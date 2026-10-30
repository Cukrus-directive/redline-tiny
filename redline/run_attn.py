"""Run a prebuilt attention artifact (kernels/attn) and check it against float64 numpy attention.

meta.json (from npu_build.py --meta): family "attn1" (one core, one Q tile), "attn32", or "attn32o" (attn32 with
OA = O as the out-projection's bfp16 A operand, OHEADS/H0 = its width/head offset, KVSPLIT = target tiles per
head in a first KV buffer, prefix cache in a second); MQ, NK, NT, NKV, H, W.
Usage (IRON env active): python run_attn.py ARTIFACT_DIR [--iters 20] [--warmup 5]
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.utils import DefaultNPURuntime
from aie.utils.npukernel import NPUKernel

from npu_layouts import V9, a_layout, a_unlayout, b_layout, c_unlayout, from_bfp16, to_bfp16
from run_prebuilt import load, pmode

D = 128
C_LOG2 = np.log2(np.e) / np.sqrt(D)  # the kernel works in log2 units; the host folds this into Q


def pack_kv(k, v, nt):
    """K0 V0 K1 V1 ...: K tile = b_layout(K_t^T, D, nt), V tile = b_layout(V_t, nt, D), each bfp16."""
    out = []
    for i in range(0, k.shape[0], nt):
        out.append(to_bfp16(b_layout(k[i:i + nt].T, D, nt)))
        out.append(to_bfp16(b_layout(v[i:i + nt], nt, D)))
    return np.concatenate(out)


def unpack_o(raw, hs, NQ, MQ):
    """The O tensor (bf16; per head, consecutive MQ x D C-layout tiles) -> heads hs as (len(hs), NQ, D) float32."""
    raw = raw.astype(np.float32).reshape(-1, NQ // MQ, MQ * D)
    return np.stack([np.concatenate([c_unlayout(raw[h, i], MQ, D, MQ, D) for i in range(NQ // MQ)]) for h in hs])


def unpack_oa(raw, hs, NQ, meta):
    """OA artifacts: O is the out-projection's bfp16 A operand, a_layout(O_all, 128, 64) with O_all (NQ, OHEADS*D)
    and head h in columns (H0 + h)*D. -> heads hs as (len(hs), NQ, D) float32."""
    oh, h0 = meta.get("OHEADS", 32), meta.get("H0", 0)
    o = a_unlayout(from_bfp16(np.frombuffer(raw.tobytes(), dtype=np.uint8)), NQ, oh * D, 128, 64)
    return np.stack([o[:, (h0 + h) * D:(h0 + h + 1) * D] for h in hs])


def split_kv(kv_pack, H, NK, NT, kvsplit):
    """A single [head][K0 V0 K1 V1 ...] pack -> the two buffers of a KVSPLIT artifact: per head the first kvsplit
    tiles (target), then the rest (prefix cache)."""
    per = kv_pack.reshape(H, NK // NT, -1)
    return [np.ascontiguousarray(per[:, :kvsplit]).reshape(-1), np.ascontiguousarray(per[:, kvsplit:]).reshape(-1)]


def o_tensor(meta, H, NQ):
    """The output tensor an attention artifact writes: bf16 C tiles, or (OA) the bfp16 A operand."""
    if meta.get("OA"):
        return iron.tensor(np.zeros(NQ * meta.get("OHEADS", 32) * D // 8, dtype=V9), dtype=V9, device="npu")
    return iron.zeros((H * NQ * D,), dtype=bfloat16, device="npu")


def attn_npu(art, q_pack, kv_pack, H, NQ, MQ):
    """Run an attention artifact once on packed bfp16 bytes: q_pack = per head, MQ-query a_layout tiles of Q *
    log2(e)/sqrt(D) in order; kv_pack = per head K0 V0 K1 V1 ... (pack_kv format), or for a KVSPLIT artifact a
    list [target pack, prefix pack], each per head. Returns O (H, NQ, D) float32."""
    meta = json.loads((Path(art) / "meta.json").read_text())
    kvs = kv_pack if isinstance(kv_pack, (list, tuple)) else [kv_pack]
    if meta.get("KVSPLIT") and len(kvs) == 1:  # one [target | prefix] pack per head: split it the artifact's way
        kvs = split_kv(kvs[0], H, meta["NK"], meta["NT"], meta["KVSPLIT"])
    ins = [iron.tensor(x.view(V9), dtype=V9, device="npu") for x in [q_pack, *kvs]]
    o_t = o_tensor(meta, H, NQ)
    DefaultNPURuntime.run(load(art), [*ins, o_t])
    if meta.get("OA"):
        return unpack_oa(o_t.numpy(), range(H), NQ, meta)
    return unpack_o(o_t.numpy(), range(H), NQ, MQ)


def pack_q_small(q):
    """attn32q Q buffer: per head one MQ-query a_layout tile of Q * log2(e)/sqrt(D) (q: (32, MQ, D))."""
    return np.concatenate([to_bfp16(a_layout(q[h] * C_LOG2, q.shape[1], D)) for h in range(q.shape[0])])


def pack_lim(lim, MQ, NT):
    """attn32q LIM buffer: one tile-sized block (MQ*NT/8 bfp16 blocks) with lim (MQ int32, last visible key per row)
    at its start."""
    buf = np.zeros(MQ * NT // 8 * 9, np.uint8)
    buf[:4 * MQ] = np.asarray(lim, np.int32).view(np.uint8)
    return buf


def check_small(art, meta, args):
    """attn32q: 32 heads x one MQ-query tile, per-row key limits, K/V at tiles P0.. of a [head][PT][K|V] buffer."""
    MQ, NK, NT = meta["MQ"], meta["NK"], meta["NT"]
    H, nkt = 32, NK // NT
    PT, P0 = meta.get("PT") or nkt, meta.get("P0", 0)
    rng = np.random.default_rng(args.seed)
    q = rng.standard_normal((H, MQ, D)).astype(bfloat16).astype(np.float32)
    k = rng.standard_normal((H, NK, D)).astype(bfloat16).astype(np.float32)
    v = rng.standard_normal((H, NK, D)).astype(bfloat16).astype(np.float32)
    lim = rng.integers(0, NK - 33, MQ)
    lim[:8] = np.arange(8)                                   # causal text rows near the start
    lim[8:40] = NK - 33 - np.arange(32)[::-1]                # rows near the end of the prefix
    kv = rng.integers(0, 256, (H, PT, 2 * NT * D // 8 * 9), dtype=np.uint8)
    kv[:, P0:P0 + nkt] = np.concatenate([pack_kv(k[h], v[h], NT) for h in range(H)]).view(np.uint8).reshape(H, nkt, -1)
    ins = [iron.tensor(x.view(V9), dtype=V9, device="npu") for x in
           (pack_q_small(q), pack_lim(lim, MQ, NT), kv.reshape(-1))]
    o_t = iron.tensor(np.zeros(128 * meta.get("OHEADS", 32) * D // 8, dtype=V9), dtype=V9, device="npu")
    rt, h_ = DefaultNPURuntime, load(art)
    us = []
    for i in range(args.warmup + args.iters):
        r = rt.run(h_, [*ins, o_t])
        if i >= args.warmup:
            us.append(r.npu_time / 1e3)
    o = a_unlayout(from_bfp16(np.frombuffer(o_t.numpy().tobytes(), np.uint8)), 128, meta.get("OHEADS", 32) * D,
                   128, 64)[:MQ]
    h0 = meta.get("H0", 0)
    worst = 1.0
    for h in range(H):
        s = q[h].astype(np.float64) @ k[h].T.astype(np.float64) / np.sqrt(D)
        s = np.where(np.arange(NK)[None, :] <= lim[:, None], s, -np.inf)
        w = np.exp(s - s.max(1, keepdims=True))
        ref = (w / w.sum(1, keepdims=True)) @ v[h].astype(np.float64)
        got = o[:, (h0 + h) * D:(h0 + h + 1) * D]
        worst = min(worst, float(np.corrcoef(got.ravel(), ref.ravel())[0, 1]))
    print(f"{art.name}: 32 heads x {MQ} queries over <= {NK} keys (per-row limits)  median {np.median(us) / 1e3:.3f} ms  "
          f"worst-head corr {worst:.5f}  {'PASS' if worst > 0.995 else 'FAIL'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diag", action="store_true", help="print per-band / per-block correlation")
    args = p.parse_args()
    art = Path(args.artifact)
    meta = json.loads((art / "meta.json").read_text())
    if meta["family"] == "attn32q":
        return check_small(art, meta, args)
    MQ, NK, NT = meta["MQ"], meta["NK"], meta["NT"]
    H = meta.get("H", 1)
    NQ = meta.get("W", 1) * 32 * MQ if meta["family"] in ("attn32", "attn32o") else MQ  # queries per head

    rng = np.random.default_rng(args.seed)
    q = rng.standard_normal((H, NQ, D)).astype(bfloat16).astype(np.float32)
    k = rng.standard_normal((H, NK, D)).astype(bfloat16).astype(np.float32)
    v = rng.standard_normal((H, NK, D)).astype(bfloat16).astype(np.float32)
    NKV = meta.get("NKV") or NK  # valid keys; the rest is zero padding the kernel must mask
    k[:, NKV:] = 0
    v[:, NKV:] = 0

    # Q/O: per head, consecutive MQ-query tiles ([head][wave][core] == tile order); KV: per head K0 V0 K1 V1 ...
    q_pack = np.concatenate([to_bfp16(a_layout(q[h, i:i + MQ] * C_LOG2, MQ, D))
                             for h in range(H) for i in range(0, NQ, MQ)])
    kv_pack = np.concatenate([pack_kv(k[h], v[h], NT) for h in range(H)])
    kvs = split_kv(kv_pack, H, NK, NT, meta["KVSPLIT"]) if meta.get("KVSPLIT") else [kv_pack]
    if any(meta.get(x) for x in ("QT", "Q0", "KT", "K0", "PT", "P0")):
        # tile-range artifacts: embed the compact packs at their tile offsets in larger buffers full of garbage
        def embed(pack, per_head, total, first, unit):
            full = rng.integers(0, 256, (H, total, unit), dtype=np.uint8)
            full[:, first:first + per_head] = pack.view(np.uint8).reshape(H, per_head, unit)
            return full.reshape(-1)
        qtile = MQ * D // 8 * 9
        q_pack = embed(q_pack, NQ // MQ, meta.get("QT") or NQ // MQ, meta.get("Q0", 0), qtile)
        kvtile = 2 * NT * D // 8 * 9
        n0 = meta.get("KVSPLIT") or NK // NT
        kvs[0] = embed(kvs[0], n0, meta.get("KT") or n0, meta.get("K0", 0), kvtile)
        if meta.get("KVSPLIT"):
            n1 = NK // NT - n0
            kvs[1] = embed(kvs[1], n1, meta.get("PT") or n1, meta.get("P0", 0), kvtile)
    q_t = iron.tensor(q_pack.view(V9), dtype=V9, device="npu")
    kv_ts = [iron.tensor(x.view(V9), dtype=V9, device="npu") for x in kvs]  # KVSPLIT: [target, prefix cache]
    o_t = o_tensor(meta, H, NQ)

    rt = DefaultNPURuntime
    us = []
    if meta.get("TRACE"):  # trace build: one traced run (it needs the extra trace buffer), cycles per kernel call
        from aie.utils.trace import TraceConfig
        from aie.utils.trace.utils import get_cycles_summary
        tc = TraceConfig(trace_size=meta["TRACE"], trace_file=str(art / "trace.txt"))
        _, r = rt.load_and_run(NPUKernel(art / "final.xclbin", art / "insts.bin", trace_config=tc), [q_t, *kv_ts, o_t])
        us.append(r.npu_time / 1e3)
        for path in tc.trace_to_json(str(art / "final.prj" / "input_with_addresses.mlir"),
                                     output_name=str(art / "trace.json")):
            for entry in get_cycles_summary(path):
                calls = entry[1:]
                if calls:
                    qk, pv = calls[0::2], calls[1::2]
                    print(f"  trace {entry[0]}: {len(calls)} calls; attn_qk median {np.median(qk):.0f} cyc, "
                          f"attn_pv median {np.median(pv):.0f} cyc (first/min/max qk {qk[0]}/{min(qk)}/{max(qk)})")
    else:
        handle = load(art)
        for i in range(args.warmup + args.iters):
            r = rt.run(handle, [q_t, *kv_ts, o_t])
            if i >= args.warmup:
                us.append(r.npu_time / 1e3)

    hs = list(range(H)) if H <= 2 else [0, H - 1]  # a float64 reference of every head would not fit in RAM
    o = unpack_oa(o_t.numpy(), hs, NQ, meta) if meta.get("OA") else unpack_o(o_t.numpy(), hs, NQ, MQ)
    s = np.einsum("hqd,hkd->hqk", q[hs].astype(np.float64), k[hs][:, :NKV].astype(np.float64)) / np.sqrt(D)
    w = np.exp(s - s.max(2, keepdims=True))
    ref = np.einsum("hqk,hkd->hqd", w / w.sum(2, keepdims=True), v[hs][:, :NKV].astype(np.float64))
    err = np.abs(o - ref)
    rel_max = float(err.max() / np.abs(ref).max())
    corr = float(np.corrcoef(o.reshape(-1), ref.reshape(-1))[0, 1])
    o, ref = o[0], ref[0]  # diagnostics below look at head 0

    us = np.asarray(us)
    flops = 4.0 * H * NQ * NK * D
    res = {"artifact": art.name, "family": meta["family"], "MQ": MQ, "NK": NK, "NT": NT, "H": H, "NQ": NQ,
           "pmode": pmode(), "npu_median_ms": float(np.median(us)) / 1e3,
           "tflops_median": flops / (np.median(us) * 1e-6) / 1e12,
           "rel_err_max": rel_max, "corr": corr, "pass": corr > 0.995,
           "host": platform.node(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "meta": meta}
    print(f"{art.name}: H={H} NQ={NQ} NK={NK} MQ={MQ} NT={NT} median {res['npu_median_ms']:.3f} ms "
          f"({res['tflops_median']:.2f} TFLOPS)  err {rel_max:.4f} corr {corr:.5f}  "
          f"{'PASS' if res['pass'] else 'FAIL'}")
    print("  o[0,:6]  ", np.round(o[0, :6], 4), "\n  ref[0,:6]", np.round(ref[0, :6], 4))
    if args.diag:
        band = [np.corrcoef(o[i:i + 16].ravel(), ref[i:i + 16].ravel())[0, 1] for i in range(0, MQ, 16)]
        dblk = [np.corrcoef(o[:, j:j + 16].ravel(), ref[:, j:j + 16].ravel())[0, 1] for j in range(0, D, 16)]
        rows = [np.corrcoef(o[i], ref[i])[0, 1] for i in range(MQ)]
        print("  corr per 16-query band:", np.round(band, 4))
        print("  corr per 16-dim block :", np.round(dblk, 4))
        print("  rows with corr < 0.99 :", [i for i, c in enumerate(rows) if c < 0.99])
    with open(Path(__file__).with_name("results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
