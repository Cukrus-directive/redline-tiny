"""Run a prebuilt glue artifact (kernels/glue) and check it against numpy.

swiglu: input = a gate|proj GEMM output in config3's C stream with columns interleaved per 128-col tile
        ([gate 64 | proj 64]); output = silu(gate) * proj as down-GEMM A tiles (128x64 a_layout).
Usage (IRON env active): python run_glue.py ARTIFACT_DIR [--iters 20] [--warmup 5]
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.utils import DefaultNPURuntime

from npu_layouts import V9, a_unlayout, c_layout_atb, c_unlayout_atb, from_bfp16, to_bfp16
from run_prebuilt import load, pmode


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(y, cos, sin):
    """Complex RoPE on adjacent dim pairs: y (n, 128), cos/sin (n, 64)."""
    a, b = y[:, 0::2], y[:, 1::2]
    out = np.empty_like(y)
    out[:, 0::2], out[:, 1::2] = a * cos - b * sin, a * sin + b * cos
    return out


def rms_rope(x, w, cos, sin):
    """RMSNorm over the last dim (eps 1e-6) * w, then RoPE."""
    return rope(x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w, cos, sin)


def qkvpost1_raw(art, q, k, v, cos, sin, wqc, wk):
    """Run the qkvpost1 artifact on T*64 tokens of one head. q/k/v (n, 128) float; cos/sin (n, 64); wqc = w_q *
    log2(e)/sqrt(128). Returns (out, xin): out = (T, 3, tile bytes), per 64-token tile the Q a_layout tile, K as
    b_layout(K^T) and the V b_layout tile, i.e. the attention kernel's input format; xin = the packed bfp16 input."""
    from ml_dtypes import bfloat16
    n = q.shape[0]
    T = n // 64

    def half_c(x):  # 64x128 -> the C-layout half-tile bfp16 stream
        return to_bfp16(c_layout_atb(x, 64, 128, rows=1, cols=1))

    def table(t0):  # per input block: 32 cos (lane r*4+p) then 32 sin, then w_q*c, w_k
        tab = np.empty((128, 64), np.float32)
        for blk in range(128):
            sbr, sbc, kk = blk // 32, (blk // 4) % 8, blk % 4
            rows = t0 + sbr * 16 + (kk // 2) * 8 + np.arange(8)
            pairs = sbc * 8 + (kk % 2) * 4 + np.arange(4)
            tab[blk, :32] = cos[np.ix_(rows, pairs)].reshape(-1)
            tab[blk, 32:] = sin[np.ix_(rows, pairs)].reshape(-1)
        return np.concatenate([tab.reshape(-1), wqc, wk])

    xin = np.concatenate([half_c(m[h * 64:(h + 1) * 64]) for h in range(T) for m in (q, k, v)])
    tabs = np.concatenate([table(h * 64) for h in range(T)]).astype(bfloat16)
    x_t = iron.tensor(xin.view(V9), dtype=V9, device="npu")
    t_t = iron.tensor(tabs, dtype=bfloat16, device="npu")
    o_t = iron.tensor(np.zeros(3 * T * HALF_BLOCKS, dtype=V9), dtype=V9, device="npu")
    DefaultNPURuntime.run(load(art), [x_t, t_t, o_t])
    return np.frombuffer(o_t.numpy().tobytes(), dtype=np.uint8).reshape(T, 3, HALF_BLOCKS * 9), xin


def qkvpost1_unpack(out):
    """qkvpost1_raw output -> Q, K, V, each (T*64, 128) float32."""
    from npu_layouts import b_unlayout
    T = out.shape[0]
    Q = np.concatenate([a_unlayout(from_bfp16(out[h, 0]), 64, 128, 64, 128) for h in range(T)])
    K = np.concatenate([b_unlayout(from_bfp16(out[h, 1]), 128, 64, 128, 64).T for h in range(T)])
    V = np.concatenate([b_unlayout(from_bfp16(out[h, 2]), 64, 128, 64, 128) for h in range(T)])
    return Q, K, V


def check_qkvpost1(art, meta, args):
    from ml_dtypes import bfloat16
    T = meta["T"]
    rng = np.random.default_rng(args.seed)
    n = T * 64
    q, k, v = (rng.standard_normal((n, 128)).astype(np.float32) for _ in range(3))
    ang = rng.uniform(-np.pi, np.pi, (n, 64))
    cos, sin = np.cos(ang).astype(bfloat16).astype(np.float32), np.sin(ang).astype(bfloat16).astype(np.float32)
    c = np.log2(np.e) / np.sqrt(128)
    wq = (1 + 0.1 * rng.standard_normal(128)).astype(bfloat16).astype(np.float32)
    wk = (1 + 0.1 * rng.standard_normal(128)).astype(bfloat16).astype(np.float32)
    wqc = (wq * c).astype(bfloat16).astype(np.float32)
    out, xin = qkvpost1_raw(art, q, k, v, cos, sin, wqc, wk)
    Q, K, V = qkvpost1_unpack(out)
    halves = from_bfp16(xin).reshape(T, 3, 64 * 128)
    un = lambda x: c_unlayout_atb(x, 64, 128, 64, 128, rows=1, cols=1)
    qq, kq, vq = (np.concatenate([un(halves[h, j]) for h in range(T)]) for j in range(3))
    worst = {}
    for name, got, ref in (("Q", Q, rms_rope(qq, wqc, cos, sin)), ("K", K, rms_rope(kq, wk, cos, sin)), ("V", V, vq)):
        e = float(np.abs(got - ref).max() / np.abs(ref).max())
        worst[name] = (e, float(np.corrcoef(got.ravel(), ref.ravel())[0, 1]))
    ok = all(cc > 0.999 for _, cc in worst.values())
    print(f"{art.name}: T={T}  " + "  ".join(f"{k}: err {e:.4f} corr {cc:.5f}" for k, (e, cc) in worst.items())
          + f"  {'PASS' if ok else 'FAIL'}")


HALF_BLOCKS = 64 * 128 // 8


def swiglu_pack(gate, proj):
    """gate/proj (M, F) -> the config3 C stream with [gate 64 | proj 64] per 128-col tile (bfp16), plus the
    bfp16-rounded gate and proj."""
    M, F = gate.shape
    N = 2 * F
    c = np.empty((M, N), np.float32)
    c.reshape(M, N // 128, 2, 64)[:, :, 0] = gate.reshape(M, F // 64, 64)
    c.reshape(M, N // 128, 2, 64)[:, :, 1] = proj.reshape(M, F // 64, 64)
    c_pack = to_bfp16(c_layout_atb(c, 128, 128))
    cq = c_unlayout_atb(from_bfp16(c_pack), M, N, 128, 128).reshape(M, N // 128, 2, 64)
    return c_pack, cq[:, :, 0].reshape(M, F), cq[:, :, 1].reshape(M, F)


def swiglu_unpack(a_t, M, F):
    """The swiglu output tensor (down-GEMM A tiles, 128x64 a_layout) -> (M, F) float32."""
    return a_unlayout(from_bfp16(np.frombuffer(a_t.numpy().tobytes(), dtype=np.uint8)), M, F, 128, 64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    art = Path(args.artifact)
    meta = json.loads((art / "meta.json").read_text())
    if meta["family"] == "qkvpost1":
        return check_qkvpost1(art, meta, args)
    M, N = meta["M"], meta["N"]
    F = N // 2

    rng = np.random.default_rng(args.seed)
    gate = (rng.standard_normal((M, F)) * 2).astype(np.float32)
    proj = rng.standard_normal((M, F)).astype(np.float32)
    c_pack, gq, pq = swiglu_pack(gate, proj)
    ref = silu(gq.astype(np.float64)) * pq

    c_t = iron.tensor(c_pack.view(V9), dtype=V9, device="npu")
    a_t = iron.tensor(np.zeros(M * F // 8, dtype=V9), dtype=V9, device="npu")
    rt = DefaultNPURuntime
    handle = load(art)
    us = []
    for i in range(args.warmup + args.iters):
        r = rt.run(handle, [c_t, a_t])
        if i >= args.warmup:
            us.append(r.npu_time / 1e3)

    out = swiglu_unpack(a_t, M, F)
    err = np.abs(out - ref)
    rel_max = float(err.max() / np.abs(ref).max())
    corr = float(np.corrcoef(out.reshape(-1), ref.reshape(-1))[0, 1])
    med = float(np.median(us))
    gbps = (c_pack.nbytes + M * F * 9 / 8) / (med * 1e-6) / 1e9
    res = {"artifact": art.name, "family": meta["family"], "M": M, "N": N, "pmode": pmode(),
           "npu_median_ms": med / 1e3, "GBps": gbps, "rel_err_max": rel_max, "corr": corr,
           "pass": corr > 0.999 and rel_max < 0.05, "host": platform.node(),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "meta": meta}
    print(f"{art.name}: {M}x{N} -> {M}x{F}  median {med / 1e3:.3f} ms ({gbps:.1f} GB/s in+out)  "
          f"err {rel_max:.4f} corr {corr:.6f}  {'PASS' if res['pass'] else 'FAIL'}")
    with open(Path(__file__).with_name("results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
