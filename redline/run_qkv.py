"""Run the 32-core QKV-post artifact (kernels/glue/qkv_design.py) and check it.

  random: random q|k|v C stream + random rotary/weights vs numpy rms_rope (Q, K) and pass-through (V), all heads
  golden: block 0 step 1 (cached, M=4096): real to_q/k/v vs golden norm_q/k + RoPE, all 32 heads
Usage (IRON env active): python run_qkv.py ARTIFACT [--golden] [--iters 10]
Host helpers for the orchestrator: positions(M), qkv_tables(...), qkv32_npu(...), unpack_head(...).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.utils import DefaultNPURuntime

from npu_layouts import V9, a_unlayout, b_unlayout, c_layout_atb, c_unlayout_atb, from_bfp16, to_bfp16
from run_glue import rms_rope, rope
from run_prebuilt import load, pmode

H, D, PREFIX = 32, 128, 4128
HALF = 64 * 128 // 8   # blocks per tile
TAB = 128 * 64 + 2 * 128
C_LOG2 = np.log2(np.e) / np.sqrt(128)
from paths import ROOT
G = ROOT / "golden" / "npy"   # developer checks only

# per input 8x8 block blk = (sbr*8 + sbc)*4 + kk of a half-tile: its 8 token rows and 4 rotary pairs
_BLK = np.arange(128)
_ROWS = (_BLK // 32)[:, None] * 16 + ((_BLK % 4) // 2)[:, None] * 8 + np.arange(8)[None, :]          # (128, 8)
_PAIRS = ((_BLK // 4) % 8)[:, None] * 8 + ((_BLK % 4) % 2)[:, None] * 4 + np.arange(4)[None, :]     # (128, 4)


def positions(M, P=None):
    """Rotary row (token position) per GEMM row, -1 = pad. M=4096: cached step (target only); M=8704: step 1
    [prefix P (default 4128) | pad | target 4096 from row 4224 | pad 384]."""
    P = PREFIX if P is None else P
    if M == 4096:
        return P + np.arange(4096)
    if M in (8704, 4608):   # 4608: text-to-image step 1, [text <= 512 | target 4096 from row 512]
        row0 = 4224 if M == 8704 else 512
        pos = np.full(M, -1)
        pos[:P] = np.arange(P)
        pos[row0:row0 + 4096] = P + np.arange(4096)
        return pos
    raise ValueError(M)


def qkv_tables(cos, sin, wqc, wk):
    """cos/sin (M, 64) per GEMM row (pad rows: anything) and w_q*log2(e)/sqrt(128), w_k (128 each) -> the TABS
    buffer (M/64, 8448) bf16. The cos/sin part depends only on the row layout: build it once per layout, and per
    block only the last 256 entries change."""
    T = cos.shape[0] // 64
    c = cos.reshape(T, 64, 64)[:, _ROWS[:, :, None], _PAIRS[:, None, :]].reshape(T, 128, 32)
    s = sin.reshape(T, 64, 64)[:, _ROWS[:, :, None], _PAIRS[:, None, :]].reshape(T, 128, 32)
    tab = np.empty((T, TAB), np.float32)
    tab[:, :128 * 64] = np.concatenate([c, s], -1).reshape(T, -1)
    tab[:, 128 * 64:128 * 64 + 128] = wqc
    tab[:, 128 * 64 + 128:] = wk
    return tab.astype(bfloat16)


def qkv32_npu(art, cin_pack, tabs, M, iters=0):
    """cin_pack: the QKV GEMM's C stream (bfp16 bytes, c_layout_atb 128x128 of M x 12288). Returns (Q, KV, ms):
    Q (32, T, tile bytes) and KV (32, T, 2, tile bytes) as the attention kernel takes them, and the median NPU ms
    over `iters` extra runs (None if 0)."""
    T = M // 64
    c_t = iron.tensor(cin_pack.view(V9), dtype=V9, device="npu")
    t_t = iron.tensor(tabs.reshape(-1), dtype=bfloat16, device="npu")
    q_t = iron.tensor(np.zeros(H * T * HALF, dtype=V9), dtype=V9, device="npu")
    kv_t = iron.tensor(np.zeros(H * T * 2 * HALF, dtype=V9), dtype=V9, device="npu")
    h = load(art)
    DefaultNPURuntime.run(h, [c_t, t_t, q_t, kv_t])
    ms = None
    if iters:
        ms = float(np.median([DefaultNPURuntime.run(h, [c_t, t_t, q_t, kv_t]).npu_time for _ in range(iters)])) / 1e6
    q = np.frombuffer(q_t.numpy().tobytes(), np.uint8).reshape(H, T, HALF * 9)
    kv = np.frombuffer(kv_t.numpy().tobytes(), np.uint8).reshape(H, T, 2, HALF * 9)
    return q, kv, ms


def unpack_head(q, kv, h):
    """Q, K, V of head h, each (T*64, 128) float32."""
    T = q.shape[1]
    Q = a_unlayout(from_bfp16(q[h].ravel()), T * 64, 128, 64, 128)
    K = b_unlayout(from_bfp16(kv[h, :, 0].ravel()), 128, T * 64, 128, 64).T
    V = b_unlayout(from_bfp16(kv[h, :, 1].ravel()), T * 64, 128, 64, 128)
    return Q, K, V


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel())[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact")
    p.add_argument("--golden", action="store_true")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    art = Path(args.artifact)
    M = json.loads((art / "meta.json").read_text())["M"]
    pos = positions(M)
    valid = pos >= 0
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    if args.golden:
        assert M == 4096, "golden check is the cached step (M=4096)"
        rot = np.load(G / "s0_rotary.npy")
        cos, sin = rot[pos, :, 0], rot[pos, :, 1]
        wq, wk = np.load(G / "norm_q.npy"), np.load(G / "norm_k.npy")
        C = np.concatenate([np.load(G / f"s1_{n}.npy") for n in ("to_q", "to_k", "to_v")], 1)
        nq, nk = np.load(G / "s1_norm_q.npy", mmap_mode="r"), np.load(G / "s1_norm_k.npy", mmap_mode="r")
    else:
        ang = rng.uniform(-np.pi, np.pi, (M, 64))
        cos, sin = np.cos(ang), np.sin(ang)
        wq, wk = 1 + 0.1 * rng.standard_normal(128), 1 + 0.1 * rng.standard_normal(128)
        C = rng.standard_normal((M, 3 * 4096)).astype(np.float32)
        C[~valid] = 0
    cos = cos.astype(bfloat16).astype(np.float32)
    sin = sin.astype(bfloat16).astype(np.float32)
    wqc = (wq * C_LOG2).astype(bfloat16).astype(np.float32)
    wk = np.asarray(wk).astype(bfloat16).astype(np.float32)
    cin = to_bfp16(c_layout_atb(C.astype(np.float32), 128, 128))
    tabs = qkv_tables(cos, sin, wqc, wk)
    print(f"host prep {time.time() - t0:.1f} s", flush=True)
    q, kv, ms = qkv32_npu(art, cin, tabs, M, iters=args.iters)
    cq = c_unlayout_atb(from_bfp16(cin), M, 3 * 4096, 128, 128)  # bfp16-rounded input
    worst = {"Q": 1.0, "K": 1.0, "V": 1.0}
    for h in range(H):
        Q, K, V = unpack_head(q, kv, h)
        hs = slice(h * D, (h + 1) * D)
        if args.golden:
            refs = (rope(np.asarray(nq[:, h], np.float32), cos, sin) * C_LOG2,
                    rope(np.asarray(nk[:, h], np.float32), cos, sin), C[:, 8192 + h * D:8192 + (h + 1) * D])
        else:
            refs = (rms_rope(cq[:, hs], wqc, cos, sin), rms_rope(cq[:, 4096 + h * D:4096 + (h + 1) * D], wk, cos, sin),
                    cq[:, 8192 + h * D:8192 + (h + 1) * D])
        for name, got, ref in zip("QKV", (Q, K, V), refs):
            worst[name] = min(worst[name], corr(got[valid], ref[valid]))
        if not args.golden:
            pad_max = max(np.abs(x[~valid]).max() if (~valid).any() else 0.0 for x in (Q, K, V))
            assert pad_max == 0.0, f"head {h}: pad rows not zero ({pad_max})"
    ok = all(c > 0.999 for c in worst.values())
    gb = (cin.nbytes + q.nbytes + kv.nbytes) / 1e9
    print(f"{art.name}: M={M} {'golden' if args.golden else 'random'}  worst-head corr Q {worst['Q']:.5f} "
          f"K {worst['K']:.5f} V {worst['V']:.5f}  {'PASS' if ok else 'FAIL'}"
          + (f"  median {ms:.3f} ms ({gb / (ms / 1e3):.1f} GB/s in+out, pmode {pmode()})" if ms else ""))


if __name__ == "__main__":
    main()
