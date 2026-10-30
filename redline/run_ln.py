"""Run the fused residual + LayerNorm + modulate artifact (kernels/glue/ln_design.py) and check it.

  random : x, branch, gate, scale random (two parameter rows, split at S) vs numpy.
  golden : block 0 of the golden step 1. (1) x = block0.in, branch 0 -> vs img_norm1.out * (1 + scale1);
           (2) x = block0.in, branch = to_out.out, g = tanh(gate1) -> vs img_norm2.out * (1 + scale2).
Usage (IRON env active): python run_ln.py ARTIFACT_DIR [--check random|golden] [--iters 20]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.utils import DefaultNPURuntime

from npu_layouts import V9, a_unlayout, c_layout_atb, c_unlayout_atb, from_bfp16, to_bfp16
from run_prebuilt import load, pmode

D = 4096
from paths import ROOT
G = ROOT / "golden" / "npy"   # developer checks only
bf = lambda a: np.asarray(a, np.float32).astype(bfloat16).astype(np.float32)


def ln_pack_x(x):
    """Residual stream (M, 4096) -> bf16 in config3's C-stream order."""
    return c_layout_atb(np.asarray(x, np.float32), 128, 128).astype(bfloat16)


def ln_pack_br(br):
    """Branch (M, 4096) -> bfp16 in config3's C-stream order (what the out-proj / down GEMM emits)."""
    return to_bfp16(c_layout_atb(np.asarray(br, np.float32), 128, 128))


def ln_params(g, sc):
    """g, sc: (2, 4096) rows [real t, t=0] -> the 2 x 8192 bf16 parameter buffer."""
    return np.concatenate([np.asarray(g, np.float32), np.asarray(sc, np.float32)], 1).astype(bfloat16).reshape(-1)


def ln_run(art, xc, brc, p, iters=0, warmup=0):
    """Run once (or warmup + iters for timing). Returns (x' bf16 C-stream, A bfp16 bytes, list of NPU us)."""
    M = xc.size // D
    x_t = iron.tensor(xc.reshape(-1), dtype=bfloat16, device="npu")
    b_t = iron.tensor(brc.view(V9), dtype=V9, device="npu")
    p_t = iron.tensor(p, dtype=bfloat16, device="npu")
    xo_t = iron.zeros((M * D,), dtype=bfloat16, device="npu")
    ao_t = iron.tensor(np.zeros(M * D // 8, dtype=V9), dtype=V9, device="npu")
    rt, h, us = DefaultNPURuntime, load(art), []
    for i in range(max(1, warmup + iters)):
        r = rt.run(h, [x_t, b_t, p_t, xo_t, ao_t])
        if i >= warmup:
            us.append(r.npu_time / 1e3)
    return xo_t.numpy(), np.frombuffer(ao_t.numpy().tobytes(), dtype=np.uint8), us


def ln_unpack(xo, ao, M):
    """-> x' (M, 4096) and y (M, 4096) float32."""
    return (c_unlayout_atb(np.asarray(xo, np.float32), M, D, 128, 128),
            a_unlayout(from_bfp16(ao), M, D, 128, 64))


def ln_ref(xq, brq, g, sc):
    """numpy model of the kernel on the exact inputs it sees: x' rounded to bf16, LayerNorm in float64."""
    xp = bf(xq + bf(g) * brq)
    x64 = xp.astype(np.float64)
    mu = x64.mean(1, keepdims=True)
    var = (x64 * x64).mean(1, keepdims=True) - mu * mu
    return xp, (x64 - mu) / np.sqrt(var + 1e-6) * bf(sc)


def stats(name, got, ref):
    got, ref = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    e = float(np.abs(got - ref).max() / np.abs(ref).max())
    cc = float(np.corrcoef(got, ref)[0, 1])
    print(f"  {name:44s} err {e:.4f}  corr {cc:.6f}")
    return e, cc


def rows_param(v, M, S):
    """Per-row parameter (M, 4096) from 2 rows: M-tiles < S use row 1."""
    out = np.repeat(v[0:1], M, 0)
    out[: S * 128] = v[1]
    return out


def check_random(art, meta, args):
    M, S = meta["M"], meta.get("S", 0)
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((M, D)).astype(np.float32) * 3
    x[:, rng.integers(0, D, 8)] *= 40  # a few outlier channels, like the residual stream
    br = rng.standard_normal((M, D)).astype(np.float32)
    g = np.tanh(rng.standard_normal((2, D))).astype(np.float32)
    sc = (1 + 0.3 * rng.standard_normal((2, D))).astype(np.float32)
    xc, brc, p = ln_pack_x(x), ln_pack_br(br), ln_params(g, sc)
    xo, ao, us = ln_run(art, xc, brc, p, args.iters, args.warmup)
    xp, y = ln_unpack(xo, ao, M)
    xq = c_unlayout_atb(xc.astype(np.float32), M, D, 128, 128)
    brq = c_unlayout_atb(from_bfp16(brc), M, D, 128, 128)
    xr, yr = ln_ref(xq, brq, rows_param(g, M, S), rows_param(sc, M, S))
    print(f"{art.name}: M={M} S={S} random")
    e1, c1 = stats("x' vs numpy", xp, xr)
    e2, c2 = stats("y (A layout) vs numpy", y, yr)
    exact = float(np.mean(xp == xr))
    print(f"  x' bit-exact with numpy: {exact:.4f}")
    if us and args.iters:
        med = float(np.median(us))
        gb = (2 * (M * D * 2 + M * D * 9 / 8) + M * D * 2 + M * D * 9 / 8) / (med * 1e-6) / 1e9
        print(f"  NPU median {med / 1e3:.3f} ms over {len(us)} runs ({gb:.1f} GB/s of DDR traffic), pmode {pmode()}")
    ok = c1 > 0.99999 and c2 > 0.9999
    print(f"  {'PASS' if ok else 'FAIL'}")


def check_golden(art, meta, args):
    M = meta["M"]
    assert M == 4096 and meta.get("S", 0) == 0, "golden check is the cached step: M=4096, S=0"
    x = np.load(G / "s1_block0_in.npy")
    mod = np.load(G / "s1_modulation.npy")  # (2, 16384): [scale1 | gate1 | scale2 | gate2], row 0 = real t
    s1, g1, s2 = mod[:, :D], mod[:, D:2 * D], mod[:, 2 * D:3 * D]
    print(f"{art.name}: golden block 0, step 1")
    # (1) ln_mod only: branch 0
    xo, ao, _ = ln_run(art, ln_pack_x(x), ln_pack_br(np.zeros((M, D), np.float32)), ln_params(np.zeros((2, D)), 1 + s1))
    xp, y = ln_unpack(xo, ao, M)
    gold1 = np.load(G / "s1_img_norm1.npy") * (1 + s1[0])
    stats("(1) x' == block0.in", xp, x)
    stats("(1) LN(x)*(1+s1) vs golden img_norm1*(1+s1)", y, gold1)
    # (2) residual + ln_mod: branch = to_out.out (as bfp16, like the out-proj GEMM output), g = tanh(gate1)
    br = np.load(G / "s1_to_out_out.npy")
    xo, ao, _ = ln_run(art, ln_pack_x(x), ln_pack_br(br), ln_params(np.tanh(g1), 1 + s2))
    xp, y = ln_unpack(xo, ao, M)
    xg = bf(x + bf(bf(np.tanh(g1[0])) * bf(br)))  # golden's bf16 residual add
    gold2 = np.load(G / "s1_img_norm2.npy") * (1 + s2[0])
    stats("(2) x' vs block0.in + tanh(g1)*to_out (bf16)", xp, xg)
    stats("(2) LN(x')*(1+s2) vs golden img_norm2*(1+s2)", y, gold2)
    brq = c_unlayout_atb(from_bfp16(ln_pack_br(br)), M, D, 128, 128)
    _, yr = ln_ref(x, brq, rows_param(np.tanh(g1), M, 0), rows_param(1 + s2, M, 0))
    stats("(2) NPU vs numpy model, same bfp16 branch", y, yr)
    stats("(2) numpy model (bfp16 branch) vs golden", yr, gold2)
    _, yx = ln_ref(x, br, rows_param(np.tanh(g1), M, 0), rows_param(1 + s2, M, 0))
    stats("(2) numpy model, unrounded branch, vs golden", yx, gold2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact")
    p.add_argument("--check", default="random", choices=["random", "golden"])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    art = Path(args.artifact)
    meta = json.loads((art / "meta.json").read_text())
    (check_random if args.check == "random" else check_golden)(art, meta, args)


if __name__ == "__main__":
    main()
