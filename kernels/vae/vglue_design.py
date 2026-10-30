"""VAE decoder glue designs (IRON, npu2) on octet-planar bf16 tensors [H+2][C/8][W+2][8].
8 cores, one per column, each with its own shim channels; column c takes the row block c*H/8 .. (c+1)*H/8 - 1
  vnorm(X, G, Y): channel RMS norm (+ SiLU), G = gamma * sqrt(C) (C bf16, broadcast once)
  vadd(A, B, Y) : flat y = a + b over L values (buffers rounded up to a multiple of 8 * N; borders stay 0)
  vup(X, Y)     : nearest 2x upsample with channel select: MODE 0 plain, 1 channel 2c+1, 2 channel 2c+i (i = row parity)
GEN = 1: shape-generic cores, so one xclbin serves every H (and every L for vadd) and each shape is only an
insts.bin: vadd / vup / vdown cores process one element / input segment per worker iteration (no trip count), vnorm
reads its segment count from a runtime parameter behind a barrier (gamma arrives once per call).
Compile-only: python vglue_design.py -d npu2 --xclbin-path X --insts-path Y --op vnorm --h 64 --w 64 --c 1152
"""

import argparse
import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker,
                       WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

_DIR = Path(__file__).resolve().parent
_INC = Path(os.environ.get("MLIR_AIE_SRC", "")) / "aie_kernels"
_SRC = str(_DIR / "vglue.cc")
N_COLS = 8
BF = np.dtype[bfloat16]


def _flags(**d):
    return [f"-I{_INC}"] + [f"-DVG_{k}={v}" for k, v in d.items()]


def seg_p(planes):
    """Pixels per row segment: a whole segment ([planes][P][8], <= 18 KB) sits in L1, double-buffered. A power of two,
    so segments tile every width (64 .. 1024)."""
    p = 1152 // planes
    return 1 << (p.bit_length() - 1)


def chunk_kp(out_planes, p, per_plane_px):
    """Output planes per chunk: the largest divisor of out_planes with a chunk <= 6 KB."""
    cap = max(1, 6144 // (per_plane_px * p * 16))
    return next(k for k in range(min(cap, out_planes), 0, -1) if out_planes % k == 0)


def add_len(h, w, c, n=4096):
    """Flat length of an OPB buffer rounded up to a multiple of 8 * n (vadd's split)."""
    L = (h + 2) * c * (w + 2)
    return -(-L // (N_COLS * n)) * N_COLS * n


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def vnorm(X: In, G: In, Y: Out, *, H: CompileTime[int] = 64, W: CompileTime[int] = 64, C: CompileTime[int] = 1152,
          SILU: CompileTime[int] = 1, GEN: CompileTime[int] = 0):
    O = C // 8
    P = seg_p(O)
    KP = chunk_kp(O, P, 1)
    NE, NK, NX = O // KP, H // N_COLS, W // P
    assert W % P == 0 and H % N_COLS == 0, (W, P, H)
    s_ty, e_ty, g_ty = np.ndarray[(O * P * 8,), BF], np.ndarray[(KP * P * 8,), BF], np.ndarray[(C,), BF]
    fl = _flags(P=P, O=O, KP=KP, SILU=SILU)
    stats = ExternalFunction("vnorm_stats", object_file_name=f"vnorm_p{P}_o{O}_kp{KP}_s{SILU}.o", source_file=_SRC,
                             arg_types=[s_ty], compile_flags=fl)
    gam = stats.object_file.bind("vnorm_gamma", [g_ty])
    app = stats.object_file.bind("vnorm_apply", [s_ty, e_ty])
    f_x = [ObjectFifo(s_ty, name=f"x_{c}", depth=2) for c in range(N_COLS)]
    f_y = [ObjectFifo(e_ty, name=f"y_{c}", depth=2) for c in range(N_COLS)]
    f_g = ObjectFifo(g_ty, name="g", depth=1)

    rtps = [Buffer(np.ndarray[(2,), np.dtype[np.int32]], name=f"rtp_{c}", use_write_rtp=True)
            for c in range(N_COLS)] if GEN else []
    bars = [WorkerRuntimeBarrier() for _ in rtps]

    def core(x, g, y, stats, gam, app, *rt):
        n_seg = NX * NK
        if rt:
            rt[1].wait_for_value(1)
            n_seg = rt[0][0]
            rt[1].release_with_value(1)
        eg = g.acquire(1)
        gam(eg)
        g.release(1)
        for _ in range_(n_seg):
            ex = x.acquire(1)
            stats(ex)
            for _ in range_(NE):
                ey = y.acquire(1)
                app(ex, ey)
                y.release(1)
            x.release(1)

    workers = [Worker(core, [f_x[c].cons(), f_g.cons(), f_y[c].prod(), stats, gam, app]
                      + ([rtps[c], bars[c]] if GEN else []), stack_size=0x400,
                      data_size=P * 8 * 4 + P * 4 + C * 2 + 256) for c in range(N_COLS)]
    PL = (W + 2) * 8
    RS = O * PL
    L = (H + 2) * RS

    def sequence(Xd, Gd, Yd, x_prods, g_prod, y_conses):
        for b, bar in zip(rtps, bars):
            b[0] = NX * NK
            bar.set(1)
        for xs in range(NX):
            tg = TaskGroup()
            if xs == 0:
                g_prod.fill(Gd, group=tg, wait=False)
            for c in range(N_COLS):
                tap = TensorAccessPattern((L,), (c * NK + 1) * RS + (xs * P + 1) * 8, [1, NK, O, P * 8],
                                          [0, RS, PL, 1])
                x_prods[c].fill(Xd, tap=tap, group=tg, wait=False)
                y_conses[c].drain(Yd, tap=tap, group=tg, wait=True)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(L,), BF], np.ndarray[(C,), BF], np.ndarray[(L,), BF],
                            [f.prod() for f in f_x], f_g.prod(), [f.cons() for f in f_y]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def vadd(A: In, B: In, Y: Out, *, L: CompileTime[int] = 8 * 4096, N: CompileTime[int] = 4096,
         GEN: CompileTime[int] = 0):
    assert L % (N_COLS * N) == 0
    per = L // N_COLS
    ne = per // N
    e_ty = np.ndarray[(N,), BF]
    add = ExternalFunction("vadd", object_file_name=f"vadd_n{N}.o", source_file=_SRC, arg_types=[e_ty, e_ty, e_ty],
                           compile_flags=_flags(N=N))
    f_a = [ObjectFifo(e_ty, name=f"a_{c}", depth=2) for c in range(N_COLS)]
    f_b = [ObjectFifo(e_ty, name=f"b_{c}", depth=2) for c in range(N_COLS)]
    f_y = [ObjectFifo(e_ty, name=f"y_{c}", depth=2) for c in range(N_COLS)]

    def core(a, b, y, add):
        for _ in range_(1 if GEN else ne):   # GEN: one element per worker iteration, no length in the core
            ea, eb, ey = a.acquire(1), b.acquire(1), y.acquire(1)
            add(ea, eb, ey)
            a.release(1)
            b.release(1)
            y.release(1)

    workers = [Worker(core, [f_a[c].cons(), f_b[c].cons(), f_y[c].prod(), add], stack_size=0x400)
               for c in range(N_COLS)]
    # per core, BDs of up to 256 elements of N each (every stride stays small)
    chunks = [(s, min(256, ne - s)) for s in range(0, ne, 256)]

    def tap(c, s, n):
        return TensorAccessPattern((L,), c * per + s * N, [1, 1, n, N], [0, 0, N, 1])

    def sequence(Ad, Bd, Yd, a_prods, b_prods, y_conses):
        for s, n in chunks:
            tg = TaskGroup()
            for c in range(N_COLS):
                a_prods[c].fill(Ad, tap=tap(c, s, n), group=tg, wait=False)
                b_prods[c].fill(Bd, tap=tap(c, s, n), group=tg, wait=False)
                y_conses[c].drain(Yd, tap=tap(c, s, n), group=tg, wait=True)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(L,), BF]] * 3 + [[f.prod() for f in f_a], [f.prod() for f in f_b],
                                                         [f.cons() for f in f_y]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def vup(X: In, Y: Out, *, H: CompileTime[int] = 64, W: CompileTime[int] = 64, CIN: CompileTime[int] = 1152,
        COUT: CompileTime[int] = 1152, MODE: CompileTime[int] = 0, GEN: CompileTime[int] = 0):
    """X: H x W x CIN -> Y: 2H x 2W x COUT. MODE 0 plain (COUT = CIN), 1: channel 2c+1, 2: channel 2c+i."""
    assert (MODE == 0 and COUT == CIN) or (MODE in (1, 2) and 2 * COUT == CIN)
    OI, OO = CIN // 8, COUT // 8
    P = seg_p(OI)
    KP = chunk_kp(OO, P, 2)
    NE, NK, NX = OO // KP, H // N_COLS, W // P
    s_ty, o_ty = np.ndarray[(OI * P * 8,), BF], np.ndarray[(KP * 2 * P * 8,), BF]
    fl = _flags(P=P, O=OI, KP=KP)
    names = {0: ("vup_plain", "vup_plain"), 1: ("vup_sel1", "vup_sel1"), 2: ("vup_sel0", "vup_sel1")}[MODE]
    f0 = ExternalFunction(names[0], object_file_name=f"vup_p{P}_o{OI}_kp{KP}.o", source_file=_SRC,
                          arg_types=[s_ty, o_ty], compile_flags=fl)
    f1 = f0.object_file.bind(names[1], [s_ty, o_ty]) if names[1] != names[0] else f0
    f_x = [ObjectFifo(s_ty, name=f"x_{c}", depth=2) for c in range(N_COLS)]
    f_y = [ObjectFifo(o_ty, name=f"y_{c}", depth=2) for c in range(N_COLS)]

    def core(x, y, *fns):
        for _ in range_(1 if GEN else NX * NK):   # GEN: one input segment per worker iteration
            ex = x.acquire(1)
            for fn in (fns[0], fns[-1]):  # output rows 2y and 2y+1
                for _ in range_(NE):
                    ey = y.acquire(1)
                    fn(ex, ey)
                    y.release(1)
            x.release(1)

    fn_args = [f0] if f1 is f0 else [f0, f1]
    workers = [Worker(core, [f_x[c].cons(), f_y[c].prod()] + fn_args, stack_size=0x400) for c in range(N_COLS)]
    PLI, PLO = (W + 2) * 8, (2 * W + 2) * 8
    RSI, RSO = OI * PLI, OO * PLO
    LI, LO = (H + 2) * RSI, (2 * H + 2) * RSO

    def sequence(Xd, Yd, x_prods, y_conses):
        for xs in range(NX):
            tg = TaskGroup()
            for c in range(N_COLS):
                x_prods[c].fill(Xd, tap=TensorAccessPattern(
                    (LI,), (c * NK + 1) * RSI + (xs * P + 1) * 8, [1, NK, OI, P * 8], [0, RSI, PLI, 1]),
                    group=tg, wait=False)
                y_conses[c].drain(Yd, tap=TensorAccessPattern(
                    (LO,), (2 * c * NK + 1) * RSO + (2 * xs * P + 1) * 8, [NK, 2, OO, 2 * P * 8],
                    [2 * RSO, RSO, PLO, 1]), group=tg, wait=True)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(LI,), BF], np.ndarray[(LO,), BF], [f.prod() for f in f_x],
                            [f.cons() for f in f_y]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def seg_p_down(planes):
    """vdown: pixels per segment of 2 rows ([2][planes][P][8] <= 18 KB)."""
    return next(p for p in (64, 32, 16, 8) if 2 * planes * p * 16 <= 18432)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def vdown(X: In, Y: Out, *, H: CompileTime[int] = 1024, W: CompileTime[int] = 1024, C: CompileTime[int] = 96,
          MODE: CompileTime[int] = 0, GEN: CompileTime[int] = 0):
    """2x2 average pool, X: H x W x C -> Y: H/2 x W/2 x (C, or 2C for MODE 1: out[2c] = 0, out[2c+1] = avg(in[c]))."""
    OI = C // 8
    OO = OI * (2 if MODE else 1)
    P = seg_p_down(OI)
    KP = chunk_kp(OO, P // 2, 1)
    if MODE and KP % 2:
        KP = next(k for k in range(KP - 1, 0, -1) if k % 2 == 0 and OO % k == 0)
    HO, WO = H // 2, W // 2
    NE, NK, NX = OO // KP, HO // N_COLS, W // P
    s_ty, o_ty = np.ndarray[(2 * OI * P * 8,), BF], np.ndarray[(KP * (P // 2) * 8,), BF]
    fn = ExternalFunction("vdown_zi" if MODE else "vdown_plain", object_file_name=f"vdown_p{P}_o{OI}_kp{KP}.o",
                          source_file=_SRC, arg_types=[s_ty, o_ty], compile_flags=_flags(P=P, O=OI, KP=KP))
    f_x = [ObjectFifo(s_ty, name=f"x_{c}", depth=2) for c in range(N_COLS)]
    f_y = [ObjectFifo(o_ty, name=f"y_{c}", depth=2) for c in range(N_COLS)]

    def core(x, y, fn):
        for _ in range_(1 if GEN else NX * NK):   # GEN: one input segment per worker iteration
            ex = x.acquire(1)
            for _ in range_(NE):
                ey = y.acquire(1)
                fn(ex, ey)
                y.release(1)
            x.release(1)

    workers = [Worker(core, [f_x[c].cons(), f_y[c].prod(), fn], stack_size=0x400) for c in range(N_COLS)]
    PLI, PLO = (W + 2) * 8, (WO + 2) * 8
    RSI, RSO = OI * PLI, OO * PLO
    LI, LO = (H + 2) * RSI, (HO + 2) * RSO

    def sequence(Xd, Yd, x_prods, y_conses):
        for xs in range(NX):
            tg = TaskGroup()
            for c in range(N_COLS):
                x_prods[c].fill(Xd, tap=TensorAccessPattern(
                    (LI,), (2 * c * NK + 1) * RSI + (xs * P + 1) * 8, [NK, 2, OI, P * 8], [2 * RSI, RSI, PLI, 1]),
                    group=tg, wait=False)
                y_conses[c].drain(Yd, tap=TensorAccessPattern(
                    (LO,), (c * NK + 1) * RSO + (xs * P // 2 + 1) * 8, [1, NK, OO, P // 2 * 8], [0, RSO, PLO, 1]),
                    group=tg, wait=True)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(LI,), BF], np.ndarray[(LO,), BF], [f.prod() for f in f_x],
                            [f.cons() for f in f_y]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="VAE glue")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("--op", required=True, choices=["vnorm", "vadd", "vup", "vdown"])
    for a, d in (("--h", 64), ("--w", 64), ("--c", 1152), ("--cout", 0), ("--silu", 1), ("--mode", 0), ("--l", 0),
                 ("--gen", 0)):
        p.add_argument(a, type=int, default=d)
    o = p.parse_args()
    if o.op == "vnorm":
        design, kw = vnorm, lambda o: dict(H=o.h, W=o.w, C=o.c, SILU=o.silu, GEN=o.gen)
    elif o.op == "vadd":
        design, kw = vadd, lambda o: dict(L=o.l or add_len(o.h, o.w, o.c), GEN=o.gen)
    elif o.op == "vdown":
        design, kw = vdown, lambda o: dict(H=o.h, W=o.w, C=o.c, MODE=o.mode, GEN=o.gen)
    else:
        design, kw = vup, lambda o: dict(H=o.h, W=o.w, CIN=o.c, COUT=o.cout or o.c, MODE=o.mode, GEN=o.gen)
    run_design_cli(design, o, compile_kwargs=kw, device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
