"""VAE decoder conv on 32 cores (IRON, npu2), v2: 3x3 or 1x1, stride 1.

Tensors are bf16 octet-planar with a 1-pixel zero border: [H+2][C/8][W+2][8].
Grid: 8 columns x R core rows. Column c computes the output rows c*H/8 .. (c+1)*H/8 - 1; its R cores take the same pixels
and R different output-channel chunks of NC (core row r: chunk g*R + r in channel group g). Per (group g,
x-segment of P pixels):
  strips : per column one DMA pattern [k (H/8 rows)][dy (TAPS)][octet (Cin/8)][(P+2)*8 contiguous] from the
           input, broadcast to the column's R cores (no reorder needed: octet-planar runs are the kernel order)
  weights: per core row one stream (chunk g*R + r: TAPS*Cin/KC step elements + 1 bias element), sent once per
           output row, broadcast to the row's 8 cores
  outputs: the column's R tiles joined in the memtile, [r][j][px][8], drained to planes g*R*NC/8 + r*NC/8 + j.
Compile-only: python vconv_design.py -d npu2 --xclbin-path X --insts-path Y --h 64 --w 64 --cin 1152 --cout 1152
"""

import argparse
import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker,
                       WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

_DIR = Path(__file__).resolve().parent
_INC = Path(os.environ.get("MLIR_AIE_SRC", "")) / "aie_kernels"
N_COLS = 8


def weight_layout(cin, cout, taps, kc, nc):
    """(steps per tile, V9 blocks per step element, V9 blocks per output-channel chunk incl. the bias element)."""
    nsteps = taps * cin // kc
    welem = taps * kc * nc // 8
    return nsteps, welem, (nsteps + 1) * welem


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def vconv(X: In, WT: In, Y: Out, *, H: CompileTime[int] = 64, W: CompileTime[int] = 64,
          CIN: CompileTime[int] = 1152, COUT: CompileTime[int] = 1152, TAPS: CompileTime[int] = 3,
          P: CompileTime[int] = 64, KC: CompileTime[int] = 48, NC: CompileTime[int] = 48,
          R: CompileTime[int] = 4, WREP: CompileTime[int] = 1, CIN_TOT: CompileTime[int] = 0,
          CIN0: CompileTime[int] = 0, DS: CompileTime[int] = 0, GEN: CompileTime[int] = 0):
    """GEN = 1: shape-generic cores. The tile count and the K steps per tile arrive as runtime parameters (written
    by the sequence, behind a barrier, as upstream's whole_array does), so the xclbin depends only on (P, KC, NC, TAPS,
    R, DS) and every shape of that config runs on the same xclbin with its own insts.bin (one hardware context).
    CIN_TOT/CIN0: the input tensor has CIN_TOT channels and the conv reads CIN of them from channel CIN0 (the
    attention's q inside the qkv output); 0 = CIN. DS = 1: the encoder's downsample (ZeroPad(0,1,0,1) + 3x3 stride
    2), i.e. the stride-1 conv at odd rows and pixels: output H/2 x W/2 (H, W are the input size)."""
    CIN_TOT = CIN_TOT or CIN
    assert H % N_COLS == 0 and W % P == 0 and CIN % KC == 0 and COUT % (R * NC) == 0, "shape"
    HO, WO, PO = (H // 2, W // 2, P // 2) if DS else (H, W, P)
    PW, NK, NX, NG = P + 2, HO // N_COLS, W // P, COUT // (R * NC)
    nsteps, welem, wchunk = weight_layout(CIN, COUT, TAPS, KC, NC)
    bf, v8 = np.dtype[bfloat16], np.dtype[v8bfp16ebs8]
    strip_ty = np.ndarray[(PW * KC,), bf]
    w_ty = np.ndarray[(welem,), v8]
    o_ty = np.ndarray[(PO * NC,), bf]
    oR_ty = np.ndarray[(R * PO * NC,), bf]

    src = str(_DIR / "vconv.cc")
    flags = [f"-I{_INC}", f"-DVC_P={P}", f"-DVC_KC={KC}", f"-DVC_NC={NC}", f"-DVC_TAPS={TAPS}"]
    flags += ["-DVC_DS"] if DS else []
    step = ExternalFunction("vconv_step", object_file_name=f"vconv_p{P}_k{KC}_n{NC}_t{TAPS}{'_ds' if DS else ''}.o",
                            source_file=src, arg_types=[strip_ty, w_ty], compile_flags=flags)
    zero = step.object_file.bind("vconv_zero", [])
    outk = step.object_file.bind("vconv_out_ds" if DS else "vconv_out", [w_ty, o_ty])

    f_s = [ObjectFifo(strip_ty, name=f"strip_{c}", depth=2) for c in range(N_COLS)]
    f_w = [ObjectFifo(w_ty, name=f"w_{r}", depth=2) for r in range(R)]
    f_o, o_l1 = [], []
    for c in range(N_COLS):
        fo = ObjectFifo(oR_ty, name=f"o_{c}", depth=2)
        f_o.append(fo)
        o_l1.append(fo.prod().join([r * PO * NC for r in range(R)], obj_types=[o_ty] * R,
                                   names=[f"o_{c}_{r}" for r in range(R)], depths=[1] * R))

    rtps = [Buffer(np.ndarray[(2,), np.dtype[np.int32]], name=f"rtp_{c}_{r}", use_write_rtp=True)
            for c in range(N_COLS) for r in range(R)] if GEN else []
    bars = [WorkerRuntimeBarrier() for _ in rtps]

    def core(s, w, o, step, zero, outk, *rt):
        n_tiles, n_steps = NG * NX * NK, nsteps
        if rt:
            rt[1].wait_for_value(1)
            n_tiles, n_steps = rt[0][0], rt[0][1]
            rt[1].release_with_value(1)
        for _ in range_(n_tiles):
            zero()
            for _ in range_(n_steps):
                es, ew = s.acquire(1), w.acquire(1)
                step(es, ew)
                s.release(1)
                w.release(1)
            eb, eo = w.acquire(1), o.acquire(1)
            outk(eb, eo)
            w.release(1)
            o.release(1)

    data = P * NC * 4 + (P // 16) * TAPS * (KC // 8) * 2 * 72 + 256
    workers = [Worker(core, [f_s[c].cons(), f_w[r].cons(), o_l1[c][r].prod(), step, zero, outk]
                      + ([rtps[c * R + r], bars[c * R + r]] if GEN else []), stack_size=0x400, data_size=data)
               for c in range(N_COLS) for r in range(R)]

    PL = (W + 2) * 8                                       # one plane of one row (elements)
    PLo = (WO + 2) * 8
    RS, RSo = (CIN_TOT // 8) * PL, (COUT // 8) * PLo        # input / output row strides
    x_len, y_len, w_len = (H + 2) * RS, (HO + 2) * RSo, COUT // NC * wchunk
    dy0 = 0 if TAPS == 3 else 1     # 1x1 convs read only the centre row
    ys = 2 if DS else 1             # input rows per output row (DS: full-res row 2y+1 = padded rows 2y+1..2y+3)
    y0 = 1 if DS else 0

    def sequence(Xd, Wd, Yd, s_prods, w_prods, o_conses):
        for b, bar in zip(rtps, bars):
            b[0] = NG * NX * NK
            b[1] = nsteps
            bar.set(1)
        for g in range(NG):
            for xs in range(NX):
                tg = TaskGroup()
                for c in range(N_COLS):
                    s_prods[c].fill(Xd, tap=TensorAccessPattern(
                        (x_len,), offset=(ys * c * NK + y0 + dy0) * RS + CIN0 // 8 * PL + xs * P * 8,
                        sizes=[NK, TAPS, CIN // 8, PW * 8], strides=[ys * RS, RS, PL, 1]), group=tg, wait=False)
                for r in range(R):
                    off = (g * R + r) * wchunk
                    if WREP:  # one pattern repeating the chunk for each of the column's NK output rows
                        w_prods[r].fill(Wd, tap=TensorAccessPattern((w_len,), offset=off, sizes=[1, 1, NK, wchunk],
                                                                    strides=[0, 0, 0, 1]), group=tg, wait=False)
                    else:
                        for _ in range(NK):
                            w_prods[r].fill(Wd, tap=TensorAccessPattern((w_len,), offset=off, sizes=[1, 1, 1, wchunk],
                                                                        strides=[0, 0, 0, 1]), group=tg, wait=False)
                for c in range(N_COLS):
                    o_conses[c].drain(Yd, tap=TensorAccessPattern(
                        (y_len,), offset=(c * NK + 1) * RSo + g * R * NC // 8 * PLo + (xs * PO + 1) * 8,
                        sizes=[1, NK, R * NC // 8, PO * 8], strides=[0, RSo, PLo, 1]), group=tg, wait=True)
                tg.finish()

    rt = Runtime(sequence, [np.ndarray[(x_len,), bf], np.ndarray[(w_len,), v8], np.ndarray[(y_len,), bf],
                            [f.prod() for f in f_s], [f.prod() for f in f_w], [f.cons() for f in f_o]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="VAE conv v2 (32 cores)")
    add_compile_args(p, default_dev="npu2")
    for a, d in (("--h", 64), ("--w", 64), ("--cin", 1152), ("--cout", 1152), ("--taps", 3), ("--p", 64),
                 ("--kc", 48), ("--nc", 48), ("--r", 4), ("--wrep", 1), ("--cin-tot", 0), ("--cin0", 0), ("--ds", 0),
                 ("--gen", 0)):
        p.add_argument(a, type=int, default=d)
    opts = p.parse_args()
    run_design_cli(vconv, opts, compile_kwargs=lambda o: dict(H=o.h, W=o.w, CIN=o.cin, COUT=o.cout, TAPS=o.taps,
                                                                P=o.p, KC=o.kc, NC=o.nc, R=o.r, WREP=o.wrep,
                                                                CIN_TOT=o.cin_tot, CIN0=o.cin0, DS=o.ds, GEN=o.gen),
                   device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
