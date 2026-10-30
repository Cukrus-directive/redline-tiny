"""Vision-tower residual add + affine LayerNorm on 8 cores (IRON, npu2), kernels from vln_mod.cc (a copy of the DiT's
ln_mod.cc design, one core per column with its own shim fifos).

vln(X, BR, P, XO, AO): X, BR, XO hold rows of NTB*128 channels in config3's C-stream order (only the first 9 C
tiles = 1152 channels are real); P = [w | b] for one virtual row, zero-padded to 2048-value chunks.
A virtual row is 1 or 4 segments of one 128-row M-tile's 9 C tiles; the kernel normalises over the whole virtual
row (VLN_DIV channels) and writes each segment's 18 A tiles at A row `amt`, k-tile offset `koff`:
  block: M-tile mt -> A row mt, k 0 (A = M x 1216, k-tile 18 is the host's ones tile)
  merge: M-tile mt = p*8 + m -> A row m, k p*18 (per-patch LN into the merger's A = 1024 x 4672, ones at k 72)
  post : M-tiles p*8 + m, p = 0..3 -> one virtual row (LN over 4608) -> A row m, k p*18
Patches are stored patch-in-block-major (row = p*1024 + merged token), so merging is A-tile re-indexing.
Compile-only: python vln_design.py -d npu2 --xclbin-path X --insts-path Y --mode block --m 4096 --ntb 16
"""

import argparse
import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

_DIR = Path(__file__).resolve().parent
_INC = Path(os.environ.get("MLIR_AIE_SRC", "")) / "aie_kernels"
N_COLS, NTS = 8, 9
TILE, BAND, AT = 128 * 128, 16 * 128, 128 * 64 // 8   # C tile values, band values, A tile blocks
MODES = {"block": (1, 19), "merge": (1, 73), "post": (4, 73)}  # segments per virtual row, A k-tiles per row


def vrows(mode, M):
    n_mt = M // 128
    if mode == "block":
        return [[(mt, mt, 0)] for mt in range(n_mt)]
    if mode == "merge":
        return [[(mt, mt % 8, (mt // 8) * 18)] for mt in range(n_mt)]
    return [[(p * 8 + m, m, p * 18) for p in range(4)] for m in range(n_mt // 4)]


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def vln(X: In, BR: In, P: In, XO: Out, AO: Out, *, M: CompileTime[int] = 4096, NTB: CompileTime[int] = 16,
        MODE: CompileTime[int] = 0):
    mode = list(MODES)[MODE]
    S, KT = MODES[mode]
    NT = S * NTS
    npc = (NT * 256 + 2047) // 2048
    v8 = np.dtype[v8bfp16ebs8]
    x_ty, b_ty = np.ndarray[(BAND,), np.dtype[bfloat16]], np.ndarray[(BAND // 8,), v8]
    rows = vrows(mode, M)
    mine = [rows[c::N_COLS] for c in range(N_COLS)]

    kp1 = ExternalFunction("ln_p1", object_file_name=f"vln_mod_nt{NT}.o", source_file=str(_DIR / "vln_mod.cc"),
                           arg_types=[x_ty, b_ty], compile_flags=[f"-I{_INC}", f"-DVLN_NT={NT}", f"-DVLN_DIV={NT * 128}"])
    bind = kp1.object_file.bind
    kpar, kfin, kp2 = bind("ln_par", [x_ty]), bind("ln_fin", []), bind("ln_p2", [x_ty, b_ty, x_ty, b_ty])

    def make_core(n):
        def core(xf, bf, xo, ao, kpar, kp1, kfin, kp2):
            for _ in range_(n) if n > 1 else range(1):
                for _ in range_(npc) if npc > 1 else range(1):
                    e = xf.acquire(1)
                    kpar(e)
                    xf.release(1)
                for _ in range_(NT * 8):
                    ex, eb = xf.acquire(1), bf.acquire(1)
                    kp1(ex, eb)
                    xf.release(1)
                    bf.release(1)
                kfin()
                for _ in range_(NT * 8):
                    ex, eb, exo, eao = xf.acquire(1), bf.acquire(1), xo.acquire(1), ao.acquire(1)
                    kp2(ex, eb, exo, eao)
                    xf.release(1)
                    bf.release(1)
                    xo.release(1)
                    ao.release(1)
        return core

    fx, fb, fxo, fao, workers = [], [], [], [], []
    for c in range(N_COLS):
        fx.append(ObjectFifo(x_ty, name=f"vx{c}", depth=2))
        fb.append(ObjectFifo(b_ty, name=f"vb{c}", depth=2))
        fxo.append(ObjectFifo(x_ty, name=f"vxo{c}", depth=2))
        fao.append(ObjectFifo(b_ty, name=f"vao{c}", depth=2))
        if mine[c]:
            workers.append(Worker(make_core(len(mine[c])),
                                  [fx[c].cons(), fb[c].cons(), fxo[c].prod(), fao[c].prod(), kpar, kp1, kfin, kp2],
                                  tile=Tile(c, 2), stack_size=0x800, data_size=npc * 4096 + 8192 + 72 + 512))
    act = [c for c in range(N_COLS) if mine[c]]
    nx, nb, na = M * NTB * 128, M * NTB * 128 // 8, (M // 128 if mode == "block" else M // 512) * KT * AT

    def seg_taps(segs):
        """(x tap, branch tap) over the virtual row's segments: tile (mt, nt) sits at (mt//4)*4*NTB + nt*4 + mt%4."""
        t0 = lambda mt: (mt // 4) * 4 * NTB + mt % 4
        mts = [s[0] for s in segs]
        if len(mts) == 1:
            sz, st = [1, 1, NTS, TILE], [0, 0, 4 * TILE, 1]
        else:  # the 4 patch tiles p*8 + m: a constant stride of 8*NTB tiles
            sz, st = [1, len(mts), NTS, TILE], [0, (t0(mts[1]) - t0(mts[0])) * TILE, 4 * TILE, 1]
        x = TensorAccessPattern((nx,), offset=t0(mts[0]) * TILE, sizes=sz, strides=st)
        b = TensorAccessPattern((nb,), offset=t0(mts[0]) * TILE // 8, sizes=[z if i < 3 else z // 8 for i, z in enumerate(sz)],
                                strides=[z // 8 for z in st[:3]] + [1])
        return x, b

    def sequence(Xd, BRd, Pd, XOd, AOd, xps, bps, xocs, aocs):
        for r in range(max(len(mine[c]) for c in act)):
            tg = TaskGroup()
            for c in act:
                if r >= len(mine[c]):
                    continue
                segs = mine[c][r]
                xt, bt = seg_taps(segs)
                xps[c].fill(Pd, tap=TensorAccessPattern((npc * 2048,), offset=0, sizes=[1, 1, 1, npc * 2048],
                                                        strides=[0, 0, 0, 1]), group=tg, wait=False)
                for _ in range(2):
                    xps[c].fill(Xd, tap=xt, group=tg, wait=False)
                    bps[c].fill(BRd, tap=bt, group=tg, wait=False)
                xocs[c].drain(XOd, tap=xt, group=tg, wait=True)
                for (_, amt, koff) in segs:
                    aocs[c].drain(AOd, tap=TensorAccessPattern((na,), offset=(amt * KT + koff) * AT,
                                                               sizes=[NTS, 8, 2, 128], strides=[2 * AT, 128, AT, 1]),
                                  group=tg, wait=True)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(nx,), np.dtype[bfloat16]], np.ndarray[(nb,), v8],
                            np.ndarray[(npc * 2048,), np.dtype[bfloat16]], np.ndarray[(nx,), np.dtype[bfloat16]],
                            np.ndarray[(na,), v8],
                            [fx[c].prod(tile=Tile(c, 0)) for c in act], [fb[c].prod(tile=Tile(c, 0)) for c in act],
                            [fxo[c].cons(tile=Tile(c, 0)) for c in act], [fao[c].cons(tile=Tile(c, 0)) for c in act]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Vision residual + affine LayerNorm")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("--m", type=int, default=4096, help="patch rows in X / BR / XO")
    p.add_argument("--ntb", type=int, default=16, help="C tiles per row in X / BR / XO (buffer N / 128)")
    p.add_argument("--mode", default="block", choices=list(MODES))
    opts = p.parse_args()
    run_design_cli(vln, opts, compile_kwargs=lambda o: dict(M=o.m, NTB=o.ntb, MODE=list(MODES).index(o.mode)),
                   device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
