"""VLM copy of kernels/glue/ln_design.py with rms_mod.cc: residual add + RMSNorm * w (P row 0: g = 1, sc = w; S = 0).

Fused residual add + LayerNorm + modulate (ln_mod.cc) on the NPU (IRON, npu2), between the GEMMs.

ln_fused(X, BR, P, XO, AO):
  X  : residual stream x, bf16 M x 4096 in config3's C-stream order (npu_layouts.c_layout_atb(x, 128, 128)).
  BR : branch (out-proj / down GEMM output), bfp16, same order.
  P  : 2 x 8192 bf16 parameter rows [g = tanh(gate) 4096 | sc = 1 + scale 4096]; row 0 = real t, row 1 = t=0.
       128-row M-tiles < S read row 1, the rest row 0.
  XO : x' = x + g * BR (bf16-rounded), same order as X.
  AO : LayerNorm(x') * sc (eps 1e-6, no affine) as bfp16 in config3's A layout (a_layout(y, 128, 64)).
8 cores, one per column (x/params and branch in on the column's 2 shim MM2S channels, x' and A out on its 2 S2MM),
core c owns M-tiles c, c + 8, ...; per M-tile pass 1 (stats) and pass 2 (outputs) each stream the tile's 32 C tiles.
Compile-only: python ln_design.py -d npu2 --xclbin-path X --insts-path Y --m 4096 --s 0
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
N_COLS = 8
D = 4096                    # hidden width: 32 C tiles of 128 per row of tiles
TILE = 128 * 128            # one C tile, in elements
BAND = 16 * 128             # one 16-row band of a C tile


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def ln_fused(X: In, BR: In, P: In, XO: Out, AO: Out, *, M: CompileTime[int] = 4096, S: CompileTime[int] = 0):
    v8 = np.dtype[v8bfp16ebs8]
    x_ty = np.ndarray[(BAND,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(BAND // 8,), v8]
    br_ty = np.ndarray[(BAND,), np.dtype[bfloat16]]  # the branch arrives as bf16
    n_mt, n_nt = M // 128, D // 128
    tiles = [[mt for mt in range(n_mt) if mt % N_COLS == c] for c in range(N_COLS)]

    kp1 = ExternalFunction("ln_p1", object_file_name="rms16_mod.o", source_file=str(_DIR / "rms16_mod.cc"),
                           arg_types=[x_ty, br_ty], compile_flags=[f"-I{_INC}"])
    bind = kp1.object_file.bind
    kpar, kfin, kp2 = bind("ln_par", [x_ty]), bind("ln_fin", []), bind("ln_p2", [x_ty, br_ty, x_ty, b_ty])

    def make_core(n):
        def core(xf, bf, xo, ao, kpar, kp1, kfin, kp2):
            for _ in range_(n):
                for _ in range_(4):
                    e = xf.acquire(1)
                    kpar(e)
                    xf.release(1)
                for _ in range_(n_nt * 8):
                    ex, eb = xf.acquire(1), bf.acquire(1)
                    kp1(ex, eb)
                    xf.release(1)
                    bf.release(1)
                kfin()
                for _ in range_(n_nt * 8):
                    ex, eb, exo, eao = xf.acquire(1), bf.acquire(1), xo.acquire(1), ao.acquire(1)
                    kp2(ex, eb, exo, eao)
                    xf.release(1)
                    bf.release(1)
                    xo.release(1)
                    ao.release(1)
        return core

    fx, fb, fxo, fao, workers = [], [], [], [], []
    for c in range(N_COLS):
        fx.append(ObjectFifo(x_ty, name=f"lx{c}", depth=2))
        fb.append(ObjectFifo(br_ty, name=f"lb{c}", depth=2))
        fxo.append(ObjectFifo(x_ty, name=f"lxo{c}", depth=2))
        fao.append(ObjectFifo(b_ty, name=f"lao{c}", depth=2))
        workers.append(Worker(make_core(len(tiles[c])),
                              [fx[c].cons(), fb[c].cons(), fxo[c].prod(), fao[c].prod(), kpar, kp1, kfin, kp2],
                              tile=Tile(c, 2), stack_size=0x800, data_size=16384 + 8192 + 72 + 512))

    nx, nb = M * D, M * D // 8

    def t0(mt):  # index of C tile (mt, 0) in the C stream (tile (mt, nt) is t0 + 4 * nt)
        return (mt // 4) * 4 * n_nt + mt % 4

    def sequence(Xd, BRd, Pd, XOd, AOd, xps, bps, xocs, aocs):
        for r in range(max(len(t) for t in tiles)):
            tg = TaskGroup()
            for c in range(N_COLS):
                if r >= len(tiles[c]):
                    continue
                mt = tiles[c][r]
                xtap = TensorAccessPattern((nx,), offset=t0(mt) * TILE, sizes=[1, 1, n_nt, TILE],
                                           strides=[0, 0, 4 * TILE, 1])
                btap = xtap  # the bf16 branch has x's layout
                atap = TensorAccessPattern((nb,), offset=mt * 128 * D // 8, sizes=[n_nt, 8, 2, 128],
                                           strides=[2048, 128, 1024, 1])
                ptap = TensorAccessPattern((2 * 8192,), offset=(1 if mt < S else 0) * 8192, sizes=[1, 1, 1, 8192],
                                           strides=[0, 0, 0, 1])
                xps[c].fill(Pd, tap=ptap, group=tg, wait=False)
                for _ in range(2):
                    xps[c].fill(Xd, tap=xtap, group=tg, wait=False)
                    bps[c].fill(BRd, tap=btap, group=tg, wait=False)
                xocs[c].drain(XOd, tap=xtap, group=tg, wait=True)
                aocs[c].drain(AOd, tap=atap, group=tg, wait=True)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(nx,), np.dtype[bfloat16]], np.ndarray[(nx,), np.dtype[bfloat16]],
                            np.ndarray[(2 * 8192,), np.dtype[bfloat16]], np.ndarray[(nx,), np.dtype[bfloat16]],
                            np.ndarray[(nb,), v8],
                            [f.prod(tile=Tile(c, 0)) for c, f in enumerate(fx)],
                            [f.prod(tile=Tile(c, 0)) for c, f in enumerate(fb)],
                            [f.cons(tile=Tile(c, 0)) for c, f in enumerate(fxo)],
                            [f.cons(tile=Tile(c, 0)) for c, f in enumerate(fao)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Qwen residual + LayerNorm + modulate")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("--m", type=int, default=4096, help="rows (tokens), a multiple of 512")
    p.add_argument("--s", type=int, default=0, help="128-row M-tiles below this use parameter row 1 (t=0)")
    opts = p.parse_args()
    run_design_cli(ln_fused, opts, compile_kwargs=lambda o: dict(M=o.m, S=o.s),
                   device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
