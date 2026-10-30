"""GELU glue on 8 cores (IRON, npu2): a GEMM's config3 C stream (M x NTB*128) -> GELU of its first NCT C tiles per
row as the next GEMM's A operand (M x KT*64, k-tiles 0..2*NCT-1; the host keeps a ones tile beyond them for the
bias row). Core c (column c, own shim fifos) owns M-tiles c, c + 8, ...; one kernel call per 16-row band.
Compile-only: python gelu_design.py -d npu2 --xclbin-path X --insts-path Y --m 4096 --ntb 40 --nct 34 --kt 69
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

_DIR = Path(__file__).resolve().parent
_INC = Path(os.environ.get("MLIR_AIE_SRC", "")) / "aie_kernels"
N_COLS = 8
TB, BB, AT = 128 * 128 // 8, 16 * 128 // 8, 128 * 64 // 8  # C tile, band, A tile (v8bfp16ebs8 blocks)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def gelu(C: In, A: Out, *, M: CompileTime[int] = 4096, NTB: CompileTime[int] = 40, NCT: CompileTime[int] = 34,
         KT: CompileTime[int] = 69):
    v8 = np.dtype[v8bfp16ebs8]
    b_ty = np.ndarray[(BB,), v8]
    n_mt = M // 128
    mine = [list(range(c, n_mt, N_COLS)) for c in range(N_COLS)]
    kern = ExternalFunction("gelu_band", object_file_name="gelu_bfp.o", source_file=str(_DIR / "gelu_bfp.cc"),
                            arg_types=[b_ty, b_ty],
                            compile_flags=[f"-I{_INC}", f"-I{_DIR.parent / 'attn'}"])

    def make_core(n):
        def core(i, o, k):
            for _ in range_(n * NCT * 8):
                ei, eo = i.acquire(1), o.acquire(1)
                k(ei, eo)
                i.release(1)
                o.release(1)
        return core

    fi, fo, workers = [], [], []
    act = [c for c in range(N_COLS) if mine[c]]
    for c in act:
        fi.append(ObjectFifo(b_ty, name=f"gi{c}", depth=2))
        fo.append(ObjectFifo(b_ty, name=f"go{c}", depth=2))
        workers.append(Worker(make_core(len(mine[c])), [fi[-1].cons(), fo[-1].prod(), kern], tile=Tile(c, 2),
                              stack_size=0x600))
    nc, na = M * NTB * 128 // 8, n_mt * KT * AT

    def sequence(Cd, Ad, ips, ocs):
        for r in range(max(len(mine[c]) for c in act)):
            tg = TaskGroup()
            for j, c in enumerate(act):
                if r >= len(mine[c]):
                    continue
                mt = mine[c][r]
                t0 = (mt // 4) * 4 * NTB + mt % 4
                ips[j].fill(Cd, tap=TensorAccessPattern((nc,), offset=t0 * TB, sizes=[1, 1, NCT, TB],
                                                        strides=[0, 0, 4 * TB, 1]), group=tg, wait=False)
                ocs[j].drain(Ad, tap=TensorAccessPattern((na,), offset=mt * KT * AT, sizes=[NCT, 8, 2, 128],
                                                         strides=[2 * AT, 128, AT, 1]), group=tg, wait=True)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(nc,), v8], np.ndarray[(na,), v8],
                            [f.prod(tile=Tile(c, 0)) for f, c in zip(fi, act)],
                            [f.cons(tile=Tile(c, 0)) for f, c in zip(fo, act)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="GELU glue")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--ntb", type=int, default=40)
    p.add_argument("--nct", type=int, default=34)
    p.add_argument("--kt", type=int, default=69)
    opts = p.parse_args()
    run_design_cli(gelu, opts, compile_kwargs=lambda o: dict(M=o.m, NTB=o.ntb, NCT=o.nct, KT=o.kt),
                   device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
