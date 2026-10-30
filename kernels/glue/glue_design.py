"""Qwen-Image glue kernels on the NPU (IRON, npu2): the element-wise work between the GEMMs and attention.

swiglu: gate|proj GEMM output (config3 C stream, weight columns interleaved per 128-col tile as [gate 64 | proj 64])
        -> silu(gate) * proj written straight into the down GEMM's A layout (128x64 a_layout tiles).
        One wave = one config3 output group (4 M-tiles x 8 N-tiles): column c gets the group's chunk c (4 tiles,
        contiguous), the memtile splits it to the column's 4 cores, and the 4 results are joined and drained to
        their A-tile slots (M-tile rg*4 + row, k-block cg*8 + c), one A strip apart.
Compile-only: python glue_design.py -d npu2 --xclbin-path X --insts-path Y --op swiglu --m 4096 --n 24576
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
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

_DIR = Path(__file__).resolve().parent
_INC = Path(os.environ.get("MLIR_AIE_SRC", "")) / "aie_kernels"
N_COLS, N_ROWS = 8, 4


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def swiglu(CIN: In, AOUT: Out, *, M: CompileTime[int] = 4096, N: CompileTime[int] = 24576):
    v8 = np.dtype[v8bfp16ebs8]
    t_in = 128 * 128 // 8   # one C tile, in v8bfp16ebs8 blocks
    t_out = 128 * 64 // 8   # one A tile
    n_rg, n_cg = M // (128 * N_ROWS), N // (128 * N_COLS)
    nj = N // 128           # A tiles per 128-row strip of the down GEMM (K = N/2, k = 64)
    in_ty, out_ty = np.ndarray[(t_in,), v8], np.ndarray[(t_out,), v8]

    kern = ExternalFunction("swiglu_tile", object_file_name="swiglu_bfp.o", source_file=str(_DIR / "swiglu_bfp.cc"),
                            arg_types=[in_ty, out_ty], compile_flags=[f"-I{_INC}"])
    fi, fo, l1_in, l1_out = [], [], [], []
    for c in range(N_COLS):
        a = ObjectFifo(np.ndarray[(N_ROWS * t_in,), v8], name=f"in_{c}", depth=2)
        fi.append(a)
        l1_in.append(a.cons().split([r * t_in for r in range(N_ROWS)], obj_types=[in_ty] * N_ROWS,
                                    names=[f"in_{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS))
        b = ObjectFifo(np.ndarray[(N_ROWS * t_out,), v8], name=f"out_{c}", depth=2)
        fo.append(b)
        l1_out.append(b.prod().join([r * t_out for r in range(N_ROWS)], obj_types=[out_ty] * N_ROWS,
                                    names=[f"out_{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS))

    def core(i, o, k):
        for _ in range_(n_rg * n_cg):
            ei = i.acquire(1)
            eo = o.acquire(1)
            k(ei, eo)
            i.release(1)
            o.release(1)

    workers = [Worker(core, [l1_in[c][r].cons(), l1_out[c][r].prod(), kern], stack_size=0x400)
               for c in range(N_COLS) for r in range(N_ROWS)]
    total_in, total_out = M * N // 8, M * N // 2 // 8

    def sequence(Cd, Ad, in_prods, out_conses):
        for rg in range(n_rg):
            for cg in range(n_cg):
                tg = TaskGroup()
                g = rg * n_cg + cg
                for c in range(N_COLS):
                    in_prods[c].fill(Cd, tap=TensorAccessPattern((total_in,), offset=(g * N_COLS + c) * N_ROWS * t_in,
                                                                 sizes=[1, 1, 1, N_ROWS * t_in], strides=[0, 0, 0, 1]),
                                     group=tg, wait=False)
                for c in range(N_COLS):
                    out_conses[c].drain(Ad, tap=TensorAccessPattern(
                        (total_out,), offset=(rg * N_ROWS * nj + cg * N_COLS + c) * t_out,
                        sizes=[1, 1, N_ROWS, t_out], strides=[0, 0, nj * t_out, 1]), group=tg, wait=True)
                tg.finish()

    rt = Runtime(sequence, [np.ndarray[(total_in,), v8], np.ndarray[(total_out,), v8],
                            [f.prod() for f in fi], [f.cons() for f in fo]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


HALF = 64 * 128 // 8          # one half-tile (64 tokens x 128 dims) in v8bfp16ebs8 blocks
TAB = 128 * 64 + 2 * 128      # qkv_post table per half-tile: 128 blocks x (32 cos + 32 sin), then w_q, w_k (bf16)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def qkvpost1(QKV: In, TABS: In, OUT: Out, *, T: CompileTime[int] = 4):
    """Single-core test of qkv_post.cc: T half-tiles of one head. QKV = per half [q, k, v] half-tiles (bfp16 C
    layout halves), TABS = per half one table, OUT = per half [Q tile, K tile, V tile] (attention layouts)."""
    v8 = np.dtype[v8bfp16ebs8]
    h_ty, tab_ty = np.ndarray[(HALF,), v8], np.ndarray[(TAB,), np.dtype[bfloat16]]
    src = str(_DIR / "qkv_post.cc")
    qk = ExternalFunction("q_post", object_file_name="qkv_post.o", source_file=src,
                          arg_types=[h_ty, tab_ty, h_ty], compile_flags=[f"-I{_INC}"])
    kk = qk.object_file.bind("k_post", [h_ty, tab_ty, h_ty])
    vk = qk.object_file.bind("v_post", [h_ty, h_ty])
    f_in = ObjectFifo(h_ty, name="qkv", depth=2)
    f_tab = ObjectFifo(tab_ty, name="tab", depth=1)
    f_out = ObjectFifo(h_ty, name="out", depth=1)  # L1: 2x9 KB in + 16.5 KB table + out + scratch; depth 2 won't fit

    def core(i, tb, o, qk, kk, vk):
        for _ in range_(T):
            et = tb.acquire(1)
            eq = i.acquire(1)
            eo = o.acquire(1)
            qk(eq, et, eo)
            o.release(1)
            i.release(1)
            ekv = i.acquire(2)  # acquire(1) twice would hand back the same element
            ek, ev = ekv[0], ekv[1]
            eo = o.acquire(1)
            kk(ek, et, eo)
            o.release(1)
            eo = o.acquire(1)
            vk(ev, eo)
            o.release(1)
            i.release(2)
            tb.release(1)

    w = Worker(core, [f_in.cons(), f_tab.cons(), f_out.prod(), qk, kk, vk], stack_size=0x600,
               data_size=32 * 64 * 2 + 72 + 256)

    def sequence(Xd, Td, Od, ih, th, oh):
        ih.fill(Xd)
        th.fill(Td)
        oh.drain(Od, wait=True)

    rt = Runtime(sequence, [np.ndarray[(3 * T * HALF,), v8], np.ndarray[(T * TAB,), np.dtype[bfloat16]],
                            np.ndarray[(3 * T * HALF,), v8], f_in.prod(), f_tab.prod(), f_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Qwen glue kernels")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("--op", default="swiglu", choices=["swiglu", "qkvpost1"])
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=24576)
    p.add_argument("--t", type=int, default=4, help="qkvpost1: half-tiles")
    opts = p.parse_args()
    if opts.op == "swiglu":
        design, kw = swiglu, lambda o: dict(M=o.m, N=o.n)
    else:
        design, kw = qkvpost1, lambda o: dict(T=o.t)
    run_design_cli(design, opts, compile_kwargs=kw, device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
