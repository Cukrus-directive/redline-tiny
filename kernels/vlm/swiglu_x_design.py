# VLM copy of kernels/atb3/n32_swiglu.py using mm_bfp_swiglu_x.cc (accurate SiLU).
# n32_swiglu.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Modified for redline-tiny (2026): derived from mlir-aie's block_datatypes GEMM examples (any-K / any-M runtime sequences, fused epilogues); see NOTICE.
#
"""Up GEMM + SwiGLU in one design (Qwen port of config3, see n32_core.py).

A (M x K, a_layout 128x64) @ B (K x N, the up weights with columns interleaved per 128-col tile as [gate 64 |
proj 64], b_layout 64x128) -> silu(gate) * proj written straight as the down GEMM's A operand: M x N/2 in
a_layout(., 128, 64) order (mm_bfp_swiglu.cc does the SwiGLU in its flush). Each C tile (M-tile i, N-tile j)
becomes down-A tile (i, j); every column drains its 4 tiles of a group to their A slots, one A strip apart.
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

_KERNEL_SRC = Path(__file__).resolve().parent / "mm_bfp_swiglu_x.cc"
_AIE_KERNELS_INC = Path(os.environ.get("MLIR_AIE_SRC", "")) / "aie_kernels"
DIV = 4


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def n32_swiglu_gemm(A: In, B: In, H: Out, *, M: CompileTime[int] = 4096, K: CompileTime[int] = 4096,
                    N: CompileTime[int] = 24576):
    m, k, n = 128, 64, 128
    n_aie_cols, n_aie_rows, tb_max_n_rows = 8, 4, 4
    v8 = np.dtype[v8bfp16ebs8]
    A_l2_ty = np.ndarray[(m, k // 8), v8]
    B_l2_ty = np.ndarray[(k, n // 8), v8]
    H_l2_ty = np.ndarray[(n_aie_rows * m, n // 2 // 8), v8]
    A_l1_ty = np.ndarray[(m // DIV, k // 8), v8]
    B_l1_ty = np.ndarray[(k, n // 8), v8]
    H_l1_ty = np.ndarray[(m, n // 2 // 8), v8]   # one down-A tile, 128 x 64 (stripe 0 accumulates in it too)

    matmul_kernel = ExternalFunction(
        "matmul_swiglu_bfp16", object_file_name=f"mm_bfp_swiglu_x_k{K}.o", source_file=str(_KERNEL_SRC),
        arg_types=[A_l1_ty, B_l1_ty, H_l1_ty], compile_flags=[f"-I{_AIE_KERNELS_INC}", f"-I{Path(__file__).resolve().parent.parent / 'attn'}", f"-DK_PROBLEMSIZE={K}"])
    zero_kernel = matmul_kernel.object_file.bind("zero_swiglu", [H_l1_ty])

    A_l3l2, A_l2l1, B_l3l2, B_l2l1 = [], [], [], []
    H_l1l2 = [[] for _ in range(n_aie_rows)]
    H_l2l3 = []
    for row in range(n_aie_rows):
        f = ObjectFifo(A_l2_ty, name=f"A_L3L2_{row}", depth=2)
        A_l3l2.append(f)
        A_l2l1.append(f.cons().forward(obj_type=A_l1_ty, name=f"A_L2L1_{row}", depth=2))
    for col in range(n_aie_cols):
        f = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=2)
        B_l3l2.append(f)
        B_l2l1.append(f.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", depth=2))
    t_out = m * (n // 2) // 8
    for col in range(n_aie_cols):
        f = ObjectFifo(H_l2_ty, name=f"H_L2L3_{col}", depth=2)
        H_l2l3.append(f)
        parts = f.prod().join([t_out * i for i in range(n_aie_rows)], obj_types=[H_l1_ty] * n_aie_rows,
                              names=[f"H_L1L2_{col}_{row}" for row in range(n_aie_rows)], depths=[1] * n_aie_rows)
        for j in range(n_aie_rows):
            H_l1l2[j].append(parts[j])

    tiles_per_core = (M // m) * (N // n) // (n_aie_cols * n_aie_rows)

    def core_fn(in_a, in_b, out_h, zero, matmul):
        for _ in range_(tiles_per_core) if tiles_per_core > 1 else range(1):
            eo = out_h.acquire(1)
            zero(eo)
            for _ in range_(K // k):
                eb = in_b.acquire(1)
                for _ in range(DIV):
                    ea = in_a.acquire(1)
                    matmul(ea, eb, eo)
                    in_a.release(1)
                in_b.release(1)
            out_h.release(1)

    # static data: c_stage (3 stripes x 32 x 128 bf16 = 24576 B) + a_tmp (2304 B) + 2 counters
    workers = Worker.grid(n_aie_rows, n_aie_cols, lambda row, col: Worker(
        core_fn, [A_l2l1[row].cons(), B_l2l1[col].cons(), H_l1l2[row][col].prod(), zero_kernel, matmul_kernel],
        stack_size=0x600, data_size=3 * 32 * 128 * 2 + 2304 + 8))

    A_ty = np.ndarray[(M * K // 8,), v8]
    B_ty = np.ndarray[(K * N // 8,), v8]
    H_ty = np.ndarray[(M * N // 2 // 8,), v8]
    A_taps = TensorAccessPattern.full((M * K // 8,)).tile((m * K // 8,))
    B_taps = TensorAccessPattern.full((N * K // 8,)).tile((n * K // 8,))
    num_row_tile, num_col_tile = M // m // n_aie_rows, N // n // n_aie_cols
    num_groups = num_row_tile * num_col_tile
    nj = N // n                                           # down-A tiles per 128-row strip
    total_out = M * N // 2 // 8

    def sequence(a, b, h, A_prods, B_prods, H_conses):
        slots = [None] * tb_max_n_rows
        for gi in range(num_groups):
            si = gi % tb_max_n_rows
            tg = TaskGroup()
            slots[si] = tg
            rg, cg = gi // num_col_tile, gi % num_col_tile
            for row in range(n_aie_rows):
                A_prods[row].fill(a, tap=A_taps[rg * n_aie_rows + row], group=tg, wait=False)
            for col in range(n_aie_cols):
                B_prods[col].fill(b, tap=B_taps[cg * n_aie_cols + col], group=tg, wait=False)
            for col in range(n_aie_cols):
                H_conses[col].drain(h, tap=TensorAccessPattern(
                    (total_out,), offset=(rg * n_aie_rows * nj + cg * n_aie_cols + col) * t_out,
                    sizes=[1, 1, n_aie_rows, t_out], strides=[0, 0, nj * t_out, 1]), group=tg, wait=True)
            # close task groups two at a time, like n32_core.py; also correct when num_groups % 4 != 0
            if si == 1 and gi != 1:
                slots[2].finish()
                slots[3].finish()
                slots[2] = slots[3] = None
            if si == 3:
                slots[0].finish()
                slots[1].finish()
                slots[0] = slots[1] = None
        for s in slots:
            if s is not None:
                s.finish()

    rt = Runtime(sequence, [A_ty, B_ty, H_ty, [f.prod() for f in A_l3l2], [f.prod() for f in B_l3l2],
                            [f.cons() for f in H_l2l3]])
    return Program(iron.get_current_device(), rt, workers=[w for row in workers for w in row]).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Up GEMM + SwiGLU (config3 + SwiGLU flush)")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-M", type=int, default=4096)
    p.add_argument("-K", type=int, default=4096)
    p.add_argument("-N", type=int, default=24576)
    opts = p.parse_args()
    run_design_cli(n32_swiglu_gemm, opts, compile_kwargs=lambda o: dict(M=o.M, K=o.K, N=o.N),
                   device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
