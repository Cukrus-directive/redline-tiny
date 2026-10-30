# VLM QKV post for the C16 QKV GEMM: bf16 half-tiles in (qkv16_post.cc), input depth 1 (L1).
"""QKV post for a GQA model on all 32 cores (IRON, npu2): VLM copy of kernels/glue/qkv_design.py for 32 q heads and
8 kv heads (Qwen3-VL text model). Kernels unchanged (kernels/glue/qkv_post.cc).

QKV GEMM C stream: M x 6144 = [q 32 x 128 | k 8 x 128 | v 8 x 128], config3 order (c_layout_atb 128x128).
Core (col c, row j): Q pass for q head 4c + j over the wave's 8 half-tiles (as in qkv_design.py); K and V passes for
kv head c, the column's 4 cores taking half-tiles t' = 4p + j (p = 0, 1). Tables are broadcast to all cores (every
core consumes all 8 per pass and uses its own).
Buffers: CIN (M x 6144 bfp16), TABS (T x 8448 bf16), Q (32 x T x 1024 blocks), KV (8 x T x 2048 blocks: [K | V] per
tile, run_attn.pack_kv order per kv head); T = M / 64, M a multiple of 512.
Compile-only: python qkv_gqa_design.py -d npu2 --xclbin-path X --insts-path Y --m 1536
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

_GLUE = Path(__file__).resolve().parent.parent / "glue"
_INC = Path(os.environ.get("MLIR_AIE_SRC", "")) / "aie_kernels"
N_COLS, N_ROWS = 8, 4
HQ, HKV = 32, 8
HALF = 64 * 128 // 8
TAB = 128 * 64 + 2 * 128
N_QKV = (HQ + 2 * HKV) * 128


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def qkv_gqa(CIN: In, TABS: In, QO: Out, KVO: Out, *, M: CompileTime[int] = 1536):
    """bf16 input (qkv16_post.cc): the C16 QKV GEMM's output."""
    v8 = np.dtype[v8bfp16ebs8]
    T, RG = M // 64, M // 512
    n_cg = N_QKV // (128 * N_COLS)
    h_ty, h4_ty = np.ndarray[(HALF,), v8], np.ndarray[(N_ROWS * HALF,), v8]
    hi_ty, hi4_ty = np.ndarray[(HALF * 8,), np.dtype[bfloat16]], np.ndarray[(N_ROWS * HALF * 8,), np.dtype[bfloat16]]
    tab_ty = np.ndarray[(TAB,), np.dtype[bfloat16]]

    src = str(Path(__file__).resolve().parent / "qkv16_post.cc")
    qk = ExternalFunction("q_post", object_file_name="qkv16_post.o", source_file=src,
                          arg_types=[hi_ty, tab_ty, h_ty], compile_flags=[f"-I{_INC}"])
    kk = qk.object_file.bind("k_post", [hi_ty, tab_ty, h_ty])
    vk = qk.object_file.bind("v_post", [hi_ty, h_ty])

    of_tab = ObjectFifo(tab_ty, name="tab", depth=1)
    f_in, f_out, l1_in, l1_out = [], [], [], []
    for c in range(N_COLS):
        a = ObjectFifo(hi4_ty, name=f"in_{c}", depth=1)
        f_in.append(a)
        l1_in.append(a.cons().split([r * HALF * 8 for r in range(N_ROWS)], obj_types=[hi_ty] * N_ROWS,
                                    names=[f"in_{c}_{r}" for r in range(N_ROWS)], depths=[1] * N_ROWS))
        b = ObjectFifo(h4_ty, name=f"out_{c}", depth=2)
        f_out.append(b)
        l1_out.append(b.prod().join([r * HALF for r in range(N_ROWS)], obj_types=[h_ty] * N_ROWS,
                                    names=[f"out_{c}_{r}" for r in range(N_ROWS)], depths=[1] * N_ROWS))

    def make_core(j):
        def core(i, tb, o, qk, kk, vk):
            for _ in range_(RG):
                for _ in range_(8):              # Q: own q head, every half-tile of the wave
                    et = tb.acquire(1)
                    ex = i.acquire(1)
                    eo = o.acquire(1)
                    qk(ex, et, eo)
                    o.release(1)
                    i.release(1)
                    tb.release(1)
                for t in range(8):               # K: the column's kv head, half-tiles t' = j, j + 4
                    et = tb.acquire(1)
                    if t % 4 == j:
                        ex = i.acquire(1)
                        eo = o.acquire(1)
                        kk(ex, et, eo)
                        o.release(1)
                        i.release(1)
                    tb.release(1)
                for _ in range(2):               # V: same half-tiles, no table
                    ex = i.acquire(1)
                    eo = o.acquire(1)
                    vk(ex, eo)
                    o.release(1)
                    i.release(1)
        return core

    workers = [Worker(make_core(r), [l1_in[c][r].cons(), of_tab.cons(), l1_out[c][r].prod(), qk, kk, vk],
                      stack_size=0x600, data_size=32 * 64 * 2 + 72 + 256)
               for c in range(N_COLS) for r in range(N_ROWS)]

    cin_all, tab_all, q_all, kv_all = M * N_QKV, T * TAB, HQ * T * HALF, HKV * T * 2 * HALF  # CIN in bf16 elements
    tile = 128 * 128 // 8

    def c_tile(mt, nt):
        return 8 * ((((mt // N_ROWS) * n_cg + nt // N_COLS) * N_COLS + nt % N_COLS) * N_ROWS * tile + (mt % N_ROWS) * tile)

    def sequence(Cd, Td, Qd, KVd, in_prods, tab_prod, out_conses):
        for rg in range(RG):
            tg = TaskGroup()
            for rep in range(2):
                tab_prod.fill(Td, tap=TensorAccessPattern((tab_all,), offset=rg * 8 * TAB, sizes=[1, 1, 8, TAB],
                                                          strides=[0, 0, TAB, 1]), group=tg, wait=False)
            for c in range(N_COLS):
                in_prods[c].fill(Cd, tap=TensorAccessPattern(       # q half-tiles of heads 4c..4c+3
                    (cin_all,), offset=c_tile(rg * N_ROWS, 4 * c), sizes=[1, 8, N_ROWS, HALF * 8],
                    strides=[0, HALF * 8, N_ROWS * tile * 8, 1]), group=tg, wait=False)
                for nt in (HQ + c, HQ + HKV + c):                    # k, then v: kv head c, 8 contiguous half-tiles
                    in_prods[c].fill(Cd, tap=TensorAccessPattern(
                        (cin_all,), offset=c_tile(rg * N_ROWS, nt), sizes=[1, 1, 1, 8 * HALF * 8],
                        strides=[0, 0, 0, 1]), group=tg, wait=False)
            for c in range(N_COLS):
                out_conses[c].drain(Qd, tap=TensorAccessPattern(
                    (q_all,), offset=(4 * c * T + rg * 8) * HALF, sizes=[1, 8, N_ROWS, HALF],
                    strides=[0, HALF, T * HALF, 1]), group=tg, wait=True)
                for kv in range(2):  # objects p = 0, 1; slot j = tile t' = 4p + j -> KV[c][rg*8 + t'][kv]
                    out_conses[c].drain(KVd, tap=TensorAccessPattern(
                        (kv_all,), offset=(c * T + rg * 8) * 2 * HALF + kv * HALF, sizes=[1, 2, N_ROWS, HALF],
                        strides=[0, 4 * 2 * HALF, 2 * HALF, 1]), group=tg, wait=True)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(cin_all,), np.dtype[bfloat16]], np.ndarray[(tab_all,), np.dtype[bfloat16]],
                            np.ndarray[(q_all,), v8], np.ndarray[(kv_all,), v8],
                            [f.prod() for f in f_in], of_tab.prod(), [f.cons() for f in f_out]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="QKV post, GQA 32/8, 32 cores")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("--m", type=int, default=1536)
    opts = p.parse_args()
    run_design_cli(qkv_gqa, opts, compile_kwargs=lambda o: dict(M=o.m),
                   device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
