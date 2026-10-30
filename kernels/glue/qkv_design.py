"""QKV post on all 32 cores (IRON, npu2): the QKV GEMM's config3 C stream -> attention Q tiles and K/V streams.

Core (col c, row j) owns head h = 4c + j for every 64-row half-tile t. Per 512-row wave (rg; t = rg*8 + t'):
  pass 1, t' = 0..7: table t' + q half-tile -> Q tile (q_post)
  pass 2, t' = 0..7: table t' + k half-tile -> K tile (k_post)
  pass 3, t' = 0..7: v half-tile -> V tile (v_post)
q/k/v half-tiles come per column from the shim (one gathered stream per column), the memtile splits each 4-tile
object to the column's cores; the cos/sin + weight tables (qkv_post.cc format) are broadcast to all 32 cores; each
core's output tiles are joined per column and drained by three 3-D patterns: Q objects to Q[h][t], K and V to
KV[h][t][K|V] (run_attn.pack_kv order per head). Passes keep every transfer 3-D: a 4-D one with a 1024-block
inner run splits into several BDs and overflows the shim's 16.
Buffers: CIN (M x 12288 bfp16, c_layout_atb 128x128), TABS (T x 8448 bf16, one table per half-tile),
         Q (32 x T x 1024 blocks), KV (32 x T x 2048 blocks); T = M / 64, M a multiple of 512.
Compile-only: python qkv_design.py -d npu2 --xclbin-path X --insts-path Y --m 4096
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
HALF = 64 * 128 // 8          # one half-tile / one attention tile, in v8bfp16ebs8 blocks
TAB = 128 * 64 + 2 * 128      # one qkv_post table (bf16)
N_QKV = 3 * 4096


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def qkv32(CIN: In, TABS: In, QO: Out, KVO: Out, *, M: CompileTime[int] = 4096, OUT_DEPTH: CompileTime[int] = 2):
    v8 = np.dtype[v8bfp16ebs8]
    T, RG = M // 64, M // 512
    n_cg = N_QKV // (128 * N_COLS)
    h_ty, h4_ty = np.ndarray[(HALF,), v8], np.ndarray[(N_ROWS * HALF,), v8]
    tab_ty = np.ndarray[(TAB,), np.dtype[bfloat16]]

    src = str(_DIR / "qkv_post.cc")
    qk = ExternalFunction("q_post", object_file_name="qkv_post.o", source_file=src,
                          arg_types=[h_ty, tab_ty, h_ty], compile_flags=[f"-I{_INC}"])
    kk = qk.object_file.bind("k_post", [h_ty, tab_ty, h_ty])
    vk = qk.object_file.bind("v_post", [h_ty, h_ty])

    of_tab = ObjectFifo(tab_ty, name="tab", depth=1)  # broadcast to all 32 cores
    f_in, f_out, l1_in, l1_out = [], [], [], []
    for c in range(N_COLS):
        a = ObjectFifo(h4_ty, name=f"in_{c}", depth=2)
        f_in.append(a)
        l1_in.append(a.cons().split([r * HALF for r in range(N_ROWS)], obj_types=[h_ty] * N_ROWS,
                                    names=[f"in_{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS))
        b = ObjectFifo(h4_ty, name=f"out_{c}", depth=2)
        f_out.append(b)
        l1_out.append(b.prod().join([r * HALF for r in range(N_ROWS)], obj_types=[h_ty] * N_ROWS,
                                    names=[f"out_{c}_{r}" for r in range(N_ROWS)], depths=[OUT_DEPTH] * N_ROWS))

    def core(i, tb, o, qk, kk, vk):
        for _ in range_(RG):
            for kern in (qk, kk):         # passes 1, 2: Q then K tiles, each with the half-tile's table
                for _ in range_(8):
                    et = tb.acquire(1)
                    ex = i.acquire(1)
                    eo = o.acquire(1)
                    kern(ex, et, eo)
                    o.release(1)
                    i.release(1)
                    tb.release(1)
            for _ in range_(8):           # pass 3: V tiles (no table)
                ex = i.acquire(1)
                eo = o.acquire(1)
                vk(ex, eo)
                o.release(1)
                i.release(1)

    workers = [Worker(core, [l1_in[c][r].cons(), of_tab.cons(), l1_out[c][r].prod(), qk, kk, vk],
                      stack_size=0x600, data_size=32 * 64 * 2 + 72 + 256)
               for c in range(N_COLS) for r in range(N_ROWS)]

    cin_all, tab_all, q_all, kv_all = M * N_QKV // 8, T * TAB, 32 * T * HALF, 32 * T * 2 * HALF
    tile = 128 * 128 // 8

    def c_tile(mt, nt):  # block offset of C tile (M-tile mt, N-tile nt) in the config3 stream
        return (((mt // N_ROWS) * n_cg + nt // N_COLS) * N_COLS + nt % N_COLS) * N_ROWS * tile + (mt % N_ROWS) * tile

    def sequence(Cd, Td, Qd, KVd, in_prods, tab_prod, out_conses):
        for rg in range(RG):
            tg = TaskGroup()
            for rep in range(2):  # the tables, once per pass
                tab_prod.fill(Td, tap=TensorAccessPattern((tab_all,), offset=rg * 8 * TAB, sizes=[1, 1, 8, TAB],
                                                          strides=[0, 0, TAB, 1]), group=tg, wait=False)
            for c in range(N_COLS):
                for part in range(3):  # q, k, v half-tiles of the column's 4 heads, t' = 0..7
                    in_prods[c].fill(Cd, tap=TensorAccessPattern(
                        (cin_all,), offset=c_tile(rg * N_ROWS, 32 * part + 4 * c), sizes=[1, 8, N_ROWS, HALF],
                        strides=[0, HALF, N_ROWS * tile, 1]), group=tg, wait=False)
            for c in range(N_COLS):
                base = 4 * c * T + rg * 8
                out_conses[c].drain(Qd, tap=TensorAccessPattern((q_all,), offset=base * HALF, sizes=[1, 8, N_ROWS, HALF],
                                                                strides=[0, HALF, T * HALF, 1]),
                                    group=tg, wait=True)
                for kv in range(2):  # K then V tiles into KV[h][t][K|V]
                    out_conses[c].drain(KVd, tap=TensorAccessPattern(
                        (kv_all,), offset=base * 2 * HALF + kv * HALF, sizes=[1, 8, N_ROWS, HALF],
                        strides=[0, 2 * HALF, T * 2 * HALF, 1]), group=tg, wait=True)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(cin_all,), v8], np.ndarray[(tab_all,), np.dtype[bfloat16]],
                            np.ndarray[(q_all,), v8], np.ndarray[(kv_all,), v8],
                            [f.prod() for f in f_in], of_tab.prod(), [f.cons() for f in f_out]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Qwen QKV post, 32 cores")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--out-depth", type=int, default=2)
    opts = p.parse_args()
    run_design_cli(qkv32, opts, compile_kwargs=lambda o: dict(M=o.m, OUT_DEPTH=o.out_depth),
                   device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
