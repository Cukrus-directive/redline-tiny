"""Causal GQA attention for the Qwen3-VL text model on all 32 cores (IRON, npu2), kernels from attn_causal.cc
(the DiT flash-attention kernel, OA build: O written as the o_proj GEMM's bfp16 A operand).

Column c holds kv head c; its core in row j computes q head 4c + j. For Q tile i (64 queries) the column's K/V
stream carries kv head c's tiles 0..NQT-1 (broadcast to the 4 cores) and the kernel computes j < i, masks j == i
above the diagonal and skips j > i (attn_*_causal, counters in the kernel). Only the first NQT tiles are computed
(the prompt's length); O rows beyond them stay as they are (the host zeroes the buffer once).
Buffers: Q (32 x T x 1024 blocks, qkv_gqa's Q), KV (8 x T x 2048 blocks: [K | V] per tile), O (a_layout(O_all, 128,
64) of the M x 4096 attention output, bfp16); T = M / 64.
Compile-only: python attn_causal_design.py -d npu2 --xclbin-path X --insts-path Y --m 1536 --nqt 17
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

_SRC = Path(__file__).resolve().parent / "attn_causal.cc"
_INC = Path(os.environ.get("MLIR_AIE_SRC", "")) / "aie_kernels"
N_COLS, N_ROWS = 8, 4
MQ = NT = 64
D, HQ, HKV, HID = 128, 32, 8, 4096
QB = MQ * D // 8           # one Q / K / V / O tile, in v8bfp16ebs8 blocks
AT = 128 * 64 // 8         # one A tile of the o_proj GEMM


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def attn_causal(Q: In, KV: In, O: Out, *, M: CompileTime[int] = 1536, NQT: CompileTime[int] = 17,
                EXP2FIX: CompileTime[int] = 0):
    v8 = np.dtype[v8bfp16ebs8]
    T = M // 64
    assert NQT <= T
    t_ty, t4_ty = np.ndarray[(QB,), v8], np.ndarray[(N_ROWS * QB,), v8]

    qk = ExternalFunction("attn_qk_causal", object_file_name=f"attn_causal_mq64_nt64_oa_q{NQT}{'_x' if EXP2FIX else ''}.o", source_file=str(_SRC),
                          arg_types=[t_ty, t_ty, t_ty],
                          compile_flags=[f"-I{_INC}", f"-I{_SRC.parent.parent / 'attn'}", f"-DATT_MQ={MQ}",
                                         f"-DATT_NT={NT}", f"-DATT_NVALID_LAST={NT}", "-DATT_OA", f"-DATT_NQT={NQT}"] + (["-DATT_EXP2_FIX"] if EXP2FIX else []))
    bind = qk.object_file.bind
    begin, pv, end = bind("attn_begin_causal", [t_ty]), bind("attn_pv_causal", [t_ty, t_ty]), bind("attn_end_a", [t_ty])

    q_l3, q_l1, kv_f, o_l3, o_l1 = [], [], [], [], []
    for c in range(N_COLS):
        fq = ObjectFifo(t4_ty, name=f"q_{c}", depth=1)
        q_l3.append(fq)
        q_l1.append(fq.cons().split([r * QB for r in range(N_ROWS)], obj_types=[t_ty] * N_ROWS,
                                    names=[f"q_{c}_{r}" for r in range(N_ROWS)], depths=[1] * N_ROWS))
        kv_f.append(ObjectFifo(t_ty, name=f"kv_{c}", depth=2))
        fo = ObjectFifo(t4_ty, name=f"o_{c}", depth=1)
        o_l3.append(fo)
        o_l1.append(fo.prod().join([r * QB for r in range(N_ROWS)], obj_types=[t_ty] * N_ROWS,
                                   names=[f"o_{c}_{r}" for r in range(N_ROWS)], depths=[1] * N_ROWS))

    def core(q, kv, o, begin, qk, pv, end):
        for _ in range_(NQT):
            eo = o.acquire(1)
            eq = q.acquire(1)
            begin(eo)
            for _ in range_(NQT):
                ek = kv.acquire(1)
                qk(eq, ek, eo)
                kv.release(1)
                ev = kv.acquire(1)
                pv(ev, eo)
                kv.release(1)
            end(eo)
            q.release(1)
            o.release(1)

    data_size = 8 * MQ + MQ // 16 * 256 + 2 * MQ * NT + MQ * NT * 9 // 8 + 256 + 2 * MQ * D - 2 * MQ * NT
    workers = [Worker(core, [q_l1[c][r].cons(), kv_f[c].cons(), o_l1[c][r].prod(), begin, qk, pv, end],
                      stack_size=0xA00, data_size=data_size)
               for c in range(N_COLS) for r in range(N_ROWS)]

    q_all, kv_all, o_all = HQ * T * QB, HKV * T * 2 * QB, M * HID // 8
    kt = HID // 64

    def sequence(Qd, KVd, Od, q_prods, kv_prods, o_conses):
        tg_in = TaskGroup()  # the long fills live in their own group, freed only at the end
        for c in range(N_COLS):  # all Q tiles of the column's 4 heads; kv head c's NQT tiles, once per Q tile
            q_prods[c].fill(Qd, tap=TensorAccessPattern((q_all,), offset=4 * c * T * QB, sizes=[1, NQT, N_ROWS, QB],
                                                        strides=[0, QB, T * QB, 1]), group=tg_in, wait=False)
            kv_prods[c].fill(KVd, tap=TensorAccessPattern((kv_all,), offset=c * T * 2 * QB,
                                                          sizes=[NQT, 1, 2 * NQT, QB], strides=[0, 0, QB, 1]),
                             group=tg_in, wait=False)
        tg = TaskGroup()
        for i in range(NQT):
            for c in range(N_COLS):  # core j: q head 4c + j -> A tiles 2(4c + j), +1 of strip i // 2, half i % 2
                o_conses[c].drain(Od, tap=TensorAccessPattern((o_all,), offset=((i // 2) * kt + 8 * c) * AT + (i % 2) * (AT // 2),
                                                              sizes=[1, N_ROWS, 2, AT // 2], strides=[0, 2 * AT, AT, 1]),
                                  group=tg, wait=True)
            if i % 4 == 3 or i == NQT - 1:
                tg.finish()
                if i < NQT - 1:
                    tg = TaskGroup()
        tg_in.finish()

    rt = Runtime(sequence, [np.ndarray[(q_all,), v8], np.ndarray[(kv_all,), v8], np.ndarray[(o_all,), v8],
                            [f.prod() for f in q_l3], [f.prod() for f in kv_f], [f.cons() for f in o_l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Causal GQA attention, 32 cores")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("--m", type=int, default=1536)
    p.add_argument("--nqt", type=int, default=17, help="64-token tiles to compute (ceil(prompt length / 64))")
    p.add_argument("--exp2fix", type=int, default=0, help="1 = correct AIE2P's piecewise-linear exp2 (exp2_fix.h)")
    opts = p.parse_args()
    run_design_cli(attn_causal, opts, compile_kwargs=lambda o: dict(M=o.m, NQT=o.nqt, EXP2FIX=o.exp2fix),
                   device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
