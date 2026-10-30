"""Qwen-Image flash attention on the NPU (IRON, npu2). Kernel: attn_bfp.cc.

Stage 1 design: one core, one head, one Q tile of MQ queries against NK keys. K and V tiles of NT keys share one
input fifo, interleaved K0 V0 K1 V1 ... (a core has two input DMA channels; Q uses the other). Host layouts (see
redline/npu_layouts.py): Q = a_layout(Q*c, MQ, 128) with c = log2(e)/sqrt(128); K tile = b_layout(K_t^T, 128, NT);
V tile = b_layout(V_t, NT, 128); O = one (MQ, 128) C tile.
Compile-only: python attn_design.py -d npu2 --xclbin-path X --insts-path Y --nk 512 [--mq 64 --nt 32]
"""

import argparse
import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.trace.events import CoreEvent
from ml_dtypes import bfloat16

_SRC = Path(__file__).resolve().parent / "attn_bfp.cc"
_INC = Path(os.environ.get("MLIR_AIE_SRC", "")) / "aie_kernels"
D = 128


def _kernels(MQ, NT, q_ty, kv_ty, o_ty, noop=False, nvalid_last=None, exp2fix=False, oa=False, abl=0, v6=0, tau=0,
             lim_ty=None, nv_rt=False):
    """attn_begin, attn_qk, attn_qk_last (keys >= nvalid_last of the last tile masked), attn_pv, attn_end, all from
    one object: they share the running max/sum statics. oa: -DATT_OA build, o_ty is the bfp16 O buffer (it also
    carries S, so attn_qk takes it), and attn_end is attn_end_a (O as the out-projection's A tiles)."""
    nv = NT if nvalid_last is None else nvalid_last
    qk = ExternalFunction(
        "attn_qk",
        object_file_name=(f"attn_bfp_mq{MQ}_nt{NT}_nv{nv}{'_noop' if noop else ''}"
                          f"{f'_x{int(exp2fix)}' if exp2fix else ''}{'_oa' if oa else ''}{f'_abl{abl}' if abl else ''}{f'_k{v6}t{tau}' if v6 else ''}{'_tr3' if os.environ.get('ATT_TRACE3') else ''}{'_lim' if lim_ty else ''}{'_rt' if nv_rt else ''}_v6.o"),
        source_file=str(_SRC),
        arg_types=[q_ty, kv_ty, o_ty] if oa else [q_ty, kv_ty],
        # Tried -mllvm -pipeliner-max-mii=200 (QK block loop then pipelines at II 75): measured slower overall,
        # qk 3504 -> 4790 cycles per 64-key tile (it takes loops the post-pipeliner scheduled better).
        compile_flags=[f"-I{_INC}", f"-I{_SRC.parent}", f"-DATT_MQ={MQ}", f"-DATT_NT={NT}", f"-DATT_NVALID_LAST={nv}"]
        + (["-DATT_NOOP"] if noop else []) + (["-DATT_OA"] if oa else []) + ([f"-DATT_ABL={abl}"] if abl else []) + ([f"-DATT_V6=1", f"-DATT_TAU={tau}"] + (["-DATT_V7=1"] if v6 >= 7 else []) if v6 else []) + (["-DATT_TRACE3=1"] if os.environ.get("ATT_TRACE3") else [])
        + (["-DATT_LIM"] if lim_ty else []) + (["-DATT_NVALID_RT"] if nv_rt else [])
        + ({1: ["-DATT_EXP2_FIX"], 2: ["-DATT_EXP2_FIX_A"], 3: ["-DATT_EXP2_FIX_P"]}.get(int(exp2fix), [])),
    )
    bind = qk.object_file.bind
    if lim_ty:  # small-query mode: begin, q_load, lim_load, attn_qk_lim (every tile), pv, end
        return (bind("attn_begin", [o_ty]), bind("attn_q_load", [q_ty]), bind("attn_lim_load", [q_ty]),
                bind("attn_qk_lim", [kv_ty, o_ty]), bind("attn_pv", [kv_ty, o_ty]), bind("attn_end_a", [o_ty]))
    ks = (bind("attn_begin", [o_ty]), qk, bind("attn_qk_last", [q_ty, kv_ty, o_ty] if oa else [q_ty, kv_ty]),
          bind("attn_pv", [kv_ty, o_ty]), bind("attn_end_a" if oa else "attn_end", [o_ty]))
    return ks + (bind("attn_set_nv", [np.int32]),) if nv_rt else ks


def _kv_loop(eq, eo, kv, qk, qk_last, pv, n_tiles, oa=False):
    """All K/V tiles of one wave; the last tile goes through attn_qk_last (key padding). oa: attn_qk also takes the
    O buffer (S lives in it)."""
    s = [eo] if oa else []
    if n_tiles > 1:
        for _ in range_(n_tiles - 1):
            ek = kv.acquire(1)
            qk(eq, ek, *s)
            kv.release(1)
            ev = kv.acquire(1)
            pv(ev, eo)
            kv.release(1)
    ek = kv.acquire(1)
    qk_last(eq, ek, *s)
    kv.release(1)
    ev = kv.acquire(1)
    pv(ev, eo)
    kv.release(1)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def attn1(Q: In, KV: In, O: Out, *, NK: CompileTime[int] = 512, MQ: CompileTime[int] = 64,
          NT: CompileTime[int] = 32, TRACE: CompileTime[int] = 0, NOOP: CompileTime[int] = 0,
          NKV: CompileTime[int] = 0, EXP2FIX: CompileTime[int] = 0, ABL: CompileTime[int] = 0,
          V6: CompileTime[int] = 0, TAU: CompileTime[int] = 0):
    nkv = NKV or NK  # valid keys; NK is the padded count (a multiple of NT)
    q_ty = np.ndarray[(MQ * D // 8,), np.dtype[v8bfp16ebs8]]
    kv_ty = np.ndarray[(NT * D // 8,), np.dtype[v8bfp16ebs8]]
    kvfull_ty = np.ndarray[(2 * NK * D // 8,), np.dtype[v8bfp16ebs8]]
    o_ty = np.ndarray[(MQ * D,), np.dtype[bfloat16]]

    of_q = ObjectFifo(q_ty, name="q", depth=1)
    of_kv = ObjectFifo(kv_ty, name="kv", depth=2)
    of_o = ObjectFifo(o_ty, name="o", depth=1)
    begin, qk, qk_last, pv, end = _kernels(MQ, NT, q_ty, kv_ty, o_ty, noop=bool(NOOP),
                                           nvalid_last=nkv - (NK - NT), exp2fix=EXP2FIX, abl=ABL, v6=V6, tau=TAU)

    def core(q, kv, o, begin, qk, qk_last, pv, end):
        eo = o.acquire(1)
        eq = q.acquire(1)
        begin(eo)
        _kv_loop(eq, eo, kv, qk, qk_last, pv, NK // NT)
        end(eo)
        q.release(1)
        o.release(1)

    # attn_bfp.cc statics: m_run, l_run (MQ floats each), ab_all (MQ/16 * 128 bf16), s_all (MQ*NT bf16),
    # p_all (MQ*NT bfp16 at 9/8 B per value), plus slack.
    data_size = 8 * MQ + MQ // 16 * 256 + 2 * MQ * NT + MQ * NT * 9 // 8 + 256
    if V6:  # mb_all (MQ/16 * 256 B) + al_all (4 * MQ B) + resc
        data_size += MQ // 16 * 256 + 4 * MQ + 64 + (4 * MQ + 64 if V6 >= 7 else 0)
    worker = Worker(core, [of_q.cons(), of_kv.cons(), of_o.prod(), begin, qk, qk_last, pv, end],
                    stack_size=0xA00, data_size=data_size)  # pipelined QK loop needs 2368 B

    def sequence(Qd, KVd, Od, qh, kvh, oh):
        qh.fill(Qd)
        kvh.fill(KVd)
        oh.drain(Od, wait=True)

    rt = Runtime(sequence, [q_ty, kvfull_ty, o_ty, of_q.prod(), of_kv.prod(), of_o.cons()])
    prog = Program(iron.get_current_device(), rt, workers=[worker])
    if TRACE > 0:  # event0/event1 bracket attn_qk and attn_pv: traced cycles per call
        prog.enable_trace(TRACE, workers=[worker], coretile_events=[
            CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.INSTR_VECTOR, CoreEvent.MEMORY_STALL,
            CoreEvent.STREAM_STALL, CoreEvent.LOCK_STALL, CoreEvent.ACTIVE, CoreEvent.DISABLED])
    return prog.resolve_program()


A_TILE = 128 * 64 // 8  # one out-projection A tile (128 x 64 bfp16), in v8bfp16ebs8 blocks
QL = 32  # attn32q: lim (64 int32 + pad = 288 B) as bfp16-block-sized slots after each head's Q tile


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def attn32(Q: In, KV: In, O: Out, *, NK: CompileTime[int] = 4096, H: CompileTime[int] = 1,
           W: CompileTime[int] = 2, MQ: CompileTime[int] = 64, NT: CompileTime[int] = 64,
           NOOP: CompileTime[int] = 0, NKV: CompileTime[int] = 0, EXP2FIX: CompileTime[int] = 0,
           OA: CompileTime[int] = 0, OHEADS: CompileTime[int] = 32, H0: CompileTime[int] = 0,
           V6: CompileTime[int] = 0, TAU: CompileTime[int] = 0, QT: CompileTime[int] = 0, Q0: CompileTime[int] = 0,
           KT: CompileTime[int] = 0, K0: CompileTime[int] = 0, NVRT: CompileTime[int] = 0):
    """Stage 2: all 32 cores. Per (head, wave): each core takes its own MQ-query Q tile (per column: shim ->
    memtile -> split to the column's 4 cores), the head's KV stream (K0 V0 K1 V1 ...) is broadcast to all 32
    cores, and O tiles are joined per column on the way out. One wave = 32 * MQ queries of one head.
    DDR layouts: Q is [head][wave][core][MQ x 128 tile] with core = col * 4 + row; KV is [head][stream].
    O: OA=0: [head][wave][core][MQ x 128 C tile] bf16. OA=1: the out-projection's A operand,
    to_bfp16(a_layout(O_all, 128, 64)) with O_all (W*32*MQ, OHEADS*128) and head h in columns (H0+h)*128."""
    return _attn32(NK, H, W, MQ, NT, NOOP, NKV, EXP2FIX, OA, OHEADS, H0, 0, V6, TAU, QT, Q0, KT, K0, NVRT=NVRT)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def attn32s(Q: In, KVT: In, KVP: In, O: Out, *, NK: CompileTime[int] = 8256, H: CompileTime[int] = 1,
            W: CompileTime[int] = 2, MQ: CompileTime[int] = 64, NT: CompileTime[int] = 64,
            NOOP: CompileTime[int] = 0, NKV: CompileTime[int] = 0, EXP2FIX: CompileTime[int] = 0,
            OA: CompileTime[int] = 1, OHEADS: CompileTime[int] = 32, H0: CompileTime[int] = 0,
            KVSPLIT: CompileTime[int] = 64, V6: CompileTime[int] = 0, TAU: CompileTime[int] = 0,
            QT: CompileTime[int] = 0, Q0: CompileTime[int] = 0, KT: CompileTime[int] = 0, K0: CompileTime[int] = 0,
            PT: CompileTime[int] = 0, P0: CompileTime[int] = 0, NVRT: CompileTime[int] = 0):
    """attn32 with the KV stream from two buffers, per head [KVT's KVSPLIT tiles | KVP's NK/NT - KVSPLIT tiles]
    (cached steps: the target's own K/V from qkv_post, then the prefix cache from step 1; padding, masked via NKV,
    only at the end of KVP). Both buffers are [head][K0 V0 K1 V1 ...] like KV."""
    return _attn32(NK, H, W, MQ, NT, NOOP, NKV, EXP2FIX, OA, OHEADS, H0, KVSPLIT, V6, TAU, QT, Q0, KT, K0, PT, P0,
                   NVRT=NVRT)


def _attn32(NK, H, W, MQ, NT, NOOP, NKV, EXP2FIX, OA, OHEADS, H0, KVSPLIT, V6=0, TAU=0, QT=0, Q0=0, KT=0, K0=0,
            PT=0, P0=0, NVRT=0):
    """QT/Q0, KT/K0, PT/P0 (0 = the compact defaults): read Q, the (target) KV and the prefix KV as tile ranges of
    larger [head][T][tile] / [head][T][K|V][tile] buffers, e.g. qkv_post's full output (step 1 at M=8704: target Q and
    KV = tiles 66..129 of T=136, prefix KV = tiles 0..64 of the same KV buffer, reference Q = tiles 0..63)."""
    n_cols, n_rows = 8, 4
    n_cores = n_cols * n_rows
    v8 = np.dtype[v8bfp16ebs8]
    q_ty = np.ndarray[(MQ * D // 8,), v8]
    q4_ty = np.ndarray[(n_rows * MQ * D // 8,), v8]
    kv_ty = np.ndarray[(NT * D // 8,), v8]
    nkt = NK // NT  # KV tiles per head
    QT = QT or W * n_cores                     # Q tiles per head in the Q buffer; this call reads Q0 .. Q0+W*32-1
    q_tile = MQ * D // 8
    q_all = H * QT * q_tile
    if OA:
        # -DATT_OA: each core's O buffer is its tile as bfp16 A-order pieces (attn_end_a; S lives in it during the
        # wave, O accumulates in a static). The drain scatters them into O_all = (W*32*MQ) rows x OHEADS*128 columns,
        # as a_layout(O_all, 128, 64) with kt = 2*OHEADS A tiles of 128x64 per 128-row strip.
        assert MQ == 64 and H0 + H <= OHEADS, "OA: two 64-query tiles per 128-row A strip"
        o_dt = v8
        o_core = MQ * D // 8
        kt = 2 * OHEADS
        o_all = W * n_cores * MQ * OHEADS * D // 8
    else:
        o_dt = np.dtype[bfloat16]
        o_core = MQ * D
        o_all = H * W * n_cores * MQ * D
    o_ty = np.ndarray[(o_core,), o_dt]
    o4_ty = np.ndarray[(n_rows * o_core,), o_dt]

    of_kv = ObjectFifo(kv_ty, name="kv", depth=2)
    q_l1, o_l1, q_l3l2, o_l2l3 = [], [], [], []
    for c in range(n_cols):
        fq = ObjectFifo(q4_ty, name=f"q_l3l2_{c}", depth=1)
        q_l3l2.append(fq)
        q_l1.append(fq.cons().split([r * MQ * D // 8 for r in range(n_rows)], obj_types=[q_ty] * n_rows,
                                    names=[f"q_{c}_{r}" for r in range(n_rows)], depths=[1] * n_rows))
        fo = ObjectFifo(o4_ty, name=f"o_l2l3_{c}", depth=1)
        o_l2l3.append(fo)
        o_l1.append(fo.prod().join([r * o_core for r in range(n_rows)], obj_types=[o_ty] * n_rows,
                                   names=[f"o_{c}_{r}" for r in range(n_rows)], depths=[1] * n_rows))
    nkv = NKV or NK  # valid keys; NK is the padded count (a multiple of NT)
    ks = _kernels(MQ, NT, q_ty, kv_ty, o_ty, noop=bool(NOOP), nvalid_last=nkv - (NK - NT), exp2fix=EXP2FIX,
                  oa=bool(OA), v6=V6, tau=TAU, nv_rt=bool(NVRT))
    # NVRT = marker id (1..255): the valid keys of the last tile come from an RTP word 0x7A4E<id><nv> that the
    # sequence writes into every core (VigenFlow-style); redline/attn_nvrt.py patches it per prompt length
    rtps = [Buffer(np.ndarray[(16,), np.dtype[np.int32]], name=f"nvrt_{i}", use_write_rtp=True)
            for i in range(n_cores)] if NVRT else [None] * n_cores
    nv_word = (0x7A4E << 16) | (NVRT << 8) | (nkv - (NK - NT))

    def core(q, kv, o, begin, qk, qk_last, pv, end, *rt):
        for _ in range_(H * W):
            eo = o.acquire(1)
            eq = q.acquire(1)
            if rt:  # read after the Q acquire: this call's sequence has written the RTP before any fill
                rt[1](rt[0][0])
            begin(eo)
            _kv_loop(eq, eo, kv, qk, qk_last, pv, nkt, oa=bool(OA))
            end(eo)
            q.release(1)
            o.release(1)

    data_size = 8 * MQ + MQ // 16 * 256 + 2 * MQ * NT + MQ * NT * 9 // 8 + 256
    if OA:  # s_all moves into the O buffer, the bf16 O accumulator (o_acc) becomes a static
        data_size += 2 * MQ * D - 2 * MQ * NT
    if V6:  # mb_all (MQ/16 * 256 B) + al_all (4 * MQ B) + resc
        data_size += MQ // 16 * 256 + 4 * MQ + 64 + (4 * MQ + 64 if V6 >= 7 else 0)
    begin, qk, qk_last, pv, end = ks[:5]
    workers = [Worker(core, [q_l1[c][r].cons(), of_kv.cons(), o_l1[c][r].prod(), begin, qk, qk_last, pv, end]
                      + ([rtps[c * n_rows + r], ks[5]] if NVRT else []), stack_size=0xA00, data_size=data_size)
               for c in range(n_cols) for r in range(n_rows)]

    def q_tap(h, w, c):  # column c's 4 consecutive Q tiles of head h, wave w
        off = (h * QT + Q0 + (w * n_cols + c) * n_rows) * q_tile
        return TensorAccessPattern((q_all,), off, [1, 1, 1, n_rows * q_tile], [0, 0, 0, 1])

    tile_v8 = 2 * NT * D // 8  # one K tile + one V tile
    kv_parts = [nkt] if not KVSPLIT else [KVSPLIT, nkt - KVSPLIT]  # tiles per head read from each KV buffer
    kv_geo = [(KT or kv_parts[0], K0)] + ([(PT or kv_parts[1], P0)] if KVSPLIT else [])  # (tiles per head, first)
    kv_alls = [H * t * tile_v8 for t, _ in kv_geo]

    def kv_tap(i, h):
        t, t0 = kv_geo[i]
        return TensorAccessPattern((kv_alls[i],), (h * t + t0) * tile_v8, [1, 1, 1, kv_parts[i] * tile_v8], [0, 0, 0, 1])
    o_taps = TensorAccessPattern.full((o_all,)).tile((n_rows * o_core,))

    def o_tap(h, w, c):
        if not OA:
            return o_taps[(h * W + w) * n_cols + c]
        # column c's 4 tiles = queries (w*32 + c*4 + r)*64: A strips w*16 + c*2 + r//2, bands (r%2)*4 of each.
        # Stream per core r = rp*2 + rr: [dims 0..63 | dims 64..127] -> A tiles kt0, kt0 + 1 of strip rp.
        off = ((w * n_cores // 2 + c * 2) * kt + 2 * (H0 + h)) * A_TILE
        return TensorAccessPattern((o_all,), off, [2, 2, 2, A_TILE // 2], [kt * A_TILE, A_TILE // 2, A_TILE, 1])

    def body(Qd, kv_ds, Od, q_prods, kv_prod, o_conses):
        for b in rtps if NVRT else []:
            b[0] = nv_word
        for h in range(H):
            for w in range(W):
                tg = TaskGroup()
                for c in range(n_cols):
                    q_prods[c].fill(Qd, tap=q_tap(h, w, c), group=tg, wait=False)
                for i, kvd in enumerate(kv_ds):
                    kv_prod.fill(kvd, tap=kv_tap(i, h), group=tg, wait=False)
                for c in range(n_cols):
                    o_conses[c].drain(Od, tap=o_tap(h, w, c), group=tg, wait=True)
                tg.finish()

    if KVSPLIT:
        def sequence(Qd, KVTd, KVPd, Od, q_prods, kv_prod, o_conses):
            body(Qd, [KVTd, KVPd], Od, q_prods, kv_prod, o_conses)
    else:
        def sequence(Qd, KVd, Od, q_prods, kv_prod, o_conses):
            body(Qd, [KVd], Od, q_prods, kv_prod, o_conses)

    rt = Runtime(sequence, [np.ndarray[(q_all,), v8]] + [np.ndarray[(n,), v8] for n in kv_alls]
                 + [np.ndarray[(o_all,), o_dt],
                    [f.prod() for f in q_l3l2], of_kv.prod(), [f.cons() for f in o_l2l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def attn32q(Q: In, LIM: In, KV: In, O: Out, *, NK: CompileTime[int] = 4160, MQ: CompileTime[int] = 64,
            NT: CompileTime[int] = 64, EXP2FIX: CompileTime[int] = 0, OHEADS: CompileTime[int] = 32,
            H0: CompileTime[int] = 0, V6: CompileTime[int] = 1, TAU: CompileTime[int] = 8, PT: CompileTime[int] = 0,
            P0: CompileTime[int] = 0):
    """Small-query mode (step 1's text and edge rows): one MQ-query tile per head, all 32 heads in one call, core
    (c, r) = head 4c + r. Query row q sees keys k <= LIM[q] only, applied on every tile (causal text rows, reference
    rows past the reference artifact's tiles). Q: [head][MQ x 128 tile] (the host gathers the rows as bfp16 blocks).
    LIM: one tile-sized buffer, MQ int32 at its start. KV: tiles P0..P0+NK/NT-1 of a [head][PT][K|V][tile] buffer
    (step 1: qkv_post's full KV, prefix = tiles 0..64). Each core has one input stream (its Q tile, the LIM block, then
    its head's K/V tiles), per column the 4 heads interleaved per object and split at the memtile (a separate Q split
    would need 8 memtile outputs per column; there are 6). O: a_layout of (128, OHEADS*128), one A strip: rows
    0..MQ-1 = the queries, head h at columns (H0+h)*128."""
    n_cols, n_rows = 8, 4
    H = n_cols * n_rows
    assert MQ == 64 and NT == 64 and H0 + H <= OHEADS
    v8 = np.dtype[v8bfp16ebs8]
    nkt = NK // NT
    PT = PT or nkt
    tile = MQ * D // 8  # Q tile = K tile = V tile (MQ == NT)
    t_ty, t4_ty = np.ndarray[(tile,), v8], np.ndarray[(n_rows * tile,), v8]
    kt = 2 * OHEADS
    q_all, kv_all, o_all = H * tile, H * PT * 2 * tile, kt * A_TILE

    kv_l1, o_l1, kv_l3l2, o_l2l3 = [], [], [], []
    for c in range(n_cols):
        fk = ObjectFifo(t4_ty, name=f"in_l3l2_{c}", depth=2)
        kv_l3l2.append(fk)
        kv_l1.append(fk.cons().split([r * tile for r in range(n_rows)], obj_types=[t_ty] * n_rows,
                                     names=[f"in_{c}_{r}" for r in range(n_rows)], depths=[2] * n_rows))
        fo = ObjectFifo(t4_ty, name=f"o_l2l3_{c}", depth=1)
        o_l2l3.append(fo)
        o_l1.append(fo.prod().join([r * tile for r in range(n_rows)], obj_types=[t_ty] * n_rows,
                                   names=[f"o_{c}_{r}" for r in range(n_rows)], depths=[1] * n_rows))
    begin, qload, limload, qk, pv, end = _kernels(MQ, NT, t_ty, t_ty, t_ty, exp2fix=EXP2FIX, oa=True, v6=V6, tau=TAU,
                                                  lim_ty=True)

    def core(kv, o, begin, qload, limload, qk, pv, end):
        eo = o.acquire(1)
        e = kv.acquire(1)
        qload(e)
        kv.release(1)
        e = kv.acquire(1)
        limload(e)
        kv.release(1)
        begin(eo)
        for _ in range_(nkt):
            ek = kv.acquire(1)
            qk(ek, eo)
            kv.release(1)
            ev = kv.acquire(1)
            pv(ev, eo)
            kv.release(1)
        end(eo)
        o.release(1)

    data_size = 8 * MQ + MQ // 16 * 256 + MQ * NT * 9 // 8 + 256 + 2 * MQ * D + 4 * MQ + 128 + tile * 9
    if V6:
        data_size += MQ // 16 * 256 + 4 * MQ + 64 + (4 * MQ + 64 if V6 >= 7 else 0)
    workers = [Worker(core, [kv_l1[c][r].cons(), o_l1[c][r].prod(), begin, qload, limload, qk, pv, end],
                      stack_size=0xA00, data_size=data_size)
               for c in range(n_cols) for r in range(n_rows)]

    def sequence(Qd, LIMd, KVd, Od, in_prods, o_conses):
        tg = TaskGroup()
        for c in range(n_cols):
            in_prods[c].fill(Qd, tap=TensorAccessPattern((q_all,), n_rows * c * tile, [1, 1, 1, n_rows * tile],
                                                         [0, 0, 0, 1]), group=tg, wait=False)
            in_prods[c].fill(LIMd, tap=TensorAccessPattern((tile,), 0, [1, 1, n_rows, tile], [0, 0, 0, 1]),
                             group=tg, wait=False)
            # per (tile, K|V): the column's 4 heads; the inner 1024 blocks as 4 x 256 so every wrap stays small
            in_prods[c].fill(KVd, tap=TensorAccessPattern((kv_all,), (n_rows * c * PT + P0) * 2 * tile,
                                                          [2 * nkt, n_rows, 4, tile // 4],
                                                          [tile, PT * 2 * tile, tile // 4, 1]),
                             group=tg, wait=False)
        for c in range(n_cols):
            o_conses[c].drain(Od, tap=TensorAccessPattern((o_all,), 2 * (H0 + n_rows * c) * A_TILE,
                                                          [1, n_rows, 2, A_TILE // 2], [0, 2 * A_TILE, A_TILE, 1]),
                              group=tg, wait=True)
        tg.finish()

    rt = Runtime(sequence, [np.ndarray[(q_all,), v8], np.ndarray[(tile,), v8], np.ndarray[(kv_all,), v8],
                            np.ndarray[(o_all,), v8], [f.prod() for f in kv_l3l2], [f.cons() for f in o_l2l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Qwen attention (stage 1, single core)")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("--nk", type=int, default=512)
    p.add_argument("--mq", type=int, default=64)
    p.add_argument("--nt", type=int, default=32)
    p.add_argument("--trace", type=int, default=0, help="trace buffer bytes (0 = no trace)")
    p.add_argument("--noop", type=int, default=0, help="1 = kernels do nothing (data-movement ablation)")
    p.add_argument("--nkv", type=int, default=0, help="valid keys (0 = all NK); the rest of the last tile is masked")
    p.add_argument("--exp2fix", type=int, default=0, help="correct AIE2P's piecewise-linear exp2: 1 = alpha and P, 2 = alpha only, 3 = P only")
    p.add_argument("--abl", type=int, default=0, help="attn1 timing ablation (wrong results): 1 S only, 2 softmax only, 3 PV MACs only")
    p.add_argument("--v6", type=int, default=0, help="1 = lazy rescale + config3-shaped PV loop (ATT_V6)")
    p.add_argument("--tau", type=int, default=0, help="--v6: a row max only moves when it rises by more than TAU (log2)")
    for a, h in (("qt", "Q tiles per head in the Q buffer (0 = W*32)"), ("q0", "first Q tile read"),
                 ("kt", "tiles per head in the (target) KV buffer (0 = compact)"), ("k0", "first (target) KV tile"),
                 ("pt", "tiles per head in the prefix KV buffer (0 = compact)"), ("p0", "first prefix KV tile")):
        p.add_argument(f"--{a}", type=int, default=0, help=f"attn32: {h}")
    p.add_argument("--nvrt", type=int, default=0, help="attn32: runtime valid-key count, RTP marker id 1..255 (0 = off)")
    p.add_argument("--small", type=int, default=0, help="1 = attn32q small-query mode (per-query key limits)")
    p.add_argument("--cores", type=int, default=1, choices=[1, 32], help="1 = attn1, 32 = attn32")
    p.add_argument("--heads", type=int, default=1, help="attn32: heads per call")
    p.add_argument("--waves", type=int, default=2, help="attn32: 32*MQ-query waves per head")
    p.add_argument("--oa", type=int, default=0, help="attn32: 1 = O as the out-projection's bfp16 A operand")
    p.add_argument("--oheads", type=int, default=32, help="attn32 --oa: heads (x128 columns) of the O_all matrix")
    p.add_argument("--h0", type=int, default=0, help="attn32 --oa: first head's column block in O_all")
    p.add_argument("--kvsplit", type=int, default=0,
                   help="attn32: KV tiles per head in a first buffer (target), the rest in a second (prefix cache)")
    opts = p.parse_args()
    if opts.small:
        design, kw = attn32q, lambda o: dict(NK=o.nk, MQ=o.mq, NT=o.nt, EXP2FIX=o.exp2fix, OHEADS=o.oheads,
                                             H0=o.h0, V6=o.v6, TAU=o.tau, PT=o.pt, P0=o.p0)
    elif opts.cores == 1:
        design, kw = attn1, lambda o: dict(NK=o.nk, MQ=o.mq, NT=o.nt, TRACE=o.trace, NOOP=o.noop, NKV=o.nkv,
                                           EXP2FIX=o.exp2fix, ABL=o.abl, V6=o.v6, TAU=o.tau)
    elif opts.kvsplit:
        design, kw = attn32s, lambda o: dict(NK=o.nk, H=o.heads, W=o.waves, MQ=o.mq, NT=o.nt, NOOP=o.noop,
                                             NKV=o.nkv, EXP2FIX=o.exp2fix, OA=o.oa, OHEADS=o.oheads, H0=o.h0,
                                             KVSPLIT=o.kvsplit, V6=o.v6, TAU=o.tau, QT=o.qt, Q0=o.q0, KT=o.kt,
                                             K0=o.k0, PT=o.pt, P0=o.p0, NVRT=o.nvrt)
    else:
        design, kw = attn32, lambda o: dict(NK=o.nk, H=o.heads, W=o.waves, MQ=o.mq, NT=o.nt, NOOP=o.noop,
                                            NKV=o.nkv, EXP2FIX=o.exp2fix, OA=o.oa, OHEADS=o.oheads, H0=o.h0, V6=o.v6,
                                            TAU=o.tau, QT=o.qt, Q0=o.q0, KT=o.kt, K0=o.k0, NVRT=o.nvrt)
    run_design_cli(design, opts, compile_kwargs=kw, device=lambda o: device_from_args(o, n_cols=None))


if __name__ == "__main__":
    main()
