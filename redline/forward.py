"""NPU forward of the Qwen-Image-2.1-Turbo DiT: host orchestration of the NPU kernels.

Weights stay resident on the NPU side (7.85 GB of iron tensors, loaded once); activations are iron tensors chained
from kernel to kernel with no host round trips (`run()` only syncs host-dirtied buffers).
Row layouts: cached steps M = 4096 (target tokens); step 1 (extract) M = 8704 = [text n | ref 4096 | pad | target
4096 (from row 4224) | pad].
Per block, cached step:
  A1 = ln_mod(x, 1+s1)            -> qkv GEMM -> qkv_post -> Q, KV_target
  O  = attn(Q, KV_target, KV_prefix[block])  (O written in the out-proj A layout)
  x  = resid(x, out GEMM(O), tanh g1); A2 = ln_mod(x, 1+s2) -> up GEMM with fused SwiGLU -> down GEMM
  x  = resid(x, down, tanh g2)
Per step: img_in GEMM -> blocks -> ln_mod(x, 1+norm_out scale) -> proj_out GEMM (target rows) -> host scheduler.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.utils import DefaultNPURuntime

from npu_layouts import V9
from run_prebuilt import load

from paths import CACHE, NPU as ART, WEIGHTS as WDIR, need_build
N_BLOCKS, HID, FF = 32, 4096, 12288
PREFIX_ROWS_PAD, TARGET_ROW0_S1, M_S1 = 4224, 4224, 8704

# GEMM artifacts per (op, M); all config3 bfp16 (A: a_layout(., 128, 64), B: b_layout(W.T, 64, 128), C: ATB order)
GEMMS = {
    # DIV=2 config3 variants: ~2% faster, same buffers and layouts
    ("qkv", 4096): "atb3d2-qkv", ("qkv", M_S1): "atb3d2-s1qkv",
    ("o", 4096): "atb3d2-4096", ("o", M_S1): "atb3d2-s1o",
    ("down", 4096): "atb3d2-down", ("down", M_S1): "atb3d2-s1down",
    # up GEMM with SwiGLU fused into its flush: writes the down GEMM's A operand (M x 12288)
    ("up_swiglu", 4096): "atb3sw2-m4096", ("up_swiglu", M_S1): "atb3sw2-m8704",
    ("img_in", 4096): "q-atb3k-imgin", ("img_in", M_S1): "q-atb3k-s1imgin",
    ("proj_out", 4096): "q-atb3k-projout",
}
GEMM_KN = {"qkv": (HID, 3 * HID), "o": (HID, HID), "up": (HID, 2 * FF), "down": (FF, HID), "img_in": (64, HID),
           "proj_out": (HID, 1024)}


# The model's distilled schedule (model_index.json sample_sigmas), used unshifted; the transformer gets bf16(sigma).
SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def model_t(sigma):
    """The timestep the transformer sees: the pipeline casts t = sigma*1000 to bf16 (steps of 4 between 512 and
    1024), then divides by 1000 in bf16. sigma 0.978453 -> 980 -> 0.98046875 (golden), not bf16(sigma) = 0.9765625."""
    from ml_dtypes import bfloat16
    return float((np.float32(sigma * 1000).astype(bfloat16).astype(np.float32) / 1000).astype(bfloat16))


def euler(latents, v, i, sigmas=SIGMAS):
    """Flow-matching Euler step i: matches every golden step (corr >= 0.999997)."""
    nxt = sigmas[i + 1] if i + 1 < len(sigmas) else 0.0
    return latents + (nxt - sigmas[i]) * v


def atb_index(r, c, N):
    """Stream position of element (r, c) in config3's C order (c_layout_atb with 128x128 tiles, rows=4, cols=8)."""
    rg, row, sbr, br, rib = r // 512, (r // 128) % 4, (r // 16) % 8, (r // 8) % 2, r % 8
    cg, col, sbc, bc, cib = c // 1024, (c // 128) % 8, (c // 16) % 8, (c // 8) % 2, c % 8
    i = (rg * (N // 1024) + cg) * 8 + col
    for d, v in ((4, row), (8, sbr), (8, sbc), (2, br), (2, bc), (8, rib), (8, cib)):
        i = i * d + v
    return i


def a_index(r, c, K):
    """Stream position of element (r, c) in the GEMM A layout (a_layout with 128x64 tiles)."""
    i = (r // 128) * (K // 64) + c // 64
    for d, v in ((8, (r // 16) % 8), (8, (c // 8) % 8), (2, (r // 8) % 2), (8, r % 8), (8, c % 8)):
        i = i * d + v
    return i


def v9(nvals):
    """An NPU tensor holding nvals bfp16 values (nvals / 8 blocks of 9 bytes)."""
    return iron.tensor(np.zeros(nvals // 8, dtype=V9), dtype=V9, device="npu")


class NPU:
    """Loaded kernels and per-op timing (NPU time and wall time, summed per op name)."""

    def __init__(self):
        self.rt = DefaultNPURuntime
        self.t_npu, self.t_wall, self.calls = defaultdict(float), defaultdict(float), defaultdict(int)
        self.elf, self.ctxs, self.pending, self.pending_ctx = {}, [], [], None
        self.elf_specs, self.nv = [], {}   # (path, route) per stage; runtime key counts patched in (attn_nvrt)

    def set_nv(self, nv):
        """Runtime valid-key counts {marker id: valid keys of the last tile} for the NVRT attention kernels: when they
        change (a new prompt length), reload the stage ELFs with the words patched (VigenFlow-style; no rebuild)."""
        if nv == self.nv:
            return
        self.sync()
        self.nv = dict(nv)
        specs, self.elf_specs, self.elf, self.ctxs = self.elf_specs, [], {}, []
        for path, route in specs:
            self.use_elf(path, route)

    def use_elf(self, elf_path, route):
        """Route artifacts to one merged full ELF (one hardware context, no context
        switch between its kernels): route = {artifact name: kernel name in the ELF}. Those runs start without waiting
        (in order within the context); sync() waits for them."""
        from elf_probe import elf_kernels
        self.elf_specs.append((elf_path, route))
        if self.nv:
            from attn_nvrt import patch
            data, tag = Path(elf_path).read_bytes(), []
            for mid, nv in sorted(self.nv.items()):
                data, n = patch(data, mid, nv)
                if n:
                    assert n == 32, f"{elf_path}: marker {mid} found {n} times"
                    tag.append(f"{mid}-{nv}")
            if tag:
                out = CACHE / "patched" / f"{Path(elf_path).stem}.nv{'_'.join(tag)}.elf"
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(data)
                elf_path = out
        handles, ctx = elf_kernels(str(elf_path), sorted(set(route.values())))
        self.ctxs.append(ctx)
        for art, name in route.items():
            self.elf[art] = (handles[name], len(self.ctxs) - 1)

    def sync(self):
        """Wait for every pending ELF run: call before any host read of a buffer they write."""
        for r, _args in self.pending:
            r.wait()
        self.pending, self.pending_ctx = [], None

    def run(self, op, art, args):
        if art in self.elf:
            import pyxrt
            h, c = self.elf[art]
            if self.pending_ctx not in (None, c):    # order is only guaranteed within one context
                self.sync()
            t = time.perf_counter()
            for a in args:
                a.to("npu")
            r = pyxrt.run(h.kernel)
            for i, a in enumerate(args):
                r.set_arg(i, a.buffer_object())
            r.start()
            # keep the argument tensors alive until the run is waited for: a temporary freed by Python while its
            # run is pending leaves the NPU reading freed memory (found: step 1's embed came out all NaN)
            self.pending.append((r, list(args)))
            self.pending_ctx = c
            if not getattr(self, "async_elf", True):
                self.sync()
            self.t_wall[op] += time.perf_counter() - t
            self.calls[op] += 1
            return
        self.sync()
        t = time.perf_counter()
        r = self.rt.run(load(ART / art), args)
        self.t_wall[op] += time.perf_counter() - t
        self.t_npu[op] += r.npu_time / 1e9
        self.calls[op] += 1

    def report(self):
        tot_w, tot_n = sum(self.t_wall.values()), sum(self.t_npu.values())
        for op in sorted(self.t_wall, key=lambda o: -self.t_wall[o]):
            print(f"  {op:12s} {self.calls[op]:5d} calls  npu {self.t_npu[op] * 1e3:9.1f} ms  "
                  f"wall {self.t_wall[op] * 1e3:9.1f} ms")
        print(f"  {'total':12s} {sum(self.calls.values()):5d} calls  npu {tot_n * 1e3:9.1f} ms  wall {tot_w * 1e3:9.1f} ms")


class Weights:
    """Packed weights, resident: per block qkv / o / up / down B operands (V9 tensors) + norm_q / norm_k."""

    def __init__(self, blocks=range(N_BLOCKS)):
        self.manifest = json.loads((WDIR / "manifest.json").read_text())
        t = time.perf_counter()
        self.b = {}
        for i in blocks:
            for m in ("qkv", "o", "up", "down"):
                raw = np.fromfile(WDIR / f"b{i:02d}_{m}.bin", np.uint8)
                self.b[i, m] = iron.tensor(raw.view(V9), dtype=V9, device="npu")
            self.b[i, "norm_q"] = np.load(WDIR / f"b{i:02d}_norm_q.npy")
            self.b[i, "norm_k"] = np.load(WDIR / f"b{i:02d}_norm_k.npy")
        for m in ("img_in", "proj_out"):
            self.b[m] = iron.tensor(np.fromfile(WDIR / f"{m}.bin", np.uint8).view(V9), dtype=V9, device="npu")
        self.host = {p.stem: np.load(p) for p in (WDIR / "host").glob("*.npy")}
        print(f"weights: {len(list(blocks))} blocks resident in {time.perf_counter() - t:.1f} s")


def gemm(npu, w, op, M, a, c, block=None):
    """C (ATB order, bfp16) = A (a_layout, bfp16) @ W for op at M rows; a and c are V9 tensors."""
    b = w.b[block, op] if block is not None else w.b[op]
    npu.run(op, GEMMS[op, M], [a, b, c])


def bf16_tensor(a):
    from ml_dtypes import bfloat16
    return iron.tensor(np.asarray(a, np.float32).astype(bfloat16).reshape(-1), dtype=bfloat16, device="npu")


LN_ART = {4096: "glue-ln-m4096-s0", M_S1: "glue-ln-m8704-s33"}
QKV_ART = {4096: "glue-qkv32-m4096", M_S1: "glue-qkv32-m8704"}


class CachedStep:
    """Device buffers + call sequence for a cached step (M = 4096 target rows). The attention call is pluggable:
    attn(npu, q, kv_target, kv_prefix, o) (two-buffer design, O in the out-proj A layout)."""

    def __init__(self, w, cos, sin, attn):
        from run_ln import ln_params
        from run_qkv import H, qkv_tables
        M = self.M = 4096
        T = M // 64
        self.w, self.attn = w, attn
        bf = __import__("ml_dtypes").bfloat16
        self.x = [iron.zeros((M * HID,), dtype=bf, device="npu") for _ in range(2)]
        self.zero_x = iron.zeros((M * HID,), dtype=bf, device="npu")  # block 0's residual input: x = 0 + img_in
        self.zero_br, self.a, self.o, self.c = v9(M * HID), v9(M * HID), v9(M * HID), v9(M * HID)
        self.c_qkv, self.h = v9(M * 3 * HID), v9(M * FF)
        self.c_in, self.c_out = v9(M * HID), v9(M * 1024)
        self.q, self.kv = v9(H * T * 64 * 128), v9(H * T * 2 * 64 * 128)
        C = np.log2(np.e) / np.sqrt(128)
        blocks = sorted({k[0] for k in w.b if isinstance(k, tuple)})
        self.tabs = {i: bf16_tensor(qkv_tables(cos, sin, w.b[i, "norm_q"] * C, w.b[i, "norm_k"])) for i in blocks}
        self.ln_params = ln_params

    def set_t(self, mod, out_scale):
        """Per-step parameters from the shared modulation (2, 16384) and the norm_out scale (2, 4096)."""
        s1, g1, s2, g2 = (mod[:, i * HID:(i + 1) * HID] for i in range(4))
        lp = lambda g, sc: bf16_tensor(self.ln_params(g, sc))
        self.p_first = lp(np.ones((2, HID)), 1 + s1)            # block 0: x = 0 + 1 * img_in
        self.p_mlp = lp(np.tanh(g1), 1 + s2)                    # after attention, before the MLP
        self.p_attn = lp(np.tanh(g2), 1 + s1)                   # after the MLP, before the next attention
        self.p_out = lp(np.tanh(g2), 1 + out_scale)             # after the last MLP: norm_out

    def blocks(self, npu, kv_prefix, first, last):
        """Run blocks first..last; self.x[0] / self.a must hold block `first`'s x and LN1 output on entry
        (block 0: call embed() first). Returns the index of the x buffer holding the result."""
        M, w = self.M, self.w
        for i in range(first, last + 1):
            gemm(npu, w, "qkv", M, self.a, self.c_qkv, block=i)
            npu.run("qkv_post", QKV_ART[M], [self.c_qkv, self.tabs[i], self.q, self.kv])
            self.attn(npu, self.q, self.kv, kv_prefix[i], self.o)
            gemm(npu, w, "o", M, self.o, self.c, block=i)
            npu.run("ln_mod", LN_ART[M], [self.x[0], self.c, self.p_mlp, self.x[1], self.a])
            npu.run("up_swiglu", GEMMS["up_swiglu", M], [self.a, w.b[i, "up"], self.h])
            gemm(npu, w, "down", M, self.h, self.c, block=i)
            p = self.p_out if i == N_BLOCKS - 1 else self.p_attn
            npu.run("ln_mod", LN_ART[M], [self.x[1], self.c, p, self.x[0], self.a])
        return 0

    def embed(self, npu, latents_pack):
        """img_in on the target latents (A layout, V9 tensor) and block 0's LN1: x[0] = img_in, a = LN1."""
        gemm(npu, self.w, "img_in", self.M, latents_pack, self.c_in)
        npu.run("ln_mod", LN_ART[self.M], [self.zero_x, self.c_in, self.p_first, self.x[0], self.a])

    def project(self, npu, a=None):
        """proj_out on a (= norm_out output after the last block) -> v (4096, 64) on the host."""
        if a is not None:
            self.a = a
        from npu_layouts import c_unlayout_atb, from_bfp16
        gemm(npu, self.w, "proj_out", self.M, self.a, self.c_out)
        npu.sync()
        c = from_bfp16(np.frombuffer(self.c_out.numpy().tobytes(), np.uint8))
        return c_unlayout_atb(c, self.M, 1024, 128, 128)[:, :64]


# Attention (V6: lazy rescale tau = 8, pipelined PV; O in the out-proj A layout)
ATTN_TARGET = "attn32o-v6-s-h32-t8-pf-nk8256-nkv8224"      # cached steps: [Q, KV target, full step-1 KV (prefix), O]
ATTN_S1_TARGET = "attn32o-v6-s-h32-t8-tr-nk8256-nkv8224"   # step 1 target: tile ranges of the full Q / KV buffers
ATTN_SMALL = "attn32q-v6-t8-nk4160-pt136"                  # step 1 host rows: per-row key limits, one tile per head
TILE_V9 = 64 * 128 // 8                          # one 64-row Q/K/V tile in V9 blocks


class ExtractStep:
    """Step 1 (kv_cache_mode extract), M = 8704 rows: [prefix p | pad | target from row 4224 | pad]. The prefix is
    the joint sequence's first p tokens (text and the reference image, interleaved as the VLM ordered them).
    Attention: target queries (Q tiles 66..129) over [target | prefix] on the NPU; prefix queries in tiles 0..63 over
    the prefix (mask count = keys the reference block sees) on the NPU; the remaining prefix rows (text tokens:
    causal; reference rows past tile 63) on the host with the exact rule (k <= q or same image block). Keeps every
    block's prefix K/V (tiles 0..64 per head) as the cache for the cached steps."""

    def __init__(self, w, cos, sin, image_ids, ref_art, M=M_S1, row0=TARGET_ROW0_S1, small_art=ATTN_SMALL):
        """M / row0: the step-1 layout (edits 8704 / 4224; text-to-image 4608 / 512, see ensure_t2i_step1)."""
        self.s1_art, self.small_art, self.row0 = ATTN_S1_TARGET, small_art, row0
        from run_ln import ln_params
        from run_qkv import H, qkv_tables
        M = self.M = M
        self.T = M // 64
        bf = __import__("ml_dtypes").bfloat16
        self.w, self.ref_art, self.ln_params = w, ref_art, ln_params
        self.ids = np.asarray(image_ids)                      # joint prefix: -1 text, else image block id
        self.p_len = len(self.ids)
        self.x = [iron.zeros((M * HID,), dtype=bf, device="npu") for _ in range(2)]
        self.zero_x = iron.zeros((M * HID,), dtype=bf, device="npu")
        self.zero_br, self.a, self.o, self.c = v9(M * HID), v9(M * HID), v9(M * HID), v9(M * HID)
        self.c_qkv, self.h = v9(M * 3 * HID), v9(M * FF)
        self.q = v9(H * self.T * 64 * 128)
        C = np.log2(np.e) / np.sqrt(128)
        blocks = sorted({k[0] for k in w.b if isinstance(k, tuple)})
        self.tabs = {i: bf16_tensor(qkv_tables(cos, sin, w.b[i, "norm_q"] * C, w.b[i, "norm_k"])) for i in blocks}
        tile_bytes = 128 * HID // 8 * 9                       # one 128-row M-tile of the A layout
        self.o_ref = self.o.subview(0, (32 * 128 * HID // 8,), V9)          # rows 0..4095
        self.o_tgt = self.o.subview(row0 // 128 * tile_bytes, (32 * 128 * HID // 8,), V9)  # the 4096 target rows
        nq = 64 * 32 * 2
        # prefix rows the two big attention calls don't get right: text rows in tiles 0..63 (causal), and every row
        # from 4096 on. They go through the small-query mode: one 64-row tile per head, row j sees keys <= lim[j];
        # one call per group of 64 rows (edits: one group; text-to-image: ceil(text tokens / 64))
        self.host_rows = np.array([r for r in range(self.p_len) if r >= nq or self.ids[r] < 0])
        from run_attn import pack_lim
        self.groups = []
        for g in range(0, len(self.host_rows), 64):
            rows = self.host_rows[g:g + 64]
            lim = np.zeros(64, np.int32)
            for j, r in enumerate(rows):   # k <= q, or anywhere in q's own image block (contiguous)
                lim[j] = max(r, np.flatnonzero(self.ids == self.ids[r]).max()) if self.ids[r] >= 0 else r
            self.groups.append((rows, iron.tensor(pack_lim(lim, 64, 64).view(V9), dtype=V9, device="npu")))
        self.q_small, self.o_small = v9(H * 64 * 128), v9(128 * HID)
        tb = 64 * 128 // 8 * 9                                  # one Q tile
        self.q_tiles = {(h, t): self.q.subview((h * self.T + t) * tb, (64 * 128 // 8,), V9)
                        for h in range(H) for t in np.unique(self.host_rows // 64)}
        self.o_tiles = {mt: self.o.subview(mt * tile_bytes, (128 * HID // 8,), V9)
                        for mt in np.unique(self.host_rows // 128)}
        # one full K/V buffer per block, resident: step 1 reads tile ranges of it, the cached steps read its prefix
        # (tiles 0..64) through the PT = 136 attention build; no host copies (2.56 GB for 32 blocks)
        self.kv_blk = {i: v9(H * self.T * 2 * 64 * 128) for i in blocks}
        self.kvp = self.kv_blk

    def _small_q(self, rows):
        """Q of host rows (<= 64) into the small-query buffer: decode only their tiles (9 KB subview syncs)."""
        from npu_layouts import a_layout, a_unlayout, from_bfp16, to_bfp16
        from run_qkv import H
        q = np.zeros((H, 64, 128), np.float32)
        for h in range(H):
            dec = {t: a_unlayout(from_bfp16(np.frombuffer(self.q_tiles[h, t].numpy().tobytes(), np.uint8)),
                                 64, 128, 64, 128) for t in np.unique(rows // 64)}
            q[h, :len(rows)] = [dec[r // 64][r % 64] for r in rows]
        # re-encoding values that are already bfp16 with the same 8-value blocks is exact
        with self.q_small.mutate() as arr:
            arr[:] = np.concatenate([to_bfp16(a_layout(q[h], 64, 128)) for h in range(H)]).view(V9)

    def _merge_small(self, rows):
        """Copy the small-query output rows (A layout, rows 0..n-1) into O's host rows, whole bfp16 blocks."""
        src = np.frombuffer(self.o_small.numpy().tobytes(), np.uint8).reshape(-1, 9)
        cols = np.arange(0, HID, 8)
        for mt, sub in self.o_tiles.items():
            sel = [(j, r) for j, r in enumerate(rows) if r // 128 == mt]
            if not sel:
                continue
            with sub.mutate() as arr:
                dst = arr.view(np.uint8).reshape(-1, 9)
                for j, r in sel:
                    dst[a_index(np.full(512, r % 128), cols, HID) // 8] = src[a_index(np.full(512, j), cols, HID) // 8]
            sub.to("npu")

    set_t = CachedStep.set_t

    def blocks(self, npu, first, last):
        from run_qkv import H
        M, w, T = self.M, self.w, self.T
        for i in range(first, last + 1):
            kv = self.kv_blk[i]
            gemm(npu, w, "qkv", M, self.a, self.c_qkv, block=i)
            npu.run("qkv_post", QKV_ART[M], [self.c_qkv, self.tabs[i], self.q, kv])
            npu.run("attn", self.s1_art, [self.q, kv, kv, self.o_tgt])   # tile ranges: Q/K/V 66.., prefix 0..64
            if self.ref_art:                            # text-to-image has no reference rows
                npu.run("attn_ref", self.ref_art, [self.q, kv, self.o_ref])
            for rows, lim_t in self.groups:
                npu.sync()                              # host reads Q tiles written by qkv_post
                t = time.perf_counter()
                self._small_q(rows)
                npu.t_wall["host_rows"] += time.perf_counter() - t
                npu.run("attn_small", self.small_art, [self.q_small, lim_t, kv, self.o_small])
                npu.sync()                              # host reads o_small
                t = time.perf_counter()
                self._merge_small(rows)
                npu.t_wall["host_rows"] += time.perf_counter() - t
                npu.calls["host_rows"] += 1
            gemm(npu, w, "o", M, self.o, self.c, block=i)
            npu.run("ln_mod", LN_ART[M], [self.x[0], self.c, self.p_mlp, self.x[1], self.a])
            npu.run("up_swiglu", GEMMS["up_swiglu", M], [self.a, w.b[i, "up"], self.h])
            gemm(npu, w, "down", M, self.h, self.c, block=i)
            p = self.p_out if i == N_BLOCKS - 1 else self.p_attn
            npu.run("ln_mod", LN_ART[M], [self.x[1], self.c, p, self.x[0], self.a])

    def target_a(self):
        """The A-layout rows of the target (32 M-tiles from row0) as a view: proj_out's input at M = 4096."""
        tile_bytes = 128 * HID // 8 * 9
        return self.a.subview(self.row0 // 128 * tile_bytes, (32 * 128 * HID // 8,), V9)


def step1_embed(ex, npu, w, txt_rows, txt_x, ref_rows, ref_lat, noise):
    """Step 1's x and LN1 on the device: X (bf16) holds the text rows (host txt_in), the img_in GEMM (M = 8704) maps
    the reference latents (at their joint positions) and the noise (target rows 4224..8319); one ln_fused adds them
    (g = 1) and applies LN1 with p_first. Rows not written stay zero (padding)."""
    from npu_layouts import a_layout, to_bfp16
    from run_ln import ln_pack_x
    M = ex.M
    lat = np.zeros((M, 64), np.float32)
    lat[ref_rows] = ref_lat
    lat[ex.row0:ex.row0 + 4096] = noise
    lat_t = iron.tensor(to_bfp16(a_layout(lat, 128, 64)).view(V9), dtype=V9, device="npu")
    xt = np.zeros((M, HID), np.float32)
    xt[txt_rows] = txt_x
    x_t = iron.tensor(ln_pack_x(xt).reshape(-1), dtype=__import__("ml_dtypes").bfloat16, device="npu")
    c_in = v9(M * HID)
    gemm(npu, w, "img_in", M, lat_t, c_in)
    npu.run("ln_mod", LN_ART[M], [x_t, c_in, ex.p_first, ex.x[0], ex.a])




def ensure_attn(base, nkv):
    """The attention artifact `base` rebuilt for another valid-key count (prompt length): same design args with
    --nkv changed; built on demand (~30 s) and cached. The mask count is compile-time here."""
    if nkv == 8224:
        return base
    import json, subprocess, sys
    name = base.replace("nkv8224", f"nkv{nkv}")
    if not (ART / name / "final.xclbin").exists():
        need_build(name)
        m = json.loads((ART / base / "meta.json").read_text())
        da = list(m["design_args"]); da[da.index("--nkv") + 1] = str(nkv)
        meta = [f"{k}={m[k]}" for k in m if k.isupper() and k != "NKV"] + [f"NKV={nkv}"]
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "build" / "npu_build.py"), "--name", name,
                        "--design", m["design"], "--family", m["family"], "--meta", *meta, "--design-args", *da],
                       check=True, env={**__import__("os").environ, "NPU_ART_ROOT": str(ART)})
    return name


M_T2I, T2I_ROW0 = 4608, 512   # text-to-image step 1: [text <= 512 | target 4096 from row 512] (VigenFlow's M)


def _build(name, design, family, shape=None, meta=(), dargs=()):
    """Compile an artifact with build/npu_build.py unless it exists (compile-only; ~5-150 s)."""
    import subprocess, sys
    if (ART / name / "final.xclbin").exists():
        return name
    need_build(name)
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root / "build" / "npu_build.py"), "--name", name, "--design", str(root / design),
           "--family", family]
    if shape:
        cmd += ["--shape", *map(str, shape), "--tile", "128", "64", "128"]
    if meta:
        cmd += ["--meta", *meta]
    subprocess.run(cmd + ["--design-args", *dargs], check=True, env={**__import__("os").environ, "NPU_ART_ROOT": str(ART)})
    return name


def ensure_t2i_step1(n):
    """Text-to-image step 1 at M = 4608 (instead of the edit layout's 8704): GEMMs, ln, qkv_post, the small-query
    text attention, and the target attention over [target | n text tiles] for steps 1 and 2+ (prefix PT = 72).
    Registers the artifact names; returns (cached attention, step-1 attention), both NVRT."""
    M = M_T2I
    for op, (K, N) in (("qkv", (HID, 3 * HID)), ("o", (HID, HID)), ("down", (FF, HID))):
        GEMMS[op, M] = _build(f"atb3d2-t2{op}", "kernels/atb3/n32_div.py", "atb-bfp", (M, K, N), dargs=["--div", "2"])
    GEMMS["img_in", M] = _build("q-atb3k-t2imgin", "kernels/atb3/n32_core.py", "atb-bfp", (M, 64, HID))
    GEMMS["up_swiglu", M] = _build(f"atb3sw2-m{M}", "kernels/atb3/n32_swiglu.py", "atb-swiglu",
                                   meta=[f"M={M}", "K=4096", "N=24576"], dargs=["-M", str(M), "-K", "4096", "-N", "24576"])
    S = T2I_ROW0 // 128                                       # M-tiles before the target use the t=0 modulation
    LN_ART[M] = _build(f"glue-ln-m{M}-s{S}", "kernels/glue/ln_design.py", "glue-ln", meta=[f"M={M}", f"S={S}"],
                       dargs=["--m", str(M), "--s", str(S)])
    QKV_ART[M] = _build(f"glue-qkv32-m{M}", "kernels/glue/qkv_design.py", "qkv32", meta=[f"M={M}"],
                        dargs=["--m", str(M), "--out-depth", "2"])
    T = M // 64
    attn = "kernels/attn/attn_design.py"
    small = _build(f"attn32q-v6-t8-nk{T2I_ROW0}-pt{T}", attn, "attn32q",
                   meta=["MQ=64", f"NK={T2I_ROW0}", "NT=64", "H=32", "OA=1", "OHEADS=32", "H0=0", "V6=1", "TAU=8",
                         f"PT={T}", "P0=0"],
                   dargs=["--small", "1", "--nk", str(T2I_ROW0), "--nt", "64", "--v6", "1", "--tau", "8", "--pt", str(T),
                          "--p0", "0", "--oheads", "32"])
    nk = 4096 + 64 * n
    common = ["--cores", "32", "--nk", str(nk), "--nkv", str(nk), "--nt", "64", "--heads", "32", "--waves", "2",
              "--oa", "1", "--oheads", "32", "--kvsplit", "64", "--v6", "1", "--tau", "8"]
    meta = ["MQ=64", f"NK={nk}", f"NKV={nk}", "NT=64", "H=32", "W=2", "OA=1", "OHEADS=32", "H0=0", "KVSPLIT=64"]
    q0 = T2I_ROW0 // 64
    a_s1 = _build(f"attn32o-v6-s-h32-t8-tr{T}-nk{nk}-nvrt2", attn, "attn32o",
                  meta=meta + [f"PT={T}", "P0=0", "NVRT=2"],
                  dargs=common + ["--qt", str(T), "--q0", str(q0), "--kt", str(T), "--k0", str(q0), "--pt", str(T),
                                  "--p0", "0", "--nvrt", "2"])
    a_c = _build(f"attn32o-v6-s-h32-t8-pf{T}-nk{nk}-nvrt1", attn, "attn32o", meta=meta + [f"PT={T}", "P0=0", "NVRT=1"],
                 dargs=common + ["--pt", str(T), "--p0", "0", "--nvrt", "1"])
    return a_c, a_s1, small


def ensure_t2i_attn(kind, n):
    """Text-to-image attention: the target's 64 KV tiles + n text tiles (NK = 4096 + 64 n), runtime key count.
    kind 'tr' = step 1 (tile ranges of the full step-1 Q/KV buffers), 'pf' = cached steps. Built on demand (~50 s,
    once per n) from the edit path's NVRT artifacts with --nk changed."""
    import json, subprocess, sys
    base = {"tr": "attn32o-v6-s-h32-t8-tr-nk8256-nvrt2", "pf": "attn32o-v6-s-h32-t8-pf-nk8256-nvrt1"}[kind]
    nk = 4096 + 64 * n
    name = base.replace("nk8256", f"nk{nk}")
    if not (ART / name / "final.xclbin").exists():
        need_build(name)
        m = json.loads((ART / base / "meta.json").read_text())
        da = list(m["design_args"])
        da[da.index("--nk") + 1] = da[da.index("--nkv") + 1] = str(nk)
        meta = [f"{k}={m[k]}" for k in m if k.isupper() and k not in ("NK", "NKV")] + [f"NK={nk}", f"NKV={nk}"]
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "build" / "npu_build.py"),
                        "--name", name, "--design", m["design"], "--family", m["family"], "--meta", *meta,
                        "--design-args", *da], check=True, env={**__import__("os").environ, "NPU_ART_ROOT": str(ART)})
    return name


def nv_variant(art, nv):
    """An NVRT artifact run as its own xclbin with the key count patched into its instructions (cached per nv)."""
    import shutil
    from attn_nvrt import patch
    out = CACHE / "nv" / f"{art}--nv{nv}"
    if not (out / "insts.bin").exists():
        out.mkdir(parents=True, exist_ok=True)
        m = json.loads((ART / art / "meta.json").read_text())
        for f in ("final.xclbin", "meta.json"):
            shutil.copyfile(ART / art / f, out / f)
        (out / "insts.bin").write_bytes(patch((ART / art / "insts.bin").read_bytes(), m["NVRT"], nv, expect=32)[0])
    return str(out)


ELF_DIR = ART / "merged"
# Merged ELFs: one hardware context per DiT stage, kernels started without waiting. Attention takes its valid-key
# count at runtime (attn_design --nvrt, attn_nvrt.py), so one ELF serves every prompt length; dit_denoise patches the
# counts per prompt (NPU.set_nv). ATTN_RT: artifact -> (marker id, padded keys).
ATTN_RT = {"attn32o-v6-s-h32-t8-pf-nk8256-nvrt1": (1, 8256), "attn32o-v6-s-h32-t8-tr-nk8256-nvrt2": (2, 8256),
           "attn32o-v6-a-h32-t8-tr-nk4160-nvrt3": (3, 4160)}
ELF_STAGES_RT = {
    "dit_cached_v7.elf": {"atb3d2-qkv": "qkv", "glue-qkv32-m4096": "qkvp", "atb3d2-4096": "o", "glue-ln-m4096-s0": "ln",
                          "atb3sw2-m4096": "upsw", "atb3d2-down": "down", "q-atb3k-imgin": "imgin",
                          "q-atb3k-projout": "projout", "attn32o-v6-s-h32-t8-pf-nk8256-nvrt1": "attn"},
    "dit_step1_v7.elf": {"atb3d2-s1qkv": "qkv", "glue-qkv32-m8704": "qkvp", ATTN_SMALL: "asmall", "atb3d2-s1o": "o",
                         "glue-ln-m8704-s33": "ln", "atb3sw2-m8704": "upsw", "atb3d2-s1down": "down",
                         "q-atb3k-s1imgin": "imgin", "attn32o-v6-s-h32-t8-tr-nk8256-nvrt2": "attn",
                         "attn32o-v6-a-h32-t8-tr-nk4160-nvrt3": "aref"},
}


def use_dit_elfs(npu):
    """Route the DiT's kernels to the merged ELFs when both exist (separate xclbins otherwise). They are loaded by
    dit_denoise's set_nv, already patched for the prompt (no unpatched load first)."""
    npu.nvrt = all((ELF_DIR / f).exists() for f in ELF_STAGES_RT)
    if npu.nvrt:
        for f, route in ELF_STAGES_RT.items():
            npu.elf_specs.append((ELF_DIR / f, route))

def dit_denoise(npu, w, prompt_embeds, vlm_image_mask, ref_lat, noise, sigmas=SIGMAS, ref_art=None, log=print,
                reuse=()):
    """The whole DiT on the NPU: step 1 (extract) + cached steps with host Euler. prompt_embeds (n_vlm, 4096) from the
    text encoder, vlm_image_mask (n_vlm,) bool, ref_lat / noise (4096, 64). Returns the final latents (4096, 64).
    Text-to-image: no image rows in the mask and ref_lat None; the prefix is the text alone (its causal rows through
    the small-query mode, attention over [target | text tiles] with the t2i artifacts).
    reuse: step indices (>= 1) that skip the DiT and reuse the previous step's velocity (lossy step caching)."""
    from host_glue import joint_layout, modulation, norm_out_scale, rope, temb, txt_in
    from ml_dtypes import bfloat16 as bf
    from run_qkv import positions
    pad, shapes = joint_layout(vlm_image_mask)
    rot = rope(pad, shapes)
    P = len(pad) - 4096                                              # prefix = everything but the target
    t2i = len(shapes) == 1
    assert P <= 4160 and len(shapes) == (1 if t2i else 2), "at most one reference image, prefix <= 4160 tokens"
    ids = np.where(pad[:P], 0, -1)
    # text-to-image with <= 512 text tokens: the smaller step-1 layout (M 4608, target from row 512)
    M, row0 = (M_T2I, T2I_ROW0) if t2i and P <= T2I_ROW0 else (M_S1, TARGET_ROW0_S1)
    pos = positions(M, P)
    cos, sin = np.zeros((M, 64), np.float32), np.zeros((M, 64), np.float32)
    cos[pos >= 0], sin[pos >= 0] = rot[pos[pos >= 0], :, 0], rot[pos[pos >= 0], :, 1]
    small = ATTN_SMALL
    if t2i:
        from attn_nvrt import nv_last
        n = -(-P // 64)                                              # text KV tiles
        if M == M_T2I:
            a_c, a_s1, small = ensure_t2i_step1(n)
            a_c, a_s1 = (nv_variant(a, nv_last(4096 + P, 4096 + 64 * n)) for a in (a_c, a_s1))
        else:                                                        # > 512 text tokens: the edit layout
            a_c, a_s1 = (nv_variant(ensure_t2i_attn(k, n), nv_last(4096 + P, 4096 + 64 * n)) for k in ("pf", "tr"))
        a_ref, ref_lat = None, np.zeros((0, 64), np.float32)
        if getattr(npu, "nvrt", False):                              # load the stage ELFs (their attention unused)
            npu.set_nv(npu.nv or {1: 32, 2: 32, 3: 8})
    elif getattr(npu, "nvrt", False) and not ref_art:   # one artifact per kernel, counts patched in per prompt
        nkv_ref = int(np.flatnonzero(ids >= 0).max()) + 1
        from attn_nvrt import nv_last
        a_c, a_s1, a_ref = ATTN_RT
        npu.set_nv({ATTN_RT[a][0]: nv_last(n, ATTN_RT[a][1]) for a, n in ((a_c, 4096 + P), (a_s1, 4096 + P),
                                                                          (a_ref, nkv_ref))})
    else:
        nkv_ref = int(np.flatnonzero(ids >= 0).max()) + 1
        a_c, a_s1 = ensure_attn(ATTN_TARGET, 4096 + P), ensure_attn(ATTN_S1_TARGET, 4096 + P)
        a_ref = ref_art or f"attn32o-v6-a-h32-t8-tr-nk4160-nkv{nkv_ref}"
    ex = ExtractStep(w, cos, sin, ids, a_ref, M, row0, small)
    ex.s1_art = a_s1
    t = time.perf_counter()
    te = temb(w.host, model_t(sigmas[0]))
    ex.set_t(modulation(w.host, te), norm_out_scale(w.host, te))
    txt_rows = np.flatnonzero(~pad[:P])                              # joint text positions = layout rows
    vlm_text = np.flatnonzero(~np.asarray(vlm_image_mask, bool))
    step1_embed(ex, npu, w, txt_rows, txt_in(w.host, prompt_embeds[vlm_text]), np.flatnonzero(pad[:P]), ref_lat, noise)
    ex.blocks(npu, 0, N_BLOCKS - 1)
    cs = CachedStep(w, cos[row0:row0 + 4096], sin[row0:row0 + 4096], None)
    cs.attn = lambda n, q, kvt, kvp, o: n.run("attn", a_c, [q, kvt, kvp, o])
    v = cs.project(npu, a=ex.target_a())
    lat = euler(noise, v, 0, sigmas).astype(np.float32).astype(bf).astype(np.float32)
    log(f"step 1 (extract): {time.perf_counter() - t:.1f} s")
    for i in range(1, len(sigmas)):
        t = time.perf_counter()
        if i not in reuse:
            te = temb(w.host, model_t(sigmas[i]))
            cs.set_t(modulation(w.host, te), norm_out_scale(w.host, te))
            from npu_layouts import a_layout, to_bfp16
            cs.embed(npu, iron.tensor(to_bfp16(a_layout(lat, 128, 64)).view(V9), dtype=V9, device="npu"))
            cs.blocks(npu, ex.kvp, 0, N_BLOCKS - 1)
            v = cs.project(npu)
        lat = euler(lat, v, i, sigmas).astype(np.float32).astype(bf).astype(np.float32)
        log(f"step {i + 1}: {time.perf_counter() - t:.1f} s{' (reused v)' if i in reuse else ''}")
    return lat
