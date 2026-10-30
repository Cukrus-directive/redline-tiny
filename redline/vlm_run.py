"""Qwen3-VL text encoder (language model) on the NPU: kernels, per-kernel checks, layer chain.
Usage (IRON env active): python vlm_run.py check-rms | check-qkv | check-attn | layers [N]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.utils import DefaultNPURuntime

from npu_layouts import V9, a_unlayout, c_layout_atb, c_unlayout_atb, from_bfp16, to_bfp16
from run_prebuilt import load

from paths import MODEL, NPU as ART, ROOT, WEIGHTS
WV = WEIGHTS / "vlm"
GN = ROOT / "golden" / "npy"   # developer checks only
M, D, L = 1536, 4096, 1070
HQ, HKV, HD = 32, 8, 128


def bf(a):
    return np.asarray(a, np.float32).astype(bfloat16).astype(np.float32)


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel())[0, 1])


def stats(name, got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    print(f"  {name:34s} corr {corr(got, ref):.6f}  max err {np.abs(got - ref).max() / np.abs(ref).max():.4f}")


def v9(nvals):
    return iron.tensor(np.zeros(nvals // 8, dtype=V9), dtype=V9, device="npu")


_handles = {}


def vload(art):
    """Like run_prebuilt.load, but survives eviction: Strix has 16 hardware contexts and a full encode uses ~22
    xclbins, so the runtime evicts least-recently-used contexts and invalidates their handles; reload those."""
    h = _handles.get(art)
    if h is None or not getattr(h, "_is_valid", True):
        from aie.utils.npukernel import NPUKernel
        h = _handles[art] = DefaultNPURuntime.load(NPUKernel(ART / art / "final.xclbin", ART / art / "insts.bin"))
    return h


def run(art, args):
    return DefaultNPURuntime.run(vload(art), args).npu_time / 1e6


# ---- RMSNorm + residual glue (kernels/vlm/rms_mod.cc via rms_design.py): x' = x + g*br, y = x' / rms(x') * w ----

def rms_params(w, g=1.0):
    """P buffer: rows [g | sc] x 2 (row 1 unused at S = 0)."""
    row = np.concatenate([np.full(D, g, np.float32), np.asarray(w, np.float32)])
    return np.concatenate([row, row]).astype(bfloat16)


def rms_ref(xq, brq, w):
    xp = bf(xq + brq)
    return xp, xp / np.sqrt((xp.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * w


def check_rms():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, D)) * 3).astype(np.float32)
    x[:, 7] += 40  # a massive-activation channel, as in real LMs
    br = rng.standard_normal((M, D)).astype(np.float32)
    w = (1 + 0.2 * rng.standard_normal(D)).astype(np.float32)
    xq = bf(x)
    brc = to_bfp16(c_layout_atb(br, 128, 128))
    brq = c_unlayout_atb(from_bfp16(brc), M, D, 128, 128)
    x_t = iron.tensor(c_layout_atb(xq, 128, 128).astype(bfloat16), dtype=bfloat16, device="npu")
    b_t = iron.tensor(brc.view(V9), dtype=V9, device="npu")
    p_t = iron.tensor(rms_params(w), dtype=bfloat16, device="npu")
    xo_t, ao_t = iron.zeros((M * D,), dtype=bfloat16, device="npu"), v9(M * D)
    ms = [run("vlm-rms-m1536", [x_t, b_t, p_t, xo_t, ao_t]) for _ in range(6)]
    xo = c_unlayout_atb(xo_t.numpy().astype(np.float32), M, D, 128, 128)
    y = a_unlayout(from_bfp16(np.frombuffer(ao_t.numpy().tobytes(), np.uint8)), M, D, 128, 64)
    xr, yr = rms_ref(xq, brq, w)
    print(f"rms_fused M={M}: {np.median(ms[1:]):.2f} ms")
    print(f"  x' bit-exact: {np.array_equal(xo, xr)}")
    stats("y vs numpy RMSNorm", y, yr)


def check_c16():
    """bf16-C GEMM (vlm-gemm-o16) and the bf16-branch RMS glue (vlm-rms16-m1536) on random data."""
    from npu_layouts import a_layout, b_layout
    rng = np.random.default_rng(3)
    a = rng.standard_normal((M, D)).astype(np.float32)
    b = (rng.standard_normal((D, D)) / 64).astype(np.float32)
    ap, bp = to_bfp16(a_layout(a, 128, 64)), to_bfp16(b_layout(b, 64, 128))
    aq, bq = a_unlayout(from_bfp16(ap), M, D, 128, 64), None
    from npu_layouts import b_unlayout
    bq = b_unlayout(from_bfp16(bp), D, D, 64, 128)
    c_t = iron.zeros((M * D,), dtype=bfloat16, device="npu")
    ms = run("vlm-gemm-o16", [iron.tensor(ap.view(V9), dtype=V9, device="npu"),
                              iron.tensor(bp.view(V9), dtype=V9, device="npu"), c_t])
    c = c_unlayout_atb(c_t.numpy().astype(np.float32), M, D, 128, 128)
    print(f"gemm-o16: {ms:.2f} ms")
    stats("bf16 C vs float64 (bfp16 inputs)", c, aq.astype(np.float64) @ bq)
    x = (rng.standard_normal((M, D)) * 3).astype(np.float32)
    w = (1 + 0.2 * rng.standard_normal(D)).astype(np.float32)
    xo_t, ao_t = iron.zeros((M * D,), dtype=bfloat16, device="npu"), v9(M * D)
    x_t = iron.tensor(c_layout_atb(bf(x), 128, 128).astype(bfloat16), dtype=bfloat16, device="npu")
    ms = run("vlm-rms16-m1536", [x_t, c_t, iron.tensor(rms_params(w), dtype=bfloat16, device="npu"), xo_t, ao_t])
    xo = c_unlayout_atb(xo_t.numpy().astype(np.float32), M, D, 128, 128)
    y = a_unlayout(from_bfp16(np.frombuffer(ao_t.numpy().tobytes(), np.uint8)), M, D, 128, 64)
    xr, yr = rms_ref(bf(x), c, w)
    print(f"rms16: {ms:.2f} ms;  x' bit-exact: {np.array_equal(xo, xr)}")
    stats("y vs numpy RMSNorm", y, yr)


# ---- QKV post, GQA (kernels/vlm/qkv_gqa_design.py): q/k RMSNorm * w, Q * log2e/sqrt(128), adjacent-pair RoPE ----

HALF9 = 64 * 128 // 8 * 9  # bytes of one attention tile


def qkv_run(cin_t, tabs):
    t_t = iron.tensor(tabs.reshape(-1), dtype=bfloat16, device="npu")
    T = M // 64
    q_t, kv_t = v9(HQ * T * 64 * 128), v9(HKV * T * 2 * 64 * 128)
    ms = run("vlm-qkv-m1536", [cin_t, t_t, q_t, kv_t])
    return q_t, kv_t, ms


def unpack_q(q_t, h):
    q = np.frombuffer(q_t.numpy().tobytes(), np.uint8).reshape(HQ, M // 64, HALF9)
    return a_unlayout(from_bfp16(q[h].ravel()), M, 128, 64, 128)


def unpack_kv(kv_t, g):
    from npu_layouts import b_unlayout
    kv = np.frombuffer(kv_t.numpy().tobytes(), np.uint8).reshape(HKV, M // 64, 2, HALF9)
    K = b_unlayout(from_bfp16(kv[g, :, 0].ravel()), 128, M, 128, 64).T
    V = b_unlayout(from_bfp16(kv[g, :, 1].ravel()), M, 128, 64, 128)
    return K, V


def check_qkv():
    from run_glue import rms_rope
    from run_qkv import qkv_tables
    rng = np.random.default_rng(1)
    C = np.log2(np.e) / np.sqrt(128)
    x = rng.standard_normal((M, (HQ + 2 * HKV) * 128)).astype(np.float32)
    cin = to_bfp16(c_layout_atb(x, 128, 128))
    xq = c_unlayout_atb(from_bfp16(cin), M, x.shape[1], 128, 128)
    ang = rng.uniform(-np.pi, np.pi, (M, 64))
    cos, sin = bf(np.cos(ang)), bf(np.sin(ang))
    wq, wk = bf(1 + 0.1 * rng.standard_normal(128)), bf(1 + 0.1 * rng.standard_normal(128))
    wqc = bf(wq * C)
    cin_t = iron.tensor(cin.view(V9), dtype=V9, device="npu")
    q_t, kv_t, ms = qkv_run(cin_t, qkv_tables(cos, sin, wqc, wk))
    ms = [qkv_run(cin_t, qkv_tables(cos, sin, wqc, wk))[2] for _ in range(5)]
    print(f"qkv_gqa M={M}: {np.median(ms):.2f} ms")
    wq_, wk_, wv_ = 1.0, 1.0, 1.0
    for h in range(HQ):
        wq_ = min(wq_, corr(unpack_q(q_t, h), rms_rope(xq[:, h * 128:(h + 1) * 128], wqc, cos, sin)))
    for g in range(HKV):
        K, V = unpack_kv(kv_t, g)
        wk_ = min(wk_, corr(K, rms_rope(xq[:, (HQ + g) * 128:(HQ + g + 1) * 128], wk, cos, sin)))
        wv_ = min(wv_, corr(V, xq[:, (HQ + HKV + g) * 128:(HQ + HKV + g + 1) * 128]))
    print(f"  worst head: Q {wq_:.6f} (32 heads)  K {wk_:.6f}  V {wv_:.6f} (8 kv heads)")


# ---- causal GQA attention (kernels/vlm/attn_causal_design.py): O straight into the o_proj A layout ----

NQT = -(-L // 64)  # 64-token tiles holding the prompt
ATTN = f"vlm-attn-m1536-t{NQT}-x"


def attn_ref(q, k, v, n):
    """float64 causal GQA attention over the first n positions. q (HQ, n, 128) unscaled, k/v (HKV, n, 128)."""
    out = np.zeros((n, HQ * HD))
    mask = np.triu(np.ones((n, n), bool), 1)
    for h in range(HQ):
        s = q[h].astype(np.float64) @ k[h // 4].T.astype(np.float64) / np.sqrt(HD)
        s[mask] = -np.inf
        w = np.exp(s - s.max(1, keepdims=True))
        out[:, h * HD:(h + 1) * HD] = (w / w.sum(1, keepdims=True)) @ v[h // 4]
    return out


def check_attn():
    from npu_layouts import a_layout
    from run_attn import pack_kv
    rng = np.random.default_rng(2)
    C = np.log2(np.e) / np.sqrt(HD)
    T = M // 64
    q = np.zeros((HQ, M, HD), np.float32)
    k, v = np.zeros((HKV, M, HD), np.float32), np.zeros((HKV, M, HD), np.float32)
    q[:, :L] = bf(rng.standard_normal((HQ, L, HD)) * 1.5)
    k[:, :L], v[:, :L] = bf(rng.standard_normal((HKV, L, HD))), bf(rng.standard_normal((HKV, L, HD)))
    qp = np.concatenate([to_bfp16(a_layout(q[h, i * 64:(i + 1) * 64] * C, 64, HD)) for h in range(HQ) for i in range(T)])
    kvp = np.concatenate([pack_kv(k[g], v[g], 64) for g in range(HKV)])
    q_t = iron.tensor(qp.view(V9), dtype=V9, device="npu")
    kv_t = iron.tensor(kvp.view(V9), dtype=V9, device="npu")
    o_t = v9(M * D)
    ms = [run(ATTN, [q_t, kv_t, o_t]) for _ in range(6)]
    o = a_unlayout(from_bfp16(np.frombuffer(o_t.numpy().tobytes(), np.uint8)), M, D, 128, 64)
    # the reference on the bfp16-rounded q/k/v the kernel saw
    qq = np.stack([a_unlayout(from_bfp16(qp[(h * T) * 9216:(h + 1) * T * 9216]), M, HD, 64, HD) for h in range(HQ)]) / C
    ref = attn_ref(qq[:, :L], k[:, :L], v[:, :L], L)
    print(f"attn_causal (NQT={NQT}, L={L}): {np.median(ms[1:]):.2f} ms")
    stats("O vs float64 causal GQA", o[:L], ref)
    worst = min(corr(o[:L, h * HD:(h + 1) * HD], ref[:, h * HD:(h + 1) * HD]) for h in range(HQ))
    print(f"  worst head corr {worst:.5f};  first rows {np.round(o[0, :4], 4)} vs {np.round(ref[0, :4], 4)}")


# ---- host prep: embeddings, mrope positions, RoPE tables ----

TE = MODEL / "text_encoder" / "model.safetensors"
THETA, N_LAYERS = 5e6, 36


def embed(ids):
    """Rows of embed_tokens (bf16, memory-mapped from the read-only checkpoint) -> float32 (len(ids), 4096)."""
    import struct
    with open(TE, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    e = hdr["model.language_model.embed_tokens.weight"]
    mm = np.memmap(TE, dtype=np.uint16, mode="r", offset=8 + n + e["data_offsets"][0], shape=tuple(e["shape"]))
    return mm[np.asarray(ids)].view(bfloat16).astype(np.float32)


def mrope_positions(ids, image_token_id, grid=(1, 64, 64), merge=2):
    """(3, len) t/h/w positions (transformers Qwen3VLModel.get_rope_index for one image)."""
    is_img = np.asarray(ids) == image_token_id
    pos, cur, i, n = np.zeros((3, len(ids)), np.int64), 0, 0, len(ids)
    while i < n:
        j = i
        while j < n and is_img[j] == is_img[i]:
            j += 1
        if not is_img[i]:
            pos[:, i:j] = cur + np.arange(j - i)
            cur += j - i
        else:
            gh, gw = grid[1] // merge, grid[2] // merge
            assert j - i == gh * gw
            hh, ww = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
            pos[0, i:j] = cur
            pos[1, i:j] = cur + hh.ravel()
            pos[2, i:j] = cur + ww.ravel()
            cur += max(grid[1], grid[2]) // merge
        i = j
    return pos


def mrope_cos_sin(pos):
    """Interleaved mrope [24, 20, 20]: pair j uses h for j % 3 == 1, w for j % 3 == 2 (j < 60), t otherwise."""
    inv = 1.0 / THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD)
    axis = np.zeros(64, int)
    axis[1:60:3], axis[2:60:3] = 1, 2
    ang = pos[axis].T * inv[None]  # (len, 64)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def prompt():
    meta = json.loads((GN / "vlm_meta.json").read_text())
    ids = np.load(GN / "vlm_input_ids.npy")
    pos = mrope_positions(ids, meta["image_token_id"])
    cos, sin = np.zeros((M, 64), np.float32), np.zeros((M, 64), np.float32)
    cos[:L], sin[:L] = mrope_cos_sin(pos)
    x = np.zeros((M, D), np.float32)
    x[:L] = embed(ids)
    return ids, meta, x, cos, sin


# ---- the language model, chained on the NPU ----

def balance_qk(wq, wk):
    """q_norm / k_norm weights (RoPE-permuted: pairs at 2j, 2j+1) -> wq * alpha, wk / alpha. q.k is unchanged, but q
    and k get similar magnitudes per dim, so their bfp16 blocks (8 dims, one exponent) lose far less: Qwen3's k_norm
    reaches 34.5 against q_norm's 3.5. alpha = sqrt(|wk| / |wq|) per dim, geometric mean over the pair (RoPE mixes
    the pair). Layer 0 simulation: attention corr 0.99868 -> 0.99978."""
    a = np.sqrt(np.maximum(np.abs(wk), 1e-2) / np.maximum(np.abs(wq), 1e-2))
    a = np.repeat(np.sqrt(a[0::2] * a[1::2]), 2)
    return wq * a, wk / a


class LM:
    """Resident weights + device buffers; layer(i) runs one decoder layer, buffers chained (no host round trips)."""

    # >0: the matrices stay in host RAM and each layer is copied into one of this many reusable NPU slots before it
    # runs (server.py). Windows caps an adapter's device-visible memory at half the RAM (~15.6 GB here), too small
    # for the DiT and all 36 LM layers at once. Runs are synchronous, so a slot is free again when the layer returns.
    STREAM = 0

    def __init__(self, cos, sin, layers=range(N_LAYERS), attn=None):
        hpf = WV / "hidden_perm.npy"  # packed weights' hidden-channel order (vlm_pack.hidden_perm)
        self.hp = np.load(hpf) if hpf.exists() else np.arange(D)
        self.layers = list(layers)
        self.w, self.tabs, self.p_in, self.p_post, self.qk = {}, {}, {}, {}, {}
        self.host = {}
        for i in self.layers:
            for m in ("qkv", "o", "up", "down"):
                f = WV / f"L{i:02d}_{m}.bin"
                if self.STREAM:  # read per request straight from the file (the file cache while RAM allows)
                    self.host[i, m] = f
                else:
                    self.w[i, m] = iron.tensor(np.fromfile(f, np.uint8).view(V9), dtype=V9, device="npu")
            nz = np.load(WV / f"L{i:02d}_norms.npz")
            self.qk[i] = balance_qk(nz["q_norm"], nz["k_norm"])
            self.p_in[i] = iron.tensor(rms_params(nz["w_in"]), dtype=bfloat16, device="npu")
            self.p_post[i] = iron.tensor(rms_params(nz["w_post"]), dtype=bfloat16, device="npu")
        if self.STREAM:
            l0 = self.layers[0]
            self.slots = [{m: iron.tensor(np.zeros(self.host[l0, m].stat().st_size // np.dtype(V9).itemsize, V9),
                                          dtype=V9, device="npu")
                           for m in ("qkv", "o", "up", "down")} for _ in range(self.STREAM)]
            self.t_stage = 0.0
        self.set_tables(cos, sin)
        T = M // 64
        z = lambda: iron.zeros((M * D,), dtype=bfloat16, device="npu")
        self.x = [z(), z(), z()]           # residual stream ping-pong (bf16 C stream)
        self.br_zero = v9(M * D)
        self.a1, self.a2, self.oa = v9(M * D), v9(M * D), v9(M * D)
        self.cqkv, self.co, self.cup, self.h, self.cd = v9(M * 6144), v9(M * D), v9(M * 24576), v9(M * 12288), v9(M * D)
        self.q, self.kv = v9(HQ * T * 64 * HD), v9(HKV * T * 2 * 64 * HD)
        self.t = {}
        self.attn = attn or ATTN
        self.fused_up = True
        self.upsw_art = "vlm-upsw-m1536"  # vlm-upswx-m1536: the accurate-SiLU copy
        self.c16_layers = set()  # layers whose o / down GEMMs write bf16 C, read by the bf16-branch RMS glue
        self.co16, self.cd16 = z(), z()
        self.br16 = False
        self.qkv16 = False  # QKV GEMM writes bf16 C, read by the bf16-input QKV post (kernels/vlm/qkv16_post.cc)
        self.cqkv16 = iron.zeros((M * 6144,), dtype=bfloat16, device="npu")

    def _run(self, op, art, args):
        self.t[op] = self.t.get(op, 0.0) + run(art, args)

    def _rms(self, x, br, br16, p, xo, a):
        self._run("rms", "vlm-rms16-m1536" if br16 else "vlm-rms-m1536", [x, br, p, xo, a])

    def set_tables(self, cos, sin):
        """qkv_post tables for the prompt's positions (cos/sin per row), per layer (its balanced q/k norm weights)."""
        from run_qkv import qkv_tables
        C = np.log2(np.e) / np.sqrt(HD)
        base = qkv_tables(cos, sin, np.zeros(HD), np.zeros(HD))   # the cos/sin layout, once per prompt
        for i in self.layers:
            wq, wk = self.qk[i]
            tab = base.copy()
            tab[:, -2 * HD:-HD], tab[:, -HD:] = bf(wq * C), wk   # per layer only the norm weights change
            self.tabs[i] = iron.tensor(tab.reshape(-1), dtype=bfloat16, device="npu")

    def load_x(self, x):
        """x in the model's channel order; stored in the packed weights' order."""
        self.x[0] = iron.tensor(c_layout_atb(bf(x)[:, self.hp], 128, 128).astype(bfloat16), dtype=bfloat16, device="npu")
        self.cur, self.br, self.br16 = 0, self.br_zero, False

    def unperm(self, xp):
        out = np.empty_like(xp)
        out[:, self.hp] = xp
        return out

    def peek(self, rows):
        """The residual stream after the last layer run (x2 + down), first `rows` rows, model channel order."""
        a = c_unlayout_atb(self.x[self.cur].numpy().astype(np.float32), M, D, 128, 128)[:rows]
        braw = self.br.numpy().astype(np.float32) if self.br16 else from_bfp16(np.frombuffer(self.br.numpy().tobytes(), np.uint8))
        b = c_unlayout_atb(braw, M, D, 128, 128)[:rows]
        return self.unperm(a.astype(np.float64) + b)

    def set_deepstack(self, feats, rows):
        """Deepstack: after layers 0..len(feats)-1, add feats[k] (len(rows) x 4096) at LM rows `rows` (the image
        tokens). Each becomes a bfp16 C-stream branch buffer, zero elsewhere, added by one extra rms call."""
        self.ds = []
        for f in feats:
            b = np.zeros((M, D), np.float32)
            b[rows] = f
            self.ds.append(iron.tensor(to_bfp16(c_layout_atb(b, 128, 128)).view(V9), dtype=V9, device="npu"))

    COPY_THREADS = 8   # STREAM mode: one layer's ~236 MB read in parallel chunks straight into the NPU slot

    def stage(self, i):
        """STREAM mode: copy layer i's matrices from host RAM into a free NPU slot."""
        from concurrent.futures import ThreadPoolExecutor
        if not hasattr(self, "_pool"):
            self._pool = ThreadPoolExecutor(self.COPY_THREADS)
        t = time.perf_counter()
        for m, buf in self.slots[i % self.STREAM].items():
            with buf.mutate() as arr:
                # readinto from the file: the OS copies from its cache in large blocks (a memmap copy page-faults
                # 4 KB at a time: ~3.4 GB/s for 8.5 GB per prompt)
                dst = memoryview(arr.view(np.uint8).reshape(-1))
                step = -(-len(dst) // self.COPY_THREADS)

                def chunk(o, f=self.host[i, m]):
                    with open(f, "rb", buffering=0) as fh:
                        fh.seek(o)
                        fh.readinto(dst[o:o + step])
                list(self._pool.map(chunk, range(0, len(dst), step)))
            buf.to("npu")
            self.w[i, m] = buf
        self.t_stage += time.perf_counter() - t

    def layer(self, i):
        """STREAM >= 2: layer i+1 is staged into the other slot on a thread while layer i runs on the NPU."""
        if self.STREAM:
            nxt = getattr(self, "_next", None)
            if nxt is not None and nxt[0] == i:
                nxt[1].join()
            else:
                self.stage(i)
            self._next = None
            if self.STREAM >= 2 and i + 1 in self.layers:
                import threading
                th = threading.Thread(target=self.stage, args=(i + 1,))
                th.start()
                self._next = (i + 1, th)
        self._layer(i)
        if i < len(getattr(self, "ds", [])):  # fold x2 + down into the stream, then add the deepstack branch
            x0, x1 = self.x[self.cur], self.x[(self.cur + 1) % 3]
            self._rms(x0, self.br, self.br16, self.p_in[i], x1, self.a1)
            self.cur, self.br, self.br16 = (self.cur + 1) % 3, self.ds[i], False

    def _layer(self, i):
        x0, x1 = self.x[self.cur], self.x[(self.cur + 1) % 3]
        self._rms(x0, self.br, self.br16, self.p_in[i], x1, self.a1)
        if self.qkv16:
            self._run("qkv", "vlm-gemm-qkv16", [self.a1, self.w[i, "qkv"], self.cqkv16])
            self._run("qkv_post", "vlm-qkv16-m1536", [self.cqkv16, self.tabs[i], self.q, self.kv])
        else:
            self._run("qkv", "vlm-gemm-qkv", [self.a1, self.w[i, "qkv"], self.cqkv])
            self._run("qkv_post", "vlm-qkv-m1536", [self.cqkv, self.tabs[i], self.q, self.kv])
        self._run("attn", self.attn, [self.q, self.kv, self.oa])
        c16 = i in self.c16_layers
        co = self.co16 if c16 else self.co
        self._run("o", "vlm-gemm-o16" if c16 else "vlm-gemm-o", [self.oa, self.w[i, "o"], co])
        x2 = self.x[(self.cur + 2) % 3]
        self._rms(x1, co, c16, self.p_post[i], x2, self.a2)
        if self.fused_up:  # up GEMM with SwiGLU in the flush: gate/proj never rounded to bfp16
            self._run("up+swiglu", self.upsw_art, [self.a2, self.w[i, "up"], self.h])
        else:
            self._run("up", "vlm-gemm-up", [self.a2, self.w[i, "up"], self.cup])
            self._run("swiglu", "vlm-swiglu-m1536", [self.cup, self.h])
        cd = self.cd16 if c16 else self.cd
        self._run("down", "vlm-gemm-down16" if c16 else "vlm-gemm-down", [self.h, self.w[i, "down"], cd])
        self.cur, self.br, self.br16 = (self.cur + 2) % 3, cd, c16   # x_out = x2 + down: folded into the next rms

    def x_out(self, last_i):
        """The residual stream after the last layer (x2 + down), via one more rms pass (its A output unused)."""
        x0, x1 = self.x[self.cur], self.x[(self.cur + 1) % 3]
        self._rms(x0, self.br, self.br16, self.p_in[last_i], x1, self.a1)
        return self.unperm(c_unlayout_atb(x1.numpy().astype(np.float32), M, D, 128, 128))


def layer_ref(x, i, cos, sin):
    """numpy float64 reference of decoder layer i (original, unpermuted weights; rotate-half RoPE) on rows < L."""
    from pack_weights import SafeTensors
    s = SafeTensors(TE)
    p = f"model.language_model.layers.{i}."
    g = lambda n: s.get(p + n).astype(np.float64)
    rms = lambda y, w: y / np.sqrt((y * y).mean(-1, keepdims=True) + 1e-6) * w
    x = x[:L].astype(np.float64)
    c2, s2 = np.concatenate([cos[:L], cos[:L]], 1), np.concatenate([sin[:L], sin[:L]], 1)
    rot = lambda y: np.concatenate([-y[..., 64:], y[..., :64]], -1)
    h = rms(x, g("input_layernorm.weight"))
    q = (h @ g("self_attn.q_proj.weight").T).reshape(L, HQ, HD)
    k = (h @ g("self_attn.k_proj.weight").T).reshape(L, HKV, HD)
    v = (h @ g("self_attn.v_proj.weight").T).reshape(L, HKV, HD)
    q, k = rms(q, g("self_attn.q_norm.weight")), rms(k, g("self_attn.k_norm.weight"))
    q = q * c2[:, None] + rot(q) * s2[:, None]
    k = k * c2[:, None] + rot(k) * s2[:, None]
    o = attn_ref(q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2), L)
    x = x + o @ g("self_attn.o_proj.weight").T
    h = rms(x, g("post_attention_layernorm.weight"))
    gt, up = h @ g("mlp.gate_proj.weight").T, h @ g("mlp.up_proj.weight").T
    return x + (gt / (1 + np.exp(-gt)) * up) @ g("mlp.down_proj.weight").T


def layers(n="36"):
    import time
    n = int(n)
    ids, meta, x, cos, sin = prompt()
    t = time.perf_counter()
    lm = LM(cos, sin, range(n))
    print(f"weights for {n} layers resident in {time.perf_counter() - t:.1f} s")
    lm.load_x(x)
    t = time.perf_counter()
    lm.layer(0)
    x1 = lm.x_out(0)
    print(f"layer 0 on the NPU: {(time.perf_counter() - t) * 1e3:.0f} ms wall (first calls)")
    stats("layer 0 vs numpy (rows < L)", x1[:L], layer_ref(bf(x), 0, cos, sin))
    if n == 1:
        return
    lm.load_x(x)
    lm.t = {}
    t = time.perf_counter()
    for i in range(n):
        lm.layer(i)
    xo = lm.x_out(n - 1)
    wall = time.perf_counter() - t
    print(f"{n} layers: wall {wall * 1e3:.0f} ms; npu " + ", ".join(f"{k} {v:.0f}" for k, v in lm.t.items())
          + f" ms (total {sum(lm.t.values()):.0f})")
    if n == N_LAYERS:
        from pack_weights import SafeTensors
        gold = SafeTensors((ROOT / "golden" / "a_text.safetensors")).get("prompt_embeds")[0]
        d = meta["drop_idx"]
        pre = np.flatnonzero(ids[d:] == meta["image_token_id"])[0]  # kept tokens before the image
        stats(f"prompt_embeds rows 0..{pre - 1} (pre-image) vs golden", xo[d:d + pre], gold[:pre])
        for r in range(pre):
            print(f"    row {r}: corr {corr(xo[d + r], gold[r]):.5f}")


if __name__ == "__main__":
    {"check-rms": check_rms, "check-qkv": check_qkv, "check-attn": check_attn, "layers": layers,
     "check-c16": check_c16}[sys.argv[1]](*sys.argv[2:])
