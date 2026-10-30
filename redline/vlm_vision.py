"""Qwen3-VL vision tower on the NPU: 27 blocks, the main merger and the 3 deepstack mergers.

Patches are stored patch-in-block-major: row = p*1024 + t, t = merged token (32x32 raster of 2x2 blocks), p = position
in the block (in_row*2 + in_col). The processor emits them block-major (row t*4 + p), so the host permutes once;
merging 4 patches then is A-tile re-indexing (vln merge/post modes). Biases ride in the GEMMs (ones tiles in the A
buffers, see vlm_pack_vision.py). Residual stream: bf16, config3 C order, rows of 2048 channels (1152 real).
Usage (IRON env active): python vlm_vision.py check
"""

import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.utils import DefaultNPURuntime

from npu_layouts import V9, a_layout, c_layout_atb, c_unlayout_atb, from_bfp16, to_bfp16
from run_prebuilt import load

from paths import MODEL, NPU as ART, ROOT, WEIGHTS
WV = WEIGHTS / "vlm" / "vision"
TE = MODEL / "text_encoder" / "model.safetensors"
GN = ROOT / "golden" / "npy"   # developer checks only
NP, NT_, HID, H, HD, DEPTH = 4096, 1024, 1152, 16, 72, 27
DEEPSTACK = (8, 16, 24)
C72 = np.log2(np.e) / np.sqrt(HD)


def bf(a):
    return np.asarray(a, np.float32).astype(bfloat16).astype(np.float32)


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel())[0, 1])


def v9(nvals):
    return iron.tensor(np.zeros(nvals // 8, dtype=V9), dtype=V9, device="npu")


def a_ones(rows, K, col):
    """An A buffer (a_layout 128x64, rows x K) holding only the ones column `col` (the bias row's input)."""
    z = np.zeros((rows, K), np.float32)
    z[:, col] = 1.0
    return iron.tensor(to_bfp16(a_layout(z, 128, 64)).view(V9), dtype=V9, device="npu")


# ---- host prep ----

def patch_order(grid=(1, 64, 64)):
    """For each stored row (p-major), the processor's row index; and the patch (row, col) in the 64x64 grid."""
    _, gh, gw = grid
    bw = gw // 2
    n = np.arange(gh * gw)
    p, t = n // (gh * gw // 4), n % (gh * gw // 4)
    src = t * 4 + p
    row = (t // bw) * 2 + p // 2
    col = (t % bw) * 2 + p % 2
    return src, row, col


def pos_embed(row, col, grid=(1, 64, 64)):
    """Bilinear (align_corners) resampling of the learned 48x48 table to the patch grid."""
    tab = np.load(WV / "pos_embed.npy").astype(np.float64)   # (2304, 1152)
    side = int(np.sqrt(tab.shape[0]))

    def taps(i, size):
        src = i * (side - 1) / max(size - 1, 1)
        f = np.floor(src)
        t = np.stack([f, f + 1], 1).clip(0, side - 1).astype(int)
        w = np.stack([1 - (src - f), src - f], 1)
        return t, w
    hr, hw_ = taps(row, grid[1])
    cr, cw = taps(col, grid[2])
    out = np.zeros((row.size, tab.shape[1]))
    for a in range(2):
        for b in range(2):
            out += (hw_[:, a] * cw[:, b])[:, None] * tab[hr[:, a] * side + cr[:, b]]
    return out


def vision_rope(row, col, theta=10000.0):
    """cos/sin (rows, 64) per qkv_post pair j (adjacent after the d' permutation = rotate-half pair (j, j+36)):
    pairs 0..17 use the row (h) position, 18..35 the column (w); 36..63 are padding."""
    inv = 1.0 / theta ** (np.arange(0, 36, 2, dtype=np.float64) / 36)   # 18
    ang = np.zeros((row.size, 64))
    ang[:, :18] = row[:, None] * inv
    ang[:, 18:36] = col[:, None] * inv
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def ln_params(w, b):
    """vln P buffer: w | b, zero-padded to whole 2048-value chunks."""
    n = w.size
    npc = (2 * n + 2047) // 2048
    p = np.zeros(npc * 2048, np.float32)
    p[:n], p[n:2 * n] = w, b
    return iron.tensor(p.astype(bfloat16), dtype=bfloat16, device="npu")


# ---- the NPU vision tower ----

class Vision:
    def __init__(self, blocks=range(DEPTH)):
        from run_qkv import qkv_tables
        self.blocks = list(blocks)
        ld = lambda f: iron.tensor(np.fromfile(WV / f, np.uint8).view(V9), dtype=V9, device="npu")
        self.w = {"patch": ld("patch.bin")}
        self.p = {}
        for i in self.blocks:
            for m in ("qkv", "proj", "fc1", "fc2"):
                self.w[i, m] = ld(f"v{i:02d}_{m}.bin")
            nz = np.load(WV / f"v{i:02d}_norms.npz")
            self.p[i, 1], self.p[i, 2] = ln_params(nz["n1w"], nz["n1b"]), ln_params(nz["n2w"], nz["n2b"])
        for k, name in enumerate(("ds0", "ds1", "ds2", "merger")):
            self.w[name, 1], self.w[name, 2] = ld(f"{name}_fc1.bin"), ld(f"{name}_fc2.bin")
            nz = np.load(WV / f"{name}_norm.npz")
            self.p[name] = ln_params(nz["w"], nz["b"])
        src, self.row, self.col = patch_order()
        self.src = src
        cos, sin = vision_rope(self.row, self.col)
        self.tabs = iron.tensor(qkv_tables(cos, sin, bf(np.full(128, C72)), np.ones(128, np.float32)).reshape(-1),
                                dtype=bfloat16, device="npu")
        z = lambda: iron.zeros((NP * 2048,), dtype=bfloat16, device="npu")
        self.x = [z(), z(), z()]
        self.xs = z()                                           # scratch XO for the merger LNs
        self.a1 = a_ones(NP, 1216, HID)                         # LN output -> qkv / fc1 (ones at col 1152)
        self.afc2 = a_ones(NP, 4416, 4352)                      # GELU output -> fc2
        self.am = a_ones(NT_, 4672, 4608)                       # merged LN output -> merger fc1
        self.am2 = a_ones(NT_, 4672, 4608)                      # merger GELU output -> merger fc2
        self.cqkv, self.cproj, self.cfc1, self.cfc2 = v9(NP * 12288), v9(NP * 2048), v9(NP * 5120), v9(NP * 2048)
        self.q, self.kv, self.o = v9(32 * 64 * 64 * 128), v9(32 * 64 * 2 * 64 * 128), v9(NP * 2048)
        self.cm1, self.cm2 = v9(NT_ * 5120), {n: v9(NT_ * 4096) for n in ("ds0", "ds1", "ds2", "merger")}
        self.t = {}

    def _run(self, op, art, args):
        from vlm_run import run
        self.t[op] = self.t.get(op, 0.0) + run(art, args)

    def merger(self, name, mode, x, br):
        self._run("vln", f"vlm-vln-{mode}", [x, br, self.p[name], self.xs, self.am])
        self._run("m_fc1", "vlm-m-fc1", [self.am, self.w[name, 1], self.cm1])
        self._run("gelu", "vlm-gelu-mrg", [self.cm1, self.am2])
        self._run("m_fc2", "vlm-m-fc2", [self.am2, self.w[name, 2], self.cm2[name]])

    def forward(self, pixel_values, upto=DEPTH):
        """pixel_values (4096, 1536) in processor order. Returns (image embeds, [deepstack x3]) as (1024, 4096)."""
        px = np.zeros((NP, 1600), np.float32)
        px[:, :1536], px[:, 1536] = pixel_values[self.src], 1.0
        apx = iron.tensor(to_bfp16(a_layout(px, 128, 64)).view(V9), dtype=V9, device="npu")
        if not hasattr(self, "pos_c"):  # fixed per resolution (1024x1024 only): computed once
            pos = np.zeros((NP, 2048), np.float32)
            pos[:, :HID] = pos_embed(self.row, self.col)
            self.pos_c = c_layout_atb(bf(pos), 128, 128).astype(bfloat16)
        self.x[0] = iron.tensor(self.pos_c, dtype=bfloat16, device="npu")
        cpatch = v9(NP * 2048)
        self._run("patch", "vlm-v-patch", [apx, self.w["patch"], cpatch])
        cur, br = 0, cpatch
        for i in self.blocks[:upto]:
            xi, xa, xb = self.x[cur], self.x[(cur + 1) % 3], self.x[(cur + 2) % 3]
            self._run("vln", "vlm-vln-block", [xi, br, self.p[i, 1], xa, self.a1])
            self._run("qkv", "vlm-v-qkv", [self.a1, self.w[i, "qkv"], self.cqkv])
            self._run("qkv_post", "vlm-vqkv-m4096", [self.cqkv, self.tabs, self.q, self.kv])
            self._run("attn", "vlm-v-attn", [self.q, self.kv, self.o])
            self._run("proj", "vlm-v-proj", [self.o, self.w[i, "proj"], self.cproj])
            self._run("vln", "vlm-vln-block", [xa, self.cproj, self.p[i, 2], xb, self.a1])
            self._run("fc1", "vlm-v-fc1", [self.a1, self.w[i, "fc1"], self.cfc1])
            self._run("gelu", "vlm-gelu-blk", [self.cfc1, self.afc2])
            self._run("fc2", "vlm-v-fc2", [self.afc2, self.w[i, "fc2"], self.cfc2])
            cur, br = (cur + 2) % 3, self.cfc2
            if i in DEEPSTACK:
                self.merger(f"ds{DEEPSTACK.index(i)}", "post", self.x[cur], br)
        self.cur, self.br = cur, br
        if upto < DEPTH:
            return None
        self.merger("merger", "merge", self.x[cur], br)
        un = lambda t: c_unlayout_atb(from_bfp16(np.frombuffer(t.numpy().tobytes(), np.uint8)), NT_, 4096, 128, 128)
        return un(self.cm2["merger"]), [un(self.cm2[f"ds{k}"]) for k in range(3)]

    def peek(self):
        """Residual stream after the last block run (x + fc2), (4096, 1152), stored (p-major) order."""
        a = c_unlayout_atb(self.x[self.cur].numpy().astype(np.float32), NP, 2048, 128, 128)
        b = c_unlayout_atb(from_bfp16(np.frombuffer(self.br.numpy().tobytes(), np.uint8)), NP, 2048, 128, 128)
        return (a.astype(np.float64) + b)[:, :HID]


# ---- float64 references (original weights) ----

def _w():
    from pack_weights import SafeTensors
    return SafeTensors(TE)


def ln(x, w, b):
    m = x.mean(-1, keepdims=True)
    return (x - m) / np.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + 1e-6) * w + b


def erf(x):
    """Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7)."""
    s, a = np.sign(x), np.abs(x)
    t = 1 / (1 + 0.3275911 * a)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def block_ref(x, i, row, col, s=None):
    """float64 vision block i on x (rows in any order, positions row/col per row)."""
    s = s or _w()
    g = lambda n: s.get(f"model.visual.blocks.{i}." + n).astype(np.float64)
    inv = 1.0 / 10000.0 ** (np.arange(0, 36, 2) / 36)
    fr = np.concatenate([row[:, None] * inv, col[:, None] * inv], 1)
    c2, s2 = np.cos(np.concatenate([fr, fr], 1)), np.sin(np.concatenate([fr, fr], 1))
    rot = lambda y: np.concatenate([-y[..., 36:], y[..., :36]], -1)
    h = ln(x, g("norm1.weight"), g("norm1.bias"))
    qkv = (h @ g("attn.qkv.weight").T + g("attn.qkv.bias")).reshape(-1, 3, H, HD)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q, k = q * c2[:, None] + rot(q) * s2[:, None], k * c2[:, None] + rot(k) * s2[:, None]
    o = np.zeros((x.shape[0], H * HD))
    for hh in range(H):
        sc = q[:, hh] @ k[:, hh].T / np.sqrt(HD)
        w = np.exp(sc - sc.max(1, keepdims=True))
        o[:, hh * HD:(hh + 1) * HD] = (w / w.sum(1, keepdims=True)) @ v[:, hh]
    x = x + o @ g("attn.proj.weight").T + g("attn.proj.bias")
    h = ln(x, g("norm2.weight"), g("norm2.bias"))
    return x + gelu_tanh(h @ g("mlp.linear_fc1.weight").T + g("mlp.linear_fc1.bias")) @ g("mlp.linear_fc2.weight").T \
        + g("mlp.linear_fc2.bias")


def merger_ref(x, name, s=None):
    """float64 merger on x (4096, 1152) in p-major order -> (1024, 4096)."""
    from math import sqrt
    s = s or _w()
    p = "model.visual.merger." if name == "merger" else f"model.visual.deepstack_merger_list.{name[2:]}."
    g = lambda n: s.get(p + n).astype(np.float64)
    xm = x.reshape(4, NT_, HID).transpose(1, 0, 2)                 # (1024, 4, 1152): token t, patches p
    if name == "merger":
        xm = ln(xm, g("norm.weight"), g("norm.bias")).reshape(NT_, 4 * HID)
    else:
        xm = ln(xm.reshape(NT_, 4 * HID), g("norm.weight"), g("norm.bias"))
    h = xm @ g("linear_fc1.weight").T + g("linear_fc1.bias")
    h = 0.5 * h * (1 + erf(h / sqrt(2)))
    return h @ g("linear_fc2.weight").T + g("linear_fc2.bias")


def check():
    px = np.load(GN / "vlm_pixel_values.npy")
    t = time.perf_counter()
    vis = Vision(blocks=range(1))
    print(f"block 0 weights resident in {time.perf_counter() - t:.1f} s")
    vis.forward(px, upto=1)
    print("npu ms:", {k: round(v, 2) for k, v in vis.t.items()})
    xn = vis.peek()
    # reference: x0 = patch_embed + pos, then block 0, in the stored (p-major) order
    s = _w()
    w = s.get("model.visual.patch_embed.proj.weight").reshape(HID, -1).astype(np.float64)
    x0 = px[vis.src].astype(np.float64) @ w.T + s.get("model.visual.patch_embed.proj.bias") + pos_embed(vis.row, vis.col)
    xr = block_ref(x0, 0, vis.row, vis.col, s)
    print(f"vision block 0 vs float64: corr {corr(xn, xr):.6f}  max err {np.abs(xn - xr).max() / np.abs(xr).max():.4f}")
    mr = merger_ref(xr, "merger", s)
    vis.merger("merger", "merge", vis.x[vis.cur], vis.br)
    un = c_unlayout_atb(from_bfp16(np.frombuffer(vis.cm2["merger"].numpy().tobytes(), np.uint8)), NT_, 4096, 128, 128)
    print(f"main merger (on block-0 output) vs float64: corr {corr(un, mr):.6f}")
    dr = merger_ref(xr, "ds0", s)
    vis.merger("ds0", "post", vis.x[vis.cur], vis.br)
    un = c_unlayout_atb(from_bfp16(np.frombuffer(vis.cm2["ds0"].numpy().tobytes(), np.uint8)), NT_, 4096, 128, 128)
    print(f"deepstack merger 0 (on block-0 output) vs float64: corr {corr(un, dr):.6f}")


if __name__ == "__main__":
    {"check": check}[sys.argv[1]](*sys.argv[2:])
