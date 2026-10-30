"""Host-side helpers for the VAE decoder on the NPU: the octet-planar activation layout, weight
packing for kernels/vae/vconv_design.py, numpy references, and the checkpoint reader.

Octet-planar (OPB): bf16 [H+2][C/8][W+2][8] with a 1-pixel zero border: element (y, x, c) sits at
((y+1)*C/8 + c//8)*(W+2)*8 + (x+1)*8 + c%8.
"""

import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from npu_layouts import to_bfp16
from pack_weights import SafeTensors

from paths import MODEL
VAE_DIR = MODEL / "vae"
VAE_ST = VAE_DIR / "diffusion_pytorch_model.safetensors"


def vae():
    return SafeTensors(VAE_ST)


def vae_config():
    return json.loads((VAE_DIR / "config.json").read_text())


def opb_pack(x):
    """(H, W, C) -> OPB bf16, flat."""
    H, W, C = x.shape
    out = np.zeros((H + 2, C // 8, W + 2, 8), bfloat16)
    out[1:-1, :, 1:-1, :] = np.asarray(x, np.float32).reshape(H, W, C // 8, 8).transpose(0, 2, 1, 3)
    return out.reshape(-1)


def opb_unpack(buf, H, W, C):
    """OPB (flat) -> (H, W, C) float32 (border dropped)."""
    x = np.asarray(buf).reshape(H + 2, C // 8, W + 2, 8)[1:-1, :, 1:-1, :].astype(np.float32)
    return x.transpose(0, 2, 1, 3).reshape(H, W, C)


def pack_vconv(w, b, kc, nc, cout_pad=None):
    """Weight stream for vconv: w (Cout, Cin, k, k) with k = 3 or 1, b (Cout,). Per output-channel chunk of nc:
    its taps*Cin/kc step elements [jp][dx][c8][h][8 rows][8 cols] (rows = outputs, cols = inputs), then one
    element whose first nc bf16 are the biases. Cout is zero-padded to cout_pad (e.g. conv_out 4 -> 16)."""
    w = np.asarray(w, np.float32)
    b = np.zeros(w.shape[0], np.float32) if b is None else np.asarray(b, np.float32)
    if cout_pad and cout_pad > w.shape[0]:
        w = np.concatenate([w, np.zeros((cout_pad - w.shape[0],) + w.shape[1:], np.float32)])
        b = np.concatenate([b, np.zeros(cout_pad - b.shape[0], np.float32)])
    cout, cin, k = w.shape[0], w.shape[1], w.shape[2]
    x = w.reshape(cout // nc, nc // 16, 2, 8, cin // kc, kc // 8, 8, k, k)  # co, jp, h, r, ci, c8, c, dy, dx
    x = np.ascontiguousarray(x.transpose(0, 7, 4, 1, 8, 5, 2, 3, 6))     # co, dy, ci, jp, dx, c8, h, r, c
    steps = to_bfp16(x).reshape(cout // nc, -1)
    welem = k * kc * nc // 8 * 9
    out = np.zeros((cout // nc, steps.shape[1] + welem), np.uint8)
    out[:, :steps.shape[1]] = steps
    out[:, steps.shape[1]:steps.shape[1] + 2 * nc] = b.astype(bfloat16).reshape(cout // nc, nc).view(np.uint8)
    return out.reshape(-1)


def conv_ref(x, w, b):
    """x (H, W, Cin), w (Cout, Cin, k, k) with k = 3 (pad 1) or 1 -> (H, W, Cout), float64."""
    H, W, _ = x.shape
    k = w.shape[2]
    y = np.zeros((H, W, w.shape[0])) + (0 if b is None else np.asarray(b, np.float64))
    if k == 1:
        return y + x.astype(np.float64) @ w[:, :, 0, 0].T.astype(np.float64)
    xp = np.zeros((H + 2, W + 2, x.shape[2]))
    xp[1:-1, 1:-1] = x
    for dy in range(3):
        for dx in range(3):
            y += xp[dy:dy + H, dx:dx + W] @ w[:, :, dy, dx].T.astype(np.float64)
    return y


def rms_norm_ref(x, gamma):
    """QwenImage21RMS_norm over channels: x / max(||x||, 1e-12) * sqrt(C) * gamma."""
    n = np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True))
    return x / np.maximum(n, 1e-12) * np.sqrt(x.shape[-1]) * gamma


def silu(x):
    return x / (1 + np.exp(-x))


def stats(name, got, ref):
    got, ref = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    e = np.abs(got - ref)
    c = np.corrcoef(got, ref)[0, 1]
    print(f"  {name}: corr {c:.6f}  max err {e.max() / np.abs(ref).max():.4f}  "
          f"rms err {np.sqrt((e ** 2).mean() / (ref ** 2).mean()):.4f}")
    return c
