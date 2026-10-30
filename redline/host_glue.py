"""Per-step host glue for the NPU forward: tiny ops that run once per step or per edit (matvecs on 2 rows, 32 text
tokens), in numpy float32 from the model's small weights.
  temb(t)            : sinusoidal(256) -> linear_1 -> SiLU -> linear_2, rows [t, 0] (causal_condition)
  modulation(temb)   : Linear(SiLU(temb)) -> (2, 16384) = [mod1.scale, mod1.gate, mod2.scale, mod2.gate] x 4096
  norm_out_scale     : Linear(SiLU(temb)) -> (2, 4096)
  txt_in(embeds)     : zero-centred RMSNorm -> Linear -> GELU(tanh) -> Linear
Developer self-check against reference tensors in golden\\ (not shipped; needs torch): python host_glue.py
"""

import math

import numpy as np

from paths import MODEL, ROOT   # ROOT/golden: developer reference tensors only

SMALL = ["time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_1.bias",
         "time_text_embed.timestep_embedder.linear_2.weight", "time_text_embed.timestep_embedder.linear_2.bias",
         "modulation.1.weight", "norm_out.linear.weight", "txt_in.text_norm.weight", "txt_in.in_layer.weight",
         "txt_in.out_layer.weight", "img_in.weight", "proj_out.weight"]


def silu(x):
    return x / (1 + np.exp(-x))


def linear(x, w, b=None):
    y = x @ w.T
    return y + b if b is not None else y


def temb(w, t):
    """t = the pipeline's timestep (sigma, in [0, 1]); returns (2, 4096): rows [t, 0]."""
    ts = np.array([t, 0.0], np.float32) * 1000.0
    half = 128
    freqs = np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / half)
    args = ts[:, None] * freqs[None]
    e = np.concatenate([np.cos(args), np.sin(args)], -1).astype(np.float32)
    p = "time_text_embed.timestep_embedder."
    h = silu(linear(e, w[p + "linear_1.weight"], w.get(p + "linear_1.bias")))
    return linear(h, w[p + "linear_2.weight"], w.get(p + "linear_2.bias"))


def modulation(w, te):
    return linear(silu(te), w["modulation.1.weight"])


def norm_out_scale(w, te):
    return linear(silu(te), w["norm_out.linear.weight"])


def txt_in(w, e):
    e = e.astype(np.float32)
    y = e / np.sqrt((e * e).mean(-1, keepdims=True) + 1e-6) * (w["txt_in.text_norm.weight"] + 1)
    h = linear(y, w["txt_in.in_layer.weight"])
    h = 0.5 * h * (1 + np.tanh(math.sqrt(2 / math.pi) * (h + 0.044715 * h ** 3)))
    return linear(h, w["txt_in.out_layer.weight"])


def joint_layout(vlm_image_mask, n_target=4096):
    """The joint sequence: VLM positions (image slots stand for 2x2 latent tokens) plus the target's slots appended.
    Returns image_pad_mask (joint, bool) and img_shapes [(1, 64, 64)] * (#images incl. target), 1024x1024 only."""
    m = np.concatenate([np.asarray(vlm_image_mask, bool).ravel(), np.ones(n_target // 4, bool)])
    pad = np.repeat(m, np.where(m, 4, 1))
    n_img = int(pad.sum()) // n_target
    return pad, [(1, 64, 64)] * n_img


def rope(image_pad_mask, img_shapes, axes=(16, 56, 56), theta=10000):
    """QwenImage21Rope as (seq, 64, 2) = (cos, sin): text advances one shared position on all 3 axes; each image
    block freezes the frame axis and lays out a centred h/w grid."""
    is_img = list(image_pad_mask)
    frame, ih, iw = [], [], []
    cursor = position = 0
    for _, h, w in img_shapes:
        start = is_img.index(True, cursor)
        frame.extend(range(position, position + start - cursor))
        position += start - cursor
        cursor = start + h * w
        frame.extend([position] * (h * w))
        position += max(h, w)
        ih.extend([y for y in range(-(h - h // 2), h // 2) for _ in range(w)])
        iw.extend([x for _ in range(h) for x in range(-(w - w // 2), w // 2)])
    if cursor < len(is_img):
        frame.extend(range(position, position + len(is_img) - cursor))
    frame = np.array(frame)
    hidx, widx = frame.copy(), frame.copy()
    hidx[np.asarray(image_pad_mask)] = ih
    widx[np.asarray(image_pad_mask)] = iw
    ang = [np.outer(idx, 1.0 / theta ** (np.arange(0, d, 2, dtype=np.float32) / d)) for idx, d in zip((frame, hidx, widx), axes)]
    a = np.concatenate(ang, -1)
    return np.stack([np.cos(a), np.sin(a)], -1).astype(np.float32)


def load_small_from_model():
    """Small weights straight from the read-only model files (needs torch for bf16)."""
    from pathlib import Path
    from safetensors import safe_open
    w = {}
    for f in sorted((MODEL / "transformer").glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in SMALL:
                if k in s.keys():
                    w[k] = s.get_tensor(k).float().numpy()
    return w


def main():
    from safetensors import safe_open
    w = load_small_from_model()
    print("small weights:", {k: v.shape for k, v in w.items()})

    def corr(a, b):
        a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
        return f"corr {np.corrcoef(a, b)[0, 1]:.6f}  max rel {np.abs(a - b).max() / np.abs(b).max():.4f}"

    for step in (0, 1):
        with safe_open(str(ROOT / "golden" / f"b_step{step}.safetensors"), "pt") as s:
            g = {k: s.get_tensor(k).float().numpy() for k in ("timestep", "time_text_embed.out", "modulation.out")}
        te = temb(w, float(g["timestep"][0]))
        print(f"step {step} (t={float(g['timestep'][0]):.6f}): temb {corr(te, g['time_text_embed.out'])};  "
              f"modulation {corr(modulation(w, te), g['modulation.out'])}")
    with safe_open(str(ROOT / "golden" / "a_text.safetensors"), "pt") as s:
        pe = s.get_tensor("prompt_embeds")[0].float().numpy()
    with safe_open(str(ROOT / "golden" / "b_step1.safetensors"), "pt") as s:
        gt = s.get_tensor("txt_in.out")[0].float().numpy()
    print(f"txt_in on all {pe.shape[0]} prompt tokens: {corr(txt_in(w, pe), gt)}")


if __name__ == "__main__":
    main()
