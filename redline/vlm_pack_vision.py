"""Pack the Qwen3-VL vision tower (27 blocks + 4 mergers + patch embed) for the NPU.
Reads the read-only checkpoint without torch; writes data\\weights\\vlm\\vision\\.

Every bias rides in the GEMM: the A operand carries a constant-1 column (a ones tile the host writes once into the A
buffer) and B gets the bias as that row. Shapes (K x N, config3: K a multiple of 64, N of 1024):
  patch.bin   1600 x 2048    rows 0..1535 = conv weight (1152, 3*2*16*16).T, row 1536 = bias
  vNN_qkv     1216 x 12288   32 head slots of 128 per q|k|v (16 used): head h dim d -> slot col h*128 + d' with
                             d' = 2d (d < 36), 2(d - 36) + 1 (d >= 36) for q and k (rotate-half pairs -> adjacent
                             pairs for qkv_post), d' = d for v; row 1152 = bias; v slot 0 col 72 gets bias 1.0, so
                             the attention output carries a constant 1 there (the proj bias row)
  vNN_proj    2048 x 2048    row h*128 + d = W.T[h*72 + d]; row 72 (head 0's constant-1 dim) = bias
  vNN_fc1     1216 x 5120    rows 0..1151 = W1.T, row 1152 = b1 (cols 0..4303 used)
  vNN_fc2     4416 x 2048    rows 0..4303 = W2.T, row 4352 = b2 (the GELU glue's A has its ones tile at k-tile 68)
  vNN_norms.npz  n1w, n1b, n2w, n2b (1152)
  merger_fc1 / dsK_fc1   4672 x 5120   rows 0..4607 = W1.T (input = 4 patches x 1152), row 4608 = b1
  merger_fc2 / dsK_fc2   4672 x 4096   rows 0..4607 = W2.T, row 4608 = b2
  merger_norm.npz (w, b: 1152, per patch), dsK_norm.npz (w, b: 4608, over the 4 patches)
  pos_embed.npy (2304, 1152) float32 (host bilinear interpolation)
Usage: python vlm_pack_vision.py [--jobs 3]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pack_weights import SafeTensors, pack_b

from paths import MODEL, WEIGHTS
SRC = MODEL / "text_encoder" / "model.safetensors"
OUT = WEIGHTS / "vlm" / "vision"
H, HD, HID, FF = 16, 72, 1152, 4304
PAIR = np.concatenate([2 * np.arange(36), 2 * np.arange(36) + 1])  # d -> d' (rotate-half -> adjacent pairs)


def blk(i):
    s = SafeTensors(SRC)
    p = f"model.visual.blocks.{i}."
    g = lambda n: s.get(p + n)
    t0 = time.time()
    w, b = g("attn.qkv.weight"), g("attn.qkv.bias")          # (3456, 1152), (3456,)
    B = np.zeros((1216, 12288), np.float32)
    for part in range(3):
        for h in range(H):
            src = slice(part * HID + h * HD, part * HID + (h + 1) * HD)
            cols = part * 4096 + h * 128 + (PAIR if part < 2 else np.arange(HD))
            B[:HID, cols] = w[src].T
            B[HID, cols] = b[src]
    B[HID, 8192 + 72] = 1.0                                     # v head 0, dim 72 = 1 -> O[:, 72] = 1
    pack_b(B).tofile(OUT / f"v{i:02d}_qkv.bin")
    wp, bp = g("attn.proj.weight"), g("attn.proj.bias")         # (1152, 1152)
    B = np.zeros((2048, 2048), np.float32)
    for h in range(H):
        B[h * 128:h * 128 + HD, :HID] = wp.T[h * HD:(h + 1) * HD]
    B[72, :HID] = bp
    pack_b(B).tofile(OUT / f"v{i:02d}_proj.bin")
    B = np.zeros((1216, 5120), np.float32)
    B[:HID, :FF], B[HID, :FF] = g("mlp.linear_fc1.weight").T, g("mlp.linear_fc1.bias")
    pack_b(B).tofile(OUT / f"v{i:02d}_fc1.bin")
    B = np.zeros((4416, 2048), np.float32)
    B[:FF, :HID], B[4352, :HID] = g("mlp.linear_fc2.weight").T, g("mlp.linear_fc2.bias")
    pack_b(B).tofile(OUT / f"v{i:02d}_fc2.bin")
    np.savez(OUT / f"v{i:02d}_norms.npz", n1w=g("norm1.weight"), n1b=g("norm1.bias"), n2w=g("norm2.weight"),
             n2b=g("norm2.bias"))
    return i, time.time() - t0


def merger(name):
    s = SafeTensors(SRC)
    p = "model.visual.merger." if name == "merger" else f"model.visual.deepstack_merger_list.{name[2:]}."
    g = lambda n: s.get(p + n)
    B = np.zeros((4672, 5120), np.float32)
    B[:4608, :4608], B[4608, :4608] = g("linear_fc1.weight").T, g("linear_fc1.bias")
    pack_b(B).tofile(OUT / f"{name}_fc1.bin")
    B = np.zeros((4672, 4096), np.float32)
    B[:4608], B[4608] = g("linear_fc2.weight").T, g("linear_fc2.bias")
    pack_b(B).tofile(OUT / f"{name}_fc2.bin")
    np.savez(OUT / f"{name}_norm.npz", w=g("norm.weight"), b=g("norm.bias"))
    return name, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s = SafeTensors(SRC)
    w = s.get("model.visual.patch_embed.proj.weight").reshape(HID, -1)   # (1152, 1536)
    B = np.zeros((1600, 2048), np.float32)
    B[:1536, :HID], B[1536, :HID] = w.T, s.get("model.visual.patch_embed.proj.bias")
    pack_b(B).tofile(OUT / "patch.bin")
    np.save(OUT / "pos_embed.npy", s.get("model.visual.pos_embed.weight"))
    with ProcessPoolExecutor(args.jobs) as ex:
        for i, dt in ex.map(blk, range(27)):
            print(f"block {i:2d}: {dt:.1f} s", flush=True)
        for n, _ in ex.map(merger, ["merger", "ds0", "ds1", "ds2"]):
            print(f"{n} done", flush=True)
    print(f"vision packed in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
