"""Pack the Qwen3-VL text encoder's language model (36 layers) for the NPU, one layer at a time.
Reads the read-only checkpoint without torch (pack_weights.SafeTensors).

Writes data\\weights\\vlm\\:
  LNN_qkv.bin   B = [Wq.T | Wk.T | Wv.T], 4096 x 6144 (q 32 heads | k 8 | v 8), q/k rows permuted per head for RoPE
  LNN_o.bin     B = Wo.T, 4096 x 4096
  LNN_up.bin    B = gate|up interleaved per 128-col tile as [gate 64 | up 64], 4096 x 24576 (feeds glue-swiglu)
  LNN_down.bin  B = Wdown.T, 12288 x 4096
  LNN_norms.npz input_layernorm (4096), post_attention_layernorm (4096), q_norm, k_norm (128, permuted), float32
  manifest.json
RoPE permutation: the model rotates dims (i, i+64) together (rotate-half); qkv_post rotates adjacent pairs (2i, 2i+1).
Head dim i -> 2i and i+64 -> 2i+1 makes them the same; q.k is unchanged because q and k are permuted alike.
Usage (IRON env or any numpy + ml_dtypes): python vlm_pack.py [--layers 0-35] [--jobs 3]
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pack_weights import SafeTensors, interleave_up, pack_b, parse_blocks

from paths import MODEL, WEIGHTS
SRC = MODEL / "text_encoder" / "model.safetensors"
OUT = WEIGHTS / "vlm"
N_LAYERS, HQ, HKV, D = 36, 32, 8, 128
PERM = np.stack([np.arange(64), np.arange(64) + 64], 1).ravel()  # new dim 2i <- i, 2i+1 <- i+64


def permute_heads(w, heads):
    """(heads*128, K) projection weight -> rows permuted per head by PERM."""
    return w.reshape(heads, D, -1)[:, PERM, :].reshape(heads * D, -1)


def hidden_perm():
    """Hidden-channel order: channels sorted by peak |x| (massive activations first), so each bfp16 block of 8
    channels holds similar magnitudes and the few massive channels (2276 at ~12800, 233, 1838, 676) share block 0
    instead of crushing 7 ordinary channels each. RMSNorm and every matmul are permutation-invariant."""
    f = OUT / "hidden_perm.npy"
    return np.load(f) if f.exists() else np.arange(4096)


def pack_layer(i):
    s = SafeTensors(SRC)
    p = f"model.language_model.layers.{i}."
    t0 = time.time()
    g = lambda n: s.get(p + n)
    hp = hidden_perm()  # new channel n = old channel hp[n]
    mats = {  # B = W.T is K x N: hidden-in rows permuted (qkv, up), hidden-out columns permuted (o, down)
        "qkv": lambda: np.concatenate([permute_heads(g("self_attn.q_proj.weight"), HQ).T,
                                       permute_heads(g("self_attn.k_proj.weight"), HKV).T,
                                       g("self_attn.v_proj.weight").T], 1)[hp],
        "o": lambda: g("self_attn.o_proj.weight").T[:, hp],
        "up": lambda: interleave_up(g("mlp.gate_proj.weight"), g("mlp.up_proj.weight"))[hp],
        "down": lambda: g("mlp.down_proj.weight").T[:, hp],
    }
    entry = {}
    for name, build in mats.items():
        B = build()
        raw = pack_b(B)
        path = OUT / f"L{i:02d}_{name}.bin"
        raw.tofile(path)
        entry[name] = {"file": path.name, "K": B.shape[0], "N": B.shape[1], "bytes": int(raw.size),
                       "sha256": hashlib.sha256(raw).hexdigest()}
        del B, raw
    np.savez(OUT / f"L{i:02d}_norms.npz", w_in=g("input_layernorm.weight")[hp], w_post=g("post_attention_layernorm.weight")[hp],
             q_norm=g("self_attn.q_norm.weight")[PERM], k_norm=g("self_attn.k_norm.weight")[PERM])
    entry["norms"] = {"file": f"L{i:02d}_norms.npz"}
    return i, entry, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=f"0-{N_LAYERS - 1}")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    layers = parse_blocks(args.layers)
    man_path = OUT / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"layers": {}}
    man.update({"source": str(SRC), "source_header_sha256": SafeTensors(SRC).header_sha,
                "layout": "to_bfp16(b_layout(B, 64, 128)), B = W.T (K x N)", "rope_perm": "head dim i -> 2i, i+64 -> 2i+1",
                "hidden_perm": "hidden_perm.npy: new channel n = old channel hidden_perm[n] (identity if absent); "
                               "embeddings and norm weights permuted alike, the output un-permuted on the host",
                "config": {"layers": N_LAYERS, "hidden": 4096, "q_heads": HQ, "kv_heads": HKV, "head_dim": D,
                           "intermediate": 12288, "rms_eps": 1e-6, "rope_theta": 5e6, "mrope_section": [24, 20, 20]}})
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        for i, entry, dt in ex.map(pack_layer, layers):
            man["layers"][str(i)] = entry
            print(f"layer {i:2d}: {dt:.1f} s", flush=True)
    man_path.write_text(json.dumps(man, indent=1))
    print(f"{len(layers)} layers in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
