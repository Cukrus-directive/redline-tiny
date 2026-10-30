"""Pack the Qwen-Image-2.1-Turbo transformer weights for the NPU forward: bfp16ebs8 config3 B operands, one block at
a time. Reads the safetensors without torch (header JSON + raw bytes, memory-mapped read-only); the weights folder is
never written.

Writes data\\weights\\ (paths.WEIGHTS):
  bNN_qkv.bin   B = [Wq.T | Wk.T | Wv.T], 4096 x 12288    (atb3d2-qkv, atb3d2-s1qkv)
  bNN_o.bin     B = Wo.T, 4096 x 4096                     (atb3d2-4096, atb3d2-s1o)
  bNN_up.bin    B = gate|proj interleaved per 128-col tile as [gate 64 | proj 64], 4096 x 24576 (atb3sw2-m4096, atb3sw2-m8704;
                SwiGLU fused into the GEMM's flush)
  bNN_down.bin  B = Wdown.T, 12288 x 4096                 (atb3d2-down, atb3d2-s1down)
  each = the raw bytes of to_bfp16(b_layout(B, 64, 128)) (round half to even), 9 bytes per 8 values, exactly what
  run_prebuilt.py feeds the atb-bfp artifacts.
  bNN_norm_q.npy, bNN_norm_k.npy    float32 (128,)
  host\\<tensor name>.npy           every non-block tensor, float32 (small-M GEMMs run on the host)
  manifest.json
Usage (any python with numpy + ml_dtypes, e.g. the IRON env): python pack_weights.py [--blocks 0-31] [--jobs 3]
"""

import argparse
import hashlib
import json
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from npu_layouts import b_layout, to_bfp16

from paths import MODEL, WEIGHTS as OUT
SRC = MODEL / "transformer"
K_TILE, N_TILE = 64, 128
_DT = {"BF16": bfloat16, "F32": np.float32, "F16": np.float16}


class SafeTensors:
    """Read-only safetensors reader (no torch): tensors come back as float32 copies."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            raw = f.read(n)
        self.header = {k: v for k, v in json.loads(raw).items() if k != "__metadata__"}
        self.header_sha = hashlib.sha256(raw).hexdigest()
        self.base = 8 + n

    def keys(self):
        return list(self.header)

    def get(self, name):
        e = self.header[name]
        a, b = e["data_offsets"]
        mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.base + a, shape=(b - a,))
        return np.frombuffer(mm, dtype=_DT[e["dtype"]]).reshape(e["shape"]).astype(np.float32)


class Model:
    """The sharded transformer checkpoint: name -> float32 tensor."""

    def __init__(self, src=SRC):
        idx = json.loads((src / "diffusion_pytorch_model.safetensors.index.json").read_text())
        self.where = idx["weight_map"]
        self.files = {f: SafeTensors(src / f) for f in sorted(set(self.where.values()))}

    def get(self, name):
        return self.files[self.where[name]].get(name)


def pack_b(B):
    """(K, N) float32 -> config3 B stream bytes (bfp16ebs8), encoded in chunks to keep memory flat."""
    lay = b_layout(np.ascontiguousarray(B, np.float32), K_TILE, N_TILE)
    out = np.empty(lay.size // 8 * 9, np.uint8)
    ch = 1 << 23
    for i in range(0, lay.size, ch):
        out[i // 8 * 9:(i + ch) // 8 * 9] = to_bfp16(lay[i:i + ch])
    return out


def interleave_up(wg, wp):
    """gate_layer / proj weights (12288, 4096) -> B (4096, 24576) with [gate 64 | proj 64] per 128-col tile."""
    K, F = wg.shape[1], wg.shape[0]
    B = np.empty((K, 2 * F), np.float32)
    v = B.reshape(K, F // 64, 2, 64)
    v[:, :, 0] = wg.T.reshape(K, F // 64, 64)
    v[:, :, 1] = wp.T.reshape(K, F // 64, 64)
    return B


def pack_block(i):
    m = Model()
    p = f"transformer_blocks.{i}."
    t0 = time.time()
    mats = {
        "qkv": (lambda: np.concatenate([m.get(p + f"attn.to_{x}.weight").T for x in "qkv"], 1),
                [p + "attn.to_q.weight", p + "attn.to_k.weight", p + "attn.to_v.weight"], "[Wq.T | Wk.T | Wv.T]"),
        "o": (lambda: m.get(p + "attn.to_out.0.weight").T, [p + "attn.to_out.0.weight"], "Wo.T"),
        "up": (lambda: interleave_up(m.get(p + "img_mlp.gate_layer.weight"), m.get(p + "img_mlp.proj.weight")),
               [p + "img_mlp.gate_layer.weight", p + "img_mlp.proj.weight"],
               "[gate 64 | proj 64] per 128-col tile of [Wgate.T, Wproj.T]"),
        "down": (lambda: m.get(p + "img_mlp.out.weight").T, [p + "img_mlp.out.weight"], "Wdown.T"),
    }
    entry = {}
    for name, (build, src, how) in mats.items():
        B = build()
        raw = pack_b(B)
        path = OUT / f"b{i:02d}_{name}.bin"
        raw.tofile(path)
        entry[name] = {"file": path.name, "K": B.shape[0], "N": B.shape[1], "bytes": int(raw.size),
                       "sha256": hashlib.sha256(raw).hexdigest(), "source": src, "B": how}
        del B, raw
    for name in ("norm_q", "norm_k"):
        path = OUT / f"b{i:02d}_{name}.npy"
        np.save(path, m.get(p + f"attn.{name}.weight"))
        entry[name] = {"file": path.name, "shape": [128], "dtype": "float32", "source": [p + f"attn.{name}.weight"]}
    return i, entry, time.time() - t0


def parse_blocks(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0-31")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "host").mkdir(exist_ok=True)
    m = Model()
    cfg = json.loads((SRC / "config.json").read_text())
    man_path = OUT / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man.update({
        "model": "Qwen-Image-2.1-Turbo", "source": str(SRC), "config": cfg,
        "source_files": {f: {"bytes": st.path.stat().st_size, "header_sha256": st.header_sha}
                         for f, st in m.files.items()},
        "layout": {"format": "bfp16ebs8, 9 bytes per 8 values ([shared exp][8 x int8]), round half to even",
                   "B": f"b_layout(B, k={K_TILE}, n={N_TILE}) of B = K x N (C = A @ B), npu_layouts.py",
                   "A_expected": "to_bfp16(a_layout(A, 128, 64))", "C": "c_unlayout_atb(from_bfp16(c), M, N, 128, 128)",
                   "up_columns": "per 128-col tile: [gate 64 | proj 64]"},
        "artifacts": {"qkv": ["atb3d2-qkv", "atb3d2-s1qkv"], "o": ["atb3d2-4096", "atb3d2-s1o"],
                      "up": ["atb3sw2-m4096", "atb3sw2-m8704"], "down": ["atb3d2-down", "atb3d2-s1down"]},
    })
    host = {}
    t0 = time.time()
    for name in sorted(m.where):
        if name.startswith("transformer_blocks."):
            continue
        a = m.get(name)
        np.save(OUT / "host" / f"{name}.npy", a)
        host[name] = {"file": f"host/{name}.npy", "shape": list(a.shape), "dtype": "float32"}
    man["host"] = host
    print(f"host tensors: {len(host)} in {time.time() - t0:.0f} s", flush=True)
    # The two non-block GEMMs (forward.GEMM_KN): img_in B = W.T (64 x 4096); proj_out B = W.T zero-padded to the
    # program's 1024 output columns (the model has 64), the extra columns ignored on the host.
    w = m.get("proj_out.weight")
    proj = np.zeros((w.shape[1], 1024), np.float32)
    proj[:, :w.shape[0]] = w.T
    for name, B in (("img_in", m.get("img_in.weight").T), ("proj_out", proj)):
        raw = pack_b(B)
        raw.tofile(OUT / f"{name}.bin")
        man[name] = {"file": f"{name}.bin", "K": B.shape[0], "N": B.shape[1], "bytes": int(raw.size),
                     "sha256": hashlib.sha256(raw).hexdigest(), "source": [f"{name}.weight"]}
    print("img_in, proj_out packed", flush=True)
    blocks = man.setdefault("blocks", {})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, entry, dt in ex.map(pack_block, parse_blocks(args.blocks)):
            blocks[f"{i:02d}"] = entry
            man_path.write_text(json.dumps(man, indent=1))
            print(f"block {i:2d}: {dt:.0f} s  (elapsed {time.time() - t0:.0f} s)", flush=True)
    total = sum(e["bytes"] for b in blocks.values() for e in b.values() if "bytes" in e)
    man["total_block_bytes"] = total
    man_path.write_text(json.dumps(man, indent=1))
    print(f"done: {len(blocks)} blocks, {total / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
