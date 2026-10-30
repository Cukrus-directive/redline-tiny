"""One-time preparation on the user's PC, run by the installer (safe to re-run: finished steps are skipped).

  1. check the official model download is complete
  2. pack the DiT, text-encoder and vision-tower weights for the NPU (numpy only, ~2-5 min, ~16 GB written)
  3. export the VAE encoder to ONNX for the iGPU and convert it to fp16 (~1-2 min)
  4. check every NPU program against the release manifest (SHA-256)
The VAE decoder's weights are packed on its first decode (the installer's self-test does that).
Usage: python prepare.py [--verify-only]
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
MODEL_FILES = ["model_index.json", "processor/tokenizer.json", "processor/processor_config.json",
               "text_encoder/model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]


def step(title):
    print(f"\n== {title}", flush=True)


def run(*args):
    t = time.time()
    r = subprocess.run([sys.executable, *map(str, args)], cwd=HERE)
    if r.returncode:
        sys.exit(f"ERROR: {' '.join(map(str, args))} failed (exit code {r.returncode})")
    print(f"   Done: {time.time() - t:.0f} s", flush=True)


def check_model():
    step(f"Model files: {paths.MODEL}")
    missing = [f for f in MODEL_FILES if not (paths.MODEL / f).exists()]
    if not list((paths.MODEL / "transformer").glob("*.safetensors")):
        missing.append("transformer/*.safetensors")
    if missing:
        sys.exit("ERROR: model incomplete. Missing: " + ", ".join(missing) + ". Re-run Install.cmd.")
    print("   OK")


def marker(d, name):
    """Written after a step's script exits cleanly. The scripts write their outputs in place, so an interrupted run
    leaves files that look complete; only the marker counts, and a step without one is redone from scratch."""
    return d / f".done-{name}"


def done(d, name):
    if marker(d, name).exists():
        print("   Skipped (already done)")
        return True
    return False


def mark(d, name):
    d.mkdir(parents=True, exist_ok=True)
    marker(d, name).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


def pack():
    w = paths.WEIGHTS
    jobs = ["--jobs", "3"]
    step("Packing DiT weights")
    if not done(w, "dit"):
        run(HERE / "pack_weights.py", *jobs)
        mark(w, "dit")
    step("Packing text encoder weights")
    if not done(w, "vlm"):
        run(HERE / "vlm_pack.py", *jobs)
        mark(w, "vlm")
    step("Packing vision tower weights")
    if not done(w, "vision"):
        run(HERE / "vlm_pack_vision.py", *jobs)
        mark(w, "vision")


def onnx():
    step("Exporting VAE encoder (ONNX fp16, iGPU)")
    o = paths.ONNX
    if done(o, "onnx"):
        return
    run(HERE / "igpu_vae.py", "export")
    run(HERE / "igpu_vae.py", "fp16")
    for f in ("vae_encoder.onnx", "vae_decoder.onnx", "vae_decoder_fp16.onnx"):   # only the fp16 encoder is used
        (o / f).unlink(missing_ok=True)
    mark(o, "onnx")


def verify_npu():
    step(f"NPU programs: {paths.NPU}")
    man = paths.NPU / "manifest.json"
    if not man.exists():
        sys.exit(f"ERROR: {man} not found (incomplete download). Re-run Install.cmd.")
    bad = []
    for rel, info in json.loads(man.read_text())["files"].items():
        f = paths.NPU / rel
        if not f.exists() or f.stat().st_size != info["size"]:
            bad.append(rel)
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != info["sha256"]:
            bad.append(rel)
    if bad:
        sys.exit(f"ERROR: {len(bad)} NPU program files missing or failed the SHA-256 check (first: {bad[0]}). "
                 "Re-run Install.cmd.")
    print("   OK (SHA-256 verified)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    check_model()
    verify_npu()
    if not a.verify_only:
        pack()
        onnx()
        mark(paths.WEIGHTS, "prepare")   # start.ps1 checks this one
    print("\nPreparation: complete", flush=True)


if __name__ == "__main__":
    main()
