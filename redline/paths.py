"""Where redline-tiny keeps everything. One install folder with fixed sub-folders:

  models\\Qwen-Image-2.1-Turbo\\   the official model, downloaded from Hugging Face (only ever read)
  data\\weights\\                  the same weights packed for the NPU (bfp16, core order), written once by prepare.py
  data\\onnx\\                     the VAE encoder exported for the iGPU (DirectML), written once by prepare.py
  data\\npu\\                      the compiled NPU programs (xclbin + instructions, merged DiT ELFs), downloaded
  data\\cache\\                    programs patched at runtime for the prompt length; safe to delete
  outputs\\                        your images
Any of them can point elsewhere: an environment variable (REDLINE_MODEL, REDLINE_WEIGHTS, REDLINE_ONNX, REDLINE_NPU,
REDLINE_CACHE, REDLINE_OUTPUTS) or the same key in lower case in settings.json at the install root.
"""

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("REDLINE_HOME") or Path(__file__).resolve().parent.parent)
try:
    _SETTINGS = json.loads((ROOT / "settings.json").read_text(encoding="utf-8-sig"))   # -sig: PowerShell 5.1 writes a BOM
except OSError:
    _SETTINGS = {}
except ValueError as e:
    raise SystemExit(f"ERROR: {ROOT / 'settings.json'} is not valid JSON ({e}). Fix or delete it, then re-run Install.cmd.")


def _p(key, default):
    v = os.environ.get(f"REDLINE_{key.upper()}") or _SETTINGS.get(key)
    return Path(v) if v else ROOT / default


MODEL = _p("model", r"models\Qwen-Image-2.1-Turbo")
WEIGHTS = _p("weights", r"data\weights")
ONNX = _p("onnx", r"data\onnx")
NPU = _p("npu", r"data\npu")
CACHE = _p("cache", r"data\cache")
OUTPUTS = _p("outputs", "outputs")
LOCK = CACHE / "npu.lock"
ASSETS = Path(__file__).resolve().parent / "assets"
HF_MODEL_REPO = "Qwen/Qwen-Image-2.1-Turbo"


class MissingProgram(RuntimeError):
    pass


def need_build(name):
    """Called where a missing NPU program would be compiled on the fly. Releases ship every program the
    app can reach, so this only happens with a broken or partial download (or in a developer checkout, where
    REDLINE_ALLOW_BUILD=1 and an mlir-aie toolchain build it)."""
    if os.environ.get("REDLINE_ALLOW_BUILD") != "1":
        raise MissingProgram(f"NPU program '{name}' is missing from {NPU}. Re-run Install.cmd to repair the download.")
