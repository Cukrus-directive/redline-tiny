"""Load a compiled NPU program (a folder with final.xclbin + insts.bin, see paths.NPU) once per process."""

import subprocess
from pathlib import Path

from aie.utils import DefaultNPURuntime
from aie.utils.npukernel import NPUKernel


def pmode():
    """The NPU power mode, as xrt-smi reports it (benchmark logs only)."""
    out = subprocess.run([r"C:\Windows\System32\AMD\xrt-smi.exe", "examine", "--report", "platform"],
                         capture_output=True, text=True).stdout
    return next((l.split(":", 1)[1].strip() for l in out.splitlines() if "Power Mode" in l), "?")


_loaded = {}


def load(art):
    """Load an artifact's xclbin + instructions once per process; returns the runtime handle. Survives eviction:
    Strix has 16 hardware contexts, the runtime evicts least-recently-used ones and invalidates their handles
    (`_is_valid` False, then 'CachedXRTKernelHandle has no attribute xclbin' on use), so reload those."""
    art = Path(art)
    h = _loaded.get(art)
    if h is None or not getattr(h, "_is_valid", True):
        h = _loaded[art] = DefaultNPURuntime.load(NPUKernel(art / "final.xclbin", art / "insts.bin"))
    return h
