"""Time kernels of a merged full ELF in ONE hw context: each alone (back to back) vs alternating,
against the same designs as separate xclbins (one context each). The difference is the switch cost per call.
Usage (IRON env active): python elf_probe.py ELF name=ARTIFACT [name=ARTIFACT ...] [--n 15]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyxrt

from aie.utils import DefaultNPURuntime
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTKernelHandle

from run_prebuilt import load

from paths import NPU as ART


def elf_kernels(elf_path, names):
    """One hw context for the whole ELF; a handle per '<name>:seq' kernel."""
    rt = DefaultNPURuntime
    ctx = pyxrt.hw_context(rt._device, pyxrt.elf(str(elf_path)))
    return {n: XRTKernelHandle(pyxrt.ext.kernel(ctx, f"{n}:seq"), None, ctx, None, name=f"{n}:seq", is_full_elf=True)
            for n in names}, ctx


def bufs_for(art):
    from residency_test import buffers
    meta = json.loads((art / "meta.json").read_text())
    if meta["family"] in ("glue-ln", "ln"):
        import aie.iron as iron
        from ml_dtypes import bfloat16
        from npu_layouts import V9
        M = meta.get("M", 4096)
        z = lambda n: iron.tensor(np.zeros(n // 8, dtype=V9), dtype=V9, device="npu")
        return [iron.zeros((M * 4096,), dtype=bfloat16, device="npu"), z(M * 4096),
                iron.tensor(np.ones(2 * 8192, dtype=bfloat16), dtype=bfloat16, device="npu"),
                iron.zeros((M * 4096,), dtype=bfloat16, device="npu"), z(M * 4096)]
    return buffers(meta)


def timed(seq, n):
    out = [[] for _ in seq]
    for r in range(n + 2):
        for i, (k, b) in enumerate(seq):
            res = DefaultNPURuntime.run(k, b)
            if r >= 2:
                out[i].append(res.npu_time / 1e6)
    return [float(np.median(v)) for v in out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("elf")
    p.add_argument("designs", nargs="+")
    p.add_argument("--n", type=int, default=15)
    args = p.parse_args()
    names = [d.split("=")[0] for d in args.designs]
    arts = [ART / d.split("=")[1] for d in args.designs]
    bufs = [bufs_for(a) for a in arts]
    ek, _ = elf_kernels(args.elf, names)
    xk = [load(a) for a in arts]
    for label, ks in (("separate xclbins", xk), ("one full ELF", [ek[n] for n in names])):
        alone = [timed([(k, b)], args.n)[0] for k, b in zip(ks, bufs)]
        alt = timed(list(zip(ks, bufs)), args.n)
        print(f"{label:16s} " + "  ".join(f"{n}: alone {a:6.2f} alt {t:6.2f} (+{t - a:4.2f})"
                                          for n, a, t in zip(names, alone, alt)), flush=True)


if __name__ == "__main__":
    main()
