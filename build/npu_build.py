"""Compile an IRON design for npu2 without an NPU (e.g. on a Linux build box) and write a runnable artifact.

The artifact dir (/opt/npu-artifacts/<name>/) gets final.xclbin, insts.bin, meta.json and compile.log;
copy it to the NPU machine (or point REDLINE_NPU at the folder).

Usage (inside an IRON env):
  python3 npu_build.py --name atb3-qkv --design <path/to/n32_core.py> --family atb-bfp \
      --shape 4096 4096 12288 --tile 128 64 128 [--design-args ...]
The design must take the standard IRON compile args (-d, --xclbin-path, --insts-path) plus -M -K -N -m -k -n.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

# NPU_ART_ROOT picks the output root (default /opt/npu-artifacts on a Linux build box).
ART_ROOT = Path(os.environ.get("NPU_ART_ROOT", "/opt/npu-artifacts"))
REPO = Path(__file__).resolve().parent.parent   # meta.json stores designs relative to this, so it holds no local paths


def git_rev(path):
    r = subprocess.run(["git", "-C", str(path), "describe", "--tags", "--always", "--dirty"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--design", required=True, type=Path)
    p.add_argument("--family", required=True, help="host data layout family, see run_prebuilt.py")
    p.add_argument("--shape", nargs=3, type=int, metavar=("M", "K", "N"), help="GEMM designs only")
    p.add_argument("--tile", nargs=3, type=int, metavar=("m", "k", "n"), help="GEMM designs only")
    p.add_argument("--meta", nargs="*", default=[], metavar="KEY=INT", help="extra meta.json fields (non-GEMM)")
    p.add_argument("--design-args", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()
    design = args.design if args.design.is_absolute() or args.design.exists() else REPO / args.design
    design = design.resolve()
    try:
        design_rel = design.relative_to(REPO).as_posix()
    except ValueError:
        design_rel = design.name   # a design outside the repo: keep only its file name

    out = ART_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(design), "-d", "npu2",
           "--xclbin-path", str(out / "final.xclbin"), "--insts-path", str(out / "insts.bin")]
    shape = {}
    if args.shape:
        (M, K, N), (m, k, n) = args.shape, args.tile
        shape = dict(M=M, K=K, N=N, m=m, k=k, n=n)
        cmd += ["-M", str(M), "-K", str(K), "-N", str(N), "-m", str(m), "-k", str(k), "-n", str(n)]
    shape.update({kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.meta})
    cmd += args.design_args
    t = time.time()
    r = subprocess.run(cmd, cwd=design.parent, capture_output=True, text=True)
    (out / "compile.log").write_text(" ".join(cmd) + "\n\n" + r.stdout + "\n" + r.stderr)
    if r.returncode != 0 or not (out / "final.xclbin").exists():
        print(f"FAILED ({r.returncode}); see {out / 'compile.log'}")
        print(r.stderr[-3000:])
        return 1

    meta = {
        "name": args.name, "family": args.family, **shape,
        "design": design_rel, "design_args": args.design_args,
        "design_rev": git_rev(design.parent),
        "mlir_aie": version("mlir_aie"), "peano": version("llvm-aie"),
        "compile_s": round(time.time() - t, 1), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"OK {out}  ({meta['compile_s']} s, mlir_aie {meta['mlir_aie']}, peano {meta['peano']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
