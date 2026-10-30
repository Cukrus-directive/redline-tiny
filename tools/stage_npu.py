"""Collect the compiled NPU programs a release needs into one folder, ready to upload to the Hugging Face repo that
the installer downloads (scripts/release.json: npu_repo).

  python tools\\stage_npu.py SOURCE_ARTIFACTS_DIR OUT_DIR

SOURCE_ARTIFACTS_DIR is a build tree (the developer's NPU_ART_ROOT). Only the files the runtime reads are copied:
final.xclbin, insts.bin, meta.json and final.prj/cdo_main/*.bin per program (tools/programs.json), plus the merged
DiT ELFs. Writes OUT_DIR/manifest.json with a SHA-256 per file; the installer checks every file against it.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_meta(f, dst):
    """meta.json with the design path made repo-relative (kernels/...): build machines' local paths stay private."""
    m = json.loads(f.read_text())
    if m.get("design"):
        p = m["design"].replace("\\", "/")
        m["design"] = "kernels/" + p.split("/kernels/", 1)[1] if "/kernels/" in p else p.rsplit("/", 1)[-1]
    dst.write_text(json.dumps(m, indent=2))


def main(src, out):
    src, out = Path(src), Path(out)
    spec = json.loads((HERE / "programs.json").read_text())
    files = []
    for name in spec["programs"]:
        d = src / name
        picks = [d / f for f in ("final.xclbin", "insts.bin", "meta.json") if (d / f).exists()]
        picks += sorted((d / "final.prj" / "cdo_main").glob("*.bin"))
        if not any(p.name in ("final.xclbin", "insts.bin") for p in picks):
            sys.exit(f"{name}: no final.xclbin / insts.bin in {d}")
        files += picks
    files += [src / e for e in spec["elfs"]]
    manifest, total = {}, 0
    for f in files:
        rel = f.relative_to(src).as_posix()
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.name == "meta.json":
            copy_meta(f, dst)
        else:
            shutil.copyfile(f, dst)
        manifest[rel] = {"sha256": sha256(dst), "size": dst.stat().st_size}
        total += dst.stat().st_size
    (out / "manifest.json").write_text(json.dumps({"format": 1, "files": manifest}, indent=1))
    print(f"{len(manifest)} files, {total / 2**20:.0f} MB -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
