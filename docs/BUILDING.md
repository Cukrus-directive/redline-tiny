# Building the NPU programs

Users never compile anything: the installer downloads the prebuilt programs from Hugging Face. This page is for developers who change a kernel or want to rebuild the set.

## What you need

- **mlir-aie v1.4.4** (IRON) and **Peano / llvm-aie 22.0.0.2026092101**. The shipped programs were built with exactly these. Install them the way mlir-aie's README describes: `utils/iron_setup.py`, or the wheels from its v1.4.4 release.
- `MLIR_AIE_SRC` set to the mlir-aie source checkout: the kernel designs include its `aie_kernels` headers. `build/env.sh` sets it on Linux.
- Compiling needs **no NPU**. Any Linux x86-64 box with the toolchain works (we use an Ubuntu 24.04 LXC), and so does the Windows laptop itself. The programs then run on any XDNA2 (npu2) machine.

## Building one program

`build/npu_build.py` compiles one IRON design into a program folder (`final.xclbin`, `insts.bin`, `meta.json`, `compile.log`, and for VAE programs `final.prj/cdo_main/*.bin`) under `$NPU_ART_ROOT/<name>/`:

```
python build/npu_build.py --name glue-ln-m4096-s0 --design kernels/glue/ln_design.py --family glue-ln --meta M=4096 S=0 --design-args --m 4096 --s 0
```

GEMMs take `--shape M K N --tile m k n`. Every shipped program's `meta.json` records the design and arguments it was built from, so it can be rebuilt exactly.

## Building on demand while developing

Inside a developer checkout, set `REDLINE_ALLOW_BUILD=1` with an IRON env active, and point `REDLINE_NPU` at your build folder. The runtime then compiles any program it's missing the first time it's needed. These are prompt-length buckets, VAE shapes and attention variants. In a release, a missing program is an error that tells the user to re-run the installer.

## Making a release set

1. Build or refresh the programs. `tools/programs.json` lists the 126 programs and 2 merged DiT ELFs the app can reach, including every prompt-length bucket:
   - text-encoder attention `vlm-attn-m1536-t1…t18`;
   - text-to-image attention `…-tr72/pf72-nk4160…4608`.
2. Stage them:
   ```
   python tools/stage_npu.py <your NPU_ART_ROOT> <out-dir>
   ```
   This copies only the files the runtime reads (about 310 MB) and writes `manifest.json` with a SHA-256 per file. The installer checks every file against it.
3. Upload `<out-dir>` to the Hugging Face repo named in `scripts/release.json` (`npu_repo`). The model card is in [hf-npu/README.md](../hf-npu/README.md).

The merged DiT ELFs (`merged/dit_cached_v7.elf`, `merged/dit_step1_v7.elf`) are built by merging the stage programs listed in `redline/forward.py` (`ELF_STAGES_RT`). The step-1 ELF's reference attention is compiled for at most 4160 prefix keys. That's where the 64-token edit prompt limit comes from, and raising it means rebuilding that ELF with a larger key budget.
