---
license: apache-2.0
tags:
- amd
- ryzen-ai
- npu
- xdna2
- qwen-image
- text-to-image
- image-editing
base_model: Qwen/Qwen-Image-2.1-Turbo
---

# redline-tiny NPU programs

These are the compiled AMD XDNA2 (Strix, `npu2`) NPU programs used by [redline-tiny](https://github.com/Cukrus-directive/redline-tiny), which runs Qwen-Image-2.1-Turbo editing and text-to-image on the Ryzen AI NPU of a Windows laptop.

**You don't need to download this by hand:** the redline-tiny installer fetches it and checks every file against `manifest.json`.

## What's in it

- 126 program folders: the text encoder (Qwen3-VL-8B), text-to-image DiT step 1 and the VAE decoder. Each folder holds an `xclbin`, its instruction stream and `meta.json`, which records the design and arguments it was built from.
- 2 merged ELFs that hold whole DiT stages (`merged/dit_step1_v7.elf`, `merged/dit_cached_v7.elf`).
- `manifest.json`: the size and SHA-256 of every file.

**No model weights are included.** The programs are compute kernels only (GEMM, attention, norms, convolutions). The app downloads the official Qwen weights from [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) and packs them locally.

## Build

The programs were built with mlir-aie v1.4.4 (IRON) and Peano / llvm-aie 22.0.0.2026092101 for `npu2`. They were tested on a Ryzen AI 9 HX 375 with NPU driver 32.0.20102. Sources and build steps are in the GitHub repo (`kernels/`, `docs/BUILDING.md`).

## Licence

Apache-2.0. Parts of the GEMM kernels are derived from AMD's mlir-aie examples (Apache-2.0 WITH LLVM-exception). The model these programs run, Qwen-Image-2.1-Turbo, is under the Qwen Research License (non-commercial only) and is not part of this repo.
