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

Compiled AMD XDNA2 NPU programs for [redline-tiny](https://github.com/Cukrus-directive/redline-tiny), which runs Qwen-Image-2.1-Turbo photo editing and text-to-image on the Ryzen AI NPU of a Windows laptop. The installer downloads this repo and checks every file against `manifest.json`, so there's no need to download it by hand.

| | |
|---|---|
| App, installer, kernel sources | [GitHub: Cukrus-directive/redline-tiny](https://github.com/Cukrus-directive/redline-tiny) |
| Gallery, benchmarks, how it works | [Space: cukrus/redline-tiny](https://huggingface.co/spaces/cukrus/redline-tiny) |
| Model weights | Not in this repo. The installer gets the official [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) files and packs them for the NPU on your PC |

## What's in it

| Part of the model | Files | Count |
|---|---|---|
| Text encoder (Qwen3-VL-8B: vision tower and language model) | `vlm-*/` | 38 programs |
| DiT, step 1 of an edit | `merged/dit_step1_v7.elf` | 1 merged ELF |
| DiT, steps 2–8 | `merged/dit_cached_v7.elf` | 1 merged ELF |
| DiT, step 1 of text-to-image | `atb3*/`, `attn32*/`, `glue-*/`, `q-atb3k-t2imgin/` | 24 programs, attention once per prompt-length bucket |
| VAE decoder | `vae2-*/` | 64 programs |
| Checksums | `manifest.json` | size and SHA-256 of every file |

Each program folder holds:

| File | |
|---|---|
| `final.xclbin` | the program the AIE cores run |
| `insts.bin` | its instruction stream (data movement, run parameters) |
| `meta.json` | the design and arguments it was built from, so it can be rebuilt exactly |
| `final.prj/cdo_main/*.bin` | VAE programs only: the static array setup, used to check that every tensor shape of one kernel shares one program |

A merged ELF holds a whole DiT stage in one hardware context, so the kernels of a step run without context switches. Attention reads its valid-key count at runtime, so one ELF serves every prompt length.

## Build

| | |
|---|---|
| Toolchain | mlir-aie v1.4.4 (IRON), Peano / llvm-aie 22.0.0.2026092101 |
| Target | `npu2`: XDNA2 (Strix), 32 AIE cores |
| Tested on | Ryzen AI 9 HX 375 and Ryzen AI 7 350, NPU driver 32.0.20102 |
| Rebuilding | Sources in `kernels/`, steps in [docs/BUILDING.md](https://github.com/Cukrus-directive/redline-tiny/blob/main/docs/BUILDING.md) |

## Licence

Apache-2.0. Parts of the GEMM kernels are derived from AMD's [mlir-aie](https://github.com/Xilinx/mlir-aie) examples (Apache-2.0 WITH LLVM-exception). The model these programs run, Qwen-Image-2.1-Turbo, is under the Qwen Research License (non-commercial use only) and is not part of this repo.
