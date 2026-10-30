# How redline-tiny works

This page covers what runs where, why, and what we learned getting Qwen-Image-2.1-Turbo onto a laptop NPU. Every number here was measured on a Ryzen AI 9 HX 375 (XDNA2 "Strix" NPU: 8 columns × 4 rows = 32 AIE-ML cores, Radeon 890M iGPU, 32 GB RAM).

![pipeline](flow.svg)

## The pipeline

| Stage | Model | Where | Warm time |
|---|---|---|---|
| Photo prep + VAE encoder (edits only) | Qwen-Image VAE encoder, fp16 | iGPU, ONNX Runtime + DirectML | 0.6–0.8 s, hidden behind the text encoder |
| Text encoder | Qwen3-VL-8B: vision tower (27 blocks) + language model (36 layers) | NPU | ~3 s text-to-image, 6–9 s edit |
| DiT, 8 steps | Qwen-Image-2.1-Turbo transformer, 32 blocks, hidden 4096 | NPU | ~25 s text-to-image, ~30 s edit |
| VAE decoder | Qwen-Image VAE decoder | NPU | 4.5 s alone, 6–7 s in the server |
| Glue | timestep embedding, modulation vectors, the Euler scheduler, a few tiny GEMMs, PNG | CPU | small |

An edit's step 1 attends over the prompt, the reference photo's latents and the target: 8704 rows, laid out as `[prefix 4128 | pad | target 4096 | pad]`. Steps 2–8 reuse the reference keys and values from step 1, so they process only the 4096 target rows. Text-to-image with up to 512 prompt tokens uses a smaller 4608-row step 1.

## The NPU kernels

**Weights are bfp16** ("bfp16ebs8"): blocks of 8 values share one 8-bit exponent and keep 8-bit mantissas, so 8 values take 9 bytes. That's the NPU's native fast matmul format. The packer stores every weight matrix in exactly the order the cores read it, so loading is a straight copy. We checked whether a more compressed format would be faster: no. Dense int8 runs at the same rate as bfp16, and decoding 4-bit weights in-core costs more time than the bandwidth it saves.

**GEMM.** The matrix multiplies use mlir-aie's asymmetric-tile-buffered bfp16 GEMM (v1.4.4's Peano rewrite, "config3"). We made it work for any K and any M and gave it fused epilogues. It runs on all 32 cores at about **33.5 TFLOPS** on Qwen's shapes. On the way we found that upstream's host-side helper reads output tiles in the wrong order. That was a host bug, not a kernel one; our layout code fixes it, and an upstream issue is being filed.

**Attention** is our own flash-attention kernel for head_dim 128. It does online softmax on the cores' native `exp2`, converts P to bfp16 in-core, and broadcasts K/V across 32 cores. It runs at about **19.5 TFLOPS**. One finding: **AIE2P's native `exp2` is piecewise linear**. It computes exactly 2^k·(1+h) for x = k + h, up to 6% high. That's harmless on random data, but on real, peaky attention it was half of the remaining error. We measured it with a known-answer probe. A cheap correction with c(1) = 1 exactly brings attention to 0.9994–0.9995 correlation with the reference, at +26% attention time.

**Glue on the NPU.** Running LayerNorm with modulation, q/k RMSNorm with RoPE, the attention packing, SwiGLU and the residuals on the CPU cost about 36 s per edit. As fused NPU kernels it's well under a second. SwiGLU runs in the up-projection GEMM's epilogue, with gate and proj columns interleaved per 128-column tile so that one core holds both halves of each output. This is the main reason our DiT is faster than the CPU-glue approach.

**Packing must round like the hardware.** The reference host packer in mlir-aie's examples truncates mantissas, but the NPU rounds half to even, so every host-packed weight carried a −0.5 LSB bias. Our packer is byte-identical to what the NPU itself produces.

## Fitting it on a laptop

**Programs and contexts.** Strix has 16 hardware contexts, and switching between full-array programs costs about 2.2 ms. An edit makes about 1800 NPU calls, so each DiT stage (step 1, cached steps) is **merged into one ELF**: one context, no switches inside the stage. Attention needs the number of valid keys, which depends on the prompt length. Rather than compiling one program per length, the count is **patched into the instruction stream at runtime**, so one program serves every prompt. That idea came from studying VigenFlow; we reimplemented it ourselves. Prompt-length buckets for the text encoder (64 tokens each) and for text-to-image are all prebuilt, so nothing compiles on your PC.

**The VAE decoder** first needed 44 shape-specific programs and lost 1.1–1.9 s per image to context switching. Now each core reads its trip counts from runtime parameters, so a program depends only on the kernel configuration. That brings it down to **20 shape-generic programs**, with a check that every shape's static configuration is byte-identical to its canonical program. This idea also came from VigenFlow, which uses 4. We tried merging the whole decoder into one ELF: it was faster alone (3.4 s), but it pins ~2.9 GB and evicted the text encoder's weights from the file cache, so overall it lost.

**Pinned RAM is the real limit.** NPU buffers are pinned host RAM. The DiT alone peaks at 12.1 GB and the text encoder at 9.4 GB, so on a 32 GB laptop they can't both stay on the device. Only the DiT stays resident. The text encoder keeps its weights hot in normal RAM and streams the language model's layers in one by one, each overlapped with the previous layer's compute. Everything else rebuilds its device buffers per job.

**Keep torch out of the NPU process.** Importing torch into the process that drives the NPU slowed every image by about 5 s (VAE decode 6.4 → 9.6 s). The host-side numpy glue gets slower once torch's threading runtime is loaded. redline-tiny counts prompt tokens with the bare `tokenizers` library, and torch only runs in helper processes.

**The iGPU helps.** The VAE encoder runs in fp16 on the 890M through DirectML (0.6 s), at the same time as the NPU text encoder, so it's free. It's also more accurate than our NPU encoder. The fp16 *decoder* hangs the 890M (a Windows GPU timeout), so decoding stays on the NPU.

## Accuracy

The NPU pipeline was validated block by block against one CPU reference run of the official diffusers pipeline:
- DiT kernels against the reference block activations: 0.9994–0.99995 correlation.
- Text-encoder rows: mean 0.983, against a float64 ceiling of 0.989.
- VAE decode: 50.7 dB PSNR against the CPU reference.

Results are deterministic: the same inputs give the same bytes, run to run and across this port. From the same starting noise, our edit and VigenFlow's agree at 23.5 dB: the same edit, with small differences in texture.

## Credits

- [mlir-aie / IRON](https://github.com/Xilinx/mlir-aie) and Peano (llvm-aie) by AMD: the compiler stack and the GEMM this is built on.
- [VigenFlow](https://github.com/jia1217/vigenflow): the first Qwen-Image port to this NPU. It showed the route, and we took two ideas from it (both described above).
- The Qwen team for [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (Qwen Research License, non-commercial).
