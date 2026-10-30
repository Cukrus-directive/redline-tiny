// VLM copy of kernels/atb3/mm_bfp_swiglu.cc with an accurate SiLU (silu_x).
//===- mm_bfp_swiglu.cc -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modified for redline-tiny (2026): derived from mlir-aie's block_datatypes GEMM examples (any-K / any-M runtime sequences, fused epilogues); see NOTICE.
//
//===----------------------------------------------------------------------===//

// Qwen port: mm_bfp_mixed.cc (config3 ATB microkernel) with SwiGLU in the flush. The up GEMM's weight columns are
// interleaved per 128-col tile as [gate 64 | proj 64], so a 128x128 C tile holds gate and proj for the same 64
// hidden units; the flush writes silu(gate) * proj as the down GEMM's 128x64 A tile (a_layout: 16-row bands, then
// 8-col blocks, then the band's two 8x8 halves), bfp16. gate/proj never round to bfp16 on the way.
//
// C accumulates in bf16 in 4 stripes of 32 rows (as in mm_bfp_mixed.cc). Stripe 0 lives in the output buffer
// (its 8 KB fits in the 9 KB A tile), stripes 1..3 in c_stage. Each stripe is converted at its own final call
// (stripes finish in order 0..3): stripe g's two A bands go to bytes [2304 g, 2304 (g + 1)) of the output, which
// for g >= 1 is stripe 0's already-consumed bf16. Stripe 0's own bands would overwrite its input, so they go
// through a_tmp and are copied in place after.

#include <aie_api/aie.hpp>

#include "common/activations.h"
#include "common/exp2_bf16.h"
#include "exp2_fix.h"

// silu(g) = g / (1 + 2^(-g log2 e)), exp2 corrected by exp2_corr (AIE2P's exp2 is piecewise linear). Replaces
// activations.h's silu_vec, whose native vtanh returns its argument for |x| <= 0.5 (sigmoid ~2.6% off at |g| ~ 1).
static inline aie::accum<accfloat, 32> silu_x(aie::vector<bfloat16, 32> g) {
  constexpr float kHi = -1.4453125f, kLo = -1.4426950409f + 1.4453125f;  // -log2(e) = hi + lo, hi exact in bf16
  aie::accum<accfloat, 32> ta = aie::mul(g, aie::broadcast<bfloat16, 32>(kHi));
  ta = aie::mac(ta, g, aie::broadcast<bfloat16, 32>(kLo));
  aie::vector<float, 32> t = aie::min(ta.to_vector<float>(), aie::broadcast<float, 32>(120.0f));
  const aie::vector<float, 32> e = exp2_corr<32>(exp2_bf16(t)).template to_vector<float>();
  aie::accum<accfloat, 32> ia;
  ia.from_vector(aie::inv(aie::add(e, aie::broadcast<float, 32>(1.0f))));
  return aie::mul(g, ia.template to_vector<bfloat16>());
}

template <typename T, unsigned N, aie_dm_resource R>
using InBufStream = aie::block_vector_restrict_input_buffer_stream<T, N, R>;
template <typename T, unsigned N, aie_dm_resource R>
using OutBufStream = aie::block_vector_output_buffer_stream<T, N, R>;

#ifndef K_PROBLEMSIZE
#define K_PROBLEMSIZE 4096
#endif
constexpr int K_Problemsize = K_PROBLEMSIZE;
static_assert(K_Problemsize % 64 == 0, "K must be a multiple of k=64");
constexpr int rho = 4;
constexpr int m = 128 / rho;
constexpr int k = 64;
constexpr int n = 128;
constexpr int r = 8;
constexpr int s = 8;
constexpr int t = 8;
constexpr int band_bytes = 16 * 72; // one 16-row band of a 128x64 A tile: 16 blocks of 8x8 bfp16
static_assert(m * rho == 128 && k == 64 && n == 128 && rho == 4);

alignas(aie::vector_decl_align) bfloat16 c_stage[(rho - 1) * m * n];
alignas(aie::vector_decl_align) uint8_t a_tmp[2 * band_bytes];

extern "C" {

static int g_counter = 0;
static int k_counter = 0;

static inline bfloat16 *stripe_ptr(bfp16ebs8 *pOut, int g) {
  return g == 0 ? (bfloat16 *)pOut : c_stage + (g - 1) * (m * n);
}

// silu(gate) * proj for one 32-row stripe -> its 2 A bands (32 blocks of 8x8 bfp16, pushed in a_layout order).
__attribute__((noinline)) static void swiglu_stripe(const bfloat16 *__restrict pS, bfp16ebs8 *__restrict pA) {
  OutBufStream<bfp16ebs8, r * t, aie_dm_resource::c> out(pA);
  for (int br = 0; br < 2; br++)
    for (int cb = 0; cb < 8; cb++)
      for (int bis = 0; bis < 2; bis++) {
        const int off = br * 2048 + (cb >> 1) * 256 + (bis * 2 + (cb & 1)) * 64;
        const aie::vector<bfloat16, 64> g = aie::load_v<64>(pS + off);
        const aie::vector<bfloat16, 64> p = aie::load_v<64>(pS + off + 4 * 256); // proj: 64 cols to the right
        const aie::accum<accfloat, 32> q0 = silu_x(g.extract<32>(0)), q1 = silu_x(g.extract<32>(1));
        const aie::accum<accfloat, 32> r0 = aie::mul(q0.to_vector<bfloat16>(), p.extract<32>(0));
        const aie::accum<accfloat, 32> r1 = aie::mul(q1.to_vector<bfloat16>(), p.extract<32>(1));
        out.push(aie::concat(r0, r1).template to_vector<bfp16ebs8>());
      }
}

void matmul_swiglu_bfp16(bfp16ebs8 *__restrict pA_in, bfp16ebs8 *__restrict pB_in,
                         bfp16ebs8 *__restrict pOut) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  using AStream = InBufStream<bfp16ebs8, r * s, aie_dm_resource::a>;
  using BStream = InBufStream<bfp16ebs8, s * t, aie_dm_resource::b>;
  using CPtr = bfloat16 __aie_dm_resource_c *__restrict;
  using MMUL = aie::mmul<r, s, t, bfp16ebs8>;

  const bfp16ebs8 *pA_base = (const bfp16ebs8 *)pA_in;
  const int g_this = g_counter;
  g_counter = (g_counter == rho - 1) ? 0 : g_counter + 1;
  CPtr const pC_base = (CPtr)stripe_ptr(pOut, g_this);

  for (int br = 0; br < m / (2 * r); br++) {
    BStream b_stream(pB_in);
    {
      CPtr pC = pC_base + br * (n / (2 * t)) * (4 * r * t);
      aie::vector<bfloat16, r * t> c0 = aie::load_v<r * t>(pC);
      aie::vector<bfloat16, r * t> c1 = aie::load_v<r * t>(pC + r * t);
      for (int blk = 0; blk < n / (2 * t); blk++) {
        CPtr pCb = pC + blk * (4 * r * t);
        CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 4 * r * t : pCb;
        MMUL acc0((aie::accum<accfloat, r * t>(c0)));
        MMUL acc1((aie::accum<accfloat, r * t>(c1)));
        MMUL acc2((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + 2 * r * t))));
        MMUL acc3((aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + 3 * r * t))));
        c0 = aie::load_v<r * t>(pNext);
        c1 = aie::load_v<r * t>(pNext + r * t);
        AStream a_stream(pA_base);
        a_stream.seek(br * (2 * k / s));
        auto kstep = [&]<bool BFILL>() __attribute__((always_inline)) {
          aie::block_vector<bfp16ebs8, r * s> A0, A1;
          aie::block_vector<bfp16ebs8, s * t> B0, B1;
          if constexpr (BFILL) {
            B0 = b_stream.pop();
            B1 = b_stream.pop();
            A0 = a_stream.pop();
            A1 = a_stream.pop();
          } else {
            A0 = a_stream.pop();
            A1 = a_stream.pop();
            B0 = b_stream.pop();
            B1 = b_stream.pop();
          }
          acc0.mac(A0, aie::op_transpose(B0));
          acc1.mac(A0, aie::op_transpose(B1));
          acc2.mac(A1, aie::op_transpose(B0));
          acc3.mac(A1, aie::op_transpose(B1));
        };
        kstep.operator()<true>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<true>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        aie::store_v(pCb, acc0.template to_vector<bfloat16>());
        aie::store_v(pCb + r * t, acc1.template to_vector<bfloat16>());
        aie::store_v(pCb + 2 * r * t, acc2.template to_vector<bfloat16>());
        aie::store_v(pCb + 3 * r * t, acc3.template to_vector<bfloat16>());
      }
    }
  }

  constexpr int total = (K_Problemsize / k) * rho;
  if (k_counter >= total - rho) { // this stripe's last k-block: SwiGLU it into its A bands
    uint8_t *pOutB = (uint8_t *)pOut;
    if (g_this == 0) {
      swiglu_stripe((const bfloat16 *)pC_base, (bfp16ebs8 *)a_tmp);
      for (int i = 0; i < 2 * band_bytes / 64; i++)
        aie::store_v(pOutB + i * 64, aie::load_v<64>(a_tmp + i * 64));
    } else {
      swiglu_stripe((const bfloat16 *)pC_base, (bfp16ebs8 *)(pOutB + g_this * 2 * band_bytes));
    }
  }
  k_counter = (k_counter == total - 1) ? 0 : k_counter + 1;
  event1();
}

void zero_swiglu(bfp16ebs8 *__restrict cOut) {
  const aie::vector<bfloat16, 64> zeros = aie::zeros<bfloat16, 64>();
  bfloat16 __aie_dm_resource_c *__restrict p0 = (bfloat16 __aie_dm_resource_c *)cOut;
  for (int i = 0; i < m * n / 64; i++)
    aie::store_v(p0 + i * 64, zeros);
  for (int i = 0; i < (rho - 1) * m * n / 64; i++)
    aie::store_v(c_stage + i * 64, zeros);
}
}
