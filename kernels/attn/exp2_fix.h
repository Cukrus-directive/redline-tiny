// Correction for AIE2P's exp2 instruction, shared by attn_bfp.cc (ATT_EXP2_FIX) and the exp2_probe unit test.
//
// AIE2P's exp2 is piecewise linear per octave: exp2(k + h) = 2^k (1 + h) for h in [0, 1), up to 6% high
// mid-octave (known-answer probe). Its result's own mantissa m is 1 + h, so the true value is
// y * c(m), c(m) = 2^(m - 1) / m ~ (C2 m + C1) m + C0: within 0.56% over every bf16 mantissa, bf16 steps included.
// c(1) must be exactly 1: alpha = exp2(0) = 1 whenever the row max doesn't move, and alpha compounds over every KV
// tile (a first fit had c(1) = 0.9961: 0.9961^126 ~ 0.6, and O (bf16) and l (fp32) drifted apart). Hence
// C0 = 1 - bf16(C2 + C1), so the bf16 steps land on exactly 1.0 at m = 1.
#pragma once

#include <aie_api/aie.hpp>

constexpr float kExpC2 = 0.24511719f, kExpC1 = -0.7285f, kExpC0 = 1.48242188f;

// The mantissa of y as a bf16 in [1, 2) (exponent field forced to 127).
template <unsigned N> static inline aie::vector<bfloat16, N> exp2_mant(aie::vector<bfloat16, N> y) {
  return aie::bit_or(aie::bit_and(y.template cast_to<int16>(), aie::broadcast<int16, N>(0x007F)),
                     aie::broadcast<int16, N>(0x3F80))
      .template cast_to<bfloat16>();
}

// c(m) for the mantissa m.
template <unsigned N> static inline aie::vector<bfloat16, N> exp2_cfac(aie::vector<bfloat16, N> m) {
  aie::accum<accfloat, N> a, b;
  a.from_vector(aie::broadcast<float, N>(kExpC1));
  b.from_vector(aie::broadcast<float, N>(kExpC0));
  const aie::vector<bfloat16, N> t =
      aie::mac(a, m, aie::broadcast<bfloat16, N>(kExpC2)).template to_vector<bfloat16>();
  return aie::mac(b, m, t).template to_vector<bfloat16>();
}

// The corrected exp2 as an accumulator, from the native result y.
template <unsigned N> static inline aie::accum<accfloat, N> exp2_corr(aie::vector<bfloat16, N> y) {
  return aie::mul(y, exp2_cfac<N>(exp2_mant<N>(y)));
}
