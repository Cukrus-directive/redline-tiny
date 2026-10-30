//===- gelu_bfp.cc - GELU between a GEMM's C stream and the next GEMM's A operand (AIE2P) -===//
//
// One call = one 16-row band of a config3 C tile (128 cols, bfp16 2x2 super-blocks of row-major 8x8) -> the same
// band of the two A tiles (cols 0-63, 64-127; a_layout order: 8 col blocks, then the band's two 8-row halves).
// GELU, tanh form: gelu(x) = x * sigmoid(2u), u = sqrt(2/pi) (x + 0.044715 x^3), with
// sigmoid(z) = 1 / (1 + 2^(-z log2 e)). AIE2P's exp2 is piecewise linear (up to 6% high), so its result goes
// through exp2_corr (exp2_fix.h, <=0.56%). The vision mergers' exact-erf GELU differs from this by ~1e-3.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "common/exp2_bf16.h"
#include "exp2_fix.h"

using vbf64 = aie::vector<bfloat16, 64>;
using vbf32 = aie::vector<bfloat16, 32>;
using vf32 = aie::vector<float, 32>;
using MMUL = aie::mmul<8, 8, 8, bfp16ebs8>;
using InStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, aie_dm_resource::a>;
using EyeStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b>;
using OutStream = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c>;

// t = -2 log2(e) u = x (A + B x^2)
constexpr float kA = -2.0f * 1.4426950409f * 0.7978845608f, kB = kA * 0.044715f;

alignas(aie::vector_decl_align) static uint8_t eye_bfp[72];
static bool eye_ready = false;

static inline aie::block_vector<bfp16ebs8, 64> eye_block() {
  if (!eye_ready) {
    for (int i = 0; i < 8; i++) {
      eye_bfp[i * 9] = 127;
      for (int j = 0; j < 8; j++)
        eye_bfp[i * 9 + 1 + j] = (i == j) ? 64 : 0;
    }
    eye_ready = true;
  }
  EyeStream es((const bfp16ebs8 *)eye_bfp);
  return es.pop();
}

static inline vbf64 decode(const aie::block_vector<bfp16ebs8, 64> &b, const aie::block_vector<bfp16ebs8, 64> &eye) {
  MMUL m;
  m.mul(b, aie::op_transpose(eye));
  return m.template to_vector<bfloat16>(); // exact: bfp16 mantissas are 7-bit
}

static inline aie::accum<accfloat, 32> gelu32(vbf32 x) {
  const vbf32 x2 = aie::mul(x, x).template to_vector<bfloat16>();
  aie::accum<accfloat, 32> pa;
  pa.from_vector(aie::broadcast<float, 32>(kA));
  const vbf32 poly = aie::mac(pa, x2, aie::broadcast<bfloat16, 32>(kB)).template to_vector<bfloat16>();
  vf32 t = aie::mul(x, poly).template to_vector<float>();
  t = aie::min(t, aie::broadcast<float, 32>(120.0f));
  const vf32 e = exp2_corr<32>(exp2_bf16(t)).template to_vector<float>();
  const vf32 den = aie::add(e, aie::broadcast<float, 32>(1.0f));
  aie::accum<accfloat, 32> ia;
  ia.from_vector(aie::inv(den));
  return aie::mul(x, ia.template to_vector<bfloat16>());
}

extern "C" {

void gelu_band(bfp16ebs8 *__restrict pIn, bfp16ebs8 *__restrict pOut) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto eye = eye_block();
  OutStream out(pOut);
  for (int ha = 0; ha < 2; ha++)      // A tile: C cols 64*ha .. 64*ha + 63
    for (int bca = 0; bca < 8; bca++) { // 8-col block of the A tile
      const int sbc = 4 * ha + (bca >> 1), bc = bca & 1;
      for (int br = 0; br < 2; br++) {  // the band's row half
        InStream bs(pIn);
        bs.seek((sbc * 2 + br) * 2 + bc);
        const vbf64 x = decode(bs.pop(), eye);
        out.push(aie::concat(gelu32(x.extract<32>(0)), gelu32(x.extract<32>(1))).template to_vector<bfp16ebs8>());
      }
    }
  event1();
}

} // extern "C"
