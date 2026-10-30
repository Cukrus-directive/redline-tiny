//===- swiglu_bfp.cc - SwiGLU between the gate|proj and down GEMMs (AIE2P) -===//
//
// In : one gate|proj GEMM output tile, 128x128 bfp16 in config3's C layout (2x2 super-blocks of row-major 8x8,
//      super-blocks row-major). Weight columns are interleaved per 128-col tile, so cols 0..63 are gate and
//      cols 64..127 are proj for the same 64 hidden units.
// Out: silu(gate) * proj as one down-GEMM A tile, 128x64 bfp16 in a_layout order (16-row bands, then 8-col
//      blocks, then the band's two 8x8 halves).
// bfp16 has no direct decode on AIE2P, so blocks are decoded through an 8x8x8 mmul against the identity (exact:
// bfp16 holds 7-bit mantissas, the fp32 accumulator holds them as-is).
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "common/activations.h"

using MMUL = aie::mmul<8, 8, 8, bfp16ebs8>;
template <aie_dm_resource R>
using InStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, R>;
using OutStream = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c>;

alignas(aie::vector_decl_align) static uint8_t eye_bfp[72]; // 8x8 identity: row i = [exp 127][64 at col i]
static bool eye_ready = false;

static inline aie::vector<float, 64> decode(const aie::block_vector<bfp16ebs8, 64> &x,
                                            const aie::block_vector<bfp16ebs8, 64> &eye) {
  MMUL m;
  m.mul(x, aie::op_transpose(eye));
  return m.template to_vector<float>();
}

extern "C" {

void swiglu_tile(bfp16ebs8 *__restrict pIn, bfp16ebs8 *__restrict pOut) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  if (!eye_ready) {
    for (int i = 0; i < 8; i++) {
      eye_bfp[i * 9] = 127;
      for (int j = 0; j < 8; j++)
        eye_bfp[i * 9 + 1 + j] = (i == j) ? 64 : 0;
    }
    eye_ready = true;
  }
  InStream<aie_dm_resource::b> es((const bfp16ebs8 *)eye_bfp);
  const auto eye = es.pop();
  OutStream out(pOut);
  for (int sbr = 0; sbr < 8; sbr++) {
    for (int cb = 0; cb < 8; cb++) {     // output 8-col block = gate/proj col block
      for (int br = 0; br < 2; br++) { // row half of the 16-row band
        const int k = br * 2 + (cb & 1), sg = cb >> 1;
        InStream<aie_dm_resource::a> gs(pIn);
        gs.seek((sbr * 8 + sg) * 4 + k);
        InStream<aie_dm_resource::a> ps(pIn);
        ps.seek((sbr * 8 + sg + 4) * 4 + k);
        const aie::vector<float, 64> g = decode(gs.pop(), eye);
        const aie::vector<float, 64> p = decode(ps.pop(), eye);
        const aie::accum<accfloat, 32> r0 = aie::mul(silu_vec<32>(g.extract<32>(0)), p.extract<32>(0));
        const aie::accum<accfloat, 32> r1 = aie::mul(silu_vec<32>(g.extract<32>(1)), p.extract<32>(1));
        aie::accum<accfloat, 64> r;
        r.from_vector(aie::concat(r0.to_vector<float>(), r1.to_vector<float>()));
        out.push(r.to_vector<bfp16ebs8>());
      }
    }
  }
  event1();
}

} // extern "C"
