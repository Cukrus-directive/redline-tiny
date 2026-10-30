//===- vconv.cc - VAE decoder conv (3x3 or 1x1, stride 1) as an implicit GEMM on AIE2P, v2 ----------------===//
//
// Activations are bf16 "octet-planar" with a 1-pixel zero border: [row][C/8][W+2][8] (8 channels of a pixel
// contiguous, one plane per channel octet). A conv strip (one input row, P+2 pixels, KC channels) is then KC/8
// contiguous runs, already in the [c8][px][8] order this kernel reads.
//
// One output tile = P pixels of one output row x NC output channels; the K loop arrives as TAPS (dy) x Cin/KC
// steps of (strip, weights):
//   strip : (P+2) x KC bf16, [c8][px][8]
//   w     : TAPS(dx) x KC x NC bfp16 8x8 blocks [jp][dx][c8][2][64] (rows = 8 outputs, cols = 8 inputs: B of A.B^T)
// After the steps, one more weight-stream element whose first NC bf16 are the biases (a core has 2 input DMA
// channels: strip + weights). bfp16 blocks are 9 bytes, so a 1-pixel tap shift isn't possible by DMA in bfp16;
// in bf16 it is a 16-byte offset: conv_step builds the dx-shifted bfp16 copies in-core, then runs config3-style
// 2x2 mmul groups over the (dx, c8) k-steps with fp32 partials in L1 (bf16 staging over 18-72 steps costs ~2%).
// vconv_out adds the biases and writes the tile octet-planar, [j][px][8] (each 8x8 block is 64 contiguous bf16).
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef VC_P
#define VC_P 64
#endif
#ifndef VC_KC
#define VC_KC 48
#endif
#ifndef VC_NC
#define VC_NC 48
#endif
#ifndef VC_TAPS
#define VC_TAPS 3
#endif

constexpr int P = VC_P, KC = VC_KC, NC = VC_NC, TAPS = VC_TAPS;
constexpr int PW = P + 2;                // strip width (1-pixel halo each side; 1x1 convs use the centre)
constexpr int DX0 = TAPS == 3 ? 0 : 1;   // first horizontal tap offset into the strip
constexpr int C8 = KC / 8;
constexpr int PGP = P / 16;              // pairs of 8-pixel groups
constexpr int JP = NC / 16;              // pairs of 8-channel output groups
constexpr int KSTEPS = TAPS * C8;        // (dx, c8) k-steps per step
static_assert(P % 16 == 0 && NC % 16 == 0 && KC % 8 == 0 && (TAPS == 1 || TAPS == 3), "tile sizes");

using MMUL = aie::mmul<8, 8, 8, bfp16ebs8>;
template <aie_dm_resource R> using InS = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, R>;
using OutS = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c>;
using facc = aie::accum<accfloat, 64>;

alignas(aie::vector_decl_align) static float acc_all[P * NC];               // [pgp][jp][4][64] fp32
alignas(aie::vector_decl_align) static uint8_t sh[PGP * KSTEPS * 2 * 72];  // [pgp][dx][c8][2][64] bfp16

// fp32 <-> accumulator in 32-lane halves (64-lane fp32 vectors miscompile under some schedules)
static inline facc load_acc(const float *p) {
  aie::accum<accfloat, 32> lo, hi;
  lo.from_vector(aie::load_v<32>(p));
  hi.from_vector(aie::load_v<32>(p + 32));
  return aie::concat(lo, hi);
}

static inline void store_acc(float *p, const facc &a) {
  aie::store_v(p, a.extract<32>(0).to_vector<float>());
  aie::store_v(p + 32, a.extract<32>(1).to_vector<float>());
}

extern "C" {

void vconv_zero() {
  const aie::vector<float, 32> z = aie::zeros<float, 32>();
  for (int i = 0; i < P * NC; i += 32)
    aie::store_v(acc_all + i, z);
}

void vconv_step(bfloat16 *__restrict strip, bfp16ebs8 *__restrict w) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  OutS out((bfp16ebs8 *)sh);
  for (int pgp = 0; pgp < PGP; pgp++)
    for (int dx = 0; dx < TAPS; dx++)
      for (int c8 = 0; c8 < C8; c8++)
        for (int h = 0; h < 2; h++) {
          const bfloat16 *src = strip + (c8 * PW + (pgp * 2 + h) * 8 + dx + DX0) * 8;
          facc a;
          a.from_vector(aie::load_unaligned_v<64>(src, 8));
          out.push(a.to_vector<bfp16ebs8>());
        }
  for (int pgp = 0; pgp < PGP; pgp++)
    for (int jp = 0; jp < JP; jp++) {
      float *pc = acc_all + (pgp * JP + jp) * 256;
      MMUL a0(load_acc(pc)), a1(load_acc(pc + 64)), a2(load_acc(pc + 128)), a3(load_acc(pc + 192));
      InS<aie_dm_resource::a> as((const bfp16ebs8 *)sh);
      as.seek(pgp * KSTEPS * 2);
      InS<aie_dm_resource::b> bs(w);
      bs.seek(jp * KSTEPS * 2);
      for (int k = 0; k < KSTEPS; k++) {
        const auto B0 = bs.pop(), B1 = bs.pop();
        const auto A0 = as.pop(), A1 = as.pop();
        a0.mac(A0, aie::op_transpose(B0));
        a1.mac(A0, aie::op_transpose(B1));
        a2.mac(A1, aie::op_transpose(B0));
        a3.mac(A1, aie::op_transpose(B1));
      }
      store_acc(pc, a0.to_accum());
      store_acc(pc + 64, a1.to_accum());
      store_acc(pc + 128, a2.to_accum());
      store_acc(pc + 192, a3.to_accum());
    }
  event1();
}

#ifdef VC_DS
// Stride-2 conv (encoder downsample: ZeroPad(0,1,0,1) + 3x3 stride 2) = the stride-1 conv at odd pixels: keep
// pixels 1, 3, 5, 7 of each 8-pixel block. out: P/2 x NC bf16, [j][px][8].
void vconv_out_ds(bfp16ebs8 *__restrict belem, bfloat16 *__restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *bias = (const bfloat16 *)belem;
  for (int pgp = 0; pgp < PGP; pgp++)
    for (int jp = 0; jp < JP; jp++)
      for (int q = 0; q < 4; q++) {
        const int pg = pgp * 2 + q / 2, j = jp * 2 + q % 2;
        const aie::vector<bfloat16, 8> b8 = aie::load_v<8>(bias + j * 8);
        facc bb;
        bb.from_vector(aie::concat(b8, b8, b8, b8, b8, b8, b8, b8));
        const aie::vector<bfloat16, 64> v =
            aie::add(load_acc(acc_all + (pgp * JP + jp) * 256 + q * 64), bb).to_vector<bfloat16>();
        aie::store_v(out + (j * (P / 2) + pg * 4) * 8, aie::filter_odd(v, 8));
      }
}
#endif

// belem: a weight-stream element whose first NC bf16 are the biases; out: P x NC bf16, octet-planar [j][px][8]
void vconv_out(bfp16ebs8 *__restrict belem, bfloat16 *__restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *bias = (const bfloat16 *)belem;
  for (int pgp = 0; pgp < PGP; pgp++)
    for (int jp = 0; jp < JP; jp++)
      for (int q = 0; q < 4; q++) {  // a0 (pg even, j even), a1 (pg even, j odd), a2, a3
        const int pg = pgp * 2 + q / 2, j = jp * 2 + q % 2;
        const aie::vector<bfloat16, 8> b8 = aie::load_v<8>(bias + j * 8);
        facc bb;
        bb.from_vector(aie::concat(b8, b8, b8, b8, b8, b8, b8, b8));  // every pixel row gets the same 8 biases
        aie::store_v(out + (j * P + pg * 8) * 8,
                     aie::add(load_acc(acc_all + (pgp * JP + jp) * 256 + q * 64), bb).to_vector<bfloat16>());
      }
}

} // extern "C"
