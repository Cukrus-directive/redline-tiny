//===- qkv_post.cc - QKV GEMM output -> attention inputs (AIE2P) -----------===//
//
// Works on half-tiles: 64 tokens x 128 dims of one head, i.e. the first or second 9216 bytes of a config3 C tile
// (bfp16, 2x2 super-blocks of row-major 8x8; a half holds super-block rows 0..3).
//   q_post: per-row RMSNorm over the 128 dims (eps 1e-6) * w_q (w_q pre-multiplied on the host by
//           log2(e)/sqrt(128), the attention kernel's score scale), complex RoPE on adjacent dim pairs, written
//           as one attention Q tile (a_layout(64, 128)).
//   k_post: the same with w_k and no score scale, written as one attention K tile (b_layout(K^T, 128, 64):
//           row-major 8x8 blocks of K, ordered key band, dim block, key half).
//   v_post: no norm or RoPE, written as one attention V tile (b_layout(V, 64, 128): transposed 8x8 blocks,
//           ordered dim band, key block, dim half).
// Table (per half-tile, one buffer): for each input 8x8 block id blk = (sbr*8 + sbc)*4 + k, 32 cos then 32 sin
// (bf16, lane r*4 + p = token row r, pair p of the block), then w_q[128], w_k[128] (bf16).
// bfp16 is decoded through an 8x8x8 mmul against the identity (exact).
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using MMUL = aie::mmul<8, 8, 8, bfp16ebs8>;
template <aie_dm_resource R>
using InStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, R>;
using OutStream = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c>;
using vbf64 = aie::vector<bfloat16, 64>;
using vbf32 = aie::vector<bfloat16, 32>;
using vbf16 = aie::vector<bfloat16, 16>;

constexpr int TAB_BLOCK = 64;              // bf16 per block in the table (32 cos + 32 sin)
constexpr int TAB_W = 128 * TAB_BLOCK;     // offset of w_q in the table (128 blocks per half-tile)
constexpr float RMS_EPS = 1e-6f;

alignas(aie::vector_decl_align) static uint8_t eye_bfp[72];
static bool eye_ready = false;
alignas(aie::vector_decl_align) static bfloat16 band_buf[32 * 64]; // one decoded 16-row band (32 blocks)

static inline aie::block_vector<bfp16ebs8, 64> eye_block() {
  if (!eye_ready) {
    for (int i = 0; i < 8; i++) {
      eye_bfp[i * 9] = 127;
      for (int j = 0; j < 8; j++)
        eye_bfp[i * 9 + 1 + j] = (i == j) ? 64 : 0;
    }
    eye_ready = true;
  }
  InStream<aie_dm_resource::b> es((const bfp16ebs8 *)eye_bfp);
  return es.pop();
}

static inline vbf64 decode_bf16(const aie::block_vector<bfp16ebs8, 64> &x, const aie::block_vector<bfp16ebs8, 64> &eye) {
  MMUL m;
  m.mul(x, aie::op_transpose(eye));
  return m.template to_vector<bfloat16>(); // exact: bfp16 mantissas are 7-bit
}

static inline void push_bfp(OutStream &out, vbf64 v) {
  aie::accum<accfloat, 64> a;
  a.from_vector(v);
  out.push(a.to_vector<bfp16ebs8>());
}

// 8x8 row-major block whose row i is filled with v[half*8 + i].
static inline vbf64 row_bcast(vbf16 v, int half) {
  aie::vector<bfloat16, 8> e = v.extract<8>(half);
  return aie::transpose(aie::concat(e, e, e, e, e, e, e, e), 8, 8);
}

// 8x8 row-major block whose column j is filled with w[j] (8 weights).
static inline vbf64 col_bcast(const bfloat16 *w) {
  aie::vector<bfloat16, 8> e = aie::load_v<8>(w);
  return aie::concat(e, e, e, e, e, e, e, e);
}

// Row sums of two 8x8 fp32-accumulated squares blocks -> lanes 0..7 rows of s0, 8..15 rows of s1.
static inline aie::vector<float, 16> rowsum2(aie::accum<accfloat, 64> s0, aie::accum<accfloat, 64> s1) {
  auto fold = [](aie::accum<accfloat, 64> s) {
    vbf64 tr = aie::transpose(s.to_vector<bfloat16>(), 8, 8); // lane c*8 + row
    aie::accum<accfloat, 32> h;
    h.from_vector(tr.extract<32>(0));
    h = aie::add(h, tr.extract<32>(1));
    aie::vector<float, 32> hv = h.to_vector<float>();
    return aie::add(hv.extract<16>(0), hv.extract<16>(1)); // lane (c%2)*8 + row
  };
  aie::vector<float, 16> q0 = fold(s0), q1 = fold(s1);
  return aie::add(aie::concat(q0.extract<8>(0), q1.extract<8>(0)), aie::concat(q0.extract<8>(1), q1.extract<8>(1)));
}

// Decode one 16-row band into `band`, return its per-row 1/rms as two row-broadcast blocks.
static inline void load_band(const bfp16ebs8 *pIn, int sbr, const aie::block_vector<bfp16ebs8, 64> &eye,
                             vbf64 &ib0, vbf64 &ib1) {
  InStream<aie_dm_resource::a> is(pIn);
  is.seek(sbr * 32);
  aie::accum<accfloat, 64> ss0 = aie::zeros<accfloat, 64>(), ss1 = aie::zeros<accfloat, 64>();
  for (int i = 0; i < 32; i++) { // i = sbc*4 + k
    const vbf64 x = decode_bf16(is.pop(), eye);
    aie::store_v(band_buf + i * 64, x);
    if ((i & 2) == 0)
      ss0 = aie::mac(ss0, x, x);
    else
      ss1 = aie::mac(ss1, x, x);
  }
  aie::vector<float, 16> ms = aie::mul(rowsum2(ss0, ss1), 1.0f / 128.0f).to_vector<float>();
  ms = aie::add(ms, aie::broadcast<float, 16>(RMS_EPS));
  aie::accum<accfloat, 16> ia;
  ia.from_vector(aie::invsqrt(ms));
  const vbf16 inv = ia.to_vector<bfloat16>();
  ib0 = row_bcast(inv, 0);
  ib1 = row_bcast(inv, 1);
}

// RMSNorm * w and RoPE of band block i (= sbc*4 + k) of band sbr.
static inline vbf64 norm_rope(int sbr, int i, vbf64 ib0, vbf64 ib1, const bfloat16 *w, const bfloat16 *pTab) {
  const int sbc = i >> 2, k = i & 3;
  const vbf64 x = aie::load_v<64>(band_buf + i * 64);
  const vbf64 xn = aie::mul(aie::mul(x, (k & 2) ? ib1 : ib0).to_vector<bfloat16>(),
                            col_bcast(w + sbc * 16 + (k & 1) * 8))
                       .to_vector<bfloat16>();
  const bfloat16 *tb = pTab + ((sbr * 8 + sbc) * 4 + k) * TAB_BLOCK;
  const vbf32 c = aie::load_v<32>(tb), s = aie::load_v<32>(tb + 32);
  const vbf32 e = aie::filter_even(xn, 1), o = aie::filter_odd(xn, 1);
  const vbf32 ne = aie::msc(aie::mul(e, c), o, s).to_vector<bfloat16>();
  const vbf32 no = aie::mac(aie::mul(e, s), o, c).to_vector<bfloat16>();
  const auto [lo, hi] = aie::interleave_zip(ne, no, 1); // [e0 o0 e1 o1 ...] split in two halves
  return aie::concat(lo, hi);
}

extern "C" {

// Q tile, a_layout(64, 128): band sbr, then 16 col blocks cb, then row half; input block = (sbr, cb/2, half*2 + cb%2).
void q_post(bfp16ebs8 *__restrict pIn, bfloat16 *__restrict pTab, bfp16ebs8 *__restrict pOut) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto eye = eye_block();
  OutStream out(pOut);
  for (int sbr = 0; sbr < 4; sbr++) {
    vbf64 ib0, ib1;
    load_band(pIn, sbr, eye, ib0, ib1);
    for (int cb = 0; cb < 16; cb++)
      for (int h = 0; h < 2; h++)
        push_bfp(out, norm_rope(sbr, (cb >> 1) * 4 + h * 2 + (cb & 1), ib0, ib1, pTab + TAB_W, pTab));
  }
  event1();
}

// K tile, b_layout(K^T, 128, 64): key band kb (= input band), then 16 dim blocks d8, then key half;
// each block is a row-major 8x8 block of K = input block (kb, d8/2, half*2 + d8%2).
void k_post(bfp16ebs8 *__restrict pIn, bfloat16 *__restrict pTab, bfp16ebs8 *__restrict pOut) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto eye = eye_block();
  OutStream out(pOut);
  for (int kb = 0; kb < 4; kb++) {
    vbf64 ib0, ib1;
    load_band(pIn, kb, eye, ib0, ib1);
    for (int d8 = 0; d8 < 16; d8++)
      for (int h = 0; h < 2; h++)
        push_bfp(out, norm_rope(kb, (d8 >> 1) * 4 + h * 2 + (d8 & 1), ib0, ib1, pTab + TAB_W + 128, pTab));
  }
  event1();
}

// V tile, b_layout(V, 64, 128): dim band db (16 dims), then 8 key blocks k8, then dim half; each block is the
// transpose of input block (k8/2, db, (k8%2)*2 + half).
void v_post(bfp16ebs8 *__restrict pIn, bfp16ebs8 *__restrict pOut) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto eye = eye_block();
  OutStream out(pOut);
  for (int db = 0; db < 8; db++)
    for (int k8 = 0; k8 < 8; k8++)
      for (int h = 0; h < 2; h++) {
        InStream<aie_dm_resource::a> is(pIn);
        is.seek(((k8 >> 1) * 8 + db) * 4 + (k8 & 1) * 2 + h);
        push_bfp(out, aie::transpose(decode_bf16(is.pop(), eye), 8, 8));
      }
  event1();
}

} // extern "C"
