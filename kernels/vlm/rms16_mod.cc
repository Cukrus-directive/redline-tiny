// rms_mod.cc with the branch in bf16 (the C16 GEMM's output) instead of bfp16.
//===- rms_mod.cc - residual add + RMSNorm * weight between the GEMMs (AIE2P); VLM copy of ln_mod.cc -===//
//
// Per 128-row M-tile (one core), two passes over the tile's 32 C tiles, one 16-row band (2048 values) per call:
//   ln_par x4 : this M-tile's parameter row, 4 chunks of 2048 bf16: g[4096] = tanh(gate), sc[4096] = 1 + scale.
//   ln_p1 x256: x' = x + g * branch (bf16-rounded, as the residual stream is stored), per-row sums of x' and x'^2.
//   ln_fin    : rstd per row (RMSNorm over 4096 channels, eps 1e-6; mean forced to 0).
//   ln_p2 x256: x' again -> stored (bf16, C order); y = (x' - mean) * rstd * sc -> bfp16 in A order.
// In : x band  = 2048 bf16 in config3's C layout (2x2 super-blocks of row-major 8x8: sbc, br, bc, rib, cib).
//      branch  = the same band as bfp16 (256 blocks of 8).
// Out: x' band = 2048 bf16, same order; A = two A-tile band pieces (cols 0-63 and 64-127 of the C tile), each 16
//      rows x 64 cols in a_layout order (8-col blocks, then the band's two 8-row halves).
// The residual stream's per-token |mean|/std is <= 0.07 in every block (golden step 1), so fp32 sums of x and x^2
// are exact enough for the variance.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using vbf64 = aie::vector<bfloat16, 64>;
using vbf32 = aie::vector<bfloat16, 32>;
using vf32 = aie::vector<float, 32>;
using vf16 = aie::vector<float, 16>;
using MMUL = aie::mmul<8, 8, 8, bfp16ebs8>;
using InStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, aie_dm_resource::a>;
using EyeStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b>;
using OutStream = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c>;

alignas(aie::vector_decl_align) static bfloat16 par[8192]; // g[4096] | sc[4096]
// Pass 1: per band, per row half br, per 4-row quarter h: 32 lanes (row*8 + col) of running sums.
// ln_fin overwrites each band's slot in place: st1 -> mean broadcast (fp32), st2 -> rstd broadcast (bf16).
alignas(aie::vector_decl_align) static float st1[8 * 128];
alignas(aie::vector_decl_align) static float st2[8 * 128];
alignas(aie::vector_decl_align) static uint8_t eye_bfp[72]; // 8x8 identity: row i = [exp 127][64 at col i]
static int par_cnt = 0, c1 = 0, c2 = 0;
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

// bfp16 block -> bf16, exactly (8-bit mantissas fit bf16): mmul against the identity.
static inline vbf64 decode(const aie::block_vector<bfp16ebs8, 64> &b, const aie::block_vector<bfp16ebs8, 64> &eye) {
  MMUL m;
  m.mul(b, aie::op_transpose(eye));
  return m.template to_vector<bfloat16>();
}

// 8 per-column values -> an 8x8 block (lane row*8 + col = p[col]).
static inline vbf64 col_bcast(const bfloat16 *p) {
  const aie::vector<bfloat16, 8> e = aie::load_v<8>(p);
  return aie::concat(e, e, e, e, e, e, e, e);
}

// Sum over the 8 columns of a 4-row x 8-col block half (lane row*8 + col) -> 4 row sums.
static inline aie::vector<float, 4> fold4(vf32 v) {
  const aie::vector<float, 16> a = aie::add(aie::filter_even(v, 4), aie::filter_odd(v, 4)); // row*4 + c
  const aie::vector<float, 8> b = aie::add(aie::filter_even(a, 2), aie::filter_odd(a, 2));  // row*2 + c
  return aie::add(aie::filter_even(b, 1), aie::filter_odd(b, 1));                          // row
}

// Rows r0..r0+3 of v, each repeated over 8 columns (lane row*8 + col).
static inline vf32 rows_bcast(vf16 v, int r0) {
  return aie::concat(aie::broadcast<float, 8>(v[r0]), aie::broadcast<float, 8>(v[r0 + 1]),
                     aie::broadcast<float, 8>(v[r0 + 2]), aie::broadcast<float, 8>(v[r0 + 3]));
}

static inline vbf32 to_bf16(vf32 v) {
  aie::accum<accfloat, 32> a;
  a.from_vector(v);
  return a.template to_vector<bfloat16>();
}

static inline vf32 to_f32(vbf32 v) {
  aie::accum<accfloat, 32> a;
  a.from_vector(v);
  return a.template to_vector<float>();
}

extern "C" {

// One of the 4 parameter chunks of an M-tile; the first one also resets the pass state.
void ln_par(bfloat16 *__restrict chunk) {
  if (par_cnt == 0) {
    for (int i = 0; i < 8 * 128; i += 32) {
      aie::store_v(st1 + i, aie::zeros<float, 32>());
      aie::store_v(st2 + i, aie::zeros<float, 32>());
    }
    c1 = 0;
    c2 = 0;
  }
  for (int i = 0; i < 2048; i += 64)
    aie::store_v(par + par_cnt * 2048 + i, aie::load_v<64>(chunk + i));
  par_cnt = (par_cnt + 1) & 3;
}

void ln_p1(bfloat16 *__restrict x, bfloat16 *__restrict pbr) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int nt = c1 >> 3, band = c1 & 7;
  c1++;
  const auto eye = eye_block();
  const bfloat16 *g = par + nt * 128;
  const vbf32 one = aie::broadcast<bfloat16, 32>(1.0f);
  for (int br = 0; br < 2; br++) {
    float *s1 = st1 + band * 128 + br * 64, *s2 = st2 + band * 128 + br * 64;
    aie::accum<accfloat, 32> a10, a11, a20, a21;
    a10.from_vector(aie::load_v<32>(s1));
    a11.from_vector(aie::load_v<32>(s1 + 32));
    a20.from_vector(aie::load_v<32>(s2));
    a21.from_vector(aie::load_v<32>(s2 + 32));
    for (int sbc = 0; sbc < 8; sbc++)
      for (int bc = 0; bc < 2; bc++) {
        const int k = (sbc * 2 + br) * 2 + bc;
        aie::accum<accfloat, 64> xa;
        xa.from_vector(aie::load_v<64>(x + k * 64));
        xa = aie::mac(xa, col_bcast(g + sbc * 16 + bc * 8), aie::load_v<64>(pbr + k * 64));
        const vbf64 xb = xa.template to_vector<bfloat16>();
        const vbf32 lo = xb.extract<32>(0), hi = xb.extract<32>(1);
        a10 = aie::mac(a10, lo, one);
        a11 = aie::mac(a11, hi, one);
        a20 = aie::mac(a20, lo, lo);
        a21 = aie::mac(a21, hi, hi);
      }
    aie::store_v(s1, a10.to_vector<float>());
    aie::store_v(s1 + 32, a11.to_vector<float>());
    aie::store_v(s2, a20.to_vector<float>());
    aie::store_v(s2 + 32, a21.to_vector<float>());
  }
  event1();
}

void ln_fin() {
  const float inv_n = 1.0f / 4096.0f;
  for (int band = 0; band < 8; band++) {
    float *s1 = st1 + band * 128, *s2 = st2 + band * 128;
    // rows of the band: br * 8 + h * 4 + r, from slot (br, h) at s + (br * 2 + h) * 32
    const vf16 S1 = aie::concat(fold4(aie::load_v<32>(s1)), fold4(aie::load_v<32>(s1 + 32)),
                                fold4(aie::load_v<32>(s1 + 64)), fold4(aie::load_v<32>(s1 + 96)));
    const vf16 S2 = aie::concat(fold4(aie::load_v<32>(s2)), fold4(aie::load_v<32>(s2 + 32)),
                                fold4(aie::load_v<32>(s2 + 64)), fold4(aie::load_v<32>(s2 + 96)));
    // RMSNorm: no mean subtraction (S1 is unused), var = E[x^2] + eps
    (void)S1;
    const vf16 mean = aie::zeros<float, 16>();
    const vf16 ex2 = aie::mul(S2, inv_n).to_vector<float>();
    const vf16 var = aie::add(ex2, aie::broadcast<float, 16>(1e-6f));
    vf16 r = aie::inv(aie::sqrt(var));
    // one Newton step on 1/sqrt(var): r * (1.5 - 0.5 * var * r^2)
    const vf16 vr2 = aie::mul(aie::mul(var, r).to_vector<float>(), r).to_vector<float>();
    r = aie::mul(r, aie::sub(aie::broadcast<float, 16>(1.5f), aie::mul(vr2, 0.5f).to_vector<float>()))
            .to_vector<float>();
    bfloat16 *rb = (bfloat16 *)s2;
    for (int q = 0; q < 4; q++) {
      aie::store_v(s1 + q * 32, rows_bcast(mean, q * 4));
      aie::store_v(rb + q * 32, to_bf16(rows_bcast(r, q * 4)));
    }
  }
}

void ln_p2(bfloat16 *__restrict x, bfloat16 *__restrict pbr, bfloat16 *__restrict xo, bfp16ebs8 *__restrict pa) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int nt = c2 >> 3, band = c2 & 7;
  c2++;
  const auto eye = eye_block();
  const bfloat16 *g = par + nt * 128, *sc = par + 4096 + nt * 128;
  const float *mb = st1 + band * 128;
  const bfloat16 *rb = (const bfloat16 *)(st2 + band * 128);
  OutStream out(pa);
  for (int ha = 0; ha < 2; ha++)      // A tile 2*nt + ha: C tile cols 64*ha .. 64*ha + 63
    for (int bca = 0; bca < 8; bca++) { // 8-col block of the A tile
      const int sbc = 4 * ha + (bca >> 1), bc = bca & 1;
      const vbf64 gb = col_bcast(g + sbc * 16 + bc * 8), sb = col_bcast(sc + sbc * 16 + bc * 8);
      for (int br = 0; br < 2; br++) { // the band's row half (A's bis)
        const int k = (sbc * 2 + br) * 2 + bc;
        aie::accum<accfloat, 64> xa;
        xa.from_vector(aie::load_v<64>(x + k * 64));
        xa = aie::mac(xa, gb, aie::load_v<64>(pbr + k * 64));
        const vbf64 xb = xa.template to_vector<bfloat16>();
        aie::store_v(xo + k * 64, xb);
        aie::accum<accfloat, 32> y[2];
        for (int h = 0; h < 2; h++) {
          const int q = br * 2 + h;
          const vbf32 d = to_bf16(aie::sub(to_f32(xb.extract<32>(h)), aie::load_v<32>(mb + q * 32)));
          const vbf32 rs = aie::mul(aie::load_v<32>(rb + q * 32), sb.extract<32>(h)).to_vector<bfloat16>();
          y[h] = aie::mul(d, rs);
        }
        out.push(aie::concat(y[0], y[1]).to_vector<bfp16ebs8>());
      }
    }
  event1();
}

} // extern "C"
