// VLM copy of kernels/attn/attn_bfp.cc + causal attention for the Qwen3-VL text model.
// OA build only. Each Q tile i walks all ATT_NQT K/V tiles j (a uniform core loop keeps the program small); the
// kernel counts i and j itself: j < i computed, j == i computed with keys above the diagonal masked, j > i skipped
// (attn_qk_causal and attn_pv_causal return at once, O and l unchanged).
//===- attn_bfp.cc - Qwen-Image flash attention microkernel (AIE2P), v2 ---===//
//
// One core holds a Q tile of MQ queries x D=128 (bfp16, pre-scaled on the host by log2(e)/sqrt(D), so scores are
// in log2 units) and consumes K and V tiles of NT keys, which arrive alternately on one channel (a core has only
// two input DMA channels; Q takes the other). Two calls per KV tile:
//   attn_qk(Q, K): S = Q K^T for the whole tile (one pipelined loop over 16x16 blocks, k-steps unrolled), then per
//                  16-query band: row max -> alpha = exp2(m_old - m_new) -> P = exp2(S - m_new),
//                  l = l*alpha + sum(P) -> P -> bfp16 straight into A-operand order (p_all)
//   attn_pv(V, O): O = O*alpha + P V (one pipelined loop over 16x16 O blocks; O is bf16, block-linear 2x2 8x8)
// attn_end divides O by l (OA builds: attn_end_a, O as the out-projection's bfp16 A tiles). Layouts (host): Q = a_layout(Q, MQ, D); K tile = b_layout(K^T, D, NT);
// V tile = b_layout(V, NT, D); KV stream = K0 V0 K1 V1 ...; O = one (MQ, D) C tile in 2x2 8x8 block-linear order.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <utility>

#include "aie_kernel_utils.h"
#include "common/exp2_bf16.h"

#ifndef ATT_MQ
#define ATT_MQ 64
#endif
#ifndef ATT_NT
#define ATT_NT 32
#endif

constexpr int MQ = ATT_MQ; // queries per core tile
constexpr int D = 128;     // head dim
constexpr int NT = ATT_NT; // keys per K/V tile
constexpr int r = 8, s = 8, t = 8;
constexpr int NB = NT / 16;    // 16-key super-block columns of S
constexpr int DB = D / 16;     // 16-dim super-block columns of O
constexpr int BANDS = MQ / 16; // 16-query bands
constexpr int KS_QK = D / s;   // k-steps per S block
constexpr int KS_PV = NT / s;  // k-steps per O block
static_assert(MQ % 16 == 0 && NT % 16 == 0, "tiles are built from 16x16 super-blocks");

template <typename T, unsigned N, aie_dm_resource R>
using InStream = aie::block_vector_restrict_input_buffer_stream<T, N, R>;
template <typename T, unsigned N, aie_dm_resource R>
using OutStream = aie::block_vector_output_buffer_stream<T, N, R>;
using MMUL = aie::mmul<r, s, t, bfp16ebs8>;
using vbf64 = aie::vector<bfloat16, 64>;
using vbf16 = aie::vector<bfloat16, 16>;
using vf16 = aie::vector<float, 16>;
using AStream = InStream<bfp16ebs8, r * s, aie_dm_resource::a>;
using BStream = InStream<bfp16ebs8, s * t, aie_dm_resource::b>;

constexpr int P_BAND_BYTES = 16 * NT * 9 / 8;
alignas(aie::vector_decl_align) static float m_run[MQ];               // running row max (log2 units)
alignas(aie::vector_decl_align) static float l_run[MQ];               // running row sum
alignas(aie::vector_decl_align) static bfloat16 ab_all[BANDS * 128];  // alpha row-broadcast blocks, 2 per band
#ifdef ATT_OA
// OA builds (O as the out-projection's bfp16 A operand): O accumulates here, and the core's O fifo buffer (MQ*D
// bfp16 = 9 KB) carries S during the wave (S needs MQ*NT bf16) and the bfp16 output at the end. Saves L1.
alignas(aie::vector_decl_align) static bfloat16 o_acc[MQ * D];
static_assert(MQ * NT * 2 <= MQ * D * 9 / 8, "OA: S must fit in the bfp16 O buffer");
#else
alignas(aie::vector_decl_align) static bfloat16 s_all[MQ * NT];       // S/P, block-linear: band, sc, 4 panels
#endif
alignas(aie::vector_decl_align) static uint8_t p_all[BANDS * P_BAND_BYTES]; // P as bfp16, A order, per band

static inline vf16 to_f32(vbf16 v) {
  aie::accum<accfloat, 16> a;
  a.from_vector(v);
  return a.to_vector<float>();
}

static inline vbf16 to_bf16(vf16 v) {
  aie::accum<accfloat, 16> a;
  a.from_vector(v);
  return a.to_vector<bfloat16>();
}

// 8x8 row-major block whose row i is filled with v[half*8 + i].
static inline vbf64 row_bcast(vbf16 v, int half) {
  aie::vector<bfloat16, 8> e = v.extract<8>(half);
  vbf64 c = aie::concat(e, e, e, e, e, e, e, e); // lane j*8+i = e[i]
  return aie::transpose(c, 8, 8);               // lane i*8+j = e[i]
}

#ifdef ATT_EXP2_FIX // both uses; ATT_EXP2_FIX_A / ATT_EXP2_FIX_P correct only alpha / only P
#define ATT_EXP2_FIX_A
#define ATT_EXP2_FIX_P
#endif
#if defined(ATT_EXP2_FIX_A) || defined(ATT_EXP2_FIX_P)
#include "exp2_fix.h"
#endif

// exp2(s - m) for one 8x8 block, computed as two 32-lane halves.
static inline vbf64 exp2_rows(vbf64 s, vbf64 m) {
  const vbf64 d = aie::sub(s, m);
  aie::accum<accfloat, 32> lo, hi;
  lo.from_vector(d.extract<32>(0));
  hi.from_vector(d.extract<32>(1));
  return aie::concat(exp2_bf16(lo.to_vector<float>()), exp2_bf16(hi.to_vector<float>()));
}

// Row reduction of two 8x8 row-major blocks: lanes 0..7 <- rows of b0, lanes 8..15 <- rows of b1.
template <bool MAX>
static inline vbf16 rowreduce2(vbf64 b0, vbf64 b1) {
  auto fold = [](vbf64 v) {
    vbf64 tr = aie::transpose(v, 8, 8); // lane c*8 + row
    aie::vector<bfloat16, 32> h;
    vbf16 q;
    if constexpr (MAX) {
      h = aie::max(tr.extract<32>(0), tr.extract<32>(1));
      q = aie::max(h.extract<16>(0), h.extract<16>(1)); // lane (c%2)*8 + row
    } else {
      h = aie::add(tr.extract<32>(0), tr.extract<32>(1));
      q = aie::add(h.extract<16>(0), h.extract<16>(1));
    }
    return q;
  };
  vbf16 q0 = fold(b0), q1 = fold(b1);
  vbf16 lo = aie::concat(q0.extract<8>(0), q1.extract<8>(0));
  vbf16 hi = aie::concat(q0.extract<8>(1), q1.extract<8>(1));
  if constexpr (MAX)
    return aie::max(lo, hi);
  else
    return aie::add(lo, hi);
}

// One k-step of a 16x16 output super-block: 2 A pops + 2 B pops + 4 8x8x8 MACs. On fill steps (every 8th B pop)
// the B pops go first so the stream fill issues ahead of the A-stream ops (same trick as upstream ATB config3).
template <bool FIRST, bool BFILL>
static inline __attribute__((always_inline)) void kstep(AStream &as, BStream &bs, MMUL &acc0, MMUL &acc1,
                                                        MMUL &acc2, MMUL &acc3) {
  aie::block_vector<bfp16ebs8, r * s> A0, A1;
  aie::block_vector<bfp16ebs8, s * t> B0, B1;
  if constexpr (BFILL) {
    B0 = bs.pop();
    B1 = bs.pop();
    A0 = as.pop();
    A1 = as.pop();
  } else {
    A0 = as.pop();
    A1 = as.pop();
    B0 = bs.pop();
    B1 = bs.pop();
  }
  if constexpr (FIRST) {
    acc0.mul(A0, aie::op_transpose(B0));
    acc1.mul(A0, aie::op_transpose(B1));
    acc2.mul(A1, aie::op_transpose(B0));
    acc3.mul(A1, aie::op_transpose(B1));
  } else {
    acc0.mac(A0, aie::op_transpose(B0));
    acc1.mac(A0, aie::op_transpose(B1));
    acc2.mac(A1, aie::op_transpose(B0));
    acc3.mac(A1, aie::op_transpose(B1));
  }
}

// k-steps 1..N-1 of a block, unrolled at compile time, with B-first fills on every 4th step.
template <int... I>
static inline __attribute__((always_inline)) void ksteps_rest(std::integer_sequence<int, I...>, AStream &as,
                                                              BStream &bs, MMUL &a0, MMUL &a1, MMUL &a2,
                                                              MMUL &a3) {
  (kstep<false, ((I + 1) % 4 == 0)>(as, bs, a0, a1, a2, a3), ...);
}

static inline void begin_impl(bfloat16 *__restrict pO) {
  const vbf64 z = aie::zeros<bfloat16, 64>();
  for (int i = 0; i < MQ * D / 64; i++)
    aie::store_v(pO + i * 64, z);
  for (int i = 0; i < MQ / 16; i++) {
    aie::store_v(m_run + i * 16, aie::broadcast<float, 16>(-1e30f));
    aie::store_v(l_run + i * 16, aie::zeros<float, 16>());
  }
}

extern "C" {

#ifdef ATT_OA
void attn_begin(bfp16ebs8 *__restrict) { begin_impl(o_acc); }
#else
void attn_begin(bfloat16 *__restrict pO) { begin_impl(pO); }
#endif

} // extern "C"

#ifndef ATT_NVALID_LAST
#define ATT_NVALID_LAST NT // valid keys in the last K/V tile (attn_qk_last)
#endif
// Masked score. Finite on purpose: with -max_bf16, S - m overflows to -inf, and exp2 of that is not a clean 0.
// exp2(-30000 - m) underflows to exactly 0 for any real row max.
constexpr bfloat16 kNegInf = bfloat16(-30000.0f);

// Key padding: set S columns >= ATT_NVALID_LAST of the current tile to -inf (all bands, both row halves).
static inline void mask_tail(bfloat16 *__restrict s_all) {
  const vbf64 ninf = aie::broadcast<bfloat16, 64>(kNegInf);
  for (int sc = 0; sc < NB; sc++) {
    for (int bc = 0; bc < 2; bc++) {
      const int v = ATT_NVALID_LAST - (sc * 16 + bc * 8); // valid columns in this 8-column block
      if (v >= 8)
        continue;
      const uint32_t row = (v <= 0) ? 0xFFu : (0xFFu & ~((1u << v) - 1u)); // invalid lanes of one row
      const uint32_t w = row | (row << 8) | (row << 16) | (row << 24);
      const auto m = aie::mask<64>::from_uint32(w, w);
      for (int b = 0; b < BANDS; b++) {
        for (int br = 0; br < 2; br++) {
          bfloat16 *ps = s_all + (b * NB + sc) * 256 + br * 128 + bc * 64;
          aie::store_v(ps, aie::select(aie::load_v<64>(ps), ninf, m));
        }
      }
    }
  }
}

// Causal: on the diagonal tile (queries and keys at the same 64 positions), key k > query q is masked. 8x8 panels:
// below the diagonal kept, above it masked, on it the strict upper triangle (lane rib*8 + cib, cib > rib).
static inline void mask_diag(bfloat16 *__restrict s_all) {
  static_assert(MQ == NT, "causal diagonal tile needs MQ == NT");
  const vbf64 ninf = aie::broadcast<bfloat16, 64>(kNegInf);
  const auto tri = aie::mask<64>::from_uint32(0xF0F8FCFEu, 0x0080C0E0u);
  const auto all = aie::mask<64>::from_uint32(0xFFFFFFFFu, 0xFFFFFFFFu);
  for (int b = 0; b < BANDS; b++)
    for (int sc = 0; sc < NB; sc++)
      for (int br = 0; br < 2; br++)
        for (int bc = 0; bc < 2; bc++) {
          const int q0 = b * 16 + br * 8, k0 = sc * 16 + bc * 8;
          if (k0 < q0)
            continue;
          bfloat16 *ps = s_all + (b * NB + sc) * 256 + br * 128 + bc * 64;
          aie::store_v(ps, aie::select(aie::load_v<64>(ps), ninf, k0 == q0 ? tri : all));
        }
}

template <bool LAST>
static inline void attn_qk_impl(bfp16ebs8 *__restrict pQ, bfp16ebs8 *__restrict pK, bfloat16 *__restrict s_all,
                                bool diag = false) {
  event0();
#ifdef ATT_NOOP // NOOP ablation: data movement only
  event1();
  return;
#endif
  aie::set_rounding(aie::rounding_mode::conv_even);

  // ---- S = Q K^T: one loop over the tile's 16x16 blocks (band-major), all k-steps unrolled ----
  AIE_LOOP_MIN_ITERATION_COUNT(BANDS * NB)
  for (int blk = 0; blk < BANDS * NB; blk++) {
    const int b = blk / NB, sc = blk % NB;
    AStream q_stream(pQ);
    q_stream.seek(b * (2 * KS_QK));
    BStream k_stream(pK);
    k_stream.seek(sc * (2 * KS_QK));
    MMUL acc0, acc1, acc2, acc3;
    kstep<true, true>(q_stream, k_stream, acc0, acc1, acc2, acc3);
    kstep<false, false>(q_stream, k_stream, acc0, acc1, acc2, acc3);
    kstep<false, false>(q_stream, k_stream, acc0, acc1, acc2, acc3);
    kstep<false, false>(q_stream, k_stream, acc0, acc1, acc2, acc3);
    AIE_LOOP_UNROLL_FULL
    for (int g = 1; g < KS_QK / 4; g++) {
      kstep<false, true>(q_stream, k_stream, acc0, acc1, acc2, acc3);
      kstep<false, false>(q_stream, k_stream, acc0, acc1, acc2, acc3);
      kstep<false, false>(q_stream, k_stream, acc0, acc1, acc2, acc3);
      kstep<false, false>(q_stream, k_stream, acc0, acc1, acc2, acc3);
    }
    bfloat16 *ps = s_all + blk * 256;
    aie::store_v(ps, acc0.template to_vector<bfloat16>());
    aie::store_v(ps + 64, acc1.template to_vector<bfloat16>());
    aie::store_v(ps + 128, acc2.template to_vector<bfloat16>());
    aie::store_v(ps + 192, acc3.template to_vector<bfloat16>());
  }

  if constexpr (LAST)
    mask_tail(s_all);
  if (diag)
    mask_diag(s_all);

  // ---- online softmax per 16-query band ----
  for (int b = 0; b < BANDS; b++) {
    const bfloat16 *sb = s_all + b * NB * 256;
    vbf64 mx0 = aie::max(aie::load_v<64>(sb), aie::load_v<64>(sb + 64));
    vbf64 mx1 = aie::max(aie::load_v<64>(sb + 128), aie::load_v<64>(sb + 192));
    for (int sc = 1; sc < NB; sc++) {
      const bfloat16 *ps = sb + sc * 256;
      mx0 = aie::max(mx0, aie::max(aie::load_v<64>(ps), aie::load_v<64>(ps + 64)));
      mx1 = aie::max(mx1, aie::max(aie::load_v<64>(ps + 128), aie::load_v<64>(ps + 192)));
    }
    const vf16 m_old = aie::load_v<16>(m_run + 16 * b);
    const vf16 m_new = aie::max(m_old, to_f32(rowreduce2<true>(mx0, mx1)));
#ifdef ATT_EXP2_FIX_A
    const vbf16 alpha = exp2_corr<16>(exp2_bf16(aie::sub(m_old, m_new))).to_vector<bfloat16>();
#else
    const vbf16 alpha = exp2_bf16(aie::sub(m_old, m_new));
#endif
    aie::store_v(m_run + 16 * b, m_new);
    aie::store_v(ab_all + b * 128, row_bcast(alpha, 0));
    aie::store_v(ab_all + b * 128 + 64, row_bcast(alpha, 1));
    const vbf16 m_new_bf = to_bf16(m_new);
    const vbf64 mb0 = row_bcast(m_new_bf, 0), mb1 = row_bcast(m_new_bf, 1);

    vbf64 rs0 = aie::zeros<bfloat16, 64>(), rs1 = aie::zeros<bfloat16, 64>();
    OutStream<bfp16ebs8, 64, aie_dm_resource::c> p_out((bfp16ebs8 *)(p_all + b * P_BAND_BYTES));
    for (int kb = 0; kb < 2 * NB; kb++) { // key block kb = sc*2 + bc; A order: key block outer, row half inner
      const bfloat16 *ps = sb + (kb >> 1) * 256 + (kb & 1) * 64;
      // exp2 in 32-lane halves (1024-bit fp32 vectors): with 64-lane fp32 vectors one 16-lane group of the
      // second row half came out wrong under some schedules (NT=32 rows 14-15, masked tail rows 10-11).
      const vbf64 p0 = exp2_rows(aie::load_v<64>(ps), mb0);
      const vbf64 p1 = exp2_rows(aie::load_v<64>(ps + 128), mb1);
#ifdef ATT_EXP2_FIX_P
      // corrected in 32-lane halves too (the polynomial's fp32 broadcasts at 64 lanes gave wrong results)
      const aie::accum<accfloat, 64> a0 = aie::concat(exp2_corr<32>(p0.extract<32>(0)), exp2_corr<32>(p0.extract<32>(1)));
      const aie::accum<accfloat, 64> a1 = aie::concat(exp2_corr<32>(p1.extract<32>(0)), exp2_corr<32>(p1.extract<32>(1)));
      rs0 = aie::add(rs0, a0.to_vector<bfloat16>());
      rs1 = aie::add(rs1, a1.to_vector<bfloat16>());
#else
      rs0 = aie::add(rs0, p0);
      rs1 = aie::add(rs1, p1);
      aie::accum<accfloat, 64> a0, a1;
      a0.from_vector(p0);
      a1.from_vector(p1);
#endif
      p_out.push(a0.to_vector<bfp16ebs8>());
      p_out.push(a1.to_vector<bfp16ebs8>());
    }
    aie::accum<accfloat, 16> la = aie::mul(aie::load_v<16>(l_run + 16 * b), to_f32(alpha));
    la = aie::add(la, to_f32(rowreduce2<false>(rs0, rs1)));
    aie::store_v(l_run + 16 * b, la.to_vector<float>());
  }
  event1();
}

static inline void attn_pv_impl(bfp16ebs8 *__restrict pV, bfloat16 *__restrict pO) {
  event0();
#ifdef ATT_NOOP
  event1();
  return;
#endif
  aie::set_rounding(aie::rounding_mode::conv_even);
#ifndef ATT_PV_BAND
  // ---- O = O*alpha + P V: one loop over the 16x16 O blocks (band-major), fresh streams per block ----
  // (measured 2898 cycles per 64-key tile; the band-outer variant below measured 3493)
  for (int blk = 0; blk < BANDS * DB; blk++) {
    const int b = blk / DB, dc = blk % DB;
    const vbf64 ab0 = aie::load_v<64>(ab_all + b * 128), ab1 = aie::load_v<64>(ab_all + b * 128 + 64);
    bfloat16 *pc = pO + blk * 256;
    MMUL acc0(aie::mul(aie::load_v<64>(pc), ab0));
    MMUL acc1(aie::mul(aie::load_v<64>(pc + 64), ab0));
    MMUL acc2(aie::mul(aie::load_v<64>(pc + 128), ab1));
    MMUL acc3(aie::mul(aie::load_v<64>(pc + 192), ab1));
    AStream p_stream((const bfp16ebs8 *)(p_all + b * P_BAND_BYTES));
    BStream v_stream(pV);
    v_stream.seek(dc * (2 * KS_PV));
    kstep<false, true>(p_stream, v_stream, acc0, acc1, acc2, acc3);
    ksteps_rest(std::make_integer_sequence<int, KS_PV - 1>{}, p_stream, v_stream, acc0, acc1, acc2, acc3);
    aie::store_v(pc, acc0.template to_vector<bfloat16>());
    aie::store_v(pc + 64, acc1.template to_vector<bfloat16>());
    aie::store_v(pc + 128, acc2.template to_vector<bfloat16>());
    aie::store_v(pc + 192, acc3.template to_vector<bfloat16>());
  }
#else
  // ---- variant: per band, one loop over its O blocks with the V stream persistent across blocks, a fresh P
  // stream per block and the next block's O pair prefetched (config3's structure) ----
  for (int b = 0; b < BANDS; b++) {
    const vbf64 ab0 = aie::load_v<64>(ab_all + b * 128), ab1 = aie::load_v<64>(ab_all + b * 128 + 64);
    const bfp16ebs8 *pP = (const bfp16ebs8 *)(p_all + b * P_BAND_BYTES);
    bfloat16 *pcb = pO + b * DB * 256;
    BStream v_stream(pV);
    vbf64 c0 = aie::load_v<64>(pcb), c1 = aie::load_v<64>(pcb + 64);
    AIE_LOOP_MIN_ITERATION_COUNT(DB)
    for (int dc = 0; dc < DB; dc++) {
      bfloat16 *pc = pcb + dc * 256;
      bfloat16 *pnext = (dc < DB - 1) ? pc + 256 : pc;
#ifdef ATT_PV_NOALPHA // experiment: accumulators straight from O, no rescale
      MMUL acc0((aie::accum<accfloat, 64>(c0)));
      MMUL acc1((aie::accum<accfloat, 64>(c1)));
      MMUL acc2((aie::accum<accfloat, 64>(aie::load_v<64>(pc + 128))));
      MMUL acc3((aie::accum<accfloat, 64>(aie::load_v<64>(pc + 192))));
#else
      MMUL acc0(aie::mul(c0, ab0));
      MMUL acc1(aie::mul(c1, ab0));
      MMUL acc2(aie::mul(aie::load_v<64>(pc + 128), ab1));
      MMUL acc3(aie::mul(aie::load_v<64>(pc + 192), ab1));
#endif
      c0 = aie::load_v<64>(pnext);
      c1 = aie::load_v<64>(pnext + 64);
      AStream p_stream(pP);
      kstep<false, true>(p_stream, v_stream, acc0, acc1, acc2, acc3);
      ksteps_rest(std::make_integer_sequence<int, KS_PV - 1>{}, p_stream, v_stream, acc0, acc1, acc2, acc3);
      aie::store_v(pc, acc0.template to_vector<bfloat16>());
      aie::store_v(pc + 64, acc1.template to_vector<bfloat16>());
      aie::store_v(pc + 128, acc2.template to_vector<bfloat16>());
      aie::store_v(pc + 192, acc3.template to_vector<bfloat16>());
    }
  }
#endif
  event1();
}

extern "C" {

#ifdef ATT_OA
void attn_qk(bfp16ebs8 *__restrict pQ, bfp16ebs8 *__restrict pK, bfp16ebs8 *__restrict pX) {
  attn_qk_impl<false>(pQ, pK, (bfloat16 *)pX);
}
void attn_qk_last(bfp16ebs8 *__restrict pQ, bfp16ebs8 *__restrict pK, bfp16ebs8 *__restrict pX) {
  attn_qk_impl<true>(pQ, pK, (bfloat16 *)pX);
}
void attn_pv(bfp16ebs8 *__restrict pV, bfp16ebs8 *__restrict) { attn_pv_impl(pV, o_acc); }

#ifndef ATT_NQT
#define ATT_NQT 1
#endif
static int c_q = 0, c_kv = 0; // Q tile i of the current call, K/V tile j within it
static bool c_skip = false;

void attn_begin_causal(bfp16ebs8 *__restrict) {
  begin_impl(o_acc);
  c_kv = 0;
}
void attn_qk_causal(bfp16ebs8 *__restrict pQ, bfp16ebs8 *__restrict pK, bfp16ebs8 *__restrict pX) {
  const int j = c_kv++;
  c_skip = j > c_q;
  if (!c_skip)
    attn_qk_impl<false>(pQ, pK, (bfloat16 *)pX, j == c_q);
}
void attn_pv_causal(bfp16ebs8 *__restrict pV, bfp16ebs8 *__restrict) {
  if (!c_skip)
    attn_pv_impl(pV, o_acc);
}

// O / l as bfp16 in the out-projection GEMM's A layout (a_layout 128x64), over the O fifo buffer pX (S is dead by
// now): the tile's MQ rows x 128 dims as [dims 0..63 | dims 64..127], each as MQ/16 16-row bands of an A tile: per
// band 8 col blocks, then the band's two 8x8 row halves (8x8 row-major, one bfp16 block per row).
void attn_end_a(bfp16ebs8 *__restrict pX) {
  c_q = (c_q + 1) % ATT_NQT; // next Q tile (wraps for the next call)
  aie::set_rounding(aie::rounding_mode::conv_even);
  OutStream<bfp16ebs8, 64, aie_dm_resource::c> out(pX);
  for (int kk = 0; kk < 2; kk++) {
    for (int b = 0; b < BANDS; b++) {
      const vbf16 il = to_bf16(aie::inv(aie::load_v<16>(l_run + 16 * b)));
      const vbf64 i0 = row_bcast(il, 0), i1 = row_bcast(il, 1);
      for (int cb = 0; cb < 8; cb++) {
        const int g = kk * 8 + cb; // 8-dim block of the head
        const bfloat16 *pc = o_acc + (b * DB + (g >> 1)) * 256 + (g & 1) * 64;
        out.push(aie::mul(aie::load_v<64>(pc), i0).template to_vector<bfp16ebs8>());
        out.push(aie::mul(aie::load_v<64>(pc + 128), i1).template to_vector<bfp16ebs8>());
      }
    }
  }
}
#else
void attn_qk(bfp16ebs8 *__restrict pQ, bfp16ebs8 *__restrict pK) { attn_qk_impl<false>(pQ, pK, s_all); }

// Last K/V tile of a padded key range: keys >= ATT_NVALID_LAST are masked to -inf.
void attn_qk_last(bfp16ebs8 *__restrict pQ, bfp16ebs8 *__restrict pK) { attn_qk_impl<true>(pQ, pK, s_all); }

void attn_pv(bfp16ebs8 *__restrict pV, bfloat16 *__restrict pO) { attn_pv_impl(pV, pO); }

void attn_end(bfloat16 *__restrict pO) {
  for (int b = 0; b < BANDS; b++) {
    const vbf16 il = to_bf16(aie::inv(aie::load_v<16>(l_run + 16 * b)));
    const vbf64 i0 = row_bcast(il, 0), i1 = row_bcast(il, 1);
    for (int dc = 0; dc < DB; dc++) {
      bfloat16 *pc = pO + (b * DB + dc) * 256;
      aie::store_v(pc, aie::mul(aie::load_v<64>(pc), i0).template to_vector<bfloat16>());
      aie::store_v(pc + 64, aie::mul(aie::load_v<64>(pc + 64), i0).template to_vector<bfloat16>());
      aie::store_v(pc + 128, aie::mul(aie::load_v<64>(pc + 128), i1).template to_vector<bfloat16>());
      aie::store_v(pc + 192, aie::mul(aie::load_v<64>(pc + 192), i1).template to_vector<bfloat16>());
    }
  }
}
#endif

} // extern "C"
