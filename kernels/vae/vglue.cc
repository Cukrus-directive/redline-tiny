//===- vglue.cc - VAE decoder glue on octet-planar bf16 ---------------------------------------------------===//
//
// A core gets a whole row segment in L1, seg = [O planes][P pixels][8 channels] (a plane is one channel octet), and
// emits output chunks of KP planes; a static plane counter walks the segment (and wraps for the next one).
//   vnorm_stats / vnorm_apply : QwenImage21RMS_norm over channels, x / ||x|| * sqrt(C) * gamma (sqrt(C) folded into
//                               gamma by the host), then SiLU if VG_SILU (silu32 below, not activations.h's
//                               silu_vec). Chunks are [KP][P][8].
//   vup_plain / vup_sel0/1    : nearest 2x upsample of the segment, chunks [KP][2P][8]. plain: same channels;
//                               sel0 / sel1: output channel c takes input channel 2c (+1), i.e. even/odd lanes of
//                               input planes 2q, 2q+1 (the DupUp shortcuts of up blocks 2 and 3).
//   vadd                      : flat y = a + b over N values.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>


#ifndef VG_P
#define VG_P 64
#endif
#ifndef VG_O
#define VG_O 18
#endif
#ifndef VG_KP
#define VG_KP 6
#endif
#ifndef VG_N
#define VG_N 4096
#endif
#ifndef VG_SILU
#define VG_SILU 1
#endif

constexpr int P = VG_P, O = VG_O, KP = VG_KP, N = VG_N, C = 8 * VG_O;
using v32f = aie::vector<float, 32>;
using v32b = aie::vector<bfloat16, 32>;

alignas(aie::vector_decl_align) static float sq[P * 8];    // lane-wise sum of squares, [px][8]
alignas(aie::vector_decl_align) static float isn[P];       // per-pixel 1 / ||x||
alignas(aie::vector_decl_align) static bfloat16 isb[P];    // the same in bf16
alignas(aie::vector_decl_align) static bfloat16 gam[C];    // gamma * sqrt(C)
static int plane = 0;                                      // next output plane within the segment

static inline v32f f32(v32b v) {
  aie::accum<accfloat, 32> a;
  a.from_vector(v);
  return a.to_vector<float>();
}

static inline v32b b16(v32f v) {
  aie::accum<accfloat, 32> a;
  a.from_vector(v);
  return a.to_vector<bfloat16>();
}

// SiLU to ~0.3 % rms. activations.h's silu_vec goes through AIE2P's vtanh, which returns its argument for
// |u| <= 0.5 and is off by up to 0.05 to |u| ~ 1.5: 2-4 % errors on the bulk of post-norm values (in situ, encoder
// resblocks came out 2.4 % off against 0.8 % for their bfp16 convs alone). Here, with e = 2^(-|z| log2 e) in
// (0, 1] and rn = e / (1 + e) = sigma(-|z|) (1 / (1 + e) as a cubic in e): silu(z) = z rn for z < 0 and
// z - z rn for z >= 0, so neither side cancels and large z don't inherit bf16's coarse steps just below 1.
// log2 e goes in as two bf16 parts (one alone is 0.18 % off: 1.4 % in e at |z| = 8). AIE2P's exp2 is piecewise
// linear, 2^k (1 + h), up to 6 % high; its result is corrected by c(m) = 2^(m - 1) / m of its own mantissa m (the
// same correction as kernels/attn/exp2_fix.h, <= 0.56 %).
static inline v32b silu32(v32b z) {
  const v32b az = aie::bit_and(z.cast_to<int16>(), aie::broadcast<int16, 32>(0x7FFF)).cast_to<bfloat16>();
  aie::accum<accfloat, 32> ta = aie::mul(az, aie::broadcast<bfloat16, 32>(-1.4453125f));
  ta = aie::mac(ta, az, aie::broadcast<bfloat16, 32>(0.00262451171875f));
  const v32f t = aie::max(ta.to_vector<float>(), aie::broadcast<float, 32>(-126.0f));
  const v32b y = aie::exp2<bfloat16>(t);
  const v32b m = aie::bit_or(aie::bit_and(y.cast_to<int16>(), aie::broadcast<int16, 32>(0x007F)),
                             aie::broadcast<int16, 32>(0x3F80)).cast_to<bfloat16>();  // y's mantissa, in [1, 2)
  aie::accum<accfloat, 32> a;
  a.from_vector(aie::broadcast<float, 32>(-0.7285f));
  const v32b c1 = aie::mac(a, m, aie::broadcast<bfloat16, 32>(0.24511719f)).to_vector<bfloat16>();
  a.from_vector(aie::broadcast<float, 32>(1.48242188f));
  const v32b c = aie::mac(a, m, c1).to_vector<bfloat16>();
  const v32b e = aie::mul(y, c).to_vector<bfloat16>();
  a.from_vector(aie::broadcast<float, 32>(0.644431917f));  // 1 / (1 + e) ~ ((C3 e + C2) e + C1) e + C0
  v32b r = aie::mac(a, e, aie::broadcast<bfloat16, 32>(-0.211914062f)).to_vector<bfloat16>();
  a.from_vector(aie::broadcast<float, 32>(-0.930307187f));
  r = aie::mac(a, e, r).to_vector<bfloat16>();
  a.from_vector(aie::broadcast<float, 32>(0.996465245f));
  r = aie::mac(a, e, r).to_vector<bfloat16>();
  const v32b rn = aie::mul(e, r).to_vector<bfloat16>();
  a.from_vector(z);
  const v32b pos = aie::msc(a, z, rn).to_vector<bfloat16>();
  const v32b neg = aie::mul(z, rn).to_vector<bfloat16>();
  return aie::select(pos, neg, aie::lt(z.cast_to<int16>(), aie::zeros<int16, 32>()));
}

// duplicate each pixel (8 values) of an 8-pixel vector: 64 -> 128 values
static inline void dup_store(bfloat16 *dst, aie::vector<bfloat16, 64> v) {
  const auto [lo, hi] = aie::interleave_zip(v, v, 8);
  aie::store_v(dst, lo);
  aie::store_v(dst + 64, hi);
}

template <int S> static inline void vup_sel(const bfloat16 *__restrict seg, bfloat16 *__restrict out) {
  for (int q = 0; q < KP; q++) {  // output plane plane+q takes input planes 2(plane+q), 2(plane+q)+1
    const bfloat16 *pa = seg + 2 * (plane + q) * P * 8, *pb = pa + P * 8;
    for (int px = 0; px < P; px += 8) {
      const aie::vector<bfloat16, 64> a = aie::load_v<64>(pa + px * 8), b = aie::load_v<64>(pb + px * 8);
      const aie::vector<bfloat16, 32> fa = S ? aie::filter_odd(a, 1) : aie::filter_even(a, 1);  // [px][4]
      const aie::vector<bfloat16, 32> fb = S ? aie::filter_odd(b, 1) : aie::filter_even(b, 1);
      const auto [lo, hi] = aie::interleave_zip(fa, fb, 4);                                     // [px][4 a | 4 b]
      dup_store(out + (q * 2 * P + 2 * px) * 8, aie::concat(lo, hi));
    }
  }
  plane += KP;
  if (plane >= O / 2)
    plane = 0;
}

extern "C" {

void vnorm_gamma(bfloat16 *__restrict g) {
  for (int i = 0; i < C; i += 16)
    aie::store_v(gam + i, aie::load_v<16>(g + i));
  plane = 0;
}

// All multiplies are bf16 x bf16 -> fp32 accumulator (native on AIE2P; fp32 vector multiplies are emulated and
// made the first version ~10x slower).
void vnorm_stats(bfloat16 *__restrict seg) {
  for (int i = 0; i < P * 8; i += 32) {  // sum of squares per lane over all planes, in the accumulator
    aie::accum<accfloat, 32> a = aie::zeros<accfloat, 32>();
    for (int q = 0; q < O; q++) {
      const v32b x = aie::load_v<32>(seg + q * P * 8 + i);
      a = aie::mac(a, x, x);
    }
    aie::store_v(sq + i, a.to_vector<float>());
  }
  for (int px = 0; px < P; px++)
    isn[px] = aie::reduce_add(aie::load_v<8>(sq + px * 8));
  for (int i = 0; i < P; i += 8) {
    const aie::vector<float, 8> s = aie::max(aie::load_v<8>(isn + i), aie::broadcast<float, 8>(1e-24f));
    aie::store_v(isn + i, aie::invsqrt(s));
  }
  for (int px = 0; px < P; px++)
    isb[px] = (bfloat16)isn[px];
  plane = 0;
}

void vnorm_apply(bfloat16 *__restrict seg, bfloat16 *__restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int q = 0; q < KP; q++) {
    const aie::vector<bfloat16, 8> g8 = aie::load_v<8>(gam + (plane + q) * 8);
    const v32b g = aie::concat(g8, g8, g8, g8);
    for (int px = 0; px < P; px += 4) {  // 4 pixels x 8 channels per 32-lane vector
      const v32b s = aie::concat(aie::broadcast<bfloat16, 8>(isb[px]), aie::broadcast<bfloat16, 8>(isb[px + 1]),
                                 aie::broadcast<bfloat16, 8>(isb[px + 2]), aie::broadcast<bfloat16, 8>(isb[px + 3]));
      const v32b sg = aie::mul(s, g).to_vector<bfloat16>();
      const v32b x = aie::load_v<32>(seg + ((plane + q) * P + px) * 8);
#if VG_SILU
      aie::store_v(out + (q * P + px) * 8, silu32(aie::mul(x, sg).to_vector<bfloat16>()));
#else
      aie::store_v(out + (q * P + px) * 8, aie::mul(x, sg).to_vector<bfloat16>());
#endif
    }
  }
  plane += KP;
  if (plane >= O)
    plane = 0;
}

void vadd(bfloat16 *__restrict a, bfloat16 *__restrict b, bfloat16 *__restrict y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < N; i += 32)
    aie::store_v(y + i, b16(aie::add(f32(aie::load_v<32>(a + i)), f32(aie::load_v<32>(b + i)))));
}

void vup_plain(bfloat16 *__restrict seg, bfloat16 *__restrict out) {
  for (int q = 0; q < KP; q++)
    for (int px = 0; px < P; px += 8)
      dup_store(out + (q * 2 * P + 2 * px) * 8, aie::load_v<64>(seg + ((plane + q) * P + px) * 8));
  plane += KP;
  if (plane >= O)
    plane = 0;
}

// 2x2 average pool (the encoder's AvgDown shortcut). seg = [2 rows][O planes][P][8]; chunks [KP][P/2][8].
// vdown_plain: same channels. vdown_zi: C -> 2C with out[2c] = 0, out[2c+1] = avg(in[c]) (the temporal
// AvgDown3D on one frame: its zero-padded time step lands on the even channels).
static inline aie::vector<bfloat16, 16> avg2x2(const bfloat16 *r0, const bfloat16 *r1) {  // 4 px -> 2 px
  aie::accum<accfloat, 32> s;
  s.from_vector(aie::load_v<32>(r0));
  s = aie::add(s, aie::load_v<32>(r1));
  const aie::vector<float, 32> v = s.to_vector<float>();
  aie::accum<accfloat, 16> t;
  t.from_vector(aie::filter_even(v, 8));
  t = aie::add(t, aie::filter_odd(v, 8));
  return aie::mul(t.to_vector<bfloat16>(), aie::broadcast<bfloat16, 16>(0.25f)).to_vector<bfloat16>();
}

void vdown_plain(bfloat16 *__restrict seg, bfloat16 *__restrict out) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int q = 0; q < KP; q++) {
    const bfloat16 *r0 = seg + (plane + q) * P * 8, *r1 = r0 + O * P * 8;
    for (int px = 0; px < P; px += 4)
      aie::store_v(out + (q * (P / 2) + px / 2) * 8, avg2x2(r0 + px * 8, r1 + px * 8));
  }
  plane += KP;
  if (plane >= O)
    plane = 0;
}

void vdown_zi(bfloat16 *__restrict seg, bfloat16 *__restrict out) {  // KP output planes = KP/2 input planes
  aie::set_rounding(aie::rounding_mode::conv_even);
  const aie::vector<bfloat16, 8> z = aie::zeros<bfloat16, 8>();
  for (int q = 0; q < KP / 2; q++) {
    const bfloat16 *r0 = seg + (plane + q) * P * 8, *r1 = r0 + O * P * 8;
    bfloat16 *o0 = out + (2 * q) * (P / 2) * 8, *o1 = o0 + (P / 2) * 8;
    for (int px = 0; px < P; px += 4) {
      const aie::vector<bfloat16, 16> a = avg2x2(r0 + px * 8, r1 + px * 8);  // [2 px][8]
      const auto [l0, l1] = aie::interleave_zip(z, aie::filter_even(a, 4), 1);  // channels 0..3 -> lanes 1,3,5,7
      const auto [h0, h1] = aie::interleave_zip(z, aie::filter_odd(a, 4), 1);   // channels 4..7
      aie::store_v(o0 + (px / 2) * 8, aie::concat(l0, l1));
      aie::store_v(o1 + (px / 2) * 8, aie::concat(h0, h1));
    }
  }
  plane += KP / 2;
  if (plane >= O)
    plane = 0;
}

void vup_sel0(bfloat16 *__restrict seg, bfloat16 *__restrict out) { vup_sel<0>(seg, out); }
void vup_sel1(bfloat16 *__restrict seg, bfloat16 *__restrict out) { vup_sel<1>(seg, out); }

} // extern "C"
