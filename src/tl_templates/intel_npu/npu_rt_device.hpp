// Device implementation of the TileLang Intel NPU runtime surface over the
// generic_compute toolkit. Included by npu_rt.hpp; compiled by moviCompile.
//
// Facts this header relies on (toolkit code, not inferred hardware behaviour):
//  * every SHAVE of a tile runs the same allocation sequence on the tile's
//    LinearAllocator, so one allocation of `shaves_per_tile` slots gives each
//    SHAVE its own slot (worker-private CMX);
//  * DMA/DPU descriptors must not lie in the DMA driver's trap-descriptor
//    window of the tile's CMX (DmaDrv40xx::init: cmx.METADATA + 0x129C0);
//  * DpuMatMul::init(M, N, K) builds a weight table of 16 bytes per (padded) N
//    row at the address given to its constructor;
//  * a SHAVE's direct DDR store (DdrAccessor) sits in its data L1 until flushed:
//    without npu::sync::memoryFence() the store can be lost when the SHAVE
//    halts (seen as an unwritten status record in ~1 of 25 launches).
#ifndef TL_TEMPLATES_INTEL_NPU_NPU_RT_DEVICE_HPP_
#define TL_TEMPLATES_INTEL_NPU_NPU_RT_DEVICE_HPP_

#include <shave/kernel/include/kernel.hpp>

#include <hw/cmx/include/cmx_flush.hpp>
#include <hw/ddr/include/ddr_accessor_40xx.hpp>
#include <lib/dma/include/dma_utils.hpp>
#include <lib/dpu/include/dpu_matmul.hpp>
#include <shave/hw/nce/include/hgl_barriers_40xx.hpp>
#include <shave/sync/include/memory_fence.hpp>

#include <cstdint>
#include <new>
#include <math.h>

// fp16 math behind the intel_npu intrinsic rules (intrin_rule_intel_npu.cc). The SAU has a
// scalar fp16 exp; the rest go through float.
static inline half tl_npu_hexp(half x) { return __builtin_shave_sau_exp_f16_r(x); }
static inline half tl_npu_hexp2(half x) { return (half)exp2f((float)x); }
static inline half tl_npu_hlog(half x) { return (half)logf((float)x); }
static inline half tl_npu_hlog2(half x) { return (half)log2f((float)x); }
static inline half tl_npu_hsqrt(half x) { return (half)sqrtf((float)x); }
static inline half tl_npu_hrsqrt(half x) { return (half)(1.0f / sqrtf((float)x)); }
static inline half tl_npu_htanh(half x) { return (half)tanhf((float)x); }
static inline half tl_npu_hfabs(half x) { return (half)fabsf((float)x); }
static inline half tl_npu_hfloor(half x) { return (half)floorf((float)x); }
static inline half tl_npu_hceil(half x) { return (half)ceilf((float)x); }
static inline half tl_npu_hround(half x) { return (half)roundf((float)x); }
// CodeGenC prints tirx.max / tirx.min as max(a, b) / min(a, b).
template <class T> static inline T max(T a, T b) { return a > b ? a : b; }
template <class T> static inline T min(T a, T b) { return a < b ? a : b; }

// Vector fp16 transcendentals: the VAU has EXP/LOG/TANH on half32 only (the v8f16
// forms are not on this SHAVE variant); half16/half8 widen to one half32 op.
#define TL_NPU_VAU_F16(NAME, BUILTIN)                                          \
  static inline half32 tl_npu_h##NAME(half32 x) { return BUILTIN(x); }         \
  static inline half16 tl_npu_h##NAME(half16 x) {                              \
    const half32 w = __builtin_shufflevector(x, x, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
                                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); \
    const half32 e = BUILTIN(w);                                               \
    return __builtin_shufflevector(e, e, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); \
  }                                                                            \
  static inline half8 tl_npu_h##NAME(half8 x) {                                \
    const half32 w = __builtin_shufflevector(x, x, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, \
                                             0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7); \
    const half32 e = BUILTIN(w);                                               \
    return __builtin_shufflevector(e, e, 0, 1, 2, 3, 4, 5, 6, 7);              \
  }
TL_NPU_VAU_F16(exp, __builtin_shave_vau_exp_r_half32)
TL_NPU_VAU_F16(log, __builtin_shave_vau_log_r_half32)
TL_NPU_VAU_F16(tanh, __builtin_shave_vau_tanh_r_half32)
#undef TL_NPU_VAU_F16
#define TL_NPU_VEC_HAS_VAU_EXP 1
#define TL_NPU_VEC_HAS_VAU_LOG 1
#define TL_NPU_VEC_HAS_VAU_TANH 1
#include "npu_rt_vecmath.hpp"

// Whole-vector reductions (the DSL's T.call_extern("tl_npu_vmax" / "tl_npu_vsum", buf[0:N])):
// log2(N) rotate-and-combine steps on the CMU/VAU instead of N lane extracts
// (moviCompile lowers a half-vector extract to one CMU.CP.16 per lane).
static inline half tl_npu_vmax(half32 v) {
  v = __builtin_shave_cmu_max_f16_rr_half32(v, __builtin_shave_cmu_alignvec_rri_half32(v, v, 32));
  v = __builtin_shave_cmu_max_f16_rr_half32(v, __builtin_shave_cmu_alignvec_rri_half32(v, v, 16));
  v = __builtin_shave_cmu_max_f16_rr_half32(v, __builtin_shave_cmu_alignvec_rri_half32(v, v, 8));
  v = __builtin_shave_cmu_max_f16_rr_half32(v, __builtin_shave_cmu_alignvec_rri_half32(v, v, 4));
  v = __builtin_shave_cmu_max_f16_rr_half32(v, __builtin_shave_cmu_alignvec_rri_half32(v, v, 2));
  return v[0];
}
static inline float tl_npu_vmax(float16 v) {
  v = __builtin_shave_cmu_max_f32_rr_float16(v, __builtin_shave_cmu_alignvec_rri_float16(v, v, 32));
  v = __builtin_shave_cmu_max_f32_rr_float16(v, __builtin_shave_cmu_alignvec_rri_float16(v, v, 16));
  v = __builtin_shave_cmu_max_f32_rr_float16(v, __builtin_shave_cmu_alignvec_rri_float16(v, v, 8));
  v = __builtin_shave_cmu_max_f32_rr_float16(v, __builtin_shave_cmu_alignvec_rri_float16(v, v, 4));
  return v[0];
}
static inline float tl_npu_vsum(float16 v) {
  v = v + __builtin_shave_cmu_alignvec_rri_float16(v, v, 32);
  v = v + __builtin_shave_cmu_alignvec_rri_float16(v, v, 16);
  v = v + __builtin_shave_cmu_alignvec_rri_float16(v, v, 8);
  v = v + __builtin_shave_cmu_alignvec_rri_float16(v, v, 4);
  return v[0];
}
static inline half tl_npu_vsum(half32 v) {
  v = v + __builtin_shave_cmu_alignvec_rri_half32(v, v, 32);
  v = v + __builtin_shave_cmu_alignvec_rri_half32(v, v, 16);
  v = v + __builtin_shave_cmu_alignvec_rri_half32(v, v, 8);
  v = v + __builtin_shave_cmu_alignvec_rri_half32(v, v, 4);
  v = v + __builtin_shave_cmu_alignvec_rri_half32(v, v, 2);
  return v[0];
}

// Packed signed int4 (two per byte, LOW nibble first, symmetric: value = (n ^ 8) - 8) to fp16:
// the LSU converts in the load (`LSU.LD512.i4.i8`: 32 bytes -> 64 int8), lo = values 0..31,
// hi = 32..63. The DSL writes `w[r, k:k+32] = tl_npu_i4x64_lo(&q[r, k/2]) * scale` (T.call_extern);
// the fp16 product of |q| <= 8 by an fp16 scale is exact. Same format as npu_dsl's DEQUANT_I4.
static inline half32 tl_npu_i4x64_lo(const void *p) {
  const schar64 v = __builtin_shave_lsu_ld512_i4_i8_r((const char64 *)p);
  return __builtin_convertvector(v.lo, half32);
}
static inline half32 tl_npu_i4x64_hi(const void *p) {
  const schar64 v = __builtin_shave_lsu_ld512_i4_i8_r((const char64 *)p);
  return __builtin_convertvector(v.hi, half32);
}
// The 64 int8 lanes themselves (`q[0:64] = tl_npu_ld_i4x64(p)` then `T.cast(q[0:32], float16x32)`).
static inline schar64 tl_npu_ld_i4x64(const void *p) { return __builtin_shave_lsu_ld512_i4_i8_r((const char64 *)p); }

// One row of int4 g`group` weights -> fp16 (`T.call_extern("handle", "tl_npu_dequant_i4_row", dst, src, scales,
// width, group)`). Blocks of 512 values: the 8 convert-loads and the scale splats are issued before the
// block's 16 stores, so the LSU latency is hidden (the schedule of npu_dsl's DEQUANT_I4 fast path, `E7-63`;
// the generic DSL statement form loaded twice and stored the loads back to CMX: 0.7 cycles per value).
// Requires width % 512 == 0 and group % 64 == 0; the fp16 product of |q| <= 8 by an fp16 scale is exact.
static inline void tl_npu_dequant_i4_row(half *__restrict dst, const unsigned char *__restrict src,
                                         const half *__restrict scales, std::uint32_t width, std::uint32_t group) {
  for (std::uint32_t b = 0; b < width / 512; ++b) {
    schar64 s[8];
    half32 sc[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      s[i] = __builtin_shave_lsu_ld512_i4_i8_r((const char64 *)(src + b * 256 + i * 32));
      sc[i] = (half32)(scales[(b * 512 + i * 64) / group]);
    }
    half32 *d = reinterpret_cast<half32 *>(dst + b * 512);
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      d[2 * i] = __builtin_convertvector(s[i].lo, half32) * sc[i];
      d[2 * i + 1] = __builtin_convertvector(s[i].hi, half32) * sc[i];
    }
  }
}

namespace npu {

// Status word 0 per SHAVE, read by the host runner; word 1 is the detail.
enum : std::uint32_t {
  kStatusOk = 1,
  kStatusCmxExhausted = 2,
  kStatusBadParams = 3,
  kStatusUnsupported = 4,
  kStatusBarrierNotLive = 5, // detail: barrier id whose config never went live
  kStatusBarrierTimeout = 6, // detail: barrier id whose count never reached zero
  kStatusRingRefused = 7,    // detail: reason code (ring_* below)
  kStatusOrderTimeout = 8,   // detail: the DPU submission-order word waited for
  kStatusDpuTimeout = 9,     // detail: the ring use whose DPU workload never completed
  kStatusDmaTimeout = 10,    // detail: which ring fill descriptor never set its watermark (see dma_wait)
};

template <class T> struct Ddr;

template <class T> struct DdrRef {
  std::uint64_t addr;
  Ddr<T> operator&() const { return Ddr<T>{addr}; }
  operator T() const { return *DdrAccessor<T>(addr); }
  DdrRef &operator=(const T &v) {
    DdrAccessor<T> acc(addr);
    acc = v;
    return *this;
  }
};

// A DDR tensor: a 64-bit address. `A[i]` is an element reference and
// `&(A[i])` the address of that element, so generated pointer arithmetic
// stays in 64 bits on the 32-bit SHAVE.
template <class T> struct Ddr {
  std::uint64_t addr;
  explicit Ddr(std::uint64_t a = 0) : addr(a) {}
  DdrRef<T> operator[](std::int64_t i) const {
    return DdrRef<T>{addr + static_cast<std::uint64_t>(i) * sizeof(T)};
  }
};

class Ctx {
public:
  static constexpr std::uint32_t kAlign = 64;
  static constexpr std::uint32_t kMaxDpuN = 1024;
  // sync_grid uses this many rotating hardware barrier ids starting at syncBarrierBase.
  static constexpr std::uint32_t kSyncBarriers = 3;
  static constexpr std::uint32_t kNoHwSync = 0xFFFFFFFFu;

  Ctx(const Environment &env, int numParams, std::uint64_t *params,
      int numTensorParams, std::uint32_t syncBarrierBase = kNoHwSync)
      : env_(env), cmx_(env.getCmxAllocator()), dma_(env.getDmaDrv()),
        sync_(env.getSync()), syncBase_(syncBarrierBase) {
    tStart_ = perf_ticks();
    const auto &info = env.getShaveInfo();
    shave_ = info.getShaveId();
    spt_ = info.getNumShavesPerTile();
    tile_ = info.getTileId();
    tiles_ = info.getNumTiles();
    statusAddr_ = numParams > numTensorParams ? params[numTensorParams] : 0;
    if (numParams < numTensorParams) {
      fail(kStatusBadParams, numParams);
      return;
    }
    cmxFree0_ = cmx_.getAvailableSize();
    desc_ = alloc_hw<DmaDescriptor>(2);
    for (std::uint32_t i = 0; i < kDpuSets; ++i) {
      dpu_[i].desc = alloc_hw<DpuMatMul::DescriptorsBuffer>(1);
      dpu_[i].wt = alloc<std::uint32_t>(4 * kMaxDpuN);
      if (ok_)
        dpu_[i].mm = new (dpu_[i].store) DpuMatMul(env_.getDpuDrv(), dpu_[i].desc, dpu_[i].wt);
    }
    if (ok_) {
      mm_ = dpu_[0].mm;
      dpuDesc_ = dpu_[0].desc;
    }
  }

  bool ok() const { return ok_; }
  std::int32_t tile_id() const { return tile_; }
  std::int32_t shave_id() const { return shave_; }
  std::int32_t num_tiles() const { return tiles_; }
  std::int32_t shaves_per_tile() const { return spt_; }

  // Worker-private CMX: one allocation of spt_ slots, this SHAVE takes slot shave_.
  template <class T> T *alloc(std::uint32_t count) {
    const std::uint32_t bytes = alignUp(count * sizeof(T), kAlign);
    auto *base = reinterpret_cast<std::uint8_t *>(cmx_.alloc<kAlign>(bytes * spt_));
    if (base == nullptr) {
      fail(kStatusCmxExhausted, bytes * spt_);
      return nullptr;
    }
    return reinterpret_cast<T *>(base + bytes * shave_);
  }
  // Tile-shared CMX: one allocation, the same address on every SHAVE of the tile
  // (identical allocation sequences) and the same offset on every tile, which is
  // what a multicast fill needs.
  template <class T> T *alloc_shared(std::uint32_t count) {
    const std::uint32_t bytes = alignUp(count * sizeof(T), kAlign);
    auto *base = cmx_.alloc<kAlign>(bytes);
    if (base == nullptr) {
      fail(kStatusCmxExhausted, bytes);
      return nullptr;
    }
    return reinterpret_cast<T *>(base);
  }

  void dma_load(void *dst, Ddr<void> src, std::uint32_t width, std::uint32_t rows,
                std::uint32_t dstStride, std::uint32_t srcStride) {
    dma_load(dst, src.addr, width, rows, dstStride, srcStride);
  }
  template <class T>
  void dma_load(T *dst, Ddr<T> src, std::uint32_t width, std::uint32_t rows,
                std::uint32_t dstStride, std::uint32_t srcStride) {
    dma_load(static_cast<void *>(dst), src.addr, width, rows, dstStride, srcStride);
  }
  template <class T>
  void dma_store(Ddr<T> dst, const T *src, std::uint32_t width, std::uint32_t rows,
                 std::uint32_t dstStride, std::uint32_t srcStride) {
    dma_store(dst.addr, static_cast<const void *>(src), width, rows, dstStride, srcStride);
  }

  // out[M,N] = a[M,K] . bt[N,K]^T, fp16, one DPU workload. Overwrites out.
  // spatialX 0 keeps DpuMatMul::init's M split; mpe -1 keeps its stencil, else
  // 0/1/2 = CUBOID_4x16/8x16/16x16 as (nthw_ntk, odu_nthw) pairs. The ODU nthw
  // must change with nthw_ntk or output rows come back permuted (npu_dsl Round 705).
  void dpu_matmul(half *out, const half *a, const half *bt, std::uint32_t M,
                  std::uint32_t N, std::uint32_t K, std::uint32_t spatialX = 0,
                  std::int32_t mpe = -1) {
    if (!dpu_prepare(M, N, K, spatialX, mpe))
      return;
    // No barriers on a synchronous workload (a ring use may have set them).
    mm_->setConsumerBarriers(0, 0);
    mm_->setProducerBarriers(0, 0);
    mm_->run(out, const_cast<half *>(a), const_cast<half *>(bt));
    dpu_account();
  }

  // out[M, N] = a[M, K] . (q * s)^T for packed int4 q [N, K] (two per byte, low nibble first) and fp16
  // group scales s [N, K/G]: ONE mixed-mode DPU workload (fp16 activations x int4 weights read in place,
  // no dequantisation) over the M x ng pseudo-rows a[m, g-th group] against C = N x ng output channels
  // (channel g * N + n = row n's group g, through the weight table's per-channel offsets), then the SHAVE
  // keeps the diagonal blocks and sums them with the scales (q_reduce). The DPU computes ng times the
  // needed products, which at decode's M = 1 is free: the workload is bound by reading the weights.
  // VPUX's merged FC; ported from npu_dsl's fused_rt.hpp matmul_qg / q_reduce (plan section 0 amendment).
  // Scratch (part [M x ng, C] fp16, sT [ng, N] fp16) is allocated on first use at that call's size.
  // `sStride` 0: s is row-major [N, ng] and the SHAVE transposes it while the DPU runs; > 0: s is group-major
  // [ng, sStride] (the packer's layout) and is read in place -- the transpose cost ~70 ns per row when ng % 8
  // != 0 (scalar path), 210 us per SHAVE in the MoE's down GEMVs (M3).
  void dpu_matmul_i4g(half *out, const half *a, const std::uint8_t *q, const half *s, std::uint32_t M,
                      std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride = 0) {
    dpu_matmul_qg(out, a, q, s, M, N, K, G, 4, sStride);
  }
  // int8 weights (one signed byte per value, rows K bytes apart), the DPU's native mixed mode; PPE 1/256.
  void dpu_matmul_i8g(half *out, const half *a, const std::int8_t *q, const half *s, std::uint32_t M,
                      std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride = 0) {
    dpu_matmul_qg(out, a, reinterpret_cast<const std::uint8_t *>(q), s, M, N, K, G, 8, sStride);
  }
  // The scales packed after the weight rows of the chunk (gemm_i4's scales_row): addressed as bytes.
  void dpu_matmul_i4g(half *out, const half *a, const std::uint8_t *q, const std::uint8_t *s, std::uint32_t M,
                      std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride) {
    dpu_matmul_qg(out, a, q, reinterpret_cast<const half *>(s), M, N, K, G, 4, sStride);
  }
  void dpu_matmul_i8g(half *out, const half *a, const std::int8_t *q, const std::int8_t *s, std::uint32_t M,
                      std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride) {
    dpu_matmul_qg(out, a, reinterpret_cast<const std::uint8_t *>(q), reinterpret_cast<const half *>(s), M, N, K, G, 8,
                  sStride);
  }
  void dpu_matmul_qg(half *out, const half *a, const std::uint8_t *q, const half *s, std::uint32_t M,
                     std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t bits, std::uint32_t sStride) {
    if (!ok_)
      return;
    const std::uint32_t ng = K / G, C = N * ng;
    if (K % G != 0 || G % 16 != 0 || N % 16 != 0 || C > kMaxDpuN) {
      fail(kStatusUnsupported, C);
      return;
    }
    if (i4Part_ == nullptr) {
      i4PartElems_ = M * ng * C;
      i4STElems_ = ng * N;
      i4Part_ = alloc<half>(i4PartElems_);
      i4ST_ = alloc<half>(i4STElems_);
      i4Desc_ = alloc_hw<DpuMatMul::DescriptorsBuffer>(1);
      i4Wt_ = alloc<std::uint32_t>(4 * kMaxDpuN);
      if (!ok_)
        return;
      i4Mm_ = new (i4Store_) DpuMatMul(env_.getDpuDrv(), i4Desc_, i4Wt_);
    } else if (M * ng * C > i4PartElems_ || ng * N > i4STElems_) {
      fail(kStatusUnsupported, M * ng * C);
      return;
    }
    const float partScale = bits == 8 ? kI8PartScale : kI4PartScale;
    const std::uint64_t key = (static_cast<std::uint64_t>(M) << 40) | (static_cast<std::uint64_t>(N) << 20) |
                              (static_cast<std::uint64_t>(K) << 8) | ((G >> 4) << 1) | (bits == 8);
    if (key != i4Key_) {
      // weight table rows {byte offset of (row n, group g) in q, 0, PPE multiplier, 0}, channel g * N + n
      union { float f; std::uint32_t u; } sc;
      sc.f = partScale;
      const std::uint32_t rowB = K * bits / 8, grpB = G * bits / 8;
      for (std::uint32_t g = 0; g < ng; ++g)
        for (std::uint32_t n = 0; n < N; ++n) {
          std::uint32_t *w = i4Wt_ + 4 * (g * N + n);
          w[0] = n * rowB + g * grpB;
          w[1] = 0;
          w[2] = sc.u;
          w[3] = 0;
        }
      i4Mm_->initMixedDescOnly(M * ng, C, G);   // sets the int8 weight mode
      if (bits == 4)
        i4Desc_->invariant.tensor_mode.tensor_mode_bf.wmode =
            static_cast<std::uint32_t>(npu40xx::VpuInputTensorDType::I4);
      i4Key_ = key;
    }
    i4Mm_->setConsumerBarriers(0, 0);
    i4Mm_->setProducerBarriers(0, 0);
    const std::uint32_t t0 = perf_ticks();
    i4Mm_->startMixed(i4Part_, const_cast<half *>(a), reinterpret_cast<std::int8_t *>(const_cast<std::uint8_t *>(q)));
    // s [N, ng] -> sT [ng, N] while the DPU runs: 8x8 vector transposes (npu_dsl's transpose_h) when the
    // shape allows, a scalar loop otherwise (1024 strided scalar copies cost ~9 us at N=64, ng=16).
    // Group-major s (sStride > 0) needs none of it.
    const half *sT = s;
    std::uint32_t sRow = sStride;
    if (sStride == 0) {
      if (N % 8 == 0 && ng % 8 == 0)
        transpose_h(i4ST_, s, N, ng);
      else
        for (std::uint32_t n = 0; n < N; ++n)
          for (std::uint32_t g = 0; g < ng; ++g)
            i4ST_[g * N + n] = s[n * ng + g];
      sT = i4ST_;
      sRow = N;
    }
    const std::uint32_t t1 = perf_ticks();
    i4Mm_->wait();
    const std::uint32_t t2 = perf_ticks();
    dpuCount_ += 1;
    dpuIduTicks_ += static_cast<volatile const DpuMatMul::DescriptorsBuffer *>(i4Desc_)->hwpStats.idu_workload_duration;
    dpuOduTicks_ += static_cast<volatile const DpuMatMul::DescriptorsBuffer *>(i4Desc_)->hwpStats.odu_workload_duration;
    q_reduce(out, i4Part_, sT, sRow, M, N, ng, 1.0f / partScale);
    const std::uint32_t t3 = perf_ticks();
    ringTicks_[kTickSubmit] += (t1 - t0) + (t3 - t2);   // SHAVE side: start + transpose, reduce
    ringTicks_[kTickDpuWait] += t2 - t1;                // DPU time not hidden behind the transpose
  }

  // dst[cols, rows] = src[rows, cols]^T in 8x8 blocks of half8 shuffles (npu_dsl's fused_rt.hpp transpose_h);
  // rows % 8 == 0 and cols % 8 == 0.
  static void transpose_h(half *__restrict dst, const half *__restrict src, std::uint32_t rows, std::uint32_t cols) {
    for (std::uint32_t r0 = 0; r0 < rows; r0 += 8)
      for (std::uint32_t c0 = 0; c0 < cols; c0 += 8) {
        half8 r[8];
        for (int i = 0; i < 8; ++i)
          r[i] = *reinterpret_cast<const half8 *>(src + (r0 + i) * cols + c0);
        const half8 t0 = __builtin_shufflevector(r[0], r[1], 0, 8, 1, 9, 2, 10, 3, 11), t1 = __builtin_shufflevector(r[0], r[1], 4, 12, 5, 13, 6, 14, 7, 15);
        const half8 t2 = __builtin_shufflevector(r[2], r[3], 0, 8, 1, 9, 2, 10, 3, 11), t3 = __builtin_shufflevector(r[2], r[3], 4, 12, 5, 13, 6, 14, 7, 15);
        const half8 t4 = __builtin_shufflevector(r[4], r[5], 0, 8, 1, 9, 2, 10, 3, 11), t5 = __builtin_shufflevector(r[4], r[5], 4, 12, 5, 13, 6, 14, 7, 15);
        const half8 t6 = __builtin_shufflevector(r[6], r[7], 0, 8, 1, 9, 2, 10, 3, 11), t7 = __builtin_shufflevector(r[6], r[7], 4, 12, 5, 13, 6, 14, 7, 15);
        const half8 u0 = __builtin_shufflevector(t0, t2, 0, 1, 8, 9, 2, 3, 10, 11), u1 = __builtin_shufflevector(t0, t2, 4, 5, 12, 13, 6, 7, 14, 15);
        const half8 u2 = __builtin_shufflevector(t1, t3, 0, 1, 8, 9, 2, 3, 10, 11), u3 = __builtin_shufflevector(t1, t3, 4, 5, 12, 13, 6, 7, 14, 15);
        const half8 u4 = __builtin_shufflevector(t4, t6, 0, 1, 8, 9, 2, 3, 10, 11), u5 = __builtin_shufflevector(t4, t6, 4, 5, 12, 13, 6, 7, 14, 15);
        const half8 u6 = __builtin_shufflevector(t5, t7, 0, 1, 8, 9, 2, 3, 10, 11), u7 = __builtin_shufflevector(t5, t7, 4, 5, 12, 13, 6, 7, 14, 15);
        half8 *d = reinterpret_cast<half8 *>(dst + c0 * rows + r0);
        const std::uint32_t st = rows / 8;
        d[0 * st] = __builtin_shufflevector(u0, u4, 0, 1, 2, 3, 8, 9, 10, 11);
        d[1 * st] = __builtin_shufflevector(u0, u4, 4, 5, 6, 7, 12, 13, 14, 15);
        d[2 * st] = __builtin_shufflevector(u1, u5, 0, 1, 2, 3, 8, 9, 10, 11);
        d[3 * st] = __builtin_shufflevector(u1, u5, 4, 5, 6, 7, 12, 13, 14, 15);
        d[4 * st] = __builtin_shufflevector(u2, u6, 0, 1, 2, 3, 8, 9, 10, 11);
        d[5 * st] = __builtin_shufflevector(u2, u6, 4, 5, 6, 7, 12, 13, 14, 15);
        d[6 * st] = __builtin_shufflevector(u3, u7, 0, 1, 2, 3, 8, 9, 10, 11);
        d[7 * st] = __builtin_shufflevector(u3, u7, 4, 5, 6, 7, 12, 13, 14, 15);
      }
  }

  // out[m, n] = inv x sum_g sT[g, n] x part[(m, g), g * N + n] (sT rows `sRow` apart): four 16-column blocks
  // at a time (independent accumulators; the sum order per output is fixed, so the result is deterministic).
  // N % 16 == 0.
  static void q_reduce(half *__restrict out, const half *__restrict part, const half *__restrict sT, std::uint32_t sRow,
                       std::uint32_t M, std::uint32_t N, std::uint32_t ng, float inv) {
    const std::uint32_t C = N * ng;
    for (std::uint32_t m = 0; m < M; ++m) {
      std::uint32_t n0 = 0;
      for (; n0 + 64 <= N; n0 += 64) {
        float16 a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (std::uint32_t g = 0; g < ng; ++g) {
          const half *p = part + (m * ng + g) * C + g * N + n0;
          const half *w = sT + g * sRow + n0;
          a0 += __builtin_convertvector(*reinterpret_cast<const half16 *>(p), float16) *
                __builtin_convertvector(*reinterpret_cast<const half16 *>(w), float16);
          a1 += __builtin_convertvector(*reinterpret_cast<const half16 *>(p + 16), float16) *
                __builtin_convertvector(*reinterpret_cast<const half16 *>(w + 16), float16);
          a2 += __builtin_convertvector(*reinterpret_cast<const half16 *>(p + 32), float16) *
                __builtin_convertvector(*reinterpret_cast<const half16 *>(w + 32), float16);
          a3 += __builtin_convertvector(*reinterpret_cast<const half16 *>(p + 48), float16) *
                __builtin_convertvector(*reinterpret_cast<const half16 *>(w + 48), float16);
        }
        half *o = out + m * N + n0;
        *reinterpret_cast<half16 *>(o) = __builtin_convertvector(a0 * inv, half16);
        *reinterpret_cast<half16 *>(o + 16) = __builtin_convertvector(a1 * inv, half16);
        *reinterpret_cast<half16 *>(o + 32) = __builtin_convertvector(a2 * inv, half16);
        *reinterpret_cast<half16 *>(o + 48) = __builtin_convertvector(a3 * inv, half16);
      }
      for (; n0 < N; n0 += 16) {
        float16 acc = 0.f;
        for (std::uint32_t g = 0; g < ng; ++g)
          acc += __builtin_convertvector(*reinterpret_cast<const half16 *>(part + (m * ng + g) * C + g * N + n0), float16) *
                 __builtin_convertvector(*reinterpret_cast<const half16 *>(sT + g * sRow + n0), float16);
        *reinterpret_cast<half16 *>(out + m * N + n0) = __builtin_convertvector(acc * inv, half16);
      }
    }
  }

  // ---- Streamed ping-pong ring (plan Phase 3b; the flow of
  // docs/examples/pingpong_multitile/README.md sections 1-3 and 6c) ----
  //
  // One operand streams through `slots` CMX slots shared by the tile's SHAVEs
  // (slots = depth x spt): chunk c is the work of SHAVE c % spt, whose use u of
  // the ring lives in slot (c % spt) + spt * (u % depth). Tile 0's SHAVE s fills
  // its slots for every tile with ONE multicast DMA each, split in two halves on
  // ports 0 and 1. Per (slot, parity of u / depth) two hardware barriers: READY
  // (the two halves produce, every tile's DPU workload consumes) and FREE (every
  // tile's DPU workload produces, the issuer consumes before it refills). A fill
  // is issued `depth` uses ahead and gated on the slot's previous occupant's FREE,
  // so with depth > 1 the fill's fixed latency overlaps the DPU and SHAVE work of
  // the uses in between (the flash phase: 0.2 us workloads behind 2.5 us fills).
  // Barrier ids alternate by parity so a config is never more than one deep in
  // the barrier FIFO and "consumer count != 0" means "this use's config is live";
  // the ids of use u are re-armed by the fill of use u (after it consumed the
  // FREE of use u - depth, the last consumer of those ids). The tile's SHAVEs
  // submit to the one DPU FIFO in chunk order through CMX order words.
  struct Ring {
    std::uint8_t *slotBase;   // shared: slots x slotBytes, same offset on every tile
    std::uint32_t slotBytes;
    std::uint32_t slots;
    std::uint32_t depth;      // slots / spt: this SHAVE's slots, fills in flight ahead
    std::int32_t port;        // -1: each fill in two halves on ports 0 and 1; 0/1: whole fill on that port;
                              // 2/3: whole fill on port (use & 1) / (use + 1) & 1, so a ring of depth 2 has one
                              // fill in flight per port (the toolkit's chain takes one job per port per SHAVE: a
                              // second start on a port waits for the first; M3's num_stages=4 arm spent 280 us there)
    bool multicast;           // true: tile 0 fills every tile's slot; false: each tile fills its own (split stream)
    bool local;               // split stream read by the SHAVE that fills it: no barriers (the fill's DMA
                              // watermark is READY; program order is FREE), no cross-tile coupling
    std::uint32_t total;      // chunks streamed in this ring
    std::uint32_t myUses;     // chunks of this SHAVE
    std::uint32_t readyBase;  // READY id = readyBase + slot * 2 + parity
    std::uint32_t freeBase;   // FREE id  = freeBase  + slot * 2 + parity
    volatile std::uint32_t *order; // shared: one word per SHAVE, uses submitted
    DmaDescriptor *fill;      // issuer's two half descriptors per owned slot
    std::uint32_t armedUpTo;  // issuer: ids of uses < armedUpTo are configured
    DpuMatMul::DescriptorsBuffer *desc[2]; // descriptor set of use u at u & 1 (submit-ahead is one use deep)
    bool armed;
  };
  enum : std::uint32_t {
    kRingSlotsNeShaves = 1,
    kRingOneTile = 2,
    kRingBarrierPool = 3,
  };

  std::uint32_t ring_slot_index(const Ring &r, std::uint32_t use) const {
    return shave_ + spt_ * (use % r.depth);
  }
  std::uint32_t ring_parity(const Ring &r, std::uint32_t use) const { return (use / r.depth) & 1u; }
  std::uint32_t ring_ready(const Ring &r, std::uint32_t use) const {
    return r.readyBase + ring_slot_index(r, use) * 2 + ring_parity(r, use);
  }
  std::uint32_t ring_free(const Ring &r, std::uint32_t use) const {
    return r.freeBase + ring_slot_index(r, use) * 2 + ring_parity(r, use);
  }
  std::uint8_t *ring_slot(const Ring &r, std::uint32_t use) const {
    return r.slotBase + ring_slot_index(r, use) * r.slotBytes;
  }
  template <class T> T *ring_slot(const Ring &r, std::uint32_t use) const {
    return reinterpret_cast<T *>(ring_slot(r, use));
  }
  DmaDescriptor *ring_fill_desc(const Ring &r, std::uint32_t use) const { return r.fill + 2 * (use % r.depth); }
  std::uint32_t ring_fill_port(const Ring &r, std::uint32_t use) const {
    return r.port >= 2 ? ((use + static_cast<std::uint32_t>(r.port)) & 1u) : static_cast<std::uint32_t>(r.port);
  }
  std::uint32_t ring_uses(const Ring &r) const { return r.myUses; }
  std::uint32_t ring_chunk(std::uint32_t use) const { return use * spt_ + shave_; }

  // `slotBase` is the compiler's shared buffer of slots x slotBytes (same address
  // on every SHAVE of the tile, same offset on every tile). Allocates the order
  // words (shared) and the issuer's fill descriptors; arms the ids of the first
  // 2 x depth uses on tile 0 (every (slot, parity) once); one grid rendezvous so
  // every config is live before any gate. `readyBase`/`freeBase` come from the
  // compiler's barrier assignment (2 x slots ids each).
  Ring ring_begin(void *slotBase, std::uint32_t slotBytes, std::uint32_t slots, std::uint32_t total,
                  std::uint32_t readyBase, std::uint32_t freeBase, std::int32_t port = -1, bool multicast = true,
                  bool local = false) {
    Ring r{};
    r.slotBase = static_cast<std::uint8_t *>(slotBase);
    r.slotBytes = slotBytes;
    r.slots = slots;
    r.total = total;
    r.readyBase = readyBase;
    r.freeBase = freeBase;
    r.port = port < 0 ? -1 : (port >= 2 ? 2 + (port & 1) : (port & 1));
    r.multicast = multicast && !local;
    r.local = local;
    if (!ok_)
      return r;
    if (slots == 0 || static_cast<std::int32_t>(slots) % spt_ != 0) {
      fail(kStatusRingRefused, kRingSlotsNeShaves);
      return r;
    }
    r.depth = slots / static_cast<std::uint32_t>(spt_);
    if (tiles_ < 2) {
      fail(kStatusRingRefused, kRingOneTile);
      return r;
    }
    r.order = alloc_shared<std::uint32_t>(16); // 64 B: one line, two words used
    r.fill = alloc_hw<DmaDescriptor>(2 * r.depth);
    if (!ok_)
      return r;
    r.myUses = total > static_cast<std::uint32_t>(shave_)
                   ? (total - shave_ + spt_ - 1) / spt_
                   : 0;
    if (shave_ == 0) {
      for (std::int32_t s = 0; s < spt_; ++s)
        r.order[s] = 0;
      flushCmxWrites();
    }
    if (local) {
      r.armed = ok_;   // no ids to configure, no rendezvous
      return r;
    }
    if (tile_ == 0) {
      for (std::uint32_t u = 0; u < 2 * r.depth && u < r.myUses; ++u)
        ring_config(r, u);
      r.armedUpTo = 2 * r.depth < r.myUses ? 2 * r.depth : r.myUses;
      for (std::uint32_t u = 0; u < r.armedUpTo; ++u)
        if (!barrier_check_live(ring_ready(r, u), 0, ring_consumers(r)) || !barrier_check_live(ring_free(r, u), 0, ring_free_consumers(r)))
          return r;
    }
    sync_grid();
    r.armed = ok_;
    return r;
  }

  // Barrier counts. Multicast: one issuer's fill (1 or 2 halves) produces READY, every tile's
  // reader consumes; every tile's reader produces FREE, the issuer consumes. Split stream: the
  // ids are still shared by the tiles, so every tile's fill produces READY and every tile's
  // issuer consumes FREE (the counts scale by the tile count; a tile waits for its peers' fills).
  std::uint8_t ring_ready_producers(const Ring &r) const {
    return static_cast<std::uint8_t>((r.port < 0 ? 2 : 1) * (r.multicast ? 1 : tiles_));
  }
  std::uint8_t ring_consumers(const Ring &) const { return static_cast<std::uint8_t>(tiles_); }
  std::uint8_t ring_free_consumers(const Ring &r) const { return static_cast<std::uint8_t>(r.multicast ? 1 : tiles_); }
  void ring_config(const Ring &r, std::uint32_t use) {
    barrier_config(ring_ready(r, use), ring_ready_producers(r), ring_consumers(r), false);
    barrier_config(ring_free(r, use), ring_consumers(r), ring_free_consumers(r), false);
  }
  bool ring_issuer(const Ring &r) const { return r.multicast ? tile_ == 0 : true; }

  // Issuer: configure the ids of `use` once, at whichever of its fill or its submit
  // comes first. Valid because ids(use) == ids(use - 2 depth), whose READY every DPU
  // consumed and whose FREE the fill of use - depth consumed, and that fill precedes
  // both callers in every lowered program.
  void ring_arm(Ring &r, std::uint32_t use) {
    if (tile_ != 0 || use < r.armedUpTo || use >= r.myUses)
      return;
    for (std::uint32_t u = r.armedUpTo; u <= use; ++u)
      ring_config(r, u);
    r.armedUpTo = use + 1;
  }

  // Issuer (tile 0, SHAVE s): fill the slot of my `use` with chunk `use * spt + s`
  // from `src` (`rows` rows of `width` bytes, `srcStride` bytes apart in DDR;
  // contiguous in the slot), two halves on ports 0 and 1, both producing
  // READY[use]. The halves split the rows (the bytes when there is one row). For
  // use >= depth it first waits until every tile's DPU has produced FREE[use - depth]
  // (the slot's previous occupant), then re-arms this use's ids when they were used
  // before (use >= 2 x depth).
  void ring_fill(Ring &r, std::uint32_t use, std::uint64_t src, std::uint32_t width,
                 std::uint32_t rows, std::uint32_t srcStride) {
    if (!ok_ || !ring_issuer(r) || use >= r.myUses)
      return;
    DmaDescriptor *d = ring_fill_desc(r, use);
    std::uint32_t t0 = perf_ticks();
    if (use >= r.depth) {
      if (!r.local) {
        const std::uint32_t freeId = ring_free(r, use - r.depth);
        if (!barrier_wait_produced(freeId))
          return;
        barrier_consume(freeId);
      }
      // The previous halves have completed (READY lifted); the chains still need the wait.
      if (!dma_wait(d, 1) || (r.port < 0 && !dma_wait(d + 1, 2)))
        return;
    }
    std::uint32_t t1 = perf_ticks();
    ringTicks_[kTickFreeWait] += t1 - t0;
    if (!r.local) {
      ring_arm(r, use);
      // A split stream's issuer on another tile produces on ids tile 0 arms: wait for the config.
      if (!r.multicast && tile_ != 0 && !barrier_wait_live(ring_ready(r, use)))
        return;
    }
    // One descriptor of this SHAVE in flight per port: a second start while another
    // ring's fill is still running on the port left one half unproduced (flash phase,
    // two rings; READY stuck at p=1). Before initDesc below: it clears the watermark
    // this wait reads when the previous descriptor is this ring's own.
    if (r.port < 0) {
      port_retire(0);
      port_retire(1);
    } else {
      port_retire(ring_fill_port(r, use));
    }
    if (!ok_)
      return;
    const std::uint32_t mask = (1u << tiles_) - 1u;
    std::uint8_t *dst0 = ring_slot(r, use);
    // multicast: the CMX address that writes every tile's copy; split: this tile's own slot
    auto cmx = [&](std::uint8_t *p) {
      return reinterpret_cast<std::uint32_t>(r.multicast ? convertToCmxMulticastAddr(p, mask) : p);
    };
    const std::uint32_t ready = r.local ? 0 : ring_ready(r, use);
    DmaDrv::initDesc(d);
    if (r.port >= 0) {
      // whole fill, one descriptor on this ring's port (several streams share the two ports)
      const std::uint32_t port = ring_fill_port(r, use);
      config_copy(d, cmx(dst0), src, width, rows, width, srcStride);
      if (!r.local)
        DmaDrv::setProducerBarrier(d, ready);
      DmaDrv::setConsumerBarriers(d, 0, 0);
      dma_.start(d, port);
      lastOnPort_[port] = d;
      ringTicks_[kTickFillIssue] += perf_ticks() - t1;
      return;
    }
    DmaDrv::initDesc(d + 1);
    if (rows <= 1) {
      const std::uint32_t half0 = width / 2;
      configDmaCopy(d, cmx(dst0), src, half0);
      configDmaCopy(d + 1, cmx(dst0 + half0), src + half0, width - half0);
    } else {
      const std::uint32_t rows0 = rows / 2;
      config_copy(d, cmx(dst0), src, width, rows0, width, srcStride);
      config_copy(d + 1, cmx(dst0 + rows0 * width), src + static_cast<std::uint64_t>(rows0) * srcStride, width,
                  rows - rows0, width, srcStride);
    }
    if (!r.local) {
      DmaDrv::setProducerBarrier(d, ready);
      DmaDrv::setProducerBarrier(d + 1, ready);
    }
    DmaDrv::setConsumerBarriers(d, 0, 0);
    DmaDrv::setConsumerBarriers(d + 1, 0, 0);
    dma_.start(d, 0);
    dma_.start(d + 1, 1);
    lastOnPort_[0] = d;
    lastOnPort_[1] = d + 1;
    ringTicks_[kTickFillIssue] += perf_ticks() - t1;
  }
  template <class T>
  void ring_fill(Ring &r, std::uint32_t use, Ddr<T> src, std::uint32_t width, std::uint32_t rows,
                 std::uint32_t srcStride) {
    ring_fill(r, use, src.addr, width, rows, srcStride);
  }

  // Every SHAVE: submit the DPU workload of my `use`, gated on READY[use] (in
  // hardware, so it may be queued before the chunk has landed) and producing
  // FREE[use], in chunk order after the other SHAVE's previous chunk. The caller
  // may submit use u+1 right after use u completes (submit-ahead, README 6c fact
  // 3), before its FREE wait, fill kick and store, so the DPU FIFO never drains
  // while a SHAVE does its own work; that needs the output buffer double-buffered
  // by use parity. The ids of a use are armed by ring_arm at its fill or its
  // submit, whichever the program reaches first.
  void ring_dpu_submit(Ring &r, std::uint32_t use, half *out, const half *a, const half *bt,
                       std::uint32_t M, std::uint32_t N, std::uint32_t K,
                       std::uint32_t spatialX = 0, std::int32_t mpe = -1) {
    if (!ok_ || !r.armed || use >= r.myUses)
      return;
    ring_arm(r, use);
    const std::uint32_t ready = ring_ready(r, use), freeId = ring_free(r, use);
    std::uint32_t t0 = perf_ticks();
    if (!barrier_wait_live(ready) || !barrier_wait_live(freeId))
      return;
    std::uint32_t t1 = perf_ticks();
    ringTicks_[kTickLive] += t1 - t0;
    // Submission order P0(u), P1(u), P0(u+1), ...: SHAVE 0 needs SHAVE 1 at `use`
    // submissions, SHAVE 1 needs SHAVE 0 at `use + 1`.
    const std::uint32_t prev = (shave_ + spt_ - 1) % spt_;
    if (!order_wait(r.order + prev, shave_ == 0 ? use : use + 1))
      return;
    t0 = perf_ticks();
    ringTicks_[kTickOrder] += t0 - t1;
    if (!dpu_prepare(M, N, K, spatialX, mpe))
      return;
    mm_->setConsumerBarrier(ready);
    mm_->setProducerBarrier(freeId);
    mm_->start(out, const_cast<half *>(a), const_cast<half *>(bt));
    r.desc[use & 1u] = dpuDesc_;
    r.order[shave_] = use + 1;
    flushCmxWrites();
    ringTicks_[kTickSubmit] += perf_ticks() - t0;
  }
  // A slot read by the SHAVE instead of the DPU (a dequant, a norm): the SHAVE is the
  // READY consumer and the FREE producer, one per tile per slot, so the same counts hold.
  void ring_slot_acquire(Ring &r, std::uint32_t use) {
    if (!ok_ || !r.armed || use >= r.myUses)
      return;
    const std::uint32_t t0 = perf_ticks();
    if (r.local) {
      // my own fill: its DMA watermark is the READY
      DmaDescriptor *d = ring_fill_desc(r, use);
      if (!dma_wait(d, 7) || (r.port < 0 && !dma_wait(d + 1, 8)))
        return;
    } else {
      ring_arm(r, use);
      const std::uint32_t ready = ring_ready(r, use);
      if (!barrier_wait_live(ready) || !barrier_wait_produced(ready))
        return;
      barrier_consume(ready);
    }
    // the acquire wait (the fill landing) is counted with the submit path's barrier-live wait (counter 0), apart
    // from the DPU wait (counter 3), so a SHAVE-read ring's DMA and DPU waits are told apart in the status words
    ringTicks_[kTickLive] += perf_ticks() - t0;
  }
  void ring_slot_release(Ring &r, std::uint32_t use) {
    if (!ok_ || !r.armed || use >= r.myUses || r.local)
      return;
    barrier_produce(ring_free(r, use));
  }

  void ring_dpu_wait(Ring &r, std::uint32_t use) {
    if (!ok_ || !r.armed || use >= r.myUses)
      return;
    // The driver's wait spins forever on a workload whose READY gate never opens; the
    // same watermark (DpuDrvBase::wait) with a bound, so a stuck gate is a report.
    DpuMatMul::DescriptorsBuffer *desc = r.desc[use & 1u];
    volatile const std::uint16_t *watermark = &desc->hwpStats.odu_workload_id;
    const std::uint16_t expected = desc->variant.hwp_wload_id.hwp_wload_id_bf.wload_id;
    const std::uint32_t t0 = perf_ticks();
    for (std::uint32_t spins = 0; spins < kBarrierSpinLimit; ++spins) {
      if (*watermark == expected) {
        ringTicks_[kTickDpuWait] += perf_ticks() - t0;
        dpu_account(desc);
        return;
      }
    }
    fail(kStatusDpuTimeout, use);
    // Diagnostic detail: READY id and its (p, c) counters, FREE id's (p, c).
    std::uint8_t rp = 0, rc = 0, fp = 0, fc = 0;
    hw::nce::barriers::hgl::getCounters(ring_ready(r, use), &rp, &rc);
    hw::nce::barriers::hgl::getCounters(ring_free(r, use), &fp, &fc);
    detail_ = (ring_ready(r, use) << 24) | (static_cast<std::uint32_t>(use & 0xff) << 16) |
              (static_cast<std::uint32_t>(rp & 0xf) << 12) | (static_cast<std::uint32_t>(rc & 0xf) << 8) |
              (static_cast<std::uint32_t>(fp & 0xf) << 4) | static_cast<std::uint32_t>(fc & 0xf);
  }

  // Issuer: the last fills must be retired on their chains before the kernel ends,
  // and the last FREE of every owned slot consumed so every id is quiescent (a
  // later ring may reuse the ids).
  void ring_end(Ring &r) {
    if (!ok_ || !ring_issuer(r) || r.myUses == 0)
      return;
    for (std::uint32_t k = 0; k < r.depth && k < r.myUses; ++k) {
      DmaDescriptor *d = r.fill + 2 * k;
      if (!dma_wait(d, 3) || (r.port < 0 && !dma_wait(d + 1, 4)))
        return;
      for (int p = 0; p < 2; ++p)
        if (lastOnPort_[p] == d || lastOnPort_[p] == d + 1)
          lastOnPort_[p] = nullptr;
    }
    if (r.local)
      return;
    const std::uint32_t first = r.myUses > r.depth ? r.myUses - r.depth : 0;
    for (std::uint32_t u = first; u < r.myUses; ++u) {
      const std::uint32_t freeId = ring_free(r, u);
      if (!barrier_wait_produced(freeId))
        return;
      barrier_consume(freeId);
    }
  }

  // Barriers. DMA is synchronous in Phase 1-2, so only SHAVE-side writes need
  // publishing: CMX writes to the tile, direct DDR stores out of the data L1.
  void sync_grid() {
    flushCmxWrites();
    sync::memoryFence();
    if (syncBase_ == kNoHwSync) {
      sync_.waitGlobal();
    } else {
      hw_sync_grid();
    }
  }
  void sync_tile() {
    flushCmxWrites();
    sync::memoryFence();
    sync_.waitLocal();
  }

  // Hardware barrier primitives (NCE barrier registers; ids from the job's pool).
  // Every wait is bounded: a barrier that never goes live or never drains
  // reports kStatusBarrierNotLive / kStatusBarrierTimeout with the id and the
  // kernel returns, instead of hanging the launch. The bound is far above any
  // legitimate wait (a grid barrier completes in a few microseconds).
  static constexpr std::uint32_t kBarrierSpinLimit = 50u * 1000u * 1000u;
  // A pushed config goes live within microseconds; this bound is a few hundred ms.
  static constexpr std::uint32_t kConfigSpinLimit = 2u * 1000u * 1000u;

  bool barrier_config(std::uint32_t id, std::uint8_t producers, std::uint8_t consumers,
                      bool waitLive) {
    hw::nce::barriers::hgl::pushCounters(id, producers, consumers);
    if (!waitLive)
      return true;
    return barrier_check_live(id, producers, consumers);
  }
  // True once the barrier's config is live. Only the consumer side is checked: a
  // dead barrier reads (0, 0), while a fast peer may already have PRODUCED on a
  // live one before a slow peer looks (seen under profiling skew: an exact
  // (p, c) check spun out and reported NOT_LIVE on a healthy barrier). Nobody
  // consumes before every producer has arrived, so c stays at its armed value.
  bool barrier_check_live(std::uint32_t id, std::uint8_t /*producers*/, std::uint8_t consumers) {
    std::uint8_t p = 0, c = 0;
    for (std::uint32_t spins = 0; spins < kConfigSpinLimit; ++spins) {
      hw::nce::barriers::hgl::getCounters(id, &p, &c);
      if (c == consumers)
        return true;
    }
    fail(kStatusBarrierNotLive, id);
    return false;
  }
  static void barrier_produce(std::uint32_t id) { hw::nce::barriers::hgl::produceBarrier(id); }
  static void barrier_consume(std::uint32_t id) { hw::nce::barriers::hgl::consumeBarrier(id); }
  // Wait until every producer has arrived (producer count reaches zero). A barrier
  // whose config is not live reads (0, 0) and would pass: callers arm with waitLive.
  bool barrier_wait_produced(std::uint32_t id) {
    std::uint8_t p = 1, c = 0;
    for (std::uint32_t spins = 0; spins < kBarrierSpinLimit; ++spins) {
      hw::nce::barriers::hgl::getCounters(id, &p, &c);
      if (p == 0)
        return true;
    }
    fail(kStatusBarrierTimeout, id);
    return false;
  }
  bool barrier_wait_consumed(std::uint32_t id) {
    std::uint8_t p = 0, c = 1;
    for (std::uint32_t spins = 0; spins < kBarrierSpinLimit; ++spins) {
      hw::nce::barriers::hgl::getCounters(id, &p, &c);
      if (c == 0)
        return true;
    }
    fail(kStatusBarrierTimeout, id);
    return false;
  }
  // A config is live once its consumer count is non-zero (a dead barrier reads
  // (0, 0)); the exact counts are not checked because producers may already have
  // arrived.
  bool barrier_wait_live(std::uint32_t id) {
    std::uint8_t p = 0, c = 0;
    for (std::uint32_t spins = 0; spins < kBarrierSpinLimit; ++spins) {
      hw::nce::barriers::hgl::getCounters(id, &p, &c);
      if (c != 0)
        return true;
    }
    fail(kStatusBarrierNotLive, id);
    return false;
  }

  // Writes the per-SHAVE status record (64 B): {code, detail, cmx_free_at_entry, cmx_used,
  // dpu_workloads, dpu_idu_ticks, dpu_odu_ticks, 0, t_entry, t_finish, extra[0..5]}. The stamps
  // are the NCE_TIME free-running counter (19.2 MHz perf clock on this box): the host takes
  // max(t_finish) - min(t_entry) over the SHAVEs as the kernel-only time. `extra` is a probe's.
  static constexpr std::uint32_t kStatusBytes = 64;
  void finish(const std::uint32_t *extra = nullptr, std::uint32_t nExtra = 0) {
    if (statusAddr_ == 0)
      return;
    const std::uint64_t a = statusAddr_ + static_cast<std::uint64_t>(kStatusBytes) * worker();
    DdrAccessor<std::uint32_t> s0(a), s1(a + 4), s2(a + 8), s3(a + 12), s4(a + 16), s5(a + 20),
        s6(a + 24), s7(a + 28), s8(a + 32), s9(a + 36);
    s0 = ok_ ? kStatusOk : code_;
    s1 = detail_;
    s2 = cmxFree0_;
    s3 = cmxFree0_ - cmx_.getAvailableSize();
    s4 = dpuCount_;
    s5 = dpuIduTicks_;
    s6 = dpuOduTicks_;
    s7 = 0;
    s8 = tStart_;
    s9 = perf_ticks();
    if (extra == nullptr) {
      extra = ringTicks_;   // ring phase counters: live-wait, order-wait, submit, dpu-wait, free-wait, fill-issue
      nExtra = 6;
    }
    for (std::uint32_t i = 0; i < nExtra && i < 6; ++i) {
      DdrAccessor<std::uint32_t> sx(a + 40 + 4 * i);
      sx = extra[i];
    }
    sync::memoryFence();
  }

  // Hardware descriptors for a hand-written probe (the ring and the DMA/DPU paths use their own).
  template <class T> T *alloc_descriptors(std::uint32_t count) { return alloc_hw<T>(count); }

private:
  static constexpr std::uint32_t alignUp(std::uint32_t v, std::uint32_t a) {
    return (v + a - 1) / a * a;
  }
  // DmaDrv40xx trap-descriptor window, relative to the tile's 2 MiB CMX window.
  static constexpr std::uint32_t kTrapLo = 0x12000, kTrapHi = 0x14000;

  std::uint32_t worker() const { return tile_ * spt_ + shave_; }

  // NCE_TIME free-running counter (kernel_profiler.hpp NCE_TIME_FREE_RUNNING_ADR).
  static std::uint32_t perf_ticks() { return getRegWordVal(0x2f020000U); }

  // Grid rendezvous on hardware barriers syncBase_ + (k mod 3): every SHAVE
  // produces, waits for the last producer, consumes; worker 0 re-arms the id
  // once every SHAVE has consumed it. No SHAVE can reach an id again before
  // worker 0 has produced on the next two, which it does only after re-arming,
  // so the re-armed config is live by then. The ids are armed once per launch
  // behind one software rendezvous.
  void hw_sync_grid() {
    if (!ok_)
      return; // a failed SHAVE stops at its status; the others time out and report too
    const std::uint8_t n = static_cast<std::uint8_t>(tiles_ * spt_);
    if (!syncArmed_) {
      if (worker() == 0)
        for (std::uint32_t i = 0; i < kSyncBarriers; ++i)
          barrier_config(syncBase_ + i, n, n, false);
      sync_.waitGlobal();
      syncArmed_ = true;
      // Every SHAVE checks liveness itself: a dead barrier reads (0, 0), which a
      // producer wait would take for "all arrived".
      for (std::uint32_t i = 0; i < kSyncBarriers; ++i)
        if (!barrier_check_live(syncBase_ + i, n, n))
          return;
    }
    const std::uint32_t id = syncBase_ + syncK_;
    syncK_ = (syncK_ + 1 == kSyncBarriers) ? 0 : syncK_ + 1;
    barrier_produce(id);
    if (!barrier_wait_produced(id))
      return;
    barrier_consume(id);
    if (worker() == 0) {
      if (!barrier_wait_consumed(id))
        return;
      barrier_config(id, n, n, false);
    }
  }

  void fail(std::uint32_t code, std::uint32_t detail) {
    if (ok_) {
      ok_ = false;
      code_ = code;
      detail_ = detail;
    }
  }

  void port_retire(int port) {
    if (lastOnPort_[port] != nullptr) {
      dma_wait(lastOnPort_[port], 5 + port);
      lastOnPort_[port] = nullptr;
    }
  }

  // The driver's wait (DmaDrvBase::wait) spins on the tail descriptor's watermark without a
  // bound; the same test with one, so a fill that never completes is a report. `where`:
  // 1/2 ring_fill previous halves, 3/4 ring_end, 5/6 port_retire.
  bool dma_wait(volatile DmaDescriptor *tail, std::uint32_t where) {
    for (std::uint32_t spins = 0; spins < kBarrierSpinLimit; ++spins) {
      if (tail->watermark != 0)
        return true;
    }
    fail(kStatusDmaTimeout, where);
    return false;
  }

  bool order_wait(volatile std::uint32_t *word, std::uint32_t need) {
    for (std::uint32_t spins = 0; spins < kBarrierSpinLimit; ++spins) {
      if (*word >= need)
        return true;
    }
    fail(kStatusOrderTimeout, need);
    return false;
  }

  // Selects the DPU descriptor set for this workload shape, programming it on a
  // miss (kDpuSets sets, least recently used replaced). One set re-initialised on
  // every submit cost 1.2 us per workload when two shapes alternated (the flash
  // phase's QK^T and PV; M1 step 3c counters).
  bool dpu_prepare(std::uint32_t M, std::uint32_t N, std::uint32_t K, std::uint32_t spatialX,
                   std::int32_t mpe) {
    if (!ok_)
      return false;
    if (N > kMaxDpuN) {
      fail(kStatusUnsupported, N);
      return false;
    }
    if (spatialX != 0 && M % spatialX != 0) {
      fail(kStatusBadParams, spatialX);
      return false;
    }
    if (mpe < -1 || mpe > 2) {
      fail(kStatusBadParams, static_cast<std::uint32_t>(mpe));
      return false;
    }
    const std::uint64_t key = (static_cast<std::uint64_t>(M) << 40) |
                              (static_cast<std::uint64_t>(N) << 20) | K;
    const std::uint32_t shapeKey = (spatialX << 8) | static_cast<std::uint32_t>(mpe + 1);
    ++dpuStamp_;
    std::uint32_t pick = 0;
    for (std::uint32_t i = 0; i < kDpuSets; ++i) {
      if (dpu_[i].key == key && dpu_[i].shapeKey == shapeKey) {
        dpu_[i].lastUse = dpuStamp_;
        mm_ = dpu_[i].mm;
        dpuDesc_ = dpu_[i].desc;
        return true;
      }
      if (dpu_[i].lastUse < dpu_[pick].lastUse)
        pick = i;
    }
    DpuSet &s = dpu_[pick];
    s.mm->init(M, N, K);
    if (spatialX != 0)
      s.mm->setSpatial(M, spatialX);
    if (mpe >= 0) {
      // No static tables: a function-local static array misreads on the SHAVE.
      const std::uint32_t nthwNtk = mpe == 0 ? 1u : (mpe == 1 ? 0u : 3u);
      const std::uint32_t oduNthw = mpe == 0 ? 1u : (mpe == 1 ? 2u : 3u);
      s.mm->setStencil(nthwNtk, oduNthw);
    }
    s.key = key;
    s.shapeKey = shapeKey;
    s.lastUse = dpuStamp_;
    mm_ = s.mm;
    dpuDesc_ = s.desc;
    return true;
  }

  // DPU hardware profile of the workload just completed. The duration fields
  // sit before the odu_workload_id watermark wait() spins on, so they are
  // committed; the timestamps after it may still be in flight and are not read.
  // Durations count 19.2 MHz perf-clock ticks on this box (npu_dsl SS690.2).
  void dpu_account(const DpuMatMul::DescriptorsBuffer *set = nullptr) {
    volatile const DpuMatMul::DescriptorsBuffer *d = set ? set : dpuDesc_;
    dpuCount_ += 1;
    dpuIduTicks_ += d->hwpStats.idu_workload_duration;
    dpuOduTicks_ += d->hwpStats.odu_workload_duration;
  }

  // Hardware descriptors avoid the trap window; every SHAVE of the tile makes
  // the same choice because the allocation sequences are identical.
  template <class T> T *alloc_hw(std::uint32_t count) {
    const std::uint32_t bytes = alignUp(count * sizeof(T), kAlign);
    for (int tries = 0; tries < 4; ++tries) {
      T *p = alloc<T>(count);
      if (p == nullptr)
        return nullptr;
      const std::uint32_t base = reinterpret_cast<std::uint32_t>(p) - bytes * shave_;
      const std::uint32_t end = base + bytes * spt_;
      const std::uint32_t lo = ((base >> 21) << 21) + kTrapLo;
      const std::uint32_t hi = ((base >> 21) << 21) + kTrapHi;
      if (end <= lo || base >= hi)
        return p;
    }
    fail(kStatusCmxExhausted, 0);
    return nullptr;
  }

  void dma_load(void *dst, std::uint64_t src, std::uint32_t width, std::uint32_t rows,
                std::uint32_t dstStride, std::uint32_t srcStride) {
    if (!ok_ || width == 0 || rows == 0)
      return;
    // The toolkit's start re-kicks a port's chain as soon as its fences ran, before the
    // previous job completed: a ring fill in flight on the port lost its READY (hoisted
    // ring, Q fill after the first K fill). One job per port per SHAVE, here too.
    port_retire(0);
    port_retire(1);
    if (!ok_)
      return;
    DmaDrv::initDesc(desc_);
    config_copy(desc_, reinterpret_cast<std::uint32_t>(dst), src, width, rows, dstStride, srcStride);
    dma_.run(desc_);
  }

  void dma_store(std::uint64_t dst, const void *src, std::uint32_t width, std::uint32_t rows,
                 std::uint32_t dstStride, std::uint32_t srcStride) {
    if (!ok_ || width == 0 || rows == 0)
      return;
    port_retire(0);
    port_retire(1);
    if (!ok_)
      return;
    flushCmxWrites();
    DmaDrv::initDesc(desc_ + 1);
    config_copy(desc_ + 1, dst, reinterpret_cast<std::uint32_t>(src), width, rows, dstStride, srcStride);
    dma_.run(desc_ + 1);
  }

  // The fewest dimensions that describe the copy: rows that are contiguous on both sides are ONE 1-D
  // transfer (a 2-D descriptor of 512 rows x 256 B ran at a third of the rate of 1024-B rows: the
  // engine's per-row cost; M3), one strided side keeps the other side as a single run.
  static void config_copy(DmaDescriptor *d, std::uint64_t dst, std::uint64_t src, std::uint32_t width,
                          std::uint32_t rows, std::uint32_t dstStride, std::uint32_t srcStride) {
    if (rows == 1 || (dstStride == width && srcStride == width))
      configDmaCopy(d, dst, src, width * rows);
    else if (dstStride == width)
      configDmaCopySrcStrided2D(d, dst, src, width, rows, srcStride);
    else if (srcStride == width)
      configDmaCopyDstStrided2D(d, dst, src, width, rows, dstStride);
    else
      configDmaCopy2D(d, dst, src, width, rows, dstStride, srcStride);
  }

  const Environment &env_;
  LinearAllocator &cmx_;
  DmaDrv &dma_;
  Sync &sync_;
  std::int32_t shave_{0}, spt_{1}, tile_{0}, tiles_{1};
  std::uint64_t statusAddr_{0};
  std::uint32_t cmxFree0_{0};
  std::uint32_t tStart_{0};
  bool ok_{true};
  std::uint32_t code_{kStatusOk}, detail_{0};
  std::uint32_t syncBase_{kNoHwSync};
  std::uint32_t syncK_{0};
  bool syncArmed_{false};

  DmaDescriptor *desc_{nullptr};
  DmaDescriptor *lastOnPort_[2]{nullptr, nullptr};
  // DPU descriptor sets, one per recently used workload shape (CheckIntelNpuBudgets
  // reserves their CMX: kDpuSets x (descriptors + 16 KiB weight table) per SHAVE).
  static constexpr std::uint32_t kDpuSets = 2;
  struct DpuSet {
    DpuMatMul::DescriptorsBuffer *desc{nullptr};
    std::uint32_t *wt{nullptr};
    alignas(DpuMatMul) unsigned char store[sizeof(DpuMatMul)];
    DpuMatMul *mm{nullptr};
    std::uint64_t key{~0ull};
    std::uint32_t shapeKey{~0u};
    std::uint32_t lastUse{0};
  };
  DpuSet dpu_[kDpuSets];
  std::uint32_t dpuStamp_{0};
  DpuMatMul *mm_{nullptr};                           // the set of the last prepared shape
  DpuMatMul::DescriptorsBuffer *dpuDesc_{nullptr};
  // Mixed-mode (fp16 x int4 weights) set for dpu_matmul_i4g, allocated on first use.
  static constexpr float kI4PartScale = 1.0f / 16.0f;  // PPE multiplier keeping the group partials in fp16 range
  static constexpr float kI8PartScale = 1.0f / 256.0f; // |q| <= 128
  alignas(DpuMatMul) unsigned char i4Store_[sizeof(DpuMatMul)];
  DpuMatMul *i4Mm_{nullptr};
  DpuMatMul::DescriptorsBuffer *i4Desc_{nullptr};
  std::uint32_t *i4Wt_{nullptr};
  half *i4Part_{nullptr}, *i4ST_{nullptr};
  std::uint32_t i4PartElems_{0}, i4STElems_{0};
  std::uint64_t i4Key_{~0ull};
  std::uint32_t dpuCount_{0}, dpuIduTicks_{0}, dpuOduTicks_{0};
  // Ring path phase counters (19.2 MHz ticks), the status record's extra words when a
  // probe gives none: where a SHAVE's time in the ring protocol goes.
  enum { kTickLive = 0, kTickOrder = 1, kTickSubmit = 2, kTickDpuWait = 3, kTickFreeWait = 4, kTickFillIssue = 5 };
  std::uint32_t ringTicks_[6]{0, 0, 0, 0, 0, 0};
};

} // namespace npu

#endif // TL_TEMPLATES_INTEL_NPU_NPU_RT_DEVICE_HPP_
