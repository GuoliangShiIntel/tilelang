// Host (CPU) implementation of the TileLang Intel NPU runtime surface, for the
// hardware-free simulation. Included by npu_rt.hpp under -DNPU_RT_HOST_SIM;
// compiled by clang++ (needs _Float16).
//
// Model: one std::thread per SHAVE; each tile owns a CMX arena and every SHAVE
// of a tile runs the same allocation sequence, taking slot `shave` of each
// allocation (the device semantics). DMA is memcpy; the DPU matmul is an fp32
// accumulate of fp16 inputs rounded to fp16 on store.
#ifndef TL_TEMPLATES_INTEL_NPU_NPU_RT_HOST_HPP_
#define TL_TEMPLATES_INTEL_NPU_NPU_RT_HOST_HPP_

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

typedef _Float16 half;

#include <cmath>
// fp16 math behind the intel_npu intrinsic rules (intrin_rule_intel_npu.cc), float math on the host.
static inline half tl_npu_hexp(half x) { return (half)expf((float)x); }
static inline half tl_npu_hexp2(half x) { return (half)exp2f((float)x); }
static inline half tl_npu_hlog(half x) { return (half)logf((float)x); }
static inline half tl_npu_hlog2(half x) { return (half)log2f((float)x); }
static inline half tl_npu_hsqrt(half x) { return (half)sqrtf((float)x); }
static inline half tl_npu_hrsqrt(half x) { return (half)(1.0f / sqrtf((float)x)); }
static inline half tl_npu_htanh(half x) { return (half)tanhf((float)x); }
static inline half tl_npu_hsigmoid(half x) { return (half)(1.0f / (1.0f + expf(-(float)x))); }
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline half tl_npu_hfabs(half x) { return (half)fabsf((float)x); }
static inline half tl_npu_hfloor(half x) { return (half)floorf((float)x); }
static inline half tl_npu_hceil(half x) { return (half)ceilf((float)x); }
static inline half tl_npu_hround(half x) { return (half)roundf((float)x); }
// CodeGenC prints tirx.max / tirx.min as max(a, b) / min(a, b).
template <class T> static inline T max(T a, T b) { return a > b ? a : b; }
template <class T> static inline T min(T a, T b) { return a < b ? a : b; }

// The SHAVE compiler's ext_vector typedefs (the codegen prints half16, float8, ...).
typedef half half4 __attribute__((ext_vector_type(4)));
typedef half half8 __attribute__((ext_vector_type(8)));
typedef half half16 __attribute__((ext_vector_type(16)));
typedef half half32 __attribute__((ext_vector_type(32)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef int int16 __attribute__((ext_vector_type(16)));
#include "npu_rt_vecmath.hpp"

// Whole-vector reductions behind T.call_extern("tl_npu_vmax" / "tl_npu_vsum", buf[0:N]).
#define TL_NPU_HOST_REDUCE(V, S, N)                                             \
  static inline S tl_npu_vmax(V v) {                                            \
    S r = v[0];                                                                 \
    for (int i = 1; i < N; ++i)                                                 \
      r = v[i] > r ? (S)v[i] : r;                                               \
    return r;                                                                   \
  }                                                                             \
  static inline S tl_npu_vsum(V v) {                                            \
    S r = v[0];                                                                 \
    for (int i = 1; i < N; ++i)                                                 \
      r = r + (S)v[i];                                                          \
    return r;                                                                   \
  }
TL_NPU_HOST_REDUCE(half32, half, 32)
TL_NPU_HOST_REDUCE(half16, half, 16)
TL_NPU_HOST_REDUCE(half8, half, 8)
TL_NPU_HOST_REDUCE(float16, float, 16)
TL_NPU_HOST_REDUCE(float8, float, 8)
TL_NPU_HOST_REDUCE(float4, float, 4)
#undef TL_NPU_HOST_REDUCE

// Perf regions (device: profiler VEC_COMPUTE spans) are no-ops in the host simulation.
static inline void tl_npu_perf_begin(const char *) {}
static inline void tl_npu_perf_end() {}

// Packed signed int4 (two per byte, low nibble first, value = (n ^ 8) - 8) to fp16, 64 values
// from 32 bytes as two half32 halves (the device's LSU convert-load).
static inline half32 tl_npu_i4x64_part(const void *p, int first) {
  const unsigned char *b = (const unsigned char *)p;
  half32 r;
  for (int i = 0; i < 32; ++i) {
    const int e = first + i;
    const int n = (e & 1) ? (b[e >> 1] >> 4) : (b[e >> 1] & 0xF);
    r[i] = (half)((n ^ 8) - 8);
  }
  return r;
}
static inline half32 tl_npu_i4x64_lo(const void *p) { return tl_npu_i4x64_part(p, 0); }
static inline half32 tl_npu_i4x64_hi(const void *p) { return tl_npu_i4x64_part(p, 32); }
typedef signed char schar64 __attribute__((ext_vector_type(64)));
typedef signed char schar32 __attribute__((ext_vector_type(32)));
static inline schar64 tl_npu_ld_i4x64(const void *p) {
  const unsigned char *b = (const unsigned char *)p;
  schar64 r;
  for (int e = 0; e < 64; ++e) {
    const int n = (e & 1) ? (b[e >> 1] >> 4) : (b[e >> 1] & 0xF);
    r[e] = (signed char)((n ^ 8) - 8);
  }
  return r;
}
static inline void tl_npu_dequant_i4_row(half *dst, const unsigned char *src, const half *scales, unsigned width,
                                         unsigned group) {
  for (unsigned k = 0; k < width; ++k) {
    const int n = (k & 1) ? (src[k >> 1] >> 4) : (src[k >> 1] & 0xF);
    dst[k] = (half)((float)((n ^ 8) - 8) * (float)scales[k / group]);
  }
}
static inline void tl_npu_dequant_i4_block(half *dst, const unsigned char *src, const half *scales, unsigned rows,
                                           unsigned width, unsigned group) {
  for (unsigned r = 0; r < rows; ++r)
    tl_npu_dequant_i4_row(dst + r * width, src + r * (width / 2), scales + r * (width / group), width, group);
}

#if defined(_WIN32)
#define NPU_SIM_EXPORT extern "C" __declspec(dllexport)
#else
#define NPU_SIM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// The generated kernel's entry point (defined in the generated .cpp).
namespace npu {
struct Environment;
}
void kernel_entry(int numParams, uint64_t *params, const npu::Environment &env);

namespace npu {

// Reusable rendezvous for a fixed party count (C++17; std::barrier is C++20).
class Barrier {
public:
  explicit Barrier(unsigned parties) : parties_(parties) {}
  void wait() {
    std::unique_lock<std::mutex> lk(m_);
    const unsigned gen = gen_;
    if (++count_ == parties_) {
      count_ = 0;
      ++gen_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lk, [&] { return gen != gen_; });
  }

private:
  std::mutex m_;
  std::condition_variable cv_;
  unsigned parties_, count_{0}, gen_{0};
};

enum : std::uint32_t {
  kStatusOk = 1,
  kStatusCmxExhausted = 2,
  kStatusBadParams = 3,
  kStatusUnsupported = 4,
};

template <class T> struct Ddr;

template <class T> struct DdrRef {
  std::uint64_t addr;
  Ddr<T> operator&() const { return Ddr<T>{addr}; }
  operator T() const { return *reinterpret_cast<const T *>(static_cast<std::uintptr_t>(addr)); }
  DdrRef &operator=(const T &v) {
    *reinterpret_cast<T *>(static_cast<std::uintptr_t>(addr)) = v;
    return *this;
  }
};

template <class T> struct Ddr {
  std::uint64_t addr;
  explicit Ddr(std::uint64_t a = 0) : addr(a) {}
  DdrRef<T> operator[](std::int64_t i) const {
    return DdrRef<T>{addr + static_cast<std::uint64_t>(i) * sizeof(T)};
  }
};

struct Environment {
  std::uint32_t tile, shave, tiles, spt;
  std::uint8_t *cmx;       // this tile's arena
  std::uint32_t cmxBytes;
  Barrier *grid;           // all SHAVEs of all tiles
  Barrier *tileBarrier;    // the SHAVEs of this tile
};

class Ctx {
public:
  static constexpr std::uint32_t kAlign = 64;
  static constexpr std::uint32_t kMaxDpuN = 1024;

  static constexpr std::uint32_t kNoHwSync = 0xFFFFFFFFu;

  Ctx(const Environment &env, int numParams, std::uint64_t *params, int numTensorParams,
      std::uint32_t /*syncBarrierBase*/ = kNoHwSync)
      : env_(env) {
    statusAddr_ = numParams > numTensorParams ? params[numTensorParams] : 0;
    if (numParams < numTensorParams) {
      fail(kStatusBadParams, numParams);
    }
  }

  bool ok() const { return ok_; }
  std::int32_t tile_id() const { return env_.tile; }
  std::int32_t shave_id() const { return env_.shave; }
  std::int32_t num_tiles() const { return env_.tiles; }
  std::int32_t shaves_per_tile() const { return env_.spt; }

  template <class T> T *alloc(std::uint32_t count) {
    const std::uint32_t bytes = alignUp(count * sizeof(T), kAlign);
    const std::uint32_t need = bytes * env_.spt;
    if (offset_ + need > env_.cmxBytes) {
      fail(kStatusCmxExhausted, need);
      return nullptr;
    }
    std::uint8_t *base = env_.cmx + offset_;
    offset_ += need;
    return reinterpret_cast<T *>(base + bytes * env_.shave);
  }
  // Tile-shared: both SHAVEs of a tile run the same allocation sequence on the same arena.
  template <class T> T *alloc_shared(std::uint32_t count) {
    const std::uint32_t bytes = alignUp(count * sizeof(T), kAlign);
    if (offset_ + bytes > env_.cmxBytes) {
      fail(kStatusCmxExhausted, bytes);
      return nullptr;
    }
    std::uint8_t *base = env_.cmx + offset_;
    offset_ += bytes;
    return reinterpret_cast<T *>(base);
  }
  // Codegen brackets every C++ scope that allocates with a CmxScope: the allocations of a phase or a
  // loop body are released at its end, so sequential phases share CMX (the device runtime does the same).
  using CmxMark = std::uint32_t;
  CmxMark cmx_mark() const { return offset_; }
  void cmx_restore(CmxMark m) { offset_ = m; }
  class CmxScope {
  public:
    explicit CmxScope(Ctx &c) : c_(c), m_(c.cmx_mark()) {}
    ~CmxScope() { c_.cmx_restore(m_); }
    CmxScope(const CmxScope &) = delete;
    CmxScope &operator=(const CmxScope &) = delete;

  private:
    Ctx &c_;
    CmxMark m_;
  };

  template <class T>
  void dma_load(T *dst, Ddr<T> src, std::uint32_t width, std::uint32_t rows,
                std::uint32_t dstStride, std::uint32_t srcStride) {
    copy2d(reinterpret_cast<std::uint8_t *>(dst), ptr(src.addr), width, rows, dstStride, srcStride);
  }
  template <class T>
  void dma_store(Ddr<T> dst, const T *src, std::uint32_t width, std::uint32_t rows,
                 std::uint32_t dstStride, std::uint32_t srcStride) {
    copy2d(ptr(dst.addr), reinterpret_cast<const std::uint8_t *>(src), width, rows, dstStride, srcStride);
  }

  void dpu_matmul(half *out, const half *a, const half *bt, std::uint32_t M, std::uint32_t N,
                  std::uint32_t K, std::uint32_t spatialX = 0, std::int32_t mpe = -1) {
    if (!ok_)
      return;
    if (N > kMaxDpuN) {
      fail(kStatusUnsupported, N);
      return;
    }
    // The workload shape only changes the DPU's rate, never its result.
    if ((spatialX != 0 && M % spatialX != 0) || mpe < -1 || mpe > 2) {
      fail(kStatusBadParams, spatialX);
      return;
    }
    for (std::uint32_t m = 0; m < M; ++m) {
      const half *ar = a + static_cast<std::size_t>(m) * K;
      for (std::uint32_t n = 0; n < N; ++n) {
        const half *br = bt + static_cast<std::size_t>(n) * K;
        float acc = 0.f;
        for (std::uint32_t k = 0; k < K; ++k)
          acc += static_cast<float>(ar[k]) * static_cast<float>(br[k]);
        out[static_cast<std::size_t>(m) * N + n] = static_cast<half>(acc);
      }
    }
  }

  // The device's merged-FC quantised GEMV, with its rounding points: each (row, group) partial product is
  // scaled by 1/16 and rounded to fp16 by the DPU's ODU, the SHAVE sums partial x scale in fp32, x16, fp16.
  // `defer` is a device scheduling hint (the reduction runs later); the host sim reduces at once.
  void dpu_matmul_i4g(half *out, const half *a, const std::uint8_t *q, const half *s, std::uint32_t M,
                      std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride = 0, bool = false) {
    dpu_matmul_qg(out, a, q, s, M, N, K, G, 4, sStride);
  }
  void dpu_matmul_i8g(half *out, const half *a, const std::int8_t *q, const half *s, std::uint32_t M,
                      std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride = 0, bool = false) {
    dpu_matmul_qg(out, a, reinterpret_cast<const std::uint8_t *>(q), s, M, N, K, G, 8, sStride);
  }
  void dpu_matmul_i4g(half *out, const half *a, const std::uint8_t *q, const std::uint8_t *s, std::uint32_t M,
                      std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride, bool = false) {
    dpu_matmul_qg(out, a, q, reinterpret_cast<const half *>(s), M, N, K, G, 4, sStride);
  }
  void dpu_matmul_i8g(half *out, const half *a, const std::int8_t *q, const std::int8_t *s, std::uint32_t M,
                      std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride, bool = false) {
    dpu_matmul_qg(out, a, reinterpret_cast<const std::uint8_t *>(q), reinterpret_cast<const half *>(s), M, N, K, G, 8,
                  sStride);
  }
  void qg_flush() {}
  // sStride 0: s row-major [N, K/G]; > 0: s group-major [K/G, sStride] (the device reads it in place).
  void reserve_qg(std::uint32_t, std::uint32_t, std::uint32_t = 0) {}   // the device's merged-FC scratch; nothing to reserve here
  void dpu_matmul_qg(half *out, const half *a, const std::uint8_t *q, const half *s, std::uint32_t M,
                     std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t bits, std::uint32_t sStride) {
    if (!ok_)
      return;
    const std::uint32_t ng = K / G;
    if (K % G != 0 || G % 16 != 0 || N % 16 != 0 || N * ng > kMaxDpuN) {
      fail(kStatusUnsupported, N * ng);
      return;
    }
    const float partScale = bits == 8 ? 1.0f / 256.0f : 1.0f / 16.0f;
    for (std::uint32_t m = 0; m < M; ++m)
      for (std::uint32_t n = 0; n < N; ++n) {
        float acc = 0.f;
        for (std::uint32_t g = 0; g < ng; ++g) {
          float part = 0.f;
          for (std::uint32_t k = g * G; k < (g + 1) * G; ++k) {
            int w;
            if (bits == 8) {
              w = static_cast<signed char>(q[static_cast<std::size_t>(n) * K + k]);
            } else {
              const unsigned byte = q[static_cast<std::size_t>(n) * (K / 2) + k / 2];
              const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
              w = (nib ^ 8) - 8;
            }
            part += static_cast<float>(a[static_cast<std::size_t>(m) * K + k]) * static_cast<float>(w);
          }
          const half p16 = static_cast<half>(part * partScale);
          const half sc = sStride ? s[static_cast<std::size_t>(g) * sStride + n] : s[static_cast<std::size_t>(n) * ng + g];
          acc += static_cast<float>(p16) * static_cast<float>(sc);
        }
        out[static_cast<std::size_t>(m) * N + n] = static_cast<half>(acc / partScale);
      }
  }

  void sync_grid() { env_.grid->wait(); }
  void sync_tile() { env_.tileBarrier->wait(); }

  void finish() {
    if (statusAddr_ == 0)
      return;
    auto *s = reinterpret_cast<std::uint32_t *>(ptr(statusAddr_ + 64ull * worker()));
    s[0] = ok_ ? kStatusOk : code_;
    s[1] = detail_;
    s[2] = env_.cmxBytes;
    s[3] = offset_;
  }

private:
  static constexpr std::uint32_t alignUp(std::uint32_t v, std::uint32_t a) {
    return (v + a - 1) / a * a;
  }
  static std::uint8_t *ptr(std::uint64_t addr) {
    return reinterpret_cast<std::uint8_t *>(static_cast<std::uintptr_t>(addr));
  }
  std::uint32_t worker() const { return env_.tile * env_.spt + env_.shave; }
  void fail(std::uint32_t code, std::uint32_t detail) {
    if (ok_) {
      ok_ = false;
      code_ = code;
      detail_ = detail;
    }
  }
  void copy2d(std::uint8_t *dst, const std::uint8_t *src, std::uint32_t width, std::uint32_t rows,
              std::uint32_t dstStride, std::uint32_t srcStride) {
    if (!ok_)
      return;
    for (std::uint32_t r = 0; r < rows; ++r)
      std::memcpy(dst + static_cast<std::size_t>(r) * dstStride,
                  src + static_cast<std::size_t>(r) * srcStride, width);
  }

  const Environment &env_;
  std::uint32_t offset_{0};
  std::uint64_t statusAddr_{0};
  bool ok_{true};
  std::uint32_t code_{kStatusOk}, detail_{0};
};

} // namespace npu

// Runs kernel_entry on tiles x spt host threads with a cmxBytes arena per tile.
// `params` is the kernel's parameter block; the status area (if any) is
// params[numTensorParams] and must hold 32 bytes per SHAVE. Returns 0.
NPU_SIM_EXPORT int npu_sim_run(int numParams, uint64_t *params, uint32_t tiles, uint32_t spt,
                               uint32_t cmxBytes) {
  std::vector<std::vector<std::uint8_t>> arenas(tiles, std::vector<std::uint8_t>(cmxBytes));
  npu::Barrier grid(tiles * spt);
  std::deque<npu::Barrier> tileBarriers;
  for (uint32_t t = 0; t < tiles; ++t)
    tileBarriers.emplace_back(spt);
  std::vector<npu::Environment> envs;
  for (uint32_t t = 0; t < tiles; ++t)
    for (uint32_t s = 0; s < spt; ++s)
      envs.push_back(npu::Environment{t, s, tiles, spt, arenas[t].data(), cmxBytes, &grid, &tileBarriers[t]});
  std::vector<std::thread> threads;
  for (auto &env : envs)
    threads.emplace_back([&] { kernel_entry(numParams, params, env); });
  for (auto &th : threads)
    th.join();
  return 0;
}

#endif // TL_TEMPLATES_INTEL_NPU_NPU_RT_HOST_HPP_
