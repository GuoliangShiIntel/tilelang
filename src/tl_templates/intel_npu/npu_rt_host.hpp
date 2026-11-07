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
