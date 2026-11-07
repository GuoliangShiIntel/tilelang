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

namespace npu {

// Status word 0 per SHAVE, read by the host runner; word 1 is the detail.
enum : std::uint32_t {
  kStatusOk = 1,
  kStatusCmxExhausted = 2,
  kStatusBadParams = 3,
  kStatusUnsupported = 4,
  kStatusBarrierNotLive = 5, // detail: barrier id whose config never went live
  kStatusBarrierTimeout = 6, // detail: barrier id whose count never reached zero
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
    dpuDesc_ = alloc_hw<DpuMatMul::DescriptorsBuffer>(1);
    wt_ = alloc<std::uint32_t>(4 * kMaxDpuN);
    if (ok_) {
      mm_ = new (mmStore_) DpuMatMul(env_.getDpuDrv(), dpuDesc_, wt_);
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
    if (!ok_)
      return;
    if (N > kMaxDpuN) {
      fail(kStatusUnsupported, N);
      return;
    }
    if (spatialX != 0 && M % spatialX != 0) {
      fail(kStatusBadParams, spatialX);
      return;
    }
    if (mpe < -1 || mpe > 2) {
      fail(kStatusBadParams, static_cast<std::uint32_t>(mpe));
      return;
    }
    const std::uint64_t key = (static_cast<std::uint64_t>(M) << 40) |
                              (static_cast<std::uint64_t>(N) << 20) | K;
    const std::uint32_t shapeKey = (spatialX << 8) | static_cast<std::uint32_t>(mpe + 1);
    if (key != mmKey_ || shapeKey != mmShapeKey_) {
      mm_->init(M, N, K);
      if (spatialX != 0)
        mm_->setSpatial(M, spatialX);
      if (mpe >= 0) {
        // No static tables: a function-local static array misreads on the SHAVE.
        const std::uint32_t nthwNtk = mpe == 0 ? 1u : (mpe == 1 ? 0u : 3u);
        const std::uint32_t oduNthw = mpe == 0 ? 1u : (mpe == 1 ? 2u : 3u);
        mm_->setStencil(nthwNtk, oduNthw);
      }
      mmKey_ = key;
      mmShapeKey_ = shapeKey;
    }
    mm_->run(out, const_cast<half *>(a), const_cast<half *>(bt));
    // DPU hardware profile of the workload just completed. The duration fields
    // sit before the odu_workload_id watermark wait() spins on, so they are
    // committed; the timestamps after it may still be in flight and are not read.
    // Durations count 19.2 MHz perf-clock ticks on this box (npu_dsl SS690.2).
    volatile const DpuMatMul::DescriptorsBuffer *d = dpuDesc_;
    dpuCount_ += 1;
    dpuIduTicks_ += d->hwpStats.idu_workload_duration;
    dpuOduTicks_ += d->hwpStats.odu_workload_duration;
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
  // True when the barrier's live counters equal the expected config.
  bool barrier_check_live(std::uint32_t id, std::uint8_t producers, std::uint8_t consumers) {
    std::uint8_t p = 0, c = 0;
    for (std::uint32_t spins = 0; spins < kConfigSpinLimit; ++spins) {
      hw::nce::barriers::hgl::getCounters(id, &p, &c);
      if (p == producers && c == consumers)
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

  // Writes the per-SHAVE status record: {code, detail, cmx_free_at_entry, cmx_used,
  // dpu_workloads, dpu_idu_ticks, dpu_odu_ticks, 0}.
  void finish() {
    if (statusAddr_ == 0)
      return;
    const std::uint64_t a = statusAddr_ + 32ull * worker();
    DdrAccessor<std::uint32_t> s0(a), s1(a + 4), s2(a + 8), s3(a + 12), s4(a + 16), s5(a + 20),
        s6(a + 24), s7(a + 28);
    s0 = ok_ ? kStatusOk : code_;
    s1 = detail_;
    s2 = cmxFree0_;
    s3 = cmxFree0_ - cmx_.getAvailableSize();
    s4 = dpuCount_;
    s5 = dpuIduTicks_;
    s6 = dpuOduTicks_;
    s7 = 0;
    sync::memoryFence();
  }

private:
  static constexpr std::uint32_t alignUp(std::uint32_t v, std::uint32_t a) {
    return (v + a - 1) / a * a;
  }
  // DmaDrv40xx trap-descriptor window, relative to the tile's 2 MiB CMX window.
  static constexpr std::uint32_t kTrapLo = 0x12000, kTrapHi = 0x14000;

  std::uint32_t worker() const { return tile_ * spt_ + shave_; }

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
    DmaDrv::initDesc(desc_);
    configDmaCopy2D(desc_, reinterpret_cast<std::uint32_t>(dst), src, width, rows,
                    dstStride, srcStride);
    dma_.run(desc_);
  }

  void dma_store(std::uint64_t dst, const void *src, std::uint32_t width, std::uint32_t rows,
                 std::uint32_t dstStride, std::uint32_t srcStride) {
    if (!ok_ || width == 0 || rows == 0)
      return;
    flushCmxWrites();
    DmaDrv::initDesc(desc_ + 1);
    configDmaCopy2D(desc_ + 1, dst, reinterpret_cast<std::uint32_t>(src), width, rows,
                    dstStride, srcStride);
    dma_.run(desc_ + 1);
  }

  const Environment &env_;
  LinearAllocator &cmx_;
  DmaDrv &dma_;
  Sync &sync_;
  std::int32_t shave_{0}, spt_{1}, tile_{0}, tiles_{1};
  std::uint64_t statusAddr_{0};
  std::uint32_t cmxFree0_{0};
  bool ok_{true};
  std::uint32_t code_{kStatusOk}, detail_{0};
  std::uint32_t syncBase_{kNoHwSync};
  std::uint32_t syncK_{0};
  bool syncArmed_{false};

  DmaDescriptor *desc_{nullptr};
  DpuMatMul::DescriptorsBuffer *dpuDesc_{nullptr};
  std::uint32_t *wt_{nullptr};
  alignas(DpuMatMul) unsigned char mmStore_[sizeof(DpuMatMul)];
  DpuMatMul *mm_{nullptr};
  std::uint64_t mmKey_{~0ull};
  std::uint32_t mmShapeKey_{~0u};
  std::uint32_t dpuCount_{0}, dpuIduTicks_{0}, dpuOduTicks_{0};
};

} // namespace npu

#endif // TL_TEMPLATES_INTEL_NPU_NPU_RT_DEVICE_HPP_
