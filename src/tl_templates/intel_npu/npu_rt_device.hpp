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

  // ---- Streamed ping-pong ring (plan Phase 3b; the flow of
  // docs/examples/pingpong_multitile/README.md sections 1-3 and 6c) ----
  //
  // One operand streams through `slots` CMX slots shared by the tile's SHAVEs;
  // chunk c lives in slot c % slots and is the work of SHAVE c % spt (slots ==
  // spt). Tile 0's SHAVE s fills slot s for every tile with ONE multicast DMA,
  // split in two halves on ports 0 and 1. Per (slot, use parity) two hardware
  // barriers: READY (the two halves produce, every tile's DPU workload consumes)
  // and FREE (every tile's DPU workload produces, the issuer consumes before it
  // refills). The next fill is kicked when FREE lifts (one outstanding DMA job
  // per SHAVE per link agent on NPU5: a held job would block the other slot's
  // fill). Barrier ids alternate by use parity so a config is never more than
  // one deep in the barrier FIFO and "consumer count != 0" means "this use's
  // config is live". The tile's two SHAVEs submit to the one DPU FIFO in chunk
  // order through two CMX order words (the FIFO runs in submission order and
  // every fill waits for the slowest tile).
  struct Ring {
    std::uint8_t *slotBase;   // shared: slots x slotBytes, same offset on every tile
    std::uint32_t slotBytes;
    std::uint32_t slots;
    std::uint32_t total;      // chunks streamed in this ring
    std::uint32_t myUses;     // chunks of this SHAVE's slot
    std::uint32_t readyBase;  // READY id = readyBase + slot * 2 + parity
    std::uint32_t freeBase;   // FREE id  = freeBase  + slot * 2 + parity
    volatile std::uint32_t *order; // shared: one word per SHAVE, uses submitted
    DmaDescriptor *fill;      // issuer's two half descriptors
    bool armed;
  };
  enum : std::uint32_t {
    kRingSlotsNeShaves = 1,
    kRingOneTile = 2,
    kRingBarrierPool = 3,
  };

  std::uint32_t ring_ready(const Ring &r, std::uint32_t use) const {
    return r.readyBase + shave_ * 2 + (use & 1u);
  }
  std::uint32_t ring_free(const Ring &r, std::uint32_t use) const {
    return r.freeBase + shave_ * 2 + (use & 1u);
  }
  std::uint8_t *ring_slot(const Ring &r) const { return r.slotBase + shave_ * r.slotBytes; }
  template <class T> T *ring_slot(const Ring &r) const {
    return reinterpret_cast<T *>(ring_slot(r));
  }
  std::uint32_t ring_uses(const Ring &r) const { return r.myUses; }
  std::uint32_t ring_chunk(std::uint32_t use) const { return use * spt_ + shave_; }

  // `slotBase` is the compiler's shared buffer of slots x slotBytes (same address
  // on every SHAVE of the tile, same offset on every tile). Allocates the order
  // words (shared) and the issuer's fill descriptors; arms the first use's
  // barriers on tile 0; one grid rendezvous so every config is live before any
  // gate. `readyBase`/`freeBase` come from the compiler's barrier assignment
  // (2 x slots ids each).
  Ring ring_begin(void *slotBase, std::uint32_t slotBytes, std::uint32_t slots, std::uint32_t total,
                  std::uint32_t readyBase, std::uint32_t freeBase) {
    Ring r{};
    r.slotBase = static_cast<std::uint8_t *>(slotBase);
    r.slotBytes = slotBytes;
    r.slots = slots;
    r.total = total;
    r.readyBase = readyBase;
    r.freeBase = freeBase;
    if (!ok_)
      return r;
    if (static_cast<std::int32_t>(slots) != spt_) {
      fail(kStatusRingRefused, kRingSlotsNeShaves);
      return r;
    }
    if (tiles_ < 2) {
      fail(kStatusRingRefused, kRingOneTile);
      return r;
    }
    r.order = alloc_shared<std::uint32_t>(16); // 64 B: one line, two words used
    r.fill = alloc_hw<DmaDescriptor>(2);
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
    if (tile_ == 0 && r.myUses > 0) {
      const std::uint8_t nt = static_cast<std::uint8_t>(tiles_);
      barrier_config(ring_ready(r, 0), 2, nt, false);
      barrier_config(ring_free(r, 0), nt, 1, false);
      if (!barrier_check_live(ring_ready(r, 0), 2, nt) || !barrier_check_live(ring_free(r, 0), nt, 1))
        return r;
    }
    sync_grid();
    r.armed = ok_;
    return r;
  }

  // Issuer (tile 0, SHAVE s): fill my slot with chunk `use * spt + s` from `src`
  // (`rows` rows of `width` bytes, `srcStride` bytes apart in DDR; contiguous in the
  // slot), two halves on ports 0 and 1, both producing READY[use]. The halves split
  // the rows (the bytes when there is one row). For use > 0 it first waits until
  // every tile's DPU has produced FREE[use-1].
  void ring_fill(Ring &r, std::uint32_t use, std::uint64_t src, std::uint32_t width,
                 std::uint32_t rows, std::uint32_t srcStride) {
    if (!ok_ || tile_ != 0 || use >= r.myUses)
      return;
    if (use > 0) {
      const std::uint32_t freeId = ring_free(r, use - 1);
      if (!barrier_wait_produced(freeId))
        return;
      barrier_consume(freeId);
      // The previous halves have completed (READY lifted); the chains still need the wait.
      dma_.wait(r.fill);
      dma_.wait(r.fill + 1);
    }
    const std::uint32_t mask = (1u << tiles_) - 1u;
    std::uint8_t *dst = ring_slot(r);
    const std::uint32_t ready = ring_ready(r, use);
    DmaDrv::initDesc(r.fill);
    DmaDrv::initDesc(r.fill + 1);
    if (rows <= 1) {
      const std::uint32_t half0 = width / 2;
      configDmaCopy(r.fill, reinterpret_cast<std::uint32_t>(convertToCmxMulticastAddr(dst, mask)), src, half0);
      configDmaCopy(r.fill + 1, reinterpret_cast<std::uint32_t>(convertToCmxMulticastAddr(dst + half0, mask)),
                    src + half0, width - half0);
    } else {
      const std::uint32_t rows0 = rows / 2;
      configDmaCopy2D(r.fill, reinterpret_cast<std::uint32_t>(convertToCmxMulticastAddr(dst, mask)), src, width,
                      rows0, width, srcStride);
      configDmaCopy2D(r.fill + 1,
                      reinterpret_cast<std::uint32_t>(convertToCmxMulticastAddr(dst + rows0 * width, mask)),
                      src + static_cast<std::uint64_t>(rows0) * srcStride, width, rows - rows0, width, srcStride);
    }
    DmaDrv::setProducerBarrier(r.fill, ready);
    DmaDrv::setProducerBarrier(r.fill + 1, ready);
    DmaDrv::setConsumerBarriers(r.fill, 0, 0);
    DmaDrv::setConsumerBarriers(r.fill + 1, 0, 0);
    // One descriptor of this SHAVE in flight per port: a second start while another
    // ring's fill is still running on the port left one half unproduced (flash phase,
    // two rings; READY stuck at p=1).
    port_retire(0);
    port_retire(1);
    dma_.start(r.fill, 0);
    dma_.start(r.fill + 1, 1);
    lastOnPort_[0] = r.fill;
    lastOnPort_[1] = r.fill + 1;
  }
  template <class T>
  void ring_fill(Ring &r, std::uint32_t use, Ddr<T> src, std::uint32_t width, std::uint32_t rows,
                 std::uint32_t srcStride) {
    ring_fill(r, use, src.addr, width, rows, srcStride);
  }

  // Every SHAVE: submit the DPU workload of my `use`, gated on READY[use] (in
  // hardware, so it may be queued before the chunk has landed) and producing
  // FREE[use], in chunk order after the other SHAVE's previous chunk; then the
  // issuer pushes the next use's configs. The caller submits use u+1 right after
  // use u completes (submit-ahead, README 6c fact 3), before its FREE wait, fill
  // kick and store, so the DPU FIFO never drains while a SHAVE does its own work;
  // that needs the output buffer double-buffered by use parity.
  void ring_dpu_submit(Ring &r, std::uint32_t use, half *out, const half *a, const half *bt,
                       std::uint32_t M, std::uint32_t N, std::uint32_t K,
                       std::uint32_t spatialX = 0, std::int32_t mpe = -1) {
    if (!ok_ || !r.armed || use >= r.myUses)
      return;
    const std::uint32_t ready = ring_ready(r, use), freeId = ring_free(r, use);
    if (!barrier_wait_live(ready) || !barrier_wait_live(freeId))
      return;
    // Submission order P0(u), P1(u), P0(u+1), ...: SHAVE 0 needs SHAVE 1 at `use`
    // submissions, SHAVE 1 needs SHAVE 0 at `use + 1`.
    const std::uint32_t prev = (shave_ + spt_ - 1) % spt_;
    if (!order_wait(r.order + prev, shave_ == 0 ? use : use + 1))
      return;
    if (!dpu_prepare(M, N, K, spatialX, mpe))
      return;
    mm_->setConsumerBarrier(ready);
    mm_->setProducerBarrier(freeId);
    mm_->start(out, const_cast<half *>(a), const_cast<half *>(bt));
    r.order[shave_] = use + 1;
    flushCmxWrites();
    if (tile_ == 0 && use + 1 < r.myUses) {
      const std::uint8_t nt = static_cast<std::uint8_t>(tiles_);
      barrier_config(ring_ready(r, use + 1), 2, nt, false);
      barrier_config(ring_free(r, use + 1), nt, 1, false);
    }
  }
  void ring_dpu_wait(Ring &r, std::uint32_t use) {
    if (!ok_ || !r.armed || use >= r.myUses)
      return;
    // The driver's wait spins forever on a workload whose READY gate never opens; the
    // same watermark (DpuDrvBase::wait) with a bound, so a stuck gate is a report.
    volatile const std::uint16_t *watermark = &dpuDesc_->hwpStats.odu_workload_id;
    const std::uint16_t expected = dpuDesc_->variant.hwp_wload_id.hwp_wload_id_bf.wload_id;
    for (std::uint32_t spins = 0; spins < kBarrierSpinLimit; ++spins) {
      if (*watermark == expected) {
        dpu_account();
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

  // Issuer: the last fills must be retired on their chains before the kernel ends.
  void ring_end(Ring &r) {
    if (!ok_ || tile_ != 0 || r.myUses == 0)
      return;
    dma_.wait(r.fill);
    dma_.wait(r.fill + 1);
    if (lastOnPort_[0] == r.fill)
      lastOnPort_[0] = nullptr;
    if (lastOnPort_[1] == r.fill + 1)
      lastOnPort_[1] = nullptr;
    // Clear the barriers' consumer side of the last FREE so the id is quiescent.
    const std::uint32_t freeId = ring_free(r, r.myUses - 1);
    if (barrier_wait_produced(freeId))
      barrier_consume(freeId);
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
      dma_.wait(lastOnPort_[port]);
      lastOnPort_[port] = nullptr;
    }
  }

  bool order_wait(volatile std::uint32_t *word, std::uint32_t need) {
    for (std::uint32_t spins = 0; spins < kBarrierSpinLimit; ++spins) {
      if (*word >= need)
        return true;
    }
    fail(kStatusOrderTimeout, need);
    return false;
  }

  // Programs the one DPU descriptor for this workload shape (cached by shape).
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
    return true;
  }

  // DPU hardware profile of the workload just completed. The duration fields
  // sit before the odu_workload_id watermark wait() spins on, so they are
  // committed; the timestamps after it may still be in flight and are not read.
  // Durations count 19.2 MHz perf-clock ticks on this box (npu_dsl SS690.2).
  void dpu_account() {
    volatile const DpuMatMul::DescriptorsBuffer *d = dpuDesc_;
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
  std::uint32_t tStart_{0};
  bool ok_{true};
  std::uint32_t code_{kStatusOk}, detail_{0};
  std::uint32_t syncBase_{kNoHwSync};
  std::uint32_t syncK_{0};
  bool syncArmed_{false};

  DmaDescriptor *desc_{nullptr};
  DpuMatMul::DescriptorsBuffer *dpuDesc_{nullptr};
  DmaDescriptor *lastOnPort_[2]{nullptr, nullptr};
  std::uint32_t *wt_{nullptr};
  alignas(DpuMatMul) unsigned char mmStore_[sizeof(DpuMatMul)];
  DpuMatMul *mm_{nullptr};
  std::uint64_t mmKey_{~0ull};
  std::uint32_t mmShapeKey_{~0u};
  std::uint32_t dpuCount_{0}, dpuIduTicks_{0}, dpuOduTicks_{0};
};

} // namespace npu

#endif // TL_TEMPLATES_INTEL_NPU_NPU_RT_DEVICE_HPP_
