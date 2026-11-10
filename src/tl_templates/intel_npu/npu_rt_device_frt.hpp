// The TileLang Intel NPU runtime surface (npu::Ctx, see npu_rt_device.hpp)
// implemented over npu_dsl's fused_rt.hpp, so a TileLang kernel runs with the
// DMA / DPU disciplines npu_dsl's fused single-launch models run with on PTL
// (M5, 2026-11-07):
//   * one synchronous load descriptor, one store descriptor, stores waited for
//   at once (FRT_SYNC_STORE);
//   * at most ONE streamed fill in flight per SHAVE, on the SHAVE's second DMA
//   port (fused_rt pf_issue: a
//     second chain on one port and direction hangs on PTL); a ring's fill ahead
//     overlaps the current chunk's compute and nothing more;
//   * every DPU workload waited for by the SHAVE that issued it (fused_rt
//   matmul / matmul_qg): a streamed slot
//     is READY when its fill's watermark is set and FREE by program order, so
//     no hardware barrier gates a workload;
//   * one fp16 and one quantised DPU descriptor set per SHAVE, rebuilt only
//   when the shape changes;
//   * grid sync is the toolkit's software rendezvous (fused_rt barrier()).
// Multicast rings (tile 0 filling every tile) are refused here
// (kStatusRingRefused / kRingMulticastUnsupported): the models' rings are all
// split streams.
#ifndef TL_TEMPLATES_INTEL_NPU_NPU_RT_DEVICE_FRT_HPP_
#define TL_TEMPLATES_INTEL_NPU_NPU_RT_DEVICE_FRT_HPP_

#ifndef TL_NPU_RUNTIME_FRT
#define TL_NPU_RUNTIME_FRT 1
#endif
#define FRT_SYNC_STORE                                                         \
  1 // generated SHAVE code writes CMX without guard() calls
#define FRT_DPU_Q_ROWMAJOR                                                     \
  0 // scales are packed group-major (T.gemm_i4 scales_row)
#define FRT_DPU_ASYNC 0
#define FRT_GATHER_DESCS 0

#include "fused_rt.hpp"
#include "npu_rt_device.hpp" // math helpers, status codes, Ddr<T>; its Ctx is compiled out (TL_NPU_RUNTIME_FRT)

namespace npu {

class Ctx : public frt::Ctx {
public:
  static constexpr std::uint32_t kAlign = 64;
  static constexpr std::uint32_t kMaxDpuN = frt::kMaxDpuN;
  static constexpr std::uint32_t kNoHwSync = 0xFFFFFFFFu;
  static constexpr std::uint32_t kRingPool = 8, kRingMaxDepth = 2;

  Ctx(const Environment &env, int numParams, std::uint64_t *params,
      int numTensorParams, std::uint32_t /*syncBarrierBase*/ = kNoHwSync)
      : frt::Ctx(env, numParams, params, numTensorParams) {
    tStart_ = perf_ticks();
    statusAddr_ = numParams > numTensorParams ? params[numTensorParams] : 0;
    if (numParams < numTensorParams) {
      fail(kStatusBadParams, numParams);
      return;
    }
    ringFill_ = alloc_hw<DmaDescriptor>(kRingPool * kRingMaxDepth);
    if (ringFill_ == nullptr)
      fail(kStatusCmxExhausted, 0);
  }

  bool ok() const { return ok_; }
  std::int32_t tile_id() const { return tile_; }
  std::int32_t shave_id() const { return shave_; }
  std::int32_t num_tiles() const { return tiles_; }
  std::int32_t shaves_per_tile() const { return spt_; }

  // ---- CMX
  // --------------------------------------------------------------------------------------------------
  template <class T> T *alloc(std::uint32_t count) {
    T *p = frt::Ctx::alloc<T>(count);
    if (p == nullptr)
      fail(kStatusCmxExhausted, count * sizeof(T));
    return p;
  }
  template <class T> T *alloc_shared(std::uint32_t count) {
    const std::uint32_t bytes = frt::alignUp(count * sizeof(T), kAlign);
    auto *base = cmx_.alloc<kAlign>(bytes);
    if (base == nullptr) {
      fail(kStatusCmxExhausted, bytes);
      return nullptr;
    }
    return reinterpret_cast<T *>(base);
  }
  template <class T> T *alloc_descriptors(std::uint32_t count) {
    return alloc_hw<T>(count);
  }
  using CmxMark = LinearAllocator;
  __attribute__((noinline)) CmxMark cmx_mark() const { return cmx_; }
  __attribute__((noinline)) void cmx_restore(const CmxMark &m) { cmx_ = m; }
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

  // ---- DMA
  // --------------------------------------------------------------------------------------------------
  __attribute__((noinline)) void dma_load(void *dst, std::uint64_t src,
                                          std::uint32_t width,
                                          std::uint32_t rows,
                                          std::uint32_t dstStride,
                                          std::uint32_t srcStride) {
    if (!ok_ || width == 0 || rows == 0)
      return;
    fills_wait(); // fused_rt: no synchronous load while a chain is in flight
    store_hazard(src, src + static_cast<std::uint64_t>(rows - 1) * srcStride +
                          width);
    DmaDrv::initDesc(desc_);
    configDmaCopy2D(desc_, reinterpret_cast<std::uint32_t>(dst), src, width,
                    rows, dstStride, srcStride);
    dma_.run(desc_);
  }
  void dma_load(void *dst, Ddr<void> src, std::uint32_t width,
                std::uint32_t rows, std::uint32_t dstStride,
                std::uint32_t srcStride) {
    dma_load(dst, src.addr, width, rows, dstStride, srcStride);
  }
  template <class T>
  void dma_load(T *dst, Ddr<T> src, std::uint32_t width, std::uint32_t rows,
                std::uint32_t dstStride, std::uint32_t srcStride) {
    dma_load(static_cast<void *>(dst), src.addr, width, rows, dstStride,
             srcStride);
  }
  template <class T>
  void dma_load_multicast(T *dst, Ddr<T> src, std::uint32_t width,
                          std::uint32_t rows, std::uint32_t dstStride,
                          std::uint32_t srcStride) {
    if (tile_id() == 0) {
      const std::uint32_t mask = (1u << num_tiles()) - 1;
      dma_load(reinterpret_cast<void *>(convertToCmxMulticastAddr(dst, mask)),
               src.addr, width, rows, dstStride, srcStride);
    }
  }
  __attribute__((noinline)) void dma_store(std::uint64_t dst, const void *src,
                                           std::uint32_t width,
                                           std::uint32_t rows,
                                           std::uint32_t dstStride,
                                           std::uint32_t srcStride) {
    if (!ok_ || width == 0 || rows == 0)
      return;
    frt::Ctx::dma_store(dst, src, width, rows, dstStride,
                        srcStride); // FRT_SYNC_STORE: waited for inside
  }
  template <class T>
  void dma_store(Ddr<T> dst, const T *src, std::uint32_t width,
                 std::uint32_t rows, std::uint32_t dstStride,
                 std::uint32_t srcStride) {
    dma_store(dst.addr, static_cast<const void *>(src), width, rows, dstStride,
              srcStride);
  }

  // ---- DPU
  // --------------------------------------------------------------------------------------------------
  __attribute__((noinline)) void dpu_matmul(half *out, const half *a,
                                            const half *bt, std::uint32_t M,
                                            std::uint32_t N, std::uint32_t K,
                                            std::uint32_t /*spatialX*/ = 0,
                                            std::int32_t /*mpe*/ = -1) {
    if (!ok_)
      return;
    frt::Ctx::matmul(out, a, bt, M, N,
                     K); // fused_rt's own M split / stencil (mm_tune)
  }
  __attribute__((noinline)) void reserve_qg(std::uint32_t partElems,
                                            std::uint32_t /*stElems*/,
                                            std::uint32_t = 0) {
    if (!ok_ || partElems <= i4PartElems_)
      return;
    i4PartElems_ = partElems;
    i4Part_ = alloc<half>(partElems);
  }
  __attribute__((noinline)) void
  dpu_matmul_i4g(half *out, const half *a, const std::uint8_t *q, const half *s,
                 std::uint32_t M, std::uint32_t N, std::uint32_t K,
                 std::uint32_t G, std::uint32_t sStride = 0,
                 bool defer = false) {
    qg(out, a, q, s, M, N, K, G, 4, sStride, defer);
  }
  __attribute__((noinline)) void
  dpu_matmul_i8g(half *out, const half *a, const std::int8_t *q, const half *s,
                 std::uint32_t M, std::uint32_t N, std::uint32_t K,
                 std::uint32_t G, std::uint32_t sStride = 0,
                 bool defer = false) {
    qg(out, a, reinterpret_cast<const std::uint8_t *>(q), s, M, N, K, G, 8,
       sStride, defer);
  }
  void dpu_matmul_i4g(half *out, const half *a, const std::uint8_t *q,
                      const std::uint8_t *s, std::uint32_t M, std::uint32_t N,
                      std::uint32_t K, std::uint32_t G, std::uint32_t sStride,
                      bool defer = false) {
    dpu_matmul_i4g(out, a, q, reinterpret_cast<const half *>(s), M, N, K, G,
                   sStride, defer);
  }
  void dpu_matmul_i8g(half *out, const half *a, const std::int8_t *q,
                      const std::int8_t *s, std::uint32_t M, std::uint32_t N,
                      std::uint32_t K, std::uint32_t G, std::uint32_t sStride,
                      bool defer = false) {
    dpu_matmul_i8g(out, a, q, reinterpret_cast<const half *>(s), M, N, K, G,
                   sStride, defer);
  }
  void qg_flush() {} // frt flushes its own pending reduction at the next
                     // matmul_qg / barrier / store

  // ---- rings: split streams only; a fill is one descriptor on the second
  // port, one in flight per SHAVE ------
  struct Ring {
    std::uint8_t *slotBase;
    std::uint32_t slotBytes, slots, depth, total, myUses;
    DmaDescriptor *fill; // depth descriptors, one per owned slot
    bool pending[kRingMaxDepth];
    bool armed;
  };
  enum : std::uint32_t {
    kRingSlotsNeShaves = 1,
    kRingOneTile = 2,
    kRingMulticastUnsupported = 3,
    kRingTooDeep = 4
  };

  std::uint32_t ring_slot_index(const Ring &r, std::uint32_t use) const {
    return shave_ + spt_ * (use % r.depth);
  }
  std::uint8_t *ring_slot(const Ring &r, std::uint32_t use) const {
    return r.slotBase + ring_slot_index(r, use) * r.slotBytes;
  }
  template <class T> T *ring_slot(const Ring &r, std::uint32_t use) const {
    return reinterpret_cast<T *>(ring_slot(r, use));
  }
  std::uint32_t ring_uses(const Ring &r) const { return r.myUses; }
  std::uint32_t ring_chunk(std::uint32_t use) const {
    return use * spt_ + shave_;
  }

  __attribute__((noinline)) Ring
  ring_begin(void *slotBase, std::uint32_t slotBytes, std::uint32_t slots,
             std::uint32_t total, std::uint32_t /*readyBase*/,
             std::uint32_t /*freeBase*/, std::int32_t /*port*/ = -1,
             bool multicast = true, bool local = false) {
    Ring r{};
    r.slotBase = static_cast<std::uint8_t *>(slotBase);
    r.slotBytes = slotBytes;
    r.slots = slots;
    r.total = total;
    if (!ok_)
      return r;
    if (slots == 0 || static_cast<std::int32_t>(slots) % spt_ != 0) {
      fail(kStatusRingRefused, kRingSlotsNeShaves);
      return r;
    }
    r.depth = slots / static_cast<std::uint32_t>(spt_);
    if (r.depth > kRingMaxDepth) {
      fail(kStatusRingRefused, kRingTooDeep);
      return r;
    }
    if (multicast && !local) {
      fail(kStatusRingRefused, kRingMulticastUnsupported);
      return r;
    }
    r.myUses = total > static_cast<std::uint32_t>(shave_)
                   ? (total - shave_ + spt_ - 1) / spt_
                   : 0;
    r.fill = ringFill_ + kRingMaxDepth * (ringSeq_++ % kRingPool);
    r.armed = true;
    return r;
  }
  __attribute__((noinline)) void
  ring_fill(Ring &r, std::uint32_t use, std::uint64_t src, std::uint32_t width,
            std::uint32_t rows, std::uint32_t srcStride) {
    if (!ok_ || !r.armed || use >= r.myUses)
      return;
    const std::uint32_t k = use % r.depth;
    DmaDescriptor *d = r.fill + k;
    fills_wait(); // one chain in flight per SHAVE (fused_rt pf_issue); also the
                  // slot's previous occupant
    store_hazard(src, src + static_cast<std::uint64_t>(rows - 1) * srcStride +
                          width);
    DmaDrv::initDesc(d);
    configDmaCopy2D(d, reinterpret_cast<std::uint32_t>(ring_slot(r, use)), src,
                    width, rows, width, srcStride);
    dma_.start(d, static_cast<int>(pfPort_));
    r.pending[k] = true;
    inflight_ = d;
    inflightRing_ = &r;
  }
  template <class T>
  void ring_fill(Ring &r, std::uint32_t use, Ddr<T> src, std::uint32_t width,
                 std::uint32_t rows, std::uint32_t srcStride) {
    ring_fill(r, use, src.addr, width, rows, srcStride);
  }
  __attribute__((noinline)) void ring_slot_acquire(Ring &r, std::uint32_t use) {
    if (!ok_ || !r.armed || use >= r.myUses)
      return;
    const std::uint32_t k = use % r.depth;
    if (r.pending[k]) {
      dma_.wait(r.fill + k);
      r.pending[k] = false;
      if (inflight_ == r.fill + k)
        inflight_ = nullptr;
    }
  }
  void ring_slot_release(Ring &, std::uint32_t) {}
  __attribute__((noinline)) void
  ring_dpu_submit(Ring &r, std::uint32_t use, half *out, const half *a,
                  const half *bt, std::uint32_t M, std::uint32_t N,
                  std::uint32_t K, std::uint32_t /*spatialX*/ = 0,
                  std::int32_t /*mpe*/ = -1) {
    if (!ok_ || !r.armed || use >= r.myUses)
      return;
    ring_slot_acquire(r, use);
    frt::Ctx::matmul(out, a, bt, M, N,
                     K); // waited for here: FREE is program order
  }
  void ring_dpu_wait(Ring &, std::uint32_t) {}
  __attribute__((noinline)) void ring_end(Ring &r) {
    if (!r.armed)
      return;
    for (std::uint32_t k = 0; k < r.depth; ++k)
      if (r.pending[k]) {
        dma_.wait(r.fill + k);
        r.pending[k] = false;
      }
    if (inflightRing_ == &r)
      inflight_ = nullptr;
    r.armed = false;
  }

  // ---- synchronisation
  // ----------------------------------------------------------------------------------------
  __attribute__((noinline)) void sync_grid() {
    if (!ok_)
      return;
    fills_wait();
    frt::Ctx::barrier(); // store_wait, dpu_wait, sync_.waitGlobal
  }
  __attribute__((noinline)) void sync_tile() {
    if (!ok_)
      return;
    frt::Ctx::sync_tile();
  }

  // ---- status record (64 B per SHAVE, the host runner's format;
  // npu_rt_device.hpp finish) ----------------------
  static constexpr std::uint32_t kStatusBytes = 64;
  __attribute__((noinline)) void finish(const std::uint32_t *extra = nullptr,
                                        std::uint32_t nExtra = 0) {
    fills_wait();
    store_wait();
    if (statusAddr_ == 0)
      return;
    const std::uint64_t a =
        statusAddr_ + static_cast<std::uint64_t>(kStatusBytes) * worker();
    DdrAccessor<std::uint32_t> s0(a), s1(a + 4), s2(a + 8), s3(a + 12),
        s4(a + 16), s5(a + 20), s6(a + 24), s7(a + 28), s8(a + 32), s9(a + 36);
    s0 = ok_ ? kStatusOk : code_;
    s1 = detail_;
    s2 = cmxFree0_;
    s3 = cmxFree0_ - cmx_.getAvailableSize();
    s4 = 0;
    s5 = 0;
    s6 = 0;
    s7 = 0xF27u; // marks the fused_rt runtime
    s8 = tStart_;
    s9 = perf_ticks();
    for (std::uint32_t i = 0; i < nExtra && i < 6; ++i) {
      DdrAccessor<std::uint32_t> sx(a + 40 + 4 * i);
      sx = extra[i];
    }
    sync::memoryFence();
  }

private:
  static std::uint32_t perf_ticks() { return getRegWordVal(0x2f020000U); }

  void fail(std::uint32_t code, std::uint32_t detail) {
    if (ok_) {
      ok_ = false;
      code_ = code;
      detail_ = detail;
    }
  }
  void fills_wait() {
    if (inflight_ != nullptr) {
      dma_.wait(inflight_);
      if (inflightRing_ != nullptr)
        inflightRing_->pending[static_cast<std::uint32_t>(
            inflight_ - inflightRing_->fill)] = false;
      inflight_ = nullptr;
    }
  }
  void qg(half *out, const half *a, const std::uint8_t *q, const half *s,
          std::uint32_t M, std::uint32_t N, std::uint32_t K, std::uint32_t G,
          std::uint32_t bits, std::uint32_t sStride, bool defer) {
    if (!ok_)
      return;
    const std::uint32_t ng = K / G;
    if (i4Part_ == nullptr || M * ng * N * ng > i4PartElems_)
      reserve_qg(M * ng * N * ng, ng * N);
    if (!ok_)
      return;
    frt::Ctx::matmul_qg(out, a, q, s, i4Part_, M, N, K, G, bits,
                        defer && sStride != 0, sStride, 0);
  }

  std::uint64_t statusAddr_ = 0;
  std::uint32_t code_ = 0, detail_ = 0;
  std::uint32_t tStart_ = 0;
  half *i4Part_ = nullptr;
  std::uint32_t i4PartElems_ = 0;
  DmaDescriptor *ringFill_ = nullptr;
  std::uint32_t ringSeq_ = 0;
  DmaDescriptor *inflight_ = nullptr;
  Ring *inflightRing_ = nullptr;
};

} // namespace npu

#endif // TL_TEMPLATES_INTEL_NPU_NPU_RT_DEVICE_FRT_HPP_
