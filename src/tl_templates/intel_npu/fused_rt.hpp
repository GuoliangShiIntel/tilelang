// VERBATIM COPY of npu_dsl's kernels_poc/common/fused_rt.hpp (D:\dsl\libraries.npu.dsl, last commit 14b47a84, sha256 60AB166AFC74)
// with one mechanical change: private: -> protected: so npu_rt_device_frt.hpp can derive from frt::Ctx.
// Do not edit otherwise; re-copy from npu_dsl when it changes.
// fused_rt.hpp -- the runtime surface the npu_dsl fused-kernel emitter (npu_dsl/fused/emit.py) writes against.
//
// Two backends with one API:
//   * device (default): generic_compute toolkit (DmaDrv, DpuMatMul, Sync) on a SHAVE;
//   * host simulation (-DFRT_HOST_SIM): plain C++ on x86, every SHAVE a thread (fused_rt_host.hpp).
// A kernel sees: worker()/workers() (global SHAVE index over all tiles), alloc<T>(n) (worker-private CMX),
// dma_load/dma_store (2-D row-strided DDR <-> CMX), matmul (fp16 DPU, out[M,N] = a[M,K] . bt[N,K]^T),
// barrier() (every SHAVE of every tile) and status(code, detail).
#pragma once

#if defined(FRT_HOST_SIM)
#include "fused_rt_host.hpp"
#else

#include <shave/kernel/include/kernel.hpp>
#include <shave/shave_lib/include/kernel_utils.hpp>

#include <common/include/buffer_arena.hpp>
#include <hw/cmx/include/cmx_flush.hpp>
#include <hw/cmx/include/cmx_multicast.hpp>
#include <shave/hw/nce/include/hgl_barriers_40xx.hpp>
#include <hw/ddr/include/ddr_accessor_40xx.hpp>
#include <lib/dma/include/dma_utils.hpp>
#include <lib/dpu/include/dpu_matmul.hpp>
#include <shave/sync/include/memory_fence.hpp>

#include <cstddef>
#include <new>
#include <cstdint>
#include <cstring>
#include <limits>

#define FRT_NOINLINE __attribute__((noinline))   // an emitted phase's own stack frame (npu_dsl/fused/emit.py)
#include <cmath>

#define FRT_KERNEL_ENTRY(numParams, params, env) void kernel_entry(int numParams, std::uint64_t* params, const npu::Environment& env)

// runtime-internal profile sub-sites (sites kProfMaxSites - 8 ..: inside a Ctx call), -DFRT_PROFILE only
#if defined(FRT_TRACE)
#if !defined(FRT_PROFILE)
#error "FRT_TRACE needs FRT_PROFILE"
#endif
#if !defined(FRT_TRACE_BYTES)
#define FRT_TRACE_BYTES (4u << 20)   // per-worker DDR trace area; keep equal to qwen36_run's kTraceBytes
#endif
#endif
#ifndef FRT_PROF_MAX_SITES
#define FRT_PROF_MAX_SITES 320   // a kernel with more profile sites defines its count (record of 4 + 3 n words)
#endif
#if defined(FRT_PROFILE)
#define FRT_SUB_BEGIN(i) const std::uint32_t frt_sub##i = now()
#define FRT_SUB_END(i) prof_add(kProfMaxSites - 8 + (i), frt_sub##i)
#else
#define FRT_SUB_BEGIN(i) (void)0
#define FRT_SUB_END(i) (void)0
#endif

namespace frt {

static constexpr std::uint32_t kMaxDpuN = 1024;   // the default weight-table rows (and the row-major scale transpose's N)
// Ctx::mc_rounds' hardware barrier ids (8: READY / FREE x slot x use parity); 12..19 run on PTL (E7 hand flow, Round 777)
static constexpr std::uint32_t kMcBarrier = 12;
// Ctx::hw_sync's hardware barrier ids (3, rotating): disjoint from mc_rounds', which a kernel may run between two syncs
static constexpr std::uint32_t kHwSyncBarrier = kMcBarrier + 8;
static_assert(kHwSyncBarrier + 3 <= 48, "PTL has 48 hardware barriers (npu_dsl/platform_configs/ptl.json)");
// one streamed block of Ctx::mc_rounds: rows x width bytes at src, rows srcStride apart
struct McBlock {
    std::uint64_t src;
    std::uint32_t width, rows, srcStride;
};
// weight-table rows per DPU path: the most output channels one of the kernel's workloads has (emit.py defines them
// when above kMaxDpuN; passes.ctx_cmx_bytes counts the same)
#ifndef FRT_DPU_F16_ROWS
#define FRT_DPU_F16_ROWS 1024
#endif
#ifndef FRT_DPU_Q_ROWS
#define FRT_DPU_Q_ROWS 1024
#endif
static constexpr std::uint32_t kDpuF16Rows = FRT_DPU_F16_ROWS, kDpuQRows = FRT_DPU_Q_ROWS;
static constexpr std::uint32_t kAlign = 64;
// prefetch chains (npu_dsl/fused/passes.plan_prefetch): kPfSets levels x 2 slots x kPfDescs descriptors. One level:
// a second chain in flight on the same DMA port and direction hangs (PTL), and the other port is the synchronous loads'.
static constexpr std::uint32_t kPfSets = 1, kPfDescs = 8;
// mm_tune stencils applied, bit i = CUBOID i (0: 4x16, 1: 8x16). 8x16 is off: with it the Qwen3-4B prefill output is
// wrong although dpu_rate_probe's DUMP2 passes for the same shapes (unexplained, 2026-11-02)
static constexpr std::uint32_t kDpuStencil = 1;
// gather chains (emit: a loop whose body is only loads, e.g. rows at indices read from CMX): up to kGatherDescs
// descriptors run as one synchronous chain on the synchronous loads' port. A kernel without one defines 0 (emit.py).
#ifndef FRT_GATHER_DESCS
#define FRT_GATHER_DESCS 64
#endif
static constexpr std::uint32_t kGatherDescs = FRT_GATHER_DESCS;
// the DPU paths a kernel uses (emit.py defines 0 for an unused one): fp16 matmul, quantised-weight matmul_i4g / i8g
#ifndef FRT_DPU_F16
#define FRT_DPU_F16 1
#endif
#ifndef FRT_DPU_Q
#define FRT_DPU_Q 1
#endif
// a quantised matmul with row-major scales [N, K/G] (transposed by the runtime into sT_); 0 = all group-major
#ifndef FRT_DPU_Q_ROWMAJOR
#define FRT_DPU_Q_ROWMAJOR 1
#endif
static constexpr bool kDpuF16 = FRT_DPU_F16, kDpuQ = FRT_DPU_Q, kDpuQRowMajor = FRT_DPU_Q && FRT_DPU_Q_ROWMAJOR;
// a quantised matmul with one group per row: the scales go into the weight table's per-channel PPE multiplier and the
// DPU writes the output itself (no partial products, no SHAVE reduction)
#ifndef FRT_DPU_Q_DIRECT
#define FRT_DPU_Q_DIRECT 0
#endif
static constexpr bool kDpuQDirect = FRT_DPU_Q_DIRECT;
// tile-shared CMX buffers (emit.py defines 1): their bytes may be reused across a global barrier
#ifndef FRT_TILE_BUFFERS
#define FRT_TILE_BUFFERS 0
#endif
// matmul_async in the kernel (emit.py defines 1): only then are the in-flight DPU checks compiled
#ifndef FRT_DPU_ASYNC
#define FRT_DPU_ASYNC 0
#endif
// the DMA driver's trap descriptors (DmaDrv40xx::init: cmx.METADATA + 0x129C0, 8 rows of 224 B), with margin
static constexpr std::uint32_t kTrapLo = 0x12000, kTrapHi = 0x14000;
// PPE multiplier of matmul_i4g's per-group partial products (keeps them in fp16 range; undone in the reduction)
static constexpr float kI4PartScale = 1.0f / 16.0f;
static constexpr float kI8PartScale = 1.0f / 256.0f;   // matmul_i8g's (|q| <= 128)

static inline std::uint32_t umin(std::uint32_t a, std::uint32_t b) { return a < b ? a : b; }
static inline std::uint32_t umax(std::uint32_t a, std::uint32_t b) { return a > b ? a : b; }
static constexpr std::uint32_t alignUp(std::uint32_t a, std::uint32_t b) { return (a + b - 1) / b * b; }

// per-SHAVE CMX the Ctx constructor allocates; the emitter's figure (passes.ctx_cmx_bytes, less its slack) must match
static constexpr std::uint32_t kCtxCmxBytes =
    alignUp(sizeof(npu::DmaDescriptor) * (2 + kPfSets * 2 * kPfDescs + kGatherDescs), kAlign) +
    (kDpuF16 ? alignUp(sizeof(npu::DpuMatMul::DescriptorsBuffer), kAlign) + alignUp(16 * kDpuF16Rows, kAlign) : 0) +
    (kDpuQ ? alignUp(sizeof(npu::DpuMatMul::DescriptorsBuffer), kAlign) + alignUp(16 * kDpuQRows, kAlign) : 0) +
    (kDpuQRowMajor ? alignUp(2 * 2 * kMaxDpuN, kAlign) : 0);
#if defined(FRT_CTX_CMX_BYTES)
static_assert(kCtxCmxBytes == FRT_CTX_CMX_BYTES, "runtime CMX reserve differs from npu_dsl/fused/passes.ctx_cmx_bytes");
#endif
// the 64-bit device address a pointer-table row (npu_dsl/fused TableSpec) stores as two u32 words (low first), read as
// one 64-bit value like a kernel parameter: composing it from two u32 values miscompiles on the SHAVE when bit 31 of
// the low word is set (the high word comes out wrong)
static inline std::uint64_t addr64_at(const std::uint32_t& lo) {
    std::uint64_t v;
    std::memcpy(&v, &lo, sizeof(v));
    return v;
}

// dst[c * rows + r] = src[r * cols + c] for rows % 8 == 0, cols % 8 == 0: 8 x 8 blocks of half8 shuffles
static inline void transpose_h(half* __restrict dst, const half* __restrict src, std::uint32_t rows, std::uint32_t cols) {
    for (std::uint32_t r0 = 0; r0 < rows; r0 += 8)
        for (std::uint32_t c0 = 0; c0 < cols; c0 += 8) {
            half8 r[8];
            for (int i = 0; i < 8; ++i) r[i] = *reinterpret_cast<const half8*>(src + (r0 + i) * cols + c0);
            const half8 t0 = __builtin_shufflevector(r[0], r[1], 0, 8, 1, 9, 2, 10, 3, 11), t1 = __builtin_shufflevector(r[0], r[1], 4, 12, 5, 13, 6, 14, 7, 15);
            const half8 t2 = __builtin_shufflevector(r[2], r[3], 0, 8, 1, 9, 2, 10, 3, 11), t3 = __builtin_shufflevector(r[2], r[3], 4, 12, 5, 13, 6, 14, 7, 15);
            const half8 t4 = __builtin_shufflevector(r[4], r[5], 0, 8, 1, 9, 2, 10, 3, 11), t5 = __builtin_shufflevector(r[4], r[5], 4, 12, 5, 13, 6, 14, 7, 15);
            const half8 t6 = __builtin_shufflevector(r[6], r[7], 0, 8, 1, 9, 2, 10, 3, 11), t7 = __builtin_shufflevector(r[6], r[7], 4, 12, 5, 13, 6, 14, 7, 15);
            const half8 u0 = __builtin_shufflevector(t0, t2, 0, 1, 8, 9, 2, 3, 10, 11), u1 = __builtin_shufflevector(t0, t2, 4, 5, 12, 13, 6, 7, 14, 15);
            const half8 u2 = __builtin_shufflevector(t1, t3, 0, 1, 8, 9, 2, 3, 10, 11), u3 = __builtin_shufflevector(t1, t3, 4, 5, 12, 13, 6, 7, 14, 15);
            const half8 u4 = __builtin_shufflevector(t4, t6, 0, 1, 8, 9, 2, 3, 10, 11), u5 = __builtin_shufflevector(t4, t6, 4, 5, 12, 13, 6, 7, 14, 15);
            const half8 u6 = __builtin_shufflevector(t5, t7, 0, 1, 8, 9, 2, 3, 10, 11), u7 = __builtin_shufflevector(t5, t7, 4, 5, 12, 13, 6, 7, 14, 15);
            half8* d = reinterpret_cast<half8*>(dst + c0 * rows + r0);
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

class Ctx {
public:
    explicit Ctx(const npu::Environment& env, int numParams, std::uint64_t* params, int statusParam)
        : env_(env), cmx_(env.getCmxAllocator()), dma_(env.getDmaDrv()), sync_(env.getSync()),
          statusAddr_(statusParam < numParams ? params[statusParam] : 0) {
        const auto& info = env.getShaveInfo();
        shave_ = info.getShaveId();
        spt_ = info.getNumShavesPerTile();
        tile_ = info.getTileId();
        tiles_ = info.getNumTiles();
        cmxFree0_ = cmx_.getAvailableSize();
        desc_ = alloc_hw<npu::DmaDescriptor>(2 + kPfSets * 2 * kPfDescs + kGatherDescs);
        stDesc_ = desc_ + 1;
        pfDesc_ = desc_ + 2;
        gaDesc_ = pfDesc_ + kPfSets * 2 * kPfDescs;
        for (std::uint32_t i = 0; i < kPfSets * 2; ++i) pf_[i].desc = pfDesc_ + i * kPfDescs;
        pfPort_ = (tile_ + shave_ + 1) % 2;   // DmaDrv40xx's default port is (tile + shave) % ports
        if (kDpuF16) {
            dpuDesc_ = alloc_hw<npu::DpuMatMul::DescriptorsBuffer>(1);
            wt_ = alloc<std::uint32_t>(4 * kDpuF16Rows);
            mm_ = new (mmStore_) npu::DpuMatMul(env_.getDpuDrv(), dpuDesc_, wt_);
        }
        if (kDpuQ) {
            dpuDescI4_ = alloc_hw<npu::DpuMatMul::DescriptorsBuffer>(1);
            wtI4_ = alloc<std::uint32_t>(4 * kDpuQRows);
            if (kDpuQRowMajor) sT_ = alloc<half>(2 * kMaxDpuN);   // transpose_h writes 8 x N for ng >= 4 groups: N <= kMaxDpuN / 4
            mmI4_ = new (mmI4Store_) npu::DpuMatMul(env_.getDpuDrv(), dpuDescI4_, wtI4_);
        }
#if defined(FRT_PROFILE)
        prof_ = alloc<std::uint32_t>(kProfWords);
        if (prof_ != nullptr) {
            for (std::uint32_t i = 0; i < kProfWords; ++i) prof_[i] = 0;
            prof_[0] = now();
        }
#endif
#if defined(FRT_TRACE)
        tr_ = alloc<std::uint32_t>(2 * (kTraceStage + kTraceSlack));
        trDesc_ = alloc_hw<npu::DmaDescriptor>(1);
#endif
    }

#if defined(FRT_PROFILE)
    // Per-site profile (-DFRT_PROFILE): every emitted DMA / DPU / SHAVE-op / barrier call site `id` accumulates
    // {ticks, calls, end timestamp of the last call} in worker-private CMX; prof_flush writes the record after the
    // status area: statusAddr + 32 * workers + worker * kProfBytes. Header words: t_begin, t_end, nsites, magic.
    static constexpr std::uint32_t kProfMaxSites = FRT_PROF_MAX_SITES;
    static constexpr std::uint32_t kProfWords = 4 + 3 * kProfMaxSites;
    static constexpr std::uint32_t kProfBytes = (4 * kProfWords + 4095) / 4096 * 4096;
    static std::uint32_t now() { return *reinterpret_cast<volatile std::uint32_t*>(0x2f020000u); }   // NCE_TIME_FREE_RUNNING
#if defined(FRT_TRACE)
    FRT_NOINLINE   // keeps the trace build's kernel text within the blob section
#endif
    void prof_add(std::uint32_t id, std::uint32_t t0) {
        const std::uint32_t t1 = now();
        if (prof_ == nullptr || id >= kProfMaxSites) return;
        std::uint32_t* s = prof_ + 4 + 3 * id;
        s[0] += t1 - t0;
        s[1] += 1;
        s[2] = t1;
#if defined(FRT_TRACE)
        trace_add(id, t0, t1);
#endif
    }
#if defined(FRT_TRACE)
    // Timeline trace (-DFRT_TRACE, implies FRT_PROFILE). Events of two words {w0, kind << 29 | w1}:
    //   kind 0 a call of site `site` (the emitter's FRT_PROF_BEGIN/END), 1 a DMA in flight (issue -> observed
    //   completion: the first instrumentation point that sees the descriptor's watermark), 2 a DPU workload as the
    //   SHAVE sees it (start -> wait returned), 3 the SHAVE waiting for the DPU: w0 = t_begin, w1 = site << 20 |
    //   min(ticks, 2^20 - 1) (kinds 1-3: the call site that issued / waited);
    //   kind 4 / 5 the DPU hardware profiler's (HWP) IDU / ODU record of the preceding kind-2 workload: w0 = stop
    //   timestamp (NCE timer, low word), w1 = duration (NPU4: dpu_clk cycles, hwp_stat_mode 3; NPU5: NCE timer
    //   ticks, mode 7);
    //   kind 6: w0 = the full ticks of the preceding event when they do not fit its 20 bits.
    // Staged in CMX and spilled (synchronous DMA on a private descriptor, logged as a call of site kTraceSpillSite)
    // to the worker's DDR area after the profile records: statusAddr + (32 + kProfBytes) * workers + worker *
    // kTraceBytes. Area header (written by prof_flush): words {events stored, events dropped (area full), magic
    // "TRCE", t_begin, npu major}, then the events from byte 64.
    static constexpr std::uint32_t kTraceStage = 256, kTraceSlack = 16;
#if defined(FRT_PROF_TRACE_CMX_BYTES)
    static_assert(alignUp(4 * kProfWords, kAlign) + alignUp(8 * (kTraceStage + kTraceSlack), kAlign) +
                      alignUp(sizeof(npu::DmaDescriptor), kAlign) == FRT_PROF_TRACE_CMX_BYTES,
                  "profile + trace CMX differs from npu_dsl/fused/passes.PROF_TRACE_CMX_BYTES");
#endif
    static constexpr std::uint32_t kTraceBytes = FRT_TRACE_BYTES;
    static constexpr std::uint32_t kTraceSpillSite = kProfMaxSites - 1;
    enum : std::uint32_t { kEvCall = 0, kEvDma = 1, kEvDpu = 2, kEvDpuWait = 3, kEvHwpIdu = 4, kEvHwpOdu = 5, kEvLong = 6 };
    std::uint64_t trace_base() const {
        return statusAddr_ + (32ull + kProfBytes) * workers() + (std::uint64_t)kTraceBytes * worker();
    }
    FRT_NOINLINE void trace_raw(std::uint32_t kind, std::uint32_t w0, std::uint32_t w1) {
        if (tr_ == nullptr || trN_ >= kTraceStage + kTraceSlack) return;
        tr_[2 * trN_] = w0;
        tr_[2 * trN_ + 1] = (kind << 29) | (w1 & 0x1FFFFFFFu);
        ++trN_;
    }
    FRT_NOINLINE void trace_ev(std::uint32_t kind, std::uint32_t site, std::uint32_t t0, std::uint32_t t1) {
        const std::uint32_t d = t1 - t0;
        trace_raw(kind, t0, ((site & 0x1FFu) << 20) | (d < 0xFFFFFu ? d : 0xFFFFFu));
        if (d >= 0xFFFFFu) trace_raw(kEvLong, d, 0);
    }
    // completion of the in-flight store / prefetch chains, observed at every call boundary
    FRT_NOINLINE void trace_poll(std::uint32_t t) {
        if (stPending_ && trStOpen_ && static_cast<volatile npu::DmaDescriptor*>(stDesc_)->watermark != 0) {
            trace_ev(kEvDma, trStSite_, trStT0_, t);
            trStOpen_ = false;
        }
        for (std::uint32_t i = 0; i < kPfSets * 2; ++i) {
            Chain& c = pf_[i];
            if (c.pending && c.trOpen && c.trDone == 0 && static_cast<volatile npu::DmaDescriptor*>(c.tail)->watermark != 0) c.trDone = t;
        }
    }
    FRT_NOINLINE void trace_add(std::uint32_t id, std::uint32_t t0, std::uint32_t t1) {
        if (tr_ == nullptr || trBusy_) return;
        trace_poll(t1);
        trace_ev(kEvCall, id, t0, t1);
        if (trN_ >= kTraceStage) trace_spill(true);
    }
    FRT_NOINLINE void trace_spill(bool log) {
        if (tr_ == nullptr || statusAddr_ == 0 || trN_ == 0 || trBusy_) return;
        store_wait();   // a second store in flight on the same port and direction hangs (PTL)
        trBusy_ = true;
        const std::uint32_t t0 = now();
        const std::uint32_t cap = (kTraceBytes - 64) / 8;
        const std::uint32_t n = trN_ < cap - trStored_ ? trN_ : cap - trStored_;
        if (n > 0) {
            npu::flushCmxWrites();
            npu::DmaDrv::initDesc(trDesc_);
            npu::configDmaCopy2D(trDesc_, trace_base() + 64 + 8ull * trStored_, (std::uint32_t)tr_, n * 8, 1, n * 8, n * 8);
            dma_.run(trDesc_);
        }
        trDropped_ += trN_ - n;
        trStored_ += n;
        trN_ = 0;
        trBusy_ = false;
        if (log) trace_ev(kEvCall, kTraceSpillSite, t0, now());
    }
    FRT_NOINLINE void trace_dpu(std::uint32_t ts, std::uint32_t tw, const npu::DpuMatMul::DescriptorsBuffer* db) {
        const std::uint32_t te = now();
        trace_ev(kEvDpuWait, trSite_, tw, te);
        trace_ev(kEvDpu, trSite_, ts, te);
        const volatile auto& h = db->hwpStats;
        trace_raw(kEvHwpIdu, h.idu_timestamp_l, h.idu_workload_duration);
        trace_raw(kEvHwpOdu, h.odu_timestamp_l, h.odu_workload_duration);
    }
    std::uint32_t trSite_ = 0;   // the call site being run (FRT_PROF_BEGIN)
#endif
    void prof_flush(std::uint32_t nsites) {
#if defined(FRT_TRACE)
        store_wait();
        trace_spill(false);
        if (tr_ != nullptr && statusAddr_ != 0 && prof_ != nullptr) {
            tr_[0] = trStored_;
            tr_[1] = trDropped_;
            tr_[2] = 0x54524345u;   // "TRCE"
            tr_[3] = prof_[0];
            tr_[4] = __npu_major__;
            npu::flushCmxWrites();
            npu::DmaDrv::initDesc(trDesc_);
            npu::configDmaCopy2D(trDesc_, trace_base(), (std::uint32_t)tr_, 32, 1, 32, 32);
            dma_.run(trDesc_);
        }
#endif
        if (prof_ == nullptr || statusAddr_ == 0) return;
        prof_[1] = now();
        prof_[2] = nsites;
        prof_[3] = 0x50524F46u;   // "PROF"
        dma_store(statusAddr_ + 32ull * workers() + (std::uint64_t)kProfBytes * worker(), prof_, kProfWords * 4, 1, kProfWords * 4);
    }
#endif

    std::uint32_t worker() const { return tile_ * spt_ + shave_; }
    std::uint32_t workers() const { return tiles_ * spt_; }
    bool ok() const { return ok_; }

    // the tile CMX this kernel asks for: the runtime's structures (with this build's profile / trace areas) and an
    // arena of `arena` bytes per SHAVE
    std::uint32_t cmx_need(std::uint32_t arena) const {
        std::uint32_t b = kCtxCmxBytes + alignUp(arena, kAlign);
#if defined(FRT_PROFILE)
        b += alignUp(4 * kProfWords, kAlign);
#endif
#if defined(FRT_TRACE)
        b += alignUp(8 * (kTraceStage + kTraceSlack), kAlign) + alignUp(sizeof(npu::DmaDescriptor), kAlign);
#endif
        return b * spt_;
    }

    // this worker's first iteration of a distributed loop of n iterations (stride workers()). The deal continues
    // where the previous distributed loop stopped, so the remainders of successive loops land on different workers;
    // every worker must run the same sequence of distributed loops (passes._check_uniform).
    std::uint32_t dist_first(std::uint32_t n) {
        const std::uint32_t w = workers(), f = (worker() + w - distRot_) % w;
        distRot_ = (distRot_ + n % w) % w;
        return f;
    }
    // the same over the tiles (every SHAVE of a tile gets the tile's iterations) and over the SHAVEs of this tile
    std::uint32_t dist_first_tile(std::uint32_t n) {
        const std::uint32_t f = (tile_ + tiles_ - tileRot_) % tiles_;
        tileRot_ = (tileRot_ + n % tiles_) % tiles_;
        return f;
    }
    std::uint32_t dist_first_shave(std::uint32_t n) {
        const std::uint32_t f = (shave_ + spt_ - shaveRot_) % spt_;
        shaveRot_ = (shaveRot_ + n % spt_) % spt_;
        return f;
    }
    std::uint32_t tiles() const { return tiles_; }
    std::uint32_t tile() const { return tile_; }
    std::uint32_t spt() const { return spt_; }
    std::uint32_t shave() const { return shave_; }

    template <class C>
    C config(std::uint64_t addr) const { return *npu::DdrAccessor<C>(addr); }

    // worker-private CMX: every SHAVE of a tile runs the same allocation sequence on the tile's allocator, so one
    // allocation of spt_ slots gives each SHAVE its own slot
    template <class T>
    T* alloc(std::uint32_t count) {
        const std::uint32_t bytes = alignUp(count * sizeof(T), kAlign);
        auto* base = reinterpret_cast<std::uint8_t*>(cmx_.alloc<kAlign>(bytes * spt_));
        if (base == nullptr) { ok_ = false; return nullptr; }
        return reinterpret_cast<T*>(base + bytes * shave_);
    }

    // an arena of `bytes` per SHAVE whose copies mirror each other around the middle of the tile's block (returned):
    // SHAVE 0's offset o of a size-s buffer is mid - o - s, SHAVE 1's mid + o, so a tile-shared buffer of 2s bytes at
    // offset 0 is [mid - s, mid + s) for both (npu_dsl/fused/passes.plan_cmx). Two SHAVEs per tile only.
    std::uint8_t* alloc_mirror(std::uint32_t bytes) {
        bytes = alignUp(bytes, kAlign);
        auto* p = alloc<std::uint8_t>(bytes);
        if (p == nullptr || spt_ != 2) { ok_ = false; return nullptr; }
        return p - bytes * shave_ + bytes;
    }
    std::uint8_t* mirror(std::uint8_t* mid, std::uint32_t off, std::uint32_t size) const {
        return shave_ ? mid + off : mid - off - size;
    }

    // ---- DMA. Loads are synchronous (dma_load) or prefetch chains (pf_*); stores are asynchronous: a store is waited
    // for before a load whose DDR range overlaps its destination, before any write to its CMX source (guard), at a
    // barrier and at the end of the kernel.
    // Hardware descriptors (DMA, DPU) must not lie in the DMA driver's trap-descriptor window of the tile's CMX
    // (cmx.METADATA, cmxBase + 0x129C0, DmaDrv40xx::init): a DMA descriptor there is taken for a trap (no transfer, or
    // a hung chain). The block of all SHAVEs is checked, so every SHAVE of a tile makes the same choice (the
    // allocation sequences must stay identical); a block that overlaps is dropped and the next one taken.
    template <class T>
    T* alloc_hw(std::uint32_t count) {
        const std::uint32_t bytes = alignUp(count * sizeof(T), kAlign);
        for (int tries = 0; tries < 4; ++tries) {
            T* p = alloc<T>(count);
            if (p == nullptr) return nullptr;
            const std::uint32_t base = (std::uint32_t)p - bytes * shave_, end = base + bytes * spt_;
            const std::uint32_t lo = ((base >> 21) << 21) + kTrapLo, hi = ((base >> 21) << 21) + kTrapHi;
            if (end <= lo || base >= hi) return p;
        }
        ok_ = false;
        return nullptr;
    }

    void dma_load(void* dst, std::uint64_t src, std::uint32_t width, std::uint32_t rows, std::uint32_t srcStride) {
        if (width == 0 || rows == 0) return;
        load_wait();
        store_hazard(src, src + (std::uint64_t)(rows - 1) * srcStride + width);
        guard(dst, width * rows);
        npu::DmaDrv::initDesc(desc_);
        npu::configDmaCopy2D(desc_, (std::uint32_t)dst, src, width, rows, width, srcStride);
#if defined(FRT_TRACE)
        const std::uint32_t trT0 = now();
        dma_.run(desc_);
        trace_ev(kEvDma, trSite_, trT0, now());
#else
        dma_.run(desc_);
#endif
    }

    // dma_load split in two (passes.plan_early_loads): started here, waited for by load_wait where the source loads.
    // It uses the synchronous loads' descriptor, so at most one is in flight.
    void load_start(void* dst, std::uint64_t src, std::uint32_t width, std::uint32_t rows, std::uint32_t srcStride) {
        if (width == 0 || rows == 0) return;
        load_wait();
        store_hazard(src, src + (std::uint64_t)(rows - 1) * srcStride + width);
        guard(dst, width * rows);
        npu::DmaDrv::initDesc(desc_);
        npu::configDmaCopy2D(desc_, (std::uint32_t)dst, src, width, rows, width, srcStride);
        dma_.start(desc_);
        ldPending_ = true;
    }

    void load_wait() {
        if (ldPending_) {
            dma_.wait(desc_);
            ldPending_ = false;
        }
    }

    void dma_store(std::uint64_t dst, const void* src, std::uint32_t width, std::uint32_t rows, std::uint32_t dstStride,
                   std::uint32_t srcStride = 0) {
        if (width == 0 || rows == 0) return;
        if (srcStride == 0) srcStride = width;
        store_wait();
        dpu_r(src, (rows - 1) * srcStride + width);
        npu::flushCmxWrites();
        npu::DmaDrv::initDesc(stDesc_);
        npu::configDmaCopy2D(stDesc_, dst, (std::uint32_t)src, width, rows, dstStride, srcStride);
#if defined(FRT_TRACE)
        trStSite_ = trSite_;
        trStT0_ = now();
        trStOpen_ = true;
#endif
        dma_.start(stDesc_);
        stPending_ = true;
#if defined(FRT_SYNC_STORE)
        store_wait();
#endif
        stSrc_ = (std::uint32_t)src;
        stSrcEnd_ = stSrc_ + (rows - 1) * srcStride + width;
        stDst_ = dst;
        stDstEnd_ = dst + (std::uint64_t)(rows - 1) * dstStride + width;
    }

    void store_wait() {
        if (stPending_) {
            dma_.wait(stDesc_);
            stPending_ = false;
#if defined(FRT_TRACE)
            if (trStOpen_ && !trBusy_) trace_ev(kEvDma, trStSite_, trStT0_, now());
            trStOpen_ = false;
#endif
        }
    }

    // before a write to CMX [p, p + bytes): the pending store must have read its source first, and the DPU workload
    // in flight must be done with its operands and output there
    void guard(const void* p, std::uint32_t bytes) {
        const std::uint32_t a = (std::uint32_t)p;
        if (stPending_ && a < stSrcEnd_ && stSrc_ < a + bytes) store_wait();
#if FRT_DPU_ASYNC
        if (dpuPending_ && ((a < dpuOe_ && dpuO_ < a + bytes) || (a < dpuAe_ && dpuA_ < a + bytes) ||
                            (a < dpuBe_ && dpuB_ < a + bytes)))
            dpu_wait();
#endif
    }

    void store_hazard(std::uint64_t src, std::uint64_t end) {
        if (stPending_ && src < stDstEnd_ && stDst_ < end) store_wait();
    }

    // gather chain: gather_add queues a 2-D copy, gather_flush runs the queued ones as one chain and waits for it
    // (a full chain is flushed by the next add)
    void gather_add(void* dst, std::uint64_t src, std::uint32_t width, std::uint32_t rows, std::uint32_t srcStride) {
        if (width == 0 || rows == 0) return;
        if (gaN_ == kGatherDescs) gather_flush();
        store_hazard(src, src + (std::uint64_t)(rows - 1) * srcStride + width);
        guard(dst, width * rows);
        npu::DmaDescriptor* d = gaDesc_ + gaN_;
        npu::DmaDrv::initDesc(d);
        npu::configDmaCopy2D(d, (std::uint32_t)dst, src, width, rows, width, srcStride);
        if (gaN_ > 0) d[-1].link_address = (std::uint32_t)d;
        gaN_ = gaN_ + 1;
    }

    void gather_flush() {
        if (gaN_ == 0) return;
        load_wait();   // an early load in flight on the same port and direction would hang the chain (PTL)
        npu::DmaDescriptor* tail = gaDesc_ + (gaN_ - 1);
        gaN_ = 0;
#if defined(FRT_TRACE)
        const std::uint32_t trT0 = now();
#endif
        dma_.start(gaDesc_, tail, npu::DmaSource::DDR);
        dma_.wait(tail);
#if defined(FRT_TRACE)
        trace_ev(kEvDma, trSite_, trT0, now());
#endif
    }

    // prefetch chain (set, slot): descriptors added with pf_add run as one chain from pf_issue; pf_wait waits for it
    void pf_add(std::uint32_t set, std::uint32_t slot, void* dst, std::uint64_t src, std::uint32_t width, std::uint32_t rows,
                std::uint32_t srcStride) {
        if (width == 0 || rows == 0) return;
        Chain& c = pf_[set * 2 + slot];
        store_hazard(src, src + (std::uint64_t)(rows - 1) * srcStride + width);
        guard(dst, width * rows);
        npu::DmaDescriptor* d = c.desc + c.n;
        npu::DmaDrv::initDesc(d);
        npu::configDmaCopy2D(d, (std::uint32_t)dst, src, width, rows, width, srcStride);
        if (c.n > 0) d[-1].link_address = (std::uint32_t)d;
        c.n = c.n + 1;
    }

    void pf_issue(std::uint32_t set, std::uint32_t slot) {
        Chain& c = pf_[set * 2 + slot];
        if (c.n == 0) return;
        c.tail = c.desc + (c.n - 1);
        c.n = 0;
        c.pending = true;
#if defined(FRT_TRACE)
        c.trT0 = now();
        c.trDone = 0;
        c.trOpen = true;
#endif
#if defined(FRT_PF_SAME_PORT)
        dma_.start(c.desc, c.tail, npu::DmaSource::DDR);
#else
        // prefetch chains run on the SHAVE's second DMA port: a synchronous load started on the same port and
        // direction while a chain is still in flight hangs the chain (seen on PTL)
        dma_.start(c.desc, c.tail, npu::DmaSource::DDR, (int)pfPort_);
#endif
    }

    void pf_wait(std::uint32_t set, std::uint32_t slot) {
        Chain& c = pf_[set * 2 + slot];
        if (!c.pending) return;
        dma_.wait(c.tail);
        c.pending = false;
#if defined(FRT_TRACE)
        if (c.trOpen) trace_ev(kEvDma, trSite_, c.trT0, c.trDone ? c.trDone : now());
        c.trOpen = false;
#endif
    }

    // the M rows' spatial grid and the accumulator stencil of an fp16 workload: npu_dsl/passes/spatial.py and
    // passes/stencil.py (measured on PTL; same output values and row order)
    void mm_tune(std::uint32_t M, std::uint32_t N) { mm_tune_on(mm_, M, N); }
    void mm_tune_on(npu::DpuMatMul* mm, std::uint32_t M, std::uint32_t N) {
#if defined(FRT_DPU_TUNE) && !FRT_DPU_TUNE
        (void)mm;
        (void)M;
        (void)N;
#else
        if (M < 16) return;
        std::uint32_t x = (M % 4 == 0) ? 4u : (M % 2 == 0 ? 2u : 1u);
        if (M % 7 == 0 && M > 14) x = 7;
        const std::uint32_t initX = x;
        if (M >= 72 && M % 12 == 0) x = M / 12;
        else if (M % 8 == 0) x = M / 8;
        if (x != initX) mm->setSpatial(M, x);
        const std::uint32_t w = x, h = M / x;
        // CUBOID 4x16, 8x16, 16x16: (grid w, h, k, nthw_ntk, odu nthw); a tie goes to the later one
        static constexpr std::uint32_t cub[3][5] = {{8, 8, 128, 1, 1}, {16, 8, 64, 0, 2}, {16, 16, 32, 3, 3}};
        std::uint32_t best = 2, bestCost = 0xFFFFFFFFu;
        for (int i = 2; i >= 0; --i) {
            const std::uint32_t c = (N + cub[i][2] - 1) / cub[i][2] * h * w +
                                    (h + cub[i][1] - 1) / cub[i][1] * ((w + cub[i][0] - 1) / cub[i][0]) * N;
            if (c < bestCost) { bestCost = c; best = (std::uint32_t)i; }
        }
        if (best != 2 && ((kDpuStencil >> best) & 1u)) mm->setStencil(cub[best][3], cub[best][4]);
#endif
    }

    // out[M, N] = a[M, K] . bt[N, K]^T (fp16); the descriptors are rebuilt only when the shape changes
    void matmul(half* out, const half* a, const half* bt, std::uint32_t M, std::uint32_t N, std::uint32_t K, float scale = 1.0f) {
        dpu_wait();
        guard(out, M * N * 2);
        const std::uint64_t key = ((std::uint64_t)M << 40) | ((std::uint64_t)N << 20) | K;
        if (key != mmKey_ || scale != mmScale_) {
            mm_->init(M, N, K);
            mm_tune(M, N);
            if (scale != 1.0f) mm_->setGlobalScale((half)scale);
            mmKey_ = key;
            mmScale_ = scale;
        }
#if defined(FRT_TRACE)
        const std::uint32_t trS = now();
        mm_->start(out, const_cast<half*>(a), const_cast<half*>(bt));
        const std::uint32_t trW = now();
        mm_->wait();
        trace_dpu(trS, trW, dpuDesc_);
#else
        mm_->run(out, const_cast<half*>(a), const_cast<half*>(bt));
#endif
    }

    // out[N, M] = (a[M, K] . bt[N, K]^T)^T: the ODU writes channel-major (permutation XYZ, as VPUX's V transpose). The
    // M rows' grid must be 8 wide for that (dpu_rate_probe DUMP2 on PTL: X = 8 / 16 / 24 / 32 / 40 right, 3 / 4 / 6 / 12 wrong)
    void matmul_t(half* out, const half* a, const half* bt, std::uint32_t M, std::uint32_t N, std::uint32_t K) {
        dpu_wait();
        guard(out, M * N * 2);
        const std::uint64_t key = (1ull << 63) | ((std::uint64_t)M << 40) | ((std::uint64_t)N << 20) | K;
        if (key != mmKey_ || mmScale_ != 1.0f) {
            mm_->init(M, N, K);
            mm_->setSpatial(M, 8);
            dpuDesc_->invariant.odu_cfg.odu_cfg_bf.permutation = 5;
            mmKey_ = key;
            mmScale_ = 1.0f;
        }
#if defined(FRT_TRACE)
        const std::uint32_t trS = now();
        mm_->start(out, const_cast<half*>(a), const_cast<half*>(bt));
        const std::uint32_t trW = now();
        mm_->wait();
        trace_dpu(trS, trW, dpuDesc_);
#else
        mm_->run(out, const_cast<half*>(a), const_cast<half*>(bt));
#endif
    }

    // matmul left running: at most one workload in flight per SHAVE. It is waited for before a CMX access that
    // conflicts with it (dpu_r for a read of its output; guard, i.e. any write, for its output or inputs), before
    // another DPU call, a barrier, a tile rendezvous and the status write.
    void matmul_async(half* out, const half* a, const half* bt, std::uint32_t M, std::uint32_t N, std::uint32_t K,
                      float scale = 1.0f) {
#if !FRT_DPU_ASYNC
        matmul(out, a, bt, M, N, K, scale);
#else
        dpu_wait();
        guard(out, M * N * 2);
        const std::uint64_t key = ((std::uint64_t)M << 40) | ((std::uint64_t)N << 20) | K;
        if (key != mmKey_ || scale != mmScale_) {
            mm_->init(M, N, K);
            mm_tune(M, N);
            if (scale != 1.0f) mm_->setGlobalScale((half)scale);
            mmKey_ = key;
            mmScale_ = scale;
        }
#if defined(FRT_TRACE)
        dpuTrS_ = now();
        dpuTrSite_ = trSite_;
#endif
        mm_->start(out, const_cast<half*>(a), const_cast<half*>(bt));
#if defined(FRT_TRACE)
        dpuTrW_ = now();
#endif
        dpuPending_ = true;
        dpuO_ = (std::uint32_t)out;
        dpuOe_ = dpuO_ + M * N * 2;
        dpuA_ = (std::uint32_t)a;
        dpuAe_ = dpuA_ + M * K * 2;
        dpuB_ = (std::uint32_t)bt;
        dpuBe_ = dpuB_ + N * K * 2;
#endif
    }

    void dpu_wait() {
#if FRT_DPU_ASYNC
        if (!dpuPending_) return;
        mm_->wait();
        dpuPending_ = false;
#if defined(FRT_TRACE)
        const std::uint32_t site = trSite_;
        trSite_ = dpuTrSite_;
        trace_dpu(dpuTrS_, dpuTrW_, dpuDesc_);
        trSite_ = site;
#endif
#endif
    }

    // before reading CMX [p, p + bytes): the workload in flight may still be writing it
    void dpu_r(const void* p, std::uint32_t bytes) {
#if FRT_DPU_ASYNC
        const std::uint32_t a = (std::uint32_t)p;
        if (dpuPending_ && a < dpuOe_ && dpuO_ < a + bytes) dpu_wait();
#else
        (void)p;
        (void)bytes;
#endif
    }

    // before handing CMX [p, p + bytes) to the tile's other SHAVE (sync_tile): the workload in flight may use it
    void dpu_t(const void* p, std::uint32_t bytes) {
#if FRT_DPU_ASYNC
        const std::uint32_t a = (std::uint32_t)p;
        if (dpuPending_ && ((a < dpuOe_ && dpuO_ < a + bytes) || (a < dpuAe_ && dpuA_ < a + bytes) ||
                            (a < dpuBe_ && dpuB_ < a + bytes)))
            dpu_wait();
#else
        (void)p;
        (void)bytes;
#endif
    }

    // out[M, N] = a[M, K] . (q * s)^T for int4 q [N, K] (signed, low nibble first, rows K/2 bytes apart) and fp16 group
    // scales s [N, K/G]: ONE DPU workload on int4 weights, no SHAVE dequantisation. Output channel c = g * N + n is row
    // n's group g (weights from byte n * K/2 + g * G/2 of q); the activations a [M, K] are read as [M * K/G, G] (row
    // (m, g) = a[m, gG:(g+1)G]). The DPU writes part[(m, g), c] = kI4PartScale * a[m, g-th group] . q[n, g-th group]
    // (the weight table is fixed per shape); meanwhile the SHAVE transposes s to [K/G, N]; then it forms
    // out[m, n] = sum_g s[n, g] * part[(m, g), g * N + n] / kI4PartScale (the diagonal blocks; the DPU computes K/G
    // times the needed products, which at decode's M = 1 is nearly free: the workload is bound by reading weights).
    // defer: the SHAVE reduction of this workload is left pending and done by the NEXT matmul_i4g while its DPU
    // workload runs (the emitter sets it only when the next statement is one; nothing reads `out` in between).
    // sStride != 0: s is group-major, [K/G, N] with rows sStride elements apart (a column block of a [K/G, *] tile),
    // read in place by the reduction (no transpose; under defer it must stay valid until the next call).
    // wStride != 0: q is group-major, [K/G, N*G] with rows wStride elements apart (VPUX's [groups, OC, IC per group]);
    // only the weight table's offsets change.
    void matmul_i4g(half* out, const half* a, const std::uint8_t* q, const half* s, half* part, std::uint32_t M,
                    std::uint32_t N, std::uint32_t K, std::uint32_t G, bool defer = false, std::uint32_t sStride = 0,
                    std::uint32_t wStride = 0) {
        matmul_qg(out, a, q, s, part, M, N, K, G, 4, defer, sStride, wStride);
    }


    // out[N, M] = (a . (q * s)^T)^T: matmul_i4g / i8g with the ODU's channel-major write (as matmul_t); one group per
    // row only, which takes the direct path (FRT_DPU_Q_DIRECT)
    void matmul_i4g_t(half* out, const half* a, const std::uint8_t* q, const half* s, half* part, std::uint32_t M,
                      std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride, std::uint32_t wStride) {
        matmul_qg(out, a, q, s, part, M, N, K, G, 4, false, sStride, wStride, true);
    }
    void matmul_i8g_t(half* out, const half* a, const std::int8_t* q, const half* s, half* part, std::uint32_t M,
                      std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride, std::uint32_t wStride) {
        matmul_qg(out, a, reinterpret_cast<const std::uint8_t*>(q), s, part, M, N, K, G, 8, false, sStride, wStride, true);
    }

    // matmul_i4g for int8 q [N, K] (signed, rows K bytes apart): the DPU's int8 weight mode (initMixed's default),
    // PPE multiplier kI8PartScale (|q| <= 128 instead of 8)
    void matmul_i8g(half* out, const half* a, const std::int8_t* q, const half* s, half* part, std::uint32_t M,
                    std::uint32_t N, std::uint32_t K, std::uint32_t G, bool defer = false, std::uint32_t sStride = 0,
                    std::uint32_t wStride = 0) {
        matmul_qg(out, a, reinterpret_cast<const std::uint8_t*>(q), s, part, M, N, K, G, 8, defer, sStride, wStride);
    }

    // out[m, n] = sum_g sT[g * sStride + n] * part[(m, g), g * N + n] * inv (sT: group-major scales; inv undoes the
    // PPE multiplier)
    void q_reduce(half* out, const half* part, const half* sT, std::uint32_t sStride, std::uint32_t M, std::uint32_t N,
                  std::uint32_t ng, float inv) {
        const std::uint32_t C = N * ng;
        for (std::uint32_t m = 0; m < M; ++m) {
            std::uint32_t n0 = 0;
            // four 16-column blocks at a time: independent accumulators instead of one serial FMA chain over the groups
            // (each output's sum is in the same order, so the result is bit-identical)
            for (; n0 + 64 <= N; n0 += 64) {
                float16 a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                for (std::uint32_t g = 0; g < ng; ++g) {
                    const half* p = part + (m * ng + g) * C + g * N + n0;
                    const half* w = sT + g * sStride + n0;
                    a0 += __builtin_convertvector(*reinterpret_cast<const half16*>(p), float16) *
                          __builtin_convertvector(*reinterpret_cast<const half16*>(w), float16);
                    a1 += __builtin_convertvector(*reinterpret_cast<const half16*>(p + 16), float16) *
                          __builtin_convertvector(*reinterpret_cast<const half16*>(w + 16), float16);
                    a2 += __builtin_convertvector(*reinterpret_cast<const half16*>(p + 32), float16) *
                          __builtin_convertvector(*reinterpret_cast<const half16*>(w + 32), float16);
                    a3 += __builtin_convertvector(*reinterpret_cast<const half16*>(p + 48), float16) *
                          __builtin_convertvector(*reinterpret_cast<const half16*>(w + 48), float16);
                }
                half* o = out + m * N + n0;
                *reinterpret_cast<half16*>(o) = __builtin_convertvector(a0 * inv, half16);
                *reinterpret_cast<half16*>(o + 16) = __builtin_convertvector(a1 * inv, half16);
                *reinterpret_cast<half16*>(o + 32) = __builtin_convertvector(a2 * inv, half16);
                *reinterpret_cast<half16*>(o + 48) = __builtin_convertvector(a3 * inv, half16);
            }
            for (; n0 < N; n0 += 16) {
                float16 acc = 0.f;
                for (std::uint32_t g = 0; g < ng; ++g) {
                    const half16 p = *reinterpret_cast<const half16*>(part + (m * ng + g) * C + g * N + n0);
                    const half16 w = *reinterpret_cast<const half16*>(sT + g * sStride + n0);
                    acc += __builtin_convertvector(p, float16) * __builtin_convertvector(w, float16);
                }
                *reinterpret_cast<half16*>(out + m * N + n0) = __builtin_convertvector(acc * inv, half16);
            }
        }
    }

#if defined(FRT_HW_SYNC)
    __attribute__((noinline))
#endif
    void barrier() {
        store_wait();
        dpu_wait();
#if FRT_TILE_BUFFERS
        npu::flushCmxWrites();   // a tile buffer's bytes may be another SHAVE's next (plan_cmx reuses them across a barrier)
#endif
#if defined(FRT_HW_SYNC)
        hw_sync();
#else
        sync_.waitGlobal();
#endif
    }

#if defined(FRT_HW_SYNC)
    // a global rendezvous on hardware barriers kHwSyncBarrier + (k % 3): every SHAVE produces, waits for the last
    // producer, consumes; worker 0 re-arms the id once every SHAVE consumed it (no SHAVE reaches the id again before
    // worker 0 has produced the next one). The ids are armed once per launch behind a software rendezvous.
    __attribute__((noinline)) void hw_sync() {
        const std::uint8_t n = (std::uint8_t)(tiles_ * spt_);
        if (!hwSyncArmed_) {
            if (worker() == 0)
                for (std::uint32_t i = 0; i < 3; ++i) hwb_push(kHwSyncBarrier + i, n, n, true);
            sync_.waitGlobal();
            hwSyncArmed_ = true;
        }
        const std::uint32_t id = kHwSyncBarrier + hwSyncK_;
        hwSyncK_ = hwSyncK_ == 2 ? 0 : hwSyncK_ + 1;
        hwb_produce(id);
        hwb_wait_produced(id);
        hwb_consume(id);
        if (worker() == 0) {
            std::uint8_t p = 0, c = 1;
            do { npu::hw::nce::barriers::hgl::getCounters(id, &p, &c); } while (c != 0);
            hwb_push(id, n, n, false);
        }
    }
#endif

    // rendezvous of the SHAVEs of this tile (a tile-shared buffer handed over; derive_tile_syncs). The pending store
    // may read CMX the other SHAVE writes next; a DPU workload in flight on the handed-over bytes is waited for by the
    // dpu_t calls the emitter puts before this.
    void sync_tile() {
        store_wait();
        npu::flushCmxWrites();
        sync_.waitLocal();
    }

    // rendezvous of every SHAVE of every tile with no DPU wait: a multicast fill handed over between tiles. The pending
    // store must finish first: the barrier's own broadcast DMA on the same port would hang behind it
    void sync_all() {
        store_wait();
        npu::flushCmxWrites();
        sync_.waitGlobal();
    }

    // ---- hardware barriers (hand PoC of the multi-tile flow; reference: kernels_poc/matmul_bias_e7_17/handpoc) ----
    void hwb_push(std::uint32_t id, std::uint8_t p, std::uint8_t c, bool wait) {
        npu::hw::nce::barriers::hgl::pushCounters(id, p, c);
        if (!wait) return;
        std::uint8_t lp = 0, lc = 0;
        do { npu::hw::nce::barriers::hgl::getCounters(id, &lp, &lc); } while (lp != p || lc != c);
    }
    // a gate on a barrier whose config is not live is open: wait for it (c != 0) before gating on it
    void hwb_wait_live(std::uint32_t id) {
        std::uint8_t p = 0, c = 0;
        do { npu::hw::nce::barriers::hgl::getCounters(id, &p, &c); } while (c == 0);
    }
    void hwb_wait_produced(std::uint32_t id) {
        std::uint8_t p = 1, c = 0;
        do { npu::hw::nce::barriers::hgl::getCounters(id, &p, &c); } while (p != 0);
    }
    void hwb_consume(std::uint32_t id) { npu::hw::nce::barriers::hgl::consumeBarrier(id); }
    void hwb_produce(std::uint32_t id) { npu::hw::nce::barriers::hgl::produceBarrier(id); }

    // rows x width bytes from DDR (rows srcStride apart) to `dst` (packed) at the same offset in every tile's CMX, the
    // rows in two halves on DMA ports 0 and 1, each producing `ready` (a single row: the second half is produced here).
    // Uses the prefetch descriptors: no prefetch chain may be in flight
    void fill_mc(void* dst, std::uint64_t src, std::uint32_t width, std::uint32_t rows, std::uint32_t srcStride, std::uint32_t ready) {
        fill_mc_wait();
        npu::DmaDescriptor* d0 = pfDesc_;
        npu::DmaDescriptor* d1 = pfDesc_ + 1;
        const std::uint32_t mask = (1u << tiles_) - 1;
        auto* const dst8 = static_cast<std::uint8_t*>(dst);
        // two halves: by rows, or one row by bytes
        const bool byRows = rows > 1;
        const std::uint32_t r0 = byRows ? (rows + 1) / 2 : 1, r1 = byRows ? rows - r0 : 0;
        const std::uint32_t w0 = byRows ? width : width / 2 / 64 * 64, w1 = byRows ? width : width - w0;
        const bool two = byRows ? r1 > 0 : w0 > 0;
        npu::DmaDrv::initDesc(d0);
        npu::DmaDrv::initDesc(d1);
        if (two) {
            npu::configDmaCopy2D(d0, (std::uint32_t)npu::convertToCmxMulticastAddr(dst8, mask), src, w0, r0, w0, srcStride);
            const std::uint32_t off = byRows ? r0 * width : w0;
            const std::uint64_t soff = byRows ? (std::uint64_t)r0 * srcStride : w0;
            npu::configDmaCopy2D(d1, (std::uint32_t)npu::convertToCmxMulticastAddr(dst8 + off, mask), src + soff, w1, byRows ? r1 : 1, w1,
                                 srcStride);
            npu::DmaDrv::setProducerBarrier(d0, ready);
            npu::DmaDrv::setProducerBarrier(d1, ready);
            dma_.start(d0, 0);
            dma_.start(d1, 1);
        } else {
            npu::configDmaCopy2D(d0, (std::uint32_t)npu::convertToCmxMulticastAddr(dst8, mask), src, width, rows, width, srcStride);
            npu::DmaDrv::setProducerBarrier(d0, ready);
            dma_.start(d0, 0);
            hwb_produce(ready);
        }
        mcHalves_ = two ? 2 : 1;
    }
    void fill_mc_wait() {
        if (mcHalves_ == 0) return;
        dma_.wait(pfDesc_);
        if (mcHalves_ == 2) dma_.wait(pfDesc_ + 1);
        mcHalves_ = 0;
    }

    // matmul gated on hardware barrier `ready` that produces `free` when it completes; waited for here
    void matmul_hwb(half* out, const half* a, const half* bt, std::uint32_t M, std::uint32_t N, std::uint32_t K, std::uint32_t ready,
                    std::uint32_t free) {
        dpu_wait();
        guard(out, M * N * 2);
        const std::uint64_t key = ((std::uint64_t)M << 40) | ((std::uint64_t)N << 20) | K;
        if (key != mmKey_ || mmScale_ != 1.0f) {
            mm_->init(M, N, K);
            mm_tune(M, N);
            mmKey_ = key;
            mmScale_ = 1.0f;
        }
        mm_->setConsumerBarrier(ready);
        mm_->setProducerBarrier(free);
        mm_->start(out, const_cast<half*>(a), const_cast<half*>(bt));
        mm_->wait();
        mm_->setConsumerBarriers(0, 0);
        mm_->setProducerBarriers(0, 0);
    }
    // matmul_i4g / i8g gated on hardware barrier `ready`, producing `free`: the SHAVE waits for the workload anyway, so
    // the gate and the release are done here around a plain workload (descriptor barriers on the int4 path hung, PTL)
    void matmul_i4g_hwb(half* out, const half* a, const std::uint8_t* q, const half* s, half* part, std::uint32_t M,
                        std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride, std::uint32_t wStride,
                        std::uint32_t ready, std::uint32_t free) {
        hwb_wait_produced(ready);
        matmul_qg(out, a, q, s, part, M, N, K, G, 4, false, sStride, wStride);
        hwb_consume(ready);
        hwb_produce(free);
    }
    void matmul_i8g_hwb(half* out, const half* a, const std::int8_t* q, const half* s, half* part, std::uint32_t M,
                        std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t sStride, std::uint32_t wStride,
                        std::uint32_t ready, std::uint32_t free) {
        hwb_wait_produced(ready);
        matmul_qg(out, a, reinterpret_cast<const std::uint8_t*>(q), s, part, M, N, K, G, 8, false, sStride, wStride);
        hwb_consume(ready);
        hwb_produce(free);
    }

    // The rounds of a tile-distributed loop over nW weight tiles whose streamed operand is tile-invariant
    // (npu_dsl/fused passes.plan_mc_rounds). The tiles run in lock-step rounds; per round every live tile runs deq(it),
    // then (one global rendezvous: every dequant done) the nb blocks are events, event g -> slot g & 1 = SHAVE g & 1.
    // Tile 0's SHAVE s reads block c (blk(c): rows x width bytes, srcStride apart) ONCE into `slot` and multicasts
    // it to every tile, as two halves on DMA ports 0 and 1. READY (the fill -> the workloads of every tile, which are
    // gated on it) and FREE (the workloads -> the next fill of the slot) are hardware barriers per slot and use parity.
    // The slot may share bytes with deq's staging: a round's first fills wait for the rendezvous.
    template <class Deq, class Blk, class Pre, class Mm, class Epi>
    void mc_rounds(std::uint32_t nW, std::uint32_t nb, void* slot, Deq deq, Blk blk, Pre pre, Mm mm, Epi epi) {
        const std::uint32_t first = dist_first_tile(nW);
        const std::uint32_t rounds = (nW + tiles_ - 1) / tiles_;
        const std::uint32_t s = shave_;
        const std::uint32_t nEv = rounds * nb;
        const std::uint32_t slotEv = nEv > s ? (nEv - s + 1) / 2 : 0;
        const bool issuer = tile_ == 0;
        auto readyId = [&](std::uint32_t u) { return kMcBarrier + 2u * s + (u & 1u); };
        auto freeId = [&](std::uint32_t u) { return kMcBarrier + 4u + 2u * s + (u & 1u); };
        auto pushUse = [&](std::uint32_t u, bool wait) {
            hwb_push(readyId(u), 2, (std::uint8_t)tiles_, wait);
            hwb_push(freeId(u), (std::uint8_t)tiles_, 1, wait);
        };
        auto fill = [&](std::uint32_t c, std::uint32_t u) {
            const McBlock b = blk(c);
            fill_mc(slot, b.src, b.width, b.rows, b.srcStride, readyId(u));
        };
        if (issuer && slotEv > 0) pushUse(0, true);
        std::uint32_t u = 0;
        bool deferred = slotEv > 0;
        for (std::uint32_t r = 0; r < rounds; ++r) {
            const std::uint32_t it = first + r * tiles_;
            const bool live = it < nW;
            if (live) {
                sync_tile();   // both SHAVEs are past the previous round's workloads on the weight tile
                deq(it);
            }
            sync_all();
            const std::uint32_t c0 = ((r * nb) & 1u) == s ? 0u : 1u;
            if (issuer && deferred && c0 < nb) {
                fill(c0, u);
                deferred = false;
            }
            for (std::uint32_t c = c0; c < nb; c += 2, ++u) {
                hwb_wait_live(readyId(u));
                if (issuer) fill_mc_wait();   // no synchronous load on a port with a fill in flight
                if (live) {
                    pre(it, c);
                    mm(it, c, readyId(u), freeId(u));
                } else {
                    hwb_wait_produced(readyId(u));
                    hwb_consume(readyId(u));
                    hwb_produce(freeId(u));
                }
                if (issuer) {
                    if (u + 1 < slotEv) pushUse(u + 1, false);
                    hwb_wait_produced(freeId(u));   // every tile's DPU is done with the slot
                    hwb_consume(freeId(u));
                    if (u + 1 < slotEv) {
                        if (c + 2 < nb) fill(c + 2, u + 1);
                        else deferred = true;
                    }
                }
                if (live) epi(it, c);
            }
            fill_mc_wait();
        }
    }

    // status slot words: code, detail, tile CMX free at kernel entry, tile CMX this kernel allocated
    void status(std::uint32_t code, std::uint32_t detail) {
        store_wait();
        dpu_wait();
        if (statusAddr_ == 0) return;
        const std::uint64_t a = statusAddr_ + 32ull * worker();
        npu::DdrAccessor<std::uint32_t> s0(a), s1(a + 4), s2(a + 8), s3(a + 12);
        s0 = code;
        s1 = detail;
        s2 = cmxFree0_;
        s3 = cmxFree0_ - cmx_.getAvailableSize();
        npu::sync::memoryFence();
    }

protected:   // TileLang: the npu_rt_device_frt.hpp shim derives from Ctx
    void matmul_qg(half* out, const half* a, const std::uint8_t* q, const half* s, half* part, std::uint32_t M,
                   std::uint32_t N, std::uint32_t K, std::uint32_t G, std::uint32_t bits, bool defer, std::uint32_t sStride,
                   std::uint32_t wStride, bool outT = false) {
        const std::uint32_t ng = K / G, C = N * ng;
        dpu_wait();
        guard(out, M * N * 2);
        guard(part, M * ng * C * 2);
        FRT_SUB_BEGIN(0);
        const std::uint64_t key = ((std::uint64_t)outT << 63) | ((std::uint64_t)M << 40) | ((std::uint64_t)N << 20) | ((std::uint64_t)K << 8) | ((G >> 4) << 1) | (bits == 8);
        const float partScale = bits == 8 ? kI8PartScale : kI4PartScale;
        if (key != i4Key_ || wStride != i4WStride_) {
            union { float f; std::uint32_t u; } sc;
            sc.f = partScale;
            const std::uint32_t rowB = K * bits / 8, grpB = G * bits / 8;
            // channel (g, n): row-major q at n * rowB + g * grpB; group-major at g * (wStride row) + n * grpB
            const std::uint32_t gStep = wStride ? wStride * bits / 8 : grpB, nStep = wStride ? grpB : rowB;
            // rows {weights offset, 0, PPE multiplier, 0}, four rows (64 B) per store: N % 16 == 0
            using u32x16 = std::uint32_t __attribute__((ext_vector_type(16)));
            const u32x16 first = {0, 0, sc.u, 0, nStep, 0, sc.u, 0, 2 * nStep, 0, sc.u, 0, 3 * nStep, 0, sc.u, 0};
            const std::uint32_t s4 = 4 * nStep;
            const u32x16 step = {s4, 0, 0, 0, s4, 0, 0, 0, s4, 0, 0, 0, s4, 0, 0, 0};
            u32x16* w = reinterpret_cast<u32x16*>(wtI4_);
            for (std::uint32_t g = 0; g < ng; ++g) {
                const std::uint32_t o = g * gStep;
                u32x16 v = first + (u32x16){o, 0, 0, 0, o, 0, 0, 0, o, 0, 0, 0, o, 0, 0, 0};
                for (std::uint32_t n = 0; n < N; n += 4, ++w, v += step) *w = v;
            }
            mmI4_->initMixedDescOnly(M * ng, C, G);
            if (bits == 4)   // initMixedDescOnly sets the int8 weight mode
                dpuDescI4_->invariant.tensor_mode.tensor_mode_bf.wmode = static_cast<std::uint32_t>(npu::npu40xx::VpuInputTensorDType::I4);
#if defined(FRT_DPU_Q_TUNE) && FRT_DPU_Q_TUNE
            if (ng == 1 && !outT) mm_tune_on(mmI4_, M, C);   // one group per row: a plain [M, N] workload
#endif
            if (outT) {   // the ODU's XYZ permutation (channel-major out), on an 8-wide grid as matmul_t
                mmI4_->setSpatial(M, 8);
                dpuDescI4_->invariant.odu_cfg.odu_cfg_bf.permutation = 5;
            }
            i4Key_ = key;
            i4WStride_ = wStride;
        }
        FRT_SUB_END(0);
        FRT_SUB_BEGIN(1);
#if defined(FRT_TRACE)
        const std::uint32_t trS = now();
#endif
        // one group per row (per-channel scales): the scale is the channel's PPE multiplier, the DPU writes `out`
        // itself and there is no SHAVE reduction (the table's offsets stay cached; only the multipliers change)
        const bool direct = kDpuQDirect && ng == 1;
        if (outT && !direct) { ok_ = false; FRT_SUB_END(1); return; }   // a transposed write needs the direct path
        if (direct) {   // s[n]: [1, N] group-major or [N, 1] row-major alike
            for (std::uint32_t n = 0; n < N; ++n) {
                union { float f; std::uint32_t u; } v;
                v.f = (float)s[n];
                wtI4_[4 * n + 2] = v.u;
            }
        }
        mmI4_->startMixed(direct ? out : part, const_cast<half*>(a), reinterpret_cast<std::int8_t*>(const_cast<std::uint8_t*>(q)));
        if (i4Pend_.out != nullptr) {   // the previous workload's reduction, overlapping this one's DPU time
            q_reduce(i4Pend_.out, i4Pend_.part, i4Pend_.sT, i4Pend_.sStride, i4Pend_.M, i4Pend_.N, i4Pend_.ng, i4Pend_.inv);
            i4Pend_.out = nullptr;
        }
        const half* sT = s;
        if (sStride == 0 && !direct) {   // row-major scales: transposed into sT_ while the DPU runs
            transpose_h(sT_, s, N, ng);
            sT = sT_;
            sStride = N;
        }
#if defined(FRT_TRACE)
        const std::uint32_t trW = now();
        mmI4_->wait();
        trace_dpu(trS, trW, dpuDescI4_);
#else
        mmI4_->wait();
#endif
        FRT_SUB_END(1);
        FRT_SUB_BEGIN(2);
        if (direct) {
        } else if (defer)
            i4Pend_ = {out, part, sT, sStride, M, N, ng, 1.0f / partScale};
        else
            q_reduce(out, part, sT, sStride, M, N, ng, 1.0f / partScale);
        FRT_SUB_END(2);
    }

    const npu::Environment& env_;
    npu::LinearAllocator& cmx_;
    npu::DmaDrv& dma_;
    npu::Sync& sync_;
    std::uint64_t statusAddr_;
    std::uint32_t shave_ = 0, spt_ = 1, tile_ = 0, tiles_ = 1;
    std::uint32_t cmxFree0_ = 0;
    std::uint32_t distRot_ = 0, tileRot_ = 0, shaveRot_ = 0;
    bool ok_ = true;
    npu::DmaDescriptor* desc_ = nullptr;
    npu::DmaDescriptor* stDesc_ = nullptr;
    npu::DmaDescriptor* pfDesc_ = nullptr;
    npu::DmaDescriptor* gaDesc_ = nullptr;
    std::uint32_t gaN_ = 0;
    struct Chain {
        npu::DmaDescriptor* desc = nullptr;
        npu::DmaDescriptor* tail = nullptr;
        std::uint32_t n = 0;
        bool pending = false;
#if defined(FRT_TRACE)
        std::uint32_t trT0 = 0, trDone = 0;
        bool trOpen = false;
#endif
    };
    Chain pf_[kPfSets * 2];
    std::uint32_t pfPort_ = 1;
    std::uint32_t mcHalves_ = 0;
#if defined(FRT_HW_SYNC)
    bool hwSyncArmed_ = false;
    std::uint32_t hwSyncK_ = 0;
#endif
    bool stPending_ = false;
    bool ldPending_ = false;
    std::uint32_t stSrc_ = 0, stSrcEnd_ = 0;
    std::uint64_t stDst_ = 0, stDstEnd_ = 0;
    npu::DpuMatMul::DescriptorsBuffer* dpuDesc_ = nullptr;
    npu::DpuMatMul::DescriptorsBuffer* dpuDescI4_ = nullptr;
    std::uint32_t* wt_ = nullptr;
    std::uint32_t* wtI4_ = nullptr;
    half* sT_ = nullptr;
    alignas(npu::DpuMatMul) unsigned char mmStore_[sizeof(npu::DpuMatMul)];
    alignas(npu::DpuMatMul) unsigned char mmI4Store_[sizeof(npu::DpuMatMul)];
    npu::DpuMatMul* mm_ = nullptr;
    bool dpuPending_ = false;
    std::uint32_t dpuO_ = 0, dpuOe_ = 0, dpuA_ = 0, dpuAe_ = 0, dpuB_ = 0, dpuBe_ = 0;
#if defined(FRT_TRACE)
    std::uint32_t dpuTrS_ = 0, dpuTrW_ = 0, dpuTrSite_ = 0;
#endif
    npu::DpuMatMul* mmI4_ = nullptr;
    std::uint64_t mmKey_ = ~0ull, i4Key_ = ~0ull;
    std::uint32_t i4WStride_ = 0;
    struct I4Pending { half* out; const half* part; const half* sT; std::uint32_t sStride, M, N, ng; float inv; } i4Pend_ = {nullptr, nullptr, nullptr, 0, 0, 0, 0, 1.0f};
    float mmScale_ = 1.0f;
#if defined(FRT_PROFILE)
    std::uint32_t* prof_ = nullptr;
#endif
#if defined(FRT_TRACE)
    std::uint32_t* tr_ = nullptr;
    npu::DmaDescriptor* trDesc_ = nullptr;
    std::uint32_t trN_ = 0, trStored_ = 0, trDropped_ = 0;
    std::uint32_t trStSite_ = 0, trStT0_ = 0;
    bool trStOpen_ = false, trBusy_ = false;
#endif
};

}  // namespace frt

#endif

// Call-site profiling hooks the emitter wraps around every DMA / DPU / SHAVE-op / barrier; empty unless the device
// build defines FRT_PROFILE (the host simulation never profiles).
#if defined(FRT_PROFILE) && !defined(FRT_HOST_SIM)
#if defined(FRT_TRACE)
#define FRT_PROF_BEGIN(id) const std::uint32_t frt_t##id = (ctx.trSite_ = (id), frt::Ctx::now())
#else
#define FRT_PROF_BEGIN(id) const std::uint32_t frt_t##id = frt::Ctx::now()
#endif
#define FRT_PROF_END(id) ctx.prof_add(id, frt_t##id)
#define FRT_PROF_FLUSH(n) ctx.prof_flush(n)
#else
#define FRT_PROF_BEGIN(id) (void)0
#define FRT_PROF_END(id) (void)0
#define FRT_PROF_FLUSH(n) (void)0
#endif

namespace frt {
// status codes written to word 0 of each SHAVE's 32-byte status slot
static constexpr std::uint32_t kStatusOk = 1;
static constexpr std::uint32_t kStatusBadParams = 2;
static constexpr std::uint32_t kStatusCmx = 3;
static constexpr std::uint32_t kStatusNotRun = 0xFFFFFFFFu;
}  // namespace frt
