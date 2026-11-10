/*!
 * \file tl/intel_npu/op/builtin.h
 * \brief Intel NPU-specific TileLang intrinsic Ops: the lowered forms of the
 * tile ops, printed by the codegen as calls on the per-SHAVE runtime context.
 */

#ifndef TVM_TL_INTEL_NPU_OP_BUILTIN_H_
#define TVM_TL_INTEL_NPU_OP_BUILTIN_H_

#include "op/builtin.h"

namespace tvm {
namespace tl {

// Pass config: derive grid barriers between DDR-communicating phases (default
// true). Off only for the negative controls that prove a barrier is live.
static constexpr const char *kIntelNpuDeriveGlobalSync =
    "tl.intel_npu.derive_global_sync";
// Pass config: implement sync_grid on hardware barriers (default true); false
// selects the toolkit's software rendezvous, for A/B measurement only.
static constexpr const char *kIntelNpuHwSyncGrid = "tl.intel_npu.hw_sync_grid";
// Pass configs: kernel-wide DPU workload shape, used by the VPUNN calibration
// sweep until PlanWorkloads sets them per op. spatial_x 0 = the runtime's own
// split; mpe -1 = the runtime's own stencil, else 0 CUBOID_4x16, 1 CUBOID_8x16,
// 2 CUBOID_16x16.
static constexpr const char *kIntelNpuDpuSpatialX =
    "tl.intel_npu.dpu_spatial_x";
static constexpr const char *kIntelNpuDpuMpe = "tl.intel_npu.dpu_mpe";
// Pass config: PlanIntelNpuWorkloads mode, "auto" (VPUNN when available),
// "off", "require".
static constexpr const char *kIntelNpuPlanWorkloads =
    "tl.intel_npu.plan_workloads";
// False: AssignBarriers gives every ring fresh ids (device A/B of the id reuse
// across phases).
static constexpr const char *kIntelNpuBarrierReuse =
    "tl.intel_npu.barrier_reuse";
// Pass config: scoped CMX (allocations sink to the block that uses them and are
// released at its end; default true); false keeps every allocation for the
// whole kernel (the A/B of the scoping).
static constexpr const char *kIntelNpuScopedCmx = "tl.intel_npu.scoped_cmx";
// Pass config: compiler-derived one-iteration DMA prefetch (default true);
// false is the paired A/B control.
static constexpr const char *kIntelNpuPrefetch = "tl.intel_npu.prefetch";
static constexpr const char *kIntelNpuBatchDma = "tl.intel_npu.batch_dma";
static constexpr const char *kIntelNpuDpuOverlap = "tl.intel_npu.dpu_overlap";
static constexpr const char *kIntelNpuMulticast = "tl.intel_npu.multicast";
static constexpr const char *kIntelNpuMulticastOverlap =
    "tl.intel_npu.multicast_overlap";
// AttrStmt key: free-text notes a pass leaves for the reader of the generated
// source (what was planned, or what was skipped and why); a StringImm value,
// printed as a comment. An AttrStmt rather than a PrimFunc attribute because
// SplitHostDevice keeps only a fixed attribute set on the device function.
static constexpr const char *kIntelNpuNotesAttr = "intel_npu.notes";

/*!
 * \brief 2-D DMA between DDR and CMX.
 *
 *   intel_npu_dma_copy(dst_ptr, src_ptr, width_bytes, rows,
 *                      dst_stride_bytes, src_stride_bytes, to_cmx[, async[,
 * multicast]])
 *
 * Pointers are `address_of(BufferLoad)`; `to_cmx` is 1 for DDR->CMX, 0 for
 * CMX->DDR. Synchronous unless `async` is 1, which only PlanIntelNpuPrefetch
 * sets: the load is left in flight and `intel_npu_dma_wait` joins it.
 * `multicast=1` is synchronous and issued by tile 0's matching SHAVE, with
 * explicit grid rendezvous around the call in the compiler-generated IR.
 */
TVM_DLL const Op &intel_npu_dma_copy();

/*!
 * \brief Join the load `intel_npu_dma_copy(..., async=1)` left in flight. The
 * runtime also joins at every synchronous DMA and at `qg_flush`, so a missing
 * wait costs time rather than correctness.
 */
TVM_DLL const Op &intel_npu_dma_wait();
TVM_DLL const Op &intel_npu_dma_batch();
TVM_DLL const Op &intel_npu_dma_batch_flush();
TVM_DLL const Op &intel_npu_axpy();

/*!
 * \brief One DPU fp16 matmul workload: out[M,N] = a[M,K] . bt[N,K]^T, every
 * operand contiguous in CMX. Overwrites `out`.
 *
 *   intel_npu_dpu_matmul(out_ptr, a_ptr, bt_ptr, M, N, K, spatial_x, mpe)
 *
 * `spatial_x` splits the M rows over an x-by-(M/x) grid (0 = runtime default);
 * `mpe` selects the accumulator stencil (-1 = runtime default, see
 * kIntelNpuDpuMpe).
 */
TVM_DLL const Op &intel_npu_dpu_matmul();
TVM_DLL const Op &intel_npu_dpu_wait();
TVM_DLL const Op &intel_npu_dequant_i4_gm();

/*!
 * \brief A DDR tensor at a run-time address: `intel_npu_ddr_ptr(addr_int64,
 * typed_zero)` is a handle a buffer is declared over (`T.ddr_view`); codegen
 * binds it as `npu::Ddr<T>(addr)` with T from the typed constant. The layer
 * table of a single-launch model points at each layer's weights and states this
 * way.
 */
TVM_DLL const Op &intel_npu_ddr_ptr();

/*!
 * \brief The streamed ping-pong ring (plan Phase 3b; README sections 1-3). A
 * T.Pipelined loop over a DUPLICATED fill becomes:
 *
 *   ring = intel_npu_ring_begin(slot_base_ptr, slot_bytes, slots, total_chunks,
 * ready_base, free_base) intel_npu_ring_uses(ring) -> int32         chunks of
 * this SHAVE's slot intel_npu_ring_slot(ring) -> ptr           this SHAVE's
 * slot intel_npu_ring_fill(ring, use, src_ptr, width_bytes, rows,
 * src_stride_bytes)   issuer: multicast fill of chunk `use`
 *   intel_npu_ring_dpu_submit(ring, use, out, a, bt, M, N, K, spatial_x, mpe)
 *   intel_npu_ring_dpu_wait(ring, use)
 *   intel_npu_ring_end(ring)
 *
 * Barrier ids are placeholders (-1) until AssignIntelNpuBarriers fills them.
 */
TVM_DLL const Op &intel_npu_ring_begin();
TVM_DLL const Op &intel_npu_ring_uses();
TVM_DLL const Op &intel_npu_ring_slot();
// (out, a, q, s, M, N, K, G): fp16 activations x packed int4 weights with fp16
// group scales, one mixed-mode DPU workload + SHAVE group reduction (the decode
// GEMV path).
TVM_DLL const Op &intel_npu_dpu_matmul_i4g();
TVM_DLL const Op &
intel_npu_dpu_matmul_i8g(); // the same with int8 weights (one byte per value)
TVM_DLL const Op &intel_npu_ring_slot_acquire(); // (ring, use): SHAVE reader
                                                 // waits READY and consumes it
TVM_DLL const Op &
intel_npu_ring_slot_release(); // (ring, use): SHAVE reader produces FREE
TVM_DLL const Op &intel_npu_ring_fill();
TVM_DLL const Op &intel_npu_ring_dpu_submit();
TVM_DLL const Op &intel_npu_ring_dpu_wait();
TVM_DLL const Op &intel_npu_ring_end();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_INTEL_NPU_OP_BUILTIN_H_
