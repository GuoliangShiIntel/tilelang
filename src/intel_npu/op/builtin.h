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
static constexpr const char *kIntelNpuDeriveGlobalSync = "tl.intel_npu.derive_global_sync";
// Pass config: implement sync_grid on hardware barriers (default true); false
// selects the toolkit's software rendezvous, for A/B measurement only.
static constexpr const char *kIntelNpuHwSyncGrid = "tl.intel_npu.hw_sync_grid";
// Pass configs: kernel-wide DPU workload shape, used by the VPUNN calibration sweep
// until PlanWorkloads sets them per op. spatial_x 0 = the runtime's own split;
// mpe -1 = the runtime's own stencil, else 0 CUBOID_4x16, 1 CUBOID_8x16, 2 CUBOID_16x16.
static constexpr const char *kIntelNpuDpuSpatialX = "tl.intel_npu.dpu_spatial_x";
static constexpr const char *kIntelNpuDpuMpe = "tl.intel_npu.dpu_mpe";
// Pass config: PlanIntelNpuWorkloads mode, "auto" (VPUNN when available), "off", "require".
static constexpr const char *kIntelNpuPlanWorkloads = "tl.intel_npu.plan_workloads";
// False: AssignBarriers gives every ring fresh ids (device A/B of the id reuse across phases).
static constexpr const char *kIntelNpuBarrierReuse = "tl.intel_npu.barrier_reuse";
// AttrStmt key: free-text notes a pass leaves for the reader of the generated source
// (what was planned, or what was skipped and why); a StringImm value, printed as a comment.
// An AttrStmt rather than a PrimFunc attribute because SplitHostDevice keeps only a fixed
// attribute set on the device function.
static constexpr const char *kIntelNpuNotesAttr = "intel_npu.notes";

/*!
 * \brief 2-D DMA between DDR and CMX.
 *
 *   intel_npu_dma_copy(dst_ptr, src_ptr, width_bytes, rows,
 *                      dst_stride_bytes, src_stride_bytes, to_cmx)
 *
 * Pointers are `address_of(BufferLoad)`; `to_cmx` is 1 for DDR->CMX, 0 for
 * CMX->DDR. Synchronous in Phase 1 (the runtime waits before returning).
 */
TVM_DLL const Op &intel_npu_dma_copy();

/*!
 * \brief One DPU fp16 matmul workload: out[M,N] = a[M,K] . bt[N,K]^T, every
 * operand contiguous in CMX. Overwrites `out`.
 *
 *   intel_npu_dpu_matmul(out_ptr, a_ptr, bt_ptr, M, N, K, spatial_x, mpe)
 *
 * `spatial_x` splits the M rows over an x-by-(M/x) grid (0 = runtime default);
 * `mpe` selects the accumulator stencil (-1 = runtime default, see kIntelNpuDpuMpe).
 */
TVM_DLL const Op &intel_npu_dpu_matmul();

/*!
 * \brief The streamed ping-pong ring (plan Phase 3b; README sections 1-3). A
 * T.Pipelined loop over a DUPLICATED fill becomes:
 *
 *   ring = intel_npu_ring_begin(slot_base_ptr, slot_bytes, slots, total_chunks, ready_base, free_base)
 *   intel_npu_ring_uses(ring) -> int32         chunks of this SHAVE's slot
 *   intel_npu_ring_slot(ring) -> ptr           this SHAVE's slot
 *   intel_npu_ring_fill(ring, use, src_ptr, width_bytes, rows, src_stride_bytes)   issuer: multicast fill of chunk `use`
 *   intel_npu_ring_dpu_submit(ring, use, out, a, bt, M, N, K, spatial_x, mpe)
 *   intel_npu_ring_dpu_wait(ring, use)
 *   intel_npu_ring_end(ring)
 *
 * Barrier ids are placeholders (-1) until AssignIntelNpuBarriers fills them.
 */
TVM_DLL const Op &intel_npu_ring_begin();
TVM_DLL const Op &intel_npu_ring_uses();
TVM_DLL const Op &intel_npu_ring_slot();
// (out, a, q, s, M, N, K, G): fp16 activations x packed int4 weights with fp16 group scales, one mixed-mode
// DPU workload + SHAVE group reduction (the decode GEMV path).
TVM_DLL const Op &intel_npu_dpu_matmul_i4g();
TVM_DLL const Op &intel_npu_ring_slot_acquire();   // (ring, use): SHAVE reader waits READY and consumes it
TVM_DLL const Op &intel_npu_ring_slot_release();   // (ring, use): SHAVE reader produces FREE
TVM_DLL const Op &intel_npu_ring_fill();
TVM_DLL const Op &intel_npu_ring_dpu_submit();
TVM_DLL const Op &intel_npu_ring_dpu_wait();
TVM_DLL const Op &intel_npu_ring_end();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_INTEL_NPU_OP_BUILTIN_H_
