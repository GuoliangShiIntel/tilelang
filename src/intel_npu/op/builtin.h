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
 *   intel_npu_dpu_matmul(out_ptr, a_ptr, bt_ptr, M, N, K)
 */
TVM_DLL const Op &intel_npu_dpu_matmul();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_INTEL_NPU_OP_BUILTIN_H_
