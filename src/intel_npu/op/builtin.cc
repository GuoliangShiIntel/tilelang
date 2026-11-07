/*!
 * \file tl/intel_npu/op/builtin.cc
 * \brief Registration of Intel NPU-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include <tvm/ir/transform.h>

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {

using namespace tirx;

TVM_REGISTER_PASS_CONFIG_OPTION(kIntelNpuDeriveGlobalSync, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kIntelNpuHwSyncGrid, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kIntelNpuDpuSpatialX, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kIntelNpuDpuMpe, Integer);

TIR_DEFINE_TL_BUILTIN(intel_npu_dma_copy)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(intel_npu_dpu_matmul)
    .set_num_inputs(8)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
