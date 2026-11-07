/*!
 * \file intrin_rule_intel_npu.cc
 * \brief Math intrinsic lowering for the Intel NPU SHAVE target.
 *
 * fp16 math maps to `tl_npu_h<name>` helpers defined by the runtime headers
 * (npu_rt_device.hpp: SAU builtins where the SHAVE has them, float math
 * otherwise; npu_rt_host.hpp: float math), fp32 to the C library's `<name>f`.
 * Vectors (after VectorizeLoop): fp16 lanes to the same `tl_npu_h<name>`
 * (overloaded on half8/half16/half32; VAU.EXP for exp), fp32 lanes to
 * `tl_npu_<name>f` lane loops. Registered as "intel_npu.FLowerIntrinsic",
 * picked up by LowerIntrin.
 */
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

using tirx::FLowerIntrinsic;

struct IntelNpuMath {
  std::string operator()(DataType t, std::string name) const {
    if (!t.is_float())
      return "";
    switch (t.bits()) {
    case 32:
      return t.lanes() == 1 ? name + 'f' : "tl_npu_" + name + 'f';
    case 16:
      return "tl_npu_h" + name;
    default:
      return "";
    }
  }
};

#define REGISTER_INTEL_NPU_INTRIN(OpName)                                      \
  TVM_REGISTER_OP("tirx." OpName)                                              \
      .set_attr<FLowerIntrinsic>("intel_npu.FLowerIntrinsic",                  \
                                 DispatchPureExtern<IntelNpuMath>)

REGISTER_INTEL_NPU_INTRIN("exp");
REGISTER_INTEL_NPU_INTRIN("exp2");
REGISTER_INTEL_NPU_INTRIN("log");
REGISTER_INTEL_NPU_INTRIN("log2");
REGISTER_INTEL_NPU_INTRIN("sqrt");
REGISTER_INTEL_NPU_INTRIN("rsqrt");
REGISTER_INTEL_NPU_INTRIN("tanh");
REGISTER_INTEL_NPU_INTRIN("fabs");
REGISTER_INTEL_NPU_INTRIN("floor");
REGISTER_INTEL_NPU_INTRIN("ceil");
REGISTER_INTEL_NPU_INTRIN("round");

#undef REGISTER_INTEL_NPU_INTRIN

} // namespace intrin
} // namespace codegen
} // namespace tvm
