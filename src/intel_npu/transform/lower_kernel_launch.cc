/*!
 * \file lower_kernel_launch.cc
 * \brief Validate the Intel NPU launch nest: a 1-D grid of tiles.
 *
 * The NPU launch is tiles x SHAVEs. blockIdx.x is the tile and threadIdx.x the
 * SHAVE; any other grid axis has no hardware meaning and is refused here, by
 * name, before MaterializeKernelLaunch binds the launch.
 */

#include "support/check.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

class IntelNpuLaunchValidator : public StmtVisitor {
private:
  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kThreadBinding && op->thread_binding) {
      const std::string tag = op->thread_binding.value()->thread_tag;
      const auto *extent = op->extent.as<IntImmNode>();
      const bool unit = extent != nullptr && extent->value == 1;
      if (!unit && (tag == "blockIdx.y" || tag == "blockIdx.z")) {
        TVM_FFI_THROW(ValueError)
            << "intel_npu: the kernel grid is one-dimensional (tiles); " << tag
            << " has no meaning on this target. Write T.Kernel(num_tiles, "
               "threads=shaves_per_tile) and deal work in the body.";
      }
      if (!unit && (tag == "threadIdx.y" || tag == "threadIdx.z")) {
        TVM_FFI_THROW(ValueError)
            << "intel_npu: SHAVEs form a one-dimensional domain; " << tag
            << " has no meaning on this target.";
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
};

} // namespace

namespace transform {

tvm::transform::Pass LowerIntelNpuKernelLaunch() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    IntelNpuLaunchValidator validator;
    validator(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.LowerIntelNpuKernelLaunch", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.intel_npu.transform.LowerIntelNpuKernelLaunch",
                        LowerIntelNpuKernelLaunch);
}

} // namespace transform
} // namespace tl
} // namespace tvm
