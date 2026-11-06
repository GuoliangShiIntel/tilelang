/*!
 * \file check_budgets.cc
 * \brief Refuse a kernel whose static CMX allocations cannot fit the tile.
 *
 * Every on-chip buffer is one allocation of `shaves_per_tile` slots on the
 * tile's allocator (npu_rt: Ctx::alloc), so the tile's cost is
 * sum(alignUp(bytes, 64)) * shaves_per_tile plus the runtime's own reserve
 * (descriptors, DPU weight table). Runs after LowerOpaqueBlock, when the
 * allocations are AllocBuffer nodes with constant shapes.
 */

#include "support/check.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

// Mirrors Ctx in src/tl_templates/intel_npu/npu_rt_device.hpp: alignment of
// every allocation, and the bytes the runtime takes before the kernel's first.
constexpr int64_t kAlign = 64;
constexpr int64_t kRuntimeReserveBytes = 32 * 1024;

int64_t AlignUp(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

bool IsCmxScope(const std::string &scope) {
  return scope == "shared" || scope == "shared.dyn" || scope == "local";
}

class CmxUsage : public StmtVisitor {
public:
  int64_t per_slot_bytes = 0;
  std::vector<std::pair<std::string, int64_t>> buffers;

private:
  void VisitStmt_(const AllocBufferNode *op) final {
    const Buffer &b = op->buffer;
    if (IsCmxScope(b.scope())) {
      int64_t count = 1;
      for (const auto &dim : b->shape) {
        const auto *imm = dim.as<IntImmNode>();
        if (imm == nullptr) {
          TVM_FFI_THROW(ValueError) << "intel_npu: CMX buffer " << b->name
                                    << " has a non-constant extent " << dim;
        }
        count *= imm->value;
      }
      const int64_t bytes = AlignUp(count * b->dtype.bytes());
      per_slot_bytes += bytes;
      buffers.emplace_back(b->name, bytes);
    }
    StmtVisitor::VisitStmt_(op);
  }
};

} // namespace

namespace transform {

tvm::transform::Pass CheckIntelNpuBudgets() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined()) {
      return func;
    }
    const int64_t cmx_bytes =
        target.value()->GetAttr<Integer>("cmx_bytes").value_or(Integer(0)).IntValue();
    const int64_t spt =
        target.value()->GetAttr<Integer>("shaves_per_tile").value_or(Integer(1)).IntValue();
    if (cmx_bytes <= 0) {
      return func;
    }
    CmxUsage usage;
    usage(func->body);
    const int64_t total = usage.per_slot_bytes * spt + kRuntimeReserveBytes;
    if (total > cmx_bytes) {
      std::ostringstream names;
      for (const auto &[name, bytes] : usage.buffers) {
        names << "\n    " << name << ": " << bytes << " B per SHAVE";
      }
      TVM_FFI_THROW(ValueError)
          << "intel_npu: CMX budget exceeded: " << usage.per_slot_bytes << " B per SHAVE x "
          << spt << " SHAVEs + " << kRuntimeReserveBytes << " B runtime = " << total
          << " B > cmx_bytes " << cmx_bytes << ". Buffers:" << names.str();
    }
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0, "tl.CheckIntelNpuBudgets", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.intel_npu.transform.CheckIntelNpuBudgets", CheckIntelNpuBudgets);
}

} // namespace transform
} // namespace tl
} // namespace tvm
