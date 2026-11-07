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
#include "op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
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
constexpr int64_t kRuntimeReserveBytes = 72 * 1024;  // 2 DPU descriptor sets x 2 SHAVEs x 16 KiB weight table + descriptors

int64_t AlignUp(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

bool IsCmxScope(const std::string &scope) {
  return scope == "shared" || scope == "shared.dyn" || scope == "local";
}

// Peak CMX of a statement under the scoped allocator (npu_rt CmxScope): the allocations declared
// directly in a C++ block live to its end, so a block costs their sum plus the largest peak of any
// nested block (for / if bodies, lexical_alloc_scope). Blocks one after another reuse the same CMX.
class CmxUsage {
public:
  int64_t per_slot_bytes = 0; // scope local: x shaves_per_tile, at the peak
  int64_t shared_bytes = 0;   // scope shared: once per tile, at the peak
  int64_t qg_part_elems = 0, qg_st_elems = 0;   // merged-FC scratch of the largest gemm_i4/i8 (per SHAVE)
  std::vector<std::pair<std::string, int64_t>> buffers;   // every allocation (the report)
  int spt = 1;

  // Mirrors Ctx::reserve_qg: part + sT halves + the 16 KiB weight table, per SHAVE, when the kernel has a
  // quantised GEMV (codegen emits the reservation from the same calls).
  int64_t qg_bytes_per_slot() const {
    if (qg_part_elems == 0)
      return 0;
    return AlignUp(2 * qg_part_elems) + AlignUp(2 * qg_st_elems) + 16 * 1024;
  }

  void Run(const Stmt &body) {
    QgScan scan(this);
    scan(body);
    const Usage root = Block(body);
    per_slot_bytes = root.local;
    shared_bytes = root.shared;
  }

private:
  struct Usage {
    int64_t local = 0, shared = 0;
    int64_t tile_bytes(int spt) const { return local * spt + shared; }
  };

  class QgScan : public StmtExprVisitor {
  public:
    explicit QgScan(CmxUsage *u) : u_(u) {}
    void VisitExpr_(const CallNode *op) final {
      const auto *op_node = op->op.as<tvm::OpNode>();
      if (op_node != nullptr && (op_node->name == "tl.intel_npu_dpu_matmul_i4g" ||
                                 op_node->name == "tl.intel_npu_dpu_matmul_i8g")) {
        const auto *M = op->args[4].as<IntImmNode>();
        const auto *N = op->args[5].as<IntImmNode>();
        const auto *K = op->args[6].as<IntImmNode>();
        const auto *G = op->args[7].as<IntImmNode>();
        if (M && N && K && G && G->value > 0) {
          const int64_t ng = K->value / G->value;
          u_->qg_part_elems = std::max(u_->qg_part_elems, M->value * ng * N->value * ng);
          u_->qg_st_elems = std::max(u_->qg_st_elems, ng * N->value);
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    CmxUsage *u_;
  };

  // The peak of one C++ block whose body is `stmt`: allocations declared in it (through SeqStmt and
  // transparent AttrStmt, which codegen prints inline) plus the heaviest nested block
  // (for / while / if bodies, lexical_alloc_scope), since blocks one after another reuse the CMX.
  Usage Block(const Stmt &stmt) {
    Usage own, deepest;
    Collect(stmt, &own, &deepest);
    return Usage{own.local + deepest.local, own.shared + deepest.shared};
  }

  void Collect(const Stmt &s, Usage *own, Usage *deepest) {
    if (const auto *alloc = s.as<AllocBufferNode>()) {
      Add(alloc, own);
    } else if (const auto *seq = s.as<SeqStmtNode>()) {
      for (const Stmt &item : seq->seq)
        Collect(item, own, deepest);
    } else if (const auto *attr = s.as<AttrStmtNode>()) {
      if (attr->attr_key == tl::attr::kLexicalAllocScope)
        Deeper(Block(attr->body), deepest);
      else
        Collect(attr->body, own, deepest);
    } else if (const auto *f = s.as<ForNode>()) {
      Deeper(Block(f->body), deepest);
    } else if (const auto *w = s.as<WhileNode>()) {
      Deeper(Block(w->body), deepest);
    } else if (const auto *ite = s.as<IfThenElseNode>()) {
      Deeper(Block(ite->then_case), deepest);
      if (ite->else_case.defined())
        Deeper(Block(ite->else_case.value()), deepest);
    }
  }

  void Deeper(const Usage &u, Usage *deepest) const {
    if (u.tile_bytes(spt) > deepest->tile_bytes(spt))
      *deepest = u;
  }

  void Add(const AllocBufferNode *op, Usage *u) {
    const Buffer &b = op->buffer;
    if (!IsCmxScope(b.scope()))
      return;
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
    if (b.scope() == "local") {
      u->local += bytes;
    } else {
      u->shared += bytes;
    }
    buffers.emplace_back(b->name + (b.scope() == "local" ? "" : " (shared)"), bytes);
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
    usage.spt = static_cast<int>(spt);
    usage.Run(func->body);
    const int64_t qg = usage.qg_bytes_per_slot() * spt;
    const int64_t total = usage.per_slot_bytes * spt + usage.shared_bytes + kRuntimeReserveBytes + qg;
    if (total > cmx_bytes) {
      std::ostringstream names;
      for (const auto &[name, bytes] : usage.buffers) {
        names << "\n    " << name << ": " << bytes << " B";
      }
      TVM_FFI_THROW(ValueError)
          << "intel_npu: CMX budget exceeded at the peak: " << usage.per_slot_bytes << " B per SHAVE x "
          << spt << " SHAVEs + " << usage.shared_bytes << " B shared + " << kRuntimeReserveBytes
          << " B runtime + " << qg << " B merged-FC scratch = " << total << " B > cmx_bytes " << cmx_bytes
          << " (blocks one after another share CMX; the peak is the heaviest nesting). Buffers:" << names.str();
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
