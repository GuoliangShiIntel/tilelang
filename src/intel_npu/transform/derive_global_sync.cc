/*!
 * \file derive_global_sync.cc
 * \brief Insert T.sync_grid() between kernel phases that communicate through DDR.
 *
 * A megakernel is a sequence of phases (the statements of the kernel body's
 * top-level sequence), each dealt over the SHAVEs by the author. A phase that
 * reads a DDR buffer another phase wrote (RAW), overwrites one an earlier phase
 * read (WAR) or wrote (WAW) needs every SHAVE of every tile to have finished
 * that earlier phase: a grid barrier. The author may write it; when they do
 * not, this pass does, at buffer granularity (conservative: two phases that
 * touch disjoint regions of one buffer still get a barrier).
 *
 * Runs after LowerIntelNpuTileOp (DMA copies are intel_npu_dma_copy calls whose
 * DDR side is known) and before LowerOpaqueBlock. Only the top-level sequence
 * is analysed; dependencies carried around a serial loop over phases are the
 * author's (Phase 5 revisits this).
 */

#include "intel_npu/op/builtin.h"
#include "op/builtin.h"
#include "support/check.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

bool IsDdr(const Buffer &buf) {
  const std::string scope = buf.scope();
  return scope.empty() || scope == "global";
}

// DDR buffers a statement reads and writes, by buffer data Var.
class DdrAccess : public StmtExprVisitor {
public:
  std::unordered_set<const VarNode *> reads, writes;
  bool has_sync_grid = false;

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsDdr(op->buffer))
      reads.insert(op->buffer->data.get());
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsDdr(op->buffer))
      writes.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(intel_npu_dma_copy())) {
      // (dst, src, width, rows, dst_stride, src_stride, to_cmx): the DDR side is
      // src when loading into CMX, dst when storing.
      const auto *to_cmx = op->args[6].as<IntImmNode>();
      ICHECK(to_cmx != nullptr);
      const PrimExpr &ddr_ptr = to_cmx->value ? op->args[1] : op->args[0];
      if (const auto *addr = ddr_ptr.as<CallNode>()) {
        if (addr->op.same_as(builtin::address_of())) {
          if (const auto *load = addr->args[0].as<BufferLoadNode>()) {
            if (IsDdr(load->buffer)) {
              (to_cmx->value ? reads : writes).insert(load->buffer->data.get());
            }
          }
        }
      }
      // Index expressions inside the pointers are not DDR accesses; do not recurse
      // into the address_of BufferLoads (they would register as reads).
      for (size_t i = 2; i < op->args.size(); ++i)
        VisitExpr(op->args[i]);
      return;
    }
    if (op->op.same_as(tl::sync_grid())) {
      has_sync_grid = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

bool Intersects(const std::unordered_set<const VarNode *> &a,
                const std::unordered_set<const VarNode *> &b) {
  for (const auto *v : a)
    if (b.count(v))
      return true;
  return false;
}

Stmt MakeSyncGrid() {
  return Evaluate(Call(DataType::Handle(), tl::sync_grid(), {}));
}

// A grid barrier every SHAVE must reach: one under a condition or loop bound
// that depends on the launch coordinates (or on a value derived from them) is
// reached by some SHAVEs and not others, and hangs the kernel.
class UniformSyncVerifier : public StmtExprVisitor {
public:
  void Run(const Stmt &body) { VisitStmt(body); }

private:
  bool DependsOnLaunch(const PrimExpr &e) const {
    bool found = false;
    PostOrderVisit(e, [&](const ffi::ObjectRef &n) {
      if (const auto *v = n.as<VarNode>())
        if (launch_vars_.count(v))
          found = true;
    });
    return found;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>())
        launch_vars_.insert(iv->var.get());
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // `worker = tile * spt + shave`: a bound value derived from the launch coordinates.
  void VisitStmt_(const BindNode *op) final {
    if (DependsOnLaunch(op->value))
      launch_vars_.insert(op->var.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    const bool dep = DependsOnLaunch(op->condition);
    if (dep)
      ++non_uniform_depth_;
    StmtExprVisitor::VisitStmt_(op);
    if (dep)
      --non_uniform_depth_;
  }

  void VisitStmt_(const ForNode *op) final {
    const bool dep = DependsOnLaunch(op->min) || DependsOnLaunch(op->extent);
    if (dep)
      ++non_uniform_depth_;
    StmtExprVisitor::VisitStmt_(op);
    if (dep)
      --non_uniform_depth_;
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::sync_grid()) && non_uniform_depth_ > 0) {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: T.sync_grid() sits under a condition or loop bound that depends on the "
             "tile or SHAVE index, so only some SHAVEs would reach it and the kernel would hang. "
             "Every SHAVE of every tile must execute the same sequence of grid barriers: move the "
             "barrier out of the worker-dependent region.";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::unordered_set<const VarNode *> launch_vars_;
  int non_uniform_depth_ = 0;
};

// Rewrites the first SeqStmt found under the launch wrappers: the phase sequence.
class PhaseSequenceRewriter : public StmtMutator {
public:
  int inserted = 0;

private:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    if (done_)
      return GetRef<Stmt>(op);
    done_ = true;
    ffi::Array<Stmt> out;
    std::unordered_set<const VarNode *> written, read;
    for (const Stmt &phase : op->seq) {
      DdrAccess acc;
      acc(phase);
      if (acc.has_sync_grid) {
        // The author's barrier: everything before it is published.
        out.push_back(phase);
        written.clear();
        read.clear();
        // A phase that both syncs and accesses DDR (e.g. a sync inside a loop
        // body) is not split further; its accesses start the next epoch.
        written.insert(acc.writes.begin(), acc.writes.end());
        read.insert(acc.reads.begin(), acc.reads.end());
        continue;
      }
      const bool raw = Intersects(acc.reads, written);
      const bool war = Intersects(acc.writes, read);
      const bool waw = Intersects(acc.writes, written);
      if (raw || war || waw) {
        out.push_back(MakeSyncGrid());
        ++inserted;
        written.clear();
        read.clear();
      }
      out.push_back(phase);
      written.insert(acc.writes.begin(), acc.writes.end());
      read.insert(acc.reads.begin(), acc.reads.end());
    }
    return SeqStmt(out);
  }

  // Descend only through launch wrappers; a loop body's sequence is the author's.
  Stmt VisitStmt_(const ForNode *op) final { return GetRef<Stmt>(op); }
  Stmt VisitStmt_(const IfThenElseNode *op) final { return GetRef<Stmt>(op); }

  bool done_ = false;
};

} // namespace

namespace transform {

tvm::transform::Pass DeriveIntelNpuGlobalSync() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    PhaseSequenceRewriter rewriter;
    Stmt body = rewriter(func->body);
    if (rewriter.inserted > 0) {
      func.CopyOnWrite()->body = body;
    }
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.DeriveIntelNpuGlobalSync", {});
}

tvm::transform::Pass VerifyIntelNpuUniformSync() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    UniformSyncVerifier verifier;
    verifier.Run(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.VerifyIntelNpuUniformSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.intel_npu.transform.DeriveIntelNpuGlobalSync", DeriveIntelNpuGlobalSync)
      .def("tl.intel_npu.transform.VerifyIntelNpuUniformSync", VerifyIntelNpuUniformSync);
}

} // namespace transform
} // namespace tl
} // namespace tvm
