/*!
 * \file lower_tile_op.cc
 * \brief Lower TileLang tile ops to Intel NPU runtime intrinsics.
 *
 * Phase 1 (docs/developer_guide/intel_npu_backend_plan.md):
 *   tl.tileop.copy  DDR<->CMX, 1-D or 2-D row-major  -> intel_npu_dma_copy
 *   tl.tileop.gemm  fp16, transpose_B, clear_accum   -> intel_npu_dpu_matmul
 * Everything else is refused by name with the plan item that adds it. Runs
 * before FlattenBuffer so row strides are still visible; the pointers it emits
 * are address_of(BufferLoad) and flatten with the buffers.
 */

#include "intel_npu/op/builtin.h"
#include "op/builtin.h"
#include "support/check.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

struct RegionInfo {
  Buffer buffer;
  ffi::Array<PrimExpr> indices;
  std::vector<int64_t> extents; // leading unit extents dropped
  std::vector<int64_t> full_extents;
};

bool IsDdr(const Buffer &buf) {
  const std::string scope = buf.scope();
  return scope.empty() || scope == "global";
}

bool IsCmx(const Buffer &buf) {
  const std::string scope = buf.scope();
  return scope == "shared" || scope == "shared.dyn" || scope == "local";
}

RegionInfo ParseRegion(const PrimExpr &e, const char *what) {
  const auto *call = e.as<CallNode>();
  if (call == nullptr || !call->op.same_as(tl::region())) {
    TVM_FFI_THROW(ValueError)
        << "intel_npu: " << what << " operand is not a tl.region: " << e;
  }
  const auto *load = call->args[0].as<BufferLoadNode>();
  ICHECK(load != nullptr) << "tl.region expects a BufferLoad";
  RegionInfo info;
  info.buffer = load->buffer;
  info.indices = load->indices;
  for (size_t i = 2; i < call->args.size(); ++i) {
    const auto *imm = call->args[i].as<IntImmNode>();
    if (imm == nullptr) {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: " << what << " on " << info.buffer->name
          << " has a non-constant extent " << call->args[i]
          << "; tile extents must be compile-time constants";
    }
    info.full_extents.push_back(imm->value);
  }
  // A [1, ..., 1, R, C] region of a higher-rank tensor is a 2-D tile.
  size_t lead = 0;
  while (lead + 1 < info.full_extents.size() && info.full_extents[lead] == 1)
    ++lead;
  info.extents.assign(info.full_extents.begin() + lead, info.full_extents.end());
  return info;
}

int64_t ConstDim(const PrimExpr &e, const Buffer &buf, const char *field) {
  const auto *imm = e.as<IntImmNode>();
  if (imm == nullptr) {
    TVM_FFI_THROW(ValueError) << "intel_npu: buffer " << buf->name << " has a non-constant " << field
                              << " (" << e << "); only static shapes are supported in Phase 1";
  }
  return imm->value;
}

// Row stride in elements of the region's second-innermost dimension (row-major),
// or the region's own width for a 1-D region.
int64_t RowStrideElems(const RegionInfo &r) {
  const Buffer &b = r.buffer;
  const size_t nd = b->shape.size();
  if (nd == 1 || r.extents.size() == 1)
    return r.extents.back();
  if (!b->strides.empty())
    return ConstDim(b->strides[nd - 2], b, "stride");
  return ConstDim(b->shape[nd - 1], b, "shape");
}

PrimExpr AddressOf(const RegionInfo &r) {
  return Call(DataType::Handle(), builtin::address_of(),
              {BufferLoad(r.buffer, r.indices)});
}

class IntelNpuTileOpLowerer : public StmtExprMutator {
public:
  IntelNpuTileOpLowerer(int64_t dpu_z_align, int64_t dpu_max_output_points)
      : dpu_z_align_(dpu_z_align), dpu_max_output_points_(dpu_max_output_points) {}

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr) {
      return StmtExprMutator::VisitStmt_(op);
    }
    const auto *op_node = call->op.as<OpNode>();
    if (op_node == nullptr) {
      return StmtExprMutator::VisitStmt_(op);
    }
    const std::string name = op_node->name;
    if (name == "tl.tileop.copy") {
      return LowerCopy(call);
    }
    if (name == "tl.tileop.gemm") {
      return LowerGemm(call);
    }
    if (name.rfind("tl.tileop.", 0) == 0) {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: tile op " << name
          << " is not yet supported on this target (plan: copy and gemm are "
             "Phase 1; fill/clear and SHAVE ops are Phase 4)";
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  Stmt LowerCopy(const CallNode *call) {
    RegionInfo src = ParseRegion(call->args[0], "copy source");
    RegionInfo dst = ParseRegion(call->args[1], "copy destination");
    if (src.extents != dst.extents) {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: copy " << src.buffer->name << " -> " << dst.buffer->name
          << " has different extents on each side";
    }
    if (src.extents.empty() || src.extents.size() > 2) {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: copy " << src.buffer->name << " -> " << dst.buffer->name
          << " is " << src.extents.size()
          << "-D; the DMA lowering takes 1-D or 2-D row-major regions (Phase 1)";
    }
    if (src.buffer->dtype != dst.buffer->dtype) {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: copy " << src.buffer->name << " -> " << dst.buffer->name
          << " changes dtype (" << src.buffer->dtype << " -> " << dst.buffer->dtype
          << "); a DMA moves bytes, convert on the SHAVE instead";
    }
    int to_cmx;
    if (IsDdr(src.buffer) && IsCmx(dst.buffer)) {
      to_cmx = 1;
    } else if (IsCmx(src.buffer) && IsDdr(dst.buffer)) {
      to_cmx = 0;
    } else {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: copy " << src.buffer->name << " (" << src.buffer.scope()
          << ") -> " << dst.buffer->name << " (" << dst.buffer.scope()
          << "): Phase 1 lowers DDR<->CMX only";
    }
    const int64_t elem_bytes = src.buffer->dtype.bytes();
    const int64_t rows = src.extents.size() == 2 ? src.extents[0] : 1;
    const int64_t width = src.extents.back() * elem_bytes;
    const int64_t src_stride = RowStrideElems(src) * elem_bytes;
    const int64_t dst_stride = RowStrideElems(dst) * elem_bytes;
    auto i32 = [](int64_t v) { return IntImm(DataType::Int(32), v); };
    return Evaluate(Call(DataType::Handle(), intel_npu_dma_copy(),
                         {AddressOf(dst), AddressOf(src), i32(width), i32(rows),
                          i32(dst_stride), i32(src_stride), i32(to_cmx)}));
  }

  Stmt LowerGemm(const CallNode *call) {
    RegionInfo a = ParseRegion(call->args[0], "gemm A");
    RegionInfo b = ParseRegion(call->args[1], "gemm B");
    RegionInfo c = ParseRegion(call->args[2], "gemm C");
    const bool trans_a = call->args[3].as<IntImmNode>() && call->args[3].as<IntImmNode>()->value != 0;
    const bool trans_b = call->args[4].as<IntImmNode>() && call->args[4].as<IntImmNode>()->value != 0;
    const int64_t M = ConstDim(call->args[5], c.buffer, "M");
    const int64_t N = ConstDim(call->args[6], c.buffer, "N");
    const int64_t K = ConstDim(call->args[7], c.buffer, "K");
    const bool clear_accum = call->args.size() > 9 && call->args[9].as<IntImmNode>() &&
                             call->args[9].as<IntImmNode>()->value != 0;

    if (trans_a || !trans_b) {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: gemm takes A[M, K] and B^T[N, K]: write "
             "T.gemm(A, B, C, transpose_B=True)";
    }
    if (!clear_accum) {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: gemm into " << c.buffer->name
          << " accumulates (clear_accum=False), but the DPU overwrites its output. Write "
             "T.gemm(..., clear_accum=True) with the whole K in one workload; a K split "
             "needs a SHAVE accumulate (plan Phase 4)";
    }
    for (const RegionInfo *r : {&a, &b, &c}) {
      if (!IsCmx(r->buffer)) {
        TVM_FFI_THROW(ValueError) << "intel_npu: gemm operand " << r->buffer->name
                                  << " must live in CMX (scope shared/local), not "
                                  << r->buffer.scope();
      }
      if (r->buffer->dtype != DataType::Float(16)) {
        TVM_FFI_THROW(ValueError) << "intel_npu: gemm operand " << r->buffer->name << " is "
                                  << r->buffer->dtype << "; the Phase 1 DPU path is fp16 in, fp16 out";
      }
      if (r->extents.size() != 2) {
        TVM_FFI_THROW(ValueError) << "intel_npu: gemm operand " << r->buffer->name << " is not 2-D";
      }
      // The DPU reads each operand as a dense [rows, cols] block.
      if (RowStrideElems(*r) != r->extents[1]) {
        TVM_FFI_THROW(ValueError)
            << "intel_npu: gemm operand " << r->buffer->name << " region width " << r->extents[1]
            << " differs from its CMX row stride " << RowStrideElems(*r)
            << "; the DPU needs a contiguous block, allocate the buffer at the operand's shape";
      }
    }
    auto check = [](const RegionInfo &r, int64_t rows, int64_t cols, const char *which) {
      if (r.extents[0] != rows || r.extents[1] != cols) {
        TVM_FFI_THROW(ValueError) << "intel_npu: gemm " << which << " region is [" << r.extents[0]
                                  << ", " << r.extents[1] << "], expected [" << rows << ", " << cols << "]";
      }
    };
    check(a, M, K, "A");
    check(b, N, K, "B");
    check(c, M, N, "C");
    // DPU workload legality (NPU5 HAS via the VPUX knowledge base, dpu.md 4.6/13).
    if (N % dpu_z_align_ != 0 || K % dpu_z_align_ != 0) {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: gemm N=" << N << ", K=" << K << " must be multiples of " << dpu_z_align_
          << ": the DPU's output and input channel extents are " << dpu_z_align_ << "-aligned";
    }
    if (M * N > dpu_max_output_points_) {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: gemm output tile " << M << "x" << N << " = " << M * N
          << " points exceeds the DPU's " << dpu_max_output_points_
          << " output points per workload; split the output tile";
    }
    auto i32 = [](int64_t v) { return IntImm(DataType::Int(32), v); };
    return Evaluate(Call(DataType::Handle(), intel_npu_dpu_matmul(),
                         {AddressOf(c), AddressOf(a), AddressOf(b), i32(M), i32(N), i32(K)}));
  }

  int64_t dpu_z_align_;
  int64_t dpu_max_output_points_;
};

} // namespace

namespace transform {

tvm::transform::Pass LowerIntelNpuTileOp() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    int64_t z_align = 16, max_points = 8192;
    if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
      z_align = target.value()->GetAttr<Integer>("dpu_z_align").value_or(Integer(16)).IntValue();
      max_points =
          target.value()->GetAttr<Integer>("dpu_max_output_points").value_or(Integer(8192)).IntValue();
    }
    IntelNpuTileOpLowerer lowerer(z_align, max_points);
    func.CopyOnWrite()->body = lowerer(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.LowerIntelNpuTileOp", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.intel_npu.transform.LowerIntelNpuTileOp",
                        LowerIntelNpuTileOp);
}

} // namespace transform
} // namespace tl
} // namespace tvm
