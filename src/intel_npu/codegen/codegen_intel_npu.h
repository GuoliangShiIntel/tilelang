/*!
 * \file intel_npu/codegen/codegen_intel_npu.h
 * \brief Generate SHAVE C++ for the Intel NPU (NPU5) target.
 */
#ifndef TILELANG_INTEL_NPU_CODEGEN_CODEGEN_INTEL_NPU_H_
#define TILELANG_INTEL_NPU_CODEGEN_CODEGEN_INTEL_NPU_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend/common/codegen/codegen_c_base.h"
#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Emits one kernel function per device PrimFunc plus the
 * `kernel_entry` the generic_compute runtime calls on every SHAVE. The kernel
 * function takes the per-SHAVE runtime context `npu::Ctx& ctx` first; launch
 * bindings, CMX allocation, DMA and DPU intrinsics are printed as calls on it.
 */
class CodeGenTileLangIntelNpu final : public CodeGenTileLangCBase {
public:
  using CodeGenC::PrintType;

  void Init(bool output_ssa);
  void AddFunction(const PrimFunc &f);
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const AllocBufferNode *op) final;
  void VisitStmt_(const BindNode *op) final;
  void VisitStmt_(const BufferStoreNode *op) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void PrintStorageSync(const CallNode *op) final;

private:
  struct EntryParam {
    std::string name;
    std::string ctype; // as printed in the kernel signature
    bool is_handle;
    bool is_ddr;       // npu::Ddr<T> (64-bit address) rather than a pointer
  };

  void EmitKernelEntry(const std::string &kernel_name,
                       const std::vector<EntryParam> &params);

  int kernel_count_{0};
  // Base of the three rotating hardware barrier ids sync_grid uses; -1 = software rendezvous.
  int64_t sync_barrier_base_{-1};
  int64_t qg_part_elems_{0}, qg_st_elems_{0};   // merged-FC scratch of the largest gemm_i4/i8 shape
  int cmx_scope_count_{0};                      // CmxScope objects emitted (one per CMX allocation)
  // SHAVE-local buffers of exactly one vector (half8/16/32, float4/8/16) are C++ vector variables, so a lane
  // accumulator lives in a register: moviCompile keeps a CMX buffer in CMX and reloads it after every store
  // (the GDN scan spent 60% of its cycles in NOPs around such round trips, M4).
  std::unordered_map<const VarNode *, int> register_lanes_;
  std::unordered_set<const VarNode *> variable_lane_;   // buffers read or written at a non-constant lane: stay in CMX
  bool IsRegisterBuffer(const VarNode *v) const { return register_lanes_.count(v) != 0; }
};

} // namespace codegen
} // namespace tvm

#endif // TILELANG_INTEL_NPU_CODEGEN_CODEGEN_INTEL_NPU_H_
