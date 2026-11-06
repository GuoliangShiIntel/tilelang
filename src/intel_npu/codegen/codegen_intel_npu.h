/*!
 * \file intel_npu/codegen/codegen_intel_npu.h
 * \brief Generate SHAVE C++ for the Intel NPU (NPU5) target.
 */
#ifndef TILELANG_INTEL_NPU_CODEGEN_CODEGEN_INTEL_NPU_H_
#define TILELANG_INTEL_NPU_CODEGEN_CODEGEN_INTEL_NPU_H_

#include <string>
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
  void VisitStmt_(const AllocBufferNode *op) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;

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
};

} // namespace codegen
} // namespace tvm

#endif // TILELANG_INTEL_NPU_CODEGEN_CODEGEN_INTEL_NPU_H_
