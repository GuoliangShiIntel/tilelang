/*!
 * \file intel_npu/codegen/rt_mod_intel_npu.cc
 * \brief Source-only build entry for the Intel NPU target.
 */

#include "codegen_intel_npu.h"
#include "config.h"
#include "support/check.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>

namespace tvm {
namespace codegen {

namespace {

std::string IntelNpuCodeGen(const IRModule &mod) {
  CodeGenTileLangIntelNpu cg;
  cg.Init(/*output_ssa=*/false);
  cg.SetEmitLineDirectives(tl::tl_config::EmitLineDirectivesEnabled());
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "intel_npu codegen: can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "intel_npu codegen: function must have "
           "calling_conv=kDeviceKernelLaunch";
    cg.AddFunction(f);
  }
  return cg.Finish();
}

} // namespace

// The compiled path does not exist: moviCompile is driven by an external
// runner. Only the source-only entry is registered, so a BackendContext that
// asks for device compilation fails at the DeviceCodegen boundary.
ffi::Module BuildTileLangIntelNpuWithoutCompile(IRModule mod, Target target) {
  std::string code = IntelNpuCodeGen(mod);
  return CSourceModuleCreate(code, "cpp", ffi::Array<ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_intel_npu_without_compile",
                        BuildTileLangIntelNpuWithoutCompile);
}

} // namespace codegen
} // namespace tvm
