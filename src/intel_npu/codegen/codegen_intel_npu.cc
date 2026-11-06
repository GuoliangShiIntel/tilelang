/*!
 * \file intel_npu/codegen/codegen_intel_npu.cc
 * \brief Generate SHAVE C++ for the Intel NPU (NPU5) target.
 */

#include "codegen_intel_npu.h"

#include <tvm/tirx/op.h>

#include <sstream>

#include "intel_npu/op/builtin.h"
#include "support/check.h"

namespace tvm {
namespace codegen {

void CodeGenTileLangIntelNpu::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  preamble_stream_ << "#include <tl_templates/intel_npu/npu_rt.hpp>\n\n";
}

void CodeGenTileLangIntelNpu::PrintFuncPrefix(std::ostream &os) {
  os << "static inline";
}

void CodeGenTileLangIntelNpu::PrintType(DataType t, std::ostream &os) {
  if (t.is_scalar()) {
    if (t.is_float16()) {
      os << "half";
      return;
    }
    if (t.is_bfloat16()) {
      os << "bfloat16";
      return;
    }
  }
  CodeGenC::PrintType(t, os);
}

void CodeGenTileLangIntelNpu::PrintStorageScope(const std::string &scope,
                                                std::ostream &os) {
  // CMX and DDR pointers carry no address-space qualifier in SHAVE C++.
}

void CodeGenTileLangIntelNpu::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == tirx::attr::thread_extent) {
    const auto *iv = op->node.as<IterVarNode>();
    ICHECK(iv != nullptr);
    const std::string &tag = iv->thread_tag;
    const auto *extent = op->value.as<IntImmNode>();
    const bool unit = extent != nullptr && extent->value == 1;
    std::string runtime_query;
    if (tag == "blockIdx.x") {
      runtime_query = "ctx.tile_id()";
    } else if (tag == "threadIdx.x") {
      runtime_query = "ctx.shave_id()";
    } else if (unit) {
      // MaterializeKernelLaunch pads the launch to 3-D with unit extents.
      runtime_query = "0";
    } else {
      TVM_FFI_THROW(ValueError)
          << "intel_npu: launch dimension " << tag
          << " has no meaning on this target (the grid is tiles x SHAVEs: "
             "blockIdx.x and threadIdx.x only)";
    }
    PrintIndent();
    stream << "const ";
    PrintType(iv->var.dtype(), stream);
    stream << " " << AllocVarID(iv->var.get()) << " = " << runtime_query
           << ";  // " << tag << " < " << op->value << "\n";
    PrintStmt(op->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangIntelNpu::VisitStmt_(const AllocBufferNode *op) {
  const std::string vid = AllocVarID(op->buffer->data.get());
  const std::string scope = GetPtrStorageScope(op->buffer->data);
  int64_t count = 1;
  for (const auto &dim : op->buffer->shape) {
    const auto *imm = dim.as<IntImmNode>();
    if (imm == nullptr) {
      TVM_FFI_THROW(ValueError) << "intel_npu: CMX buffer " << op->buffer->name
                                << " has a non-constant extent " << dim;
    }
    count *= imm->value;
  }
  if (scope != "shared" && scope != "shared.dyn" && scope != "local") {
    TVM_FFI_THROW(ValueError)
        << "intel_npu: buffer " << op->buffer->name << " has scope '" << scope
        << "'; on-chip buffers are scope shared (tile) or local (SHAVE)";
  }
  // Phase 1: every on-chip buffer is a SHAVE-private CMX slice; tile-shared
  // placement for scope "shared" arrives with the multi-tile flow (Phase 3).
  PrintIndent();
  PrintType(op->buffer->dtype, stream);
  stream << "* " << vid << " = ctx.alloc<";
  PrintType(op->buffer->dtype, stream);
  stream << ">(" << count << ");  // " << scope << "\n";
  RegisterHandleType(op->buffer->data.get(), op->buffer->dtype);
}

void CodeGenTileLangIntelNpu::VisitExpr_(const CallNode *op,
                                         std::ostream &os) {
  auto args = [&](size_t first, size_t last) {
    std::string s;
    for (size_t i = first; i < last; ++i) {
      if (i != first)
        s += ", ";
      s += PrintExpr(op->args[i]);
    }
    return s;
  };
  if (op->op.same_as(tl::intel_npu_dma_copy())) {
    // (dst, src, width_bytes, rows, dst_stride, src_stride, to_cmx)
    const auto *to_cmx = op->args[6].as<IntImmNode>();
    ICHECK(to_cmx != nullptr);
    os << (to_cmx->value ? "ctx.dma_load(" : "ctx.dma_store(") << args(0, 6)
       << ")";
    return;
  }
  if (op->op.same_as(tl::intel_npu_dpu_matmul())) {
    os << "ctx.dpu_matmul(" << args(0, 6) << ")";
    return;
  }
  if (op->op.same_as(tl::sync_grid())) {
    // Every SHAVE of every tile: the megakernel phase boundary.
    os << "ctx.sync_grid()";
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangIntelNpu::PrintStorageSync(const CallNode *op) {
  const auto *scope = op->args[0].as<StringImmNode>();
  ICHECK(scope != nullptr);
  const std::string s = scope->value;
  if (s == "shared" || s == "shared.dyn") {
    PrintIndent();
    stream << "ctx.sync_tile();\n";
    return;
  }
  if (s == "global") {
    PrintIndent();
    stream << "ctx.sync_grid();\n";
    return;
  }
  TVM_FFI_THROW(ValueError) << "intel_npu: storage sync scope '" << s
                            << "' has no meaning on this target (shared = the tile's SHAVEs, "
                               "global / T.sync_grid() = every SHAVE)";
}

void CodeGenTileLangIntelNpu::AddFunction(const PrimFunc &f) {
  ICHECK_EQ(kernel_count_, 0)
      << "intel_npu: one kernel_entry per module; a second device function "
         "was found";
  ++kernel_count_;

  this->InitFuncState(f);
  ReserveKeywordsAsUnique();
  name_supply_->ReserveName("kernel_entry");
  name_supply_->ReserveName("params");
  name_supply_->ReserveName("env");
  name_supply_->ReserveName("ctx");

  auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol)
      << "intel_npu: expect PrimFunc to have the global_symbol attribute";
  const std::string kernel_name =
      static_cast<std::string>(global_symbol.value());
  bool no_alias = f->HasNonzeroAttr(tirx::attr::kNoAlias);

  for (const auto &param : f->params) {
    if (param->dtype.is_handle()) {
      alloc_storage_scope_[param.get()] = GetPtrStorageScope(param);
    }
  }

  std::vector<EntryParam> entry_params;

  this->PrintFuncPrefix(stream);
  stream << " ";
  CodeGenC::PrintType(f->ret_type, stream);
  stream << " " << kernel_name << "(npu::Ctx& ctx";
  for (size_t i = 0; i < f->params.size(); ++i) {
    tirx::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    stream << ", ";
    std::ostringstream ctype;
    bool is_ddr = false;
    if (v.dtype().is_handle()) {
      const std::string scope = GetPtrStorageScope(v);
      is_ddr = scope.empty() || scope == "global";
      std::ostringstream elem;
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
          PrintType(prim->dtype, elem);
        } else {
          elem << "void";
        }
      } else {
        elem << "void";
      }
      // DDR addresses are 64-bit on a 32-bit SHAVE: a typed address, not a pointer.
      if (is_ddr) {
        ctype << "npu::Ddr<" << elem.str() << ">";
      } else {
        ctype << elem.str() << "*";
      }
      stream << ctype.str();
      if (no_alias && !is_ddr) {
        PrintRestrict(v, stream);
      }
    } else {
      PrintType(v.dtype(), ctype);
      stream << ctype.str();
    }
    stream << " " << vid;
    entry_params.push_back({vid, ctype.str(), v.dtype().is_handle(), is_ddr});
  }
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  stream << "}\n\n";

  EmitKernelEntry(kernel_name, entry_params);
}

void CodeGenTileLangIntelNpu::EmitKernelEntry(
    const std::string &kernel_name, const std::vector<EntryParam> &params) {
  // generic_compute calls this on every SHAVE of every tile with the same
  // parameter block: params[i] is the i-th kernel argument as a 64-bit word,
  // and an optional params[n] is the per-SHAVE status area the host reads.
  stream << "void kernel_entry(int numParams, uint64_t* params, const "
            "npu::Environment& env) {\n";
  stream << "  npu::Ctx ctx(env, numParams, params, " << params.size()
         << ");\n";
  stream << "  if (!ctx.ok()) { ctx.finish(); return; }\n";
  stream << "  " << kernel_name << "(ctx";
  for (size_t i = 0; i < params.size(); ++i) {
    stream << ", ";
    if (params[i].is_ddr) {
      stream << params[i].ctype << "(params[" << i << "])";
    } else if (params[i].is_handle) {
      stream << "reinterpret_cast<" << params[i].ctype << ">(params[" << i
             << "])";
    } else {
      stream << "static_cast<" << params[i].ctype << ">(params[" << i << "])";
    }
  }
  stream << ");\n";
  stream << "  ctx.finish();\n";
  stream << "}\n";
}

} // namespace codegen
} // namespace tvm
