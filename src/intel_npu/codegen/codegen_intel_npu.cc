/*!
 * \file intel_npu/codegen/codegen_intel_npu.cc
 * \brief Generate SHAVE C++ for the Intel NPU (NPU5) target.
 */

#include "codegen_intel_npu.h"

#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
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
  } else if (t.lanes() >= 2 && t.lanes() <= 32 && (t.lanes() & (t.lanes() - 1)) == 0) {
    // moviCompile's ext_vector typedefs: half8/16/32, float4/8/16, int4/8/16, ...
    // (VectorizeLoop decides the lanes; the SHAVE VRF is 512 bits).
    if (t.is_float16()) {
      os << "half" << t.lanes();
      return;
    }
    if (t.is_float() && t.bits() == 32) {
      os << "float" << t.lanes();
      return;
    }
    if (t.is_int() && t.bits() == 32) {
      os << "int" << t.lanes();
      return;
    }
    if (t.is_uint() && t.bits() == 32) {
      os << "uint" << t.lanes();
      return;
    }
    if (t.is_int() && t.bits() == 16) {
      os << "short" << t.lanes();
      return;
    }
    if (t.is_uint() && t.bits() == 16) {
      os << "ushort" << t.lanes();
      return;
    }
    if (t.is_int() && t.bits() == 8) {
      os << "schar" << t.lanes();
      return;
    }
    if (t.is_uint() && t.bits() == 8) {
      os << "uchar" << t.lanes();
      return;
    }
  }
  CodeGenC::PrintType(t, os);
}

void CodeGenTileLangIntelNpu::VisitExpr_(const BroadcastNode *op, std::ostream &os) {
  // A scalar cast to an ext_vector type splats.
  os << "((";
  PrintType(op->dtype, os);
  os << ")(" << PrintExpr(op->value) << "))";
}

void CodeGenTileLangIntelNpu::VisitExpr_(const CastNode *op, std::ostream &os) {
  if (op->dtype.lanes() == 1) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }
  // A C cast between same-size vectors is a bitcast; lane-wise conversion is this builtin.
  os << "__builtin_convertvector(" << PrintExpr(op->value) << ", ";
  PrintType(op->dtype, os);
  os << ")";
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
  if (op->attr_key == tl::kIntelNpuNotesAttr) {
    const auto *notes = op->value.as<StringImmNode>();
    ICHECK(notes != nullptr) << "intel_npu.notes must carry a StringImm";
    std::istringstream lines(notes->value);
    for (std::string line; std::getline(lines, line);) {
      PrintIndent();
      stream << "// intel_npu: " << line << "\n";
    }
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
  // scope local: a SHAVE-private slice of one allocation of shaves_per_tile
  // slots; scope shared: one tile-wide allocation at the same address on every
  // SHAVE (and the same offset on every tile, which a multicast fill needs).
  const bool shared = scope != "local";
  PrintIndent();
  PrintType(op->buffer->dtype, stream);
  // Distinct allocations never alias; without __restrict moviCompile keeps every
  // vector accumulator in CMX and re-issues each load after the previous store
  // (NOP 8 per load in the flash softmax row loop, M1 step 3c).
  stream << "* __restrict " << vid << " = ctx." << (shared ? "alloc_shared<" : "alloc<");
  PrintType(op->buffer->dtype, stream);
  stream << ">(" << count << ");  // " << scope << "\n";
  RegisterHandleType(op->buffer->data.get(), op->buffer->dtype);
}

void CodeGenTileLangIntelNpu::VisitStmt_(const BindNode *op) {
  if (const auto *call = op->value.as<CallNode>()) {
    if (call->op.same_as(tl::intel_npu_ring_begin())) {
      PrintIndent();
      stream << "npu::Ctx::Ring " << AllocVarID(op->var.get()) << " = " << PrintExpr(op->value)
             << ";\n";
      return;
    }
  }
  CodeGenC::VisitStmt_(op);
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
    os << "ctx.dpu_matmul(" << args(0, 8) << ")";
    return;
  }
  if (op->op.same_as(tl::intel_npu_ring_begin())) {
    os << "ctx.ring_begin(" << args(0, 6) << ")";
    return;
  }
  if (op->op.same_as(tl::intel_npu_ring_uses())) {
    os << "ctx.ring_uses(" << args(0, 1) << ")";
    return;
  }
  if (op->op.same_as(tl::intel_npu_ring_slot())) {
    // fp16 is the only DPU operand type lowered today.
    os << "ctx.ring_slot<half>(" << args(0, 1) << ")";
    return;
  }
  if (op->op.same_as(tl::intel_npu_ring_fill())) {
    os << "ctx.ring_fill(" << args(0, 6) << ")";
    return;
  }
  if (op->op.same_as(tl::intel_npu_ring_dpu_submit())) {
    os << "ctx.ring_dpu_submit(" << args(0, 10) << ")";
    return;
  }
  if (op->op.same_as(tl::intel_npu_ring_dpu_wait())) {
    os << "ctx.ring_dpu_wait(" << args(0, 2) << ")";
    return;
  }
  if (op->op.same_as(tl::intel_npu_ring_end())) {
    os << "ctx.ring_end(" << args(0, 1) << ")";
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

  // sync_grid on hardware barriers: the top kSyncBarriers ids of the JOB's pool,
  // [barrier_pool_first, barriers_per_tile x launch tiles). An id beyond the
  // job's allocation is a context violation that hangs the run, so the pool
  // follows the kernel's own tile count, not the platform's.
  sync_barrier_base_ = -1;
  const bool hw_sync = tvm::transform::PassContext::Current()
                           ->GetConfig<Bool>(tl::kIntelNpuHwSyncGrid, Bool(true))
                           .value();
  if (hw_sync) {
    if (auto target = f->GetAttr<Target>(tvm::attr::kTarget)) {
      const int64_t first =
          target.value()->GetAttr<Integer>("barrier_pool_first").value_or(Integer(-1)).IntValue();
      const int64_t per_tile =
          target.value()->GetAttr<Integer>("barriers_per_tile").value_or(Integer(0)).IntValue();
      int64_t launch_tiles = 0;
      tirx::PostOrderVisit(f->body, [&](const ffi::ObjectRef &n) {
        if (const auto *attr = n.as<AttrStmtNode>()) {
          if (attr->attr_key == tirx::attr::thread_extent) {
            if (const auto *iv = attr->node.as<IterVarNode>()) {
              if (iv->thread_tag == "blockIdx.x") {
                if (const auto *imm = attr->value.as<IntImmNode>())
                  launch_tiles = imm->value;
              }
            }
          }
        }
      });
      const int64_t pool_end = per_tile * launch_tiles;
      if (first >= 0 && pool_end - first >= 3) {
        sync_barrier_base_ = pool_end - 3;
      } else {
        TVM_FFI_THROW(ValueError)
            << "intel_npu: the hardware-barrier pool [" << first << ", " << pool_end
            << ") of a " << launch_tiles
            << "-tile launch has no room for the 3 sync_grid barriers; set "
               "tl.intel_npu.hw_sync_grid=False or launch more tiles";
      }
    }
  }

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
  if (sync_barrier_base_ >= 0) {
    // Named here so a launch that never returns can be matched to the ids it touches.
    stream << "  // hardware barrier ids used by this kernel: sync_grid " << sync_barrier_base_ << ".."
           << sync_barrier_base_ + 2 << "\n";
  }
  stream << "  npu::Ctx ctx(env, numParams, params, " << params.size();
  if (sync_barrier_base_ >= 0) {
    stream << ", /*sync_barrier_base=*/" << sync_barrier_base_;
  }
  stream << ");\n";
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
