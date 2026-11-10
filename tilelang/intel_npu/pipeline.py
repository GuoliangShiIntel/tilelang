"""Intel NPU lowering pipeline.

Phases 0-1 of docs/developer_guide/intel_npu_backend_plan.md: launch binding,
shared normalisation, tile-op lowering to DMA/DPU intrinsics, and the
host/device split. Tile ops outside {copy, gemm} are refused by name in
LowerIntelNpuTileOp.
"""

from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline.pipeline_utils import allow_vectorize, should_force_let_inline

DEFAULT_SHAVES_PER_TILE = 2
DERIVE_GLOBAL_SYNC_KEY = "tl.intel_npu.derive_global_sync"
SCOPED_CMX_KEY = "tl.intel_npu.scoped_cmx"
PREFETCH_KEY = "tl.intel_npu.prefetch"


def _shaves_per_tile(target: Target) -> int:
    value = target.attrs.get("shaves_per_tile")
    return int(value) if value is not None else DEFAULT_SHAVES_PER_TILE


def _derive_global_sync_enabled() -> bool:
    pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx.config.get(DERIVE_GLOBAL_SYNC_KEY, True)) if pass_ctx else True


def _scoped_cmx_enabled() -> bool:
    pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx.config.get(SCOPED_CMX_KEY, True)) if pass_ctx else True


def _prefetch_enabled() -> bool:
    pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx.config.get(PREFETCH_KEY, True)) if pass_ctx else True


def IntelNpuPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.intel_npu.transform.LowerIntelNpuKernelLaunch()(mod)
    # blockIdx.x -> tile, threadIdx.x -> SHAVE: both are real launch dimensions
    # that codegen reads back from the runtime.
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_thread_binding=True,
        default_threads=_shaves_per_tile(target),
        unsupported_annotations=["cluster_dims"],
        lower_grid_binding=True,
    )(mod)

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.VerifyBufferInit()(mod)
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.Simplify()(mod)

    # Before FlattenBuffer: the DMA lowering reads 2-D shapes and strides.
    mod = tilelang.intel_npu.transform.LowerIntelNpuTileOp()(mod)
    # Phase 3b: SEGMENTED/DUPLICATED per fill, then the streamed ring (plan 3.2 order).
    mod = tilelang.intel_npu.transform.PlanIntelNpuDistribution()(mod)
    mod = tilelang.intel_npu.transform.LowerIntelNpuPipelined()(mod)
    # Grid barriers between phases that communicate through DDR, unless written by hand.
    if _derive_global_sync_enabled():
        mod = tilelang.intel_npu.transform.DeriveIntelNpuGlobalSync()(mod)
    # A barrier some SHAVEs would skip is a hang, derived or hand-written.
    mod = tilelang.intel_npu.transform.VerifyIntelNpuUniformSync()(mod)
    # Each DPU workload's cuboid and spatial split, by the VPUNN cost model (plan 3.2).
    mod = tilelang.intel_npu.transform.PlanIntelNpuWorkloads()(mod)
    # Hardware barrier ids for every ring from the job's pool.
    mod = tilelang.intel_npu.transform.AssignIntelNpuBarriers()(mod)

    # CMX allocations sink to the innermost scope covering their uses; codegen frees a scope's
    # allocations at its end (a bump-pointer mark / restore), so phases share CMX (plan M5 PlanCmx).
    if _scoped_cmx_enabled():
        mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    if _scoped_cmx_enabled():
        # phases between grid barriers get their own CMX scope; allocations under SHAVE-dependent branches hoisted
        mod = tilelang.intel_npu.transform.ScopeIntelNpuCmx()(mod)
    # a leading DDR -> CMX load whose address follows the loop variable runs an iteration ahead, into a twin of
    # its destination; before the budget check, which must price the twin
    if _prefetch_enabled():
        mod = tilelang.intel_npu.transform.PlanIntelNpuPrefetch()(mod)
    mod = tilelang.intel_npu.transform.PlanIntelNpuDpuOverlap()(mod)
    mod = tilelang.intel_npu.transform.CheckIntelNpuBudgets()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    # T.vectorized loops become halfN/floatN lane ops (codegen prints moviCompile's vector types).
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=tilelang.transform.get_pass_context()))(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod
