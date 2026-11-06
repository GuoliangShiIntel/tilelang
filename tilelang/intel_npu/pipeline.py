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
from tilelang.backend.pass_pipeline.pipeline_utils import should_force_let_inline

DEFAULT_SHAVES_PER_TILE = 2
DERIVE_GLOBAL_SYNC_KEY = "tl.intel_npu.derive_global_sync"


def _shaves_per_tile(target: Target) -> int:
    value = target.attrs.get("shaves_per_tile")
    return int(value) if value is not None else DEFAULT_SHAVES_PER_TILE


def _derive_global_sync_enabled() -> bool:
    pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx.config.get(DERIVE_GLOBAL_SYNC_KEY, True)) if pass_ctx else True


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
    # Grid barriers between phases that communicate through DDR, unless written by hand.
    if _derive_global_sync_enabled():
        mod = tilelang.intel_npu.transform.DeriveIntelNpuGlobalSync()(mod)

    # No PlanAndUpdateBufferAllocationLocation: it sinks CMX buffers into the
    # loops that use them, and a CMX allocation is made once per kernel.
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.intel_npu.transform.CheckIntelNpuBudgets()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
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
