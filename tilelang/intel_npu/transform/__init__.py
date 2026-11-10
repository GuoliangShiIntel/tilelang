"""Intel NPU-specific transformation frontends (native passes in src/intel_npu/transform,
Python passes alongside this file)."""

from .. import _ffi_api
from .plan_workloads import PlanIntelNpuWorkloads  # noqa: F401
from .lower_pipelined import PlanIntelNpuDistribution, LowerIntelNpuPipelined  # noqa: F401
from .assign_barriers import AssignIntelNpuBarriers  # noqa: F401
from .scope_cmx import ScopeIntelNpuCmx  # noqa: F401
from .prefetch import PlanIntelNpuPrefetch, PlanIntelNpuDpuOverlap, PlanIntelNpuDmaBatch  # noqa: F401


def LowerIntelNpuKernelLaunch():
    """Validate the launch nest for the NPU: a 1-D grid of tiles, one SHAVE
    domain per tile. Rejects blockIdx.y/z before MaterializeKernelLaunch."""
    return _ffi_api.LowerIntelNpuKernelLaunch()  # type: ignore


def LowerIntelNpuTileOp():
    """Lower tile ops to NPU runtime intrinsics: copy (DDR<->CMX, 1-D/2-D) ->
    DMA, gemm (fp16, transpose_B, clear_accum) -> one DPU workload. Refuses
    every other tile op by name. Runs before FlattenBuffer."""
    return _ffi_api.LowerIntelNpuTileOp()  # type: ignore


def CheckIntelNpuBudgets():
    """Refuse a kernel whose CMX allocations (per SHAVE x shaves_per_tile +
    runtime reserve) exceed the target's cmx_bytes, naming the buffers."""
    return _ffi_api.CheckIntelNpuBudgets()  # type: ignore


def DeriveIntelNpuGlobalSync():
    """Insert T.sync_grid() between top-level phases that communicate through
    DDR (RAW/WAR/WAW at buffer granularity) when the author did not write one.
    Runs after LowerIntelNpuTileOp and before LowerOpaqueBlock."""
    return _ffi_api.DeriveIntelNpuGlobalSync()  # type: ignore


def VerifyIntelNpuUniformSync():
    """Refuse a T.sync_grid() under a condition or loop bound that depends on
    the tile/SHAVE index: some SHAVEs would never reach it and the kernel hangs."""
    return _ffi_api.VerifyIntelNpuUniformSync()  # type: ignore
