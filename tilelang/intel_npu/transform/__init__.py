"""Intel NPU-specific transformation frontends (native passes in src/intel_npu/transform)."""

from .. import _ffi_api


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
