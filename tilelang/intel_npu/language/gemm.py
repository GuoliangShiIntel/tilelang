"""Bounded runtime-row FP16 GEMM for Intel NPU CMX buffers."""

from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.language.gemm_op import _gemm_impl
from tilelang.tileop.base import GemmWarpPolicy
from tvm import tirx
from tvm.tirx import Buffer

from ..hw_facts import fact
from .quant import _addr

__all__ = ["gemm_async", "gemm_rows", "dpu_wait"]


def gemm_async(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    annotations: dict | None = None,
):
    """Issue an FP16 DPU GEMM without waiting; join it with `dpu_wait` before reading its output."""
    async_annotations = dict(annotations or {})
    async_annotations["intel_npu_async"] = tirx.IntImm("int32", 1)
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        None,
        annotations=async_annotations,
    )


def dpu_wait():
    """Wait for the in-flight FP16 DPU GEMM before consuming its output."""
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.intel_npu_dpu_wait"))


def gemm_rows(activation: Buffer, weights: Buffer, output: Buffer, rows):
    """Overwrite output[:rows] with activation[:rows] @ weights.T.

    Buffers have static, dense FP16 shapes. The runtime row count is clamped
    to [0, activation.shape[0]]; the remaining output rows are untouched.
    """
    buffers = (activation, weights, output)
    if any(len(buffer.shape) != 2 or buffer.dtype != "float16" for buffer in buffers):
        raise ValueError("intel_npu: gemm_rows requires 2-D FP16 buffers")
    if any(buffer.scope() not in ("local", "shared", "shared.dyn") for buffer in buffers):
        raise ValueError("intel_npu: gemm_rows operands must live in CMX")
    capacity, width = (int(dim) for dim in activation.shape)
    columns, weight_width = (int(dim) for dim in weights.shape)
    if weight_width != width or [int(dim) for dim in output.shape] != [capacity, columns]:
        raise ValueError("intel_npu: gemm_rows requires A[M,K], B[N,K], output[M,N]")
    for buffer in buffers:
        if buffer.strides and [int(dim) for dim in buffer.strides] != [int(buffer.shape[1]), 1]:
            raise ValueError("intel_npu: gemm_rows operands must be contiguous")
    if (
        width % fact("dpu_z_align")
        or columns % fact("dpu_z_align")
        or min(capacity, width, columns) < 1
        or max(capacity, width, columns) > fact("dpu_max_workload_dim")
    ):
        raise ValueError("intel_npu: gemm_rows has unsupported workload dimensions")
    live = tirx.max(0, tirx.min(capacity, rows))
    return tirx.call_intrin(
        "handle", tirx.op.Op.get("tl.intel_npu_dpu_matmul"), _addr(output), _addr(activation), _addr(weights), live, columns, width, 0, -1
    )
