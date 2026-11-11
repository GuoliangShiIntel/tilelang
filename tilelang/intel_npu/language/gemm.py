"""Bounded runtime-row FP16 GEMM for Intel NPU CMX buffers."""

from __future__ import annotations

from tvm import tirx
from tvm.tirx import Buffer

from ..hw_facts import fact
from .quant import _addr

__all__ = ["gemm_rows"]


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
