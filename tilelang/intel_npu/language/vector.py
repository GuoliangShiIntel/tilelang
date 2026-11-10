"""Typed SHAVE vector operations for the Intel NPU dialect."""

from tvm import tirx

__all__ = ["axpy"]


def axpy(output, source, weight, output_row=0, source_row=0):
    """Accumulate one fp16 source row into an fp32 CMX row with an fp32 weight."""
    if output.dtype != "float32" or source.dtype != "float16":
        raise ValueError("intel_npu: axpy requires fp32 output and fp16 source")
    if len(output.shape) != 2 or len(source.shape) != 2 or int(output.shape[1]) != int(source.shape[1]):
        raise ValueError("intel_npu: axpy requires equal-width two-dimensional rows")
    if any(buffer.scope() not in ("local", "shared", "shared.dyn") for buffer in (output, source)):
        raise ValueError("intel_npu: axpy operands must live in CMX")
    if output.data.same_as(source.data):
        raise ValueError("intel_npu: axpy operands must not alias")
    destination = tirx.address_of(tirx.BufferLoad(output, [output_row, 0]))
    origin = tirx.address_of(tirx.BufferLoad(source, [source_row, 0]))
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.intel_npu_axpy"), destination, origin,
                           tirx.Cast("float32", weight), int(output.shape[1]))