"""DDR tensors at run-time addresses (the layer table of a single-launch model).

`ddr_view(addr, shape, dtype)` declares a global buffer over `npu::Ddr<T>(addr)`, where `addr` is an int64 read
from a table the host filled with device addresses (`Session.address`). Every tile op that takes a DDR tensor
(`T.copy`, the streamed fills of `T.Pipelined`) takes the view like a kernel parameter. The view is bound where it is
written, so a view declared inside the layer loop names that layer's weights, and LowerIntelNpuPipelined refuses to
hoist a ring over a loop that binds its fill's source (the fill ahead would read a view not yet bound).
"""

from __future__ import annotations

from tvm import tirx
from tvm.ir import PointerType, PrimType
from tvm.script import tirx as B

__all__ = ["ddr_view"]


def ddr_view(addr, shape, dtype):
    dt = str(dtype) if not isinstance(dtype, str) else dtype
    dt = {"float16": "float16", "half": "float16"}.get(dt, dt)
    call = tirx.call_intrin("handle", tirx.op.Op.get("tl.intel_npu_ddr_ptr"), tirx.Cast("int64", addr), tirx.const(0, dt))
    ptr = B.bind(call, PointerType(PrimType(dt), "global"))
    return B.decl_buffer(shape, dt, data=ptr, scope="global")
