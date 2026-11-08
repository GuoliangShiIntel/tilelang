"""Lowering helper shared by the host-sim and device harnesses.

SplitHostDevice may reorder the device kernel's parameters relative to the
prim_func signature, so a harness must bind arrays by the device order, which
is read from the lowered device module.
"""

from __future__ import annotations

from dataclasses import dataclass

import tilelang
from tvm import tirx


@dataclass
class Lowered:
    source: str
    param_names: list[str]  # device kernel order; each is the prim_func buffer name


def lower(func, target=None) -> Lowered:
    artifact = tilelang.lower(func, target=target or "intel_npu")
    funcs = [f for _, f in artifact.device_mod.functions.items() if isinstance(f, tirx.PrimFunc)]
    assert len(funcs) == 1, f"expected one device function, got {len(funcs)}"
    dev = funcs[0]
    names = []
    for v in dev.params:
        buf = dev.buffer_map.get(v)
        name = buf.name if buf is not None else v.name
        names.append(name.removesuffix("_handle"))
    return Lowered(source=artifact.kernel_source, param_names=names)


def order_arrays(lowered: Lowered, by_name: dict) -> list:
    missing = [n for n in lowered.param_names if n not in by_name]
    if missing:
        raise KeyError(f"no array for device parameter(s) {missing}; have {sorted(by_name)}")
    return [by_name[n] for n in lowered.param_names]
