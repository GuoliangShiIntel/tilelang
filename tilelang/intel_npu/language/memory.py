"""CMX allocation in the Intel NPU dialect.

`alloc_shared` is one tile-wide CMX buffer (the same address on both SHAVEs of a tile and the
same offset on every tile); `alloc_local` is one buffer per SHAVE. The NPU has no "dynamic
shared memory", so the scope is plain `shared`, not the GPU dialect's `shared.dyn` default.
"""

from __future__ import annotations

import importlib

# `tilelang.language.allocate` the attribute is a function; the module needs importlib.
_allocate = importlib.import_module("tilelang.language.allocate")
alloc_local = _allocate.alloc_local

__all__ = ["alloc_shared", "alloc_local"]


def alloc_shared(shape, dtype, scope: str = "shared"):
    return _allocate.alloc_shared(shape, dtype, scope=scope)
