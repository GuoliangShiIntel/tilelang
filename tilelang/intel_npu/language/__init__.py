"""Intel NPU language dialect: common TileLang plus NPU extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

from .kernel import *  # noqa: F401,F403
from .kernel import __all__ as _KERNEL_ALL
from .memory import *  # noqa: F401,F403  (alloc_shared with scope "shared" overrides the common one)
from .memory import __all__ as _MEMORY_ALL

__tilelang_dialect__ = "intel_npu"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_KERNEL_ALL, *_MEMORY_ALL)))

del _COMMON_ALL, _KERNEL_ALL, _MEMORY_ALL
