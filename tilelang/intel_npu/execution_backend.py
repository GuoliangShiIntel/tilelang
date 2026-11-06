from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec

# No shared execution backend can launch on the NPU. `source_only` declares
# that compilation stops at generated source (`tilelang.lower(...).kernel_source`);
# the device runner under testing/python/intel_npu/device/ compiles and launches it.
EXECUTION_BACKENDS = (ExecutionBackendSpec("source_only"),)
