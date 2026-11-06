from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

# Source-only: emits the SHAVE C++ and a `kernel_entry`; an external runner
# compiles it with moviCompile. No in-process device compile exists.
build_source = global_func_device_codegen("target.build.tilelang_intel_npu_without_compile")
