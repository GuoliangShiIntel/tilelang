"""Phase 0 of the Intel NPU backend: registration, target resolution, launch
validation and source-only codegen. Hardware-free.

Acceptance (docs/developer_guide/intel_npu_backend_plan.md, Phase 0), each
with its negative control:
  * the package imports and registers the ``intel_npu`` backend;
  * ``intel_npu`` / ``ptl`` / ``npu5`` resolve to the target kind; another
    backend's spelling returns None from the normalizer;
  * lowering a scalar kernel returns source with ``kernel_entry`` and the
    tile/SHAVE launch bindings; a kernel lowered for ``c`` has neither;
  * a 2-D grid is refused naming ``intel_npu`` both at the dialect and, for
    a common-dialect kernel, in the pipeline;
  * a tile op is refused by name until Phase 1.
"""

import pytest

import tilelang
import tilelang.intel_npu.language as T
import tilelang.language as TC  # common dialect: bypasses the intel_npu T.Kernel check
from tilelang import tvm
from tilelang.backend import get_backend
from tilelang.backend.target import determine_target
from tilelang.intel_npu.target import normalize_intel_npu_target
from tvm.target import Target


def _scalar_kernel():
    @T.prim_func
    def main(A: T.Tensor((8, 16), T.float16)):
        with T.Kernel(2, threads=2) as tile:
            shave = T.get_thread_binding()
            for i in T.serial(4):
                A[tile * 2 + shave, i] = T.cast(0, T.float16)

    return main


def test_backend_registered():
    backend = get_backend("intel_npu")
    assert backend.target_kinds == ("intel_npu",)
    assert tuple(spec.name for spec in backend.execution_backends) == ("source_only",)


@pytest.mark.parametrize("spelling", ["intel_npu", "ptl", "npu5"])
def test_target_aliases_resolve(spelling):
    target = determine_target(spelling, return_object=True)
    assert isinstance(target, Target)
    assert target.kind.name == "intel_npu"
    assert str(target.attrs["arch"]) == "npu5"
    assert int(target.attrs["shaves_per_tile"]) == 2


def test_explicit_target_dict_with_attrs():
    target = Target({"kind": "intel_npu", "arch": "npu5", "num_tiles": 1})
    assert target.kind.name == "intel_npu"
    assert int(target.attrs["num_tiles"]) == 1
    # The usable CMX workspace (1439 KiB), not the physical 1536 KiB.
    assert int(target.attrs["cmx_bytes"]) == 1473536
    assert int(target.attrs["barrier_pool_first"]) == 8 and int(target.attrs["barrier_pool_count"]) == 40
    assert int(target.attrs["dpu_max_output_points"]) == 8192


@pytest.mark.parametrize("other", ["cuda", "c", "ascend", "intel_npu_x", ""])
def test_normalizer_ignores_other_backends(other):
    assert normalize_intel_npu_target(other) is None


def test_lower_scalar_kernel_emits_kernel_entry():
    artifact = tilelang.lower(_scalar_kernel(), target="intel_npu")
    src = artifact.kernel_source
    assert "#include <tl_templates/intel_npu/npu_rt.hpp>" in src
    assert "void kernel_entry(int numParams, uint64_t* params, const npu::Environment& env)" in src
    assert "npu::Ctx ctx(env, numParams, params, 1);" in src
    assert "ctx.tile_id()" in src
    assert "ctx.shave_id()" in src
    assert "npu::Ddr<half>(params[0])" in src
    assert src.count("kernel_entry(") == 1


def test_negative_control_cpu_source_has_no_npu_entry():
    # The same assertions must be able to fail: the CPU backend emits neither symbol.
    import tilelang.cpu.language as TCPU

    @TCPU.prim_func
    def main(A: T.Tensor((8,), T.float16)):
        with TCPU.Kernel(2) as bx:
            A[bx] = T.cast(0, T.float16)

    src = tilelang.lower(main, target="c").kernel_source
    assert "kernel_entry" not in src
    assert "ctx.tile_id()" not in src


def test_dialect_refuses_multi_dim_grid():
    with pytest.raises(ValueError, match="intel_npu"):

        @T.prim_func
        def main(A: T.Tensor((4, 4), T.float16)):
            with T.Kernel(2, 2) as (bx, by):
                A[bx, by] = T.cast(0, T.float16)


def test_pipeline_refuses_multi_dim_grid_from_common_dialect():
    @TC.prim_func
    def main(A: T.Tensor((4, 4), T.float16)):
        with TC.Kernel(2, 2, threads=2) as (bx, by):
            A[bx, by] = T.cast(0, T.float16)

    with pytest.raises(Exception, match="intel_npu.*blockIdx.y"):
        tilelang.lower(main, target="intel_npu")


def test_tile_op_refused_by_name():
    @T.prim_func
    def main(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=2):
            A_cmx = T.alloc_shared((64, 64), T.float16)
            T.copy(A, A_cmx)
            T.fill(A_cmx, 0)
            T.copy(A_cmx, B)

    with pytest.raises(ValueError, match="tl.tileop.fill"):
        tilelang.lower(main, target="intel_npu")


def test_device_compile_mode_is_absent():
    # Only the source-only entry exists; asking for a compiled device module must fail at the codegen boundary.
    from tilelang.backend.module import create_backend_context
    from tilelang.engine.lower import lower_with_context

    context = create_backend_context("intel_npu")
    with pytest.raises(ValueError, match="does not support lowering with compilation"):
        lower_with_context(_scalar_kernel(), context, enable_device_compile=True)


if __name__ == "__main__":
    tilelang.testing.main()
