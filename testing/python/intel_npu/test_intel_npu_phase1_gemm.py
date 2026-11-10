"""Phase 1 of the Intel NPU backend: fp16 GEMM lowered to DMA + DPU intrinsics,
checked in the host simulation against numpy. Hardware-free (needs clang++).

Acceptance (plan Phase 1) with negative controls:
  * generated source contains the DMA/DPU runtime calls with the right byte
    widths, row counts and strides;
  * the host sim reproduces numpy's result (rel < 2e-3) for a shape where the
    blocks do not divide evenly over the workers (the dealing guard is live);
  * a perturbed input makes the comparison FAIL (the check can fail);
  * the K-split form (accumulating gemm) is refused by name;
  * a CMX over-allocation is reported through the status record.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang
import tvm
import tilelang.intel_npu.language as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import make_gemm, make_gemm_ksplit  # noqa: E402


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except RuntimeError:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")


def _source(func) -> str:
    # The raw lowering: workload planning (Phase 3b, test_intel_npu_phase3b_cost.py) is off here.
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return tilelang.lower(func, target="intel_npu").kernel_source


def test_gemm_source_has_dma_and_dpu_calls():
    src = _source(make_gemm(256, 256, 512, BM=64, BN=64, tiles=2, shaves=2))
    assert "npu::Ddr<half> A, npu::Ddr<half> B, npu::Ddr<half> C" in src
    assert "ctx.alloc<half>(32768);" in src  # A tile: 64 x 512
    # A tile: 64 rows of 512 halves = 1024 B, source row stride 1024 B
    assert ", 1024, 64, 1024, 1024);" in src
    # default workload shape: runtime's own spatial split (0) and stencil (-1)
    assert "ctx.dpu_matmul(" in src and ", 64, 64, 512, 0, -1);" in src
    # C block: 64 rows of 64 halves = 128 B into rows of 256 halves = 512 B
    assert "ctx.dma_store(" in src and ", 128, 64, 512, 128);" in src


@needs_cxx
@pytest.mark.parametrize("count", [0, 3, 8])
@pytest.mark.parametrize("shared", [False, True])
def test_dynamic_dma_rows_preserve_unwritten_output(count, shared):
    @T.prim_func
    def row_copy(A: T.Tensor((8, 64), T.float16), Limit: T.Tensor((1,), T.int32),
                 B: T.Tensor((9, 64), T.float16)):
        with T.Kernel(1, threads=1):
            if shared:
                scratch = T.alloc_shared((8, 64), T.float16)
            else:
                scratch = T.alloc_local((8, 64), T.float16)
            count_cmx = T.alloc_local((1,), T.int32)
            T.copy(Limit, count_cmx)
            rows = T.max(0, T.min(8, count_cmx[0]))
            T.copy(A[0:rows, 0:64], scratch[0:rows, 0:64])
            T.copy(scratch[0:rows, 0:64], B[1:1 + rows, 0:64])

    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        low = lowering.lower(row_copy)
    source = np.arange(512, dtype=np.float16).reshape(8, 64)
    output = np.full((9, 64), -123, np.float16)
    arrays = lowering.order_arrays(low, {"A": source, "Limit": np.array([count], np.int32), "B": output})
    lib = host_sim.build(low.source, "dynamic_dma_rows")
    host_sim.check_status(host_sim.run(lib, arrays, tiles=1, shaves=1))
    expected = np.full_like(output, -123)
    expected[1:1 + count] = source[:count]
    np.testing.assert_array_equal(output, expected)


def test_dynamic_dma_width_is_still_refused():
    @T.prim_func
    def width_copy(A: T.Tensor((8, 64), T.float16), Limit: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1):
            scratch = T.alloc_local((8, 64), T.float16)
            T.copy(A[0:8, 0:Limit[0]], scratch[0:8, 0:Limit[0]])

    with pytest.raises(ValueError, match="non-constant extent"):
        _source(width_copy)


def _run_gemm(M, N, K, BM, BN, tiles, shaves, perturb=False):
    func = make_gemm(M, N, K, BM=BM, BN=BN, tiles=tiles, shaves=shaves)
    lib = host_sim.build(_source(func), f"gemm_{M}x{N}x{K}")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((N, K)).astype(np.float16)
    c = np.zeros((M, N), np.float16)
    status = host_sim.run(lib, [a, b, c], tiles, shaves)
    host_sim.check_status(status)
    ref = a.astype(np.float32) @ b.astype(np.float32).T
    if perturb:
        c[3, 5] += np.float16(1.0)
    err = np.abs(c.astype(np.float32) - ref).max() / np.abs(ref).max()
    return err


@needs_cxx
def test_gemm_host_sim_matches_numpy():
    # 4 x 2 = 8 blocks over 4 workers (even) and 6 x 2 = 12 blocks over 4 workers.
    assert _run_gemm(256, 128, 512, 64, 64, tiles=2, shaves=2) < 2e-3
    assert _run_gemm(384, 128, 256, 64, 64, tiles=2, shaves=2) < 2e-3


@needs_cxx
def test_gemm_host_sim_uneven_dealing():
    # 3 x 3 = 9 blocks over 4 workers: the `blk < nblk` guard must be live.
    assert _run_gemm(192, 192, 128, 64, 64, tiles=2, shaves=2) < 2e-3


@needs_cxx
def test_negative_control_perturbed_output_fails():
    assert _run_gemm(256, 128, 512, 64, 64, tiles=2, shaves=2, perturb=True) > 1e-2


def test_ksplit_refused_by_name():
    with pytest.raises(ValueError, match="clear_accum=False.*DPU overwrites"):
        _source(make_gemm_ksplit(64, 64, 256))


def test_dpu_workload_dim_limit_refused():
    # A dimension above 8192 is refused (dpu.md 4.6); the output tile's M x N PRODUCT is not bounded.
    # dpu.md 13's "max output points per operation 8192" was read as that product until
    # device/run_dpu_points_probe.py ran 16384- and 65536-value matmuls correctly on PTL.
    with pytest.raises(ValueError, match="has a dimension above the DPU's 8192"):
        _source(make_gemm(16384, 64, 256, BM=16384, BN=64, tiles=1, shaves=1))
    # the shape that used to be refused, and the one npu_dsl's prefill scan issues
    assert "ctx.dpu_matmul(" in _source(make_gemm(128, 128, 256, BM=128, BN=128, tiles=1, shaves=1))


def test_dpu_z_alignment_refused():
    with pytest.raises(ValueError, match="must be multiples of 16"):
        _source(make_gemm(64, 72, 256, BM=64, BN=72, tiles=1, shaves=1))  # N = 72
    with pytest.raises(ValueError, match="must be multiples of 16"):
        _source(make_gemm(64, 64, 200, BM=64, BN=64, tiles=1, shaves=1))  # K = 200


@needs_cxx
def test_cmx_exhaustion_reported_in_status():
    # The compile-time budget check (Phase 2) refuses this for the real target; lower it for a
    # target that claims 64 MiB of CMX so the RUNTIME allocator is the one that reports the failure.
    from tvm.target import Target

    func = make_gemm(256, 64, 8192, BM=256, BN=64, tiles=1, shaves=2)  # A tile 256 x 8192 fp16 = 4 MiB per SHAVE
    big = Target({"kind": "intel_npu", "arch": "npu5", "cmx_bytes": 64 * 1024 * 1024})
    lib = host_sim.build(tilelang.lower(func, target=big).kernel_source, "gemm_too_big")
    a = np.zeros((256, 8192), np.float16)
    b = np.zeros((64, 8192), np.float16)
    c = np.zeros((256, 64), np.float16)
    status = host_sim.run(lib, [a, b, c], tiles=1, shaves=2)
    with pytest.raises(RuntimeError, match="CMX_EXHAUSTED"):
        host_sim.check_status(status)


def test_scalar_kernel_still_lowers():
    @T.prim_func
    def main(A: T.Tensor((8, 16), T.float16)):
        with T.Kernel(2, threads=2) as tile:
            shave = T.get_thread_binding()
            for i in T.serial(4):
                A[tile * 2 + shave, i] = T.cast(0, T.float16)

    src = _source(main)
    assert "A[" in src and "npu::Ddr<half> A" in src


if __name__ == "__main__":
    tilelang.testing.main()
