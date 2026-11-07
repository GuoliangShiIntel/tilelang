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
import tilelang.intel_npu.language as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
from kernels import make_gemm, make_gemm_ksplit  # noqa: E402


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except RuntimeError:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")


def _source(func) -> str:
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


def test_dpu_output_points_limit_refused():
    # 128 x 128 = 16384 output points: over the NPU5 DPU's 8192 per workload.
    with pytest.raises(ValueError, match="16384 points exceeds the DPU's 8192"):
        _source(make_gemm(128, 128, 256, BM=128, BN=128, tiles=1, shaves=1))
    # 64 x 128 = 8192 is the limit and is accepted.
    assert "ctx.dpu_matmul(" in _source(make_gemm(64, 128, 256, BM=64, BN=128, tiles=1, shaves=1))


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

    func = make_gemm(64, 64, 32768, BM=64, BN=64, tiles=1, shaves=2)  # A tile 64 x 32768 fp16 = 4 MiB per SHAVE
    big = Target({"kind": "intel_npu", "arch": "npu5", "cmx_bytes": 64 * 1024 * 1024})
    lib = host_sim.build(tilelang.lower(func, target=big).kernel_source, "gemm_too_big")
    a = np.zeros((64, 32768), np.float16)
    b = np.zeros((64, 32768), np.float16)
    c = np.zeros((64, 64), np.float16)
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
