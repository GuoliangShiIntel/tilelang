"""Phase 2 of the Intel NPU backend: megakernel primitives in the host sim.

  * T.sync_grid() lowers to ctx.sync_grid(); T.sync_threads() to ctx.sync_tile();
  * a two-phase kernel (split-K partials -> sync_grid -> combine) matches numpy;
  * negative: the same kernel without the barrier is a race; in the sim the
    combine reads partials that other threads have not written (zeros) often
    enough to fail, so the barrier is shown to be load-bearing;
  * CheckIntelNpuBudgets refuses an over-budget kernel at compile time, naming
    the buffers; a fitting kernel passes; a different cmx_bytes flips the verdict.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang
import tilelang.intel_npu.language as T
from tvm.target import Target

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import make_gemm, make_gemm_two_phase  # noqa: E402


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except RuntimeError:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")

M, N, K = 256, 256, 512
TILES, SHAVES = 2, 2


def test_sync_grid_and_sync_threads_lower():
    @T.prim_func
    def main(A: T.Tensor((8,), T.float16)):
        with T.Kernel(2, threads=2) as tile:
            shave = T.get_thread_binding()
            A[tile * 2 + shave] = T.cast(1, T.float16)
            T.sync_threads()
            T.sync_grid()
            A[tile * 2 + shave] = A[(tile * 2 + shave + 1) % 4]

    src = lowering.lower(main).source
    assert "ctx.sync_tile();" in src
    assert "ctx.sync_grid();" in src
    assert src.index("ctx.sync_tile();") < src.index("ctx.sync_grid();")


def _run_two_phase(with_sync: bool, seed: int = 0):
    low = lowering.lower(make_gemm_two_phase(M, N, K, tiles=TILES, shaves=SHAVES, with_sync=with_sync))
    lib = host_sim.build(low.source, f"two_phase_{'sync' if with_sync else 'nosync'}")
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((N, K)).astype(np.float16)
    p = np.zeros((2, M, N), np.float16)
    c = np.zeros((M, N), np.float16)
    status = host_sim.run(lib, lowering.order_arrays(low, {"A": a, "B": b, "P": p, "C": c}), TILES, SHAVES)
    host_sim.check_status(status)
    ref = a.astype(np.float32) @ b.astype(np.float32).T
    return np.abs(c.astype(np.float32) - ref).max() / np.abs(ref).max()


@needs_cxx
def test_two_phase_matches_numpy():
    assert _run_two_phase(with_sync=True) < 4e-3  # two fp16 roundings (partials) plus the sum


@needs_cxx
def test_negative_without_sync_grid_races():
    # Without the barrier at least one of several runs must read unfinished partials.
    errs = [_run_two_phase(with_sync=False, seed=s) for s in range(5)]
    assert max(errs) > 1e-2, f"the no-barrier kernel never failed: {errs}"


def test_budget_refuses_oversized_kernel():
    func = make_gemm(2048, 64, 1024, BM=2048, BN=64, tiles=1, shaves=2)  # 4 MiB A tile per SHAVE
    with pytest.raises(ValueError, match=r"(?s)CMX budget exceeded.*A_cmx: 4194304 B per SHAVE"):
        tilelang.lower(func, target="intel_npu")


def test_budget_follows_target_attr():
    func = make_gemm(256, 256, 512, BM=64, BN=64, tiles=2, shaves=2)  # 2 x 64 KiB + 8 KiB per SHAVE
    tilelang.lower(func, target="intel_npu")  # fits 1.5 MiB
    small = Target({"kind": "intel_npu", "arch": "npu5", "cmx_bytes": 256 * 1024})
    with pytest.raises(ValueError, match="CMX budget exceeded"):
        tilelang.lower(func, target=small)


if __name__ == "__main__":
    tilelang.testing.main()
