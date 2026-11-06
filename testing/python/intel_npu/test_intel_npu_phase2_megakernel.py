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
from tilelang import tvm
from tvm.target import Target

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import make_gemm, make_gemm_two_phase  # noqa: E402

NO_DERIVE = {"tl.intel_npu.derive_global_sync": False}


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


def _run_two_phase(with_sync: bool, seed: int = 0, derive: bool = True):
    func = make_gemm_two_phase(M, N, K, tiles=TILES, shaves=SHAVES, with_sync=with_sync)
    with tvm.transform.PassContext(config={} if derive else NO_DERIVE):
        low = lowering.lower(func)
    lib = host_sim.build(low.source, f"two_phase_{'sync' if with_sync else 'nosync'}_{'d' if derive else 'nd'}")
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
    # With derivation off and no hand-written barrier: at least one of several runs reads unfinished partials.
    errs = [_run_two_phase(with_sync=False, seed=s, derive=False) for s in range(5)]
    assert max(errs) > 1e-2, f"the no-barrier kernel never failed: {errs}"


@needs_cxx
def test_derived_barrier_makes_unsynced_source_correct():
    assert _run_two_phase(with_sync=False, derive=True) < 4e-3


def test_derived_sync_identical_to_hand_written():
    hand = lowering.lower(make_gemm_two_phase(M, N, K, tiles=TILES, shaves=SHAVES, with_sync=True)).source
    auto = lowering.lower(make_gemm_two_phase(M, N, K, tiles=TILES, shaves=SHAVES, with_sync=False)).source
    assert hand == auto
    assert hand.count("ctx.sync_grid();") == 1
    with tvm.transform.PassContext(config=NO_DERIVE):
        off = lowering.lower(make_gemm_two_phase(M, N, K, tiles=TILES, shaves=SHAVES, with_sync=False)).source
    assert "ctx.sync_grid()" not in off  # the kill switch is live


def test_no_cross_phase_dependency_gets_no_barrier():
    # Phase A writes P from A, B; phase B writes C from A, B: no DDR buffer flows between them.
    @T.prim_func
    def main(A: T.Tensor((64, 128), T.float16), B: T.Tensor((64, 128), T.float16),
             P: T.Tensor((64, 64), T.float16), C: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            A_cmx = T.alloc_local((64, 128), T.float16)
            B_cmx = T.alloc_local((64, 128), T.float16)
            O_cmx = T.alloc_local((64, 64), T.float16)
            if shave == 0:
                T.copy(A, A_cmx)
                T.copy(B, B_cmx)
                T.gemm(A_cmx, B_cmx, O_cmx, transpose_B=True, clear_accum=True)
                T.copy(O_cmx, P)
            if shave == 1:
                T.copy(A, A_cmx)
                T.copy(B, B_cmx)
                T.gemm(A_cmx, B_cmx, O_cmx, transpose_B=True, clear_accum=True)
                T.copy(O_cmx, C)

    src = lowering.lower(main).source
    assert "ctx.sync_grid()" not in src
    assert src.count("ctx.dpu_matmul(") == 2


def test_war_dependency_gets_barrier():
    # Phase A reads X (DMA load); phase B overwrites X (DMA store): a WAR hazard across SHAVEs.
    @T.prim_func
    def main(X: T.Tensor((64, 64), T.float16), Y: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            X_cmx = T.alloc_local((64, 64), T.float16)
            Y_cmx = T.alloc_local((64, 64), T.float16)
            if shave == 0:
                T.copy(X, X_cmx)
                T.copy(X_cmx, Y)
            if shave == 1:
                T.copy(Y, Y_cmx)
                T.copy(Y_cmx, X)

    src = lowering.lower(main).source
    assert src.count("ctx.sync_grid();") == 1
    # The barrier sits between the two phases.
    first_store = src.index("ctx.dma_store(")
    assert src.index("ctx.sync_grid();") > first_store


def test_author_barrier_is_kept_not_duplicated():
    @T.prim_func
    def main(X: T.Tensor((64, 64), T.float16), Y: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            X_cmx = T.alloc_local((64, 64), T.float16)
            if shave == 0:
                T.copy(X, X_cmx)
                T.copy(X_cmx, Y)
            T.sync_grid()
            if shave == 1:
                T.copy(Y, X_cmx)
                T.copy(X_cmx, X)

    assert lowering.lower(main).source.count("ctx.sync_grid();") == 1


@pytest.mark.parametrize("derive", [True, False])
def test_sync_grid_under_worker_condition_is_refused(derive):
    # Only SHAVE 0 would reach the barrier: a hang. Refused with or without derivation.
    @T.prim_func
    def main(X: T.Tensor((64, 64), T.float16), Y: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            X_cmx = T.alloc_local((64, 64), T.float16)
            if shave == 0:
                T.copy(X, X_cmx)
                T.sync_grid()
                T.copy(X_cmx, Y)

    with tvm.transform.PassContext(config={} if derive else NO_DERIVE):
        with pytest.raises(ValueError, match="depends on the tile or SHAVE index"):
            lowering.lower(main)


def test_sync_grid_under_derived_worker_value_is_refused():
    # The condition reads `worker`, a value computed from the launch coordinates.
    @T.prim_func
    def main(X: T.Tensor((64, 64), T.float16), Y: T.Tensor((64, 64), T.float16)):
        with T.Kernel(2, threads=2) as tile:
            shave = T.get_thread_binding()
            worker = tile * 2 + shave
            X_cmx = T.alloc_local((64, 64), T.float16)
            for w in T.serial(worker + 1):
                T.copy(X, X_cmx)
                T.sync_grid()
                T.copy(X_cmx, Y)

    with pytest.raises(ValueError, match="depends on the tile or SHAVE index"):
        lowering.lower(main)


def test_sync_grid_under_uniform_condition_is_allowed():
    @T.prim_func
    def main(X: T.Tensor((64, 64), T.float16), Y: T.Tensor((64, 64), T.float16), n: T.int32):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            X_cmx = T.alloc_local((64, 64), T.float16)
            for i in T.serial(2):
                if shave == 0:
                    T.copy(X, X_cmx)
                    T.copy(X_cmx, Y)
                T.sync_grid()

    assert lowering.lower(main).source.count("ctx.sync_grid();") == 1


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
