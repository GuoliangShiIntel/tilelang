"""Phase 3b: the multi-tile ping-pong flow from the author's loops (plan 3.2; README sections 1-3).

Hardware-free: these tests read the generated source. The executable acceptance is the device
golden `device/run_ring_flow_device.py` (PASS 2026-11-06, see device/evidence/INDEX.md); the host
simulation does not implement the ring yet.

Acceptance with negative controls:
  * the flow source lowers to one ring: shared slots, SEGMENTED resident fill split over the
    SHAVEs with a tile sync, multicast fills, READY/FREE ids from the job's pool, submit-ahead
    with a parity double-buffered output, and the source says what was derived;
  * generality: a source with the roles swapped (streamed operand second), other names and
    loops lowers with zero compiler edits;
  * refused by name: ring depth != SHAVEs per tile; a streamed fill that depends on the tile;
    a one-tile launch; more rings than the barrier pool holds.
"""

import re
import sys
from pathlib import Path

import pytest
import numpy as np

import tilelang
import tilelang.intel_npu.language as T
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import make_ring_flow, make_ring_flow_swapped  # noqa: E402
import host_sim  # noqa: E402
import lowering  # noqa: E402


def _src(func):
    # Workload planning off: these tests read the structure, not VPUNN's choice.
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return tilelang.lower(func, target="intel_npu").kernel_source


def test_flow_lowers_to_one_ring():
    src = _src(make_ring_flow(2048, 768, 2048, BM=32, BN=256, tiles=3))
    assert "half* __restrict W_cmx = ctx.alloc_shared<half>(524288);" in src  # resident, one per tile
    assert "half* __restrict X_cmx = ctx.alloc_shared<half>(131072);" in src  # 2 slots x 32 x 2048
    assert "half* __restrict Y_cmx = ctx.alloc<half>(16384);" in src  # 2 x 32 x 256, per SHAVE
    # resident fill split over the SHAVEs (128 of 256 rows each), then the tile sync
    assert re.search(r"ctx\.dma_load\(\(&\(W_cmx\[\(tx \* 262144\)\]\)\).*, 4096, 128, 4096, 4096\);\n\s*ctx\.sync_tile\(\);", src)
    assert "npu::Ctx::Ring ring = ctx.ring_begin((&(X_cmx[0])), 131072, 2, 64, 8, 12, -1, 1, 0);" in src  # one stream: halves on both ports
    assert "ctx.ring_fill(ring, 0, (&(X[(tx * 65536)])), 4096, 32, 4096);" in src  # prologue: chunk = SHAVE index
    assert src.count("ctx.ring_dpu_submit(") == 2 and "ctx.ring_dpu_wait(ring, u);" in src
    assert "(&(Y_cmx[(((u + 1) & 1) * 8192)]))" in src  # submit-ahead output parity
    assert "ctx.ring_end(ring);" in src
    assert "READY 8..11, FREE 12..15" in src and "sync_grid 45..47" in src
    assert "W_cmx: SEGMENTED resident" in src and "X_cmx: DUPLICATED (streamed" in src
    assert "ctx.dpu_matmul(" not in src


def test_flow_is_generic_over_roles_and_names():
    src = _src(make_ring_flow_swapped(768, 2048, 2048, BM=256, BN=32, tiles=3))
    assert "rows_cmx: SEGMENTED resident" in src and "wstream: DUPLICATED (streamed" in src
    # the streamed operand is the gemm's second operand here
    assert re.search(
        r"ctx\.ring_dpu_submit\(ring, 0, \(&\(acc\[0\]\)\), \(&\(rows_cmx\[0\]\)\), ctx\.ring_slot<half>\(ring, 0\), 256, 32, 2048", src
    )
    assert "ctx.ring_begin((&(wstream[0])), 131072, 2, 64, 8, 12, -1, 1, 0);" in src


def _serial_projection(split=False, guarded=False, shave_guard=False, repeated_output=False):
    @T.prim_func
    def projection(X: T.Tensor((128, 64), T.float16), W: T.Tensor((64, 64), T.float16), Y: T.Tensor((64, 64), T.float16)):
        with T.Kernel(2, threads=2) as tile:
            shave = T.get_thread_binding()
            weights = T.alloc_shared((32, 64), T.float16)
            activation = T.alloc_local((16, 64), T.float16)
            result = T.alloc_local((16, 32), T.float16)
            T.copy(W[tile * 32, 0], weights)
            for block in T.serial(2):
                if (not guarded or tile == 0) and (not shave_guard or shave == 0):
                    T.copy(X[(tile * 64 if split else 0) + (block * 2 + shave) * 16, 0], activation)
                    T.gemm(activation, weights, result, transpose_B=True, clear_accum=True)
                    T.copy(result, Y[(shave if repeated_output else block * 2 + shave) * 16, tile * 32])

    return projection


@pytest.mark.parametrize("split,guarded", [(False, False), (True, False), (False, True)])
def test_serial_activation_multicast_preserves_worker_ownership(split, guarded):
    with tvm.transform.PassContext(
        config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.multicast": True, "tl.intel_npu.prefetch": False}
    ):
        src = tilelang.lower(_serial_projection(split, guarded), target="intel_npu").kernel_source
    assert ("ctx.dma_load_multicast(" in src) == (not split and not guarded)
    assert "ctx.ring_begin(" not in src
    assert "ctx.alloc_shared<half>(2048)" in src
    assert "ctx.alloc<half>(1024)" in src


def test_serial_multicast_is_opt_in_and_skips_shave_branches():
    assert "ctx.dma_load_multicast(" not in _src(_serial_projection())
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.multicast": True}):
        src = tilelang.lower(_serial_projection(shave_guard=True), target="intel_npu").kernel_source
    assert "ctx.dma_load_multicast(" not in src


def test_serial_overlap_derives_a_ring_without_changing_token_ownership():
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.multicast_overlap": True}):
        src = tilelang.lower(_serial_projection(), target="intel_npu").kernel_source
    assert "ctx.ring_begin(" in src
    assert "ctx.ring_dpu_submit(" in src
    assert "ctx.alloc_shared<half>(2048)" in src
    assert "ctx.dma_load_multicast(" not in src
    assert "ownership-preserving" in src
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.multicast_overlap": True}):
        split_src = tilelang.lower(_serial_projection(split=True), target="intel_npu").kernel_source
        guarded_src = tilelang.lower(_serial_projection(guarded=True), target="intel_npu").kernel_source
        repeated_src = tilelang.lower(_serial_projection(repeated_output=True), target="intel_npu").kernel_source
    assert "ctx.ring_begin(" not in split_src
    assert "ctx.ring_begin(" not in guarded_src
    assert "ctx.ring_begin(" not in repeated_src


@pytest.mark.parametrize("split,guarded", [(False, False), (True, False), (False, True)])
def test_serial_activation_multicast_numerical(split, guarded):
    try:
        host_sim.find_clangxx()
    except RuntimeError:
        pytest.skip("no host clang++ for the simulation")
    with tvm.transform.PassContext(
        config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.multicast": True, "tl.intel_npu.prefetch": False}
    ):
        low = lowering.lower(_serial_projection(split, guarded))
    rng = np.random.default_rng(42)
    activation = rng.normal(size=(128, 64)).astype(np.float16)
    weights = rng.normal(size=(64, 64)).astype(np.float16)
    output = np.zeros((64, 64), np.float16)
    lib = host_sim.build(low.source, "serial_multicast")
    arrays = lowering.order_arrays(low, {"X": activation, "W": weights, "Y": output})
    host_sim.check_status(host_sim.run(lib, arrays, tiles=2, shaves=2))
    expected = np.zeros_like(output)
    for tile in range(1 if guarded else 2):
        start = tile * 64 if split else 0
        expected[:, tile * 32 : (tile + 1) * 32] = (
            activation[start : start + 64].astype(np.float32) @ weights[tile * 32 : (tile + 1) * 32].astype(np.float32).T
        ).astype(np.float16)
    np.testing.assert_allclose(output, expected, rtol=2e-3, atol=2e-3)


def test_ring_depth_must_match_shaves():
    with pytest.raises(Exception, match=r"num_stages=3\) on a tile with 2 SHAVEs"):
        _src(make_ring_flow(2048, 768, 2048, BM=32, BN=256, tiles=3, num_stages=3))


def test_tile_split_on_streamed_loop_is_a_segmented_stream():
    # Since M2 (2026-11-07): a streamed fill that depends on the tile is a SPLIT stream (each tile fills its own
    # slot, unicast, the ring's multicast flag off), not a refusal; the resident operand is tile-invariant.
    M, N, K, BM, BN, tiles = 768, 768, 2048, 32, 256, 3

    @T.prim_func
    def split(X: T.Tensor((M, K), T.float16), W: T.Tensor((N, K), T.float16), Y: T.Tensor((M, N), T.float16)):
        with T.Kernel(tiles, threads=2) as tile:
            W_cmx = T.alloc_shared((BN, K), T.float16)
            X_cmx = T.alloc_shared((BM, K), T.float16)
            Y_cmx = T.alloc_local((BM, BN), T.float16)
            T.copy(W[0, 0], W_cmx)
            for c in T.Pipelined(M // BM // tiles, num_stages=2):
                T.copy(X[(tile * (M // BM // tiles) + c) * BM, 0], X_cmx)  # the streamed fill depends on the tile
                T.gemm(X_cmx, W_cmx, Y_cmx, transpose_B=True, clear_accum=True)
                T.copy(Y_cmx, Y[(tile * (M // BM // tiles) + c) * BM, 0])

    src = _src(split)
    assert "X_cmx: SEGMENTED (streamed, each tile fills its own slot)" in src
    assert "W_cmx: tile-invariant resident" in src
    assert "ctx.ring_begin((&(X_cmx[0])), 131072, 2, 8, 8, 12, -1, 0, 0);" in src
    assert "ctx.ring_fill(ring, 0, (&(X[((bx * 524288) + (tx * 65536))])), 4096, 32, 4096);" in src


def test_one_tile_is_refused():
    with pytest.raises(Exception, match="hardware ring needs at least 2 tiles"):
        _src(make_ring_flow(2048, 256, 2048, BM=32, BN=256, tiles=1))


def test_barrier_pool_exhaustion_is_refused():
    M, N, K, BM, BN, tiles = 512, 512, 1024, 32, 256, 2  # 2 tiles: pool [8, 32), 21 usable, 8 per ring

    def make(nrings):
        @T.prim_func
        def many(X: T.Tensor((M, K), T.float16), W: T.Tensor((N, K), T.float16), Y: T.Tensor((M, N), T.float16)):
            with T.Kernel(tiles, threads=2) as tile:
                W_cmx = T.alloc_shared((BN, K), T.float16)
                X_cmx = T.alloc_shared((BM, K), T.float16)
                Y_cmx = T.alloc_local((BM, BN), T.float16)
                T.copy(W[tile * BN, 0], W_cmx)
                for r in T.serial(nrings):
                    for c in T.Pipelined(M // BM, num_stages=2):
                        T.copy(X[c * BM, 0], X_cmx)
                        T.gemm(X_cmx, W_cmx, Y_cmx, transpose_B=True, clear_accum=True)
                        T.copy(Y_cmx, Y[c * BM, tile * BN])

        return many

    # one ring inside a serial loop is one ring_begin: fine
    assert "READY 8..11, FREE 12..15" in _src(make(1))

    @T.prim_func
    def three(X: T.Tensor((M, K), T.float16), W: T.Tensor((N, K), T.float16), Y: T.Tensor((M, N), T.float16)):
        with T.Kernel(tiles, threads=2) as tile:
            W_cmx = T.alloc_shared((BN, K), T.float16)
            X_cmx = T.alloc_shared((BM, K), T.float16)
            Y_cmx = T.alloc_local((BM, BN), T.float16)
            T.copy(W[tile * BN, 0], W_cmx)
            for c in T.Pipelined(M // BM, num_stages=2):
                T.copy(X[c * BM, 0], X_cmx)
                T.gemm(X_cmx, W_cmx, Y_cmx, transpose_B=True, clear_accum=True)
                T.copy(Y_cmx, Y[c * BM, tile * BN])
            for c in T.Pipelined(M // BM, num_stages=2):
                T.copy(X[c * BM, 0], X_cmx)
                T.gemm(X_cmx, W_cmx, Y_cmx, transpose_B=True, clear_accum=True)
                T.copy(Y_cmx, Y[c * BM, tile * BN])
            for c in T.Pipelined(M // BM, num_stages=2):
                T.copy(X[c * BM, 0], X_cmx)
                T.gemm(X_cmx, W_cmx, Y_cmx, transpose_B=True, clear_accum=True)
                T.copy(Y_cmx, Y[c * BM, tile * BN])

    # three rings one after another: each begins after the previous ended, so all three share one block
    src = _src(three)
    assert src.count("READY 8..11, FREE 12..15") == 3 and "READY 16" not in src
    assert "3 rings, 1 live at once, 8 ids by reuse" in src
    assert src.count("ctx.ring_begin((&(X_cmx[0])), 65536, 2, 16, 8, 12, -1, 1, 0);") == 3

    # three operands streamed through ONE loop are three rings live at once: 24 ids do not fit in 21
    @T.prim_func
    def three_at_once(
        X1: T.Tensor((M, K), T.float16),
        X2: T.Tensor((M, K), T.float16),
        X3: T.Tensor((M, K), T.float16),
        W: T.Tensor((N, K), T.float16),
        Y1: T.Tensor((M, N), T.float16),
        Y2: T.Tensor((M, N), T.float16),
        Y3: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(tiles, threads=2) as tile:
            W_cmx = T.alloc_shared((BN, K), T.float16)
            X1_cmx = T.alloc_shared((BM, K), T.float16)
            X2_cmx = T.alloc_shared((BM, K), T.float16)
            X3_cmx = T.alloc_shared((BM, K), T.float16)
            Y1_cmx = T.alloc_local((BM, BN), T.float16)
            Y2_cmx = T.alloc_local((BM, BN), T.float16)
            Y3_cmx = T.alloc_local((BM, BN), T.float16)
            T.copy(W[tile * BN, 0], W_cmx)
            for c in T.Pipelined(M // BM, num_stages=2):
                T.copy(X1[c * BM, 0], X1_cmx)
                T.copy(X2[c * BM, 0], X2_cmx)
                T.copy(X3[c * BM, 0], X3_cmx)
                T.gemm(X1_cmx, W_cmx, Y1_cmx, transpose_B=True, clear_accum=True)
                T.gemm(X2_cmx, W_cmx, Y2_cmx, transpose_B=True, clear_accum=True)
                T.gemm(X3_cmx, W_cmx, Y3_cmx, transpose_B=True, clear_accum=True)
                T.copy(Y1_cmx, Y1[c * BM, tile * BN])
                T.copy(Y2_cmx, Y2[c * BM, tile * BN])
                T.copy(Y3_cmx, Y3[c * BM, tile * BN])

    with pytest.raises(Exception, match=r"3 ring\(s\), 3 live at once, need 24 hardware barrier ids .* short by 3"):
        _src(three_at_once)


# -- the flow checker on recorded traces (device/evidence) -------------------------------------

EVIDENCE = Path(__file__).resolve().parent / "device" / "evidence"


def _check(trace, **kw):
    sys.path.insert(0, str(EVIDENCE.parent))
    import trace_timeline

    events, names = trace_timeline.load(trace)
    return trace_timeline.check(events, names, **kw)


def test_flow_checker_passes_the_flow_trace_and_fails_a_plain_gemm():
    flow = EVIDENCE / "tl_ring_flow_bm64_bn128_k2048_2048x384x2048_trace.json"
    gemm = EVIDENCE / "tl_gemm_f16_256x256x512_trace.json"
    if not (flow.exists() and gemm.exists()):
        pytest.skip("recorded traces missing")
    assert _check(flow, chunks=32, tiles=3, min_busy=50) == []
    problems = _check(gemm, tiles=2)
    assert problems and any("multicast" in p for p in problems), problems
    # the busy threshold is a live check: an impossible bar fails the flow trace too
    assert any("DPU busy" in p for p in _check(flow, chunks=32, tiles=3, min_busy=99))


def test_a_shared_fill_under_a_shave_branch_syncs_after_the_branch():
    """PlanIntelNpuDistribution ends every DDR -> shared fill with a tile sync. Under `if shave == 0` that sync
    used to land INSIDE the branch, where the other SHAVE never arrives and the tile waits forever -- emitted
    silently from six ordinary lines. The sync now goes after the branch, which is where the tile actually meets.
    The control is the unguarded twin, whose fill the pass hands to SHAVE 0 by itself: both lower, and in neither
    does a sync sit inside a SHAVE branch."""

    def make(guard: bool):
        @T.prim_func
        def k(A: T.Tensor((1, 64), T.float16), Y: T.Tensor((4, 64), T.float16)):
            with T.Kernel(2, threads=2) as tile:
                shave = T.get_thread_binding()
                s_cmx = T.alloc_shared((1, 64), T.float16)
                if guard:  # Python-level: the two arms are different kernels
                    if shave == 0:
                        T.copy(A[0, 0], s_cmx)
                else:
                    T.copy(A[0, 0], s_cmx)
                T.copy(s_cmx, Y[tile * 2 + shave, 0])

        return k

    for guard in (True, False):
        body = _src(make(guard)).split("kernel_entry")[0]
        assert "ctx.sync_tile();" in body, guard
        inside = body.split("if (shave == 0) {", 1)[1].split("}", 1)[0]
        assert "sync_tile" not in inside, f"the tile barrier is unreachable for the other SHAVE (guard={guard})"


def test_a_loop_of_unsplittable_fills_is_dealt_over_the_shaves():
    """A one-row DDR -> shared fill cannot be split across the tile's SHAVEs, so the per-statement rule gives it
    to SHAVE 0 and brackets it with barriers. In a loop that is once per ITERATION: the MoE gather of 16 token
    rows became 16 serialized DMAs on SHAVE 0 and 32 tile barriers, with SHAVE 1 only waiting. Iterations writing
    disjoint rows have no hazard between them, so the loop is dealt over the SHAVEs and meets once."""

    def make(by_iteration: bool):
        @T.prim_func
        def k(A: T.Tensor((16, 64), T.float16), Y: T.Tensor((4, 64), T.float16)):
            with T.Kernel(2, threads=2) as tile:
                shave = T.get_thread_binding()
                s_cmx = T.alloc_shared((16, 64), T.float16)
                for i in T.serial(16):
                    # control: every iteration writing the SAME row is a real serial chain, never dealt out
                    T.copy(A[i, 0], s_cmx[i if by_iteration else 0, 0:64])
                T.copy(s_cmx[0:1, 0:64], Y[tile * 2 + shave, 0])

        return k

    dealt = _src(make(True)).split("kernel_entry")[0]
    assert "if ((i & 1) == shave)" in dealt, "the iterations are not dealt over the tile's SHAVEs"
    loop = dealt.split("for (int32_t i = 0;", 1)[1].split("\n  }", 1)[0]
    assert "sync_tile" not in loop, "a barrier per iteration is exactly the cost being removed"

    same_row = _src(make(False)).split("kernel_entry")[0]
    assert "if ((i & 1) == shave)" not in same_row, "iterations that alias must stay serialized"
    assert "if (shave == 0)" in same_row
