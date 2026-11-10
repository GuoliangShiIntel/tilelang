"""Compiler-derived DDR->CMX prefetch: allocation, alternating slots, wait placement and conservative fallback."""

import re

import numpy as np
import pytest

import tilelang.intel_npu.language as T
import tvm

import host_sim
import lowering

BLOCK, ITERS = 32, 4


def _copy_kernel(iters=ITERS, outside_read=False, pin_write=False):
    @T.prim_func
    def copy_kernel(X: T.Tensor((iters * BLOCK, BLOCK), T.float16),
                    W: T.Tensor((BLOCK, BLOCK), T.float16),
                    Y: T.Tensor((iters * BLOCK + BLOCK, BLOCK), T.float16)):
        with T.Kernel(1, threads=1):
            slot = T.alloc_local((BLOCK, BLOCK), T.float16)
            weight = T.alloc_local((BLOCK, BLOCK), T.float16)
            out = T.alloc_local((BLOCK, BLOCK), T.float16)
            T.copy(W[0, 0], weight)
            if pin_write:
                slot[0, 0] = T.float16(0)
            for i in T.serial(iters):
                T.copy(X[i * BLOCK, 0], slot)
                T.gemm(slot, weight, out, transpose_B=True, clear_accum=True)
                T.copy(out, Y[i * BLOCK, 0])
            if outside_read:
                T.copy(slot, Y[iters * BLOCK, 0])

    return copy_kernel


def _lower(func, prefetch=True):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off",
                                           "tl.intel_npu.prefetch": prefetch}):
        return lowering.lower(func)


def _has_host_cxx():
    try:
        host_sim.find_clangxx()
        return True
    except RuntimeError:
        return False


def test_prefetch_doubles_the_real_allocation_and_alternates_slots():
    src = _lower(_copy_kernel()).source
    assert "half* __restrict slot = ctx.alloc<half>(2048);" in src
    assert src.count("ctx.dma_load_async(") == 2   # prologue plus steady-state refill
    assert src.count("ctx.dma_load_wait()") == 1   # one wait at the top of each loop iteration
    assert "ctx.dma_load_async((&(slot[0])), (&(X[0]))" in src
    assert "slot[(1024 - ((i & 1) * 1024))]" in src
    assert "slot[((i & 1) * 1024)]" in src
    no_prefetch = _lower(_copy_kernel(), prefetch=False).source
    assert "ctx.dma_load_async(" not in no_prefetch


def test_prefetch_leaves_single_iteration_and_outside_live_buffer_synchronous():
    single = _lower(_copy_kernel(iters=1)).source
    outside = _lower(_copy_kernel(outside_read=True)).source
    pinned = _lower(_copy_kernel(pin_write=True)).source
    assert "ctx.dma_load_async(" not in single
    assert "ctx.dma_load_async(" not in outside
    assert "ctx.dma_load_async(" in pinned
    assert re.search(r"ctx\.alloc<half>\(1024\)", single)
    assert re.search(r"ctx\.alloc<half>\(1024\)", outside)
    assert re.search(r"ctx\.alloc<half>\(2048\)", pinned)


@pytest.mark.skipif(not _has_host_cxx(), reason="host-sim needs clang++")
def test_prefetched_copy_is_correct_in_host_sim():
    low = _lower(_copy_kernel())
    lib = host_sim.build(low.source, "intel_npu_prefetch_copy")
    x = np.arange(ITERS * BLOCK * BLOCK, dtype=np.float16).reshape(ITERS * BLOCK, BLOCK)
    w = np.eye(BLOCK, dtype=np.float16)
    y = np.full((ITERS * BLOCK + BLOCK, BLOCK), -7, dtype=np.float16)
    arrays = lowering.order_arrays(low, {"X": x, "W": w, "Y": y})
    host_sim.check_status(host_sim.run(lib, arrays, tiles=1, shaves=1))
    np.testing.assert_array_equal(y[:ITERS * BLOCK], x)
    np.testing.assert_array_equal(y[ITERS * BLOCK:], -7)