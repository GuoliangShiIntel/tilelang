"""`T.ddr_view` (plan M5): a DDR tensor at a run-time address read from a table, the mechanism of the single-launch
model's layer loop. Host sim: a loop over 3 "layers" whose rows live in separate arrays named by the table; each is
doubled into the output. Negative: a wrong address in the table gives the wrong row."""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401
import tilelang.intel_npu.language as T
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402

N, L = 256, 3


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except RuntimeError:
        return False


@T.prim_func
def layers(Table: T.Tensor((L, 2), T.int64), Y: T.Tensor((L, N), T.float16)):
    with T.Kernel(2, threads=2) as tile:
        shave = T.get_thread_binding()
        a_cmx = T.alloc_local((N,), T.float16)
        for li in T.serial(L):
            A = T.ddr_view(Table[li, 0], (N,), T.float16)
            T.copy(A[0], a_cmx)
            for i in T.serial(N):
                a_cmx[i] = a_cmx[i] * T.float16(2)
            if tile == 0:
                if shave == 0:
                    T.copy(a_cmx, Y[li, 0])


def test_ddr_view_binds_a_typed_ddr_tensor_from_the_table():
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        src = lowering.lower(layers).source
    assert "const npu::Ddr<half> A((uint64_t)Table[(li * 2)]);" in src
    assert "ctx.dma_load((&(a_cmx[0])), (&(A[0])), 512, 1, 512, 512);" in src


@pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")
def test_ddr_view_reads_the_rows_the_table_names_in_host_sim():
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        low = lowering.lower(layers)
    lib = host_sim.build(low.source, "tl_ddr_view_sim")
    rng = np.random.default_rng(0)
    rows = [np.ascontiguousarray(rng.standard_normal(N).astype(np.float16)) for _ in range(L)]
    table = np.zeros((L, 2), np.int64)
    for i, r in enumerate(rows):
        table[i, 0] = r.ctypes.data
    y = np.zeros((L, N), np.float16)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, dict(Table=table, Y=y)), 2, 2))
    for i, r in enumerate(rows):
        assert np.array_equal(y[i], (r.astype(np.float32) * 2).astype(np.float16))
    # negative: layer 2's entry pointing at layer 0's row
    table[2, 0] = table[0, 0]
    y2 = np.zeros((L, N), np.float16)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, dict(Table=table, Y=y2)), 2, 2))
    assert np.array_equal(y2[2], y[0]) and not np.array_equal(y2[2], y[2])
