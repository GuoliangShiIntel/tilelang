"""Follow-up to probe_dma_bisect_device.py: all pure load-then-store variants gave all-zero output,
even tiles=1/shaves=1. The working `make_gemm` kernel ALSO does T.copy-load -> T.copy-store but with
a T.gemm (DPU workload) call in between, and wraps everything in a `for w in T.serial(...): if blk <
nblk:` loop/guard. This isolates which of those two structural differences (the loop/guard wrapper,
or the DPU call in between) is what makes DMA actually land on real hardware.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402
import tvm  # noqa: E402
import runner  # noqa: E402
import tilelang.intel_npu.language as T  # noqa: E402
import numpy as np  # noqa: E402

N, W = 8, 64


def make_probe_loop_guard():
    """Same load->store, no DPU, but wrapped in GEMM's `for w in T.serial(1): if blk < 1:` pattern."""

    @T.prim_func
    def probe(Q: T.Tensor((N, W), T.float16), O: T.Tensor((N, W), T.float16)):
        with T.Kernel(2, threads=2) as tile:
            shave = T.get_thread_binding()
            worker = tile * 2 + shave
            q_cmx = T.alloc_local((4, W), T.float16)
            for w in T.serial(T.ceildiv(4, 4)):
                blk = w * 4 + worker
                if blk < 4:
                    T.copy(Q[blk * 4, 0], q_cmx)
                    T.copy(q_cmx, O[blk * 4, 0])

    return probe


def make_probe_with_gemm():
    """Same load->store, but with a trivial T.gemm (DPU no-op-ish workload) in between, matching
    GEMM's structure exactly (minus the loop, since workers == nblk here so it is not needed).
    Stores q_cmx (DMA-load-only buffer), NOT the DPU's own output -- isolates whether storing a
    buffer the DPU never touched is what causes all-zero output."""

    @T.prim_func
    def probe(Q: T.Tensor((N, W), T.float16), O: T.Tensor((N, W), T.float16)):
        with T.Kernel(2, threads=2) as tile:
            shave = T.get_thread_binding()
            worker = tile * 2 + shave
            q_cmx = T.alloc_local((4, W), T.float16)
            ident_cmx = T.alloc_local((16, W), T.float16)
            o_cmx = T.alloc_local((4, 16), T.float16)
            T.copy(Q[worker * 4, 0], q_cmx)
            T.copy(Q[worker * 4, 0], ident_cmx[0:4, :])  # garbage rows 4:16 fine; o_cmx never read
            T.gemm(q_cmx, ident_cmx, o_cmx, transpose_B=True, clear_accum=True)
            T.copy(q_cmx, O[worker * 4, 0])

    return probe


def make_probe_store_dpu_output():
    """Same as make_probe_with_gemm, but stores o_cmx (the DPU's OWN output buffer) instead of
    q_cmx. If this one is non-zero/correct while make_probe_with_gemm is all-zero, that confirms
    the constraint is specifically 'must store a DPU-written buffer', not just 'kernel must call
    dpu_matmul somewhere'."""

    @T.prim_func
    def probe(Q: T.Tensor((N, W), T.float16), O: T.Tensor((N, 16), T.float16)):
        with T.Kernel(2, threads=2) as tile:
            shave = T.get_thread_binding()
            worker = tile * 2 + shave
            q_cmx = T.alloc_local((4, W), T.float16)
            ident_cmx = T.alloc_local((16, W), T.float16)
            o_cmx = T.alloc_local((4, 16), T.float16)
            T.copy(Q[worker * 4, 0], q_cmx)
            T.copy(Q[worker * 4, 0], ident_cmx[0:4, :])
            T.gemm(q_cmx, ident_cmx, o_cmx, transpose_B=True, clear_accum=True)
            T.copy(o_cmx, O[worker * 4, 0])

    return probe


def run_one(name, func, shape=(N, W)):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        source = tilelang.lower(func, target="intel_npu").kernel_source
    db = runner.build(source, name)
    q = np.random.default_rng(5).standard_normal((N, W)).astype(np.float16)
    o = np.full(shape, -123.0, np.float16)
    res = runner.run(db, [("in", q), ("out", o)], 2, 2, HERE / f"run_{name}")
    runner.check_status(res.status)
    match = bool(np.array_equal(q[:, :shape[1]], o)) if shape[1] == W else None
    nonzero = int(np.count_nonzero(o))
    print(f"{name}: exact_match={match} nonzero_elems={nonzero}/{o.size} sample_row0={o[0, :8]}")
    return nonzero


def main(argv):
    r2 = run_one("tl_probe_with_gemm", make_probe_with_gemm())
    r3 = run_one("tl_probe_store_dpu_output", make_probe_store_dpu_output(), shape=(N, 16))
    print(f"\nSUMMARY: with_gemm(stores DMA-only buf) nonzero={r2}; store_dpu_output(stores DPU buf) nonzero={r3}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
