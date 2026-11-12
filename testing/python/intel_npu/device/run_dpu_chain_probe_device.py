"""Bisection probe #3 (see repo memory "OPEN BUG" note and `run_tiny_dpu_matmul_probe_device.py`):
12 sequential 16x16x16 `ctx.dpu_matmul` calls back-to-back on a single active SHAVE, NO scalar
loops in between -- matches the VOLUME of DPU calls in `make_fgclip2_encoder_layer_v0` (3
projections + 8 per-head attention matmuls + 1 o_proj, all at small shapes) but none of its
LayerNorm/softmax/GELU scalar transcendental loops (`sqrtf`/`expf`/`tanhf`).

Probes #1 (one 16x16x16 matmul) and #2 (two different-shaped matmuls back-to-back) both PASSED on
real NPU6 hardware. If THIS also passes, the hang is specific to the scalar transcendental loops
(LayerNorm/softmax/GELU), not DPU call volume. If this hangs, DPU call volume/accumulated real
time alone is enough.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_dpu_chain_probe_device.py [--no-build]
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401
import tilelang.intel_npu.language as T  # noqa: E402
import tvm  # noqa: E402

import runner  # noqa: E402
import lowering  # noqa: E402

N = 16
NCALLS = 12   # matches make_fgclip2_encoder_layer_v0's dpu_matmul call count
TILES, SHAVES = 1, 2
KERNEL = "tl_dpu_chain_probe"


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def make_probe():
    @T.prim_func
    def probe(A: T.Tensor((N, N), T.float16), B: T.Tensor((N, N), T.float16), C: T.Tensor((N, N), T.float16)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            a = T.alloc_shared((N, N), T.float16)
            b = T.alloc_shared((N, N), T.float16)
            c = T.alloc_shared((N, N), T.float16)
            if shave == 0:
                T.copy(A, a)
                T.copy(B, b)
                # NOTE: a bare Python `for` here does NOT trace-time-unroll in this DSL's eager builder (it
                # becomes a real runtime TIR loop whose buffer handles can't be reassigned across iterations,
                # per repo memory); write the NCALLS calls out explicitly instead, ping-ponging a/c by hand.
                T.gemm(a, b, c, transpose_B=True, clear_accum=True)
                T.gemm(c, b, a, transpose_B=True, clear_accum=True)
                T.gemm(a, b, c, transpose_B=True, clear_accum=True)
                T.gemm(c, b, a, transpose_B=True, clear_accum=True)
                T.gemm(a, b, c, transpose_B=True, clear_accum=True)
                T.gemm(c, b, a, transpose_B=True, clear_accum=True)
                T.gemm(a, b, c, transpose_B=True, clear_accum=True)
                T.gemm(c, b, a, transpose_B=True, clear_accum=True)
                T.gemm(a, b, c, transpose_B=True, clear_accum=True)
                T.gemm(c, b, a, transpose_B=True, clear_accum=True)
                T.gemm(a, b, c, transpose_B=True, clear_accum=True)
                T.gemm(c, b, a, transpose_B=True, clear_accum=True)
                T.copy(a, C)
    return probe


def main(argv):
    low = _lower(make_probe())
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, KERNEL)
        print(f"build {time.time() - t0:.0f} s")

    rng = np.random.default_rng(0)
    a = rng.standard_normal((N, N)).astype(np.float16) * 0.1
    b = rng.standard_normal((N, N)).astype(np.float16) * 0.1
    c = np.zeros((N, N), np.float16)
    by_name = {"A": ("in", a), "B": ("in", b), "C": ("out", c)}
    arrays = lowering.order_arrays(low, by_name)
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_dpu_chain_probe", repeat=3, warmup=1, timeout_s=60)
    print(f"status per worker (code, detail, cmx_free0, cmx_used): {res.status[:, :4].tolist()}")
    runner.check_status(res.status)
    ref = a.astype(np.float64)
    bb = b.astype(np.float64)
    for _ in range(NCALLS):
        ref = ref @ bb.T
    err = float(np.abs(c.astype(np.float64) - ref).max())
    rel = err / max(float(np.abs(ref).max()), 1e-12)
    verdict = "PASS" if rel < 5e-2 else "FAIL"
    print(f"{NCALLS}x chained 16x16x16 DPU matmuls, single active SHAVE, on REAL NPU HARDWARE: rel err {rel:.6f} -> {verdict}")
    print(f"device: median {res.median_ms:.3f} ms; kernel sha256 {db.source_sha256[:12]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
