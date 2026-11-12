"""Minimal real-NPU-hardware bisection probe for the FG-CLIP2 barrier-timeout hang (see
`run_fgclip2_v0_device.py` / repo memory "OPEN BUG" note): a single-active-SHAVE kernel doing
ONLY one tiny (16x16x16) DPU matmul, nothing else, then `ctx.sync_grid()` (via T.Kernel exit).

If THIS hangs the same way (SHAVE 0 / SHAVE 1 timeout on adjacent sync_grid barrier ids), the
16x16x16 DPU matmul itself (DPU-alignment minimum size) is the culprit on this NPU6 hardware,
not FGCLIP2's broader structure. If it passes, the hang needs a bigger kernel to reproduce (point
at FGCLIP2's broader sequential/asymmetric-arrival structure instead).

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_tiny_dpu_matmul_probe_device.py [--no-build]
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
TILES, SHAVES = 1, 2
KERNEL = "tl_tiny_dpu_matmul_probe"


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def make_probe():
    @T.prim_func
    def probe(A: T.Tensor((N, N), T.float16), B: T.Tensor((N, N), T.float16), C: T.Tensor((N, N), T.float16),
              A2: T.Tensor((N, 4 * N), T.float16), B2: T.Tensor((N, 4 * N), T.float16), C2: T.Tensor((N, N), T.float16)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            a = T.alloc_shared((N, N), T.float16)
            b = T.alloc_shared((N, N), T.float16)
            c = T.alloc_shared((N, N), T.float16)
            a2 = T.alloc_shared((N, 4 * N), T.float16)
            b2 = T.alloc_shared((N, 4 * N), T.float16)
            c2 = T.alloc_shared((N, N), T.float16)
            if shave == 0:
                # FIRST a DIFFERENT-shaped matmul (16x16x64), THEN the 16x16x16 one -- bisects whether a
                # DPU descriptor shape CHANGE between consecutive calls (as FGCLIP2's V0 does: 16x64x64
                # projections, then four 16x16x16 attention heads, then 16x64x64 o_proj, ...) is the hang,
                # vs a single isolated tiny matmul (which `run_tiny_dpu_matmul_probe_device.py`'s original
                # single-call version already confirmed PASSES).
                T.copy(A2, a2)
                T.copy(B2, b2)
                T.gemm(a2, b2, c2, transpose_B=True, clear_accum=True)
                T.copy(c2, C2)
                T.copy(A, a)
                T.copy(B, b)
                T.gemm(a, b, c, transpose_B=True, clear_accum=True)
                T.copy(c, C)
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
    a = rng.standard_normal((N, N)).astype(np.float16)
    b = rng.standard_normal((N, N)).astype(np.float16)
    c = np.zeros((N, N), np.float16)
    a2 = rng.standard_normal((N, 4 * N)).astype(np.float16)
    b2 = rng.standard_normal((N, 4 * N)).astype(np.float16)
    c2 = np.zeros((N, N), np.float16)
    by_name = {"A": ("in", a), "B": ("in", b), "C": ("out", c), "A2": ("in", a2), "B2": ("in", b2), "C2": ("out", c2)}
    arrays = lowering.order_arrays(low, by_name)
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_tiny_dpu_matmul_probe", repeat=3, warmup=1, timeout_s=60)
    print(f"status per worker (code, detail, cmx_free0, cmx_used): {res.status[:, :4].tolist()}")
    runner.check_status(res.status)
    ref = a.astype(np.float32) @ b.astype(np.float32).T
    ref2 = a2.astype(np.float32) @ b2.astype(np.float32).T
    err = max(float(np.abs(c.astype(np.float32) - ref).max()), float(np.abs(c2.astype(np.float32) - ref2).max()))
    rel = err / float(np.abs(ref).max())
    verdict = "PASS" if rel < 2e-3 else "FAIL"
    print(f"TWO sequential DIFFERENT-shape DPU matmuls (16x64x64 then 16x16x16), single active SHAVE, on REAL NPU HARDWARE: rel err {rel:.6f} -> {verdict}")
    print(f"device: median {res.median_ms:.3f} ms; kernel sha256 {db.source_sha256[:12]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
