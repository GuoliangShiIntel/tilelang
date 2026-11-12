"""Bisection probe #7 (see repo memory "OPEN BUG" note): generic_compute's OWN `kernel_layernorm.cpp`
(tests/kernels/layernorm, CI-validated on NVL) uses per-SHAVE PRIVATE CMX buffers (`npu::BufferArena`,
TileLang's `alloc_local` equivalent) and NEVER calls any barrier/sync_grid at all -- every SHAVE
does independent work and returns. It PASSES on this exact NPU6 hardware (verified directly,
`tests/kernels/layernorm` built and run standalone: all tests PASSED).

This probe isolates whether the hang is specifically about a SHARED CMX buffer (`T.alloc_shared`,
one copy per tile, used by every prior failing probe) as opposed to a scalar store in general:
both SHAVEs do independent scalar work into their OWN alloc_local buffer (no cross-SHAVE data
dependency at all, matching generic_compute's own layernorm pattern), and ctx.sync_grid() is still
called (TileLang always inserts it at kernel exit).

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_scalar_loop_private_probe_device.py [--no-build]
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

ROWS, H = 16, 64   # 8 rows per SHAVE, each SHAVE's rows in its OWN private CMX buffer
TILES, SHAVES = 1, 2
KERNEL = "tl_scalar_loop_private_probe"


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def make_probe():
    @T.prim_func
    def probe(Hio: T.Tensor((ROWS, H), T.float32)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            half = ROWS // SHAVES
            h32 = T.alloc_local((half, H), T.float32)   # PRIVATE per SHAVE, not alloc_shared
            if shave == 0:
                T.copy(Hio[0, 0], h32)
                for r in T.serial(half):
                    for d in T.serial(H):
                        h32[r, d] = h32[r, d] * T.float32(2)
                T.copy(h32, Hio[0, 0])
            if shave == 1:
                T.copy(Hio[half, 0], h32)
                for r in T.serial(half):
                    for d in T.serial(H):
                        h32[r, d] = h32[r, d] * T.float32(2)
                T.copy(h32, Hio[half, 0])
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
    x = (rng.standard_normal((ROWS, H)) * 0.3).astype(np.float32)
    hio = x.copy()
    by_name = {"Hio": ("inout", hio)}
    arrays = lowering.order_arrays(low, by_name)
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_scalar_loop_private_probe", repeat=5, warmup=1, timeout_s=60)
    print(f"status per worker (code, detail, cmx_free0, cmx_used): {res.status[:, :4].tolist()}")
    runner.check_status(res.status)

    ref = x * 2
    err = float(np.abs(hio.astype(np.float64) - ref.astype(np.float64)).max())
    rel = err / float(np.abs(ref).max())
    verdict = "PASS" if rel < 5e-2 else "FAIL"
    print(f"Scalar loop into PRIVATE (alloc_local) per-SHAVE buffers, on REAL NPU HARDWARE: rel err {rel:.6f} -> {verdict}")
    print(f"device: median {res.median_ms:.3f} ms; kernel sha256 {db.source_sha256[:12]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
