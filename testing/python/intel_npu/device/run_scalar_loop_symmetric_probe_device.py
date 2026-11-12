"""Bisection probe #6 (see repo memory "OPEN BUG" note): same scalar elementwise loop as
`run_scalar_loop_probe_device.py` (which reliably hangs), but BOTH SHAVEs do an equal share of the
work instead of leaving SHAVE 1 fully idle. Tests whether the hang is specifically about an idle
SHAVE waiting at `ctx.sync_grid()` for a partner that was doing scalar (SAU) work, vs. scalar SAU
work being broken in general regardless of symmetry.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_scalar_loop_symmetric_probe_device.py [--no-build]
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

ROWS, H = 16, 64   # split 8 rows per SHAVE
TILES, SHAVES = 1, 2
KERNEL = "tl_scalar_loop_symmetric_probe"


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def make_probe():
    @T.prim_func
    def probe(Hio: T.Tensor((ROWS, H), T.float32)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            h32 = T.alloc_shared((ROWS, H), T.float32)
            T.copy(Hio, h32)
            half = ROWS // SHAVES
            if shave == 0:
                for r in T.serial(half):
                    for d in T.serial(H):
                        h32[r, d] = h32[r, d] * T.float32(2)
            if shave == 1:
                for r in T.serial(half):
                    for d in T.serial(H):
                        h32[half + r, d] = h32[half + r, d] * T.float32(2)
            T.copy(h32, Hio)
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
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_scalar_loop_symmetric_probe", repeat=3, warmup=1, timeout_s=60)
    print(f"status per worker (code, detail, cmx_free0, cmx_used): {res.status[:, :4].tolist()}")
    runner.check_status(res.status)

    ref = x * 2
    err = float(np.abs(hio.astype(np.float64) - ref.astype(np.float64)).max())
    rel = err / float(np.abs(ref).max())
    verdict = "PASS" if rel < 5e-2 else "FAIL"
    print(f"Scalar elementwise loop SPLIT across BOTH SHAVEs (8 rows each), on REAL NPU HARDWARE: rel err {rel:.6f} -> {verdict}")
    print(f"device: median {res.median_ms:.3f} ms; kernel sha256 {db.source_sha256[:12]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
