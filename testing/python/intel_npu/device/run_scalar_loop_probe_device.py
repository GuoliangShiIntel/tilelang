"""Bisection probe #5 (see repo memory "OPEN BUG" note): the SIMPLEST possible scalar loop on a
single active SHAVE -- no DPU, no accumulation, no transcendental, just `xn[r,d] = h32[r,d] * 2`
over 16x64 elements. Probe #4 (LayerNorm's sum/sumsq accumulation + a second scalar pass, WITHOUT
sqrtf even) reproduced the hang; this checks whether ANY scalar double-loop hangs, or specifically
the accumulating-register pattern (`st[0] = st[0] + ...` read-modify-write across many iterations).

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_scalar_loop_probe_device.py [--no-build]
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

ROWS, H = 16, 64
TILES, SHAVES = 1, 2
KERNEL = "tl_scalar_loop_probe"


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def make_probe():
    @T.prim_func
    def probe(Hio: T.Tensor((ROWS, H), T.float32)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            h32 = T.alloc_shared((ROWS, H), T.float32)
            if shave == 0:
                T.copy(Hio, h32)
                for r in T.serial(ROWS):
                    for d in T.serial(H):
                        h32[r, d] = h32[r, d] * T.float32(2)
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
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_scalar_loop_probe", repeat=3, warmup=1, timeout_s=60)
    print(f"status per worker (code, detail, cmx_free0, cmx_used): {res.status[:, :4].tolist()}")
    runner.check_status(res.status)

    ref = x * 2
    err = float(np.abs(hio.astype(np.float64) - ref.astype(np.float64)).max())
    rel = err / float(np.abs(ref).max())
    verdict = "PASS" if rel < 5e-2 else "FAIL"
    print(f"Simplest scalar elementwise loop (16x64, x*2, no accum/transcendental), single active SHAVE, on REAL NPU HARDWARE: rel err {rel:.6f} -> {verdict}")
    print(f"device: median {res.median_ms:.3f} ms; kernel sha256 {db.source_sha256[:12]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
