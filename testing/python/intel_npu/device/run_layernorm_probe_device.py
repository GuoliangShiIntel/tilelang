"""Bisection probe #4 (see repo memory "OPEN BUG" note): a LayerNorm-shaped scalar reduction loop
(sum/sumsq over H=64 elements per row, `sqrtf`, 16 rows) on a single active SHAVE, NO DPU matmul
calls at all. Probes #1-#3 (tiny matmul, shape-changing matmuls, 12x chained matmuls) all PASSED;
this isolates whether the SCALAR transcendental path (LayerNorm/softmax/GELU use `sqrtf`/`expf`/
`tanhf`) -- present in `make_fgclip2_encoder_layer_v0` but absent from probes #1-#3 -- is what
hangs on real NPU6 hardware.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_layernorm_probe_device.py [--no-build]
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
SKIP_SQRT = "--skip-sqrt" in sys.argv   # bisect: accumulation loop alone (no sqrtf) vs sqrtf itself
KERNEL = "tl_layernorm_probe_nosqrt" if SKIP_SQRT else "tl_layernorm_probe"


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def make_probe():
    @T.prim_func
    def probe(Hio: T.Tensor((ROWS, H), T.float32)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            h32 = T.alloc_shared((ROWS, H), T.float32)
            xn = T.alloc_shared((ROWS, H), T.float16)
            st = T.alloc_local((2,), T.float32)
            if shave == 0:
                T.copy(Hio, h32)
                for r in T.serial(ROWS):
                    st[0] = T.float32(0)
                    st[1] = T.float32(0)
                    for d in T.serial(H):
                        st[0] = st[0] + h32[r, d]
                        st[1] = st[1] + h32[r, d] * h32[r, d]
                    mean = st[0] / T.float32(H)
                    var = st[1] / T.float32(H) - mean * mean
                    invstd = T.float32(1) if SKIP_SQRT else T.float32(1) / T.sqrt(var + T.float32(1e-6))
                    for d in T.serial(H):
                        xn[r, d] = T.cast((h32[r, d] - mean) * invstd, T.float16)
                for r in T.serial(ROWS):
                    for d in T.serial(H):
                        h32[r, d] = T.cast(xn[r, d], T.float32)
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
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_layernorm_probe", repeat=3, warmup=1, timeout_s=60)
    print(f"status per worker (code, detail, cmx_free0, cmx_used): {res.status[:, :4].tolist()}")
    runner.check_status(res.status)

    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    ref = (x - mean) if SKIP_SQRT else (x - mean) / np.sqrt(var + 1e-6)
    err = float(np.abs(hio.astype(np.float64) - ref.astype(np.float64)).max())
    rel = err / float(np.abs(ref).max())
    verdict = "PASS" if rel < 5e-2 else "FAIL"
    label = "accumulation-loop-only (NO sqrtf)" if SKIP_SQRT else "full (sqrtf)"
    print(f"LayerNorm-only scalar loop (16x64, {label}), single active SHAVE, on REAL NPU HARDWARE: rel err {rel:.6f} -> {verdict}")
    print(f"device: median {res.median_ms:.3f} ms; kernel sha256 {db.source_sha256[:12]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
