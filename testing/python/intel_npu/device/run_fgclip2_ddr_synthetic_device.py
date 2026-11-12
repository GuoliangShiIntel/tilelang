"""Real-NPU-hardware smoke test for `make_fgclip2_encoder_layer_ddr` at a SMALLER real-channel-width
scale (rows=256, valid=200) than the full real-layer0 device run (rows=1024). Device counterpart of
`test_intel_npu_fgclip2_ddr_synthetic.py` -- same synthetic weights/inputs, same numpy oracle, but
launched on actual NPU hardware via `runner.py`/`npu_run.exe`.

Purpose: `run_fgclip2_layer0_device.py` (rows=1024, real checkpoint) hit a device-side
BARRIER_TIMEOUT (status 6, detail 13) -- a hardcoded 50M-spin limit baked into the SHAVE firmware
(`src/tl_templates/intel_npu/npu_rt_device.hpp`'s `kBarrierSpinLimit`), NOT adjustable at runtime.
This script checks whether a 4x-smaller same-shape kernel clears all its barriers on real hardware,
to tell a genuine scale/timeout issue apart from a correctness/deadlock bug.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_fgclip2_ddr_synthetic_device.py [--no-build]
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
import tvm  # noqa: E402

import runner  # noqa: E402
import lowering  # noqa: E402
from fgclip2_kernels import encoder_layer_reference, make_fgclip2_encoder_layer_ddr  # noqa: E402

ROWS, VALID, HIDDEN, HEADS, HD_PAD, INTER_PAD = 256, 200, 1152, 16, 80, 4320
BR, TN, BQ, C = 32, 32, 64, 128
QW = HEADS * HD_PAD
TILES, SHAVES = 1, 2
KERNEL = "tl_fgclip2_encoder_ddr_synth"


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _weights(rng):
    def w(n, k):
        return (rng.standard_normal((n, k)) * 0.02).astype(np.float16)

    def b(n):
        return (rng.standard_normal(n) * 0.01).astype(np.float16).reshape(1, n)

    return dict(
        ln1w=np.ones((1, HIDDEN), np.float16), ln1b=b(HIDDEN),
        wq=w(QW, HIDDEN), bq=b(QW), wk=w(QW, HIDDEN), bk=b(QW), wv=w(QW, HIDDEN), bv=b(QW),
        wo=w(HIDDEN, QW), bo=b(HIDDEN),
        ln2w=np.ones((1, HIDDEN), np.float16), ln2b=b(HIDDEN),
        w1=w(INTER_PAD, HIDDEN), b1=b(INTER_PAD), w2=w(HIDDEN, INTER_PAD), b2=b(HIDDEN),
    )


def main(argv):
    func = make_fgclip2_encoder_layer_ddr(ROWS, VALID, HIDDEN, HEADS, HD_PAD, INTER_PAD, BR=BR, TN=TN, BQ=BQ, C=C)
    low = _lower(func)
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, KERNEL)
        print(f"build {time.time() - t0:.0f} s")

    rng = np.random.default_rng(0)
    W = _weights(rng)
    x = (rng.standard_normal((ROWS, HIDDEN)) * 0.3).astype(np.float32)
    x[VALID:] = (rng.standard_normal((ROWS - VALID, HIDDEN)) * 5.0).astype(np.float32)
    hio = x.copy()
    by_name = {
        "Hio": ("inout", hio), "Ln1W": ("in", W["ln1w"]), "Ln1B": ("in", W["ln1b"]),
        "Wq": ("in", W["wq"]), "Bq": ("in", W["bq"]), "Wk": ("in", W["wk"]), "Bk": ("in", W["bk"]),
        "Wv": ("in", W["wv"]), "Bv": ("in", W["bv"]), "Wo": ("in", W["wo"]), "Bo": ("in", W["bo"]),
        "Ln2W": ("in", W["ln2w"]), "Ln2B": ("in", W["ln2b"]), "W1": ("in", W["w1"]), "B1": ("in", W["b1"]),
        "W2": ("in", W["w2"]), "B2": ("in", W["b2"]),
        "Xn": ("out", np.zeros((ROWS, HIDDEN), np.float16)), "Qp": ("out", np.zeros((ROWS, QW), np.float16)),
        "Kp": ("out", np.zeros((ROWS, QW), np.float16)), "Vp": ("out", np.zeros((ROWS, QW), np.float16)),
        "Ao": ("out", np.zeros((ROWS, QW), np.float16)), "Act": ("out", np.zeros((ROWS, INTER_PAD), np.float16)),
    }
    arrays = lowering.order_arrays(low, by_name)
    t0 = time.time()
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_fgclip2_ddr_synth", repeat=3, warmup=1, timeout_s=180)
    print(f"device run {time.time() - t0:.0f} s")
    print(f"status per worker (code, detail, cmx_free0, cmx_used): {res.status[:, :4].tolist()}")
    runner.check_status(res.status)

    def masked_encoder_layer(x_full):
        out = x_full.copy()
        v = x_full[:VALID]
        ref_v = encoder_layer_reference(v, W["ln1w"], W["ln1b"], W["wq"], W["bq"], W["wk"], W["bk"], W["wv"], W["bv"],
                                        W["wo"], W["bo"], W["ln2w"], W["ln2b"], W["w1"], W["b1"], W["w2"], W["b2"],
                                        HEADS, HD_PAD)
        out[:VALID] = ref_v
        return out

    ref = masked_encoder_layer(x)
    rel = np.abs(hio[:VALID].astype(np.float64) - ref[:VALID]).max() / np.abs(ref[:VALID]).max()
    verdict = "PASS" if rel < 5e-2 else "FAIL"
    print(f"FG-CLIP2 synthetic encoder layer (rows=256,valid=200) on REAL NPU HARDWARE: rel err {rel:.6f} -> {verdict}")
    print(f"device: median {res.median_ms:.3f} ms [UNPROFILED host-wall incl. submit]; kernel sha256 {db.source_sha256[:12]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
