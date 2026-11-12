"""Real-NPU-hardware isolation test for the V0 (small, fully CMX-resident, no DDR streaming) FG-CLIP2
encoder layer -- run while diagnosing a BARRIER_TIMEOUT deadlock seen on real hardware for the
DDR-resident kernel (`run_fgclip2_ddr_synthetic_device.py`, both SHAVEs stuck on adjacent barrier
ids 13/14, a genuine cross-SHAVE deadlock not reproduced by host-sim). If V0 also deadlocks, the bug
is general to the encoder-layer attention/pipeline structure; if V0 passes, it is specific to the
DDR row-block/weight-tile/key-chunk streaming machinery.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_fgclip2_v0_device.py [--no-build]
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
from fgclip2_kernels import encoder_layer_reference, make_fgclip2_encoder_layer_v0  # noqa: E402

ROWS, HIDDEN, HEADS, HEAD_DIM, INTER = 16, 64, 4, 16, 128
TILES, SHAVES = 1, 2
KERNEL = "tl_fgclip2_encoder_v0"


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _weights(rng):
    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float16)

    def b(n):
        return (rng.standard_normal(n) * 0.02).astype(np.float16)

    return dict(
        ln1w=np.ones((1, HIDDEN), np.float16), ln1b=b(HIDDEN).reshape(1, HIDDEN),
        wq=w(HIDDEN, HIDDEN), bq=b(HIDDEN).reshape(1, HIDDEN),
        wk=w(HIDDEN, HIDDEN), bk=b(HIDDEN).reshape(1, HIDDEN),
        wv=w(HIDDEN, HIDDEN), bv=b(HIDDEN).reshape(1, HIDDEN),
        wo=w(HIDDEN, HIDDEN), bo=b(HIDDEN).reshape(1, HIDDEN),
        ln2w=np.ones((1, HIDDEN), np.float16), ln2b=b(HIDDEN).reshape(1, HIDDEN),
        w1=w(INTER, HIDDEN), b1=b(INTER).reshape(1, INTER),
        w2=w(HIDDEN, INTER), b2=b(HIDDEN).reshape(1, HIDDEN),
    )


def main(argv):
    func = make_fgclip2_encoder_layer_v0(ROWS, HIDDEN, HEADS, HEAD_DIM, INTER)
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
    hio = x.copy()
    by_name = {
        "Hio": ("inout", hio), "Ln1W": ("in", W["ln1w"]), "Ln1B": ("in", W["ln1b"]),
        "Wq": ("in", W["wq"]), "Bq": ("in", W["bq"]), "Wk": ("in", W["wk"]), "Bk": ("in", W["bk"]),
        "Wv": ("in", W["wv"]), "Bv": ("in", W["bv"]), "Wo": ("in", W["wo"]), "Bo": ("in", W["bo"]),
        "Ln2W": ("in", W["ln2w"]), "Ln2B": ("in", W["ln2b"]), "W1": ("in", W["w1"]), "B1": ("in", W["b1"]),
        "W2": ("in", W["w2"]), "B2": ("in", W["b2"]),
    }
    arrays = lowering.order_arrays(low, by_name)
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_fgclip2_v0", repeat=3, warmup=1, timeout_s=60)
    print(f"status per worker (code, detail, cmx_free0, cmx_used): {res.status[:, :4].tolist()}")
    runner.check_status(res.status)

    ref = encoder_layer_reference(x, W["ln1w"], W["ln1b"], W["wq"], W["bq"], W["wk"], W["bk"], W["wv"], W["bv"],
                                   W["wo"], W["bo"], W["ln2w"], W["ln2b"], W["w1"], W["b1"], W["w2"], W["b2"],
                                   HEADS, HEAD_DIM)
    rel = np.abs(hio.astype(np.float64) - ref).max() / np.abs(ref).max()
    verdict = "PASS" if rel < 5e-2 else "FAIL"
    print(f"FG-CLIP2 V0 encoder layer on REAL NPU HARDWARE: rel err {rel:.6f} -> {verdict}")
    print(f"device: median {res.median_ms:.3f} ms; kernel sha256 {db.source_sha256[:12]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
