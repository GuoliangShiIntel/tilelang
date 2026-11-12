"""Real NPU device golden for FG-CLIP2: one vision-tower encoder layer (layer 0), REAL pretrained
weights (`model.safetensors`), REAL image patches (972, from the model's own golden trace), run on
the actual NPU hardware (not host-sim) and compared against the real HF model's own recorded
layer-0 output (`golden_fp32.pt`'s `trace_image0.layers[0]`).

This is the device-execution counterpart of
`test_intel_npu_fgclip2_real_layer0.py::test_encoder_layer0_matches_the_real_models_own_layer0_trace`
(host-sim, 0.73% rel error) -- same kernel, same real data, same comparison, but launched through
`runner.py`/`npu_run.exe` on real hardware instead of the host-sim clang++ binary.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set; TL_NPU_DEVICES/TL_NPU_CMAKE_GENERATOR as needed
for a non-PTL/NPU5, non-VS2022 machine):
    python testing/python/intel_npu/device/run_fgclip2_layer0_device.py [--no-build] [--profile]
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
from fgclip2_kernels import make_fgclip2_encoder_layer_ddr  # noqa: E402
from fgclip2_pack import Checkpoint, pack_layer  # noqa: E402

MODEL_ROOT = Path(r"C:\Intel\guoliang\dsl\fg-clip2-so400m")
GOLDEN_PATH = Path(r"C:\Intel\guoliang\dsl\fgclip2_data\golden_fp32.pt")
HIDDEN, HEADS, HD_PAD, INTER_PAD = 1152, 16, 80, 4320
BR, TN, BQ, C = 32, 32, 64, 128
TILES, SHAVES = 1, 2   # matches make_fgclip2_encoder_layer_ddr's own grid (available as a blob on every NPU generation)
KERNEL = "tl_fgclip2_real_layer0"


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def main(argv):
    if not MODEL_ROOT.exists() or not GOLDEN_PATH.exists():
        print(f"real checkpoint/golden not found ({MODEL_ROOT}, {GOLDEN_PATH})")
        return 2
    import torch

    golden = torch.load(GOLDEN_PATH, weights_only=False)
    embed = golden["trace_image0"]["embed"].numpy().astype(np.float32)   # [972, 1152]
    layer0_out = golden["trace_image0"]["layers"][0].numpy().astype(np.float64)
    valid = embed.shape[0]
    rows = -(-valid // BQ) * BQ
    rows = -(-rows // C) * C

    ck = Checkpoint(MODEL_ROOT)
    w = pack_layer(ck, "vision_model", 0, hd_pad=HD_PAD, inter_pad=INTER_PAD)

    hio = np.zeros((rows, HIDDEN), np.float32)
    hio[:valid] = embed

    func = make_fgclip2_encoder_layer_ddr(rows, valid, HIDDEN, HEADS, HD_PAD, INTER_PAD, BR=BR, TN=TN, BQ=BQ, C=C)
    low = _lower(func)
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, KERNEL)
        print(f"build {time.time() - t0:.0f} s")

    qw = HEADS * HD_PAD
    by_name = {
        "Hio": ("inout", hio), "Ln1W": ("in", w["ln1w"]), "Ln1B": ("in", w["ln1b"]),
        "Wq": ("in", w["wq"]), "Bq": ("in", w["bq"]), "Wk": ("in", w["wk"]), "Bk": ("in", w["bk"]),
        "Wv": ("in", w["wv"]), "Bv": ("in", w["bv"]), "Wo": ("in", w["wo"]), "Bo": ("in", w["bo"]),
        "Ln2W": ("in", w["ln2w"]), "Ln2B": ("in", w["ln2b"]), "W1": ("in", w["w1"]), "B1": ("in", w["b1"]),
        "W2": ("in", w["w2"]), "B2": ("in", w["b2"]),
        "Xn": ("out", np.zeros((rows, HIDDEN), np.float16)), "Qp": ("out", np.zeros((rows, qw), np.float16)),
        "Kp": ("out", np.zeros((rows, qw), np.float16)), "Vp": ("out", np.zeros((rows, qw), np.float16)),
        "Ao": ("out", np.zeros((rows, qw), np.float16)), "Act": ("out", np.zeros((rows, INTER_PAD), np.float16)),
    }
    arrays = lowering.order_arrays(low, by_name)
    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{KERNEL}_trace.json"
    t0 = time.time()
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_fgclip2_layer0", repeat=5, warmup=2, timeout_s=300, profile_json=profile)
    print(f"device run {time.time() - t0:.0f} s")
    print(f"status per worker (code, detail, cmx_free0, cmx_used): {res.status[:, :4].tolist()}")
    runner.check_status(res.status)

    kernel_out = hio[:valid].astype(np.float64)
    err = float(np.abs(kernel_out - layer0_out).max())
    rel = err / float(np.abs(layer0_out).max())
    kind = "PROFILED (observability, not a timing)" if profile else "UNPROFILED host-wall incl. submit"
    verdict = "PASS" if rel < 5e-2 else "FAIL"
    print(f"FG-CLIP2 real layer-0 on REAL NPU HARDWARE, real image (972 patches): "
          f"max|kernel-real model|/max|real model| = {rel:.6f} -> {verdict}")
    print(f"device: median {res.median_ms:.3f} ms [{kind}]; kernel sha256 {db.source_sha256[:12]}")
    if profile:
        print(f"trace: {profile}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
