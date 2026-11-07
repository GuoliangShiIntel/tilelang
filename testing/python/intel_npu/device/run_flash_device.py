"""M1 device golden: one flash-attention phase compiled from kernels.make_flash_attention_phase.

Q block resident per tile (SEGMENTED), K and V^T streamed through two rings (DUPLICATED), each SHAVE keeps
online-softmax partials over its chunks, the tile merges them. Golden against a numpy softmax attention.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set; VPUNN_ROOT for planned workloads):
    python testing/python/intel_npu/device/run_flash_device.py [--no-build] [--profile] [--negative] [--kv N]
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
from kernels import make_flash_attention_phase  # noqa: E402

D, BQ, BKV = 64, 32, 128
TILES, SHAVES = 3, 2
PERF_CLK_MHZ = 19.2


def main(argv):
    numKv = int(argv[argv.index("--kv") + 1]) if "--kv" in argv else 1024
    numQ = BQ * TILES
    kernel = f"tl_flash_phase_q{numQ}_kv{numKv}_d{D}" + ("_vec" if "--vec" in argv else "")
    cfg = {"tl.intel_npu.plan_workloads": "off"} if "--no-vpunn" in argv else {}
    with tvm.transform.PassContext(config=cfg):
        low = lowering.lower(make_flash_attention_phase(numQ, numKv, D=D, BQ=BQ, BKV=BKV, tiles=TILES, shaves=SHAVES,
                                                        vectorized="--vec" in argv))
    for ln in low.source.splitlines():
        if "// intel_npu:" in ln:
            print(ln.strip()[:220])
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, kernel)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=kernel, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, kernel)
        print(f"build {time.time() - t0:.0f} s")
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((numQ, D)) * 0.5).astype(np.float16)
    k = (rng.standard_normal((numKv, D)) * 0.5).astype(np.float16)
    vt = (rng.standard_normal((D, numKv)) * 0.5).astype(np.float16)
    o = np.zeros((numQ, D), np.float16)
    k_dev = k.copy()
    negative = "--negative" in argv
    if negative:
        k_dev[3, 5] += np.float16(6.0)
    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{kernel}_trace.json"
    arrays = lowering.order_arrays(low, {"Q": ("in", q), "K": ("in", k_dev), "Vt": ("in", vt), "O": ("out", o)})
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_flash", profile_json=profile, repeat=10, warmup=3, timeout_s=120)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    s = q.astype(np.float32) @ k.astype(np.float32).T / np.sqrt(D)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    ref = p @ vt.astype(np.float32).T
    err = float(np.abs(o.astype(np.float32) - ref).max())
    rel = err / float(np.abs(ref).max())
    verdict = "PASS" if rel < 2e-2 else "FAIL"
    kind = "PROFILED (observability, not a timing)" if profile else "UNPROFILED"
    flops = 4.0 * numQ * numKv * D
    idu_us = [float(res.status[t * SHAVES:(t + 1) * SHAVES, 5].astype(np.int64).sum()) / PERF_CLK_MHZ for t in range(TILES)]
    print(f"device: kernel-only {res.median_kernel_ms:.3f} ms (host wall {res.median_ms:.3f} ms) [{kind}]; "
          f"max |o - ref| = {err:.3e} (rel {rel:.2e}) -> {verdict}; kernel sha256 {db.source_sha256[:12]}")
    print(f"DPU workloads per tile {res.status[::SHAVES, 4].tolist()} + {res.status[1::SHAVES, 4].tolist()}; DPU IDU busy per tile "
          f"{[f'{u:.0f}' for u in idu_us]} us of {res.median_kernel_ms * 1e3:.0f} us -> {[f'{100 * u / (res.median_kernel_ms * 1e3):.0f}%' for u in idu_us]}; "
          f"{flops / 1e9:.3f} GFLOP, {flops / (res.median_kernel_ms * 1e-3) / 1e12:.2f} TFLOP/s kernel-only (the SHAVE softmax is the rest)")
    if profile:
        print(f"trace: {profile}")
    if negative:
        print("negative control (perturbed K): " + ("behaved (FAIL reported)" if verdict == "FAIL" else "BROKEN: PASS reported"))
        return 0 if verdict == "FAIL" else 1
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
