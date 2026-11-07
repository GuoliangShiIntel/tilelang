"""Phase 3b step A device golden: the hand-written multi-tile ping-pong flow (probes/ring_flow_hand.cpp).

Usage (GENERIC_COMPUTE_ROOT and MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_ring_flow_device.py [--no-build] [--profile] [--negative]

Y[2048, 384] = X[2048, 1024] . W[384, 1024]^T fp16 on 3 tiles x 2 SHAVEs: W split by tile, X
streamed through a 2-slot multicast ring with READY/FREE hardware barriers. Unprofiled by default;
reports host wall (incl. submit), the DPU IDU time per tile from the HWP status words and the
DPU busy % it implies, and an ops/time figure. --profile captures a trace (observability run).
--negative perturbs W so the comparison must FAIL.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401  (puts tvm on the path for runner)
import runner  # noqa: E402

M, K, N = 2048, 2048, 768
TILES, SHAVES = 3, 2
KERNEL = "tl_ring_flow_hand"
PERF_CLK_MHZ = 19.2


def main(argv):
    source = (HERE / "probes" / "ring_flow_hand.cpp").read_text(encoding="utf-8")
    if "--no-build" in argv:
        _, sha = runner.write_kernel(source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(source, KERNEL)
        print(f"build {time.time() - t0:.0f} s")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((M, K)).astype(np.float16)
    w = rng.standard_normal((N, K)).astype(np.float16)
    y = np.zeros((M, N), np.float16)
    w_dev = w.copy()
    negative = "--negative" in argv
    if negative:
        w_dev[5, 7] += np.float16(8.0)
    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{KERNEL}_{M}x{N}x{K}_trace.json"
    res = runner.run(db, [("in", x), ("in", w_dev), ("out", y)], TILES, SHAVES, HERE / "run_ring_flow",
                     profile_json=profile, repeat=10, warmup=3, timeout_s=120)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    ref = x.astype(np.float32) @ w.astype(np.float32).T
    err = float(np.abs(y.astype(np.float32) - ref).max())
    rel = err / float(np.abs(ref).max())
    verdict = "PASS" if rel < 2e-3 else "FAIL"
    kind = "PROFILED (observability, not a timing)" if profile else "UNPROFILED host-wall incl. submit"
    ops = 2.0 * M * N * K
    idu_us_per_tile = [float(res.status[t * SHAVES:(t + 1) * SHAVES, 5].astype(np.int64).sum()) / PERF_CLK_MHZ for t in range(TILES)]
    wl_per_tile = [int(res.status[t * SHAVES:(t + 1) * SHAVES, 4].astype(np.int64).sum()) for t in range(TILES)]
    print(f"device: median {res.median_ms:.3f} ms [{kind}]; max |y - ref| = {err:.4f} (rel {rel:.2e}) -> {verdict}; "
          f"kernel sha256 {db.source_sha256[:12]}")
    print(f"DPU workloads per tile {wl_per_tile}; DPU IDU busy per tile {[f'{u:.0f}' for u in idu_us_per_tile]} us "
          f"of {res.median_ms * 1e3:.0f} us host wall -> busy {[f'{100 * u / (res.median_ms * 1e3):.0f}%' for u in idu_us_per_tile]} (lower bound: the wall includes submit)")
    print(f"ops {ops / 1e9:.2f} GFLOP; {ops / (res.median_ms * 1e-3) / 1e12:.2f} TOPS over host wall (lower bound; peak basis 22.73 TOPS at 1.85 GHz)")
    if profile:
        print(f"trace: {profile}")
    if negative:
        print("negative control (perturbed W): " + ("behaved (FAIL reported)" if verdict == "FAIL" else "BROKEN: PASS reported"))
        return 0 if verdict == "FAIL" else 1
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
