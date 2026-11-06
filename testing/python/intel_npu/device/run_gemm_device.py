"""Phase 1 device golden: the TileLang fp16 GEMM on the PTL/NPU5 against numpy.

Usage (GENERIC_COMPUTE_ROOT and MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_gemm_device.py [--no-build] [--profile] [--negative] [--cmx-negative]

Unprofiled by default (a timing run); --profile captures the runtime trace to
evidence/ and makes it an observability run. Prints the status per SHAVE, the
max error against numpy, PASS/FAIL, and the host-wall median.
--negative feeds the device a perturbed B so the comparison must FAIL;
--cmx-negative builds a kernel whose CMX tiles cannot fit so the per-SHAVE
status must report CMX_EXHAUSTED (exit 0 means the control behaved).
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402
import runner  # noqa: E402
from kernels import make_gemm  # noqa: E402

M, N, K = 256, 256, 512
BM, BN = 64, 64
TILES, SHAVES = 2, 2
KERNEL = "tl_gemm_f16"


def main(argv):
    if "--cmx-negative" in argv:
        return cmx_negative()
    func = make_gemm(M, N, K, BM=BM, BN=BN, tiles=TILES, shaves=SHAVES)
    source = tilelang.lower(func, target="intel_npu").kernel_source
    if "--no-build" in argv:
        path, sha = runner.write_kernel(source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(source, KERNEL)
        print(f"build {time.time() - t0:.0f} s")

    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((N, K)).astype(np.float16)
    c = np.zeros((M, N), np.float16)
    b_dev = b.copy()
    negative = "--negative" in argv
    if negative:
        b_dev[7, 11] += np.float16(8.0)
    profile = None
    if "--profile" in argv:
        evidence = HERE / "evidence"
        evidence.mkdir(exist_ok=True)
        profile = evidence / f"{KERNEL}_{M}x{N}x{K}_trace.json"
    res = runner.run(db, [("in", a), ("in", b_dev), ("out", c)], TILES, SHAVES, HERE / "run_gemm", profile_json=profile)
    print(f"status per SHAVE (code, detail, cmx_free0, cmx_used): {res.status[:, :4].tolist()}")
    runner.check_status(res.status)
    ref = a.astype(np.float32) @ b.astype(np.float32).T
    err = float(np.abs(c.astype(np.float32) - ref).max())
    rel = err / float(np.abs(ref).max())
    kind = "PROFILED (observability, not a timing)" if profile else "UNPROFILED host-wall incl. submit"
    verdict = "PASS" if rel < 2e-3 else "FAIL"
    print(f"device: median {res.median_ms:.3f} ms [{kind}]; max |c - ref| = {err:.4f} (rel {rel:.2e}) -> "
          f"{verdict}; kernel sha256 {db.source_sha256[:12]}")
    if profile:
        print(f"trace: {profile}")
    if negative:
        print("negative control (perturbed B): " + ("behaved (FAIL reported)" if verdict == "FAIL" else "BROKEN: PASS reported"))
        return 0 if verdict == "FAIL" else 1
    return 0 if verdict == "PASS" else 1


def cmx_negative():
    # 2048 x 1024 fp16 A tile = 4 MiB per SHAVE: cannot fit a 1.5 MiB tile; the status record must say so.
    func = make_gemm(2048, 64, 1024, BM=2048, BN=64, tiles=1, shaves=2)
    source = tilelang.lower(func, target="intel_npu").kernel_source
    db = runner.build(source, KERNEL + "_toobig")
    a = np.zeros((2048, 1024), np.float16)
    b = np.zeros((64, 1024), np.float16)
    c = np.zeros((2048, 64), np.float16)
    try:
        runner.run(db, [("in", a), ("in", b), ("out", c)], 1, 2, HERE / "run_gemm_toobig", repeat=1, warmup=0)
    except RuntimeError as e:
        ok = "status 2 " in str(e)
        print("cmx negative control: " + ("behaved (CMX_EXHAUSTED reported by the kernel)" if ok else f"BROKEN: {e}"))
        return 0 if ok else 1
    print("cmx negative control: BROKEN: the kernel reported OK")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
