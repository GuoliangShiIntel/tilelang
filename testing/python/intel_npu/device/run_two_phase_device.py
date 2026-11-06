"""Phase 2 device golden: the two-phase megakernel (split-K partials -> T.sync_grid() -> combine)
on the PTL/NPU5 against numpy, one launch.

Usage (GENERIC_COMPUTE_ROOT and MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_two_phase_device.py [--no-build] [--profile] [--nosync] [--runs N]

--nosync builds the kernel WITHOUT the barrier: the negative control, expected to FAIL the
comparison on at least one of the --runs (default 5) launches; exit 0 means the control behaved.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import runner  # noqa: E402
import lowering  # noqa: E402
from kernels import make_gemm_two_phase  # noqa: E402

M, N, K = 256, 256, 512
TILES, SHAVES = 2, 2


def main(argv):
    nosync = "--nosync" in argv
    runs = int(argv[argv.index("--runs") + 1]) if "--runs" in argv else 5
    kernel = "tl_gemm_two_phase" + ("_nosync" if nosync else "")
    low = lowering.lower(make_gemm_two_phase(M, N, K, tiles=TILES, shaves=SHAVES, with_sync=not nosync))
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, kernel)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=kernel, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, kernel)
        print(f"build {time.time() - t0:.0f} s")

    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{kernel}_{M}x{N}x{K}_trace.json"

    rels = []
    for r in range(runs if nosync else 1):
        rng = np.random.default_rng(r)
        a = rng.standard_normal((M, K)).astype(np.float16)
        b = rng.standard_normal((N, K)).astype(np.float16)
        p = np.zeros((2, M, N), np.float16)
        c = np.zeros((M, N), np.float16)
        arrays = lowering.order_arrays(low, {"A": ("in", a), "B": ("in", b), "P": ("inout", p), "C": ("out", c)})
        res = runner.run(db, arrays, TILES, SHAVES, HERE / f"run_{kernel}", repeat=3, warmup=1, profile_json=profile)
        runner.check_status(res.status)
        ref = a.astype(np.float32) @ b.astype(np.float32).T
        rel = float(np.abs(c.astype(np.float32) - ref).max() / np.abs(ref).max())
        rels.append(rel)
        kind = "PROFILED (observability)" if profile else "UNPROFILED host-wall incl. submit"
        print(f"run-set {r}: median {res.median_ms:.3f} ms [{kind}]; rel err {rel:.2e} -> {'PASS' if rel < 4e-3 else 'FAIL'}")
    print(f"status per SHAVE: {res.status[:, :4].tolist()}; kernel sha256 {db.source_sha256[:12]}")
    if profile:
        print(f"trace: {profile}")
    if nosync:
        behaved = max(rels) >= 4e-3
        print("negative control (no sync_grid): " + ("behaved (a FAIL was observed)" if behaved else "BROKEN: every launch passed"))
        return 0 if behaved else 1
    return 0 if rels[0] < 4e-3 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
