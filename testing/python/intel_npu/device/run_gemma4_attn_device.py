"""Gemma4 V5: make_gemma4_attn_prefill (the gate-less, sliding-window flash attention phase) on REAL
NVL/NPU6 hardware, at real-ish Gemma4 sliding-attention dimensions (head_dim=256, 8 kv heads, 2 query
heads per kv group). A small handful of DDR parameters (Q, Kc, Vt, Pos, O) -- kept deliberately small
component-level, unlike the fused 61-parameter whole-layer kernel in run_gemma4_layer_device.py, which
hits a host-harness parameter-count limit on launch (STATUS_STACK_BUFFER_OVERRUN, host-side, not a
device/kernel-correctness issue -- see docs/developer_guide/gemma4_enablement_plan.md's V5 section).
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
import tvm  # noqa: E402
import runner  # noqa: E402
from gemma4_kernels import gemma4_attn_prefill_reference, make_gemma4_attn_prefill  # noqa: E402

N, NKV, GQ, D, SMAX = 32, 8, 2, 256, 512
TB, BKV, WINDOW = 8, 128, 1024
TILES, SHAVES = 4, 2
KERNEL = "tl_gemma4_attn_real_dims"


def main(argv):
    func = make_gemma4_attn_prefill(N=N, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, window=WINDOW, TB=TB, BKV=BKV,
                                    tiles=TILES, shaves=SHAVES)
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.hw_sync_grid": False}):
        source = tilelang.lower(func, target="intel_npu").kernel_source
    if "--no-build" in argv:
        path, sha = runner.write_kernel(source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(source, KERNEL)
        print(f"build {time.time() - t0:.0f} s")

    rng = np.random.default_rng(3)
    pos0 = 40
    q = rng.standard_normal((NKV, N, GQ, D), np.float32).astype(np.float16)
    kc = np.zeros((NKV, SMAX, D), np.float16)
    vt = np.zeros((NKV, D, SMAX), np.float16)
    last = pos0 + N
    kc[:, :last] = rng.standard_normal((NKV, last, D), np.float32).astype(np.float16)
    vt[:, :, :last] = rng.standard_normal((NKV, D, last), np.float32).astype(np.float16)
    o = np.zeros((N, NKV * GQ * D), np.float16)

    res = runner.run(db, [("in", q), ("in", kc), ("in", vt), ("in", np.array([[pos0]], np.int32)), ("out", o)],
                     TILES, SHAVES, HERE / "run_gemma4_attn")
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)

    ref = gemma4_attn_prefill_reference(q, kc, vt, pos0, window=WINDOW)
    rel = float(np.abs(o.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-6))
    verdict = "PASS" if rel < 3e-2 else "FAIL"
    print(f"device (REAL NVL/NPU6): median {res.median_ms:.3f} ms [UNPROFILED host-wall incl. submit]; "
         f"rel_err={rel:.4f} -> {verdict}; kernel sha256 {db.source_sha256[:12]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
