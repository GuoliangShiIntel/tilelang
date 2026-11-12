"""Minimal probe: pure DMA load+store, no DPU, no compute at all, at the exact tiny shape that gave
all-zero output in make_gemma4_attn_prefill on real NVL/NPU6 device. Isolates whether basic DMA
load/store itself is the problem, or whether it's specific to the DPU/softmax compute path.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402
import tvm  # noqa: E402
import runner  # noqa: E402
import tilelang.intel_npu.language as T  # noqa: E402
import numpy as np  # noqa: E402

N = 8
KERNEL = "tl_probe_dma_passthrough"


@T.prim_func
def probe(Q: T.Tensor((N, 64), T.float16), O: T.Tensor((N, 64), T.float16)):
    with T.Kernel(2, threads=2) as tile:
        shave = T.get_thread_binding()
        worker = tile * 2 + shave
        q_cmx = T.alloc_local((4, 64), T.float16)
        T.copy(Q[worker * 4, 0], q_cmx)
        T.copy(q_cmx, O[worker * 4, 0])


def main(argv):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        source = tilelang.lower(probe, target="intel_npu").kernel_source
    if "--no-build" in argv:
        path, sha = runner.write_kernel(source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(source, KERNEL)
        print(f"build {time.time() - t0:.0f} s")

    q = np.random.default_rng(5).standard_normal((N, 64)).astype(np.float16)
    o = np.full((N, 64), -123.0, np.float16)
    res = runner.run(db, [("in", q), ("out", o)], 2, 2, HERE / "run_probe_dma_passthrough")
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    match = bool(np.array_equal(q, o))
    print(f"exact match: {match}")
    print(f"q row0: {q[0, :8]}")
    print(f"o row0: {o[0, :8]}")
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
