"""Bisection ladder for the all-zero-output DMA bug found via probe_dma_passthrough_device.py.
Runs several minimal variants (tile/shave count, buffer size) to find exactly which factor matters.
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


def make_probe(N, W, tiles, shaves, rows_per_worker):
    @T.prim_func
    def probe(Q: T.Tensor((N, W), T.float16), O: T.Tensor((N, W), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            q_cmx = T.alloc_local((rows_per_worker, W), T.float16)
            T.copy(Q[worker * rows_per_worker, 0], q_cmx)
            T.copy(q_cmx, O[worker * rows_per_worker, 0])

    return probe


def run_one(name, N, W, tiles, shaves, rows_per_worker):
    func = make_probe(N, W, tiles, shaves, rows_per_worker)
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        source = tilelang.lower(func, target="intel_npu").kernel_source
    db = runner.build(source, name)
    q = np.random.default_rng(5).standard_normal((N, W)).astype(np.float16)
    o = np.full((N, W), -123.0, np.float16)
    res = runner.run(db, [("in", q), ("out", o)], tiles, shaves, HERE / f"run_{name}")
    runner.check_status(res.status)
    match = bool(np.array_equal(q, o))
    nonzero = int(np.count_nonzero(o))
    print(f"{name}: N={N} W={W} tiles={tiles} shaves={shaves} rpw={rows_per_worker} "
         f"-> exact_match={match} nonzero_elems={nonzero}/{o.size}")
    return match


def main(argv):
    results = {}
    # 1. single worker, tiny shape (isolates multi-tile/shave addressing entirely)
    results["1x1_tiny"] = run_one("tl_probe_1x1_tiny", 4, 64, 1, 1, 4)
    # 2. 2x2 workers, tiny shape (the exact failing config)
    results["2x2_tiny"] = run_one("tl_probe_2x2_tiny", 8, 64, 2, 2, 4)
    # 3. 2x2 workers, GEMM-like LARGE shape (same worker config, bigger buffers)
    results["2x2_large"] = run_one("tl_probe_2x2_large", 256, 512, 2, 2, 128)
    # 4. 1 tile, 2 shaves, tiny shape (isolates "shave" dimension specifically)
    results["1x2_tiny"] = run_one("tl_probe_1x2_tiny", 8, 64, 1, 2, 4)
    # 5. 2 tiles, 1 shave, tiny shape (isolates "tile" dimension specifically)
    results["2x1_tiny"] = run_one("tl_probe_2x1_tiny", 8, 64, 2, 1, 4)
    print("\nSUMMARY:", results)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
