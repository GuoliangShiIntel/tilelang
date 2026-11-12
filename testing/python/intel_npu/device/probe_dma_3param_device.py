"""Test whether the all-zero bug is actually a parameter-count artifact: all failing probes had
exactly 2 DDR params (Q, O); the working GEMM kernel has 3 (A, B, C). Add a dummy unused 3rd DDR
param to the minimal passthrough probe and see if that alone changes the outcome.
"""
import sys
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

N, W = 8, 64


@T.prim_func
def probe(Q: T.Tensor((N, W), T.float16), Dummy: T.Tensor((N, W), T.float16), O: T.Tensor((N, W), T.float16)):
    with T.Kernel(2, threads=2) as tile:
        shave = T.get_thread_binding()
        worker = tile * 2 + shave
        q_cmx = T.alloc_local((4, W), T.float16)
        T.copy(Q[worker * 4, 0], q_cmx)
        T.copy(q_cmx, O[worker * 4, 0])


def main(argv):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        source = tilelang.lower(probe, target="intel_npu").kernel_source
    db = runner.build(source, "tl_probe_3param")
    q = np.random.default_rng(5).standard_normal((N, W)).astype(np.float16)
    d = np.zeros((N, W), np.float16)
    o = np.full((N, W), -123.0, np.float16)
    res = runner.run(db, [("in", q), ("in", d), ("out", o)], 2, 2, HERE / "run_probe_3param")
    runner.check_status(res.status)
    match = bool(np.array_equal(q, o))
    nonzero = int(np.count_nonzero(o))
    print(f"3param probe: exact_match={match} nonzero_elems={nonzero}/{o.size}")
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
