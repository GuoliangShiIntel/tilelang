"""Does ONE DPU workload really cap at 8192 output points on NPU5?

`LowerIntelNpuTileOp` refuses a gemm whose output tile has M x N > `dpu_max_output_points` (8192), and
`hw_facts.py` records that bound as the PRODUCT of a workload's output. npu_dsl contradicts it: its
device-verified prefill GDN layer issues `ctx.matmul(dS, kdT, vnT, 128, 128, 64)` and
`ctx.matmul_async(ST, eye, S16, 128, 128, 128)` as single workloads, i.e. 16384 output values each
(`kernels_poc/qwen36_dsl_poc/generated/qwen36_prefill_gdn_layer.cpp`). One of the two is wrong, and the
answer decides whether the prefill scan has to split its state matmuls.

So: raise the target bound and run the shapes against numpy on the device. The point is NOT that a big
gemm is fast -- it is whether the DPU computes it CORRECTLY. Each shape is its own launch, so one bad
shape cannot be blamed on another, and a shape at the default 8192 is included as the positive control.

Usage:  python testing/python/intel_npu/device/run_dpu_points_probe.py [--max-points 65536]
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401
import tilelang.language as T  # noqa: E402
import runner  # noqa: E402
import lowering  # noqa: E402
from tilelang.intel_npu.target import make_intel_npu_target  # noqa: E402

# (M, N, K, what it tests)
SHAPES = [
    (64, 128, 64, "8192 = the default bound: positive control, must pass whatever the answer is"),
    (128, 128, 64, "16384 as a product; npu_dsl issues exactly this one (dS = kd^T . v_new)"),
    (128, 128, 128, "16384; npu_dsl's state transpose (S^T = I . S16)"),
    (256, 64, 64, "16384 with M = 256: is the bound on the ROWS rather than the product?"),
    (64, 256, 64, "16384 with N = 256: is the bound on the CHANNELS?"),
    (128, 512, 64, "65536: far past any reading of the bound"),
]


def make(M, N, K):

    @T.prim_func
    def probe(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(1, threads=1) as tile:
            a = T.alloc_local((M, K), T.float16)
            b = T.alloc_local((N, K), T.float16)
            c = T.alloc_local((M, N), T.float16)
            T.copy(A[0, 0], a)
            T.copy(B[0, 0], b)
            T.gemm(a, b, c, transpose_B=True, clear_accum=True)
            T.copy(c, C[0, 0])

    return probe


def main(argv):
    cap = int(argv[argv.index("--max-points") + 1]) if "--max-points" in argv else 65536
    target = make_intel_npu_target(dpu_max_output_points=cap)
    rng = np.random.default_rng(0)
    rows = []
    for M, N, K in [(m, n, k) for m, n, k, _ in SHAPES]:
        why = next(w for m, n, k, w in SHAPES if (m, n, k) == (M, N, K))
        try:
            low = lowering.lower(make(M, N, K), target=target)
        except Exception as e:                                   # noqa: BLE001
            rows.append((M, N, K, "REFUSED", f"{e}"[:80], why))
            continue
        a = (rng.standard_normal((M, K), np.float32) * 0.5).astype(np.float16)
        b = (rng.standard_normal((N, K), np.float32) * 0.5).astype(np.float16)
        c = np.zeros((M, N), np.float16)
        db = runner.build(low.source, f"tl_dpu_points_{M}x{N}x{K}")
        res = runner.run(db, lowering.order_arrays(low, {"A": ("in", a), "B": ("in", b), "C": ("out", c)}),
                         1, 1, HERE / "run_dpu_points", repeat=3, warmup=1, timeout_s=120)
        runner.check_status(res.status)
        ref = a.astype(np.float32) @ b.astype(np.float32).T
        rel = float(np.abs(c.astype(np.float32) - ref).max() / np.abs(ref).max())
        # a wrong-but-plausible result is the danger here, so also count how many outputs are badly off
        bad = int((np.abs(c.astype(np.float32) - ref) > 0.05 * np.abs(ref).max()).sum())
        rows.append((M, N, K, "PASS" if rel < 2e-2 else "WRONG", f"rel {rel:.2e}, {bad}/{M * N} outputs off", why))

    print(f"\ndpu_max_output_points raised to {cap}\n")
    print(f"{'M':>5} {'N':>5} {'K':>5} {'points':>7}  {'verdict':<8} {'detail':<34} why")
    for M, N, K, verdict, detail, why in rows:
        print(f"{M:5} {N:5} {K:5} {M * N:7}  {verdict:<8} {detail:<34} {why}")
    return 0 if all(r[3] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
