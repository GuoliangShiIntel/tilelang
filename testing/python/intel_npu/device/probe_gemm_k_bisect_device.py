"""Single-variable bisection from the KNOWN-GOOD make_gemm config (BM=BN=64, K=512, tiles=2,
shaves=2): shrink ONLY K to 64 (minimum 16-aligned), keeping BM=BN=64 identical to the working
config. Isolates whether the output tile footprint (BM,BN=64) or the K depth is what matters.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402
import runner  # noqa: E402
from kernels import make_gemm  # noqa: E402
import numpy as np  # noqa: E402

BM, BN = 64, 64
TILES, SHAVES = 2, 2


def run_shape(K, M=None, N=None, name=None, BM=64, BN=64):
    M = M or BM * TILES * SHAVES
    N = N or BN
    func = make_gemm(M, N, K, BM=BM, BN=BN, tiles=TILES, shaves=SHAVES)
    source = tilelang.lower(func, target="intel_npu").kernel_source
    db = runner.build(source, name)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((N, K)).astype(np.float16)
    c = np.full((M, N), -123.0, np.float16)
    res = runner.run(db, [("in", a), ("in", b), ("out", c)], TILES, SHAVES, HERE / f"run_{name}")
    runner.check_status(res.status)
    ref = a.astype(np.float32) @ b.astype(np.float32).T
    err = float(np.abs(c.astype(np.float32) - ref).max())
    rel = err / float(np.abs(ref).max())
    nonzero = int(np.count_nonzero(c))
    print(f"{name}: M={M} N={N} K={K} BM={BM} BN={BN} -> rel_err={rel:.4e} nonzero={nonzero}/{c.size} "
         f"verdict={'PASS' if rel < 2e-3 else 'FAIL'}")
    return rel < 2e-3


def main(argv):
    results = {}
    results["BM32"] = run_shape(64, name="tl_bisect_bm32", BM=32, BN=32)
    results["BM48"] = run_shape(64, name="tl_bisect_bm48", BM=48, BN=48)
    results["BM16"] = run_shape(64, name="tl_bisect_bm16", BM=16, BN=16)
    results["BM64_N16"] = run_shape(64, name="tl_bisect_bm64_n16", BM=64, BN=16)
    results["BM16_N64"] = run_shape(64, name="tl_bisect_bm16_n64", BM=16, BN=64)
    print("\nSUMMARY:", results)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
