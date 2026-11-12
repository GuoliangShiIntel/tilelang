"""Bisect from the KNOWN-GOOD make_gemm direction (not from the minimal probe): a single-iteration,
small, 3-param GEMM-shaped kernel where the STORED buffer is the DIRECT output of dpu_matmul,
matching GEMM's structure exactly minus the loop. This checks whether "store the DPU's own output"
(vs "store an untouched DMA-loaded buffer") is the actual differentiator.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402
import runner  # noqa: E402
import tilelang.intel_npu.language as T  # noqa: E402
import numpy as np  # noqa: E402

M, N, K = 64, 64, 64  # smallest 16-aligned, 1 block per worker exactly (no loop)
TILES, SHAVES = 2, 2


@T.prim_func
def probe(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float16)):
    with T.Kernel(TILES, threads=SHAVES) as tile:
        shave = T.get_thread_binding()
        worker = tile * SHAVES + shave
        A_cmx = T.alloc_local((16, K), T.float16)
        B_cmx = T.alloc_local((16, K), T.float16)
        C_cmx = T.alloc_local((16, 16), T.float16)
        T.copy(A[worker * 16, 0], A_cmx)
        T.copy(B[worker * 16, 0], B_cmx)
        T.gemm(A_cmx, B_cmx, C_cmx, transpose_B=True, clear_accum=True)
        T.copy(C_cmx, C[worker * 16, 0])


def main(argv):
    source = tilelang.lower(probe, target="intel_npu").kernel_source
    db = runner.build(source, "tl_probe_gemm_singleshot")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((N, K)).astype(np.float16)
    c = np.full((M, N), -123.0, np.float16)
    res = runner.run(db, [("in", a), ("in", b), ("out", c)], TILES, SHAVES, HERE / "run_probe_gemm_singleshot")
    runner.check_status(res.status)
    ref = a.astype(np.float32) @ b.astype(np.float32).T
    err = float(np.abs(c.astype(np.float32) - ref).max())
    rel = err / float(np.abs(ref).max())
    nonzero = int(np.count_nonzero(c))
    print(f"single-shot small gemm: rel_err={rel:.4e} nonzero={nonzero}/{c.size} verdict={'PASS' if rel < 2e-3 else 'FAIL'}")
    for w in range(4):
        blk_c = c[w * 16:(w + 1) * 16, :].astype(np.float32)
        blk_ref = ref[w * 16:(w + 1) * 16, :]
        blk_rel = float(np.abs(blk_c - blk_ref).max()) / float(np.abs(blk_ref).max())
        print(f"  worker {w} (rows {w*16}:{(w+1)*16}): nonzero={int(np.count_nonzero(blk_c))}/256 rel_err={blk_rel:.4e}")
    print("cross-check: does worker i's output match a DIFFERENT worker j's expected ref block?")
    for i in range(4):
        blk_c = c[i * 16:(i + 1) * 16, :].astype(np.float32)
        best_j, best_rel = None, 1e9
        for j in range(4):
            blk_ref = ref[j * 16:(j + 1) * 16, :]
            r = float(np.abs(blk_c - blk_ref).max()) / float(np.abs(blk_ref).max())
            if r < best_rel:
                best_rel, best_j = r, j
        print(f"  got(worker {i}) best matches ref(block {best_j}) rel={best_rel:.4e}")
    # also check: is c just literally A @ A^T instead of A @ B^T (B/A mixed up)?
    alt_ref = a.astype(np.float32) @ a.astype(np.float32).T
    alt_rel = float(np.abs(c.astype(np.float32) - alt_ref).max()) / float(np.abs(alt_ref).max())
    print(f"check vs A@A^T (B/A swap hypothesis): rel={alt_rel:.4e}")
    return 0 if rel < 2e-3 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
