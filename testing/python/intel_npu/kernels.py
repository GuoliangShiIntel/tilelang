"""Phase 1 GEMM kernel for the Intel NPU: C[M, N] = A[M, K] . B[N, K]^T in fp16.

One persistent launch over `tiles` x `shaves_per_tile` SHAVEs; the output blocks
are dealt round-robin over the workers. The DPU takes the whole K per
workload, so A and B tiles are [BM, K] and [BN, K] in CMX.

No `from __future__ import annotations`: T.prim_func evaluates the tensor
annotations against the factory's locals.
"""

import tilelang.intel_npu.language as T


def make_gemm(M: int, N: int, K: int, BM: int = 64, BN: int = 64, tiles: int = 2, shaves: int = 2):
    workers = tiles * shaves
    nblk = (M // BM) * (N // BN)
    assert M % BM == 0 and N % BN == 0

    @T.prim_func
    def gemm(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            A_cmx = T.alloc_local((BM, K), T.float16)
            B_cmx = T.alloc_local((BN, K), T.float16)
            C_cmx = T.alloc_local((BM, BN), T.float16)
            for w in T.serial(T.ceildiv(nblk, workers)):
                blk = w * workers + worker
                if blk < nblk:
                    bm = blk // (N // BN)
                    bn = blk % (N // BN)
                    T.copy(A[bm * BM, 0], A_cmx)
                    T.copy(B[bn * BN, 0], B_cmx)
                    T.gemm(A_cmx, B_cmx, C_cmx, transpose_B=True, clear_accum=True)
                    T.copy(C_cmx, C[bm * BM, bn * BN])

    return gemm


def make_gemm_ksplit(M: int, N: int, K: int, BM: int = 64, BN: int = 64, BK: int = 64):
    """GPU-style K loop with an accumulating gemm: the backend must refuse it by name."""

    @T.prim_func
    def gemm_ksplit(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(1, threads=1):
            A_cmx = T.alloc_local((BM, BK), T.float16)
            B_cmx = T.alloc_local((BN, BK), T.float16)
            C_cmx = T.alloc_local((BM, BN), T.float16)
            for k in T.serial(K // BK):
                T.copy(A[0, k * BK], A_cmx)
                T.copy(B[0, k * BK], B_cmx)
                T.gemm(A_cmx, B_cmx, C_cmx, transpose_B=True)
            T.copy(C_cmx, C[0, 0])

    return gemm_ksplit
