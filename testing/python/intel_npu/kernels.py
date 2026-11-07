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


def make_gemm_repeat(BM: int, BN: int, K: int, repeat: int, tiles: int = 1, shaves: int = 2):
    """DPU rate probe: every worker loads one [BM, K] x [BN, K] block pair once and runs the
    same workload `repeat` times before storing. The difference between two repeat counts
    isolates the DPU time per workload from launch, DMA and store costs; the result is the
    same for any repeat, so the golden still applies. Used to calibrate VPUNN (plan 3.2)."""
    workers = tiles * shaves
    M, N = BM * workers, BN

    @T.prim_func
    def gemm_repeat(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            A_cmx = T.alloc_local((BM, K), T.float16)
            B_cmx = T.alloc_local((BN, K), T.float16)
            C_cmx = T.alloc_local((BM, BN), T.float16)
            T.copy(A[worker * BM, 0], A_cmx)
            T.copy(B[0, 0], B_cmx)
            for _ in T.serial(repeat):
                T.gemm(A_cmx, B_cmx, C_cmx, transpose_B=True, clear_accum=True)
            T.copy(C_cmx, C[worker * BM, 0])

    return gemm_repeat


def make_gemm_two_phase(M: int, N: int, K: int, BM: int = 64, BN: int = 64, tiles: int = 2, shaves: int = 2,
                        with_sync: bool = True):
    """Megakernel shape: split-K partial products -> T.sync_grid() -> combine, one launch.

    Phase A runs on the first half of the workers (the producers) and writes
    partial products to DDR scratch P[half]; phase B deals output blocks over
    ALL workers. The phases have different parallelism, as megakernel phases
    do, and without the barrier the non-producer workers read partials that
    have not been written yet, so `with_sync=False` is a negative control that
    fails without depending on scheduling luck.
    """
    workers = tiles * shaves
    producers = max(1, workers // 2)
    nbm, nbn = M // BM, N // BN
    nblk = nbm * nbn
    KH = K // 2
    assert M % BM == 0 and N % BN == 0 and K % 2 == 0

    @T.prim_func
    def gemm_two_phase(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16),
                       P: T.Tensor((2, M, N), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            A_cmx = T.alloc_local((BM, KH), T.float16)
            B_cmx = T.alloc_local((BN, KH), T.float16)
            P_cmx = T.alloc_local((BM, BN), T.float16)
            Q_cmx = T.alloc_local((BM, BN), T.float16)
            # Phase A: 2 * nblk items (block, k-half) over the producer workers only
            for w in T.serial(T.ceildiv(2 * nblk, producers)):
                item = w * producers + worker
                if worker < producers and item < 2 * nblk:
                    kh = item % 2
                    blk = item // 2
                    bm = blk // nbn
                    bn = blk % nbn
                    T.copy(A[bm * BM, kh * KH], A_cmx)
                    T.copy(B[bn * BN, kh * KH], B_cmx)
                    T.gemm(A_cmx, B_cmx, P_cmx, transpose_B=True, clear_accum=True)
                    T.copy(P_cmx, P[kh, bm * BM, bn * BN])
            if with_sync:
                T.sync_grid()
            # Phase B: output blocks over every worker
            for w in T.serial(T.ceildiv(nblk, workers)):
                blk = w * workers + worker
                if blk < nblk:
                    bm = blk // nbn
                    bn = blk % nbn
                    T.copy(P[0, bm * BM, bn * BN], P_cmx)
                    T.copy(P[1, bm * BM, bn * BN], Q_cmx)
                    for i, j in T.grid(BM, BN):
                        P_cmx[i, j] = P_cmx[i, j] + Q_cmx[i, j]
                    T.copy(P_cmx, C[bm * BM, bn * BN])

    return gemm_two_phase


def make_sync_chain(rounds: int, tiles: int = 3, shaves: int = 2):
    """A chain of `rounds` grid barriers with a token passed around the workers through DDR.

    Round r: worker w writes T[r+1, w] = T[r, (w+1) % W] + 1, then everyone syncs. After the
    chain, T[rounds, w] == rounds for every w, and a single missed or racing barrier breaks
    the value. Measures sync_grid cost (rounds barriers per launch) and checks correctness.

    Each worker's token sits in its own 16-byte slot: the SHAVE data L1 is write-back with
    16-byte lines, so two SHAVEs storing scalars into one line clobber each other's bytes.
    """
    W = tiles * shaves
    LINE = 4  # int32 per 16-byte DL1 line

    @T.prim_func
    def sync_chain(Tok: T.Tensor((rounds + 1, W * LINE), T.int32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            for r in T.serial(rounds):
                Tok[r + 1, worker * LINE] = Tok[r, ((worker + 1) % W) * LINE] + 1
                T.sync_grid()

    return sync_chain
