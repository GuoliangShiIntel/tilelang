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


def make_ring_flow(M: int, N: int, K: int, BM: int = 32, BN: int = 256, tiles: int = 3, shaves: int = 2,
                   num_stages: int = 2):
    """THE FLOW (docs/examples/pingpong_multitile/README.md 1-3) as the author writes it:
    Y[M, N] = X[M, K] . W[N, K]^T. W's fill depends on the tile -> SEGMENTED, resident in shared CMX;
    X's fill inside the pipelined loop does not -> DUPLICATED (one multicast per chunk), streamed
    through `num_stages` slots dealt round-robin to the tile's SHAVEs. The compiler derives the rest."""
    assert N == BN * tiles and M % BM == 0

    @T.prim_func
    def ring_flow(X: T.Tensor((M, K), T.float16), W: T.Tensor((N, K), T.float16), Y: T.Tensor((M, N), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            W_cmx = T.alloc_shared((BN, K), T.float16)
            X_cmx = T.alloc_shared((BM, K), T.float16)
            Y_cmx = T.alloc_local((BM, BN), T.float16)
            T.copy(W[tile * BN, 0], W_cmx)
            for c in T.Pipelined(M // BM, num_stages=num_stages):
                T.copy(X[c * BM, 0], X_cmx)
                T.gemm(X_cmx, W_cmx, Y_cmx, transpose_B=True, clear_accum=True)
                T.copy(Y_cmx, Y[c * BM, tile * BN])

    return ring_flow


def make_ring_flow_two_phase(M: int, N: int, K: int, BM: int = 32, BN: int = 256, tiles: int = 3, shaves: int = 2,
                             num_stages: int = 2):
    """Two flows one after the other against the same resident W: Y1 = X1 . W^T then Y2 = X2 . W^T.
    The second ring begins after the first has ended, so AssignBarriers gives both the same ids
    (barrier id reuse across phases); the device golden checks both outputs."""
    assert N == BN * tiles and M % BM == 0

    @T.prim_func
    def ring_flow_two_phase(X1: T.Tensor((M, K), T.float16), X2: T.Tensor((M, K), T.float16), W: T.Tensor((N, K), T.float16),
                            Y1: T.Tensor((M, N), T.float16), Y2: T.Tensor((M, N), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            W_cmx = T.alloc_shared((BN, K), T.float16)
            X_cmx = T.alloc_shared((BM, K), T.float16)
            Y_cmx = T.alloc_local((BM, BN), T.float16)
            T.copy(W[tile * BN, 0], W_cmx)
            for c in T.Pipelined(M // BM, num_stages=num_stages):
                T.copy(X1[c * BM, 0], X_cmx)
                T.gemm(X_cmx, W_cmx, Y_cmx, transpose_B=True, clear_accum=True)
                T.copy(Y_cmx, Y1[c * BM, tile * BN])
            for c in T.Pipelined(M // BM, num_stages=num_stages):
                T.copy(X2[c * BM, 0], X_cmx)
                T.gemm(X_cmx, W_cmx, Y_cmx, transpose_B=True, clear_accum=True)
                T.copy(Y_cmx, Y2[c * BM, tile * BN])

    return ring_flow_two_phase


def make_ring_flow_swapped(M: int, N: int, K: int, BM: int = 256, BN: int = 32, tiles: int = 3, shaves: int = 2,
                          num_stages: int = 2):
    """The flow with the roles swapped (activation-stationary, README section 1 "roles, not names"):
    the tile keeps its block of rows of Xa resident and streams the weight rows Wb. The gemm's
    SECOND operand is the streamed one. Different names, loops and operand order: a source the
    passes were not written for."""
    assert M == BM * tiles and N % BN == 0

    @T.prim_func
    def ring_flow_swapped(Xa: T.Tensor((M, K), T.float16), Wb: T.Tensor((N, K), T.float16), Out: T.Tensor((M, N), T.float16)):
        with T.Kernel(tiles, threads=shaves) as t:
            rows_cmx = T.alloc_shared((BM, K), T.float16)
            wstream = T.alloc_shared((BN, K), T.float16)
            acc = T.alloc_local((BM, BN), T.float16)
            T.copy(Xa[t * BM, 0], rows_cmx)
            for j in T.Pipelined(N // BN, num_stages=num_stages):
                T.copy(Wb[j * BN, 0], wstream)
                T.gemm(rows_cmx, wstream, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, Out[t * BM, j * BN])

    return ring_flow_swapped


def make_flash_attention_phase(numQ: int, numKv: int, D: int = 64, BQ: int = 32, BKV: int = 128, tiles: int = 3,
                               shaves: int = 2, num_stages: int = 2, pipelined: bool = True, vectorized: bool = False):
    """One attention phase of a megakernel, written in the flow's form (plan 3b, README 1-3):
    the tile's Q block is resident (SEGMENTED), K and V^T stream through the ring (DUPLICATED),
    chunk c -> SHAVE c % 2 keeps its own online-softmax partials (m, l, O) over its chunks, and
    the two partials are merged per tile at the end (flash-decoding's split-KV combine). Used
    first as a probe of what the passes refuse (plan section 8), then as the Phase 4 target.
    `vectorized`: the same math with T.vectorized inner loops (half32 / float16 lanes on the
    SHAVE: VAU.EXP for the fp16 exp, lane accumulators for the row max / sum)."""
    assert numQ % (BQ * tiles) == 0 and numKv % BKV == 0
    NKV, QB = numKv // BKV, numQ // (BQ * tiles)
    scale = 1.0 / (D ** 0.5)
    HL, FL = 32, 16   # fp16 and fp32 lanes of the 512-bit SHAVE vector unit
    assert BKV % HL == 0 and D % FL == 0

    @T.prim_func
    def flash(Q: T.Tensor((numQ, D), T.float16), K: T.Tensor((numKv, D), T.float16),
              Vt: T.Tensor((D, numKv), T.float16), O: T.Tensor((numQ, D), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            q_cmx = T.alloc_shared((BQ, D), T.float16)
            k_cmx = T.alloc_shared((BKV, D), T.float16)
            v_cmx = T.alloc_shared((D, BKV), T.float16)
            s_cmx = T.alloc_local((BQ, BKV), T.float16)
            pv_cmx = T.alloc_local((BQ, D), T.float16)
            o_acc = T.alloc_local((BQ, D), T.float32)     # fp32 running stats: fp16 sums of 1024 keys lose ~1%
            o_out = T.alloc_local((BQ, D), T.float16)
            m_i = T.alloc_local((BQ,), T.float32)
            l_i = T.alloc_local((BQ,), T.float32)
            m_new = T.alloc_local((BQ,), T.float32)
            corr = T.alloc_local((BQ,), T.float32)
            mx = T.alloc_local((HL,), T.float16)           # lane accumulators (vectorized path)
            ls = T.alloc_local((FL,), T.float32)
            part = T.alloc_shared((shaves, BQ, D + 2), T.float32)  # per-SHAVE (O, m, l) for the merge
            # Q blocks dealt round-robin over the tiles; each block is one pass over the whole KV
            for qb in T.serial(QB):
                T.copy(Q[(qb * tiles + tile) * BQ, 0], q_cmx)
                # T.fill is a Phase 4 tile op (refused today); the scalar loops lower
                for i, d in T.grid(BQ, D):
                    o_acc[i, d] = T.float32(0)
                for i in T.serial(BQ):
                    m_i[i] = T.float32(-1e30)
                    l_i[i] = T.float32(0)
                for kv in (T.Pipelined(NKV, num_stages=num_stages) if pipelined else T.serial(NKV)):
                    T.copy(K[kv * BKV, 0], k_cmx)
                    T.copy(Vt[0, kv * BKV], v_cmx)
                    T.gemm(q_cmx, k_cmx, s_cmx, transpose_B=True, clear_accum=True)
                    # online softmax on the SHAVE: m_new = max(m_i, rowmax(S)); P = exp(S*scale - m_new); l = l*corr + rowsum(P)
                    if vectorized:
                        for i in T.serial(BQ):
                            for v in T.vectorized(HL):
                                mx[v] = s_cmx[i, v]
                            for j0 in T.serial(1, BKV // HL):
                                for v in T.vectorized(HL):
                                    mx[v] = T.max(mx[v], s_cmx[i, j0 * HL + v])
                            m_new[i] = m_i[i]
                            for v in T.serial(HL):
                                m_new[i] = T.max(m_new[i], T.cast(mx[v], T.float32) * scale)
                            corr[i] = T.exp(m_i[i] - m_new[i])
                            # P in fp16 on the VAU: exp(s * scale - m_new), |arg| <= ~11 so fp16 holds it
                            for j0 in T.serial(BKV // HL):
                                for v in T.vectorized(HL):
                                    s_cmx[i, j0 * HL + v] = T.exp(s_cmx[i, j0 * HL + v] * T.float16(scale) - T.cast(m_new[i], T.float16))
                            for v in T.vectorized(FL):
                                ls[v] = T.float32(0)
                            for j0 in T.serial(BKV // FL):
                                for v in T.vectorized(FL):
                                    ls[v] = ls[v] + T.cast(s_cmx[i, j0 * FL + v], T.float32)
                            l_i[i] = l_i[i] * corr[i]
                            for v in T.serial(FL):
                                l_i[i] = l_i[i] + ls[v]
                            m_i[i] = m_new[i]
                    else:
                        for i in T.serial(BQ):
                            m_new[i] = m_i[i]
                            for j in T.serial(BKV):
                                m_new[i] = T.max(m_new[i], T.cast(s_cmx[i, j], T.float32) * scale)
                            corr[i] = T.exp(m_i[i] - m_new[i])
                            l_i[i] = l_i[i] * corr[i]
                            for j in T.serial(BKV):
                                p = T.exp(T.cast(s_cmx[i, j], T.float32) * scale - m_new[i])
                                s_cmx[i, j] = T.cast(p, T.float16)
                                l_i[i] = l_i[i] + p
                            m_i[i] = m_new[i]
                    T.gemm(s_cmx, v_cmx, pv_cmx, transpose_B=True, clear_accum=True)
                    if vectorized:
                        for i in T.serial(BQ):
                            for d0 in T.serial(D // FL):
                                for v in T.vectorized(FL):
                                    o_acc[i, d0 * FL + v] = o_acc[i, d0 * FL + v] * corr[i] + T.cast(pv_cmx[i, d0 * FL + v], T.float32)
                    else:
                        for i, d in T.grid(BQ, D):
                            o_acc[i, d] = o_acc[i, d] * corr[i] + T.cast(pv_cmx[i, d], T.float32)
                # merge the two SHAVEs' partials in the tile, SHAVE 0 writes the block
                for i, d in T.grid(BQ, D):
                    part[shave, i, d] = o_acc[i, d]
                for i in T.serial(BQ):
                    part[shave, i, D] = m_i[i]
                    part[shave, i, D + 1] = l_i[i]
                T.sync_threads()
                if shave == 0:
                    for i in T.serial(BQ):
                        m_new[i] = T.max(part[0, i, D], part[1, i, D])
                        corr[i] = T.exp(part[0, i, D] - m_new[i])
                        l_i[i] = T.exp(part[1, i, D] - m_new[i])
                        for d in T.serial(D):
                            o_out[i, d] = T.cast((part[0, i, d] * corr[i] + part[1, i, d] * l_i[i]) /
                                                 (part[0, i, D + 1] * corr[i] + part[1, i, D + 1] * l_i[i]), T.float16)
                    T.copy(o_out, O[(qb * tiles + tile) * BQ, 0])

    return flash


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
