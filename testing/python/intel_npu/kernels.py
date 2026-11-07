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
                               shaves: int = 2, num_stages: int = 2, pipelined: bool = True, vectorized: bool = False,
                               probe_skip_shave: bool = False, kv_offset_by_qb: bool = False):
    """One attention phase of a megakernel, written in the flow's form (plan 3b, README 1-3):
    the tile's Q block is resident (SEGMENTED), K and V^T stream through the ring (DUPLICATED),
    chunk c -> SHAVE c % 2 keeps its own online-softmax partials (m, l, O) over its chunks, and
    the two partials are merged per tile at the end (flash-decoding's split-KV combine). Used
    first as a probe of what the passes refuse (plan section 8), then as the Phase 4 target.
    `vectorized`: the same math with T.vectorized inner loops (half32 / float16 lanes on the
    SHAVE: VAU.EXP for the fp16 exp, lane accumulators for the row max / sum).
    `probe_skip_shave`: MEASUREMENT ARM ONLY -- no SHAVE math between the gemms (the output is
    wrong by design); what remains is the DPU + ring + fill floor the softmax is added to.
    `kv_offset_by_qb`: TEST ARM -- the K/V fills read (kv + qb) % NKV, so they depend on the Q-block
    loop; the ring is still hoisted (M3: a fill ahead reads the chunk's own iteration; output wrong)."""
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
            part = T.alloc_shared((shaves, BQ, D + FL), T.float32)  # per-SHAVE (O, m, l) for the merge; row stride a vector multiple
            # Q blocks dealt round-robin over the tiles; each block is one pass over the whole KV
            for qb in T.serial(QB):
                T.copy(Q[(qb * tiles + tile) * BQ, 0], q_cmx)
                # T.fill is a Phase 4 tile op (refused today); the scalar loops lower
                if vectorized:
                    for i, d0 in T.grid(BQ, D // FL):
                        for v in T.vectorized(FL):
                            o_acc[i, d0 * FL + v] = T.float32(0)
                else:
                    for i, d in T.grid(BQ, D):
                        o_acc[i, d] = T.float32(0)
                for i in T.serial(BQ):
                    m_i[i] = T.float32(-1e30)
                    l_i[i] = T.float32(0)
                for kv in (T.Pipelined(NKV, num_stages=num_stages) if pipelined else T.serial(NKV)):
                    if kv_offset_by_qb:
                        T.copy(K[((kv + qb) % NKV) * BKV, 0], k_cmx)
                        T.copy(Vt[0, ((kv + qb) % NKV) * BKV], v_cmx)
                    else:
                        T.copy(K[kv * BKV, 0], k_cmx)
                        T.copy(Vt[0, kv * BKV], v_cmx)
                    T.gemm(q_cmx, k_cmx, s_cmx, transpose_B=True, clear_accum=True)
                    # online softmax on the SHAVE: m_new = max(m_i, rowmax(S)); P = exp(S*scale - m_new); l = l*corr + rowsum(P)
                    if probe_skip_shave:
                        pass
                    elif vectorized:
                        for i in T.serial(BQ):
                            for v in T.vectorized(HL):
                                mx[v] = s_cmx[i, v]
                            for j0 in T.serial(1, BKV // HL):
                                for v in T.vectorized(HL):
                                    mx[v] = T.max(mx[v], s_cmx[i, j0 * HL + v])
                            # whole-vector lane reductions (runtime library ops on the CMU: log2(lanes) rotate-and-combine);
                            # scale once (max is monotone, scale > 0)
                            m_new[i] = T.max(m_i[i], T.cast(T.call_extern(T.float16, "tl_npu_vmax", mx[0:HL]), T.float32) * scale)
                            corr[i] = T.exp(m_i[i] - m_new[i])
                            # P in fp16 on the VAU: exp(s * scale - m_new), |arg| <= ~11 so fp16 holds it
                            for j0 in T.serial(BKV // HL):
                                for v in T.vectorized(HL):
                                    s_cmx[i, j0 * HL + v] = T.exp(s_cmx[i, j0 * HL + v] * T.float16(scale) - T.cast(m_new[i], T.float16))
                            for v in T.vectorized(FL):
                                ls[v] = T.cast(s_cmx[i, v], T.float32)
                            for j0 in T.serial(1, BKV // FL):
                                for v in T.vectorized(FL):
                                    ls[v] = ls[v] + T.cast(s_cmx[i, j0 * FL + v], T.float32)
                            l_i[i] = l_i[i] * corr[i] + T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL])
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
                    if probe_skip_shave:
                        pass
                    elif vectorized:
                        for i in T.serial(BQ):
                            for d0 in T.serial(D // FL):
                                for v in T.vectorized(FL):
                                    o_acc[i, d0 * FL + v] = o_acc[i, d0 * FL + v] * corr[i] + T.cast(pv_cmx[i, d0 * FL + v], T.float32)
                    else:
                        for i, d in T.grid(BQ, D):
                            o_acc[i, d] = o_acc[i, d] * corr[i] + T.cast(pv_cmx[i, d], T.float32)
                # merge the two SHAVEs' partials in the tile, SHAVE 0 writes the block
                if vectorized:
                    for i, d0 in T.grid(BQ, D // FL):
                        for v in T.vectorized(FL):
                            part[shave, i, d0 * FL + v] = o_acc[i, d0 * FL + v]
                else:
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
                        if vectorized:
                            # the row's two weights over the merged denominator, then one fused vector update
                            m_i[i] = T.float32(1) / (part[0, i, D + 1] * corr[i] + part[1, i, D + 1] * l_i[i])
                            corr[i] = corr[i] * m_i[i]
                            l_i[i] = l_i[i] * m_i[i]
                            for d0 in T.serial(D // FL):
                                for v in T.vectorized(FL):
                                    o_out[i, d0 * FL + v] = T.cast(part[0, i, d0 * FL + v] * corr[i] + part[1, i, d0 * FL + v] * l_i[i], T.float16)
                        else:
                            for d in T.serial(D):
                                o_out[i, d] = T.cast((part[0, i, d] * corr[i] + part[1, i, d] * l_i[i]) /
                                                     (part[0, i, D + 1] * corr[i] + part[1, i, D + 1] * l_i[i]), T.float16)
                    T.copy(o_out, O[(qb * tiles + tile) * BQ, 0])

    return flash


def make_w4a16_flow(M: int, N: int, K: int, BN: int = 64, G: int = 128, tiles: int = 3, shaves: int = 2,
                    num_stages: int = 2, pipelined: bool = True, split: str = "m"):
    """M2: Y[M, N] = X[M, K] . dequant(Wq[N, K])^T with symmetric signed int4 weights, two per byte (low
    nibble first), one fp16 scale per group of G along K (npu_dsl's DEQUANT_I4 / W4A16 format).

    Two role assignments (README section 1: roles, not names):
      split="m": the activation rows are resident and SEGMENTED over the tiles (M / tiles rows each); the packed
        weight chunk and its scales stream through two multicast rings (read from DDR once) and EVERY tile
        dequantizes every chunk -- right when M is large (the DPU work per chunk grows with M, the dequant does not);
      split="n": X is resident and tile-invariant (broadcast, one copy per tile); the weight chunks are split across
        the tiles (each tile streams its own third: a SEGMENTED stream, unicast fills) so the dequant is done once
        -- right when M is small (a short prompt, the W4A16 probe's shape).
    Each SHAVE dequantizes its chunk into a private fp16 buffer with the runtime's int4 row op (LSU int4 -> int8
    convert-load) and runs the DPU on it; the slots' last reader is the SHAVE, so it produces the ring's FREE."""
    assert N % BN == 0 and K % 512 == 0 and G % 64 == 0 and K % G == 0 and split in ("m", "n")
    KG = K // G
    if split == "m":
        assert M % tiles == 0
        BM, chunks = M // tiles, N // BN
    else:
        assert (N // BN) % tiles == 0
        BM, chunks = M, N // BN // tiles

    @T.prim_func
    def w4a16(X: T.Tensor((M, K), T.float16), Wq: T.Tensor((N, K // 2), T.uint8), S: T.Tensor((N, KG), T.float16),
              Y: T.Tensor((M, N), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            x_cmx = T.alloc_shared((BM, K), T.float16)
            wq_cmx = T.alloc_shared((BN, K // 2), T.uint8)
            s_cmx = T.alloc_shared((BN, KG), T.float16)
            w_cmx = T.alloc_local((BN, K), T.float16)
            y_cmx = T.alloc_local((BM, BN), T.float16)
            if split == "m":
                T.copy(X[tile * BM, 0], x_cmx)
            else:
                T.copy(X[0, 0], x_cmx)
            for n in (T.Pipelined(chunks, num_stages=num_stages) if pipelined else T.serial(chunks)):
                if split == "m":
                    T.copy(Wq[n * BN, 0], wq_cmx)
                    T.copy(S[n * BN, 0], s_cmx)
                else:
                    T.copy(Wq[(tile * chunks + n) * BN, 0], wq_cmx)
                    T.copy(S[(tile * chunks + n) * BN, 0], s_cmx)
                # dequant: a row at a time through the runtime's int4 row op (LSU convert-load, loads of a 512-value
                # block issued before its stores); the statement forms of the same math ran at 0.7 cycles per value
                for r in T.serial(BN):
                    T.call_extern("handle", "tl_npu_dequant_i4_row", T.address_of(w_cmx[r, 0]), T.address_of(wq_cmx[r, 0]),
                                  T.address_of(s_cmx[r, 0]), K, G)
                T.gemm(x_cmx, w_cmx, y_cmx, transpose_B=True, clear_accum=True)
                if split == "m":
                    T.copy(y_cmx, Y[tile * BM, n * BN])
                else:
                    T.copy(y_cmx, Y[0, (tile * chunks + n) * BN])

    return w4a16


def make_w4a16_gemv(N: int, K: int, BN: int = 256, G: int = 128, tiles: int = 3, shaves: int = 2, num_stages: int = 2,
                    pipelined: bool = True, M: int = 1, rows_per_workload: int = 64, bits: int = 4):
    """M2 decode: y[M, N] = x[M, K] . dequant(Wq[N, K])^T at M = 1 (a few rows at most) through the DPU's int4-weight
    mode (`T.gemm_i4`: VPUX's merged FC, no SHAVE dequantisation). x is resident on every tile; the weight rows are
    split over the tiles and stream through two rings (packed bytes, scales) the DPU reads in place, in chunks of BN
    rows (one DMA job each: decode is bound by weight bytes, and the DMA wants few big jobs), each consumed by
    BN / rows_per_workload DPU workloads (rows_per_workload x K / G <= 1024 channels). `bits=8`: int8 weights
    (`T.gemm_i8`, one byte per value; the lm_head's format)."""
    assert N % (BN * tiles) == 0 and K % G == 0 and BN % rows_per_workload == 0 and rows_per_workload % 16 == 0
    assert rows_per_workload * (K // G) <= 1024 and bits in (4, 8)
    chunks = N // BN // tiles
    KG = K // G
    R = rows_per_workload
    KB = K // 2 if bits == 4 else K
    wdtype = T.uint8 if bits == 4 else T.int8

    @T.prim_func
    def gemv(X: T.Tensor((M, K), T.float16), Wq: T.Tensor((N, KB), wdtype), S: T.Tensor((N, KG), T.float16),
             Y: T.Tensor((M, N), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            x_cmx = T.alloc_shared((M, K), T.float16)
            wq_cmx = T.alloc_shared((BN, KB), wdtype)
            s_cmx = T.alloc_shared((BN, KG), T.float16)
            y_cmx = T.alloc_local((M, BN), T.float16)
            T.copy(X[0, 0], x_cmx)
            for n in (T.Pipelined(chunks, num_stages=num_stages) if pipelined else T.serial(chunks)):
                T.copy(Wq[(tile * chunks + n) * BN, 0], wq_cmx)
                T.copy(S[(tile * chunks + n) * BN, 0], s_cmx)
                for sub in T.serial(BN // R):
                    if bits == 4:
                        T.gemm_i4(x_cmx, wq_cmx, s_cmx, y_cmx, group=G, row0=sub * R, rows=R)
                    else:
                        T.gemm_i8(x_cmx, wq_cmx, s_cmx, y_cmx, group=G, row0=sub * R, rows=R)
                T.copy(y_cmx, Y[0, (tile * chunks + n) * BN])

    return gemv


def make_moe_decode(H: int = 2048, I: int = 512, E: int = 256, TOPK: int = 8, G: int = 128, tiles: int = 3,
                    shaves: int = 2, num_stages: int = 4, BNR: int = 16, BNG: int = 128, BND: int = 256, RG: int = 64,
                    RD: int = 256, pipelined: bool = True, probe_skip_dpu: bool = False):
    """M3 decode: one token through a routed + shared MoE layer in ONE launch (Qwen3.6's shape by default:
    hidden H, E experts of SwiGLU inter I, top-k softmax routing renormalised over the chosen k, a shared expert
    of the same shape gated by sigmoid(router row E)).

      out = res + sum_k wts[k] * down_e(silu(gate_e(x)) * up_e(x)) + sigmoid(l[E]) * down_E(silu(gate_E(x)) * up_E(x))

    Packing (the host's job, `pack_moe`): the shared expert is expert E of the same int4 g128 tensors; each
    projection is stored as CHUNK BLOCKS, one DMA each: BN packed weight rows then the chunk's group-major fp16
    scales [K/G, BN] in whole rows (`WguP[(E + 1) * CG * (BNG + SRG), H / 2]`, `WdP[(E + 1) * CD * (BND + SRD), I / 2]`;
    the DPU path reads the scales in place, and one job per chunk instead of two halves the engine's fixed
    per-job cost); the router's E + 1 fp16 rows padded with zeros to RR = a multiple of tiles x shaves x BNR. Phases:
      1. router: x (padded to 16 rows for the DPU) resident on every tile; the router rows split over the tiles
         and streamed (SEGMENTED ring, fp16 gemm, submit-ahead); logits land in DDR, then T.sync_grid();
      2. routing: every tile reads the logits and SHAVE 0 picks the top-k (lowest id on a tie) and the
         renormalised softmax weights, plus the shared expert's sigmoid gate -- redundantly, so no second grid
         barrier; ids / wts sit in tile-shared CMX;
      3. experts: the TOPK + 1 items dealt over the tiles (IPT each); per item the gate/up GEMV streams the
         expert's int4 rows through a ring whose fill source reads `ids[...]` (an indirect, tile-split stream
         the DPU reads in place, `T.gemm_i4`), SwiGLU on the SHAVEs into h, the down GEMV likewise into the item's
         SHAVE accumulates its half of H x wts into the tile's fp32 partial. Both rings are hoisted over the item
         loop (the fill ahead reads the next item's expert: the ring never drains between experts), are LOCAL
         (split + SHAVE-read: no hardware barriers), and with num_stages = 4 each SHAVE owns two slots whose
         fills alternate the DMA ports (one job in flight per port). Measured on PTL (2026-11-08, TURBO): 128/256-row
         chunks at two slots per SHAVE 0.551 ms vs 256/512-row chunks at one slot 0.729 ms;
      4. combine: the tiles' partials meet in DDR (T.sync_grid()); tile 0 adds them to the residual.
    `probe_skip_dpu`: MEASUREMENT ARM ONLY -- the expert GEMVs are not run (output wrong by design): what remains
    is the rings' DMA + SHAVE floor of the expert phase.
    """
    items = TOPK + 1
    assert items % tiles == 0 and H % G == 0 and I % G == 0 and (2 * I) % BNG == 0 and H % BND == 0
    assert BNG % RG == 0 and BND % RD == 0 and RG * (H // G) <= 1024 and RD * (I // G) <= 1024
    assert I % shaves == 0 and H % shaves == 0 and BNR % 16 == 0
    IPT = items // tiles
    per_round = tiles * shaves * BNR
    RR = ((E + 1 + per_round - 1) // per_round) * per_round
    CR = RR // (tiles * BNR)          # router chunks per tile
    CG, CD = (2 * I) // BNG, H // BND   # gate/up and down chunks per item
    HG, IG = H // G, I // G
    MR = 16                           # the router gemm's M: x padded to the DPU's row granule
    SRG, SRD = moe_scale_rows(H, I, G, BNG, BND)   # rows of the chunk block holding the scales
    BG, BD = BNG + SRG, BND + SRD                  # rows per chunk block

    @T.prim_func
    def moe(X: T.Tensor((1, H), T.float16), Wr: T.Tensor((RR, H), T.float16),
            WguP: T.Tensor(((E + 1) * CG * BG, H // 2), T.uint8), WdP: T.Tensor(((E + 1) * CD * BD, I // 2), T.uint8),
            Res: T.Tensor((1, H), T.float16), Logits: T.Tensor((MR, RR), T.float16),
            Part: T.Tensor((tiles, H), T.float32), Out: T.Tensor((1, H), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            x16 = T.alloc_shared((MR, H), T.float16)
            x_cmx = T.alloc_shared((1, H), T.float16)   # the GEMVs' M = 1 operand (the mixed-mode scratch scales with M)
            wr_cmx = T.alloc_shared((BNR, H), T.float16)
            yr_cmx = T.alloc_local((MR, BNR), T.float16)
            lg_cmx = T.alloc_shared((1, RR), T.float16)
            ids = T.alloc_shared((16,), T.int32)
            wts = T.alloc_shared((16,), T.float32)
            pick = T.alloc_local((16,), T.float32)
            best = T.alloc_local((2,), T.float32)
            wq_cmx = T.alloc_shared((BG, H // 2), T.uint8)
            y_cmx = T.alloc_local((1, BNG), T.float16)
            gu_cmx = T.alloc_shared((1, 2 * I), T.float16)
            h_cmx = T.alloc_shared((1, I), T.float16)
            wqd_cmx = T.alloc_shared((BD, I // 2), T.uint8)
            yd_cmx = T.alloc_local((1, BND), T.float16)
            ye_cmx = T.alloc_shared((1, H), T.float16)
            acc = T.alloc_shared((1, H), T.float32)
            p_cmx = T.alloc_shared((tiles, H), T.float32)
            res_cmx = T.alloc_shared((1, H), T.float16)
            out_cmx = T.alloc_shared((1, H), T.float16)
            # --- 1. router -------------------------------------------------------------------------------
            for r in T.serial(MR):
                if r > 0:
                    if r % shaves == shave:
                        for c in T.serial(H):
                            x16[r, c] = T.float16(0)
            for c in T.serial(H // shaves):
                acc[0, shave * (H // shaves) + c] = T.float32(0)
            T.copy(X[0:1, 0:H], x16[0:1, 0:H])   # one row: SHAVE 0 fills, tile sync after (also orders the zeroing above)
            T.copy(X[0, 0], x_cmx)
            for n in (T.Pipelined(CR, num_stages=num_stages) if pipelined else T.serial(CR)):
                T.copy(Wr[(tile * CR + n) * BNR, 0], wr_cmx)
                T.gemm(x16, wr_cmx, yr_cmx, transpose_B=True, clear_accum=True)
                T.copy(yr_cmx, Logits[0, (tile * CR + n) * BNR])
            T.sync_grid()
            # --- 2. routing (every tile, SHAVE 0) ---------------------------------------------------------
            T.copy(Logits[0, 0], lg_cmx)
            if shave == 0:
                for k in T.serial(TOPK):
                    best[0] = T.float32(-3.0e38)
                    best[1] = T.float32(0)
                    for e in T.serial(E):
                        if T.cast(lg_cmx[0, e], T.float32) > best[0]:
                            best[0] = T.cast(lg_cmx[0, e], T.float32)
                            best[1] = T.cast(e, T.float32)
                    ids[k] = T.cast(best[1], T.int32)
                    pick[k] = best[0]
                    lg_cmx[0, ids[k]] = T.float16(-65504)
                best[1] = T.float32(0)
                for k in T.serial(TOPK):
                    wts[k] = T.exp(pick[k] - pick[0])   # pick[0] is the row max: the softmax over the chosen k
                    best[1] = best[1] + wts[k]
                for k in T.serial(TOPK):
                    wts[k] = wts[k] / best[1]
                ids[TOPK] = E
                wts[TOPK] = T.float32(1) / (T.float32(1) + T.exp(T.float32(0) - T.cast(lg_cmx[0, E], T.float32)))
            T.sync_threads()
            # --- 3. experts: IPT items per tile -----------------------------------------------------------
            for j in T.serial(IPT):
                for n in (T.Pipelined(CG, num_stages=num_stages) if pipelined else T.serial(CG)):
                    T.copy(WguP[(ids[tile * IPT + j] * CG + n) * BG, 0], wq_cmx)
                    if probe_skip_dpu:
                        y_cmx[0, 0] = T.cast(wq_cmx[0, 0], T.float16)   # keep the slot read (the ring needs a reader)
                    else:
                        for sub in T.serial(BNG // RG):
                            T.gemm_i4(x_cmx, wq_cmx, None, y_cmx, group=G, row0=sub * RG, rows=RG, scales_row=BNG)
                    for c in T.serial(BNG):
                        gu_cmx[0, n * BNG + c] = y_cmx[0, c]
                T.sync_threads()
                for c in T.serial(I // shaves):
                    h_cmx[0, shave * (I // shaves) + c] = T.cast(
                        T.cast(gu_cmx[0, shave * (I // shaves) + c], T.float32)
                        / (T.float32(1) + T.exp(T.float32(0) - T.cast(gu_cmx[0, shave * (I // shaves) + c], T.float32)))
                        * T.cast(gu_cmx[0, I + shave * (I // shaves) + c], T.float32), T.float16)
                T.sync_threads()
                for n in (T.Pipelined(CD, num_stages=num_stages) if pipelined else T.serial(CD)):
                    T.copy(WdP[(ids[tile * IPT + j] * CD + n) * BD, 0], wqd_cmx)
                    if probe_skip_dpu:
                        yd_cmx[0, 0] = T.cast(wqd_cmx[0, 0], T.float16)
                    else:
                        for sub in T.serial(BND // RD):
                            T.gemm_i4(h_cmx, wqd_cmx, None, yd_cmx, group=G, row0=sub * RD, rows=RD, scales_row=BND)
                    for c in T.serial(BND):
                        ye_cmx[0, n * BND + c] = yd_cmx[0, c]   # idempotent: the serial form runs every chunk on both SHAVEs
                T.sync_threads()
                for c in T.serial(H // shaves):
                    acc[0, shave * (H // shaves) + c] = (acc[0, shave * (H // shaves) + c]
                                                         + wts[tile * IPT + j] * T.cast(ye_cmx[0, shave * (H // shaves) + c], T.float32))
                T.sync_threads()   # h and ye are rewritten by the next item
            if shave == 0:
                T.copy(acc, Part[tile, 0])
            T.sync_grid()
            # --- 4. combine on tile 0 ---------------------------------------------------------------------
            if tile == 0:
                T.copy(Part, p_cmx)
                T.copy(Res[0, 0], res_cmx)
                for c in T.serial(H // shaves):
                    out_cmx[0, shave * (H // shaves) + c] = T.cast(
                        T.cast(res_cmx[0, shave * (H // shaves) + c], T.float32)
                        + p_cmx[0, shave * (H // shaves) + c] + p_cmx[1, shave * (H // shaves) + c]
                        + p_cmx[2, shave * (H // shaves) + c], T.float16)
                T.sync_threads()
                if shave == 0:
                    T.copy(out_cmx, Out[0, 0])

    return moe


def moe_scale_rows(H: int, I: int, G: int, BNG: int, BND: int):
    """Rows of a chunk block that hold the chunk's group-major fp16 scales, for the gate/up and down projections."""
    HG, IG = H // G, I // G
    assert (HG * BNG * 2) % (H // 2) == 0 and (IG * BND * 2) % (I // 2) == 0, "scales must fill whole weight rows"
    return HG * BNG * 2 // (H // 2), IG * BND * 2 // (I // 2)


def pack_chunk_blocks(w_int: "np.ndarray", scales: "np.ndarray", BN: int):
    """[N, K] int4 values and [N, K/G] scales -> the chunk blocks of `T.gemm_i4(..., scales_row=BN)`: for each
    chunk of BN rows, the packed weight rows then the chunk's group-major scales [K/G, BN] as whole uint8 rows.
    Returns uint8 [N/BN x (BN + SR), K/2]."""
    import numpy as np
    N, K = w_int.shape
    ng = scales.shape[1]
    assert N % BN == 0 and (ng * BN * 2) % (K // 2) == 0
    SR = ng * BN * 2 // (K // 2)
    wq = pack_int4(w_int)
    blocks = []
    for n0 in range(0, N, BN):
        sT = np.ascontiguousarray(scales[n0:n0 + BN].astype(np.float16).T)   # [ng, BN]
        blocks.append(np.concatenate([wq[n0:n0 + BN], sT.view(np.uint8).reshape(SR, K // 2)], axis=0))
    return np.concatenate(blocks, axis=0)


def pack_moe(router: "np.ndarray", gu_int: "np.ndarray", gu_scales: "np.ndarray", d_int: "np.ndarray",
             d_scales: "np.ndarray", tiles: int = 3, shaves: int = 2, BNR: int = 16, BNG: int = 128, BND: int = 256,
             G: int = 128):
    """The kernel's weight layout from per-expert arrays (the shared expert LAST, as expert E): router fp16
    [E + 1, H] (row E the shared gate) padded with zero rows to RR; gu_int int8 [E + 1, 2I, H] in [-8, 7] and its
    scales [E + 1, 2I, H / G]; d_int [E + 1, H, I] and scales [E + 1, H, I / G]. Returns the kernel's arrays: the
    router and the two projections as chunk blocks (`pack_chunk_blocks`, BNG / BND rows per chunk)."""
    import numpy as np
    R, H = router.shape
    per_round = tiles * shaves * BNR
    RR = ((R + per_round - 1) // per_round) * per_round
    wr = np.zeros((RR, H), np.float16)
    wr[:R] = router
    gu = np.concatenate([pack_chunk_blocks(gu_int[e], gu_scales[e], BNG) for e in range(gu_int.shape[0])], axis=0)
    d = np.concatenate([pack_chunk_blocks(d_int[e], d_scales[e], BND) for e in range(d_int.shape[0])], axis=0)
    return dict(Wr=wr, WguP=gu, WdP=d)


def moe_reference(x: "np.ndarray", router: "np.ndarray", gu_int, gu_scales, d_int, d_scales, res, topk: int, G: int):
    """numpy oracle of make_moe_decode's math (fp32; the shared expert is the last expert, gated by router row E)."""
    import numpy as np
    E = router.shape[0] - 1
    xf = x.astype(np.float32).reshape(-1)
    logits = router.astype(np.float32) @ xf
    probs = np.exp(logits[:E] - logits[:E].max())
    probs /= probs.sum()
    order = np.argsort(-probs, kind="stable")[:topk]
    w = probs[order] / probs[order].sum()

    def ffn(e):
        gu = (gu_int[e].astype(np.float32) * np.repeat(gu_scales[e].astype(np.float32), G, axis=1)) @ xf
        g, u = gu[: gu.size // 2], gu[gu.size // 2:]
        h = g / (1 + np.exp(-g)) * u
        return (d_int[e].astype(np.float32) * np.repeat(d_scales[e].astype(np.float32), G, axis=1)) @ h

    out = res.astype(np.float32).reshape(-1).copy()
    for e, we in zip(order, w):
        out += we * ffn(int(e))
    out += 1 / (1 + np.exp(-logits[E])) * ffn(E)
    return out, order, w


def pack_int4(w_int: "np.ndarray") -> "np.ndarray":
    """[N, K] signed int4 values in [-8, 7] -> [N, K/2] bytes, low nibble first (the reference packer)."""
    import numpy as np
    q = (w_int.astype(np.int16) & 0xF).astype(np.uint8)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)


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
