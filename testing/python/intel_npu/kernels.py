"""Phase 1 GEMM kernel for the Intel NPU: C[M, N] = A[M, K] . B[N, K]^T in fp16.

One persistent launch over `tiles` x `shaves_per_tile` SHAVEs; the output blocks
are dealt round-robin over the workers. The DPU takes the whole K per
workload, so A and B tiles are [BM, K] and [BN, K] in CMX.

No `from __future__ import annotations`: T.prim_func evaluates the tensor
annotations against the factory's locals.
"""

import os

import tilelang.intel_npu.language as T

PERF_REGIONS = os.environ.get("TL_NPU_PERF_REGIONS") == "1"   # named SHAVE compute spans in the profiler trace (perf_attr.py)


@T.macro
def perf_begin(name: str):
    """Opens a named SHAVE region on the profiler track (4-char label; a no-op unless TL_NPU_PERF_REGIONS=1). A macro:
    a bare extern call inside a plain helper is evaluated and dropped by the eager builder."""
    if PERF_REGIONS:
        T.call_extern("handle", "tl_npu_perf_begin", name)


@T.macro
def perf_end():
    if PERF_REGIONS:
        T.call_extern("handle", "tl_npu_perf_end")


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
    moe_phases = moe_block(H=H, I=I, E=E, TOPK=TOPK, G=G, tiles=tiles, shaves=shaves, num_stages=num_stages, BNR=BNR,
                           BNG=BNG, BND=BND, RG=RG, RD=RD, pipelined=pipelined, probe_skip_dpu=probe_skip_dpu)

    @T.prim_func
    def moe(X: T.Tensor((1, H), T.float16), Wr: T.Tensor((RR, H), T.float16),
            WguP: T.Tensor(((E + 1) * CG * BG, H // 2), T.uint8), WdP: T.Tensor(((E + 1) * CD * BD, I // 2), T.uint8),
            Res: T.Tensor((1, H), T.float16), Logits: T.Tensor((MR, RR), T.float16),
            Part: T.Tensor((tiles, H), T.float32), Out: T.Tensor((1, H), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            x_cmx = T.alloc_shared((1, H), T.float16)
            p_cmx = T.alloc_shared((tiles, H), T.float32)
            res_cmx = T.alloc_shared((1, H), T.float16)
            out_cmx = T.alloc_shared((1, H), T.float16)
            T.copy(X[0, 0], x_cmx)
            moe_phases(x_cmx, Wr, WguP, WdP, Logits, Part, tile, shave)
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


def moe_block(H: int, I: int, E: int, TOPK: int, G: int, tiles: int, shaves: int, num_stages: int = 4, BNR: int = 16,
              BNG: int = 128, BND: int = 256, RG: int = 64, RD: int = 256, pipelined: bool = True, probe_skip_dpu: bool = False,
              num_stages_g: int | None = None, defer_gu: bool = False):
    """The MoE layer's phases 1-3 as a `T.macro` (make_moe_decode's docstring): `moe_phases(x_cmx, Wr, WguP, WdP,
    Logits, Part, tile, shave)` takes the normed token in tile-shared CMX (`x_cmx` fp16 [1, H]) and leaves every tile's
    fp32 partial of the MoE output in `Part[tile]` behind a grid barrier; the caller combines (the standalone kernel on
    tile 0 with the residual, a layer kernel into its fp32 residual stream on every tile)."""
    items = TOPK + 1
    assert items % tiles == 0 and H % G == 0 and I % G == 0 and (2 * I) % BNG == 0 and H % BND == 0
    assert BNG % RG == 0 and BND % RD == 0 and RG * (H // G) <= 1024 and RD * (I // G) <= 1024
    assert I % shaves == 0 and H % shaves == 0 and BNR % 16 == 0
    IPT = items // tiles
    per_round = tiles * shaves * BNR
    RR = ((E + 1 + per_round - 1) // per_round) * per_round
    CR = RR // (tiles * BNR)
    assert RR % tiles == 0 and (RR // tiles) % 16 == 0
    RRT = RR // tiles   # router rows per tile: resident, one DPU workload (a ring of 16-row chunks cost 67 us: ~5 us per chunk of protocol)
    CG, CD = (2 * I) // BNG, H // BND
    MR = 16
    FL = 16   # fp32 lanes
    assert (I // shaves) % (2 * FL) == 0 and (H // shaves) % FL == 0 and BNG % (2 * FL) == 0 and BND % (2 * FL) == 0 and H % (2 * FL) == 0
    assert E + 1 <= RR   # the shared expert's gate logit is row E
    topk_lanes = E % (2 * FL) == 0   # the row max on 32 fp16 lanes; a small instance scans scalars
    NSG = num_stages_g if num_stages_g is not None else num_stages   # resolved here: `or` inside the macro parses as a TIR boolean
    # (workload, defer its reduction into the next workload's DPU time): all but the last. Measured -11 us/layer for
    # 2 x RG x H/G x 2 B of CMX (the part double buffer), which the GDN + MoE layer kernel cannot spare: opt-in.
    GU_SUBS = [(sub, defer_gu and sub < BNG // RG - 1) for sub in range(BNG // RG)]
    SRG, SRD = moe_scale_rows(H, I, G, BNG, BND)
    BG, BD = BNG + SRG, BND + SRD

    @T.macro
    def moe_phases(x_cmx, Wr, WguP, WdP, Logits, Part, tile, shave):
        x16 = T.alloc_shared((MR, H), T.float16)
        wr_cmx = T.alloc_shared((RRT, H), T.float16)
        yr_cmx = T.alloc_local((MR, RRT), T.float16)
        lg_cmx = T.alloc_shared((1, RR), T.float16)
        ids = T.alloc_shared((16,), T.int32)
        wts = T.alloc_shared((16,), T.float32)
        pick = T.alloc_local((16,), T.float32)
        best = T.alloc_local((2,), T.float32)
        mx = T.alloc_local((2 * FL,), T.float16)
        wq_cmx = T.alloc_shared((BG, H // 2), T.uint8)
        y_cmx = T.alloc_local((1, BNG), T.float16)
        gu_cmx = T.alloc_shared((1, 2 * I), T.float16)
        h_cmx = T.alloc_shared((1, I), T.float16)
        wqd_cmx = T.alloc_shared((BD, I // 2), T.uint8)
        yd_cmx = T.alloc_local((1, BND), T.float16)
        ye_cmx = T.alloc_shared((1, H), T.float16)
        acc = T.alloc_shared((1, H), T.float32)
        HL = 2 * FL   # fp16 lanes
        # --- 1. router -------------------------------------------------------------------------------
        perf_begin("x16i")
        for r in T.serial(MR):
            if r % shaves == shave:
                for c0 in T.serial(H // HL):
                    for v in T.vectorized(HL):
                        if r == 0:
                            x16[0, c0 * HL + v] = x_cmx[0, c0 * HL + v]
                        else:
                            x16[r, c0 * HL + v] = T.float16(0)
        for c0 in T.serial(H // shaves // FL):
            for v in T.vectorized(FL):
                acc[0, shave * (H // shaves) + c0 * FL + v] = T.float32(0)
        perf_end()
        T.sync_threads()
        perf_begin("rout")
        T.copy(Wr[tile * RRT, 0], wr_cmx)
        if shave == 0:
            T.gemm(x16, wr_cmx, yr_cmx, transpose_B=True, clear_accum=True)
            T.copy(yr_cmx, Logits[0, tile * RRT])
        perf_end()
        T.sync_grid()
        # --- 2. routing (every tile, SHAVE 0) ---------------------------------------------------------
        T.copy(Logits[0, 0], lg_cmx)
        perf_begin("topk")
        if shave == 0:
            for k in T.serial(TOPK):
                # the row max, then the first index holding it (the lowest id on a tie)
                if topk_lanes:
                    for v in T.vectorized(HL):
                        mx[v] = lg_cmx[0, v]
                    for e0 in T.serial(1, E // HL):
                        for v in T.vectorized(HL):
                            mx[v] = T.max(mx[v], lg_cmx[0, e0 * HL + v])
                    best[0] = T.cast(T.call_extern(T.float16, "tl_npu_vmax", mx[0:HL]), T.float32)
                    # argmax on lanes: E - index where the logit equals the max (0 elsewhere), lane max -> E - lowest index
                    for v in T.vectorized(HL):
                        mx[v] = T.if_then_else(lg_cmx[0, v] == T.cast(best[0], T.float16), T.cast(E - v, T.float16), T.float16(0))
                    for e0 in T.serial(1, E // HL):
                        for v in T.vectorized(HL):
                            mx[v] = T.max(mx[v], T.if_then_else(lg_cmx[0, e0 * HL + v] == T.cast(best[0], T.float16),
                                                                T.cast(E - (e0 * HL + v), T.float16), T.float16(0)))
                    best[1] = T.float32(E) - T.cast(T.call_extern(T.float16, "tl_npu_vmax", mx[0:HL]), T.float32)
                else:
                    best[0] = T.float32(-3.0e38)
                    for e in T.serial(E):
                        best[0] = T.max(best[0], T.cast(lg_cmx[0, e], T.float32))
                    best[1] = T.float32(E)
                    for e in T.serial(E):
                        if T.cast(lg_cmx[0, e], T.float32) == best[0]:
                            best[1] = T.min(best[1], T.cast(e, T.float32))
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
        perf_end()
        T.sync_threads()
        # --- 3. experts: IPT items per tile -----------------------------------------------------------
        for j in T.serial(IPT):
            perf_begin("gupr")
            for n in (T.Pipelined(CG, num_stages=NSG) if pipelined else T.serial(CG)):
                T.copy(WguP[(ids[tile * IPT + j] * CG + n) * BG, 0], wq_cmx)
                if probe_skip_dpu:
                    y_cmx[0, 0] = T.cast(wq_cmx[0, 0], T.float16)   # keep the slot read (the ring needs a reader)
                else:
                    for sub, dfr in GU_SUBS:   # Python-level: `defer` is a compile-time flag
                        # all but the chunk's last workload defer their reduction into the next workload's DPU time
                        T.gemm_i4(x_cmx, wq_cmx, None, y_cmx, group=G, row0=sub * RG, rows=RG, scales_row=BNG, defer=dfr)
                for c0 in T.serial(BNG // HL):
                    for v in T.vectorized(HL):
                        gu_cmx[0, n * BNG + c0 * HL + v] = y_cmx[0, c0 * HL + v]
            perf_end()
            T.sync_threads()
            # SwiGLU on 32 fp16 lanes (VAU sigmoid), as npu_dsl's silu_h / mul_h: h = gate * sigmoid(gate) * up
            perf_begin("swig")
            for c0 in T.serial(I // shaves // HL):
                for v in T.vectorized(HL):
                    h_cmx[0, shave * (I // shaves) + c0 * HL + v] = (
                        gu_cmx[0, shave * (I // shaves) + c0 * HL + v]
                        * T.sigmoid(gu_cmx[0, shave * (I // shaves) + c0 * HL + v])
                        * gu_cmx[0, I + shave * (I // shaves) + c0 * HL + v])
            perf_end()
            T.sync_threads()
            perf_begin("down")
            for n in (T.Pipelined(CD, num_stages=num_stages) if pipelined else T.serial(CD)):
                T.copy(WdP[(ids[tile * IPT + j] * CD + n) * BD, 0], wqd_cmx)
                if probe_skip_dpu:
                    yd_cmx[0, 0] = T.cast(wqd_cmx[0, 0], T.float16)
                else:
                    for sub in T.serial(BND // RD):
                        T.gemm_i4(h_cmx, wqd_cmx, None, yd_cmx, group=G, row0=sub * RD, rows=RD, scales_row=BND)
                for c0 in T.serial(BND // HL):
                    for v in T.vectorized(HL):
                        ye_cmx[0, n * BND + c0 * HL + v] = yd_cmx[0, c0 * HL + v]   # idempotent: the serial form runs every chunk on both SHAVEs
            perf_end()
            T.sync_threads()
            perf_begin("accm")
            for c0 in T.serial(H // shaves // FL):
                for v in T.vectorized(FL):
                    acc[0, shave * (H // shaves) + c0 * FL + v] = (
                        acc[0, shave * (H // shaves) + c0 * FL + v]
                        + wts[tile * IPT + j] * T.cast(ye_cmx[0, shave * (H // shaves) + c0 * FL + v], T.float32))
            perf_end()
            T.sync_threads()   # h and ye are rewritten by the next item
        if shave == 0:
            T.copy(acc, Part[tile, 0])
        T.sync_grid()

    return moe_phases


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


def make_gdn_decode(H: int = 2048, KH: int = 16, VH: int = 32, DK: int = 128, DV: int = 128, G: int = 128,
                    tiles: int = 3, shaves: int = 2, num_stages: int = 4, num_stages2: int = 2, BN1: int = 128, R1: int = 64,
                    BN2: int = 64, R2: int = 32, CB: int = 512, eps: float = 1e-6, pipelined: bool = True,
                    probe_skip: str | None = None):
    """M4 decode: one token through a Gated DeltaNet mixer layer in ONE launch (Qwen3.6's shape by default: hidden H,
    KH k-heads and VH v-heads of DK = DV = 128, conv kernel 4, int4 g128 projections).

      mixed | z = x . Win^T                   (in_proj GEMV, int4 g128 rows [q | k | v | z | pad])
      b | a = x . Wba^T                        (the 2 VH gate projections the checkpoint keeps in bf16: fp16 rows,
                                                fp32 dot products on the SHAVE, dealt over the workers)
      conv  = silu(conv1d_k4(mixed; state))    (per channel, the state carried in place)
      q, k  = l2norm(conv_q) / sqrt(DK), l2norm(conv_k); v = conv_v;  beta = sigmoid(b);  g = -exp(A_log) softplus(a + dt_bias)
      S     = S e^g + k (v - e^g S^T k) beta^T;  o = S^T q           (per v-head, S [DK, DV] fp32 carried in place)
      out   = res + rmsnorm(o) * g_norm * silu(z) . Wout^T          (gated RMSNorm per head, out_proj GEMV)

    Phases: A. in_proj GEMV, rows split over the tiles (chunk blocks, local rings) -> Y in DDR, then each worker's
    RBA rows of Wba as fp32 lane dot products -> BA; T.sync_grid();
    B1. causal conv + SiLU over CONV channels in blocks of CB dealt over the workers, conv state updated, -> CV;
    T.sync_grid(); B2. each worker takes HPW consecutive v-heads (whole k-heads): one DMA each for its q, k, v, z
    rows, then per head S in (64 KiB), the two-pass delta rule in fp32 on vector lanes, S out, gated norm -> O;
    T.sync_grid(); C. out_proj GEMV + residual, rows split over the tiles -> Out (padded to NOUT_pad); its ring has
    `num_stages2` slots (one per SHAVE by default: both rings at two slots exceed the CMX budget by 34 KiB).
    Scratch (DDR): Y [1, YLEN], BA [1, BA_LEN] (b then a), CV [1, CVLEN], O [1, workers x HPW x DV] (padded so the last worker's whole-block
    row loads stay in bounds; a DDR read past a buffer hangs the device). Packing: `pack_gdn`; sizes: `gdn_sizes`.
    `probe_skip`: MEASUREMENT ARM ONLY (output wrong by design): 'conv' leaves out the conv SHAVE loop, 'scan' the
    per-head delta rule and gated norm, 'gemv' both GEMVs' DPU workloads (the rings still stream)."""
    gdn_mixer, sz = gdn_block(H=H, KH=KH, VH=VH, DK=DK, DV=DV, G=G, tiles=tiles, shaves=shaves, num_stages=num_stages,
                              num_stages2=num_stages2, BN1=BN1, R1=R1, BN2=BN2, R2=R2, CB=CB, eps=eps, pipelined=pipelined,
                              probe_skip=probe_skip, res_add=True)
    CONV, NIN_pad, NOUT_pad, BR1, BR2, VC = sz["CONV"], sz["NIN_pad"], sz["NOUT_pad"], sz["BR1"], sz["BR2"], sz["VC"]
    WBA_pad, YLEN, BA_LEN, CVLEN, OPAD = sz["WBA_pad"], sz["YLEN"], sz["BA_LEN"], sz["CVLEN"], sz["OPAD"]

    @T.prim_func
    def gdn(X: T.Tensor((1, H), T.float16), Res: T.Tensor((1, H), T.float16),
            WinP: T.Tensor(((NIN_pad // BN1) * BR1, H // 2), T.uint8), Wba: T.Tensor((WBA_pad, H), T.float16),
            ConvW: T.Tensor((4, CONV), T.float16),
            ALog: T.Tensor((1, VH), T.float32), DtBias: T.Tensor((1, VH), T.float32), GNorm: T.Tensor((1, DV), T.float16),
            ConvState: T.Tensor((3, CONV), T.float16), Ssm: T.Tensor((VH, DK, DV), T.float32),
            WoutP: T.Tensor(((NOUT_pad // BN2) * BR2, VC // 2), T.uint8),
            Y: T.Tensor((1, YLEN), T.float16), BA: T.Tensor((1, BA_LEN), T.float16), CV: T.Tensor((1, CVLEN), T.float16),
            O: T.Tensor((1, OPAD), T.float16),
            Out: T.Tensor((1, NOUT_pad), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            x_cmx = T.alloc_shared((1, H), T.float16)
            T.copy(X[0, 0], x_cmx)
            gdn_mixer(x_cmx, WinP, Wba, ConvW, ALog, DtBias, GNorm, ConvState, Ssm, WoutP, Y, BA, CV, O, Res, Out, tile, shave)

    return gdn


def gdn_block(H: int, KH: int, VH: int, DK: int, DV: int, G: int, tiles: int, shaves: int, num_stages: int = 4,
              num_stages2: int = 2, BN1: int = 128, R1: int = 64, BN2: int = 64, R2: int = 32, CB: int = 512, eps: float = 1e-6,
              pipelined: bool = True, probe_skip: str | None = None, res_add: bool = True):
    """The Gated DeltaNet mixer as a `T.macro` (make_gdn_decode's docstring): `gdn_mixer(x_cmx, WinP, Wba, ConvW, ALog,
    DtBias, GNorm, ConvState, Ssm, WoutP, Y, BA, CV, O, Res, Out, tile, shave)` takes the normed token in tile-shared CMX
    and writes the mixer output (+ Res when `res_add`) to Out, every tile its third of the rows. Returns (macro, sizes)."""
    assert DK == DV and VH % KH == 0 and H % G == 0 and CB % 16 == 0
    REP = VH // KH
    QK, VC = KH * DK, VH * DV
    CONV = 2 * QK + VC
    ZOFF, BOFF, AOFF = CONV, CONV + VC, CONV + VC + VH
    NIN = CONV + VC
    NIN_pad = -(-NIN // (BN1 * tiles)) * BN1 * tiles
    NOUT_pad = -(-H // (BN2 * tiles)) * BN2 * tiles
    C1, C2 = NIN_pad // (BN1 * tiles), NOUT_pad // (BN2 * tiles)
    HG1, KG2 = H // G, VC // G
    assert R1 * HG1 <= 1024 and R2 * KG2 <= 1024 and BN1 % R1 == 0 and BN2 % R2 == 0
    assert (HG1 * BN1 * 2) % (H // 2) == 0 and (KG2 * BN2 * 2) % (VC // 2) == 0
    SR1, SR2 = HG1 * BN1 * 2 // (H // 2), KG2 * BN2 * 2 // (VC // 2)
    BR1, BR2 = BN1 + SR1, BN2 + SR2
    workers = tiles * shaves
    RBA = -(-2 * VH // workers)                # b / a rows per worker
    assert RBA <= BN1
    WBA_pad = workers * RBA
    HPW = -(-VH // workers)
    HPW += (-HPW) % REP                       # whole k-heads per worker
    KHW = HPW // REP
    BA_LEN = max(WBA_pad, VH + workers * HPW)  # the last worker's a reads reach past VH heads (skipped, in bounds)
    assert CONV % CB == 0
    NB = CONV // CB
    OPAD = workers * HPW * DV
    YLEN = max(NIN_pad, ZOFF + OPAD)          # the last worker's z rows reach past VH heads (skipped, but in bounds)
    CVLEN = max(CONV, 2 * QK + OPAD)
    sizes = dict(CONV=CONV, NIN_pad=NIN_pad, NOUT_pad=NOUT_pad, BR1=BR1, BR2=BR2, VC=VC, WBA_pad=WBA_pad, YLEN=YLEN,
                 BA_LEN=BA_LEN, CVLEN=CVLEN, OPAD=OPAD)
    FL = 16                                   # fp32 lanes
    assert DV % FL == 0
    qscale = 1.0 / (DK ** 0.5)
    skip_conv, skip_scan, skip_gemv = (probe_skip == "conv"), (probe_skip == "scan"), (probe_skip == "gemv")

    @T.macro
    def gdn_head(S_cmx, q_cmx, k_cmx, v_cmx, z_cmx, ba_cmx, al_cmx, db_cmx, gn_cmx, qn, kn, egv, betav, kv, sq, delta,
                 o, kr, kr2, qr, qr2, rs, dr, st, ls, ls2, eh, on_cmx, j, h):
        """One v-head's step: l2-normalised q / k of its k-head, the gates, the delta rule on S (fp32 lanes), the
        gated RMSNorm into on_cmx[j]. j: the head's index within the worker's block, h: its index.
        The delta rule as npu_dsl's gdn_scan (kernels_poc/common/fused_ops.hpp): ONE read of S accumulates both
        kv = S^T k and sq = S^T q, then o = e^g sq + (k . q) delta algebraically (S_new^T q = e^g S^T q + delta (k^T q)),
        and S = e^g S + k delta^T is one read-modify-write. k_d / q_d are scalar splats (CMU.CP.512.32.R, one op; not a
        [DK, FL] row expansion that costs a load per FMA). moviCompile does not software-pipeline these loops, so every
        iteration carries its own parallelism: pass 1 four register accumulators over two column vectors, pass 2 a
        whole row in eight register vectors loaded before any store (a load after a store to S serialises on it)."""
        for v in T.vectorized(FL):
            ls[v] = T.float32(0)
            ls2[v] = T.float32(0)
        for d0 in T.serial(DK // FL):
            for v in T.vectorized(FL):
                ls[v] = ls[v] + (T.cast(q_cmx[0, (j // REP) * DK + d0 * FL + v], T.float32)
                                 * T.cast(q_cmx[0, (j // REP) * DK + d0 * FL + v], T.float32))
                ls2[v] = ls2[v] + (T.cast(k_cmx[0, (j // REP) * DK + d0 * FL + v], T.float32)
                                   * T.cast(k_cmx[0, (j // REP) * DK + d0 * FL + v], T.float32))
        st[0] = T.float32(0)
        st[1] = T.float32(0)
        for v in T.serial(FL):
            st[0] = st[0] + ls[v]
            st[1] = st[1] + ls2[v]
        st[0] = T.float32(qscale) / T.sqrt(st[0] + T.float32(1e-6))
        st[1] = T.float32(1) / T.sqrt(st[1] + T.float32(1e-6))
        for v in T.vectorized(FL):
            ls[v] = T.float32(0)
        for d0 in T.serial(DK // FL):
            for v in T.vectorized(FL):
                qn[d0 * FL + v] = T.cast(q_cmx[0, (j // REP) * DK + d0 * FL + v], T.float32) * st[0]
                kn[d0 * FL + v] = T.cast(k_cmx[0, (j // REP) * DK + d0 * FL + v], T.float32) * st[1]
                ls[v] = ls[v] + qn[d0 * FL + v] * kn[d0 * FL + v]
        st[4] = T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL])   # k . q
        # gates: beta = sigmoid(b); g = -exp(A_log) softplus(a + dt_bias); e^g
        st[2] = T.float32(1) / (T.float32(1) + T.exp(T.float32(0) - T.cast(ba_cmx[0, j], T.float32)))
        st[3] = T.cast(ba_cmx[1, j], T.float32) + db_cmx[0, h]
        st[3] = T.if_then_else(st[3] > T.float32(20), st[3], T.log(T.float32(1) + T.exp(st[3])))
        st[3] = T.exp(T.float32(0) - T.exp(al_cmx[0, h]) * st[3])
        for v in T.vectorized(FL):
            egv[v] = st[3]
            betav[v] = st[2]
        # pass 1: kv = S^T k and sq = S^T q from one read of S; two adjacent column vectors outer, rows inner
        for v0 in T.serial(DV // (2 * FL)):
            for v in T.vectorized(FL):
                kr[v] = T.float32(0)
                kr2[v] = T.float32(0)
                qr[v] = T.float32(0)
                qr2[v] = T.float32(0)
            for d in T.serial(DK):
                for v in T.vectorized(FL):
                    kr[v] = kr[v] + S_cmx[d, 2 * v0 * FL + v] * kn[d]
                    qr[v] = qr[v] + S_cmx[d, 2 * v0 * FL + v] * qn[d]
                    kr2[v] = kr2[v] + S_cmx[d, (2 * v0 + 1) * FL + v] * kn[d]
                    qr2[v] = qr2[v] + S_cmx[d, (2 * v0 + 1) * FL + v] * qn[d]
            for v in T.vectorized(FL):
                kv[2 * v0 * FL + v] = kr[v]
                sq[2 * v0 * FL + v] = qr[v]
                kv[(2 * v0 + 1) * FL + v] = kr2[v]
                sq[(2 * v0 + 1) * FL + v] = qr2[v]
        # delta = beta (v - e^g kv); o = e^g sq + (k . q) delta
        for v0 in T.unroll(DV // FL):
            for v in T.vectorized(FL):
                dr[v] = (T.cast(v_cmx[0, j * DV + v0 * FL + v], T.float32) - egv[v] * kv[v0 * FL + v]) * betav[v]
                delta[v0 * FL + v] = dr[v]
                o[v0 * FL + v] = egv[v] * sq[v0 * FL + v] + dr[v] * st[4]
        # pass 2: S = e^g S + k delta^T, a row at a time: its DV / FL vectors loaded (rs, registers), computed, then stored
        for d in T.serial(DK):
            st[5] = kn[d]
            for i, r in enumerate(rs):   # Python-level: `range` would become a TIR loop
                for v in T.vectorized(FL):
                    r[v] = S_cmx[d, i * FL + v]
            for i, r in enumerate(rs):
                for v in T.vectorized(FL):
                    r[v] = r[v] * egv[v] + delta[i * FL + v] * st[5]
            for i, r in enumerate(rs):
                for v in T.vectorized(FL):
                    S_cmx[d, i * FL + v] = r[v]
        # gated RMSNorm: o * rsqrt(mean(o^2) + eps) * g_norm * silu(z), on lanes (the exp of -z on half16, VAU.EXP)
        for v in T.vectorized(FL):
            ls[v] = T.float32(0)
        for v0 in T.unroll(DV // FL):
            for v in T.vectorized(FL):
                ls[v] = ls[v] + o[v0 * FL + v] * o[v0 * FL + v]
        st[0] = T.float32(0)
        for v in T.serial(FL):
            st[0] = st[0] + ls[v]
        st[0] = T.float32(1) / T.sqrt(st[0] / T.float32(DV) + T.float32(eps))
        for v0 in T.unroll(DV // FL):
            for v in T.vectorized(FL):
                eh[v] = T.exp(T.float16(0) - z_cmx[0, j * DV + v0 * FL + v])
                on_cmx[0, j * DV + v0 * FL + v] = T.cast(
                    o[v0 * FL + v] * st[0] * T.cast(gn_cmx[0, v0 * FL + v], T.float32)
                    * T.cast(z_cmx[0, j * DV + v0 * FL + v], T.float32) / (T.float32(1) + T.cast(eh[v], T.float32)), T.float16)

    @T.macro
    def gdn_mixer(x_cmx, WinP, Wba, ConvW, ALog, DtBias, GNorm, ConvState, Ssm, WoutP, Y, BA, CV, O, Res, Out, tile, shave):
            worker = tile * shaves + shave
            w1_cmx = T.alloc_shared((BR1, H // 2), T.uint8)
            y1_cmx = T.alloc_local((1, BN1), T.float16)
            wba_cmx = T.alloc_local((RBA, H), T.float16)
            cs_cmx = T.alloc_local((3, CB), T.float16)
            cw_cmx = T.alloc_local((4, CB), T.float16)
            cx_cmx = T.alloc_local((1, CB), T.float16)
            cv_cmx = T.alloc_local((1, CB), T.float16)
            q_cmx = T.alloc_local((1, KHW * DK), T.float16)
            k_cmx = T.alloc_local((1, KHW * DK), T.float16)
            v_cmx = T.alloc_local((1, HPW * DV), T.float16)
            z_cmx = T.alloc_local((1, HPW * DV), T.float16)
            ba_cmx = T.alloc_local((2, HPW), T.float16)
            gn_cmx = T.alloc_shared((1, DV), T.float16)
            al_cmx = T.alloc_shared((1, VH), T.float32)
            db_cmx = T.alloc_shared((1, VH), T.float32)
            S_cmx = T.alloc_local((DK, DV), T.float32)
            qn = T.alloc_local((DK,), T.float32)
            kn = T.alloc_local((DK,), T.float32)
            egv = T.alloc_local((FL,), T.float32)
            betav = T.alloc_local((FL,), T.float32)
            kv = T.alloc_local((DV,), T.float32)
            sq = T.alloc_local((DV,), T.float32)
            delta = T.alloc_local((DV,), T.float32)
            o = T.alloc_local((DV,), T.float32)
            kr = T.alloc_local((FL,), T.float32)
            kr2 = T.alloc_local((FL,), T.float32)
            qr = T.alloc_local((FL,), T.float32)
            qr2 = T.alloc_local((FL,), T.float32)
            rs = [T.alloc_local((FL,), T.float32) for _ in range(DV // FL)]   # one S row in registers (pass 2)
            dr = T.alloc_local((FL,), T.float32)
            acc = T.alloc_local((FL,), T.float32)
            ls = T.alloc_local((FL,), T.float32)
            ls2 = T.alloc_local((FL,), T.float32)
            eh = T.alloc_local((FL,), T.float16)
            st = T.alloc_local((8,), T.float32)
            on_cmx = T.alloc_local((1, HPW * DV), T.float16)
            o_cmx = T.alloc_shared((1, VC), T.float16)
            res_cmx = T.alloc_shared((1, NOUT_pad), T.float16)
            w2_cmx = T.alloc_shared((BR2, VC // 2), T.uint8)
            y2_cmx = T.alloc_local((1, BN2), T.float16)
            # --- A. in_proj GEMV: rows [q | k | v | z | pad] split over the tiles; b | a on the SHAVE ------------
            for n in (T.Pipelined(C1, num_stages=num_stages) if pipelined else T.serial(C1)):
                T.copy(WinP[(tile * C1 + n) * BR1, 0], w1_cmx)
                if skip_gemv:
                    y1_cmx[0, 0] = T.cast(w1_cmx[0, 0], T.float16)
                else:
                    for sub in T.serial(BN1 // R1):
                        T.gemm_i4(x_cmx, w1_cmx, None, y1_cmx, group=G, row0=sub * R1, rows=R1, scales_row=BN1)
                T.copy(y1_cmx, Y[0, (tile * C1 + n) * BN1])
            T.copy(Wba[worker * RBA, 0], wba_cmx)
            for r in T.serial(RBA):
                for v in T.vectorized(FL):
                    ls[v] = T.float32(0)
                for d0 in T.serial(H // FL):
                    for v in T.vectorized(FL):
                        ls[v] = ls[v] + T.cast(x_cmx[0, d0 * FL + v], T.float32) * T.cast(wba_cmx[r, d0 * FL + v], T.float32)
                y1_cmx[0, r] = T.cast(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]), T.float16)
            T.copy(y1_cmx[0:1, 0:RBA], BA[0, worker * RBA])
            T.sync_grid()
            # --- B1. causal conv1d (k = 4) + SiLU over the CONV channels, blocks of CB dealt over the workers -------
            for it in T.serial(T.ceildiv(NB, workers)):
                blk = it * workers + worker
                if blk < NB:
                    T.copy(ConvState[0, blk * CB], cs_cmx)
                    T.copy(ConvW[0, blk * CB], cw_cmx)
                    T.copy(Y[0, blk * CB], cx_cmx)
                    if not skip_conv:
                        # time-major state and weights: unit-stride lanes (the channel-major form auto-vectorised
                        # into per-lane gathers at ~100 cycles per channel); fp32 accumulate, half16 exp for the SiLU
                        for c0 in T.serial(CB // FL):
                            for v in T.vectorized(FL):
                                acc[v] = (T.cast(cs_cmx[0, c0 * FL + v], T.float32) * T.cast(cw_cmx[0, c0 * FL + v], T.float32)
                                          + T.cast(cs_cmx[1, c0 * FL + v], T.float32) * T.cast(cw_cmx[1, c0 * FL + v], T.float32)
                                          + T.cast(cs_cmx[2, c0 * FL + v], T.float32) * T.cast(cw_cmx[2, c0 * FL + v], T.float32)
                                          + T.cast(cx_cmx[0, c0 * FL + v], T.float32) * T.cast(cw_cmx[3, c0 * FL + v], T.float32))
                                eh[v] = T.exp(T.cast(T.float32(0) - acc[v], T.float16))
                                cv_cmx[0, c0 * FL + v] = T.cast(acc[v] / (T.float32(1) + T.cast(eh[v], T.float32)), T.float16)
                                cs_cmx[0, c0 * FL + v] = cs_cmx[1, c0 * FL + v]
                                cs_cmx[1, c0 * FL + v] = cs_cmx[2, c0 * FL + v]
                                cs_cmx[2, c0 * FL + v] = cx_cmx[0, c0 * FL + v]
                    T.copy(cv_cmx, CV[0, blk * CB])
                    T.copy(cs_cmx, ConvState[0, blk * CB])
            T.sync_grid()
            # --- B2. the delta rule per v-head: worker w takes heads [w * HPW, w * HPW + HPW) ------------------------
            T.copy(GNorm[0, 0], gn_cmx)
            T.copy(ALog[0, 0], al_cmx)
            T.copy(DtBias[0, 0], db_cmx)
            T.copy(CV[0, worker * KHW * DK], q_cmx)
            T.copy(CV[0, QK + worker * KHW * DK], k_cmx)
            T.copy(CV[0, 2 * QK + worker * HPW * DV], v_cmx)
            T.copy(Y[0, ZOFF + worker * HPW * DV], z_cmx)
            T.copy(BA[0, worker * HPW], ba_cmx[0:1, 0:HPW])
            T.copy(BA[0, VH + worker * HPW], ba_cmx[1:2, 0:HPW])
            for j in T.serial(HPW):
                if worker * HPW + j < VH:
                    T.copy(Ssm[worker * HPW + j, 0, 0], S_cmx)
                    if skip_scan:
                        on_cmx[0, j * DV] = T.cast(S_cmx[0, 0], T.float16)
                    else:
                        gdn_head(S_cmx, q_cmx, k_cmx, v_cmx, z_cmx, ba_cmx, al_cmx, db_cmx, gn_cmx, qn, kn, egv, betav,
                                 kv, sq, delta, o, kr, kr2, qr, qr2, rs, dr, st, ls, ls2, eh, on_cmx, j, worker * HPW + j)
                    T.copy(S_cmx, Ssm[worker * HPW + j, 0, 0])
            T.copy(on_cmx, O[0, worker * HPW * DV])
            T.sync_grid()
            # --- C. out_proj GEMV (+ residual), rows split over the tiles ---------------------------------------
            if res_add:
                for c in T.serial(NOUT_pad - H):
                    if shave == 0:
                        res_cmx[0, H + c] = T.float16(0)
                T.copy(Res[0:1, 0:H], res_cmx[0:1, 0:H])
            T.copy(O[0, 0], o_cmx)
            for n in (T.Pipelined(C2, num_stages=num_stages2) if pipelined else T.serial(C2)):
                T.copy(WoutP[(tile * C2 + n) * BR2, 0], w2_cmx)
                for sub in T.serial(BN2 // R2):
                    T.gemm_i4(o_cmx, w2_cmx, None, y2_cmx, group=G, row0=sub * R2, rows=R2, scales_row=BN2)
                if res_add:
                    for c in T.serial(BN2):
                        y2_cmx[0, c] = T.cast(T.cast(y2_cmx[0, c], T.float32)
                                              + T.cast(res_cmx[0, (tile * C2 + n) * BN2 + c], T.float32), T.float16)
                T.copy(y2_cmx, Out[0, (tile * C2 + n) * BN2])

    return gdn_mixer, sizes


def gdn_sizes(H: int, KH: int, VH: int, DK: int, tiles: int = 3, shaves: int = 2, BN1: int = 128, BN2: int = 64, **_):
    """make_gdn_decode's scratch and output sizes: dict(YLEN, BA_LEN, CVLEN, OPAD, NOUT_pad, WBA_pad)."""
    DV = DK
    QK, VC = KH * DK, VH * DV
    CONV = 2 * QK + VC
    NIN = CONV + VC
    NIN_pad = -(-NIN // (BN1 * tiles)) * BN1 * tiles
    NOUT_pad = -(-H // (BN2 * tiles)) * BN2 * tiles
    workers = tiles * shaves
    HPW = -(-VH // workers)
    HPW += (-HPW) % (VH // KH)
    OPAD = workers * HPW * DV
    WBA_pad = workers * (-(-2 * VH // workers))
    return dict(YLEN=max(NIN_pad, CONV + OPAD), BA_LEN=max(WBA_pad, VH + workers * HPW), CVLEN=max(CONV, 2 * QK + OPAD),
                OPAD=OPAD, NOUT_pad=NOUT_pad, WBA_pad=WBA_pad)


def pack_gdn(win_int, win_scales, wout_int, wout_scales, wba=None, tiles: int = 3, shaves: int = 2, BN1: int = 128, BN2: int = 64):
    """make_gdn_decode's weights from the in_proj rows (int4 values [NIN, H] in [-8, 7] with scales [NIN, H/G], rows
    [q | k | v | z]) and the out_proj ([H, VC] with [H, VC/G]): zero-padded to the tile multiple, then chunk
    blocks of BN1 / BN2 rows (`pack_chunk_blocks`); `wba` [2 VH, H] (b rows then a rows, any float dtype) -> fp16 rows
    zero-padded to the worker multiple."""
    import numpy as np

    def padded(w, s, BN):
        n_pad = -(-w.shape[0] // (BN * tiles)) * BN * tiles
        wp = np.zeros((n_pad, w.shape[1]), np.int8)
        sp = np.zeros((n_pad, s.shape[1]), np.float16)
        wp[: w.shape[0]] = w
        sp[: s.shape[0]] = s
        return pack_chunk_blocks(wp, sp, BN)

    out = dict(WinP=padded(win_int, win_scales, BN1), WoutP=padded(wout_int, wout_scales, BN2))
    if wba is not None:
        rows = -(-wba.shape[0] // (tiles * shaves)) * tiles * shaves
        w = np.zeros((rows, wba.shape[1]), np.float16)
        w[: wba.shape[0]] = wba.astype(np.float16)
        out["Wba"] = w
    return out


def gdn_reference(x, res, win_int, win_scales, wout_int, wout_scales, conv_w, a_log, dt_bias, g_norm, conv_state, ssm,
                  KH: int, VH: int, DK: int, G: int, eps: float = 1e-6, details: bool = False, wba=None):
    """numpy oracle of make_gdn_decode (fp32; conv_state [3, CONV] and conv_w [4, CONV] time-major; returns out [H],
    the new conv_state [3, CONV] and ssm [VH, DK, DV]; `details` adds a dict of the intermediates y, conv, on).
    `wba` [2 VH, H]: the b / a projections (fp16-valued); None means they are the last 2 VH rows of win_int."""
    import numpy as np
    DV = DK
    QK, VC = KH * DK, VH * DV
    CONV = 2 * QK + VC
    xf = x.astype(np.float32).reshape(-1)
    win = win_int.astype(np.float32) * np.repeat(win_scales.astype(np.float32), G, axis=1)
    y = win @ xf
    mixed, z = y[:CONV], y[CONV:CONV + VC]
    if wba is not None:
        ba = wba.astype(np.float32) @ xf
        b, a = ba[:VH], ba[VH:2 * VH]
    else:
        b, a = y[CONV + VC:CONV + VC + VH], y[CONV + VC + VH:CONV + VC + 2 * VH]
    cs = conv_state.astype(np.float32)                                     # [3, CONV] time-major, oldest first
    seq = np.concatenate([cs, mixed[None, :]], axis=0)                      # [4, CONV]
    conv = (seq * conv_w.astype(np.float32)).sum(axis=0)
    conv = conv / (1 + np.exp(-conv))
    new_cs = seq[1:].astype(np.float16)
    q = conv[:QK].reshape(KH, DK)
    k = conv[QK:2 * QK].reshape(KH, DK)
    v = conv[2 * QK:].reshape(VH, DV)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(DK)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    rep = VH // KH
    q, k = np.repeat(q, rep, axis=0), np.repeat(k, rep, axis=0)
    beta = 1 / (1 + np.exp(-b))
    g = -np.exp(a_log.astype(np.float32).reshape(-1)) * np.logaddexp(0, a + dt_bias.astype(np.float32).reshape(-1))
    S = ssm.astype(np.float32).copy()
    o = np.zeros((VH, DV), np.float32)
    for h in range(VH):
        S[h] *= np.exp(g[h])
        kvh = S[h].T @ k[h]
        d = (v[h] - kvh) * beta[h]
        S[h] += np.outer(k[h], d)
        o[h] = S[h].T @ q[h]
    zf = z.reshape(VH, DV)
    on = o / np.sqrt((o * o).mean(-1, keepdims=True) + eps) * g_norm.astype(np.float32).reshape(-1) * (zf / (1 + np.exp(-zf)))
    wout = wout_int.astype(np.float32) * np.repeat(wout_scales.astype(np.float32), G, axis=1)
    out = res.astype(np.float32).reshape(-1) + wout @ on.reshape(-1)
    if details:
        return out, new_cs, S, dict(y=y, conv=conv, on=on.reshape(-1), g=g, beta=beta)
    return out, new_cs, S


def make_attn_decode(H: int = 2048, NH: int = 16, NKV: int = 2, D: int = 256, ROT: int = 64, SMAX: int = 1536, G: int = 128,
                     tiles: int = 3, shaves: int = 2, BN1: int = 128, R1: int = 64, BN2: int = 64, R2: int = 32, BKV: int = 128,
                     num_stages: int = 4, num_stages2: int = 2, kv_stages: int = 2, eps: float = 1e-6, pipelined: bool = True):
    """M1/M5 decode: one token through a full-attention layer in ONE launch (Qwen3.6's shape by default: NH q-heads
    over NKV kv-heads (GQA), head dim D, RoPE on the first ROT dims, an output gate per head, zero-centred q / k norms,
    int4 g128 projections, a KV cache of SMAX positions).

      q | gate | k | v = x . Wqkv^T         (rows [per q-head: q(D) gate(D)] x NH, then k (NKV x D), v (NKV x D))
      q, k = rope(rmsnorm_zc(.))           (x rsqrt(mean x^2 + eps) (1 + w); rotate-half on the first ROT dims)
      Kc[g, pos] = k_g;  Vt[g, :, pos] = v_g
      o_h = softmax(q_h . Kc[g]^T / sqrt(D)) . V_g     over positions <= pos (g = h / (NH / NKV))
      out = res + (o * sigmoid(gate)) . Wo^T

    Phases: A. qkv GEMV (rows split over the tiles, chunk blocks) -> Y; T.sync_grid(); B. the NH + NKV head items
    dealt over the workers: q heads normed and roped into QN, k heads normed, roped and appended to the K cache, v
    heads appended to the transposed V cache; T.sync_grid(); C. flash-decoding: every tile streams its third of the
    positions through K / Vt rings for each kv group (fills read the group: the rings are hoisted over the groups),
    online softmax per SHAVE on the group's GQ = NH / NKV rows, positions past pos masked, the tile's two partials
    merged into Part; T.sync_grid(); D. the three tiles' partials merged per head, the gate applied -> O;
    T.sync_grid(); E. o_proj GEMV + residual -> Out. Positions past pos must hold zeros in the cache (the host
    zero-initialises it; the mask then only has to zero their probabilities). Packing: `pack_attn`; sizes: `attn_sizes`."""
    attn_mixer, sz = attn_block(H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, G=G, tiles=tiles, shaves=shaves, BN1=BN1, R1=R1,
                                BN2=BN2, R2=R2, BKV=BKV, num_stages=num_stages, num_stages2=num_stages2, kv_stages=kv_stages,
                                eps=eps, pipelined=pipelined, res_add=True)
    NQKV, NOUT_pad, BR1, BR2, KO, GQ, PW = sz["NQKV"], sz["NOUT_pad"], sz["BR1"], sz["BR2"], sz["KO"], sz["GQ"], sz["PW"]

    @T.prim_func
    def attn(X: T.Tensor((1, H), T.float16), Res: T.Tensor((1, H), T.float16),
             WqkvP: T.Tensor(((NQKV // BN1) * BR1, H // 2), T.uint8), QNormW: T.Tensor((1, D), T.float16),
             KNormW: T.Tensor((1, D), T.float16), CosSin: T.Tensor((2, ROT), T.float32), Pos: T.Tensor((1, 1), T.int32),
             Kc: T.Tensor((NKV, SMAX, D), T.float16), Vt: T.Tensor((NKV, D, SMAX), T.float16),
             WoP: T.Tensor(((NOUT_pad // BN2) * BR2, KO // 2), T.uint8),
             Y: T.Tensor((1, NQKV), T.float16), QN: T.Tensor((NKV, GQ, D), T.float16),
             Part: T.Tensor((NKV, GQ, tiles, PW), T.float32), O: T.Tensor((1, KO), T.float16),
             Out: T.Tensor((1, NOUT_pad), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            x_cmx = T.alloc_shared((1, H), T.float16)
            T.copy(X[0, 0], x_cmx)
            attn_mixer(x_cmx, WqkvP, QNormW, KNormW, CosSin, Pos, Kc, Vt, WoP, Y, QN, Part, O, Res, Out, tile, shave, Pos, 0)

    return attn


def attn_block(H: int, NH: int, NKV: int, D: int, ROT: int, SMAX: int, G: int, tiles: int, shaves: int, BN1: int = 128,
               R1: int = 64, BN2: int = 64, R2: int = 32, BKV: int = 128, num_stages: int = 4, num_stages2: int = 2,
               kv_stages: int = 2, eps: float = 1e-6, pipelined: bool = True, res_add: bool = True, stages: str = "ABCDE",
               gemv_pipelined: bool = True, kv4: bool = False):
    """The full-attention mixer as a `T.macro` (make_attn_decode's docstring): `attn_mixer(x_cmx, WqkvP, QNormW, KNormW,
    CosSin, Pos, Kc, Vt, WoP, Y, QN, Part, O, Res, Out, tile, shave, KvSlot, kvi)` takes the normed token in tile-shared CMX and writes
    the mixer output (+ Res when `res_add`) to Out, every tile its third of the rows. `kv4`: Kc / Vt carry a leading layer
    axis indexed by `KvSlot[kvi, 1]` -- one cache per attention layer of the launch, reached as a kernel parameter rather
    than through a table address (a DDR store whose base is a `T.ddr_view` AND whose offset is a runtime value loses the
    device). A `T.macro` parameter must be a variable or a constant, which is why the slot arrives as a buffer and an
    index rather than as the loaded value. Returns (macro, sizes)."""
    assert NH % NKV == 0 and D % 32 == 0 and ROT % 2 == 0 and SMAX % (tiles * BKV) == 0 and BKV % 32 == 0
    GQ = NH // NKV
    NQKV = NH * 2 * D + 2 * NKV * D
    KOFF, VOFF = NH * 2 * D, NH * 2 * D + NKV * D
    assert NQKV % (BN1 * tiles) == 0
    C1 = NQKV // (BN1 * tiles)
    NOUT_pad = -(-H // (BN2 * tiles)) * BN2 * tiles
    C2 = NOUT_pad // (BN2 * tiles)
    KO = NH * D                                 # o_proj K
    HG1, KG2 = H // G, KO // G
    assert R1 * HG1 <= 1024 and R2 * KG2 <= 1024 and BN1 % R1 == 0 and BN2 % R2 == 0
    assert (HG1 * BN1 * 2) % (H // 2) == 0 and (KG2 * BN2 * 2) % (KO // 2) == 0
    BR1, BR2 = BN1 + HG1 * BN1 * 2 // (H // 2), BN2 + KG2 * BN2 * 2 // (KO // 2)
    workers = tiles * shaves
    NIT = NH + NKV
    CPT = SMAX // (tiles * BKV)                 # key chunks per tile per group
    assert CPT % shaves == 0, "the kv ring's runtime chunk count is a multiple of the SHAVEs (hoisted over the groups)"
    HL, FL = 32, 16
    PW = D + FL                                 # partial row: O then m, l
    scale = 1.0 / (D ** 0.5)
    HR = ROT // 2
    sizes = dict(NQKV=NQKV, NOUT_pad=NOUT_pad, BR1=BR1, BR2=BR2, KO=KO, GQ=GQ, PW=PW)

    @T.macro
    def attn_mixer(x_cmx, WqkvP, QNormW, KNormW, CosSin, Pos, Kc, Vt, WoP, Y, QN, Part, O, Res, Out, tile, shave, KvSlot, kvi):
            worker = tile * shaves + shave
            w1_cmx = T.alloc_shared((BR1, H // 2), T.uint8)
            y1_cmx = T.alloc_local((1, BN1), T.float16)
            pos_cmx = T.alloc_shared((1, 1), T.int32)
            cs_cmx = T.alloc_shared((2, ROT), T.float32)
            qw_cmx = T.alloc_shared((1, D), T.float16)
            kw_cmx = T.alloc_shared((1, D), T.float16)
            row = T.alloc_local((1, D), T.float16)
            rowf = T.alloc_local((D,), T.float32)
            rown = T.alloc_local((1, D), T.float16)
            vcol = T.alloc_local((D, 1), T.float16)
            ls = T.alloc_local((FL,), T.float32)
            st = T.alloc_local((4,), T.float32)
            q_cmx = T.alloc_shared((GQ, D), T.float16)
            k_cmx = T.alloc_shared((BKV, D), T.float16)
            v_cmx = T.alloc_shared((D, BKV), T.float16)
            s_cmx = T.alloc_local((GQ, BKV), T.float16)
            pv_cmx = T.alloc_local((GQ, D), T.float16)
            o_acc = T.alloc_local((GQ, D), T.float32)
            m_i = T.alloc_local((GQ,), T.float32)
            l_i = T.alloc_local((GQ,), T.float32)
            m_new = T.alloc_local((GQ,), T.float32)
            corr = T.alloc_local((GQ,), T.float32)
            mx = T.alloc_local((HL,), T.float16)
            gh = T.alloc_local((FL,), T.float16)
            part = T.alloc_shared((shaves, GQ, PW), T.float32)
            pout = T.alloc_local((GQ, PW), T.float32)
            p3 = T.alloc_local((tiles, PW), T.float32)
            gate = T.alloc_local((1, D), T.float16)
            oh = T.alloc_local((1, D), T.float16)
            o_cmx = T.alloc_shared((1, KO), T.float16)
            res_cmx = T.alloc_shared((1, NOUT_pad), T.float16)
            w2_cmx = T.alloc_shared((BR2, KO // 2), T.uint8)
            y2_cmx = T.alloc_local((1, BN2), T.float16)
            # --- A. qkv GEMV -----------------------------------------------------------------------------------------
            if "A" in stages:
              for n in (T.Pipelined(C1, num_stages=num_stages) if (pipelined and gemv_pipelined) else T.serial(C1)):
                T.copy(WqkvP[(tile * C1 + n) * BR1, 0], w1_cmx)
                for sub in T.serial(BN1 // R1):
                    T.gemm_i4(x_cmx, w1_cmx, None, y1_cmx, group=G, row0=sub * R1, rows=R1, scales_row=BN1)
                T.copy(y1_cmx, Y[0, (tile * C1 + n) * BN1])
            T.sync_grid()
            # --- B. norms, RoPE, cache append: NH q heads then NKV kv heads, dealt over the workers ------------------
            T.copy(Pos[0, 0], pos_cmx)
            T.copy(CosSin[0, 0], cs_cmx)
            T.copy(QNormW[0, 0], qw_cmx)
            T.copy(KNormW[0, 0], kw_cmx)
            if "B" in stages:
              for it in T.serial(T.ceildiv(NIT, workers)):
                item = it * workers + worker
                if item < NIT:
                    if item < NH:
                        T.copy(Y[0, item * 2 * D], row)
                    else:
                        T.copy(Y[0, KOFF + (item - NH) * D], row)
                    # zero-centred RMSNorm: x rsqrt(mean(x^2) + eps) (1 + w)
                    for v in T.vectorized(FL):
                        ls[v] = T.float32(0)
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rowf[d0 * FL + v] = T.cast(row[0, d0 * FL + v], T.float32)
                            ls[v] = ls[v] + rowf[d0 * FL + v] * rowf[d0 * FL + v]
                    st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(D) + T.float32(eps))
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            if item < NH:
                                rowf[d0 * FL + v] = rowf[d0 * FL + v] * st[0] * (T.float32(1) + T.cast(qw_cmx[0, d0 * FL + v], T.float32))
                            else:
                                rowf[d0 * FL + v] = rowf[d0 * FL + v] * st[0] * (T.float32(1) + T.cast(kw_cmx[0, d0 * FL + v], T.float32))
                    # RoPE, rotate-half on the first ROT dims (cos / sin of this position from the host)
                    for r in T.serial(HR):
                        st[1] = rowf[r]
                        st[2] = rowf[HR + r]
                        rowf[r] = st[1] * cs_cmx[0, r] - st[2] * cs_cmx[1, r]
                        rowf[HR + r] = st[2] * cs_cmx[0, HR + r] + st[1] * cs_cmx[1, HR + r]
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rown[0, d0 * FL + v] = T.cast(rowf[d0 * FL + v], T.float16)
                    if item < NH:
                        T.copy(rown, QN[item // GQ, item % GQ, 0])
                    else:
                        if kv4:
                            T.copy(rown, Kc[KvSlot[kvi, 1], item - NH, pos_cmx[0, 0], 0])
                        else:
                            T.copy(rown, Kc[item - NH, pos_cmx[0, 0], 0])
                        T.copy(Y[0, VOFF + (item - NH) * D], row)
                        for d in T.serial(D):
                            vcol[d, 0] = row[0, d]
                        if "v" not in stages:   # diagnostic 'v': skip the 2-byte-wide strided V^T column store
                            if kv4:
                                T.copy(vcol, Vt[KvSlot[kvi, 1], item - NH, 0, pos_cmx[0, 0]])
                            else:
                                T.copy(vcol, Vt[item - NH, 0, pos_cmx[0, 0]])
            T.sync_grid()
            # --- C. flash-decoding over this tile's third of the positions, per kv group -----------------------------
            if "C" in stages:
              for g in T.serial(NKV):
                T.copy(QN[g, 0, 0], q_cmx)
                for i, d0 in T.grid(GQ, D // FL):
                    for v in T.vectorized(FL):
                        o_acc[i, d0 * FL + v] = T.float32(0)
                for i in T.serial(GQ):
                    m_i[i] = T.float32(-1e30)
                    l_i[i] = T.float32(0)
                # The K/V rings are local, so each tile streams only its own chunks up to pos.
                kv_extent = T.min(CPT, T.ceildiv(T.max(0, pos_cmx[0, 0] + 1 - tile * BKV), BKV * tiles))
                for kv in (T.Pipelined(kv_extent, num_stages=kv_stages) if (pipelined and kv_stages > 0) else T.serial(CPT)):
                    if kv4:
                        T.copy(Kc[KvSlot[kvi, 1], g, (kv * tiles + tile) * BKV, 0], k_cmx)
                        T.copy(Vt[KvSlot[kvi, 1], g, 0, (kv * tiles + tile) * BKV], v_cmx)
                    else:
                        T.copy(Kc[g, (kv * tiles + tile) * BKV, 0], k_cmx)
                        T.copy(Vt[g, 0, (kv * tiles + tile) * BKV], v_cmx)
                    if (kv * tiles + tile) * BKV <= pos_cmx[0, 0]:
                      if "g" not in stages:
                        T.gemm(q_cmx, k_cmx, s_cmx, transpose_B=True, clear_accum=True)
                      # positions past pos: their scores are 0 (zeroed cache) -> -65504 before the max, so P = 0
                      if (kv * tiles + tile) * BKV + BKV > pos_cmx[0, 0] + 1:
                        for i, j in T.grid(GQ, BKV):
                            if (kv * tiles + tile) * BKV + j > pos_cmx[0, 0]:
                                s_cmx[i, j] = T.float16(-65504)
                      for i in T.serial(GQ):
                        for v in T.vectorized(HL):
                            mx[v] = s_cmx[i, v]
                        for j0 in T.serial(1, BKV // HL):
                            for v in T.vectorized(HL):
                                mx[v] = T.max(mx[v], s_cmx[i, j0 * HL + v])
                        m_new[i] = T.max(m_i[i], T.cast(T.call_extern(T.float16, "tl_npu_vmax", mx[0:HL]), T.float32) * scale)
                        corr[i] = T.exp(m_i[i] - m_new[i])
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
                      if "g" not in stages:
                        T.gemm(s_cmx, v_cmx, pv_cmx, transpose_B=True, clear_accum=True)
                      for i in T.serial(GQ):
                        for d0 in T.serial(D // FL):
                            for v in T.vectorized(FL):
                                o_acc[i, d0 * FL + v] = o_acc[i, d0 * FL + v] * corr[i] + T.cast(pv_cmx[i, d0 * FL + v], T.float32)
                # the tile's two partials -> one (unnormalised O, m, l) per row into Part[g, :, tile, :]
                for i, d0 in T.grid(GQ, D // FL):
                    for v in T.vectorized(FL):
                        part[shave, i, d0 * FL + v] = o_acc[i, d0 * FL + v]
                for i in T.serial(GQ):
                    part[shave, i, D] = m_i[i]
                    part[shave, i, D + 1] = l_i[i]
                T.sync_threads()
                if shave == 0:
                    for i in T.serial(GQ):
                        m_new[i] = T.max(part[0, i, D], part[1, i, D])
                        corr[i] = T.exp(part[0, i, D] - m_new[i])
                        l_i[i] = T.exp(part[1, i, D] - m_new[i])
                        for d0 in T.serial(D // FL):
                            for v in T.vectorized(FL):
                                pout[i, d0 * FL + v] = part[0, i, d0 * FL + v] * corr[i] + part[1, i, d0 * FL + v] * l_i[i]
                        pout[i, D] = m_new[i]
                        pout[i, D + 1] = part[0, i, D + 1] * corr[i] + part[1, i, D + 1] * l_i[i]
                    for i in T.serial(GQ):
                        T.copy(pout[i:i + 1, 0:PW], Part[g, i, tile, 0])
                T.sync_threads()
            T.sync_grid()
            # --- D. merge the tiles per head, apply the gate -> O ------------------------------------------------------
            if "D" in stages or "d" in stages:
              for it in T.serial(T.ceildiv(NH, workers)):
                h = it * workers + worker
                if h < NH:
                    T.copy(Part[h // GQ, h % GQ, 0, 0], p3)
                    T.copy(Y[0, h * 2 * D + D], gate)
                    if "D" in stages:
                      st[0] = p3[0, D]
                      for t in T.serial(1, tiles):
                        st[0] = T.max(st[0], p3[t, D])
                      st[1] = T.float32(0)
                      for t in T.serial(tiles):
                        if "x" in stages:
                            p3[t, D] = p3[t, D] - st[0]
                        else:
                            p3[t, D] = T.exp(p3[t, D] - st[0])     # the tile's weight
                        st[1] = st[1] + p3[t, D] * p3[t, D + 1]
                      st[1] = T.float32(1) / st[1]
                      for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rowf[d0 * FL + v] = T.float32(0)
                      for t in T.serial(tiles):
                        for d0 in T.serial(D // FL):
                            for v in T.vectorized(FL):
                                rowf[d0 * FL + v] = rowf[d0 * FL + v] + p3[t, d0 * FL + v] * p3[t, D]
                      for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            if "x" in stages:
                                gh[v] = T.float16(0) - gate[0, d0 * FL + v]
                            else:
                                gh[v] = T.exp(T.float16(0) - gate[0, d0 * FL + v])
                            oh[0, d0 * FL + v] = T.cast(rowf[d0 * FL + v] * st[1] / (T.float32(1) + T.cast(gh[v], T.float32)), T.float16)
                    T.copy(oh, O[0, h * D])
            T.sync_grid()
            # --- E. o_proj GEMV (+ residual) -------------------------------------------------------------------------
            if res_add:
                for c in T.serial(NOUT_pad - H):
                    if shave == 0:
                        res_cmx[0, H + c] = T.float16(0)
                T.copy(Res[0:1, 0:H], res_cmx[0:1, 0:H])
            T.copy(O[0, 0], o_cmx)
            if "E" in stages:
              for n in (T.Pipelined(C2, num_stages=num_stages2) if (pipelined and gemv_pipelined) else T.serial(C2)):
                T.copy(WoP[(tile * C2 + n) * BR2, 0], w2_cmx)
                for sub in T.serial(BN2 // R2):
                    T.gemm_i4(o_cmx, w2_cmx, None, y2_cmx, group=G, row0=sub * R2, rows=R2, scales_row=BN2)
                if res_add:
                    for c in T.serial(BN2):
                        y2_cmx[0, c] = T.cast(T.cast(y2_cmx[0, c], T.float32)
                                              + T.cast(res_cmx[0, (tile * C2 + n) * BN2 + c], T.float32), T.float16)
                T.copy(y2_cmx, Out[0, (tile * C2 + n) * BN2])

    return attn_mixer, sizes


def attn_sizes(H: int, NH: int, NKV: int, D: int, tiles: int = 3, BN2: int = 64, **_):
    """make_attn_decode's scratch and output sizes: dict(NQKV, GQ, PW, KO, NOUT_pad)."""
    return dict(NQKV=NH * 2 * D + 2 * NKV * D, GQ=NH // NKV, PW=D + 16, KO=NH * D,
                NOUT_pad=-(-H // (BN2 * tiles)) * BN2 * tiles)


def pack_attn(wqkv_int, wqkv_scales, wo_int, wo_scales, tiles: int = 3, BN1: int = 128, BN2: int = 64):
    """make_attn_decode's weights: the qkv rows ([per q-head q | gate] x NH, k, v; int4 values with scales) as chunk
    blocks of BN1 rows, the o_proj rows zero-padded to the tile multiple as chunk blocks of BN2 rows."""
    import numpy as np
    n_pad = -(-wo_int.shape[0] // (BN2 * tiles)) * BN2 * tiles
    wo = np.zeros((n_pad, wo_int.shape[1]), np.int8)
    so = np.zeros((n_pad, wo_scales.shape[1]), np.float16)
    wo[: wo_int.shape[0]] = wo_int
    so[: wo_scales.shape[0]] = wo_scales
    return dict(WqkvP=pack_chunk_blocks(wqkv_int, wqkv_scales, BN1), WoP=pack_chunk_blocks(wo, so, BN2))


def rope_cos_sin(pos: int, ROT: int, theta: float):
    """cos / sin rows [2, ROT] of one position (emb = cat(freqs, freqs), as the reference)."""
    import numpy as np
    inv = 1.0 / (theta ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT))
    f = pos * inv
    emb = np.concatenate([f, f])
    return np.stack([np.cos(emb), np.sin(emb)]).astype(np.float32)


def make_attn_prefill(N: int = 64, NKV: int = 2, GQ: int = 8, D: int = 256, SMAX: int = 1536, TB: int = 4,
                      BKV: int = 128, tiles: int = 3, shaves: int = 2, pipelined: bool = False):
    """P1: the full-attention PREFILL phase -- flash attention over `N` query tokens already normed, roped and
    appended to the cache, with the three things the decode phase does not need: **GQA**, the **causal mask** and
    the **output gate**.

      out[g, t, hh, :] = softmax_j(q[g, t, hh] . Kc[g, j] / sqrt(D))[j <= pos0 + t] . Vt[g, :, j]^T
                         * sigmoid(gate[g, t, hh, :])

    Work items are (kv head, token block of TB tokens) dealt round-robin over the tiles x SHAVEs; one worker owns a
    block end to end, so there is no cross-worker merge (the decode phase splits the positions instead, because at
    one token there is nothing else to split). The DPU operand is the block's `TB x GQ` query rows: Q is laid out
    `[NKV, N, GQ, D]` so one block is one contiguous DMA. Only the chunks a block can see are read (`ck` below), and
    only the chunk that straddles the diagonal is masked.

    `Pos[0, 0]` is the absolute position of token 0 of this call, so a prompt is prefilled in chunks of `N` tokens.
    Oracle: `attn_prefill_reference`."""
    assert D % 32 == 0 and BKV % 32 == 0 and SMAX % BKV == 0 and N % TB == 0
    ROWS = TB * GQ
    # One `T.gemm` is one DPU workload and the DPU takes at most 8192 output points, so the PV product bounds the
    # query block: TB x GQ x D <= 8192. Splitting that output over D would allow a larger TB, which matters because
    # a bigger block amortises the K/V re-read -- the next step, not this one.
    assert ROWS * D <= 8192, f"query block {ROWS} x {D} = {ROWS * D} output points exceeds the DPU's 8192: lower TB"
    assert ROWS * BKV <= 8192, f"score block {ROWS} x {BKV} exceeds the DPU's 8192 output points: lower TB or BKV"
    HL, FL = 32, 16
    assert BKV % HL == 0 and D % FL == 0
    scale = 1.0 / (D ** 0.5)
    workers = tiles * shaves
    NIT = NKV * (N // TB)
    CMAX = SMAX // BKV

    @T.prim_func
    def attn_prefill(Q: T.Tensor((NKV, N, GQ, D), T.float16), Kc: T.Tensor((NKV, SMAX, D), T.float16),
                     Vt: T.Tensor((NKV, D, SMAX), T.float16), Gate: T.Tensor((NKV, N, GQ, D), T.float16),
                     Pos: T.Tensor((1, 1), T.int32), O: T.Tensor((NKV, N, GQ, D), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            pos_cmx = T.alloc_shared((1, 1), T.int32)
            q_cmx = T.alloc_local((ROWS, D), T.float16)
            k_cmx = T.alloc_local((BKV, D), T.float16)
            v_cmx = T.alloc_local((D, BKV), T.float16)
            s_cmx = T.alloc_local((ROWS, BKV), T.float16)
            pv_cmx = T.alloc_local((ROWS, D), T.float16)
            o_acc = T.alloc_local((ROWS, D), T.float32)
            gt_cmx = T.alloc_local((ROWS, D), T.float16)
            m_i = T.alloc_local((ROWS,), T.float32)
            l_i = T.alloc_local((ROWS,), T.float32)
            m_new = T.alloc_local((ROWS,), T.float32)
            corr = T.alloc_local((ROWS,), T.float32)
            mx = T.alloc_local((HL,), T.float16)
            ls = T.alloc_local((FL,), T.float32)
            gh = T.alloc_local((FL,), T.float16)
            T.copy(Pos[0, 0], pos_cmx)
            for it in T.serial(T.ceildiv(NIT, workers)):
                item = it * workers + worker
                if item < NIT:
                    g = item // (N // TB)
                    t0 = (item % (N // TB)) * TB
                    T.copy(Q[g, t0, 0, 0], q_cmx)
                    for i, d0 in T.grid(ROWS, D // FL):
                        for v in T.vectorized(FL):
                            o_acc[i, d0 * FL + v] = T.float32(0)
                    for i in T.serial(ROWS):
                        m_i[i] = T.float32(-1e30)
                        l_i[i] = T.float32(0)
                    # the block's last query is at pos0 + t0 + TB - 1, so it sees chunks 0 .. that // BKV
                    ck = T.min(CMAX, T.ceildiv(pos_cmx[0, 0] + t0 + TB, BKV))
                    for kv in (T.Pipelined(ck, num_stages=2) if pipelined else T.serial(ck)):
                        T.copy(Kc[g, kv * BKV, 0], k_cmx)
                        T.copy(Vt[g, 0, kv * BKV], v_cmx)
                        T.gemm(q_cmx, k_cmx, s_cmx, transpose_B=True, clear_accum=True)
                        # causal: row i is token t0 + i // GQ at position pos0 + t0 + i // GQ
                        if kv * BKV + BKV > pos_cmx[0, 0] + t0:
                            for i, j in T.grid(ROWS, BKV):
                                if kv * BKV + j > pos_cmx[0, 0] + t0 + i // GQ:
                                    s_cmx[i, j] = T.float16(-65504)
                        for i in T.serial(ROWS):
                            for v in T.vectorized(HL):
                                mx[v] = s_cmx[i, v]
                            for j0 in T.serial(1, BKV // HL):
                                for v in T.vectorized(HL):
                                    mx[v] = T.max(mx[v], s_cmx[i, j0 * HL + v])
                            m_new[i] = T.max(m_i[i], T.cast(T.call_extern(T.float16, "tl_npu_vmax", mx[0:HL]), T.float32) * scale)
                            corr[i] = T.exp(m_i[i] - m_new[i])
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
                        T.gemm(s_cmx, v_cmx, pv_cmx, transpose_B=True, clear_accum=True)
                        for i in T.serial(ROWS):
                            for d0 in T.serial(D // FL):
                                for v in T.vectorized(FL):
                                    o_acc[i, d0 * FL + v] = o_acc[i, d0 * FL + v] * corr[i] + T.cast(pv_cmx[i, d0 * FL + v], T.float32)
                    # normalise and apply the per-head output gate
                    T.copy(Gate[g, t0, 0, 0], gt_cmx)
                    for i in T.serial(ROWS):
                        corr[i] = T.float32(1) / l_i[i]
                        for d0 in T.serial(D // FL):
                            for v in T.vectorized(FL):
                                gh[v] = T.exp(T.float16(0) - gt_cmx[i, d0 * FL + v])
                                pv_cmx[i, d0 * FL + v] = T.cast(o_acc[i, d0 * FL + v] * corr[i]
                                                                / (T.float32(1) + T.cast(gh[v], T.float32)), T.float16)
                    T.copy(pv_cmx, O[g, t0, 0, 0])

    return attn_prefill


def attn_prefill_reference(q, kc, vt, gate, pos0: int, eps: float = 0.0):
    """numpy oracle of make_attn_prefill (fp32). q / gate / out are [NKV, N, GQ, D], kc [NKV, SMAX, D],
    vt [NKV, D, SMAX]; query token t of the call is at absolute position pos0 + t and sees keys 0 .. pos0 + t."""
    import numpy as np
    NKV, N, GQ, D = q.shape
    out = np.zeros((NKV, N, GQ, D), np.float32)
    for g in range(NKV):
        for t in range(N):
            p = pos0 + t
            k = kc[g, :p + 1].astype(np.float32)        # [p + 1, D]
            v = vt[g, :, :p + 1].astype(np.float32)     # [D, p + 1]
            for hh in range(GQ):
                s = k @ q[g, t, hh].astype(np.float32) / np.sqrt(D)
                pr = np.exp(s - s.max())
                pr /= pr.sum()
                gt = gate[g, t, hh].astype(np.float32)
                out[g, t, hh] = (v @ pr) / (1.0 + np.exp(-gt))
    return out


def attn_reference(x, res, wqkv_int, wqkv_scales, wo_int, wo_scales, qnorm_w, knorm_w, cos_sin, pos, kc, vt,
                   NH: int, NKV: int, D: int, ROT: int, G: int, eps: float = 1e-6):
    """numpy oracle of make_attn_decode (fp32): returns out [H], the K cache and transposed V cache after the append."""
    import numpy as np
    GQ = NH // NKV
    xf = x.astype(np.float32).reshape(-1)
    w = wqkv_int.astype(np.float32) * np.repeat(wqkv_scales.astype(np.float32), G, axis=1)
    y = w @ xf
    KOFF, VOFF = NH * 2 * D, NH * 2 * D + NKV * D

    def norm_rope(v, wn):
        v = v * (1.0 / np.sqrt((v * v).mean() + eps)) * (1.0 + wn.astype(np.float32).reshape(-1))
        r = v[:ROT].copy()
        h = ROT // 2
        rot = np.concatenate([-r[h:], r[:h]])
        v = v.copy()
        v[:ROT] = r * cos_sin[0] + rot * cos_sin[1]
        return v

    kc = kc.copy()
    vt = vt.copy()
    q = np.zeros((NH, D), np.float32)
    for hh in range(NH):
        q[hh] = norm_rope(y[hh * 2 * D:hh * 2 * D + D], qnorm_w)
    for g in range(NKV):
        kc[g, pos] = norm_rope(y[KOFF + g * D:KOFF + (g + 1) * D], knorm_w).astype(np.float16)
        vt[g, :, pos] = y[VOFF + g * D:VOFF + (g + 1) * D].astype(np.float16)
    o = np.zeros((NH, D), np.float32)
    for hh in range(NH):
        g = hh // GQ
        sc = kc[g, :pos + 1].astype(np.float32) @ q[hh] / np.sqrt(D)
        p = np.exp(sc - sc.max())
        p /= p.sum()
        gate = y[hh * 2 * D + D:hh * 2 * D + 2 * D]
        o[hh] = (vt[g, :, :pos + 1].astype(np.float32) @ p) / (1 + np.exp(-gate))
    wo = wo_int.astype(np.float32) * np.repeat(wo_scales.astype(np.float32), G, axis=1)
    out = res.astype(np.float32).reshape(-1) + wo @ o.reshape(-1)
    return out, kc, vt


def pack_int4(w_int: "np.ndarray") -> "np.ndarray":
    """[N, K] signed int4 values in [-8, 7] -> [N, K/2] bytes, low nibble first (the reference packer)."""
    import numpy as np
    q = (w_int.astype(np.int16) & 0xF).astype(np.uint8)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)


def make_head(H: int = 2048, V: int = 248320, G: int = 128, tiles: int = 3, shaves: int = 2, BN: int = 128, R: int = 64,
              num_stages: int = 4, eps: float = 1e-6, pipelined: bool = True, h_dtype=T.float16, as_block: bool = False):
    """M5 decode head in ONE launch: logits = rmsnorm_zc(h) . dequant_i8(Wlm)^T over the vocabulary, greedy argmax.

      x = h rsqrt(mean h^2 + eps) (1 + w)                 (every tile, SHAVE 0; the final norm)
      logits[v] = x . Wlm[v]                              (int8 g128 rows as chunk blocks, split over the tiles, a
                                                           local ring per SHAVE; the lm_head is the largest decode
                                                           weight: V x H bytes)
      tok = argmax_v logits[v]                            (per SHAVE on the fly: a lane max per chunk, the index scan
                                                           only when the chunk beats the running max; the six partials
                                                           meet in DDR and worker 0 picks the lowest index on a tie)

    The vocabulary is zero-padded to a multiple of BN x tiles x shaves (`head_sizes`); padded columns are masked to
    -65504 before the max. Each SHAVE keeps its chunks' logits in CMX and stores them once at the end (a strided
    2-D job; `Logits` is [tiles, CS, shaves x BN] = the flat [V_pad] row). Partials: `Part[worker, 0:2]` = (max, index);
    `Tok[0, 0]` the token. Packing: `pack_head`; sizes: `head_sizes`."""
    assert H % G == 0 and BN % R == 0 and R % 16 == 0 and R * (H // G) <= 1024
    ng = H // G
    V_pad = -(-V // (BN * tiles * shaves)) * BN * tiles * shaves
    C = V_pad // (BN * tiles)
    CS = C // shaves
    assert (ng * BN * 2) % H == 0
    BR = BN + ng * BN * 2 // H
    workers = tiles * shaves
    HL, FL = 32, 16
    PW = 32

    @T.macro
    def consume(x_cmx, w_cmx, y_cmx, yl, mx, st, bi, tile, n):
        """One chunk after its fill: the BN / R merged-FC workloads, the pad mask, the running max, the logits row."""
        for sub in T.serial(BN // R):
            T.gemm_i8(x_cmx, w_cmx, None, y_cmx, group=G, row0=sub * R, rows=R, scales_row=BN)
        if (tile * C + n) * BN + BN > V:
            for c in T.serial(BN):
                if (tile * C + n) * BN + c >= V:
                    y_cmx[0, c] = T.float16(-65504)
        for v in T.vectorized(HL):
            mx[v] = y_cmx[0, v]
        for j0 in T.serial(1, BN // HL):
            for v in T.vectorized(HL):
                mx[v] = T.max(mx[v], y_cmx[0, j0 * HL + v])
        st[1] = T.cast(T.call_extern(T.float16, "tl_npu_vmax", mx[0:HL]), T.float32)
        if st[1] > st[0]:
            for c in T.serial(BN):
                if T.cast(y_cmx[0, c], T.float32) > st[0]:
                    st[0] = T.cast(y_cmx[0, c], T.float32)
                    bi[0] = (tile * C + n) * BN + c
        for j0 in T.serial(BN // HL):
            for v in T.vectorized(HL):
                yl[n // shaves, j0 * HL + v] = y_cmx[0, j0 * HL + v]

    @T.macro
    def head_phases(x_cmx, WlmP, Logits, Part, Tok, tile, shave):
        """Phases B and C over the normed token `x_cmx` (shared fp16 [1, H]): the vocabulary GEMV with the running
        argmax, the partials, the grid barrier, worker 0's pick into Tok."""
        worker = tile * shaves + shave
        w_cmx = T.alloc_shared((BR, H), T.int8)
        y_cmx = T.alloc_local((1, BN), T.float16)
        yl = T.alloc_local((CS, BN), T.float16)
        st = T.alloc_local((4,), T.float32)
        bi = T.alloc_local((4,), T.int32)
        mx = T.alloc_local((HL,), T.float16)
        pw = T.alloc_local((1, PW), T.float32)
        pall = T.alloc_local((workers, PW), T.float32)
        tok_cmx = T.alloc_local((1, 8), T.int32)
        # chunk n belongs to SHAVE n % shaves (the ring's dealing; the serial form deals it by hand)
        st[0] = T.float32(-3e38)
        bi[0] = 0
        if pipelined:
            for n in T.Pipelined(C, num_stages=num_stages):
                T.copy(WlmP[(tile * C + n) * BR, 0], w_cmx)
                consume(x_cmx, w_cmx, y_cmx, yl, mx, st, bi, tile, n)
        else:
            for n in T.serial(C):
                T.copy(WlmP[(tile * C + n) * BR, 0], w_cmx)
                if n % shaves == shave:
                    consume(x_cmx, w_cmx, y_cmx, yl, mx, st, bi, tile, n)
        T.copy(yl, Logits[tile, 0, shave * BN])
        for c in T.serial(PW):
            pw[0, c] = T.float32(0)
        pw[0, 0] = st[0]
        pw[0, 1] = T.cast(bi[0], T.float32)
        T.copy(pw, Part[worker, 0])
        T.sync_grid()
        # worker 0: the lowest index among the partials sharing the max
        if worker == 0:
            T.copy(Part[0, 0], pall)
            st[0] = pall[0, 0]
            bi[0] = T.cast(pall[0, 1], T.int32)
            for w in T.serial(1, workers):
                if pall[w, 0] > st[0] or (pall[w, 0] == st[0] and T.cast(pall[w, 1], T.int32) < bi[0]):
                    st[0] = pall[w, 0]
                    bi[0] = T.cast(pall[w, 1], T.int32)
            for c in T.serial(8):
                tok_cmx[0, c] = 0
            tok_cmx[0, 0] = bi[0]
            T.copy(tok_cmx, Tok[0, 0])

    if as_block:
        return head_phases, dict(V_pad=V_pad, C=C, CS=CS, BR=BR, workers=workers, PW=PW)

    @T.prim_func
    def head(Hs: T.Tensor((1, H), h_dtype), NormW: T.Tensor((1, H), T.float16),
             WlmP: T.Tensor(((V_pad // BN) * BR, H), T.int8),
             Logits: T.Tensor((tiles, CS, shaves * BN), T.float16), Part: T.Tensor((workers, PW), T.float32),
             Tok: T.Tensor((1, 8), T.int32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            h_cmx = T.alloc_shared((1, H), h_dtype)
            nw_cmx = T.alloc_shared((1, H), T.float16)
            x_cmx = T.alloc_shared((1, H), T.float16)
            ls = T.alloc_local((FL,), T.float32)
            st = T.alloc_local((4,), T.float32)
            # --- A. final norm (zero-centred) on SHAVE 0 of every tile -----------------------------------------------
            T.copy(Hs[0, 0], h_cmx)
            T.copy(NormW[0, 0], nw_cmx)
            if shave == 0:
                for v in T.vectorized(FL):
                    ls[v] = T.float32(0)
                for d0 in T.serial(H // FL):
                    for v in T.vectorized(FL):
                        ls[v] = ls[v] + T.cast(h_cmx[0, d0 * FL + v], T.float32) * T.cast(h_cmx[0, d0 * FL + v], T.float32)
                st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(H) + T.float32(eps))
                for d0 in T.serial(H // FL):
                    for v in T.vectorized(FL):
                        x_cmx[0, d0 * FL + v] = T.cast(T.cast(h_cmx[0, d0 * FL + v], T.float32) * st[0]
                                                       * (T.float32(1) + T.cast(nw_cmx[0, d0 * FL + v], T.float32)), T.float16)
            T.sync_threads()
            head_phases(x_cmx, WlmP, Logits, Part, Tok, tile, shave)

    return head


def head_sizes(H: int, V: int, G: int = 128, tiles: int = 3, shaves: int = 2, BN: int = 128, **_):
    """make_head's padded vocabulary and block geometry: dict(V_pad, C, CS, BR, workers)."""
    V_pad = -(-V // (BN * tiles * shaves)) * BN * tiles * shaves
    C = V_pad // (BN * tiles)
    return dict(V_pad=V_pad, C=C, CS=C // shaves, BR=BN + (H // G) * BN * 2 // H, workers=tiles * shaves)


def pack_chunk_blocks_i8(w_int: "np.ndarray", scales: "np.ndarray", BN: int):
    """`pack_chunk_blocks` for int8 weights (`T.gemm_i8(..., scales_row=BN)`): [N, K] int8 and [N, K/G] scales ->
    int8 [N/BN x (BN + SR), K], each chunk's rows then its group-major fp16 scales [K/G, BN] in whole rows."""
    import numpy as np
    N, K = w_int.shape
    ng = scales.shape[1]
    assert N % BN == 0 and (ng * BN * 2) % K == 0
    SR = ng * BN * 2 // K
    blocks = []
    for n0 in range(0, N, BN):
        sT = np.ascontiguousarray(scales[n0:n0 + BN].astype(np.float16).T)   # [ng, BN]
        blocks.append(np.concatenate([w_int[n0:n0 + BN].astype(np.int8), sT.view(np.int8).reshape(SR, K)], axis=0))
    return np.concatenate(blocks, axis=0)


def pack_head(wlm_int, wlm_scales, tiles: int = 3, shaves: int = 2, BN: int = 128):
    """make_head's lm_head: int8 rows zero-padded to the V_pad multiple, as chunk blocks of BN rows."""
    import numpy as np
    V, H = wlm_int.shape
    V_pad = head_sizes(H, V, tiles=tiles, shaves=shaves, BN=BN)["V_pad"]
    w = np.zeros((V_pad, H), np.int8)
    s = np.zeros((V_pad, wlm_scales.shape[1]), np.float16)
    w[:V] = wlm_int
    s[:V] = wlm_scales
    return dict(WlmP=pack_chunk_blocks_i8(w, s, BN))


def head_reference(h, norm_w, wlm_int, wlm_scales, G: int, eps: float = 1e-6, block: int = 8192):
    """numpy oracle of make_head (fp32, the rows dequantised `block` at a time): returns logits [V] and the token."""
    import numpy as np
    hf = h.astype(np.float32).reshape(-1)
    x = hf * (1.0 / np.sqrt((hf * hf).mean() + eps)) * (1.0 + norm_w.astype(np.float32).reshape(-1))
    V = wlm_int.shape[0]
    logits = np.empty(V, np.float32)
    for v0 in range(0, V, block):
        w = wlm_int[v0:v0 + block].astype(np.float32) * np.repeat(wlm_scales[v0:v0 + block].astype(np.float32), G, axis=1)
        logits[v0:v0 + block] = w @ x
    return logits, int(np.argmax(logits))


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


def norm_block(H: int, shaves: int, eps: float = 1e-6):
    """The residual stream's norms as `T.macro`s over tile-shared CMX: `rmsnorm_zc(h32, w_cmx, x_cmx, shave)` writes
    x = h rsqrt(mean h^2 + eps) (1 + w) in fp16 (SHAVE 0; tile sync after); `add_rows(h32, a_cmx, shave)` adds an
    fp16 row into the fp32 stream (both SHAVEs, halves; tile sync after); `add_parts(h32, p_cmx, parts, shave)` adds
    `parts` fp32 rows of p_cmx. Returns (rmsnorm_zc, add_rows, add_parts)."""
    FL = 16
    assert H % (shaves * FL) == 0
    HS = H // shaves

    @T.macro
    def rmsnorm_zc(h32, w_cmx, x_cmx, shave):
        ls = T.alloc_local((FL,), T.float32)
        st = T.alloc_local((4,), T.float32)
        if shave == 0:
            for v in T.vectorized(FL):
                ls[v] = T.float32(0)
            for d0 in T.serial(H // FL):
                for v in T.vectorized(FL):
                    ls[v] = ls[v] + h32[0, d0 * FL + v] * h32[0, d0 * FL + v]
            st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(H) + T.float32(eps))
            for d0 in T.serial(H // FL):
                for v in T.vectorized(FL):
                    x_cmx[0, d0 * FL + v] = T.cast(h32[0, d0 * FL + v] * st[0] * (T.float32(1) + T.cast(w_cmx[0, d0 * FL + v], T.float32)),
                                                   T.float16)
        T.sync_threads()

    @T.macro
    def add_rows(h32, a_cmx, shave):
        for d0 in T.serial(HS // FL):
            for v in T.vectorized(FL):
                h32[0, shave * HS + d0 * FL + v] = h32[0, shave * HS + d0 * FL + v] + T.cast(a_cmx[0, shave * HS + d0 * FL + v], T.float32)
        T.sync_threads()

    @T.macro
    def add_parts(h32, p_cmx, parts, shave):
        for t in T.serial(parts):
            for d0 in T.serial(HS // FL):
                for v in T.vectorized(FL):
                    h32[0, shave * HS + d0 * FL + v] = h32[0, shave * HS + d0 * FL + v] + p_cmx[t, shave * HS + d0 * FL + v]
        T.sync_threads()

    return rmsnorm_zc, add_rows, add_parts


def make_gdn_layer(H: int = 2048, KH: int = 16, VH: int = 32, DK: int = 128, DV: int = 128, I: int = 512, E: int = 256,
                   TOPK: int = 8, G: int = 128, tiles: int = 3, shaves: int = 2, eps: float = 1e-6, pipelined: bool = True,
                   gdn_kw: dict | None = None, moe_kw: dict | None = None):
    """M5: one whole Gated DeltaNet decoder layer in ONE launch, the residual stream in fp32 in DDR (`Hio`, read at
    entry and written at exit, every tile holding a copy in CMX): x = rmsnorm_zc(h, in_norm) -> the GDN mixer
    (`gdn_block`, no residual) -> h += mix -> x = rmsnorm_zc(h, post_norm) -> MoE (`moe_block`) -> h += the tiles'
    partials. The host launches it with the previous layer's `Hio` and does nothing else. Scratch as the two
    standalone kernels (Y, BA, CV, O, Mix for the mixer; Logits, PartM for the MoE)."""
    gdn_mixer, gs = gdn_block(H=H, KH=KH, VH=VH, DK=DK, DV=DV, G=G, tiles=tiles, shaves=shaves, eps=eps, pipelined=pipelined,
                              res_add=False, **(gdn_kw or {}))
    mk = dict(num_stages=4, BNR=16, BNG=128, BND=256)
    mk.update(moe_kw or {})
    moe_phases = moe_block(H=H, I=I, E=E, TOPK=TOPK, G=G, tiles=tiles, shaves=shaves, pipelined=pipelined, **mk)
    rmsnorm_zc, add_rows, add_parts = norm_block(H, shaves, eps)
    BN1, BN2 = (gdn_kw or {}).get("BN1", 128), (gdn_kw or {}).get("BN2", 64)
    CONV, NIN_pad, NOUT_pad, BR1, BR2, VC = gs["CONV"], gs["NIN_pad"], gs["NOUT_pad"], gs["BR1"], gs["BR2"], gs["VC"]
    WBA_pad, YLEN, BA_LEN, CVLEN, OPAD = gs["WBA_pad"], gs["YLEN"], gs["BA_LEN"], gs["CVLEN"], gs["OPAD"]
    per_round = tiles * shaves * mk["BNR"]
    RR = -(-(E + 1) // per_round) * per_round
    CG, CD = (2 * I) // mk["BNG"], H // mk["BND"]
    SRG, SRD = moe_scale_rows(H, I, G, mk["BNG"], mk["BND"])
    BG, BD = mk["BNG"] + SRG, mk["BND"] + SRD
    MR = 16

    @T.prim_func
    def gdn_layer(Hio: T.Tensor((1, H), T.float32), InNorm: T.Tensor((1, H), T.float16), PostNorm: T.Tensor((1, H), T.float16),
                  WinP: T.Tensor(((NIN_pad // BN1) * BR1, H // 2), T.uint8), Wba: T.Tensor((WBA_pad, H), T.float16),
                  ConvW: T.Tensor((4, CONV), T.float16), ALog: T.Tensor((1, VH), T.float32), DtBias: T.Tensor((1, VH), T.float32),
                  GNorm: T.Tensor((1, DV), T.float16), ConvState: T.Tensor((3, CONV), T.float16),
                  Ssm: T.Tensor((VH, DK, DV), T.float32), WoutP: T.Tensor(((NOUT_pad // BN2) * BR2, VC // 2), T.uint8),
                  Wr: T.Tensor((RR, H), T.float16), WguP: T.Tensor(((E + 1) * CG * BG, H // 2), T.uint8),
                  WdP: T.Tensor(((E + 1) * CD * BD, I // 2), T.uint8),
                  Y: T.Tensor((1, YLEN), T.float16), BA: T.Tensor((1, BA_LEN), T.float16), CV: T.Tensor((1, CVLEN), T.float16),
                  O: T.Tensor((1, OPAD), T.float16), Mix: T.Tensor((1, NOUT_pad), T.float16),
                  Logits: T.Tensor((MR, RR), T.float16), PartM: T.Tensor((tiles, H), T.float32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            h32 = T.alloc_shared((1, H), T.float32)
            x_cmx = T.alloc_shared((1, H), T.float16)
            nw_cmx = T.alloc_shared((1, H), T.float16)
            mix_cmx = T.alloc_shared((1, H), T.float16)
            p_cmx = T.alloc_shared((tiles, H), T.float32)
            T.copy(Hio[0, 0], h32)
            T.copy(InNorm[0, 0], nw_cmx)
            rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
            gdn_mixer(x_cmx, WinP, Wba, ConvW, ALog, DtBias, GNorm, ConvState, Ssm, WoutP, Y, BA, CV, O, Mix, Mix, tile, shave)
            T.sync_grid()
            T.copy(Mix[0:1, 0:H], mix_cmx[0:1, 0:H])
            add_rows(h32, mix_cmx, shave)
            T.copy(PostNorm[0, 0], nw_cmx)
            rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
            moe_phases(x_cmx, Wr, WguP, WdP, Logits, PartM, tile, shave)
            T.copy(PartM[0, 0], p_cmx)
            add_parts(h32, p_cmx, tiles, shave)
            if tile == 0:
                if shave == 0:
                    T.copy(h32, Hio[0, 0])

    return gdn_layer


def make_attn_layer(H: int = 2048, NH: int = 16, NKV: int = 2, D: int = 256, ROT: int = 64, SMAX: int = 1536, I: int = 512,
                    E: int = 256, TOPK: int = 8, G: int = 128, tiles: int = 3, shaves: int = 2, eps: float = 1e-6,
                    pipelined: bool = True, attn_kw: dict | None = None, moe_kw: dict | None = None):
    """M5: one whole full-attention decoder layer in ONE launch (make_gdn_layer's shape with `attn_block` as the mixer;
    Pos / CosSin as the standalone attention kernel)."""
    attn_mixer, asz = attn_block(H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, G=G, tiles=tiles, shaves=shaves, eps=eps,
                                 pipelined=pipelined, res_add=False, **(attn_kw or {}))
    mk = dict(num_stages=4, BNR=16, BNG=128, BND=256)
    mk.update(moe_kw or {})
    moe_phases = moe_block(H=H, I=I, E=E, TOPK=TOPK, G=G, tiles=tiles, shaves=shaves, pipelined=pipelined, **mk)
    rmsnorm_zc, add_rows, add_parts = norm_block(H, shaves, eps)
    BN1, BN2 = (attn_kw or {}).get("BN1", 128), (attn_kw or {}).get("BN2", 64)
    NQKV, NOUT_pad, BR1, BR2, KO, GQ, PW = asz["NQKV"], asz["NOUT_pad"], asz["BR1"], asz["BR2"], asz["KO"], asz["GQ"], asz["PW"]
    per_round = tiles * shaves * mk["BNR"]
    RR = -(-(E + 1) // per_round) * per_round
    CG, CD = (2 * I) // mk["BNG"], H // mk["BND"]
    SRG, SRD = moe_scale_rows(H, I, G, mk["BNG"], mk["BND"])
    BG, BD = mk["BNG"] + SRG, mk["BND"] + SRD
    MR = 16

    @T.prim_func
    def attn_layer(Hio: T.Tensor((1, H), T.float32), InNorm: T.Tensor((1, H), T.float16), PostNorm: T.Tensor((1, H), T.float16),
                   WqkvP: T.Tensor(((NQKV // BN1) * BR1, H // 2), T.uint8), QNormW: T.Tensor((1, D), T.float16),
                   KNormW: T.Tensor((1, D), T.float16), CosSin: T.Tensor((2, ROT), T.float32), Pos: T.Tensor((1, 1), T.int32),
                   Kc: T.Tensor((NKV, SMAX, D), T.float16), Vt: T.Tensor((NKV, D, SMAX), T.float16),
                   WoP: T.Tensor(((NOUT_pad // BN2) * BR2, KO // 2), T.uint8),
                   Wr: T.Tensor((RR, H), T.float16), WguP: T.Tensor(((E + 1) * CG * BG, H // 2), T.uint8),
                   WdP: T.Tensor(((E + 1) * CD * BD, I // 2), T.uint8),
                   Y: T.Tensor((1, NQKV), T.float16), QN: T.Tensor((NKV, GQ, D), T.float16),
                   PartA: T.Tensor((NKV, GQ, tiles, PW), T.float32), O: T.Tensor((1, KO), T.float16),
                   Mix: T.Tensor((1, NOUT_pad), T.float16), Logits: T.Tensor((MR, RR), T.float16),
                   PartM: T.Tensor((tiles, H), T.float32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            h32 = T.alloc_shared((1, H), T.float32)
            x_cmx = T.alloc_shared((1, H), T.float16)
            nw_cmx = T.alloc_shared((1, H), T.float16)
            mix_cmx = T.alloc_shared((1, H), T.float16)
            p_cmx = T.alloc_shared((tiles, H), T.float32)
            T.copy(Hio[0, 0], h32)
            T.copy(InNorm[0, 0], nw_cmx)
            rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
            attn_mixer(x_cmx, WqkvP, QNormW, KNormW, CosSin, Pos, Kc, Vt, WoP, Y, QN, PartA, O, Mix, Mix, tile, shave, Pos, 0)
            T.sync_grid()
            T.copy(Mix[0:1, 0:H], mix_cmx[0:1, 0:H])
            add_rows(h32, mix_cmx, shave)
            T.copy(PostNorm[0, 0], nw_cmx)
            rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
            moe_phases(x_cmx, Wr, WguP, WdP, Logits, PartM, tile, shave)
            T.copy(PartM[0, 0], p_cmx)
            add_parts(h32, p_cmx, tiles, shave)
            if tile == 0:
                if shave == 0:
                    T.copy(h32, Hio[0, 0])

    return attn_layer


# Layer table columns (`make_decode_model`): one int64 device address per column, filled by the host from
# `Session.address`; a column a layer kind does not use holds 0.
TABLE_COLS = ["InNorm", "PostNorm", "W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "Wr", "WguP", "WdP", "S0", "S1", "pad"]
GDN_TABLE = dict(W0="WinP", W1="Wba", W2="ConvW", W3="ALog", W4="DtBias", W5="GNorm", W6="WoutP", S0="ConvState", S1="Ssm")
# The attention KV cache is a kernel parameter (`KcAll` / `VtAll`), not a table address: see `attn_block`'s `kv4`.
# `Kind[li]` is (kind, kv_slot, 0, 0): kv_slot is the layer's index among the launch's attention layers.
ATTN_TABLE = dict(W0="WqkvP", W1="QNormW", W2="KNormW", W3="WoP")


def make_decode_model(L: int, H: int = 2048, KH: int = 16, VH: int = 32, DK: int = 128, DV: int = 128, NH: int = 16, NKV: int = 2,
                      D: int = 256, ROT: int = 64, SMAX: int = 1536, I: int = 512, E: int = 256, TOPK: int = 8, V: int = 248320,
                      G: int = 128, tiles: int = 3, shaves: int = 2, eps: float = 1e-6, pipelined: bool = True,
                      gdn_kw: dict | None = None, attn_kw: dict | None = None, moe_kw: dict | None = None, head_kw: dict | None = None,
                      skip: frozenset = frozenset(), KVL: int | None = None):
    """M5: the whole decode step in ONE launch: a device-side loop over the L layers (`Kind[li]` 0 = Gated DeltaNet,
    1 = full attention), each layer's weights and states reached through `Table[li, :]` (device addresses,
    `T.ddr_view`, columns `TABLE_COLS`), the residual stream fp32 in CMX for the whole step (written back to Hio at the
    end), then the final norm and the lm_head with the greedy argmax into Tok. The host uploads the embedding row into
    Hio, Pos and CosSin, launches once, reads Tok. The two layer kinds' phases share CMX (scoped), the
    `if Kind == 0` branches are uniform across the SHAVEs, and every ring stays inside its layer (LowerIntelNpuPipelined
    refuses to hoist a ring whose fill source is a view bound inside the loop). The out_proj gemms submit
    R2 = 16 rows (not 32): the runtime's merged-FC scratch is sized by the largest quantised gemm of the whole kernel
    (M x ng^2 x N, K = 4096 -> ng = 32) and is not scoped, so at R2 = 32 it costs 64 KiB more than the head alone
    and the 1 MB lm_head ring no longer fits next to it (CheckIntelNpuBudgets refuses by 16 KiB). `skip` (diagnostic
    only) leaves out whole stages: any of "gdn", "attn", "moe", "head"."""
    gk = dict(R2=16)
    gk.update(gdn_kw or {})
    ak = dict(R2=16)
    ak.update(attn_kw or {})
    ak.setdefault("pipelined", pipelined)
    gdn_mixer, gs = gdn_block(H=H, KH=KH, VH=VH, DK=DK, DV=DV, G=G, tiles=tiles, shaves=shaves, eps=eps, pipelined=pipelined,
                              res_add=False, **gk)
    attn_mixer, asz = attn_block(H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, G=G, tiles=tiles, shaves=shaves, eps=eps,
                                 res_add=False, kv4=True, **ak)
    mk = dict(num_stages=4, BNR=16, BNG=128, BND=256)
    mk.update(moe_kw or {})
    moe_phases = moe_block(H=H, I=I, E=E, TOPK=TOPK, G=G, tiles=tiles, shaves=shaves, pipelined=pipelined, **mk)
    head_phases, hs = make_head(H=H, V=V, G=G, tiles=tiles, shaves=shaves, eps=eps, pipelined=pipelined, as_block=True, **(head_kw or {}))
    rmsnorm_zc, add_rows, add_parts = norm_block(H, shaves, eps)
    BN1g, BN2g = (gdn_kw or {}).get("BN1", 128), (gdn_kw or {}).get("BN2", 64)
    BN1a, BN2a = (attn_kw or {}).get("BN1", 128), (attn_kw or {}).get("BN2", 64)
    BNh = (head_kw or {}).get("BN", 128)
    CONV, NIN_pad, NOUTg, BR1g, BR2g, VC = gs["CONV"], gs["NIN_pad"], gs["NOUT_pad"], gs["BR1"], gs["BR2"], gs["VC"]
    WBA_pad, YLEN, BA_LEN, CVLEN, OPADg = gs["WBA_pad"], gs["YLEN"], gs["BA_LEN"], gs["CVLEN"], gs["OPAD"]
    NQKV, NOUTa, BR1a, BR2a, KO, GQ, PWa = asz["NQKV"], asz["NOUT_pad"], asz["BR1"], asz["BR2"], asz["KO"], asz["GQ"], asz["PW"]
    per_round = tiles * shaves * mk["BNR"]
    RR = -(-(E + 1) // per_round) * per_round
    CG, CD = (2 * I) // mk["BNG"], H // mk["BND"]
    SRG, SRD = moe_scale_rows(H, I, G, mk["BNG"], mk["BND"])
    BG, BD = mk["BNG"] + SRG, mk["BND"] + SRD
    MR = 16
    NCOL = len(TABLE_COLS)
    col = {n: i for i, n in enumerate(TABLE_COLS)}
    KVL = L if KVL is None else KVL   # KV cache slots: one per attention layer of this launch, not per layer

    @T.prim_func
    def model(Hio: T.Tensor((1, H), T.float32), Pos: T.Tensor((1, 1), T.int32), CosSin: T.Tensor((2, ROT), T.float32),
              Kind: T.Tensor((L, 4), T.int32), Table: T.Tensor((L, NCOL), T.int64),
              KcAll: T.Tensor((KVL, NKV, SMAX, D), T.float16), VtAll: T.Tensor((KVL, NKV, D, SMAX), T.float16),
              Y: T.Tensor((1, YLEN), T.float16), BA: T.Tensor((1, BA_LEN), T.float16), CV: T.Tensor((1, CVLEN), T.float16),
              O: T.Tensor((1, OPADg), T.float16), Mix: T.Tensor((1, NOUTg), T.float16),
              YA: T.Tensor((1, NQKV), T.float16), QN: T.Tensor((NKV, GQ, D), T.float16),
              PartA: T.Tensor((NKV, GQ, tiles, PWa), T.float32), OA: T.Tensor((1, KO), T.float16), MixA: T.Tensor((1, NOUTa), T.float16),
              Logits: T.Tensor((MR, RR), T.float16), PartM: T.Tensor((tiles, H), T.float32),
              NormW: T.Tensor((1, H), T.float16), WlmP: T.Tensor(((hs["V_pad"] // BNh) * hs["BR"], H), T.int8),
              HLogits: T.Tensor((tiles, hs["CS"], shaves * BNh), T.float16), HPart: T.Tensor((hs["workers"], hs["PW"]), T.float32),
              Tok: T.Tensor((1, 8), T.int32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            h32 = T.alloc_shared((1, H), T.float32)
            x_cmx = T.alloc_shared((1, H), T.float16)
            nw_cmx = T.alloc_shared((1, H), T.float16)
            mix_cmx = T.alloc_shared((1, H), T.float16)
            p_cmx = T.alloc_shared((tiles, H), T.float32)
            T.copy(Hio[0, 0], h32)
            for li in T.serial(L):
                InNorm = T.ddr_view(Table[li, col["InNorm"]], (1, H), T.float16)
                PostNorm = T.ddr_view(Table[li, col["PostNorm"]], (1, H), T.float16)
                Wr = T.ddr_view(Table[li, col["Wr"]], (RR, H), T.float16)
                WguP = T.ddr_view(Table[li, col["WguP"]], ((E + 1) * CG * BG, H // 2), T.uint8)
                WdP = T.ddr_view(Table[li, col["WdP"]], ((E + 1) * CD * BD, I // 2), T.uint8)
                perf_begin("lhed")
                T.copy(InNorm[0, 0], nw_cmx)
                if "norm" not in skip:
                    rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
                perf_end()
                if Kind[li, 0] == 0:
                    WinP = T.ddr_view(Table[li, col["W0"]], ((NIN_pad // BN1g) * BR1g, H // 2), T.uint8)
                    Wba = T.ddr_view(Table[li, col["W1"]], (WBA_pad, H), T.float16)
                    ConvW = T.ddr_view(Table[li, col["W2"]], (4, CONV), T.float16)
                    ALog = T.ddr_view(Table[li, col["W3"]], (1, VH), T.float32)
                    DtBias = T.ddr_view(Table[li, col["W4"]], (1, VH), T.float32)
                    GNorm = T.ddr_view(Table[li, col["W5"]], (1, DV), T.float16)
                    WoutP = T.ddr_view(Table[li, col["W6"]], ((NOUTg // BN2g) * BR2g, VC // 2), T.uint8)
                    ConvState = T.ddr_view(Table[li, col["S0"]], (3, CONV), T.float16)
                    Ssm = T.ddr_view(Table[li, col["S1"]], (VH, DK, DV), T.float32)
                    if "gdn" not in skip:
                        gdn_mixer(x_cmx, WinP, Wba, ConvW, ALog, DtBias, GNorm, ConvState, Ssm, WoutP, Y, BA, CV, O, Mix, Mix, tile, shave)
                        perf_begin("mixh")
                        T.sync_grid()
                        T.copy(Mix[0:1, 0:H], mix_cmx[0:1, 0:H])
                        perf_end()
                    else:
                        if shave == 0:
                            for c in T.serial(H):
                                mix_cmx[0, c] = x_cmx[0, c]
                        T.sync_threads()
                else:
                    WqkvP = T.ddr_view(Table[li, col["W0"]], ((NQKV // BN1a) * BR1a, H // 2), T.uint8)
                    QNormW = T.ddr_view(Table[li, col["W1"]], (1, D), T.float16)
                    KNormW = T.ddr_view(Table[li, col["W2"]], (1, D), T.float16)
                    WoP = T.ddr_view(Table[li, col["W3"]], ((NOUTa // BN2a) * BR2a, KO // 2), T.uint8)
                    if "attn" not in skip:
                        attn_mixer(x_cmx, WqkvP, QNormW, KNormW, CosSin, Pos, KcAll, VtAll, WoP, YA, QN, PartA, OA, MixA, MixA,
                                   tile, shave, Kind, li)
                        T.sync_grid()
                        T.copy(MixA[0:1, 0:H], mix_cmx[0:1, 0:H])
                    else:
                        if shave == 0:
                            for c in T.serial(H):
                                mix_cmx[0, c] = x_cmx[0, c]
                        T.sync_threads()
                perf_begin("ltal")
                add_rows(h32, mix_cmx, shave)
                T.copy(PostNorm[0, 0], nw_cmx)
                if "norm" not in skip:
                    rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
                perf_end()
                if "moe" not in skip:
                    moe_phases(x_cmx, Wr, WguP, WdP, Logits, PartM, tile, shave)
                    perf_begin("lprt")
                    T.copy(PartM[0, 0], p_cmx)
                    add_parts(h32, p_cmx, tiles, shave)
                    perf_end()
            # a grid barrier closes the layer loop's CMX phase before the head's ring (the lm_head ring is 1 MB)
            T.sync_grid()
            T.copy(NormW[0, 0], nw_cmx)
            rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
            if "head" not in skip:
                head_phases(x_cmx, WlmP, HLogits, HPart, Tok, tile, shave)
            if tile == 0:
                if shave == 0:
                    T.copy(h32, Hio[0, 0])

    return model
