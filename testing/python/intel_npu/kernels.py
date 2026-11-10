"""Phase 1 GEMM kernel for the Intel NPU: C[M, N] = A[M, K] . B[N, K]^T in fp16.

One persistent launch over `tiles` x `shaves_per_tile` SHAVEs; the output blocks
are dealt round-robin over the workers. The DPU takes the whole K per
workload, so A and B tiles are [BM, K] and [BN, K] in CMX.

No `from __future__ import annotations`: T.prim_func evaluates the tensor
annotations against the factory's locals.
"""

import math
import os

import tilelang.intel_npu.language as T


@T.macro
def perf_begin(name: str):
    """A named SHAVE site inside a phase (a dequant, a gemm, a gather): one per-task record under that name, so
    it gets its own track. A macro, because a bare extern call in a plain helper is dropped by the builder.

    Recorded only in a build with `TL_NPU_PHASE_PROF=1` (the runners' `--phases`): the records cost ~5%, so
    they are opt-in, as VPUX gates on PERF_COUNT and npu_dsl on -DFRT_PROFILE."""
    T.call_extern("handle", "ctx.task_begin", name)


@T.macro
def perf_end():
    T.call_extern("handle", "ctx.task_end")


@T.macro
def perf_phase_begin(name: str):
    """A PHASE boundary: the unit between two grid barriers."""
    T.call_extern("handle", "ctx.phase_begin", name)


@T.macro
def perf_phase_end(name: str):
    T.call_extern("handle", "ctx.phase_end")


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
    projection is the packed values `[N, K/2]` and a SEPARATE group-major fp16 scale tensor `[K/G, N]`, which the
    DPU path reads in place. The chunk size is a scheduling choice and stays out of the files: re-tiling must not
    mean re-packing the 17.91 GB of weights a prompt already moves, and prefill reads the same two tensors.
    The router's E + 1 fp16 rows are padded with zeros to RR = a multiple of tiles x shaves x BNR. Phases:
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
    moe_phases = moe_block(H=H, I=I, E=E, TOPK=TOPK, G=G, tiles=tiles, shaves=shaves, num_stages=num_stages, BNR=BNR,
                           BNG=BNG, BND=BND, RG=RG, RD=RD, pipelined=pipelined, probe_skip_dpu=probe_skip_dpu)

    @T.prim_func
    def moe(X: T.Tensor((1, H), T.float16), Wr: T.Tensor((RR, H), T.float16),
            WguQ: T.Tensor(((E + 1) * 2 * I, H // 2), T.uint8), WguS: T.Tensor((HG, (E + 1) * 2 * I), T.float16),
            WdQ: T.Tensor(((E + 1) * H, I // 2), T.uint8), WdS: T.Tensor((IG, (E + 1) * H), T.float16),
            Res: T.Tensor((1, H), T.float16), Logits: T.Tensor((MR, RR), T.float16),
            Part: T.Tensor((tiles, H), T.float32), Out: T.Tensor((1, H), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            x_cmx = T.alloc_shared((1, H), T.float16)
            p_cmx = T.alloc_shared((tiles, H), T.float32)
            res_cmx = T.alloc_shared((1, H), T.float16)
            out_cmx = T.alloc_shared((1, H), T.float16)
            T.copy(X[0, 0], x_cmx)
            moe_phases(x_cmx, Wr, WguQ, WguS, WdQ, WdS, Logits, Part, tile, shave)
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
    """The MoE layer's phases 1-3 as a `T.macro` (make_moe_decode's docstring): `moe_phases(x_cmx, Wr, WguQ, WguS,
    WdQ, WdS, Logits, Part, tile, shave)` takes the normed token in tile-shared CMX (`x_cmx` fp16 [1, H]) and leaves
    every tile's fp32 partial of the MoE output in `Part[tile]` behind a grid barrier; the caller combines (the
    standalone kernel on tile 0 with the residual, a layer kernel into its fp32 residual stream on every tile)."""
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
    HG, IG = H // G, I // G

    @T.macro
    def moe_phases(x_cmx, Wr, WguQ, WguS, WdQ, WdS, Logits, Part, tile, shave):
        x16 = T.alloc_shared((MR, H), T.float16)
        wr_cmx = T.alloc_shared((RRT, H), T.float16)
        yr_cmx = T.alloc_local((MR, RRT), T.float16)
        lg_cmx = T.alloc_shared((1, RR), T.float16)
        ids = T.alloc_shared((16,), T.int32)
        wts = T.alloc_shared((16,), T.float32)
        pick = T.alloc_local((16,), T.float32)
        best = T.alloc_local((2,), T.float32)
        mx = T.alloc_local((2 * FL,), T.float16)
        wq_cmx = T.alloc_shared((BNG, H // 2), T.uint8)
        y_cmx = T.alloc_local((1, BNG), T.float16)
        gu_cmx = T.alloc_shared((1, 2 * I), T.float16)
        h_cmx = T.alloc_shared((1, I), T.float16)
        wqd_cmx = T.alloc_shared((BND, I // 2), T.uint8)
        # the tile's IPT experts' scales in ONE DMA each, before the item loop: per chunk they would be CG + CD
        # extra jobs of 4 KiB (the engine's fixed cost per job, not the bytes), and per ITEM they would need a
        # barrier inside the loop, which drains the weight ring the item loop exists to keep full
        wsg_all = T.alloc_shared((HG, IPT * 2 * I), T.float16)
        wsd_all = T.alloc_shared((IG, IPT * H), T.float16)
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
            T.copy(WguS[0, ids[tile * IPT + j] * 2 * I], wsg_all[0:HG, j * 2 * I:(j + 1) * 2 * I])
            T.copy(WdS[0, ids[tile * IPT + j] * H], wsd_all[0:IG, j * H:(j + 1) * H])
        T.sync_threads()
        for j in T.serial(IPT):
            perf_begin("gupr")
            for n in (T.Pipelined(CG, num_stages=NSG) if pipelined else T.serial(CG)):
                T.copy(WguQ[ids[tile * IPT + j] * 2 * I + n * BNG, 0], wq_cmx)
                if probe_skip_dpu:
                    y_cmx[0, 0] = T.cast(wq_cmx[0, 0], T.float16)   # keep the slot read (the ring needs a reader)
                else:
                    for sub, dfr in GU_SUBS:   # Python-level: `defer` is a compile-time flag
                        # all but the chunk's last workload defer their reduction into the next workload's DPU time
                        T.gemm_i4(x_cmx, wq_cmx, wsg_all, y_cmx, group=G, row0=sub * RG, rows=RG, scales_major="group",
                                  scales_at=(0, j * 2 * I + n * BNG), defer=dfr)
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
                T.copy(WdQ[ids[tile * IPT + j] * H + n * BND, 0], wqd_cmx)
                if probe_skip_dpu:
                    yd_cmx[0, 0] = T.cast(wqd_cmx[0, 0], T.float16)
                else:
                    for sub in T.serial(BND // RD):
                        T.gemm_i4(h_cmx, wqd_cmx, wsd_all, yd_cmx, group=G, row0=sub * RD, rows=RD,
                                  scales_major="group", scales_at=(0, j * H + n * BND))
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


def make_moe_route_prefill(N: int = 128, H: int = 2048, E: int = 256, TOPK: int = 8, BM: int = 64, BNR: int = 16,
                           tiles: int = 3, shaves: int = 2, as_block: bool = False):
    """P2a: MoE routing and **grouping by expert** for `N` prompt tokens.

    Decode reads every chosen expert's weights once per token. Prefill must read them once per expert, so the
    (token, choice) pairs are sorted into expert order first and the FFN then walks one contiguous group per
    expert. Phases, each closed by a grid barrier:

      A. router GEMM for all N tokens, the padded router rows split over the tiles (`RRT` each), token blocks of
         `BM` rows so one workload stays inside the DPU's output-point limit -> `Logits[N, RR]`;
      B. the tokens dealt over the workers: top-k of `E` on fp16 lanes (lowest id on a tie, as decode), the
         renormalised softmax over the chosen k, and the shared expert's sigmoid gate -> `Ids`, `Wts`;
      C. worker 0 builds the grouping: `Cnt[e]` tokens per expert (the shared expert E takes every token),
         `Off` its exclusive prefix sum, and `Slot[s] = t * (TOPK + 1) + k` -- the (token, choice) pair that
         occupies grouped slot `s`, so slot `s` of expert `e` is `Off[e] .. Off[e + 1]`.

    `Wts[t, TOPK]` is the shared expert's gate. Oracle: `moe_route_prefill_reference`."""
    FL, HL = 16, 32
    per_round = tiles * shaves * BNR
    RR = ((E + 1 + per_round - 1) // per_round) * per_round
    RRT = RR // tiles
    workers = tiles * shaves
    SLOTS = N * (TOPK + 1)
    assert N % BM == 0 and E % HL == 0 and H % FL == 0 and RR % tiles == 0 and RRT % 16 == 0
    assert BM <= 8192 and RRT <= 8192, f"router workload {BM} x {RRT} has a dimension above the DPU's 8192"
    assert TOPK + 1 <= 16, "the per-token choice buffers are one 16-wide vector"

    @T.macro
    def route_phase(X, Wr, Logits, Ids, Wts, Cnt, Off, Slot, tile, shave):
        """The phase as a macro so a layer kernel can call it; `make_moe_route_prefill` is the standalone wrapper."""
        worker = tile * shaves + shave
        x_cmx = T.alloc_shared((BM, H), T.float16)
        wr_cmx = T.alloc_shared((RRT, H), T.float16)
        yr_cmx = T.alloc_local((BM, RRT), T.float16)
        lg_cmx = T.alloc_local((1, RR), T.float16)
        idl = T.alloc_local((1, 16), T.int32)
        wl = T.alloc_local((1, 16), T.float32)
        pick = T.alloc_local((16,), T.float32)
        best = T.alloc_local((4,), T.float32)
        mx = T.alloc_local((HL,), T.float16)
        ids_all = T.alloc_local((N, TOPK), T.int32)
        cnt_cmx = T.alloc_local((E + 1,), T.int32)
        off_cmx = T.alloc_local((E + 2,), T.int32)
        cur_cmx = T.alloc_local((E + 1,), T.int32)
        slot_cmx = T.alloc_local((SLOTS,), T.int32)
        # --- A. router GEMM over the token blocks ----------------------------------------------------
        T.copy(Wr[tile * RRT, 0], wr_cmx)
        for b in T.serial(N // BM):
            T.copy(X[b * BM, 0], x_cmx)
            if shave == 0:
                T.gemm(x_cmx, wr_cmx, yr_cmx, transpose_B=True, clear_accum=True)
                T.copy(yr_cmx, Logits[b * BM, tile * RRT])
            T.sync_threads()
        T.sync_grid()
        # --- B. top-k per token, the tokens dealt over the workers -----------------------------------
        for it in T.serial(T.ceildiv(N, workers)):
            t = it * workers + worker
            if t < N:
                T.copy(Logits[t, 0], lg_cmx)
                for k in T.serial(TOPK):
                    for v in T.vectorized(HL):
                        mx[v] = lg_cmx[0, v]
                    for e0 in T.serial(1, E // HL):
                        for v in T.vectorized(HL):
                            mx[v] = T.max(mx[v], lg_cmx[0, e0 * HL + v])
                    best[0] = T.cast(T.call_extern(T.float16, "tl_npu_vmax", mx[0:HL]), T.float32)
                    # argmax on lanes: E - index where the logit equals the max -> the lowest index wins
                    for v in T.vectorized(HL):
                        mx[v] = T.if_then_else(lg_cmx[0, v] == T.cast(best[0], T.float16), T.cast(E - v, T.float16), T.float16(0))
                    for e0 in T.serial(1, E // HL):
                        for v in T.vectorized(HL):
                            mx[v] = T.max(mx[v], T.if_then_else(lg_cmx[0, e0 * HL + v] == T.cast(best[0], T.float16),
                                                                T.cast(E - (e0 * HL + v), T.float16), T.float16(0)))
                    best[1] = T.float32(E) - T.cast(T.call_extern(T.float16, "tl_npu_vmax", mx[0:HL]), T.float32)
                    idl[0, k] = T.cast(best[1], T.int32)
                    pick[k] = best[0]
                    lg_cmx[0, idl[0, k]] = T.float16(-65504)
                best[1] = T.float32(0)
                for k in T.serial(TOPK):
                    wl[0, k] = T.exp(pick[k] - pick[0])   # pick[0] is the row max
                    best[1] = best[1] + wl[0, k]
                for k in T.serial(TOPK):
                    wl[0, k] = wl[0, k] / best[1]
                wl[0, TOPK] = T.float32(1) / (T.float32(1) + T.exp(T.float32(0) - T.cast(lg_cmx[0, E], T.float32)))
                T.copy(idl[0:1, 0:TOPK], Ids[t, 0])
                T.copy(wl, Wts[t, 0])
        T.sync_grid()
        # --- C. grouping: histogram, exclusive prefix sum, slot table (worker 0) ----------------------
        if worker == 0:
            T.copy(Ids[0, 0], ids_all)
            for e in T.serial(E + 1):
                cnt_cmx[e] = 0
            for t in T.serial(N):
                for k in T.serial(TOPK):
                    cnt_cmx[ids_all[t, k]] = cnt_cmx[ids_all[t, k]] + 1
            cnt_cmx[E] = N                      # every token also goes to the shared expert
            off_cmx[0] = 0
            for e in T.serial(E + 1):
                off_cmx[e + 1] = off_cmx[e] + cnt_cmx[e]
                cur_cmx[e] = off_cmx[e]
            for t in T.serial(N):
                for k in T.serial(TOPK):
                    slot_cmx[cur_cmx[ids_all[t, k]]] = t * (TOPK + 1) + k
                    cur_cmx[ids_all[t, k]] = cur_cmx[ids_all[t, k]] + 1
                slot_cmx[cur_cmx[E]] = t * (TOPK + 1) + TOPK
                cur_cmx[E] = cur_cmx[E] + 1
            T.copy(cnt_cmx, Cnt[0])
            T.copy(off_cmx, Off[0])
            T.copy(slot_cmx, Slot[0])

    if as_block:
        return route_phase, dict(RR=RR, RRT=RRT, SLOTS=SLOTS)

    @T.prim_func
    def moe_route(X: T.Tensor((N, H), T.float16), Wr: T.Tensor((RR, H), T.float16),
                  Logits: T.Tensor((N, RR), T.float16), Ids: T.Tensor((N, TOPK), T.int32),
                  Wts: T.Tensor((N, 16), T.float32), Cnt: T.Tensor((E + 1,), T.int32),
                  Off: T.Tensor((E + 2,), T.int32), Slot: T.Tensor((SLOTS,), T.int32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            route_phase(X, Wr, Logits, Ids, Wts, Cnt, Off, Slot, tile, T.get_thread_binding())

    return moe_route


def moe_route_prefill_reference(x, wr, E: int, TOPK: int, logits=None):
    """numpy oracle of make_moe_route_prefill (fp32): the top-k ids (lowest id on a tie), the renormalised
    weights with the shared expert's gate last, and the expert grouping (counts, exclusive offsets, slots).
    `logits`: rank THESE instead of recomputing them. The device stores its logits in fp16, and with hundreds of
    experts a near-tie ranks differently in fp16 than in fp32, so a device golden passes its own logits back in
    and the grouping is then compared exactly -- the GEMM is checked separately, against a tolerance."""
    import numpy as np
    N = x.shape[0]
    logits = (x.astype(np.float32) @ wr.astype(np.float32).T) if logits is None else logits.astype(np.float32)
    ids = np.zeros((N, TOPK), np.int32)
    wts = np.zeros((N, 16), np.float32)
    for t in range(N):
        row = logits[t, :E].copy()
        for k in range(TOPK):
            m = row.max()
            i = int(np.flatnonzero(row == m)[0])                     # the lowest index among equals
            ids[t, k] = i
            row[i] = -65504.0
        p = np.exp(logits[t, ids[t]] - logits[t, ids[t, 0]])
        wts[t, :TOPK] = p / p.sum()
        wts[t, TOPK] = 1.0 / (1.0 + np.exp(-logits[t, E]))
    cnt = np.zeros(E + 1, np.int32)
    for t in range(N):
        for k in range(TOPK):
            cnt[ids[t, k]] += 1
    cnt[E] = N
    off = np.zeros(E + 2, np.int32)
    off[1:] = np.cumsum(cnt)
    cur = off[:E + 1].copy()
    slot = np.zeros(N * (TOPK + 1), np.int32)
    for t in range(N):
        for k in range(TOPK):
            e = int(ids[t, k])
            slot[cur[e]] = t * (TOPK + 1) + k
            cur[e] += 1
        slot[cur[E]] = t * (TOPK + 1) + TOPK
        cur[E] += 1
    return dict(logits=logits, ids=ids, wts=wts, cnt=cnt, off=off, slot=slot)


def make_moe_expert_prefill(N: int = 32, H: int = 256, I: int = 128, E: int = 16, TOPK: int = 4, G: int = 128,
                            MB: int = 16, BNG: int = 64, BND: int = 64, tiles: int = 3, shaves: int = 2,
                            as_block: bool = False, probe_skip: str | None = None):
    """P2b: the grouped expert FFN for `N` prompt tokens, over P2a's grouping.

    Each worker owns whole experts, so **an expert's weights are read once per launch** instead of once per
    (token, choice) as decode does -- the point of prefill. An expert's slots `Off[e] .. Off[e + 1]` are walked in
    chunks of `MB` tokens: the chunk's rows are gathered out of `X` by `Slot`, the gate/up and down projections run
    as **SHAVE dequant into fp16 then an fp16 DPU gemm** (`make_w4a16_flow`'s path, not decode's merged-FC: the
    merged-FC scratch grows with M and is sized for M = 1, while the dequant's cost does not grow with M at all and
    so amortises over the chunk), SwiGLU between them.

    A token's TOPK + 1 choices land on different experts and therefore on different workers, so the weighted
    scatter-add would race. Each worker accumulates into its own `Part[worker]` and a combine phase behind a grid
    barrier sums them, the tokens dealt over the workers. That costs `workers x N x H` fp32 of DDR; a slot-major
    output plus the inverse of `Slot` would cost `N x (TOPK + 1) x H` fp16 instead and is the next form.

    Oracle: `moe_expert_prefill_reference`.

    The weights arrive as `make_w4a16_flow`'s pair -- packed values and a SEPARATE scale tensor, carried
    **group-major** `[K/G, N]`. That is the layout decode's DPU int4 path reads in place (`dpu_matmul_i4`'s
    `sStride > 0`), so prefill and decode share one weight upload rather than one each; at 40 layers and 256
    experts the second copy is the difference between fitting the box and not. The SHAVE reads it through
    `tl_npu_dequant_i4_gm`, which keeps the VPUX assembly by transposing a chunk's scales into a stack buffer
    first, so the layout costs the scales only -- 1/64 of the bytes at G = 128.

    `probe_skip='dequant'`: MEASUREMENT ARM ONLY -- the weights are still read from DDR and the gemms still run,
    but the int4 rows are not expanded (the output is wrong by design). The difference is the dequant's share."""
    FL, HL = 16, 32
    skip_dequant = probe_skip == "dequant"
    workers = tiles * shaves
    SLOTS = N * (TOPK + 1)
    CG, CD = (2 * I) // BNG, H // BND
    KGH, KGI = H // G, I // G
    assert H % G == 0 and I % G == 0 and (2 * I) % BNG == 0 and H % BND == 0
    assert H % HL == 0 and I % HL == 0 and MB * BNG <= 8192 and MB * BND <= 8192
    assert I % (shaves * HL) == 0 and N % shaves == 0
    # down chunks per DMA: at BND = 64 a job is 16 KiB and 70% of it is the 1.55 us per-job fixed cost, and the
    # PACKED buffer is 8x smaller than wd_cmx, so DMA granularity is cheap to decouple from compute granularity
    DB = next(d for d in (4, 2, 1) if CD % (d * shaves) == 0)
    EPT = -(-(E + 1) // tiles)            # experts per TILE: the tile's SHAVEs cooperate on one expert

    @T.macro
    def expert_phase(X, WguQ, WguS, WdQ, WdS, Wts, Off, Slot, Part, Out, tile, shave):
        """The phase as a macro so a layer kernel can call it; `make_moe_expert_prefill` is the standalone wrapper.

        A TILE owns whole experts and its two SHAVEs split that expert's weight chunks, rather than each worker
        owning its own experts. The flow's ping-pong needs the tile's SHAVEs alternating on chunks of ONE loop
        (docs/examples/pingpong_multitile/README.md 1-3), which per-worker ownership cannot offer; it also halves
        the partials, so the zero fill, the accumulation and the combine all read and write three rows per token
        instead of six."""
        worker = tile * shaves + shave
        off_cmx = T.alloc_local((E + 2,), T.int32)
        slot_cmx = T.alloc_local((SLOTS,), T.int32)
        wts_cmx = T.alloc_local((N, 16), T.float32)
        x_cmx = T.alloc_shared((MB, H), T.float16)          # the chunk's tokens: both SHAVEs read them
        wq_cmx = T.alloc_local((BNG, H // 2), T.uint8)      # a SHAVE's own chunk of weights
        ws_cmx = T.alloc_local((KGH, BNG), T.float16)
        wg_cmx = T.alloc_local((BNG, H), T.float16)
        y_cmx = T.alloc_local((MB, BNG), T.float16)
        gu_cmx = T.alloc_shared((MB, 2 * I), T.float16)     # the SHAVEs write disjoint column ranges
        h_cmx = T.alloc_shared((MB, I), T.float16)
        wqd_cmx = T.alloc_local((DB * BND, I // 2), T.uint8)
        # the expert's down scales in one DMA instead of one per chunk: CD = 32 jobs for 16 KiB, where the same
        # hoist on the gate/up side would cost 32 KiB to save 16, and the layer kernel has no room for both.
        # Group-major, so a chunk is a COLUMN range of this and the dequant reads it in place at pitch H.
        wsd_all = T.alloc_shared((KGI, H), T.float16)
        wd_cmx = T.alloc_local((BND, I), T.float16)
        yd_cmx = T.alloc_local((MB, BND), T.float16)
        yo = T.alloc_shared((MB, H), T.float16)     # the chunk's whole down output, so Part is touched once per token
        pw = T.alloc_local((1, H), T.float32)
        pa = T.alloc_local((1, H), T.float32)
        zr = T.alloc_local((1, H), T.float32)
        seen = T.alloc_shared((N,), T.int32)       # per token: has an expert of this TILE written its partial yet?
        T.copy(Off[0], off_cmx)
        T.copy(Slot[0], slot_cmx)
        T.copy(Wts[0, 0], wts_cmx)
        for c0 in T.serial(H // FL):
            for v in T.vectorized(FL):
                zr[0, c0 * FL + v] = T.float32(0)
        for t in T.serial(N):
            if t % shaves == shave:
                seen[t] = 0
        T.sync_threads()
        for ei in T.serial(EPT):
            e = ei * tiles + tile
            if e < E + 1:
                # a DDR -> shared fill is split over the tile's SHAVEs and gets a tile sync, so it must not sit
                # under a SHAVE-dependent branch: guarding one would leave the other waiting at that sync
                T.copy(WdS[0, e * H], wsd_all)
                for c in T.serial(T.ceildiv(off_cmx[e + 1] - off_cmx[e], MB)):
                    # gather the chunk's token rows; a short tail keeps the last row, whose result is dropped
                    perf_begin("gath")
                    for i in T.serial(MB):
                        T.copy(X[slot_cmx[T.min(off_cmx[e] + c * MB + i, off_cmx[e + 1] - 1)] // (TOPK + 1), 0],
                               x_cmx[i:i + 1, 0:H])
                    perf_end()
                    T.sync_threads()
                    for n in T.serial(CG):
                        if n % shaves == shave:
                            T.copy(WguQ[e * 2 * I + n * BNG, 0], wq_cmx)
                            T.copy(WguS[0, e * 2 * I + n * BNG], ws_cmx)
                            perf_begin("gdeq")
                            if not skip_dequant:
                                T.call_extern("handle", "tl_npu_dequant_i4_gm", T.address_of(wg_cmx[0, 0]),
                                              T.address_of(wq_cmx[0, 0]), T.address_of(ws_cmx[0, 0]), BNG, H, G, BNG)
                            perf_end()
                            perf_begin("ggem")
                            T.gemm(x_cmx, wg_cmx, y_cmx, transpose_B=True, clear_accum=True)
                            perf_end()
                            perf_begin("gcpy")
                            for i, c0 in T.grid(MB, BNG // HL):
                                for v in T.vectorized(HL):
                                    gu_cmx[i, n * BNG + c0 * HL + v] = y_cmx[i, c0 * HL + v]
                            perf_end()
                    T.sync_threads()
                    perf_begin("swig")
                    for i, c0 in T.grid(MB, I // shaves // HL):
                        for v in T.vectorized(HL):
                            h_cmx[i, shave * (I // shaves) + c0 * HL + v] = (
                                gu_cmx[i, shave * (I // shaves) + c0 * HL + v]
                                * T.sigmoid(gu_cmx[i, shave * (I // shaves) + c0 * HL + v])
                                * gu_cmx[i, I + shave * (I // shaves) + c0 * HL + v])
                    perf_end()
                    T.sync_threads()
                    for nb in T.serial(CD // (DB * shaves)):
                        T.copy(WdQ[e * H + ((nb * shaves + shave) * DB) * BND, 0], wqd_cmx)
                        for j in T.serial(DB):
                            perf_begin("ddeq")
                            if not skip_dequant:
                                T.call_extern("handle", "tl_npu_dequant_i4_gm", T.address_of(wd_cmx[0, 0]),
                                              T.address_of(wqd_cmx[j * BND, 0]),
                                              T.address_of(wsd_all[0, ((nb * shaves + shave) * DB + j) * BND]),
                                              BND, I, G, H)
                            perf_end()
                            perf_begin("dgem")
                            T.gemm(h_cmx, wd_cmx, yd_cmx, transpose_B=True, clear_accum=True)
                            perf_end()
                            perf_begin("dcpy")
                            for i, c0 in T.grid(MB, BND // HL):
                                for v in T.vectorized(HL):
                                    yo[i, ((nb * shaves + shave) * DB + j) * BND + c0 * HL + v] = yd_cmx[i, c0 * HL + v]
                            perf_end()
                    T.sync_threads()
                    # one read-modify-write of the whole row per live token: the chunk-wise form cost CD of
                    # them, and at BND = 64 those are 256-byte DDR jobs whose count dominated the phase
                    perf_begin("accm")
                    for i in T.serial(MB):
                        if i % shaves == shave and off_cmx[e] + c * MB + i < off_cmx[e + 1]:
                            # the first touch of a row stores, so the partial never has to be zeroed up front
                            if seen[slot_cmx[off_cmx[e] + c * MB + i] // (TOPK + 1)] == 0:
                                seen[slot_cmx[off_cmx[e] + c * MB + i] // (TOPK + 1)] = 1
                                for c0 in T.serial(H // FL):
                                    for v in T.vectorized(FL):
                                        pw[0, c0 * FL + v] = T.float32(0)
                            else:
                                T.copy(Part[tile, slot_cmx[off_cmx[e] + c * MB + i] // (TOPK + 1), 0], pw)
                            for c0 in T.serial(H // FL):
                                for v in T.vectorized(FL):
                                    pw[0, c0 * FL + v] = (pw[0, c0 * FL + v]
                                                          + wts_cmx[slot_cmx[off_cmx[e] + c * MB + i] // (TOPK + 1),
                                                                    slot_cmx[off_cmx[e] + c * MB + i] % (TOPK + 1)]
                                                          * T.cast(yo[i, c0 * FL + v], T.float32))
                            T.copy(pw, Part[tile, slot_cmx[off_cmx[e] + c * MB + i] // (TOPK + 1), 0])
                    perf_end()
                    T.sync_threads()
        T.sync_threads()
        for t in T.serial(N):                      # only the tokens no expert of this tile saw still need zeros
            if t % shaves == shave and seen[t] == 0:
                T.copy(zr, Part[tile, t, 0])
        T.sync_grid()
        # --- combine: the tokens dealt over the workers, the tiles' partials summed -------------------
        for it in T.serial(T.ceildiv(N, workers)):
            t = it * workers + worker
            if t < N:
                for c0 in T.serial(H // FL):
                    for v in T.vectorized(FL):
                        pw[0, c0 * FL + v] = T.float32(0)
                for w in T.serial(tiles):
                    T.copy(Part[w, t, 0], pa)     # a DDR row is a DMA into CMX, never a vector load
                    for c0 in T.serial(H // FL):
                        for v in T.vectorized(FL):
                            pw[0, c0 * FL + v] = pw[0, c0 * FL + v] + pa[0, c0 * FL + v]
                T.copy(pw, Out[t, 0])

    if as_block:
        return expert_phase, dict(SLOTS=SLOTS, CG=CG, CD=CD, KGH=KGH, KGI=KGI, workers=workers, parts=tiles)

    @T.prim_func
    def moe_expert(X: T.Tensor((N, H), T.float16),
                   WguQ: T.Tensor(((E + 1) * 2 * I, H // 2), T.uint8), WguS: T.Tensor((KGH, (E + 1) * 2 * I), T.float16),
                   WdQ: T.Tensor(((E + 1) * H, I // 2), T.uint8), WdS: T.Tensor((KGI, (E + 1) * H), T.float16),
                   Wts: T.Tensor((N, 16), T.float32), Off: T.Tensor((E + 2,), T.int32),
                   Slot: T.Tensor((SLOTS,), T.int32), Part: T.Tensor((tiles, N, H), T.float32),
                   Out: T.Tensor((N, H), T.float32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            expert_phase(X, WguQ, WguS, WdQ, WdS, Wts, Off, Slot, Part, Out, tile, T.get_thread_binding())

    return moe_expert


def moe_expert_prefill_reference(x, gu_int, gu_scales, d_int, d_scales, wts, off, slot, TOPK: int, G: int):
    """numpy oracle of make_moe_expert_prefill (fp32): every (token, choice) pair's expert FFN, weighted and summed
    per token. Walks the SAME grouping the kernel walks, so a wrong slot table shows up as a wrong result."""
    import numpy as np
    N, H = x.shape
    E = gu_int.shape[0] - 1
    out = np.zeros((N, H), np.float32)
    for e in range(E + 1):
        gu = gu_int[e].astype(np.float32) * np.repeat(gu_scales[e].astype(np.float32), G, axis=1)
        dw = d_int[e].astype(np.float32) * np.repeat(d_scales[e].astype(np.float32), G, axis=1)
        for s in range(int(off[e]), int(off[e + 1])):
            t, k = int(slot[s]) // (TOPK + 1), int(slot[s]) % (TOPK + 1)
            y = gu @ x[t].astype(np.float32)
            g, u = y[: y.size // 2], y[y.size // 2:]
            h = g / (1 + np.exp(-g)) * u
            out[t] += float(wts[t, k]) * (dw @ h)
    return out


def pack_moe(router: "np.ndarray", gu_int: "np.ndarray", gu_scales: "np.ndarray", d_int: "np.ndarray",
             d_scales: "np.ndarray", tiles: int = 3, shaves: int = 2, BNR: int = 16, BNG: int = 128, BND: int = 256,
             G: int = 128):
    """The kernel's weight layout from per-expert arrays (the shared expert LAST, as expert E): router fp16
    [E + 1, H] (row E the shared gate) padded with zero rows to RR; gu_int int8 [E + 1, 2I, H] in [-8, 7] and its
    scales [E + 1, 2I, H / G]; d_int [E + 1, H, I] and scales [E + 1, H, I / G]. Returns the kernel's arrays: the
    router, and each projection as packed values plus a separate group-major scale tensor (`pack_int4` /
    `gm_scales`). BNG / BND no longer reach the layout -- the chunk size is a scheduling choice."""
    import numpy as np
    R, H = router.shape
    per_round = tiles * shaves * BNR
    RR = ((R + per_round - 1) // per_round) * per_round
    wr = np.zeros((RR, H), np.float16)
    wr[:R] = router
    gu_q = np.concatenate([pack_int4(gu_int[e]) for e in range(gu_int.shape[0])], axis=0)
    gu_s = gm_scales(np.concatenate([gu_scales[e] for e in range(gu_int.shape[0])], axis=0).astype(np.float16))
    d_q = np.concatenate([pack_int4(d_int[e]) for e in range(d_int.shape[0])], axis=0)
    d_s = gm_scales(np.concatenate([d_scales[e] for e in range(d_int.shape[0])], axis=0).astype(np.float16))
    return dict(Wr=wr, WguQ=gu_q, WguS=gu_s, WdQ=d_q, WdS=d_s)


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
    CONV, NIN_pad, NOUT_pad, HG1, KG2, VC = sz["CONV"], sz["NIN_pad"], sz["NOUT_pad"], sz["HG1"], sz["KG2"], sz["VC"]
    WBA_pad, YLEN, BA_LEN, CVLEN, OPAD = sz["WBA_pad"], sz["YLEN"], sz["BA_LEN"], sz["CVLEN"], sz["OPAD"]

    @T.prim_func
    def gdn(X: T.Tensor((1, H), T.float16), Res: T.Tensor((1, H), T.float16),
            WinQ: T.Tensor((NIN_pad, H // 2), T.uint8), WinS: T.Tensor((HG1, NIN_pad), T.float16),
            Wba: T.Tensor((WBA_pad, H), T.float16),
            ConvW: T.Tensor((4, CONV), T.float16),
            ALog: T.Tensor((1, VH), T.float32), DtBias: T.Tensor((1, VH), T.float32), GNorm: T.Tensor((1, DV), T.float16),
            ConvState: T.Tensor((3, CONV), T.float16), Ssm: T.Tensor((VH, DK, DV), T.float32),
            WoutQ: T.Tensor((NOUT_pad, VC // 2), T.uint8), WoutS: T.Tensor((KG2, NOUT_pad), T.float16),
            Y: T.Tensor((1, YLEN), T.float16), BA: T.Tensor((1, BA_LEN), T.float16), CV: T.Tensor((1, CVLEN), T.float16),
            O: T.Tensor((1, OPAD), T.float16),
            Out: T.Tensor((1, NOUT_pad), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            x_cmx = T.alloc_shared((1, H), T.float16)
            T.copy(X[0, 0], x_cmx)
            gdn_mixer(x_cmx, WinQ, WinS, Wba, ConvW, ALog, DtBias, GNorm, ConvState, Ssm, WoutQ, WoutS, Y, BA, CV, O,
                      Res, Out, tile, shave)

    return gdn


def gdn_block(H: int, KH: int, VH: int, DK: int, DV: int, G: int, tiles: int, shaves: int, num_stages: int = 4,
              num_stages2: int = 2, BN1: int = 128, R1: int = 64, BN2: int = 64, R2: int = 32, CB: int = 512, eps: float = 1e-6,
              pipelined: bool = True, probe_skip: str | None = None, res_add: bool = True):
    """The Gated DeltaNet mixer as a `T.macro` (make_gdn_decode's docstring): `gdn_mixer(x_cmx, WinQ, WinS, Wba, ConvW,
    ALog, DtBias, GNorm, ConvState, Ssm, WoutQ, WoutS, Y, BA, CV, O, Res, Out, tile, shave)` takes the normed token in
    tile-shared CMX
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
    sizes = dict(CONV=CONV, NIN_pad=NIN_pad, NOUT_pad=NOUT_pad, HG1=HG1, KG2=KG2, VC=VC, WBA_pad=WBA_pad, YLEN=YLEN,
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
    def gdn_mixer(x_cmx, WinQ, WinS, Wba, ConvW, ALog, DtBias, GNorm, ConvState, Ssm, WoutQ, WoutS, Y, BA, CV, O, Res,
                  Out, tile, shave):
            worker = tile * shaves + shave
            w1_cmx = T.alloc_shared((BN1, H // 2), T.uint8)
            s1_cmx = T.alloc_shared((HG1, BN1), T.float16)
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
            w2_cmx = T.alloc_shared((BN2, VC // 2), T.uint8)
            s2_cmx = T.alloc_shared((KG2, BN2), T.float16)
            y2_cmx = T.alloc_local((1, BN2), T.float16)
            # --- A. in_proj GEMV: rows [q | k | v | z | pad] split over the tiles; b | a on the SHAVE ------------
            for n in (T.Pipelined(C1, num_stages=num_stages) if pipelined else T.serial(C1)):
                T.copy(WinQ[(tile * C1 + n) * BN1, 0], w1_cmx)
                T.copy(WinS[0, (tile * C1 + n) * BN1], s1_cmx)
                if skip_gemv:
                    y1_cmx[0, 0] = T.cast(w1_cmx[0, 0], T.float16)
                else:
                    for sub in T.serial(BN1 // R1):
                        T.gemm_i4(x_cmx, w1_cmx, s1_cmx, y1_cmx, group=G, row0=sub * R1, rows=R1, scales_major="group")
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
                T.copy(WoutQ[(tile * C2 + n) * BN2, 0], w2_cmx)
                T.copy(WoutS[0, (tile * C2 + n) * BN2], s2_cmx)
                for sub in T.serial(BN2 // R2):
                    T.gemm_i4(o_cmx, w2_cmx, s2_cmx, y2_cmx, group=G, row0=sub * R2, rows=R2, scales_major="group")
                if res_add:
                    for c in T.serial(BN2):
                        y2_cmx[0, c] = T.cast(T.cast(y2_cmx[0, c], T.float32)
                                              + T.cast(res_cmx[0, (tile * C2 + n) * BN2 + c], T.float32), T.float16)
                T.copy(y2_cmx, Out[0, (tile * C2 + n) * BN2])

    return gdn_mixer, sizes


def gdn_sizes(H: int, KH: int, VH: int, DK: int, tiles: int = 3, shaves: int = 2, BN1: int = 128, BN2: int = 64, **_):
    """make_gdn_decode's scratch and output sizes: dict(YLEN, BA_LEN, CVLEN, OPAD, NOUT_pad, WBA_pad, NIN_pad)."""
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
                OPAD=OPAD, NOUT_pad=NOUT_pad, WBA_pad=WBA_pad, NIN_pad=NIN_pad)


def pack_gdn(win_int, win_scales, wout_int, wout_scales, wba=None, tiles: int = 3, shaves: int = 2, BN1: int = 128, BN2: int = 64):
    """make_gdn_decode's weights from the in_proj rows (int4 values [NIN, H] in [-8, 7] with scales [NIN, H/G], rows
    [q | k | v | z]) and the out_proj ([H, VC] with [H, VC/G]): zero-padded to the tile multiple, then the packed
    values and group-major scales as two tensors; `wba` [2 VH, H] (b rows then a rows, any float dtype) -> fp16 rows
    zero-padded to the worker multiple.

    BN1 / BN2 only decide the padding here, not the layout: the chunk size is a scheduling choice and re-tiling
    must not mean re-packing 17.91 GB."""
    import numpy as np

    def padded(w, s, BN):
        n_pad = -(-w.shape[0] // (BN * tiles)) * BN * tiles
        wp = np.zeros((n_pad, w.shape[1]), np.int8)
        sp = np.zeros((n_pad, s.shape[1]), np.float16)
        wp[: w.shape[0]] = w
        sp[: s.shape[0]] = s
        return pack_int4(wp), gm_scales(sp)

    wi, si = padded(win_int, win_scales, BN1)
    wo, so = padded(wout_int, wout_scales, BN2)
    out = dict(WinQ=wi, WinS=si, WoutQ=wo, WoutS=so)
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
    NQKV, NOUT_pad, HG1, KG2, KO, GQ, PW = sz["NQKV"], sz["NOUT_pad"], sz["HG1"], sz["KG2"], sz["KO"], sz["GQ"], sz["PW"]

    @T.prim_func
    def attn(X: T.Tensor((1, H), T.float16), Res: T.Tensor((1, H), T.float16),
             WqkvQ: T.Tensor((NQKV, H // 2), T.uint8), WqkvS: T.Tensor((HG1, NQKV), T.float16),
             QNormW: T.Tensor((1, D), T.float16),
             KNormW: T.Tensor((1, D), T.float16), CosSin: T.Tensor((2, ROT), T.float32), Pos: T.Tensor((1, 1), T.int32),
             Kc: T.Tensor((NKV, SMAX, D), T.float16), Vt: T.Tensor((NKV, D, SMAX), T.float16),
             WoQ: T.Tensor((NOUT_pad, KO // 2), T.uint8), WoS: T.Tensor((KG2, NOUT_pad), T.float16),
             Y: T.Tensor((1, NQKV), T.float16), QN: T.Tensor((NKV, GQ, D), T.float16),
             Part: T.Tensor((NKV, GQ, tiles, PW), T.float32), O: T.Tensor((1, KO), T.float16),
             Out: T.Tensor((1, NOUT_pad), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            x_cmx = T.alloc_shared((1, H), T.float16)
            T.copy(X[0, 0], x_cmx)
            attn_mixer(x_cmx, WqkvQ, WqkvS, QNormW, KNormW, CosSin, Pos, Kc, Vt, WoQ, WoS, Y, QN, Part, O, Res, Out,
                       tile, shave, Pos, 0)

    return attn


def attn_block(H: int, NH: int, NKV: int, D: int, ROT: int, SMAX: int, G: int, tiles: int, shaves: int, BN1: int = 128,
               R1: int = 64, BN2: int = 64, R2: int = 32, BKV: int = 128, num_stages: int = 4, num_stages2: int = 2,
               kv_stages: int = 2, eps: float = 1e-6, pipelined: bool = True, res_add: bool = True, stages: str = "ABCDE",
               gemv_pipelined: bool = True, kv4: bool = False):
    """The full-attention mixer as a `T.macro` (make_attn_decode's docstring): `attn_mixer(x_cmx, WqkvQ, WqkvS, QNormW,
    KNormW, CosSin, Pos, Kc, Vt, WoQ, WoS, Y, QN, Part, O, Res, Out, tile, shave, KvSlot, kvi)` takes the normed token in
    tile-shared CMX and writes
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
    workers = tiles * shaves
    NIT = NH + NKV
    CPT = SMAX // (tiles * BKV)                 # key chunks per tile per group
    assert CPT % shaves == 0, "the kv ring's runtime chunk count is a multiple of the SHAVEs (hoisted over the groups)"
    HL, FL = 32, 16
    PW = D + FL                                 # partial row: O then m, l
    scale = 1.0 / (D ** 0.5)
    HR = ROT // 2
    sizes = dict(NQKV=NQKV, NOUT_pad=NOUT_pad, HG1=HG1, KG2=KG2, KO=KO, GQ=GQ, PW=PW)

    @T.macro
    def attn_mixer(x_cmx, WqkvQ, WqkvS, QNormW, KNormW, CosSin, Pos, Kc, Vt, WoQ, WoS, Y, QN, Part, O, Res, Out, tile,
                   shave, KvSlot, kvi):
            worker = tile * shaves + shave
            w1_cmx = T.alloc_shared((BN1, H // 2), T.uint8)
            s1_cmx = T.alloc_shared((HG1, BN1), T.float16)
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
            w2_cmx = T.alloc_shared((BN2, KO // 2), T.uint8)
            s2_cmx = T.alloc_shared((KG2, BN2), T.float16)
            y2_cmx = T.alloc_local((1, BN2), T.float16)
            # --- A. qkv GEMV -----------------------------------------------------------------------------------------
            if "A" in stages:
              for n in (T.Pipelined(C1, num_stages=num_stages) if (pipelined and gemv_pipelined) else T.serial(C1)):
                T.copy(WqkvQ[(tile * C1 + n) * BN1, 0], w1_cmx)
                T.copy(WqkvS[0, (tile * C1 + n) * BN1], s1_cmx)
                for sub in T.serial(BN1 // R1):
                    T.gemm_i4(x_cmx, w1_cmx, s1_cmx, y1_cmx, group=G, row0=sub * R1, rows=R1, scales_major="group")
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
                T.copy(WoQ[(tile * C2 + n) * BN2, 0], w2_cmx)
                T.copy(WoS[0, (tile * C2 + n) * BN2], s2_cmx)
                for sub in T.serial(BN2 // R2):
                    T.gemm_i4(o_cmx, w2_cmx, s2_cmx, y2_cmx, group=G, row0=sub * R2, rows=R2, scales_major="group")
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
    """make_attn_decode's weights: the qkv rows ([per q-head q | gate] x NH, k, v; int4 values with scales) and the
    o_proj rows zero-padded to the tile multiple, each as packed values plus a group-major scale tensor."""
    import numpy as np
    n_pad = -(-wo_int.shape[0] // (BN2 * tiles)) * BN2 * tiles
    wo = np.zeros((n_pad, wo_int.shape[1]), np.int8)
    so = np.zeros((n_pad, wo_scales.shape[1]), np.float16)
    wo[: wo_int.shape[0]] = wo_int
    so[: wo_scales.shape[0]] = wo_scales
    return dict(WqkvQ=pack_int4(wqkv_int), WqkvS=gm_scales(wqkv_scales.astype(np.float16)),
                WoQ=pack_int4(wo), WoS=gm_scales(so))


def rope_cos_sin(pos: int, ROT: int, theta: float):
    """cos / sin rows [2, ROT] of one position (emb = cat(freqs, freqs), as the reference)."""
    import numpy as np
    inv = 1.0 / (theta ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT))
    f = pos * inv
    emb = np.concatenate([f, f])
    return np.stack([np.cos(emb), np.sin(emb)]).astype(np.float32)


GDN_Q_SCALE = 64.0
"""The chunked scan multiplies q by this before the DPU and divides it out in the gated norm (npu_dsl's
`kGdnQScale`). The DPU flushes fp16 subnormals, and `q k^T . D` has many entries below 2^-14 in a head that decays
fast; losing them is visible after the per-head RMSNorm of a small output. A power of two, so it is exact."""


def make_gdn_scan_prefill(N: int = 128, KH: int = 16, VH: int = 32, DK: int = 128, DV: int = 128, C: int = 64,
                          tiles: int = 3, shaves: int = 2, eps: float = 1e-6, as_block: bool = False):
    """P3: the Gated DeltaNet recurrence over `n` prompt tokens in **CHUNKED** form -- the prefill shape of the
    delta rule that `make_gdn_decode` runs one token at a time. Takes the conv phase's output (q / k l2-normalised
    fp32, v, and the two gates) and writes the mixer's gated-RMSNorm output.

    The recurrence `S <- S e^g + k (v - e^g S^T k) beta^T`, `o = S^T q` is sequential in t, so a prefill that walks
    it token by token leaves the DPU idle. The chunked form makes a block of C tokens into matrix products and keeps
    only the chunk-to-chunk carry sequential. With `gc = cumsum(g)` over the chunk, `gl = gc[C-1]`,
    `D[i][j] = e^(gc_i - gc_j)` for `j <= i` and 0 above, and S0 the state entering the chunk:

      T     = (I + strictly_lower(beta k k^T . D))^-1        (forward substitution, the one serial step left)
      U     = T (beta v);  W = T (e^gc beta k);  v_new = U - W S0
      P     = (q k^T) . D
      o     = (e^gc q) S0 + P v_new;   S = e^gl S0 + (e^(gl - gc) k)^T v_new

    Eleven DPU matmuls per chunk carry it, including the transposes (`out = a . b^T` with `a = I` transposes `b`
    exactly); the SHAVE only scales rows, builds D, solves the triangle and keeps the fp32 state. **Two of the
    eleven -- `dS` and the state transpose, both 128x128 = 16384 output values -- were refused by this backend until
    `device/run_dpu_points_probe.py` falsified the "8192 output points" reading of dpu.md 13.**

    v-heads are dealt round-robin over the tiles x SHAVEs (32 heads over 6 workers balances 6/6/6/5/5/4, which
    blocking into 6 would not); chunks within a head stay ordered because the state carries. `Nt[0, 0]` is the live
    token count, so one kernel serves any prompt up to NP = ceil(N / C) * C; a chunk's rows past it are zeroed and
    contribute nothing to gc, to the products or to the state. Tensors are head-major so a chunk is one contiguous
    DMA, except `Out`, which is token-major like npu_dsl's layer output. Oracle: `gdn_scan_prefill_reference`,
    which runs the RECURRENCE, not this algebra."""
    assert DK == DV and VH % KH == 0 and C % 32 == 0 and DK % 32 == 0 and DV % 32 == 0
    assert C % 16 == 0 and DK % 16 == 0, "N and K of every workload are 16-aligned"
    REP = VH // KH
    NP = -(-N // C) * C
    NC = NP // C
    VC = VH * DV
    workers = tiles * shaves
    HL, FL = 32, 16
    INV = 1.0 / GDN_Q_SCALE

    @T.macro
    def scan_phase(QK, V, Gb, Z, GNorm, Nt, Ssm, Out, tile, shave):
        """The phase as a macro so P4 can put it in a layer launch after the conv;
        `make_gdn_scan_prefill` is the standalone wrapper."""
        worker = tile * shaves + shave
        eye = T.alloc_shared((DK, DK), T.float16)
        gn = T.alloc_shared((1, DV), T.float16)
        nt = T.alloc_shared((1, 1), T.int32)
        Sf = T.alloc_local((DK, DV), T.float32)
        S16 = T.alloc_local((DK, DV), T.float16)
        ST = T.alloc_local((DV, DK), T.float16)
        qkc = T.alloc_local((C, 2 * DK), T.float32)
        vin = T.alloc_local((C, DV), T.float16)
        gbv = T.alloc_local((C, 2), T.float32)
        zb = T.alloc_local((C, DV), T.float16)
        kq = T.alloc_local((2 * C, DK), T.float16)
        khb = T.alloc_local((C, DK), T.float16)
        qg = T.alloc_local((C, DK), T.float16)
        kg = T.alloc_local((C, DK), T.float16)
        kd = T.alloc_local((C, DK), T.float16)
        vb = T.alloc_local((C, DV), T.float16)
        dm = T.alloc_local((C, C), T.float16)
        pm = T.alloc_local((C, C), T.float16)
        tri = T.alloc_local((C, C), T.float16)
        ap = T.alloc_local((2 * C, C), T.float16)
        vbT = T.alloc_local((DK, C), T.float16)
        kgT = T.alloc_local((DK, C), T.float16)
        kdT = T.alloc_local((DK, C), T.float16)
        uT = T.alloc_local((DV, C), T.float16)
        vpT = T.alloc_local((DV, C), T.float16)
        vnT = T.alloc_local((DV, C), T.float16)
        W = T.alloc_local((C, DK), T.float16)
        qi = T.alloc_local((C, DV), T.float16)
        o1 = T.alloc_local((C, DV), T.float16)
        on = T.alloc_local((C, DV), T.float16)
        dS = T.alloc_local((DK, DV), T.float16)
        gc = T.alloc_local((C,), T.float32)
        la = T.alloc_local((C,), T.float32)
        ac = T.alloc_local((C,), T.float32)
        xr = T.alloc_local((DV,), T.float32)
        lq = T.alloc_local((FL,), T.float32)
        lk = T.alloc_local((FL,), T.float32)
        ls = T.alloc_local((FL,), T.float32)
        eh = T.alloc_local((FL,), T.float16)
        st = T.alloc_local((8,), T.float32)
        T.copy(GNorm[0, 0], gn)
        T.copy(Nt[0, 0], nt)
        if shave == 0:
            for i, j0 in T.grid(DK, DK // HL):
                for v in T.vectorized(HL):
                    eye[i, j0 * HL + v] = T.float16(0)
            for i in T.serial(DK):
                eye[i, i] = T.float16(1)
        T.sync_threads()
        for ht in T.serial(T.ceildiv(VH, workers)):
            h = ht * workers + worker
            if h < VH:
                T.copy(Ssm[h, 0, 0], Sf)
                for d, c0 in T.grid(DK, DV // FL):
                    for v in T.vectorized(FL):
                        S16[d, c0 * FL + v] = T.cast(Sf[d, c0 * FL + v], T.float16)
                T.gemm(eye, S16, ST, transpose_B=True, clear_accum=True)
                for ci in T.serial(NC):
                    tb = T.max(0, T.min(C, nt[0, 0] - ci * C))
                    T.copy(QK[h // REP, ci * C, 0], qkc)
                    T.copy(V[h, ci * C, 0], vin)
                    T.copy(Gb[h, ci * C, 0], gbv)
                    T.copy(Z[h, ci * C, 0], zb)
                    # --- the chunk's row operands: gc, then every row scaled by its gate -----------------
                    st[0] = T.float32(0)
                    for t in T.serial(C):
                        st[0] = st[0] + T.if_then_else(t < tb, gbv[t, 0], T.float32(0))
                        gc[t] = st[0]
                    for t in T.serial(C):
                        if t < tb:
                            st[2] = gbv[t, 1]                           # beta
                            st[3] = T.exp(gc[t])
                            st[4] = T.exp(gc[C - 1] - gc[t])
                            for d0 in T.serial(DK // FL):
                                for v in T.vectorized(FL):
                                    lq[v] = qkc[t, d0 * FL + v] * T.float32(GDN_Q_SCALE)
                                    lk[v] = qkc[t, DK + d0 * FL + v]
                                    kq[C + t, d0 * FL + v] = T.cast(lq[v], T.float16)
                                    khb[t, d0 * FL + v] = T.cast(lk[v], T.float16)
                                    kq[t, d0 * FL + v] = T.cast(lk[v] * st[2], T.float16)
                                    qg[t, d0 * FL + v] = T.cast(lq[v] * st[3], T.float16)
                                    kg[t, d0 * FL + v] = T.cast(lk[v] * st[2] * st[3], T.float16)
                                    kd[t, d0 * FL + v] = T.cast(lk[v] * st[4], T.float16)
                            for d0 in T.serial(DV // FL):
                                for v in T.vectorized(FL):
                                    vb[t, d0 * FL + v] = T.cast(
                                        T.cast(vin[t, d0 * FL + v], T.float32) * st[2], T.float16)
                        else:
                            for d0 in T.serial(DK // HL):
                                for v in T.vectorized(HL):
                                    kq[t, d0 * HL + v] = T.float16(0)
                                    kq[C + t, d0 * HL + v] = T.float16(0)
                                    khb[t, d0 * HL + v] = T.float16(0)
                                    qg[t, d0 * HL + v] = T.float16(0)
                                    kg[t, d0 * HL + v] = T.float16(0)
                                    kd[t, d0 * HL + v] = T.float16(0)
                            for d0 in T.serial(DV // HL):
                                for v in T.vectorized(HL):
                                    vb[t, d0 * HL + v] = T.float16(0)
                    # D: the whole row vectorised, then the strictly-upper part zeroed. exp overflows to inf
                    # above the diagonal and is overwritten there; a per-lane select does not vectorise.
                    for i in T.serial(C):
                        for j0 in T.serial(C // FL):
                            for v in T.vectorized(FL):
                                dm[i, j0 * FL + v] = T.cast(T.exp(gc[i] - gc[j0 * FL + v]), T.float16)
                        for j in T.serial(i + 1, C):
                            dm[i, j] = T.float16(0)
                    # --- products: [beta k k^T; s q k^T], and the three transposes -----------------------
                    T.gemm(kq, khb, ap, transpose_B=True, clear_accum=True)
                    T.gemm(eye, vb, vbT, transpose_B=True, clear_accum=True)
                    T.gemm(eye, kg, kgT, transpose_B=True, clear_accum=True)
                    T.gemm(eye, kd, kdT, transpose_B=True, clear_accum=True)
                    # --- T = (I + strictly_lower(A . D))^-1 by forward substitution ----------------------
                    for j0 in T.serial(C // HL):
                        for v in T.vectorized(HL):
                            tri[0, j0 * HL + v] = T.float16(0)
                    for i in T.serial(1, C):
                        for j0 in T.serial(C // FL):
                            for v in T.vectorized(FL):
                                la[j0 * FL + v] = T.float32(0) - (T.cast(ap[i, j0 * FL + v], T.float32)
                                                                  * T.cast(dm[i, j0 * FL + v], T.float32))
                        for j in T.serial(i, C):
                            la[j] = T.float32(0)        # row i is strictly lower; column i on is not its own
                        for j0 in T.serial(C // FL):
                            for v in T.vectorized(FL):
                                ac[j0 * FL + v] = la[j0 * FL + v]
                        for m in T.serial(i):
                            st[5] = la[m]
                            for j0 in T.serial(C // FL):
                                for v in T.vectorized(FL):
                                    ac[j0 * FL + v] = ac[j0 * FL + v] + T.cast(tri[m, j0 * FL + v],
                                                                               T.float32) * st[5]
                        for j0 in T.serial(C // FL):
                            for v in T.vectorized(FL):
                                tri[i, j0 * FL + v] = T.cast(ac[j0 * FL + v], T.float16)
                    for i in T.serial(C):
                        tri[i, i] = T.float16(1)
                    for i in T.serial(C):
                        for j0 in T.serial(C // HL):
                            for v in T.vectorized(HL):
                                pm[i, j0 * HL + v] = ap[C + i, j0 * HL + v] * dm[i, j0 * HL + v]
                    # --- v_new, the output halves and the state increment --------------------------------
                    T.gemm(vbT, tri, uT, transpose_B=True, clear_accum=True)
                    T.gemm(tri, kgT, W, transpose_B=True, clear_accum=True)
                    T.gemm(ST, W, vpT, transpose_B=True, clear_accum=True)
                    T.gemm(qg, ST, qi, transpose_B=True, clear_accum=True)
                    for i in T.serial(DV):
                        for j0 in T.serial(C // HL):
                            for v in T.vectorized(HL):
                                vnT[i, j0 * HL + v] = uT[i, j0 * HL + v] - vpT[i, j0 * HL + v]
                    T.gemm(pm, vnT, o1, transpose_B=True, clear_accum=True)
                    T.gemm(kdT, vnT, dS, transpose_B=True, clear_accum=True)
                    st[6] = T.exp(gc[C - 1])
                    for d in T.serial(DK):
                        for c0 in T.serial(DV // FL):
                            for v in T.vectorized(FL):
                                Sf[d, c0 * FL + v] = (Sf[d, c0 * FL + v] * st[6]
                                                      + T.cast(dS[d, c0 * FL + v], T.float32))
                                S16[d, c0 * FL + v] = T.cast(Sf[d, c0 * FL + v], T.float16)
                    T.gemm(eye, S16, ST, transpose_B=True, clear_accum=True)
                    # --- gated RMSNorm, with the q pre-scale divided back out ----------------------------
                    for i in T.serial(C):
                        for v in T.vectorized(FL):
                            ls[v] = T.float32(0)
                        for c0 in T.serial(DV // FL):
                            for v in T.vectorized(FL):
                                xr[c0 * FL + v] = (T.cast(qi[i, c0 * FL + v], T.float32)
                                                   + T.cast(o1[i, c0 * FL + v], T.float32))
                                ls[v] = ls[v] + xr[c0 * FL + v] * xr[c0 * FL + v]
                        st[7] = T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL])
                        st[7] = T.float32(INV) / T.sqrt(st[7] * T.float32(INV * INV) / T.float32(DV)
                                                        + T.float32(eps))
                        for c0 in T.serial(DV // FL):
                            for v in T.vectorized(FL):
                                eh[v] = T.exp(T.float16(0) - zb[i, c0 * FL + v])
                                on[i, c0 * FL + v] = T.cast(
                                    xr[c0 * FL + v] * st[7] * T.cast(gn[0, c0 * FL + v], T.float32)
                                    * T.cast(zb[i, c0 * FL + v], T.float32)
                                    / (T.float32(1) + T.cast(eh[v], T.float32)), T.float16)
                    T.copy(on, Out[ci * C, h * DV])
                T.copy(Sf, Ssm[h, 0, 0])

    if as_block:
        return scan_phase, dict(NP=NP, NC=NC, VC=VC)

    @T.prim_func
    def gdn_scan(QK: T.Tensor((KH, NP, 2 * DK), T.float32), V: T.Tensor((VH, NP, DV), T.float16),
                 Gb: T.Tensor((VH, NP, 2), T.float32), Z: T.Tensor((VH, NP, DV), T.float16),
                 GNorm: T.Tensor((1, DV), T.float16), Nt: T.Tensor((1, 1), T.int32),
                 Ssm: T.Tensor((VH, DK, DV), T.float32), Out: T.Tensor((NP, VC), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            scan_phase(QK, V, Gb, Z, GNorm, Nt, Ssm, Out, tile, T.get_thread_binding())

    return gdn_scan


def gdn_scan_prefill_reference(qk, v, gb, z, gnorm, ssm, n: int, eps: float = 1e-6):
    """numpy oracle of make_gdn_scan_prefill (fp32), as the RECURRENCE rather than the chunked algebra: the kernel's
    chunking, its triangular inverse and its decay matrix are all claims about this loop, so the oracle must not
    share them. Returns (out [NP, VH*DV], ssm after the scan)."""
    import numpy as np
    KH, NP, _, DK = qk.shape
    VH, _, DV = v.shape
    REP = VH // KH
    out = np.zeros((NP, VH * DV), np.float32)
    s = ssm.astype(np.float64).copy()
    for h in range(VH):
        for t in range(n):
            q = qk[h // REP, t, 0].astype(np.float64)
            k = qk[h // REP, t, 1].astype(np.float64)
            g, beta = float(gb[h, t, 0]), float(gb[h, t, 1])
            eg = np.exp(g)
            delta = (v[h, t].astype(np.float64) - eg * (s[h].T @ k)) * beta
            s[h] = s[h] * eg + np.outer(k, delta)
            o = s[h].T @ q
            zt = z[h, t].astype(np.float64)
            o = o / np.sqrt((o * o).mean() + eps) * gnorm.astype(np.float64).reshape(-1)
            out[t, h * DV:(h + 1) * DV] = o * zt / (1.0 + np.exp(-zt))
    return out, s.astype(np.float32)


def gdn_conv_prefill_reference(y, conv_w, conv_state, a_log, dt_bias, n: int, KH: int, VH: int, DK: int, DV: int,
                               ba=None):
    """numpy oracle of make_gdn_conv_prefill (fp32): the GDN prefill phase BETWEEN the input projection and the
    chunked scan. Returns exactly what `make_gdn_scan_prefill` consumes -- (qk, v, gb, z, new conv_state) -- so the
    two compose. Semantics are `gdn_reference`'s, run over `n` tokens instead of one: a causal conv1d of width 4 with
    SiLU over the CONV channels (the state is the 3 tokens before this chunk, time-major, oldest first), then q and k
    l2-normalised per k-head with q carrying the `1/sqrt(DK)` of attention, beta = sigmoid(b) and
    g = -exp(a_log) softplus(a + dt_bias) per v-head, and z passed through unconvolved.

    `ba` [NP, 2 VH]: the gate projections, which the checkpoint keeps outside the quantised in_proj. None takes
    them from the trailing columns of `y`, the layout the synthetic device-runner weights use."""
    import numpy as np
    QK, VC = KH * DK, VH * DV
    CONV = 2 * QK + VC
    NP = y.shape[0]
    mixed = y[:, :CONV].astype(np.float32)
    seq = np.concatenate([conv_state.astype(np.float32), mixed], axis=0)      # [3 + NP, CONV], oldest first
    w = conv_w.astype(np.float32)
    conv = sum(seq[j:j + NP] * w[j] for j in range(4))                        # token t sees rows t-3 .. t
    conv = conv / (1.0 + np.exp(-conv))
    q = conv[:, :QK].reshape(NP, KH, DK)
    k = conv[:, QK:2 * QK].reshape(NP, KH, DK)
    q = q / (np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * np.sqrt(DK))
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    qk = np.zeros((KH, NP, 2, DK), np.float32)
    qk[:, :, 0], qk[:, :, 1] = q.transpose(1, 0, 2), k.transpose(1, 0, 2)
    v = conv[:, 2 * QK:].reshape(NP, VH, DV).transpose(1, 0, 2).astype(np.float16)
    z = y[:, CONV:CONV + VC].astype(np.float32).reshape(NP, VH, DV).transpose(1, 0, 2).astype(np.float16)
    bb = y[:, CONV + VC:CONV + VC + VH].astype(np.float32) if ba is None else ba[:, :VH].astype(np.float32)
    a = (y[:, CONV + VC + VH:CONV + VC + 2 * VH].astype(np.float32) if ba is None
         else ba[:, VH:2 * VH].astype(np.float32))
    gb = np.zeros((VH, NP, 2), np.float32)
    gb[:, :, 0] = (-np.exp(a_log.astype(np.float32).reshape(1, VH)) * np.logaddexp(0, a + dt_bias.astype(np.float32).reshape(1, VH))).T
    gb[:, :, 1] = (1.0 / (1.0 + np.exp(-bb))).T
    # rows past the live tokens contribute nothing downstream; the state is the last three LIVE rows
    qk[:, n:], v[:, n:], z[:, n:], gb[:, n:] = 0, 0, 0, 0
    new_cs = seq[n:n + 3].astype(np.float16)
    return qk, v, gb, z, new_cs


def prefill_ba_block(H: int, VH: int, NP: int, MB: int, tiles: int, shaves: int):
    """`BAs[NP, 2 VH] = Xs[NP, H] . Wba[2 VH, H]^T`, fp16 on the DPU.

    The checkpoint keeps the two Gated DeltaNet gate projections OUTSIDE the quantised in_proj -- bf16 rows that
    `pack_gdn` emits as `Wba` -- which is why decode computes them with per-row SHAVE dot products instead of
    reading them out of the projection output. Prefill has NP tokens, so the same thing is an ordinary small gemm.
    Writing it the other way (expecting b | a as columns of the in_proj output) is what made the first E2E prefill
    read 12352 columns of a 12288-column tensor and return NaN."""
    RB = 16                      # the DPU's output granule; 2 VH = 64 is four chunks
    NCH = (2 * VH) // RB
    workers = tiles * shaves
    assert (2 * VH) % RB == 0 and NP % MB == 0

    @T.macro
    def ba_phase(Xs, Wba, BAs, tile, shave):
        worker = tile * shaves + shave
        w = T.alloc_local((RB, H), T.float16)
        x = T.alloc_local((MB, H), T.float16)
        y = T.alloc_local((MB, RB), T.float16)
        w[0, 0] = T.float16(0)     # pins the scope at kernel level (question 10)
        x[0, 0] = T.float16(0)
        y[0, 0] = T.float16(0)
        for it in T.serial(T.ceildiv(NCH, workers)):
            n = it * workers + worker
            if n < NCH:
                T.copy(Wba[n * RB, 0], w)
                for b in T.serial(NP // MB):
                    T.copy(Xs[b * MB, 0], x)
                    T.gemm(x, w, y, transpose_B=True, clear_accum=True)
                    T.copy(y, BAs[b * MB, n * RB])

    return ba_phase


def make_gdn_conv_prefill(N: int = 128, KH: int = 16, VH: int = 32, DK: int = 128, DV: int = 128, TB: int = 32,
                          tiles: int = 3, shaves: int = 2, as_block: bool = False, blocked_items: bool = False,
                          item_worker: str = "tile"):
    """P3b: the Gated DeltaNet PREFILL phase between the input projection and the chunked scan -- a causal conv1d of
    width 4 with SiLU, then the per-head normalisations -- producing exactly what `make_gdn_scan_prefill` consumes.

    The conv is causal in t and independent per channel, so the work splits on **heads**: item `hb < KH` owns k-head
    `hb` and does its q and k channels, item `hb >= KH` owns v-head `hb - KH` and does its v channels plus that
    head's z, beta and g. One head's DK channels stay in one worker, which is what lets the l2 norm be a local
    reduction; the conv state of those channels is then also private, so no worker touches another's.

    Tokens are walked in blocks of TB with the 3-token window carried in CMX across blocks, so a block is one 2-D
    DMA (TB rows of DK out of a row of NIN) rather than TB small ones. `Nt[0, 0]` is the live token count: rows past
    it are zeroed, and the conv state handed on is the last three LIVE rows.

    Every buffer is touched before the loops. ScopeIntelNpuCmx hoists loop-invariant allocations itself now, but
    only out to the innermost enclosing non-loop scope; the touch pins them at kernel level, which is the form
    this kernel's device gate was run in (plan question 10).

    Oracle: `gdn_conv_prefill_reference`.

    `blocked_items`: a worker takes a CONTIGUOUS run of heads instead of every `workers`-th one, which is how
    npu_dsl splits the same work (by channel block). The channels a worker owns are then contiguous, so its state
    slices could become one wide DMA; it also changes WHICH worker owns a channel, which is what makes it a
    diagnostic for the 3-tile multi-layer defect. `item_worker='shave'` deals those contiguous runs shave-major
    (`shave * tiles + tile`), which keeps the runs contiguous but spreads them over the tiles -- the control that
    separates "contiguous per worker" from "how many tiles write the same region"."""
    assert DK == DV and VH % KH == 0 and DK % 32 == 0 and TB % 16 == 0
    QK, VC = KH * DK, VH * DV
    CONV = 2 * QK + VC
    NIN = CONV + VC          # b | a are NOT here: the checkpoint keeps them in a separate fp16 `Wba` (see
                             # `prefill_ba_block`), so the in_proj output is exactly what `pack_gdn` packs
    NP = -(-N // TB) * TB
    NT = NP // TB
    ITEMS = KH + VH
    workers = tiles * shaves
    PER = -(-ITEMS // workers)
    FL, HL = 16, 32

    @T.macro
    def conv_phase(Y, Ba, ConvW, ConvState, ALog, DtBias, Nt, QKo, Vo, Gb, Zo, CsOut, tile, shave, bufs=None):
        """The phase as a macro so P4 can put it in a layer launch; `make_gdn_conv_prefill` is the
        standalone wrapper. `tile` and `shave` come from the caller's launch.

        `bufs`: a dict of pre-allocated buffers from the caller. A kernel whose layer loop runs this phase inside
        an `if` on the layer kind must pass them: ScopeIntelNpuCmx hoists only a loop body's TOP-LEVEL
        declarations, so allocations inside the branch are re-made every iteration -- the state plan question 10
        showed to be unsound, and the one npu_dsl never enters (it packs one arena once)."""
        worker = tile * shaves + shave
        nt = T.alloc_shared((1, 1), T.int32) if bufs is None else bufs["nt"]
        al = T.alloc_shared((1, VH), T.float32) if bufs is None else bufs["al"]
        db = T.alloc_shared((1, VH), T.float32) if bufs is None else bufs["db"]
        win = (T.alloc_local((3, DK), T.float16) if bufs is None else bufs["win"])    # the 3 tokens before the row
        keep = (T.alloc_local((3, DK), T.float16) if bufs is None else bufs["keep"])  # that window at the last LIVE token
        cw = T.alloc_local((4, DK), T.float16) if bufs is None else bufs["cw"]
        xb = (T.alloc_local((TB, DK), T.float16) if bufs is None else bufs["xb"])     # a token block of this head's channels
        cv = (T.alloc_local((TB, DK), T.float32) if bufs is None else bufs["cv"])     # conv + SiLU
        ob = T.alloc_local((TB, DK), T.float32) if bufs is None else bufs["ob"]
        zb = T.alloc_local((TB, DV), T.float16) if bufs is None else bufs["zb"]
        bb = (T.alloc_local((TB, 1), T.float16) if bufs is None else bufs["bb"])      # b and a are VH columns apart
        aa = T.alloc_local((TB, 1), T.float16) if bufs is None else bufs["aa"]
        gbb = T.alloc_local((TB, 2), T.float32) if bufs is None else bufs["gbb"]
        acc = T.alloc_local((FL,), T.float32) if bufs is None else bufs["acc"]
        eh = T.alloc_local((FL,), T.float16) if bufs is None else bufs["eh"]
        ls = T.alloc_local((FL,), T.float32) if bufs is None else bufs["ls"]
        st = T.alloc_local((8,), T.float32) if bufs is None else bufs["st"]
        if bufs is None:
            # first use here pins every CMX scope at kernel level (question 10)
            win[0, 0] = T.float16(0)
            keep[0, 0] = T.float16(0)
            cw[0, 0] = T.float16(0)
            xb[0, 0] = T.float16(0)
            cv[0, 0] = T.float32(0)
            ob[0, 0] = T.float32(0)
            zb[0, 0] = T.float16(0)
            bb[0, 0] = T.float16(0)
            aa[0, 0] = T.float16(0)
            gbb[0, 0] = T.float32(0)
            acc[0] = T.float32(0)
            ls[0] = T.float32(0)
            eh[0] = T.float16(0)
            st[0] = T.float32(0)
        T.copy(Nt[0, 0], nt)
        T.copy(ALog[0, 0], al)
        T.copy(DtBias[0, 0], db)
        T.sync_threads()
        for it in T.serial(T.ceildiv(ITEMS, workers)):
            wid = (shave * tiles + tile) if item_worker == "shave" else worker
            hb = (wid * PER + it) if blocked_items else (it * workers + worker)
            if hb < ITEMS:
                # channel base of this item: k-heads take q then k, v-heads take their v slice
                for part in T.serial(2):
                    if hb >= KH and part == 1:
                        st[0] = T.float32(0)        # a v-head has one channel slice, not two
                    else:
                        st[1] = T.float32(0)
                        cb = T.if_then_else(hb < KH, hb * DK + part * QK, 2 * QK + (hb - KH) * DV)
                        T.copy(ConvState[0, cb], win)
                        T.copy(ConvW[0, cb], cw)
                        for tb in T.serial(NT):
                            T.copy(Y[tb * TB, cb], xb)
                            for r in T.serial(TB):
                                for c0 in T.serial(DK // FL):
                                    for v in T.vectorized(FL):
                                        acc[v] = (T.cast(win[0, c0 * FL + v], T.float32) * T.cast(cw[0, c0 * FL + v], T.float32)
                                                  + T.cast(win[1, c0 * FL + v], T.float32) * T.cast(cw[1, c0 * FL + v], T.float32)
                                                  + T.cast(win[2, c0 * FL + v], T.float32) * T.cast(cw[2, c0 * FL + v], T.float32)
                                                  + T.cast(xb[r, c0 * FL + v], T.float32) * T.cast(cw[3, c0 * FL + v], T.float32))
                                        eh[v] = T.exp(T.cast(T.float32(0) - acc[v], T.float16))
                                        cv[r, c0 * FL + v] = acc[v] / (T.float32(1) + T.cast(eh[v], T.float32))
                                        win[0, c0 * FL + v] = win[1, c0 * FL + v]
                                        win[1, c0 * FL + v] = win[2, c0 * FL + v]
                                        win[2, c0 * FL + v] = xb[r, c0 * FL + v]
                                if tb * TB + r == nt[0, 0] - 1:
                                    for c0 in T.serial(DK // HL):
                                        for v in T.vectorized(HL):
                                            keep[0, c0 * HL + v] = win[0, c0 * HL + v]
                                            keep[1, c0 * HL + v] = win[1, c0 * HL + v]
                                            keep[2, c0 * HL + v] = win[2, c0 * HL + v]
                            # l2 norm per token, then the slice's destination
                            for r in T.serial(TB):
                                for v in T.vectorized(FL):
                                    ls[v] = T.float32(0)
                                for c0 in T.serial(DK // FL):
                                    for v in T.vectorized(FL):
                                        ls[v] = ls[v] + cv[r, c0 * FL + v] * cv[r, c0 * FL + v]
                                st[2] = T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL])
                                st[3] = T.float32(1) / T.sqrt(st[2] + T.float32(1e-6))
                                if hb < KH and part == 0:
                                    st[3] = st[3] * T.float32(1.0 / (DK ** 0.5))
                                st[4] = T.if_then_else(tb * TB + r < nt[0, 0], T.float32(1), T.float32(0))
                                for c0 in T.serial(DK // FL):
                                    for v in T.vectorized(FL):
                                        ob[r, c0 * FL + v] = T.if_then_else(
                                            hb < KH, cv[r, c0 * FL + v] * st[3], cv[r, c0 * FL + v]) * st[4]
                            if hb < KH:
                                T.copy(ob, QKo[hb, tb * TB, part * DK])
                            else:
                                for r in T.serial(TB):
                                    for c0 in T.serial(DV // FL):
                                        for v in T.vectorized(FL):
                                            zb[r, c0 * FL + v] = T.cast(ob[r, c0 * FL + v], T.float16)
                                T.copy(zb, Vo[hb - KH, tb * TB, 0])
                        T.copy(keep, CsOut[0, cb])
        T.sync_grid()
        # z, beta and g: one v-head per item, read straight out of the projection row
        for it in T.serial(T.ceildiv(VH, workers)):
            h = it * workers + worker
            if h < VH:
                for tb in T.serial(NT):
                    T.copy(Y[tb * TB, CONV + h * DV], zb)
                    for r in T.serial(TB):
                        st[5] = T.if_then_else(tb * TB + r < nt[0, 0], T.float32(1), T.float32(0))
                        for c0 in T.serial(DV // FL):
                            for v in T.vectorized(FL):
                                zb[r, c0 * FL + v] = T.cast(T.cast(zb[r, c0 * FL + v], T.float32) * st[5], T.float16)
                    T.copy(zb, Zo[h, tb * TB, 0])
                    T.copy(Ba[tb * TB, h], bb)
                    T.copy(Ba[tb * TB, VH + h], aa)
                    for r in T.serial(TB):
                        st[6] = T.if_then_else(tb * TB + r < nt[0, 0], T.float32(1), T.float32(0))
                        gbb[r, 1] = st[6] / (T.float32(1) + T.exp(T.float32(0) - T.cast(bb[r, 0], T.float32)))
                        st[7] = T.cast(aa[r, 0], T.float32) + db[0, h]
                        gbb[r, 0] = (T.float32(0) - T.exp(al[0, h])) * T.log(T.float32(1) + T.exp(st[7])) * st[6]
                    T.copy(gbb, Gb[h, tb * TB, 0])

    if as_block:
        return conv_phase, dict(NP=NP, NIN=NIN, CONV=CONV, QK=QK, VC=VC)

    @T.prim_func
    def gdn_conv(Y: T.Tensor((NP, NIN), T.float16), Ba: T.Tensor((NP, 2 * VH), T.float16),
                 ConvW: T.Tensor((4, CONV), T.float16),                 ConvState: T.Tensor((3, CONV), T.float16), ALog: T.Tensor((1, VH), T.float32),
                 DtBias: T.Tensor((1, VH), T.float32), Nt: T.Tensor((1, 1), T.int32),
                 QKo: T.Tensor((KH, NP, 2 * DK), T.float32), Vo: T.Tensor((VH, NP, DV), T.float16),
                 Gb: T.Tensor((VH, NP, 2), T.float32), Zo: T.Tensor((VH, NP, DV), T.float16),
                 CsOut: T.Tensor((3, CONV), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            conv_phase(Y, Ba, ConvW, ConvState, ALog, DtBias, Nt, QKo, Vo, Gb, Zo, CsOut,
                       tile, T.get_thread_binding())

    return gdn_conv


def make_gdn_mixer_prefill(N: int = 128, KH: int = 16, VH: int = 32, DK: int = 128, DV: int = 128, TB: int = 32,
                           C: int = 64, tiles: int = 3, shaves: int = 2, eps: float = 1e-6):
    """P4 step 3: the GDN prefill mixer in ONE launch -- the conv phase (P3b) and the chunked scan (P3) with a grid
    barrier between them, instead of two launches with the four intermediates round-tripping through the host.

    The intermediates stay in DDR as scratch (`QKs`, `Vs`, `Gbs`, `Zs`); the barrier is what makes them safe, since
    the conv deals heads over the workers one way and the scan another. Both phases keep their own token geometry:
    NP is a multiple of both TB and C, so one prompt length serves both.

    Composing changes the loop nesting, which is what plan question 10 acts on, so the test asserts this kernel
    still lowers with no allocation inside a loop body -- the detector, not a comment."""
    conv_phase, _ = make_gdn_conv_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB, tiles=tiles, shaves=shaves,
                                          as_block=True)
    scan_phase, _ = make_gdn_scan_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, C=C, tiles=tiles, shaves=shaves, eps=eps,
                                          as_block=True)
    QK, VC = KH * DK, VH * DV
    CONV = 2 * QK + VC
    NIN = CONV + VC
    step = TB * C // math.gcd(TB, C)
    NP = -(-N // step) * step
    assert NP == -(-N // TB) * TB == -(-N // C) * C, "one token count must serve both phases"

    @T.prim_func
    def gdn_mixer_prefill(Y: T.Tensor((NP, NIN), T.float16), Ba: T.Tensor((NP, 2 * VH), T.float16),
                          ConvW: T.Tensor((4, CONV), T.float16),
                          ConvState: T.Tensor((3, CONV), T.float16), ALog: T.Tensor((1, VH), T.float32),
                          DtBias: T.Tensor((1, VH), T.float32), GNorm: T.Tensor((1, DV), T.float16),
                          Nt: T.Tensor((1, 1), T.int32), Ssm: T.Tensor((VH, DK, DV), T.float32),
                          QKs: T.Tensor((KH, NP, 2 * DK), T.float32), Vs: T.Tensor((VH, NP, DV), T.float16),
                          Gbs: T.Tensor((VH, NP, 2), T.float32), Zs: T.Tensor((VH, NP, DV), T.float16),
                          CsOut: T.Tensor((3, CONV), T.float16), Out: T.Tensor((NP, VC), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            conv_phase(Y, Ba, ConvW, ConvState, ALog, DtBias, Nt, QKs, Vs, Gbs, Zs, CsOut, tile, shave)
            T.sync_grid()
            scan_phase(QKs, Vs, Gbs, Zs, GNorm, Nt, Ssm, Out, tile, shave)

    return gdn_mixer_prefill


def make_proj_prefill(NP: int = 128, M: int = 1024, K: int = 2048, BN: int = 64, MB: int = 32, G: int = 128,
                      tiles: int = 3, shaves: int = 2, as_block: bool = False, scales_gm: bool = False):
    """P4 step 4: a PREFILL projection -- `Yout[NP, M] = X[NP, K] . dequant(Wq[M, K])^T`, int4 symmetric with one
    fp16 scale per group of G along K.

    Decode's projection is a GEMV and goes through the DPU's merged-FC (`T.gemm_i4`), whose scratch is sized for
    M = 1. Prefill takes `make_w4a16_flow`'s path instead -- the SHAVE dequantises a chunk of output rows into fp16
    and the DPU runs an ordinary fp16 gemm -- because the dequant's cost does not grow with the token count and so
    amortises over the whole prompt, while the merged-FC scratch grows with it.

    Output-row chunks of BN are dealt over the workers and a worker dequantises its chunk ONCE, then walks the
    tokens in blocks of MB against it. That ordering is the point: the other way round re-dequantises the same
    weights for every token block.

    `scales_gm`: the scales are GROUP-MAJOR `[K/G, M]`, which is the layout the DPU's int4 path reads in place, so
    one scale tensor serves prefill and decode and the weights are uploaded once (npu_dsl's arrangement). The
    chunk's scales are then a `[K/G, BN]` 2-D read at stride M, and the dequant is `tl_npu_dequant_i4_gm`, which
    keeps the VPUX assembly by transposing a chunk's scales first. `False` is the ROW-MAJOR `[M, K/G]` form.
    Oracle: `proj_prefill_reference`."""
    assert M % BN == 0 and NP % MB == 0 and K % G == 0 and K % 32 == 0
    KG = K // G
    NCH, NTB = M // BN, NP // MB
    workers = tiles * shaves
    HL = 32
    assert MB <= 8192 and BN <= 8192 and K <= 8192, "every workload dimension is bounded by 8192 (dpu.md 4.6)"

    @T.macro
    def proj_phase(X, Wq, S, Yout, tile, shave):
        worker = tile * shaves + shave
        wq_cmx = T.alloc_local((BN, K // 2), T.uint8)
        s_cmx = T.alloc_local((KG, BN) if scales_gm else (BN, KG), T.float16)
        w_cmx = T.alloc_local((BN, K), T.float16)
        x_cmx = T.alloc_local((MB, K), T.float16)
        y_cmx = T.alloc_local((MB, BN), T.float16)
        wq_cmx[0, 0] = T.uint8(0)      # pins the scope at kernel level (question 10)
        s_cmx[0, 0] = T.float16(0)
        w_cmx[0, 0] = T.float16(0)
        x_cmx[0, 0] = T.float16(0)
        y_cmx[0, 0] = T.float16(0)
        for it in T.serial(T.ceildiv(NCH, workers)):
            n = it * workers + worker
            if n < NCH:
                T.copy(Wq[n * BN, 0], wq_cmx)
                if scales_gm:
                    T.copy(S[0, n * BN], s_cmx)
                    T.call_extern("handle", "tl_npu_dequant_i4_gm", T.address_of(w_cmx[0, 0]),
                                  T.address_of(wq_cmx[0, 0]), T.address_of(s_cmx[0, 0]), BN, K, G)
                else:
                    T.copy(S[n * BN, 0], s_cmx)
                    T.call_extern("handle", "tl_npu_dequant_i4_block", T.address_of(w_cmx[0, 0]),
                                  T.address_of(wq_cmx[0, 0]), T.address_of(s_cmx[0, 0]), BN, K, G)
                for b in T.serial(NTB):
                    T.copy(X[b * MB, 0], x_cmx)
                    T.gemm(x_cmx, w_cmx, y_cmx, transpose_B=True, clear_accum=True)
                    T.copy(y_cmx, Yout[b * MB, n * BN])

    if as_block:
        return proj_phase, dict(NCH=NCH, NTB=NTB, KG=KG)

    @T.prim_func
    def proj(X: T.Tensor((NP, K), T.float16), Wq: T.Tensor((M, K // 2), T.uint8),
             S: T.Tensor((KG, M) if scales_gm else (M, KG), T.float16), Yout: T.Tensor((NP, M), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            proj_phase(X, Wq, S, Yout, tile, T.get_thread_binding())

    return proj


def proj_prefill_reference(x, w_int, scales, G: int):
    """numpy oracle of make_proj_prefill (fp32): y[NP, M] = x @ dequant(w)^T."""
    import numpy as np
    w = w_int.astype(np.float32) * np.repeat(scales.astype(np.float32), G, axis=1)
    return x.astype(np.float32) @ w.T


def prefill_norm_block(H: int, NP: int, workers: int, eps: float = 1e-6, FL: int = 16):
    """The prefill layers' per-token zero-centred RMSNorm: the fp32 residual stream `Hio` into the fp16 scratch
    `Xs`, tokens dealt over the workers. The caller owns the buffers so that every phase of a layer shares them
    (an allocation per call would land inside the layer's loops: plan question 10)."""

    @T.macro
    def norm_tokens(Hio, Wnorm, Xs, nw, hrow, xrow, ls, st, worker):
        T.copy(Wnorm[0, 0], nw)
        for it in T.serial(NP // workers):
            t = it * workers + worker
            T.copy(Hio[t, 0], hrow)
            for v in T.vectorized(FL):
                ls[v] = T.float32(0)
            for d0 in T.serial(H // FL):
                for v in T.vectorized(FL):
                    ls[v] = ls[v] + hrow[0, d0 * FL + v] * hrow[0, d0 * FL + v]
            st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(H)
                                          + T.float32(eps))
            for d0 in T.serial(H // FL):
                for v in T.vectorized(FL):
                    xrow[0, d0 * FL + v] = T.cast(hrow[0, d0 * FL + v] * st[0]
                                                  * (T.float32(1) + T.cast(nw[0, d0 * FL + v], T.float32)),
                                                  T.float16)
            T.copy(xrow, Xs[t, 0])

    return norm_tokens


def make_gdn_layer_prefill(N: int = 128, H: int = 2048, KH: int = 16, VH: int = 32, DK: int = 128, DV: int = 128,
                           G: int = 128, TB: int = 32, C: int = 64, BN1: int = 64, BN2: int = 64, MB: int = 32,
                           tiles: int = 3, shaves: int = 2, eps: float = 1e-6):
    """P4 step 4: a whole Gated DeltaNet PREFILL mixer layer in ONE launch, over `N` prompt tokens --
    `make_gdn_layer`'s shape with every phase in its prefill form:

      x = rmsnorm_zc(h, in_norm) per token -> in_proj (P4 step 4's SHAVE-dequant projection) -> the conv phase
      (P3b) -> the chunked scan (P3) -> out_proj -> h += mix

    The residual stream is fp32 in DDR (`Hio[NP, H]`), read at entry and written at exit, as the decode layer does.
    The phases are separated by grid barriers, because each deals its work differently: the projections by
    output-row chunk, the conv by head, the scan by v-head.

    The MoE half of the layer is `make_gdn_moe_layer_prefill` (step 5), which adds a second norm, P2a's routing
    and P2b's grouped expert FFN after this. Oracle: `gdn_layer_prefill_reference`."""
    conv_phase, _ = make_gdn_conv_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB, tiles=tiles, shaves=shaves,
                                          as_block=True)
    scan_phase, _ = make_gdn_scan_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, C=C, tiles=tiles, shaves=shaves, eps=eps,
                                          as_block=True)
    QK, VC = KH * DK, VH * DV
    CONV = 2 * QK + VC
    NIN = CONV + VC
    step = TB * C // math.gcd(TB, C)
    NP = -(-N // step) * step
    NIN_pad = -(-NIN // BN1) * BN1
    in_proj, _ = make_proj_prefill(NP=NP, M=NIN_pad, K=H, BN=BN1, MB=MB, G=G, tiles=tiles, shaves=shaves,
                                   as_block=True, scales_gm=True)
    out_proj, _ = make_proj_prefill(NP=NP, M=H, K=VC, BN=BN2, MB=MB, G=G, tiles=tiles, shaves=shaves, as_block=True,
                                    scales_gm=True)
    workers = tiles * shaves
    FL = 16
    assert H % FL == 0 and NP % workers == 0
    norm_tokens = prefill_norm_block(H, NP, workers, eps, FL)
    ba_phase = prefill_ba_block(H, VH, NP, MB, tiles, shaves)

    @T.prim_func
    def gdn_layer_prefill(Hio: T.Tensor((NP, H), T.float32), InNorm: T.Tensor((1, H), T.float16),
                          WinQ: T.Tensor((NIN_pad, H // 2), T.uint8), WinS: T.Tensor((H // G, NIN_pad), T.float16),
                          Wba: T.Tensor((2 * VH, H), T.float16),
                          ConvW: T.Tensor((4, CONV), T.float16), ConvState: T.Tensor((3, CONV), T.float16),
                          ALog: T.Tensor((1, VH), T.float32), DtBias: T.Tensor((1, VH), T.float32),
                          GNorm: T.Tensor((1, DV), T.float16), Nt: T.Tensor((1, 1), T.int32),
                          Ssm: T.Tensor((VH, DK, DV), T.float32),
                          WoutQ: T.Tensor((H, VC // 2), T.uint8), WoutS: T.Tensor((VC // G, H), T.float16),
                          Xs: T.Tensor((NP, H), T.float16), Ys: T.Tensor((NP, NIN_pad), T.float16),
                          BAs: T.Tensor((NP, 2 * VH), T.float16),
                          QKs: T.Tensor((KH, NP, 2 * DK), T.float32), Vs: T.Tensor((VH, NP, DV), T.float16),
                          Gbs: T.Tensor((VH, NP, 2), T.float32), Zs: T.Tensor((VH, NP, DV), T.float16),
                          Os: T.Tensor((NP, VC), T.float16), Mix: T.Tensor((NP, H), T.float16),
                          CsOut: T.Tensor((3, CONV), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            nw = T.alloc_shared((1, H), T.float16)
            hrow = T.alloc_local((1, H), T.float32)
            xrow = T.alloc_local((1, H), T.float16)
            mrow = T.alloc_local((1, H), T.float16)
            ls = T.alloc_local((FL,), T.float32)
            st = T.alloc_local((4,), T.float32)
            hrow[0, 0] = T.float32(0)     # pins the scope at kernel level (question 10)
            xrow[0, 0] = T.float16(0)
            mrow[0, 0] = T.float16(0)
            ls[0] = T.float32(0)
            st[0] = T.float32(0)
            norm_tokens(Hio, InNorm, Xs, nw, hrow, xrow, ls, st, worker)
            T.sync_grid()
            in_proj(Xs, WinQ, WinS, Ys, tile, shave)
            ba_phase(Xs, Wba, BAs, tile, shave)
            T.sync_grid()
            conv_phase(Ys, BAs, ConvW, ConvState, ALog, DtBias, Nt, QKs, Vs, Gbs, Zs, CsOut, tile, shave)
            T.sync_grid()
            scan_phase(QKs, Vs, Gbs, Zs, GNorm, Nt, Ssm, Os, tile, shave)
            T.sync_grid()
            out_proj(Os, WoutQ, WoutS, Mix, tile, shave)
            T.sync_grid()
            # --- residual: the tokens dealt over the workers ---------------------------------------------
            for it in T.serial(NP // workers):
                t = it * workers + worker
                T.copy(Hio[t, 0], hrow)
                T.copy(Mix[t, 0], mrow)
                for d0 in T.serial(H // FL):
                    for v in T.vectorized(FL):
                        hrow[0, d0 * FL + v] = hrow[0, d0 * FL + v] + T.cast(mrow[0, d0 * FL + v], T.float32)
                T.copy(hrow, Hio[t, 0])

    return gdn_layer_prefill


def gdn_layer_prefill_reference(h, in_norm, win_int, win_scales, conv_w, conv_state, a_log, dt_bias, gnorm,
                                wout_int, wout_scales, ssm, n: int, KH: int, VH: int, DK: int, DV: int, G: int,
                                eps: float = 1e-6, wba=None):
    """numpy oracle of make_gdn_layer_prefill (fp32), built from the phase oracles so the layer's claim is that
    they compose: norm -> in_proj -> conv -> scan -> out_proj -> residual. Returns (h after the layer, conv state,
    ssm). `wba` [2 VH, H]: the gate projections the checkpoint keeps outside the quantised in_proj."""
    import numpy as np
    hf = h.astype(np.float32)
    x = hf / np.sqrt((hf * hf).mean(-1, keepdims=True) + eps) * (1.0 + in_norm.astype(np.float32).reshape(1, -1))
    y = proj_prefill_reference(x.astype(np.float16), win_int, win_scales, G)
    ba = None if wba is None else (x.astype(np.float16).astype(np.float32) @ wba.astype(np.float32).T)
    qk, v, gb, z, new_cs = gdn_conv_prefill_reference(y.astype(np.float16), conv_w, conv_state, a_log, dt_bias,
                                                      n, KH, VH, DK, DV, ba=ba)
    o, new_ssm = gdn_scan_prefill_reference(qk, v, gb, z, gnorm, ssm, n)
    mix = proj_prefill_reference(o.astype(np.float16), wout_int, wout_scales, G)
    out = hf.copy()
    out[:n] += mix[:n]
    return out, new_cs, new_ssm


def make_gdn_moe_layer_prefill(N: int = 128, H: int = 2048, KH: int = 16, VH: int = 32, DK: int = 128, DV: int = 128,
                               I: int = 512, E: int = 256, TOPK: int = 8, G: int = 128, TB: int = 32, C: int = 64,
                               BN1: int = 16, BN2: int = 16, MB: int = 16, BM: int = 64, BNR: int = 16,
                               MBe: int = 16, BNG: int = 64, BND: int = 64, tiles: int = 3, shaves: int = 2,
                               eps: float = 1e-6):
    """P4 step 5: a WHOLE Gated DeltaNet PREFILL decoder layer in ONE launch -- `make_gdn_layer_prefill`'s mixer
    half followed by the second norm, P2a's routing and grouping, P2b's grouped expert FFN, and the weighted sum
    back into the fp32 residual stream.

    Nine phases, each dealing its work differently -- projections by output-row chunk, conv by head, scan by
    v-head, router by token, experts by expert, the combine by token -- so every boundary is a grid barrier.

    Oracle: `gdn_moe_layer_prefill_reference`, built from the phase oracles, because what the layer claims is that
    they compose."""
    conv_phase, _ = make_gdn_conv_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB, tiles=tiles, shaves=shaves,
                                          as_block=True)
    scan_phase, _ = make_gdn_scan_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, C=C, tiles=tiles, shaves=shaves, eps=eps,
                                          as_block=True)
    QK, VC = KH * DK, VH * DV
    CONV = 2 * QK + VC
    NIN = CONV + VC
    step = TB * C // math.gcd(TB, C)
    NP = -(-N // step) * step
    NIN_pad = -(-NIN // BN1) * BN1
    in_proj, _ = make_proj_prefill(NP=NP, M=NIN_pad, K=H, BN=BN1, MB=MB, G=G, tiles=tiles, shaves=shaves,
                                   as_block=True, scales_gm=True)
    out_proj, _ = make_proj_prefill(NP=NP, M=H, K=VC, BN=BN2, MB=MB, G=G, tiles=tiles, shaves=shaves, as_block=True,
                                    scales_gm=True)
    route_phase, rs = make_moe_route_prefill(N=NP, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=tiles, shaves=shaves,
                                             as_block=True)
    expert_phase, es = make_moe_expert_prefill(N=NP, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=MBe, BNG=BNG, BND=BND,
                                               tiles=tiles, shaves=shaves, as_block=True)
    RR, SLOTS = rs["RR"], rs["SLOTS"]
    KGH, KGI = H // G, I // G
    workers = tiles * shaves
    FL = 16
    assert H % FL == 0 and NP % workers == 0
    norm_tokens = prefill_norm_block(H, NP, workers, eps, FL)
    ba_phase = prefill_ba_block(H, VH, NP, MB, tiles, shaves)

    @T.prim_func
    def gdn_moe_layer(Hio: T.Tensor((NP, H), T.float32), InNorm: T.Tensor((1, H), T.float16),
                      PostNorm: T.Tensor((1, H), T.float16),
                      WinQ: T.Tensor((NIN_pad, H // 2), T.uint8), WinS: T.Tensor((H // G, NIN_pad), T.float16),
                      Wba: T.Tensor((2 * VH, H), T.float16),
                      ConvW: T.Tensor((4, CONV), T.float16), ConvState: T.Tensor((3, CONV), T.float16),
                      ALog: T.Tensor((1, VH), T.float32), DtBias: T.Tensor((1, VH), T.float32),
                      GNorm: T.Tensor((1, DV), T.float16), Nt: T.Tensor((1, 1), T.int32),
                      Ssm: T.Tensor((VH, DK, DV), T.float32),
                      WoutQ: T.Tensor((H, VC // 2), T.uint8), WoutS: T.Tensor((VC // G, H), T.float16),
                      Wr: T.Tensor((RR, H), T.float16),
                      WguQ: T.Tensor(((E + 1) * 2 * I, H // 2), T.uint8),
                      WguS: T.Tensor((KGH, (E + 1) * 2 * I), T.float16),
                      WdQ: T.Tensor(((E + 1) * H, I // 2), T.uint8), WdS: T.Tensor((KGI, (E + 1) * H), T.float16),
                      Xs: T.Tensor((NP, H), T.float16), Ys: T.Tensor((NP, NIN_pad), T.float16),
                      BAs: T.Tensor((NP, 2 * VH), T.float16),
                      QKs: T.Tensor((KH, NP, 2 * DK), T.float32), Vs: T.Tensor((VH, NP, DV), T.float16),
                      Gbs: T.Tensor((VH, NP, 2), T.float32), Zs: T.Tensor((VH, NP, DV), T.float16),
                      Os: T.Tensor((NP, VC), T.float16), Mix: T.Tensor((NP, H), T.float16),
                      CsOut: T.Tensor((3, CONV), T.float16),
                      Logits: T.Tensor((NP, RR), T.float16), Ids: T.Tensor((NP, TOPK), T.int32),
                      Wts: T.Tensor((NP, 16), T.float32), Cnt: T.Tensor((E + 1,), T.int32),
                      Off: T.Tensor((E + 2,), T.int32), Slot: T.Tensor((SLOTS,), T.int32),
                      PartE: T.Tensor((tiles, NP, H), T.float32), MoeOut: T.Tensor((NP, H), T.float32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            nw = T.alloc_shared((1, H), T.float16)
            hrow = T.alloc_local((1, H), T.float32)
            xrow = T.alloc_local((1, H), T.float16)
            mrow = T.alloc_local((1, H), T.float16)
            erow = T.alloc_local((1, H), T.float32)
            ls = T.alloc_local((FL,), T.float32)
            st = T.alloc_local((4,), T.float32)
            hrow[0, 0] = T.float32(0)     # pins the scope at kernel level (question 10)
            xrow[0, 0] = T.float16(0)
            mrow[0, 0] = T.float16(0)
            erow[0, 0] = T.float32(0)
            ls[0] = T.float32(0)
            st[0] = T.float32(0)

            norm_tokens(Hio, InNorm, Xs, nw, hrow, xrow, ls, st, worker)
            T.sync_grid()
            in_proj(Xs, WinQ, WinS, Ys, tile, shave)
            ba_phase(Xs, Wba, BAs, tile, shave)
            T.sync_grid()
            conv_phase(Ys, BAs, ConvW, ConvState, ALog, DtBias, Nt, QKs, Vs, Gbs, Zs, CsOut, tile, shave)
            T.sync_grid()
            scan_phase(QKs, Vs, Gbs, Zs, GNorm, Nt, Ssm, Os, tile, shave)
            T.sync_grid()
            out_proj(Os, WoutQ, WoutS, Mix, tile, shave)
            T.sync_grid()
            for it in T.serial(NP // workers):     # h += mix
                t = it * workers + worker
                T.copy(Hio[t, 0], hrow)
                T.copy(Mix[t, 0], mrow)
                for d0 in T.serial(H // FL):
                    for v in T.vectorized(FL):
                        hrow[0, d0 * FL + v] = hrow[0, d0 * FL + v] + T.cast(mrow[0, d0 * FL + v], T.float32)
                T.copy(hrow, Hio[t, 0])
            T.sync_grid()
            norm_tokens(Hio, PostNorm, Xs, nw, hrow, xrow, ls, st, worker)
            T.sync_grid()
            route_phase(Xs, Wr, Logits, Ids, Wts, Cnt, Off, Slot, tile, shave)
            T.sync_grid()      # worker 0's grouping is read back by everyone (derived at top level; written here)
            expert_phase(Xs, WguQ, WguS, WdQ, WdS, Wts, Off, Slot, PartE, MoeOut, tile, shave)
            T.sync_grid()
            for it in T.serial(NP // workers):     # h += the MoE output
                t = it * workers + worker
                T.copy(Hio[t, 0], hrow)
                T.copy(MoeOut[t, 0], erow)
                for d0 in T.serial(H // FL):
                    for v in T.vectorized(FL):
                        hrow[0, d0 * FL + v] = hrow[0, d0 * FL + v] + erow[0, d0 * FL + v]
                T.copy(hrow, Hio[t, 0])

    return gdn_moe_layer


def gdn_moe_layer_prefill_reference(h, in_norm, post_norm, win_int, win_scales, conv_w, conv_state, a_log, dt_bias,
                                    gnorm, wout_int, wout_scales, ssm, router, gu_int, gu_scales, d_int, d_scales,
                                    n: int, KH: int, VH: int, DK: int, DV: int, E: int, TOPK: int, G: int,
                                    eps: float = 1e-6, wba=None):
    """numpy oracle of make_gdn_moe_layer_prefill (fp32), built from the phase oracles: the mixer half, then the
    post norm, P2a's routing and P2b's grouped expert FFN, summed back into the stream."""
    import numpy as np
    out, new_cs, new_ssm = gdn_layer_prefill_reference(h, in_norm, win_int, win_scales, conv_w, conv_state, a_log,
                                                       dt_bias, gnorm, wout_int, wout_scales, ssm, n, KH, VH, DK,
                                                       DV, G, eps, wba=wba)
    x = out / np.sqrt((out * out).mean(-1, keepdims=True) + eps) * (1.0 + post_norm.astype(np.float32).reshape(1, -1))
    x16 = x.astype(np.float16)
    r = moe_route_prefill_reference(x16, router, E=E, TOPK=TOPK)
    moe = moe_expert_prefill_reference(x16, gu_int, gu_scales, d_int, d_scales, r["wts"], r["off"], r["slot"],
                                       TOPK=TOPK, G=G)
    out = out + moe
    return out, new_cs, new_ssm


def make_attn_prefill(N: int = 64, NKV: int = 2, GQ: int = 8, D: int = 256, SMAX: int = 1536, TB: int = 4,
                      BKV: int = 128, tiles: int = 3, shaves: int = 2, pipelined: bool = False,
                      as_block: bool = False, kv4: bool = False):
    """P1: the full-attention PREFILL phase -- flash attention over `N` query tokens already normed, roped and
    appended to the cache, with the three things the decode phase does not need: **GQA**, the **causal mask** and
    the **output gate**.

      O[t, (g * GQ + hh) * D ...] = softmax_j(q[g, t, hh] . Kc[g, j] / sqrt(D))[j <= pos0 + t] . Vt[g, :, j]^T
                                    * sigmoid(Gate[t, (g * GQ + hh) * D ...])

    Work items are (kv head, token block of TB tokens) dealt round-robin over the tiles x SHAVEs; one worker owns a
    block end to end, so there is no cross-worker merge (the decode phase splits the positions instead, because at
    one token there is nothing else to split). The DPU operand is the block's `TB x GQ` query rows: Q is laid out
    `[NKV, N, GQ, D]` so one block is one contiguous DMA. Only the chunks a block can see are read (`ck` below), and
    only the chunk that straddles the diagonal is masked.

    The gate and the output are **token-major** `[N, NKV * GQ * D]` while the query is head-major, because the
    o_proj that reads the output is an ordinary `[N, KO] x [H, KO]^T` gemm. A block's `TB x GQ` rows are then the
    `[TB, GQ * D]` sub-block at column `g * GQ * D`, so the gate is one 2-D DMA in and the gated output one 2-D DMA
    out of the same CMX buffer. Only the query has a reason to stay head-major: it is a DPU operand.

    `Pos[0, 0]` is the absolute position of token 0 of this call, so a prompt is prefilled in chunks of `N` tokens.
    `kv4`: Kc / Vt carry a leading layer axis indexed by `KvSlot[kvi, 1]`, one cache per attention layer of the
    launch, as `attn_block` does for decode -- the model kernel's form. Oracle: `attn_prefill_reference`."""
    assert D % 32 == 0 and BKV % 32 == 0 and SMAX % BKV == 0 and N % TB == 0
    ROWS = TB * GQ
    KO = NKV * GQ * D
    # One `T.gemm` is one DPU workload. Only each DIMENSION is bounded (8192, dpu.md 4.6) -- the output tile's
    # M x N product is not, which device/run_dpu_points_probe.py established after an earlier reading of dpu.md 13
    # had capped TB at 4 here for nothing.
    assert ROWS <= 8192 and D <= 8192 and BKV <= 8192, f"query block {ROWS} x {D} / score block {ROWS} x {BKV}"
    HL, FL = 32, 16
    assert BKV % HL == 0 and D % FL == 0
    scale = 1.0 / (D ** 0.5)
    workers = tiles * shaves
    NIT = NKV * (N // TB)
    CMAX = SMAX // BKV

    @T.macro
    def flash_phase(Q, Kc, Vt, Gate, Pos, O, tile, shave, KvSlot, kvi):
        worker = tile * shaves + shave
        pos_cmx = T.alloc_shared((1, 1), T.int32)
        q_cmx = T.alloc_local((ROWS, D), T.float16)
        k_cmx = T.alloc_local((BKV, D), T.float16)
        v_cmx = T.alloc_local((D, BKV), T.float16)
        s_cmx = T.alloc_local((ROWS, BKV), T.float16)
        pv_cmx = T.alloc_local((ROWS, D), T.float16)
        o_acc = T.alloc_local((ROWS, D), T.float32)
        gt_cmx = T.alloc_local((TB, GQ * D), T.float16)   # the gate in, the gated output out: O's token-major shape
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
                    if kv4:
                        T.copy(Kc[KvSlot[kvi, 1], g, kv * BKV, 0], k_cmx)
                        T.copy(Vt[KvSlot[kvi, 1], g, 0, kv * BKV], v_cmx)
                    else:
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
                T.copy(Gate[t0, g * GQ * D], gt_cmx)
                for r, hh in T.grid(TB, GQ):
                    corr[r * GQ + hh] = T.float32(1) / l_i[r * GQ + hh]
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            gh[v] = T.exp(T.float16(0) - gt_cmx[r, hh * D + d0 * FL + v])
                            gt_cmx[r, hh * D + d0 * FL + v] = T.cast(
                                o_acc[r * GQ + hh, d0 * FL + v] * corr[r * GQ + hh]
                                / (T.float32(1) + T.cast(gh[v], T.float32)), T.float16)
                T.copy(gt_cmx, O[t0, g * GQ * D])

    if as_block:
        return flash_phase, dict(KO=KO, ROWS=ROWS, NIT=NIT, CMAX=CMAX)

    @T.prim_func
    def attn_prefill(Q: T.Tensor((NKV, N, GQ, D), T.float16), Kc: T.Tensor((NKV, SMAX, D), T.float16),
                     Vt: T.Tensor((NKV, D, SMAX), T.float16), Gate: T.Tensor((N, KO), T.float16),
                     Pos: T.Tensor((1, 1), T.int32), O: T.Tensor((N, KO), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            flash_phase(Q, Kc, Vt, Gate, Pos, O, tile, T.get_thread_binding(), Pos, 0)

    return attn_prefill


def attn_prefill_reference(q, kc, vt, gate, pos0: int, eps: float = 0.0):
    """numpy oracle of make_attn_prefill (fp32). q is [NKV, N, GQ, D], kc [NKV, SMAX, D], vt [NKV, D, SMAX]; the
    gate and the output are token-major [N, NKV * GQ * D] as the o_proj needs them. Query token t of the call is at
    absolute position pos0 + t and sees keys 0 .. pos0 + t."""
    import numpy as np
    NKV, N, GQ, D = q.shape
    out = np.zeros((N, NKV * GQ * D), np.float32)
    for g in range(NKV):
        for t in range(N):
            p = pos0 + t
            k = kc[g, :p + 1].astype(np.float32)        # [p + 1, D]
            v = vt[g, :, :p + 1].astype(np.float32)     # [D, p + 1]
            for hh in range(GQ):
                s = k @ q[g, t, hh].astype(np.float32) / np.sqrt(D)
                pr = np.exp(s - s.max())
                pr /= pr.sum()
                h = g * GQ + hh
                gt = gate[t, h * D:(h + 1) * D].astype(np.float32)
                out[t, h * D:(h + 1) * D] = (v @ pr) / (1.0 + np.exp(-gt))
    return out


def make_attn_qkv_prefill(N: int = 64, H: int = 2048, NH: int = 16, NKV: int = 2, D: int = 256, ROT: int = 64,
                          SMAX: int = 1536, TB: int = 8, tiles: int = 3, shaves: int = 2, eps: float = 1e-6,
                          as_block: bool = False, kv4: bool = False):
    """P1b: the attention layer's PREFILL phase between the qkv projection and the flash phase -- what
    `attn_block`'s stage B does for ONE token, over `N` of them.

      q_h  = rope(rmsnorm_zc(Y[t, h*2D : h*2D+D], qnorm), pos0 + t)      -> Q[h // GQ, t, h % GQ, :]
      gate = Y[t, h*2D+D : h*2D+2D]                                      -> Gate[t, h*D : h*D+D]
      k_g  = rope(rmsnorm_zc(Y[t, KOFF + g*D ...], knorm), pos0 + t)     -> Kc[g, pos0 + t, :]
      v_g  = Y[t, VOFF + g*D ...]                                        -> Vt[g, :, pos0 + t]

    `make_attn_prefill` reads Kc / Vt and never writes them, so without this phase nothing fills the cache and
    prefill cannot feed decode. The work item is a (token block, kv group) pair, which is what makes the stores
    cheap: one kv group's GQ query heads are adjacent in Y AND adjacent in Q, so a block is one [TB, GQ*2D] read
    and two contiguous writes -- the queries head-major [TB*GQ, D] for the DPU, the gates token-major [TB, GQ*D]
    for the flash phase's gate read -- and the keys of TB tokens are one contiguous run of Kc. Only V needs a
    transpose, into one [D, TB] store instead of TB stores of [D, 1] -- a 256-row DMA of 2 bytes per row is the
    shape the per-job cost punishes hardest.

    `Nt[0, 0]` is the live token count: a row past it is written as ZERO, because `make_attn_prefill` relies on
    unseen cache positions holding zeros rather than masking them. `kv4` is `make_attn_prefill`'s: the caches carry
    a leading layer axis indexed by `KvSlot[kvi, 1]`. Oracle: `attn_qkv_prefill_reference`."""
    GQ = NH // NKV
    NQKV = NH * 2 * D + 2 * NKV * D
    KOFF, VOFF = NH * 2 * D, NH * 2 * D + NKV * D
    NP = -(-N // TB) * TB
    NB = NP // TB
    workers = tiles * shaves
    FL, HL = 16, 32
    HR = ROT // 2
    assert NH % NKV == 0 and D % HL == 0 and ROT % 2 == 0 and ROT <= D

    @T.macro
    def qkv_phase(Ys, QNormW, KNormW, CosSin, Pos, Nt, Q, Gate, Kc, Vt, tile, shave, KvSlot, kvi):
        """The phase as a macro so the attention layer can call it; `make_attn_qkv_prefill` is the wrapper."""
        worker = tile * shaves + shave
        qw = T.alloc_shared((1, D), T.float16)
        kw = T.alloc_shared((1, D), T.float16)
        pos = T.alloc_shared((1, 1), T.int32)
        nt = T.alloc_shared((1, 1), T.int32)
        qg = T.alloc_local((TB, GQ * 2 * D), T.float16)   # one kv group's q and gate for TB tokens: one DMA
        cs = T.alloc_local((TB, 2 * ROT), T.float32)
        qo = T.alloc_local((TB * GQ, D), T.float16)
        go = T.alloc_local((TB, GQ * D), T.float16)   # token-major, the layout the flash phase's gate read wants
        kb = T.alloc_local((TB, D), T.float16)
        vb = T.alloc_local((TB, D), T.float16)
        vt = T.alloc_local((D, TB), T.float16)
        rowf = T.alloc_local((D,), T.float32)
        ls = T.alloc_local((FL,), T.float32)
        st = T.alloc_local((8,), T.float32)
        T.copy(QNormW[0, 0], qw)
        T.copy(KNormW[0, 0], kw)
        T.copy(Pos[0, 0], pos)
        T.copy(Nt[0, 0], nt)
        T.sync_threads()
        for it in T.serial(T.ceildiv(NB * NKV, workers)):
            item = it * workers + worker
            if item < NB * NKV:
                b = item // NKV
                g = item % NKV
                T.copy(CosSin[b * TB, 0], cs)
                # --- the group's GQ query heads and their gates ------------------------------------------
                T.copy(Ys[b * TB, (g * GQ) * 2 * D], qg)
                for r, hh in T.grid(TB, GQ):
                    st[3] = T.if_then_else(b * TB + r < nt[0, 0], T.float32(1), T.float32(0))
                    for v in T.vectorized(FL):
                        ls[v] = T.float32(0)
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rowf[d0 * FL + v] = T.cast(qg[r, hh * 2 * D + d0 * FL + v], T.float32)
                            ls[v] = ls[v] + rowf[d0 * FL + v] * rowf[d0 * FL + v]
                    st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(D)
                                                  + T.float32(eps))
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rowf[d0 * FL + v] = (rowf[d0 * FL + v] * st[0]
                                                 * (T.float32(1) + T.cast(qw[0, d0 * FL + v], T.float32)))
                    for rr in T.serial(HR):      # rotate-half on the first ROT dims
                        st[1] = rowf[rr]
                        st[2] = rowf[HR + rr]
                        rowf[rr] = st[1] * cs[r, rr] - st[2] * cs[r, ROT + rr]
                        rowf[HR + rr] = st[2] * cs[r, HR + rr] + st[1] * cs[r, ROT + HR + rr]
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            qo[r * GQ + hh, d0 * FL + v] = T.cast(rowf[d0 * FL + v] * st[3], T.float16)
                            go[r, hh * D + d0 * FL + v] = T.cast(
                                T.cast(qg[r, hh * 2 * D + D + d0 * FL + v], T.float32) * st[3], T.float16)
                T.copy(qo, Q[g, b * TB, 0, 0])
                T.copy(go, Gate[b * TB, g * GQ * D])
                # --- the group's key: normed, roped, appended to the cache -------------------------------
                T.copy(Ys[b * TB, KOFF + g * D], kb)
                for r in T.serial(TB):
                    st[3] = T.if_then_else(b * TB + r < nt[0, 0], T.float32(1), T.float32(0))
                    for v in T.vectorized(FL):
                        ls[v] = T.float32(0)
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rowf[d0 * FL + v] = T.cast(kb[r, d0 * FL + v], T.float32)
                            ls[v] = ls[v] + rowf[d0 * FL + v] * rowf[d0 * FL + v]
                    st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(D)
                                                  + T.float32(eps))
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rowf[d0 * FL + v] = (rowf[d0 * FL + v] * st[0]
                                                 * (T.float32(1) + T.cast(kw[0, d0 * FL + v], T.float32)))
                    for rr in T.serial(HR):
                        st[1] = rowf[rr]
                        st[2] = rowf[HR + rr]
                        rowf[rr] = st[1] * cs[r, rr] - st[2] * cs[r, ROT + rr]
                        rowf[HR + rr] = st[2] * cs[r, HR + rr] + st[1] * cs[r, ROT + HR + rr]
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            kb[r, d0 * FL + v] = T.cast(rowf[d0 * FL + v] * st[3], T.float16)
                if kv4:
                    T.copy(kb, Kc[KvSlot[kvi, 1], g, pos[0, 0] + b * TB, 0])
                else:
                    T.copy(kb, Kc[g, pos[0, 0] + b * TB, 0])
                # --- the group's value, transposed once per block into the cache's column layout ---------
                T.copy(Ys[b * TB, VOFF + g * D], vb)
                for r in T.serial(TB):
                    st[3] = T.if_then_else(b * TB + r < nt[0, 0], T.float32(1), T.float32(0))
                    for d in T.serial(D):
                        vt[d, r] = T.cast(T.cast(vb[r, d], T.float32) * st[3], T.float16)
                if kv4:
                    T.copy(vt, Vt[KvSlot[kvi, 1], g, 0, pos[0, 0] + b * TB])
                else:
                    T.copy(vt, Vt[g, 0, pos[0, 0] + b * TB])

    if as_block:
        return qkv_phase, dict(NP=NP, NB=NB, NQKV=NQKV, GQ=GQ, KOFF=KOFF, VOFF=VOFF)

    @T.prim_func
    def attn_qkv(Ys: T.Tensor((NP, NQKV), T.float16), QNormW: T.Tensor((1, D), T.float16),
                 KNormW: T.Tensor((1, D), T.float16), CosSin: T.Tensor((NP, 2 * ROT), T.float32),
                 Pos: T.Tensor((1, 1), T.int32), Nt: T.Tensor((1, 1), T.int32),
                 Q: T.Tensor((NKV, NP, GQ, D), T.float16), Gate: T.Tensor((NP, NH * D), T.float16),
                 Kc: T.Tensor((NKV, SMAX, D), T.float16), Vt: T.Tensor((NKV, D, SMAX), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            qkv_phase(Ys, QNormW, KNormW, CosSin, Pos, Nt, Q, Gate, Kc, Vt, tile, T.get_thread_binding(), Pos, 0)

    return attn_qkv


def attn_qkv_prefill_reference(y, qnorm_w, knorm_w, cos_sin, pos0: int, n: int, NH: int, NKV: int, D: int,
                               ROT: int, kc, vt, eps: float = 1e-6):
    """numpy oracle of make_attn_qkv_prefill (fp32). `y` [NP, NQKV] is the projection output, `cos_sin` [NP, 2*ROT]
    (cos then sin per token, the layout the kernel reads a whole block of in one copy); returns (Q, Gate, kc, vt)
    with the caches updated in place for positions pos0 .. pos0 + n - 1. Q is head-major [NKV, NP, GQ, D] (a DPU
    operand), Gate token-major [NP, NH*D] (what the flash phase's gate read and the o_proj want). The norm and the
    rotate-half are `attn_reference`'s, so a prefill that disagrees with decode on one token shows up here."""
    import numpy as np
    NP = y.shape[0]
    GQ = NH // NKV
    KOFF, VOFF = NH * 2 * D, NH * 2 * D + NKV * D
    HR = ROT // 2
    Q = np.zeros((NKV, NP, GQ, D), np.float16)
    Gate = np.zeros((NP, NH * D), np.float16)
    kc, vt = kc.copy(), vt.copy()

    def norm_rope(v, wn, cs):
        v = v * (1.0 / np.sqrt((v * v).mean() + eps)) * (1.0 + wn.astype(np.float32).reshape(-1))
        r = v[:ROT].copy()
        out = v.copy()
        out[:HR] = r[:HR] * cs[0, :HR] - r[HR:] * cs[1, :HR]
        out[HR:ROT] = r[HR:] * cs[0, HR:] + r[:HR] * cs[1, HR:]
        return out

    for t in range(n):
        cs = cos_sin[t].astype(np.float32).reshape(2, ROT)
        for h in range(NH):
            q = norm_rope(y[t, h * 2 * D:h * 2 * D + D].astype(np.float32), qnorm_w, cs)
            Q[h // GQ, t, h % GQ] = q.astype(np.float16)
            Gate[t, h * D:(h + 1) * D] = y[t, h * 2 * D + D:h * 2 * D + 2 * D]
        for g in range(NKV):
            kc[g, pos0 + t] = norm_rope(y[t, KOFF + g * D:KOFF + (g + 1) * D].astype(np.float32),
                                        knorm_w, cs).astype(np.float16)
            vt[g, :, pos0 + t] = y[t, VOFF + g * D:VOFF + (g + 1) * D]
    return Q, Gate, kc, vt


def make_attn_layer_prefill(N: int = 64, H: int = 2048, NH: int = 16, NKV: int = 2, D: int = 256, ROT: int = 64,
                            SMAX: int = 1536, G: int = 128, TB: int = 8, TBF: int = 4, BKV: int = 128,
                            BN1: int = 64, BN2: int = 64, MB: int = 32, tiles: int = 3, shaves: int = 2,
                            eps: float = 1e-6):
    """P1 step 2: a whole full-attention PREFILL mixer layer in ONE launch, over `N` prompt tokens -- the
    attention counterpart of `make_gdn_layer_prefill`:

      x = rmsnorm_zc(h, in_norm) -> qkv projection (P4 step 4's SHAVE-dequant projection) -> P1b (per-head norms,
      RoPE, the cache append, the flash phase's query / gate layout) -> P1's flash attention -> o_proj -> h += mix

    `Kc` / `Vt` are kernel PARAMETERS, not addresses read from a layer table: that is the one cell of plan
    question 8's table which does not lose the device, and the append is exactly its "parameter base + runtime
    offset" shape.

    Tokens past `Nt[0, 0]` are skipped by the residual add. The GDN layer needs no such guard because a dead token
    flows through its phases as a zero; through attention it does not -- a zero query gives a UNIFORM softmax over
    the keys it may see, so its mix is the mean of V, not zero.

    Every phase deals its work differently (projections by output-row chunk, P1b by (token block, kv group), the
    flash phase by (kv head, token block), the norm and the residual by token), so every boundary is a grid
    barrier. Oracle: `attn_layer_prefill_reference`."""
    GQ = NH // NKV
    KO = NH * D
    workers = tiles * shaves
    step = TB * TBF // math.gcd(TB, TBF)
    step = step * workers // math.gcd(step, workers)
    NP = -(-N // step) * step
    qkv_phase, qs = make_attn_qkv_prefill(N=NP, H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, TB=TB, tiles=tiles,
                                          shaves=shaves, eps=eps, as_block=True)
    flash_phase, fs = make_attn_prefill(N=NP, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, TB=TBF, BKV=BKV, tiles=tiles,
                                        shaves=shaves, as_block=True)
    NQKV = qs["NQKV"]
    NQKV_pad = -(-NQKV // BN1) * BN1
    qkv_proj, _ = make_proj_prefill(NP=NP, M=NQKV_pad, K=H, BN=BN1, MB=MB, G=G, tiles=tiles, shaves=shaves,
                                    as_block=True, scales_gm=True)
    o_proj, _ = make_proj_prefill(NP=NP, M=H, K=KO, BN=BN2, MB=MB, G=G, tiles=tiles, shaves=shaves, as_block=True,
                                  scales_gm=True)
    assert fs["KO"] == KO
    FL = 16
    assert H % FL == 0 and NP % workers == 0
    norm_tokens = prefill_norm_block(H, NP, workers, eps, FL)

    @T.prim_func
    def attn_layer_prefill(Hio: T.Tensor((NP, H), T.float32), InNorm: T.Tensor((1, H), T.float16),
                           WqkvQ: T.Tensor((NQKV_pad, H // 2), T.uint8),
                           WqkvS: T.Tensor((H // G, NQKV_pad), T.float16),
                           QNormW: T.Tensor((1, D), T.float16), KNormW: T.Tensor((1, D), T.float16),
                           CosSin: T.Tensor((NP, 2 * ROT), T.float32), Pos: T.Tensor((1, 1), T.int32),
                           Nt: T.Tensor((1, 1), T.int32), Kc: T.Tensor((NKV, SMAX, D), T.float16),
                           Vt: T.Tensor((NKV, D, SMAX), T.float16),
                           WoQ: T.Tensor((H, KO // 2), T.uint8), WoS: T.Tensor((KO // G, H), T.float16),
                           Xs: T.Tensor((NP, H), T.float16), Ys: T.Tensor((NP, NQKV_pad), T.float16),
                           Q: T.Tensor((NKV, NP, GQ, D), T.float16), Gate: T.Tensor((NP, KO), T.float16),
                           Os: T.Tensor((NP, KO), T.float16), Mix: T.Tensor((NP, H), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            nw = T.alloc_shared((1, H), T.float16)
            nt = T.alloc_shared((1, 1), T.int32)
            hrow = T.alloc_local((1, H), T.float32)
            xrow = T.alloc_local((1, H), T.float16)
            mrow = T.alloc_local((1, H), T.float16)
            ls = T.alloc_local((FL,), T.float32)
            st = T.alloc_local((4,), T.float32)
            hrow[0, 0] = T.float32(0)     # pins the scope at kernel level (question 10)
            xrow[0, 0] = T.float16(0)
            mrow[0, 0] = T.float16(0)
            ls[0] = T.float32(0)
            st[0] = T.float32(0)
            T.copy(Nt[0, 0], nt)

            norm_tokens(Hio, InNorm, Xs, nw, hrow, xrow, ls, st, worker)
            T.sync_grid()
            qkv_proj(Xs, WqkvQ, WqkvS, Ys, tile, shave)
            T.sync_grid()
            qkv_phase(Ys, QNormW, KNormW, CosSin, Pos, Nt, Q, Gate, Kc, Vt, tile, shave, Pos, 0)
            T.sync_grid()
            flash_phase(Q, Kc, Vt, Gate, Pos, Os, tile, shave, Pos, 0)
            T.sync_grid()
            o_proj(Os, WoQ, WoS, Mix, tile, shave)
            T.sync_grid()
            for it in T.serial(NP // workers):     # h += mix, live tokens only
                t = it * workers + worker
                if t < nt[0, 0]:
                    T.copy(Hio[t, 0], hrow)
                    T.copy(Mix[t, 0], mrow)
                    for d0 in T.serial(H // FL):
                        for v in T.vectorized(FL):
                            hrow[0, d0 * FL + v] = hrow[0, d0 * FL + v] + T.cast(mrow[0, d0 * FL + v], T.float32)
                    T.copy(hrow, Hio[t, 0])

    return attn_layer_prefill


def attn_layer_prefill_reference(h, in_norm, wqkv_int, wqkv_scales, qnorm_w, knorm_w, cos_sin, wo_int, wo_scales,
                                 kc, vt, pos0: int, n: int, NH: int, NKV: int, D: int, ROT: int, G: int,
                                 eps: float = 1e-6):
    """numpy oracle of make_attn_layer_prefill (fp32), built from the phase oracles because what the layer claims
    is that they compose. Returns (h after the layer, the K cache, the transposed V cache)."""
    import numpy as np
    hf = h.astype(np.float32)
    x = hf / np.sqrt((hf * hf).mean(-1, keepdims=True) + eps) * (1.0 + in_norm.astype(np.float32).reshape(1, -1))
    y = proj_prefill_reference(x.astype(np.float16), wqkv_int, wqkv_scales, G).astype(np.float16)
    q, gate, kc, vt = attn_qkv_prefill_reference(y, qnorm_w, knorm_w, cos_sin, pos0, n, NH, NKV, D, ROT, kc, vt,
                                                 eps)
    o = attn_prefill_reference(q, kc, vt, gate, pos0)
    mix = proj_prefill_reference(o.astype(np.float16), wo_int, wo_scales, G)
    out = hf.copy()
    out[:n] += mix[:n]
    return out, kc, vt


def make_attn_moe_layer_prefill(N: int = 64, H: int = 2048, NH: int = 16, NKV: int = 2, D: int = 256, ROT: int = 64,
                                SMAX: int = 1536, I: int = 512, E: int = 256, TOPK: int = 8, G: int = 128,
                                TB: int = 8, TBF: int = 4, BKV: int = 128, BN1: int = 16, BN2: int = 16,
                                MB: int = 16, BM: int = 64, BNR: int = 16, MBe: int = 16, BNG: int = 64,
                                BND: int = 64, tiles: int = 3, shaves: int = 2, eps: float = 1e-6):
    """P1 step 3: a WHOLE full-attention PREFILL decoder layer in ONE launch -- `make_attn_layer_prefill`'s mixer
    half followed by the second norm, P2a's routing and grouping, P2b's grouped expert FFN, and the weighted sum
    back into the fp32 residual stream. The attention counterpart of `make_gdn_moe_layer_prefill`, and the second
    of the two layer kinds the prefill model kernel runs.

    Ten phases, each dealing its work differently, so every boundary is a grid barrier. The dead-token guard of the
    mixer half applies to the MoE half too: past `Nt` the router sees a normed zero row and would route it.

    Oracle: `attn_moe_layer_prefill_reference`."""
    GQ = NH // NKV
    KO = NH * D
    workers = tiles * shaves
    step = TB * TBF // math.gcd(TB, TBF)
    step = step * workers // math.gcd(step, workers)
    step = step * BM // math.gcd(step, BM)
    NP = -(-N // step) * step
    qkv_phase, qs = make_attn_qkv_prefill(N=NP, H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, TB=TB, tiles=tiles,
                                          shaves=shaves, eps=eps, as_block=True)
    flash_phase, _ = make_attn_prefill(N=NP, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, TB=TBF, BKV=BKV, tiles=tiles,
                                       shaves=shaves, as_block=True)
    NQKV = qs["NQKV"]
    NQKV_pad = -(-NQKV // BN1) * BN1
    qkv_proj, _ = make_proj_prefill(NP=NP, M=NQKV_pad, K=H, BN=BN1, MB=MB, G=G, tiles=tiles, shaves=shaves,
                                    as_block=True, scales_gm=True)
    o_proj, _ = make_proj_prefill(NP=NP, M=H, K=KO, BN=BN2, MB=MB, G=G, tiles=tiles, shaves=shaves, as_block=True,
                                  scales_gm=True)
    route_phase, rs = make_moe_route_prefill(N=NP, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=tiles, shaves=shaves,
                                             as_block=True)
    expert_phase, _ = make_moe_expert_prefill(N=NP, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=MBe, BNG=BNG, BND=BND,
                                              tiles=tiles, shaves=shaves, as_block=True)
    RR, SLOTS = rs["RR"], rs["SLOTS"]
    KGH, KGI = H // G, I // G
    FL = 16
    assert H % FL == 0 and NP % workers == 0
    norm_tokens = prefill_norm_block(H, NP, workers, eps, FL)

    @T.prim_func
    def attn_moe_layer(Hio: T.Tensor((NP, H), T.float32), InNorm: T.Tensor((1, H), T.float16),
                       PostNorm: T.Tensor((1, H), T.float16),
                       WqkvQ: T.Tensor((NQKV_pad, H // 2), T.uint8), WqkvS: T.Tensor((H // G, NQKV_pad), T.float16),
                       QNormW: T.Tensor((1, D), T.float16), KNormW: T.Tensor((1, D), T.float16),
                       CosSin: T.Tensor((NP, 2 * ROT), T.float32), Pos: T.Tensor((1, 1), T.int32),
                       Nt: T.Tensor((1, 1), T.int32), Kc: T.Tensor((NKV, SMAX, D), T.float16),
                       Vt: T.Tensor((NKV, D, SMAX), T.float16),
                       WoQ: T.Tensor((H, KO // 2), T.uint8), WoS: T.Tensor((KO // G, H), T.float16),
                       Wr: T.Tensor((RR, H), T.float16),
                       WguQ: T.Tensor(((E + 1) * 2 * I, H // 2), T.uint8),
                       WguS: T.Tensor((KGH, (E + 1) * 2 * I), T.float16),
                       WdQ: T.Tensor(((E + 1) * H, I // 2), T.uint8), WdS: T.Tensor((KGI, (E + 1) * H), T.float16),
                       Xs: T.Tensor((NP, H), T.float16), Ys: T.Tensor((NP, NQKV_pad), T.float16),
                       Q: T.Tensor((NKV, NP, GQ, D), T.float16), Gate: T.Tensor((NP, KO), T.float16),
                       Os: T.Tensor((NP, KO), T.float16), Mix: T.Tensor((NP, H), T.float16),
                       Logits: T.Tensor((NP, RR), T.float16), Ids: T.Tensor((NP, TOPK), T.int32),
                       Wts: T.Tensor((NP, 16), T.float32), Cnt: T.Tensor((E + 1,), T.int32),
                       Off: T.Tensor((E + 2,), T.int32), Slot: T.Tensor((SLOTS,), T.int32),
                       PartE: T.Tensor((tiles, NP, H), T.float32), MoeOut: T.Tensor((NP, H), T.float32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            nw = T.alloc_shared((1, H), T.float16)
            nt = T.alloc_shared((1, 1), T.int32)
            hrow = T.alloc_local((1, H), T.float32)
            xrow = T.alloc_local((1, H), T.float16)
            mrow = T.alloc_local((1, H), T.float16)
            erow = T.alloc_local((1, H), T.float32)
            ls = T.alloc_local((FL,), T.float32)
            st = T.alloc_local((4,), T.float32)
            hrow[0, 0] = T.float32(0)     # pins the scope at kernel level (question 10)
            xrow[0, 0] = T.float16(0)
            mrow[0, 0] = T.float16(0)
            erow[0, 0] = T.float32(0)
            ls[0] = T.float32(0)
            st[0] = T.float32(0)
            T.copy(Nt[0, 0], nt)

            norm_tokens(Hio, InNorm, Xs, nw, hrow, xrow, ls, st, worker)
            T.sync_grid()
            qkv_proj(Xs, WqkvQ, WqkvS, Ys, tile, shave)
            T.sync_grid()
            qkv_phase(Ys, QNormW, KNormW, CosSin, Pos, Nt, Q, Gate, Kc, Vt, tile, shave, Pos, 0)
            T.sync_grid()
            flash_phase(Q, Kc, Vt, Gate, Pos, Os, tile, shave, Pos, 0)
            T.sync_grid()
            o_proj(Os, WoQ, WoS, Mix, tile, shave)
            T.sync_grid()
            for it in T.serial(NP // workers):     # h += mix, live tokens only
                t = it * workers + worker
                if t < nt[0, 0]:
                    T.copy(Hio[t, 0], hrow)
                    T.copy(Mix[t, 0], mrow)
                    for d0 in T.serial(H // FL):
                        for v in T.vectorized(FL):
                            hrow[0, d0 * FL + v] = hrow[0, d0 * FL + v] + T.cast(mrow[0, d0 * FL + v], T.float32)
                    T.copy(hrow, Hio[t, 0])
            T.sync_grid()
            norm_tokens(Hio, PostNorm, Xs, nw, hrow, xrow, ls, st, worker)
            T.sync_grid()
            route_phase(Xs, Wr, Logits, Ids, Wts, Cnt, Off, Slot, tile, shave)
            T.sync_grid()      # worker 0's grouping is read back by everyone (derived at top level; written here)
            expert_phase(Xs, WguQ, WguS, WdQ, WdS, Wts, Off, Slot, PartE, MoeOut, tile, shave)
            T.sync_grid()
            for it in T.serial(NP // workers):     # h += the MoE output, live tokens only
                t = it * workers + worker
                if t < nt[0, 0]:
                    T.copy(Hio[t, 0], hrow)
                    T.copy(MoeOut[t, 0], erow)
                    for d0 in T.serial(H // FL):
                        for v in T.vectorized(FL):
                            hrow[0, d0 * FL + v] = hrow[0, d0 * FL + v] + erow[0, d0 * FL + v]
                    T.copy(hrow, Hio[t, 0])

    return attn_moe_layer


def attn_moe_layer_prefill_reference(h, in_norm, post_norm, wqkv_int, wqkv_scales, qnorm_w, knorm_w, cos_sin,
                                     wo_int, wo_scales, kc, vt, router, gu_int, gu_scales, d_int, d_scales,
                                     pos0: int, n: int, NH: int, NKV: int, D: int, ROT: int, E: int, TOPK: int,
                                     G: int, eps: float = 1e-6):
    """numpy oracle of make_attn_moe_layer_prefill (fp32), built from the phase oracles: the mixer half, then the
    post norm, P2a's routing and P2b's grouped expert FFN, summed back into the stream."""
    import numpy as np
    out, kc, vt = attn_layer_prefill_reference(h, in_norm, wqkv_int, wqkv_scales, qnorm_w, knorm_w, cos_sin,
                                               wo_int, wo_scales, kc, vt, pos0, n, NH, NKV, D, ROT, G, eps)
    x = out / np.sqrt((out * out).mean(-1, keepdims=True) + eps) * (1.0 + post_norm.astype(np.float32).reshape(1, -1))
    x16 = x.astype(np.float16)
    r = moe_route_prefill_reference(x16, router, E=E, TOPK=TOPK)
    moe = moe_expert_prefill_reference(x16, gu_int, gu_scales, d_int, d_scales, r["wts"], r["off"], r["slot"],
                                       TOPK=TOPK, G=G)
    out[:n] += moe[:n]
    return out, kc, vt


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


def make_dma_page_probe(KH: int = 16, VH: int = 32, DK: int = 128, ROWS: int = 3, rounds: int = 2,
                        blocked: bool = False, item_worker: str = "tile", rw: bool = False,
                        tiles: int = 3, shaves: int = 2):
    """Plan question 11's store pattern with the computation removed -- and it does NOT reproduce the defect.

    `Dst[ROWS, CONV]` has the conv state's shape and a slot is the same 2-D job the real kernel issues: ROWS rows
    of `DK` halves at column `cb`, i.e. `DK * 2` bytes per row at stride `CONV * 2`. The addresses are the point --
    slots 0..KH-1 are `DK * 2` bytes apart, so **they all live in the first 4 KiB of a row** -- and `blocked`
    decides whether that page is written by one tile or interleaved by all of them. Each slot is written once per
    round with `slot + 1 + 100 * round`, so the host can tell a LOST store (an earlier round's value, or the -1 the
    buffer was pre-filled with) from a wrong one and say which round lost it; `rw` adds the load the real phase
    does before its store, so a page carries both directions from different tiles.

    **Result (2026-11-09, `device/run_dma_page_probe.py`): all six arms CLEAN** -- tiles 1/2/3 x
    {round-robin, blocked}, with and without `rw`. So "three tiles writing interleaved 256 B pieces of one 4 KiB
    DDR page, read-modify-write, over several rounds" is **not sufficient** to lose a store, and the hypothesis the
    probe was built to test is falsified. Something the probe leaves out is necessary; the model kernel's `--skip
    moe` arm says the MoE phases (about 0.4 GB of concurrent expert-weight DMA and the DPU work beside it) are part
    of it. Kept as the skeleton: the next version adds that load rather than re-deriving this one."""
    QK = KH * DK
    CONV = 2 * QK + VH * DK
    NSLOT = 2 * KH + VH
    workers = tiles * shaves
    PER = -(-NSLOT // workers)

    @T.prim_func
    def dma_page_probe(Dst: T.Tensor((ROWS, CONV), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            wid = (shave * tiles + tile) if item_worker == "shave" else worker
            buf = T.alloc_local((ROWS, DK), T.float16)
            tmp = T.alloc_local((ROWS, DK), T.float16)
            buf[0, 0] = T.float16(0)
            tmp[0, 0] = T.float16(0)
            for r in T.serial(rounds):
                for it in T.serial(PER):
                    s = (wid * PER + it) if blocked else (it * workers + wid)
                    if s < NSLOT:
                        cb = T.if_then_else(s < 2 * KH, s * DK, 2 * QK + (s - 2 * KH) * DK)
                        if rw:
                            T.copy(Dst[0, cb], tmp)
                        for i, c in T.grid(ROWS, DK):
                            buf[i, c] = T.cast(s + 1 + 100 * r, T.float16)
                        T.copy(buf, Dst[0, cb])
                T.sync_grid()

    return dma_page_probe


def dma_page_probe_sizes(KH: int = 16, VH: int = 32, DK: int = 128, tiles: int = 3, shaves: int = 2, **_):
    """(CONV halves per row, slot count, slots per worker) for `make_dma_page_probe`."""
    NSLOT = 2 * KH + VH
    return dict(CONV=2 * KH * DK + VH * DK, NSLOT=NSLOT, PER=-(-NSLOT // (tiles * shaves)))


def pack_int4(w_int: "np.ndarray") -> "np.ndarray":
    """[N, K] signed int4 values in [-8, 7] -> [N, K/2] bytes, low nibble first (the reference packer)."""
    import numpy as np
    q = (w_int.astype(np.int16) & 0xF).astype(np.uint8)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)


def gm_scales(s: "np.ndarray") -> "np.ndarray":
    """Row-major scales `[N, K/G]` -> the GROUP-MAJOR `[K/G, N]` the kernels read.

    One layout serves both paths: the DPU's int4 GEMV takes it in place (`dpu_matmul_i4`'s `sStride > 0`) and the
    prefill SHAVE dequant reads it through `tl_npu_dequant_i4_gm`. That is what lets prefill and decode share a
    single weight upload -- decode alone moves 17.91 GB of the 40 layers per prompt, so a second copy does not
    fit the box. It is npu_dsl's arrangement, deliberately copied.

    This is the TEST-side packer; `e2e/pack.py` writes the same bytes from the checkpoint."""
    import numpy as np
    return np.ascontiguousarray(s.T)


def make_head(H: int = 2048, V: int = 248320, G: int = 128, tiles: int = 3, shaves: int = 2, BN: int = 128, R: int = 64,
              num_stages: int = 4, eps: float = 1e-6, pipelined: bool = True, h_dtype=T.float16, as_block: bool = False):
    """M5 decode head in ONE launch: logits = rmsnorm_zc(h) . dequant_i8(Wlm)^T over the vocabulary, greedy argmax.

      x = h rsqrt(mean h^2 + eps) (1 + w)                 (every tile, SHAVE 0; the final norm)
      logits[v] = x . Wlm[v]                              (int8 g128 rows with a separate group-major scale tensor,
                                                           split over the tiles, a local ring per SHAVE; the lm_head
                                                           is the largest decode weight: V x H bytes)
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
    # the scales a BLOCK of chunks at a time: per chunk they would be C extra DMA jobs a tile, and the engine's
    # fixed cost per job (not the 4 KiB) is what the lm_head feels at 648 chunks. The lm_head's scales are stored
    # CHUNK-BLOCKED -- `[V_pad / BN x K/G, BN]`, each chunk's [K/G, BN] contiguous -- so a block is ONE contiguous
    # burst; a column window of a flat `[K/G, V_pad]` would be K/G lines of BN halves at the vocabulary's stride,
    # and that strided read cost the WEIGHT stream 6.8 GB/s, far more than the scale bytes. Only this 8 MB tensor
    # carries BN; the 517 MB of weights stays a plain `[V_pad, H]`.
    # The buffer is per-SHAVE so a block needs no tile sync, which would drain the weight ring.
    SB = max(d for d in range(1, C + 1) if C % d == 0 and (H // G) * d * BN * 2 <= 16 * 1024)
    NBLK = C // SB
    workers = tiles * shaves
    HL, FL = 32, 16
    PW = 32

    @T.macro
    def consume(x_cmx, w_cmx, s_cmx, y_cmx, yl, mx, st, bi, tile, n, cn):
        """One chunk after its fill: the BN / R merged-FC workloads, the pad mask, the running max, the logits row.
        `n` is the chunk inside the scales block (its column there), `cn` the chunk inside the tile."""
        for sub in T.serial(BN // R):
            T.gemm_i8(x_cmx, w_cmx, s_cmx, y_cmx, group=G, row0=sub * R, rows=R, scales_major="group",
                      scales_at=(n * ng, 0))
        if (tile * C + cn) * BN + BN > V:   # the tail chunk's columns past V, masked before the max
            for c in T.serial(BN):
                if (tile * C + cn) * BN + c >= V:
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
                    bi[0] = (tile * C + cn) * BN + c
        for j0 in T.serial(BN // HL):
            for v in T.vectorized(HL):
                yl[cn // shaves, j0 * HL + v] = y_cmx[0, j0 * HL + v]

    @T.macro
    def head_phases(x_cmx, WlmQ, WlmS, Logits, Part, Tok, tile, shave):
        """Phases B and C over the normed token `x_cmx` (shared fp16 [1, H]): the vocabulary GEMV with the running
        argmax, the partials, the grid barrier, worker 0's pick into Tok."""
        worker = tile * shaves + shave
        w_cmx = T.alloc_shared((BN, H), T.int8)
        s_cmx = T.alloc_local((SB * ng, BN), T.float16)
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
            # the block loop stays SERIAL: making it a second T.Pipelined nests one ring inside another, and the
            # inner ring's slot rotation moves s_cmx under the addresses `scales_at` computed -- it hangs the device
            for b in T.serial(NBLK):
                T.copy(WlmS[(tile * C + b * SB) * ng, 0], s_cmx)
                for n in T.Pipelined(SB, num_stages=num_stages):
                    T.copy(WlmQ[(tile * C + b * SB + n) * BN, 0], w_cmx)
                    consume(x_cmx, w_cmx, s_cmx, y_cmx, yl, mx, st, bi, tile, n, b * SB + n)
        else:
            for b in T.serial(NBLK):
                T.copy(WlmS[(tile * C + b * SB) * ng, 0], s_cmx)
                for n in T.serial(SB):
                    T.copy(WlmQ[(tile * C + b * SB + n) * BN, 0], w_cmx)
                    if (b * SB + n) % shaves == shave:
                        consume(x_cmx, w_cmx, s_cmx, y_cmx, yl, mx, st, bi, tile, n, b * SB + n)
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
        return head_phases, dict(V_pad=V_pad, C=C, CS=CS, ng=ng, workers=workers, PW=PW)

    @T.prim_func
    def head(Hs: T.Tensor((1, H), h_dtype), NormW: T.Tensor((1, H), T.float16),
             WlmQ: T.Tensor((V_pad, H), T.int8), WlmS: T.Tensor((V_pad // BN * ng, BN), T.float16),
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
            head_phases(x_cmx, WlmQ, WlmS, Logits, Part, Tok, tile, shave)

    return head


def head_sizes(H: int, V: int, G: int = 128, tiles: int = 3, shaves: int = 2, BN: int = 128, **_):
    """make_head's padded vocabulary and block geometry: dict(V_pad, C, CS, ng, workers)."""
    V_pad = -(-V // (BN * tiles * shaves)) * BN * tiles * shaves
    C = V_pad // (BN * tiles)
    return dict(V_pad=V_pad, C=C, CS=C // shaves, ng=H // G, workers=tiles * shaves)


def pack_head(wlm_int, wlm_scales, tiles: int = 3, shaves: int = 2, BN: int = 128):
    """make_head's lm_head: int8 rows zero-padded to the V_pad multiple, plus CHUNK-BLOCKED group-major scales
    `[V_pad / BN x K/G, BN]` -- each chunk's `[K/G, BN]` contiguous, so a run of chunks is one DMA burst.

    The head is the one place the scales carry BN. At 648 chunks a tile, a column window of a flat `[K/G, V_pad]`
    is K/G lines of BN halves at the vocabulary's stride, and that strided read slowed the WEIGHT stream from
    58.9 to 52.1 GB/s -- far more than the scale bytes themselves. The 517 MB of weights stays BN-free; only this
    8 MB tensor is re-packed when BN changes (and `head_BN` is in `pack.PAD_KEYS` for exactly that reason)."""
    import numpy as np
    V, H = wlm_int.shape
    V_pad = head_sizes(H, V, tiles=tiles, shaves=shaves, BN=BN)["V_pad"]
    w = np.zeros((V_pad, H), np.int8)
    s = np.zeros((V_pad, wlm_scales.shape[1]), np.float16)
    w[:V] = wlm_int
    s[:V] = wlm_scales
    blocks = [np.ascontiguousarray(s[n0:n0 + BN].T) for n0 in range(0, V_pad, BN)]
    return dict(WlmQ=w, WlmS=np.concatenate(blocks, axis=0))


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
    CONV, NIN_pad, NOUT_pad, HG1, KG2, VC = gs["CONV"], gs["NIN_pad"], gs["NOUT_pad"], gs["HG1"], gs["KG2"], gs["VC"]
    WBA_pad, YLEN, BA_LEN, CVLEN, OPAD = gs["WBA_pad"], gs["YLEN"], gs["BA_LEN"], gs["CVLEN"], gs["OPAD"]
    per_round = tiles * shaves * mk["BNR"]
    RR = -(-(E + 1) // per_round) * per_round
    CG, CD = (2 * I) // mk["BNG"], H // mk["BND"]
    HG, IG = H // G, I // G
    MR = 16

    @T.prim_func
    def gdn_layer(Hio: T.Tensor((1, H), T.float32), InNorm: T.Tensor((1, H), T.float16), PostNorm: T.Tensor((1, H), T.float16),
                  WinQ: T.Tensor((NIN_pad, H // 2), T.uint8), WinS: T.Tensor((HG1, NIN_pad), T.float16),
                  Wba: T.Tensor((WBA_pad, H), T.float16),
                  ConvW: T.Tensor((4, CONV), T.float16), ALog: T.Tensor((1, VH), T.float32), DtBias: T.Tensor((1, VH), T.float32),
                  GNorm: T.Tensor((1, DV), T.float16), ConvState: T.Tensor((3, CONV), T.float16),
                  Ssm: T.Tensor((VH, DK, DV), T.float32),
                  WoutQ: T.Tensor((NOUT_pad, VC // 2), T.uint8), WoutS: T.Tensor((KG2, NOUT_pad), T.float16),
                  Wr: T.Tensor((RR, H), T.float16),
                  WguQ: T.Tensor(((E + 1) * 2 * I, H // 2), T.uint8), WguS: T.Tensor((HG, (E + 1) * 2 * I), T.float16),
                  WdQ: T.Tensor(((E + 1) * H, I // 2), T.uint8), WdS: T.Tensor((IG, (E + 1) * H), T.float16),
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
            gdn_mixer(x_cmx, WinQ, WinS, Wba, ConvW, ALog, DtBias, GNorm, ConvState, Ssm, WoutQ, WoutS, Y, BA, CV, O,
                      Mix, Mix, tile, shave)
            T.sync_grid()
            T.copy(Mix[0:1, 0:H], mix_cmx[0:1, 0:H])
            add_rows(h32, mix_cmx, shave)
            T.copy(PostNorm[0, 0], nw_cmx)
            rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
            moe_phases(x_cmx, Wr, WguQ, WguS, WdQ, WdS, Logits, PartM, tile, shave)
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
    NQKV, NOUT_pad, HG1, KG2, KO, GQ, PW = asz["NQKV"], asz["NOUT_pad"], asz["HG1"], asz["KG2"], asz["KO"], asz["GQ"], asz["PW"]
    per_round = tiles * shaves * mk["BNR"]
    RR = -(-(E + 1) // per_round) * per_round
    CG, CD = (2 * I) // mk["BNG"], H // mk["BND"]
    HG, IG = H // G, I // G
    MR = 16

    @T.prim_func
    def attn_layer(Hio: T.Tensor((1, H), T.float32), InNorm: T.Tensor((1, H), T.float16), PostNorm: T.Tensor((1, H), T.float16),
                   WqkvQ: T.Tensor((NQKV, H // 2), T.uint8), WqkvS: T.Tensor((HG1, NQKV), T.float16),
                   QNormW: T.Tensor((1, D), T.float16),
                   KNormW: T.Tensor((1, D), T.float16), CosSin: T.Tensor((2, ROT), T.float32), Pos: T.Tensor((1, 1), T.int32),
                   Kc: T.Tensor((NKV, SMAX, D), T.float16), Vt: T.Tensor((NKV, D, SMAX), T.float16),
                   WoQ: T.Tensor((NOUT_pad, KO // 2), T.uint8), WoS: T.Tensor((KG2, NOUT_pad), T.float16),
                   Wr: T.Tensor((RR, H), T.float16),
                   WguQ: T.Tensor(((E + 1) * 2 * I, H // 2), T.uint8), WguS: T.Tensor((HG, (E + 1) * 2 * I), T.float16),
                   WdQ: T.Tensor(((E + 1) * H, I // 2), T.uint8), WdS: T.Tensor((IG, (E + 1) * H), T.float16),
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
            attn_mixer(x_cmx, WqkvQ, WqkvS, QNormW, KNormW, CosSin, Pos, Kc, Vt, WoQ, WoS, Y, QN, PartA, O, Mix, Mix,
                       tile, shave, Pos, 0)
            T.sync_grid()
            T.copy(Mix[0:1, 0:H], mix_cmx[0:1, 0:H])
            add_rows(h32, mix_cmx, shave)
            T.copy(PostNorm[0, 0], nw_cmx)
            rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
            moe_phases(x_cmx, Wr, WguQ, WguS, WdQ, WdS, Logits, PartM, tile, shave)
            T.copy(PartM[0, 0], p_cmx)
            add_parts(h32, p_cmx, tiles, shave)
            if tile == 0:
                if shave == 0:
                    T.copy(h32, Hio[0, 0])

    return attn_layer


# Layer table columns (`make_decode_model`): one int64 device address per column, filled by the host from
# `Session.address`; a column a layer kind does not use holds 0.
TABLE_COLS = ["InNorm", "PostNorm", "W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "Wr", "WguQ", "WguS",
              "WdQ", "WdS", "S0", "S1"]
GDN_TABLE = dict(W0="WinQ", W1="WinS", W2="Wba", W3="ConvW", W4="ALog", W5="DtBias", W6="GNorm", W7="WoutQ",
                 W8="WoutS", S0="ConvState", S1="Ssm")
# The attention KV cache is a kernel parameter (`KcAll` / `VtAll`), not a table address: see `attn_block`'s `kv4`.
# `Kind[li]` is (kind, kv_slot, 0, 0): kv_slot is the layer's index among the launch's attention layers.
ATTN_TABLE = dict(W0="WqkvQ", W1="WqkvS", W2="QNormW", W3="KNormW", W4="WoQ", W5="WoS")


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
    CONV, NIN_pad, NOUTg, HG1g, KG2g, VC = gs["CONV"], gs["NIN_pad"], gs["NOUT_pad"], gs["HG1"], gs["KG2"], gs["VC"]
    WBA_pad, YLEN, BA_LEN, CVLEN, OPADg = gs["WBA_pad"], gs["YLEN"], gs["BA_LEN"], gs["CVLEN"], gs["OPAD"]
    NQKV, NOUTa, HG1a, KG2a, KO, GQ, PWa = asz["NQKV"], asz["NOUT_pad"], asz["HG1"], asz["KG2"], asz["KO"], asz["GQ"], asz["PW"]
    per_round = tiles * shaves * mk["BNR"]
    RR = -(-(E + 1) // per_round) * per_round
    CG, CD = (2 * I) // mk["BNG"], H // mk["BND"]
    HG, IG = H // G, I // G
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
              NormW: T.Tensor((1, H), T.float16), WlmQ: T.Tensor((hs["V_pad"], H), T.int8),
              WlmS: T.Tensor((hs["V_pad"] // BNh * hs["ng"], BNh), T.float16),
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
                WguQ = T.ddr_view(Table[li, col["WguQ"]], ((E + 1) * 2 * I, H // 2), T.uint8)
                WguS = T.ddr_view(Table[li, col["WguS"]], (HG, (E + 1) * 2 * I), T.float16)
                WdQ = T.ddr_view(Table[li, col["WdQ"]], ((E + 1) * H, I // 2), T.uint8)
                WdS = T.ddr_view(Table[li, col["WdS"]], (IG, (E + 1) * H), T.float16)
                perf_begin("lhed")
                T.copy(InNorm[0, 0], nw_cmx)
                if "norm" not in skip:
                    rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
                perf_end()
                if Kind[li, 0] == 0:
                    WinQ = T.ddr_view(Table[li, col["W0"]], (NIN_pad, H // 2), T.uint8)
                    WinS = T.ddr_view(Table[li, col["W1"]], (HG1g, NIN_pad), T.float16)
                    Wba = T.ddr_view(Table[li, col["W2"]], (WBA_pad, H), T.float16)
                    ConvW = T.ddr_view(Table[li, col["W3"]], (4, CONV), T.float16)
                    ALog = T.ddr_view(Table[li, col["W4"]], (1, VH), T.float32)
                    DtBias = T.ddr_view(Table[li, col["W5"]], (1, VH), T.float32)
                    GNorm = T.ddr_view(Table[li, col["W6"]], (1, DV), T.float16)
                    WoutQ = T.ddr_view(Table[li, col["W7"]], (NOUTg, VC // 2), T.uint8)
                    WoutS = T.ddr_view(Table[li, col["W8"]], (KG2g, NOUTg), T.float16)
                    ConvState = T.ddr_view(Table[li, col["S0"]], (3, CONV), T.float16)
                    Ssm = T.ddr_view(Table[li, col["S1"]], (VH, DK, DV), T.float32)
                    if "gdn" not in skip:
                        gdn_mixer(x_cmx, WinQ, WinS, Wba, ConvW, ALog, DtBias, GNorm, ConvState, Ssm, WoutQ, WoutS,
                                  Y, BA, CV, O, Mix, Mix, tile, shave)
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
                    WqkvQ = T.ddr_view(Table[li, col["W0"]], (NQKV, H // 2), T.uint8)
                    WqkvS = T.ddr_view(Table[li, col["W1"]], (HG1a, NQKV), T.float16)
                    QNormW = T.ddr_view(Table[li, col["W2"]], (1, D), T.float16)
                    KNormW = T.ddr_view(Table[li, col["W3"]], (1, D), T.float16)
                    WoQ = T.ddr_view(Table[li, col["W4"]], (NOUTa, KO // 2), T.uint8)
                    WoS = T.ddr_view(Table[li, col["W5"]], (KG2a, NOUTa), T.float16)
                    if "attn" not in skip:
                        attn_mixer(x_cmx, WqkvQ, WqkvS, QNormW, KNormW, CosSin, Pos, KcAll, VtAll, WoQ, WoS, YA, QN,
                                   PartA, OA, MixA, MixA, tile, shave, Kind, li)
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
                    moe_phases(x_cmx, Wr, WguQ, WguS, WdQ, WdS, Logits, PartM, tile, shave)
                    perf_begin("lprt")
                    T.copy(PartM[0, 0], p_cmx)
                    add_parts(h32, p_cmx, tiles, shave)
                    perf_end()
            # a grid barrier closes the layer loop's CMX phase before the head's ring (the lm_head ring is 1 MB)
            T.sync_grid()
            T.copy(NormW[0, 0], nw_cmx)
            rmsnorm_zc(h32, nw_cmx, x_cmx, shave)
            if "head" not in skip:
                head_phases(x_cmx, WlmQ, WlmS, HLogits, HPart, Tok, tile, shave)
            if tile == 0:
                if shave == 0:
                    T.copy(h32, Hio[0, 0])

    return model


# Layer table columns for `make_prefill_model`. Prefill and decode read the SAME weight tensors (packed values
# plus group-major scales); the two tables differ only in which of them a layer kind needs.
PREFILL_TABLE_COLS = ["InNorm", "PostNorm", "W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7",
                      "Wr", "WguQ", "WguS", "WdQ", "WdS", "S0", "S1", "W8"]
GDN_PREFILL_TABLE = dict(W0="WinQ", W1="WinS", W2="ConvW", W3="ALog", W4="DtBias", W5="GNorm", W6="WoutQ",
                         W7="WoutS", W8="Wba", S0="ConvState", S1="Ssm")
# As in decode, the KV cache is a kernel parameter (`KcAll` / `VtAll`) and NOT a table address (question 8);
# `Kind[li]` is (kind, kv_slot, 0, 0).
ATTN_PREFILL_TABLE = dict(W0="WqkvQ", W1="WqkvS", W2="QNormW", W3="KNormW", W4="WoQ", W5="WoS")


def make_prefill_model(L: int, N: int = 192, H: int = 2048, KH: int = 16, VH: int = 32, DK: int = 128,
                       DV: int = 128, NH: int = 16, NKV: int = 2, D: int = 256, ROT: int = 64, SMAX: int = 1536,
                       I: int = 512, E: int = 256, TOPK: int = 8, G: int = 128, TB: int = 32, C: int = 64,
                       TBq: int = 8, TBF: int = 16, BKV: int = 128, BN1: int = 32, BN2: int = 32, MB: int = 16,
                       BM: int = 64, BNR: int = 16, MBe: int = 16, BNG: int = 64, BND: int = 64, tiles: int = 3,
                       shaves: int = 2, eps: float = 1e-6, KVL: int | None = None, skip: frozenset = frozenset(),
                       unroll_layers: bool = False, conv_blocked: bool = True, conv_item_worker: str = "tile",
                       HW: int | None = None):
    """P5: a whole PREFILL chunk of `N` prompt tokens through `L` layers in ONE launch -- the prefill counterpart
    of `make_decode_model`. A device-side loop over the layers, `Kind[li, 0]` selecting the Gated DeltaNet or the
    full-attention mixer, every layer's weights reached through `Table[li, :]` (`PREFILL_TABLE_COLS`), the residual
    stream fp32 in DDR (`Hio[NP, H]`: at `N` tokens it does not fit CMX, which is the one structural difference
    from the decode kernel), and the DDR scratch shared by every layer and both kinds.

    `Ys` and `Os` are sized for the WIDER of the two kinds and each phase indexes within its own extent -- a 2-D
    DDR copy takes its row stride from the buffer, so a narrower phase reading a wider buffer is correct by
    construction (the same mechanism the attention layer already relies on for its padded `Ys`).

    The GDN conv state is updated IN PLACE: `Table[li, "S0"]` is passed as both the phase's input state and its
    output. Within one work item the read of a channel slice precedes the write of the same slice, and items own
    disjoint slices, so no barrier is owed.

    The head is NOT here: it is wanted on the last token of the last chunk only, so the driver launches the
    standalone `make_head` kernel on `Hio[nt - 1]` instead of this kernel carrying 8.8 ms of lm_head per chunk.

    `skip` (diagnostic only) leaves out whole stages: any of "gdn", "attn", "moe", "norm".

    `HW` is the ROW COUNT the out projections were PACKED with. The packer pads a projection's rows to the
    DECODE kernel's `BN2 * tiles`, so `WoutS`/`WoS` can be wider than `H`; the scale view must carry that pitch
    or every group after the first is read from the wrong column. Only the first `H` rows are computed."""
    GQ, KO = NH // NKV, NH * D
    QK, VC = KH * DK, VH * DV
    CONV = 2 * QK + VC
    NIN = CONV + VC
    HW = H if HW is None else HW
    assert HW >= H, f"packed out-proj rows {HW} < H {H}"
    workers = tiles * shaves
    step = 1
    for k in (TB, C, TBq, TBF, workers, BM):
        step = step * k // math.gcd(step, k)
    NP = -(-N // step) * step
    conv_phase, _ = make_gdn_conv_prefill(N=NP, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB, tiles=tiles, shaves=shaves,
                                          as_block=True, blocked_items=conv_blocked, item_worker=conv_item_worker)
    scan_phase, _ = make_gdn_scan_prefill(N=NP, KH=KH, VH=VH, DK=DK, DV=DV, C=C, tiles=tiles, shaves=shaves,
                                          eps=eps, as_block=True)
    qkv_phase, qs = make_attn_qkv_prefill(N=NP, H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, TB=TBq, tiles=tiles,
                                          shaves=shaves, eps=eps, as_block=True, kv4=True)
    flash_phase, _ = make_attn_prefill(N=NP, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, TB=TBF, BKV=BKV, tiles=tiles,
                                       shaves=shaves, as_block=True, kv4=True)
    NIN_pad = -(-NIN // BN1) * BN1
    NQKV_pad = -(-qs["NQKV"] // BN1) * BN1
    YW, OW = max(NIN_pad, NQKV_pad), max(VC, KO)
    in_proj, _ = make_proj_prefill(NP=NP, M=NIN_pad, K=H, BN=BN1, MB=MB, G=G, tiles=tiles, shaves=shaves,
                                   as_block=True, scales_gm=True)
    gdn_out_proj, _ = make_proj_prefill(NP=NP, M=H, K=VC, BN=BN2, MB=MB, G=G, tiles=tiles, shaves=shaves,
                                        as_block=True, scales_gm=True)
    qkv_proj, _ = make_proj_prefill(NP=NP, M=NQKV_pad, K=H, BN=BN1, MB=MB, G=G, tiles=tiles, shaves=shaves,
                                    as_block=True, scales_gm=True)
    o_proj, _ = make_proj_prefill(NP=NP, M=H, K=KO, BN=BN2, MB=MB, G=G, tiles=tiles, shaves=shaves, as_block=True,
                                  scales_gm=True)
    route_phase, rs = make_moe_route_prefill(N=NP, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=tiles,
                                             shaves=shaves, as_block=True)
    expert_phase, _ = make_moe_expert_prefill(N=NP, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=MBe, BNG=BNG, BND=BND,
                                              tiles=tiles, shaves=shaves, as_block=True)
    RR, SLOTS = rs["RR"], rs["SLOTS"]
    KGH, KGI = H // G, I // G
    FL = 16
    assert H % FL == 0 and NP % workers == 0
    norm_tokens = prefill_norm_block(H, NP, workers, eps, FL)
    ba_phase = prefill_ba_block(H, VH, NP, MB, tiles, shaves)
    NCOL = len(PREFILL_TABLE_COLS)
    col = {n: i for i, n in enumerate(PREFILL_TABLE_COLS)}
    KVL = L if KVL is None else KVL

    @T.macro
    def add_mix(Hio, Src, hrow, mrow, nt, worker):
        """h += the mixer output, live tokens only: past Nt a zero query gives a uniform softmax, not a zero."""
        for it in T.serial(NP // workers):
            t = it * workers + worker
            if t < nt[0, 0]:
                T.copy(Hio[t, 0], hrow)
                T.copy(Src[t, 0], mrow)
                for d0 in T.serial(H // FL):
                    for v in T.vectorized(FL):
                        hrow[0, d0 * FL + v] = hrow[0, d0 * FL + v] + T.cast(mrow[0, d0 * FL + v], T.float32)
                T.copy(hrow, Hio[t, 0])

    @T.prim_func
    def prefill_model(Hio: T.Tensor((NP, H), T.float32), Pos: T.Tensor((1, 1), T.int32),
                      Nt: T.Tensor((1, 1), T.int32), CosSin: T.Tensor((NP, 2 * ROT), T.float32),
                      Kind: T.Tensor((L, 4), T.int32), Table: T.Tensor((L, NCOL), T.int64),
                      KcAll: T.Tensor((KVL, NKV, SMAX, D), T.float16),
                      VtAll: T.Tensor((KVL, NKV, D, SMAX), T.float16),
                      Xs: T.Tensor((NP, H), T.float16), Ys: T.Tensor((NP, YW), T.float16),
                      BAs: T.Tensor((NP, 2 * VH), T.float16),
                      QKs: T.Tensor((KH, NP, 2 * DK), T.float32), Vs: T.Tensor((VH, NP, DV), T.float16),
                      Gbs: T.Tensor((VH, NP, 2), T.float32), Zs: T.Tensor((VH, NP, DV), T.float16),
                      Q: T.Tensor((NKV, NP, GQ, D), T.float16), Gate: T.Tensor((NP, KO), T.float16),
                      Os: T.Tensor((NP, OW), T.float16), Mix: T.Tensor((NP, H), T.float16),
                      Logits: T.Tensor((NP, RR), T.float16), Ids: T.Tensor((NP, TOPK), T.int32),
                      Wts: T.Tensor((NP, 16), T.float32), Cnt: T.Tensor((E + 1,), T.int32),
                      Off: T.Tensor((E + 2,), T.int32), Slot: T.Tensor((SLOTS,), T.int32),
                      PartE: T.Tensor((tiles, NP, H), T.float32), MoeOut: T.Tensor((NP, H), T.float32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            shave = T.get_thread_binding()
            worker = tile * shaves + shave
            nw = T.alloc_shared((1, H), T.float16)
            nt = T.alloc_shared((1, 1), T.int32)
            hrow = T.alloc_local((1, H), T.float32)
            xrow = T.alloc_local((1, H), T.float16)
            mrow = T.alloc_local((1, H), T.float16)
            erow = T.alloc_local((1, H), T.float32)
            ls = T.alloc_local((FL,), T.float32)
            st = T.alloc_local((4,), T.float32)
            hrow[0, 0] = T.float32(0)     # pins the scope at kernel level (question 10)
            xrow[0, 0] = T.float16(0)
            mrow[0, 0] = T.float16(0)
            erow[0, 0] = T.float32(0)
            ls[0] = T.float32(0)
            st[0] = T.float32(0)
            T.copy(Nt[0, 0], nt)
            # The GDN conv phase's buffers live HERE, not inside the layer loop's Kind branch: ScopeIntelNpuCmx
            # hoists only a loop body's top-level declarations, so an allocation inside the branch is re-made
            # every iteration -- question 10's unsound state, and the one npu_dsl never enters (one packed arena).
            cbuf = dict(nt=T.alloc_shared((1, 1), T.int32), al=T.alloc_shared((1, VH), T.float32),
                        db=T.alloc_shared((1, VH), T.float32), win=T.alloc_local((3, DK), T.float16),
                        keep=T.alloc_local((3, DK), T.float16), cw=T.alloc_local((4, DK), T.float16),
                        xb=T.alloc_local((TB, DK), T.float16), cv=T.alloc_local((TB, DK), T.float32),
                        ob=T.alloc_local((TB, DK), T.float32), zb=T.alloc_local((TB, DV), T.float16),
                        bb=T.alloc_local((TB, 1), T.float16), aa=T.alloc_local((TB, 1), T.float16),
                        gbb=T.alloc_local((TB, 2), T.float32), acc=T.alloc_local((FL,), T.float32),
                        eh=T.alloc_local((FL,), T.float16), ls=T.alloc_local((FL,), T.float32),
                        st=T.alloc_local((8,), T.float32))
            cbuf["win"][0, 0] = T.float16(0)     # first use pins each scope at kernel level (question 10)
            cbuf["keep"][0, 0] = T.float16(0)
            cbuf["cw"][0, 0] = T.float16(0)
            cbuf["xb"][0, 0] = T.float16(0)
            cbuf["cv"][0, 0] = T.float32(0)
            cbuf["ob"][0, 0] = T.float32(0)
            cbuf["zb"][0, 0] = T.float16(0)
            cbuf["bb"][0, 0] = T.float16(0)
            cbuf["aa"][0, 0] = T.float16(0)
            cbuf["gbb"][0, 0] = T.float32(0)
            cbuf["acc"][0] = T.float32(0)
            cbuf["ls"][0] = T.float32(0)
            cbuf["eh"][0] = T.float16(0)
            cbuf["st"][0] = T.float32(0)
            for li in (range(L) if unroll_layers else T.serial(L)):
                InNorm = T.ddr_view(Table[li, col["InNorm"]], (1, H), T.float16)
                PostNorm = T.ddr_view(Table[li, col["PostNorm"]], (1, H), T.float16)
                Wr = T.ddr_view(Table[li, col["Wr"]], (RR, H), T.float16)
                WguQ = T.ddr_view(Table[li, col["WguQ"]], ((E + 1) * 2 * I, H // 2), T.uint8)
                WguS = T.ddr_view(Table[li, col["WguS"]], (KGH, (E + 1) * 2 * I), T.float16)
                WdQ = T.ddr_view(Table[li, col["WdQ"]], ((E + 1) * H, I // 2), T.uint8)
                WdS = T.ddr_view(Table[li, col["WdS"]], (KGI, (E + 1) * H), T.float16)
                if "norm" not in skip:
                    perf_phase_begin("nrm1")
                    norm_tokens(Hio, InNorm, Xs, nw, hrow, xrow, ls, st, worker)
                    perf_phase_end("nrm1")
                T.sync_grid()
                if Kind[li, 0] == 0:
                    WinQ = T.ddr_view(Table[li, col["W0"]], (NIN_pad, H // 2), T.uint8)
                    WinS = T.ddr_view(Table[li, col["W1"]], (H // G, NIN_pad), T.float16)
                    ConvW = T.ddr_view(Table[li, col["W2"]], (4, CONV), T.float16)
                    ALog = T.ddr_view(Table[li, col["W3"]], (1, VH), T.float32)
                    DtBias = T.ddr_view(Table[li, col["W4"]], (1, VH), T.float32)
                    GNorm = T.ddr_view(Table[li, col["W5"]], (1, DV), T.float16)
                    WoutQ = T.ddr_view(Table[li, col["W6"]], (HW, VC // 2), T.uint8)
                    WoutS = T.ddr_view(Table[li, col["W7"]], (VC // G, HW), T.float16)
                    Wba = T.ddr_view(Table[li, col["W8"]], (2 * VH, H), T.float16)
                    ConvState = T.ddr_view(Table[li, col["S0"]], (3, CONV), T.float16)
                    Ssm = T.ddr_view(Table[li, col["S1"]], (VH, DK, DV), T.float32)
                    if "gdn" not in skip:
                        perf_phase_begin("ginp")
                        in_proj(Xs, WinQ, WinS, Ys, tile, shave)
                        ba_phase(Xs, Wba, BAs, tile, shave)
                        perf_phase_end("ginp")
                        T.sync_grid()
                        perf_phase_begin("gcnv")
                        conv_phase(Ys, BAs, ConvW, ConvState, ALog, DtBias, Nt, QKs, Vs, Gbs, Zs, ConvState, tile,
                                   shave, cbuf)
                        perf_phase_end("gcnv")
                        T.sync_grid()
                        perf_phase_begin("gscn")
                        scan_phase(QKs, Vs, Gbs, Zs, GNorm, Nt, Ssm, Os, tile, shave)
                        perf_phase_end("gscn")
                        T.sync_grid()
                        perf_phase_begin("gout")
                        gdn_out_proj(Os, WoutQ, WoutS, Mix, tile, shave)
                        perf_phase_end("gout")
                        T.sync_grid()
                        perf_phase_begin("gres")
                        add_mix(Hio, Mix, hrow, mrow, nt, worker)
                        perf_phase_end("gres")
                else:
                    WqkvQ = T.ddr_view(Table[li, col["W0"]], (NQKV_pad, H // 2), T.uint8)
                    WqkvS = T.ddr_view(Table[li, col["W1"]], (H // G, NQKV_pad), T.float16)
                    QNormW = T.ddr_view(Table[li, col["W2"]], (1, D), T.float16)
                    KNormW = T.ddr_view(Table[li, col["W3"]], (1, D), T.float16)
                    WoQ = T.ddr_view(Table[li, col["W4"]], (HW, KO // 2), T.uint8)
                    WoS = T.ddr_view(Table[li, col["W5"]], (KO // G, HW), T.float16)
                    if "attn" not in skip:
                        perf_phase_begin("aqkv")
                        qkv_proj(Xs, WqkvQ, WqkvS, Ys, tile, shave)
                        perf_phase_end("aqkv")
                        T.sync_grid()
                        perf_phase_begin("arop")
                        qkv_phase(Ys, QNormW, KNormW, CosSin, Pos, Nt, Q, Gate, KcAll, VtAll, tile, shave, Kind, li)
                        perf_phase_end("arop")
                        T.sync_grid()
                        perf_phase_begin("afls")
                        flash_phase(Q, KcAll, VtAll, Gate, Pos, Os, tile, shave, Kind, li)
                        perf_phase_end("afls")
                        T.sync_grid()
                        perf_phase_begin("aout")
                        o_proj(Os, WoQ, WoS, Mix, tile, shave)
                        perf_phase_end("aout")
                        T.sync_grid()
                        perf_phase_begin("ares")
                        add_mix(Hio, Mix, hrow, mrow, nt, worker)
                        perf_phase_end("ares")
                T.sync_grid()
                if "norm" not in skip:
                    perf_phase_begin("nrm2")
                    norm_tokens(Hio, PostNorm, Xs, nw, hrow, xrow, ls, st, worker)
                    perf_phase_end("nrm2")
                T.sync_grid()
                if "moe" not in skip:
                    perf_phase_begin("mrot")
                    route_phase(Xs, Wr, Logits, Ids, Wts, Cnt, Off, Slot, tile, shave)
                    perf_phase_end("mrot")
                    # the grouping is worker 0's alone and the expert phase's first act is to read it back;
                    # DeriveIntelNpuGlobalSync only sees the TOP-LEVEL phase sequence, so inside the layer loop
                    # this barrier has to be written (without it the expert loop bounds are a race)
                    T.sync_grid()
                    perf_phase_begin("mexp")
                    expert_phase(Xs, WguQ, WguS, WdQ, WdS, Wts, Off, Slot, PartE, MoeOut, tile, shave)
                    perf_phase_end("mexp")
                    T.sync_grid()
                    perf_phase_begin("madd")
                    for it in T.serial(NP // workers):     # h += the MoE output, live tokens only
                        t = it * workers + worker
                        if t < nt[0, 0]:
                            T.copy(Hio[t, 0], hrow)
                            T.copy(MoeOut[t, 0], erow)
                            for d0 in T.serial(H // FL):
                                for v in T.vectorized(FL):
                                    hrow[0, d0 * FL + v] = hrow[0, d0 * FL + v] + erow[0, d0 * FL + v]
                            T.copy(hrow, Hio[t, 0])
                    perf_phase_end("madd")
                    T.sync_grid()

    return prefill_model


def prefill_model_reference(layers: list, h, cos_sin, pos0: int, n: int, NH: int, NKV: int, D: int, ROT: int,
                            KH: int, VH: int, DK: int, DV: int, E: int, TOPK: int, G: int, kc=None, vt=None,
                            eps: float = 1e-6):
    """numpy oracle of make_prefill_model (fp32): the per-layer oracles chained, which is what the model kernel
    claims. `layers` is one dict per layer in run order with `kind` ("gdn" / "attn"), the layer's weights, and for
    attention a `kv` index into the `kc` / `vt` cache lists. The GDN layer oracle adds the MoE output to the dead
    rows too (its kernel does); the model kernel guards both halves of both kinds, so the dead rows are restored
    here -- they cannot influence a live token, which is why the two kernels may differ there at all."""
    import numpy as np
    out = h.astype(np.float32)
    kc = [] if kc is None else kc
    vt = [] if vt is None else vt
    for p in layers:
        dead = out[n:].copy()
        if p["kind"] == "gdn":
            out, p["conv_state"], p["ssm"] = gdn_moe_layer_prefill_reference(
                out, p["in_norm"], p["post_norm"], p["win_int"], p["win_s"], p["conv_w"], p["conv_state"],
                p["a_log"], p["dt_bias"], p["gnorm"], p["wout_int"], p["wout_s"], p["ssm"], p["wr"],
                p["gu_int"], p["gu_s"], p["d_int"], p["d_s"], n=n, KH=KH, VH=VH, DK=DK, DV=DV, E=E, TOPK=TOPK,
                G=G, eps=eps, wba=p.get("wba"))
        else:
            k = p["kv"]
            out, kc[k], vt[k] = attn_moe_layer_prefill_reference(
                out, p["in_norm"], p["post_norm"], p["wqkv_int"], p["wqkv_s"], p["qnorm"], p["knorm"], cos_sin,
                p["wo_int"], p["wo_s"], kc[k], vt[k], p["wr"], p["gu_int"], p["gu_s"], p["d_int"], p["d_s"],
                pos0=pos0, n=n, NH=NH, NKV=NKV, D=D, ROT=ROT, E=E, TOPK=TOPK, G=G, eps=eps)
        out[n:] = dead
    return out, kc, vt
