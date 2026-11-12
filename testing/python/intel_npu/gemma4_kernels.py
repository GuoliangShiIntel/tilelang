"""Gemma-4-26B-A4B kernel building blocks for the Intel NPU backend.

See docs/developer_guide/gemma4_enablement_plan.md for the model architecture, the gap analysis
against the existing Qwen3.6 kernels in kernels.py, and the milestone log (V0 is this file's first
content: a flash-attention phase without Qwen3.6's output gate and with an optional sliding-window
lower bound, the one piece of the mixer that has no existing counterpart to reuse at all).

No `from __future__ import annotations`: T.prim_func evaluates the tensor annotations against the
factory's locals, same convention as kernels.py.
"""

import tilelang.intel_npu.language as T


def make_gemma4_attn_prefill(
    N: int,
    NKV: int,
    GQ: int,
    D: int,
    SMAX: int,
    window: int | None = None,
    TB: int = 4,
    BKV: int = 128,
    tiles: int = 3,
    shaves: int = 2,
    as_block: bool = False,
):
    """Gemma4's flash-attention PREFILL phase: GQA + causal mask (+ an optional sliding-window lower
    bound), with NO output gate -- unlike `kernels.make_attn_prefill`, which Gemma4 has no counterpart
    of at all (every Gemma4 attention layer's `o_proj` reads the raw context, see
    `Gemma4Ref._attention`'s `return context @ weights["o"].T`).

      O[t, (g * GQ + hh) * D ...] = softmax_j(q[g, t, hh] . Kc[g, j] / sqrt(D))[visible(t, j)] . Vt[g, :, j]^T
      visible(t, j) := j <= pos0 + t  and  (window is None or j > pos0 + t - window)

    Derived from `make_attn_prefill`'s online-softmax / DPU-call skeleton (same work split: one worker
    owns a (kv head, token block of TB tokens) item end to end). `window=None` reduces to plain causal
    attention, structurally identical to `make_attn_prefill` minus the gate.

    V0 simplification (see the plan's open questions): chunks are still walked `0 .. ck` (the existing
    causal upper bound only); a windowed call does NOT yet skip whole chunks before the window, it masks
    every element of every loaded chunk against both bounds. This is correct but not the efficient form
    (chunks entirely outside the window are loaded and fully masked to zero) -- a chunk-skip lower bound
    is future work once correctness is established, not a V0 concern.

    Oracle: `gemma4_attn_prefill_reference`."""
    assert D % 32 == 0 and BKV % 32 == 0 and SMAX % BKV == 0 and N % TB == 0
    ROWS = TB * GQ
    KO = NKV * GQ * D
    assert ROWS <= 8192 and D <= 8192 and BKV <= 8192, f"query block {ROWS} x {D} / score block {ROWS} x {BKV}"
    HL, FL = 32, 16
    assert BKV % HL == 0 and D % FL == 0
    scale = 1.0 / (D**0.5)
    workers = tiles * shaves
    NIT = NKV * (N // TB)
    CMAX = SMAX // BKV

    @T.macro
    def flash_phase(Q, Kc, Vt, Pos, O, tile, shave, KvSlot, kvi):
        worker = tile * shaves + shave
        pos_cmx = T.alloc_shared((1, 1), T.int32)
        q_cmx = T.alloc_local((ROWS, D), T.float16)
        k_cmx = T.alloc_local((BKV, D), T.float16)
        v_cmx = T.alloc_local((D, BKV), T.float16)
        s_cmx = T.alloc_local((ROWS, BKV), T.float16)
        pv_cmx = T.alloc_local((ROWS, D), T.float16)
        o_acc = T.alloc_local((ROWS, D), T.float32)
        o_cmx = T.alloc_local((TB, GQ * D), T.float16)  # the normalised output, token-major: O's shape
        m_i = T.alloc_local((ROWS,), T.float32)
        l_i = T.alloc_local((ROWS,), T.float32)
        m_new = T.alloc_local((ROWS,), T.float32)
        corr = T.alloc_local((ROWS,), T.float32)
        mx = T.alloc_local((HL,), T.float16)
        ls = T.alloc_local((FL,), T.float32)
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
                for kv in T.serial(ck):
                    if KvSlot is None:
                        T.copy(Kc[g, kv * BKV, 0], k_cmx)
                        T.copy(Vt[g, 0, kv * BKV], v_cmx)
                    else:
                        T.copy(Kc[KvSlot[kvi, 1], g, kv * BKV, 0], k_cmx)
                        T.copy(Vt[KvSlot[kvi, 1], g, 0, kv * BKV], v_cmx)
                    T.gemm(q_cmx, k_cmx, s_cmx, transpose_B=True, clear_accum=True)
                    # causal: row i is token t0 + i // GQ at position pos0 + t0 + i // GQ
                    if kv * BKV + BKV > pos_cmx[0, 0] + t0:
                        for i, j in T.grid(ROWS, BKV):
                            if kv * BKV + j > pos_cmx[0, 0] + t0 + i // GQ:
                                s_cmx[i, j] = T.float16(-65504)
                    if window is not None:
                        # the sliding-window lower bound: a key is visible only if j > pos - window
                        for i, j in T.grid(ROWS, BKV):
                            if kv * BKV + j <= pos_cmx[0, 0] + t0 + i // GQ - window:
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
                # normalise -- no output gate (Gemma4 has none)
                for r, hh in T.grid(TB, GQ):
                    corr[r * GQ + hh] = T.float32(1) / l_i[r * GQ + hh]
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            o_cmx[r, hh * D + d0 * FL + v] = T.cast(o_acc[r * GQ + hh, d0 * FL + v] * corr[r * GQ + hh], T.float16)
                T.copy(o_cmx, O[t0, g * GQ * D])

    if as_block:
        return flash_phase, dict(KO=KO, ROWS=ROWS, NIT=NIT, CMAX=CMAX)

    @T.prim_func
    def attn_prefill(
        Q: T.Tensor((NKV, N, GQ, D), T.float16),
        Kc: T.Tensor((NKV, SMAX, D), T.float16),
        Vt: T.Tensor((NKV, D, SMAX), T.float16),
        Pos: T.Tensor((1, 1), T.int32),
        O: T.Tensor((N, KO), T.float16),
    ):
        with T.Kernel(tiles, threads=shaves) as tile:
            flash_phase(Q, Kc, Vt, Pos, O, tile, T.get_thread_binding(), None, 0)

    return attn_prefill


def gemma4_attn_prefill_reference(q, kc, vt, pos0: int, window: int | None = None):
    """numpy oracle of `make_gemma4_attn_prefill` (fp32). q is [NKV, N, GQ, D], kc [NKV, SMAX, D],
    vt [NKV, D, SMAX]; the output is token-major [N, NKV * GQ * D], the shape o_proj needs. Query token
    t of the call is at absolute position pos0 + t and sees keys j with j <= pos0 + t and (window is
    None or j > pos0 + t - window) -- `Gemma4Ref._attention`'s masking rule, no output gate."""
    import numpy as np

    NKV, N, GQ, D = q.shape
    out = np.zeros((N, NKV * GQ * D), np.float32)
    for g in range(NKV):
        for t in range(N):
            p = pos0 + t
            lo = 0 if window is None else max(0, p - window + 1)
            k = kc[g, lo : p + 1].astype(np.float32)  # [p + 1 - lo, D]
            v = vt[g, :, lo : p + 1].astype(np.float32)  # [D, p + 1 - lo]
            for hh in range(GQ):
                s = k @ q[g, t, hh].astype(np.float32) / np.sqrt(D)
                pr = np.exp(s - s.max())
                pr /= pr.sum()
                h = g * GQ + hh
                out[t, h * D : (h + 1) * D] = v @ pr
    return out
