"""Gemma-4-26B-A4B kernel building blocks for the Intel NPU backend.

See docs/developer_guide/gemma4_enablement_plan.md for the model architecture, the gap analysis
against the existing Qwen3.6 kernels in kernels.py, and the milestone log (V0 is this file's first
content: a flash-attention phase without Qwen3.6's output gate and with an optional sliding-window
lower bound, the one piece of the mixer that has no existing counterpart to reuse at all).

No `from __future__ import annotations`: T.prim_func evaluates the tensor annotations against the
factory's locals, same convention as kernels.py.
"""

import tilelang.intel_npu.language as T


def gemma4_norm_block(H: int, shaves: int, eps: float = 1e-6):
    """The residual stream's norms, as `T.macro`s over tile-shared CMX -- `kernels.norm_block`'s counterpart
    with Gemma4's PLAIN (non-zero-centered) convention: `rmsnorm(h32, w_cmx, x_cmx, shave)` writes
    `x = h rsqrt(mean h^2 + eps) w` in fp16 (SHAVE 0; tile sync after), `w_cmx=None` for an UNWEIGHTED norm
    (`post_attn`/`pre_ffn`/etc. all carry a real weight in Gemma4; only the attention V path is unweighted,
    and that lives in `make_gemma4_qkv_prefill`, not here). `add_rows`/`add_parts`/`scale_rows` are identical
    to `kernels.norm_block`'s (the residual-add and the final `layer_scalar` multiply have no norm-convention
    dependence). Returns `(rmsnorm, add_rows, add_parts, scale_rows)`."""
    FL = 16
    assert H % (shaves * FL) == 0
    HS = H // shaves

    @T.macro
    def rmsnorm(h32, w_cmx, x_cmx, shave):
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
                    scale = st[0] if w_cmx is None else st[0] * T.cast(w_cmx[0, d0 * FL + v], T.float32)
                    x_cmx[0, d0 * FL + v] = T.cast(h32[0, d0 * FL + v] * scale, T.float16)
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

    @T.macro
    def scale_rows(h32, scalar, shave):
        """`layer_scalar`'s final per-layer multiply: `h *= scalar` (a runtime fp32 scalar read from the
        layer table, not a Python constant)."""
        for d0 in T.serial(HS // FL):
            for v in T.vectorized(FL):
                h32[0, shave * HS + d0 * FL + v] = h32[0, shave * HS + d0 * FL + v] * scalar
        T.sync_threads()

    return rmsnorm, add_rows, add_parts, scale_rows


def make_gemma4_qkv_prefill(
    N: int,
    NH: int,
    NKV: int,
    D: int,
    SMAX: int,
    full: bool,
    TB: int = 8,
    tiles: int = 3,
    shaves: int = 2,
    eps: float = 1e-6,
    as_block: bool = False,
):
    """Gemma4's PREFILL phase between the qkv projection and the flash phase -- `kernels.make_attn_qkv_prefill`'s
    counterpart, with the three differences `Gemma4Ref._attention` has and Qwen3.6 does not:

      * NO output gate at all (Ys carries only Q | K (| V), never a doubled `2 * D` per head);
      * PLAIN (non-zero-centered) per-head RMSNorm: `x * rsqrt(mean(x**2) + eps) * w`, no `(1 + w)`;
      * V is ALWAYS an UNWEIGHTED RMSNorm (`rms_norm(raw, None, eps)`, no weight, no RoPE) of a raw projection
        that is SEPARATE for sliding layers (`v_proj`'s own output, read from Ys) but is the SAME raw K
        projection full-attention layers read for K itself (`full=True`: Ys carries no V section at all; V is
        derived in this phase from the K section BEFORE the weighted norm + RoPE that produces the real K).

    RoPE is Gemma4's own full-width rotate-half convention (`Gemma4Ref.apply_rope`/`rotate_half`), not Qwen's
    partial `ROT < D` rotary -- which is the SAME arithmetic as `make_attn_qkv_prefill`'s rotate-half with
    `ROT = D` (verified by direct derivation, see the plan doc section 2.1 and the module docstring above):
    `CosSin`'s layout must then be the full `[cos(D) | sin(D)]` per position (each half-width table
    `rope_cos_sin` returns, duplicated to full width at pack time), not npu_dsl's half-width `RopeS`/`RopeF`
    storage -- the duplication is a cheap pack-time transform, not a kernel difference.

    `full=True`: `Ys` is `[NP, NH*D + NKV*D]` (Q | K only). `full=False`: `Ys` is `[NP, NH*D + 2*NKV*D]` (Q | K | V).
    Oracle: `gemma4_qkv_prefill_reference`."""
    GQ = NH // NKV
    KOFF = NH * D
    VOFF = NH * D + NKV * D  # only used when not full
    NQKV = KOFF + NKV * D if full else KOFF + 2 * NKV * D
    NP = -(-N // TB) * TB
    NB = NP // TB
    workers = tiles * shaves
    FL, HL = 16, 32
    HR = D // 2
    assert NH % NKV == 0 and D % HL == 0 and D % 2 == 0

    @T.macro
    def qkv_phase(Ys, QNormW, KNormW, CosSin, Pos, Nt, Q, Kc, Vt, tile, shave):
        worker = tile * shaves + shave
        qw = T.alloc_shared((1, D), T.float16)
        kw = T.alloc_shared((1, D), T.float16)
        pos = T.alloc_shared((1, 1), T.int32)
        nt = T.alloc_shared((1, 1), T.int32)
        qg = T.alloc_local((TB, GQ * D), T.float16)  # one kv group's q for TB tokens: one DMA
        cs = T.alloc_local((TB, 2 * D), T.float32)
        qo = T.alloc_local((TB * GQ, D), T.float16)
        kb = T.alloc_local((TB, D), T.float16)  # K's raw projection; ALSO V's raw source when full
        vb = T.alloc_local((TB, D), T.float16)  # V's raw projection when not full
        vt = T.alloc_local((D, TB), T.float16)
        rowf = T.alloc_local((D,), T.float32)
        vf = T.alloc_local((D,), T.float32)  # the unweighted-norm V row, kept separate from K's weighted/roped row
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
                # --- the group's GQ query heads: plain weighted norm + full rotate-half RoPE --------------
                T.copy(Ys[b * TB, (g * GQ) * D], qg)
                for r, hh in T.grid(TB, GQ):
                    st[3] = T.if_then_else(b * TB + r < nt[0, 0], T.float32(1), T.float32(0))
                    for v in T.vectorized(FL):
                        ls[v] = T.float32(0)
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rowf[d0 * FL + v] = T.cast(qg[r, hh * D + d0 * FL + v], T.float32)
                            ls[v] = ls[v] + rowf[d0 * FL + v] * rowf[d0 * FL + v]
                    st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(D) + T.float32(eps))
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rowf[d0 * FL + v] = rowf[d0 * FL + v] * st[0] * T.cast(qw[0, d0 * FL + v], T.float32)
                    for rr in T.serial(HR):  # rotate-half on the full D dims (ROT = D)
                        st[1] = rowf[rr]
                        st[2] = rowf[HR + rr]
                        rowf[rr] = st[1] * cs[r, rr] - st[2] * cs[r, D + rr]
                        rowf[HR + rr] = st[2] * cs[r, HR + rr] + st[1] * cs[r, D + HR + rr]
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            qo[r * GQ + hh, d0 * FL + v] = T.cast(rowf[d0 * FL + v] * st[3], T.float16)
                T.copy(qo, Q[g, b * TB, 0, 0])
                # --- the group's key (and, if full, V's raw source): read once -----------------------------
                T.copy(Ys[b * TB, KOFF + g * D], kb)
                if not full:
                    T.copy(Ys[b * TB, VOFF + g * D], vb)
                for r in T.serial(TB):
                    st[3] = T.if_then_else(b * TB + r < nt[0, 0], T.float32(1), T.float32(0))
                    # --- V: unweighted RMSNorm of the raw source, no RoPE ------------------------------
                    for v in T.vectorized(FL):
                        ls[v] = T.float32(0)
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            vf[d0 * FL + v] = T.cast((kb if full else vb)[r, d0 * FL + v], T.float32)
                            ls[v] = ls[v] + vf[d0 * FL + v] * vf[d0 * FL + v]
                    st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(D) + T.float32(eps))
                    for d in T.serial(D):
                        vt[d, r] = T.cast(vf[d] * st[0] * st[3], T.float16)
                    # --- K: plain weighted RMSNorm + full rotate-half RoPE -----------------------------
                    for v in T.vectorized(FL):
                        ls[v] = T.float32(0)
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rowf[d0 * FL + v] = T.cast(kb[r, d0 * FL + v], T.float32)
                            ls[v] = ls[v] + rowf[d0 * FL + v] * rowf[d0 * FL + v]
                    st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(D) + T.float32(eps))
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            rowf[d0 * FL + v] = rowf[d0 * FL + v] * st[0] * T.cast(kw[0, d0 * FL + v], T.float32)
                    for rr in T.serial(HR):
                        st[1] = rowf[rr]
                        st[2] = rowf[HR + rr]
                        rowf[rr] = st[1] * cs[r, rr] - st[2] * cs[r, D + rr]
                        rowf[HR + rr] = st[2] * cs[r, HR + rr] + st[1] * cs[r, D + HR + rr]
                    for d0 in T.serial(D // FL):
                        for v in T.vectorized(FL):
                            kb[r, d0 * FL + v] = T.cast(rowf[d0 * FL + v] * st[3], T.float16)
                T.copy(kb, Kc[g, pos[0, 0] + b * TB, 0])
                T.copy(vt, Vt[g, 0, pos[0, 0] + b * TB])

    if as_block:
        return qkv_phase, dict(NP=NP, NB=NB, NQKV=NQKV, GQ=GQ, KOFF=KOFF, VOFF=VOFF)

    @T.prim_func
    def attn_qkv(
        Ys: T.Tensor((NP, NQKV), T.float16),
        QNormW: T.Tensor((1, D), T.float16),
        KNormW: T.Tensor((1, D), T.float16),
        CosSin: T.Tensor((NP, 2 * D), T.float32),
        Pos: T.Tensor((1, 1), T.int32),
        Nt: T.Tensor((1, 1), T.int32),
        Q: T.Tensor((NKV, NP, GQ, D), T.float16),
        Kc: T.Tensor((NKV, SMAX, D), T.float16),
        Vt: T.Tensor((NKV, D, SMAX), T.float16),
    ):
        with T.Kernel(tiles, threads=shaves) as tile:
            qkv_phase(Ys, QNormW, KNormW, CosSin, Pos, Nt, Q, Kc, Vt, tile, T.get_thread_binding())

    return attn_qkv


def gemma4_qkv_prefill_reference(y, qnorm_w, knorm_w, cos_sin, pos0: int, n: int, NH: int, NKV: int, D: int, full: bool,
                                 kc, vt, eps: float = 1e-6):
    """numpy oracle of `make_gemma4_qkv_prefill` (fp32). `y` is `[NP, NQKV]` (Q | K (| V)); `cos_sin` is
    `[NP, 2 * D]` (full-width cos then full-width sin, duplicated-half -- see the kernel docstring). Returns
    `(Q, kc, vt)` with the caches updated in place for positions `pos0 .. pos0 + n - 1`. `Q` is head-major
    `[NKV, NP, GQ, D]`. The weighted/unweighted-norm split and the full-attention V-from-raw-K rule are
    `Gemma4Ref._attention`'s exactly, so a kernel/oracle disagreement here is a real bug, not a convention gap."""
    import numpy as np

    NP = y.shape[0]
    GQ = NH // NKV
    KOFF = NH * D
    VOFF = NH * D + NKV * D
    Q = np.zeros((NKV, NP, GQ, D), np.float16)
    kc, vt = kc.copy(), vt.copy()

    def rope(v, cs):
        half = D // 2
        r = v[:half].copy()
        out = v.copy()
        out[:half] = r * cs[0, :half] - v[half:] * cs[1, :half]
        out[half:] = v[half:] * cs[0, half:] + r * cs[1, half:]
        return out

    def rmsnorm(v, w):
        s = v * (1.0 / np.sqrt((v * v).mean() + eps))
        return s if w is None else s * w.astype(np.float32).reshape(-1)

    for t in range(n):
        cs = cos_sin[t].astype(np.float32).reshape(2, D)
        for h in range(NH):
            q = rope(rmsnorm(y[t, h * D : (h + 1) * D].astype(np.float32), qnorm_w), cs)
            Q[h // GQ, t, h % GQ] = q.astype(np.float16)
        for g in range(NKV):
            raw_k = y[t, KOFF + g * D : KOFF + (g + 1) * D].astype(np.float32)
            raw_v = raw_k if full else y[t, VOFF + g * D : VOFF + (g + 1) * D].astype(np.float32)
            vt[g, :, pos0 + t] = rmsnorm(raw_v, None).astype(np.float16)
            kc[g, pos0 + t] = rope(rmsnorm(raw_k, knorm_w), cs).astype(np.float16)
    return Q, kc, vt


def gemma4_prefill_norm_block(H: int, NP: int, workers: int, eps: float = 1e-6, FL: int = 16,
                              in_dtype=T.float32, weighted: bool = True):
    """The prefill layer's PLAIN (non-zero-centered) RMSNorm over `NP` DDR-resident `in_dtype` rows,
    tokens dealt over the workers -- `kernels.prefill_norm_block`'s counterpart, `w` not `1 + w`.
    `Hio` is the residual stream (fp32 for the seven weighted residual-site norms; fp16 for `post_ffn_1`,
    which norms the dense FFN's fp16 output directly, not a copy promoted to fp32 first), `Xs` the fp16
    normed output a projection reads next. `weighted=False` is the router's `rms_norm(h1, None, eps) *
    router_scale` step -- same arithmetic, `Wnorm` still provided (it IS `router_scale`, just applied
    without being folded into a per-norm-site weight the way every other site's weight already is)."""
    assert NP % workers == 0 and H % FL == 0

    @T.macro
    def norm_tokens(Hio, Wnorm, Xs, worker):
        nw = T.alloc_local((1, H), T.float16)
        hrow = T.alloc_local((1, H), in_dtype)
        xrow = T.alloc_local((1, H), T.float16)
        ls = T.alloc_local((FL,), T.float32)
        st = T.alloc_local((1,), T.float32)
        if weighted:
            T.copy(Wnorm[0, 0], nw)
        for it in T.serial(NP // workers):
            t = it * workers + worker
            T.copy(Hio[t, 0], hrow)
            for v in T.vectorized(FL):
                ls[v] = T.float32(0)
            for d0 in T.serial(H // FL):
                for v in T.vectorized(FL):
                    ls[v] = ls[v] + T.cast(hrow[0, d0 * FL + v], T.float32) * T.cast(hrow[0, d0 * FL + v], T.float32)
            st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(H) + T.float32(eps))
            for d0 in T.serial(H // FL):
                for v in T.vectorized(FL):
                    scale = st[0] * T.cast(nw[0, d0 * FL + v], T.float32) if weighted else st[0]
                    xrow[0, d0 * FL + v] = T.cast(T.cast(hrow[0, d0 * FL + v], T.float32) * scale, T.float16)
            T.copy(xrow, Xs[t, 0])

    return norm_tokens


def gemma4_residual_block(H: int, NP: int, workers: int, FL: int = 16):
    """`add_tokens(Hio, A, worker)`: the fp32 residual stream `Hio[NP, H] += A[NP, H]` (fp16), dealt by token.
    `scale_tokens(Hio, Scalar, worker)`: `Hio *= Scalar[0, 0]` (a runtime fp32 scalar, `layer_scalar`)."""
    assert NP % workers == 0 and H % FL == 0

    @T.macro
    def add_tokens(Hio, A, worker):
        hrow = T.alloc_local((1, H), T.float32)
        arow = T.alloc_local((1, H), T.float16)
        for it in T.serial(NP // workers):
            t = it * workers + worker
            T.copy(Hio[t, 0], hrow)
            T.copy(A[t, 0], arow)
            for d0 in T.serial(H // FL):
                for v in T.vectorized(FL):
                    hrow[0, d0 * FL + v] = hrow[0, d0 * FL + v] + T.cast(arow[0, d0 * FL + v], T.float32)
            T.copy(hrow, Hio[t, 0])

    @T.macro
    def scale_tokens(Hio, Scalar, worker):
        hrow = T.alloc_local((1, H), T.float32)
        sc = T.alloc_local((1, 1), T.float32)
        T.copy(Scalar[0, 0], sc)
        for it in T.serial(NP // workers):
            t = it * workers + worker
            T.copy(Hio[t, 0], hrow)
            for d0 in T.serial(H // FL):
                for v in T.vectorized(FL):
                    hrow[0, d0 * FL + v] = hrow[0, d0 * FL + v] * sc[0, 0]
            T.copy(hrow, Hio[t, 0])

    return add_tokens, scale_tokens


def gemma4_add_norm_block(H: int, NP: int, workers: int, eps: float = 1e-6, FL: int = 16):
    """`combined = rmsnorm(dense_normed + moe_normed, post_ffn_norm)`: the one layer site that sums two
    fp16 DDR tensors (in fp32) before the plain weighted RMSNorm, instead of normalising a single
    residual-stream row. Returns `add_norm(A, B, Wnorm, Xs, worker)`."""
    assert NP % workers == 0 and H % FL == 0

    @T.macro
    def add_norm(A, B, Wnorm, Xs, worker):
        nw = T.alloc_local((1, H), T.float16)
        arow = T.alloc_local((1, H), T.float16)
        brow = T.alloc_local((1, H), T.float16)
        srow = T.alloc_local((1, H), T.float32)
        xrow = T.alloc_local((1, H), T.float16)
        ls = T.alloc_local((FL,), T.float32)
        st = T.alloc_local((1,), T.float32)
        T.copy(Wnorm[0, 0], nw)
        for it in T.serial(NP // workers):
            t = it * workers + worker
            T.copy(A[t, 0], arow)
            T.copy(B[t, 0], brow)
            for v in T.vectorized(FL):
                ls[v] = T.float32(0)
            for d0 in T.serial(H // FL):
                for v in T.vectorized(FL):
                    srow[0, d0 * FL + v] = T.cast(arow[0, d0 * FL + v], T.float32) + T.cast(brow[0, d0 * FL + v], T.float32)
                    ls[v] = ls[v] + srow[0, d0 * FL + v] * srow[0, d0 * FL + v]
            st[0] = T.float32(1) / T.sqrt(T.call_extern(T.float32, "tl_npu_vsum", ls[0:FL]) / T.float32(H) + T.float32(eps))
            for d0 in T.serial(H // FL):
                for v in T.vectorized(FL):
                    xrow[0, d0 * FL + v] = T.cast(srow[0, d0 * FL + v] * st[0] * T.cast(nw[0, d0 * FL + v], T.float32), T.float16)
            T.copy(xrow, Xs[t, 0])

    return add_norm


def make_gemma4_geglu_activation(N: int, FI: int, tiles: int = 3, shaves: int = 2, as_block: bool = False, FL: int = 16):
    """`Act[N, FI] = gelu_tanh(Gate[N, FI]) * Up[N, FI]`, tokens dealt over the workers. Algebraically
    simplified so it reuses the `exp`-based sigmoid pattern already proven on this backend (the attention
    output gate, the MoE router's sigmoid gate) rather than `T.tanh` (exists in the common TIR op set but
    unproven on this backend's SHAVE codegen path):

      gelu_tanh(x) = 0.5 x (1 + tanh(c (x + 0.044715 x^3))), c = sqrt(2 / pi)
                   = x * sigmoid(2 c (x + 0.044715 x^3))                         (derivation in the plan doc)

    See `Gemma4Ref`'s `F.gelu(..., approximate="tanh")` calls (the dense FFN and every expert's gate)."""
    workers = tiles * shaves
    assert N % workers == 0 and FI % FL == 0
    C2 = 2.0 * 0.7978845608028654  # 2 * sqrt(2 / pi)

    @T.macro
    def act_phase(Gate, Up, Act, tile, shave):
        worker = tile * shaves + shave
        g_cmx = T.alloc_local((1, FI), T.float16)
        u_cmx = T.alloc_local((1, FI), T.float16)
        a_cmx = T.alloc_local((1, FI), T.float16)
        for it in T.serial(N // workers):
            t = it * workers + worker
            T.copy(Gate[t, 0], g_cmx)
            T.copy(Up[t, 0], u_cmx)
            for d0 in T.serial(FI // FL):
                for v in T.vectorized(FL):
                    x = T.cast(g_cmx[0, d0 * FL + v], T.float32)
                    z2 = T.float32(C2) * (x + T.float32(0.044715) * x * x * x)
                    e = T.exp(T.float32(0) - z2)
                    gelu = x / (T.float32(1) + e)
                    a_cmx[0, d0 * FL + v] = T.cast(gelu * T.cast(u_cmx[0, d0 * FL + v], T.float32), T.float16)
            T.copy(a_cmx, Act[t, 0])

    if as_block:
        return act_phase

    @T.prim_func
    def geglu(Gate: T.Tensor((N, FI), T.float16), Up: T.Tensor((N, FI), T.float16), Act: T.Tensor((N, FI), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            act_phase(Gate, Up, Act, tile, T.get_thread_binding())

    return geglu


def gemma4_geglu_reference(gate, up):
    import numpy as np

    g = gate.astype(np.float32)
    half = 0.5 * g * (1.0 + np.tanh(0.7978845608028654 * (g + 0.044715 * g**3)))
    return (half * up.astype(np.float32)).astype(np.float16)


def make_gemma4_dense_ffn_prefill(N: int, H: int, FI: int, G: int = 64, BN: int = 32, MB: int = 8,
                                  tiles: int = 3, shaves: int = 2, as_block: bool = False):
    """The dense FFN every Gemma4 layer runs unconditionally on every token: `down(gelu_tanh(gate(x)) *
    up(x))`. Reuses `kernels.make_proj_prefill` for the three int4 group-G quantised GEMMs (the SAME
    dequant+fp16-DPU machinery the MoE expert FFN and the attention projections use) -- only the activation
    in between is Gemma4-specific (`make_gemma4_geglu_activation`). V0 simplification: gate and up are
    separate projections (not `pack.py`'s tile-interleaved single-DMA layout `kernels.py`'s MoE expert
    FFN uses) -- correct, not yet the efficient form; see the plan doc section 2.3."""
    from kernels import make_proj_prefill

    gate_phase, _ = make_proj_prefill(NP=N, M=FI, K=H, BN=BN, MB=MB, G=G, tiles=tiles, shaves=shaves, as_block=True, scales_gm=True)
    up_phase, _ = make_proj_prefill(NP=N, M=FI, K=H, BN=BN, MB=MB, G=G, tiles=tiles, shaves=shaves, as_block=True, scales_gm=True)
    down_phase, _ = make_proj_prefill(NP=N, M=H, K=FI, BN=BN, MB=MB, G=G, tiles=tiles, shaves=shaves, as_block=True, scales_gm=True)
    act_phase = make_gemma4_geglu_activation(N, FI, tiles=tiles, shaves=shaves, as_block=True)

    @T.macro
    def dense_ffn(Xn, WgQ, WgS, WuQ, WuS, WdQ, WdS, Gate, Up, Act, Dense, tile, shave):
        gate_phase(Xn, WgQ, WgS, Gate, tile, shave)
        T.sync_grid()  # gate/up/down have no DDR dependency on each other, so nothing derives this
        up_phase(Xn, WuQ, WuS, Up, tile, shave)  # barrier automatically; force it so their CMX doesn't co-scope
        T.sync_grid()
        act_phase(Gate, Up, Act, tile, shave)
        down_phase(Act, WdQ, WdS, Dense, tile, shave)

    if as_block:
        return dense_ffn

    KG = H // G

    @T.prim_func
    def dense(Xn: T.Tensor((N, H), T.float16), WgQ: T.Tensor((FI, H // 2), T.uint8), WgS: T.Tensor((KG, FI), T.float16),
             WuQ: T.Tensor((FI, H // 2), T.uint8), WuS: T.Tensor((KG, FI), T.float16),
             WdQ: T.Tensor((H, FI // 2), T.uint8), WdS: T.Tensor((FI // G, H), T.float16),
             Gate: T.Tensor((N, FI), T.float16), Up: T.Tensor((N, FI), T.float16), Act: T.Tensor((N, FI), T.float16),
             Dense: T.Tensor((N, H), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            dense_ffn(Xn, WgQ, WgS, WuQ, WuS, WdQ, WdS, Gate, Up, Act, Dense, tile, T.get_thread_binding())

    return dense


def gemma4_fp16_proj_prefill(NP: int, M: int, K: int, BN: int, MB: int, tiles: int = 3, shaves: int = 2, as_block: bool = False):
    """`Yout[NP, M] = X[NP, K] . W[M, K]^T`, plain fp16 weights (not quantised) -- Gemma4's router projection
    is small and stored fp16 in the device table, unlike every other projection in this model. A simple
    output-column-chunk round-robin, V0-grade (not amortising the weight read across token blocks the way
    `make_proj_prefill` does -- correctness over performance, see the plan doc)."""
    assert M % BN == 0 and NP % MB == 0
    NCH, NTB = M // BN, NP // MB
    workers = tiles * shaves

    @T.macro
    def proj_phase(X, W, Yout, tile, shave):
        worker = tile * shaves + shave
        w_cmx = T.alloc_local((BN, K), T.float16)
        x_cmx = T.alloc_local((MB, K), T.float16)
        y_cmx = T.alloc_local((MB, BN), T.float16)
        for it in T.serial(T.ceildiv(NCH, workers)):
            n = it * workers + worker
            if n < NCH:
                T.copy(W[n * BN, 0], w_cmx)
                for b in T.serial(NTB):
                    T.copy(X[b * MB, 0], x_cmx)
                    T.gemm(x_cmx, w_cmx, y_cmx, transpose_B=True, clear_accum=True)
                    T.copy(y_cmx, Yout[b * MB, n * BN])

    if as_block:
        return proj_phase

    @T.prim_func
    def proj(X: T.Tensor((NP, K), T.float16), W: T.Tensor((M, K), T.float16), Yout: T.Tensor((NP, M), T.float16)):
        with T.Kernel(tiles, threads=shaves) as tile:
            proj_phase(X, W, Yout, tile, T.get_thread_binding())

    return proj


def make_gemma4_moe_route_prefill(N: int, E: int, TOPK: int, tiles: int = 3, shaves: int = 2, as_block: bool = False):
    """Gemma4's router tail, AFTER the plain fp16 logits projection (`gemma4_fp16_proj_prefill`): top-`TOPK`
    of `E` by raw logit (mathematically identical to top-k of the softmax, see the plan doc section 5 item
    1), softmax over the chosen k only, renormalised, THEN multiplied by a gathered `PerExpertScale` --
    `Gemma4Ref._route`'s exact order, and with NO shared-expert row (every expert is an ordinary routed
    expert; `kernels.make_moe_route_prefill`'s `E + 1`-th universal row is Qwen3.6-only).

    V0-grade grouping: the histogram / prefix-sum / slot table is built by worker 0 alone, serially (same
    as `kernels.make_moe_route_prefill`'s phase C, minus the shared-expert row) -- correctness-first, not
    yet load-balanced over the workers. Oracle: `gemma4_moe_route_prefill_reference`."""
    HL = 32
    assert E % HL == 0 and TOPK <= 16
    SLOTS = N * TOPK
    workers = tiles * shaves

    @T.macro
    def route_phase(Logits, PerExpertScale, Ids, Wts, Off, Slot, tile, shave):
        worker = tile * shaves + shave
        lg_cmx = T.alloc_local((1, E), T.float16)
        idl = T.alloc_local((1, 16), T.int32)
        pick = T.alloc_local((16,), T.float32)
        best = T.alloc_local((2,), T.float32)
        mx = T.alloc_local((HL,), T.float16)
        pes = T.alloc_local((1, 1), T.float32)
        cnt_cmx = T.alloc_shared((E + 1,), T.int32)
        off_cmx = T.alloc_shared((E + 1,), T.int32)
        cur_cmx = T.alloc_shared((E,), T.int32)
        ids_all = T.alloc_shared((N, TOPK), T.int32)
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
                    for v in T.vectorized(HL):
                        mx[v] = T.if_then_else(lg_cmx[0, v] == T.cast(best[0], T.float16), T.cast(E - v, T.float16), T.float16(0))
                    for e0 in T.serial(1, E // HL):
                        for v in T.vectorized(HL):
                            mx[v] = T.max(
                                mx[v],
                                T.if_then_else(
                                    lg_cmx[0, e0 * HL + v] == T.cast(best[0], T.float16), T.cast(E - (e0 * HL + v), T.float16), T.float16(0)
                                ),
                            )
                    best[1] = T.float32(E) - T.cast(T.call_extern(T.float16, "tl_npu_vmax", mx[0:HL]), T.float32)
                    idl[0, k] = T.cast(best[1], T.int32)
                    pick[k] = best[0]
                    lg_cmx[0, idl[0, k]] = T.float16(-65504)
                best[0] = pick[0]  # the row max, saved before `pick` is overwritten by the exp below
                best[1] = T.float32(0)
                for k in T.serial(TOPK):
                    pick[k] = T.exp(pick[k] - best[0])
                    best[1] = best[1] + pick[k]
                for k in T.serial(TOPK):
                    T.copy(PerExpertScale[idl[0, k], 0], pes)
                    pick[k] = pick[k] / best[1] * pes[0, 0]
                T.copy(idl[0:1, 0:TOPK], Ids[t, 0])
                for k in T.serial(TOPK):
                    Wts[t, k] = pick[k]
        T.sync_grid()
        if worker == 0:
            T.copy(Ids[0, 0], ids_all)
            for e in T.serial(E + 1):
                cnt_cmx[e] = 0
            for t in T.serial(N):
                for k in T.serial(TOPK):
                    cnt_cmx[ids_all[t, k]] = cnt_cmx[ids_all[t, k]] + 1
            off_cmx[0] = 0
            for e in T.serial(E):
                off_cmx[e + 1] = off_cmx[e] + cnt_cmx[e]
                cur_cmx[e] = off_cmx[e]
            for t in T.serial(N):
                for k in T.serial(TOPK):
                    Slot[cur_cmx[ids_all[t, k]]] = t * TOPK + k
                    cur_cmx[ids_all[t, k]] = cur_cmx[ids_all[t, k]] + 1
            T.copy(off_cmx, Off[0])

    if as_block:
        return route_phase, dict(SLOTS=SLOTS)

    @T.prim_func
    def moe_route(Logits: T.Tensor((N, E), T.float16), PerExpertScale: T.Tensor((E, 1), T.float32),
                 Ids: T.Tensor((N, TOPK), T.int32), Wts: T.Tensor((N, TOPK), T.float32),
                 Off: T.Tensor((E + 1,), T.int32), Slot: T.Tensor((SLOTS,), T.int32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            route_phase(Logits, PerExpertScale, Ids, Wts, Off, Slot, tile, T.get_thread_binding())

    return moe_route


def gemma4_moe_route_prefill_reference(logits, per_expert_scale, E: int, TOPK: int):
    """numpy oracle of `make_gemma4_moe_route_prefill` (fp32): `Gemma4Ref._route`'s exact order (top-k by
    raw logit, softmax over the chosen k, renormalise, THEN multiply by the gathered per-expert scale --
    no shared-expert row)."""
    import numpy as np

    n = logits.shape[0]
    lg = logits.astype(np.float32).copy()
    ids = np.zeros((n, TOPK), np.int64)
    wts = np.zeros((n, TOPK), np.float32)
    for t in range(n):
        row = lg[t].copy()
        for k in range(TOPK):
            i = int(np.argmax(row))
            ids[t, k] = i
            wts[t, k] = row[i]
            row[i] = -1e30
        w = np.exp(wts[t] - wts[t, 0])
        w = w / w.sum()
        wts[t] = w * per_expert_scale.reshape(-1)[ids[t]]
    return ids, wts


def make_gemma4_moe_expert_prefill(N: int, H: int, I: int, E: int, TOPK: int, G: int = 64, MB: int = 16,
                                   BNW_gu: "int | None" = None, BNW_d: "int | None" = None,
                                   tiles: int = 3, shaves: int = 2, as_block: bool = False, FL: int = 16):
    """The grouped expert FFN for `N` prompt tokens, after `make_gemma4_moe_route_prefill`'s grouping
    (`Off`/`Slot`). V0-grade simplification vs `kernels.make_moe_expert_prefill` (see the plan doc section
    4): every WORKER (not every tile) owns whole experts round-robin -- no tile-shared weight buffers, no
    pipelining, no slot-major output -- so every buffer is `T.alloc_local` and no SHAVE-to-SHAVE
    coordination is needed at all within the expert loop. Correctness over performance.

    `BNW_gu` / `BNW_d` tile the gate|up output width (`I`) and the down output width (`H`) into chunks a
    SHAVE dequantises/computes one at a time, instead of the whole `(I, H)` / `(H, I)` weight matrix at
    once -- required at real model scale (`I` ~ 704, `H` ~ 2816: a single untiled `(I, H)` fp16 tile is
    ~3.96 MiB, far over one SHAVE's CMX share) and a no-op at V0's small test shapes (default `None` keeps
    the old untiled behaviour, one chunk the whole width). `I % BNW_gu == 0` and `H % BNW_d == 0` required.

    A token's `TOPK` choices land on different experts and therefore (generally) on different workers, so
    each worker accumulates its own expert(s)' contribution into a PRIVATE fp32 partial `Part[worker, N, H]`
    (zeroed at the start of this phase); `make_gemma4_moe_combine` sums the `workers` partials afterwards.
    Oracle: `gemma4_moe_expert_prefill_reference`."""
    assert H % G == 0 and I % G == 0 and H % FL == 0 and I % FL == 0
    BNW_gu = I if BNW_gu is None else BNW_gu
    BNW_d = H if BNW_d is None else BNW_d
    assert I % BNW_gu == 0 and H % BNW_d == 0
    workers = tiles * shaves
    EPT = -(-E // workers)
    KGH, KGI = H // G, I // G
    SLOTS = N * TOPK

    @T.macro
    def expert_phase(X, WgateQ, WgateS, WupQ, WupS, WdownQ, WdownS, Wts, Off, Slot, Part, tile, shave):
        worker = tile * shaves + shave
        off_cmx = T.alloc_local((E + 1,), T.int32)
        slot_cmx = T.alloc_local((SLOTS,), T.int32)
        wts_cmx = T.alloc_local((N, TOPK), T.float32)
        T.copy(Off, off_cmx)
        T.copy(Slot, slot_cmx)
        T.copy(Wts[0, 0], wts_cmx)
        zero_row = T.alloc_local((1, H), T.float32)
        for d0 in T.serial(H // FL):
            for v in T.vectorized(FL):
                zero_row[0, d0 * FL + v] = T.float32(0)
        for t in T.serial(N):
            T.copy(zero_row, Part[worker, t, 0])
        wgq_cmx = T.alloc_local((BNW_gu, H // 2), T.uint8)
        wgs_cmx = T.alloc_local((KGH, BNW_gu), T.float16)
        wg_cmx = T.alloc_local((BNW_gu, H), T.float16)
        wuq_cmx = T.alloc_local((BNW_gu, H // 2), T.uint8)
        wus_cmx = T.alloc_local((KGH, BNW_gu), T.float16)
        wu_cmx = T.alloc_local((BNW_gu, H), T.float16)
        wdq_cmx = T.alloc_local((BNW_d, I // 2), T.uint8)
        wds_cmx = T.alloc_local((KGI, BNW_d), T.float16)
        wd_cmx = T.alloc_local((BNW_d, I), T.float16)
        x_cmx = T.alloc_local((MB, H), T.float16)
        gate_chunk = T.alloc_local((MB, BNW_gu), T.float16)
        up_chunk = T.alloc_local((MB, BNW_gu), T.float16)
        act_cmx = T.alloc_local((MB, I), T.float16)
        y_cmx = T.alloc_local((MB, BNW_d), T.float16)
        acc = T.alloc_local((1, H), T.float32)
        for ei in T.serial(EPT):
            e = ei * workers + worker
            if e < E:
                for c in T.serial(T.ceildiv(off_cmx[e + 1] - off_cmx[e], MB)):
                    for r in T.serial(MB):
                        idx = off_cmx[e] + c * MB + r
                        if idx < off_cmx[e + 1]:
                            T.copy(X[slot_cmx[idx] // TOPK, 0:H], x_cmx[r, 0:H])
                        else:
                            for d0 in T.serial(H // FL):
                                for v in T.vectorized(FL):
                                    x_cmx[r, d0 * FL + v] = T.float16(0)
                    for gu in T.serial(I // BNW_gu):
                        T.copy(WgateQ[e, gu * BNW_gu, 0], wgq_cmx)
                        T.copy(WgateS[e, 0, gu * BNW_gu], wgs_cmx)
                        T.call_extern("handle", "tl_npu_dequant_i4_gm", T.address_of(wg_cmx[0, 0]),
                                     T.address_of(wgq_cmx[0, 0]), T.address_of(wgs_cmx[0, 0]), BNW_gu, H, G)
                        T.gemm(x_cmx, wg_cmx, gate_chunk, transpose_B=True, clear_accum=True)
                        T.copy(WupQ[e, gu * BNW_gu, 0], wuq_cmx)
                        T.copy(WupS[e, 0, gu * BNW_gu], wus_cmx)
                        T.call_extern("handle", "tl_npu_dequant_i4_gm", T.address_of(wu_cmx[0, 0]),
                                     T.address_of(wuq_cmx[0, 0]), T.address_of(wus_cmx[0, 0]), BNW_gu, H, G)
                        T.gemm(x_cmx, wu_cmx, up_chunk, transpose_B=True, clear_accum=True)
                        for r in T.serial(MB):
                            for d0 in T.serial(BNW_gu // FL):
                                for v in T.vectorized(FL):
                                    g = T.cast(gate_chunk[r, d0 * FL + v], T.float32)
                                    z2 = T.float32(1.5957691216057308) * (g + T.float32(0.044715) * g * g * g)
                                    e_ = T.exp(T.float32(0) - z2)
                                    act_cmx[r, gu * BNW_gu + d0 * FL + v] = T.cast(
                                        g / (T.float32(1) + e_) * T.cast(up_chunk[r, d0 * FL + v], T.float32), T.float16
                                    )
                    for dd in T.serial(H // BNW_d):
                        T.copy(WdownQ[e, dd * BNW_d, 0], wdq_cmx)
                        T.copy(WdownS[e, 0, dd * BNW_d], wds_cmx)
                        T.call_extern("handle", "tl_npu_dequant_i4_gm", T.address_of(wd_cmx[0, 0]),
                                     T.address_of(wdq_cmx[0, 0]), T.address_of(wds_cmx[0, 0]), BNW_d, I, G)
                        T.gemm(act_cmx, wd_cmx, y_cmx, transpose_B=True, clear_accum=True)
                        for r in T.serial(MB):
                            idx = off_cmx[e] + c * MB + r
                            if idx < off_cmx[e + 1]:
                                tok = slot_cmx[idx] // TOPK
                                kk = slot_cmx[idx] % TOPK
                                T.copy(Part[worker, tok, dd * BNW_d : dd * BNW_d + BNW_d], acc[0:1, 0:BNW_d])
                                for d0 in T.serial(BNW_d // FL):
                                    for v in T.vectorized(FL):
                                        acc[0, d0 * FL + v] = acc[0, d0 * FL + v] + T.cast(y_cmx[r, d0 * FL + v], T.float32) * wts_cmx[tok, kk]
                                T.copy(acc[0:1, 0:BNW_d], Part[worker, tok, dd * BNW_d : dd * BNW_d + BNW_d])

    if as_block:
        return expert_phase, dict(EPT=EPT, KGH=KGH, KGI=KGI)

    @T.prim_func
    def moe_expert(
        X: T.Tensor((N, H), T.float16),
        WgateQ: T.Tensor((E, I, H // 2), T.uint8), WgateS: T.Tensor((E, KGH, I), T.float16),
        WupQ: T.Tensor((E, I, H // 2), T.uint8), WupS: T.Tensor((E, KGH, I), T.float16),
        WdownQ: T.Tensor((E, H, I // 2), T.uint8), WdownS: T.Tensor((E, KGI, H), T.float16),
        Wts: T.Tensor((N, TOPK), T.float32), Off: T.Tensor((E + 1,), T.int32), Slot: T.Tensor((SLOTS,), T.int32),
        Part: T.Tensor((workers, N, H), T.float32),
    ):
        with T.Kernel(tiles, threads=shaves) as tile:
            expert_phase(X, WgateQ, WgateS, WupQ, WupS, WdownQ, WdownS, Wts, Off, Slot, Part, tile, T.get_thread_binding())

    return moe_expert


def gemma4_moe_expert_prefill_reference(x, ids, wts, gate_w, up_w, down_w):
    """numpy oracle of `make_gemma4_moe_expert_prefill` (fp32): `gate_w`/`up_w` are `[E, I, H]`, `down_w` is
    `[E, H, I]` (already dequantised); `ids`/`wts` are `[N, TOPK]`. `out[n] = sum_k wts[n,k] *
    down_{ids[n,k]}(gelu_tanh(gate_{ids[n,k]}(x[n])) * up_{ids[n,k]}(x[n]))`."""
    import numpy as np

    N, H = x.shape
    out = np.zeros((N, H), np.float32)
    for n in range(N):
        for k in range(ids.shape[1]):
            e = int(ids[n, k])
            gate = x[n].astype(np.float32) @ gate_w[e].astype(np.float32).T
            up = x[n].astype(np.float32) @ up_w[e].astype(np.float32).T
            act = gemma4_geglu_reference(gate.astype(np.float16), up.astype(np.float16)).astype(np.float32)
            y = act @ down_w[e].astype(np.float32).T
            out[n] += wts[n, k] * y
    return out


def make_gemma4_moe_combine(N: int, H: int, tiles: int = 3, shaves: int = 2, as_block: bool = False, FL: int = 16):
    """Sum the `workers` private partials from `make_gemma4_moe_expert_prefill` into `Out[N, H]` fp32,
    tokens dealt over the workers."""
    workers = tiles * shaves
    assert H % FL == 0

    @T.macro
    def combine_phase(Part, Out, tile, shave):
        worker = tile * shaves + shave
        acc = T.alloc_local((1, H), T.float32)
        row = T.alloc_local((1, H), T.float32)
        for it in T.serial(T.ceildiv(N, workers)):
            t = it * workers + worker
            if t < N:
                T.copy(Part[0, t, 0], acc)
                for w in T.serial(1, workers):
                    T.copy(Part[w, t, 0], row)
                    for d0 in T.serial(H // FL):
                        for v in T.vectorized(FL):
                            acc[0, d0 * FL + v] = acc[0, d0 * FL + v] + row[0, d0 * FL + v]
                T.copy(acc, Out[t, 0])

    if as_block:
        return combine_phase

    @T.prim_func
    def combine(Part: T.Tensor((workers, N, H), T.float32), Out: T.Tensor((N, H), T.float32)):
        with T.Kernel(tiles, threads=shaves) as tile:
            combine_phase(Part, Out, tile, T.get_thread_binding())

    return combine


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


def make_gemma4_layer_prefill(
    N: int, H: int, NH: int, NKV: int, D: int, FI: int, E: int, TOPK: int, I: int,
    G: int = 64, SMAX: int = 256, full: bool = False, window: "int | None" = None,
    TB: int = 4, BKV: int = 128, BN: int = 32, MB_expert: int = 4, MB_proj: "int | None" = None,
    BNW_gu: "int | None" = None, BNW_d: "int | None" = None,
    tiles: int = 3, shaves: int = 2, eps: float = 1e-6,
):
    """A WHOLE Gemma4 decoder layer in ONE launch, V0: one of the two layer KINDS (`full` selects which),
    composed from every macro above -- `Gemma4Ref.decoder_layer`'s exact phase order:

      x = rmsnorm(H, input_norm) -> qkv proj -> qkv_phase (norm/RoPE/V-derivation) -> flash attention
          -> o_proj -> h1 = H + rmsnorm(attn_o, post_attn_norm)
      dense_in = rmsnorm(h1, pre_ffn_norm) -> dense FFN -> dense_n = rmsnorm(dense, post_ffn_1_norm)
      route_in = rmsnorm(h1, router_scale) -> router logits -> route (top-k, per_expert_scale)
      moe_in = rmsnorm(h1, pre_ffn_2_norm) -> grouped expert FFN -> combine -> moe_n = rmsnorm(moe, post_ffn_2_norm)
      combined = rmsnorm(dense_n + moe_n, post_ffn_norm)
      H = (h1 + combined) * layer_scalar

    Every phase deals its work differently, so every boundary is a grid barrier (`DeriveIntelNpuGlobalSync`
    derives them from the DDR read/write dependencies below -- no manual `T.sync_grid()` in this macro).
    `CosSin` must be the full-width `[cos(D) | sin(D)]` duplicated-half table (see
    `make_gemma4_qkv_prefill`'s docstring). `H` is read AND written (the fp32 residual stream, mutated
    through the layer exactly as `Gemma4Ref.decoder_layer` mutates `hidden`/`h1` in place).
    Oracle: `gemma4_decoder_layer_reference`."""
    workers = tiles * shaves
    GQ = NH // NKV
    KO = NH * D
    MB_proj = N if MB_proj is None else MB_proj
    from kernels import make_proj_prefill

    qkv_phase, qs = make_gemma4_qkv_prefill(N=N, NH=NH, NKV=NKV, D=D, SMAX=SMAX, full=full, TB=TB,
                                            tiles=tiles, shaves=shaves, eps=eps, as_block=True)
    NQKV = qs["NQKV"]
    attn_phase, _ = make_gemma4_attn_prefill(N=N, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, window=window, TB=TB,
                                             BKV=BKV, tiles=tiles, shaves=shaves, as_block=True)
    qkv_proj, _ = make_proj_prefill(NP=N, M=NQKV, K=H, BN=BN, MB=MB_proj, G=G, tiles=tiles, shaves=shaves,
                                    as_block=True, scales_gm=True)
    o_proj, _ = make_proj_prefill(NP=N, M=H, K=KO, BN=BN, MB=MB_proj, G=G, tiles=tiles, shaves=shaves,
                                  as_block=True, scales_gm=True)
    dense_ffn = make_gemma4_dense_ffn_prefill(N=N, H=H, FI=FI, G=G, BN=BN, MB=MB_proj, tiles=tiles, shaves=shaves, as_block=True)
    router_proj = gemma4_fp16_proj_prefill(NP=N, M=E, K=H, BN=BN, MB=MB_proj, tiles=tiles, shaves=shaves, as_block=True)
    route_phase, rs = make_gemma4_moe_route_prefill(N=N, E=E, TOPK=TOPK, tiles=tiles, shaves=shaves, as_block=True)
    SLOTS = rs["SLOTS"]
    expert_phase, es = make_gemma4_moe_expert_prefill(N=N, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=MB_expert,
                                                      BNW_gu=BNW_gu, BNW_d=BNW_d, tiles=tiles, shaves=shaves, as_block=True)
    KGH_e, KGI_e = es["KGH"], es["KGI"]
    combine_phase = make_gemma4_moe_combine(N=N, H=H, tiles=tiles, shaves=shaves, as_block=True)

    norm_f32 = gemma4_prefill_norm_block(H, N, workers, eps, in_dtype=T.float32)
    norm_f16 = gemma4_prefill_norm_block(H, N, workers, eps, in_dtype=T.float16)
    add_tokens, scale_tokens = gemma4_residual_block(H, N, workers)
    add_norm = gemma4_add_norm_block(H, N, workers, eps)

    @T.macro
    def layer(
        Hio, InNorm, QNorm, KNorm, CosSin, PostAttnNorm, PreFfnNorm, PostFfn1Norm, PreFfn2Norm, PostFfn2Norm,
        PostFfnNorm, RouterScale, WqkvQ, WqkvS, WoQ, WoS, WgateQ, WgateS, WupQ, WupS, WdownQ, WdownS,
        WR, EWgateQ, EWgateS, EWupQ, EWupS, EWdownQ, EWdownS, PerExpertScale, LayerScalar, Kc, Vt, Pos, Nt,
        Xn, Ys, Q, AttnO, AttnP, PostAttnN, Xdense, Dense, DenseN, Gate, Up, Act,
        Xroute, Logits, Ids, Wts, Off, Slot, Xmoe, MoePart, MoeOut, MoeN, Combined,
        tile, shave,
    ):
        worker = tile * shaves + shave
        norm_f32(Hio, InNorm, Xn, worker)
        qkv_proj(Xn, WqkvQ, WqkvS, Ys, tile, shave)
        qkv_phase(Ys, QNorm, KNorm, CosSin, Pos, Nt, Q, Kc, Vt, tile, shave)
        attn_phase(Q, Kc, Vt, Pos, AttnO, tile, shave, None, 0)
        o_proj(AttnO, WoQ, WoS, AttnP, tile, shave)
        norm_f16(AttnP, PostAttnNorm, PostAttnN, worker)
        add_tokens(Hio, PostAttnN, worker)  # Hio now holds h1

        norm_f32(Hio, PreFfnNorm, Xdense, worker)
        dense_ffn(Xdense, WgateQ, WgateS, WupQ, WupS, WdownQ, WdownS, Gate, Up, Act, Dense, tile, shave)
        norm_f16(Dense, PostFfn1Norm, DenseN, worker)

        norm_f32(Hio, RouterScale, Xroute, worker)
        router_proj(Xroute, WR, Logits, tile, shave)
        route_phase(Logits, PerExpertScale, Ids, Wts, Off, Slot, tile, shave)

        norm_f32(Hio, PreFfn2Norm, Xmoe, worker)
        expert_phase(Xmoe, EWgateQ, EWgateS, EWupQ, EWupS, EWdownQ, EWdownS, Wts, Off, Slot, MoePart, tile, shave)
        combine_phase(MoePart, MoeOut, tile, shave)
        norm_f32(MoeOut, PostFfn2Norm, MoeN, worker)  # MoeOut is fp32 -- norm_f32 reads it directly

        add_norm(DenseN, MoeN, PostFfnNorm, Combined, worker)
        add_tokens(Hio, Combined, worker)
        scale_tokens(Hio, LayerScalar, worker)

    return layer, dict(NQKV=NQKV, GQ=GQ, KO=KO, SLOTS=SLOTS, KGH_e=KGH_e, KGI_e=KGI_e, workers=workers)


def gemma4_decoder_layer_reference(
    h, weights: dict, NH: int, NKV: int, D: int, full: bool, window: "int | None", eps: float = 1e-6,
):
    """numpy oracle of `make_gemma4_layer_prefill` (fp32) -- `Gemma4Ref.decoder_layer`'s exact math, with
    the model's OvIr-backed weight loading replaced by a plain `weights` dict of already-fp32 arrays (no
    OvIr/torch dependency, so this runs standalone against synthetic weights). `h` is `[N, H]`. Weight
    keys: `input_norm, post_attn_norm, pre_ffn_norm, post_ffn_1_norm, pre_ffn_2_norm, post_ffn_2_norm,
    post_ffn_norm, q_norm, k_norm, router_scale` (vectors); `wq [NH*D+(1|2)*NKV*D, H], wv [NKV*D, H]`
    (sliding only), `wo [H, NH*D]`, `wg [FI, H], wu [FI, H], wd [H, FI]`, `router_proj [E, H]`,
    `ewg [E, I, H], ewu [E, I, H], ewd [E, H, I]`, `per_expert_scale [E]`; `layer_scalar` (float);
    `cos_sin [N, 2 * D]` (full-width, duplicated-half, pos0 = 0 always here -- V0 has no KV-cache prefix)."""
    import numpy as np

    def rms(x, w=None):
        s = x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))
        return s if w is None else s * w.reshape(1, -1)

    def rope(x, cs):
        # x: [N, D]; cs: [N, 2D] (cos(D) | sin(D), duplicated-half)
        half = D // 2
        cos, sin = cs[:, :D], cs[:, D:]
        r1, r2 = x[:, :half], x[:, half:]
        out = np.empty_like(x)
        out[:, :half] = r1 * cos[:, :half] - r2 * sin[:, :half]
        out[:, half:] = r2 * cos[:, half:] + r1 * sin[:, half:]
        return out

    def gelu_tanh(x):
        return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))

    N = h.shape[0]
    GQ = NH // NKV
    w = weights
    xin = rms(h, w["input_norm"])
    q_all = (xin @ w["wq"][: NH * D].T).reshape(N, NH, D)
    raw_k = (xin @ w["wq"][NH * D : NH * D + NKV * D].T).reshape(N, NKV, D)
    if full:
        raw_v = raw_k
    else:
        raw_v = (xin @ w["wq"][NH * D + NKV * D : NH * D + 2 * NKV * D].T).reshape(N, NKV, D)
    q = np.stack([rope(rms(q_all[:, hh], w["q_norm"]), w["cos_sin"]) for hh in range(NH)], axis=1)
    k = np.stack([rope(rms(raw_k[:, g], w["k_norm"]), w["cos_sin"]) for g in range(NKV)], axis=1)
    v = np.stack([rms(raw_v[:, g]) for g in range(NKV)], axis=1)
    k_rep = np.repeat(k, GQ, axis=1)
    v_rep = np.repeat(v, GQ, axis=1)
    context = np.zeros((N, NH, D), np.float32)
    for t in range(N):
        lo = 0 if window is None else max(0, t - window + 1)
        for hh in range(NH):
            s = k_rep[lo : t + 1, hh] @ q[t, hh] / np.sqrt(D)
            pr = np.exp(s - s.max())
            pr /= pr.sum()
            context[t, hh] = v_rep[lo : t + 1, hh].T @ pr
    attn_o = context.reshape(N, NH * D) @ w["wo"].T
    h1 = h + rms(attn_o, w["post_attn_norm"])

    dense_in = rms(h1, w["pre_ffn_norm"])
    gate = gelu_tanh(dense_in @ w["wg"].T)
    up = dense_in @ w["wu"].T
    dense = (gate * up) @ w["wd"].T
    dense_n = rms(dense, w["post_ffn_1_norm"])

    route_in = rms(h1, None) * w["router_scale"].reshape(1, -1)
    logits = route_in @ w["router_proj"].T
    E = logits.shape[1]
    moe_in = rms(h1, w["pre_ffn_2_norm"])
    moe = np.zeros((N, h.shape[1]), np.float32)
    for t in range(N):
        row = logits[t].copy()
        TOPK = w["ids_topk"]
        ids = []
        vals = []
        for _ in range(TOPK):
            i = int(np.argmax(row))
            ids.append(i)
            vals.append(row[i])
            row[i] = -1e30
        vals = np.array(vals)
        wts = np.exp(vals - vals[0])
        wts = wts / wts.sum() * w["per_expert_scale"][ids]
        for k_, e in enumerate(ids):
            g_ = gelu_tanh(moe_in[t] @ w["ewg"][e].T)
            u_ = moe_in[t] @ w["ewu"][e].T
            y_ = (g_ * u_) @ w["ewd"][e].T
            moe[t] += wts[k_] * y_
    moe_n = rms(moe, w["post_ffn_2_norm"])

    combined = rms(dense_n + moe_n, w["post_ffn_norm"])
    return (h1 + combined) * w["layer_scalar"]

