"""FG-CLIP2 so400m building blocks, V0 (plan: docs/developer_guide/fgclip2_enablement_plan.md; model walkthrough:
docs/developer_guide/fgclip2_npu_dsl_explained.md). Ported from npu_dsl's `fgclip2_layers_source.py` /
`fgclip2_e2e/reference.py`.

V0 deliberately skips every packing trick the original npu_dsl implementation uses (head-dim padding to a DPU
multiple of 16, q prescale, k-bias drop, v-bias-into-o_proj fold, LayerNorm-into-GEMM fold, broadcast/ping-pong
GEMM scheduling, flash-attention chunking) and every weight-stacking trick (q/k/v in one GEMM). The point of V0
is a literal, unoptimised port of `reference.py::encoder_layer`'s math -- one worker, one tile, full resident
buffers, a straightforward softmax instead of an online one (the row counts here are small enough that nothing
needs to stream) -- so a numpy oracle can be a near-verbatim transcription of the reference, not a re-derivation,
and any mismatch is a bug in the port rather than in an optimisation. Tiling for real model sizes (hidden=1152,
heads=16, head_dim=72->80, inter=4304->4320, 27 layers) is V1 (see the plan doc); the broadcast/ping-pong/ring
scheduling tricks are performance work for V5, after device validation, not before.
"""

import math

import tilelang.intel_npu.language as T

GELU_TANH_C = math.sqrt(2.0 / math.pi)


@T.macro
def layernorm_bias_row(x32, w_cmx, b_cmx, out_cmx, r, H: int, eps: float):
    """out[r, :] = (x32[r, :] - mean) * rsqrt(var + eps) * w + b, fp32 in (one row of a resident [rows, H]
    buffer) / fp16 out; w_cmx / b_cmx are [1, H] (shared across rows, loaded once by the caller)."""
    st = T.alloc_local((2,), T.float32)
    st[0] = T.float32(0)
    st[1] = T.float32(0)
    for d in T.serial(H):
        st[0] = st[0] + x32[r, d]
        st[1] = st[1] + x32[r, d] * x32[r, d]
    mean = st[0] / T.float32(H)
    var = st[1] / T.float32(H) - mean * mean
    invstd = T.float32(1) / T.sqrt(var + T.float32(eps))
    for d in T.serial(H):
        out_cmx[r, d] = T.cast((x32[r, d] - mean) * invstd * T.cast(w_cmx[0, d], T.float32) + T.cast(b_cmx[0, d], T.float32),
                               T.float16)


@T.macro
def linear_bias(x_cmx, w_cmx, b_cmx, y_cmx, rows: int, N: int, gelu: bool):
    """y[0:rows, 0:N] = x[0:rows, :] . w[0:N, :]^T + b, optionally GELU(tanh) (reference.py's `gelu_tanh`, the
    tanh approximation -- NOT erf). No N/K tiling: V0's shapes are small enough that every operand is resident."""
    T.gemm(x_cmx, w_cmx, y_cmx, transpose_B=True, clear_accum=True)
    for r in T.serial(rows):
        for n in T.serial(N):
            v = T.cast(y_cmx[r, n], T.float32) + T.cast(b_cmx[0, n], T.float32)
            if gelu:
                inner = T.float32(GELU_TANH_C) * (v + T.float32(0.044715) * v * v * v)
                v = T.float32(0.5) * v * (T.float32(1) + T.tanh(inner))
            y_cmx[r, n] = T.cast(v, T.float16)


@T.macro
def residual_add_fp32(h32, y_cmx, rows: int, N: int):
    """h32[0:rows, 0:N] += y_cmx[0:rows, 0:N] (the fp32 residual stream accumulates an fp16 update)."""
    for r in T.serial(rows):
        for n in T.serial(N):
            h32[r, n] = h32[r, n] + T.cast(y_cmx[r, n], T.float32)


@T.macro
def mha_full(q16, k16, v16, o16, s_cmx, qh, kh, vth, oh, mx, sm, rows: int, heads: int, D: int, scale: float):
    """o16[:, h*D:(h+1)*D] = softmax(q_h . k_h^T * scale) . v_h for every head h (`heads` is a Python int: the
    per-head loop is unrolled at trace time, so every CMX slice offset below is a compile-time literal) -- full,
    UNMASKED attention (no causal mask, no padding: reference.py's `attention`, every row attends every key).
    q16/k16/v16/o16 are [rows, heads*D] fp16; qh/kh/oh [rows, D], vth [D, rows] and s_cmx [rows, rows] are
    scratch the caller allocates once (so the pipeline packer can reuse the bytes across heads)."""
    for h in range(heads):
        for r in T.serial(rows):
            for d in T.serial(D):
                qh[r, d] = q16[r, h * D + d]
                kh[r, d] = k16[r, h * D + d]
                vth[d, r] = v16[r, h * D + d]
        T.gemm(qh, kh, s_cmx, transpose_B=True, clear_accum=True)
        for r in T.serial(rows):
            mx[0] = T.float32(-1e30)
            for j in T.serial(rows):
                mx[0] = T.max(mx[0], T.cast(s_cmx[r, j], T.float32) * T.float32(scale))
            sm[0] = T.float32(0)
            for j in T.serial(rows):
                p = T.exp(T.cast(s_cmx[r, j], T.float32) * T.float32(scale) - mx[0])
                s_cmx[r, j] = T.cast(p, T.float16)
                sm[0] = sm[0] + p
            for j in T.serial(rows):
                s_cmx[r, j] = T.cast(T.cast(s_cmx[r, j], T.float32) / sm[0], T.float16)
        T.gemm(s_cmx, vth, oh, transpose_B=True, clear_accum=True)
        for r in T.serial(rows):
            for d in T.serial(D):
                o16[r, h * D + d] = oh[r, d]


def make_fgclip2_encoder_layer_v0(rows: int, hidden: int, heads: int, head_dim: int, inter: int, eps: float = 1e-6):
    """One pre-LN FG-CLIP2 encoder layer (`reference.py::encoder_layer`, literally): LN1 -> separate q/k/v
    projections (no Wqkv stacking yet) -> full MHA (no mask) -> o_proj -> residual -> LN2 -> fc1+GELU(tanh) ->
    fc2 -> residual. One tile, one working SHAVE (`shave == 0`); everything resident in CMX (no row/weight
    tiling) -- V0 is a correctness check of the building blocks, not a performance kernel. `Hio` is the fp32
    residual stream, read at entry and overwritten at exit (matches the Qwen layers' in-place residual
    convention elsewhere in this backend)."""
    assert hidden == heads * head_dim, (hidden, heads, head_dim)
    scale = head_dim ** -0.5

    @T.prim_func
    def layer(Hio: T.Tensor((rows, hidden), T.float32),
              Ln1W: T.Tensor((1, hidden), T.float16), Ln1B: T.Tensor((1, hidden), T.float16),
              Wq: T.Tensor((hidden, hidden), T.float16), Bq: T.Tensor((1, hidden), T.float16),
              Wk: T.Tensor((hidden, hidden), T.float16), Bk: T.Tensor((1, hidden), T.float16),
              Wv: T.Tensor((hidden, hidden), T.float16), Bv: T.Tensor((1, hidden), T.float16),
              Wo: T.Tensor((hidden, hidden), T.float16), Bo: T.Tensor((1, hidden), T.float16),
              Ln2W: T.Tensor((1, hidden), T.float16), Ln2B: T.Tensor((1, hidden), T.float16),
              W1: T.Tensor((inter, hidden), T.float16), B1: T.Tensor((1, inter), T.float16),
              W2: T.Tensor((hidden, inter), T.float16), B2: T.Tensor((1, hidden), T.float16)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            h32 = T.alloc_local((rows, hidden), T.float32)
            ln1w = T.alloc_local((1, hidden), T.float16)
            ln1b = T.alloc_local((1, hidden), T.float16)
            ln2w = T.alloc_local((1, hidden), T.float16)
            ln2b = T.alloc_local((1, hidden), T.float16)
            xn = T.alloc_local((rows, hidden), T.float16)
            wq = T.alloc_local((hidden, hidden), T.float16)
            wk = T.alloc_local((hidden, hidden), T.float16)
            wv = T.alloc_local((hidden, hidden), T.float16)
            wo = T.alloc_local((hidden, hidden), T.float16)
            bq = T.alloc_local((1, hidden), T.float16)
            bk = T.alloc_local((1, hidden), T.float16)
            bv = T.alloc_local((1, hidden), T.float16)
            bo = T.alloc_local((1, hidden), T.float16)
            qp = T.alloc_local((rows, hidden), T.float16)
            kp = T.alloc_local((rows, hidden), T.float16)
            vp = T.alloc_local((rows, hidden), T.float16)
            ao = T.alloc_local((rows, hidden), T.float16)
            aop = T.alloc_local((rows, hidden), T.float16)
            s_cmx = T.alloc_local((rows, rows), T.float16)
            qh = T.alloc_local((rows, head_dim), T.float16)
            kh = T.alloc_local((rows, head_dim), T.float16)
            vth = T.alloc_local((head_dim, rows), T.float16)
            oh = T.alloc_local((rows, head_dim), T.float16)
            mx = T.alloc_local((1,), T.float32)
            sm = T.alloc_local((1,), T.float32)
            w1 = T.alloc_local((inter, hidden), T.float16)
            b1 = T.alloc_local((1, inter), T.float16)
            w2 = T.alloc_local((hidden, inter), T.float16)
            b2 = T.alloc_local((1, hidden), T.float16)
            act = T.alloc_local((rows, inter), T.float16)
            mlp_out = T.alloc_local((rows, hidden), T.float16)

            if shave == 0:
                T.copy(Hio[0, 0], h32)
                T.copy(Ln1W[0, 0], ln1w)
                T.copy(Ln1B[0, 0], ln1b)
                T.copy(Ln2W[0, 0], ln2w)
                T.copy(Ln2B[0, 0], ln2b)
                T.copy(Wq[0, 0], wq)
                T.copy(Bq[0, 0], bq)
                T.copy(Wk[0, 0], wk)
                T.copy(Bk[0, 0], bk)
                T.copy(Wv[0, 0], wv)
                T.copy(Bv[0, 0], bv)
                T.copy(Wo[0, 0], wo)
                T.copy(Bo[0, 0], bo)
                T.copy(W1[0, 0], w1)
                T.copy(B1[0, 0], b1)
                T.copy(W2[0, 0], w2)
                T.copy(B2[0, 0], b2)

                for r in T.serial(rows):
                    layernorm_bias_row(h32, ln1w, ln1b, xn, r, hidden, eps)
                linear_bias(xn, wq, bq, qp, rows, hidden, False)
                linear_bias(xn, wk, bk, kp, rows, hidden, False)
                linear_bias(xn, wv, bv, vp, rows, hidden, False)
                mha_full(qp, kp, vp, ao, s_cmx, qh, kh, vth, oh, mx, sm, rows, heads, head_dim, scale)
                linear_bias(ao, wo, bo, aop, rows, hidden, False)
                residual_add_fp32(h32, aop, rows, hidden)

                for r in T.serial(rows):
                    layernorm_bias_row(h32, ln2w, ln2b, xn, r, hidden, eps)
                linear_bias(xn, w1, b1, act, rows, inter, True)
                linear_bias(act, w2, b2, mlp_out, rows, hidden, False)
                residual_add_fp32(h32, mlp_out, rows, hidden)

                T.copy(h32, Hio[0, 0])
            T.sync_threads()

    return layer


def encoder_layer_reference(x, ln1w, ln1b, wq, bq, wk, bk, wv, bv, wo, bo, ln2w, ln2b, w1, b1, w2, b2,
                            heads: int, head_dim: int, eps: float = 1e-6):
    """numpy oracle of `make_fgclip2_encoder_layer_v0`, a transcription of `fgclip2_e2e/reference.py`'s
    `layer_norm` / `gelu_tanh` / `linear` / `attention` / `encoder_layer` (no torch dependency). `x` is
    [rows, hidden] fp32; every weight is [out, in] (`linear`'s `x @ W.T + b` convention); returns [rows, hidden]
    fp32. Every op computed in fp64 except the GEMM operands (fp16, matching the kernel's weight/activation
    dtype) so the comparison isolates the kernel's own fp16 rounding rather than adding the oracle's own."""
    import numpy as np

    def f16(a):
        return a.astype(np.float16).astype(np.float64)

    def layer_norm(v, w, b):
        mu = v.mean(-1, keepdims=True)
        var = ((v - mu) ** 2).mean(-1, keepdims=True)
        return (v - mu) / np.sqrt(var + eps) * w + b

    def gelu_tanh(v):
        return 0.5 * v * (1.0 + np.tanh(GELU_TANH_C * (v + 0.044715 * v ** 3)))

    def linear(v, w, b):
        return f16(v) @ f16(w).T + b.astype(np.float64)

    def attention(q, k, v):
        d = q.shape[-1] // heads
        qh = q.reshape(q.shape[0], heads, d).transpose(1, 0, 2)
        kh = k.reshape(k.shape[0], heads, d).transpose(1, 0, 2)
        vh = v.reshape(v.shape[0], heads, d).transpose(1, 0, 2)
        s = f16(qh) @ f16(kh).transpose(0, 2, 1) * d ** -0.5
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        o = f16(p) @ f16(vh)
        return o.transpose(1, 0, 2).reshape(q.shape[0], -1)

    x = x.astype(np.float64)
    h = layer_norm(x, ln1w.astype(np.float64), ln1b.astype(np.float64))
    a = attention(linear(h, wq, bq), linear(h, wk, bk), linear(h, wv, bv))
    x = x + linear(a, wo, bo)
    h = layer_norm(x, ln2w.astype(np.float64), ln2b.astype(np.float64))
    return x + linear(gelu_tanh(linear(h, w1, b1)), w2, b2)


# ---------------------------------------------------------------------------------------------------------------------
# V1 (plan: docs/developer_guide/fgclip2_enablement_plan.md): real model proportions, one layer
# ---------------------------------------------------------------------------------------------------------------------
@T.macro
def linear_bias_tiled(x_cmx, Wddr, b_cmx, y_cmx, w_tile, y_tile, rows: int, N: int, K: int, TN: int, gelu: bool):
    """y[0:rows, 0:N] = x[0:rows, 0:K] . W[0:N, 0:K]^T + b (b_cmx is [1, N], resident -- bias is small enough
    not to need tiling), optionally GELU(tanh). V1's difference from V0's `linear_bias`: at real model widths
    (hidden=1152, Q/K/V width 16*80=1280, inter_pad=4320) the weight matrix no longer fits CMX resident whole,
    so it streams TN-row tiles from DDR one at a time through `w_tile` ([TN, K], caller-allocated so the same
    buffer is reused across every linear layer whose K matches). No K-tiling yet (every K here still fits
    resident as part of the activation block) -- that is only needed once CMX is tighter than this (real NVL
    budget) or K grows further (not expected for this model)."""
    for n0 in T.serial(N // TN):
        T.copy(Wddr[n0 * TN, 0], w_tile)
        T.gemm(x_cmx, w_tile, y_tile, transpose_B=True, clear_accum=True)
        for r in T.serial(rows):
            for nn in T.serial(TN):
                v = T.cast(y_tile[r, nn], T.float32) + T.cast(b_cmx[0, n0 * TN + nn], T.float32)
                if gelu:
                    inner = T.float32(GELU_TANH_C) * (v + T.float32(0.044715) * v * v * v)
                    v = T.float32(0.5) * v * (T.float32(1) + T.tanh(inner))
                y_cmx[r, n0 * TN + nn] = T.cast(v, T.float16)


def make_fgclip2_encoder_layer_v1(rows: int, hidden: int, heads: int, hd_pad: int, inter_pad: int, tn: int = 32,
                                  eps: float = 1e-6):
    """One pre-LN FG-CLIP2 encoder layer at REAL model proportions (hidden=1152, heads=16, hd_pad=80, inter_pad=4320
    are the intended callers; `rows` stays small -- a handful of tokens -- because V1 is about real CHANNEL widths,
    not real token counts yet, and attention here is still the same plain (non-chunked) softmax as V0, which needs
    all `rows` keys resident at once).

    `hd_pad` is the kernel's only notion of head dimension: it never sees the real head_dim=72, only the
    DPU-aligned pad to 80 (see fgclip2_npu_dsl_explained.md section 6). Zero-padding the extra 8 channels so this
    is numerically IDENTICAL to a real head_dim=72 model is pack.py's job (V4), not tested here -- this kernel
    and its oracle both operate on the padded width end to end, so this test validates the real-shape TILED GEMM
    plumbing (the new thing since V0), not the padding-is-exact claim.

    V0 -> V1's one structural change: every linear layer (`linear_bias` -> `linear_bias_tiled`) streams its
    weight in row tiles from DDR instead of loading the whole matrix resident -- required once CMX can no
    longer hold a [1152, 1152] or [4320, 1152] fp16 matrix whole. Q/K/V/attention-output width (`QW =
    heads * hd_pad = 1280`) is now distinct from the residual width `hidden` (1152); `linear_bias`/`mha_full`
    already supported this (the oracle, `encoder_layer_reference`, already does -- it never assumed square
    weights). Still one layer, one tile, one working SHAVE; the device-side 27-layer loop is the next increment.

    All scratch is `alloc_shared`, not `alloc_local`: only `shave == 0` ever touches it, but `local` reserves
    its bytes PER SHAVE regardless of which branch runs, so declaring these buffers `local` silently doubled
    the CMX footprint and blew the budget on the first real run here (`CheckIntelNpuBudgets`: 1,995,136 B
    needed > 1,473,536 B available, because the per-SHAVE total was counted twice). `shared` is one copy per
    tile, which is what a single-active-SHAVE kernel actually needs.
    """
    QW = heads * hd_pad
    assert hidden % tn == 0 and QW % tn == 0 and inter_pad % tn == 0 and tn % 16 == 0 and rows % 16 == 0

    @T.prim_func
    def layer(Hio: T.Tensor((rows, hidden), T.float32),
              Ln1W: T.Tensor((1, hidden), T.float16), Ln1B: T.Tensor((1, hidden), T.float16),
              Wq: T.Tensor((QW, hidden), T.float16), Bq: T.Tensor((1, QW), T.float16),
              Wk: T.Tensor((QW, hidden), T.float16), Bk: T.Tensor((1, QW), T.float16),
              Wv: T.Tensor((QW, hidden), T.float16), Bv: T.Tensor((1, QW), T.float16),
              Wo: T.Tensor((hidden, QW), T.float16), Bo: T.Tensor((1, hidden), T.float16),
              Ln2W: T.Tensor((1, hidden), T.float16), Ln2B: T.Tensor((1, hidden), T.float16),
              W1: T.Tensor((inter_pad, hidden), T.float16), B1: T.Tensor((1, inter_pad), T.float16),
              W2: T.Tensor((hidden, inter_pad), T.float16), B2: T.Tensor((1, hidden), T.float16)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            h32 = T.alloc_shared((rows, hidden), T.float32)
            ln1w = T.alloc_shared((1, hidden), T.float16)
            ln1b = T.alloc_shared((1, hidden), T.float16)
            ln2w = T.alloc_shared((1, hidden), T.float16)
            ln2b = T.alloc_shared((1, hidden), T.float16)
            xn = T.alloc_shared((rows, hidden), T.float16)
            bq = T.alloc_shared((1, QW), T.float16)
            bk = T.alloc_shared((1, QW), T.float16)
            bv = T.alloc_shared((1, QW), T.float16)
            bo = T.alloc_shared((1, hidden), T.float16)
            b1 = T.alloc_shared((1, inter_pad), T.float16)
            b2 = T.alloc_shared((1, hidden), T.float16)
            qp = T.alloc_shared((rows, QW), T.float16)
            kp = T.alloc_shared((rows, QW), T.float16)
            vp = T.alloc_shared((rows, QW), T.float16)
            ao = T.alloc_shared((rows, QW), T.float16)
            aop = T.alloc_shared((rows, hidden), T.float16)
            act = T.alloc_shared((rows, inter_pad), T.float16)
            mlp_out = T.alloc_shared((rows, hidden), T.float16)
            w_tile_h = T.alloc_shared((tn, hidden), T.float16)       # K == hidden: q/k/v/fc1
            w_tile_q = T.alloc_shared((tn, QW), T.float16)           # K == QW: o_proj
            w_tile_i = T.alloc_shared((tn, inter_pad), T.float16)    # K == inter_pad: fc2
            y_tile = T.alloc_shared((rows, tn), T.float16)
            s_cmx = T.alloc_shared((rows, rows), T.float16)
            qh = T.alloc_shared((rows, hd_pad), T.float16)
            kh = T.alloc_shared((rows, hd_pad), T.float16)
            vth = T.alloc_shared((hd_pad, rows), T.float16)
            oh = T.alloc_shared((rows, hd_pad), T.float16)
            mx = T.alloc_shared((1,), T.float32)
            sm = T.alloc_shared((1,), T.float32)

            if shave == 0:
                T.copy(Hio[0, 0], h32)
                T.copy(Ln1W[0, 0], ln1w)
                T.copy(Ln1B[0, 0], ln1b)
                T.copy(Ln2W[0, 0], ln2w)
                T.copy(Ln2B[0, 0], ln2b)
                T.copy(Bq[0, 0], bq)
                T.copy(Bk[0, 0], bk)
                T.copy(Bv[0, 0], bv)
                T.copy(Bo[0, 0], bo)
                T.copy(B1[0, 0], b1)
                T.copy(B2[0, 0], b2)

                for r in T.serial(rows):
                    layernorm_bias_row(h32, ln1w, ln1b, xn, r, hidden, eps)
                linear_bias_tiled(xn, Wq, bq, qp, w_tile_h, y_tile, rows, QW, hidden, tn, False)
                linear_bias_tiled(xn, Wk, bk, kp, w_tile_h, y_tile, rows, QW, hidden, tn, False)
                linear_bias_tiled(xn, Wv, bv, vp, w_tile_h, y_tile, rows, QW, hidden, tn, False)
                mha_full(qp, kp, vp, ao, s_cmx, qh, kh, vth, oh, mx, sm, rows, heads, hd_pad, hd_pad ** -0.5)
                linear_bias_tiled(ao, Wo, bo, aop, w_tile_q, y_tile, rows, hidden, QW, tn, False)
                residual_add_fp32(h32, aop, rows, hidden)

                for r in T.serial(rows):
                    layernorm_bias_row(h32, ln2w, ln2b, xn, r, hidden, eps)
                linear_bias_tiled(xn, W1, b1, act, w_tile_h, y_tile, rows, inter_pad, hidden, tn, True)
                linear_bias_tiled(act, W2, b2, mlp_out, w_tile_i, y_tile, rows, hidden, inter_pad, tn, False)
                residual_add_fp32(h32, mlp_out, rows, hidden)

                T.copy(h32, Hio[0, 0])
            T.sync_threads()

    return layer


# ---------------------------------------------------------------------------------------------------------------------
# V1, continued: a device-side loop over N layers through a weight table (the other half of V1 -- see
# fgclip2_enablement_plan.md). Reuses Qwen's layer-table design (`kernels.py`'s `TABLE_COLS`/`GDN_TABLE`/
# `T.ddr_view`, `make_decode_model`) rather than re-deriving it: one `Table[n_layers, NCOL]` of device
# addresses, the host fills one row per layer, the kernel loops `for li in T.serial(n_layers):` and reads
# `T.ddr_view(Table[li, col["Wq"]], ...)` for that layer's weights -- a static Python unroll of 27 copies of
# the layer body would also work but bloats `.text` for no benefit (every layer is the identical op sequence,
# only the weight addresses differ), which is exactly why Qwen's M5 did it this way for its 40 layers.
# ---------------------------------------------------------------------------------------------------------------------
FGCLIP2_TABLE_COLS = ["Ln1W", "Ln1B", "Wq", "Bq", "Wk", "Bk", "Wv", "Bv", "Wo", "Bo", "Ln2W", "Ln2B", "W1", "B1", "W2", "B2"]


def make_fgclip2_encoder_layers_v1(n_layers: int, rows: int, hidden: int, heads: int, hd_pad: int, inter_pad: int,
                                   tn: int = 32, eps: float = 1e-6):
    """`n_layers` copies of `make_fgclip2_encoder_layer_v1`'s single layer, in ONE launch, as a device-side loop
    over a weight table (`Table[n_layers, len(FGCLIP2_TABLE_COLS)]`, one int64 device address per cell, host-filled
    the way `Session.address` would for a real model). Every layer shares the same CMX scratch (declared once,
    outside the loop) and reads its own row's `T.ddr_view`s inside it. Still: real channel widths (hidden=1152,
    heads*hd_pad=1280, inter_pad=4320), small `rows`, one working SHAVE, plain (non-chunked) softmax -- real
    patch/token counts and the attention-pooling/dense heads are later increments."""
    QW = heads * hd_pad
    assert hidden % tn == 0 and QW % tn == 0 and inter_pad % tn == 0 and tn % 16 == 0 and rows % 16 == 0
    NCOL = len(FGCLIP2_TABLE_COLS)
    col = {n: i for i, n in enumerate(FGCLIP2_TABLE_COLS)}

    @T.prim_func
    def layers(Hio: T.Tensor((rows, hidden), T.float32), Table: T.Tensor((n_layers, NCOL), T.int64)):
        with T.Kernel(1, threads=2) as tile:
            shave = T.get_thread_binding()
            h32 = T.alloc_shared((rows, hidden), T.float32)
            xn = T.alloc_shared((rows, hidden), T.float16)
            ln1w = T.alloc_shared((1, hidden), T.float16)
            ln1b = T.alloc_shared((1, hidden), T.float16)
            ln2w = T.alloc_shared((1, hidden), T.float16)
            ln2b = T.alloc_shared((1, hidden), T.float16)
            bq = T.alloc_shared((1, QW), T.float16)
            bk = T.alloc_shared((1, QW), T.float16)
            bv = T.alloc_shared((1, QW), T.float16)
            bo = T.alloc_shared((1, hidden), T.float16)
            b1 = T.alloc_shared((1, inter_pad), T.float16)
            b2 = T.alloc_shared((1, hidden), T.float16)
            qp = T.alloc_shared((rows, QW), T.float16)
            kp = T.alloc_shared((rows, QW), T.float16)
            vp = T.alloc_shared((rows, QW), T.float16)
            ao = T.alloc_shared((rows, QW), T.float16)
            aop = T.alloc_shared((rows, hidden), T.float16)
            act = T.alloc_shared((rows, inter_pad), T.float16)
            mlp_out = T.alloc_shared((rows, hidden), T.float16)
            w_tile_h = T.alloc_shared((tn, hidden), T.float16)
            w_tile_q = T.alloc_shared((tn, QW), T.float16)
            w_tile_i = T.alloc_shared((tn, inter_pad), T.float16)
            y_tile = T.alloc_shared((rows, tn), T.float16)
            s_cmx = T.alloc_shared((rows, rows), T.float16)
            qh = T.alloc_shared((rows, hd_pad), T.float16)
            kh = T.alloc_shared((rows, hd_pad), T.float16)
            vth = T.alloc_shared((hd_pad, rows), T.float16)
            oh = T.alloc_shared((rows, hd_pad), T.float16)
            mx = T.alloc_shared((1,), T.float32)
            sm = T.alloc_shared((1,), T.float32)

            if shave == 0:
                T.copy(Hio[0, 0], h32)
                for li in T.serial(n_layers):
                    Ln1W = T.ddr_view(Table[li, col["Ln1W"]], (1, hidden), T.float16)
                    Ln1B = T.ddr_view(Table[li, col["Ln1B"]], (1, hidden), T.float16)
                    Wq = T.ddr_view(Table[li, col["Wq"]], (QW, hidden), T.float16)
                    Bq = T.ddr_view(Table[li, col["Bq"]], (1, QW), T.float16)
                    Wk = T.ddr_view(Table[li, col["Wk"]], (QW, hidden), T.float16)
                    Bk = T.ddr_view(Table[li, col["Bk"]], (1, QW), T.float16)
                    Wv = T.ddr_view(Table[li, col["Wv"]], (QW, hidden), T.float16)
                    Bv = T.ddr_view(Table[li, col["Bv"]], (1, QW), T.float16)
                    Wo = T.ddr_view(Table[li, col["Wo"]], (hidden, QW), T.float16)
                    Bo = T.ddr_view(Table[li, col["Bo"]], (1, hidden), T.float16)
                    Ln2W = T.ddr_view(Table[li, col["Ln2W"]], (1, hidden), T.float16)
                    Ln2B = T.ddr_view(Table[li, col["Ln2B"]], (1, hidden), T.float16)
                    W1 = T.ddr_view(Table[li, col["W1"]], (inter_pad, hidden), T.float16)
                    B1 = T.ddr_view(Table[li, col["B1"]], (1, inter_pad), T.float16)
                    W2 = T.ddr_view(Table[li, col["W2"]], (hidden, inter_pad), T.float16)
                    B2 = T.ddr_view(Table[li, col["B2"]], (1, hidden), T.float16)

                    T.copy(Ln1W[0, 0], ln1w)
                    T.copy(Ln1B[0, 0], ln1b)
                    T.copy(Ln2W[0, 0], ln2w)
                    T.copy(Ln2B[0, 0], ln2b)
                    T.copy(Bq[0, 0], bq)
                    T.copy(Bk[0, 0], bk)
                    T.copy(Bv[0, 0], bv)
                    T.copy(Bo[0, 0], bo)
                    T.copy(B1[0, 0], b1)
                    T.copy(B2[0, 0], b2)

                    for r in T.serial(rows):
                        layernorm_bias_row(h32, ln1w, ln1b, xn, r, hidden, eps)
                    linear_bias_tiled(xn, Wq, bq, qp, w_tile_h, y_tile, rows, QW, hidden, tn, False)
                    linear_bias_tiled(xn, Wk, bk, kp, w_tile_h, y_tile, rows, QW, hidden, tn, False)
                    linear_bias_tiled(xn, Wv, bv, vp, w_tile_h, y_tile, rows, QW, hidden, tn, False)
                    mha_full(qp, kp, vp, ao, s_cmx, qh, kh, vth, oh, mx, sm, rows, heads, hd_pad, hd_pad ** -0.5)
                    linear_bias_tiled(ao, Wo, bo, aop, w_tile_q, y_tile, rows, hidden, QW, tn, False)
                    residual_add_fp32(h32, aop, rows, hidden)

                    for r in T.serial(rows):
                        layernorm_bias_row(h32, ln2w, ln2b, xn, r, hidden, eps)
                    linear_bias_tiled(xn, W1, b1, act, w_tile_h, y_tile, rows, inter_pad, hidden, tn, True)
                    linear_bias_tiled(act, W2, b2, mlp_out, w_tile_i, y_tile, rows, hidden, inter_pad, tn, False)
                    residual_add_fp32(h32, mlp_out, rows, hidden)

                T.copy(h32, Hio[0, 0])
            T.sync_threads()

    return layers


def fgclip2_layers_reference(x, weights_per_layer, heads: int, head_dim: int, eps: float = 1e-6):
    """numpy oracle of `make_fgclip2_encoder_layers_v1`: `encoder_layer_reference` applied once per entry of
    `weights_per_layer` (each a dict with keys ln1w/ln1b/wq/bq/wk/bk/wv/bv/wo/bo/ln2w/ln2b/w1/b1/w2/b2), in
    order, each layer's output feeding the next -- the same thing the device-side loop does one `Table` row
    at a time."""
    for p in weights_per_layer:
        x = encoder_layer_reference(x, p["ln1w"], p["ln1b"], p["wq"], p["bq"], p["wk"], p["bk"], p["wv"], p["bv"],
                                    p["wo"], p["bo"], p["ln2w"], p["ln2b"], p["w1"], p["b1"], p["w2"], p["b2"],
                                    heads, head_dim, eps)
    return x
