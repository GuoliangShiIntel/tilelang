"""Minimal FG-CLIP2 so400m weight packer: reads the real checkpoint's `model.safetensors` directly (numpy only,
no torch) and produces per-layer tensors in THIS backend's kernel layout (`make_fgclip2_encoder_layer_ddr` /
`make_fgclip2_encoder_layers_v1`'s `FGCLIP2_TABLE_COLS`).

Deliberately simpler than npu_dsl's `fgclip2_e2e/pack.py`: this kernel keeps q/k/v/o separate (no Wqkv
stacking), keeps the real k bias (no drop), keeps the real v bias (no fold into o_proj), and keeps LayerNorm's
real affine gamma/beta (no fold into the following GEMM) -- none of those optimisations exist in this backend's
kernels yet (plan V4), so packing for them would just be dead weight. The one padding that IS unavoidable
(DPU alignment, not an optimisation) is kept: head_dim 72 -> hd_pad (a multiple of 16) and MLP width 4304 ->
inter_pad (a multiple of 16), both zero-padded -- exact, not approximate (pad_cols_rows below).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors import safe_open

HIDDEN, HEADS, HEAD_DIM, INTER, LAYERS = 1152, 16, 72, 4304, 27


class Checkpoint:
    """Lazy fp32 reader of `model.safetensors`, cached per-tensor-name (the same tensor is read at most once)."""

    def __init__(self, root):
        self._f = safe_open(str(Path(root) / "model.safetensors"), "np")
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = self._f.get_tensor(name).astype(np.float64)
        return self._cache[name]


def _f16(a: np.ndarray) -> np.ndarray:
    out = np.asarray(a, np.float32).astype(np.float16)
    assert np.isfinite(out).all(), "fp16 overflow while packing"
    return out


def pad_head_rows(w: np.ndarray, hd_pad: int) -> np.ndarray:
    """[heads*head_dim, k] -> [heads*hd_pad, k], zero rows 72..hd_pad-1 of every head's block."""
    out = np.zeros((HEADS, hd_pad, w.shape[1]), np.float64)
    out[:, :HEAD_DIM] = w.reshape(HEADS, HEAD_DIM, -1)
    return out.reshape(HEADS * hd_pad, -1)


def pad_head_vec(b: np.ndarray, hd_pad: int) -> np.ndarray:
    out = np.zeros((HEADS, hd_pad), np.float64)
    out[:, :HEAD_DIM] = b.reshape(HEADS, HEAD_DIM)
    return out.reshape(-1)


def pad_head_cols(w: np.ndarray, hd_pad: int) -> np.ndarray:
    """[n, heads*head_dim] -> [n, heads*hd_pad], zero columns 72..hd_pad-1 of every head's block."""
    out = np.zeros((w.shape[0], HEADS, hd_pad), np.float64)
    out[:, :, :HEAD_DIM] = w.reshape(w.shape[0], HEADS, HEAD_DIM)
    return out.reshape(w.shape[0], -1)


def pad_mlp_rows(w: np.ndarray, b: np.ndarray, inter_pad: int):
    wo = np.zeros((inter_pad, w.shape[1]), np.float64)
    wo[:INTER] = w
    bo = np.zeros(inter_pad, np.float64)
    bo[:INTER] = b
    return wo, bo


def pad_mlp_cols(w: np.ndarray, inter_pad: int) -> np.ndarray:
    out = np.zeros((w.shape[0], inter_pad), np.float64)
    out[:, :INTER] = w
    return out


def pack_layer(ck: Checkpoint, tower: str, i: int, hd_pad: int = 80, inter_pad: int = 4320) -> dict:
    """One encoder layer's weights (`tower` is 'vision_model' or 'text_model'), fp16, in this backend's kernel
    layout -- a dict with exactly `make_fgclip2_encoder_layer_ddr`'s weight parameter names (lowercased)."""
    p = f"{tower}.encoder.layers.{i}."
    ln1w, ln1b = ck[p + "layer_norm1.weight"], ck[p + "layer_norm1.bias"]
    ln2w, ln2b = ck[p + "layer_norm2.weight"], ck[p + "layer_norm2.bias"]
    wq, bq = ck[p + "self_attn.q_proj.weight"], ck[p + "self_attn.q_proj.bias"]
    wk, bk = ck[p + "self_attn.k_proj.weight"], ck[p + "self_attn.k_proj.bias"]
    wv, bv = ck[p + "self_attn.v_proj.weight"], ck[p + "self_attn.v_proj.bias"]
    wo, bo = ck[p + "self_attn.out_proj.weight"], ck[p + "self_attn.out_proj.bias"]
    w1, b1 = ck[p + "mlp.fc1.weight"], ck[p + "mlp.fc1.bias"]
    w2, b2 = ck[p + "mlp.fc2.weight"], ck[p + "mlp.fc2.bias"]
    w1p, b1p = pad_mlp_rows(w1, b1, inter_pad)
    return dict(
        ln1w=_f16(ln1w.reshape(1, -1)), ln1b=_f16(ln1b.reshape(1, -1)),
        wq=_f16(pad_head_rows(wq, hd_pad)), bq=_f16(pad_head_vec(bq, hd_pad).reshape(1, -1)),
        wk=_f16(pad_head_rows(wk, hd_pad)), bk=_f16(pad_head_vec(bk, hd_pad).reshape(1, -1)),
        wv=_f16(pad_head_rows(wv, hd_pad)), bv=_f16(pad_head_vec(bv, hd_pad).reshape(1, -1)),
        wo=_f16(pad_head_cols(wo, hd_pad)), bo=_f16(bo.reshape(1, -1)),
        ln2w=_f16(ln2w.reshape(1, -1)), ln2b=_f16(ln2b.reshape(1, -1)),
        w1=_f16(w1p), b1=_f16(b1p.reshape(1, -1)),
        w2=_f16(pad_mlp_cols(w2, inter_pad)), b2=_f16(b2.reshape(1, -1)),
    )


def pack_vision_head(ck: Checkpoint, hd_pad: int = 80, inter_pad: int = 4320) -> dict:
    """The image tower's attention-pooling head (`vision_model.head.*`): one learned `probe` vector attends
    over every patch token (post-`post_layernorm`), then out_proj + LayerNorm + GELU-MLP + residual ->
    the image embedding (`reference.py::vision`'s tail, after the 27 encoder layers). `MapQ` is the probe's
    query PRECOMPUTED on the host (`probe @ Wq^T + bq`, a single row -- there is nothing to gain from doing a
    1-row GEMM on the device when the probe never changes): the one piece of "host does the constant-folding"
    this packer allows itself, since it changes no number a device GEMM would have produced, just when it
    runs."""
    p = "vision_model.head."
    ipw, ipb = ck[p + "attention.in_proj_weight"], ck[p + "attention.in_proj_bias"]
    wq, bq = ipw[:HIDDEN], ipb[:HIDDEN]
    wk, bk = ipw[HIDDEN:2 * HIDDEN], ipb[HIDDEN:2 * HIDDEN]
    wv, bv = ipw[2 * HIDDEN:], ipb[2 * HIDDEN:]
    probe = ck[p + "probe"].reshape(HIDDEN)
    wq_pad, bq_pad = pad_head_rows(wq, hd_pad), pad_head_vec(bq, hd_pad)
    map_q = probe @ wq_pad.T + bq_pad
    wo, bo = ck[p + "attention.out_proj.weight"], ck[p + "attention.out_proj.bias"]
    hlnw, hlnb = ck[p + "layernorm.weight"], ck[p + "layernorm.bias"]
    w1, b1 = ck[p + "mlp.fc1.weight"], ck[p + "mlp.fc1.bias"]
    w2, b2 = ck[p + "mlp.fc2.weight"], ck[p + "mlp.fc2.bias"]
    w1p, b1p = pad_mlp_rows(w1, b1, inter_pad)
    post_w, post_b = ck["vision_model.post_layernorm.weight"], ck["vision_model.post_layernorm.bias"]
    return dict(
        post_ln_w=_f16(post_w.reshape(1, -1)), post_ln_b=_f16(post_b.reshape(1, -1)),
        map_q=_f16(map_q.reshape(1, -1)),
        map_wk=_f16(pad_head_rows(wk, hd_pad)), map_bk=_f16(pad_head_vec(bk, hd_pad).reshape(1, -1)),
        map_wv=_f16(pad_head_rows(wv, hd_pad)), map_bv=_f16(pad_head_vec(bv, hd_pad).reshape(1, -1)),
        map_wo=_f16(pad_head_cols(wo, hd_pad)), map_bo=_f16(bo.reshape(1, -1)),
        hln_w=_f16(hlnw.reshape(1, -1)), hln_b=_f16(hlnb.reshape(1, -1)),
        map_w1=_f16(w1p), map_b1=_f16(b1p.reshape(1, -1)),
        map_w2=_f16(pad_mlp_cols(w2, inter_pad)), map_b2=_f16(b2.reshape(1, -1)),
    )


def pack_text_tail(ck: Checkpoint) -> dict:
    """The text tower's tail (`reference.py::text_short`, after the 27 encoder layers): final LayerNorm +
    head projection. `boxtext_head` (the `box` walk's separate head) is not packed -- out of scope for now,
    see the enablement plan."""
    return dict(
        final_ln_w=_f16(ck["text_model.final_layer_norm.weight"].reshape(1, -1)),
        final_ln_b=_f16(ck["text_model.final_layer_norm.bias"].reshape(1, -1)),
        head_w=_f16(ck["text_model.head.weight"]),
        head_b=_f16(ck["text_model.head.bias"].reshape(1, -1)),
    )


def pack_text_embedding(ck: Checkpoint, ids: np.ndarray) -> np.ndarray:
    """H0[t, :] = TokEmb[ids[t]] + PosEmb[t] for one caption's 64 token ids -- host-side gather (exact, no
    device kernel needed; same precedent as the Qwen decode kernel's host-uploaded embedding row)."""
    tok_emb = ck["text_model.embeddings.token_embedding.weight"]
    pos_emb = ck["text_model.embeddings.position_embedding.weight"]
    t = ids.shape[0]
    return (tok_emb[ids] + pos_emb[:t]).astype(np.float32)
