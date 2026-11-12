"""Gemma4 V4: the SAME `make_gemma4_layer_prefill` composition, REAL Gemma-4-26B-A4B-it weights
(layer 0, sliding attention) read directly from the real checkpoint's OpenVINO IR via npu_dsl's
`OvIr` (pure numpy, no torch/Gemma4Ref dependency -- see docs/developer_guide/
gemma4_enablement_plan.md section 0 for the checkpoint path and section 4 for the milestone). The
reference is `gemma4_decoder_layer_reference` fed the SAME real (dequantised) weights, so a
disagreement here is a real bug in the kernel or the packing, not a convention mismatch between two
independently-built oracles.

Token input: REAL embedded tokens (via the IR's own `embed_i8`), not synthetic -- an end-to-end-ish
check from real token ids through a real transformer layer's real weights.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import pack_int4  # noqa: E402
from gemma4_kernels import gemma4_decoder_layer_reference, make_gemma4_layer_prefill  # noqa: E402
import tilelang.intel_npu.language as T  # noqa: E402

NPU_DSL_ROOT = r"C:\Intel\guoliang\dsl\libraries.npu.dsl"
CHECKPOINT = r"C:\Intel\guoliang\model_pkg\Google-Gemma-4-26B-A4B-it_int4_sym_qat_group-64_dyn_stateful"

N = 8
TILES, SHAVES = 4, 2
TB, BKV, BN = 4, 32, 16
MB_EXPERT, MB_PROJ, BNW_GU, BNW_D = 4, 4, 16, 16
SMAX = 256
REL = 1.5e-1


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.prefetch": False}):
        return lowering.lower(func)


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


def _has_checkpoint() -> bool:
    return Path(CHECKPOINT, "openvino_language_model.xml").exists() and Path(NPU_DSL_ROOT, "gemma4_e2e").exists()


needs_cxx_and_checkpoint = pytest.mark.skipif(
    not (_has_host_cxx() and _has_checkpoint()), reason="needs host-sim clang++ AND the real Gemma4 checkpoint + npu_dsl checkout"
)


def _pack(w_int):
    if w_int.ndim == 2:
        return pack_int4(w_int)
    return np.stack([_pack(w_int[i]) for i in range(w_int.shape[0])])


def _gm(scale_row_major):
    """[N, K/G] row-major -> [K/G, N] group-major (kernels.py's gm_scales, the layout our kernels read)."""
    return np.ascontiguousarray(scale_row_major.T)


def _dequant(q_int8, scale_row_major, G):
    n, k = q_int8.shape
    ng = scale_row_major.shape[1]
    return (q_int8.astype(np.float32).reshape(n, ng, G) * scale_row_major.astype(np.float32)[:, :, None]).reshape(n, k)


@needs_cxx_and_checkpoint
def test_real_checkpoint_layer0_matches_reference():
    sys.path.insert(0, NPU_DSL_ROOT)
    from gemma4_e2e.reference.ov_ir import OvIr

    ir = OvIr(CHECKPOINT)
    assert ir.layer_types[0] == "sliding_attention"
    G = ir.group
    assert G == 64
    H, NH = 2816, 16
    NKV, D = 8, 256
    FI = 2112
    E, TOPK = 128, 8
    I_EXP = 704
    GQ = NH // NKV
    KO = NH * D
    NQKV = NH * D + 2 * NKV * D

    def q_s(name):
        q, s = ir.int4_cw(name)
        return q.astype(np.int8), s

    wq_q, wq_s = q_s(ir.lname(0, "q"))
    wk_q, wk_s = q_s(ir.lname(0, "k"))
    wv_q, wv_s = q_s(ir.lname(0, "v"))
    wo_q, wo_s = q_s(ir.lname(0, "o"))
    wg_q, wg_s = q_s(ir.lname(0, "gate"))
    wu_q, wu_s = q_s(ir.lname(0, "up"))
    wd_q, wd_s = q_s(ir.lname(0, "down"))
    router_proj = ir.router_proj(0).astype(np.float32)  # already dequantised by ov_ir if int4
    router_scale = ir.router_scale(0).astype(np.float32)
    per_expert_scale = ir.per_expert_scale(0).astype(np.float32)
    layer_scalar = ir.layer_scalar(0)
    norms = {f"{w}_norm": ir.norm(0, w).astype(np.float32) for w in (
        "input", "post_attn", "pre_ffn", "post_ffn_1", "pre_ffn_2", "post_ffn_2", "post_ffn")}
    q_norm = ir.norm(0, "q").astype(np.float32)
    k_norm = ir.norm(0, "k").astype(np.float32)

    egate_name, eup_name, edown_name = ir.lname(0, "egate"), ir.lname(0, "eup"), ir.lname(0, "edown")
    ewg_q = np.zeros((E, I_EXP, H), np.int8)
    ewg_s = np.zeros((E, H // G, I_EXP), np.float16)
    ewu_q = np.zeros((E, I_EXP, H), np.int8)
    ewu_s = np.zeros((E, H // G, I_EXP), np.float16)
    ewd_q = np.zeros((E, H, I_EXP), np.int8)
    ewd_s = np.zeros((E, I_EXP // G, H), np.float16)
    for e in range(E):
        q, s = ir.int4_cw(egate_name, expert=e)
        ewg_q[e], ewg_s[e] = q.astype(np.int8), _gm(s)
        q, s = ir.int4_cw(eup_name, expert=e)
        ewu_q[e], ewu_s[e] = q.astype(np.int8), _gm(s)
        q, s = ir.int4_cw(edown_name, expert=e)
        ewd_q[e], ewd_s[e] = q.astype(np.int8), _gm(s)

    # --- real token embeddings as this layer's input ------------------------------------------------
    embed_q, embed_scale, embed_mult = ir.embed_i8()
    rng = np.random.default_rng(7)
    ids = rng.integers(0, embed_q.shape[0], N)
    h = (embed_q[ids].astype(np.float32) * embed_scale[ids].astype(np.float32)[:, None] * float(embed_mult))

    # --- real RoPE: ov_ir.rope_cos_sin already returns FULL-WIDTH [N, D] cos/sin (not half-width --
    # only gemma4_e2e/pack.py's device packer slices to half; our full-width "duplicated-half" kernel
    # convention is satisfied directly since cos/sin here already repeat across the two halves) --------
    positions = np.arange(N, dtype=np.int64)
    cos_full, sin_full = ir.rope_cos_sin("sliding", positions)
    cos_sin = np.concatenate([cos_full, sin_full], axis=1).astype(np.float32)

    weights_ref = dict(
        **norms, q_norm=q_norm, k_norm=k_norm, cos_sin=cos_sin,
        wq=np.concatenate([_dequant(wq_q, wq_s, G), _dequant(wk_q, wk_s, G), _dequant(wv_q, wv_s, G)], axis=0),
        wo=_dequant(wo_q, wo_s, G), wg=_dequant(wg_q, wg_s, G), wu=_dequant(wu_q, wu_s, G), wd=_dequant(wd_q, wd_s, G),
        router_proj=router_proj, router_scale=router_scale, per_expert_scale=per_expert_scale, layer_scalar=float(layer_scalar),
        ewg=np.stack([_dequant(ewg_q[e], ewg_s[e].T, G) for e in range(E)]),
        ewu=np.stack([_dequant(ewu_q[e], ewu_s[e].T, G) for e in range(E)]),
        ewd=np.stack([_dequant(ewd_q[e], ewd_s[e].T, G) for e in range(E)]),
        ids_topk=TOPK,
    )

    layer_macro, sizes = make_gemma4_layer_prefill(
        N=N, H=H, NH=NH, NKV=NKV, D=D, FI=FI, E=E, TOPK=TOPK, I=I_EXP, G=G, SMAX=SMAX, full=False, window=None,
        TB=TB, BKV=BKV, BN=BN, MB_expert=MB_EXPERT, MB_proj=MB_PROJ, BNW_gu=BNW_GU, BNW_d=BNW_D,
        tiles=TILES, shaves=SHAVES,
    )
    NQKV_s, GQ_s, KO_s, SLOTS = sizes["NQKV"], sizes["GQ"], sizes["KO"], sizes["SLOTS"]
    assert (NQKV_s, GQ_s, KO_s) == (NQKV, GQ, KO)

    @T.prim_func
    def layer_kernel(
        Hio: T.Tensor((N, H), T.float32), InNorm: T.Tensor((1, H), T.float16), QNorm: T.Tensor((1, D), T.float16),
        KNorm: T.Tensor((1, D), T.float16), CosSin: T.Tensor((N, 2 * D), T.float32),
        PostAttnNorm: T.Tensor((1, H), T.float16), PreFfnNorm: T.Tensor((1, H), T.float16),
        PostFfn1Norm: T.Tensor((1, H), T.float16), PreFfn2Norm: T.Tensor((1, H), T.float16),
        PostFfn2Norm: T.Tensor((1, H), T.float16), PostFfnNorm: T.Tensor((1, H), T.float16),
        RouterScale: T.Tensor((1, H), T.float16),
        WqkvQ: T.Tensor((NQKV, H // 2), T.uint8), WqkvS: T.Tensor((H // G, NQKV), T.float16),
        WoQ: T.Tensor((H, KO // 2), T.uint8), WoS: T.Tensor((KO // G, H), T.float16),
        WgateQ: T.Tensor((FI, H // 2), T.uint8), WgateS: T.Tensor((H // G, FI), T.float16),
        WupQ: T.Tensor((FI, H // 2), T.uint8), WupS: T.Tensor((H // G, FI), T.float16),
        WdownQ: T.Tensor((H, FI // 2), T.uint8), WdownS: T.Tensor((FI // G, H), T.float16),
        WR: T.Tensor((E, H), T.float16),
        EWgateQ: T.Tensor((E, I_EXP, H // 2), T.uint8), EWgateS: T.Tensor((E, H // G, I_EXP), T.float16),
        EWupQ: T.Tensor((E, I_EXP, H // 2), T.uint8), EWupS: T.Tensor((E, H // G, I_EXP), T.float16),
        EWdownQ: T.Tensor((E, H, I_EXP // 2), T.uint8), EWdownS: T.Tensor((E, I_EXP // G, H), T.float16),
        PerExpertScale: T.Tensor((E, 1), T.float32), LayerScalar: T.Tensor((1, 1), T.float32),
        Kc: T.Tensor((NKV, SMAX, D), T.float16), Vt: T.Tensor((NKV, D, SMAX), T.float16),
        Pos: T.Tensor((1, 1), T.int32), Nt: T.Tensor((1, 1), T.int32),
        Xn: T.Tensor((N, H), T.float16), Ys: T.Tensor((N, NQKV), T.float16), Q: T.Tensor((NKV, N, GQ, D), T.float16),
        AttnO: T.Tensor((N, KO), T.float16), AttnP: T.Tensor((N, H), T.float16), PostAttnN: T.Tensor((N, H), T.float16),
        Xdense: T.Tensor((N, H), T.float16), Dense: T.Tensor((N, H), T.float16), DenseN: T.Tensor((N, H), T.float16),
        Gate: T.Tensor((N, FI), T.float16), Up: T.Tensor((N, FI), T.float16), Act: T.Tensor((N, FI), T.float16),
        Xroute: T.Tensor((N, H), T.float16), Logits: T.Tensor((N, E), T.float16),
        Ids: T.Tensor((N, TOPK), T.int32), Wts: T.Tensor((N, TOPK), T.float32),
        Off: T.Tensor((E + 1,), T.int32), Slot: T.Tensor((SLOTS,), T.int32), Xmoe: T.Tensor((N, H), T.float16),
        MoePart: T.Tensor((TILES * SHAVES, N, H), T.float32), MoeOut: T.Tensor((N, H), T.float32),
        MoeN: T.Tensor((N, H), T.float16), Combined: T.Tensor((N, H), T.float16),
    ):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            layer_macro(Hio, InNorm, QNorm, KNorm, CosSin, PostAttnNorm, PreFfnNorm, PostFfn1Norm, PreFfn2Norm,
                       PostFfn2Norm, PostFfnNorm, RouterScale, WqkvQ, WqkvS, WoQ, WoS, WgateQ, WgateS, WupQ, WupS,
                       WdownQ, WdownS, WR, EWgateQ, EWgateS, EWupQ, EWupS, EWdownQ, EWdownS, PerExpertScale,
                       LayerScalar, Kc, Vt, Pos, Nt, Xn, Ys, Q, AttnO, AttnP, PostAttnN, Xdense, Dense, DenseN,
                       Gate, Up, Act, Xroute, Logits, Ids, Wts, Off, Slot, Xmoe, MoePart, MoeOut, MoeN, Combined,
                       tile, T.get_thread_binding())

    low = _lower(layer_kernel)
    lib = host_sim.build(low.source, "tl_gemma4_real_checkpoint_layer0_sim")
    arrays = dict(
        Hio=h.copy(), InNorm=norms["input_norm"].astype(np.float16).reshape(1, H),
        QNorm=q_norm.astype(np.float16).reshape(1, D), KNorm=k_norm.astype(np.float16).reshape(1, D),
        CosSin=cos_sin,
        PostAttnNorm=norms["post_attn_norm"].astype(np.float16).reshape(1, H),
        PreFfnNorm=norms["pre_ffn_norm"].astype(np.float16).reshape(1, H),
        PostFfn1Norm=norms["post_ffn_1_norm"].astype(np.float16).reshape(1, H),
        PreFfn2Norm=norms["pre_ffn_2_norm"].astype(np.float16).reshape(1, H),
        PostFfn2Norm=norms["post_ffn_2_norm"].astype(np.float16).reshape(1, H),
        PostFfnNorm=norms["post_ffn_norm"].astype(np.float16).reshape(1, H),
        RouterScale=router_scale.astype(np.float16).reshape(1, H),
        WqkvQ=_pack(np.concatenate([wq_q, wk_q, wv_q], axis=0)), WqkvS=_gm(np.concatenate([wq_s, wk_s, wv_s], axis=0)),
        WoQ=_pack(wo_q), WoS=_gm(wo_s),
        WgateQ=_pack(wg_q), WgateS=_gm(wg_s), WupQ=_pack(wu_q), WupS=_gm(wu_s), WdownQ=_pack(wd_q), WdownS=_gm(wd_s),
        WR=router_proj.astype(np.float16),
        EWgateQ=_pack(ewg_q), EWgateS=ewg_s, EWupQ=_pack(ewu_q), EWupS=ewu_s, EWdownQ=_pack(ewd_q), EWdownS=ewd_s,
        PerExpertScale=per_expert_scale.reshape(E, 1), LayerScalar=np.array([[layer_scalar]], np.float32),
        Kc=np.zeros((NKV, SMAX, D), np.float16), Vt=np.zeros((NKV, D, SMAX), np.float16),
        Pos=np.array([[0]], np.int32), Nt=np.array([[N]], np.int32),
        Xn=np.zeros((N, H), np.float16), Ys=np.zeros((N, NQKV), np.float16),
        Q=np.zeros((NKV, N, GQ, D), np.float16), AttnO=np.zeros((N, KO), np.float16),
        AttnP=np.zeros((N, H), np.float16), PostAttnN=np.zeros((N, H), np.float16),
        Xdense=np.zeros((N, H), np.float16), Dense=np.zeros((N, H), np.float16), DenseN=np.zeros((N, H), np.float16),
        Gate=np.zeros((N, FI), np.float16), Up=np.zeros((N, FI), np.float16), Act=np.zeros((N, FI), np.float16),
        Xroute=np.zeros((N, H), np.float16), Logits=np.zeros((N, E), np.float16),
        Ids=np.zeros((N, TOPK), np.int32), Wts=np.zeros((N, TOPK), np.float32),
        Off=np.zeros((E + 1,), np.int32), Slot=np.zeros((N * TOPK,), np.int32),
        Xmoe=np.zeros((N, H), np.float16),
        MoePart=np.zeros((TILES * SHAVES, N, H), np.float32), MoeOut=np.zeros((N, H), np.float32),
        MoeN=np.zeros((N, H), np.float16), Combined=np.zeros((N, H), np.float16),
    )
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))

    ref = gemma4_decoder_layer_reference(h, weights_ref, NH, NKV, D, False, None)
    out = arrays["Hio"]
    rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)
    cos = float(np.sum(out * ref) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-9))
    print(f"REAL CHECKPOINT layer-0 result: rel_err={rel:.4f} cosine={cos:.6f}")
    assert rel < REL, (rel, cos)
