"""Gemma4 V0/V1 closing milestone: a WHOLE decoder layer, BOTH kinds (sliding-window and full
attention), in ONE launch, host-sim, synthetic weights, against `gemma4_decoder_layer_reference` --
a from-scratch numpy reimplementation of `Gemma4Ref.decoder_layer`'s exact math (no torch/OvIr
dependency). See docs/developer_guide/gemma4_enablement_plan.md section 4 (V0/V1) for the milestone
definition.
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

N, H, NH, NKV, D, FI, E, TOPK, I_EXP, G = 8, 64, 4, 2, 32, 64, 32, 4, 64, 64
SMAX, TILES, SHAVES, TB, BKV, BN, MB_EXPERT = 64, 2, 2, 4, 32, 32, 4
REL = 8e-2


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")


def _dequant_gm(w_int, scale_gm, G):
    n, k = w_int.shape[-2:]
    s = np.moveaxis(scale_gm, -2, -1).astype(np.float32)  # [..., N, K/G]
    w = w_int.astype(np.float32)
    lead = w.shape[:-2]
    return (w.reshape(*lead, n, k // G, G) * s[..., None]).reshape(*lead, n, k)


def _rand_q(rng, shape):
    return rng.integers(-7, 8, shape, np.int8)


def _pack(w_int):
    if w_int.ndim == 2:
        return pack_int4(w_int)
    return np.stack([_pack(w_int[i]) for i in range(w_int.shape[0])])


def _build(full: bool, window, seed=0):
    rng = np.random.default_rng(seed)
    GQ = NH // NKV
    KO = NH * D
    NQKV = NH * D + (NKV * D if full else 2 * NKV * D)

    h = rng.standard_normal((N, H), np.float32) * 0.5
    norms = {name: rng.uniform(0.5, 1.5, H).astype(np.float32) for name in (
        "input_norm", "post_attn_norm", "pre_ffn_norm", "post_ffn_1_norm", "pre_ffn_2_norm",
        "post_ffn_2_norm", "post_ffn_norm", "router_scale")}
    q_norm = rng.uniform(0.5, 1.5, D).astype(np.float32)
    k_norm = rng.uniform(0.5, 1.5, D).astype(np.float32)
    half = D // 2
    angles = rng.uniform(0, 2 * np.pi, (N, half)).astype(np.float32)
    cos_sin = np.concatenate([np.tile(np.cos(angles), 2), np.tile(np.sin(angles), 2)], axis=1)

    wq_int = _rand_q(rng, (NQKV, H))
    wq_s = rng.uniform(0.01, 0.05, (H // G, NQKV)).astype(np.float16)
    wo_int = _rand_q(rng, (H, KO))
    wo_s = rng.uniform(0.01, 0.05, (KO // G, H)).astype(np.float16)
    wg_int, wg_s = _rand_q(rng, (FI, H)), rng.uniform(0.01, 0.05, (H // G, FI)).astype(np.float16)
    wu_int, wu_s = _rand_q(rng, (FI, H)), rng.uniform(0.01, 0.05, (H // G, FI)).astype(np.float16)
    wd_int, wd_s = _rand_q(rng, (H, FI)), rng.uniform(0.01, 0.05, (FI // G, H)).astype(np.float16)
    wr = rng.standard_normal((E, H), np.float32).astype(np.float16)
    per_expert_scale = rng.uniform(0.5, 1.5, E).astype(np.float32)
    layer_scalar = 1.0

    ewg_int = _rand_q(rng, (E, I_EXP, H))
    ewg_s = rng.uniform(0.01, 0.05, (E, H // G, I_EXP)).astype(np.float16)
    ewu_int = _rand_q(rng, (E, I_EXP, H))
    ewu_s = rng.uniform(0.01, 0.05, (E, H // G, I_EXP)).astype(np.float16)
    ewd_int = _rand_q(rng, (E, H, I_EXP))
    ewd_s = rng.uniform(0.01, 0.05, (E, I_EXP // G, H)).astype(np.float16)

    weights_ref = dict(
        **{k: v for k, v in norms.items()}, q_norm=q_norm, k_norm=k_norm, cos_sin=cos_sin,
        wq=_dequant_gm(wq_int, wq_s, G), wo=_dequant_gm(wo_int, wo_s, G),
        wg=_dequant_gm(wg_int, wg_s, G), wu=_dequant_gm(wu_int, wu_s, G), wd=_dequant_gm(wd_int, wd_s, G),
        router_proj=wr.astype(np.float32), per_expert_scale=per_expert_scale, layer_scalar=layer_scalar,
        ewg=_dequant_gm(ewg_int, ewg_s, G), ewu=_dequant_gm(ewu_int, ewu_s, G), ewd=_dequant_gm(ewd_int, ewd_s, G),
        ids_topk=TOPK,
    )

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
        RouterScale=norms["router_scale"].astype(np.float16).reshape(1, H),
        WqkvQ=_pack(wq_int), WqkvS=wq_s, WoQ=_pack(wo_int), WoS=wo_s,
        WgateQ=_pack(wg_int), WgateS=wg_s, WupQ=_pack(wu_int), WupS=wu_s, WdownQ=_pack(wd_int), WdownS=wd_s,
        WR=wr, EWgateQ=_pack(ewg_int), EWgateS=ewg_s, EWupQ=_pack(ewu_int), EWupS=ewu_s,
        EWdownQ=_pack(ewd_int), EWdownS=ewd_s, PerExpertScale=per_expert_scale.reshape(E, 1),
        LayerScalar=np.array([[layer_scalar]], np.float32),
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
    return h, weights_ref, arrays


@needs_cxx
def test_two_layers_chain_matches_reference():
    """V2 (lightweight): TWO sliding layers back to back in ONE launch, proving the residual stream
    correctly carries from one layer's output into the next's input, and that the compiler's automatic
    barrier derivation/CMX reuse works across a REPEATED large phase sequence, not just one. A
    python-unrolled fixed-count loop (two separate weight sets, two separate macro calls) -- the real
    device-side runtime layer TABLE (variable layer count, two-kind alternation) is deferred to a later
    milestone; this proves the chaining mechanism itself."""
    import tilelang.intel_npu.language as T

    full, window = False, None
    h0, w0, a0 = _build(full, window, seed=10)
    _, w1, a1 = _build(full, window, seed=11)
    # RoPE is purely positional -- both layers see the SAME N tokens at the SAME positions, so they share
    # ONE CosSin table (the kernel below only has a single CosSin parameter, reused for both layer_macro
    # calls); force the reference's second layer to use layer 0's table instead of its own independently
    # randomised one, matching what the kernel actually computes.
    w1["cos_sin"] = w0["cos_sin"]
    a1["CosSin"] = a0["CosSin"]
    layer_macro, sizes = make_gemma4_layer_prefill(
        N=N, H=H, NH=NH, NKV=NKV, D=D, FI=FI, E=E, TOPK=TOPK, I=I_EXP, G=G, SMAX=SMAX, full=full, window=window,
        TB=TB, BKV=BKV, BN=BN, MB_expert=MB_EXPERT, tiles=TILES, shaves=SHAVES,
    )
    NQKV, GQ, KO, SLOTS = sizes["NQKV"], sizes["GQ"], sizes["KO"], sizes["SLOTS"]

    scratch_args = dict(
        Kc=T.Tensor((NKV, SMAX, D), T.float16), Vt=T.Tensor((NKV, D, SMAX), T.float16),
        Pos=T.Tensor((1, 1), T.int32), Nt=T.Tensor((1, 1), T.int32),
        Xn=T.Tensor((N, H), T.float16), Ys=T.Tensor((N, NQKV), T.float16), Q=T.Tensor((NKV, N, GQ, D), T.float16),
        AttnO=T.Tensor((N, KO), T.float16), AttnP=T.Tensor((N, H), T.float16), PostAttnN=T.Tensor((N, H), T.float16),
        Xdense=T.Tensor((N, H), T.float16), Dense=T.Tensor((N, H), T.float16), DenseN=T.Tensor((N, H), T.float16),
        Gate=T.Tensor((N, FI), T.float16), Up=T.Tensor((N, FI), T.float16), Act=T.Tensor((N, FI), T.float16),
        Xroute=T.Tensor((N, H), T.float16), Logits=T.Tensor((N, E), T.float16),
        Ids=T.Tensor((N, TOPK), T.int32), Wts=T.Tensor((N, TOPK), T.float32),
        Off=T.Tensor((E + 1,), T.int32), Slot=T.Tensor((SLOTS,), T.int32), Xmoe=T.Tensor((N, H), T.float16),
        MoePart=T.Tensor((TILES * SHAVES, N, H), T.float32), MoeOut=T.Tensor((N, H), T.float32),
        MoeN=T.Tensor((N, H), T.float16), Combined=T.Tensor((N, H), T.float16),
    )

    @T.prim_func
    def two_layer_kernel(
        Hio: T.Tensor((N, H), T.float32), CosSin: T.Tensor((N, 2 * D), T.float32),
        L0InNorm: T.Tensor((1, H), T.float16), L0QNorm: T.Tensor((1, D), T.float16), L0KNorm: T.Tensor((1, D), T.float16),
        L0PostAttnNorm: T.Tensor((1, H), T.float16), L0PreFfnNorm: T.Tensor((1, H), T.float16),
        L0PostFfn1Norm: T.Tensor((1, H), T.float16), L0PreFfn2Norm: T.Tensor((1, H), T.float16),
        L0PostFfn2Norm: T.Tensor((1, H), T.float16), L0PostFfnNorm: T.Tensor((1, H), T.float16),
        L0RouterScale: T.Tensor((1, H), T.float16),
        L0WqkvQ: T.Tensor((NQKV, H // 2), T.uint8), L0WqkvS: T.Tensor((H // G, NQKV), T.float16),
        L0WoQ: T.Tensor((H, KO // 2), T.uint8), L0WoS: T.Tensor((KO // G, H), T.float16),
        L0WgateQ: T.Tensor((FI, H // 2), T.uint8), L0WgateS: T.Tensor((H // G, FI), T.float16),
        L0WupQ: T.Tensor((FI, H // 2), T.uint8), L0WupS: T.Tensor((H // G, FI), T.float16),
        L0WdownQ: T.Tensor((H, FI // 2), T.uint8), L0WdownS: T.Tensor((FI // G, H), T.float16),
        L0WR: T.Tensor((E, H), T.float16),
        L0EWgateQ: T.Tensor((E, I_EXP, H // 2), T.uint8), L0EWgateS: T.Tensor((E, H // G, I_EXP), T.float16),
        L0EWupQ: T.Tensor((E, I_EXP, H // 2), T.uint8), L0EWupS: T.Tensor((E, H // G, I_EXP), T.float16),
        L0EWdownQ: T.Tensor((E, H, I_EXP // 2), T.uint8), L0EWdownS: T.Tensor((E, I_EXP // G, H), T.float16),
        L0PerExpertScale: T.Tensor((E, 1), T.float32), L0LayerScalar: T.Tensor((1, 1), T.float32),
        L1InNorm: T.Tensor((1, H), T.float16), L1QNorm: T.Tensor((1, D), T.float16), L1KNorm: T.Tensor((1, D), T.float16),
        L1PostAttnNorm: T.Tensor((1, H), T.float16), L1PreFfnNorm: T.Tensor((1, H), T.float16),
        L1PostFfn1Norm: T.Tensor((1, H), T.float16), L1PreFfn2Norm: T.Tensor((1, H), T.float16),
        L1PostFfn2Norm: T.Tensor((1, H), T.float16), L1PostFfnNorm: T.Tensor((1, H), T.float16),
        L1RouterScale: T.Tensor((1, H), T.float16),
        L1WqkvQ: T.Tensor((NQKV, H // 2), T.uint8), L1WqkvS: T.Tensor((H // G, NQKV), T.float16),
        L1WoQ: T.Tensor((H, KO // 2), T.uint8), L1WoS: T.Tensor((KO // G, H), T.float16),
        L1WgateQ: T.Tensor((FI, H // 2), T.uint8), L1WgateS: T.Tensor((H // G, FI), T.float16),
        L1WupQ: T.Tensor((FI, H // 2), T.uint8), L1WupS: T.Tensor((H // G, FI), T.float16),
        L1WdownQ: T.Tensor((H, FI // 2), T.uint8), L1WdownS: T.Tensor((FI // G, H), T.float16),
        L1WR: T.Tensor((E, H), T.float16),
        L1EWgateQ: T.Tensor((E, I_EXP, H // 2), T.uint8), L1EWgateS: T.Tensor((E, H // G, I_EXP), T.float16),
        L1EWupQ: T.Tensor((E, I_EXP, H // 2), T.uint8), L1EWupS: T.Tensor((E, H // G, I_EXP), T.float16),
        L1EWdownQ: T.Tensor((E, H, I_EXP // 2), T.uint8), L1EWdownS: T.Tensor((E, I_EXP // G, H), T.float16),
        L1PerExpertScale: T.Tensor((E, 1), T.float32), L1LayerScalar: T.Tensor((1, 1), T.float32),
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
            shave = T.get_thread_binding()
            layer_macro(Hio, L0InNorm, L0QNorm, L0KNorm, CosSin, L0PostAttnNorm, L0PreFfnNorm, L0PostFfn1Norm,
                       L0PreFfn2Norm, L0PostFfn2Norm, L0PostFfnNorm, L0RouterScale, L0WqkvQ, L0WqkvS, L0WoQ, L0WoS,
                       L0WgateQ, L0WgateS, L0WupQ, L0WupS, L0WdownQ, L0WdownS, L0WR, L0EWgateQ, L0EWgateS, L0EWupQ,
                       L0EWupS, L0EWdownQ, L0EWdownS, L0PerExpertScale, L0LayerScalar, Kc, Vt, Pos, Nt, Xn, Ys, Q,
                       AttnO, AttnP, PostAttnN, Xdense, Dense, DenseN, Gate, Up, Act, Xroute, Logits, Ids, Wts, Off,
                       Slot, Xmoe, MoePart, MoeOut, MoeN, Combined, tile, shave)
            layer_macro(Hio, L1InNorm, L1QNorm, L1KNorm, CosSin, L1PostAttnNorm, L1PreFfnNorm, L1PostFfn1Norm,
                       L1PreFfn2Norm, L1PostFfn2Norm, L1PostFfnNorm, L1RouterScale, L1WqkvQ, L1WqkvS, L1WoQ, L1WoS,
                       L1WgateQ, L1WgateS, L1WupQ, L1WupS, L1WdownQ, L1WdownS, L1WR, L1EWgateQ, L1EWgateS, L1EWupQ,
                       L1EWupS, L1EWdownQ, L1EWdownS, L1PerExpertScale, L1LayerScalar, Kc, Vt, Pos, Nt, Xn, Ys, Q,
                       AttnO, AttnP, PostAttnN, Xdense, Dense, DenseN, Gate, Up, Act, Xroute, Logits, Ids, Wts, Off,
                       Slot, Xmoe, MoePart, MoeOut, MoeN, Combined, tile, shave)

    low = _lower(two_layer_kernel)
    lib = host_sim.build(low.source, "tl_gemma4_two_layer_sim")
    arrays = {"Hio": h0.copy(), "CosSin": a0["CosSin"]}
    for k, v in a0.items():
        if k not in ("CosSin", "Hio") and k not in scratch_args:
            arrays[f"L0{k}"] = v
    for k, v in a1.items():
        if k not in ("CosSin", "Hio") and k not in scratch_args:
            arrays[f"L1{k}"] = v
    for k in scratch_args:
        arrays[k] = a0[k]
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))

    ref = gemma4_decoder_layer_reference(h0, w0, NH, NKV, D, full, window)
    ref = gemma4_decoder_layer_reference(ref, w1, NH, NKV, D, full, window)
    out = arrays["Hio"]
    rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel < REL, (rel, out, ref)


@needs_cxx
@pytest.mark.parametrize("full,window", [(False, None), (False, 6), (True, None)])
def test_layer_matches_reference(full, window):
    h, weights_ref, arrays = _build(full, window)
    layer_macro, sizes = make_gemma4_layer_prefill(
        N=N, H=H, NH=NH, NKV=NKV, D=D, FI=FI, E=E, TOPK=TOPK, I=I_EXP, G=G, SMAX=SMAX, full=full, window=window,
        TB=TB, BKV=BKV, BN=BN, MB_expert=MB_EXPERT, tiles=TILES, shaves=SHAVES,
    )
    NQKV, GQ, KO, SLOTS = sizes["NQKV"], sizes["GQ"], sizes["KO"], sizes["SLOTS"]
    import tilelang.intel_npu.language as T

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
    lib = host_sim.build(low.source, "tl_gemma4_layer_sim")
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))

    ref = gemma4_decoder_layer_reference(h, weights_ref, NH, NKV, D, full, window)
    out = arrays["Hio"]
    rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel < REL, (rel, out, ref)
