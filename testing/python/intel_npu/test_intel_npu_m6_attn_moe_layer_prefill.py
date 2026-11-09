"""P1 step 3: a WHOLE full-attention PREFILL decoder layer in one launch (`make_attn_moe_layer_prefill`) --
the mixer half of P1 step 2 followed by the second norm, P2a's routing, P2b's grouped expert FFN and the sum
back into the fp32 residual stream. With `make_gdn_moe_layer_prefill` this is the second of the two layer kinds
the prefill model kernel runs.

The oracle is the phase oracles chained, so the tests go after composition: that BOTH halves add to the stream
rather than replacing it, that the layer still holds the dead-token guard the attention mixer needs and the GDN
layer does not, and that the nesting did not reintroduce question 10.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import (attn_moe_layer_prefill_reference, gm_scales, make_attn_moe_layer_prefill,  # noqa: E402
                     make_moe_route_prefill, pack_int4, rope_cos_sin)
from test_intel_npu_scoped_cmx import allocations_inside_loops  # noqa: E402

TILES, SHAVES = 2, 2
N, H, NH, NKV, D, ROT, SMAX, G = 32, 256, 4, 2, 32, 16, 128, 128
I, E, TOPK = 128, 32, 4
TB, TBF, BKV, BN1, BN2, MB = 8, 4, 32, 16, 16, 16
BM, BNR, MBe, BNG, BND = 32, 16, 16, 64, 64
GQ, KO = NH // NKV, NH * D
NQKV = NH * 2 * D + 2 * NKV * D
NQKV_pad = -(-NQKV // BN1) * BN1
KGH, KGI = H // G, I // G
_, _RS = make_moe_route_prefill(N=N, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=TILES, shaves=SHAVES, as_block=True)
RR, SLOTS = _RS["RR"], _RS["SLOTS"]
THETA = 1e6
REL = 8e-2   # int4 weights through two projections, an fp16 flash phase and TOPK + 1 expert FFNs


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")


def _kernel():
    return make_attn_moe_layer_prefill(N=N, H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, I=I, E=E, TOPK=TOPK,
                                       G=G, TB=TB, TBF=TBF, BKV=BKV, BN1=BN1, BN2=BN2, MB=MB, BM=BM, BNR=BNR,
                                       MBe=MBe, BNG=BNG, BND=BND, tiles=TILES, shaves=SHAVES)


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        h=(rng.standard_normal((N, H), np.float32) * 0.5).astype(np.float32),
        inn=(rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16),
        post=(rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16),
        wq=rng.integers(-8, 8, (NQKV_pad, H), dtype=np.int8),
        wqs=(rng.standard_normal((NQKV_pad, H // G), np.float32) * 0.02).astype(np.float16),
        wo=rng.integers(-8, 8, (H, KO), dtype=np.int8),
        wos=(rng.standard_normal((H, KO // G), np.float32) * 0.02).astype(np.float16),
        qn=(0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16),
        kn=(0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16),
        wr=(rng.standard_normal((E + 1, H), np.float32) * 0.1).astype(np.float16),
        gu_int=rng.integers(-8, 8, (E + 1, 2 * I, H), dtype=np.int8),
        gu_s=(rng.standard_normal((E + 1, 2 * I, H // G), np.float32) * 0.02).astype(np.float16),
        d_int=rng.integers(-8, 8, (E + 1, H, I), dtype=np.int8),
        d_s=(rng.standard_normal((E + 1, H, I // G), np.float32) * 0.02).astype(np.float16),
    )


def _cos_sin(pos0):
    return np.stack([rope_cos_sin(pos0 + t, ROT, THETA).reshape(-1) for t in range(N)]).astype(np.float32)


def _run(p, n=N, pos0=0):
    low = lowering.lower(_kernel())
    lib = host_sim.build(low.source, "attn_moe_layer_prefill")
    hio = p["h"].copy()
    kc = np.zeros((NKV, SMAX, D), np.float16)
    vt = np.zeros((NKV, D, SMAX), np.float16)
    wr_pad = np.zeros((RR, H), np.float16)
    wr_pad[:E + 1] = p["wr"]
    arrays = {"Hio": hio, "InNorm": p["inn"], "PostNorm": p["post"],
              "WqkvQ": pack_int4(p["wq"]), "WqkvS": gm_scales(p["wqs"]), "QNormW": p["qn"], "KNormW": p["kn"],
              "CosSin": _cos_sin(pos0), "Pos": np.array([[pos0]], np.int32), "Nt": np.array([[n]], np.int32),
              "Kc": kc, "Vt": vt, "WoQ": pack_int4(p["wo"]), "WoS": gm_scales(p["wos"]), "Wr": wr_pad,
              "WguQ": pack_int4(p["gu_int"].reshape(-1, H)), "WguS": gm_scales(p["gu_s"].reshape(-1, KGH)),
              "WdQ": pack_int4(p["d_int"].reshape(-1, I)), "WdS": gm_scales(p["d_s"].reshape(-1, KGI)),
              "Xs": np.zeros((N, H), np.float16), "Ys": np.zeros((N, NQKV_pad), np.float16),
              "Q": np.zeros((NKV, N, GQ, D), np.float16), "Gate": np.zeros((N, KO), np.float16),
              "Os": np.zeros((N, KO), np.float16), "Mix": np.zeros((N, H), np.float16),
              "Logits": np.zeros((N, RR), np.float16), "Ids": np.zeros((N, TOPK), np.int32),
              "Wts": np.zeros((N, 16), np.float32), "Cnt": np.zeros((E + 1,), np.int32),
              "Off": np.zeros((E + 2,), np.int32), "Slot": np.zeros((SLOTS,), np.int32),
              "PartE": np.zeros((TILES, N, H), np.float32), "MoeOut": np.zeros((N, H), np.float32)}
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), tiles=TILES, shaves=SHAVES))
    return hio, kc, vt


def _oracle(p, n=N, pos0=0):
    return attn_moe_layer_prefill_reference(p["h"], p["inn"], p["post"], p["wq"], p["wqs"], p["qn"], p["kn"],
                                            _cos_sin(pos0), p["wo"], p["wos"],
                                            np.zeros((NKV, SMAX, D), np.float16),
                                            np.zeros((NKV, D, SMAX), np.float16),
                                            p["wr"], p["gu_int"], p["gu_s"], p["d_int"], p["d_s"],
                                            pos0=pos0, n=n, NH=NH, NKV=NKV, D=D, ROT=ROT, E=E, TOPK=TOPK, G=G)


def test_the_layer_does_not_reintroduce_question_10():
    src = lowering.lower(_kernel()).source
    # depth, not membership, for the reason the GDN+MoE layer's test gives: the expert phase's buffers sit at
    # depth 1, which is the form P2b earned its device trials in
    assert max((d for _, d in allocations_inside_loops(src)), default=0) <= 1
    assert src.count("ctx.sync_grid();") >= 8   # ten phases, each dealt differently
    assert "ctx.alloc" in src      # control: the needle the first assertion looks for is present at all
    assert "ddr_view" not in src   # the caches stay kernel parameters (question 8)


@needs_cxx
def test_the_layer_matches_the_composed_oracle():
    p = _weights()
    hio, kc, vt = _run(p)
    ref_h, kc_ref, vt_ref = _oracle(p)
    assert np.max(np.abs(hio - ref_h)) / np.max(np.abs(ref_h)) < REL
    for name, got, exp in (("Kc", kc, kc_ref), ("Vt", vt, vt_ref)):
        rel = (np.abs(got.astype(np.float32) - exp.astype(np.float32)).max()
               / max(float(np.abs(exp.astype(np.float32)).max()), 1e-6))
        assert rel < REL, f"{name} rel {rel}"


@needs_cxx
def test_both_halves_add_to_the_residual_rather_than_replacing_it():
    """Zero the o_proj and every expert's down projection: the layer then contributes nothing and the stream must
    come back bit-identical. A layer that assigned instead of accumulating passes the tolerance test above and
    fails this one."""
    p = _weights()
    p["wo"] = np.zeros_like(p["wo"])
    p["wos"] = np.zeros_like(p["wos"])
    p["d_int"] = np.zeros_like(p["d_int"])
    p["d_s"] = np.zeros_like(p["d_s"])
    hio, kc, _ = _run(p)
    assert np.max(np.abs(hio - p["h"])) == 0.0
    assert np.abs(kc.astype(np.float32)).max() > 0, "positive control: the cache was still filled"


@needs_cxx
def test_a_short_prompt_leaves_the_dead_tokens_alone():
    """Unlike the GDN+MoE layer, BOTH halves here skip t >= Nt. The mixer half must (a zero query gives a uniform
    softmax, not a zero), and the MoE half follows it so that the stream past the live count is exactly what came
    in -- which is what a chunked prefill's last partial chunk relies on."""
    p = _weights()
    n = N - TB
    hio, kc, _ = _run(p, n=n)
    ref_h, _, _ = _oracle(p, n=n)
    assert np.max(np.abs(hio[:n] - ref_h[:n])) / np.max(np.abs(ref_h[:n])) < REL
    assert np.max(np.abs(hio[n:] - p["h"][n:])) == 0.0, "a dead token was added to the stream"
    assert np.abs(kc[:, n:].astype(np.float32)).max() == 0, "a dead token was appended to the cache"
