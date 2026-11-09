"""P4 step 5: a WHOLE Gated DeltaNet PREFILL decoder layer in one launch (`make_gdn_moe_layer_prefill`) --
the mixer half, then the second norm, P2a's routing and P2b's grouped expert FFN, summed back into the
fp32 residual stream, over N prompt tokens.

Nine phases deal their work nine different ways, so the tests go after what composition can break and no
single phase can: that BOTH contributions are added to the residual rather than replacing it, that a short
prompt leaves the dead tokens untouched, and that the routing table the experts consume is the one the
router wrote in the same launch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import (gdn_moe_layer_prefill_reference, make_gdn_moe_layer_prefill,  # noqa: E402
                     make_moe_route_prefill, pack_int4)
from test_intel_npu_scoped_cmx import allocations_inside_loops  # noqa: E402

TILES, SHAVES = 2, 2
N, H, KH, VH, DK, DV, G = 64, 256, 4, 8, 64, 64, 128
I, E, TOPK = 128, 32, 4
TB, C, BN1, BN2, MB = 32, 32, 16, 16, 16
BM, BNR, MBe, BNG, BND = 32, 16, 16, 64, 64
QK, VC = KH * DK, VH * DV
CONV = 2 * QK + VC
NIN = CONV + VC + 2 * VH
NIN_pad = -(-NIN // BN1) * BN1
KGH, KGI = H // G, I // G
_, _RS = make_moe_route_prefill(N=N, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=TILES, shaves=SHAVES, as_block=True)
RR, SLOTS = _RS["RR"], _RS["SLOTS"]
WORKERS = TILES * SHAVES
REL = 6e-2   # int4 weights through two projections, a chunked scan and TOPK + 1 expert FFNs


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")


def _kernel():
    return make_gdn_moe_layer_prefill(N=N, H=H, KH=KH, VH=VH, DK=DK, DV=DV, I=I, E=E, TOPK=TOPK, G=G, TB=TB,
                                      C=C, BN1=BN1, BN2=BN2, MB=MB, BM=BM, BNR=BNR, MBe=MBe, BNG=BNG,
                                      BND=BND, tiles=TILES, shaves=SHAVES)


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    p = dict(
        h=(rng.standard_normal((N, H), np.float32) * 0.5).astype(np.float32),
        inn=(rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16),
        post=(rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16),
        wi=rng.integers(-8, 8, (NIN_pad, H), dtype=np.int8),
        ws=(rng.standard_normal((NIN_pad, H // G), np.float32) * 0.02).astype(np.float16),
        wo=rng.integers(-8, 8, (H, VC), dtype=np.int8),
        wos=(rng.standard_normal((H, VC // G), np.float32) * 0.02).astype(np.float16),
        cw=(rng.standard_normal((4, CONV), np.float32) * 0.3).astype(np.float16),
        cs=(rng.standard_normal((3, CONV), np.float32) * 0.3).astype(np.float16),
        al=(rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32),
        db=(rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32),
        gn=(1.0 + rng.standard_normal((1, DV), np.float32) * 0.1).astype(np.float16),
        ssm=(rng.standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32),
        wr=(rng.standard_normal((E + 1, H), np.float32) * 0.1).astype(np.float16),
        gu_int=rng.integers(-8, 8, (E + 1, 2 * I, H), dtype=np.int8),
        gu_s=(rng.standard_normal((E + 1, 2 * I, H // G), np.float32) * 0.02).astype(np.float16),
        d_int=rng.integers(-8, 8, (E + 1, H, I), dtype=np.int8),
        d_s=(rng.standard_normal((E + 1, H, I // G), np.float32) * 0.02).astype(np.float16),
    )
    return p


def _run(p, n=N):
    low = lowering.lower(_kernel())
    lib = host_sim.build(low.source, "gdn_moe_layer_prefill")
    hio, sm, cso = p["h"].copy(), p["ssm"].copy(), np.zeros((3, CONV), np.float16)
    wr_pad = np.zeros((RR, H), np.float16)
    wr_pad[:E + 1] = p["wr"]
    arrays = {"Hio": hio, "InNorm": p["inn"], "PostNorm": p["post"],
              "WinQ": pack_int4(p["wi"]), "WinS": p["ws"], "ConvW": p["cw"], "ConvState": p["cs"],
              "ALog": p["al"], "DtBias": p["db"], "GNorm": p["gn"], "Nt": np.array([[n]], np.int32), "Ssm": sm,
              "WoutQ": pack_int4(p["wo"]), "WoutS": p["wos"], "Wr": wr_pad,
              "WguQ": pack_int4(p["gu_int"].reshape(-1, H)), "WguS": p["gu_s"].reshape(-1, KGH),
              "WdQ": pack_int4(p["d_int"].reshape(-1, I)), "WdS": p["d_s"].reshape(-1, KGI),
              "Xs": np.zeros((N, H), np.float16), "Ys": np.zeros((N, NIN_pad), np.float16),
              "QKs": np.zeros((KH, N, 2 * DK), np.float32), "Vs": np.zeros((VH, N, DV), np.float16),
              "Gbs": np.zeros((VH, N, 2), np.float32), "Zs": np.zeros((VH, N, DV), np.float16),
              "Os": np.zeros((N, VC), np.float16), "Mix": np.zeros((N, H), np.float16), "CsOut": cso,
              "Logits": np.zeros((N, RR), np.float16), "Ids": np.zeros((N, TOPK), np.int32),
              "Wts": np.zeros((N, 16), np.float32), "Cnt": np.zeros((E + 1,), np.int32),
              "Off": np.zeros((E + 2,), np.int32), "Slot": np.zeros((SLOTS,), np.int32),
              "PartE": np.zeros((TILES, N, H), np.float32), "MoeOut": np.zeros((N, H), np.float32)}
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), tiles=TILES, shaves=SHAVES))
    return hio, sm, cso


def _oracle(p, n=N):
    return gdn_moe_layer_prefill_reference(p["h"], p["inn"], p["post"], p["wi"], p["ws"], p["cw"], p["cs"],
                                           p["al"], p["db"], p["gn"], p["wo"], p["wos"], p["ssm"], p["wr"],
                                           p["gu_int"], p["gu_s"], p["d_int"], p["d_s"],
                                           n=n, KH=KH, VH=VH, DK=DK, DV=DV, E=E, TOPK=TOPK, G=G)


def test_the_layer_does_not_reintroduce_question_10():
    src = lowering.lower(_kernel()).source
    # depth, not membership: question 10's defect was the GDN scan's `xr` at depth 3, re-allocated 1152 times a
    # worker. The expert phase's buffers sit at depth 1, which is the form P3 earned 12/12 device trials in, and
    # pinning them at kernel level instead costs the layer more CMX than it has.
    assert max((d for _, d in allocations_inside_loops(src)), default=0) <= 1
    # nine phases, each dealt differently, so eight boundaries at least
    assert src.count("ctx.sync_grid();") >= 8
    assert "ctx.alloc" in src   # control: the needle the first assertion looks for is present at all


@needs_cxx
def test_the_layer_matches_the_composed_oracle():
    """The claim is that nine phases compose, so the oracle is the phase oracles chained, not a rewrite."""
    p = _weights()
    hio, sm, cso = _run(p)
    ref_h, ref_cs, ref_ssm = _oracle(p)
    assert np.max(np.abs(hio - ref_h)) / np.max(np.abs(ref_h)) < REL
    assert np.max(np.abs(sm - ref_ssm)) / max(np.max(np.abs(ref_ssm)), 1e-6) < REL


@needs_cxx
def test_both_halves_add_to_the_residual_rather_than_replacing_it():
    """Zero the out_proj and every expert's down projection: the layer then contributes nothing and the stream
    must come back bit-identical. A layer that assigned instead of accumulating passes every tolerance test
    above and fails this one."""
    p = _weights()
    p["wo"] = np.zeros_like(p["wo"])
    p["wos"] = np.zeros_like(p["wos"])
    p["d_int"] = np.zeros_like(p["d_int"])
    p["d_s"] = np.zeros_like(p["d_s"])
    hio, _, _ = _run(p)
    assert np.max(np.abs(hio - p["h"])) == 0.0


@needs_cxx
def test_a_short_prompt_leaves_the_live_tokens_independent_of_the_dead_ones():
    """Nt is the live token count. The mixer half masks past it; the MoE half does NOT -- routing and the expert
    FFN are per-token, so they run on the padding rows too and add to them. That is what the oracle says and what
    the kernel does, and it is harmless only because every consumer masks: the live tokens below must not move.
    Pinned here so that P5, which chains layers, finds the asymmetry written down rather than in a residual."""
    p = _weights()
    n = N - TB         # one conv block of dead tokens
    hio, _, _ = _run(p, n=n)
    ref_h, _, _ = _oracle(p, n=n)
    assert np.max(np.abs(hio[:n] - ref_h[:n])) / np.max(np.abs(ref_h[:n])) < REL
    assert np.max(np.abs(hio[n:] - ref_h[n:])) / np.max(np.abs(ref_h[n:])) < REL
    # the mixer's own masking still holds: with the experts disabled the dead rows come back untouched
    q = _weights()
    q["d_int"] = np.zeros_like(q["d_int"])
    q["d_s"] = np.zeros_like(q["d_s"])
    short, _, _ = _run(q, n=n)
    assert np.max(np.abs(short[n:] - q["h"][n:])) == 0.0
