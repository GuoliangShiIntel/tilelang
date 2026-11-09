"""P4 step 4: a whole Gated DeltaNet PREFILL mixer layer in one launch (`make_gdn_layer_prefill`) --
norm -> in_proj -> conv -> chunked scan -> out_proj -> residual, over N prompt tokens.

The layer's claim is that the five phases compose, so the oracle is built from the phase oracles rather than
written again, and the tests go after what composition can break and a single phase cannot: that the residual is
added (not replaced), that a short prompt leaves the dead tokens untouched, and that the barriers really separate
phases whose work is dealt differently -- by output-row chunk, by head, by v-head.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import gdn_layer_prefill_reference, gm_scales, make_gdn_layer_prefill, pack_int4  # noqa: E402
from test_intel_npu_scoped_cmx import allocations_inside_loops  # noqa: E402

TILES, SHAVES = 2, 2
N, H, KH, VH, DK, DV, G = 64, 256, 4, 8, 64, 64, 128
TB, C, BN1, BN2, MB = 32, 32, 64, 64, 32
QK, VC = KH * DK, VH * DV
CONV = 2 * QK + VC
NIN = CONV + VC + 2 * VH
NIN_pad = -(-NIN // BN1) * BN1
REL = 5e-2


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    h = (rng.standard_normal((N, H), np.float32) * 0.5).astype(np.float32)
    inn = (rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16)
    wi = rng.integers(-8, 8, (NIN_pad, H), dtype=np.int8)
    ws = (rng.standard_normal((NIN_pad, H // G), np.float32) * 0.02).astype(np.float16)
    wo = rng.integers(-8, 8, (H, VC), dtype=np.int8)
    wos = (rng.standard_normal((H, VC // G), np.float32) * 0.02).astype(np.float16)
    cw = (rng.standard_normal((4, CONV), np.float32) * 0.3).astype(np.float16)
    cs = (rng.standard_normal((3, CONV), np.float32) * 0.3).astype(np.float16)
    al = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    db = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    gn = (1.0 + rng.standard_normal((1, DV), np.float32) * 0.1).astype(np.float16)
    ssm = (rng.standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32)
    return h, inn, wi, ws, wo, wos, cw, cs, al, db, gn, ssm


def _run(h, inn, wi, ws, wo, wos, cw, cs, al, db, gn, ssm, n=N):
    low = lowering.lower(make_gdn_layer_prefill(N=N, H=H, KH=KH, VH=VH, DK=DK, DV=DV, G=G, TB=TB, C=C,
                                                BN1=BN1, BN2=BN2, MB=MB, tiles=TILES, shaves=SHAVES))
    lib = host_sim.build(low.source, "gdn_layer_prefill")
    hio, sm, cso = h.copy(), ssm.copy(), np.zeros((3, CONV), np.float16)
    arrays = {"Hio": hio, "InNorm": inn, "WinQ": pack_int4(wi), "WinS": gm_scales(ws), "ConvW": cw, "ConvState": cs,
              "ALog": al, "DtBias": db, "GNorm": gn, "Nt": np.array([[n]], np.int32), "Ssm": sm,
              "WoutQ": pack_int4(wo), "WoutS": gm_scales(wos),
              "Xs": np.zeros((N, H), np.float16), "Ys": np.zeros((N, NIN_pad), np.float16),
              "QKs": np.zeros((KH, N, 2 * DK), np.float32), "Vs": np.zeros((VH, N, DV), np.float16),
              "Gbs": np.zeros((VH, N, 2), np.float32), "Zs": np.zeros((VH, N, DV), np.float16),
              "Os": np.zeros((N, VC), np.float16), "Mix": np.zeros((N, H), np.float16), "CsOut": cso}
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), tiles=TILES, shaves=SHAVES))
    return hio, sm, cso


def test_the_layer_does_not_reintroduce_question_10():
    src = lowering.lower(make_gdn_layer_prefill(N=N, H=H, KH=KH, VH=VH, DK=DK, DV=DV, G=G, TB=TB, C=C,
                                                BN1=BN1, BN2=BN2, MB=MB, tiles=TILES, shaves=SHAVES)).source
    assert allocations_inside_loops(src) == []
    # five phases deal their work differently, so each boundary needs a barrier
    assert src.count("ctx.sync_grid") >= 5, f"grid barriers: {src.count('ctx.sync_grid')}"


@needs_cxx
def test_matches_the_composed_oracle():
    w = _weights()
    hio, sm, cso = _run(*w)
    ref, cs_ref, ssm_ref = gdn_layer_prefill_reference(w[0], w[1], w[2], w[3], w[6], w[7], w[8], w[9], w[10],
                                                       w[4], w[5], w[11], N, KH, VH, DK, DV, G)
    assert np.abs(hio - ref).max() / np.abs(ref).max() < REL
    assert np.abs(sm - ssm_ref).max() / np.abs(ssm_ref).max() < REL
    # not bit-exact here, unlike the standalone conv test: inside the layer the conv's input is the fp16 gemm
    # output of the projection, so the handed-on state carries the projection's own error
    cs_rel = (np.abs(cso.astype(np.float32) - cs_ref.astype(np.float32)).max()
              / np.abs(cs_ref.astype(np.float32)).max())
    assert cs_rel < REL, f"conv state rel {cs_rel}"


@needs_cxx
def test_the_residual_is_added_not_replaced():
    """Zero every projection and the layer must hand its input straight back. A layer that wrote the mixer output
    over the stream instead of adding to it passes the oracle test (the oracle would be wrong the same way) but
    fails this one."""
    h, inn, wi, ws, wo, wos, cw, cs, al, db, gn, ssm = _weights()
    hio, _, _ = _run(h, inn, wi, ws, np.zeros_like(wo), np.zeros_like(wos), cw, cs, al, db, gn, ssm)
    assert np.abs(hio - h).max() == 0, "with a zero out_proj the layer must be the identity on the stream"


@needs_cxx
def test_a_short_prompt_leaves_the_dead_tokens_alone():
    n = 50
    h, inn, wi, ws, wo, wos, cw, cs, al, db, gn, ssm = _weights()
    hio, _, _ = _run(h, inn, wi, ws, wo, wos, cw, cs, al, db, gn, ssm, n=n)
    assert np.abs(hio[n:] - h[n:]).max() == 0, "tokens past the live count were written"
    assert np.abs(hio[:n] - h[:n]).max() > 0
