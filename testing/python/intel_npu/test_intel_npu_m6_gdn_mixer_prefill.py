"""P4 step 3: the GDN prefill mixer in ONE launch (`make_gdn_mixer_prefill`) -- the conv phase and the chunked
scan with a grid barrier between them, instead of two launches round-tripping the intermediates through the host.

The point of the fused form is that it must equal the two-launch form exactly, so that is the test: the same
inputs through the pair and through the fused kernel, compared to each other and to the two oracles chained. And
because composing changes the loop nesting -- which is what plan question 10 acts on -- the exposure detector runs
on the fused kernel too.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import (gdn_conv_prefill_reference, gdn_scan_prefill_reference, make_gdn_conv_prefill,  # noqa: E402
                     make_gdn_mixer_prefill, make_gdn_scan_prefill)
from test_intel_npu_scoped_cmx import allocations_inside_loops  # noqa: E402

TILES, SHAVES = 2, 2
KH, VH, DK, DV, TB, C = 4, 8, 64, 64, 32, 32
QK, VC = KH * DK, VH * DV
CONV = 2 * QK + VC
NIN = CONV + VC + 2 * VH
N = 64
REL = 5e-2


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.standard_normal((N, NIN), np.float32) * 0.5).astype(np.float16)
    cw = (rng.standard_normal((4, CONV), np.float32) * 0.3).astype(np.float16)
    cs = (rng.standard_normal((3, CONV), np.float32) * 0.3).astype(np.float16)
    al = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    db = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    gn = (1.0 + rng.standard_normal((1, DV), np.float32) * 0.1).astype(np.float16)
    ssm = (rng.standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32)
    return y, cw, cs, al, db, gn, ssm


def _fused(y, cw, cs, al, db, gn, ssm, n=N):
    low = lowering.lower(make_gdn_mixer_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB, C=C,
                                                tiles=TILES, shaves=SHAVES))
    lib = host_sim.build(low.source, "gdn_mixer_prefill")
    out, sm, cso = np.zeros((N, VC), np.float16), ssm.copy(), np.zeros((3, CONV), np.float16)
    arrays = {"Y": y, "ConvW": cw, "ConvState": cs, "ALog": al, "DtBias": db, "GNorm": gn,
              "Nt": np.array([[n]], np.int32), "Ssm": sm,
              "QKs": np.zeros((KH, N, 2 * DK), np.float32), "Vs": np.zeros((VH, N, DV), np.float16),
              "Gbs": np.zeros((VH, N, 2), np.float32), "Zs": np.zeros((VH, N, DV), np.float16),
              "CsOut": cso, "Out": out}
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), tiles=TILES, shaves=SHAVES))
    return out, sm, cso


def _two_launches(y, cw, cs, al, db, gn, ssm, n=N):
    clow = lowering.lower(make_gdn_conv_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB, tiles=TILES, shaves=SHAVES))
    clib = host_sim.build(clow.source, "gdn_conv_for_pair")
    c = dict(QKo=np.zeros((KH, N, 2 * DK), np.float32), Vo=np.zeros((VH, N, DV), np.float16),
             Gb=np.zeros((VH, N, 2), np.float32), Zo=np.zeros((VH, N, DV), np.float16),
             CsOut=np.zeros((3, CONV), np.float16))
    host_sim.check_status(host_sim.run(clib, lowering.order_arrays(clow, {
        "Y": y, "ConvW": cw, "ConvState": cs, "ALog": al, "DtBias": db,
        "Nt": np.array([[n]], np.int32), **c}), tiles=TILES, shaves=SHAVES))
    slow = lowering.lower(make_gdn_scan_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, C=C, tiles=TILES, shaves=SHAVES))
    slib = host_sim.build(slow.source, "gdn_scan_for_pair")
    out, sm = np.zeros((N, VC), np.float16), ssm.copy()
    host_sim.check_status(host_sim.run(slib, lowering.order_arrays(slow, {
        "QK": c["QKo"], "V": c["Vo"], "Gb": c["Gb"], "Z": c["Zo"], "GNorm": gn,
        "Nt": np.array([[n]], np.int32), "Ssm": sm, "Out": out}), tiles=TILES, shaves=SHAVES))
    return out, sm, c["CsOut"]


def test_fusing_does_not_reintroduce_question_10():
    src = lowering.lower(make_gdn_mixer_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB, C=C,
                                                tiles=TILES, shaves=SHAVES)).source
    assert allocations_inside_loops(src) == []
    assert src.count("ctx.sync_grid") >= 1, "the two phases must be separated by a grid barrier"


@needs_cxx
def test_one_launch_equals_two():
    """The fused kernel must equal the pair bit for bit in the simulation: the only difference is where the
    intermediates live, and a barrier that did not cover them would show up here."""
    y, cw, cs, al, db, gn, ssm = _problem()
    a_out, a_sm, a_cs = _fused(y, cw, cs, al, db, gn, ssm)
    b_out, b_sm, b_cs = _two_launches(y, cw, cs, al, db, gn, ssm)
    assert np.array_equal(a_out, b_out), "fused and two-launch outputs differ"
    assert np.array_equal(a_sm, b_sm) and np.array_equal(a_cs, b_cs)


@needs_cxx
def test_it_matches_the_chained_oracles():
    y, cw, cs, al, db, gn, ssm = _problem()
    out, sm, cso = _fused(y, cw, cs, al, db, gn, ssm)
    qk, v, gb, z, new_cs = gdn_conv_prefill_reference(y, cw, cs, al, db, N, KH, VH, DK, DV)
    ref, sref = gdn_scan_prefill_reference(qk, v, gb, z, gn, ssm, N)
    rel = np.abs(out.astype(np.float32) - ref).max() / np.abs(ref).max()
    assert rel < REL, f"mixer rel {rel}"
    assert np.abs(sm - sref).max() / np.abs(sref).max() < REL
    assert np.abs(cso.astype(np.float32) - new_cs.astype(np.float32)).max() == 0


@needs_cxx
def test_a_short_prompt_is_masked():
    n = 50
    y, cw, cs, al, db, gn, ssm = _problem()
    y[n:] = 0
    out, _, _ = _fused(y, cw, cs, al, db, gn, ssm, n=n)
    assert np.abs(out[n:]).max() == 0 and np.abs(out[:n]).max() > 0
