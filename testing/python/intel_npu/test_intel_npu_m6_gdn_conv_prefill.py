"""P3b of prefill: the Gated DeltaNet phase between the input projection and the chunked scan
(`make_gdn_conv_prefill`) -- a causal conv1d of width 4 with SiLU, then the per-head normalisations.

It is written to produce exactly what `make_gdn_scan_prefill` consumes, so the two compose; the test that matters
is the one that feeds P3b's output straight into P3 and compares against the two oracles chained the same way.
The conv is the part a per-token loop would get right by construction and a blocked one can get wrong, so the
tests pin causality, the chunk boundary (the 3-token window handed from one chunk to the next) and the masking of
rows past the live token count.
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
from kernels import (gdn_conv_prefill_reference, gdn_scan_prefill_reference, make_gdn_conv_prefill,  # noqa: E402
                     make_gdn_scan_prefill)

TILES, SHAVES = 2, 2
KH, VH, DK, DV, TB = 4, 8, 64, 64, 32
QK, VC = KH * DK, VH * DV
CONV = 2 * QK + VC
NIN = CONV + VC
REL = 5e-2


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")


def _problem(n, NP, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.standard_normal((NP, NIN), np.float32) * 0.5).astype(np.float16)
    y[n:] = 0
    ba = (rng.standard_normal((NP, 2 * VH), np.float32) * 0.5).astype(np.float16)
    ba[n:] = 0
    conv_w = (rng.standard_normal((4, CONV), np.float32) * 0.3).astype(np.float16)
    cs = (rng.standard_normal((3, CONV), np.float32) * 0.3).astype(np.float16)
    a_log = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    dt_bias = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    return y, ba, conv_w, cs, a_log, dt_bias


def _sim(n, N, y, ba, conv_w, cs, a_log, dt_bias, tiles=TILES, shaves=SHAVES, register_conv=False, conv_fp16=False,
         live_chunks=False, sentinel=False):
    NP = -(-N // TB) * TB
    low = lowering.lower(make_gdn_conv_prefill(N=NP, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB,
                                               tiles=tiles, shaves=shaves,
                                               register_conv=register_conv, conv_fp16=conv_fp16, live_chunks=live_chunks))
    lib = host_sim.build(low.source, f"gdn_conv_prefill_{NP}_{tiles}x{shaves}")
    out = dict(QKo=np.zeros((KH, NP, 2 * DK), np.float32), Vo=np.zeros((VH, NP, DV), np.float16),
               Gb=np.zeros((VH, NP, 2), np.float32), Zo=np.zeros((VH, NP, DV), np.float16),
               CsOut=np.zeros((3, CONV), np.float16))
    if sentinel:
        for value in out.values():
            value.fill(np.nan)
    arrays = {"Y": y, "Ba": ba, "ConvW": conv_w, "ConvState": cs, "ALog": a_log, "DtBias": dt_bias,
              "Nt": np.array([[n]], np.int32), **out}
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), tiles=tiles, shaves=shaves))
    return out


def test_lowers_without_allocating_inside_a_loop():
    """The authoring rule question 10 left behind: a buffer first used before the loops keeps its CMX scope at
    kernel level. This kernel is new, so it starts clean and the test says so."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_intel_npu_scoped_cmx import allocations_inside_loops
    src = lowering.lower(make_gdn_conv_prefill(N=64, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB,
                                               tiles=TILES, shaves=SHAVES)).source
    assert allocations_inside_loops(src) == []
    assert "ctx.alloc" in src   # the control: the detector is looking at a kernel that does allocate


@needs_cxx
@pytest.mark.parametrize("register_conv,conv_fp16", [(False, False), (True, False), (True, True)])
def test_matches_the_reference(register_conv, conv_fp16):
    n = N = 64
    y, ba, cw, cs, al, db = _problem(n, N)
    got = _sim(n, N, y, ba, cw, cs, al, db, register_conv=register_conv, conv_fp16=conv_fp16)
    qk, v, gb, z, new_cs = gdn_conv_prefill_reference(y, cw, cs, al, db, n, KH, VH, DK, DV, ba=ba)
    for name, dev, ref in (("qk", got["QKo"].reshape(KH, N, 2, DK), qk), ("v", got["Vo"], v),
                           ("gb", got["Gb"], gb), ("z", got["Zo"], z), ("conv_state", got["CsOut"], new_cs)):
        d = np.abs(dev.astype(np.float32) - ref.astype(np.float32)).max() / (np.abs(ref.astype(np.float32)).max() + 1e-9)
        assert d < REL, f"{name} rel {d}"


@needs_cxx
@pytest.mark.parametrize("register_conv,conv_fp16", [(False, False), (True, False), (True, True)])
def test_the_conv_is_causal_in_the_tokens(register_conv, conv_fp16):
    """Perturbing one token may move its own output and the next three, and nothing earlier -- the width of the
    kernel. A conv that read the whole block at once, or lost the window at a block boundary, would not."""
    n = N = 64
    y, ba, cw, cs, al, db = _problem(n, N)
    base = _sim(n, N, y, ba, cw, cs, al, db, register_conv=register_conv, conv_fp16=conv_fp16)["QKo"]
    t = 29                                        # straddles the TB = 32 block boundary with its 3-token reach
    y2 = y.copy()
    y2[t] = (y2[t].astype(np.float32) * 1.5).astype(np.float16)
    moved = _sim(n, N, y2, ba, cw, cs, al, db, register_conv=register_conv, conv_fp16=conv_fp16)["QKo"]
    d = np.abs(moved.astype(np.float32) - base.astype(np.float32)).max(axis=(0, 2))
    assert d[:t].max() == 0, "an earlier token moved"
    assert d[t] > 0 and d[t + 3] > 0, "the token and its three successors must move"
    assert d[t + 4:].max() == 0, "the kernel is 4 wide; a fifth token must not move"


@needs_cxx
@pytest.mark.parametrize("register_conv,conv_fp16", [(False, False), (True, False), (True, True)])
def test_the_handed_on_state_continues_the_sequence(register_conv, conv_fp16):
    """Two chunks of 32 tokens, the second started from the first's returned conv state, must equal one chunk of
    64. This is the only property that makes a long prompt possible and a single-chunk test cannot see it."""
    y, ba, cw, cs, al, db = _problem(64, 64)
    whole = _sim(64, 64, y, ba, cw, cs, al, db, register_conv=register_conv, conv_fp16=conv_fp16)
    first = _sim(32, 32, y[:32], ba[:32], cw, cs, al, db, register_conv=register_conv, conv_fp16=conv_fp16)
    second = _sim(32, 32, y[32:64], ba[32:64], cw, first["CsOut"], al, db, register_conv=register_conv, conv_fp16=conv_fp16)
    for name, key in (("qk", "QKo"), ("v", "Vo")):
        a = whole[key].astype(np.float32)
        joined = np.concatenate([first[key].astype(np.float32), second[key].astype(np.float32)], axis=1)
        d = np.abs(a - joined).max() / (np.abs(a).max() + 1e-9)
        assert d < REL, f"{name} across the chunk boundary rel {d}"


@needs_cxx
@pytest.mark.parametrize("register_conv,conv_fp16", [(False, False), (True, False), (True, True)])
def test_rows_past_the_live_count_are_zero(register_conv, conv_fp16):
    n, N = 50, 64
    y, ba, cw, cs, al, db = _problem(n, N)
    got = _sim(n, N, y, ba, cw, cs, al, db, register_conv=register_conv, conv_fp16=conv_fp16)
    assert np.abs(got["QKo"][:, n:]).max() == 0 and np.abs(got["Vo"][:, n:]).max() == 0
    assert np.abs(got["Gb"][:, n:]).max() == 0 and np.abs(got["Zo"][:, n:]).max() == 0
    assert np.abs(got["QKo"][:, :n]).max() > 0, "the live rows are not all zero either"


@needs_cxx
@pytest.mark.parametrize("live", [0, 1, 31, 32, 33, 50])
def test_live_chunks_zero_inactive_capacity_and_preserve_state(live):
    capacity = 128
    y, ba, cw, state, al, db = _problem(live, capacity, seed=19)
    got = _sim(live, capacity, y, ba, cw, state, al, db, conv_fp16=True,
               live_chunks=True, sentinel=True)
    for value in got.values():
        assert np.isfinite(value).all()
    assert np.count_nonzero(got["QKo"][:, live:]) == 0
    for name in ("Vo", "Gb", "Zo"):
        assert np.count_nonzero(got[name][:, live:]) == 0
    expected_state = gdn_conv_prefill_reference(y, cw, state, al, db, live, KH, VH, DK, DV, ba=ba)[-1]
    np.testing.assert_array_equal(got["CsOut"], expected_state)


@needs_cxx
def test_it_feeds_the_scan():
    """P3b into P3, against the two oracles chained the same way. This is what the pair has to do for P4."""
    n = N = 64
    C = 32
    y, ba, cw, cs, al, db = _problem(n, N)
    conv = _sim(n, N, y, ba, cw, cs, al, db)
    gnorm = (1.0 + np.random.default_rng(1).standard_normal((1, DV), np.float32) * 0.1).astype(np.float16)
    ssm = (np.random.default_rng(2).standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32)
    low = lowering.lower(make_gdn_scan_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, C=C, tiles=TILES, shaves=SHAVES))
    lib = host_sim.build(low.source, "gdn_scan_after_conv")
    out, sm = np.zeros((N, VC), np.float16), ssm.copy()
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, {
        "QK": conv["QKo"], "V": conv["Vo"], "Gb": conv["Gb"], "Z": conv["Zo"], "GNorm": gnorm,
        "Nt": np.array([[n]], np.int32), "Ssm": sm, "Out": out}), tiles=TILES, shaves=SHAVES))
    qk, v, gb, z, _ = gdn_conv_prefill_reference(y, cw, cs, al, db, n, KH, VH, DK, DV, ba=ba)
    ref, _ = gdn_scan_prefill_reference(qk, v, gb, z, gnorm, ssm, n)
    rel = np.abs(out.astype(np.float32)[:n] - ref[:n]).max() / np.abs(ref[:n]).max()
    assert rel < REL, f"conv -> scan rel {rel}"
