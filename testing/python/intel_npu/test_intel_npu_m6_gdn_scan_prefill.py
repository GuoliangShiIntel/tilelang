"""P3 of prefill: the Gated DeltaNet recurrence in chunked form (`make_gdn_scan_prefill`).

The oracle runs the RECURRENCE token by token, so these tests check the chunked algebra -- the triangular
inverse, the decay matrix D and the chunk-to-chunk carry -- against the definition it is supposed to equal,
not against a second copy of itself.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401  (loads the dev build before tvm)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import kernels  # noqa: E402
import lowering  # noqa: E402
from kernels import gdn_scan_prefill_reference, make_gdn_scan_prefill  # noqa: E402


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")

TILES, SHAVES = 3, 2
KH, VH, DK, DV = 4, 8, 128, 128
C = 64
EPS = 1e-6


def _problem(n, NP, seed=0):
    rng = np.random.default_rng(seed)
    qk = rng.standard_normal((KH, NP, 2, DK), np.float32) * 0.3
    for h in range(KH):                                  # q and k reach the scan l2-normalised
        for t in range(NP):
            for i in range(2):
                qk[h, t, i] /= np.sqrt((qk[h, t, i] ** 2).sum()) + 1e-6
    qk[:, n:] = 0
    v = (rng.standard_normal((VH, NP, DV), np.float32) * 0.5).astype(np.float16)
    gb = np.zeros((VH, NP, 2), np.float32)
    gb[:, :, 0] = -np.abs(rng.standard_normal((VH, NP), np.float32)) * 0.05    # g < 0: the decay
    gb[:, :, 1] = 1.0 / (1.0 + np.exp(-rng.standard_normal((VH, NP), np.float32)))   # beta in (0, 1)
    z = (rng.standard_normal((VH, NP, DV), np.float32) * 0.5).astype(np.float16)
    gnorm = (1.0 + rng.standard_normal((1, DV), np.float32) * 0.1).astype(np.float16)
    ssm = (rng.standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32)
    return qk, v, gb, z, gnorm, ssm


def _sim(n, N, qk, v, gb, z, gnorm, ssm, tiles=TILES, shaves=SHAVES, register_tri=False,
         live_chunks=False, sentinel=False):
    NP = -(-N // C) * C
    low = lowering.lower(make_gdn_scan_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, C=C, tiles=tiles, shaves=shaves,
                                               eps=EPS, register_tri=register_tri, live_chunks=live_chunks))
    lib = host_sim.build(low.source, "gdn_scan_prefill")
    out = np.zeros((NP, VH * DV), np.float16)
    if sentinel:
        out.fill(np.nan)
    sm = ssm.copy()
    arrays = {"QK": qk.reshape(KH, qk.shape[1], 2 * DK), "V": v, "Gb": gb, "Z": z, "GNorm": gnorm,
              "Nt": np.array([[n]], np.int32), "Ssm": sm, "Out": out}
    status = host_sim.run(lib, lowering.order_arrays(low, arrays), tiles=tiles, shaves=shaves)
    host_sim.check_status(status)
    return out, sm


def test_lowers_with_the_eleven_chunk_matmuls():
    src = lowering.lower(make_gdn_scan_prefill(N=128, KH=KH, VH=VH, DK=DK, DV=DV, C=C,
                                               tiles=TILES, shaves=SHAVES)).source
    assert "kernel_entry" in src
    # the two workloads that the old "8192 output points" reading refused outright: dS and the state transpose
    assert src.count("ctx.dpu_matmul((&(dS[0]))") == 1
    assert f", {DK}, {DV}, {C}," in src                 # dS: 128 x 128 output values from a K of 64
    # the chunk loop must carry eleven, and S0's transpose must stay OUT of it -- one per head, not per chunk.
    # Nesting depth is what separates them, so hoisting or sinking a matmul by mistake shows up here.
    depths = [len(m.group(1)) for m in re.finditer(r"^(\s*)ctx\.dpu_matmul(?:_start)?\(", src, re.M)]
    assert len(depths) == 12, f"matmuls emitted: {len(depths)}"
    assert depths.count(min(depths)) == 1, "exactly one matmul sits above the chunk loop (the S0 transpose)"
    assert depths.count(max(depths)) == 11, f"per-chunk matmuls: {depths.count(max(depths))}"
    assert "ctx.dpu_matmul_start(" not in src
    assert "ctx.dpu_matmul_not_a_real_symbol(" not in src    # control: the needle still discriminates


@needs_cxx
@pytest.mark.parametrize("register_tri", [False, True])
def test_chunked_scan_matches_the_recurrence(register_tri):
    n, N = 128, 128
    qk, v, gb, z, gnorm, ssm = _problem(n, N)
    got, sm = _sim(n, N, qk, v, gb, z, gnorm, ssm, register_tri=register_tri)
    ref, sref = gdn_scan_prefill_reference(qk, v, gb, z, gnorm, ssm, n, eps=EPS)
    rel = np.abs(got[:n].astype(np.float32) - ref[:n]).max() / np.abs(ref[:n]).max()
    assert rel < 5e-2, f"output rel {rel}"
    srel = np.abs(sm - sref).max() / np.abs(sref).max()
    assert srel < 5e-2, f"carried state rel {srel}"


@needs_cxx
@pytest.mark.parametrize("register_tri", [False, True])
def test_a_partial_last_chunk_is_masked(register_tri):
    """n = 100 leaves 28 padding rows in the second chunk. They must not reach gc, the products or the state --
    the whole-chunk answer for the first 100 tokens has to be identical to the n = 128 run's first 100."""
    N = 128
    qk, v, gb, z, gnorm, ssm = _problem(128, N)
    full, _ = _sim(128, N, qk, v, gb, z, gnorm, ssm, register_tri=register_tri)
    part, sm = _sim(100, N, qk, v, gb, z, gnorm, ssm, register_tri=register_tri)
    ref, sref = gdn_scan_prefill_reference(qk, v, gb, z, gnorm, ssm, 100, eps=EPS)
    assert np.abs(part[:100].astype(np.float32) - ref[:100]).max() / np.abs(ref[:100]).max() < 5e-2
    assert np.abs(sm - sref).max() / np.abs(sref).max() < 5e-2
    # and the padding really is padding: the first chunk is untouched by the second chunk's row count
    assert np.array_equal(full[:C], part[:C])


@needs_cxx
@pytest.mark.parametrize("register_tri", [False, True])
def test_the_chunk_size_does_not_change_the_answer(register_tri):
    """C is a blocking choice, not a model choice: C = 32 and C = 64 must agree. If D, the triangular solve or
    the carry were wrong, they would disagree -- a single-chunk test cannot see any of it."""
    n = N = 128
    qk, v, gb, z, gnorm, ssm = _problem(n, N)
    a, sa = _sim(n, N, qk, v, gb, z, gnorm, ssm, register_tri=register_tri)
    low = lowering.lower(kernels.make_gdn_scan_prefill(N=N, KH=KH, VH=VH, DK=DK, DV=DV, C=32,
                                                       tiles=TILES, shaves=SHAVES, eps=EPS))
    lib = host_sim.build(low.source, "gdn_scan_prefill_c32")
    out = np.zeros((N, VH * DV), np.float16)
    sm = ssm.copy()
    status = host_sim.run(lib, lowering.order_arrays(low, {
        "QK": qk.reshape(KH, qk.shape[1], 2 * DK), "V": v, "Gb": gb, "Z": z, "GNorm": gnorm,
        "Nt": np.array([[n]], np.int32), "Ssm": sm, "Out": out}), tiles=TILES, shaves=SHAVES)
    host_sim.check_status(status)
    rel = np.abs(a.astype(np.float32) - out.astype(np.float32)).max() / np.abs(a.astype(np.float32)).max()
    assert rel < 5e-2, f"C = 32 and C = 64 disagree by {rel}"
    assert np.abs(sm - sa).max() / np.abs(sa).max() < 5e-2


@needs_cxx
@pytest.mark.parametrize("register_tri", [False, True])
def test_a_perturbed_token_moves_only_later_outputs(register_tri):
    """The negative control, and the one that catches a scan running the wrong way: changing token 40's v may
    move tokens 40.. and must leave 0..39 bit-identical. A non-causal or reversed scan fails this."""
    n = N = 128
    qk, v, gb, z, gnorm, ssm = _problem(n, N)
    base, _ = _sim(n, N, qk, v, gb, z, gnorm, ssm, register_tri=register_tri)
    v2 = v.copy()
    v2[3, 40] = (v2[3, 40].astype(np.float32) + 1.0).astype(np.float16)
    moved, _ = _sim(n, N, qk, v2, gb, z, gnorm, ssm, register_tri=register_tri)
    head = slice(3 * DV, 4 * DV)
    assert np.array_equal(base[:40, head], moved[:40, head]), "an earlier token moved: the scan is not causal"
    assert not np.array_equal(base[40:, head], moved[40:, head]), "no later token moved: the state is not carried"


@needs_cxx
@pytest.mark.parametrize("live", [0, 1, 64, 100])
def test_live_chunks_skip_full_padding_without_stale_outputs(live):
    capacity = 192
    inputs = _problem(live, capacity, seed=13)
    output, state = _sim(live, capacity, *inputs, register_tri=True, live_chunks=True, sentinel=True)
    assert np.isfinite(output).all() and np.isfinite(state).all()
    assert np.count_nonzero(output[live:]) == 0
    reference, reference_state = gdn_scan_prefill_reference(*inputs, live, eps=EPS)
    assert np.abs(state - reference_state).max() / max(float(np.abs(reference_state).max()), 1e-6) < 5e-2
    if live:
        assert (np.abs(output[:live].astype(np.float32) - reference[:live]).max()
                / max(float(np.abs(reference[:live]).max()), 1e-6)) < 5e-2


def test_an_unaligned_chunk_size_is_refused():
    with pytest.raises(AssertionError):
        make_gdn_scan_prefill(N=128, KH=KH, VH=VH, DK=DK, DV=DV, C=48, tiles=TILES, shaves=SHAVES)
