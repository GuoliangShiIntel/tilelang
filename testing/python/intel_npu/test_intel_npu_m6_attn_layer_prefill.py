"""P1 step 2: a whole full-attention PREFILL mixer layer in one launch (`kernels.make_attn_layer_prefill`) --
norm -> qkv projection -> P1b (per-head norms, RoPE, the cache append) -> P1's flash attention -> o_proj ->
residual, over N prompt tokens.

The oracle is built from the phase oracles, so the tests go after what composition can break and a single phase
cannot: that the residual is added rather than replaced, that the cache this layer fills is the cache its own
flash phase reads (a layout disagreement between P1b and P1 is invisible to either alone), that a dead token is
not added to the stream -- the one place the attention layer must do what the GDN layer gets for free, since a
zero query gives a UNIFORM softmax, not a zero -- and that nothing allocates inside a loop (plan question 10).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import (attn_layer_prefill_reference, make_attn_layer_prefill, pack_int4,  # noqa: E402
                     rope_cos_sin)
from test_intel_npu_scoped_cmx import allocations_inside_loops  # noqa: E402

TILES, SHAVES = 2, 2
N, H, NH, NKV, D, ROT, SMAX, G = 32, 256, 4, 2, 32, 16, 128, 128
TB, TBF, BKV, BN1, BN2, MB = 8, 4, 32, 32, 32, 16
GQ = NH // NKV
KO = NH * D
NQKV = NH * 2 * D + 2 * NKV * D
NQKV_pad = -(-NQKV // BN1) * BN1
THETA = 1e6
REL = 6e-2   # int4 weights through two projections and an fp16 flash phase


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")


def _kernel():
    return make_attn_layer_prefill(N=N, H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, G=G, TB=TB, TBF=TBF,
                                   BKV=BKV, BN1=BN1, BN2=BN2, MB=MB, tiles=TILES, shaves=SHAVES)


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    h = (rng.standard_normal((N, H), np.float32) * 0.5).astype(np.float32)
    inn = (rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16)
    wq = rng.integers(-8, 8, (NQKV_pad, H), dtype=np.int8)
    wqs = (rng.standard_normal((NQKV_pad, H // G), np.float32) * 0.05).astype(np.float16)
    wo = rng.integers(-8, 8, (H, KO), dtype=np.int8)
    wos = (rng.standard_normal((H, KO // G), np.float32) * 0.05).astype(np.float16)
    qn = (0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16)
    kn = (0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16)
    return h, inn, wq, wqs, wo, wos, qn, kn


def _run(h, inn, wq, wqs, wo, wos, qn, kn, pos0=0, n=N, kc=None, vt=None):
    low = lowering.lower(_kernel())
    lib = host_sim.build(low.source, "attn_layer_prefill")
    cs = np.stack([rope_cos_sin(pos0 + t, ROT, THETA).reshape(-1) for t in range(N)]).astype(np.float32)
    kc = np.zeros((NKV, SMAX, D), np.float16) if kc is None else kc.copy()
    vt = np.zeros((NKV, D, SMAX), np.float16) if vt is None else vt.copy()
    hio = h.copy()
    arrays = {"Hio": hio, "InNorm": inn, "WqkvQ": pack_int4(wq), "WqkvS": wqs, "QNormW": qn, "KNormW": kn,
              "CosSin": cs, "Pos": np.array([[pos0]], np.int32), "Nt": np.array([[n]], np.int32),
              "Kc": kc, "Vt": vt, "WoQ": pack_int4(wo), "WoS": wos,
              "Xs": np.zeros((N, H), np.float16), "Ys": np.zeros((N, NQKV_pad), np.float16),
              "Q": np.zeros((NKV, N, GQ, D), np.float16), "Gate": np.zeros((N, KO), np.float16),
              "Os": np.zeros((N, KO), np.float16), "Mix": np.zeros((N, H), np.float16)}
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), tiles=TILES, shaves=SHAVES))
    return hio, kc, vt, cs


def test_the_layer_does_not_reintroduce_question_10():
    src = lowering.lower(_kernel()).source
    assert allocations_inside_loops(src) == []
    # five phases deal their work differently, so each boundary needs a barrier
    assert src.count("ctx.sync_grid") >= 5, f"grid barriers: {src.count('ctx.sync_grid')}"


def test_the_cache_is_a_kernel_parameter():
    """Plan question 8: a cache reached through a layer-table address and appended at a runtime position loses the
    device. The layer takes Kc / Vt as parameters, so the table form cannot creep back in unnoticed."""
    src = lowering.lower(_kernel()).source
    assert "ddr_view" not in src


@needs_cxx
@pytest.mark.parametrize("pos0", [0, 48])
def test_matches_the_composed_oracle(pos0):
    w = _weights()
    hio, kc, vt, cs = _run(*w, pos0=pos0)
    ref, kc_ref, vt_ref = attn_layer_prefill_reference(w[0], w[1], w[2], w[3], w[6], w[7], cs, w[4], w[5],
                                                       np.zeros((NKV, SMAX, D), np.float16),
                                                       np.zeros((NKV, D, SMAX), np.float16),
                                                       pos0, N, NH, NKV, D, ROT, G)
    assert np.abs(hio - ref).max() / np.abs(ref).max() < REL
    for name, got, exp in (("Kc", kc, kc_ref), ("Vt", vt, vt_ref)):
        rel = (np.abs(got.astype(np.float32) - exp.astype(np.float32)).max()
               / max(float(np.abs(exp.astype(np.float32)).max()), 1e-6))
        assert rel < REL, f"{name} rel {rel}"


@needs_cxx
def test_the_residual_is_added_not_replaced():
    """Zero the o_proj and the layer must hand its input straight back. A layer that wrote the mixer output over
    the stream passes the oracle test (the oracle would be wrong the same way) and fails this one."""
    h, inn, wq, wqs, wo, wos, qn, kn = _weights()
    hio, kc, _, _ = _run(h, inn, wq, wqs, np.zeros_like(wo), np.zeros_like(wos), qn, kn)
    assert np.abs(hio - h).max() == 0, "with a zero o_proj the layer must be the identity on the stream"
    assert np.abs(kc.astype(np.float32)).max() > 0, "positive control: the cache was still filled"


@needs_cxx
def test_a_short_prompt_leaves_the_dead_tokens_alone():
    n = 20
    w = _weights()
    hio, kc, vt, _ = _run(*w, n=n)
    assert np.abs(hio[n:] - w[0][n:]).max() == 0, "tokens past the live count were added to the stream"
    assert np.abs(hio[:n] - w[0][:n]).max() > 0
    assert np.abs(kc[:, n:].astype(np.float32)).max() == 0, "a dead token was written into the cache"


@needs_cxx
def test_the_flash_phase_reads_the_cache_this_layer_wrote():
    """The two phases only agree if P1b's append layout is P1's read layout. Perturbing one earlier position of
    the cache this layer inherits must move every token's output; leaving the cache ahead of the chunk alone must
    not (it is masked), which is what separates "reads the right place" from "reads something"."""
    w = _weights()
    pos0 = 48
    kc0 = np.zeros((NKV, SMAX, D), np.float16)
    vt0 = np.zeros((NKV, D, SMAX), np.float16)
    rng = np.random.default_rng(1)
    kc0[:, :pos0] = rng.standard_normal((NKV, pos0, D), np.float32).astype(np.float16)
    vt0[:, :, :pos0] = rng.standard_normal((NKV, D, pos0), np.float32).astype(np.float16)
    a, _, _, _ = _run(*w, pos0=pos0, kc=kc0, vt=vt0)
    kc1 = kc0.copy()
    kc1[0, 7] += np.float16(8.0)      # a position every token of the chunk attends to
    b, _, _, _ = _run(*w, pos0=pos0, kc=kc1, vt=vt0)
    assert np.abs(a - b).max() > 0, "the chunk's output did not depend on the inherited cache"
    kc2 = kc0.copy()
    kc2[0, pos0 + N] += np.float16(8.0)   # one position past the chunk: the causal mask must absorb it
    c, _, _, _ = _run(*w, pos0=pos0, kc=kc2, vt=vt0)
    assert np.abs(a - c).max() == 0, "a token saw a key past the chunk: the causal mask is not live in the layer"
