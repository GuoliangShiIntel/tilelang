"""P1b of prefill: the attention layer's qkv phase -- what `attn_block`'s stage B does for one token, over N.

`kernels.make_attn_qkv_prefill` turns the qkv projection's output into the flash phase's query and gate
layout and APPENDS the keys and values to the cache. Nothing else writes `Kc` / `Vt`: `make_attn_prefill`
only reads them, so without this phase a prefilled prompt leaves decode with an empty cache.

Coverage here is hardware-free: the lowering, and the host simulation against
`attn_qkv_prefill_reference` with the negatives that make each half falsifiable:

  * a perturbed projection row FAILs the golden (the comparison bites);
  * **the RoPE is live**: the same input at a different pos0 must give a different key, so a kernel that
    forgot to rotate cannot pass;
  * **the norm is live**: zeroing the norm weight changes the output;
  * rows past `Nt` are written as ZERO, which is what `make_attn_prefill` relies on instead of masking.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401  (loads the dev build before tvm)
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import attn_qkv_prefill_reference, make_attn_qkv_prefill, rope_cos_sin  # noqa: E402

# small instance: the shapes only have to satisfy the kernel's asserts
N, H, NH, NKV, D, ROT, SMAX, TB, TILES, SHAVES = 16, 128, 4, 2, 64, 32, 128, 4, 3, 2
GQ = NH // NKV
NQKV = NH * 2 * D + 2 * NKV * D
REL = 2e-2   # fp16 stores of an fp32 norm-and-rotate


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


def _kernel():
    return make_attn_qkv_prefill(N=N, H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, TB=TB, tiles=TILES,
                                 shaves=SHAVES)


def _problem(pos0: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((N, NQKV), np.float32).astype(np.float16)
    qw = (0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16)
    kw = (0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16)
    cs = np.stack([rope_cos_sin(pos0 + t, ROT, 1e6).reshape(-1) for t in range(N)]).astype(np.float32)
    kc = np.zeros((NKV, SMAX, D), np.float16)
    vt = np.zeros((NKV, D, SMAX), np.float16)
    return y, qw, kw, cs, kc, vt


def _sim(y, qw, kw, cs, kc, vt, pos0: int, nt: int):
    low = _lower(_kernel())
    lib = host_sim.build(low.source, "tl_attn_qkv_prefill_sim")
    q = np.zeros((NKV, N, GQ, D), np.float16)
    gate = np.zeros((NKV, N, GQ, D), np.float16)
    kc, vt = kc.copy(), vt.copy()
    arrays = dict(Ys=y, QNormW=qw, KNormW=kw, CosSin=cs, Pos=np.array([[pos0]], np.int32),
                  Nt=np.array([[nt]], np.int32), Q=q, Gate=gate, Kc=kc, Vt=vt)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    return q, gate, kc, vt


def _rel(a, b):
    b = b.astype(np.float32)
    scale = max(np.abs(b).max(), 1e-6)
    return float(np.abs(a.astype(np.float32) - b).max() / scale)


def test_lowers_and_writes_both_caches():
    src = _lower(_kernel()).source
    assert "kernel_entry" in src
    # the keys of a token block are one contiguous run of the cache; V is transposed into one block store
    assert src.count("ctx.dma_store") >= 4, "Q, Gate, Kc and Vt must each be written"
    assert "tl_npu_vsum" in src          # the RMSNorm reduction
    assert "ring_begin" not in src       # nothing is streamed here


@pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")
@pytest.mark.parametrize("pos0", [0, 37])
def test_host_sim_matches_reference(pos0):
    y, qw, kw, cs, kc, vt = _problem(pos0)
    rq, rg, rkc, rvt = attn_qkv_prefill_reference(y, qw, kw, cs, pos0, N, NH, NKV, D, ROT, kc, vt)
    q, gate, okc, ovt = _sim(y, qw, kw, cs, kc, vt, pos0, N)
    assert _rel(q, rq) < REL, f"Q pos0={pos0} rel {_rel(q, rq):.3e}"
    assert _rel(gate, rg) < REL, f"Gate rel {_rel(gate, rg):.3e}"
    assert _rel(okc, rkc) < REL, f"Kc rel {_rel(okc, rkc):.3e}"
    assert _rel(ovt, rvt) < REL, f"Vt rel {_rel(ovt, rvt):.3e}"


@pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")
def test_a_perturbed_projection_row_fails():
    y, qw, kw, cs, kc, vt = _problem(0)
    rq, _, rkc, _ = attn_qkv_prefill_reference(y, qw, kw, cs, 0, N, NH, NKV, D, ROT, kc, vt)
    y[3] = y[3] + np.float16(4.0)
    q, _, okc, _ = _sim(y, qw, kw, cs, kc, vt, 0, N)
    assert _rel(q, rq) > REL and _rel(okc, rkc) > REL, "the golden passed with a perturbed row"


@pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")
def test_the_rotation_is_live():
    """The same projection at a different pos0 must give a different key: a kernel that skipped the RoPE,
    or read one token's angles for all of them, would be invariant here."""
    y, qw, kw, cs0, kc, vt = _problem(0)
    cs1 = np.stack([rope_cos_sin(37 + t, ROT, 1e6).reshape(-1) for t in range(N)]).astype(np.float32)
    _, _, kc_a, _ = _sim(y, qw, kw, cs0, kc, vt, 0, N)
    _, _, kc_b, _ = _sim(y, qw, kw, cs1, kc, vt, 0, N)
    assert _rel(kc_a[:, :N], kc_b[:, :N]) > REL, "the key did not rotate with the position"


@pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")
def test_the_norm_weight_is_live():
    y, qw, kw, cs, kc, vt = _problem(0)
    q_a, _, _, _ = _sim(y, qw, kw, cs, kc, vt, 0, N)
    q_b, _, _, _ = _sim(y, np.zeros_like(qw), kw, cs, kc, vt, 0, N)
    assert _rel(q_a, q_b) > REL, "the query norm weight did nothing"


@pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")
def test_rows_past_the_live_count_are_zero():
    """`make_attn_prefill` does not mask unseen positions -- it relies on them holding zeros."""
    live = N // 2
    y, qw, kw, cs, kc, vt = _problem(0)
    q, gate, okc, ovt = _sim(y, qw, kw, cs, kc, vt, 0, live)
    assert np.abs(okc[:, live:N]).max() == 0, "a dead token was appended to the key cache"
    assert np.abs(ovt[:, :, live:N]).max() == 0, "a dead token was appended to the value cache"
    assert np.abs(q[:, live:N]).max() == 0 and np.abs(gate[:, live:N]).max() == 0
    assert np.abs(okc[:, :live]).max() > 0, "control: the live rows were written"
