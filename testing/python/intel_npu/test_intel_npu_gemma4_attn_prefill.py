"""Gemma4 V0: the gate-less, optionally-sliding-window flash-attention prefill phase
(gemma4_kernels.make_gemma4_attn_prefill), the one piece of the Gemma4 mixer with no existing
Qwen3.6 counterpart at all (kernels.make_attn_prefill always applies an output gate Gemma4 has none
of, and has no window lower bound). See docs/developer_guide/gemma4_enablement_plan.md section 2.1.

Hardware-free coverage here: the host simulation against `gemma4_attn_prefill_reference`, with the
negatives that make both masks falsifiable:

  * a perturbed key FAILs the golden;
  * window=None reduces to plain causal attention (cross-checked against a later pos0's oracle
    disagreeing, same negative control as kernels.py's own attn_prefill test);
  * a small window actually drops old keys: the windowed oracle disagrees with the unwindowed one at
    the same (q, k, v), so a passing comparison against the WINDOWED oracle is not an accident of the
    causal mask alone catching everything.
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
from gemma4_kernels import gemma4_attn_prefill_reference, make_gemma4_attn_prefill  # noqa: E402

# small instance: the shapes only have to satisfy the kernel's asserts
N, NKV, GQ, D, SMAX, TB, BKV, TILES, SHAVES = 16, 2, 2, 32, 256, 4, 32, 3, 2
WINDOW = 20  # smaller than N + any pos0 used below, so the window actually trims live keys
REL = 3e-2   # fp16 scores and an fp16 PV product over a short context


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


def _problem(pos0: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((NKV, N, GQ, D), np.float32).astype(np.float16)
    kc = np.zeros((NKV, SMAX, D), np.float16)
    vt = np.zeros((NKV, D, SMAX), np.float16)
    last = pos0 + N
    kc[:, :last] = rng.standard_normal((NKV, last, D), np.float32).astype(np.float16)
    vt[:, :, :last] = rng.standard_normal((NKV, D, last), np.float32).astype(np.float16)
    return q, kc, vt


def _sim(q, kc, vt, pos0: int, window):
    low = _lower(make_gemma4_attn_prefill(N=N, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, window=window, TB=TB, BKV=BKV, tiles=TILES, shaves=SHAVES))
    lib = host_sim.build(low.source, "tl_gemma4_attn_prefill_sim")
    out = np.zeros((N, NKV * GQ * D), np.float16)
    arrays = dict(Q=q, Kc=kc, Vt=vt, Pos=np.array([[pos0]], np.int32), O=out)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    return out


def test_lowers_with_no_gate_and_runtime_chunk_bound():
    src = _lower(make_gemma4_attn_prefill(N=N, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, TB=TB, BKV=BKV, tiles=TILES)).source
    assert "kernel_entry" in src
    assert "Gate" not in src  # no output gate parameter at all, unlike kernels.make_attn_prefill
    assert "pos_cmx" in src or "Pos" in src


@needs_cxx
def test_causal_only_matches_reference_at_pos0_zero():
    pos0 = 0
    q, kc, vt = _problem(pos0)
    out = _sim(q, kc, vt, pos0, window=None)
    ref = gemma4_attn_prefill_reference(q, kc, vt, pos0, window=None)
    rel = np.abs(out.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel < REL, rel


@needs_cxx
def test_causal_only_matches_reference_with_cache_prefix():
    pos0 = 48
    q, kc, vt = _problem(pos0)
    out = _sim(q, kc, vt, pos0, window=None)
    ref = gemma4_attn_prefill_reference(q, kc, vt, pos0, window=None)
    rel = np.abs(out.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel < REL, rel


@needs_cxx
def test_window_matches_windowed_reference_not_plain_causal():
    pos0 = 48
    q, kc, vt = _problem(pos0)
    out = _sim(q, kc, vt, pos0, window=WINDOW)
    ref_windowed = gemma4_attn_prefill_reference(q, kc, vt, pos0, window=WINDOW)
    ref_causal = gemma4_attn_prefill_reference(q, kc, vt, pos0, window=None)
    # the window must actually change the answer at this pos0/window (a dead test otherwise)
    assert np.abs(ref_windowed - ref_causal).max() > 1e-3
    rel = np.abs(out.astype(np.float32) - ref_windowed).max() / (np.abs(ref_windowed).max() + 1e-6)
    assert rel < REL, rel
    # falsifiable: comparing the windowed kernel output against the UNWINDOWED oracle must fail
    rel_vs_causal = np.abs(out.astype(np.float32) - ref_causal).max() / (np.abs(ref_causal).max() + 1e-6)
    assert rel_vs_causal > REL


@needs_cxx
def test_perturbed_key_fails_golden():
    pos0 = 48
    q, kc, vt = _problem(pos0)
    out = _sim(q, kc, vt, pos0, window=WINDOW)
    kc2 = kc.copy()
    kc2[0, pos0] += np.float16(5.0)
    ref_bad = gemma4_attn_prefill_reference(q, kc2, vt, pos0, window=WINDOW)
    rel = np.abs(out.astype(np.float32) - ref_bad).max() / (np.abs(ref_bad).max() + 1e-6)
    assert rel > REL
