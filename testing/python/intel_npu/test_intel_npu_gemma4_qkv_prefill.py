"""Gemma4 V0: the QKV prefill phase (gemma4_kernels.make_gemma4_qkv_prefill) -- plain (non-zero-
centered) per-head RMSNorm, Gemma4's full-width rotate-half RoPE, and the always-unweighted-RMSNorm
V (from a separate v_proj for sliding layers, from the raw K projection for full-attention layers).
See docs/developer_guide/gemma4_enablement_plan.md section 2.1/2.2.
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
from gemma4_kernels import gemma4_qkv_prefill_reference, make_gemma4_qkv_prefill  # noqa: E402

N, NH, NKV, D, SMAX, TB, TILES, SHAVES = 8, 4, 2, 32, 64, 4, 3, 2
REL = 3e-2


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


def _problem(full: bool, pos0: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    NQKV = NH * D + NKV * D if full else NH * D + 2 * NKV * D
    y = rng.standard_normal((N, NQKV), np.float32).astype(np.float16)
    qnorm = rng.standard_normal(D, np.float32).astype(np.float16)
    knorm = rng.standard_normal(D, np.float32).astype(np.float16)
    half = D // 2
    angles = rng.uniform(0, 2 * np.pi, (N, half)).astype(np.float32)
    cos_h, sin_h = np.cos(angles), np.sin(angles)
    cos_sin = np.concatenate([np.tile(cos_h, 2), np.tile(sin_h, 2)], axis=1)  # [N, 2 * D]
    kc = np.zeros((NKV, SMAX, D), np.float16)
    vt = np.zeros((NKV, D, SMAX), np.float16)
    return y, qnorm, knorm, cos_sin, kc, vt


def _sim(full, y, qnorm, knorm, cos_sin, kc, vt, pos0, n):
    low = _lower(make_gemma4_qkv_prefill(N=N, NH=NH, NKV=NKV, D=D, SMAX=SMAX, full=full, TB=TB, tiles=TILES, shaves=SHAVES))
    lib = host_sim.build(low.source, "tl_gemma4_qkv_prefill_sim")
    GQ = NH // NKV
    Q = np.zeros((NKV, N, GQ, D), np.float16)
    arrays = dict(Ys=y, QNormW=qnorm.reshape(1, D), KNormW=knorm.reshape(1, D), CosSin=cos_sin,
                  Pos=np.array([[pos0]], np.int32), Nt=np.array([[n]], np.int32), Q=Q, Kc=kc, Vt=vt)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    return arrays["Q"], arrays["Kc"], arrays["Vt"]


@needs_cxx
@pytest.mark.parametrize("full", [False, True])
def test_matches_reference(full):
    pos0 = 16
    y, qnorm, knorm, cos_sin, kc, vt = _problem(full, pos0)
    q, kc_out, vt_out = _sim(full, y, qnorm, knorm, cos_sin, kc, vt, pos0, N)
    ref_q, ref_kc, ref_vt = gemma4_qkv_prefill_reference(y, qnorm, knorm, cos_sin, pos0, N, NH, NKV, D, full, kc, vt)
    rel_q = np.abs(q.astype(np.float32) - ref_q).max() / (np.abs(ref_q).max() + 1e-6)
    rel_k = np.abs(kc_out.astype(np.float32) - ref_kc).max() / (np.abs(ref_kc).max() + 1e-6)
    rel_v = np.abs(vt_out.astype(np.float32) - ref_vt).max() / (np.abs(ref_vt).max() + 1e-6)
    assert rel_q < REL, rel_q
    assert rel_k < REL, rel_k
    assert rel_v < REL, rel_v


@needs_cxx
def test_full_attention_v_equals_unweighted_norm_of_raw_k_not_weighted_k():
    """A dead-test guard: full=True's V must NOT equal the (weighted, roped) K it is derived alongside."""
    pos0 = 0
    y, qnorm, knorm, cos_sin, kc, vt = _problem(True, pos0)
    _, kc_out, vt_out = _sim(True, y, qnorm, knorm, cos_sin, kc, vt, pos0, N)
    assert np.abs(kc_out.astype(np.float32) - vt_out.transpose(0, 2, 1).astype(np.float32)[:, : kc_out.shape[1]]).max() > 0.1
