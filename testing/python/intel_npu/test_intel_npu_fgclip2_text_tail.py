"""V3 (plan: docs/developer_guide/fgclip2_enablement_plan.md): the text tower's tail
(`make_fgclip2_text_tail_ddr`: final LayerNorm + head projection), synthetic weights, against a numpy oracle.
The 27 encoder layers themselves reuse `make_fgclip2_encoder_layers_ddr` unchanged (same math, different
weights, `valid == rows == 64`, no masking -- see the real-data test for that combination)."""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from fgclip2_kernels import make_fgclip2_text_tail_ddr, text_tail_reference  # noqa: E402

ROWS, HIDDEN = 64, 1152
BR, TN = 64, 32


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except RuntimeError:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")


def _kernel():
    return make_fgclip2_text_tail_ddr(ROWS, HIDDEN, BR=BR, TN=TN)


def test_text_tail_lowers():
    assert "kernel_entry" in _lower(_kernel()).source


@needs_cxx
def test_text_tail_matches_numpy_reference():
    low = _lower(_kernel())
    lib = host_sim.build(low.source, "tl_fgclip2_text_tail")
    rng = np.random.default_rng(0)
    ln_w = np.ones((1, HIDDEN), np.float16)
    ln_b = (rng.standard_normal(HIDDEN) * 0.01).astype(np.float16).reshape(1, -1)
    head_w = (rng.standard_normal((HIDDEN, HIDDEN)) * 0.02).astype(np.float16)
    head_b = (rng.standard_normal(HIDDEN) * 0.01).astype(np.float16).reshape(1, -1)
    x = (rng.standard_normal((ROWS, HIDDEN)) * 0.3).astype(np.float32)
    hio = x.copy()
    arrays = {
        "Hio": hio, "FinalLnW": ln_w, "FinalLnB": ln_b, "HeadW": head_w, "HeadB": head_b,
        "XL": np.zeros((ROWS, HIDDEN), np.float16), "EmbAll": np.zeros((ROWS, HIDDEN), np.float16),
    }
    status = host_sim.run(lib, lowering.order_arrays(low, arrays), 1, 2)
    host_sim.check_status(status)

    def layer_norm(v, w, b, eps=1e-6):
        mu = v.mean(-1, keepdims=True)
        var = ((v - mu) ** 2).mean(-1, keepdims=True)
        return (v - mu) / np.sqrt(var + eps) * w + b

    last = ROWS - 1
    x_ln_last = layer_norm(x[last].astype(np.float64), ln_w.astype(np.float64).reshape(-1),
                           ln_b.astype(np.float64).reshape(-1))
    ref = text_tail_reference(x_ln_last, head_w, head_b)
    out = arrays["EmbAll"][last].astype(np.float64)
    rel = np.abs(out - ref).max() / np.abs(ref).max()
    assert rel < 5e-2, rel
