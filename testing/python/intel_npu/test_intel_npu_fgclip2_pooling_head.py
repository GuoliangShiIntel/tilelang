"""V2 (plan: docs/developer_guide/fgclip2_enablement_plan.md): the FG-CLIP2 attention-pooling head
(`make_fgclip2_pooling_head_ddr`) against a numpy oracle, synthetic weights first (fast iteration before the
real-checkpoint test)."""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from fgclip2_kernels import make_fgclip2_pooling_head_ddr, pooling_head_reference  # noqa: E402

ROWS, VALID, HIDDEN, HEADS, HD_PAD, INTER_PAD = 256, 200, 1152, 16, 80, 4320
BR, TN, BQ, C = 32, 32, 16, 128
QW = HEADS * HD_PAD


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


def _weights(rng):
    def w(n, k):
        return (rng.standard_normal((n, k)) * 0.02).astype(np.float16)

    def b(n):
        return (rng.standard_normal(n) * 0.01).astype(np.float16).reshape(1, n)

    return dict(
        post_ln_w=np.ones((1, HIDDEN), np.float16), post_ln_b=b(HIDDEN),
        map_q=np.tile((rng.standard_normal((1, QW)) * 0.05).astype(np.float16), (BQ, 1)),  # BQ identical rows
        map_wk=w(QW, HIDDEN), map_bk=b(QW), map_wv=w(QW, HIDDEN), map_bv=b(QW),
        map_wo=w(HIDDEN, QW), map_bo=b(HIDDEN),
        hln_w=np.ones((1, HIDDEN), np.float16), hln_b=b(HIDDEN),
        map_w1=w(INTER_PAD, HIDDEN), map_b1=b(INTER_PAD), map_w2=w(HIDDEN, INTER_PAD), map_b2=b(HIDDEN),
    )


def _kernel():
    return make_fgclip2_pooling_head_ddr(ROWS, VALID, HIDDEN, HEADS, HD_PAD, INTER_PAD, BR=BR, TN=TN, BQ=BQ, C=C)


def test_pooling_head_lowers():
    assert "kernel_entry" in _lower(_kernel()).source


@needs_cxx
def test_pooling_head_matches_numpy_reference():
    low = _lower(_kernel())
    lib = host_sim.build(low.source, "tl_fgclip2_pool_head")
    rng = np.random.default_rng(0)
    W = _weights(rng)
    x = (rng.standard_normal((ROWS, HIDDEN)) * 0.3).astype(np.float32)
    x[VALID:] = (rng.standard_normal((ROWS - VALID, HIDDEN)) * 5.0).astype(np.float32)
    hio = x.copy()
    arrays = {
        "Hio": hio, "PostLnW": W["post_ln_w"], "PostLnB": W["post_ln_b"], "MapQ": W["map_q"],
        "MapWk": W["map_wk"], "MapBk": W["map_bk"], "MapWv": W["map_wv"], "MapBv": W["map_bv"],
        "MapWo": W["map_wo"], "MapBo": W["map_bo"], "HLnW": W["hln_w"], "HLnB": W["hln_b"],
        "MapW1": W["map_w1"], "MapB1": W["map_b1"], "MapW2": W["map_w2"], "MapB2": W["map_b2"],
        "Xn": np.zeros((ROWS, HIDDEN), np.float16), "Ks": np.zeros((ROWS, QW), np.float16),
        "Vs": np.zeros((ROWS, QW), np.float16), "Ao": np.zeros((BQ, QW), np.float16),
        "ALn": np.zeros((BQ, HIDDEN), np.float16), "Act1": np.zeros((BQ, INTER_PAD), np.float16),
        "Emb": np.zeros((BQ, HIDDEN), np.float32),
    }
    status = host_sim.run(lib, lowering.order_arrays(low, arrays), 1, 2)
    host_sim.check_status(status)

    # the kernel's post_layernorm is applied inside the kernel too (from the raw residual `x`), so the oracle
    # must start from the same raw `x` and apply the identical LayerNorm (ones weight / small bias here) before
    # the pooling math -- not pass `x` directly as if it were already post-LN.
    def layer_norm(v, w, b, eps=1e-6):
        mu = v.mean(-1, keepdims=True)
        var = ((v - mu) ** 2).mean(-1, keepdims=True)
        return (v - mu) / np.sqrt(var + eps) * w + b

    x_post_ln = layer_norm(x.astype(np.float64), W["post_ln_w"].astype(np.float64).reshape(-1),
                           W["post_ln_b"].astype(np.float64).reshape(-1))
    ref = pooling_head_reference(x_post_ln, W["map_q"][0:1], W["map_wk"], W["map_bk"], W["map_wv"], W["map_bv"],
                                 W["map_wo"], W["map_bo"], W["hln_w"], W["hln_b"], W["map_w1"], W["map_b1"],
                                 W["map_w2"], W["map_b2"], HEADS, HD_PAD, VALID)
    out = arrays["Emb"][0].astype(np.float64)
    rel = np.abs(out - ref).max() / np.abs(ref).max()
    assert rel < 5e-2, rel
