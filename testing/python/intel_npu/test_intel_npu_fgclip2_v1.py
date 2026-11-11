"""V1 (plan: docs/developer_guide/fgclip2_enablement_plan.md): one FG-CLIP2 encoder layer at REAL model
proportions (hidden=1152, heads=16, hd_pad=80, inter_pad=4320), host-sim only, against the same numpy oracle
V0 uses (`encoder_layer_reference` is already generic in Q/K/V width vs hidden -- no change needed there).

V0 -> V1's one structural change under test here: every linear layer streams its weight in row tiles from DDR
(`linear_bias_tiled`) instead of loading the whole matrix resident -- real hidden=1152 / inter_pad=4320 weight
matrices no longer fit CMX whole. `rows` stays small (16): V1 is about real CHANNEL widths, not real token
counts -- attention is still the same plain (non-chunked) softmax as V0.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from fgclip2_kernels import encoder_layer_reference, make_fgclip2_encoder_layer_v1  # noqa: E402

ROWS, HIDDEN, HEADS, HD_PAD, INTER_PAD, TN = 16, 1152, 16, 80, 4320, 32
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
        ln1w=np.ones((1, HIDDEN), np.float16), ln1b=b(HIDDEN),
        wq=w(QW, HIDDEN), bq=b(QW), wk=w(QW, HIDDEN), bk=b(QW), wv=w(QW, HIDDEN), bv=b(QW),
        wo=w(HIDDEN, QW), bo=b(HIDDEN),
        ln2w=np.ones((1, HIDDEN), np.float16), ln2b=b(HIDDEN),
        w1=w(INTER_PAD, HIDDEN), b1=b(INTER_PAD), w2=w(HIDDEN, INTER_PAD), b2=b(HIDDEN),
    )


def test_encoder_layer_v1_lowers_at_real_model_proportions():
    # Phase 0/1-style smoke check: real channel widths (1152 / 1280 / 4320) and the tiled weight stream lower
    # with no `C-*` source refusal before ever touching the host-sim build.
    src = _lower(make_fgclip2_encoder_layer_v1(ROWS, HIDDEN, HEADS, HD_PAD, INTER_PAD, tn=TN)).source
    assert "kernel_entry" in src


@needs_cxx
def test_encoder_layer_v1_matches_numpy_reference_in_host_sim():
    low = _lower(make_fgclip2_encoder_layer_v1(ROWS, HIDDEN, HEADS, HD_PAD, INTER_PAD, tn=TN))
    lib = host_sim.build(low.source, "tl_fgclip2_encoder_v1")
    rng = np.random.default_rng(0)
    W = _weights(rng)
    x = (rng.standard_normal((ROWS, HIDDEN)) * 0.3).astype(np.float32)
    hio = x.copy()
    arrays = {
        "Hio": hio, "Ln1W": W["ln1w"], "Ln1B": W["ln1b"],
        "Wq": W["wq"], "Bq": W["bq"], "Wk": W["wk"], "Bk": W["bk"], "Wv": W["wv"], "Bv": W["bv"],
        "Wo": W["wo"], "Bo": W["bo"], "Ln2W": W["ln2w"], "Ln2B": W["ln2b"],
        "W1": W["w1"], "B1": W["b1"], "W2": W["w2"], "B2": W["b2"],
    }
    status = host_sim.run(lib, lowering.order_arrays(low, arrays), 1, 2)
    host_sim.check_status(status)

    ref = encoder_layer_reference(x, W["ln1w"], W["ln1b"], W["wq"], W["bq"], W["wk"], W["bk"], W["wv"], W["bv"],
                                  W["wo"], W["bo"], W["ln2w"], W["ln2b"], W["w1"], W["b1"], W["w2"], W["b2"],
                                  HEADS, HD_PAD)
    rel = np.abs(hio.astype(np.float64) - ref).max() / np.abs(ref).max()
    assert rel < 5e-2, rel

    # negative control
    x2 = x.copy()
    x2[2, 5] += np.float32(1.0)
    hio2 = x2.copy()
    arrays2 = dict(arrays, Hio=hio2)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays2), 1, 2))
    assert np.abs(hio2 - hio).max() > 1e-2
