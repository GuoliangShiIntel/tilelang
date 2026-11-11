"""V1 completion (plan: docs/developer_guide/fgclip2_enablement_plan.md): one FG-CLIP2 encoder layer at a
REAL (padded) patch count, DDR-resident residual stream + row-block + weight-tile + key-chunk tiling
(`make_fgclip2_encoder_layer_ddr`). Synthetic weights/inputs first (fast iteration); the real-checkpoint,
real-golden-trace validation is a separate test (`test_intel_npu_fgclip2_real_layer0.py`) once this plumbing
is proven, so a real-data test failure is a real-data bug, not a tiling bug.

`rows=256, valid=200`: small enough to compile and host-sim quickly, large enough to force the row-block
(BR=64, 4 blocks), weight-tile (TN=32) and multi-chunk (`C=128`, 2 chunks, the second mostly masked) paths all
to actually iterate more than once -- a `rows == BR == C` kernel would not exercise its own loop-carried state.
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
from fgclip2_kernels import encoder_layer_reference, make_fgclip2_encoder_layer_ddr  # noqa: E402

ROWS, VALID, HIDDEN, HEADS, HD_PAD, INTER_PAD = 256, 200, 1152, 16, 80, 4320
BR, TN, BQ, C = 32, 32, 64, 128
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


def _kernel():
    return make_fgclip2_encoder_layer_ddr(ROWS, VALID, HIDDEN, HEADS, HD_PAD, INTER_PAD, BR=BR, TN=TN, BQ=BQ, C=C)


def test_encoder_layer_ddr_lowers_at_a_real_patch_count():
    src = _lower(_kernel()).source
    assert "kernel_entry" in src


@needs_cxx
def test_encoder_layer_ddr_matches_numpy_reference_on_the_valid_rows():
    low = _lower(_kernel())
    lib = host_sim.build(low.source, "tl_fgclip2_encoder_ddr")
    rng = np.random.default_rng(0)
    W = _weights(rng)
    x = (rng.standard_normal((ROWS, HIDDEN)) * 0.3).astype(np.float32)
    x[VALID:] = (rng.standard_normal((ROWS - VALID, HIDDEN)) * 5.0).astype(np.float32)  # garbage padding rows
    hio = x.copy()
    arrays = {
        "Hio": hio, "Ln1W": W["ln1w"], "Ln1B": W["ln1b"],
        "Wq": W["wq"], "Bq": W["bq"], "Wk": W["wk"], "Bk": W["bk"], "Wv": W["wv"], "Bv": W["bv"],
        "Wo": W["wo"], "Bo": W["bo"], "Ln2W": W["ln2w"], "Ln2B": W["ln2b"],
        "W1": W["w1"], "B1": W["b1"], "W2": W["w2"], "B2": W["b2"],
        "Xn": np.zeros((ROWS, HIDDEN), np.float16), "Qp": np.zeros((ROWS, QW), np.float16),
        "Kp": np.zeros((ROWS, QW), np.float16), "Vp": np.zeros((ROWS, QW), np.float16),
        "Ao": np.zeros((ROWS, QW), np.float16), "Act": np.zeros((ROWS, INTER_PAD), np.float16),
    }
    status = host_sim.run(lib, lowering.order_arrays(low, arrays), 1, 2)
    host_sim.check_status(status)

    # Oracle: the SAME masked-attention math, computed densely in numpy (every one of the VALID rows attends
    # only the first VALID keys; padding rows are not compared -- the kernel does not promise anything about
    # their output, only that they do not corrupt the valid rows' softmax, which this oracle call enforces by
    # construction: it is given only the valid keys).
    def masked_encoder_layer(x_full):
        out = x_full.copy()
        v = x_full[:VALID]
        ref_v = encoder_layer_reference(v, W["ln1w"], W["ln1b"], W["wq"], W["bq"], W["wk"], W["bk"], W["wv"], W["bv"],
                                        W["wo"], W["bo"], W["ln2w"], W["ln2b"], W["w1"], W["b1"], W["w2"], W["b2"],
                                        HEADS, HD_PAD)
        out[:VALID] = ref_v
        return out

    ref = masked_encoder_layer(x)
    rel = np.abs(hio[:VALID].astype(np.float64) - ref[:VALID]).max() / np.abs(ref[:VALID]).max()
    assert rel < 5e-2, rel
