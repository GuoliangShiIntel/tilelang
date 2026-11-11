"""V0 (plan: docs/developer_guide/fgclip2_enablement_plan.md): one FG-CLIP2 pre-LN encoder layer, host-sim only,
against a numpy transcription of `fgclip2_e2e/reference.py::encoder_layer` (the model's own oracle in the
original npu_dsl implementation; see docs/developer_guide/fgclip2_npu_dsl_explained.md for the full walkthrough).

Small, clean test shapes on purpose (hidden=64, 4 heads of 16, inter=128 -- all already DPU-alignment-friendly,
no head-dim/MLP padding needed): V0 is a correctness check of the building blocks (LayerNorm with bias,
GELU-tanh, bias-add GEMM, full unmasked multi-head attention), not a test at the real model's shapes
(hidden=1152, 16 heads of 72->80, inter=4304->4320, 27 layers) -- that is V1.

ROWS=16 (not 8): the DPU requires every GEMM's N and K to be a multiple of 16 (`src/intel_npu/transform/
lower_tile_op.cc`'s `LowerIntelNpuTileOp` refuses otherwise), and the attention GEMMs use `rows` as their N
(Q.K^T) or K (P.V) dimension -- found the hard way: 8 lowers with `ValueError: gemm N=8, K=16 must be
multiples of 16`.
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
from fgclip2_kernels import encoder_layer_reference, make_fgclip2_encoder_layer_v0  # noqa: E402

ROWS, HIDDEN, HEADS, HEAD_DIM, INTER = 16, 64, 4, 16, 128


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
    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float16)

    def b(n):
        return (rng.standard_normal(n) * 0.02).astype(np.float16)

    return dict(
        ln1w=np.ones((1, HIDDEN), np.float16), ln1b=b(HIDDEN).reshape(1, HIDDEN),
        wq=w(HIDDEN, HIDDEN), bq=b(HIDDEN).reshape(1, HIDDEN),
        wk=w(HIDDEN, HIDDEN), bk=b(HIDDEN).reshape(1, HIDDEN),
        wv=w(HIDDEN, HIDDEN), bv=b(HIDDEN).reshape(1, HIDDEN),
        wo=w(HIDDEN, HIDDEN), bo=b(HIDDEN).reshape(1, HIDDEN),
        ln2w=np.ones((1, HIDDEN), np.float16), ln2b=b(HIDDEN).reshape(1, HIDDEN),
        w1=w(INTER, HIDDEN), b1=b(INTER).reshape(1, INTER),
        w2=w(HIDDEN, INTER), b2=b(HIDDEN).reshape(1, HIDDEN),
    )


def test_encoder_layer_v0_lowers_with_no_distributed_loop_refusal():
    # Phase 0/1-style smoke check (plan section 1 acceptance): it lowers at all, and the tile op vocabulary used
    # (LayerNorm's reduction loops, the bias-add epilogue, the per-head attention) is accepted by this backend's
    # pipeline with no `C-*` source refusal before ever touching the host-sim build.
    src = _lower(make_fgclip2_encoder_layer_v0(ROWS, HIDDEN, HEADS, HEAD_DIM, INTER)).source
    assert "kernel_entry" in src or "FRT_KERNEL_ENTRY" in src or "void" in src  # smoke: some entry point emitted


@needs_cxx
def test_encoder_layer_v0_matches_numpy_reference_in_host_sim():
    low = _lower(make_fgclip2_encoder_layer_v0(ROWS, HIDDEN, HEADS, HEAD_DIM, INTER))
    lib = host_sim.build(low.source, "tl_fgclip2_encoder_v0")
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
                                  HEADS, HEAD_DIM)
    rel = np.abs(hio.astype(np.float64) - ref).max() / np.abs(ref).max()
    assert rel < 5e-2, rel

    # negative control: perturbing one input row must change the output by more than the pass tolerance
    x2 = x.copy()
    x2[2, 5] += np.float32(1.0)
    hio2 = x2.copy()
    arrays2 = dict(arrays, Hio=hio2)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays2), 1, 2))
    assert np.abs(hio2 - hio).max() > 1e-2
