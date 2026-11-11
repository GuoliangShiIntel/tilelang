"""V1, continued (plan: docs/developer_guide/fgclip2_enablement_plan.md): a device-side loop over N FG-CLIP2
encoder layers through a weight table (`T.ddr_view`), the other half of V1 that `test_intel_npu_fgclip2_v1.py`
(one layer, real channel widths) did not cover yet. Reuses Qwen's layer-table pattern (`kernels.py`'s
`TABLE_COLS`/`T.ddr_view`/`make_decode_model`) end to end, including how a host-sim test fills the table: one
contiguous numpy array per (layer, tensor), kept alive in a dict, its `.ctypes.data` pointer written into the
table cell -- exactly what a real device driver does with `Session.address`.
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
from fgclip2_kernels import FGCLIP2_TABLE_COLS, fgclip2_layers_reference, make_fgclip2_encoder_layers_v1  # noqa: E402

N_LAYERS, ROWS, HIDDEN, HEADS, HD_PAD, INTER_PAD, TN = 3, 16, 1152, 16, 80, 4320, 32
QW = HEADS * HD_PAD
COL = {c: i for i, c in enumerate(FGCLIP2_TABLE_COLS)}


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


def _layer_weights(rng, seed_scale):
    """One layer's weights, a different (tiny) scale per layer so a kernel that read layer 0's table row for
    every layer would disagree with the oracle (the negative control `test_every_layer_reads_its_own_row`
    mirrors `test_intel_npu_m6_prefill_model.py`'s own version of this check)."""
    def w(n, k):
        return (rng.standard_normal((n, k)) * seed_scale).astype(np.float16)

    def b(n):
        return (rng.standard_normal(n) * seed_scale * 0.5).astype(np.float16).reshape(1, n)

    return dict(
        ln1w=np.ones((1, HIDDEN), np.float16), ln1b=b(HIDDEN),
        wq=w(QW, HIDDEN), bq=b(QW), wk=w(QW, HIDDEN), bk=b(QW), wv=w(QW, HIDDEN), bv=b(QW),
        wo=w(HIDDEN, QW), bo=b(HIDDEN),
        ln2w=np.ones((1, HIDDEN), np.float16), ln2b=b(HIDDEN),
        w1=w(INTER_PAD, HIDDEN), b1=b(INTER_PAD), w2=w(HIDDEN, INTER_PAD), b2=b(HIDDEN),
    )


def _host_table(layers):
    """One `Table[N_LAYERS, NCOL]` of device addresses (`arr.ctypes.data`) plus a dict keeping every backing
    array alive (a garbage-collected array's address would go stale in the table)."""
    table = np.zeros((N_LAYERS, len(FGCLIP2_TABLE_COLS)), np.int64)
    keep = {}
    for li, p in enumerate(layers):
        for name, key in (("Ln1W", "ln1w"), ("Ln1B", "ln1b"), ("Wq", "wq"), ("Bq", "bq"), ("Wk", "wk"), ("Bk", "bk"),
                          ("Wv", "wv"), ("Bv", "bv"), ("Wo", "wo"), ("Bo", "bo"), ("Ln2W", "ln2w"), ("Ln2B", "ln2b"),
                          ("W1", "w1"), ("B1", "b1"), ("W2", "w2"), ("B2", "b2")):
            arr = np.ascontiguousarray(p[key])
            keep[f"L{li}_{name}"] = arr
            table[li, COL[name]] = arr.ctypes.data
    return table, keep


def test_encoder_layers_v1_lowers_with_a_device_side_loop_over_the_table():
    src = _lower(make_fgclip2_encoder_layers_v1(N_LAYERS, ROWS, HIDDEN, HEADS, HD_PAD, INTER_PAD, tn=TN)).source
    assert "kernel_entry" in src
    assert src.count("uint64_t)Table") >= 16   # every weight of the 16-column row bound inside the layer loop


@needs_cxx
def test_encoder_layers_v1_matches_numpy_reference_in_host_sim():
    low = _lower(make_fgclip2_encoder_layers_v1(N_LAYERS, ROWS, HIDDEN, HEADS, HD_PAD, INTER_PAD, tn=TN))
    lib = host_sim.build(low.source, "tl_fgclip2_encoder_layers_v1")
    rng = np.random.default_rng(0)
    layers = [_layer_weights(rng, 0.02 * (li + 1)) for li in range(N_LAYERS)]
    table, _keep = _host_table(layers)
    x = (rng.standard_normal((ROWS, HIDDEN)) * 0.3).astype(np.float32)
    hio = x.copy()
    status = host_sim.run(lib, lowering.order_arrays(low, {"Hio": hio, "Table": table}), 1, 2)
    host_sim.check_status(status)

    ref = fgclip2_layers_reference(x, layers, HEADS, HD_PAD)
    rel = np.abs(hio.astype(np.float64) - ref).max() / np.abs(ref).max()
    assert rel < 5e-2, rel


@needs_cxx
def test_every_layer_reads_its_own_row_of_the_table():
    """The easiest way to get a layer-table kernel wrong: hoist a weight view out of the loop so every layer
    reads layer 0's row. That bug would still pass a single-layer or all-identical-layers test, so this test
    gives each layer a DIFFERENT seed and checks the device result tracks a change to layer 1 alone (not
    layer 0) -- `test_intel_npu_m6_prefill_model.py`'s own version of the same check."""
    low = _lower(make_fgclip2_encoder_layers_v1(N_LAYERS, ROWS, HIDDEN, HEADS, HD_PAD, INTER_PAD, tn=TN))
    lib = host_sim.build(low.source, "tl_fgclip2_encoder_layers_v1")
    rng = np.random.default_rng(1)
    layers = [_layer_weights(rng, 0.02 * (li + 1)) for li in range(N_LAYERS)]
    x = (rng.standard_normal((ROWS, HIDDEN)) * 0.3).astype(np.float32)

    table_a, _keep_a = _host_table(layers)
    hio_a = x.copy()
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, {"Hio": hio_a, "Table": table_a}), 1, 2))

    layers_b = [dict(p) for p in layers]
    layers_b[1] = dict(layers_b[1], wq=layers_b[1]["wq"] + np.float16(0.05))   # perturb layer 1 ONLY
    table_b, _keep_b = _host_table(layers_b)
    hio_b = x.copy()
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, {"Hio": hio_b, "Table": table_b}), 1, 2))

    assert np.abs(hio_b - hio_a).max() > 1e-2, "perturbing layer 1's table row did not change the result"
