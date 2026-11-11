"""Real-checkpoint validation, all 27 layers (plan: docs/developer_guide/fgclip2_enablement_plan.md V1):
`make_fgclip2_encoder_layers_ddr` (the device-side N-layer loop over the DDR-resident, real-patch-count single
layer) against the REAL model's own golden trace after its 27th layer (`trace_image0['layers'][26]`), on the
real 972-patch test image, real pretrained weights for every layer.
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
from fgclip2_kernels import FGCLIP2_TABLE_COLS, make_fgclip2_encoder_layers_ddr  # noqa: E402
from fgclip2_pack import Checkpoint, pack_layer  # noqa: E402

MODEL_ROOT = Path(r"C:\Intel\guoliang\dsl\fg-clip2-so400m")
GOLDEN_PATH = Path(r"C:\Intel\guoliang\dsl\fgclip2_data\golden_fp32.pt")
HIDDEN, HEADS, HD_PAD, INTER_PAD, N_LAYERS = 1152, 16, 80, 4320, 27
BR, TN, BQ, C = 32, 32, 64, 128
COL = {c: i for i, c in enumerate(FGCLIP2_TABLE_COLS)}

needs_real_data = pytest.mark.skipif(
    not (MODEL_ROOT.exists() and GOLDEN_PATH.exists()),
    reason=f"real checkpoint/golden not found ({MODEL_ROOT}, {GOLDEN_PATH})")


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except RuntimeError:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _host_table(ck, tower):
    table = np.zeros((N_LAYERS, len(FGCLIP2_TABLE_COLS)), np.int64)
    keep = {}
    for li in range(N_LAYERS):
        w = pack_layer(ck, tower, li, hd_pad=HD_PAD, inter_pad=INTER_PAD)
        for name, key in (("Ln1W", "ln1w"), ("Ln1B", "ln1b"), ("Wq", "wq"), ("Bq", "bq"), ("Wk", "wk"), ("Bk", "bk"),
                          ("Wv", "wv"), ("Bv", "bv"), ("Wo", "wo"), ("Bo", "bo"), ("Ln2W", "ln2w"), ("Ln2B", "ln2b"),
                          ("W1", "w1"), ("B1", "b1"), ("W2", "w2"), ("B2", "b2")):
            arr = np.ascontiguousarray(w[key])
            keep[f"L{li}_{name}"] = arr
            table[li, COL[name]] = arr.ctypes.data
    return table, keep


@needs_real_data
@needs_cxx
def test_27_layers_match_the_real_models_own_post_layer27_trace():
    import torch

    golden = torch.load(GOLDEN_PATH, weights_only=False)
    embed = golden["trace_image0"]["embed"].numpy().astype(np.float32)
    layer26_out = golden["trace_image0"]["layers"][N_LAYERS - 1].numpy().astype(np.float64)
    valid = embed.shape[0]
    rows = -(-valid // BQ) * BQ
    rows = -(-rows // C) * C

    ck = Checkpoint(MODEL_ROOT)
    table, keep = _host_table(ck, "vision_model")

    hio = np.zeros((rows, HIDDEN), np.float32)
    hio[:valid] = embed

    func = make_fgclip2_encoder_layers_ddr(N_LAYERS, rows, valid, HIDDEN, HEADS, HD_PAD, INTER_PAD, BR=BR, TN=TN, BQ=BQ, C=C)
    low = _lower(func)
    lib = host_sim.build(low.source, "tl_fgclip2_real_27layers")
    qw = HEADS * HD_PAD
    arrays = {
        "Hio": hio, "Table": table,
        "Xn": np.zeros((rows, HIDDEN), np.float16), "Qp": np.zeros((rows, qw), np.float16),
        "Kp": np.zeros((rows, qw), np.float16), "Vp": np.zeros((rows, qw), np.float16),
        "Ao": np.zeros((rows, qw), np.float16), "Act": np.zeros((rows, INTER_PAD), np.float16),
    }
    status = host_sim.run(lib, lowering.order_arrays(low, arrays), 1, 2)
    host_sim.check_status(status)

    kernel_out = hio[:valid].astype(np.float64)
    rel = np.abs(kernel_out - layer26_out).max() / np.abs(layer26_out).max()
    print(f"\nFG-CLIP2 all 27 real layers, real image (972 patches): "
          f"max |kernel - real model| / max|real model| = {rel:.6f}")
    assert rel < 1e-1, rel
