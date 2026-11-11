"""Real-checkpoint validation (plan: docs/developer_guide/fgclip2_enablement_plan.md V1/V4 overlap): one FG-CLIP2
vision encoder layer's REAL pretrained weights (layer 0, `model.safetensors`) against the REAL model's own
per-layer trace (`golden_fp32.pt`'s `trace_image0`, produced by npu_dsl's `fgclip2_e2e/golden.py` from the
actual HF reference implementation on a real test image, 972 patches). This is the strongest correctness check
available without device hardware: not a numpy oracle written to match this port's own assumptions, but the
real model's own recorded intermediate activation.

Skipped entirely if the checkpoint / golden file are not present on this machine (both are machine-local,
multi-GB artifacts outside the repo, matching npu_dsl's own hardcoded paths in `fgclip2_e2e/prep.py`).
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
from fgclip2_kernels import make_fgclip2_encoder_layer_ddr  # noqa: E402
from fgclip2_pack import Checkpoint, pack_layer  # noqa: E402

MODEL_ROOT = Path(r"C:\Intel\guoliang\dsl\fg-clip2-so400m")
GOLDEN_PATH = Path(r"C:\Intel\guoliang\dsl\fgclip2_data\golden_fp32.pt")
HIDDEN, HEADS, HD_PAD, INTER_PAD = 1152, 16, 80, 4320
BR, TN, BQ, C = 32, 32, 64, 128

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


@needs_real_data
@needs_cxx
def test_encoder_layer0_matches_the_real_models_own_layer0_trace():
    import torch

    golden = torch.load(GOLDEN_PATH, weights_only=False)
    embed = golden["trace_image0"]["embed"].numpy().astype(np.float32)   # [972, 1152]: patch+pos embed, pre-layer-0
    layer0_out = golden["trace_image0"]["layers"][0].numpy().astype(np.float64)   # the REAL model's own layer-0 output
    valid = embed.shape[0]
    rows = -(-valid // BQ) * BQ   # round up to a multiple of lcm(BR, BQ, C) -- BQ=64 is the binding one here
    rows = -(-rows // C) * C

    ck = Checkpoint(MODEL_ROOT)
    w = pack_layer(ck, "vision_model", 0, hd_pad=HD_PAD, inter_pad=INTER_PAD)

    hio = np.zeros((rows, HIDDEN), np.float32)
    hio[:valid] = embed

    func = make_fgclip2_encoder_layer_ddr(rows, valid, HIDDEN, HEADS, HD_PAD, INTER_PAD, BR=BR, TN=TN, BQ=BQ, C=C)
    low = _lower(func)
    lib = host_sim.build(low.source, "tl_fgclip2_real_layer0")
    qw = HEADS * HD_PAD
    arrays = {
        "Hio": hio, "Ln1W": w["ln1w"], "Ln1B": w["ln1b"],
        "Wq": w["wq"], "Bq": w["bq"], "Wk": w["wk"], "Bk": w["bk"], "Wv": w["wv"], "Bv": w["bv"],
        "Wo": w["wo"], "Bo": w["bo"], "Ln2W": w["ln2w"], "Ln2B": w["ln2b"],
        "W1": w["w1"], "B1": w["b1"], "W2": w["w2"], "B2": w["b2"],
        "Xn": np.zeros((rows, HIDDEN), np.float16), "Qp": np.zeros((rows, qw), np.float16),
        "Kp": np.zeros((rows, qw), np.float16), "Vp": np.zeros((rows, qw), np.float16),
        "Ao": np.zeros((rows, qw), np.float16), "Act": np.zeros((rows, INTER_PAD), np.float16),
    }
    status = host_sim.run(lib, lowering.order_arrays(low, arrays), 1, 2)
    host_sim.check_status(status)

    kernel_out = hio[:valid].astype(np.float64)
    rel = np.abs(kernel_out - layer0_out).max() / np.abs(layer0_out).max()
    print(f"\nFG-CLIP2 real layer-0, real image (972 patches): max |kernel - real model| / max|real model| = {rel:.6f}")
    assert rel < 5e-2, rel
