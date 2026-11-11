"""Full image embedding against the real model (plan: docs/developer_guide/fgclip2_enablement_plan.md V1/V2):
27 real encoder layers (`make_fgclip2_encoder_layers_ddr`) followed by the attention-pooling head
(`make_fgclip2_pooling_head_ddr`), real pretrained weights throughout, compared against the REAL model's own
recorded `pooled` image embedding (`golden_fp32.pt`'s `trace_image0`) AND the golden file's own `emb_i[0]` (the
same number, computed a different way in `golden.py`, as a self-consistency check of the golden file itself).

Two separate host-sim builds/runs chained through plain numpy arrays (not a single launch) -- correctness
first; whether this should be one launch is a later (perf) question, not this test's.
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
from fgclip2_kernels import FGCLIP2_TABLE_COLS, make_fgclip2_encoder_layers_ddr, make_fgclip2_pooling_head_ddr  # noqa: E402
from fgclip2_pack import Checkpoint, pack_layer, pack_vision_head  # noqa: E402

MODEL_ROOT = Path(r"C:\Intel\guoliang\dsl\fg-clip2-so400m")
GOLDEN_PATH = Path(r"C:\Intel\guoliang\dsl\fgclip2_data\golden_fp32.pt")
HIDDEN, HEADS, HD_PAD, INTER_PAD, N_LAYERS = 1152, 16, 80, 4320, 27
BR, TN, BQ, C = 32, 32, 16, 128
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
def test_image_embedding_matches_the_real_models_pooled_output():
    import torch

    golden = torch.load(GOLDEN_PATH, weights_only=False)
    embed = golden["trace_image0"]["embed"].numpy().astype(np.float32)
    pooled_real = golden["trace_image0"]["pooled"].numpy().astype(np.float64)
    emb_i0 = golden["emb_i"][0].numpy().astype(np.float64)
    assert np.abs(pooled_real - emb_i0).max() < 1e-3, "golden file's own pooled vs emb_i[0] disagree"

    valid = embed.shape[0]
    rows = -(-valid // BQ) * BQ
    rows = -(-rows // C) * C
    qw = HEADS * HD_PAD

    ck = Checkpoint(MODEL_ROOT)
    table, keep_layers = _host_table(ck, "vision_model")
    head_w = pack_vision_head(ck, hd_pad=HD_PAD, inter_pad=INTER_PAD)

    hio = np.zeros((rows, HIDDEN), np.float32)
    hio[:valid] = embed

    # stage 1: 27 real encoder layers
    layers_func = make_fgclip2_encoder_layers_ddr(N_LAYERS, rows, valid, HIDDEN, HEADS, HD_PAD, INTER_PAD,
                                                  BR=BR, TN=TN, BQ=64, C=C)
    low1 = _lower(layers_func)
    lib1 = host_sim.build(low1.source, "tl_fgclip2_image_layers")
    arrays1 = {
        "Hio": hio, "Table": table,
        "Xn": np.zeros((rows, HIDDEN), np.float16), "Qp": np.zeros((rows, qw), np.float16),
        "Kp": np.zeros((rows, qw), np.float16), "Vp": np.zeros((rows, qw), np.float16),
        "Ao": np.zeros((rows, qw), np.float16), "Act": np.zeros((rows, INTER_PAD), np.float16),
    }
    status = host_sim.run(lib1, lowering.order_arrays(low1, arrays1), 1, 2)
    host_sim.check_status(status)

    # stage 2: post_layernorm + attention-pooling head
    head_func = make_fgclip2_pooling_head_ddr(rows, valid, HIDDEN, HEADS, HD_PAD, INTER_PAD, BR=BR, TN=TN, BQ=BQ, C=C)
    low2 = _lower(head_func)
    lib2 = host_sim.build(low2.source, "tl_fgclip2_image_head")
    arrays2 = {
        "Hio": hio, "PostLnW": head_w["post_ln_w"], "PostLnB": head_w["post_ln_b"], "MapQ": np.tile(head_w["map_q"], (BQ, 1)),
        "MapWk": head_w["map_wk"], "MapBk": head_w["map_bk"], "MapWv": head_w["map_wv"], "MapBv": head_w["map_bv"],
        "MapWo": head_w["map_wo"], "MapBo": head_w["map_bo"], "HLnW": head_w["hln_w"], "HLnB": head_w["hln_b"],
        "MapW1": head_w["map_w1"], "MapB1": head_w["map_b1"], "MapW2": head_w["map_w2"], "MapB2": head_w["map_b2"],
        "Xn": np.zeros((rows, HIDDEN), np.float16), "Ks": np.zeros((rows, qw), np.float16),
        "Vs": np.zeros((rows, qw), np.float16), "Ao": np.zeros((BQ, qw), np.float16),
        "ALn": np.zeros((BQ, HIDDEN), np.float16), "Act1": np.zeros((BQ, INTER_PAD), np.float16),
        "Emb": np.zeros((BQ, HIDDEN), np.float32),
    }
    status = host_sim.run(lib2, lowering.order_arrays(low2, arrays2), 1, 2)
    host_sim.check_status(status)

    kernel_emb = arrays2["Emb"][0].astype(np.float64)
    rel = np.abs(kernel_emb - pooled_real).max() / np.abs(pooled_real).max()
    cos = (kernel_emb @ pooled_real) / (np.linalg.norm(kernel_emb) * np.linalg.norm(pooled_real))
    print(f"\nFG-CLIP2 full image embedding, real image (972 patches), real pretrained weights: "
          f"max|kernel-real|/max|real| = {rel:.6f}, cosine similarity = {cos:.6f}")
    assert rel < 1.5e-1, rel
    assert cos > 0.99, cos
