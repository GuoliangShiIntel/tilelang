"""The capstone measurement (plan: docs/developer_guide/fgclip2_enablement_plan.md V4): the actual CLIP
similarity logit for one real (image, caption) pair, computed end to end from THIS backend's own kernel
outputs (both towers, real pretrained weights), compared against the real model's own logit for the same
pair. Reuses `test_intel_npu_fgclip2_real_image_embedding.py` / `..._real_text_embedding.py`'s exact pipelines;
this is the same vectors, just carried one step further into the similarity formula
(`reference.py::similarity`) instead of stopping at per-tower cosine similarity.
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
from fgclip2_kernels import (FGCLIP2_TABLE_COLS, make_fgclip2_encoder_layers_ddr,  # noqa: E402
                             make_fgclip2_pooling_head_ddr, make_fgclip2_text_tail_ddr)
from fgclip2_pack import Checkpoint, pack_layer, pack_text_embedding, pack_text_tail, pack_vision_head  # noqa: E402

MODEL_ROOT = Path(r"C:\Intel\guoliang\dsl\fg-clip2-so400m")
GOLDEN_PATH = Path(r"C:\Intel\guoliang\dsl\fgclip2_data\golden_fp32.pt")
HIDDEN, HEADS, HD_PAD, INTER_PAD, N_LAYERS = 1152, 16, 80, 4320, 27
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


def _host_table(ck, tower, n_layers):
    table = np.zeros((n_layers, len(FGCLIP2_TABLE_COLS)), np.int64)
    keep = {}
    for li in range(n_layers):
        w = pack_layer(ck, tower, li, hd_pad=HD_PAD, inter_pad=INTER_PAD)
        for name, key in (("Ln1W", "ln1w"), ("Ln1B", "ln1b"), ("Wq", "wq"), ("Bq", "bq"), ("Wk", "wk"), ("Bk", "bk"),
                          ("Wv", "wv"), ("Bv", "bv"), ("Wo", "wo"), ("Bo", "bo"), ("Ln2W", "ln2w"), ("Ln2B", "ln2b"),
                          ("W1", "w1"), ("B1", "b1"), ("W2", "w2"), ("B2", "b2")):
            arr = np.ascontiguousarray(w[key])
            keep[f"L{li}_{name}"] = arr
            table[li, COL[name]] = arr.ctypes.data
    return table, keep


def _image_embedding(ck):
    import torch

    golden = torch.load(GOLDEN_PATH, weights_only=False)
    embed = golden["trace_image0"]["embed"].numpy().astype(np.float32)
    valid = embed.shape[0]
    BR, TN, BQc, C = 32, 32, 64, 128
    rows = -(-valid // BQc) * BQc
    rows = -(-rows // C) * C
    qw = HEADS * HD_PAD

    table, keep = _host_table(ck, "vision_model", N_LAYERS)
    hio = np.zeros((rows, HIDDEN), np.float32)
    hio[:valid] = embed

    layers_func = make_fgclip2_encoder_layers_ddr(N_LAYERS, rows, valid, HIDDEN, HEADS, HD_PAD, INTER_PAD,
                                                  BR=BR, TN=TN, BQ=BQc, C=C)
    low1 = _lower(layers_func)
    lib1 = host_sim.build(low1.source, "tl_fgclip2_e2e_image_layers")
    arrays1 = {
        "Hio": hio, "Table": table,
        "Xn": np.zeros((rows, HIDDEN), np.float16), "Qp": np.zeros((rows, qw), np.float16),
        "Kp": np.zeros((rows, qw), np.float16), "Vp": np.zeros((rows, qw), np.float16),
        "Ao": np.zeros((rows, qw), np.float16), "Act": np.zeros((rows, INTER_PAD), np.float16),
    }
    host_sim.check_status(host_sim.run(lib1, lowering.order_arrays(low1, arrays1), 1, 2))

    BQh = 16
    head_w = pack_vision_head(ck, hd_pad=HD_PAD, inter_pad=INTER_PAD)
    head_func = make_fgclip2_pooling_head_ddr(rows, valid, HIDDEN, HEADS, HD_PAD, INTER_PAD, BR=BR, TN=TN, BQ=BQh, C=C)
    low2 = _lower(head_func)
    lib2 = host_sim.build(low2.source, "tl_fgclip2_e2e_image_head")
    arrays2 = {
        "Hio": hio, "PostLnW": head_w["post_ln_w"], "PostLnB": head_w["post_ln_b"],
        "MapQ": np.tile(head_w["map_q"], (BQh, 1)),
        "MapWk": head_w["map_wk"], "MapBk": head_w["map_bk"], "MapWv": head_w["map_wv"], "MapBv": head_w["map_bv"],
        "MapWo": head_w["map_wo"], "MapBo": head_w["map_bo"], "HLnW": head_w["hln_w"], "HLnB": head_w["hln_b"],
        "MapW1": head_w["map_w1"], "MapB1": head_w["map_b1"], "MapW2": head_w["map_w2"], "MapB2": head_w["map_b2"],
        "Xn": np.zeros((rows, HIDDEN), np.float16), "Ks": np.zeros((rows, qw), np.float16),
        "Vs": np.zeros((rows, qw), np.float16), "Ao": np.zeros((BQh, qw), np.float16),
        "ALn": np.zeros((BQh, HIDDEN), np.float16), "Act1": np.zeros((BQh, INTER_PAD), np.float16),
        "Emb": np.zeros((BQh, HIDDEN), np.float32),
    }
    host_sim.check_status(host_sim.run(lib2, lowering.order_arrays(low2, arrays2), 1, 2))
    return arrays2["Emb"][0].astype(np.float64)


def _text_embedding(ck):
    import torch

    golden = torch.load(GOLDEN_PATH, weights_only=False)
    ids = golden["ids"][0].numpy().astype(np.int64)
    rows = ids.shape[0]
    BR, TN, BQ, C = 32, 32, 64, 64
    qw = HEADS * HD_PAD

    h0 = pack_text_embedding(ck, ids)
    table, keep = _host_table(ck, "text_model", N_LAYERS)
    tail_w = pack_text_tail(ck)

    hio = h0.copy()
    layers_func = make_fgclip2_encoder_layers_ddr(N_LAYERS, rows, rows, HIDDEN, HEADS, HD_PAD, INTER_PAD,
                                                  BR=BR, TN=TN, BQ=BQ, C=C)
    low1 = _lower(layers_func)
    lib1 = host_sim.build(low1.source, "tl_fgclip2_e2e_text_layers")
    arrays1 = {
        "Hio": hio, "Table": table,
        "Xn": np.zeros((rows, HIDDEN), np.float16), "Qp": np.zeros((rows, qw), np.float16),
        "Kp": np.zeros((rows, qw), np.float16), "Vp": np.zeros((rows, qw), np.float16),
        "Ao": np.zeros((rows, qw), np.float16), "Act": np.zeros((rows, INTER_PAD), np.float16),
    }
    host_sim.check_status(host_sim.run(lib1, lowering.order_arrays(low1, arrays1), 1, 2))

    tail_func = make_fgclip2_text_tail_ddr(rows, HIDDEN, BR=BR, TN=TN)
    low2 = _lower(tail_func)
    lib2 = host_sim.build(low2.source, "tl_fgclip2_e2e_text_tail")
    arrays2 = {
        "Hio": hio, "FinalLnW": tail_w["final_ln_w"], "FinalLnB": tail_w["final_ln_b"],
        "HeadW": tail_w["head_w"], "HeadB": tail_w["head_b"],
        "XL": np.zeros((rows, HIDDEN), np.float16), "EmbAll": np.zeros((rows, HIDDEN), np.float16),
    }
    host_sim.check_status(host_sim.run(lib2, lowering.order_arrays(low2, arrays2), 1, 2))
    return arrays2["EmbAll"][rows - 1].astype(np.float64)


@needs_real_data
@needs_cxx
def test_end_to_end_clip_similarity_matches_the_real_model():
    import torch

    golden = torch.load(GOLDEN_PATH, weights_only=False)
    ls, lb = float(golden["logit_scale"]), float(golden["logit_bias"])
    real_sim = float(golden["sim"][0, 0])   # image 0 vs caption 0: caption 0 belongs to image 0 (truth[0] == 0)
    assert int(golden["truth"][0]) == 0

    ck = Checkpoint(MODEL_ROOT)
    img = _image_embedding(ck)
    txt = _text_embedding(ck)

    img_n = img / np.linalg.norm(img)
    txt_n = txt / np.linalg.norm(txt)
    kernel_sim = float(img_n @ txt_n * np.exp(ls) + lb)

    print(f"\nFG-CLIP2 end-to-end CLIP similarity logit, real image x real caption, real pretrained weights, "
          f"this backend's own kernel outputs for BOTH towers: "
          f"kernel = {kernel_sim:.4f}, real model = {real_sim:.4f}, abs diff = {abs(kernel_sim - real_sim):.4f}")
    assert abs(kernel_sim - real_sim) < 2.0, (kernel_sim, real_sim)
