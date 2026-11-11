"""Full text embedding against the real model (plan: docs/developer_guide/fgclip2_enablement_plan.md V3):
token+position embedding (host-side gather) -> 27 real text-tower layers (`make_fgclip2_encoder_layers_ddr`,
valid == rows == 64, unmasked) -> final LayerNorm + head projection (`make_fgclip2_text_tail_ddr`), real
pretrained weights throughout, compared against the REAL model's own recorded `pooled` text embedding
(`golden_fp32.pt`'s `trace_text0`, the first caption) and the golden file's own `emb_t[0]`.
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
from fgclip2_kernels import FGCLIP2_TABLE_COLS, make_fgclip2_encoder_layers_ddr, make_fgclip2_text_tail_ddr  # noqa: E402
from fgclip2_pack import Checkpoint, pack_layer, pack_text_embedding, pack_text_tail  # noqa: E402

MODEL_ROOT = Path(r"C:\Intel\guoliang\dsl\fg-clip2-so400m")
GOLDEN_PATH = Path(r"C:\Intel\guoliang\dsl\fgclip2_data\golden_fp32.pt")
HIDDEN, HEADS, HD_PAD, INTER_PAD, N_LAYERS, ROWS = 1152, 16, 80, 4320, 27, 64
BR, TN, BQ, C = 32, 32, 64, 64
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
def test_text_embedding_matches_the_real_models_pooled_output():
    import torch

    golden = torch.load(GOLDEN_PATH, weights_only=False)
    ids = golden["ids"][0].numpy().astype(np.int64)   # first caption, 64 token ids
    assert ids.shape[0] == ROWS
    pooled_real = golden["trace_text0"]["pooled"].numpy().astype(np.float64)
    emb_t0 = golden["emb_t"][0].numpy().astype(np.float64)
    assert np.abs(pooled_real - emb_t0).max() < 1e-3, "golden file's own pooled vs emb_t[0] disagree"

    ck = Checkpoint(MODEL_ROOT)
    h0 = pack_text_embedding(ck, ids)   # [64, hidden] fp32, host-side gather
    table, keep_layers = _host_table(ck, "text_model")
    tail_w = pack_text_tail(ck)

    hio = h0.copy()
    qw = HEADS * HD_PAD
    layers_func = make_fgclip2_encoder_layers_ddr(N_LAYERS, ROWS, ROWS, HIDDEN, HEADS, HD_PAD, INTER_PAD,
                                                  BR=BR, TN=TN, BQ=BQ, C=C)
    low1 = _lower(layers_func)
    lib1 = host_sim.build(low1.source, "tl_fgclip2_text_layers")
    arrays1 = {
        "Hio": hio, "Table": table,
        "Xn": np.zeros((ROWS, HIDDEN), np.float16), "Qp": np.zeros((ROWS, qw), np.float16),
        "Kp": np.zeros((ROWS, qw), np.float16), "Vp": np.zeros((ROWS, qw), np.float16),
        "Ao": np.zeros((ROWS, qw), np.float16), "Act": np.zeros((ROWS, INTER_PAD), np.float16),
    }
    status = host_sim.run(lib1, lowering.order_arrays(low1, arrays1), 1, 2)
    host_sim.check_status(status)

    tail_func = make_fgclip2_text_tail_ddr(ROWS, HIDDEN, BR=BR, TN=TN)
    low2 = _lower(tail_func)
    lib2 = host_sim.build(low2.source, "tl_fgclip2_text_tail_real")
    arrays2 = {
        "Hio": hio, "FinalLnW": tail_w["final_ln_w"], "FinalLnB": tail_w["final_ln_b"],
        "HeadW": tail_w["head_w"], "HeadB": tail_w["head_b"],
        "XL": np.zeros((ROWS, HIDDEN), np.float16), "EmbAll": np.zeros((ROWS, HIDDEN), np.float16),
    }
    status = host_sim.run(lib2, lowering.order_arrays(low2, arrays2), 1, 2)
    host_sim.check_status(status)

    kernel_emb = arrays2["EmbAll"][ROWS - 1].astype(np.float64)
    rel = np.abs(kernel_emb - pooled_real).max() / np.abs(pooled_real).max()
    cos = (kernel_emb @ pooled_real) / (np.linalg.norm(kernel_emb) * np.linalg.norm(pooled_real))
    print(f"\nFG-CLIP2 full text embedding, real caption, real pretrained weights: "
          f"max|kernel-real|/max|real| = {rel:.6f}, cosine similarity = {cos:.6f}")
    assert rel < 1.5e-1, rel
    assert cos > 0.99, cos
