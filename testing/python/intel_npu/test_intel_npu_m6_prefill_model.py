"""P5: a whole PREFILL chunk through several layers of BOTH kinds in one launch (`kernels.make_prefill_model`)
-- a device-side loop over the layer table, the Gated DeltaNet or the full-attention mixer per `Kind[li, 0]`,
the MoE after each, the fp32 residual stream in DDR.

The oracle is the two layer oracles chained, so the tests go after what only the model kernel can get wrong: that
each layer reads ITS OWN row of the table (a kernel that read layer 0's weights for every layer matches nothing
but is easy to write), that the GDN conv state survives being updated in place, that the attention layers index
their own KV slot, and that the layer kinds may be interleaved in one launch -- the shape that lost the device
for decode until the caches became kernel parameters (plan question 8).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import (GDN_PREFILL_TABLE, PREFILL_TABLE_COLS, gm_scales, make_moe_route_prefill,  # noqa: E402
                     make_prefill_model, pack_int4, prefill_model_reference, rope_cos_sin)

TILES, SHAVES = 2, 2
N, H, KH, VH, DK, DV, G = 64, 256, 4, 8, 64, 64, 128
NH, NKV, D, ROT, SMAX = 4, 2, 32, 16, 128
I, E, TOPK = 128, 32, 4
TB, C, TBq, TBF, BKV = 32, 32, 8, 4, 32
BN1, BN2, MB, BM, BNR, MBe, BNG, BND = 16, 16, 16, 32, 16, 16, 64, 64
KINDS = ["gdn", "attn", "gdn", "attn"]   # interleaved on purpose (question 8's shape), and TWO attention
L = len(KINDS)                            # layers so that the KV slot index of Kind[li, 1] is exercised
KVL = sum(1 for k in KINDS if k == "attn")
GQ, KO = NH // NKV, NH * D
QK, VC = KH * DK, VH * DV
CONV = 2 * QK + VC
NIN = CONV + VC
NIN_pad = -(-NIN // BN1) * BN1
NQKV = NH * 2 * D + 2 * NKV * D
NQKV_pad = -(-NQKV // BN1) * BN1
YW, OW = max(NIN_pad, NQKV_pad), max(VC, KO)
KGH, KGI = H // G, I // G
_, _RS = make_moe_route_prefill(N=N, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=TILES, shaves=SHAVES, as_block=True)
RR, SLOTS = _RS["RR"], _RS["SLOTS"]
THETA = 1e6
REL = 1e-1   # three layers of int4 projections, a chunked scan, a flash phase and 9 expert FFNs each


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")


def _kernel():
    return make_prefill_model(L=L, N=N, H=H, KH=KH, VH=VH, DK=DK, DV=DV, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX,
                              I=I, E=E, TOPK=TOPK, G=G, TB=TB, C=C, TBq=TBq, TBF=TBF, BKV=BKV, BN1=BN1, BN2=BN2,
                              MB=MB, BM=BM, BNR=BNR, MBe=MBe, BNG=BNG, BND=BND, tiles=TILES, shaves=SHAVES,
                              KVL=KVL)


def _layers(seed=0):
    """One weight set per layer, with a different seed each, so a kernel that read one row of the table for every
    layer cannot pass."""
    out, kv = [], 0
    for li, kind in enumerate(KINDS):
        rng = np.random.default_rng(seed + 100 * li + 1)
        p = dict(kind=kind,
                 in_norm=(rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16),
                 post_norm=(rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16),
                 wr=(rng.standard_normal((E + 1, H), np.float32) * 0.1).astype(np.float16),
                 gu_int=rng.integers(-8, 8, (E + 1, 2 * I, H), dtype=np.int8),
                 gu_s=(rng.standard_normal((E + 1, 2 * I, KGH), np.float32) * 0.02).astype(np.float16),
                 d_int=rng.integers(-8, 8, (E + 1, H, I), dtype=np.int8),
                 d_s=(rng.standard_normal((E + 1, H, KGI), np.float32) * 0.02).astype(np.float16))
        if kind == "gdn":
            p.update(win_int=rng.integers(-8, 8, (NIN_pad, H), dtype=np.int8),
                     win_s=(rng.standard_normal((NIN_pad, H // G), np.float32) * 0.02).astype(np.float16),
                     wba=(rng.standard_normal((2 * VH, H), np.float32) * 0.05).astype(np.float16),
                     conv_w=(rng.standard_normal((4, CONV), np.float32) * 0.3).astype(np.float16),
                     conv_state=(rng.standard_normal((3, CONV), np.float32) * 0.3).astype(np.float16),
                     a_log=(rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32),
                     dt_bias=(rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32),
                     gnorm=(1.0 + rng.standard_normal((1, DV), np.float32) * 0.1).astype(np.float16),
                     wout_int=rng.integers(-8, 8, (H, VC), dtype=np.int8),
                     wout_s=(rng.standard_normal((H, VC // G), np.float32) * 0.02).astype(np.float16),
                     ssm=(rng.standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32))
        else:
            p.update(kv=kv,
                     wqkv_int=rng.integers(-8, 8, (NQKV_pad, H), dtype=np.int8),
                     wqkv_s=(rng.standard_normal((NQKV_pad, H // G), np.float32) * 0.02).astype(np.float16),
                     qnorm=(0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16),
                     knorm=(0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16),
                     wo_int=rng.integers(-8, 8, (H, KO), dtype=np.int8),
                     wo_s=(rng.standard_normal((H, KO // G), np.float32) * 0.02).astype(np.float16))
            kv += 1
        out.append(p)
    return out


def _cos_sin(pos0):
    return np.stack([rope_cos_sin(pos0 + t, ROT, THETA).reshape(-1) for t in range(N)]).astype(np.float32)


def _host_arrays(layers):
    """One contiguous array per table entry, kept alive by the returned dict; the table holds their addresses,
    exactly as the device driver fills it from `Session.address`."""
    rows = []
    keep = {}
    for li, p in enumerate(layers):
        wr_pad = np.zeros((RR, H), np.float16)
        wr_pad[:E + 1] = p["wr"]
        a = {"InNorm": p["in_norm"], "PostNorm": p["post_norm"], "Wr": wr_pad,
             "WguQ": pack_int4(p["gu_int"].reshape(-1, H)), "WguS": gm_scales(p["gu_s"].reshape(-1, KGH)),
             "WdQ": pack_int4(p["d_int"].reshape(-1, I)), "WdS": gm_scales(p["d_s"].reshape(-1, KGI))}
        if p["kind"] == "gdn":
            a.update({"W0": pack_int4(p["win_int"]), "W1": gm_scales(p["win_s"]), "W2": p["conv_w"],
                      "W3": p["a_log"], "W4": p["dt_bias"], "W5": p["gnorm"], "W6": pack_int4(p["wout_int"]),
                      "W7": gm_scales(p["wout_s"]), "W8": p["wba"],
                      "S0": p["conv_state"].copy(), "S1": p["ssm"].copy()})
        else:
            a.update({"W0": pack_int4(p["wqkv_int"]), "W1": gm_scales(p["wqkv_s"]), "W2": p["qnorm"],
                      "W3": p["knorm"], "W4": pack_int4(p["wo_int"]), "W5": gm_scales(p["wo_s"])})
        a = {k: np.ascontiguousarray(v) for k, v in a.items()}
        keep.update({f"L{li}_{k}": v for k, v in a.items()})
        rows.append(a)
    return rows, keep


def _run(layers, h, pos0=0, n=N):
    low = lowering.lower(_kernel())
    lib = host_sim.build(low.source, "prefill_model")
    rows, keep = _host_arrays(layers)
    col = {c: i for i, c in enumerate(PREFILL_TABLE_COLS)}
    table = np.zeros((L, len(PREFILL_TABLE_COLS)), np.int64)
    kind = np.zeros((L, 4), np.int32)
    kv = 0
    for li, p in enumerate(layers):
        kind[li, 0] = 0 if p["kind"] == "gdn" else 1
        if p["kind"] == "attn":
            kind[li, 1] = kv
            kv += 1
        for name, arr in rows[li].items():
            table[li, col[name]] = arr.ctypes.data
    hio = h.copy()
    kc = np.zeros((KVL, NKV, SMAX, D), np.float16)
    vt = np.zeros((KVL, NKV, D, SMAX), np.float16)
    arrays = {"Hio": hio, "Pos": np.array([[pos0]], np.int32), "Nt": np.array([[n]], np.int32),
              "CosSin": _cos_sin(pos0), "Kind": kind, "Table": table, "KcAll": kc, "VtAll": vt,
              "Xs": np.zeros((N, H), np.float16), "Ys": np.zeros((N, YW), np.float16),
              "BAs": np.zeros((N, 2 * VH), np.float16),
              "QKs": np.zeros((KH, N, 2 * DK), np.float32), "Vs": np.zeros((VH, N, DV), np.float16),
              "Gbs": np.zeros((VH, N, 2), np.float32), "Zs": np.zeros((VH, N, DV), np.float16),
              "Q": np.zeros((NKV, N, GQ, D), np.float16), "Gate": np.zeros((N, KO), np.float16),
              "Os": np.zeros((N, OW), np.float16), "Mix": np.zeros((N, H), np.float16),
              "Logits": np.zeros((N, RR), np.float16), "Ids": np.zeros((N, TOPK), np.int32),
              "Wts": np.zeros((N, 16), np.float32), "Cnt": np.zeros((E + 1,), np.int32),
              "Off": np.zeros((E + 2,), np.int32), "Slot": np.zeros((SLOTS,), np.int32),
              "PartE": np.zeros((TILES, N, H), np.float32), "MoeOut": np.zeros((N, H), np.float32)}
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), tiles=TILES, shaves=SHAVES))
    return hio, kc, vt, rows, keep


def _oracle(layers, h, pos0=0, n=N):
    """Returns (stream, kc, vt, rows) -- `rows` are the oracle's own layer dicts, whose `conv_state` / `ssm` it
    rebinds to the UPDATED states; reading them off `layers` instead compares against the initial state."""
    kc = [np.zeros((NKV, SMAX, D), np.float16) for _ in range(KVL)]
    vt = [np.zeros((NKV, D, SMAX), np.float16) for _ in range(KVL)]
    rows = [dict(p) for p in layers]
    out, kc, vt = prefill_model_reference(rows, h, _cos_sin(pos0), pos0, n, NH=NH, NKV=NKV, D=D,
                                          ROT=ROT, KH=KH, VH=VH, DK=DK, DV=DV, E=E, TOPK=TOPK, G=G, kc=kc, vt=vt)
    return out, kc, vt, rows


def test_the_model_does_not_reintroduce_question_10():
    src = lowering.lower(_kernel()).source
    from test_intel_npu_scoped_cmx import allocations_inside_loops
    # the layer loop puts every phase one level deeper than the layer kernels; the expert phase's own buffers are
    # the depth-1 form P2b earned its device trials in, so the bar is the layer loop plus that
    assert max((d for _, d in allocations_inside_loops(src)), default=0) <= 2
    assert src.count("ctx.sync_grid();") >= 10
    # the caches are kernel parameters; only the weights come from the table (question 8)
    assert "KcAll" in src and "VtAll" in src
    rows = [ln for ln in src.splitlines() if "uint64_t)Table" in ln]
    assert len(rows) >= 17, f"{len(rows)} weights bound from the layer table"
    stride = f"* {len(PREFILL_TABLE_COLS)})"   # every one indexed by the layer variable, i.e. inside the loop
    assert all(stride in ln for ln in rows), [ln.strip()[:90] for ln in rows if stride not in ln]


@needs_cxx
def test_the_model_matches_the_layer_oracles_chained():
    layers = _layers()
    h = (np.random.default_rng(0).standard_normal((N, H), np.float32) * 0.5).astype(np.float32)
    hio, kc, vt, rows, _ = _run(layers, h)
    ref, kc_ref, vt_ref, ref_rows = _oracle(layers, h)
    assert np.abs(hio - ref).max() / np.abs(ref).max() < REL
    for k in range(KVL):
        rel = (np.abs(kc[k].astype(np.float32) - kc_ref[k].astype(np.float32)).max()
               / max(float(np.abs(kc_ref[k].astype(np.float32)).max()), 1e-6))
        assert rel < REL, f"KV slot {k} rel {rel}"
    # both GDN states are updated IN PLACE through the table address
    for li, p in enumerate(layers):
        if p["kind"] == "gdn":
            for name, key in (("S0", "conv_state"), ("S1", "ssm")):
                got = rows[li][name].astype(np.float32)
                exp = ref_rows[li][key].astype(np.float32)
                rel = np.abs(got - exp).max() / max(float(np.abs(exp).max()), 1e-6)
                assert rel < REL, f"layer {li} {key} rel {rel}"


@needs_cxx
def test_every_layer_reads_its_own_row_of_the_table():
    """Point layer 2's weights at layer 0's row and the output must move. Without it, a kernel that hoisted the
    table read out of the layer loop -- the easiest way to get this wrong -- would pass the oracle test whenever
    two layers of a kind happen to agree."""
    layers = _layers()
    h = (np.random.default_rng(0).standard_normal((N, H), np.float32) * 0.5).astype(np.float32)
    a, _, _, _, _ = _run(layers, h)
    swapped = [dict(p) for p in layers]
    for k in ("win_int", "win_s", "wout_int", "wout_s", "in_norm", "post_norm"):
        swapped[2][k] = layers[0][k]
    b, _, _, _, _ = _run(swapped, h)
    assert np.abs(a - b).max() > 0, "layer 2 did not read its own table row"


@needs_cxx
def test_a_short_prompt_leaves_the_dead_tokens_alone():
    layers = _layers()
    h = (np.random.default_rng(0).standard_normal((N, H), np.float32) * 0.5).astype(np.float32)
    n = N - TB
    hio, kc, _, _, _ = _run(layers, h, n=n)
    ref, _, _, _ = _oracle(layers, h, n=n)
    assert np.abs(hio[:n] - ref[:n]).max() / np.abs(ref[:n]).max() < REL
    assert np.abs(hio[n:] - h[n:]).max() == 0.0, "a dead token was added to the stream"
    assert np.abs(kc[:, :, n:].astype(np.float32)).max() == 0, "a dead token was appended to a cache"
