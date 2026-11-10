"""M5 fused decoder-layer kernels (plan section 4): one launch per layer with the norms and the fp32 residual stream
in the kernel (`kernels.make_gdn_layer`, `make_attn_layer`, composed from the `gdn_block` / `attn_block` / `moe_block`
/ `norm_block` macros). Hardware-free: the small GDN layer matches a composed numpy oracle (gdn_reference, moe_reference
and the zero-centred norms) in the host sim, the states are updated, and a perturbed expert is a FAIL; both layer
kernels lower at Qwen3.6's shape within the CMX budget (only possible with scoped CMX: the standalone GDN kernel alone
was near the budget)."""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import gdn_reference, gdn_sizes, make_attn_layer, make_gdn_layer, moe_reference, pack_gdn, pack_moe  # noqa: E402
from test_intel_npu_m3_moe import _problem as moe_problem  # noqa: E402
from test_intel_npu_m4_gdn import gdn_problem  # noqa: E402

G, TILES, SHAVES = 128, 3, 2
GDN_SMALL = dict(H=512, KH=4, VH=8, DK=64, DV=64, G=G, tiles=TILES, shaves=SHAVES)
GDN_KW = dict(BN1=64, R1=64, BN2=64, R2=32, CB=256)
MOE_SMALL = dict(H=512, I=128, E=16, TOPK=2)
MOE_KW = dict(BNG=128, BND=128, RG=64, RD=128)
EPS = 1e-6


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except RuntimeError:
        return False


def rmsnorm_zc(h, w):
    return h * (1.0 / np.sqrt((h * h).mean() + EPS)) * (1.0 + w.astype(np.float32).reshape(-1))


def test_both_layer_kernels_lower_at_the_model_shape():
    for f in (make_gdn_layer, make_attn_layer):
        src = _lower(f()).source
        assert src.count("ctx.sync_grid();") >= 6 and "kernel_entry" in src


@pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")
def test_small_gdn_layer_matches_the_composed_oracle():
    H = GDN_SMALL["H"]
    gp = gdn_problem(0, **GDN_SMALL)
    x, router, gu_int, gu_scales, d_int, d_scales, _ = moe_problem(1, **MOE_SMALL)
    rng = np.random.default_rng(2)
    h = (rng.standard_normal(H) * 0.5).astype(np.float32)
    in_norm = rng.uniform(-0.3, 0.3, size=(1, H)).astype(np.float16)
    post_norm = rng.uniform(-0.3, 0.3, size=(1, H)).astype(np.float16)
    gpk = pack_gdn(gp["win_int"], gp["win_scales"], gp["wout_int"], gp["wout_scales"], wba=gp["wba"], tiles=TILES, shaves=SHAVES,
                   BN1=GDN_KW["BN1"], BN2=GDN_KW["BN2"])
    mpk = pack_moe(router, gu_int, gu_scales, d_int, d_scales, tiles=TILES, shaves=SHAVES, BNR=16, BNG=MOE_KW["BNG"], BND=MOE_KW["BND"], G=G)
    low = _lower(make_gdn_layer(I=MOE_SMALL["I"], E=MOE_SMALL["E"], TOPK=MOE_SMALL["TOPK"], pipelined=False, gdn_kw=GDN_KW, moe_kw=MOE_KW,
                                **GDN_SMALL))
    lib = host_sim.build(low.source, "tl_gdn_layer_sim")
    sz = gdn_sizes(H, GDN_SMALL["KH"], GDN_SMALL["VH"], GDN_SMALL["DK"], tiles=TILES, shaves=SHAVES, BN1=GDN_KW["BN1"], BN2=GDN_KW["BN2"])
    hio = h.reshape(1, H).copy()
    arrays = dict(Hio=hio, InNorm=in_norm, PostNorm=post_norm, WinQ=gpk["WinQ"], WinS=gpk["WinS"], Wba=gpk["Wba"],
                  ConvW=gp["conv_w"], ALog=gp["a_log"],
                  DtBias=gp["dt_bias"], GNorm=gp["g_norm"], ConvState=gp["conv_state"].copy(), Ssm=gp["ssm"].copy(),
                  WoutQ=gpk["WoutQ"], WoutS=gpk["WoutS"],
                  Wr=mpk["Wr"], WguQ=mpk["WguQ"], WguS=mpk["WguS"], WdQ=mpk["WdQ"], WdS=mpk["WdS"],
                  Y=np.zeros((1, sz["YLEN"]), np.float16),
                  BA=np.zeros((1, sz["BA_LEN"]), np.float16), CV=np.zeros((1, sz["CVLEN"]), np.float16),
                  O=np.zeros((1, sz["OPAD"]), np.float16), Mix=np.zeros((1, sz["NOUT_pad"]), np.float16),
                  Logits=np.zeros((16, mpk["Wr"].shape[0]), np.float16), PartM=np.zeros((TILES, H), np.float32))
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    # the oracle: h -> norm -> mixer (no residual) -> h2 -> norm -> moe -> h3
    x1 = rmsnorm_zc(h, in_norm).astype(np.float16)
    zero = np.zeros((1, H), np.float16)
    mix, ref_cs, ref_ssm = gdn_reference(x1.reshape(1, H), zero, gp["win_int"], gp["win_scales"], gp["wout_int"], gp["wout_scales"],
                                         gp["conv_w"], gp["a_log"], gp["dt_bias"], gp["g_norm"], gp["conv_state"], gp["ssm"],
                                         KH=GDN_SMALL["KH"], VH=GDN_SMALL["VH"], DK=GDN_SMALL["DK"], G=G, wba=gp["wba"])
    h2 = h + mix
    x2 = rmsnorm_zc(h2, post_norm).astype(np.float16)
    moe_out, _, _ = moe_reference(x2.reshape(1, H), router, gu_int, gu_scales, d_int, d_scales, zero, MOE_SMALL["TOPK"], G)
    h3 = h2 + moe_out
    scale = float(np.abs(h3 - h).max())
    err = float(np.abs(hio[0] - h3).max())
    assert err / scale < 2e-2, (err, scale)
    assert np.abs(arrays["Ssm"] - ref_ssm).max() < 1e-3 * np.abs(ref_ssm).max()
    assert np.array_equal(arrays["ConvState"][:2], ref_cs[:2])
    # negative: a routed expert's weights perturbed -> the layer output moves
    gu2 = gu_int.copy()
    _, order, _ = moe_reference(x2.reshape(1, H), router, gu_int, gu_scales, d_int, d_scales, zero, MOE_SMALL["TOPK"], G)
    gu2[int(order[0]), :, :G] = 7
    mpk2 = pack_moe(router, gu2, gu_scales, d_int, d_scales, tiles=TILES, shaves=SHAVES, BNR=16, BNG=MOE_KW["BNG"], BND=MOE_KW["BND"], G=G)
    arrays2 = dict(arrays, Hio=h.reshape(1, H).copy(), ConvState=gp["conv_state"].copy(), Ssm=gp["ssm"].copy(),
                   WguQ=mpk2["WguQ"], WguS=mpk2["WguS"])
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays2), TILES, SHAVES))
    assert np.abs(arrays2["Hio"][0] - h3).max() / scale > 2e-2
