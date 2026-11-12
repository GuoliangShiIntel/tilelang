"""Gemma4 V0: the MoE expert FFN (gemma4_kernels.make_gemma4_moe_expert_prefill, V0-grade -- every
WORKER owns whole experts round-robin, no tile sharing) and the partial-sum combine
(make_gemma4_moe_combine), chained behind make_gemma4_moe_route_prefill's grouping. Each kernel is run
as its own host-sim launch here (a staged pipeline), matching how the full layer composes them as
macros within one launch later. See docs/developer_guide/gemma4_enablement_plan.md section 2.4/4.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import pack_int4  # noqa: E402
from gemma4_kernels import (  # noqa: E402
    gemma4_moe_expert_prefill_reference,
    gemma4_moe_route_prefill_reference,
    make_gemma4_moe_combine,
    make_gemma4_moe_expert_prefill,
    make_gemma4_moe_route_prefill,
)

N, H, I, E, TOPK, G = 8, 64, 64, 32, 4, 64
TILES, SHAVES = 2, 2
WORKERS = TILES * SHAVES
REL = 5e-2


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")


def _dequant_gm(w_int, scale_gm, G):
    n, k = w_int.shape
    s = scale_gm.astype(np.float32).T  # [N, K/G]
    return (w_int.astype(np.float32).reshape(n, k // G, G) * s[:, :, None]).reshape(n, k)


@needs_cxx
def test_expert_ffn_and_combine_match_reference():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((N, H), np.float32).astype(np.float16)
    logits = rng.standard_normal((N, E), np.float32).astype(np.float16)
    per_expert_scale = rng.uniform(0.5, 1.5, E).astype(np.float32)

    # --- stage 1: route ---------------------------------------------------------------------------
    route_func = make_gemma4_moe_route_prefill(N=N, E=E, TOPK=TOPK, tiles=TILES, shaves=SHAVES, as_block=False)
    low = _lower(route_func)
    lib = host_sim.build(low.source, "tl_gemma4_moe_route_for_expert_sim")
    route_arrays = dict(
        Logits=logits, PerExpertScale=per_expert_scale.reshape(E, 1),
        Ids=np.zeros((N, TOPK), np.int32), Wts=np.zeros((N, TOPK), np.float32),
        Off=np.zeros((E + 1,), np.int32), Slot=np.zeros((N * TOPK,), np.int32),
    )
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, route_arrays), TILES, SHAVES))

    # --- weights ------------------------------------------------------------------------------------
    KGH, KGI = H // G, I // G
    wg_int = rng.integers(-7, 8, (E, I, H), np.int8)
    wg_s = rng.uniform(0.01, 0.1, (E, KGH, I)).astype(np.float16)
    wu_int = rng.integers(-7, 8, (E, I, H), np.int8)
    wu_s = rng.uniform(0.01, 0.1, (E, KGH, I)).astype(np.float16)
    wd_int = rng.integers(-7, 8, (E, H, I), np.int8)
    wd_s = rng.uniform(0.01, 0.1, (E, KGI, H)).astype(np.float16)

    # --- stage 2: expert FFN ------------------------------------------------------------------------
    expert_func = make_gemma4_moe_expert_prefill(N=N, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=4, tiles=TILES, shaves=SHAVES)
    low2 = _lower(expert_func)
    lib2 = host_sim.build(low2.source, "tl_gemma4_moe_expert_sim")
    expert_arrays = dict(
        X=x,
        WgateQ=np.stack([pack_int4(wg_int[e]) for e in range(E)]), WgateS=wg_s,
        WupQ=np.stack([pack_int4(wu_int[e]) for e in range(E)]), WupS=wu_s,
        WdownQ=np.stack([pack_int4(wd_int[e]) for e in range(E)]), WdownS=wd_s,
        Wts=route_arrays["Wts"], Off=route_arrays["Off"], Slot=route_arrays["Slot"],
        Part=np.zeros((WORKERS, N, H), np.float32),
    )
    host_sim.check_status(host_sim.run(lib2, lowering.order_arrays(low2, expert_arrays), TILES, SHAVES))

    # --- stage 3: combine ----------------------------------------------------------------------------
    combine_func = make_gemma4_moe_combine(N=N, H=H, tiles=TILES, shaves=SHAVES)
    low3 = _lower(combine_func)
    lib3 = host_sim.build(low3.source, "tl_gemma4_moe_combine_sim")
    combine_arrays = dict(Part=expert_arrays["Part"], Out=np.zeros((N, H), np.float32))
    host_sim.check_status(host_sim.run(lib3, lowering.order_arrays(low3, combine_arrays), TILES, SHAVES))

    # --- oracle --------------------------------------------------------------------------------------
    ref_ids, ref_wts = gemma4_moe_route_prefill_reference(logits, per_expert_scale, E, TOPK)
    assert np.array_equal(route_arrays["Ids"], ref_ids)
    wg_f = np.stack([_dequant_gm(wg_int[e], wg_s[e], G) for e in range(E)])
    wu_f = np.stack([_dequant_gm(wu_int[e], wu_s[e], G) for e in range(E)])
    wd_f = np.stack([_dequant_gm(wd_int[e], wd_s[e], G) for e in range(E)])
    ref_out = gemma4_moe_expert_prefill_reference(x, ref_ids, ref_wts, wg_f, wu_f, wd_f)

    out = combine_arrays["Out"]
    rel = np.abs(out - ref_out).max() / (np.abs(ref_out).max() + 1e-6)
    assert rel < REL, (rel, out, ref_out)


@needs_cxx
def test_expert_ffn_weight_tiling_matches_untiled():
    """`BNW_gu`/`BNW_d` tile the expert FFN's weight-row dequant+gemm into chunks (required at real model
    scale: an untiled (I, H) ~ (704, 2816) fp16 tile is ~3.96 MiB, far over one SHAVE's CMX share). Here
    I/H are small enough that BOTH the untiled (BNW=None) and tiled (BNW < I/H, 2+ chunks) forms fit CMX,
    so this checks they agree with EACH OTHER (not just the numpy oracle) -- the real evidence that tiling
    the gate|up|down computation into column chunks changes nothing except the schedule."""
    H2, I2 = 128, 128
    rng = np.random.default_rng(3)
    x = rng.standard_normal((N, H2), np.float32).astype(np.float16)
    E2, TOPK2 = 32, 2
    logits = rng.standard_normal((N, E2), np.float32).astype(np.float16)
    per_expert_scale = rng.uniform(0.5, 1.5, E2).astype(np.float32)

    route_func = make_gemma4_moe_route_prefill(N=N, E=E2, TOPK=TOPK2, tiles=TILES, shaves=SHAVES, as_block=False)
    low = _lower(route_func)
    lib = host_sim.build(low.source, "tl_gemma4_moe_route_tiling_sim")
    route_arrays = dict(
        Logits=logits, PerExpertScale=per_expert_scale.reshape(E2, 1),
        Ids=np.zeros((N, TOPK2), np.int32), Wts=np.zeros((N, TOPK2), np.float32),
        Off=np.zeros((E2 + 1,), np.int32), Slot=np.zeros((N * TOPK2,), np.int32),
    )
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, route_arrays), TILES, SHAVES))

    KGH, KGI = H2 // G, I2 // G
    wg_int = rng.integers(-7, 8, (E2, I2, H2), np.int8)
    wg_s = rng.uniform(0.01, 0.1, (E2, KGH, I2)).astype(np.float16)
    wu_int = rng.integers(-7, 8, (E2, I2, H2), np.int8)
    wu_s = rng.uniform(0.01, 0.1, (E2, KGH, I2)).astype(np.float16)
    wd_int = rng.integers(-7, 8, (E2, H2, I2), np.int8)
    wd_s = rng.uniform(0.01, 0.1, (E2, KGI, H2)).astype(np.float16)
    base_arrays = dict(
        X=x,
        WgateQ=np.stack([pack_int4(wg_int[e]) for e in range(E2)]), WgateS=wg_s,
        WupQ=np.stack([pack_int4(wu_int[e]) for e in range(E2)]), WupS=wu_s,
        WdownQ=np.stack([pack_int4(wd_int[e]) for e in range(E2)]), WdownS=wd_s,
        Wts=route_arrays["Wts"], Off=route_arrays["Off"], Slot=route_arrays["Slot"],
    )

    def run(BNW_gu, BNW_d):
        func = make_gemma4_moe_expert_prefill(N=N, H=H2, I=I2, E=E2, TOPK=TOPK2, G=G, MB=4,
                                              BNW_gu=BNW_gu, BNW_d=BNW_d, tiles=TILES, shaves=SHAVES)
        low2 = _lower(func)
        lib2 = host_sim.build(low2.source, f"tl_gemma4_moe_expert_tiling_{BNW_gu}_{BNW_d}_sim")
        arrays = dict(base_arrays, Part=np.zeros((WORKERS, N, H2), np.float32))
        host_sim.check_status(host_sim.run(lib2, lowering.order_arrays(low2, arrays), TILES, SHAVES))
        combine_func = make_gemma4_moe_combine(N=N, H=H2, tiles=TILES, shaves=SHAVES)
        low3 = _lower(combine_func)
        lib3 = host_sim.build(low3.source, f"tl_gemma4_moe_combine_tiling_{BNW_gu}_{BNW_d}_sim")
        combine_arrays = dict(Part=arrays["Part"], Out=np.zeros((N, H2), np.float32))
        host_sim.check_status(host_sim.run(lib3, lowering.order_arrays(low3, combine_arrays), TILES, SHAVES))
        return combine_arrays["Out"]

    out_untiled = run(None, None)
    out_tiled = run(64, 64)  # 2 chunks each
    rel = np.abs(out_tiled - out_untiled).max() / (np.abs(out_untiled).max() + 1e-6)
    assert rel < 1e-2, (rel, out_tiled, out_untiled)

