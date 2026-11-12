"""Gemma4 V0: dense FFN (gate|up GeGLU with gelu-tanh, down) and the MoE router tail (top-k by raw
logit, softmax-over-chosen, per_expert_scale, NO shared-expert row), isolated from the full layer.
See docs/developer_guide/gemma4_enablement_plan.md sections 2.3/2.4/5.
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
    gemma4_geglu_reference,
    gemma4_moe_route_prefill_reference,
    make_gemma4_dense_ffn_prefill,
    make_gemma4_moe_route_prefill,
)

TILES, SHAVES = 2, 2
REL = 4e-2


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


@needs_cxx
def test_dense_ffn_matches_reference():
    N, H, FI, G = 8, 64, 64, 64
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, H), np.float32).astype(np.float16)
    wg = rng.integers(-7, 8, (FI, H), np.int8)
    sg = rng.uniform(0.01, 0.1, (H // G, FI)).astype(np.float16)
    wu = rng.integers(-7, 8, (FI, H), np.int8)
    su = rng.uniform(0.01, 0.1, (H // G, FI)).astype(np.float16)
    wd = rng.integers(-7, 8, (H, FI), np.int8)
    sd = rng.uniform(0.01, 0.1, (FI // G, H), np.int8.__class__) if False else rng.uniform(0.01, 0.1, (FI // G, H)).astype(np.float16)

    def dequant(w_int, scale, G):
        # scale is group-major [K/G, N]; w_int is [N, K]
        n, k = w_int.shape
        s = scale.astype(np.float32).T  # [N, K/G]
        return (w_int.astype(np.float32).reshape(n, k // G, G) * s[:, :, None]).reshape(n, k)

    wg_f = dequant(wg, sg, G)
    wu_f = dequant(wu, su, G)
    wd_f = dequant(wd, sd, G)
    gate_ref = x.astype(np.float32) @ wg_f.T
    up_ref = x.astype(np.float32) @ wu_f.T
    act_ref = gemma4_geglu_reference(gate_ref.astype(np.float16), up_ref.astype(np.float16))
    dense_ref = act_ref.astype(np.float32) @ wd_f.T

    wgq = pack_int4(wg)
    wuq = pack_int4(wu)
    wdq = pack_int4(wd)

    func = make_gemma4_dense_ffn_prefill(N=N, H=H, FI=FI, G=G, BN=32, MB=8, tiles=TILES, shaves=SHAVES)
    low = _lower(func)
    lib = host_sim.build(low.source, "tl_gemma4_dense_ffn_sim")
    arrays = dict(
        Xn=x, WgQ=wgq, WgS=sg, WuQ=wuq, WuS=su, WdQ=wdq, WdS=sd,
        Gate=np.zeros((N, FI), np.float16), Up=np.zeros((N, FI), np.float16), Act=np.zeros((N, FI), np.float16),
        Dense=np.zeros((N, H), np.float16),
    )
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    out = arrays["Dense"].astype(np.float32)
    rel = np.abs(out - dense_ref).max() / (np.abs(dense_ref).max() + 1e-6)
    assert rel < REL, rel


@needs_cxx
def test_moe_route_matches_reference_no_shared_expert_row():
    N, E, TOPK = 8, 32, 4
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((N, E), np.float32).astype(np.float16)
    per_expert_scale = rng.uniform(0.5, 1.5, E).astype(np.float32)

    func, sizes = make_gemma4_moe_route_prefill(N=N, E=E, TOPK=TOPK, tiles=TILES, shaves=SHAVES, as_block=False), None
    low = _lower(func)
    lib = host_sim.build(low.source, "tl_gemma4_moe_route_sim")
    arrays = dict(
        Logits=logits, PerExpertScale=per_expert_scale.reshape(E, 1),
        Ids=np.zeros((N, TOPK), np.int32), Wts=np.zeros((N, TOPK), np.float32),
        Off=np.zeros((E + 1,), np.int32), Slot=np.zeros((N * TOPK,), np.int32),
    )
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    ref_ids, ref_wts = gemma4_moe_route_prefill_reference(logits, per_expert_scale, E, TOPK)
    assert np.array_equal(arrays["Ids"], ref_ids), (arrays["Ids"], ref_ids)
    rel = np.abs(arrays["Wts"] - ref_wts).max() / (np.abs(ref_wts).max() + 1e-6)
    assert rel < REL, rel
    # grouping check: Off is a valid exclusive prefix sum of the per-expert counts implied by Ids
    cnt = np.zeros(E, np.int64)
    for row in ref_ids:
        for e in row:
            cnt[e] += 1
    off_ref = np.concatenate([[0], np.cumsum(cnt)])
    assert np.array_equal(arrays["Off"], off_ref), (arrays["Off"], off_ref)
