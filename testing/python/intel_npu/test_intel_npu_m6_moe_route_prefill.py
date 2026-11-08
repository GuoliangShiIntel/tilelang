"""P2a of prefill (plan section 4, M6): MoE routing and grouping by expert for N prompt tokens.

Decode reads every chosen expert's weights once per token; prefill must read them once per expert, so
`kernels.make_moe_route_prefill` sorts the (token, choice) pairs into expert order. The grouping is the
whole point of the phase, so the tests check it as a permutation and not only element by element:

  * the ids, weights, counts, offsets and slot table match `moe_route_prefill_reference`;
  * `Off` is the exclusive prefix sum of `Cnt` and `Off[-1]` is every pair, so no pair is lost or doubled;
  * every slot of expert e really holds a pair whose expert is e -- the property the FFN will rely on;
  * the slot table is a permutation of all N x (TOPK + 1) pairs;
  * negative: a perturbed router row changes the routing, so the comparison can fail.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401  (loads the dev build before tvm)
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import make_moe_route_prefill, moe_route_prefill_reference  # noqa: E402

N, H, E, TOPK, BM, BNR = 32, 128, 64, 4, 16, 16
TILES, SHAVES = 3, 2
RR = -(-(E + 1) // (TILES * SHAVES * BNR)) * (TILES * SHAVES * BNR)
SLOTS = N * (TOPK + 1)


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")


def _problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, H), np.float32).astype(np.float16)
    wr = np.zeros((RR, H), np.float16)
    wr[:E + 1] = rng.standard_normal((E + 1, H), np.float32).astype(np.float16) * np.float16(0.1)
    return x, wr


def _sim(x, wr):
    low = _lower(make_moe_route_prefill(N=N, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=TILES, shaves=SHAVES))
    lib = host_sim.build(low.source, "tl_moe_route_sim")
    out = dict(Logits=np.zeros((N, RR), np.float16), Ids=np.zeros((N, TOPK), np.int32),
               Wts=np.zeros((N, 16), np.float32), Cnt=np.zeros(E + 1, np.int32),
               Off=np.zeros(E + 2, np.int32), Slot=np.zeros(SLOTS, np.int32))
    arrays = dict(X=x, Wr=wr, **out)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    return out


def test_lowers_with_a_router_workload_inside_the_dpu_limit():
    src = _lower(make_moe_route_prefill(N=N, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=TILES, shaves=SHAVES)).source
    assert "kernel_entry" in src
    assert src.count("ctx.sync_grid();") == 2          # one after the router, one after the top-k
    # the top-k's row max is a lane reduction, not a scalar scan over E. Only the call is asserted: whether the
    # accumulator is a register or a CMX buffer is the register-promotion pass's choice, not this phase's decision.
    assert "tl_npu_vmax(" in src
    assert "tl_npu_vmax_not_a_real_symbol(" not in src   # control: the needle still discriminates
    assert BM * (RR // TILES) <= 8192


def test_an_oversized_router_block_is_refused():
    """The DPU output-point limit is a refusal, not a silent slowdown."""
    with pytest.raises(AssertionError, match="8192 output points"):
        make_moe_route_prefill(N=N, H=H, E=E, TOPK=TOPK, BM=N, BNR=256, tiles=TILES, shaves=SHAVES)


@needs_cxx
def test_routing_and_grouping_match_the_oracle():
    x, wr = _problem()
    ref = moe_route_prefill_reference(x, wr, E=E, TOPK=TOPK)
    got = _sim(x, wr)
    assert np.array_equal(got["Ids"], ref["ids"]), "the chosen experts differ"
    assert np.abs(got["Wts"][:, :TOPK + 1] - ref["wts"][:, :TOPK + 1]).max() < 2e-3
    assert np.array_equal(got["Cnt"], ref["cnt"])
    assert np.array_equal(got["Off"], ref["off"])
    assert np.array_equal(got["Slot"], ref["slot"])


@needs_cxx
def test_the_grouping_is_a_sound_permutation():
    """Checked as a structure, not value by value: this is what the expert FFN will rely on."""
    x, wr = _problem()
    got = _sim(x, wr)
    cnt, off, slot = got["Cnt"], got["Off"], got["Slot"]
    assert off[0] == 0 and off[E + 1] == SLOTS, "the offsets do not cover every (token, choice) pair"
    assert np.array_equal(off[1:], np.cumsum(cnt)), "Off is not the exclusive prefix sum of Cnt"
    assert cnt[E] == N, "the shared expert must take every token"
    assert cnt[:E].sum() == N * TOPK, "the routed pairs do not add up"
    assert np.array_equal(np.sort(slot), np.arange(SLOTS)), "the slot table is not a permutation of the pairs"
    for e in range(E + 1):                              # every slot of a group belongs to that group's expert
        for s in range(off[e], off[e + 1]):
            t, k = int(slot[s]) // (TOPK + 1), int(slot[s]) % (TOPK + 1)
            assert e == (E if k == TOPK else int(got["Ids"][t, k])), f"slot {s} of expert {e} holds a pair of another"


@needs_cxx
def test_a_perturbed_router_row_changes_the_routing():
    x, wr = _problem()
    base = _sim(x, wr)
    wr2 = wr.copy()
    wr2[3] = wr2[3] + np.float16(4.0)                   # expert 3 now wins for most tokens
    got = _sim(x, wr2)
    assert not np.array_equal(got["Ids"], base["Ids"]), "the routing did not move: the comparison cannot fail"
