"""P2b of prefill (plan section 4, M6): the grouped expert FFN over P2a's grouping.

Each worker owns whole experts, so an expert's weights are read once per launch instead of once per
(token, choice) pair. The projections take `make_w4a16_flow`'s path -- SHAVE dequant into fp16, then an
fp16 DPU gemm -- because the merged-FC scratch decode uses grows with M and is sized for M = 1, while the
dequant's cost does not grow with M and so amortises over a chunk of tokens.

The oracle walks the SAME grouping the kernel walks, so a wrong slot table shows up as a wrong result,
and the tests pin the property that makes prefill worth doing:

  * the result matches the oracle;
  * a perturbed group of ONE expert moves only the tokens routed to that expert, which is what "grouped by
    expert" means and what a per-token loop would not prove;
  * a token whose weight is zeroed drops out, so the weights are really applied per (token, choice).
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
from kernels import (gm_scales, make_moe_expert_prefill, moe_expert_prefill_reference,  # noqa: E402
                     moe_route_prefill_reference, pack_int4)

N, H, I, E, TOPK, G = 12, 128, 128, 8, 2, 128
MB, BNG, BND, TILES, SHAVES = 4, 64, 64, 3, 2
WORKERS = TILES * SHAVES
SLOTS = N * (TOPK + 1)
REL = 4e-2   # int4 weights, fp16 gemms and an fp32 accumulate over TOPK + 1 experts


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
    wr = rng.standard_normal((E + 1, H), np.float32).astype(np.float16) * np.float16(0.1)
    gu_int = rng.integers(-8, 8, (E + 1, 2 * I, H), dtype=np.int8)
    gu_s = (rng.standard_normal((E + 1, 2 * I, H // G), np.float32) * 0.02).astype(np.float16)
    d_int = rng.integers(-8, 8, (E + 1, H, I), dtype=np.int8)
    d_s = (rng.standard_normal((E + 1, H, I // G), np.float32) * 0.02).astype(np.float16)
    # the grouping comes from the routing oracle: P2b is tested against the table P2a produces
    r = moe_route_prefill_reference(x, np.pad(wr, ((0, 0), (0, 0))), E=E, TOPK=TOPK)
    return dict(x=x, gu_int=gu_int, gu_s=gu_s, d_int=d_int, d_s=d_s, wts=r["wts"], off=r["off"], slot=r["slot"])


def _sim(p):
    low = _lower(make_moe_expert_prefill(N=N, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=MB, BNG=BNG, BND=BND,
                                         tiles=TILES, shaves=SHAVES))
    lib = host_sim.build(low.source, "tl_moe_expert_sim")
    out = np.zeros((N, H), np.float32)
    arrays = dict(X=p["x"],
                  WguQ=pack_int4(p["gu_int"].reshape(-1, H)), WguS=gm_scales(p["gu_s"].reshape(-1, H // G)),
                  WdQ=pack_int4(p["d_int"].reshape(-1, I)), WdS=gm_scales(p["d_s"].reshape(-1, I // G)),
                  Wts=p["wts"].astype(np.float32), Off=p["off"].astype(np.int32),
                  Slot=p["slot"].astype(np.int32), Part=np.zeros((TILES, N, H), np.float32), Out=out)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    return out


def test_lowers_with_a_dequant_per_weight_chunk_and_no_merged_fc():
    src = _lower(make_moe_expert_prefill(N=N, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=MB, BNG=BNG, BND=BND,
                                         tiles=TILES, shaves=SHAVES)).source
    assert "tl_npu_dequant_i4_gm(" in src          # a whole chunk per call, from the SHARED group-major scales
    assert "tl_npu_dequant_i4_block(" not in src   # ... never the row-major form, which needs a second upload
    assert "tl_npu_dequant_i4_row(" not in src     # ... never a call per row
    assert "dpu_matmul_i4g" not in src             # ... and does not use decode's merged-FC
    assert "ctx.sync_grid();" in src               # the combine sits behind a grid barrier
    assert "tl_npu_dequant_i8_row" not in src      # control: the needle still discriminates


@needs_cxx
def test_matches_the_oracle():
    p = _problem()
    ref = moe_expert_prefill_reference(p["x"], p["gu_int"], p["gu_s"], p["d_int"], p["d_s"],
                                       p["wts"], p["off"], p["slot"], TOPK=TOPK, G=G)
    got = _sim(p)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < REL, f"rel {rel:.3e}"


@needs_cxx
def test_perturbing_one_expert_moves_only_its_tokens():
    """What "grouped by expert" means: a per-token loop would pass the oracle test too, this one it would not."""
    p = _problem()
    base = _sim(p)
    victim = int(np.argmax(np.diff(p["off"][:E + 1])))        # the routed expert with the most tokens
    touched = {int(p["slot"][s]) // (TOPK + 1) for s in range(int(p["off"][victim]), int(p["off"][victim + 1]))}
    assert 0 < len(touched) < N, "the probe needs an expert that some but not all tokens use"
    p["d_int"][victim] = np.clip(p["d_int"][victim].astype(np.int16) + 3, -8, 7).astype(np.int8)
    got = _sim(p)
    moved = {t for t in range(N) if np.abs(got[t] - base[t]).max() > 1e-3}
    assert moved == touched, f"expert {victim}: tokens {sorted(moved)} moved, expected {sorted(touched)}"


@needs_cxx
def test_a_zeroed_choice_weight_drops_that_contribution():
    p = _problem()
    base = _sim(p)
    t0, k0 = int(p["slot"][0]) // (TOPK + 1), int(p["slot"][0]) % (TOPK + 1)
    p["wts"] = p["wts"].copy()
    p["wts"][t0, k0] = 0.0
    got = _sim(p)
    assert np.abs(got[t0] - base[t0]).max() > 1e-3, "zeroing a choice's weight changed nothing"
    others = [t for t in range(N) if t != t0]
    assert np.abs(got[others] - base[others]).max() < 1e-3, "zeroing one token's weight moved another token"
