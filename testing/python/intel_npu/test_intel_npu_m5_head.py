"""M5 decode head (plan section 4 / 9): final norm + lm_head int8 g128 GEMV + greedy argmax in one launch.

`kernels.make_head`: zero-centred RMSNorm on SHAVE 0 of every tile -> the vocabulary rows split over the tiles as
int8 chunk blocks through a local ring, the argmax kept on the fly per SHAVE (lane max per chunk, index scan only
when the chunk beats the running max, padded columns masked) -> partials in DDR -> grid barrier -> worker 0 picks the
lowest index among the partials sharing the max. The device golden is `device/run_head_device.py`. Here, hardware-free:

  * the serial form matches the numpy oracle in the host sim (logits and token), with a perturbed-weight negative
    and a tie fixture (two equal maximal logits in different tiles: the lowest index must win);
  * the Qwen3.6 shape (V 248320 -> 248832 padded, H 2048) lowers to one local int8 ring of 648 chunks, the masked
    last chunks, a strided logits store per SHAVE and one grid barrier.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401  (sets up the tvm path)
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import head_reference, head_sizes, make_head, pack_head  # noqa: E402

G, TILES, SHAVES = 128, 3, 2
SMALL = dict(H=256, V=1000, G=G, tiles=TILES, shaves=SHAVES, BN=64, R=64)


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except RuntimeError:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")


def head_problem(seed, H, V, G, **_):
    rng = np.random.default_rng(seed)
    h = (rng.standard_normal((1, H)) * 2.0).astype(np.float16)
    norm_w = rng.uniform(-0.3, 0.3, size=(1, H)).astype(np.float16)
    wlm_int = rng.integers(-127, 128, size=(V, H), dtype=np.int8)
    wlm_scales = rng.uniform(0.0005, 0.002, size=(V, H // G)).astype(np.float16)
    return dict(h=h, norm_w=norm_w, wlm_int=wlm_int, wlm_scales=wlm_scales)


def head_arrays(p, packed, H, V, tiles=TILES, shaves=SHAVES, BN=128, **_):
    sz = head_sizes(H, V, tiles=tiles, shaves=shaves, BN=BN)
    return dict(Hs=p["h"], NormW=p["norm_w"], WlmQ=packed["WlmQ"], WlmS=packed["WlmS"],
                Logits=np.zeros((tiles, sz["CS"], shaves * BN), np.float16),
                Part=np.zeros((sz["workers"], 32), np.float32), Tok=np.full((1, 8), -1, np.int32))


def test_qwen36_shape_lowers_to_one_local_int8_ring_with_masked_tail():
    src = _lower(make_head()).source
    sz = head_sizes(2048, 248320)
    assert sz["V_pad"] == 248832 and sz["C"] == 648 and sz["CS"] == 324 and sz["ng"] == 16
    # ONE ring, for the weights. The scales are not streamed per chunk: a block of SB chunks is loaded into a
    # per-SHAVE buffer and read in place, so the ring keeps its single begin/end across all 648 chunks.
    assert "ctx.ring_begin((&(w_cmx[0])), 262144, 4, 648, 0, 0, 2, 0, 1);" in src
    assert src.count("ctx.ring_begin(") == 1
    assert "w_cmx: SEGMENTED (streamed, each tile fills its own slot)" in src
    assert "s_cmx" not in src.split("ring_begin(", 1)[0]          # the scales buffer is not a ring's stream
    # the block's scales are ONE CONTIGUOUS burst (width == stride): a column window of a flat [K/G, V_pad] would
    # be K/G short lines at the vocabulary's stride, and that cost the weight stream 58.9 -> 52.1 GB/s
    assert "ctx.dma_load((&(s_cmx[0])), (&(WlmS[((tile * 1327104) + (b * 8192))])), 256, 64, 256, 256);" in src
    assert "ctx.dpu_matmul_i8g(" in src and "= (half)-6.550400e+04f;" in src
    assert "tl_npu_vmax(mx)" in src and "tl_npu_vsum(ls)" in src
    # each SHAVE's 324 chunk rows of 256 B land 512 B apart (the other SHAVE's chunks interleave)
    assert "ctx.dma_store((&(Logits[((tile * 82944) + (shave * 128))])), (&(yl[0])), 256, 324, 512, 256);" in src
    assert src.count("ctx.sync_grid();") == 1
    # the merged-FC scratch is sized by the kernel's largest quantised gemm; left open at the end so that a later
    # argument (the deferred-reduction elements) does not break an assertion that is about the first two
    assert "ctx.reserve_qg(16384, 1024" in src
    assert "ctx.reserve_qg(16384, 2048" not in src   # control: the pattern still discriminates on the second argument


@needs_cxx
def test_small_instance_matches_the_oracle_in_host_sim():
    p = head_problem(0, **SMALL)
    packed = pack_head(p["wlm_int"], p["wlm_scales"], tiles=TILES, shaves=SHAVES, BN=SMALL["BN"])
    low = _lower(make_head(pipelined=False, **SMALL))
    lib = host_sim.build(low.source, "tl_head_sim")
    arrays = head_arrays(p, packed, **SMALL)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    ref, tok_ref = head_reference(p["h"], p["norm_w"], p["wlm_int"], p["wlm_scales"], G=G)
    V = SMALL["V"]
    logits = arrays["Logits"].reshape(-1)[:V].astype(np.float32)
    scale = float(np.abs(ref).max())
    assert np.abs(logits - ref).max() / scale < 2e-2, (np.abs(logits - ref).max(), scale)
    tok = int(arrays["Tok"][0, 0])
    assert tok == int(np.argmax(logits)), (tok, int(np.argmax(logits)))      # the device's own argmax
    top2 = np.sort(ref)[-2:]
    if top2[1] - top2[0] > 2 * np.abs(logits - ref).max():
        assert tok == tok_ref
    # the padded columns were masked, not left at zero
    assert np.all(arrays["Logits"].reshape(-1)[V:] == np.float16(-65504))
    # negative: the winning row's weights perturbed -> its logit moves and the comparison must fail
    p2 = dict(p, wlm_int=p["wlm_int"].copy())
    p2["wlm_int"][tok_ref, :G] = 127
    packed2 = pack_head(p2["wlm_int"], p2["wlm_scales"], tiles=TILES, shaves=SHAVES, BN=SMALL["BN"])
    arrays2 = head_arrays(p2, packed2, **SMALL)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays2), TILES, SHAVES))
    logits2 = arrays2["Logits"].reshape(-1)[:V].astype(np.float32)
    assert np.abs(logits2 - ref).max() / scale > 2e-2


@needs_cxx
def test_tie_picks_the_lowest_index_across_tiles():
    """Two rows equal to the maximum, one in tile 0 and one in tile 2: the token is the lower index."""
    p = head_problem(1, **SMALL)
    sz = head_sizes(**SMALL)
    hi = sz["C"] * SMALL["BN"] * 2 + 5                  # a row in tile 2's third
    lo = 7                                               # a row in tile 0's first chunk
    p["wlm_int"][[lo, hi]] = 127                         # identical rows with a large scale: the maximum, if x sums positive
    p["wlm_scales"][[lo, hi]] = np.float16(0.02)
    ref, _ = head_reference(p["h"], p["norm_w"], p["wlm_int"], p["wlm_scales"], G=G)
    if ref[lo] < ref.max():                              # otherwise the negated rows are
        p["wlm_int"][[lo, hi]] = -127
        ref, _ = head_reference(p["h"], p["norm_w"], p["wlm_int"], p["wlm_scales"], G=G)
    assert ref[lo] == ref[hi] == ref.max()
    packed = pack_head(p["wlm_int"], p["wlm_scales"], tiles=TILES, shaves=SHAVES, BN=SMALL["BN"])
    low = _lower(make_head(pipelined=False, **SMALL))
    lib = host_sim.build(low.source, "tl_head_sim")
    arrays = head_arrays(p, packed, **SMALL)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    logits = arrays["Logits"].reshape(-1)[: SMALL["V"]]
    assert logits[lo] == logits[hi] == logits.max()
    assert int(arrays["Tok"][0, 0]) == lo
