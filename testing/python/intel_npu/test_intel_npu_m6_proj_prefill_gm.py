"""The prefill projection reading GROUP-MAJOR scales (`make_proj_prefill(scales_gm=True)`).

Why the layout matters: the DPU's int4 path (`dpu_matmul_i4` with `sStride > 0`) reads scales group-major
`[K/G, N]` in place, and prefill's SHAVE dequant read them row-major `[N, K/G]`. Two layouts means two uploads of
the same weights, and decode alone moves 17.91 GB per prompt -- so the model would not fit the box. This test
pins the one layout both paths can share: the kernel reads `[K/G, M]`, and its output must equal the row-major
kernel's on the same weights, to the bit.

The dequant behind it is `tl_npu_dequant_i4_gm`, ported from npu_dsl's `dequant_i4_gm`, which keeps the VPUX
assembly by transposing a chunk's scales into a stack buffer first. The device's fast path needs
`G == 128 and K % 128 == 0 and K >= 256`; this test runs in the host simulation, which takes the scalar form, so
what it checks is the CONTRACT -- the indexing `s[(k / G) * M + n]` and the kernel's 2-D strided read of it. The
assembly path is the device runner's to confirm.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401
import tvm
import tilelang.intel_npu.language as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import make_proj_prefill, proj_prefill_reference  # noqa: E402

TILES, SHAVES = 2, 2
NP, M, K, G = 32, 128, 256, 128
BN, MB = 32, 32
KG = K // G
REL = 5e-2


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


needs_cxx = pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((NP, K), np.float32) * 0.5).astype(np.float16)
    w_int = rng.integers(-8, 8, (M, K), dtype=np.int8)
    # the packed form both kernels read: two values per byte, low nibble first
    wq = ((w_int[:, 0::2] & 0xF) | ((w_int[:, 1::2] & 0xF) << 4)).astype(np.uint8)
    s_row = (rng.standard_normal((M, KG), np.float32) * 0.1 + 0.2).astype(np.float16)
    return x, w_int, wq, s_row


def _sim(scales_gm, x, wq, s):
    low = lowering.lower(make_proj_prefill(NP=NP, M=M, K=K, BN=BN, MB=MB, G=G, tiles=TILES, shaves=SHAVES, scales_gm=scales_gm))
    lib = host_sim.build(low.source, f"proj_prefill_{'gm' if scales_gm else 'rm'}_{NP}_{M}_{K}")
    out = np.zeros((NP, M), np.float16)
    arrays = {"X": x, "Wq": wq, "S": s, "Yout": out}
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), tiles=TILES, shaves=SHAVES))
    return out


@needs_cxx
def test_matches_the_reference():
    x, w_int, wq, s_row = _problem()
    got = _sim(True, x, wq, np.ascontiguousarray(s_row.T))  # [KG, M]
    ref = proj_prefill_reference(x, w_int, s_row, G)
    d = np.abs(got.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-9)
    assert d < REL, f"rel {d}"


@needs_cxx
def test_is_bit_identical_to_the_row_major_kernel():
    """The two layouts are the same numbers in a different order, so the two kernels must agree exactly -- a
    tolerance here would hide a transposed read that happened to be close."""
    x, _, wq, s_row = _problem(seed=1)
    rm = _sim(False, x, wq, s_row)
    gm = _sim(True, x, wq, np.ascontiguousarray(s_row.T))
    assert np.array_equal(rm, gm)


@needs_cxx
@pytest.mark.parametrize("scales_gm", [False, True])
@pytest.mark.parametrize("weight_rows", [32, 64])
def test_tile_shared_weights_match_private_weights(scales_gm, weight_rows):
    x, _, wq, s_row = _problem(seed=4)
    scales = np.ascontiguousarray(s_row.T) if scales_gm else s_row
    expected = _sim(scales_gm, x, wq, scales)
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.multicast": True}):
        low = lowering.lower(
            make_proj_prefill(
                NP=NP, M=M, K=K, BN=weight_rows, MB=16, G=G, tiles=TILES, shaves=SHAVES, scales_gm=scales_gm, shared_weights=True
            )
        )
    assert "ctx.dma_load_multicast(" in low.source
    assert f"ctx.alloc_shared<half>({weight_rows * K})" in low.source
    lib = host_sim.build(low.source, "shared_proj_prefill")
    output = np.zeros_like(expected)
    arrays = lowering.order_arrays(low, {"X": x, "Wq": wq, "S": scales, "Yout": output})
    host_sim.check_status(host_sim.run(lib, arrays, tiles=TILES, shaves=SHAVES))
    np.testing.assert_array_equal(output, expected)


def test_shared_weights_fit_the_reference_projection_geometry():
    kwargs = dict(NP=192, M=12288, K=2048, BN=128, MB=48, G=128, tiles=3, shaves=2, scales_gm=True)
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.multicast": True}):
        low = lowering.lower(make_proj_prefill(**kwargs, shared_weights=True))
        assert "ctx.alloc_shared<half>(262144)" in low.source
        assert "ctx.dma_load_multicast(" in low.source
        with pytest.raises(Exception, match="CMX budget exceeded"):
            lowering.lower(make_proj_prefill(**kwargs))


def test_shared_weight_overlap_stays_after_weight_preparation():
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.multicast_overlap": True}):
        low = lowering.lower(make_proj_prefill(NP=192, M=12288, K=2048, BN=128, MB=48, scales_gm=True, shared_weights=True))
    assert "ctx.ring_begin(" in low.source
    assert low.source.index("tl_npu_dequant_i4_gm(") < low.source.index("ctx.ring_begin(")
    assert "ownership-preserving" in low.source
    assert "hoisted over it" not in low.source


@needs_cxx
@pytest.mark.parametrize("block_k", [128, 256])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("tiles", [1, 3])
def test_k_blocked_projection_preserves_output_with_incomplete_tile_round(block_k, overlap, tiles):
    x, weights, packed, scales = _problem(seed=17)
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.multicast": True,
                                           "tl.intel_npu.multicast_overlap": overlap}):
        low = lowering.lower(make_proj_prefill(NP=NP, M=M, K=K, BN=32, MB=8, G=G,
                                              tiles=tiles, shaves=2, scales_gm=True, shared_weights=True, block_k=block_k))
    expected_ring = overlap and tiles > 1
    assert ("ctx.ring_begin(" in low.source) == expected_ring
    output = np.full((NP, M), np.nan, np.float16)
    arrays = lowering.order_arrays(low, {"X": x, "Wq": packed, "S": np.ascontiguousarray(scales.T), "Yout": output})
    if expected_ring:
        return
    lib = host_sim.build(low.source, "blocked_projection_tail")
    host_sim.check_status(host_sim.run(lib, arrays, tiles=tiles, shaves=2))
    expected = proj_prefill_reference(x, weights, scales, G)
    assert np.isfinite(output).all()
    assert np.abs(output.astype(np.float32) - expected).max() / np.abs(expected).max() < 2e-3


@needs_cxx
def test_projection_weight_and_output_row_offsets():
    offset, prefix = 32, 8
    phase, _ = make_proj_prefill(NP=NP, M=M, K=K, BN=BN, MB=MB, G=G, tiles=TILES,
                                 shaves=SHAVES, scales_gm=True, as_block=True,
                                 weight_row_offset=offset, output_row_base=True)

    @T.prim_func
    def shifted(X: T.Tensor((NP, K), T.float16), Wq: T.Tensor((M + offset, K // 2), T.uint8),
                S: T.Tensor((KG, M + offset), T.float16), Yout: T.Tensor((NP + prefix, M), T.float16)):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            phase(X, Wq, S, Yout, tile, T.get_thread_binding(), prefix)

    x, _, wq, scales = _problem(seed=9)
    expected = _sim(True, x, wq, np.ascontiguousarray(scales.T))
    weights = np.pad(wq, ((offset, 0), (0, 0)))
    padded_scales = np.ascontiguousarray(np.pad(scales.T, ((0, 0), (offset, 0)), constant_values=3))
    low = lowering.lower(shifted)
    output = np.full((NP + prefix, M), np.nan, np.float16)
    arrays = lowering.order_arrays(low, {"X": x, "Wq": weights, "S": padded_scales, "Yout": output})
    lib = host_sim.build(low.source, "shifted_projection")
    host_sim.check_status(host_sim.run(lib, arrays, tiles=TILES, shaves=SHAVES))
    assert np.isnan(output[:prefix]).all()
    np.testing.assert_array_equal(output[prefix:], expected)


@needs_cxx
def test_a_transposed_scale_table_is_rejected():
    """The negative control the pair above needs: feeding the group-major kernel the ROW-major table (reshaped to
    the same extent, so nothing complains about sizes) must produce a different answer. Without this, a kernel
    that ignored `scales_gm` would pass both tests above."""
    x, w_int, wq, s_row = _problem(seed=2)
    assert s_row.shape[0] != s_row.shape[1], "the two layouts must not be the same bytes"
    wrong = _sim(True, x, wq, np.ascontiguousarray(s_row).reshape(KG, M))
    ref = proj_prefill_reference(x, w_int, s_row, G)
    d = np.abs(wrong.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-9)
    assert d > REL, f"the wrong scale layout still matched the oracle (rel {d}): the kernel is not reading it"
