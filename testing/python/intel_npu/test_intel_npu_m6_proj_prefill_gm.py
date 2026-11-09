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
    low = lowering.lower(make_proj_prefill(NP=NP, M=M, K=K, BN=BN, MB=MB, G=G, tiles=TILES, shaves=SHAVES,
                                           scales_gm=scales_gm))
    lib = host_sim.build(low.source, f"proj_prefill_{'gm' if scales_gm else 'rm'}_{NP}_{M}_{K}")
    out = np.zeros((NP, M), np.float16)
    arrays = {"X": x, "Wq": wq, "S": s, "Yout": out}
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), tiles=TILES, shaves=SHAVES))
    return out


@needs_cxx
def test_matches_the_reference():
    x, w_int, wq, s_row = _problem()
    got = _sim(True, x, wq, np.ascontiguousarray(s_row.T))      # [KG, M]
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
