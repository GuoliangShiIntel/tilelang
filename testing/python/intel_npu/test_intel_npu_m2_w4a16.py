"""M2 (plan section 4 / 9): W4A16 projections -- int4 g128 weights dequantized on the SHAVE, fp16 on the DPU.

`kernels.make_w4a16_flow` in its two role assignments; the device goldens are `device/run_w4a16_device.py`
(PASS 2026-11-07, device/evidence/INDEX.md). Here, hardware-free:

  * the serial form matches numpy in the host sim for both role assignments, with a perturbed-nibble negative;
  * the pipelined form lowers the streamed slots the SHAVE reads (no gemm reads them) to acquire / release
    around the dequant, and the N-split form to SEGMENTED streams (every tile fills its own slot) with the
    ring's multicast flag off; the M-split form keeps multicast rings;
  * a streamed buffer nothing reads is refused by name.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang
import tilelang.intel_npu.language as T
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import make_w4a16_flow, pack_int4  # noqa: E402

G, TILES = 128, 3


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


def test_pipelined_m_split_streams_weights_through_multicast_rings_the_shave_reads():
    src = _lower(make_w4a16_flow(96, 2048, 2048, BN=64, G=G, tiles=TILES, split="m")).source
    assert "wq_cmx: DUPLICATED (streamed, one multicast per chunk)" in src
    assert "x_cmx: SEGMENTED resident" in src
    assert "ctx.ring_begin((&(wq_cmx[0])), 65536, 2, 32, 8, 12, 0, 1);" in src     # port 0, multicast
    assert "ctx.ring_begin((&(s_cmx[0])), 2048, 2, 32, 16, 20, 1, 1);" in src     # port 1, multicast
    assert "2 slot(s) read by the SHAVE" in src
    # acquire both slots before the dequant, release each after it and refill; the gemm reads the private fp16 buffer
    assert re.search(r"ctx\.ring_slot_acquire\(ring, u\);\s*ctx\.ring_slot_acquire\(ring_1, u\);", src)
    assert re.search(r"tl_npu_dequant_i4_row\(.*\n.*\}\s*ctx\.ring_slot_release\(ring, u\);\s*ctx\.ring_fill\(ring, \(u \+ 1\)", src)
    assert "ctx.dpu_matmul((&(y_cmx[0])), (&(x_cmx[0])), (&(w_cmx[0])), 32, 64, 2048" in src
    assert "ring_dpu_submit" not in src
    # the dequant reads this SHAVE's slot of the use
    assert "tl_npu_dequant_i4_row((&(w_cmx[(r * 2048)])), (&(wq_cmx[((tx * 65536) + (r * 1024))])), (&(s_cmx[((tx * 1024) + (r * 16))])), 2048, 128)" in src


def test_pipelined_n_split_streams_each_tiles_own_weights():
    src = _lower(make_w4a16_flow(96, 1536, 2048, BN=64, G=G, tiles=TILES, split="n")).source
    assert "wq_cmx: SEGMENTED (streamed, each tile fills its own slot)" in src
    assert "x_cmx: tile-invariant resident" in src
    assert "ctx.ring_begin((&(wq_cmx[0])), 65536, 2, 8, 8, 12, 0, 0);" in src      # multicast off
    assert "ctx.ring_fill(ring, 0, (&(Wq[((bx * 524288) + (tx * 65536))])), 1024, 64, 1024);" in src


def test_streamed_buffer_nothing_reads_is_refused():
    @T.prim_func
    def bad(X: T.Tensor((96, 512), T.float16), W: T.Tensor((256, 512), T.float16), Y: T.Tensor((96, 256), T.float16)):
        with T.Kernel(TILES, threads=2) as tile:
            x_cmx = T.alloc_shared((32, 512), T.float16)
            w_cmx = T.alloc_shared((64, 512), T.float16)
            w2 = T.alloc_local((64, 512), T.float16)
            y_cmx = T.alloc_local((32, 64), T.float16)
            T.copy(X[tile * 32, 0], x_cmx)
            for n in T.Pipelined(4, num_stages=2):
                T.copy(W[n * 64, 0], w_cmx)
                T.gemm(x_cmx, w2, y_cmx, transpose_B=True, clear_accum=True)   # reads w2, never w_cmx
                T.copy(y_cmx, Y[tile * 32, n * 64])

    with pytest.raises(Exception, match="w_cmx is filled but nothing in the loop reads it"):
        _lower(bad)


@needs_cxx
@pytest.mark.parametrize("split", ["m", "n"])
def test_serial_form_matches_numpy_in_host_sim(split):
    M, N, K, BN = 96, 384, 512, 64
    low = _lower(make_w4a16_flow(M, N, K, BN=BN, G=G, tiles=TILES, pipelined=False, split=split))
    lib = host_sim.build(low.source, f"tl_w4a16_serial_sim_{split}")
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    w_int = rng.integers(-8, 8, size=(N, K)).astype(np.int8)
    scales = rng.uniform(0.01, 0.05, size=(N, K // G)).astype(np.float16)
    wq = pack_int4(w_int)
    y = np.zeros((M, N), np.float16)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, {"X": x, "Wq": wq, "S": scales, "Y": y}), TILES, 2))
    w = (w_int.astype(np.float32) * np.repeat(scales.astype(np.float32), G, axis=1)).astype(np.float16).astype(np.float32)
    ref = x.astype(np.float32) @ w.T
    assert np.abs(y.astype(np.float32) - ref).max() / np.abs(ref).max() < 2e-3
    # negative: one nibble changed
    wq2 = wq.copy()
    wq2[3, 5] ^= np.uint8(0x70)
    y2 = np.zeros((M, N), np.float16)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, {"X": x, "Wq": wq2, "S": scales, "Y": y2}), TILES, 2))
    assert np.abs(y2.astype(np.float32) - ref).max() / np.abs(ref).max() > 2e-3
