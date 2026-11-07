"""M1 (plan section 4, model target section 9): one flash-attention phase through the generic ring.

The source is `kernels.make_flash_attention_phase`: Q resident (SEGMENTED), K and Vt streamed
through one T.Pipelined loop with SHAVE statements (online softmax) between the two gemms. The
device golden is `device/run_flash_device.py` (PASS 2026-11-07, device/evidence/INDEX.md); here:

  * the pipelined phase lowers to TWO rings over the same loop, the strided Vt chunk as a 2-D
    multicast fill, the general body (gemm -> wait -> next fill, author order kept, no
    submit-ahead because SHAVE statements sit between the gemms) and distinct barrier ids;
  * the serial phase (pipelined=False) matches numpy in the host sim at the device's tolerance
    (the host sim does not run rings);
  * negative: the serial source lowers to no ring at all, and a perturbed K fails the golden.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import make_flash_attention_phase  # noqa: E402

NUMQ, NUMKV, D, BQ, BKV, TILES = 96, 1024, 64, 32, 128, 3


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


def _reference(q, k, vt):
    s = q.astype(np.float32) @ k.astype(np.float32).T / np.sqrt(D)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p @ vt.astype(np.float32).T


def test_pipelined_phase_lowers_to_two_rings_over_one_loop():
    src = _lower(make_flash_attention_phase(NUMQ, NUMKV, D=D, BQ=BQ, BKV=BKV, tiles=TILES)).source
    assert "npu::Ctx::Ring ring = ctx.ring_begin((&(k_cmx[0])), 16384, 2, 8, 8, 12);" in src
    assert "npu::Ctx::Ring ring_1 = ctx.ring_begin((&(v_cmx[0])), 16384, 2, 8, 16, 20);" in src
    # K chunk: 128 rows of 128 B, contiguous; Vt chunk: 64 rows of 256 B, 2048 B apart (2-D fill)
    assert "ctx.ring_fill(ring, 0, (&(K[(shave * 8192)])), 128, 128, 128);" in src
    assert "ctx.ring_fill(ring_1, 0, (&(Vt[(shave * 128)])), 256, 64, 2048);" in src
    # general body: each gemm reads its ring's slot, waits, then the issuer fills use u+1
    assert re.search(r"ring_dpu_submit\(ring, u, \(&\(s_cmx\[0\]\)\), \(&\(q_cmx\[0\]\)\), ctx\.ring_slot<half>\(ring\), 32, 128, 64", src)
    assert re.search(r"ring_dpu_submit\(ring_1, u, \(&\(pv_cmx\[0\]\)\), \(&\(s_cmx\[0\]\)\), ctx\.ring_slot<half>\(ring_1\), 32, 64, 128", src)
    assert "ctx.ring_dpu_wait(ring, u);\n  ctx.ring_fill(ring, (u + 1)," in src.replace("    ", "  ")
    assert src.count("ctx.ring_end(") == 2
    assert "READY 8..11, FREE 12..15; ring: READY 16..19, FREE 20..23" in src
    assert "2 gemms with SHAVE statements between, no submit-ahead" in src
    assert "q_cmx: SEGMENTED resident, 32 rows split 16 per SHAVE, tile sync after" in src
    assert "ctx.dpu_matmul(" not in src


def test_serial_phase_has_no_ring():
    src = _lower(make_flash_attention_phase(NUMQ, NUMKV, D=D, BQ=BQ, BKV=BKV, tiles=TILES, pipelined=False)).source
    assert "ring_begin" not in src and "ctx.dpu_matmul(" in src


@needs_cxx
def test_serial_phase_matches_numpy_in_host_sim():
    low = _lower(make_flash_attention_phase(NUMQ, 256, D=D, BQ=BQ, BKV=BKV, tiles=TILES, pipelined=False))
    lib = host_sim.build(low.source, "tl_flash_serial_sim_kv256")
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((NUMQ, D)) * 0.5).astype(np.float16)
    k = (rng.standard_normal((256, D)) * 0.5).astype(np.float16)
    vt = (rng.standard_normal((D, 256)) * 0.5).astype(np.float16)
    o = np.zeros((NUMQ, D), np.float16)
    status = host_sim.run(lib, lowering.order_arrays(low, {"Q": q, "K": k, "Vt": vt, "O": o}), TILES, 2)
    host_sim.check_status(status)
    ref = _reference(q, k, vt)
    rel = np.abs(o.astype(np.float32) - ref).max() / np.abs(ref).max()
    assert rel < 2e-3, rel
    # negative control: a perturbed K is not the same attention
    k2 = k.copy()
    k2[3, 5] += np.float16(6.0)
    o2 = np.zeros((NUMQ, D), np.float16)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, {"Q": q, "K": k2, "Vt": vt, "O": o2}), TILES, 2))
    assert np.abs(o2.astype(np.float32) - ref).max() / np.abs(ref).max() > 2e-3
