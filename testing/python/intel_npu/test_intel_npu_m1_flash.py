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
    assert "npu::Ctx::Ring ring = ctx.ring_begin((&(k_cmx[0])), 16384, 2, 8, 8, 12, 0, 1, 0);" in src       # stream 0 on port 0
    assert "npu::Ctx::Ring ring_1 = ctx.ring_begin((&(v_cmx[0])), 16384, 2, 8, 16, 20, 1, 1, 0);" in src   # stream 1 on port 1
    # K chunk: 128 rows of 128 B, contiguous; Vt chunk: 64 rows of 256 B, 2048 B apart (2-D fill)
    assert "ctx.ring_fill(ring, 0, (&(K[(shave * 8192)])), 128, 128, 128);" in src
    assert "ctx.ring_fill(ring_1, 0, (&(Vt[(shave * 128)])), 256, 64, 2048);" in src
    # general body: each gemm reads its ring's slot, waits, then the issuer fills use u+1
    assert re.search(r"ring_dpu_submit\(ring, u, \(&\(s_cmx\[0\]\)\), \(&\(q_cmx\[0\]\)\), ctx\.ring_slot<half>\(ring, u\), 32, 128, 64", src)
    assert re.search(r"ring_dpu_submit\(ring_1, u, \(&\(pv_cmx\[0\]\)\), \(&\(s_cmx\[0\]\)\), ctx\.ring_slot<half>\(ring_1, u\), 32, 64, 128", src)
    assert "ctx.ring_dpu_wait(ring, u);\n  ctx.ring_fill(ring, (u + 1)," in src.replace("    ", "  ")
    assert src.count("ctx.ring_end(") == 2
    assert "ring: READY 8..11, FREE 12..15; ring_1: READY 16..19, FREE 20..23" in src
    assert "by reuse" not in src                                             # both live at once: no sharing
    assert "2 gemm(s) with SHAVE statements between, no submit-ahead" in src
    assert "SHAVE s owns 1 slot(s), fills 1 use(s) ahead" in src
    assert "q_cmx: SEGMENTED resident in a loop, 32 rows split 16 per SHAVE, tile sync before and after" in src
    assert "ctx.dpu_matmul(" not in src


def test_ring_depth_two_per_shave_fills_two_uses_ahead():
    # num_stages=4 on 2 SHAVEs: each SHAVE owns 2 slots, the prologue fills uses 0 and 1 of every stream,
    # the body refills use u+2 after use u's gemm, and each ring takes 4 x 4 = 16 ids.
    src = _lower(make_flash_attention_phase(NUMQ, NUMKV, D=D, BQ=BQ, BKV=BKV, tiles=TILES, num_stages=4)).source
    assert "half* __restrict k_cmx = ctx.alloc_shared<half>(32768);" in src        # 4 slots x 128 x 64
    # two slots per SHAVE: the fills alternate ports by use (2 = use & 1, 3 = the opposite phase) so a ring keeps
    # one fill in flight per port under the runtime's one-job-per-port rule
    assert "ctx.ring_begin((&(k_cmx[0])), 16384, 4, 8, 8, 16, 2, 1, 0);" in src
    assert "ctx.ring_begin((&(v_cmx[0])), 16384, 4, 8, 24, 32, 3, 1, 0);" in src
    assert "ctx.ring_fill(ring, 0, (&(K[(shave * 8192)])), 128, 128, 128);" in src
    assert "ctx.ring_fill(ring, 1, (&(K[((shave * 8192) + 16384)])), 128, 128, 128);" in src
    assert "ctx.ring_fill(ring, (u + 2)," in src and "ctx.ring_fill(ring_1, (u + 2)," in src
    assert "READY 8..15, FREE 16..23; ring_1: READY 24..31, FREE 32..39" in src
    assert "SHAVE s owns 2 slot(s), fills 2 use(s) ahead" in src
    with pytest.raises(Exception, match=r"num_stages=3\) on a tile with 2 SHAVEs: the ring depth must be a multiple"):
        _lower(make_flash_attention_phase(NUMQ, NUMKV, D=D, BQ=BQ, BKV=BKV, tiles=TILES, num_stages=3))


def test_serial_phase_has_no_ring():
    src = _lower(make_flash_attention_phase(NUMQ, NUMKV, D=D, BQ=BQ, BKV=BKV, tiles=TILES, pipelined=False)).source
    assert "ring_begin" not in src and "ctx.dpu_matmul(" in src


def test_ring_is_hoisted_over_an_enclosing_loop_its_fills_do_not_read():
    # 10 Q blocks: the K/V fills read only kv, so one ring streams 10 x 8 chunks with one begin/end per kernel;
    # inside the Q-block loop each SHAVE's uses are qb * 4 + u and the fill source wraps modulo the 8 kv chunks.
    src = _lower(make_flash_attention_phase(960, NUMKV, D=D, BQ=BQ, BKV=BKV, tiles=TILES)).source
    assert src.count("ctx.ring_begin(") == 2 and src.count("ctx.ring_end(") == 2
    assert "ctx.ring_begin((&(k_cmx[0])), 16384, 2, 80, 8, 12, 0, 1, 0);" in src
    assert "ring over kv hoisted over qb (10 iterations: one begin/end)" in src
    assert re.search(r"for \(int32_t qb = 0; qb < 10; \+\+qb\) \{", src)
    assert "ctx.ring_dpu_submit(ring, ((qb * 4) + u)," in src
    assert re.search(r"ctx\.ring_fill\(ring, \(\(\(qb \* 4\) \+ u\) \+ 1\), \(&\(K\[\(\(\(\(\(u \* 2\) \+ shave\) \+ 2\) & 7\) \* 8192\)\]\)\)", src)
    # a fill that reads the outer variable is hoisted too (M3): the fill ahead reads the chunk's own iteration,
    # qb = chunk / 8, so the last fills of one Q block already stream the next block's first K/V chunks
    outer_dep = _lower(make_flash_attention_phase(960, NUMKV, D=D, BQ=BQ, BKV=BKV, tiles=TILES, kv_offset_by_qb=True)).source
    assert "ring over kv hoisted over qb (10 iterations: one begin/end, fills read the iteration)" in outer_dep
    assert "ctx.ring_begin((&(k_cmx[0])), 16384, 2, 80, 8, 12, 0, 1, 0);" in outer_dep
    assert "ctx.ring_fill(ring, (((qb * 4) + u) + 1), (&(K[((((((((u * 2) + shave) + 2) >> 3) + qb) + ((((u * 2) + shave) + 2) & 7)) & 7) * 8192)])), 128, 128, 128);" in outer_dep


def test_vectorized_loops_lower_to_shave_vector_ops():
    # T.vectorized(32) over fp16 -> half32 (VAU.EXP through tl_npu_hexp), T.vectorized(16) over fp32 -> float16;
    # broadcasts splat, fp16 -> fp32 lane casts are __builtin_convertvector; no per-kernel emitter involved.
    src = _lower(make_flash_attention_phase(NUMQ, NUMKV, D=D, BQ=BQ, BKV=BKV, tiles=TILES, vectorized=True)).source
    assert re.search(r"\*\(half32\*\)\(mx \+ 0\) = max\(\*\(half32\*\)\(mx \+ 0\), \*\(half32\*\)\(s_cmx \+ ", src)
    assert "= tl_npu_hexp(((*(half32*)(s_cmx + " in src and "((half32)(" in src
    assert "__builtin_convertvector(*(half16*)(s_cmx + " in src and ", float16)" in src
    assert re.search(r"\*\(float16\*\)\(o_acc \+ .*\) = \(\(\*\(float16\*\)\(o_acc \+ .*\) \* \(\(float16\)\(corr\[", src)
    scalar = _lower(make_flash_attention_phase(NUMQ, NUMKV, D=D, BQ=BQ, BKV=BKV, tiles=TILES)).source
    assert "half32" not in scalar and "float16" not in scalar


@needs_cxx
@pytest.mark.parametrize("vectorized", [False, True])
def test_serial_phase_matches_numpy_in_host_sim(vectorized):
    low = _lower(make_flash_attention_phase(NUMQ, 256, D=D, BQ=BQ, BKV=BKV, tiles=TILES, pipelined=False, vectorized=vectorized))
    lib = host_sim.build(low.source, f"tl_flash_serial_sim_kv256_{'vec' if vectorized else 'scalar'}")
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
