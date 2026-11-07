"""M1 decode (plan section 4 / 9): the full-attention layer at decode in one launch.

`kernels.make_attn_decode`: qkv GEMV -> grid barrier -> per head zero-centred norm, RoPE, K / V^T cache append ->
grid barrier -> flash-decoding (each tile streams its third of the positions through K / Vt rings per kv group, online
softmax on the SHAVE, positions past pos masked) -> grid barrier -> merge over the tiles + output gate -> grid barrier
-> o_proj GEMV + residual. The device golden is `device/run_attn_device.py`. Here, hardware-free:

  * the serial form matches the numpy oracle in the host sim (output and both cache entries), with a perturbed-cache
    negative that only fires on a position <= pos;
  * the Qwen3.6 shape lowers to the qkv ring, two split K / Vt rings hoisted over the kv groups (fills read the
    group), the masked softmax, four grid barriers.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401  (sets up the tvm path)
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import attn_reference, attn_sizes, make_attn_decode, pack_attn, rope_cos_sin  # noqa: E402

G, TILES, SHAVES = 128, 3, 2
SMALL = dict(H=512, NH=4, NKV=2, D=64, ROT=32, SMAX=768, G=G, tiles=TILES, shaves=SHAVES, BN1=64, R1=64, BN2=64, R2=32, BKV=128)
THETA = 1e7


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


def attn_problem(seed, pos, H, NH, NKV, D, ROT, SMAX, G, **_):
    """Random weights, a cache filled up to pos - 1 (zeros past it) and the token at `pos`."""
    rng = np.random.default_rng(seed)
    NQKV = NH * 2 * D + 2 * NKV * D
    x = (rng.standard_normal((1, H)) * 0.5).astype(np.float16)
    res = (rng.standard_normal((1, H)) * 0.5).astype(np.float16)
    wqkv_int = rng.integers(-8, 8, size=(NQKV, H), dtype=np.int8)
    wqkv_scales = rng.uniform(0.002, 0.01, size=(NQKV, H // G)).astype(np.float16)
    wo_int = rng.integers(-8, 8, size=(H, NH * D), dtype=np.int8)
    wo_scales = rng.uniform(0.002, 0.01, size=(H, NH * D // G)).astype(np.float16)
    qnorm_w = rng.uniform(-0.2, 0.2, size=(1, D)).astype(np.float16)
    knorm_w = rng.uniform(-0.2, 0.2, size=(1, D)).astype(np.float16)
    kc = np.zeros((NKV, SMAX, D), np.float16)
    vt = np.zeros((NKV, D, SMAX), np.float16)
    kc[:, :pos] = (rng.standard_normal((NKV, pos, D)) * 0.5).astype(np.float16)
    vt[:, :, :pos] = (rng.standard_normal((NKV, D, pos)) * 0.5).astype(np.float16)
    return dict(x=x, res=res, wqkv_int=wqkv_int, wqkv_scales=wqkv_scales, wo_int=wo_int, wo_scales=wo_scales,
                qnorm_w=qnorm_w, knorm_w=knorm_w, cos_sin=rope_cos_sin(pos, ROT, THETA), pos=pos, kc=kc, vt=vt)


def attn_arrays(p, packed, H, NH, NKV, D, tiles=TILES, BN2=64, **_):
    """The kernel's parameter arrays (caches copied: the kernel appends in place)."""
    sz = attn_sizes(H, NH, NKV, D, tiles=tiles, BN2=BN2)
    return dict(X=p["x"], Res=p["res"], WqkvP=packed["WqkvP"], QNormW=p["qnorm_w"], KNormW=p["knorm_w"], CosSin=p["cos_sin"],
                Pos=np.array([[p["pos"]]], np.int32), Kc=p["kc"].copy(), Vt=p["vt"].copy(), WoP=packed["WoP"],
                Y=np.zeros((1, sz["NQKV"]), np.float16), QN=np.zeros((NKV, sz["GQ"], D), np.float16),
                Part=np.zeros((NKV, sz["GQ"], tiles, sz["PW"]), np.float32), O=np.zeros((1, sz["KO"]), np.float16),
                Out=np.zeros((1, sz["NOUT_pad"]), np.float16))


def test_qwen36_shape_lowers_to_split_kv_rings_hoisted_over_the_groups():
    src = _lower(make_attn_decode()).source
    # qkv: 9216 rows = 3 tiles x 24 chunks of 128 (no padding), local ring; o_proj 2048 -> 2112, K 4096
    assert "ctx.ring_begin((&(w1_cmx[0])), 135168, 4, 24, 0, 0, 2, 0, 1);" in src
    assert "ctx.ring_begin((&(w2_cmx[0])), 135168, 2, 11, 0, 0, -1, 0, 1);" in src
    # K and Vt: split streams (the tile's third of 1536 positions, 4 chunks of 128) read by the DPU, hoisted over g
    assert "k_cmx: SEGMENTED (streamed, each tile fills its own slot)" in src
    assert "ring over kv hoisted over g (2 iterations: one begin/end, fills read the iteration)" in src
    assert "ctx.ring_begin((&(k_cmx[0])), 65536, 2, 8, 8, 12, 0, 0, 0);" in src
    assert "ctx.ring_begin((&(v_cmx[0])), 65536, 2, 8, 16, 20, 1, 0, 0);" in src
    assert "q_cmx: tile-invariant resident in a loop, 8 rows split 4 per SHAVE, tile sync before and after" in src
    # the mask on the chunk that straddles pos, the softmax on half32 lanes with the lane reductions
    assert "= (half)-6.550400e+04f;" in src
    assert "tl_npu_vmax(mx)" in src and "tl_npu_vsum(ls)" in src
    assert "ctx.ring_dpu_submit(ring_1, ((g * 2) + u_1), (&(s_cmx[0])), (&(q_cmx[0])), ctx.ring_slot<half>(ring_1, ((g * 2) + u_1)), 8, 128, 256, 0, -1);" in src
    assert src.count("ctx.sync_grid();") == 4
    # the V^T append is a strided DMA store of one column (2 B x 256 rows, 3072 B apart)
    assert re.search(r"ctx\.dma_store\(\(&\(Vt\[.*pos_cmx\[\(int64_t\)0\].*\]\)\), \(&\(vcol\[0\]\)\), 2, 256, 3072, 2\);", src)


@needs_cxx
def test_small_instance_matches_the_oracle_in_host_sim():
    pos = 300   # inside the second tile's third of the 768 positions, mid-chunk: the mask has work to do
    p = attn_problem(0, pos, **SMALL)
    packed = pack_attn(p["wqkv_int"], p["wqkv_scales"], p["wo_int"], p["wo_scales"], tiles=TILES, BN1=SMALL["BN1"], BN2=SMALL["BN2"])
    low = _lower(make_attn_decode(pipelined=False, **SMALL))
    lib = host_sim.build(low.source, "tl_attn_sim")
    arrays = attn_arrays(p, packed, **SMALL)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    ref, ref_kc, ref_vt = attn_reference(p["x"], p["res"], p["wqkv_int"], p["wqkv_scales"], p["wo_int"], p["wo_scales"],
                                         p["qnorm_w"], p["knorm_w"], p["cos_sin"], pos, p["kc"], p["vt"],
                                         NH=SMALL["NH"], NKV=SMALL["NKV"], D=SMALL["D"], ROT=SMALL["ROT"], G=G)
    H = SMALL["H"]
    out = arrays["Out"][0, :H].astype(np.float32)
    scale = float(np.abs(ref - p["res"].astype(np.float32).reshape(-1)).max())
    assert np.abs(out - ref).max() / scale < 2e-2, (np.abs(out - ref).max(), scale)
    # the appended cache entries (fp16 of the oracle's fp32 row: merged-FC rounding of y)
    assert np.abs(arrays["Kc"][:, pos].astype(np.float32) - ref_kc[:, pos].astype(np.float32)).max() < 2e-2
    assert np.abs(arrays["Vt"][:, :, pos].astype(np.float32) - ref_vt[:, :, pos].astype(np.float32)).max() < 2e-2
    assert np.array_equal(arrays["Kc"][:, :pos], p["kc"][:, :pos]) and np.array_equal(arrays["Kc"][:, pos + 1:], p["kc"][:, pos + 1:])
    # negative 1: a key inside the window perturbed -> the output moves
    p2 = dict(p, kc=p["kc"].copy())
    p2["kc"][1, 17] += np.float16(2.0)
    arrays2 = attn_arrays(p2, packed, **SMALL)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays2), TILES, SHAVES))
    assert np.abs(arrays2["Out"][0, :H].astype(np.float32) - ref).max() / scale > 2e-2
    # negative 2: a key past pos (masked) perturbed -> unchanged output (the mask, not the zeroed cache, excludes it)
    p3 = dict(p, kc=p["kc"].copy())
    p3["kc"][1, pos + 5] = np.float16(3.0)
    arrays3 = attn_arrays(p3, packed, **SMALL)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays3), TILES, SHAVES))
    assert np.array_equal(arrays3["Out"], arrays["Out"])
