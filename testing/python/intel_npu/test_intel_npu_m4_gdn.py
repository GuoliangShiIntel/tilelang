"""M4 (plan section 4 / 9): the Gated DeltaNet mixer layer at decode in one launch.

`kernels.make_gdn_decode`: in_proj GEMV (chunk blocks, local rings) -> grid barrier -> causal conv1d + SiLU per channel
block with the conv state carried in place -> grid barrier -> per v-head the two-pass delta rule on fp32 vector lanes
with the SSM state carried in place and the gated RMSNorm -> grid barrier -> out_proj GEMV + residual. The device
golden is `device/run_gdn_device.py`. Here, hardware-free:

  * the serial form matches the numpy oracle in the host sim (output, new conv state, new SSM state), with a
    perturbed-state negative;
  * the Qwen3.6 shape lowers to two local GEMV rings, fp32 vector lanes in the scan and three grid barriers.
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
from kernels import gdn_reference, gdn_sizes, make_gdn_decode, pack_gdn  # noqa: E402

G, TILES, SHAVES = 128, 3, 2
SMALL = dict(H=512, KH=4, VH=8, DK=64, DV=64, G=G, tiles=TILES, shaves=SHAVES, BN1=64, R1=64, BN2=64, R2=32, CB=256)


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


def gdn_problem(seed, H, KH, VH, DK, G, **_):
    """Random weights and states of a GDN layer at decode (the device script uses the same)."""
    rng = np.random.default_rng(seed)
    DV = DK
    QK, VC = KH * DK, VH * DV
    CONV = 2 * QK + VC
    NIN = CONV + VC
    x = (rng.standard_normal((1, H)) * 0.5).astype(np.float16)
    res = (rng.standard_normal((1, H)) * 0.5).astype(np.float16)
    win_int = rng.integers(-8, 8, size=(NIN, H), dtype=np.int8)
    win_scales = rng.uniform(0.002, 0.01, size=(NIN, H // G)).astype(np.float16)
    wba = (rng.standard_normal((2 * VH, H)) * 0.02).astype(np.float16)
    wout_int = rng.integers(-8, 8, size=(H, VC), dtype=np.int8)
    wout_scales = rng.uniform(0.002, 0.01, size=(H, VC // G)).astype(np.float16)
    conv_w = (rng.standard_normal((4, CONV)) * 0.3).astype(np.float16)
    a_log = rng.uniform(-1.0, 1.0, size=(1, VH)).astype(np.float32)
    dt_bias = rng.uniform(-1.0, 1.0, size=(1, VH)).astype(np.float32)
    g_norm = rng.uniform(0.5, 1.5, size=(1, DV)).astype(np.float16)
    conv_state = (rng.standard_normal((3, CONV)) * 0.5).astype(np.float16)
    ssm = (rng.standard_normal((VH, DK, DV)) * 0.1).astype(np.float32)
    return dict(x=x, res=res, win_int=win_int, win_scales=win_scales, wba=wba, wout_int=wout_int, wout_scales=wout_scales,
                conv_w=conv_w, a_log=a_log, dt_bias=dt_bias, g_norm=g_norm, conv_state=conv_state, ssm=ssm)


def gdn_pack(p, BN1, BN2):
    return pack_gdn(p["win_int"], p["win_scales"], p["wout_int"], p["wout_scales"], wba=p["wba"], tiles=TILES, shaves=SHAVES, BN1=BN1, BN2=BN2)


def gdn_ref(p, KH, VH, DK, **_):
    return gdn_reference(p["x"], p["res"], p["win_int"], p["win_scales"], p["wout_int"], p["wout_scales"], p["conv_w"],
                         p["a_log"], p["dt_bias"], p["g_norm"], p["conv_state"], p["ssm"], KH=KH, VH=VH, DK=DK, G=G, wba=p["wba"])


def gdn_arrays(p, packed, H, KH, VH, DK, tiles=TILES, shaves=SHAVES, BN1=64, BN2=64, **_):
    """The kernel's parameter arrays (states copied: the kernel updates them in place)."""
    sz = gdn_sizes(H, KH, VH, DK, tiles=tiles, shaves=shaves, BN1=BN1, BN2=BN2)
    arrays = dict(X=p["x"], Res=p["res"], WinP=packed["WinP"], Wba=packed["Wba"], ConvW=p["conv_w"], ALog=p["a_log"], DtBias=p["dt_bias"],
                  GNorm=p["g_norm"], ConvState=p["conv_state"].copy(), Ssm=p["ssm"].copy(), WoutP=packed["WoutP"],
                  Y=np.zeros((1, sz["YLEN"]), np.float16), BA=np.zeros((1, sz["BA_LEN"]), np.float16), CV=np.zeros((1, sz["CVLEN"]), np.float16),
                  O=np.zeros((1, sz["OPAD"]), np.float16), Out=np.zeros((1, sz["NOUT_pad"]), np.float16))
    return arrays


def test_qwen36_shape_lowers_to_two_local_gemv_rings_and_a_vector_scan():
    src = _lower(make_gdn_decode()).source
    # in_proj: 12288 rows = 3 tiles x 32 chunks of 128 (+4 scale rows), two slots per SHAVE, local; b | a: 64 fp16
    # rows dealt 11 per worker as fp32 lane dot products
    assert "ctx.ring_begin((&(w1_cmx[0])), 135168, 4, 32, 0, 0, 2, 0, 1);" in src
    assert "ctx.dma_load((&(wba_cmx[0])), (&(Wba[((tile * 45056) + (shave * 22528))])), 4096, 11, 4096, 4096);" in src
    assert "y1_cmx[r] = ((half)tl_npu_vsum(" in src
    assert "ctx.dpu_matmul_i4g((&(y1_cmx[(sub * 64)])), (&(x_cmx[0])), (&(w1_cmx[((((u & 1) * 270336) + (shave * 135168)) + (sub * 65536))]))" in src
    # out_proj: 2048 rows padded to 2112 = 3 x 11 chunks of 64 (+2 scale rows), K 4096 -> 32 groups, 32-row workloads;
    # one slot per SHAVE (the budget), so the lone depth-1 stream is split in two halves over the ports
    assert "ctx.ring_begin((&(w2_cmx[0])), 135168, 2, 11, 0, 0, -1, 0, 1);" in src
    assert ", 1, 32, 4096, 128, 64);" in src
    assert "ring: local (no ids); ring_1: local (no ids)" in src
    # the scan on fp32 lanes: column block outer, two rows per iteration, the block accumulators and the new row
    # in one-vector register buffers (kr, kr2, sn, sn2, dr); S is the only CMX round trip per row
    assert "float16 kr = {};  // local, one vector: a register" in src
    assert re.search(r"sn = \(\(\*\(float16\*\)\(S_cmx \+ \(\((d0_\d+) \* 256\) \+ \((v0_\d+) \* 16\)\)\) \* egv\) \+ \(\*\(float16\*\)\(knv \+ \(\1 \* 32\)\) \* dr\)\);", src)
    assert re.search(r"\*\(float16\*\)\(S_cmx \+ \(\((d0_\d+) \* 256\) \+ \((v0_\d+) \* 16\)\)\) = sn;", src)
    assert re.search(r"kr = \(kr \+ \(sn \* \*\(float16\*\)\(qnv \+ \((d0_\d+) \* 32\)\)\)\);", src)
    # the conv on time-major state: unit-stride half16 lanes converted to fp32, the SiLU's exp on half16 lanes (VAU.EXP)
    assert re.search(r"eh = tl_npu_hexp\(__builtin_convertvector\(\(\(\(float16\)\(broadcast_var\w*\)\) - acc\), half16\)\);", src)
    assert "__builtin_convertvector(*(half16*)(cs_cmx + ((c0 * 16) + 512)), float16)" in src
    assert src.count("ctx.sync_grid();") == 3
    # the SSM state of one head in and out of DDR (64 KiB fp32 each way), head = worker * 6 + j
    assert "ctx.dma_load((&(S_cmx[0])), (&(Ssm[(((tile * 196608) + (shave * 98304)) + (j * 16384))])), 512, 128, 512, 512);" in src
    assert "ctx.dma_store((&(Ssm[(((tile * 196608) + (shave * 98304)) + (j * 16384))])), (&(S_cmx[0])), 512, 128, 512, 512);" in src


@needs_cxx
def test_small_instance_matches_the_oracle_in_host_sim():
    p = gdn_problem(0, **SMALL)
    packed = gdn_pack(p, SMALL["BN1"], SMALL["BN2"])
    low = _lower(make_gdn_decode(pipelined=False, **SMALL))
    lib = host_sim.build(low.source, "tl_gdn_sim")
    arrays = gdn_arrays(p, packed, **SMALL)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    ref, ref_cs, ref_ssm = gdn_ref(p, **SMALL)
    H = SMALL["H"]
    out = arrays["Out"][0, :H].astype(np.float32)
    scale = float(np.abs(ref - p["res"].astype(np.float32).reshape(-1)).max())   # the mixer's own contribution
    assert np.abs(out - ref).max() / scale < 2e-2, (np.abs(out - ref).max(), scale)
    assert np.array_equal(arrays["ConvState"][:2], ref_cs[:2])                         # the shift: exact
    new_row = arrays["ConvState"][2].astype(np.float32)                                # the GEMV output (merged-FC rounding)
    assert np.abs(new_row - ref_cs[2].astype(np.float32)).max() / np.abs(new_row).max() < 4e-3
    assert np.abs(arrays["Ssm"] - ref_ssm).max() < 1e-3 * np.abs(ref_ssm).max()
    # negative: a perturbed SSM state of one head moves the output (the head's output feeds every out_proj row)
    p2 = dict(p, ssm=p["ssm"].copy())
    p2["ssm"][3] += 0.5
    arrays2 = gdn_arrays(p2, packed, **SMALL)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays2), TILES, SHAVES))
    out2 = arrays2["Out"][0, :H].astype(np.float32)
    assert np.abs(out2 - ref).max() / scale > 2e-2

