"""M3 (plan section 4 / 9): a routed + shared MoE decode layer in one launch.

`kernels.make_moe_decode`: router GEMM -> grid barrier -> top-k routing on the SHAVE -> per item the gate/up and
down GEMVs through rings whose fill source reads the routed expert id (indirect, tile-split streams the DPU reads
in place) -> grid barrier -> combine into the residual. The device golden is `device/run_moe_device.py`. Here,
hardware-free:

  * the serial and the pipelined forms match the numpy oracle in the host sim (fp32 reference, the shared expert
    last), with a perturbed-expert negative that only fires when the routing picked that expert;
  * the expert rings are hoisted over the item loop although their fills read the item (the fill ahead reads the
    next item's expert id), both GEMV loops of one item get their own rings, and the barrier pool is shared by reuse;
  * a fill whose source index reads a buffer the iteration writes is NOT hoisted (one ring per iteration).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang.intel_npu.language as T
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import make_moe_decode, moe_reference, pack_moe  # noqa: E402

G, TILES, SHAVES = 128, 3, 2
# a small instance of the same structure: E + 1 = 17 router rows pad to 96, 3 items -> one per tile
SMALL = dict(H=512, I=128, E=16, TOPK=2, G=G, tiles=TILES, shaves=SHAVES, BNG=128, BND=128, RG=64, RD=128)


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


def _problem(seed, H, I, E, TOPK, **_):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((1, H)) * 0.5).astype(np.float16)
    router = (rng.standard_normal((E + 1, H)) * 0.05).astype(np.float16)
    gu_int = rng.integers(-8, 8, size=(E + 1, 2 * I, H)).astype(np.int8)
    gu_scales = rng.uniform(0.005, 0.02, size=(E + 1, 2 * I, H // G)).astype(np.float16)
    d_int = rng.integers(-8, 8, size=(E + 1, H, I)).astype(np.int8)
    d_scales = rng.uniform(0.005, 0.02, size=(E + 1, H, I // G)).astype(np.float16)
    res = (rng.standard_normal((1, H)) * 0.5).astype(np.float16)
    return x, router, gu_int, gu_scales, d_int, d_scales, res


def _run(low, lib, x, packed, res, H, tiles=TILES):
    out = np.zeros((1, H), np.float16)
    arrays = dict(X=x, Res=res, Logits=np.zeros((16, packed["Wr"].shape[0]), np.float16),
                  Part=np.zeros((tiles, H), np.float32), Out=out, **packed)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), tiles, SHAVES))
    return out.astype(np.float32).reshape(-1)


def test_qwen36_shape_lowers_to_indirect_split_rings_hoisted_over_the_items():
    src = _lower(make_moe_decode()).source   # H 2048, I 512, E 256, top-8 + shared: 9 items, 3 per tile
    # the router: 288 padded rows, 6 chunks of 16 per tile, a split (unicast) fp16 ring with submit-ahead, two
    # slots per SHAVE whose fills alternate the ports (2 = use & 1)
    assert "ctx.ring_begin((&(wr_cmx[0])), 65536, 4, 6, 8, 16, 2, 0, 0);" in src
    assert "x_cmx: tile-invariant resident, 1 row(s) filled by SHAVE 0" in src   # x16 is built from it in CMX
    # the expert rings: fills read ids[tile * 3 + j]; hoisted over j with the fill ahead reading the chunk's item
    assert "wq_cmx: SEGMENTED (streamed, each tile fills its own slot)" in src
    assert "ring over n hoisted over j (3 iterations: one begin/end, fills read the iteration)" in src
    assert src.count("hoisted over j") == 2
    # one chunk block per fill (128 weight rows + 4 rows of group-major scales = 132 KiB; 256 + 8 rows = 66 KiB),
    # 8 chunks x 3 items per tile, two slots per SHAVE; split + SHAVE-read = LOCAL rings (no barrier ids), their
    # fills on alternating ports in opposite phases (2 / 3)
    assert "ctx.ring_begin((&(wq_cmx[0])), 135168, 4, 24, 0, 0, 2, 0, 1);" in src
    assert "ctx.ring_begin((&(wqd_cmx[0])), 67584, 4, 24, 0, 0, 3, 0, 1);" in src
    assert "ring: READY 8..15, FREE 16..23; ring_1: local (no ids); ring_2: local (no ids)" in src
    # the fill two uses ahead: item = chunk / 8 (the next item's expert when it wraps), chunk % 8 within the expert
    assert re.search(r"ctx\.ring_fill\(ring_1, \(\(\(j \* 4\) \+ u_1\) \+ 2\), \(&\(WguP\[\(\(\(\(int64_t\)ids\[\(\(\(\(\(int64_t\)tile\) \* "
                     r"\(int64_t\)3\) \+ \(\(\(\(\(\(int64_t\)u_1\) \* \(int64_t\)2\) \+ \(\(int64_t\)shave\)\) \+ \(int64_t\)4\) >> \(int64_t\)3\)\) "
                     r"\+ \(\(int64_t\)j\)\)\]\) \* \(int64_t\)1081344\)", src)   # expert stride: 8 blocks x 135168 B
    # the workload's scales: column row0 of the group-major block after the 128 weight rows (stride 128), in this
    # SHAVE's slot of the use (slot = shave + 2 x (use & 1))
    assert "ctx.dpu_matmul_i4g((&(y_cmx[(sub * 64)])), (&(x_cmx[0])), (&(wq_cmx[((((u_1 & 1) * 270336) + (shave * 135168)) + (sub * 65536))])), (&(wq_cmx[(((((u_1 & 1) * 270336) + (shave * 135168)) + (sub * 128)) + 131072)])), 1, 64, 2048, 128, 128);" in src
    assert "ctx.dpu_matmul_i4g((&(yd_cmx[0])), (&(h_cmx[0])), (&(wqd_cmx[(((u_2 & 1) * 135168) + (shave * 67584))])), (&(wqd_cmx[((((u_2 & 1) * 135168) + (shave * 67584)) + 65536)])), 1, 256, 512, 128, 256);" in src
    assert src.count("ctx.sync_grid();") == 2


def test_fill_index_read_from_a_buffer_the_iteration_writes_is_not_hoisted():
    @T.prim_func
    def per_item(X: T.Tensor((32, 512), T.float16), W: T.Tensor((8 * 256, 512), T.float16), Sel: T.Tensor((4,), T.int32),
                 Y: T.Tensor((4, 32, 256), T.float16)):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            x_cmx = T.alloc_shared((32, 512), T.float16)
            w_cmx = T.alloc_shared((64, 512), T.float16)
            y_cmx = T.alloc_local((32, 64), T.float16)
            which = T.alloc_shared((1,), T.int32)
            T.copy(X[0, 0], x_cmx)
            for it in T.serial(4):
                which[0] = Sel[it]            # computed inside the iteration: the fill ahead could not know it
                T.sync_threads()
                for n in T.Pipelined(4, num_stages=2):
                    T.copy(W[which[0] * 256 + n * 64, 0], w_cmx)
                    T.gemm(x_cmx, w_cmx, y_cmx, transpose_B=True, clear_accum=True)
                    T.copy(y_cmx, Y[it, 0, n * 64])

    src = _lower(per_item).source
    assert "ring over n not hoisted over it: a fill's source index reads a buffer the iteration writes" in src
    assert "hoisted over it (" not in src
    assert "ctx.ring_begin((&(w_cmx[0])), 65536, 2, 4, 8, 12, -1, 1, 0);" in src   # 4 chunks: one ring per iteration


@needs_cxx
def test_small_instance_matches_the_oracle_in_host_sim():
    # the serial form (the host sim has no rings; the pipelined form is the device golden). Both SHAVEs run every
    # chunk in this form, so every per-chunk write of the source is idempotent and the accumulate is per SHAVE half.
    pipelined = False
    x, router, gu_int, gu_scales, d_int, d_scales, res = _problem(0, **SMALL)
    low = _lower(make_moe_decode(pipelined=pipelined, **SMALL))
    lib = host_sim.build(low.source, f"tl_moe_sim_{int(pipelined)}")
    packed = pack_moe(router, gu_int, gu_scales, d_int, d_scales, tiles=TILES, shaves=SHAVES, BNG=SMALL["BNG"], BND=SMALL["BND"])
    ref, order, w = moe_reference(x, router, gu_int, gu_scales, d_int, d_scales, res, SMALL["TOPK"], G)
    out = _run(low, lib, x, packed, res, SMALL["H"])
    scale = np.abs(ref - res.astype(np.float32).reshape(-1)).max()   # the MoE's own contribution sets the bar
    assert np.abs(out - ref).max() / scale < 2e-2, (np.abs(out - ref).max(), scale)
    # negative 1: a whole group of a ROUTED expert's gate rows perturbed -> the output moves
    e = int(order[0])
    gu2 = gu_int.copy()
    gu2[e, :SMALL["I"], :G] = -gu2[e, :SMALL["I"], :G] - 1
    packed2 = pack_moe(router, gu2, gu_scales, d_int, d_scales, tiles=TILES, shaves=SHAVES, BNG=SMALL["BNG"], BND=SMALL["BND"])
    out2 = _run(low, lib, x, packed2, res, SMALL["H"])
    assert np.abs(out2 - ref).max() / scale > 2e-2
    # negative 2: the same perturbation on an expert the routing did NOT pick leaves the output unchanged
    unpicked = next(i for i in range(SMALL["E"]) if i not in set(order.tolist()))
    gu3 = gu_int.copy()
    gu3[unpicked, :SMALL["I"], :G] = -gu3[unpicked, :SMALL["I"], :G] - 1
    packed3 = pack_moe(router, gu3, gu_scales, d_int, d_scales, tiles=TILES, shaves=SHAVES, BNG=SMALL["BNG"], BND=SMALL["BND"])
    out3 = _run(low, lib, x, packed3, res, SMALL["H"])
    assert np.array_equal(out3, out)

