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
import re
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401  (loads the dev build before tvm)
import tvm
import tilelang.intel_npu.language as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import (gm_scales, make_moe_expert_prefill, moe_expert_prefill_reference,  # noqa: E402
                     moe_route_prefill_reference, pack_int4, make_moe_combine)

N, H, I, E, TOPK, G = 12, 128, 128, 8, 2, 128
MB, BNG, BND, TILES, SHAVES = 4, 64, 64, 3, 2
WORKERS = TILES * SHAVES
SLOTS = N * (TOPK + 1)
REL = 4e-2   # int4 weights, fp16 gemms and an fp32 accumulate over TOPK + 1 experts


def _lower(func, dpu_overlap=False):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off",
                                           "tl.intel_npu.dpu_overlap": dpu_overlap}):
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


def _sim(p, slot_output=False, separate_shared=False, reuse_input=False, expert_mb=MB, shared_gate_up_mb=None,
        shared_gate_up_bn=None, shared_down_mb=None, shared_down_bn=None, bndc=None, pipelined=False):
    low = _lower(make_moe_expert_prefill(N=N, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=expert_mb, BNG=BNG, BND=BND,
                                         tiles=TILES, shaves=SHAVES, reuse_input=reuse_input, BNDc=bndc,
                                         pipelined=pipelined,
                                         **({"slot_output": True} if slot_output else {}),
                                         **({"separate_shared": True} if separate_shared else {}),
                                         **({"shared_gate_up_mb": shared_gate_up_mb,
                                             "shared_gate_up_bn": shared_gate_up_bn,
                                             "shared_down_mb": shared_down_mb,
                                             "shared_down_bn": shared_down_bn} if separate_shared else {})))
    lib = host_sim.build(low.source, "tl_moe_expert_sim")
    out = np.zeros((N, H), np.float32)
    arrays = dict(X=p["x"],
                  WguQ=pack_int4(p["gu_int"].reshape(-1, H)), WguS=gm_scales(p["gu_s"].reshape(-1, H // G)),
                  WdQ=pack_int4(p["d_int"].reshape(-1, I)), WdS=gm_scales(p["d_s"].reshape(-1, I // G)),
                  Wts=p["wts"].astype(np.float32), Off=p["off"].astype(np.int32),
                  Slot=p["slot"].astype(np.int32),
                  Part=np.full((SLOTS, H), np.nan, np.float16) if slot_output else np.zeros((TILES, N, H), np.float32),
                  Out=out, SharedGu=np.full((N, 2 * I), np.nan, np.float16),
                  SharedAct=np.full((N, I), np.nan, np.float16))
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
def test_bndc_decouples_down_projection_compute_from_bnd():
    """BND (64 here) still sizes the DMA fetch and scale-fetch amortization; BNDc (32, half of it) sizes
    the SHAVE-resident dequant+GEMM tile. The result must be identical to BNDc defaulting to BND, which is what
    the whole test suite above already pins -- this is the ONE new variable, isolated."""
    p = _problem()
    baseline = _sim(p)
    narrower = _sim(p, bndc=32)
    assert BND % 32 == 0 and 32 != BND, "the probe must actually exercise a smaller compute chunk"
    np.testing.assert_allclose(narrower, baseline, rtol=0, atol=0)  # same math, so bit-identical, not just REL-close


def test_bndc_fits_npu_dsl_moe_expert_geometry_within_cmx_budget():
    """The real gap this closes: npu_dsl's routed-expert geometry (BNG=32, BND=256, MBe=88) at this checkpoint's
    H=2048/I=512/E=256/TOPK=8/N=1152 scale. Without BNDc (it defaults to BND=256) the down-projection's
    SHAVE-resident wd_cmx/yd_cmx alone blow the CMX budget; BNDc=64 (matching the PREVIOUS default BND=64's
    compute-tile size, now fed by a 4x larger DMA/scale-fetch chunk) fits. Fails are a budget ValueError with a
    `cmx_bytes` figure; a wrong failure (e.g. a shape assertion) is not evidence that BNDc is the blocker."""
    H2, I2, E2, TOPK2, G2, N2 = 2048, 512, 256, 8, 128, 1152

    def geometry(**kw):
        return make_moe_expert_prefill(N=N2, H=H2, I=I2, E=E2, TOPK=TOPK2, G=G2, MB=88, BNG=32, BND=256,
                                       tiles=TILES, shaves=SHAVES, slot_output=True, reuse_input=True, **kw)

    with pytest.raises(Exception, match="CMX budget exceeded"):
        _lower(geometry())
    _lower(geometry(BNDc=64, DB=1))  # must not raise


def test_pipelined_gate_up_loop_lowers_to_a_hardware_ring():
    """`pipelined=True` runs the gate/up chunk loop over the FULL, un-split `CG` range as `T.Pipelined`
    instead of `T.serial`: the mechanism a smaller `BNG` (more, smaller chunks) needs to stop regressing,
    since each extra chunk's WguQ/WguS DMA fetch is currently paid fully serially (see kernels.py's
    `pipelined` docstring on `make_moe_expert_prefill`, and `make_moe_expert_decode`'s identical,
    already-pipelined CG loop). The ring itself deals chunk `n` to SHAVE `n % num_stages_gu`
    (`docs/examples/pingpong_multitile/README.md` section 3), so the loop body must NOT also pre-split `n`
    by shave the way the non-pipelined branches do -- an earlier version did, and it lowered fine but gave
    WRONG results on hardware (oracle FAIL, worst rel 2.4e1, 2-layer canary at BNG=32; see evidence/INDEX.md
    and the `pipelined` branch's comment in `kernels.py`) because the manual split and the ring's own
    round-robin both decided slot ownership and disagreed.

    Correctness here is LOWERING ONLY, not a host_sim run: host_sim's `npu::Ctx` never implemented the
    `ring_*` methods `LowerIntelNpuPipelined` emits (`ring_begin`/`ring_fill`/`ring_slot_acquire`/
    `ring_slot_release`/`ring_end`), the same pre-existing gap the scope_cmx ring-lifetime bug hit this
    session -- a pipelined loop can only be validated on real hardware. A wrong lowering (a Python/shape bug)
    still raises here; only the ring's numeric behavior is untested by this suite."""
    func = make_moe_expert_prefill(N=N, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=MB, BNG=BNG, BND=BND,
                                   tiles=TILES, shaves=SHAVES, pipelined=True)
    source = _lower(func).source
    assert "ctx.ring_begin(" in source and "ctx.ring_fill(" in source and "ctx.ring_end(" in source


def test_pipelined_gate_up_loop_works_even_when_cg_not_divisible_by_shaves():
    """Unlike the non-pipelined branches (which must manually split `CG` over the SHAVEs, and the `else`
    one -- `CG % shaves != 0` -- does it with a runtime guard `LowerIntelNpuPipelined` cannot see through),
    `pipelined=True` always iterates the FULL `CG` range and lets the ring's own round-robin own the split
    -- so it does not need, and does not have, a `CG % shaves != 0` special case at all."""
    odd_bng = 2 * I // 1  # CG = 1, 1 % SHAVES (2) != 0: would hit the non-pipelined "else" branch
    assert (2 * I) % odd_bng == 0 and (2 * I // odd_bng) % SHAVES != 0
    func = make_moe_expert_prefill(N=N, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=MB, BNG=odd_bng, BND=BND,
                                   tiles=TILES, shaves=SHAVES, pipelined=True)
    source = _lower(func).source  # must not raise
    assert "ctx.ring_begin(" in source


def test_compiler_derives_dequant_dpu_overlap():
    func = make_moe_expert_prefill(N=N, H=512, I=I, E=E, TOPK=TOPK, G=G, MB=MB,
                                   BNG=BNG, BND=BND, tiles=TILES, shaves=SHAVES)
    assert "ctx.dpu_matmul_start(" not in _lower(func).source
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.dpu_overlap": True}):
        source = lowering.lower(func).source
    assert source.count("ctx.dpu_matmul_start(") >= 2, "both gate/up and down loops are independent candidates"
    assert "ctx.dpu_matmul_wait();" in source


def test_moe_extern_gate_up_dequant_is_async_only_when_enabled():
    func = make_moe_expert_prefill(N=N, H=512, I=128, E=E, TOPK=TOPK, G=G, MB=MB,
                                   BNG=32, BND=BND, tiles=TILES, shaves=SHAVES)
    serial = _lower(func).source
    async_source = _lower(func, dpu_overlap=True).source
    assert "ctx.dpu_matmul_start(" not in serial
    assert re.search(r"ctx\.dpu_matmul_start\([^\n]*, 4, 32, 512,", async_source)


def _overlap_kernel(chunks=3, rewrite_input=False, escape=False, dependent_index=False):
    @T.prim_func
    def quant_tiles(X: T.Tensor((4, 128), T.float16), Q: T.Tensor((chunks * 16, 64), T.uint8),
                    S: T.Tensor((1, chunks * 16), T.float16), Y: T.Tensor((chunks, 4, 16), T.float16)):
        with T.Kernel(1, threads=1):
            activation = T.alloc_local((4, 128), T.float16)
            packed = T.alloc_local((chunks * 16, 64), T.uint8)
            scales = T.alloc_local((1, chunks * 16), T.float16)
            weights = T.alloc_local((16, 128), T.float16)
            result = T.alloc_local((4, 16), T.float16)
            index_cmx = T.alloc_local((2,), T.int32)
            index_cmx[0] = 0
            T.copy(X, activation)
            T.copy(Q, packed)
            T.copy(S, scales)
            for block in T.serial(chunks):
                T.dequant_i4_gm(weights, packed, scales, 128,
                               packed_row=block * 16 + (index_cmx[0] if dependent_index else 0), scales_col=block * 16)
                T.gemm(activation, weights, result, transpose_B=True, clear_accum=True)
                if rewrite_input:
                    activation[0, 0] = T.float16(0)
                if dependent_index:
                    for row in T.serial(4):
                        if row == 0:
                            index_cmx[0] = 1 - index_cmx[0]
                        T.copy(result[row:row + 1, 0:16], Y[block, row:row + 1, 0:16])
                else:
                    T.copy(result, Y[block, 0, 0])
            if escape:
                Y[0, 0, 0] = weights[0, 0]
    return quant_tiles


def _overlap_kernel_with_weight_loads(chunks=3):
    @T.prim_func
    def quant_tiles(X: T.Tensor((4, 128), T.float16), Q: T.Tensor((chunks * 16, 64), T.uint8),
                    S: T.Tensor((1, chunks * 16), T.float16), Y: T.Tensor((chunks, 4, 16), T.float16)):
        with T.Kernel(1, threads=1):
            activation = T.alloc_local((4, 128), T.float16)
            packed = T.alloc_local((16, 64), T.uint8)
            scales = T.alloc_local((1, 16), T.float16)
            weights = T.alloc_local((16, 128), T.float16)
            result = T.alloc_local((4, 16), T.float16)
            T.copy(X, activation)
            for block in T.serial(chunks):
                T.copy(Q[block * 16, 0], packed)
                T.copy(S[0, block * 16:block * 16 + 16], scales)
                T.dequant_i4_gm(weights, packed, scales, 128)
                T.gemm(activation, weights, result, transpose_B=True, clear_accum=True)
                T.copy(result, Y[block, 0, 0])
    return quant_tiles


def _two_overlap_kernel(chunks=3):
    @T.prim_func
    def quant_tiles(X0: T.Tensor((4, 128), T.float16), Q0: T.Tensor((chunks * 16, 64), T.uint8),
                    S0: T.Tensor((1, chunks * 16), T.float16), Y0: T.Tensor((chunks, 4, 16), T.float16),
                    X1: T.Tensor((4, 128), T.float16), Q1: T.Tensor((chunks * 16, 64), T.uint8),
                    S1: T.Tensor((1, chunks * 16), T.float16), Y1: T.Tensor((chunks, 4, 16), T.float16)):
        with T.Kernel(1, threads=1):
            a0 = T.alloc_local((4, 128), T.float16)
            a1 = T.alloc_local((4, 128), T.float16)
            q0 = T.alloc_local((chunks * 16, 64), T.uint8)
            q1 = T.alloc_local((chunks * 16, 64), T.uint8)
            s0 = T.alloc_local((1, chunks * 16), T.float16)
            s1 = T.alloc_local((1, chunks * 16), T.float16)
            w0 = T.alloc_local((16, 128), T.float16)
            w1 = T.alloc_local((16, 128), T.float16)
            y0 = T.alloc_local((4, 16), T.float16)
            y1 = T.alloc_local((4, 16), T.float16)
            T.copy(X0, a0)
            T.copy(X1, a1)
            T.copy(Q0, q0)
            T.copy(Q1, q1)
            T.copy(S0, s0)
            T.copy(S1, s1)
            for block0 in T.serial(chunks):
                T.dequant_i4_gm(w0, q0, s0, 128, packed_row=block0 * 16, scales_col=block0 * 16)
                T.gemm(a0, w0, y0, transpose_B=True, clear_accum=True)
                T.copy(y0, Y0[block0, 0, 0])
            for block1 in T.serial(chunks):
                T.dequant_i4_gm(w1, q1, s1, 128, packed_row=block1 * 16, scales_col=block1 * 16)
                T.gemm(a1, w1, y1, transpose_B=True, clear_accum=True)
                T.copy(y1, Y1[block1, 0, 0])
    return quant_tiles


@needs_cxx
@pytest.mark.parametrize("chunks", [1, 3, 4])
def test_dpu_overlap_matches_serial_with_odd_and_even_tails(chunks):
    rng = np.random.default_rng(42)
    values = rng.integers(-8, 8, size=(chunks * 16, 128), dtype=np.int8)
    arrays = {"X": rng.normal(size=(4, 128)).astype(np.float16), "Q": pack_int4(values),
              "S": rng.uniform(0.01, 0.1, size=(1, chunks * 16)).astype(np.float16)}
    outputs = []
    for enabled in (False, True):
        low = _lower(_overlap_kernel(chunks), dpu_overlap=enabled)
        assert ("ctx.dpu_matmul_start(" in low.source) == (enabled and chunks > 1)
        output = np.full((chunks, 4, 16), np.nan, np.float16)
        lib = host_sim.build(low.source, "dpu_overlap_tiles")
        args = lowering.order_arrays(low, {**arrays, "Y": output})
        host_sim.check_status(host_sim.run(lib, args, tiles=1, shaves=1))
        outputs.append(output)
    assert np.isfinite(outputs[1]).all()
    np.testing.assert_array_equal(outputs[1], outputs[0])


@needs_cxx
def test_dpu_overlap_selects_multiple_disjoint_loops_in_one_function():
    func = _two_overlap_kernel(chunks=3)
    rng = np.random.default_rng(42)
    arrays = {}
    for suffix in ("0", "1"):
        values = rng.integers(-8, 8, size=(3 * 16, 128), dtype=np.int8)
        arrays[f"X{suffix}"] = rng.normal(size=(4, 128)).astype(np.float16)
        arrays[f"Q{suffix}"] = pack_int4(values)
        arrays[f"S{suffix}"] = rng.uniform(0.01, 0.1, size=(1, 3 * 16)).astype(np.float16)
    outputs = []
    for enabled in (False, True):
        low = _lower(func, dpu_overlap=enabled)
        starts = low.source.count("ctx.dpu_matmul_start(")
        assert (starts >= 4) == enabled
        result = {f"Y{suffix}": np.full((3, 4, 16), np.nan, np.float16) for suffix in ("0", "1")}
        lib = host_sim.build(low.source, "dpu_overlap_two_disjoint_loops")
        host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, {**arrays, **result}), 1, 1))
        outputs.append(result)
    np.testing.assert_array_equal(outputs[1]["Y0"], outputs[0]["Y0"])
    np.testing.assert_array_equal(outputs[1]["Y1"], outputs[0]["Y1"])


@needs_cxx
def test_dpu_overlap_with_loop_local_weight_loads_matches_serial():
    func = _overlap_kernel_with_weight_loads()
    rng = np.random.default_rng(42)
    values = rng.integers(-8, 8, size=(3 * 16, 128), dtype=np.int8)
    arrays = {"X": rng.normal(size=(4, 128)).astype(np.float16), "Q": pack_int4(values),
              "S": rng.uniform(0.01, 0.1, size=(1, 3 * 16)).astype(np.float16)}
    outputs = []
    for enabled in (False, True):
        with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off",
                                               "tl.intel_npu.dpu_overlap": enabled,
                                               "tl.intel_npu.prefetch": False}):
            low = lowering.lower(func)
        assert ("ctx.dpu_matmul_start(" in low.source) == enabled
        output = np.full((3, 4, 16), np.nan, np.float16)
        lib = host_sim.build(low.source, "dpu_overlap_loop_local_weights")
        host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, {**arrays, "Y": output}), 1, 1))
        outputs.append(output)
    np.testing.assert_array_equal(outputs[1], outputs[0])


@pytest.mark.parametrize("rewrite_input,escape", [(True, False), (False, True)])
def test_dpu_overlap_rejects_mutation_and_escaping_weights(rewrite_input, escape):
    source = _lower(_overlap_kernel(rewrite_input=rewrite_input, escape=escape), dpu_overlap=True).source
    assert "ctx.dpu_matmul_start(" not in source


def test_dpu_overlap_rejects_next_iteration_index_dependency():
    source = _lower(_overlap_kernel(dependent_index=True), dpu_overlap=True).source
    assert "ctx.dpu_matmul_start(" not in source


def test_typed_dequant_rejects_invalid_groups_and_offsets():
    output = tvm.tirx.decl_buffer((16, 128), "float16", scope="local")
    packed = tvm.tirx.decl_buffer((16, 64), "uint8", scope="local")
    scales = tvm.tirx.decl_buffer((1, 16), "float16", scope="local")
    with pytest.raises(ValueError, match="incompatible"):
        T.dequant_i4_gm(output, packed, scales, group=0)
    with pytest.raises(ValueError, match="packed row offset"):
        T.dequant_i4_gm(output, packed, scales, packed_row=1)
    with pytest.raises(ValueError, match="scale column offset"):
        T.dequant_i4_gm(output, packed, scales, scales_col=1)


@needs_cxx
@pytest.mark.parametrize("tokens,block_rows", [(12, 4), (13, 4), (13, 8)])
def test_blocked_combine_matches_slot_order_with_invalid_capacity(tokens, block_rows):
    choices, hidden, experts = 9, 128, 16
    slots, live = tokens * choices, tokens - 2
    rng = np.random.default_rng(23)
    order = np.zeros(slots, np.int32)
    order[:live * choices] = rng.permutation(live * choices)
    values = np.full((slots, hidden), np.nan, np.float16)
    values[:live * choices] = rng.normal(size=(live * choices, hidden)).astype(np.float16)
    weights = rng.uniform(size=(tokens, 16)).astype(np.float32)
    offsets = np.full(experts + 2, live * choices, np.int32)
    output = np.full((tokens, hidden), np.nan, np.float32)
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.batch_dma": True}):
        low = lowering.lower(make_moe_combine(tokens, hidden, experts, choices - 1, block_rows, 3, 2))
    assert "ctx.dma_batch_add(" in low.source
    inverse = np.empty(live * choices, np.int32)
    inverse[order[:live * choices]] = np.arange(live * choices)
    expected = np.zeros_like(output)
    for token in range(live):
        for choice in range(choices):
            expected[token] += weights[token, choice] * values[inverse[token * choices + choice]].astype(np.float32)
    arrays = lowering.order_arrays(low, {"Part": values, "Slot": order, "Wts": weights, "Off": offsets, "Out": output})
    lib = host_sim.build(low.source, "blocked_moe_combine")
    host_sim.check_status(host_sim.run(lib, arrays, tiles=3, shaves=2))
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


@needs_cxx
@pytest.mark.parametrize("slot_output", [False, True])
@pytest.mark.parametrize("reuse_input", [False, True])
@pytest.mark.parametrize("expert_mb", [MB, 88])
def test_matches_the_oracle(slot_output, reuse_input, expert_mb):
    p = _problem()
    ref = moe_expert_prefill_reference(p["x"], p["gu_int"], p["gu_s"], p["d_int"], p["d_s"],
                                       p["wts"], p["off"], p["slot"], TOPK=TOPK, G=G)
    got = _sim(p, slot_output=slot_output, reuse_input=reuse_input, expert_mb=expert_mb)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < REL, f"rel {rel:.3e}"


def test_slot_output_refuses_uneven_shave_rows():
    with pytest.raises(AssertionError, match="slot-output token rows must divide"):
        make_moe_expert_prefill(N=N, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=3,
                               BNG=BNG, BND=BND, tiles=TILES, shaves=SHAVES, slot_output=True)


@needs_cxx
@pytest.mark.parametrize("reference_blocks", [False, True])
def test_separate_shared_matches_grouped_output(reference_blocks):
    problem = _problem(seed=9)
    expected = _sim(problem, slot_output=True)
    blocks = ({"shared_gate_up_mb": 2, "shared_gate_up_bn": 32,
               "shared_down_mb": 2, "shared_down_bn": 32} if reference_blocks else {})
    actual = _sim(problem, slot_output=True, separate_shared=True, **blocks)
    assert np.isfinite(actual).all()
    assert np.abs(actual - expected).max() / max(float(np.abs(expected).max()), 1e-6) < 1e-3


def test_separate_shared_requires_slots_and_complete_blocks():
    kwargs = dict(N=N, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=MB, BNG=BNG, BND=BND,
                  tiles=TILES, shaves=SHAVES, separate_shared=True)
    with pytest.raises(AssertionError, match="requires slot-major output"):
        make_moe_expert_prefill(**kwargs)
    kwargs["N"] = N + 2
    with pytest.raises(AssertionError, match="requires complete projection token blocks"):
        make_moe_expert_prefill(**kwargs, slot_output=True)


@needs_cxx
@pytest.mark.parametrize("slot_output,separate_shared", [(False, False), (True, False), (True, True)])
def test_perturbing_one_expert_moves_only_its_tokens(slot_output, separate_shared):
    """What "grouped by expert" means: a per-token loop would pass the oracle test too, this one it would not."""
    p = _problem()
    base = _sim(p, slot_output=slot_output, separate_shared=separate_shared)
    victim = int(np.argmax(np.diff(p["off"][:E + 1])))        # the routed expert with the most tokens
    touched = {int(p["slot"][s]) // (TOPK + 1) for s in range(int(p["off"][victim]), int(p["off"][victim + 1]))}
    assert 0 < len(touched) < N, "the probe needs an expert that some but not all tokens use"
    p["d_int"][victim] = np.clip(p["d_int"][victim].astype(np.int16) + 3, -8, 7).astype(np.int8)
    got = _sim(p, slot_output=slot_output, separate_shared=separate_shared)
    moved = {t for t in range(N) if np.abs(got[t] - base[t]).max() > 1e-3}
    assert moved == touched, f"expert {victim}: tokens {sorted(moved)} moved, expected {sorted(touched)}"


@needs_cxx
@pytest.mark.parametrize("slot_output,separate_shared", [(False, False), (True, False), (True, True)])
def test_a_zeroed_choice_weight_drops_that_contribution(slot_output, separate_shared):
    p = _problem()
    base = _sim(p, slot_output=slot_output, separate_shared=separate_shared)
    t0, k0 = int(p["slot"][0]) // (TOPK + 1), int(p["slot"][0]) % (TOPK + 1)
    p["wts"] = p["wts"].copy()
    p["wts"][t0, k0] = 0.0
    got = _sim(p, slot_output=slot_output, separate_shared=separate_shared)
    assert np.abs(got[t0] - base[t0]).max() > 1e-3, "zeroing a choice's weight changed nothing"
    others = [t for t in range(N) if t != t0]
    assert np.abs(got[others] - base[others]).max() < 1e-3, "zeroing one token's weight moved another token"


@needs_cxx
def test_perturbing_shared_expert_moves_all_tokens():
    problem = _problem(seed=17)
    expected = _sim(problem, slot_output=True, separate_shared=True)
    problem["d_int"][E] = np.clip(problem["d_int"][E].astype(np.int16) + 3, -8, 7).astype(np.int8)
    actual = _sim(problem, slot_output=True, separate_shared=True)
    assert set(np.flatnonzero(np.max(np.abs(actual - expected), axis=1) > 1e-3)) == set(range(N))
