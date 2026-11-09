"""Scoped CMX (plan M5, PlanCmx): allocations sink to the block that uses them (PlanAndUpdateBufferAllocationLocation
for loops, ScopeIntelNpuCmx for the phases between grid barriers), codegen brackets every allocating block with
`npu::Ctx::CmxScope`, and CheckIntelNpuBudgets charges the peak nesting instead of the sum.

  * two phases with a big buffer each fit only with scoping: refused by name with `tl.intel_npu.scoped_cmx=False`
    (the negative control of the budget rule) and accepted by default, each phase in its own C++ block;
  * an allocation under a SHAVE-dependent branch is hoisted to the enclosing uniform block (every SHAVE must run the
    same allocation sequence) while a tile-dependent branch keeps it;
  * the host simulation runs a two-phase kernel whose phases reuse the same CMX (golden against numpy).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401
import tilelang.intel_npu.language as T
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402

TILES, SHAVES = 2, 2
BIG = 384 * 1024   # halves: 768 KiB per buffer; two do not fit the 1439 KiB workspace next to the runtime reserve


def _lower(func, **cfg):
    config = {"tl.intel_npu.plan_workloads": "off"}
    config.update(cfg)
    with tvm.transform.PassContext(config=config):
        return lowering.lower(func)


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except RuntimeError:
        return False


def make_two_phase(n: int):
    @T.prim_func
    def two(A: T.Tensor((n,), T.float16), B: T.Tensor((n,), T.float16), Y: T.Tensor((2, n), T.float16)):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            a_cmx = T.alloc_local((n,), T.float16)
            b_cmx = T.alloc_local((n,), T.float16)
            T.copy(A[0], a_cmx)
            for i in T.serial(n):
                a_cmx[i] = a_cmx[i] * T.float16(2)
            T.copy(a_cmx, Y[0, 0])
            T.sync_grid()
            T.copy(B[0], b_cmx)
            for i in T.serial(n):
                b_cmx[i] = b_cmx[i] + T.float16(1)
            T.copy(b_cmx, Y[1, 0])

    return two


def test_two_phases_share_cmx_and_the_sum_is_refused_without_scoping():
    src = _lower(make_two_phase(BIG // SHAVES)).source
    # each phase is a C++ block that allocates and releases its buffer
    assert src.count("npu::Ctx::CmxScope") == 2
    assert re.search(r"\{\n\s+npu::Ctx::CmxScope cmx_scope_0\(ctx\);\n\s+half\* __restrict a_cmx = ctx.alloc<half>\(196608\);", src)
    assert re.search(r"\{\n\s+npu::Ctx::CmxScope cmx_scope_1\(ctx\);\n\s+half\* __restrict b_cmx = ctx.alloc<half>\(196608\);", src)
    with pytest.raises(Exception, match="CMX budget exceeded"):
        _lower(make_two_phase(BIG // SHAVES), **{"tl.intel_npu.scoped_cmx": False})


def test_allocation_under_a_shave_branch_is_hoisted():
    @T.prim_func
    def k(A: T.Tensor((64,), T.float16), Y: T.Tensor((TILES * SHAVES, 64), T.float16)):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            shave = T.get_thread_binding()
            for it in T.serial(2):
                if shave == 0:
                    s_cmx = T.alloc_local((64,), T.float16)
                    T.copy(A[0], s_cmx)
                    T.copy(s_cmx, Y[tile * SHAVES, 0])

    src = _lower(k).source
    body = src.split("kernel_entry")[0]
    s_pos, if_shave = body.index("half* __restrict s_cmx"), body.index("if (shave == 0)")
    assert body.index("for (int32_t it") < s_pos < if_shave, "hoisted out of the SHAVE-dependent branch, inside the loop body"


@pytest.mark.xfail(strict=True, reason="plan question 10: the hoist that fixes this hangs the 40-layer model kernel")
def test_a_loop_invariant_allocation_bubbles_out_of_every_enclosing_loop():
    """What a correct question-10 fix must achieve. A buffer sunk into a loop body is re-marked and re-allocated
    every iteration for an address that never changes, and that made the GDN prefill scan intermittently wrong on
    2+ tiles (4/12 trials clean at 3 x 2). Strict xfail: it fails the suite the day it starts passing, which is
    the signal that the fix landed."""

    @T.prim_func
    def k(A: T.Tensor((64,), T.float16), Y: T.Tensor((TILES * SHAVES, 64), T.float16)):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            shave = T.get_thread_binding()
            for outer in T.serial(3):
                for inner in T.serial(2):
                    c = T.alloc_local((64,), T.float16)
                    T.copy(A[0], c)
                    T.copy(c, Y[tile * SHAVES + shave, 0])

    body = _lower(k).source.split("kernel_entry")[0]
    alloc = body.index("half* __restrict c = ctx.alloc")
    assert alloc < body.index("for (int32_t outer"), "hoisted past the OUTER loop, not just the inner one"


def allocations_inside_loops(src: str) -> list[tuple[str, int]]:
    """(buffer, loop depth) for every `ctx.alloc` the generated source makes inside a loop body. Those are the
    allocations question 10's defect acts on: the runtime re-runs mark / alloc / restore every iteration for an
    address that never changes, and one of them made the GDN prefill scan wrong on 2+ tiles. Until the pass is
    fixed this is the exposure list, so a kernel that grows one is seen rather than discovered on the device."""
    out, depth, loops = [], 0, []
    for ln in src.splitlines():
        st = ln.strip()
        if re.search(r"ctx\.alloc(_shared)?<", st) and loops:
            m = re.search(r"__restrict (\w+)", st)
            out.append((m.group(1) if m else "?", len(loops)))
        if st.startswith("for ("):
            loops.append(depth)
        depth += st.count("{") - st.count("}")
        while loops and depth <= loops[-1]:
            loops.pop()
    return out


def test_the_kernels_that_allocate_inside_a_loop_are_the_ones_we_know_about():
    """Question 10's exposure list, pinned. A kernel that starts allocating inside a loop either stops doing it or
    is added here with a reason -- it cannot arrive unnoticed, which is how P3 shipped a kernel that was wrong two
    runs in three. The author's lever is a first use of the buffer before the loops (`make_gdn_scan_prefill`)."""
    import kernels

    cases = {
        "gemm": kernels.make_gemm(256, 256, 512, 64, 64, TILES, SHAVES),
        "attn_prefill": kernels.make_attn_prefill(tiles=TILES, shaves=SHAVES),
        "moe_route_prefill": kernels.make_moe_route_prefill(tiles=TILES, shaves=SHAVES),
    }
    seen = {name: len(allocations_inside_loops(_lower(func).source)) for name, func in cases.items()}
    expected = {"gemm": 3, "attn_prefill": 11, "moe_route_prefill": 7}
    assert seen == expected, f"exposure to question 10 changed: {seen} != {expected}"
    # the control: the detector can report zero. A kernel whose allocations are all above its loops has none.

    @T.prim_func
    def clean(A: T.Tensor((64,), T.float16), Y: T.Tensor((TILES * SHAVES, 64), T.float16)):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            shave = T.get_thread_binding()
            c = T.alloc_local((64,), T.float16)
            T.copy(A[0], c)
            for it in T.serial(2):
                T.copy(c, Y[tile * SHAVES + shave, 0])

    assert allocations_inside_loops(_lower(clean).source) == []


@pytest.mark.skipif(not _has_host_cxx(), reason="host sim needs clang++ (TL_HOST_CXX or PATH)")
def test_two_phase_kernel_is_correct_in_the_host_sim():
    n = 4096
    low = _lower(make_two_phase(n))
    lib = host_sim.build(low.source, "tl_scoped_two_phase")
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float16)
    b = rng.standard_normal(n).astype(np.float16)
    y = np.zeros((2, n), np.float16)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, dict(A=a, B=b, Y=y)), TILES, SHAVES))
    assert np.array_equal(y[0], (a.astype(np.float32) * 2).astype(np.float16))
    assert np.array_equal(y[1], (b.astype(np.float32) + 1).astype(np.float16))
