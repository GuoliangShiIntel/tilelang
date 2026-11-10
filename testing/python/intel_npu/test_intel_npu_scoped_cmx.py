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
    assert s_pos < body.index("for (int32_t it") < if_shave, "out of the SHAVE-dependent branch and its loop"


def test_a_one_vector_buffer_whose_address_is_taken_stays_in_cmx():
    """A local buffer of exactly one vector is promoted to a vector register. A register has no address, so a
    buffer handed to a DMA or an extern call must be disqualified -- the promotion rule used to look only at
    variable-lane access, so `s` below became a register and taking its address was refused outright."""

    @T.prim_func
    def k(A: T.Tensor((32,), T.float16), Y: T.Tensor((TILES * SHAVES, 32), T.float16)):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            shave = T.get_thread_binding()
            s = T.alloc_local((32,), T.float16)
            for v in T.vectorized(32):
                s[v] = A[v]
            T.call_extern("handle", "tl_npu_dequant_i4_row", T.address_of(s[0]), T.address_of(s[0]),
                          T.address_of(s[0]), 32, 128)
            T.copy(s, Y[tile * SHAVES + shave, 0])

    body = _lower(k).source.split("kernel_entry")[0]
    assert "half* __restrict s = ctx.alloc<half>(32);" in body     # CMX, addressable
    assert "half32 s = {};" not in body                            # ... not the register form

    @T.prim_func                                                   # positive control: s's address is never taken
    def reg(A: T.Tensor((32,), T.float16), Y: T.Tensor((TILES * SHAVES, 64), T.float16)):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            shave = T.get_thread_binding()
            s = T.alloc_local((32,), T.float16)
            o = T.alloc_local((64,), T.float16)
            for v in T.vectorized(32):
                s[v] = A[v]
            for v in T.vectorized(32):
                o[v] = s[v]
            T.copy(o, Y[tile * SHAVES + shave, 0])

    assert "half32 s = {};  // local, one vector: a register" in _lower(reg).source


def test_a_loop_invariant_allocation_bubbles_out_of_every_enclosing_loop():
    """A buffer sunk into a loop body is re-marked and re-allocated every iteration for an address that never
    changes, and that made the GDN prefill scan intermittently wrong on 2+ tiles (4/12 trials clean at 3 x 2).
    Was a strict xfail while the hoist that fixes it hung the 40-layer model kernel; the hoist now skips loops
    holding a ring, which is where the hang came from (plan question 10)."""

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


def _ring_kernel(pipelined: bool):
    M, N, K, BM, BN = 512, TILES * 256, 256, 32, 256

    @T.prim_func
    def k(X: T.Tensor((M, K), T.float16), W: T.Tensor((N, K), T.float16), Y: T.Tensor((M, N), T.float16)):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            W_cmx = T.alloc_shared((BN, K), T.float16)
            X_cmx = T.alloc_shared((BM, K), T.float16)
            Y_cmx = T.alloc_local((BM, BN), T.float16)
            T.copy(W[tile * BN, 0], W_cmx)
            for c in (T.Pipelined(M // BM, num_stages=SHAVES) if pipelined else T.serial(M // BM)):
                T.copy(X[c * BM, 0], X_cmx)
                T.gemm(X_cmx, W_cmx, Y_cmx, transpose_B=True, clear_accum=True)
                s = T.alloc_local((BM, BN), T.float16)   # used only in the loop: the allocator sinks it in
                for i, j in T.grid(BM, BN):
                    s[i, j] = Y_cmx[i, j]
                T.copy(s, Y[c * BM, tile * BN])

    return k


def test_a_loop_holding_a_ring_keeps_its_allocations():
    """The hoist stops at a loop that holds a ring. Hoisting out of one anyway builds a decode model kernel that
    takes the device down (ZE_RESULT_ERROR_DEVICE_LOST at the watchdog, sha 9d4b9a4e4634) where the restricted
    hoist runs 10/10 golden (sha 6e54ddf02e1a) -- plan question 10. The serial twin is the control: the same
    loop-invariant buffer does leave the loop when no ring is in it, so a pass that hoisted nothing at all would
    fail here rather than look correct."""
    ringed = _lower(_ring_kernel(True)).source.split("kernel_entry")[0]
    assert "ctx.ring_begin(" in ringed
    assert [b for b, _ in allocations_inside_loops(ringed)], "nothing left inside the ring's loop to guard"

    serial = _lower(_ring_kernel(False)).source.split("kernel_entry")[0]
    assert "ctx.ring_begin(" not in serial
    assert allocations_inside_loops(serial) == [], "the control must hoist: without a ring the loop keeps nothing"


def allocations_inside_loops(src: str) -> list[tuple[str, int]]:
    """(buffer, loop depth) for every `ctx.alloc` the generated source makes inside a loop body. Those are the
    allocations question 10's defect acts on: the runtime re-runs mark / alloc / restore every iteration for an
    address that never changes, and one of them made the GDN prefill scan wrong on 2+ tiles. Read the depths, not
    the count: the hoist took the scan's `xr` from depth 3 to depth 1 -- a hundredfold drop in allocations per
    worker -- while the count of buffers under some loop only went 32 -> 29."""
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


def unseparated_scope_siblings(src: str) -> list[tuple[str, int]]:
    """(buffer, line) for every CMX scope block inside a loop that reuses the bytes of an earlier sibling scope
    with no rendezvous between them.

    Two sibling scopes share CMX -- CheckIntelNpuBudgets charges the heavier, not the sum -- and the tile's two
    SHAVEs are not in lockstep, so one may hand its bytes to the next block while the other still reads them.
    A `ctx.sync_tile()` (or `ctx.sync_grid()`) between them is what makes the reuse sound; npu_dsl's `plan_cmx`
    checks the same condition by name (`_separated_by_rendezvous`). This is the invariant question 10's bare
    depth bound was standing in for: depth alone cannot tell a sound reuse from an unsound one."""
    out, depth, loops = [], 0, []
    opened = {}        # brace depth -> the buffer named by the scope block opened there
    pending = {}       # brace depth -> a scope closed here and no rendezvous since
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        st = ln.strip()
        if re.search(r"ctx\.sync_tile\(\)|ctx\.sync_grid\(\)", st):
            pending = {d: v for d, v in pending.items() if d < depth}
        m = re.search(r"ctx\.alloc(?:_shared)?<", st)
        if m and depth in opened and opened[depth] is None:
            nm = re.search(r"__restrict (\w+)", st)
            opened[depth] = nm.group(1) if nm else "?"
            if loops and pending.get(depth):
                out.append((opened[depth], i + 1))
        if st == "{" or st.endswith("{"):
            depth += 1
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            opened[depth] = None if "npu::Ctx::CmxScope" in nxt else "-"
            if st.startswith("for ("):
                loops.append(depth - 1)
        if st.startswith("}"):
            if opened.get(depth) not in (None, "-"):
                pending[depth] = True
            opened.pop(depth, None)
            depth -= 1
            while loops and depth <= loops[-1]:
                loops.pop()
    return out


def test_unseparated_scope_siblings_detects_an_unsound_reuse():
    """The positive control for `unseparated_scope_siblings`: a zero from it must mean the source is sound, not
    that the detector cannot see. Two sibling scopes inside a loop with a sync between them are fine; the same
    pair with the sync removed is the unsound reuse, and it must be named."""
    def src(sep):
        return ("for (int32_t c = 0; c < 4; ++c) {\n"
                "  {\n"
                "    npu::Ctx::CmxScope cmx_scope_1(ctx);\n"
                "    half* __restrict a_cmx = ctx.alloc<half>(64);\n"
                "  }\n"
                + ("  ctx.sync_tile();\n" if sep else "")
                + "  {\n"
                "    npu::Ctx::CmxScope cmx_scope_2(ctx);\n"
                "    half* __restrict b_cmx = ctx.alloc<half>(64);\n"
                "  }\n"
                "}\n")

    assert unseparated_scope_siblings(src(True)) == []
    bad = unseparated_scope_siblings(src(False))
    assert [b for b, _ in bad] == ["b_cmx"], f"the detector missed the unseparated reuse: {bad}"


def test_the_kernels_that_allocate_inside_a_loop_are_the_ones_we_know_about():
    """Question 10's exposure list, pinned. A kernel that starts allocating inside a loop either stops doing it or
    is added here with a reason -- it cannot arrive unnoticed, which is how P3 shipped a kernel that was wrong two
    runs in three. All three now report zero: the pass hoists loop-invariant allocations itself."""
    import kernels

    # positive control first: a zero below means the pass hoisted, not that the detector is broken
    assert allocations_inside_loops(
        "for (int32_t it = 0; it < 2; ++it) {\n"
        "  half* __restrict c = ctx.alloc<half>(64);\n"
        "}\n") == [("c", 1)]

    cases = {
        "gemm": kernels.make_gemm(256, 256, 512, 64, 64, TILES, SHAVES),
        "attn_prefill": kernels.make_attn_prefill(tiles=TILES, shaves=SHAVES),
        "moe_route_prefill": kernels.make_moe_route_prefill(tiles=TILES, shaves=SHAVES),
    }
    seen = {name: len(allocations_inside_loops(_lower(func).source)) for name, func in cases.items()}
    expected = {"gemm": 0, "attn_prefill": 0, "moe_route_prefill": 0}
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
