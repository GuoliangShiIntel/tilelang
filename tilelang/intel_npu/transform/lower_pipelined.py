"""PlanIntelNpuDistribution and LowerIntelNpuPipelined: the multi-tile flow from the author's loops.

docs/examples/pingpong_multitile/README.md sections 1-3, as the plan's section 3.2 orders it.
Both passes run after LowerIntelNpuTileOp (copies are `intel_npu_dma_copy`, gemms are
`intel_npu_dpu_matmul`) and before LowerOpaqueBlock (allocations are still SBlock buffers).

PlanIntelNpuDistribution derives, for every DDR -> CMX copy, how it is distributed:
    * with tl.intel_npu.multicast enabled (default false), a local DPU operand loaded from a
        parameter DDR buffer at a tile-invariant address in uniform serial control flow is
        broadcast by tile 0's matching SHAVE, between grid rendezvous; ownership is unchanged;
  * destination scope local  -> per-SHAVE copy, left alone (the source names the worker);
  * destination scope shared, outside a pipelined loop -> SEGMENTED resident operand: the rows
    are split over the tile's SHAVEs (both DMA ports) and a tile sync follows;
  * destination scope shared, inside a pipelined loop -> the streamed operand, left for
    LowerIntelNpuPipelined; its source must not depend on the tile (a tile split on the
    streamed loop is refused by name).
  A copy whose source depends on neither the tile nor the SHAVE into a local buffer is the
  same DDR bytes read once per worker; it is legal (a bias, a probe) and noted in the source.

LowerIntelNpuPipelined lowers a `T.Pipelined(num_stages=S)` loop whose body is one streamed
fill, one DPU matmul reading it and the stores of the result into the ring program of the
runtime (`npu::Ctx::Ring`): S must equal the SHAVEs per tile (slot = SHAVE), the loop is dealt
round-robin (chunk c -> SHAVE c % S), the streamed buffer becomes S slots, the output buffer is
double-buffered by use parity (submit-ahead), and the per-use program is

    wait(u); submit(u+1); fill(u+1) [issuer: tile 0, SHAVE s]; store(u)

A pipelined loop directly inside a serial loop is hoisted: one ring over all the iterations (several
pipelined loops in one outer loop each get their own), and a fill issued ahead reads the chunk's own
iteration (its source may read the outer variable, e.g. an expert id), unless the source index reads
a buffer the iteration writes, in which case the loop gets one ring per iteration.
Barrier ids are left as -1 for AssignIntelNpuBarriers.
"""

from __future__ import annotations

import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform, post_order_visit, substitute
from tvm.tirx.transform import prim_func_pass

DMA_COPY = "tl.intel_npu_dma_copy"
DPU_MATMUL = "tl.intel_npu_dpu_matmul"
RING_BEGIN = "tl.intel_npu_ring_begin"
RING_SLOT = "tl.intel_npu_ring_slot"
RING_FILL = "tl.intel_npu_ring_fill"
RING_SUBMIT = "tl.intel_npu_ring_dpu_submit"
RING_WAIT = "tl.intel_npu_ring_dpu_wait"
RING_ACQUIRE = "tl.intel_npu_ring_slot_acquire"
RING_RELEASE = "tl.intel_npu_ring_slot_release"
RING_END = "tl.intel_npu_ring_end"
NOTES_ATTR = "intel_npu.notes"


def _op(name):
    return tvm.ir.Op.get(name)


def _is_call(stmt, name):
    return isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call) and stmt.value.op.same_as(_op(name))


def _vars_of(expr):
    found = []

    def visit(node):
        if isinstance(node, tirx.Var):
            found.append(node)

    post_order_visit(expr, visit)
    return found


def _uses(expr, var):
    return any(v.same_as(var) for v in _vars_of(expr))


def _load_of(addr):
    """The BufferLoad under an address_of(...)."""
    assert isinstance(addr, tirx.Call) and isinstance(addr.args[0], tirx.BufferLoad), addr
    return addr.args[0]


def _address_of(buffer, indices):
    return tirx.Call("handle", _op("tirx.address_of"), [tirx.BufferLoad(buffer, indices)])


def _i32(v):
    return tirx.IntImm("int32", int(v))


def _const(e):
    assert isinstance(e, tirx.IntImm), f"intel_npu: expected a constant, got {e}"
    return int(e.value)


def _const_or(e, default):
    return int(e.value) if isinstance(e, tirx.IntImm) else default


def _as_i32(e):
    return e if e.dtype == "int32" else tirx.Cast("int32", e)


def _divisible(e, n) -> bool:
    """Whether the extent `e` (constant or runtime) is provably a multiple of `n`."""
    if isinstance(e, tirx.IntImm):
        return int(e.value) % n == 0
    return bool(tvm.arith.Analyzer().can_prove_equal(tirx.floormod(e, _i32(n)), _i32(0)))


class _Launch:
    """The launch variables and extents of the kernel (thread_extent AttrStmts)."""

    def __init__(self, func: tirx.PrimFunc):
        self.tile_var = self.shave_var = None
        self.tiles = self.shaves = 1

        def visit(node):
            if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent":
                iv = node.node
                if iv.thread_tag == "blockIdx.x":
                    self.tile_var, self.tiles = iv.var, _const(node.value)
                elif iv.thread_tag == "threadIdx.x":
                    self.shave_var, self.shaves = iv.var, _const(node.value)

        post_order_visit(func.body, visit)
        if self.tile_var is None or self.shave_var is None:
            raise ValueError("intel_npu: kernel has no blockIdx.x / threadIdx.x launch binding")


def _reads_buffer(stmt, buf) -> bool:
    hit = [False]

    def visit(node):
        if isinstance(node, tirx.BufferLoad) and node.buffer.same_as(buf):
            hit[0] = True

    post_order_visit(stmt, visit)
    return hit[0]


def _offset_loads(stmt, buf, row_offset):
    """BufferLoads of `buf` inside `stmt` with `row_offset` added to the leading index (the slot of the use)."""

    def rewrite(node):
        if isinstance(node, tirx.BufferLoad) and node.buffer.same_as(buf):
            idx = list(node.indices)
            idx[0] = idx[0] + row_offset
            return tirx.BufferLoad(node.buffer, idx)
        return None

    return ir_transform(stmt, None, rewrite, ["tirx.BufferLoad"])


def _storage_sync_shared():
    return tirx.Evaluate(tirx.Call("int32", _op("tirx.tvm_storage_sync"), [tirx.StringImm("shared")]))


def _with_notes(func: tirx.PrimFunc, text: str) -> tirx.PrimFunc:
    def wrap(stmt):
        if isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "thread_extent":
            notes = tirx.AttrStmt(tirx.IntImm("int32", 0), NOTES_ATTR, tirx.StringImm(text), stmt.body)
            return tirx.AttrStmt(stmt.node, stmt.attr_key, stmt.value, notes)
        return None

    return func.with_body(ir_transform(func.body, wrap, None, ["tirx.AttrStmt"]))


# ---------------------------------------------------------------------------------------------
# PlanIntelNpuDistribution
# ---------------------------------------------------------------------------------------------


def PlanIntelNpuDistribution():
    @prim_func_pass(opt_level=0, name="tl.PlanIntelNpuDistribution")
    def _pass(func: tirx.PrimFunc, mod, ctx):
        launch = _Launch(func)
        notes = []
        pipelined_depth = [0]
        loop_depth = [0]
        shave_branch = []  # one flag per enclosing SHAVE-dependent branch: did it fill a shared buffer?
        multicast = bool(ctx.config.get("tl.intel_npu.multicast", False)) and launch.tiles > 1
        uniform = [list(func.params)]
        dpu_inputs = []
        written = []

        def find_writes(node):
            if isinstance(node, tirx.BufferStore):
                written.append(node.buffer.data)
            elif isinstance(node, tirx.Call) and getattr(node.op, "name", "") == DMA_COPY:
                if _const_or(node.args[6], 1) == 0:
                    written.append(_load_of(node.args[0]).buffer.data)

        post_order_visit(func.body, find_writes)
        readonly = [buffer.data for buffer in func.buffer_map.values()
                    if not any(buffer.data.same_as(data) for data in written)]

        def find_input(node):
            if isinstance(node, tirx.Call) and getattr(node.op, "name", "") == DPU_MATMUL:
                dpu_inputs.extend(_load_of(arg).buffer.data for arg in node.args[1:3])

        post_order_visit(func.body, find_input)

        def is_uniform(expr):
            safe = [True]

            def visit(node):
                if isinstance(node, tirx.BufferLoad):
                    if not any(node.buffer.data.same_as(data) for data in readonly):
                        safe[0] = False
                elif isinstance(node, tirx.Call):
                    safe[0] = False
                elif isinstance(node, tirx.Var) and not any(node.same_as(var) for var in uniform[-1]):
                    safe[0] = False

            post_order_visit(expr, visit)
            return safe[0]

        overlap_buffers = []

        def plan_overlap(loop):
            if not bool(ctx.config.get("tl.intel_npu.multicast_overlap", False)) or launch.tiles < 2:
                return None
            if not uniform[-1] or not is_uniform(loop.min) or not is_uniform(loop.extent):
                return None
            if loop.kind != tirx.ForKind.SERIAL or _const_or(loop.min, -1) != 0:
                return None
            trips = _const_or(loop.extent, 0)
            if trips < 2 or int(loop.annotations.get("num_stages", 0)) > 0:
                return None
            items = list(loop.body.seq) if isinstance(loop.body, tirx.SeqStmt) else [loop.body]
            if len(items) != 3 or not all(_is_call(item, name) for item, name in zip(items, [DMA_COPY, DPU_MATMUL, DMA_COPY])):
                return None
            fill, gemm, store = (item.value for item in items)
            if len(fill.args) != 7 or _const_or(fill.args[6], 0) != 1 or _const_or(store.args[6], 1) != 0:
                return None
            activation, source = _load_of(fill.args[0]), _load_of(fill.args[1])
            output, weights = _load_of(gemm.args[0]), _load_of(gemm.args[2])
            destination = _load_of(store.args[0])
            if activation.buffer.scope() != "local" or output.buffer.scope() != "local":
                return None
            if weights.buffer.scope() not in ("shared", "shared.dyn"):
                return None
            if not _load_of(gemm.args[1]).buffer.same_as(activation.buffer):
                return None
            if not _load_of(store.args[1]).buffer.same_as(output.buffer):
                return None
            if source.buffer.data.same_as(destination.buffer.data):
                return None
            if any(_uses(weights, var) for var in (loop.loop_var, launch.shave_var)):
                return None
            if not any(source.buffer.data.same_as(buf.data) for buf in func.buffer_map.values()):
                return None
            if any(_uses(index, launch.tile_var) for index in source.indices):
                return None
            allowed = uniform[-1] + [loop.loop_var, launch.shave_var]
            safe_address = [True]

            def check_address(node):
                if (
                    isinstance(node, (tirx.BufferLoad, tirx.Call))
                    or isinstance(node, tirx.Var)
                    and not any(node.same_as(var) for var in allowed)
                ):
                    safe_address[0] = False

            for index in source.indices:
                post_order_visit(index, check_address)
            if not safe_address[0]:
                return None
            from .prefetch import _full_buffer_load, _refs

            if not _full_buffer_load(fill, activation.buffer):
                return None
            if not all(_const_or(index, -1) == 0 for index in output.indices):
                return None
            for buffer in (activation.buffer, output.buffer):
                if _refs(func.body, buffer.data) != _refs(loop.body, buffer.data):
                    return None
            chunk = tirx.Var(loop.loop_var.name + "_chunk", loop.loop_var.dtype)
            sub = {loop.loop_var: tirx.floordiv(chunk, launch.shaves), launch.shave_var: tirx.floormod(chunk, launch.shaves)}
            body = substitute(loop.body, sub)
            new_items = list(body.seq)
            new_source = _load_of(new_items[0].value.args[1])
            new_destination = _load_of(new_items[2].value.args[0])
            rows = _const_or(gemm.args[3], 0)
            analyzer = tvm.arith.Analyzer()
            for address in (new_source, new_destination):
                if len(address.indices) != 2 or _uses(address.indices[1], chunk):
                    return None
                row = address.indices[0]
                if not analyzer.can_prove_equal(substitute(row, {chunk: chunk + 1}) - row, rows):
                    return None
            shared = tirx.decl_buffer(list(activation.buffer.shape), activation.buffer.dtype, activation.buffer.name, scope="shared")
            overlap_buffers.append((activation.buffer, shared))
            annotations = dict(loop.annotations)
            annotations["num_stages"] = launch.shaves
            annotations["intel_npu.no_hoist"] = 1
            notes.append(
                f"{activation.buffer.name}: ownership-preserving automatic multicast ring; "
                f"{trips} uses per SHAVE, shared resident operand, no hoist across weight preparation"
            )
            return tirx.For(chunk, loop.min, _i32(trips * launch.shaves), loop.kind, body, loop.thread_binding, annotations)

        def multicast_fill(stmt):
            args = stmt.value.args
            dst, src = _load_of(args[0]), _load_of(args[1])
            if not multicast or len(args) != 7 or not uniform[-1]:
                return None
            if dst.buffer.scope() != "local" or not any(dst.buffer.data.same_as(var) for var in dpu_inputs):
                return None
            if not any(src.buffer.data.same_as(buf.data) for buf in func.buffer_map.values()):
                return None
            allowed = uniform[-1] + [launch.shave_var]
            for expr in list(dst.indices) + list(src.indices) + list(args[2:6]):
                bad = [False]

                def visit(node):
                    if (
                        isinstance(node, (tirx.BufferLoad, tirx.Call))
                        or isinstance(node, tirx.Var)
                        and not any(node.same_as(var) for var in allowed)
                    ):
                        bad[0] = True

                post_order_visit(expr, visit)
                if bad[0]:
                    return None
            fill = tirx.Evaluate(tirx.Call(stmt.value.dtype, stmt.value.op, list(args) + [_i32(0), _i32(1)]))
            sync = tirx.Evaluate(tirx.Call("int32", _op("tl.sync_grid"), []))
            notes.append(f"{dst.buffer.name}: MULTICAST (serial, worker ownership unchanged)")
            return tirx.SeqStmt([sync, fill, sync])

        def deal_fill_loop(stmt):
            """A loop whose whole body is ONE unsplittable DDR -> shared fill writing a different row each
            iteration: deal the ITERATIONS over the tile's SHAVEs and meet once after the loop.

            The per-statement rule below cannot see the loop, so it gives every iteration to SHAVE 0 between two
            barriers -- a gather of MB token rows became MB serialized DMAs on one SHAVE and 2 x MB tile barriers,
            with the other SHAVE only waiting. Iterations that write disjoint rows have no hazard between them, so
            the only barriers needed are one before (WAR against the previous contents) and one after (RAW)."""
            if shave_branch or pipelined_depth[0] > 0:
                return None
            if stmt.kind != tirx.ForKind.SERIAL or not isinstance(stmt.extent, tirx.IntImm):
                return None
            if int(stmt.extent) < launch.shaves:
                return None
            body = stmt.body
            while isinstance(body, tirx.SeqStmt) and len(body.seq) == 1:
                body = body.seq[0]
            if not _is_call(body, DMA_COPY):
                return None
            args = body.value.args
            if _const_or(args[6], 0) != 1:
                return None  # a store
            dst = _load_of(args[0])
            if dst.buffer.scope() == "local":
                return None  # SHAVE-private: nothing to share
            if _const_or(args[3], 0) % launch.shaves == 0:
                return None  # the row split already uses both SHAVEs
            if not any(_uses(i, stmt.loop_var) for i in dst.indices):
                return None  # every iteration writes the same rows: a real serial chain
            if _uses(body.value, launch.shave_var):
                return None  # the author already chose which SHAVE fills
            notes.append(
                f"{dst.buffer.name}: unsplittable fill, the loop's {int(stmt.extent)} iterations dealt "
                f"over the tile's SHAVEs, tile sync before and after the loop"
            )
            guarded = tirx.IfThenElse(tirx.floormod(stmt.loop_var, _i32(launch.shaves)) == launch.shave_var, body, None)
            loop = tirx.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind, guarded, stmt.thread_binding, stmt.annotations)
            return tirx.SeqStmt([_storage_sync_shared(), loop, _storage_sync_shared()])

        def pre(stmt):
            if isinstance(stmt, tirx.For):
                planned = plan_overlap(stmt)
                if planned is not None:
                    return planned
                dealt = deal_fill_loop(stmt)
                if dealt is not None:
                    return dealt
                uniform.append(uniform[-1] + [stmt.loop_var] if uniform[-1] and is_uniform(stmt.min) and is_uniform(stmt.extent) else [])
                loop_depth[0] += 1
                if int(stmt.annotations.get("num_stages", 0)) > 0:
                    pipelined_depth[0] += 1
            elif isinstance(stmt, tirx.IfThenElse):
                uniform.append(uniform[-1] if uniform[-1] and is_uniform(stmt.condition) else [])
                if _uses(stmt.condition, launch.shave_var):
                    shave_branch.append(False)
            elif isinstance(stmt, tirx.While):
                uniform.append([])
            return None

        def post(stmt):
            if isinstance(stmt, tirx.For):
                uniform.pop()
                loop_depth[0] -= 1
                if int(stmt.annotations.get("num_stages", 0)) > 0:
                    pipelined_depth[0] -= 1
                return None
            if isinstance(stmt, tirx.IfThenElse):
                uniform.pop()
                if _uses(stmt.condition, launch.shave_var) and shave_branch.pop():
                    # the fill ran on one SHAVE; the tile still has to meet, and the meeting point is HERE,
                    # after the branch, where every SHAVE arrives
                    return tirx.SeqStmt([stmt, _storage_sync_shared()])
                return None
            if isinstance(stmt, tirx.While):
                uniform.pop()
                return None
            if not _is_call(stmt, DMA_COPY):
                return None
            args = stmt.value.args
            if _const(args[6]) != 1:
                return None  # a store
            dst = _load_of(args[0])
            src = _load_of(args[1])
            dst_scope = dst.buffer.scope()
            src_expr = src.indices
            src_tile = any(_uses(i, launch.tile_var) for i in src_expr)
            src_shave = any(_uses(i, launch.shave_var) for i in src_expr)
            if dst_scope == "local":
                if pipelined_depth[0] == 0:
                    fill = multicast_fill(stmt)
                    if fill is not None:
                        return fill
                if not src_tile and not src_shave and pipelined_depth[0] == 0:
                    # The same DDR bytes read once per worker (a bias, a probe's operand): legal, and
                    # said in the source so a large one is noticed; the flow's operands are shared.
                    loop_vars = [v for i in src_expr for v in _vars_of(i)]
                    if not loop_vars:
                        notes.append(
                            f"{dst.buffer.name}: tile-invariant source copied into a SHAVE-private "
                            f"buffer on every worker (make it shared to read it once per tile)"
                        )
                return None
            if pipelined_depth[0] > 0:
                if src_tile:
                    # Each tile reads its own disjoint chunks: a split stream (one unicast fill per tile), legal by
                    # the rule (split or broadcast, never copied per tile); the weights of an N-split projection.
                    notes.append(f"{dst.buffer.name}: SEGMENTED (streamed, each tile fills its own slot)")
                else:
                    notes.append(f"{dst.buffer.name}: DUPLICATED (streamed, one multicast per chunk)")
                return None
            # Resident SEGMENTED (or tile-invariant) fill into a tile-shared buffer: split the rows
            # over the tile's SHAVEs, one per DMA port, then meet.
            if shave_branch:
                # the author already chose which SHAVE fills, so there is nothing to split, and the sync cannot
                # go here because the other SHAVE never reaches it: the enclosing branch emits it after itself
                shave_branch[-1] = True
                notes.append(f"{dst.buffer.name}: filled under a SHAVE-dependent branch, tile sync after the branch")
                return None
            rows = _const(args[3])
            spt = launch.shaves
            tx = launch.shave_var
            kind = "SEGMENTED" if src_tile else "tile-invariant"
            if rows % spt != 0:
                # Too few rows to share (a decode activation row): SHAVE 0 does the whole fill, the tile meets.
                fill = tirx.IfThenElse(tx == 0, stmt, None)
                notes.append(f"{dst.buffer.name}: {kind} resident, {rows} row(s) filled by SHAVE 0, tile sync after")
                return tirx.SeqStmt(([_storage_sync_shared()] if loop_depth[0] > 0 else []) + [fill, _storage_sync_shared()])
            per = rows // spt
            dst_idx = list(dst.indices)
            src_idx = list(src.indices)
            # the rows of a 2-D region are its second-to-last index (a [g, row, col] cache source, not only [row, col])
            dst_idx[-2 if len(dst_idx) >= 2 else -1] = dst_idx[-2 if len(dst_idx) >= 2 else -1] + tx * per
            src_idx[-2 if len(src_idx) >= 2 else -1] = src_idx[-2 if len(src_idx) >= 2 else -1] + tx * per
            new_args = [_address_of(dst.buffer, dst_idx), _address_of(src.buffer, src_idx), args[2], _i32(per), args[4], args[5], args[6]]
            fill = tirx.Evaluate(tirx.Call(stmt.value.dtype, stmt.value.op, new_args))
            if loop_depth[0] > 0:
                # Refilled in a loop: the other SHAVE may still read the previous contents (WAR).
                notes.append(
                    f"{dst.buffer.name}: {'SEGMENTED' if src_tile else 'tile-invariant'} resident in a loop, "
                    f"{rows} rows split {per} per SHAVE, tile sync before and after"
                )
                return tirx.SeqStmt([_storage_sync_shared(), fill, _storage_sync_shared()])
            notes.append(
                f"{dst.buffer.name}: {'SEGMENTED' if src_tile else 'tile-invariant'} resident, "
                f"{rows} rows split {per} per SHAVE, tile sync after"
            )
            return tirx.SeqStmt([fill, _storage_sync_shared()])

        body = ir_transform(func.body, pre, post, ["tirx.For", "tirx.IfThenElse", "tirx.While", "tirx.Evaluate"])
        if overlap_buffers:

            def mapped(buffer):
                return next((new for old, new in overlap_buffers if old.same_as(buffer)), buffer)

            def rebuffer(node):
                if isinstance(node, tirx.BufferLoad):
                    buffer = mapped(node.buffer)
                    if not buffer.same_as(node.buffer):
                        return tirx.BufferLoad(buffer, node.indices)
                elif isinstance(node, tirx.SBlock):
                    return tirx.SBlock(
                        node.iter_vars,
                        node.reads,
                        node.writes,
                        node.name_hint,
                        node.body,
                        node.init,
                        [mapped(buffer) for buffer in node.alloc_buffers],
                        node.match_buffers,
                        node.annotations,
                    )
                return None

            body = ir_transform(body, None, rebuffer, ["tirx.BufferLoad", "tirx.SBlock"])
        func = func.with_body(body)
        if notes:
            func = _with_notes(func, "distribution: " + "; ".join(notes))
        return func

    return _pass


# ---------------------------------------------------------------------------------------------
# LowerIntelNpuPipelined
# ---------------------------------------------------------------------------------------------

_WIDENED: dict = {}  # hash(data var) -> the widened Buffer; one object per CMX buffer however many rings use it


def _widen(buf, factor: int):
    """`buf` with its leading extent x factor (slots, output parities) on the same data var. Rings one after
    another over the same source buffer get the SAME Buffer object, so the allocation planner sees one buffer."""
    key = hash(buf.data)
    hit = _WIDENED.get(key)
    if hit is not None:
        if int(hit.shape[0]) != int(buf.shape[0]) * factor:
            raise ValueError(
                f"intel_npu: {buf.name} is widened x{factor} by one ring and x{int(hit.shape[0]) // int(buf.shape[0])} "
                "by another; give each pipelined loop its own buffer"
            )
        return hit
    new = tirx.decl_buffer([buf.shape[0] * factor] + list(buf.shape[1:]), buf.dtype, buf.name, data=buf.data, scope=buf.scope())
    _WIDENED[key] = new
    return new


class _Stream:
    """One streamed operand of a pipelined loop: its fill, slot buffer and the gemm that reads it."""

    def __init__(self, fill: tirx.Evaluate, stages: int):
        fa = fill.value.args
        self.fill = fill
        self.slot_buf = _load_of(fa[0]).buffer
        if self.slot_buf.scope() not in ("shared", "shared.dyn"):
            raise ValueError(f"intel_npu: the streamed buffer {self.slot_buf.name} must be shared (one ring per tile)")
        if _const(fa[2]) != _const(fa[4]):
            raise ValueError(
                f"intel_npu: the streamed fill into {self.slot_buf.name} is not contiguous in the slot (width {fa[2]}, "
                f"slot row stride {fa[4]}); allocate the slot buffer at the chunk's shape"
            )
        self.src = fa[1]
        self.width, self.rows, self.src_stride = _const(fa[2]), _const(fa[3]), _const(fa[5])
        self.slot_bytes = self.width * self.rows
        self.slots = _widen(self.slot_buf, stages)
        self.ring = tirx.Var("ring", "handle")
        self.gemm = None  # the one gemm that reads the slot (its last reader produces FREE)
        self.stream_is_a = None

    def bind_gemm(self, gemm: tirx.Evaluate):
        da = gemm.value.args
        a_buf, bt_buf = _load_of(da[1]).buffer, _load_of(da[2]).buffer
        is_a, is_bt = a_buf.same_as(self.slot_buf), bt_buf.same_as(self.slot_buf)
        if not (is_a or is_bt):
            return False
        if self.gemm is not None:
            raise ValueError(
                f"intel_npu: the streamed buffer {self.slot_buf.name} is read by two gemms in one "
                f"pipelined loop; the slot has one DPU reader (its FREE producer)"
            )
        if is_a and is_bt:
            raise ValueError(f"intel_npu: a gemm reads {self.slot_buf.name} as both operands")
        self.gemm, self.stream_is_a = gemm, is_a
        return True


class _Pipeline:
    """A T.Pipelined loop classified into streams, gemms, stores and other (SHAVE) statements."""

    def __init__(self, loop: tirx.For, launch: _Launch, outer: tirx.For | None = None):
        self.loop = loop
        self.outer = outer  # the enclosing serial loop the ring is hoisted over, if any
        self.c = loop.loop_var
        # The chunk count may be a runtime value (a decode step streams the cache up to its position, not
        # SMAX): it must be the same on every SHAVE of every tile (the barrier counts are shared), which a
        # value read from tile-shared CMX is. The runtime ignores fills / uses past it.
        self.total = _as_i32(loop.extent)
        outer_n = _const(outer.extent) if outer is not None else 1
        if isinstance(loop.extent, tirx.IntImm):
            self.total_chunks = _i32(int(loop.extent.value) * outer_n)
            self.total_text = str(int(loop.extent.value) * outer_n)
        else:
            self.total_chunks = self.total * _i32(outer_n)
            self.total_text = f"({loop.extent})" + (f" x {outer_n}" if outer is not None else "")
        self.stages = int(loop.annotations["num_stages"])
        if _const(loop.min) != 0:
            raise ValueError("intel_npu: a pipelined loop must start at 0")
        if self.stages % launch.shaves != 0 or self.stages == 0:
            raise ValueError(
                f"intel_npu: T.Pipelined(num_stages={self.stages}) on a tile with {launch.shaves} SHAVEs: the "
                f"ring depth must be a multiple of the SHAVEs per tile (SHAVE s owns slots s, s + {launch.shaves}, "
                f"...; README section 3)"
            )
        self.depth = self.stages // launch.shaves  # slots per SHAVE: fills issued this many uses ahead
        if launch.tiles < 2:
            raise ValueError(
                "intel_npu: the hardware ring needs at least 2 tiles (one tile has no multicast and a barrier "
                "pool of 8 ids, 3 of them sync_grid's); launch more tiles or write a serial loop"
            )
        body = loop.body
        self.stmts = list(body.seq) if isinstance(body, tirx.SeqStmt) else [body]
        fills = [s for s in self.stmts if _is_call(s, DMA_COPY) and _const(s.value.args[6]) == 1]
        if not fills:
            raise ValueError("intel_npu: a pipelined loop has no DDR -> CMX fill to stream")
        self.streams = [_Stream(f, self.stages) for f in fills]
        # Ring-submitted gemms read a slot through the DPU FIFO; any other reader of a slot (a SHAVE loop, a
        # synchronous DPU op such as gemm_i4) goes through acquire/release below.
        self.gemms = [s for s in self.stmts if _is_call(s, DPU_MATMUL)]
        for g in self.gemms:
            readers = [st for st in self.streams if st.bind_gemm(g)]
            if len(readers) > 1:
                raise ValueError("intel_npu: a gemm reads two streamed operands; one slot per gemm is lowered")
        for st in self.streams:
            if st.gemm is not None:
                continue
            # No gemm reads the slot: its readers are SHAVE statements (a dequant, a norm). The SHAVE acquires
            # the slot (waits READY) before the first and releases it (produces FREE) after the last.
            readers = [
                i
                for i, s in enumerate(self.stmts)
                if not (_is_call(s, DMA_COPY) and _const(s.value.args[6]) == 1) and _reads_buffer(s, st.slot_buf)
            ]
            if not readers:
                raise ValueError(f"intel_npu: the streamed buffer {st.slot_buf.name} is filled but nothing in the loop reads it")
            st.first_reader, st.last_reader = readers[0], readers[-1]
        self.shave_streams = [st for st in self.streams if st.gemm is None]
        self.stores = [s for s in self.stmts if _is_call(s, DMA_COPY) and _const(s.value.args[6]) == 0]
        others = [s for s in self.stmts if not (_is_call(s, DMA_COPY) or _is_call(s, DPU_MATMUL))]
        # The simple body (one stream, its gemm, stores) gets the submit-ahead program.
        self.simple = len(self.streams) == 1 and len(self.gemms) == 1 and bool(self.stores) and not others and not self.shave_streams
        for st in self.stores:
            if _load_of(st.value.args[1]).buffer.scope() != "local":
                raise ValueError("intel_npu: a store in a pipelined loop must read a SHAVE-local buffer")


def LowerIntelNpuPipelined():
    @prim_func_pass(opt_level=0, name="tl.LowerIntelNpuPipelined")
    def _pass(func: tirx.PrimFunc, mod, ctx):
        launch = _Launch(func)
        remap = []  # (old Buffer, new Buffer) shape changes
        _WIDENED.clear()
        notes = []
        spt = launch.shaves
        tx = launch.shave_var

        def chunk(use):
            return use * spt + tx

        def inner_chunk(p, use):
            # the pipelined loop's own index of global chunk `use * spt + tx` (hoisted: modulo the inner extent);
            # truncating division: both operands are non-negative, and a runtime extent would otherwise cost the
            # SHAVE the signed floordiv fix-up on every fill address
            return tirx.truncmod(chunk(use), p.total) if p.outer is not None else chunk(use)

        def fill(st, use):
            # the source of global chunk `use * spt + tx`: the pipelined index is that chunk modulo the inner extent
            # and (hoisted) the outer index is its quotient, so a fill issued ahead reads the next iteration's rows
            sub = {st.pipe_c: inner_chunk(st.pipe, use)}
            if st.pipe.outer is not None:
                sub[st.pipe.outer.loop_var] = tirx.floordiv(chunk(use), st.pipe.total)
            src = substitute(st.src, sub)
            return tirx.Evaluate(
                tirx.Call("handle", _op(RING_FILL), [st.ring, use, src, _i32(st.width), _i32(st.rows), _i32(st.src_stride)])
            )

        def submit(st, use, out_addr):
            da = list(st.gemm.value.args)
            slot = tirx.Call("handle", _op(RING_SLOT), [st.ring, use])
            a = slot if st.stream_is_a else da[1]
            bt = da[2] if st.stream_is_a else slot
            args = [st.ring, use, out_addr, a, bt] + list(da[3:8])
            return tirx.Evaluate(tirx.Call("handle", _op(RING_SUBMIT), args))

        def wait(st, use):
            return tirx.Evaluate(tirx.Call("handle", _op(RING_WAIT), [st.ring, use]))

        def acquire(st, use):
            return tirx.Evaluate(tirx.Call("handle", _op(RING_ACQUIRE), [st.ring, use]))

        def release(st, use):
            return tirx.Evaluate(tirx.Call("handle", _op(RING_RELEASE), [st.ring, use]))

        def slot_rows(st, p, use):
            # leading-index offset of this SHAVE's slot for `use` within the slots buffer
            return (tx + spt * tirx.floormod(use, _i32(p.depth))) * _i32(int(st.slot_buf.shape[0]))

        def begin(st, p):
            # A stream whose source reads the tile is split: every tile fills its own slot (no multicast). A split
            # stream the SHAVE reads is LOCAL: the SHAVE that fills the slot is its only reader, so the fill's DMA
            # watermark is READY and program order is FREE -- no hardware barriers, no coupling between tiles.
            multicast = 0 if _uses(st.src, launch.tile_var) else 1
            local = 1 if (multicast == 0 and st.gemm is None) else 0
            st.local = local
            return tirx.Bind(
                st.ring,
                tirx.Call(
                    "handle",
                    _op(RING_BEGIN),
                    [
                        _address_of(st.slots, [_i32(0)] * len(st.slots.shape)),
                        _i32(st.slot_bytes),
                        _i32(p.stages),
                        p.total_chunks,
                        _i32(-1),
                        _i32(-1),
                        _i32(st.port),
                        _i32(multicast),
                        _i32(local),
                    ],
                ),
            )

        def assign_ports(pipes):
            """DMA ports for the streams of the rings live together. The runtime takes one job in flight per port
            per SHAVE (a second start on a port waits for the first), so a ring whose SHAVE owns two or more
            slots alternates its fills between the ports by use (port 2 = use & 1, 3 = the opposite phase,
            dealt alternately to such streams), which is what lets its fills overlap. Of the depth-1 streams,
            one alone is split in two halves (port -1, the halves' fixed costs overlap across the ports);
            several are dealt whole to the port with the smaller load so far, largest first, where a port's
            load is the serial sum of its jobs' costs, `fixed + bytes / rate` (plan 7.4: ~2.5 us + bytes /
            70 GB/s per port under turbo; 72 scale fills of 8 KiB cost as much as 72 x 175 KiB)."""
            streams = [st for p in pipes for st in p.streams]
            deep = [st for st in streams if st.pipe.depth >= 2]
            for i, st in enumerate(deep):
                st.port = 2 + i % 2
            shallow = [st for st in streams if st.pipe.depth < 2]
            if len(shallow) == 1 and not deep:
                shallow[0].port = -1
                return
            fixed_bytes = 175 * 1024  # 2.5 us at 70 GB/s
            load = [0, 0]
            for st in deep:
                for port in (0, 1):
                    load[port] += _const_or(st.pipe.total_chunks, 1) * (st.slot_bytes + fixed_bytes) // 2
            for st in sorted(shallow, key=lambda s: -(s.slot_bytes)):
                st.port = 0 if load[0] <= load[1] else 1
                load[st.port] += _const_or(st.pipe.total_chunks, 1) * (st.slot_bytes + fixed_bytes)

        def end(st):
            return tirx.Evaluate(tirx.Call("handle", _op(RING_END), [st.ring]))

        def use_loop(p, u, body):
            """This SHAVE's uses: all of them, or (hoisted) those of one outer iteration, as a loop over `u`
            with the body written against the global use `g`."""
            if p.outer is None:
                uses = tirx.floordiv(p.total - tx + _i32(spt - 1), _i32(spt))
                return tirx.For(u, _i32(0), uses, tirx.ForKind.SERIAL, body(u))
            per = tirx.truncdiv(p.total, _i32(spt))  # hoisted only when the extent is a multiple of spt
            return tirx.For(u, _i32(0), per, tirx.ForKind.SERIAL, body(p.outer.loop_var * per + u))

        def lower_simple(p: _Pipeline):
            # One stream, its gemm, stores: submit u+1 right after u completes, output double-buffered by parity.
            st = p.streams[0]
            out_buf = _load_of(st.gemm.value.args[0]).buffer
            if out_buf.scope() != "local":
                raise ValueError(f"intel_npu: the gemm output {out_buf.name} in a pipelined loop must be SHAVE-local")
            M = _const(st.gemm.value.args[3])
            outs = _widen(out_buf, 2)
            remap.append((out_buf, outs))
            u = tirx.Var("u", "int32")

            def out_addr(parity):
                idx = list(_load_of(st.gemm.value.args[0]).indices)
                idx[0] = idx[0] + parity * M
                return _address_of(outs, idx)

            def store(s, use, parity):
                sa = list(s.value.args)
                sidx = list(_load_of(sa[1]).indices)
                sidx[0] = sidx[0] + parity * M
                sa[1] = _address_of(outs, sidx)
                return substitute(tirx.Evaluate(tirx.Call(s.value.dtype, s.value.op, sa)), {p.c: inner_chunk(p, use)})

            def loop_body(g):
                return tirx.SeqStmt(
                    [wait(st, g), submit(st, g + 1, out_addr(tirx.floormod(g + 1, 2))), fill(st, g + p.depth)]
                    + [store(s, g, tirx.floormod(g, 2)) for s in p.stores]
                )

            notes.append(
                f"ring over {p.c.name} ({p.total_text} chunks of {st.slot_bytes} B into {st.slot_buf.name}"
                f"[{p.stages} slots], SHAVE s owns {p.depth} slot(s), fills {p.depth} use(s) ahead, output "
                f"{out_buf.name} x2 by use parity, submit-ahead; issuer tile 0, two halves on ports 0/1)"
            )
            prologue = [begin(st, p)] + [fill(st, _i32(k)) for k in range(p.depth)] + [submit(st, _i32(0), out_addr(_i32(0)))]
            return prologue, use_loop(p, u, loop_body), [end(st)]

        def lower_general(p: _Pipeline):
            # Several streams and SHAVE statements between the gemms: the body runs in the author's order
            # for each of this SHAVE's chunks; a gemm reading a slot is submit + wait on that slot's ring,
            # and the issuer refills the slot for the next use right after (the gemm is the slot's last
            # reader, so its FREE is the refill's gate). No submit-ahead: the SHAVE code between the
            # gemms depends on the gemm's output.
            u = tirx.Var("u", "int32")

            def loop_body(g):
                seq = []
                for i, s in enumerate(p.stmts):
                    if _is_call(s, DMA_COPY) and _const(s.value.args[6]) == 1:
                        continue  # fills are issued by the ring, after the previous use's last reader
                    for st in p.shave_streams:
                        if st.first_reader == i:
                            seq.append(acquire(st, g))
                    hit = [st for st in p.streams if st.gemm is not None and st.gemm.same_as(s)]
                    if hit:
                        st = hit[0]
                        seq += [submit(st, g, st.gemm.value.args[0]), wait(st, g), fill(st, g + p.depth)]
                    else:
                        s2 = s
                        for st in p.shave_streams:
                            if st.first_reader <= i <= st.last_reader:
                                s2 = _offset_loads(s2, st.slot_buf, slot_rows(st, p, g))
                        seq.append(substitute(s2, {p.c: inner_chunk(p, g)}))
                    for st in p.shave_streams:
                        if st.last_reader == i:
                            seq += [release(st, g), fill(st, g + p.depth)]
                return tirx.SeqStmt(seq)

            prologue = [begin(st, p) for st in p.streams] + [fill(st, _i32(k)) for k in range(p.depth) for st in p.streams]
            kinds = f"{len(p.gemms)} gemm(s)" + (f", {len(p.shave_streams)} slot(s) read by the SHAVE" if p.shave_streams else "")
            notes.append(
                f"ring over {p.c.name} ({p.total_text} chunks; streams "
                + ", ".join(f"{st.slot_buf.name}[{p.stages} slots, {st.slot_bytes} B]" for st in p.streams)
                + f"; SHAVE s owns {p.depth} slot(s), fills {p.depth} use(s) ahead; {kinds} with SHAVE "
                f"statements between, no submit-ahead)"
            )
            return prologue, use_loop(p, u, loop_body), [end(st) for st in p.streams]

        def is_pipelined(s):
            return isinstance(s, tirx.For) and int(s.annotations.get("num_stages", 0)) > 0

        _pipes = []

        def make_pipe(stmt, outer):
            p = _Pipeline(stmt, launch, outer)
            _pipes.append(p)
            for st in p.streams:
                st.pipe_c = p.c
                st.pipe = p
                remap.append((st.slot_buf, st.slots))
            return p

        def lower_pipe(p):
            return lower_simple(p) if p.simple else lower_general(p)

        def _written_in(stmt):
            """Buffers a statement writes: BufferStores and the destinations of DDR -> CMX copies."""
            found = []

            def visit(node):
                if isinstance(node, tirx.BufferStore):
                    found.append(node.buffer)
                elif isinstance(node, tirx.Call) and node.op.same_as(_op(DMA_COPY)) and _const(node.args[6]) == 1:
                    found.append(_load_of(node.args[0]).buffer)

            post_order_visit(stmt, visit)
            return found

        def _index_reads(src):
            """Buffers read inside the index of a fill's DDR source (an expert id, a token's row)."""
            found = []

            def visit(node):
                if isinstance(node, tirx.BufferLoad):
                    found.append(node.buffer)

            for i in _load_of(src).indices:
                post_order_visit(i, visit)
            return found

        def _let_bound_in(stmt):
            """Vars a statement binds (`tirx.Bind`): a `T.ddr_view` inside a loop body names that iteration's DDR
            tensor through such a binding, so a fill reading the view is per-iteration."""
            found = []

            def visit(node):
                if isinstance(node, tirx.Bind):
                    found.append(node.var)

            post_order_visit(stmt, visit)
            return found

        def hoistable(outer):
            """The pipelined loops directly inside a serial loop whose rings can run over all its iterations: a
            fill issued ahead then reads the next iteration's chunk, which is sound when nothing the outer body
            writes feeds the fill's source index (an index read from a buffer the iteration computes is not) and
            the fill's source tensor is not a view bound inside the iteration."""
            if not isinstance(outer, tirx.For) or outer.kind != tirx.ForKind.SERIAL or is_pipelined(outer):
                return []
            if not isinstance(outer.extent, tirx.IntImm):
                return []  # a runtime trip count (an expert's token chunks): one ring per iteration
            stmts = list(outer.body.seq) if isinstance(outer.body, tirx.SeqStmt) else [outer.body]
            written = _written_in(outer.body)
            bound = _let_bound_in(outer.body)
            hoisted = []
            for inner in (s for s in stmts if is_pipelined(s)):
                if int(inner.annotations.get("intel_npu.no_hoist", 0)):
                    continue
                fills = [
                    s
                    for s in (list(inner.body.seq) if isinstance(inner.body, tirx.SeqStmt) else [inner.body])
                    if _is_call(s, DMA_COPY) and _const(s.value.args[6]) == 1
                ]
                if not fills or not _divisible(inner.extent, spt):
                    continue  # uses would straddle outer iterations unevenly; one ring per iteration instead
                if any(any(w.same_as(b) for w in written) for f in fills for b in _index_reads(f.value.args[1])):
                    notes.append(
                        f"ring over {inner.loop_var.name} not hoisted over {outer.loop_var.name}: a fill's "
                        f"source index reads a buffer the iteration writes"
                    )
                    continue
                if any(any(v.same_as(_load_of(f.value.args[1]).buffer.data) for v in bound) for f in fills):
                    notes.append(
                        f"ring over {inner.loop_var.name} not hoisted over {outer.loop_var.name}: a fill's "
                        f"source is a DDR view bound inside the iteration"
                    )
                    continue
                hoisted.append(inner)
            return hoisted

        def pre(stmt):
            inners = hoistable(stmt)
            if not inners:
                return None
            # One ring per hoisted loop over (outer x inner): begin and the first fills before the outer loop,
            # the use loop in the inner loop's place, the ends after. A fill's source may read the outer
            # variable: the fill ahead substitutes the chunk's own iteration (see `fill`).
            prologue, ends, loops = [], [], {}
            pipes = [make_pipe(inner, stmt) for inner in inners]
            assign_ports(pipes)
            for inner, p in zip(inners, pipes):
                pro, loop, end = lower_pipe(p)
                prologue += pro
                ends += end
                loops[inner] = loop
                reads_outer = any(_uses(st.src, stmt.loop_var) for st in p.streams)
                notes.append(
                    f"ring over {inner.loop_var.name} hoisted over {stmt.loop_var.name} "
                    f"({_const(stmt.extent)} iterations: one begin/end" + (", fills read the iteration" if reads_outer else "") + ")"
                )
            stmts = list(stmt.body.seq) if isinstance(stmt.body, tirx.SeqStmt) else [stmt.body]
            replaced = [next((l for k, l in loops.items() if s.same_as(k)), s) for s in stmts]
            new_body = replaced[0] if len(replaced) == 1 else tirx.SeqStmt(replaced)
            outer = tirx.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind, new_body, stmt.thread_binding, stmt.annotations)
            return tirx.SeqStmt(prologue + [outer] + ends)

        def lower(stmt):
            if not is_pipelined(stmt):
                return None
            p = make_pipe(stmt, None)
            assign_ports([p])
            prologue, loop, ends = lower_pipe(p)
            return tirx.SeqStmt(prologue + [loop] + ends)

        body = ir_transform(func.body, pre, lower, ["tirx.For"])
        if not remap:
            return func

        def mapped(buf):
            for old, new in remap:
                if buf.same_as(old):
                    return new
            return buf

        def rebuf(node):
            if isinstance(node, tirx.BufferLoad):
                new = mapped(node.buffer)
                if not new.same_as(node.buffer):
                    return tirx.BufferLoad(new, node.indices)
            if isinstance(node, tirx.SBlock):
                allocs = [mapped(b) for b in node.alloc_buffers]
                if any(not a.same_as(b) for a, b in zip(allocs, node.alloc_buffers)):
                    return tirx.SBlock(
                        node.iter_vars,
                        node.reads,
                        node.writes,
                        node.name_hint,
                        node.body,
                        node.init,
                        allocs,
                        node.match_buffers,
                        node.annotations,
                    )
            return None

        body = ir_transform(body, None, rebuf, ["tirx.BufferLoad", "tirx.SBlock"])
        func = func.with_body(body)
        return _with_notes(func, "pipelined: " + "; ".join(notes))

    return _pass
