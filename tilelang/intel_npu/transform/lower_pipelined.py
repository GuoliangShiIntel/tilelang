"""PlanIntelNpuDistribution and LowerIntelNpuPipelined: the multi-tile flow from the author's loops.

docs/examples/pingpong_multitile/README.md sections 1-3, as the plan's section 3.2 orders it.
Both passes run after LowerIntelNpuTileOp (copies are `intel_npu_dma_copy`, gemms are
`intel_npu_dpu_matmul`) and before LowerOpaqueBlock (allocations are still SBlock buffers).

PlanIntelNpuDistribution derives, for every DDR -> CMX copy, how it is distributed:
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

        def pre(stmt):
            if isinstance(stmt, tirx.For):
                loop_depth[0] += 1
                if int(stmt.annotations.get("num_stages", 0)) > 0:
                    pipelined_depth[0] += 1
            return None

        def post(stmt):
            if isinstance(stmt, tirx.For):
                loop_depth[0] -= 1
                if int(stmt.annotations.get("num_stages", 0)) > 0:
                    pipelined_depth[0] -= 1
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
                if not src_tile and not src_shave and pipelined_depth[0] == 0:
                    # The same DDR bytes read once per worker (a bias, a probe's operand): legal, and
                    # said in the source so a large one is noticed; the flow's operands are shared.
                    loop_vars = [v for i in src_expr for v in _vars_of(i)]
                    if not loop_vars:
                        notes.append(f"{dst.buffer.name}: tile-invariant source copied into a SHAVE-private "
                                     f"buffer on every worker (make it shared to read it once per tile)")
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
            rows = _const(args[3])
            spt = launch.shaves
            if rows % spt != 0:
                raise ValueError(
                    f"intel_npu: resident fill {src.buffer.name} -> {dst.buffer.name} has {rows} rows, not a "
                    f"multiple of the {spt} SHAVEs that share the load")
            per = rows // spt
            tx = launch.shave_var
            dst_idx = list(dst.indices)
            src_idx = list(src.indices)
            dst_idx[0] = dst_idx[0] + tx * per
            src_idx[0] = src_idx[0] + tx * per
            new_args = [_address_of(dst.buffer, dst_idx), _address_of(src.buffer, src_idx), args[2], _i32(per),
                        args[4], args[5], args[6]]
            fill = tirx.Evaluate(tirx.Call(stmt.value.dtype, stmt.value.op, new_args))
            if loop_depth[0] > 0:
                # Refilled in a loop: the other SHAVE may still read the previous contents (WAR).
                notes.append(f"{dst.buffer.name}: {'SEGMENTED' if src_tile else 'tile-invariant'} resident in a loop, "
                             f"{rows} rows split {per} per SHAVE, tile sync before and after")
                return tirx.SeqStmt([_storage_sync_shared(), fill, _storage_sync_shared()])
            notes.append(f"{dst.buffer.name}: {'SEGMENTED' if src_tile else 'tile-invariant'} resident, "
                         f"{rows} rows split {per} per SHAVE, tile sync after")
            return tirx.SeqStmt([fill, _storage_sync_shared()])

        body = ir_transform(func.body, pre, post, ["tirx.For", "tirx.Evaluate"])
        func = func.with_body(body)
        if notes:
            func = _with_notes(func, "distribution: " + "; ".join(notes))
        return func

    return _pass


# ---------------------------------------------------------------------------------------------
# LowerIntelNpuPipelined
# ---------------------------------------------------------------------------------------------

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
                f"slot row stride {fa[4]}); allocate the slot buffer at the chunk's shape")
        self.src = fa[1]
        self.width, self.rows, self.src_stride = _const(fa[2]), _const(fa[3]), _const(fa[5])
        self.slot_bytes = self.width * self.rows
        self.slots = tirx.decl_buffer([self.slot_buf.shape[0] * stages] + list(self.slot_buf.shape[1:]),
                                      self.slot_buf.dtype, self.slot_buf.name, data=self.slot_buf.data,
                                      scope=self.slot_buf.scope())
        self.ring = tirx.Var("ring", "handle")
        self.gemm = None        # the one gemm that reads the slot (its last reader produces FREE)
        self.stream_is_a = None

    def bind_gemm(self, gemm: tirx.Evaluate):
        da = gemm.value.args
        a_buf, bt_buf = _load_of(da[1]).buffer, _load_of(da[2]).buffer
        is_a, is_bt = a_buf.same_as(self.slot_buf), bt_buf.same_as(self.slot_buf)
        if not (is_a or is_bt):
            return False
        if self.gemm is not None:
            raise ValueError(f"intel_npu: the streamed buffer {self.slot_buf.name} is read by two gemms in one "
                             f"pipelined loop; the slot has one DPU reader (its FREE producer)")
        if is_a and is_bt:
            raise ValueError(f"intel_npu: a gemm reads {self.slot_buf.name} as both operands")
        self.gemm, self.stream_is_a = gemm, is_a
        return True


class _Pipeline:
    """A T.Pipelined loop classified into streams, gemms, stores and other (SHAVE) statements."""

    def __init__(self, loop: tirx.For, launch: _Launch, outer: tirx.For | None = None):
        self.loop = loop
        self.outer = outer   # the enclosing serial loop the ring is hoisted over, if any
        self.c = loop.loop_var
        self.total = _const(loop.extent)
        self.total_chunks = self.total * (_const(outer.extent) if outer is not None else 1)
        self.stages = int(loop.annotations["num_stages"])
        if _const(loop.min) != 0:
            raise ValueError("intel_npu: a pipelined loop must start at 0")
        if self.stages % launch.shaves != 0 or self.stages == 0:
            raise ValueError(
                f"intel_npu: T.Pipelined(num_stages={self.stages}) on a tile with {launch.shaves} SHAVEs: the "
                f"ring depth must be a multiple of the SHAVEs per tile (SHAVE s owns slots s, s + {launch.shaves}, "
                f"...; README section 3)")
        self.depth = self.stages // launch.shaves   # slots per SHAVE: fills issued this many uses ahead
        if launch.tiles < 2:
            raise ValueError(
                "intel_npu: the hardware ring needs at least 2 tiles (one tile has no multicast and a barrier "
                "pool of 8 ids, 3 of them sync_grid's); launch more tiles or write a serial loop")
        body = loop.body
        self.stmts = list(body.seq) if isinstance(body, tirx.SeqStmt) else [body]
        fills = [s for s in self.stmts if _is_call(s, DMA_COPY) and _const(s.value.args[6]) == 1]
        if not fills:
            raise ValueError("intel_npu: a pipelined loop has no DDR -> CMX fill to stream")
        self.streams = [_Stream(f, self.stages) for f in fills]
        self.gemms = [s for s in self.stmts if _is_call(s, DPU_MATMUL)]
        if not self.gemms:
            raise ValueError("intel_npu: a pipelined loop has no gemm reading its streamed operand")
        for g in self.gemms:
            readers = [st for st in self.streams if st.bind_gemm(g)]
            if len(readers) > 1:
                raise ValueError("intel_npu: a gemm reads two streamed operands; one slot per gemm is lowered")
        for st in self.streams:
            if st.gemm is not None:
                continue
            # No gemm reads the slot: its readers are SHAVE statements (a dequant, a norm). The SHAVE acquires
            # the slot (waits READY) before the first and releases it (produces FREE) after the last.
            readers = [i for i, s in enumerate(self.stmts)
                       if not (_is_call(s, DMA_COPY) and _const(s.value.args[6]) == 1) and _reads_buffer(s, st.slot_buf)]
            if not readers:
                raise ValueError(f"intel_npu: the streamed buffer {st.slot_buf.name} is filled but nothing in the loop "
                                 f"reads it")
            st.first_reader, st.last_reader = readers[0], readers[-1]
        self.shave_streams = [st for st in self.streams if st.gemm is None]
        self.stores = [s for s in self.stmts if _is_call(s, DMA_COPY) and _const(s.value.args[6]) == 0]
        others = [s for s in self.stmts if not (_is_call(s, DMA_COPY) or _is_call(s, DPU_MATMUL))]
        # The simple body (one stream, its gemm, stores) gets the submit-ahead program.
        self.simple = (len(self.streams) == 1 and len(self.gemms) == 1 and bool(self.stores) and not others
                       and not self.shave_streams)
        for st in self.stores:
            if _load_of(st.value.args[1]).buffer.scope() != "local":
                raise ValueError("intel_npu: a store in a pipelined loop must read a SHAVE-local buffer")


def LowerIntelNpuPipelined():
    @prim_func_pass(opt_level=0, name="tl.LowerIntelNpuPipelined")
    def _pass(func: tirx.PrimFunc, mod, ctx):
        launch = _Launch(func)
        remap = []  # (old Buffer, new Buffer) shape changes
        notes = []
        spt = launch.shaves
        tx = launch.shave_var

        def chunk(use):
            return use * spt + tx

        def inner_chunk(p, use):
            # the pipelined loop's own index of global chunk `use * spt + tx` (hoisted: modulo the inner extent)
            return tirx.floormod(chunk(use), _i32(p.total)) if p.outer is not None else chunk(use)

        def fill(st, use):
            src = substitute(st.src, {st.pipe_c: inner_chunk(st.pipe, use)})
            return tirx.Evaluate(tirx.Call("handle", _op(RING_FILL),
                                           [st.ring, use, src, _i32(st.width), _i32(st.rows), _i32(st.src_stride)]))

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
            # one stream: each fill split in two halves over both ports; several streams: stream i whole on
            # port i % 2, so the streams' fills run concurrently (one job in flight per port per SHAVE).
            # A stream whose source reads the tile is split: every tile fills its own slot (no multicast).
            port = -1 if len(p.streams) == 1 else p.streams.index(st) % 2
            multicast = 0 if _uses(st.src, launch.tile_var) else 1
            return tirx.Bind(st.ring, tirx.Call("handle", _op(RING_BEGIN),
                                                [_address_of(st.slots, [_i32(0)] * len(st.slots.shape)),
                                                 _i32(st.slot_bytes), _i32(p.stages), _i32(p.total_chunks), _i32(-1),
                                                 _i32(-1), _i32(port), _i32(multicast)]))

        def end(st):
            return tirx.Evaluate(tirx.Call("handle", _op(RING_END), [st.ring]))

        def use_loop(p, u, body):
            """This SHAVE's uses: all of them, or (hoisted) those of one outer iteration, as a loop over `u`
            with the body written against the global use `g`."""
            if p.outer is None:
                uses = tirx.floordiv(_i32(p.total) - tx + _i32(spt - 1), _i32(spt))
                return tirx.For(u, _i32(0), uses, tirx.ForKind.SERIAL, body(u))
            per = p.total // spt
            return tirx.For(u, _i32(0), _i32(per), tirx.ForKind.SERIAL, body(p.outer.loop_var * per + u))

        def lower_simple(p: _Pipeline):
            # One stream, its gemm, stores: submit u+1 right after u completes, output double-buffered by parity.
            st = p.streams[0]
            out_buf = _load_of(st.gemm.value.args[0]).buffer
            if out_buf.scope() != "local":
                raise ValueError(f"intel_npu: the gemm output {out_buf.name} in a pipelined loop must be SHAVE-local")
            M = _const(st.gemm.value.args[3])
            outs = tirx.decl_buffer([out_buf.shape[0] * 2] + list(out_buf.shape[1:]), out_buf.dtype, out_buf.name,
                                    data=out_buf.data, scope=out_buf.scope())
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
                return tirx.SeqStmt([wait(st, g), submit(st, g + 1, out_addr(tirx.floormod(g + 1, 2))), fill(st, g + p.depth)]
                                    + [store(s, g, tirx.floormod(g, 2)) for s in p.stores])

            notes.append(f"ring over {p.c.name} ({p.total_chunks} chunks of {st.slot_bytes} B into {st.slot_buf.name}"
                         f"[{p.stages} slots], SHAVE s owns {p.depth} slot(s), fills {p.depth} use(s) ahead, output "
                         f"{out_buf.name} x2 by use parity, submit-ahead; issuer tile 0, two halves on ports 0/1)")
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
            kinds = (f"{len(p.gemms)} gemm(s)" + (f", {len(p.shave_streams)} slot(s) read by the SHAVE" if p.shave_streams else ""))
            notes.append(f"ring over {p.c.name} ({p.total_chunks} chunks; streams " +
                         ", ".join(f"{st.slot_buf.name}[{p.stages} slots, {st.slot_bytes} B]" for st in p.streams) +
                         f"; SHAVE s owns {p.depth} slot(s), fills {p.depth} use(s) ahead; {kinds} with SHAVE "
                         f"statements between, no submit-ahead)")
            return prologue, use_loop(p, u, loop_body), [end(st) for st in p.streams]

        def is_pipelined(s):
            return isinstance(s, tirx.For) and int(s.annotations.get("num_stages", 0)) > 0

        def lower_pipe(stmt, outer):
            p = _Pipeline(stmt, launch, outer)
            for st in p.streams:
                st.pipe_c = p.c
                st.pipe = p
                remap.append((st.slot_buf, st.slots))
            return lower_simple(p) if p.simple else lower_general(p)

        def hoistable(outer):
            """The one pipelined loop directly inside a serial loop whose variable no stream's fill reads."""
            if not isinstance(outer, tirx.For) or outer.kind != tirx.ForKind.SERIAL or is_pipelined(outer):
                return None
            stmts = list(outer.body.seq) if isinstance(outer.body, tirx.SeqStmt) else [outer.body]
            pipes = [s for s in stmts if is_pipelined(s)]
            if len(pipes) != 1:
                return None
            inner = pipes[0]
            fills = [s for s in (list(inner.body.seq) if isinstance(inner.body, tirx.SeqStmt) else [inner.body])
                     if _is_call(s, DMA_COPY) and _const(s.value.args[6]) == 1]
            if not fills or any(_uses(f.value.args[1], outer.loop_var) for f in fills):
                return None
            if _const(inner.extent) % spt != 0:
                return None  # uses would straddle outer iterations unevenly; one ring per iteration instead
            return inner

        def pre(stmt):
            inner = hoistable(stmt)
            if inner is None:
                return None
            # One ring over (outer x inner): begin and the first fills before the outer loop, the use loop in
            # the inner loop's place, the ends after. The fills' sources do not depend on the outer variable.
            prologue, loop, ends = lower_pipe(inner, stmt)
            stmts = list(stmt.body.seq) if isinstance(stmt.body, tirx.SeqStmt) else [stmt.body]
            replaced = [loop if s.same_as(inner) else s for s in stmts]
            new_body = replaced[0] if len(replaced) == 1 else tirx.SeqStmt(replaced)
            outer = tirx.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind, new_body, stmt.thread_binding,
                             stmt.annotations)
            notes.append(f"ring hoisted over {stmt.loop_var.name} ({_const(stmt.extent)} iterations: one begin/end)")
            return tirx.SeqStmt(prologue + [outer] + ends)

        def lower(stmt):
            if not is_pipelined(stmt):
                return None
            prologue, loop, ends = lower_pipe(stmt, None)
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
                    return tirx.SBlock(node.iter_vars, node.reads, node.writes, node.name_hint, node.body, node.init,
                                       allocs, node.match_buffers, node.annotations)
            return None

        body = ir_transform(body, None, rebuf, ["tirx.BufferLoad", "tirx.SBlock"])
        func = func.with_body(body)
        return _with_notes(func, "pipelined: " + "; ".join(notes))

    return _pass
