"""AssignIntelNpuBarriers: hand hardware barrier ids to every ring from the job's pool.

The job owns ids [barrier_pool_first, barriers_per_tile x launch tiles) (plan section 3.2,
measured in Phase 3a). The top three are sync_grid's rotating ids (codegen's convention);
each ring takes READY/FREE x slots x 2 parities = 4 x slots ids, READY first.

Ids are reused across phases: a ring is live from its `ring_begin` to its `ring_end` in program
order, and a ring that begins after another has ended takes that ring's block (interval
colouring, first fit). Rings live at the same time (two streams of one pipelined loop) never
share an id. Reuse is sound because `ring_end` leaves every id of the issuer's slot quiescent
(the last FREE produced by every tile's DPU and consumed; the READYs consumed before that) and
`ring_begin` re-arms the first use on the same SHAVE, behind a grid rendezvous. The pass refuses
by name, saying by how much, when the live rings do not fit.
"""

from __future__ import annotations

import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform, post_order_visit
from tvm.tirx.transform import prim_func_pass

from .lower_pipelined import NOTES_ATTR, RING_BEGIN, RING_END, _with_notes

SYNC_BARRIERS = 3
REUSE_KEY = "tl.intel_npu.barrier_reuse"   # False: every ring takes fresh ids (device A/B of the reuse)


def _launch_tiles(func: tirx.PrimFunc) -> int:
    tiles = [1]

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent" and node.node.thread_tag == "blockIdx.x":
            tiles[0] = int(node.value.value)

    post_order_visit(func.body, visit)
    return tiles[0]


def _live_ranges(body: tirx.Stmt, begin_op, end_op):
    """[(bind, begin_seq, end_seq)] in program order; a ring without an end stays live to the end."""
    rings, ends, seq = [], [], [0]

    def visit(node):
        if isinstance(node, tirx.Bind) and isinstance(node.value, tirx.Call) and node.value.op.same_as(begin_op):
            seq[0] += 1
            if int(node.value.args[8].value) == 0:   # a local ring (split, SHAVE-read) uses no ids
                rings.append((node, seq[0]))
        elif isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call) and node.value.op.same_as(end_op):
            seq[0] += 1
            ends.append((node.value.args[0], seq[0]))

    post_order_visit(body, visit)

    def end_of(var):
        return next((e for v, e in ends if v.same_as(var)), seq[0] + 1)

    return [(b, s, end_of(b.var)) for b, s in rings]


def _colour(ranges, first, reuse=True):
    """First-fit interval colouring over blocks of 4 x slots ids; returns ({id(bind): base}, next free id, peak live)."""
    events = []
    for b, s, e in ranges:
        events.append((s, 1, b))
        events.append((e, 0, b))
    events.sort(key=lambda t: (t[0], t[1]))   # an end at seq k frees before a begin at seq k
    free, assigned, size_of, nxt, live, peak = [], {}, {}, first, 0, 0
    for _, is_begin, b in events:
        need = 4 * int(b.value.args[2].value)
        if is_begin:
            live += 1
            peak = max(peak, live)
            for i, (base, size) in enumerate(free):
                if size == need and reuse:
                    assigned[id(b)] = base
                    del free[i]
                    break
            else:
                assigned[id(b)] = nxt
                nxt += need
            size_of[id(b)] = need
        else:
            live -= 1
            free.append((assigned[id(b)], size_of[id(b)]))
            free.sort()
    return assigned, nxt, peak


def AssignIntelNpuBarriers():
    @prim_func_pass(opt_level=0, name="tl.AssignIntelNpuBarriers")
    def _pass(func: tirx.PrimFunc, mod, ctx):
        op = tvm.ir.Op.get(RING_BEGIN)
        ranges = _live_ranges(func.body, op, tvm.ir.Op.get(RING_END))
        any_ring = [False]

        def seen(node):
            if isinstance(node, tirx.Bind) and isinstance(node.value, tirx.Call) and node.value.op.same_as(op):
                any_ring[0] = True

        post_order_visit(func.body, seen)
        if not any_ring[0]:
            return func
        rings = [b for b, _, _ in ranges]
        target = func.attrs["target"]
        first = int(target.attrs["barrier_pool_first"])
        per_tile = int(target.attrs["barriers_per_tile"])
        tiles = _launch_tiles(func)
        pool_end = per_tile * tiles
        usable_end = pool_end - SYNC_BARRIERS
        assigned, nxt, peak = _colour(ranges, first, bool(ctx.config.get(REUSE_KEY, True)))
        need = nxt - first
        if nxt > usable_end:
            raise ValueError(
                f"intel_npu: {len(rings)} ring(s), {peak} live at once, need {need} hardware barrier ids but the job's pool "
                f"[{first}, {pool_end}) minus sync_grid's {SYNC_BARRIERS} leaves {max(0, usable_end - first)}: "
                f"short by {nxt - usable_end}; launch more tiles or stream fewer operands at once")
        notes = []
        names = {}

        def rewrite(stmt):
            if isinstance(stmt, tirx.Bind) and isinstance(stmt.value, tirx.Call) and stmt.value.op.same_as(op):
                if int(stmt.value.args[8].value) == 1:
                    args = list(stmt.value.args)
                    args[4] = args[5] = tirx.IntImm("int32", 0)
                    n = names.get(stmt.var.name, 0)
                    names[stmt.var.name] = n + 1
                    notes.append(f"{stmt.var.name if n == 0 else f'{stmt.var.name}_{n}'}: local (no ids)")
                    return tirx.Bind(stmt.var, tirx.Call(stmt.value.dtype, op, args))
                for r in rings:
                    if r.same_as(stmt):
                        slots = int(stmt.value.args[2].value)
                        ready = assigned[id(r)]
                        free = ready + 2 * slots
                        args = list(stmt.value.args)
                        args[4] = tirx.IntImm("int32", ready)
                        args[5] = tirx.IntImm("int32", free)
                        n = names.get(stmt.var.name, 0)
                        names[stmt.var.name] = n + 1
                        label = stmt.var.name if n == 0 else f"{stmt.var.name}_{n}"
                        notes.append(f"{label}: READY {ready}..{ready + 2 * slots - 1}, FREE {free}..{free + 2 * slots - 1}")
                        return tirx.Bind(stmt.var, tirx.Call(stmt.value.dtype, op, args))
            return None

        body = ir_transform(func.body, None, rewrite, ["tirx.Bind"])
        func = func.with_body(body)
        reuse_on = bool(ctx.config.get(REUSE_KEY, True))
        reuse = "" if peak == len(rings) else (f"; {len(rings)} rings, {peak} live at once, {need} ids by reuse" if reuse_on
                                               else f"; {len(rings)} rings, {peak} live at once, {need} ids (reuse off)")
        return _with_notes(func, f"hardware barriers (pool [{first}, {pool_end}), sync_grid {usable_end}..{pool_end - 1}): "
                                 + "; ".join(notes) + reuse)

    return _pass
