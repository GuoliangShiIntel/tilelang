"""AssignIntelNpuBarriers: hand hardware barrier ids to every ring from the job's pool.

The job owns ids [barrier_pool_first, barriers_per_tile x launch tiles) (plan section 3.2,
measured in Phase 3a). The top three are sync_grid's rotating ids (codegen's convention);
each ring takes READY/FREE x slots x 2 parities = 4 x slots ids, READY first. The pass refuses
by name, saying by how much, when the pool runs out; ids are never shared between users.
"""

from __future__ import annotations

import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform, post_order_visit
from tvm.tirx.transform import prim_func_pass

from .lower_pipelined import NOTES_ATTR, RING_BEGIN, _with_notes

SYNC_BARRIERS = 3


def _launch_tiles(func: tirx.PrimFunc) -> int:
    tiles = [1]

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent" and node.node.thread_tag == "blockIdx.x":
            tiles[0] = int(node.value.value)

    post_order_visit(func.body, visit)
    return tiles[0]


def AssignIntelNpuBarriers():
    @prim_func_pass(opt_level=0, name="tl.AssignIntelNpuBarriers")
    def _pass(func: tirx.PrimFunc, mod, ctx):
        op = tvm.ir.Op.get(RING_BEGIN)
        rings = []

        def find(stmt):
            if isinstance(stmt, tirx.Bind) and isinstance(stmt.value, tirx.Call) and stmt.value.op.same_as(op):
                rings.append(stmt)

        post_order_visit(func.body, find)
        if not rings:
            return func
        target = func.attrs["target"]
        first = int(target.attrs["barrier_pool_first"])
        per_tile = int(target.attrs["barriers_per_tile"])
        tiles = _launch_tiles(func)
        pool_end = per_tile * tiles
        usable_end = pool_end - SYNC_BARRIERS
        need = sum(4 * int(r.value.args[2].value) for r in rings)
        if first + need > usable_end:
            raise ValueError(
                f"intel_npu: {len(rings)} ring(s) need {need} hardware barrier ids but the job's pool "
                f"[{first}, {pool_end}) minus sync_grid's {SYNC_BARRIERS} leaves {max(0, usable_end - first)}: "
                f"short by {first + need - usable_end}; launch more tiles or stream fewer operands at once")
        assigned = {}
        nxt = first
        for r in rings:
            slots = int(r.value.args[2].value)
            assigned[id(r)] = (nxt, nxt + 2 * slots)
            nxt += 4 * slots
        notes = []

        def rewrite(stmt):
            if isinstance(stmt, tirx.Bind) and isinstance(stmt.value, tirx.Call) and stmt.value.op.same_as(op):
                for r in rings:
                    if r.same_as(stmt):
                        ready, free = assigned[id(r)]
                        slots = int(stmt.value.args[2].value)
                        args = list(stmt.value.args)
                        args[4] = tirx.IntImm("int32", ready)
                        args[5] = tirx.IntImm("int32", free)
                        notes.append(f"{stmt.var.name}: READY {ready}..{ready + 2 * slots - 1}, FREE {free}..{free + 2 * slots - 1}")
                        return tirx.Bind(stmt.var, tirx.Call(stmt.value.dtype, op, args))
            return None

        body = ir_transform(func.body, None, rewrite, ["tirx.Bind"])
        func = func.with_body(body)
        return _with_notes(func, f"hardware barriers (pool [{first}, {pool_end}), sync_grid {usable_end}..{pool_end - 1}): "
                                 + "; ".join(notes))

    return _pass
