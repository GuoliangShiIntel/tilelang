"""ScopeIntelNpuCmx: put every CMX allocation in the smallest C++ block that covers its uses, so blocks one after
another share CMX through the runtime's scoped bump allocator (codegen brackets each allocating block with
`npu::Ctx::CmxScope`; CheckIntelNpuBudgets charges the peak nesting, not the sum).

Runs after LowerOpaqueBlock, on flat `AllocBuffer` statements. Two moves:

1. **Phases.** Within any block body (the kernel's root sequence, a loop body, a scope), the statements between
   consecutive grid barriers (`T.sync_grid()` / `tvm_storage_sync("global")`) are phases. An allocation declared in
   that block and used by exactly one phase moves into it, and the phase becomes a `lexical_alloc_scope` block.
   PlanAndUpdateBufferAllocationLocation (run just before) already sank allocations into loop bodies; this is the
   same idea for the straight-line phases of a megakernel, which have no block of their own in TIR.
2. **Uniformity.** An allocation under a condition that depends on the SHAVE index is hoisted to the enclosing
   uniform block: every SHAVE of a tile must run the same allocation sequence, or shared buffers get different
   addresses on the two SHAVEs (the ring slots, the x row). Tile-dependent conditions are fine: a tile's SHAVEs
   agree, and the per-tile offsets stay equal because every tile runs the same program.
"""

from __future__ import annotations

import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit
from tvm.tirx.transform import prim_func_pass

LEXICAL_SCOPE = "lexical_alloc_scope"


def _is_sync_grid(stmt) -> bool:
    if not (isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call)):
        return False
    call = stmt.value
    name = getattr(call.op, "name", "")
    if name == "tl.sync_grid":
        return True
    if name == "tir.tvm_storage_sync" and len(call.args) == 1 and isinstance(call.args[0], tirx.StringImm):
        return call.args[0].value == "global"
    return False


def _shave_var(func: tirx.PrimFunc):
    found = [None]

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent" and node.node.thread_tag == "threadIdx.x":
            found[0] = node.node.var

    post_order_visit(func.body, visit)
    return found[0]


def _uses_var(expr, var) -> bool:
    hit = [False]

    def visit(node):
        if isinstance(node, tirx.Var) and node.same_as(var):
            hit[0] = True

    post_order_visit(expr, visit)
    return hit[0]


def _buffers_used(stmt) -> set:
    """Identity keys (hash) of every buffer data var the statement reads, writes or takes the address of."""
    used = set()

    def visit(node):
        if isinstance(node, (tirx.BufferLoad, tirx.BufferStore)):
            used.add(hash(node.buffer.data))
        elif isinstance(node, tirx.Var):
            used.add(hash(node))

    post_order_visit(stmt, visit)
    return used


def _items(stmt) -> list:
    return list(stmt.seq) if isinstance(stmt, tirx.SeqStmt) else [stmt]


def _seq(items: list):
    return items[0] if len(items) == 1 else tirx.SeqStmt(items)


def _nop():
    return tirx.Evaluate(tirx.IntImm("int32", 0))


def _for(s, body):
    return tirx.For(s.loop_var, s.min, s.extent, s.kind, body, s.thread_binding, s.annotations, s.step)


def _is_decl(s) -> bool:
    """An allocation or a DeclBuffer alias of one (LowerOpaqueBlock emits both; they move together)."""
    return isinstance(s, (tirx.AllocBuffer, tirx.DeclBuffer))


def _strip_allocs(stmt):
    """(stmt without its AllocBuffer / DeclBuffer statements (None if nothing is left), those in program order)."""
    allocs = []

    def rec(s):
        if _is_decl(s):
            allocs.append(s)
            return None
        if isinstance(s, tirx.SeqStmt):
            kept = [r for r in (rec(x) for x in s.seq) if r is not None]
            return _seq(kept) if kept else None
        if isinstance(s, tirx.For):
            body = rec(s.body)
            return _for(s, body) if body is not None else None
        if isinstance(s, tirx.IfThenElse):
            then_case = rec(s.then_case)
            else_case = rec(s.else_case) if s.else_case is not None else None
            if then_case is None and else_case is None:
                return None
            return tirx.IfThenElse(s.condition, then_case if then_case is not None else _nop(), else_case)
        if isinstance(s, tirx.AttrStmt):
            body = rec(s.body)
            return tirx.AttrStmt(s.node, s.attr_key, s.value, body) if body is not None else None
        if isinstance(s, tirx.While):
            body = rec(s.body)
            return tirx.While(s.condition, body) if body is not None else None
        return s

    return rec(stmt), allocs


class _Scoper:
    def __init__(self, shave):
        self.shave = shave
        self.moved = 0
        self.hoisted = 0
        self.nonuniform = 0

    # ---- a block body ------------------------------------------------------------------------------------------
    def block(self, stmt):
        """Rewrite one C++ block's body: hoist allocations out of SHAVE-dependent branches, recurse into nested
        blocks, then sink this block's allocations into the phases that use them. Transparent AttrStmt wrappers
        (thread_extent, notes) around the body are peeled and put back."""
        wrappers = []
        while isinstance(stmt, tirx.AttrStmt) and stmt.attr_key != LEXICAL_SCOPE:
            wrappers.append(stmt)
            stmt = stmt.body
        out = self._block_items(stmt)
        for w in reversed(wrappers):
            out = tirx.AttrStmt(w.node, w.attr_key, w.value, out)
        return out

    def _block_items(self, stmt):
        items = []
        for s in _items(stmt):
            items.extend(self.flatten(s))
        # phases of this block, split by grid barriers
        phases, cur = [], []
        for s in items:
            if _is_sync_grid(s):
                phases.append(cur)
                phases.append([s])
                cur = []
            else:
                cur.append(s)
        phases.append(cur)
        if len(phases) == 1:
            return _seq(items) if items else stmt
        allocs = [s for s in items if _is_decl(s)]
        uses = []
        for ph in phases:
            u = set()
            for s in ph:
                if not _is_decl(s):
                    u |= _buffers_used(s)
            uses.append(u)
        home = {}
        for a in allocs:
            key = hash(a.buffer.data)
            where = [i for i, u in enumerate(uses) if key in u]
            if len(where) == 1:
                home[id(a)] = where[0]
        out = []
        for i, ph in enumerate(phases):
            moved = [a for a in allocs if home.get(id(a)) == i]
            body = [s for s in ph if not _is_decl(s)]
            if moved:
                self.moved += sum(isinstance(a, tirx.AllocBuffer) for a in moved)
                out.append(tirx.AttrStmt(tirx.IntImm("int32", 0), LEXICAL_SCOPE, tirx.IntImm("int32", 1), _seq(moved + body)))
            else:
                out.extend(body)
        # allocations that stay in this block come first, where they were
        kept = [a for a in allocs if id(a) not in home]
        return _seq(kept + out)

    def flatten(self, s) -> list:
        """One statement of a block, with nested blocks rewritten; a SHAVE-dependent `if` gives up its allocations."""
        if isinstance(s, tirx.SeqStmt):
            out = []
            for x in s.seq:
                out.extend(self.flatten(x))
            return out
        if isinstance(s, tirx.For):
            return [_for(s, self.block(s.body))]
        if isinstance(s, tirx.While):
            return [tirx.While(s.condition, self.block(s.body))]
        if isinstance(s, tirx.IfThenElse):
            if self.shave is not None and self.nonuniform == 0 and _uses_var(s.condition, self.shave):
                stripped, allocs = _strip_allocs(s)
                self.hoisted += sum(isinstance(a, tirx.AllocBuffer) for a in allocs)
                if stripped is None:
                    return list(allocs)
                # nothing allocates inside any more: nested SHAVE-dependent branches are plain blocks now
                self.nonuniform += 1
                rest = self.flatten(stripped)
                self.nonuniform -= 1
                return list(allocs) + rest
            then_case = self.block(s.then_case)
            else_case = self.block(s.else_case) if s.else_case is not None else None
            return [tirx.IfThenElse(s.condition, then_case, else_case)]
        if isinstance(s, tirx.AttrStmt):
            if s.attr_key == LEXICAL_SCOPE:
                return [tirx.AttrStmt(s.node, s.attr_key, s.value, self.block(s.body))]
            # thread_extent / notes: transparent, its body is part of this block
            return [tirx.AttrStmt(s.node, s.attr_key, s.value, _seq(self.flatten(s.body)))]
        return [s]


def ScopeIntelNpuCmx():
    def transform(func: tirx.PrimFunc, mod, ctx):
        if func.attrs is None or "target" not in func.attrs or func.attrs["target"].kind.name != "intel_npu":
            return func
        scoper = _Scoper(_shave_var(func))
        body = scoper.block(func.body)
        if scoper.moved == 0 and scoper.hoisted == 0:
            return func
        return func.with_body(body)

    return prim_func_pass(transform, opt_level=0, name="tl.ScopeIntelNpuCmx")
