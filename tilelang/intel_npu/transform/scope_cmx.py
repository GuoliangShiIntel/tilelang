"""ScopeIntelNpuCmx: put every CMX allocation in the smallest C++ block that covers its uses, so blocks one after
another share CMX through the runtime's scoped bump allocator (codegen brackets each allocating block with
`npu::Ctx::CmxScope`; CheckIntelNpuBudgets charges the peak nesting, not the sum).

Runs after LowerOpaqueBlock, on flat `AllocBuffer` statements. Two moves:

1. **Phases.** Within any block body (the kernel's root sequence, a loop body, a scope), the statements between
   consecutive RENDEZVOUS points are phases: a grid barrier (`T.sync_grid()`) or a TILE sync
   (`T.sync_threads()`). An allocation declared in that block and used by exactly one phase moves into it, and
   the phase becomes a `lexical_alloc_scope` block. PlanAndUpdateBufferAllocationLocation (run just before)
   already sank allocations into loop bodies; this is the same idea for the straight-line phases of a megakernel,
   which have no block of their own in TIR.
   The rendezvous is what makes the sharing sound, and it is the condition npu_dsl's `plan_cmx` checks by name
   (`_separated_by_rendezvous`): the tile's SHAVEs are not in lockstep, so one may reuse bytes the other is still
   reading unless a barrier separates the two live ranges BOTH ways round -- between them, and across the
   enclosing loop's back edge. Phases split this way satisfy both, since each ends at a sync.
2. **Uniformity.** An allocation under a condition that depends on the SHAVE index is hoisted to the enclosing
   uniform block: every SHAVE of a tile must run the same allocation sequence, or shared buffers get different
   addresses on the two SHAVEs (the ring slots, the x row). Tile-dependent conditions are fine: a tile's SHAVEs
   agree, and the per-tile offsets stay equal because every tile runs the same program.
3. **Loops.** An allocation whose extents do not depend on a loop variable leaves that loop's body and bubbles
   out through every enclosing loop. Inside one loop it buys nothing -- the buffer is live in every iteration, so
   the peak over that loop is the same either way -- while the runtime re-runs mark / alloc / restore once per
   iteration for an address that never changes. That cost correctness, not just time: the GDN prefill scan had
   one 512 B scratch buffer sunk into a per-row loop (1152 allocations per worker) and was intermittently wrong
   on 2+ tiles, 4/12 trials clean at 3 x 2, 12/12 once it stayed put (plan question 10). Two SIBLING loops with
   different buffers therefore land side by side in the parent block rather than sharing bytes -- unless a
   rendezvous separates them, in which case rule 1 puts each in its own phase scope. That is what the MoE
   expert's gate/up and down weight tiles rely on.
"""

from __future__ import annotations

import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit
from tvm.tirx.transform import prim_func_pass

LEXICAL_SCOPE = "lexical_alloc_scope"
NOTES_ATTR = "intel_npu.notes"


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


def _storage_sync_scope(stmt):
    """The scope string of a `tvm_storage_sync` statement, else None. The op is registered under both the `tir.`
    and the `tirx.` prefix depending on how it was built (`T.sync_threads` emits the latter)."""
    if not (isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call)):
        return None
    call = stmt.value
    if getattr(call.op, "name", "").split(".")[-1] != "tvm_storage_sync" or not call.args:
        return None
    return call.args[0].value if isinstance(call.args[0], tirx.StringImm) else None


def _is_sync_tile(stmt) -> bool:
    return _storage_sync_scope(stmt) in ("shared", "shared.dyn")


def _is_phase_split(stmt) -> bool:
    """A rendezvous every SHAVE of the tile passes, so the phases either side may share CMX bytes: the grid
    barrier, and the TILE sync -- which is what separates an expert's gate/up weights from its down weights."""
    return _is_sync_grid(stmt) or _is_sync_tile(stmt)


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


def _is_ddr_bind(s) -> bool:
    return (isinstance(s, tirx.Bind) and isinstance(s.value, tirx.Call)
            and getattr(s.value.op, "name", "") == "tl.intel_npu_ddr_ptr")


def _is_ring_bind(s) -> bool:
    """A `ring = ctx.ring_begin(...)` Bind: codegen (`VisitStmt_(const BindNode*)`) prints it as a named
    `npu::Ctx::Ring` C++ variable at the Bind's own position, so wherever this statement ends up IS where the
    variable's C++ scope starts. The ring's use loop and `ring_end` are almost always in LATER phases, so this
    must never be nested inside a `lexical_alloc_scope` that one of those later phases falls outside of -- the
    bug a ring in the SAME phase as a scoped allocation (its own backing `k_cmx`/`v_cmx`, say) used to trigger:
    the allocation's phase-only scope is correct for the buffer, but it is not a scope the Bind may share."""
    return (isinstance(s, tirx.Bind) and isinstance(s.value, tirx.Call)
            and getattr(s.value.op, "name", "") == "tl.intel_npu_ring_begin")


def _is_view_def(s, items) -> bool:
    """A `T.ddr_view` definition: the Bind of the pointer, or a DeclBuffer over a pointer bound that way in `items`.
    (A ring's `ring_begin` bind is NOT one of these: unlike a DDR pointer, it reads a CMX buffer argument, so it
    cannot simply move to the block's head ahead of that buffer's own (possibly phase-scoped) allocation -- see
    `_is_ring_bind` and its use in `_phase_scope`.)"""
    if _is_ddr_bind(s):
        return True
    if isinstance(s, tirx.DeclBuffer):
        return any(_is_ddr_bind(b) and b.var.same_as(s.buffer.data) for b in items)
    return False


def _has_ring(stmt) -> bool:
    """Whether the statement contains a ring call: LowerIntelNpuPipelined's slots, fills and DPU handshakes."""
    hit = [False]

    def visit(node):
        if isinstance(node, tirx.Call) and getattr(node.op, "name", "").startswith("tl.intel_npu_ring"):
            hit[0] = True

    post_order_visit(stmt, visit)
    return hit[0]


def _alloc_keys(stmt) -> set:
    """Identity keys of the data vars the statement's AllocBuffers define (anywhere inside)."""
    keys = set()

    def visit(node):
        if isinstance(node, tirx.AllocBuffer):
            keys.add(hash(node.buffer.data))

    post_order_visit(stmt, visit)
    return keys


def _is_decl(s, alloc_keys=None) -> bool:
    """An allocation or a DeclBuffer alias of one (LowerOpaqueBlock emits both; they move together). A DeclBuffer
    over a pointer defined otherwise (a `T.ddr_view` over a `Bind`) is not movable: it must follow its binding."""
    if isinstance(s, tirx.AllocBuffer):
        return True
    if isinstance(s, tirx.DeclBuffer):
        return alloc_keys is None or hash(s.buffer.data) in alloc_keys
    return False


def _strip_allocs(stmt):
    """(stmt without its AllocBuffer / DeclBuffer statements (None if nothing is left), those in program order)."""
    allocs = []
    keys = _alloc_keys(stmt)

    def rec(s):
        if _is_decl(s, keys):
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
        items = self._sink(_items(stmt))
        flat = []
        for s in items:
            flat.extend(self.flatten(s))
        items = flat
        # Grid barriers first; a block with none falls back to its TILE syncs. The order matters and is not a
        # preference: an allocation moves only when ONE phase uses it, so splitting finer can scope FEWER of them
        # and push the rest back to the block head, where they are summed. Measured on the n=1024 prefill model:
        # splitting on both at once took the peak from 1634944 B to 2016640 B.
        out = self._phase_scope(items, _is_sync_grid)
        if out is None:
            # Keep a ring wholly inside its sync-delimited phase; its slots and use loop are addressed by parity.
            # A ring in one phase can still release unrelated allocations before the next tile rendezvous.
            has_ring = any(_has_ring(s) for s in items)
            out = self._phase_scope(items, _is_sync_tile, need_two_owners=not has_ring)
        return out if out is not None else (_seq(items) if items else stmt)

    def _phase_scope(self, items: list, is_split, need_two_owners: bool = False):
        """Scope allocations over non-crossing intervals of rendezvous-separated phases.
        None when `is_split` finds no boundary, so the caller can try a coarser rendezvous."""
        phases, cur = [], []
        for s in items:
            if is_split(s):
                phases.append(cur)
                phases.append([s])
                cur = []
            else:
                cur.append(s)
        phases.append(cur)
        if len(phases) == 1:
            return None
        # DDR views (`T.ddr_view`: a Bind of a ddr_ptr call and its DeclBuffer) go to the block's head: a phase
        # becomes its own C++ block, and a view declared inside one would be out of scope for the next. Their
        # values read only kernel parameters and loop variables, all defined at the block's start.
        views = [s for s in items if _is_view_def(s, items)]
        if views:
            view_ids = {id(s) for s in views}
            items = [s for s in items if id(s) not in view_ids]
            phases = [[s for s in ph if id(s) not in view_ids] for ph in phases]
        keys = {hash(s.buffer.data) for s in items if isinstance(s, tirx.AllocBuffer)}
        allocs = [s for s in items if _is_decl(s, keys)]
        uses = []
        for ph in phases:
            u = set()
            for s in ph:
                if not _is_decl(s, keys):
                    u |= _buffers_used(s)
            uses.append(u)
        # A ring Bind (`ring = ctx.ring_begin(...)`) prints as a named C++ variable AT ITS OWN POSITION
        # (`VisitStmt_(const BindNode*)`), so wherever this statement lands IS where its C++ scope starts. Most
        # rings begin, run their use loop and `ring_end` all within ONE phase (no sync_tile interrupts them) --
        # that phase may be scoped exactly like any other, ring included, as test_ring_phase_releases_unrelated_
        # cmx_at_tile_rendezvous relies on. A ring whose later read (a fill, a submit, `ring_end`) falls in a
        # DIFFERENT phase from its `ring_begin` is the unsafe case (the flash-attention phase3 regression this
        # guards): every phase that ring is read from or defined in must then be withdrawn from `home` entirely,
        # for two reasons at once -- such a phase must not become a `lexical_alloc_scope` itself (the Ring
        # variable would go out of scope before a later read), and no OTHER buffer those phases also read may be
        # homed to some single later phase either (it would move after an earlier phase's read of it).
        binding_def_phase = {hash(s.var): i for i, ph in enumerate(phases) for s in ph
                     if isinstance(s, tirx.Bind)}
        unsafe_phases = set()
        for key, def_phase in binding_def_phase.items():
            spans = {def_phase} | {i for i, u in enumerate(uses) if key in u}
            if len(spans) > 1:
                unsafe_phases |= spans
        unsafe_keys = {key for i in unsafe_phases for key in uses[i]}
        home = {}
        intervals = set()
        for a in allocs:
            key = hash(a.buffer.data)
            if key in unsafe_keys:
                continue
            where = [i for i, u in enumerate(uses) if key in u and i not in unsafe_phases]
            if not where:
                continue
            first, last = where[0], where[-1]
            if any(first <= i <= last for i in unsafe_phases):
                continue
            if first != last and any(_has_ring(s) for ph in phases[first:last + 1] for s in ph):
                continue
            if any(first < start <= last < end or start < first <= end < last
                   for start, end in intervals):
                continue
            home[id(a)] = (first, last)
            intervals.add((first, last))
        owners = intervals | {(i, i) for i, ph in enumerate(phases)
                      if any(_alloc_keys(s) for s in ph if not _is_decl(s, keys))}
        if need_two_owners and not any(end < other_start
                   for start, end in owners
                   for other_start, other_end in owners):
            # nothing to overlap, so scoping would move allocations INTO a loop body for no CMX at all -- and that
            # is the state question 10 was about. Only pay it where two groups actually share the bytes.
            return None
        out = []
        stack = [out]
        for i, ph in enumerate(phases):
            for interval in sorted((span for span in intervals if span[0] == i), key=lambda span: -span[1]):
                moved = [a for a in allocs if home.get(id(a)) == interval]
                self.moved += sum(isinstance(a, tirx.AllocBuffer) for a in moved)
                stack.append(moved)
            body = [s for s in ph if not _is_decl(s, keys)]
            stack[-1].extend(body)
            for interval in sorted((span for span in intervals if span[1] == i), key=lambda span: -span[0]):
                scoped = stack.pop()
                stack[-1].append(tirx.AttrStmt(tirx.IntImm("int32", 0), LEXICAL_SCOPE,
                                             tirx.IntImm("int32", 1), _seq(scoped)))
        # allocations that stay in this block come first, where they were
        kept = [a for a in allocs if id(a) not in home]
        return _seq(views + kept + out)

    def _sink(self, items: list) -> list:
        """3. **Branches and loops.** An allocation of this block used by exactly one statement of it, that statement
        being a loop or a SHAVE-uniform `if`, moves into that loop body / the one branch that uses it (together with its
        DeclBuffer aliases), where the phase rule then places it. This is what puts a layer kind's buffers under its
        `if Kind[li] == 0` branch, so the two kinds of a model loop share CMX instead of adding up."""
        flat = []

        def expand(s):
            # a notes wrapper is a comment: keep it as an empty-bodied marker and let its body join this block
            if isinstance(s, tirx.AttrStmt) and s.attr_key == NOTES_ATTR:
                flat.append(tirx.AttrStmt(s.node, s.attr_key, s.value, _nop()))
                for x in _items(s.body):
                    expand(x)
            else:
                flat.extend(_items(s))

        for s in items:
            expand(s)
        items = flat
        keys = {hash(s.buffer.data) for s in items if isinstance(s, tirx.AllocBuffer)}
        decls = [s for s in items if _is_decl(s, keys)]
        if not decls:
            return items
        by_key = {}
        for d in decls:
            by_key.setdefault(hash(d.buffer.data), []).append(d)
        uses = [None if _is_decl(s, keys) else _buffers_used(s) for s in items]
        sunk = {}  # id(item) -> {"then": [...], "else": [...], "body": [...]}
        gone = set()
        for key, ds in by_key.items():
            where = [i for i, u in enumerate(uses) if u is not None and key in u]
            if len(where) != 1:
                continue
            tgt = items[where[0]]
            if isinstance(tgt, tirx.For):
                slot = "body"
            elif isinstance(tgt, tirx.IfThenElse) and not (self.shave is not None and _uses_var(tgt.condition, self.shave)):
                in_then = key in _buffers_used(tgt.then_case)
                in_else = tgt.else_case is not None and key in _buffers_used(tgt.else_case)
                if in_then == in_else:
                    continue
                slot = "then" if in_then else "else"
            else:
                continue
            sunk.setdefault(id(tgt), {}).setdefault(slot, []).extend(ds)
            gone.update(id(d) for d in ds)
        if not gone:
            return items
        out = []
        for s in items:
            if id(s) in gone:
                continue
            mv = sunk.get(id(s))
            if mv is None:
                out.append(s)
                continue
            n = sum(isinstance(d, tirx.AllocBuffer) for v in mv.values() for d in v)
            self.moved += n
            if isinstance(s, tirx.For):
                out.append(_for(s, _seq(mv["body"] + _items(s.body))))
            else:
                then_case = _seq(mv.get("then", []) + _items(s.then_case))
                else_case = s.else_case
                if "else" in mv:
                    else_case = _seq(mv["else"] + _items(s.else_case))
                out.append(tirx.IfThenElse(s.condition, then_case, else_case))
        return out

    def _hoist_from_loop(self, s, body) -> list:
        """Loop-invariant allocations at the top of `body` move in front of the loop, so the runtime allocates them
        once instead of once per iteration. Nesting these calls hoists through every enclosing loop in turn."""
        if _has_ring(body):
            # A ring's slots and its use loop are addressed by use parity; leave the loop alone until the
            # interaction with the hoist is understood (plan question 10).
            return [_for(s, body)]
        items = _items(body)
        keys = _alloc_keys(body)
        lead, rest = [], []
        for x in items:
            if _is_decl(x, keys) and not _uses_var(x, s.loop_var):
                lead.append(x)
            else:
                rest.append(x)
        if not lead or not rest:
            return [_for(s, body)]
        self.hoisted += sum(isinstance(a, tirx.AllocBuffer) for a in lead)
        return lead + [_for(s, _seq(rest))]

    def flatten(self, s) -> list:
        """One statement of a block, with nested blocks rewritten; a SHAVE-dependent `if` gives up its allocations."""
        if isinstance(s, tirx.SeqStmt):
            out = []
            for x in s.seq:
                out.extend(self.flatten(x))
            return out
        if isinstance(s, tirx.For):
            return self._hoist_from_loop(s, self.block(s.body))
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
