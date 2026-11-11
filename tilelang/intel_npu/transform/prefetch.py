"""PlanIntelNpuPrefetch: software-pipelined loads, so a DMA runs while the DPU and the SHAVE work.

The per-task records showed the prefill phases to be strictly serial -- load, compute, store, load, compute --
with the `ring` columns empty and the DPU busy 3-10%% of a phase. A ring (`T.Pipelined`) does not fix that here:
it is a work DISTRIBUTION mechanism and deals its chunks round-robin over the tile's SHAVEs, which is wrong for a
loop every worker must walk in full.

So the overlap is derived instead, as npu_dsl's `plan_prefetch` derives it (npu_dsl/fused/passes/prefetch.py).
In a loop whose body STARTS with a DDR -> CMX load whose address depends on the loop variable, whose destination
nothing else in the loop writes and nothing outside the loop reads, that load runs ONE ITERATION AHEAD: the
destination gets a twin (its leading extent doubled, the same data var -- the ring's `_widen` trick) and the
iterations alternate between the halves, so iteration i computes out of one while i + 1 streams into the other.

The author writes nothing. A missing wait would cost time rather than correctness in any case: the runtime
registers the in-flight load on its port, so every synchronous DMA and every `qg_flush` joins it (npu_rt_device
`dma_load_async`).
"""

from __future__ import annotations

import os
import math

import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit
from tvm.tirx.transform import prim_func_pass

DMA_COPY = "tl.intel_npu_dma_copy"
DMA_WAIT = "tl.intel_npu_dma_wait"
DEBUG = os.environ.get("TL_NPU_PREFETCH_DEBUG")


def _int(expr):
    """`expr` as a Python int when it is a constant, else None. Comparing a PrimExpr with `!=` builds a node
    rather than a bool, so every such test has to go through here."""
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    return None


def _is_compact(buf):
    """Whether the buffer has row-major compact strides, or leaves them implicit."""
    if buf.strides is None:
        return True
    stride = 1
    for shape, actual in zip(reversed(buf.shape), reversed(buf.strides)):
        if _int(actual) != stride:
            return False
        dim = _int(shape)
        if dim is None:
            return False
        stride *= dim
    return True


def _call(stmt):
    """The Call a statement evaluates, else None."""
    if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
        return stmt.value
    return None


def _is_load(stmt):
    """A DDR -> CMX `intel_npu_dma_copy` (to_cmx = 1) that is not already asynchronous."""
    call = _call(stmt)
    if call is None or getattr(call.op, "name", "") != DMA_COPY:
        return None
    to_cmx = call.args[6]
    if not isinstance(to_cmx, tirx.IntImm) or to_cmx.value != 1:
        return None
    if len(call.args) > 7:
        return None
    return call


def _dst_buffer(call):
    """The CMX buffer a dma_copy writes: its first argument is `address_of(BufferLoad(buf, ...))`."""
    arg = call.args[0]
    if isinstance(arg, tirx.Call) and arg.args and isinstance(arg.args[0], tirx.BufferLoad):
        return arg.args[0].buffer
    return None


def _items(stmt) -> list:
    return list(stmt.seq) if isinstance(stmt, tirx.SeqStmt) else [stmt]


def _seq(items: list):
    return items[0] if len(items) == 1 else tirx.SeqStmt(items)


def _uses_var(node, var) -> bool:
    hit = [False]

    def visit(n):
        if isinstance(n, tirx.Var) and n.same_as(var):
            hit[0] = True

    post_order_visit(node, visit)
    return hit[0]


def _refs(node, data_var):
    """How many times `node` reads, writes or takes the address of the buffer behind `data_var`."""
    n = [0]

    def visit(x):
        if isinstance(x, (tirx.BufferLoad, tirx.BufferStore)) and x.buffer.data.same_as(data_var):
            n[0] += 1

    post_order_visit(node, visit)
    return n[0]


def _reads(node, data_var):
    n = [0]

    def visit(x):
        if isinstance(x, tirx.BufferLoad) and x.buffer.data.same_as(data_var):
            n[0] += 1

    post_order_visit(node, visit)
    return n[0]


def _writes(node, data_var) -> bool:
    hit = [False]

    def visit(x):
        if isinstance(x, tirx.BufferStore) and x.buffer.data.same_as(data_var):
            hit[0] = True

    post_order_visit(node, visit)
    return hit[0]


def _full_buffer_load(call, buf) -> bool:
    """The DMA overwrites every byte of a compact target starting at its origin; only then can an earlier
    scope-pin store to slot zero be discarded safely when the loop's accesses are rebased to alternating slots."""
    addr = call.args[0]
    if not (isinstance(addr, tirx.Call) and addr.args and isinstance(addr.args[0], tirx.BufferLoad)):
        return False
    load = addr.args[0]
    if not all(_int(i) == 0 for i in load.indices) or not _is_compact(buf):
        return False
    width, rows, dst_stride = (_int(call.args[i]) for i in (2, 3, 4))
    dims = [_int(d) for d in buf.shape]
    if width is None or rows is None or dst_stride is None or any(d is None for d in dims):
        return False
    if dst_stride != width or width * rows != math.prod(dims) * (buf.dtype.bits // 8):
        return False
    return True


def _dpu_reads_a(stmts, data_var) -> bool:
    """The loaded tile is the activation (A) operand of a DPU matmul, not just any buffer somewhere in the body."""
    found = [False]

    def visit(node):
        if not isinstance(node, tirx.Call):
            return
        name = getattr(node.op, "name", "")
        if name not in ("tl.intel_npu_dpu_matmul", "tl.intel_npu_dpu_matmul_i4g", "tl.intel_npu_dpu_matmul_i8g"):
            return
        if len(node.args) > 1 and _reads(node.args[1], data_var):
            found[0] = True

    post_order_visit(_seq(list(stmts)), visit)
    return found[0]


class _Rebase:
    """Every access to `data_var` moved into slot `parity` of the widened buffer: its leading index gains
    `parity * extent`, and the Buffer object becomes the widened one so the flattened address is right."""

    def __init__(self, data_var, wide, extent, parity):
        self.data_var, self.wide, self.extent, self.parity = data_var, wide, extent, parity

    def _idx(self, indices):
        return [indices[0] + self.parity * self.extent] + list(indices[1:])

    def __call__(self, node):
        rebase = self

        def post(n):
            if isinstance(n, tirx.BufferLoad) and n.buffer.data.same_as(rebase.data_var):
                return tirx.BufferLoad(rebase.wide, rebase._idx(n.indices))
            if isinstance(n, tirx.BufferStore) and n.buffer.data.same_as(rebase.data_var):
                return tirx.BufferStore(rebase.wide, n.value, rebase._idx(n.indices))
            return None

        return tirx.stmt_functor.ir_transform(node, None, post, ["tirx.BufferLoad", "tirx.BufferStore"])


def _async(call):
    """`call` with the asynchronous flag set."""
    return tirx.Evaluate(tirx.Call(call.dtype, call.op, list(call.args) + [tirx.IntImm("int32", 1)]))


def _wait():
    return tirx.Evaluate(tirx.call_intrin("int32", tirx.op.Op.get(DMA_WAIT)))


def _subst(node, var, value):
    return tirx.stmt_functor.substitute(node, {var: value})


class _Planner:
    def __init__(self, func):
        self.func = func
        self.done = 0
        self.widened = {}

    def body(self, stmt):
        """Rewrite a statement tree, prefetching the loops inside it."""
        if isinstance(stmt, tirx.SeqStmt):
            return _seq([self.body(s) for s in stmt.seq])
        if isinstance(stmt, tirx.For):
            inner = self.body(stmt.body)
            loop = tirx.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind, inner, stmt.thread_binding,
                            stmt.annotations, stmt.step)
            return self.prefetch(loop)
        if isinstance(stmt, tirx.IfThenElse):
            return tirx.IfThenElse(stmt.condition, self.body(stmt.then_case),
                                   self.body(stmt.else_case) if stmt.else_case is not None else None)
        if isinstance(stmt, tirx.AttrStmt):
            return tirx.AttrStmt(stmt.node, stmt.attr_key, stmt.value, self.body(stmt.body))
        if isinstance(stmt, tirx.While):
            return tirx.While(stmt.condition, self.body(stmt.body))
        return stmt

    def widen_allocations(self, stmt):
        """Make the allocation match every widened view; changing BufferLoad/Store shapes alone is not storage."""
        widened = self.widened

        def post(node):
            if isinstance(node, tirx.AllocBuffer):
                buf = widened.get(hash(node.buffer.data))
                if buf is not None:
                    return tirx.AllocBuffer(buf)
            return None

        return tirx.stmt_functor.ir_transform(stmt, None, post, ["tirx.AllocBuffer"])

    def prefetch(self, loop):
        """The loop with its leading load run an iteration ahead, or unchanged when any condition fails."""
        def no(why):
            if DEBUG:
                print(f"[prefetch] loop {loop.loop_var.name} extent={loop.extent}: {why}")
            return loop

        if self.done >= 1:
            return no("initial rollout selects one candidate per PrimFunc; CMX-aware selection follows after this proof")
        if loop.kind != tirx.ForKind.SERIAL or loop.thread_binding is not None:
            return no("non-serial or thread-bound")
        if loop.step is not None and _int(loop.step) != 1:
            return no(f"step {loop.step}")
        extent = loop.extent
        trip_count = _int(extent)
        if trip_count is None or trip_count < 2:
            return no("trip count is dynamic or less than two")
        items = _items(loop.body)
        if not items:
            return no("empty body")
        call = _is_load(items[0])
        if call is None:
            return no(f"body does not start with a sync DDR->CMX load (starts with {type(items[0]).__name__})")
        dst = _dst_buffer(call)
        if dst is None:
            return no("destination is not address_of(BufferLoad)")
        if dst.scope() not in ("local",):
            # shared would be filled by every SHAVE of the tile: that is the ring's job, not this one
            return no(f"destination {dst.name} is scope {dst.scope()}")
        var = loop.loop_var
        if not _uses_var(call.args[1], var):
            return no("the source address does not move with the loop")
        rest = items[1:]
        if not rest:
            return no("nothing to overlap with")
        data = dst.data
        key = hash(data)
        if key in self.widened:
            return no(f"{dst.name} is already widened by another loop")
        if not _is_compact(dst):
            return no(f"{dst.name} has non-compact strides")
        if any(_writes(s, data) for s in rest):
            return no(f"the body rewrites {dst.name}")
        if not _dpu_reads_a(rest, data):
            return no(f"{dst.name} is not consumed as a DPU A operand")
        if _reads(self.func.body, data) != _reads(loop, data):
            return no(f"{dst.name} is read outside the loop")
        if _refs(self.func.body, data) != _refs(loop, data) and not _full_buffer_load(call, dst):
            return no(f"{dst.name} has an outside write and the loop load does not overwrite the whole buffer")
        extent0 = dst.shape[0]
        wide = tirx.decl_buffer([extent0 * 2] + list(dst.shape[1:]), dst.dtype, dst.name,
                                data=data, scope=dst.scope())
        self.widened[key] = wide
        parity = (var - loop.min) % 2
        # iteration i computes out of slot `parity` while i + 1 streams into the other half
        ahead = _Rebase(data, wide, extent0, 1 - parity)
        here = _Rebase(data, wide, extent0, parity)
        next_call = _subst(_async(call), var, var + 1)
        guarded = tirx.IfThenElse(var + 1 < loop.min + extent, ahead(next_call), None)
        new_body = _seq([_wait(), guarded] + [here(s) for s in rest])
        first = _Rebase(data, wide, extent0, 0)(_subst(_async(call), var, loop.min))
        if DEBUG:
            print(f"[prefetch] selected loop {var.name}, destination {dst.name}, bytes={math.prod(_int(d) for d in dst.shape) * (dst.dtype.bits // 8)}")
        self.done += 1
        return _seq([first, tirx.For(var, loop.min, extent, loop.kind, new_body, loop.thread_binding,
                                     loop.annotations, loop.step)])


class _DpuPlanner(_Planner):
    def __init__(self, func, early_packed=()):
        super().__init__(func)
        self.early_packed = set(early_packed)
        self.packed_candidates = []

    def prefetch(self, loop):
        if loop.kind != tirx.ForKind.SERIAL or loop.thread_binding is not None:
            return loop
        if _int(loop.extent) is None or _int(loop.extent) < 2 or (loop.step is not None and _int(loop.step) != 1):
            return loop
        declarations, items = [], []

        def linearize(stmt):
            if isinstance(stmt, tirx.SeqStmt):
                for child in stmt.seq:
                    linearize(child)
            elif isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "lexical_alloc_scope":
                linearize(stmt.body)
            elif isinstance(stmt, (tirx.AllocBuffer, tirx.DeclBuffer)):
                declarations.append(stmt)
            else:
                items.append(stmt)

        linearize(loop.body)
        if any(any(_int(dim) is None for dim in stmt.buffer.shape) for stmt in declarations):
            return loop
        groups = []
        index = 0

        def marker(stmt, name):
            call = _call(stmt)
            return (call is not None and getattr(call.op, "name", "") == "tirx.call_extern"
                    and call.args and isinstance(call.args[0], tirx.StringImm) and call.args[0].value == name)

        def tile_sync(stmt):
            call = _call(stmt)
            return (call is not None and getattr(call.op, "name", "").split(".")[-1] == "tvm_storage_sync"
                and len(call.args) == 1 and isinstance(call.args[0], tirx.StringImm)
                and call.args[0].value == "shared")

        def dequant_call(group):
            call = _call(group[0][group[1]])
            if call is None:
                return None, None
            name = getattr(call.op, "name", "")
            if name == "tl.intel_npu_dequant_i4_gm":
                return call, list(call.args)
            if (name == "tirx.call_extern" and call.args and isinstance(call.args[0], tirx.StringImm)
                    and call.args[0].value == "tl_npu_dequant_i4_gm"):
                return call, list(call.args[1:])
            return None, None

        while index < len(items):
            if marker(items[index], "ctx.task_begin"):
                if index + 2 >= len(items) or not marker(items[index + 2], "ctx.task_end"):
                    return loop
                groups.append((items[index:index + 3], 1))
                index += 3
            else:
                groups.append(([items[index]], 0))
                index += 1
        dequant_groups = [(i, g, dequant_call(g)) for i, g in enumerate(groups)
                  if dequant_call(g)[0] is not None]
        gemm_groups = [(i, g) for i, g in enumerate(groups)
                       if _call(g[0][g[1]]) is not None
                       and getattr(_call(g[0][g[1]]).op, "name", "") == "tl.intel_npu_dpu_matmul"]
        if len(dequant_groups) != 1 or len(gemm_groups) != 1:
            return loop
        dequant_index, dequant_group, (dequant, dequant_args) = dequant_groups[0]
        gemm_index, gemm_group = gemm_groups[0]
        cooperative = (gemm_index == dequant_index + 2
                   and tile_sync(groups[dequant_index + 1][0][groups[dequant_index + 1][1]]))
        if (gemm_index != dequant_index + 1 and not cooperative) or gemm_index + 1 == len(groups):
            return loop
        gemm = _call(gemm_group[0][gemm_group[1]])
        if gemm is None or getattr(gemm.op, "name", "") != "tl.intel_npu_dpu_matmul" or len(gemm.args) != 8:
            return loop
        preamble = [stmt for group, _ in groups[:dequant_index] for stmt in group]
        if any(not (isinstance(address, tirx.Call) and address.args
                    and isinstance(address.args[0], tirx.BufferLoad)) for address in dequant_args[:3]):
            return loop
        packed_buffers = [dequant_args[1].args[0].buffer, dequant_args[2].args[0].buffer]
        for stmt in preamble:
            fill = _is_load(stmt)
            if fill is None:
                return loop
            loaded = _dst_buffer(fill)
            if (loaded is None or loaded.scope() != "local" or not _is_compact(loaded)
                    or not any(loaded.data.same_as(buf.data) for buf in packed_buffers)
                    or not _uses_var(fill.args[1], loop.loop_var)
                    or _refs(self.func.body, loaded.data) != _refs(loop.body, loaded.data)):
                return loop
        tail = _seq([stmt for group, _ in groups[gemm_index + 1:] for stmt in group])
        weights = dequant_args[0].args[0].buffer
        output = _dst_buffer(gemm)
        if weights is None or output is None or not _reads(gemm.args[2], weights.data):
            return loop
        if cooperative:
            if weights.scope() != "shared" or not tile_sync(groups[-1][0][groups[-1][1]]):
                return loop
            threads = []

            def find_threads(node):
                if (isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent"
                        and node.node.thread_tag == "threadIdx.x"):
                    threads.append((node.node.var, _int(node.value)))

            post_order_visit(self.func.body, find_threads)
            if len(threads) != 1 or threads[0][1] is None:
                return loop
            rows, width = (_int(value) for value in dequant_args[3:5])
            if rows is None or width is None or rows * threads[0][1] != _int(weights.shape[0]):
                return loop
            indices = dequant_args[0].args[0].indices
            analyzer = tvm.arith.Analyzer()
            if (len(indices) != 2 or not analyzer.can_prove_equal(indices[0], threads[0][0] * rows)
                    or _int(indices[1]) != 0 or width != _int(weights.shape[1])):
                return loop
        if weights.data.same_as(output.data) or _reads(gemm.args[1], weights.data):
            return loop
        for buffer in (weights, output):
            allowed_scope = "shared" if cooperative and buffer.data.same_as(weights.data) else "local"
            if buffer.scope() != allowed_scope or not _is_compact(buffer) or len(buffer.shape) != 2:
                return loop
            if any(_int(dim) is None for dim in buffer.shape):
                return loop
            if _refs(self.func.body, buffer.data) != _refs(loop.body, buffer.data):
                return loop
        if not cooperative and [_int(arg) for arg in dequant_args[3:5]] != [_int(dim) for dim in weights.shape]:
            return loop
        if _int(gemm.args[4]) != _int(output.shape[1]):
            return loop
        row_bound = tvm.arith.Analyzer().const_int_bound(gemm.args[3])
        if (row_bound.min_value < 0 or row_bound.max_value > _int(output.shape[0])
                or _uses_var(gemm.args[3], loop.loop_var)):
            return loop
        for address in (dequant_args[0], gemm.args[0]):
            if cooperative and address.same_as(dequant_args[0]):
                continue
            load = address.args[0]
            if not all(_int(value) == 0 for value in load.indices):
                return loop
        if not preamble and not _uses_var(dequant_args[1], loop.loop_var) and not _uses_var(dequant_args[2], loop.loop_var):
            return loop
        if not _reads(tail, output.data) or _refs(tail, weights.data):
            return loop
        protected = []
        for address in list(dequant_args[1:3]) + [gemm.args[1]]:
            if not (isinstance(address, tirx.Call) and isinstance(address.args[0], tirx.BufferLoad)):
                return loop
            def protect(node):
                if isinstance(node, tirx.BufferLoad):
                    protected.append(node.buffer.data)
            post_order_visit(address, protect)
        def protect_weight_indices(node):
            if isinstance(node, tirx.BufferLoad) and not node.buffer.data.same_as(weights.data):
                protected.append(node.buffer.data)
        post_order_visit(gemm.args[2], protect_weight_indices)
        post_order_visit(gemm.args[3], protect_weight_indices)
        if any(data.same_as(weights.data) or data.same_as(output.data) for data in protected):
            return loop
        safe = [True]

        def inspect(node):
            if isinstance(node, tirx.BufferStore):
                if any(node.buffer.data.same_as(data) for data in protected + [output.data, weights.data]):
                    safe[0] = False
            elif isinstance(node, tirx.Call):
                name = getattr(node.op, "name", "")
                if cooperative and tile_sync(tirx.Evaluate(node)):
                    return
                if name == "tirx.call_extern":
                    if not (isinstance(node.args[0], tirx.StringImm)
                            and node.args[0].value in ("ctx.task_begin", "ctx.task_end")):
                        safe[0] = False
                elif name == DMA_COPY:
                    destination = _dst_buffer(node)
                    if (_int(node.args[6]) != 0 or destination is None
                            or any(destination.data.same_as(data) for data in protected + [output.data, weights.data])):
                        safe[0] = False
                elif name != "tirx.address_of":
                    if (not isinstance(node.op, tvm.ir.Op)
                            or _int(node.op.get_attr("TCallEffectKind")) != tirx.CallEffectKind.Pure.value):
                        safe[0] = False

        post_order_visit(tail, inspect)
        if not safe[0]:
            return loop
        pipeline_sync = tirx.Evaluate(tirx.call_intrin(
            "handle", tirx.op.Op.get("tl.intel_npu_dpu_async_sync"))) if cooperative else None
        if cooperative:
            tail = _seq([stmt for group, _ in groups[gemm_index + 1:-1] for stmt in group] + [pipeline_sync])
        packed_prefetch = (cooperative and bool(tvm.transform.PassContext.current().config.get(
            "tl.intel_npu.prefetch", True)) and bool(preamble)
            and _dst_buffer(_is_load(preamble[0])).data.same_as(packed_buffers[0].data)
            and _full_buffer_load(_is_load(preamble[0]), packed_buffers[0]))
        early_packed = packed_prefetch and hash(loop.loop_var) in self.early_packed
        if packed_prefetch:
            self.packed_candidates.append(hash(loop.loop_var))
        if early_packed:
            packed = packed_buffers[0]
            wide_packed = tirx.decl_buffer([packed.shape[0] * 2] + list(packed.shape[1:]), packed.dtype,
                                           packed.name, data=packed.data, scope=packed.scope())
            self.widened[hash(packed.data)] = wide_packed
        wide_weights = tirx.decl_buffer([weights.shape[0] * 2, weights.shape[1]], weights.dtype,
                                        weights.name, data=weights.data, scope=weights.scope())
        wide_output = tirx.decl_buffer([output.shape[0] * 2, output.shape[1]], output.dtype,
                                       output.name, data=output.data, scope=output.scope())
        self.widened[hash(weights.data)] = wide_weights
        self.widened[hash(output.data)] = wide_output

        def rebase(stmt, value):
            parity = (value - loop.min) % 2
            stmt = _subst(stmt, loop.loop_var, value)
            stmt = _Rebase(weights.data, wide_weights, weights.shape[0], parity)(stmt)
            if early_packed:
                stmt = _Rebase(packed.data, wide_packed, packed.shape[0], parity)(stmt)
            return _Rebase(output.data, wide_output, output.shape[0], parity)(stmt)

        def prepare(value):
            rendezvous = [pipeline_sync] if cooperative else []
            if packed_prefetch:
                ahead = tirx.IfThenElse(value + 1 < loop.min + loop.extent,
                                        rebase(_async(_is_load(preamble[0])), value + 1), None)
                work = rebase(_seq(dequant_group[0] + rendezvous), value)
                return _seq([_wait()] + [rebase(stmt, value) for stmt in preamble[1:]]
                            + ([ahead, work] if early_packed else [work, ahead]))
            return rebase(_seq(preamble + dequant_group[0] + rendezvous), value)

        def submit(value):
            group = list(gemm_group[0])
            group[gemm_group[1]] = tirx.Evaluate(tirx.Call(gemm.dtype, gemm.op,
                                                          list(gemm.args) + [tirx.IntImm("int32", 1)]))
            return rebase(_seq(group), value)

        following = loop.loop_var + 1
        valid = following < loop.min + loop.extent
        wait = tirx.Evaluate(tirx.call_intrin("handle", tirx.op.Op.get("tl.intel_npu_dpu_wait")))
        body = _seq([tirx.IfThenElse(valid, prepare(following), None), wait,
                     tirx.IfThenElse(valid, submit(following), None), rebase(tail, loop.loop_var)])
        initial = [rebase(_async(_is_load(preamble[0])), loop.min)] if packed_prefetch else []
        scheduled = _seq(declarations + initial + [prepare(loop.min), submit(loop.min),
                          tirx.For(loop.loop_var, loop.min, loop.extent, loop.kind, body, loop.thread_binding,
                                    loop.annotations, loop.step)])
        if cooperative:
            scheduled = _seq([scheduled, groups[-1][0][groups[-1][1]]])
            return tirx.AttrStmt(tirx.IntImm("int32", 0), "lexical_alloc_scope", tirx.IntImm("int32", 1), scheduled)
        return scheduled


def PlanIntelNpuDmaBatch():
    def transform(func: tirx.PrimFunc, mod, ctx):
        if not bool(ctx.config.get("tl.intel_npu.batch_dma", False)):
            return func
        live_rings = [0]

        def pre(node):
            call = node.value if isinstance(node, tirx.Bind) and isinstance(node.value, tirx.Call) else _call(node)
            if call is not None:
                name = getattr(call.op, "name", "")
                if name == "tl.intel_npu_ring_begin":
                    live_rings[0] += 1
                elif name == "tl.intel_npu_ring_end":
                    live_rings[0] = max(0, live_rings[0] - 1)
            if not isinstance(node, tirx.For) or live_rings[0] or node.kind != tirx.ForKind.SERIAL:
                return None
            if node.thread_binding is not None or (node.step is not None and _int(node.step) != 1):
                return None
            body, conditions, bindings = node.body, [], {}
            while True:
                if isinstance(body, tirx.SeqStmt) and len(body.seq) == 1:
                    body = body.seq[0]
                elif (isinstance(body, tirx.SeqStmt) and len(body.seq) > 1
                      and all(isinstance(stmt, tirx.Bind) for stmt in body.seq[:-1])):
                    for stmt in body.seq[:-1]:
                        bindings[stmt.var] = tirx.stmt_functor.substitute(stmt.value, bindings)
                    body = body.seq[-1]
                elif isinstance(body, tirx.IfThenElse) and body.else_case is None:
                    conditions.append(body.condition)
                    body = body.then_case
                else:
                    break
            fill = _is_load(body)
            if fill is None:
                return None
            original_fill = fill
            fill = tirx.stmt_functor.substitute(fill, bindings)
            conditions = [tirx.stmt_functor.substitute(condition, bindings) for condition in conditions]
            dst = _dst_buffer(fill)
            source = fill.args[1]
            if dst is None or dst.scope() not in ("local", "shared") or len(dst.shape) != 2 or not _is_compact(dst):
                return None
            if not (isinstance(source, tirx.Call) and source.args and isinstance(source.args[0], tirx.BufferLoad)):
                return None
            rows, width, stride = (_int(fill.args[index]) for index in (3, 2, 4))
            if rows is None or rows <= 0 or width is None or width != stride:
                return None
            address = fill.args[0].args[0]
            if len(address.indices) != 2 or _uses_var(address.indices[1], node.loop_var):
                return None
            row = address.indices[0]
            analyzer = tvm.arith.Analyzer()
            delta = _subst(row, node.loop_var, node.loop_var + 1) - row
            if not analyzer.can_prove_equal(delta, rows):
                return None
            safe = [True]

            def inspect(expr):
                if isinstance(expr, tirx.Call):
                    safe[0] = False
                elif isinstance(expr, tirx.BufferLoad) and expr.buffer.data.same_as(dst.data):
                    safe[0] = False

            for expr in (conditions + list(bindings.values()) + list(source.args[0].indices)
                         + list(address.indices) + [node.min, node.extent]):
                post_order_visit(expr, inspect)
            if not safe[0]:
                return None

            def batch(stmt):
                if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(original_fill):
                    return tirx.Evaluate(tirx.call_intrin("handle", tirx.op.Op.get("tl.intel_npu_dma_batch"),
                                                         *list(original_fill.args[:6])))
                return None

            transformed = tirx.stmt_functor.ir_transform(node.body, None, batch, ["tirx.Evaluate"])
            loop = tirx.For(node.loop_var, node.min, node.extent, node.kind, transformed,
                            node.thread_binding, node.annotations, node.step)
            flush = tirx.Evaluate(tirx.call_intrin("handle", tirx.op.Op.get("tl.intel_npu_dma_batch_flush")))
            return _seq([loop, flush])

        body = tirx.stmt_functor.ir_transform(func.body, pre, None, ["tirx.For", "tirx.Bind", "tirx.Evaluate"])
        return func.with_body(body)

    return prim_func_pass(transform, opt_level=0, name="tl.PlanIntelNpuDmaBatch")


def PlanIntelNpuDpuOverlap():
    def transform(func: tirx.PrimFunc, mod, ctx):
        if not bool(ctx.config.get("tl.intel_npu.dpu_overlap", False)):
            return func
        planner = _DpuPlanner(func)
        result = func.with_body(planner.widen_allocations(planner.body(func.body)))
        selected = set()
        check_budget = tvm.get_global_func("tl.intel_npu.transform.CheckIntelNpuBudgets")()
        for key in planner.packed_candidates:
            candidate = _DpuPlanner(func, selected | {key})
            proposed = func.with_body(candidate.widen_allocations(candidate.body(func.body)))
            try:
                check_budget(tvm.IRModule.from_expr(proposed))
            except Exception as error:
                if "intel_npu: CMX budget exceeded" not in str(error):
                    raise
            else:
                selected.add(key)
                result = proposed
        return result

    return prim_func_pass(transform, opt_level=0, name="tl.PlanIntelNpuDpuOverlap")


def PlanIntelNpuPrefetch():
    def transform(func: tirx.PrimFunc, mod, ctx):
        if func.attrs is None or "target" not in func.attrs or func.attrs["target"].kind.name != "intel_npu":
            return func
        planner = _Planner(func)
        body = planner.widen_allocations(planner.body(func.body))
        if planner.done == 0:
            return func
        return func.with_body(body)

    return prim_func_pass(transform, opt_level=0, name="tl.PlanIntelNpuPrefetch")
