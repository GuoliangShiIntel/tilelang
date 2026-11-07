"""PlanIntelNpuWorkloads: choose each DPU workload's shape with the VPUNN cost model.

VPUX picks the MPE cuboid and the spatial split of every DPU workload by cost
(`split_nce_ops_onto_workloads.cpp`, `dpu_tiler.cpp`; plan section 3.2). This
pass does the same for every `intel_npu_dpu_matmul(out, a, bt, M, N, K, spatial_x,
mpe)` call left by LowerIntelNpuTileOp whose knobs are still the runtime
defaults (0, -1): it ranks the (cuboid, x) candidates with
`tilelang.intel_npu.vpunn.GemmCost` and writes the cheapest one into the call.
Knobs already set (the kernel-wide `tl.intel_npu.dpu_spatial_x` / `dpu_mpe`
configs, used by the calibration sweep) are left alone.

Modes, `tl.intel_npu.plan_workloads`:
  * "auto" (default): plan when VPUNN is available; otherwise leave the
    runtime defaults and say so in the generated source (`intel_npu.notes`).
  * "require": refuse by name when VPUNN is unavailable.
  * "off": leave every call as it is.

Device calibration 2026-11-06 (`device/evidence/vpunn_calibration.json`): the
runtime default is 36-42% slower than VPUNN's pick on four tile shapes.
"""

from __future__ import annotations

from functools import lru_cache

import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform
from tvm.tirx.transform import prim_func_pass

from .. import vpunn

PLAN_WORKLOADS_KEY = "tl.intel_npu.plan_workloads"
NOTES_ATTR = "intel_npu.notes"
DPU_MATMUL_OP = "tl.intel_npu_dpu_matmul"
# Runtime encoding of the MPE cuboid (npu_rt_device.hpp dpu_matmul).
MPE_CODE = {"CUBOID_4x16": 0, "CUBOID_8x16": 1, "CUBOID_16x16": 2}


def _mode() -> str:
    ctx = tvm.transform.PassContext.current()
    mode = str(ctx.config.get(PLAN_WORKLOADS_KEY, "auto"))
    if mode not in ("auto", "require", "off"):
        raise ValueError(f"intel_npu: {PLAN_WORKLOADS_KEY}={mode!r} is not auto, require or off")
    return mode


@lru_cache(maxsize=None)
def _cost_model():
    return vpunn.GemmCost()


@lru_cache(maxsize=None)
def plan_gemm(M: int, N: int, K: int) -> tuple[str, int, int]:
    """(cuboid, spatial_x, vpunn_cycles) of the cheapest legal configuration."""
    xs = [x for x in range(1, min(M, 64) + 1) if M % x == 0]
    ranked = _cost_model().best_stencil(M, N, K, spatial_xs=xs)
    if not ranked:
        raise ValueError(f"intel_npu: VPUNN costed no configuration for a {M}x{N}x{K} DPU workload")
    cycles, mpe, x = ranked[0]
    return mpe, x, cycles


def _with_notes(func: tirx.PrimFunc, text: str) -> tirx.PrimFunc:
    # Inside the outermost thread_extent: that is where the device region starts
    # (AnnotateDeviceRegions), and SplitHostDevice drops PrimFunc attributes it does not know.
    def wrap(stmt):
        if isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "thread_extent":
            notes = tirx.AttrStmt(tirx.IntImm("int32", 0), NOTES_ATTR, tirx.StringImm(text), stmt.body)
            return tirx.AttrStmt(stmt.node, stmt.attr_key, stmt.value, notes)
        return None

    return func.with_body(ir_transform(func.body, wrap, None, ["tirx.AttrStmt"]))


def PlanIntelNpuWorkloads():
    mode = _mode()

    @prim_func_pass(opt_level=0, name="tl.PlanIntelNpuWorkloads")
    def _pass(func: tirx.PrimFunc, mod, ctx):
        if mode == "off":
            return func
        op = tvm.ir.Op.get(DPU_MATMUL_OP)
        calls = []

        def find(stmt):
            if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call) and stmt.value.op.same_as(op):
                calls.append(stmt)
            return None

        ir_transform(func.body, None, find, ["tirx.Evaluate"])
        if not calls:
            return func
        try:
            cost = _cost_model()
        except vpunn.VpunnUnavailable as e:
            if mode == "require":
                raise ValueError(f"intel_npu: {PLAN_WORKLOADS_KEY}=require but {e}") from e
            return _with_notes(func, f"DPU workloads NOT planned ({len(calls)} left at the runtime default): {e}")

        planned = []

        def rewrite(stmt):
            if not (isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call) and stmt.value.op.same_as(op)):
                return None
            args = list(stmt.value.args)
            M, N, K = (int(args[i].value) for i in (3, 4, 5))
            if int(args[6].value) != 0 or int(args[7].value) != -1:
                planned.append(f"{M}x{N}x{K}: kept author/config knobs x={int(args[6].value)} mpe={int(args[7].value)}")
                return None
            try:
                mpe, x, cycles = plan_gemm(M, N, K)
                default_cycles = cost.dpu_cycles(M, N, K)
            except (ValueError, RuntimeError) as e:
                # Outside the model's domain (e.g. a huge K): the runtime default stays, visibly.
                planned.append(f"{M}x{N}x{K}: left at the runtime default, VPUNN: {str(e).splitlines()[-1]}")
                return None
            args[6] = tirx.IntImm("int32", x)
            args[7] = tirx.IntImm("int32", MPE_CODE[mpe])
            planned.append(f"{M}x{N}x{K} -> {mpe} x={x} ({cycles} cyc; runtime default {default_cycles})")
            return tirx.Evaluate(tirx.Call(stmt.value.dtype, op, args))

        body = ir_transform(func.body, None, rewrite, ["tirx.Evaluate"])
        func = func.with_body(body)
        return _with_notes(func, f"DPU workloads planned by VPUNN {vpunn.version()}: " + "; ".join(planned))

    return _pass
