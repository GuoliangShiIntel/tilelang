"""Phase 3b Step 0: the VPUNN cost model and the DPU workload-shape knobs.

Hardware-free. The VPUNN tests need the pybind module (`VPUNN_ROOT` set and
`maint/scripts/build_intel_npu_vpunn.py` run) and SKIP otherwise, naming the
reason; the knob tests only need the compiler.

Acceptance with negative controls:
  * a GEMM tile is costed as a 1x1 CONVOLUTION and the cycle count moves with
    the MPE mode / spatial split (the knob reaches the model);
  * N or K not a multiple of 16 is refused; a spatial split that does not divide
    M is refused;
  * the pass configs `tl.intel_npu.dpu_spatial_x` / `dpu_mpe` reach the emitted
    `ctx.dpu_matmul(...)` call, and an invalid value is refused by name;
  * the device calibration (evidence/vpunn_calibration.json) is consistent:
    every row's golden PASSed and VPUNN's pick is within 5% of the measured best;
  * PlanIntelNpuWorkloads sets every DPU call's knobs and records what it chose
    in the source; it works unchanged on a kernel it was not written for; it
    keeps explicit knobs; "off" leaves defaults; without VPUNN "auto" leaves the
    defaults and says so, "require" refuses by name.
"""

import json
import sys
from pathlib import Path

import pytest

import tilelang
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import make_gemm_repeat  # noqa: E402
from tilelang.intel_npu import vpunn  # noqa: E402
from tilelang.intel_npu.transform import plan_workloads  # noqa: E402

HERE = Path(__file__).resolve().parent


def _vpunn_available() -> bool:
    try:
        vpunn.load()
        vpunn.model_path()
        return True
    except vpunn.VpunnUnavailable:
        return False


needs_vpunn = pytest.mark.skipif(not _vpunn_available(),
                                 reason="VPUNN not built: set VPUNN_ROOT and run maint/scripts/build_intel_npu_vpunn.py")


@needs_vpunn
def test_gemm_cost_moves_with_workload_shape():
    c = vpunn.GemmCost()
    default = c.dpu_cycles(64, 64, 512)  # x = 4, CUBOID_16x16 as the runtime programs
    ranked = c.best_stencil(64, 64, 512)
    assert ranked[0][0] < default, ranked
    cycles = {(m, x) for _, m, x in ranked}
    assert len(cycles) == len(ranked)
    assert c.dpu_cycles(64, 64, 2048) > default
    assert c.dma_cycles(65536) > 0


@needs_vpunn
def test_gemm_cost_refuses_illegal_shapes():
    c = vpunn.GemmCost()
    with pytest.raises(ValueError, match="N % 16"):
        c.dpu_cycles(64, 60, 512)
    with pytest.raises(ValueError, match="does not divide"):
        c.dpu_cycles(64, 64, 512, spatial_x=7)


def test_default_spatial_mirrors_runtime():
    assert vpunn.default_spatial_x(64) == 4
    assert vpunn.default_spatial_x(21) == 7
    assert vpunn.default_spatial_x(14) == 2
    assert vpunn.default_spatial_x(9) == 1


def _dpu_call(cfg):
    func = make_gemm_repeat(64, 64, 512, repeat=2, tiles=1, shaves=2)
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", **cfg}):
        src = tilelang.lower(func, target="intel_npu").kernel_source
    return [ln for ln in src.splitlines() if "ctx.dpu_matmul(" in ln][0]


def test_workload_knobs_reach_codegen():
    assert ", 64, 64, 512, 0, -1);" in _dpu_call({})
    assert ", 64, 64, 512, 16, 1);" in _dpu_call({"tl.intel_npu.dpu_spatial_x": 16, "tl.intel_npu.dpu_mpe": 1})


def test_workload_knobs_refused_by_name():
    with pytest.raises(Exception, match="dpu_spatial_x=7 must be 0 or divide"):
        _dpu_call({"tl.intel_npu.dpu_spatial_x": 7})
    with pytest.raises(Exception, match="dpu_mpe=5 is not"):
        _dpu_call({"tl.intel_npu.dpu_mpe": 5})


def test_calibration_evidence_is_consistent():
    path = HERE / "device" / "evidence" / "vpunn_calibration.json"
    if not path.exists():
        pytest.skip(f"no device calibration at {path} (run device/run_vpunn_calib_device.py)")
    data = json.loads(path.read_text())
    assert data["rows"], "empty calibration"
    assert all(r["golden"] for r in data["rows"]), [r for r in data["rows"] if not r["golden"]]
    for s in data["summary"]:
        assert s["regret_pct"] < 5.0, s


# -- PlanIntelNpuWorkloads ---------------------------------------------------------------

def _lower(func, cfg=None):
    with tvm.transform.PassContext(config=cfg or {}):
        return tilelang.lower(func, target="intel_npu").kernel_source


def _dpu_calls(src):
    return [ln.strip() for ln in src.splitlines() if "ctx.dpu_matmul(" in ln]


@needs_vpunn
def test_plan_workloads_sets_every_dpu_call_and_says_so():
    from kernels import make_gemm
    src = _lower(make_gemm(256, 256, 512, BM=64, BN=64, tiles=2, shaves=2))
    assert "// intel_npu: DPU workloads planned by VPUNN" in src
    calls = _dpu_calls(src)
    assert calls and all(not c.endswith(", 0, -1);") for c in calls), calls
    mpe, x, _ = plan_workloads.plan_gemm(64, 64, 512)
    assert calls[0].endswith(f", 64, 64, 512, {x}, {plan_workloads.MPE_CODE[mpe]});"), calls[0]


@needs_vpunn
def test_plan_workloads_is_generic_over_kernels():
    # Generality clause: a kernel the pass was not written for (two gemm phases, K/2), zero compiler edits.
    from kernels import make_gemm_two_phase
    src = _lower(make_gemm_two_phase(256, 256, 512, BM=64, BN=64, tiles=2, shaves=2))
    calls = _dpu_calls(src)
    assert calls and all(not c.endswith(", 0, -1);") for c in calls), calls
    assert "64x64x256 ->" in src


@needs_vpunn
def test_plan_workloads_keeps_explicit_knobs_and_off_mode():
    from kernels import make_gemm
    f = make_gemm(256, 256, 512, BM=64, BN=64, tiles=2, shaves=2)
    kept = _lower(f, {"tl.intel_npu.dpu_spatial_x": 4, "tl.intel_npu.dpu_mpe": 2})
    assert _dpu_calls(kept)[0].endswith(", 4, 2);") and "kept author/config knobs" in kept
    off = _lower(f, {"tl.intel_npu.plan_workloads": "off"})
    assert _dpu_calls(off)[0].endswith(", 0, -1);") and "intel_npu: DPU workloads" not in off


def test_plan_workloads_without_vpunn(monkeypatch):
    from kernels import make_gemm

    def unavailable():
        raise vpunn.VpunnUnavailable("VPUNN not configured (test)")

    monkeypatch.setattr(plan_workloads, "_cost_model", unavailable)
    f = make_gemm(256, 256, 512, BM=64, BN=64, tiles=2, shaves=2)
    src = _lower(f)  # auto: runtime defaults, and the source says why
    assert _dpu_calls(src)[0].endswith(", 0, -1);")
    assert "// intel_npu: DPU workloads NOT planned (1 left at the runtime default): VPUNN not configured (test)" in src
    with pytest.raises(Exception, match="plan_workloads=require but VPUNN not configured"):
        _lower(f, {"tl.intel_npu.plan_workloads": "require"})
    with pytest.raises(Exception, match="is not auto, require or off"):
        _lower(f, {"tl.intel_npu.plan_workloads": "bogus"})
