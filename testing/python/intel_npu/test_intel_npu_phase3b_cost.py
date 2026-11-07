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
    every row's golden PASSed and VPUNN's pick is within 5% of the measured best.
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
    with tvm.transform.PassContext(config=cfg):
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
