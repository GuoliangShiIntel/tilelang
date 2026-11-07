"""VPUNN cost model access for the Intel NPU backend.

VPUNN is the production NPU compiler's cost model (plan section 3.2). It is an
external pinned dependency: ``VPUNN_ROOT`` names the checkout and
``maint/scripts/build_intel_npu_vpunn.py`` builds its pybind module into
``<VPUNN_ROOT>/build-py/lib`` (override with ``VPUNN_PYTHON_DIR``). Nothing here is
imported by the pass pipeline unless a cost-driven pass asks for it, so a tree
without VPUNN still lowers and runs; the cost-driven passes refuse by name instead.

A DPU GEMM ``[M, K] x [N, K]^T`` is costed the way VPUX costs ``NCE.MatMul``: as a
1x1 CONVOLUTION whose activation is ``W x H x K`` with ``W * H = M`` and whose
output has ``N`` channels (``vpux_compiler/src/core/cost_model_utils.cpp``,
``getDPUWorkload``). The spatial split follows ``DpuMatMul::init`` in the device
runtime (x = 4 when 4 | M, 7 when 7 | M > 14, else 2 or 1) unless given, and the
default stencil is the one ``init`` programs (``nthw_ntk`` 16_4 = ``CUBOID_16x16``).
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# DPU clock VPUNN cycles are counted at; VPUX uses get_dpu_fclk(NPU_5_0) = 1950 MHz
# (vpux_compiler/src/dialect/VPU/IR/attributes.cpp). Our measured peak basis is 1850.
VPUNN_DPU_FCLK_MHZ = 1950.0

# DpuMatMul::setStencil(nthwNtk) values -> VPUNN execution mode names.
STENCIL_TO_MPE = {3: "CUBOID_16x16", 0: "CUBOID_8x16", 1: "CUBOID_4x16"}
MPE_TO_STENCIL = {v: k for k, v in STENCIL_TO_MPE.items()}

_mod = None
# Cycles::START_ERROR_RANGE (vpu/cycles_interface_types.h): values above are error codes.
_ERROR_RANGE_START = 0xFFFFFFFF - 1000


class VpunnUnavailable(RuntimeError):
    pass


def root() -> Optional[Path]:
    r = os.environ.get("VPUNN_ROOT")
    return Path(r) if r else None


def python_dir() -> Optional[Path]:
    d = os.environ.get("VPUNN_PYTHON_DIR")
    if d:
        return Path(d)
    r = root()
    return r / "build-py" / "lib" if r else None


def load():
    """Import and return the ``_VPUNN`` pybind module, or raise ``VpunnUnavailable``."""
    global _mod
    if _mod is not None:
        return _mod
    d = python_dir()
    if d is None:
        raise VpunnUnavailable("VPUNN not configured: set VPUNN_ROOT (and build it with "
                               "maint/scripts/build_intel_npu_vpunn.py) or VPUNN_PYTHON_DIR")
    if not any(d.glob("_VPUNN*.pyd")) and not any(d.glob("_VPUNN*.so")):
        raise VpunnUnavailable(f"no _VPUNN module in {d}; run maint/scripts/build_intel_npu_vpunn.py")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    _mod = importlib.import_module("_VPUNN")
    return _mod


def version() -> str:
    r = root()
    if r is None:
        return "unknown"
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", str(r), "describe", "--tags", "--always"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def model_path(name: str = "vpu_5_1.vpunn") -> Path:
    r = root()
    if r is None:
        raise VpunnUnavailable("VPUNN_ROOT not set; cannot locate the .vpunn model files")
    p = r / "models" / name
    if not p.exists():
        raise VpunnUnavailable(f"model file missing: {p}")
    return p


def default_spatial_x(M: int) -> int:
    """Mirror DpuMatMul::init's spatial choice (dpu_matmul.hpp)."""
    x = 1
    if M % 2 == 0:
        x = 2
    if M % 4 == 0:
        x = 4
    if M % 7 == 0 and M > 14:
        x = 7
    return x


@dataclass(frozen=True)
class GemmWorkload:
    M: int
    N: int
    K: int
    spatial_x: int
    mpe: str = "CUBOID_16x16"
    dtype: str = "FLOAT16"
    output_write_tiles: int = 1
    isi: str = "CLUSTERING"

    @property
    def spatial_y(self) -> int:
        return self.M // self.spatial_x

    @property
    def macs(self) -> int:
        return self.M * self.N * self.K


class GemmCost:
    """DPU and DMA cycle estimates for the backend's GEMM building blocks."""

    def __init__(self, device: str = "NPU_5_0", dpu_model: str = "vpu_5_1.vpunn",
                 dma_model: str = "dma_5_0.vpunn"):
        self.v = load()
        self.device = getattr(self.v.VPUDevice, device)
        self.dpu = self.v.VPUCostModel(self.device, str(model_path(dpu_model)))
        self._dma = None
        self._dma_model = dma_model

    # -- DPU -------------------------------------------------------------------
    def workload(self, M: int, N: int, K: int, spatial_x: Optional[int] = None,
                 mpe: str = "CUBOID_16x16", dtype: str = "FLOAT16",
                 output_write_tiles: int = 1, isi: str = "CLUSTERING") -> GemmWorkload:
        if N % 16 or K % 16:
            raise ValueError(f"DPU GEMM needs N % 16 == 0 and K % 16 == 0 (N={N}, K={K})")
        x = spatial_x if spatial_x is not None else default_spatial_x(M)
        if M % x:
            raise ValueError(f"spatial_x={x} does not divide M={M}")
        return GemmWorkload(M, N, K, x, mpe, dtype, output_write_tiles, isi)

    def to_dpu_workload(self, w: GemmWorkload):
        v = self.v
        dt = getattr(v.DataType, w.dtype)
        wl = v.DPUWorkload()
        wl.device = self.device
        wl.op = v.Operation.CONVOLUTION
        wl.inputs = [v.VPUTensor(w.spatial_x, w.spatial_y, w.K, 1, dt, v.Layout.ZXY)]
        wl.outputs = [v.VPUTensor(w.spatial_x, w.spatial_y, w.N, 1, dt, v.Layout.ZXY)]
        wl.kernels = [1, 1]
        wl.strides = [1, 1]
        wl.padding = [0, 0, 0, 0]
        wl.execution_order = getattr(v.ExecutionMode, w.mpe)
        wl.output_write_tiles = w.output_write_tiles
        wl.isi_strategy = getattr(v.ISIStrategy, w.isi)
        return wl

    def dpu_cycles(self, M: int, N: int, K: int, **kw) -> int:
        w = self.workload(M, N, K, **kw)
        cycles, msg = self.dpu.DPUMsg(self.to_dpu_workload(w))
        if cycles > _ERROR_RANGE_START:
            raise RuntimeError(f"VPUNN refused {w}: {msg}")
        return int(cycles)

    def dpu_ideal_cycles(self, M: int, N: int, K: int, **kw) -> int:
        w = self.workload(M, N, K, **kw)
        return int(self.dpu.DPUIdealCycles(self.to_dpu_workload(w)))

    def best_stencil(self, M: int, N: int, K: int, spatial_xs=None, **kw):
        """Rank (mpe, spatial_x) pairs by VPUNN DPU cycles; the first entry is the best."""
        xs = spatial_xs or sorted({x for x in (1, 2, 4, 7, 8, 16, 32) if M % x == 0})
        rows = []
        for mpe in STENCIL_TO_MPE.values():
            for x in xs:
                try:
                    rows.append((self.dpu_cycles(M, N, K, spatial_x=x, mpe=mpe, **kw), mpe, x))
                except RuntimeError:
                    continue
        return sorted(rows)

    # -- DMA -------------------------------------------------------------------
    @property
    def dma(self):
        if self._dma is None:
            self._dma = self.v.DMACostModel_NPU40_50(str(model_path(self._dma_model)))
        return self._dma

    def dma_cycles(self, nbytes: int, direction: str = "DDR2CMX", engines: int = 1) -> int:
        """1-D transfer cost in DPU cycles (the binding exposes no strided dims)."""
        v = self.v
        wl = v.DMANNWorkload_NPU40_50(self.device)
        wl.src_width = int(nbytes)
        wl.dst_width = int(nbytes)
        wl.num_dim = 0
        wl.num_engine = v.Num_DMA_Engine.Num_Engine_2 if engines == 2 else v.Num_DMA_Engine.Num_Engine_1
        wl.transfer_direction = getattr(v.MemoryDirection, direction)
        cycles, msg = self.dma.computeCyclesMsg(wl)
        if cycles > _ERROR_RANGE_START:
            raise RuntimeError(f"VPUNN DMA refused {nbytes} B {direction}: {msg}")
        return int(cycles)

    # -- units -----------------------------------------------------------------
    @staticmethod
    def cycles_to_us(cycles: int, fclk_mhz: float = VPUNN_DPU_FCLK_MHZ) -> float:
        return cycles / fclk_mhz
