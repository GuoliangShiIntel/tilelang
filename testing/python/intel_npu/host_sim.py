"""Host simulation of Intel NPU kernels: compile the generated SHAVE C++ against
`npu_rt_host.hpp` with clang++, load it, and run `kernel_entry` on
tiles x shaves_per_tile threads.

The simulation is a CPU model of the runtime surface (DMA = memcpy, DPU = fp32
accumulate rounded to fp16), not of the hardware: it checks the compiler's
addressing, dealing and operand plumbing, which is what a wrong tile stride or
block index breaks.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TL_ROOT = Path(__file__).resolve().parents[3]
INCLUDE_ROOT = TL_ROOT / "src"  # generated code includes <tl_templates/intel_npu/npu_rt.hpp>
STATUS_BYTES_PER_SHAVE = 32
STATUS_OK = 1
STATUS_NAMES = {1: "OK", 2: "CMX_EXHAUSTED", 3: "BAD_PARAMS", 4: "UNSUPPORTED", 5: "BARRIER_NOT_LIVE", 6: "BARRIER_TIMEOUT"}


def find_clangxx() -> str:
    for cand in (os.environ.get("CXX"), os.environ.get("TL_HOST_CXX"), shutil.which("clang++")):
        if cand and Path(cand).exists():
            return cand
    raise RuntimeError("clang++ not found: set TL_HOST_CXX or put clang++ on PATH (the sim needs _Float16)")


def build(source: str, name: str, out_dir: str | Path | None = None) -> Path:
    """Compile `source` (generated kernel C++) into a shared library exporting npu_sim_run."""
    out_dir = Path(out_dir) if out_dir else Path(tempfile.gettempdir()) / "tl_intel_npu_sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    src = out_dir / f"{name}_{digest}.cpp"
    lib = out_dir / f"{name}_{digest}.{'dll' if sys.platform == 'win32' else 'so'}"
    if lib.exists():
        return lib
    src.write_text(source, encoding="utf-8")
    cmd = [find_clangxx(), "-std=c++17", "-O2", "-shared", "-DNPU_RT_HOST_SIM", f"-I{INCLUDE_ROOT}", str(src), "-o", str(lib)]
    # Inline fp16<->fp32 conversions (F16C): without them clang emits __extendhfsf2 libcalls that are not in the MSVC link.
    cmd[2:2] = ["-mavx2", "-mf16c", "-mfma"]
    if sys.platform != "win32":
        cmd.insert(3, "-fPIC")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"host-sim build failed:\n{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return lib


def run(lib_path: Path, arrays: list[np.ndarray], tiles: int, shaves: int, cmx_bytes: int = 1_572_864) -> np.ndarray:
    """Run the kernel with `arrays` as its tensor parameters (in order; modified in place).
    Returns the per-SHAVE status records as an int array [tiles*shaves, 8]."""
    lib = ctypes.CDLL(str(lib_path))
    lib.npu_sim_run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.npu_sim_run.restype = ctypes.c_int
    workers = tiles * shaves
    status = np.full(workers * STATUS_BYTES_PER_SHAVE // 4, 0xFFFFFFFF, dtype=np.uint32)
    for a in arrays:
        assert a.flags["C_CONTIGUOUS"], "tensor parameters must be C-contiguous"
    words = [a.ctypes.data for a in arrays] + [status.ctypes.data]
    params = (ctypes.c_uint64 * len(words))(*words)
    rc = lib.npu_sim_run(len(words), params, tiles, shaves, cmx_bytes)
    if rc != 0:
        raise RuntimeError(f"npu_sim_run returned {rc}")
    return status.reshape(workers, 8)


def check_status(status: np.ndarray) -> None:
    bad = [(w, int(s[0]), int(s[1])) for w, s in enumerate(status) if s[0] != STATUS_OK]
    if bad:
        desc = ", ".join(f"shave {w}: {STATUS_NAMES.get(c, c)} detail={d}" for w, c, d in bad)
        raise RuntimeError(f"kernel reported failure: {desc}")
