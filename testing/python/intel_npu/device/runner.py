"""Device runner for TileLang Intel NPU kernels (PTL/NPU5 on this machine).

Flow: lower -> write generated/<name>.cpp -> cmake configure + build (host runner
with MSVC, kernels with moviCompile) -> write parameter files -> run npu_run ->
read outputs and the per-SHAVE status area.

Requires GENERIC_COMPUTE_ROOT and MV_TOOLS_PATH in the environment (never
defaulted here). Timing printed by npu_run is host wall time around
submit+wait and is UNPROFILED unless `profile_json` is given, in which case the
run is an observability run and its time is not a latency claim.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
STATUS_BYTES_PER_SHAVE = 32
STATUS_OK = 1
STATUS_NAMES = {1: "OK", 2: "CMX_EXHAUSTED", 3: "BAD_PARAMS", 4: "UNSUPPORTED"}


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; the device runner never defaults SDK paths")
    if not Path(value).exists():
        raise RuntimeError(f"{name}={value} does not exist")
    return value


def find_cmake() -> str:
    for cand in (shutil.which("cmake"), os.environ.get("CMAKE")):
        if cand and Path(cand).exists():
            return cand
    vs = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")
    if vs.exists():
        return str(vs)
    raise RuntimeError("cmake not found")


@dataclass
class DeviceBuild:
    build_dir: Path
    exe: Path
    kernel: str
    source_sha256: str


def write_kernel(source: str, kernel: str) -> tuple[Path, str]:
    GENERATED.mkdir(exist_ok=True)
    path = GENERATED / f"{kernel}.cpp"
    path.write_text(source, encoding="utf-8")
    return path, hashlib.sha256(source.encode()).hexdigest()


def build(source: str, kernel: str, build_dir: Path | None = None, log=print) -> DeviceBuild:
    gc_root = require_env("GENERIC_COMPUTE_ROOT")
    mv_tools = require_env("MV_TOOLS_PATH")
    build_dir = build_dir or (HERE / "build")
    path, sha = write_kernel(source, kernel)
    log(f"kernel source: {path} sha256 {sha[:12]}")
    cmake = find_cmake()
    env = dict(os.environ)
    # Always configure: the kernel list is a CMake glob, so a new generated/<kernel>.cpp needs a re-run.
    r = subprocess.run(
        [cmake, "-G", "Visual Studio 17 2022", "-A", "x64", "-S", str(HERE), "-B", str(build_dir),
         f"-DGENERIC_COMPUTE_ROOT={gc_root}", f"-DMV_TOOLS_PATH={mv_tools}"],
        capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"cmake configure failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    r = subprocess.run([cmake, "--build", str(build_dir), "--config", "Release", "--", "/m", "/v:minimal"],
                       capture_output=True, text=True, env=env)
    errs = [ln for ln in (r.stdout + r.stderr).splitlines() if " error" in ln.lower()]
    if r.returncode:
        raise RuntimeError("device build failed:\n" + "\n".join(errs[:30] or [r.stdout[-4000:]]))
    exe = build_dir / "Release" / "npu_run.exe"
    if not exe.exists():
        raise RuntimeError(f"runner not found at {exe}")
    elf = next(build_dir.rglob(f"*{kernel}*.elf"), None)
    if elf is None:
        raise RuntimeError(f"no .elf for kernel {kernel} under {build_dir}")
    log(f"built {exe.name} and {elf.name}")
    return DeviceBuild(build_dir=build_dir, exe=exe, kernel=kernel, source_sha256=sha)


@dataclass
class DeviceResult:
    median_ms: float
    times_ms: list[float]
    status: np.ndarray  # [workers, 8] uint32
    stdout: str


def run(db: DeviceBuild, arrays: list[tuple[str, np.ndarray]], tiles: int, shaves: int, run_dir: Path,
        repeat: int = 10, warmup: int = 2, profile_json: Path | None = None, timeout_s: int = 300) -> DeviceResult:
    """`arrays` are (mode, array) in kernel parameter order; mode in {in, out, inout}. Outputs are read back in place."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in run_dir.iterdir():
        f.unlink()
    lines = []
    for i, (mode, a) in enumerate(arrays):
        name = f"p{i}.bin"
        if mode in ("in", "inout"):
            np.ascontiguousarray(a).tofile(run_dir / name)
        lines.append(f"{mode} {a.nbytes} {name}")
    (run_dir / "params.txt").write_text("\n".join(lines) + "\n")
    cmd = [str(db.exe), "--kernel", db.kernel, "--dir", str(run_dir), "--tiles", str(tiles), "--shaves", str(shaves),
           "--repeat", str(repeat), "--warmup", str(warmup)]
    env = dict(os.environ)
    if profile_json is not None:
        cmd += ["--profile", str(profile_json)]
        env["NPU_PROFILE_FRC_MHZ"] = "19.2"  # the runtime's 38.4 default halves every duration on this box
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(db.exe.parent), env=env, timeout=timeout_s)
    if r.returncode or "median_ms" not in r.stdout:
        raise RuntimeError(f"device run failed (exit {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    for i, (mode, a) in enumerate(arrays):
        if mode in ("out", "inout"):
            data = np.fromfile(run_dir / f"p{i}.bin", dtype=a.dtype)
            a[...] = data.reshape(a.shape)
    workers = tiles * shaves
    status = np.fromfile(run_dir / "status.bin", dtype=np.uint32)[: workers * 8].reshape(workers, 8)
    times = [float(ln.split()[-1]) for ln in r.stdout.splitlines() if ln.startswith("run ")]
    median = float(r.stdout.split("median_ms")[1].split()[0])
    return DeviceResult(median_ms=median, times_ms=times, status=status, stdout=r.stdout)


def check_status(status: np.ndarray) -> None:
    bad = [(w, int(s[0]), int(s[1])) for w, s in enumerate(status) if s[0] != STATUS_OK]
    if bad:
        desc = ", ".join(f"shave {w}: {STATUS_NAMES.get(c, c)} detail={d}" for w, c, d in bad)
        raise RuntimeError(f"kernel reported failure: {desc}")
