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
STATUS_BYTES_PER_SHAVE = 64
STATUS_WORDS = STATUS_BYTES_PER_SHAVE // 4
STATUS_OK = 1
STATUS_NAMES = {1: "OK", 2: "CMX_EXHAUSTED", 3: "BAD_PARAMS", 4: "UNSUPPORTED", 5: "BARRIER_NOT_LIVE(id=detail)", 6: "BARRIER_TIMEOUT(id=detail)",
                7: "RING_REFUSED(1 slots!=shaves, 2 one tile, 3 barrier pool)", 8: "ORDER_TIMEOUT(need=detail)",
                9: "DPU_TIMEOUT(ring use=detail)", 10: "DMA_TIMEOUT(where=detail: 1/2 fill prev halves, 3/4 ring_end, 5/6 port_retire)"}


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
    text_bytes: int = 0  # executable section bytes of the SHAVE .elf (0 = not measured)


def elf_section_sizes(elf: Path) -> dict[str, tuple[int, int]]:
    """Section name -> (flags, size) of a little-endian ELF32/ELF64 (the SHAVE .elf is ELF32)."""
    import struct

    data = elf.read_bytes()
    assert data[:4] == b"\x7fELF", f"{elf} is not an ELF"
    is64 = data[4] == 2
    if is64:
        shoff, = struct.unpack_from("<Q", data, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    else:
        shoff, = struct.unpack_from("<I", data, 0x20)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x2E)
    headers = []
    for i in range(shnum):
        off = shoff + i * shentsize
        if is64:
            name, typ, flags, _addr, _off, size = struct.unpack_from("<IIQQQQ", data, off)
        else:
            name, typ, flags, _addr, _off, size = struct.unpack_from("<IIIIII", data, off)
        headers.append((name, typ, flags, size))
    str_off = shoff + shstrndx * shentsize
    strtab_off = struct.unpack_from("<Q" if is64 else "<I", data, str_off + (0x18 if is64 else 0x10))[0]
    out = {}
    for name, typ, flags, size in headers:
        end = data.index(b"\0", strtab_off + name)
        out[data[strtab_off + name:end].decode()] = (flags, size)
    return out


SHF_EXECINSTR = 0x4


def text_bytes(elf: Path) -> int:
    return sum(size for _, (flags, size) in elf_section_sizes(elf).items() if flags & SHF_EXECINSTR)


def write_kernel(source: str, kernel: str) -> tuple[Path, str]:
    GENERATED.mkdir(exist_ok=True)
    path = GENERATED / f"{kernel}.cpp"
    path.write_text(source, encoding="utf-8")
    return path, hashlib.sha256(source.encode()).hexdigest()


def remove_kernels(names) -> int:
    """Sweeps call this when done: every device build compiles the whole generated/ glob through MSBuild
    (~0.2 s per kernel plus a 300-project configure), so 267 leftover sweep arms once made each build ~50 s."""
    n = 0
    for name in names:
        p = GENERATED / f"{name}.cpp"
        if p.exists():
            p.unlink()
            n += 1
    return n


def build(source: str, kernel: str, build_dir: Path | None = None, log=print, text_limit_bytes: int | None = None) -> DeviceBuild:
    """`text_limit_bytes` defaults to the intel_npu target's attribute; the build is refused above it."""
    return build_all({kernel: source}, build_dir=build_dir, log=log, text_limit_bytes=text_limit_bytes)[kernel]


def build_all(sources: dict[str, str], build_dir: Path | None = None, log=print,
              text_limit_bytes: int | None = None) -> dict[str, DeviceBuild]:
    """Writes every kernel, builds the project once (the glob compiles all of generated/), checks each `.text`."""
    if text_limit_bytes is None:
        from tvm.target import Target

        text_limit_bytes = int(Target({"kind": "intel_npu"}).attrs["text_limit_bytes"])
    gc_root = require_env("GENERIC_COMPUTE_ROOT")
    mv_tools = require_env("MV_TOOLS_PATH")
    build_dir = build_dir or (HERE / "build")
    shas = {}
    for kernel, source in sources.items():
        path, shas[kernel] = write_kernel(source, kernel)
        log(f"kernel source: {path} sha256 {shas[kernel][:12]}")
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
    out = {}
    for kernel, sha in shas.items():
        elf = next(build_dir.rglob(f"*{kernel}*.elf"), None)
        if elf is None:
            raise RuntimeError(f"no .elf for kernel {kernel} under {build_dir}")
        text = text_bytes(elf)
        log(f"built {exe.name} and {elf.name}; SHAVE .text {text} B of {text_limit_bytes} B ({100.0 * text / text_limit_bytes:.1f}%)")
        if text > text_limit_bytes:
            raise RuntimeError(f"kernel {kernel}: SHAVE .text {text} B exceeds text_limit_bytes {text_limit_bytes}")
        out[kernel] = DeviceBuild(build_dir=build_dir, exe=exe, kernel=kernel, source_sha256=sha, text_bytes=text)
    return out


@dataclass
class DeviceResult:
    median_ms: float          # host wall incl. submit
    median_kernel_ms: float   # earliest SHAVE entry stamp to latest finish stamp (NCE_TIME, 19.2 MHz)
    times_ms: list[float]
    status: np.ndarray  # [workers, 16] uint32
    stdout: str
    turbo: bool = True        # NPU_TURBO=YES was set for the run (the measurement condition of every quoted number)
    kernel_times_ms: list[float] | None = None   # per timed run, kernel-only


def turbo_requested() -> bool:
    """NPU_TURBO=YES is the default (npu_dsl's Frequency Validation Rule; the DPU and SHAVE clocks are ~2x lower
    without it). TL_NPU_TURBO=0 runs without it, for an A/B only."""
    return os.environ.get("TL_NPU_TURBO", "1") not in ("0", "", "NO", "no")


def condition(res: "DeviceResult", profiled: bool) -> str:
    return ("PROFILED (observability, not a timing)" if profiled else "UNPROFILED") + (", TURBO" if res.turbo else ", NO TURBO")


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
    turbo = turbo_requested()
    if turbo:
        env["NPU_TURBO"] = "YES"
    else:
        env.pop("NPU_TURBO", None)
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
    status = np.fromfile(run_dir / "status.bin", dtype=np.uint32)[: workers * STATUS_WORDS].reshape(workers, STATUS_WORDS)
    times = [float(ln.split()[3]) for ln in r.stdout.splitlines() if ln.startswith("run ")]
    kernel_times = [float(ln.split()[5]) for ln in r.stdout.splitlines() if ln.startswith("run ") and "kernel_ms" in ln]
    median = float(r.stdout.split("median_ms")[1].split()[0])
    median_kernel = float(r.stdout.split("median_kernel_ms")[1].split()[0])
    return DeviceResult(median_ms=median, median_kernel_ms=median_kernel, times_ms=times, status=status, stdout=r.stdout, turbo=turbo,
                        kernel_times_ms=kernel_times)


def check_status(status: np.ndarray) -> None:
    bad = [(w, int(s[0]), int(s[1])) for w, s in enumerate(status) if s[0] != STATUS_OK]
    if bad:
        desc = ", ".join(f"shave {w}: {STATUS_NAMES.get(c, c)} detail={d}" for w, c, d in bad)
        raise RuntimeError(f"kernel reported failure: {desc}")


class Session:
    """`npu_run --session`: one process, kernels created once, device buffers resident across launches. Buffers are
    named; `launch` takes the names in the kernel's device parameter order and returns (host_ms, kernel_ms).
    Host <-> device bytes go through files in `run_dir` (a one-time cost for weights; a few KB per token after)."""

    def __init__(self, exe: Path, run_dir: Path, tiles: int, shaves: int):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if turbo_requested():
            env["NPU_TURBO"] = "YES"
        else:
            env.pop("NPU_TURBO", None)
        self.turbo = turbo_requested()
        self.proc = subprocess.Popen([str(exe), "--session", "--dir", str(self.run_dir), "--tiles", str(tiles), "--shaves", str(shaves)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     bufsize=1, cwd=str(Path(exe).parent), env=env)
        ready = self.proc.stdout.readline()
        if not ready.startswith("session ready"):
            raise RuntimeError(f"session did not start: {ready!r}\n{self.proc.stderr.read()[-2000:]}")
        self.sizes: dict[str, int] = {}
        self.launches: list[tuple[str, float, float]] = []   # (alias, host_ms, kernel_ms)
        self._n = 0
        self._fills = 0
        self._filled = 0        # bytes spent keeping buffers off a 4 GiB boundary (see `alloc`)
        self._next = None       # predicted address of the next allocation; resynced from the real one each time
        self._gran = None

    def _cmd(self, line: str) -> str:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        r = self.proc.stdout.readline().rstrip("\n")
        if not r:
            raise RuntimeError(f"session died on {line!r}: {self.proc.stderr.read()[-3000:]}")
        if r.startswith("error") or r.startswith("fatal"):
            tail = ""
            if r.startswith("fatal"):   # the process is exiting: its stderr carries the watchdog / health dump
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                tail = "\n" + self.proc.stderr.read()[-4000:]
            raise RuntimeError(f"session {line.split()[0]} {line.split()[1] if len(line.split()) > 1 else ''}: {r}{tail}")
        return r

    # A DMA whose buffer STRADDLES a 4 GiB boundary faults the device (ZE_RESULT_ERROR_DEVICE_LOST); a buffer
    # wholly above one is fine. Measured on the prefill model: 9 layers at pad 0.2 GiB (top 4.12 GB) returns,
    # pad 0.4 GiB (top 4.32 GB) faults, and 10 layers -- which always faulted -- returns once a 4.5 GiB pad puts
    # every weight above the line with nothing across it. The big MoE expert tensors (269 MB) hit it first simply
    # because they are wide enough to land across a boundary.
    _BOUNDARY = 1 << 32

    @staticmethod
    def _straddles(addr: int, nbytes: int) -> bool:
        return nbytes > 0 and addr // Session._BOUNDARY != (addr + nbytes - 1) // Session._BOUNDARY

    def alloc(self, name: str, nbytes: int) -> None:
        nbytes = int(nbytes)
        if self._next is not None and self._straddles(self._next, nbytes):
            fill = (self._next // self._BOUNDARY + 1) * self._BOUNDARY - self._next
            self._cmd(f"alloc _fill{self._fills} {fill}")
            self._fills += 1
            self._filled += fill
            self._next = None          # resynced from the real address below
        self._cmd(f"alloc {name} {nbytes}")
        self.sizes[name] = nbytes
        addr = int(self._cmd(f"addr {name}").split()[1])
        # the prediction is only a guess about the allocator; this is the check that the buffer is actually clear
        assert not self._straddles(addr, nbytes), (
            f"{name}: {nbytes} B at {addr:#x} straddles a 4 GiB boundary -- the allocator model in Session.alloc "
            f"is wrong, and a DMA of this buffer would fault the device")
        if self._gran is None:
            self._gran = 1 << 12
        self._next = addr + -(-nbytes // self._gran) * self._gran

    def upload(self, name: str, a: np.ndarray, offset: int = 0) -> None:
        """Allocates `name` on first use (exact size), then copies `a` in."""
        a = np.ascontiguousarray(a)
        if name not in self.sizes:
            self.alloc(name, a.nbytes)
        self._n += 1
        f = self.run_dir / f"xfer_{self._n % 4}.bin"
        a.tofile(f)
        self._cmd(f"load {name} {f.name} {int(offset)}")

    def upload_npy(self, name: str, path: Path) -> int:
        """A .npy file straight into a new buffer `name` (its data read by npu_run itself: no copy through Python)."""
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read_header(f)
            data_off = f.tell()
        assert not fortran
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.alloc(name, nbytes)
        self._cmd(f"load {name} {Path(path).resolve()} 0 {data_off} {nbytes}")
        return nbytes

    def download(self, name: str, dtype, shape, offset: int = 0) -> np.ndarray:
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self._n += 1
        f = self.run_dir / f"xfer_{self._n % 4}.bin"
        self._cmd(f"save {name} {f.name} {int(offset)} {n}")
        return np.fromfile(f, dtype=dtype).reshape(shape)

    def zero(self, name: str) -> None:
        self._cmd(f"zero {name}")

    def address(self, name: str) -> int:
        """The buffer's device address: what a kernel parameter carries, so a layer table can name it."""
        return int(self._cmd(f"addr {name}").split()[1])

    def kernel(self, alias: str, elf_name: str, profile: bool = False) -> None:
        self._cmd(f"kernel {alias} {elf_name}" + (" profile" if profile else ""))

    def launch(self, alias: str, *buffers: str) -> tuple[float, float]:
        r = self._cmd(f"launch {alias} " + " ".join(buffers))
        parts = r.split()
        host_ms, kernel_ms = float(parts[2]), float(parts[4])
        if "status" in parts:   # TL_NPU_DUMP_STATUS: the 16 status words of every SHAVE, kept for the caller
            self.last_status = [int(x) for x in parts[parts.index("status") + 1:]]
        self.launches.append((alias, host_ms, kernel_ms))
        return host_ms, kernel_ms

    def profile(self, alias: str, path: Path) -> None:
        self._cmd(f"profile {alias} {path}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
            except OSError:
                pass
            self.proc.wait(timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
