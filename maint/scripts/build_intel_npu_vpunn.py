"""Build the VPUNN cost-model Python binding for the Intel NPU backend.

VPUNN (``libraries.performance.modeling.vpu.nn-cost-model``, Intel innersource,
Apache 2.0) is an external, pinned dependency like ``GENERIC_COMPUTE_ROOT``: it is
never vendored into this tree. Point ``VPUNN_ROOT`` at a checkout (pin: see
``docs/developer_guide/intel_npu_backend_plan.md`` section 0) and run::

    python maint/scripts/build_intel_npu_vpunn.py [--deps-dir <dir with flatbuffers-src,json-src,cxxopts-src>] [--check]

The module is built with MSVC (VPUNN's own ``-Wall -Werror`` set does not compile
under clang-cl) into ``<VPUNN_ROOT>/build-py/lib``, which is where
``tilelang.intel_npu.vpunn`` loads it from unless ``VPUNN_PYTHON_DIR`` says otherwise.
``--deps-dir`` serves machines without GitHub access: VPUNN fetches flatbuffers,
nlohmann/json and cxxopts with FetchContent, and an existing ``_deps`` directory of a
vpux-plugin build has the three source trees.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _vs_dev_cmd() -> Path:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        raise SystemExit(f"vswhere.exe not found at {vswhere}; install Visual Studio 2022 Build Tools")
    inst = subprocess.check_output([str(vswhere), "-latest", "-property", "installationPath"], text=True).strip()
    dev = Path(inst) / "Common7" / "Tools" / "VsDevCmd.bat"
    if not dev.exists():
        raise SystemExit(f"VsDevCmd.bat not found under {inst}")
    return dev


def _pybind11_dir() -> str:
    try:
        import pybind11  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install pybind11 (needed for the VPUNN binding)") from e
    return pybind11.get_cmake_dir()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vpunn-root", default=os.environ.get("VPUNN_ROOT"),
                    help="VPUNN checkout (default: $VPUNN_ROOT)")
    ap.add_argument("--build-dir", default=None, help="default <VPUNN_ROOT>/build-py")
    ap.add_argument("--deps-dir", default=os.environ.get("VPUNN_DEPS_DIR"),
                    help="offline FetchContent sources: <dir>/{flatbuffers,json,cxxopts}-src")
    ap.add_argument("--check", action="store_true", help="import the module and cost one workload")
    args = ap.parse_args()

    if not args.vpunn_root:
        raise SystemExit("set VPUNN_ROOT or pass --vpunn-root")
    root = Path(args.vpunn_root).resolve()
    if not (root / "src" / "python" / "VPUNN.cpp").exists():
        raise SystemExit(f"{root} is not a VPUNN checkout (src/python/VPUNN.cpp missing)")
    build = Path(args.build_dir).resolve() if args.build_dir else root / "build-py"
    build.mkdir(parents=True, exist_ok=True)

    cfg = [
        "cmake", "-S", str(root), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl",
        "-DVPUNN_BUILD_TESTS=OFF", "-DVPUNN_BUILD_APPS=OFF", "-DVPUNN_BUILD_EXAMPLES=OFF",
        "-DENABLE_PYTHON_BINDING=ON",
        f"-Dpybind11_DIR={_pybind11_dir()}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    if args.deps_dir:
        deps = Path(args.deps_dir).resolve()
        for name in ("flatbuffers", "json", "cxxopts"):
            src = deps / f"{name}-src"
            if not src.exists():
                raise SystemExit(f"--deps-dir: {src} missing")
            cfg.append(f"-DFETCHCONTENT_SOURCE_DIR_{name.upper()}={src}")
        cfg.append("-DFETCHCONTENT_FULLY_DISCONNECTED=ON")

    bld = ["cmake", "--build", str(build), "--target", "vpunn_python_bindings", "vpunn_python_create_init_file"]

    def q(parts):
        return " ".join(f'"{p}"' if " " in p else p for p in parts)

    # cl.exe needs the VS environment; a .bat keeps that out of the caller's shell.
    bat = build / "build_vpunn.bat"
    bat.write_text(
        f'@call "{_vs_dev_cmd()}" -no_logo -arch=x64 -host_arch=x64 || exit /b 1\n'
        f"{q(cfg)} || exit /b 1\n{q(bld)} || exit /b 1\n",
        encoding="ascii",
    )
    print(f"[build_vpunn] {root} (git: {_git_describe(root)}) -> {build}")
    rc = subprocess.call(["cmd", "/c", str(bat)])
    if rc != 0:
        return rc

    lib = build / "lib"
    pyd = sorted(lib.glob("_VPUNN*.pyd"))
    if not pyd:
        print(f"[build_vpunn] no _VPUNN*.pyd in {lib}", file=sys.stderr)
        return 1
    print(f"[build_vpunn] built {pyd[0].name}; set VPUNN_PYTHON_DIR={lib} (or leave VPUNN_ROOT={root})")

    if args.check:
        env = dict(os.environ, VPUNN_ROOT=str(root), VPUNN_PYTHON_DIR=str(lib))
        code = ("from tilelang.intel_npu import vpunn as v; c = v.GemmCost(); "
                "print('vpunn', v.version(), 'M64 N64 K512 ->', c.dpu_cycles(64, 64, 512), 'cycles')")
        return subprocess.call([sys.executable, "-c", code], env=env)
    return 0


def _git_describe(root: Path) -> str:
    if shutil.which("git") is None:
        return "git not found"
    try:
        return subprocess.check_output(["git", "-C", str(root), "describe", "--tags", "--always"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return "not a git checkout"


if __name__ == "__main__":
    sys.exit(main())
