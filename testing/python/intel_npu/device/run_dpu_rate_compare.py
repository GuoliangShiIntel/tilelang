"""DPU rate in the npu_dsl method, side by side with its record (plan section 7 question 5).

Builds probes/dpu_rate_probe.cpp for a sweep of (shape, stencil, spatial, data) and reports each arm's
median idu_workload_duration (19.2 MHz ticks) as MAC/ns and as % of the npu_dsl basis
2048 MAC/cycle x 2.0454 GHz (its `rate_pct_of_2048xDPU_Hz`), next to the npu_dsl numbers for the same
shape from evidence/e7_21/stencil_table_validated.txt (fp16w: 64x64 100.1 with 8_8 ODU left, 32x192
77.7 with 4_16, 16x192 46.0 ...). Same instrument, same box, same day.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_dpu_rate_compare.py [--no-build] [--quick]
"""

import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401
import runner  # noqa: E402
from tilelang.intel_npu import hw_facts  # noqa: E402

TILES, SHAVES = 3, 2
ITERS = 64
PERF_CLK_MHZ = 19.2
NPU_DSL_BASIS_MAC_PER_NS = hw_facts.fact("dpu_fp16_macs_per_clk") * hw_facts.fact("dpu_clk_turbo_ghz")  # the prior art's 100%: 2048 x 2.0454
# (M, N, K): npu_dsl's fp16 shapes at K=1600 (its stencil table), plus our calibration's 64x64x2048.
# Shapes whose operands fit the ~1.44 MiB of free CMX over two SHAVE slots (64x192 and 32x192 at K=1600 and
# 32x256 at K=2048 do not: status 2 CMX_EXHAUSTED); the headline arm is 64x64x1600 8_8 x=8.
SHAPES = [(64, 64, 1600), (16, 192, 1600), (64, 64, 2048)]
# (nthw_ntk, odu_nthw): 99 = keep init's; "left" = ODU left at init's 3 (the npu_dsl table's best arms).
STENCILS = [("init 16_4", 99, 99), ("8_8 odu left(3)", 0, 99), ("8_8 odu 2", 0, 2), ("4_16 odu left(3)", 1, 99), ("4_16 odu 1", 1, 1)]
SPATIALS = [0, 8, 16]
DATA = [("zeros", 1), ("random", 0)]
# npu_dsl best VALID fp16w arm per (M, N) at K=1600 (evidence/e7_21/stencil_table_validated.txt).
NPU_DSL_BEST = {(64, 64): ("8_8 odu left(3)", 100.1), (64, 192): ("8_8 odu left(3)", 100.1),
                (32, 192): ("4_16 odu left(3)", 77.7), (16, 192): ("4_16 odu left(3)", 46.0)}


def kernel_name(M, N, K, x, nthw, odu, zeros):
    return f"tl_dpurate_{M}x{N}x{K}_x{x}_s{nthw}_{odu}_z{zeros}_k"


def main(argv):
    quick = "--quick" in argv
    shapes = SHAPES[:1] if quick else SHAPES
    stencils = STENCILS[:3] if quick else STENCILS
    spatials = SPATIALS[:2] if quick else SPATIALS
    template = (HERE / "probes" / "dpu_rate_probe.cpp").read_text(encoding="utf-8")
    jobs = []
    for (M, N, K), (sname, nthw, odu), x, (dname, zeros) in itertools.product(shapes, stencils, spatials, DATA):
        if x and M % x:
            continue
        name = kernel_name(M, N, K, x, nthw, odu, zeros)
        defines = (f"#define TL_M {M}\n#define TL_N {N}\n#define TL_K {K}\n#define TL_X {x}\n#define TL_NTHW {nthw}\n"
                   f"#define TL_ODU {odu}\n#define TL_ZEROS {zeros}\n#define TL_ITERS {ITERS}\n")
        source = template.replace("#ifndef TL_M", defines + "#ifndef TL_M", 1)
        jobs.append(((M, N, K), sname, nthw, odu, x, dname, name, source))
    for *_, name, source in jobs:
        runner.write_kernel(source, name)
    if "--no-build" not in argv:
        last = runner.build(jobs[-1][-1], jobs[-1][-2])
        exe = last.exe
    else:
        exe = HERE / "build" / "Release" / "npu_run.exe"
    print(f"{len(jobs)} arms; median of {ITERS} workloads each; % of {NPU_DSL_BASIS_MAC_PER_NS:.0f} MAC/ns (2048 x 2.0454 GHz, the npu_dsl basis)")
    rows = []
    for shape, sname, nthw, odu, x, dname, name, source in jobs:
        M, N, K = shape
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=exe, kernel=name, source_sha256=hashlib.sha256(source.encode()).hexdigest())
        durs = np.zeros((ITERS,), np.uint32)
        res = runner.run(db, [("out", durs)], TILES, SHAVES, HERE / "run_dpurate", repeat=3, warmup=2, timeout_s=120)
        runner.check_status(res.status)
        med = float(np.median(durs.astype(np.int64)))
        if med <= 0:
            rows.append((shape, sname, x, dname, None, None))
            print(f"{M}x{N}x{K} {sname:16s} x={x:<2d} {dname:6s}: INVALID (median duration 0: the setter refused or nothing ran)")
            continue
        us = med / PERF_CLK_MHZ
        mac_per_ns = M * N * K / us * 1e-3
        pct = 100.0 * mac_per_ns / NPU_DSL_BASIS_MAC_PER_NS
        rows.append((shape, sname, x, dname, mac_per_ns, pct))
        print(f"{M}x{N}x{K} {sname:16s} x={x:<2d} {dname:6s}: {med:6.1f} ticks = {us:6.2f} us, {mac_per_ns:6.0f} MAC/ns = {pct:5.1f}%")
    print("\n| shape | our best arm (random data) | ours % | our best (zeros) % | npu_dsl best fp16w arm | npu_dsl % |")
    print("|---|---|---|---|---|---|")
    for shape in shapes:
        M, N, K = shape
        ours = [r for r in rows if r[0] == shape and r[4] is not None]
        if not ours:
            continue
        best_r = max((r for r in ours if r[3] == "random"), key=lambda r: r[4])
        best_z = max((r for r in ours if r[3] == "zeros"), key=lambda r: r[4])
        ref = NPU_DSL_BEST.get((M, N)) if K == 1600 else None
        print(f"| {M}x{N}x{K} | {best_r[1]} x={best_r[2]} | {best_r[5]:.1f} | {best_z[5]:.1f} ({best_z[1]} x={best_z[2]}) | "
              f"{ref[0] if ref else '-'} | {ref[1] if ref else '-'} |")
    print(f"removed {runner.remove_kernels(j[-2] for j in jobs)} sweep kernels from generated/ (every device build compiles the glob)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
