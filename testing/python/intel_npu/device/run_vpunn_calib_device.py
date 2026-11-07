"""VPUNN calibration on the PTL/NPU5 DPU (plan section 3.2, Phase 3b Step 0).

For each tile shape and each DPU workload configuration (spatial split x, MPE
cuboid), builds `kernels.make_gemm_repeat` at two repeat counts and runs both
UNPROFILED. Two per-workload numbers come out:

* DPU time from the DPU's own hardware profile (status words 4-6: workload
  count, sum of IDU durations, sum of ODU durations). The duration counters tick
  at the 19.2 MHz perf clock on this box (npu_dsl HANDOFF SS690.2: the only
  reading consistent with the event durations; the DPU core clock itself is not
  observable and varies with DVFS), so they are reported in microseconds and
  compared with VPUNN's cycles at the 1950 MHz VPUX assumes for NPU_5_0;
* host wall time (t_R - t_1) / (shaves * (R - 1)), which also contains the
  SHAVE's synchronous submit/wait round trip (the two SHAVEs queue on one DPU).

The ranking of configurations is compared with VPUNN's. Golden against numpy on
every run; a configuration whose output is wrong is recorded as such (the ODU
stencil pairing is a known placement hazard) and excluded from the ranking.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_vpunn_calib_device.py [--quick] [--no-build]

Writes evidence/vpunn_calibration.json and prints a markdown table.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402
import tvm  # noqa: E402
import runner  # noqa: E402
from kernels import make_gemm_repeat  # noqa: E402
from tilelang.intel_npu import vpunn  # noqa: E402

TILES, SHAVES = 1, 2
R_LOW, R_HIGH = 1, 401
PERF_CLK_MHZ = 19.2
# (BM, BN, K): the DPU's 8192 output points cap BM x BN.
SHAPES = [(64, 64, 512), (64, 128, 512), (128, 64, 512), (64, 64, 2048)]
# (spatial_x, mpe): 0/-1 = the runtime's own (x = 4, CUBOID_16x16).
CONFIGS = [(0, -1)] + [(x, m) for x in (4, 8, 16) for m in (0, 1, 2)]
MPE_NAME = {-1: "default", 0: "CUBOID_4x16", 1: "CUBOID_8x16", 2: "CUBOID_16x16"}


def kernel_name(bm, bn, k, x, mpe, rep):
    return f"tl_dpucal_{bm}x{bn}x{k}_x{x}_m{mpe + 1}_r{rep}_k"


def lower(bm, bn, k, x, mpe, rep):
    cfg = {"tl.intel_npu.dpu_spatial_x": x, "tl.intel_npu.dpu_mpe": mpe}
    with tvm.transform.PassContext(config=cfg):
        return tilelang.lower(make_gemm_repeat(bm, bn, k, rep, TILES, SHAVES), target="intel_npu").kernel_source


def main(argv):
    quick = "--quick" in argv
    shapes = SHAPES[:1] if quick else SHAPES
    # quick: the default, then the explicit spelling of the default split by knob, then one real variant
    configs = [(0, -1), (4, -1), (0, 2), (4, 2), (8, 1)] if quick else CONFIGS
    cost = vpunn.GemmCost()
    print(f"vpunn {vpunn.version()} from {vpunn.root()}")

    jobs = []  # (shape, x, mpe, rep, name)
    for (bm, bn, k) in shapes:
        for (x, mpe) in configs:
            for rep in (R_LOW, R_HIGH):
                jobs.append(((bm, bn, k), x, mpe, rep, kernel_name(bm, bn, k, x, mpe, rep)))

    # One device build for every kernel: the CMake kernel list is a glob over generated/.
    sources = {}
    for shape, x, mpe, rep, name in jobs:
        sources[name] = lower(*shape, x, mpe, rep)
        runner.write_kernel(sources[name], name)
    builds = {}
    if "--no-build" not in argv:
        t0 = time.time()
        last = runner.build(sources[jobs[-1][4]], jobs[-1][4])
        print(f"built {len(jobs)} kernels in {time.time() - t0:.0f} s")
        exe = last.exe
    else:
        exe = HERE / "build" / "Release" / "npu_run.exe"
    for shape, x, mpe, rep, name in jobs:
        import hashlib
        builds[name] = runner.DeviceBuild(build_dir=HERE / "build", exe=exe, kernel=name,
                                          source_sha256=hashlib.sha256(sources[name].encode()).hexdigest())

    rng = np.random.default_rng(0)
    rows = []
    for (bm, bn, k) in shapes:
        M, N = bm * TILES * SHAVES, bn
        a = rng.standard_normal((M, k)).astype(np.float16)
        b = rng.standard_normal((N, k)).astype(np.float16)
        ref = a.astype(np.float32) @ b.astype(np.float32).T
        for (x, mpe) in configs:
            t, hw = {}, {}
            golden = True
            for rep in (R_LOW, R_HIGH):
                name = kernel_name(bm, bn, k, x, mpe, rep)
                c = np.zeros((M, N), np.float16)
                res = runner.run(builds[name], [("in", a), ("in", b), ("out", c)], TILES, SHAVES,
                                 HERE / "run_dpucal", repeat=15, warmup=20)
                runner.check_status(res.status)
                rel = float(np.abs(c.astype(np.float32) - ref).max() / np.abs(ref).max())
                golden = golden and rel < 2e-3
                t[rep] = res.median_ms
                # status words 4..7 summed over the tile's SHAVEs: workloads, IDU cycles, ODU cycles, ticks
                hw[rep] = res.status[:, 4:8].astype(np.int64).sum(axis=0)
            per_wl_us = (t[R_HIGH] - t[R_LOW]) * 1e3 / (SHAVES * (R_HIGH - R_LOW))
            n_wl = int(hw[R_HIGH][0])
            idu_us = float(hw[R_HIGH][1]) / n_wl / PERF_CLK_MHZ if n_wl else float("nan")
            odu_us = float(hw[R_HIGH][2]) / n_wl / PERF_CLK_MHZ if n_wl else float("nan")
            # VPUNN sees the configuration the runtime actually programs.
            vx = x if x else vpunn.default_spatial_x(bm)
            vmpe = MPE_NAME[mpe] if mpe >= 0 else "CUBOID_16x16"
            cyc = cost.dpu_cycles(bm, bn, k, spatial_x=vx, mpe=vmpe)
            vpunn_us = cost.cycles_to_us(cyc)
            macs = bm * bn * k
            rows.append(dict(shape=[bm, bn, k], spatial_x=x, mpe=mpe, mpe_name=MPE_NAME[mpe], golden=golden,
                             t_low_ms=t[R_LOW], t_high_ms=t[R_HIGH], host_us_per_wl=per_wl_us,
                             dpu_workloads=n_wl, idu_us=idu_us, odu_us=odu_us,
                             vpunn_cycles=cyc, vpunn_us_1950=vpunn_us, ratio_idu=idu_us / vpunn_us,
                             idu_mac_per_ns=macs / idu_us * 1e-3 if idu_us else float("nan"),
                             kernel_sha=builds[kernel_name(bm, bn, k, x, mpe, R_HIGH)].source_sha256[:12]))
            r = rows[-1]
            print(f"{bm}x{bn}x{k} x={x} {MPE_NAME[mpe]:13s}: {'PASS' if golden else 'GOLDEN FAIL'}; "
                  f"IDU {idu_us:6.3f} us (ODU {odu_us:6.3f}), vpunn {vpunn_us:6.3f} us ({cyc} cyc), "
                  f"ratio {r['ratio_idu']:.2f}, {r['idu_mac_per_ns']:.0f} MAC/ns; host {per_wl_us:.3f} us/wl")

    # Ranking agreement per shape (golden-correct rows only): concordant pairs, and the regret of VPUNN's pick.
    summary = []
    for (bm, bn, k) in shapes:
        sub = [r for r in rows if r["shape"] == [bm, bn, k] and r["golden"]]
        bad = [f"x={r['spatial_x']} {r['mpe_name']}" for r in rows if r["shape"] == [bm, bn, k] and not r["golden"]]
        pairs = [(i, j) for i in range(len(sub)) for j in range(i + 1, len(sub))]
        conc = sum(1 for i, j in pairs if (sub[i]["idu_us"] - sub[j]["idu_us"]) *
                   (sub[i]["vpunn_cycles"] - sub[j]["vpunn_cycles"]) > 0)
        best_m = min(sub, key=lambda r: r["idu_us"])
        best_v = min(sub, key=lambda r: r["vpunn_cycles"])
        summary.append(dict(shape=[bm, bn, k], concordant_pairs=f"{conc}/{len(pairs)}", golden_fail=bad,
                            best_measured=f"x={best_m['spatial_x']} {best_m['mpe_name']} {best_m['idu_us']:.3f} us",
                            best_vpunn=f"x={best_v['spatial_x']} {best_v['mpe_name']} {best_v['idu_us']:.3f} us measured",
                            regret_pct=100.0 * (best_v["idu_us"] / best_m["idu_us"] - 1.0)))
        print(f"{bm}x{bn}x{k}: ranking concordance {conc}/{len(pairs)}; best measured {summary[-1]['best_measured']}; "
              f"VPUNN's pick measures {best_v['idu_us']:.3f} us (regret {summary[-1]['regret_pct']:.1f}%)"
              + (f"; golden FAIL: {', '.join(bad)}" if bad else ""))

    ratios = np.array([r["ratio_idu"] for r in rows if r["golden"]])
    print(f"IDU us / VPUNN us@1950: median {np.median(ratios):.3f}, min {ratios.min():.3f}, max {ratios.max():.3f}")
    out = HERE / "evidence" / "vpunn_calibration.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(vpunn=vpunn.version(), tiles=TILES, shaves=SHAVES, repeats=[R_LOW, R_HIGH],
                                   timing="UNPROFILED; DPU IDU/ODU durations from the HWP status words of the R_HIGH "
                                          "run, 19.2 MHz ticks -> us; host us per workload = delta / (shaves * delta_repeat)",
                                   rows=rows, summary=summary, ratio_idu_median=float(np.median(ratios))), indent=1))
    print(f"wrote {out}")
    print("\n| shape | x | MPE | golden | IDU us | VPUNN us @1950 | ratio | MAC/ns | host us/wl |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {'x'.join(map(str, r['shape']))} | {r['spatial_x']} | {r['mpe_name']} | {'PASS' if r['golden'] else 'FAIL'} | "
              f"{r['idu_us']:.3f} | {r['vpunn_us_1950']:.3f} | {r['ratio_idu']:.2f} | {r['idu_mac_per_ns']:.0f} | "
              f"{r['host_us_per_wl']:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
