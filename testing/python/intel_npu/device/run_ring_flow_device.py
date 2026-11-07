"""Phase 3b device golden: the multi-tile ping-pong flow, compiled from the DSL (kernels.make_ring_flow)
or, with --hand, the hand-written reference (probes/ring_flow_hand.cpp) it was built from.

Usage (GENERIC_COMPUTE_ROOT and MV_TOOLS_PATH set; VPUNN_ROOT for planned workloads):
    python testing/python/intel_npu/device/run_ring_flow_device.py [--hand] [--no-build] [--profile] [--negative]

Y[2048, 768] = X[2048, 2048] . W[768, 2048]^T fp16 on 3 tiles x 2 SHAVEs: W split by tile, X
streamed through a 2-slot multicast ring with READY/FREE hardware barriers. Unprofiled by default;
reports host wall (incl. submit), the DPU IDU time per tile from the HWP status words and the
DPU busy % it implies, and an ops/time figure. --profile captures a trace (observability run).
--negative perturbs W so the comparison must FAIL.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401  (puts tvm on the path for runner)
import tvm  # noqa: E402
import runner  # noqa: E402
import lowering  # noqa: E402
from tilelang.intel_npu import hw_facts  # noqa: E402
from kernels import make_ring_flow, make_ring_flow_swapped, make_ring_flow_two_phase  # noqa: E402

M, K, N = 2048, 2048, 768
BM, BN = 32, 256
TILES, SHAVES = 3, 2
PERF_CLK_MHZ = 19.2


def main(argv):
    global M, N, K, BM, BN
    hand = "--hand" in argv
    swapped = "--swapped" in argv
    two_phase = "--two-phase" in argv
    if "--bm" in argv:
        BM = int(argv[argv.index("--bm") + 1])
    if "--bn" in argv:
        BN = int(argv[argv.index("--bn") + 1])
        N = BN * TILES
    if "--k" in argv:
        K = int(argv[argv.index("--k") + 1])
    if hand:
        kernel = "tl_ring_flow_hand"
        source = (HERE / "probes" / "ring_flow_hand.cpp").read_text(encoding="utf-8")
        names = ["X", "W", "Y"]
    elif swapped:
        # roles swapped: X rows resident per tile, W streamed (kernels.make_ring_flow_swapped)
        kernel = "tl_ring_flow_swapped"
        M, N, BM, BN = 768, 2048, 256, 32
        low = lowering.lower(make_ring_flow_swapped(M, N, K, BM=BM, BN=BN, tiles=TILES, shaves=SHAVES))
        source, names = low.source, [{"Xa": "X", "Wb": "W", "Out": "Y"}[n] for n in low.param_names]
        for ln in source.splitlines():
            if "// intel_npu:" in ln:
                print(ln.strip())
    elif two_phase:
        # two sequential rings on the same barrier ids (AssignBarriers reuse); both outputs checked.
        # --no-reuse gives the second ring fresh ids: the A/B that isolates the reuse.
        reuse = "--no-reuse" not in argv
        kernel = f"tl_ring_flow_two_phase_bm{BM}_bn{BN}_k{K}" + ("" if reuse else "_noreuse")
        with tvm.transform.PassContext(config={"tl.intel_npu.barrier_reuse": reuse}):
            low = lowering.lower(make_ring_flow_two_phase(M, N, K, BM=BM, BN=BN, tiles=TILES, shaves=SHAVES))
        source, names = low.source, low.param_names
        for ln in source.splitlines():
            if "// intel_npu:" in ln:
                print(ln.strip())
        assert ("2 rings, 1 live at once, 8 ids by reuse" in source) == reuse
    else:
        kernel = f"tl_ring_flow_bm{BM}_bn{BN}_k{K}"
        low = lowering.lower(make_ring_flow(M, N, K, BM=BM, BN=BN, tiles=TILES, shaves=SHAVES))
        source, names = low.source, low.param_names
        for ln in source.splitlines():
            if "// intel_npu:" in ln:
                print(ln.strip())
    if "--no-build" in argv:
        _, sha = runner.write_kernel(source, kernel)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=kernel, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(source, kernel)
        print(f"build {time.time() - t0:.0f} s")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((M, K)).astype(np.float16)
    w = rng.standard_normal((N, K)).astype(np.float16)
    y = np.zeros((M, N), np.float16)
    w_dev = w.copy()
    negative = "--negative" in argv
    if negative:
        w_dev[5, 7] += np.float16(8.0)
    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{kernel}_{M}x{N}x{K}_trace.json"
    by_name = {"X": ("in", x), "W": ("in", w_dev), "Y": ("out", y)}
    if two_phase:
        x2 = rng.standard_normal((M, K)).astype(np.float16)
        y2 = np.zeros((M, N), np.float16)
        by_name = {"X1": ("in", x), "X2": ("in", x2), "W": ("in", w_dev), "Y1": ("out", y), "Y2": ("out", y2)}
    res = runner.run(db, [by_name[n] for n in names], TILES, SHAVES, HERE / "run_ring_flow",
                     profile_json=profile, repeat=10, warmup=3, timeout_s=120)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    ref = x.astype(np.float32) @ w.astype(np.float32).T
    err = float(np.abs(y.astype(np.float32) - ref).max())
    rel = err / float(np.abs(ref).max())
    if two_phase:
        ref2 = x2.astype(np.float32) @ w.astype(np.float32).T
        err2 = float(np.abs(y2.astype(np.float32) - ref2).max())
        rel2 = err2 / float(np.abs(ref2).max())
        print(f"phase 1 rel {rel:.2e}, phase 2 (reused ids) rel {rel2:.2e}")
        err, rel = max(err, err2), max(rel, rel2)
    verdict = "PASS" if rel < 2e-3 else "FAIL"
    kind = runner.condition(res, profile is not None)
    ops = 2.0 * M * N * K * (2 if two_phase else 1)
    idu_us_per_tile = [float(res.status[t * SHAVES:(t + 1) * SHAVES, 5].astype(np.int64).sum()) / PERF_CLK_MHZ for t in range(TILES)]
    wl_per_tile = [int(res.status[t * SHAVES:(t + 1) * SHAVES, 4].astype(np.int64).sum()) for t in range(TILES)]
    kus = res.median_kernel_ms * 1e3
    print(f"device: median kernel-only {res.median_kernel_ms:.3f} ms (NCE_TIME stamps), host wall {res.median_ms:.3f} ms incl. submit "
          f"[{kind}]; max |y - ref| = {err:.4f} (rel {rel:.2e}) -> {verdict}; kernel sha256 {db.source_sha256[:12]}")
    print(f"DPU workloads per tile {wl_per_tile}; DPU IDU busy per tile {[f'{u:.0f}' for u in idu_us_per_tile]} us "
          f"of {kus:.0f} us kernel -> busy {[f'{100 * u / kus:.0f}%' for u in idu_us_per_tile]}")
    tops = ops / (res.median_kernel_ms * 1e-3) / 1e12
    peak = hw_facts.fp16_peak_tops(hw_facts.fact("dpu_clk_turbo_ghz"), TILES)   # 2048 fp16 MAC/clk/DPU (dpu.md 3.4) x 2.0454 GHz x 3
    print(f"ops {ops / 1e9:.2f} GFLOP; {tops:.2f} TOPS kernel-only = {100 * tops / peak:.1f}% of {peak:.2f} TOPS "
          f"(dense fp16 peak, {TILES} tiles at {hw_facts.fact('dpu_clk_turbo_ghz')} GHz){' [PROFILED: not a timing claim]' if profile else ''}")
    if profile:
        print(f"trace: {profile}")
    if negative:
        print("negative control (perturbed W): " + ("behaved (FAIL reported)" if verdict == "FAIL" else "BROKEN: PASS reported"))
        return 0 if verdict == "FAIL" else 1
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
