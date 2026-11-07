"""M2 device golden: W4A16 projection Y[M, N] = X[M, K] . dequant(Wq[N, K])^T, int4 g128 weights
streamed through two rings (packed bytes + scales), dequantized on the SHAVE (LSU int4 -> int8
convert-load), fp16 on the DPU (kernels.make_w4a16_flow).

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_w4a16_device.py [--m 96] [--n 2048] [--k 2048] [--bn 64]
                                                               [--no-build] [--negative] [--profile]
Unprofiled by default (a timing run, TURBO); the ops basis is 2 M N K; --negative perturbs one nibble so
the comparison must FAIL (the wrong-nibble-order control of npu_dsl's W4A16 harness).
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401
import runner  # noqa: E402
import lowering  # noqa: E402
from kernels import make_w4a16_flow, make_w4a16_gemv, pack_int4  # noqa: E402
from tilelang.intel_npu import hw_facts  # noqa: E402

TILES, SHAVES = 3, 2
G = 128
PERF_CLK_MHZ = 19.2


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def main(argv):
    M, N, K, BN = arg(argv, "--m", 96), arg(argv, "--n", 2048), arg(argv, "--k", 2048), arg(argv, "--bn", 64)
    split = "n" if "--split-n" in argv else "m"
    stages = arg(argv, "--stages", 2)
    gemv = "--gemv" in argv
    bits = arg(argv, "--bits", 4)
    if gemv:
        M = arg(argv, "--m", 1)
        kernel = f"tl_w4a16_gemv_m{M}_n{N}_k{K}_bn{BN}_b{bits}" + (f"_s{stages}" if stages != 2 else "")
        low = lowering.lower(make_w4a16_gemv(N, K, BN=BN, G=G, tiles=TILES, shaves=SHAVES, num_stages=stages, M=M, bits=bits))
    else:
        kernel = f"tl_w4a16_m{M}_n{N}_k{K}_bn{BN}_split{split}" + (f"_s{stages}" if stages != 2 else "")
        low = lowering.lower(make_w4a16_flow(M, N, K, BN=BN, G=G, tiles=TILES, shaves=SHAVES, split=split, num_stages=stages))
    for ln in low.source.splitlines():
        if "// intel_npu:" in ln:
            print(ln.strip()[:220])
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, kernel)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=kernel, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, kernel)
        print(f"build {time.time() - t0:.0f} s")
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    if bits == 8:
        w_int = rng.integers(-128, 128, size=(N, K)).astype(np.int8)
        scales = rng.uniform(0.001, 0.004, size=(N, K // G)).astype(np.float16)
        wq = w_int.copy()
    else:
        w_int = rng.integers(-8, 8, size=(N, K)).astype(np.int8)
        scales = rng.uniform(0.01, 0.05, size=(N, K // G)).astype(np.float16)
        wq = pack_int4(w_int)
    wq_dev = wq.copy()
    negative = "--negative" in argv
    if negative:
        wq_dev[3, :64] ^= (np.int8(0x77) if bits == 8 else np.uint8(0x77))   # one whole group of row 3 (a single nibble can meet a near-zero activation)
    y = np.zeros((M, N), np.float16)
    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{kernel}_trace.json"
    arrays = lowering.order_arrays(low, {"X": ("in", x), "Wq": ("in", wq_dev), "S": ("in", scales), "Y": ("out", y)})
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_w4a16", profile_json=profile, repeat=10, warmup=10, timeout_s=120)   # a launch set steps 0.33 -> 0.21 ms after ~7 launches (a clock state?)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    w = (w_int.astype(np.float32) * np.repeat(scales.astype(np.float32), G, axis=1)).astype(np.float16).astype(np.float32)
    ref = x.astype(np.float32) @ w.T
    err = float(np.abs(y.astype(np.float32) - ref).max())
    rel = err / float(np.abs(ref).max())
    verdict = "PASS" if rel < (4e-3 if gemv else 2e-3) else "FAIL"   # the merged-FC path rounds each group partial to fp16
    kind = runner.condition(res, profile is not None)
    ops = 2.0 * M * N * K
    tops = ops / (res.median_kernel_ms * 1e-3) / 1e12
    peak = hw_facts.fp16_peak_tops(hw_facts.fact("dpu_clk_turbo_ghz"), TILES)
    idu_us = [float(res.status[t * SHAVES:(t + 1) * SHAVES, 5].astype(np.int64).sum()) / PERF_CLK_MHZ for t in range(TILES)]
    print(f"device: kernel-only {res.median_kernel_ms:.3f} ms (host wall {res.median_ms:.3f} ms) [{kind}]; "
          f"max |y - ref| = {err:.4f} (rel {rel:.2e}) -> {verdict}; kernel sha256 {db.source_sha256[:12]}")
    if res.kernel_times_ms:
        print("  kernel-only per run ms: " + " ".join(f"{t:.3f}" for t in res.kernel_times_ms))
    print(f"ops {ops / 1e9:.2f} GFLOP; {tops:.2f} TOPS kernel-only = {100 * tops / peak:.1f}% of {peak:.2f} (dense fp16 peak); "
          f"DPU IDU busy per tile {[f'{100 * u / (res.median_kernel_ms * 1e3):.0f}%' for u in idu_us]}; "
          f"weights {wq.nbytes / 1e6:.1f} MB read once -> {wq.nbytes / (res.median_kernel_ms * 1e-3) / 1e9:.1f} GB/s")
    names = ["live_wait", "order_wait", "submit", "slot_wait", "free_wait", "fill_issue"]
    phases = res.status[:, 10:16].astype(np.int64) / PERF_CLK_MHZ
    for w_ in (0, 1, 2):
        print(f"  tile {w_ // SHAVES} SHAVE {w_ % SHAVES} ring phases us: " + ", ".join(f"{n} {phases[w_, i]:.0f}" for i, n in enumerate(names)))
    if profile:
        print(f"trace: {profile}")
    if negative:
        print("negative control (perturbed nibble): " + ("behaved (FAIL reported)" if verdict == "FAIL" else "BROKEN: PASS reported"))
        return 0 if verdict == "FAIL" else 1
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
