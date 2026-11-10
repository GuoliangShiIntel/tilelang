"""M5 head device golden: final norm + lm_head int8 g128 GEMV + greedy argmax in one launch (kernels.make_head,
Qwen3.6's shape by default: H 2048, V 248320 -> 248832 padded; 509 MB of int8 weights + 8 MB of scales).

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_head_device.py [--v 248320] [--bn 128] [--r 64] [--stages 4]
                                                              [--no-build] [--negative] [--profile] [--no-pipeline]
Unprofiled by default (a timing run, TURBO). --negative perturbs the winning row's weights so the comparison must
FAIL. --v 24576 is a quick arm (the same kernel at a 10x smaller vocabulary).
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
from kernels import head_reference, head_sizes, make_head, pack_head  # noqa: E402
from test_intel_npu_m5_head import head_arrays, head_problem  # noqa: E402

TILES, SHAVES = 3, 2
G = 128
H = 2048
PERF_CLK_MHZ = 19.2


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def main(argv):
    V, BN, R, stages = arg(argv, "--v", 248320), arg(argv, "--bn", 128), arg(argv, "--r", 64), arg(argv, "--stages", 4)
    pipelined = "--no-pipeline" not in argv
    cfg = dict(H=H, V=V, G=G, tiles=TILES, shaves=SHAVES, BN=BN, R=R)
    sz = head_sizes(**cfg)
    kernel = (f"tl_head_h{H}_v{V}" + (f"_bn{BN}" if BN != 128 else "") + (f"_r{R}" if R != 64 else "")
              + (f"_q{stages}" if stages != 4 else "") + ("" if pipelined else "_serial"))
    low = lowering.lower(make_head(num_stages=stages, pipelined=pipelined, **cfg))
    for ln in low.source.splitlines():
        if "// intel_npu:" in ln:
            print(ln.strip()[:260])
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, kernel)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=kernel, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, kernel)
        print(f"build {time.time() - t0:.0f} s")
    t0 = time.time()
    p = head_problem(0, **cfg)
    ref, tok_ref = head_reference(p["h"], p["norm_w"], p["wlm_int"], p["wlm_scales"], G=G)
    if "--negative" in argv:
        p["wlm_int"][tok_ref, :G] = 127
    packed = pack_head(p["wlm_int"], p["wlm_scales"], tiles=TILES, shaves=SHAVES, BN=BN)
    arrays = head_arrays(p, packed, **cfg)
    print(f"problem + oracle + packing {time.time() - t0:.0f} s; V {V} -> {sz['V_pad']} padded, {sz['C']} chunks of {BN} per tile")
    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{kernel}_trace.json"
    io = {k: ("out" if k in ("Logits", "Part", "Tok") else "in") for k in arrays}
    ordered = lowering.order_arrays(low, {k: (io[k], v) for k, v in arrays.items()})
    res = runner.run(db, ordered, TILES, SHAVES, HERE / "run_head", profile_json=profile, repeat=10, warmup=10, timeout_s=600)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    logits = arrays["Logits"].reshape(-1)[:V].astype(np.float32)
    scale = float(np.abs(ref).max())
    err = float(np.abs(logits - ref).max())
    rel = err / scale
    tok = int(arrays["Tok"][0, 0])
    tok_own = int(np.argmax(logits))
    top2 = np.sort(ref)[-2:]
    pad_ok = bool(np.all(arrays["Logits"].reshape(-1)[V:] == np.float16(-65504)))
    verdict = "PASS" if (rel < 2e-2 and tok == tok_own and pad_ok and (tok == tok_ref or top2[1] - top2[0] <= 2 * err)) else "FAIL"
    kind = runner.condition(res, profile is not None)
    weight_bytes = packed["WlmQ"].nbytes + packed["WlmS"].nbytes
    idu_us = [float(res.status[t * SHAVES:(t + 1) * SHAVES, 5].astype(np.int64).sum()) / PERF_CLK_MHZ for t in range(TILES)]
    print(f"device: kernel-only {res.median_kernel_ms:.3f} ms (host wall {res.median_ms:.3f} ms) [{kind}]; "
          f"max |logits - ref| = {err:.4f} over {scale:.3f} (rel {rel:.2e}); token {tok} (own argmax {tok_own}, oracle {tok_ref}, "
          f"oracle top-2 margin {top2[1] - top2[0]:.4f}); pad masked {pad_ok} -> {verdict}; kernel sha256 {db.source_sha256[:12]}")
    print(f"  partials (max, index) per worker: {[(round(float(a), 3), int(b)) for a, b in arrays['Part'][:, :2]]}")
    if res.kernel_times_ms:
        print("  kernel-only per run ms: " + " ".join(f"{t:.3f}" for t in res.kernel_times_ms))
    print(f"weights {weight_bytes / 1e6:.1f} MB -> {weight_bytes / (res.median_kernel_ms * 1e-3) / 1e9:.1f} GB/s kernel-only; "
          f"DPU IDU busy per tile {[f'{100 * u / (res.median_kernel_ms * 1e3):.0f}%' for u in idu_us]}")
    names = ["acquire/live_wait", "order_wait", "submit", "dpu_wait", "free_wait", "fill_issue"]
    phases = res.status[:, 10:16].astype(np.int64) / PERF_CLK_MHZ
    for w_ in range(TILES * SHAVES):
        print(f"  tile {w_ // SHAVES} SHAVE {w_ % SHAVES} ring phases us: " + ", ".join(f"{n} {phases[w_, i]:.0f}" for i, n in enumerate(names)))
    if profile:
        print(f"trace: {profile}")
    if "--negative" in argv:
        print("negative control (perturbed winning row): " + ("behaved (FAIL reported)" if verdict == "FAIL" else "BROKEN: PASS reported"))
        return 0 if verdict == "FAIL" else 1
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
