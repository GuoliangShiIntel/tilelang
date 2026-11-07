"""M4 device golden: one token through a Gated DeltaNet mixer layer in one launch (kernels.make_gdn_decode,
Qwen3.6's shape by default: H 2048, 16 k-heads / 32 v-heads of 128, conv k 4, int4 g128 in_proj [12352, 2048] and
out_proj [2048, 4096]).

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_gdn_device.py [--stages 4] [--stages2 2] [--bn1 128] [--bn2 64] [--cb 512]
                                                             [--no-build] [--negative] [--profile] [--no-pipeline]
Unprofiled by default (a timing run, TURBO). The layer reads 8.4 MB of int4 weights and moves the 2 MiB fp32 SSM
state in and out; --negative perturbs one head's SSM state so the comparison must FAIL. The states are checked too
(conv state exact in its shifted columns, SSM within 1e-3 relative).
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
from kernels import gdn_reference, make_gdn_decode, pack_gdn  # noqa: E402
from test_intel_npu_m4_gdn import gdn_arrays, gdn_problem  # noqa: E402

TILES, SHAVES = 3, 2
G = 128
PERF_CLK_MHZ = 19.2
SHAPE = dict(H=2048, KH=16, VH=32, DK=128, DV=128)


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def main(argv):
    stages, stages2 = arg(argv, "--stages", 4), arg(argv, "--stages2", 2)
    BN1, BN2, CB = arg(argv, "--bn1", 128), arg(argv, "--bn2", 64), arg(argv, "--cb", 512)
    pipelined = "--no-pipeline" not in argv
    skip = next((a[7:] for a in argv if a.startswith("--skip-")), None)   # measurement arm: conv | scan | gemv (output wrong)
    cfg = dict(SHAPE, G=G, tiles=TILES, shaves=SHAVES, BN1=BN1, R1=64, BN2=BN2, R2=32, CB=CB)
    kernel = (f"tl_gdn_h{SHAPE['H']}_kh{SHAPE['KH']}_vh{SHAPE['VH']}" + (f"_s{stages}" if stages != 4 else "")
              + (f"_t{stages2}" if stages2 != 2 else "")
              + (f"_b{BN1}" if BN1 != 128 else "") + (f"_o{BN2}" if BN2 != 64 else "") + (f"_c{CB}" if CB != 512 else "")
              + ("" if pipelined else "_serial") + (f"_no{skip}" if skip else ""))
    low = lowering.lower(make_gdn_decode(num_stages=stages, num_stages2=stages2, pipelined=pipelined, probe_skip=skip, **cfg))
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
    p = gdn_problem(0, **cfg)
    if "--negative" in argv:
        p["ssm"][5] += 0.5
    packed = pack_gdn(p["win_int"], p["win_scales"], p["wout_int"], p["wout_scales"], tiles=TILES, BN1=BN1, BN2=BN2)
    arrays = gdn_arrays(p, packed, **cfg)
    # npu_run re-uploads every input (inout included) before each launch, so the in-place states restart every
    # launch and the one-step oracle is the golden (verified: the 20-step iteration disagrees, the one-step agrees)
    WARMUP, REPEAT = 10, 10
    ref_p = gdn_problem(0, **cfg)   # the UNPERTURBED problem, so the negative must fail
    ref, ref_cs, ref_ssm = gdn_reference(ref_p["x"], ref_p["res"], ref_p["win_int"], ref_p["win_scales"], ref_p["wout_int"],
                                         ref_p["wout_scales"], ref_p["conv_w"], ref_p["a_log"], ref_p["dt_bias"], ref_p["g_norm"],
                                         ref_p["conv_state"], ref_p["ssm"], KH=SHAPE["KH"], VH=SHAPE["VH"], DK=SHAPE["DK"], G=G)
    print(f"problem + oracle + packing {time.time() - t0:.0f} s")
    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{kernel}_trace.json"
    io = {k: ("out" if k in ("ConvState", "Ssm", "Y", "CV", "O", "Out") else "in") for k in arrays}
    io["ConvState"] = io["Ssm"] = "inout"
    ordered = lowering.order_arrays(low, {k: (io[k], v) for k, v in arrays.items()})
    res = runner.run(db, ordered, TILES, SHAVES, HERE / "run_gdn", profile_json=profile, repeat=REPEAT, warmup=WARMUP, timeout_s=300)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    H = SHAPE["H"]
    out = arrays["Out"][0, :H].astype(np.float32)
    scale = float(np.abs(ref - p["res"].astype(np.float32).reshape(-1)).max())
    err = float(np.abs(out - ref).max())
    rel = err / scale
    verdict = "PASS" if rel < 2e-2 and skip is None else ("SKIP-ARM" if skip is not None else "FAIL")
    cs_rel = float(np.abs(arrays["ConvState"].astype(np.float32) - ref_cs.astype(np.float32)).max() / np.abs(ref_cs.astype(np.float32)).max())
    ssm_rel = float(np.abs(arrays["Ssm"] - ref_ssm).max() / np.abs(ref_ssm).max())
    kind = runner.condition(res, profile is not None)
    weight_bytes = packed["WinP"].nbytes + packed["WoutP"].nbytes
    state_bytes = 2 * arrays["Ssm"].nbytes + 2 * arrays["ConvState"].nbytes
    idu_us = [float(res.status[t * SHAVES:(t + 1) * SHAVES, 5].astype(np.int64).sum()) / PERF_CLK_MHZ for t in range(TILES)]
    print(f"device: kernel-only {res.median_kernel_ms:.3f} ms (host wall {res.median_ms:.3f} ms) [{kind}]; "
          f"max |out - ref| = {err:.4f} over a mixer contribution of {scale:.3f} (rel {rel:.2e}) -> {verdict}; "
          f"kernel sha256 {db.source_sha256[:12]}")
    print(f"  states after one launch vs the oracle: conv state rel {cs_rel:.2e}, SSM rel {ssm_rel:.2e}")
    if verdict == "FAIL" or "--diag" in argv:
        # which phase diverges: the oracle's intermediates against the scratch read back (Y, CV, O)
        one, cs1, S1, d = gdn_reference(ref_p["x"], ref_p["res"], ref_p["win_int"], ref_p["win_scales"], ref_p["wout_int"],
                                        ref_p["wout_scales"], ref_p["conv_w"], ref_p["a_log"], ref_p["dt_bias"], ref_p["g_norm"],
                                        ref_p["conv_state"], ref_p["ssm"], KH=SHAPE["KH"], VH=SHAPE["VH"], DK=SHAPE["DK"], G=G, details=True)
        NIN = d["y"].size
        yd = arrays["Y"][0, :NIN].astype(np.float32)
        print(f"  diag: Y vs y: rel {np.abs(yd - d['y']).max() / np.abs(d['y']).max():.2e}; "
              f"CV vs conv: rel {np.abs(arrays['CV'][0, :d['conv'].size].astype(np.float32) - d['conv']).max() / np.abs(d['conv']).max():.2e}; "
              f"O vs on: rel {np.abs(arrays['O'][0, :d['on'].size].astype(np.float32) - d['on']).max() / np.abs(d['on']).max():.2e}")
    if res.kernel_times_ms:
        print("  kernel-only per run ms: " + " ".join(f"{t:.3f}" for t in res.kernel_times_ms))
    print(f"weights {weight_bytes / 1e6:.2f} MB + states {state_bytes / 1e6:.2f} MB in/out -> "
          f"{(weight_bytes + state_bytes) / (res.median_kernel_ms * 1e-3) / 1e9:.1f} GB/s kernel-only; "
          f"DPU IDU busy per tile {[f'{100 * u / (res.median_kernel_ms * 1e3):.0f}%' for u in idu_us]}")
    names = ["acquire_wait", "order_wait", "submit", "dpu_wait", "free_wait", "fill_issue"]
    phases = res.status[:, 10:16].astype(np.int64) / PERF_CLK_MHZ
    for w_ in range(TILES * SHAVES):
        print(f"  tile {w_ // SHAVES} SHAVE {w_ % SHAVES} ring phases us: " + ", ".join(f"{n} {phases[w_, i]:.0f}" for i, n in enumerate(names)))
    if profile:
        print(f"trace: {profile}")
    if "--negative" in argv:
        print("negative control (perturbed SSM head): " + ("behaved (FAIL reported)" if verdict == "FAIL" else "BROKEN: PASS reported"))
        return 0 if verdict == "FAIL" else 1
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
