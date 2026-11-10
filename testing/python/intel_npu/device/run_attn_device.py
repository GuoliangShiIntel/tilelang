"""M1 decode device golden: one token through a full-attention layer in one launch (kernels.make_attn_decode,
Qwen3.6's shape by default: H 2048, 16 q-heads over 2 kv-heads of 256, RoPE 64, output gate, int4 g128 qkv [9216, 2048]
and o_proj [2048, 4096], a KV cache of 1536 positions).

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_attn_device.py [--pos 1000] [--smax 1536] [--stages 4] [--kv-stages 2]
                                                              [--no-build] [--negative] [--profile] [--no-pipeline]
Unprofiled by default (a timing run, TURBO). npu_run re-uploads the inputs before each launch, so the caches restart
every launch and the one-step oracle is the golden. --negative perturbs a cached key inside the window so the
comparison must FAIL.
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
from kernels import attn_reference, make_attn_decode, pack_attn  # noqa: E402
from test_intel_npu_m1_attn_decode import attn_arrays, attn_problem  # noqa: E402

TILES, SHAVES = 3, 2
G = 128
PERF_CLK_MHZ = 19.2
SHAPE = dict(H=2048, NH=16, NKV=2, D=256, ROT=64)


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def main(argv):
    pos, SMAX = arg(argv, "--pos", 1000), arg(argv, "--smax", 1536)
    stages, kv_stages = arg(argv, "--stages", 4), arg(argv, "--kv-stages", 2)
    pipelined = "--no-pipeline" not in argv
    cfg = dict(SHAPE, SMAX=SMAX, G=G, tiles=TILES, shaves=SHAVES, BN1=128, R1=64, BN2=64, R2=32, BKV=128)
    kernel = (f"tl_attn_h{SHAPE['H']}_nh{SHAPE['NH']}_kv{SHAPE['NKV']}_s{SMAX}" + (f"_q{stages}" if stages != 4 else "")
              + (f"_k{kv_stages}" if kv_stages != 2 else "") + ("" if pipelined else "_serial"))
    low = lowering.lower(make_attn_decode(num_stages=stages, kv_stages=kv_stages, pipelined=pipelined, **cfg))
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
    p = attn_problem(0, pos, **cfg)
    ref_p = attn_problem(0, pos, **cfg)   # unperturbed, so the negative must fail
    if "--negative" in argv:
        p["kc"][1, 17] += np.float16(2.0)
    packed = pack_attn(p["wqkv_int"], p["wqkv_scales"], p["wo_int"], p["wo_scales"], tiles=TILES, BN1=128, BN2=64)
    arrays = attn_arrays(p, packed, **cfg)
    ref, ref_kc, ref_vt = attn_reference(ref_p["x"], ref_p["res"], ref_p["wqkv_int"], ref_p["wqkv_scales"], ref_p["wo_int"],
                                         ref_p["wo_scales"], ref_p["qnorm_w"], ref_p["knorm_w"], ref_p["cos_sin"], pos,
                                         ref_p["kc"], ref_p["vt"], NH=SHAPE["NH"], NKV=SHAPE["NKV"], D=SHAPE["D"], ROT=SHAPE["ROT"], G=G)
    print(f"problem + oracle + packing {time.time() - t0:.0f} s; pos {pos} of {SMAX}")
    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{kernel}_trace.json"
    io = {k: ("out" if k in ("Y", "QN", "Part", "O", "Out") else "in") for k in arrays}
    io["Kc"] = io["Vt"] = "inout"
    ordered = lowering.order_arrays(low, {k: (io[k], v) for k, v in arrays.items()})
    res = runner.run(db, ordered, TILES, SHAVES, HERE / "run_attn", profile_json=profile, repeat=10, warmup=10, timeout_s=300)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    H = SHAPE["H"]
    out = arrays["Out"][0, :H].astype(np.float32)
    scale = float(np.abs(ref - ref_p["res"].astype(np.float32).reshape(-1)).max())
    err = float(np.abs(out - ref).max())
    rel = err / scale
    verdict = "PASS" if rel < 2e-2 else "FAIL"
    k_err = float(np.abs(arrays["Kc"][:, pos].astype(np.float32) - ref_kc[:, pos].astype(np.float32)).max())
    v_err = float(np.abs(arrays["Vt"][:, :, pos].astype(np.float32) - ref_vt[:, :, pos].astype(np.float32)).max())
    kind = runner.condition(res, profile is not None)
    weight_bytes = packed["WqkvQ"].nbytes + packed["WqkvS"].nbytes + packed["WoQ"].nbytes + packed["WoS"].nbytes
    kv_bytes = 2 * SHAPE["NKV"] * SMAX * SHAPE["D"] * 2
    idu_us = [float(res.status[t * SHAVES:(t + 1) * SHAVES, 5].astype(np.int64).sum()) / PERF_CLK_MHZ for t in range(TILES)]
    print(f"device: kernel-only {res.median_kernel_ms:.3f} ms (host wall {res.median_ms:.3f} ms) [{kind}]; "
          f"max |out - ref| = {err:.4f} over an attention contribution of {scale:.3f} (rel {rel:.2e}) -> {verdict}; "
          f"kernel sha256 {db.source_sha256[:12]}")
    print(f"  appended K / V^T entries vs oracle: max abs {k_err:.2e} / {v_err:.2e}")
    if res.kernel_times_ms:
        print("  kernel-only per run ms: " + " ".join(f"{t:.3f}" for t in res.kernel_times_ms))
    print(f"weights {weight_bytes / 1e6:.2f} MB + KV cache streamed {kv_bytes / 1e6:.2f} MB -> "
          f"{(weight_bytes + kv_bytes) / (res.median_kernel_ms * 1e-3) / 1e9:.1f} GB/s kernel-only; "
          f"DPU IDU busy per tile {[f'{100 * u / (res.median_kernel_ms * 1e3):.0f}%' for u in idu_us]}")
    names = ["acquire/live_wait", "order_wait", "submit", "dpu_wait", "free_wait", "fill_issue"]
    phases = res.status[:, 10:16].astype(np.int64) / PERF_CLK_MHZ
    for w_ in range(TILES * SHAVES):
        print(f"  tile {w_ // SHAVES} SHAVE {w_ % SHAVES} ring phases us: " + ", ".join(f"{n} {phases[w_, i]:.0f}" for i, n in enumerate(names)))
    if profile:
        print(f"trace: {profile}")
    if "--negative" in argv:
        print("negative control (perturbed cached key): " + ("behaved (FAIL reported)" if verdict == "FAIL" else "BROKEN: PASS reported"))
        return 0 if verdict == "FAIL" else 1
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
