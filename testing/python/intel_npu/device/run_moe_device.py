"""M3 device golden: one token through a routed + shared MoE layer in one launch (kernels.make_moe_decode,
Qwen3.6's shape by default: H 2048, 256 experts of inter 512, top-8 + a shared expert, int4 g128 weights).

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_moe_device.py [--e 256] [--topk 8] [--h 2048] [--i 512] [--stages 4]
                                                             [--bng 128] [--bnd 256] [--no-build] [--negative] [--profile]
                                                             [--no-pipeline] [--skip-dpu]
Unprofiled by default (a timing run, TURBO). The number that matters at decode is the weight bytes the layer
reads (router + 9 experts' int4 rows + scales) over the kernel-only time. --negative perturbs a group of the
FIRST ROUTED expert's gate rows (computed from the oracle's routing) so the comparison must FAIL; --e small
keeps the host-side packing quick (E + 1 experts are materialised).
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
from kernels import make_moe_decode, moe_reference, pack_moe  # noqa: E402

TILES, SHAVES = 3, 2
G = 128
PERF_CLK_MHZ = 19.2


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def main(argv):
    H, I, E, TOPK = arg(argv, "--h", 2048), arg(argv, "--i", 512), arg(argv, "--e", 256), arg(argv, "--topk", 8)
    stages = arg(argv, "--stages", 4)
    BNG, BND = arg(argv, "--bng", 128), arg(argv, "--bnd", 256)
    pipelined = "--no-pipeline" not in argv
    skip_dpu = "--skip-dpu" in argv
    kernel = (f"tl_moe_h{H}_i{I}_e{E}_k{TOPK}" + (f"_s{stages}" if stages != 4 else "") + (f"_g{BNG}" if BNG != 128 else "")
              + (f"_d{BND}" if BND != 256 else "") + ("" if pipelined else "_serial") + ("_nodpu" if skip_dpu else ""))
    low = lowering.lower(make_moe_decode(H=H, I=I, E=E, TOPK=TOPK, G=G, tiles=TILES, shaves=SHAVES, num_stages=stages,
                                         BNG=BNG, BND=BND, pipelined=pipelined, probe_skip_dpu=skip_dpu))
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
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, H)) * 0.5).astype(np.float16)
    router = (rng.standard_normal((E + 1, H)) * 0.05).astype(np.float16)
    gu_int = rng.integers(-8, 8, size=(E + 1, 2 * I, H), dtype=np.int8)
    gu_scales = rng.uniform(0.005, 0.02, size=(E + 1, 2 * I, H // G)).astype(np.float16)
    d_int = rng.integers(-8, 8, size=(E + 1, H, I), dtype=np.int8)
    d_scales = rng.uniform(0.005, 0.02, size=(E + 1, H, I // G)).astype(np.float16)
    res_in = (rng.standard_normal((1, H)) * 0.5).astype(np.float16)
    t0 = time.time()
    ref, order, w = moe_reference(x, router, gu_int, gu_scales, d_int, d_scales, res_in, TOPK, G)
    negative = "--negative" in argv
    if negative:
        e = int(order[0])
        gu_int[e, :I, :G] = -gu_int[e, :I, :G] - 1   # one whole group of the first routed expert's gate rows
    packed = pack_moe(router, gu_int, gu_scales, d_int, d_scales, tiles=TILES, shaves=SHAVES, G=G, BNG=BNG, BND=BND)
    print(f"oracle + packing {time.time() - t0:.0f} s; routed experts {order.tolist()} weights {np.round(w, 3).tolist()}")
    out = np.zeros((1, H), np.float16)
    logits = np.zeros((16, packed["Wr"].shape[0]), np.float16)
    part = np.zeros((TILES, H), np.float32)
    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{kernel}_trace.json"
    arrays = lowering.order_arrays(low, {"X": ("in", x), "Res": ("in", res_in), "Logits": ("out", logits), "Part": ("out", part),
                                         "Out": ("out", out), **{k: ("in", v) for k, v in packed.items()}})
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_moe", profile_json=profile, repeat=10, warmup=10, timeout_s=300)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    dev = out.astype(np.float32).reshape(-1)
    scale = float(np.abs(ref - res_in.astype(np.float32).reshape(-1)).max())   # the MoE's own contribution sets the bar
    err = float(np.abs(dev - ref).max())
    rel = err / scale
    verdict = "PASS" if rel < 2e-2 else "FAIL"
    dev_logits = logits[0, :E + 1].astype(np.float32)
    ref_logits = router.astype(np.float32) @ x.astype(np.float32).reshape(-1)
    dev_order = np.argsort(-dev_logits[:E], kind="stable")[:TOPK]
    kind = runner.condition(res, profile is not None)
    weight_bytes = packed["Wr"].nbytes + (TOPK + 1) * (packed["WguP"].nbytes + packed["WdP"].nbytes) // (E + 1)
    idu_us = [float(res.status[t * SHAVES:(t + 1) * SHAVES, 5].astype(np.int64).sum()) / PERF_CLK_MHZ for t in range(TILES)]
    print(f"device: kernel-only {res.median_kernel_ms:.3f} ms (host wall {res.median_ms:.3f} ms) [{kind}]; "
          f"max |out - ref| = {err:.4f} over a MoE contribution of {scale:.3f} (rel {rel:.2e}) -> {verdict}; "
          f"kernel sha256 {db.source_sha256[:12]}")
    print(f"  logits max |dev - ref| {np.abs(dev_logits - ref_logits).max():.4f}; device top-{TOPK} by its logits {dev_order.tolist()}"
          f" {'==' if dev_order.tolist() == order.tolist() else '!='} oracle {order.tolist()}")
    if res.kernel_times_ms:
        print("  kernel-only per run ms: " + " ".join(f"{t:.3f}" for t in res.kernel_times_ms))
    print(f"weights read {weight_bytes / 1e6:.2f} MB (router {packed['Wr'].nbytes / 1e6:.2f} + {TOPK + 1} experts) -> "
          f"{weight_bytes / (res.median_kernel_ms * 1e-3) / 1e9:.1f} GB/s kernel-only; "
          f"DPU IDU busy per tile {[f'{100 * u / (res.median_kernel_ms * 1e3):.0f}%' for u in idu_us]}")
    names = ["acquire_wait", "order_wait", "submit", "dpu_wait", "free_wait", "fill_issue"]
    phases = res.status[:, 10:16].astype(np.int64) / PERF_CLK_MHZ
    for w_ in range(TILES * SHAVES):
        print(f"  tile {w_ // SHAVES} SHAVE {w_ % SHAVES} ring phases us: " + ", ".join(f"{n} {phases[w_, i]:.0f}" for i, n in enumerate(names)))
    if profile:
        print(f"trace: {profile}")
    if negative:
        print("negative control (perturbed routed expert): " + ("behaved (FAIL reported)" if verdict == "FAIL" else "BROKEN: PASS reported"))
        return 0 if verdict == "FAIL" else 1
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
