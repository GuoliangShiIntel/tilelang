"""P1b device golden: the attention layer's qkv prefill phase (`kernels.make_attn_qkv_prefill`) -- the
projection output for `N` prompt tokens normed, roped and split into the flash phase's query / gate layout,
with the keys and values APPENDED to the cache. Nothing else writes Kc / Vt.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_attn_qkv_prefill_device.py [--n 192] [--pos0 0] [--tb 8]
                                                                         [--smax 1536] [--no-build] [--negative]
Unprofiled by default (a timing run, TURBO). `--negative` perturbs one projection row: the comparison must FAIL,
which is what proves the golden can bite.
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
from kernels import attn_qkv_prefill_reference, make_attn_qkv_prefill, rope_cos_sin  # noqa: E402

TILES, SHAVES = 3, 2
H, NH, NKV, D, ROT = 2048, 16, 2, 256, 64
GQ = NH // NKV
NQKV = NH * 2 * D + 2 * NKV * D
THETA = 1e6


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def main(argv):
    N, pos0 = arg(argv, "--n", 192), arg(argv, "--pos0", 0)
    TB, SMAX = arg(argv, "--tb", 8), arg(argv, "--smax", 1536)
    assert pos0 + N <= SMAX, "the chunk must fit the cache window"
    NP = -(-N // TB) * TB
    cfg = dict(N=N, H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, TB=TB, tiles=TILES, shaves=SHAVES)
    kernel = f"tl_attn_qkv_prefill_n{N}_nh{NH}_kv{NKV}_d{D}_s{SMAX}_tb{TB}"
    low = lowering.lower(make_attn_qkv_prefill(**cfg))
    for ln in low.source.splitlines():
        if "// intel_npu:" in ln:
            print(ln.strip()[:260])
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, kernel)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe",
                                kernel=kernel, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, kernel)
        print(f"build {time.time() - t0:.0f} s")

    rng = np.random.default_rng(0)
    y = rng.standard_normal((NP, NQKV), np.float32).astype(np.float16)
    qw = (0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16)
    kw = (0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16)
    cs = np.stack([rope_cos_sin(pos0 + t, ROT, THETA).reshape(-1) for t in range(NP)]).astype(np.float32)
    kc = np.zeros((NKV, SMAX, D), np.float16)
    vt = np.zeros((NKV, D, SMAX), np.float16)
    rq, rg, rkc, rvt = attn_qkv_prefill_reference(y, qw, kw, cs, pos0, N, NH, NKV, D, ROT, kc, vt)
    if "--negative" in argv:
        y[3] = y[3] + np.float16(4.0)

    q = np.zeros((NKV, NP, GQ, D), np.float16)
    gate = np.zeros((NKV, NP, GQ, D), np.float16)
    arrays = {"Ys": ("in", y), "QNormW": ("in", qw), "KNormW": ("in", kw), "CosSin": ("in", cs),
              "Pos": ("in", np.array([[pos0]], np.int32)), "Nt": ("in", np.array([[N]], np.int32)),
              "Q": ("out", q), "Gate": ("out", gate), "Kc": ("inout", kc), "Vt": ("inout", vt)}
    res = runner.run(db, lowering.order_arrays(low, arrays), TILES, SHAVES, HERE / "run_attn_qkv_prefill",
                     repeat=10, warmup=10, timeout_s=300)
    runner.check_status(res.status)

    worst, which = 0.0, ""
    for name, got, ref in (("Q", arrays["Q"][1], rq), ("Gate", arrays["Gate"][1], rg),
                           ("Kc", arrays["Kc"][1], rkc), ("Vt", arrays["Vt"][1], rvt)):
        scale = max(float(np.abs(ref.astype(np.float32)).max()), 1e-6)
        rel = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max()) / scale
        if rel > worst:
            worst, which = rel, name
        print(f"  {name:5s} rel {rel:.2e}")
    verdict = "PASS" if worst < 2e-2 else "FAIL"
    kind = runner.condition(res, False)
    print(f"attn qkv prefill: {res.median_kernel_ms:.3f} ms [{kind}] -> {verdict}; worst rel {worst:.2e} ({which}); "
          f"sha {db.source_sha256[:12]}")
    print(f"  {N} tokens at pos0 {pos0}, {NH} q-heads over {NKV} kv-heads of {D}, RoPE on {ROT}: "
          f"{1e3 * res.median_kernel_ms / N:.1f} us per token")
    if "--negative" in argv:
        print("negative (one projection row perturbed): " + ("behaved (FAIL)" if verdict == "FAIL" else "BROKEN: PASS"))
        return 0 if verdict == "FAIL" else 1
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
