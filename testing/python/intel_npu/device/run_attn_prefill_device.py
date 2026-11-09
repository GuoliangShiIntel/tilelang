"""P1 prefill device golden: `N` prompt tokens through the full-attention prefill phase in one launch
(`kernels.make_attn_prefill`, Qwen3.8's shape by default: 16 q-heads over 2 kv-heads of 256, GQA 8:1,
causal mask, per-head output gate, a KV cache of 1536 positions).

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_attn_prefill_device.py [--n 128] [--pos0 0] [--tb 8] [--bkv 128]
                                                                     [--smax 1536] [--no-build] [--negative] [--profile]
Unprofiled by default (a timing run, TURBO). The cache holds every position the call can see and zeros past it,
as it does on the device after the chunk's own append. `--negative` perturbs a key a token must NOT see: the
causal mask makes that a no-op, so the run must still PASS -- and `--negative-seen` perturbs one it must see,
which must FAIL. Both controls are needed: the first proves the mask is live, the second proves the oracle bites.
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
from kernels import attn_prefill_reference, make_attn_prefill  # noqa: E402

TILES, SHAVES = 3, 2
PERF_CLK_MHZ = 19.2
NKV, GQ, D = 2, 8, 256


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def main(argv):
    N, pos0 = arg(argv, "--n", 128), arg(argv, "--pos0", 0)
    TB, BKV, SMAX = arg(argv, "--tb", 4), arg(argv, "--bkv", 128), arg(argv, "--smax", 1536)
    assert pos0 + N <= SMAX, "the chunk must fit the cache window"
    cfg = dict(N=N, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, TB=TB, BKV=BKV, tiles=TILES, shaves=SHAVES)
    kernel = f"tl_attn_prefill_n{N}_kv{NKV}_g{GQ}_d{D}_s{SMAX}" + (f"_tb{TB}" if TB != 4 else "")
    low = lowering.lower(make_attn_prefill(**cfg))
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
    q = rng.standard_normal((NKV, N, GQ, D), np.float32).astype(np.float16)
    gate = rng.standard_normal((N, NKV * GQ * D), np.float32).astype(np.float16)
    kc = np.zeros((NKV, SMAX, D), np.float16)
    vt = np.zeros((NKV, D, SMAX), np.float16)
    last = pos0 + N
    kc[:, :last] = rng.standard_normal((NKV, last, D), np.float32).astype(np.float16)
    vt[:, :, :last] = rng.standard_normal((NKV, D, last), np.float32).astype(np.float16)
    ref = attn_prefill_reference(q, kc, vt, gate, pos0)
    if "--negative" in argv:          # a key no token of this chunk may see: the mask must absorb it
        kc[:, last - 1] += np.float16(8.0)
        ref_mask = attn_prefill_reference(q, kc, vt, gate, pos0)
        ref[N - 1] = ref_mask[N - 1]   # the last token legitimately sees it
    if "--negative-seen" in argv:     # a key every token sees: the comparison must FAIL
        kc[:, 0] += np.float16(8.0)

    out = np.zeros((N, NKV * GQ * D), np.float16)
    arrays = {"Q": ("in", q), "Kc": ("in", kc), "Vt": ("in", vt), "Gate": ("in", gate),
              "Pos": ("in", np.array([[pos0]], np.int32)), "O": ("out", out)}
    profile = None
    if "--profile" in argv:
        (HERE / "evidence").mkdir(exist_ok=True)
        profile = HERE / "evidence" / f"{kernel}_trace.json"
    res = runner.run(db, lowering.order_arrays(low, arrays), TILES, SHAVES, HERE / "run_attn_prefill",
                     profile_json=profile, repeat=10, warmup=10, timeout_s=300)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)

    got = arrays["O"][1].astype(np.float32)
    err = float(np.abs(got - ref).max())
    scale = float(np.abs(ref).max())
    rel = err / scale
    verdict = "PASS" if rel < 3e-2 else "FAIL"
    kind = runner.condition(res, profile is not None)
    # ops: per (kv head, token, q head) two gemms over the keys it sees -> 2 * 2 * sum_t (pos0 + t + 1) * GQ * D
    seen = sum(pos0 + t + 1 for t in range(N))
    ops = 4.0 * NKV * GQ * D * seen
    print(f"device: kernel-only {res.median_kernel_ms:.3f} ms (host wall {res.median_ms:.3f} ms) [{kind}]; "
          f"max |out - ref| = {err:.4f} over {scale:.3f} (rel {rel:.2e}) -> {verdict}; "
          f"kernel sha256 {db.source_sha256[:12]}")
    print(f"  {N} tokens at pos0 {pos0}: {seen} (query, key) pairs per kv head, "
          f"{ops / (res.median_kernel_ms * 1e-3) / 1e12:.2f} TOPS; {1e3 * res.median_kernel_ms / N:.1f} us per token")
    if res.kernel_times_ms:
        print("  kernel-only per run ms: " + " ".join(f"{t:.3f}" for t in res.kernel_times_ms))
    idu_us = [float(res.status[t * SHAVES:(t + 1) * SHAVES, 5].astype(np.int64).sum()) / PERF_CLK_MHZ for t in range(TILES)]
    print(f"  DPU IDU busy per tile {[f'{100 * u / (res.median_kernel_ms * 1e3):.0f}%' for u in idu_us]}")
    if profile:
        print(f"trace: {profile}")
    if "--negative-seen" in argv:
        print("negative control (a key every token sees): " + ("behaved (FAIL)" if verdict == "FAIL" else "BROKEN: PASS"))
        return 0 if verdict == "FAIL" else 1
    if "--negative" in argv:
        print("causal control (a key only the last token may see): " + ("behaved (PASS)" if verdict == "PASS" else "BROKEN: FAIL"))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
