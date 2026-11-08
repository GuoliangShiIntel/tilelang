"""P3 prefill device golden: the Gated DeltaNet chunked scan (`make_gdn_scan_prefill`) at Qwen3.6's GDN shape.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_gdn_scan_prefill_device.py [--n 128] [--c 64] [--kh 16] [--vh 32]
                                                                          [--no-build] [--negative] [--profile]
Unprofiled by default (a timing run, TURBO). The oracle is the RECURRENCE, not the chunked algebra, so a wrong
triangular inverse, decay matrix or chunk carry shows up as a wrong result. `--negative` perturbs ONE token: only
outputs at or after it may move, which is the causality a per-token loop would give for free and the chunked form
has to earn.

This kernel is also the backend's CMX high-water mark -- its `T.alloc_local` buffers are per-SHAVE -- so the
`cmx_used` it reports against the 1,473,536 B workspace is part of the result.
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
from kernels import gdn_scan_prefill_reference, make_gdn_scan_prefill  # noqa: E402

TILES, SHAVES = 3, 2
DK = DV = 128
EPS = 1e-6


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def problem(n, NP, KH, VH, seed=0):
    rng = np.random.default_rng(seed)
    qk = rng.standard_normal((KH, NP, 2, DK), np.float32) * 0.3
    qk /= np.sqrt((qk ** 2).sum(-1, keepdims=True)) + 1e-6      # q and k reach the scan l2-normalised
    qk[:, n:] = 0
    v = (rng.standard_normal((VH, NP, DV), np.float32) * 0.5).astype(np.float16)
    gb = np.zeros((VH, NP, 2), np.float32)
    gb[:, :, 0] = -np.abs(rng.standard_normal((VH, NP), np.float32)) * 0.05
    gb[:, :, 1] = 1.0 / (1.0 + np.exp(-rng.standard_normal((VH, NP), np.float32)))
    z = (rng.standard_normal((VH, NP, DV), np.float32) * 0.5).astype(np.float16)
    gnorm = (1.0 + rng.standard_normal((1, DV), np.float32) * 0.1).astype(np.float16)
    ssm = (rng.standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32)
    return qk, v, gb, z, gnorm, ssm


def main(argv):
    n = arg(argv, "--n", 128)
    C = arg(argv, "--c", 64)
    KH, VH = arg(argv, "--kh", 16), arg(argv, "--vh", 32)
    tiles, shaves = arg(argv, "--tiles", TILES), arg(argv, "--shaves", SHAVES)
    NP = -(-n // C) * C
    profile = "--profile" in argv

    low = lowering.lower(make_gdn_scan_prefill(N=NP, KH=KH, VH=VH, DK=DK, DV=DV, C=C,
                                               tiles=tiles, shaves=shaves, eps=EPS))
    name = f"tl_gdn_scan_prefill_n{NP}_kh{KH}_vh{VH}_c{C}_t{tiles}s{shaves}"
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, name)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe",
                                kernel=name, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, name)
        print(f"build {time.time() - t0:.0f} s")

    qk, v, gb, z, gnorm, ssm = problem(n, NP, KH, VH)

    def run(qk_in, repeat):
        out = np.zeros((NP, VH * DV), np.float16)
        sm = ssm.copy()
        arrays = {"QK": ("in", qk_in.reshape(KH, NP, 2 * DK)), "V": ("in", v), "Gb": ("in", gb), "Z": ("in", z),
                  "GNorm": ("in", gnorm), "Nt": ("in", np.array([[n]], np.int32)),
                  "Ssm": ("inout", sm), "Out": ("out", out)}
        r = runner.run(db, lowering.order_arrays(low, arrays), tiles, shaves, HERE / "run_gdn_scan_prefill",
                       repeat=repeat, warmup=repeat, timeout_s=600,
                       profile_json=(HERE / "evidence" / f"{name}_trace.json") if profile else None)
        runner.check_status(r.status)
        return out, sm, r

    # npu_run restores every input before each launch (npu_run.cpp, the memcpy over `ps`), so repeating does not
    # advance the state -- it only rolls the dice again, and the readback is the LAST launch. The defect this
    # kernel still has is intermittent and needs more than one tile, so the gate is independent trials.
    trials = arg(argv, "--trials", 3)
    rels, fails = [], 0
    ref, sref = gdn_scan_prefill_reference(qk, v, gb, z, gnorm, ssm, n, eps=EPS)
    got, res = None, None
    for _ in range(trials):
        o, sm, r = run(qk, 10)
        rel = float(np.abs(o[:n].astype(np.float32) - ref[:n]).max() / np.abs(ref[:n]).max())
        srel = float(np.abs(sm - sref).max() / np.abs(sref).max())
        rels.append(rel)
        if not (rel < 5e-2 and srel < 5e-2):
            fails += 1
        if got is None:
            got, res = o.copy(), r
    verdict = "PASS" if fails == 0 else "FAIL"
    kind = runner.condition(res, profile)
    held = int(res.status[0][3])
    worst = max(rels) if not any(np.isnan(rels)) else float("nan")
    print(f"gdn scan prefill: {res.median_kernel_ms:.3f} ms [{kind}] -> {verdict}; "
          f"{trials - fails}/{trials} trials agree with the recurrence, worst rel {worst:.2e}; "
          f"sha {db.source_sha256[:12]}")
    print(f"  {n} tokens, {VH} v-heads, chunk {C}: {NP // C} chunks x {-(-VH // (tiles * shaves))} head rounds "
          f"on {tiles}x{shaves}; fits the 1473536 B workspace ({held} B still held at finish); "
          f"{1e3 * res.median_kernel_ms / n:.1f} us per token")
    rc = 0 if verdict == "PASS" else 1

    if "--negative" in argv:
        t = n // 2
        q2 = qk.copy()
        q2[:, t] *= 1.5
        moved, _, _ = run(q2, 1)
        d = np.abs(moved[:n].astype(np.float32) - got[:n].astype(np.float32)).max(axis=1)
        before, after = float(d[:t].max()), float(d[t:].max())
        good = before < 1e-3 < after
        print(f"negative (token {t} perturbed): {'behaved' if good else 'BROKEN'} -- "
              f"max change before it {before:.2e}, at or after it {after:.2e}")
        rc |= 0 if good else 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
