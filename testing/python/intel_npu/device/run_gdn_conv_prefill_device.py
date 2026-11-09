"""P3b prefill device golden: the GDN conv phase (`make_gdn_conv_prefill`) at Qwen3.6's GDN shape, and the
composition P3b -> P3 that P4 needs.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_gdn_conv_prefill_device.py [--n 192] [--tb 32] [--trials 12]
                                                                          [--negative] [--chain] [--no-build]
Unprofiled (a timing run, TURBO). Correctness is `--trials` independent runs, not one run inside a tolerance:
plan question 10 showed a single launch cannot tell a correct kernel from one that is right four times in five.
`--negative` perturbs one token and checks the conv's causality on the device -- only that token and its three
successors may move. `--chain` feeds the device's conv output into the scan kernel and compares against the two
oracles chained the same way.
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
from kernels import (gdn_conv_prefill_reference, gdn_scan_prefill_reference,  # noqa: E402
                     make_gdn_conv_prefill, make_gdn_mixer_prefill, make_gdn_scan_prefill)

TILES, SHAVES = 3, 2
KH, VH, DK, DV = 16, 32, 128, 128
QK, VC = KH * DK, VH * DV
CONV = 2 * QK + VC
NIN = CONV + VC + 2 * VH


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def problem(n, NP, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.standard_normal((NP, NIN), np.float32) * 0.5).astype(np.float16)
    y[n:] = 0
    conv_w = (rng.standard_normal((4, CONV), np.float32) * 0.3).astype(np.float16)
    cs = (rng.standard_normal((3, CONV), np.float32) * 0.3).astype(np.float16)
    a_log = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    dt_bias = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    return y, conv_w, cs, a_log, dt_bias


def main(argv):
    n = arg(argv, "--n", 192)
    TB = arg(argv, "--tb", 32)
    trials = arg(argv, "--trials", 12)
    NP = -(-n // TB) * TB

    low = lowering.lower(make_gdn_conv_prefill(N=NP, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB,
                                               tiles=TILES, shaves=SHAVES))
    name = f"tl_gdn_conv_prefill_n{NP}_kh{KH}_vh{VH}_tb{TB}"
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, name)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe",
                                kernel=name, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, name)
        print(f"build {time.time() - t0:.0f} s")

    y, cw, cs, al, dbias = problem(n, NP)

    def run(y_in):
        out = dict(QKo=np.zeros((KH, NP, 2 * DK), np.float32), Vo=np.zeros((VH, NP, DV), np.float16),
                   Gb=np.zeros((VH, NP, 2), np.float32), Zo=np.zeros((VH, NP, DV), np.float16),
                   CsOut=np.zeros((3, CONV), np.float16))
        arrays = {"Y": ("in", y_in), "ConvW": ("in", cw), "ConvState": ("in", cs), "ALog": ("in", al),
                  "DtBias": ("in", dbias), "Nt": ("in", np.array([[n]], np.int32)),
                  **{k: ("out", v) for k, v in out.items()}}
        r = runner.run(db, lowering.order_arrays(low, arrays), TILES, SHAVES, HERE / "run_gdn_conv_prefill",
                       repeat=10, warmup=10, timeout_s=600)
        runner.check_status(r.status)
        return out, r

    ref = dict(zip(("QKo", "Vo", "Gb", "Zo", "CsOut"),
                   gdn_conv_prefill_reference(y, cw, cs, al, dbias, n, KH, VH, DK, DV)))
    ref["QKo"] = ref["QKo"].reshape(KH, NP, 2 * DK)
    good, worst, first, res = 0, 0.0, None, None
    for _ in range(trials):
        got, r = run(y)
        rels = {k: float(np.abs(got[k].astype(np.float32) - ref[k].astype(np.float32)).max()
                         / (np.abs(ref[k].astype(np.float32)).max() + 1e-9)) for k in ref}
        worst = max(worst, max(rels.values()))
        good += int(max(rels.values()) < 5e-2)
        if first is None:
            first, res = got, r
    verdict = "PASS" if good == trials else "FAIL"
    kind = runner.condition(res, False)
    print(f"gdn conv prefill: {res.median_kernel_ms:.3f} ms [{kind}] -> {verdict}; "
          f"{good}/{trials} trials match the reference, worst rel {worst:.2e}; sha {db.source_sha256[:12]}")
    print(f"  {n} tokens of {NIN} channels, {KH} k-heads + {VH} v-heads, token block {TB}: "
          f"{1e3 * res.median_kernel_ms / n:.1f} us per token; "
          f"{NP * NIN * 2 / 1e6:.1f} MB of projection read")
    rc = 0 if verdict == "PASS" else 1

    if "--negative" in argv:
        t = n // 2 - 3                      # straddles a token block boundary with its 3-token reach
        y2 = y.copy()
        y2[t] = (y2[t].astype(np.float32) * 1.5).astype(np.float16)
        moved, _ = run(y2)
        d = np.abs(moved["QKo"].astype(np.float32) - first["QKo"].astype(np.float32)).max(axis=(0, 2))
        good_c = d[:t].max() == 0 and d[t] > 0 and d[t + 3] > 0 and d[t + 4:].max() == 0
        print(f"negative (token {t} perturbed): {'behaved' if good_c else 'BROKEN'} -- before it "
              f"{d[:t].max():.2e}, at it {d[t]:.2e}, +3 {d[t + 3]:.2e}, +4 and later {d[t + 4:].max():.2e}")
        rc |= 0 if good_c else 1

    if "--chain" in argv:
        C = 64
        gnorm = (1.0 + np.random.default_rng(1).standard_normal((1, DV), np.float32) * 0.1).astype(np.float16)
        ssm = (np.random.default_rng(2).standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32)
        slow = lowering.lower(make_gdn_scan_prefill(N=NP, KH=KH, VH=VH, DK=DK, DV=DV, C=C,
                                                    tiles=TILES, shaves=SHAVES))
        sdb = runner.build(slow.source, f"tl_gdn_scan_after_conv_n{NP}_c{C}")
        out, sm = np.zeros((NP, VC), np.float16), ssm.copy()
        r = runner.run(sdb, lowering.order_arrays(slow, {
            "QK": ("in", first["QKo"]), "V": ("in", first["Vo"]), "Gb": ("in", first["Gb"]),
            "Z": ("in", first["Zo"]), "GNorm": ("in", gnorm), "Nt": ("in", np.array([[n]], np.int32)),
            "Ssm": ("inout", sm), "Out": ("out", out)}), TILES, SHAVES, HERE / "run_gdn_conv_prefill",
            repeat=3, warmup=3, timeout_s=600)
        runner.check_status(r.status)
        qk, v, gb, z, _ = gdn_conv_prefill_reference(y, cw, cs, al, dbias, n, KH, VH, DK, DV)
        oracle, _ = gdn_scan_prefill_reference(qk, v, gb, z, gnorm, ssm, n)
        rel = float(np.abs(out.astype(np.float32)[:n] - oracle[:n]).max() / np.abs(oracle[:n]).max())
        print(f"chain conv -> scan: {'PASS' if rel < 5e-2 else 'FAIL'} rel {rel:.2e}; "
              f"{res.median_kernel_ms + r.median_kernel_ms:.3f} ms for the pair, "
              f"{1e3 * (res.median_kernel_ms + r.median_kernel_ms) / n:.1f} us per token")
        rc |= 0 if rel < 5e-2 else 1
    if "--fused" in argv:
        C = 64
        gnorm = (1.0 + np.random.default_rng(1).standard_normal((1, DV), np.float32) * 0.1).astype(np.float16)
        ssm = (np.random.default_rng(2).standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32)
        flow = lowering.lower(make_gdn_mixer_prefill(N=NP, KH=KH, VH=VH, DK=DK, DV=DV, TB=TB, C=C,
                                                     tiles=TILES, shaves=SHAVES))
        fdb = runner.build(flow.source, f"tl_gdn_mixer_prefill_n{NP}_tb{TB}_c{C}")
        oracle = gdn_scan_prefill_reference(*gdn_conv_prefill_reference(y, cw, cs, al, dbias, n, KH, VH, DK, DV)[:4],
                                            gnorm, ssm, n)[0]
        goodf, worstf, rf = 0, 0.0, None
        for _ in range(trials):
            out, sm = np.zeros((NP, VC), np.float16), ssm.copy()
            rf = runner.run(fdb, lowering.order_arrays(flow, {
                "Y": ("in", y), "ConvW": ("in", cw), "ConvState": ("in", cs), "ALog": ("in", al),
                "DtBias": ("in", dbias), "GNorm": ("in", gnorm), "Nt": ("in", np.array([[n]], np.int32)),
                "Ssm": ("inout", sm), "QKs": ("out", np.zeros((KH, NP, 2 * DK), np.float32)),
                "Vs": ("out", np.zeros((VH, NP, DV), np.float16)), "Gbs": ("out", np.zeros((VH, NP, 2), np.float32)),
                "Zs": ("out", np.zeros((VH, NP, DV), np.float16)), "CsOut": ("out", np.zeros((3, CONV), np.float16)),
                "Out": ("out", out)}), TILES, SHAVES, HERE / "run_gdn_conv_prefill", repeat=3, warmup=3, timeout_s=600)
            runner.check_status(rf.status)
            rel = float(np.abs(out.astype(np.float32)[:n] - oracle[:n]).max() / np.abs(oracle[:n]).max())
            worstf = max(worstf, rel)
            goodf += int(rel < 5e-2)
        print(f"fused mixer (one launch): {'PASS' if goodf == trials else 'FAIL'}; {goodf}/{trials} trials, "
              f"worst rel {worstf:.2e}; {rf.median_kernel_ms:.3f} ms = "
              f"{1e3 * rf.median_kernel_ms / n:.1f} us per token; sha {fdb.source_sha256[:12]}")
        rc |= 0 if goodf == trials else 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
