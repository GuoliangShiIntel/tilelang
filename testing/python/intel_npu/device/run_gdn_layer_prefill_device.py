"""P4 step 4 device golden: a whole Gated DeltaNet PREFILL mixer layer in ONE launch
(`make_gdn_layer_prefill`) at Qwen3.6's shape -- norm, in_proj, conv, chunked scan, out_proj, residual.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_gdn_layer_prefill_device.py [--n 192] [--trials 12] [--negative]
Unprofiled (a timing run, TURBO). Correctness is `--trials` independent runs. `--negative` zeroes the out_proj:
the layer must then hand the residual stream back untouched, which a layer that wrote over the stream instead of
adding to it would fail while still matching the oracle.
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
from kernels import gdn_layer_prefill_reference, gm_scales, make_gdn_layer_prefill, pack_int4  # noqa: E402

TILES, SHAVES = 3, 2
H, KH, VH, DK, DV, G = 2048, 16, 32, 128, 128, 128
# BN=64 overruns at 813120 B per SHAVE, and that is one projection phase's own buffers, not phases failing
# to share: the number is byte-identical with and without question 10's hoist. `--bn` sweeps what does fit.
TB, C, MB = 32, 64, 16
QK, VC = KH * DK, VH * DV
CONV = 2 * QK + VC
NIN = CONV + VC


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def main(argv):
    n = arg(argv, "--n", 192)
    trials = arg(argv, "--trials", 12)
    BN1 = BN2 = arg(argv, "--bn", 32)   # 32 is 120.3 us per token against 16's 152.3; 64 does not fit
    NIN_pad = -(-NIN // BN1) * BN1
    NP = -(-n // (TB * C // np.gcd(TB, C))) * (TB * C // np.gcd(TB, C))

    low = lowering.lower(make_gdn_layer_prefill(N=NP, H=H, KH=KH, VH=VH, DK=DK, DV=DV, G=G, TB=TB, C=C,
                                                BN1=BN1, BN2=BN2, MB=MB, tiles=TILES, shaves=SHAVES))
    name = f"tl_gdn_layer_prefill_n{NP}_h{H}_bn{BN1}"
    if "--no-build" in argv:
        _, sha = runner.write_kernel(low.source, name)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe",
                                kernel=name, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(low.source, name)
        print(f"build {time.time() - t0:.0f} s")

    rng = np.random.default_rng(0)
    h = (rng.standard_normal((NP, H), np.float32) * 0.5).astype(np.float32)
    inn = (rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16)
    wi = rng.integers(-8, 8, (NIN_pad, H), dtype=np.int8)
    ws = (rng.standard_normal((NIN_pad, H // G), np.float32) * 0.02).astype(np.float16)
    wba = (rng.standard_normal((2 * VH, H), np.float32) * 0.05).astype(np.float16)
    wo = rng.integers(-8, 8, (H, VC), dtype=np.int8)
    wos = (rng.standard_normal((H, VC // G), np.float32) * 0.02).astype(np.float16)
    cw = (rng.standard_normal((4, CONV), np.float32) * 0.3).astype(np.float16)
    cs = (rng.standard_normal((3, CONV), np.float32) * 0.3).astype(np.float16)
    al = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    dbias = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    gn = (1.0 + rng.standard_normal((1, DV), np.float32) * 0.1).astype(np.float16)
    ssm = (rng.standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32)

    def run(wo_in, wos_in):
        hio, sm = h.copy(), ssm.copy()
        arrays = {"Hio": ("inout", hio), "InNorm": ("in", inn), "WinQ": ("in", pack_int4(wi)), "WinS": ("in", gm_scales(ws)),
                  "Wba": ("in", wba),
                  "ConvW": ("in", cw), "ConvState": ("in", cs), "ALog": ("in", al), "DtBias": ("in", dbias),
                  "GNorm": ("in", gn), "Nt": ("in", np.array([[n]], np.int32)), "Ssm": ("inout", sm),
                  "WoutQ": ("in", pack_int4(wo_in)), "WoutS": ("in", gm_scales(wos_in)),
                  "Xs": ("out", np.zeros((NP, H), np.float16)), "Ys": ("out", np.zeros((NP, NIN_pad), np.float16)),
                  "BAs": ("out", np.zeros((NP, 2 * VH), np.float16)),
                  "QKs": ("out", np.zeros((KH, NP, 2 * DK), np.float32)),
                  "Vs": ("out", np.zeros((VH, NP, DV), np.float16)),
                  "Gbs": ("out", np.zeros((VH, NP, 2), np.float32)),
                  "Zs": ("out", np.zeros((VH, NP, DV), np.float16)),
                  "Os": ("out", np.zeros((NP, VC), np.float16)), "Mix": ("out", np.zeros((NP, H), np.float16)),
                  "CsOut": ("out", np.zeros((3, CONV), np.float16))}
        r = runner.run(db, lowering.order_arrays(low, arrays), TILES, SHAVES, HERE / "run_gdn_layer_prefill",
                       repeat=3, warmup=3, timeout_s=900)
        runner.check_status(r.status)
        return hio, sm, r

    ref, _, ssm_ref = gdn_layer_prefill_reference(h, inn, wi, ws, cw, cs, al, dbias, gn, wo, wos, ssm,
                                                  n, KH, VH, DK, DV, G, wba=wba)
    good, worst, res = 0, 0.0, None
    for _ in range(trials):
        hio, sm, r = run(wo, wos)
        rel = float(np.abs(hio - ref).max() / np.abs(ref).max())
        srel = float(np.abs(sm - ssm_ref).max() / np.abs(ssm_ref).max())
        worst = max(worst, rel, srel)
        good += int(rel < 5e-2 and srel < 5e-2)
        res = r if res is None else res
    verdict = "PASS" if good == trials else "FAIL"
    print(f"gdn layer prefill: {res.median_kernel_ms:.3f} ms [{runner.condition(res, False)}] -> {verdict}; "
          f"{good}/{trials} trials match the composed oracle, worst rel {worst:.2e}; sha {db.source_sha256[:12]}")
    print(f"  {n} tokens, H={H}, in_proj {NIN_pad}x{H} + out_proj {H}x{VC} int4 g128, conv block {TB}, scan chunk "
          f"{C}: {1e3 * res.median_kernel_ms / n:.1f} us per token")
    rc = 0 if verdict == "PASS" else 1

    if "--negative" in argv:
        hio, _, _ = run(np.zeros_like(wo), np.zeros_like(wos))
        d = float(np.abs(hio - h).max())
        print(f"negative (out_proj zeroed): {'behaved' if d == 0 else 'BROKEN'} -- "
              f"max |stream - input| {d:.2e}; the layer must ADD to the residual, not replace it")
        rc |= 0 if d == 0 else 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
