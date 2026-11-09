"""P4 step 5 device golden: a WHOLE Gated DeltaNet PREFILL decoder layer in ONE launch
(`make_gdn_moe_layer_prefill`) at Qwen3.6's shape -- the mixer half, then the second norm, routing,
the grouped expert FFN and the weighted sum back into the residual stream.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_gdn_moe_layer_prefill_device.py [--n 192] [--trials 12]
        [--bn 32] [--negative]
Unprofiled (a timing run, TURBO). Correctness is `--trials` independent runs: one run inside a tolerance
cannot tell a correct kernel from one that is right four times in five. `--negative` zeroes BOTH the
out_proj and every expert's down projection, so the layer contributes nothing and must hand the residual
stream back bit-identical -- which a layer that wrote over the stream instead of adding would fail while
still matching the oracle.
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
from kernels import (gdn_moe_layer_prefill_reference, make_gdn_moe_layer_prefill,  # noqa: E402
                     make_moe_route_prefill, pack_int4)

TILES, SHAVES = 3, 2
H, KH, VH, DK, DV, G = 2048, 16, 32, 128, 128, 128
I, E, TOPK = 512, 256, 8
TB, C, MB = 32, 64, 16
BM, BNR, MBe, BNG, BND = 64, 16, 16, 64, 64
QK, VC = KH * DK, VH * DV
CONV = 2 * QK + VC
NIN = CONV + VC + 2 * VH
KGH, KGI = H // G, I // G
WORKERS = TILES * SHAVES


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def main(argv):
    n = arg(argv, "--n", 192)
    trials = arg(argv, "--trials", 12)
    BN1 = BN2 = arg(argv, "--bn", 32)
    NIN_pad = -(-NIN // BN1) * BN1
    NP = -(-n // (TB * C // np.gcd(TB, C))) * (TB * C // np.gcd(TB, C))
    _, rs = make_moe_route_prefill(N=NP, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=TILES, shaves=SHAVES,
                                   as_block=True)
    RR, SLOTS = rs["RR"], rs["SLOTS"]

    low = lowering.lower(make_gdn_moe_layer_prefill(N=NP, H=H, KH=KH, VH=VH, DK=DK, DV=DV, I=I, E=E, TOPK=TOPK,
                                                    G=G, TB=TB, C=C, BN1=BN1, BN2=BN2, MB=MB, BM=BM, BNR=BNR,
                                                    MBe=MBe, BNG=BNG, BND=BND, tiles=TILES, shaves=SHAVES))
    name = f"tl_gdn_moe_layer_prefill_n{NP}_h{H}_bn{BN1}"
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
    post = (rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16)
    wi = rng.integers(-8, 8, (NIN_pad, H), dtype=np.int8)
    ws = (rng.standard_normal((NIN_pad, H // G), np.float32) * 0.02).astype(np.float16)
    wo = rng.integers(-8, 8, (H, VC), dtype=np.int8)
    wos = (rng.standard_normal((H, VC // G), np.float32) * 0.02).astype(np.float16)
    cw = (rng.standard_normal((4, CONV), np.float32) * 0.3).astype(np.float16)
    cs = (rng.standard_normal((3, CONV), np.float32) * 0.3).astype(np.float16)
    al = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    dbias = (rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32)
    gn = (1.0 + rng.standard_normal((1, DV), np.float32) * 0.1).astype(np.float16)
    ssm = (rng.standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32)
    wr = (rng.standard_normal((E + 1, H), np.float32) * 0.1).astype(np.float16)
    gu_int = rng.integers(-8, 8, (E + 1, 2 * I, H), dtype=np.int8)
    gu_s = (rng.standard_normal((E + 1, 2 * I, KGH), np.float32) * 0.02).astype(np.float16)
    d_int = rng.integers(-8, 8, (E + 1, H, I), dtype=np.int8)
    d_s = (rng.standard_normal((E + 1, H, KGI), np.float32) * 0.02).astype(np.float16)
    wr_pad = np.zeros((RR, H), np.float16)
    wr_pad[:E + 1] = wr

    def run(wo_in, wos_in, d_in, ds_in):
        hio, sm = h.copy(), ssm.copy()
        arrays = {"Hio": ("inout", hio), "InNorm": ("in", inn), "PostNorm": ("in", post),
                  "WinQ": ("in", pack_int4(wi)), "WinS": ("in", ws),
                  "ConvW": ("in", cw), "ConvState": ("in", cs), "ALog": ("in", al), "DtBias": ("in", dbias),
                  "GNorm": ("in", gn), "Nt": ("in", np.array([[n]], np.int32)), "Ssm": ("inout", sm),
                  "WoutQ": ("in", pack_int4(wo_in)), "WoutS": ("in", wos_in), "Wr": ("in", wr_pad),
                  "WguQ": ("in", pack_int4(gu_int.reshape(-1, H))), "WguS": ("in", gu_s.reshape(-1, KGH)),
                  "WdQ": ("in", pack_int4(d_in.reshape(-1, I))), "WdS": ("in", ds_in.reshape(-1, KGI)),
                  "Xs": ("out", np.zeros((NP, H), np.float16)), "Ys": ("out", np.zeros((NP, NIN_pad), np.float16)),
                  "QKs": ("out", np.zeros((KH, NP, 2 * DK), np.float32)),
                  "Vs": ("out", np.zeros((VH, NP, DV), np.float16)),
                  "Gbs": ("out", np.zeros((VH, NP, 2), np.float32)),
                  "Zs": ("out", np.zeros((VH, NP, DV), np.float16)),
                  "Os": ("out", np.zeros((NP, VC), np.float16)), "Mix": ("out", np.zeros((NP, H), np.float16)),
                  "CsOut": ("out", np.zeros((3, CONV), np.float16)),
                  "Logits": ("out", np.zeros((NP, RR), np.float16)),
                  "Ids": ("out", np.zeros((NP, TOPK), np.int32)),
                  "Wts": ("out", np.zeros((NP, 16), np.float32)),
                  "Cnt": ("out", np.zeros((E + 1,), np.int32)), "Off": ("out", np.zeros((E + 2,), np.int32)),
                  "Slot": ("out", np.zeros((SLOTS,), np.int32)),
                  "PartE": ("out", np.zeros((WORKERS, NP, H), np.float32)),
                  "MoeOut": ("out", np.zeros((NP, H), np.float32))}
        r = runner.run(db, lowering.order_arrays(low, arrays), TILES, SHAVES, HERE / "run_gdn_moe_layer_prefill",
                       repeat=3, warmup=3, timeout_s=1800)
        runner.check_status(r.status)
        return hio, sm, r

    ref, _, ssm_ref = gdn_moe_layer_prefill_reference(h, inn, post, wi, ws, cw, cs, al, dbias, gn, wo, wos, ssm,
                                                      wr, gu_int, gu_s, d_int, d_s, n=n, KH=KH, VH=VH, DK=DK,
                                                      DV=DV, E=E, TOPK=TOPK, G=G)
    good, worst, res = 0, 0.0, None
    for _ in range(trials):
        hio, sm, r = run(wo, wos, d_int, d_s)
        rel = float(np.abs(hio - ref).max() / np.abs(ref).max())
        srel = float(np.abs(sm - ssm_ref).max() / np.abs(ssm_ref).max())
        worst = max(worst, rel, srel)
        good += int(rel < 6e-2 and srel < 6e-2)
        res = r if res is None else res
    verdict = "PASS" if good == trials else "FAIL"
    print(f"gdn+moe layer prefill: {res.median_kernel_ms:.3f} ms [{runner.condition(res, False)}] -> {verdict}; "
          f"{good}/{trials} trials match the composed oracle, worst rel {worst:.2e}; sha {db.source_sha256[:12]}")
    print(f"  {n} tokens, H={H}, E={E} top-{TOPK} I={I}, in_proj {NIN_pad}x{H} + out_proj {H}x{VC} int4 g128, "
          f"conv block {TB}, scan chunk {C}, BN={BN1}: {1e3 * res.median_kernel_ms / n:.1f} us per token")
    rc = 0 if verdict == "PASS" else 1

    if "--negative" in argv:
        hio, _, _ = run(np.zeros_like(wo), np.zeros_like(wos), np.zeros_like(d_int), np.zeros_like(d_s))
        d = float(np.abs(hio - h).max())
        print(f"negative (out_proj and every expert down zeroed): {'behaved' if d == 0 else 'BROKEN'} -- "
              f"max |stream - input| {d:.2e}; both halves must ADD to the residual, not replace it")
        rc |= 0 if d == 0 else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
