"""P1 step 3 device golden: a WHOLE full-attention PREFILL decoder layer in ONE launch
(`make_attn_moe_layer_prefill`) at Qwen3.8's shape -- norm, qkv projection, per-head norms + RoPE + the cache
append, flash attention (GQA, causal mask, output gate), o_proj, residual, post norm, MoE routing, the grouped
expert FFN, and the MoE sum back into the stream.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_attn_moe_layer_prefill_device.py [--n 192] [--pos0 0]
                                                                                [--trials 3] [--bn 32]
                                                                                [--no-build] [--negative]
Unprofiled (a timing run, TURBO). Correctness is `--trials` independent runs against the composed oracle.
`--negative` zeroes the o_proj and every expert's down projection: the stream must then come back bit-identical,
which a layer that wrote over the residual instead of adding to it would fail while matching the oracle.
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
from kernels import (attn_moe_layer_prefill_reference, gm_scales, make_attn_moe_layer_prefill,  # noqa: E402
                     make_moe_route_prefill, pack_int4, rope_cos_sin)

TILES, SHAVES = 3, 2
H, NH, NKV, D, ROT, G = 2048, 16, 2, 256, 64, 128
I, E, TOPK = 512, 256, 8
GQ, KO = NH // NKV, NH * D
NQKV = NH * 2 * D + 2 * NKV * D
KGH, KGI = H // G, I // G
TB, TBF, BKV, MB = 8, 16, 128, 16
BM, BNR, MBe, BNG, BND = 64, 16, 16, 64, 64
THETA = 1e6
REL = 6e-2


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def main(argv):
    n, pos0 = arg(argv, "--n", 192), arg(argv, "--pos0", 0)
    trials = arg(argv, "--trials", 3)
    BN1 = BN2 = arg(argv, "--bn", 32)
    SMAX = arg(argv, "--smax", 1536)
    step = int(np.lcm(np.lcm(np.lcm(TB, TBF), TILES * SHAVES), BM))
    NP = -(-n // step) * step
    assert pos0 + NP <= SMAX, "the chunk must fit the cache window"
    NQKV_pad = -(-NQKV // BN1) * BN1
    _, rs = make_moe_route_prefill(N=NP, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=TILES, shaves=SHAVES,
                                   as_block=True)
    RR, SLOTS = rs["RR"], rs["SLOTS"]

    low = lowering.lower(make_attn_moe_layer_prefill(N=NP, H=H, NH=NH, NKV=NKV, D=D, ROT=ROT, SMAX=SMAX, I=I, E=E,
                                                     TOPK=TOPK, G=G, TB=TB, TBF=TBF, BKV=BKV, BN1=BN1, BN2=BN2,
                                                     MB=MB, BM=BM, BNR=BNR, MBe=MBe, BNG=BNG, BND=BND,
                                                     tiles=TILES, shaves=SHAVES))
    name = f"tl_attn_moe_layer_prefill_n{NP}_h{H}_s{SMAX}_bn{BN1}"
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
    wq = rng.integers(-8, 8, (NQKV_pad, H), dtype=np.int8)
    wqs = (rng.standard_normal((NQKV_pad, H // G), np.float32) * 0.02).astype(np.float16)
    wo = rng.integers(-8, 8, (H, KO), dtype=np.int8)
    wos = (rng.standard_normal((H, KO // G), np.float32) * 0.02).astype(np.float16)
    qn = (0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16)
    kn = (0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16)
    cs = np.stack([rope_cos_sin(pos0 + t, ROT, THETA).reshape(-1) for t in range(NP)]).astype(np.float32)
    wr = (rng.standard_normal((E + 1, H), np.float32) * 0.1).astype(np.float16)
    gu_int = rng.integers(-8, 8, (E + 1, 2 * I, H), dtype=np.int8)
    gu_s = (rng.standard_normal((E + 1, 2 * I, KGH), np.float32) * 0.02).astype(np.float16)
    d_int = rng.integers(-8, 8, (E + 1, H, I), dtype=np.int8)
    d_s = (rng.standard_normal((E + 1, H, KGI), np.float32) * 0.02).astype(np.float16)
    wr_pad = np.zeros((RR, H), np.float16)
    wr_pad[:E + 1] = wr
    kc0 = np.zeros((NKV, SMAX, D), np.float16)
    vt0 = np.zeros((NKV, D, SMAX), np.float16)
    if pos0:
        kc0[:, :pos0] = rng.standard_normal((NKV, pos0, D), np.float32).astype(np.float16)
        vt0[:, :, :pos0] = rng.standard_normal((NKV, D, pos0), np.float32).astype(np.float16)

    def run(wo_in, wos_in, d_in, ds_in):
        hio, kc, vt = h.copy(), kc0.copy(), vt0.copy()
        arrays = {"Hio": ("inout", hio), "InNorm": ("in", inn), "PostNorm": ("in", post),
                  "WqkvQ": ("in", pack_int4(wq)), "WqkvS": ("in", gm_scales(wqs)), "QNormW": ("in", qn),
                  "KNormW": ("in", kn),
                  "CosSin": ("in", cs), "Pos": ("in", np.array([[pos0]], np.int32)),
                  "Nt": ("in", np.array([[n]], np.int32)), "Kc": ("inout", kc), "Vt": ("inout", vt),
                  "WoQ": ("in", pack_int4(wo_in)), "WoS": ("in", gm_scales(wos_in)), "Wr": ("in", wr_pad),
                  "WguQ": ("in", pack_int4(gu_int.reshape(-1, H))), "WguS": ("in", gm_scales(gu_s.reshape(-1, KGH))),
                  "WdQ": ("in", pack_int4(d_in.reshape(-1, I))), "WdS": ("in", gm_scales(ds_in.reshape(-1, KGI))),
                  "Xs": ("out", np.zeros((NP, H), np.float16)),
                  "Ys": ("out", np.zeros((NP, NQKV_pad), np.float16)),
                  "Q": ("out", np.zeros((NKV, NP, GQ, D), np.float16)),
                  "Gate": ("out", np.zeros((NP, KO), np.float16)),
                  "Os": ("out", np.zeros((NP, KO), np.float16)), "Mix": ("out", np.zeros((NP, H), np.float16)),
                  "Logits": ("out", np.zeros((NP, RR), np.float16)),
                  "Ids": ("out", np.zeros((NP, TOPK), np.int32)),
                  "Wts": ("out", np.zeros((NP, 16), np.float32)),
                  "Cnt": ("out", np.zeros((E + 1,), np.int32)), "Off": ("out", np.zeros((E + 2,), np.int32)),
                  "Slot": ("out", np.zeros((SLOTS,), np.int32)),
                  "PartE": ("out", np.zeros((TILES, NP, H), np.float32)),
                  "MoeOut": ("out", np.zeros((NP, H), np.float32))}
        r = runner.run(db, lowering.order_arrays(low, arrays), TILES, SHAVES, HERE / "run_attn_moe_layer_prefill",
                       repeat=3, warmup=3, timeout_s=1800)
        runner.check_status(r.status)
        return hio, kc, vt, r

    ref, kc_ref, vt_ref = attn_moe_layer_prefill_reference(h, inn, post, wq, wqs, qn, kn, cs, wo, wos, kc0, vt0,
                                                           wr, gu_int, gu_s, d_int, d_s, pos0=pos0, n=n, NH=NH,
                                                           NKV=NKV, D=D, ROT=ROT, E=E, TOPK=TOPK, G=G)
    good, worst, res = 0, 0.0, None
    for _ in range(trials):
        hio, kc, vt, r = run(wo, wos, d_int, d_s)
        rel = float(np.abs(hio - ref).max() / np.abs(ref).max())
        krel = float(np.abs(kc.astype(np.float32) - kc_ref.astype(np.float32)).max()
                     / max(float(np.abs(kc_ref.astype(np.float32)).max()), 1e-6))
        vrel = float(np.abs(vt.astype(np.float32) - vt_ref.astype(np.float32)).max()
                     / max(float(np.abs(vt_ref.astype(np.float32)).max()), 1e-6))
        worst = max(worst, rel, krel, vrel)
        good += int(rel < REL and krel < REL and vrel < REL)
        res = r if res is None else res
    verdict = "PASS" if good == trials else "FAIL"
    print(f"attn+moe layer prefill: {res.median_kernel_ms:.3f} ms [{runner.condition(res, False)}] -> {verdict}; "
          f"{good}/{trials} trials match the composed oracle (stream and both caches), worst rel {worst:.2e}; "
          f"sha {db.source_sha256[:12]}")
    print(f"  {n} tokens at pos0 {pos0}, H={H}, E={E} top-{TOPK} I={I}, qkv {NQKV_pad}x{H} + o_proj {H}x{KO} "
          f"int4 g128, flash block {TBF}, BN={BN1}: {1e3 * res.median_kernel_ms / n:.1f} us per token")
    if res.kernel_times_ms:
        print("  kernel-only per run ms: " + " ".join(f"{t:.3f}" for t in res.kernel_times_ms))
    rc = 0 if verdict == "PASS" else 1

    if "--negative" in argv:
        hio, kc, _, _ = run(np.zeros_like(wo), np.zeros_like(wos), np.zeros_like(d_int), np.zeros_like(d_s))
        d = float(np.abs(hio - h).max())
        filled = float(np.abs(kc.astype(np.float32)).max())
        print(f"negative (o_proj and every expert down zeroed): {'behaved' if d == 0 and filled > 0 else 'BROKEN'}"
              f" -- max |stream - input| {d:.2e} (must be 0: both halves ADD to the residual), cache max "
              f"{filled:.2e} (must be > 0: the append still ran)")
        rc |= 0 if d == 0 and filled > 0 else 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
