"""Gemma4 V5: the SAME `make_gemma4_layer_prefill` composition, REAL Gemma-4-26B-A4B-it dimensions,
on REAL NVL/NPU6 hardware (synthetic weights -- the real-checkpoint numeric validation already ran in
host-sim, see test_intel_npu_gemma4_real_checkpoint_layer0.py; this script is specifically about
proving the kernel runs on SILICON, which host-sim cannot tell you). Requires GENERIC_COMPUTE_ROOT and
MV_TOOLS_PATH (see docs/developer_guide/gemma4_enablement_plan.md's V5 section for this machine's
paths). `tl.intel_npu.hw_sync_grid: False` is applied because this kernel has plenty of scalar SHAVE
work (every RMSNorm) before its grid barriers -- the known NVL hang this backend already has a fix for.

Usage: python testing/python/intel_npu/device/run_gemma4_layer_device.py [--no-build]
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402
import tvm  # noqa: E402
import runner  # noqa: E402
from kernels import pack_int4  # noqa: E402
from gemma4_kernels import gemma4_decoder_layer_reference, make_gemma4_layer_prefill  # noqa: E402
import tilelang.intel_npu.language as T  # noqa: E402

N = 8
H = 2816
NH = 16
NKV, D = 8, 256  # sliding attention layer
FI = 2112
E, TOPK = 128, 8
I_EXP = 704
G = 64
SMAX = 256
TILES, SHAVES = 4, 2
TB, BKV, BN = 4, 32, 16
MB_EXPERT, MB_PROJ, BNW_GU, BNW_D = 4, 4, 16, 16
KERNEL = "tl_gemma4_layer_real_scale"


def _dequant_gm(w_int, scale_gm, G):
    n, k = w_int.shape[-2:]
    s = np.moveaxis(scale_gm, -2, -1).astype(np.float32)
    w = w_int.astype(np.float32)
    lead = w.shape[:-2]
    return (w.reshape(*lead, n, k // G, G) * s[..., None]).reshape(*lead, n, k)


def _rand_q(rng, shape):
    return rng.integers(-7, 8, shape, np.int8)


def _pack(w_int):
    if w_int.ndim == 2:
        return pack_int4(w_int)
    return np.stack([_pack(w_int[i]) for i in range(w_int.shape[0])])


def main(argv):
    full = False
    GQ = NH // NKV
    KO = NH * D
    NQKV = NH * D + 2 * NKV * D

    rng = np.random.default_rng(42)
    h = rng.standard_normal((N, H), np.float32) * 0.1
    norms = {name: rng.uniform(0.5, 1.5, H).astype(np.float32) for name in (
        "input_norm", "post_attn_norm", "pre_ffn_norm", "post_ffn_1_norm", "pre_ffn_2_norm",
        "post_ffn_2_norm", "post_ffn_norm", "router_scale")}
    q_norm = rng.uniform(0.5, 1.5, D).astype(np.float32)
    k_norm = rng.uniform(0.5, 1.5, D).astype(np.float32)
    half = D // 2
    angles = rng.uniform(0, 2 * np.pi, (N, half)).astype(np.float32)
    cos_sin = np.concatenate([np.tile(np.cos(angles), 2), np.tile(np.sin(angles), 2)], axis=1).astype(np.float32)

    wq_int = _rand_q(rng, (NQKV, H))
    wq_s = rng.uniform(0.01, 0.05, (H // G, NQKV)).astype(np.float16)
    wo_int = _rand_q(rng, (H, KO))
    wo_s = rng.uniform(0.01, 0.05, (KO // G, H)).astype(np.float16)
    wg_int, wg_s = _rand_q(rng, (FI, H)), rng.uniform(0.01, 0.05, (H // G, FI)).astype(np.float16)
    wu_int, wu_s = _rand_q(rng, (FI, H)), rng.uniform(0.01, 0.05, (H // G, FI)).astype(np.float16)
    wd_int, wd_s = _rand_q(rng, (H, FI)), rng.uniform(0.01, 0.05, (FI // G, H)).astype(np.float16)
    wr = rng.standard_normal((E, H), np.float32).astype(np.float16)
    per_expert_scale = rng.uniform(0.5, 1.5, E).astype(np.float32)
    layer_scalar = 1.0

    ewg_int = _rand_q(rng, (E, I_EXP, H))
    ewg_s = rng.uniform(0.01, 0.05, (E, H // G, I_EXP)).astype(np.float16)
    ewu_int = _rand_q(rng, (E, I_EXP, H))
    ewu_s = rng.uniform(0.01, 0.05, (E, H // G, I_EXP)).astype(np.float16)
    ewd_int = _rand_q(rng, (E, H, I_EXP))
    ewd_s = rng.uniform(0.01, 0.05, (E, I_EXP // G, H)).astype(np.float16)

    weights_ref = dict(
        **norms, q_norm=q_norm, k_norm=k_norm, cos_sin=cos_sin,
        wq=_dequant_gm(wq_int, wq_s, G), wo=_dequant_gm(wo_int, wo_s, G),
        wg=_dequant_gm(wg_int, wg_s, G), wu=_dequant_gm(wu_int, wu_s, G), wd=_dequant_gm(wd_int, wd_s, G),
        router_proj=wr.astype(np.float32), per_expert_scale=per_expert_scale, layer_scalar=layer_scalar,
        ewg=_dequant_gm(ewg_int, ewg_s, G), ewu=_dequant_gm(ewu_int, ewu_s, G), ewd=_dequant_gm(ewd_int, ewd_s, G),
        ids_topk=TOPK,
    )

    layer_macro, sizes = make_gemma4_layer_prefill(
        N=N, H=H, NH=NH, NKV=NKV, D=D, FI=FI, E=E, TOPK=TOPK, I=I_EXP, G=G, SMAX=SMAX, full=full, window=None,
        TB=TB, BKV=BKV, BN=BN, MB_expert=MB_EXPERT, MB_proj=MB_PROJ, BNW_gu=BNW_GU, BNW_d=BNW_D,
        tiles=TILES, shaves=SHAVES,
    )
    NQKV_s, GQ_s, KO_s, SLOTS = sizes["NQKV"], sizes["GQ"], sizes["KO"], sizes["SLOTS"]
    assert (NQKV_s, GQ_s, KO_s) == (NQKV, GQ, KO)

    @T.prim_func
    def layer_kernel(
        Hio: T.Tensor((N, H), T.float32), InNorm: T.Tensor((1, H), T.float16), QNorm: T.Tensor((1, D), T.float16),
        KNorm: T.Tensor((1, D), T.float16), CosSin: T.Tensor((N, 2 * D), T.float32),
        PostAttnNorm: T.Tensor((1, H), T.float16), PreFfnNorm: T.Tensor((1, H), T.float16),
        PostFfn1Norm: T.Tensor((1, H), T.float16), PreFfn2Norm: T.Tensor((1, H), T.float16),
        PostFfn2Norm: T.Tensor((1, H), T.float16), PostFfnNorm: T.Tensor((1, H), T.float16),
        RouterScale: T.Tensor((1, H), T.float16),
        WqkvQ: T.Tensor((NQKV, H // 2), T.uint8), WqkvS: T.Tensor((H // G, NQKV), T.float16),
        WoQ: T.Tensor((H, KO // 2), T.uint8), WoS: T.Tensor((KO // G, H), T.float16),
        WgateQ: T.Tensor((FI, H // 2), T.uint8), WgateS: T.Tensor((H // G, FI), T.float16),
        WupQ: T.Tensor((FI, H // 2), T.uint8), WupS: T.Tensor((H // G, FI), T.float16),
        WdownQ: T.Tensor((H, FI // 2), T.uint8), WdownS: T.Tensor((FI // G, H), T.float16),
        WR: T.Tensor((E, H), T.float16),
        EWgateQ: T.Tensor((E, I_EXP, H // 2), T.uint8), EWgateS: T.Tensor((E, H // G, I_EXP), T.float16),
        EWupQ: T.Tensor((E, I_EXP, H // 2), T.uint8), EWupS: T.Tensor((E, H // G, I_EXP), T.float16),
        EWdownQ: T.Tensor((E, H, I_EXP // 2), T.uint8), EWdownS: T.Tensor((E, I_EXP // G, H), T.float16),
        PerExpertScale: T.Tensor((E, 1), T.float32), LayerScalar: T.Tensor((1, 1), T.float32),
        Kc: T.Tensor((NKV, SMAX, D), T.float16), Vt: T.Tensor((NKV, D, SMAX), T.float16),
        Pos: T.Tensor((1, 1), T.int32), Nt: T.Tensor((1, 1), T.int32),
        Xn: T.Tensor((N, H), T.float16), Ys: T.Tensor((N, NQKV), T.float16), Q: T.Tensor((NKV, N, GQ, D), T.float16),
        AttnO: T.Tensor((N, KO), T.float16), AttnP: T.Tensor((N, H), T.float16), PostAttnN: T.Tensor((N, H), T.float16),
        Xdense: T.Tensor((N, H), T.float16), Dense: T.Tensor((N, H), T.float16), DenseN: T.Tensor((N, H), T.float16),
        Gate: T.Tensor((N, FI), T.float16), Up: T.Tensor((N, FI), T.float16), Act: T.Tensor((N, FI), T.float16),
        Xroute: T.Tensor((N, H), T.float16), Logits: T.Tensor((N, E), T.float16),
        Ids: T.Tensor((N, TOPK), T.int32), Wts: T.Tensor((N, TOPK), T.float32),
        Off: T.Tensor((E + 1,), T.int32), Slot: T.Tensor((SLOTS,), T.int32), Xmoe: T.Tensor((N, H), T.float16),
        MoePart: T.Tensor((TILES * SHAVES, N, H), T.float32), MoeOut: T.Tensor((N, H), T.float32),
        MoeN: T.Tensor((N, H), T.float16), Combined: T.Tensor((N, H), T.float16),
    ):
        with T.Kernel(TILES, threads=SHAVES) as tile:
            layer_macro(Hio, InNorm, QNorm, KNorm, CosSin, PostAttnNorm, PreFfnNorm, PostFfn1Norm, PreFfn2Norm,
                       PostFfn2Norm, PostFfnNorm, RouterScale, WqkvQ, WqkvS, WoQ, WoS, WgateQ, WgateS, WupQ, WupS,
                       WdownQ, WdownS, WR, EWgateQ, EWgateS, EWupQ, EWupS, EWdownQ, EWdownS, PerExpertScale,
                       LayerScalar, Kc, Vt, Pos, Nt, Xn, Ys, Q, AttnO, AttnP, PostAttnN, Xdense, Dense, DenseN,
                       Gate, Up, Act, Xroute, Logits, Ids, Wts, Off, Slot, Xmoe, MoePart, MoeOut, MoeN, Combined,
                       tile, T.get_thread_binding())

    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.prefetch": False,
                                           "tl.intel_npu.hw_sync_grid": False}):
        source = tilelang.lower(layer_kernel, target="intel_npu").kernel_source

    if "--no-build" in argv:
        path, sha = runner.write_kernel(source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(source, KERNEL)
        print(f"build {time.time() - t0:.0f} s")

    zeros16 = lambda shape: np.zeros(shape, np.float16)  # noqa: E731
    zeros32 = lambda shape: np.zeros(shape, np.float32)  # noqa: E731
    zerosi = lambda shape: np.zeros(shape, np.int32)  # noqa: E731
    arrays = [
        ("inout", h.copy()),
        ("in", norms["input_norm"].astype(np.float16).reshape(1, H)),
        ("in", q_norm.astype(np.float16).reshape(1, D)), ("in", k_norm.astype(np.float16).reshape(1, D)),
        ("in", cos_sin),
        ("in", norms["post_attn_norm"].astype(np.float16).reshape(1, H)),
        ("in", norms["pre_ffn_norm"].astype(np.float16).reshape(1, H)),
        ("in", norms["post_ffn_1_norm"].astype(np.float16).reshape(1, H)),
        ("in", norms["pre_ffn_2_norm"].astype(np.float16).reshape(1, H)),
        ("in", norms["post_ffn_2_norm"].astype(np.float16).reshape(1, H)),
        ("in", norms["post_ffn_norm"].astype(np.float16).reshape(1, H)),
        ("in", norms["router_scale"].astype(np.float16).reshape(1, H)),
        ("in", _pack(wq_int)), ("in", wq_s), ("in", _pack(wo_int)), ("in", wo_s),
        ("in", _pack(wg_int)), ("in", wg_s), ("in", _pack(wu_int)), ("in", wu_s),
        ("in", _pack(wd_int)), ("in", wd_s),
        ("in", wr),
        ("in", _pack(ewg_int)), ("in", ewg_s), ("in", _pack(ewu_int)), ("in", ewu_s),
        ("in", _pack(ewd_int)), ("in", ewd_s),
        ("in", per_expert_scale.reshape(E, 1)), ("in", np.array([[layer_scalar]], np.float32)),
        ("in", zeros16((NKV, SMAX, D))), ("in", zeros16((NKV, D, SMAX))),
        ("in", np.array([[0]], np.int32)), ("in", np.array([[N]], np.int32)),
        ("in", zeros16((N, H))), ("in", zeros16((N, NQKV))), ("in", zeros16((NKV, N, GQ, D))),
        ("in", zeros16((N, KO))), ("in", zeros16((N, H))), ("in", zeros16((N, H))),
        ("in", zeros16((N, H))), ("in", zeros16((N, H))), ("in", zeros16((N, H))),
        ("in", zeros16((N, FI))), ("in", zeros16((N, FI))), ("in", zeros16((N, FI))),
        ("in", zeros16((N, H))), ("in", zeros16((N, E))),
        ("in", zerosi((N, TOPK))), ("in", zeros32((N, TOPK))),
        ("in", zerosi((E + 1,))), ("in", zerosi((N * TOPK,))), ("in", zeros16((N, H))),
        ("in", zeros32((TILES * SHAVES, N, H))), ("in", zeros32((N, H))),
        ("in", zeros16((N, H))), ("in", zeros16((N, H))),
    ]
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_gemma4_layer")
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)

    ref = gemma4_decoder_layer_reference(h, weights_ref, NH, NKV, D, full, None)
    out = arrays[0][1]
    rel = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6))
    cos = float(np.sum(out * ref) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-9))
    verdict = "PASS" if rel < 1.2e-1 else "FAIL"
    print(f"device (REAL NVL/NPU6): median {res.median_ms:.3f} ms [UNPROFILED host-wall incl. submit]; "
         f"rel_err={rel:.4f} cosine={cos:.6f} -> {verdict}; kernel sha256 {db.source_sha256[:12]}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
