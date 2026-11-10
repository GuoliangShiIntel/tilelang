"""P5 device golden: a whole PREFILL chunk through L layers in ONE launch (`kernels.make_prefill_model`) at
Qwen3.8's shape -- a device-side layer loop, `Kind[li, 0]` picking the Gated DeltaNet or the full-attention
decoder layer, every layer's weights reached through a table of device addresses, the KV caches carried as
kernel parameters and indexed by `Kind[li, 1]`.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_prefill_model_device.py [--kinds gdn,attn] [--n 192] [--pos0 0]
                                                                       [--trials 3] [--bn 32] [--no-build]
                                                                       [--negative]
Unprofiled (a timing run, TURBO). Correctness is `--trials` independent runs against the layer oracles chained.
`--negative` zeroes every layer's output projection and every expert's down projection: the stream must then come
back bit-identical, which a kernel that wrote over the residual instead of adding to it would fail.

Start at two layers. Plan question 8 cost a day of bisection on the decode model, and `--layer-ids 3,7` was the
cheap canary that found it; the same applies here, so `--kinds` takes any mix.
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
import tvm  # noqa: E402
import runner  # noqa: E402
import lowering  # noqa: E402
from kernels import (ATTN_PREFILL_TABLE, GDN_PREFILL_TABLE, PREFILL_TABLE_COLS, gm_scales,  # noqa: E402
                     make_moe_route_prefill, make_prefill_model, pack_int4, prefill_model_reference, rope_cos_sin)

TILES, SHAVES = 3, 2
H, KH, VH, DK, DV, G = 2048, 16, 32, 128, 128, 128
NH, NKV, D, ROT = 16, 2, 256, 64
I, E, TOPK = 512, 256, 8
TB, C, TBq, TBF, BKV, MB = 32, 64, 8, 16, 128, 16
BM, BNR, MBe, BNG, BND = 64, 16, 16, 64, 64
DB = 4
GQ, KO = NH // NKV, NH * D
QK, VC = KH * DK, VH * DV
CONV = 2 * QK + VC
NIN = CONV + VC
NQKV = NH * 2 * D + 2 * NKV * D
KGH, KGI = H // G, I // G
THETA = 1e6
REL = 1e-1   # several layers of int4 projections, a chunked scan, a flash phase and 9 expert FFNs each


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def layer_weights(kind: str, kv: int, seed: int) -> dict:
    """One layer's weights. Generated per layer and uploaded straight away: at this shape the expert weights are
    ~0.8 GB of int8 per layer before packing."""
    rng = np.random.default_rng(seed)
    p = dict(kind=kind, kv=kv,
             in_norm=(rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16),
             post_norm=(rng.standard_normal((1, H), np.float32) * 0.1).astype(np.float16),
             wr=(rng.standard_normal((E + 1, H), np.float32) * 0.1).astype(np.float16),
             gu_int=rng.integers(-8, 8, (E + 1, 2 * I, H), dtype=np.int8),
             gu_s=(rng.standard_normal((E + 1, 2 * I, KGH), np.float32) * 0.02).astype(np.float16),
             d_int=rng.integers(-8, 8, (E + 1, H, I), dtype=np.int8),
             d_s=(rng.standard_normal((E + 1, H, KGI), np.float32) * 0.02).astype(np.float16))
    if kind == "gdn":
        NIN_pad = p["NIN_pad"] = -(-NIN // BN1) * BN1
        p.update(win_int=rng.integers(-8, 8, (NIN_pad, H), dtype=np.int8),
                 win_s=(rng.standard_normal((NIN_pad, H // G), np.float32) * 0.02).astype(np.float16),
                 wba=(rng.standard_normal((2 * VH, H), np.float32) * 0.05).astype(np.float16),
                 conv_w=(rng.standard_normal((4, CONV), np.float32) * 0.3).astype(np.float16),
                 conv_state=(rng.standard_normal((3, CONV), np.float32) * 0.3).astype(np.float16),
                 a_log=(rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32),
                 dt_bias=(rng.standard_normal((1, VH), np.float32) * 0.2).astype(np.float32),
                 gnorm=(1.0 + rng.standard_normal((1, DV), np.float32) * 0.1).astype(np.float16),
                 wout_int=rng.integers(-8, 8, (H, VC), dtype=np.int8),
                 wout_s=(rng.standard_normal((H, VC // G), np.float32) * 0.02).astype(np.float16),
                 ssm=(rng.standard_normal((VH, DK, DV), np.float32) * 0.05).astype(np.float32))
    else:
        NQKV_pad = p["NQKV_pad"] = -(-NQKV // BN1) * BN1
        p.update(wqkv_int=rng.integers(-8, 8, (NQKV_pad, H), dtype=np.int8),
                 wqkv_s=(rng.standard_normal((NQKV_pad, H // G), np.float32) * 0.02).astype(np.float16),
                 qnorm=(0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16),
                 knorm=(0.1 * rng.standard_normal((1, D), np.float32)).astype(np.float16),
                 wo_int=rng.integers(-8, 8, (H, KO), dtype=np.int8),
                 wo_s=(rng.standard_normal((H, KO // G), np.float32) * 0.02).astype(np.float16))
    return p


def table_arrays(p: dict, RR: int, zero_out: bool = False) -> dict:
    """The layer's table entries by column name, in the kernel's layouts. `zero_out` is the negative control:
    the output projection and every expert's down projection become zero, so the layer must contribute nothing."""
    wr_pad = np.zeros((RR, H), np.float16)
    wr_pad[:E + 1] = p["wr"]
    d_int = np.zeros_like(p["d_int"]) if zero_out else p["d_int"]
    d_s = np.zeros_like(p["d_s"]) if zero_out else p["d_s"]
    a = {"InNorm": p["in_norm"], "PostNorm": p["post_norm"], "Wr": wr_pad,
         "WguQ": pack_int4(p["gu_int"].reshape(-1, H)), "WguS": gm_scales(p["gu_s"].reshape(-1, KGH)),
         "WdQ": pack_int4(d_int.reshape(-1, I)), "WdS": gm_scales(d_s.reshape(-1, KGI))}
    if p["kind"] == "gdn":
        wout = np.zeros_like(p["wout_int"]) if zero_out else p["wout_int"]
        wos = np.zeros_like(p["wout_s"]) if zero_out else p["wout_s"]
        a.update({"W0": pack_int4(p["win_int"]), "W1": gm_scales(p["win_s"]), "W2": p["conv_w"], "W3": p["a_log"],
                  "W4": p["dt_bias"], "W5": p["gnorm"], "W6": pack_int4(wout), "W7": gm_scales(wos),
                  "W8": p["wba"],
                  "S0": p["conv_state"], "S1": p["ssm"]})
    else:
        wo = np.zeros_like(p["wo_int"]) if zero_out else p["wo_int"]
        wos = np.zeros_like(p["wo_s"]) if zero_out else p["wo_s"]
        a.update({"W0": pack_int4(p["wqkv_int"]), "W1": gm_scales(p["wqkv_s"]), "W2": p["qnorm"], "W3": p["knorm"],
                  "W4": pack_int4(wo), "W5": gm_scales(wos)})
    return {k: np.ascontiguousarray(v) for k, v in a.items()}


BN1 = BN2 = 32


def main(argv):
    global BN1, BN2, TILES
    kinds = (argv[argv.index("--kinds") + 1] if "--kinds" in argv else "gdn,attn").split(",")
    TILES = arg(argv, "--tiles", TILES)
    n, pos0 = arg(argv, "--n", 192), arg(argv, "--pos0", 0)
    trials = arg(argv, "--trials", 3)
    BN1 = arg(argv, "--bn1", arg(argv, "--bn", 32))
    BN2 = arg(argv, "--bn2", arg(argv, "--bn", 32))
    global MBe, DB, MB
    MBe = arg(argv, "--mbe", MBe)
    DB = arg(argv, "--db", DB)
    MB = arg(argv, "--mb", MB)       # the projections' token block: sets their DPU workload's M
    SMAX = arg(argv, "--smax", 1536)
    skip = frozenset(x for x in (argv[argv.index("--skip") + 1].split(",") if "--skip" in argv else []) if x)
    L = len(kinds)
    KVL = max(1, sum(1 for k in kinds if k == "attn"))
    step = int(np.lcm.reduce([TB, C, TBq, TBF, TILES * SHAVES, BM]))
    NP = -(-n // step) * step
    assert pos0 + NP <= SMAX, "the chunk must fit the cache window"
    _, rs = make_moe_route_prefill(N=NP, H=H, E=E, TOPK=TOPK, BM=BM, BNR=BNR, tiles=TILES, shaves=SHAVES,
                                   as_block=True)
    RR, SLOTS = rs["RR"], rs["SLOTS"]
    NIN_pad = -(-NIN // BN1) * BN1
    NQKV_pad = -(-NQKV // BN1) * BN1
    YW, OW = max(NIN_pad, NQKV_pad), max(VC, KO)

    cfg = {"tl.intel_npu.scoped_cmx": "--no-scoped-cmx" not in argv,
            "tl.intel_npu.batch_dma": "--batch-dma" in argv,
            "tl.intel_npu.dpu_overlap": "--dpu-overlap" in argv,
            "tl.intel_npu.prefetch": "--no-prefetch" not in argv,
            "tl.intel_npu.multicast": "--shared-projections" in argv,
            "tl.intel_npu.multicast_overlap": "--overlap" in argv}
    t_low = time.time()
    with tvm.transform.PassContext(config=cfg):
        low = lowering.lower(make_prefill_model(L=L, N=NP, H=H, KH=KH, VH=VH, DK=DK, DV=DV, NH=NH, NKV=NKV, D=D,
                                                ROT=ROT, SMAX=SMAX, I=I, E=E, TOPK=TOPK, G=G, TB=TB, C=C, TBq=TBq,
                                                TBF=TBF, BKV=BKV, BN1=BN1, BN2=BN2, MB=MB, BM=BM, BNR=BNR,
                                                MBe=MBe, BNG=BNG, BND=BND, DB=DB, tiles=TILES, shaves=SHAVES, KVL=KVL,
                                                skip=skip, unroll_layers="--unroll" in argv,
                                                conv_blocked="--roundrobin" not in argv,
                                                conv_item_worker="shave" if "--shave-major" in argv else "tile",
                                                shared_projections="--shared-projections" in argv,
                                                moe_slot_output="--moe-slots" in argv,
                                                separate_shared_expert="--shared-expert" in argv,
                                                moe_combine_rows=arg(argv, "--combine-rows", 1),
                                                register_conv="--register-conv" in argv,
                                                conv_fp16="--conv-fp16" in argv, register_tri="--register-tri" in argv,
                                                gdn_live_chunks="--gdn-live" in argv,
                                                moe_reuse_input="--moe-reuse-input" in argv,
                                                output_block_k=arg(argv, "--out-block-k", None),
                                                output_mb=arg(argv, "--out-mb", None)))
    print(f"lower {time.time() - t_low:.0f} s")
    name = (f"tl_prefill_model_l{L}_n{NP}_h{H}_s{SMAX}_bn{BN1}"
            + (f"_bn2{BN2}" if BN2 != BN1 else "")
            + (f"_mbe{MBe}" if MBe != 16 else "")
            + (f"_mb{MB}" if MB != 16 else "")
            + ("_noprefetch" if "--no-prefetch" in argv else "")
            + ("_sharedproj" if "--shared-projections" in argv else "")
            + ("_overlap" if "--overlap" in argv else "")
            + ("_moeslots" if "--moe-slots" in argv else "")
            + ("_sharedexpert" if "--shared-expert" in argv else "")
            + ("_dpuov" if "--dpu-overlap" in argv else "")
            + (f"_db{DB}" if DB != 4 else "")
            + (f"_t{TILES}" if TILES != 3 else "")
            + ("_sm" if "--shave-major" in argv else "")
            + ("_rr" if "--roundrobin" in argv else "")
            + ("_unroll" if "--unroll" in argv else "")
            + ("_frt" if "--frt" in argv else "")
            + ("_skip_" + "_".join(sorted(skip)) if skip else "")
            + ("_noscope" if "--no-scoped-cmx" in argv else ""))
    source = low.source
    if "--register-conv" in argv:
        name = name.replace("_sharedproj_overlap_moeslots_sharedexpert", "_spov_ms_se") + "_rcv"
    if "--conv-fp16" in argv:
        name = name.replace("_sharedproj_overlap_moeslots_sharedexpert", "_spov_ms_se") + "_cv16"
    if "--combine-rows" in argv or "--batch-dma" in argv:
        name = f"tl_pf_l{L}_n{NP}_bn{BN1}_{BN2}_mb{MB}_me{MBe}_cb{arg(argv, '--combine-rows', 1)}"
        name += "_batch" if "--batch-dma" in argv else "_sync"
        name += "_rcv" if "--register-conv" in argv else ""
        name += "_cv16" if "--conv-fp16" in argv else ""
        assert "--shared-projections" in argv and "--overlap" in argv and "--moe-slots" in argv and "--shared-expert" in argv
    if "--dpu-overlap" in argv:
        name = name.replace("_sharedproj_overlap_moeslots_sharedexpert", "_spov_ms_se")
    if "--register-tri" in argv:
        name = name.replace("_sharedproj_overlap_moeslots_sharedexpert", "_spov_ms_se") + "_rtri"
    if "--gdn-live" in argv:
        name = name.replace("tl_prefill_model", "tl_pf") + "_live"
    if "--moe-reuse-input" in argv:
        name = name.replace("tl_prefill_model", "tl_pf") + "_reuse"
    if "--out-block-k" in argv:
        name = name.replace("tl_prefill_model", "tl_pf") + f"_ok{arg(argv, '--out-block-k', 0)}_om{arg(argv, '--out-mb', MB)}"
    if "--phases" in argv:
        # instrumentation is opt-in: it costs ~5% (npu_rt_device.hpp's TL_NPU_PHASE_PROF), so the default
        # build is the one every latency number comes from
        source = "#define TL_NPU_PHASE_PROF 1\n" + source
        name += "_prof"
        low = type(low)(source=source, param_names=low.param_names)
    if "--frt" in argv:
        # the SAME generated kernel on npu_dsl's fused_rt: both arms share this lowering, the generated C++,
        # the toolkit and the hardware, so a difference can only be the runtime
        source = "#define TL_NPU_RUNTIME_FRT 1\n" + source
        low = type(low)(source=source, param_names=low.param_names)
    if "--no-build" in argv:
        runner.write_kernel(low.source, name)
    else:
        t0 = time.time()
        runner.build_all({name: low.source})
        print(f"build {time.time() - t0:.0f} s")

    exe = HERE / "build" / "Release" / "npu_run.exe"
    s = runner.Session(exe, HERE / "run_prefill_model", TILES, SHAVES)
    profiled = "--profile" in argv      # every launch of the alias, so this run is observability, not a timing
    s.kernel("model", name, profile=profiled)
    print(f"session up, TURBO {s.turbo}")

    # ---- weights: one layer at a time, uploaded and then kept only for the oracle -------------------------
    layers, kv = [], 0
    uploaded = {}      # name -> the array as uploaded, for --audit
    t_gen = t_up = 0.0
    up_bytes = 0
    for li, kind in enumerate(kinds):
        t0 = time.time()
        p = layer_weights(kind, kv if kind == "attn" else 0, seed=100 + li)
        arrs = table_arrays(p, RR)
        t_gen += time.time() - t0
        kv += 1 if kind == "attn" else 0
        t0 = time.time()
        for cname, arr in arrs.items():
            s.upload(f"L{li}_{cname}", arr)
            up_bytes += arr.nbytes
            if arr.nbytes < (1 << 24):      # the expert weights are ~0.4 GB a layer: not worth reading back
                uploaded[f"L{li}_{cname}"] = arr
        t_up += time.time() - t0
        layers.append(p)
    print(f"weights: generate+pack {t_gen:.0f} s, upload {t_up:.0f} s "
          f"({up_bytes / 2**30:.2f} GiB, {up_bytes / 2**20 / max(t_up, 1e-9):.0f} MiB/s), {L} layers")

    col = {c: i for i, c in enumerate(PREFILL_TABLE_COLS)}
    table = np.zeros((L, len(PREFILL_TABLE_COLS)), np.int64)
    kind_a = np.zeros((L, 4), np.int32)
    kv = 0
    for li, p in enumerate(layers):
        kind_a[li, 0] = 0 if p["kind"] == "gdn" else 1
        if p["kind"] == "attn":
            kind_a[li, 1] = kv
            kv += 1
        names = (GDN_PREFILL_TABLE if p["kind"] == "gdn" else ATTN_PREFILL_TABLE)
        for cname in list(names) + ["InNorm", "PostNorm", "Wr", "WguQ", "WguS", "WdQ", "WdS"]:
            table[li, col[cname]] = s.address(f"L{li}_{cname}")
    s.upload("Kind", kind_a)
    s.upload("Table", table)

    h0 = (np.random.default_rng(0).standard_normal((NP, H), np.float32) * 0.5).astype(np.float32)
    cs = np.stack([rope_cos_sin(pos0 + t, ROT, THETA).reshape(-1) for t in range(NP)]).astype(np.float32)
    s.upload("Pos", np.array([[pos0]], np.int32))
    s.upload("Nt", np.array([[n]], np.int32))
    s.upload("CosSin", cs)
    for nm, nbytes in (("KcAll", KVL * NKV * SMAX * D * 2), ("VtAll", KVL * NKV * D * SMAX * 2),
                       ("Xs", NP * H * 2), ("Ys", NP * YW * 2), ("BAs", NP * 2 * VH * 2),
                       ("QKs", KH * NP * 2 * DK * 4),
                       ("Vs", VH * NP * DV * 2), ("Gbs", VH * NP * 2 * 4), ("Zs", VH * NP * DV * 2),
                       ("Q", NKV * NP * GQ * D * 2), ("Gate", NP * KO * 2), ("Os", NP * OW * 2),
                       ("Mix", NP * H * 2), ("Logits", NP * RR * 2), ("Ids", NP * TOPK * 4),
                       ("Wts", NP * 16 * 4), ("Cnt", (E + 1) * 4), ("Off", (E + 2) * 4), ("Slot", SLOTS * 4),
                       ("PartE", SLOTS * H * 2 if "--moe-slots" in argv else TILES * NP * H * 4),
                       ("MoeOut", NP * H * 4), ("Hio", NP * H * 4),
                       ("SharedGu", NP * 2 * I * 2), ("SharedAct", NP * I * 2)):
        s.alloc(nm, nbytes)
    bufs = [n for n in low.param_names]

    def run_once():
        """One launch from a clean state: the stream, the caches and the GDN states are restored first."""
        s.upload("Hio", h0)
        s.zero("KcAll")
        s.zero("VtAll")
        for nm in ("QKs", "Vs", "Gbs", "Zs", "Xs", "Ys", "BAs", "Os", "Mix"):
            s.zero(nm)      # a Session buffer persists across launches: stale scratch reads as written output
        for li, p in enumerate(layers):
            if p["kind"] == "gdn":
                s.upload(f"L{li}_S0", np.ascontiguousarray(p["conv_state"]))
                s.upload(f"L{li}_S1", np.ascontiguousarray(p["ssm"]))
        host_ms, kernel_ms = s.launch("model", *bufs)
        hio = s.download("Hio", np.float32, (NP, H))
        kc = s.download("KcAll", np.float16, (KVL, NKV, SMAX, D))
        # the conv phase writes q into QKs[:, :, :DK] and k into [:, :, DK:]; an untouched half says the
        # part == 0 iteration never ran, which is a different defect from a lost store
        qk = s.download("QKs", np.float32, (KH, NP, 2 * DK))
        print(f"  QKs q half: absmax {np.abs(qk[:, :, :DK]).max():.3e}, exact zeros "
              f"{int((qk[:, :, :DK] == 0).sum())} of {qk[:, :, :DK].size}; "
              f"k half: absmax {np.abs(qk[:, :, DK:]).max():.3e}, exact zeros "
              f"{int((qk[:, :, DK:] == 0).sum())} of {qk[:, :, DK:].size}")
        states = {li: (s.download(f"L{li}_S0", np.float16, (3, CONV)),
                       s.download(f"L{li}_S1", np.float32, (VH, DK, DV)))
                  for li, p in enumerate(layers) if p["kind"] == "gdn"}
        if "--qks" in argv:   # did the conv phase produce its q half at all? (scratch was zeroed above)
            qk = s.download("QKs", np.float32, (KH, NP, 2 * DK))
            for nm, half in (("q", qk[:, :, :DK]), ("k", qk[:, :, DK:])):
                print(f"  QKs {nm}: absmax {np.abs(half).max():.3e}, exact zeros "
                      f"{100.0 * (half == 0).mean():.1f}%")
            if "--save-qks" in argv:
                np.save(argv[argv.index("--save-qks") + 1], qk)
            if "--save-ys" in argv:
                np.save(argv[argv.index("--save-ys") + 1], s.download("Ys", np.float16, (NP, YW)))
        return hio, kc, states, kernel_ms

    t0 = time.time()
    # the oracle is a pure function of the seeded weights and the fixed inputs, so a geometry sweep (MBe, DB,
    # BNG ...) recomputes the same 20 s of numpy for every arm; the key carries the mtime of the two files that
    # define it, so editing either one misses rather than returning a stale reference
    import hashlib as _hl
    _key = _hl.sha256(repr((kinds, NP, pos0, n, SMAX, KVL, H, E, TOPK, G, NH, NKV, D, ROT, KH, VH, DK, DV,
                            Path(__file__).stat().st_mtime_ns,
                            (HERE.parent / "kernels.py").stat().st_mtime_ns)).encode()).hexdigest()[:16]
    _cache = HERE / "build" / f"oracle_{_key}.npz"
    ref_rows = [dict(p) for p in layers]
    if _cache.exists() and "--no-oracle-cache" not in argv:
        _z = np.load(_cache)
        ref, kc_ref = _z["ref"], [_z[f"kc{i}"] for i in range(KVL)]
        # prefill_model_reference ADVANCES conv_state/ssm in ref_rows in place and the trial loop compares
        # against them, so the cache has to carry the advanced states too, not just the stream
        for li, p in enumerate(ref_rows):
            if p["kind"] == "gdn":
                p["conv_state"], p["ssm"] = _z[f"cs{li}"], _z[f"ssm{li}"]
        print(f"oracle {time.time() - t0:.0f} s (cached {_cache.name})")
    else:
        ref, kc_ref, _ = prefill_model_reference(ref_rows, h0, cs, pos0, n, NH=NH, NKV=NKV, D=D, ROT=ROT, KH=KH,
                                                 VH=VH, DK=DK, DV=DV, E=E, TOPK=TOPK, G=G,
                                                 kc=[np.zeros((NKV, SMAX, D), np.float16) for _ in range(KVL)],
                                                 vt=[np.zeros((NKV, D, SMAX), np.float16) for _ in range(KVL)])
        _st = {f"cs{li}": p["conv_state"] for li, p in enumerate(ref_rows) if p["kind"] == "gdn"}
        _st |= {f"ssm{li}": p["ssm"] for li, p in enumerate(ref_rows) if p["kind"] == "gdn"}
        _cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(_cache, ref=ref, **{f"kc{i}": a for i, a in enumerate(kc_ref)}, **_st)
        print(f"oracle {time.time() - t0:.0f} s (cached to {_cache.name})")

    good, worst, kms = 0, 0.0, []
    for trial in range(trials):
        hio, kc, states, km = run_once()
        kms.append(km)
        named = [("stream", float(np.abs(hio - ref).max() / np.abs(ref).max()))]
        for k in range(KVL):
            named.append((f"Kc[{k}]",
                          float(np.abs(kc[k].astype(np.float32) - kc_ref[k].astype(np.float32)).max()
                                / max(float(np.abs(kc_ref[k].astype(np.float32)).max()), 1e-6))))
        for li, (cs_got, ssm_got) in states.items():
            exp_cs = ref_rows[li]["conv_state"].astype(np.float32)
            got_cs = cs_got.astype(np.float32)
            named.append((f"L{li}.conv_state",
                          float(np.abs(got_cs - exp_cs).max() / max(float(np.abs(exp_cs).max()), 1e-6))))
            if trial == 0:   # the conv state is three channel groups: q, k, v -- a wrong one names the slice
                for gname, lo, hi in (("q", 0, QK), ("k", QK, 2 * QK), ("v", 2 * QK, CONV)):
                    sub = np.abs(got_cs[:, lo:hi] - exp_cs[:, lo:hi]).max() / max(float(np.abs(exp_cs[:, lo:hi]).max()), 1e-6)
                    named.append((f"  L{li}.cs[{gname}]", float(sub)))
            named.append((f"L{li}.ssm",
                          float(np.abs(ssm_got - ref_rows[li]["ssm"]).max()
                                / max(float(np.abs(ref_rows[li]["ssm"]).max()), 1e-6))))
        if trial == 0:
            print("  " + "  ".join(f"{k} {v:.2e}" for k, v in named))
        worst = max(worst, max(v for _, v in named))
        good += int(max(v for _, v in named) < REL)
    km = float(np.median(kms))
    verdict = "PASS" if good == trials else "FAIL"
    condition = ("INSTRUMENTED (observability, NOT a timing)" if "--phases" in argv else
                 "PROFILED (observability, NOT a timing)" if profiled else "UNPROFILED")
    print(f"prefill model: {km:.3f} ms [{'TURBO' if s.turbo else 'NO TURBO'}, "
            f"{condition}] -> {verdict}; "
          f"{good}/{trials} trials match the layer oracles chained (stream, KV caches, both GDN states), "
          f"worst rel {worst:.2e}")
    print(f"  {'/'.join(kinds)}: {L} layers, {n} tokens at pos0 {pos0}, H={H}, E={E} top-{TOPK}: "
          f"{1e3 * km / n:.1f} us per token, {1e3 * km / n / L:.1f} us per token per layer")
    print("  kernel-only per run ms: " + " ".join(f"{t:.3f}" for t in kms))
    rc = 0 if verdict == "PASS" else 1

    if profiled:
        trace = HERE / "evidence" / f"{name}_trace.json"
        s.profile("model", trace)
        print(f"  PROFILED trace: {trace}")
        print(f"  python {Path(__file__).with_name('perf_attr.py')} {trace}")

    if "--phases" in argv:
        f = s.phases()
        sys.path.insert(0, str(HERE))
        import task_report
        per, dropped = task_report.load(f, TILES * SHAVES)
        task_report.table(per, dropped)
        print(task_report.chrome_json(per, dropped, HERE / "evidence" / f"{name}_tasks.json"))

    if "--audit" in argv:
        # Which HEADS of the conv phase are wrong? The q half, the k half and the state slice of head hb are
        # written by the same two iterations, so a per-head map says whether an iteration ran at all.
        li = next((i for i, p in enumerate(layers) if p["kind"] == "gdn"), None)
        if li is not None:
            from kernels import gdn_conv_prefill_reference, proj_prefill_reference
            p = layers[li]
            hf = h0.astype(np.float32)
            x = hf / np.sqrt((hf * hf).mean(-1, keepdims=True) + 1e-6) * (1.0 + p["in_norm"].astype(np.float32).reshape(1, -1))
            y = proj_prefill_reference(x.astype(np.float16), p["win_int"], p["win_s"], G).astype(np.float16)
            ba = x.astype(np.float16).astype(np.float32) @ p["wba"].astype(np.float32).T
            qk_ref, _, _, _, cs_ref = gdn_conv_prefill_reference(y, p["conv_w"], p["conv_state"], p["a_log"],
                                                                 p["dt_bias"], n, KH, VH, DK, DV, ba=ba)
            qk = s.download("QKs", np.float32, (KH, NP, 2 * DK))
            cs = s.download(f"L{li}_S0", np.float16, (3, CONV)).astype(np.float32)
            # upstream first: if the norm or the projection is wrong, every head is wrong for free
            xs = s.download("Xs", np.float16, (NP, H)).astype(np.float32)
            ys = s.download("Ys", np.float16, (NP, YW)).astype(np.float32)
            print(f"  upstream: Xs rel {np.abs(xs - x).max() / np.abs(x).max():.2e}, "
                  f"Ys[:, :NIN] rel {np.abs(ys[:, :NIN] - y.astype(np.float32)[:, :NIN]).max() / np.abs(y.astype(np.float32)[:, :NIN]).max():.2e} "
                  f"(Ys is shared scratch: a later layer overwrites it, so this is only meaningful for the LAST layer)")
            print(f"  per-head map of layer {li} (q | k | state q | state k), rel vs the conv oracle:")
            for hb in range(KH):
                worker = hb % (TILES * SHAVES)
                def rel(a, b):
                    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-6))
                rq = rel(qk[hb, :, :DK], qk_ref[hb, :, 0])
                rk = rel(qk[hb, :, DK:], qk_ref[hb, :, 1])
                sq = rel(cs[:, hb * DK:(hb + 1) * DK], cs_ref[:, hb * DK:(hb + 1) * DK].astype(np.float32))
                sk = rel(cs[:, QK + hb * DK:QK + (hb + 1) * DK], cs_ref[:, QK + hb * DK:QK + (hb + 1) * DK].astype(np.float32))
                print(f"    head {hb:2d} (worker {worker}): q {rq:8.2e}  k {rk:8.2e} | state q {sq:8.2e}  state k {sk:8.2e}")

    if "--buffers" in argv:
        # which uploaded buffers did the launch change? S0 / S1 are the states and are MEANT to change; anything
        # else means a store landed outside its own buffer, and the set of victims gives its reach
        print("audit: buffers that changed across the launch (S0 / S1 are the states and must)")
        for nm, arr in uploaded.items():
            got = s.download(nm, arr.dtype, arr.shape)
            if not np.array_equal(got, arr):
                d = float(np.abs(got.astype(np.float64) - arr.astype(np.float64)).max())
                nz = int((got != arr).sum())
                print(f"  {nm:16s} changed: {nz} of {arr.size} elements, max |delta| {d:.3e}")
        # and WHAT is in the wrong channels: the content names the writer, where the address arithmetic cannot
        for li, p in enumerate(layers):
            if p["kind"] != "gdn":
                continue
            got = s.download(f"L{li}_S0", np.float16, (3, CONV)).astype(np.float32)
            exp = ref_rows[li]["conv_state"].astype(np.float32)
            print(f"  L{li}_S0 row 0, first 6 of q: got {np.round(got[0, :6], 4)}")
            print(f"  {' ' * len(str(li))}        expected {np.round(exp[0, :6], 4)}")
            print(f"  L{li}_S0 row 0, first 6 of k: got {np.round(got[0, QK:QK + 6], 4)}")
            print(f"  L{li}_S0 row 0, first 6 of v: got {np.round(got[0, 2 * QK:2 * QK + 6], 4)}")
            cand = {"the k slice": exp[:, QK:2 * QK], "the v slice": exp[:, 2 * QK:2 * QK + QK],
                    "zeros": np.zeros_like(exp[:, :QK]), "the state as uploaded": p["conv_state"][:, :QK].astype(np.float32)}
            for nm, c in cand.items():
                if c.shape == got[:, :QK].shape:
                    print(f"    q == {nm}? max |delta| {float(np.abs(got[:, :QK] - c).max()):.3e}")

    if "--negative" in argv:
        for li, p in enumerate(layers):
            for cname, arr in table_arrays(p, RR, zero_out=True).items():
                s.upload(f"L{li}_{cname}", arr)
        hio, _, _, _ = run_once()
        d = float(np.abs(hio - h0).max())
        print(f"negative (every output projection and expert down zeroed): {'behaved' if d == 0 else 'BROKEN'} "
              f"-- max |stream - input| {d:.2e}; every layer must ADD to the residual, not replace it")
        rc |= 0 if d == 0 else 1
    s.close()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
