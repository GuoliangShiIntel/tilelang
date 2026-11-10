"""Decode a Qwen3.5-MoE-family model on the NPU with this backend's kernels. Default: ONE launch per token
(`kernels.make_decode_model`: a device-side loop over the layers, each layer's weights and states reached through a
table of device addresses, the fp32 residual stream in CMX for the whole step, final norm + lm_head + argmax at the end).
`--per-layer`: one launch per fused layer (`make_gdn_layer` / `make_attn_layer`) and one for the head (the A/B baseline).
Weights and states are resident in one `npu_run --session`; the host uploads the embedding row, launches, and reads the
token back.

`--prefill [N]`: the prompt goes through `kernels.make_prefill_model` in chunks of N tokens (one launch a chunk, all
40 layers) instead of token by token through the decode kernels. Both paths read the SAME weight tensors -- packed
values plus group-major scales -- and write the same KV cache, conv state and SSM state, so decode continues from
where prefill stopped; that shared layout is what makes one 17.91 GB upload serve both. Without the flag the prompt
is replayed through decode, which is correct but is not prefill and must not be quoted as TTFT.

    python testing/python/intel_npu/e2e/decode.py --ckpt D:\\models\\... --pack-dir D:\\models\\..._tlpack
        [--prefill 192] [--layers N] [--prompt "text" | --ids 1,2,3] [--new 16] [--smax 1536] [--golden golden.npz]
        [--no-build] [--per-layer]

--golden FILE: the fp32 reference's saved answer for the same prompt (`e2e/reference_golden.py`, run separately: the
reference and 18 GB of pinned weights do not fit one 32 GB box together); the hidden state after every layer of the
first token (per-layer mode only) and the argmax of every step are compared, teacher-forced (the device is fed the
reference's tokens). Timing is UNPROFILED, TURBO unless TL_NPU_TURBO=0; the per-token number is the kernel-only span of
the launch(es) plus the host wall of the whole step (file transfers included).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "device"))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401
import lowering  # noqa: E402
import runner  # noqa: E402
from kernels import (ATTN_PREFILL_TABLE, ATTN_TABLE, GDN_PREFILL_TABLE, GDN_TABLE, PREFILL_TABLE_COLS,  # noqa: E402
                     TABLE_COLS, attn_sizes, gdn_sizes, head_sizes, make_attn_layer, make_decode_model,
                     make_gdn_layer, make_head, make_prefill_model, rope_cos_sin)
import tilelang.intel_npu.language as T  # noqa: E402
from checkpoint import Checkpoint, LM  # noqa: E402
import pack  # noqa: E402

TILES, SHAVES = 3, 2

# `make_prefill_model`'s block geometry. Only the token count N reaches a buffer SHAPE; the rest are scheduling,
# and none of them reach the packed weights any more (pack.PAD_KEYS), so a sweep here costs a rebuild, not a repack.
PREFILL_GEOM = dict(TB=32, C=64, TBq=8, TBF=16, BKV=128, BN1=32, BN2=32, MB=16, BM=64, BNR=16, MBe=16,
                    BNG=64, BND=64)


def prefill_sizes(c, n: int, G: dict) -> dict:
    """The padded token count and every scratch extent `make_prefill_model` declares, derived the same way the
    kernel derives them -- a mismatch here is a DDR read past a buffer, which hangs the device rather than
    failing."""
    import math
    workers = TILES * SHAVES
    step = 1
    for k in (G["TB"], G["C"], G["TBq"], G["TBF"], workers, G["BM"]):
        step = step * k // math.gcd(step, k)
    NP = -(-n // step) * step
    QK, VC = c.lin_kh * c.lin_dk, c.lin_vh * c.lin_dv
    CONV = 2 * QK + VC
    NIN = CONV + VC
    NIN_pad = -(-NIN // G["BN1"]) * G["BN1"]
    NQKV = c.n_heads * 2 * c.head_dim + 2 * c.n_kv * c.head_dim
    NQKV_pad = -(-NQKV // G["BN1"]) * G["BN1"]
    KO = c.n_heads * c.head_dim
    per_round = workers * G["BNR"]
    RR = -(-(c.n_experts + 1) // per_round) * per_round
    return dict(NP=NP, YW=max(NIN_pad, NQKV_pad), OW=max(VC, KO), KO=KO, RR=RR, NIN=NIN, NQKV=NQKV,
                SLOTS=NP * (c.top_k + 1), GQ=c.n_heads // c.n_kv, VC=VC, CONV=CONV)


class DeviceModel:
    def __init__(self, ck: Checkpoint, pack_dir: Path, layers: int, smax: int, build: bool, single: bool = True, log=print,
                 layer_ids: list[int] | None = None, chunked: bool = False, prefill: int = 0,
                 prefill_ls: list[int] | None = None, prefill_shared: bool = False, prefill_overlap: bool = False,
                 prefill_bn1: int | None = None, prefill_mb: int | None = None, prefill_mbe: int | None = None,
                 prefill_moe_slots: bool = False):
        self.ck, self.c, self.log = ck, ck.cfg, log
        c = self.c
        self.layer_ids = list(layer_ids) if layer_ids is not None else list(range(layers))   # checkpoint layers, in run order
        layers = len(self.layer_ids)
        self.L = layers
        self.smax = smax
        self.single = single
        self.prefill_n = prefill
        self.prefill_moe_slots = prefill_moe_slots
        prefill_shared = prefill_shared or prefill_overlap
        pgeom = dict(PREFILL_GEOM)
        if prefill_bn1 is not None:
            pgeom["BN1"] = prefill_bn1
        if prefill_mb is not None:
            pgeom["MB"] = prefill_mb
        if prefill_mbe is not None:
            pgeom["MBe"] = prefill_mbe
        # launch units: the whole model, or (chunked) runs of up to 3 GDN layers and single attention layers, the head in the last
        self.chunks = self._chunks() if (single and chunked) else [self.layer_ids]
        G = pack.GEOM
        moe_kw = dict(BNR=G["moe_BNR"], BNG=G["moe_BNG"], BND=G["moe_BND"])
        if os.environ.get("TL_NPU_MOE_NSG"):   # gate/up ring depth (slots), independent of the down ring's
            moe_kw["num_stages_g"] = int(os.environ["TL_NPU_MOE_NSG"])
        if os.environ.get("TL_NPU_MOE_DEFER"):   # defer the gate/up reductions (64 KB more CMX per SHAVE)
            moe_kw["defer_gu"] = True
        if single:
            base_skip = frozenset(x for x in os.environ.get("TL_NPU_DBG_SKIP", "").split(",") if x)
            r2 = dict(R2=int(os.environ["TL_NPU_DBG_R2"])) if os.environ.get("TL_NPU_DBG_R2") else {}   # out_proj rows (model default 16)
            kernels, names = {}, {}
            for ci, chunk in enumerate(self.chunks):
                key = self._chunk_kernel(ci)
                if key in kernels:
                    continue
                skip = base_skip if ci == len(self.chunks) - 1 else base_skip | {"head"}
                kernels[key] = make_decode_model(L=len(chunk), KVL=self._kv_slots(ci), H=c.hidden, KH=c.lin_kh, VH=c.lin_vh, DK=c.lin_dk,
                                                 DV=c.lin_dv, NH=c.n_heads,
                                                 NKV=c.n_kv, D=c.head_dim, ROT=c.rot_dim, SMAX=smax, I=c.moe_inter, E=c.n_experts,
                                                 TOPK=c.top_k, V=c.vocab, G=ck.group, tiles=TILES, shaves=SHAVES, eps=c.eps,
                                                 pipelined=os.environ.get("TL_NPU_PIPELINED", "1") != "0",
                                                 skip=skip,
                                                 gdn_kw=dict(BN1=G["gdn_BN1"], BN2=G["gdn_BN2"], **r2),
                                                 attn_kw=dict(BN1=G["attn_BN1"], BN2=G["attn_BN2"], **r2,
                                                              pipelined=os.environ.get("TL_NPU_ATTN_SERIAL") != "1",
                                                              gemv_pipelined=os.environ.get("TL_NPU_ATTN_GEMV_SERIAL") != "1",
                                                              BKV=int(os.environ.get("TL_NPU_DBG_BKV", "128")),
                                                              num_stages=int(os.environ.get("TL_NPU_DBG_ATTN_NSTAGES", "4")),
                                                              kv_stages=int(os.environ.get("TL_NPU_DBG_KV_STAGES", "2")),
                                                              stages=os.environ.get("TL_NPU_DBG_ATTN_STAGES", "ABCDE")), moe_kw=moe_kw,
                                                 head_kw=dict(BN=G["head_BN"]))
                names[key] = f"tl_e2e_{key}_h{c.hidden}_s{smax}"
            if self.prefill_n:
                # the prompt through the PREFILL kernels instead of token by token through decode. It reads the
                # same weight tensors decode already uploaded -- that is what the shared group-major scale layout
                # bought -- and writes the same KV cache, conv state and SSM state, so decode continues from it.
                assert len(self.chunks) == 1, "--prefill needs the whole model in one launch unit (not --chunked)"
                ps = prefill_sizes(c, self.prefill_n, pgeom)
                self.ps = ps
                # a sweep builds one kernel per layer count in THIS pass: they then share the build, the 17.91 GB
                # upload and the session, so a bisect point costs a launch instead of ~100 s of fixed overhead
                self.prefill_ls = sorted(prefill_ls) if prefill_ls else [layers]
                assert max(self.prefill_ls) <= layers
                pskip = frozenset(x for x in os.environ.get("TL_NPU_PREFILL_SKIP", "").split(",") if x)
                # the out projections were PACKED at the DECODE geometry, so their rows are padded to that
                # kernel's BN2 * tiles and the scales tensor is wider than H. Prefill must read them with the
                # packer's pitch, not its own.
                hw_g = gdn_sizes(c.hidden, c.lin_kh, c.lin_vh, c.lin_dk, tiles=TILES, shaves=SHAVES,
                                 BN1=pack.GEOM["gdn_BN1"], BN2=pack.GEOM["gdn_BN2"])["NOUT_pad"]
                hw_a = attn_sizes(c.hidden, c.n_heads, c.n_kv, c.head_dim, tiles=TILES,
                                  BN2=pack.GEOM["attn_BN2"])["NOUT_pad"]
                assert hw_g == hw_a, (f"the two mixers' out projections were packed to different row counts "
                                      f"(gdn {hw_g}, attn {hw_a}); the prefill kernel carries one pitch")
                # the IN projections carry no such knob: prefill derives its own pad, so if the packer's differs
                # the group-major scales are read from the wrong column and the layer silently returns NaN.
                gsz_p = gdn_sizes(c.hidden, c.lin_kh, c.lin_vh, c.lin_dk, tiles=TILES, shaves=SHAVES,
                                  BN1=pack.GEOM["gdn_BN1"], BN2=pack.GEOM["gdn_BN2"])
                asz_p = attn_sizes(c.hidden, c.n_heads, c.n_kv, c.head_dim, tiles=TILES,
                                   BN2=pack.GEOM["attn_BN2"])
                assert gsz_p["NIN_pad"] == -(-ps["NIN"] // pgeom["BN1"]) * pgeom["BN1"], (
                    f"in_proj packed to {gsz_p['NIN_pad']} rows; prefill reads "
                    f"{-(-ps['NIN'] // pgeom['BN1']) * pgeom['BN1']}")
                assert asz_p["NQKV"] == -(-ps["NQKV"] // pgeom["BN1"]) * pgeom["BN1"], (
                    f"qkv_proj packed to {asz_p['NQKV']} rows; prefill reads "
                    f"{-(-ps['NQKV'] // pgeom['BN1']) * pgeom['BN1']}")
                for pl in self.prefill_ls:
                    kvl = max(1, sum(1 for i in self.layer_ids[:pl] if c.layer_types[i] != "linear_attention"))
                    kernels[f"prefill{pl}"] = make_prefill_model(
                        L=pl, N=ps["NP"], H=c.hidden, KH=c.lin_kh, VH=c.lin_vh, DK=c.lin_dk, DV=c.lin_dv,
                        NH=c.n_heads, NKV=c.n_kv, D=c.head_dim, ROT=c.rot_dim, SMAX=smax, I=c.moe_inter,
                        E=c.n_experts, TOPK=c.top_k, G=ck.group, tiles=TILES, shaves=SHAVES, eps=c.eps,
                        KVL=kvl, skip=pskip, HW=hw_g, shared_projections=prefill_shared,
                        moe_slot_output=prefill_moe_slots, **pgeom)
                    names[f"prefill{pl}"] = (f"tl_e2e_prefill_l{pl}_n{ps['NP']}_h{c.hidden}_s{smax}"
                                             + (f"_bn{pgeom['BN1']}_mb{pgeom['MB']}" if prefill_shared or prefill_bn1 is not None or prefill_mb is not None else "")
                                             + ("_sharedproj" if prefill_shared else "")
                                             + ("_overlap" if prefill_overlap else "")
                                             + (f"_mbe{prefill_mbe}" if prefill_mbe is not None else "")
                                             + ("_moeslots" if prefill_moe_slots else "")
                                             + ("_skip_" + "_".join(sorted(pskip)) if pskip else ""))
        else:
            kernels = {
                "gdn": make_gdn_layer(H=c.hidden, KH=c.lin_kh, VH=c.lin_vh, DK=c.lin_dk, DV=c.lin_dv, I=c.moe_inter, E=c.n_experts,
                                      TOPK=c.top_k, G=ck.group, tiles=TILES, shaves=SHAVES, eps=c.eps,
                                      gdn_kw=dict(BN1=G["gdn_BN1"], BN2=G["gdn_BN2"]), moe_kw=moe_kw),
                "attn": make_attn_layer(H=c.hidden, NH=c.n_heads, NKV=c.n_kv, D=c.head_dim, ROT=c.rot_dim, SMAX=smax, I=c.moe_inter,
                                        E=c.n_experts, TOPK=c.top_k, G=ck.group, tiles=TILES, shaves=SHAVES, eps=c.eps,
                                        attn_kw=dict(BN1=G["attn_BN1"], BN2=G["attn_BN2"]), moe_kw=moe_kw),
                "head": make_head(H=c.hidden, V=c.vocab, G=ck.group, tiles=TILES, shaves=SHAVES, BN=G["head_BN"], eps=c.eps, h_dtype=T.float32),
            }
            names = {"gdn": f"tl_e2e_gdn_layer_h{c.hidden}", "attn": f"tl_e2e_attn_layer_h{c.hidden}_s{smax}",
                     "head": f"tl_e2e_head32_h{c.hidden}_v{c.vocab}"}
        self.names = names
        import tvm
        cfg = {"tl.intel_npu.hw_sync_grid": os.environ.get("TL_NPU_HW_SYNC", "1") != "0",
               "tl.intel_npu.scoped_cmx": os.environ.get("TL_NPU_SCOPED_CMX", "1") != "0"}
        if os.environ.get("TL_NPU_PLAN_WORKLOADS") == "off":
            cfg["tl.intel_npu.plan_workloads"] = "off"
        self.low = {}
        for key, func in kernels.items():
            options = dict(cfg)
            if key.startswith("prefill"):
                options["tl.intel_npu.multicast"] = prefill_shared
                options["tl.intel_npu.multicast_overlap"] = prefill_overlap
            with tvm.transform.PassContext(config=options):
                self.low[key] = lowering.lower(func)
        prefix = os.environ.get("TL_NPU_SOURCE_PREFIX")   # diagnostics: C++ lines prepended to every kernel (e.g. a #define)
        if prefix:
            for k in kernels:
                self.low[k].source = prefix.replace("\\n", "\n") + "\n" + self.low[k].source
                names[k] += "_dbg"
        if build:
            t0 = time.perf_counter()
            runner.build_all({names[k]: self.low[k].source for k in kernels}, log=log)
            log(f"build {time.perf_counter() - t0:.0f} s")
        else:
            for k in kernels:
                runner.write_kernel(self.low[k].source, names[k])
        exe = runner.HERE / "build" / "Release" / "npu_run.exe"
        self.s = runner.Session(exe, runner.HERE / "run_e2e", TILES, SHAVES)
        for k, n in names.items():
            self.s.kernel(k, n, profile=os.environ.get("TL_NPU_SESSION_PROFILE") == "1")   # PROFILED runs are not timings
        self.pack_dir = Path(pack_dir)
        self.embed = ck.raw(LM + "embed_tokens.weight")   # bf16 [V, H], rows taken on demand
        self._upload_all()

    def _chunks(self) -> list[list[int]]:
        out, run = [], []
        for i in self.layer_ids:
            if self.c.layer_types[i] == "linear_attention" and len(run) < 3:
                run.append(i)
                continue
            if run:
                out.append(run)
                run = []
            if self.c.layer_types[i] == "linear_attention":
                run.append(i)
            else:
                out.append([i])
        if run:
            out.append(run)
        return out

    def _kv_slots(self, ci: int) -> int:
        """KV cache slots of chunk ci: one per full-attention layer (at least one, so the parameter is never empty)."""
        return max(1, sum(1 for i in self.chunks[ci] if self.c.layer_types[i] != "linear_attention"))

    def _chunk_kernel(self, ci: int) -> str:
        """Kernel key of chunk ci: its layer count, its KV slot count, and whether the head runs in it (the last chunk)."""
        if len(self.chunks) == 1:
            return "model"
        return f"model_l{len(self.chunks[ci])}k{self._kv_slots(ci)}" + ("_head" if ci == len(self.chunks) - 1 else "")

    # ---- buffers ------------------------------------------------------------------------------------------------
    def _upload_all(self):
        c, s = self.c, self.s
        H = c.hidden
        t0 = time.perf_counter()
        total = 0
        # diagnostic: a dead buffer allocated FIRST, pushing every weight to a higher device address. Prefill
        # faults at a layer count whose CUMULATIVE upload crosses 4 GiB, so padding is what separates "too many
        # layers" from "a weight above 2^32".
        pad_gb = float(os.environ.get("TL_NPU_PAD_GB", "0"))
        if pad_gb:
            s.alloc("PAD", int(pad_gb * (1 << 30)))
            self.log(f"pad: {pad_gb} GiB before the weights")
        # The prefill scratch goes FIRST, with decode's, because the 17.91 GB of weights follows and anything
        # allocated after it lands above 4 GiB. A prefill whose scratch sits there faults the device
        # (ZE_RESULT_ERROR_DEVICE_LOST): measured sharply -- 9 layers with a 0.2 GiB pad keeps the scratch at
        # 4.12 GB and returns, 0.4 GiB puts it at 4.32 GB and faults. Decode never hit it because its own scratch
        # was always allocated before the weights.
        if self.prefill_n:
            self._alloc_prefill_scratch()
        s.upload("Hio", np.zeros((1, H), np.float32))
        s.upload("Pos", np.zeros((1, 1), np.int32))
        s.upload("CosSin", np.zeros((2, c.rot_dim), np.float32))
        gsz = gdn_sizes(H, c.lin_kh, c.lin_vh, c.lin_dk, tiles=TILES, shaves=SHAVES, BN1=pack.GEOM["gdn_BN1"], BN2=pack.GEOM["gdn_BN2"])
        asz = attn_sizes(H, c.n_heads, c.n_kv, c.head_dim, tiles=TILES, BN2=pack.GEOM["attn_BN2"])
        hsz = head_sizes(H, c.vocab, tiles=TILES, shaves=SHAVES, BN=pack.GEOM["head_BN"])
        self.gsz, self.asz, self.hsz = gsz, asz, hsz
        # scratch shared by every layer of a kind
        for n, shape, dt in [("G_Y", (1, gsz["YLEN"]), np.float16), ("G_BA", (1, gsz["BA_LEN"]), np.float16), ("G_CV", (1, gsz["CVLEN"]), np.float16),
                             ("G_O", (1, gsz["OPAD"]), np.float16), ("G_Mix", (1, gsz["NOUT_pad"]), np.float16),
                             ("A_Y", (1, asz["NQKV"]), np.float16), ("A_QN", (c.n_kv, asz["GQ"], c.head_dim), np.float16),
                             ("A_Part", (c.n_kv, asz["GQ"], TILES, asz["PW"]), np.float32), ("A_O", (1, asz["KO"]), np.float16),
                             ("A_Mix", (1, asz["NOUT_pad"]), np.float16),
                             ("M_Part", (TILES, H), np.float32),
                             ("H_Logits", (TILES, hsz["CS"], SHAVES * pack.GEOM["head_BN"]), np.float16),
                             ("H_Part", (hsz["workers"], 32), np.float32), ("H_Tok", (1, 8), np.int32)]:
            s.alloc(n, int(np.prod(shape)) * np.dtype(dt).itemsize)
        for i in self.layer_ids:
            paths = pack.cached_paths(self.pack_dir, pack.layer_key(i), lambda: pack.pack_layer(self.ck, i), log=self.log)
            for k, p in paths.items():
                if k in ("in_norm", "post_norm"):
                    s.upload(f"L{i}_{k}", np.load(p).astype(np.float16).reshape(1, H))
                    continue
                total += s.upload_npy(f"L{i}_{k}", p)
            if c.layer_types[i] == "linear_attention":
                s.alloc(f"L{i}_ConvState", 3 * c.conv_dim * 2)
                s.alloc(f"L{i}_Ssm", c.lin_vh * c.lin_dk * c.lin_dv * 4)
            elif not self.single:
                s.alloc(f"L{i}_Kc", c.n_kv * self.smax * c.head_dim * 2)
                s.alloc(f"L{i}_Vt", c.n_kv * self.smax * c.head_dim * 2)
            # the MoE router logits scratch is per kernel, not per layer
        if self.single:
            # one KV cache per launch unit, a kernel parameter indexed by (kv slot, head, position): a DDR store whose
            # base is a table address AND whose offset is a runtime value loses the device (watchdog, DEVICE_LOST)
            kv = c.n_kv * self.smax * c.head_dim * 2
            for ci in range(len(self.chunks)):
                s.alloc(f"KcAll{ci}", self._kv_slots(ci) * kv)
                s.alloc(f"VtAll{ci}", self._kv_slots(ci) * kv)
        s.alloc("M_Logits", 16 * self._router_rows() * 2)
        head = pack.cached_paths(self.pack_dir, pack.head_key(), lambda: pack.pack_head_weights(self.ck), log=self.log)
        self.final_norm = np.load(head["final_norm"])
        total += s.upload_npy("H_WlmQ", head["WlmQ"])
        total += s.upload_npy("H_WlmS", head["WlmS"])
        s.upload("H_NormW", self.final_norm.astype(np.float16).reshape(1, H))
        self.log(f"uploaded {self.L} layers + head: {total / 1e9:.2f} GB in {time.perf_counter() - t0:.0f} s")
        if self.single:
            self._upload_table()
        if self.prefill_n:
            self._alloc_prefill()

    def _alloc_prefill_scratch(self):
        """The prefill kernel's DDR scratch. Called before the weights: see `_upload_all`."""
        c, s, ps = self.c, self.s, self.ps
        NP, H = ps["NP"], c.hidden
        for nm, nbytes in (("P_Hio", NP * H * 4), ("P_Pos", 4), ("P_Nt", 4), ("P_CosSin", NP * 2 * c.rot_dim * 4),
                           ("P_Xs", NP * H * 2), ("P_Ys", NP * ps["YW"] * 2), ("P_BAs", NP * 2 * c.lin_vh * 2),
                           ("P_QKs", c.lin_kh * NP * 2 * c.lin_dk * 4), ("P_Vs", c.lin_vh * NP * c.lin_dv * 2),
                           ("P_Gbs", c.lin_vh * NP * 2 * 4), ("P_Zs", c.lin_vh * NP * c.lin_dv * 2),
                           ("P_Q", c.n_kv * NP * ps["GQ"] * c.head_dim * 2), ("P_Gate", NP * ps["KO"] * 2),
                           ("P_Os", NP * ps["OW"] * 2), ("P_Mix", NP * H * 2),
                           ("P_Logits", NP * ps["RR"] * 2), ("P_Ids", NP * c.top_k * 4), ("P_Wts", NP * 16 * 4),
                           ("P_Cnt", (c.n_experts + 1) * 4), ("P_Off", (c.n_experts + 2) * 4),
                           ("P_Slot", ps["SLOTS"] * 4),
                           ("P_PartE", ps["SLOTS"] * H * 2 if self.prefill_moe_slots else TILES * NP * H * 4),
                           ("P_MoeOut", NP * H * 4)):
            s.alloc(nm, nbytes)

    def _alloc_prefill(self):
        """The prefill kernel's layer table. Weights, conv state, SSM and the KV cache are the buffers decode
        already has: the two tables name the same allocations, in each kernel's own column order."""
        c, s, ps = self.c, self.s, self.ps
        col = {n: i for i, n in enumerate(PREFILL_TABLE_COLS)}
        table = np.zeros((self.L, len(PREFILL_TABLE_COLS)), np.int64)
        kind = np.zeros((self.L, 4), np.int32)
        kv_slot = 0
        for r, i in enumerate(self.layer_ids):
            gdn = c.layer_types[i] == "linear_attention"
            kind[r, 0] = 0 if gdn else 1
            if not gdn:
                kind[r, 1] = kv_slot      # the same slot numbering decode's table uses: both walk layers in order
                kv_slot += 1
            names = dict(InNorm=f"L{i}_in_norm", PostNorm=f"L{i}_post_norm", Wr=f"L{i}_Wr",
                         WguQ=f"L{i}_WguQ", WguS=f"L{i}_WguS", WdQ=f"L{i}_WdQ", WdS=f"L{i}_WdS")
            for k, v in (GDN_PREFILL_TABLE if gdn else ATTN_PREFILL_TABLE).items():
                names[k] = f"L{i}_{v}"
            for k, n in names.items():
                table[r, col[k]] = s.address(n)
        assert (table[:, [col["InNorm"], col["PostNorm"], col["Wr"]]] != 0).all()
        s.upload("P_Kind", kind)
        s.upload("P_Table", table)
        self.log(f"prefill: {ps['NP']} tokens a launch, {self.L} rows x {len(PREFILL_TABLE_COLS)} addresses")

    def _prefill_bufs(self, alias: str):
        m = dict(Hio="P_Hio", Pos="P_Pos", Nt="P_Nt", CosSin="P_CosSin", Kind="P_Kind", Table="P_Table",
                 KcAll="KcAll0", VtAll="VtAll0", Xs="P_Xs", Ys="P_Ys", BAs="P_BAs", QKs="P_QKs", Vs="P_Vs",
                 Gbs="P_Gbs", Zs="P_Zs", Q="P_Q", Gate="P_Gate", Os="P_Os", Mix="P_Mix", Logits="P_Logits",
                 Ids="P_Ids", Wts="P_Wts", Cnt="P_Cnt", Off="P_Off", Slot="P_Slot", PartE="P_PartE",
                 MoeOut="P_MoeOut")
        return [m[n] for n in self.low[alias].param_names]

    def prefill(self, ids: list[int], L: int | None = None) -> float:
        """Runs `ids` through the prefill kernels in chunks of NP, advancing `self.pos`. Returns the device
        kernel-only milliseconds. `L` picks a kernel of the sweep; a kernel built for L reads the first L rows of
        the one shared table, so every layer count runs off the same upload."""
        import torch
        c, s, ps = self.c, self.s, self.ps
        alias = f"prefill{self.prefill_ls[-1] if L is None else L}"
        NP, H = ps["NP"], c.hidden
        dev_ms = 0.0
        for c0 in range(0, len(ids), NP):
            blk = ids[c0:c0 + NP]
            h = np.zeros((NP, H), np.float32)
            for j, t in enumerate(blk):
                h[j] = self.embed[t].to(torch.float32).numpy()
            s.upload("P_Hio", h)
            s.upload("P_Pos", np.array([[self.pos]], np.int32))
            s.upload("P_Nt", np.array([[len(blk)]], np.int32))
            s.upload("P_CosSin", np.stack([rope_cos_sin(self.pos + t, c.rot_dim, c.rope_theta).reshape(-1)
                                           for t in range(NP)]).astype(np.float32))
            _, km = s.launch(alias, *self._prefill_bufs(alias))
            dev_ms += km
            if os.environ.get("TL_NPU_PREFILL_DBG") == "1":
                hio = s.download("P_Hio", np.float32, (NP, H))
                live = hio[:len(blk)]
                print(f"  prefill out: live rows absmax {np.abs(live).max():.3e}, nan {int(np.isnan(live).sum())}, "
                      f"inf {int(np.isinf(live).sum())}; row0[:4] {np.round(live[0, :4], 3)}", file=sys.stderr)
            self.pos += len(blk)
        return dev_ms

    def _upload_table(self):
        """Kind[L, 4] and Table[L, len(TABLE_COLS)]: the device addresses of every layer's weights and states, in the
        column order `make_decode_model` reads (`TABLE_COLS`, `GDN_TABLE` / `ATTN_TABLE`); unused columns 0."""
        c, s = self.c, self.s
        col = {n: i for i, n in enumerate(TABLE_COLS)}
        for ci, chunk in enumerate(self.chunks):
            kind = np.zeros((len(chunk), 4), np.int32)
            table = np.zeros((len(chunk), len(TABLE_COLS)), np.int64)
            kv_slot = 0
            for r, i in enumerate(chunk):
                gdn = c.layer_types[i] == "linear_attention"
                kind[r, 0] = 0 if gdn else 1
                if not gdn:
                    kind[r, 1] = kv_slot   # the layer's slot in KcAll / VtAll
                    kv_slot += 1
                names = dict(InNorm=f"L{i}_in_norm", PostNorm=f"L{i}_post_norm", Wr=f"L{i}_Wr",
                             WguQ=f"L{i}_WguQ", WguS=f"L{i}_WguS", WdQ=f"L{i}_WdQ", WdS=f"L{i}_WdS")
                for k, v in (GDN_TABLE if gdn else ATTN_TABLE).items():
                    names[k] = f"L{i}_{v}"
                for k, n in names.items():
                    table[r, col[k]] = s.address(n)
            assert (table[:, [col["InNorm"], col["PostNorm"], col["Wr"]]] != 0).all()
            s.upload(f"Kind{ci}", kind)
            s.upload(f"Table{ci}", table)
        self.log(f"layer table: {self.L} rows x {len(TABLE_COLS)} device addresses in {len(self.chunks)} chunk(s)")

    def _router_rows(self) -> int:
        c = self.c
        per_round = TILES * SHAVES * pack.GEOM["moe_BNR"]
        return -(-(c.n_experts + 1) // per_round) * per_round

    def reset_state(self):
        for i in self.layer_ids:
            if self.c.layer_types[i] == "linear_attention":
                for k in ("ConvState", "Ssm"):
                    self.s.zero(f"L{i}_{k}")
            elif not self.single:
                for k in ("Kc", "Vt"):
                    self.s.zero(f"L{i}_{k}")
        if self.single:
            for ci in range(len(self.chunks)):
                self.s.zero(f"KcAll{ci}")
                self.s.zero(f"VtAll{ci}")
        self.pos = 0

    # ---- one token ----------------------------------------------------------------------------------------------
    def _bufs(self, kind, i):
        """Buffer names in the kernel's device parameter order."""
        if kind == "gdn":
            m = dict(Hio="Hio", InNorm=f"L{i}_in_norm", PostNorm=f"L{i}_post_norm", WinQ=f"L{i}_WinQ", WinS=f"L{i}_WinS",
                     Wba=f"L{i}_Wba", ConvW=f"L{i}_ConvW",
                     ALog=f"L{i}_ALog", DtBias=f"L{i}_DtBias", GNorm=f"L{i}_GNorm", ConvState=f"L{i}_ConvState", Ssm=f"L{i}_Ssm",
                     WoutQ=f"L{i}_WoutQ", WoutS=f"L{i}_WoutS", Wr=f"L{i}_Wr", WguQ=f"L{i}_WguQ", WguS=f"L{i}_WguS",
                     WdQ=f"L{i}_WdQ", WdS=f"L{i}_WdS", Y="G_Y", BA="G_BA", CV="G_CV", O="G_O",
                     Mix="G_Mix", Logits="M_Logits", PartM="M_Part")
        elif kind == "attn":
            m = dict(Hio="Hio", InNorm=f"L{i}_in_norm", PostNorm=f"L{i}_post_norm", WqkvQ=f"L{i}_WqkvQ",
                     WqkvS=f"L{i}_WqkvS", QNormW=f"L{i}_QNormW",
                     KNormW=f"L{i}_KNormW", CosSin="CosSin", Pos="Pos", Kc=f"L{i}_Kc", Vt=f"L{i}_Vt",
                     WoQ=f"L{i}_WoQ", WoS=f"L{i}_WoS", Wr=f"L{i}_Wr",
                     WguQ=f"L{i}_WguQ", WguS=f"L{i}_WguS", WdQ=f"L{i}_WdQ", WdS=f"L{i}_WdS",
                     Y="A_Y", QN="A_QN", PartA="A_Part", O="A_O", Mix="A_Mix", Logits="M_Logits",
                     PartM="M_Part")
        else:
            m = dict(Hs="Hio", NormW="H_NormW", WlmQ="H_WlmQ", WlmS="H_WlmS", Logits="H_Logits", Part="H_Part", Tok="H_Tok")
        return [m[n] for n in self.low[kind].param_names]

    def _model_bufs(self, ci: int = 0):
        m = dict(Hio="Hio", Pos="Pos", CosSin="CosSin", Kind=f"Kind{ci}", Table=f"Table{ci}", KcAll=f"KcAll{ci}", VtAll=f"VtAll{ci}",
                 Y="G_Y", BA="G_BA", CV="G_CV", O="G_O", Mix="G_Mix",
                 YA="A_Y", QN="A_QN", PartA="A_Part", OA="A_O", MixA="A_Mix", Logits="M_Logits", PartM="M_Part", NormW="H_NormW",
                 WlmQ="H_WlmQ", WlmS="H_WlmS", HLogits="H_Logits", HPart="H_Part", Tok="H_Tok")
        return [m[n] for n in self.low[self._chunk_kernel(ci)].param_names]

    def step(self, token: int, trace: list | None = None) -> tuple[int, dict]:
        """Runs `token` at the current position through every layer and the head; returns (next token, timing).
        `trace` (per-layer mode; slow: a download per layer) collects the residual stream after every layer."""
        c, s = self.c, self.s
        H = c.hidden
        import torch
        tt = time.perf_counter()
        h = self.embed[token].to(torch.float32).numpy().copy()
        t0 = time.perf_counter()
        dev_ms = dict(layers=0.0, head=0.0)
        s.upload("Hio", h.reshape(1, H))
        s.upload("Pos", np.array([[self.pos]], np.int32))
        s.upload("CosSin", rope_cos_sin(self.pos, c.rot_dim, c.rope_theta))
        t1 = time.perf_counter()
        if self.single:
            dev_ms = dict(model=0.0)
            for ci in range(len(self.chunks)):
                if os.environ.get("TL_NPU_DBG_WARM_LAUNCH") == "1":   # diagnostic: a throwaway launch right before the timed one (clock ramp)
                    s.launch(self._chunk_kernel(ci), *self._model_bufs(ci))
                hm, km = s.launch(self._chunk_kernel(ci), *self._model_bufs(ci))
                dev_ms["model"] += km
            if os.environ.get("TL_NPU_DBG_STEP_TIMING") == "1":
                print(f"step timing: embed {1e3 * (t0 - tt):.1f} ms, uploads {1e3 * (t1 - t0):.1f} ms, launch round trip "
                      f"{1e3 * (time.perf_counter() - t1):.1f} ms (runner host {hm:.1f} ms, kernel {km:.1f} ms)", file=sys.stderr)
            if getattr(s, "last_status", None):
                ws = s.last_status
                for w in range(len(ws) // 16):
                    print("status shave", w, " ".join(f"{x:#x}" for x in ws[w * 16:(w + 1) * 16]), file=sys.stderr)
        else:
            for i in self.layer_ids:
                kind = "gdn" if c.layer_types[i] == "linear_attention" else "attn"
                _, km = s.launch(kind, *self._bufs(kind, i))
                dev_ms["layers"] += km
                if trace is not None:
                    trace.append(dict(layer=i, out=s.download("Hio", np.float32, (1, H))[0].copy()))
            _, km = s.launch("head", *self._bufs("head", 0))
            dev_ms["head"] += km
        tok = int(s.download("H_Tok", np.int32, (1, 8))[0, 0])
        self.pos += 1
        dev_ms["host_wall"] = (time.perf_counter() - t0) * 1e3
        return tok, dev_ms

    def head_logits(self) -> np.ndarray:
        V = self.c.vocab
        return self.s.download("H_Logits", np.float16, (TILES, self.hsz["CS"], SHAVES * pack.GEOM["head_BN"])).reshape(-1)[:V].astype(np.float32)

    def close(self):
        self.s.close()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("TL_NPU_CKPT"))
    ap.add_argument("--pack-dir", default=os.environ.get("TL_NPU_PACK_DIR"))
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--prompt", default="Give me a short introduction to large language models.")
    ap.add_argument("--ids", default=None, help="prompt token ids, comma-separated (instead of --prompt)")
    ap.add_argument("--raw", action="store_true", help="no chat template")
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--smax", type=int, default=1536)
    ap.add_argument("--golden", default=None, help="reference_golden.py's .npz: teacher-forced comparison (prompt and layers from it)")
    ap.add_argument("--trials", type=int, default=1,
                    help="repeat the teacher-forced comparison N times and report n/N; one run cannot tell a "
                         "correct kernel from one that is right four times in five (plan question 10)")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--per-layer", action="store_true", help="one launch per layer + head (the A/B baseline) instead of one per token")
    ap.add_argument("--layer-ids", default=None, help="run only these checkpoint layers, in order (e.g. 3 or 0,1,3): device bisection")
    ap.add_argument("--chunked", action="store_true",
                    help="one launch per run of up to 3 GDN layers and per attention layer (head in the last): the A/B "
                         "baseline for the single-launch model")
    ap.add_argument("--prefill", type=int, nargs="?", const=192, default=0, metavar="N",
                    help="run the prompt through make_prefill_model in chunks of N tokens (default 192) instead of "
                         "token by token through the decode kernels; this is what TTFT should measure")
    ap.add_argument("--prefill-only", action="store_true",
                    help="stop after the prefill launches, before any decode step: the prefill bisect, which must "
                         "not depend on the decode path (a partial --layers / --layer-ids set hangs decode)")
    ap.add_argument("--prefill-shared", action="store_true")
    ap.add_argument("--prefill-overlap", action="store_true")
    ap.add_argument("--prefill-bn1", type=int, default=None)
    ap.add_argument("--prefill-mb", type=int, default=None)
    ap.add_argument("--prefill-mbe", type=int, default=None)
    ap.add_argument("--prefill-moe-slots", action="store_true")
    ap.add_argument("--prefill-sweep", default=None, metavar="L1,L2,...",
                    help="layer counts to launch prefill at, ascending, from ONE session: the kernels are built in "
                         "one pass and share the weight upload, so a bisect point costs a launch instead of a "
                         "rebuild. A hang ends the session, so the sweep stops there and names the layer count")
    a = ap.parse_args(argv)
    if not a.ckpt or not a.pack_dir:
        print("--ckpt and --pack-dir (or TL_NPU_CKPT / TL_NPU_PACK_DIR) are required")
        return 2
    ck = Checkpoint(a.ckpt)
    gold = None
    if a.golden:
        if Path(a.golden).suffix == ".pt":
            import torch
            saved = torch.load(a.golden, map_location="cpu", weights_only=True)
            gold = {name: value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                    for name, value in saved.items()}
            gold.setdefault("layers", np.int64(ck.cfg.layers))
        else:
            gold = np.load(a.golden)
    layers = int(gold["layers"]) if gold is not None else (a.layers if a.layers is not None else ck.cfg.layers)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.ckpt) / "tokenizer.json"))
    if gold is not None:
        ids = [int(x) for x in gold["prompt_ids"]]
    elif a.ids:
        ids = [int(x) for x in a.ids.split(",")]
    elif a.raw:
        ids = tok.encode(a.prompt, add_special_tokens=False).ids
    else:
        text = f"<|im_start|>user\n{a.prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) + a.new <= a.smax, "prompt + new tokens must fit the KV window (--smax)"
    mode = "one launch per layer" if a.per_layer else ("one launch per chunk" if a.chunked else "one launch per token")
    print(f"prompt: {len(ids)} tokens; layers {layers} of {ck.cfg.layers}; new {a.new}; {mode}")
    m = DeviceModel(ck, Path(a.pack_dir), layers, a.smax, build=not a.no_build, single=not a.per_layer,
                    layer_ids=[int(x) for x in a.layer_ids.split(",")] if a.layer_ids else None, chunked=a.chunked,
                    prefill=a.prefill if not a.per_layer else 0,
                    prefill_ls=[int(x) for x in a.prefill_sweep.split(",")] if a.prefill_sweep else None,
                    prefill_shared=a.prefill_shared, prefill_overlap=a.prefill_overlap,
                    prefill_bn1=a.prefill_bn1, prefill_mb=a.prefill_mb, prefill_mbe=a.prefill_mbe,
                    prefill_moe_slots=a.prefill_moe_slots)
    if a.chunked:
        print(f"chunks: {len(m.chunks)} launches per token: " + " ".join("/".join(str(i) for i in ch) for ch in m.chunks))
    try:
        m.reset_state()
        if a.prefill_sweep:
            for pl in m.prefill_ls:
                m.reset_state()
                t0 = time.perf_counter()
                km = m.prefill(ids[:-1], L=pl)
                print(f"  L={pl:3d}: returned, {km:7.1f} ms device, {time.perf_counter() - t0:.1f} s wall",
                      flush=True)
            print(f"prefill sweep: every layer count in {m.prefill_ls} returned")
            return 0
        if gold is not None:
            t0 = time.perf_counter()
            dev_trace = [] if a.per_layer else None
            if m.prefill_n:
                if len(ids) > 1:
                    m.prefill(ids[:-1])
                dev_tok, ms = m.step(ids[-1])
            else:
                dev_tok, ms = m.step(ids[0], trace=dev_trace)
            if dev_trace is not None:
                print("layer-by-layer, first prompt token: max |h_dev - h_ref| / max |h_ref| after the layer:")
                worst = 0.0
                for i in range(layers):
                    ref_out = gold["trace_out"][i]
                    rel_h = np.abs(dev_trace[i]["out"] - ref_out).max() / np.abs(ref_out).max()
                    worst = max(worst, rel_h)
                    print(f"  layer {i:2d} {ck.cfg.layer_types[i][:6]}: h rel {rel_h:.2e}")
                print(f"  worst h rel over the layers: {worst:.2e}")
            elif not m.prefill_n and "trace_out" in gold:
                ref_out = gold["trace_out"][layers - 1]
                h_dev = m.s.download("Hio", np.float32, (1, ck.cfg.hidden))[0]
                print(f"first prompt token, residual stream after layer {layers - 1}: max |h_dev - h_ref| / max |h_ref| = "
                      f"{np.abs(h_dev - ref_out).max() / np.abs(ref_out).max():.2e}")
            # the rest of the prompt, then the reference's greedy tokens, teacher-forced
            ref_logits = gold["logits"].astype(np.float32)        # [1 + new, V]
            out_ids = [int(x) for x in gold["out_ids"][:a.new]]
            clean = 0
            for trial in range(a.trials):
                if trial:
                    m.reset_state()
                    if m.prefill_n:
                        if len(ids) > 1:
                            m.prefill(ids[:-1])
                        dev_tok, ms = m.step(ids[-1])
                    else:
                        dev_tok, ms = m.step(ids[0])
                if not m.prefill_n:
                    for t in ids[1:]:
                        dev_tok, ms = m.step(t)
                agree, worst = 0, 0.0
                dev_logits = m.head_logits()
                for k, r in enumerate(out_ids):
                    ok = dev_tok == r
                    agree += int(ok)
                    dl_err = float(np.abs(dev_logits - ref_logits[k]).max())
                    worst = max(worst, dl_err)
                    if a.trials == 1:
                        print(f"  step {k:2d}: dev {dev_tok:6d} ref {r:6d} {'OK  ' if ok else 'DIFF'} ({tok.decode([r])!r}); "
                              f"max |logits dev - ref| {dl_err:.3f} over {np.abs(ref_logits[k]).max():.1f}; "
                              f"device {sum(v for kk, v in ms.items() if kk != 'host_wall'):.1f} ms, "
                              f"host wall {ms['host_wall']:.0f} ms", flush=True)
                    dev_tok, ms = m.step(r)
                    dev_logits = m.head_logits()
                clean += int(agree == len(out_ids))
                if a.trials > 1:
                    print(f"  trial {trial}: {agree}/{len(out_ids)} agree, worst |logits dev - ref| {worst:.3f}", flush=True)
            print(f"golden: {clean}/{a.trials} trials agree with the reference at all {len(out_ids)} greedy tokens "
                  f"(teacher-forced; {time.perf_counter() - t0:.0f} s)")
            return 0 if clean == a.trials else 1
        # free generation
        out, times = [], []
        t_pre = time.perf_counter()
        pre_ms = 0.0
        if m.prefill_n:
            # the prompt through the prefill kernels, all but its last token; the last one goes through decode so
            # that the first output token comes out of the same path every later token does
            if len(ids) > 1:
                pre_ms = m.prefill(ids[:-1])
            if a.prefill_only:
                print(f"prefill only: {len(ids) - 1} tokens through {m.L} layers in "
                      f"{-(-(len(ids) - 1) // m.ps['NP'])} launch(es), {pre_ms:.1f} ms device [UNPROFILED, "
                      f"{'TURBO' if m.s.turbo else 'NO TURBO'}] -> the prefill kernels returned")
                return 0
        else:
            for t in ids[:-1]:
                m.step(t)
        nxt, ms = m.step(ids[-1])
        ttft = time.perf_counter() - t_pre      # prompt through the model plus the first token
        for _ in range(a.new):
            out.append(nxt)
            times.append(ms)
            if nxt in (248044, 248046):
                break
            nxt, ms = m.step(nxt)
        print("output:", tok.decode(out))
        print("ids:", ",".join(map(str, out)))
        if m.prefill_n:
            print(f"TTFT: {ttft:.2f} s for {len(ids)} prompt tokens + the first output "
                  f"({1e3 * ttft / len(ids):.0f} ms per prompt token; prefill kernels {pre_ms:.0f} ms device for "
                  f"{len(ids) - 1} tokens = {pre_ms / max(1, len(ids) - 1):.1f} ms/token, plus one decode step)")
        else:
            # the prompt replayed through the DECODE kernels, one token per step: an upper bound, not prefill's number
            print(f"TTFT: {ttft:.2f} s for {len(ids)} prompt tokens + the first output "
                  f"({1e3 * ttft / len(ids):.0f} ms per prompt token, decode kernels -- pass --prefill for the real one)")
        if times:
            dev = sorted(sum(v for k, v in t.items() if k != "host_wall") for t in times)
            wall = sorted(t["host_wall"] for t in times)
            med = times[len(times) // 2]
            cond = "UNPROFILED, " + ("TURBO" if m.s.turbo else "NO TURBO")
            split = (f"one launch, kernel-only {med['model']:.1f} ms" if "model" in med
                     else f"layers {med['layers']:.1f} + head {med['head']:.1f} of the median step")
            print(f"per token: device launches sum median {dev[len(dev) // 2]:.1f} ms ({split}), "
                  f"host wall median {wall[len(wall) // 2]:.0f} ms = {1000 / wall[len(wall) // 2]:.1f} tok/s [{cond}]")
            print(f"generate: {len(out)} tokens in {sum(t['host_wall'] for t in times) / 1e3:.2f} s = "
                  f"{len(out) / (sum(t['host_wall'] for t in times) / 1e3):.1f} tok/s (host wall, generation only)")
            if os.environ.get("TL_NPU_SESSION_PROFILE") == "1" and m.single:   # the last launch's timeline per kernel
                for ci in range(len(m.chunks)):
                    p = Path(os.environ.get("TL_NPU_SESSION_PROFILE_DIR", str(HERE / "run_e2e"))) / f"trace_{m._chunk_kernel(ci)}.json"
                    m.s.profile(m._chunk_kernel(ci), p)
                    print(f"trace: {p}")
        return 0
    finally:
        m.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
