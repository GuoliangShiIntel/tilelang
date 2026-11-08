"""Decode a Qwen3.5-MoE-family model on the NPU with this backend's kernels. Default: ONE launch per token
(`kernels.make_decode_model`: a device-side loop over the layers, each layer's weights and states reached through a
table of device addresses, the fp32 residual stream in CMX for the whole step, final norm + lm_head + argmax at the end).
`--per-layer`: one launch per fused layer (`make_gdn_layer` / `make_attn_layer`) and one for the head (the A/B baseline).
Weights and states are resident in one `npu_run --session`; the host uploads the embedding row, launches, and reads the
token back. The prompt is run token by token through the same decode kernels (correct, slow: prefill kernels are a later
milestone).

    python testing/python/intel_npu/e2e/decode.py --ckpt D:\\models\\... --pack-dir D:\\models\\..._tlpack
        [--layers N] [--prompt "text" | --ids 1,2,3] [--new 16] [--smax 1536] [--golden golden.npz] [--no-build] [--per-layer]

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
from kernels import (ATTN_TABLE, GDN_TABLE, TABLE_COLS, attn_sizes, gdn_sizes, head_sizes, make_attn_layer, make_decode_model,  # noqa: E402
                     make_gdn_layer, make_head, rope_cos_sin)
import tilelang.intel_npu.language as T  # noqa: E402
from checkpoint import Checkpoint, LM  # noqa: E402
import pack  # noqa: E402

TILES, SHAVES = 3, 2


class DeviceModel:
    def __init__(self, ck: Checkpoint, pack_dir: Path, layers: int, smax: int, build: bool, single: bool = True, log=print,
                 layer_ids: list[int] | None = None, chunked: bool = False):
        self.ck, self.c, self.log = ck, ck.cfg, log
        c = self.c
        self.layer_ids = list(layer_ids) if layer_ids is not None else list(range(layers))   # checkpoint layers, in run order
        layers = len(self.layer_ids)
        self.L = layers
        self.smax = smax
        self.single = single
        # launch units: the whole model, or (chunked) runs of up to 3 GDN layers and single attention layers, the head in the last
        self.chunks = self._chunks() if (single and chunked) else [self.layer_ids]
        G = pack.GEOM
        moe_kw = dict(BNR=G["moe_BNR"], BNG=G["moe_BNG"], BND=G["moe_BND"])
        if single:
            base_skip = frozenset(x for x in os.environ.get("TL_NPU_DBG_SKIP", "").split(",") if x)
            r2 = dict(R2=int(os.environ["TL_NPU_DBG_R2"])) if os.environ.get("TL_NPU_DBG_R2") else {}   # out_proj rows (model default 16)
            kernels, names = {}, {}
            for ci, chunk in enumerate(self.chunks):
                key = self._chunk_kernel(ci)
                if key in kernels:
                    continue
                skip = base_skip if ci == len(self.chunks) - 1 else base_skip | {"head"}
                kernels[key] = make_decode_model(L=len(chunk), H=c.hidden, KH=c.lin_kh, VH=c.lin_vh, DK=c.lin_dk, DV=c.lin_dv, NH=c.n_heads,
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
        with tvm.transform.PassContext(config=cfg):
            self.low = {k: lowering.lower(f) for k, f in kernels.items()}
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

    def _chunk_kernel(self, ci: int) -> str:
        """Kernel key of chunk ci: its layer count, and whether the head runs in it (the last chunk)."""
        if len(self.chunks) == 1:
            return "model"
        return f"model_l{len(self.chunks[ci])}" + ("_head" if ci == len(self.chunks) - 1 else "")

    # ---- buffers ------------------------------------------------------------------------------------------------
    def _upload_all(self):
        c, s = self.c, self.s
        H = c.hidden
        t0 = time.perf_counter()
        total = 0
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
            else:
                s.alloc(f"L{i}_Kc", c.n_kv * self.smax * c.head_dim * 2)
                s.alloc(f"L{i}_Vt", c.n_kv * self.smax * c.head_dim * 2)
            # the MoE router logits scratch is per kernel, not per layer
        s.alloc("M_Logits", 16 * self._router_rows() * 2)
        head = pack.cached_paths(self.pack_dir, pack.head_key(), lambda: pack.pack_head_weights(self.ck), log=self.log)
        self.final_norm = np.load(head["final_norm"])
        total += s.upload_npy("H_WlmP", head["WlmP"])
        s.upload("H_NormW", self.final_norm.astype(np.float16).reshape(1, H))
        self.log(f"uploaded {self.L} layers + head: {total / 1e9:.2f} GB in {time.perf_counter() - t0:.0f} s")
        if self.single:
            self._upload_table()

    def _upload_table(self):
        """Kind[L, 4] and Table[L, len(TABLE_COLS)]: the device addresses of every layer's weights and states, in the
        column order `make_decode_model` reads (`TABLE_COLS`, `GDN_TABLE` / `ATTN_TABLE`); unused columns 0."""
        c, s = self.c, self.s
        col = {n: i for i, n in enumerate(TABLE_COLS)}
        for ci, chunk in enumerate(self.chunks):
            kind = np.zeros((len(chunk), 4), np.int32)
            table = np.zeros((len(chunk), len(TABLE_COLS)), np.int64)
            for r, i in enumerate(chunk):
                gdn = c.layer_types[i] == "linear_attention"
                kind[r, 0] = 0 if gdn else 1
                names = dict(InNorm=f"L{i}_in_norm", PostNorm=f"L{i}_post_norm", Wr=f"L{i}_Wr", WguP=f"L{i}_WguP", WdP=f"L{i}_WdP")
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
            for k in (("ConvState", "Ssm") if self.c.layer_types[i] == "linear_attention" else ("Kc", "Vt")):
                self.s.zero(f"L{i}_{k}")
        self.pos = 0

    # ---- one token ----------------------------------------------------------------------------------------------
    def _bufs(self, kind, i):
        """Buffer names in the kernel's device parameter order."""
        if kind == "gdn":
            m = dict(Hio="Hio", InNorm=f"L{i}_in_norm", PostNorm=f"L{i}_post_norm", WinP=f"L{i}_WinP", Wba=f"L{i}_Wba", ConvW=f"L{i}_ConvW",
                     ALog=f"L{i}_ALog", DtBias=f"L{i}_DtBias", GNorm=f"L{i}_GNorm", ConvState=f"L{i}_ConvState", Ssm=f"L{i}_Ssm",
                     WoutP=f"L{i}_WoutP", Wr=f"L{i}_Wr", WguP=f"L{i}_WguP", WdP=f"L{i}_WdP", Y="G_Y", BA="G_BA", CV="G_CV", O="G_O",
                     Mix="G_Mix", Logits="M_Logits", PartM="M_Part")
        elif kind == "attn":
            m = dict(Hio="Hio", InNorm=f"L{i}_in_norm", PostNorm=f"L{i}_post_norm", WqkvP=f"L{i}_WqkvP", QNormW=f"L{i}_QNormW",
                     KNormW=f"L{i}_KNormW", CosSin="CosSin", Pos="Pos", Kc=f"L{i}_Kc", Vt=f"L{i}_Vt", WoP=f"L{i}_WoP", Wr=f"L{i}_Wr",
                     WguP=f"L{i}_WguP", WdP=f"L{i}_WdP", Y="A_Y", QN="A_QN", PartA="A_Part", O="A_O", Mix="A_Mix", Logits="M_Logits",
                     PartM="M_Part")
        else:
            m = dict(Hs="Hio", NormW="H_NormW", WlmP="H_WlmP", Logits="H_Logits", Part="H_Part", Tok="H_Tok")
        return [m[n] for n in self.low[kind].param_names]

    def _model_bufs(self, ci: int = 0):
        m = dict(Hio="Hio", Pos="Pos", CosSin="CosSin", Kind=f"Kind{ci}", Table=f"Table{ci}", Y="G_Y", BA="G_BA", CV="G_CV", O="G_O", Mix="G_Mix",
                 YA="A_Y", QN="A_QN", PartA="A_Part", OA="A_O", MixA="A_Mix", Logits="M_Logits", PartM="M_Part", NormW="H_NormW",
                 WlmP="H_WlmP", HLogits="H_Logits", HPart="H_Part", Tok="H_Tok")
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
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--per-layer", action="store_true", help="one launch per layer + head (the A/B baseline) instead of one per token")
    ap.add_argument("--layer-ids", default=None, help="run only these checkpoint layers, in order (e.g. 3 or 0,1,3): device bisection")
    ap.add_argument("--chunked", action="store_true",
                    help="one launch per run of up to 3 GDN layers and per attention layer (head in the last): the device is lost "
                         "after a launch holding an attention layer and another layer")
    a = ap.parse_args(argv)
    if not a.ckpt or not a.pack_dir:
        print("--ckpt and --pack-dir (or TL_NPU_CKPT / TL_NPU_PACK_DIR) are required")
        return 2
    ck = Checkpoint(a.ckpt)
    gold = np.load(a.golden) if a.golden else None
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
                    layer_ids=[int(x) for x in a.layer_ids.split(",")] if a.layer_ids else None, chunked=a.chunked)
    if a.chunked:
        print(f"chunks: {len(m.chunks)} launches per token: " + " ".join("/".join(str(i) for i in ch) for ch in m.chunks))
    try:
        m.reset_state()
        if gold is not None:
            t0 = time.perf_counter()
            dev_trace = [] if a.per_layer else None
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
            else:
                ref_out = gold["trace_out"][layers - 1]
                h_dev = m.s.download("Hio", np.float32, (1, ck.cfg.hidden))[0]
                print(f"first prompt token, residual stream after layer {layers - 1}: max |h_dev - h_ref| / max |h_ref| = "
                      f"{np.abs(h_dev - ref_out).max() / np.abs(ref_out).max():.2e}")
            # the rest of the prompt, then the reference's greedy tokens, teacher-forced
            for t in ids[1:]:
                dev_tok, ms = m.step(t)
            ref_logits = gold["logits"].astype(np.float32)        # [1 + new, V]
            out_ids = [int(x) for x in gold["out_ids"]]
            agree, lines = 0, []
            dev_logits = m.head_logits()
            for k, r in enumerate(out_ids):
                ok = dev_tok == r
                agree += int(ok)
                dl_err = float(np.abs(dev_logits - ref_logits[k]).max())
                lines.append(f"  step {k:2d}: dev {dev_tok:6d} ref {r:6d} {'OK  ' if ok else 'DIFF'} ({tok.decode([r])!r}); "
                             f"max |logits dev - ref| {dl_err:.3f} over {np.abs(ref_logits[k]).max():.1f}; "
                             f"device {sum(v for kk, v in ms.items() if kk != 'host_wall'):.1f} ms, host wall {ms['host_wall']:.0f} ms")
                print(lines[-1], flush=True)
                dev_tok, ms = m.step(r)
                dev_logits = m.head_logits()
            print(f"golden: {agree}/{len(out_ids)} greedy tokens agree with the reference (teacher-forced; {time.perf_counter() - t0:.0f} s)")
            return 0 if agree == len(out_ids) else 1
        # free generation
        out, times = [], []
        for t in ids[:-1]:
            m.step(t)
        nxt, ms = m.step(ids[-1])
        for _ in range(a.new):
            out.append(nxt)
            times.append(ms)
            if nxt in (248044, 248046):
                break
            nxt, ms = m.step(nxt)
        print("output:", tok.decode(out))
        print("ids:", ",".join(map(str, out)))
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
