"""Decode a Qwen3.5-MoE-family model on the NPU with this backend's decode kernels, one launch per block
(mixer, MoE, head), weights resident in one `npu_run --session`, the residual stream in fp32 on the host and the two
RMSNorms per layer on the host for now (M5 moves both into the kernels). The prompt is run token by token through the
same decode kernels (correct, slow: prefill kernels are a later milestone).

    python testing/python/intel_npu/e2e/decode.py --ckpt D:\\models\\... --pack-dir D:\\models\\..._tlpack
        [--layers N] [--prompt "text" | --ids 1,2,3] [--new 16] [--smax 1536] [--golden golden.npz] [--no-build]

--golden FILE: the fp32 reference's saved answer for the same prompt (`e2e/reference_golden.py`, run separately: the
reference and 18 GB of pinned weights do not fit one 32 GB box together); the hidden state after every layer of the
first token and the argmax of every step are compared, teacher-forced (the device is fed the reference's tokens). Timing
is UNPROFILED, TURBO unless TL_NPU_TURBO=0; the per-token number is the sum of the launches' kernel-only spans plus the
host wall of the whole step (file transfers included).
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
from kernels import attn_sizes, gdn_sizes, head_sizes, make_attn_decode, make_gdn_decode, make_head, make_moe_decode, rope_cos_sin  # noqa: E402
from checkpoint import Checkpoint, LM  # noqa: E402
import pack  # noqa: E402

TILES, SHAVES = 3, 2


def rmsnorm_zc(h: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return (h * (1.0 / np.sqrt((h * h).mean() + eps)) * (1.0 + w)).astype(np.float16)


class DeviceModel:
    def __init__(self, ck: Checkpoint, pack_dir: Path, layers: int, smax: int, build: bool, log=print):
        self.ck, self.c, self.log = ck, ck.cfg, log
        c = self.c
        self.L = layers
        self.smax = smax
        G = pack.GEOM
        kernels = {
            "gdn": make_gdn_decode(H=c.hidden, KH=c.lin_kh, VH=c.lin_vh, DK=c.lin_dk, DV=c.lin_dv, G=ck.group, tiles=TILES,
                                   shaves=SHAVES, BN1=G["gdn_BN1"], BN2=G["gdn_BN2"], eps=c.eps),
            "attn": make_attn_decode(H=c.hidden, NH=c.n_heads, NKV=c.n_kv, D=c.head_dim, ROT=c.rot_dim, SMAX=smax, G=ck.group,
                                     tiles=TILES, shaves=SHAVES, BN1=G["attn_BN1"], BN2=G["attn_BN2"], eps=c.eps),
            "moe": make_moe_decode(H=c.hidden, I=c.moe_inter, E=c.n_experts, TOPK=c.top_k, G=ck.group, tiles=TILES, shaves=SHAVES,
                                   BNR=G["moe_BNR"], BNG=G["moe_BNG"], BND=G["moe_BND"]),
            "head": make_head(H=c.hidden, V=c.vocab, G=ck.group, tiles=TILES, shaves=SHAVES, BN=G["head_BN"], eps=c.eps),
        }
        names = {"gdn": f"tl_e2e_gdn_h{c.hidden}", "attn": f"tl_e2e_attn_h{c.hidden}_s{smax}", "moe": f"tl_e2e_moe_h{c.hidden}",
                 "head": f"tl_e2e_head_h{c.hidden}_v{c.vocab}"}
        self.low = {k: lowering.lower(f) for k, f in kernels.items()}
        if build:
            t0 = time.time()
            runner.build_all({names[k]: self.low[k].source for k in kernels}, log=log)
            log(f"build {time.time() - t0:.0f} s")
        else:
            for k in kernels:
                runner.write_kernel(self.low[k].source, names[k])
        exe = runner.HERE / "build" / "Release" / "npu_run.exe"
        self.s = runner.Session(exe, runner.HERE / "run_e2e", TILES, SHAVES)
        for k, n in names.items():
            self.s.kernel(k, n)
        self.pack_dir = Path(pack_dir)
        self.embed = ck.raw(LM + "embed_tokens.weight")   # bf16 [V, H], rows taken on demand
        self._upload_all()

    # ---- buffers ------------------------------------------------------------------------------------------------
    def _upload_all(self):
        c, s = self.c, self.s
        H = c.hidden
        t0 = time.time()
        total = 0
        s.upload("Zero", np.zeros((1, H), np.float16))
        s.upload("X", np.zeros((1, H), np.float16))
        s.upload("Pos", np.zeros((1, 1), np.int32))
        s.upload("CosSin", np.zeros((2, c.rot_dim), np.float32))
        gsz = gdn_sizes(H, c.lin_kh, c.lin_vh, c.lin_dk, tiles=TILES, shaves=SHAVES, BN1=pack.GEOM["gdn_BN1"], BN2=pack.GEOM["gdn_BN2"])
        asz = attn_sizes(H, c.n_heads, c.n_kv, c.head_dim, tiles=TILES, BN2=pack.GEOM["attn_BN2"])
        hsz = head_sizes(H, c.vocab, tiles=TILES, shaves=SHAVES, BN=pack.GEOM["head_BN"])
        self.gsz, self.asz, self.hsz = gsz, asz, hsz
        # scratch shared by every layer of a kind
        for n, shape, dt in [("G_Y", (1, gsz["YLEN"]), np.float16), ("G_BA", (1, gsz["BA_LEN"]), np.float16), ("G_CV", (1, gsz["CVLEN"]), np.float16),
                             ("G_O", (1, gsz["OPAD"]), np.float16), ("G_Out", (1, gsz["NOUT_pad"]), np.float16),
                             ("A_Y", (1, asz["NQKV"]), np.float16), ("A_QN", (c.n_kv, asz["GQ"], c.head_dim), np.float16),
                             ("A_Part", (c.n_kv, asz["GQ"], TILES, asz["PW"]), np.float32), ("A_O", (1, asz["KO"]), np.float16),
                             ("A_Out", (1, asz["NOUT_pad"]), np.float16),
                             ("M_Part", (TILES, H), np.float32), ("M_Out", (1, H), np.float16),
                             ("H_Hs", (1, H), np.float16), ("H_Logits", (TILES, hsz["CS"], SHAVES * pack.GEOM["head_BN"]), np.float16),
                             ("H_Part", (hsz["workers"], 32), np.float32), ("H_Tok", (1, 8), np.int32)]:
            s.alloc(n, int(np.prod(shape)) * np.dtype(dt).itemsize)
        self.layer_w = []
        for i in range(self.L):
            paths = pack.cached_paths(self.pack_dir, pack.layer_key(i), lambda: pack.pack_layer(self.ck, i), log=self.log)
            self.layer_w.append({k: np.load(paths[k]) for k in ("in_norm", "post_norm")})
            for k, p in paths.items():
                if k in ("in_norm", "post_norm"):
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
        self.log(f"uploaded {self.L} layers + head: {total / 1e9:.2f} GB in {time.time() - t0:.0f} s")

    def _router_rows(self) -> int:
        c = self.c
        per_round = TILES * SHAVES * pack.GEOM["moe_BNR"]
        return -(-(c.n_experts + 1) // per_round) * per_round

    def reset_state(self):
        for i in range(self.L):
            for k in (("ConvState", "Ssm") if self.c.layer_types[i] == "linear_attention" else ("Kc", "Vt")):
                self.s.zero(f"L{i}_{k}")
        self.pos = 0

    # ---- one token ----------------------------------------------------------------------------------------------
    def _bufs(self, kind, i):
        """Buffer names in the kernel's device parameter order."""
        c = self.c
        if kind == "gdn":
            m = dict(X="X", Res="Zero", WinP=f"L{i}_WinP", Wba=f"L{i}_Wba", ConvW=f"L{i}_ConvW", ALog=f"L{i}_ALog", DtBias=f"L{i}_DtBias",
                     GNorm=f"L{i}_GNorm", ConvState=f"L{i}_ConvState", Ssm=f"L{i}_Ssm", WoutP=f"L{i}_WoutP", Y="G_Y", BA="G_BA",
                     CV="G_CV", O="G_O", Out="G_Out")
        elif kind == "attn":
            m = dict(X="X", Res="Zero", WqkvP=f"L{i}_WqkvP", QNormW=f"L{i}_QNormW", KNormW=f"L{i}_KNormW", CosSin="CosSin", Pos="Pos",
                     Kc=f"L{i}_Kc", Vt=f"L{i}_Vt", WoP=f"L{i}_WoP", Y="A_Y", QN="A_QN", Part="A_Part", O="A_O", Out="A_Out")
        elif kind == "moe":
            m = dict(X="X", Wr=f"L{i}_Wr", WguP=f"L{i}_WguP", WdP=f"L{i}_WdP", Res="Zero", Logits="M_Logits", Part="M_Part", Out="M_Out")
        else:
            m = dict(Hs="H_Hs", NormW="H_NormW", WlmP="H_WlmP", Logits="H_Logits", Part="H_Part", Tok="H_Tok")
        return [m[n] for n in self.low[kind].param_names]

    def step(self, token: int, trace: list | None = None) -> tuple[int, dict]:
        """Runs `token` at the current position through every layer and the head; returns (next token, timing)."""
        c, s = self.c, self.s
        H = c.hidden
        import torch
        h = self.embed[token].to(torch.float32).numpy().copy()
        t0 = time.time()
        dev_ms = dict(mixer=0.0, moe=0.0, head=0.0)
        s.upload("Pos", np.array([[self.pos]], np.int32))
        s.upload("CosSin", rope_cos_sin(self.pos, c.rot_dim, c.rope_theta))
        for i in range(self.L):
            w = self.layer_w[i]
            s.upload("X", rmsnorm_zc(h, w["in_norm"], c.eps).reshape(1, H))
            if c.layer_types[i] == "linear_attention":
                _, km = s.launch("gdn", *self._bufs("gdn", i))
                mix = s.download("G_Out", np.float16, (1, self.gsz["NOUT_pad"]))[0, :H].astype(np.float32)
            else:
                _, km = s.launch("attn", *self._bufs("attn", i))
                mix = s.download("A_Out", np.float16, (1, self.asz["NOUT_pad"]))[0, :H].astype(np.float32)
            dev_ms["mixer"] += km
            h = h + mix
            s.upload("X", rmsnorm_zc(h, w["post_norm"], c.eps).reshape(1, H))
            _, km = s.launch("moe", *self._bufs("moe", i))
            dev_ms["moe"] += km
            m = s.download("M_Out", np.float16, (1, H))[0].astype(np.float32)
            h = h + m
            if trace is not None:
                trace.append(dict(layer=i, mix=mix, moe=m, out=h.copy()))
        s.upload("H_Hs", h.astype(np.float16).reshape(1, H))
        _, km = s.launch("head", *self._bufs("head", 0))
        dev_ms["head"] += km
        tok = int(s.download("H_Tok", np.int32, (1, 8))[0, 0])
        self.pos += 1
        dev_ms["host_wall"] = (time.time() - t0) * 1e3
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
    print(f"prompt: {len(ids)} tokens; layers {layers} of {ck.cfg.layers}; new {a.new}")
    m = DeviceModel(ck, Path(a.pack_dir), layers, a.smax, build=not a.no_build)
    try:
        m.reset_state()
        if gold is not None:
            t0 = time.time()
            dev_trace = []
            dev_tok, ms = m.step(ids[0], trace=dev_trace)
            print("layer-by-layer, first prompt token: max |h_dev - h_ref| / max |h_ref| (mixer contribution, MoE contribution):")
            worst = 0.0
            for i in range(layers):
                dr = dev_trace[i]
                ref_out, ref_mix, ref_moe = gold["trace_out"][i], gold["trace_mix"][i], gold["trace_moe"][i]
                rel_h = np.abs(dr["out"] - ref_out).max() / np.abs(ref_out).max()
                rel_mix = np.abs(dr["mix"] - ref_mix).max() / max(np.abs(ref_mix).max(), 1e-6)
                rel_moe = np.abs(dr["moe"] - ref_moe).max() / max(np.abs(ref_moe).max(), 1e-6)
                worst = max(worst, rel_h)
                print(f"  layer {i:2d} {ck.cfg.layer_types[i][:6]}: h rel {rel_h:.2e}  mixer rel {rel_mix:.2e}  moe rel {rel_moe:.2e}")
            print(f"  worst h rel over the layers: {worst:.2e}")
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
            print(f"golden: {agree}/{len(out_ids)} greedy tokens agree with the reference (teacher-forced; {time.time() - t0:.0f} s)")
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
            print(f"per token: device launches sum median {dev[len(dev) // 2]:.1f} ms (mixer {med['mixer']:.1f} + moe {med['moe']:.1f} + head "
                  f"{med['head']:.1f} of the median step), host wall median {wall[len(wall) // 2]:.0f} ms [{cond}]")
        return 0
    finally:
        m.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
