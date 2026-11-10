"""Packs a checkpoint's layers into the decode kernels' parameter arrays (kernels.pack_gdn / pack_attn / pack_moe /
pack_head), cached as .npy files per layer under a pack directory so a model loads in seconds after the first time.

The layout is packed values plus a separate group-major scale tensor, which prefill reads too, so `GEOM` now only
decides the row PADDING, not the bytes: a sweep over chunk rows no longer invalidates the 17.91 GB of weights.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from kernels import pack_attn, pack_gdn, pack_head, pack_moe  # noqa: E402
from checkpoint import Checkpoint, layer_prefix  # noqa: E402

TILES, SHAVES = 3, 2
GEOM = dict(gdn_BN1=128, gdn_BN2=64, attn_BN1=128, attn_BN2=64, moe_BNR=16, moe_BNG=128, moe_BND=256, head_BN=128)
# Only these reach the packed BYTES, and only as the row padding. The rest are pure scheduling now that the scales
# have left the weight tensor, so a chunk-size sweep (the usual one: moe_BNG / moe_BND) reuses the cache instead of
# re-packing 17.91 GB at ~8 s a layer. `test_intel_npu_e2e_pack_geometry` holds this set to the bytes.
PAD_KEYS = ("gdn_BN1", "gdn_BN2", "attn_BN2", "moe_BNR", "head_BN")
_GEOM_DEFAULTS = dict(GEOM)
GEOM_TAG = "g2"   # g1 was the chunk-block layout; a g1 cache is not readable by these kernels
if __import__("os").environ.get("TL_NPU_GEOM"):   # sweeps: "moe_BNG=256,moe_BND=512"
    for _kv in __import__("os").environ["TL_NPU_GEOM"].split(","):
        _k, _v = _kv.split("=")
        assert _k in GEOM, _k
        GEOM[_k] = int(_v)
    _diff = {k: GEOM[k] for k in PAD_KEYS if GEOM[k] != _GEOM_DEFAULTS[k]}
    if _diff:
        GEOM_TAG += "_" + "_".join(f"{k}{v}" for k, v in sorted(_diff.items()))


def pack_gdn_layer(ck: Checkpoint, i: int) -> dict:
    c = ck.cfg
    p = layer_prefix(i) + "linear_attn."
    qkv_v, qkv_s, _ = ck.quantized(p + "in_proj_qkv")
    z_v, z_s, _ = ck.quantized(p + "in_proj_z")
    out_v, out_s, _ = ck.quantized(p + "out_proj")
    wba = np.concatenate([ck.f32(p + "in_proj_b.weight"), ck.f32(p + "in_proj_a.weight")], axis=0)   # [2 VH, H]
    packed = pack_gdn(np.concatenate([qkv_v, z_v]), np.concatenate([qkv_s, z_s]), out_v, out_s, wba=wba,
                      tiles=TILES, shaves=SHAVES, BN1=GEOM["gdn_BN1"], BN2=GEOM["gdn_BN2"])
    conv = ck.f32(p + "conv1d.weight").reshape(c.conv_dim, c.conv_k)       # [C, 1, K] -> [C, K], tap 0 the oldest
    packed["ConvW"] = np.ascontiguousarray(conv.T).astype(np.float16)       # [K, C] time-major
    packed["ALog"] = ck.f32(p + "A_log").reshape(1, c.lin_vh).astype(np.float32)
    packed["DtBias"] = ck.f32(p + "dt_bias").reshape(1, c.lin_vh).astype(np.float32)
    packed["GNorm"] = ck.f16(p + "norm.weight").reshape(1, c.lin_dv)
    return packed


def pack_attn_layer(ck: Checkpoint, i: int) -> dict:
    c = ck.cfg
    p = layer_prefix(i) + "self_attn."
    q_v, q_s, _ = ck.quantized(p + "q_proj")       # [NH x 2D, H]: per head [q | gate]
    k_v, k_s, _ = ck.quantized(p + "k_proj")
    v_v, v_s, _ = ck.quantized(p + "v_proj")
    o_v, o_s, _ = ck.quantized(p + "o_proj")
    packed = pack_attn(np.concatenate([q_v, k_v, v_v]), np.concatenate([q_s, k_s, v_s]), o_v, o_s,
                       tiles=TILES, BN1=GEOM["attn_BN1"], BN2=GEOM["attn_BN2"])
    packed["QNormW"] = ck.f16(p + "q_norm.weight").reshape(1, c.head_dim)
    packed["KNormW"] = ck.f16(p + "k_norm.weight").reshape(1, c.head_dim)
    return packed


def pack_moe_layer(ck: Checkpoint, i: int) -> dict:
    c = ck.cfg
    p = layer_prefix(i) + "mlp."
    E, I, H, G = c.n_experts, c.moe_inter, c.hidden, ck.group
    assert c.shared_inter == I
    gu_int = np.empty((E + 1, 2 * I, H), np.int8)
    gu_s = np.empty((E + 1, 2 * I, H // G), np.float16)
    d_int = np.empty((E + 1, H, I), np.int8)
    d_s = np.empty((E + 1, H, I // G), np.float16)
    for e in range(E + 1):
        q = f"{p}experts.{e}." if e < E else f"{p}shared_expert."
        g_v, g_sc, _ = ck.quantized(q + "gate_proj")
        u_v, u_sc, _ = ck.quantized(q + "up_proj")
        dv, dsc, _ = ck.quantized(q + "down_proj")
        gu_int[e, :I], gu_int[e, I:] = g_v, u_v
        gu_s[e, :I], gu_s[e, I:] = g_sc, u_sc
        d_int[e], d_s[e] = dv, dsc
    router = np.concatenate([ck.f32(p + "gate.weight"), ck.f32(p + "shared_expert_gate.weight")], axis=0).astype(np.float16)
    return pack_moe(router, gu_int, gu_s, d_int, d_s, tiles=TILES, shaves=SHAVES, BNR=GEOM["moe_BNR"], BNG=GEOM["moe_BNG"],
                    BND=GEOM["moe_BND"], G=G)


def pack_layer(ck: Checkpoint, i: int) -> dict:
    """Every device array of layer i (mixer + MoE), plus the two norms (host side for now) as fp32 [H]."""
    p = layer_prefix(i)
    out = pack_gdn_layer(ck, i) if ck.cfg.layer_types[i] == "linear_attention" else pack_attn_layer(ck, i)
    out.update(pack_moe_layer(ck, i))
    out["in_norm"] = ck.f32(p + "input_layernorm.weight")
    out["post_norm"] = ck.f32(p + "post_attention_layernorm.weight")
    return out


def pack_head_weights(ck: Checkpoint) -> dict:
    v, s, bits = ck.quantized("lm_head")
    assert bits == 8, bits
    out = pack_head(v, s, tiles=TILES, shaves=SHAVES, BN=GEOM["head_BN"])
    out["final_norm"] = ck.f32("model.language_model.norm.weight")
    return out


def cached_paths(pack_dir: Path, key: str, fn, log=print) -> dict:
    """{array name: .npy path} under pack_dir/key, packed by `fn()` on a miss (a manifest marks completeness)."""
    d = pack_dir / key
    manifest = d / "manifest.json"
    if not manifest.exists():
        t0 = time.time()
        arrays = fn()
        d.mkdir(parents=True, exist_ok=True)
        for n, a in arrays.items():
            np.save(d / f"{n}.npy", np.ascontiguousarray(a))
        tmp = d / "manifest.json.tmp"
        tmp.write_text(json.dumps({"arrays": list(arrays), "geom": GEOM}), encoding="utf-8")
        tmp.replace(manifest)
        log(f"packed {key} in {time.time() - t0:.0f} s ({sum(a.nbytes for a in arrays.values()) / 1e6:.0f} MB)")
        del arrays
    m = json.loads(manifest.read_text(encoding="utf-8"))
    if "geom" not in m:   # a manifest with no geometry cannot be checked, so it is refused rather than trusted
        raise RuntimeError(f"{d}: manifest records no geometry; delete the directory and repack")
    stale = {k: (m["geom"].get(k), GEOM[k]) for k in PAD_KEYS if m["geom"].get(k) != GEOM[k]}
    if stale:   # the key dropped a value that does reach the bytes: refuse rather than serve the wrong padding
        raise RuntimeError(f"{d}: cached under a different padding geometry {stale} (cached, wanted); "
                           f"PAD_KEYS and GEOM_TAG disagree -- delete the directory or fix the key")
    return {n: d / f"{n}.npy" for n in m["arrays"]}


def cached(pack_dir: Path, key: str, fn, log=print) -> dict:
    """The arrays themselves (loads every .npy of `cached_paths`)."""
    return {n: np.load(p) for n, p in cached_paths(pack_dir, key, fn, log).items()}


def layer_key(i: int) -> str:
    return f"{GEOM_TAG}_layer{i:02d}"


def head_key() -> str:
    return f"{GEOM_TAG}_head"


def main(argv):
    """python e2e/pack.py --ckpt DIR --pack-dir DIR [--layers N]: pack ahead of time."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args(argv)
    ck = Checkpoint(a.ckpt)
    n = a.layers if a.layers is not None else ck.cfg.layers
    for i in range(n):
        cached(Path(a.pack_dir), layer_key(i), lambda: pack_layer(ck, i))
    cached(Path(a.pack_dir), head_key(), lambda: pack_head_weights(ck))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
