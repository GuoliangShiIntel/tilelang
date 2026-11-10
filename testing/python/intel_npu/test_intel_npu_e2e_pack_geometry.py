"""`pack.PAD_KEYS` held to the packed bytes.

The cache key `pack.GEOM_TAG` only names the geometry parameters in `PAD_KEYS`. That is what lets a chunk-size
sweep reuse the 17.91 GB of packed weights instead of re-packing at ~8 s a layer -- but it is only safe while the
other parameters genuinely do not reach the bytes. They do not today because the scales left the weight tensor:
`BNG` / `BND` / `attn_BN1` pick DMA chunk sizes, and the packers no longer see them. If that ever changes, the
cache would silently serve the wrong padding, which is worse than a slow repack.

So this compares SETS, not counts: double each geometry value in turn, re-pack, and collect the keys whose bytes
moved. That set must equal `PAD_KEYS` exactly -- a key that starts mattering fails here, and a key wrongly listed
in `PAD_KEYS` fails here too. The shapes are chosen so that doubling a padding key really does change the padded
row count (otherwise the test would pass by accident).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "e2e"))
import pack as e2e_pack  # noqa: E402
from kernels import pack_attn, pack_gdn, pack_head, pack_moe  # noqa: E402

TILES, SHAVES, G = 3, 2, 16
H, NIN, NOUT, V = 64, 100, 100, 100
E, I2 = 4, 32          # E + 1 experts, 2I columns of gate/up


def _problem(seed=0):
    rng = np.random.default_rng(seed)

    def q4(n, k):
        return rng.integers(-8, 8, (n, k), dtype=np.int8), (rng.standard_normal((n, k // G)) * 0.1).astype(np.float16)

    p = {}
    p["win"], p["win_s"] = q4(NIN, H)
    p["wout"], p["wout_s"] = q4(NOUT, H)
    p["wqkv"], p["wqkv_s"] = q4(NIN, H)
    p["wo"], p["wo_s"] = q4(NOUT, H)
    p["router"] = (rng.standard_normal((E + 1, H)) * 0.1).astype(np.float16)
    p["gu"] = rng.integers(-8, 8, (E + 1, I2, H), dtype=np.int8)
    p["gu_s"] = (rng.standard_normal((E + 1, I2, H // G)) * 0.1).astype(np.float16)
    p["d"] = rng.integers(-8, 8, (E + 1, H, I2), dtype=np.int8)
    p["d_s"] = (rng.standard_normal((E + 1, H, I2 // G)) * 0.1).astype(np.float16)
    p["wlm"] = rng.integers(-128, 127, (V, H), dtype=np.int8)
    p["wlm_s"] = (rng.standard_normal((V, H // G)) * 0.1).astype(np.float16)
    return p


def _pack_all(p, geom):
    """Every packer driven by one geometry dict, as `pack.pack_layer` + `pack.pack_head_weights` do."""
    out = {}
    out.update({f"gdn_{k}": v for k, v in pack_gdn(p["win"], p["win_s"], p["wout"], p["wout_s"], tiles=TILES,
                                                   shaves=SHAVES, BN1=geom["gdn_BN1"], BN2=geom["gdn_BN2"]).items()})
    out.update({f"attn_{k}": v for k, v in pack_attn(p["wqkv"], p["wqkv_s"], p["wo"], p["wo_s"], tiles=TILES,
                                                     BN1=geom["attn_BN1"], BN2=geom["attn_BN2"]).items()})
    out.update({f"moe_{k}": v for k, v in pack_moe(p["router"], p["gu"], p["gu_s"], p["d"], p["d_s"], tiles=TILES,
                                                   shaves=SHAVES, BNR=geom["moe_BNR"], BNG=geom["moe_BNG"],
                                                   BND=geom["moe_BND"], G=G).items()})
    out.update({f"head_{k}": v for k, v in pack_head(p["wlm"], p["wlm_s"], tiles=TILES, shaves=SHAVES,
                                                     BN=geom["head_BN"]).items()})
    return out


def _same(a, b):
    return set(a) == set(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype
                                    and np.array_equal(a[k], b[k]) for k in a)


def test_pad_keys_is_exactly_the_set_that_reaches_the_bytes():
    p = _problem()
    base_geom = dict(e2e_pack.GEOM)
    base = _pack_all(p, base_geom)
    moved = set()
    for k in base_geom:
        g = dict(base_geom)
        g[k] = base_geom[k] * 2
        if not _same(base, _pack_all(p, g)):
            moved.add(k)
    assert moved == set(e2e_pack.PAD_KEYS), (
        f"keys that change the packed bytes: {sorted(moved)}; PAD_KEYS: {sorted(e2e_pack.PAD_KEYS)}. "
        f"Extra in PAD_KEYS means a needless repack; missing means the cache serves the wrong padding.")


def test_the_detector_can_see_a_change():
    """Positive control: the comparison above must be capable of reporting a difference at all."""
    p = _problem()
    base = _pack_all(p, dict(e2e_pack.GEOM))
    perturbed = dict(base)
    k = sorted(base)[0]
    perturbed[k] = base[k].copy()
    perturbed[k].reshape(-1)[0] += 1
    assert not _same(base, perturbed)
    assert _same(base, dict(base))


def test_a_chunk_size_sweep_keeps_the_cache_key():
    """The point of the split: `moe_BNG` / `moe_BND` are DMA chunk sizes now, so a sweep over them must land on the
    same cache directory -- otherwise every sweep point repacks 17.91 GB."""
    assert "moe_BNG" not in e2e_pack.PAD_KEYS and "moe_BND" not in e2e_pack.PAD_KEYS
    p = _problem()
    g = dict(e2e_pack.GEOM, moe_BNG=256, moe_BND=512)
    assert _same(_pack_all(p, dict(e2e_pack.GEOM)), _pack_all(p, g))


def test_a_cache_packed_under_another_padding_is_refused(tmp_path):
    """The cache key names only `PAD_KEYS`, so the manifest is the last line of defence -- and a gate is only
    evidence once it has been SEEN to fail. All three arms run here: a matching geometry is served, a differing
    one raises, and a manifest too old to carry a geometry raises instead of being trusted."""
    import json

    def never(*_):
        raise AssertionError("repacked a directory whose manifest exists")

    d = tmp_path / "L0"
    d.mkdir()

    def manifest(geom):
        body = {"arrays": ["w"]} if geom is None else {"arrays": ["w"], "geom": geom}
        (d / "manifest.json").write_text(json.dumps(body), encoding="utf-8")

    manifest(dict(e2e_pack.GEOM))
    assert e2e_pack.cached_paths(tmp_path, "L0", never, log=never) == {"w": d / "w.npy"}

    manifest(dict(e2e_pack.GEOM, **{e2e_pack.PAD_KEYS[0]: e2e_pack.GEOM[e2e_pack.PAD_KEYS[0]] * 2}))
    with pytest.raises(RuntimeError, match="different padding geometry"):
        e2e_pack.cached_paths(tmp_path, "L0", never, log=never)

    manifest(None)
    with pytest.raises(RuntimeError, match="records no geometry"):
        e2e_pack.cached_paths(tmp_path, "L0", never, log=never)
