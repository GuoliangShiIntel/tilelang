"""Follow-up to probe_qwen_attn_tiny_device.py's DEVICE_LOST hang: that test used BKV=32, D=32,
SMAX=64, dimensions smaller than ANY shape in device/evidence/INDEX.md (which documents extensive
real-device PASSES only at BKV>=128, D>=64, SMAX>=1024). This bumps BKV/D/SMAX toward the smallest
previously-PROVEN-on-device combination while keeping N (token count) small/fast to build, to find
whether "small BKV/D" (untested regime) or "NVL generally" is the actual boundary.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402
import runner  # noqa: E402
from kernels import make_attn_prefill, attn_prefill_reference  # noqa: E402
import numpy as np  # noqa: E402

TILES, SHAVES = 2, 2


def run_shape(name, N, NKV, GQ, D, SMAX, TB, BKV, pos0):
    func = make_attn_prefill(N=N, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, TB=TB, BKV=BKV, tiles=TILES, shaves=SHAVES)
    source = tilelang.lower(func, target="intel_npu").kernel_source
    db = runner.build(source, name)
    rng = np.random.default_rng(7)
    q = rng.standard_normal((NKV, N, GQ, D)).astype(np.float16)
    kc = rng.standard_normal((NKV, SMAX, D)).astype(np.float16)
    vt = rng.standard_normal((NKV, D, SMAX)).astype(np.float16)
    gate = rng.standard_normal((N, NKV * GQ * D)).astype(np.float16)
    pos = np.array([[pos0]], np.int32)
    o = np.full((N, NKV * GQ * D), -123.0, np.float16)
    res = runner.run(db, [("in", q), ("in", kc), ("in", vt), ("in", gate), ("in", pos), ("out", o)],
                     TILES, SHAVES, HERE / f"run_{name}")
    runner.check_status(res.status)
    ref = attn_prefill_reference(q, kc, vt, gate, pos0)
    err = float(np.abs(o.astype(np.float32) - ref.astype(np.float32)).max())
    rel = err / float(np.abs(ref).max())
    nonzero = int(np.count_nonzero(o))
    verdict = "PASS" if rel < 5e-2 else "FAIL"
    print(f"{name}: N={N} NKV={NKV} D={D} SMAX={SMAX} BKV={BKV} -> rel_err={rel:.4e} "
         f"nonzero={nonzero}/{o.size} verdict={verdict}")
    return verdict == "PASS"


def main(argv):
    results = {}
    try:
        results["BKV128_D64"] = run_shape("tl_attn_bkv128_d64", N=8, NKV=2, GQ=2, D=64, SMAX=256, TB=4, BKV=128, pos0=16)
    except RuntimeError as e:
        print(f"BKV128_D64: EXCEPTION {e}")
        results["BKV128_D64"] = "EXC"
    try:
        # N=96 matches device/evidence/INDEX.md's documented, extensively-PASSING-on-real-device
        # flash shape (Sq=96, Skv=1024-ish, d=64); this is the cleanest "replicate known-good
        # shape on NVL hardware" test to isolate whether N (token count) is the real factor.
        results["N96_proven_shape"] = run_shape("tl_attn_n96_proven", N=96, NKV=2, GQ=8, D=64, SMAX=1024, TB=4, BKV=128, pos0=0)
    except RuntimeError as e:
        print(f"N96_proven_shape: EXCEPTION {e}")
        results["N96_proven_shape"] = "EXC"
    print("\nSUMMARY:", results)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
