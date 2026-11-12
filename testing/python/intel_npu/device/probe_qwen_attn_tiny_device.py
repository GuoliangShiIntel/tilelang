"""Decisive test: does the PROVEN (per device/evidence/INDEX.md, extensively PASSING on real device)
Qwen3.6/3.8 `kernels.make_attn_prefill` flash-attention kernel ALSO fail at a tiny shape on THIS
NVL/NPU6 machine? If yes, this is a general small-shape-on-NVL gap, not Gemma4-specific. If it PASSES,
the bug is specific to Gemma4's kernel code (gemma4_kernels.make_gemma4_attn_prefill).
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

N, NKV, GQ, D, SMAX, TB, BKV = 8, 2, 2, 32, 64, 4, 32
TILES, SHAVES = 2, 2
POS0 = 16


def main(argv):
    func = make_attn_prefill(N=N, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, TB=TB, BKV=BKV, tiles=TILES, shaves=SHAVES)
    source = tilelang.lower(func, target="intel_npu").kernel_source
    db = runner.build(source, "tl_attn_qwen_tiny_nvl")

    rng = np.random.default_rng(7)
    q = rng.standard_normal((NKV, N, GQ, D)).astype(np.float16)
    kc = rng.standard_normal((NKV, SMAX, D)).astype(np.float16)
    vt = rng.standard_normal((NKV, D, SMAX)).astype(np.float16)
    gate = rng.standard_normal((N, NKV * GQ * D)).astype(np.float16)
    pos = np.array([[POS0]], np.int32)
    o = np.full((N, NKV * GQ * D), -123.0, np.float16)

    res = runner.run(db, [("in", q), ("in", kc), ("in", vt), ("in", gate), ("in", pos), ("out", o)],
                     TILES, SHAVES, HERE / "run_attn_qwen_tiny_nvl")
    print(f"status per SHAVE: {res.status[:, :2].tolist()}")
    runner.check_status(res.status)

    ref = attn_prefill_reference(q, kc, vt, gate, POS0)
    err = float(np.abs(o.astype(np.float32) - ref.astype(np.float32)).max())
    rel = err / float(np.abs(ref).max())
    nonzero = int(np.count_nonzero(o))
    print(f"Qwen attn_prefill tiny shape on NVL: rel_err={rel:.4e} nonzero={nonzero}/{o.size} "
         f"verdict={'PASS' if rel < 5e-2 else 'FAIL'}")
    return 0 if rel < 5e-2 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
