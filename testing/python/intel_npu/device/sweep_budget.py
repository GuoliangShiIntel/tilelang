"""Offline CMX feasibility sweep: lowering only, no ELF build and no device.

`lowering.lower()` runs CheckIntelNpuBudgets, so a configuration's CMX verdict costs one
lowering (~10 s) instead of a build-and-run (~4 min). Use this to find the feasible frontier,
then put only the survivors on the device.

  python sweep_budget.py --n 1024,1536,2048 --mbe 16,32,64 --db 2,4
"""
import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

_OVER = re.compile(r"peak: (\d+) B per SHAVE x (\d+) SHAVEs \+ (\d+) B shared \+ (\d+) B runtime"
                   r".*?= (\d+) B > cmx_bytes (\d+)", re.S)


def probe(cfg):
    """Lower one configuration; return (cfg, verdict, peak, limit)."""
    for p in (HERE.parents[3], HERE.parent, HERE):          # the worker does not inherit the parent's sys.path
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import numpy as np
    import tilelang  # noqa: F401
    import tvm
    import lowering
    from kernels import make_prefill_model

    n, mbe, db, tiles, shaves = cfg["n"], cfg["mbe"], cfg["db"], cfg["tiles"], cfg["shaves"]
    H, I, E, TOPK, G = 2048, 512, 256, 8, 128
    KH, VH, DK, DV, NH, NKV, D, ROT = 16, 32, 128, 128, 16, 2, 256, 64
    TB, C, TBq, TBF, BKV, BM, BNR = 32, 64, 8, 16, 128, 32, 16
    MB, BNG, BND, SMAX, BN1 = 16, cfg["bng"], cfg["bnd"], cfg["smax"], 32
    kinds = ["gdn", "attn"]

    step = int(np.lcm.reduce([TB, C, TBq, TBF, tiles * shaves, BM]))
    NP = -(-n // step) * step
    if NP > SMAX:
        return cfg, "chunk > smax", 0, 0
    try:
        with tvm.transform.PassContext(config={"tl.intel_npu.scoped_cmx": True}):
            lowering.lower(make_prefill_model(
                L=len(kinds), N=NP, H=H, KH=KH, VH=VH, DK=DK, DV=DV, NH=NH, NKV=NKV, D=D, ROT=ROT,
                SMAX=SMAX, I=I, E=E, TOPK=TOPK, G=G, TB=TB, C=C, TBq=TBq, TBF=TBF, BKV=BKV,
                BN1=BN1, BN2=BN1, MB=MB, BM=BM, BNR=BNR, MBe=mbe, BNG=BNG, BND=BND,
                DB=db, tiles=tiles, shaves=shaves, KVL=1))
        return cfg, "OK", 0, 0
    except Exception as e:                                     # the budget pass throws on overflow
        m = _OVER.search(str(e))
        if m is None:
            return cfg, f"ERR {str(e).strip().splitlines()[-1][:70]}", 0, 0
        per, nsh, shared, rt, total, limit = (int(x) for x in m.groups())
        return cfg, f"over by {total - limit} B", total, limit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1024")
    ap.add_argument("--mbe", default="16")
    ap.add_argument("--db", default="2")
    ap.add_argument("--bng", default="64")
    ap.add_argument("--bnd", default="64")
    ap.add_argument("--smax", type=int, default=1536)
    ap.add_argument("--tiles", type=int, default=3)
    ap.add_argument("--shaves", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    a = ap.parse_args()

    grid = [dict(n=n, mbe=m, db=d, bng=g, bnd=b, smax=a.smax, tiles=a.tiles, shaves=a.shaves)
            for n in (int(x) for x in a.n.split(","))
            for m in (int(x) for x in a.mbe.split(","))
            for d in (int(x) for x in a.db.split(","))
            for g in (int(x) for x in a.bng.split(","))
            for b in (int(x) for x in a.bnd.split(","))]
    print(f"{len(grid)} configurations on {a.jobs} workers\n")
    print(f"{'n':>6} {'MBe':>5} {'DB':>4} {'BNG':>5} {'BND':>5}  {'verdict':<22} {'peak B':>10}")
    print("-" * 70)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for cfg, verdict, peak, limit in ex.map(probe, grid):
            print(f"{cfg['n']:>6} {cfg['mbe']:>5} {cfg['db']:>4} {cfg['bng']:>5} {cfg['bnd']:>5}  "
                  f"{verdict:<22} {peak or '':>10}")


if __name__ == "__main__":
    main()
