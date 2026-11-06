"""Probe which hardware barrier ids a kernel can use on this device/driver.

Runs probes/barrier_probe.cpp once per id in a fresh npu_run process (an id that aborts the
run must not take the others down), on all tiles x SHAVEs, and writes
evidence/barrier_probe.json plus a summary table. The usable pool is the maximal run of
consecutive OK ids; it is what target attributes barrier_pool_first/count should hold.

Usage: python probe_barriers.py [--ids 0-47] [--tiles T] [--include-outside] [--no-build]
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))

import runner  # noqa: E402
import tilelang  # noqa: E402,F401  (sets up the tvm import path for a dev checkout)
from tvm.target import Target  # noqa: E402

KERNEL = "tl_barrier_probe"
CODES = {1: "OK", 2: "config-not-live", 3: "producer-stuck", 4: "consumer-stuck", 0: "unwritten"}


def parse_ids(spec: str):
    lo, hi = spec.split("-")
    return list(range(int(lo), int(hi) + 1))


def main(argv):
    target = Target({"kind": "intel_npu"})
    tiles, shaves = int(target.attrs["num_tiles"]), int(target.attrs["shaves_per_tile"])
    per_tile = int(target.attrs["barriers_per_tile"])
    # A job of fewer tiles: the HAS allocates barriers_per_tile per tile IN THE JOB. An id at or
    # above that range is a context violation: the launch never completes (seen 2026-11-06 with
    # --tiles 2 and ids >= 32), so such ids are probed only on request, with a short timeout.
    if "--tiles" in argv:
        tiles = int(argv[argv.index("--tiles") + 1])
    job_ids = per_tile * tiles
    ids = parse_ids(argv[argv.index("--ids") + 1]) if "--ids" in argv else list(range(0, job_ids))
    if "--include-outside" not in argv:
        outside = [b for b in ids if b >= job_ids]
        ids = [b for b in ids if b < job_ids]
        if outside:
            print(f"skipping ids outside the {tiles}-tile job's allocation [0, {job_ids}): {outside} (--include-outside to probe)")
    source = (HERE / "probes" / "barrier_probe.cpp").read_text(encoding="utf-8")
    if "--no-build" in argv:
        _, sha = runner.write_kernel(source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        db = runner.build(source, KERNEL)
    workers = tiles * shaves
    results = {}
    for bid in ids:
        rec = np.zeros((workers, 8), np.uint32)
        idw = np.array([bid], np.uint32)
        t0 = time.time()
        try:
            res = runner.run(db, [("out", rec), ("in", idw)], tiles, shaves, HERE / "run_barrier_probe", repeat=1, warmup=0, timeout_s=20)
            runner.check_status(res.status)
            codes = [int(c) for c in rec[:, 0]]
            verdict = "OK" if all(c == 1 for c in codes) else ",".join(sorted({CODES.get(c, str(c)) for c in codes}))
            results[bid] = {"verdict": verdict, "codes": codes, "live_prod": int(rec[0, 1]), "live_cons": int(rec[0, 2]),
                            "spins": int(rec[:, 5].max()), "seconds": round(time.time() - t0, 2)}
        except subprocess.TimeoutExpired:
            # The kernel never returned: the id is outside the job's allocation or otherwise fatal.
            results[bid] = {"verdict": "hang", "seconds": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001 - a faulting id must be recorded, not fatal
            msg = str(e).strip().splitlines()[-1][:160]
            results[bid] = {"verdict": "run-failed", "error": msg, "seconds": round(time.time() - t0, 2)}
        print(f"barrier {bid:2d}: {results[bid]['verdict']}" + (f"  ({results[bid].get('error')})" if "error" in results[bid] else ""))

    ok = [b for b, r in results.items() if r["verdict"] == "OK"]
    # maximal run of consecutive OK ids
    best = (0, 0)
    start = None
    for b in ids + [None]:
        if b is not None and b in ok:
            start = b if start is None else start
        else:
            if start is not None and (b or ids[-1] + 1) - start > best[1]:
                best = (start, (b if b is not None else ids[-1] + 1) - start)
            start = None
    out = HERE / "evidence" / ("barrier_probe.json" if tiles == int(target.attrs["num_tiles"]) else f"barrier_probe_{tiles}tiles.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "tiles": tiles, "shaves": shaves, "kernel_sha256": db.source_sha256,
                               "results": results, "usable_pool": {"first": best[0], "count": best[1]}}, indent=1))
    print(f"\nusable pool (longest consecutive OK run): first={best[0]} count={best[1]}; OK ids: {ok}")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
