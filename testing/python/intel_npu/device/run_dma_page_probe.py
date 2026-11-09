"""Plan question 11: which DMA stores reach DDR when several tiles write one page -- a probe that came back CLEAN.

`kernels.make_dma_page_probe` is the GDN conv phase's state store with every computation removed: the same 2-D
job (ROWS rows of DK halves at stride CONV*2), the same addresses (slots 0..KH-1 all inside the first 4 KiB of a
row), the same outer loop that the model kernel's layer loop provides. Nothing else is left, so whatever this
shows is about the stores alone.

**It shows nothing: all six arms are clean** (2026-11-09; tiles 1/2/3 x {round-robin, blocked}, with and without
`--rw`). The hypothesis it was built to test -- three tiles writing interleaved 256 B pieces of one 4 KiB DDR
page, read-modify-write, over several rounds -- is therefore **not sufficient** to lose a store. Something this
probe leaves out is necessary, and the model kernel's `--skip moe` arm says the MoE phases are part of it. The
file is kept as the skeleton for the next version, which adds that load instead of re-deriving this one.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH set):
    python testing/python/intel_npu/device/run_dma_page_probe.py [--tiles 3] [--rounds 2] [--blocked]
                                                                 [--shave-major] [--rows 3] [--trials 3]
                                                                 [--rw] [--sweep]
`--sweep` runs the grid that matters: tiles 1..3 x {round-robin, blocked} at the default rounds.

The buffer is pre-filled with -1, and round r writes `slot + 1 + 100 * r`, so the report separates a LOST store
(an earlier round's value, or -1) from a wrong one and says which round lost it.
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
import runner  # noqa: E402
import lowering  # noqa: E402
from kernels import dma_page_probe_sizes, make_dma_page_probe  # noqa: E402

SHAVES = 2
KH, VH, DK = 16, 32, 128


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def one(argv, tiles: int, blocked: bool, rounds: int, rows: int, trials: int, build: bool) -> int:
    sm = "--shave-major" in argv
    sz = dma_page_probe_sizes(KH=KH, VH=VH, DK=DK, tiles=tiles, shaves=SHAVES)
    CONV, NSLOT = sz["CONV"], sz["NSLOT"]
    low = lowering.lower(make_dma_page_probe(KH=KH, VH=VH, DK=DK, ROWS=rows, rounds=rounds, blocked=blocked,
                                             item_worker="shave" if sm else "tile", rw="--rw" in argv,
                                             tiles=tiles, shaves=SHAVES))
    name = (f"tl_dma_page_probe_t{tiles}_r{rounds}_rows{rows}"
            + ("_blocked" if blocked else "_rr") + ("_sm" if sm else "") + ("_rw" if "--rw" in argv else ""))
    if build:
        runner.build_all({name: low.source})
    else:
        runner.write_kernel(low.source, name)
    db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe",
                            kernel=name, source_sha256="")

    bad_total = 0
    for _ in range(trials):
        dst = np.full((rows, CONV), -1.0, np.float16)
        res = runner.run(db, [("inout", dst)], tiles, SHAVES, HERE / "run_dma_page_probe", timeout_s=120)
        runner.check_status(res.status)
        got = dst.astype(np.float32)
        lost, wrong = [], []
        for s in range(NSLOT):
            cb = s * DK if s < 2 * KH else 2 * KH * DK + (s - 2 * KH) * DK
            block = got[:, cb:cb + DK]
            want = s + 1 + 100 * (rounds - 1)
            if np.all(block == want):
                continue
            vals = sorted({float(v) for v in np.unique(block)})
            # an earlier round's value or the -1 pre-fill means the LAST round's store never landed
            if all(v == -1.0 or (v - s - 1) % 100 == 0 for v in vals):
                lost.append((s, vals))
            else:
                wrong.append((s, vals))
        bad_total += len(lost) + len(wrong)
        if lost or wrong:
            q = [s for s, _ in lost if s < KH]
            k = [s for s, _ in lost if KH <= s < 2 * KH]
            v = [s for s, _ in lost if s >= 2 * KH]
            print(f"    lost {len(lost)} (q {len(q)}, k {len(k)}, v {len(v)}), wrong {len(wrong)}"
                  + (f"; first lost slot {lost[0][0]} holds {lost[0][1]}" if lost else ""))
    tag = ("blocked" + ("/shave-major" if sm else "")) if blocked else "round-robin"
    print(f"  tiles {tiles}, {tag:22s} rounds {rounds}, rows {rows}: "
          + ("CLEAN" if bad_total == 0 else f"{bad_total} bad slots over {trials} trials"))
    return bad_total


def main(argv):
    rounds, rows = arg(argv, "--rounds", 2), arg(argv, "--rows", 3)
    trials = arg(argv, "--trials", 3)
    build = "--no-build" not in argv
    if "--sweep" in argv:
        t0 = time.time()
        srcs, cases = {}, []
        for tiles in (1, 2, 3):
            for blocked in (False, True):
                cases.append((tiles, blocked))
        for tiles, blocked in cases:   # one build for the whole sweep
            low = lowering.lower(make_dma_page_probe(KH=KH, VH=VH, DK=DK, ROWS=rows, rounds=rounds,
                                                     blocked=blocked, rw="--rw" in argv, tiles=tiles,
                                                     shaves=SHAVES))
            srcs[f"tl_dma_page_probe_t{tiles}_r{rounds}_rows{rows}" + ("_blocked" if blocked else "_rr")
                 + ("_rw" if "--rw" in argv else "")] = low.source
        runner.build_all(srcs)
        print(f"build {time.time() - t0:.0f} s")
        rc = 0
        for tiles, blocked in cases:
            rc |= one(argv, tiles, blocked, rounds, rows, trials, build=False)
        return 1 if rc else 0
    return 1 if one(argv, arg(argv, "--tiles", 3), "--blocked" in argv, rounds, rows, trials, build) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
