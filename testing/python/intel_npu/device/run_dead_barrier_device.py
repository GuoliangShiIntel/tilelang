"""Negative control: sync_grid on dead barrier ids must REPORT, not hang.

Builds probes/dead_barrier_sync.cpp (sync_grid armed on ids 0..2, which never go live on this
driver) and expects every SHAVE's status to be BARRIER_NOT_LIVE with detail 0, returned well
within the runner timeout. Exit 0 = the bounded wait behaved.

Usage: python run_dead_barrier_device.py [--no-build]
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))

import runner  # noqa: E402
import tilelang  # noqa: E402,F401

KERNEL = "tl_dead_barrier_sync"
TILES, SHAVES = 2, 2


def main(argv):
    source = (HERE / "probes" / "dead_barrier_sync.cpp").read_text(encoding="utf-8")
    if "--no-build" in argv:
        _, sha = runner.write_kernel(source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        db = runner.build(source, KERNEL)
    t0 = time.time()
    try:
        runner.run(db, [], TILES, SHAVES, HERE / "run_dead_barrier", repeat=1, warmup=0, timeout_s=60)
    except RuntimeError as e:
        msg = str(e)
        status = np.fromfile(HERE / "run_dead_barrier" / "status.bin", dtype=np.uint32).reshape(-1, 8)[: TILES * SHAVES, :2]
        ok = all(int(c) == 5 and int(d) == 0 for c, d in status)
        print(f"returned in {time.time() - t0:.1f} s; status (code, detail) per SHAVE: {status.tolist()}")
        print("dead-barrier control: " + ("behaved (BARRIER_NOT_LIVE id 0 reported, no hang)" if ok else f"BROKEN: {msg[-300:]}"))
        return 0 if ok else 1
    print("dead-barrier control: BROKEN: the kernel reported OK on a dead barrier")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
