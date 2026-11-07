"""DMA rate probe (probes/dma_rate_probe.cpp): 128 KiB DDR -> CMX in six configurations, from tile 0
SHAVE 0, timed with NCE_TIME. Prints GB/s per configuration.

Usage: python testing/python/intel_npu/device/run_dma_rate_probe.py [--no-build]
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401
import runner  # noqa: E402

KERNEL = "tl_dma_rate_probe"
TILES, SHAVES = 3, 2
BYTES, REPS = 128 * 1024, 16
LABELS = ["unicast 1-D, 1 port, 1 issuer", "multicast 1-D, 1 port, 1 issuer",
          "multicast, every SHAVE of every tile issues 1/6 as 2 halves (6 issuers x 2 ports)",
          "multicast, SHAVE 0 of every tile issues 1/3 as 2 halves (3 issuers x 2 ports)",
          "multicast 1-D two halves ports 0+1, 1 issuer (the ring fill today)", "unicast 1-D two halves ports 0+1, 1 issuer"]


def main(argv):
    source = (HERE / "probes" / "dma_rate_probe.cpp").read_text(encoding="utf-8")
    if "--no-build" in argv:
        _, sha = runner.write_kernel(source, KERNEL)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=KERNEL, source_sha256=sha)
    else:
        db = runner.build(source, KERNEL)
    x = np.random.default_rng(0).standard_normal((BYTES // 2,)).astype(np.float16)
    res = runner.run(db, [("in", x)], TILES, SHAVES, HERE / "run_dma_rate", repeat=5, warmup=2, timeout_s=120)
    runner.check_status(res.status)
    ticks = res.status[0, 10:16].astype(np.int64)
    print(f"kernel sha256 {db.source_sha256[:12]}; {REPS} x {BYTES} B per configuration, tile 0 SHAVE 0, NCE_TIME 19.2 MHz")
    for label, t in zip(LABELS, ticks):
        us = t / 19.2
        print(f"  {label:82s}: {us / REPS:7.2f} us per 128 KiB, {REPS * BYTES / (us * 1e-6) / 1e9:6.2f} GB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
