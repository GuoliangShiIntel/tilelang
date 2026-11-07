"""DMA rate probe (probes/dma_rate_probe.cpp): N KiB DDR -> CMX in six configurations, from tile 0
SHAVE 0, timed with NCE_TIME. Prints GB/s per configuration. The job size separates the per-job
fixed latency from the streaming bandwidth (the prior art's 28.7 / 51.6 GB/s used 512 KiB jobs).

Usage: python testing/python/intel_npu/device/run_dma_rate_probe.py [--no-build] [--kib 128]
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
LABELS = ["unicast 1-D, 1 port, 1 issuer", "unicast 1-D, 6 issuers each its OWN region, port = shave & 1 (x6 bytes)",
          "multicast, every SHAVE of every tile issues 1/6 as 2 halves (6 issuers x 2 ports)",
          "multicast, SHAVE 0 of every tile issues 1/3 as 2 halves (3 issuers x 2 ports)",
          "multicast 1-D two halves ports 0+1, 1 issuer (the ring fill today)", "unicast 1-D two halves ports 0+1, 1 issuer"]
MULT = [1, 6, 1, 1, 1, 1]   # bytes moved per rep relative to one job


def main(argv):
    kib = int(argv[argv.index("--kib") + 1]) if "--kib" in argv else 128
    nbytes = kib * 1024
    kernel = f"{KERNEL}_{kib}k"
    source = (HERE / "probes" / "dma_rate_probe.cpp").read_text(encoding="utf-8").replace(
        "#define TL_PROBE_KIB 128", f"#define TL_PROBE_KIB {kib}")
    if "--no-build" in argv:
        _, sha = runner.write_kernel(source, kernel)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=kernel, source_sha256=sha)
    else:
        db = runner.build(source, kernel)
    x = np.random.default_rng(0).standard_normal((TILES * SHAVES * nbytes // 2,)).astype(np.float16)   # six regions
    res = runner.run(db, [("in", x)], TILES, SHAVES, HERE / "run_dma_rate", repeat=5, warmup=2, timeout_s=120)
    runner.check_status(res.status)
    ticks = res.status[0, 10:16].astype(np.int64)
    print(f"kernel sha256 {db.source_sha256[:12]}; {REPS} x {nbytes} B per configuration, tile 0 SHAVE 0, NCE_TIME 19.2 MHz")
    for label, t, m in zip(LABELS, ticks, MULT):
        us = t / 19.2
        print(f"  {label:82s}: {us / REPS:7.2f} us per {kib} KiB, {m * REPS * nbytes / (us * 1e-6) / 1e9:6.2f} GB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
