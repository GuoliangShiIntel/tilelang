"""Per-task profiling records of an intel_npu launch: a table, and a Trace Event file for ui.perfetto.dev.

This is VPUX's arrangement (`guides/how_to_use_profiling.md`): the kernel writes one fixed record per TASK --
every DMA, DPU workload, barrier and phase -- into a DDR area, and the host decodes it offline. Per-task is
what gives DMA, DPU and SHAVE their own REAL tracks with real start and end; a per-phase sum cannot, which is
why VPUX instruments dma / dpu / sw as three independent paths.

On NPU50xx every timestamp comes from the one fixed-frequency counter (19.2 MHz here), so unlike NPU37xx/40xx
no cross-engine clock calibration is applied -- see the guide's "Timestamp collection and synchronization".

Layout per worker: a 4-word header {n, dropped, magic 'TASK', 0} then `n` records of {t0, t1, kind, phase}.
`dropped` is reported on every path: a truncated capture has to be loud.
"""

import numpy as np

FRC_MHZ = 19.2
MAGIC = 0x4B534154
KINDS = {0: "dma_load", 1: "dma_store", 2: "ring", 3: "dpu", 4: "barrier", 5: "phase", 6: "ring_wait",
         7: "shave"}
HDR = 4          # header words
REC = 4          # words per record


def name4(packed: int) -> str:
    b = [(int(packed) >> (8 * i)) & 0xFF for i in range(4)]
    return "".join(chr(c) if 32 <= c < 127 else "" for c in b) or "-"


def load(path, workers: int = 6):
    """-> (list of per-worker record arrays [n, 4], list of dropped counts)."""
    raw = np.fromfile(path, dtype=np.uint32)
    per_worker = raw.size // workers
    out, dropped = [], []
    for w in range(workers):
        a = raw[w * per_worker:(w + 1) * per_worker]
        n, drop, magic = int(a[0]), int(a[1]), int(a[2])
        if magic != MAGIC:
            raise SystemExit(f"worker {w}: no TASK header (magic {magic:#x}); the kernel was built with "
                             f"TL_NPU_PHASE_PROF=0, or the area layout disagrees with npu_rt_device.hpp")
        out.append(a[HDR:HDR + n * REC].reshape(n, REC))
        dropped.append(drop)
    return out, dropped


def walk(recs):
    """-> list of (phase_name, [records]) in execution order.

    A phase record is written when the phase ENDS, so everything since the previous one belongs to it. That
    walk is what attributes a site (`gdeq`, `ggem`) to its phase: the record's own name field carries the
    SITE, not the phase, because the site is what needs its own track."""
    out, pending = [], []
    for r in recs:
        if int(r[2]) == 5:
            out.append((name4(int(r[3])), pending))
            pending = []
        else:
            pending.append(r)
    if pending:
        out.append(("-", pending))
    return out


def table(per, dropped) -> None:
    us = 1.0 / FRC_MHZ
    nw = len(per)
    if any(dropped):
        print(f"DROPPED {sum(dropped)} records ({max(dropped)} on the worst worker): the capture is TRUNCATED, "
              f"raise kTaskMax in npu_rt_device.hpp\n")
    span_of = {}          # phase -> summed span over workers
    kind_of = {}          # (phase, kind) -> [count, us]
    site_of = {}          # (phase, site) -> [count, us]
    for recs in per:
        for ph, items in walk(recs):
            # the phase record's own extent is the last one seen; recompute from the items' envelope instead
            if items:
                t0 = min(int(r[0]) for r in items)
                t1 = max(int(r[1]) for r in items)
                span_of[ph] = span_of.get(ph, 0.0) + ((t1 - t0) & 0xFFFFFFFF) * us
            for r in items:
                d = ((int(r[1]) - int(r[0])) & 0xFFFFFFFF) * us
                k = KINDS.get(int(r[2]), str(int(r[2])))
                a = kind_of.setdefault((ph, k), [0, 0.0])
                a[0] += 1
                a[1] += d
                if int(r[2]) == 7:
                    b = site_of.setdefault((ph, name4(int(r[3]))), [0, 0.0])
                    b[0] += 1
                    b[1] += d
    print(f"{nw} workers, {sum(len(r) for r in per)} records\n")
    cols = ("dma_load", "dma_store", "ring", "ring_wait", "dpu", "barrier", "shave")
    phases = sorted(span_of, key=lambda p: -span_of[p])
    tot = sum(span_of.values()) / nw
    print(f"{'phase':>7} {'span us':>9} {'share':>6}  " + "  ".join(f"{c:>12}" for c in cols))
    for p in phases:
        sp = span_of[p] / nw
        cells = [f"{kind_of.get((p, c), [0, 0.0])[1] / nw:>7.0f}/{kind_of.get((p, c), [0, 0.0])[0] // nw:<4}"
                 for c in cols]
        print(f"{p:>7} {sp:>9.0f} {100 * sp / tot if tot else 0:>5.1f}%  " + "  ".join(cells))
    print(f"{'(us per worker / task count)':>7}")
    if site_of:
        print(f"\nnamed SHAVE sites:\n{'phase':>7} {'site':>6} {'calls':>7} {'total us':>9} {'per call':>9}")
        for (p, s), (c, d) in sorted(site_of.items(), key=lambda kv: -kv[1][1]):
            print(f"{p:>7} {s:>6} {c // nw:>7} {d / nw:>9.0f} {d / max(1, c):>9.2f}")



def chrome_json(per, dropped, path) -> str:
    """Real tracks: one per kind per worker, every task at its own start and end."""
    import json

    t0 = min(int(r[:, 0].min()) for r in per if len(r))
    us = 1.0 / FRC_MHZ
    shaves = 2 if len(per) % 2 == 0 else 1
    ev = []
    for w, recs in enumerate(per):
        tile, shave = w // shaves, w % shaves
        pid = 100 + w
        ev.append({"ph": "M", "pid": pid, "tid": 0, "name": "process_name",
                   "args": {"name": f"Cluster {tile} SHAVE {shave}"}})
        for kind, label in KINDS.items():
            ev.append({"ph": "M", "pid": pid, "tid": kind + 1, "name": "thread_name", "args": {"name": label}})
        for phase, items in walk(recs):
            for r in items:
                a, b, kind, nm = int(r[0]), int(r[1]), int(r[2]), int(r[3])
                ev.append({"ph": "X", "pid": pid, "tid": kind + 1, "name": name4(nm),
                           "ts": ((a - t0) & 0xFFFFFFFF) * us, "dur": ((b - a) & 0xFFFFFFFF) * us,
                           "args": {"phase": phase}})
            if items:   # the phase itself, on its own track, spanning what it contained
                s = min(int(r[0]) for r in items)
                e = max(int(r[1]) for r in items)
                ev.append({"ph": "X", "pid": pid, "tid": 6, "name": phase,
                           "ts": ((s - t0) & 0xFFFFFFFF) * us, "dur": ((e - s) & 0xFFFFFFFF) * us})
    out = {"traceEvents": ev, "displayTimeUnit": "ns",
           "tl": {"source": "task records", "frc_mhz": FRC_MHZ, "workers": len(per),
                  "dropped": int(sum(dropped))}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    drop = f"; DROPPED {sum(dropped)}" if any(dropped) else ""
    return f"{len(ev)} events -> {path}{drop}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks_bin")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--json", default=None, help="write a Chrome trace for ui.perfetto.dev")
    a = ap.parse_args()
    per, dropped = load(a.tasks_bin, a.workers)
    table(per, dropped)
    if a.json:
        print(chrome_json(per, dropped, a.json))


if __name__ == "__main__":
    main()
