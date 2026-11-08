"""Attribution table of a profiled intel_npu trace (the first step of the intel-npu-perf-loop skill).

Segments one cluster's DPU track at idle gaps (start-to-start, since a dpu event's `dur` is when a SHAVE observed
completion, not when the hardware finished: E7-28) and prints, per segment: span, DPU workloads and busy, every
SHAVE's dma_wait / dpu_wait / sync_local, the SHAVE compute left over (span - waits: the time the SHAVE was neither
waiting nor synchronising), and the DMA jobs the DMA tracks ran inside the window. The gaps between segments are
SHAVE-only sections (no DPU work) and are listed too. Byte counts are not in the trace (`dst_width_bytes` is 0), so
bandwidth comes from the kernel's own notes, not from here. The DMA tracks hold the toolkit's synchronous copies
(`dma_load` / `dma_store`) only: ring fills (`DmaDrv::start`) are not DMA-track events, so a streamed section shows
`dma jobs 0` while its SHAVEs' `dma_wait` is the fills' cost.

    python perf_attr.py trace.json [--cluster "Cluster 0"] [--gap 25] [--vs other.json]

A PROFILED span is a split, never a timing claim. Capture with NPU_PROFILE_FRC_MHZ=19.2.
"""

import argparse
import collections
import json
from pathlib import Path


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    ev = d["traceEvents"] if isinstance(d, dict) and "traceEvents" in d else d
    names = {}
    for e in ev:
        if e.get("ph") == "M":
            names[(e.get("pid"), e.get("tid"), e.get("name"))] = e.get("args", {}).get("name")
    tracks = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X":
            pn = names.get((e["pid"], 0, "process_name")) or e["pid"]
            tn = names.get((e["pid"], e["tid"], "thread_name")) or e["tid"]
            tracks[(str(pn), str(tn))].append(e)
    for es in tracks.values():
        es.sort(key=lambda e: e["ts"])
    t0 = min(e["ts"] for es in tracks.values() for e in es)
    return tracks, t0


def segments(tracks, t0, cluster, gap):
    dpu = tracks.get((cluster, "DPU"))
    if not dpu:
        raise SystemExit(f"no DPU track for {cluster}; tracks: {sorted(tracks)}")
    segs, cur = [], [dpu[0]]
    for e in dpu[1:]:
        if e["ts"] - cur[-1]["ts"] > gap:
            segs.append(cur)
            cur = []
        cur.append(e)
    segs.append(cur)
    out = []
    for g in segs:
        s = g[0]["ts"] - t0
        e_ = g[-1]["ts"] + g[-1].get("dur", 0) - t0
        # busy: a workload's dur clipped at the next start (observed completion lags the hardware)
        busy = sum(min(x.get("dur", 0), (g[k + 1]["ts"] - x["ts"]) if k + 1 < len(g) else x.get("dur", 0)) for k, x in enumerate(g))
        out.append(dict(start=s, end=e_, n=len(g), busy=busy))
    return out


def window_stats(tracks, t0, cluster, a, b):
    waits = {}
    regions = collections.Counter()
    for (pn, tn), es in tracks.items():
        if pn != cluster or not tn.startswith("SHAVE"):
            continue
        w = collections.Counter()
        for x in es:
            xs = x["ts"] - t0
            if a <= xs <= b:
                if x["name"] in ("dma_wait", "dpu_wait", "sync_local"):
                    w[x["name"]] += x.get("dur", 0)
                else:   # a named region (tl_npu_perf_begin / end): measured SHAVE compute, by label
                    regions[(tn, str(x.get("args", {}).get("name", x["name"])))] += x.get("dur", 0)
        waits[tn] = dict(dma=w["dma_wait"], dpu=w["dpu_wait"], sync=w["sync_local"], compute=max(0.0, (b - a) - sum(w.values())))
    jobs, jdur = 0, 0.0
    for (pn, tn), es in tracks.items():
        if not pn.startswith("DMA"):
            continue
        for x in es:
            xs = x["ts"] - t0
            if a <= xs <= b:
                jobs += 1
                jdur += x.get("dur", 0)
    return waits, jobs, jdur, regions


def table(path, cluster, gap):
    tracks, t0 = load(path)
    segs = segments(tracks, t0, cluster, gap)
    rows = []
    prev_end = None
    for i, sg in enumerate(segs):
        if prev_end is not None and sg["start"] - prev_end > gap:
            waits, jobs, jdur, regions = window_stats(tracks, t0, cluster, prev_end, sg["start"])
            rows.append(dict(kind="gap", start=prev_end, end=sg["start"], n=0, busy=0.0, waits=waits, jobs=jobs, jdur=jdur, regions=regions))
        waits, jobs, jdur, regions = window_stats(tracks, t0, cluster, sg["start"], sg["end"])
        rows.append(dict(kind=f"seg{i}", **sg, waits=waits, jobs=jobs, jdur=jdur, regions=regions))
        prev_end = sg["end"]
    return rows, t0


def fmt(rows, title):
    print(f"== {title}")
    print(f"{'section':8} {'start':>7} {'end':>7} {'span':>6} {'dpu':>4} {'busy%':>5} {'dma jobs':>8} {'jobs us':>7}  SHAVE: dma_wait / dpu_wait / sync / compute (us)")
    for r in rows:
        span = r["end"] - r["start"]
        busy = f"{100 * r['busy'] / span:4.0f}%" if r["n"] else "   -"
        sh = "; ".join(f"{k[-1]}: {v['dma']:.0f}/{v['dpu']:.0f}/{v['sync']:.0f}/{v['compute']:.0f}" for k, v in sorted(r["waits"].items()))
        print(f"{r['kind']:8} {r['start']:7.0f} {r['end']:7.0f} {span:6.0f} {r['n']:4d} {busy:>5} {r['jobs']:8d} {r['jdur']:7.0f}  {sh}")
        if r["regions"]:
            print("         regions: " + "; ".join(f"{tn[-1]}:{name} {d:.0f}" for (tn, name), d in sorted(r["regions"].items())))
    total = rows[-1]["end"] - rows[0]["start"] if rows else 0
    print(f"total {total:.0f} us over {len([r for r in rows if r['kind'] != 'gap'])} DPU segments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--vs", default=None, help="a second trace printed after the first (same cluster and gap)")
    ap.add_argument("--cluster", default="Cluster 0")
    ap.add_argument("--gap", type=float, default=25.0, help="DPU start-to-start idle gap (us) that ends a segment")
    a = ap.parse_args()
    rows, _ = table(a.trace, a.cluster, a.gap)
    fmt(rows, f"{Path(a.trace).name} ({a.cluster}, gap {a.gap:g} us)")
    if a.vs:
        rows2, _ = table(a.vs, a.cluster, a.gap)
        fmt(rows2, f"{Path(a.vs).name} ({a.cluster}, gap {a.gap:g} us)")


if __name__ == "__main__":
    main()
