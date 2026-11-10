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

ADMISSIBILITY. A SHAVE records at most 1984 events and `Profiler::mark` drops the rest with a bare `return`:
no flag, no overflow bit. A truncated trace is therefore well-formed JSON that simply stops early, and its
attribution table is a table of the kernel's first few milliseconds wearing the whole kernel's name. `Session.
profile` stamps every trace it writes with the NCE_TIME duration of the launches it covers (`tl.kernel_ms`);
this tool refuses to print a table when the span falls short of it. `--allow-truncated` prints anyway and
labels every line, for when the first N ms really is the question.
"""

import argparse
import collections
import json
from pathlib import Path


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    ev = d["traceEvents"] if isinstance(d, dict) and "traceEvents" in d else d
    meta = d.get("tl", {}) if isinstance(d, dict) else {}
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
    return tracks, t0, meta


COVERAGE_MIN = 0.90      # below this the table describes a prefix of the kernel, not the kernel


def coverage(tracks, t0, meta):
    """(span_us, kernel_us or None, fraction or None) -- what the trace covers of the kernel it came from."""
    span = max(e["ts"] + e.get("dur", 0) for es in tracks.values() for e in es) - t0
    kms = meta.get("kernel_ms")
    if not kms:
        return span, None, None
    kus = float(kms) * 1000.0
    return span, kus, (span / kus if kus > 0 else None)


def check_admissible(path, tracks, t0, meta, allow_truncated=False):
    """Refuse a trace that covers only a prefix of its kernel. Returns the coverage line to print."""
    span, kus, frac = coverage(tracks, t0, meta)
    per = collections.Counter()
    for (pn, tn), es in tracks.items():
        per[pn] += len(es)
    if kus is None:
        return (f"coverage: span {span:.0f} us; kernel time UNKNOWN -- this trace carries no `tl.kernel_ms`, so "
                f"whether it covers its kernel CANNOT BE CHECKED. Recapture through Session.profile.")
    line = (f"coverage: span {span:.0f} us of a {kus:.0f} us kernel = {100 * frac:.1f}% "
            f"({meta.get('launches', '?')} launches of '{meta.get('alias', '?')}', "
            f"{sum(per.values())} events)")
    if frac >= COVERAGE_MIN:
        return line
    msg = (f"{line}\n"
           f"TRUNCATED: a SHAVE records at most 1984 events and silently drops the rest "
           f"(kernel_wrapper.cpp Profiler::mark), so this trace stops after the kernel's first {span:.0f} us and "
           f"an attribution table from it would describe {100 * frac:.1f}% of the work while naming all of it.\n"
           f"For whole-kernel attribution use the per-task records instead (`--phases`, `task_report.py`), "
           f"which cost O(tasks) in DDR rather than 1984 events in CMX.\n"
           f"--allow-truncated prints the table anyway, labelled.")
    if not allow_truncated:
        raise SystemExit(msg)
    return msg


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


WAIT_NAMES = ("dma_wait", "dpu_wait", "sync_local", "sync_global", "dma", "dpu")
def phase_rows(tracks, t0):
    """Per named region: wall time per worker and share of the capture. Used when a trace has no DPU track to
    segment on."""
    shaves = [(k, es) for k, es in tracks.items() if k[1].startswith("SHAVE")]
    span = max(e["ts"] + e.get("dur", 0) for es in tracks.values() for e in es) - t0
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    cnt = collections.Counter()
    for k, es in shaves:
        for e in es:
            if e["name"] in WAIT_NAMES:
                continue
            per[e["name"]][k] += e.get("dur", 0)
            cnt[e["name"]] += 1
    rows = []
    for name, byw in per.items():
        vals = [byw.get(k, 0.0) for k, _ in shaves]
        rows.append(dict(name=name, mean=sum(vals) / len(vals), lo=min(vals), hi=max(vals),
                         n=cnt[name] // max(1, len(shaves))))
    rows.sort(key=lambda r: -r["mean"])
    return rows, span


def fmt_phases(rows, span, title, cov):
    print(f"== {title}")
    print(cov)
    print(f"{'phase':>6} {'n':>4} {'mean us':>10} {'share':>7} {'min us':>10} {'max us':>10} {'imbal':>6}")
    tot = 0.0
    for r in rows:
        tot += r["mean"]
        imb = (r["hi"] / r["lo"]) if r["lo"] > 0 else float("inf")
        print(f"{r['name']:>6} {r['n']:>4} {r['mean']:>10.0f} {100 * r['mean'] / span:>6.1f}% "
              f"{r['lo']:>10.0f} {r['hi']:>10.0f} {imb:>6.2f}")
    print(f"{'sum':>6} {'':>4} {tot:>10.0f} {100 * tot / span:>6.1f}%   of a {span:.0f} us capture; "
          f"the remainder is grid barriers and unbracketed code")


def table(path, cluster, gap, allow_truncated=False):
    tracks, t0, meta = load(path)
    cov = check_admissible(path, tracks, t0, meta, allow_truncated)
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
    return rows, t0, cov


def fmt(rows, title, cov=None):
    print(f"== {title}")
    if cov:
        print(cov)
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
    ap.add_argument("--allow-truncated", action="store_true",
                    help="print the table even when the trace covers only a prefix of its kernel")
    ap.add_argument("--phases", action="store_true",
                    help="per-region phase table, for a trace with no DPU track")
    a = ap.parse_args()

    def one(path):
        tracks, t0, meta = load(path)
        cov = check_admissible(path, tracks, t0, meta, a.allow_truncated)
        title = f"{Path(path).name} ({a.cluster}, gap {a.gap:g} us)"
        if a.phases or (a.cluster, "DPU") not in tracks:
            rows, span = phase_rows(tracks, t0)
            fmt_phases(rows, span, f"{Path(path).name} (phases, all workers)", cov)
        else:
            rows, _, _ = table(path, a.cluster, a.gap, a.allow_truncated)
            fmt(rows, title, cov)

    one(a.trace)
    if a.vs:
        one(a.vs)


if __name__ == "__main__":
    main()
