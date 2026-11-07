"""Render and check a runtime profiling trace of the Intel NPU backend.

    python trace_timeline.py <trace.json> [--window-us W] [--cols N]     one steady-state window, ASCII
    python trace_timeline.py <trace.json> --check [--chunks C] [--tiles T] [--min-busy P]

The trace is the generic_compute runtime's JSON (`saveProfilingJson`, captured with
`NPU_PROFILE_FRC_MHZ=19.2`): pid 0 = the DMA engines (one row per tile), pid t+1 = cluster t with
rows DPU (hardware HWP: `idu_workload_duration` ticks), SHAVE 0 and SHAVE 1 (what each SHAVE
submitted and waited for). `dma` and `dpu` event ends on the SHAVE rows are when the SHAVE
*observed* completion, not when the hardware finished; the DPU row's IDU ticks are the hardware's.

`--check` is the flow verdict (plan Phase 3 acceptance; README section 6a): exit 0 when every
cluster ran the same number of DPU workloads (= --chunks when given), the two SHAVEs of every
cluster submitted them alternately (slot = SHAVE, chunk order), tile 0's DMA row carries the
fills (at least two per chunk: the two halves) and the others do not, no SHAVE's event count is
at the profiler's cap, and the steady-state DPU busy of every cluster is at least --min-busy.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

PERF_CLK_MHZ = 19.2
PROFILER_CAP = 1984


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    events = [e for e in d["traceEvents"] if e.get("ph") == "X"]
    names = {}
    for e in d["traceEvents"]:
        if e.get("ph") == "M" and e["name"] == "thread_name":
            names[(e["pid"], e["tid"])] = e["args"]["name"]
        if e.get("ph") == "M" and e["name"] == "process_name":
            names[(e["pid"], 0)] = e["args"]["name"]
    return events, names


def rows(events):
    by = collections.defaultdict(list)
    for e in events:
        by[(e["pid"], e["tid"])].append(e)
    for v in by.values():
        v.sort(key=lambda e: e["ts"])
    return by


def clusters(names):
    return sorted({pid for (pid, tid), n in names.items() if tid == 0 and n.startswith("Cluster")})


def dpu_stats(by, pid):
    """(count, span_us, idu_busy_us, mean gap between consecutive hardware workloads) of a cluster's DPU row."""
    ev = by.get((pid, 1), [])
    if len(ev) < 2:
        return len(ev), 0.0, 0.0, 0.0
    idu = sum(e["args"].get("idu_workload_duration", 0) for e in ev) / PERF_CLK_MHZ
    span = ev[-1]["ts"] + ev[-1]["dur"] - ev[0]["ts"]
    gaps = [ev[i + 1]["ts"] - (ev[i]["ts"] + ev[i]["dur"]) for i in range(len(ev) - 1)]
    return len(ev), span, idu, sum(gaps) / len(gaps)


def steady_window(by, pid, frac=0.25):
    ev = by.get((pid, 1), [])
    if not ev:
        return 0.0, 0.0
    t0, t1 = ev[0]["ts"], ev[-1]["ts"] + ev[-1]["dur"]
    mid = (t0 + t1) / 2
    half = (t1 - t0) * frac / 2
    return mid - half, mid + half


def render(events, names, window_us=None, cols=120):
    by = rows(events)
    out = []
    for pid in clusters(names):
        w0, w1 = steady_window(by, pid)
        if window_us:
            mid = (w0 + w1) / 2
            w0, w1 = mid - window_us / 2, mid + window_us / 2
        if w1 <= w0:
            continue
        scale = cols / (w1 - w0)
        out.append(f"{names[(pid, 0)]}: window {w0:.0f}..{w1:.0f} us, {(w1 - w0) / cols:.2f} us per column")
        lanes = [(pid, 1, "DPU   "), (pid, 2, "SHAVE0"), (pid, 3, "SHAVE1"), (0, pid, f"DMA{pid - 1}  ")]
        for (p, t, label) in lanes:
            line = [" "] * cols
            for e in by.get((p, t), []):
                a, b = e["ts"], e["ts"] + e["dur"]
                if b < w0 or a > w1:
                    continue
                c0 = max(0, int((a - w0) * scale))
                c1 = min(cols - 1, int((b - w0) * scale))
                ch = {"dpu": "#", "dpu_wait": "w", "dma": ">", "dma_wait": "r", "sync_local": "|"}.get(e["name"], "?")
                for c in range(c0, c1 + 1):
                    line[c] = ch if line[c] == " " or ch == "#" else line[c]
            out.append(f"  {label} {''.join(line)}")
        n, span, idu, gap = dpu_stats(by, pid)
        out.append(f"  DPU: {n} workloads, IDU busy {idu:.0f} us of {span:.0f} us span = {100 * idu / span if span else 0:.0f}%, "
                   f"{span / n if n else 0:.2f} us per workload, mean gap {gap:.2f} us")
    out.append("legend: # dpu (hardware on the DPU row; submitted..observed on SHAVE rows), w dpu_wait, > dma, r dma_wait, | sync_local")
    return "\n".join(out)


def check(events, names, chunks=None, tiles=None, min_busy=0.0):
    by = rows(events)
    cl = clusters(names)
    problems = []
    if tiles is not None and len(cl) != tiles:
        problems.append(f"{len(cl)} clusters in the trace, expected {tiles}")
    counts = [len(by.get((pid, 1), [])) for pid in cl]
    if len(set(counts)) != 1:
        problems.append(f"DPU workloads per cluster differ: {counts} (every tile computes every chunk)")
    if chunks is not None and counts and counts[0] != chunks:
        problems.append(f"{counts[0]} DPU workloads per cluster, expected {chunks} chunks")
    for pid in cl:
        subs = sorted([(e["ts"], e["tid"]) for t in (2, 3) for e in by.get((pid, t), []) if e["name"] == "dpu"])
        order = [t for _, t in subs]
        if any(order[i] == order[i + 1] for i in range(len(order) - 1)):
            problems.append(f"{names[(pid, 0)]}: DPU submissions do not alternate between the SHAVEs: {order[:12]}...")
        for t in (2, 3):
            n = len(by.get((pid, t), []))
            if n >= PROFILER_CAP:
                problems.append(f"{names[(pid, 0)]} {names[(pid, t)]}: {n} events = profiler cap, trace SATURATED")
        n, span, idu, _ = dpu_stats(by, pid)
        busy = 100 * idu / span if span else 0
        if busy < min_busy:
            problems.append(f"{names[(pid, 0)]}: DPU busy {busy:.0f}% < {min_busy}%")
    dma_counts = {names.get((0, t), f"DMA{t}"): len(by.get((0, t), [])) for t in range(1, len(cl) + 1)}
    if counts:
        n0 = dma_counts.get("DMA 0", 0)
        others = [v for k, v in dma_counts.items() if k != "DMA 0"]
        # tile 0: 2 halves per chunk + its stores (+ resident fills); other tiles: stores (+ resident fills) only
        if n0 < 3 * counts[0]:
            problems.append(f"DMA 0 has {n0} events, fewer than 2 fills + 1 store per chunk x {counts[0]}: the fills are not issued by tile 0")
        if any(v >= n0 for v in others):
            problems.append(f"another tile's DMA row has as many events as tile 0's ({dma_counts}): fills are not multicast from one issuer")
    return problems


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--chunks", type=int)
    ap.add_argument("--tiles", type=int)
    ap.add_argument("--min-busy", type=float, default=0.0)
    ap.add_argument("--window-us", type=float)
    ap.add_argument("--cols", type=int, default=120)
    a = ap.parse_args(argv)
    events, names = load(a.trace)
    per_shave = {f"{names[(p, 0)]}/{names[(p, t)]}": len(v) for (p, t), v in rows(events).items() if (p, t) in names and p != 0}
    print(f"events per row: {per_shave} (profiler cap {PROFILER_CAP})")
    print(render(events, names, a.window_us, a.cols))
    if a.check:
        problems = check(events, names, a.chunks, a.tiles, a.min_busy)
        for p in problems:
            print("FLOW CHECK FAIL:", p)
        print("FLOW CHECK:", "PASS (the flow ran)" if not problems else f"FAIL ({len(problems)})")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
