"""Hot-loop statistics of a moviCompile `.s` file (the intel-npu-perf-loop skill's step 1.3).

For every basic block that ends in a backward branch (a loop body) and holds vector math, prints the instruction
line count, NOPs (and the cycles they stand for), vector loads / stores, VAU ops and splats, and the label, so the
static cycle count of a hot loop is read instead of guessed. Produce the `.s` with the device include paths:

    moviCompile -mcpu=5000xx -O3 -DNPU_GENERIC_COMPUTE -I<GENERIC_COMPUTE_ROOT>/src -I<GENERIC_COMPUTE_ROOT>/src/shave/toolkit
                -I<GENERIC_COMPUTE_ROOT> -I<tilelang>/src -S kernel.cpp -o kernel.s
    python asm_loops.py kernel.s [--min-vau 4]

Reading the numbers: one line is at most one VLIW bundle (`||` continues a bundle), so `lines - continuation lines`
approximates issue cycles and `NOP n` adds n stall cycles; a loop whose body is mostly NOPs after loads carries too
little independent work per iteration (moviCompile does not software-pipeline these loops).
"""

import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asm")
    ap.add_argument("--min-vau", type=int, default=4, help="only loops with at least this many VAU ops")
    a = ap.parse_args()
    lines = open(a.asm, encoding="utf-8", errors="replace").read().splitlines()
    labels = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^(\.?L\w+):", ln)
        if m:
            labels[m.group(1)] = i
    # a loop = a backward branch to a label: body = [label, branch line]
    loops = []
    for i, ln in enumerate(lines):
        m = re.search(r"BRU\.BRA\s+(\.?L\w+)", ln)
        if m and m.group(1) in labels and labels[m.group(1)] < i:
            loops.append((labels[m.group(1)], i))
    print(f"{'label':12} {'lines':>5} {'bundles':>7} {'NOPcyc':>6} {'ld.v':>4} {'st.v':>4} {'VAU':>4} {'splat':>5} {'~cycles':>7}")
    for s, e in loops:
        body = lines[s:e + 1]
        bundles = sum(1 for b in body if b.strip() and not b.strip().startswith("||") and not b.strip().endswith(":"))
        nopcyc = 0
        for b in body:
            m = re.match(r"^\s+NOP\s*(\d*)", b)
            if m:
                nopcyc += int(m.group(1) or 1)
        ldv = sum(1 for b in body if re.search(r"LSU[01]\.LD[OX]?\.(256|512)", b))
        stv = sum(1 for b in body if re.search(r"LSU[01]\.ST[O]?\.(256|512)", b))
        vau = sum(1 for b in body if re.search(r"VAU\.", b))
        splat = sum(1 for b in body if "CP.512.32.R" in b)
        if vau < a.min_vau:
            continue
        print(f"{lines[s].strip()[:-1]:12} {len(body):5d} {bundles:7d} {nopcyc:6d} {ldv:4d} {stv:4d} {vau:4d} {splat:5d} {bundles + nopcyc:7d}")


if __name__ == "__main__":
    main()
