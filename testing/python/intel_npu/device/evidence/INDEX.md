# Device evidence index (Intel NPU backend)

One row per device session. Timing numbers come from UNPROFILED runs only; a
trace's own span is profiled and is never a timing claim. `events/SHAVE` is the
profiler event count per SHAVE: a count at the profiler's cap is SATURATED and
the trace inadmissible. Every row names the generated kernel (sha256 prefix of
the `.cpp`) and the `gc_dsl_barrier` commit it was built against.

| Date | Phase | Kernel (sha256) | Shape | Tiles x SHAVEs | Golden | Unprofiled median (host wall) | Trace | events/SHAVE | gc_dsl_barrier |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-06 | 1 | `tl_gemm_f16` (`814aa2e5d092`) | fp16 GEMM 256x256x512, BM=BN=64, whole K | 2 x 2 | PASS, max abs 0.0313, rel 3.24e-4 vs numpy fp32; 5 runs x 12 launches, all status OK | 0.219-0.225 ms (incl. submit; ~0.07 GFLOP, i.e. not a kernel-only number) | `tl_gemm_f16_256x256x512_trace.json` (profiled run: 0.236 ms host wall) | 8-24 (far below cap) | `daf6c05` |

Negative controls run in the same session: `--negative` (perturbed B) reported FAIL (rel 2.06e-1);
`--cmx-negative` (4 MiB A tile per SHAVE) reported `CMX_EXHAUSTED` through the per-SHAVE status
record and the runner exited non-zero.

Open item: the very first launch after the first device build reported SHAVE 2 status unwritten
(`0xFFFFFFFF`); every one of the 70+ launches since (rebuilt with the same kernel source) wrote all
four status records. Not reproduced; recorded here so it is not forgotten if it recurs.
