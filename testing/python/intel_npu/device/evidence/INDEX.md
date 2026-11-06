# Device evidence index (Intel NPU backend)

One row per device session. Timing numbers come from UNPROFILED runs only; a
trace's own span is profiled and is never a timing claim. `events/SHAVE` is the
profiler event count per SHAVE: a count at the profiler's cap is SATURATED and
the trace inadmissible. Every row names the generated kernel (sha256 prefix of
the `.cpp`) and the `gc_dsl_barrier` commit it was built against.

| Date | Phase | Kernel (sha256) | Shape | Tiles x SHAVEs | Golden | Unprofiled median (host wall) | Trace | events/SHAVE | gc_dsl_barrier |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-06 | 1 | `tl_gemm_f16` (`814aa2e5d092`) | fp16 GEMM 256x256x512, BM=BN=64, whole K | 2 x 2 | PASS, max abs 0.0313, rel 3.24e-4 vs numpy fp32; 5 runs x 12 launches, all status OK | 0.219-0.225 ms (incl. submit; ~0.07 GFLOP, i.e. not a kernel-only number) | `tl_gemm_f16_256x256x512_trace.json` (profiled run: 0.236 ms host wall) | 8-24 (far below cap) | `daf6c05` |
| 2026-11-06 | 2 | `tl_gemm_two_phase` (`2e0bd7becea4`), runtime with memoryFence | split-K partials (2 producer SHAVEs) -> `T.sync_grid()` -> combine on all 4 SHAVEs, 256x256x512, one launch | 2 x 2 | PASS rel 5.67e-4; after the fence fix 15 process runs x 4 launches + Phase 1 GEMM 10 x 12 launches: 0 failures | 0.45 ms (incl. submit; scalar fp16 combine loop on the SHAVE, not a kernel-only number) | `tl_gemm_two_phase_256x256x512_trace.json` (profiled: 0.565 ms host wall) | 12-120 (below cap; `dma`/`dpu` + waits) | `daf6c05` |
| 2026-11-06 | 2b | `tl_gemm_two_phase_auto` (source byte-identical to `tl_gemm_two_phase`, barrier derived by `DeriveIntelNpuGlobalSync`) | as above | 2 x 2 | PASS rel 5.67e-4; `.text` 20096 B (7.7% of 262272) | 0.59 ms (incl. submit) | - | - | `daf6c05` |

Negative controls, Phase 2: `run_two_phase_device.py --nosync` (same kernel without the barrier and
with `tl.intel_npu.derive_global_sync=False`, since Phase 2b the derivation would otherwise insert it):
5/5 launches FAIL (rel 0.89-1.00), so the barrier is load-bearing on hardware. The kernel is shaped
so phase A runs on half the workers and phase B on all of them; the first shape (balanced phase A,
rotated combine) passed 28/28 launches without the barrier because the SHAVEs ran in lockstep, and
was replaced for that reason.

Negative controls run in the same session: `--negative` (perturbed B) reported FAIL (rel 2.06e-1);
`--cmx-negative` (4 MiB A tile per SHAVE) reported `CMX_EXHAUSTED` through the per-SHAVE status
record and the runner exited non-zero.

**Resolved (Phase 2):** the "unwritten status record" first seen after the Phase 1 build recurred
in about 1 of 25 launches (SHAVE varied; the SHAVE's outputs were all correct, so it had run and
only its status words were missing; the words never arrived within 50 ms). Root cause: a SHAVE's
direct DDR store (`DdrAccessor`) sits in its data L1 and is lost when the SHAVE halts unless
`npu::sync::memoryFence()` (DL1 flush) follows it; the prior-art runtime did exactly that after its
status write. `npu_rt_device.hpp` now fences after the status record and before every barrier.
After the fix: 180 launches, 0 missing records; before it, on the same kernels: 4 of ~100.
