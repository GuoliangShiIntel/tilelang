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
| 2026-11-06 | 3a | `tl_sync_chain_hw` (`7c12cc0714d7`) / `tl_sync_chain_sw` (`bb872fdbb4b4`) | 200 grid barriers per launch with a DDR token chain | 3 x 2 | both correct (token = 200 on every worker); `--nosync` FAILs | **hw 0.469 ms vs sw 1.604 ms** (3 interleaved rounds x 5 launches; -5.7 us per barrier) | - | - | `daf6c05` |
| 2026-11-06 | 3a | `tl_barrier_probe` (`7f2f41dfcb5a`) | one hardware barrier id per process: arm, produce, wait, consume, bounded spins | 3 x 2 and 2 x 2 | 3-tile job: 0-3, 5, 6 config-not-live; 4, 7-47 OK. 2-tile job: 28-31 OK, 32-33 **hang** (outside the job's 16 x 2 ids) | n/a | `barrier_probe.json`, `barrier_probe_2tiles.json` | - | `daf6c05` |
| 2026-11-06 | 3a | `tl_dead_barrier_sync` (hand-written) | sync_grid armed on dead ids 0-2 | 2 x 2 | negative control: every SHAVE reports `BARRIER_NOT_LIVE id 0` in 0.4 s, no hang | n/a | - | - | `daf6c05` |
| 2026-11-06 | 3a | `tl_gemm_two_phase` (`c954a8d995f2`, hw sync ids 29-31) | as Phase 2 | 2 x 2 | PASS rel 5.67e-4 | 0.454 ms | - | - | `daf6c05` |
| 2026-11-06 | 3b-0 | `tl_dpucal_<shape>_x<X>_m<mpe+1>_r<R>_k` (80 kernels; e.g. `..._64x64x512_x8_m2_r401_k` `8d7776157209`) | VPUNN calibration: DPU repeat probe, 4 tile shapes x 10 (spatial, MPE) configs | 1 x 2 | all 80 golden PASS | DPU IDU time per workload from the HWP counters (19.2 MHz ticks): default 1.564 us vs best 1.001 us at 64x64x512; VPUNN pick within 0.0-2.8% of best on every shape; IDU / VPUNN@1950 median 1.58 | `vpunn_calibration.json` (not a timeline trace) | - | `daf6c05` |
| 2026-11-06 | 3b | `tl_gemm_f16` (`0039c507d6e3`, workloads planned by VPUNN: CUBOID_8x16 x=16) and `tl_gemm_two_phase` | Phase 1 GEMM and Phase 2 two-phase with `PlanIntelNpuWorkloads` on by default | 2 x 2 | PASS rel 3.24e-4; PASS rel 5.67e-4 | 0.225 ms / 0.452 ms (incl. submit; launch-dominated at this shape, the DPU gain is in the 3b-0 row) | - | - | `daf6c05` |
| 2026-11-06 | 3b-A | `tl_ring_flow_hand` (hand-written, `probes/ring_flow_hand.cpp`; `7c48df9ee845` at 2048x384x1024 BM64/BN128, `5f29ca6fa8ca` at 2048x768x2048 BM32/BN256) | multi-tile ping-pong ring: W split + resident, X multicast through 2 slots, READY/FREE hw barriers ids 8-15, sync 45-47 | 3 x 2 | PASS rel 3.8e-4 / 2.86e-4; perturbed W FAILs | 0.944 ms incl. ~0.2 ms submit at 2048x768x2048 (6.44 GFLOP; DPU IDU 696 us per tile, 64 workloads); TOPS: not measured kernel-only | `tl_ring_flow_hand_2048x384x1024_trace.json` (first shape; profiled run PASS after the liveness fix) | 33-99 (below cap) | `daf6c05` |

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
