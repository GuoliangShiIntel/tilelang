---
name: intel-npu-perf-loop
description: The one way to optimise a TileLang intel_npu kernel on PTL/NPU5 — attribution table first (the per-task records: a phase's DMA / DPU / barrier / SHAVE split and its named SHAVE sites, over the WHOLE launch), a roofline for whatever bounds the section, the hot-loop cycle count from the .s, a written prediction, ONE change, a measurement under the SAME condition as the baseline, then the prediction reconciled. Use before any change whose purpose is speed (kernel source, pass parameter, chunk geometry, runtime), when a kernel is slower than npu_dsl's at the same shape, when two numbers from different runs are about to be compared, and when an optimisation did not deliver. Not for correctness bugs or hangs.
---

# Perf loop for intel_npu kernels: attribute, predict, change one thing, reconcile

Why this exists (2026-11-07, the GDN / MoE work): five optimisation attempts in one day were wasted for
three reasons that a procedure prevents — the available trace was not read to the end before acting (its
`dma_wait ~ 0` said SHAVE-bound while a DMA-bound fix was built), two numbers measured under different
conditions were compared (a warm back-to-back kernel against a cold single launch: a 35% "slowdown" that
was the runner's 50 ms poll), and a loop was rewritten before its assembly was read (an old comment about
splat cost was wrong). None of it was a profiler error.

And once more on 2026-11-09, which is why step 1 now names a different instrument: an attribution table was
built from a trace covering **3.1%** of its kernel, and it looked entirely reasonable (24% DPU busy, waits
dominating). The toolkit's profiler keeps 1984 events per SHAVE and drops the rest with no flag, so a
truncated capture is a well-formed file that simply stops early. **Check what a table covers before reading
it**; the per-task records exist so that the answer is "all of it".

## Step 0: the measurement condition, or stop

Every number carries its condition, and two numbers are compared only under the same one:

| field | values | how |
|---|---|---|
| warm / cold | `warm=N` (N back-to-back launches before the timed ones) or `cold` (one launch on an idle device) | `runner.run(warmup=...)`; `decode.py` launches are cold unless `TL_NPU_DBG_WARM_LAUNCH=1` |
| TURBO | TURBO / NO TURBO | the runner's result line |
| profiled | UNPROFILED (a timing) / PROFILED (observability only) | the runner's result line |
| instrumented | a `--phases` build (`TL_NPU_PHASE_PROF=1`) is **never a timing**: measured +5.2% at 112k records and +19% at 189k, and the spread widens 13x | the kernel name carries `_prof` |
| host round trip | ms per launch, from `TL_NPU_DBG_STEP_TIMING=1` or the session self-check | a kernel of 1 ms with a host round trip above ~3 ms means the harness is the subject, not the kernel |

The runner refuses to print a number without `warm`/`cold`; the session prints a null-launch round trip at
start. If either side of a comparison lacks a field, the comparison is void: re-measure, do not reason.

## Step 0.5: audit the reference path before proposing a change

When a reference implementation is available, this audit is mandatory **before the first performance edit**. A
phase label, operation name or tile-size constant is not enough to establish that two kernels do the same work.

1. **Identify the exact artifact and selected branch.** Record the generated reference source path and SHA, build
  configuration, checkpoint/shape/dtype, live versus padded tokens, and the runtime branch actually taken. Read
  the generated C++ at the measured call site plus its owning DSL/library helper. Generated models often have
  separate small-expert and large-expert paths; a shared constant may describe only one of them.
2. **Trace one tile of data and ownership.** For each operand, write its shape, bytes, source/destination layout,
  owner (tile/SHAVE), chunk extent, and reuse lifetime. Then write the exact order of DMA issue/wait, SHAVE
  dequant, DPU submit/completion, stores, and rendezvous. Distinguish ping/pong or twin-buffer overlap from
  serial reuse, and note whether the two SHAVEs submit to one tile-local DPU FIFO.
3. **Map compiler idioms, not just DSL names.** Check whether the reference uses a synchronous GEMM, async submit,
  merged FC, DMA prefetch, or explicit double buffering. Search the generated source and the actual emitter/runtime
  consumer; do not conclude a capability is absent from a high-level helper name or one source snippet.
4. **Name the falsifiable gap.** State the exact reference behavior TileLang lacks, the nearest IR/runtime boundary
  that owns it, and one cheap check that can disprove the hypothesis (generated source, task schedule, or a matched
  narrow A/B). If this evidence is missing, keep the proposed change as a probe, not an optimization claim.

## Step 1: the attribution table, before any edit

Add `--phases` to the device runner and read the table it prints (`device/task_report.py <phases.bin>
[--json out.json]`, the JSON for ui.perfetto.dev). The kernel writes one 16-byte record per TASK — every DMA,
DPU workload, barrier, named SHAVE site and phase — straight to DDR, and the host decodes it offline. That is
VPUX's arrangement (`vpux-plugin/guides/how_to_use_profiling.md`) and npu_dsl's (`-DFRT_PROFILE`), and the
reason for it is coverage: **the cost is O(tasks) in DDR, so a 40-layer launch is attributed whole.**

The table is per phase: `dma_load / dma_store / ring / ring_wait / dpu / barrier / shave`, each as
`us per worker / task count`, plus named SHAVE sites with their per-call spans. Use sites to locate work, not to
declare two engines' work serial or comparable. Always include each site's call count and the corresponding DPU
task count/workload shape; a per-call number without its amount of work is incomplete.

Four things to check before believing it:

* **`DROPPED` must be 0.** The decoder prints it; a non-zero count means `kTaskMax` was hit and the capture is
  a prefix. Truncation is reported, never silent — which is the one thing the toolkit's event profiler gets
  wrong (`Profiler::mark` drops with a bare `return` past 1984 events per SHAVE, no flag).
* **A `--phases` run is observability.** Take the before / after numbers from the clean build.
* **DPU extents are separate from API-site spans.** For a synchronous DPU call, the SHAVE site may include
  submit/wait overhead and observation latency. For an async `matmul_start`/`matmul_async`, the site can end at
  submission while the DPU task executes later. Never compare `gdeq` with an async `ggem` site's duration as if
  that were GEMM compute time.
* **Prove overlap from intervals and generated order.** Use the per-task timeline/Perfetto tracks to measure
  whether a given SHAVE dequant interval intersects a real DPU workload interval, and whether DPU gaps remain.
  A dequant site being shorter than a GEMM site is neither necessary nor sufficient proof of overlap. The target
  is to hide dequant behind preceding DPU work without leaving an exposed dequant tail; compare actual work
  shapes and counts before deciding whether `gdeq < ggem` is a meaningful per-chunk target.
* **DPU extents are derived** from the hardware profiler's duration counted back from the observation point.
  Durations are sound; exact placement against a DMA is not. Use task interval overlap only for ordering the
  instrumented tasks, and reconcile it with the emitted async schedule.
* **Zero `ring` columns can be correct.** Prefill uses `T.serial` + `T.copy` and has no `T.Pipelined` ring at
  all; its DMA is all `dma_load` / `dma_store`. Rings are the decode path.

PTL makes this cheaper than VPUX's general case: every engine's timestamp comes from the one fixed-frequency
19.2 MHz counter, so none of the cross-engine clock calibration the guide describes for NPU37xx/40xx applies.

The older toolkit event trace (`--profile`, `perf_attr.py`) still reads the frozen traces under
`device/evidence/`; it has real `dma_wait` / `dpu_wait` sub-intervals but only until its 1984-event budget
runs out. `perf_attr.py` refuses a trace whose span falls short of the `tl.kernel_ms` stamped into it.

For the phase you intend to touch, write down:

1. **What bounds it** — the largest column. `shave` dominating means the arithmetic is the bound; a large
   `dma_load` with a small `dpu` means the DPU is starved; a large `barrier` means imbalance between workers.
2. **The roofline** of that bound for this section (the number it cannot beat): DMA `bytes / 126 GB/s` two
   ports under turbo plus ~2.5 us a job, DPU `workloads x 2.7 us` at decode shapes or VPUNN cycles for large
   ones, SHAVE the lane-count budget below.
3. **For a SHAVE-bound section, the hot loop's static cycle count** from the assembly
   (`moviCompile -mcpu=5000xx -O3 -S` with the device include paths, then `device/asm_loops.py <file.s>`:
   per loop body the instruction, NOP, load, store and splat counts). Lanes: 16 fp32 / 32 fp16 per 512-bit
   register; `CMU.CP.512.32.R` is a one-instruction splat; VAU fp32 has MUL and ADD (no FMA), 2-cycle
   dependent latency; loads land after ~7 cycles; **moviCompile does not software-pipeline these loops**, so an
   iteration must carry its own independent work (register accumulators, a whole row in registers before any
   store).
4. **The prediction**: section time before -> after, and the per-token effect (x layers). No prediction, no edit.

## Step 2: one change

One variable: the algorithm, or the schedule, or the chunk geometry, or a runtime path — not two. Geometry
changes are checked against the CMX budget on paper first (slots x bytes, both live rings when hoisted over
the same loop) — the lowering's `CheckIntelNpuBudgets` refusal is the gate, not a surprise.

For schedule or geometry experiments, the control and candidate must differ in that one variable. If adding a
capability changes which loop is selected, first prove that in generated source and run a matched control with
the capability enabled but the target loop unchanged. Do not compare a BNG32+async arm with an older BNG64
synchronous arm and attribute the result to either difference.

## Step 3: measure under step 0's condition, then reconcile

Measured gain within ~20% of the prediction: record both in the log entry and move on. Otherwise the
prediction was wrong and the *model* is fixed before the next edit (re-read the trace or the `.s`; a wrong
model that is kept produces the next wasted attempt). A golden failure is a stop, not a note.

## Standing bars

* Not slower than npu_dsl's kernel at the same shape, same day, same runner (`run_npudsl_*_device.py`);
  the per-section table is the comparison, not the total.
* Every quoted number: condition fields, kernel sha256 prefix, `gc_dsl_barrier` commit, INDEX row.
* The hardware facts come from the `intel-npu-hardware` skill first, measurement second, the owner third.

## Cost facts measured on PTL (use these in predictions; re-measure when the runtime changes)

| what | cost | source |
|---|---|---|
| decode ring, per chunk (local ring, one 64-132 KB fill): `fill_issue` ~2.3 us + `acquire` ~2.5 us | ~5 us of SHAVE time per chunk, regardless of bytes | status words 10-15 (ring ticks), MoE 2026-11-07 |
| one `gemm_i4` workload (M=1, 64-256 rows x K 512-4096): submit ~0.85 us + dpu_wait ~2 us | ~2.9 us round trip; a 16-row router ring was 67 us for 6 us of DPU | same |
| the merged-FC SHAVE reduce (q_reduce, 64 cols x 16 groups) | ~1 us: deferring it into the next workload's DPU time (`gemm_i4(defer=True)`) saved 11 us per MoE layer, not the predicted 70 — the dpu_wait is the DPU's own kick-to-watermark latency (hardware question 7), not the reduce | defer A/B 2026-11-07 |
| ring geometry at decode (128-row x 2 slots vs 256-row x 1 slot per SHAVE) | fewer, larger fills are WORSE (+170 us): `DmaDrv::start` spins on the port's previous job (one in flight per port), so one slot per SHAVE serialises fill(u+1) behind fill(u). The per-chunk ~5 us is a latency chain, not a byte or job count | geometry A/B 2026-11-07 |
| fp16 DPU workload M=16, N=96, K=2048 | ~15 us (not the 2.7 us of the M=1 i4 form) | MoE resident router |
| a vectorised elementwise pass over 1024 fp16/fp32 values on one SHAVE (SwiGLU, axpy) | < 2 us: never the bound | perf regions `swig` / `accm` |
| a scalar top-k over 256 logits on one SHAVE | 43 us; on 32 lanes 12 us | perf region `topk` |
| GDN delta rule, DK = DV = 128, both passes at npu_dsl's gdn_scan code quality | ~12 us per head of SHAVE time | `asm_loops.py` on `gdn_scan3.s` |
| synchronous 64 KB DDR<->CMX `dma_load` / `dma_store` | ~4-5 us each | S state per head |
| host round trip per launch through `npu_run --session` | 1.2 ms (was 54 with the 50 ms poll) | `TL_NPU_DBG_STEP_TIMING=1` |

## Instruments, and how to turn them on

* `--phases` on a device runner: the per-task records (`TL_NPU_PHASE_PROF=1` into the build, `task_report.py`
  to decode). The default build has none of it; a `--phases` kernel is named `_prof` so the two ELFs never mix.
* `perf_begin("name")` / `perf_end()` in kernels.py: a named SHAVE site, recorded as its own task under that
  name, so the table gives its per-call span and count. These separate named work from other SHAVE work, but do
  not by themselves establish overlap or provide DPU execution time for an asynchronous GEMM.
* `TL_NPU_DUMP_STATUS=1` + `decode.py`: the status record's ring tick counters per SHAVE (words 10-15, 19.2 MHz ticks:
  acquire, order, submit, dpu_wait, free, fill_issue) — the per-chunk protocol split without any instrumentation.
* `TL_NPU_GEOM="moe_BND=512,..."` (e2e/pack.py): chunk geometry sweeps; the pack cache key follows.
* `run_*_device.py --skip-<stage>`: measurement arms (output wrong by design) for a subtraction when a region is
  not bracketed.

## Mistakes this procedure exists to prevent (dated, so they are not re-learned)

* 2026-11-07: a "0.25 ms per layer in-model slowdown" chased through R2, `.text`, CMX layout — it was the runner
  polling the stream every 50 ms, so the model kernel was measured cold. Step 0 first.
* 2026-11-07: SwiGLU and the expert accumulate predicted at 45 us per item and vectorised first; the bracket then
  measured them at < 2 us. The trace's `dma_wait ~ 0` and the ring tick counters had already said the section was
  ring protocol. Read what is on hand to the end before predicting.
* 2026-11-07: the GDN scan rewritten on a comment claiming a splat costs ~16 lane copies; the `.s` shows
  `CMU.CP.512.32.R` is one instruction, and the real cost was an unpipelined pass 2. Count the hot loop first.
* 2026-11-07: a chunk geometry (BNG 256) that exceeds CMX by 360 KB was built before the slots were added up.
* 2026-11-10: `gdeq` (8.8 us/call) was compared directly with `ggem` (3.8 us/call) in TileLang MoE and read as
  proof that dequant was both slower and unhidden. That comparison skipped the reference generated path: its
  large-expert branch double-buffers per-SHAVE gate/up weights and calls `matmul_async`; its <=2-token branch is a
  separate 32-row path. The TileLang site around a synchronous GEMM and a reference async-submit site do not
  measure the same interval. Initial BNG32+async E2E at 4.77 s was compared with an older 4.54 s BNG64 arm,
  which did not isolate the cause. A later same-prompt BNG32 pair measured sync4.79 s vs async4.77 s once; that
  20 ms difference is within cold-run spread, so there is no stable E2E win. BNG64+async is rejected by the CMX
  budget. Correction: inspect the selected generated branch, compare actual workload shapes/counts and overlap
  intervals, then take a one-variable pair. The target is hidden dequant with no DPU bubble, not an unqualified
  named-site inequality.
