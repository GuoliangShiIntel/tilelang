---
name: intel-npu-perf-loop
description: The one way to optimise a TileLang intel_npu kernel on PTL/NPU5 — attribution table first (trace segments, wait breakdown, SHAVE compute by direct brackets or subtraction, roofline per section, hot-loop cycle count from the .s), a written prediction, ONE change, a measurement under the SAME condition as the baseline, then the prediction reconciled. Use before any change whose purpose is speed (kernel source, pass parameter, chunk geometry, runtime), when a kernel is slower than npu_dsl's at the same shape, when two numbers from different runs are about to be compared, and when an optimisation did not deliver. Not for correctness bugs or hangs.
---

# Perf loop for intel_npu kernels: attribute, predict, change one thing, reconcile

Why this exists (2026-11-07, the GDN / MoE work): five optimisation attempts in one day were wasted for
three reasons that a procedure prevents — the available trace was not read to the end before acting (its
`dma_wait ~ 0` said SHAVE-bound while a DMA-bound fix was built), two numbers measured under different
conditions were compared (a warm back-to-back kernel against a cold single launch: a 35% "slowdown" that
was the runner's 50 ms poll), and a loop was rewritten before its assembly was read (an old comment about
splat cost was wrong). None of it was a profiler error.

## Step 0: the measurement condition, or stop

Every number carries its condition, and two numbers are compared only under the same one:

| field | values | how |
|---|---|---|
| warm / cold | `warm=N` (N back-to-back launches before the timed ones) or `cold` (one launch on an idle device) | `runner.run(warmup=...)`; `decode.py` launches are cold unless `TL_NPU_DBG_WARM_LAUNCH=1` |
| TURBO | TURBO / NO TURBO | the runner's result line |
| profiled | UNPROFILED (a timing) / PROFILED (observability only) | the runner's result line |
| host round trip | ms per launch, from `TL_NPU_DBG_STEP_TIMING=1` or the session self-check | a kernel of 1 ms with a host round trip above ~3 ms means the harness is the subject, not the kernel |

The runner refuses to print a number without `warm`/`cold`; the session prints a null-launch round trip at
start. If either side of a comparison lacks a field, the comparison is void: re-measure, do not reason.

## Step 1: the attribution table, before any edit

One command per trace: `python testing/python/intel_npu/device/perf_attr.py <trace.json> [--vs <other.json>]`
(segments one cluster's DPU track at idle gaps; per segment: span, DPU busy, `dma_wait` / `dpu_wait` /
`sync_local` per SHAVE, SHAVE compute = span - waits, DMA bytes and GB/s from the DMA track; `--vs` prints the
other trace's segments side by side). Capture with `NPU_PROFILE_FRC_MHZ=19.2`; a PROFILED span is never a
timing claim, only a split.

For the segment you intend to touch, write down:

1. **What bounds it** — the largest of: DMA (`bytes / 126 GB/s` two ports under turbo, + ~2.5 us per job),
   DPU (`workloads x 2.7 us` round trip at decode shapes, or the VPUNN cycles for large ones), SHAVE
   compute (span - waits). `dma_wait ~ 0` means the DMA is not the bound, whatever the bytes say.
2. **The roofline** of that bound for this section (the number it cannot beat).
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
| fp16 DPU workload M=16, N=96, K=2048 | ~15 us (not the 2.7 us of the M=1 i4 form) | MoE resident router |
| a vectorised elementwise pass over 1024 fp16/fp32 values on one SHAVE (SwiGLU, axpy) | < 2 us: never the bound | perf regions `swig` / `accm` |
| a scalar top-k over 256 logits on one SHAVE | 43 us; on 32 lanes 12 us | perf region `topk` |
| GDN delta rule, DK = DV = 128, both passes at npu_dsl's gdn_scan code quality | ~12 us per head of SHAVE time | `asm_loops.py` on `gdn_scan3.s` |
| synchronous 64 KB DDR<->CMX `dma_load` / `dma_store` | ~4-5 us each | S state per head |
| host round trip per launch through `npu_run --session` | 1.2 ms (was 54 with the 50 ms poll) | `TL_NPU_DBG_STEP_TIMING=1` |

## Instruments, and how to turn them on

* `TL_NPU_PERF_REGIONS=1` (kernels.py `perf_begin("name")` / `perf_end()`): named SHAVE spans in the trace, read by
  `perf_attr.py` as `regions:`. Two profiler events per pair against the 1984-event cap.
* `TL_NPU_DUMP_STATUS=1` + `decode.py`: the status record's ring tick counters per SHAVE (words 10-15, 19.2 MHz ticks:
  acquire, order, submit, dpu_wait, free, fill_issue) — the per-chunk protocol split without a trace.
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
