---
name: intel-npu-hardware
description: Look up Intel NPU (NPU4/NPU5 PTL/NPU6) hardware and VPUX compiler facts before measuring, planning or asking the hardware owner — CMX sizes and workspace, SHAVE vector width and ports, DPU MAC rates per data type, DMA bandwidth and link agents, barriers, clocks — and record every fact the TileLang intel_npu backend depends on in `tilelang/intel_npu/hw_facts.py` with its provenance. Use when a number about the NPU is needed, when a device measurement disagrees with an assumption, when writing a plan's hardware question, or when a constant enters a pass, the runtime or a device script.
---

# Intel NPU hardware facts: look up, then measure, then ask

## Where the facts are

The VPUX knowledge base, on this machine at
`C:\Intel\xiong\applications.ai.vpu-accelerators.vpux-plugin\knowledges` (override with
`VPUX_KNOWLEDGE_ROOT`). Start at `KNOWLEDGE.md` (the index), then read only the one file and section
that answers the question:

| question about | file |
|---|---|
| tiles, CMX per tile, CMX layout/alignment, NoC bandwidth, barriers per job, FP16/BF16/FP8 handling, clock ratio DPU:CMX | `hw/npu5/nce-subsystem.md` |
| DPU MACs, MAC throughput per data type, NTHW/NTK stencils, workload/tensor padding, halos | `hw/npu5/dpu.md` |
| SHAVE VLIW slots, VRF width, LSU ports, load latency, L1/L2, addressing windows | `hw/npu5/shave.md` |
| DMA bandwidth per group/port, 6D throughput cap, link agents and their FIFOs, descriptor fields and alignment, barrier fields | `hw/npu5/dma-controller.md` |
| compiler resource constants: CMX workspace size and offset, SHAVE stack, DPU groups, SHAVEs per tile, DMA ports per platform | `compiler/platform-resources.md` |
| how VPUX schedules the same problem (workloads, memory scheduler, multi-SHAVE, flash SDPA) | `compiler/*.md` |

The text is Intel-internal: **numbers, register/field names and `file#section` citations travel into
this repository; copied paragraphs and tables do not.**

## The order of evidence

1. **Look it up.** A fact that is in the knowledge base is cited (`hw/npu5/dpu.md#3.4`), not measured
   into existence. A measurement then *confirms* the fact or *exposes a difference* (driver, toolkit,
   firmware, a clock setting), and the difference is the finding.
2. **Measure** what the knowledge base does not state (driver reservations such as which barrier ids a
   job may use, the toolkit's DMA chain behaviour, turbo clocks) with a probe under
   `testing/python/intel_npu/device/`, UNPROFILED and under `NPU_TURBO=YES` (the runner's default).
3. **Ask the hardware owner** only what neither gives, and write the question into the plan's
   section 7 with what was looked up and what was measured.

Two mistakes this skill exists to prevent (both happened in this backend, 2026-11-06/07): deriving the
DPU's fp16 rate and "a 2.1x slowdown since October" from probes when `dpu.md#3.4` states FP16 runs 8 C
steps per clock against INT8's 16 and the real difference was `NPU_TURBO`; and diagnosing a CMX
shortfall at 1.47 MiB as if the tile had 2 MiB, when `platform-resources.md#3` gives the NPU5010
workspace as 1439 KiB.

## Recording a fact

Every constant the backend depends on lives in `tilelang/intel_npu/hw_facts.py` as a `Fact(value,
source, measured, note)`:

* `source` is `"<file>#<section> (what it says, in your words)"`; several citations are separated by
  `;` (never use `;` inside a parenthetical: the provenance test splits on it);
* `measured` names the probe, the date and the value when this backend measured it;
* `source=None` is allowed only for a driver/toolkit/measured-only fact, and then `note` names the
  open question.

`testing/python/intel_npu/test_intel_npu_hw_facts.py` checks that every citation resolves to an
existing file and section heading, that the intel_npu target's defaults (`src/intel_npu/target.cc`)
equal the facts, and that the fp16 peak basis is derived (`hw_facts.fp16_peak_tops`). Device scripts
take their peak basis from `hw_facts`, never from a literal.

When a fact changes (new knowledge file revision, new driver), change `hw_facts.py` and the target
default together, and say so in the plan log with the old and new citation.

## Reporting

A number in a round, a plan entry or an INDEX row carries its basis: for TOPS, "dense fp16 peak,
2048 MAC/clk per DPU (`dpu.md#3.4`) x 2 x tiles x DPU clock", with the clock and TURBO / NO TURBO
stated. A measured per-op cost on the SHAVE is compared against the 512-bit VRF (`shave.md#2.1`: 32
fp16 or 16 fp32 lanes) at the CMX clock (4/7 of the DPU clock, `nce-subsystem.md#2.1`).
