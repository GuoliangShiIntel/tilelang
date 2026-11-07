"""Intel NPU (NPU5 / PTL) hardware facts the backend depends on, each with its provenance.

The source of truth is the VPUX knowledge base (`VPUX_KNOWLEDGE_ROOT`, default the local
`applications.ai.vpu-accelerators.vpux-plugin/knowledges`): `hw/npu5/*.md` summarise the hardware
specification, `compiler/platform-resources.md` the compiler's resource constants. A fact is
recorded here as (value, knowledge file and section, and -- when the backend measured it -- the
measured value and where). The numbers and section names travel; the knowledge text does not
(it is Intel-internal), so this file quotes none of it.

Rules (the `intel-npu-hardware` skill, `.agents/skills/intel-npu-hardware/SKILL.md`):
  * before measuring or asking the hardware owner, look the fact up; a measurement confirms a
    sourced fact or exposes a difference, it does not establish the fact;
  * a constant used by a pass, the runtime, a device script or a plan number comes from here, so
    `test_intel_npu_hw_facts.py` can hold the target defaults and the scripts to it;
  * a fact that is only measured says `source=None` and names the probe, and is a question for
    the knowledge base first and the hardware owner second.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KNOWLEDGE_ROOT = r"C:\Intel\xiong\applications.ai.vpu-accelerators.vpux-plugin\knowledges"


def knowledge_root() -> Path:
    return Path(os.environ.get("VPUX_KNOWLEDGE_ROOT", DEFAULT_KNOWLEDGE_ROOT))


@dataclass(frozen=True)
class Fact:
    value: object
    source: str | None          # "<file>#<section>" under the knowledge root, or None if only measured
    measured: str | None = None  # what this backend measured and where (device script / evidence), if anything
    note: str = ""


FACTS: dict[str, Fact] = {
    # --- tiles, SHAVEs, CMX ------------------------------------------------------------------
    "num_tiles": Fact(3, "hw/npu5/nce-subsystem.md#1 (PTL 3T)"),
    "shaves_per_tile": Fact(2, "hw/npu5/nce-subsystem.md#2.2; compiler/platform-resources.md#5"),
    "cmx_bytes_per_tile": Fact(1536 * 1024, "hw/npu5/nce-subsystem.md#1 (CMX per tile, PTL)"),
    "cmx_workspace_bytes": Fact(1439 * 1024, "compiler/platform-resources.md#3 (NPU5010 CMX_WORKSPACE_SIZE)",
                                measured="Ctx reads 1473536 B free at kernel entry (status word 2; run_dpu_rate_compare 2026-11-07)",
                                note="= the target's cmx_bytes: the budget CheckIntelNpuBudgets enforces"),
    "cmx_workspace_offset_bytes": Fact(0x18000, "compiler/platform-resources.md#2 (CMX_WORKSPACE_OFFSET)"),
    "cmx_shave_stack_bytes": Fact(7 * 1024, "compiler/platform-resources.md#2 (CMX_SHAVE_STACK_SIZE)"),
    "cmx_alignment_bytes": Fact(32, "hw/npu5/nce-subsystem.md#2.3 (32 B aligned addresses for DPU/SHAVE/DMA)"),
    # --- DPU ---------------------------------------------------------------------------------
    "dpu_macs_8bit": Fact(4096, "hw/npu5/dpu.md#1 (256 MPE x 16 fused MACs)"),
    "dpu_fp16_c_steps_per_clk": Fact(8, "hw/npu5/dpu.md#3.4 (FP16: 8 C steps per clock, INT8/FP8: 16)",
                                     note="so the dense fp16 rate is 2048 MAC/clk per DPU: the peak basis of every TOPS figure"),
    "dpu_fp16_macs_per_clk": Fact(2048, "hw/npu5/dpu.md#1 with #3.4",
                                  measured="96.9% of 2048 x 2.0454 GHz at 64x64x1600 8_8 x=8 under NPU_TURBO=YES (run_dpu_rate_compare 2026-11-07)"),
    "dpu_clk_to_cmx_clk": Fact((7, 4), "hw/npu5/nce-subsystem.md#2.1 and #3 (DPU_CLK:CMX_CLK = 7:4)",
                               note="the SHAVE and DMA run on the CMX clock: 4/7 of the DPU clock"),
    "dpu_clk_turbo_ghz": Fact(2.0454, None,
                              measured="npu_dsl's basis (2048 x 2.0454 GHz); IDU durations reach 96.9% of it under turbo",
                              note="question for the knowledge base / hardware owner: the nominal DPU_CLK under NPU_TURBO=YES and without it"),
    "dpu_z_align": Fact(16, "hw/npu5/dpu.md#3 (C mod 16 padding)", note="target dpu_z_align"),
    # --- SHAVE -------------------------------------------------------------------------------
    "shave_vector_bits": Fact(512, "hw/npu5/shave.md#1 and #2.1 (VRF: 32 x 512-bit)",
                              note="32 fp16 / 16 fp32 lanes; the 128-bit (v8f16, SAU.SUMX.f32) builtins are refused by moviCompile -mcpu=5000xx"),
    "shave_lsu_ports": Fact((2, 256), "hw/npu5/shave.md#1 (LSU0/LSU1, 256-bit each)",
                            note="a 512-bit store is two 256-bit stores; 32 B/clk per port to CMX (nce-subsystem.md#2.2)"),
    "shave_cmx_load_latency_clk": Fact(8, "hw/npu5/nce-subsystem.md#2.2 (~8 CMX clocks request to data)"),
    # --- DMA ---------------------------------------------------------------------------------
    "dma_ports": Fact(2, "compiler/platform-resources.md#6 (MAX_DMA_PORTS NPU5010 = 2); hw/npu5/dma-controller.md#1 (2 CTRGs)"),
    "dma_ddr_read_bytes_per_clk_per_group": Fact(64, "hw/npu5/dma-controller.md#1 and #3 (AXI read 64 B/clk per group, 128 B/clk both)",
                                                 measured="TURBO (run_dma_rate_probe 2026-11-07, 512->1024 KiB increments): ~70 GB/s one port, ~126 GB/s two ports streaming, ~2-2.5 us fixed per job; = 64 B/clk at a CMX clock of ~1.1-1.2 GHz (4/7 of 2.0454). NO TURBO (2026-11-06): 34 / 67 GB/s, 4.2-4.5 us"),
    "dma_min_src_width_full_rate_bytes": Fact(64, "hw/npu5/dma-controller.md#3 (6D cap: min(64, SRC_WIDTH) B/clk per job)",
                                              note="a 2-D fill whose row is narrower than 64 B runs at the row width per clock"),
    "dma_link_agent_fifo_entries": Fact(8, "hw/npu5/dma-controller.md#4 (each LA holds a FIFO of 8 entries, a linked list completes before the next entry starts)",
                                        note="the hardware queues jobs per LA; the one-in-flight-per-port rule in the ring is a toolkit DmaDrv chain constraint (plan section 7 question 6)"),
    "dma_descriptor_alignment_bytes": Fact(32, "hw/npu5/dma-controller.md#6 (LINK_ADDR 32 B aligned)"),
    # --- barriers ----------------------------------------------------------------------------
    "barriers_per_tile": Fact(16, "hw/npu5/nce-subsystem.md#4.2 (16 per allocated tile, 127:48 unused on PTL)",
                              measured="probe_barriers.py 2026-11-06: a 3-tile job owns [0, 48); 0-3, 5, 6 never go live, 7 is the toolkit mutex -> pool [8, 48)"),
    "barrier_pool_first": Fact(8, None, measured="probe_barriers.py 2026-11-06 (driver/toolkit reservation, not a hardware fact)"),
    # --- profiling / clocks ------------------------------------------------------------------
    "perf_clk_mhz": Fact(19.2, None,
                         measured="npu_dsl Round 112 and this backend's NCE_TIME stamps vs host wall; the runtime's 38.4 default is MTL/NPU3",
                         note="question for the knowledge base: the FRC / NCE_TIME counter rate on PTL"),
}


def fact(name: str):
    return FACTS[name].value


def fp16_peak_tops(dpu_clk_ghz: float, tiles: int = 3) -> float:
    """Dense fp16 peak on the knowledge-base basis: 2048 MAC/clk per DPU x 2 ops x tiles x clock."""
    return fact("dpu_fp16_macs_per_clk") * 2 * tiles * dpu_clk_ghz / 1e3


def provenance_table() -> str:
    rows = ["| fact | value | source (knowledge root) | measured |", "|---|---|---|---|"]
    for k, f in FACTS.items():
        rows.append(f"| {k} | {f.value} | {f.source or '-'} | {f.measured or '-'} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(f"knowledge root: {knowledge_root()} ({'present' if knowledge_root().is_dir() else 'ABSENT'})")
    print(provenance_table())
