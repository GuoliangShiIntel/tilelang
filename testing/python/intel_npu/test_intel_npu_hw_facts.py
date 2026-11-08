"""Hardware facts have provenance, and the backend's constants are those facts.

  * every sourced fact names a file under the VPUX knowledge root and a section that exists there
    (skipped when the knowledge root is not on this machine; the facts themselves still hold);
  * the intel_npu target's defaults equal the facts (CMX budget, SHAVEs, tiles, barriers, DMA ports,
    Z alignment): a constant changed in one place without the other fails here;
  * the fp16 peak basis is derived from the facts, not typed into scripts; negative: the int8 MAC count
    is not the fp16 basis.
"""

import re

import pytest

import tilelang  # noqa: F401  (puts tvm on the path)
from tvm.target import Target

from tilelang.intel_npu import hw_facts

SECTION = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+)*)\b", re.M)


@pytest.mark.skipif(not hw_facts.knowledge_root().is_dir(), reason="VPUX knowledge root not present")
def test_every_sourced_fact_names_an_existing_file_and_section():
    root = hw_facts.knowledge_root()
    for name, f in hw_facts.FACTS.items():
        if f.source is None:
            continue
        for ref in f.source.split(";"):
            ref = ref.strip()
            path, _, rest = ref.partition("#")
            p = root / path
            assert p.is_file(), f"{name}: {p} missing"
            section = rest.split()[0].rstrip(".") if rest else ""
            if section and section[0].isdigit():
                headings = set(SECTION.findall(p.read_text(encoding="utf-8", errors="replace")))
                assert section in headings, f"{name}: section {section} not in {path} (has {sorted(headings)[:12]}...)"


def test_target_defaults_are_the_facts():
    t = Target("intel_npu")
    assert int(t.attrs["cmx_bytes"]) == hw_facts.fact("cmx_workspace_bytes") == 1473536
    assert int(t.attrs["shaves_per_tile"]) == hw_facts.fact("shaves_per_tile")
    assert int(t.attrs["num_tiles"]) == hw_facts.fact("num_tiles")
    assert int(t.attrs["barriers_per_tile"]) == hw_facts.fact("barriers_per_tile")
    assert int(t.attrs["barrier_pool_first"]) == hw_facts.fact("barrier_pool_first")
    assert int(t.attrs["dma_ports"]) == hw_facts.fact("dma_ports")
    assert int(t.attrs["dpu_z_align"]) == hw_facts.fact("dpu_z_align")
    assert int(t.attrs["dpu_max_workload_dim"]) == hw_facts.fact("dpu_max_workload_dim") == 8192
    assert int(t.attrs["dpu_z_align"]) == hw_facts.fact("dpu_z_align")
    # the workspace is the physical CMX minus the runtime's reserved head, not the physical size
    assert hw_facts.fact("cmx_workspace_bytes") < hw_facts.fact("cmx_bytes_per_tile")


def test_fp16_peak_basis_is_derived():
    assert hw_facts.fact("dpu_fp16_macs_per_clk") == hw_facts.fact("dpu_macs_8bit") * hw_facts.fact("dpu_fp16_c_steps_per_clk") // 16
    assert abs(hw_facts.fp16_peak_tops(1.85) - 22.73) < 0.01
    assert abs(hw_facts.fp16_peak_tops(2.0454) - 25.13) < 0.01
    assert hw_facts.fp16_peak_tops(2.0454) != hw_facts.fact("dpu_macs_8bit") * 2 * 3 * 2.0454 / 1e3
