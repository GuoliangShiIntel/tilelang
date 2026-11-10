"""`perf_attr`'s admissibility gate, seen to fail.

A SHAVE keeps 1984 profiling events and `Profiler::mark` drops the rest with a bare `return` -- no flag, no
overflow bit (kernel_wrapper.cpp). A truncated capture is therefore well-formed JSON that simply stops early,
and the attribution table built from it is believable: the one that started this test covered 3.1% of its
kernel and reported a perfectly plausible 24% DPU busy. The only first-party way to notice is to compare the
span against the kernel's own NCE_TIME duration, which is why `Session.profile` stamps it into the file.

So the gate is exercised in all four states, not merely observed passing in one.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOL = HERE / "device" / "perf_attr.py"


def _trace(span_us: float, kernel_ms: float | None):
    """One cluster, one SHAVE, one DPU track and a single named region spanning `span_us`."""
    ev = [
        {"ph": "M", "pid": 1, "tid": 0, "name": "process_name", "args": {"name": "Cluster 0"}},
        {"ph": "M", "pid": 1, "tid": 1, "name": "thread_name", "args": {"name": "DPU"}},
        {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "SHAVE 0"}},
        {"ph": "X", "pid": 1, "tid": 1, "name": "dpu", "ts": 0.0, "dur": span_us / 2},
        {"ph": "X", "pid": 1, "tid": 2, "name": "work", "ts": 0.0, "dur": span_us},
    ]
    d = {"traceEvents": ev, "displayTimeUnit": "ns"}
    if kernel_ms is not None:
        d["tl"] = {"alias": "k", "kernel_ms": kernel_ms, "launches": 1}
    return d


def _run(tmp_path, d, *extra):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return subprocess.run([sys.executable, str(TOOL), str(p), *extra], capture_output=True, text=True)


def test_a_truncated_trace_is_refused(tmp_path):
    """The failing arm. 3 ms of a 100 ms kernel: the table would describe 3% of the work under the whole
    kernel's name."""
    r = _run(tmp_path, _trace(3000.0, 100.0))
    assert r.returncode != 0, f"the gate passed a 3% trace:\n{r.stdout}"
    out = r.stdout + r.stderr
    assert "TRUNCATED" in out and "3.0%" in out
    assert "1984" in out, "the message must name the cause, not just the symptom"


def test_a_complete_trace_is_accepted(tmp_path):
    """The passing arm, so the refusal above is not simply a tool that always fails."""
    r = _run(tmp_path, _trace(9800.0, 10.0))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "98.0%" in r.stdout and "TRUNCATED" not in r.stdout


def test_an_unstamped_trace_says_it_cannot_be_checked(tmp_path):
    """Silence is the failure this whole gate exists to stop: a trace with no kernel time must SAY so rather
    than print a table as if it were whole."""
    r = _run(tmp_path, _trace(3000.0, None))
    assert "CANNOT BE CHECKED" in r.stdout


def test_allow_truncated_prints_but_keeps_the_label(tmp_path):
    r = _run(tmp_path, _trace(3000.0, 100.0), "--allow-truncated")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "TRUNCATED" in r.stdout, "an override must not also remove the warning"


@pytest.mark.parametrize("frac,refused", [(0.80, True), (0.95, False)])
def test_the_threshold_is_where_it_claims_to_be(tmp_path, frac, refused):
    r = _run(tmp_path, _trace(100.0 * frac * 1000.0, 100.0))
    assert (r.returncode != 0) is refused, f"frac {frac}:\n{r.stdout}{r.stderr}"
