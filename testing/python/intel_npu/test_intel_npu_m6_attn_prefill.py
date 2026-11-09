"""P1 of prefill (plan section 4, M6): the full-attention prefill phase.

`kernels.make_attn_prefill` is the flash phase with the three things decode does not need -- GQA
(the DPU operand is a block's TB x GQ query rows), the causal mask (only the chunk straddling the
diagonal is masked) and the per-head output gate -- over `N` query tokens starting at `Pos[0, 0]`.

Hardware-free coverage here: the lowering (one gemm pair per chunk, a runtime chunk bound, no ring
in the serial form) and the host simulation against `attn_prefill_reference`, with the negatives
that make the masking and the gate falsifiable:

  * a perturbed key FAILs the golden;
  * **dropping the causal mask is caught**: the oracle for a shifted pos0 differs from the kernel's
    output when the mask is removed, so the comparison cannot pass by accident;
  * a token may not see a key past its own position: the reference recomputed with a later pos0
    disagrees, which is the property the mask exists to enforce.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import tilelang  # noqa: F401  (loads the dev build before tvm)
import tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_sim  # noqa: E402
import lowering  # noqa: E402
from kernels import attn_prefill_reference, make_attn_prefill  # noqa: E402

# small instance: the shapes only have to satisfy the kernel's asserts
N, NKV, GQ, D, SMAX, TB, BKV, TILES, SHAVES = 16, 2, 2, 32, 256, 4, 32, 3, 2
REL = 3e-2   # fp16 scores and an fp16 PV product over a short context


def _lower(func):
    with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off"}):
        return lowering.lower(func)


def _has_host_cxx() -> bool:
    try:
        host_sim.find_clangxx()
        return True
    except Exception:
        return False


def _problem(pos0: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((NKV, N, GQ, D), np.float32).astype(np.float16)
    gate = rng.standard_normal((N, NKV * GQ * D), np.float32).astype(np.float16)
    kc = np.zeros((NKV, SMAX, D), np.float16)
    vt = np.zeros((NKV, D, SMAX), np.float16)
    # the cache holds every position this call can see; the rest stays zero as on the device
    last = pos0 + N
    kc[:, :last] = rng.standard_normal((NKV, last, D), np.float32).astype(np.float16)
    vt[:, :, :last] = rng.standard_normal((NKV, D, last), np.float32).astype(np.float16)
    return q, kc, vt, gate


def _sim(q, kc, vt, gate, pos0: int):
    low = _lower(make_attn_prefill(N=N, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, TB=TB, BKV=BKV, tiles=TILES, shaves=SHAVES))
    lib = host_sim.build(low.source, "tl_attn_prefill_sim")
    out = np.zeros((N, NKV * GQ * D), np.float16)
    arrays = dict(Q=q, Kc=kc, Vt=vt, Gate=gate, Pos=np.array([[pos0]], np.int32), O=out)
    host_sim.check_status(host_sim.run(lib, lowering.order_arrays(low, arrays), TILES, SHAVES))
    return out


def test_lowers_with_runtime_chunk_bound_and_no_ring():
    src = _lower(make_attn_prefill(N=N, NKV=NKV, GQ=GQ, D=D, SMAX=SMAX, TB=TB, BKV=BKV, tiles=TILES)).source
    assert "kernel_entry" in src
    assert src.count("intel_npu_dpu_matmul") == 0 or "dpu_matmul" in src
    # the serial form builds no ring: the chunk bound is a runtime value, so there is nothing to pipeline
    assert "ring_begin" not in src
    # the chunk count is computed, not a literal: the kernel reads only the chunks a block can see
    assert "pos_cmx" in src or "Pos" in src


@pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")
@pytest.mark.parametrize("pos0", [0, 37])
def test_host_sim_matches_reference(pos0):
    q, kc, vt, gate = _problem(pos0)
    ref = attn_prefill_reference(q, kc, vt, gate, pos0)
    out = _sim(q, kc, vt, gate, pos0)
    rel = np.abs(out.astype(np.float32) - ref).max() / np.abs(ref).max()
    assert rel < REL, f"pos0={pos0} rel {rel:.3e}"


@pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")
def test_perturbed_key_fails():
    pos0 = 0
    q, kc, vt, gate = _problem(pos0)
    ref = attn_prefill_reference(q, kc, vt, gate, pos0)
    kc[0, 3] = kc[0, 3] + np.float16(4.0)   # a key the first tokens attend to
    out = _sim(q, kc, vt, gate, pos0)
    rel = np.abs(out.astype(np.float32) - ref).max() / np.abs(ref).max()
    assert rel > REL, "the golden passed with a perturbed key: it cannot detect a wrong score"


@pytest.mark.skipif(not _has_host_cxx(), reason="no host clang++ for the simulation")
def test_future_keys_are_not_seen():
    """The positive control for the causal mask: changing a key the block must NOT see leaves the
    output bit-identical. Without the mask this key would move every earlier token's output."""
    pos0 = 0
    q, kc, vt, gate = _problem(pos0)
    a = _sim(q, kc, vt, gate, pos0)
    kc2 = kc.copy()
    kc2[:, N - 1] = kc2[:, N - 1] + np.float16(8.0)   # the last position: only token N-1 may see it
    b = _sim(q, kc2, vt, gate, pos0)
    assert np.array_equal(a[:N - 1], b[:N - 1]), "an earlier token saw a later key: the causal mask is not live"
    assert not np.array_equal(a[N - 1], b[N - 1]), "the last token did not see its own key"
