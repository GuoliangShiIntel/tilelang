"""P2 prefill device golden: MoE routing + grouping (P2a) and the grouped expert FFN (P2b) for `N` prompt
tokens, at Qwen3.8's shape by default (H 2048, 256 experts of SwiGLU inter 512, top-8 + a shared expert,
int4 g128).

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, VPUNN_ROOT set):
    python testing/python/intel_npu/device/run_moe_prefill_device.py [--n 64] [--phase route|expert|both]
                                                                     [--mb 16] [--no-build] [--negative]
Unprofiled by default (a timing run, TURBO). The expert phase's point is that an expert's weights are read
ONCE per launch; the line it prints compares the bytes it must read against what decode would read for the
same tokens, which is the number prefill exists to move. `--negative` perturbs one expert's weights: only
the tokens routed to that expert may move, and the runner checks exactly that set.
"""

import sys
import os
import re
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401
import runner  # noqa: E402
import lowering  # noqa: E402
from kernels import (gm_scales, make_moe_expert_prefill, make_moe_route_prefill,  # noqa: E402
                     moe_expert_prefill_reference, moe_route_prefill_reference, pack_int4, make_moe_combine)

TILES, SHAVES = 3, 2
WORKERS = TILES * SHAVES
H, I, E, TOPK, G, BNR = 2048, 512, 256, 8, 128, 16
RR = -(-(E + 1) // (WORKERS * BNR)) * (WORKERS * BNR)


def arg(argv, name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def build(src, name, argv):
    if "--no-build" in argv:
        _, sha = runner.write_kernel(src, name)
        return runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe",
                                  kernel=name, source_sha256=sha)
    t0 = time.time()
    db = runner.build(src, name)
    print(f"build {time.time() - t0:.0f} s")
    return db


def reference_combine_source(tokens, choices):
    root = Path(runner.require_env("NPU_DSL_ROOT"))
    sys.path.insert(0, str(root))
    from dataclasses import replace
    from npu_dsl.fused.ir import KernelDef, TensorSpec
    from npu_dsl.fused.tracer import trace
    from npu_dsl.fused.emit import emit
    from npu_dsl.frontend.kernel_specs.qwen36_fused import Qwen36FusedConfig
    from npu_dsl.lib.row_phase import row_phase, Rows, Gather
    from run_npudsl_layer_device import WRAPPER

    def source(nl, cfg, values, inverse, weights, residual, output):
        def combine(row, token):
            for choice in nl.range(0, choices, name="matchedSlot"):
                nl.shave.axpy(row.acc, row.y[choice, :], row.wts.at(choice))
        row_phase(nl, name="matched", rows=tokens, block_rows=4, compute=combine,
                  inputs={"acc": Rows(residual), "wts": Rows(weights, cols=(0, choices)),
                          "y": Gather(values, Rows(inverse, width=choices), count=choices)},
                  outputs={"acc": Rows(output)})

    tensors = [TensorSpec("Values", (tokens * choices, H), "fp16"),
               TensorSpec("Inverse", (tokens * choices,), "u32"),
               TensorSpec("Weights", (tokens, 16), "fp32"),
               TensorSpec("Residual", (tokens, H), "fp32"), TensorSpec("Output", (tokens, H), "fp32")]
    cfg = replace(Qwen36FusedConfig(), max_tokens=tokens, max_seq=1536, tiles=(("matched_row_block", 4),))
    text = emit(trace(KernelDef("matched_combine", source, tensors, []), cfg))
    common = root / "kernels_poc" / "common"
    for header in ("fused_rt.hpp", "fused_ops.hpp"):
        text = text.replace(f'#include "{header}"', f'#include "{(common / header).as_posix()}"')
    text, count = re.subn(r"FRT_KERNEL_ENTRY\(numParams, params, env\)",
                          "static void frt_layer_entry(int numParams, std::uint64_t* params, const npu::Environment& env)", text)
    assert count == 1 and "ctx.gather_add(" in text
    return text + WRAPPER


def combine_benchmark(argv):
    import tvm
    tokens, choices = arg(argv, "--n", 1152), TOPK + 1
    rng = np.random.default_rng(42)
    order = rng.permutation(tokens * choices).astype(np.int32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    values = rng.normal(size=(order.size, H)).astype(np.float16)
    weights = rng.uniform(size=(tokens, 16)).astype(np.float32)
    offsets = np.full(E + 2, order.size, np.int32)
    sources, lowered = {}, {}
    for block, batch in ((1, False), (4, True)):
        name = f"tl_comb_n{tokens}_cb{block}"
        with tvm.transform.PassContext(config={"tl.intel_npu.plan_workloads": "off", "tl.intel_npu.batch_dma": batch}):
            low = lowering.lower(make_moe_combine(tokens, H, E, TOPK, block, TILES, SHAVES))
        sources[name], lowered[name] = low.source, low
    reference = "--reference" in argv
    if reference:
        sources[f"tl_npudsl_comb_n{tokens}"] = reference_combine_source(tokens, choices)
    built = runner.build_all(sources)
    expected = np.zeros((tokens, H), np.float32)
    for choice in range(choices):
        expected += weights[:, choice:choice + 1] * values[inverse.reshape(tokens, choices)[:, choice]].astype(np.float32)
    for name in sources:
        output = np.full_like(expected, np.nan)
        if name in lowered:
            by_name = {"Part": ("in", values), "Slot": ("in", order), "Wts": ("in", weights),
                       "Off": ("in", offsets), "Out": ("out", output)}
            arrays = lowering.order_arrays(lowered[name], by_name)
        else:
            arrays = [("in", np.zeros(1, np.uint32)), ("in", values), ("in", inverse.astype(np.uint32)),
                      ("in", weights), ("in", np.zeros_like(expected)), ("out", output),
                      ("out", np.zeros(WORKERS * 8, np.uint32))]
        result = runner.run(built[name], arrays, TILES, SHAVES, HERE / "run_moe_combine",
                            repeat=10, warmup=10, timeout_s=120)
        runner.check_status(result.status)
        relative = float(np.abs(output - expected).max() / max(float(np.abs(expected).max()), 1e-6))
        assert np.isfinite(output).all() and relative < 2e-3, (name, relative)
        print(f"{name}: {result.median_kernel_ms:.3f} ms [{runner.condition(result, False)}], warm=10/repeat=10; "
              f"rel={relative:.2e} PASS; sha={built[name].source_sha256[:12]}")
    if reference:
        print("Reference: actual npu_dsl row_phase/Gather/axpy at block=4, 9 matched choices; includes an extra zero-residual read.")
    return 0


def main(argv):
    N, MB = arg(argv, "--n", 64), arg(argv, "--mb", 16)
    phase = argv[argv.index("--phase") + 1] if "--phase" in argv else "both"
    if phase == "combine":
        return combine_benchmark(argv)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, H), np.float32).astype(np.float16)
    wr = np.zeros((RR, H), np.float16)
    wr[:E + 1] = (rng.standard_normal((E + 1, H), np.float32) * 0.05).astype(np.float16)
    ref = moe_route_prefill_reference(x, wr, E=E, TOPK=TOPK)
    trials = arg(argv, "--trials", 5)
    rc = 0

    if phase in ("route", "both"):
        low = lowering.lower(make_moe_route_prefill(N=N, H=H, E=E, TOPK=TOPK, BM=min(N, 64), BNR=BNR,
                                                    tiles=TILES, shaves=SHAVES))
        db = build(low.source, f"tl_moe_route_prefill_n{N}_h{H}_e{E}", argv)

        def route_once():
            out = dict(Logits=np.zeros((N, RR), np.float16), Ids=np.zeros((N, TOPK), np.int32),
                       Wts=np.zeros((N, 16), np.float32), Cnt=np.zeros(E + 1, np.int32),
                       Off=np.zeros(E + 2, np.int32), Slot=np.zeros(N * (TOPK + 1), np.int32))
            arrays = {"X": ("in", x), "Wr": ("in", wr), **{k: ("out", v) for k, v in out.items()}}
            r = runner.run(db, lowering.order_arrays(low, arrays), TILES, SHAVES, HERE / "run_moe_prefill",
                           repeat=10, warmup=10, timeout_s=300)
            runner.check_status(r.status)
            # the GEMM against fp32, with a tolerance; the grouping against the ranking of the device's OWN fp16
            # logits, exactly -- in fp16 a near-tie among hundreds of experts ranks differently than in fp32
            lg = float(np.abs(out["Logits"][:, :E + 1].astype(np.float32) - ref["logits"][:, :E + 1]).max()
                       / np.abs(ref["logits"][:, :E + 1]).max())
            dev = moe_route_prefill_reference(x, wr, E=E, TOPK=TOPK, logits=out["Logits"])
            exact = (np.array_equal(out["Ids"], dev["ids"]) and np.array_equal(out["Cnt"], dev["cnt"])
                     and np.array_equal(out["Off"], dev["off"]) and np.array_equal(out["Slot"], dev["slot"])
                     and np.abs(out["Wts"][:, :TOPK + 1] - dev["wts"][:, :TOPK + 1]).max() < 2e-3)
            perm = np.array_equal(np.sort(out["Slot"]), np.arange(N * (TOPK + 1)))
            return out, r, lg, exact and perm and lg < 5e-2

        good, worst_lg, first = 0, 0.0, None
        for _ in range(trials):
            out, res, lg, okt = route_once()
            good += int(okt)
            worst_lg = max(worst_lg, lg)
            if first is None:
                first = out
        flips = int((first["Ids"] != ref["ids"]).sum())
        kind = runner.condition(res, False)
        print(f"route: {res.median_kernel_ms:.3f} ms [{kind}] -> {'PASS' if good == trials else 'FAIL'}; "
              f"{good}/{trials} trials exact on the device's own logits; worst logits rel {worst_lg:.2e}; "
              f"sha {db.source_sha256[:12]}")
        print(f"  {N} tokens, {E + 1} experts: {int(first['Cnt'][:E].max())} tokens on the busiest routed expert, "
              f"{int((first['Cnt'][:E] > 0).sum())} of {E} routed experts used; "
              f"{flips} of {N * TOPK} choices differ from the fp32 ranking (fp16 near-ties)")
        rc |= 0 if good == trials else 1

    if phase in ("expert", "both"):
        gu_int = rng.integers(-8, 8, (E + 1, 2 * I, H), dtype=np.int8)
        gu_s = (rng.standard_normal((E + 1, 2 * I, H // G), np.float32) * 0.02).astype(np.float16)
        d_int = rng.integers(-8, 8, (E + 1, H, I), dtype=np.int8)
        d_s = (rng.standard_normal((E + 1, H, I // G), np.float32) * 0.02).astype(np.float16)
        BNG, BND = arg(argv, "--bng", 64), arg(argv, "--bnd", 64)
        low = lowering.lower(make_moe_expert_prefill(N=N, H=H, I=I, E=E, TOPK=TOPK, G=G, MB=MB,
                                                     BNG=BNG, BND=BND, tiles=TILES, shaves=SHAVES,
                                                     probe_skip="dequant" if "--skip-dequant" in argv else None))
        db = build(low.source, f"tl_moe_expert_prefill_n{N}_h{H}_i{I}_e{E}_mb{MB}_bg{BNG}_bd{BND}"
                   + ("_nodeq" if "--skip-dequant" in argv else ""), argv)

        def run(gu, dw, profile=False):
            out = np.zeros((N, H), np.float32)
            arrays = {"X": ("in", x), "WguQ": ("in", pack_int4(gu.reshape(-1, H))),
                      "WguS": ("in", gm_scales(gu_s.reshape(-1, H // G))),
                      "WdQ": ("in", pack_int4(dw.reshape(-1, I))),
                      "WdS": ("in", gm_scales(d_s.reshape(-1, I // G))), "Wts": ("in", ref["wts"].astype(np.float32)),
                      "Off": ("in", ref["off"].astype(np.int32)), "Slot": ("in", ref["slot"].astype(np.int32)),
                      "Part": ("out", np.zeros((TILES, N, H), np.float32)), "Out": ("out", out)}
            trace = HERE / "evidence" / f"tl_moe_expert_prefill_n{N}_trace.json" if profile else None
            if trace is not None:
                trace.parent.mkdir(parents=True, exist_ok=True)
            r = runner.run(db, lowering.order_arrays(low, arrays), TILES, SHAVES, HERE / "run_moe_prefill",
                           repeat=1 if profile else 3, warmup=0 if profile else 3, timeout_s=600,
                           profile_json=trace)
            runner.check_status(r.status)
            return out, r

        got, res = run(gu_int, d_int)
        if "--skip-dequant" in argv:
            # a measurement arm: the weights are read and the gemms run, but the int4 rows are not expanded,
            # so the result is wrong by design and only the time means anything
            print(f"expert [SKIP-DEQUANT probe, output wrong by design]: {res.median_kernel_ms:.3f} ms "
                  f"[{runner.condition(res, False)}]; sha {db.source_sha256[:12]}")
            return rc
        oracle = moe_expert_prefill_reference(x, gu_int, gu_s, d_int, d_s, ref["wts"], ref["off"], ref["slot"],
                                              TOPK=TOPK, G=G)
        scale = np.abs(oracle).max()
        rels = [float(np.abs(got - oracle).max() / scale)]
        for _ in range(trials - 1):
            rels.append(float(np.abs(run(gu_int, d_int)[0] - oracle).max() / scale))
        good = sum(1 for r in rels if r < 5e-2)
        verdict = "PASS" if good == trials else "FAIL"
        kind = runner.condition(res, False)
        # the kernel skips an expert with no slots, so only the VISITED experts' weights are read
        visited = int((np.diff(ref["off"][:E + 2]) > 0).sum())
        per_expert = (2 * I * H + H * I) // 2
        once, decode_like = visited * per_expert, N * (TOPK + 1) * per_expert
        print(f"expert: {res.median_kernel_ms:.3f} ms [{kind}]; {good}/{trials} trials match the oracle, "
              f"worst rel {max(rels):.2e} -> {verdict}; sha {db.source_sha256[:12]}")
        print(f"  grouped: {visited} of {E + 1} experts visited, {once / 1e6:.0f} MB of int4 read once "
              f"({once / (res.median_kernel_ms * 1e-3) / 1e9:.1f} GB/s); a per-token walk of the same {N} tokens "
              f"would read {decode_like / 1e6:.0f} MB ({decode_like / once:.1f}x)")
        print(f"  the win IS tokens-per-expert: {N * (TOPK + 1) / visited:.1f} here, and it grows with N -- "
              f"at this N the tokens still scatter over most of the {E} experts")
        rc |= 0 if verdict == "PASS" else 1

        if "--profile" in argv:
            # an observability run, never a latency claim: the timing above is the unprofiled one
            _, pr = run(gu_int, d_int, profile=True)
            print(f"  trace [{runner.condition(pr, True)}]: "
                  f"{HERE / 'evidence' / f'tl_moe_expert_prefill_n{N}_trace.json'}")

        if "--negative" in argv:
            victim = int(np.argmax(np.diff(ref["off"][:E + 1])))
            touched = {int(ref["slot"][s]) // (TOPK + 1)
                       for s in range(int(ref["off"][victim]), int(ref["off"][victim + 1]))}
            d2 = d_int.copy()
            d2[victim] = np.clip(d2[victim].astype(np.int16) + 3, -8, 7).astype(np.int8)
            got2, _ = run(gu_int, d2)
            moved = {t for t in range(N) if np.abs(got2[t] - got[t]).max() > 1e-3}
            good = moved == touched
            print(f"negative (expert {victim}, {len(touched)} tokens routed to it): "
                  f"{'behaved' if good else 'BROKEN'} -- {len(moved)} tokens moved, expected {len(touched)}")
            rc |= 0 if good else 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
