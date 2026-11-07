"""Same-day A/B of npu_dsl's decoder-layer kernels (`qwen36_decode_gdn_layer.cpp`, `qwen36_decode_attn_layer.cpp`:
one launch = mixer + MoE of one layer, norms and fp32 residual inside) through this backend's device runner, on REAL
layer weights of the checkpoint and the first prompt token of a saved reference golden, so their per-layer time can be
set against this backend's GDN / attention + MoE kernels measured the same way.

The kernel is wrapped, not edited (as `run_npudsl_head_device.py`): its `kernel_entry` becomes a static function, the
runtime includes resolve against `NPU_DSL_ROOT/kernels_poc/common`, and our `kernel_entry` stamps NCE_TIME around it.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, NPU_DSL_ROOT set):
    python testing/python/intel_npu/device/run_npudsl_layer_device.py --kind gdn|attn --ckpt DIR --golden golden.npz
                                                                      [--layer N] [--no-build]
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "e2e"))
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401
import runner  # noqa: E402
from kernels import pack_int4  # noqa: E402
from checkpoint import LM, Checkpoint, layer_prefix  # noqa: E402

TILES, SHAVES = 3, 2

WRAPPER = r"""
// ---- tilelang intel_npu: wrapper around npu_dsl's layer kernel for a same-method timing ----
#include <hw/common/include/hw_utils.hpp>
void kernel_entry(int numParams, std::uint64_t* params, const npu::Environment& env) {
  const std::uint32_t t0 = npu::getRegWordVal(0x2f020000U);
  frt_layer_entry(numParams - 1, params, env);            // frt status record: params[numParams - 2], 32 B per SHAVE
  const std::uint32_t t1 = npu::getRegWordVal(0x2f020000U);
  const auto& info = env.getShaveInfo();
  const std::uint32_t worker = info.getTileId() * info.getNumShavesPerTile() + info.getShaveId();
  const std::uint64_t f = params[numParams - 2] + 32ull * worker;
  const std::uint64_t a = params[numParams - 1] + 64ull * worker;
  npu::DdrAccessor<std::uint32_t> f0(f), f1(f + 4), s0(a), s1(a + 4), s8(a + 32), s9(a + 36);
  s0 = (std::uint32_t)f0;
  s1 = (std::uint32_t)f1;
  s8 = t0;
  s9 = t1;
  npu::sync::memoryFence();
}
"""


def wrap_source(npu_dsl_root: Path, kind: str) -> str:
    gen = npu_dsl_root / "kernels_poc" / "qwen36_dsl_poc" / "generated"
    common = npu_dsl_root / "kernels_poc" / "common"
    src = (gen / f"qwen36_decode_{kind}_layer.cpp").read_text(encoding="utf-8")
    for hdr, d in (("fused_rt.hpp", common), ("fused_ops.hpp", common), ("qwen36_swiglu.hpp", gen)):
        src = src.replace(f'#include "{hdr}"', f'#include "{(d / hdr).as_posix()}"')
    src, n = re.subn(r"FRT_KERNEL_ENTRY\(numParams, params, env\)",
                     "static void frt_layer_entry(int numParams, std::uint64_t* params, const npu::Environment& env)", src)
    assert n == 1
    return src + WRAPPER


def q4(ck, prefix):
    """npu_dsl's int4 operand: (packed bytes [N, K/2] low nibble first, group-major fp16 scales [K/G, N])."""
    v, s, bits = ck.quantized(prefix)
    assert bits == 4
    return pack_int4(v), np.ascontiguousarray(s.T)


def moe_arrays(ck, i):
    c = ck.cfg
    p = layer_prefix(i) + "mlp."
    E, I, H = c.n_experts, c.moe_inter, c.hidden
    gu = np.empty((E, 2 * I, H // 2), np.uint8)
    gus = np.empty((E, H // 128, 2 * I), np.float16)
    d = np.empty((E, H, I // 2), np.uint8)
    ds = np.empty((E, I // 128, H), np.float16)
    for e in range(E):
        g, gs = q4(ck, f"{p}experts.{e}.gate_proj")
        u, us = q4(ck, f"{p}experts.{e}.up_proj")
        gu[e, :I], gu[e, I:] = g, u
        gus[e, :, :I], gus[e, :, I:] = gs, us
        d[e], ds[e] = q4(ck, f"{p}experts.{e}.down_proj")
    sg, sgs = q4(ck, p + "shared_expert.gate_proj")
    su, sus = q4(ck, p + "shared_expert.up_proj")
    sd, sds = q4(ck, p + "shared_expert.down_proj")
    dense = np.zeros((320, H), np.float16)
    dense[:E] = ck.f16(p + "gate.weight")
    dense[E] = ck.f16(p + "shared_expert_gate.weight")[0]
    return dict(PostNorm=ck.f16(layer_prefix(i) + "post_attention_layernorm.weight"), WDense=dense,
                WSh=np.concatenate([sg, su]), WShS=np.concatenate([sgs, sus], axis=1), ExpGU=gu, ExpGUS=gus, ExpD=d, ExpDS=ds,
                ShDown=sd, ShDownS=sds)


def main(argv):
    root = os.environ.get("NPU_DSL_ROOT")
    if not root:
        print("NPU_DSL_ROOT is not set")
        return 2
    kind = argv[argv.index("--kind") + 1]
    ckpt = Path(argv[argv.index("--ckpt") + 1])
    gold = np.load(argv[argv.index("--golden") + 1])
    ck = Checkpoint(ckpt)
    c = ck.cfg
    H = c.hidden
    layer = int(argv[argv.index("--layer") + 1]) if "--layer" in argv else (0 if kind == "gdn" else c.layer_types.index("full_attention"))
    assert c.layer_types[layer] == ("linear_attention" if kind == "gdn" else "full_attention"), layer
    kernel = f"tl_npudsl_{kind}_layer_h{H}"
    src = wrap_source(Path(root), kind)
    if "--no-build" in argv:
        _, sha = runner.write_kernel(src, kernel)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=kernel, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(src, kernel)
        print(f"build {time.time() - t0:.0f} s")
    t0 = time.time()
    import torch
    tok0 = int(gold["prompt_ids"][0])
    h_in = (ck.raw(LM + "embed_tokens.weight")[tok0].to(torch.float32).numpy() if layer == 0 else gold["trace_out"][layer - 1]).astype(np.float32)
    h_ref = gold["trace_out"][layer].astype(np.float32)
    p = layer_prefix(layer)
    moe = moe_arrays(ck, layer)
    frt_status = ("out", np.full(TILES * SHAVES * 8, 0xFFFFFFFF, np.uint32))
    if kind == "gdn":
        a = p + "linear_attn."
        qkv, qkvs = q4(ck, a + "in_proj_qkv")
        z, zs = q4(ck, a + "in_proj_z")
        wo, wos = q4(ck, a + "out_proj")
        wba = np.concatenate([ck.f16(a + "in_proj_b.weight"), ck.f16(a + "in_proj_a.weight")])
        conv = ck.f16(a + "conv1d.weight").reshape(c.conv_dim, c.conv_k)
        arrays = [("in", np.zeros(1, np.uint32)), ("in", h_in), ("out", np.zeros(H, np.float32)), ("in", ck.f16(p + "input_layernorm.weight")),
                  ("in", np.concatenate([qkv, z])), ("in", np.concatenate([qkvs, zs], axis=1)), ("in", wba), ("in", conv),
                  ("in", ck.f32(a + "A_log")), ("in", ck.f32(a + "dt_bias")), ("in", ck.f16(a + "norm.weight")), ("in", wo), ("in", wos),
                  ("in", moe["PostNorm"]), ("in", moe["WDense"]), ("in", moe["WSh"]), ("in", moe["WShS"]), ("in", moe["ExpGU"]), ("in", moe["ExpGUS"]),
                  ("in", moe["ExpD"]), ("in", moe["ExpDS"]), ("in", moe["ShDown"]), ("in", moe["ShDownS"]),
                  ("inout", np.zeros((3, c.conv_dim), np.float16)), ("inout", np.zeros((c.lin_vh, c.lin_dk, c.lin_dv), np.float32)),
                  ("out", np.zeros(12288, np.float16)), ("out", np.zeros(64, np.float16)), ("out", np.zeros(4096, np.float16)),
                  ("out", np.zeros(H, np.float32)), ("out", np.zeros(320, np.float16)), ("out", np.zeros((8, 1024), np.float16)),
                  ("out", np.zeros(1024, np.float16)), ("out", np.zeros((9, H), np.float16)), ("out", np.zeros((16, 2, 128), np.float32)),
                  ("out", np.zeros(4096, np.float16)), ("out", np.zeros((32, 2), np.float32)), frt_status]
        ours_ms = "0.811 (GDN) + 0.560 (MoE) = 1.37"
    else:
        a = p + "self_attn."
        q, qs = q4(ck, a + "q_proj")
        k, ks = q4(ck, a + "k_proj")
        v, vs = q4(ck, a + "v_proj")
        wo, wos = q4(ck, a + "o_proj")
        SMAX = 65792
        arrays = [("in", np.array([0], np.uint32)), ("in", h_in), ("out", np.zeros(H, np.float32)), ("in", ck.f16(p + "input_layernorm.weight")),
                  ("in", np.concatenate([q, k, v])), ("in", np.concatenate([qs, ks, vs], axis=1)), ("in", ck.f16(a + "q_norm.weight")),
                  ("in", ck.f16(a + "k_norm.weight")), ("in", wo), ("in", wos),
                  ("in", moe["PostNorm"]), ("in", moe["WDense"]), ("in", moe["WSh"]), ("in", moe["WShS"]), ("in", moe["ExpGU"]), ("in", moe["ExpGUS"]),
                  ("in", moe["ExpD"]), ("in", moe["ExpDS"]), ("in", moe["ShDown"]), ("in", moe["ShDownS"]),
                  ("inout", np.zeros((c.n_kv, SMAX, c.head_dim), np.float16)), ("inout", np.zeros((c.n_kv, c.head_dim, SMAX), np.float16)),
                  ("out", np.zeros(9216, np.float16)), ("out", np.zeros((16, 256), np.float16)), ("out", np.zeros((96, 8, 256), np.float16)),
                  ("out", np.zeros((96, 8, 2), np.float32)), ("out", np.zeros(4096, np.float16)), ("out", np.zeros(H, np.float32)),
                  ("out", np.zeros(320, np.float16)), ("out", np.zeros((8, 1024), np.float16)), ("out", np.zeros(1024, np.float16)),
                  ("out", np.zeros((9, H), np.float16)), frt_status]
        ours_ms = "0.563 (attention) + 0.560 (MoE) = 1.12"
    wbytes = sum(x.nbytes for m, x in arrays if m == "in") - h_in.nbytes
    print(f"layer {layer} ({kind}) weights {wbytes / 1e6:.0f} MB prepared in {time.time() - t0:.0f} s")
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_npudsl_layer", repeat=10, warmup=10, timeout_s=900)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    h_out = arrays[2][1]
    contrib = h_ref - h_in
    err = float(np.abs(h_out - h_ref).max())
    scale = float(np.abs(contrib).max())
    verdict = "PASS" if err / scale < 2e-2 else "FAIL"
    print(f"npu_dsl {kind} layer {layer} (mixer + MoE, norms + fp32 residual in-kernel): kernel-only {res.median_kernel_ms:.3f} ms "
          f"(host wall {res.median_ms:.3f}) [{runner.condition(res, False)}]; max |h_out - h_ref| {err:.4f} over the layer's "
          f"contribution {scale:.3f} (rel {err / scale:.2e}) -> {verdict}; kernel sha256 {db.source_sha256[:12]}")
    if res.kernel_times_ms:
        print("  kernel-only per run ms: " + " ".join(f"{t:.3f}" for t in res.kernel_times_ms))
    print(f"  this backend, same blocks measured standalone (TURBO): {ours_ms} ms")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
