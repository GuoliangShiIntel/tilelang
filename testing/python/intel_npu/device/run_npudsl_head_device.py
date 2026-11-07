"""Same-day A/B of npu_dsl's lm_head kernel (`kernels_poc/qwen36_dsl_poc/generated/qwen36_head.cpp`, the fused
emitter's final norm + int8 g128 vocab GEMV + greedy argmax) through this backend's device runner, on the same
random problem as `run_head_device.py` (seed 0), unprofiled and TURBO, so the two head kernels are compared under one
method on one day.

The npu_dsl kernel is wrapped, not edited: its `kernel_entry` becomes a static function, its two quoted runtime
includes are resolved against `NPU_DSL_ROOT/kernels_poc/common`, and a `kernel_entry` of ours stamps NCE_TIME around
the call and writes this runner's 64 B status record (the frt record, 32 B per SHAVE, lives in a parameter of its
own). Nothing of npu_dsl is imported or linked.

Usage (GENERIC_COMPUTE_ROOT, MV_TOOLS_PATH, NPU_DSL_ROOT set):
    python testing/python/intel_npu/device/run_npudsl_head_device.py [--v 248320] [--no-build]
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
sys.path.insert(0, str(HERE))

import tilelang  # noqa: E402,F401
import runner  # noqa: E402
from kernels import head_reference  # noqa: E402
from test_intel_npu_m5_head import head_problem  # noqa: E402

TILES, SHAVES = 3, 2
G, H, V_KERNEL = 128, 2048, 248320

WRAPPER = r"""
// ---- tilelang intel_npu: wrapper around npu_dsl's head kernel for a same-method timing ----
#include <hw/common/include/hw_utils.hpp>
void kernel_entry(int numParams, std::uint64_t* params, const npu::Environment& env) {
  const std::uint32_t t0 = npu::getRegWordVal(0x2f020000U);   // NCE_TIME free-running counter
  frt_head_entry(numParams - 1, params, env);             // frt status record: params[numParams - 2], 32 B per SHAVE
  const std::uint32_t t1 = npu::getRegWordVal(0x2f020000U);
  const auto& info = env.getShaveInfo();
  const std::uint32_t worker = info.getTileId() * info.getNumShavesPerTile() + info.getShaveId();
  const std::uint64_t f = params[numParams - 2] + 32ull * worker;
  const std::uint64_t a = params[numParams - 1] + 64ull * worker;
  npu::DdrAccessor<std::uint32_t> f0(f), f1(f + 4), s0(a), s1(a + 4), s8(a + 32), s9(a + 36);
  s0 = (std::uint32_t)f0;   // frt::kStatusOk == 1 == this runner's OK
  s1 = (std::uint32_t)f1;
  s8 = t0;
  s9 = t1;
  npu::sync::memoryFence();
}
"""


def wrap_source(npu_dsl_root: Path) -> str:
    common = npu_dsl_root / "kernels_poc" / "common"
    src = (npu_dsl_root / "kernels_poc" / "qwen36_dsl_poc" / "generated" / "qwen36_head.cpp").read_text(encoding="utf-8")
    assert "LmHead i8[248320, 2048]" in src, "not the Qwen3.6 head kernel this probe was written against"
    for hdr in ("fused_rt.hpp", "fused_ops.hpp"):
        src = src.replace(f'#include "{hdr}"', f'#include "{(common / hdr).as_posix()}"')
    src, n = re.subn(r"FRT_KERNEL_ENTRY\(numParams, params, env\)",
                     "static void frt_head_entry(int numParams, std::uint64_t* params, const npu::Environment& env)", src)
    assert n == 1
    return src + WRAPPER


def main(argv):
    root = os.environ.get("NPU_DSL_ROOT")
    if not root:
        print("NPU_DSL_ROOT is not set (the libraries.npu.dsl checkout)")
        return 2
    kernel = "tl_npudsl_head_h2048_v248320"
    src = wrap_source(Path(root))
    if "--no-build" in argv:
        _, sha = runner.write_kernel(src, kernel)
        db = runner.DeviceBuild(build_dir=HERE / "build", exe=HERE / "build" / "Release" / "npu_run.exe", kernel=kernel, source_sha256=sha)
    else:
        t0 = time.time()
        db = runner.build(src, kernel)
        print(f"build {time.time() - t0:.0f} s")
    t0 = time.time()
    p = head_problem(0, H=H, V=V_KERNEL, G=G)
    ref, tok_ref = head_reference(p["h"], p["norm_w"], p["wlm_int"], p["wlm_scales"], G=G)
    arrays = [
        ("in", np.zeros(1, np.uint32)),                                         # Qwen36HeadConfigs
        ("in", p["h"].astype(np.float32).reshape(-1)),                          # HIn fp32[2048]
        ("in", p["norm_w"].reshape(-1)),                                        # FinalNorm fp16[2048]
        ("in", np.ascontiguousarray(p["wlm_int"])),                             # LmHead i8[V, H]
        ("in", np.ascontiguousarray(p["wlm_scales"].astype(np.float16).T)),     # LmHeadS fp16[16, V] group-major
        ("out", np.zeros(V_KERNEL, np.float16)),                                # Logits
        ("out", np.zeros((TILES * SHAVES, 2), np.float32)),                     # Best
        ("out", np.zeros(1, np.uint32)),                                        # TokenOut
        ("out", np.full(TILES * SHAVES * 8, 0xFFFFFFFF, np.uint32)),            # frt status, 32 B per SHAVE
    ]
    print(f"problem + oracle + packing {time.time() - t0:.0f} s")
    res = runner.run(db, arrays, TILES, SHAVES, HERE / "run_npudsl_head", repeat=10, warmup=10, timeout_s=600)
    print(f"status per SHAVE (code, detail): {res.status[:, :2].tolist()}")
    runner.check_status(res.status)
    logits = arrays[5][1].astype(np.float32)
    err = float(np.abs(logits - ref).max())
    scale = float(np.abs(ref).max())
    tok = int(arrays[7][1][0])
    verdict = "PASS" if err / scale < 2e-2 and tok == tok_ref else "FAIL"
    kind = runner.condition(res, False)
    wbytes = arrays[3][1].nbytes + arrays[4][1].nbytes
    print(f"npu_dsl head (qwen36_head.cpp, vocab_rows=128, 2 slots per worker): kernel-only {res.median_kernel_ms:.3f} ms "
          f"(host wall {res.median_ms:.3f} ms) [{kind}]; max |logits - ref| = {err:.4f} over {scale:.3f} (rel {err / scale:.2e}); "
          f"token {tok} (oracle {tok_ref}) -> {verdict}; kernel sha256 {db.source_sha256[:12]}")
    if res.kernel_times_ms:
        print("  kernel-only per run ms: " + " ".join(f"{t:.3f}" for t in res.kernel_times_ms))
    print(f"weights {wbytes / 1e6:.1f} MB -> {wbytes / (res.median_kernel_ms * 1e-3) / 1e9:.1f} GB/s kernel-only")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
