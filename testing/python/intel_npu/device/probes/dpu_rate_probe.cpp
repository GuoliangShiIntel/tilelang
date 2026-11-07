// DPU rate probe in the npu_dsl method (kernels_poc/dpu_rate_probe): one SHAVE (tile 0, SHAVE 0)
// runs the same fp16 workload TL_ITERS times, start/wait each, and records every workload's
// idu_workload_duration (19.2 MHz ticks) into the output parameter; the host takes the median.
// Knobs are compile-time so the runner can build a sweep in one device build:
//   TL_M TL_N TL_K    workload shape (K is the aligned reduction length)
//   TL_X              DpuMatMul::setSpatial(M, X) when non-zero
//   TL_NTHW TL_ODU    DpuMatMul::setStencil(nthw_ntk, odu_nthw); 99 = keep init()'s value
//   TL_ZEROS          1: operands all zero (the npu_dsl probe's data); 0: pseudo-random fp16
#ifndef TL_M
#define TL_M 64
#define TL_N 64
#define TL_K 1600
#define TL_X 0
#define TL_NTHW 99
#define TL_ODU 99
#define TL_ZEROS 1
#define TL_ITERS 64
#endif
#include <tl_templates/intel_npu/npu_rt.hpp>

void kernel_entry(int numParams, uint64_t *params, const npu::Environment &env) {
  npu::Ctx ctx(env, numParams, params, 1, /*sync_barrier_base=*/16 * 3 - 3);
  if (!ctx.ok()) { ctx.finish(); return; }
  if (ctx.tile_id() != 0 || ctx.shave_id() != 0) { ctx.finish(); return; }
  constexpr std::uint32_t M = TL_M, N = TL_N, K = TL_K;
  half *a = ctx.alloc<half>(M * K);
  half *bt = ctx.alloc<half>(N * K);
  half *out = ctx.alloc<half>(M * N);
  auto *desc = ctx.alloc_descriptors<npu::DpuMatMul::DescriptorsBuffer>(1);
  std::uint32_t *wt = ctx.alloc<std::uint32_t>(4 * 1024);
  if (!ctx.ok()) { ctx.finish(); return; }
  std::uint32_t seed = 12345u;
  auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (half)(float(int((seed >> 9) & 0xFFF) - 2048) / 2048.0f); };
  for (std::uint32_t i = 0; i < M * K; ++i) a[i] = TL_ZEROS ? (half)0.0f : rnd();
  for (std::uint32_t i = 0; i < N * K; ++i) bt[i] = TL_ZEROS ? (half)0.0f : rnd();
  npu::DpuMatMul mm(env.getDpuDrv(), desc, wt);
  mm.init(M, N, K);
  if (TL_X != 0 && !mm.setSpatial(M, TL_X)) { ctx.finish(); return; }
  if (TL_NTHW != 99 && !mm.setStencil(TL_NTHW, TL_ODU == 99 ? npu::DpuMatMul::kKeepOduNthw : TL_ODU)) { ctx.finish(); return; }
  npu::Ddr<std::uint32_t> durs(params[0]);
  for (std::uint32_t it = 0; it < TL_ITERS; ++it) {
    mm.start(out, a, bt);
    mm.wait();
    const std::uint32_t d = *(volatile const std::uint32_t *)&desc->hwpStats.idu_workload_duration;
    durs[it] = d;
  }
  npu::sync::memoryFence();
  ctx.finish();
}
