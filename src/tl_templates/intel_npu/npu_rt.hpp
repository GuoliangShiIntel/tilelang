// Intel NPU (NPU5) runtime surface for TileLang-generated kernels.
//
// The generated kernel receives `npu::Ctx& ctx` (one per SHAVE) and calls:
//   ctx.tile_id() / ctx.shave_id()               launch coordinates
//   ctx.alloc<T>(n)                              SHAVE-private CMX slice
//   ctx.dma_load(cmx, ddr, width, rows, dst_stride, src_stride)
//   ctx.dma_store(ddr, cmx, width, rows, dst_stride, src_stride)
//   ctx.dpu_matmul(out, a, bt, M, N, K)          fp16, out[M,N] = a[M,K] . bt[N,K]^T
// DDR tensors are `npu::Ddr<T>`: a 64-bit address with `[]` and `&` so the
// generated `(&(A[i]))` is an address computation, not a 32-bit pointer.
//
// Two implementations of the same surface:
//   device (default): over the generic_compute toolkit, compiled by moviCompile
//   host (-DNPU_RT_HOST_SIM): a CPU model, compiled by clang++, for the host sim
#ifndef TL_TEMPLATES_INTEL_NPU_NPU_RT_HPP_
#define TL_TEMPLATES_INTEL_NPU_NPU_RT_HPP_

#if defined(NPU_RT_HOST_SIM)
#include "npu_rt_host.hpp"
#elif defined(TL_NPU_RUNTIME_FRT)
#include "npu_rt_device_frt.hpp"   // the same surface over npu_dsl's fused_rt.hpp (its DMA / DPU disciplines)
#else
#include "npu_rt_device.hpp"
#endif

#endif // TL_TEMPLATES_INTEL_NPU_NPU_RT_HPP_
