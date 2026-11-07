// Vector math behind the intel_npu intrinsic rules for VectorizeLoop'd loops
// (intrin_rule_intel_npu.cc): fp16 lanes -> tl_npu_h<name>(halfN), fp32 lanes ->
// tl_npu_<name>f(floatN). Included by npu_rt_device.hpp (which first defines the
// hardware fast paths: VAU.EXP/LOG/TANH on half32) and npu_rt_host.hpp (clang++
// ext_vector typedefs). Everything here is a lane loop over the scalar helper, so
// the host and the SHAVE agree bit for bit except where the SHAVE uses the VAU.
#ifndef TL_TEMPLATES_INTEL_NPU_NPU_RT_VECMATH_HPP_
#define TL_TEMPLATES_INTEL_NPU_NPU_RT_VECMATH_HPP_

#define TL_NPU_LANEWISE(NAME, V, SCALAR_EXPR)                                  \
  static inline V NAME(V x) {                                                  \
    V r;                                                                       \
    for (int i = 0; i < (int)(sizeof(V) / sizeof(x[0])); ++i) {               \
      const auto v = x[i];                                                     \
      r[i] = SCALAR_EXPR;                                                      \
    }                                                                          \
    return r;                                                                  \
  }

#define TL_NPU_VEC_F16(NAME)                                                   \
  TL_NPU_LANEWISE(tl_npu_h##NAME, half8, tl_npu_h##NAME(v))                    \
  TL_NPU_LANEWISE(tl_npu_h##NAME, half16, tl_npu_h##NAME(v))                   \
  TL_NPU_LANEWISE(tl_npu_h##NAME, half32, tl_npu_h##NAME(v))
#define TL_NPU_VEC_F32(NAME)                                                   \
  TL_NPU_LANEWISE(tl_npu_##NAME##f, float4, NAME##f(v))                        \
  TL_NPU_LANEWISE(tl_npu_##NAME##f, float8, NAME##f(v))                        \
  TL_NPU_LANEWISE(tl_npu_##NAME##f, float16, NAME##f(v))

#ifndef TL_NPU_VEC_HAS_VAU_EXP
TL_NPU_VEC_F16(exp)
#endif
#ifndef TL_NPU_VEC_HAS_VAU_LOG
TL_NPU_VEC_F16(log)
#endif
#ifndef TL_NPU_VEC_HAS_VAU_TANH
TL_NPU_VEC_F16(tanh)
#endif
TL_NPU_VEC_F16(exp2)
TL_NPU_VEC_F16(log2)
TL_NPU_VEC_F16(sqrt)
TL_NPU_VEC_F16(rsqrt)
TL_NPU_VEC_F16(fabs)
TL_NPU_VEC_F16(floor)
TL_NPU_VEC_F16(ceil)
TL_NPU_VEC_F16(round)

TL_NPU_VEC_F32(exp)
TL_NPU_VEC_F32(exp2)
TL_NPU_VEC_F32(log)
TL_NPU_VEC_F32(log2)
TL_NPU_VEC_F32(sqrt)
TL_NPU_LANEWISE(tl_npu_rsqrtf, float4, 1.0f / sqrtf(v))
TL_NPU_LANEWISE(tl_npu_rsqrtf, float8, 1.0f / sqrtf(v))
TL_NPU_LANEWISE(tl_npu_rsqrtf, float16, 1.0f / sqrtf(v))
TL_NPU_VEC_F32(tanh)
TL_NPU_VEC_F32(fabs)
TL_NPU_VEC_F32(floor)
TL_NPU_VEC_F32(ceil)
TL_NPU_VEC_F32(round)

#undef TL_NPU_VEC_F32
#undef TL_NPU_VEC_F16
#undef TL_NPU_LANEWISE

#endif // TL_TEMPLATES_INTEL_NPU_NPU_RT_VECMATH_HPP_
