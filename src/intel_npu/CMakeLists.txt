# Intel NPU backend: target kind, launch validation and SHAVE C++ codegen.
#
# These sources emit text only and need neither the generic_compute SDK nor
# MoviTools, so they are always compiled. SDK-dependent sources (none yet)
# would get a USE_INTEL_NPU gate when they appear.

file(GLOB TILE_LANG_INTEL_NPU_SRCS
  src/intel_npu/*.cc
  src/intel_npu/codegen/*.cc
  src/intel_npu/op/*.cc
  src/intel_npu/transform/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_INTEL_NPU_SRCS})
