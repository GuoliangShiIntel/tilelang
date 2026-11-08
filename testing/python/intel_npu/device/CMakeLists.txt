# Device project for TileLang Intel NPU kernels: the host runner (npu_run) over
# the generic_compute npu_runtime, plus every generated kernel under
# generated/*.cpp compiled for the SHAVEs with moviCompile.
#
# Configure with GENERIC_COMPUTE_ROOT and MV_TOOLS_PATH set (environment or
# -D); both are refused when missing, never defaulted.
cmake_minimum_required(VERSION 3.20)
project(tilelang_intel_npu_device)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED GENERIC_COMPUTE_ROOT)
  if(DEFINED ENV{GENERIC_COMPUTE_ROOT})
    set(GENERIC_COMPUTE_ROOT "$ENV{GENERIC_COMPUTE_ROOT}")
  else()
    message(SEND_ERROR "GENERIC_COMPUTE_ROOT is not defined (path to the generic_compute checkout)")
  endif()
endif()
if(NOT DEFINED MV_TOOLS_PATH AND NOT DEFINED ENV{MV_TOOLS_PATH})
  message(SEND_ERROR "MV_TOOLS_PATH is not defined (path to the MoviTools release)")
endif()
set(GENERIC_COMPUTE_ROOT "${GENERIC_COMPUTE_ROOT}" CACHE PATH "generic_compute checkout")
add_subdirectory(${GENERIC_COMPUTE_ROOT} ${CMAKE_BINARY_DIR}/npu_framework)

add_executable(npu_run "${CMAKE_CURRENT_SOURCE_DIR}/host/npu_run.cpp")
target_link_libraries(npu_run PRIVATE npu_runtime)
target_include_directories(npu_run PRIVATE ${GENERIC_COMPUTE_ROOT})

# The generated kernels include <tl_templates/intel_npu/npu_rt.hpp> from the
# TileLang source tree (this file lives at testing/python/intel_npu/device/).
get_filename_component(TILELANG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../../src" ABSOLUTE)

include(${GENERIC_COMPUTE_ROOT}/scripts/kernels_build.cmake)
set(TL_NPU_DEVICES "5010" CACHE STRING "NPU generations to build the kernels for")
file(GLOB TL_NPU_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/generated/*.cpp")
foreach(device ${TL_NPU_DEVICES})
  foreach(src ${TL_NPU_KERNELS})
    add_kernels(
      SOURCES "${src}"
      INCLUDEDIRS "${TILELANG_SRC}"
      DEVICE "${device}"
    )
  endforeach()
endforeach()

# npu_runtime links `generated_kernels_registry` for npu::getKernelsRegistry(). generic_compute's
# generate_kernels_registry() would embed every ELF of the build directory in a generated source and
# recompile + relink the runner on every kernel change (~21 s of the ~26 s a one-kernel rebuild took);
# this registry reads the ELFs from the build directory at first use (host/kernels_registry.cpp.in).
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/host/kernels_registry.cpp.in" "${CMAKE_BINARY_DIR}/generated_kernels_registry.cpp" @ONLY)
add_library(generated_kernels_registry STATIC "${CMAKE_BINARY_DIR}/generated_kernels_registry.cpp")
add_dependencies(npu_run build_user_kernels)
