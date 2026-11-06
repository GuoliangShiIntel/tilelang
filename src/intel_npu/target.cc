/*!
 * \file tl/intel_npu/target.cc
 * \brief Intel NPU (NPU5 / Panther Lake) target kind registration.
 *
 * Every platform constant the compiler reads lives on this target kind and
 * nowhere else (plan T1). Defaults describe PTL/NPU5; they are overridable
 * per target so a second platform is a set of attribute values, not code.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include "dlpack/dlpack.h"

namespace tvm {

namespace refl = tvm::ffi::reflection;

TVM_REGISTER_TARGET_KIND("intel_npu", kDLExtDev)
    // Platform generation: "npu5" is PTL. Aliases are resolved in Python.
    .add_attr_option<ffi::String>("arch", refl::DefaultValue("npu5"))
    // Tiles (clusters) available to one kernel launch.
    .add_attr_option<int64_t>("num_tiles", refl::DefaultValue(3))
    // SHAVE cores per tile; threadIdx.x ranges over them.
    .add_attr_option<int64_t>("shaves_per_tile", refl::DefaultValue(2))
    // CMX bytes per tile the kernel may allocate: the workspace after the
    // runtime's reserved regions (1439 KiB on PTL, observed at kernel entry and
    // equal to VPUX's NPU5010 workspace), not the 1536 KiB physical size.
    .add_attr_option<int64_t>("cmx_bytes", refl::DefaultValue(1473536))
    // DMA engines per tile a fill may be split across.
    .add_attr_option<int64_t>("dma_ports", refl::DefaultValue(2))
    // Hardware barriers: the job gets barriers_per_tile x (tiles in the job)
    // ids starting at 0; ids below barrier_pool_first are taken or never go
    // live on this driver (device probe 2026-11-06: 0-3, 5, 6 config-not-live;
    // 7 is the toolkit mutex). An id at or above the job's range hangs the run.
    .add_attr_option<int64_t>("barrier_pool_first", refl::DefaultValue(8))
    .add_attr_option<int64_t>("barriers_per_tile", refl::DefaultValue(16))
    // DPU workload legality (NPU5): Z extents multiples of 16, at most this
    // many output points per workload.
    .add_attr_option<int64_t>("dpu_z_align", refl::DefaultValue(16))
    .add_attr_option<int64_t>("dpu_max_output_points", refl::DefaultValue(8192))
    // SHAVE .text budget per kernel, bytes.
    .add_attr_option<int64_t>("text_limit_bytes", refl::DefaultValue(262272))
    // kernel_entry returns void; status travels through a parameter.
    .add_attr_option<bool>("supports_kernel_status_return",
                           refl::DefaultValue(false))
    .set_default_keys({"intel_npu"});

} // namespace tvm
