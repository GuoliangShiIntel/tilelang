// Negative control for the bounded barrier waits: sync_grid on barrier ids 0..2, which never go
// live on this driver. The kernel must return with status kStatusBarrierNotLive (5), detail = 0,
// within the spin bound, instead of hanging the launch. Hand-written; params[0] is the status area.
#include <tl_templates/intel_npu/npu_rt.hpp>

void kernel_entry(int numParams, uint64_t *params, const npu::Environment &env) {
  npu::Ctx ctx(env, numParams, params, 0, /*sync_barrier_base=*/0);
  if (!ctx.ok()) { ctx.finish(); return; }
  ctx.sync_grid();
  ctx.finish();
}
