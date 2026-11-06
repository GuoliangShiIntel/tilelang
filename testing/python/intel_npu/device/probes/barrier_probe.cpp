// Hardware-barrier probe for the Intel NPU5: is barrier id `id` usable by a kernel on this driver?
//
// Hand-written (not generated). Runs on every SHAVE of every tile. Worker 0 pushes the config
// (producers = consumers = number of SHAVEs), everyone rendezvous (software), reads the live
// counters, produces, waits for the producer count to reach zero, consumes, waits for the
// consumer count to reach zero. Every wait is bounded, so a dead barrier reports instead of
// hanging. Result per SHAVE, 8 x u32 at params[0] + 32 * worker:
//   [0] code: 1 OK; 2 config never went live; 3 producer count never reached 0;
//       4 consumer count never reached 0; 5 counters after reset not zero
//   [1] live prod count after push   [2] live cons count after push
//   [3] prod count after produce+wait [4] cons count after consume+wait
//   [5] spins used (producer wait)    [6] barrier id
// params[1]: the barrier id (u32 in DDR). params[2]: the usual status area.
#include <tl_templates/intel_npu/npu_rt.hpp>
#include <shave/hw/nce/include/hgl_barriers_40xx.hpp>

namespace hgl = npu::hw::nce::barriers::hgl;

static constexpr std::uint32_t kSpinLimit = 4000000u;

void kernel_entry(int numParams, uint64_t *params, const npu::Environment &env) {
  npu::Ctx ctx(env, numParams, params, 2);
  if (!ctx.ok()) { ctx.finish(); return; }
  const std::uint32_t id = *npu::DdrAccessor<std::uint32_t>(params[1]);
  const std::uint32_t worker = ctx.tile_id() * ctx.shaves_per_tile() + ctx.shave_id();
  const std::uint8_t n = static_cast<std::uint8_t>(ctx.num_tiles() * ctx.shaves_per_tile());
  const std::uint64_t out = params[0] + 32ull * worker;
  std::uint32_t r[8] = {0, 0, 0, 0, 0, 0, id, 0};

  if (worker == 0) hgl::pushCounters(id, n, n);
  ctx.sync_grid();  // software rendezvous: the config has been pushed

  std::uint8_t p = 0, c = 0;
  std::uint32_t spins = 0;
  do { hgl::getCounters(id, &p, &c); } while ((p != n || c != n) && ++spins < kSpinLimit);
  r[1] = p; r[2] = c;
  if (p != n || c != n) {
    r[0] = 2;
  } else {
    hgl::produceBarrier(id);
    spins = 0;
    do { hgl::getCounters(id, &p, &c); } while (p != 0 && ++spins < kSpinLimit);
    r[3] = p; r[5] = spins;
    if (p != 0) {
      r[0] = 3;
    } else {
      hgl::consumeBarrier(id);
      spins = 0;
      do { hgl::getCounters(id, &p, &c); } while (c != 0 && ++spins < kSpinLimit);
      r[4] = c;
      r[0] = (c != 0) ? 4 : 1;
    }
  }
  for (int i = 0; i < 8; ++i) {
    npu::DdrAccessor<std::uint32_t> w(out + 4u * i);
    w = r[i];
  }
  ctx.sync_grid();  // nobody leaves before every SHAVE has recorded
  ctx.finish();
}
