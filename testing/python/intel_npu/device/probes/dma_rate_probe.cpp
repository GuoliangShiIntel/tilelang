// DMA rate probe: 128 KiB DDR -> CMX repeatedly in several configurations, timed with NCE_TIME
// (19.2 MHz) on tile 0 SHAVE 0 and reported in status words 10..15 (ticks for kReps transfers):
//   10: unicast 1-D, one port, one issuer          11: unicast 1-D, every SHAVE of every tile streams its OWN
//                                                      kBytes region (6 distinct DDR sources) on port (shave & 1)
//                                                      -- the split weight stream of a decode GEMV / MoE; the
//                                                      rate is the aggregate over the six
//   12: multicast, every SHAVE of every tile issues its 1/(2*tiles) piece as two halves on ports 0/1
//   13: multicast, SHAVE 0 of every tile issues its 1/tiles piece as two halves on ports 0/1
//   14: multicast 1-D, two halves on ports 0 and 1, one issuer (the ring's fill today)
//   15: unicast 1-D, two halves on ports 0 and 1, one issuer
// Concurrent configurations are bracketed by sync_grid so every issuer starts together.
#include <tl_templates/intel_npu/npu_rt.hpp>

#ifndef TL_PROBE_KIB
#define TL_PROBE_KIB 128
#endif
static constexpr std::uint32_t kBytes = TL_PROBE_KIB * 1024, kReps = 16;

static inline std::uint32_t ticks() { return npu::getRegWordVal(0x2f020000U); }

void kernel_entry(int numParams, uint64_t *params, const npu::Environment &env) {
  npu::Ctx ctx(env, numParams, params, 1, /*sync_barrier_base=*/16 * 3 - 3);
  if (!ctx.ok()) { ctx.finish(); return; }
  const std::uint32_t tiles = ctx.num_tiles(), spt = ctx.shaves_per_tile();
  const std::uint32_t tile = ctx.tile_id(), shave = ctx.shave_id();
  const std::uint32_t mask = (1u << tiles) - 1u;
  std::uint8_t *dst = ctx.alloc_shared<std::uint8_t>(kBytes);
  npu::DmaDescriptor *d = ctx.alloc_descriptors<npu::DmaDescriptor>(2);
  std::uint32_t r[6] = {0, 0, 0, 0, 0, 0};
  if (!dst || !d) { ctx.finish(); return; }
  auto &dma = env.getDmaDrv();
  const std::uint64_t src = params[0];
  const bool lead = tile == 0 && shave == 0;
  auto mc = [&](std::uint8_t *p) { return reinterpret_cast<std::uint32_t>(npu::convertToCmxMulticastAddr(p, mask)); };
  // one issuer, one port
  auto run1 = [&](std::uint32_t slot, bool multicast) {
    if (!lead) return;
    const std::uint32_t t0 = ticks();
    for (std::uint32_t i = 0; i < kReps; ++i) {
      npu::DmaDrv::initDesc(d);
      npu::configDmaCopy(d, multicast ? mc(dst) : reinterpret_cast<std::uint32_t>(dst), src, kBytes);
      dma.start(d, 0);
      dma.wait(d);
    }
    r[slot] = ticks() - t0;
  };
  // one issuer, two halves on ports 0 and 1
  auto run2 = [&](std::uint32_t slot, bool multicast) {
    if (!lead) return;
    const std::uint32_t t0 = ticks();
    const std::uint32_t half = kBytes / 2;
    for (std::uint32_t i = 0; i < kReps; ++i) {
      npu::DmaDrv::initDesc(d);
      npu::DmaDrv::initDesc(d + 1);
      npu::configDmaCopy(d, multicast ? mc(dst) : reinterpret_cast<std::uint32_t>(dst), src, half);
      npu::configDmaCopy(d + 1, multicast ? mc(dst + half) : reinterpret_cast<std::uint32_t>(dst + half), src + half, half);
      dma.start(d, 0);
      dma.start(d + 1, 1);
      dma.wait(d);
      dma.wait(d + 1);
    }
    r[slot] = ticks() - t0;
  };
  // several issuers (every tile's SHAVE 0, or every SHAVE), each moving its piece as two halves; multicast
  auto runN = [&](std::uint32_t slot, bool allShaves) {
    const bool mine = allShaves || shave == 0;
    const std::uint32_t issuers = allShaves ? tiles * spt : tiles;
    const std::uint32_t index = allShaves ? tile * spt + shave : tile;
    const std::uint32_t piece = kBytes / issuers, half = piece / 2;
    ctx.sync_grid();
    const std::uint32_t t0 = ticks();
    if (mine) {
      for (std::uint32_t i = 0; i < kReps; ++i) {
        npu::DmaDrv::initDesc(d);
        npu::DmaDrv::initDesc(d + 1);
        const std::uint32_t off = index * piece;
        npu::configDmaCopy(d, mc(dst + off), src + off, half);
        npu::configDmaCopy(d + 1, mc(dst + off + half), src + off + half, half);
        dma.start(d, 0);
        dma.start(d + 1, 1);
        dma.wait(d);
        dma.wait(d + 1);
      }
    }
    ctx.sync_grid();
    if (lead) r[slot] = ticks() - t0;
  };
  // every SHAVE streams its own kBytes region (src + index * kBytes), unicast, one whole job on port shave & 1
  auto runOwn = [&](std::uint32_t slot) {
    const std::uint32_t index = tile * spt + shave;
    ctx.sync_grid();
    const std::uint32_t t0 = ticks();
    for (std::uint32_t i = 0; i < kReps; ++i) {
      npu::DmaDrv::initDesc(d);
      npu::configDmaCopy(d, reinterpret_cast<std::uint32_t>(dst), src + static_cast<std::uint64_t>(index) * kBytes, kBytes);
      dma.start(d, shave & 1);
      dma.wait(d);
    }
    ctx.sync_grid();
    if (lead) r[slot] = ticks() - t0;
  };
  run1(0, false);
  runOwn(1);
  runN(2, true);
  runN(3, false);
  run2(4, true);
  run2(5, false);
  ctx.sync_grid();
  ctx.finish(r, 6);
}
