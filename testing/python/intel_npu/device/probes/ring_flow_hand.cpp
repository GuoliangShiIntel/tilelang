// Phase 3b step A: the multi-tile ping-pong matmul flow, hand-written against npu::Ctx in the
// shape the compiler will emit (docs/examples/pingpong_multitile/README.md sections 1-3).
//
//   Y[M, N] = X[M, K] . W[N, K]^T   fp16
//   W is SPLIT across tiles (tile t owns columns [t*BN, (t+1)*BN)) and resident in shared CMX;
//   X streams in chunks of BM rows through a 2-slot ring: chunk c -> slot c % 2 = SHAVE c % 2,
//   one multicast fill per chunk issued by tile 0's SHAVE (c % 2) as two halves on ports 0/1,
//   READY/FREE hardware barriers; each SHAVE stores its own output block.
#include <tl_templates/intel_npu/npu_rt.hpp>

static constexpr std::uint32_t M = 2048, K = 2048, N = 768;
static constexpr std::uint32_t BM = 32, BN = 256;   // BM * BN = 8192 DPU output points
static constexpr std::uint32_t TILES = 3;           // N / BN
static constexpr std::uint32_t kReadyBase = 8, kFreeBase = 12, kSyncBase = 16 * TILES - 3;

static inline void ring_flow(npu::Ctx &ctx, npu::Ddr<half> X, npu::Ddr<half> W, npu::Ddr<half> Y) {
  const int32_t tile = ctx.tile_id();
  const int32_t shave = ctx.shave_id();
  half *W_cmx = ctx.alloc_shared<half>(BN * K);
  half *X_cmx = ctx.alloc_shared<half>(2 * BM * K);  // the ring's two slots
  half *Y_cmx = ctx.alloc<half>(2 * BM * BN);        // double-buffered by use parity (submit-ahead)
  // Resident SEGMENTED operand: the tile's SHAVEs each load half the rows, then meet.
  {
    const std::uint32_t rows = BN / 2;
    ctx.dma_load(W_cmx + shave * rows * K, &(W[(tile * BN + shave * rows) * K]), K * 2, rows, K * 2, K * 2);
  }
  ctx.sync_tile();
  const std::uint32_t nchunks = M / BM;
  npu::Ctx::Ring ring = ctx.ring_begin(X_cmx, BM * K * sizeof(half), 2, nchunks, kReadyBase, kFreeBase);
  const half *X_slot = ctx.ring_slot<half>(ring);
  ctx.ring_fill(ring, 0, (&(X[(ctx.ring_chunk(0) * BM) * K])).addr, BM * K * sizeof(half));
  ctx.ring_dpu_submit(ring, 0, Y_cmx, X_slot, W_cmx, BM, BN, K, 4, 0);  // VPUNN: CUBOID_4x16 x=4 for 32x256x1024
  for (std::uint32_t u = 0; u < ctx.ring_uses(ring); ++u) {
    const std::uint32_t c = ctx.ring_chunk(u);
    ctx.ring_dpu_wait(ring, u);
    ctx.ring_dpu_submit(ring, u + 1, Y_cmx + ((u + 1) & 1u) * BM * BN, X_slot, W_cmx, BM, BN, K, 4, 0);
    ctx.ring_fill(ring, u + 1, (&(X[(ctx.ring_chunk(u + 1) * BM) * K])).addr, BM * K * sizeof(half));
    ctx.dma_store(&(Y[(c * BM) * N + tile * BN]), Y_cmx + (u & 1u) * BM * BN, BN * 2, BM, N * 2, BN * 2);
  }
  ctx.ring_end(ring);
}

void kernel_entry(int numParams, uint64_t *params, const npu::Environment &env) {
  // hardware barrier ids used by this kernel: ring READY 8..11, FREE 12..15; sync_grid 45..47
  npu::Ctx ctx(env, numParams, params, 3, /*sync_barrier_base=*/kSyncBase);
  if (!ctx.ok()) { ctx.finish(); return; }
  ring_flow(ctx, npu::Ddr<half>(params[0]), npu::Ddr<half>(params[1]), npu::Ddr<half>(params[2]));
  ctx.finish();
}
