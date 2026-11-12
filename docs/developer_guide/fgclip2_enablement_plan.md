# FG-CLIP2-so400m on the Intel NPU TileLang backend: enablement plan

Companion to [fgclip2_npu_dsl_explained.md](fgclip2_npu_dsl_explained.md) (what the original `npu_dsl`
implementation does and why) and [intel_npu_backend_plan.md](intel_npu_backend_plan.md) (the backend's own
phase/milestone history; this model follows the same method: host sim first, numpy oracle, correctness before
tuning, "prune -> measure -> pin" rather than porting npu_dsl's tile-size constants as assumptions).

Source of truth for the model: `C:\Intel\guoliang\dsl\libraries.npu.dsl`, branch `feat/fgclip2-so400m`
(`npu_dsl/frontend/examples/fgclip2_layers_source.py`, `npu_dsl/frontend/kernel_specs/fgclip2_fused.py`,
`fgclip2_e2e/reference.py`, `fgclip2_e2e/pack.py`). As of 2026-11-11 this repo (`tilelang`,
`feat/intel-npu-backend`) has no FGCLIP2 code yet — this is a new model target, additive to the Qwen3.8-MoE work
already underway.

## 0. Why this model is a good second target

Mathematically simpler than Qwen3.8-MoE (no MoE, no GDN, no int4 quant in the reference math, no causal mask, no
RoPE, no KV cache — every tower is a single full-length forward pass), so it mainly exercises **compiler/runtime
breadth** the Qwen work has not yet needed:
* LayerNorm **with bias** (Qwen only needed zero-centred RMSNorm).
* GELU-tanh activation (Qwen only needed SiLU/sigmoid).
* **Bidirectional** (non-causal) attention, including the degenerate case of a single-query "pooling" attention
  head (image tower) and a fixed-length unmasked attention (text tower, deliberately including the pad tokens —
  verified against the reference, not a bug to fix).
* Two independent towers (image, text) sharing the same per-layer kernel shape but different tiling constants.
* fp16-weight GEMM at several N/K shapes (no dequant), a good test of the plain `T.gemm` path end to end.

## 1. Building blocks to add (none exist in this backend yet)

| Block | Status | Notes |
|---|---|---|
| LayerNorm with bias, fp32 residual in -> fp16 out | **to write** | Mirror `norm_block`/`prefill_norm_block`'s structure (`testing/python/intel_npu/kernels.py`): vectorized sum/sumsq accumulate (`FL=16` lanes), `tl_npu_vsum` reduce, write back scaled+shifted fp16 row. Needs both the affine (`w`,`b` read) and non-affine (fold-ready) forms, matching `_ln_rows(affine=...)`. |
| GELU-tanh | **to write** | `T.tanh` is a TVM builtin and `npu_rt_vecmath.hpp` already has a VAU.TANH fast path (fp16/fp32 lanes) -- write it the same way the attention kernels use `T.exp` inline in a `T.vectorized` loop; no new runtime op needed. |
| Bias-add GEMM (fp16 weights, fp16/fp32 bias add epilogue, optional GELU, optional residual-add epilogue) | **partially have** | `T.gemm` + a manual epilogue loop; follow `make_proj_prefill`'s `proj_phase` shape (dequant-free simplification) for the loop skeleton. |
| Bidirectional flash attention with a "valid key count" boundary (no causal mask) | **adapt, don't rewrite** | `make_flash_attention_phase` (`testing/python/intel_npu/kernels.py`) is *already* full (non-causal) attention over the whole KV range. Needed changes: replace the fixed `numKv` boundary with a runtime "valid" count for the image tower's padded patches (compare `_flash_chunks`'s `valid - c0` argument), and verify there is no accidental causal assumption hiding in the online-softmax code (there should not be any -- `make_flash_attention_phase` has no causal masking already). |
| Patch embedding / resized position embedding | **host-side, not a kernel concern** | Reference does this on the host with `torch.nn.functional.interpolate`; the kernel only ever receives the already-resized position tensor plus the patch bias added in (`PosB`, see `pack.py::image_inputs`). TileLang side: same split (host prep script, kernel takes a precomputed tensor). |
| Token + position embedding (text tower) | **to write** | Trivial: per-token gather row from `TokEmb[id]` + `PosEmb[t % 64]`, `residual_add`. Pattern already exists for embedding-like gathers elsewhere in the GDN/MoE kernels (indexed DMA load with a runtime offset). |
| Attention-pooling head (single learned query attends over all patches) | **to write, but trivial given flash attention block** | Degenerate case of the same attention building block with `rows=1`. |
| Device-side 27-layer loop with a weight table | **adapt from Qwen's M5** | Reuse the `T.ddr_view` address-table pattern (`GDN_TABLE`/`GDN_PREFILL_TABLE` in `kernels.py`) rather than re-deriving it; FGCLIP2 needs one table row per layer with `Wqkv`/`Bqkv`/`Wo`/`W1`/`B1`/`W2`/`LV` addresses, no `Kind[li]` branch needed (every layer is the same op). |

## 2. Milestones (host sim first; device validation and perf tuning follow, same bar as Qwen: "measure, don't
assume npu_dsl's tile constants transfer")

* **V0 (written 2026-11-11, NOT YET HOST-SIM VALIDATED -- see status note below): building blocks + one
  encoder layer, host sim only.** LayerNorm(bias), GELU-tanh, bias-add GEMM, full unmasked multi-head attention
  (a plain softmax, not `make_flash_attention_phase`'s online one -- V0's row counts are small enough that
  nothing needs to stream; chunked/online attention is deferred to V1 when real patch counts are in play),
  assembled into one pre-LN encoder layer kernel matching `reference.py::encoder_layer` exactly (small clean
  test shapes -- hidden=64, 4 heads of 16, inter=128 -- not the real model's, so no head-dim/MLP padding is
  needed yet; fp16 weights, no folding tricks yet, no dense-head/pooling-head). Code:
  `testing/python/intel_npu/fgclip2_kernels.py` (`make_fgclip2_encoder_layer_v0`, `encoder_layer_reference`),
  test: `testing/python/intel_npu/test_intel_npu_fgclip2_v0.py`. Acceptance: host-sim output within
  `rel < 5e-2` of the numpy oracle (loose on purpose for this first pass -- tighten once it is green), matching
  the backend's existing Phase 1 acceptance style.

  **Status note (updated 2026-11-11)**: the repo has since been built (`pip install -e . --no-build-isolation`,
  a from-scratch build: cloned `3rdparty/cutlass` + `3rdparty/tvm`, compiled with clang-cl/Ninja, no CUDA/ROCm/
  device SDK needed) and the V0 test above **has been run and passes, repeated 6/6 times** (no intermittency).
  One real bug was found and fixed on the first run: `ROWS=8` was refused by `LowerIntelNpuTileOp` ("gemm N=8,
  K=16 must be multiples of 16" -- the attention GEMMs use `rows` as their N or K dimension, not just `hidden`/
  `head_dim`); fixed by using `ROWS=16`. See
  [fgclip2_tilelang_pipeline_explained.md](fgclip2_tilelang_pipeline_explained.md) for the full pipeline
  walkthrough with real captured TIR/C++ from this exact kernel.

  **V1 progress (2026-11-11): the channel-width half is done, the token-count half is not.**
  `make_fgclip2_encoder_layer_v1` (same file) runs one layer at real hidden=1152 / 16 heads of hd_pad=80 /
  inter_pad=4320, with every linear layer streaming TN-row weight tiles from DDR (`linear_bias_tiled`) instead
  of loading the whole matrix resident -- the real weight matrices no longer fit CMX whole. Host-sim validated,
  stable over repeats (`test_intel_npu_fgclip2_v1.py`). Real bug found and fixed: every scratch buffer was
  `alloc_local` (reserved PER SHAVE even though only `shave==0` uses any of it), which doubled the CMX
  footprint and blew the budget (1,995,136 B needed > 1,473,536 B available) -- switched to `alloc_shared`
  (one copy per tile). Still open for V1: real patch/token counts (which will need attention chunking again,
  since V0/V1's plain softmax assumes all keys resident at once), patch embedding, and the pooling/dense heads.

  **V1 progress, part 2 (2026-11-11): the device-side layer loop is done.**
  `make_fgclip2_encoder_layers_v1` runs N copies of the single-layer kernel in ONE launch, reading each layer's
  weights through `T.ddr_view(Table[li, col["Wq"]], ...)` (`FGCLIP2_TABLE_COLS`), reusing Qwen's M5 layer-table
  design verbatim rather than re-deriving it. Host-sim validated at `n_layers=3`, stable over 3 repeats
  (`test_intel_npu_fgclip2_layers_v1.py`), including the "every layer reads its own row" negative control
  (mirrors `test_intel_npu_m6_prefill_model.py`'s own version of that check). No new bugs found on the first
  run this time -- the layer-table mechanics transferred cleanly once the CMX budget lesson from part 1 was
  already applied (every buffer `alloc_shared`, declared once outside the loop).
* **V1: 27-layer vision tower, single launch, no pooling/dense head.** Device-side layer loop (weight table),
  patch embedding + position embedding as kernel inputs (host-prepared), fp32 residual stream in DDR. Oracle:
  `reference.py::vision(..., trace=False)` up to `post_layernorm`, no pooling head yet.
* **V2: attention-pooling head + full vision-tower embedding.** Single-query attention block, out_proj + LN +
  MLP epilogue (`_gemv`-equivalent for the single row). Oracle: `reference.py::vision` full return value.
* **V3: text tower.** Token/position embedding, 27-layer loop (shared layer kernel with V1 if the tiling
  constants allow it, separate if not -- decide by measurement, not by assumption), last-token pooling + head
  projection (+ optional boxtext_head). Oracle: `reference.py::text_short`.
* **V4: weight packing + end-to-end golden check.** Port `pack.py`'s folding tricks (head-dim pad, q
  prescale, k-bias drop, v-bias-into-o_proj-bias fold, LN-into-GEMM fold, MLP pad) verbatim -- they are pure
  math identities, not hardware-specific, so they transfer unchanged. Validate against `golden.py`'s retrieval
  metrics (R@1/R@5), not just per-tensor cosine similarity.
* **V5: device validation + perf tuning.** Device golden check, then the same "prune -> measure -> pin" loop
  Qwen's plan documents (section 4 of `intel_npu_backend_plan.md`) -- do **not** copy npu_dsl's tile-size
  constants (`tn=32`, `mb_a=256`, `bcast_min_rows=384`, ...) as if they were still optimal; they were tuned for
  npu_dsl's hand-written C++ codegen on a specific hardware unit and need re-measuring against this backend's
  own codegen.

## 3. Status summary (2026-11-11): both towers validated end to end against REAL pretrained weights

Everything below is from actually running host-sim (`clang++`, `TL_NPU_HOST_BARRIER_TIMEOUT_S=3600` for the
multi-hundred-second real-scale runs -- the default 120 s watchdog is tuned for much smaller kernels and will
falsely report "STUCK" on a real 27-layer pass, which is not a hang), not read off source. Real checkpoint:
`C:\Intel\guoliang\dsl\fg-clip2-so400m\model.safetensors`. Real oracle: `C:\Intel\guoliang\dsl\fgclip2_data\
golden_fp32.pt` (npu_dsl's own `fgclip2_e2e/golden.py` output -- the REAL model's own recorded activations, not
a numpy re-derivation of the reference math).

| Measurement | Result | Test |
|---|---|---|
| Layer 0 (real weights) vs the real model's own layer-0 trace, real 972-patch image | `max\|kernel-real\|/max\|real\| = 0.0073` | `test_intel_npu_fgclip2_real_layer0.py` |
| All 27 layers (real weights) vs the real model's own post-layer-27 trace, same image | `max\|kernel-real\|/max\|real\| = 0.0352` | `test_intel_npu_fgclip2_real_27layers.py` (~1000 s) |
| Full image embedding (27 layers + pooling head, real weights) vs the real model's pooled embedding | `max\|kernel-real\|/max\|real\| = 0.0279`, **cosine similarity = 0.9983** | `test_intel_npu_fgclip2_real_image_embedding.py` (~960 s) |
| Full text embedding (27 layers + tail, real weights) vs the real model's pooled embedding, real caption | `max\|kernel-real\|/max\|real\| = 0.0228`, **cosine similarity = 0.9989** | `test_intel_npu_fgclip2_real_text_embedding.py` (~90 s) |
| End-to-end CLIP similarity logit (this backend's own image AND text embeddings) vs the real model's logit for the same real (image, caption) pair | **kernel = 8.2944, real model = 8.1210, abs diff = 0.17** (on a scale where this real pair's true logit is a strongly positive ~8.1 -- a confident correct match, reproduced end to end) | `test_intel_npu_fgclip2_e2e_similarity.py` (~1225 s) |

**What this means**: both towers of the model are *enabled* in the sense that matters -- real pretrained
weights, real input data, numbers that agree with the actual reference implementation to within fp16
rounding error at 27-layer depth. What is **not** covered yet:

* **Device validation.** This sandbox has no `GENERIC_COMPUTE_ROOT` set and no attached NPU hardware
  (`device/runner.py` refuses immediately: `"GENERIC_COMPUTE_ROOT is not set; the device runner never
  defaults SDK paths"`). Every number above is **host-sim** (a CPU model of the runtime surface: DMA = memcpy,
  DPU = fp32-accumulate-rounded-to-fp16 -- it checks addressing/dealing/operand plumbing, not real hardware
  timing or hardware-specific numerics). V5's device golden check and the "prune -> measure -> pin" performance
  loop need real hardware access this environment does not have; nothing here should be read as a performance
  claim.
* **Dense head, `box` walk (boxtext_head)**: not packed/kernelled -- out of scope so far, see section 1.
* **Weight-packing optimisations** (q prescale, k-bias drop, v-bias/LN folding, Wqkv stacking,
  broadcast/ping-pong GEMM scheduling, real flash-attention chunk/tile sizes tuned rather than picked for CMX
  headroom): none ported. The packer (`fgclip2_pack.py`) is deliberately the simple/literal layout this
  backend's kernels already expect (padding only, no folding) -- correct, not fast.
* **Retrieval metrics (R@1/R@5) over the full 32-image x 160-caption test set**: not run. Each real image's
  27-layer pass costs roughly 1000 s of host-sim wall time (not a fast numpy call -- a real C++ host-sim
  compiled and executed per run); a full-set benchmark would be tens of hours of host-sim time for no
  additional correctness information beyond the single-sample checks already done above. Revisit once device
  access exists and host-sim is no longer the only way to run the kernel.

Real bugs found and fixed in this process (all recorded in repo memory for future sessions): `ROWS=8` refused
(gemm N/K must be multiples of 16); `alloc_local` on a single-active-SHAVE kernel silently doubles CMX
reservation; a GEMM operand with a literal leading extent of 1 is refused ("not 2-D") -- the pooling head's
single probe row had to ride along as `BQ` (>=16) redundant identical rows instead.

## 3b. Device validation attempt (2026-11-11 follow-up): infra works, FGCLIP2 itself is BLOCKED by a real-hardware bug

Section 3's "no GENERIC_COMPUTE_ROOT, no NPU hardware" claim above no longer describes every environment this
plan might run in -- a later session on a machine with a real NVL (NPU6) card and local `generic_compute` /
MoviTools SDK checkouts got the device build/runner working (see repo memory
`intel_npu_backend_overview.md`'s "Real NPU device execution" + "OPEN BUG" sections for the full account).
Headline results:

* **Device infra confirmed working**: after three portability fixes (CMake generator, `TL_NPU_DEVICES` for
  NPU6, a guarded `npu40xx` reference for int4 quant that doesn't exist on NPU6), `run_gemm_device.py` ran for
  real on the NVL/NPU6 card: all SHAVEs OK, rel error 3.24e-04, PASS. This is the first non-host-sim execution
  of any TileLang Intel NPU kernel in this backend.
* **FGCLIP2 itself cannot currently be validated on this NPU6 hardware**: every FGCLIP2 kernel tried for real
  (`device/run_fgclip2_v0_device.py` at the smallest V0 shape, `run_fgclip2_ddr_synthetic_device.py` at V1's
  real channel width, `run_fgclip2_layer0_device.py` at the real checkpoint's real shape) hits the identical
  `BARRIER_TIMEOUT` (status 6, detail 13) on real hardware. Bisection with five minimal probe kernels (all kept
  in `device/run_*_probe_device.py`) found the exact cause: **a plain scalar (non-DPU) loop executed on the
  single active SHAVE** -- present in LayerNorm/softmax/GELU/bias-add/residual-add, i.e. almost every op this
  backend's kernels use outside the GEMM itself -- reliably reproduces the hang within 1-3 repeated launches,
  while DPU-matmul-only kernels (any shape, any call count, any sequence of different shapes) were robust
  across 20+ repeated launches with zero failures. This looks like an NPU6-specific runtime/hardware issue in
  this generic_compute revision (`dc/dsl_barrier`/`6bb585c`, which per `device/evidence/INDEX.md` was developed
  and evidenced only against NPU5/PTL hardware), not a TileLang codegen bug -- fixing it needs either a newer,
  NPU6-validated generic_compute revision or real hardware debug tooling (moviDebug2/JTAG trace) beyond source
  inspection.
* **All of section 3's host-sim numbers above remain exactly as valid as before** -- host-sim is a CPU model
  that never exercises real SHAVE hardware, so it was never going to reproduce (or be affected by) this bug.
  Do not re-run the host-sim suite expecting a change; the open item is device validation only.
* If resuming device work: re-run `device/run_scalar_loop_probe_device.py` first as a quick health check (cheap,
  ~30s build) before investing time in a bigger kernel -- if it still fails, FGCLIP2 (and any other
  scalar-op-using kernel) is still blocked on this hardware/SDK combination.
* **Mitigation attempts tried (2026-11-11), all failed** -- do not re-try without new information: (1) a 10x
  larger barrier spin budget just moves the failure to the HOST's own Level Zero event-wait timeout instead
  (proves it is a genuine stall, not a timing budget); (2) giving BOTH SHAVEs symmetric work instead of leaving
  one idle still hangs (rules out "idle worker waits too long for a busy partner"); (3) shrinking the scalar
  loop to a single iteration, or even a single bare scalar store with no loop at all, still hangs (rules out
  loop volume/duration); (4) a dummy DPU matmul inserted right after the scalar store does not clear whatever
  bad state it leaves. The trigger is the mere presence of a plain scalar store to a CMX buffer before
  `ctx.sync_grid()`, independent of size, symmetry, or timeout. See repo memory's "Mitigation attempts tried and
  FAILED" note and `device/run_scalar_loop_symmetric_probe_device.py` for the probe kernels. Root-causing this
  further needs real hardware debug tooling (moviDebug2/JTAG) or SDK-owner engineering support, not more
  source-level reading or black-box device experiments.



* GELU is the **tanh approximation**, not erf -- a numpy `scipy.special.erf`-based GELU reference will disagree
  at the ~1e-3 level, large enough to look like a real bug.
* The text tower's attention is **deliberately unmasked** over all 64 positions including pad tokens (no
  attention mask from the tokenizer) -- do not add causal or padding masking there "to be safe"; it must match
  the reference's actual (slightly unusual) behavior.
* DMA does not convert dtypes (general backend rule, also page 4 of the explainer doc): the position-embedding
  tensor is fp32, patches/weights are fp16 -- any place they are combined needs an explicit `T.cast`, not a copy
  that silently reinterprets bytes.
* The vision tower's row count is **padded** (`n_pad`, a multiple of 128) but the real patch count `n_valid` is
  a runtime value -- every reduction (attention, and anything summing over rows) must bound itself by `n_valid`,
  not by the buffer's padded shape.
