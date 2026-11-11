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

## 3. Known traps carried over from the npu_dsl implementation (do not re-discover these the hard way)

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
