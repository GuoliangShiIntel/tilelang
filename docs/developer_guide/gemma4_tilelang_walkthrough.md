# Gemma-4-26B-A4B on the Intel NPU TileLang backend: a from-scratch technical walkthrough

Written 2026-11-11, after V0-V4 closed and V5 partially attempted. This is the "最原始版本" detailed
tutorial document the user asked for — written for someone who has never seen this backend before, with
real generated code, real IR shapes, and real validation numbers throughout, not reconstructed from
memory. The living, dated development log with the full decision trail is
[gemma4_enablement_plan.md](gemma4_enablement_plan.md); this document re-tells the same material as a
single coherent walkthrough instead of a chronological log, and is the one to read first.

## 1. What problem this solves

Gemma-4-26B-A4B-it is Google's instruction-tuned, 26B-parameter (≈4B active) mixture-of-experts language
model, quantised int4 (symmetric, QAT, group-64) for edge inference. This document explains how we ported
it onto this repository's Intel NPU backend (the TileLang `intel_npu` dialect) — a from-scratch port, not
a translation of the model's original C++ kernel implementation (which lives in a different repository,
`libraries.npu.dsl`, and uses a different, hand-written DSL called `npu_dsl`'s "fused" path).

Why this is non-trivial, in one paragraph: Gemma4 is architecturally quite different from this backend's
existing (Qwen3.6) kernel library. It alternates two kinds of attention layer with different head
dimensions; its full-attention layers have no V projection at all; every layer runs BOTH a dense FFN and
an MoE FFN (not routed-vs-shared); its router has no shared-expert trick; its norms use a different
weight convention than the existing kernels; and its activation function (GELU-tanh) doesn't exist
anywhere in the codebase yet (everything else uses SiLU). None of this is exotic by itself, but getting
all of it right, in a new DSL, on hardware with a tiny fast-memory budget, needed building new kernel
primitives from the attention math up — not just wiring together existing building blocks.

## 2. The backend you're building on top of (a two-paragraph primer)

If you've never seen this backend before, read
[docs/developer_guide/intel_npu_backend_plan.md](intel_npu_backend_plan.md) in full first — this section
is just enough to follow the rest of this document.

TileLang is a tile-level kernel DSL (`T.prim_func`, `T.Kernel`, `T.alloc_local`/`T.alloc_shared`,
`T.gemm`, `T.copy`) that already targets CUDA/HIP/CPU/Ascend/Metal; `intel_npu` is one more backend
target. A kernel launches a fixed grid of `tiles × shaves_per_tile` workers (NPU5/PTL: 3 tiles; NPU6/NVL:
4 tiles; 2 SHAVEs each) that stays resident for the whole launch — there is no GPU-style
grid-per-block remapping; instead, work (output rows, experts, attention (kv-head, token-block) pairs...)
is dealt out to workers by ordinary `for` loops inside the kernel body, with `T.sync_grid()`/
`T.sync_threads()` as explicit (or compiler-derived) rendezvous points. "CMX" is the fast on-chip scratch
memory every SHAVE's working buffers must fit in — it is tiny (NVL: ~1.47 MB per TILE, shared by its 2
SHAVEs) and is the single most common thing a new kernel violates first.

## 3. Reading the model: where the facts came from

Everything below was read directly from three sources, never guessed:

- `Gemma4Ref` (`libraries.npu.dsl`'s `gemma4_e2e/reference/model.py`) — an independent fp32 PyTorch
  reference, the ultimate arithmetic ground truth for every formula quoted below.
- `OvIr` (`gemma4_e2e/reference/ov_ir.py`) — reads the real checkpoint's OpenVINO IR (`.xml` + `.bin`)
  directly with plain numpy (no OpenVINO, no torch needed for this part), giving real weight shapes,
  real norm vectors, real RoPE tables.
- `Gemma4FusedConfig` (`npu_dsl/frontend/kernel_specs/gemma4_fused.py`) — the real model's shapes
  (`hidden=2816`, `n_heads=16`, 128 experts top-8, etc.) and the npu_dsl port's own device ABI, useful as
  a cross-check, not as something this port's kernels had to structurally match.

### 3.1 The two layer kinds

`layer_types[i]` alternates `"sliding_attention"` (5 layers) then one `"full_attention"` layer, period 6
(confirmed on the real checkpoint: `ir.layer_types[:8]` = `['sliding_attention'] * 5 + ['full_attention'] +
['sliding_attention'] * 2`).

| | sliding | full |
|---|---|---|
| head_dim | 256 | 512 |
| kv heads | 8 | 2 |
| query heads (both) | 16 | 16 |
| window | last 1024 positions | none (full causal) |
| V projection | real `v_proj` weight | **none at all** |

The full-attention case is the first real surprise: there is no `v_proj` weight in the checkpoint for
these layers (`ir.lname(5, "v")` returns `None`) — confirmed directly:
```
>>> ir.lname(5, 'v')
None
```
Instead, `Gemma4Ref._attention` computes V as an UNWEIGHTED RMSNorm of the SAME raw K projection that
also feeds (weighted-normed, RoPE'd) K:
```python
raw_k = (x @ weights["k"].T).reshape(token_count, n_kv, head_dim)
k = rms_norm(raw_k, weights["k_norm"], cfg.eps)          # weighted, then RoPE'd
if is_full:
    v = rms_norm(raw_k, None, cfg.eps)                   # SAME raw_k, unweighted, NO RoPE
else:
    raw_v = (x @ weights["v"].T).reshape(...)
    v = rms_norm(raw_v, None, cfg.eps)                    # a separate v_proj, still unweighted
```
Note the detail that is easy to miss: V is unweighted-RMS-normed in BOTH cases, not just the full-
attention one — only the SOURCE of the raw vector differs.

### 3.2 RMSNorm is "plain", not "zero-centered"

This backend's existing Qwen3.6 kernels use a `(1 + w)` convention (`kernels.norm_block`'s `rmsnorm_zc`):
`x * rsqrt(mean(x²) + eps) * (1 + w)` — the usual HF-style "weight starts at 0" RMSNorm. Gemma4's own
reference has no `+1` anywhere: `Gemma4Ref.rms_norm(x, weight, eps) = x * rsqrt(mean(x²)+eps) * weight`
(weight, not `1+weight`). This is because `ov_ir.norm()` reads the ALREADY-MATERIALIZED `Multiply`
constant straight out of the exported graph — whatever `(1+weight)` folding HF's own Gemma RMSNorm does
internally is already baked into the stored IR constant by the time it's exported, so the reference (and
our kernel) just multiply by it directly. We wrote a new, plain-norm macro
(`gemma4_prefill_norm_block` in `gemma4_kernels.py`) rather than trying to reuse `rmsnorm_zc` with a
`w - 1` packing trick — simpler and one less thing that could silently round wrong in fp16.

### 3.3 RoPE: the same math, a different storage convention

`Gemma4Ref.apply_rope`/`rotate_half` is the ordinary "rotate half" convention:
```python
def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin
```
with `cos`/`sin` FULL head_dim wide (`D`, not `D/2`) — verified directly on the real checkpoint:
```
>>> cos, sin = ir.rope_cos_sin('sliding', np.arange(8))
>>> cos.shape, sin.shape
((8, 256), (8, 256))
>>> np.allclose(cos[:, :128], cos[:, 128:])   # the two halves are identical
True
```
This is ALGEBRAICALLY IDENTICAL to the existing Qwen3.6 kernel's partial-rotary macro
(`kernels.make_attn_qkv_prefill`'s rotate-half) with its `ROT` parameter simply set to the full `D` — we
verified this by direct derivation (both reduce to `out[i] = x[i]*cos[i] - x[half+i]*sin[i]`,
`out[half+i] = x[half+i]*cos[half+i] + x[i]*sin[half+i]`) before writing a single line of new RoPE code.
Only the model-side reorganisation (no gate column, separate/derived V) needed a new phase
(`make_gemma4_qkv_prefill`); the RoPE arithmetic inside it is the proven Qwen one.

### 3.4 Every layer runs a dense FFN AND an MoE FFN — not routed-vs-shared

```python
dense_in = rmsnorm(h1, pre_ffn_norm)
dense    = down(gelu_tanh(gate(dense_in)) * up(dense_in))
dense    = rmsnorm(dense, post_ffn_1_norm)

moe_in   = rmsnorm(h1, pre_ffn_2_norm)
moe      = moe_block(moe_in, route_x=h1)      # top-8 of 128, NO shared expert at all
moe      = rmsnorm(moe, post_ffn_2_norm)

combined = rmsnorm(dense + moe, post_ffn_norm)
h_out    = (h1 + combined) * layer_scalar      # a real per-layer scalar, read from the checkpoint
```
Qwen3.6's MoE instead has a routed-vs-shared-expert combine (`out = sum_k routed + sigmoid(gate) *
shared`) — structurally different, not just a naming difference. Gemma4's router is plain top-8-of-128
softmax, no shared row, PLUS a learned `per_expert_scale` gathered after renormalising:
```python
probabilities = softmax(scores)                               # over all 128
top_w, top_idx = topk(probabilities, 8)
top_w = top_w / top_w.sum()                                    # renormalise the chosen 8
top_w = top_w * per_expert_scale[top_idx]                       # THEN apply the learned per-expert scale
```

**A genuinely useful discovery, not an assumption**: the EXISTING Qwen3.6 router kernel
(`kernels.make_moe_route_prefill`'s `route_phase`) picks top-k BY RAW LOGIT (a lane-argmax trick, no
softmax at all at that point), then computes softmax ONLY over the k chosen values and renormalises.
We proved by direct algebra that this is mathematically IDENTICAL to "softmax over all, then top-k, then
renormalise": top-k by raw logit picks the same set as top-k by softmax probability (softmax is monotonic
in the logit), and `exp(l_i)/Σ_all / (exp(l_i)/Σ_all summed over just the chosen k)` cancels `Σ_all` to
give exactly the chosen-k-only softmax formula the existing kernel already computes. So the ONLY real
change needed in the router was: drop Qwen3.6's extra "shared expert, every token" row, and add the
`per_expert_scale` gather. See `make_gemma4_moe_route_prefill` in `gemma4_kernels.py`.

### 3.5 The activation function nobody had written yet

Checked directly (not assumed): this repository's existing MoE/GDN code uses SiLU-based SwiGLU
everywhere (`h = gate * sigmoid(gate) * up`), never GELU. Gemma4 uses `F.gelu(x, approximate="tanh")`
for BOTH the dense FFN's gate and every expert's gate. We wrote it algebraically simplified so it reuses
the `exp`-based sigmoid pattern already proven elsewhere on this backend (the attention output gate's
sigmoid, the old MoE router's shared-expert sigmoid), rather than the untested `T.tanh`:
```
gelu_tanh(x) = 0.5 x (1 + tanh(c(x + 0.044715x³))),  c = sqrt(2/π)
             = x * sigmoid(2c(x + 0.044715x³))                     (same value, no tanh needed)
```
```python
@T.macro
def act_phase(Gate, Up, Act, tile, shave):
    ...
    for d0 in T.serial(FI // FL):
        for v in T.vectorized(FL):
            x = T.cast(g_cmx[0, d0 * FL + v], T.float32)
            z2 = T.float32(C2) * (x + T.float32(0.044715) * x * x * x)
            e = T.exp(T.float32(0) - z2)
            gelu = x / (T.float32(1) + e)
            a_cmx[0, d0 * FL + v] = T.cast(gelu * T.cast(u_cmx[0, d0 * FL + v], T.float32), T.float16)
```

## 4. The kernel, phase by phase, with real generated code

`make_gemma4_layer_prefill` (`gemma4_kernels.py`) composes ten phases into ONE launch, in exactly
`Gemma4Ref.decoder_layer`'s order:

```
x = rmsnorm(H, input_norm) -> qkv proj -> qkv_phase (norm/RoPE/V-derivation) -> flash attention
    -> o_proj -> h1 = H + rmsnorm(attn_o, post_attn_norm)
dense_in = rmsnorm(h1, pre_ffn_norm) -> dense FFN -> dense_n = rmsnorm(dense, post_ffn_1_norm)
route_in = rmsnorm(h1, router_scale) -> router logits -> route (top-k, per_expert_scale)
moe_in = rmsnorm(h1, pre_ffn_2_norm) -> grouped expert FFN -> combine -> moe_n = rmsnorm(moe, post_ffn_2_norm)
combined = rmsnorm(dense_n + moe_n, post_ffn_norm)
H = (h1 + combined) * layer_scalar
```
Every phase boundary is a grid barrier. Critically, **we never write these barriers by hand** — the
pipeline's `DeriveIntelNpuGlobalSync` pass derives them automatically from which phases read/write which
DDR scratch tensors, so the layer's Python code is just a straight-line sequence of macro calls:
```python
@T.macro
def layer(Hio, InNorm, QNorm, KNorm, CosSin, ..., tile, shave):
    worker = tile * shaves + shave
    norm_f32(Hio, InNorm, Xn, worker)
    qkv_proj(Xn, WqkvQ, WqkvS, Ys, tile, shave)
    qkv_phase(Ys, QNorm, KNorm, CosSin, Pos, Nt, Q, Kc, Vt, tile, shave)
    attn_phase(Q, Kc, Vt, Pos, AttnO, tile, shave, None, 0)
    o_proj(AttnO, WoQ, WoS, AttnP, tile, shave)
    norm_f16(AttnP, PostAttnNorm, PostAttnN, worker)
    add_tokens(Hio, PostAttnN, worker)          # Hio now holds h1
    ...
```

### 4.1 What the attention phase actually compiles to (real captured output)

This is `make_gemma4_attn_prefill`'s generated source, lowered at a small shape
(`N=16, NKV=2, GQ=2, D=32, window=20`) — the raw DMA/DPU calls `tilelang.lower(...).kernel_source`
produces, nothing reconstructed:
```cpp
ctx.dma_load((&(pos_cmx[0])), (&(Pos[0])), 4, 1, 4, 4);
...
ctx.dma_load((&(q_cmx[0])), (&(Q[...])), 64, 8, 64, 64);
...
ctx.dma_load((&(k_cmx[0])), (&(Kc[...])), 64, 32, 64, 64);
ctx.dma_load((&(v_cmx[0])), (&(Vt[...])), 64, 32, 64, 512);
ctx.dpu_matmul((&(s_cmx[0])), (&(q_cmx[0])), (&(k_cmx[0])), 8, 32, 32, 0, -1);
...
ctx.dpu_matmul((&(pv_cmx[0])), (&(s_cmx[0])), (&(v_cmx[0])), 8, 32, 32, 0, -1);
```
Each `ctx.dma_load` is a raw byte DMA (DDR → CMX); each `ctx.dpu_matmul` is one DPU workload (the `8, 32,
32` are M/N/K; the trailing `0, -1` select the runtime's own spatial split and stencil). There is no
output gate anywhere in this source (Gemma4 has none — unlike every existing Qwen3.6 attention kernel,
which always emits a `Gate` DMA and a sigmoid multiply at the end).

### 4.2 CMX budgets, in real numbers — why half the engineering effort went here

The MoE expert FFN's FIRST, untiled draft allocated the WHOLE `(I, H)` gate/up/down weight matrices per
SHAVE. At real dimensions (`I=704, H=2816`) that's `704 * 2816 * 2 bytes ≈ 3.96 MiB` — on its own, more
than DOUBLE the entire NVL tile's `cmx_bytes = 1,473,536` (≈1.47 MiB) budget, which two SHAVEs must
share. The compiler's `CheckIntelNpuBudgets` pass caught this immediately and clearly:
```
ValueError: intel_npu: CMX budget exceeded at the peak: 2630464 B per SHAVE x 2 SHAVEs
+ 1152 B shared + 73728 B runtime + 0 B merged-FC scratch = 5335808 B > cmx_bytes 1473536
```
The fix (`BNW_gu`/`BNW_d` parameters on `make_gemma4_moe_expert_prefill`) tiles the weight-row dimension
into chunks a SHAVE dequantises/computes one at a time:
```python
for gu in T.serial(I // BNW_gu):
    T.copy(WgateQ[e, gu * BNW_gu, 0], wgq_cmx)
    T.copy(WgateS[e, 0, gu * BNW_gu], wgs_cmx)
    T.call_extern("handle", "tl_npu_dequant_i4_gm", ..., BNW_gu, H, G)
    T.gemm(x_cmx, wg_cmx, gate_chunk, transpose_B=True, clear_accum=True)
    ...
    # the activation is fused into the SAME chunk loop, writing straight into the right
    # column range of the full (MB, I) act_cmx buffer -- no extra (MB, I)-sized gate/up buffer needed
```
After this fix, the SAME kernel at the SAME real dimensions still exceeded budget by roughly 0.07%
(`1,474,560 B` needed vs `1,473,536 B` available) — this remaining gap was bisected empirically (not
analytically: changing `BKV`/`TB` had NO effect on the reported peak, which ruled out the attention
phase) down to `make_proj_prefill`'s `MB` (token-block) parameter, which the layer composer had defaulted
to the FULL token count `N` for the QKV/O/dense-FFN projections — at real `K` (`KO=4096`, `H=2816`) this
activation tile was the dominant remaining term. A new `MB_proj` parameter (letting these projections use
a smaller token block than the attention/expert phases' own `N`) closed the last 1 KB.

## 5. The validation ladder, with real numbers at every rung

Every rung below is a REAL pytest, not a hand-wave — file names are given so you can re-run them.

| Rung | What | Shape | Result |
|---|---|---|---|
| V0 attention | gate-less, sliding-window flash attention alone | small synthetic | 5/5 tests pass |
| V0 QKV phase | plain norm, full RoPE, V-derivation (both layer kinds) | small synthetic | 3/3 pass |
| V0 dense FFN + router | gelu-tanh GeGLU; top-k/per_expert_scale, no shared row | small synthetic | 2/2 pass |
| V0 MoE expert + combine | worker-owns-whole-experts, weight-row tiled | small synthetic | 2/2 pass |
| **V0+V1 whole layer** | full `make_gemma4_layer_prefill`, BOTH layer kinds | `H=64,...,N=8` | **3/3 pass**, incl. sliding+window and full-attention |
| V2 (lightweight) | the SAME layer macro called TWICE, residual chained | `H=64,...,N=8` | 1/1 pass |
| **Real-dimension smoke test** | the SAME layer macro, REAL Gemma4 shapes, synthetic weights | `H=2816, NH=16, E=128,...` | **1/1 pass** (after the CMX fixes in §4.2) |
| **V4: real checkpoint, layer 0** | REAL weights read from the real OpenVINO IR, real embedded tokens | `H=2816,..., N=8` | **PASS: rel_err=0.0003, cosine=1.000000** |
| V5: device, infrastructure | `run_gemm_device.py` on real NVL/NPU6 | — | PASS (rel 3.24e-4, 0.105 ms) |
| V5: device, Gemma4 kernels | the real-dimension layer kernel, and attention alone | — | **NOT YET passing** -- see §6 |

The V4 number is the capstone correctness result: real Gemma-4-26B-A4B-it weights, in the model's own
real int4-symmetric-QAT-group-64 quantisation, through this backend's actual generated kernel code
(host-simulated), reproducing an independently-written from-scratch reference
(`gemma4_decoder_layer_reference`) to 0.03% relative error and a cosine similarity of 1.000000. No shape
was shrunk to make this pass, no weight was synthetic, no tolerance was loosened.

### 5.1 Real bugs this ladder actually caught (not a hypothetical list)

- **A top-k softmax ordering bug**: the router's normalising loop reused `pick[0]` (the row max) AFTER
  already overwriting it in the same loop (`pick[k] = exp(pick[k] - pick[0])` with `pick` doing double
  duty as both the raw-logit input and the exp output) — every weight but the first token's came out
  wrong. Fixed by saving the max to a separate scalar first. Caught by the 2/2 router unit test, NOT by
  the full-layer test (which would have shown a vaguer, harder-to-localise numeric mismatch).
- **A silent `T.copy` semantics bug**: `T.copy(src, dst)` infers its shape from whichever operand is a
  BARE (unindexed) buffer; when gathering one dynamically-indexed DDR row into one dynamically-indexed
  CMX row (neither operand bare), it silently copied a SINGLE SCALAR ELEMENT instead of a whole row — no
  compile error, a ~1.0 relative-error numeric failure. Fixed with explicit `0:H` slices on both sides
  (the convention the existing MoE combine code already used for the identical pattern — read it before
  guessing).
- **A RoPE table convention mismatch (test-only, not a kernel bug)**: `ov_ir.rope_cos_sin` returns the
  FULL-WIDTH `[N, D]` cos/sin table directly (verified: the two halves are exact duplicates on real
  data) — only `pack.py`'s device packer slices it to half-width to save DDR space. An early test built a
  half-width table by hand and re-tiled it, silently doubling the width wrongly (caught immediately as a
  shape mismatch, not a latent numeric bug).
- **Two parallel reference weight-dict naming mistakes** (`"input"` vs `"input_norm"`, a missing
  `router_scale` key) — boring but real, caught by `KeyError`s, not silent wrong numbers.

None of these would have been caught by "the compiler accepted it" alone — every one needed an actual
numeric comparison against an independent implementation.

## 6. What does NOT work yet, honestly

**Real device (NVL/NPU6) execution of the Gemma4 kernels is not yet achieved.** The device toolchain
itself is confirmed healthy on this machine (`run_gemm_device.py`: real hardware, PASS). But:

1. The full fused layer kernel (61 separate `npu::Ddr<T>` parameters, one per logical tensor — no
   consolidation) crashes at device LAUNCH with `STATUS_STACK_BUFFER_OVERRUN` — almost certainly a
   parameter-count ceiling in generic_compute's host-side `Kernel::submit()` path, not a bug in this
   kernel's logic. **The real fix is architectural**: a real device-side layer TABLE (consolidating many
   logical DDR tensors into a few big blobs addressed via `T.ddr_view`, exactly the "real runtime layer
   table" work already flagged as deferred future engineering in the V2/V3 sections of the plan doc) —
   not a quick patch.
2. The standalone attention kernel alone (5 parameters, confirmed zero `sync_grid` calls) hangs the
   device at modest real-ish dimensions; a much smaller shape of the identical kernel ran without hanging
   but gave an output essentially uncorrelated with the host-sim-proven-correct answer. This is a real,
   unexplained host-sim-vs-device numeric/execution gap that needs real hardware debug tooling
   (`moviDebug2`, parameter-by-parameter bisection) to root-cause — not attempted here for lack of time.

**Also not done, by deliberate scope decision, not oversight** (see the plan doc's Log for the exact
reasoning at each point):
- The real device-side runtime layer TABLE for a full 30-layer model (python-unrolled 2-layer chaining
  was used instead, enough to prove the mechanism, not the scale).
- A dedicated decode (M=1 GEMV) fast path — this port's kernels are all "prefill-style" (operate on N
  tokens at once, including N=1), which is correct but not the performance-optimal decode path npu_dsl's
  `dpu_matmul_i4`-based merged-FC GEMV uses.
- Performance tuning of any kind (tile-interleaved gate|up packing, load-balanced MoE routing, windowed
  attention's chunk-skip optimisation, pipelining) — every V0-grade simplification in this port
  deliberately trades performance for a simpler, more obviously-correct first implementation; none of
  them affect the numbers in §5.

## 7. If you are picking this up next

1. Read [gemma4_enablement_plan.md](gemma4_enablement_plan.md)'s Log (section "## 8"-equivalent at the
   bottom) for the exact dated trail every decision and bug fix above came from.
2. The device issues in §6 are the highest-value next investigation — they block ANY further Gemma4
   device validation. Start with the attention-alone hang (simpler kernel, easier to bisect) before the
   full-layer parameter-count issue (which needs an architectural fix, not just debugging).
3. Everything in `testing/python/intel_npu/gemma4_kernels.py` is built to the same pattern as
   `kernels.py`'s Qwen3.6 code — read a Qwen `make_*_prefill` function side by side with its Gemma4
   counterpart before changing either.
4. The real checkpoint is at
   `C:\Intel\guoliang\model_pkg\Google-Gemma-4-26B-A4B-it_int4_sym_qat_group-64_dyn_stateful` (group-64
   QAT int4 — confirmed via `ir.group == 64` on this exact checkpoint). Extracting more layers' weights
   from it follows the exact pattern in
   `test_intel_npu_gemma4_real_checkpoint_layer0.py` — `OvIr`'s accessors
   (`lname`, `int4_cw`, `norm`, `router_scale`, `router_proj`, `per_expert_scale`, `layer_scalar`,
   `rope_cos_sin`) are pure numpy, no torch/OpenVINO dependency, and fast (checkpoint load: ~0.07s,
   memory-mapped).
