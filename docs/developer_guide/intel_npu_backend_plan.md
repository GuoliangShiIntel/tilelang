# Intel NPU (NPU5 / Panther Lake) backend: implementation plan

Status: plan, 2026-11-06. Branch `feat/intel-npu-backend`. Living document: every phase below records
its outcome in place when it closes (date, commit, evidence path). Decisions that change get a dated
amendment, not a silent edit.

## 0. Decision

Build a **pure TileLang target backend** for the Intel NPU5 (PTL), owned entirely by this repo:
dialect, target, pass pipeline and C++ codegen, plus a small SHAVE runtime header of our own.
Nothing imports or links the previous `npu_dsl` compiler (`D:\dsl\libraries.npu.dsl`). Its design
documents are read as prior art only; its runtime headers (`fused_rt*.hpp`) are not reused.

The only external dependency is the hardware SDK: the `generic_compute` toolkit (SHAVE `kernel_entry`
ABI, `npu::Environment`, DMA/DPU drivers, host `npu_runtime` over Level Zero) and MoviTools
(`moviCompile`, `moviLLD`). Their roots come from `GENERIC_COMPUTE_ROOT` and `MV_TOOLS_PATH`; they
are never hardcoded (CMake `SEND_ERROR` if undefined).

**Pinned SDK state on this machine (2026-11-06).** `GENERIC_COMPUTE_ROOT=D:\dsl\gc_dsl_barrier`, a
standalone local git repository (`main` @ `daf6c05`, "Snapshot of generic_compute build root"). It
was a worktree of a clone that no longer exists, so it was re-initialised as its own repo with the two
former submodules vendored as plain files. Its commit message records the provenance claimed by
`libraries.npu.dsl/process/generic_compute.pin` (upstream `6bb585c`, branch `dc/dsl_barrier`); that
claim is **not verified** against upstream. Every device result in this plan cites the
`gc_dsl_barrier` commit it was built against, and any edit to that tree is its own commit there.
`MV_TOOLS_PATH=C:\Intel\xiong\applications.ai.vpu-accelerators.vpux-plugin\bin\MoviTools\26.08.2-internal`.

Why pure rather than a TIR-to-npu_dsl bridge: one IR (TIR), one pass framework, TileLang's
verifiers/arith/layout/autotune for free, and no second schema to keep in sync. The cost is accepted:
the NPU5-specific derivations (split vs broadcast, dual-port DMA, barrier derivation, CMX budget,
`.text` budget) are rewritten as TIR passes.

Prior PoC (2026-10-29): `tilelang/ptl/codegen.py` emitted npu_dsl-runtime C++ straight from the
frontend TIR, with no `BackendModule`. It was never committed; it and the two adapter edits it needed
(`jit/adapter/libgen.py`, `jit/adapter/wrapper.py`) were removed from the working tree when this
branch was created. The backend starts from zero.

### 0.1 Why migrate: npu_dsl versus TileLang

npu_dsl works. Three models run on it as single-launch fused kernels, with device goldens and
measured numbers (`docs/RUN_AND_RESULTS.md` there). The reasons to move are about what it costs to
keep going, not about what it has failed to do.

| Dimension | npu_dsl (`D:\dsl\libraries.npu.dsl`) | TileLang `intel_npu` backend | Why it matters |
|---|---|---|---|
| Frontend | `nl.*`, a home-grown Python runtime tracer (body runs once; `if` folds at trace time) | `@T.prim_func` / `T.macro`, TVMScript AST parsing into TIR | A documented, externally known DSL; GPU authors can read an NPU kernel; `T.macro` libraries have explicit signatures and are unit-testable |
| IR | Python dataclasses (`Loop/Dma/Dpu/Shave/Barrier/TileSync`, `npu_dsl/fused/ir.py`); ad-hoc annotations (P2 debt) | TVM TIR with typed nodes, `arith::Analyzer`, `StmtExprMutator`, span tracking | Simplification, bounds proofs and symbolic shapes come for free; passes are written against a stable, typed IR |
| Pass framework | 9 passes in `passes/pipeline.py`; `check_order()` only; **no per-pass verifier** (P1 debt, open since the fused path began) | `PassPipeline` over `tvm.transform`; per-pass instrumentation (`tilelang/instrumentation`), `VerifyBufferInit`, `VerifyParallelLoop`, race checks, lower-trace tooling | Debt P1 and P2 close by construction instead of by a future project |
| Scheduling | `autotile` with a six-coefficient linear time model that **mis-ranks** (S3 debt); Gemma-4 pins its tiles by hand after a device sweep | `tilelang.autotune` (device-measured search), `T.Pipelined`, `T.Persistent`, layout inference, `T.use_swizzle` | The sweep tooling exists and is maintained upstream; the cost model is not ours to calibrate alone |
| Megakernel | native: one `kernel_entry` per model, `derive_barriers` from DDR dataflow | native: `T.Persistent` + `T.sync_grid()` (MLA decode example); barrier derivation to be added (Phase 2b) | Parity on the capability that makes the NPU path fast; npu_dsl's one unique strength is reproducible |
| Heterogeneous cores | DPU + SHAVE, modelled directly | Ascend backend precedent (`T.Cube()`/`T.Vector()`, `MixedKernel(bx, sid)`), ~2.5k Py + 8k C++ | The architecture has already carried an NPU with two core kinds and explicit scratchpads |
| Codegen | `emit.py` string assembly (Python) against `fused_rt.hpp` | `CodeGenTileLangCBase` subclass (C++), `#line` directives, source-only module | Same C++-for-moviCompile output; the emitter is a visitor over typed IR rather than string templates |
| Host sim / CI | `npu_dsl/fused/sim` builds the generated C++ with host clang | `npu_rt_host.hpp` over the same generated source, plus the TileLang CPU backend as a second reference | Equivalent; CI stays hardware-free |
| Generality bar (G2) | enforced by review and by the "renamed-loops" test pattern | the same bar, but TIR passes select by IR properties by nature (no names reach a pass) | G1/G2 become easier to keep than to break |
| New-model harness | 15-33 files per model, copied by renaming (the "new model (harness)" debt) | rebuilt once, model-agnostic, under `testing/python/intel_npu/device/` | A known debt is paid as part of the move instead of carried |
| Dependencies | none beyond the SDK; pure Python | TVM C++ build (~15 min cold on this box), Ninja + clang-cl | **Cost.** Mitigated by incremental builds and by keeping SDK-dependent code out of the native library |
| Ecosystem | one repo, four lanes, ~38k-line round log | upstream TileLang: other backends, tests, reviews, future features (e.g. a Triton/Inductor front door would attach to TIR, not to `nl`) | Work we do not have to do; a path to users outside this team |

What is given up, stated plainly:

* **Sunk work.** ~13k lines of Python (`fused/`, `lib/`, `frontend/`, `registry.py`) and the three
  model sources are not ported; they are read as prior art. Their *rules* (`fused_compiler_rules.md`
  G/Q/T/S/O/P/N/V) and their *pitfalls* (the `enable-new-model` skill's list) are carried over as
  tests in Phases 1-5.
* **The V1 identity gate** across the migration. Byte-identical output against npu_dsl is not a goal;
  device goldens and unprofiled A/Bs against npu_dsl's recorded numbers are the acceptance (Phase 5).
* **Effort before parity.** Until Phase 5 closes, enabling a new model costs what npu_dsl's *first*
  model cost, not its fourth. The `enable-new-model` procedure itself (model card, reference, golden,
  sim, device, first numbers, offline sweep) is DSL-independent and is reused as written.
* **Build weight.** A C++ toolchain in the loop; see the Windows notes in section 1.

The decision is therefore: pay a bounded rebuild (Phases 0-5) to replace a bespoke compiler that has
reached the point where its next items (verifiers, typed annotations, a trustworthy cost model, a
model-agnostic harness) are exactly the things a mature TIR-based framework already provides.

## 1. Preflight decisions (integration-checklist, "Preflight")

| Item | Decision |
|---|---|
| Backend / package name | `intel_npu` (`tilelang/intel_npu/`, `src/intel_npu/`) |
| TVM target kind | `intel_npu`, registered in `src/intel_npu/target.cc` (`TVM_REGISTER_TARGET_KIND("intel_npu", kDLExtDev)`, as Ascend does); TVM does not know it |
| Target attributes | `arch` (`npu5`; alias `ptl`), `num_tiles` (PTL: 3), `shaves_per_tile` (2), `cmx_bytes` (per tile; PTL 1572864), `dma_ports` (2), `text_limit_bytes` (262272, measured on Gemma-4 decode), `supports_kernel_status_return` (false). Every platform constant lives here and nowhere else. **Barrier ids are not an attribute yet**: two prior sources disagree (12..19 vs a pool of 8..47 read from a platform config) and the question is open in section 7. NPU60XX is reserved, not modelled: no value is entered until the hardware owner provides it |
| Auto-detection | none. Explicit `target="intel_npu"` (or `"ptl"` alias via the normalizer). The device is driven by an external runner, not by TileLang's process |
| Launch / thread model | `T.Kernel(num_tiles, threads=shaves_per_tile)`: `blockIdx.x` = tile (cluster), `threadIdx.x` = SHAVE in the tile. The grid is **persistent**: it equals the physical tile count, and work is dealt by loops (`T.Persistent` / `T.serial(waves)`). One `prim_func` = one `kernel_entry` = one launch, holding any number of phases |
| Memory scopes | `global` = DDR; `shared` = tile-shared CMX; `local` = SHAVE-private CMX. No fragments / registers as a tile-op operand: the DPU and SHAVE ops consume CMX |
| Sync | `T.sync_grid()` = all SHAVEs of all tiles (global barrier; Phase 2); `tvm_storage_sync("shared")` = the SHAVEs of one tile. Hardware barriers (ids 12..19) are an implementation of the pipelined fill/free protocol in Phase 3, not a dialect symbol |
| Initial tile ops | `T.copy` (DDR<->CMX, CMX<->CMX), `T.gemm` (DPU, fp16, `transpose_B=True`, whole K per workload), `T.fill`/`T.clear`, `T.Pipelined` (num_stages 2 = ping-pong), `T.sync_grid`, `T.call_extern` for SHAVE ops (Phase 4 adds a typed op surface) |
| Explicit exclusions (refused by name at lowering) | `T.Parallel` element loops on fragments, `T.reduce_*` on fragments, atomics, TMA/mbarrier/warp-level builtins, `gemm` with `transpose_A`, K-split gemm with a fragment accumulator (the DPU takes whole K; a K split is a SHAVE accumulate and is Phase 4), 3-D grids, `cluster_dims` |
| Device output format | C++ source for `moviCompile` (one `kernel_entry` per `prim_func`) plus a JSON sidecar: parameter list (index, shape, dtype, direction), CMX footprint, tile/SHAVE counts, `.text` estimate |
| Host targets | `STANDARD_HOST_CODEGENS` is **not** used in Phases 0-4: the host is the external runner. Host codegen is revisited with the execution backend (Phase 5) |
| Execution backends | none of `tvm_ffi`/`nvrtc`/`cython`/`torch` can launch on the NPU. Phases 0-4 declare a single `source_only` mode: `build_without_compile` emits the source and the sidecar; a runner under `testing/python/intel_npu/` compiles with moviCompile and launches through the `generic_compute` host app. A real `intel_npu` execution backend (ctypes over the `npu_runtime` host library) is Phase 5 and only if the runner proves insufficient |
| Eager/deferred compile | deferred; the runner compiles |
| Hardware-free coverage | import, target resolution, dialect construction, pipeline lowering with refusal diagnostics, source generation with asserted text, and the **host simulation**: the generated C++ also compiles with the host `clang++` against `npu_rt_host.hpp` (a CPU implementation of the same `npu_rt.hpp` API) and is checked against numpy. This is the CI path |
| Hardware-required coverage | device golden vs numpy/PyTorch, device trace admissibility, unprofiled timing. Pre-authorised on the developer's PTL box; every lane may run it |
| Windows | first-class (the device box is Windows). `PYTHONIOENCODING=utf-8`; `subst` short drives for `MAX_PATH`; no `/tmp` |

## 2. Layout

```
tilelang/intel_npu/
  __init__.py           target -> transform -> language -> op -> backend (cycle-safe order)
  target.py             normalizer: "ptl"/"npu5" aliases -> intel_npu -arch=npu5 + canonical attrs
  backend.py            register_backend(BackendModule(name="intel_npu", target_kinds=("intel_npu",), ...))
  pipeline.py           the complete ordered pass list (PassPipeline("intel_npu", ...))
  codegen.py            thin wrappers: target.build.tilelang_intel_npu[_without_compile]
  execution_backend.py  ExecutionBackendSpec("source_only", ...)
  toolchain.py          moviCompile/moviLLD/host clang++ lookup from MV_TOOLS_PATH; never a literal path
  language/__init__.py  common surface + extensions (__tilelang_dialect__ = "intel_npu")
  language/kernel.py    T.Kernel override: (tiles, threads=shaves), persistent grid, refuses 2-D/3-D grids
  language/dma.py       copy attrs: port=0|1|"both", multicast
  language/shave.py     typed SHAVE op surface over call_extern (Phase 4)
  transform/__init__.py Python wrappers of src/intel_npu/transform/*
  op/                   Python tile-op selectors (gemm -> DPU workload, copy -> DMA plan)
src/intel_npu/
  target.cc             target kind + attributes
  op/                   gemm.cc, copy.cc, builtin.cc (DMA / DPU / barrier builtins)
  transform/            lower_kernel_launch.cc, plan_cmx.cc, derive_global_sync.cc, plan_distribution.cc,
                        plan_dual_port.cc, lower_pipelined.cc, insert_sync.cc, check_budgets.cc
  codegen/              codegen_intel_npu.{h,cc} (CodeGenTileLangCBase subclass), rt_mod_intel_npu.cc
  CMakeLists.txt        source selection; always compiled (text-only, no SDK); a USE_INTEL_NPU gate appears
                        only with SDK-dependent runtime sources
src/tl_templates/intel_npu/
  npu_rt.hpp            dispatcher: device or host implementation of one surface (npu::Ctx, npu::Ddr<T>)
  npu_rt_device.hpp     over generic_compute: dma_load/store (2-D), dpu_matmul (fp16, whole K), status record;
                        Phase 3 adds ports, multicast and barriers
  npu_rt_host.hpp       CPU implementation of the same API for the host simulation (threads, memcpy, fp32 acc)
testing/python/intel_npu/
  test_intel_npu_phase0.py, test_intel_npu_phase1_gemm.py, then test_pipeline_refusals.py, test_host_sim_*.py
  kernels.py            the kernels under test (make_gemm, make_gemm_ksplit)
  host_sim.py           clang++ build of generated source against npu_rt_host.hpp + ctypes runner
  show_phase0_source.py / show_gemm_source.py   print emitted C++
  device/               CMakeLists.txt + host/npu_run.cpp (runner over npu_runtime), runner.py,
                        run_gemm_device.py, evidence/INDEX.md; build/, generated/, run_*/ are ignored
docs/developer_guide/intel_npu_backend_plan.md   this file
```

Registration points touched: `tilelang/__init__.py` (import `intel_npu` next to `cpu`/`ascend`),
root `CMakeLists.txt` delegation block (`src/intel_npu/CMakeLists.txt`). `cmake/BackendOptions.cmake`
(`TILELANG_BACKENDS`, `USE_INTEL_NPU`) only when a gated source appears; `pyproject.toml` only if a
resource is added.

## 3. Pass pipeline (target order; Phase column says when each lands)

| # | Pass | Purpose | Phase |
|---|---|---|---|
| 1 | `BindTarget` | bind `intel_npu` target | 0 |
| 2 | `LowerNpuKernelLaunch` | `blockIdx.x`->tile id, `threadIdx.x`->SHAVE id; refuse >1-D grid, `cluster_dims`; persistent grid attr | 0 |
| 3 | `MaterializeKernelLaunch(lower_thread_binding=False)` | shared | 0 |
| 4 | shared normalisation: `LetInline`, `LegalizeNegativeIndex`, `InjectAssumes`, `Simplify`, `VerifyBufferInit` | shared | 0 |
| 5 | `LayoutInference` + `NpuLayoutRules` | CMX buffers are row-major contiguous; DPU operands need no swizzle; refuse fragments | 1 |
| 6 | `NpuLowerTileOp` | `copy`->DMA builtin (direction, rows, width, stride, port); `gemm`->DPU builtin; `fill` on a DPU output dropped | 1 |
| 7 | `PlanDistribution` | per `copy` from DDR: source independent of the tile var => broadcast (one multicast fill, one issuing SHAVE), else split; refuse a per-tile unicast copy of a tile-invariant source | 3 |
| 8 | `PlanDualPort` | split each DDR fill in half, one half per port, issued together | 3 |
| 9 | `LowerPipelined` | `T.Pipelined(num_stages=2)` over a streamed loop -> ping/pong slots, SHAVE 0 = ping, SHAVE 1 = pong, pre-enqueued next fill gated on FREE, READY on completion | 3 |
| 10 | `DeriveGlobalSync` | insert `sync_grid` from global-buffer RAW/WAR across phases when the author did not; reuse the access-set machinery of `thread_storage_sync.cc` at `global` scope | 2b |
| 11 | `InsertSync` | tile-level sync for shared CMX; map `sync_grid` to a software rendezvous, hardware barrier ids for the pipelined protocol | 2 |
| 12 | `PlanCmx` | static CMX allocation per scope with live-range packing against `cmx_bytes`; refuse overflow naming the buffer | 1 (bump) / 3 (packing) |
| 13 | shared legalisation: `DecoupleTypeCast`, `LegalizeSafeMemoryAccess`, `LowerOpaqueBlock`, `FlattenBuffer`, `NarrowDataType(32)`, `Simplify`, `RemoveNoOp` | shared | 1 |
| 14 | `CheckBudgets` | `.text` estimate vs `text_limit_bytes`; CMX; barrier id count; emit the sidecar numbers | 2 |
| 15 | `SplitHostDevice`, `AnnotateDeviceRegions`, `LowerDeviceKernelLaunch` | shared; host side is a stub in source-only mode | 0 |

Every pass ships with a negative control (a fixture it must refuse or leave unchanged, shown to
fail when the check is removed). A pass is not done when it is only observed passing.

## 4. Phases

Each phase has an executable acceptance. A phase closes only when its acceptance **has been seen to
fail** on a negative fixture and then pass on the real one.

### Phase 0: scaffold (hardware-free)
* Package, target kind, manifest, dialect (common only), pipeline items 1-4 and 15, source-only
  codegen that emits a `kernel_entry` skeleton for a kernel with no tile ops.
* Sources are text-only and always compiled; no CMake gate (amended 2026-11-06, see section 2).
* Acceptance: fresh-process `import tilelang.intel_npu`; `Target("intel_npu -arch=npu5")` and alias
  `"ptl"` resolve; `tilelang.lower(kernel, target="intel_npu")` returns source containing
  `kernel_entry`; a 2-D grid is refused with a message naming `intel_npu`. Negative: a target string
  of another backend returns `None` from the normalizer; the same source assertions fail on the
  `c` backend's output; a tile op is refused by name.
* Launch binding (decided here): `MaterializeKernelLaunch(lower_grid_binding=True,
  lower_thread_binding=True)`, so `blockIdx.x` / `threadIdx.x` reach codegen as `thread_extent`
  scopes and are printed as `npu::tile_id()` / `npu::shave_id()`. `default_threads` is the target's
  `shaves_per_tile`.

### Phase 1: single-phase GEMM, one tile, sim + device
* Pipeline items 5, 6, 12 (bump allocation), 13. `npu_rt.hpp` + `npu_rt_host.hpp` with
  `dma_load/store`, `dpu_matmul`, worker dealing.
* Kernel: `[M,K] x [N,K]^T` fp16, grid persistent over `(M/BM)*(N/BN)` blocks.
* Acceptance: host sim matches numpy (`rel < 2e-3`); device golden matches; the K-split form is
  refused by name. Negative: perturb one byte of `B` in the sim harness -> FAIL reported.

### Phase 2: megakernel primitives
* `T.sync_grid()` -> global rendezvous (item 11); `CheckBudgets` (item 14); `T.serial(waves)` and
  `T.Persistent` dealing over tiles x SHAVEs.
* Kernel: two phases in one launch, e.g. split-K attention partials -> `sync_grid` -> combine (the
  shape of `examples/deepseek_mla/example_mla_decode_persistent.py`), or rmsnorm -> `sync_grid` ->
  GEMM.
* Acceptance: one `kernel_entry`, one launch, device golden passes; removing the `sync_grid` from
  the source makes the device golden FAIL (the negative control that proves the barrier is live).
* 2b: `DeriveGlobalSync` inserts it automatically; acceptance = identical generated source with and
  without the hand-written `sync_grid`, and a fixture with no cross-phase dependency gets none.

### Phase 3: multi-tile ping-pong matmul flow
* Items 7, 8, 9; `PlanCmx` live-range packing; hardware barriers. Target schedule: A split across
  tiles and resident; B broadcast (multicast), ping-ponged, each chunk split across both DMA ports;
  DPU FIFO alternated by the two SHAVEs with no sync between them.
* Acceptance: device trace admissible (per-SHAVE event count below the profiler's saturation
  value; `NPU_PROFILE_FRC_MHZ=19.2` exported), a ported flow checker
  (`testing/python/intel_npu/device/check_flow_trace.py`) exits 0, DPU busy % and TOPS at the
  reference shape reported **unprofiled** with the peak basis stated. Negative: a source whose tile
  split is on the streamed loop is refused; a per-tile unicast copy of a tile-invariant source is
  refused.

### Phase 4: SHAVE ops and quantised GEMM
* Typed SHAVE op surface (`T.npu.rmsnorm`, `softmax`, `silu`, `dequant_i4`, ...) over
  `call_extern`, each backed by a function in `npu_rt.hpp`; int4 group-quantised weights:
  dequantise on the SHAVE, fp16 DPU matmul (`DEQUANT_THEN_FP16`), later the DPU-native per-channel
  path. K-split with a SHAVE accumulate.
* Acceptance: conformance matrix {per-channel, grouped} x {int4, int8} x {M=1, M=32} vs numpy in
  the host sim and on the device, with a negative control (wrong weight layout must miss).
* Library guard rails carried over from npu_dsl's three bring-ups (`enable-new-model` skill,
  "Pitfalls"), each as a refusal with a negative test: top-k gating with fewer than 32 experts;
  an attention key chunk or head dim not a multiple of 32; two layouts for one state key across
  decode and prefill kernels; a multicast round with a load in its epilogue.

### Phase 5: one decoder layer, then a model, as one megakernel
* Layer = norm -> qkv -> attention -> o_proj -> norm -> FFN, all in one `prim_func`, phases joined
  by (derived) `sync_grid`; then `for layer in T.serial(L)`.
* Acceptance: golden vs a PyTorch reference on a small model; `.text` under `text_limit_bytes`;
  decode ms/token and prefill TTFT measured unprofiled as interleaved A/B against the recorded
  npu_dsl numbers (external reference only; `docs/RUN_AND_RESULTS.md` in that repo), with repeats.
* Decide here whether a real `intel_npu` execution backend (Build/JIT/Runtime over `npu_runtime`)
  is worth it or the runner stays.

## 5. Measurement and evidence rules carried over

* A run is either profiled (observability) or unprofiled (timing), never both; every duration says
  which. `NPU_PROFILE_FRC_MHZ=19.2` for every capture (the runtime's 38.4 default halves durations).
* A profiled trace reports per-SHAVE event counts; a count at the profiler's cap is SATURATED and
  inadmissible.
* Every quoted TOPS states ops (`2 M N K`), the unprofiled kernel-only time, and the peak basis.
* Evidence (traces, logs, generated `.cpp` with sha256 prefix) goes under
  `testing/python/intel_npu/device/evidence/` with an `INDEX.md` row per device session.
* Hardware facts come from the hardware owner, never inferred from code or registry text. Open
  questions are listed in section 7 and asked in chat, not answered by guessing.

## 6. Out of scope for this plan

NPU60XX values; a TVM upstream change (`3rdparty/` is not edited); OpenVINO/VPUX graph path
interop; a Python `nl.*`-compatible shim.

## 7. Open hardware questions (for the hardware owner)

1. For a persistent grid, is the SHAVE count per launch fixed at `num_tiles * shaves_per_tile`, or
   may a kernel request fewer SHAVEs on a tile?
2. Which hardware barrier ids are free for a kernel's own use on this driver (plan assumes 12..19)?
3. Is a `sync_grid` across tiles implementable with one hardware barrier, or only as a software
   rendezvous through DDR/CMX flags?

## 8. Log

* 2026-11-06: plan written; branch `feat/intel-npu-backend` created from `main` (`da53da96`);
  the untracked 2026-10-29 `tilelang/ptl` PoC and its adapter edits removed; working tree is `main`
  plus this file.
* 2026-11-06: **Phase 2 closed (2a).** `T.sync_grid()` -> `ctx.sync_grid()` (npu::Sync::waitGlobal),
  `T.sync_threads()` -> `ctx.sync_tile()` (waitLocal); `CheckIntelNpuBudgets` refuses a kernel whose
  CMX allocations x shaves_per_tile + runtime reserve exceed the target's `cmx_bytes`, naming the
  buffers (negative: the Phase 1 over-sized kernel; the verdict follows the target attribute). Two-phase
  megakernel (`kernels.make_gemm_two_phase`: split-K partials on half the workers -> `sync_grid` ->
  fp16 combine on all, one launch): host sim PASS and device **PASS** (rel 5.67e-4); without the
  barrier 5/5 device launches FAIL, so the barrier is live. Harness: `lowering.py` binds arrays by the
  device parameter order (`SplitHostDevice` reorders parameters). Device finding, fixed: a SHAVE's
  direct DDR store needs `npu::sync::memoryFence()` or it can be lost at kernel end (the Phase 1
  "unwritten status" one-off, ~1 in 25 launches); the runtime now fences after the status record and
  before every barrier; 180 launches since, 0 lost. Details and the trace row in
  `device/evidence/INDEX.md`. **Open for 2b:** `DeriveGlobalSync` (automatic `sync_grid` from
  global-buffer dataflow) is not started; `.text` is still unmeasured (the runner can read the .elf).
* 2026-11-06: **Phase 1 closed.** `LowerIntelNpuTileOp` (C++, before `FlattenBuffer`) lowers `copy`
  (DDR<->CMX, 1-D/2-D, byte widths and row strides from the 2-D buffers) to `intel_npu_dma_copy` and
  `gemm` (fp16, `transpose_B`, `clear_accum=True`) to `intel_npu_dpu_matmul`; every other tile op is
  refused by name. Codegen prints them as calls on a per-SHAVE `npu::Ctx`; DDR tensors are
  `npu::Ddr<T>` (64-bit addresses, since a SHAVE pointer is 32-bit); CMX buffers are `ctx.alloc<T>(n)`
  (runtime bump allocation, SHAVE-private slots of one tile allocation). Runtime:
  `src/tl_templates/intel_npu/npu_rt_device.hpp` over generic_compute (DmaDrv 2-D copy, DpuMatMul,
  trap-window-aware descriptor placement, per-SHAVE status record) and `npu_rt_host.hpp` (threads +
  memcpy + fp32-accumulate model). Host sim: `testing/python/intel_npu/test_intel_npu_phase1_gemm.py`
  (23 tests in the directory pass: numpy match at rel < 2e-3 on even and uneven block dealing,
  perturbed-output FAIL, K-split refusal, CMX exhaustion in the status record). Device:
  `testing/python/intel_npu/device/` (CMake project, `npu_run` host runner, `runner.py`,
  `run_gemm_device.py`): 256x256x512 fp16 GEMM on 2 tiles x 2 SHAVEs **PASS** vs numpy (rel 3.24e-4),
  both negative controls behaved; evidence row in `device/evidence/INDEX.md` (trace 8-24 events per
  SHAVE, built against `gc_dsl_barrier` `daf6c05`).
  Amendments: (a) the gemm K-split refusal keys on `clear_accum=False` (semantic: the DPU overwrites),
  not on "gemm inside a serial loop"; (b) `PlanCmx` is a runtime bump allocator in Phase 1, with the
  compile-time budget check deferred to `CheckBudgets` (Phase 2); (c) `PlanAndUpdateBufferAllocationLocation`
  is not run (it sinks CMX buffers into loops; a CMX allocation is once per kernel); (d) scope `shared`
  and `local` are both SHAVE-private until Phase 3 places tile-shared buffers. One unexplained
  one-off (first launch after the first build reported one SHAVE's status unwritten) was recorded in
  `device/evidence/INDEX.md`; it recurred and was root-caused and fixed in Phase 2 (missing DL1 fence).
* 2026-11-06: **Phase 0 closed.** `tilelang/intel_npu/` (target, manifest, pipeline, dialect with a
  1-D `T.Kernel`, source-only codegen, `source_only` execution spec), `src/intel_npu/` (target kind
  with PTL defaults, `LowerIntelNpuKernelLaunch`, `CodeGenTileLangIntelNpu` + `kernel_entry`
  emission, `target.build.tilelang_intel_npu_without_compile`), `src/tl_templates/intel_npu/npu_rt.hpp`
  (declarations). `testing/python/intel_npu/test_intel_npu_phase0.py`: 16 passed, including the
  negative controls (other backends' spellings return None; the CPU backend's source lacks
  `kernel_entry`/the ctx launch queries; a 2-D grid is refused at the dialect and in the pipeline; a tile op
  is refused by name; a compiled-device request fails at the codegen boundary).
  `testing/python/intel_npu/show_phase0_source.py` prints the emitted skeleton. Build note: on
  Windows `tilelang_objs` links into `tvm_compiler.dll`; build with
  `cmake --build build --target tvm_compiler tvm_runtime tvm_ffi_shared`, never interrupt ninja
  (a killed run corrupts `.ninja_deps` and forces full rebuilds until it is deleted).
