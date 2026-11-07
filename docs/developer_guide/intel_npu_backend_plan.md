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

**Cost model (2026-11-06, section 3.2).** VPUNN is the second external dependency, pinned like the SDK:
`VPUNN_ROOT=D:\dsl\vpunn`, a standalone clone of `intel-innersource/libraries.performance.modeling.vpu.nn-cost-model`
at tag **v1.10.2** (`8fbfa294`, the version `vpux-plugin` pins). It is Intel innersource with embargo markers,
so it is **never vendored** into this public tree; `maint/scripts/build_intel_npu_vpunn.py` builds its pybind
module (MSVC; `--deps-dir` points at a vpux-plugin build's `_deps` when GitHub is unreachable) and
`tilelang.intel_npu.vpunn` loads it. A tree without it still compiles and runs; the cost tests SKIP by name.

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
| 6b | `PlanIntelNpuWorkloads` (Python) | per DPU workload, the MPE cuboid and spatial split ranked by VPUNN (section 3.2); explicit knobs kept; modes auto / require / off; what was chosen or skipped is written into the generated source | 3b (done) |
| 7 | `PlanIntelNpuDistribution` (Python, done 3b) | per DDR->CMX copy: into a shared buffer outside a pipelined loop = resident (SEGMENTED when the source names the tile), rows split over the tile's SHAVEs on both ports + tile sync; into a shared buffer inside a pipelined loop = the streamed DUPLICATED operand, refused if its source depends on the tile (a tile split on the streamed loop); into a local buffer = per SHAVE, noted when the source is tile-invariant. The tile-variable rule is the legality floor; the cost model's role is advisory (section 3.2) | 3 |
| 8 | `PlanDualPort` | folded into the ring: every multicast fill is two contiguous halves, one per port, issued together (`Ctx::ring_fill`) | 3 (done) |
| 9 | `LowerIntelNpuPipelined` (Python, done 3b) | `T.Pipelined(num_stages=S)` over one streamed fill + gemm + stores -> the ring program: S == SHAVEs per tile (slot = SHAVE), chunk c -> SHAVE c % S, streamed buffer x S slots, output x 2 by use parity (submit-ahead), per use `wait(u); submit(u+1); fill(u+1) [issuer]; store(u)`; issuer = tile 0's SHAVE s; barrier ids placeholders | 3 |
| 9b | `AssignIntelNpuBarriers` (Python, done 3b) | READY/FREE x slots x 2 parities per ring from the job's pool `[barrier_pool_first, barriers_per_tile x tiles)` below sync_grid's 3; refuses by name with the shortfall | 3 |
| 10 | `DeriveGlobalSync` | insert `sync_grid` from global-buffer RAW/WAR/WAW across top-level phases when the author did not; buffer granularity (done 2b; region granularity and loop-carried phases later) | 2b |
| 11 | `InsertSync` | tile-level sync for shared CMX; map `sync_grid` to a software rendezvous, hardware barrier ids for the pipelined protocol | 2 |
| 12 | `PlanCmx` | static CMX allocation per scope with live-range packing against `cmx_bytes`; refuse overflow naming the buffer | 1 (bump) / 3 (packing) |
| 13 | shared legalisation: `DecoupleTypeCast`, `LegalizeSafeMemoryAccess`, `LowerOpaqueBlock`, `FlattenBuffer`, `NarrowDataType(32)`, `Simplify`, `RemoveNoOp` | shared | 1 |
| 14 | `CheckBudgets` | CMX: allocations x shaves_per_tile + runtime reserve vs `cmx_bytes` (done 2a); `.text` is measured from the built `.elf` by the device runner vs `text_limit_bytes` (done 2b), not estimated | 2 |
| 15 | `SplitHostDevice`, `AnnotateDeviceRegions`, `LowerDeviceKernelLaunch` | shared; host side is a stub in source-only mode | 0 |

Every pass ships with a negative control (a fixture it must refuse or leave unchanged, shown to
fail when the check is removed). A pass is not done when it is only observed passing.

### 3.1 Design inputs from the production compiler (VPUX), read 2026-11-06

npu_dsl was a PoC; the production NPU compiler is VPUX (`applications.ai.vpu-accelerators.vpux-plugin`),
and its local knowledge base (`knowledges/`: `hw/npu5/{nce-subsystem,dma-controller,dpu,shave}.md`,
`compiler/{workload-management,memory-scheduler,multi-shave,sdpa-extended,platform-resources}.md`) is
the reference this backend is designed against from Phase 3 on. The HAS-derived files are Intel-internal:
they are cited here by file and section, not copied. What changes in this plan because of them:

| Topic | VPUX / HAS fact (citation) | Consequence for `intel_npu` |
|---|---|---|
| Synchronisation | DPU, DMA and SHAVE communicate through **hardware barriers only**; IPC/software sync is a measured performance loss (`nce-subsystem.md` §5, §8). VPUX models every op as a task with *wait* and *update* barrier sets: virtual barriers -> legalisation -> physical assignment by graph colouring -> **barrier simulation** that proves deadlock-freedom (`workload-management.md` §5, §11) | Phase 3 adopts the task/barrier model: every DMA and DPU op in a pipelined region carries wait/update barrier sets; `AssignBarriers` colours them onto the tile's physical pool; a compile-time `BarrierSimulation` pass refuses a schedule that can deadlock. `sync_grid` phase boundaries stay on the toolkit's software rendezvous until a hardware version is measured faster |
| Barrier resources | 16 barriers per tile in the job, 48 on PTL, ids 48-127 fault; each barrier has a 32-deep FIFO that **auto-reloads** producer/consumer counts when both reach zero (`nce-subsystem.md` §4.2, §5). Ids 0-7 are taken or faulting on this driver (npu_dsl measurement, `platform_configs/ptl.json`), so the usable pool is 8-47 | Target attributes `barrier_pool_first=8`, `barrier_pool_count=40` (the 0-7 exclusion is measured, not from the HAS; section 7 question 2 stays open). Auto-reload replaces the prior art's manual re-arm for pipelined rounds |
| Distribution | VPUX names: `SEGMENTED` (disjoint slice per tile), `DUPLICATED` (full copy on every tile, one DMA), `OVERLAPPED`, `MULTICASTED`; a distributed DMA consumes **all DMA ports** (`workload-management.md` §8, §10). Multicast is one DDR read at unicast throughput (`dma-controller.md` §3). A DUPLICATED operand sits at the **same CMX offset on every tile** | `PlanDistribution` emits these modes by name; a DUPLICATED fill is one multicast DMA (same offset everywhere: our allocator's identical per-tile sequence guarantees it), split in two halves over the two ports |
| Remote CMX | DPU and SHAVE can **write** a remote tile's CMX (ITI) but never read it; remote reads fault (`nce-subsystem.md` §2.1, §9) | Scope `shared` means "the two SHAVEs of one tile"; cross-tile data moves only by DMA multicast. The dialect never offers a cross-tile read |
| CMX | 1536 KiB physical per tile on PTL; the compiler-usable workspace is **1439 KiB** after the runtime's reserved regions (`platform-resources.md` §3). Our kernels observe exactly 1473536 B free at entry. Accesses 32 B aligned; VPUX allocates at 64 B; a tiling heuristic keeps ~10% slack for fragmentation (`FRAGMENTATION_AVOID_RATIO`) | `cmx_bytes` default becomes the **workspace** 1473536, not the physical size; `CheckBudgets` compares against it; allocations stay 64 B aligned |
| DPU workload legality | Output and input Z (N, K) multiples of 16; any workload dimension <= 8192; max output points per operation 8192 on VPU5 (`dpu.md` §4.6, §13). Stencils 16/2, 8/4, 4/8 with reduced activation reuse vs VPU4 (§4.5); IC-split secondary accumulator exists but is only enabled for a grouped 1x1 pattern (§4.7) | `LowerIntelNpuTileOp` refuses N % 16, K % 16 and M*N > `dpu_max_output_points` (target attribute 8192) with the reason; stencil selection (`setSpatial`/`setStencil` in the prior art) is a Phase 3 performance item driven by M, N |
| DPU FIFO | One workload FIFO per tile, 256 entries (`nce-subsystem.md` §4.2) | Two SHAVEs enqueue on one DPU without a sync between them, as the flow assumes |
| SHAVE and DDR | A SHAVE reaching DDR through its cache gets < 10% of DMA bandwidth (12 vs 256 outstanding) (`nce-subsystem.md` §7, §8) | Scalar `Ddr<T>` element access is for status words and the odd scalar, never a data path; a pass flagging scalar DDR access inside a loop is a Phase 4 item. Per-SHAVE SW-kernel inputs 32 B aligned (`multi-shave.md` §8.1) |
| Scheduling | VPUX list-schedules with per-op cycle costs (VPUNN) over executor FIFOs, prefetch window 35 levels, loop ping-pong with two banks freed every second iteration, loop regions allocated first (`memory-scheduler.md` §8, §10; `workload-management.md` §3, §7) | Phase 3 keeps the author's explicit `T.Pipelined(num_stages=2)`; a cost-driven scheduler is a later item, and when it comes it is this model, not npu_dsl's six-coefficient fit |
| Two SHAVEs, one DPU | SDPA runtime: SHAVE submits pre-built DPU descriptors; two SHAVEs split the L dimension and each runs the full head pipeline on its slice (`sdpa-extended.md` §6) | Confirms the Phase 3 ping/pong ownership (SHAVE 0 = PING chunks, SHAVE 1 = PONG chunks) is the production pattern |

### 3.2 GEMM tiling, multi-cluster and workload split in VPUX; what is reusable (read 2026-11-06)

`<user>` asked, before Phase 3b: how does VPUX tile a GEMM, and can its code be used here. Sources are
under `applications.ai.vpu-accelerators.vpux-plugin/src/vpux_compiler/` (Apache 2.0); paths below are
relative to it. Line numbers are as read on 2026-11-06.

**How VPUX does it.**

| Layer | Mechanism | Where |
|---|---|---|
| MatMul on the DPU | No standalone MatMul op. `NCE.MatMul` is a **grouped 1x1 convolution** over a 5-D `[G,N,H,W,C]` tensor: M on the spatial dims, N = output channels, K = input channels; legal only in GNHWC with groups >= cluster count, default multi-cluster strategy SplitOverGroup | `src/dialect/VPU/utils/nce_matmul_utils.cpp` (`inferNCEMatmulOutputType`, `isNCEMatMulSupported`); interfaces `tblgen/.../VPU/ops_interfaces.td:60-177` (`ClusteredOpInterface`), `:487-551` (`TilingBuilderOpInterface`: `getTilingStrategy`, `backInferTileInfo`, `adjustAttrs`) |
| Tiling of one layer | Three modes, `TilingMode { ISOLATED, PIPELINING, PREFETCHING }` (`include/vpux/compiler/core/tiling.hpp:109-119`): ISOLATED = fewest tiles that fit CMX; PIPELINING = more tiles so this layer's DMA overlaps its own DPU; PREFETCHING = more still so the next layer's DMA overlaps this layer's DPU. Search: start at one tile, increase the divisor on the lowest-indexed dimension until the per-tile footprint fits CMX minus reserved minus a ~10% fragmentation margin, cost each candidate with VPUNN, keep the cheapest legal one | `src/dialect/VPU/transforms/passes/tiling_strategy_assignment.cpp`, `src/dialect/VPU/utils/cost_model/layer_vpunn_cost.cpp:436-500`, `src/dialect/VPU/utils/tiling_algorithm/tiling_general_algorithm.cpp` (`fillDividedTiles`), `passes/apply_tiling.cpp` |
| Multi-cluster | `StrategyManager` evaluates SOH / SOK / Clustering / HKSwitch / SplitOverBatch / SplitOverGroup per layer: cost = VPUNN DPU cycles + weights DMA cost (theoretical DMA model), constrained by 4-D/5-D rank, channel % 16, SOK needs OC >= 16 x tiles, `doesLayerFitIntoCMX` | `include/.../VPU/utils/strategy_manager/strategy_manager.hpp:19-48`, `src/dialect/VPU/utils/strategy_manager/multi_cluster_strategy_utils.cpp` (MPE volume per cluster at `:443`), `passes/multi_cluster_strategy_assignment.cpp` |
| DPU workload split | `DpuTiler` enumerates a split pool (over H, over W, H x W factor pairs; `generateSplitNumberPool(numDPU, maxSplits)`) for each MPE mode `CUBOID_16x16 / 8x16 / 4x16` (the 16/2, 8/4, 4/8 stencils of `dpu.md` §4.5), costs every candidate with VPUNN L1 and materialises `VPU.DPU.Workload {outOffsets, outSizes, padding, mpeMode, clusterId}` | `src/dialect/VPUIP/interfaces/dpu_tiler.cpp:149-284`, `passes/split_nce_ops_onto_workloads.cpp:40-71`, `include/.../VPU/utils/workload_split_utils.hpp:29-65` |
| Scheduling | `FeasibleAllocation`: already recorded in section 3.1 (Scheduling row) | `knowledges/compiler/memory-scheduler.md` |

All three decisions that Phase 3b needs (tile shape and depth, SEGMENTED vs DUPLICATED per operand,
stencil per workload) are made by **one cost model, VPUNN**, not by rules. npu_dsl's `dpu_rate_probe`
had measured the stencil/spatial choice moving DPU efficiency from 56% to 97% at one shape; VPUX computes
that choice instead of guessing it.

**What can be reused.**

| Component | Verdict | Why |
|---|---|---|
| **VPUNN** (`thirdparty/vpucostmodel/`) | **Reusable as-is**: link or pip-install | Standalone CMake library, Apache 2.0, **no MLIR dependency**. Input is a plain struct `DPUWorkload { VPUDevice device; Operation op; VPUTensor inputs[1], outputs[1]; kernels[2]; strides[2]; padding[4]; ExecutionMode execution_order; act/weight sparsity; Swizzling; output_write_tiles; ISIStrategy isi_strategy; ... }` (`include/vpu/dpu_workload.h:32`); `Operation` is `{CONVOLUTION, DW_CONVOLUTION, ELTWISE, MAXPOOL, AVEPOOL, CM_CONVOLUTION}` (`include/vpu/dpu_types.h:106`), so a GEMM is costed exactly as VPUX does it, as a 1x1 CONVOLUTION. `VPUDevice` has `NPU_5_0` / `NPU_5_0_W`; model files `models/vpu_5_1*.vpunn` (0.8 MB), `models/dma_5_0.vpunn`. Layer-level API: `VPULayerCostModel` + `VPULayerStrategy { nDPUs, nSHVs, nTiles, tiling_strategy, prefetching }`. Optional Python binding (`ENABLE_PYTHON_BINDING`, `python/`, `pyproject.toml`) and CLI (`apps/cost_model_cli`) |
| Tiling search, strategy choice, workload-split **algorithms** | **Portable, not linkable** (a few hundred lines of logic each) | The arithmetic depends only on shapes, CMX size and a cost; the MLIR coupling is the interface layer (`TilingBuilderOpInterface`, `ClusteredOpInterface`, `NDTypeInterface`, `mlir::Operation*`) |
| The passes themselves, `FeasibleAllocation`, distributed tensor types | **Not reusable** | Bound to the VPU / VPUIP / async dialects and MLIR 17+, flatbuffers, OpenVINO |
| "Graph compiler versus kernel compiler" | **Not an obstacle** | Inside one megakernel phase the objects are the same: a VPUX layer is our tile op; its tiling is our expansion of `T.gemm` into a workload loop; its multi-cluster strategy is our SEGMENTED/DUPLICATED choice; its memory scheduler is our CMX plan plus pipelined prefetch. The cross-layer machinery (vertical fusion, spilling) is not needed yet |

**Consequence for Phase 3b** (replaces the rule-only item 7 of section 3):

1. **Step 0, cost model.** Use VPUNN as an external dependency through its Python binding built from
   `thirdparty/vpucostmodel` (no coupling into the `tvm_compiler` build; the same model serves offline
   autotuning). `testing/python/intel_npu/gemm_cost.py` maps `(M, N, K, dtype, MPE mode, distribution,
   tiles)` to a `DPUWorkload` by the same MatMul -> grouped 1x1 CONVOLUTION construction as
   `layer_vpunn_cost.cpp`, and returns DPU cycles plus the theoretical DMA cost.
2. **Calibrate before trusting.** Compare the prediction with the device at two or three shapes (the
   Phase 1 GEMM, the npu_dsl stencil-sweep shapes) and record the ratio; VPUX also calibrates it.
3. **Three cost-driven passes**, Python first, C++ once stable: `PlanWorkloads` (stencil and split pool
   per `T.gemm`, VPUNN-ranked, legality from `dpu.md` §4.6), `PlanDistribution` (SEGMENTED vs
   DUPLICATED per operand by DPU + DMA cost, with the tile-variable rule as the legality floor),
   `TileGemm` (ISOLATED -> PIPELINING divisor search under the CMX fit of `CheckBudgets`).

Decided by `<user>` 2026-11-06: (a) VPUNN through its Python binding, not linked into `tvm_compiler.dll`;
(b) cost decisions live in Python-level passes, C++ is reserved for lowering and codegen.

**3b pass order and barrier budget (agreed with `<user>` 2026-11-06).** Only decisions that change a
workload's (M, N, K) run before `PlanWorkloads`; it runs last and reads the final shape. In a DSL where the
author writes the loop order, which operand is split and which is broadcast is *derived* from the source
(a fill whose DDR address depends on the tile variable is SEGMENTED, otherwise DUPLICATED; README section 3
"nothing is named"), so `PlanDistribution` derives and refuses, and the cost model's role there is advisory
(the notes say what the swapped roles would cost). The shape search (`TileGemm`: BM, BN, stages, tiles) is a
VPUNN-pruned search over the kernel factory's parameters **before lowering**, where TileLang's autotuner
sits, not an IR rewrite. Every candidate's DPU cost is `plan_gemm(M, N, K)` (the cost at the best stencil),
so the passes cannot disagree.

```
[Python, pre-lower]  TileGemm: VPUNN-pruned search over (BM, BN, stages, tiles)        <- changes shapes
[IR] LowerTileOp -> PlanDistribution (derive SEGMENTED/DUPLICATED, refuse unicast of an invariant source)
     -> LowerPipelined (slots, ports, READY/FREE) -> DeriveGlobalSync -> VerifyUniformSync
     -> PlanWorkloads (final shape -> cuboid, x)                                           <- last cost pass
     -> AssignBarriers -> BarrierSimulation -> CheckBudgets -> ... -> codegen
```

Barriers: the job owns `[8, 16 x tiles)`; `sync_grid` takes 3 rotating ids however many syncs are derived;
one ping-pong ring takes READY/FREE x 2 slots x 2 parities = **8**. So 3 tiles: 40 - 3 = 37, up to 4 rings;
2 tiles: 21, 2 rings; **1 tile: 5 < 8, the hardware ring is refused by name** (no multicast on one tile
anyway). `AssignBarriers` becomes the one owner of the pool (the 3 sync ids move from codegen's
`pool_end - 3` into it), hands ids to codegen as IR annotations, and refuses when the pool runs out, saying
by how much. The two users never share an id; id reuse across phases separated by a `sync_grid` (VPUX's
graph colouring) is deferred until a kernel needs it. Build order: **A** runtime `Ring` + a hand-written
kernel verified on the device, **B** the passes emit the same program from the DSL, **C** `check_flow_trace`
+ TOPS.

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
* 3a (closed): `sync_grid` on hardware barriers; see section 8.
* 3b (closed 2026-11-06; A, B, C): the pipelined flow above, with tile shape, distribution and stencil chosen by
  the VPUNN-backed passes of section 3.2 (Step 0 cost model and calibration first). Still open after 3b: a
  host-simulation ring so the suite runs the flow without hardware; `DeriveGlobalSync` does not yet see
  `ring_fill` as a DDR read; rounds for a resident operand larger than CMX; TileGemm (the shape search).
* Acceptance (met, section 8 step C): device trace admissible (per-SHAVE event counts 32-66, cap 1984;
  `NPU_PROFILE_FRC_MHZ=19.2` exported), the flow checker (`device/trace_timeline.py --check`) exits 0 on
  the flow trace and 1 on a plain GEMM trace, DPU busy % and TOPS at the reference shape reported
  **unprofiled, kernel-only** (NCE_TIME entry/finish stamps in the status record) with the peak basis stated.
  Negatives: a source whose tile split is on the streamed loop is refused; a per-tile unicast copy of a
  tile-invariant source is noted in the source (relaxed from refusal: fixtures and probes do it legitimately).

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
2. ~~Which hardware barrier ids are free for a kernel's own use on this driver (plan assumes 12..19)?~~
   **Measured 2026-11-06** (`device/evidence/barrier_probe*.json`): a job owns `16 x tiles` ids from 0;
   0-3, 5, 6 never go live and 7 is the toolkit mutex, so the pool is `[8, 16 x tiles)`. Re-probe after
   any firmware or toolkit change (`python device/probe_barriers.py`).
3. ~~Is a `sync_grid` across tiles implementable with one hardware barrier, or only as a software
   rendezvous through DDR/CMX flags?~~ **Done 2026-11-06**: three rotating hardware barriers per
   launch, 5.7 us faster per barrier than the toolkit's software rendezvous (Phase 3a log entry).
4. **DMA DDR->CMX ceiling.** Measured ~17.5 GB/s on one port and ~21.7 GB/s on two, from one issuer,
   and no more with 3 or 6 concurrent issuers (`device/run_dma_rate_probe.py`, 128 KiB 1-D copies).
   Is ~22 GB/s the expected DMA read bandwidth from DDR on this PTL box, or is a descriptor setting
   (burst / `dst_width`, 2-D vs 1-D, 2 MiB pages) leaving bandwidth on the table?
5. **fp16 x fp16 DPU rate.** The best measured fp16 workload reaches 2743 MAC/ns per tile (calibration,
   64x64x2048, CUBOID_8x16) and the flow's 32x256x2048 workload 1540 MAC/ns, against the 3790 MAC/ns
   implied by the 22.73 TOPS basis (2048 MAC/cycle x 1.85 GHz x 3 tiles). Is 2048 MAC/cycle the fp16 x fp16
   rate on NPU5, or the int8 (or fp16 x int8 mixed) rate, with fp16 x fp16 at half? The answer sets the
   honest peak basis for every TOPS percentage in this plan.

## 8. Log

* 2026-11-06: plan written; branch `feat/intel-npu-backend` created from `main` (`da53da96`);
  the untracked 2026-10-29 `tilelang/ptl` PoC and its adapter edits removed; working tree is `main`
  plus this file.
* 2026-11-06: **Phase 3b step C closed: the flow is measured kernel-only and checked from its trace.**
  (1) Kernel-only time: the status record grew to 64 B with NCE_TIME stamps at `Ctx` entry and `finish`;
  `npu_run` reports `median_kernel_ms` = max(finish) - min(entry) over the SHAVEs (19.2 MHz); the host wall
  carries ~0.15-0.2 ms of submit on top. (2) `device/trace_timeline.py` renders one steady-state window per
  cluster (DPU, SHAVE 0/1, the tile's DMA row) and `--check` is the flow verdict (same workload count on every
  cluster, alternating SHAVE submissions, fills on tile 0's DMA row only, no row at the profiler cap, DPU busy
  >= a bar): exit 0 on the flow trace, 1 on the Phase 1 GEMM trace; a hardware-free test pins both. (3) **Numbers
  (UNPROFILED, kernel-only, 3 tiles x 2 SHAVEs, fp16 x fp16):** reference shape Y[2048, 768] = X[2048, 2048] .
  W[768, 2048]^T, BM=32, BN=256: **0.799 ms = 8.07 TOPS = 35.5% of 22.73 TOPS** (basis 2048 MAC/cycle/tile at
  1.85 GHz; 32.1% of 25.13 turbo), DPU IDU busy 87%, so the flow is DPU-bound and the DPU's own rate (10.9 us
  per 32x256x2048 workload = 1540 MAC/ns) is the gap. Same ops at BM=64, BN=128 (N=384): 0.401 ms = 8.04 TOPS,
  DPU 70% busy, DMA-bound; roles swapped (`make_ring_flow_swapped`): 1.090 ms = 5.91 TOPS. The compiled flow
  equals the hand kernel (0.799 vs 0.799). (4) **DMA ceiling, measured** (`device/run_dma_rate_probe.py`,
  128 KiB DDR->CMX, NCE_TIME): one issuer one port 17.5 GB/s, one issuer two ports **21.7 GB/s**, multicast to 3
  tiles costs nothing (17.6 / 21.7), and more issuers do NOT add bandwidth (3 issuers x 2 ports 14.7, 6 x 2 8.7
  GB/s). So the box's DMA DDR->CMX ceiling is ~22 GB/s and the fill-vs-compute balance is a shape property:
  bytes per MAC = 2 / (3 BN) for the streamed operand, 14.8 GB/s needed at BN=256 (fits), 29.7 at BN=128 (does
  not), matching the two measurements. The prior art's 62 GB/s "golden" was a vpuem simulator trace, not
  hardware. TileGemm has its two constants: VPUNN DPU cycles x ~1.6 and 22 GB/s. (5) Lessons: the host-sim
  library cache now keys on the runtime headers too (a stale 32-byte-status library passed as "unwritten
  status"); `--no-build` reuses stale device ELFs after a runtime header change, so rebuild after one.
* 2026-11-06: **Phase 3b step B: the compiler emits the flow from the author's loops.** Three Python
  passes after `LowerIntelNpuTileOp` (section 3 items 7, 9, 9b): `PlanIntelNpuDistribution`,
  `LowerIntelNpuPipelined`, `AssignIntelNpuBarriers`; `PlanIntelNpuWorkloads` now also plans
  `ring_dpu_submit`. The source (`kernels.make_ring_flow`) is the README's: `alloc_shared` for the
  resident and streamed operands (the dialect's `alloc_shared` is scope `shared`, not the GPU
  `shared.dyn`, which the late kernel-launch pass takes for dynamic shared memory), one
  `T.Pipelined(num_stages=2)` loop with fill, gemm, store. Codegen prints `alloc_shared`, the ring
  builtins and `npu::Ctx::Ring` for the `Bind`; `CheckBudgets` counts shared buffers once. Every
  derivation is written into the generated source as `// intel_npu:` notes. **Device: the compiled
  flow PASSes** (rel 2.86e-4; 0.970 ms host wall incl. submit vs the hand kernel's 0.944; 6 SHAVEs OK;
  perturbed W FAILs), and the **generality clause holds**: `make_ring_flow_swapped` (roles swapped,
  streamed operand second, other names and loops) lowers and PASSes with zero compiler edits
  (rel 2.78e-4). Negatives, each seen to fire: depth != SHAVEs, tile-dependent streamed fill,
  one-tile launch, 3 rings on 2 tiles (short by 3 ids). Suite 55 passed (50 + 5 skips without VPUNN).
  Two rules relaxed on evidence: a tile-invariant copy into a local buffer is legal and noted, not
  refused (fixtures and probes do it legitimately); the ring borrows the compiler's shared slot
  buffer so `CheckBudgets` sees the slots.
* 2026-11-06: **Phase 3b step A: the ping-pong ring runs on the device from the runtime.** `npu::Ctx`
  gained `alloc_shared` (one tile-shared allocation, same offset on every tile), `dpu_prepare` /
  async DPU with barrier setters, and `Ring`: `ring_begin` (slots = SHAVEs, order words, first-use
  configs on tile 0, one grid rendezvous), `ring_fill` (issuer = tile 0's SHAVE s for slot s: wait
  FREE[use-1], retire the previous halves, kick two multicast halves on ports 0 and 1 producing
  READY[use]), `ring_dpu_submit` (live checks, submission order through the order words, DPU gated on
  READY and producing FREE, next use's configs pushed by the issuer), `ring_dpu_wait`, `ring_end`.
  Barrier ids alternate by use parity (README 6c fact 4); the next fill is kicked at FREE, not
  pre-queued (6e, one outstanding job per SHAVE per link agent); submit-ahead with a parity
  double-buffered output (6c fact 3). Hand kernel `device/probes/ring_flow_hand.cpp` in the shape the
  compiler will emit: 3 tiles, W split by tile and resident (each SHAVE loads half the rows, then
  `sync_tile`), X streamed. **Golden PASS on the device** (rel 2.9e-4 at 2048x768x2048; perturbed W
  FAILs; profiled run PASS with a trace). Two findings: (1) **a latent Phase 3a race**: `hw_sync_grid`
  checked liveness with exact (p, c) counts, and under profiling skew a fast SHAVE produced before a
  slow one looked, which then reported NOT_LIVE on a healthy barrier (all six SHAVEs failed on id 45);
  liveness is now the consumer count alone, since nobody consumes before every producer arrived.
  (2) **The window rule is real** (README Round 774): at BM=32, BN=256, K=1024 (5.2 us DPU per
  workload) the FREE -> fill -> READY chain does not fit one workload and the DPU sits at ~70% of the
  kernel; at K=2048 (10.9 us) it is ~94%: 0.944 ms host wall incl. submit for 6.44 GFLOP, DPU IDU
  696 us per tile. Submit-ahead alone was worth 5%. Choosing K per chunk so that D >= chain is
  TileGemm's job (section 3.2), with VPUNN's DPU and DMA costs. Timing caveat: host wall includes
  ~0.2 ms of submit; a kernel-only stamp is step C's work before any TOPS claim.
* 2026-11-06: **`PlanIntelNpuWorkloads` landed (section 3 item 6b), the first cost-driven pass.** A Python
  `prim_func_pass` after the sync passes: for every `intel_npu_dpu_matmul` call still at the runtime
  defaults it ranks (cuboid, spatial x | M, x <= 64) with `vpunn.GemmCost.best_stencil` and writes the
  cheapest pair into the call; explicit kernel-wide knobs are kept. Modes `tl.intel_npu.plan_workloads`
  = auto (default; without VPUNN the defaults stay and the generated source carries
  `// intel_npu: DPU workloads NOT planned ...`), require (refuses by name), off. The notes travel as an
  `intel_npu.notes` AttrStmt inside the outermost `thread_extent` because SplitHostDevice keeps only a
  fixed set of PrimFunc attributes. A shape VPUNN cannot cost (K=32768 in the CMX negative) is left at
  the default and noted, so the budget check still fires. Generality: the two-phase kernel (64x64x256)
  is planned with zero edits. Device: Phase 1 GEMM golden PASS (rel 3.24e-4, kernel `0039c507d6e3`,
  `64x64x512 -> CUBOID_8x16 x=16`), two-phase PASS (rel 5.67e-4). Suite: 49 passed with VPUNN;
  44 passed, 5 skipped by name without it.
* 2026-11-06: **Phase 3b Step 0 done: VPUNN wired and calibrated on the device.** (1) VPUNN v1.10.2
  built as a pybind module from a standalone clone (section 0); `tilelang/intel_npu/vpunn.py` costs a
  GEMM tile as VPUX does (1x1 CONVOLUTION, ZXY, `W x H = M`, `N` output channels), plus a 1-D DMA cost.
  (2) The DPU builtin gained the two workload-shape knobs VPUX chooses per workload, `spatial_x` and
  `mpe` (CUBOID_4x16/8x16/16x16 as `(nthw_ntk, odu_nthw)` pairs (1,1)/(0,2)/(3,3), the ODU field set
  with the stencil), reachable kernel-wide through `tl.intel_npu.dpu_spatial_x` / `dpu_mpe` until
  `PlanWorkloads` sets them per op; the runtime publishes the DPU's own HWP counters (workloads, IDU and
  ODU durations) in status words 4-6. (3) Calibration, `device/run_vpunn_calib_device.py`, 4 tile shapes
  x 10 configurations x 2 repeat counts = 80 kernels, every golden PASS, UNPROFILED: **VPUNN's pick is
  within 0.0-2.8% of the measured best on every shape** (ranking concordance 30-39 of 45 pairs), while the
  runtime's default (x=4, CUBOID_16x16) is **36-42% slower than the best** (64x64x512: 1.564 vs 1.001 us;
  64x64x2048: 4.33 vs 3.06 us = 2743 MAC/ns). Absolute: measured IDU time / VPUNN time at 1950 MHz has
  median **1.58** (1.14-2.17; small workloads worst), so VPUNN is used for *ranking* and the ratio is the
  calibration constant, as in VPUX. Evidence: `device/evidence/vpunn_calibration.json`, INDEX row.
  Two findings recorded: the HWP `idu/odu_workload_duration` counters tick at the 19.2 MHz perf clock on
  this box (npu_dsl SS690.2), so they are reported in microseconds; and a function-local `static constexpr`
  array in SHAVE runtime code reads back wrong under moviCompile (it produced a full sweep of golden FAILs
  before being replaced by ternaries), so the runtime holds no static tables.
* 2026-11-06: **Pre-3b research filed (section 3.2).** `<user>` asked how VPUX tiles a GEMM and whether its
  tiling / multi-cluster / scheduling code can be used. Findings: MatMul is a grouped 1x1 convolution;
  tile count, SOH/SOK/Clustering and the DPU stencil per workload are all chosen by the VPUNN cost
  model under a CMX-fit constraint; VPUNN itself (`thirdparty/vpucostmodel`) is MLIR-free, has an
  `NPU_5_0` device and models, and is reusable directly; the algorithms are portable, the passes are
  not. Section 3 item 7 amended from rule-based to cost-based; Phase 3b gains a Step 0 (cost model +
  calibration). Two decisions taken by `<user>` the same day: VPUNN via its Python binding; cost
  passes in Python.
* 2026-11-06: **Phase 3a closed: `sync_grid` on hardware barriers.** `<user>` ruled: HAS is right, use
  hardware barriers as far as possible; which low ids are already taken is uncertain, so it is
  measured. (1) `device/probe_barriers.py` + `probes/barrier_probe.cpp` sweep every id of a job in its
  own process: 3-tile job ids 0-47: **0-3, 5, 6 never go live; 4 and 7-47 OK** (7 is the toolkit mutex,
  excluded); 2-tile job: 28-31 OK, **32 and 33 hang the launch** (context violation: the job owns
  `barriers_per_tile x tiles` ids). So the pool is **per job**, `[barrier_pool_first=8, 16 x launch tiles)`,
  and codegen derives it from the kernel's own `T.Kernel(tiles)`; `barrier_pool_count` is gone. (2)
  `Ctx::hw_sync_grid`: 3 rotating ids at the top of the job's pool, armed once per launch, worker 0
  re-arms after every SHAVE consumed. Chain of 200 grid barriers (`kernels.make_sync_chain`, token
  passed through DDR; wrong without every barrier): hw 0.469 ms vs sw 1.604 ms per launch, **5.7 us
  less per barrier**, both correct; no-barrier negative FAILs. Kill switch
  `tl.intel_npu.hw_sync_grid=False` keeps the software rendezvous for A/B. (3) **A hang is turned
  into a report**: every barrier wait is bounded (`kBarrierSpinLimit`), every SHAVE verifies the armed
  config is live itself (a dead barrier reads (0, 0), which a producer wait would take for "all
  arrived"), and failures surface as status `BARRIER_NOT_LIVE(id)` / `BARRIER_TIMEOUT(id)`; the
  generated source names the ids it uses. Negative control `run_dead_barrier_device.py` (sync on ids
  0-2): all four SHAVEs report `BARRIER_NOT_LIVE id 0` in 0.4 s. An id outside the job's allocation
  still hangs before any code runs; only the compile-time pool rule prevents that, and the runner's
  wall-clock timeout catches it. Finding, recorded in `kernels.py`: SHAVE scalar DDR stores go through
  a write-back DL1 with 16-byte lines, so two SHAVEs writing scalars into one line clobber each other;
  per-SHAVE scalars need their own 16-byte slot.
* 2026-11-06: **Phase 2b closed.** `DeriveIntelNpuGlobalSync` (C++, after tile-op lowering) inserts
  `sync_grid` between top-level phases with a DDR RAW/WAR/WAW hazard at buffer granularity; an author's
  barrier is kept and not duplicated; kill switch `tl.intel_npu.derive_global_sync=False` (used only by
  the negative controls). Acceptance met: the two-phase kernel without a hand-written barrier lowers to
  **byte-identical** source as with one (host sim and device, PASS rel 5.67e-4); a dependency-free
  two-phase fixture gets no barrier; a WAR-only fixture gets one. Scope limits, recorded: buffer (not
  region) granularity, so disjoint regions of one buffer still get a barrier; only the top-level phase
  sequence is analysed, a serial loop over phases (Phase 5) is the author's or a later refinement.
  `.text` is now measured from the SHAVE `.elf` by the runner and refused above `text_limit_bytes`:
  GEMM 17920 B, two-phase 20096 B (7.7% of 262272; mostly the kernel wrapper and runtime).
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
