# Intel NPU (NPU5 / Panther Lake) backend: implementation plan

Status: plan, 2026-11-06. Branch `feat/intel-npu-backend`. Living document: every phase below records
its outcome in place when it closes (date, commit, evidence path). Decisions that change get a dated
amendment, not a silent edit.

## 0. Decision

Build a **pure TileLang target backend** for the Intel NPU5 (PTL), owned entirely by this repo:
dialect, target, pass pipeline and C++ codegen, plus a small SHAVE runtime header of our own.
Nothing imports or links the previous `npu_dsl` compiler (`D:\dsl\libraries.npu.dsl`). Its design
documents are read as prior art only; its runtime headers (`fused_rt*.hpp`) are not reused. Amended
2026-11-06 (`<user>`): its hardware-verified SHAVE *algorithms* (vector math loops, dequant, the GDN scan in
`kernels_poc/common/fused_ops.hpp` and siblings) may be ported into this backend's runtime with the origin
named; the compiler side stays pure TileLang (section 4, M1-M6).

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

### Phase 4 and 5, re-planned 2026-11-06 around the model target (section 9): milestones M1-M6

`<user>`'s final target is **Qwen3.8-35B-A3B-Distill** (`empero-ai/Qwen3.8-35B-A3B-Distill`), a fine-tune of
Qwen3.6-35B-A3B (`qwen3_5_moe`: 40 layers, hybrid Gated DeltaNet + full attention, 256 experts top-8 with a shared
expert, int4 group-128 weights), enabled the way npu_dsl's `enable-new-model` skill prescribes. npu_dsl already runs
Qwen3.6-35B-A3B, so section 9 lists what the model needs against what this backend has, and the milestones below
replace the earlier Phase 4 / Phase 5 text. Every milestone: host sim first, device golden, then kernel-only time
against npu_dsl's same-shape number (the standing bar: not slower than npu_dsl).

* **M1 full-attention prefill phase** (in progress): the ring generalised to several streamed operands, SHAVE
  statements in the body and "the slot's last reader produces FREE"; RMSNorm, RoPE, q/k norm as SHAVE ops;
  fp16 qkv / o_proj. Acceptance: one attention block's prefill, device golden, kernel-only vs npu_dsl.
* **M2 int4 group-128 weights**: SHAVE dequant feeding the fp16 DPU (prefill) and the SHAVE GEMV (decode, M=1);
  int8 group-128 for the lm_head. Decides decode TPS.
* **M3 MoE layer**: top-8 softmax router over 256 experts, routed expert FFN (SwiGLU pairs), shared expert
  (sigmoid gate), combine; the routed-work distribution rule (npu_dsl's `e7_24_moe_routed_work.md` is the
  readable precedent).
* **M4 Gated DeltaNet layer**: causal conv1d and the delta-rule scan (the largest SHAVE op).
* **M5 one-kernel model**: layer table with a device-side layer loop (`when` on the layer kind), long-prompt chunk
  loop, `PlanCmx` live-range packing across phases, barrier id reuse across phases, `.text` budget per kernel;
  one decode kernel and one prefill kernel.
* **M6 e2e**: reuse npu_dsl's `qwen36_e2e` reference model, pack and golden check (model-side Python, reused
  across models by that skill itself) on this backend's ABI; golden 32/32; `longseq` TTFT / TPS as an interleaved
  A/B against npu_dsl's Qwen3.6 numbers.

Rules carried into M1-M6:
* **SHAVE algorithms may be ported from npu_dsl's hardware-verified runtime code** (`<user>`, 2026-11-06:
  `kernels_poc/common/fused_ops.hpp` vector loops, dequant, GDN scan), with the origin named at the port;
  the compiler side (IR, passes, emitter) stays pure TileLang. Section 0's "fused_rt not reused" is amended to
  this.
* **Performance is guaranteed by prune -> measure -> pin, not by a model** (`<user>`'s concern: npu_dsl's
  tiling choices were ad hoc). `TileGemm`: legal candidates (CMX, 8192 points, 16-alignment, the window rule
  (d - 1) x D >= chain) ranked by the composite model `t_chunk = max(DPU = VPUNN x calibration, fill = 4.5 us +
  bytes / 67 GB/s + 2 barrier latencies, SHAVE = measured ns per element)`, top-k built and timed kernel-only
  on the device in one build (80 kernels in ~30 s today), the measured best pinned with its evidence, and the
  model's regret recorded. A GEMM class may skip the sweep only once the model's first pick has regret <= 3% on
  every shape tried; MoE routed work is swept with representative routing, never random (skill 6b's lesson);
  decode GEMV is SHAVE/bandwidth bound and is measured, not modelled. Whole-model A/B before any pin (skill 6b).
* Library guard rails from npu_dsl's bring-ups stay as refusals with negative tests: top-k over fewer than 32
  experts; an attention key chunk or head dim not a multiple of 32; two layouts for one state key across decode
  and prefill; a multicast round with a load in its epilogue.
* The execution-backend question (Build/JIT/Runtime over `npu_runtime` versus the runner) is decided at M6.

## 5. Measurement and evidence rules carried over

* A run is either profiled (observability) or unprofiled (timing), never both; every duration says
  which. `NPU_PROFILE_FRC_MHZ=19.2` for every capture (the runtime's 38.4 default halves durations).
* A profiled trace reports per-SHAVE event counts; a count at the profiler's cap is SATURATED and
  inadmissible.
* Every quoted TOPS states ops (`2 M N K`), the unprofiled kernel-only time, and the peak basis.
* **Every timing runs under `NPU_TURBO=YES`** (the runner's default since 2026-11-07; `TL_NPU_TURBO=0` only
  for an A/B) and says TURBO / NO TURBO. Without it the DPU and the SHAVE run at about half the clock: every
  number in this plan dated before 2026-11-07 is NO TURBO (log entry of that date).
* Evidence (traces, logs, generated `.cpp` with sha256 prefix) goes under
  `testing/python/intel_npu/device/evidence/` with an `INDEX.md` row per device session.
* **Hardware facts are looked up before they are measured and before they are asked** (`<user>`, 2026-11-07):
  the VPUX knowledge base (`knowledges/hw/npu5/*.md`, `knowledges/compiler/platform-resources.md`; the
  `intel-npu-hardware` skill) is the first stop; a measurement confirms a sourced fact or exposes a
  difference; the hardware owner is asked what neither gives. Every constant the backend depends on is a
  `Fact` in `tilelang/intel_npu/hw_facts.py` with its `file#section` and, if measured, the probe;
  `test_intel_npu_hw_facts.py` checks the citations resolve and the target defaults equal the facts.
  Numbers and citations travel; the knowledge text does not. Open questions stay in section 7 with what was
  looked up and what was measured.

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
4. ~~**DMA DDR->CMX ceiling.** Measured ~17.5 GB/s on one port and ~21.7 GB/s on two, from one issuer,
   and no more with 3 or 6 concurrent issuers (`device/run_dma_rate_probe.py`, 128 KiB 1-D copies).
   Is ~22 GB/s the expected DMA read bandwidth from DDR on this PTL box?~~ **Answered by measurement
   2026-11-06** (`<user>` suspected it low; the npu_dsl record has 28.7 / 51.6 GB/s with 512 KiB jobs,
   `evidence/e7_20/multicast_probe`). A job-size sweep (`run_dma_rate_probe.py --kib 64..1024`, one issuer,
   NCE_TIME, synchronous start/wait per job) gives, one port: 64 KiB 6.15 us, 128 7.51, 256 11.4, 512 19.1,
   1024 34.6 us -> **~4.2 us fixed per job + ~34 GB/s streaming**; two ports: 5.47 / 6.07 / 8.12 / 12.0 / 19.7 us
   -> **~4.5 us fixed + ~67 GB/s aggregate** (53 GB/s at 1 MiB; the npu_dsl 51.6 at 512 KiB is reproduced
   within 15%). Multicast to 3 tiles costs nothing at any size; more issuers add bandwidth only above
   ~512 KiB. So the ring's 64 KiB halves run at ~22 GB/s because the **fixed latency** (descriptor issue to
   observed completion) dominates, not a bandwidth ceiling. TileGemm's DMA model is
   `t_fill = 4.5 us + bytes / 67 GB/s` for a two-port fill, and the lever is fewer, larger fills (deeper K
   per chunk), not more ports. Remaining hardware question: is ~34 GB/s per port the expected streaming
   rate, and what the ~4 us per job is made of (descriptor fetch, link-agent arbitration, completion
   write-back)?
   **Knowledge base (2026-11-07):** `hw/npu5/dma-controller.md#1,#3`: one DMA engine with two compute
   resource groups (the two "ports"), AXI DDR read 64 B/clk per group, 128 B/clk both, at the CMX clock
   (4/7 of the DPU clock); the 6D cap is `min(64, SRC_WIDTH)` B/clk per job, so a 2-D fill narrower than 64 B
   per row runs at the row width. The 34 / 67 GB/s above were measured **without turbo**. **Under turbo
   (2026-11-07, same probe, 64 KiB .. 1 MiB):** one port 4.37 / 6.98 / 5.76 / 9.51 / 16.98 us, two ports 3.49 /
   4.85 / 4.35 / 6.68 / 10.84 us; from the 512 KiB -> 1 MiB increments **~70 GB/s one port, ~126 GB/s two ports,
   ~2-2.5 us fixed per job**, which is the knowledge base's 64 B/clk per group at a CMX clock near 1.1-1.2 GHz
   (4/7 of the 2.0454 GHz DPU clock). TileGemm's DMA model under turbo: `t_fill = 2.5 us + bytes / 126 GB/s`
   for a two-port fill. The fixed ~2 us per job remains the only open part.
5. **fp16 x fp16 DPU rate.** The best measured fp16 workload reaches 2743 MAC/ns per tile (calibration,
   64x64x2048, CUBOID_8x16) and the flow's 32x256x2048 workload 1540 MAC/ns, against the 3790 MAC/ns
   implied by the 22.73 TOPS basis (2048 MAC/cycle x 1.85 GHz x 3 tiles). Is 2048 MAC/cycle the fp16 x fp16
   rate on NPU5, or the int8 (or fp16 x int8 mixed) rate, with fp16 x fp16 at half? The answer sets the
   honest peak basis for every TOPS percentage in this plan.
   **Narrowed 2026-11-06 by a same-day A/B against npu_dsl** (`<user>`: "we must not be slower than
   npu_dsl"). `device/run_dpu_rate_compare.py` reimplements npu_dsl's `dpu_rate_probe` method (one SHAVE,
   start/wait, median `idu_workload_duration`) and the npu_dsl probe **binary itself** (built Nov 2,
   `kernels_poc/dpu_rate_probe/build/Release/test_dpu_rate_probe.exe`) was run unchanged on the same
   shapes: both give **64 ticks = 46.9%** of 2048 x 2.0454 GHz for fp16 64x64x1600 with 8_8 x=8, **30.7%**
   at x=4, **36.0%** for 32x192 4_16, where npu_dsl's table (Oct, `evidence/e7_21/stencil_table_validated.txt`)
   recorded **100.1% / 77.7%** for the same arms. Operand data (zeros vs random) and the ODU nthw make no
   difference. So the backend is not slower than npu_dsl; **the box's DPU is ~2.1x slower per workload today
   than when that table was taken**, which also explains the VPUNN ratio of ~1.6 and every TOPS percentage
   here. Question for `<user>`: what changed on the box since early October (NPU driver/firmware, Windows
   power plan, DVFS cap, thermal)? Until it is answered, percentages are quoted against the measured
   same-day probe, not against the October table.
   **CLOSED 2026-11-07, by the knowledge base and one environment variable.** (a) `hw/npu5/dpu.md#1,#3.4`:
   the DPU has 4096 MACs (the 8-bit count) and FP16 runs **8 C steps per clock against INT8's 16**, so the dense
   fp16 rate is **2048 MAC/clk per DPU**: the 22.73 / 25.13 TOPS bases were right, and `hw_facts.fp16_peak_tops`
   now derives them. (b) Nothing changed on the box: npu_dsl's table was taken under `NPU_TURBO=YES` (its
   Frequency Validation Rule) and every run of this backend before 2026-11-07 was without it. Under turbo
   (`run_dpu_rate_compare.py`, same method and arms): **64x64x1600 8_8 x=8 = 96.9%** of 2048 x 2.0454 GHz
   (October: 100.1%), 16x192x1600 4_16 = 43.7% (46.0%), 64x64x2048 8_8 x=8 = 96.1%. The ring flow:
   **0.398 ms = 16.19 TOPS = 64.4% of 25.13** (was 0.806 ms NO TURBO). VPUNN recalibrated under turbo:
   IDU / VPUNN@1950 median **0.92** (was 1.58), `evidence/vpunn_calibration.json`.
6. **Two in-flight DMA jobs per port from one SHAVE (2026-11-07, M1).** Through the toolkit's `DmaDrv::start`
   (chained descriptors, one chain per SHAVE per port and direction), starting a second job on a port before
   the previous one completed left the previous job's barrier unproduced (READY stuck at p=1, observed for
   seconds) in the flash phase's two-ring schedule, every run. Serialising this SHAVE's jobs per port fixes it.
   Is queueing a second descriptor chain behind a running one supported by the driver, or is one outstanding
   job per port per SHAVE the contract? The answer decides whether a kernel with several streams needs one
   linked chain per port per fill round (supported by `start(desc, tail, ...)`) or may keep the serialisation.
   **Knowledge base (2026-11-07):** `hw/npu5/dma-controller.md#4`: each link agent holds a **FIFO of 8
   entries** and a linked list completes before the next entry starts, so the hardware queues jobs per LA. The
   lost READY is therefore a property of the toolkit's `DmaDrv40xx` trap/fence chain, not of the DMA; the
   question to the toolkit owner is whether `start` may be called on a chain whose previous job is still in
   flight.
7. **Per-workload latency of a barrier-gated DPU workload (2026-11-07, M1 3c).** With the SHAVE submitting one
   0.2 us workload and waiting for it, the round trip costs 2-3 us: ~0.7 us on the SHAVE (descriptor writes,
   barrier fields, FIFO push, CMX flush) and the rest between the consumer barrier lifting and the IDU starting,
   and between the ODU finishing and the producer barrier decrementing / the watermark landing. What are those
   two hardware latencies on NPU5? They bound how far "submit ahead and never wait per workload" can go, and
   whether tiny workloads (decode, M = 1) should be batched into one DPU task list the way VPUX does.

## 9. Model target: Qwen3.8-35B-A3B-Distill, and what it needs (2026-11-06)

The model is Qwen3.6-35B-A3B's architecture with new weights. The coverage matrix below is npu_dsl's
`docs/design/lib_coverage.md` filtered to the rows Qwen3.6 uses, against this backend. "have" means lowered,
device-verified and measured here.

| Block | Needed | This backend | Milestone |
|---|---|---|---|
| Projections, prefill | int4 group-128: SHAVE dequant -> fp16 DPU; fp16 (`baProj`) | **have** (log 2026-11-07 M2): W4A16 flow in both role assignments, device PASS; 0.82x npu_dsl's tuned probe at its shape, dequant row op 0.26 cyc/value | M2 |
| Projections, decode (M = 1) | SHAVE GEMV over int4 g128 rows; int8 g128 lm_head | **have** (log 2026-11-07 M2): `T.gemm_i4` / `T.gemm_i8` merged-FC DPU path, device PASS, 30 / 41 GB/s of weights (DMA bound ~126) | M2 |
| Norms, embedding, RoPE | RMSNorm rows (fp32 residual), gated RMSNorm, per-head q/k norm, RoPE angles on the SHAVE, embedding rows | `T.vectorized` loops lower to half32/float16 SHAVE vector ops with VAU exp/log/tanh (**have**, log 2026-11-07); the ops themselves not yet written | M1 |
| Full attention | flash prefill (GQA, output gate), flash-decoding, KV cache layout / append | one flash prefill phase (Q resident, K/Vt streamed through two rings, fp32 online softmax) **device golden PASS**, 960x1024x64 vectorised 1.042 ms TURBO; **decode layer have** (log 2026-11-08): qkv GEMV, zero-centred q/k norm, RoPE, cache append, flash-decoding over 1536 positions with GQA 8:1 and the output gate, o_proj, in one launch, device PASS, **0.563 ms** TURBO; prefill GQA / gate / causal mask open | M1 |
| Gated DeltaNet | causal conv1d, delta-rule chunked scan, gated norm | **decode have** (log 2026-11-08 M4): one launch at Qwen3.6's shape, device PASS with both states checked, **0.750 ms** TURBO; the chunked prefill scan open | M4 |
| MoE | top-8 softmax router over 256 experts, routed expert FFN (SwiGLU pairs), shared expert (sigmoid gate), combine | **have** (log 2026-11-08 M3): one launch at Qwen3.6's shape, device PASS (routing == oracle), **0.560 ms** TURBO = 28 GB/s of the 15.8 MB it reads; expert phase ~0.43 ms at 34 GB/s (DMA-latency bound, see the log's cost split) | M3 |
| Head | final norm + lm_head (int8 g128) + greedy argmax | **have** (log 2026-11-08): one launch at Qwen3.6's vocabulary, device PASS, **8.78 ms** TURBO = 58.9 GB/s vs npu_dsl's head 9.44 ms same day; both at <50% of the DMA bound | M5 |
| Compiler | layer table + device-side layer loop with two layer kinds; long-prompt chunk loop; CMX reuse across phases; barrier id reuse; `.text` per kernel | phases + derived `sync_grid`, budgets as sums (**have**); PlanCmx and id reuse open | M5 |
| e2e | pack, export, fp32 reference on the quantised weights, golden, `longseq` | **decode have** (log 2026-11-08): the real checkpoint packed and resident, 40 layers through the session runner, golden 16/16 vs the reference, 57 ms of device time per token (185 ms host wall); prefill forms, `longseq`, the npu_dsl A/B open | M6 |
| Measurement | kernel-only time, traces, flow check, VPUNN-planned workloads, DMA model | **have** | - |

Baseline to beat: npu_dsl's Qwen3.6-35B-A3B numbers (`qwen36_e2e`, its `docs/design/fused_compiler_log.md`),
re-measured on the same day as ours, since the box's DPU rate has changed since October (section 7 question 5).

## 8. Log

* 2026-11-08: **M5 step 2: scoped CMX, and one launch per decoder layer: 5.4 -> 13.7 tok/s by `<user>`'s measure.** (1)
  **Scoped CMX** (the PlanCmx item, as scopes rather than live-range packing): `PlanAndUpdateBufferAllocationLocation`
  sinks allocations into the loops that use them, `ScopeIntelNpuCmx` gives the phases between grid barriers their own
  `lexical_alloc_scope` block and hoists any allocation out of a SHAVE-dependent branch (every SHAVE must run one
  allocation sequence or shared buffers split), codegen brackets each allocating block with `npu::Ctx::CmxScope` (the
  toolkit's `LinearAllocator` is a value: save, restore), `CheckIntelNpuBudgets` charges the peak nesting; kill switch
  `tl.intel_npu.scoped_cmx=False` is the budget negative (`test_intel_npu_scoped_cmx.py`). The four decode kernels
  re-pass on the device unchanged in time. (2) The three block kernels became `T.macro`s that allocate their own CMX
  (`gdn_block`, `attn_block`, `moe_block`; the standalone kernels are thin wrappers, their tests unchanged), plus
  `norm_block` (zero-centred RMSNorm and the fp32 residual adds), and **`make_gdn_layer` / `make_attn_layer`** run a whole
  decoder layer per launch with the residual stream `Hio` fp32 in DDR; both fit the budget only because the phases
  release their CMX. Host sim: the small GDN layer matches the composed oracle with a perturbed-expert negative
  (`test_intel_npu_m5_layer.py`). **Device, whole model: golden 16/16, worst h rel 2.06e-3; host wall 73 ms per token =
  13.7 tok/s** (device launches 55.5 ms: layers 49.2 + head 6.4); the 41 launches still cost ~0.43 ms each of submit +
  protocol, which the device-side layer loop (one launch per token) removes next. 89 passed.
* 2026-11-08: **M5 step 1: same-day per-layer A/B against npu_dsl's decoder-layer kernels** (`device/run_npudsl_layer_device.py`,
  real Qwen3.8 weights, the golden's first token): npu_dsl's GDN layer + MoE **1.165 ms** vs this backend's 0.811 + 0.560 =
  1.37; its attention layer + MoE **1.058 ms** vs 0.563 + 0.560 = 1.12; head 9.44 vs 8.78. So the kernels are within
  ~15% (GDN the one behind), and the whole gap between `<user>`'s measure (generate time / tokens: npu_dsl ~25 tok/s in its
  `--single` mode = one launch per step; this backend 5.4 tok/s = 185 ms wall of which 57 ms device) is the 81 launches per
  token and the host round trips. **Next: the single-launch decode kernel** (norms + fp32 residual in-kernel, layer table,
  device-side layer loop, head), then the GDN gap.
* 2026-11-08: **First end-to-end decode of Qwen3.8-35B-A3B-Distill on the device, golden 16/16** (`testing/python/intel_npu/e2e/`:
  `checkpoint.py` reads the compressed-tensors checkpoint, `pack.py` packs every layer into the kernels' arrays (cached as
  .npy under a pack dir; 8 s per layer the first time), `reference_golden.py` saves the fp32 reference's answer (npu_dsl's
  `qwen36_e2e.reference`, model-side code from `NPU_DSL_ROOT`, run on its own), `decode.py` drives the device). Runtime: `npu_run
  --session` (kernels created once, named resident buffers, launches over stdin; `runner.Session`, `build_all`): the 40 layers'
  17.91 GB upload straight from the cache files in 19 s and stay resident; per token one launch per block (30 GDN + 10 attention
  mixers, 40 MoE, the head) with the residual stream in fp32 and the two RMSNorms per layer on the host (a 4 KB round trip
  each), the prompt run token by token through the decode kernels. **Layer by layer against the reference: worst h rel
  2.4e-3 over 40 layers; 16/16 greedy tokens agree teacher-forced; free generation is coherent.** Per token: device launches
  **57.4 ms** (mixer 32.9 + MoE 17.9 + head 6.3; kernel-only, TURBO) = 17 tok/s of device time, host wall 185 ms (the 81
  launches' protocol and file transfers: what M5's single-kernel decode removes). Two model-side findings: (1) GDN's
  `in_proj_a / in_proj_b` are bf16 in the checkpoint, and int4 g128 would put ~13% noise on the gates, so the kernel now
  takes them as 64 fp16 rows (fp32 lane dot products dealt over the workers, their own DDR scratch so the derived sync sees
  no WAW on Y; 0.811 ms, within the clock-step spread of 0.750); (2) running the fp32 reference in the same process as 18 GB
  of pinned weights swaps a 32 GB box (its expert cache and dense fp32 layer caches reach ~17 GB), so the golden is a saved
  file. **Open for M5/M6:** norms and the residual into the kernels, the layer loop in one launch, the head's DMA pattern,
  prefill kernels (TTFT), the same-day A/B against npu_dsl's Qwen3.6 decode step.
* 2026-11-08: **Decode head closed, and the same-day A/B against npu_dsl's head** (`kernels.make_head`, `device/run_head_device.py`,
  `test_intel_npu_m5_head.py`; `device/run_npudsl_head_device.py`). Qwen3.6's shape: the zero-centred final norm (SHAVE 0 of
  every tile) -> lm_head int8 g128 GEMV over 248320 rows (padded to 248832 = a multiple of BN x tiles x SHAVEs; int8 chunk
  blocks of 128 rows + 2 group-major scale rows, `pack_chunk_blocks_i8`, a local ring of 2 slots per SHAVE, two merged-FC
  workloads per chunk) with the greedy argmax on the fly (half32 lane max per chunk, index scan only when a chunk beats the
  running max, padded columns masked to -65504, each SHAVE's logit rows kept in CMX and stored once as a strided 2-D job)
  -> (max, index) partials in DDR -> `T.sync_grid()` -> worker 0 takes the lowest index among equals. Device: PASS rel
  4.6e-4, token = own argmax = oracle, negative FAIL as required; **8.780 ms** (8.10-8.96, TURBO, UNPROFILED) = 58.9 GB/s of
  517.6 MB. **npu_dsl's `qwen36_head.cpp` on the same problem, same runner, same day: 9.441 ms (9.07-9.63) = 54.7 GB/s**, so
  the bar holds (-7%); both are bound by the same DMA pattern (six unicast issuers, one job in flight per port per SHAVE,
  acquire wait 6.3 of 8.8 ms) at less than half the two-port bound (517 MB / 126 GB/s = 4.1 ms, the "3-4 ms" `<user>`
  remembered). Sweep: 256-row chunks x 1 slot 8.79 ms, 64-row x 4 slots 9.69 ms: the chunk size is not the lever; a
  tile-shared chunk filled by one issuer and read by both SHAVEs (an M5 ring variant) is. Host-sim tests: the small
  instance against the oracle with a perturbed-row negative, and a cross-tile tie (two equal maxima, tiles 0 and 2) that
  must pick the lower index. The npu_dsl A/B probe wraps its generated kernel unchanged (its `kernel_entry` becomes a static
  function; ours stamps NCE_TIME around it and writes this runner's status record), imports nothing, and takes the checkout
  from `NPU_DSL_ROOT`. **The decode path now has every block; next is the host session (one process, weights resident,
  per-token launch sequence over the layer kernels) and the real checkpoint.**
* 2026-11-08: **Attention decode layer closed** (`kernels.make_attn_decode`, `device/run_attn_device.py`,
  `test_intel_npu_m1_attn_decode.py`): Qwen3.6's shape (16 q-heads over 2 kv-heads of 256, RoPE 64, output gate,
  zero-centred norms, int4 g128 qkv [9216, 2048] / o_proj [2048, 4096], a 1536-position cache). qkv GEMV -> the 18
  head items dealt over the workers (norm, RoPE from host cos/sin, K row and V^T column appended at pos) -> flash-
  decoding with every tile streaming its third of the positions through K / Vt rings hoisted over the kv groups, GQA
  rows M = 8, positions past pos masked on the straddling chunk (the cache past pos holds zeros) -> tile merge ->
  head merge over the tiles + gate -> o_proj + residual. Device: PASS (rel 6.3e-3 of the attention contribution),
  cache entries within 2e-3, negative FAIL as required; **0.563 ms** (0.556-0.566, TURBO, UNPROFILED) for 14.2 MB of
  weights + 3.15 MB of cache. Two generic fixes found by it: a one-vector buffer indexed by a loop variable stays in
  CMX (a register's variable-lane extract feeding a splat is unselectable: `LANESPLAT`), and the planner's SHAVE
  split of a resident fill offsets the second-to-last index (a 3-D cache source). Open: a chunk count that follows
  pos (today every chunk of the window streams and the mask does the rest: the long-context lever for M5), decode
  attention cost split. **With this the decode path has every block but the head**: embedding row, final norm,
  lm_head int8 GEMV + argmax, and then the layer loop (M5).
* 2026-11-08: **M4 decode closed: the Gated DeltaNet mixer layer in one launch** (`kernels.make_gdn_decode`,
  `device/run_gdn_device.py`, `test_intel_npu_m4_gdn.py`). Qwen3.6's shape (H 2048, 16 k-heads / 32 v-heads of 128,
  conv k 4, int4 g128 in_proj [12352, 2048] and out_proj [2048, 4096]): in_proj GEMV (rows padded to the tile
  multiple, chunk blocks, local ring) -> `T.sync_grid()` -> causal conv + SiLU per channel block with the conv state
  carried in place -> `T.sync_grid()` -> 6 consecutive v-heads per worker (whole k-heads; one DMA each for the worker's
  q, k, v, z rows), per head the fp32 S [128, 128] in from DDR, l2-norm q / k, gates, the two-pass delta rule, S out,
  gated RMSNorm -> `T.sync_grid()` -> out_proj GEMV + residual. Device: PASS rel 8.7e-4 against the numpy oracle,
  conv state and SSM state checked (7.7e-4 / 5.4e-4), negative (one head's SSM perturbed) FAIL as required; **0.750 ms**
  kernel-only (0.745-0.754 over 10, TURBO, UNPROFILED). `npu_run` re-uploads every input before each launch, so the
  one-step oracle is the golden (the 20-step iteration disagreed, the one-step agreed: verified before adopting).
  **Compiler/runtime capability added, generic:** (1) **one-vector SHAVE-local buffers are vector register variables**
  in codegen (half8/16/32, float4/8/16; whole-vector or single-lane access, no address): moviCompile kept every CMX
  lane accumulator in CMX and reloaded it after each store, `__restrict` on local pointers notwithstanding, so the
  scan spent 60% of its cycles in NOPs; flash's softmax accumulators became registers too (1.130 -> 1.042 ms);
  (2) the merged-FC scratch is sized by the largest `gemm_i4/i8` of the kernel, reserved in `kernel_entry`
  (`ctx.reserve_qg`) and counted by `CheckIntelNpuBudgets` (the budget gate refused the first GDN form by 34 KiB:
  the out_proj ring took one slot per SHAVE); (3) `T.macro` carries the per-head body. **Cost split** (skip arms):
  1.095 -> 0.918 (conv state time-major: the channel-major loop auto-vectorised into per-lane gathers, 1671 cycles
  per 16 channels) -> 0.882 (norms on lanes) -> 0.750 (register accumulators, column-block-outer scan); the scan phase
  is still ~0.25 ms (42 us per head; the pass loops run at ~440 cycles per row pair with 61% NOPs: no software
  pipelining across rows), the GEMV + conv phases ~0.5 ms. Levers left: 4-row unroll / fp16 lanes for the S update /
  a DPU matvec for S^T k and S^T q; the GEMV rings' DMA as in M3. Two device hangs on the way, both root-caused: a
  DDR read past a scratch buffer (the last worker's whole-block row loads; scratch now padded, `gdn_sizes`) hangs the
  device, and the CMX budget must include the merged-FC scratch. Prefill GDN (chunked scan on the DPU, npu_dsl's
  `gated_delta_net_prefill` is the precedent) is open. 79 passed.
* 2026-11-08: **M3 closed: the MoE decode layer in one launch** (`kernels.make_moe_decode`, `device/run_moe_device.py`,
  `test_intel_npu_m3_moe.py`). Qwen3.6's shape (H 2048, 256 experts of inter 512, top-8 softmax renormalised + a
  shared expert gated by sigmoid(router row E), int4 g128): router fp16 GEMM (x padded to 16 rows, 288 padded router
  rows split over the tiles) -> `T.sync_grid()` -> top-k on SHAVE 0 of every tile (redundant, so no second grid
  barrier) -> 9 items dealt 3 per tile, each a gate/up GEMV and a down GEMV through rings whose **fill source reads
  `ids[tile * 3 + j]`** (an indirect, tile-split stream the DPU reads in place) with SwiGLU between -> partials meet
  in DDR -> `T.sync_grid()` -> tile 0 combines into the residual. Device: PASS rel 8.7e-4 against a numpy oracle,
  routing identical, negative (a routed expert's group perturbed) FAIL as required; **0.560 ms** kernel-only
  (0.555-0.562 over 10, TURBO, UNPROFILED) = 28.2 GB/s of the 15.78 MB read. **Compiler capability added, all
  generic:** (1) a ring hoisted over an outer loop may have fills that read the outer variable (the fill ahead
  substitutes `chunk / inner_extent`), several pipelined loops in one outer loop each get a ring, and a fill whose
  source index reads a buffer the iteration writes is not hoisted (test); (2) `T.gemm_i4(..., scales_row=N)`: the
  chunk block carries its group-major scales after the weight rows (one DMA per chunk; row-major and group-major
  `s` buffers still accepted, runtime `sStride`); (3) DMA ports assigned over all live rings by a per-port serial
  job-cost model, and rings of depth >= 2 alternate ports by use (runtime port modes 2/3) because the toolkit's chain
  takes one job per port per SHAVE; (4) **local rings**: a split stream read by the SHAVE that fills it needs no
  hardware barriers (DMA watermark = READY, program order = FREE), so the barrier pool is untouched by decode GEMVs.
  **Cost split (per SHAVE, from the status counters):** 0.950 ms at first -> 0.823 (no per-workload scale transpose:
  the 256x4 scalar transpose of the down GEMV cost 18 us per workload) -> 0.751 (weights not all on port 0) -> 0.729
  (one job per chunk) -> 0.607 (two slots per SHAVE, ports alternating) -> 0.551 (local rings) -> 0.560 final. What
  remains: acquire wait ~130 us, DPU wait ~100 (36 workloads x 2.7 us), fill issue ~85, submit ~37; the intercept
  (router + routing + combine + prologue, from a top-2 run at 0.284 ms) is ~0.12 ms; the expert phase moves 13.9 MiB
  at ~34 GB/s where the DMA probe's six-issuer unicast pattern reaches 75 GB/s at 128 KiB jobs (94 at 256 KiB). The
  `--skip-dpu` arm (0.545 ms) shows the phase is DMA/ring bound, not DPU bound. Levers left for M5: larger chunks
  (CMX reuse across phases would allow 256-row chunks at two slots), the per-fill SHAVE cost (3 us), the router's
  fp16 ring and the scalar top-k (~0.12 ms together). Flash re-measured after the runtime changes: 1.130 ms (1.112
  before, within the launch-set spread); 77 passed. **Filed for later (`<user>`, 2026-11-08: "GDN first"):** (a) move
  the M2 decode GEMV (`make_w4a16_gemv`) onto chunk blocks and re-measure (its row-major scales still cost a SHAVE
  transpose per workload); (b) unify `dpu_matmul` / `dpu_matmul_qg` / `ring_dpu_submit` into one gated, optionally
  asynchronous `dpu_run` with double-buffered merged-FC scratch (M5); (c) the maintainability rules: cost model as the
  acceptance gate (>25% deviation = runtime/model defect), automatic predicted-vs-measured phase split in the runner,
  one-command re-measure of every pinned kernel after a runtime change.
* 2026-11-06: plan written; branch `feat/intel-npu-backend` created from `main` (`da53da96`);
  the untracked 2026-10-29 `tilelang/ptl` PoC and its adapter edits removed; working tree is `main`
  plus this file.
* 2026-11-07: **M2 closed: int8 g128 GEMV (`T.gemm_i8`, the lm_head's format), and the GEMV's variance explained.**
  `dpu_matmul_qg(bits)` shares the merged-FC body: int8 rows K bytes apart, PPE 1/256. Device int8 N 6144 K 2048:
  PASS, **0.308 ms = 41 GB/s** of weights. The 0.22-0.32 ms spread of the int4 GEMV was a **launch-set step**: the
  per-run kernel-only list (now printed by the runner) reads 0.333 x4 then 0.213 x6 -- a clock or power state that
  settles after ~7 launches; with warmup 10 the int4 GEMV is **0.208 ms (0.207-0.209) = 30 GB/s**. Rule for the
  runners: warmup 10, and a quoted number carries its per-run list. The decode GEMV's remaining distance to the
  ~126 GB/s DMA bound is the per-SHAVE serialisation (acquire -> 4 workloads -> release -> store) and six
  concurrent issuers; the lever is a tile-shared chunk filled by one issuer and consumed by both SHAVEs (a ring
  variant), left for the model's profile to justify. M2 coverage: prefill W4A16 both roles, decode int4 and int8;
  74 passed.
* 2026-11-07: **M2 step 2: the decode GEMV through the DPU's int4-weight mode.** New dialect op `T.gemm_i4(a, wq, s,
  out, group, row0, rows)` -> runtime `dpu_matmul_i4g`: one mixed-mode DPU workload (fp16 activations x packed int4
  weights read in place, `wmode = I4`) over the M x ng pseudo-rows against N x ng channels whose weight-table
  offsets point at (row, group), PPE multiplier 1/16, then the SHAVE keeps the diagonal blocks and sums them with the
  scales (VPUX's merged FC; ported from npu_dsl's `fused_rt.hpp` `matmul_qg` / `q_reduce` / `transpose_h` under the
  section 0 amendment). Dequantising first would cost the SHAVE ~10x the weight DMA at M = 1. Also: a resident fill
  of fewer rows than SHAVEs is done by SHAVE 0 (a decode activation row); a pipelined loop needs no ring-submitted
  gemm when its slots have SHAVE-side readers (the synchronous `gemm_i4` is one). **Device** (`--gemv`, N 6144 K 2048,
  256-row chunks): PASS rel 4.5e-4, 0.22-0.32 ms over 5 runs = 20-29 GB/s of weights against a ~126 GB/s DMA bound.
  Per workload: DPU ~2 us, SHAVE transpose + reduce 2.1 us (the scalar transpose was 9 us: `transpose_h` vector
  shuffles), slot wait ~2 us; 6 concurrent unicast issuers of 256 KiB each is also the DMA pattern the Phase 3b probe
  measured lowest. Levers, recorded: scales packed group-major by the model packer (no transpose), 512-row chunks with
  one issuer per tile, the run-to-run variance (0.22-0.32) to be understood before any further number is quoted.
  Tests: `test_intel_npu_m2_w4a16.py` (GEMV lowering, host-sim golden), 73 passed. The negative controls of the W4A16
  runner now perturb a whole group of a row: one nibble met a 0.02 activation and stayed under the bar (a gate seen
  to pass is not evidence; this one was caught by the rule).
* 2026-11-07: **M2 step 1: W4A16 projections on the device, both role assignments.** Source
  `kernels.make_w4a16_flow` (int4 g128, two per byte low nibble first, fp16 scale per group: npu_dsl's
  DEQUANT_I4 / W4A16 format, `pack_int4` the reference packer). Three generic capabilities landed for it:
  (1) **slots the SHAVE reads** -- a streamed buffer no gemm reads gets `ring_slot_acquire` (wait READY, consume)
  before its first SHAVE reader and `ring_slot_release` (produce FREE) + the refill after its last; the readers
  index this SHAVE's slot of the use; nothing reading the slot is refused by name; (2) **split streams** -- a
  streamed fill that reads the tile is SEGMENTED (each tile's SHAVE fills its own slot, unicast; the barrier
  counts scale by the tile count on the shared ids), no longer a refusal: the weights of an N-split projection;
  (3) `#pragma unroll` for `T.unroll` loops, and the runtime library op `tl_npu_dequant_i4_row(dst, src, scales,
  width, group)` (LSU `LD512.i4.i8` convert-load, a 512-value block's 8 loads issued before its stores, after
  npu_dsl's `E7-63` schedule) called through `T.call_extern` -- the same math written as DSL statements
  (`tl_npu_i4x64_lo/hi` or an int8x64 register buffer) compiled to 0.7 cycles per value because moviCompile
  loaded twice or round-tripped the register through CMX; the row op is 0.26 (npu_dsl's tuned body 0.13).
  **Device (TURBO):** 96x2048x2048 split-m 0.414 ms (1.382 with the statement dequant); 96x1536x2048 split-m
  0.319 vs split-n **0.199 ms** (every tile dequantizing everything is the 3x the role rule warns about: for a
  small M the weights must be the split operand); npu_dsl's probe shape 16x3072x1024 **0.197 ms vs its 0.161**
  (0.82x; its dequant body is 2x tighter and its rings are per tile without the shared-id coupling). Depth 2 per
  SHAVE moved the wait from READY to FREE/port, total unchanged. Host-sim goldens for both splits with a
  perturbed-nibble negative (`test_intel_npu_m2_w4a16.py`, 71 passed). Also: `generated/` held 300 sweep
  kernels and every device build compiled the glob (~50 s); sweeps now remove their arms (`runner.remove_kernels`),
  builds are ~21 s. Next for M2: the GEMV (decode, M = 1) on the SHAVE; the dequant row op's inner schedule
  (0.26 -> 0.13) is a known, bounded item.
* 2026-11-07: **M1 step 3c (end): ring hoisting; the split is now SHAVE math 57%, per-Q-block overhead 26%,
  ring protocol 17%.** `LowerIntelNpuPipelined` hoists a `T.Pipelined` loop out of an enclosing serial loop
  when no stream's fill reads the outer variable and the inner chunk count is a multiple of the SHAVEs: one
  ring over outer x inner chunks, the use loop in the inner loop's place with global use `outer * per + u`, the
  fill source wrapped modulo the inner extent; a fill that reads the outer variable keeps one ring per iteration
  (test with the negative). Found on the device: the toolkit's `DmaDrv::start` re-kicks a port's chain once its
  fences ran, before the previous job completed, so a synchronous `dma_load` issued while a ring fill was in
  flight on the port lost that fill's READY (status 9 at use 0); `dma_load/store` now retire the ports first.
  **960x1024x64 TURBO: 1.112 ms** (3c started at 1.765), floor 0.478. Remaining: SHAVE math ~0.63 ms (the row loop
  364 static cycles for 128 scores, 57% NOP: one row's dependency chain at a time; unroll-by-2 or two rows in
  flight is the lever), per-Q-block ~30 us (Q fill with two tile syncs, merge on one SHAVE, output store),
  protocol ~0.18 ms. Stopping the single-kernel work here (`<user>`: model first); the levers are recorded, the
  counters stay in the status record for every later kernel.
* 2026-11-07: **M1 step 3c (continued): where the flash phase's time goes, and four generic fixes.** Instrument
  first: the ring path now accumulates six per-SHAVE phase counters (live-wait, order-wait, submit, dpu-wait,
  free-wait, fill-issue; NCE_TIME ticks) into the status record's extra words, and `run_flash_device.py --floor`
  runs the phase with no SHAVE math (output wrong by design) to expose the protocol floor. Findings on
  960x1024x64 TURBO: floor 0.675 ms of 1.327; per workload the **submit cost 1.9 us** because the two gemm
  shapes (QK^T 32x128x64, PV 32x64x128) alternate and the one DPU descriptor set was re-initialised every time
  (the single-shape flow: 0.66 us); **fill issue 1.5 us** because both rings split every fill over both ports and
  the one-in-flight-per-port rule serialised K behind V; the fill's fixed latency was NOT on the critical path
  (ring depth 4 -- two slots per SHAVE, fills two uses ahead, now supported: `num_stages` a multiple of the
  SHAVEs, ids 4 x slots, `ring_arm` at fill-or-submit -- moved nothing). Fixes, all generic: (1) `Ctx` keeps
  **two DPU descriptor sets, LRU by shape** (budget reserve 32 -> 72 KiB); (2) with several streams in one loop
  **each stream's fill is one descriptor on its own port** (`ring_begin(..., port)`, -1 = halves on both for a
  single stream), so K and V fills run concurrently; (3) `__restrict` on allocations and `tl_npu_vmax/vsum`
  lane reductions (above); (4) the per-Q-block init, partial copy and merge vectorised in the source (`part` row
  stride padded to a vector multiple). **Result: 1.181 ms, floor 0.536 ms**; ring phases are now 0.13-0.2 ms of
  the 1.18. What remains: SHAVE math ~0.55 ms (static count; the row loop is still 57% NOP) and **~35 us per
  Q block outside the ring**: two `ring_begin` grid rendezvous, the Q fill with two tile syncs, the merge. Next:
  hoist a ring whose fills do not reference the enclosing serial loop's variable out of that loop (one
  begin/end per kernel instead of per Q block), then submit QK^T(u+1) before softmax(u) (multi-gemm
  submit-ahead). Hardware question 7 asks for the DPU gate/complete latencies this protocol pays per workload.
* 2026-11-07: **Turbo, and hardware facts with provenance.** (1) `<user>` pointed at the VPUX knowledge base
  (`applications.ai.vpu-accelerators.vpux-plugin/knowledges`): it already held what this plan had been
  measuring and asking for: the NPU5010 CMX workspace of 1439 KiB (`compiler/platform-resources.md#3`; equals the
  1,473,536 B `Ctx` sees), FP16 at 8 C steps per clock = 2048 MAC/clk per DPU (`hw/npu5/dpu.md#3.4`), the
  512-bit SHAVE VRF (`hw/npu5/shave.md#2.1`), the DMA's 64 B/clk per group and 8-entry link-agent FIFOs
  (`hw/npu5/dma-controller.md#1,#4`), 16 barriers per tile (`hw/npu5/nce-subsystem.md#4.2`). The facts the
  backend depends on are now `tilelang/intel_npu/hw_facts.py` (value, `file#section`, measured value) checked by
  `test_intel_npu_hw_facts.py` (every citation resolves to a file and section heading; the target defaults equal
  the facts; the fp16 peak is derived), the device scripts take their peak basis from it, and the
  `intel-npu-hardware` skill (`.agents/skills/intel-npu-hardware/SKILL.md`) makes the lookup the first step;
  section 5 carries the rule. (2) **Every number before today was NO TURBO.** `NPU_TURBO=YES` is npu_dsl's
  measurement condition and the runner's default now (`runner.turbo_requested`, `TL_NPU_TURBO=0` for an A/B;
  every result line says TURBO / NO TURBO). Under turbo: ring flow **0.398 ms = 16.19 TOPS = 64.4%** of the
  25.13 dense fp16 peak (was 35%); flash 960x1024x64 vectorised **1.765 ms** (was 3.588; npu_dsl 20.3 ms under
  turbo -> 11.5x per op); DPU probe 64x64x1600 8_8 x=8 **96.9%** = npu_dsl's October 100.1%, which closes
  question 5; VPUNN ratio **0.92** (full recalibration, `evidence/vpunn_calibration.json`). (3)
  `run_dpu_rate_compare.py` keeps the shapes whose operands fit the 1439 KiB workspace over two SHAVE slots
  (64x192 / 32x192 at K=1600 and 32x256 at K=2048 do not: status 2 CMX_EXHAUSTED); why the 2026-11-06 run of
  the same script reported a 32x192 row is not established (the script, shapes and `Ctx` allocations are
  unchanged in git; `Ctx` reads 1,473,536 B free today), so that row is not quoted again.
* 2026-11-07: **M1 step 3b: same-shape A/B against npu_dsl's flash attention.** `make_flash_attention_phase`
  takes any `numQ` that is a multiple of `BQ x tiles`: Q blocks are dealt round-robin over the tiles in a serial
  loop around the phase (the resident Q fill becomes "SEGMENTED resident in a loop, tile sync before and after";
  the two rings are re-begun per block on the same ids). **Sq 960 x Skv 1024 x d 64, vectorised, 3 x 2: 3.588 ms
  kernel-only = 0.070 TOPS on the `4 Sq Skv d` basis (0.28% of 25.13 turbo)**, PASS rel 6.19e-4. npu_dsl's record
  for its 32x8 class (Sq = Skv = 1024, same basis, `docs/RUN_AND_RESULTS.md` section 2, Round 1195, UNPROFILED
  NCE_TIME): walker 20.38 ms / legacy 20.32 ms = 0.0132 TOPS. **So this backend's flash phase is 5.3x faster per op
  than npu_dsl's, with the DPU at 1% on both sides**: the difference is the K/V hardware rings (npu_dsl's W3, not
  built) and the vector softmax. Both are still SHAVE-bound by a wide margin: 3.588 ms over 40 chunk steps per
  SHAVE is ~90 us per 32 x 128 score block (~22 ns per score), far from 32-lane throughput; the next step is a
  SHAVE-side cost breakdown (`ScopedVecCompute` brackets or the static cycle count of the `.s`) before touching
  the source. Caveat for the comparison: npu_dsl's figures are its October record on a box whose DPU is now ~2.1x
  slower (section 7 question 5); the DPU is ~1% of either kernel, so the SHAVE-bound comparison stands, but a
  same-day re-run of its harness would make it exact.
* 2026-11-07: **M1 step 3a: SHAVE vector codegen; the flash phase's softmax is 9.4x faster from `T.vectorized`
  alone.** Generic path, no per-kernel code: `VectorizeLoop` joins the pipeline after `ConfigIndexBitwidth`;
  `CodeGenTileLangIntelNpu` prints moviCompile's ext_vector types (`half8/16/32`, `float4/8/16`, `int4/8/16`,
  ...), splats a `Broadcast` as `((halfN)(x))`, lowers a vector `Cast` to `__builtin_convertvector` (a C cast
  between same-size vectors is a bitcast), and CodeGenC's own `*(halfN*)(p + i)` loads/stores and infix
  arithmetic do the rest. `intrin_rule_intel_npu.cc` maps vector fp16 math to the `tl_npu_h<name>` overloads
  and vector fp32 math to `tl_npu_<name>f` lane loops (`npu_rt_vecmath.hpp`, shared by device and host sim);
  the device defines `exp/log/tanh` on `half32` with `VAU.EXP/LOG/TANH` (the v8f16 forms and `SAU.SUMX.f32` are
  **not on this SHAVE variant**: `moviCompile -mcpu=5000xx` refuses them; `half16`/`half8` widen to one half32
  op). Probed first (`D:\\tmp\\vecprobe.cpp`): splat casts, `__builtin_convertvector`, `__builtin_elementwise_max`,
  vector `?:`, `v[i]`/`v.s3` all compile. **Source**: `make_flash_attention_phase(vectorized=True)` writes the
  online softmax with `T.vectorized(32)` over fp16 (row max into a 32-lane accumulator, `exp(s*scale - m)` on the
  VAU) and `T.vectorized(16)` over fp32 (row sums, O update); reductions finish with a scalar lane loop. **Device**:
  PASS rel 6.23e-4, negative FAIL, **0.397 ms vs 3.754 ms** scalar; DPU still 1% busy, so the SHAVE is still the
  bound (next: register-resident accumulators, fewer scalar lane reductions, and the kernel-only A/B against
  npu_dsl's flash at the same shape). Host sim of the vectorised serial phase passes at the same tolerance
  (`test_intel_npu_m1_flash.py`, 61 passed).
* 2026-11-07: **M1 step 2a: barrier id reuse across phases, device-verified.** `AssignIntelNpuBarriers`
  colours rings by live range (`ring_begin`..`ring_end` in program order, first fit): rings one after another
  share one block, rings live at once (the flash phase's two streams) never do; the note says `N rings, P live at
  once, I ids by reuse`; the refusal counts live rings (`test_barrier_pool_exhaustion_is_refused`: three
  sequential rings now fit in 8 ids, three operands streamed through one loop at 2 tiles are refused `3 live at
  once, need 24 .. short by 3`). Soundness: `ring_end` leaves the issuer's slot ids quiescent (last FREE produced
  by every DPU and consumed; READYs consumed before that) and `ring_begin` re-arms on the same SHAVE behind a
  grid rendezvous. **Device** (`run_ring_flow_device.py --two-phase`, `kernels.make_ring_flow_two_phase`): both
  phases PASS on ids 8..15, 1.563 ms for two flows = 2 x the single flow, so a phase boundary costs nothing;
  `--no-reuse` (`tl.intel_npu.barrier_reuse=False`) is the A/B arm. **Found on the way, both in the runtime:**
  (a) the per-port retire added in step 1 ran after `initDesc` of the same descriptor, so a single ring's next
  fill waited on a watermark it had just cleared -- every single-ring kernel hung since step 1 (the flash phase,
  with two rings alternating descriptors, could not see it); retire now precedes `initDesc`, and the single flow
  (0.804 ms) and the flash phase (3.710 ms) re-pass. (b) The toolkit's `DmaDrvBase::wait` is unbounded; the ring
  paths now use `dma_wait` with the barrier bound and status 10 `DMA_TIMEOUT` (detail says which wait), which is
  how (a) became a status instead of a 120 s host timeout. Standing rule from the two: **no toolkit wait without a
  bound on the kernel's control path** -- `mm_->wait()` and `dma_.wait()` remain only in the synchronous
  Phase 1-2 paths (`dpu_matmul`, `dma_load/store`).
* 2026-11-07: **M1 step 1: the flash-attention phase lowers through the generalised ring and passes its
  device golden.** Source `kernels.make_flash_attention_phase` (Q resident, K and Vt streamed in one
  `T.Pipelined` loop, two `T.gemm`, online softmax in fp32 scalar loops between them, merge of the two SHAVEs'
  partials after `T.sync_threads`). Compiler: `LowerIntelNpuPipelined` grew a general body (any number of
  streams and gemms, author order kept: gemm reading a slot -> submit, wait, that stream's next fill; no
  submit-ahead when SHAVE statements sit between gemms) next to the simple submit-ahead body; a stream whose
  DDR chunk is strided (Vt columns) is a **2-D multicast fill** (`ring_fill(ring, use, src, width, rows,
  src_stride)`, the halves split the rows); a resident SEGMENTED fill inside a loop gets a tile sync before
  and after. Two rings over the same loop get disjoint ids (8..15, 16..23). Math intrinsics lower via
  `intrin_rule_intel_npu.cc` (fp16 `exp` -> `tl_npu_hexp`, SAU). **Device** (`device/run_flash_device.py`,
  Q 96x64, KV 1024, 3 x 2): PASS rel 5.73e-4 (equal to the host sim of the serial phase), negative control
  (K perturbed) FAIL, trace `evidence/tl_flash_phase_q96_kv1024_d64_trace.json`. Kernel-only **3.754 ms** of
  which the DPU is 3 us: the scalar softmax is the whole time, which is the expected state before the SHAVE
  ops are vectorised (M1 step 3); no TOPS claim. **Two defects found by the device, fixed in the runtime:**
  (a) `ring_dpu_wait` used the driver's unbounded `wait()`, so a stuck gate hung the launch (120 s host
  timeout, no status); it now spins on the same `odu_workload_id` watermark with the barrier bound and reports
  status 9 `DPU_TIMEOUT` with the READY/FREE counters in the detail. (b) With two rings, a SHAVE issued a
  second fill on a port while its previous fill (the other ring's) was still running; one half of the earlier
  fill never produced READY (counters p=1 on READY 10 after seconds) and the in-order DPU FIFO stalled behind
  it. The ring now retires this SHAVE's previous descriptor on a port before starting another
  (`lastOnPort_`), which costs nothing in this schedule (the previous fill has long landed) and the golden
  passes. Hardware question 6 records it. Tests: `test_intel_npu_m1_flash.py` (two rings over one loop, 2-D
  fill, general body, serial phase host-sim golden with a negative) -- 59 passed.
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

  > Amendment 2026-11-06, same day: the "~22 GB/s ceiling" in (4) is a 128 KiB-job artifact. The job-size
  > sweep (section 7 question 4) shows ~4.5 us fixed per job plus ~67 GB/s streaming on two ports (~34 per
  > port), reproducing the npu_dsl record's 51.6 GB/s at 512 KiB. TileGemm's constant is
  > `4.5 us + bytes / 67 GB/s`, and the lever is larger fills, not more ports.
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
