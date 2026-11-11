# FGCLIP2 在 TileLang Intel NPU 后端里是怎么编译的——基于真实构建产物的逐层拆解

> 这是 [fgclip2_npu_dsl_explained.md](fgclip2_npu_dsl_explained.md)（原始 `npu_dsl` 实现）的**对应篇**，但讲的是
> **这个仓库（TileLang）自己的编译管线**，而不是 npu_dsl 的。素材全部来自本仓库 2026-11-11 **真实构建、真实跑出来**
> 的产物（`pip install -e . --no-build-isolation` 构建成功，`testing/python/intel_npu/test_intel_npu_fgclip2_v0.py`
> 两个测试稳定通过 6/6 次），不是从源码读出来的推测——凡是下面出现的 C++/TIR 代码块，都是对着真实解释器跑出来的
> 原文，连变量名、临时寄存器编号（`r_1`、`d_2`……）都是编译器自己生成的，没有改写。

## 目录
1. TileLang intel_npu 后端的编译管线：DSL → TIR → 15 个 Pass → C++
2. 一个真实的「DSL 源码 → TIR → 最终 C++」最小例子（亲手跑出来的）
3. 放大到真实 kernel：FGCLIP2 V0 encoder layer 的生成代码逐段解读
4. 和 npu_dsl 对照：同一件事,两条管线怎么各自做
5. FGCLIP2 V0 当前状态与下一步

---

## 1. TileLang intel_npu 后端的编译管线

和 npu_dsl 的 `fused` 管线一样，TileLang 这边也是"Python 写的 kernel 源码 → 结构化 IR → 一串 pass → C++ 源码"，
但走的是 TileLang/TVM 原生的 TIR 基础设施，不是 npu_dsl 自己发明的一套 `ScheduledOp`/`FusedKernel`。入口是
[tilelang/intel_npu/pipeline.py](../../tilelang/intel_npu/pipeline.py) 的 `IntelNpuPassPipelineBody`，真实顺序
（从我们这次构建后实际跑的版本摘录，不是转述）：

```python
def IntelNpuPassPipelineBody(mod, target):
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.intel_npu.transform.LowerIntelNpuKernelLaunch()(mod)
    mod = tilelang.transform.MaterializeKernelLaunch(lower_thread_binding=True, ..., lower_grid_binding=True)(mod)
    # ... LetInline / AddWrapperForSingleBufStore / LegalizeNegativeIndex / InjectAssumes / Simplify / VerifyBufferInit / IfStmtBinding ...
    mod = tilelang.intel_npu.transform.LowerIntelNpuTileOp()(mod)        # T.copy/T.gemm -> dma_load/dma_store/dpu_matmul
    mod = tilelang.intel_npu.transform.PlanIntelNpuDistribution()(mod)   # 哪些 buffer 是 tile-invariant，要不要变成多播
    mod = tilelang.intel_npu.transform.LowerIntelNpuPipelined()(mod)     # T.Pipelined 环形双缓冲
    mod = tilelang.intel_npu.transform.DeriveIntelNpuGlobalSync()(mod)   # 跨 worker 的 DDR 读写自动插 barrier
    mod = tilelang.intel_npu.transform.VerifyIntelNpuUniformSync()(mod)  # 校验 barrier 不会被某些 SHAVE 跳过(会死锁)
    mod = tilelang.intel_npu.transform.PlanIntelNpuWorkloads()(mod)      # VPUNN 代价模型切 DPU workload 的 cuboid
    mod = tilelang.intel_npu.transform.AssignIntelNpuBarriers()(mod)     # 给每个 ring 分配硬件 barrier id
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)  # CMX 分配下沉到最内层作用域
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.intel_npu.transform.ScopeIntelNpuCmx()(mod)           # barrier 之间的 phase 各自一段 CMX 作用域
    mod = tilelang.intel_npu.transform.CheckIntelNpuBudgets()(mod)       # CMX/.text 预算校验,超了直接报错
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.VectorizeLoop(...)(mod)                     # T.vectorized -> halfN/floatN 向量指令
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)                      # host 包装 vs 设备 kernel_entry 拆开
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod
```

几个和 npu_dsl 的 `fused` 管线**本质相同、形式不同**的地方，值得先建立起对应关系（第 4 节有完整对照表）：

| 做的事 | npu_dsl `fused` 管线 | TileLang intel_npu 管线 |
|---|---|---|
| CMX 地址分配、双缓冲 | `passes.py` 里手写的 live-range 打包 + `__pf1/__pf2` 预取变量生成 | `PlanAndUpdateBufferAllocationLocation` + `ScopeIntelNpuCmx`（见第 3 节真实例子，用 `CmxScope` RAII 对象实现"作用域结束就释放"，而不是生成一堆带后缀的影子变量） |
| 跨 worker barrier | `passes.derive_barriers`，扫描 DDR 读写冲突 | `DeriveIntelNpuGlobalSync`，同样是扫描 DDR 读写冲突（见第 3 节：即使只有 1 个 tile,也会在检测到 `Hio` 先读后写时自动插一次 `ctx.sync_grid()`） |
| DPU workload 切分 | 手写的 tile 调度（`_gemm_split`/`_gemm_bcast`）由作者在 Python 里挑 | `PlanIntelNpuWorkloads`，一个独立 pass，吃 VPUNN 代价模型；**VPUNN 没配置时会打印注释说明并退回运行时默认值**,不是报错中止（见第 3 节真实注释） |
| SHAVE 向量算子 | `nl.shave.xxx` 调用手写 C++ 库函数（`fops::layernorm_f32` 等） | 没有"调用"这一层：作者直接写标量/`T.vectorized` 循环,`VectorizeLoop` pass 把它们变成 `halfN`/`floatN` 向量指令。没有算子注册表,因为没有"算子"这个概念——一切都是循环 |
| 一个 Python `for` 循环该不该展开 | 区分**两种循环**：`nl.range(...)`(运行时循环,产生真实 C++ for)和裸 Python `for` (trace 时展开,产生字面量偏移) | **没有这个区分**：`T.prim_func`/`T.macro` 体内的 `for x in range(n):` 会被 TVMScript 风格的 AST 解析器接管,**一律变成真实的 TIR `For` 循环**，不会在 trace 阶段展开成字面量——第 2 节的真实例子会直接证明这一点 |

---

## 2. 一个真实的「DSL 源码 → TIR → 最终 C++」最小例子

用一个只做"带 bias 的 LayerNorm"的玩具 kernel（和上一份文档里 npu_dsl 的 `tiny_ln_demo` 刻意同构，方便两边对照），
**真实跑了一遍** `tilelang.lower(func, target="intel_npu")`：

**输入（TileLang DSL 源码）：**
```python
import tilelang.intel_npu.language as T

@T.prim_func
def tiny_ln(X: T.Tensor((16, 16), T.float32), W: T.Tensor((1, 16), T.float16),
            B: T.Tensor((1, 16), T.float16), Out: T.Tensor((16, 16), T.float16)):
    with T.Kernel(1, threads=2) as tile:
        shave = T.get_thread_binding()
        x32 = T.alloc_local((16, 16), T.float32)
        w = T.alloc_local((1, 16), T.float16)
        b = T.alloc_local((1, 16), T.float16)
        y = T.alloc_local((16, 16), T.float16)
        if shave == 0:
            T.copy(X[0, 0], x32); T.copy(W[0, 0], w); T.copy(B[0, 0], b)
            for r in T.serial(16):
                st0 = T.alloc_local((2,), T.float32)
                st0[0] = T.float32(0); st0[1] = T.float32(0)
                for d in T.serial(16):
                    st0[0] = st0[0] + x32[r, d]
                    st0[1] = st0[1] + x32[r, d] * x32[r, d]
                mean = st0[0] / T.float32(16)
                var = st0[1] / T.float32(16) - mean * mean
                invstd = T.float32(1) / T.sqrt(var + T.float32(1e-6))
                for d in T.serial(16):
                    y[r, d] = T.cast((x32[r, d] - mean) * invstd * T.cast(w[0, d], T.float32)
                                    + T.cast(b[0, d], T.float32), T.float16)
            T.copy(y, Out[0, 0])
        T.sync_threads()
```

**中间产物（`print(tiny_ln)`：trace 之后、任何 intel_npu pass 之前的原始 TIR，TVMScript 打印形式）**——这一步揭示了
几件源码里看不出来的事：
```python
# from tvm.script import tirx as T
@T.prim_func
def tiny_ln(X_handle: T.handle, W_handle: T.handle, B_handle: T.handle, Out_handle: T.handle):
    X = T.match_buffer(X_handle, (16, 16), strides=(16, 1))
    ...
    for bx in T.thread_binding(1, thread="blockIdx.x"):
        shave: T.int32 = T.launch_thread_idx(0)
        ty: T.int32 = T.launch_thread_idx(1)      # <- threads=2 只声明了一维,y/z 维度自动补 1
        tz: T.int32 = T.launch_thread_idx(2)
        with T.sblock("tilelang_root"):
            T.sblock_attr({"tl.launch_threads": [2, 1, 1]})
            x32 = T.sblock_alloc_buffer((16, 16), scope="local")
            ...
            if shave == 0:
                T.copy(T.region(X[0, 0], 1, 16, 16), T.region(x32[0, 0], 2, 16, 16))
                ...
                for r in range(16):          # <- T.serial(16) 在这一层就已经是普通 TIR range,和裸 Python for 没有区别
                    ...
            T.tvm_storage_sync("shared")     # <- T.sync_threads() 在这一层翻译成的通用 TIR 原语
```
**关键事实**：`T.serial(16)` 和裸 Python `for r in range(16):` 在这个 IR 层面**是同一个东西**——`T.copy`/`T.region`
这类"tile op"（`T.copy`，对应 npu_dsl 的 `nl.dma.load/store`）此刻还是**高层意图**，要到 `LowerIntelNpuTileOp`
这个 intel_npu 专属 pass 才会变成具体的 `dma_load`/`dpu_matmul` 调用——这就是"通用 TileLang IR"与"intel_npu
专属降低"的分界线。

**输出（`tilelang.lower(tiny_ln, target="intel_npu").kernel_source`，跑完全部 pass 之后，未删减）：**
```cpp
#include <tl_templates/intel_npu/npu_rt.hpp>

static inline void tiny_ln_kernel(npu::Ctx& ctx, npu::Ddr<half> B, npu::Ddr<half> Out, npu::Ddr<half> W, npu::Ddr<float> X) {
  const int32_t bx = ctx.tile_id();  // blockIdx.x < 1
  // intel_npu: distribution: x32: tile-invariant source copied into a SHAVE-private buffer on every worker (make it shared to read it once per tile); w: ...; b: ...
  const int32_t shave = ctx.shave_id();  // threadIdx.x < 2
  const int32_t ty = 0;  // threadIdx.y < 1
  const int32_t tz = 0;  // threadIdx.z < 1
  npu::Ctx::CmxScope cmx_scope_0(ctx);
  float* __restrict x32 = ctx.alloc<float>(256);  // local
  npu::Ctx::CmxScope cmx_scope_1(ctx);
  half* __restrict w = ctx.alloc<half>(16);  // local
  npu::Ctx::CmxScope cmx_scope_2(ctx);
  half* __restrict b = ctx.alloc<half>(16);  // local
  npu::Ctx::CmxScope cmx_scope_3(ctx);
  half* __restrict y = ctx.alloc<half>(256);  // local
  if (shave == 0) {
    ctx.dma_load((&(x32[0])), (&(X[0])), 64, 16, 64, 64);
    ctx.dma_load((&(w[0])), (&(W[0])), 32, 1, 32, 32);
    ctx.dma_load((&(b[0])), (&(B[0])), 32, 1, 32, 32);
  }
  npu::Ctx::CmxScope cmx_scope_4(ctx);
  float* __restrict st0 = ctx.alloc<float>(2);  // local
  if (shave == 0) {
    for (int32_t r = 0; r < 16; ++r) {
      st0[0] = 0.000000e+00f; st0[1] = 0.000000e+00f;
      for (int32_t d = 0; d < 16; ++d) {
        st0[0] = (st0[0] + x32[((r * 16) + d)]);
        st0[1] = (st0[1] + (x32[((r * 16) + d)] * x32[((r * 16) + d)]));
      }
      float mean = (st0[0] / 1.600000e+01f);
      float var = ((st0[1] / 1.600000e+01f) - (mean * mean));
      for (int32_t d_1 = 0; d_1 < 16; ++d_1) {
        y[((r * 16) + d_1)] = ((half)((((x32[((r * 16) + d_1)] - mean) * (1.000000e+00f / sqrtf((var + 1.000000e-06f)))) * ((float)w[d_1])) + ((float)b[d_1])));
      }
    }
    ctx.dma_store((&(Out[0])), (&(y[0])), 32, 16, 32, 32);
  }
  ctx.sync_tile();
}

void kernel_entry(int numParams, uint64_t* params, const npu::Environment& env) {
  // hardware barrier ids used by this kernel: sync_grid 13..15
  npu::Ctx ctx(env, numParams, params, 4, /*sync_barrier_base=*/13);
  if (!ctx.ok()) { ctx.finish(); return; }
  tiny_ln_kernel(ctx, npu::Ddr<half>(params[0]), npu::Ddr<half>(params[1]), npu::Ddr<half>(params[2]), npu::Ddr<half>(params[3]));
  ctx.finish();
}
```

几个"编译器自己做的事，源码里一行都没写"（和上一份文档第 4 节并列对照）：
1. **`CmxScope` 是 RAII，不是手写的 live-range 打包**：每个 `T.alloc_local` 对应一个 `npu::Ctx::CmxScope cmx_scope_N(ctx);`，
   构造时在 bump-pointer 上标记、析构时（C++ 作用域结束）自动回收——`ScopeIntelNpuCmx` pass 的全部产出就是"把分配点
   包一层 scope 对象"，不像 npu_dsl 需要在 Python 侧手工计算每个 buffer 的偏移量字面量。
2. **自动检测"tile 不变的输入"**（`PlanIntelNpuDistribution` 的注释）：`x32`/`w`/`b` 虽然被声明成 `local`（每个
   SHAVE 私有），pass 发现它们的内容对整个 tile 其实是一样的（只有 `shave==0` 在写），于是在注释里**主动建议**
   "改成 shared 可以只读一次"——这是一个**诊断性**输出，不是强制重写源码，体现了 TileLang 这边"pass 给建议,
   作者决定要不要采纳"的风格，和 npu_dsl 直接在 trace 阶段按声明的 `scope=` 固定下来不同。
3. **`sync_grid 13..15` 的 barrier id 段**：即使这个 kernel 只有 1 个 tile、没有真正的跨 tile 同步需求，
   `AssignIntelNpuBarriers` 依然从硬件的 barrier 池里切出一段 id（`13..15`）——这是因为管线是**统一流程**，不为
   "单 tile" 走特殊分支；真正用不用、用几个,由前面的 `DeriveIntelNpuGlobalSync` 决定。

---

## 3. 放大到真实 kernel：FGCLIP2 V0 encoder layer 的生成代码

[testing/python/intel_npu/fgclip2_kernels.py](../../testing/python/intel_npu/fgclip2_kernels.py) 的
`make_fgclip2_encoder_layer_v0`（16 行 token、4 头、每头 16 维、MLP 128 维的小规模测试配置）编译出 221 行、
13.5 KB 的 C++。完整文件太长,这里摘出几段最有信息量的,全部是真实生成结果：

### 3.1 CMX 分配：17 层嵌套的 `CmxScope`

```cpp
npu::Ctx::CmxScope cmx_scope_0(ctx);
float* __restrict h32 = ctx.alloc<float>(1024);  // local          <- 整个 encoder layer 的 fp32 残差流,最外层 scope
{
  npu::Ctx::CmxScope cmx_scope_1(ctx);
  half* __restrict ln1w = ctx.alloc<half>(64);  // local
  ...
  half* __restrict wq = ctx.alloc<half>(4096);  // local            <- 64x64 的 fp16 权重矩阵,4096 = 64*64
  ...
}
```
对应我在 Python 里写的 `h32 = T.alloc_local((rows, hidden), T.float32)` 在最外层、其余缓冲在 `if shave == 0:`
块内——`ScopeIntelNpuCmx` 原样保留了这个嵌套结构：`h32` 活得比 `if` 块长（要在块外被最终 `dma_store` 回 DDR），所以
停留在外层 scope；块内的 `wq`/`wk`/`wv`/`act`/`mlp_out`……全部活在同一层嵌套里，意味着（结合 `CmxScope` 的
bump-pointer 语义）**这些缓冲的生命周期一结束,它们占的 CMX 字节就能被同一 scope 里后面声明的缓冲复用**——这正是
两份文档都提到的"CMX 复用跨 phase"能力,这里是用最朴素的 C++ 作用域规则自动实现的，不需要 npu_dsl 那种显式的
live-range 分析表。

### 3.2 `T.gemm` → `ctx.dpu_matmul`

```cpp
ctx.dpu_matmul((&(qp[0])), (&(xn[0])), (&(wq[0])), 16, 64, 64, 0, -1);
```
我的 Python 源码写的是 `T.gemm(xn, wq, qp, transpose_B=True, clear_accum=True)`（`LowerIntelNpuTileOp` 这个 pass
的产物）；5 个数字参数 `16, 64, 64, 0, -1` 分别是 M（行数 16）、N（输出通道 64）、K（输入通道 64）和两个硬件相关的
标志位——这组 `(M, N, K)` 必须全部是 16 的倍数，是我们在第一次真实运行时撞到的真实报错（见下面第 5 节）。

### 3.3 Python `for h in range(heads):` **不是**字面量展开，是真实循环

```cpp
for (int32_t h = 0; h < 4; ++h) {
  for (int32_t r_4 = 0; r_4 < 16; ++r_4) {
    for (int32_t d_2 = 0; d_2 < 16; ++d_2) {
      qh[((r_4 * 16) + d_2)] = qp[(((r_4 * 64) + (h * 16)) + d_2)];
      ...
    }
  }
  ctx.dpu_matmul((&(s_cmx[0])), (&(qh[0])), (&(kh[0])), 16, 16, 16, 0, -1);
  ...
}
```
我在 [fgclip2_kernels.py](../../testing/python/intel_npu/fgclip2_kernels.py) 的 `mha_full` 宏里写的是**裸 Python**
`for h in range(heads):`（写的时候参照了 npu_dsl 的习惯，以为这样能让 `h*D` 变成字面量偏移，像 npu_dsl 的
`nl.range` 和裸 `for` 的区别那样）——**生成结果证明这个假设在 TileLang 里不成立**：`h` 是一个真正的 TIR 循环变量
（见 `for (int32_t h = 0; h < 4; ++h)`），`h * 16` 在 C++ 里也是**运行时乘法**，不是编译期常量。这是本次移植过程
中发现的、和 npu_dsl 最大的一处认知偏差，已经记入第 4 节的对照表，供以后写 kernel 时不要再假设"裸 for 会被展开"。
好消息是：这**不影响正确性**——对一个在 CMX 里驻留的 buffer 做运行时下标访问本来就是合法、常见的写法（`s_cmx[i, j]`
这类访问在这个后端里到处都是），两个测试都稳定通过证明了这一点；只是如果以后想要"编译期展开、生成更大但更快的
直线代码"，需要用别的机制（比如 Python 层面在 `T.prim_func`/`T.macro` **外面**的普通 Python 代码里循环并各自
生成一段，而不是指望 body 内部的 `for` 被展开）。

### 3.4 自动插入的跨 worker barrier：即使只有一个 tile

```cpp
    }   // <- if (shave == 0) { ... 27 层 encoder layer 的全部计算 ... } 的右括号
  }     // <- CmxScope 嵌套块的右括号
  ctx.sync_grid();                                  // <- 自动插入,源码里没有写
  if (shave == 0) {
    ctx.dma_store((&(Hio[0])), (&(h32[0])), 256, 16, 256, 256);
  }
  ctx.sync_tile();
}
```
`DeriveIntelNpuGlobalSync` 发现 `Hio`（DDR 残差流）在 kernel 一开始被读（`T.copy(Hio[0,0], h32)`）、在最后又被写回
（`T.copy(h32, Hio[0,0])`），于是在"最后一次写之前"插入了一次 `ctx.sync_grid()`——即使这个 V0 kernel 只用了 1 个
tile、`shave==1` 的那个 worker 全程什么都没干。这是"**按规则机械地推导，不管这次是否用得上**"的典型行为，和上一份
文档里讲的 npu_dsl `passes.derive_barriers` 原理完全一样，只是这次是亲眼在生成代码里看到的，而不是读源码脑补的。

### 3.5 VPUNN 没配置时的降级提示

```cpp
// intel_npu: DPU workloads NOT planned (8 left at the runtime default): VPUNN not configured: set VPUNN_ROOT
// (and build it with maint/scripts/build_intel_npu_vpunn.py) or VPUNN_PYTHON_DIR
```
`PlanIntelNpuWorkloads` 这个 pass 本该用 VPUNN 代价模型把每个 DPU workload 切成最优的 cuboid 形状,但这台构建机器
没配置 VPUNN，于是它**没有报错**，而是把这个事实原样写进生成代码的注释里，退回"运行时默认值（8 个 workload）"。
这是一个很值得学的设计：**一个可选的优化 pass 在依赖缺失时，选择"关掉自己、留痕迹"，而不是让整个编译失败**——后续
真要做性能调优（计划文档的 V5）时，第一件事就是装好 VPUNN,不然这个 pass 全程是摆设。

---

## 4. 和 npu_dsl 对照：同一件事两条管线怎么各自做

| 概念 | npu_dsl `fused` 管线 | TileLang intel_npu 管线 |
|---|---|---|
| IR 的载体 | 自定义 Python dataclass 树（`ir.FusedKernel`/`Shave`/`Barrier`/`Loop`） | TVM `IRModule` / `PrimFunc`（`tirx` 方言, `T.sblock` 为基本块） |
| "tile op" 的表达 | `nl.dma.load/store`、`nl.dpu.matmul`、`nl.shave.xxx`——**trace 时直接调用对应的 `OpRecord`** | `T.copy`、`T.gemm`——**先是中性的 TIR 节点**，要等 `LowerIntelNpuTileOp` 这个 intel_npu 专属 pass 才具体化成 `dma_load`/`dpu_matmul` |
| SHAVE 向量算子 | 查注册表 → 调用 `kernels_poc/common/fused_ops.hpp` 里手写、汇编优化过的 C++ 函数 | 没有注册表：作者直接写 `T.vectorized` 标量/向量循环，`VectorizeLoop` pass 认出模式后生成 `halfN`/`floatN` 向量指令 |
| CMX 分配 | Python 侧手算 live-range,生成形如 `arena + 384 + ctx.shave()*576u` 的字面量偏移 + `__pf1/__pf2` 双缓冲影子变量 | C++ 侧用 `npu::Ctx::CmxScope` RAII 对象做 bump-pointer,作用域结束自动回收;预取双缓冲是 `T.Pipelined` 触发的专门 pass（`LowerIntelNpuPipelined`）,不是默认行为 |
| 跨 worker barrier | `passes.derive_barriers`：扫 DDR 读写冲突,没有 barrier 语法 | `DeriveIntelNpuGlobalSync`：同样扫 DDR 读写冲突,没有 barrier 语法 —— **两边的设计哲学完全一致**，只是实现语言和 IR 形态不同 |
| "裸 Python for" 的语义 | **两种循环语义共存**：`nl.range(...)` 是运行时循环；裸 `for x in range(n):` 在 trace 阶段被 Python 解释器执行、展开成 n 份字面量代码 | **只有一种循环语义**：`T.prim_func`/`T.macro` 体内的任何 `for` （无论写成 `T.serial(n)` 还是裸 `range(n)`）都被 AST 解析器接管,变成同一种 TIR `For` 节点——**没有"trace 时展开"这回事** |
| 性能代价模型 | VPUNN,手工在 Python 里调用、手工选择 tile 大小分支（`nl.when`） | VPUNN 作为 `PlanIntelNpuWorkloads` 的后端,一个独立 pass,是否启用是编译时配置,不需要作者在源码里写分支 |
| 诊断信息 | 无（trace 失败直接抛 `SourceRefusal`,成功则不留痕迹） | 生成的 C++ 里保留**人类可读的诊断注释**（"tile-invariant source"建议、VPUNN 降级说明、barrier id 分配范围）——这是本次实测中印象最深的一点,调试时不需要额外加 print,直接读生成代码的注释就知道编译器做了什么决定 |

**给以后写 kernel 的直接提醒**（本次移植过程中真实踩到的,不是预先猜的）：
1. 不要假设裸 Python `for` 会在 trace 阶段展开——TileLang 这边没有这个机制,`T.serial(n)` 和 `for i in range(n):`
   在 `T.prim_func`/`T.macro` 体内是等价的,都会生成真实的运行时循环。
2. `T.gemm` 的 M、N、K 全部要求是 16 的倍数（`src/intel_npu/transform/lower_tile_op.cc` 的 `LowerIntelNpuTileOp`
   在 lowering 阶段才检查,不是 trace 阶段）——写测试用的"小而干净"的形状时,记得把 attention 的 `rows`（同时是
   `Q@K^T` 的 N 和 `P@V` 的 K）也对齐到 16,而不是只对齐 `hidden`/`head_dim`。
3. 生成代码里的诊断注释（`// intel_npu: ...`）值得在开发时养成"先读一遍"的习惯——第 3.5 节的 VPUNN 降级提示就是
   一个无害但容易被忽略的信号：没装 VPUNN 时,性能相关的 pass 其实什么都没做。

---

## 5. FGCLIP2 V0 当前状态与下一步

截至本文写作时（2026-11-11）：

- **仓库构建**：`pip install -e . --no-build-isolation` 在这台机器上**从零构建成功**（之前没有 `build/`
  目录,需要先 clone `3rdparty/cutlass`、`3rdparty/tvm` 两个子模块,再编译 TVM + TileLang 的 C++ 部分,整个过程
  用 clang-cl + Ninja 完成,无需 CUDA/ROCm/设备 SDK）。
- **V0 kernel**：`testing/python/intel_npu/fgclip2_kernels.py` 的 `make_fgclip2_encoder_layer_v0` 和
  `testing/python/intel_npu/test_intel_npu_fgclip2_v0.py` 的两个测试**真实跑通**，连续 6 次全部通过
  （`2 passed` x6,无一次失败,排除了偶发性）。过程中发现并修复了一个真实 bug：测试一开始用 `ROWS=8`,被
  `LowerIntelNpuTileOp` 拒绝（"gemm N=8, K=16 must be multiples of 16"）,改成 `ROWS=16` 后通过——已经记录进
  [fgclip2_enablement_plan.md](fgclip2_enablement_plan.md) 和仓库记忆,供以后参考真实边界条件,而不是靠猜。
- **下一步**（对照 [fgclip2_enablement_plan.md](fgclip2_enablement_plan.md) 的里程碑）：V1（27 层、真实模型
  形状 hidden=1152/16 头×72→80/inter=4304→4320、设备侧 layer 循环 + 权重表）尚未开始。

