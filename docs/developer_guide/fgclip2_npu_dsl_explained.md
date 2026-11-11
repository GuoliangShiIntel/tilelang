# FG-CLIP2-so400m 在 npu_dsl（原始实现）里是怎么跑起来的——给新人的逐层拆解

> 写作目的：这不是"做了什么改动"的变更记录，而是面向**完全新人**的原理讲解，目标是让你读完就能动手在 TileLang 的
> Intel NPU 后端里重新实现/优化这个模型。素材全部来自 `C:\Intel\guoliang\dsl\libraries.npu.dsl` 仓库
> `feat/fgclip2-so400m` 分支的真实代码（不是转述），关键结论都标了文件路径，自己可以去对照原文件验证。
> 行文沿用用户此前引用的技术报告标题风格（`fgclip2_nl_operators_explained.md`），但内容是独立重新梳理的。

## 目录
1. 模型是什么：FG-CLIP2-so400m
2. npu_dsl 里有两条编译管线，FG-CLIP2 走的是哪一条
3. `fused` 管线的核心概念（IR、Tracer、Pass、Emit）
4. 一个真实的「DSL 源码 → 生成 C++」最小例子（亲手跑出来的，非臆测）
5. FG-CLIP2 的模型数学（`reference.py` 逐段讲解）
6. 权重打包与设备内存布局（`pack.py`：几处"偷懒但等价"的折叠技巧）
7. Kernel 源码逐段解读（`fgclip2_layers_source.py`）
8. Kernel 配置项详解（`kernel_specs/fgclip2_fused.py`）
9. 测试与验证方法论
10. 给 TileLang 迁移工作的直接启示

---

## 1. 模型是什么：FG-CLIP2-so400m

`qihoo360/fg-clip2-so400m`，一个 **CLIP 风格的双塔（image tower + text tower）对比学习模型**：

- **Vision tower**：27 层 pre-LN Transformer encoder（SigLIP/so400m 架构），输入是图像 patch；最后接一个
  **attention-pooling head**（一个可学习的 probe 向量作为唯一 query，对全部 patch 做一次 attention）得到全局图像
  embedding；另外还有一个可选的 **dense head**（`get_image_dense_feature`：每个 patch 自己当 query，输出逐
  patch 的稠密特征图，用于区域/检测类下游任务）。
- **Text tower**：27 层结构相同的 encoder，输入是 64 token 的 caption（padding 到定长，**没有 attention mask**
  ——分词器根本不返回 mask，所以模型其实对 pad 位置也做了 attention，这是模型自身的行为，不是 bug，移植时必须
  原样保留）。最后取每个 caption **最后一个 token** 位置的输出做 LayerNorm + 投影头，得到文本 embedding；还有一个
  独立的 `boxtext_head`（`box` walk，用于和 dense map 配对的区域检索任务）。
- **相似度**：`logits = cosine(img, txt) * exp(logit_scale) + logit_bias`。

每层 encoder（`fgclip2_e2e/reference.py::encoder_layer`，这是整份实现的"数学真理来源"，被反复拿来做 oracle）：

```python
def encoder_layer(x, W, pre):
    h = layer_norm(x, W[pre+"layer_norm1.weight"], W[pre+"layer_norm1.bias"])
    a = attention(linear(h, W, pre+"self_attn.q_proj"), linear(h, W, pre+"self_attn.k_proj"),
                  linear(h, W, pre+"self_attn.v_proj"))
    x = x + linear(a, W, pre+"self_attn.out_proj")
    h = layer_norm(x, W[pre+"layer_norm2.weight"], W[pre+"layer_norm2.bias"])
    return x + linear(gelu_tanh(linear(h, W, pre+"mlp.fc1")), W, pre+"mlp.fc2")
```

16 head、每 head 维度 72（`hidden=1152=16×72`）、MLP 宽度 4304、GELU 用 **tanh 近似**（不是 erf 版本，这点在移植
时必须对上，否则数值会有百分之几的系统性偏差）、attention **无 causal mask、无 GQA**（纯双塔 MHA，所有 token 互相
可见）、LayerNorm **带 bias**（不是 Qwen 那种 RMSNorm）。

这比 tilelang 已经实现的 Qwen3.6-MoE 在数学上**简单**很多：没有 MoE、没有 Gated DeltaNet、没有 int4 量化、没有
causal mask、没有 RoPE、没有 KV cache（每次都是满长度一次性过完）。真正的难度全部在**怎么把它tile 到硬件上跑得
快**——这也是下面 90% 篇幅要讲的东西。

---

## 2. npu_dsl 里有两条编译管线，FG-CLIP2 走的是哪一条

`npu_dsl` 仓库（见 `sw_architecture.md`）其实有两套并存的编译路径：

| | "经典"管线（`npu_dsl/frontend/trace_*.py` + `npu_dsl/passes/`） | **`fused` 管线**（`npu_dsl/fused/`） |
|---|---|---|
| 定位 | 单算子/单 kernel，精细 pass 流水线（CMX 分配、barrier、naming……） | **多阶段大 kernel（"megakernel"）**，一个 launch 里塞下一整层甚至一整个模型 |
| IR | `ScheduledOp` 列表 + 各种 pass | `fused/ir.py` 的 `FusedKernel`（`Shave`/`Barrier`/`TileSync`/`Loop`/`DdrRegion`/`View` 构成的语句树） |
| 典型使用者 | 单个算子的 conformance 测试、早期 PoC | Qwen3.6 的 M1-M6、**FG-CLIP2**、GDN |
| 这份文档关心的 | 简单带过 | **主角** |

FG-CLIP2 的两个 kernel（`vision` 和 `text`）都定义在 `npu_dsl/frontend/kernel_specs/fgclip2_fused.py` 的
`KERNELS` 字典里，走的是 `fused` 管线：一次 launch 跑完 27 层 + pooling head（甚至 + dense head），这和 TileLang
这边 Qwen3.6 "一个 `prim_func` = 一个 `kernel_entry` = 多个 phase"的设计哲学是**同一回事**——这也是为什么这个模型
相对好移植：TileLang intel_npu 后端已经为这种"大 kernel 多 phase"的模式打好了地基（`T.Kernel` 持久 1D grid +
`T.Pipelined` + `T.sync_grid`），不需要从头发明。

---

## 3. `fused` 管线的核心概念

读 `npu_dsl/fused/ir.py`、`tracer.py`、`emit.py` 三个文件就能看懂整条链路。核心抽象：

### 3.1 `KernelDef`（kernel 的"签名"）

```python
@dataclass
class KernelDef:
    name: str
    source: Callable        # 被 trace 的 Python 函数：fgclip2_vision_source / fgclip2_text_source
    tensors: List[TensorSpec]   # DDR 参数（params[1..n]），每个 TensorSpec 记 name/dims/dtype
    config: List[ConfigField] = ...  # 运行时配置字段（params[0] 那个 struct 里的字段，每次 launch 可以不同）
    constants: Dict[str, int] = ...
```
`FgClip2Config`（见第 8 节）里的字段分两类：**编译期**（Python dataclass 字段，trace 时就已知，比如
`hidden=1152`、`tn=32`）和**运行时**（`config: List[ConfigField]`，比如 `n_valid`——同一份编译好的二进制，每次
推理传入不同的图片 patch 数，不用重新编译）。

### 3.2 `nl` 命名空间（DSL 的"语言"）

一个 kernel 的 `source` 函数签名是 `def fgclip2_vision_source(nl, cfg, *tensors)`。`nl` 是
`frontend/language.make_namespace(tracer)` 构造出来的一个**纯记录型代理**——它自己不做任何检查，只是把每次调用
记到 `tracer` 上：

- `nl.alloc_cmx(dtype, shape, scope=, name=)` → 登记一个 `ir.Buffer`（`scope="worker"` 每个 SHAVE 私有一份，
  `scope="tile"` 一个 tile 的两个 SHAVE 共享一份，`across_tiles="broadcast"` 则是所有 tile 共享同一份字节、由一次
  多播 DMA 填充）。
- `nl.dma.load(dst, src)` / `nl.dma.store(dst, src)` → `COPY`（DMA，**不做 dtype 转换**，和 TileLang 这边的
  `T.copy` 是同一回事）。
- `nl.dpu.matmul(out, a, bt, bias=, scale=)` → `GEMM`（`out = a · bt^T`，fp16）。
- `nl.shave.xxx(...)` → 查 `registry.py` 里的 `OpRecord`，映射到 `kernels_poc/common/fused_ops.hpp` 里一个手写
  的 C++ 函数（比如 `nl.shave.layernorm_f32` → `fops::layernorm_f32`）。**这是和 TileLang 风格最大的不同**：
  npu_dsl 的 SHAVE 算子是"调用一个已经手工汇编/C++ 优化好的库函数"，TileLang 这边（见下面第 10 节）则是直接用
  `T.vectorized`/`T.serial` 写标量/向量循环，由编译器把它们降低成 SHAVE 向量指令——两种风格都存在，TileLang 选的
  是后者。
- `nl.range(lo, hi, step, distribute="shaves"|None, name=)` → `Loop`。`distribute="shaves"` 是关键字：它说
  "这个循环的各次迭代轮转分配到每个 tile 的每个 SHAVE 上"（generic_compute 一个 kernel 默认 4 个 tile × 2
  SHAVE = 8 个 worker）；不带 `distribute` 的循环是**每个 worker 都完整跑一遍**（uniform）。
- `with nl.when(lhs, cmp, rhs):` → 运行时 if（比如根据 `cfg.bc`——一个运行时字段——选择用 broadcast 方案还是
  split 方案，同一份编译产物覆盖两种 launch 形状）。
- **没有显式的 barrier 语法**：`passes.derive_barriers`（trace 之后的一个 pass）自动扫描"哪个 DDR 张量被一个
  worker 写、又被另一个 worker 读"，在那里插入全局 rendezvous。TileLang 这边对应的是 `DeriveGlobalSync` pass
  （`src/intel_npu/transform/derive_global_sync.cc`），**同一个设计思路**，不用重新发明。

### 3.3 Pass 与 Emit

`trace(kd, cfg)` 跑完 Python 源码后得到一棵 `FusedKernel` 语句树，再经过 `fused/passes.py` 里的若干 pass
（CMX 活跃区间分配、barrier 推导、ring/pipeline 识别……），最后 `fused/emit.py::emit(k)` 把这棵树"打印"成一份
`FRT_KERNEL_ENTRY` 形式的 C++ 源码，`#include "fused_rt.hpp"` + `"fused_ops.hpp"`，每个 worker（SHAVE）各自跑一份
同样的代码（SPMD），靠 `ctx.worker()` / `ctx.shave()` 取自己的编号来分工。

---

## 4. 一个真实的「DSL 源码 → 生成 C++」最小例子

光看文字不如看一次真实运行结果。下面这段不是凭空编的——是在 `libraries.npu.dsl` 仓库里**实际跑了一遍**
`trace()` + `emit()`（一个只含 LayerNorm 的玩具 kernel，刻意选小，方便整份贴出来）：

**输入（DSL 源码）：**
```python
def tiny_source(nl, cfg, X, W, B, Out):
    x = nl.alloc_cmx(nl.fp32, [4, 8], scope="worker", name="x")
    w = nl.alloc_cmx(nl.fp16, [8], scope="worker", name="w")
    b = nl.alloc_cmx(nl.fp16, [8], scope="worker", name="b")
    y = nl.alloc_cmx(nl.fp16, [4, 8], scope="worker", name="y")
    for _ in nl.range(0, 1, distribute="shaves", name="one"):
        nl.dma.load(x, X[0:4, :])
        nl.dma.load(w, W[:])
        nl.dma.load(b, B[:])
        for r in nl.range(0, 4, name="row"):
            nl.shave.layernorm_f32(y[r, :], x[r, :], w, b, 1e-6)
        nl.dma.store(Out[0:4, :], y)
```

**输出（`emit(trace(kd, cfg))` 的真实结果，未删减）：**
```cpp
// Generated by npu_dsl.fused from __main__.tiny_source -- do not edit.
//
// params[0]: TinyLnConfigs
// params[1]: X fp32[4, 8]
// params[2]: W fp16[8]
// params[3]: B fp16[8]
// params[4]: Out fp16[4, 8]
// params[5]: status, 32 B per SHAVE
#include "fused_rt.hpp"
#include "fused_ops.hpp"

struct TinyLnConfigs {
    std::uint32_t unused;
};

FRT_KERNEL_ENTRY(numParams, params, env) {
    frt::Ctx ctx(env, numParams, params, 5);
    if (numParams != 6) { ctx.status(frt::kStatusBadParams, (std::uint32_t)numParams); return; }
    const TinyLnConfigs cfg = ctx.config<TinyLnConfigs>(params[0]);

    std::uint8_t* arena = ctx.alloc_tile<std::uint8_t>(1152);   // the tile's buffers, live-range packed
    float* x = reinterpret_cast<float*>(arena + 0 + ctx.shave() * 576u);   // fp32[4, 8]
    half* w = reinterpret_cast<half*>(arena + 384 + ctx.shave() * 576u);   // fp16[8]
    half* b = reinterpret_cast<half*>(arena + 256 + ctx.shave() * 576u);   // fp16[8]
    half* y = reinterpret_cast<half*>(arena + 512 + ctx.shave() * 576u);   // fp16[4, 8]
    half* b__pf1 = reinterpret_cast<half*>(arena + 320 + ctx.shave() * 576u);   // fp16[8]   (预取用的第二份缓冲)
    half* w__pf2 = reinterpret_cast<half*>(arena + 448 + ctx.shave() * 576u);   // fp16[8]
    float* x__pf3 = reinterpret_cast<float*>(arena + 128 + ctx.shave() * 576u);   // fp32[4, 8]
    if (!ctx.ok()) { ctx.status(frt::kStatusCmx, 0); return; }

    [&]() FRT_NOINLINE {
        std::uint32_t pfs_one = 0;
        half* const b__s0 = b;
        half* const w__s0 = w;
        float* const x__s0 = x;
        const std::uint32_t one_w0 = ctx.dist_first();
        FRT_PROF_BEGIN(0);
        if (one_w0 < 1) {
            ctx.pf_add(0, 0, x, params[1], 128, 1, 128);   // X
            ctx.pf_add(0, 0, w, params[2], 16, 1, 16);   // W
            ctx.pf_add(0, 0, b, params[3], 16, 1, 16);   // B
            ctx.pf_issue(0, 0);
        }
        FRT_PROF_END(0);
        for (std::uint32_t one_it = one_w0; one_it < 1; one_it += ctx.workers()) {
            const std::uint32_t one = one_it;
            FRT_PROF_BEGIN(1);
            ctx.pf_wait(0, pfs_one);
            FRT_PROF_END(1);
            half* const b = pfs_one ? b__pf1 : b__s0;
            half* const w = pfs_one ? w__pf2 : w__s0;
            float* const x = pfs_one ? x__pf3 : x__s0;
            FRT_PROF_BEGIN(2);
            if (one_it + ctx.workers() < 1) {
                ctx.pf_add(0, pfs_one ^ 1u, (pfs_one ? x__s0 : x__pf3), params[1], 128, 1, 128);   // X
                ctx.pf_add(0, pfs_one ^ 1u, (pfs_one ? w__s0 : w__pf2), params[2], 16, 1, 16);   // W
                ctx.pf_add(0, pfs_one ^ 1u, (pfs_one ? b__s0 : b__pf1), params[3], 16, 1, 16);   // B
                ctx.pf_issue(0, pfs_one ^ 1u);
            }
            FRT_PROF_END(2);
            for (std::uint32_t row = 0; row < 4; row += 1) {
                FRT_PROF_BEGIN(3);
                ctx.guard((y + (row * 8)), 16);
                fops::layernorm_f32((y + (row * 8)), (x + (row * 8)), w, b, 8, 1e-06f);
                FRT_PROF_END(3);
            }
            FRT_PROF_BEGIN(4);
            ctx.dma_store(params[4], y, 64, 1, 64);   // Out
            FRT_PROF_END(4);
            pfs_one ^= 1u;
        }
        ctx.dist_advance(1);
    }();
    FRT_PROF_FLUSH(5);
    ctx.status(frt::kStatusOk, 0);
}
// @site 0 load 0 - pfissue
// @site 1 load 0 one* pfwait:X,W,B
// @site 2 load 0 one* pfissue
// @site 3 shave 0 one*/row layernorm_f32
// @site 4 store 0 one* Out
```

几个值得注意的"编译器自己做的事"（**源码里一行都没写**，全是 pass 自动插入的）：
1. **CMX 地址分配**：`arena`（每个 tile 1152 字节）由 live-range 打包自动给每个 buffer 分配偏移量（`x` 在偏移 0，
   `w` 在 384……），且自动按 `ctx.shave()` 做了 per-SHAVE 的地址偏移（`+ ctx.shave()*576u`）——这就是
   `scope="worker"` 的真实含义：同一份代码对两个 SHAVE 读到的是两块不同的内存。
2. **双缓冲预取（software pipelining）**：源码只写了"循环体内 dma.load 再用"，emit 出来的代码却自动生成了
   `b`/`w`/`x` 各自的 `__pf1/__pf2/__pf3` 第二份缓冲，并在循环体**顶部**先 `pf_wait` 当前这份、同时为**下一次**
   迭代发起 `pf_add/pf_issue`——这是一个自动插入的 2 级流水（ping-pong），后面"3-slot weight ring"
   （`passes._ring_ok`）是这个机制的升级版。TileLang 这边对应的正是 `T.Pipelined(num_stages=2)`。
3. **Profile 站点（`@site`）自动编号**：每个 DMA/SHAVE/barrier 调用自动包一层 `FRT_PROF_BEGIN/END(n)`，文件尾部
   自动生成一张"站点号 → 源码位置"映射表，供 `perf_attr.py`/`profile_report.py` 做 trace 归因——这也正是 TileLang
   `perf_attr.py` 想做的事（参见 intel-npu-perf-loop skill）。

理解了这个"麻雀虽小五脏俱全"的例子，再去读 552 行的 `fgclip2_layers_source.py` 就不会觉得无从下手了——那份文件
里的每一行 `nl.xxx(...)`，脑子里都应该在自动"翻译"成类似上面这样的 C++。

---

## 5. FG-CLIP2 的模型数学（`reference.py`）

`fgclip2_e2e/reference.py` 是**唯一的数学真理来源**（不依赖 `transformers`，纯手写 + `safetensors` 读权重），后面
的每一步（打包、host 模拟、设备运行）都拿它当 oracle 核对。关键几点：

- `layer_norm`：标准 `(x-mean)/sqrt(var+eps)*w+b`，`eps=1e-6`。
- `gelu_tanh`：`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`——**tanh 近似**，不是 erf 版本。
- `attention(q, k, v)`：16 head，`scale = head_dim^-0.5 = 72^-0.5`，**无 mask**。
- **Vision**：`x = patch_embed(patches) + resized_pos_embedding`（位置编码是 16×16 的网格双线性插值到实际
  patch 网格大小，这一步在 host 上用 `torch.nn.functional.interpolate(..., antialias=True)` 做，**不在 NPU
  kernel 里**）→ 27 层 encoder → `post_layernorm` → **attention pooling head**（一个可学习 `probe` 向量做
  query，对全部 patch 做一次 attention，再过 out_proj + LayerNorm + MLP，取第 0 行当 embedding）→ 可选
  `dense_head`（每个 patch 自己当 query 的同构 attention + MLP）。
- **Text**：`x = token_embedding[ids] + position_embedding[:64]` → 27 层 encoder（**无 mask**）→
  `final_layer_norm` → 取**每个 caption 最后一个 token**（`x[-1:]`）→ 投影头（`text_model.head` 或
  `boxtext_head`）。
- `similarity`：先对 image/text embedding 做 L2 归一化，再 `cosine * exp(logit_scale) + logit_bias`。

---

## 6. 权重打包与设备内存布局（`pack.py`）

`fgclip2_e2e/pack.py` 把 `model.safetensors` 转换成 kernel 直接能用的二进制布局，过程中做了好几处**数学上等价、
但能省掉设备上重复计算**的折叠技巧——这些技巧**必须在打包阶段原样搬过去**，TileLang 这边的权重导出脚本如果漏掉
任何一条，模型数值就会错：

| 技巧 | 原因 |
|---|---|
| **head_dim 72 → 80 补零**（`hd_pad`） | DPU 的 N、K 维度要求是 16 的倍数；attention 的向量化 SHAVE 算子在 32-lane 向量上操作（奇数维需要一条 16-lane 的）。q/k/v 的权重行、o_proj 的权重列都按 head 补零。 |
| **q 预缩放 `× 72^-0.5`**（权重和 bias 都乘） | 省掉设备上"算完 QK^T 再乘 scale"这一步，GEMM 出来的分数已经是缩放过的。 |
| **丢弃 k 的 bias** | `q · (k+bk) = q·k + 常数`，而 softmax 对每行加同一个常数不变——所以 k_proj 的 bias 可以安全丢弃，省一次 bias add。 |
| **v 的 bias 折进 o_proj 的 bias**：`bo' = bo + Wo·bv` | softmax 的权重对每行求和为 1，所以 `softmax(·)·(V+bv) = softmax(·)·V + bv`，这个常数项经过 `o_proj` 线性变换后就是 `Wo·bv`，可以提前在打包阶段加到 `bo` 里，**设备上 v_proj 完全不用算 bias**。 |
| **q/k/v 三个投影权重堆成一个 `Wqkv`** | 一次 GEMM 出 q\|k\|v，而不是三次——这是"Qwen layout"的堆叠写法，减少一次权重 DMA + 一次 kernel launch 开销。 |
| **LayerNorm 的 scale/shift 折进其后的 GEMM**（`ln_fold=True`，默认开启） | `LN(x)*γ+β` 之后立刻接一个线性层 `W·(·)+b` 时，等价于 `W' = W·diag(γ)`、`b' = b + W·β`，设备上 LayerNorm 就只需要做"纯归一化"（`layernorm_nw_f32`，无 affine），两个 LayerNorm 各省一次逐元素乘加。 |
| **MLP 宽度 4304 → 4320 补零**（32 的倍数） | `fc1` 补的行、`fc2` 补的列全是 0，`gelu(0)=0`，所以补零部分对最终结果零影响，纯粹是为了让 DPU 的 N 维对齐 32。 |
| **V 不转置存，让 DPU 用"单位矩阵乘法"转置** | `V^T[c,t] = Eye[c,c] · V[t,c]^T`——与其写一个 SHAVE 转置循环，不如让 DPU（本来就擅长做矩阵乘）用一个预先生成的单位矩阵"乘"出转置结果，这样 `P·V` 这一步就不需要再转置。 |
| **Attention pooling head 的 query 是常数**：`probe @ Wq + bq`，整条打包进 `MapQ` 一个固定向量 | 因为 probe 和 Wq/bq 都是训练好的常量，`q = probe·Wq+bq` 在打包阶段就能提前算好，设备上连这次 GEMV 都不用做。 |

每个 encoder 层的 6 个"杂项"向量（两个 LayerNorm 的 w/b、折叠后的 o_proj bias、fc2 bias）打包进一个 `LV`
`[layers, 8, hidden]` 张量，省掉很多小 DMA。27 层权重按 `[layer, ...]` 堆叠成大张量（`Wqkv`/`Wo`/`W1`/`W2`
等），vision tower 还多留一个 slot（第 27 个）存 dense head 的权重——**同一套 `_layer()` 代码**通过换一个 layer
下标就能跑 dense head，这是个很值得学的"复用已有循环体"的技巧（见第 7.6 节）。

---

## 7. Kernel 源码逐段解读（`npu_dsl/frontend/examples/fgclip2_layers_source.py`，552 行）

这是全文件信息密度最高的部分，强烈建议对照本地文件（路径见标题）边读边看。按出现顺序讲。

### 7.1 `_ln_rows`：带 bias 的 LayerNorm（可选 affine）

```python
def _ln_rows(nl, cfg, Hres, wref, bref, Xn, rows, name, affine=True):
    ...
    for t0 in nl.range(0, rows, B, distribute="shaves", name=f"{name}Tok"):
        tb = emin(B, rows - t0)
        nl.dma.load(h[0:tb, :], Hres[t0:t0 + tb, :])
        for tt in nl.range(0, tb, name=f"{name}Row"):
            if affine:
                nl.shave.layernorm_f32(x[tt, :], h[tt, :], w, b, cfg.eps)
            else:
                nl.shave.layernorm_nw_f32(x[tt, :], h[tt, :], cfg.eps)
        nl.dma.store(Xn[t0:t0 + tb, :], x[0:tb, :])
```
`rows` 个 token 按块（每块 `B=cfg.ln_tb` 个）**分给各 SHAVE**（`distribute="shaves"`），每个 SHAVE 内部再逐行跑
LayerNorm。`affine=not cfg.ln_fold`——因为默认 `ln_fold=True`，两个 LayerNorm 的 scale/shift 已经被打包阶段折进
后面的 GEMM 了，这里只需要**纯归一化**（`layernorm_nw_f32`，不读 `wref`/`bref`）。`cfg.ln_tb`（token block 大小）
是一个被实测调过的参数：代码注释直接写了"4/8/16/32 分别是 99.2/97.1/97.4/97.5 ms"——**这是典型的"测量后钉死"而非
"建模后假设"**的工程方法论，和 Qwen plan 里"prune → measure → pin"的原则一致。

### 7.2 `_gemm` / `_gemm_split` / `_gemm_bcast`：同一个数学操作的三种硬件调度方案

`_gemm` 本身只是个**分发器**：根据 `cfg.bcast_gemm`（编译期）和 `cfg.bc`（运行时，由 `vision_launch()` 按本次
patch 数算出）选择两套完全不同的 DMA/计算调度策略，但数学结果完全一样（`Out = X · W^T (+bias)(+GELU)`）：

- **`_gemm_split`**（"X-stationary"）：每个 worker 认领一块（行块, 权重 tile 切片）的"活"，激活行块留在 CMX 里，
  权重 tile 一块块流过去做 GEMM。**权重会被每个认领它的行块各读一次**——行块少（小图/单 caption）时用这个更快，
  因为读权重的次数本来就少。
- **`_gemm_bcast`**（"权重常驻广播"）：反过来，每个 worker 认领一批固定的行（用 `mbw` 算），所有 worker **同步
  地**流过同一批权重 tile——权重从 DDR 读一次，**多播**进一个 `scope="tile", across_tiles="broadcast"` 的共享
  buffer（`passes` 会给这种 buffer 插入"tile 内部 SHAVE 互相同步"的逻辑，是和 ping-pong 预取不同的另一种
  同步原语：`TileSync`）。行数多（大图）时这个更快，因为权重只读一次、分摊到更多行上。
- **`cfg.bcast_min_rows=384`** 是这两条路径的切换阈值，**同样是实测出来的**（注释写了"128/256 patch 时实测
  split 快 38%/9%"）。
- **`parts`**（K 维切分，`fc2` 用 `f2_parts=3`）：K=4320 太大放不进 CMX（激活块+权重 tile 同时常驻），于是切成
  3 份分别过 GEMM 再累加——这是**纯粹被 CMX 容量逼出来的**切分，不是为了并行。
- **`dpu_overlap` + `matmul_start`/`matmul_wait`**：DPU 算当前 tile 的同时，SHAVE 去做上一个 tile 的
  bias/GELU/残差/store——这是手工软件流水，目的是让 DPU 和 SHAVE 两条执行单元都不空闲。

**这三个函数的本质**：同一个"行分块 × 权重分块"的 GEMM 问题，在"行块很少"和"行块很多"两种情况下**最优的 DMA 调度
方向是相反的**——这是一个很通用的教训，TileLang 这边 Qwen 的 `gemm_strategy_sweep.py`（"prune→measure→pin"）
本质上就是在系统化地做这件事，而不是像这里一样手写两条路径再用 `nl.when` 挑选。**迁移建议**：TileLang 版本不必
照搬两条路径，先用一条简单路径把正确性跑通，再用相同的"扫描+实测+钉死"方法论决定是否需要第二条路径。

### 7.3 `_vt_gemm` 系列：用 DPU"转置"V

```python
def _vt_gemm_split(nl, cfg, Xn, Wrows, VT, rows, name):
    """VT[h*DP+d, t] = sum_k Wv[h*DP+d, k] . Xn[t, k]"""
    ...
    for h in nl.range(0, NH, distribute="shaves", name=f"{name}Head"):
        nl.dma.load(wv, Wrows(h * DP, h * DP + DP))     # 这个 head 的 V 权重常驻 CMX
        for t0 in nl.range(0, rows, TT, name=f"{name}Tile"):
            nl.dma.load(xt, Xn[t0:t0 + TT, :])
            nl.dpu.matmul(yv, wv, xt)                    # 权重当 "a"，激活当 "bt"：天然出转置结果
            nl.dma.store(VT[h * DP:h * DP + DP, t0:t0 + TT], yv)
```
这里的技巧是：`nl.dpu.matmul(out, a, bt)` 算的是 `out = a · bt^T`。正常算 `V = X·Wv^T` 应该 `a=X, bt=Wv`；但如果
**把权重放在 `a` 的位置、激活放在 `bt` 的位置**，算出来的 `out = Wv · X^T` 恰好就是**转置过的 V**——不需要额外的
转置步骤，白捡一次转置。`_layer()` 里单独的 `_v_transpose` 函数则是对"标准 QKV 投影"算出来的 V（还没转置）用
真正的单位矩阵乘法转置（见上面第 6 节的 Eye 技巧），两者服务于同一个目的（attention 要的是 `P·V`，如果 V 已经是
`[D, T]` 的转置形式，`P[Tq,Tk]·V^T[D,Tk]`这个乘法可以用 `nl.dpu.matmul(out, P, VT, transpose_B=True)` 直接做）。

### 7.4 `_attn_state` / `_flash_chunks`：在线 softmax（flash attention）状态机

```python
def _attn_state(nl, rows, C, D, name):
    return _NS(qb=..., kc=..., vc=..., s=..., pv=..., oacc=...,
               ml=nl.alloc_cmx(nl.fp32, [rows, 2], ...),      # 在线 (max, sum)
               corr=nl.alloc_cmx(nl.fp32, [rows, 2], ...))     # rescale 系数

def _flash_chunks(nl, st, K, VT, h, k0, nch, valid, C, D, rows, name, kcol=0):
    nl.shave.attn_init(st.ml, st.oacc)
    for cc in nl.range(0, nch, name=f"{name}Chunk"):
        c0 = cc * C
        nl.dma.load(st.kc, K[k0+c0:k0+c0+C, kcol+h*D:kcol+h*D+D])
        nl.dma.load(st.vc, VT[h*D:h*D+D, k0+c0:k0+c0+C])
        nl.dpu.matmul(st.s, st.qb, st.kc)                         # S = Q . K^T
        nl.shave.attn_online_v(st.s, st.ml, st.corr, st.s, valid - c0, rows)  # 在线 max/exp/sum + rescale 系数
        nl.dpu.matmul(st.pv, st.s, st.vc)                          # P . V
        nl.shave.attn_acc_v(st.oacc, st.corr, st.pv)               # O = O*corr + P.V
```
这是标准 flash-attention 的在线 softmax：按 key 分 chunk（大小 `C`）过一遍，每个 chunk 更新一次运行中的
`(max, sum, O)`，用 `corr = exp(old_max - new_max)` 做数值稳定的 rescale。**`valid` 参数是这里处理"padding"的
方式**：图像的 patch 数 `n_valid` 不一定是 chunk 大小的整数倍，补齐到 `n_pad` 后，最后一个 chunk 里超出
`n_valid` 的 key 需要被当作"不存在"——`attn_online_v` 的第五个参数 `valid - c0` 就是"这个 chunk 里还剩几个真
key 有效"，这比显式造一个 mask buffer 更省：**传一个整数边界而不是一整条 mask 向量**。文本塔同理：64 个 token
全部有效（`_flash_chunks(..., valid=TL, C=TL, ...)`，一次性一个 chunk 吃完整条 caption，`test_fgclip2.py` 专门
有一个测试断言这一点：`valid = all 64 keys`，**验证"确实无 mask"**）。

`attn_overlap=True` 时（`attention_c` 函数内），**上一个 chunk 的 `P·V` 累加**和**当前 chunk 的 `Q·K`**被交错成
`matmul_start`/`matmul_wait` 对，让 DPU 不间断工作——这是和 7.2 节同一个"手工软件流水"手法在 attention 里的版本。

### 7.5 `_v_transpose`：批量转置 QKV 里已经算出的 V

和 7.3 节 `_vt_gemm` 做的是同一件事（V 转置成 `VT[C, T]` 形式），区别是这里的输入已经是 `Qkv` 这个大投影张量里的
一段切片（`T.Qkv[t0:t0+TB, 2*QW+c0:2*QW+c0+TV]`），用的是真正的单位矩阵技巧（`Eye`）而不是"权重当激活"的技巧——
因为这里没有"权重"可以拿来复用，输入纯粹是激活。

### 7.6 `_layer`：一层 encoder 的完整装配

```python
def _layer(nl, cfg, T, layer, rows, attention, dense=False):
    lv = lambda i: T.LV[layer, i, :]
    if not dense:
        _ln_rows(nl, cfg, T.H, lv(0), lv(1), T.Xn, rows, "ln1", affine=not cfg.ln_fold)
    _gemm(nl, cfg, T.Xn, lambda r0,r1,k0,k1: T.Wqkv[layer, r0:r1, k0:k1], T.Qkv, 3*QW, Hd, cfg.tn, cfg.mb_a, "qkv",
          rows, bias=T.Bqkv[layer, :], ns=_ns(cfg.ns_a))
    _v_transpose(nl, cfg, T, rows)
    attention(layer)
    _gemm(..., T.Wo, ..., bias=lv(4), resid=(T.Zr if dense else T.H, T.H), ...)   # o_proj + 残差相加
    _ln_rows(nl, cfg, T.H, lv(2), lv(3), T.Xn, rows, "ln2", ...)
    _gemm(..., T.W1, T.Act, ..., bias=T.B1[layer,:], gelu=True, ...)              # fc1 + GELU
    _gemm(..., T.W2, ..., bias=lv(5), resid=(T.H, T.H), parts=cfg.f2_parts, ...)  # fc2 + 残差相加
```
**`layer` 是一个参数，不是 Python 循环展开的下标**——`fgclip2_vision_source` 外层用
`for layer in nl.range(0, cfg.n_layers, name="layer"):` 包着调用 `_layer`，这是一个**设备侧的循环**
（`nl.range` 不带 `distribute` 时每个 worker 都跑全部迭代），`T.Wqkv[layer, ...]` 里的 `layer` 变成生成的 C++
里一个真正的循环变量去做指针偏移——**不是在 trace 阶段把 27 层展开成 27 份重复代码**（这与 TileLang Qwen 版本里
"device-side layer loop" `for li in ...: Kind[li]` 的设计完全一致，均避免 `.text` 段爆炸）。

**`dense=True` 复用同一个 `_layer` 函数跑 dense head**：dense head 的 attention 其实就是"每个 patch 自己是
query"的 attention（和编码层的 self-attention 结构一样），区别只是：(1) 跳过第一个 LayerNorm（dense head 的
输入已经是 post-LayerNorm 过的 `Xn`）；(2) attention 输出不加到真实残差流上，而是加到一个全零张量 `Zr` 上（因为
数学上 dense head 的输出是 `a + MLP(LN(a))`，没有外部残差输入）。`cfg.dense_slot`（权重第 27 个 slot）让
`T.LV[cfg.dense_slot, ...]`/`T.Wqkv[cfg.dense_slot, ...]` 自动取到 dense head 自己的权重——**“一份代码、不同
下标”复用的极致**。

### 7.7 `fgclip2_vision_source`：image tower 整体流程

```
Patches, PosB --[_gemm, resid=(PosB,H)]--> H（fp32 残差流，= patch_embed + 位置编码）
for layer in range(27): _layer(...)                              # 27 层 encoder（attention 见 attention_c 下面讲）
H --[_ln_rows]--> Xn                                              # post_layernorm
Xn --[_gemm]--> Ks (pooling head的key投影)
Xn --[_vt_gemm]--> VT (pooling head的value，已转置)
for h in range(16): _flash_chunks(单 query 行 MapQ, Ks, VT) --> MapAo   # pooling attention：probe 当唯一 query
MapAo --[_gemv + LN + _gemv(GELU) + _gemv]--> Emb                 # out_proj, LN, fc1+GELU, fc2, 残差相加 = 最终 embedding
for _ in range(cfg.dense): _layer(layer=27, dense=True)            # 可选：dense head，复用 _layer
```
`attention_c`（vision 塔自己的 encoder-attention 实现，不是 pooling head 那个）是一个**展平过的双重循环**：
```python
for g in nl.range(0, (NH * nqb // W) * nch, name="vAttnStep"):
    c = g % nch                      # 本次迭代对应第几个 key chunk
    it = (g // nch) * W + cfg.worker  # 本次迭代对应第几个 (head, query块) 任务项
    h, q0 = it // nqb, (it % nqb) * TB
```
代码注释解释了为什么要"展平成一个循环"而不是"外层按任务项、内层按 chunk 嵌套两层循环"：**让下一步（无论是不是
跨任务项边界）的 Q/K/V 加载提前发出去**，这样没有任何一个任务项的第一个 chunk 需要"空等"它自己的加载完成——是
预取跨迭代边界持续流动的一种写法技巧，值得在 TileLang 里复刻（`T.Pipelined` 应该能自然处理，需要验证)。

`cfg.v_chunk_big`：如果 chunk 数是偶数，可以把两个 `v_chunk=256` 的 chunk 合并成一个 `v_chunk_big=512` 的大
chunk 跑（`nl.when` 运行时选择）——**chunk 越大、DMA/launch 开销摊得越薄**，但 `n_valid` 运行时才知道、奇偶性
不固定，所以两条路径都要生成，运行时选。

### 7.8 `fgclip2_text_source`：text tower 整体流程

```
Ids --[逐 token embedding lookup + 加位置编码]--> H
for layer in range(27): _layer(..., attention=text_attention)   # 同一个 _layer，但 attention 是双塔共享结构不同实现
H --[取每个 caption 最后一个 token, LN]--> XL
XL --[_gemm, bias=head_bias]--> Emb       (short walk)
XL --[_gemm, bias=boxhead_bias]--> EmbBox (box walk)
```
text tower 的 attention 更简单：每个 caption 固定 64 token，**一次 chunk 吃完整条 caption**（`_flash_chunks(nl,
st, ..., nch=1, valid=TL, C=TL, ...)`），所以这里的"flash"其实退化成了普通 attention——**真正的 online-softmax
分块逻辑只在 vision 塔被用上**（因为一张图可能几千个 patch，key 太多放不进一次 CMX；一条 caption 固定 64 个
token，一次就放得下）。

---

## 8. Kernel 配置项详解（`kernel_specs/fgclip2_fused.py`）

`FgClip2Config` 的字段大致分四类，理解这个分类对设计 TileLang 版本的配置很有参考价值：

1. **模型结构常量**（永远不变）：`hidden=1152`、`n_layers=27`、`heads=16`、`head_dim=72`、`inter=4304`、
   `eps=1e-6`、`vocab=256000`……
2. **硬件/对齐常量**（由硬件约束决定，不是"调出来的"）：`hd_pad=80`（16 的倍数）、`inter_pad=4320`（32 的倍数）、
   `n_pad_align=128`（图像 patch 数补齐粒度）、`cmx_budget_bytes`（硬件 CMX 容量减掉安全余量）。
3. **调度策略旋钮**（编译期，**实测后钉死**，注释里往往直接写着对比数字）：`ln_tb`、`mb`/`mb_a`/`mb_b`、
   `tn`/`tn_o`/`tn_1`/`tn_2`/`tn_b*`、`bcast_gemm`/`bcast_min_rows`、`dpu_overlap`/`pf_ring`/`split_sync`、
   `v_chunk`/`v_chunk_big`/`v_tb`/`attn_overlap`。**这些就是"性能工程"的全部战场**，第 10 节会讲 TileLang
   这边应该怎么系统化地重新决定它们而不是照抄数字（不同硬件代际、不同 compiler 后端，最优值不保证一样）。
4. **运行时字段**（每次 launch 可能不同，`vision_launch()`/`text_launch()` 按实际输入形状算出来）：
   `n_valid`/`n_pad`/`n_caps`/`bc`/`mbw`/`ns_a`/`ns_o`/`ns_2`——这些决定了同一份编译产物在不同输入大小下走哪条
   分支、分几片。

`_slices()` / `_mbw()` 两个辅助函数是"简易代价模型"：`_slices` 用一个解析公式（而不是实测扫描）估算"把权重 tile
切成几份能让所有 SHAVE 最忙"，`_mbw` 则纯粹是"行数平均分给 workers，再向上取整到 DPU 的行粒度 16"。这是"建模"和
"实测钉死"两种方法论在同一份代码里**共存**的例子——粗粒度的运行时决策用便宜的解析模型，细粒度的编译期常数用实测。

---

## 9. 测试与验证方法论

这条流水线上下游一共四层验证，每一层都在前一层基础上做"差分"，而不是每次都从头对比到 fp32：

1. **`reference.py`**：fp32 PyTorch 直接实现，用 `check_vs_hf.py` 对照 HuggingFace 官方 `transformers` 模型，
   确认自己手写的这份 reference 是对的（"oracle 的 oracle"）。
2. **`golden.py`**：用 `reference.py` 在一个固定测试集（若干图片 + captions）上跑一遍，存下 image/text
   embedding、相似度矩阵、检索指标（R@1/R@5）、以及**第一张图/第一条 caption 的逐层中间结果**（`trace=True`），
   作为后续每一层数值比对的"标准答案"文件 `golden_fp32.pt`。
3. **`npu_dsl/tests/test_fgclip2.py`**：**不跑设备，只验证 kernel 能不能 trace 成功、CMX 预算是否超标、配置
   字段是否完整、关键的硬件对齐断言（pad 到 16/32 的倍数等）、以及三个新增 SHAVE 算子（带 bias 的 LayerNorm、
   bias add、bias+GELU）在 host simulation 下和 numpy 实现是否一致**（见 `_ops_def` 那段小 kernel）。这是最快的
   反馈循环——不需要真实权重、不需要设备。
4. **`fgclip2_e2e/device_gate.py`**（host 模拟 或 真实 NVL 设备）：用真实打包权重（`pack.py` 产出）跑完整
   vision/text kernel，和 `golden.py` 的结果比对数值误差。

**对新人最有用的教训**：**先写只测"这个 kernel 能不能 trace、CMX 够不够"的单元测试，再写"数值对不对"的单元测试，
最后才是"接真实权重、接真实设备"的端到端测试**——如果反过来（一上来就追求端到端打通），出错时完全不知道该往
哪个方向查。这与 TileLang 已有的 `test_intel_npu_m*.py`（host-sim 优先）→ `device/run_*_device.py`（设备验证）
两段式流程是同一个方法论。

---

## 10. 给 TileLang 迁移工作的直接启示

| npu_dsl 的做法 | TileLang intel_npu 后端的对应物 | 迁移时的结论 |
|---|---|---|
| `nl.shave.layernorm_f32(...)` 调用手写 C++ 函数 `fops::layernorm_f32` | 没有现成的"调用式" SHAVE op；M4/M5 的 `rmsnorm_zc` 是直接用 `T.serial`/`T.vectorized` + `T.sqrt`/`T.exp` 写出归约循环，编译器自己降低成 SHAVE 向量指令（`testing/python/intel_npu/kernels.py` 的 `norm_block`/`prefill_norm_block`） | **带 bias 的 LayerNorm 要新写**，照抄 `rmsnorm_zc` 的结构（按 `FL=16` 向量化累加 `ls[v]`，`tl_npu_vsum` 归约，最后写回 fp16），只是把 `rsqrt*(1+w)` 换成标准 LN 的 `(x-mean)*invstd*w+b`，**两趟**（先 mean 后 var，或用 `sum`/`sumsq` 一趟） |
| `nl.shave.bias_gelu_h` 调 `fops::...` 里的 GELU-tanh | 无现成实现，但 `T.tanh` 是 TVM 内建 intrinsic，且 `src/tl_templates/intel_npu/npu_rt_vecmath.hpp` 已经有 **VAU.TANH 的 half32/fp32 向量化快速路径**（`TL_NPU_VAU_F16(tanh, ...)`） | GELU-tanh **可以直接用 `T.tanh(...)` 写**在 `T.vectorized` 循环里，硬件加速路径已经打通，不需要新增 runtime 支持 |
| `nl.dpu.matmul`（`out=a·bt^T`） | `T.gemm(A, B, C, transpose_B=True, clear_accum=True)` | 一一对应，直接用 |
| `nl.when(cond): ...` 运行时分支 | `T.Pipelined`/if 普通 TIR 分支（需确认 intel_npu 后端对运行时标量条件分支的支持程度） | 需要先用一条简单路径验证正确性，**不必一开始就复刻 split/bcast 双路径** |
| flash attention 的 online softmax + chunk 循环 | `make_flash_attention_phase`（已经是双向、非 causal 的 flash attention！）与 `make_attn_prefill`（按位置做因果 mask） | **FGCLIP2 的 attention 本质上就是 `make_flash_attention_phase` 去掉 causal 部分、换成"剩余有效 key 数"的边界判断**——已有的 Qwen attention kernel 可以直接作为起点大幅改写，不需要从零写 |
| DMA 不做 dtype 转换 | 同样不做 dtype 转换（`T.copy` 是裸拷贝） | 完全一致的坑，注意 patch 输入 fp16/位置编码 fp32 的混合精度处理需要显式转换（`T.cast`） |
| 权重打包（`pack.py` 的折叠技巧：q 预缩放、k bias 丢弃、v bias 折进 o_proj、LN 折进 GEMM、head_dim/MLP pad） | 尚不存在；需要新写一份 TileLang 版本的打包脚本 | 这些折叠技巧和硬件无关，是**纯数学等价变换**，可以**直接照搬** `pack.py` 的公式，不需要重新推导 |
| 一个 `prim_func` 内部用 Python `for layer in range(27)` 展开 vs. 设备侧 `nl.range` 循环 | Qwen 的 M5 已经验证了"device-side layer loop + 每层查表取权重地址"（`T.ddr_view` 地址表）的方案，`.text` 预算校验过 | FGCLIP2 27 层如果全部静态展开，`.text` 段大概率超预算（Qwen 40 层都需要设备侧循环）——**应该直接复用 Qwen 的 layer-table 设计**，而不是新发明 |
| `tn`/`mb`/`bcast_gemm` 等調度旋钮靠"实测后钉死"，注释里带着具体实验数字 | Qwen plan 第 4 节记录的"`TileGemm`: 合法候选集排序、top-k 编译、设备上一次性测、钉死、记录 regret"方法论 | **不要把 npu_dsl 里的具体 tile 大小数字照抄过来**——NPU5/NVL 和这次的硬件代际、以及 TileLang 的 codegen 和 npu_dsl 的手写 C++ 性能特性都不同，数字需要在新后端上重新扫描；但"先用一条正确的慢路径跑通，再按同一方法论扫描调优"的**过程**可以照搬 |

下一步（配套的 `docs/developer_guide/fgclip2_enablement_plan.md` 已经把这些拆成了具体的里程碑）：先搭 LayerNorm
(bias) + GELU-tanh + 双向 flash attention 三个构建块，拼成一个能在 host-sim 下对上 numpy oracle 的单层
encoder，再逐步叠加 27 层循环、pooling head、text 塔、权重打包，最后才是性能调优。
