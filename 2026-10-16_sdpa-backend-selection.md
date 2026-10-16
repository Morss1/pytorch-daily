# PyTorch 每日一课 · 第 038 期

## SDPA 的后端选择逻辑：flash / mem-efficient / cuDNN / math 谁上，以及它是怎么静默降级的

**日期**：2026-10-16　**难度**：⭐⭐⭐⭐　**预计阅读时间**：28 分钟

**前置知识**：知道 `F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False)` 的签名；读过第 005 期（Flash Attention 的在线 softmax）；第 032 期（MHA/MQA/GQA/MLA）的结论最好还记得——本期的门表里有一项就是 GQA。

**关联**：第 034/035/036 期在做稀疏注意力、KV 驱逐、attention sink 时，**三次撞到同一堵墙**——`SDPBackend` 枚举里找不到自己要的后端。本期把那堵墙拆开。第 005 期讲的是「flash 算法为什么省显存」，本期第 6 节把那个「省」量成 **64 倍**。第 037 期讲「判断一个数字之前先问机制本身长什么样」，本期就是同一件事在框架内部的版本。

---

## 0. 一句话

> **SDPA 的后端不是「你选的」，是「一套三层结构选的」：一张可行性门表（这份输入能不能用）、一个优先级序列（能用的时候谁先上）、一个永远可用的兜底（math）。而它挑不出好的、退到最慢那条路的时候，是不会告诉你的。**

---

## 1. 这个领域解决什么问题

同一个数学式子

$$\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

在 PyTorch 里至少有五种实现：

| 实现 | 在哪跑 | 中间张量 | 一句话 |
|---|---|---|---|
| **math**（朴素） | 任意设备 | $O(S^2)$ | 物化分数矩阵，最慢但最通用，是兜底 |
| **FlashAttention** | NVIDIA SM80+ / ROCm / CPU | $O(S)$ | 分块 + 在线 softmax，不物化 $S\times S$ |
| **Memory-Efficient**（xFormers 血统） | NVIDIA SM50+ | $O(S)$ | 分块 + 自定义反向 |
| **cuDNN Attention** | NVIDIA SM80+ | $O(S)$ | cuDNN 9 的融合内核，mask 支持更宽 |
| 各家自研（AMD CK / AOTriton / 编译器生成的 kernel） | 对应硬件 | 各自 | 见第 7 节 |

**这五种实现不是「可选优化」，而是「必须挑一个」**——因为它们的输入假设互不相同：flash 不接受非空的 `attn_mask`，cuDNN 对 `head_dim` 有硬上限，mem-efficient 要求最后一维 stride 为 1，math 什么都能吃但会把显存吃光（第 6.3 节：$S=4096$ 时差 64 倍）。

于是框架必须在**每一次调用**上做一次运行时决策。这件决策有三个特点，凑在一起就成了这一期的全部内容：

1. **它是运行时做的，不是编译期的。** 换个 batch size、换个 dtype、把 mask 从 `None` 换成一个 bool 张量，出来的后端都可能不一样。
2. **它是本地的、贪心的。** 没有全局规划：挨个问「你能用吗」，第一个说能用的上。
3. **它失败时不报错。** 门表全部拒绝 flash，就悄悄退到 math——**没有 warning，没有日志，什么都没有**（第 4.3 节实测：所有降级场景的警告数都是 0）。

第 3 条是这一期真正想讲的。前两点是机制，第 3 点是代价。

### 1.1 为什么这件事值得单独讲一期

因为前面三期都被它卡过：

- **第 034 期**要跑 DSA 的稀疏 indexer，去 `SDPBackend` 里找「稀疏后端」——**没有**。枚举里只有 `ERROR / MATH / FLASH_ATTENTION / EFFICIENT_ATTENTION / CUDNN_ATTENTION / OVERRIDEABLE`。
- **第 035 期**要做 KV 驱逐，答案同样是「PyTorch 没有这个 API」。
- **第 036 期**要往 attention 里加一个 sink 列，而 `scaled_dot_product_attention` 的 schema 里**没有 sink 槽**。

三次的共同结论是「框架不帮你，你自己写 kernel」。但当时那个结论是**从枚举里看不见**得来的，是黑箱。这一期把枚举打开：这五个成员分别由谁负责、门表长什么样、什么条件下会拒绝、拒绝之后去哪里。

### 1.2 本期的四组问题

1. 怎么**观测**实际走了哪条路？（第 3 节）
2. 门表具体有哪些条件？（第 4 节，本机实测 + 上游源码逐条对照）
3. 「优先级」到底是谁的优先级？（第 5 节，实测 + 源码）
4. 静默降级的代价有多大？（第 6 节，四笔账：数值 / 训练 / 内存 / 梯度）

---

## 2. 核心思想：三层结构

打个比方：这是一家医院的**分诊台**。

- **第一层是门表（可行性）**：每个科室门前挂着牌子，写着「本诊室不接 3 岁以下 / 不接外伤 / 不接需要 24 小时监护的」。护士拿着单子一项一项对，任何一项不合，这个科室就出局。
- **第二层是优先级序列（顺序）**：护士按一个**固定顺序**去问科室。顺序不同，同样一批病人会分到不同的地方。
- **第三层是兜底（math）**：所有科室都拒收时，进观察室——**观察室永远有空床**，但那里只能做最基础的处置。

三层的性质完全不同，这是最容易混的地方：

| 层 | 形态 | 在哪 | 变化频率 |
|---|---|---|---|
| 门表 | 一堆 `bool(sdp_params, bool debug)` 的纯函数 | `sdp_utils_cpp.h` / `cuda/sdp_utils.cpp` | 换 PyTorch 版本才变 |
| 优先级序列 | 一个整数数组 | 全局 Context，可用 `torch._C._get_sdp_priority_order()` 读到 | 用户可改，**且是全局状态** |
| 兜底 | 一个 `if (userEnabledMathSDP())` | 选择循环的末尾 | 由全局开关控制 |

**门表和优先级是解耦的。** 门表说「这个后端能不能处理这份输入」，优先级说「都能处理的时候先用谁」。一个后端「不可用」和「可用但不是首选」是两件完全不同的事——混起来就会把「它被降级了」误读成「它不支持」。

### 2.1 选择循环本身只有十几行

上游 `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp` 里的 CUDA 版本（逐字引用）：

```cpp
for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::cudnn_attention:
        if (sdp::can_use_cudnn_attention(kernel_params, print_debug)) return SDPBackend::cudnn_attention;
        break;
      case SDPBackend::flash_attention:
        if (sdp::can_use_flash_attention(kernel_params, print_debug)) return SDPBackend::flash_attention;
        break;
      case SDPBackend::efficient_attention:
        if (sdp::can_use_mem_efficient_attention(kernel_params, print_debug)) return SDPBackend::efficient_attention;
        break;
      case SDPBackend::math:
        if (ctx.userEnabledMathSDP()) return SDPBackend::math;
        break;
      case SDPBackend::overrideable:
        if (ctx.userEnabledOverrideableSDP()) { TORCH_CHECK(false, "Invalid backend"); }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
}
```

三个细节值得停一下：

- **`print_debug = false` 是第一轮的默认值。** 整个循环跑一遍、全被拒之后，代码才把它置成 `true` **再跑一遍**，这时各个 `check_*` 才会 `TORCH_WARN` 打印「我为什么拒绝你」。也就是说——**正常情况下你什么都看不到；只有全军覆没时才会突然收到一堆 warning**（紧接着 `TORCH_CHECK` 抛错收场）。这个「安静 / 吵闹」的开关是理解整件事的关键。
- **math 那一支不看输入。** 它只问 `userEnabledMathSDP()`。**math 没有门表**——这就是「兜底」的含义。反过来说，一旦你关掉 math，就必须保证至少有一个融合后端能完整吃下你所有的输入，否则直接抛错。
- **先到先得。** 没有任何「收集所有候选再打分」的逻辑。排在前面而且门表放行，就它了。

---

## 3. 观测：先学会看走了哪条路

要判断「我是不是被静默降级了」，得先有观测手段。这里有两个层次，**它们看到的东西不一样**——这一点本身就很值得知道。

### 3.1 后端的枚举、默认状态、全局开关

先把「有哪些后端、默认开哪些、谁排第一」摸清楚。

```python
import torch
from torch.nn.attention import SDPBackend, _backend_names, _cur_sdpa_kernel_backends, sdpa_kernel
import torch.backends.cuda as bc

print("SDPBackend 枚举成员（值）:")
for n, m in SDPBackend.__members__.items():
    print(f"   {n:<20} {int(m):>3}")
print("_backend_names 映射:", _backend_names)

print("\n默认开启的后端:", [b.name for b in _cur_sdpa_kernel_backends()])
po = torch._C._get_sdp_priority_order()
print("默认优先级序列（全局状态）:", po, "->", [SDPBackend(x).name for x in po])

print("\n四个全局开关的 getter / setter:")
for g, s in [("flash_sdp_enabled","enable_flash_sdp"),
             ("mem_efficient_sdp_enabled","enable_mem_efficient_sdp"),
             ("math_sdp_enabled","enable_math_sdp"),
             ("cudnn_sdp_enabled","enable_cudnn_sdp")]:
    print(f"   {g}() = {getattr(bc,g)()}   设置用的是 {s}(bool)")

print("\n用 set_priority=True 把 MATH 提到第一位:")
with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION], set_priority=True):
    p2 = torch._C._get_sdp_priority_order()
    print("   上下文内:", [SDPBackend(x).name for x in p2])
    print("   上下文内开启集:", [b.name for b in _cur_sdpa_kernel_backends()])
print("   退出后恢复:", [SDPBackend(x).name for x in torch._C._get_sdp_priority_order()])
```

```text
SDPBackend 枚举成员（值）:
   ERROR                 -1
   MATH                   0
   FLASH_ATTENTION        1
   EFFICIENT_ATTENTION    2
   CUDNN_ATTENTION        3
   OVERRIDEABLE           4
_backend_names 映射: {'cudnn': 'CUDNN_ATTENTION', 'flash': 'FLASH_ATTENTION', 'mem_efficient': 'EFFICIENT_ATTENTION', 'math': 'MATH', 'overrideable': 'OVERRIDEABLE'}

默认开启的后端: ['CUDNN_ATTENTION', 'FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'MATH', 'OVERRIDEABLE']
默认优先级序列（全局状态）: [1, 2, 0, 3, 4] -> ['FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'MATH', 'CUDNN_ATTENTION', 'OVERRIDEABLE']

四个全局开关的 getter / setter:
   flash_sdp_enabled() = True   设置用的是 enable_flash_sdp(bool)
   mem_efficient_sdp_enabled() = True   设置用的是 enable_mem_efficient_sdp(bool)
   math_sdp_enabled() = True   设置用的是 enable_math_sdp(bool)
   cudnn_sdp_enabled() = True   设置用的是 enable_cudnn_sdp(bool)

用 set_priority=True 把 MATH 提到第一位:
   上下文内: ['MATH', 'FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'CUDNN_ATTENTION', 'OVERRIDEABLE']
   上下文内开启集: ['FLASH_ATTENTION', 'MATH']
   退出后恢复: ['FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'MATH', 'CUDNN_ATTENTION', 'OVERRIDEABLE']
```

两个立刻会咬人的点：

1. **`_cur_sdpa_kernel_backends()` 返回的顺序不是优先级顺序。** 它按名字字典（`_backend_names`）的插入顺序拼列表，所以打印出来是 `CUDNN_ATTENTION` 打头；而真正的优先级序列是 `[FLASH, EFFICIENT, MATH, CUDNN, OVERRIDEABLE]`。**想知道谁先上，只能看 `_get_sdp_priority_order()`。**
2. **全局开关的 getter / setter 名字不对称**：`math_sdp_enabled()` 是读，写要用 `enable_math_sdp(bool)`。写成 `bc.math_sdp_enabled(False)` 会直接 `TypeError: math_sdp_enabled() takes 0 positional arguments but 1 was given`——这个我实测踩到了。

### 3.2 两种观测层看到的东西不一样

这是本节最实用的一段。同样一次调用，用 profiler 看和用 dispatcher 看，**报出来的算子名不同，因为它们回答的是不同的问题**。

```python
import torch, torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(1)
torch.manual_seed(0)
q, k, v = (torch.randn(2, 4, 64, 32) for _ in range(3))

class Rec(TorchDispatchMode):
    """记录 dispatcher 层实际派发到的原子算子"""
    def __init__(self): self.seen = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func)); return func(*args, **(kwargs or {}))

def prof_ops(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        fn()
    return [e.key for e in p.events() if "scaled_dot_product" in e.key]

def disp_ops(fn):
    m = Rec()
    with m: fn()
    return [s for s in m.seen if any(t in s for t in ("scaled_dot_product","bmm","safe_softmax","softmax"))]

print("== 默认（不干预）==")
print("  profiler 看到:", prof_ops(lambda: F.scaled_dot_product_attention(q, k, v)))
print("  dispatcher 看到:", disp_ops(lambda: F.scaled_dot_product_attention(q, k, v)))

print("\n== sdpa_kernel([MATH]) 强制 math ==")
with sdpa_kernel([SDPBackend.MATH]):
    print("  profiler 看到:", prof_ops(lambda: F.scaled_dot_product_attention(q, k, v)))
    print("  dispatcher 看到:", disp_ops(lambda: F.scaled_dot_product_attention(q, k, v)))

print("\n== 绕开选择器，直接调底层算子 ==")
o1, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(q, k, v)
print("  返回:", type(o1).__name__, "output", tuple(o1.shape), "| logsumexp", tuple(lse.shape))
print("  与 F.scaled_dot_product_attention 的最大差:",
      (o1 - F.scaled_dot_product_attention(q, k, v)).abs().max().item())

print("\n== 都不许用时会怎样 ==")
import torch.backends.cuda as bc
bc.enable_math_sdp(False); bc.enable_flash_sdp(False)
try:
    F.scaled_dot_product_attention(q, k, v)
except RuntimeError as e:
    print("  RuntimeError:", str(e).split("\n")[0][:110])
finally:
    bc.enable_math_sdp(True); bc.enable_flash_sdp(True)
```

```text
== 默认（不干预）==
  profiler 看到: ['aten::scaled_dot_product_attention', 'aten::_scaled_dot_product_flash_attention_for_cpu']
  dispatcher 看到: ['aten._scaled_dot_product_flash_attention_for_cpu.default']

== sdpa_kernel([MATH]) 强制 math ==
  profiler 看到: ['aten::scaled_dot_product_attention', 'aten::_scaled_dot_product_attention_math']
  dispatcher 看到: ['aten.bmm.default', 'aten._safe_softmax.default', 'aten.bmm.default']

== 绕开选择器，直接调底层算子 ==
  返回: Tensor output (2, 4, 64, 32) | logsumexp (2, 4, 64)
  与 F.scaled_dot_product_attention 的最大差: 0.0

== 都不许用时会怎样 ==
  RuntimeError: No viable backend for scaled_dot_product_attention was found. This is likely due to turning off both the math 
```

四件事：

1. **profiler 看到的是「逻辑算子」，dispatcher 看到的是「原子算子」。** 强制 math 时 profiler 报 `aten::_scaled_dot_product_attention_math` 一条，dispatcher 层看到的是 `bmm → _safe_softmax → bmm` 三条。原因是 `_scaled_dot_product_attention_math` 是个**复合算子**（composite：它自己就是一段由原子算子拼成的实现），profiler 的 RecordFunction 挂在复合算子的入口上；而 `TorchDispatchMode` 拦在派发层，看到的是真正下发给 kernel 的那些原子算子。
   两个都不会骗你，问的问题不同：**「框架认为它走了哪条路」问 profiler；「硬件实际执行了什么」问 dispatcher。**
2. **底层算子可以直接调。** `torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(q,k,v)` 返回 `(output, logsumexp)` 二元组，`output` 与 `F.scaled_dot_product_attention` 的结果**逐比特相同（最大差 0.0）**。生产工程普遍这么干（推理引擎直接调 flash kernel 而不走 SDPA），代价是你得自己负责所有门表检查。
   注意第二个返回值：`logsumexp` 形状 `[B, H, S]`。**融合后端会把 LSE 顺手给你**——这是它与 math 的又一处接口差异（反向要用它，chunked prefill 类算法也要用它）。
3. **把所有后端都关掉，报错信息其实是准确的**：`No viable backend for scaled_dot_product_attention was found.` 这是整套机制里**唯一一处失败会发声**的地方。
4. **强制 math 完全生效**（`[MATH]` 那一行），而且结果与默认路径**不同**——数值差异第 6.1 节算。

### 3.3 顺带把索引名也列清楚

`torch._C` 里跟这件事相关的读写接口，成对性并不整齐：

```python
import torch
names = ["_get_sdp_priority_order", "_set_sdp_priority_order",
         "_get_overrideable_sdp_enabled", "_set_overrideable_sdp_enabled",
         "_get_flash_sdp_enabled", "_set_flash_sdp_enabled",
         "_get_mem_efficient_sdp_enabled", "_set_mem_efficient_sdp_enabled",
         "_get_math_sdp_enabled", "_set_math_sdp_enabled",
         "_get_cudnn_sdp_enabled", "_set_cudnn_sdp_enabled",
         "_get_fa3_sdp_enabled", "_set_fa3_sdp_enabled",
         "_get_fa4_sdp_enabled", "_set_fa4_sdp_enabled"]
print("torch._C 里与后端选择有关的开关（True = 本构建暴露了）")
for i in range(0, len(names), 2):
    g, s = names[i], names[i + 1]
    print(f"   {g:<34}{str(hasattr(torch._C, g)):<6}   {s:<34}{hasattr(torch._C, s)}")
gets = [n for n in names if n.startswith("_get") and "priority" not in n and hasattr(torch._C, n)]
print("\n这些开关的当前值:", {n[5:].replace("_sdp_enabled", ""): getattr(torch._C, n)() for n in gets})
import torch._inductor.config as ic
print("\nInductor 里名字带 attention / sdpa / flex 的配置项:")
print("  ", [k for k in dir(ic) if any(t in k.lower() for t in ("attention", "sdpa", "flex"))])
```

```text
torch._C 里与后端选择有关的开关（True = 本构建暴露了）
   _get_sdp_priority_order           True     _set_sdp_priority_order           True
   _get_overrideable_sdp_enabled     True     _set_overrideable_sdp_enabled     False
   _get_flash_sdp_enabled            True     _set_flash_sdp_enabled            False
   _get_mem_efficient_sdp_enabled    True     _set_mem_efficient_sdp_enabled    False
   _get_math_sdp_enabled             True     _set_math_sdp_enabled             False
   _get_cudnn_sdp_enabled            True     _set_cudnn_sdp_enabled            False
   _get_fa3_sdp_enabled              True     _set_fa3_sdp_enabled              False
   _get_fa4_sdp_enabled              False    _set_fa4_sdp_enabled              False

这些开关的当前值: {'overrideable': True, 'flash': True, 'mem_efficient': True, 'math': True, 'cudnn': True, 'fa3': False}

Inductor 里名字带 attention / sdpa / flex 的配置项:
   []
```

结论：**单个后端的 `_set_*` 一个都没暴露到 `torch._C`**，唯一的 setter 是 `_set_sdp_priority_order`。要开/关某个后端，走 `torch.backends.cuda.enable_*()` 或 `sdpa_kernel()` 上下文管理器。另外注意 **FA4 的开关在这台机器的构建里连 getter 都没有**（FA3 有 getter 没有 setter），所以「用 FA4 跑一把」这个动作在本机构建下不成立——`check_fa4_constraints` 只能通过 C++ 侧的 `userEnabledFA4SDP()` 改（源码里它读的是 global context）。

---

## 4. 门表实测：什么条件下会被降级

这是本期的核心实验。做法：用 3.2 节的 dispatcher 观测法，对一组配置逐个问「这一次实际走了哪条路」，并同时记录**有没有 warning**。

```python
import warnings, torch, torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_num_threads(1)
torch.manual_seed(0)
B, H, S, D = 2, 4, 64, 32

class Rec(TorchDispatchMode):
    def __init__(self): self.seen = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func)); return func(*args, **(kwargs or {}))

def probe(label, fn):
    m = Rec()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            with m: fn()
            err = ""
        except Exception as e:
            err = type(e).__name__
    if any("_scaled_dot_product_flash_attention_for_cpu" in s for s in m.seen):
        be = "flash_for_cpu"
    elif any("bmm" in s for s in m.seen):
        be = "math"
    else:
        be = "-"
    nw = sum(1 for x in w if issubclass(x.category, (UserWarning, RuntimeWarning)))
    tail = f" [{err}]" if err else ""
    print(f"{label:<34} {be:<13} 警告 {nw}{tail}")

def mk(dtype=torch.float32, D2=D, contig=True):
    q = torch.randn(B, H, S, D2).to(dtype)
    if not contig:
        q = q.transpose(2, 3).contiguous().transpose(2, 3)
    return q, torch.randn(B, H, S, D2).to(dtype), torch.randn(B, H, S, D2).to(dtype)

print("配置                               实际后端        警告数")
print("-" * 62)
q, k, v = mk()
probe("fp32 / 无 mask", lambda: F.scaled_dot_product_attention(q, k, v))
probe("fp32 / is_causal", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
probe("fp32 / bool 2D mask", lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(S, S, dtype=torch.bool).tril()))
probe("fp32 / float 4D mask", lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=torch.zeros(B, H, S, S)))
probe("fp32 / dropout_p=0.1", lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.1))
for dt in (torch.float16, torch.bfloat16, torch.float64):
    a, b, c = mk(dt)
    probe(f"{str(dt).replace('torch.','')} / 无 mask", lambda a=a, b=b, c=c: F.scaled_dot_product_attention(a, b, c))
for D2 in (16, 64, 256, 272, 512):
    a, b, c = mk(D2=D2)
    probe(f"fp32 / head_dim={D2}", lambda a=a, b=b, c=c: F.scaled_dot_product_attention(a, b, c))
qc, kc, vc = mk(contig=False)
probe(f"fp32 / 非连续 q (stride={qc.stride()[-1]})", lambda: F.scaled_dot_product_attention(qc, kc, vc))
probe("fp32 / 3D 无 batch 维", lambda: F.scaled_dot_product_attention(q[0], k[0], v[0]))
qg = torch.randn(B, H, S, D); kg = torch.randn(B, 1, S, D)
probe("fp32 / enable_gqa=True (H:1)", lambda: F.scaled_dot_product_attention(qg, kg, kg, enable_gqa=True))
```

```text
配置                               实际后端        警告数
--------------------------------------------------------------
fp32 / 无 mask                      flash_for_cpu 警告 0
fp32 / is_causal                   flash_for_cpu 警告 0
fp32 / bool 2D mask                flash_for_cpu 警告 0
fp32 / float 4D mask               flash_for_cpu 警告 0
fp32 / dropout_p=0.1               math          警告 0
float16 / 无 mask                   flash_for_cpu 警告 0
bfloat16 / 无 mask                  flash_for_cpu 警告 0
float64 / 无 mask                   flash_for_cpu 警告 0
fp32 / head_dim=16                 flash_for_cpu 警告 0
fp32 / head_dim=64                 flash_for_cpu 警告 0
fp32 / head_dim=256                flash_for_cpu 警告 0
fp32 / head_dim=272                flash_for_cpu 警告 0
fp32 / head_dim=512                flash_for_cpu 警告 0
fp32 / 非连续 q (stride=64)           math          警告 0
fp32 / 3D 无 batch 维                flash_for_cpu 警告 0
fp32 / enable_gqa=True (H:1)       flash_for_cpu 警告 0
```

### 4.1 读这张表

**18 个配置里只有 2 个降级，而且都是「非数值」的原因：**

| 降级原因 | 配置 |
|---|---|
| `dropout > 0` | `dropout_p=0.1` |
| 最后一维 stride ≠ 1 | 非连续 q（stride 为 64） |

其余全部走 `flash_for_cpu`，包括几个**跟 CUDA 直觉完全相反**的：

- **`float64` 走 flash。** 在 CUDA 上 flash 只吃 fp16/bf16（SM80+ 的允许集是 `{kHalf, kBFloat16}`），但在 CPU 上 `cpp_supported_flash_dtypes = {kFloat, kDouble, kBFloat16, kHalf}`——**fp64 是明写在允许集里的**。
- **`head_dim` 从 16 到 512 全部走 flash。** CUDA 侧 flash 的硬上限是 `head_dim <= 256`；CPU 侧的 `check_head_dim_size_cpp` **只检查 q/k/v 的最后一维相等，没有任何上界**。
- **`attn_mask` 无论 bool 还是 float 4D 都走 flash。** CUDA 侧 flash 的 `check_for_attn_mask` 是「只要 `attn_mask.has_value()` 就拒绝」；CPU 侧这条约束叫 `check_attn_mask_shape`——**它检查形状，不检查有无**。
- **`enable_gqa=True` 走 flash**，flash 门表里 GQA 是显式支持的。
- **3D 输入（没有 batch 维）走 flash**，因为有个 `normalize_unbatched_input` 会先给补一个单例 batch 维。

**为什么 CPU 和 CUDA 的门表差这么多？** 因为 `atorch` 的 CPU 路径有**自己独立的一套门表和优先级**。上游 `sdp_utils_cpp.cpp` 里逐字写着：

```cpp
std::array<SDPBackend, num_backends> priority_order_cpp(sdp_params const& params) {
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::flash_attention,
      SDPBackend::math};
  return default_order;
}
```

**只有两项。** 在 CPU 上，「flash / mem-efficient / cuDNN」这条三选一的故事不存在——它退化成 **flash 或 math 二选一**。这也解释了第 3 节里那个观测结果：在 CPU 上 `SDPBackend.EFFICIENT_ATTENTION` 和 `CUDNN_ATTENTION` 从来没有被选中过。

CPU 侧的 `use_flash_attention_cpp` 约束序列（逐字摘录顺序）：

```cpp
constexpr auto constraints = std::to_array<bool (*)(sdp_params const&, bool)>(
    {check_runtime_disabled_flash,
     check_nested_tensor,
     check_for_dropout,
     check_tensor_shapes,
     check_batch_size_and_num_heads_dense<true, true, true>,
     check_attn_mask_shape,
     check_head_dim_size_cpp,
     check_nonzero_sequence_lengths_dense,
     check_last_dim_stride_equals_1_dense<false>});
```

对照实测：**`check_for_dropout` 和 `check_last_dim_stride_equals_1_dense` 就是那 2 个降级**，其它 7 条在这批配置下都放行了。源码与实测逐条对上。

### 4.2 「静默」这个词的准确含义

**18 行里，警告数全是 0。**

这比它看起来更值得强调。`select_sdp_backend` 里那套 debug 机制是有设计意图的：**它会给你一个详尽的拒绝理由，但只在「所有后端都拒绝」的时候给。** 只要 math 还开着（默认就开着），总有一个后端能接住——于是那条 warning 永远不会触发。

所以工程上的准确表述是：

> **「降级」在 PyTorch 里不是异常，不是 warning，是一个正常返回值。**

要发现它，只有两个办法：**自己观测实际算子**（3.2 节），或者**跑 benchmark 发现这条路径莫名其妙比预期慢**。第二条是绝大多数团队实际采用的路径，也是绝大多数团队发现问题时已经浪费了两周的原因。

### 4.3 兜底不是「安全网」

一个容易产生的误解：既然 math 什么都能接，那写错了也总会跑通。

**不是。** 门表能拒掉的是「不支持」，拒不掉「错误」。看这两个边界（第 11 节练习会用到）：

```python
import torch, torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_num_threads(1); torch.manual_seed(0)
class Rec(TorchDispatchMode):
    def __init__(s): s.seen = []
    def __torch_dispatch__(s, f, t, a=(), k=None):
        s.seen.append(str(f)); return f(*a, **(k or {}))
def which(fn):
    m = Rec()
    try:
        with m: fn()
    except RuntimeError as e:
        return "RuntimeError: " + str(e).split(".")[0][:36]
    if any("flash_attention_for_cpu" in s for s in m.seen): return "flash_for_cpu"
    if any("bmm" in s for s in m.seen): return "math"
    return "其它路径"

B, H, S, D = 2, 4, 64, 32
q, k, v = (torch.randn(B, H, S, D) for _ in range(3))

def nc(t):
    return t.transpose(2, 3).contiguous().transpose(2, 3)

rows = []
q1 = torch.randn(B, H, S, 1)
rows.append(("head_dim=1（连续）", lambda: F.scaled_dot_product_attention(q1, q1, q1)))
rows.append((f"只有 k 非连续（stride[-1]={nc(k).stride()[-1]}）", lambda: F.scaled_dot_product_attention(q, nc(k), v)))
rows.append((f"只有 v 非连续（stride[-1]={nc(v).stride()[-1]}）", lambda: F.scaled_dot_product_attention(q, k, nc(v))))
rows.append(("3D 输入 + dropout=0.1", lambda: F.scaled_dot_product_attention(q[0], k[0], v[0], dropout_p=0.1)))
rows.append(("float 4D mask + dropout=0.1", lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=torch.zeros(B, H, S, S), dropout_p=0.1)))
rows.append(("5 维输入", lambda: F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))))
rows.append(("q 的 head_dim 与 k 不同 (32 vs 16)", lambda: F.scaled_dot_product_attention(q, torch.randn(B, H, S, 16), v)))

print(f"{'边界配置':<44}{'实际后端':>18}")
print("-" * 62)
for label, fn in rows:
    print(f"{label:<44}{which(fn):>18}")

print("\nS=16384 / B=1 / H=32 / D=128 的中间张量（fp16）：")
sq = 1 * 32 * 16384 * 16384 * 2
lin = 1 * 32 * 16384 * 128 * 2
print(f"   B·H·S²·2  = {sq:>18,}  = {sq/2**30:.2f} GiB")
print(f"   B·H·S·D·2 = {lin:>18,}  = {lin/2**30:.2f} GiB")
print(f"   比值 = {sq//lin}×")
```

```text
边界配置                                                      实际后端
--------------------------------------------------------------
head_dim=1（连续）                                   flash_for_cpu
只有 k 非连续（stride[-1]=64）                                   math
只有 v 非连续（stride[-1]=64）                                   math
3D 输入 + dropout=0.1                                       math
float 4D mask + dropout=0.1                               math
5 维输入                                                     math
q 的 head_dim 与 k 不同 (32 vs 16)              RuntimeError: Expected size for first two dimensio

S=16384 / B=1 / H=32 / D=128 的中间张量（fp16）：
   B·H·S²·2  =     17,179,869,184  = 16.00 GiB
   B·H·S·D·2 =        134,217,728  = 0.12 GiB
   比值 = 128×
```

注意最后一行：**q 与 k 的 head_dim 不一致时，flash 被 `check_head_dim_size_cpp` 拒掉（它检查 q/k/v 最后一维相等），math 接住，然后在 `bmm` 里以一个 matmul 形状错误炸出来。** 你看到的报错和花在这条路径上的时间，都与「后端选择」毫无关系——这是一个非常常见的排查方向性错误。

另外注意 `head_dim=1` 那一行：它是连续的，所以 stride 检查放行；`check_head_dim_size_cpp` 也不管大小——**head_dim=1 走的是 flash**。而在 CUDA 上 mem-efficient 的 `minimum_gemm_alignment` 会要求对齐到 8（fp16/bf16）或 4（fp32）。同一份输入、换台机器、换条路。

---

## 5. 优先级序列到底是谁的优先级

第 3.1 节用 `set_priority=True` 把 MATH 提到了优先级第一位。那它**真的会改变选择**吗？

```python
import torch, torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(1)
torch.manual_seed(0)
q, k, v = (torch.randn(2, 4, 64, 32) for _ in range(3))

class Rec(TorchDispatchMode):
    def __init__(self): self.seen = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func)); return func(*args, **(kwargs or {}))

def which():
    m = Rec()
    try:
        with m: F.scaled_dot_product_attention(q, k, v)
    except RuntimeError:
        return "RuntimeError"
    for s in m.seen:
        if "flash_attention_for_cpu" in s: return "flash_for_cpu"
        if "bmm" in s: return "math"
    return "-"

def case(label, ctx):
    if ctx is None:
        print(f"{label:<50}{which():>15}")
    else:
        with ctx:
            print(f"{label:<50}{which():>15}")

print(f"{'传给 sdpa_kernel 的参数':<50}{'实际后端':>15}")
print("-" * 65)
case("（不干预）", None)
case("[FLASH_ATTENTION]", sdpa_kernel([SDPBackend.FLASH_ATTENTION]))
case("[MATH]", sdpa_kernel([SDPBackend.MATH]))
case("[FLASH_ATTENTION, MATH]", sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]))
case("[MATH, FLASH_ATTENTION]", sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]))
case("[MATH, FLASH] + set_priority=True", sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION], set_priority=True))
case("[FLASH, MATH] + set_priority=True", sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH], set_priority=True))
case("[EFFICIENT_ATTENTION, MATH]", sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]))
case("[EFFICIENT_ATTENTION]", sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]))
case("[CUDNN_ATTENTION, MATH]", sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]))
```

```text
传给 sdpa_kernel 的参数                                           实际后端
-----------------------------------------------------------------
（不干预）                                               flash_for_cpu
[FLASH_ATTENTION]                                   flash_for_cpu
[MATH]                                                       math
[FLASH_ATTENTION, MATH]                             flash_for_cpu
[MATH, FLASH_ATTENTION]                             flash_for_cpu
[MATH, FLASH] + set_priority=True                   flash_for_cpu
[FLASH, MATH] + set_priority=True                   flash_for_cpu
[EFFICIENT_ATTENTION, MATH]                                  math
[EFFICIENT_ATTENTION]                                RuntimeError
[CUDNN_ATTENTION, MATH]                                      math
```

三行结论，每一行都能对着源码验证：

1. **列表顺序 ≠ 优先级顺序。** `[MATH, FLASH_ATTENTION]` 和 `[FLASH_ATTENTION, MATH]` 的结果一模一样（都走 flash）。因为不带 `set_priority` 时，列表只是个**集合**——决定「谁先上」的仍然是那个全局优先级序列。
2. **`set_priority=True` 在本机上是 no-op。** 把 MATH 放到列表第一位、并且 `set_priority=True` 改写了全局序列（3.1 节已实测改成了 `['MATH','FLASH_ATTENTION',...]`），**结果还是走 flash**。原因是源码里 CPU 路径的 `priority_order_cpp` 返回的是一个**硬编码常量** `{flash_attention, math}`，它根本不读全局序列。
   **这是一条实测与源码互相印证的重要结论：CUDA 路径用全局优先级序列，CPU 路径用硬编码。** 上游 CUDA 侧确实是 `return at::globalContext().sDPPriorityOrder();`，而 CPU 侧是 `return default_order;`。
3. **不可用的后端 + 可用的 math = 静默用 math**（`[EFFICIENT_ATTENTION, MATH]`、`[CUDNN_ATTENTION, MATH]` 都落到 math）；**不可用的后端 + 没有 math = 抛错**（`[EFFICIENT_ATTENTION]`）。这两行的对比精确刻画了「兜底」的角色：**math 不是备选之一，是地板。**

### 5.1 CUDA 侧：cuDNN 会在满足条件时被插到第一位

上游 CUDA 源码里，默认顺序本来是 `flash → efficient → math`，但有一个**开机时执行一次**的重排：

```cpp
std::array<SDPBackend, num_backends> priority_order(sdp_params const& params) {
  if (!priority_order_init_) {
    priority_order_init_ = true;
    if (check_prefer_cudnn_attention()) {
        const std::vector<int64_t> cudnn_order = {
            static_cast<int64_t>(at::SDPBackend::cudnn_attention),
            static_cast<int64_t>(at::SDPBackend::flash_attention),
            static_cast<int64_t>(at::SDPBackend::efficient_attention),
            static_cast<int64_t>(at::SDPBackend::math)};
        at::globalContext().setSDPPriorityOrder(cudnn_order);
    }
  }
  return at::globalContext().sDPPriorityOrder();
}
```

`check_prefer_cudnn_attention()` 读环境变量 `TORCH_CUDNN_SDPA_DEPRIORITIZED`（设了就放弃提升），并要求 **cuDNN 版本 > 9.15.00 且 SM major 是 9 或 10**；CUDA < 13 时源码还额外交了一句 `(!minor || minor == 3)`——**即 minor 必须是 0 或 3**，所以实际命中的是 SM 9.0 / 10.0 / 10.3 这几种。

**这几条串起来解释了一个容易困惑的现象**：同一份模型代码，在 H100 上跑的是 cuDNN 的 attention kernel、在 A100 上跑的是 FlashAttention——**不是你的代码换了，是开机时那个重排换了。** 而 `_get_sdp_priority_order()` 是唯一能直接看到它的入口。

（本机实测的序列 `[1, 2, 0, 3, 4] = FLASH, EFFICIENT, MATH, CUDNN, OVERRIDEABLE` 里，cuDNN 落在 math **之后**——因为这台机器没有编进 CUDA/cuDNN，提升条件不成立。这和源码逻辑一致。）

---

## 6. 静默降级的代价

降级了会怎样？四笔账，逐笔量。

### 6.1 第一笔：数值 —— 换后端就是换了一个数值实现

```python
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(1)
torch.manual_seed(0)
B, H, S, D = 4, 8, 256, 64
q, k, v = (torch.randn(B, H, S, D) for _ in range(3))

# 参考解：fp64 + 强制 math（最接近定义式的实现）
with sdpa_kernel([SDPBackend.MATH]):
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)
ref32 = ref.float()

def run(force_math, causal=True):
    if force_math:
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

print("同一份输入，三个实现的差异（参考 = fp64 + math）")
print(f"{'实现':<22}{'max|Δ|':>13}{'相对误差 nMSE':>16}")
print("-" * 51)
for name, force in (("fp32 + flash_for_cpu", False), ("fp32 + math", True)):
    o = run(force)
    d = (o - ref32)
    nmse = (d.double() ** 2).sum().item() / (ref32.double() ** 2).sum().item()
    print(f"{name:<22}{d.abs().max().item():>13.3e}{nmse:>16.3e}")

print("\n两个 fp32 实现彼此之间的差：")
d = (run(False) - run(True))
nmse = (d.double() ** 2).sum().item() / (run(True).double() ** 2).sum().item()
print(f"   max|Δ| = {d.abs().max().item():.3e}   nMSE = {nmse:.3e}")

print("\nfp32 的机器 epsilon 量级对比：")
print(f"   eps(fp32) = {torch.finfo(torch.float32).eps:.3e}   输出典型幅值 = {ref32.abs().mean().item():.3f}")
print("\n聚合量（下游 loss 只看到聚合值，看看差多少）：")
print(f"   sum 差 = {abs(run(False).sum().item() - run(True).sum().item()):.3e}")
print(f"   mean 差 = {abs(run(False).mean().item() - run(True).mean().item()):.3e}")
```

```text
同一份输入，三个实现的差异（参考 = fp64 + math）
实现                           max|Δ|       相对误差 nMSE
---------------------------------------------------
fp32 + flash_for_cpu      9.537e-07       5.862e-14
fp32 + math               9.537e-07       5.923e-14

两个 fp32 实现彼此之间的差：
   max|Δ| = 7.153e-07   nMSE = 5.743e-14

fp32 的机器 epsilon 量级对比：
   eps(fp32) = 1.192e-07   输出典型幅值 = 0.142

聚合量（下游 loss 只看到聚合值，看看差多少）：
   sum 差 = 1.831e-04
   mean 差 = 3.492e-10
```

读法（**这里有一个反直觉的结论**）：

- **两个 fp32 实现离 fp64 参考一样近**（nMSE 分别是 5.862e-14 与 5.923e-14，同量级）。所以「flash 比 math 不准」这个说法在本机 fp32 下**不成立**——两者都只是被 fp32 的舍入精度限制住了。
- **但它们彼此不同**（`max|Δ| = 7.153e-07`，约 6 倍 `eps(fp32)`；相对误差 nMSE 5.743e-14）。而且这个差异**就在 fp32 舍入地板附近**，不是谁「算错了」。
- **聚合量上几乎看不见**：整块的 `mean` 差是 3.492e-10，而 `sum` 差是 1.831e-04（因为 sum 是 $4\times8\times256\times64 = 524288$ 个元素的和，逐元素的舍入会累积）。

所以第一笔账的准确表述是：**换后端 ≠ 得到不同质量的答案，而是得到同一个答案的另一个舍入样本。** 这个扰动有多小？看第二笔账。

### 6.2 第二笔：训练 —— loss 读数骗你，参数不会

这一节的做法是**同种子、同数据、同步数，唯一变量是 SDPA 走哪条路**，并且**加一个对照组**（两次都走默认路径）。

```python
import torch, torch.nn as nn, torch.nn.functional as F, time
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(4)
V, T, B, D, NH, STEPS = 32, 64, 16, 64, 4, 800

g = torch.Generator().manual_seed(7)
Pm = torch.rand(V, V, generator=g); Pm = Pm / Pm.sum(-1, keepdim=True); cum = Pm.cumsum(-1)
N = 33000; u = torch.rand(N, generator=g)
toks = torch.zeros(N, dtype=torch.long); cur = 0
for i in range(N):
    cur = int(torch.searchsorted(cum[cur], u[i]).clamp(max=V - 1)); toks[i] = cur
data = toks[: N // (T + 1) * (T + 1)].view(-1, T + 1)
BATCHES = [(s * 37) % (data.shape[0] - B) for s in range(STEPS)]

class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D)
    def forward(s, x):
        b, t, _ = x.shape
        q, k, v = s.qkv(x).chunk(3, -1)
        q, k, v = (z.view(b, t, NH, D // NH).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # ← 后端在这里决定
        return s.o(y.transpose(1, 2).reshape(b, t, D))

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1 = nn.LayerNorm(D); s.at = Attn(); s.ln2 = nn.LayerNorm(D)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x):
        x = x + s.at(s.ln1(x)); return x + s.mlp(s.ln2(x))

class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.te = nn.Embedding(V, D); s.b = nn.ModuleList([Block(), Block()])
        s.lnf = nn.LayerNorm(D); s.head = nn.Linear(D, V, bias=False)
    def forward(s, idx):
        x = s.te(idx)
        for blk in s.b: x = blk(x)
        return s.head(s.lnf(x))

def run(force_math, seed):
    torch.manual_seed(seed)
    model = LM(); opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(); marks, snaps = {}, {}
    for step in range(STEPS):
        i = BATCHES[step]; xb, yb = data[i:i+B, :-1], data[i:i+B, 1:]
        opt.zero_grad()
        if force_math:
            with sdpa_kernel([SDPBackend.MATH]):
                l = lossf(model(xb).view(-1, V), yb.reshape(-1))
        else:
            l = lossf(model(xb).view(-1, V), yb.reshape(-1))
        l.backward(); opt.step()
        if step + 1 in (10, 100, 200, 400, 800):
            marks[step + 1] = l.item()
            snaps[step + 1] = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    return marks, snaps

def rel(a, b):
    return (a - b).norm().item() / b.norm().item()

print("A = 默认路径(flash_for_cpu)，B = 强制 math。对照组 = 两次都走默认路径。")
print(f"{'seed':<5}{'步数':>6}{'A loss':>13}{'B loss':>13}{'|Δloss|':>10}"
      f"{'对照组 ‖Δθ‖/‖θ‖':>19}{'实验组 ‖Δθ‖/‖θ‖':>19}")
print("-" * 85)
t0 = time.time()
for seed in (0, 1, 2):
    mA, sA = run(False, seed)
    mB, sB = run(False, seed)      # 对照组：同一个后端
    mC, sC = run(True,  seed)      # 实验组：换后端
    for st in (10, 100, 200, 400, 800):
        print(f"{seed:<5}{st:>6}{mA[st]:>13.9f}{mC[st]:>13.9f}{abs(mC[st]-mA[st]):>10.2e}"
              f"{rel(sB[st], sA[st]):>19.3e}{rel(sC[st], sA[st]):>19.3e}")
```

```text
A = 默认路径(flash_for_cpu)，B = 强制 math。对照组 = 两次都走默认路径。
seed     步数       A loss       B loss   |Δloss|       对照组 ‖Δθ‖/‖θ‖       实验组 ‖Δθ‖/‖θ‖
-------------------------------------------------------------------------------------
0        10  3.432523966  3.432523966  0.00e+00          0.000e+00          7.850e-06
0       100  3.294420958  3.294420958  0.00e+00          0.000e+00          1.968e-05
0       200  3.289544106  3.289544106  0.00e+00          0.000e+00          7.686e-05
0       400  3.017071486  3.017071009  4.77e-07          0.000e+00          3.012e-04
0       800  2.422361374  2.422362566  1.19e-06          0.000e+00          1.335e-03
1        10  3.419694901  3.419694901  0.00e+00          0.000e+00          6.685e-06
1       100  3.296090126  3.296089649  4.77e-07          0.000e+00          2.276e-05
1       200  3.298892260  3.298892260  0.00e+00          0.000e+00          6.852e-05
1       400  3.046492815  3.046492815  0.00e+00          0.000e+00          3.117e-04
1       800  2.455093384  2.455094337  9.54e-07          0.000e+00          1.360e-03
2        10  3.426055431  3.426055431  0.00e+00          0.000e+00          8.229e-06
2       100  3.296395540  3.296395540  0.00e+00          0.000e+00          2.234e-05
2       200  3.289067030  3.289067030  0.00e+00          0.000e+00          6.848e-05
2       400  3.037135839  3.037136078  2.38e-07          0.000e+00          3.163e-04
2       800  2.448451519  2.448484182  3.27e-05          0.000e+00          1.247e-03
```

这是本期最硬的一条结果，三层读法：

**第一层：对照组恰好是 0。** 三个种子 × 五个检查点，同一后端跑两遍的 `‖Δθ‖/‖θ‖` **15 个读数全部是 `0.000e+00`**——**在这个线程数下训练是逐比特可复现的**。这一步是必须的：没有它，后面那些 1e-5 量级的差异根本没法归因，因为你不知道它们是后端差异还是线程调度噪声。**对照组为 0 是这一节所有推论的前提。**

**第二层：扰动一旦注入就单调放大。** 换后端注入的相对扰动在 10 步时是 7.85e-06，然后 1.97e-05 → 7.69e-05 → 3.01e-04 → 1.34e-03。**800 步放大了约 170 倍**（三个种子的数量级一致：3.0e-4 / 3.1e-4 / 3.2e-4 在 400 步，1.25e-3~1.36e-3 在 800 步）。这是一个正的李雅普诺夫指数的直接体现：训练动力学本身把微扰往外推。

**第三层：loss 读数几乎完全看不见这件事。** 前 200 步三个种子的 loss 在 **9 位小数下完全相同**（`|Δloss|` 恰为 0）；到 400 步才出现 4.77e-07 这种量级，800 步时是 1.19e-06 / 9.54e-07 / 3.27e-05。**用 `%.4f` 打印，三个种子的最终 loss 看起来一模一样**；要到小数第 5~6 位才露出痕迹（seed 2：`2.44845` vs `2.44848`）。而同时参数已经分开了 0.13%。

> 这一条和**第 036 期的「指标与行为脱钩」是同一形状**：036 是 sink 质量指标冻结而行为代价涨 650 倍；这里是 loss 读数冻结而参数距离涨 170 倍。**共同教训是同一个：不要用一个聚合标量去判断一个扰动的存在性。**

工程含义很直接：**你在 A100 上训到第 5000 步的 checkpoint，和同事在 H100 上从同一个 checkpoint 接着训，两个模型的参数不是同一组。** 这不是 bug，也不是谁错了——但如果你在做 A/B 对照、要做 bitwise 复现、或者要排查一个只在某个机器上出现的 loss spike，这个量级必须提前知道。

### 6.3 第三笔：内存 —— $O(S)$ 和 $O(S^2)$ 的实测分界

这笔账和**第 035 期那本「字节账」是同一本**。用一个能精确归属到算子的内存 profiling：

```python
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(1)
torch.manual_seed(0)
B, H, D = 1, 8, 64
M_OP = "aten::_scaled_dot_product_attention_math"
F_OP = "aten::_scaled_dot_product_flash_attention_for_cpu"

def alloc(opname, S, force_math):
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    fn = lambda: F.scaled_dot_product_attention(q, k, v)
    fn(); fn()                                   # 预热，把一次性初始化开销排掉
    with torch.profiler.profile(profile_memory=True) as p:
        if force_math:
            with sdpa_kernel([SDPBackend.MATH]): fn()
        else:
            fn()
    for e in p.key_averages():
        if e.key == opname:
            return e.cpu_memory_usage
    return None

print(f"B={B}, H={H}, D={D}, fp32。cpu_memory_usage = 该算子链分配的总字节数")
print(f"{'S':>6}{'B·H·S²·4 (O(S²))':>18}{'B·H·S·(D+1)·4 (O(S))':>22}{'math 实测':>14}{'flash_for_cpu 实测':>20}")
print("-" * 82)
for S in (256, 512, 1024, 2048, 4096):
    q2 = B * H * S * S * 4
    lin = B * H * S * (D + 1) * 4
    print(f"{S:>6}{q2:>18,}{lin:>22,}{alloc(M_OP, S, True):>14,}{alloc(F_OP, S, False):>20,}")

print("\n与解析式的逐档核对（实测 ÷ 解析）：")
for S in (256, 512, 1024, 2048, 4096):
    q2 = B * H * S * S * 4; lin = B * H * S * (D + 1) * 4
    a = alloc(M_OP, S, True); b = alloc(F_OP, S, False)
    print(f"  S={S:<5} math 实测/（S²+线性项）= {a/(q2+B*H*S*D*4):.6f}   flash 实测/线性项 = {b/lin:.6f}")

print("\n相邻两档倍率（S 翻倍时 O(S) 应 ×2，O(S²) 应 ×4）：")
prev = None
for S in (512, 1024, 2048, 4096):
    a = alloc(M_OP, S, True); b = alloc(F_OP, S, False)
    if prev:
        print(f"  S {prev[0]:>5}→{S:<5} math ×{a/prev[1]:.2f}    flash_for_cpu ×{b/prev[2]:.2f}")
    prev = (S, a, b)
```

```text
B=1, H=8, D=64, fp32。cpu_memory_usage = 该算子链分配的总字节数
     S  B·H·S²·4 (O(S²))  B·H·S·(D+1)·4 (O(S))       math 实测    flash_for_cpu 实测
----------------------------------------------------------------------------------
   256         2,097,152               532,480     2,621,440             532,480
   512         8,388,608             1,064,960     9,437,184           1,064,960
  1024        33,554,432             2,129,920    35,651,584           2,129,920
  2048       134,217,728             4,259,840   138,412,032           4,259,840
  4096       536,870,912             8,519,680   545,259,520           8,519,680

与解析式的逐档核对（实测 ÷ 解析）：
  S=256   math 实测/（S²+线性项）= 1.000000   flash 实测/线性项 = 1.000000
  S=512   math 实测/（S²+线性项）= 1.000000   flash 实测/线性项 = 1.000000
  S=1024  math 实测/（S²+线性项）= 1.000000   flash 实测/线性项 = 1.000000
  S=2048  math 实测/（S²+线性项）= 1.000000   flash 实测/线性项 = 1.000000
  S=4096  math 实测/（S²+线性项）= 1.000000   flash 实测/线性项 = 1.000000

相邻两档倍率（S 翻倍时 O(S) 应 ×2，O(S²) 应 ×4）：
  S   512→1024  math ×3.78    flash_for_cpu ×2.00
  S  1024→2048  math ×3.88    flash_for_cpu ×2.00
  S  2048→4096  math ×3.94    flash_for_cpu ×2.00
```

两个公式被测量**精确确认**（五档全对，比值 1.000000）：

$$\text{alloc}_{\rm math} = \underbrace{B\,H\,S^2\cdot 4}_{\text{分数矩阵}} + \underbrace{B\,H\,S\,D\cdot 4}_{\text{一份 } S\times D \text{ 的临时张量}},\qquad \text{alloc}_{\rm flash\_cpu} = B\,H\,S\,(D+1)\cdot 4$$

- `flash_for_cpu` 的倍率**恰好 ×2.00**（连续三档），严格 $O(S)$，而且它的常数项只有一行 $(D+1)$ 个标量。
- `math` 的倍率是 ×3.78 / ×3.88 / ×3.94——**从下方逼近 4**，因为它的分配里混着一项线性的 $S\cdot D$，S 越大线性项占比越小，斜率越靠近 $S^2$。
- $S=4096$ 那一行：**math 545,259,520 B（520 MiB） vs flash_for_cpu 8,519,680 B（8.1 MiB），差 64 倍。** 而这是**单层、单条序列、8 个头**。

> 顺带说明为什么不用进程峰值 RSS 来量这笔账：我试过，`ru_maxrss` 在 $S$ 从 1024 到 2048 那一档**增量是 0**——因为分配器复用了刚释放的块。峰值 RSS 在「分配-释放-再分配」的循环里会被掩盖，而 per-op 的 `cpu_memory_usage` 不会。这也是量内存代价时一个常见的坑。

### 6.4 第四笔：梯度和可复现性

```python
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(1)
torch.manual_seed(0)
q0, k0, v0 = (torch.randn(2, 4, 64, 32) for _ in range(3))

def fwd_bwd(force_math, det=False):
    q = q0.clone().requires_grad_(True); k = k0.clone().requires_grad_(True); v = v0.clone().requires_grad_(True)
    torch.use_deterministic_algorithms(det)
    try:
        if force_math:
            with sdpa_kernel([SDPBackend.MATH]):
                F.scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()
        else:
            F.scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()
    finally:
        torch.use_deterministic_algorithms(False)
    return (q.grad.clone(), k.grad.clone(), v.grad.clone())

def cmp(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))

print("前向 + 反向的可复现性（grad 逐比特比较）")
print(f"{'配置':<28}{'两次同后端':>13}{'与另一后端':>13}{'‖grad_q‖':>12}")
print("-" * 66)
for det in (False, True):
    for name, force in (("flash_for_cpu", False), ("math", True)):
        g1 = fwd_bwd(force, det); g2 = fwd_bwd(force, det)
        other = fwd_bwd(not force, det)
        tag = f"det={'on ' if det else 'off'} / {name}"
        print(f"{tag:<28}{'一致' if cmp(g1,g2) else '不一致':>13}{'一致' if cmp(g1,other) else '不同':>13}{g1[0].norm().item():>12.6f}")

# gate 函数：能不能离线问「这个配置能用 flash 吗」
import torch.backends.cuda as bc
print("\n门函数可以直接问（参数结构 torch.backends.cuda.SDPAParams，共 7 个字段）：")
p = bc.SDPAParams(q0, k0, v0, None, 0.0, True, False)
print("   SDPAParams(q,k,v,mask=None,dropout=0.0,is_causal=True,enable_gqa=False)")
print("   can_use_flash_attention(...)     ->", bc.can_use_flash_attention(p))
print("   can_use_efficient_attention(...) ->", bc.can_use_efficient_attention(p))
print("   can_use_cudnn_attention(...)     ->", bc.can_use_cudnn_attention(p))
print("   三个都 False —— 这三个门是 CUDA 侧的，在没有 CUDA 的构建里对任何配置都不放行")
```

```text
前向 + 反向的可复现性（grad 逐比特比较）
配置                                  两次同后端        与另一后端    ‖grad_q‖
------------------------------------------------------------------
det=off / flash_for_cpu                一致           不同   35.184708
det=off / math                         一致           不同   35.184711
det=on  / flash_for_cpu                一致           不同   35.184708
det=on  / math                         一致           不同   35.184711

门函数可以直接问（参数结构 torch.backends.cuda.SDPAParams，共 7 个字段）：
   SDPAParams(q,k,v,mask=None,dropout=0.0,is_causal=True,enable_gqa=False)
   can_use_flash_attention(...)     -> False
   can_use_efficient_attention(...) -> False
   can_use_cudnn_attention(...)     -> False
   三个都 False —— 这三个门是 CUDA 侧的，在没有 CUDA 的构建里对任何配置都不放行
```

- **每个后端自己都是逐比特可复现的**（两次同后端：一致），`torch.use_deterministic_algorithms` 在这条路径上没有改变行为（因为这里没有非确定的 CPU kernel）。
- **两个后端的梯度不同**（`‖grad_q‖` 35.184708 vs 35.184711，第 7 位有效数字才分开）。
- **`SDPAParams` 现在是 7 个字段**——第 7 个是 `enable_gqa`。这个结构体还在长大，是个不算稳定的内部 API，但它的存在方式很有启发：**门表被设计成「可以离线问的纯函数」**。`can_use_flash_attention(SDPAParams(...), debug=True)` 本来打算给你一个「这个配置能不能用 flash」的即时答复。可惜在本机（无 CUDA 构建）三个门一律 False，问不出有用信息——**这也正说明为什么 CPU 路径需要自己那套 `use_flash_attention_cpp`：CUDA 门表在 CPU 上等于全关。**
- 关于确定性有一层 CUDA 侧的机制没在本机验证到（**标为「源码里看到的」**）：`can_use_cudnn_attention` 的通用约束里有一条 `check_cudnn_deterministic`——**打开确定性算法时会直接拒绝 cuDNN 后端**。所以 `torch.use_deterministic_algorithms(True)` 在 H100 上会让 attention 从 cuDNN 掉到 flash。这是一个「开了确定性会变慢」的具体来源。

---

## 7. 围绕该领域展开

前面六节都在一台没有 CUDA 的机器上做。本节把 CUDA 侧的全貌补齐——**所有数字都标了来源**：`【上游源码】` = 从 pytorch/pytorch main 分支的 `aten/src/ATen/native/transformers/` 下读到的；`【本机实测】` = 前面几节的 stdout；`【推断】` = 我基于前两者推的。

### 7.1 CUDA 侧的三张门表（NVIDIA 专属参数）

**FlashAttention（`can_use_flash_attention`）** 【上游源码】

| 门 | 条件 | 为什么 |
|---|---|---|
| `check_flash_attention_hardware_support` | SM ∈ **[8.0, 12.1]** | SM80 才有可用的 async copy 与 tensor core 路径 |
| `check_head_dim_size_flash` | `head_dim <= 256`，q/k/v 最后一维相等 | kernel 的 tile 布局假设 |
| `check_flash_attention_grid_size` | `batch <= 65535` 且 `num_heads <= 65535` | 网格维度上限 |
| `check_for_attn_mask` | **`attn_mask` 必须为 `None`** | flash 的 causal 掩码是编译进 kernel 的，任意 mask 无处安放 |
| `check_flash_causal_non_square_seqlens` | 若 `is_causal` 则要求 `seqlen_q == seqlen_k` | 对角对齐假设 |
| `check_last_dim_stride_equals_1_dense<true>` | 最后一维 stride 为 1（忽略单例维） | 向量化访存 |
| `check_requires_grad_and_head_dim_gt192_...` | SM86/89/120/121 且在训练：`head_dim <= 192` 或（无 dropout 且 `head_dim > 224`） | 这些卡的 d=192~224 区间的反向有数值问题 |
| `check_dtypes_flash_attention` | SM90+ 且启用了 FA3：`{fp8_e4m3fn, fp16, bf16}`；否则 `{fp16, bf16}` | FA3 引入了 fp8 |
| **FA4 额外** | SM90 / SM100 且 **`dropout == 0`**；SM100 上 head_dim 有 `<=128` / `(q=192,v=128)` / `(q=256,v=256)` 三档 | FA4 是独立的一套 kernel |

**Memory-Efficient（`can_use_mem_efficient_attention`）** 【上游源码】

| 门 | 条件 |
|---|---|
| `check_mem_efficient_hardware_support` | SM ∈ **[5.0, 12.1]** |
| `check_last_dim_stride_equals_1_dense<false>` | 最后一维 stride 为 1，**不忽略单例维** |
| `check_head_dim_size_mem_efficient` | 按 `minimum_gemm_alignment()` 对齐且 `> 0` |
| `check_data_ptr_alignment_mem_efficient` | **指针**也要对齐 |
| dtype | SM < 80：`{fp16, fp32}`；**SM >= 80：`{fp16, fp32, bf16}`** |
| GQA / MQA | CUDA 上支持；**ROCm 上 `supports_gqa = supports_mqa = false`** |

其中对齐要求由这段决定：

```cpp
int64_t minimum_gemm_alignment(sdp_params const& params) {
  bool is_half = (params.query.dtype() == at::kHalf) || (params.query.dtype() == at::kBFloat16);
  bool use_tc = use_tensor_cores(params, dprops, is_half);
  int64_t matmul_alignment_mn = 1;
  if (dprops->major >= 8) { matmul_alignment_mn = 4; }
  int64_t bits_per_scalar = is_half ? 16 : 32;
  if (use_tc) { matmul_alignment_mn = std::max(matmul_alignment_mn, 128 / bits_per_scalar); }
  return matmul_alignment_mn;
}
```

量纲自校验一下：`128 / bits_per_scalar` —— fp16 给 8，fp32 给 4。**这是「128 位怎么切」而不是「128 个元素」**，单位是元素数、量纲无量纲，结果 8 与 4 满足「fp16 要求 8 的倍数、fp32 要求 4 的倍数」的直觉。配合 `use_tensor_cores`（SM80+ 恒 true；SM70~79 仅 fp16；更低 false），得出的结论是：**fp32 在这条路上要求 4 对齐，fp16/bf16 要求 8 对齐**。

**cuDNN Attention（`can_use_cudnn_attention`）** 【上游源码】

| 门 | 条件 |
|---|---|
| `check_cudnn_hardware_support` | SM ∈ **[8.0, 12.1]** |
| cuDNN 版本 | `< 8903` 拒绝；`< 8906` 且带 dropout 或 `s_k % 64 != 0` 拒绝；`< 90000` 要求 `s_q >= 64` |
| `check_dtypes_low_precision` | SM >= 80：`{fp16, bf16}`（**注意这里没有 fp32**，与 mem-efficient 不同） |
| `check_cudnn_deterministic` | **开了确定性算法 → 直接拒绝** |
| `check_cudnn_hardware_support` + head_dim | 默认上限 **128**；SM90/SM10x 且 cudnn > 92200 且 frontend >= 12400 → **256**；`d_qk==192 && d_v==128`（DeepSeek 形态）→ 192；**且必须是 8 的倍数** |
| `check_cudnn_dropout` | `dropout` 必须是 **1/16 的整数倍** |
| `check_cudnn_query_seq_len_nested` | 嵌套输入下 `max_seqlen_q <= 128` 拒绝 |
| decode 禁用名单 | SM 10.x / 11.x 上 cuDNN **9.19 ~ 9.25（9.24.1 除外）** 的 decode 被禁用 |
| `check_cudnn_d256_bprop_head_dim` | 要反向时：`d_qk <= 128 && d_v <= 128`，或 `(192, 128)`，或 `(256, 256)` |

第 034 期记过一个数：FlashInfer 的 DSA kernel 名里 `head_dim_with_scale = 132 = 128 + 4`。把它放进上面这张表：**132 落在 128 与 192 之间**——如果哪天有人想拿 cuDNN 后端跑 MLA，这个 132 恰好踩在「默认上限 128」之外，能不能过取决于 `d_qk == 192 && d_v == 128` 那一条放不放行。【推断】

### 7.2 哪些是 NVIDIA 特有，哪些是通用抽象

这一节是为了让你能把这套东西迁移到别的加速器上（比如自研卡）。

| 概念 | 性质 | 迁移时要做什么 |
|---|---|---|
| **三层结构**（门表 / 优先级 / 兜底） | **通用** | 任何后端都要有这三层。缺了兜底就会在某些输入上直接崩；缺了门表就会给出错误结果 |
| **门表是纯函数、可离线问** | **通用** | 设计时就把「配置 → bool」抽出来，别把检查散在 kernel 里 |
| **「失败静默、全败才发声」** | **通用（且危险）** | 强烈建议自己加一层「后端选完 → 记一个 metric / 打一条日志」 |
| **优先级序列是全局状态** | **通用** | 全局状态在并发场景（多线程编译、多进程服务）下是隐患 |
| roofline、算术强度、occupancy | 通用抽象 | 用来判断「这个后端理论上有多少空间」 |
| `attn_mask` 有无 / 最后一维 stride / 对齐 | 通用约束 | 任何向量化 kernel 都会要求 |
| **`head_dim <= 256` / `<= 128`** | **NVIDIA 特有** | 换成 tile 布局假设不同的硬件，数字要重填 |
| **SM 版本区间 `[8.0, 12.1]`** | **NVIDIA 特有** | 换成目标芯片的能力枚举 |
| **`{fp16, bf16, fp8_e4m3fn}`** | 半通用 | fp8 的格式（e4m3 / e5m2）各家不完全一致，别直接搬 |
| TMA / wgmma / tcgen05 / 2:4 稀疏 / 线程块簇 | **NVIDIA 特有** | 迁移要重新映射到目标硬件的对应机制 |
| **cuDNN / FA3 / FA4 / AOTriton / CK 这些后端名字** | **厂商特有** | 目标平台的等价物是「谁提供的融合 kernel」 |

一条对迁移最有用的推论【推断】：**门表里最容易被漏掉的一类约束是「对齐」**（stride 对齐、指针对齐、head_dim 整除）。这类约束在 source 里看着琐碎，但它们决定了你的 kernel 能不能用向量化指令——也是迁移到新硬件时最容易踩、最难查的一类。第 035 期里那个 `page_size=16` 的「硬地板」是同一个家族的东西。

### 7.3 谁来接管：compile / flex_attention / 自定义后端

`SDPBackend` 只有 5 个真实成员，但现实里 attention 的实现远多于 5 个。**多出来的那些都走别的入口**，这是理解这张枚举为什么「看起来这么少」的关键：

| 入口 | 怎么做 | 什么时候用 |
|---|---|---|
| `SDPBackend` + `sdpa_kernel()` | 就在这套机制里选 | 想要标准三选一 / 兜底 |
| `torch.nn.attention.flex_attention` | 自己给 `score_mod`（逐元素打分改动）和 `block_mask`（块级稀疏） | 需要**自定义掩码结构**，而不是自定义 kernel |
| `torch.compile` / Inductor | 编译器自己生成/选择 kernel，绕过这套运行时选择 | 整个模型要编译时 |
| 自己的 kernel + `torch.library` | 注册成新算子，自己负责后端选择和门表 | 框架里根本没有的能力（第 034/035/036 期的三种情况） |

**这张表解释了那三期的结论为什么是「自己写 kernel」而不是「找个开关」**：第 034 期的稀疏 indexer、第 036 期的 sink 列，都是**输入语义**变了（多了内容相关的选择、多了一列），而 `SDPBackend` 那一层的门表是按「执行方式」分类的，它的接口（`SDPAParams` 的 7 个字段）里根本没有表达这类语义的位置。`flex_attention` 的 `score_mod` 能表达「给 logits 加偏移」但不能表达「多一列 KV」（第 036 期实测过），`block_mask` 只能表达**结构**稀疏、不能表达**内容相关**的 top-k（第 035 期实测过，元素级 0.2207 → 块级 0.4375）。**两者都不够，于是只能往下走一层。**

### 7.4 与已发各期的接口

把这一期接回前面：

- **第 005 期（Flash Attention）**：讲的是算法——在线 softmax 为什么不用物化 $S\times S$。本期 6.3 节把那个「不用」量成了**实测 64 倍**（$S=4096$，单层单序列），并且给出了两条精确的分配公式。005 给机制，038 给数字。
- **第 021 期（动态形状）**：门表里的 `check_*` 大量使用 `sym_size` 和 `TORCH_GUARD_OR_FALSE`——**在符号形状下遇到「算不出来」就返回 false（拒绝）**。也就是符号形状下的默认行为是**保守拒绝、退到 math**。这解释了为什么动态形状的模型更容易掉进慢路径。
- **第 032 期（MHA/MQA/GQA/MLA）**：那一期讲的 head 数配比，在本期是门表的一项（`check_grouped_query_attention`、`enable_gqa`）。实测确认 CPU 的 flash 门表支持 GQA（`H:1` 那行走了 flash）。
- **第 001 期（CUDA Graph）**：捕获期间不能有同步、不能有新分配。**后端选择必须在捕获那一刻固定下来**，捕获后不能变——所以「第一次跑对、capture 之后变了」是一类真实故障【推断】。
- **第 034/035/036 期**：本期的 1.1 节和 7.3 节把这三期撞的墙解释清楚了——不是配置没找对，是那一层根本没有对应的位置。
- **第 037 期（零假设基准）**：本期 6.2 节的**对照组为 0** 就是那条方法论的直接应用——**先证明「什么都不改」时差异是 0，才有资格解释「改了一件事」时差异是 1e-5。**

---

## 8. 什么时候该用 / 不该用

**该 pin 后端（用 `sdpa_kernel` 明确指定）的场景：**

- **做 A/B 对照实验或复现某一篇论文的数字。** 6.2 节已经量到：不 pin 的话，换台机器参数就会分叉。
- **需要 bitwise 可复现的 CI。** 版本升级最容易悄悄换掉后端。
- **排一个只在某个后端上出现的数值问题。** `sdpa_kernel([MATH])` 是最快的二分法：math 正确而 flash 不正确，问题就在 kernel 或门表；两个都错，问题在你的输入。
- **需要 LSE 返回值**（chunked prefill、自定义反向）——直接调底层算子而不是走 SDPA。

**不该 pin 的场景：**

- **生产推理。** 后端是给硬件挑的，你比框架更了解这台机器的可能性很低。而且 pin 之后框架的升级红利全丢了。
- **只是为了「让结果稳定」而随便 pin 一个。** 6.1 节量化过：两个 fp32 后端离 fp64 一样近，pin math 不会让模型更准，只会更慢更占内存。
- **想用 `sdpa_kernel` 关掉 cuDNN 来省显存。** 显存是 $O(S)$ 还是 $O(S^2)$ 决定的（6.3 节），关掉 cuDNN 会退到 flash 或 math，可能反而更差。

**该做但很少有人做的：**

- **在生产里埋一个「实际后端」的计数指标。** 用 `TorchDispatchMode` 或 profiler 采样，把「有多少比例走的是 math」报出来。第 4.3 节证明了降级完全静默——**这是唯一能在用户抱怨之前发现它的手段。**

---

## 9. 常见坑

**坑 1：以为 `sdpa_kernel` 的列表顺序就是优先级。**
不带 `set_priority=True` 时它只是个集合。实测：`[MATH, FLASH_ATTENTION]` 和 `[FLASH_ATTENTION, MATH]` 走同一条路。（而且在本机上加不加 `set_priority` 也一样，因为 CPU 路径硬编码了顺序——**5.1 节**。）

**坑 2：以为 `_cur_sdpa_kernel_backends()` 的返回顺序有意义。**
它按名字字典的顺序拼列表，所以打印出来 `CUDNN_ATTENTION` 排第一，看着像默认首选。真正的顺序只能从 `torch._C._get_sdp_priority_order()` 读。（**3.1 节**）

**坑 3：`math_sdp_enabled(False)` 报 TypeError 就以为这个开关不存在。**
getter / setter 名字不对称：读是 `math_sdp_enabled()`，写是 `enable_math_sdp(bool)`。四个后端全这样。（**3.1 节**）

**坑 4：把「降级到 math」当成性能问题去查 kernel。**
它跟 kernel 优化毫无关系，是门表放行与否。**判断方法**：用 dispatcher 观测（3.2 节），两分钟内就能确认。**常见的触发源**远不止代码里的显式参数：`attn_mask` 是不是 `None`、输入是不是连续、`dropout_p` 是否非零、head_dim 是否越界、是不是第一次调用（编译/初始化）。

**坑 5：拿进程峰值 RSS 去量 attention 的中间张量。**
我实测过：$S$ 从 1024 到 2048，峰值 RSS **增量是 0**——分配器复用了刚释放的块。要用 per-op 的 `cpu_memory_usage`（6.3 节）或者显式跟踪。

**坑 6：看到「所有后端被拒」的报错就去翻门表。**
那条报错只在**一个后端都不剩**的时候出现。只要你没关 math，它就永远不会出现——你看到的任何 RuntimeError 几乎都不是「后端选择」的问题，而是输入形状/数值的问题（4.3 节那个 head_dim 不一致的例子）。

---

## 10. 一句话总结

> **`SDPBackend` 那一层做的事只有三件——问门表、按优先级、兜底 math；它的全部风险也只有一件——这三件事全都不出声。所以真正要装进工具箱的不是那五个枚举值的名字，而是「怎么观测实际走了哪条路」这一条：`TorchDispatchMode` 看原子算子，profiler 看逻辑算子，`_get_sdp_priority_order()` 看顺序。**

---

## 11. 今日练习

假设你在给一个服务做性能排查，配置是 **$B=1$、$H=32$、$S=16384$、$D=128$、fp16、单层**，同事说「attention 这一段比预期慢十倍」。你已经确认了 kernel 本身没有问题。

1. 写出**一个可运行的断言工具**：给一段 attention 调用，让它报出实际走了哪个后端（要求用 dispatcher 观测，不能看输出数值）。
2. 用这个工具跑一遍 **$S=16384$** 的常见配置，列出哪些会掉到 math。
3. 算出 math 与 flash 在**中间张量**上的差距。$S=16384$、$H=32$、$D=128$、fp16，两边的量各是多少？
4. 如果这个服务跑在一块**没有 `attn_mask`、也没有 dropout、输入全部连续**的路径上，第 3 步的差距还成立吗？为什么？
5. 给出一条**能自动化执行**的判据，用来在 CI 里拦住「某次改动让 attention 静默掉到 math」。

<details><summary>今日练习（参考答案）</summary>

### 1. 断言工具

```python
from torch.utils._python_dispatch import TorchDispatchMode

class SDPARecorder(TorchDispatchMode):
    """记录一次 SDPA 调用实际派发到的原子算子，推断后端。"""
    def __init__(self): self.seen = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func))
        return func(*args, **(kwargs or {}))

    @property
    def backend(self):
        s = " ".join(self.seen)
        if "flash_attention_for_cpu" in s: return "flash_for_cpu"
        if "_scaled_dot_product_flash_attention" in s: return "flash"     # CUDA
        if "_scaled_dot_product_efficient_attention" in s: return "mem_efficient"
        if "_scaled_dot_product_cudnn_attention" in s: return "cudnn"
        if "bmm" in s and "_safe_softmax" in s: return "math"
        return "unknown"

# 用法
rec = SDPARecorder()
with rec:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
assert rec.backend != "math", f"attention 掉到了 math：{rec.backend}"
```

**为什么不能靠看输出数值**：第 6.1 节量过，两个 fp32 后端的输出差只有 `7e-7` 量级——用数值判断后端，等于用一把误差比目标还大的尺子去量。

### 2. 常见配置的判定

第 4 节那张 18 行的表就是答案的模板。**$S=16384$ 只影响内存与时间，不影响门表**——门表里唯一与序列长度有关的约束是「非零」和「causal 时 q/k 等长」，$16384$ 全部通过。所以降级条件与 $S=64$ 时完全一样：

| 条件 | 结果 |
|---|---|
| 无 mask、无 dropout、连续 | flash |
| `dropout_p > 0` | **math** |
| 任一输入的最后一维 stride ≠ 1 | **math** |
| head_dim > 256（CUDA flash）/ > 128（CUDA cuDNN） | **math**（CPU 无上界） |
| `attn_mask` 非 `None`（CUDA flash） | **math** |

### 3. 中间张量

第 4.3 节的实跑输出（逐字）：

    B·H·S²·2  =     17,179,869,184  = 16.00 GiB
    B·H·S·D·2 =        134,217,728  = 0.12 GiB
    比值 = 128×

**math 需要的分数矩阵是 16.00 GiB，flash 路径的线性量是 0.12 GiB，差 128 倍。**

量纲自校验：$B\,H\,S^2\cdot 2$ 的单位是（批次 × 头数 × 元素数 × 2 字节/元素）= 字节。$1\times32\times16384^2\times2 = 1.72\times10^{10}$，除以 $2^{30}$ 得 16.00 GiB，对得上。

比值 $S/(D/2 \cdot \ldots)$ 其实有更简洁的写法：$\frac{B H S^2 b}{B H S D b} = \frac{S}{D} = \frac{16384}{128} = 128$。**这个倍数只与 $S/D$ 有关，与 batch、头数、dtype 全都无关**——这是这套账里最有用的一个结论：序列越长、head_dim 越小，物化分数矩阵的代价越大。

### 4. 这个差距还成立吗

**仍然成立，而且这是本节的要点。** 那三个条件（无 mask、无 dropout、连续）影响的是**「会不会降级」**，不影响**「降级之后要付多少钱」**。

要注意区分两件事：
- **门表**决定走哪条路（第 4 节实测：三个条件都满足 → 走 flash）。
- **公式**决定每条路要多少钱（第 6.3 节实测：math = $O(S^2)$ + 线性项，flash = 严格 $O(S)$）。

在本例中，因为条件全部满足，**它不会降级**——所以同事说的「慢十倍」的根因不在后端选择上，需要往别处查（比如 cv 形 batch 的分块、CPU 侧的 launch 开销、或者根本没有用上融合后端）。**这恰恰是第 1 步那个工具的用法：先用它排除后端选择，再去查 kernel。** 如果第 2 步显示走的就是 flash，那么「静默降级」这个假设被排除，可以直接进入下一层。

### 5. CI 判据

```python
# 在 CI 里对固定的一组配置逐个断言，任何一个掉到 math 就 fail
CANONICAL_CONFIGS = [
    dict(B=1, H=8,  S=512,  D=64,  dtype=torch.float32, causal=None),
    dict(B=1, H=8,  S=2048, D=128, dtype=torch.float16, causal="causal"),
    dict(B=1, H=8,  S=128,  D=64,  dtype=torch.bfloat16, causal="causal"),
]
```

四条设计原则，每一条都对着本期的一个实测：

1. **断言的是「走了哪个后端」，不是「跑得多快」。** 速度有机器噪声，后端名是确定的。（6.2 节：噪声能到 15%，而对照组本该是 0。）
2. **要覆盖「边界型」配置**：短序列（$S=128$，容易触发某些 kernel 的最小长度要求）、`is_causal=True`（CPU 走不同分支）、非连续输入、有 mask。**只跑一个「标准形状」的 CI 是发现不了降级的**——因为降级正是由那些不标准的输入触发的。
3. **加一个对照组**：同一份配置跑两次，断言 `‖Δθ‖/‖θ‖ == 0`（6.2 节）。这一条能把「环境本身有问题」和「这次改动有问题」分开。
4. **不要只断言「没降到 math」，还要断言「没悄悄换了后端」**：把实测的后端名写进 golden 文件。换机器/换版本时它会响亮地失败——而这是**你想要的**，因为那正是参数会分叉的时刻。

**一句话的判据**：**把「实际后端」当成一个需要断言的输出，而不是一个可以忽略的实现细节。**

</details>

---

*本期所有输出块均由 `scratch-038/art038/` 下的脚本真实执行产生，`scratch-038/check_article.py` 逐块比对 stdout（每块独立子进程）。实验环境：torch 2.14.0，CPU（本机无 CUDA，SM/head_dim/对齐等 CUDA 侧数字均标注为「上游源码」而非实测）。*
