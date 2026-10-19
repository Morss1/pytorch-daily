# PyTorch 每日一课 · 第 039 期

## ATen 与 Dispatch：一次 `a + b` 到底是谁在执行

**日期**：2026-10-19　**难度**：⭐⭐⭐⭐　**预计阅读时间**：30 分钟

**前置知识**：知道 `torch.Tensor` 有一堆"属性"（`device`、`dtype`、`layout`、`requires_grad`）；见过 `torch.library` 或 `TORCH_LIBRARY` 这两个词；如果读过第 038 期（SDPA 的后端选择），本期是它的下一层。

**关联**：第 038 期问的是「**一个算子内部**怎么选实现」（门表 + 优先级 + 兜底）；本期问的是「**一个算子调用**怎么找到任何一个实现」。038 期的三层结构（门表 / 优先级序列 / 兜底 math）在机制上就是本期这套东西的局部特例——只不过 SDPA 把它手写了一遍，而 aten 的每一个算子都免费拥有。第 014 期（profiler）看到的是逻辑算子，本期解释那一层之下的原子算子从哪来；第 001 期（CUDA Graph）捕获期固定后端，本质是"把查表结果冻住"；第 037 期的"参考量要先算机制"，本期的 `Undefined` 条目就是派发表里那个"永远存在但不干活"的参考量。

---

## 0. 一句话

> **`a + b` 不是一次函数调用，是一次查表。** 这张张量（以及它的每个参数）身上贴着一组标签（`DispatchKey`），派发器取这组标签里**优先级最高、且对这个算子有实现**的那一个，调用它下面的函数。整套机制存在的理由只有一个：3592 个算子 × 136 个标签，不可能用 `if-else` 或虚函数表写出来。

---

## 1. 这个领域解决什么问题

### 1.1 三条正交的维度，乘起来就是一张爆炸的表

一个 `aten::add` 的语义维度至少有这么多条，而且**可以同时成立**：

| 维度 | 取值（举例） | 谁来决定 |
|---|---|---|
| 数据在哪 | CPU / CUDA / MPS / PrivateUse1 / Meta | 设备的类型 |
| 数据长什么样 | dense / sparse COO / sparse CSR / nested / 量化 | layout 与 dtype |
| 要不要建图 | 建 / 不建 / 建了但记在 functionalize 里 | `requires_grad` + 全局模式 |
| 数值上是不是"虚"的 | 实数 / 共轭视图 / 取负视图 / 零张量 | 视图操作 |
| 要不要降精度 | 是（bf16/fp16） / 否 | autocast 上下文 |
| 是不是在"假张量"上跑 | 真张量 / FakeTensor / meta | 编译期、shape 推导 |

关键在于这些维度**不是互斥的**。一个张量完全可以同时是：CPU 上的、需要建图的、共轭视图、正处在 autocast 区域里。如果每个算子自己去 `if` 这些条件，`aten::add` 一个函数要写 $2^6$ 个分支，而 aten 有 3592 个算子（含重载）。

这就是 Dispatch 要解决的问题：**把"这个调用需要经过哪些关卡"编码成一组位（bit），把"每个关卡由谁负责"编码成一张按位索引的数组**。查表是 $O(1)$，加一个维度只要多一个比特位，加一个后端只要多一列。

### 1.2 为什么不能是虚函数表

C++ 里干这件事的常规手段是虚函数：`Tensor` 是个基类，`CPUFloatTensor`、`CUDATensor` 各自 override。这条路在 PyTorch 里走不通，原因有三个，都很具体：

1. **一个张量同时属于多个"子类"**。虚表只允许单继承的覆盖关系；而"CPU + 需要建图 + 是共轭视图"是三件独立的事，需要**叠加**而不是覆盖。
2. **同一份计算里可能混后端**。`sparse + dense` 的加法、`meta + cpu` 的形状推导，都不是"以哪个张量的虚表为准"能回答的——需要运行时按优先级裁决。
3. **第三方要能接管，而不改 PyTorch 源码**。`torch.library` 注册的算子在编译期根本不存在于 PyTorch 里，虚表没法给"还不存在的类型"留槽位。

位图 + 查表同时解决了这三点：位图天然可叠加，查表可以在运行时按优先级裁决，表项可以在运行时插入。

---

## 2. 核心思想：三个部件

### 2.1 部件清单

| 部件 | 是什么 | 一行代码摸到它 |
|---|---|---|
| **schema** | 算子的签名（名字 + 参数类型 + 返回类型），派发前用来把 Python 的调用转成统一的参数栈 | `torch.ops.aten.add.Tensor._schema` |
| **keyset** | 一张 64 位的位图，描述"这次调用要经过哪些关卡" | `torch._C._dispatch_keys(t)` |
| **派发表** | 每个算子一张 `key → 函数指针` 的表，用 keyset 的最高位去索引 | `torch._C._dispatch_dump_table("aten::add.Tensor")` |

最容易被误解的是 keyset：它**不是**"这个张量是什么类型"，而是"这个张量需要哪些层参与"。这一点决定了后面所有的行为，第 3 节会拿实测把这句话钉死。

### 2.2 位是怎么排的

源码里 `c10/core/DispatchKey.h` 的注释写得很直白（上游 main 分支）：

> In implementation terms, the dispatch key identifies a specific "bit" in a DispatchKeySet. **Higher bit indexes get handled by dispatching first** (because we "count leading zeros" when we extract the highest priority dispatch key.)

以及位布局：

> The bits in the DispatchKeySet are split between the **bottom ~12 "BackendComponent" bits**, while the remaining **upper bits are assigned to functionalities**.

于是整个 keyset 可以写成

$$
\text{bit}(f, b) = f \cdot N_{\text{backend}} + b,\qquad b \in [0, N_{\text{backend}}),\quad f \in [0, F)
$$

其中 $N_{\text{backend}} \le 16$（头文件里有一个 `static_assert` 专门守这条），$F$ 是功能（functionality）的个数。这个式子只是**布局示意**，别拿它去算具体位号——真实位号还夹着若干个"per-backend functionality"的合并（`Dense`/`Quantized`/`Sparse`/`SparseCsr`/`NestedTensor` 各自占 1 位、但在算子表里映射到 $N_{\text{backend}}$ 个槽）。一个 key 的位次既编码了"哪一类语义"（$f$），也编码了"哪个后端"（$b$）。

这解释了一件在 dump 表里看得见、但很容易看漏的事：`CPU`、`SparseCPU`、`QuantizedCPU`、`AutogradCPU` **是四个不同的位**，而不是"CPU 张量的四种状态"。同一个 `SparseCPU` 张量身上只有 `SparseCPU` 一个后端位。

### 2.3 派发的四步

```
1. 有效集合 = (张量 keyset ∪ TLS include) − TLS exclude
2. 取有效集合里位次最高的 key        # count leading zeros
3. 在那个算子的表里查这个 key
4. 命中 → 调用；没命中 → 顺次往下问下一个 key
```

第 4 步是"fallback 链"的来源：一个 key 没实现，不代表调用失败，只代表"这一关没人管，往下走"。**整个机制里最反直觉的部分都在第 4 步**——第 4 节会看到一张 96 行的表里只有 15 行是"真实现"，其余 81 行全是"往下走"或者"用别人的默认实现"。

### 2.4 一个类比

把它想成机场：

- 张量身上贴的是**行李标签**（"我有液体"、"我是国际航班"、"我要退税"）。
- keyset 是"这趟行程需要经过的所有检查口"。
- 派发表是"每个检查口的职责清单 + 具体执行人"。
- 优先级是检查口的固定动线：先报关、再安检、最后登机。
- `backend fallback` 是"这个口对**所有**航班都开，但它自己不做具体业务，只负责把人领到下一个口"。
- `composite`（第 4 节）是"这个口不检查，直接给你一张纸条，上面写着'自己去 3 号、5 号、7 号窗口办'"。

---

## 3. 张量身上到底贴了什么（实测）

先看最普通的那张纸：一个 CPU 上的 float32 张量。

```python
import torch, warnings
warnings.simplefilter("ignore")

def keys(t):
    s = torch._C._dispatch_keyset_to_string(torch._C._dispatch_keys(t))
    return set(s[len("DispatchKeySet("):-1].split(", "))

def row(name, t):
    s = torch._C._dispatch_keyset_to_string(torch._C._dispatch_keys(t))
    body = s[len("DispatchKeySet("):-1]
    print(f"{name} → {body}")

z = torch.randn(3)
row("普通 float32", z)
row("requires_grad=True", torch.randn(3, requires_grad=True))
row("复数 conj 视图", torch.randn(3, dtype=torch.complex64).conj())
row("neg 视图", torch._neg_view(torch.randn(3)))
row("零张量 _efficientzerotensor", torch.ops.aten._efficientzerotensor([3]))
row("量化张量", torch.quantize_per_tensor(torch.randn(3), 0.1, 0, torch.quint8))
row("sparse COO", torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.tensor([1.0, 2.0]), (3,)))
row("meta 张量", torch.randn(3, device="meta"))

from torch._subclasses.fake_tensor import FakeTensorMode
with FakeTensorMode():
    row("FakeTensor", torch.randn(3))

with torch.inference_mode():
    row("inference_mode 内创建", torch.randn(3))

base = keys(z)
print()
print("相对普通张量的差异：")
print("  requires_grad=True ->", keys(torch.randn(3, requires_grad=True)) - base or "无差别")
with torch.inference_mode():
    print("  inference_mode    -> 少", sorted(base - keys(torch.randn(3))))
```

```text
普通 float32 → CPU, ADInplaceOrView, AutogradCPU, AutocastCPU
requires_grad=True → CPU, ADInplaceOrView, AutogradCPU, AutocastCPU
复数 conj 视图 → CPU, Conjugate, ADInplaceOrView, AutogradCPU, AutocastCPU
neg 视图 → CPU, Negative, ADInplaceOrView, AutogradCPU, AutocastCPU
零张量 _efficientzerotensor → CPU, ZeroTensor, ADInplaceOrView, AutogradCPU, AutocastCPU
量化张量 → QuantizedCPU, ADInplaceOrView, AutogradCPU, AutocastCPU
sparse COO → SparseCPU, ADInplaceOrView, AutogradCPU, AutocastCPU
meta 张量 → Meta, ADInplaceOrView, AutogradMeta
FakeTensor → CPU, Python, ADInplaceOrView, AutogradCPU, AutocastCPU, PythonTLSSnapshot
inference_mode 内创建 → CPU, AutocastCPU

相对普通张量的差异：
  requires_grad=True -> 无差别
  inference_mode    -> 少 ['ADInplaceOrView', 'AutogradCPU']
```

这张表里有四条值得停下来看。

**第一条：`requires_grad=True` 不改变 keyset。** 两行输出逐字相同。原因很朴素：`AutogradCPU` 这个 key **本来就在**每个 CPU 张量身上（不管它需不需要梯度）。真正决定"建不建图"的是另外两样东西——张量上的 `AutogradMeta` 里那个 `requires_grad` 布尔值，以及全局的 `GradMode`（`torch.no_grad()` 改的就是它）。autograd 的 kernel 只是在这一关被问到"你要不要建图"，答"不要"就往下走。

这条推论很重要：**你不能靠读 keyset 判断一个张量会不会建图**。第 7 节的表里 `torch.no_grad()` 那一行同样是"无变化"。

**第二条：`Conjugate` / `Negative` / `ZeroTensor` 是"语义位"，不是"数据类型"。** 一个共轭视图和它的基张量是同一块内存、同一个 `dtype`（`complex64`），唯一的区别就是多了一个 `Conjugate` 位。这个位的作用是告诉下一层："我这里的数是共轭过的，你要么当场把它物化，要么用能理解共轭的算法"。

**第三条：量化 / sparse 是"换后端位"，不是"加位"。** `CPU` 消失了，换成 `QuantizedCPU` / `SparseCPU`。因为它们改的是"数据怎么存"（后端位的 $b$ 变成 Quantized/Sparse 这一类功能 $f$），而不是"这次调用要不要多做一件事"。

**第四条：`inference_mode` 是"删位"，不是"加标志位"。** 在 `inference_mode` 里创建的张量，keyset 里**少了 `ADInplaceOrView` 和 `AutogradCPU`**。这就是为什么 inference 张量不能参与反向传播——它身上根本没有 autograd 那一关，不是被一个 flag 挡住的。这也解释了那条著名的报错 `Inference tensors cannot be saved for backward`：不是"不允许"，是"没这条通道"。

另外注意 `meta 张量`：`AutogradCPU` 变成了 `AutogradMeta`，`AutocastCPU` 也不见了。**autograd 是按后端分开的位**，不是全局一个。这一点在第 10 节接新硬件时会再遇到一次（`AutogradPrivateUse1`）。

---

## 4. 一张派发表长什么样

现在把表打开。下面这段按"条目类型"给三个算子的表做了分类统计。

```python
import torch, collections

def table(op):
    rows = []
    for line in torch._C._dispatch_dump_table(op).strip().split("\n"):
        key, rest = line.split(":", 1)
        tag = rest[rest.rfind("[") + 1: rest.rfind("]")]
        rows.append((key.strip(), tag))
    return rows

for op in ["aten::add.Tensor", "aten::randn", "aten::softmax.int"]:
    rows = table(op)
    c = collections.Counter(t for _, t in rows)
    print(f"{op}：{len(rows)} 条")
    print("  ", " / ".join(f"{k}×{v}" for k, v in c.most_common()))

print()
rows = table("aten::add.Tensor")
kern = [k for k, t in rows if t == "kernel"]
print("add.Tensor 有专属 kernel 的 key：", ", ".join(kern))
print("add.Tensor 的 CUDA 条目：", dict(rows)["CUDA"])
fb = [k for k, t in rows if t == "backend fallback"]
print("add.Tensor 的 backend fallback（按优先级，左低右高）：")
print("  " + " → ".join(fb))

print()
print("randn 的 BackendSelect：", dict(table("aten::randn"))["BackendSelect"])
print("randn 有专属 kernel 的 key：", ", ".join(k for k, t in table("aten::randn") if t == "kernel"))
print()
print("softmax.int 的 CPU：", dict(table("aten::softmax.int"))["CPU"])
print("softmax.int 里 math kernel 的条目数：",
      sum(1 for _, t in table("aten::softmax.int") if t == "math kernel"))

names = torch._C._dispatch_get_all_op_names()
D = torch._C.DispatchKey
print()
print("整个 aten 的规模：")
print("  op 总数（含重载）:", len(names))
for k in ["CPU", "Meta", "CompositeImplicitAutograd", "CompositeExplicitAutograd", "FuncTorchBatched"]:
    c = sum(1 for n in names if torch._C._dispatch_has_kernel_for_dispatch_key(n, getattr(D, k)))
    print(f"  在 {k} 上有实现的 op 数: {c}")
```

```text
aten::add.Tensor：96 条
   default backend kernel×42 / backend fallback×21 / autograd kernel×18 / kernel×15
aten::randn：113 条
   default backend kernel×69 / backend fallback×21 / autograd kernel×18 / kernel×5
aten::softmax.int：129 条
   math kernel×104 / backend fallback×18 / kernel×6 / batched kernel×1

add.Tensor 有专属 kernel 的 key： CPU, MPS, Meta, MkldnnCPU, SparseCPU, SparseMPS, SparseMeta, SparseCsrCPU, SparseCsrMeta, NestedTensorCPU, NestedTensorHPU, ZeroTensor, Tracer, FuncTorchBatched, Batched
add.Tensor 的 CUDA 条目： default backend kernel
add.Tensor 的 backend fallback（按优先级，左低右高）：
  BackendSelect → Python → FuncTorchDynamicLayerBackMode → Functionalize → Conjugate → Negative → ADInplaceOrView → AutocastCPU → AutocastMTIA → AutocastMAIA → AutocastXPU → AutocastMPS → AutocastCUDA → BatchedNestedTensor → FuncTorchVmapMode → VmapMode → FuncTorchGradWrapper → PythonTLSSnapshot → FuncTorchDynamicLayerFrontMode → PreDispatch → PythonDispatcher

randn 的 BackendSelect： kernel
randn 有专属 kernel 的 key： Meta, BackendSelect, Tracer, FuncTorchVmapMode, VmapMode

softmax.int 的 CPU： math kernel
softmax.int 里 math kernel 的条目数： 104

整个 aten 的规模：
  op 总数（含重载）: 3592
  在 CPU 上有实现的 op 数: 1173
  在 Meta 上有实现的 op 数: 1606
  在 CompositeImplicitAutograd 上有实现的 op 数: 783
  在 CompositeExplicitAutograd 上有实现的 op 数: 1223
  在 FuncTorchBatched 上有实现的 op 数: 614
```

### 4.1 五种条目，读表先认标签

| 标签 | 含义 | 谁写的 |
|---|---|---|
| `kernel` | 这一关的**真实现** | 人写的（`RegisterCPU_*.cpp` 之类） |
| `backend fallback` | 这个 key 对所有算子都挂了一个兜底函数 | 框架（`PythonFallbackKernel.cpp` 等） |
| `default backend kernel` | 这个后端**没有**专属实现，用 composite 的默认实现顶上 | 代码生成（`RegisterCompositeExplicitAutograd*`） |
| `math kernel` | composite implicit 的"配方"，展开到每个后端 | 代码生成（`RegisterCompositeImplicitAutograd*`） |
| `autograd kernel` | `VariableType` 生成的建图/改图逻辑 | 代码生成（`VariableType_*.cpp`） |

`add.Tensor` 的 96 条里只有 **15 条**是 `kernel`。剩下 81 条里，21 条是"往下走"，18 条是"帮你建图"，42 条是"用别人的默认实现"。**这张表的绝大部分不是实现，是路由。**

### 4.2 `Undefined`：表里第一行的那个万能兜底

第一行永远是 `Undefined`。它是"所有后端位都没匹配上"时的落点，通常装的是 composite explicit 的默认实现（`add.Tensor` 那一行就是 `[default backend kernel]`）。它的存在解释了一个常见现象：**报错信息里出现的算子名，经常不是你调的那个**（第 6 节的实验会抓一个现行）。

### 4.3 `randn` 的 `BackendSelect` 是 `kernel`，不是 `fallback`

这一条差别很关键。`add.Tensor` 的 `BackendSelect` 是 `fallthrough`（标记为 backend fallback）——因为它的后端由张量参数决定，根本不需要 `BackendSelect` 出手。

而 `randn` 一个张量参数都没有，后端只能从 `device=` 参数里读。所以它必须有一个**真的 `BackendSelect` kernel**：先看 `device` 决定后端，再把调用重新派发过去。这就是"工厂函数"和"普通算子"在派发层面的分界线，也是第 6 节那个实验里"把 `BackendSelect` 挖掉之后 `randn` 全挂"的原因。

### 4.4 `softmax.int` 的 104 条 `math kernel`

104 条这个数字说明了 composite 的本质：**它不是"一条实现"，而是"填满所有槽位的配方"**。`DispatchKey.h` 里明确把 `CompositeImplicitAutograd` 划在别名键区间（`StartOfAliasKeys .. EndOfAliasKeys`）内，注释原话是"registered at `build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp`"。注册一次，代码生成会把它复制到每一个 runtime key 的槽位上——所以表里的条数约等于"后端位数 × 相关功能位数"，而不是 1。

顺带一个规模感（第 4 节表里的最后五行）：3592 个算子里，**783 个**在 `CompositeImplicitAutograd` 上有实现（只写配方、不写 kernel），**1223 个**在 `CompositeExplicitAutograd` 上有实现。这两个数是"覆盖数"而非"互斥分类"（一个算子可以两条路都有），但量级已经说明：aten 里有相当大一部分算子**没有手写的 CPU kernel**，它们是配方，或者是"用默认实现顶上"。

### 4.5 一个必须诚实说明的观察

上表里 `add.Tensor` 的 CUDA 条目是 `default backend kernel`。这**不代表**上游 PyTorch 的 CUDA 没有专属 `add` kernel——本机这个 wheel 的 `torch.version.cuda` 是 `None`（macOS 的 CPU + MPS 构建，没有编译 `RegisterCUDA_*.cpp`），没编译进去的后端自然只能靠 composite 默认实现顶上。

这是"源码里有"和"本机构建里有"的差别。凡是拿 dump 表下结论，都得先问一句：**这个后端在当前这个 wheel 里被编译了吗？** 我在后文提到 CUDA 行为时都会标注这一条。

---

## 5. 优先级：谁先动手

`DispatchKeySet` 的打印顺序是**升序**（位次从低到高），所以**最后一个才是优先级最高的**。这一点不能靠猜，下面用两种互相独立的方法测。

```python
import torch

print("[方法 A] 给同一个自定义 op 的三个 key 各注册一个实现，看运行时选谁")
lib = torch.library.Library("priotest", "DEF")
lib.define("whois(Tensor x) -> str")
for key in ["CPU", "Conjugate", "Negative"]:
    lib.impl("whois", (lambda k: (lambda x: k))(key), key)

x = torch.randn(4, dtype=torch.complex64)
cases = [
    ("普通复数张量", x),
    ("conj 视图", x.conj()),
    ("neg 视图", torch._neg_view(x)),
    ("conj + neg", torch._neg_view(x.conj())),
]
for name, t in cases:
    ks = torch._C._dispatch_keyset_to_string(torch._C._dispatch_keys(t))
    print(f"  {name} → 命中 {torch.ops.priotest.whois(t)}")
    print(f"      keyset: {ks[15:-1]}")

print()
print("[方法 B] 数一数「排在它下面（优先级更低）的 key 有多少个」")
D = torch._C.DispatchKey
total = len(torch._C._dispatch_keyset_to_string(torch._C._dispatch_keyset_full()).split(", "))
print("  运行时 key 总数：", total)
for k in ["CPU", "Python", "Conjugate", "ADInplaceOrView", "AutogradCPU",
          "AutocastCPU", "PythonTLSSnapshot", "PreDispatch"]:
    low = len(torch._C._dispatch_keyset_to_string(
        torch._C._dispatch_keyset_full_after(getattr(D, k))).split(", "))
    print(f"  {k:18s} 下方 {low:3d} 个")
```

```text
[方法 A] 给同一个自定义 op 的三个 key 各注册一个实现，看运行时选谁
  普通复数张量 → 命中 CPU
      keyset: CPU, ADInplaceOrView, AutogradCPU, AutocastCPU
  conj 视图 → 命中 Conjugate
      keyset: CPU, Conjugate, ADInplaceOrView, AutogradCPU, AutocastCPU
  neg 视图 → 命中 Negative
      keyset: CPU, Negative, ADInplaceOrView, AutogradCPU, AutocastCPU
  conj + neg → 命中 Negative
      keyset: CPU, Conjugate, Negative, ADInplaceOrView, AutogradCPU, AutocastCPU

[方法 B] 数一数「排在它下面（优先级更低）的 key 有多少个」
  运行时 key 总数： 136
  CPU                下方   1 个
  Python             下方  88 个
  Conjugate          下方  91 个
  ADInplaceOrView    下方  94 个
  AutogradCPU        下方  96 个
  AutocastCPU        下方 114 个
  PythonTLSSnapshot  下方 131 个
  PreDispatch        下方 135 个
```

两种方法给出同一个结论：

- **A** 是行为证据：同一个 op、四份实现，随 keyset 里多了哪个位，命中的实现就换一个；`conj + neg` 同时有两个位时，`Negative` 赢——所以 `Negative` 的位次高于 `Conjugate`。
- **B** 是结构证据：`full_after(k)` 返回"位次低于 k 的全部 key"，数出来的个数就是 k 的位次序号。CPU 只有 1 个 key 在它下面，`PreDispatch` 有 135 个。

把两条对齐，就得到这台机器的完整阶梯（从左到右 = 位次从低到高，只列日常打交道的部分）：

| 位次 | key | 类别 |
|---|---|---|
| 最低 | `CPU` / `SparseCPU` / `QuantizedCPU` … | 后端位 |
| ↑ | `Python` | 功能位（mode） |
| ↑ | `Conjugate` / `Negative` / `ZeroTensor` | 功能位（视图语义） |
| ↑ | `ADInplaceOrView` | 功能位（视图的 in-place 检查） |
| ↑ | `AutogradCPU` …（按后端各一个） | 功能位（建图） |
| ↑ | `AutocastCPU` …（按后端各一个） | 功能位（降精度） |
| ↑ | `PythonTLSSnapshot` | 功能位（TLS 快照） |
| 最高 | `PreDispatch` / `PythonDispatcher` | 功能位（给外部抓钩子） |

这个阶梯直接对应源码里那句 "Autograd > Sparse > Quantized > Dense"——**后端位在最底下，功能位层层叠上去**，而 `PreDispatch` / `PythonDispatcher` 这两个"给外部抓钩子用"的位在最顶上。

### 5.1 读 dump 表的正确姿势

因为打印是升序，第 4.1 节那张 `add.Tensor` 的 fallback 链

```
BackendSelect → Python → FuncTorchDynamicLayerBackMode → Functionalize
→ Conjugate → Negative → ADInplaceOrView → Autocast* → ...
→ PreDispatch → PythonDispatcher
```

**要从右往左读才是执行顺序**。最右边的 `PythonDispatcher` / `PreDispatch` 最先被问，最左边的 `BackendSelect` 最后被问。

这条顺序也解释了第 3 节那个"共轭视图能正确计算"的事实：`Conjugate` 在 `CPU` 之上，所以共轭视图上的加法会先被 `Conjugate` 的 fallback 接住（把视图物化），再落在 CPU 的实现上——而不是让 CPU kernel 读着一块没共轭过的内存算出错误结果。

---

## 6. 有效集合 ≠ keyset：TLS 那两把剪刀

第 2.3 节第 1 步里提到的 `include` / `exclude` 是**线程局部的全局状态**（TLS），可以用 `torch._C` 里的私有 API 直接读。下面这段把两个集合打印出来，然后做一件更狠的事：**把某个 key 从有效集合里挖掉**，看会发生什么。

```python
import torch, re
from torch.utils._python_dispatch import TorchDispatchMode
D = torch._C.DispatchKey

def show(label, st):
    print(f"{label}: {torch._C._dispatch_keyset_to_string(st)[15:-1]}")

show("默认 include", torch._C._dispatch_tls_local_include_set())
show("默认 exclude", torch._C._dispatch_tls_local_exclude_set())

class Noop(TorchDispatchMode):
    def __torch_dispatch__(self, f, t, a, k=None):
        return f(*a, **(k or {}))
with Noop():
    show("mode 里 include", torch._C._dispatch_tls_local_include_set())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        show("autocast 里 exclude", torch._C._dispatch_tls_local_exclude_set())

print()
x = torch.randn(3)

def probe(label, fn):
    try:
        fn()
        print(f"  {label} → 成功")
    except Exception as e:
        m = re.search(r"registered for schema ([\w:.]+)", str(e))
        print(f"  {label} → {type(e).__name__}：卡在 {(m.group(1).rstrip('.') if m else '?')}")

print("把 BackendSelect 从有效集合里挖掉：")
torch._C._dispatch_tls_set_dispatch_key_excluded(D.BackendSelect, True)
for label, fn in [("torch.randn(3)", lambda: torch.randn(3)),
                  ("torch.empty(3)", lambda: torch.empty(3)),
                  ("x * 2", lambda: x * 2),
                  ("torch.add(x, x)", lambda: torch.add(x, x))]:
    probe(label, fn)
torch._C._dispatch_tls_set_dispatch_key_excluded(D.BackendSelect, False)

print("把 CPU 从有效集合里挖掉：")
torch._C._dispatch_tls_set_dispatch_key_excluded(D.CPU, True)
probe("x + 1", lambda: x + 1)
torch._C._dispatch_tls_set_dispatch_key_excluded(D.CPU, False)

print()
probe("恢复后 torch.randn(3)", lambda: torch.randn(3))
probe("恢复后 x + 1", lambda: x + 1)
```

```text
默认 include: BackendSelect, ADInplaceOrView
默认 exclude: AutocastCPU, AutocastMTIA, AutocastMAIA, AutocastXPU, AutocastIPU, AutocastHPU, AutocastXLA, AutocastMPS, AutocastCUDA, AutocastPrivateUse1
mode 里 include: BackendSelect, Python, ADInplaceOrView, PythonTLSSnapshot
autocast 里 exclude: AutocastMTIA, AutocastMAIA, AutocastXPU, AutocastIPU, AutocastHPU, AutocastXLA, AutocastMPS, AutocastCUDA, AutocastPrivateUse1

把 BackendSelect 从有效集合里挖掉：
  torch.randn(3) → NotImplementedError：卡在 aten::empty.memory_format
  torch.empty(3) → NotImplementedError：卡在 aten::empty.memory_format
  x * 2 → NotImplementedError：卡在 aten::empty_strided
  torch.add(x, x) → 成功
把 CPU 从有效集合里挖掉：
  x + 1 → NotImplementedError：卡在 aten::add.out

  恢复后 torch.randn(3) → 成功
  恢复后 x + 1 → 成功
```

这里有三个可以直接改变调试方式的事实。

### 6.1 `autocast` 不是"加一层"，是"把 key 从 exclude 里拿出来"

对比两行：

```
默认 exclude:  AutocastCPU, AutocastMTIA, AutocastMAIA, ..., AutocastPrivateUse1
autocast 里:   AutocastMTIA, AutocastMAIA, ..., AutocastPrivateUse1
```

**只有 `AutocastCPU` 从 exclude 里消失了**，其他后端一个没动。这就是 autocast 的全部实现：第 3 节看到 `AutocastCPU` 位**本来就在**每个 CPU 张量的 keyset 里，只是默认被 TLS exclude 剪掉了；进入 `torch.autocast("cpu", ...)` 相当于把剪刀挪开，让这个位重新生效。

于是"key 在 keyset 里"和"这层会生效"是两件完全不同的事。这也顺便解释了为什么 `torch.autocast("cuda")` 在 CPU 张量上什么都不做——它解开的是 `AutocastCUDA` 那把剪刀，而 CPU 张量身上根本没有那个位。

### 6.2 `TorchDispatchMode` 也不是"加一层"，是"把 Python 位塞进 include"

```
默认 include:  BackendSelect, ADInplaceOrView
mode 里 include: BackendSelect, Python, ADInplaceOrView, PythonTLSSnapshot
```

多了 `Python` 和 `PythonTLSSnapshot`。所以 mode 也不是"包在外面的一层壳"，而是**让 Python 位在派发中生效**。这个视角在写 `__torch_dispatch__` 时很有用：你拦到的是"这一关被问到的人"，不是"最外层的包装"。

### 6.3 挖掉 `BackendSelect`：工厂函数全挂，而 `x * 2` 也挂

`torch.randn(3)` 和 `torch.empty(3)` 都卡在 `aten::empty.memory_format`，这是意料之中的——工厂函数要先决定后端才能分配内存。

**意外的是 `x * 2` 也挂了**，而且卡在 `aten::empty_strided`。原因：`mul` 本身有 CPU kernel，但它要**分配输出**，而分配输出调的是 `empty_strided`，那是个工厂函数。**一次 `x * 2` 至少两次派发**，第二次在你没有意识到的位置。

而 `torch.add(x, x)` 活着。两者的差别在"输出分配时 device 从哪来"：`torch.empty(3)` 没给 device，只能靠 `BackendSelect` 从默认值推；`add` 内部的分配用的是从输入张量推出来的 device。所以它不是"没派发 `empty`"，而是"派发 `empty` 时不需要 `BackendSelect` 出手"（这一步是从两个对照结果推的，属于推断）。

### 6.4 挖掉 CPU：报错指向 `aten::add.out`

这是本期最有用的一条。`x + 1` 卡在 **`aten::add.out`**，不是 `aten::add.Tensor`。

为什么？因为 `add.Tensor` 有两套实现：CPU 上的专属 kernel（第 4 节那 15 条之一），以及一条 composite 的默认实现（`default backend kernel`）。CPU 位被剪掉之后，专属 kernel 被跳过，落到 composite 默认实现上；那条默认实现内部再去调 `add.out`，而 `add.out` 的 CPU kernel 同样被剪掉了——于是报错出现在**下游的 `add.out`**。

（这里是"从报错反推机制"，属于推断；能直接看到的是报错点名 `aten::add.out` 这个事实。）

**工程推论**：以后看到 `Could not run 'aten::xxx' with arguments from the 'yyy' backend` 这种报错，先别断定"`xxx` 没有实现"。它很可能是被某个上游算子的**兜底路径**带出来的，真正缺实现的是上游那个。第 10 节会撞到一个更极端的版本：报错点名的 `aten::empty_strided` 跟你要调的东西毫无关系。

---

## 7. 你认识的每一个"特性"，都是这张表上的一行

把第 3 节那张表的差分整理一下，就得到一张"PyTorch 功能 ↔ dispatch key"的对照表。这张表是本期真正的"围绕展开"：**你以为的六个独立特性，其实是同一套机制的不同位**。

```python
import torch, warnings
warnings.simplefilter("ignore")

def keys(t):
    s = torch._C._dispatch_keyset_to_string(torch._C._dispatch_keys(t))
    return set(s[len("DispatchKeySet("):-1].split(", "))

base = keys(torch.randn(3))
print("基线（普通 CPU float32）:", ", ".join(sorted(base)))
print()

def diff(label, t):
    k = keys(t)
    add, rm = k - base, base - k
    if not add and not rm:
        print(f"  {label}: 无变化")
        return
    parts = []
    if add:
        parts.append("+" + ", ".join(sorted(add)))
    if rm:
        parts.append("−" + ", ".join(sorted(rm)))
    print(f"  {label}: " + "  ".join(parts))

diff("conj() 视图", torch.randn(3, dtype=torch.complex64).conj())
diff("_neg_view", torch._neg_view(torch.randn(3)))
diff("_efficientzerotensor", torch.ops.aten._efficientzerotensor([3]))
diff("量化张量", torch.quantize_per_tensor(torch.randn(3), 0.1, 0, torch.quint8))
diff("sparse COO", torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.tensor([1.0, 2.0]), (3,)))
diff("meta 张量", torch.randn(3, device="meta"))
with torch.no_grad():
    diff("torch.no_grad() 里创建", torch.randn(3))
with torch.autocast("cpu", dtype=torch.bfloat16):
    diff("autocast 里创建", torch.randn(3))
with torch.inference_mode():
    diff("inference_mode 里创建", torch.randn(3))
from torch._subclasses.fake_tensor import FakeTensorMode
with FakeTensorMode():
    diff("FakeTensorMode 里创建", torch.randn(3))
diff("requires_grad=True", torch.randn(3, requires_grad=True))
```

```text
基线（普通 CPU float32）: ADInplaceOrView, AutocastCPU, AutogradCPU, CPU

  conj() 视图: +Conjugate
  _neg_view: +Negative
  _efficientzerotensor: +ZeroTensor
  量化张量: +QuantizedCPU  −CPU
  sparse COO: +SparseCPU  −CPU
  meta 张量: +AutogradMeta, Meta  −AutocastCPU, AutogradCPU, CPU
  torch.no_grad() 里创建: 无变化
  autocast 里创建: 无变化
  inference_mode 里创建: −ADInplaceOrView, AutogradCPU
  FakeTensorMode 里创建: +Python, PythonTLSSnapshot
  requires_grad=True: 无变化
```

这张表最重要的一列不是"加了什么"，而是**"无变化"那三行**。

| 触发方式 | keyset 变化 | 说明 |
|---|---|---|
| `conj()` / `_neg_view()` | `+Conjugate` / `+Negative` | **代码生成**：`Tensor.real` 之类会顺手检查并物化 |
| `_efficientzerotensor` | `+ZeroTensor` | 让所有算子知道"这个张量恒为 0，别真去算" |
| 量化 / sparse | 换后端位 | **存储**变了，不是语义变了 |
| `meta` / FakeTensor | 换或加位 | `Meta` 换掉 `CPU`；FakeTensor 额外带 `Python` |
| `torch.no_grad()` | **无变化** | 改的是全局 `GradMode`，不是 keyset |
| `torch.autocast` | **无变化** | keyset 里本来就有 `AutocastCPU`，改的是 exclude 集合 |
| `requires_grad=True` | **无变化** | 改的是 `AutogradMeta` 里的布尔值 |
| `inference_mode` | 删两个位 | 从"有 autograd 通道"变成"没有" |

### 7.1 那句"无变化"为什么重要

因为它把 PyTorch 里所有"影响计算行为"的开关分成了两类，而这两类的**作用范围完全不同**：

- **贴在张量上的**（`Conjugate`、`Negative`、`ZeroTensor`、后端位）：跟着张量走。你把这张量传进任何函数、任何线程、任何进程，行为不变。
- **挂在全局/线程局部上的**（`GradMode`、TLS exclude、mode 栈）：只对**当前线程当前时刻**的派发生效。张量一旦离开这个上下文，它的行为就回到默认。

这解释了日常里那些"为什么换个函数就不一样"的问题：`torch.no_grad()` 里创建的张量，出了上下文照样能建图（因为它的 keyset 从来没变过）；而 `conj()` 视图出了任何上下文都还是共轭视图（因为位在张量身上）。

### 7.2 六个邻居：这张表上的其他居民

顺着这张表往外看，同一套机制还养着这些邻居。它们的共同点是：**都能用"多一个位"表达**，所以都不需要改任何算子的实现。

| 邻居 | 加的位 | 你什么时候会遇到它 |
|---|---|---|
| autograd | `AutogradCPU` 等（**按后端各一个**） | 任何 `requires_grad` 的调用；`torch.autograd.Function` |
| autocast | `AutocastCPU` / `AutocastCUDA` | 混合精度训练；`torch.autocast` |
| 共轭 / 取负视图 | `Conjugate` / `Negative` | 复数运算；`t.conj()`、`t.imag` |
| functionalize | `Functionalize` | `torch.func.functionalize`、AOTAutograd 内部 |
| vmap | `FuncTorchBatched` / `Batched` | `torch.vmap`、`torch.func` |
| Meta / Fake | `Meta` / `Python` | `torch.export`、`torch.compile` 的形状推导 |
| 量化 / sparse | `QuantizedCPU` / `SparseCPU` | 量化推理；稀疏算子 |
| 新硬件 | `PrivateUse1` 及 `AutogradPrivateUse1` | 接国产加速器、自定义后端（第 10 节） |

`FuncTorchBatched` 那 614 个 op 说明了这一层的覆盖范围：**vmap 也需要给每个算子准备一份"批量语义"的实现**，600 多个算子有，其余的靠 fallback（vmap 不认识的算子会直接报错，而不是悄悄算错）。

---

## 8. 观测：`TorchDispatchMode` 看到的是什么

第 038 期留过一个二分：**profiler 看逻辑算子，dispatch mode 看原子算子**。这一节把那个二分做细——它不总成立。

```python
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

class Log(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))

def trace(label, fn, extra=None):
    m = Log()
    m.ops = []
    if extra is None:
        with m:
            fn()
    else:
        with m, extra:
            fn()
    names = [o.split(".")[1] for o in m.ops]
    print(f"  {label} → {len(names)} 个原子算子 {names}")

x = torch.randn(4, 8)
y = torch.randn(8, 8)
w = torch.ones(8)
b = torch.zeros(8)

trace("y @ y（有 CPU 内核）", lambda: y @ y)
trace("F.layer_norm（有 CPU 内核）", lambda: F.layer_norm(x, (8,), w, b))
trace("F.rms_norm（composite 配方）", lambda: F.rms_norm(x, (8,)))
trace("F.softmax（composite 配方）", lambda: F.softmax(x, -1))
trace("aten::softmax.int 直呼", lambda: torch.ops.aten.softmax.int(x, -1))
trace("autocast 下的 y @ y", lambda: y @ y,
      extra=torch.autocast("cpu", dtype=torch.bfloat16))
```

```text
  y @ y（有 CPU 内核） → 1 个原子算子 ['mm']
  F.layer_norm（有 CPU 内核） → 1 个原子算子 ['native_layer_norm']
  F.rms_norm（composite 配方） → 5 个原子算子 ['pow', 'mean', 'add_', 'rsqrt', 'mul']
  F.softmax（composite 配方） → 1 个原子算子 ['_softmax']
  aten::softmax.int 直呼 → 1 个原子算子 ['_softmax']
  autocast 下的 y @ y → 3 个原子算子 ['_to_copy', '_to_copy', 'mm']
```

四行对比出三件事。

**第一，同样"看起来一个算子"，派发次数差 5 倍。** `F.rms_norm` 展开成 5 个原子算子，因为它在 aten 里是 **composite implicit**（`aten::rms_norm`，第 4 节统计里那 783 个之一）——它没有 kernel，只有配方：`pow → mean → add_ → rsqrt → mul`。而 `F.layer_norm` 有 `native_layer_norm` 的融合 kernel，1 条。

这也解释了为什么"优化模型时数算子个数"经常没有意义：**算子的个数取决于实现的方式，而不取决于数学式子的样子**。

**第二，composite 本身在 mode 里看不见。** `aten::softmax.int` 在 aten 里是 composite（第 4 节那张 129 条的表里，104 条都是 `math kernel`），但直呼它一次，trace 里只有一条 `_softmax`——`softmax.int` 自己那条**不出现**。

（机制上我只确认了行为：三个不同来源的 composite 算子——`aten::softmax.int`、`aten::chain_matmul`、以及我自己用 `Library.impl(..., "CompositeImplicitAutograd")` 注册的算子——在 mode 里都不可见，只看到展开后的原子算子。至于它为什么越过 Python 层，我在 `DispatchKey.h` 里只找到"composite 是别名键、展开到多个 runtime 槽位"这一条，没有找到明确描述位次顺序的注释，这里存疑。）

**第三，autocast 是真在改图，不是只改 dtype。** `y @ y` 从 1 条变成 3 条：两次 `_to_copy`（把两个 fp32 输入各转一次 bf16）加一次 `mm`。所以 autocast 的"额外开销"不是零——每次 matmul 多两个转换算子。这也是"为什么 autocast 应该只包住 forward、不要包住整个 step"的一个具体理由。

### 8.1 一个和直觉相反的细节

mode 里 `func(*args)` 那次重派发是**把 Python 层关掉**的。所以你在 `__torch_dispatch__` 里调用的 `func`，它内部再调用的 aten 算子**不会再被同一个 mode 看到**。

这一条和"composite 展开能看到 5 条"并不矛盾，因为两者发生在不同位置：composite 的展开发生在 mode **之前**（所以每条都重新经过 Python 层），而 mode 里 `func()` 的重派发发生在 mode **之后**（Python 层已经被关掉）。

实用推论：**别拿 mode 的 trace 当"这个算子内部调了什么"的完整答案**。它给的是"这一层之上发生了什么"。

---

## 9. 注册：一个算子要接几个 key 才算接全

现在反过来做一个算子。下面两条路线都是官方 API，差别在"默认帮你接几个 key"。

```python
import torch

def tags(op):
    out = {}
    for line in torch._C._dispatch_dump_table(op).strip().split("\n"):
        k, rest = line.split(":", 1)
        out[k.strip()] = rest[rest.rfind("[") + 1: rest.rfind("]")]
    return out

def show(op, keys):
    t = tags(op)
    for k in keys:
        print(f"  {k:12s} {t.get(k, '（没注册）')}")

print("[路线 A] torch.library.Library + impl(..., 'CPU')")
lib = torch.library.Library("rawops", "DEF")
lib.define("rmsnorm(Tensor x, Tensor w, float eps=1e-6) -> Tensor")
lib.impl("rmsnorm", lambda x, w, eps=1e-6: w * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps), "CPU")
show("rawops::rmsnorm", ["CPU", "Meta", "AutogradCPU", "Undefined"])

x = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
w = torch.ones(4, dtype=torch.double, requires_grad=True)
y = torch.ops.rawops.rmsnorm(x, w)
print("  前向 requires_grad:", y.requires_grad, "| grad_fn:", type(y.grad_fn).__name__)
y.sum().backward()
print("  反向能跑，dx 的形状:", tuple(x.grad.shape))
try:
    torch.ops.rawops.rmsnorm(torch.randn(2, 4, device="meta"), torch.randn(4, device="meta"))
except Exception as e:
    print("  meta 张量 →", type(e).__name__, "：",
          str(e).split("You may have run into")[0].strip())

print()
print("[路线 B] torch.library.custom_op 四件套")

@torch.library.custom_op("cooked::rmsnorm", mutates_args=())
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return w * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

@rmsnorm.register_fake
def _(x, w, eps=1e-6):
    return torch.empty_like(x)

@rmsnorm.register_kernel("cpu")
def _(x, w, eps=1e-6):
    return w * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def setup_context(ctx, inputs, output):
    x, w, eps = inputs
    ctx.save_for_backward(x, w)
    ctx.eps = eps

def backward(ctx, grad_out):
    x, w = ctx.saved_tensors
    n = x.shape[-1]                      # 归一化维度的长度
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + ctx.eps)
    # dL/dx = rstd*(w*g) - rstd^3 * x * sum_k(w_k g_k x_k) / n
    dx = rstd * (grad_out * w - rstd ** 2 * x * (grad_out * w * x).sum(-1, keepdim=True) / n)
    dw = (grad_out * x * rstd).sum(dim=0)  # 沿 batch 维求和，回到 w 的形状
    return dx, dw, None

torch.library.register_autograd("cooked::rmsnorm", backward, setup_context=setup_context)
show("cooked::rmsnorm", ["CPU", "Meta", "AutogradCPU", "Undefined"])

x2 = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
w2 = torch.ones(4, dtype=torch.double, requires_grad=True)
y2 = torch.ops.cooked.rmsnorm(x2, w2)
print("  前向 requires_grad:", y2.requires_grad, "| grad_fn:", type(y2.grad_fn).__name__)
print("  meta 张量 → 形状", tuple(torch.ops.cooked.rmsnorm(
    torch.randn(2, 4, device="meta"), torch.randn(4, device="meta")).shape))
print("  gradcheck（拿数值微分对账）:", torch.autograd.gradcheck(
    lambda a, b: torch.ops.cooked.rmsnorm(a, b), (x2, w2)))
```

```text
[路线 A] torch.library.Library + impl(..., 'CPU')
  CPU          kernel
  Meta         backend fallback
  AutogradCPU  backend fallback
  Undefined    （没注册）
  前向 requires_grad: True | grad_fn: DivBackward0
  反向能跑，dx 的形状: (2, 4)
  meta 张量 → NotImplementedError ： rawops::rmsnorm: attempted to run this operator with Meta tensors, but there was no fake impl or Meta kernel registered.

[路线 B] torch.library.custom_op 四件套
  CPU          kernel
  Meta         kernel
  AutogradCPU  autograd kernel
  Undefined    default backend kernel
  前向 requires_grad: True | grad_fn: GeneratedBackwardFor_cooked_rmsnorm_defaultBackward
  meta 张量 → 形状 (2, 4)
  gradcheck（拿数值微分对账）: True
```

### 9.1 两条路的差别，全在"接了几个 key"

| key | 路线 A（只 `impl("CPU")`） | 路线 B（`custom_op` 四件套） |
|---|---|---|
| `CPU` | `kernel` | `kernel` |
| `Meta` | `backend fallback`（报错） | `kernel`（能推形状） |
| `AutogradCPU` | `backend fallback`（`VariableFallbackKernel`） | `autograd kernel`（`GeneratedBackwardFor...`） |
| `Undefined` | 没注册 | `default backend kernel`（catch-all） |

路线 A 的 `grad_fn` 是 **`DivBackward0`**——注意这个名字：它不是我那个算子的反向节点，而是**实现内部那个除法**的反向节点。也就是说，`requires_grad` 传下来了、反向也能跑，但图里展开的是我实现里的每一步，而不是"一个 rmsnorm 节点"。

更麻烦的是它同时会打一条 warning（我这次运行时的原文）：

```
rawops::rmsnorm: an autograd kernel was not registered to the Autograd key(s)
but we are trying to backprop through it. This may lead to silently incorrect
behavior. This behavior is deprecated and will be removed in a future version ...
```

"may lead to silently incorrect behavior" 这句是实话：`VariableFallbackKernel` 的做法是"让反向穿过去，指望实现内部的算子自己建图"。如果实现内部有 `detach`、`item()`、或者写回 in-place 的地方，梯度就断在那里，**而你不会收到任何报错**。

路线 B 的 `grad_fn` 是 `GeneratedBackwardFor_cooked_rmsnorm_defaultBackward`——一个以我的算子命名的节点，图里就是一个 rmsnorm。加上 `register_fake` 之后，meta 张量也能跑（`torch.export` / `torch.compile` 靠这一条）。

### 9.2 手写 backward 一定会写错的地方

这次的 backward 我第一版只写了 `grad_out * w`，`gradcheck` 直接报 `Jacobian mismatch`：**解析出来的是单位阵，数值微分算出来的是稠密矩阵**。

原因：RMSNorm 的输入$x$在归一化维度上是**互相耦合**的——$y_k = w_k x_k / s$，而 $s$ 依赖整个向量：

$$
\frac{\partial y_k}{\partial x_j} = \frac{w_k \delta_{kj}}{s} - \frac{w_k x_k x_j}{n s^3}
$$

第二项就是"耦合"。只写 `grad*w` 等于只保留了 $\delta_{kj}$ 那一半。正确形式：

$$
\frac{\partial L}{\partial x_j} = \frac{1}{s}\left(w_j g_j - \frac{x_j}{n}\sum_k \frac{w_k g_k x_k}{s^2}\right)
$$

凡是"归一化"类算子（LayerNorm、RMSNorm、Softmax、BatchNorm）的反向都有这一项，而它恰好是最容易漏的。

`gradcheck` 是这件事的**唯一可靠检查**：它不看你的推导，直接拿数值微分对账。写自定义算子时，它是必跑项，不是可选项。

---

## 10. 接一块新硬件：`PrivateUse1` 的三层

`PrivateUse1` 是 PyTorch 留给"框架之外的后端"的那个位。这一节把它的接入过程拆成三层，每层都会撞到一个**不同的报错**——这三个报错本身就是一张路线图。

```python
import sys, types, torch

def dump_lines(op):
    return torch._C._dispatch_dump_table(op).strip().split("\n")

def count(op, word):
    return sum(1 for l in dump_lines(op) if word in l)

print("[第 1 层] 只改名")
before = count("aten::add.Tensor", "PrivateUse1")
torch.utils.rename_privateuse1_backend("ifwa")
print("  torch.device('ifwa:0') →", torch.device("ifwa:0"))
print("  aten::add.Tensor 里含 'PrivateUse1' 的行数：改名前", before,
      "→ 改名后", count("aten::add.Tensor", "PrivateUse1"),
      "| 含 'ifwa' 的行数：", count("aten::add.Tensor", "ifwa"))
try:
    torch.empty(2, device="ifwa")
except Exception as e:
    print(f"  torch.empty(2, device='ifwa') → {type(e).__name__}：{e}")

print()
print("[第 2 层] 补一个 device module")
sys.modules["torch.ifwa"] = types.ModuleType("torch.ifwa")
torch._register_device_module("ifwa", sys.modules["torch.ifwa"])
try:
    torch.empty(2, device="ifwa")
except Exception as e:
    msg = str(e)
    head = msg.split(". This could be because")[0].strip()
    lst = msg[msg.find("only available for these backends"):].split("[")[1].rstrip("]")
    items = [x.strip() for x in lst.split(",")]
    print(f"  torch.empty(2, device='ifwa') → {type(e).__name__}")
    print("    ", head)
    print("     only available for these backends:", ", ".join(items[:9]),
          f"…（共 {len(items)} 个，不含 PrivateUse1）")

print()
print("[第 3 层] 把一个算子注册到 PrivateUse1")
lib = torch.library.Library("myops", "DEF")
lib.define("foo(Tensor x) -> Tensor")
lib.impl("foo", lambda x: x + 1, "PrivateUse1")
for l in dump_lines("myops::foo"):
    k = l.split(":")[0]
    if k in ("PrivateUse1", "AutogradPrivateUse1", "CPU"):
        print(f"  {k:18s} {l[l.rfind('[') + 1: l.rfind(']')]}")
print("  Tensor.ifwa 方法存在？", hasattr(torch.Tensor, "ifwa"))
torch.utils.generate_methods_for_privateuse1_backend()
print("  generate_methods 之后：", hasattr(torch.Tensor, "ifwa"))
try:
    torch.ones(2).ifwa()
except Exception as e:
    print(f"  torch.ones(2).ifwa() → {type(e).__name__}："
          + str(e).split(". This could be because")[0])
```

```text
[第 1 层] 只改名
  torch.device('ifwa:0') → ifwa:0
  aten::add.Tensor 里含 'PrivateUse1' 的行数：改名前 4 → 改名后 4 | 含 'ifwa' 的行数： 0
  torch.empty(2, device='ifwa') → ModuleNotFoundError：No module named 'torch.ifwa'

[第 2 层] 补一个 device module
  torch.empty(2, device='ifwa') → NotImplementedError
     Could not run 'aten::empty.memory_format' with arguments from the 'ifwa' backend
     only available for these backends: CPU, MPS, Meta, QuantizedCPU, QuantizedMeta, MkldnnCPU, SparseCPU, SparseMPS, SparseMeta …（共 54 个，不含 PrivateUse1）

[第 3 层] 把一个算子注册到 PrivateUse1
  PrivateUse1        kernel
  AutogradPrivateUse1 backend fallback
  Tensor.ifwa 方法存在？ False
  generate_methods 之后： True
  torch.ones(2).ifwa() → NotImplementedError：Could not run 'aten::empty_strided' with arguments from the 'ifwa' backend
```

### 10.1 三个报错就是三张任务单

| 层 | 报错 | 缺什么 |
|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'torch.ifwa'` | 缺 **device module**（`torch.<name>` 模块，PyTorch 会去 import 它） |
| 2 | `Could not run 'aten::empty.memory_format' with arguments from the 'ifwa' backend` | 缺 **内存分配器 / device guard**（C++ 侧 `PrivateUse1HooksInterface`），54 个可用后端里没有它 |
| 3 | `Could not run 'aten::empty_strided' ...` | 缺**所有**算子的实现（`torch.ones(2).ifwa()` 要分配内存，又回到 `empty_strided`） |

第 1 层的输出里还有一个细节值得盯：**改名前后 dump 表里 `PrivateUse1` 的行数都是 4，含 `ifwa` 的行数是 0**。也就是说，`rename_privateuse1_backend("ifwa")` 改的是**设备名的解析**（`torch.device("ifwa:0")` 能造出来），但**dispatch key 的名字不变**——dump 表里永远打印 `PrivateUse1`。

这条对读日志很重要：**日志里出现 `PrivateUse1` 不代表你的后端没被识别，那是这个 key 的固定名字。**

### 10.2 第 3 层看到的两个 key

注册一个算子到 `PrivateUse1` 之后，dump 表里出现的是**两个**位：

- `PrivateUse1` → `kernel`（你注册的那个）
- `AutogradPrivateUse1` → `backend fallback`（框架自动挂的）

注意它是 `AutogradPrivateUse1` **而不是** `AutogradCPU`。这印证了第 3 节那条观察：**autograd 是按后端分位的**，所以接一块新硬件时，autograd 那一关也要单独接——这就是为什么"能跑前向"和"能训练"在接入新后端时是两个独立的里程碑。

### 10.3 剩下的部分在 C++ 侧

Python 能给的东西到第 3 层为止。真正要让 `torch.empty(2, device="ifwa")` 成功，还需要 C++ 侧注册：allocator、device guard、`PrivateUse1HooksInterface`、以及各算子的实现（这一批通常由代码生成从 `native_functions.yaml` 生成，而不是手写）。

**这一段是"文档里看到的"，不是本机实测**——本机没有可用的私有后端，我只能验证到"缺什么会报什么"。真要接硬件，`torch.utils.rename_privateuse1_backend` + `generate_methods_for_privateuse1_backend` 是 Python 侧的全部入口，剩下都在 C++。

---

## 11. 派发本身有多贵

最后一个问题：这套机制的开销。下面用成对交替采样（每轮测一次候选再测一次基线，取逐对比值的中位数）测 5 种调用写法。张量只有 4 个元素，所以测到的几乎全是**框架开销**，不是计算。

```python
import torch, time, statistics

N = 20000          # 每轮迭代次数
ROUNDS = 11        # 交替采样轮数

def timed(fn):
    t0 = time.perf_counter_ns()
    for _ in range(N):
        fn()
    return time.perf_counter_ns() - t0

pyops = torch.library.Library("pyops", "DEF")
pyops.define("bump(Tensor x) -> Tensor")
pyops.impl("bump", lambda x: x + 1, "CPU")

a = torch.randn(4)
b = torch.randn(4)
base_fn = lambda: a + b

variants = [
    ("a + b（Tensor.__add__）", base_fn),
    ("torch.add(a, b)", lambda: torch.add(a, b)),
    ("torch.ops.aten.add.Tensor(a, b)", lambda: torch.ops.aten.add.Tensor(a, b)),
    ("torch.ops.aten.add(a, b)（包）", lambda: torch.ops.aten.add(a, b)),
    ("torch.ops.pyops.bump(a)（Python kernel）", lambda: torch.ops.pyops.bump(a)),
]

for fn in (base_fn, *[f for _, f in variants[1:]]):      # 预热
    timed(fn)

print(f"每次调用耗时（中位数，{N} 次/轮 × {ROUNDS} 轮交替采样）")
base_rounds = []
for name, fn in variants[:1]:
    ts = [timed(fn) for _ in range(ROUNDS)]
    base_rounds = ts
    med = statistics.median(ts) / N / 1000
    print(f"  {name}: {med:.2f} µs")

for name, fn in variants[1:]:
    ratios, samples = [], []
    for i in range(ROUNDS):
        t = timed(fn)
        tb = timed(base_fn)
        ratios.append(t / tb)
        samples.append(t / N)
    med = statistics.median(samples) / 1000
    r = statistics.median(ratios)
    print(f"  {name}: {med:.2f} µs，是 a+b 的 {r:.2f}×")

base_med = statistics.median(base_rounds) / N / 1000
print(f"  基线 a+b: {base_med:.2f} µs")
```

下面是一次真实运行的输出（微秒级的数字每次会有几个百分点的抖动，所以没有放进逐行比对的自检里）：

    每次调用耗时（中位数，20000 次/轮 × 11 轮交替采样）
      a + b（Tensor.__add__）: 0.36 µs
      torch.add(a, b): 0.38 µs，是 a+b 的 1.06×
      torch.ops.aten.add.Tensor(a, b): 0.66 µs，是 a+b 的 1.88×
      torch.ops.aten.add(a, b)（包）: 0.66 µs，是 a+b 的 1.88×
      torch.ops.pyops.bump(a)（Python kernel）: 1.30 µs，是 a+b 的 3.61×
      基线 a+b: 0.36 µs

三点结论：

1. **`a + b` 和 `torch.add(a, b)` 几乎同价**（约 0.36 µs）。`Tensor.__add__` 这个 slot wrapper 和 `torch.add` 这个 Python 函数最终走的是同一条 C 路径，没有谁更"直接"。
2. **`torch.ops.aten.add.Tensor(...)` 贵了近一倍**（约 0.66 µs，1.9×）。多出来的那 0.3 µs 是 Python 对象的一层包装（`OpOverload` 的 `__call__`），**不是查表变慢了**。查表本身是"取 keyset 的最高位当数组下标"，在 C++ 侧是几条指令。
3. **Python 实现的自定义算子再翻一倍**（约 1.30 µs，3.6×）。每调用一次要跨一次 Python/C++ 边界、组一次参数栈。

量的意义在这里：在 CPU 上 0.36 µs 无所谓；但 GPU 上一个 kernel 的端到端耗时常常也就是几微秒量级，如果你的 step 里有几千个 Python 级 op 调用，**光派发就是毫秒量级**，而这部分完全串行、完全在 CPU 上、完全不和 GPU 重叠。第 001 期（CUDA Graph）和第 017 期（CUDA Streams）解决的就是这一层的浪费——捕获期把"查表结果"冻成一条固定序列，运行时不再查。

---

## 12. 什么时候该用 / 不该用

### 该用（把逻辑放进 dispatcher）

| 场景 | 为什么 |
|---|---|
| 给一块新硬件接 PyTorch | 只有走 `PrivateUse1`，上层所有 `nn.Module`、`torch.compile`、autograd 才会自动认得它（第 10 节） |
| 写一个会被别人复用的算子 | 注册成 op 之后，autocast / vmap / fake tensor / export **免费**继承——前提是你把该接的 key 接全（第 9 节） |
| 需要被 `torch.compile` 看见 | Dynamo 抓的是 aten 图；纯 Python 函数会被 inline 展开成一堆小算子（第 8 节的 `rms_norm` 就是反面教材） |
| 需要给框架打补丁式的接管 | `Library.impl(..., "Backend")` 可以在不改源码的情况下换掉某个后端的某个算子 |

### 不该用（绕开 dispatcher）

| 场景 | 为什么 |
|---|---|
| kernel 内部的热循环 | 循环体里每一次 aten 调用都要重新走一遍派发（0.36 µs 起）。`TensorIterator`、`at::parallel_for` 这类东西存在的理由就是"进一次派发，干很多活" |
| 纯 Python 原型 | `def my_op(x): return x * 2` 就够了，注册 op 只会多一层开销和一堆要维护的 key |
| 只为一个模型写的一次性算子 | 注册它意味着你要同时维护 fake / autograd / 各后端；除非要 export，否则 `torch.autograd.Function` 更省事 |
| 想"优化掉"派发开销 | 派发不是瓶颈，`Python → C++` 的边界才是。减少 Python 侧调用次数，比换调用写法有效一个量级 |

### 一条判据

**问自己："这个算子的正确性依赖于哪些语义层？"**

- 只依赖"数据在哪"→ 注册一个后端 kernel 就够（路线 A）。
- 依赖 autograd → 必须 `register_autograd`（否则会拿到 `VariableFallbackKernel` 的"可能静默错误"）。
- 依赖 shape 推导（要 export / compile）→ 必须 `register_fake`。
- 依赖 autocast → 要 `register_autocast`。

少接哪一层，就会在**那一层生效的那天**炸——而且往往不是在你的代码里炸。

---

## 13. 常见坑

### 坑 1：没有 autograd 注册 ≠ 不可导

路线 A 的表现极具欺骗性：`requires_grad=True` 传下去了、`backward()` 能跑、`x.grad` 有值、形状也对——但图里是**实现内部的算子**，而且框架只打一条 warning（第 9 节引了原文），不会报错。

判据很简单：看 `grad_fn` 的名字。**以你的算子命名的 `grad_fn` 才说明你接上了 autograd 那一层**；看到 `DivBackward0`、`MulBackward0` 这种，说明你在靠实现的副作用建图。

### 坑 2：没有 `register_fake` 的算子，报错会指向别的地方

路线 A 的算子对 meta 张量调用，报错是"这个算子没有 fake/Meta 实现"——这条还算清楚。但如果你在 `torch.compile` / `torch.export` 里用，报错经常出现在**上游的分配调用**上（第 10 节 `aten::empty_strided` 那个形状），离你真正的问题隔了三四层。

判据：注册完 op，立刻跑一次 `torch.export.export` 或者 `torch.ops.myop(x.to("meta"))`。**这一步不通过，就不要往下写训练代码**。

### 坑 3：别用 dispatch mode 数 composite 算子的调用次数

第 8 节实测：`F.softmax` 在 mode 里只有 1 条 `_softmax`，`aten::softmax.int` 自己**不出现**。如果你的统计口径是"数 mode 里出现的算子"，那么：

- composite 算子会被算成它展开后的原子算子（数量偏多）；
- 反过来说，`F.rms_norm` 那种 5 条的，也不是"模型真的调了 5 次"。

**做算子计数请用 profiler（逻辑算子）**，做"底层到底执行了什么"请用 mode。两者不能混。

### 坑 4：只注册了一个后端，却在另一个后端上跑

最常见的形态是：在 CPU 上测通了，换到 CUDA（或自定义设备）就报 `Could not run 'aten::xxx' with arguments from the 'cuda' backend`。这时候的正确反应不是"我没注册这个算子"——而是**把那个算子的 dump 表打出来，看目标后端的条目是 `kernel` 还是 `default backend kernel`**。

`default backend kernel` 意味着"这个后端用的是 composite 的默认实现"，它可能能跑（比如 `add.Tensor`），也可能跑到一半才失败（第 6.4 节 `add.out` 那个例子）。**看表比看报错可靠。**

---

## 14. 一句话总结

> **keyset 说的是"这次调用要经过哪些关卡"，派发表说的是"每关谁负责"，优先级决定先问谁。** 你认识的每一个 PyTorch 特性——autograd、autocast、共轭视图、vmap、fake tensor、量化、sparse、新硬件——都只是这张表上的一个位。所以判断"某个 API 能不能用"的正确方法不是读文档，而是 `torch._C._dispatch_dump_table(...)` 看一眼那个位置上是 `kernel` 还是 `backend fallback`。

---

## 15. 今日练习

<details>
<summary>今日练习（点击展开参考答案）</summary>

**练习 1**：不用 `torch._C._dispatch_keys`，只用"注册三个实现"的办法，判断 `torch._C.DispatchKey.ZeroTensor` 和 `DispatchKey.Negative` 谁优先级更高。

参考答案：

```python
import torch

lib = torch.library.Library("q1", "DEF")
lib.define("whois(Tensor x) -> str")
for key in ["CPU", "Negative", "ZeroTensor"]:
    lib.impl("whois", (lambda k: (lambda x: k))(key), key)

x = torch.randn(4)
z = torch.ops.aten._efficientzerotensor([4])          # 带 ZeroTensor 位
n = torch._neg_view(z)                                # 再叠一个 Negative 位
print(torch._C._dispatch_keyset_to_string(torch._C._dispatch_keys(n)))
print(torch.ops.q1.whois(n))
```

`_neg_view` 一个零张量，两个位都在；谁被命中就说明谁位次高。实跑结果是 **`ZeroTensor` 赢**——keyset 打印出来是

    DispatchKeySet(CPU, Negative, ZeroTensor, ADInplaceOrView, AutogradCPU, AutocastCPU)

`ZeroTensor` 排在 `Negative` **后面**，而第 5 节测出来"打印顺序 = 位次升序"，所以 `ZeroTensor` 更高。这条也顺便证明了"ZeroTensor 是一个真会参与派发的位"，而不是某个 Python 层的标记。

**练习 2**：`torch.no_grad()` 里创建的张量，出了上下文还能不能 `backward()`？先预测，再验证，然后用 keyset 解释。

参考答案：能（如果它 `requires_grad=True` 且参与了一次在 grad 模式下的计算）。`no_grad` 改的是全局 `GradMode`，张量的 keyset 一个位都没变（第 7 节实测"无变化"）。所以"在 no_grad 里创建的张量不能求导"这个说法是错的——不能求导的是**在 no_grad 里做的那些运算**（因为它们没建图），而张量本身的能力没变。

**练习 3**：写一个自定义算子，要求它在 `torch.export` 下可用、能训练、且 `gradcheck` 通过。最少要注册几个东西？

参考答案：三个。`custom_op` 定义 + `register_fake`（export 要）+ `register_autograd`（训练要，且必须 gradcheck 通过）。如果想让它被 `torch.autocast` 覆盖，还要第四个 `register_autocast`。第 9 节的 `cooked::rmsnorm` 就是前三件。

**练习 4**：一个算子在 CUDA 上跑出 `Could not run 'aten::empty_strided'`，但你的代码里根本没调用过 `empty_strided`。怎么定位？

参考答案：先怀疑"某个上游算子走了 composite 兜底"。三步：① 打出那个算子的 dump 表，看目标后端的条目类型；② 如果看到 `default backend kernel`，说明走的是兜底，兜底内部会调 `empty_strided` 分配输出；③ 用 `TorchDispatchMode` 把调用序列打印出来，看最后一条原子算子是什么。第 6.4 节的 `add.Tensor` → `add.out` 是同一个形状的例子。

</details>
