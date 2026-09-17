# PyTorch 每日一课 · 第 021 期

## 动态形状：PyTorch 的自由，编译器的噩梦

> **日期**：2026-09-17
> **难度**：⭐⭐⭐⭐
> **前置知识**：了解 `torch.compile` 大致在做什么即可（第 003 期），本篇独立可读
> **预计阅读时间**：15 分钟

---

## 一、这个领域解决什么问题

先看一段最普通的 PyTorch 代码：

```python
import torch

for batch in dataloader:
    x = batch["input_ids"]          # 形状可能是 (8, 512)，也可能是 (8, 137)
    y = model(x)                    # 每次形状不同，代码照跑不误
```

在 eager 模式下，张量形状变化是**完全透明**的：`x` 的形状不同，PyTorch 就按那个形状重新分配显存、重新发射 kernel。你甚至不会意识到"形状"这件事存在。

但只要你开始**编译**——`torch.compile`、`torch.export`、CUDA Graph、TensorRT、ONNX——形状立刻从"背景设施"变成"头号矛盾"。

原因很直白：**编译的本质是把"逐次运行时决策"提前到"编译期一次性决策"**。编译器需要知道：这个 kernel 的循环要展开几次？这块 buffer 要开多大？共享内存要分配多少？能不能用向量化的 128-bit 加载？

- 如果形状是 `(8, 512)`，编译器可以直接写一个专为 `8×512` 定制的 kernel：循环边界是常数，能完全展开、能对齐、能做常量折叠。
- 如果形状是"某个未知的 `s0 × s1`"，编译器只能写一个通用循环——它失去了所有常量信息，只能保守地生成代码。

于是冲突出现了：

| | 静态形状 | 动态形状 |
|---|---|---|
| 编译器视角 | 天堂：形状是常数，一切可特化 | 噩梦：边界未知，只能保守生成 |
| 用户视角 | 束缚：必须 pad 到固定长度 | 自由：想传什么形状就传什么 |
| 换形状时 | 需要重新编译 / 或大量浪费算力在 padding | 直接跑，但优化空间变小 |

**这个领域要解决的核心问题就是：如何让"用户享受形状自由"和"编译器享受特化收益"这两件事尽量兼得。**

它不是一个新问题。TensorFlow 1.x 的 graph mode 逼着所有人把序列 pad 到固定长度，这就是当年 NLP 开发者最痛恨的体验之一。PyTorch 靠"define-by-run"赢得了人心，但当 PyTorch 自己也走向编译（2.0 之后的 `torch.compile`）时，它必须正面回答这个问题——而且答案不能是"请你自己 pad"。

真正让它变成必须掌握的知识的场景：

- 训练 NLP 模型，序列长度从 12 到 2048 都有；
- 训练的最后一个 batch 往往不满（`drop_last=False`）；
- 目标检测里每张图的候选框数量不同；
- 推理服务里每个用户请求的长度、batch 组成都不同；
- 你用了 `torch.compile`，却发现"怎么比 eager 还慢"——大概率就是形状问题。

---

## 二、核心思想：从"具体数字"到"符号变量"

### 类比：裁缝与成衣

- **Eager 模式** = 每次给你现量体裁衣。慢，但任何身材都合身。
- **静态形状编译** = 提前做好的成衣。极快，但只有标准身材能穿；身材变了只能重做一件（重编译）。
- **动态形状编译** = 可调节松紧的成衣。把"腰围"从固定数字变成一个**旋钮**（符号变量），一件衣服适配一个范围。代价是版型不能为某个具体腰围做到极致贴身了。

这就是全部故事的核心：**把形状里的某些维度从常量提升为符号（Symbolic）变量。**

### 三个关键机制

#### 1. SymInt：形状可以是一个"表达式"

在 `torch.compile` 的世界里，`x.size(0)` 返回的可能不是 `8`，而是一个符号整数 `s0`。它支持的运算也变成符号运算：

```python
s0, s2 = x.size(0), x.size(2)
out = torch.empty(s0, s2)     # 形状是一个"关于符号的表达式"
```

这样编译器就能生成**一个**同时适用于多种形状的图，而不是为每种形状生成一个图。

#### 2. Guard（守卫）：什么情况下这张图还能用？

编译器为每个编译产物记下一组**前提条件**，运行时逐条检查：

```
guard: x.dtype == torch.float32
guard: x.device == cuda:0
guard: x.stride() == (512, 1)
guard: x.size(0) == 8          # ← 静态形状时是这一条
```

- 全部通过 → 复用已编译的图，零开销。
- 任一条不通过 → **重编译（recompile）**：丢掉旧图，从头再编译一次。

静态形状之所以脆，就是因为 `x.size(0) == 8` 这种 guard 极其容易被打破。一旦把这一维符号化，guard 变成"`x.size(0)` 落在 [1, ∞) 内"，就能容纳一系列形状——这就是动态形状省下重编译的原理。

#### 3. 重编译是有上限的，超了就悄悄认输

`torch._dynamo.config.cache_size_limit` 默认是 **8**：同一个函数最多缓存 8 个编译产物。第 9 种形状出现时，Dynamo 会**放弃编译，静默回退到 eager**。

这个设计极其阴险——它不会报错，你只会发现"训练好像没变快"。很多人第一次用 `torch.compile` 觉得"没效果"，根源就在这。

### 动态形状不是免费的

把维度符号化之后，编译器丢了什么？

- **无法常量折叠**：`512 * 4` 能算成 `2048`，`s0 * 4` 只能留成表达式。
- **无法完全展开循环**：不知道循环几次，就没法 unroll。
- **无法保证对齐**：不知道 `s1 % 8 == 0`，就不能放心用向量化加载。
- **需要间接寻址**：地址偏移变成运行时计算，多几条指令、多几个寄存器。
- **自动调优更难**：Triton kernel 的 `BLOCK_SIZE` 该怎么选？静态时能大胆选，动态时只能妥协。

实践中的经验判断：**同一个 kernel，动态形状版本通常比静态版本慢一些（常见是几个百分点到百分之几十），但比因为重编译而回退到完全未优化的执行路径要快得多。** 所以动态形状的正确理解不是"更快"，而是**"在形状会变的场景下，把最坏情况变好"**。

### 一个更聪明的中间态：形状分桶

既然"全静态"太脆、"全动态"太慢，工业界普遍走中间路线——**把连续无限的形状空间，离散化成有限几个桶（bucket）**：

```python
def bucket_size(n, buckets=(1, 2, 4, 8, 16, 32)):
    for b in buckets:
        if n <= b:
            return b
    return n
```

- 训练变长序列：按长度分桶、桶内 padding（`group_by_length` 就是干这个的）。
- 推理服务：只为几个常见 batch size 捕获 CUDA Graph（vLLM 就是这个思路：1/2/4/8/16… 各抓一张图）。
- TensorRT：显式声明 optimization profile 的 min/opt/max。

这是本领域最值得记住的工程直觉：**动态形状是能力上限，分桶是把能力花在刀刃上的方式。**

---

## 三、在 PyTorch 中怎么用

以下代码在 CPU 上即可运行（把 `device` 换成 `"cuda"` 收益更明显）。

### 1. 先学会"看见"重编译

不度量就没有优化。最省事的办法是打开编译日志：

```bash
# 终端里运行（或设进环境变量）
TORCH_LOGS="recompiles" python train.py
```

你会看到类似这样的输出：

```
Recompiling function forward in /path/model.py:42
triggered by the following guard failure(s):
    - tensor 'L['x']' size mismatch at index 0. expected 8, actual 4
```

**"size mismatch at index 0"** 这句话，就是本领域所有问题的起点。它精确告诉你：哪个张量的哪一维打破了 guard。

Python 里也可以用计数器统计：

```python
import torch
import torch._dynamo as dynamo

@torch.compile
def f(x):
    return (x * 2).relu().sum()

# 用 5 种不同 batch size 调用
for n in [8, 16, 32, 8, 16]:
    f(torch.randn(n, 64))

# 注意：counters 是 defaultdict 风格的内部对象，没有 .get()，
# 直接用 [] 取值即可（缺失的 key 会返回空字典，不会报错）
stats = dynamo.utils.counters["stats"]
print("unique_graphs:", stats["unique_graphs"])   # 编译出了几张图

for reason, cnt in dynamo.utils.counters["recompiles"].items():
    print(f"{cnt:>3} 次  <- {reason[:70]}")
```

默认（`dynamic=None`）时，Dynamo 会先按静态编译，只有当它检测到"形状反复变化、重编译快要撞上限"时，才**自动**把动态的那一维符号化。这就是 `unique_graphs` 通常停在 2 而不是 5 的原因：第一张静态图之后，它自己转成了动态图。

### 2. 主动指定哪些维度是动态的

别让编译器猜。你比它清楚"哪些维是业务上会变的"。

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = Model().eval()
opt_model = torch.compile(model)

x = torch.randn(8, 64)

# 只把第 0 维（batch）标记为动态；第 1 维（特征维）保持静态
torch.compiler.mark_dynamic(x, 0)
# 老版本从 torch._dynamo 导入：from torch._dynamo import mark_dynamic

with torch.no_grad():
    for n in [8, 3, 17, 64]:       # 形状五花八门
        y = opt_model(torch.randn(n, 64))
        print(n, "->", tuple(y.shape))
```

这里的关键是**精确标记**：只让真正会变的那一维动态化，其余维保持静态，编译器才能在能特化的地方继续特化。

### 3. `dynamic=True` 一把梭

不想逐个标记，就整体开启动态形状：

```python
# 所有输入维度都按动态处理：最省心，也最容易损失性能
opt_model = torch.compile(model, dynamic=True)

# 反过来：强制静态，形状一变就重编译。形状真的固定时，这是最优选择
opt_model_static = torch.compile(model, dynamic=False)
```

三种模式的取舍：

| 写法 | 行为 | 适合 |
|---|---|---|
| `dynamic=False` | 永远按静态编译 | 形状绝对固定（如固定 224×224 分类） |
| `dynamic=None`（默认） | 先静态，压力大时自动转动态 | 不知道该选什么时，从这开始 |
| `dynamic=True` | 从一开始就动态 | 形状一定会变、且变化范围大 |
| `mark_dynamic(x, dim)` | 精确控制哪一维动态 | 生产环境推荐，收益最高 |

### 4. 用 `min`/`max` 告诉编译器"范围"

动态形状不只是"未知"，还可以是"已知区间"。给范围能让编译器做出更好的决策（比如知道上界，就能预留 buffer 或选择合适的 tile 尺寸）：

```python
x = torch.randn(8, 64)
torch.compiler.mark_dynamic(x, 0, min=1, max=128)   # 只接受 1..128 的 batch
```

### 5. 导出时声明动态形状：`torch.export`

做推理部署时，动态形状必须在**导出那一刻**就说清楚，否则导出的是死图。

```python
import torch
from torch.export import export, Dim

class Model(torch.nn.Module):
    def forward(self, x):
        return (x * 2).relu()

m = Model().eval()
x = torch.randn(8, 64)

batch = Dim("batch", min=1, max=64)   # 声明一个带范围的符号维度
length = Dim("length")                # 不限范围

ep = export(m, (x,), dynamic_shapes={"x": {0: batch, 1: length}})

# 检查导出时的 guard/断言，确认动态性真的生效了
print(ep.graph_signature.input_specs)
print(ep.range_constraints)           # 每个符号维的取值区间
```

`range_constraints` 是导出的"契约"：它写明这个模型承诺只在哪些形状区间内被调用。**部署方如果打破了它，不保证正确性**——这是动态形状最容易踩的坑之一。

### 6. 序列分桶：训练侧的落地套路

这是把动态形状用好的经典模式。核心思路是"让形状变化落在有限几个值上"：

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        n = self.lengths[i]
        return torch.randn(n, 64), n

def collate(batch):
    """把同一批里的序列 pad 到该批的最大长度（批内对齐，不跨批硬凑）"""
    seqs, lens = zip(*batch)
    maxlen = max(lens)
    padded = torch.stack([
        torch.nn.functional.pad(s, (0, 0, 0, maxlen - s.size(0))) for s in seqs
    ])
    return padded, torch.tensor(lens)

ds = TextDataset([12, 15, 130, 500, 9, 480, 33, 200])

# 关键：先按长度排序再切 batch —— 让同一批内的长度接近，减少 padding 浪费，
# 同时让每个 batch 的 maxlen 尽量落在少数几个值上，从而减少重编译
def bucket_by_length(dataset, batch_size):
    order = sorted(range(len(dataset)), key=lambda i: dataset.lengths[i])
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

from torch.utils.data import Sampler

class BucketSampler(Sampler):
    def __init__(self, lengths, batch_size):
        self.lengths, self.bs = lengths, batch_size

    def __iter__(self):
        # 每个 epoch 重新打乱桶的顺序，避免固定顺序影响收敛
        buckets = bucket_by_length(self, self.bs)
        perm = torch.randperm(len(buckets)).tolist()
        for i in perm:
            yield from buckets[i]

    def __len__(self):
        return len(self.lengths)

loader = DataLoader(
    ds, batch_size=4, sampler=BucketSampler(ds.lengths, 4), collate_fn=collate
)
for x, lens in loader:
    print("batch shape:", tuple(x.shape), "lengths:", lens.tolist())
```

排序本身会破坏随机性，所以 Sampler 里每个 epoch 重新打乱"桶的顺序"（而不是桶内样本），既保住了随机性，又保住了分桶收益。这是所有主流训练框架的标准做法。

---

## 四、围绕该领域展开：它在生态里的位置

动态形状不是一个孤立的开关，它是**编译/部署链条上多个环节的共同约束**。理解这一点，比记住几个 API 重要得多。

### 1. 与 Dynamo 的 guard 体系

Dynamo 的 guard 不只盯形状，还盯 dtype、device、stride、`requires_grad`，甚至 Python 层面的类型和闭包常量。所以"重编译"的原因经常不是形状，而是：

```python
if self.training:      # ← 训练/推理切换会触发重编译
    ...
if x.size(0) > 1:      # ← 把一个符号值当 Python 布尔值用了，产生"真值守卫"
    ...
```

第二行尤其隐蔽：`x.size(0)` 现在是符号 `s0`，Python 的 `if` 需要一个确定的布尔值，于是 Dynamo 不得不**假设 `s0 > 1` 成立**并给这个假设加 guard。下次来个 batch size = 1，guard 破，重编译。**在编译范围内的函数里，尽量别用张量形状做 Python 分支**——该用 `torch.where`、masked 操作或把分支挪到编译范围外。

### 2. 与 Inductor 的 kernel 生成

Inductor 把图变成 Triton kernel。动态形状下它做了大量 `size hint`（形状提示）工作：运行时第一次见到的具体形状被当作 hint，用于选 `BLOCK_SIZE`、决定是否向量化。因此：

- 同一个动态 kernel，**第一次**遇到形状 `(8, 512)` 和第一次遇到 `(512, 8)` 的性能可能差很多——它们被调成了不同的配置。
- 动态形状的 kernel 一般无法做极致的对齐假设，所以**别指望动态版本和静态版本一样快**。

这也解释了为什么"标记动态维度"要克制：把 4 个维度都标动态，等于放弃了几乎所有特化机会。

### 3. 与 CUDA Graph 的根本冲突

CUDA Graph（第 001 期）要求**固定的显存地址 + 固定的执行序列**。形状一变，中间激活的 buffer 大小就变，图就没法重放。两者的关系是"勉强共存"：

- **padding 到固定形状**：牺牲算力换图重放。小模型、小 batch 时常常划算。
- **按形状捕获多张图**：为几个常见形状各抓一张，运行时按形状查表。vLLM 就是这么做的。
- **用分段图 + 中间拼接**：复杂，收益有限，少见。

记住结论：**CUDA Graph 是"形状必须收敛"，动态形状是"形状可以发散"——它们天生对着干。实际系统里通常是"用分桶把形状收敛到有限几个值，再对每个值捕获 CUDA Graph"。**

### 4. 与显存分配器的相互作用

形状频繁变化意味着 Caching Allocator（第 004 期）要不断申请不同大小的 block。不同 maxlen 的 batch 会留下大小不一的缓存块，长期训练容易出现碎片和"显存明明够却 OOM"。

应对手段通常就是那几招：分桶（让尺寸种类收敛）、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（允许扩展已分配的段，缓解碎片），以及避免在训练循环里创建大量临时张量。

### 5. 变长注意力：这一领域的"终极形态"

NLP 里 padding 到批内最大长度意味着：若一批里有 1 个长度 2048 的样本，其余 31 个长度 128 的样本就要被 pad 到 2048——**算力浪费可以高达 90%**。

所以 Flash Attention（第 005 期）提供了 varlen 接口（`flash_attn_varlen_func`）：把所有序列首尾相接成一个一维张量，用 `cu_seqlens` 记录每条的边界，然后**只对有效 token 计算**。这样"变长"不再靠 padding 抹平，而是被算法本身消化掉了。

与之配套的数据结构是 **Nested Tensor（NJT）**：

```python
import torch

# 一个"参差不齐"的批次：长度为 3、5、2 的三条序列
nt = torch.nested.nested_tensor([
    torch.randn(3, 8),
    torch.randn(5, 8),
    torch.randn(2, 8),
])
print(nt.shape)          # 打印成 [sum(3+5+2)=10, j1=8] 这种"带符号"的形状
print(nt.is_nested)      # True
```

Nested Tensor 的思路是：**让形状本身携带"不确定"语义**，而不是靠外部约定。它是对动态形状最彻底的回答——只是生态支持（尤其在 `torch.compile`、算子覆盖度上）还在演进中，用之前建议先验证你需要的算子是否走得通。

### 6. 与导出/部署链路

一旦离开 PyTorch 进入部署，动态形状就变成一份**必须交接清楚的契约**：

- `torch.export` → 用 `dynamic_shapes` + `Dim(min, max)` 声明范围；
- ONNX → 用 `dynamic_axes` 标记哪一维是动态的；
- TensorRT → optimization profile 的 min/opt/max，选错 profile 会导致昂贵的运行时重规划；
- vLLM/TensorRT-LLM 等推理引擎 → 内部就是"按形状分桶 + 每桶优化"的组合拳。

**一句话：训练侧的动态形状是"减少重编译"，部署侧的动态形状是"声明清楚边界"。两者目标不同，但技术底座（符号形状 + 范围约束）是同一套。**

---

## 五、什么时候该用 / 不该用

**该用动态形状：**

- 变长序列的训练（NLP、语音、点云），尤其长度分布很宽；
- batch size 会变化的训练（多卡下最后一批、梯度累积步数不整除）；
- 推理服务：请求长度、并发数天然不定；
- 目标检测/分割：候选框数量、图像尺寸不定；
- 一份编译产物希望服务多种输入规格（"编译一次，多处复用"）。

**不该用、或该收敛成静态：**

- 形状真的固定：固定分辨率图像分类（224×224）、固定窗口时序模型。此时 `dynamic=False` 最稳最快；
- 编译范围内的关键路径：已经定位到热点的 kernel 就该给它静态形状，全力特化；
- 用 CUDA Graph 追求极限延迟时：形状必须收敛，动态形状只会带来麻烦；
- 动态维度过宽导致 guard 开销显著时：与其全动态，不如分桶成 3~5 个静态桶。

**一个实用的决策顺序：**

1. 形状固定 → 直接静态，别折腾。
2. 形状变化但种类有限（比如 ≤ 8 种）→ 分桶 + 静态编译，或让默认的 `dynamic=None` 自动处理。
3. 形状变化连续且范围宽（序列长度 10~4096）→ 先分桶降低种类数，再对少数桶使用 `mark_dynamic`。
4. 无论如何都要监控 `unique_graphs` 和重编译次数。

---

## 六、常见坑

**坑 1：训练最后一个小 batch 触发重编译**

`DataLoader(drop_last=False)` 时，最后一个 batch 往往比 `batch_size` 小，于是 `size mismatch at index 0` 出现，重编译一次。如果数据量每次都能被整除就看不出来，换数据集就炸。

应对：`drop_last=True`；或把最后一个 batch 丢掉/补一份；或干脆把 batch 维标记为动态。

```python
# 明确标记 batch 维动态，一劳永逸地容纳"最后那个不满的 batch"
x = torch.randn(8, 64)
torch.compiler.mark_dynamic(x, 0)
```

**坑 2：用形状做 Python 分支，制造出脆弱的真值守卫**

```python
def forward(self, x):
    if x.size(1) > 512:        # ❌ 符号值参与 Python 判断 -> 隐式 guard
        x = x[:, :512]
    return self.net(x)
```

Dynamo 会为 `s1 > 512` 固化一个假设，形状一越界就重编译（甚至直接图断裂）。改成张量级操作或把分支挪出编译范围：

```python
def forward(self, x):
    # ✅ 用张量操作表达，编译器可以保留为图中的条件，而不是 Python 的 guard
    return self.net(x[..., :512])
```

**坑 3：撞上 `cache_size_limit`，静默回退 eager**

默认 8 个编译产物。形状种类一多就超限，Dynamo **不报错**，直接回退到 eager。你的日志一切正常，但速度毫无提升。

应对：

```python
import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 64      # 先调大，争取时间做分桶
# 但更正确的做法是：分桶降低形状种类，而不是无限扩大缓存
```

调大缓存只是止血。缓存本身要占显存和编译时间，形状种类真的爆炸时，正确解是分桶。

**坑 4：`mark_dynamic` 标错了张量**

`mark_dynamic` 必须作用在**真正传入被编译函数的那个张量对象**上。常见的错误是标在了拷贝上：

```python
x = data.to(device)
torch.compiler.mark_dynamic(x, 0)

x2 = x.clone()                 # ❌ 动态标记不会继承给新对象
y = opt_model(x2)              # 编译看到的是未标记的 x2

y = opt_model(x)               # ✅ 标在真正入参的那个张量上
```

另外，标记模块内部产生的中间张量是无效的——`mark_dynamic` 只对输入生效。

---

## 七、一句话总结

**动态形状是把"张量的某一个维度"从编译期常量提升为运行期符号变量的机制——它用一部分 kernel 特化收益，换取"形状自由"与"减少重编译"；用好它的关键不是全开或全关，而是精确标记会变的那一维，并用分桶把无限形状空间收敛成有限几个桶。**

---

## 今日练习

给定一个会接收 batch size 在 1~128 之间任意变化的模型，要求：**最多只允许 2 个编译产物**，同时不允许静态地固定 batch 大小。

<details>
<summary>今日练习（点击查看参考答案）</summary>

**思路**：不允许固定 batch、又要限制编译产物数量 → 必须把 batch 维**有限地离散化**，再配合符号化维度。

```python
import torch
import torch._dynamo as dynamo

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# 把任意 batch size 归到有限几个桶，桶内 padding
BUCKETS = [1, 8, 32, 128]

def bucketize(n):
    for b in BUCKETS:
        if n <= b:
            return b
    return 128

# 只留 2 个编译产物：桶 32 和桶 128（小桶其实也能被大桶覆盖，这里示意手法）
dynamo.config.cache_size_limit = 2

model = Model().eval()
opt = torch.compile(model, dynamic=True)     # 全动态：形状由分桶控制

def run(x):
    n = x.size(0)
    target = bucketize(n)
    if target != n:                          # 桶内 padding
        pad = x.new_zeros(target - n, x.size(1))
        x = torch.cat([x, pad], dim=0)
    y = opt(x)
    return y[:n]                             # 裁掉 padding 的输出

with torch.no_grad():
    for n in [1, 3, 7, 8, 20, 32, 100, 128]:  # 形状五花八门
        assert run(torch.randn(n, 64)).shape[0] == n

stats = dynamo.utils.counters["stats"]
print("unique_graphs:", stats["unique_graphs"])   # 期望 <= 2
```

**为什么这样写能成立**：

1. `dynamic=True` 让形状不再是 guard 的一部分，Dynamo 不会因为 `n` 变化而重编译——这是"最多 2 个产物"的先决条件。
2. 分桶不是为了省重编译（这里已经由 `dynamic=True` 解决了），而是为了**限制实际参与 kernel 调优的形状种类**。Inductor 按 `size hint` 选 `BLOCK_SIZE`，桶数量一多，tuning 质量就被稀释；把桶限制在少数几个，让编译器把主要的形状见到、调好。
3. `y[:n]` 裁掉 padding 结果，保证语义正确（注意：这种方法适用于逐样本独立的算子；如果模型里有跨样本的 BatchNorm、跨序列的注意力交互，padding 会污染结果，必须改用 `cu_seqlens` 一类的变长方案）。
4. `cache_size_limit = 2` 在真实项目里**不要设这么小**——一旦真超出了，Dynamo 会静默退回 eager。这里只是为了逼出"必须靠分桶控制形状种类"这个结论。

**更优的解法**：如果模型对 padding 敏感（例如含跨样本归一化或变长注意力），就不要 padding，改为只对"会变的那一维"精确标记：

```python
x = torch.randn(8, 64)
torch.compiler.mark_dynamic(x, 0, min=1, max=128)   # 只放开 batch 维
```

配合 `dynamic=False`（其余维度保持静态），既限制住了动态范围，又保住了其他维度的特化机会——这通常是生产环境里性价比最高的写法。

</details>
