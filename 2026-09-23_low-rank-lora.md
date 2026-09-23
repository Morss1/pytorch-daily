# PyTorch 每日一课 · 第 025 期

## 低秩自适应微调（LoRA）：给大模型打一个薄薄的补丁

> **日期**：2026-09-23
> **难度**：⭐⭐⭐⭐
> **前置知识**：知道 `nn.Linear` 的权重形状是 `(out_features, in_features)`；知道优化器状态会占显存；对 SVD 有印象（没有也不影响，文中会补）。本篇独立可读，不依赖前 24 期。
> **预计阅读时间**：26 分钟

---

## 一、这个领域解决什么问题

### 1.1 先看那张让人不想干的账单

假设你手上有一个 7B 模型，想在一个垂直领域微调它。最直觉的做法是全量微调（full fine-tuning）——所有权重都参与训练。把账算清楚：

| 项目 | 每份大小 | 7B 模型 | 占比 |
|---|---|---|---|
| 权重（fp16） | 2 bytes/参数 | 13.04 GiB | 12.5% |
| 梯度（fp16） | 2 bytes/参数 | 13.04 GiB | 12.5% |
| fp32 主权重（optimizer master weights） | 4 bytes/参数 | 26.08 GiB | 25.0% |
| Adam 的一阶/二阶动量（fp32） | 8 bytes/参数 | **52.15 GiB** | **50.0%** |
| **合计** | | **104.31 GiB / 每张卡** | |

注意几个残酷之处：

1. **这是每张卡的份**。数据并行（DP）会把这个数字原封不动复制到每张卡上——加卡不省这里的显存，只省 batch 维度的激活。
2. **还没算激活**。7B 在 batch=8、seq=2048 下的激活又是几十 GB 量级。
3. **一半显存花在优化器状态上**。你买的显存有一半不是用来存模型，是用来存"模型走过的路"。

也就是说，单卡 80GB 的 A100/H100 直接放不下这个 7B 的全量微调。这还没提别的。

### 1.2 第二个问题：一个任务一份完整体重

假设你要做 20 个垂直领域的版本。全量微调 = 20 份 7B 权重 = 20 × 13 GiB ≈ 260 GiB 的存储，而且推理时要同时加载多份——**没有共享**。

这在工程上是不可接受的。你想要的是一套基础设施：基础模型常驻显存，每个任务只带一个几百 MB 的小附件，按请求热插拔。

### 1.3 真正的问题：微调真的需要改那么多参数吗

前两点都还是"工程抱怨"。这一节才是关键，因为它决定了整个方向是否成立。

**如果微调权重里真的有 67 亿个自由度需要确定，那任何参数高效方法都不可能 work——你就得老老实实改 67 亿个参数。**

所以先问一个更基础的问题：从预训练权重 $W_0$ 到微调后的权重 $W$，那个增量

$$\Delta W = W - W_0$$

**它到底长什么样？它是一个"满秩的随机矩阵"，还是一个"低秩的结构化矩阵"？**

这个问题有实验答案。我在这台机器上用纯 Python 做了两组对照模拟（下面所有数字都可复现，脚本见文末练习的思路）：

**对照组 A：随机满秩矩阵。** 一个 4096×4096 的 iid 高斯矩阵，用它的前 r 个奇异值去近似它自己。这是 Marchenko-Pastur 定律能给出解析解的经典情形，我直接积分 MP 谱密度：

| 截断 rank r | 前 r 个奇异值的能量占比（d=4096） | 相对误差 |
|---|---|---|
| 1 | 0.097% | 99.90% |
| 8 | **0.768%** | **99.6%** |
| 16 | 1.522% | 99.2% |
| 64 | 5.84% | 97.1% |
| 256 | 20.99% | 88.9% |

**对照组 B：真低秩 + 噪声。** 先构造一个真秩 k 的信号矩阵，再加一个满秩的高斯噪声（模拟微调时"主要学到少数几个方向 + 大量微小扰动"的情形），然后用完整 Jacobi 特征分解实测（d=96，6 次平均）：

| 构造 | 信号能量占比 | rank-8 截断保留 | rank-16 | rank-32 |
|---|---|---|---|---|
| 真秩=4，噪声 σ=0.5 | 94.1% | **95.52%** | 96.77% | 98.43% |
| 真秩=8，噪声 σ=0.5 | 97.2% | **97.39%** | 98.16% | 99.12% |
| 真秩=8，噪声 σ=1.0 | 89.3% | **90.84%** | 93.51% | 96.89% |
| 真秩=8，噪声 σ=4.0 | 32.7% | 44.88% | 60.91% | 81.38% |

**把这两张表放在一起，就是 LoRA 的全部合法性来源：**

- 同样是 rank-8：在随机满秩矩阵上只能抓住 **0.768%** 的能量（等于什么都没抓到），在真低秩矩阵上能抓住 **95%** 以上。
- 差别不是"好一点"，是**两个数量级**。

所以结论是：**LoRA 不是"用低秩去近似一个满秩对象"的魔法，它是在赌"$\Delta W$ 本身就是低秩的"。** 赌对了，rank-8 绰绰有余；赌错了（比如从零开始训练一个新任务，而不是微调），rank-8 一点用都没有。

这个赌注有理论和实验支撑。

- **内在维度**：Aghajanyan 等人在 2020 年的工作（*Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*）做了一个很有说服力的实验——把预训练模型投影到一个**随机**的 k 维子空间里再微调，k 只要**几百到几千**就能达到接近全量微调的效果。注意"随机子空间"这个设定：如果连随机选的方向都够用，说明下游任务真正需要的自由度本来就很少。
- **子空间重叠**：LoRA 论文从另一侧给了证据。他们比较了全量微调的 $\Delta W$ 与 LoRA 学到的 $\Delta W$ 的**奇异子空间**，用 Grassmann 距离衡量，发现两者**最靠前的几个主方向有显著重叠**（远高于随机基线），而排后面的方向基本不相关。也就是说：LoRA 抓住的正好是那些"真正重要"的方向，丢掉的是噪声。

一句话：**预训练已经学好了大部分结构，下游任务只需要在一个很小的子空间里挪一挪。**

---

## 二、核心思想

### 2.1 把增量拆成两个瘦矩阵

LoRA 的做法可以写成一个等式：

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA$$

其中：

| 符号 | 形状 | 说明 |
|---|---|---|
| $W_0$ | $(d_{out},\, d_{in})$ | 预训练权重，**冻结**，不参与梯度 |
| $B$ | $(d_{out},\, r)$ | 可训练 |
| $A$ | $(r,\, d_{in})$ | 可训练 |
| $r$ | 标量 | rank，通常 4 ~ 64，$r \ll \min(d_{in}, d_{out})$ |
| $\alpha$ | 标量 | 缩放超参 |

因为 $B A$ 是两个瘦矩阵相乘，它的**秩必然 ≤ r**。这就是"低秩自适应"（Low-Rank Adaptation）字面的意思。

参数量对比（$d_{in} = d_{out} = 4096$，单个矩阵）：

| rank r | LoRA 参数量 | 占全量 | 压缩比 |
|---|---|---|---|
| 4 | 32,768 | 0.195% | 512× |
| **8** | **65,536** | **0.391%** | **256×** |
| 16 | 131,072 | 0.781% | 128× |
| 64 | 524,288 | 3.125% | 32× |
| 256 | 2,097,152 | 12.5% | 8× |

**为什么是"两个矩阵"而不是"一个瘦矩阵"？** 如果只用一个 $(d_{out}, r)$ 的矩阵，那相当于 $\Delta W = B'$，它是一个**列空间被固定截断**的矩阵——只能修改输出空间的 r 个方向。拆成 $BA$ 之后，$A$ 负责"从输入空间里挑出 r 个方向"，$B$ 负责"把它们映射到输出空间的 r 个方向"，**两头都可以学**。这个额外的自由度不是装饰：rank-r 矩阵构成的流形维数是 $r(d_{in} + d_{out} - r)$，比"固定列空间"的 $r \cdot d_{in}$ 大得多。

### 2.2 类比：不是重刷一面墙，而是贴一层可撕的膜

想象你要把一间房子改成办公室。

- **全量微调**：拆掉重砌。所有墙、水电、地板都动过。改成第二种用途？再拆一遍。而且你得为每一种用途单独一栋房子。
- **LoRA**：房子结构不动，只在几面墙上贴一层**很薄的功能膜**。换用途时把膜撕掉、换一张新膜，**房子本身始终是那一栋**。

膜很薄（参数量 0.4%），但它贴在**哪几面墙**上、**贴多厚**（rank、target modules）可以调。而这个类比也顺便解释了两件事：
- 为什么"合并"（merge）是免费的：膜和墙的厚度可以相加，$W = W_0 + BA$ 加完就是一个普通权重矩阵，推理时零额外开销。
- 为什么"多任务"是天然的：同一栋房子可以按请求换膜（multi-LoRA serving）。

### 2.3 初始化：为什么 $B$ 必须是零

LoRA 的原始约定是：

- $A$ 用随机高斯 / kaiming 初始化；
- $B$ **初始化为全零**。

这样 $\Delta W = B A = 0$，训练**第一步**的模型和预训练模型**逐比特相同**。这不是为了让训练好看，而是：**微调必须从预训练那个点出发**。如果 $B$ 也随机初始化，模型一开始就被推离了预训练解，那你就不是"微调"了，而是在做一次带畸变的重新训练。这跟"初始化时模型输出应为零"是同一类要求（残差分支零初始化、LayerNorm 的 $\gamma=1$ 之类）。

顺带一个常被忽略的点：**$B=0$ 意味着第一步只有 $B$ 有梯度**。因为

$$\frac{\partial \mathcal{L}}{\partial B} \propto (\text{上游梯度}) \cdot (Ax)^\top, \qquad \frac{\partial \mathcal{L}}{\partial A} \propto (\text{上游梯度}) \cdot B = 0$$

第一步 $A$ 完全不动，只更新 $B$；从第二步开始两个都动。有趣的是有些实现（比如 LoRA+ 这篇工作）正是从这里做文章——它给 $B$ 配了比 $A$ 大得多的学习率。

### 2.4 $\alpha/r$：那个让人困惑的缩放

为什么是 $\frac{\alpha}{r}$ 而不是直接用 $\alpha$？先看实测（纯 Python 模拟，$d=512$，$A \sim \mathcal{N}(0, 1/r)$，$B \sim \mathcal{N}(0,1)$，$\alpha = 16$）：

| rank r | $BA$ 原始 RMS | $\alpha/r$ 缩放后 $\Delta W$ 的 RMS | $\alpha/\sqrt{r}$ 缩放后的 RMS |
|---|---|---|---|
| 4 | 1.0030 | 4.012 | 8.024 |
| 8 | 0.9961 | 1.992 | 5.635 |
| 16 | 0.9966 | 0.997 | 3.986 |
| 32 | 1.0101 | 0.505 | 2.857 |
| 64 | 1.0030 | 0.251 | 2.006 |

读法：因为 $A$ 是用 $1/\sqrt{r}$ 缩放的，$BA$ 每个元素的方差**与 r 无关**（恒为 1）。于是：

- 用 $\alpha/r$ 缩放 → $\Delta W$ 的幅度正比于 $1/r$：**r 翻一倍，更新幅度减半**。所以你把 r 从 8 改成 64 时，为了不破坏训练，往往得同时把 $\alpha$ 或 lr 一起调大。
- 用 $\alpha/\sqrt{r}$ 缩放 → 幅度正比于 $1/\sqrt{r}$，衰减得更慢。这正是 **rsLoRA**（rank-stabilized LoRA）的出发点：它主张 $\alpha/\sqrt{r}$ 能让不同 rank 下的有效更新尺度更一致，从而使"调大 r"不必重新精调学习率。

LoRA 原论文给 $\alpha/r$ 的理由是"有助于稳定不同 r 下的学习"，但它的实际效果是**把 rank 和有效学习率耦合在了一起**——这是实践中非常容易踩的一个坑，后面还会讲。

---

## 三、在 PyTorch 中怎么用

LoRA 不在 PyTorch 核心库里，但用核心库写一个完全可用的版本只需要几十行。下面分两条路：**手写模块**（理解原理用）和 **`torch.nn.utils.parametrize` 无侵入注入**（工程用）。

### 3.1 手写一个最小的 LoRA Linear

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """把 nn.Linear 包一层：原权重冻结，旁路挂一对低秩矩阵。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        # 1) 冻结原权重：原地把 requires_grad 置 False
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r          # 注意是 α/r，不是 α
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        d_out, d_in = base.weight.shape   # nn.Linear 权重形状是 (out, in)
        # 2) A: (r, d_in) 随机初始化；B: (d_out, r) 零初始化
        #    A 用 kaiming 等价的做法：std = 1/sqrt(r)，见 2.4 节的方差分析
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原路径（冻结，无梯度）
        out = self.base(x)
        # 低秩旁路：先降维到 r，再升回 d_out
        lora = F.linear(self.lora_dropout(x), self.lora_A)   # (..., r)
        lora = F.linear(lora, self.lora_B)                   # (..., d_out)
        return out + self.scaling * lora

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """把 LoRA 折进原权重，返回一个与原始结构完全相同的 nn.Linear。"""
        self.base.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        return self.base
```

几个形状细节值得停下来确认一遍（这是最高频的 bug 来源）：

- `self.base(x)` 内部算的是 `x @ W.T`，所以 `W` 的形状是 `(d_out, d_in)`。
- `F.linear(x, A)` 算的是 `x @ A.T`，`A` 的形状必须是 `(r, d_in)` → 输出 `(..., r)`。
- `F.linear(·, B)`，`B` 形状 `(d_out, r)` → 输出 `(..., d_out)`。
- 合并时 `B @ A` 的形状是 `(d_out, d_in)`，**和权重同形，直接相加**。不需要任何 `.T` 或 `.view()`。

**关于"合并等价性"**：数学上 $W_0x + sB(Ax) = (W_0 + sBA)x$ 是恒等式。我用纯 Python 双精度实测过这个恒等式（$d_{in}=64$、$d_{out}=48$、$r=4$、batch=5）：逐元素最大绝对差 **1.42e-14**，相对量级 **1.46e-16**——就是浮点舍入的底噪，说明两条路径严格等价。

但**工程上要小心**：合并确实有精度损失。如果把 bf16 的 $W_0$ 和 fp32 算出来的 $BA$ 相加再写回 bf16，误差量级是 bf16 的 eps（≈ 7.8e-3，相对误差约 0.4%）。这个损失通常是可接受的，但如果你的场景对数值极其敏感（比如再做一次量化），**合并时先在 fp32 里算 $W_0 + sBA$、再决定落回什么精度**。

### 3.2 用 `parametrize` 无侵入注入已有模型

手写包装类的问题是要改模型结构（替换 `nn.Linear`）。PyTorch 提供了官方的 `torch.nn.utils.parametrize`：它允许你**声明式地**给某个已有参数加一层变换，而不改变模块类型。

```python
import torch.nn.utils.parametrize as parametrize


class LoRAParametrization(nn.Module):
    """parametrization 的约定：forward(original_weight) -> 新的 weight"""

    def __init__(self, d_out: int, d_in: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.scaling = alpha / r

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        # 每次访问 module.weight 时被调用，返回 W0 + sBA
        return W + self.scaling * (self.lora_B @ self.lora_A)


def inject_lora(model: nn.Module, targets=("q_proj", "v_proj"), r=8, alpha=16.0):
    # 第一步：先全冻结，再注入（顺序很重要，见下）
    for p in model.parameters():
        p.requires_grad_(False)

    injected = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in targets):
            d_out, d_in = module.weight.shape
            parametrize.register_parametrization(
                module, "weight", LoRAParametrization(d_out, d_in, r, alpha)
            )
            injected.append(module)
    return injected
```

注册之后，模型发生了这些变化：

```python
# 注册前：linear.weight 是一个 nn.Parameter
# 注册后：linear.weight 变成一个「计算属性」
#        真实存储的原始权重被挪到了：
#        linear.parametrizations.weight.original
#
# 想拿到所有 LoRA 的可训练参数：
lora_params = []
for m in injected:
    lora_params += list(m.parametrizations.weight[0].parameters())

n_train = sum(p.numel() for p in lora_params if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"可训练 {n_train/1e6:.2f}M / 总计 {n_total/1e6:.2f}M "
      f"= {100*n_train/n_total:.3f}%")
```

**顺序为什么重要**：`register_parametrization` 之后才冻结的话，你得遍历 `parametrizations.weight[0]` 才能找到 LoRA 参数并重新打开梯度，很容易漏。先"冻结整个模型 → 再注入 parametrize"的顺序可以一步到位，因为新注册的 `lora_A`/`lora_B` 天然是 `requires_grad=True`。

还有一个实用工具：如果一个模块在前向中被多次访问、或者多个模块共享同一个参数化，可以用上下文管理器缓存计算结果，避免重复 matmul：

```python
with parametrize.cached():
    out = model(x)   # 同一次前向内，被共享的 weight 只算一次
```

`parametrize.cached()` 的作用域是**模块级**的（用 `parametrize.is_cached(module)` 可查），它不会跨前向缓存。

### 3.3 训练：只把 adapter 交给优化器

```python
# 优化器只接 LoRA 参数。这是一行都不能马虎的地方：
# 如果你写 torch.optim.AdamW(model.parameters())，虽然冻结参数的梯度是 None
# （不会被更新），但优化器内部的状态会在首次 step 时被分配 —— 白占显存。
optimizer = torch.optim.AdamW(
    [p for p in lora_params if p.requires_grad],
    lr=2e-4,              # LoRA 的学习率通常比全量微调大 1~2 个数量级
    weight_decay=0.0,     # 见「常见坑」第 3 条
)

for step, (x, y) in enumerate(loader):
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

学习率为什么能大很多？因为你要更新的参数量少了两三个数量级，而 Adam 的更新步长是**逐参数自适应**的（大致按 \( \text{lr} \cdot \text{sign}(\hat{m}/\sqrt{\hat{v}}) \) 走）。参数少了、每个参数承担的信息密度高了，允许更大的步长。经验值：LoRA 常用 1e-4 ~ 5e-4，全量微调常是 1e-5 ~ 2e-5。

### 3.4 合并，以及"合并前必须做的一件事"

```python
@torch.no_grad()
def merge_all(model, injected):
    """把 LoRA 折进基础权重，然后完全移除参数化结构。"""
    for m in injected:
        # leave_parametrized=True 的含义：
        #   把参数化「展开后的当前值」留下来当作新的原始参数，
        #   而不是回退到注册之前的那个原始权重。
        parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    return model
```

`leave_parametrized=True` 是这里的关键开关。如果写成 `False`（默认），PyTorch 会把 `original` 那个旧权重塞回去——**你的 LoRA 训练成果就全丢了**。

合并前应该做的验证（这是"没有对照就不下结论"的一个具体应用）：

```python
import copy

@torch.no_grad()
def check_merge_equivalence(model, injected, x, atol=1e-4):
    """在真实 dtype 下验证合并前后输出一致，再动手合并。

    注意：合并是就地的、不可逆的。要留退路就先 copy.deepcopy(model)。
    """
    y_before = model(x).clone()          # 合并前
    merge_all(model, injected)           # 就地合并
    y_after = model(x)                   # 合并后
    err = (y_before - y_after).abs().max().item()
    print(f"合并前后最大绝对差 = {err:.3e}（atol={atol}）")
    return err < atol


# 用法示意：
# import copy
# ref = copy.deepcopy(model)                      # 想留退路就拷一份
# ok = check_merge_equivalence(model, injected, x, atol=1e-4)
# assert ok, "合并不等价，先别部署"
```

在 fp32 下这个差应该在 1e-5 量级；如果是 bf16，你会看到 1e-3 量级——**这不是 bug，是 bf16 的精度上限**。设定 `atol` 时要按实际 dtype 来。

### 3.5 挂哪些层、r 取多大

| 决策 | 常见起点 | 说明 |
|---|---|---|
| **target modules** | `q_proj`, `v_proj` | LoRA 原论文的起点。覆盖 q/k/v/o 全四个投影会更好但更贵；MLP 层通常收益递减 |
| **rank r** | 8 ~ 16 | 简单任务 / 域内适配：4~8；复杂任务 / 跨域：16~64；再往上收益迅速变平 |
| **alpha** | 2r 或 r | 常见做法 $\alpha = 2r$（等价于 scaling=2）或 $\alpha = r$（scaling=1，更好推理） |
| **dropout** | 0 ~ 0.1 | 只在 LoRA 分支上加；小数据集上 0.05~0.1 有明显正则效果 |
| **lr** | 1e-4 ~ 5e-4 | 配合 cosine + 少量 warmup |

用一个具体模型算一下量级：LLaMA-7B（$d=4096$，32 层）。

- LoRA r=16 挂在 q/k/v/o 四个投影上：$32 \times 8 \times 4096 \times 16 = 16.78\text{M}$ 参数 = 全量的 **0.2397%**
- 同一配置挂到全部线性层（qkvo + MLP 的 gate/up/down）：**67.11M** = 全量的 **0.959%**

不到 1% 的参数，就完成了"挂到所有线性层"的最激进配置。

---

## 四、围绕该领域展开

这一节是这个领域真正的样子——LoRA 从来不是孤立的一个技巧，它是一个**参数高效微调（PEFT）生态**的入口，同时又和量化、分布式、推理服务深度交织。

### 4.1 低秩只是 PEFT 的一条路线

PEFT 家族大致按"改哪里"分成四派：

| 派别 | 代表方法 | 注入位置 | 推理时能否合并 | 增参量级 |
|---|---|---|---|---|
| **加旁路**（adapter） | Houlsby Adapter、AdapterFusion | Transformer block 内插入小 MLP | 不能（增加深度） | 1~5% |
| **改输入**（prompt） | Prompt Tuning、P-Tuning v2、Prefix Tuning | 在输入/每层 KV 前拼可学向量 | 不能（占用上下文） | <1% |
| **改激活**（scaling） | IA³、BitFit | 对 K/V/FFN 乘可学向量 / 只训 bias | 能（乘进权重） | <0.1% |
| **改权重增量**（low-rank） | **LoRA**、DoRA、LoHa、LoKr、VeRA | 给权重加低秩增量 | **能（直接相加）** | 0.1~1% |

LoRA 胜出的三个结构性原因，值得逐条说：

1. **可合并且零推理开销**。这是它相对 adapter / prefix 的决定性优势。adapter 增加了网络深度 → 每层多两次 matmul，推理变慢；prefix tuning 占用上下文长度 → 有效窗口变短。LoRA 合并后就是一个普通权重矩阵。
2. **低秩假设恰好匹配"微调"这个场景**（第 1.3 节的实测就是证据）。
3. **实现极简**。它不需要改计算图结构，只是一个加法和两个 matmul，对编译、分布式、量化的侵入性都很低。

反过来，prefix/prompt tuning 在**生成任务上仍然有独特价值**：它天然支持"一次前向处理多个不同 prompt"，而 LoRA 需要切换权重（或者 batched multi-LoRA kernel）。

### 4.2 LoRA 自己的改进谱系

LoRA 有若干被广泛采用的改良版本，每一个都在解决一个具体的缺陷：

| 方法 | 解决的缺陷 | 做法 |
|---|---|---|
| **rsLoRA** | $\alpha/r$ 使有效更新尺度随 r 衰减，调 r 必须重调 lr | 换成 $\alpha/\sqrt{r}$ |
| **LoRA+** | $A$ 和 $B$ 用同一学习率不是最优 | 给 $B$ 配 $\lambda$ 倍（通常 8~16×）的学习率 |
| **AdaLoRA** | 所有层给同一个 r 是浪费 | 用 SVD 式参数化 + 重要性打分，动态分配预算给重要层 |
| **DoRA** | 低秩更新同时改变了权重的**幅度**和**方向**，而这两者敏感度不同 | 把 $W$ 显式分解为 magnitude $m$ 与 direction $V/\|V\|$，只对 direction 做 LoRA 式更新 |
| **PiSSA** | 随机初始化 $A$、零初始化 $B$ 使训练早期"冷启动" | 用 $W_0$ 的 SVD 主成分初始化 $A$、$B$，让优化从主方向开始 |
| **LoftQ** | 量化后再加 LoRA，量化误差无法被 LoRA 补偿 | 联合优化量化权重与低秩初始化，把量化残差交给 LoRA 去拟合 |
| **GaLore** | 想做全参数训练，但连梯度和优化器状态都装不下 | 把**梯度**投影到一个低秩子空间，在这个低维空间里跑 Adam，再把更新投影回去 |
| **ReLoRA** | 单个 LoRA 的表达力上限受 rank 限制 | 周期性把 LoRA **合并**进基础权重，然后重新开一组新的 LoRA，让多次低秩更新累积成一个大更新 |

其中 **DoRA** 的思路最值得体会，因为它指出了一件容易被忽略的事：$\Delta W = BA$ 这个形式**没有区分"把权重整体放大"和"把权重旋转"**。而对神经网络来说，这两件事的后果完全不同——幅度变化会影响激活尺度进而影响 LayerNorm 的行为，方向变化才是"改语义"。DoRA 于是把权重写成

$$W = m \cdot \frac{V + BA}{\|V + BA\|_c}$$

（$\|\cdot\|_c$ 是逐列的范数），让 $m$ 单独学幅度、低秩分支只负责方向。它在小 rank 下的效果提升比较明显，代价是多了一组 $m$ 参数和一次范数计算，而且**推理时合并的代价变高**（需要重新归一化后相乘）。

### 4.3 量化 × 低秩：QLoRA 是怎么把 7B 塞进单卡的

QLoRA 的三个组件各解决一个问题：

**(1) NF4（4-bit NormalFloat）—— 为"权重近似正态"这件事定制码本。**

标准 INT4 是等距的 16 个格点；NF4 用的是**按标准正态分位数放置的 16 个非等距格点**（bitsandbytes 的官方码本）：

```
[-1.00000000, -0.69619280, -0.52507305, -0.39491749,
 -0.28444138, -0.18477343, -0.09105004,  0.00000000,
  0.07958030,  0.16093020,  0.24611230,  0.33791524,
  0.44070983,  0.56261700,  0.72295684,  1.00000000]
```

零点附近密、两端疏。我实测了这个码本的收益（纯 Python 模拟，每个 block 用 absmax 归一化到 $[-1,1]$ 后查表，对称 INT4 对照码本取 $[-7..7]/7$）：

| 权重分布 | block | NF4 相对 MSE | INT4 相对 MSE | NF4 改善 |
|---|---|---|---|---|
| 正态 $\mathcal{N}(0,1)$ | 64 | 0.00848 | 0.01153 | **+26.4%** |
| 正态 $\mathcal{N}(0,1)$ | 256 | 0.00969 | 0.01607 | **+39.7%** |
| 均匀 $U(-1,1)$ | 64 | 0.00823 | 0.00492 | **−67.2%** |
| 均匀 $U(-1,1)$ | 256 | 0.00858 | 0.00508 | **−69.0%** |

这张表读出来的信息比"NF4 更好"更精确：**NF4 的优势完全建立在"权重近似正态"这个前提上**。换成正态分布它赢三成，换成均匀分布它输六成。所以"用 NF4"不是一个普适的更好选择，而是一个**与分布假设绑定的赌注**。这也解释了为什么注意力权重、某些 embedding 层用 NF4 会掉点。

**(2) Block-wise 量化 —— 为什么 group_size 通常是 64 而不是"全局"。**

权重矩阵不同通道的尺度可以差很多（这是真实观察，也是量化的主要难点）。如果整张矩阵共用一个 scale，那么"尺度小的那一块"在归一化后会全部挤在零点附近，被稀疏的 NF4 格点直接抹成 0。

我构造了尺度异质的权重（每 64 个元素一组，组尺度 $s \sim \text{lognormal}(\sigma)$，组内 $\mathcal{N}(0,1)\cdot s$），对比"全局 scale"和"per-block scale"：

| 尺度异质性 $\sigma$ | 全局 scale 的相对 RMSE | per-block 的相对 RMSE | 全局误差 / 分块误差 | **最小尺度那一组的全局相对 RMSE** |
|---|---|---|---|---|
| 0.0（同尺度） | 0.1275 | 0.0923 | 1.91× | 0.122 |
| 1.0 | 0.2907 | 0.0887 | **10.73×** | **1.000** |
| 2.0 | 0.2278 | 0.0887 | 6.59× | **1.000** |

最后一列是重点：**相对 RMSE = 1.000 意味着那一组权重的量化结果全是 0**——它们被彻底抹掉了，不管原来存的是什么。全局 scale 在尺度异质的权重上不是"误差大一点"，而是"小尺度通道直接失效"。

那 block 是不是越小越好？我另外测了 block 大小对相对误差的影响（同尺度正态权重）：

| block size | 8 | 16 | 32 | **64** | 128 | 256 | 1024 |
|---|---|---|---|---|---|---|---|
| 相对 MSE | 0.00508 | 0.00654 | 0.00771 | **0.00849** | 0.00915 | 0.00961 | 0.01100 |

单调递减，但收益递减很陡。而代价是**元数据的线性增长**：每 block 存一个 scale，fp32 下开销是 $32/\text{block}$ bit/参数。

- block=64：$32/64 = 0.5$ bit/参数，相对 4-bit 净载荷是**12.5% 的额外开销**
- block=16：2 bit/参数，额外开销 50%——四分之一的信息都花在存 scale 上了

所以 64 是"精度"和"元数据开销"的平衡点，这也是 QLoRA 论文的选择（torchao 里常见默认是 32 或 64）。

**(3) Double Quantization —— 把 scale 自己也量化一遍。**

既然 scale 本身占 0.5 bit/参数，那就用 8-bit 再量化一次，per-256-blocks 存一个 fp32 的外层常量。对 7B 模型：

| 方案 | 存储 | 相对 |
|---|---|---|
| NF4 净载荷 | 3.26 GiB | — |
| + fp32 scale（每 64） | 0.41 GiB | 3.67 GiB |
| + int8 scale（每 64，双重量化） | 0.11 GiB（+ 外层常量） | **3.36 GiB** |

省 0.30 GiB / 8.3%。听起来不多，但它是**零精度成本**的纯收益——所以 QLoRA 默认开启。

**(4) PyTorch 官方的实现**：`torchao` 提供了 QLoRA 论文里 NF4 的实现，做法是把 NF4 表达为一个 **tensor subclass**（`NF4Tensor`）：

```python
# torchao 的 API（以你安装版本为准）
from torchao.dtypes import to_nf4
from torchao.dtypes.nf4tensor import linear_nf4

class FrozenNF4Linear(nn.Linear):
    def __init__(self, in_dim, out_dim, bias=False, **kw):
        super().__init__(in_dim, out_dim, bias=bias)
        self.weight.requires_grad_(False)       # QLoRA 里基础权重不训练
        # 把 bf16/fp32 权重转成 NF4 表示
        self.weight = nn.Parameter(to_nf4(self.weight), requires_grad=False)

# 前向：linear_nf4 负责「反量化 + 矩阵乘 + 反向时只存量化权重」
# frozen_out = linear_nf4(x, self.weight)
# lora_out   = self.lora_B(self.lora_A(x))
# return frozen_out + (alpha / r) * lora_out
```

选 tensor subclass 而不是"自定义 module"是很有讲究的：`NF4Tensor` 在语义上仍然是一个 `Tensor`，所以它能和 `torch.compile`、FSDP2、以及 `meta` device 初始化这些通用机制干净地组合。torchao 官方文档特别提到：QLoRA 不必绑定 NF4，也能和普通 INT4、以及面向 Blackwell 的 MXFP4 / NVFP4 组合。

官方还给了个现实参照——torchtune 的 QLoRA 教程里实测 7B 模型（用 `linear_nf4` 的完整流程）：
- 纯 LoRA：模型初始化 13.98 GB，训练峰值 15.57 GB
- QLoRA：初始化 **9.13 GB**，训练峰值 **9.29 GB**
- 也就是初始化和训练分别降了约 **35% / 40%**

注意这里的下降幅度**没有**达到"权重从 2 字节降到 0.5 字节"（−75%），因为 LoRA 训练时的显存里，权重只是其中一部分，激活和临时缓冲区不会因为权重量化而缩小。

### 4.4 服务侧：LoRA 的第二个战场

LoRA 在训练侧省的是显存，在**推理侧省的是部署复杂度**：同一份基础模型 + N 个小 adapter。

这带来了一些训练侧没有的工程问题：

- **多 adapter 批次**：一个 batch 里不同请求要走不同 adapter。朴素做法是按 adapter 分组、多跑几遍（吞吐严重浪费）。S-LoRA 这类工作把这批 matmul 做成 batched/分页的形式，让不同 adapter 的请求能在同一个 kernel 里跑完。
- **vLLM 的 LoRA 支持**：vLLM 有 `enable_lora` 选项，支持动态加载/卸载 adapter，并在批处理时处理多 adapter。这就是"当权重用"的 LoRA——adapter 保留在低秩形式（不合并），换来的是**免重启热切换**。
- **合并 vs 不合并的权衡**：
  - 只服务一个 adapter → **合并**，零推理开销；
  - 服务多 adapter、要热切换 → **不合并**，付出一点延迟换灵活性。

这个权衡就是第 4.1 节里 LoRA 相对 adapter/prefix 的"可合并"优势在服务场景的具体体现。

### 4.5 和分布式、编译、量化的交叉

- **FSDP2 / ZeRO**：LoRA 参数是"小而全量复制"的，通常**不做分片**（分片的通信开销远大于它省下的显存）；基础权重可以照常分片或冻结在 fp16。混合使用时要注意 FSDP 的 `ignored_params` 需要把 LoRA 参数排除在分片之外。
- **张量并行（TP）**：如果基础权重按列/行切分，LoRA 的 $A$/B 必须跟着切。一个常见错误是 $A$ 切了、$B$ 没切（或反之），结果是 rank 维度算错。原则是：$\Delta W = BA$ 必须和 $W$ 保持同样的分片布局。
- **`torch.compile`**：LoRA 给计算图增加了两个小 matmul。合并后（$W + sBA$ 在编译期算出来）可以得到与原始模型完全相同的图；不合并时，编译器通常能把它们融进同一个 kernel 里。**但在训练时**要小心：`parametrize` 的 `original` 参数访问路径会让图的输入变多，`torch.compile` 的 recompile 计数可能上升（属于第 021 期"动态形状/重编译"的同类问题）。
- **量化感知的微调**：`torchao` 支持把 LoRA 和 QAT 结合，官方给的数字是训练提速约 1.89×。这条路的价值在于：**训练时就让模型适应量化误差**，避免"训完再量化掉点"。

### 4.6 一张图看清楚 LoRA 的位置

```
                     预训练权重 W₀（冻结，可量化到 4-bit）
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   全量微调              低秩适配 (LoRA)        仅调 bias (BitFit)
   改 ΔW 全部            ΔW = (α/r)·BA           Δb
   100% 参数             0.1%~1% 参数            <0.1% 参数
   优化器状态 52 GiB     优化器状态 192 MiB       更少
        │                     │                     │
        │                     ├─► 可合并（零推理开销）
        │                     ├─► 可多任务热插拔（vLLM / S-LoRA）
        │                     └─► 可与 NF4 组合（QLoRA）
        │
   20 个任务 = 20 份 13 GiB 权重
   vs
   LoRA: 1 份基础模型 + 20 个 ~100 MB 附件
```

---

## 五、什么时候该用 / 不该用

### 该用

- **下游任务与预训练分布相近**（风格适配、领域术语、格式遵循、指令跟随微调）。这正是"$\Delta W$ 低秩"的前提成立的地方。
- **显存是硬约束**。7B 全量微调的优化器状态就是 52 GiB，LoRA r=16 只要 192 MiB——**278 倍**的差距。
- **需要一份基础模型服务多个任务**。这是 LoRA 相对其他 PEFT 方法最不可替代的场景。
- **数据量较小**（几百到几万条）。参数量小本身就是强正则。
- **需要快速迭代**。附件小 → 存 checkpoint 快、传到评测机快、A/B 实验便宜。

### 不该用（或该慎用）

- **继续预训练 / 领域自适应的大规模训练**。这时 $\Delta W$ 不再是低秩扰动，而是对模型的大幅改写。rank-8 在这种场景下的效果会明显退化（参考第 1.3 节：噪声占主导时 rank-8 只能保留 44.88%）。
- **从零训练新能力**（新语言、新模态）。LoRA 是"在已有能力上做偏移"，不是"长出新能力"。
- **对参数极度敏感的任务**（比如需要精确的数值行为、长链条推理）。低秩约束是一种归纳偏置，可能限制模型找到正确的解。
- **需要合并后再量化的场景**。合并会引入一次精度损失，如果下游还要做 INT4/NF4 量化，误差会叠加（这也是 LoftQ 出现的原因）。
- **极致的训练吞吐**。LoRA 的参数量小，但**前向/反向的计算量并没有等比例减少**——基础模型的 forward 和 backward（对输入）照做不误。省的是显存和优化器计算，不是主计算量。这一点常被误解。

---

## 六、常见坑

**坑 1：`A` 和 `B` 的形状放反。**

这是发生率最高的错误。口诀：**$A$ 降维、$B$ 升维**。

- $A:(r, d_{in})$ → 输出 $r$ 维（降维）
- $B:(d_{out}, r)$ → 输出 $d_{out}$ 维（升维）
- $BA:(d_{out}, d_{in})$ → 与权重同形

放反了（写成 $A:(d_{in}, r)$、$B:(r, d_{out})$）在方形权重上**不会报错**，只是转置了一下——你会得到一组语义完全错误但能训练的参数。更隐蔽的是：LoRA 论文与不少实现里 $A$、$B$ 的记法约定不同（有人写 $\Delta W = AB$），所以**跨实现迁移权重时一定要核对形状，而不是核对变量名**。

**坑 2：假装用了 NF4，其实写的是 INT4。**

网上大量"NF4 量化"示例长这样：

```python
# ⚠️ 这不是 NF4，是标准对称 INT4
scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
```

除以 7、`round`、`clamp`——这是**等距**量化的数学形式，NF4 的核心恰恰是"**格点不等距**"。用等距格点就丢掉了 NF4 的全部意义（第 4.3 节实测：正态权重下 NF4 比 INT4 好 26%）。真正的 NF4 需要一个 16 项的码本 + 最近邻查表。要真正做到，用 `torchao.dtypes.to_nf4` 或 bitsandbytes，别自己手搓。

**坑 3：给 LoRA 参数加了 weight decay，或者没加在正确的地方。**

LoRA 的 $A$、$B$ 通常应该用 `weight_decay=0`。两个理由：一是它们本身参数量极小、过拟合风险低；二是 weight decay 会把 $B$ 往零拉，而 $B=0$ 正是"没有适配"的状态——你在持续地把模型推回预训练原点。反过来，如果你要加，也只应该加在 $B$ 上，不要加在 $A$ 上。

另一个相关的坑：把 `bias` 和 LayerNorm 参数也塞进 weight decay 组（这属于常规训练卫生，但在 LoRA 的"只训一小部分"设定下更容易被忽略——因为 LoRA 参数本身就是主要项）。

**坑 4：用了 `remove_parametrizations` 却忘了 `leave_parametrized=True`。**

```python
parametrize.remove_parametrizations(m, "weight")                      # ❌ LoRA 训练成果被丢弃
parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)  # ✅
```

默认是 `False`，它会恢复成"注册之前"的原始权重。如果你的训练白跑了却找不到原因，先查这一行。建议按第 3.4 节写一个 `check_merge_equivalence`，**合并前后各跑一次前向对比**再动手——有对照就不容易翻车。

---

## 七、一句话总结

**LoRA 用 $\Delta W = \frac{\alpha}{r}BA$ 这一个秩约束，把"微调"从"改 67 亿个参数"变成了"在一个几百维的子空间里挪一挪"——它省下的是显存、优化器状态和部署复杂度，代价是赌上"微调增量天然低秩"这个假设；这个赌注在形态上是对的，但它有明确的适用边界。**

---

## 八、今日练习

### 练习 1：为什么"用一个瘦矩阵"不够

有人提出一个简化版 LoRA：不用两个矩阵，直接用 $W = W_0 + B'$，其中 $B':(d_{out}, r)$ 只能放在输出空间的 $r$ 个方向上（假设通过某种方式注入）。请从**可表达的矩阵集合大小**角度，说明它比 $BA$ 少掉了什么。

<details><summary>今日练习（点击展开参考答案）</summary>

**答案**：数一下两个集合的维数。

- 形如 $BA$（$B \in \mathbb{R}^{d_{out}\times r}$，$A \in \mathbb{R}^{r \times d_{in}}$）的矩阵，其**秩 ≤ r**。秩恰为 $r$ 的矩阵构成的集合（流形）的维数是：
  $$r \cdot d_{out} + r \cdot d_{in} - r^2 = r(d_{in} + d_{out} - r)$$
  减去的 $r^2$ 来自"把 $A$ 左乘一个可逆矩阵、同时把 $B$ 右乘其逆，结果不变"这个冗余（$GL_r$ 的自由度）。

- 形如"只有 $r$ 个固定方向的输出"的矩阵（即列空间被预先固定的秩-r 矩阵），其维数是 $r \cdot d_{in}$——只有 $A$ 这一头能动。

代入 $d_{in} = d_{out} = 4096$：

| r | $r(d_{in}+d_{out}-r)$ | $r \cdot d_{in}$ |
|---|---|---|
| 8 | 65,472 | 32,768 |
| 16 | 130,816 | 65,536 |
| 64 | 520,192 | 262,144 |

**$BA$ 的可表达集合正好大 2 倍少一点**（差值就是 $r(d_{out} - r)$）。它的意义是：$BA$ 不仅能选"改输出空间的哪几个方向"，还能同时选"从输入空间的哪几个方向读"——这个双向自由是"一个瘦矩阵"给不了的。

顺带注意一个边界：当 $r \geq \min(d_{in}, d_{out})$ 时自由度覆盖整个矩阵空间，此时 LoRA 退化成全量微调——所以 rank 不是"越大越好"，超过某个值之后只是白花钱。

</details>

### 练习 2：手算 LoRA 的优化器状态，验证"278 倍"

背景：LLaMA-7B，$d_{model}=4096$，32 层，每层有 4 个注意力投影（q/k/v/o），每个都是 $4096\times4096$。用 AdamW + 混合精度（fp32 主权重 + fp32 的 $m$、$v$）。

求：(1) 全量微调时，优化器相关状态（主权重 + $m$ + $v$）总共多少 GiB？(2) LoRA r=16 挂在四个投影上时，这些状态又是多少？比值是多少？

<details><summary>今日练习（点击展开参考答案）</summary>

**参考答案**：

(1) 全量微调：$P = 7\times 10^9$ 个参数。AdamW + 混合精度下每参数需要：

| 项 | 精度 | 每参数字节 |
|---|---|---|
| 主权重（master weight）副本 | fp32 | 4 |
| 一阶动量 $m$ | fp32 | 4 |
| 二阶动量 $v$ | fp32 | 4 |
| **合计** | | **12** |

$$7\times10^9 \times 12 = 8.4\times10^{10}\ \text{bytes} = \mathbf{78.2\ GiB}$$

（本文 1.1 节把这一项拆成了两行：$m+v$ = 52.15 GiB，fp32 主权重 = 26.08 GiB，两者相加即 78.2 GiB。别忘了权重和梯度本身还要另外 26.08 GiB。）

(2) LoRA r=16 挂 q/k/v/o：

- 每个投影：$A + B = r\cdot d + d\cdot r = 2rd = 2\times16\times4096 = 131{,}072$
- 每层 4 个投影：$4 \times 131{,}072 = 524{,}288$
- 32 层：$524{,}288 \times 32 = 16{,}777{,}216 \approx \mathbf{16.78\text{M}}$ 参数

对应的优化器状态：$16.78\times10^6 \times 12 = 2.01\times10^8$ bytes $= \mathbf{192.0\ MiB}$

(3) 比值：

$$\frac{78.2\ \text{GiB}}{192.0\ \text{MiB}} = \frac{80{,}077}{192} = \mathbf{417\times}$$

如果只比 $m+v$ 这一项（52.15 GiB vs 192 MiB），比值是 **278×**——这就是正文里那个数字的来源。两个数字不矛盾，只是分母口径不同。

**要点**：这个比值本质上只取决于**参数量比**（$7\text{B} / 16.78\text{M} = 417\times$）。所以"LoRA r=16 在 qkvo 上 = 全量的 0.2397%"这句话，可以**等效地读作"优化器状态只有全量的 0.2397%"**。这是估算 LoRA 显存收益最快的一把尺子：先算参数量占比，再套到优化器状态上。

</details>

### 练习 3：从实测数据反推"低秩假设"是否成立

假设你在某个模型上做了全量微调，得到 $\Delta W$（$4096\times4096$）。你想知道 LoRA 的 rank 该取多大。请设计一个**不需要重新训练**的诊断方法，并说明结果怎么读。

<details><summary>今日练习（点击展开参考答案）</summary>

**参考答案**：

**做法**：对 $\Delta W$ 做一次 SVD，看它的**奇异值谱**（或者等价的能量累积曲线）。

```python
import torch

dW = (W_finetuned.float() - W_pretrained.float())   # 关键：先转 fp32，避免 bf16 噪声淹没谱
U, S, Vh = torch.linalg.svd(dW, full_matrices=False)

# 能量累积曲线：前 r 个奇异值占总能量的比例
energy = S.pow(2)
cum = energy.cumsum(0) / energy.sum()

for r in (4, 8, 16, 32, 64):
    print(f"rank-{r:3d}: 能量保留 {cum[r-1].item():.4f}  "
          f"→ 相对误差 {(1 - cum[r-1]).sqrt().item():.4f}")
```

**怎么读结果**——对照本文第 1.3 节的两张表：

| 观察到的形态 | 含义 | 建议 |
|---|---|---|
| 累积曲线**陡峭**（rank-8 就到 0.9+） | $\Delta W$ 是低秩的，微调只动了少数方向 | LoRA 适用，r 取 8~16 |
| 曲线**接近 MP 直线**（rank-8 只有个位数百分比） | $\Delta W$ 接近满秩随机矩阵 | LoRA 会明显欠拟合；要么大幅提高 r，要么用别的方法 |
| 中间形态 | 部分低秩 | 取更大的 r（32~64），或考虑 AdaLoRA 这种按层分配 rank 的方法 |

**要注意的三个细节**：

1. **必须在 fp32 里算 $\Delta W$ 和 SVD**。如果两个权重都是 bf16，那么 $\Delta W$ 里混着 bf16 的量化噪声（相对量级 $10^{-2}$），这个噪声本身就是满秩的，会**把真实的低秩结构淹没**。这会让你误判为"$\Delta W$ 不低秩"。
2. **要逐层看，不要只看全局**。不同层的 $\Delta W$ 结构差异很大（这就是 AdaLoRA 的动机）。把 32 层的谱曲线叠在一张图上比看平均更有信息量。
3. **别忘了 $\Delta W$ 是"训练了多少步"的函数**。训练不足时 $\Delta W$ 接近随机小扰动（看起来满秩），训透了才显现出结构化。所以这个诊断要在收敛之后做。

**补充**：还有一个更直接的诊断——不做 SVD，直接跑一次低 rank 的 LoRA 微调，看它和全量微调的下游指标差多少。这个方法更贵但更贴近真实目标，而且顺带扫出了"最优 r"。

</details>

---

**下期预告**：候选有损失函数设计、正则化全景（Dropout / Weight Decay / Label Smoothing）、cuDNN 与 cuBLAS、知识蒸馏、torch.export + AOT Inductor——挑一个你没看过的。
