# PyTorch 每日一课 · 第 022 期

## 模型参数初始化：一场关于方差的精密平衡

> **日期**：2026-09-18
> **难度**：⭐⭐⭐
> **前置知识**：知道 `nn.Linear` 在做矩阵乘法即可，理解方差/标准差的基本含义；本篇独立可读
> **预计阅读时间**：16 分钟

---

## 一、这个领域解决什么问题

先看一段"完全正确"的代码：

```python
import torch
import torch.nn as nn

model = nn.Sequential(*[
    nn.Sequential(nn.Linear(256, 256), nn.ReLU())
    for _ in range(20)          # 只是堆 20 层而已
])

x = torch.randn(64, 256)
y = model(x)
print(y.std())                  # 猜猜是多少？
```

输出大约会是 **1e-8 量级**——输入信号标准差是 1，穿过 20 层之后变成了亿分之一。此时你想给这个网络的第 20 层接一个分类头，它的输入是一片"几乎全零"的噪声，梯度自然也接近零。训练不会报错，只是**永远学不动**。

更极端的情况：

```python
model = nn.Sequential(*[
    nn.Sequential(nn.Linear(4096, 4096), nn.ReLU())
    for _ in range(10)
])
print(model(torch.randn(8, 4096)).std())    # inf 或 nan
```

这一次是反过来：信号被放大到溢出，反向传播时梯度变成 `inf` 或 `nan`。

**这两个 bug 都不在模型的"结构"里，而在参数被赋予的初始值里。** 这就是参数初始化这个领域要解决的问题。

它要同时满足三件互相拉扯的事：

| 目标 | 不满足时会怎样 |
|---|---|
| **数值稳定** | 前向/反向出现 `inf`/`nan`，或者小到浮点下溢成 0 |
| **信号保真** | 前向激活、反向梯度的尺度随层数指数衰减或爆炸 |
| **打破对称性** | 同层所有神经元学出完全一样的东西（全零或全相同初始化） |

**为什么这个问题至今没过时？** 因为深度学习是非凸优化——初始化决定了你从损失曲面的哪个位置出发。大模型时代，网络从 20 层变成 100+ 层、从几百维变成上万维，**指数衰减这件事被放大了**，同时残差连接、归一化层、低精度训练的加入，又让"一个标准初始化"不再够用。

时间线上有几个关键节点：

- **1998（LeCun）**：`1/sqrt(fan_in)` 缩放，第一次把"尺度"当作显式设计对象；
- **2010（Glorot & Bengio）**：Xavier 初始化，提出"保持前向/反向方差"的统一框架；
- **2015（He et al.）**：针对 ReLU 修正，成为 CNN 时代的事实标准；
- **2019-2022（Fixup / ReZero / muP）**：面向残差网络和"超参迁移"的第二代理论。

---

## 二、核心思想：让方差穿过每一层时不增不减

### 类比：一盏接一盏的灯

想象一列串联的灯，每盏灯有一个"增益"旋钮。信号从第一盏传到最后一盏：

- 每盏增益 **0.9** → 传 100 盏后剩 0.9¹⁰⁰ ≈ **0.0000266**，灯全灭了；
- 每盏增益 **1.1** → 传 100 盏后剩 1.1¹⁰⁰ ≈ **13780**，全部烧爆；
- 每盏增益 **1.0** → 传多远都是原来的亮度。

深度网络就是这列灯，**每层的"增益"由该层权重的方差决定**。初始化的全部追求，就是把这个增益调到 1 附近。

### 推导：临界点在哪里

考虑一层 `z = Wx + b`，其中 `x` 是输入向量（维度 fan_in），`w` 是权重。假设 `w` 与 `x` 独立、都近似零均值：

```
Var(z_j) = Σ_i Var(w_ji · x_i)
         = fan_in · Var(w) · Var(x)        ← 独立同分布求和的方差是可加的
```

要让 `Var(z) = Var(x)`（增益为 1），唯一的条件是：

```
Var(w) = 1 / fan_in          ← 前向传播的临界点
```

**反向传播给出了另一个约束。** 梯度回传时，同样是矩阵乘法，只是"扇入"方向反转了，所以要保持梯度方差必须：

```
Var(w) = 1 / fan_out
```

fan_in 和 fan_out 一般不相等，两个条件冲突。Xavier 的答案是**取折中**：

```
Var(w) = 2 / (fan_in + fan_out)
```

### 三个必须掌握的关键洞见

**洞见 1：分布形状不重要，方差才重要。**

只要 `Var(w)` 对了，用均匀分布还是正态分布，在二阶近似下没有区别。这就是为什么 `xavier_uniform_` 和 `xavier_normal_` 可以互换使用——它们只差一个形状。

**洞见 2：均匀分布的 `bound` 与 `std` 差一个 `sqrt(3)`。**

均匀分布 `U(-b, b)` 的方差是 `b²/3`，所以 `std = b/sqrt(3)`，反过来：

```
bound = sqrt(3) · std
```

**这是读懂 PyTorch 初始化源码的钥匙**——`torch.nn.init` 里几乎所有函数算的都是 `std`（或 `gain`），最后一步统一乘 `sqrt(3)` 转成 `bound`。看到 `sqrt(3)` 不要慌，它只是分布形状的换算。

**洞见 3（最重要）：误差是指数累积的。**

如果每层的增益是 `g`（`g = fan_in · Var(w) · c`，`c` 是激活函数的贡献系数），那么 L 层之后：

```
Var(h_L) / Var(h_0) = g^L
```

关键不是 `g` 比 1 大还是小，而是**它离 1 有多远，被 L 次方之后放大了多少**：

| 每层增益 g | 10 层 | 20 层 | 50 层 | 100 层 |
|---|---|---|---|---|
| 0.90 | 0.35 | 0.12 | 0.0052 | 2.7e-5 |
| 0.97 | 0.74 | 0.54 | 0.22 | 0.048 |
| **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| 1.03 | 1.34 | 1.81 | 4.38 | 19.2 |
| 1.10 | 2.59 | 6.73 | 117 | 13780 |

你会发现 `g = 0.97` 看起来"只差 3%"，100 层后也掉了 20 倍。**深度网络对每层增益的精度要求，是指数级的苛刻。** 这条指数律也是"残差连接为什么是里程碑"的答案之一：残差把乘法变成了加法（`h = h + f(h)`），方差是累加而非连乘，指数律被换成了线性律。

---

## 三、在 PyTorch 中怎么用

### 3.1 先看默认初始化到底做了什么

```python
import math
import torch
import torch.nn as nn

lin = nn.Linear(256, 256)              # 构造时会自动 reset_parameters()
print(lin.weight.std().item())         # 实测约 0.036
print(1 / math.sqrt(256))              # 0.0625
print(lin.weight.min(), lin.weight.max())   # 边界约 ±0.0625
```

边界正好是 `±1/sqrt(fan_in) = ±0.0625`。翻开 PyTorch 源码，`nn.Linear.reset_parameters` 是这样写的：

```python
def reset_parameters(self):
    # 注意这个 a=math.sqrt(5)——它不是什么玄学常数
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
```

`a=sqrt(5)` 的含义可以算出来。`kaiming_uniform_` 里 `gain = sqrt(2 / (1 + a²))`，代入 `a² = 5` 得 `gain = sqrt(1/3)`，于是：

```
bound = sqrt(3) · gain / sqrt(fan_in) = sqrt(3) · (1/sqrt(3)) / sqrt(fan_in) = 1/sqrt(fan_in)
```

也就是说，**PyTorch 的默认初始化等价于一个"简化版 Xavier uniform"**，它刻意选择了让 `fan_out` 消失、只保留 `fan_in` 的形式。

问题在于：它对应的是 **tanh/sigmoid 时代的假设**，而不是 ReLU。把三种尺度放在一起对比（假设 `fan_in = fan_out = n`）：

| 初始化 | `Var(w)` | 相对默认的方差倍数 |
|---|---|---|
| PyTorch `nn.Linear` 默认 | `1/(3n)` | 1× |
| Xavier（tanh） | `1/n` | 3× |
| Kaiming（ReLU） | `2/n` | **6×** |

**默认值比 ReLU 网络真正需要的尺度小了 6 倍。** 每层增益因此只有 `1/6 ≈ 0.167`（更精确地说，ReLU 会再砍掉一半方差，理论增益 `1/sqrt(6) ≈ 0.408`），层数一多信号就消失了。这不是 PyTorch 的 bug，而是为了兼容性保留的历史默认值——**对浅层网络无所谓，对深层网络必须自己接管。**

### 3.2 手写一套初始化：把公式翻译成代码

```python
import math
import torch
import torch.nn as nn

def init_weights(module, nonlinearity="relu"):
    """按层类型分别初始化。用 model.apply(init_weights) 施加到整个模型。"""
    if isinstance(module, nn.Linear):
        # Step 1: 算 gain —— 激活函数决定了方差要保留多少
        #   relu       : gain = sqrt(2)     （一半神经元被砍掉，方差减半，故补 2 倍）
        #   leaky_relu : 由 slope 决定
        #   tanh/sigmoid: gain = 1（近似线性区）
        gain_map = {"relu": math.sqrt(2.0), "tanh": 1.0, "linear": 1.0}
        gain = gain_map.get(nonlinearity, 1.0)

        # Step 2: 算目标标准差（fan_in 用上采样的方式统计，忽略了 fan_out）
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        std = gain / math.sqrt(fan_in)

        # Step 3: 换算成均匀分布的 bound —— 别漏了这个 sqrt(3)
        bound = math.sqrt(3.0) * std
        nn.init.uniform_(module.weight, -bound, bound)

        # Step 4: bias 置零（配合后面的归一化层时，bias 甚至可以直接不要）
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        # 归一化层是"单位变换"起步：缩放 1、偏移 0
        # 忘了这一步，BN 默认的 weight=1/bias=0 恰好没事，但自定义实现容易踩空
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        # embedding 的行是"查表"，方差按 1 处理即可
        nn.init.normal_(module.weight, mean=0.0, std=1.0)


model = nn.Sequential(
    nn.Linear(256, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10),
)
model.apply(init_weights)         # 关键：apply 会递归遍历所有子模块
```

`gain` 的含义值得单独记一下：**它衡量"激活函数会损失多少方差"**。ReLU 把负半轴压成 0，零均值输入下有一半神经元被砍，方差减半，所以需要 `gain=2` 来补回——这正是 Xavier 与 Kaiming 的唯一差别。

### 3.3 亲手验证：激活方差逐层怎么变

这段代码可以直接跑（纯 CPU）：

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)

def build(depth=20, width=128, init="default"):
    layers = []
    for _ in range(depth):
        lin = nn.Linear(width, width)
        if init == "xavier":
            nn.init.xavier_uniform_(lin.weight)
        elif init == "kaiming":
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        elif init == "tiny":
            nn.init.normal_(lin.weight, std=0.01)
        # init == "default" 时什么都不做，保留 PyTorch 默认
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU()]
    return nn.Sequential(*layers)

def layer_stds(model, x):
    """用 hook 记录每一层输出（ReLU 之后）的标准差。"""
    stats = []
    def hook(module, inp, out):
        stats.append(out.detach().std().item())
    handles = [m.register_forward_hook(hook)
               for m in model if isinstance(m, nn.ReLU)]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return stats

x = torch.randn(128, 128)          # 输入已经归一化到 std ≈ 1
for name in ["default", "xavier", "kaiming", "tiny"]:
    s = layer_stds(build(init=name), x)
    pick = [s[i] for i in (0, 1, 4, 9, 19)]     # 第 1/2/5/10/20 层
    # 逐层增益用"全层几何平均"来度量，比只看相邻两层稳健得多
    gain = (s[-1] / s[0]) ** (1 / (len(s) - 1))
    print(f"{name:8s} 逐层增益≈{gain:.3f}  " +
          "  ".join(f"L{i+1}={v:.2e}" for i, v in zip((0, 1, 4, 9, 19), pick)))
```

实测输出（20 层、宽 128 的 ReLU MLP，输入 std=1）：

```
default 逐层增益≈0.408  L1=3.60e-01  L2=1.42e-01  L5=1.00e-02  L10=1.04e-04  L20=1.44e-08
xavier  逐层增益≈0.720  L1=4.83e-01  L2=3.51e-01  L5=1.32e-01  L10=2.94e-02  L20=9.47e-04
kaiming 逐层增益≈0.982  L1=7.49e-01  L2=6.93e-01  L5=6.28e-01  L10=5.23e-01  L20=5.30e-01
tiny    逐层增益≈0.078  L1=7.83e-02  L2=5.87e-03  L5=2.41e-06  L10=6.10e-12  L20=6.34e-23
```

数据和理论严丝合缝（理论预测：default `1/sqrt(6) = 0.4082`、xavier `1/sqrt(2) = 0.7071`、kaiming `1.0`）：

- **default**：实测 0.408，与理论 0.4082 吻合到小数点后三位 —— 20 层后信号只剩 1e-8，**这就是开头那段代码失败的完整解释**；
- **xavier**：实测 0.720，比默认好，但 20 层后依然掉了三个数量级 —— **因为它没有为 ReLU 补偿那被砍掉的一半方差**；
- **kaiming**：实测 0.982（有限宽度带来的轻微偏差），20 层后仍在 0.5 量级，**成功把信号送到底**；
- **tiny**（`std=0.01`）：增益只有 0.078，第 20 层是 `1e-23`，已经彻底下溢。

### 3.4 现代做法：Transformer 的初始化

Transformer 把上述理论又推进了一步。GPT-2 的原始实现里有这样一段：

```python
class Block(nn.Module):
    """一个 Transformer block，说明常见的三处特殊处理。"""
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model),
        )
        # 处理 ①：残差分支的“最后一个线性层”额外缩放 1/sqrt(2L)
        #          因为残差流是 2L 条支路相加，方差会累积，必须把每条支路缩小
        self.apply(lambda m: _scaled_init(m, n_layers))
        # 处理 ②：不变量——所有 LayerNorm 的权重必须是 1、偏置必须是 0
        self.apply(lambda m: _fix_norm(m))


def _scaled_init(module, n_layers):
    if isinstance(module, nn.Linear):
        std = 0.02                                    # GPT-2 的经验基准尺度
        if hasattr(module, "SPECIAL_SCALE"):          # 只给残差出口的 Linear 打这个标签
            std *= (2 * n_layers) ** -0.5
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _fix_norm(module):
    if isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

三处值得注意的地方：

1. **`std = 0.02` 不是从理论推出来的**，而是实验调出来的经验值，在这个尺度上配合 LayerNorm 工作得很好。理论给你的是"什么量级"，不是"确切数字"。
2. **残差分支的缩放 `1/sqrt(2L)`** 才是真正关键的一步，它是 GPT-2 论文里明确写出的改动，不加它深层 Transformer 会训练不稳定。
3. **LayerNorm 的 weight 必须重置为 1**：如果先做了全局 `normal_(std=0.02)` 再忘了修回来，等于把每一层的归一化输出都缩小 50 倍——这是一个非常隐蔽的 bug。

---

## 四、围绕该领域展开：初始化在生态中的位置

初始化不是一个孤立技巧，它是"尺度管理"这条主线上的一环。理解它周围的邻居，才能知道什么时候该动它。

### 4.1 与归一化层：事前预防 vs 事后修正

| | 初始化 | 归一化层（BN/LN） |
|---|---|---|
| 作用时点 | 训练开始前，一次性 | 每次前向，动态 |
| 手段 | 设定权重的尺度 | 强制激活的统计量回到 N(0,1) |
| 擅长 | 让信号"起步就在好位置" | 修正训练过程中跑偏的分布 |
| 弱点 | 训练中权重会自己漂移，init 管不着 | 掩盖问题、依赖 batch 统计、有训练/推理不一致 |

**有了 LayerNorm，初始化还重要吗？** 重要，但机理变了：

- 有 LN 时，进入每个子层的输入尺度已经被归一化，所以"逐层方差衰减"这条老问题被大幅缓解；
- 但 **LN 之后的路径没有保护**：残差流本身会累积（见 4.2），attention 的 logits 尺度决定 softmax 的熵（logits 太大 → softmax 饱和 → 梯度趋零），这两处仍然直接由初始化决定；
- 所以现代 Transformer 的做法是：**LN 保证基本稳定，init 负责残差流和注意力两支的精细尺度**。

### 4.2 与残差连接：把指数律改成线性律

残差连接 `h_{l+1} = h_l + f_l(h_l)` 对尺度的影响是根本性的：

```
无残差：Var(h_L) ∝ g^L        ← 乘法，指数
有残差：Var(h_L) ≈ Var(h_0) + Σ Var(f_l)   ← 加法，线性
```

这带来两个直接后果：

1. **深度变得可行**：ResNet 能上 100 层、Transformer 能上 100 层，不是靠"初始化得更好"，而是靠"不再依赖指数律"。
2. **但有新代价**：残差流是**方差累加器**，L 层之后累积了 2L 条支路的方差。如果每条支路的输出尺度和输入一样大，残差流的方差就会线性增长到 `2L` 倍。所以必须**给支路出口乘上 `1/sqrt(2L)`**——这就是 3.4 里那个 `** -0.5` 的来历。
3. **最激进的做法是零初始化支路**（Fixup、ReZero）：把 `f_l` 的最后一层权重初始化为 0，训练开始时整个网络等价于恒等映射，然后逐步"长出来"。这与 LoRA 的 `B = 0` 是同一个思想。

### 4.3 与优化器：Adam 会掩盖它，SGD 会暴露它

这是最容易被忽略的耦合。**Adam 是尺度不变的**：它用梯度自身的二阶矩做归一化，把权重乘 10 倍，更新步长几乎不变（`Δw = lr · m/sqrt(v)`，分子分母同时放大）。

后果有两面：

- **好的一面**：初始化偏差被 Adam 部分掩盖了，所以很多"用 Adam 就能训起来"的模型，换成 SGD 立刻崩——问题一直在，只是被藏起来了；
- **坏的一面**：尺度不变性意味着**不同层的有效学习率事实上不同**，超参（尤其是学习率）与宽度/深度的耦合关系变得难以迁移。你在这个宽度下调好的 lr，换成 4 倍宽度就未必好用。

**muP（Maximal Update Parametrization）** 就是为了解决这件事：它主张把初始化和学习率**同时**按宽度归一化（输出层用 `1/fan_in` 而不是 `1/sqrt(fan_in)`，每层 lr 按 `1/fan_in` 缩放），从而让"小模型上搜出来的超参可以直接搬到大模型"。如果你在做逐层学习率或者缩放律相关的工作，这是必修内容。

### 4.4 与混合精度训练：小权重会在 fp16 下溢

`Var(w) = 1/(3n)`，当 `n = 4096` 时标准差只有 0.009；再深一层，某些初始化会让权重落到 1e-5 量级。而 **fp16 的最小正规数是 6.1e-5**，比它更小就进入 subnormal 甚至直接归零。

于是出现一种诡异现象：**同一份代码，fp32 能训、AMP 下 loss 不下降**。原因就是小尺度权重在 fp16 里被截断，梯度更新量也被一起截断了。

应对手段：

- 优先用 **bf16** 而不是 fp16（bf16 的指数位和 fp32 一样宽，动态范围不缩水，代价是尾数精度低）；
- 用 `torch.amp.GradScaler` 的 loss scaling（它正是为 fp16 设计的）；
- **别把初始化设得过小**——`std=0.001` 这种在小网络上没事，一上 AMP 就出事。模型的权重尺度应当在 1e-2 ~ 1e-1 量级，才在 fp16 的安全区里。

### 4.5 与量化：初始化决定了权重的分布形状

后训练量化（PTQ）要先把一组权重映射到 int8 的 256 个整数格点上，映射区间由 `max(abs(w))` 决定：

```
scale = max|w| / 127
```

如果某个初始化（或训练过程）让权重分布长出几个远离主群的 **outlier**，`max|w|` 就会被这几个值拉大，主群权重全被压进少数几个格点——精度损失由此产生。这是 LLM 量化里"激活异常值"问题的同构版本，只是发生在权重侧。

所以"初始化是否友好"有一个可观测的判据：**看权重的分布是紧凑、近似高斯，还是有长尾**。紧凑分布 → 量化 scale 利用率高 → int8 掉点少。这也是为什么很多量化流程会在训练阶段加权重正则或裁剪。

### 4.6 与微调、LoRA：初始化思想的现代复用

- **从头训练**：初始化是头等大事；
- **微调预训练模型**：初始化几乎不重要（权重来自 checkpoint，你自己的新模块别忘了初始化即可）；
- **LoRA**：A 用 Kaiming 随机初始化、**B 用全零初始化**——这正是 4.2 里的"零初始化支路"：保证训练起点时 `BA = 0`，模型输出与预训练模型**完全等价**，增量从 0 平滑长出。如果 B 也随机初始化，你等于在预训练模型上叠加了一个随机扰动，起步就掉点。
- **`nn.Linear` 的额外头**（分类头、回归头）：**必须缩小初始化尺度**。常见坑是把一个 `std` 很大的头接在预训练特征后面，一开始就产生巨大 loss，反向传播把主干特征打崩。经验做法是分类头的 std 设成 `1e-2` 量级甚至更小。

### 4.7 与分布式训练：一致性和分片

两个分布式场景下的初始化的隐藏坑：

- **数据并行 / DDP**：初始权重必须在所有 rank 上**完全相同**（DDP 只在启动时广播一次参数，之后依靠梯度同步保持一致）。做法是在 `init_process_group` 之后、`DistributedDataParallel` 包装之前设 `torch.manual_seed(0)`，且保证各 rank 数据结构一致。如果各 rank 的 init 不同，第一次 allreduce 之后的平均，大家各占一半，模型就成"缝合怪"。
- **张量并行 / FSDP**：权重被切分到多卡，**`fan_in` 应该是全局扇入还是本地扇入？** 答案是：如果初始化后再做分片，就用全局 `fan_in` 算 std，然后按分片方式切；如果直接在各 rank 上构造分片，一定要用全局 `fan_in` 算 `std`，用本地形状去填充。用本地 `fan_in` 会让 std 偏大 `sqrt(world_size)` 倍——这是 TP 实现里最常见的静默错误（第 019 期讲过 TP 的切法，这里正好是它的一个应用细节）。

### 4.8 其他初始化家族：知道它们为什么存在

| 初始化 | 核心主张 | 适用 |
|---|---|---|
| `1/sqrt(fan_in)` | 只保前向、只保一阶矩 | 最朴素的起点 |
| Xavier | 前向+反向方差折中 | tanh/sigmoid 网络 |
| Kaiming | 为 ReLU 补回一半方差 | CNN、MLP（ReLU/GELU） |
| **正交初始化** | 让权重矩阵的**奇异值全为 1**，不只是方差对 | RNN、极深网络、想要 dynamical isometry 时 |
| LSUV | 先正交初始化，再用一个 batch 实测各层 std，逐层除以实测值 | 对尺度极敏感的网络，自动化 |
| fixup / ReZero | 不靠 init，靠零初始化支路 + 特殊的缩放结构 | 免归一化的深层残差网络 |
| muP | 按宽度归一化 init 与 lr | 超参迁移、缩放律研究 |

**正交初始化值得多看一眼**：前面的方差推导只用了二阶矩，它忽略了一个事实——**方差对不代表信号能原样传递**。真正强的条件是让层间雅可比矩阵的**整个奇异值谱**都接近 1（叫 dynamical isometry）。对一个方阵，正交矩阵恰好满足这个条件（所有奇异值 = 1）。这就是正交初始化在某些极深网络里明显优于 Kaiming 的原因——它管得比方差更细。

---

## 五、什么时候该用 / 不该用

| 场景 | 该怎么做 |
|---|---|
| 浅层网络（< 10 层）+ ReLU + Adam | 默认初始化够用，先别折腾 |
| 深层网络（≥ 20 层）无归一化 | **必须**接管：Kaiming/正交 + 残差缩放 |
| 深层 Transformer | 用 LN + 残差分支 `1/sqrt(2L)` 缩放 + 经验尺度 |
| 有 LayerNorm/BN 的前馈网络 | 默认初始化通常可行，但残差流的累积仍需留意 |
| 极深 RNN / 无残差的深网络 | 优先考虑**正交初始化** |
| 微调预训练模型 | 初始化不重要；只关心自己新增的模块（头、LoRA 的 A/B） |
| 用 SGD + fp16 训练 | 初始化最敏感，务必按理论算 std 并检查下溢 |
| 要做的只是"跑通一个 demo" | 别在这上面花时间，先跑起来再看 |

**一条实用判据**：先打印第一层和最后一层激活的标准差。如果从 1 掉到 1e-3 以下，或者涨到 1e3 以上，就别继续调学习率了，先修初始化——**这是一个 5 分钟就能做的诊断，而它能省下几小时的瞎调。**

---

## 六、常见坑

**坑 1：`apply` 顺手把所有层都改了，包括不该动的层**

`model.apply(fn)` 会递归遍历**所有**子模块，包括归一化层和自己的自定义模块。写 `isinstance` 分支时一定要把 `nn.LayerNorm`/`nn.BatchNorm` 单独列出并重置为 `weight=1, bias=0`，否则它们会被当成普通线性层一起随机化，网络直接瘫掉。更隐蔽的是**对 `nn.Embedding` 用 Kaiming**——embedding 的 fan_out 恒为 1，算出的 std 完全不对。

**坑 2：初始化写在 `load_state_dict` 之后，或者写在了加载预训练权重之前**

顺序必须是：`构造模型 → apply(init) → load_state_dict(预训练)`。如果你在 load 之后又 apply 了一次 init，预训练权重会被全部覆盖，而且**不报错**，你只会看到"这个模型怎么学不动"。

**坑 3：分布式下各 rank 的随机数不同步**

如果 init 之前各 rank 已经执行了不同的随机操作（比如不同的 dropout mask、不同的数据增强），即使 seed 相同，随机数状态也已经分叉，各 rank 的初始权重就会不同。安全做法：**先 `manual_seed`，再构造模型**，把模型构造放在最前面，不要有任何随机操作夹在中间。

**坑 4：自己写的 `nn.Parameter` 用了 `torch.empty`**

```python
class MyModule(nn.Module):
    def __init__(self, n):
        super().__init__()
        # ❌ empty 不初始化，内容是完全随机的垃圾值，可能是 1e30 也可能是 nan
        self.w = nn.Parameter(torch.empty(n, n))
        # ✅ 永远显式初始化
        # self.w = nn.Parameter(torch.randn(n, n) * (1.0 / math.sqrt(n)))
```

`nn.Linear`/`nn.Conv2d` 这类内置层会自动 `reset_parameters()`，但**你自己声明的 `Parameter` 不会**。这个 bug 的症状是"随机地 nan"，而且换个 seed 有时就好了，极难排查。

**坑 5：把初始化当"一次性任务"，训到一半才发现尺度漂移**

初始化只管训练开始。权重范数在训练中会自己变化（尤其 Adam + weight decay 会用一种复杂的动态把尺度推向某个平衡点）。如果训练中期出现 loss 尖峰、激活爆炸，问题可能不在 init，而在 lr、eps、weight decay 的组合。诊断顺序应该是：**先看 init 是否在合理量级 → 再看前若干步的激活/梯度范数 → 最后才去怀疑 lr 调度**。

**坑 6：换了激活函数却没换初始化**

GELU、SiLU(Swish)、GLU 的方差贡献系数都和 ReLU 不同（GLU 会乘性交互，更接近"方差再减半"）。把 ReLU 换成 GELU 一般问题不大（两者足够接近），但换成 `GLU`/`GEGLU`，或者网络是"深而窄"的，就值得实测一遍激活方差——**别用理论算，直接用 3.3 的 hook 打印出来看，这是最可靠的检查手段。**

---

## 七、一句话总结

**初始化就是给每层选一个"增益"，目标让信号穿过几百层后还是原来的亮度；理论给你量级（`1/sqrt(fan_in)` 及其 ReLU/残差修正），实验给你确切数字。**

---

## 八、今日练习

<details><summary>今日练习</summary>

**练习 1：诊断一个"学不动"的模型**

下面这个模型训练时 loss 几乎不下降。请指出至少两个初始化层面的问题，并给出修复代码。

```python
import math
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, d=512, depth=24):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)               # 只在最后做一次归一化
        self.head = nn.Linear(d, 10)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)   # "看起来很温和"
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        for lin in self.layers:
            x = torch.relu(lin(x))
        return self.head(self.norm(x))
```

**参考答案：**

问题有四处：

1. **`U(-0.1, 0.1)` 的 std 是 `0.1/sqrt(3) ≈ 0.0577`，它和 fan_in 无关。**当 `d = 512` 时，理论需要的 std 是 `sqrt(2/512) ≈ 0.0625`——看着很接近，纯属巧合。把 `d` 改成 2048，理论值变成 0.031，而代码里还是 0.0577，就错了。**初始化必须写成 `d` 的函数，不能是常数。**
2. **depth=24 的无残差纯 MLP**，方差按 `(fan_in · Var(w) · 0.5)^L` 指数衰减。`fan_in·Var(w)·0.5 = 512 × 0.00333 × 0.5 ≈ 0.853`，24 层后只剩 `0.853^24 ≈ 0.021`——掉了近 50 倍。
3. **bias 也用了 `U(-0.1, 0.1)`**，非零偏置把激活推到 ReLU 的一侧，加剧方差失衡。bias 应当置零。
4. **只在最后归一化**：LN 放在 head 之前，整条主干（24 层）完全没有保护。应该改成 Pre-LN 结构（每个子层前归一化 + 残差）。

修复版本：

```python
class Net(nn.Module):
    def __init__(self, d=512, depth=24):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.head = nn.Linear(d, 10)

        # ① 按 fan_in 算 std，而不是拍一个常数
        std = math.sqrt(2.0 / d)                       # Kaiming 尺度（ReLU）
        bound = math.sqrt(3.0) * std
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)              # ② bias 必须置零
            elif isinstance(m, nn.LayerNorm):           # ③ LN 保持单位变换
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # ④ 分类头单独缩小尺度，避免起步 loss 过大
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        for norm, lin in zip(self.norms, self.layers):
            # Pre-LN + 残差：把"乘法"换成"加法"，指数衰减的根就此拔掉
            x = x + torch.relu(lin(norm(x)))
        return self.head(x)
```

修复后可以用第 3.3 节的 hook 代码再测一次：逐层增益应该接近 1.0，第 24 层的激活 std 应该仍在 0.5 以上。

---

**练习 2：把"缩放律"用上**

假设你有一个 40 层、`d_model=1024` 的 Transformer，采用 Pre-LN + 残差结构。请回答：

(a) 残差分支出口（每个子层的最后一个 `nn.Linear`）应该乘什么缩放因子？
(b) 如果你把层数从 40 改成 80，这个因子怎么变？变大还是变小，为什么？

**参考答案：**

(a) 乘 `1/sqrt(2L)`，其中 `L = 40`，即 `1/sqrt(80) ≈ 0.112`。实现方式是把这个因子折进该层权重的 std：`std = base_std * (2 * L) ** -0.5`。

(b) **变小**（`1/sqrt(160) ≈ 0.079`）。原因是残差流是方差累加器：`Var(h_L) ≈ Var(h_0) + Σ_{l=1}^{2L} Var(f_l)`。支路条数是 `2L` 条（每层有 attention 和 MLP 两支），要让总和保持在 `O(1)`，每条支路的方差就得是 `O(1/L)`，尺度因子自然是 `O(1/sqrt(L))`。

这正是"深度"和"初始化"在残差网络里耦合的体现：**改深度不只是改层数，还要同步改初始化尺度**。GPT-2 论文里这一项是明确的架构改动，不是可选优化。

---

**练习 3（动手）：验证均匀分布与正态分布等价**

用 3.3 的代码框架，把 `kaiming_normal_` 换成"方差相同但是均匀分布"的手写初始化（提示：`bound = sqrt(3) · sqrt(2/fan_in)`），跑两遍比较逐层 std 曲线。

**参考答案：**

```python
std = math.sqrt(2.0 / width)
bound = math.sqrt(3.0) * std
nn.init.uniform_(lin.weight, -bound, bound)
```

两条曲线应当基本重合（相对差异在有限宽度和单次采样的噪声范围内，通常只有几个百分点）。结论：**只要方差对，分布形状无所谓**。这个结论只在二阶近似下成立——如果你用正交初始化去追奇异值谱，那就不是"方差对了就行"，而是更强的一阶近似要求。

</details>
