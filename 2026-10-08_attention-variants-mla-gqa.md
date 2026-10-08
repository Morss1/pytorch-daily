# PyTorch 每日一课 · 第 032 期

## Attention 变体谱系：MHA → MQA → GQA → MLA —— KV 侧的结构设计

| | |
|---|---|
| **日期** | 2026-10-08 |
| **难度** | ⭐⭐⭐⭐ |
| **前置知识** | attention 的基本形式、RoPE、KV cache 的概念、roofline 里「算术强度 / 平衡点」的用法 |
| **预计阅读** | 35 分钟 |
| **关联** | 第 005 期 Flash Attention（同一个注意力的**计算实现效率**）· 第 011 期 KV Cache 与 PagedAttention（缓存**怎么放**）· 第 028 期 连续批处理（同一个「B·L = 295」平衡点）· 第 029 期 前缀缓存（**命中率恒等于省下的 prefill FLOPs 比例**）· 第 030 期 PD 分离（跨机搬的就是 KV，体积由本期决定） |

> **本期要回答的问题**：一个 token 的 KV，到底需要用多少字节表示才算够？
>
> 这是**架构决策**，不是实现技巧（005 期），也不是缓存管理（011 期）。前面几期反复用到「7B / GQA-8 / bf16 → 128 KiB/token」这个常数，本期往回退一步，把「这个常数是谁定的、还能不能再压」讲透。

---

## 1 这个领域解决什么问题

### 1.1 decode 的账本上只有一项

第 027 期算过一个数：7B 模型 BF16 在 H800 量级的卡上，一次 decode 前向的理论下限是

$$t_{\text{mem}} = \frac{N \cdot \text{bytes}}{\text{BW}} = 4.179 \ \text{ms}$$

而同样的 FLOPs 如果按峰值算力跑，只要 0.014 ms —— **算力利用率 0.34%**。第 028 期把这件事总结成一句话：

> decode 阶段的时间几乎完全由「要读多少字节」决定，FLOPs 是免费的。

于是有两个字节数进入视野：

- **权重字节**：$N \cdot \text{bytes}$，与请求无关，是固定的地板；
- **KV cache 字节**：$\text{per-token bytes} \times S \times B$，**随上下文长度和并发数线性增长**。

前者加量化、加 MoE 就压得动；后者只有一条路 —— 让「每个 token 的 KV」本来就更小。这就是本期的话题。

### 1.2 和已经发过的几期怎么分工

| 期数 | 管的是 | 对象 |
|---|---|---|
| 005 Flash Attention | 同一个注意力的**计算实现效率** | tiling / online softmax / IO 复杂度 |
| 011 KV Cache / PagedAttention | 缓存**怎么放** | 分页、块表、显存碎片 |
| 029 前缀缓存 | 缓存**怎么复用** | 命中率 = 省下的 prefill FLOPs 比例 |
| 030 PD 分离 | 缓存**怎么搬** | 跨机传输、分层存储 |
| **032（本期）** | **缓存本身的表示** | **一个 token 的 KV 用多少字节** |

前四期都吃同一个账本。第 029 期用了一整天算「277,305 个去重 token = 33.9 GiB」，那个 33.9 GiB 是用 `128 KiB/token` 乘出来的 —— 而 128 KiB 这个数，来自 GQA 这件事本身。

### 1.3 KV 体积账本：用真张量量，不套公式

先别急着背公式。为每种机制真的建一个 module，跑一次 forward，直接数 cache 里躺着多少个元素。

```python
"""第 032 期 代码块 1：KV 体积账本 —— 用真张量量，不套公式。"""
import torch
import torch.nn as nn

torch.manual_seed(0)


def kv_elements(n_h, n_kv, d_h):
    """GroupedAttention 的 cache 里，每层每 token 存多少元素。"""
    return 2 * n_kv * d_h          # K 和 V 各 n_kv 个头、每头 d_h 维


class GroupedAttention(nn.Module):
    """n_kv == n_h → MHA；n_kv == 1 → MQA；中间 → GQA。"""

    def __init__(self, d_model, n_h, n_kv, d_h):
        super().__init__()
        self.n_h, self.n_kv, self.d_h = n_h, n_kv, d_h
        self.wq = nn.Linear(d_model, n_h * d_h, bias=False)
        self.wk = nn.Linear(d_model, n_kv * d_h, bias=False)
        self.wv = nn.Linear(d_model, n_kv * d_h, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        k = self.wk(x).view(B, S, self.n_kv, self.d_h)
        v = self.wv(x).view(B, S, self.n_kv, self.d_h)
        # 关键：cache 里只有 n_kv 个头的 K/V；
        # 算 attention 时的 repeat_interleave 是临时张量，不进 cache
        return [k, v]


class MLA(nn.Module):
    """DeepSeek 风格：cache 的是压缩 latent + 共享的解耦 RoPE key。"""

    def __init__(self, d_model, n_h, d_h, d_c, d_rope):
        super().__init__()
        self.n_h, self.d_h = n_h, d_h
        self.w_dkv = nn.Linear(d_model, d_c + d_rope, bias=False)
        self.w_uk = nn.Linear(d_c, n_h * d_h, bias=False)
        self.w_uv = nn.Linear(d_c, n_h * d_h, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        latent, k_rope = self.w_dkv(x).split([self.w_dkv.out_features - 64, 64], dim=-1)
        # 展开出的 k_nope / v 只在算 attention 时用，不进 cache
        _ = self.w_uk(latent), self.w_uv(latent)
        return [latent, k_rope]


def cached_elements(mod, x):
    with torch.no_grad():
        return sum(t.numel() for t in mod(x)) / x.shape[1]


d_model, n_h, d_h = 7168, 128, 128
x = torch.randn(1, 8, d_model)      # 只关心张量形状，数值随机无妨
BYTES = 2                            # bf16
L = 61

print("统一基准：61 层 / 128 个 Q 头 / d_h=128 / bf16")
print(f"{'机制':<10}{'每 token 每层':>14}{'元素数':>10}{'61 层':>14}{'相对 MHA':>11}")
rows = []
for name, n_kv in [("MHA", 128), ("GQA-8", 8), ("GQA-4", 4), ("MQA", 1)]:
    e = cached_elements(GroupedAttention(d_model, n_h, n_kv, d_h), x)
    rows.append((name, e))
e = cached_elements(MLA(d_model, n_h, d_h, 512, 64), x)
rows.append(("MLA(V3)", e))
base = rows[0][1]
for name, e in rows:
    print(f"{name:<10}{e * BYTES:>11.0f} B{e:>10.0f}{e * BYTES * L / 1024:>11.2f} KiB"
          f"{base / e:>10.2f}x")

print()
print("论文 Table 1 的解析式（元素数）")
print("  MHA       2*n_h*d_h      = 32768")
print("  GQA(n_g)  2*n_g*d_h      = 256*n_g")
print("  MQA       2*d_h          = 256")
print("  MLA       d_c + d_h^R    = 576")
print(f"  MLA 等效 GQA 组数 = 576 / 256 = {576 / 256:.2f} 组（论文原文的原话）")
print(f"  MLA 相对 MHA      = {32768 / 576:.2f}x   相对 GQA-8 = {2048 / 576:.2f}x")
```

```text
统一基准：61 层 / 128 个 Q 头 / d_h=128 / bf16
机制            每 token 每层       元素数          61 层     相对 MHA
MHA             65536 B     32768    3904.00 KiB      1.00x
GQA-8            4096 B      2048     244.00 KiB     16.00x
GQA-4            2048 B      1024     122.00 KiB     32.00x
MQA               512 B       256      30.50 KiB    128.00x
MLA(V3)          1152 B       576      68.62 KiB     56.89x

论文 Table 1 的解析式（元素数）
  MHA       2*n_h*d_h      = 32768
  GQA(n_g)  2*n_g*d_h      = 256*n_g
  MQA       2*d_h          = 256
  MLA       d_c + d_h^R    = 576
  MLA 等效 GQA 组数 = 576 / 256 = 2.25 组（论文原文的原话）
  MLA 相对 MHA      = 56.89x   相对 GQA-8 = 3.56x
```

三个直接结论：

1. **MHA → GQA-8 是 16 倍，MHA → MQA 是 128 倍**（$n_h = 128$）。压缩比就是头数比 $n_h / n_g$，没别的。
2. **MLA 是 56.89 倍**，但它的压缩比**不是整数比，也不是有理数比** —— 稍后细说。
3. MLA 的 576 个元素里，`latent` 占 512（88.9%），共享的 RoPE key 占 64（11.1%）。后者是**刚性开销**，$d_c$ 越小它占比越大。

### 1.4 真实模型落在哪

把上面那张账套到真实模型上（层数不同，所以不能直接横向比机制，只能比「同一块卡能装多少」）：

| 模型 | 机制 | 层数 | 每 token KV | 4K 上下文单序列 |
|---|---|---|---|---|
| Llama-2-7B | MHA-32 | 32 | 512 KiB | 2.00 GiB |
| Llama-3-70B | GQA-8 | 80 | 320 KiB | 1.25 GiB |
| Qwen2.5-72B | GQA-8 | 80 | 320 KiB | 1.25 GiB |
| DeepSeek-V2 | MLA | 60 | 67.50 KiB | 270 MiB |
| DeepSeek-V3 | MLA | 61 | **68.62 KiB** | **274.5 MiB** |

最后一行就是 030 期反复用到的那个数：$68.62 \ \text{KiB/token} = 61 \times (512+64) \times 2 \ \text{B}$。

同一个 52.02 GiB 的 KV 池（80GB 卡扣掉 7B 权重与激活）能装多少：

- **MHA-32（Llama-2-7B 口径）**：0.107M token ≈ 26 条 @4K
- **GQA-8（Llama-3-70B 口径）**：0.170M token ≈ 42 条 @4K
- **MLA（DeepSeek-V2 口径）**：0.808M token ≈ 197 条 @4K

同一个显存池，并发数差 7.6 倍。这就是为什么「KV 压缩」在服务侧是一等公民 —— 它不是省显存，它是**买并发**。

---

## 2 核心思想：三条不同的压缩轴

MHA 的 KV cache 之所以大，是因为它同时冗余了两件事。看清楚冗余在哪，压缩轴就自然出现了。

### 2.1 轴一：减 KV 头数（MQA → GQA）

MHA 里每个 query 头都有自己的一套 K/V。**但它们真的需要吗？**

如果所有 query 头共用同一套 K/V，就是 **MQA**（Shazeer, 2019）。这直接砍掉 $n_h$ 倍：KV cache 从 $2 n_h d_h$ 变成 $2 d_h$。

问题是质量。MQA 原始论文在 WMT14 EN-DE 上（6 层、211M 参数、$d_{\text{model}}=1024$、$h=8$、$d_k=d_v=128$）给的是：

| | 训练 µs/token | 增量解码 µs/token | 增量 beam-4 | PPL(dev) |
|---|---|---|---|---|
| MHA | 13.2 | 46 | 203 | 1.424 |
| MQA | 13.0 | **3.8** | **32** | 1.439 |

训练几乎不变（13.2 → 13.0），解码快 **12.1 倍**。但 PPL 从 1.424 涨到 1.439，Billion-Word 上从 29.9 涨到 30.2 —— 不大，但是真的变差了。

**GQA**（Ainslie et al., EMNLP 2023）就是把「1 套」和「$n_h$ 套」之间的空间填上：取 $n_g$ 组 K/V，$1 < n_g < n_h$。论文的原话是「a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads」，并且给出了关键工程结论：

> 用**原预训练算力的 5%** 把已有的 MHA checkpoint uptrain 成 MQA（以及 GQA），uptrained GQA 的质量接近 MHA，速度接近 MQA。

### 2.2 轴二：降维（MLA 的低秩瓶颈）

GQA 的思路是「少留几份」，**MLA 的思路是「每份都变小」**。

DeepSeek 的 **MLA**（Multi-head Latent Attention）不再分别投影出 K 和 V，而是先把 hidden state 压成一个 $d_c$ 维的 latent $\mathbf{c}^{KV}$，K 和 V 都从这个 latent 上投出来：

$$
\mathbf{c}_t^{KV} = W^{DKV}\mathbf{h}_t, \qquad
\mathbf{k}_t^C = W^{UK}\mathbf{c}_t^{KV}, \qquad
\mathbf{v}_t^C = W^{UV}\mathbf{c}_t^{KV}
$$

**推理时只缓存 $\mathbf{c}_t^{KV}$（以及后文要讲的解耦 RoPE key）**，不缓存展开后的 K/V。于是每层每 token 的元素数是

$$
\underbrace{d_c}_{\text{latent } 512} + \underbrace{d_h^R}_{\text{RoPE key } 64} = 576
$$

而不是 $2 n_h d_h = 32768$。

### 2.3 两条轴的本质差别：离散 vs 连续

这是整个谱系里最重要的一句话：

- **MQA / GQA 的压缩比是「选出来的」**，取值为 $n_h / n_g$，是一串**离散点**，而且 $n_g$ 必须整除 $n_h$。$n_h=128$ 时可选的压缩比就是 ${2, 4, 8, 16, 32, 64, 128}$ —— 中间什么都没有。
- **MLA 的压缩比是「学出来的」**，取值 $2 n_h d_h / (d_c + d_h^R)$，$d_c$ 取任意正整数都合法，是一条**连续曲线**。

DeepSeek-V2 论文的 Table 1 底下有一句常常被忽略的注释：

> For DeepSeek-V2, $d_c$ is set to $4d_h$ and $d_h^R$ is set to $d_h/2$. So, its KV cache is equal to **GQA with only 2.25 groups**, but its performance is stronger than MHA.

$576 / 256 = 2.25$ —— **一个 GQA 永远取不到的中间值**。这就是 MLA 的全部卖点：用 GQA-2 的钱，买到比 MHA-128 更强的质量。

### 2.4 那「低秩」这个前提成立吗？——不，是被逼出来的

一个自然的疑问：把 16384 维的 K 空间压到 512 维，凭什么没损失？

无结构的随机矩阵有这个问题的解析解（Marchenko-Pastur / 四分之一圆律）。设总维度 $d$、保留前 $r$ 个奇异值、$\rho = r/d$：

$$
T(\theta) = 1 - \frac{2\theta}{\pi} - \frac{\sin(2\theta)}{\pi}, \qquad
E(\theta) = 1 - \frac{2\theta}{\pi} + \frac{\sin(4\theta)}{2\pi}, \qquad x = 4\sin^2\theta
$$

解 $T(\theta) = \rho$ 得到 $\theta$，代入 $E$ 就是「前 $r$ 个奇异值占的能量比」。

```python
"""第 032 期 代码块 2：低秩瓶颈的代价 —— 随机矩阵的谱截断解析解 + 真 SVD 验证。"""
import math

import torch


def tail_theta(rho):
    """解 T(θ) = ρ，其中 T(θ) = 1 - 2θ/π - sin(2θ)/π。"""
    lo, hi = 0.0, math.pi / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - 2 * mid / math.pi - math.sin(2 * mid) / math.pi > rho:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def top_energy(rho):
    """前 ρ 比例的奇异值占总能量的多少（四分之一圆律）。"""
    t = tail_theta(rho)
    return 1 - 2 * t / math.pi + math.sin(4 * t) / (2 * math.pi)


print("§1  用真 SVD 验证解析式")
print(f"{'d':>6}{'r':>6}{'ρ':>10}{'解析':>12}{'真 SVD':>12}{'相对差':>10}")
for d in (256, 512, 1024):
    g = torch.Generator().manual_seed(d)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    s = torch.linalg.svdvals(A)
    tot = (s ** 2).sum()
    for r in (d // 32, d // 8, d // 2):
        real = ((s[:r] ** 2).sum() / tot).item()
        ana = top_energy(r / d)
        print(f"{d:>6}{r:>6}{r / d:>10.4f}{ana:>12.4f}{real:>12.4f}"
              f"{(ana - real) / real * 100:>9.2f}%")

print()
print("§2  外推到真实规模：n_h*d_h = 128*128 = 16384")
d = 128 * 128
print(f"{'保留维度 r':>12}{'ρ=r/d':>12}{'能量占比':>12}{'等效 GQA 组数':>16}")
for r in (256, 512, 1024, 2048, 4096, 8192):
    print(f"{r:>12}{r / d:>12.5f}{top_energy(r / d):>11.2%}{r / 256:>16.2f}")
print()
print(f"rank-512 在无结构随机矩阵上只能保住 {top_energy(512 / d):.1%} 的能量。")
print("MLA 用 512 维跑赢 16384 维，说明低维结构是被瓶颈「逼」出来的，不是先验存在的。")
```

```text
§1  用真 SVD 验证解析式
     d     r         ρ          解析       真 SVD       相对差
   256     8    0.0312      0.1122      0.1128    -0.57%
   256    32    0.1250      0.3759      0.3760    -0.03%
   256   128    0.5000      0.8937      0.8949    -0.13%
   512    16    0.0312      0.1122      0.1119     0.22%
   512    64    0.1250      0.3759      0.3758     0.00%
   512   256    0.5000      0.8937      0.8939    -0.02%
  1024    32    0.0312      0.1122      0.1120     0.17%
  1024   128    0.1250      0.3759      0.3754     0.11%
  1024   512    0.5000      0.8937      0.8935     0.03%

§2  外推到真实规模：n_h*d_h = 128*128 = 16384
      保留维度 r       ρ=r/d        能量占比       等效 GQA 组数
         256     0.01562      5.84%            1.00
         512     0.03125     11.22%            2.00
        1024     0.06250     20.99%            4.00
        2048     0.12500     37.59%            8.00
        4096     0.25000     62.29%           16.00
        8192     0.50000     89.37%           32.00

rank-512 在无结构随机矩阵上只能保住 11.2% 的能量。
MLA 用 512 维跑赢 16384 维，说明低维结构是被瓶颈「逼」出来的，不是先验存在的。
```

**结论很硬**：如果 K/V 是「无结构」的，rank-512 只能保住 **11.2%** 的能量。

MLA 能用 512 维跑赢 16384 维的 MHA，唯一解释是：**训练压力把 K/V 逼进了一个低维结构里**。这个结构不是先验存在的，是瓶颈「逼」出来的 —— 这也是为什么 MLA 不能像 GQA 那样直接从 MHA 权重池化初始化，而必须从头训（或者做专门的重构）。

顺带把 $d_c$ 的代价侧账本算清楚：

| $d_c$ | per-token KV/层 | 61 层全模型 | 压缩比 vs MHA | 相对 $d_c=512$ |
|---|---|---|---|---|
| 128 | 384 B | 22.88 KiB | 170.67× | 0.33× |
| 256 | 640 B | 38.12 KiB | 102.40× | 0.56× |
| **512** | **1152 B** | **68.62 KiB** | **56.89×** | **1.00×** |
| 1024 | 2176 B | 129.62 KiB | 30.12× | 1.89× |
| 2048 | 4224 B | 251.62 KiB | 15.52× | 3.67× |

$d_c$ 512 → 1024 换来的是「latent 多一点表达能力」，代价是 KV 体积变成 1.89 倍（+88.9%）。**收益侧要靠论文的消融实验来定，这里只说明「为什么不是无脑加宽」。** 另外注意 $d_h^R = 64$ 是刚性开销：$d_c=128$ 时它已经占 33.3% 了。

### 2.5 谱系总表

| 机制 | 每 token KV（元素） | 压缩比（$n_h=128, d_h=128$） | 质量 | 压缩比取值 | 权重侧等价物 |
|---|---|---|---|---|---|
| MHA | $2 n_h d_h = 32768$ | 1× | 强 | — | — |
| GQA($n_g$) | $2 n_g d_h = 256 n_g$ | $128/n_g$ | 中 | 离散 | 少量头 |
| MQA | $2 d_h = 256$ | 128× | 弱 | 单点 | 1 个头 |
| **MLA** | $d_c + d_h^R = 576$ | **56.89×** | **更强** | **连续** | 一个 latent 向量 |

来源：DeepSeek-V2 论文 Table 1（前四行的公式与「capability」列直接来自原文；最后一行的数字由本期核算）。

---

## 3 MLA 的三个零件

MLA 看起来复杂，其实是三个独立零件拼起来的。缺任何一个都不成立。

### 3.1 零件一：KV 联合低秩压缩

注意是**联合**压缩 —— K 和 V 共用同一个 latent，而不是各自压一次。为什么？

看权重侧的账：DeepSeek-V3 的 $W^{UK}$ 形状是 $(n_h d_h, d_c) = (16384, 512) = 8.39\text{M}$ 参数，$W^{UV}$ 一样；两者合计 16.78M。而 MHA 的 $W_K + W_V$ 是 $2 \times 16384 \times 7168 = 234.9\text{M}$ 参数 —— **14 倍**。

如果 K 和 V 各压各的，就要两套下投影 + 两套 latent，cache 从 512 变 1024，压缩比直接从 56.89 掉到 30.12。共用一份 latent 是**把「K 和 V 来自同一个 hidden state」这件事利用到了极致**。

论文里 $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$，即把 512 维的 latent 扩张回 128 个头 × 128 维。**同一个 latent 被上投影了 32768 次**（每个头、每个 K/V 维度都要一份）—— 这个「一份变两份、两份变 128 份」的扩张，正是下一节要消掉的东西。

### 3.2 零件二：解耦 RoPE（否则零件三不成立）

DeepSeek 想用 RoPE。论文自己承认这里撞了墙，原文（§2.1.3）：

> However, **RoPE is incompatible with low-rank KV compression**. To be specific, RoPE is position-sensitive for both keys and queries. If we apply RoPE for the keys $\mathbf{k}_t^C$, $W^{UK}$ in Equation 10 will be coupled with a position-sensitive RoPE matrix. In this way, $W^{UK}$ **cannot be absorbed into** $W^{Q}$ any more during inference, since a RoPE matrix related to the currently generating token will lie between $W^{Q}$ and $W^{UK}$ and **matrix multiplication does not obey a commutative law**.

用一行公式就能看懂这句话：

带 RoPE 的 score 是 $\mathbf{q}_i^\top R_{j-i} \mathbf{k}_j$（$R$ 是旋转矩阵）。要把它写成 latent 空间的内积，需要

$$
\underbrace{\mathbf{q}_i^\top R_{j-i} W^{UK}}_{\text{必须与 } j \text{ 无关才算「吸收」}}\ \mathbf{c}_j
$$

但 $R_{j-i}$ **随 key 位置 $j$ 变**，而你想吸收出来的那个 query 只有一个。要精确表出，必须 $W^{UK\top} R_{j-i} W^{UK} = I$ 对所有 $j$ 成立 —— 只有 $R = I$ 才行。

MLA 的解法是**不要让 RoPE 碰到可吸收的部分**：

$$
\mathbf{k}_t^R = \operatorname{RoPE}(W^{KR}\mathbf{h}_t)
$$

从 hidden state 直接投出一个 $d_h^R = 64$ 维的 key，**所有头共享**，只对它做 RoPE；而可吸收的 $d_h = 128$ 维那部分（论文叫 nope，"no position embedding"）完全不碰 RoPE。两段拼接起来：

$$
\mathbf{k}_{t,i} = [\underbrace{\mathbf{k}_{t,i}^C}_{\text{可吸收}}; \underbrace{\mathbf{k}_t^R}_{\text{带位置，共享}}]
$$

代价是 cache 里要多存那 64 维（占 11.1%），收益是**吸收成立**。这是一笔极其划算的交易。

**实验验证。** 先看两种形态（展开 vs 吸收）的输出差多少：

```python
"""第 032 期 代码块 3：MLA 的矩阵吸收 —— 两种形态在数学上等价。"""
import torch
import torch.nn as nn

torch.manual_seed(1)
D_MODEL, N_H, D_H, D_C, D_ROPE, Q_RANK = 512, 8, 64, 256, 32, 192


def rope(x, pos, base=10000.0):
    B, S, H, D = x.shape
    half = D // 2
    inv = base ** (-torch.arange(0, half, dtype=torch.float32) / half)
    ang = pos.float().unsqueeze(-1) * inv.unsqueeze(0)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x1 * cos[:, :, None, :] - x2 * sin[:, :, None, :]
    out[..., half:] = x1 * sin[:, :, None, :] + x2 * cos[:, :, None, :]
    return out


class Norm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class MLA(nn.Module):
    def __init__(self, rope_on_nope=False):
        super().__init__()
        self.rope_on_nope = rope_on_nope
        self.w_dq = nn.Linear(D_MODEL, Q_RANK, bias=False)
        self.q_norm = Norm(Q_RANK)
        self.w_uq = nn.Linear(Q_RANK, N_H * (D_H + D_ROPE), bias=False)
        self.w_dkv = nn.Linear(D_MODEL, D_C + D_ROPE, bias=False)
        self.kv_norm = Norm(D_C)
        self.w_uk = nn.Parameter(torch.randn(N_H, D_C, D_H) * 0.02)
        self.w_uv = nn.Parameter(torch.randn(N_H, D_C, D_H) * 0.02)
        self.w_o = nn.Linear(N_H * D_H, D_MODEL, bias=False)
        self.scale = (D_H + D_ROPE) ** -0.5

    def forward(self, x, pos, absorbed):
        B, S, _ = x.shape
        q = self.w_uq(self.q_norm(self.w_dq(x))).view(B, S, N_H, D_H + D_ROPE)
        q_nope, q_pe = q[..., :D_H], rope(q[..., D_H:], pos)
        raw = self.w_dkv(x)
        latent, k_pe_raw = self.kv_norm(raw[..., :D_C]), raw[..., D_C:]
        k_pe = rope(k_pe_raw.unsqueeze(2), pos).squeeze(2)     # 共享，无 head 维
        lat = latent[:, :, None, :].expand(B, S, N_H, D_C)
        mask = torch.ones(S, S).tril().bool()

        if not absorbed:
            # 形态 A：先把 latent 展开成每头 K/V
            knope = torch.einsum("bsl,hld->bshd", latent, self.w_uk)
            v = torch.einsum("bsl,hld->bshd", latent, self.w_uv)
            if self.rope_on_nope:
                knope = rope(knope, pos)
            k = torch.cat([knope, k_pe[:, :, None].expand(B, S, N_H, D_ROPE)], -1)
            att = torch.einsum("bqhd,bkhd->bhqk", torch.cat([q_nope, q_pe], -1), k)
            att = (att * self.scale).masked_fill(~mask, float("-inf")).softmax(-1)
            o = torch.einsum("bhqk,bkhd->bqhd", att, v)
        else:
            # 形态 B：W_UK 吸进 Q，W_UV 吸进 O，V 直接用 latent
            ql = torch.einsum("bshd,hld->bshl", q_nope, self.w_uk)   # (B,S,H,D_C)
            ka = torch.cat([lat, k_pe[:, :, None].expand(B, S, N_H, D_ROPE)], -1)
            att = torch.einsum("bqhd,bkhd->bhqk", torch.cat([ql, q_pe], -1), ka)
            att = (att * self.scale).masked_fill(~mask, float("-inf")).softmax(-1)
            o_lat = torch.einsum("bhqk,bkhd->bqhd", att, lat)
            o = torch.einsum("bshl,hld->bshd", o_lat, self.w_uv)     # 吸收 W_UV
        return self.w_o(o.reshape(B, S, N_H * D_H))


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


x = torch.randn(2, 24, D_MODEL)
pos = torch.arange(24).unsqueeze(0).expand(2, 24)
for flag in (False, True):
    torch.manual_seed(7)
    m = MLA(rope_on_nope=flag).eval()
    with torch.no_grad():
        yA, yB = m(x, pos, False), m(x, pos, True)
    tag = "RoPE 加在 nope 维（旧方案）" if flag else "解耦 RoPE（DeepSeek 方案）"
    print(f"{tag}：输出相对误差 {rel(yB, yA):.3e}"
          f"   绝对误差 {(yB - yA).abs().max():.3e}   输出尺度 {yA.abs().max():.4f}")
```

```text
解耦 RoPE（DeepSeek 方案）：输出相对误差 5.023e-07   绝对误差 3.576e-07   输出尺度 0.6275
RoPE 加在 nope 维（旧方案）：输出相对误差 6.575e-02   绝对误差 2.566e-02   输出尺度 0.6275
```

解耦 RoPE 下，两种形态的输出相对误差是 **5.02e-07** —— 就是 float32 的舍入误差，恒等式成立。如果把 RoPE 加在 nope 维上（旧方案），误差跳到 **6.58e-02**（6.6%）。

再给一个更强的命题：**就算允许为每个 (batch, head, query位置) 各自最优地挑一个 latent 空间向量**，也拟合不出对 nope 加 RoPE 的 score。用最小二乘给出这个下界：

```python
"""第 032 期 代码块 4：RoPE 为什么不能加在可吸收的维度上（最小二乘下界）。
接上一块，沿用同一个命名空间。
"""
S2 = 1024
torch.manual_seed(7)
m = MLA().eval()
x = torch.randn(1, S2, D_MODEL)
pos = torch.arange(S2).unsqueeze(0)

with torch.no_grad():
    q = m.w_uq(m.q_norm(m.w_dq(x))).view(1, S2, N_H, D_H + D_ROPE)
    q_nope = q[..., :D_H]
    latent = m.kv_norm(m.w_dkv(x)[..., :D_C])                 # (1, S2, D_C)
    knope = torch.einsum("bsl,hld->bshd", latent, m.w_uk)       # (1, S2, H, D_H)
    k_rot = rope(knope, pos)
    A = latent[0]                                               # (S2, D_C)

    print(f"{S2} 个 key 位置，latent 维度 {D_C}（方程数 {S2} > 未知数 {D_C}，超定）")
    plain, rotated = [], []
    for h in range(N_H):
        qh = q_nope[0, :, h, :]
        t_plain = qh @ knope[0, :, h, :].T
        t_rot = qh @ k_rot[0, :, h, :].T
        for i in (0, S2 // 4, S2 // 2, S2 - 1):
            a = torch.linalg.lstsq(A, t_plain[i]).solution
            ar = torch.linalg.lstsq(A, t_rot[i]).solution
            plain.append(((A @ a - t_plain[i]).norm() / t_plain[i].norm()).item())
            rotated.append(((A @ ar - t_rot[i]).norm() / t_rot[i].norm()).item())

print("用「一个 latent 空间的 query a_i」拟合 q_i·k_j（k 不旋转）：")
print(f"  残差 中位 {sorted(plain)[len(plain) // 2]:.3e}  最大 {max(plain):.3e}")
print("  可以精确表出 —— a_i = W_UKᵀ q_i 就是那个解")
print("用「一个 latent 空间的 query a_i」拟合 q_i·R_(j-i)·k_j（k 旋转）：")
print(f"  残差 中位 {sorted(rotated)[len(rotated) // 2]:.4f}  最大 {max(rotated):.4f}")
print("  拟合不了：要精确表出须 W_UKᵀ R_(j-i) W_UK = I 对所有 j 成立，只有 R = I 才行")
```

```text
1024 个 key 位置，latent 维度 256（方程数 1024 > 未知数 256，超定）
用「一个 latent 空间的 query a_i」拟合 q_i·k_j（k 不旋转）：
  残差 中位 4.246e-07  最大 4.637e-07
  可以精确表出 —— a_i = W_UKᵀ q_i 就是那个解
用「一个 latent 空间的 query a_i」拟合 q_i·R_(j-i)·k_j（k 旋转）：
  残差 中位 0.7182  最大 0.7981
  拟合不了：要精确表出须 W_UKᵀ R_(j-i) W_UK = I 对所有 j 成立，只有 R = I 才行
```

- 不给 nope 加 RoPE：残差 **4.25e-07**（精确可表出，$a_i = W^{UK\top}\mathbf{q}_i$ 就是解）
- 给 nope 加 RoPE：残差 **0.7182**（中位），最大 0.7981 —— 和 score 自己是同一量级

**这里有一个必须提醒的坑**：这个最小二乘下界只在 $S \gg d_c$ 时才有意义。key 位置比 latent 维度还少时，线性系统欠定，任何目标都能被精确拟合，做出来的结论是**假阳性**。上面用 1024 个 key 位置对 256 维 latent（超定 4 倍）才是真实推理的区间。

### 3.3 零件三：矩阵吸收

论文 §2.1.2 的原话：

> During inference, since $W^{UK}$ can be absorbed into $W^{Q}$, and $W^{UV}$ can be absorbed into $W^{O}$, we **even do not need to compute keys and values out for attention**.

吸收就是两次结合律。对每个头：

$$
\underbrace{(\mathbf{q}_{\text{nope}})_h^\top}_{\text{1}\times d_h} \underbrace{W^{UK}_h}_{d_h \times d_c} \underbrace{\mathbf{c}_j}_{d_c} = \underbrace{\left(W^{UK\top}_h \mathbf{q}_{\text{nope}}\right)_h^\top}_{\text{1} \times d_c}\ \mathbf{c}_j
$$

也就是把 $W^{UK}$ 从 K 侧搬到 Q 侧。V 侧类似：$\sum_j \alpha_j (W^{UV}\mathbf{c}_j) = W^{UV}\left(\sum_j \alpha_j \mathbf{c}_j\right)$，把 $W^{UV}$ 搬到输出侧。

vLLM 的 MLA 实现按 $S_q / S_{kv}$ 的比值走两条路径（这两段伪代码来自 vLLM 源码文档，不是我的推断）：

**形态 A —— 展开（prefill / extend，$S_q/S_{kv}$ 接近 1）**：先把 latent 展开成每头 K/V，再跑标准 MHA kernel：

```text
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c = cat([new_kv_c, cache_kv_c], dim=0)
k_nope = (kv_c @ W_UK.view(Lkv, N*P)).view(Skv, N, P)
v      = (kv_c @ W_UV.view(Lkv, N*V)).view(Skv, N, V)
sdpa_o = sdpa(cat([q_nope, q_pe], -1), cat([k_nope, k_pe.expand(-1, N, -1)], -1), v)
```

**形态 B —— 吸收（decode，$S_q/S_{kv}$ 接近 0）**：直接在 latent 空间做 attention：

```text
q_nope = (q_c @ W_UQ).view(-1, N, P)
ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)     # ← 吸收 W_UK
q_pe   = RoPE(q_c @ W_QR).view(Sq, N, R)
sdpa_o = sdpa(cat([ql_nope, q_pe], -1), cat([kv_c, k_pe], -1), kv_c)
o      = einsum("snl,lnv->snv", sdpa_o.reshape(-1, N, Lkv), W_UV)   # ← 吸收 W_UV
```

注意两件事：

1. **两种形态 cache 里存的东西完全一样**（都是 576 维 latent + RoPE key）。差别只在「谁来展开」。
2. 形态 B 里注意力的 **K 维变成 $L_{kv} + R = 576$、V 维变成 $L_{kv} = 512$**，而形态 A 是 $P + R = 192$ 和 $V = 128$。

**所以「MLA 在 decode 时退化成一个 MQA」不是比喻，是字面事实**：形态 B 里所有 query 头共享同一份 576 维的 K 和 512 维的 V。

还有一个细节值得单独说：`kv_a_layernorm` 的 weight 能不能折进上投影？

判据是「**能折的东西不能依赖被缓存的 token**」：

- $\gamma$（RMSNorm 的 weight）与 $\mathbf{c}$ 无关 → 可以折进 $W^{UK}$（也可以折进 query，因为 $(\mathbf{q} \odot \gamma)\cdot\mathbf{c} = \mathbf{q}\cdot(\gamma \odot \mathbf{c})$）。实测折完之后 $k_{\text{nope}}$ 的相对误差是 **4.06e-07**。
- $1/\|\mathbf{c}_j\|$ 依赖第 $j$ 个被缓存的 token → **只能逐 key 位置乘回 score**，不能折进 query 侧。

---

## 4 吸收的代价与临界点

吸收听上去是免费的午餐。它不是。展开形态做的事情被省掉了，但 attention 本身变宽了 3.4 倍。

### 4.1 两种形态的账

只算两种形态**不同**的部分（query 的投影两边都有，不参与比较）：

```python
"""第 032 期 代码块 5：吸收的代价模型与临界点。"""
N_H, P, R, V, LKV, L = 128, 128, 64, 128, 512, 61


def expand(sq, skv):
    """展开形态：把每个被缓存 token 的 latent 上投影成每头 K/V，再跑窄 head attention。"""
    return skv * N_H * (P + V) * LKV, sq * skv * N_H * (P + R + V)


def absorb(sq, skv):
    """吸收形态：latent 空间的 query，宽 head attention。"""
    return sq * N_H * P * LKV, sq * skv * N_H * (2 * LKV + R)


def f(x):
    for u, s in [(1e12, "T"), (1e9, "G"), (1e6, "M")]:
        if x >= u:
            return f"{x / u:.2f}{s}"
    return f"{x:.0f}"


print("DeepSeek-V3 配置：N_H=128 P=128 R=64 V=128 Lkv=512 L=61")
print()
print("§1  decode（Sq=1）的边际成本：每多缓存一个 token")
print(f"{'形态':<8}{'上投影':>12}{'attention':>14}{'合计':>12}")
pe, ae = N_H * (P + V) * LKV, N_H * (P + R + V)
pa, aa = 0, N_H * (2 * LKV + R)
print(f"{'展开':<8}{f(pe):>12}{f(ae):>14}{f(pe + ae):>12}")
print(f"{'吸收':<8}{f(pa):>12}{f(aa):>14}{f(pa + aa):>12}")
print(f"上投影 / 吸收 attention = {pe / aa:.1f}x")
print()
print("§2  decode 下两种形态的总量随上下文 S 变化")
print(f"{'S':>8}{'展开':>12}{'吸收':>12}{'展开/吸收':>12}")
for S in (1024, 4096, 16384, 65536, 131072):
    e, a = sum(expand(1, S)), sum(absorb(1, S))
    print(f"{S:>8}{f(e):>12}{f(a):>12}{e / a:>11.1f}x")
print()
print("§3  prefill（Sq=Skv=S）—— 展开反而便宜")
print(f"{'S':>8}{'展开':>12}{'吸收':>12}{'展开/吸收':>12}")
for S in (64, 128, 256, 1024, 4096):
    e, a = sum(expand(S, S)), sum(absorb(S, S))
    print(f"{S:>8}{f(e):>12}{f(a):>12}{e / a:>11.2f}x")
print()
# 推导：Skv(P+V)Lkv + Sq·Skv(P+R+V) = Sq·P·Lkv + Sq·Skv(2Lkv+R)
# Sq=Skv=S 时化简为 S·Lkv·V = S²(2Lkv - P - V) → S* = Lkv·V/(2Lkv-P-V)
s_star = LKV * V / (2 * LKV - P - V)
print(f"prefill 临界点 S* = Lkv*V/(2Lkv-P-V) = {LKV}*{V}/{2 * LKV - P - V}"
      f" = {LKV * V}/{2 * LKV - P - V} = {s_star:.1f}")
print(f"S < {s_star:.1f} 该吸收；S > {s_star:.1f} 该展开。常见 prefill chunk ≥ 512，所以 prefill 一律展开。")
print()
print("§4  吸收形态的隐藏账单：query 权重变大")
w0, w1 = 1536 * N_H * (P + R), 1536 * (N_H * (P + R) + N_H * LKV)
print(f"W_UQ 展开时  1536 x {N_H * (P + R)} = {w0:,} 参数 = {w0 * 2 / 1e6:.1f} MB")
print(f"W_UQ 吸收后  1536 x {N_H * (P + R) + N_H * LKV} = {w1:,} 参数 = {w1 * 2 / 1e6:.1f} MB")
print(f"每层多 {w1 - w0:,} 参数，{L} 层共多 {(w1 - w0) * L * 2 / 1e9:.1f} GB")
```

```text
DeepSeek-V3 配置：N_H=128 P=128 R=64 V=128 Lkv=512 L=61

§1  decode（Sq=1）的边际成本：每多缓存一个 token
形态               上投影     attention          合计
展开            16.78M         40960      16.82M
吸收                 0        139264      139264
上投影 / 吸收 attention = 120.5x

§2  decode 下两种形态的总量随上下文 S 变化
       S          展开          吸收       展开/吸收
    1024      17.22G     150.99M      114.1x
    4096      68.89G     578.81M      119.0x
   16384     275.55G       2.29G      120.3x
   65536       1.10T       9.14G      120.7x
  131072       2.20T      18.26G      120.7x

§3  prefill（Sq=Skv=S）—— 展开反而便宜
       S          展开          吸收       展开/吸收
      64       1.24G       1.11G       1.12x
     128       2.82G       3.36G       0.84x
     256       6.98G      11.27G       0.62x
    1024      60.13G     154.62G       0.39x
    4096     755.91G       2.37T       0.32x

prefill 临界点 S* = Lkv*V/(2Lkv-P-V) = 512*128/768 = 65536/768 = 85.3
S < 85.3 该吸收；S > 85.3 该展开。常见 prefill chunk ≥ 512，所以 prefill 一律展开。

§4  吸收形态的隐藏账单：query 权重变大
W_UQ 展开时  1536 x 24576 = 37,748,736 参数 = 75.5 MB
W_UQ 吸收后  1536 x 90112 = 138,412,032 参数 = 276.8 MB
每层多 100,663,296 参数，61 层共多 12.3 GB
```

### 4.2 三个能背下来的结论

**① decode 的边际成本差 120.5 倍。**
展开形态每多缓存一个 token，就要多做 $N(P+V)L_{kv} = 16.78\text{M}$ 的上投影；而吸收形态每多缓存一个 token，attention 总共才 $N(2L_{kv}+R) = 139{,}264$。两者都随 $S$ 线性增长，比的是常数 —— 120.5 : 1。

**② decode 里不存在临界点。**
展开的常数项是 $N(P+V)L_{kv} = 16.78\text{M}$，吸收的一次性项是 $N P L_{kv} = 8.39\text{M}$。$S \geq 1$ 时吸收就已经赢了，而且差距随 $S$ 拉大（114× → 120.7×）。**在这个配置下 decode 永远该吸收。**

**③ prefill 有临界点，而且是解析的。**
令 $S_q = S_{kv} = S$，两边相等化简后 $S$ 的二次项和一次项对齐，得到

$$
\boxed{\ S^{*} = \frac{L_{kv} \cdot V}{2L_{kv} - P - V}\ }
$$

代入 DeepSeek-V3：$S^* = 512 \times 128 / (1024 - 128 - 128) = 65536/768 = \mathbf{85.3}$。

- $S < 85.3$：展开更贵，该吸收
- $S > 85.3$：吸收更贵，该展开

常见 prefill chunk 是 512 起，所以 **prefill 一律展开** —— 这与 vLLM 用 $S_q/S_{kv}$ 做判据的做法完全吻合（prefill 时 $S_q/S_{kv} \approx 1$，decode 时 $\approx 0$）。

推导里那个 $-P - V$ 值得看一眼：分子 $L_{kv}V$ 来自「展开时 V 侧要付的上投影」，分母 $2L_{kv} - P - V$ 来自「吸收时宽出来的那部分维度减去原来窄 head 的维度」。量纲上两边都是「元素数」，自洽。

### 4.3 「data-movement friendly」到底 friendly 在哪

如果两种形态都从 HBM 读同样多的 latent，那吸收的好处不在带宽上。真正被省掉的是**展开形态必须物化的那片中间张量**：

| 形态 | 每层每 token 需要摆开的中间张量 | 元素数 | 字节（bf16） |
|---|---|---|---|
| 展开（每头 K/V） | $N(P+V)$ | 32768 | 65,536 B |
| 吸收（latent） | $L_{kv}+R$ | 576 | 1,152 B |

**56.9 倍 —— 正好等于「MLA 相对 MHA 的 KV 压缩比」**。道理很简单：**展开后的形态本来就是 MHA**。

它同时还把 AV 的算术强度改了：

- 展开：$2 N V = 32{,}768$ FLOPs / 1152 B = **28.4 FLOP/byte**
- 吸收：$2 N L_{kv} = 131{,}072$ FLOPs / 1152 B = **113.8 FLOP/byte**

吸收用了 4 倍的 FLOPs 换同样的字节。在纯带宽瓶颈下这是免费的（decode 算力利用率零点几个百分点），但 batch 和上下文一上去就要开始还账 —— 这正是 028 期那个「$B \cdot L = 295$ 平衡点」在另一处的现身。

### 4.4 还有两笔隐形账单

**① 权重变大。** 吸收要在加载时把 $W^{UK}$ 折进 $W^{UQ}$：$W^{UQ}$ 从 $1536 \times 24576$ 涨到 $1536 \times 90112$，**每层多 100.66M 参数，61 层共多 12.3 GB**。生产实现要同时留着展开形态的 $W^{UK}$（prefill 用）和吸收后的 $W^{UQ}$（decode 用），这是「同一套权重、两种形态、按阶段切换」的物理代价。

**② 长上下文里吸收的 attention 会变成大头。** 吸收形态每 token 的 attention FLOPs 是 $2 N (2L_{kv}+R) S = 278{,}528 \cdot S$。$S = 131072$ 时是 **36.5 GFLOP**，而 DeepSeek-V3 激活参数 37B、每 token 全部前向约 74 GFLOP —— attention 已经占到 **49%**。到那个长度，MLA 这份「3.4 倍」的账就不再免费了。

---

## 5 围绕这个领域展开

### 5.1 GQA-8 为什么是事实标准

看一圈现代模型的 KV 配置：

| 模型 | $n_g$ |
|---|---|
| Llama-2-7B / 13B | 32 / 40（即 MHA） |
| Llama-2-70B | 8 |
| Llama-3 全系列 | 8 |
| Qwen2.5-72B | 8 |
| Mistral / Mixtral | 8 |
| Yi-34B | 8 |

**8 组几乎是全行业收敛点**。这不是数学最优，是工程甜点。为什么是 8 而不是 4 或 16？

GQA 论文给的答案是「uptrained GQA 的质量接近 MHA」，但那是大规模预训练实验的结论。小模型上量不出来 —— 下面这个实验正好说明**为什么**：

```python
"""第 032 期 代码块 6：GQA 的组数饱和 —— mean-pool 到 g 组，再继续训练看能否恢复。

任务：in-context copy（必须靠 attention）。刻意欠训练，让 CE 留下改进空间。
CPU 上约 15 秒。
"""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
torch.manual_seed(0)

VOCAB, HALF, SEP = 128, 24, 128
VOCAB_ALL, D, N_H, N_LAYER, FFN = VOCAB + 1, 128, 8, 2, 256
D_H, STEPS, RECOVERY, BATCH = D // N_H, 500, 250, 24


def batch(bs, gen):
    h = torch.randint(0, VOCAB, (bs, HALF), generator=gen)
    seq = torch.cat([h, torch.full((bs, 1), SEP), h], 1)
    tgt = torch.full_like(seq, -100)
    tgt[:, HALF + 1:] = h
    return seq, tgt


class Blk(nn.Module):
    def __init__(self, n_kv):
        super().__init__()
        self.n_kv = n_kv
        self.n1 = nn.LayerNorm(D)
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, n_kv * D_H, bias=False)
        self.wv = nn.Linear(D, n_kv * D_H, bias=False)
        self.wo = nn.Linear(D, D, bias=False)
        self.n2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, FFN, bias=False)
        self.fc2 = nn.Linear(FFN, D, bias=False)
        self.scale = D_H ** -0.5

    def forward(self, x, sink=None):
        B, S, _ = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B, S, N_H, D_H).transpose(1, 2)
        k = self.wk(h).view(B, S, self.n_kv, D_H).transpose(1, 2)
        v = self.wv(h).view(B, S, self.n_kv, D_H).transpose(1, 2)
        k = k.repeat_interleave(N_H // self.n_kv, 1)
        v = v.repeat_interleave(N_H // self.n_kv, 1)
        att = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.ones(S, S).tril().bool()
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        if sink is not None:
            sink.append(att.detach())
        o = (att @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.wo(o)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class LM(nn.Module):
    def __init__(self, n_kv=N_H):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_ALL, D)
        self.pos = nn.Embedding(64, D)
        self.layers = nn.ModuleList([Blk(n_kv) for _ in range(N_LAYER)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB_ALL, bias=False)

    def forward(self, x, sink=None):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        for i, b in enumerate(self.layers):
            h = b(h, None if sink is None else sink.setdefault(i, []))
        return self.head(self.norm(h))


def pool(mha, g):
    """把 MHA 的 K/V 头按 g 组 mean-pool —— GQA 论文从 MHA 初始化 GQA 的做法。"""
    gqa = LM(g)
    sd, td = mha.state_dict(), gqa.state_dict()
    for k in sd:
        if k.endswith(("wk.weight", "wv.weight")):
            td[k] = sd[k].view(g, N_H // g, D_H, D).mean(1).reshape(g * D_H, D)
        else:
            td[k] = sd[k].clone()
    gqa.load_state_dict(td)
    return gqa


def loss_of(m, gen, n=16):
    m.eval()
    tot = cnt = 0
    with torch.no_grad():
        for _ in range(n):
            x, t = batch(BATCH, gen)
            tot += F.cross_entropy(m(x)[:, :-1].reshape(-1, VOCAB_ALL),
                                   t[:, 1:].reshape(-1), ignore_index=-100,
                                   reduction="sum").item()
            cnt += (t[:, 1:] != -100).sum().item()
    return tot / cnt


def fit(m, gen, steps):
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=0.3)
    m.train()
    for _ in range(steps):
        x, t = batch(BATCH, gen)
        loss = F.cross_entropy(m(x)[:, :-1].reshape(-1, VOCAB_ALL),
                               t[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    return m


def map_sim(m, gen):
    """同层内不同头的 attention 概率矩阵两两余弦相似度（均值）。"""
    x, _ = batch(8, gen)
    sink = {}
    m.eval()
    with torch.no_grad():
        m(x, sink)
    out = {}
    for li, atts in sink.items():
        a = atts[0]
        f = a.reshape(a.shape[0], a.shape[1], -1)
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cs = [(f[:, i] * f[:, j]).sum(-1).mean().item()
              for i, j in itertools.combinations(range(N_H), 2)]
        out[li] = sum(cs) / len(cs)
    return out


gen = torch.Generator().manual_seed(0)
mha = fit(LM(N_H), gen, STEPS)
base = loss_of(mha, gen)
print(f"MHA base CE = {base:.4f}")
print()
print(f"{'g (KV 组数)':<14}{'KV 压缩':>10}{'池化后 CE':>14}"
      f"{'attention 输出误差':>20}{'恢复 250 步后':>16}")
for g in (8, 4, 2, 1):
    if g == N_H:
        print(f"{'8 (= MHA)':<14}{'1x':>10}{base:>14.4f}{0.0:>20.4f}{base:>16.4f}")
        continue
    torch.manual_seed(1000 + g)
    q = pool(mha, g)
    x, _ = batch(16, gen)
    a, b = {}, {}
    with torch.no_grad():
        mha(x, a)
        q(x, b)
    err = math.sqrt(sum(((b[i][-1] - a[i][-1]) ** 2).sum().item() for i in a)
                    / sum((a[i][-1] ** 2).sum().item() for i in a))
    z = loss_of(q, gen)
    fit(q, gen, RECOVERY)
    print(f"{g:<14}{N_H // g:>9}x{z:>14.4f}{err:>20.4f}{loss_of(q, gen):>16.4f}")

print()
sim = map_sim(mha, gen)
for li, v in sim.items():
    print(f"第 {li} 层不同头的 attention 概率矩阵两两余弦相似度均值 = {v:.4f}")
print("相似度很高说明各头实际用到的 K/V 结构相似 —— 这是「共享 K/V 便宜」的直接证据。")
print("但池化的零点损失很大、而且 250 步就能恢复 → 损失的性质是「参数错位」而非「容量不足」。")
```

```text
MHA base CE = 0.0030

g (KV 组数)          KV 压缩        池化后 CE      attention 输出误差       恢复 250 步后
8 (= MHA)             1x        0.0030              0.0000          0.0030
4                     2x        0.3555              0.6482          0.0010
2                     4x        3.7044              0.8411          0.0022
1                     8x        5.2713              0.8972          0.0038

第 0 层不同头的 attention 概率矩阵两两余弦相似度均值 = 0.8263
第 1 层不同头的 attention 概率矩阵两两余弦相似度均值 = 0.8350
相似度很高说明各头实际用到的 K/V 结构相似 —— 这是「共享 K/V 便宜」的直接证据。
但池化的零点损失很大、而且 250 步就能恢复 → 损失的性质是「参数错位」而非「容量不足」。
```

这张表要分两层读：

**第一层：池化的零点损失很陡。** $g=4$ 时 CE 从 0.0030 涨到 0.3555，$g=1$（MQA）时涨到 5.27 —— 掉了三个数量级。而且注意 attention 输出的相对误差：$g=4$ 就已经 0.6482 了。

**第二层（关键）：恢复 250 步就全回来了。** $g=1$ 的 CE 从 5.27 回到 0.0038，和基线 0.0030 基本无差。**这说明池化造成的损失性质是「参数错位」，不是「容量不足」。**

还有一个佐证（实验 6，同一个模型）：

| 诊断 | 结果 |
|---|---|
| 同一层内 K/V 头投影矩阵两两余弦相似度（均值 \|cos\|） | 0.0375 / 0.0186（几乎正交） |
| 同一层内不同头的 attention 概率矩阵两两余弦相似度 | 0.86 / 0.81（高度相似） |
| 组内 K/V 头**求和**（只改尺度不改方向）后的 CE 增量 | $g=4$: +0.2845，$g=1$: +5.1564 |

三行连起来读：**权重几乎正交（不是权重冗余），但功能高度相似（attention 用到的 K/V 结构很像）**；而「求和」保留了方向只改了尺度，照样崩 —— 所以崩的原因是「丢掉了各头自己的方向」，不是「尺度变了」。

**这恰好解释了 GQA 论文为什么能用 5% 的算力做 uptraining**：被拿掉的不是一个容量，是一组自由度，而自由度可以通过继续训练重新分配。

同时它也解释了**为什么小区间实验量不出 GQA 组数的甜点**：在 tiny 模型上，连 MQA 都能靠 250 步找回来。真正的区分需要「训练预算受限」「任务复杂度高」两个条件同时成立 —— 那是大模型预训练的尺度，不是笔记本 CPU 能复现的。

所以「8 组」这个数字的来源是这样的：**上限由 MQA 的质量损失定，下限由 GQA 的压缩收益定，中间找一个「upside 已经消失、downside 还没出现」的位置**。对 $n_h = 64$ 的模型来说，$n_g = 8$ 正好是 8 倍压缩、8 个 query 头共享一组 —— 一个既够粗又还有余量的点。

### 5.2 MLA 的三笔额外工程账

**① TP 装不下它。** 这一条是 MLA 最容易被低估的代价。GQA 的 KV 有 $n_g$ 个头，TP 可以按头切；**MLA 的 KV 是「一个」latent 向量**（等价于 $n_g = 1$），TP 切不动。

vLLM 的官方文档（ROCm 优化指南）把这件事写得很直白：

> For models with Multi-head Latent Attention (MLA) architecture like DeepSeek V2, V3, and R1, vLLM supports **Data Parallel Attention** … This **avoids KV cache duplication across tensor parallel ranks**, significantly reducing memory usage and enabling larger batch sizes.

社区的实测口径也一致：TP=8 时 MLA 的 latent cache 会在 8 个 rank 上各复制一份（有人算过 R1 在 H100 上 DP 路线的 KV 池要 29.2 GiB，而实际只剩 10.8 GiB 可用）。官方推荐的两条路线：

```bash
# vLLM：attention 走 DP，MoE 走 EP
VLLM_ALL2ALL_BACKEND="allgather_reducescatter" \
vllm serve deepseek-ai/DeepSeek-R1 \
  --data-parallel-size 8 --enable-expert-parallel --disable-nccl-for-dp-synchronization

# SGLang：同样的思路
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 --dp-size 8 --enable-dp-attention --enable-ep-moe --trust-remote-code
```

**注意 DP attention 在这里的含义和普通的 DP 副本不一样**：这些 rank 是**同一个模型实例**，要在共享的 MoE 层上锁步前进（vLLM 会用一个空 forward 来对齐没有活的 rank）。这是一个专为 MLA 存在的新型并行维度，是 MLA 带进推理工程的**结构性**变化。

**② 与 PagedAttention 的接口变了。** 第 011 期讲的分页机制本身不动（对象还是「块」），但块里装的东西从「$n_g$ 个头的 K + V」变成「latent + 共享 RoPE key 的拼接」。以 block = 16 为例，每层每块的字节数是

$$
16 \times 576 \times 2 = 18{,}432 \ \text{B} = 18 \ \text{KiB}
$$

而 GQA-8 是 $16 \times 2048 \times 2 = 64 \ \text{KiB}$。**块粒度带来的损失规律（029 期的 `floor(L/16)`）完全不变** —— 因为它是按 token 数算的，与每 token 多少字节无关。

**③ 跨机传输的判据被推着走。** 第 030 期给过一个判据：一个满负荷 decode 实例需要的入带宽正比于 per-token KV 体积，临界 $S/G \approx 12$（IB400）。MLA 把 per-token KV 压到 GQA-8 的 $1/3.556$，同一个判据下临界 $S/G$ 会抬到 **约 43**（本期按 030 的公式线性外推，不是实测）。这正是 DeepSeek 能用长 prompt 跑 PD 分离的底层理由。

### 5.3 往前一步：从「压得小」到「看得少」

MLA 压的是**每个 token 的表示**。还有一条正交的轴：**少看几个 token**。

- **NSA**（Native Sparse Attention）和 DeepSeek-V3.2-Exp 的 **DSA**（DeepSeek Sparse Attention）走的是后者：用一个轻量 indexer 打分，只对 top-k 个 key 做完整 attention。

两条轴**可以叠加**：MLA 决定「每个 key 花多少字节」，稀疏决定「读几个 key」。乘起来就是 attention 的实际字节数。这也是为什么现在的架构讨论里，MLA 和稀疏注意力被放在同一张图里。

顺便，它们和已经发过的几期也接得上：

- 028（连续批处理）那套调度器里**没有 prefill/decode 之分**，只有 `num_computed_tokens` 追 `num_tokens_with_spec`。MLA 的两种形态恰好对应调度器眼里的两种步型 —— **吸收形态是 decode 步的默认路径，展开形态是大 chunk prefill 的默认路径**，而调度器只需要知道这一步有多少个 query token。
- 029（前缀缓存）的「命中率 = 省下的 prefill FLOPs 比例」是精确等式，与模型无关。但**命中之后剩多少时间**取决于 per-token KV —— MLA 让那条「带宽地板」低了 3.56 倍。
- 030（PD 分离）里「搬 KV vs 重算 prefill」的临界带宽（单请求 4K 上下文只要 3.87 GB/s）会被 MLA 再除以 3.556 → **约 1.1 GB/s**。跨机搬运变得更容易成立。

### 5.4 一张图看谱系

```text
                    压缩轴一：减 KV 头数            压缩轴二：降维
                    （离散，n_g 必须整除 n_h）      （连续，d_c 任意）

   MHA n_h=128  ──────────────────────►  GQA n_g=8  ──────►  MQA n_g=1
   32768 elem        16x / 32x / 128x      2048 elem          256 elem
                                            │
                                            │  两者正交，可以叠加
                                            ▼
                                        MLA  576 elem
                                        （d_c=512 + d_h^R=64）
                                        56.89x，等效 GQA 2.25 组
                                        ▸ 低秩联合压缩：每层每 token 存 576
                                        ▸ 解耦 RoPE：让吸收成立（代价 +64）
                                        ▸ 矩阵吸收：decode 用吸收形态
                                                  prefill 用展开形态
```

---

## 6 什么时候该用 / 不该用

| 场景 | 选择 | 理由 |
|---|---|---|
| 从头训一个新模型，长上下文是核心卖点 | **MLA** | 压缩比可以连续调，质量上限比 GQA 高 |
| 已经有 MHA checkpoint，想省推理显存 | **GQA-8 + 5% 算力 uptraining** | GQA 论文的原方案；MLA 需要从头训 |
| 短上下文、高 QPS、延迟敏感 | **GQA-8** | MLA 在 S < 85 时反而更贵，且 TP 装不下 |
| 单卡或小规模部署，不想改并行策略 | **GQA-8** | MLA 的 DP attention 需要额外的工程改造 |
| 已经用 MLA 了，要跑长上下文 | **确认 attention backend 走的是吸收形态** | 展开形态在 $S=4096$ 时多付 119 倍的上投影 |
| 已经用 MLA 了，要跑大 batch prefill | **确认 prefill 走的是展开形态** | $S=4096$ 时吸收形态多花 3.1 倍 FLOPs |
| 追求极致压缩又不想改架构 | **KV 量化（fp8）** | 与 MLA/GQA 正交，直接再砍一半字节 |

**最该记住的一条**：MLA 的「两种形态」不是实现细节，是**必须按阶段切换的两个 kernel 路径**。选 engine 的时候，先确认它有没有把这两条路都实现好 —— 只实现一条的 engine 在另一半负载上会明显吃亏。

---

## 7 常见坑

**坑 1：以为 MLA 的收益是「低秩」，所以可以拿来压已有的 MHA 权重。**
不是。本期的实验 2 说得很清楚：无结构随机矩阵上 rank-512 只保住 11.2% 能量。MLA 的低维结构是**训练逼出来的**，不是 MHA 权重里本来就有、可以 SVD 出来的。GQA 可以靠池化初始化 + 5% 算力 uptrain，MLA 不能 —— 把 MHA 直接投到低维 latent 上再接着训，等于重新学一遍。

**坑 2：把 `num_key_value_heads` 当成 MLA 的组数。**
DeepSeek-V3 的 `config.json` 里写着 `"num_key_value_heads": 128`（和 `num_attention_heads` 一样）—— 这个字段在 MLA 路径下**根本不参与计算**。MLA 的「KV 头数」逻辑上是 1，KV 的维度是 $d_c + d_h^R$。看到 `num_key_value_heads: 128` 就推断「这是 MHA」，是错的。

**坑 3：把「吸收」当成无损的免费优化，不看 $S$。**
吸收在 decode 里永远赢（本配置下），但在 prefill 里只要 $S > 85.3$ 就开始亏，$S=4096$ 时亏 3.1 倍 FLOPs。更隐蔽的是：吸收**不影响 cache 里存什么**，所以只看显存占用的 profiling 根本看不出这两种形态的区别 —— 必须在 kernel 级别看 attention 的 head dim 是 192 还是 576。

**坑 4：用小模型实验去定「GQA 该用几组」。**
本期实验 6 已经演示了：tiny 模型上，把 8 个 KV 头池化成 1 个（MQA），CE 从 0.0030 崩到 5.27 —— 但继续训练 250 步就回到 0.0038。**在训练预算充裕、任务简单的区间里，连 MQA 都能救回来**，这种实验对「组数甜点」这个问题完全没有区分度。定组数需要大模型 + 受限训练预算，或者直接抄已有大规模模型的收敛值（8）。

**坑 5（最容易踩的）：做「吸收等价性」验证时，key 数比 latent 维还少。**
本期的实验 4 第一版就踩了：用 24 个 key 位置对 256 维 latent 做最小二乘，线性系统欠定（24 个方程、256 个未知数），**任何目标都能被精确拟合**，于是得出「RoPE 不影响吸收」的错误结论（残差 0.0000）。改成 1024 个 key 位置（超定 4 倍）才得到真实的 0.7182。**凡是拿最小二乘残差当证据，先检查方程数是否大于未知数。**

---

## 8 一句话总结

> **MQA/GQA 是「少留几份」（压缩比必须是有理数 $n_h/n_g$），MLA 是「每份都变小」（压缩比连续可调）；而 MLA 能做到这一点的前提是解耦 RoPE 让矩阵吸收成立 —— 代价是 decode 的 attention 变宽 3.4 倍，换来的是省掉 120 倍的上投影。**

---

## 9 今日练习

<details><summary>今日练习（点击展开参考答案）</summary>

### 练习 1：换个模型，算它的 prefill 临界点

Llama-3-70B 是 GQA-8：$n_h = 64$、$n_g = 8$、$d_h = 128$、80 层。假设有人想给它也加一个 MLA 式的低秩瓶颈，取 $d_c = 256$、$d_h^R = 64$。

问：(a) 新的 per-token KV 是多少？相对 GQA-8 的压缩比？(b) 用 §4 的公式算 prefill 临界点 $S^*$；(c) 解释为什么 $S^*$ 比 DeepSeek-V3 的 85.3 小。

**参考答案：**

(a) 每层每 token = $d_c + d_h^R = 256 + 64 = 320$ 元素 = 640 B。GQA-8 是 $2 \times 8 \times 128 = 2048$ 元素 = 4096 B。压缩比 $2048/320 = 6.4$ 倍。（注意：这里是拿 MLA 去比 GQA 而不是比 MHA —— 因为原模型就是 GQA。）80 层全模型：$320 \times 80 \times 2 = 51{,}200$ B = 50 KiB/token。

(b) 注意 §4 的公式是用 $\{P, R, V, L_{kv}\}$ 表达的，其中 $P = V = d_h = 128$、$L_{kv} = 256$：

$$S^* = \frac{L_{kv} V}{2L_{kv} - P - V} = \frac{256 \times 128}{512 - 128 - 128} = \frac{32768}{256} = \mathbf{128}$$

(c) 因为 $S^*$ 的分母 $2L_{kv} - P - V$ 随 $L_{kv}$ 增大而增大，分子 $L_{kv}V$ 只是线性增长。$L_{kv}$ 从 512 降到 256 时，分母从 768 掉到 256（÷3），分子只掉一半 —— 净效果是 $S^*$ 从 85.3 升到 128。

**物理解释**：latent 越小，「展开」的每-token 上投影成本越低（正比于 $L_{kv}$），而「吸收」的宽 head 也越窄，两边的成本同时下降，但下降速率不同。$L_{kv}$ 越大，展开越不划算 —— 也就是说 **latent 越宽，越应该在 prefill 里展开**。这反直觉但符合公式：展开的代价是 $L_{kv}(P+V)$，吸收的额外代价是 $2L_{kv} - P - V$，前者的 $L_{kv}$ 系数是 256、后者是 2，所以 $L_{kv}$ 一大，展开先撑不住。

### 练习 2：算清「MLA 让跨机传输便宜了多少」

第 030 期给出的结论是：单请求搬 KV 与重算 prefill 的临界带宽，$S=4096$ 时是 **3.87 GB/s**（7B / 32 层 / GQA-8 / bf16，128 KiB/token）。

问：**模型其余部分完全不动**（还是 32 层、$d_{\text{model}}=4096$、$n_h=32$、$d_h=128$），只把 KV 侧换成 MLA（$d_c=512$、$d_h^R=64$）。这时同样的 $S=4096$ 临界带宽是多少？

**参考答案：**

因为除了 KV 的表示方式什么都没变，临界带宽只按 per-token KV 体积等比缩放（临界带宽 = 「搬完整个序列 KV 的字节数」÷「重算 prefill 的墙钟时间」，后者不变）。

先算清楚两边：

- GQA-8：$2 n_g d_h = 2 \times 8 \times 128 = 2048$ 元素/层/token → $32 \times 2048 \times 2 = 131{,}072$ B = 128 KiB/token
- MLA：$d_c + d_h^R = 512 + 64 = 576$ 元素/层/token → $32 \times 576 \times 2 = 36{,}864$ B = **36 KiB/token**

$$\text{临界带宽} = 3.87 \times \frac{36}{128} = 3.87 \times \frac{1}{3.556} = \mathbf{1.09 \ \text{GB/s}}$$

**量纲校验**：$[\text{GB/s}] \times [\text{KiB}]/[\text{KiB}] = \text{GB/s}$ ✓。层数在两边都一样（32），所以没有被层数差污染 —— 这正是本练习特意把模型固定住的原因。

**结论**：**一条 1 GB/s 的链路就能养活 MLA 的 PD 分离** —— 这是个极低的门槛（PCIe Gen4 x1 单向就有 ~2 GB/s）。030 期说「一条 30 Gbps（≈3.75 GB/s）的链路就能赢重算十几倍」，在 MLA 口径下这个余量要再乘 $128/36 = 3.56$ 倍。

（提醒：这是**按比例外推**，不是实测。真实数字还要看 prefill 重算那一边的实现效率。）

### 练习 3：找出下面这段 MLA 实现的 bug

```python
def mla_decode(q_nope, q_pe, kv_cache_latent, k_pe, w_uk, w_uv):
    """q_nope: (Sq, N, P)  q_pe: (Sq, N, R)
       kv_cache_latent: (Skv, Lkv)   k_pe: (Skv, R)"""
    ql = torch.einsum("snh,lnh->snl", q_nope, w_uk)      # (Sq, N, Lkv)
    q = torch.cat([ql, q_pe], dim=-1)                     # (Sq, N, Lkv+R)
    k = torch.cat([kv_cache_latent, k_pe], dim=-1)        # (Skv, Lkv+R)
    att = torch.einsum("snh,th->snt", q, k) / math.sqrt(P + R)
    att = att.softmax(-1)
    o = torch.einsum("snt,tl->snl", att, kv_cache_latent)  # (Sq, N, Lkv)
    return torch.einsum("snl,lnv->snv", o, w_uv)           # (Sq, N, V)
```

**参考答案：** 三处问题，一处比一处隐蔽。

**① 缺 causal mask，而且缺得很危险。**

在 decode 时 $S_q = 1$，单看这一步是对的。但这个函数**没有任何守卫**。一旦被复用去跑 extend / prefill（$S_q = S_{kv} = S$），它会静默地算出「每个 query 都能看到所有 key」—— **形状完全对得上，不报错，只是结果错了**。这是 MLA 实现里最经典的事故。

修法有两条：要么在函数开头断言 `Sq == 1`，要么按 vLLM 的做法把 prefill 和 decode 写成两条独立路径（形态 A / 形态 B），绝不共用一个函数。

**② `softmax_scale` 漏了 YaRN 的 `mscale` 修正。**

$\sqrt{P + R}$ 这个基数是对的（DeepSeek 参考实现里就是 `self.q_head_dim ** (-0.5)`，而 `q_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192`）。但参考实现的下一段是：

```python
self.softmax_scale = self.q_head_dim ** (-0.5)
if self.config.rope_scaling is not None:
    mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
    scaling_factor = self.config.rope_scaling["factor"]
    if mscale_all_dim:
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        self.softmax_scale = self.softmax_scale * mscale * mscale
```

而 `yarn_get_mscale` 是（源码原文）：

```python
def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
```

DeepSeek-V3 的 `config.json` 里 `rope_scaling.factor = 40`、`mscale_all_dim = 1.0`，所以

- `yarn_get_mscale(40, 1.0) = 0.1 × ln(40) + 1 = 1.36889`
- `mscale² = 1.87385`
- 真实的 `softmax_scale = 192^-0.5 × 1.87385 = 0.135234`，而不是 $192^{-0.5} = 0.072169$

**差了 87.4%。** 这个 bug 不会报错、不会 NaN，只会让 attention 分布整体变尖锐（或变平），训练/推理质量悄悄下滑。原因是 YaRN 把 RoPE 的旋转角度按 $\log$ 拉伸了，为了保持 score 的方差不变，必须把 scale 乘回去。

**③ `w_uk` 的布局约定 —— 静默的数值错误。**

`einsum("snh,lnh->snl", q_nope, w_uk)` 隐含 $W^{UK}$ 的形状是 $(L_{kv}, N, P)$，也就是**按 head 拆开存**。但 HuggingFace 参考实现里 `kv_b_proj` 是**一个** `nn.Linear(Lkv, N*(P+V))` —— 它的权重形状是 $(N(P+V), L_{kv})$，head 维和 $P/V$ 维是**混在一起**的。

要做吸收，你必须按 `[head, P+V]` 的次序把那个大矩阵切开再重排成 $(L_{kv}, N, P)$。**切错顺序（比如按 $[P, V]$ 先切、再分 head）形状照样对得上，结果全错。** 这就是为什么生产实现宁愿写一段带断言的权重加载代码，也不愿在 forward 里做重排。

**建议的自检**：把吸收形态和展开形态在**同一个 batch** 上跑一遍，断言两者的输出相对误差 $< 10^{-5}$。本期实验 3 做的就是这件事 —— 只要权重布局错了，这个断言一定会炸。

</details>

---

*本期所有实验均在 CPU 上真跑（torch 2.14.0），脚本见 `scratch-032/`。文中标注「来自源码/文档」的事实已逐处核对来源；标注「本期推断/按比例外推」的地方不可直接当作实测数据引用。*
