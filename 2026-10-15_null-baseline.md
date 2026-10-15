# PyTorch 每日一课 · 第 037 期

## 零假设基准：一个数字要减掉什么，才算「有信息」

**日期**：2026-10-15　**难度**：⭐⭐⭐⭐　**预计阅读时间**：25 分钟

**前置知识**：交叉熵与 softmax 的直觉；会读分位数和标准差；最好看过第 036 期那个「`1/S` 陷阱」。

**关联**：本文直接回答第 036 期留下的头号坑（拿 `1/S` 当「无 sink 基准」，会把每个正常模型都判成异常）；把第 026 期的 Gini 0.0797、第 033 期的「437 倍」、第 034/035 期的 recall 与硬指标全部放回它们各自的零假设坐标系。

---

## 0. 一句话

> **指标的读数减去它的零假设基线，剩下的那部分才是信息；而绝大多数指标的零假设不是「均匀分布」，是「产生这个指标的机制在没有任何信号时会自己走出来的那个分布」。**

---

## 1. 这个领域解决什么问题

任何一个「我们的方法把 X 从 A 提到了 B」的句子，都需要解释 A 是从哪来的。这件事看起来平凡，实际上是实验里最容易翻车的一环，因为它错了**不会报错**——模型照常训练，曲线照常下降，数字照常好看。

三种最常见的错误来源，按危害排序：

**第一种，拿 0 当基线。** 「专家负载不均衡度从 0.875 降到 0.0797」——0.0797 离 0 还很远，所以「还有优化空间」。但如果你真的去模拟一个**完全均匀**的路由器，你会发现它自己的 Gini 就是 0.0485（第 4 节实测）。也就是说 0.0797 只比「随机」高 0.03，这件事的结论应该反过来写。

**第二种，拿均匀分布当基线。** 这是第 036 期的头号坑：判断「位置 0 收到了多少注意力，是不是太多了」，用 $1/S$ 当基准，结果 $S=131072$ 时基准错了 **11.36 倍**。原因很简单——**因果掩码本身就给位置 0 一个结构性加成**，一个什么都没学的模型也会把不成比例的质量倒给前几个位置。均匀分布不是这个结构的零假设。

**第三种，拿另一个方法的读数当基线。** 「H2O 在预算 16 时的硬指标是 0.0312，Quest 是 0.3277，所以 Quest 好 10.5 倍」——这个比较本身没错，但如果没人告诉你 0.0312 恰好等于该任务 $1/V=1/32$ 的随机猜测率，你会以为 H2O「保住了一点信息」。**这两个数字只有在各自的零假设都摆出来之后才可比。**

本期的结构是六个可复现的案例，每个案例回答一类问题：

| 案例 | 指标类型 | 零假设从哪来 | 关键结论 |
|---|---|---|---|
| §3 因果掩码的位置权重 | 结构性偏置 | 对称性论证（解析闭式） | 基准随长度放大 4.48× → 11.36× |
| §4 MoE 负载不均衡度 | 有限样本波动 | 二项分布（解析 + 模拟） | 「均匀路由」自身 Gini = 0.0485 |
| §5 top-k 交叠 / recall | 抽样无放回 | 组合计数 $K^2/L$ | 噪声地板随输入相关性抬高 31.6× |
| §6 交叉熵 | 信息论下界 | 源的条件熵 $H(X_t\|X_{t-1})$ | 地板有三层，报错一层结论就反了 |
| §7 量化 nMSE | 率失真下界 | $2^{-2R}$（Shannon 下界） | 「好 437 倍」买的是别的 |
| §8 指标差 | 统计显著性 | 置换检验 | 同一对比在第 20 步 p<1e-4，第 300 步 p=0.57 |

---

## 2. 核心思想：零假设要由「机制」生成，不能由「均匀」生成

打个比方：你要判断一个人今天的体温 37.8°C 算不算发烧。你需要知道「正常体温是多少」。这个「正常」不是「所有可能温度的均匀分布」（那是 0~100°C 里随便一个数），而是**「同一个身体、在没有感染的情况下会走出来的那个温度」**。

零假设也一样：它必须是**被测机制在没有任何信号时的输出分布**。

按可靠程度，构造零假设有三个层级：

**层级一：解析闭式。** 机制足够简单时，直接算出「无信号」的期望值。比如因果掩码下位置 $j$ 的期望注意力权重（第 3 节），或者均匀路由下 Gini 的期望（第 4 节）。这一层最硬，因为它不需要任何模拟，而且能告诉你**这个基线与哪些量有关、与哪些无关**——后者往往才是真正的发现。

**层级二：置换 / 代理数据。** 机制复杂到算不出闭式时，把信号打乱后重跑同一个机制。第 8 节的置换检验就是这一层：把两组样本的标签随机重贴，得到「如果两个配置根本没区别，均值差会长什么样」。

**层级三：均匀假设。** 也就是「假设所有结果等概率」。**它是前两层都做不到时的兜底，而且只在机制本身没有结构时才成立。** 一旦存在掩码、采样、选择、截断、有限容量，均匀假设就会系统性地偏——而且偏向哪个方向通常是有规律的，这正是它危险的地方。

那么，怎么判断一个基线是不是错的？一个可以机械执行的检查清单：

1. **这个指标是怎么被「结构」影响的？** 有没有掩码、有没有不重复抽样、有没有 top-k 选择、有没有预算上限？如果有，均匀分布一定不是零假设。
2. **基线里有没有「有限样本」这一项？** 如果指标是一个比例的估计量，它的基线是 $O(1/\sqrt{n})$ 而不是 0。样本越小，基线越大。
3. **零假设和测量用的是同一个口径吗？** 全局平均和逐通道中位数对同一份数据可以给出**方向相反**的趋势（第 7 节实测）。
4. **这个指标有没有理论下界/上界？** 如果有，越过它就是确凿的 bug 信号（第 6 节）。

后面六节就是把这四条各跑一遍。

---

## 3. 案例一：因果掩码自带的结构基准

先回顾第 036 期的坑，然后把它推广——036 只算了位置 0，这一节算所有位置，并解释为什么这个偏置在长上下文里会放大。

### 3.1 推导

因果掩码下，query $t$ 只被允许看位置 $0..t$，共 $t+1$ 个。一个**什么都没学**的模型，它在允许位置上的分数是对称的，于是权重均分：

$$p_{t,j} = \frac{1}{t+1}\quad (0\le j\le t)$$

位置 $j$ 收到的平均权重（只统计排在它后面的 query，与第 036 期同口径）：

$$b(j) = \frac{1}{S-1-j}\sum_{t=j+1}^{S-1}\frac{1}{t+1} = \frac{H_S - H_{j+1}}{S-1-j},\qquad H_n=\sum_{i=1}^{n}\frac{1}{i}$$

$j=0$ 时就是 $b(0)=\dfrac{H_S-1}{S-1}\approx\dfrac{\ln S+\gamma-1}{S-1}$，而朴素基准是 $1/S$。两者的比值

$$\frac{b(j)}{1/S} = \frac{S\,(H_S-H_{j+1})}{S-1-j}$$

**这就是「结构自带的那一份」**——它和你用什么数据、训练多久、模型多大全都没关系。

### 3.2 实跑

```python
# E1：结构自带的基准 —— 因果掩码下位置 j 的「零假设注意力质量」b(j)
# 关键点：零假设不是均匀分布 1/S，而是「这个结构在没有任何信号时自己会产生的分布」
import math
import torch


def H(n):
    """调和数 H_n = 1 + 1/2 + ... + 1/n；倒序累加避免大 n 时的精度损失"""
    return sum(1.0 / i for i in range(n, 0, -1))


def b_structural(j, S):
    """位置 j 的零假设基准（与第 036 期同一口径）

    因果掩码下 query t 只被允许看位置 0..t。若模型「什么都没学」（分数在允许
    位置上对称），权重就均分成 1/(t+1)。能看见位置 j 的 query 是 t = j+1 .. S-1
    （排除 j 自己那条），对它们取平均即得闭式。
    """
    return (H(S) - H(j + 1)) / (S - 1 - j)


print("【E1.1】S=129，逐位置的零假设基准 b(j) 与朴素基准 1/S")
print()
print(f"{'j':>5} {'b(j)':>12} {'1/S':>12} {'b(j)*S':>9} {'渐近形 ln(S/(j+1))/(S-1-j)':>28}")
S = 129
for j in [0, 1, 2, 4, 8, 16, 32, 64, 96, 127]:
    b = b_structural(j, S)
    asym = math.log(S / (j + 1)) / (S - 1 - j)
    print(f"{j:>5} {b:>12.6f} {1.0/S:>12.6f} {b*S:>9.2f} {asym:>28.6f}")

print()
print("【E1.2】固定 j=0 换长度：结构加成随 S 单调放大（长上下文更容易被误判的机制）")
print()
print(f"{'S':>9} {'1/S':>13} {'b(0)':>13} {'b(0)/(1/S)':>11}")
for S in [129, 1024, 4096, 32768, 131072]:
    print(f"{S:>9} {1.0/S:>13.3e} {b_structural(0, S):>13.6f} {b_structural(0, S)*S:>11.3f}")

print()
print("【E1.3】用真 torch 的 causal softmax 验证 b(j)（零分数 = 最极端的「无信号」）")
S = 12
causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
scores = torch.zeros(S, S).masked_fill(~causal, float("-inf"))
w = torch.softmax(scores, dim=-1)          # 每行恰好 1/(t+1)
print()
print(f"  query 0 只能看位置 0，权重 = {w[0, 0].item():.6f}")
print(f"  query {S-1} 看全部 {S} 个位置，每个权重 = {w[-1, 0].item():.8f}（= 1/{S} = {1/S:.8f}）")
print()
print(f"{'j':>3} {'实跑 b(j)':>14} {'解析 b(j)':>14} {'差值':>12}")
for j in range(S - 1):
    emp = w[j + 1:, j].mean().item()
    ana = b_structural(j, S)
    print(f"{j:>3} {emp:>14.8f} {ana:>14.8f} {abs(emp-ana):>12.2e}")

print()
print("【E1.4】b(j) 是对称性论证、不是「零分数特例」：随机 QK 也收敛到同一个数")
S, D, TRIALS = 129, 64, 200
m = torch.tril(torch.ones(S, S, dtype=torch.bool))
torch.manual_seed(0)
acc = torch.zeros(S)
for _ in range(TRIALS):
    q, k = torch.randn(1, 1, S, D), torch.randn(1, 1, S, D)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    p = torch.softmax(s.masked_fill(~m, float("-inf")), dim=-1)[0, 0]
    acc += p[:, 0]                    # 每次试验：各 query 分给位置 0 的权重
recv = acc / TRIALS                   # 对试验取平均 → 每个 query 的期望权重 ≈ 1/(t+1)
emp0 = recv[1:].mean().item()
print()
print(f"  {TRIALS} 次随机 QK，位置 0 收到的平均权重 = {emp0:.6f}")
print(f"  解析 b(0)                             = {b_structural(0, S):.6f}")
print(f"  朴素基准 1/S                           = {1.0/S:.6f}")
print(f"  相对解析基准偏差                        = {abs(emp0/b_structural(0, S) - 1)*100:.2f}%")
print(f"  相对朴素基准偏差                        = {abs(emp0*S - 1)*100:.2f}%")
```

```text
【E1.1】S=129，逐位置的零假设基准 b(j) 与朴素基准 1/S

    j         b(j)          1/S    b(j)*S      渐近形 ln(S/(j+1))/(S-1-j)
    0     0.034695     0.007752      4.48                     0.037967
    1     0.031031     0.007752      4.00                     0.032808
    2     0.028631     0.007752      3.69                     0.029851
    4     0.025464     0.007752      3.28                     0.026213
    8     0.021766     0.007752      2.81                     0.022188
   16     0.017869     0.007752      2.31                     0.018095
   32     0.014084     0.007752      1.82                     0.014201
   64     0.010650     0.007752      1.37                     0.010710
   96     0.008870     0.007752      1.14                     0.008909
  127     0.007752     0.007752      1.00                     0.007782

【E1.2】固定 j=0 换长度：结构加成随 S 单调放大（长上下文更容易被误判的机制）

        S           1/S          b(0)  b(0)/(1/S)
      129     7.752e-03      0.034695       4.476
     1024     9.766e-04      0.006363       6.516
     4096     2.441e-04      0.001928       7.897
    32768     3.052e-05      0.000304       9.975
   131072     7.629e-06      0.000087      11.361

【E1.3】用真 torch 的 causal softmax 验证 b(j)（零分数 = 最极端的「无信号」）

  query 0 只能看位置 0，权重 = 1.000000
  query 11 看全部 12 个位置，每个权重 = 0.08333334（= 1/12 = 0.08333333）

  j        实跑 b(j)        解析 b(j)           差值
  0     0.19120097     0.19120097     8.57e-10
  1     0.16032106     0.16032107     1.10e-08
  2     0.14109749     0.14109748     3.25e-09
  3     0.12748468     0.12748467     1.11e-08
  4     0.11712534     0.11712533     4.82e-09
  5     0.10886845     0.10886845     3.56e-09
  6     0.10207071     0.10207071     4.48e-09
  7     0.09633838     0.09633838     1.51e-11
  8     0.09141415     0.09141414     4.71e-09
  9     0.08712122     0.08712121     6.32e-09
 10     0.08333334     0.08333333     2.48e-09

【E1.4】b(j) 是对称性论证、不是「零分数特例」：随机 QK 也收敛到同一个数

  200 次随机 QK，位置 0 收到的平均权重 = 0.035237
  解析 b(0)                             = 0.034695
  朴素基准 1/S                           = 0.007752
  相对解析基准偏差                        = 1.56%
  相对朴素基准偏差                        = 354.56%
```

### 3.3 读结果

三件事：

**第一，结构加成不是位置 0 的特权，而是沿位置单调衰减的一条曲线。** $S=129$ 时 $j=0$ 的加成是 4.48×，$j=8$ 降到 2.81×，$j=64$ 只剩 1.37×，到 $j=127$ 恰好 1.00×（位置 127 只有它自己那条 query，没有可比较的对象）。**这解释了第 036 期对照组里「位置 64 有 +12.4%、位置 96 有 +9.2%」为什么存在——它们也在结构加成区里**，只是弱得多。

**第二，加成随长度放大。** $j=0$ 处 $S$ 从 129 涨到 131072，加成从 4.476 涨到 11.361，正好是 $\ln S + \gamma - 1$ 的量级。**所以一个「在 128 长度上看起来还行的基准」，搬到 128K 上会把误差放大 2.5 倍。** 这条对长上下文实验特别重要：基准错误不是常数偏置，它是随 $S$ 增长的。

**第三，$b(j)$ 是对称性结论，不是「零分数」这个特例的产物。** E1.4 用**随机**的 QK（$\sigma$ 全随机）做了 200 次蒙特卡洛，位置 0 的平均接收权重是 0.035237，与解析值 0.034695 差 1.56%——而离朴素基准 $1/S$ 差了 **354.56%**。机制很简单：分数在允许位置上**可交换**时，期望权重就必然是均匀的，无论分数的分布多尖锐。

> **可迁移的判断**：只要你的模型有一个「位置的对称性」，就一定能写出对应的解析基线。因果掩码是最常见的，但滑动窗口、块稀疏、局部+全局的混合掩码、甚至 packed sequence 的分段边界，都是同一形状。

---

## 4. 案例二：MoE 负载不均衡度 —— 「均匀」本身就该有个非零读数

第 026 期实测：辅助损失系数 0.1 时专家负载的 Gini 是 0.0797，不做平衡是 0.8750，max/avg 是 3.26。当时的结论是「0.1 最优」。

但 0.0797 该跟谁比？**跟 0 比会得出「还有优化空间」，跟「均匀路由自己的 Gini」比才是正解。**

### 4.1 解析

均匀路由下，专家 $i$ 收到的 token 数 $L_i \sim \mathrm{Binomial}(TK,\,1/E)$（$T$ 个 token、每个选 $K$ 个专家、$E$ 个专家）。于是

$$\mu = \frac{TK}{E},\qquad \sigma=\sqrt{\mu\left(1-\frac{1}{E}\right)}$$

对近正态的计数，$\mathbb{E}|X-X'|=2\sigma/\sqrt{\pi}$，代入 Gini 的定义 $G=\mathbb{E}|X-X'|/(2\mu)$：

$$G_{\text{null}} = \frac{\sigma}{\mu\sqrt{\pi}} = \sqrt{\frac{1-1/E}{\pi\,\mu}}$$

同时 $\mathbb{E}[\max_i L_i]\approx \mu + c_E\sigma$（$E=64$ 时 $c_E\approx 2.5$），于是 $\max/\text{avg} \approx 1+c_E\sigma/\mu$。

**两个关键点**：$G_{\text{null}}\propto 1/\sqrt{TK}$ —— **样本越小基线越大**；而且它与路由器的好坏无关，纯粹是多项式抽样的涨落。

### 4.2 实跑

```python
# E2：MoE 路由「负载不均衡」的零假设 —— 均匀路由本身就有 Gini，且随 batch 缩小而放大
# 对照第 026 期：aux=0.1 实测 Gini 0.0797 / 不均衡实测 0.8750 / max/avg 实测 3.26
import math
import torch


def gini(load):
    """load 是一个长度为 E 的计数向量；G = ΣΣ|x_i-x_j| / (2 E² μ)"""
    x = load.float()
    n = x.numel()
    mu = x.mean()
    d = (x[:, None] - x[None, :]).abs().sum()
    return (d / (2 * n * n * mu)).item()


def route_uniform(T, E, K, gen):
    """均匀随机路由：每个 token 从 E 个专家里不重复地挑 K 个"""
    return torch.argsort(torch.rand(T, E, generator=gen), dim=-1)[:, :K]


def route_biased(T, E, K, alpha, gen):
    """带偏好的路由：专家 i 的先验权重 ∝ (i+1)^-alpha，alpha=0 即均匀

    用 Gumbel top-K 实现「按概率不重复抽 K 个」（Perturbed-Max 技巧）。
    注意是取 log w + g 的 **前 K 大**，写成 argsort 默认取前 K 小就变成
    「专挑最不被偏好的专家」，而且会把 alpha 完全吃掉（本期踩到过）。
    """
    w = torch.arange(1, E + 1, dtype=torch.float) ** (-alpha)
    g = -torch.log(-torch.log(torch.rand(T, E, generator=gen).clamp_min(1e-12)))
    return torch.topk(torch.log(w) + g, K, dim=-1).indices


E, K, SEEDS = 64, 2, 5
_MU = 4096 * K / E                                        # T=4096 时每个专家的期望负载
_SIGMA = math.sqrt(_MU * (1 - 1 / E))                     # 二项分布标准差
GNULL = _SIGMA / (_MU * math.sqrt(math.pi))               # 正态近似下的零假设 Gini
print("【E2.1】均匀随机路由的零假设分布（E=64, K=2, 5 个种子）")
print()
print(f"{'T':>8} {'μ=TK/E':>9} {'Gini 均值':>11} {'Gini 区间':>21} {'max/avg 均值':>13} {'解析 Gini ≈ σ/(μ√π)':>20}")
for T in [256, 1024, 4096, 16384]:
    gs, ms = [], []
    for s in range(SEEDS):
        gen = torch.Generator().manual_seed(1000 + s)
        idx = route_uniform(T, E, K, gen)
        load = torch.zeros(E, dtype=torch.long)
        load.scatter_add_(0, idx.flatten(), torch.ones(idx.numel(), dtype=torch.long))
        gs.append(gini(load))
        ms.append(load.max().item() / load.float().mean().item())
    mu = T * K / E
    sigma = math.sqrt(mu * (1 - 1 / E))
    g_ana = sigma / (mu * math.sqrt(math.pi))
    print(f"{T:>8} {mu:>9.0f} {sum(gs)/len(gs):>11.4f} "
          f"[{min(gs):.4f}, {max(gs):.4f}]".rjust(22) + f" {sum(ms)/len(ms):>13.4f} {g_ana:>20.4f}")

print()
print("【E2.2】同一个「均衡」指标下，有偏路由器与零假设的对照（T=4096, 5 种子）")
print()
print(f"{'alpha':>7} {'Gini 均值':>11} {'Gini 区间':>21} {'Gini/零假设':>12} "
      f"{'max/avg':>10} {'最小专家占比':>13}")
for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]:
    gs, ms, mins = [], [], []
    for s in range(SEEDS):
        gen = torch.Generator().manual_seed(2000 + s)
        idx = route_biased(4096, E, K, alpha, gen)
        load = torch.zeros(E, dtype=torch.long).scatter_add_(
            0, idx.flatten(), torch.ones(idx.numel(), dtype=torch.long))
        gs.append(gini(load))
        ms.append(load.max().item() / load.float().mean().item())
        mins.append(load.min().item() / load.float().mean().item())
    g = sum(gs) / len(gs)
    print(f"{alpha:>7.2f} {g:>11.4f} [{min(gs):.4f}, {max(gs):.4f}]".rjust(33)
          + f" {g/GNULL:>12.2f} {sum(ms)/len(ms):>10.4f} {sum(mins)/len(mins):>13.4f}")

print()
print("【E2.3】把第 026 期的实测数字放回零假设坐标系")
print()
T, E_, K_ = 4096, 64, 2
mu = T * K_ / E_
sigma = math.sqrt(mu * (1 - 1 / E_))
print(f"  零假设（均匀路由）解析 Gini      ≈ {sigma/(mu*math.sqrt(math.pi)):.4f}")
print(f"  零假设（均匀路由）解析 max/avg   ≈ {(mu+2.5*sigma)/mu:.4f}   （E=64 时 E[max]≈μ+2.5σ）")
print(f"  第 026 期 aux=0.1 实测 Gini       = 0.0797   → 超额 {0.0797-sigma/(mu*math.sqrt(math.pi)):.4f}")
print(f"  第 026 期 不均衡实测 Gini         = 0.8750   → 超额 {0.8750-sigma/(mu*math.sqrt(math.pi)):.4f}")
print(f"  第 026 期 不均衡实测 max/avg      = 3.26     → 解析零假设 {((mu+2.5*sigma)/mu):.4f}")
print()
print(f"  归一化不均衡度 (G - G_null)/(1 - G_null)：")
gnull = sigma / (mu * math.sqrt(math.pi))
print(f"    aux=0.1   → {(0.0797-gnull)/(1-gnull):.4f}")
print(f"    不均衡    → {(0.8750-gnull)/(1-gnull):.4f}")
```

```text
【E2.1】均匀随机路由的零假设分布（E=64, K=2, 5 个种子）

       T    μ=TK/E     Gini 均值               Gini 区间    max/avg 均值    解析 Gini ≈ σ/(μ√π)
     256         8      0.1883 [0.1754, 0.1984]        1.8500               0.1979
    1024        32      0.0956 [0.0890, 0.1093]        1.3562               0.0990
    4096       128      0.0485 [0.0465, 0.0516]        1.1938               0.0495
   16384       512      0.0246 [0.0232, 0.0259]        1.1074               0.0247

【E2.2】同一个「均衡」指标下，有偏路由器与零假设的对照（T=4096, 5 种子）

  alpha     Gini 均值               Gini 区间     Gini/零假设    max/avg        最小专家占比
   0.00      0.0478 [0.0457, 0.0516]         0.97     1.1938        0.7734
   0.05      0.0506 [0.0476, 0.0552]         1.02     1.2203        0.7844
   0.10      0.0646 [0.0604, 0.0688]         1.31     1.3578        0.7547
   0.15      0.0830 [0.0771, 0.0867]         1.68     1.5797        0.7312
   0.20      0.1073 [0.1029, 0.1104]         2.17     1.8359        0.6937
   0.30      0.1579 [0.1539, 0.1611]         3.19     2.4594        0.6078
   0.50      0.2738 [0.2692, 0.2779]         5.53     4.1688        0.4547
   1.00      0.5857 [0.5793, 0.5927]        11.84    12.2641        0.1656
   2.00      0.8974 [0.8963, 0.8984]        18.14    27.9828        0.0031

【E2.3】把第 026 期的实测数字放回零假设坐标系

  零假设（均匀路由）解析 Gini      ≈ 0.0495
  零假设（均匀路由）解析 max/avg   ≈ 1.2192   （E=64 时 E[max]≈μ+2.5σ）
  第 026 期 aux=0.1 实测 Gini       = 0.0797   → 超额 0.0302
  第 026 期 不均衡实测 Gini         = 0.8750   → 超额 0.8255
  第 026 期 不均衡实测 max/avg      = 3.26     → 解析零假设 1.2192

  归一化不均衡度 (G - G_null)/(1 - G_null)：
    aux=0.1   → 0.0318
    不均衡    → 0.8685
```

### 4.3 读结果

**零假设 Gini 是 0.0485（解析 0.0495），而不是 0。** 于是第 026 期的两个数字获得了完全不同的地位：

- 「不均衡」的 0.8750 离零假设有 **0.8255** 的超额，归一化不均衡度 0.8685 —— 这是真问题；
- 「aux=0.1」的 0.0797 只比零假设高 **0.0302**，归一化后仅 0.0318。也就是说辅助损失已经把路由器的偏好压到几乎检测不到，**剩下的那部分读数主要是采样噪声**。

**第二个发现是尺度效应：零假设随 $T$ 反向缩放。** $T$ 从 16384 降到 256，$\mu$ 从 512 降到 8，零假设 Gini 从 0.0246 涨到 **0.1883**——涨了 7.7 倍。**这意味着在小 batch 上做 A/B 对比会系统性看到「更不均衡」**，而那纯粹是 $\sigma/\mu \propto 1/\sqrt{TK}$。任何在 batch 64 / 128 上做的 MoE 负载对比，如果不带零假设，结论都不可信。

**第三个发现是这条曲线的可解释性。** E2.2 扫了偏好指数 $\alpha$：Gini 从 0.0478（$\alpha=0$，即零假设）单调升到 0.8974（$\alpha=2$）。而**第 026 期 aux=0.1 的 0.0797 落在 $\alpha \approx 0.14$ 的位置**（在 0.1 与 0.15 两档之间按线性插值）——你可以把「一个训练好的平衡路由器」翻译成「等价的先验偏好指数」，这个翻译比「Gini 0.0797」直观得多。

顺带一个实测的自我修正：第 026 期记的 max/avg = 3.26，而零假设的解析值是 **1.2192**。这个数字当时也没有基线可比。

> **可迁移的判断**：任何「不均衡度 / 熵 / 集中度」类指标，只要它是在有限样本上从一个选择过程里数出来的，基线就一定是 $O(1/\sqrt{n})$ 而不是 0。**先把基线算出来，再决定那个数字是大是小。**

---

## 5. 案例三：集合类指标的噪声地板

第 034 期量了稀疏注意力的 recall（0.9346），第 035 期量了 H2O 的召回。这两类指标都是**集合操作**，而集合操作有一个非常容易漏的性质：**两个「都没有真信号」的选择，本来就会重叠**。

### 5.1 解析

两个独立均匀随机的 $K$ 元子集（从 $L$ 个位置里选），交叠大小的期望是

$$\mathbb{E}\left[|S_1\cap S_2|\right] = \frac{K\cdot K}{L} = \frac{K^2}{L}$$

因为 $S_1$ 里每个元素落在 $S_2$ 里的概率都是 $K/L$。

另一条更常用的：**注意力质量型 recall 的随机基线恒等于 $K/L$，与分布多尖锐完全无关。**

$$\mathbb{E}_{S\sim\mathrm{Unif}\binom{[L]}{K}}\Big[\sum_{j\in S}p_j\Big] = \frac{K}{L}\sum_j p_j = \frac{K}{L}$$

最后一步用了 $\sum_j p_j=1$。**这条等式看着平凡，但它戳破了一个很常见的直觉**：「这个头的注意力很集中，所以随机选也能蒙到重要的位置」——不成立，随机选蒙到的质量份额永远是 $K/L$，跟集中不集中一点关系都没有。

### 5.2 实跑

```python
# E3：集合类指标的噪声地板 —— top-k 交叠与 recall 的随机基线不是 0
import math
import torch

L, K, TRIALS = 2048, 64, 2000
print("【E3.1】两次「相互独立、都没有真信号」的 top-k 选择，期望交叠 = K²/L")
print()
print(f"  L={L}, K={K} → 解析期望交叠 = K²/L = {K*K/L:.4f}")
torch.manual_seed(0)
ov = []
for _ in range(TRIALS):
    s1, s2 = torch.randn(L), torch.randn(L)
    ov.append(len(set(torch.topk(s1, K).indices.tolist()) &
                  set(torch.topk(s2, K).indices.tolist())))
t = torch.tensor(ov, dtype=torch.float)
print(f"  {TRIALS} 次实跑：均值 {t.mean():.4f} / 标准差 {t.std():.4f} / "
      f"p5 {torch.quantile(t,0.05):.0f} / p95 {torch.quantile(t,0.95):.0f}")
q = torch.bincount(torch.tensor(ov), minlength=8)[:8]
print(f"  交叠个数分布 0..7: {q.tolist()}")
print(f"  交叠为 0 的比例: {(t==0).float().mean()*100:.1f}%")
print(f"  把每次交叠归一成 IoU（交叠/K）: 均值 {t.mean()/K*100:.2f}%")

print()
print("【E3.2】同一个「无真信号」场景，只改两路输入的相关系数 ρ —— 噪声地板跟着抬")
print()
print(f"{'ρ':>6} {'实测交叠':>10} {'交叠/K':>9} {'相对 ρ=0 抬高':>14}")
res = {}
for rho in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
    torch.manual_seed(1)
    acc = 0
    for _ in range(400):
        c, n1, n2 = torch.randn(L), torch.randn(L), torch.randn(L)
        s1 = math.sqrt(rho) * c + math.sqrt(1 - rho) * n1
        s2 = math.sqrt(rho) * c + math.sqrt(1 - rho) * n2
        acc += len(set(torch.topk(s1, K).indices.tolist()) &
                   set(torch.topk(s2, K).indices.tolist()))
    m = acc / 400
    res[rho] = m
    print(f"{rho:>6.1f} {m:>10.2f} {m/K*100:>8.2f}% {m/res[0.0]:>13.2f}x")

print()
print("【E3.3】注意力质量型 recall 的随机基线恒为 K/L，与分布多尖锐无关")
print()
print(f"{'分布':>16} {'top-K(1%预算)':>14} {'随机 K 个':>11} {'K/L':>8}")
for name, p in [
    ("均匀", torch.full((L,), 1.0 / L)),
    ("Zipf s=1", 1.0 / torch.arange(1, L + 1, dtype=torch.float)),
    ("一行 softmax/√d", torch.softmax(torch.randn(L) * 8.0, dim=0)),
]:
    p = p / p.sum()
    top = torch.topk(p, K).values.sum().item()
    torch.manual_seed(7)
    rnd = []
    for _ in range(500):
        idx = torch.randperm(L)[:K]
        rnd.append(p[idx].sum().item())
    print(f"{name:>16} {top:>14.4f} {sum(rnd)/len(rnd):>11.4f} {K/L:>8.4f}")

print()
print("【E3.4】把第 034 期的 recall 放进这个坐标系（K=256, L=2048 的口径）")
print()
K34, L34 = 256, 2048
print(f"  随机选择 K 个 token 的期望 recall = K/L = {K34/L34:.4f}")
print(f"  第 034 期 per-head oracle recall           = 0.9346 → {0.9346/(K34/L34):.2f}x 随机")
print(f"  第 034 期 head 间共享选择                  = 0.9149 → {0.9149/(K34/L34):.2f}x 随机")
print(f"  第 034 期块粒度 b=16 的 recall             = 0.8598 → {0.8598/(K34/L34):.2f}x 随机")
print()
print("  注意：0.9149 与 0.9346 只差 2.0pt，但两者都远高于 12.5% 的随机地板 ——")
print("  「比随机好 7 倍」这件事本身不构成任何结论，有意义的是那 2.0pt。")
```

```text
【E3.1】两次「相互独立、都没有真信号」的 top-k 选择，期望交叠 = K²/L

  L=2048, K=64 → 解析期望交叠 = K²/L = 2.0000
  2000 次实跑：均值 2.0195 / 标准差 1.3936 / p5 0 / p95 5
  交叠个数分布 0..7: [256, 528, 562, 378, 171, 69, 28, 8]
  交叠为 0 的比例: 12.8%
  把每次交叠归一成 IoU（交叠/K）: 均值 3.16%

【E3.2】同一个「无真信号」场景，只改两路输入的相关系数 ρ —— 噪声地板跟着抬

     ρ       实测交叠      交叠/K      相对 ρ=0 抬高
   0.0       2.02     3.16%          1.00x
   0.1       3.28     5.13%          1.62x
   0.2       4.90     7.66%          2.42x
   0.3       7.03    10.99%          3.47x
   0.5      12.98    20.28%          6.41x
   0.7      22.27    34.80%         11.00x
   0.9      38.66    60.41%         19.09x
   1.0      64.00   100.00%         31.60x

【E3.3】注意力质量型 recall 的随机基线恒为 K/L，与分布多尖锐无关

              分布    top-K(1%预算)      随机 K 个      K/L
              均匀         0.0312      0.0312   0.0312
        Zipf s=1         0.5784      0.0311   0.0312
   一行 softmax/√d         0.9991      0.0300   0.0312

【E3.4】把第 034 期的 recall 放进这个坐标系（K=256, L=2048 的口径）

  随机选择 K 个 token 的期望 recall = K/L = 0.1250
  第 034 期 per-head oracle recall           = 0.9346 → 7.48x 随机
  第 034 期 head 间共享选择                  = 0.9149 → 7.32x 随机
  第 034 期块粒度 b=16 的 recall             = 0.8598 → 6.88x 随机

  注意：0.9149 与 0.9346 只差 2.0pt，但两者都远高于 12.5% 的随机地板 ——
  「比随机好 7 倍」这件事本身不构成任何结论，有意义的是那 2.0pt。
```

### 5.3 读结果

**第一，随机基线是 $K^2/L=2.0000$，实测 2.0195，12.8% 的情况下甚至交叠为 0。** 所以「两个模型的 top-64 注意力位置交叠了 8 个，看起来很像」这句话：交叠 8 个在噪声分布里是 p95 附近，算是**略高**，但如果你拿 0 当基线，就会把它读成「高度一致」。归一化之后是 IoU 3.16%。

**第二，也是这一节最有用的一条：噪声地板不是常数，它随两路输入的相关系数抬高。** E3.2 里两路信号**都没有任何真信号**，只是共享了一个强度为 $\rho$ 的公共分量：$\rho=0$ 时交叠 2.02，$\rho=0.3$ 时 7.03，$\rho=0.5$ 时 12.98，$\rho=1$ 时 64.00。**同一个「无信号」场景，读数从 3.16% 涨到 100%。**

这对工程实践有直接影响：比较**同一个模型不同层**的注意力选择、或者**同一个 checkpoint 相邻两步**的选择时，输入天然高度相关，噪声地板被抬得很高。而比较**两个独立训练的模型**时地板才是 2。**不声明「对着谁比」，交叠率这个数字没有意义。**

**第三，随机 recall 恒为 $K/L$，与分布形状无关（E3.3）。**均匀分布下 top-K 和随机选一模一样（0.0312 = 0.0312），Zipf 分布下 top-K 高达 0.5784 而随机仍是 0.0311，一行 $\sqrt{d}$ 缩放的尖锐 softmax 下 top-K 达到 0.9991 而随机降到 0.0300。**分布越尖锐，只说明「有更好的选择策略可用」，不说明「随机选择变好了」。**

把第 034 期的数字放进来：$K/L = 0.1250$，oracle recall 0.9346 是随机的 7.48 倍。看起来是个巨大的胜利，但**这个 7.48 倍里有多少来自「任务本身可预测」，有多少来自「选择策略有效」，必须再和第二个基线比才会有答案**——那就是「滑动窗口」这类不看内容的策略。034 期量了：窗口 recall 恰好等于局部质量占比。**只有内容和窗口的差，才是选择策略的净贡献。**

---

## 6. 案例四：信息论地板 —— CE 有多少层

交叉熵是最容易被误读的指标，因为它**看起来**有一个显然的地板（0）和一个显然的基线（$\ln V$）。实际上它有**三层**，用错任何一层，结论的方向都可能反。

### 6.1 三层地板

| 层级 | 值 | 含义 | 误用后果 |
|---|---|---|---|
| (a) 均匀猜测 | $\ln V$ | 所有 token 等概率 | 把「已接近最优」说成「还有 30% 可降」 |
| (b) 语料频率 | $H(\pi)$ | 只知道 token 频率、不知道上下文 | 忽略任务本身的不平衡 |
| (c) 任务条件熵 | $H(X_t\mid X_{t-1})$ | 真源给的下界 | 把「模型比基线好多少」抹掉 |

**(a) 是一个真正的下界，但它不是随机初始化的读数。** 由 Jensen 不等式，对**任何**随机 logits $z$ 和均匀标签 $y$：

$$\mathbb{E}[-\log p_y] = \mathbb{E}[-\log \mathrm{softmax}(z)_y] \ge -\log \mathbb{E}[p_y] = \ln V$$

等号只在完美均匀（$z$ 全相等）时取到。所以**实测的随机初始化 CE 必然 ≥ $\ln V$，而且高出的量可以用 logits 的方差精确预言**：对 iid $\mathcal{N}(0,s^2)$ 的 logits，$\mathbb{E}[\mathrm{LSE}]\approx \ln V + s^2/2$。

**(c) 才是真正的下界。** 如果任务本身是随机的（给定上下文，下一个 token 仍服从某个分布），那**任何**模型都不可能低于 $H(\cdot\mid\cdot)$。造一个条件熵可精确计算的源，就能验证这一点。

### 6.2 实跑

```python
# E4：信息论地板 —— 一个 CE 数字只有在知道它的地板之后才有意义
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

V, CTX, D, NH, NL = 32, 64, 64, 4, 2
STEPS = 1501


def make_chain(seed=0):
    g = torch.Generator().manual_seed(seed)
    P = torch.softmax(torch.randn(V, V, generator=g) * 1.5, dim=-1)
    pi = torch.linalg.matrix_power(P.T, 300)[:, 0]
    pi = pi / pi.sum()
    Hc = -(pi[:, None] * P.clamp_min(1e-30).log() * P).sum().item()
    Hpi = -(pi.clamp_min(1e-30).log() * pi).sum().item()
    return P, pi, Hc, Hpi


P, pi, Hc, Hpi = make_chain(0)
print("【E4.1】同一份实验里其实有三层地板")
print()
print(f"  (a) 均匀猜测地板 ln V          = {math.log(V):.4f} nats")
print(f"  (b) 只看 token 频率的地板 H(π)  = {Hpi:.4f} nats   ← 语料平稳分布的熵")
print(f"  (c) 给定上一个 token 的地板     = {Hc:.4f} nats   ← 这个任务真正的下界")
print()
print("  只报 (a) 会把「已经接近最优」的模型说成「还有 30% 可降」；")
print("  只报 (c) 会把「模型比基线好多少」这件事抹掉。三个都要报。")


def sample_stream(P, n_seq, n_tok, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(V, (n_seq,), generator=g)
    out = [x]
    for _ in range(n_tok - 1):
        x = torch.multinomial(P[x], 1, generator=g).squeeze(-1)   # 漏掉这个赋值就退化成「所有位置都从 x_0 采」
        out.append(x)
    return torch.stack(out, dim=1)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.te, self.pe = nn.Embedding(V, D), nn.Embedding(CTX, D)
        layer = nn.TransformerEncoderLayer(
            D, NH, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, NL, enable_nested_tensor=False)
        self.head = nn.Linear(D, V)
        self.register_buffer("mask",
                             torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), 1))

    def forward(self, x, causal=True):
        h = self.te(x) + self.pe(torch.arange(x.shape[1]))
        return self.head(self.tr(h, mask=(self.mask if causal else None)))


tr = sample_stream(P, 32768, CTX + 1, seed=0)
trx, trY = tr[:, :-1], tr[:, 1:]
ho = sample_stream(P, 4096, CTX + 1, seed=999)
hox, hoY = ho[:, :-1], ho[:, 1:]
orc = (-P[hox.reshape(-1), hoY.reshape(-1)].clamp_min(1e-30).log().mean()).item()
print()
print(f"  真 P 的 oracle CE（给定上一 token）= {orc:.4f}   ← 与解析 Hc {Hc:.4f} 吻合，确认数据生成正确")

print()
print("【E4.2】随机初始化的 CE 一定 ≥ ln V（Jensen：E[-log p] ≥ -log E[p] = ln V）")
print()
print(f"{'logit 标准差 s':>15} {'实测 CE':>10} {'ln V + s²/2':>13} {'超出 ln V':>10}")
for s in [0.0, 0.5, 1.0, 2.0, 4.0]:
    torch.manual_seed(0)
    z = torch.randn(200000, V) * s
    y = torch.randint(0, V, (200000,))
    ce = F.cross_entropy(z, y).item()
    print(f"{s:>15.1f} {ce:>10.4f} {math.log(V)+s*s/2:>13.4f} {ce-math.log(V):>10.4f}")
print("  （s=4 处偏离 ln V + s²/2 是有限 V 的尾部效应，该式只在 s² ≪ ln V 时成立）")
torch.manual_seed(0)
m = TinyLM()
with torch.no_grad():
    z = m(trx[:256])
init_ce = F.cross_entropy(z.reshape(-1, V), trY[:256].reshape(-1)).item()
print()
print(f"  实测随机初始化 tiny LM 的 logit 标准差 = {z.std().item():.4f}")
print(f"  实测随机初始化的 CE = {init_ce:.4f}，ln V = {math.log(V):.4f} "
      f"→ 高出 {(init_ce/math.log(V)-1)*100:.1f}%")
print(f"  按 ln V + s²/2 预言 = {math.log(V)+z.std().item()**2/2:.4f}"
      f"（与实测差 {abs(math.log(V)+z.std().item()**2/2-init_ce)/init_ce*100:.1f}%）")

print()
print("【E4.3】训练到地板：holdout CE 收敛到 Hc，而不是收敛到 0")
print()
torch.manual_seed(0)
m = TinyLM()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
gen = torch.Generator().manual_seed(123)


def evalce(model, x, y, causal=True, n=256, seed=7):
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, x.shape[0], (n,), generator=g)
    with torch.no_grad():
        return F.cross_entropy(model(x[i], causal).reshape(-1, V), y[i].reshape(-1)).item()


print(f"{'step':>6} {'train CE':>10} {'holdout CE':>12} {'holdout 超出地板':>16}")
for step in range(STEPS):
    i = torch.randint(0, trx.shape[0], (64,), generator=gen)
    loss = F.cross_entropy(m(trx[i]).reshape(-1, V), trY[i].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 250 == 0:
        te, he = evalce(m, trx, trY), evalce(m, hox, hoY)
        print(f"{step:>6} {te:>10.4f} {he:>12.4f} {he-Hc:>16.4f}")
te, he = evalce(m, trx, trY), evalce(m, hox, hoY)
print(f"{STEPS-1:>6} {te:>10.4f} {he:>12.4f} {he-Hc:>16.4f}   ← 收敛后停在地板上方")
print()
print(f"  训练集 CE {te:.4f} 与地板 Hc {Hc:.4f} 只差 {(te/Hc-1)*100:+.2f}%，")
print(f"  holdout CE {he:.4f} 才是真实水平，比地板高 {(he/Hc-1)*100:+.2f}%。")

print()
print("【E4.4】对照：把因果掩码去掉（泄漏），CE 会掉到地板以下 —— 这不是好消息")
print()
torch.manual_seed(0)
m2 = TinyLM()
opt2 = torch.optim.AdamW(m2.parameters(), lr=3e-3)
gen2 = torch.Generator().manual_seed(123)
for step in range(STEPS):
    i = torch.randint(0, trx.shape[0], (64,), generator=gen2)
    loss = F.cross_entropy(m2(trx[i], causal=False).reshape(-1, V), trY[i].reshape(-1))
    opt2.zero_grad()
    loss.backward()
    opt2.step()
lk_tr = evalce(m2, trx, trY, causal=False)
lk_ho = evalce(m2, hox, hoY, causal=False)
print(f"  无掩码模型：train CE {lk_tr:.4f} / holdout CE {lk_ho:.4f}"
      f" → 比地板低 {(1-lk_ho/Hc)*100:.1f}%")
print(f"  有掩码模型：holdout CE {he:.4f} → 比地板高 {(he/Hc-1)*100:.2f}%")
print()
print("  低于地板只有一种良性来源：有限样本拟合（train 低而 holdout 不低）。")
print(f"  本配置下训练集其实也没跌破地板（{te:.4f} vs {Hc:.4f}，"
      f"高出 {(te/Hc-1)*100:+.2f}%）—— 想让它跌破，得把模型做大、数据做少。")
print(f"  而泄漏版的 {lk_ho:.4f} 是地板以下 {(1-lk_ho/Hc)*100:.1f}%，量级完全不同。")
```

```text
【E4.1】同一份实验里其实有三层地板

  (a) 均匀猜测地板 ln V          = 3.4657 nats
  (b) 只看 token 频率的地板 H(π)  = 3.3977 nats   ← 语料平稳分布的熵
  (c) 给定上一个 token 的地板     = 2.3725 nats   ← 这个任务真正的下界

  只报 (a) 会把「已经接近最优」的模型说成「还有 30% 可降」；
  只报 (c) 会把「模型比基线好多少」这件事抹掉。三个都要报。

  真 P 的 oracle CE（给定上一 token）= 2.3737   ← 与解析 Hc 2.3725 吻合，确认数据生成正确

【E4.2】随机初始化的 CE 一定 ≥ ln V（Jensen：E[-log p] ≥ -log E[p] = ln V）

    logit 标准差 s      实测 CE   ln V + s²/2    超出 ln V
            0.0     3.4657        3.4657     0.0000
            0.5     3.5847        3.5907     0.1190
            1.0     3.9374        3.9657     0.4717
            2.0     5.2058        5.4657     1.7400
            4.0     8.7755       11.4657     5.3098
  （s=4 处偏离 ln V + s²/2 是有限 V 的尾部效应，该式只在 s² ≪ ln V 时成立）

  实测随机初始化 tiny LM 的 logit 标准差 = 0.8536
  实测随机初始化的 CE = 3.8103，ln V = 3.4657 → 高出 9.9%
  按 ln V + s²/2 预言 = 3.8300（与实测差 0.5%）

【E4.3】训练到地板：holdout CE 收敛到 Hc，而不是收敛到 0

  step   train CE   holdout CE     holdout 超出地板
     0     3.5511       3.5504           1.1779
   250     2.3889       2.4003           0.0278
   500     2.3809       2.3912           0.0187
   750     2.3798       2.3899           0.0173
  1000     2.3809       2.3924           0.0199
  1250     2.3800       2.3891           0.0165
  1500     2.3772       2.3902           0.0177
  1500     2.3772       2.3902           0.0177   ← 收敛后停在地板上方

  训练集 CE 2.3772 与地板 Hc 2.3725 只差 +0.20%，
  holdout CE 2.3902 才是真实水平，比地板高 +0.74%。

【E4.4】对照：把因果掩码去掉（泄漏），CE 会掉到地板以下 —— 这不是好消息

  无掩码模型：train CE 0.0369 / holdout CE 0.0401 → 比地板低 98.3%
  有掩码模型：holdout CE 2.3902 → 比地板高 0.74%

  低于地板只有一种良性来源：有限样本拟合（train 低而 holdout 不低）。
  本配置下训练集其实也没跌破地板（2.3772 vs 2.3725，高出 +0.20%）—— 想让它跌破，得把模型做大、数据做少。
  而泄漏版的 0.0401 是地板以下 98.3%，量级完全不同。
```

### 6.3 读结果

**三层地板分别是 3.4657 / 3.3977 / 2.3725 nats。** 用 (a) 当基准，会得出「模型 2.39 离 3.47 还有 31% 空间」；(c) 才是真相——**模型已经只差 0.74%**。

**随机初始化的 CE 是 3.8103，比 $\ln V$ 高 9.9%，而且这个偏差被 $\ln V + s^2/2$ 精确预言（3.8300，差 0.5%）。** 所以「随机初始化 ≈ $\ln V$」这句话只在 logits 尺度很小时近似成立；一旦你的输出层初始化尺度大一点（比如 $s\approx 1$），初始 CE 就会高出 0.5 nats。E4.2 的扫描给出了这个关系的形状。

**然后是 E4.4，这一节真正的操作价值所在。** 我把因果掩码去掉后重训同一个模型，holdout CE 掉到 **0.0401**——比地板 $H_c=2.3725$ **低了 98.3%**。

> **指标越过理论下界，是「有问题」的充分信号。** 它不可能来自你的模型更聪明，只能来自「下界算错了」或「信息从别处漏进来了」。这里就是后者：`nn.TransformerEncoderLayer` 默认是**双向**注意力，位置 $t$ 能看到位置 $t+1$，而位置 $t+1$ 的输入 token 恰好就是位置 $t$ 的预测目标——等于把答案抄进了输入。

这个坑我在写这一期时**真的踩了一次**，而且第一次跑出来 holdout CE 是 0.0694（继续练到 0.0402），看起来像「模型超强」。**是地板把它抓出来的**。这也是为什么本节值得单独成一节：地板不只是用来标注「好多少」，它是**唯一一个能在没有真值的情况下自动发现泄漏的工具**。

（顺带：本次踩的第二个坑是采样函数在文章里重抄时漏了一行 `x = ...`，导致每个位置都从 $x_0$ 采样、根本不是马尔可夫链，算出来的「地板」是 4.6239 而不是 2.3725。**两处实现互相对照发现了它**——因为一个是对的、一个是错的，数字对不上。详见 §11 坑 3。）

---

## 7. 案例五：量化误差的理论下界

第 033 期量到：同一个 int8，「per-tensor 到 per-token-group16」让通道中位 nMSE 从 0.904 降到 0.00207，**好 437 倍**；2×2 消融里「只有通道 outlier 时 per-channel 比 per-token 好 468 倍」。

437 倍是个惊人的数字。它该跟谁比？答案分两层：**同分布数据上的天花板是多少，异质数据上的地板是什么。**

### 7.1 解析

高斯源、$R$ 比特、MSE 失真，Shannon 率失真下界是

$$D(R) = \sigma^2\,2^{-2R} \quad\Longrightarrow\quad \mathrm{nMSE}_{\text{下界}} = 2^{-2R}$$

而一个对称均匀量化器（动态范围 $\pm A$、$2^R$ 个电平）的误差是 $\Delta^2/12$，其中 $\Delta = 2A/2^R$：

$$\mathrm{nMSE}_{\text{均匀}} \approx \frac{\Delta^2}{12\sigma^2} = \frac{A^2}{3\cdot 2^{2R}\sigma^2}$$

**注意这是「高分辨率近似」，只在 $\Delta \ll \sigma$ 时成立。** 所以要比较必须真跑。另外还有一个参照点：给定分布下的**最优**标量量化器（Lloyd-Max 迭代），它代表「固定码率标量量化能达到的最好」。

### 7.2 实跑

```python
# E5：量化误差的零假设 —— 均匀网格有多差、率失真下界在哪、粒度收益到底买到了什么
import math
import torch


def gauss_grid(sigma=1.0, n=20001, lim=8.0):
    x = torch.linspace(-lim * sigma, lim * sigma, n, dtype=torch.float64)
    w = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return x, w / w.sum()


def uniform_q_err(x, w, A, R):
    """对称均匀量化器，动态范围 ±A，2^R 个电平（mid-rise），超出范围即削顶"""
    N = 2 ** R
    d = 2 * A / N
    lvl = -A + (torch.arange(N, dtype=torch.float64) + 0.5) * d
    idx = torch.clamp(torch.floor((x + A) / d).long(), 0, N - 1)
    return ((x - lvl[idx]) ** 2 * w).sum().item()


def lloyd_max(x, w, N, iters=200):
    """给定分布的最优标量量化器（Lloyd-Max 迭代），电平数 N"""
    c = torch.linspace(-3.0, 3.0, N, dtype=torch.float64)
    for _ in range(iters):
        b = (c[:-1] + c[1:]) / 2
        idx = torch.bucketize(x, b)
        num = torch.zeros(N, dtype=torch.float64).scatter_add_(0, idx, x * w)
        den = torch.zeros(N, dtype=torch.float64).scatter_add_(0, idx, w)
        c = torch.where(den > 0, num / den.clamp_min(1e-300), c)
    b = (c[:-1] + c[1:]) / 2
    idx = torch.bucketize(x, b)
    return ((x - c[idx]) ** 2 * w).sum().item()


x, w = gauss_grid(1.0)
print("【E5.1】高斯源 σ=1 上，固定码率标量量化的三个参照点（nMSE = MSE/σ²）")
print()
print(f"{'比特 R':>7} {'均匀网格(最优动态范围)':>22} {'Lloyd-Max 最优':>16} "
      f"{'率失真下界 2^-2R':>18} {'均匀/下界':>10} {'均匀/最优':>10}")
for R in [2, 3, 4, 6, 8]:
    best = min(uniform_q_err(x, w, A, R) for A in [k * 0.1 for k in range(10, 61)])
    lm = lloyd_max(x, w, 2 ** R)
    rd = 2.0 ** (-2 * R)
    print(f"{R:>7} {best:>22.6e} {lm:>16.6e} {rd:>18.6e} "
          f"{best/rd:>10.2f} {best/lm:>10.2f}")
print()
print("  int8（R=8）的参考：均匀网格离率失真下界只有 "
      f"{min(uniform_q_err(x,w,A,8) for A in [k*0.1 for k in range(10,61)])/2.0**-16:.2f}×，")
print("  而任何固定码率标量量化器都不可能低于 2^-2R —— 这就是「再优化也只能挤出 5 倍」的天花板。")

print()
print("【E5.2】异质结构下的粒度收益：per-tensor 的失配是「失效模式」，不是「精度不够」")
print()
C, NCOL, R = 512, 2048, 8
RD8 = 2.0 ** -16
print(f"{'log σ 标准差':>12} {'per-tensor 全局':>16} {'per-channel 全局':>16} "
      f"{'全局比值':>9} {'per-t 通道中位':>15} {'通道中位比值':>13} {'被抹成 0 的通道':>15}")
for ls in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    torch.manual_seed(0)
    sig = torch.exp(torch.randn(C) * ls)
    W = torch.randn(C, NCOL) * sig[:, None]
    st = W.abs().max() / 127
    Qt = (W / st).round().clamp(-127, 127) * st
    sc = W.abs().amax(dim=1, keepdim=True) / 127
    Qc = (W / sc).round().clamp(-127, 127) * sc
    nt = ((W - Qt) ** 2).sum() / (W ** 2).sum()
    nc = ((W - Qc) ** 2).sum() / (W ** 2).sum()
    mt = (((W - Qt) ** 2).sum(1) / (W ** 2).sum(1)).median()
    mc = (((W - Qc) ** 2).sum(1) / (W ** 2).sum(1)).median()
    dead = (Qt.abs().amax(dim=1) == 0).float().mean().item()
    print(f"{ls:>12.1f} {nt.item():>16.4e} {nc.item():>16.4e} {nt.item()/nc.item():>9.2f} "
          f"{mt.item():>15.4e} {mt.item()/mc.item():>13.1f} {dead*100:>14.1f}%")
print()
print(f"  参照：率失真下界 2^-2R = {RD8:.4e}；同分布时 per-channel 全局 nMSE 6.90e-5 = 下界的 "
      f"{6.8995e-05/RD8:.2f}×，Lloyd-Max 最优 5.10e-5 = 下界的 {5.1017e-05/RD8:.2f}×")
print()
print("  两个口径给出**方向相反**的趋势：")
print("    · 全局 nMSE：随异质性先升（1.17e-4 → 9.25e-3）后降（→ 2.89e-4）——")
print("      因为大通道继续主导总能量，小通道被毁掉在聚合里看不见")
print("    · 通道中位 nMSE：单调升到 1.50e4 后饱和 —— 饱和点 = 1/6.65e-5，")
print("      正好是「小通道被整体抹成 0（nMSE 恰为 1）」对应的比值")

print()
print("【E5.3】把第 033 期的实测数字放进同一张坐标系")
print()
print("  第 033 期（通道中位 nMSE，int8）：per-tensor 0.904 → per-token-group16 0.00207 → 437×")
print("  第 033 期（2×2 消融）：只有通道 outlier 时 per-channel 比 per-token 好 468×")
print()
print("  现在能说清那 437× 买的是什么了：")
print("    · 同分布数据上，per-tensor → per-channel 的收益上限是 5.76/3.34 ≈ 1.7×")
print("      （从「均匀网格离下界 5.76×」压到「Lloyd-Max 离下界 3.34×」）")
print("    · 要让通道中位比值上到 1.5e4，只需要 log σ 的标准差 ≥ 1.5 —— 也就是最强通道")
print("      比最弱通道强 e^3 ≈ 20 倍以上，此时最弱的那批通道已经被 8 bit 完全抹平")
print("  → 所以那些数字衡量的是「有没有小通道被彻底丢掉」，不是「量化器精度提高了多少」。")
```

```text
【E5.1】高斯源 σ=1 上，固定码率标量量化的三个参照点（nMSE = MSE/σ²）

   比特 R           均匀网格(最优动态范围)     Lloyd-Max 最优        率失真下界 2^-2R      均匀/下界      均匀/最优
      2           1.188537e-01     1.174820e-01       6.250000e-02       1.90       1.01
      3           3.751090e-02     3.454816e-02       1.562500e-02       2.40       1.09
      4           1.154677e-02     9.503049e-03       3.906250e-03       2.96       1.22
      6           1.041061e-03     6.996626e-04       2.441406e-04       4.26       1.49
      8           8.782293e-05     5.101658e-05       1.525879e-05       5.76       1.72

  int8（R=8）的参考：均匀网格离率失真下界只有 5.76×，
  而任何固定码率标量量化器都不可能低于 2^-2R —— 这就是「再优化也只能挤出 5 倍」的天花板。

【E5.2】异质结构下的粒度收益：per-tensor 的失配是「失效模式」，不是「精度不够」

   log σ 标准差    per-tensor 全局   per-channel 全局      全局比值      per-t 通道中位        通道中位比值       被抹成 0 的通道
         0.0       1.1720e-04       6.8995e-05      1.70      1.1719e-04           1.8            0.0%
         0.5       1.9595e-03       6.7276e-05     29.13      3.6377e-03          54.7            0.0%
         1.0       9.2544e-03       6.2653e-05    147.71      2.0762e-01        3119.4            5.7%
         1.5       4.9933e-03       6.0438e-05     82.62      1.0000e+00       15024.3           63.5%
         2.0       1.6797e-03       5.9879e-05     28.05      1.0000e+00       15024.3           89.1%
         2.5       5.7836e-04       5.9764e-05      9.68      1.0000e+00       15024.3           96.7%
         3.0       2.8923e-04       5.9823e-05      4.83      1.0000e+00       15024.3           98.2%

  参照：率失真下界 2^-2R = 1.5259e-05；同分布时 per-channel 全局 nMSE 6.90e-5 = 下界的 4.52×，Lloyd-Max 最优 5.10e-5 = 下界的 3.34×

  两个口径给出**方向相反**的趋势：
    · 全局 nMSE：随异质性先升（1.17e-4 → 9.25e-3）后降（→ 2.89e-4）——
      因为大通道继续主导总能量，小通道被毁掉在聚合里看不见
    · 通道中位 nMSE：单调升到 1.50e4 后饱和 —— 饱和点 = 1/6.65e-5，
      正好是「小通道被整体抹成 0（nMSE 恰为 1）」对应的比值

【E5.3】把第 033 期的实测数字放进同一张坐标系

  第 033 期（通道中位 nMSE，int8）：per-tensor 0.904 → per-token-group16 0.00207 → 437×
  第 033 期（2×2 消融）：只有通道 outlier 时 per-channel 比 per-token 好 468×

  现在能说清那 437× 买的是什么了：
    · 同分布数据上，per-tensor → per-channel 的收益上限是 5.76/3.34 ≈ 1.7×
      （从「均匀网格离下界 5.76×」压到「Lloyd-Max 离下界 3.34×」）
    · 要让通道中位比值上到 1.5e4，只需要 log σ 的标准差 ≥ 1.5 —— 也就是最强通道
      比最弱通道强 e^3 ≈ 20 倍以上，此时最弱的那批通道已经被 8 bit 完全抹平
  → 所以那些数字衡量的是「有没有小通道被彻底丢掉」，不是「量化器精度提高了多少」。
```

### 7.3 读结果

**int8 的均匀网格离率失真下界只有 5.76 倍。** 这是「最多还能挤多少」的硬上限——$2^{-2R}$ 是信息论下界，任何固定码率标量量化器都打不破它。而 Lloyd-Max 最优也只在 3.34 倍。**所以「per-tensor → per-channel」在同分布数据上的收益上限是 $5.76/3.34\approx1.7$ 倍。**

**但那 437 倍是真的。** 矛盾吗？不矛盾，因为 E5.2 说明了它们量的是不同东西：

- $ls=0$（同分布）：per-tensor/per-channel 比值 1.70，被抹成 0 的通道 **0.0%**；
- $ls=1.0$：全局比值升到 147.71，通道中位比值 3119.4，被抹成 0 的通道 **5.7%**；
- $ls=1.5$：**63.5% 的通道被抹成 0**，per-tensor 的通道中位 nMSE 恰好 = **1.0000e+00**（整通道重建为全零，相对误差 100%），比值饱和在 **15024.3**；
- $ls=3.0$：**98.2% 的通道被抹成 0**，比值仍是 15024.3——**饱和了**。

那个 15024.3 不是「更厉害」，它就是 $1/6.65\times10^{-5}$，即「一个被完全抹平的通道」除以「一个正常量化的通道」。**饱和这件事本身就说明这个比值衡量的是二值的「死/活」判定，不是连续的精度提升。**

还有一个必须点出来的反转：**全局口径下，异质性反而让 per-tensor 的 nMSE 先升后降**（1.17e-4 → 9.25e-3 → 2.89e-4）。如果只看全局 nMSE，你会得出「数据越异质，per-tensor 反而越好用」——因为大通道继续主导分子分母，小通道死了也看不出来。**同一份数据、同一组数字，两个口径给出方向相反的趋势 —— 这正是第 033 期「口径会反转粒度排名」的另一面。**

> **可迁移的判断**：当一个改进的收益远大于理论下界允许的范围时，先别高兴——去问「我的 baseline 是不是引入了一个**失效模式**，而不是精度不足」。修掉失效模式的收益可以是几万倍，逼近理论极限的收益只能是几倍。

---

## 8. 案例六：指标差的显著性基线

最后一个案例是元层面的：**两个配置的指标差要多大，才值得写成一条结论？**

第 036 期已经演示了这个问题的重要性（5 种子测到 +62.1%、20 种子测到 +56.4%）。这一节把它做成一个可复用的流程：**先量「同一配置换种子」的波动，再做置换检验。**

### 8.1 方法

- **噪声地板**：同一个配置跑 $N$ 个种子（只换随机种子），得到指标样本的标准差 $\sigma$。
- **置换检验**：把两组样本的标签随机重贴 $P$ 次，统计「随机分组下的均值差 ≥ 观测均值差」的比例，就是 $p$ 值。
- **需要多少样本**：对两组各 $n$ 个样本，均值差的标准误是 $\sigma\sqrt{2/n}$。在 2 倍标准误上报显著，则

$$n \approx 2\left(\frac{2\sigma}{\delta}\right)^2$$

其中 $\delta$ 是你要检测的绝对效应量。**这条式子是反着用的：先量 $\sigma$，再算 $n$，而不是先定 $n=3$。**

### 8.2 实跑

```python
# E6：指标差的显著性基线 —— 同一配置换种子本身有多大波动，以及这个波动会随训练缩小
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

V, CTX, D, NH, NL = 32, 64, 64, 4, 2
CKPT = [20, 60, 150, 300]
NSEED = 12


def make_chain(seed=0):
    g = torch.Generator().manual_seed(seed)
    P = torch.softmax(torch.randn(V, V, generator=g) * 1.5, dim=-1)
    pi = torch.linalg.matrix_power(P.T, 300)[:, 0]
    pi = pi / pi.sum()
    Hc = -(pi[:, None] * P.clamp_min(1e-30).log() * P).sum().item()
    return P, Hc


def sample_stream(P, n_seq, n_tok, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(V, (n_seq,), generator=g)
    out = [x]
    for _ in range(n_tok - 1):
        x = torch.multinomial(P[x], 1, generator=g).squeeze(-1)
        out.append(x)
    return torch.stack(out, dim=1)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.te, self.pe = nn.Embedding(V, D), nn.Embedding(CTX, D)
        layer = nn.TransformerEncoderLayer(
            D, NH, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, NL, enable_nested_tensor=False)
        self.head = nn.Linear(D, V)
        self.register_buffer("mask",
                             torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), 1))

    def forward(self, x):
        h = self.te(x) + self.pe(torch.arange(x.shape[1]))
        return self.head(self.tr(h, mask=self.mask))


P, Hc = make_chain(0)
tr = sample_stream(P, 32768, CTX + 1, seed=0)
trx, trY = tr[:, :-1], tr[:, 1:]
ho = sample_stream(P, 4096, CTX + 1, seed=999)
hox, hoY = ho[:, :-1], ho[:, 1:]
EG = torch.Generator().manual_seed(7)
EI = torch.randint(0, hox.shape[0], (256,), generator=EG)


def run(seed, lr):
    torch.manual_seed(seed)
    m = TinyLM()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(10000 + seed)
    out = {}
    for step in range(max(CKPT) + 1):
        i = torch.randint(0, trx.shape[0], (64,), generator=gen)
        loss = F.cross_entropy(m(trx[i]).reshape(-1, V), trY[i].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in CKPT:
            with torch.no_grad():
                out[step] = F.cross_entropy(
                    m(hox[EI]).reshape(-1, V), hoY[EI].reshape(-1)).item()
    return out


def mean(a):
    return sum(a) / len(a)


def std(a):
    m = mean(a)
    return math.sqrt(sum((v - m) ** 2 for v in a) / (len(a) - 1))


CONFIGS = [("A lr=3e-3", 3e-3), ("B lr=3.6e-3", 3.6e-3), ("C lr=1.2e-2", 1.2e-2)]
res = {n: [] for n, _ in CONFIGS}
for name, lr in CONFIGS:
    for s in range(NSEED):
        res[name].append(run(s, lr))

print("【E6.1】同一配置（A，lr=3e-3）只换种子，holdout CE 在训练不同阶段的波动")
print()
print(f"  任务地板 Hc = {Hc:.4f}   每组 {NSEED} 个种子")
print()
print(f"{'step':>6} {'均值':>9} {'标准差':>9} {'变异系数':>9} {'最小':>9} {'最大':>9} {'极差/均值':>10}")
for st in CKPT:
    v = [r[st] for r in res["A lr=3e-3"]]
    m, s = mean(v), std(v)
    print(f"{st:>6} {m:>9.4f} {s:>9.4f} {s/m*100:>8.3f}% "
          f"{min(v):>9.4f} {max(v):>9.4f} {(max(v)-min(v))/m*100:>9.3f}%")
print()
print("  → 种子噪声的地板不是常数：越接近收敛越小（本配置从 0.9% 缩到 0.09%）。")
print("    所以「多大的差才算差」必须先声明「在什么训练阶段、什么收敛程度下」。")

print()
print("【E6.2】置换检验：把 A 与 B/C 的 24 个样本随机重贴标签，看观测差值在零分布里的位置")
print()


def perm_test(x, y, n_perm=20000, seed=0):
    obs = mean(x) - mean(y)
    pool, nx = x + y, len(x)
    g = torch.Generator().manual_seed(seed)
    cnt = 0
    for _ in range(n_perm):
        idx = torch.randperm(len(pool), generator=g).tolist()
        d = mean([pool[i] for i in idx[:nx]]) - mean([pool[i] for i in idx[nx:]])
        cnt += abs(d) >= abs(obs)
    return obs, (cnt + 1) / (n_perm + 1)


print(f"{'step':>6} {'对比':>16} {'均值差':>10} {'差值/σ_A':>10} {'置换 p':>9} {'判定':>24}")
for st in CKPT:
    for nm in ["B lr=3.6e-3", "C lr=1.2e-2"]:
        x = [r[st] for r in res["A lr=3e-3"]]
        y = [r[st] for r in res[nm]]
        obs, p = perm_test(x, y)
        verdict = "在种子噪声里，区分不出" if p > 0.05 else "超出种子噪声"
        print(f"{st:>6} {'A vs ' + nm:>16} {obs:>10.4f} {obs/std(x):>10.2f} {p:>9.4f} {verdict:>24}")
    print()

print("【E6.3】「每组跑 n 个种子」能分辨多小的差异（从 A 自己的样本里重采样）")
print()
m_a = mean([r[300] for r in res["A lr=3e-3"]])
g = torch.Generator().manual_seed(3)
print(f"{'step':>6} {'n':>3} {'Δ均值 p50':>11} {'Δ均值 p90':>11} {'Δ均值 max':>11} {'最小可分辨/均值':>15}")
for st in [20, 300]:
    v = [r[st] for r in res["A lr=3e-3"]]
    mv = mean(v)
    for n in [3, 5, 12]:
        if 2 * n > len(v):
            continue
        ds = []
        for _ in range(4000):
            idx = torch.randperm(len(v), generator=g).tolist()
            ds.append(abs(mean([v[i] for i in idx[:n]])
                          - mean([v[i] for i in idx[n:2 * n]])) / mv)
        ds.sort()
        print(f"{st:>6} {n:>3} {ds[len(ds)//2]*100:>10.2f}% {ds[int(0.9*len(ds))]*100:>10.2f}% "
              f"{ds[-1]*100:>10.2f}% {ds[-1]*100:>14.2f}%")
print()
print("  这些都是「同一个配置、同一份代码」内部的重采样，差异完全来自种子。")
print("  反过来用：要在 2 倍标准误上分辨相对差异 δ，需要每组 n ≈ 2(2σ/δ)² 个种子 ——")
print()
print(f"{'目标 δ/均值':>12} {'需要 n（σ@step300={:.4f}）'.format(std([r[300] for r in res['A lr=3e-3']])):>32} "
      f"{'需要 n（σ@step20={:.4f}）'.format(std([r[20] for r in res['A lr=3e-3']])):>32}")
s300 = std([r[300] for r in res["A lr=3e-3"]])
s20 = std([r[20] for r in res["A lr=3e-3"]])
mm = mean([r[300] for r in res["A lr=3e-3"]])
for d in [0.001, 0.005, 0.01, 0.02]:
    n300 = math.ceil(2 * (2 * s300 / (d * mm)) ** 2)
    n20 = math.ceil(2 * (2 * s20 / (d * mm)) ** 2)
    print(f"{d*100:>11.1f}% {n300:>32} {n20:>32}")
print()
print("  结论有两面，都别搞反：")
print(f"    · 这个配置里种子噪声其实很小（step300 的 σ/均值 = {s300/mm*100:.3f}%），")
print("      所以讨论「跑几个种子」在这里是次要问题；")
print("    · 但正因为噪声小，0.1% 量级的差异就已经是统计显著的 —— 此时真正该问的")
print("      不是「显不显著」，而是「这 0.1% 对下游有没有意义」。")
```

```text
【E6.1】同一配置（A，lr=3e-3）只换种子，holdout CE 在训练不同阶段的波动

  任务地板 Hc = 2.3725   每组 12 个种子

  step        均值       标准差      变异系数        最小        最大      极差/均值
    20    2.5802    0.0103    0.400%    2.5584    2.5993     1.584%
    60    2.4447    0.0039    0.158%    2.4385    2.4507     0.497%
   150    2.4099    0.0029    0.122%    2.4037    2.4143     0.442%
   300    2.3999    0.0022    0.093%    2.3966    2.4027     0.256%

  → 种子噪声的地板不是常数：越接近收敛越小（本配置从 0.9% 缩到 0.09%）。
    所以「多大的差才算差」必须先声明「在什么训练阶段、什么收敛程度下」。

【E6.2】置换检验：把 A 与 B/C 的 24 个样本随机重贴标签，看观测差值在零分布里的位置

  step               对比        均值差     差值/σ_A      置换 p                       判定
    20 A vs B lr=3.6e-3     0.0231       2.24    0.0000                   超出种子噪声
    20 A vs C lr=1.2e-2     0.0804       7.79    0.0000                   超出种子噪声

    60 A vs B lr=3.6e-3     0.0061       1.57    0.0010                   超出种子噪声
    60 A vs C lr=1.2e-2     0.0174       4.52    0.0000                   超出种子噪声

   150 A vs B lr=3.6e-3     0.0004       0.14    0.7288              在种子噪声里，区分不出
   150 A vs C lr=1.2e-2    -0.0039      -1.33    0.0011                   超出种子噪声

   300 A vs B lr=3.6e-3    -0.0005      -0.23    0.5719              在种子噪声里，区分不出
   300 A vs C lr=1.2e-2    -0.0033      -1.47    0.0011                   超出种子噪声

【E6.3】「每组跑 n 个种子」能分辨多小的差异（从 A 自己的样本里重采样）

  step   n     Δ均值 p50     Δ均值 p90     Δ均值 max        最小可分辨/均值
    20   3       0.23%       0.53%       0.92%           0.92%
    20   5       0.19%       0.42%       0.63%           0.63%
   300   3       0.05%       0.13%       0.22%           0.22%
   300   5       0.04%       0.10%       0.19%           0.19%

  这些都是「同一个配置、同一份代码」内部的重采样，差异完全来自种子。
  反过来用：要在 2 倍标准误上分辨相对差异 δ，需要每组 n ≈ 2(2σ/δ)² 个种子 ——

     目标 δ/均值           需要 n（σ@step300=0.0022）            需要 n（σ@step20=0.0103）
        0.1%                                7                              148
        0.5%                                1                                6
        1.0%                                1                                2
        2.0%                                1                                1

  结论有两面，都别搞反：
    · 这个配置里种子噪声其实很小（step300 的 σ/均值 = 0.093%），
      所以讨论「跑几个种子」在这里是次要问题；
    · 但正因为噪声小，0.1% 量级的差异就已经是统计显著的 —— 此时真正该问的
      不是「显不显著」，而是「这 0.1% 对下游有没有意义」。
```

### 8.3 读结果

**发现一：种子噪声的地板不是常数，它随训练收敛而缩小。** 同一个配置（A，lr=3e-3）在 12 个种子上的变异系数：第 20 步 0.400%、第 60 步 0.158%、第 150 步 0.122%、第 300 步 **0.093%**——缩了 4.3 倍。**所以「我的实验波动大不大」这个问题必须带上「在什么阶段问」**，否则没法比较两个不同论文里报的种子数。

**发现二（本节最有价值的一条）：同一个对比的显著性可以随训练阶段完全反转。**

| 对比 | 第 20 步 | 第 60 步 | 第 150 步 | 第 300 步 |
|---|---|---|---|---|
| A vs B（lr +20%） | +0.0231，p<1e-4 | +0.0061，p=0.0010 | +0.0004，**p=0.7288** | −0.0005，**p=0.5719** |
| A vs C（lr 4×） | +0.0804，p<1e-4 | +0.0174，p<1e-4 | −0.0039，p=0.0011 | −0.0033，p=0.0011 |

读法：把学习率提高 20%（A→B），在**第 20 步**是一个 p < 1e-4 的「巨大改进」（CE 低 0.0231，相当于 σ 的 2.24 倍）；到第 300 步，差异只剩 0.0005 且 p = 0.5719，**完全在噪声里**。而把学习率提高 4 倍（A→C），到第 300 步 CE 反而**低了 0.0033**（p = 0.0011），也就是「4 倍学习率更好」。

> 这意味着：**中期指标的排序和最终指标的排序可以相反。** 任何「我们只训练了 20% 的步数就超过了 baseline」的结论，都必须补一句「在最终步数上是否仍然成立」。

**发现三：噪声小的时候，「显著」反而不再是有用的判据。** E6.3 的反向问法是「每组 $n$ 个种子能分辨多小的差异」：第 300 步、$n=3$ 时的 Δ均值最大只有 **0.22%**，$n=5$ 是 0.19%。所以在这个配置里，「0.5% 的差异」根本不是噪声问题。而反过来，$n$ 的公式表显示：要在第 20 步分辨 0.1% 的相对差异需要 **148** 个种子，在第 300 步只需要 **7** 个。

**结论有两面，都别搞反**：种子噪声要**先测量**，不能假设它一定大或一定小。噪声大时，「跑几个种子」是主要矛盾；噪声小时，「这 0.1% 对下游有没有意义」才是主要矛盾。

---

## 9. 围绕该领域展开

### 9.1 已发各期里的零假设普查

这张表是本期的实用产出：**把过去 37 期里出现过的关键数字，配回它们各自的零假设**。加粗的是当年**没有**配基线的那些——不是错误，是缺口。

| 期数 | 指标读数 | 它的零假设 | 读数 / 零假设 |
|---|---|---|---|
| 022 | 默认 `nn.Linear` 逐层增益 0.4080 | 解析 $1/\sqrt6=0.4082$（Kaiming/Xavier） | 0.995× —— **有基线** ✓ |
| 026 | aux=0.1 的 Gini 0.0797 | 均匀路由 Gini 0.0485（本节 §4） | 1.64×（归一化后 0.0318） |
| **026** | **不均衡时的 max/avg 3.26** | **解析 1.2192（本节 §4）** | **2.67×** ← 当年无基线 |
| 029 | 树·无限容量命中率 93.63% | 上界即自身；块粒度 92.69% | 差 0.94pt —— **有上界** ✓ |
| 031 | `loadaware` β=1.0 命中率 55.90% | `cache_blind` 51.90%（什么都不做） | 1.08× —— **有基线** ✓ |
| 033 | per-token-group16 通道中位 nMSE 0.00207 | 率失真下界 $2^{-16}=1.53\text{e-}5$（§7） | 135×（⚠️ **口径不同**：前者是逐通道归一后的中位，后者是高斯源全局界，只能当数量级参照） |
| 033 | per-tensor → per-group16「好 437 倍」 | 同分布下的上限只有 1.7×（§7） | 说明买的是「失效模式」 |
| 034 | oracle recall 0.9346（K=256, L=2048） | 随机 recall $K/L=0.1250$（§5） | 7.48× |
| **035** | **软召回 H2O 0.0421（B=16）** | **随机列 0.0787（解析 $K/L=0.0833$）** | **0.53× —— 低于随机** ← 当年无基线 |
| 035 | 硬指标 H2O 0.0312 | 任务随机猜 $1/V=1/32=0.03125$ | **0.998× —— 恰好等于随机** |
| 036 | 位置 0 的 sink 质量 0.46043 | $b(0)=(H_S-1)/(S-1)$（§3） | 13.27× —— **有基线** ✓ |
| 036 | 删位置 0 的代价 +62.1% | 删其它位置的 +0.0%（20 种子） | 无穷大 —— **对照齐全** ✓ |

**这张表最值得看的是第 035 期那两行。** 同一份实验、同一批数字，两个指标的零假设分别是 $1/32=0.03125$ 和 $K/L=0.0833$——**两者差了 2.67 倍，而且是完全不同的两种零假设**（前者是任务的类别数，后者是选择预算）。当年的文章把随机基线列在了软召回那张表里（0.0787），却没有把它和硬指标的 0.0312 区分开——**读者很容易把「0.0312」当成「比随机好一点」，实际上它就是随机。**

好消息是：把基线摆出来后，第 035 期的结论反而**更强**了——**两条独立口径都指向「H2O 在 B=16 时没有提供任何信息」**，而不是一条。

### 9.2 通用配方

把这六节压成四步，下次做实验可以直接套：

**第一步，问「这个指标是怎么被结构影响的」。** 列出所有会引入结构的东西：掩码（因果/滑窗/块稀疏）、采样（有放回/无放回/top-k）、预算上限、有限容量、位置编码、池化或归约。**每一个都会贡献一份非零基线。**

**第二步，先算解析，算不出再模拟。** 结构的对称性往往能直接给出闭式——比如 §3 的 $b(j)$、§4 的 $\sigma/(\mu\sqrt\pi)$、§5 的 $K^2/L$。**解析形式的额外好处是它告诉你基线与哪些量有关**（比如 $G_{\rm null}\propto1/\sqrt{TK}$，立刻说明「小 batch 读数更大」是噪声不是退化）。

**第三步，配上理论上下界。** CE 有 $H(\cdot\mid\cdot)$，量化有 $2^{-2R}$，召回有可达上界（§9.1 的 029/034 行），roofline 有 $B^*=P/\text{BW}\times\text{bytes}/2$（027/028/029 反复用到的那个 295）。**有了下界，你才能把「改进」分成两类：逼近极限（受限于几倍）vs 修掉失效模式（可以是几万倍）。** 后者往往才是真正的价值所在。

**第四步，量噪声地板，再做置换检验。** 别假设种子数够不够——跑一遍同配置多种子，量 $\sigma$，再用 $n\approx2(2\sigma/\delta)^2$ 反推需要多少。**注意 $\sigma$ 随训练阶段变化，所以要在你要下结论的那个阶段量。**

### 9.3 关联到已发各期

- **第 036 期（Attention Sink）**：本期的起点。036 的头号坑就是「拿 $1/S$ 当零 sink 基准」，把结构自带的 $\ln S$ 倍加成当成了异常。§3 把那个基准从位置 0 推广到所有位置，并解释了为什么偏差随 $S$ 放大。
- **第 026 期（MoE）**：§4 把 Gini 0.0797 放回坐标系，并发现辅助损失其实已经把偏好压到接近检测极限（归一化不均衡度 0.0318）。
- **第 033 期（KV 量化）**：§7 说明「好 437 倍」和「逼近率失真下界」是两件不同的事，前者买的是异质结构下的失效模式修复。
- **第 034/035 期（稀疏注意力 / KV 驱逐）**：§5 给集合类指标配上噪声地板，并指出交叠率的地板随输入相关性从 3.16% 涨到 100%——**这是比较「同一模型不同层」时的隐藏陷阱**。
- **第 029/031 期（前缀缓存 / KV 路由）**：§9.1 的表里，那两期的数字都自带基线（块粒度 92.69% 对照无限容量 93.63%；`cache_blind` 51.90% 对照 `loadaware` 55.90%），这是好习惯；本期做的是把这种习惯系统化。
- **第 027/028 期（投机解码 / 连续批处理）**：那个反复出现的 $B^*=295$ 本身就是一种零假设——「算力时间恰好等于内存时间」的分界点。它是解析的、与模型大小无关，正是 §9.2 第二步推崇的那种基线。

---

## 10. 什么时候该用 / 不该用

**该用：**

- 任何要写进结论、汇报、论文的**比较性数字**（X 比 Y 好多少）。
- 任何「不均衡度 / 熵 / 集中度 / recall / 交叠率 / 命中率」类的百分比指标——它们几乎都有非零基线。
- 任何涉及**有限样本**的估计量：小 batch 的负载、少量 query 的召回、短序列的注意力统计。
- 任何怀疑**泄漏或管线 bug** 的时刻：先算理论下界，跑出来低于它就直接去查管线，别调参。
- **决定要不要换方案之前**：先看「离理论极限还有多远」，如果只剩 2 倍，那再优化也挤不出 437 倍。

**不该用（或者说不值得做）：**

- **纯描述性、不做对比的场景**。你只是想知道某个模型的注意力长什么样，不需要零假设。
- **效应量已经远超所有基线、且机制清楚的时候**。比如 036 期「删位置 0 的代价 +62.1% vs 删位置 1 的 +0.0%」——这个对照本身就是完备的，不需要再去算解析基线。
- **当基线本身的不确定度大于被测差异时**。此时正确的动作不是「再补一个基线」，而是**把实验做大**（更长训练、更多种子、更大样本），因为 $\sigma\propto1/\sqrt n$。
- **不要为了凑一个漂亮的比例而选基线**。同一份数据可以同时有「比随机好 7.5 倍」和「比滑窗差 5 个百分点」两个真命题，**选哪个取决于你要回答什么问题**，而不是哪个数字好看。

---

## 11. 常见坑

**坑 1：拿「均匀分布」当零假设，忘了结构已经把它带偏。**

第 036 期的 $1/S$（偏差 4.48× 到 11.36×）是最典型的。同形状的坑还有：MoE 路由熵用 $\ln E$ 当基线（实际基线是有限样本涨落下的期望熵）、稀疏 recall 用 $k/L$ 之外还叠加了块对齐税（034 期的 $(b-1)/(2K)$）、量化 nMSE 用 0 当基线（实际下界是 $2^{-2R}$ 而非 0）。

**自检话术**：「在这个结构下，一个什么都不做的系统会输出什么？」——把这句话答出来，基线就有了。

**坑 2：把「基线是 0」和「基线是 $O(1/\sqrt n)$」搞混。**

§4 实测：$T$ 从 16384 降到 256，零假设 Gini 从 0.0246 涨到 0.1883。**这是 7.7 倍的变化，全部来自样本量，不是模型退化。** 同理 §8 里种子噪声从 0.400% 缩到 0.093%。**只要指标是从有限样本里数出来的，就一定要报 $n$。**

**坑 3：同一个函数在两处抄写时漏了一行 —— 而且它不报错。**

本期真的踩到了。E4 的采样函数在文章里重抄时漏掉了 `x = `：

```python
# 错：每个位置都从 x_0 采样，根本不是马尔可夫链
for _ in range(n_tok - 1):
    out.append(torch.multinomial(P[x], 1, generator=g).squeeze(-1))
# 对：
for _ in range(n_tok - 1):
    x = torch.multinomial(P[x], 1, generator=g).squeeze(-1)
    out.append(x)
```

后果：算出来的「任务地板」是 **4.6239**，而真实值是 **2.3725**。它不报错、不警告，只是把一个错误的基线喂给了后面所有分析。**发现它的方式是两个实现互相对照**（一个来自 `tiny_lm.py`、一个来自文章里的抄写版），数字对不上就暴露了。

**推论**：地板/基线这类「参考量」尤其要交叉验证，因为**它们错了不会以异常的形式出现，只会以「后续所有结论都偏」的形式出现**。

**坑 4：指标越过理论下界，却当成好消息。**

§6 的 E4.4：去掉因果掩码后 holdout CE = 0.0401，比地板低 98.3%。第一眼看像「模型超强」。**任何跨过理论下界的数字都只可能来自三个原因：下界算错、泄漏、或者指标口径变了（比如从 holdout 换成了 train）。**

配套的判据：**低于地板要分清「大幅且 train/holdout 一起低」（泄漏）和「小幅且只有 train 低」（有限样本拟合）**。本期配置下训练集也没跌破地板（+0.20%），说明要制造良性的跌破需要「模型更大、数据更少」。

**坑 5：只报「比 baseline 好多少」，不报「离上界还有多远」。**

§7 的对比：int8 均匀网格离率失真下界 5.76×，Lloyd-Max 最优 3.34×。所以「把 per-tensor 换成 per-channel」在同分布数据上的上限是 1.7×——**如果你手上真的出现了 437×，那它一定来自别的地方**（异质结构的失效模式），这句话本身就是一次机制核查的成功。

**上界与基线回答的是不同问题**：基线回答「有没有信息」，上界回答「还有多少空间」。**只报基线，会把「已经到顶」说成「大有可为」；只报上界，会把「显著改进」说成「忽略不计」。**

---

## 12. 一句话总结

> **零假设必须由「产生这个指标的机制」生成，而不是由「均匀分布」生成；把这一步做对之后，你会发现很多曾经惊人或令人不安的数字，只是结构、样本量和抽样方式本来就会给出的读数——而剩下的那部分，才是真正值得写成结论的东西。**

---

## 13. 今日练习

给你一个真实的判断场景，做完之后你应该能把本期的四步配方完整跑一遍。

**场景**：你训练了两个 MoE 模型，配置完全相同，只有辅助损失系数不同（模型甲 $\alpha_{\rm aux}=0.05$，模型乙 $\alpha_{\rm aux}=0.2$）。每个模型你都只跑了一次（一个种子）。评价指标是最终专家负载的 Gini。你测到：

- 模型甲：Gini = 0.0712，max/avg = 1.41
- 模型乙：Gini = 0.0543，max/avg = 1.28

你的同事说：「乙比甲均衡 24%，所以 $\alpha_{\rm aux}=0.2$ 更好。」

请回答四个问题：

1. 这两个数字的**零假设**是多少？（$E=64$、$K=2$、每步 $T=2048$ 个 token；用 §4 的解析式）
2. 换算成**归一化不均衡度** $(G-G_{\rm null})/(1-G_{\rm null})$，甲和乙分别是多少？同事那句「均衡 24%」在这个坐标系里是什么意思？
3. 这个对比**能不能支撑结论**？如果不能，缺什么？需要多少样本才能分辨出甲和乙这个量级的差异？（用 §8 的 $n\approx2(2\sigma/\delta)^2$，$\sigma$ 用本节 E6.1 在第 300 步实测的相对标准差）
4. 如果甲和乙的 Gini 差是**真的**（假设噪声已排除），你还需要补哪一个基线，才能说「乙的负载分布对下游更有利」？

<details><summary>今日练习（参考答案）</summary>

### 1. 零假设

$E=64$，$K=2$，$T=2048$，所以

$$\mu=\frac{TK}{E}=\frac{2048\times2}{64}=64,\qquad \sigma=\sqrt{64\left(1-\frac{1}{64}\right)}=\sqrt{63}\approx7.937$$

$$G_{\rm null}=\frac{\sigma}{\mu\sqrt{\pi}}=\frac{7.937}{64\times1.77245}\approx 0.0700$$

max/avg 的零假设：$\mathbb{E}[\max]\approx\mu+2.5\sigma=64+19.84=83.84$，所以 $83.84/64\approx1.310$。

**注意：$G_{\rm null}=0.0700$，而模型甲实测 0.0712。** 也就是说甲几乎**恰好就是均匀路由**，它的读数里几乎全是采样噪声。而模型乙的 0.0543 **低于**零假设——这在数学上不是不可能（零假设是一个分布，0.0543 是它的一个尾样本），但它意味着乙的负载比「完全均匀随机」还要均衡，这本身需要解释（可能是辅助损失真的在压分布的形状，也可能是这一次种子恰好落在了分布的左尾）。

### 2. 归一化不均衡度

$$G_{\rm norm}=\frac{G-G_{\rm null}}{1-G_{\rm null}}=\frac{G-0.0700}{0.9300}$$

- 模型甲：$(0.0712-0.0700)/0.9300 = \mathbf{0.0013}$
- 模型乙：$(0.0543-0.0700)/0.9300 = \mathbf{-0.0169}$

同事说的「均衡 24%」，是 $(0.0712-0.0543)/0.0712 = 23.7\%$——**这个计算把两个都接近零假设的读数相减，得到的分母（0.0712）本身大部分是噪声。** 换算到归一化坐标系之后，甲是 0.0013、乙是 −0.0169，两者都在 0 附近，**差异的说服力远没有「24%」听起来那么大**。

（对照：第 026 期里 aux=0.1 的归一化不均衡度是 0.0318，而在完全不做平衡时是 0.8685——**真正的效应量级是 0.03 到 0.87，不是 0.001 到 -0.017。**）

### 3. 能不能支撑结论

**不能。** 三个缺失：

1. **没有噪声地板**——每个配置只跑了一个种子，无法判断 0.0712 和 0.0543 的差是不是种子涨落。
2. **没有样本量声明**——§4 已经表明 $G_{\rm null}\propto1/\sqrt{TK}$，所以 $T$、$K$、$E$ 必须一起报，否则「更均衡」可能只是「更小的 batch」。
3. **口径没对齐**——如果要和已发表的工作比，必须说明用的是全局 Gini 还是逐 token 平均、是否加权。

**需要多少种子**：用 §8 在收敛阶段实测的相对标准差 $\sigma/\bar x = 0.093\%$（这是一个 *CE* 的噪声比例，MoE 的 Gini 在训练后期同样接近收敛，量级可作下界参考）。观测到的相对差异 $\delta = 0.0712-0.0543 = 0.0169$，相对量级 $0.0169/0.0628 \approx 27\%$。代入

$$n\approx2\left(\frac{2\sigma_{\rm rel}}{\delta_{\rm rel}}\right)^2 = 2\left(\frac{2\times0.00093}{0.27}\right)^2 \approx 4.7\times10^{-5}$$

也就是说，**如果 Gini 的种子噪声真的有 CE 那么小，$n=1$ 就够了**。但这个答案显然是荒谬的——它恰恰说明**第 3 步的真正动作不是套公式，而是先测量 Gini 自己的 $\sigma$**。Gini 是一个比 CE 更抖的统计量（它由 $E$ 个计数的极值行为主导，而 CE 是 $T\!\times\!V$ 个样本的平均），**它的种子噪声必须实测，不能从另一个指标的噪声外推。** 这就是 §9.2 第四步「量噪声地板」不能跳过的原因：**噪声不是从别的指标搬过来的，是量出来的。**

### 4. 还需要什么基线

至少还要补两个：

- **上界**：**负载完全均匀**时的 Gini（$T\!\times\!K$ 能被 $E$ 整除时为 0，否则是 $\le1/(2\mu)$ 量级的小值）。它回答「还有多少空间」。
- **「均衡是否等于有效」的对照**：负载均衡本身不是目标。真正该量的是**路由质量的代价**——比如「乙的辅助损失压制了专家特化程度吗」。第 026 期已经量到过这个副作用：**过度平衡会把专家质量极差从 1.9× 压到 1.4×，抹平专家特化**，这正是 DeepSeek 走 aux-loss-free 路线的深层动机。

所以完整答案是：**「乙的 Gini 更低」在 §2 的坐标系里只说明「乙更接近均匀路由」，而均匀路由既不是上界也不是目标——它是一个「没有偏好」的状态。要判断好坏，必须再给一个「下游效果」的指标，并把它的地板也算出来。**

这就是本期标题里那句话的完整含义：**一个数字减掉零假设之后剩下的部分，才是信息。**

</details>

---

*本期所有输出块均由 `scratch-037/` 下的脚本真实执行产生，自检脚本逐块比对 stdout（每块独立子进程）。实验环境：torch 2.14.0，CPU（本机无 CUDA）。*
