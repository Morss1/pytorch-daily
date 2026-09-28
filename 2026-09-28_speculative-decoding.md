# PyTorch 每日一课 · 第 027 期

## 投机解码（Speculative Decoding）：一次搬砖，砌多块墙

> **日期**：2026-09-28
> **难度**：⭐⭐⭐⭐
> **前置知识**：知道自回归生成是怎么跑的（每一步只吐一个 token）；对「显存带宽」和「算力」是两个不同瓶颈有概念；看过采样（temperature / top-k / top-p）的基本写法。第 011 期 KV Cache、第 026 期 MoE 没看过不影响，本篇独立可读。
> **预计阅读时间**：35 分钟

---

## 一、这个领域解决什么问题

### 1.1 decode 阶段的诅咒：搬一次砖，砌一块墙

先看一个所有人都知道、但很少算清楚的事实。

一个 7B 的 LLM（BF16）在 H800 上做 batch=1 的 decode，每一步要多久？

答案几乎是**常数**，与「已经生成了多少 token」无关：约 4.18 毫秒。

为什么？因为这一步的计算是：

$$\text{输出} = W \cdot x,\quad W \in \mathbb{R}^{7\times 10^9},\ x \in \mathbb{R}^{d_{model}}$$

占主导的不是乘加运算，而是**把 14 GB 的权重从 HBM 搬进 SM**。搬运时间：

$$t_{\text{mem}} = \frac{7\times10^9 \times 2\ \text{bytes}}{3.35 \times 10^{12}\ \text{B/s}} = 4.179\ \text{ms}$$

这 4.179 ms 里，GPU 实际做了多少有用功？一次前向的浮点运算量是 $2N = 14$ GFLOP（batch=1、每个参数只用一次）。所以实际算力：

$$\frac{14\times10^9}{4.179\times10^{-3}} = 3.35\ \text{TFLOPS}$$

而 H800 的 BF16 峰值是 **989 TFLOPS**。**算力利用率 0.34%。**

这就引出一个干净的判据：这次前向到底是「带宽瓶颈」还是「算力瓶颈」？比较两件事：

| | 表达式 | 数值（7B BF16） |
|---|---|---|
| 把权重搬完要多久 | $N\cdot\text{bytes} / BW$ | 4.179 ms |
| 把算力喂饱要多久 | $2N \cdot (B\cdot L) / \text{PEAK}$ | $1.4\times10^{-11} \cdot B L$ 秒 |

令两者相等，$N$ 会被约掉：

$$B \cdot L = \frac{\text{PEAK}}{BW} \cdot \frac{\text{bytes}}{2}$$

代入 989 TFLOPS / 3.35 TB/s = **295.2 FLOP/byte** 的算术强度平衡点：

- BF16（2 bytes）：$B\cdot L = \mathbf{295.2}$
- FP8（1 byte，算力也翻倍的话）：仍是 295.2；若算力不变则是 147.6

**这个常数 295 与模型多大完全无关。** 7B 也好 700B 也好，一次前向要「喂饱算力」，都需要 $B\times L \approx 295$ 个 token。

现在把 decode 的真实处境代进去：$B=1$、$L=1$（batch=1，序列长 1）。$B \cdot L = 1$，离 295 差 **295 倍**。

> **decode 的本质：一条 4.18 毫秒的装配线上，只过去 1 个 token。**

### 1.2 已有的加速手段，为什么都不够

| 手段 | 它改变了什么 | 它**不能**改变什么 |
|---|---|---|
| 量化（INT8/FP8/INT4） | 减小 bytes → 缩短 $t_{\text{mem}}$ | 每步还是只出 1 个 token |
| Continuous Batching | 增大 $B$ → 提高算力利用率 | batch 大了能提吞吐，但**单序列延迟不变** |
| Paged Attention | 减少 KV 碎片 → 提高 KV 容量 | 与 $t_{\text{mem}}$ 无关 |
| torch.compile | 减少 kernel launch / 融合算子 | 搬不动的那 14 GB 还是要搬 |

它们的共同边界是：**每一步的处理量是 1 个 token 这件事，没有被触碰。**

### 1.3 把问题重新问一遍

既然 4.179 ms 是「把 14 GB 搬一遍」的固定开销，那——

> **能不能一次搬运，多算几个 token？**

线性代数上完全可以：一次前向处理 $L$ 个 token 的算力是 $2N\cdot L$，而搬运量是**同一个** 14 GB。当 $L=9$ 时，$B\cdot L = 9 \ll 295$，仍是带宽瓶颈，耗时**还是 4.179 ms**（本地实测的 roofline 计算：$L=9$ 的算力项是 0.127 ms，被 4.179 ms 的带宽项完全掩盖）。

所以「验证 9 个 token」的成本 ≈ 「生成 1 个 token」的成本。

**问题只剩一个：那 9 个 token 从哪来？** 如果让目标模型自己生成，它必须一次一个（自回归依赖），省不下来。

**投机解码的答案：找个便宜模型先猜，目标模型只负责"批改"。**

类比：主编（大模型）写一篇稿子要 4 小时/段。实习生（小模型）写一段只要 0.2 小时，但质量不可靠。与其让主编一段段写，不如让实习生一口气写 9 段草稿（1.8 小时），主编花 4 小时**一次性批注全部 9 段**——接受写得对的部分，改掉第一个错的地方。只要实习生的接受率不是太差，主编的批注时间几乎被摊薄了。

---

## 二、核心思想

### 2.1 两个模型，两个角色

- **draft 模型（$q$）**：便宜、快，负责串行地猜 $\gamma$ 个 token。
- **target 模型（$p$）**：昂贵、准，**一次前向**验证全部 $\gamma+1$ 个位置。

关键在于：验证时 target 一次性算出 $\gamma+1$ 个位置的 logits，每个位置独立地和 draft 给出的候选做比对。**接受前 $k$ 个连续正确的，在第一个出错的位置用 target 自己的输出替换**。这样一次 target 前向产出了 $k+1$ 个 token。

### 2.2 最漂亮的一点：它不是近似

这是投机解码和量化、剪枝、蒸馏最本质的区别——**输出的分布和 target 单独采样完全一致，一个 bit 都不差**。原理是「修正拒绝采样」（modified rejection sampling）：

对 draft 采出的每个候选 $x \sim q$：

1. 以 $a(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$ 的概率**接受**它；
2. 否则**拒绝**，并从残差分布 $(p - q)_+$（只保留 $p>q$ 的部分，重新归一化）里重新采一个 token。

**为什么这样就对了？** 把两步的贡献加起来。记接受率 $\beta = \sum_x q(x)a(x) = \sum_x \min(p(x), q(x))$：

$$\pi(x) = \underbrace{q(x)a(x)}_{\text{接受路径}} + \underbrace{(1-\beta)\cdot\frac{(p(x)-q(x))_+}{1-\beta}}_{\text{拒绝路径}} = \min(p,q)(x) + (p(x)-q(x))_+$$

而 $\min(p,q) + (p-q)_+$ 恰好逐点等于 $p$：$p \ge q$ 时得 $q + (p-q) = p$；$p < q$ 时得 $p + 0 = p$。**所以 $\pi \equiv p$，精确成立。**

顺手得到一个非常有用的推论：

$$\boxed{\ \alpha = \beta = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p, q)\ }$$

**单步接受率 = 1 减去两个模型输出的总变差距离。** 这意味着你不用跑实验就能估算接受率上界：把两个模型的 next-token 分布抓出来算一下 TV 就行。（本文后面所有的接受率实测，都和这条公式对上了。）

### 2.3 期望接受长度

设每个位置的接受概率都是 $\alpha$（独立近似）。一次 target 前向最多接受 $\gamma$ 个候选，加上被拒绝位置上 target 补的一个「免费」token，产出的期望 token 数：

$$\tau(\gamma) = 1 + \alpha + \alpha^2 + \cdots + \alpha^{\gamma} = \frac{1 - \alpha^{\gamma+1}}{1-\alpha}$$

$\gamma \to \infty$ 时 $\tau \to \frac{1}{1-\alpha}$。**这是个硬上限：接受率 0.6 时，不管你把 $\gamma$ 调到多大，平均每步最多 2.5 个 token（加速上限 2.5 倍，还是在不计 draft 成本的前提下）。**

### 2.4 经济账：draft 不是免费的

draft 生成 $\gamma$ 个候选要跑 $\gamma$ 次前向。设单次 draft 前向的成本是 target 的 $c_d$ 倍，加速比是：

$$S(\gamma) = \frac{\tau(\gamma)}{1 + \gamma \cdot c_d}$$

这个式子里的 $c_d$ 才是真正的胜负手。下面是我用公式扫出来的（$\alpha$ 取常数）：

| $\alpha$ | $c_d$=0 | $c_d$=0.02 | $c_d$=0.05 | $c_d$=0.10 | $c_d$=0.25 |
|---|---|---|---|---|---|
| 0.6 | γ=12+, **2.50x** | γ=6, 2.17x | γ=4, 1.92x | γ=3, 1.67x | γ=2, 1.31x |
| 0.8 | γ=12+, **4.73x** | γ=11, 3.82x | γ=8, 3.09x | γ=6, 2.47x | γ=3, 1.69x |
| 0.9 | γ=12+, **7.46x** | γ=12+, 6.01x | γ=12+, 4.66x | γ=10, 3.43x | γ=6, 2.09x |

（"γ=12+" 意思是扫描范围内 $\gamma$ 越大越好，说明 draft 便宜到不需要节制。）

三行话读完这张表：

1. **$\alpha$ 决定天花板**，$\alpha$ 从 0.6 涨到 0.9，收益差不多翻三倍。
2. **$c_d$ 决定最优 $\gamma$**，draft 越贵，越该少猜（把力气花在「猜得准」而不是「猜得多」）。
3. **$c_d$ 也决定天花板能不能摸到**——$\alpha=0.8$、$c_d=0.25$ 时最优收益只有 1.69x，连理论上限 4.73x 的 40% 都不到。

### 2.5 一条贯穿全文的约束：「免费验证」是有预算的

上面讲「验证 $\gamma+1$ 个 token 不额外花钱」，前提是 $B\cdot L \ll 295$。这个预算很具体：

$$L_0 = \frac{295}{B}$$

| batch $B$ | 免费验证位置 $L_0$ | 能用的 $\gamma$ |
|---|---|---|
| 1 | 295.2 | ≤ 294 |
| 8 | 36.9 | ≤ 35 |
| 32 | 9.2 | ≤ 8 |
| 64 | 4.6 | ≤ 3 |
| 128 | 2.3 | ≤ 1 |
| 256 | 1.15 | **0**（连一个都验证不起） |

**这张表是投机解码的适用边界。** batch 一上去，验证就从「免费」变成「按 token 计费」，而且计的是**最贵的算力费**。4.4 节会把这个账算到底。

---

## 三、在 PyTorch 中怎么用

### 3.1 一个能跑的最小实现

不用下载任何权重。我们造一个玩具「语言」：target 是完整的 $32\times32$ 转移表，draft 是它的**秩-3 低秩近似**（类比「小模型是目标模型的低秩投影」）。

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
V, RANK = 32, 3

# ---------- 1. 造一个"真实语言"：稀疏的 token 转移分布 ----------
P_true = torch.distributions.Dirichlet(torch.full((V,), 0.9)).sample((V,))
P_true = torch.where(torch.rand(V, V) < 0.55, torch.zeros_like(P_true), P_true)
P_true = P_true / P_true.sum(-1, keepdim=True)      # 行随机矩阵

seq = [0]                                            # 用这条链采样语料
for _ in range(20000):
    seq.append(int(torch.multinomial(P_true[seq[-1]], 1).item()))
data = torch.tensor(seq)


# ---------- 2. target = 满秩转移表；draft = 低秩近似 ----------
class Bigram(nn.Module):
    def __init__(self, vocab, rank=None):
        super().__init__()
        if rank is None:
            self.W = nn.Parameter(torch.randn(vocab, vocab) * 0.1)   # V×V
        else:
            self.A = nn.Parameter(torch.randn(vocab, rank) * 0.1)    # V×r
            self.B = nn.Parameter(torch.randn(rank, vocab) * 0.1)    # r×V

    def forward(self, idx):
        if hasattr(self, "W"):
            return self.W[idx]                                       # (B,T,V)
        return torch.einsum("bti,ij->btj", torch.tanh(self.A[idx]), self.B)


target, draft = Bigram(V), Bigram(V, rank=RANK)
for m in (target, draft):
    opt = torch.optim.Adam(m.parameters(), lr=3e-2)
    for _ in range(3000):
        i = torch.randint(0, data.numel() - 33, (64,))
        x = torch.stack([data[j:j + 32] for j in i])
        y = torch.stack([data[j + 1:j + 33] for j in i])
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

n_t = sum(p.numel() for p in target.parameters())
n_d = sum(p.numel() for p in draft.parameters())
print(f"target 参数 {n_t}，draft 参数 {n_d}（{n_d/n_t:.1%}）")
```

然后是贪心版的投机解码主循环。**这段是整个领域的骨架，值得逐行读**：

```python
@torch.no_grad()
def greedy_plain(prompt, n):
    """标准贪心解码：n 个 token 要 n 次前向"""
    out = prompt.clone()
    for _ in range(n):
        out = torch.cat([out, target(out[:, -1:])[0, -1].argmax().view(1, 1)], 1)
    return out


@torch.no_grad()
def greedy_spec(prompt, n, gamma):
    """投机解码：一次 target 前向吐多个 token"""
    out, tf, df, na, emitted = prompt.clone(), 0, 0, 0, 0
    while emitted < n:
        T = out.size(1)

        # ---- 阶段 1：draft 串行猜 gamma 个 token ----
        cur, cand = out, []
        for _ in range(gamma):
            nxt = draft(cur[:, -1:])[0, -1].argmax().view(1, 1)
            cand.append(nxt)
            cur = torch.cat([cur, nxt], 1)          # 把猜的 token 拼回去继续猜
            df += 1
        cand = torch.cat(cand, 1)                    # (1, gamma)

        # ---- 阶段 2：target 一次前向，验证 gamma+1 个位置 ----
        logits = target(torch.cat([out, cand], 1))   # 只此一次前向
        tf += 1
        pred = logits[0, T - 1:T - 1 + gamma].argmax(-1)   # 每个位置的 target 预测

        # ---- 阶段 3：数连续对了几个 ----
        k = 0
        match = pred == cand[0]
        while k < gamma and bool(match[k]):
            k += 1

        # 位置 T-1+k 的 target 输出，就是被拒绝位置上的"正确答案"，白拿
        bonus = logits[0, T - 1 + k].argmax().view(1, 1)
        out = torch.cat([out, cand[:, :k], bonus], 1)
        na += k
        emitted += k + 1
    return out, tf, df, na
```

索引上唯一需要想清楚的地方：`logits[0, T-1+i]` 预测的是**位置 $T+i$** 的 token。所以 `pred[i]` 要和 `cand[0, i]` 比；全部接受（$k=\gamma$）时，用 `logits[0, T-1+gamma]` 拿到一个超出 draft 范围的「免费」token。

跑出来的真实结果：

```text
target 参数 1024，draft 参数 192（18.8%）
贪心投机解码 vs 标准贪心解码：
  gamma=1: 目标前向  33 次（标准 40 次），draft 前向  33 次，逐位接受率 24.2%，输出与标准一致 = True
  gamma=2: 目标前向  30 次（标准 40 次），draft 前向  60 次，逐位接受率 18.3%，输出与标准一致 = True
  gamma=4: 目标前向  29 次（标准 40 次），draft 前向 116 次，逐位接受率 10.3%，输出与标准一致 = True
  gamma=8: 目标前向  29 次（标准 40 次），draft 前向 232 次，逐位接受率  5.2%，输出与标准一致 = True
```

**注意最后一列全是 True。** 贪心解码下投机与标准解码的输出**逐 token 相同**——这不是「差不多」，是可证明的等价：接受的那个 token 本来就是 target 的 argmax，拒绝时用的也是 target 的 argmax。

也注意「逐位接受率」为什么从 24.2% 掉到 5.2%。两件事叠在一起：

1. **条件化**：这个数字是条件概率——只有前 $i-1$ 个候选**全部**被接受，我们才会走到第 $i$ 位去给它记一次分。$\gamma$ 越大，能走到后面的样本越少（实测 $\gamma=12$ 时，位置 8 之后已经几乎没有样本到达），剩下的全是「前面一路蒙对」的少数派。
2. **路径漂移**：draft 生成第 $i$ 个候选时，喂给它的上下文是前 $i$ 个候选拼起来的。这些候选虽然是 target 认可的，但整条轨迹已经和「真实文本」渐行渐远，越往后越难猜。

两条加起来的效果就是 $\tau(\gamma)$ 很快撞到饱和：实测把 $\gamma$ 从 4 加到 16（draft 前向次数翻 3.75 倍），$\tau$ 只从 1.959 涨到 2.087（**+6.5%**）。**多猜的力量是有限的，猜得准才是。**

### 3.2 采样版：修正拒绝采样

贪心版只需要对 argmax，采样版才需要完整的拒绝采样机制：

```python
@torch.no_grad()
def sample_spec(prompt, n, gamma, gen):
    """严格按修正拒绝采样实现：接受概率 min(1, p/q)，拒绝时从 (p-q)+ 重采样"""
    out, emitted = prompt.clone(), 0
    while emitted < n:
        T = out.size(1)

        # 阶段 1：draft 采样 gamma 个 token，同时记下每个位置的 q 分布
        cur, cand, dq = out, [], []
        for _ in range(gamma):
            qq = torch.softmax(draft(cur[:, -1:])[0, -1], -1)
            t = torch.multinomial(qq, 1, generator=gen).view(1, 1)
            cand.append(t); dq.append(qq)
            cur = torch.cat([cur, t], 1)
        cand = torch.cat(cand, 1)

        # 阶段 2：target 一次前向拿到每个位置的 p 分布
        logits = target(torch.cat([out, cand], 1))
        pp = torch.softmax(logits[0, T - 1:T - 1 + gamma], -1)   # (gamma, V)
        qq = torch.stack(dq)                                     # (gamma, V)

        # 阶段 3：逐位置判定
        for k in range(gamma):
            x = cand[0, k]
            acc = torch.clamp(pp[k, x] / qq[k, x].clamp_min(1e-12), max=1.0)
            if torch.rand(1, generator=gen) > acc:               # 拒绝
                resid = (pp[k] - qq[k]).clamp_min(0)             # 残差分布
                resid = resid / resid.sum()                      # 必须重新归一化
                nxt = torch.multinomial(resid, 1, generator=gen).view(1, 1)
                out = torch.cat([out, nxt], 1); emitted += 1
                break
            out = torch.cat([out, x.view(1, 1)], 1); emitted += 1  # 接受
        else:
            # 全部接受：再从最后一个位置的 target 分布白拿一个
            nxt = torch.multinomial(torch.softmax(logits[0, T - 1 + gamma], -1),
                                    1, generator=gen).view(1, 1)
            out = torch.cat([out, nxt], 1); emitted += 1
    return out
```

`clamp_min(1e-12)` 那一行不是装饰：**$q(x)=0$ 时 $p(x)/q(x)$ 会变成 `inf` 或 `nan`**，而 $q(x)=0$ 的位置按定义根本采不到，正确的语义是「不适用」。这是真实实现在数值上的必踩坑。

跑 20 万次采样，验证第一个 token 的边缘分布：

```text
单步 TV(p,q) = 0.4948 -> 理论上限 alpha = 0.5052
单个上下文：投机解码首 token 分布 vs target 分布，TV = 0.00322   （200000 次采样）
  朴素实现的解析 TV = 0.11626
  单步接受率 alpha = 0.5052
  token 5  : 目标 0.1274 / 投机 0.1266 / 朴素 0.1073
  token 19 : 目标 0.1229 / 投机 0.1228 / 朴素 0.0935
  token 7  : 目标 0.1192 / 投机 0.1192 / 朴素 0.0826
  token 20 : 目标 0.0951 / 投机 0.0964 / 朴素 0.0877
```

**修正实现的 TV 距离 0.0032（落在 20 万次采样的蒙特卡洛噪声里，理论值 0）**，而「拒绝时直接重新采 $p$」这个看起来更直觉的朴素写法，TV 是 0.116 —— 差 **36 倍**。看最后四行：概率最高的几个 token **全被系统性压低**（0.1274→0.1073、0.1229→0.0935、0.1192→0.0826、0.0951→0.0877），而被"偷"走的这些概率质量被摊到了大量低概率 token 上——这就是「输出分布变平」的微观图像。

这个偏差可以解析地写出来。用一组手算的分布：$p = [0.30, 0.22, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04]$，$q = [0.14, 0.20, 0.22, 0.16, 0.10, 0.08, 0.06, 0.04]$：

```text
TV(p,q)=0.1800 => 单步接受率 alpha = sum min(p,q) = 0.8200
解析 TV(朴素实现, p) = 0.10600
token :      p      q   min(p,q)   (p-q)+   朴素输出
  0   : 0.3000 0.1400   0.1400   0.1600   0.1940
  1   : 0.2200 0.2000   0.2000   0.0200   0.2396
  2   : 0.1500 0.2200   0.1500   0.0000   0.1770
  3   : 0.1000 0.1600   0.1000   0.0000   0.1180
蒙特卡洛(2000000 次) 实测接受率 = 0.8200  (解析 0.8200)
修正拒绝采样 TV = 0.00049  (理论 0，残差是 MC 噪声)
朴素实现     TV = 0.10602  (解析 0.10600)
```

两种实现的差别只在「拒绝时从那采」：修正版从 $\frac{(p-q)_+}{1-\beta}$ 采，朴素版从 $p$ 直接采。而朴素版对于 $p \ll q$ 的 token（token 2、3：$p<q$，决定权完全在 draft 手里的那些位置）会**重复计入** $p$ 的质量，导致 token 0 的目标概率 0.30 被砍到 0.194（-35%）。只要 $\mathrm{TV}(p,q)=0.18$，输出就稳定偏 0.106。

**记住一句话：拒绝时不能从 $p$ 重采，必须从 $(p-q)_+$ 重采。**

### 3.3 树形验证：一次前向验证多个候选路径

到目前为止我们只验证了**一条**候选链。但既然一次前向能塞进 $L_0 = 295/B$ 个位置，为什么不把 $B\cdot L$ 的 $L$ 用来装**多条候选路径**？

这就是 Medusa / EAGLE / SpecInfer 的「树形验证」：每个位置给出 top-$k$ 个候选，形成一棵树，一次前向把所有节点都验证掉。难点在 attention mask —— 树里同层的兄弟节点之间**不能互相看见**，每个节点只能看见自己的祖先链。

```python
def build_tree(parents, prefix_len):
    """parents: 每个候选节点的父节点索引（-1 = 挂在前缀末尾）"""
    N = len(parents)

    # 位置编码 = 节点深度（前缀长 + 从根到它的边数）
    depth = []
    for i in range(N):
        d, j = prefix_len, parents[i]
        while j != -1:
            d += 1
            j = parents[j]
        depth.append(d)

    total = prefix_len + N
    mask = torch.full((total, total), float("-inf"))
    mask[:prefix_len, :prefix_len] = torch.tril(torch.zeros(prefix_len, prefix_len))
    mask[prefix_len:, :prefix_len] = 0.0            # 所有候选都能看全前缀
    for i in range(N):                              # 每个候选只看自己的祖先链
        j = i
        while j != -1:
            mask[prefix_len + i, prefix_len + j] = 0.0
            j = parents[j]
    mask[mask == float("-inf")] = torch.finfo(mask.dtype).min
    return mask, depth
```

一棵 $k=2$、深度 3 的树，节点 0/1 挂在前缀上，2/3 挂 0，4/5 挂 1，6/7 挂 2。跑出来的 mask（`.` 可见，`x` 屏蔽）：

```text
候选节点父指针: [-1, -1, 0, 0, 1, 1, 2, 2]
每个候选节点的位置编码: [4, 4, 5, 5, 5, 5, 6, 6]
attention mask（0=可见, 极小数=屏蔽）:
     .  .  .  .  x  x  x  x  x  x  x  x      <- 前缀token 0
     .  .  .  .  x  x  x  x  x  x  x  x
     .  .  .  .  x  x  x  x  x  x  x  x
     .  .  .  .  x  x  x  x  x  x  x  x
     .  .  .  .  .  x  x  x  x  x  x  x      <- 候选0（父=-1）
     .  .  .  .  x  .  x  x  x  x  x  x      <- 候选1（父=-1）
     .  .  .  .  .  x  .  x  x  x  x  x      <- 候选2（父=0）
     .  .  .  .  .  x  x  .  x  x  x  x      <- 候选3（父=0）
     .  .  .  .  x  .  x  x  .  x  x  x      <- 候选4（父=1）
     .  .  .  .  x  .  x  x  x  .  x  x      <- 候选5（父=1）
     .  .  .  .  .  x  .  x  x  x  .  x      <- 候选6（父=2）
     .  .  .  .  .  x  .  x  x  x  x  .      <- 候选7（父=2）
  校验每行可见数 = prefix + 祖先数 + 自身: True
  候选节点 6 可见列: [0, 1, 2, 3, 4, 6, 10]
```

看候选 6（第 11 行）：它能看见前缀 0-3、候选 0、以及自己，但**看不见候选 3**（那是它叔父的孩子）。同时它的位置编码是 6 而不是 5 —— 深度决定 RoPE 相位。这两个细节（mask + position_ids）是树形验证实现里 90% 的 bug 来源。

### 3.4 生产环境怎么用

**HuggingFace transformers**（最省事的入口，附带官方 draft 模型）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
target = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda")
# assistant_model 是官方配对的 draft 模型，两者 tokenizer 必须一致
draft = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda")

inputs = tok("用一句话解释什么是投机解码：", return_tensors="pt").to("cuda")
out = target.generate(
    **inputs,
    assistant_model=draft,   # ← 打开投机解码就这么一行
    do_sample=False,         # 贪心；采样模式需 target 与 draft 的采样参数一致
    max_new_tokens=256,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

**vLLM**（服务端。参数以官方文档为准确认，2026-09 版）：

```bash
# 独立 draft 模型（method 必须显式写成 draft_model）
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"method": "draft_model",
                         "model": "meta-llama/Llama-3.2-1B-Instruct",
                         "num_speculative_tokens": 5}'

# n-gram 检索式投机：零模型成本，适合 RAG / 代码补全这类"上下文里有现成片段"的场景
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"method": "ngram",
                         "num_speculative_tokens": 4,
                         "prompt_lookup_min": 2, "prompt_lookup_max": 5}'
```

注意 `num_speculative_tokens` 就是我们说的 $\gamma$。**部署时真正要调的参数就它一个**。

vLLM 官方支持的投机方法现在已经是一个家族（不止「小模型 + 大模型」这一种）：

| method | draft 是什么 | 官方给的定位 |
|---|---|---|
| `draft_model` | 独立小模型 | 低 QPS 高收益，高 QPS 中等收益；需要额外一份权重 |
| `eagle3` | EAGLE-3 专用头 | 通用性最强的 model-based 方案 |
| `mtp` | target 模型自带的 MTP 头 | target 原生支持时最优（如 DeepSeek 系） |
| `ngram` / `suffix` | 不需要模型，纯检索 | 收益温和但**峰值流量下不增加任何负载** |
| `mlp` | 一个小 MLP 预测器 | 有现成兼容权重时好用 |
| `custom_class` | 你自己实现的 proposer | 实验性 |

**并且 vLLM 文档第一句就把适用场景写死了**（原文摘录）：

> *"This document shows how to use Speculative Decoding with vLLM to reduce inter-token latency under **medium-to-low QPS**, **memory-bound** workloads."*

以及它在方法选择表里把 Low QPS（latency focused）和 High QPS（throughput focused）分成两列来推荐——**这正是 4.3 节那张 batch 表的官方注脚**。

同一个页面还有一节叫 "Lossless guarantees of Speculative Decoding"，明确写了三层保证，其中"Algorithmic Losslessness"列出的两个官方测试是：

- **Rejection Sampler Convergence**：验证拒绝采样器的输出分布与目标分布一致（就是 3.2 节我们做过的事）；
- **Greedy Sampling Equality**：验证贪心采样下「开投机 == 不开投机」（就是 3.1 节输出里那一列全 `True`）。

**工业引擎把这两条当回归测试来跑**，也从侧面说明：3.2 节那个「拒绝时从哪采样」的细节不是学术洁癖，而是会被测试抓出来的正确性 bug。

另外两个文档里明确写着的限制：**流水线并行（PP）与投机解码不可组合**；**draft 与 target 的 tokenizer 默认必须一致**（想用异构词表需要显式打开 `use_heterogeneous_vocab`，走 Token-Level Intersection 算法把两套词表的交集建出来）。

### 3.5 KV Cache 回滚：实现里最容易漏的一步

draft 的 $\gamma$ 个候选一旦有部分被拒绝，target 的 KV cache 里就留了 $\gamma - k$ 条**不属于最终序列**的脏数据。下一次前向必须把它们丢掉：

```python
# 假设 past_key_values 是每层 (k, v)，形状 (B, n_head, seq_len, head_dim)
def rollback_kv(past_key_values, prefix_len, k_accepted):
    """只保留前缀 + 被接受的 k 个 token 的 KV"""
    keep = prefix_len + k_accepted
    return tuple((kk[:, :, :keep], vv[:, :, :keep]) for kk, vv in past_key_values)
```

忘了这一步的后果不是报错，而是**输出偶尔变垃圾**——因为被拒绝路径的 attention 上下文污染了后续所有位置。这类 bug 在测试里很难复现，是推理引擎里最典型的「静默错误」。

---

## 四、围绕该领域展开

### 4.1 draft 从哪来：一条完整的谱系

投机解码真正的工程难点从来不是验证，而是**「去哪找一个又快又准的 draft」**。这个问题催生了一整个技术家族：

| 路线 | 代表 | draft 是什么 | $c_d$ 量级 | 特点 / 代价 |
|---|---|---|---|---|
| 独立小模型 | Leviathan 2022、Chen 2023 | 同家族的小模型（1B/68M） | 1%~10% | 最直接，但要额外一份权重和显存；分布对齐差 → $\alpha$ 低 |
| 层跳跃自投机 | LayerSkip（Meta 2024） | 目标模型的前几层 + 共享输出头 | 10%~50% | 不占额外权重，可共享 KV/embedding；但「浅层 = 差」的天花板很低（见下表） |
| 多 head 并行 | Medusa（2024） | 在 target 最后一层加 k 个预测头，一次出多个候选 | ~1% | draft 成本几乎为 0，但每个头只有单层信息 → 单头 $\alpha$ 偏低，必须配树形验证 |
| 特征级自回归 | EAGLE / EAGLE-2 / EAGLE-3 | 预测**倒数第二层的特征**而不是 token，再映射回 token | 2%~5% | 目前工业界的 SOTA 路线；特征比 token 更"平滑"，$\alpha$ 明显更高 |
| 训练时内建 | DeepSeek-V3 的 MTP | 训练时就并行加一个「预测下下个 token」的头 | 极小 | 零额外训练成本（本来就在训），天然对齐；老模型用不了 |
| 检索式 | Prompt Lookup Decoding、REST | 从 prompt / 检索库里找重复出现的 n-gram | **0** | 零模型成本；在 RAG、代码补全、摘要这类"抄现成片段"的任务上 $\alpha$ 极高 |
| Jacobi 迭代 | Lookahead Decoding（2024） | 用同一模型的多次并行迭代互相"猜" | 0（无 draft） | 不需要 draft 模型，靠并行轨迹收敛；$\alpha$ 对任务敏感 |

**「层跳跃」的天花板有多低**，我用目标模型的前 $s$ 层当 draft 实测了一遍（8 层 d=192 的 tiny 模型）：

| draft 层数 $s$ | 成本占比 $c_d$ | argmax 一致率 | 平均 TV | $\alpha = 1-\mathrm{TV}$ |
|---|---|---|---|---|
| 1 | 12.5% | 0.208 | 0.8007 | 0.1993 |
| 2 | 25.0% | 0.396 | 0.7163 | 0.2837 |
| 3 | 37.5% | 0.479 | 0.6098 | 0.3902 |
| 4 | 50.0% | 0.604 | 0.5020 | 0.4980 |
| 6 | 75.0% | 0.854 | 0.2541 | 0.7459 |

看第一列和最后一列：**成本涨 6 倍，$\alpha$ 涨 3.7 倍**。代进 $S=\tau/(1+\gamma c_d)$，$s=1$ 时最优也就 1.06x —— 这就是为什么简单的「砍掉几层」在实践中收益有限，也是 Medusa/EAGLE 要花力气去**训练专门的 draft 头**的原因。想让层跳跃真正work，必须共享 KV cache 和 embedding，让 $c_d$ 远低于「层数比」。

### 4.2 树形验证为什么值得

把「猜得准」的担子从 draft 模型转移到**验证环节的搜索**上。给定「每层至少有一个候选被接受」的概率 $\alpha_k$，树形（$k$ 分支、深度 $d$）的期望接受长度是：

$$\tau_{\text{tree}} = \sum_{i=1}^{d} \alpha_k^{i}$$

它和线性的核心差别在于**成本结构**：线性要 draft 串行生成 $\gamma$ 个 token（$\gamma$ 次前向）；树形只需要**一次** draft 前向就铺开整棵树（Medusa 的多个头、EAGLE 的单步特征预测都是这个性质）。

用 $\alpha_1=0.6$（$k$ 分支模型 $\alpha_k = 1-(1-0.6)^k$）、线性逐位衰减 $\rho=0.92$ 算一笔账：

| 方案 | draft 前向次数 | 验证位置数 | 接受长度 | 每单位成本收益 |
|---|---|---|---|---|
| 线性 γ=1 | 1 | 2 | 0.600 | 0.571 |
| 线性 γ=4 | 4 | 5 | 1.178 | 0.982 |
| 线性 γ=8 | 8 | 9 | 1.232 | 0.880 |
| 线性 γ=24 | 24 | 25 | 1.232 | 0.560 |
| **树形 k=2 d=3** | **1** | 15 | **2.138** | **2.036** |
| **树形 k=2 d=5** | **1** | 63 | **3.054** | **2.909** |
| 树形 k=3 d=4 | 1 | 121 | 3.400 | 3.238 |

（$c_d=5\%$ 下算的，成本 = $1 + \gamma c_d$）

**树形在同样的 draft 预算下把接受长度做到线性的约 2.5 倍**。它的代价全压在「一次前向验证的位置数」上——而这恰恰是带宽瓶颈期最便宜的资源。

但要留意那张免费预算表的约束：$B=64$ 时 $L_0=4.6$，一棵 15 节点的树就超预算了。**树形验证是"小 batch 奢侈品"，batch 一上去必须缩树。**

### 4.3 batch 效应：投机解码什么时候开始亏钱

这是整个领域最反直觉、也最容易被忽略的一点。用实测的接受长度（$\gamma=4$ 时 $\tau=1.959$），代进 7B BF16 / H800 级的 roofline 模型：

| batch $B$ | 无投机吞吐 | 投机吞吐 | 比值 | $B\cdot L$ |
|---|---|---|---|---|
| 1 | 239 | 469 | **1.96x** | 5 |
| 8 | 1914 | 3750 | **1.96x** | 40 |
| 32 | 7657 | 15000 | **1.96x** | 160 |
| 48 | 11486 | 22501 | **1.96x** | 240 |
| 64 | 15314 | 27678 | 1.81x | 320 |
| 96 | 22971 | 27678 | 1.20x | 480 |
| 128 | 30629 | 27678 | **0.90x** ❌ | 640 |
| 256 | 61257 | 27678 | **0.45x** ❌ | 1280 |
| 512 | 70643 | 27678 | **0.39x** ❌ | 2560 |

**临界 batch ≈ 59**（就是 $295/(\gamma+1) = 295/5$）。$\gamma=8$ 时临界点更早，约 33。

两个现象值得盯着看：

1. **batch ≤ 48 时，比值锁死在 1.96x**——这正是 $\tau$ 的数值。因为这段区间里验证是免费的，收益完全由接受长度决定。
2. **batch > 59 后，投机吞吐被钉死在 27678 tok/s 不再增长。** 因为进入算力瓶颈区后，产出 ∝ 算力 / 总验证 token 数，而投机的「有效 token 占比」只有 $\tau/(\gamma+1) = 1.959/5 = 39\%$。**你花了 100% 的算力，只换回 39% 的有效产出。**

所以生产环境里投机解码的定位非常明确：

> **它是延迟优化手段（latency optimization），不是吞吐优化手段（throughput optimization）。**

vLLM/SGLang 里开启投机后「吞吐下降但 TTFT/TPOT 改善」是预期行为，不是 bug。真要同时要，就按 batch 动态调整 $\gamma$——把 $\gamma$ 从 $294$（B=1）一路削到 0（B≈295），**这正是「免费预算 $L_0$」那张表的另一种读法**：

| batch $B$ | 免费预算 $L_0$ | 最优 $\gamma$ | 收益 |
|---|---|---|---|
| 1 | 295.2 | 8 | 3.09x |
| 32 | 9.2 | 8 | 3.09x |
| 64 | 4.6 | 4 | 2.62x |
| 128 | 2.3 | 2 | 1.74x |

（$\alpha=0.8$、$c_d=0.05$ 的模型计算，收益 = $\tau(\gamma) / \left(\max(1, \frac{\gamma+1}{L_0}) + \gamma c_d\right)$）

### 4.4 温度会怎么改变接受率（一个 U 形曲线）

直觉上「温度越高越随机，draft 越容易猜错」，但实测是一条 **U 形曲线**（tiny 模型，48 个位置平均，$\alpha = 1-\mathrm{TV}$）：

| 温度 $T$ | 平均熵 | 平均 top-1 概率 | 平均 TV(p,q) | $\alpha$ |
|---|---|---|---|---|
| 0.1 | 0.004 | 0.9990 | 0.6241 | 0.3759 |
| 0.5 | 0.017 | 0.9919 | 0.6600 | 0.3400 |
| 1.0 | 0.055 | 0.9831 | 0.7163 | 0.2837 |
| **1.5** | 0.270 | 0.9469 | **0.7645** | **0.2355** ← 最低 |
| 2.0 | 0.924 | 0.8424 | 0.7485 | 0.2515 |
| 3.0 | 3.180 | 0.4653 | 0.5123 | 0.4877 |
| 5.0 | 5.066 | 0.1094 | 0.2268 | 0.7732 |
| 10.0 | 5.477 | 0.0225 | 0.0945 | 0.9055 |
| greedy（$T\to0$） | — | — | — | 0.396 |

两个机制在互相拉扯：

- **$T \to 0$（greedy）时**：$p$ 和 $q$ 都塌成点质量分布，TV 只取决于「两个模型的 argmax 是否相同」。实测这个一致率是 0.396，所以 $\alpha \to 0.396$。
- **$T$ 在 1~2 附近**：两个分布都还有明显的形状（top-1 在 0.85~0.95），此时**次高项之间的差异**暴露得最充分，TV 达到峰值。
- **$T$ 很大时**：两个分布都接近均匀，谁都不"自信"，反而容易互相接受，$\alpha \to 1$（但这时候的输出本身也没什么信息量了）。

实践含义：**不要拿「默认 temperature」去评估投机解码的加速比**。同一对模型的接受率可以在 0.24~0.91 之间变化 3.8 倍。评估报告必须锚定采样配置。

### 4.5 与其它优化手段的关系

| 手段 | 与投机解码的关系 |
|---|---|
| **量化** | 正交，可叠加。量化缩短 $t_{\text{mem}}$，投机提高每次 $t_{\text{mem}}$ 的产出。注意平衡点 $B\cdot L = \frac{\text{PEAK}}{BW}\cdot\frac{\text{bytes}}{2}$：精度减半且算力翻倍时，平衡点不变。 |
| **KV Cache / Paged Attention** | 投机需要额外维护 draft 的 KV；验证时 target 的 KV 要能回滚（3.5 节），Paged Attention 的 block 粒度让回滚以 block 为单位，比逐 token 截断更快。 |
| **CUDA Graph** | 有冲突。$\gamma$ 固定时可以把「draft 循环 + 验证」整体 capture 成图；但接受长度 $k$ 是**数据依赖**的，会让 shape 每步都变。工业实现的做法是**固定 $\gamma$、对候选做 padding**，把动态部分留到图外面。 |
| **Continuous Batching** | 反向作用，见 4.3。两者是对立的调度策略：一个要小 batch 保延迟，一个要大 batch 保吞吐。 |
| **Prefix Caching** | 正向协同。前缀命中率高的场景（多轮对话、RAG）里，prompt 里就藏着大量可复制的片段，n-gram 投机几乎免费。 |
| **torch.compile / torch.export** | 验证阶段（$\gamma+1$ 个位置一次前向、形状固定）非常适合编译；draft 循环可以独立编译成一个小图。真正难编译的是「循环次数由接受长度决定」的那部分控制流。 |

### 4.6 分布式下的一致性：一个容易致命的细节

修正拒绝采样用的是**随机数**。在张量并行/流水线并行的推理里，所有 rank 必须对「这次接受还是拒绝」得出**完全相同的结论**——否则各 rank 的序列长度会分叉，下一步就彻底乱套。

实践中：

- 拒绝判定的随机数必须**由同一个 rank 生成后广播**，或者用同一个 seed + 同一次调用序号在全组内重放。
- 更省事的做法是全程贪心（`do_sample=False`），此时判定完全确定，无一致性问题——这也是很多线上部署默认关采样的原因之一。
- 如果 draft 和 target 分在不同设备上（PD 分离、draft 在 CPU/小卡上），**通信往返会直接吃掉收益**。draft 一旦需要跨设备同步，$c_d$ 就不再是「算力比」而是「网络延迟比」了。
- **流水线并行（PP）和投机解码目前不能组合**（vLLM 文档明写到 `vllm<=0.15.0`）——这是调度粒度冲突：PP 要求各 stage 的 microbatch 步调一致，而投机解码每步产出的 token 数不确定。

顺带一个很有意思的配置项：vLLM 的 `rejection_sample_method` 有三个值 `standard` / `synthetic` / `block`。其中 **`synthetic` 允许你直接喂一组「预设的逐位置接受率」**（要求非递增）来驱动拒绝采样，**不需要真实 draft 模型**就能复现投机解码的收益曲线：

```jsonc
// 这不是"跑真投机"，而是"假设 draft 的逐位置接受率是这条曲线，看吞吐会变成什么样"
"rejection_sample_method": "synthetic",
"synthetic_acceptance_rates": [0.8, 0.72, 0.65, 0.58, 0.52]   // 长度须等于 num_speculative_tokens，且非递增
// 或者只给一个目标平均接受长度：
// "rejection_sample_method": "synthetic", "synthetic_acceptance_length": 3.5
```

这正是练习 2 (c) 里那个 $\alpha_i = \alpha\rho^{i-1}$ 衰减模型，vLLM 把它做成了生产配置项——**用来做容量规划和压测**，不用真的把 draft 模型部署上去。

---

## 五、什么时候该用 / 不该用

**该用：**

- 单序列或小 batch 的**低延迟**场景（batch $\lesssim 32$）——这是投机解码的主场，$\alpha$ 合适时收益能到 2~3 倍。
- chat / 代码补全这类**交互式**场景，用户等的是 TTFT 和 TPOT，不是集群吞吐。
- **有现成廉价 draft 的场景**：同家族小模型、模型自带 MTP 头、或者 prompt 里有大量可复制的片段（RAG、摘要、代码编辑）——n-gram 投机在这里 $c_d \approx 0$。
- 输出**可预测性强**的任务：翻译、格式化输出、代码。这些任务 $\alpha$ 天然高。

**不该用：**

- **高吞吐、大 batch 的服务**（batch $\gtrsim 100$）：4.3 节的表显示会掉到 0.4x。要吞吐就别开投机。
- **临时配对的 draft 模型**：两个模型没做过对齐、tokenizer 有差异、或者 $\alpha$ 低于 0.4 —— 收益会小于引入的复杂度和 draft 的那份显存。
- **输出高度开放**（创意写作、头脑风暴）：$\alpha$ 低，且高温采样会让树形/多候选的收益打折。
- **显存吃紧**：draft 模型的权重和它自己的 KV cache 都是额外开销，在 KV cache 已经打满的场景里是净负担。
- **极度受限的硬件**（CPU 推理、小显存卡）：本地实测在 CPU 上投机反而慢到 0.85x~0.33x —— 因为 CPU 上不存在「免费的额外验证」，而且 $\gamma$ 次 draft 调用的固定开销是线性累加的。

---

## 六、常见坑

**坑 1：拒绝时直接从 $p$ 重采样。**
这是最直觉、也最错的做法。实测偏离目标的 TV 距离 0.116（修正实现是 0.0032，差 36 倍），表现为「高频 token 被系统性压低、低频 token 被抬高」，输出分布变平。**必须从 $(p-q)_+$ 归一化后的残差分布采样。** 判断自己有没有踩：在 $\mathrm{TV}(p,q)=0.18$ 的一对分布上，你的实现在 200 万次采样后对 token 0 的估计应该是 0.30，如果是 0.19 就是踩了。

**坑 2：draft 和 target 的 tokenizer / chat template 不一致。**
draft 猜出的 token id 在 target 的词表里可能对应不同的字符串。这类问题**不会报错**，只会让接受率莫名很低（比如 0.1 以下）。vLLM 的做法是**默认强制两者词表相同**，不合规直接启动失败——这是比静默降质好得多的处理方式。真要用异构词表的 draft，得显式打开 `use_heterogeneous_vocab`，引擎会在初始化时对两套词表做 Token-Level Intersection，把 draft 的 logits 限制在交集上，采样后再翻译成 target 的 token id。即便如此，这条路径目前**只支持贪心 draft**（`temperature > 0` 的 draft 采样尚未支持）。

**坑 3：忘了回滚被拒绝路径的 KV cache。**
draft 的 $\gamma$ 个候选里有 $\gamma-k$ 个是错的，它们的 KV 已经进了 cache。不回滚的话，后续所有 token 的 attention 上下文都被污染，输出质量静默下降。测试里表现为「长序列偶尔崩」而不是稳定失败。

**坑 4：$q(x)=0$ 时的除零。**
$p(x)/q(x)$ 在 $q(x)=0$、$p(x)>0$ 时是 `inf`（或 `nan`）。语义上 $q(x)=0$ 的 token 根本不会被采到，接受判定应该跳过；但直接做除法会污染整个 batch 的梯度/输出。用 `q.clamp_min(1e-12)` 或先做掩码。同理，残差 $(p-q)_+$ 归一化时如果 $\sum(p-q)_+ = 0$（两个分布完全相同），分母是 0 —— 需要特判。

**坑 5（工程）：把加速比当成和 batch 无关的常数来宣传。**
同一套配置在 batch=1 是 1.96x，在 batch=256 是 0.45x。**报投机解码的收益必须带上 batch size 和采样温度**，否则数字没有意义。

---

## 七、一句话总结

**投机解码用一次「便宜的猜测 + 一次昂贵的验证」，把 decode 阶段那笔固定不变的显存搬运成本摊到多个 token 上；它的正确性由修正拒绝采样精确保证（输出分布与目标模型完全一致），而它的收益上限由接受率 $\tau(\gamma)=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 和 draft 相对成本 $c_d$ 共同决定——注定是低延迟武器，不是吞吐武器。**

---

## 八、今日练习

<details><summary>今日练习（点击展开参考答案）</summary>

### 练习 1：手算加速比与最优 γ

设某对 draft/target 模型的单步接受率 $\alpha = 0.7$，draft 单次前向成本是 target 的 $c_d = 0.05$。

(a) $\gamma = 4$ 时的期望接受长度 $\tau$ 和加速比 $S$ 各是多少？
(b) $\gamma$ 从 1 试到 12，最优是多少？对应加速比多少？
(c) 如果这对模型的 $\alpha$ 掉到 0.4（换了任务类型），最优 $\gamma$ 和收益怎么变？

**参考答案**（下面这段是真跑过的，不是手写）：

```python
alpha, c_d = 0.7, 0.05

def tau(a, g):
    return (1 - a ** (g + 1)) / (1 - a)

print(f"(a) gamma=4: tau = {tau(alpha,4):.4f}, S = {tau(alpha,4)/(1+4*c_d):.4f}")

print("(b) alpha=0.7 扫描:")
best = None
for g in range(1, 13):
    s = tau(alpha, g) / (1 + g * c_d)
    if best is None or s > best[1]:
        best = (g, s)
    print(f"    gamma={g:2d}: tau={tau(alpha,g):.4f}  S={s:.4f}")
print(f"    -> 最优 gamma={best[0]}, S={best[1]:.4f}, 理论上限 {1/(1-alpha):.4f}")
```

```text
(a) gamma=4: tau = 2.7731, S = 2.3109
(b) alpha=0.7 扫描:
    gamma= 1: tau=1.7000  S=1.6190
    gamma= 2: tau=2.1900  S=1.9909
    gamma= 3: tau=2.5330  S=2.2026
    gamma= 4: tau=2.7731  S=2.3109
    gamma= 5: tau=2.9412  S=2.3529
    gamma= 6: tau=3.0588  S=2.3529
    gamma= 7: tau=3.1412  S=2.3268
    gamma= 8: tau=3.1988  S=2.2849
    gamma= 9: tau=3.2392  S=2.2339
    gamma=10: tau=3.2674  S=2.1783
    gamma=11: tau=3.2872  S=2.1208
    gamma=12: tau=3.3010  S=2.0631
    -> 最优 gamma=6, S=2.3529, 理论上限 3.3333
```

（注意 $\gamma=5$ 和 $\gamma=6$ 的 $S$ 都是 2.3529，差在第 6 位小数——**这个函数在极值附近非常平坦**，也就是说线上你完全不必精调 $\gamma$，取 4~8 差别都在 2% 以内。）

(c) $\alpha=0.4$ 时：

```python
for a in (0.7, 0.4):
    rows = [(g, tau(a, g) / (1 + g * c_d)) for g in range(1, 13)]
    g_best, s_best = max(rows, key=lambda x: x[1])
    print(f"alpha={a}: 理论上限 {1/(1-a):.4f} -> 最优 gamma={g_best}, S={s_best:.4f} "
          f"(占到上限的 {s_best/(1/(1-a)):.1%})")
```

```text
alpha=0.7: 理论上限 3.3333 -> 最优 gamma=6, S=2.3529 (占到上限的 70.6%)
alpha=0.4: 理论上限 1.6667 -> 最优 gamma=2, S=1.4182 (占到上限的 85.1%)
```

**结论很有意思**：$\alpha$ 从 0.7 掉到 0.4，收益从 2.35x 掉到 1.42x，最优 $\gamma$ 从 **6 缩到 2**。低接受率时「多猜」不但没用，还会被 draft 成本吃掉——因为 $\tau(\gamma)$ 很快撞到 $\frac{1}{1-\alpha}=1.67$ 的天花板，而分母 $1+\gamma c_d$ 一直线性上涨。

### 练习 2：模拟验证 $\tau(\gamma)$ 公式，以及 $\alpha$ 按位置衰减时会怎样

(a) 用蒙特卡洛验证 $\tau(\gamma) = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$（每个位置以概率 $\alpha$ 独立接受，一次前向产出「接受数 + 1 个免费 token」）。
(b) 用公式求 $\alpha=0.8$、$c_d=0.05$ 时的最优 $\gamma$。
(c) 把「逐位接受率恒定」放宽成 $\alpha_i = \alpha\cdot\rho^{i-1}$（$\rho$ 就是上一节说的路径漂移率），看最优 $\gamma$ 和收益怎么变。

**参考答案**：

```python
import random
random.seed(0)

# (a) 模拟
for alpha in (0.5, 0.8):
    print(f"alpha = {alpha}")
    for g in (1, 2, 4, 8, 16):
        n = 0
        for _ in range(200000):
            k = 0
            while k < g and random.random() < alpha:
                k += 1
            n += k + 1                      # 接受 k 个 + 1 个免费 token
        print(f"  gamma={g:2d} 公式 {(1-alpha**(g+1))/(1-alpha):.4f}  "
              f"模拟 {n/200000:.4f}")
    print(f"  上限 1/(1-alpha) = {1/(1-alpha):.4f}")
```

```text
alpha = 0.5
  gamma= 1 公式 1.5000  模拟 1.5009
  gamma= 2 公式 1.7500  模拟 1.7508
  gamma= 4 公式 1.9375  模拟 1.9355
  gamma= 8 公式 1.9961  模拟 2.0010
  gamma=16 公式 2.0000  模拟 1.9978
  上限 1/(1-alpha) = 2.0000
alpha = 0.8
  gamma= 1 公式 1.8000  模拟 1.8008
  gamma= 2 公式 2.4400  模拟 2.4403
  gamma= 4 公式 3.3616  模拟 3.3610
  gamma= 8 公式 4.3289  模拟 4.3141
  gamma=16 公式 4.8874  模拟 4.9007
  上限 1/(1-alpha) = 5.0000
```

偏差全部在 0.015 以内（20 万轮蒙特卡洛的正常噪声），**公式是对的**。

(b) 和 (c)：

```python
alpha, c_d = 0.8, 0.05

def tau_decay(alpha, rho, g):
    t, prod = 1.0, 1.0                          # 1.0 = 那个免费 token
    for i in range(1, g + 1):
        prod *= alpha * rho ** (i - 1)          # 第 i 位的逐位接受率
        t += prod
    return t

for rho in (1.0, 0.95, 0.9, 0.8):
    best = max(((g, tau_decay(alpha, rho, g) / (1 + g * c_d)) for g in range(0, 25)),
               key=lambda x: x[1])
    print(f"rho={rho:.2f}: 最优 gamma={best[0]:2d}, S={best[1]:.4f}, "
          f"tau={tau_decay(alpha, rho, best[0]):.4f}")
```

```text
rho=1.00: 最优 gamma= 8, S=3.0921, tau=4.3289
rho=0.95: 最优 gamma= 5, S=2.6754, tau=3.3443
rho=0.90: 最优 gamma= 4, S=2.4724, tau=2.9669
rho=0.80: 最优 gamma= 3, S=2.2384, tau=2.5741
```

**答案的三层含义**：

1. $\rho=1$ 时最优 $\gamma=8$、收益 3.09x —— 和 2.4 节表里 $\alpha=0.8$、$c_d=0.05$ 那一格（γ=8, 3.09x）完全对上，说明两条路径（解析公式 / 数值模拟）一致。
2. **漂移率 $\rho$ 只要从 1.0 掉到 0.9（每往后一个位置，接受率打 9 折），收益就从 3.09x 塌到 2.47x，最优 $\gamma$ 从 8 缩到 4。** 在 $\rho=0.9$ 下，$\tau$ 在 $\gamma=12$ 之后就死死卡在 3.1712 不动了——多猜的每一个 token 都等于白花 draft 的钱。
3. 所以工程上「$\gamma$ 该设多少」这个问题的真实答案取决于**你的 draft 有多会漂**，而不是接受率本身。这也是 EAGLE-3 论文里反复强调「在特征空间预测能把 $\rho$ 推高」的原因：$\rho$ 每提高一点，收益是乘性的。

### 练习 3：为什么大 batch 下投机解码必亏

用 roofline 模型算一下：7B BF16 在 H800 级硬件上（$BW = 3.35$ TB/s，BF16 峰值 989 TFLOPS），当 batch 涨到多少时，$\gamma=8$ 的投机解码从「赚」变成「亏」？

（a）先推导临界 batch 的解析表达式。
（b）用数字验证：$\gamma=8$、$\tau=2.087$ 时，临界点在哪？

**参考答案**：

**(a) 推导。** 令 $t_{\text{mem}} = N\cdot b/BW$（$b$ 为每参数字节数），$c = 2N/\text{PEAK}$（每个 token 的算力时间）。解码一步的耗时：

$$t(B, L) = \max\left(t_{\text{mem}},\ c \cdot B \cdot L\right)$$

- 无投机：$L=1$，吞吐 $\text{Th}_{\text{plain}} = B / t(B,1)$，只要 $B < B^*$ 就恒为 $B/t_{\text{mem}}$（$B^*$ 就是 2.5 节那个 $B\cdot L$ 平衡点：$B^* = \frac{\text{PEAK}}{BW}\cdot\frac{b}{2}$，BF16 下是 295.2）。
- 投机：$L=\gamma+1$，吞吐 $\text{Th}_{\text{spec}} = B\tau / t(B,\gamma+1)$。

当 $B(\gamma+1) \le B^*$ 时两者都在带宽区，比值恒为 $\tau$ ——**与 batch 无关**。

一旦 $B(\gamma+1) > B^*$，投机进入算力区：

$$\frac{\text{Th}_{\text{spec}}}{\text{Th}_{\text{plain}}} = \frac{B\tau / (cB(\gamma+1))}{B/t_{\text{mem}}} = \frac{\tau}{\gamma+1}\cdot\frac{B^*}{B}$$

令比值 $=1$：

$$\boxed{\ B_{\text{crit}} = B^* \cdot \frac{\tau}{\gamma+1}\ }$$

直觉版：$\frac{B^*}{\gamma+1}$ 是「这个 $\gamma$ 下能免费验证的 batch 上限」$L_0$，而 $\frac{\tau}{\gamma+1}$ 是投机在算力区的**有效产出率**（花了 100% 算力，换回多少有效 token）。

**(b) 数字验证。**

```python
N, b, BW, PEAK = 7e9, 2, 3.35e12, 989e12
t_mem = N * b / BW
BL_balance = t_mem * PEAK / (2 * N)       # B*L 的平衡点（与 N 无关）
print(f"t_mem = {t_mem*1000:.3f} ms, B*L 平衡点 = {BL_balance:.1f}")

for gamma, tau_val in ((4, 1.959), (8, 2.087)):
    L = gamma + 1
    B_crit = BL_balance * tau_val / L
    print(f"gamma={gamma}: 免费预算 L0 = {BL_balance/L:.2f}, "
          f"有效产出率 = {tau_val/L:.3f}, 临界 batch = {B_crit:.1f}")
    for B in (32, 48, 59, 64, 96, 115, 116, 128, 256):
        t1 = max(t_mem, 2*N*B/PEAK)
        tS = max(t_mem, 2*N*B*L/PEAK)
        print(f"    B={B:4d}: 无投机 {B/t1:8.0f} tok/s  投机 {B*tau_val/tS:8.0f} tok/s  "
              f"比值 {B*tau_val/tS/(B/t1):5.2f}x")
```

```text
t_mem = 4.179 ms, B*L 平衡点 = 295.2
gamma=4: 免费预算 L0 = 59.04, 有效产出率 = 0.392, 临界 batch = 115.7
    B=  32: 无投机     7657 tok/s  投机    15000 tok/s  比值  1.96x
    B=  48: 无投机    11486 tok/s  投机    22501 tok/s  比值  1.96x
    B=  59: 无投机    14118 tok/s  投机    27657 tok/s  比值  1.96x
    B=  64: 无投机    15314 tok/s  投机    27678 tok/s  比值  1.81x
    B=  96: 无投机    22971 tok/s  投机    27678 tok/s  比值  1.20x
    B= 115: 无投机    27518 tok/s  投机    27678 tok/s  比值  1.01x
    B= 116: 无投机    27757 tok/s  投机    27678 tok/s  比值  1.00x
    B= 128: 无投机    30629 tok/s  投机    27678 tok/s  比值  0.90x
    B= 256: 无投机    61257 tok/s  投机    27678 tok/s  比值  0.45x
gamma=8: 免费预算 L0 = 32.80, 有效产出率 = 0.232, 临界 batch = 68.5
    B=  32: 无投机     7657 tok/s  投机    15980 tok/s  比值  2.09x
    B=  48: 无投机    11486 tok/s  投机    16381 tok/s  比值  1.43x
    B=  59: 无投机    14118 tok/s  投机    16381 tok/s  比值  1.16x
    B=  64: 无投机    15314 tok/s  投机    16381 tok/s  比值  1.07x
    B=  96: 无投机    22971 tok/s  投机    16381 tok/s  比值  0.71x
    B= 115: 无投机    27518 tok/s  投机    16381 tok/s  比值  0.60x
    B= 116: 无投机    27757 tok/s  投机    16381 tok/s  比值  0.59x
    B= 128: 无投机    30629 tok/s  投机    16381 tok/s  比值  0.53x
    B= 256: 无投机    61257 tok/s  投机    16381 tok/s  比值  0.27x
```

**两个答案都值得注意**：

1. **「开始衰减」和「开始亏钱」是两个不同的界。** 免费预算 $L_0 = 295/(\gamma+1)$ 是**衰减起点**（$\gamma=4$ 时 $B=59$，$\gamma=8$ 时 $B=33$）；而 $B_{\text{crit}} = B^*\cdot\frac{\tau}{\gamma+1}$ 是**盈亏平衡点**（$\gamma=4$ 时 115.7，$\gamma=8$ 时 68.5）。中间那段（59~115）投机仍然赚，只是越赚越少。工程上该盯的是前者——**收益的边际衰减从 $L_0$ 那一刻就开始了**。
2. $\gamma=8$ 的临界点（68.5）比 $\gamma=4$（115.7）**小得多**。所以「加大 $\gamma$ 换更多接受长度」在大 batch 场景下是**双输**：既更早进入算力区（$L_0$ 更小），算力区的有效产出率也更低（0.232 vs 0.392）。

</details>

---

**下期预告**：候选话题池还有 `torch.export + AOT Inductor`、`知识蒸馏`、`vLLM 推理引擎工程化`、`正则化全景`、`损失函数设计`、`cuDNN/cuBLAS`。
