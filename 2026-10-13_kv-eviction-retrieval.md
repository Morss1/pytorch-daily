# PyTorch 每日一课 · 第 035 期

## 推理期 KV 驱逐与检索：谁被丢掉，谁被找回

**日期** 2026-10-13　**难度** ⭐⭐⭐⭐　**前置知识** attention / KV cache 的基本概念（看过第 011 期更顺，不看也能读）　**预计阅读** 约 50 分钟

> **关联**
> - 第 032 期回答了「一个 token 的 KV 有多少字节」，第 033 期问「这些字节能不能压」，第 034 期问「能不能少看几条」——这一期问**最后一种省法：干脆丢掉**。
> - 第 034 期的选择信号是**训练出来的 indexer**；这一期的方法**全都不训练**，直接读模型自己现成的注意力分数。两条路线的分水岭在「谁来当裁判」。
> - 第 029 期讲前缀缓存的驱逐（基数树上的 LRU/LFU，块级），第 004 期讲显存分配器的驱逐（LRU，字节级）——**「有限缓存里丢什么」这个问题在逐层重复出现**，本期是它在「KV 条目」这一层的样子。

---

## 1. 这个领域解决什么问题

### 1.1 KV 是唯一只增不减的东西

模型权重是常量。激活是瞬时的。**只有 KV cache 会随着上下文线性长大，而且从来不缩小。**

用第 032 期量出的常数（7B / GQA-8 / bf16 = 128 KiB 每 token）：

| 上下文长度 | 单个请求的 KV | 一张 80GB 卡（扣掉权重 14GB + 运行时 4GB）能装几个 |
| --- | --- | --- |
| 8K | 1.00 GiB | 约 62 个 |
| 32K | 4.00 GiB | 约 15 个 |
| 128K | 16.00 GiB | 约 3 个 |
| 1M | 128.00 GiB | **装不下一个** |

1M 上下文那一行就是问题的全部：**权重没变，上下文变长 128 倍，KV 就涨 128 倍**，直到超过显卡。而推理服务里，KV 池大小直接决定了并发 batch —— batch 又直接决定了吞吐。这不是「优化」，是「能做 / 不能做」。

### 1.2 四种省法，以及本期是第几种

| 路线 | 做法 | 省的是什么 | 本系列哪一期 |
| --- | --- | --- | --- |
| 算清账 | 先搞明白一个 token 到底占多少字节 | 不省，只是量清楚 | 032 |
| 压字节 | fp8 / int4 / per-block scale | 每条条目变短 | 033 |
| 少看条目 | 训练一个 indexer，只算 top-k | 每条 query 少读条目 | 034 |
| **丢条目** | **用现成的注意力分数决定丢谁** | **缓存里根本不存** | **035 ← 本期** |

注意 034 和 035 的区别不在「省什么」，而在**谁当裁判**：

- 034（NSA / DSA / MoBA）的裁判是一个**专门训练的浅层网络**（indexer），它要额外花训练成本、额外占显存、还要额外算 64 个头的小 attention。换来的是「可训练、可端到端优化」。
- 035 的裁判是**模型自己已经算出来的注意力分数**——反正 dense attention 每步都要算一遍 `softmax(qKᵀ)`，分数是现成的副产品，白拿。代价是：这些分数本来是给「加权求和」用的，**不是给「排序」用的**。

### 1.3 核心张力：一个矩阵 vs 一个标量

这是本期的题眼，值得单独说清楚。

一次 attention 的中间结果是一个矩阵：`P[t, j]` = 第 t 个 query 分给第 j 个 key 的权重。它有 `T × T` 个自由度。

而所有「丢谁」的策略最后都要产出一个**向量** `s[j]`，然后「按 s 排序、丢掉最小的那些」。从 `T × T` 的矩阵到长度 `T` 的向量，这一步是**降维**，必然丢信息。

问题在于丢的是哪一部分信息：

- 如果第 t 个 query 需要的 key 是「第 7 号位置」，第 t+1 个 query 需要的还是「第 7 号位置」—— 那降维完全无损，一个向量足够了。**这就是文档问答、摘要、检索增强这些任务的形状**：文档区块的重要性对整段生成是稳定的。
- 如果第 t 个 query 需要的是「t−95」，第 t+1 个需要的是「t−94」—— 那这个向量几乎不携带信息，因为「哪一位重要」本身就在移动。**这就是链式推理、复制任务、passkey 检索的形状**。

第 4 节会用实验把这条差别量出来。先记住一个数：**同一份注意力，逐 query 看位置重要性有 111 倍差距，把 query 轴求和之后只剩 4.55 倍**（§ 3.2）。

---

## 2. 核心思想

### 2.1 三个自由度

任何一个「推理期 KV 稀疏化」方案，都可以用三个旋钮描述：

```
① 打分信号：用什么给每个 key 打一个分？
     位置规则   → 只留最近 W 个（滑窗）
     历史统计   → 累积注意力 / 上一步注意力（H2O / TOVA / Scissorhands）
     观察窗投票 → 用 prompt 末尾若干行的注意力投票（SnapKV）
     当前 query → 用这一条 query 自己估（Quest / MInference）

② 粒度：一次决定多少个 token？
     token（1 个） / 块（16~64 个） / 页（vLLM 的 block）

③ 动作：决定之后干什么？
     永久驱逐（evict，腾显存，不可逆）
     按需取回（retrieve，不腾显存但省带宽）
     补零跳过（zero-mask，省带宽也省算力但块还占着）
```

三个旋钮可以自由组合，这就是为什么这个领域看起来有那么多论文 —— 其实绝大多数是「同一个旋钮换一档」，或者「把两个信号线性加权」。**真正带来质变的只有两处**：信号从「query 无关」变成「query 相关」（②里最贵的一档），以及粒度从 token 变成块（③里唯一能真的腾出显存的一档）。这两个才是本期的主线。

### 2.2 打分就是「沿 query 轴做归约」

把三个信号写成公式，特点一眼就看清了（$a_{t,j}$ 是第 t 个 query 分给第 j 个 key 的注意力概率）：

$$\text{滑窗}:\ s_j = \frac{j\,\ge\,T-W}{\text{常数（与内容无关）}}$$

$$\text{累积注意力 (H2O)}:\ s_j = \sum_{\tau \ge j} a_{\tau,j}$$

$$\text{最后一条 query (TOVA)}:\ s_j = a_{T,j}$$

$$\text{观察窗投票 (SnapKV)}:\ s_j = \sum_{\tau \in [T-W,\,T)} a_{\tau,j}$$

$$\text{当前 query (Quest)}:\ s_j \approx \text{upper bound of } q_t \cdot k_j \quad\text{（只对当前这一步有效）}$$

前四个都是 $\sum_\tau a_{\tau,j}$ 的不同形式 —— **沿 query 轴求和**。第五个不求和，它按当前 query 重新估一次，所以是唯一「query 相关」的那个。

沿 query 轴求和带来一个立刻可见的后果：求和会把「峰」抹平。因为每个 query 的峰落在**不同的位置**上，加起来的形状就是一堆尖峰被摊平后的平均。这就是 § 2.3 要量的东西。

### 2.3 量化「聚合损失」

设某个头在序列上有 $Q$ 个 query，每个 query 的注意力向量归一化后行和为 1。定义两个峰度：

$$\kappa_{\text{per-query}} = \mathbb{E}_t\left[\frac{\max_j a_{t,j}}{\frac{1}{t+1}\sum_j a_{t,j}}\right]
\qquad
\kappa_{\text{agg}} = \frac{\max_j s_j}{\frac{1}{S}\sum_j s_j}$$

$\kappa_{\text{per-query}}$ 是「用这条 query 自己的分数选位置，信号有多尖」；$\kappa_{\text{agg}}$ 是「用聚合分数选位置，信号有多尖」。两者的比值就是**聚合这一步付出的信息代价**。

在真正执行复制的那个头（下面会看到是 L1H0）上：

- $\kappa_{\text{per-query}} = 83.9$
- $\kappa_{\text{agg}} = 4.55$
- 比值 $= 18.4$

也就是说：**逐 query 看，一个位置能比平均位置重要 84 倍；聚合之后只剩 4.55 倍。** 而 $S = 192$ 时 4.55 倍的峰度，扣掉「前 96 位 / 后 96 位」这个粗糙结构之外，几乎没有区分度 —— 实验里答案位在累积分数里的平均排名是 **51.1 / 144**，而纯随机的期望排名是 72.2。比随机好，但离「够用」远得很。

### 2.4 两个「兜底占位」：sink 和 recency

既然聚合分数不可靠，为什么这些方法还在生产里用？因为它们不靠分数**精确**，它们靠两个**结构性的兜底**：

**① Attention Sink（注意力汇）。** 实验和文献都观察到：序列最前面几个 token，无论语义是否重要，都会拿到异常高的注意力。StreamingLLM 的解释是：softmax 的权重和必须为 1，当一条 query 与所有 key 都不相关时，**它必须把「无处安放的注意力」倒在某个地方**——训练过程会让某个常驻 token（通常是 BOS）长成一个「注意力垃圾桶」。

这个机制有一个可以用第一性原理算清的量。假设某个 key 相对其他所有 key 有一个恒定的对数几率优势 $\Delta$，其余 $T-1$ 个 key 打平，那么它分到的质量是：

$$m = \frac{e^{\Delta}}{e^{\Delta} + (T-1)} = \frac{1}{1 + (T-1)e^{-\Delta}}$$

反解出「要守住质量 $m$ 需要多大的优势」：

$$\Delta = \ln\frac{(T-1)(1-m)}{m} \approx \ln T + \text{常数}\quad (m = 1/2\ \text{时})$$

质量 $m$ | $T=10^3$ 需要的 $\Delta$ | $T=10^5$ | $T=10^6$
--- | --- | --- | ---
0.5 | 6.91 | 11.51 | 13.82
0.2 | 8.29 | 12.90 | 15.20

**关键不是 $\Delta$ 的绝对值，而是它只按 $\ln T$ 增长。** 上下文从 1K 涨到 1M（一千倍），守住同一份质量需要的优势只从 6.91 nat 涨到 13.82 nat。这意味着一个**训练时定下来的常数偏置，在任何长度上都够用** —— 这就是 sink 为什么能「免费」存在，也是为什么删掉它会立刻崩：它独自扛着「所有无匹配 query 的归一化需求」。

> **本机的实测结果与文献不一致，必须说清楚。** 我在实验台上训了一个 2 层的小模型，**它没有 sink**：位置 0 只拿到 0.005~0.009 的质量，和背景（0.0052）同量级；我甚至人为构造了「query 的 token 在第一段从未出现过」的无匹配输入，注意力也只是变得更平均（峰/均从 3.3 掉到 2.3），峰值位置随机散落在 54/56/59/60/70/80/85/95，没有任何一个位置变成垃圾桶。
>
> 原因是这个任务**每条 query 都有答案**（唯一的例外是 t−95 落在序列外的那些），所以 softmax 不需要垃圾桶。sink 是**训练出来的**，不是 softmax 自动产生的；真实 LLM 有 BOS、有大量「这步没什么可看」的位置，才把它长了出来。这是「文献里看到的」与「本机实测」的差别，我把它标出来而不是含糊过去。

**② Recency（最近性）。** 最近几个 token 几乎总在被看 —— 因为语言模型做的是续写，最近的上下文就是最直接的依据。所以滑窗在**局部**永远是安全的，问题只出在窗口够不到的远处。

两个兜底合起来就是工业界最常见的配方：**钉子户（sink）+ 滑动窗（recency）+ 用某个分数从中间捞一批（heavy hitter）**。所有论文的差别就在「分数怎么算」和「抢到多少预算」。

### 2.5 谱系

| 年份 | 方法 | 打分信号 | 粒度 | 什么时候决定 | 要注意力分数吗 |
| --- | --- | --- | --- | --- | --- |
| 2023.05 | Scissorhands | 历史累积（量化后） | token | 每步 | 是 |
| 2023.06 | H2O | 历史累积 | token | 每步 | 是（每步） |
| 2023.09 | StreamingLLM | 无（sink + 窗口） | token | 每步 | **否** |
| 2024.04 | SnapKV | 观察窗投票 | token（per-head） | prefill 末尾一次 | 是（只 16 行） |
| 2024.06 | Quest | **当前 query × 页 min/max** | 页（16） | 每步 | **否**（只用 K） |
| 2024.07 | MInference | 三种固定模式 | 块 | prefill 前离线搜一次 | 否（只用结构） |

谱系里有两条清晰的演化方向：**一是从「query 无关」走向「query 相关」**（StreamingLLM / H2O / SnapKV → Quest），**二是从「token 粒度」走向「块粒度」**（Scissorhands / H2O / SnapKV → Quest / MInference）。第 4 节会说明，后者不是精度上的选择，而是**显存能不能真的回来**的问题。

---

## 3. 在 PyTorch 中怎么用

下面的代码全部在本机 CPU 上真跑过（torch 2.14.0，CPU + MPS 环境）。**请按顺序执行**：3.1 会训练一个小模型并把注意力概率落到 `kv035.pt`，后面几段直接读它。

### 3.1 先造一个「必须回看 96 步之前」的任务

要研究「丢谁」，得有一个**丢掉就会坏**的任务。我选了最简单也最苛刻的一种：

```
序列 = [A, A]         A 是 96 个随机 token，词表 32
```

预测第二个 A 的第 k 位时，模型必须回头找到「当前 token 上一次出现的位置」，再取它的下一个 token。因为第二段与第一段逐个相同，**预测位置 t 的 token，答案固定在第 t−95 位** —— 一个距离恒为 95 的长程依赖，而且离序列末尾越来越远（早期的 t 除外）。

这和 passkey 检索同构：答案位置固定、离 query 很远、且**不在** sink 也不在最近窗口里。对它来说，只保留最近 W 个 token 的策略，必须 $W \ge 97$ 才能覆盖到答案，即 50% 的预算。

```python
"""3.1 块：训练一个「必须回看 96 步之前」的玩具模型，看清它的注意力结构"""
import torch
import torch.nn as nn
import torch.nn.functional as F

V, D, NH, DH, LAYERS, HALF, S = 32, 96, 4, 24, 2, 96, 192      # 总共 192 个位置
CKPT = "kv035.pt"


def make_batch(bs, gen):                      # 序列 = [A, A]，A 是 96 个随机 token
    a = torch.randint(0, V, (bs, HALF), generator=gen)
    return torch.cat([a, a], dim=1)


class Blk(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.wq = nn.Linear(D, NH * DH, bias=False)
        self.wk = nn.Linear(D, NH * DH, bias=False)
        self.wv = nn.Linear(D, NH * DH, bias=False)
        self.wo = nn.Linear(NH * DH, D, bias=False)
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(self, x, keep=None):           # keep 是 [B,S,S] 的 bool 掩码，None 表示 dense
        B, T, _ = x.shape
        y = self.ln1(x)
        q = self.wq(y).view(B, T, NH, DH).transpose(1, 2)
        k = self.wk(y).view(B, T, NH, DH).transpose(1, 2)
        v = self.wv(y).view(B, T, NH, DH).transpose(1, 2)
        allow = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)
        if keep is not None:
            allow = allow & keep.reshape(keep.shape[0], 1, T, T)
        att = torch.softmax((q @ k.transpose(-1, -2) / DH ** 0.5).masked_fill(~allow, -float("inf")), -1)
        o = (att @ v).transpose(1, 2).reshape(B, T, NH * DH)
        x = x + self.wo(o)
        return x + self.mlp(self.ln2(x)), att


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(S, D)
        self.blocks = nn.ModuleList([Blk() for _ in range(LAYERS)])
        self.head = nn.Linear(D, V)

    def forward(self, x, keep=None):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        atts = []
        for blk in self.blocks:
            h, a = blk(h, keep)
            atts.append(a)
        return self.head(h), torch.stack(atts)


torch.manual_seed(0)
gen = torch.Generator().manual_seed(7)
m = Tiny()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1200, eta_min=3e-4)
for _ in range(1200):
    x = make_batch(32, gen)
    logits, _ = m(x)
    loss = F.cross_entropy(logits[:, HALF - 1:-1].reshape(-1, V), x[:, HALF:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()

x, P = make_batch(24, torch.Generator().manual_seed(1234)), None
with torch.no_grad():
    _, P = m(x)
BATCH = x.shape[0]
Q = list(range(HALF, S))                       # 复制段的 96 个 query
print(f"copy 任务：query 在 t（token A[t-96]）时要输出 A[t-95]，即必须精确回看第 t-95 位")
print(f"模型 {LAYERS} 层 × {NH} 头，d={D}，序列 {S}；训练 1200 步后的复制段准确率：", end="")
gen2 = torch.Generator().manual_seed(999)
cor = tot = 0
with torch.no_grad():
    for _ in range(8):
        xs = make_batch(64, gen2)
        lg, _ = m(xs)
        cor += (lg[:, HALF - 1:-1].argmax(-1) == xs[:, HALF:]).sum().item()
        tot += 64 * HALF
print(f"{cor / tot:.4f}   （随机猜 {1 / V:.4f}）")
print()
print("== 各层各头把质量放在哪里（query t ∈ [96,191] 的 batch 平均）==")
print(f"{'':7}{'答案位 t-95':>12}{'自己 t-96':>11}{'位置 0':>9}{'lag<16 合计':>12}{'lag 16-64':>11}")
for l in range(LAYERS):
    for h in range(NH):
        ans = torch.stack([P[l, :, h, t, t - 95] for t in Q]).mean().item()
        slf = torch.stack([P[l, :, h, t, t - 96] for t in Q]).mean().item()
        snk = torch.stack([P[l, :, h, t, 0] for t in Q]).mean().item()
        near = torch.stack([P[l, :, h, t, t - 15:t + 1].sum(-1) for t in Q]).mean().item()
        mid = torch.stack([P[l, :, h, t, t - 63:t - 15].sum(-1) for t in Q]).mean().item()
        print(f" L{l}H{h}{ans:>12.5f}{slf:>11.5f}{snk:>9.5f}{near:>12.4f}{mid:>11.4f}")
print("  解读：L1 四个头把 0.52~0.58 的质量精确压在 t-95 上 —— 这是归纳头在干活；")
print("        'lag<16' 是最近 16 个位置的平均质量，均匀分布下应该是 16/192 = 0.0833，")
print("        L1 实测只有 0.034~0.040 —— **比均匀还低**，说明这个模型完全没有局部偏置；")
print("        位置 0 只拿到 0.005~0.009，和背景一个量级 —— **这个模型没有 attention sink**。")
print()
print("== 集中度：真需要的位置 vs 整个序列的平均 ==")
for l in range(LAYERS):
    for h in range(NH):
        pk = torch.stack([P[l, :, h, t, :t + 1].max() / P[l, :, h, t, :t + 1].mean() for t in Q]).mean()
        print(f" L{l}H{h}: 逐 query 的 峰/均 = {pk:>6.1f}", end="")
    print()
print()
with torch.no_grad():
    h1 = m.tok(x) + m.pos(torch.arange(S))
    h1, _ = m.blocks[0](h1, None)
    y1 = m.blocks[1].ln1(h1)
    q1 = m.blocks[1].wq(y1).view(BATCH, S, NH, DH).transpose(1, 2)
    k1 = m.blocks[1].wk(y1).view(BATCH, S, NH, DH).transpose(1, 2)
torch.save({"P": P, "x": x, "q1": q1, "k1": k1, "sd": m.state_dict()}, CKPT)
print(f"（已把注意力概率、L1 的 q/k 与权重存到 {CKPT}，后面几段直接读它）")
```

真实输出（本机 CPU，约 35 秒）：

```text
copy 任务：query 在 t（token A[t-96]）时要输出 A[t-95]，即必须精确回看第 t-95 位
模型 2 层 × 4 头，d=96，序列 192；训练 1200 步后的复制段准确率：1.0000   （随机猜 0.0312）

== 各层各头把质量放在哪里（query t ∈ [96,191] 的 batch 平均）==
           答案位 t-95    自己 t-96     位置 0   lag<16 合计  lag 16-64
 L0H0     0.18488    0.00558  0.00502      0.0807     0.2546
 L0H1     0.24529    0.00610  0.00297      0.0784     0.2408
 L0H2     0.14174    0.00804  0.00451      0.0892     0.2864
 L0H3     0.21082    0.00544  0.00632      0.0829     0.2636
 L1H0     0.57912    0.00593  0.00491      0.0345     0.1157
 L1H1     0.54375    0.00381  0.00747      0.0373     0.1320
 L1H2     0.52203    0.00335  0.00864      0.0393     0.1389
 L1H3     0.52101    0.00362  0.00449      0.0402     0.1356
  解读：L1 四个头把 0.52~0.58 的质量精确压在 t-95 上 —— 这是归纳头在干活；
        'lag<16' 是最近 16 个位置的平均质量，均匀分布下应该是 16/192 = 0.0833，
        L1 实测只有 0.034~0.040 —— **比均匀还低**，说明这个模型完全没有局部偏置；
        位置 0 只拿到 0.005~0.009，和背景一个量级 —— **这个模型没有 attention sink**。

== 集中度：真需要的位置 vs 整个序列的平均 ==
 L0H0: 逐 query 的 峰/均 =   50.9 L0H1: 逐 query 的 峰/均 =   63.1 L0H2: 逐 query 的 峰/均 =   44.3 L0H3: 逐 query 的 峰/均 =   54.8
 L1H0: 逐 query 的 峰/均 =  106.3 L1H1: 逐 query 的 峰/均 =  106.1 L1H2: 逐 query 的 峰/均 =  104.2 L1H3: 逐 query 的 峰/均 =  104.1

（已把注意力概率、L1 的 q/k 与权重存到 kv035.pt，后面几段直接读它）
```

三点值得停下来看：

1. **0.58 的质量压在一个位置上。** 逐 query 的峰/均是 104~106 —— 对一个 query 来说，位置的重要性差别是两位数量级。这就是「稀疏」在注意力里的真实含义。
2. **这个模型没有局部偏置。** 最近 16 位拿到的比均匀分布（0.0833）还少。所以在这个模型上做滑窗，没有任何「局部先验」可依赖。
3. **这个模型没有 sink。** 见 § 2.4 的说明。

### 3.2 把 query 轴求和，损失了多少

```python
"""3.2 块：聚合损失 —— 把 query×key 的矩阵压成一个向量，丢掉了多少"""
import torch

z = torch.load("kv035.pt", weights_only=False)      # 3.1 存下的注意力概率与 L1 的 q/k
P, q1, k1 = z["P"], z["q1"], z["k1"]
L, BATCH, H, S, _ = P.shape
Q = list(range(96, 192))
causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
Pm = P.mean(1)                                       # 跨 batch 平均 → [L,H,S_t,S_j]
l, h = 1, 0                                          # 看真正执行复制的那个头

print("== 1) 同一个头（L1H0）：逐 query 的峰度 vs 聚合后的峰度 ==")
print(f"{'query t':>8}{'真实 top1 位置':>14}{'峰值':>9}{'前缀均值':>10}{'峰/均':>8}"
      f"{'答案位在累积分里的排名':>22}")
ranks = []
aas = (Pm[l, h] * causal).sum(0)                     # 累积注意力：每个 key 收到的质量之和
for t in Q:
    row = Pm[l, h, t, :t + 1]
    r = int((aas[:t + 1] > aas[t - 95]).sum()) + 1
    ranks.append(r)
    if t in (96, 128, 160, 191):
        print(f"{t:>8}{int(row.argmax()):>14}{row.max():>9.4f}{row.mean():>10.5f}"
              f"{row.max() / row.mean():>8.1f}{r:>22}")
perq = torch.stack([Pm[l, h, t, :t + 1].max() / Pm[l, h, t, :t + 1].mean() for t in Q]).mean()
agg = aas.max() / aas[:96].mean()
print(f"\n  逐 query 的峰/均（96 个 query 平均） : {perq:.1f}")
print(f"  聚合分数 aas 的峰/均               : {agg:.2f}（最大落在 j={int(aas.argmax())}）")
print(f"  → 把 query 轴求和这一步，峰度掉了 {perq / agg:.1f} 倍")
print(f"  答案位在 aas 里的平均排名 {sum(ranks) / len(ranks):.1f} / 中位 "
      f"{sorted(ranks)[len(ranks) // 2]}；纯随机的期望排名是 {sum(range(97, 193)) / 96 / 2:.1f}")
print()
print("== 2) 四种打分信号与「逐 query 真实需求」的秩相关（L1H0）==")
truth = torch.tensor([float(Pm[l, h, t, t - 95]) for t in Q])
sigs = {"累积注意力 AAS（H2O）": aas,
        "最后一条 query LTA（TOVA）": Pm[l, h, 191],
        "观察窗投票（末 16 位，SnapKV）": Pm[l, h, 175:191].sum(0),
        "纯最近邻（滑窗）": torch.tensor([float(j > S - 16) for j in range(S)])}
rk = lambda v: v.argsort().argsort().float()
for name, sig in sigs.items():
    s = sig[[t - 95 for t in Q]]
    rho = torch.corrcoef(torch.stack([rk(s), rk(truth)]))[0, 1].item()
    print(f"  {name:<30} Spearman ρ = {rho:+.4f}")
print("  注意 ρ 最高的 AAS 仍然是「粗粒度对了、细粒度没对」：这个任务的答案位全部落在前")
print("  96 位里，aas 恰好能分开「前半段 / 后半段」，所以相关系数看着漂亮，排名却很差。")
print()
print("== 3) 逐 query 看，位置的重要性差多少倍 ==")
n = torch.stack([Pm[l, h, t, t - 95] for t in Q]).mean()
bg = Pm[l, h, :96].mean()
print(f"  query t=160 的答案位（65）拿到 {Pm[l, h, 160, 65]:.4f} —— 一条 query 的 62% 都给了它")
print(f"  答案位的平均质量 {n:.4f}，背景位平均 {bg:.5f}")
print(f"  比值 {n / bg:.1f}×。逐 query 看位置重要性有百倍差距，聚合之后就只剩 4.55 了。")
```

真实输出：

```text
== 1) 同一个头（L1H0）：逐 query 的峰度 vs 聚合后的峰度 ==
 query t    真实 top1 位置       峰值      前缀均值     峰/均           答案位在累积分里的排名
      96             1   0.9719   0.01031    94.3                     3
     128            33   0.4117   0.00775    53.1                    57
     160            65   0.6163   0.00621    99.2                    58
     191            60   0.2918   0.00521    56.0                   131

  逐 query 的峰/均（96 个 query 平均） : 83.9
  聚合分数 aas 的峰/均               : 4.55（最大落在 j=0）
  → 把 query 轴求和这一步，峰度掉了 18.4 倍
  答案位在 aas 里的平均排名 51.1 / 中位 50；纯随机的期望排名是 72.2

== 2) 四种打分信号与「逐 query 真实需求」的秩相关（L1H0）==
  累积注意力 AAS（H2O）                 Spearman ρ = +0.7955
  最后一条 query LTA（TOVA）           Spearman ρ = -0.1507
  观察窗投票（末 16 位，SnapKV）           Spearman ρ = +0.5500
  纯最近邻（滑窗）                       Spearman ρ = -0.2157
  注意 ρ 最高的 AAS 仍然是「粗粒度对了、细粒度没对」：这个任务的答案位全部落在前
  96 位里，aas 恰好能分开「前半段 / 后半段」，所以相关系数看着漂亮，排名却很差。

== 3) 逐 query 看，位置的重要性差多少倍 ==
  query t=160 的答案位（65）拿到 0.6163 —— 一条 query 的 62% 都给了它
  答案位的平均质量 0.5791，背景位平均 0.00521
  比值 111.2×。逐 query 看位置重要性有百倍差距，聚合之后就只剩 4.55 了。
```

第 2 段的那个 ρ = +0.7955 是个**陷阱**，值得单独讲：它看着很高，会让人以为「累积分数挺好用的」。但它是被「答案位全在前 96 位、后半段永远不会是答案」这个**粗结构**撑起来的。换成绝对排名就露馅了 —— 平均 51.1 / 144，比随机（72.2）好一点，离 top-16 差得远。**相关性高 ≠ 能排序**，因为 Spearman 对「整体顺序」敏感，而选择策略要的是「头部顺序」正确。

### 3.3 七种策略同台

```python
"""3.3 块：七种策略同台 —— 软召回 + 「那一位在不在」"""
import torch

z = torch.load("kv035.pt", weights_only=False)
P, q1, k1 = z["P"], z["q1"], z["k1"]
L, BATCH, H, S, _ = P.shape
Q = list(range(96, 192))
causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
Pm = P.mean(1)
SINK, WLOC, WOBS, PAGE = 4, 16, 16, 16          # sink 位数、局部窗、观察窗、页大小
DH = k1.shape[-1]


def recall(ss):
    k = torch.zeros(S, dtype=torch.bool); k[list(ss)] = True
    return (P[:, :, :, Q, :] * k.view(1, 1, 1, -1).float()).sum() / (L * BATCH * H * len(Q))


def hit(ss):
    return sum(int((t - 95) in ss) for t in Q) / len(Q)


def top_set(vec, B, first=0, last=0):
    s = set(range(first)) if first else set()
    if last: s |= set(range(S - last, S))
    v = vec.clone()
    for i in s: v[i] = -1e9
    s |= set(int(i) for i in v.topk(max(B - len(s), 0)).indices)
    return s


aas_all = (Pm * causal).sum(-2).mean(0).mean(0)          # 跨层跨头平均的累积注意力
aas_1 = (Pm * causal).sum(-2)[1, 0]                      # 只用 L1H0（单头上限）
lta = Pm[:, :, S - 1].mean(0).mean(0)                     # 最后一条 query 的注意力
voc = torch.nn.functional.max_pool1d(
    Pm[:, :, S - WOBS:S, :].sum(2).mean(0).mean(0).view(1, 1, -1), 5, 1, 2).view(-1)
import random
rng = random.Random(0)

print(f"模型与 3.1 同一份；S={S}，复制段 {len(Q)} 个 query，答案固定在 t-95")
print(f"dense 时答案位在 L1H0 拿到 0.5791 的质量，全序列平均 0.00521")
print()
print("== 软召回（每个 query 的质量有多少落在保留集里）==")
print(f"{'预算 B':>7}{'随机':>9}{'滑窗':>9}{'sink+窗':>9}{'H2O':>9}{'TOVA':>9}"
      f"{'SnapKV':>9}{'累积(单头)':>11}{'Quest页16':>11}{'逐query上界':>12}")
print("-" * 104)
rows = {}
for B in (16, 32, 64, 96):
    r = {}
    r["随机"] = sum(recall(set(rng.sample(range(S), B))) for _ in range(5)) / 5
    r["滑窗"] = recall(set(range(S - B, S)))
    r["sink+窗"] = recall(set(range(SINK)) | set(range(S - B + SINK, S)))
    r["H2O"] = recall(top_set(aas_all, B, SINK, WLOC))
    r["TOVA"] = recall(top_set(lta, B, SINK, WLOC))
    r["SnapKV"] = recall(top_set(voc, B, 0, WOBS))
    r["累积(单头)"] = recall(top_set(aas_1, B, SINK, WLOC))
    nb, nk = S // PAGE, max(1, B // PAGE)
    kmin = k1.view(BATCH, H, nb, PAGE, -1).min(3).values
    kmax = k1.view(BATCH, H, nb, PAGE, -1).max(3).values
    tot = hits = 0.0
    for t in Q:
        qt = q1[:, :, t] / DH ** 0.5
        ub = torch.maximum(qt[:, :, None, :] * kmin, qt[:, :, None, :] * kmax).sum(-1)
        pg = ub.topk(min(nk, nb), -1).indices
        kk = torch.zeros(BATCH, H, S, dtype=torch.bool)
        for g in range(pg.shape[-1]):
            kk.scatter_(2, pg[..., g].unsqueeze(-1) * PAGE + torch.arange(PAGE), True)
        tot += (P[:, :, :, t, :] * kk.float()).sum() / (L * BATCH * H)
        hits += float(kk[..., t - 95].float().mean())
    r["Quest页16"] = tot / len(Q)
    r["qh"] = hits / len(Q)
    tot = hits = 0.0
    for t in Q:
        idx = Pm[1, 0, t, :t + 1].topk(min(B, t + 1)).indices
        k = torch.zeros(S, dtype=torch.bool); k[idx] = True
        tot += (P[:, :, :, t, :] * k.float()).sum() / (L * BATCH * H)
        hits += float(k[t - 95])
    r["逐query上界"] = tot / len(Q)
    r["oh"] = hits / len(Q)
    rows[B] = r
    print(f"{B:>7}{r['随机']:>9.4f}{r['滑窗']:>9.4f}{r['sink+窗']:>9.4f}{r['H2O']:>9.4f}"
          f"{r['TOVA']:>9.4f}{r['SnapKV']:>9.4f}{r['累积(单头)']:>11.4f}"
          f"{r['Quest页16']:>11.4f}{r['逐query上界']:>12.4f}")
print()
print("== 硬指标：答案位 t-95 有没有被保住（复制任务真正在乎的）==")
print(f"{'预算 B':>7}{'滑窗':>9}{'sink+窗':>9}{'H2O':>9}{'TOVA':>9}"
      f"{'SnapKV':>9}{'累积(单头)':>11}{'Quest页16':>11}{'逐query上界':>12}")
print("-" * 92)
for B in (16, 32, 64, 96):
    r = rows[B]
    print(f"{B:>7}{hit(set(range(S - B, S))):>9.4f}{hit(set(range(SINK)) | set(range(S - B + SINK, S))):>9.4f}"
          f"{hit(top_set(aas_all, B, SINK, WLOC)):>9.4f}{hit(top_set(lta, B, SINK, WLOC)):>9.4f}"
          f"{hit(top_set(voc, B, 0, WOBS)):>9.4f}{hit(top_set(aas_1, B, SINK, WLOC)):>11.4f}"
          f"{r['qh']:>11.4f}{r['oh']:>12.4f}")
print()
print("滑窗要够到答案位必须 B ≥ 97（答案距末尾 95）—— 所以它在前四档全为 0。")
print(f"最划算的一档 B=16（8.3% 预算）：H2O {rows[16]['H2O']:.4f} vs 逐 query 上界 "
      f"{rows[16]['逐query上界']:.4f}，差 {rows[16]['逐query上界'] / rows[16]['H2O']:.1f} 倍。")
```

真实输出：

```text
模型与 3.1 同一份；S=192，复制段 96 个 query，答案固定在 t-95
dense 时答案位在 L1H0 拿到 0.5791 的质量，全序列平均 0.00521

== 软召回（每个 query 的质量有多少落在保留集里）==
   预算 B       随机       滑窗   sink+窗      H2O     TOVA   SnapKV     累积(单头)   Quest页16    逐query上界
--------------------------------------------------------------------------------------------------------
     16   0.0787   0.0050   0.0399   0.0421   0.0421   0.0050     0.0421     0.2480      0.4797
     32   0.1704   0.0194   0.0520   0.1593   0.1364   0.1246     0.1496     0.4209      0.5812
     64   0.3242   0.0740   0.1019   0.4452   0.3517   0.4127     0.4349     0.6625      0.7408
     96   0.5057   0.1681   0.1909   0.7152   0.4919   0.6967     0.7115     0.8138      0.8592

== 硬指标：答案位 t-95 有没有被保住（复制任务真正在乎的）==
   预算 B       滑窗   sink+窗      H2O     TOVA   SnapKV     累积(单头)   Quest页16    逐query上界
--------------------------------------------------------------------------------------------
     16   0.0000   0.0312   0.0312   0.0312   0.0000     0.0312     0.3277      0.9167
     32   0.0000   0.0312   0.1562   0.1458   0.1562     0.1562     0.5460      0.9375
     64   0.0000   0.0312   0.4896   0.3750   0.4896     0.4896     0.8146      0.9792
     96   0.0104   0.0312   0.8229   0.4896   0.8229     0.8229     0.9610      0.9896

滑窗要够到答案位必须 B ≥ 97（答案距末尾 95）—— 所以它在前四档全为 0。
最划算的一档 B=16（8.3% 预算）：H2O 0.0421 vs 逐 query 上界 0.4797，差 11.4 倍。
```

这张表要连着读三遍：

- **在最关键的预算档（B=16，8.3%）**：累积类信号（H2O / TOVA / SnapKV / 单头累积）全部挤在 0.0421，和随机（0.0787）**同一量级甚至更低**；而 query 相关的那一列是 0.2480，逐 query 上界 0.4797。**差 11.4 倍。**
- **TOVA < H2O < 单头累积**：更新的信号（只看最后一条 query）比累积更差，单头比跨头平均好。这符合直觉 —— 聚合得越狠、越只看近期，信息丢得越多。
- **硬指标比软召回更残酷**：H2O 在 B=16 时保住答案位的概率只有 3.12%（大概就是那 4 个 sink 位蒙对了 t=96 那一条 query），而 Quest 是 32.77%，逐 query 上界 91.67%。

**为什么 Quest 好这么多？** 因为它不沿 query 轴求和：它对**当前这一条 query** 用 key 的 min/max 箱子算了一个分数上界，于是选出来的页是「对这条 query 可能最响的那几页」，而不是「历史上平均最响的那几页」。代价是每一步都要重算，而且要额外存 min/max 元数据。

### 3.4 粒度：页级 oracle 的税，和 Quest 上界够不够用

```python
"""3.4 块：粒度 —— 页级 oracle 的税，和 Quest 那个箱式上界到底够不够用"""
import torch

z = torch.load("kv035.pt", weights_only=False)
P, q1, k1 = z["P"], z["q1"], z["k1"]
L, BATCH, H, S, _ = P.shape
Q = list(range(96, 192))
DH = k1.shape[-1]
l, h = 1, 0
K, BUD = 16, 64                      # 真正需要 K 个 token，允许加载 BUD 个

print(f"靶子 = 每个 query 真实分数 q·k 最高的 {K} 个 token；加载预算 {BUD} 个 token；S={S}")
print()
print("== A) 页级 oracle：页的分数取页内真实最大分（最好的页级策略）==")
print(f"{'页大小':>7}{'选几页':>8}{'实载 token':>11}{'命中靶子':>10}{'相对 b=1':>10}")
base = None
for b in (1, 2, 4, 8, 16, 32, 64):
    nb, nk = S // b, max(1, BUD // b)
    got = load = 0.0
    for t in Q:
        sc = (q1[:, h, t, None, :] * k1[:, h, :t + 1]).sum(-1) / DH ** 0.5
        tgt = set(int(v) for v in sc[0].topk(K).indices)
        psc = torch.full((nb,), -1e9)
        for p in range(nb):
            a, z_ = p * b, min((p + 1) * b, t + 1)
            if a <= t: psc[p] = sc[0, a:z_].max()
        keep = set()
        for p in psc.topk(nk).indices.tolist():
            keep |= set(range(p * b, min((p + 1) * b, t + 1)))
        got += len(tgt & keep) / K; load += len(keep)
    r = got / len(Q)
    base = r if base is None else base
    print(f"{b:>7}{nk:>8}{load / len(Q):>11.1f}{r:>10.4f}{r / base:>10.4f}")
print("  → 页大小 ≤ 4 完全不损失（靶子足够稀疏，一页里最多一个），从 8 开始掉：")
print("     b=16 保住 0.6452，b=32 剩 0.5944，b=64 只剩 0.5430 —— 拿页换效率，代价直接可测。")
print()
print("== B) Quest 的 min/max 箱式上界：够不够用来排序 ==")
print(f"{'页大小':>7}{'上界/真值':>11}{'上界命中靶子':>14}{'页oracle命中':>13}{'上界相对 oracle':>16}")
for b in (1, 4, 16, 64):
    nb, nk = S // b, max(1, BUD // b)
    kmin = k1[:, h].view(BATCH, nb, b, DH).min(2).values
    kmax = k1[:, h].view(BATCH, nb, b, DH).max(2).values
    got = ora = 0.0
    tight = []
    for t in Q:
        qt = q1[:, h, t] / DH ** 0.5
        ub = torch.maximum(qt[:, None, :] * kmin, qt[:, None, :] * kmax).sum(-1)
        sc = (qt[:, None, :] * k1[:, h]).sum(-1)
        tgt = set(int(v) for v in sc[0, :t + 1].topk(K).indices)
        tmax = torch.full((nb,), -1e9)
        for p in range(nb):
            a, z_ = p * b, min((p + 1) * b, t + 1)
            if a <= t: tmax[p] = sc[0, a:z_].max()
        for sig, acc in ((ub[0], "ub"), (tmax, "or")):
            keep = set()
            for p in sig.topk(nk).indices.tolist():
                keep |= set(range(p * b, min((p + 1) * b, t + 1)))
            if acc == "ub": got += len(tgt & keep) / K
            else: ora += len(tgt & keep) / K
        tight.append(float(ub[0, :t + 1].max() / tmax[:t + 1].max()))
    print(f"{b:>7}{sum(tight) / len(tight):>11.3f}{got / len(Q):>14.4f}"
          f"{ora / len(Q):>13.4f}{got / ora:>16.3f}")
print("  → 上界本身并不紧（页 16 时乐观 2.271 倍），但它的**排序**几乎和页 oracle 一样好")
print("     （相对损失 0.959，也就是 4%）。松的上界不一定是坏的上界。")
print("     另外：上界要额外存 2×head_dim 的 min/max，页 16 时相当于每 token 额外 1/16 的 K；")
print("     页 1 时元数据就等于 K 本身，所以页大小不能取 1。")
```

真实输出：

```text
靶子 = 每个 query 真实分数 q·k 最高的 16 个 token；加载预算 64 个 token；S=192

== A) 页级 oracle：页的分数取页内真实最大分（最好的页级策略）==
    页大小     选几页   实载 token      命中靶子    相对 b=1
      1      64       64.0    1.0000    1.0000
      2      32       63.8    1.0000    1.0000
      4      16       63.7    1.0000    1.0000
      8       8       63.3    0.8105    0.8105
     16       4       63.4    0.6452    0.6452
     32       2       63.4    0.5944    0.5944
     64       1       63.9    0.5430    0.5430
  → 页大小 ≤ 4 完全不损失（靶子足够稀疏，一页里最多一个），从 8 开始掉：
     b=16 保住 0.6452，b=32 剩 0.5944，b=64 只剩 0.5430 —— 拿页换效率，代价直接可测。

== B) Quest 的 min/max 箱式上界：够不够用来排序 ==
    页大小      上界/真值        上界命中靶子    页oracle命中     上界相对 oracle
      1      1.000        1.0000       1.0000           1.000
      4      1.639        0.7422       1.0000           0.742
     16      2.271        0.6185       0.6452           0.959
     64      2.812        0.5599       0.5430           1.031
  → 上界本身并不紧（页 16 时乐观 2.271 倍），但它的**排序**几乎和页 oracle 一样好
     （相对损失 0.959，也就是 4%）。松的上界不一定是坏的上界。
     另外：上界要额外存 2×head_dim 的 min/max，页 16 时相当于每 token 额外 1/16 的 K；
     页 1 时元数据就等于 K 本身，所以页大小不能取 1。
```

这里有两个反直觉的点：

1. **上界可以不紧，但排序仍然准。** 箱式上界只用到每维的 min/max，是最粗的信息，所以它对任何 token 都偏乐观（页 16 时偏 2.27 倍）。但只要**偏差在页之间是均匀的**，排序就还能对 —— 实测相对页 oracle 只损失 4%。这解释了 Quest 为什么用这么糙的估计还能work：它不需要准确的分数，只需要正确的**相对顺序**。
2. **页大小 ≤ 4 的白送。** 靶子每 page 至多命中一个，所以 b=4 时「选 16 页」正好装下 16 个靶子，一点都不亏。真正开始付税的是 b≥8。这和 034 期的结论形状一样：**块粒度要挑，但不是越小越好，而是「让靶子的平均间隔 ≈ 页大小」最好**。

### 3.5 观察窗的「有效期」

SnapKV 那条假设是：「prompt 末尾的十几行注意力，能预测生成阶段的注意力」。这条假设什么时候成立？

```python
"""3.5 块：观察窗的「有效期」—— SnapKV 那条假设什么时候成立"""
import math
import torch

z = torch.load("kv035.pt", weights_only=False)
P = z["P"]
L, BATCH, H, S, _ = P.shape
Q = list(range(112, 192))
Pm = P[:, :, 0]                       # [L,B,S,S]，取 L1H0
l, h, W, BUD = 1, 0, 16, 32

print(f"观察窗 = 结束于 t-d 的 {W} 行；用它们的投票替 query t 选 {BUD} 个位置")
print()
print("== A) 在真实的 copy 模型上（靶位 = t-95，每步漂 1 位）==")
print(f"{'d':>5}{'软召回 (B=32)':>15}{'窗口投票 top1':>15}{'答案位(65)票数排名':>20}")
for d in (1, 4, 16, 64):
    tot, n = 0.0, 0
    for t in Q:
        if t - d - W < 0: continue
        vote = Pm[l, :, t - d - W:t - d, :].sum(1).mean(0)
        tot += float(Pm[l, :, t, vote.topk(BUD).indices].sum(-1).mean())
        n += 1
    vote = Pm[l, :, 160 - d - W:160 - d, :].sum(1).mean(0)
    rk = int((vote[:161] > vote[65]).sum()) + 1
    print(f"{d:>5}{tot / n:>15.4f}{int(vote[:161].argmax()):>15}{rk:>20}")
print("  → 软召回看着还有 0.30~0.45（随机选 32/192 只有 0.104），但真正决定成败的那一位")
print("     票数排名是 61/65/111/18 —— **从来没进过 top-32**。这是聚合指标的经典骗局：")
print("     大部分质量被背景位凑出来了，而任务成败只取决于那一位在不在。")
print()
print("== B) 换成可控漂移率的合成注意力：靶位 = round(mu + rho*t) ==")
SS = 512

def build(rho, mu=64.0, peak=6.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(SS).view(SS, 1)
    j = torch.arange(SS).view(1, SS)
    logits = peak * (j == (mu + rho * t).round().clamp(0, SS - 1)).float() \
        + 0.35 * torch.randn(SS, SS, generator=g)
    return torch.softmax(logits.masked_fill(j > t, -float("inf")), -1)

print(f"{'漂移率 ρ':>9}{'含义':>22}" + "".join(f"{'d=' + str(d):>8}" for d in (1, 2, 4, 16, 64)))
for rho, desc in ((0.0, "靶位不动（文档区块）"), (0.25, "每 4 步挪 1 位"),
                  (0.5, "每 2 步挪 1 位"), (1.0, "每步挪 1 位（同 copy）")):
    Ph = build(rho)
    line = []
    for d in (1, 2, 4, 16, 64):
        tot, n = 0.0, 0
        for t in range(256, SS):
            if t - d - W < 0: continue
            vote = Ph[t - d - W:t - d, :].sum(0)
            tot += float(Ph[t, vote.topk(BUD).indices].sum())
            n += 1
        line.append(tot / n)
    print(f"{rho:>9.2f}{desc:>22}" + "".join(f"{v:>8.4f}" for v in line))
print()
print("  解析：窗口的票落在靶位 mu+rho*(t-d)，而真正的靶位是 mu+rho*t，差 rho*d 位；")
print("  且票永远「偏后」。所以有效期 d* ≈ 1/rho：ρ=0.25 实测 3 步、ρ=0.5 实测 2 步，")
print("  ρ=1（本期的 copy 任务）d=1 就已经失效。靶位完全不漂时，d=64 也毫无衰减。")
print()
print("  这条 ρ 就是「观察窗能不能用」的全部判据：文档问答里目标区域固定（ρ≈0）→ 能用；")
print("  链式推理里「下一步要回看哪里」随推理推进而变（ρ>0）→ 窗口很快就过期。")
```

真实输出：

```text
观察窗 = 结束于 t-d 的 16 行；用它们的投票替 query t 选 32 个位置

== A) 在真实的 copy 模型上（靶位 = t-95，每步漂 1 位）==
    d     软召回 (B=32)      窗口投票 top1         答案位(65)票数排名
    1         0.3076             60                  61
    4         0.3032             47                  65
   16         0.3338             42                 111
   64         0.4533              0                  18
  → 软召回看着还有 0.30~0.45（随机选 32/192 只有 0.104），但真正决定成败的那一位
     票数排名是 61/65/111/18 —— **从来没进过 top-32**。这是聚合指标的经典骗局：
     大部分质量被背景位凑出来了，而任务成败只取决于那一位在不在。

== B) 换成可控漂移率的合成注意力：靶位 = round(mu + rho*t) ==
    漂移率 ρ                    含义     d=1     d=2     d=4    d=16    d=64
     0.00            靶位不动（文档区块）  0.5496  0.5496  0.5494  0.5497  0.5493
     0.25            每 4 步挪 1 位  0.3118  0.1958  0.0849  0.0842  0.0902
     0.50            每 2 步挪 1 位  0.2013  0.0873  0.0931  0.0882  0.0862
     1.00       每步挪 1 位（同 copy）  0.0864  0.0862  0.0862  0.0868  0.0858

  解析：窗口的票落在靶位 mu+rho*(t-d)，而真正的靶位是 mu+rho*t，差 rho*d 位；
  且票永远「偏后」。所以有效期 d* ≈ 1/rho：ρ=0.25 实测 3 步、ρ=0.5 实测 2 步，
  ρ=1（本期的 copy 任务）d=1 就已经失效。靶位完全不漂时，d=64 也毫无衰减。

  这条 ρ 就是「观察窗能不能用」的全部判据：文档问答里目标区域固定（ρ≈0）→ 能用；
  链式推理里「下一步要回看哪里」随推理推进而变（ρ>0）→ 窗口很快就过期。
```

**这是本期最实用的一条结论。** 观察窗不是「不够大就不行」的问题，而是**存在一个由漂移速度决定的硬性有效期**：$d^* \approx 1/\rho$。它解释了为什么：

- SnapKV 在 LongBench 这类**目标区域固定**的任务上表现好（$\rho \approx 0$，观察窗随便多远都行）；
- 但同一族方法在**多轮对话 / 链式推理**上明显退化 —— 因为「下一步要回看哪里」在推理过程中一直在挪；
- NVIDIA 那篇 KV cache compression 的工程分析里有一句「只有最近的约 25 条 query 反映模型真正在看哪里」，说的就是同一件事的经验版本。

### 3.6 端到端：把召回换成任务成功率

```python
"""3.6 块：端到端 —— 把召回换成任务成功率（逐 batch 两遍法：先 dense 取 q/k，再带掩码跑）"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

V, D, NH, DH, LAYERS, HALF, S = 32, 96, 4, 24, 2, 96, 192


class Blk(nn.Module):                      # 与 3.1 完全相同的结构（为了载入它的权重）
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.wq = nn.Linear(D, NH * DH, bias=False)
        self.wk = nn.Linear(D, NH * DH, bias=False)
        self.wv = nn.Linear(D, NH * DH, bias=False)
        self.wo = nn.Linear(NH * DH, D, bias=False)
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(self, x, keep=None):
        B, T, _ = x.shape
        y = self.ln1(x)
        q = self.wq(y).view(B, T, NH, DH).transpose(1, 2)
        k = self.wk(y).view(B, T, NH, DH).transpose(1, 2)
        v = self.wv(y).view(B, T, NH, DH).transpose(1, 2)
        allow = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)
        if keep is not None:
            allow = allow & keep.reshape(keep.shape[0], 1, T, T)
        att = torch.softmax((q @ k.transpose(-1, -2) / DH ** 0.5).masked_fill(~allow, -float("inf")), -1)
        o = (att @ v).transpose(1, 2).reshape(B, T, NH * DH)
        x = x + self.wo(o)
        return x + self.mlp(self.ln2(x)), att


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(S, D)
        self.blocks = nn.ModuleList([Blk() for _ in range(LAYERS)])
        self.head = nn.Linear(D, V)

    def forward(self, x, keep=None):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        for blk in self.blocks:
            h, _ = blk(h, keep)
        return self.head(h), None


z = torch.load("kv035.pt", weights_only=False)
m = Tiny(); m.load_state_dict(z["sd"]); m.eval()
P, q1, k1 = z["P"], z["q1"], z["k1"]
L, BATCH, H = P.shape[0], P.shape[1], P.shape[2]
causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
Pm = P.mean(1)
aas = (Pm * causal).sum(-2).mean(0).mean(0)
lta = Pm[:, :, S - 1].mean(0).mean(0)
voc = torch.nn.functional.max_pool1d(
    Pm[:, :, S - 16:S, :].sum(2).mean(0).mean(0).view(1, 1, -1), 5, 1, 2).view(-1)
gen0 = torch.Generator().manual_seed(777)
EVA = [torch.cat([a := torch.randint(0, V, (48, HALF), generator=gen0), a], 1) for _ in range(8)]


def static_set(vec, B, first=0, last=0):
    s = set(range(first)) if first else set()
    if last: s |= set(range(S - last, S))
    v = vec.clone()
    for i in s: v[i] = -1e9
    return s | set(int(i) for i in v.topk(max(B - len(s), 0)).indices)


def mask_static(ss):
    k = torch.zeros(S, dtype=torch.bool); k[list(ss)] = True
    return (k.view(1, S) & causal)[None]


@torch.no_grad()
def dense_qk(xs):
    h = m.tok(xs) + m.pos(torch.arange(S))
    h, _ = m.blocks[0](h)
    y = m.blocks[1].ln1(h)
    return (m.blocks[1].wq(y).view(-1, S, H, DH).transpose(1, 2),
            m.blocks[1].wk(y).view(-1, S, H, DH).transpose(1, 2))


@torch.no_grad()
def mask_quest(xs, B, PAGE=16):
    q, k = dense_qk(xs)
    nb, nk = S // PAGE, max(1, B // PAGE)
    kmin = k.view(-1, H, nb, PAGE, DH).min(3).values
    kmax = k.view(-1, H, nb, PAGE, DH).max(3).values
    keep = torch.zeros(xs.shape[0], S, S, dtype=torch.bool)
    for t in range(S):
        qt = q[:, :, t] / DH ** 0.5
        ub = torch.maximum(qt[:, :, None, :] * kmin, qt[:, :, None, :] * kmax).sum(-1).amax(1)
        pg = ub.topk(min(nk, nb), -1).indices
        kk = torch.zeros(xs.shape[0], S, dtype=torch.bool)
        for g in range(pg.shape[-1]):
            kk.scatter_(1, pg[:, g:g + 1] * PAGE + torch.arange(PAGE), True)
        keep[:, t] = kk & causal[t].view(1, S)
    return keep


@torch.no_grad()
def mask_hindsight(xs, B):
    keep = torch.zeros(xs.shape[0], S, S, dtype=torch.bool)
    for t in range(S):
        idx = Pm[1, 0, t, :t + 1].topk(min(B, t + 1)).indices
        keep[:, t].scatter_(1, idx.unsqueeze(0).expand(xs.shape[0], -1), True)
    return keep


@torch.no_grad()
def acc(kind, B, seed=0):
    cor = n = 0
    rng = random.Random(seed)
    for xs in EVA:
        if kind == "dense":       k = None
        elif kind == "滑窗":       k = mask_static(set(range(S - B, S)))
        elif kind == "sink+滑窗":  k = mask_static(set(range(4)) | set(range(S - B + 4, S)))
        elif kind == "H2O":       k = mask_static(static_set(aas, B, first=4, last=16))
        elif kind == "TOVA":      k = mask_static(static_set(lta, B, first=4, last=16))
        elif kind == "SnapKV":    k = mask_static(static_set(voc, B, last=16))
        elif kind == "随机":       k = mask_static(set(rng.sample(range(S), B)))
        elif kind == "Quest":     k = mask_quest(xs, B)
        elif kind == "事后最优":    k = mask_hindsight(xs, B)
        lg, _ = m(xs, k)
        cor += (lg[:, HALF - 1:-1].argmax(-1) == xs[:, HALF:]).sum().item()
        n += xs.shape[0] * HALF
    return cor / n


print("dense（不驱逐）成功率 %.4f      随机猜 %.4f（词表 %d）" % (acc("dense", 0), 1 / V, V))
print()
print(f"{'策略':<11}" + "".join(f"{'B=' + str(B):>9}" for B in (16, 32, 64, 96, 128, 160)))
print("-" * 64)
for kind in ("滑窗", "sink+滑窗", "H2O", "TOVA", "SnapKV", "随机", "Quest", "事后最优"):
    print(f"{kind:<11}" + "".join(f"{acc(kind, B, seed=B):>9.4f}" for B in (16, 32, 64, 96, 128, 160)))
print()
print("注意两件事：")
print(" 1) 滑窗是一道峭壁：B<97 全部 0.0298（等于瞎猜），B≥97 才突然可用 —— 因为答案位")
print("    固定在第 t-95 位，距末尾 95。这类「单个答案位」的任务会把差异放大成 0/1。")
print(" 2) 这正说明 passkey / needle-in-a-haystack 这类基准不能单独用来选策略：")
print("    它们的打分本质是「那一位在不在」，和软召回是两个不同的量。")
```

真实输出：

```text
dense（不驱逐）成功率 1.0000      随机猜 0.0312（词表 32）

策略              B=16     B=32     B=64     B=96    B=128    B=160
----------------------------------------------------------------
滑窗            0.0298   0.0298   0.0298   0.0298   0.0298   0.0298
sink+滑窗       0.0715   0.0710   0.0702   0.0720   0.3535   0.6764
H2O           0.0710   0.1921   0.5157   0.8393   1.0000   1.0000
TOVA          0.0710   0.1817   0.4054   0.5072   0.6569   0.8186
SnapKV        0.0298   0.0298   0.0298   0.0298   1.0000   1.0000
随机            0.0298   0.0960   0.1521   0.2188   0.4487   0.6291
Quest         0.0298   0.0298   0.0606   0.8248   0.9899   1.0000
事后最优          0.9292   0.9493   0.9899   0.9899   1.0000   1.0000

注意两件事：
 1) 滑窗是一道峭壁：B<97 全部 0.0298（等于瞎猜），B≥97 才突然可用 —— 因为答案位
    固定在第 t-95 位，距末尾 95。这类「单个答案位」的任务会把差异放大成 0/1。
 2) 这正说明 passkey / needle-in-a-haystack 这类基准不能单独用来选策略：
    它们的打分本质是「那一位在不在」，和软召回是两个不同的量。
```

两点观测：

- **滑窗是一道峭壁，不是斜线。** B 从 96 到 97 只差一个 token，成功率从 0.0298（瞎猜）跳到可用。因为答案位只有一个，而且它离末尾恰好 95。「保留率」这个连续量在这里被压成了 0/1。
- **Quest 在 B=16/32 是瞎猜，B=96 就跳到 0.8248。** 因为 B=16 只够选 1 页，而答案位散落在前 96 个位置里，一页根本盖不住。页粒度在**小预算**时是致命的 —— 这和第 4 节的块空洞问题是同一枚硬币的两面。

### 3.7 PyTorch 侧到底给了什么入口

```python
"""3.7 块：PyTorch 侧到底给了什么入口 —— 「内容相关的 KV 选择」有一等公民 API 吗"""
import inspect
import torch

print("torch", torch.__version__, "| CUDA 可用:", torch.cuda.is_available())
print()
print("== 1) SDPA 的后端枚举里有没有稀疏这一项 ==")
from torch.nn.attention import SDPBackend, sdpa_kernel
print(" ", list(SDPBackend.__members__.keys()))
print("  → 没有 sparse / eviction。而且 sdpa_kernel 只能在这些后端之间切，不能自定义选择逻辑。")
print()
print("== 2) flex_attention：能表达结构稀疏，表达不了内容相关的 top-k ==")
from torch.nn.attention.flex_attention import (
    flex_attention, create_block_mask, create_mask)
B, H, S_, D = 1, 2, 64, 16
q, k, v = (torch.randn(B, H, S_, D) for _ in range(3))


def sliding(b, h, qi, kj):
    return (qi >= kj) & (qi - kj < 16)


bm = create_block_mask(sliding, B, H, S_, S_, device="cpu", BLOCK_SIZE=16)
out = flex_attention(q, k, v, block_mask=bm)
elem = create_mask(sliding, B, H, S_, S_, device="cpu").float().mean().item()
blk = bm.to_dense().float().mean().item()
print(f"  滑窗 W=16，S=64：元素级密度 {elem:.4f}（解析 "
      f"{sum(min(i + 1, 16) for i in range(64)) / 64 ** 2:.4f}）")
print(f"  BlockMask 的块级密度 {blk:.4f} → 放大 {blk / elem:.2f}×（块里只要有一个元素命中就整块放行）")
print(f"  输出 shape {tuple(out.shape)}，和 dense 一致")
print(f"  mask_mod 的签名：{inspect.signature(sliding)}")
print("  → 参数是 (b, h, qi, kj)，能拿到的是**下标**，拿不到 q/k 向量。")
print("     所以首方 API 只能表达「由位置决定」的稀疏；")
print("     「由 q·k 决定」的选择要自己先在别处算出下标，再传进 mask_mod。")
print()
print("== 3) 有没有现成的 KV 驱逐入口 ==")
names = set(dir(torch.nn.attention)) | set(dir(torch.nn.functional)) | set(dir(torch))
hit = sorted({n for n in names
              for c in ("evict", "heavy_hitter", "h2o", "snapkv", "quest", "sink")
              if c in n.lower()})
print("  在 torch.nn.attention / torch.nn.functional / torch 里搜相关词：", hit or "无")
print("  → PyTorch 本身不提供 KV 驱逐 API。这是推理引擎层的职责。")
print()
print("== 4) 手动执行一次「按分数裁剪 KV」，注意 K/V 必须同下标 ==")
K = torch.arange(32).float().view(8, 4)
Vv = torch.arange(32).float().view(8, 4) + 100
score = torch.tensor([0.1, 8.0, 0.2, 0.3, 7.0, 0.1, 0.05, 9.0])
keep = score.topk(3).indices.sort().values
print(f"  原 K {tuple(K.shape)} → 裁剪后 {tuple(K[keep].shape)}，保留下标 {keep.tolist()}")
print(f"  K 的值域 [{K.min():.0f}, {K.max():.0f}] → [{K[keep].min():.0f}, {K[keep].max():.0f}]")
print(f"  V 的值域 [{Vv.min():.0f}, {Vv.max():.0f}] → [{Vv[keep].min():.0f}, {Vv[keep].max():.0f}]")
print("  若 K 与 V 用了不同的下标，键值会整体错行 —— 而且不会报任何错。")
```

真实输出：

```text
torch 2.14.0 | CUDA 可用: False

== 1) SDPA 的后端枚举里有没有稀疏这一项 ==
  ['ERROR', 'MATH', 'FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'CUDNN_ATTENTION', 'OVERRIDEABLE']
  → 没有 sparse / eviction。而且 sdpa_kernel 只能在这些后端之间切，不能自定义选择逻辑。

== 2) flex_attention：能表达结构稀疏，表达不了内容相关的 top-k ==
  滑窗 W=16，S=64：元素级密度 0.2207（解析 0.2207）
  BlockMask 的块级密度 0.4375 → 放大 1.98×（块里只要有一个元素命中就整块放行）
  输出 shape (1, 2, 64, 16)，和 dense 一致
  mask_mod 的签名：(b, h, qi, kj)
  → 参数是 (b, h, qi, kj)，能拿到的是**下标**，拿不到 q/k 向量。
     所以首方 API 只能表达「由位置决定」的稀疏；
     「由 q·k 决定」的选择要自己先在别处算出下标，再传进 mask_mod。

== 3) 有没有现成的 KV 驱逐入口 ==
  在 torch.nn.attention / torch.nn.functional / torch 里搜相关词： 无
  → PyTorch 本身不提供 KV 驱逐 API。这是推理引擎层的职责。

== 4) 手动执行一次「按分数裁剪 KV」，注意 K/V 必须同下标 ==
  原 K (8, 4) → 裁剪后 (3, 4)，保留下标 [1, 4, 7]
  K 的值域 [0, 31] → [4, 31]
  V 的值域 [100, 131] → [104, 131]
  若 K 与 V 用了不同的下标，键值会整体错行 —— 而且不会报任何错。
```

结论很干脆：**PyTorch 把「稀疏」理解成「结构稀疏」**（`flex_attention` 的 `mask_mod` 只拿到下标，拿不到 q/k），**把「KV 驱逐」理解成引擎的职责**（搜不到任何相关 API）。所以想在生产里用这些方法，你面对的是 § 4.1 的两个基础设施问题，而不是缺一行 API。

一个顺带量出来的数字：`flex_attention` 的 `BlockMask` 在 `BLOCK_SIZE=16` 时，**元素级稀疏度 0.2207 被放大成块级 0.4375，接近 2 倍**。这就是「块粒度税」在一个官方 API 里的样子 —— 你想省的算力，被块边界吃掉了一半。

---

## 4. 围绕该领域展开

### 4.1 两个基础设施问题（这一节是本期的重心）

论文里的方法大多在「离线复现」里有效，上了生产服务引擎就失灵。原因不是算法有问题，而是两个基础设施的硬约束。我这边把两条都量化了。

#### 问题一：token 级驱逐**腾不出显存**

vLLM 的 PagedAttention 按固定块分配 KV（默认块大小 16）。它的回收条件是：**一整块里一个存活 token 都没有**，才把这 16 个槽位还给分配器。

而 token 级驱逐（H2O / SnapKV / Scissorhands 全是）把幸存者撒得满序列都是。于是一整块变空的概率是：

$$P(\text{块空}) = (1-r)^{16}$$

$r$ 是保留比例。$r=0.2$（H2O 论文的招牌配置）时：$(0.8)^{16} = 0.0281$ —— 只有 2.81% 的块能变空。

```python
"""4.1 块：块空洞 —— token 级驱逐到底能回收多少显存"""
import random

BLK, T = 16, 32768                    # vLLM 默认块大小 16；一个长上下文请求 32K token
NBLK = T // BLK


def simulate(keep, window, trials=40, seed=0):
    """keep：中段按内容选中的比例；window：末尾连续保留的 token 数"""
    rng = random.Random(seed)
    mid = T - window
    fre, kept = [], []
    for _ in range(trials):
        flags = [rng.random() < keep for _ in range(mid)]
        flags += [True] * (window - window % BLK) + [False] * (window % BLK)
        flags += [False] * (T - len(flags))
        nz = sum(any(flags[b * BLK:(b + 1) * BLK]) for b in range(NBLK))
        fre.append(NBLK - nz); kept.append(sum(flags))
    f, k = sum(fre) / trials, sum(kept) / trials
    return k / T, f / NBLK


print(f"单请求 T={T} token，块大小 {BLK}，共 {NBLK} 块")
print()
print("== 1) 纯内容驱逐（H2O / SnapKV 的做法：幸存者撒在中段）==")
print(f"{'保留比例':>9}{'驱逐比例':>10}{'整块变空':>10}{'真正回收显存':>13}{'差额':>10}")
for r in (0.8, 0.5, 0.2, 0.1):
    kept, empty = simulate(r, 0)
    gap = (1 - kept) / empty if empty > 0 else float("inf")
    print(f"{kept:>9.3f}{1 - kept:>10.3f}{empty:>10.4f}{empty:>13.4f}{gap:>9.1f}×")
print()
print("== 2) 解析：幸存者撒向 N 块，一块空的概率 = (1-r)^16 ==")
for r in (0.8, 0.5, 0.2, 0.1):
    print(f"  保留 {r:.1f} → 空块概率 {(1 - r) ** BLK:.4f}，驱逐了 {1 - r:.1f}，"
          f"差额 {((1 - r) / (1 - r) ** BLK):.0f}×")
print()
print("== 3) 加末尾 recency 窗口（H2O 和 StreamingLLM 都保留最近若干 token）==")
print(f"{'保留比例':>9}{'窗口 W':>8}{'驱逐比例':>10}{'真正回收显存':>13}{'差额':>10}")
for r in (0.2, 0.1):
    for w in (0, 512, 1024):
        kept, empty = simulate(r, w)
        gap = (1 - kept) / empty if empty > 0 else float("inf")
        print(f"{kept:>9.3f}{w:>8}{1 - kept:>10.3f}{empty:>13.4f}{gap:>9.1f}×")
print("  → 窗口对碎片毫无帮助：最近的那些是连续满块，动不了；中段照样碎。")
print()
print("== 4) 对照：纯滑窗（只留末尾 W 个连续 token）==")
print(f"{'窗口 W':>8}{'驱逐比例':>10}{'真正回收显存':>13}{'差额':>10}")
for w in (1024, 2048, 4096):
    kept, empty = simulate(0.0, w)
    print(f"{w:>8}{1 - kept:>10.3f}{empty:>13.4f}{(1 - kept) / empty:>9.1f}×")
print("  → 连续窗口天然对齐块边界：驱逐多少就回收多少，差额 1.0×。")
print("     代价是窗口外信息永久丢失 —— 但至少它承诺的省显存是真的。")
print()
print("== 5) 修一次要花多少：前向压实（把幸存者 gather 成密集前缀）==")
PER_TOK, BW = 128 * 1024, 3.35e12
for r in (0.2, 0.1):
    surv = int(T * r); byt = surv * PER_TOK
    print(f"  保留 {r:.0%}（{surv} token = {byt / 2 ** 30:.2f} GiB）：读+写 "
          f"{2 * byt / 2 ** 30:.2f} GiB → 在 3.35 TB/s 上约 {2 * byt / BW * 1e3:.2f} ms")
print(f"  对比：dense 每生成一步都要读一遍整个 KV = {T * PER_TOK / BW * 1e3:.2f} ms")
print("  → 一次压实的成本约等于「半步」的常态读取，而且是偶发的一次性动作。")
```

真实输出：

```text
单请求 T=32768 token，块大小 16，共 2048 块

== 1) 纯内容驱逐（H2O / SnapKV 的做法：幸存者撒在中段）==
     保留比例      驱逐比例      整块变空       真正回收显存        差额
    0.800     0.200    0.0000       0.0000      inf×
    0.500     0.500    0.0001       0.0001   8185.6×
    0.201     0.799    0.0285       0.0285     28.0×
    0.100     0.900    0.1843       0.1843      4.9×

== 2) 解析：幸存者撒向 N 块，一块空的概率 = (1-r)^16 ==
  保留 0.8 → 空块概率 0.0000，驱逐了 0.2，差额 30517578125×
  保留 0.5 → 空块概率 0.0000，驱逐了 0.5，差额 32768×
  保留 0.2 → 空块概率 0.0281，驱逐了 0.8，差额 28×
  保留 0.1 → 空块概率 0.1853，驱逐了 0.9，差额 5×

== 3) 加末尾 recency 窗口（H2O 和 StreamingLLM 都保留最近若干 token）==
     保留比例    窗口 W      驱逐比例       真正回收显存        差额
    0.201       0     0.799       0.0285     28.0×
    0.213     512     0.787       0.0280     28.1×
    0.226    1024     0.774       0.0275     28.1×
    0.100       0     0.900       0.1843      4.9×
    0.114     512     0.886       0.1812      4.9×
    0.128    1024     0.872       0.1785      4.9×
  → 窗口对碎片毫无帮助：最近的那些是连续满块，动不了；中段照样碎。

== 4) 对照：纯滑窗（只留末尾 W 个连续 token）==
    窗口 W      驱逐比例       真正回收显存        差额
    1024     0.969       0.9688      1.0×
    2048     0.938       0.9375      1.0×
    4096     0.875       0.8750      1.0×
  → 连续窗口天然对齐块边界：驱逐多少就回收多少，差额 1.0×。
     代价是窗口外信息永久丢失 —— 但至少它承诺的省显存是真的。

== 5) 修一次要花多少：前向压实（把幸存者 gather 成密集前缀）==
  保留 20%（6553 token = 0.80 GiB）：读+写 1.60 GiB → 在 3.35 TB/s 上约 0.51 ms
  保留 10%（3276 token = 0.40 GiB）：读+写 0.80 GiB → 在 3.35 TB/s 上约 0.26 ms
  对比：dense 每生成一步都要读一遍整个 KV = 1.28 ms
  → 一次压实的成本约等于「半步」的常态读取，而且是偶发的一次性动作。
```

**表格第一行是这件事最刺眼的形态**：保留 80%、驱逐 20% 的时候，**一个块都没变空**，显存回收率恰好是 0。差额那一列是 `inf`，因为分母是 0。保留 20%（H2O 的招牌配置）时差额 **28.0×** —— 你承诺省 80% 的显存，实际只省了 2.85%。而纯滑窗的差额恒为 **1.0×**，因为它天然对齐块边界。

修法有两个方向，且都便宜：

- **前向压实**：把幸存者 gather 到密集前缀，尾部整块变空。代价是一次读+写约 1.6 GiB → **0.51 ms**，而 dense 每步读一遍 KV 就要 1.28 ms。也就是说**压一次的成本不到「半步」**，偶发执行完全可接受。
- **只在块粒度上做决定**：这是 vLLM 的 RFC #39076 选的路 —— 它维护 `block_attn_mass`（每块的累积注意力质量），EMA 平滑后低于阈值 θ 就把整块标记成 `EXPIRED` 并立刻释放物理页。因为决定的对象本来就是整块，**碎片问题从根上不存在**。

#### 问题二：FlashAttention **不把注意力分数写回显存**

H2O 需要每一个 decode 步的完整注意力行。但 FlashAttention 的核心设计就是**不物化 `T×T` 的分数矩阵** —— 它在 SRAM 里分块算完就扔，只把结果写回 HBM。这不是实现细节，这是它快的原因。

vLLM 的 RFC #39076 把这条约束写得很直白：

> FlashAttention (v2/v3/v4) never materializes the full attention weight matrix — the fused kernel avoids this precisely to reduce HBM bandwidth.

它给了两条绕法：

- **Path A（目标方案）**：用 FlashInfer 的 `BatchDecodeWithPagedKVCacheWrapper` 返回的 **LSE（log-sum-exp）** 反推每块的注意力质量。LSE 编码了 softmax 的归一化因子，从它就能重建质量分布，**不需要物化矩阵**；而且输出缓冲预分配后与 CUDA Graph 捕获兼容。
- **Path B（过渡方案）**：走 Triton 注意力后端（需 `--enforce-eager`），在 softmax 归约旁边挂一个 side pass 用 atomic add 累加到 `block_attn_mass`。代价是**放弃 FlashAttention**。

同一份 RFC 还描述了现状的尴尬：

> H2O, SnapKV, PyramidKV: Model-level methods operating outside the serving engine. Require significant kernel changes and are not integrated into the block allocator.

以及它列出的另一条思路：PagedEviction（arXiv:2509.04377）**干脆不算注意力分数**，改用 key norm 这类静态代理，理由正是「FlashAttention 不返回注意力权重」。

### 4.2 打分本身要花多少

既然「要打分」，那就把账算清楚。

```python
"""4.2 块：打分本身要花多少 —— 「想知道该丢谁」和「把 KV 全留下」哪个更贵"""
KVH, QH, LAYER, DH, BYTES = 8, 32, 32, 128, 2     # GQA-8，32 层，head_dim 128，fp16
PER_TOK = LAYER * KVH * DH * 2 * BYTES            # 每 token 全部层的 K+V
BW = 3.35e12                                      # H800 级 HBM 带宽


def h(x):
    for u, k in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10)):
        if x >= k:
            return f"{x / k:.2f} {u}"
    return f"{x:.0f} B"


print(f"一个 token 的 KV = {PER_TOK} B = {PER_TOK / 1024:.0f} KiB "
      f"（{LAYER} 层 × KV 头 {KVH} × head_dim {DH} × K,V × {BYTES} B）")
print()
print("== 注意力分数矩阵要占多少（一层、一条序列、fp16）==")
print(f"{'S':>8}{'全矩阵 H×S²×2B':>18}{'观察窗 16 行':>15}{'单行(decode)':>15}"
      f"{'全部层 KV':>13}{'全矩阵/KV':>11}")
print("-" * 82)
for S in (4096, 8192, 32768, 131072):
    full, obs, one, kv = QH * S * S * BYTES, QH * 16 * S * BYTES, QH * S * BYTES, S * PER_TOK
    print(f"{S:>8}{h(full):>18}{h(obs):>15}{h(one):>15}{h(kv):>13}{full / kv:>10.1f}x")
print()
print(f"  S=32768 时，一层一条序列的全分数矩阵 {h(QH * 32768 ** 2 * BYTES)}，"
      f"是整个模型 KV（{h(32768 * PER_TOK)}）的 {QH * 32768 * BYTES / PER_TOK:.0f} 倍。")
print(f"  只要 16 行的观察窗就降到 {h(QH * 16 * 32768 * BYTES)}。")
print(f"  而且这个比例与 S 无关：观察窗/KV = {QH * 16 * BYTES / PER_TOK:.4f}"
      f"（两边都含 S，约掉了）。")
print("  → 这就是为什么「只用观察窗打分」的 SnapKV 能落地，「要全部行打分」的 H2O 很难：")
print("     分数矩阵在 S=32768 时是 KV 本身的 16 倍，FlashAttention 干脆不物化它。")
print()
print("== 读 K 的代价（decode 单步，roofline）==")
print(f"{'S':>8}{'dense 读 KV':>14}{'留 20% 读':>12}{'每步省下':>12}{'打分额外读':>12}")
print("-" * 60)
for S in (8192, 32768, 131072):
    d = S * PER_TOK / BW * 1e3
    k = d * 0.2
    print(f"{S:>8}{d:>11.3f} ms{k:>9.3f} ms{d - k:>9.3f} ms{0:>12}")
print("  → decode 时打分和注意力共用同一次 K 读取，分数是搭便车来的。")
print()
print("== 一次性全量打分的成本上限 ==")
for S in (32768, 131072):
    kv = S * PER_TOK
    print(f"  S={S}: 把整个 K 扫一遍 = 读 {h(kv)} → {kv / BW * 1e3:.3f} ms，"
          f"正好等于一步 dense decode 的读取量")
print("  结论：扫一遍不贵（约 1 步的量）；真正贵的是「必须物化才能拿到分数」。")
print(f"       要物化就得退回 eager attention：S=131072 时那是 "
      f"{h(QH * 131072 ** 2 * BYTES)}。")
```

真实输出（roofline 推算，非实测；带宽取 H800 级的 3.35 TB/s）：

```text
一个 token 的 KV = 131072 B = 128 KiB （32 层 × KV 头 8 × head_dim 128 × K,V × 2 B）

== 注意力分数矩阵要占多少（一层、一条序列、fp16）==
       S       全矩阵 H×S²×2B       观察窗 16 行     单行(decode)       全部层 KV     全矩阵/KV
----------------------------------------------------------------------------------
    4096          1.00 GiB       4.00 MiB     256.00 KiB   512.00 MiB       2.0x
    8192          4.00 GiB       8.00 MiB     512.00 KiB     1.00 GiB       4.0x
   32768         64.00 GiB      32.00 MiB       2.00 MiB     4.00 GiB      16.0x
  131072       1024.00 GiB     128.00 MiB       8.00 MiB    16.00 GiB      64.0x

  S=32768 时，一层一条序列的全分数矩阵 64.00 GiB，是整个模型 KV（4.00 GiB）的 16 倍。
  只要 16 行的观察窗就降到 32.00 MiB。
  而且这个比例与 S 无关：观察窗/KV = 0.0078（两边都含 S，约掉了）。
  → 这就是为什么「只用观察窗打分」的 SnapKV 能落地，「要全部行打分」的 H2O 很难：
     分数矩阵在 S=32768 时是 KV 本身的 16 倍，FlashAttention 干脆不物化它。

== 读 K 的代价（decode 单步，roofline）==
       S    dense 读 KV     留 20% 读        每步省下       打分额外读
------------------------------------------------------------
    8192      0.321 ms    0.064 ms    0.256 ms           0
   32768      1.282 ms    0.256 ms    1.026 ms           0
  131072      5.128 ms    1.026 ms    4.103 ms           0
  → decode 时打分和注意力共用同一次 K 读取，分数是搭便车来的。

== 一次性全量打分的成本上限 ==
  S=32768: 把整个 K 扫一遍 = 读 4.00 GiB → 1.282 ms，正好等于一步 dense decode 的读取量
  S=131072: 把整个 K 扫一遍 = 读 16.00 GiB → 5.128 ms，正好等于一步 dense decode 的读取量
  结论：扫一遍不贵（约 1 步的量）；真正贵的是「必须物化才能拿到分数」。
       要物化就得退回 eager attention：S=131072 时那是 1024.00 GiB。
```

三条要分开说：

1. **decode 阶段的打分是免费的。** 一条 query 对全部 K 做一次点积，本来就是 dense attention 要干的事；分数是副产品。所以 H2O 在 decode 每一步打分，**不额外花读取带宽**，它省下的恰好是「不再读被驱逐的那部分 K/V」。
2. **prefill 末尾的一次性打分才贵，而且贵在「不能物化」。** 要拿全部 S 行的分数，就得算 `H×S²` 的矩阵：S=32768 时一层就是 **64 GiB**，是整个模型 KV 的 16 倍；S=131072 时是 **1 TiB**。物理上不可能。而如果只要 16 行的观察窗，就降到 **32 MiB**，且这个比例与 S 无关（`Q头数×16×2 / (层数×KV头数×head_dim×4) = 0.78%`）。
3. **所以「只用观察窗」不是 SnapKV 的精度选择，是它的工程选择。** 它把一个 O(S²) 的中间量降到了 O(16·S)。这也解释了 § 3.5 的有效期代价：**为了便宜，它只敢看 16 行。**

### 4.3 两条绕开路线

把两个问题合起来看，工业界的解法有一个共同特点 —— **不解决它们，而是绕开**：

| 问题 | 正面解决 | 绕开的办法 | 代价 |
| --- | --- | --- | --- |
| 分数矩阵物化不起 | 改 FlashAttention 内核 | ① 只用 16 行观察窗（SnapKV）② 从 LSE 反推（RFC Path A）③ 用 K 的 min/max 箱式上界代替真分数（Quest）④ 用 key norm 等静态代理（PagedEviction） | ① 有效期只有 1/ρ 步 ② 依赖 FlashInfer ③ 上界松 2.3 倍但排序够用 ④ 精度弱 |
| 驱逐腾不出显存 | 前向压实（gather） | 只在**整块**粒度上做决定（RFC 的 `EXPIRED` block、Quest 的页选择） | 块粒度税：§ 3.4 里 b=16 只剩 0.6452 的命中 |

这张表值得记住：**四条「绕开分数物化」的路，全部是「用一个便宜得多的代理量代替真分数」**；而「绕开碎片」的唯一办法是**接受块粒度**。§ 3.4 的实测告诉你块粒度的价格是多少 —— b=16 时 0.6452，b=64 时 0.5430。

### 4.4 和已发各期的关系

| 期数 | 关系 |
| --- | --- |
| 011 PagedAttention | 本期问题一的根源。块分配是它的优点（无外部碎片），也是 token 级驱逐的坑（内部碎片无法回收，除非整块空） |
| 029 前缀缓存 | **同一个「有限缓存里丢什么」的问题，换了一层**。029 在基数树上做叶子优先 LRU（块级），本期在 KV 序列上做注意力打分驱逐（token 级）。有趣的是 029 的结论在这里重现了：**块级的驱逐天然对齐、回收得到；token 级的分离了「驱逐」和「回收」** |
| 032 KV 的字节账 | 本期 4.2 的 128 KiB/token、KV 体积表都直接引用它 |
| 033 KV 量化 | **正交且可叠加**。033 压每条条目的字节，本期减少条目数。两者相乘：fp8（÷2）+ 驱逐 20%（÷5）= ÷10 |
| 034 稀疏注意力 | **同一个「选谁」问题的另一条路线**。034 的 indexer 需要训练（KL 蒸馏 + 两阶段配方），本期的信号是白拿的；034 的 indexer 只需 64 个头 × 128 维的小计算，代价是额外参数；本期不需要额外参数，代价是信号不可训练、不能端到端优化。两者的验收指标可以合并：**都用 recall（选中集合覆盖了多少注意力质量）**，但本期 § 3.5/3.6 说明 recall 单独不够用 |
| 027 投机解码 / 028 连续批处理 | 都在抢同一份额度 —— 那个 295 平衡点（B·L）管的是「算力时间 vs 内存时间」。驱逐的直接收益是把它往下压（少读字节），这让两者更容易同时成立 |

### 4.5 五种主流方法的对照

| | StreamingLLM | H2O | SnapKV | Quest | MInference |
| --- | --- | --- | --- | --- | --- |
| 打分信号 | 无（sink + 窗口） | 累积注意力 | 观察窗投票（per-head） | **当前 query × 页 min/max** | 三种固定模式 |
| 粒度 | token | token | token | **页（16）** | 块 |
| 决定时机 | 每步 | 每步 | prefill 末尾一次 | 每步 | prefill 前离线搜一次 |
| 需要真注意力分数 | **否** | 是（每步） | 是（只 16 行） | **否**（只用 K） | 否 |
| 会腾出显存吗 | **是**（连续窗口对齐块） | 否（碎片） | 否（碎片） | 否（只省带宽） | 否（只省算力） |
| 论文报告的速度 | 22.2× vs 滑窗重算基线 | 最多 29× 吞吐（OPT-6.7B/30B）、1.9× 延迟 | 3.6× 生成速度、8.2× 显存效率（16K 输入） | 7.03× 自注意力 / 2.23× 端到端 | 1M 上下文 10× prefill |
| 本机实测（B=16，硬指标） | 0.0312 | 0.0312 | 0.0000 | **0.3277** | 未测（结构性方法） |

表里最后一行是我在 § 3.3 那个 192 token 的 copy 任务上量出来的「答案位保住率」，**只能同表内互相比较，不能和论文数字对照** —— 论文用的是 7B~1M 级别的模型和真实长文档基准，我这里是个 0.4M 参数的玩具。放在这张表里是为了展示**排序**：query 相关的信号（Quest）在这个任务上把 query 无关的一族甩开了一个数量级。

---

## 5. 什么时候该用 / 不该用

**该用：**

- **目标区域在生成过程中基本不动**（单轮文档问答、摘要、代码库检索、RAG stuffing）。这时 $\rho \approx 0$，观察窗投票（SnapKV）是最优性价比 —— 一次打分、零额外参数、3.6× 生成加速是论文实测的。
- **延迟敏感、显存不紧张**。Quest 不删任何东西，只决定「这步读哪几页」，KV 池大小不变，但每步读的字节少。适合显存够、算力/带宽是瓶颈的场景。
- **极长上下文的 prefill**。MInference 这一类卖的是 TTFT（1M token 从 30 分钟到 3 分钟），不改 KV 缓存策略，只改注意力计算方式。

**不该用：**

- **链式推理 / 长思维链**。$\rho > 0$，观察窗的有效期只有 $1/\rho$ 步，而且推理过程中会反复回到早前算过的东西。NVIDIA 的工程分析明确点名：观察窗不能开大，因为 RoPE 让每个 query 由自己的位置定向，只有最近的约 25 条 query 反映模型实际在看哪里 —— 结果是「某一阶段没被注意到的信息会在那一阶段被驱逐，即使后面的推理依赖它」。
- **你要的是显存，不是带宽**。H2O / SnapKV 在 vLLM 里跑不出论文的显存收益（§ 4.1 问题一：80% 驱逐 → 2.85% 回收）。要显存就用连续窗口（差额 1.0×），或者上块粒度的方案 + 前向压实。
- **显存 <= 权重 + 必须的 KV**。InfiniGen 那种「全量 KV 放 CPU、每步按需取回」的思路在长生成上会被 PCIe 串行化拖垮 —— 有研究报告在 Llama-3.1-8B 上生成 8192 token：vLLM 1.1 分钟 / H2O 3.2 分钟 / InfiniGen 16.4 分钟。
- **还在追最新模型的行为**。这些方法都建立在「注意力分布高度非均匀」这个经验规律上，而这条规律会随训练配方变化（比如推理模型的长 CoT 会让分布更平）。

---

## 6. 常见坑

**坑 1：把 recall 当成唯一验收指标。**
§ 3.5 是最好的反例：观察窗的软召回还有 0.30~0.45（随机基线 0.104），看着「保住了三成质量」；但决定成败的那一位票数排名是 61/65/111/18，**从来没进过 top-32**。原因是那 0.3 是被背景位凑出来的。
**对策**：软召回必须配一个「关键位保住率」。这和 033 期「top-1 一致率会漏掉 logit 分布的漂移」是同一类陷阱 —— **聚合指标会掩盖尾部，而尾部才是失败发生的地方。**

**坑 2：用相关性证明打分信号可靠。**
§ 3.2 里累积注意力的 Spearman ρ = +0.7955，看着很漂亮。但那是因为这个任务里「答案位全部落在前 96 位」—— 累积分数只要能分开「前半段 / 后半段」就能拿到高相关。**Spearman 对整体顺序敏感，而选择策略只关心头部。**
**对策**：看**排名**（答案位在分数里的位次），不看相关系数；或者只看 top-K 区间内的 AUC。

**坑 3：把「省算力」和「省显存」当同一件事。**
§ 4.1 的第一行是极端的例子：保留 80%、驱逐 20%，**一个块都没变空，显存回收率恰好 0**。而 H2O 类方法在论文里的宣传语常是「80% KV cache reduction」——那是**条目数的减少**，不是**显存占用的减少**。两者在块分配器面前差了 28 倍。
**对策**：交付前实测 `nvidia-smi` 的占用，或者直接数「有多少块变成了空块」。

**坑 4：忘了 K 和 V 必须用同一组下标裁剪。**
手动实现时 `K[keep]` 和 `V[keep2]` 用了不同下标（哪怕只是排序不一致），键值就会整体错行 —— **不报任何错**，只是输出质量下降。这类 bug 极难从 loss 曲线上看出来。
**对策**：写一个 assert，检查 `keep_k == keep_v`；或者在离线阶段把 K/V 一起 gather，不让它们分开走。

**坑 5：以为「观察窗越大越准」。**
观察窗的有效期由漂移速度决定（$d^* \approx 1/\rho$），**不由窗口宽度决定**。把 W 从 16 加到 256，在 $\rho = 0.25$ 的任务上并不会让它更耐用（§ 3.5 表 B：d=4 之后就直接掉到谷底，和 W 无关）。而且窗口越大，需要打分的行数越多、工程上越贵（§ 4.2）。
**对策**：先量这个任务/模型层的 $\rho$（把靶位的移动速度测出来），再决定观察窗放哪、放多大。

---

## 7. 一句话总结

**注意力是一个 query × key 的矩阵，而所有「丢谁」的策略都要求一个标量分数 —— 这一步降维是免费的午餐里唯一要付的钱；付得起的场景（目标区域不动）用观察窗，付不起的场景（推理链不固定）只能退回 query 相关估计或干脆别丢；而不管用哪种，token 级的驱逐在 vLLM 的块分配器面前腾不出显存，8 成的驱逐只换回 2.85 成的空间。**

---

<details>
<summary>今日练习</summary>

### 练习 1：把 sink 生出来

§ 2.4 里我的玩具模型没有 sink。请改造任务，让它长出一个：

1. 在序列最前面加一个 BOS token（词表加一个 id），但**不参与**损失；
2. 让一部分 query 没有答案 —— 例如把第二段的 30% 位置换成「第一段从未出现过的符号」，这些位置的目标是一个专门的 `UNK`；
3. 重新训练，然后测量：位置 0 收到的质量、无匹配 query 的注意力熵 vs 有匹配 query 的注意力熵。

**预期**：无匹配 query 的质量会被位置 0 吸收（因为它的 key 是唯一「不用挑」的），位置 0 的质量占比会从 0.005 跳到 0.2 以上，且熵明显下降。
**提示**：相邻两位不要同时是「无答案」位，否则一个 NOVEL token 能在它前面找到另一个 NOVEL，任务就变模糊了。

### 练习 2：验证块空洞的解析式

§ 4.1 的 § 2 段给出了 $P(\text{块空}) = (1-r)^{16}$。请验证它，并回答：

1. 块大小改成 8 / 32 / 64，要做到「驱逐 80% 就回收 80%」，块要多大？
2. 如果驱逐策略不是随机的，而是「保留一段连续的中段」（比如 j ∈ [0.4T, 0.6T]），回收率会变成多少？为什么？

**预期**：`(1-r)^b = 1-r` 要求 $b=1$ —— **只有 token 级分配才能做到驱逐即回收**。而连续中段的对齐方式和末尾窗口一样：中间部分整块满、两头整块空，回收率接近 1.0×。这说明**「连续」比「按内容」更适合 paged 显存**。

### 练习 3：把 Quest 的上界换成更好的估计

§ 3.4 里箱式上界（每维 min/max）偏乐观 2.27 倍。请试两种改进，比较「命中靶子」和「每页元数据字节数」：

1. **分段 min/max**：每页内再分 4 个子段，各存 min/max（元数据 ×4）；
2. **中心 + 半径**：存每页 key 的均值向量 $\mu$ 和残差范数上界 $\rho$，用 $q\cdot\mu + \|q\|\rho$ 当上界（元数据 ≈ 2×dim，和 min/max 一样）。

**提示**：关键不是上界有多紧，而是**排序**对不对。第 2 种在 key 分布接近球对称时排序更好；第 1 种在你只需要「哪一页可能有极大值」时更稳。用 § 3.4 的代码把两者接进去，对比 `上界相对 oracle` 那一列。

</details>
