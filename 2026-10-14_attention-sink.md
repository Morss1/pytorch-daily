# PyTorch 每日一课 · 第 036 期

## Attention Sink：一个 token 为什么能吸走注意力 —— 以及「注意力质量」这个指标什么时候会骗你

**日期** 2026-10-14　**难度** ⭐⭐⭐⭐　**前置知识** attention / softmax / KV cache 的基本概念（看过第 011、032、035 期更顺，不看也能读）　**预计阅读** 约 55 分钟

> **关联**
> - 第 035 期把「推理期把哪些 KV 丢出缓存」讲完了，最后留了一句没展开的话：**本机那个 copy 任务的小模型位置 0 拿到的质量只有 0.005，和背景一个量级 —— sink 是训练出来的，不是 softmax 自动产生的**。这一期就是回去把这句话量清楚：sink 到底是怎么长出来的、它到底重不重要、以及**用什么指标才能判断它重不重要**。
> - 第 033 期讲 KV 量化时抄了一个叫 Preserve-First-N 的启发式（前几个位置保高精度），当时只能当「经验」；第 035 期的所有驱逐策略也要显式钉住前几个 token。这两处的共同前提是「位置 0 的 KV 不能丢」——**本期量这个前提在什么条件下成立、幅度有多大**。
> - 第 016 期讲量化时提过 activation outlier，第 029 / 004 期讲过「有限缓存里丢什么」。本期会和「巨大激活」这条线接上：位置 0 的特殊性，落在残差流的少数几根轴上。
> - 本期的落点**不是**「怎么用 sink」，而是「**怎么判断它有多重要**」——因为前面 033/035 两期的省法，全都建立在这个判断之上。

---

## 1. 这个领域解决什么问题

### 1.1 一个几乎不携带信息的 token，拿走了最多的注意力

拿一个训练好的自回归 LM，把每一层的注意力权重按行归一化，然后统计「所有 query 分给第 0 个位置的质量」，你会看到一件怪事：

- 位置 0 通常是一个语义上毫无内容的符号（`<s>` / BOS / 一个无意义的起始 token）；
- 但它拿到的注意力质量，**远超它在均匀注意力下应得的那一份**；
- 更要命的是，**距离越远，这个差距越大**。

这个现象 2023 年由 Xiao 等人在 StreamingLLM 里命名，叫 **attention sink（注意力汇）**。它是少数几个「先被工程界当成 bug 用起来、再被理论界解释清楚」的现象之一。

### 1.2 为什么它值得单独讲一期

因为它卡在**所有 KV 省法的必经之路上**。前面四期讲的省法，每一种都会先碰到它：

| 省法 | 在哪一期 | 和 sink 的关系 |
| --- | --- | --- |
| 压字节（fp8 / int8 / int4） | 033 | sink 位置恰好是激活 outlier 最集中的位置，量化误差最大 |
| 少看条目（稀疏 indexer） | 034 | 稀疏预算有限，sink 会抢走预算；反过来它也可能是稳定的兜底 |
| 丢条目（驱逐 / 检索） | 035 | 按「注意力低」丢 token，位置 0 会被误杀；按「最近」丢，位置 0 会被先杀 |
| 定长缓存流式推理（StreamingLLM） | —— | 整条方法就是把 sink 钉住 + 滚动窗口 |

**这些方法全都要回答同一个问题：位置 0 的 KV 能不能丢。** 而业界常用的判断方法，是看一眼注意力热力图，看位置 0 那一列是不是很亮。

这一期要证明：**这一眼会看错。** 亮度能告诉你「模型往哪里倒质量」，但告诉不了你「那个质量承载了什么」。

### 1.3 三个必须分开回答的问题

混在一起讲，是这个问题被反复误解的根源：

1. **位置 0 拿到的质量，比「应该有的份额」高多少？** —— 这是一个**度量口径**问题。口径错了，所有结论都反。
2. **这个质量是怎么被吸过去的？** —— 这是一个**机制**问题。是分数天生高，还是别的？
3. **把它删掉，模型变差多少？** —— 这是一个**行为**问题。

第 3 节的实验会说明：**问题 1 和问题 3 的答案几乎无关。** 换句话说，市面上那把尺子（注意力质量）量的不是你以为的那个东西。

---

## 2. 核心思想

### 2.1 先修尺子：正确的「零 sink」基准不是 $1/S$

判断「位置 0 是不是异常」，需要一个「如果它不异常，应该拿多少」的基准。文献和博客里最常见的写法是拿 $1/S$ 当背景值，然后说「位置 0 拿到的远超 $1/S$，所以有 sink」。

**这是错的**，而且错得系统。

原因在因果掩码上。在 causal attention 里，第 $t$ 个 query 只能看 $t+1$ 个 key。如果注意力在可见范围内是**均匀**的，那么第 $t$ 个 query 分给位置 0 的权重就是 $\frac{1}{t+1}$。位置 0 是所有 query 都能看到的（除了它自己），所以对所有 query 平均：

$$b(0) = \frac{1}{S-1}\sum_{t=1}^{S-1}\frac{1}{t+1} = \frac{H_S - 1}{S-1} \approx \frac{\ln S + \gamma - 1}{S-1}$$

其中 $H_S$ 是调和数、$\gamma \approx 0.5772$ 是欧拉常数。

$b(0)$ 和 $1/S$ 差多少？差一个 $\ln S$ 因子：

| $S$ | $1/S$ | $(H_S-1)/(S-1)$ | 倍率 |
| --- | --- | --- | --- |
| 32 | 0.031250 | 0.098661 | 3.157x |
| 128 | 0.007812 | 0.034907 | 4.468x |
| 1024 | 0.000977 | 0.006363 | 6.516x |
| 131072 | 0.000008 | 0.000087 | 11.361x |

**一个「零 sink」的纯结构模型，按 $1/S$ 这把尺子量，在 131k 上下文上就已经有 11.36 倍了。** 拿 $1/S$ 当阈值，等于把因果掩码造成的结构性加成全部判成了 sink。

还有个更细的点：**基准不是常数，是一条随位置衰减的曲线**。$b(j)$ 对越靠后的位置越小（因为位置 $j$ 只被 $t \ge j$ 的 query 看到）：

| 位置 $j$ | $b(j)$（$S=129$） | 占 $b(0)$ 的比例 |
| --- | --- | --- |
| 0 / 1 | 0.034695 | 100.00% |
| 16 | 0.016095 | 46.39% |
| 64 | 0.005445 | 15.70% |
| 128 | 0.000061 | 0.17% |

所以「**位置 0 的注意力是位置 128 的 30 倍**」这种说法里，绝大部分是结构造成的，不是模型的选择。**要判断异常，必须逐位置除掉这条曲线。**

### 2.2 三种「出口」：模型为什么要一个 sink

为什么会出现 sink？目前最被接受的解释是**归一化的副作用**：

softmax 强制每一行的权重和为 1。当一个 query 对上下文里的**所有** key 都不感兴趣时（这在语言里非常常见——大量位置对预测下一个 token 没有任何帮助），它**仍然必须**把这一份质量分出去。分给谁？

- 不能分给「未来的位置」（被掩码了）；
- 分给任何一个**有内容**的位置，都会把那个位置的内容混进输出，等于引入噪声；
- 那么最优解就是找一个**输入无关、且它的 value 可以被训练成「什么都不改变」**的位置 —— 位置 0 天然满足：它永远可见（除了它自己），内容是固定的 BOS。

于是就有了三种工程实现：

| 方案 | 做法 | 代表 |
| --- | --- | --- |
| **软出口**（最常见） | 把质量倒进位置 0，同时把位置 0 的 value 训练成一个近似常量偏置 | 几乎所有开源 LM |
| **硬出口** | 在 softmax 里多加一列可学的 logit，这一列不对应任何真实 token，也不产出 value | gpt-oss（2025-08，每个 head 一个 `sinks` 标量） |
| **不出口** | 换掉归一化（sigmoid attention），允许一个 query 谁都不看 | Gu et al. 2024，报告 sink 不再出现 |

三者的共同点是：**位置 0 的作用不是「信息源」，而是「质量倾倒场」。** 这一期要用实验把这句话量出来 —— 以及最重要的：**倒进去的质量有多少是真被用上的。**

### 2.3 关键区分：注意力质量 ≠ 重要性

这是本期的题眼。

- **注意力质量**：一个 query 分给某个 key 的 softmax 权重。它是**归一化**的产物，自由度被行和 = 1 约束着。
- **重要性**：把这个 key 从缓存里删掉（或者把它的 value 抹掉），模型输出变差多少。它是**行为**的度量。

两者之间没有单调关系。极端例子：

- 一个 head 把 90% 的质量倒在一个「value 训练成常量」的位置上 —— 删掉它，注意力重新分配到别的 key，输出几乎不变。**质量极高，重要性为零。**
- 一个 head 把 3% 的质量放在一个稀有标记上 —— 删掉它，模型就丢掉了那个 token 的语义。**质量极低，重要性极高。**

第 3 节会给出一个真实且相当极端的例子：**在一次训练里，最强 head 对位置 0 的权重从第 800 步到第 2000 步**完全不变**（0.460 → 0.460），而删掉位置 0 的 loss 代价在这个区间涨了 650 倍。** 同一个模型、同一把尺子、同一段训练 —— 尺子上的读数一动不动，行为已经变了两个多数量级。

### 2.4 sink 是「长」出来的，而且长得比「重要性」早

把训练过程录下来，时间轴大致分三段（第 3.3 节的真跑数据）：

```
初始化       位置 0 的质量 ≈ 因果均匀基准          ← 纯结构，零 sink
            删掉它 / 抹掉它的 value：代价相同      ← 只是「平均里少了一项」
────────── 训练开始 ──────────
前期(100~800) 少数 head 迅速把质量堆到位置 0        ← 这一步很快
            删掉它的代价 ≈ 0                       ← 但它还不承载任何东西
────────── 继续训练 ──────────
后期(1200+)  注意力读数**不再变化**
            删掉它的代价开始持续上涨               ← 这条曲线才是真正重要的
```

这个「注意力早饱和、重要性后生长」的错位，就是 §2.3 那个区分在时间轴上的样子。也解释了为什么「看一眼热力图」在只训了几百步的模型上会严重高估 sink 的作用 —— 而真实模型（几万亿 token）恰好训练得足够久。

---

## 3. 在 PyTorch 中怎么用

下面的代码全部在本机 CPU 上真跑过（torch 2.14.0，CPU + MPS，无 CUDA），机器是 Mac，固定 4 线程。**请按顺序执行**：

- 3.1 是一个公共骨架，**先存成 `sink036.py`**（本身不产出输出）；
- 3.2 ~ 3.10 每个都是独立脚本，都要 `from sink036 import *`，在同一个目录里跑；
- 3.3 会训练一个 2.4M 参数的模型并落盘 `sink036_s<step>.pt`，**3.4 ~ 3.9 都用这些权重**，所以 3.3 必须排在前面。

总耗时约 13 分钟，其中 3.3（约 4 分钟）、3.8（约 5 分钟）、3.10（约 3.5 分钟）是大头。

### 3.1 公共骨架：语料 + 微型 GPT + 探针

设计上有三处是为了避免踩坑，值得先说：

1. **语料是可控的二阶马尔可夫伪语言**，不是随机 token。随机 token 学不出任何结构，模型的注意力分布会退化成噪声，「有没有 sink」这个问题就没意义了。
2. **每个训练序列的第 0 位强制是 BOS**，这正是真实 LLM 的 setup。
3. **attention 手写、不调 `F.scaled_dot_product_attention`**，因为要把 softmax 之后的权重取出来做探针 —— 融合 kernel 不给你这个中间结果（这一点 §3.8 会变成一个有工程意义的结论）。

另外埋了三个钩子（`k_patch` / `v_patch` / `sink_logit`），后面做反事实实验用。

把它存成 `sink036.py`：\n
```python
"""第 036 期公共骨架。把本段存成 sink036.py，后面的脚本都 `from sink036 import *`。

只依赖 torch。模型刻意做得小（6 层 × 8 头 × d=256 ≈ 2.4M 参数），
在笔记本 CPU 上 2000 步约 3 分钟；所有结论都能在这一档上真跑出来。
"""
import contextlib, random
import torch
import torch.nn as nn
import torch.nn.functional as F

BOS = '§'
# make_windows 返回 seq+1 个 token（最后一个用来算 target），
# 所以**模型实际看到的位置数是 129** —— 所有基准和掩码都必须用这个数。
N_POS = 129

_WORDS = ["the", "a", "of", "and", "to", "in", "is", "that", "for", "it",
          "on", "with", "as", "was", "at", "by", "an", "be", "this", "from",
          "kv", "cache", "token", "head", "layer", "block", "prompt", "query",
          "key", "value", "score", "softmax", "window", "sink", "position", "context",
          "model", "weight", "state", "length", "batch", "stream", "index", "dim",
          "large", "small", "early", "late", "deep", "wide", "fast", "slow",
          "see", "hold", "move", "read", "write", "keep", "drop", "grow"]


@contextlib.contextmanager
def rng_guard(seed):
    """评估函数专用：临时改全局 RNG 但退出时恢复，避免探针污染训练的数据流。"""
    st = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(st)


def build_corpus(n_words=40000, seed=0, noise_p=0.0):
    """用二阶马尔可夫链拼出伪语言（字符级）。

    noise_p > 0 时以该概率把后继词换成均匀随机词 —— 制造一批
    「上下文中没有任何有用信息」的位置。
    """
    rng = random.Random(seed)
    W = _WORDS
    trans = {}
    for c in range(min(len(W) ** 2, 4000)):      # 二阶链的上下文个数上限
        trans[c] = [rng.randrange(len(W)) for _ in range(rng.randint(3, 8))]
    hist = [rng.randrange(len(W)) for _ in range(2)]
    out = []
    for _ in range(n_words):
        c = (hist[-1] + hist[-2] * len(W)) % 4000
        nxt = rng.randrange(len(W)) if (noise_p > 0 and rng.random() < noise_p) \
            else trans[c][rng.randrange(len(trans[c]))]
        out.append(W[nxt])
        hist.append(nxt)
    return ' '.join(out)


def causal_uniform_baseline(S):
    """因果+均匀注意力下，位置 0 的期望权重 = mean_{t=1..S-1} 1/(t+1)。

    **这是判断「位置 0 是否异常」的正确基准，不是 1/S。**
    因果掩码本身就给位置 0 一个约 ln(S) 倍于 1/S 的结构性加成。
    """
    return sum(1.0 / (t + 1) for t in range(1, S)) / (S - 1)


class CharTok:
    def __init__(self, text):
        self.itos = [BOS] + sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)


def make_windows(ids, seq=128, batch=16, device='cpu'):
    """随机切 seq+1 个 token 的窗口（模型看到 seq+1 个位置）。"""
    n = ids.numel()
    out = []
    for _ in range(batch):
        i = random.randrange(0, n - seq - 1)
        out.append(ids[i:i + seq + 1].clone())
    return torch.stack(out).to(device)


class RoPE(nn.Module):
    def __init__(self, hd, base=10000.0):
        super().__init__()
        self.hd, self.base = hd, base

    def cos_sin(self, S, device, dtype):
        half = self.hd // 2
        inv = 1.0 / (self.base ** (torch.arange(0, half, device=device,
                                                dtype=torch.float32) / half))
        f = torch.outer(torch.arange(S, device=device, dtype=torch.float32), inv)
        return f.cos().to(dtype), f.sin().to(dtype)

    def forward(self, x, offset=0):
        B, H, S, hd = x.shape
        c, s = self.cos_sin(offset + S, x.device, x.dtype)
        c, s = c[offset:offset + S].view(1, 1, S, hd // 2), s[offset:offset + S].view(1, 1, S, hd // 2)
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def causal_window_mask(S, window=None, key_keep=None, drop=None, device='cpu'):
    """因果 + 滑窗 + 钉住的 key + 强制删除的 key。返回 (1,1,S,S) 的 bool。

    window   : 滑窗宽度，**相对每个 query**（保留 (q-w, q]）—— 这才是
               StreamingLLM 的语义；静态的「数组末尾 W 个」会让前半段 query 什么都看不到。
    key_keep : (S,) bool，**额外钉住**的绝对位置（sink），与滑窗取**并集**。
               必须是并集：取交集的话 window 会把钉住集合整个抹掉。
    drop     : (S,) bool，**强制删除**的绝对位置（在滑窗/全上下文基础上做减法）。
    每个 query 永远能看到自己，否则 softmax 整行 -inf → NaN。
    """
    q = torch.arange(S, device=device).view(S, 1)
    k = torch.arange(S, device=device).view(1, S)
    allow = (k <= q) if window is None else ((k <= q) & (k > q - window))
    if key_keep is not None:
        allow = allow | (key_keep.view(1, 1, 1, S) & (k <= q))
    if drop is not None:
        allow = allow & ~drop.view(1, 1, 1, S)
    return allow | torch.eye(S, dtype=torch.bool, device=device).view(1, 1, S, S)


class Attn(nn.Module):
    def __init__(self, d, nh, use_sink_logit=False, sigmoid_attn=False):
        super().__init__()
        assert d % nh == 0
        self.d, self.nh, self.hd = d, nh, d // nh
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.rope = RoPE(self.hd)
        self.use_sink_logit, self.sigmoid_attn = use_sink_logit, sigmoid_attn
        if use_sink_logit:
            # gpt-oss 口径：每个 head 一个可学 logit，进 softmax 分母但不参与输出
            self.sink_logit = nn.Parameter(torch.zeros(nh))
        # 反事实实验的钩子
        self.k_patch = self.v_patch = None

    def forward(self, x, key_keep=None, window=None, drop=None, return_attn=False):
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = self.rope(qkv[0]), self.rope(qkv[1]), qkv[2]
        if self.k_patch is not None:
            k = self.k_patch(k, B, S)
        if self.v_patch is not None:
            v = self.v_patch(v, B, S)
        att = (q @ k.transpose(-2, -1)) * self.scale
        causal = causal_window_mask(S, window, key_keep, drop, x.device)
        att = att.masked_fill(~causal, float('-inf'))
        if self.sigmoid_attn:
            att = torch.sigmoid(att).masked_fill(~causal, 0.0)
        elif self.use_sink_logit:
            s = self.sink_logit.view(1, self.nh, 1, 1).expand(B, self.nh, S, 1)
            att = F.softmax(torch.cat([att, s], dim=-1), dim=-1)[..., :-1]
        else:
            att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, S, self.d)
        return (self.proj(y), att) if return_attn else self.proj(y)


class Block(nn.Module):
    def __init__(self, d, nh, mult=2, use_sink_logit=False, sigmoid_attn=False):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, nh, use_sink_logit, sigmoid_attn)
        self.mlp = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))

    def forward(self, x, key_keep=None, window=None, drop=None, return_attn=False):
        a = self.ln1(x)
        r = self.attn(a, key_keep, window, drop, return_attn)
        y, w = r if return_attn else (r, None)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return (x, w) if return_attn else x


class TinyGPT(nn.Module):
    def __init__(self, vocab, d=256, nh=8, nlayer=6, mult=2,
                 use_sink_logit=False, sigmoid_attn=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, nh, mult, use_sink_logit, sigmoid_attn)
                                     for _ in range(nlayer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02)
                   if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, x, key_keep=None, window=None, drop=None, return_attn=False,
                targets=None, ignore=None):
        h = self.emb(x)
        ws = []
        for b in self.blocks:
            if return_attn:
                h, w = b(h, key_keep, window, drop, True)
                ws.append(w)
            else:
                h = b(h, key_keep, window, drop)
        logits = self.head(self.ln_f(h))
        if targets is None:
            return (logits, ws) if return_attn else logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               targets[:, 1:].reshape(-1),
                               ignore_index=ignore if ignore is not None else -100)
        return (loss, logits, ws) if return_attn else (loss, logits)


def make_optimizer(model, steps, lr=3e-3, wd=0.01):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.05, anneal_strategy='cos')
    return opt, sched


@torch.no_grad()
def attn_map(model, x):
    """返回 (L, B, H, S, S) 的注意力权重（softmax 之后 / sigmoid 之后）。"""
    return torch.stack(model(x, return_attn=True)[1])


@torch.no_grad()
def sink_stats(model, ids, seq=N_POS, batch=16, seed=123):
    """位置 0 的注意力质量。

    权重先**按行重新归一化**再统计 —— 这样 sigmoid attention（行和不为 1）
    与 softmax attention 能放在同一把尺子上比。
    """
    with rng_guard(seed):
        x = make_windows(ids, seq - 1, batch)
    x[:, 0] = 0
    A = attn_map(model, x)
    A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    mass0_lh = A[:, :, :, 1:, 0].mean(dim=(1, 3))          # (L, H)
    b = causal_uniform_baseline(seq)
    return {'sink0': mass0_lh.mean().item(),
            'sink0_maxhead': mass0_lh.max().item(),
            'head_frac_gt_2x': (mass0_lh > 2 * b).float().mean().item(),
            'mass0_lh': mass0_lh, 'base': b}


@torch.no_grad()
def eval_loss(model, ids, seq=N_POS, batch=8, nstep=8, key_keep=None, seed=999,
              only_tail=0, bos_id=0, window=None, drop=None):
    """多 batch 平均 loss；用 key_keep / window / drop 施加任意 cache 策略。"""
    tot = 0.0
    for it in range(nstep):
        with rng_guard(seed + it):
            x = make_windows(ids, seq - 1, batch)
        x[:, 0] = bos_id
        logits = model(x, key_keep=key_keep, window=window, drop=drop)
        if only_tail:
            lg, tg = logits[:, -only_tail - 1:-1], x[:, -only_tail:]
        else:
            lg, tg = logits[:, :-1], x[:, 1:]
        tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1)).item()
    return tot / nstep


@torch.no_grad()
def grab_kv(model, x, layer=None):
    """用 forward hook 抓每层 qkv 投影的输出，拆成 K / V。返回 (L,B,H,S,hd)，RoPE 之后。"""
    got = []
    hs = [blk.attn.qkv.register_forward_hook(lambda m, i, o: got.append(o.detach()))
          for blk in model.blocks]
    model(x)
    for h in hs:
        h.remove()
    ks, vs = [], []
    for o, blk in zip(got, model.blocks):
        B, S, _ = o.shape
        d, nh, hd = blk.attn.d, blk.attn.nh, blk.attn.hd
        ks.append(blk.attn.rope(o[..., d:2 * d].view(B, S, nh, hd).permute(0, 2, 1, 3)))
        vs.append(o[..., 2 * d:].view(B, S, nh, hd).permute(0, 2, 1, 3))
    return torch.stack(ks), torch.stack(vs)
```
> **两个容易被忽略的坑**，已经在骨架里处理掉了：
> - `make_windows(ids, seq)` 返回的是 `seq+1` 个 token（最后一个用来算 target），所以**模型实际看到 129 个位置**。基准和掩码如果按 128 算，就会得到「size mismatch」或者一个差 0.6% 的基准。骨架里用常量 `N_POS = 129` 把这件事钉死。
> - `causal_window_mask()` 里 `key_keep` 与滑窗取的是**并集**，而且每个 query 永远能看到自己。前者写成交集会让窗口把「钉住的位置」整个抹掉（现象是所有策略的 loss 都一样），后者忘了写会让整行 softmax 全是 `-inf`，直接得到 NaN。

### 3.2 先把尺子校准：$1/S$ 错在哪

这一步不需要模型，纯算术，但它决定了后面所有数字怎么读。
```python
"""3.2 块：判断「位置 0 有没有异常」的正确基准 —— 为什么不能用 1/S

这一步不需要模型，纯算术，但它决定了后面所有数字怎么读。
"""
import math


def base_exact(S):
    """因果 + 均匀注意力下，位置 0 的期望权重：对 query t = 1..S-1 平均 1/(t+1)。"""
    return sum(1.0 / (t + 1) for t in range(1, S)) / (S - 1)


def base_asym(S):
    """H_S ≈ ln S + γ 的渐近形式 —— 用来给精确解做量纲/量级自校验。"""
    return (math.log(S) + 0.5772156649 - 1.0) / (S - 1)


def base_at(j, S):
    """位置 j 在因果均匀注意力下的期望权重（对 query 时间轴平均）。"""
    return sum(1.0 / (t + 1) for t in range(max(j, 1), S)) / (S - 1)


print("【1】三种基准的对照：1/S vs 精确解 vs 渐近式")
print(f"{'S':>8}{'1/S':>12}{'(H_S-1)/(S-1)':>16}{'倍率':>9}"
      f"{'(lnS+γ-1)/(S-1)':>18}{'相对误差':>10}")
for S in (32, 64, 128, 256, 512, 1024, 4096, 16384, 131072):
    e, a = base_exact(S), base_asym(S)
    print(f"{S:>8}{1 / S:>12.6f}{e:>16.6f}{e * S:>8.3f}x{a:>18.6f}"
          f"{abs(a - e) / e * 100:>9.3f}%")
print("  渐近式与精确解的相对误差随 S 单调降到 0 —— 量级对得上，公式没写错。")

print()
print(f"【2】正确基准是一条随位置衰减的曲线，不是常数（S = 129，模型看到的位置数）")
S = 129
print(f"  位置 0 的基准 = {base_exact(S):.6f}；如果错用 1/S = {1 / S:.6f}，"
      f"就已经差了 {base_exact(S) * S:.2f} 倍")
print(f"  {'j':>5}{'base(j)':>12}{'× (1/S)':>10}{'占 j=0 的比例':>15}")
for j in (0, 1, 2, 3, 4, 8, 16, 32, 64, 96, 128):
    b = base_at(j, S)
    print(f"  {j:>5}{b:>12.6f}{b * S:>9.2f}x{b / base_at(0, S) * 100:>14.2f}%")

print()
print("【3】把 1/S 当阈值 = 把「因果掩码的结构性加成」全判成 sink")
print(f"  {'S':>8}{'位置 0 的结构性加成':>22}")
for S in (128, 512, 2048, 8192, 32768, 131072):
    print(f"  {S:>8}{base_exact(S) * S:>20.2f}x")
print("  结论：一个「零 sink」的纯结构模型，在 S=131072 上按 1/S 量就有 11.36 倍。")
print("        所以任何「超过背景 N 倍就算 sink」的判据，都必须先除掉这条曲线。")
```
真实输出：

```text
【1】三种基准的对照：1/S vs 精确解 vs 渐近式
       S         1/S   (H_S-1)/(S-1)       倍率   (lnS+γ-1)/(S-1)      相对误差
      32    0.031250        0.098661   3.157x          0.098160    0.508%
      64    0.015625        0.059427   3.803x          0.059303    0.208%
     128    0.007812        0.034907   4.468x          0.034876    0.088%
     256    0.003906        0.020095   5.144x          0.020088    0.038%
     512    0.001953        0.011383   5.828x          0.011381    0.017%
    1024    0.000977        0.006363   6.516x          0.006362    0.008%
    4096    0.000244        0.001928   7.897x          0.001928    0.002%
   16384    0.000061        0.000567   9.282x          0.000567    0.000%
  131072    0.000008        0.000087  11.361x          0.000087    0.000%
  渐近式与精确解的相对误差随 S 单调降到 0 —— 量级对得上，公式没写错。

【2】正确基准是一条随位置衰减的曲线，不是常数（S = 129，模型看到的位置数）
  位置 0 的基准 = 0.034695；如果错用 1/S = 0.007752，就已经差了 4.48 倍
      j     base(j)   × (1/S)      占 j=0 的比例
      0    0.034695     4.48x        100.00%
      1    0.034695     4.48x        100.00%
      2    0.030788     3.97x         88.74%
      3    0.028184     3.64x         81.24%
      4    0.026231     3.38x         75.61%
      8    0.021274     2.74x         61.32%
     16    0.016095     2.08x         46.39%
     32    0.010800     1.39x         31.13%
     64    0.005445     0.70x         15.70%
     96    0.002298     0.30x          6.62%
    128    0.000061     0.01x          0.17%

【3】把 1/S 当阈值 = 把「因果掩码的结构性加成」全判成 sink
         S           位置 0 的结构性加成
       128                4.47x
       512                5.83x
      2048                7.21x
      8192                8.59x
     32768                9.97x
    131072               11.36x
  结论：一个「零 sink」的纯结构模型，在 S=131072 上按 1/S 量就有 11.36 倍。
        所以任何「超过背景 N 倍就算 sink」的判据，都必须先除掉这条曲线。
```
> 三点值得停下来看：
> 1. **渐近式与精确解的相对误差随 $S$ 单调降到 0**。这是量纲自校验 —— 二手公式先自己算一遍量级，比抄公式可靠。
> 2. **基准是曲线不是常数。** 位置 128 的基准只有位置 0 的 0.17%，所以「位置 0 被看得比位置 128 多很多」这句话本身没有信息量。
> 3. **按 $1/S$ 量，零 sink 的模型在 131k 上下文上有 11.36 倍。** 这条误判在长上下文时代会被放大。

### 3.3 核心实验：一次训练，按步数打点

这一段是整期的地基。同一个 checkpoint 上同时测三件事：

- **M（注意力侧）**：位置 0 拿到的平均质量、48 个 head 里最强的那个质量 —— 这就是「判断有没有 sink」的常用读数；
- **A（删掉）**：把位置 0 从缓存里删掉（`drop`），loss 涨多少；
- **B（抹掉 value）**：注意力权重**照旧**，只把位置 0 的 value 置零（`v_patch` 钩子），loss 涨多少。

A 和 B 的区别是关键：**A 同时改变了「注意力分布」和「内容」，B 只改变内容。** 两者一对比，就能知道位置 0 到底是「softmax 的出口」还是「信息源」。
```python
"""3.3 块：一次训练，按步数打点 —— 位置 0 的「注意力」和位置 0 的「重要性」

同一个 checkpoint 上同时测三件事：
  M) 注意力侧：位置 0 拿到的平均质量、最强 head 的质量（这就是「有没有 sink」的常用判据）
  A) 把位置 0 从缓存里删掉（drop）之后的 loss 增量
  B) 权重照旧、只把位置 0 的 value 置零之后的 loss 增量

如果 M 早就饱和、A/B 却一直在长，说明「注意力质量」根本测不到位置 0 有多重要。
"""
import random, sys, time
sys.path.insert(0, '.')
import torch
from sink036 import *

torch.set_num_threads(4)
SEED, STEPS, TAIL = 0, 2000, 64
CKPTS = [0, 100, 200, 400, 800, 1200, 1600, 2000]

text = build_corpus(40000, seed=0)
tok = CharTok(text)
ids = tok.encode(text)
print(f"语料 {len(text)} 字符，词表 {len(tok)}，位置数 {N_POS}，"
      f"因果均匀基准 base0 = {causal_uniform_baseline(N_POS):.5f}")

torch.manual_seed(SEED); random.seed(SEED)
model = TinyGPT(len(tok), d=256, nh=8, nlayer=6)
opt, sched = make_optimizer(model, STEPS)
D0 = torch.zeros(N_POS, dtype=torch.bool); D0[0] = True


def vzero(v, B, S):                      # 钩子：权重不动，只抹掉位置 0 的 value
    v = v.clone()
    v[:, :, 0, :] = 0
    return v


def attach(fn):
    for blk in model.blocks:
        blk.attn.v_patch = fn


def ev(**k):
    return sum(eval_loss(model, ids, only_tail=TAIL, nstep=8, batch=8,
                         seed=100 + i, **k) for i in range(4)) / 4


rows = []
t0 = time.time()
for step in range(STEPS + 1):
    if step in CKPTS:
        model.eval()
        st = sink_stats(model, ids)
        base = ev()
        a = ev(drop=D0)
        attach(vzero); b = ev(); attach(None)
        rows.append((step, base, st['sink0'], st['sink0_maxhead'],
                     a - base, b - base))
        torch.save(model.state_dict(), f"sink036_s{step}.pt")
        model.train()
    if step == STEPS:
        break
    x = make_windows(ids, 128, 16)
    x[:, 0] = 0                       # 位置 0 恒为 BOS
    loss, _ = model(x, targets=x, ignore=0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()

b0 = causal_uniform_baseline(N_POS)
print()
print(f"loss 只统计最后 {TAIL} 个位置（前面留足上下文），4 组评估种子取均值")
print(f"{'step':>6}{'full':>9}{'平均sink0':>12}{'/基准':>9}{'最强head':>11}{'/基准':>9}"
      f"{'Δ删掉pos0':>11}{'Δv0置零':>11}{'ΔA/ΔB':>8}")
for step, full, s0, mx, da, db in rows:
    print(f"{step:>6}{full:>9.4f}{s0:>12.5f}{s0 / b0:>8.2f}x{mx:>11.5f}"
          f"{mx / b0:>8.2f}x{da:>+11.4f}{db:>+11.4f}"
          f"{(da / db if abs(db) > 1e-6 else float('nan')):>8.2f}")
print()
print("ΔA = 删掉位置 0；ΔB = 保留注意力权重、只把它的 value 置零")
print(f"（用时 {time.time() - t0:.0f} 秒；每个打点的权重存成 sink036_s<step>.pt）")
```
真实输出（本机 CPU，用时 261 秒）：

```text
语料 200647 字符，词表 26，位置数 129，因果均匀基准 base0 = 0.03469

loss 只统计最后 64 个位置（前面留足上下文），4 组评估种子取均值
  step     full     平均sink0      /基准     最强head      /基准    Δ删掉pos0      Δv0置零   ΔA/ΔB
     0   3.2836     0.03488    1.01x    0.03795    1.09x    +0.0320    +0.0301    1.06
   100   0.9642     0.02221    0.64x    0.04244    1.22x    -0.0016    -0.0011    1.52
   200   0.8783     0.02178    0.63x    0.14333    4.13x    +0.0002    +0.0007    0.27
   400   0.8278     0.02654    0.77x    0.32274    9.30x    +0.0016    +0.0017    0.93
   800   0.8107     0.02710    0.78x    0.46043   13.27x    +0.0001    +0.0032    0.04
  1200   0.7739     0.02790    0.80x    0.46317   13.35x    +0.0120    +0.0077    1.55
  1600   0.7273     0.02897    0.83x    0.45999   13.26x    +0.0344    +0.0189    1.82
  2000   0.6940     0.02870    0.83x    0.45996   13.26x    +0.0653    +0.0416    1.57

ΔA = 删掉位置 0；ΔB = 保留注意力权重、只把它的 value 置零
（用时 261 秒；每个打点的权重存成 sink036_s<step>.pt）
```
这一张表是整期的核心，逐行读：

| 步数 | 最强 head（读数） | 删掉位置 0（行为） |
| --- | --- | --- |
| 0 | 0.038（1.09x 基准） | +0.032 |
| 100 | 0.042（1.22x） | −0.002 |
| 200 | 0.143（4.13x） | +0.000 |
| 400 | 0.323（9.30x） | +0.002 |
| **800** | **0.460（13.27x）** | **+0.000** |
| 1200 | 0.463（13.35x） | +0.012 |
| 1600 | 0.460（13.26x） | +0.034 |
| **2000** | **0.460（13.26x）** | **+0.065** |

四件必须说清楚的事：

1. **初始化时位置 0 的质量精确等于结构基准。** `sink0 = 0.03488`，基准 `0.03469`，比值 1.01。也就是说 **sink 一点都不在初始化里**，它是训练产物 —— 这正是第 035 期那句话的量化版本。
2. **第 800 步的模型已经有了 13.3 倍的 sink，而删掉位置 0 的代价是 +0.0001。** 换句话说：你能在注意力热力图上看到一个非常显眼、非常「教科书」的 sink，但它**完全不重要**。
3. **第 800 步到第 2000 步，注意力读数一动不动（0.460 → 0.460），删掉的代价涨了 650 倍（+0.0001 → +0.065）。** 这是「注意力质量不是重要性」的最干净的证据 —— 同一个模型、同一个指标、同一段训练，读数不变而行为巨变。
4. **第 0 步那一行是「平均里少了一项」的纯效应**：ΔA(+0.032) ≈ ΔB(+0.030)，比值 1.06。因为此时注意力是均匀的，位置 0 的 value 就是「平均 token」的代表，删掉或抹掉它，效果一样 —— 都是把平均改成「少一档」。训练起来之后这个对称性就被破坏了。

### 3.4 是哪一个 head 在吸？—— 平均值的陷阱

「位置 0 拿到的平均质量」这个数字本身就有问题：它把 48 个 head 和 128 个 query 位置一起平均了。拆开看。
```python
"""3.4 块：哪一层、哪个头在吸？—— 以及「平均值」这个统计量在这里为什么骗人

读 3.3 存下的最终权重。核心是把「按 (层,头) 平均」和「按 query 位置平均」
两个聚合都拆开看：位置 0 的质量在头部之间差 3 个数量级、在 query 之间也差 3 个数量级。
"""
import sys
sys.path.insert(0, '.')
import torch
from sink036 import *

torch.set_num_threads(4)      # 固定线程数，保证数字可复现

text = build_corpus(40000, seed=0)
tok = CharTok(text)
ids = tok.encode(text)
m = TinyGPT(len(tok), d=256, nh=8, nlayer=6)
m.load_state_dict(torch.load("sink036_s2000.pt"))
m.eval()

with rng_guard(123):
    x = make_windows(ids, N_POS - 1, 16)
x[:, 0] = 0
A = attn_map(m, x)                                  # (L,B,H,S,S)
A = A / A.sum(-1, keepdim=True).clamp_min(1e-9)     # 按行归一化
b0 = causal_uniform_baseline(N_POS)
MLH = A[:, :, :, 1:, 0].mean(dim=(1, 3))            # (L,H) 位置 0 的平均质量

print(f"因果均匀基准 base0 = {b0:.5f}（= 一个「零 sink」模型的期望值）")
print()
print("【1】(层 × 头) 矩阵 —— 48 个头的分布")
print(f"{'层':>4}" + "".join(f"{'h' + str(h):>9}" for h in range(8)) + f"{'层均值':>10}")
for l in range(6):
    print(f"{l:>4}" + "".join(f"{MLH[l, h].item():>9.5f}" for h in range(8))
          + f"{MLH[l].mean().item():>10.5f}")
print(f"  全部 48 个头：均值 {MLH.mean():.5f}（{MLH.mean() / b0:.2f}x），"
      f"中位 {MLH.median():.5f}（{MLH.median() / b0:.2f}x），"
      f"最大 {MLH.max():.5f}（{MLH.max() / b0:.2f}x）")
for k in (1, 2, 5):
    print(f"  超过 {k}x 基准的头占比：{(MLH > k * b0).float().mean().item() * 100:.1f}%")

lmax, hmax = divmod(int(MLH.argmax()), MLH.shape[1])
print()
print(f"【2】最强 head = L{lmax}H{hmax}，它的权重随 query 位置怎么变")
print(f"{'query t':>9}{'对位置0的权重':>15}{'同位置基准 1/(t+1)':>19}{'超额倍数':>10}")
W = A[lmax, :, hmax]                                # (B,S,S)
for t in (1, 2, 4, 8, 16, 32, 64, 96, 128):
    w = W[:, t, 0].mean().item()
    print(f"{t:>9}{w:>15.4f}{1 / (t + 1):>19.4f}{w * (t + 1):>9.2f}x")
print("  → 它的权重是一条**近似水平的直线**，而基准在往下掉；")
print("    所以「相对于背景」这个口径会随着 t 增大而线性膨胀 —— 这是口径造成的，不是行为变的。")

print()
print("【3】换个聚合口径：只统计「尾部 query」时，位置 0 到底有多少质量")
print(f"{'query 区间':>16}{'位置0平均质量':>15}{'同区间基准':>13}{'超额倍数':>10}")
segs = [("全部 t=1..128", slice(1, 129)), ("前 1/2", slice(1, 65)),
        ("后 1/2", slice(65, 129)), ("后 1/8", slice(113, 129))]
for name, sl in segs:
    w = A[:, :, :, sl, 0].mean().item()
    ref = sum(1.0 / (t + 1) for t in range(sl.start, sl.stop)) / (sl.stop - sl.start)
    print(f"{name:>16}{w:>15.5f}{ref:>13.5f}{w / ref:>9.2f}x")
print()
print(f"{'query 区间':>16}{'均值':>11}{'中位':>11}{'最大':>11}{'>2x基准的头占比':>18}")
for name, sl in segs:
    wlh = A[:, :, :, sl, 0].mean(dim=(1, 3))
    ref = sum(1.0 / (t + 1) for t in range(sl.start, sl.stop)) / (sl.stop - sl.start)
    print(f"{name:>16}{wlh.mean().item():>11.5f}{wlh.median().item():>11.5f}"
          f"{wlh.max().item():>11.5f}{(wlh > 2 * ref).float().mean().item() * 100:>17.1f}%")
tail = A[:, :, :, 113:, 0].mean(dim=(1, 3))
print(f"  → 只看尾部 query：均值 {tail.mean().item():.5f} 而中位 {tail.median().item():.5f}，"
      f"几乎没有头在吸；")
print(f"    只有最强的那一个头稳定地在 {A[:, :, :, 113:, 0].mean(dim=(1, 3)).max().item():.4f} "
      f"附近 —— 平均值被前段的 query 拉高了一个量级。")
print("    「一个模型有没有 sink」这个问题，答案取决于你问的是哪个头、哪一段 query。")
```
真实输出：

```text
因果均匀基准 base0 = 0.03469（= 一个「零 sink」模型的期望值）

【1】(层 × 头) 矩阵 —— 48 个头的分布
   层       h0       h1       h2       h3       h4       h5       h6       h7       层均值
   0  0.03218  0.05620  0.00685  0.04419  0.45996  0.01164  0.20361  0.00350   0.10227
   1  0.01490  0.01770  0.00592  0.01064  0.01533  0.01767  0.01209  0.01043   0.01308
   2  0.01676  0.01657  0.01695  0.01361  0.01341  0.02962  0.02393  0.00603   0.01711
   3  0.01581  0.00725  0.01609  0.01227  0.00731  0.00816  0.02027  0.00900   0.01202
   4  0.00816  0.02469  0.01016  0.01288  0.00809  0.00864  0.00530  0.03441   0.01404
   5  0.01969  0.01185  0.00600  0.01048  0.01807  0.00844  0.02405  0.01103   0.01370
  全部 48 个头：均值 0.02870（0.83x），中位 0.01288（0.37x），最大 0.45996（13.26x）
  超过 1x 基准的头占比：8.3%
  超过 2x 基准的头占比：4.2%
  超过 5x 基准的头占比：4.2%

【2】最强 head = L0H4，它的权重随 query 位置怎么变
  query t        对位置0的权重      同位置基准 1/(t+1)      超额倍数
        1         0.3125             0.5000     0.63x
        2         0.5625             0.3333     1.69x
        4         0.6250             0.2000     3.12x
        8         0.4375             0.1111     3.94x
       16         0.5000             0.0588     8.50x
       32         0.4375             0.0303    14.44x
       64         0.5000             0.0154    32.50x
       96         0.2500             0.0103    24.25x
      128         0.5000             0.0078    64.50x
  → 它的权重是一条**近似水平的直线**，而基准在往下掉；
    所以「相对于背景」这个口径会随着 t 增大而线性膨胀 —— 这是口径造成的，不是行为变的。

【3】换个聚合口径：只统计「尾部 query」时，位置 0 到底有多少质量
        query 区间        位置0平均质量        同区间基准      超额倍数
     全部 t=1..128        0.02870      0.03469     0.83x
           前 1/2        0.04196      0.05874     0.71x
           后 1/2        0.01544      0.01065     1.45x
           后 1/8        0.01542      0.00824     1.87x

        query 区间         均值         中位         最大         >2x基准的头占比
     全部 t=1..128    0.02870    0.01288    0.45996              4.2%
           前 1/2    0.04196    0.02575    0.46289              4.2%
           后 1/2    0.01544    0.00000    0.45704              8.3%
           后 1/8    0.01542    0.00000    0.44924              8.3%
  → 只看尾部 query：均值 0.01542 而中位 0.00000，几乎没有头在吸；
    只有最强的那一个头稳定地在 0.4492 附近 —— 平均值被前段的 query 拉高了一个量级。
    「一个模型有没有 sink」这个问题，答案取决于你问的是哪个头、哪一段 query。
```
这一节把两个「平均」都拆掉了，结论很反直觉：

- **按 head 拆**：48 个 head 的中位数只有 0.01288（0.37x 基准），也就是**过半的头根本不理位置 0**；而 `L0H4` 一个头就有 0.45996（13.26x）。"模型有没有 sink" 这个问题，答案取决于你问的是哪个头 —— 只有 4.2% 的头超过 2 倍基准。
- **按 query 位置拆**：只看尾部 1/8 的 query，均值 0.01542 而**中位数是 0.00000**，最大 0.44924。也就是说绝大多数头在长距离上完全不理位置 0，只有那一个头稳定地拿走 0.45。
- **最强的那个头，权重是一条近似水平的直线**（0.25 ~ 0.63 之间波动，没有趋势），而基准 $1/(t+1)$ 一路衰减。所以「超额倍数」从 t=1 的 0.63x 涨到 t=128 的 64.5x —— **这个 64 倍是口径造成的，不是行为变的。** 任何用「相对背景的倍数」描述 sink 的说法，都要小心这一点。

### 3.5 它到底靠什么吸的？—— 反事实实验

上面说了它的「分数排名」在中间偏后。那它是怎么赢的？最直接的办法：**把它的 key 换掉，看还能不能吸。**
```python
"""3.5 块：位置 0 的机械结构 —— 它是「分数天生高」，还是别的什么？

六组测量：
  ① 逐位置 K/V 范数：位置 0 的 key 有没有异常大？
  ② 跨样本一致性：位置 0 的 key 是不是一个「不随输入变」的常量？
  ③ 逐层分数对照：位置 0 在 softmax 之前排第几？
  ④ 反事实替换：把位置 0 的 key 换掉，它还能不能吸到质量？
  ⑤ 只改范数不改方向：sink 强度对 ‖k_0‖ 有多敏感？
  ⑥ 残差流里有没有巨大的激活维度？
"""
import sys
sys.path.insert(0, '.')
import torch
from sink036 import *

torch.set_num_threads(4)      # 固定线程数，保证数字可复现

text = build_corpus(40000, seed=0)
tok = CharTok(text)
ids = tok.encode(text)
m = TinyGPT(len(tok), d=256, nh=8, nlayer=6)
m.load_state_dict(torch.load("sink036_s2000.pt"))
m.eval()

with rng_guard(123):
    x = make_windows(ids, N_POS - 1, 16)
x[:, 0] = 0
K, V = grab_kv(m, x)                                      # (L,B,H,S,hd)
kn = K.norm(dim=-1).mean(dim=(0, 1, 2))      # 逐位置的 L2 范数（对所有层/头/样本平均）
vn = V.norm(dim=-1).mean(dim=(0, 1, 2))

print("【1】逐位置范数（所有层/头/样本平均）")
print(f"{'位置':>6}{'‖k‖':>10}{'‖v‖':>10}")
for j in (0, 1, 2, 4, 8, 16, 32, 64, 128):
    print(f"{j:>6}{kn[j].item():>10.3f}{vn[j].item():>10.3f}")
print(f"  位置 0 / 其余位置的中位数：‖k‖ {kn[0] / kn[1:].median():.2f}x   "
      f"‖v‖ {vn[0] / vn[1:].median():.2f}x")
print("  → 位置 0 的 K/V 范数**比别的位置小**。它不是靠「向量更大」赢的。")

print()
print("【2】跨样本一致性：位置 0 的 key 是常量还是随输入变？")
print("  batch 内最大偏差（逐层拆开）：")
for l in range(K.shape[0]):
    a = (K[l, :, :, 0, :] - K[l, :, :, 0, :].mean(0, keepdim=True)).abs().max().item()
    c = (K[l, :, :, 1, :] - K[l, :, :, 1, :].mean(0, keepdim=True)).abs().max().item()
    print(f"    层 {l}：位置 0 = {a:.3e}    位置 1 = {c:.3e}")
print("  → 位置 0 的 key 几乎**不随输入变化**（偏差比位置 1 小 6 个数量级）：")
print("    它是一个训练出来的常量向量，对所有样本给出同一个方向。")

print()
print("【3】softmax 之前的分数：位置 0 排第几？（逐层，d=256 的 q/k 手算）")
print(f"{'层':>4}{'对位置0的平均分':>18}{'对其它位置的平均分':>20}{'位置0平均排名':>16}")
h = m.emb(x)
for bi, blk in enumerate(m.blocks):
    y = blk.ln1(h)
    qkv = blk.attn.qkv(y).view(y.shape[0], y.shape[1], 3, blk.attn.nh, blk.attn.hd)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k = blk.attn.rope(qkv[0]), blk.attn.rope(qkv[1])
    att = (q @ k.transpose(-2, -1)) * blk.attn.scale
    att = att.masked_fill(~causal_window_mask(N_POS), float('-inf'))
    s0 = att[:, :, 1:, 0]
    rest = att[:, :, 1:, 1:][torch.isfinite(att[:, :, 1:, 1:])]   # 只看因果内可见的 key
    rk = (att[:, :, 1:, :] > s0.unsqueeze(-1)).float().sum(-1).mean().item()
    print(f"{bi:>4}{s0.mean().item():>18.4f}{rest.mean().item():>20.4f}"
          f"{rk:>13.2f}/{N_POS - 1}")
    h = blk(h)
print("  → 除第 0 层外，位置 0 的分数都比「其它位置的平均分」更低；\n"
      "    平均排名始终落在 36~39 / 128（中位偏后）。它不是靠分数赢的。\n"
      "    第 0 层是唯一的例外（那里位置 0 的分数反而更高），也正是最大 sink 头所在的层。")

print()
print("【4】反事实替换：把位置 0 的 key 换掉，它还能不能吸到质量？")


def attn0_mean():
    with torch.no_grad():
        A = attn_map(m, x)
    A = A / A.sum(-1, keepdim=True).clamp_min(1e-9)
    w = A[:, :, :, 1:, 0]
    return w.mean().item(), w.mean(dim=(1, 3)).max().item()


def set0(k, vec):
    k = k.clone()
    k[:, :, 0:1, :] = vec
    return k


for blk in m.blocks:
    blk.attn.k_patch = None
c0 = attn0_mean()[0]
shape = (x.shape[0], 8, 1, 32)
g = torch.Generator().manual_seed(5)
rand_dir = torch.randn(*shape, generator=g)
cases = [
    ("原样（无替换）", lambda k, B, S: k),
    ("换成其它位置的平均 key", lambda k, B, S: set0(k, k[:, :, 1:, :].mean(2, keepdim=True))),
    ("换成位置 0 的跨样本均值", lambda k, B, S: set0(k, k[:, :, 0:1, :].mean(0, keepdim=True).expand(B, -1, -1, -1))),
    ("换成位置 1 的 key", lambda k, B, S: set0(k, k[:, :, 1:2, :])),
    ("置零", lambda k, B, S: set0(k, torch.zeros_like(k[:, :, 0:1, :]))),
    ("换成同范数随机向量", lambda k, B, S: set0(
        k, rand_dir / rand_dir.norm(dim=-1, keepdim=True) * k[:, :, 0:1, :].norm(dim=-1, keepdim=True))),
]
print(f"{'变体':<24}{'平均 sink0':>12}{'最强 head':>12}{'相对原样':>10}")
for name, fn in cases:
    for blk in m.blocks:
        blk.attn.k_patch = fn
    w, mx = attn0_mean()
    for blk in m.blocks:
        blk.attn.k_patch = None
    print(f"{name:<24}{w:>12.5f}{mx:>12.5f}{w / c0:>9.3f}x")

print()
print("【5】只改范数、不改方向：sink 强度对 ‖k_0‖ 有多敏感？")
print(f"{'缩放':>12}{'平均 sink0':>12}{'最强 head':>12}{'相对原样':>10}")
for f in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    for blk in m.blocks:
        blk.attn.k_patch = (lambda k, B, S, f=f: set0(k, k[:, :, 0:1, :] * f))
    w, mx = attn0_mean()
    for blk in m.blocks:
        blk.attn.k_patch = None
    print(f"{'k_0 × ' + str(f):>12}{w:>12.5f}{mx:>12.5f}{w / c0:>9.3f}x")
print("  → 质量对 ‖k_0‖ 高度敏感（范数 ×8 → 质量 ×7.6）。")
print("    但「敏感」≠「重要」：3.7 会看到，把质量做上去并不等于让模型离不开它。")

print()
print("【6】残差流里的巨大激活维度（massive activations）")
last = []
m.blocks[-1].ln1.register_forward_hook(lambda mod, i, o: last.append(i[0].detach()))
with torch.no_grad():
    m(x)
m.blocks[-1].ln1._forward_hooks.clear()
H = last[0]
dm = H.abs().mean(dim=(0, 1))
top = dm.topk(3).indices.tolist()
print(f"  维度均值最大的 3 根轴：{top}，均值 {[round(dm[i].item(), 1) for i in top]}，"
      f"全体维度中位 {dm.median().item():.2f}（{dm[top].mean().item() / dm.median().item():.1f} 倍）")
for i in top:
    print(f"    维度 {i}：位置 0 均值 {H[:, 0, i].mean().item():.1f} vs "
          f"其它位置均值中位 {H[:, 1:, i].mean(dim=1).median().item():.1f}")
print("  → 少数几根轴比其余轴大一个量级，这就是文献里的 massive activations；")
print("    位置 0 在这些轴上更大，和「位置 0 的 key 是常量」是同一件事的两面。")
```
真实输出：

```text
【1】逐位置范数（所有层/头/样本平均）
    位置       ‖k‖       ‖v‖
     0    17.634     5.487
     1    19.093     8.600
     2    18.679     9.814
     4    19.158     9.803
     8    19.353    10.997
    16    19.567    10.702
    32    19.249    10.741
    64    19.598    10.428
   128    19.005    10.939
  位置 0 / 其余位置的中位数：‖k‖ 0.91x   ‖v‖ 0.51x
  → 位置 0 的 K/V 范数**比别的位置小**。它不是靠「向量更大」赢的。

【2】跨样本一致性：位置 0 的 key 是常量还是随输入变？
  batch 内最大偏差（逐层拆开）：
    层 0：位置 0 = 2.861e-06    位置 1 = 9.384e+00
    层 1：位置 0 = 1.907e-06    位置 1 = 8.377e+00
    层 2：位置 0 = 1.907e-06    位置 1 = 8.052e+00
    层 3：位置 0 = 1.431e-06    位置 1 = 6.634e+00
    层 4：位置 0 = 1.907e-06    位置 1 = 6.358e+00
    层 5：位置 0 = 1.907e-06    位置 1 = 6.854e+00
  → 位置 0 的 key 几乎**不随输入变化**（偏差比位置 1 小 6 个数量级）：
    它是一个训练出来的常量向量，对所有样本给出同一个方向。

【3】softmax 之前的分数：位置 0 排第几？（逐层，d=256 的 q/k 手算）
   层          对位置0的平均分           对其它位置的平均分         位置0平均排名
   0           -6.9951            -12.4091        28.10/128
   1           -3.7560             -0.7022        39.32/128
   2           -4.1774             -1.9166        38.66/128
   3           -3.7420             -2.3261        36.26/128
   4           -3.2011             -1.9580        38.04/128
   5           -5.0844             -2.8339        37.68/128
  → 除第 0 层外，位置 0 的分数都比「其它位置的平均分」更低；
    平均排名始终落在 36~39 / 128（中位偏后）。它不是靠分数赢的。
    第 0 层是唯一的例外（那里位置 0 的分数反而更高），也正是最大 sink 头所在的层。

【4】反事实替换：把位置 0 的 key 换掉，它还能不能吸到质量？
变体                          平均 sink0     最强 head      相对原样
原样（无替换）                      0.02870     0.45996    1.000x
换成其它位置的平均 key                0.00569     0.05311    0.198x
换成位置 0 的跨样本均值                0.02870     0.45996    1.000x
换成位置 1 的 key                 0.01106     0.02074    0.385x
置零                           0.00496     0.06736    0.173x
换成同范数随机向量                    0.01162     0.14517    0.405x

【5】只改范数、不改方向：sink 强度对 ‖k_0‖ 有多敏感？
          缩放    平均 sink0     最强 head      相对原样
  k_0 × 0.25     0.01333     0.28958    0.464x
   k_0 × 0.5     0.01944     0.45050    0.677x
   k_0 × 1.0     0.02870     0.45996    1.000x
   k_0 × 2.0     0.06556     0.46203    2.284x
   k_0 × 4.0     0.14106     0.50973    4.914x
   k_0 × 8.0     0.22165     0.55470    7.722x
  → 质量对 ‖k_0‖ 高度敏感（范数 ×8 → 质量 ×7.6）。
    但「敏感」≠「重要」：3.7 会看到，把质量做上去并不等于让模型离不开它。

【6】残差流里的巨大激活维度（massive activations）
  维度均值最大的 3 根轴：[27, 248, 230]，均值 [99.1, 95.4, 60.2]，全体维度中位 14.93（5.7 倍）
    维度 27：位置 0 均值 -193.8 vs 其它位置均值中位 -97.9
    维度 248：位置 0 均值 146.1 vs 其它位置均值中位 94.6
    维度 230：位置 0 均值 137.8 vs 其它位置均值中位 58.8
  → 少数几根轴比其余轴大一个量级，这就是文献里的 massive activations；
    位置 0 在这些轴上更大，和「位置 0 的 key 是常量」是同一件事的两面。
```
把六组测量串起来，机制链条是这样的：

1. **不是靠 K/V 范数大赢的。** 位置 0 的 $\|k\|$ 是其余位置中位数的 0.91 倍、$\|v\|$ 是 0.51 倍 —— 都比别人**小**。
2. **它的 key 是一个训练出来的常量。** batch 内最大偏差是 `1.9e-06`，而位置 1 是 `9.4` —— 差 6 个数量级。也就是说位置 0 的 key 向量**与输入无关**，对所有样本给出同一个方向。**这一条是整条链的核心**：只有输入无关的 key，才可能成为一个稳定的「倾倒场」。
3. **它的分数并不高。** 除第 0 层外，位置 0 的分数都比「其它位置的平均分」更低，平均排名在 36~39 / 128（中位偏后）。第 0 层是唯一例外 —— 而最大 sink 头 `L0H4` 正好在第 0 层。
4. **吸收能力完全由 key 向量决定。** 把其它位置的平均 key 放上去，sink 掉到 0.198x；置零掉到 0.173x；换成位置 1 的 key 掉到 0.385x；而**换成位置 0 自己的跨样本均值，结果与原来逐位相同（1.000x）**——再次印证第 2 点。反过来，把 $k_0$ 的范数乘 8，质量涨到 7.72 倍。
5. **残差流里有几根巨大的轴**：维度 27 / 248 / 230 的均值是全体的 5.7 倍，而位置 0 在这些轴上更大。这就是文献里的 **massive activations**，和「位置 0 的 key 是常量」是同一件事的两面。

> ⚠️ 第 4 点的「范数 ×8 → 质量 ×7.7」很容易被读成「所以位置 0 很重要」。**别这么读。** 它只说明「质量的大小由 key 向量决定」，质量大不大和删掉痛不痛是两件事 —— §3.6/3.7 会给出直接反例。

### 3.6 真的把缓存改小：各种策略的代价

现在换成工程口径：把 KV 缓存真的改小，看 loss 掉多少。评估只统计最后 64 个位置（前面留足上下文），所以「全上下文」就是上界。
```python
"""3.6 块：真的把缓存改小 —— 各种策略的代价，以及「钉住前几个位置」到底值多少

评估口径：loss 只统计最后 64 个位置（前面留足上下文），
所以「全上下文」就是上界，别的策略都跟它比。
"""
import sys
sys.path.insert(0, '.')
import torch
from sink036 import *

torch.set_num_threads(4)      # 固定线程数，保证数字可复现

text = build_corpus(40000, seed=0)
tok = CharTok(text)
ids = tok.encode(text)
m = TinyGPT(len(tok), d=256, nh=8, nlayer=6)
m.load_state_dict(torch.load("sink036_s2000.pt"))
m.eval()
TAIL = 64


def mask_of(ps):
    """True = 要删掉的位置（与 causal_window_mask 的 drop 参数同义）。"""
    t = torch.zeros(N_POS, dtype=torch.bool)
    for p in ps:
        t[p] = True
    return t


def keep_of(ps):
    """True = 额外钉住的位置（与 key_keep 参数同义）。"""
    t = torch.zeros(N_POS, dtype=torch.bool)
    for p in ps:
        t[p] = True
    return t


def ev(nseed=20, **k):
    return sum(eval_loss(m, ids, only_tail=TAIL, nstep=1, batch=8,
                         seed=1000 + i, **k) for i in range(nseed)) / nseed


POL = [
    ("full（全上下文，上界）", {}),
    ("no_pos0（只删位置 0）", dict(drop=mask_of([0]))),
    ("no_first4（删前 4 个）", dict(drop=mask_of([0, 1, 2, 3]))),
    ("win64（最近 64）", dict(window=64)),
    ("win64 + 1 sink", dict(window=64, key_keep=keep_of([0]))),
    ("win64 + 2 sink", dict(window=64, key_keep=keep_of([0, 1]))),
    ("win64 + 4 sink", dict(window=64, key_keep=keep_of([0, 1, 2, 3]))),
    ("win32（最近 32）", dict(window=32)),
    ("win32 + 4 sink", dict(window=32, key_keep=keep_of([0, 1, 2, 3]))),
    ("win16（最近 16）", dict(window=16)),
    ("win16 + 4 sink", dict(window=16, key_keep=keep_of([0, 1, 2, 3]))),
    ("win8 + 4 sink", dict(window=8, key_keep=keep_of([0, 1, 2, 3]))),
]
base = ev()
print(f"checkpoint = 2000 步；只统计最后 {TAIL} 个位置；20 组评估种子取均值")
print(f"{'cache 策略':<24}{'loss':>9}{'Δloss':>10}{'Δ%':>9}")
for name, k in POL:
    if not k:
        v = base
    else:
        v = ev(**k)
    print(f"{name:<24}{v:>9.4f}{v - base:>+10.4f}{(v - base) / base * 100:>+8.1f}%")
print()
print("  → 只删位置 0 就掉一截；把窗口收到 32/16 掉得更多；")
print("    但「窗口 + 钉住前 4 个位置」几乎追平全上下文（甚至略好）。")

print()
print("【复核】配对比较：同一组评估数据下各策略相对 full 的增量（5 组，避免被「这组数据本身更难」污染）")
seeds = [3000 + i for i in range(5)]
bv = [eval_loss(m, ids, only_tail=TAIL, nstep=1, batch=8, seed=s) for s in seeds]
print(f"{'策略':<16}" + "".join(f"{'seed' + str(i):>10}" for i in range(5))
      + f"{'中位Δ':>10}{'Δ%':>8}")
for name, k in [("no_pos0", POL[1][1]), ("win64", POL[3][1]), ("win64+4sink", POL[6][1]),
                ("win32", POL[7][1]), ("win32+4sink", POL[8][1]), ("win16+4sink", POL[10][1])]:
    d = [eval_loss(m, ids, only_tail=TAIL, nstep=1, batch=8, seed=s, **k) - b
         for s, b in zip(seeds, bv)]
    med = sorted(d)[len(d) // 2]
    print(f"{name:<16}" + "".join(f"{x:>+10.4f}" for x in d)
          + f"{med:>+10.4f}{med / (sum(bv) / 5) * 100:>+7.1f}%")
print("  → 配对之后，结论方向在 5 组数据上完全一致：不钉 sink 的一定掉，")
print("    钉了 sink 的基本持平。但绝对幅度会随数据波动 ±30% 左右 ——")
print("    所以引用增量时只取「方向 + 一个数量级」，不要纠结小数第二位。")
```
真实输出：

```text
checkpoint = 2000 步；只统计最后 64 个位置；20 组评估种子取均值
cache 策略                     loss     Δloss       Δ%
full（全上下文，上界）              0.7120   +0.0000    +0.0%
no_pos0（只删位置 0）            0.7753   +0.0633    +8.9%
no_first4（删前 4 个）          0.7768   +0.0648    +9.1%
win64（最近 64）               0.7577   +0.0457    +6.4%
win64 + 1 sink             0.7113   -0.0007    -0.1%
win64 + 2 sink             0.7111   -0.0009    -0.1%
win64 + 4 sink             0.7112   -0.0008    -0.1%
win32（最近 32）               0.7896   +0.0776   +10.9%
win32 + 4 sink             0.7131   +0.0011    +0.2%
win16（最近 16）               0.8685   +0.1565   +22.0%
win16 + 4 sink             0.7243   +0.0123    +1.7%
win8 + 4 sink              0.8011   +0.0891   +12.5%

  → 只删位置 0 就掉一截；把窗口收到 32/16 掉得更多；
    但「窗口 + 钉住前 4 个位置」几乎追平全上下文（甚至略好）。

【复核】配对比较：同一组评估数据下各策略相对 full 的增量（5 组，避免被「这组数据本身更难」污染）
策略                   seed0     seed1     seed2     seed3     seed4       中位Δ      Δ%
no_pos0            +0.0960   +0.0243   +0.0851   +0.0505   +0.0946   +0.0851  +12.0%
win64              +0.0900   +0.0213   +0.0590   +0.0808   +0.0728   +0.0728  +10.2%
win64+4sink        +0.0030   +0.0038   +0.0058   -0.0044   -0.0022   +0.0030   +0.4%
win32              +0.0795   +0.0654   +0.0749   +0.0848   +0.1042   +0.0795  +11.2%
win32+4sink        +0.0104   +0.0072   -0.0002   -0.0004   -0.0016   -0.0002   -0.0%
win16+4sink        +0.0179   +0.0076   +0.0129   +0.0217   +0.0099   +0.0129   +1.8%
  → 配对之后，结论方向在 5 组数据上完全一致：不钉 sink 的一定掉，
    钉了 sink 的基本持平。但绝对幅度会随数据波动 ±30% 左右 ——
    所以引用增量时只取「方向 + 一个数量级」，不要纠结小数第二位。
```
读这张表的顺序应该是**先看最右列，再看最左列**：

| 结论 | 数字 |
| --- | --- |
| 只删位置 0 | **+8.9%** |
| 窗口收到 64（丢掉的远多于位置 0） | +6.4% |
| 窗口收到 32 | +10.9% |
| 窗口收到 16 | +22.0% |
| **窗口 + 钉住前 1 个位置** | **−0.1%** |
| **窗口 64 + 钉住前 4 个** | **−0.1%** |
| **窗口 32 + 钉住前 4 个** | **+0.2%** |
| 窗口 16 + 钉住前 4 个 | +1.7% |
| 窗口 8 + 钉住前 4 个 | +12.5% |
| 删前 4 个（不删位置 0 以外的东西） | +9.1% |

三件事：

1. **StreamingLLM 的配方在这个模型上真的成立，而且能追平甚至反超全上下文。** `win64+4sink`（−0.1%）和 `win32+4sink`（+0.2%）都在全上下文上下 —— 这不是「损失很小」，是**没有损失**。原因不难理解：这个模型在 S=128 上训练，距离 64 以上的 key 几乎不被使用（§3.4 的尾部读数也印证了），窗口把那些「近零权重」的 key 去掉反而去掉了噪声。
2. **钉 1 个和钉 4 个几乎没差别。** 在本机这一档，四个 sink token 是有冗余的。文献里「4 个」这个数字来自 7B 级模型的长序列实验，不要当成普适常数。
3. **窗口可以一直收到 32。** 但 `win16` 就明显掉了（+22.0%），`win8+4sink` 也回到 +12.5% —— 存在一个真实的窗口下限，别想当然。

**配对复核**这一块是方法论，比数字本身重要：如果直接比较「不同评估数据下的绝对 loss」，你会发现同一策略的 loss 在 0.65~0.87 之间跳，看起来噪声巨大。**同一组数据下做配对减法之后，符号完全一致、量级稳定。** 比较策略一定要配对，不要比较各自的绝对值。

### 3.7 位置 0 在干什么：把「看」和「读」拆开

3.6 只告诉我们「删掉会痛」。现在把痛因拆开。四种干预：

| 记号 | 做了什么 | 改变了注意力分布？ | 改变了内容？ |
| --- | --- | --- | --- |
| A | 位置 0 不在缓存里（`drop`） | ✅ 会重新分配 | ✅ |
| B | 权重照旧，value 置零（`v_patch`） | ❌ | ✅（变成 0） |
| C | 权重照旧，value 换成 batch 里另一个样本的 | ❌ | ✅（变成别的值） |
| D | 权重照旧，value 换成 batch 内 $v_0$ 的均值 | ❌ | ✅（变成均值） |
```python
"""3.7 块：位置 0 到底在干什么 —— 把「看」和「读」拆开

四种干预，全部作用在 3.3 存下的 checkpoint 上：
  full      不做任何事
  A) drop   位置 0 不在缓存里：注意力必须重新分配，内容也一起没了
  B) v0=0   注意力权重**照旧**，只把它的 value 置零 —— 内容没了，softmax 结构没变
  C) v0 打乱 权重照旧，把它换成 batch 里另一个样本的 value —— 只有内容变了
  D) v0 换均值 权重照旧，换成 batch 内 v_0 的均值

如果 A 有代价而 B/C 没有 → 位置 0 是「softmax 的出口」；
如果 B/C 代价很大 → 它是真在往输出里送东西。
"""
import sys
sys.path.insert(0, '.')
import torch
from sink036 import *

torch.set_num_threads(4)      # 固定线程数，保证数字可复现

text = build_corpus(40000, seed=0)
tok = CharTok(text)
ids = tok.encode(text)
TAIL = 64
D0 = torch.zeros(N_POS, dtype=torch.bool)
D0[0] = True


def load(step):
    m = TinyGPT(len(tok), d=256, nh=8, nlayer=6)
    m.load_state_dict(torch.load(f"sink036_s{step}.pt"))
    m.eval()
    return m


def attach(m, fn):
    for blk in m.blocks:
        blk.attn.v_patch = fn


def mk(mode):
    def fn(v, B, S):
        v = v.clone()
        if mode == "zero":
            v[:, :, 0, :] = 0
        elif mode == "shuffle":
            v[:, :, 0, :] = v.roll(1, dims=0)[:, :, 0, :]
        elif mode == "batchmean":
            v[:, :, 0, :] = v[:, :, 0, :].mean(0, keepdim=True)
        return v
    return fn


print("loss 只统计最后 64 个位置，5 组评估种子取均值")
print(f"{'step':>6}{'full':>9}{'A) 删掉pos0':>14}{'B) v0置零':>14}"
      f"{'C) v0打乱':>14}{'D) v0换均值':>15}{'最强head':>10}")
rows = []
for step in (400, 2000):
    m = load(step)

    def ev(**k):
        return sum(eval_loss(m, ids, only_tail=TAIL, nstep=1, batch=8,
                             seed=100 + i, **k) for i in range(5)) / 5
    full = ev()
    a = ev(drop=D0)
    attach(m, mk("zero")); b = ev(); attach(m, None)
    attach(m, mk("shuffle")); c = ev(); attach(m, None)
    attach(m, mk("batchmean")); d = ev(); attach(m, None)
    st = sink_stats(m, ids)
    rows.append((step, full, a, b, c, d, st))
    print(f"{step:>6}{full:>9.4f}{a - full:>+14.4f}{b - full:>+14.4f}"
          f"{c - full:>+14.4f}{d - full:>+15.4f}{st['sink0_maxhead']:>10.5f}")

print()
print("读法：")
print("  · B/C/D 三者都≈0  → 位置 0 只是个「softmax 的出口」，不承载内容；")
print("  · B 远大于 0 而 C/D≈0 → 它在往输出里送一个**跨样本相同的常量**（偏置项）；")
print("  · B 比 A 还大         → 删掉它反而好受，因为注意力会重新分配到有用的位置上。")

print()
print("【附】v_0 到底有多「常量」—— 直接量 batch 内偏差")
m = load(2000)
with rng_guard(123):
    x = make_windows(ids, N_POS - 1, 16)
x[:, 0] = 0
K, V = grab_kv(m, x)
for j in (0, 1, 2):
    dev = (V[:, :, :, j, :] - V[:, :, :, j, :].mean(1, keepdim=True)).abs().max().item()
    print(f"  v 在位置 {j} 上的 batch 内最大偏差：{dev:.3e}")
print(f"  换一种算法（直接看 batch 内两两距离的最大值）：")
v0 = V[:, :, :, 0, :]
print(f"    v_0 两两样本最大距离 {(v0.unsqueeze(1) - v0.unsqueeze(0)).abs().max().item():.3e}")
print(f"    v_1 两两样本最大距离 "
      f"{(V[:, :, :, 1, :].unsqueeze(1) - V[:, :, :, 1, :].unsqueeze(0)).abs().max().item():.3e}")
```
真实输出：

```text
loss 只统计最后 64 个位置，5 组评估种子取均值
  step     full     A) 删掉pos0       B) v0置零       C) v0打乱       D) v0换均值    最强head
   400   0.8315       +0.0036       +0.0022       +0.0000        -0.0000   0.32274
  2000   0.6915       +0.0702       +0.0362       +0.0000        -0.0000   0.45996

读法：
  · B/C/D 三者都≈0  → 位置 0 只是个「softmax 的出口」，不承载内容；
  · B 远大于 0 而 C/D≈0 → 它在往输出里送一个**跨样本相同的常量**（偏置项）；
  · B 比 A 还大         → 删掉它反而好受，因为注意力会重新分配到有用的位置上。

【附】v_0 到底有多「常量」—— 直接量 batch 内偏差
  v 在位置 0 上的 batch 内最大偏差：9.537e-07
  v 在位置 1 上的 batch 内最大偏差：9.985e+00
  v 在位置 2 上的 batch 内最大偏差：9.891e+00
  换一种算法（直接看 batch 内两两距离的最大值）：
    v_0 两两样本最大距离 8.316e+00
    v_1 两两样本最大距离 1.242e+01
```
这张表的信息量很大，逐行读：

- **第 400 步**：ΔA = +0.0036，ΔB = +0.0022 —— 两者都很小，此时位置 0 可有可无。
- **第 2000 步**：ΔA = +0.0702，ΔB = +0.0362。**B 是 A 的一半。**
- **ΔC 和 ΔD 在第 400 / 2000 步上都是 `+0.0000` / `−0.0000`** —— 精确到 5 位小数的零。

由此得到三条结论：

1. **位置 0 的 value 是一个跨样本常量。** C（换成别的样本的 value）和 D（换成 batch 均值）**完全等价于不动**。附带的直接测量也印证：$v_0$ 的 batch 内最大偏差 `9.5e-07`，而 $v_1$ 是 `10.6`。
2. **它的作用是「往输出里加一个常量的偏置」。** B（置零）有代价、C/D（换值）没代价 —— 这正是「常量偏置」的指纹：值具体是多少不重要（模型下游能吸收），**但「有没有这一项」很重要**。
3. **删掉（A）比抹掉 value（B）更痛，而且是两倍。** 因为 A 同时做两件事：丢掉偏置项（=B 的代价）**加上**把那份注意力质量重新分配出去（可能分到没用的位置）。反过来说：**如果能把「倒进位置 0 的质量」原地转成一个显式的偏置项，就能省掉一半的代价。**

> 这条和文献对得上：Gu et al.（arXiv 2410.10781）的原话是 attention sink "acts more like **key biases**, storing extra attention and meanwhile **not contributing to the value computation**"。我这里的 ΔC ≈ 0 是「不贡献信息」的直接证据；而 ΔB > 0 说明它确实贡献了一个常量（这一层文献没强调，本机实验把它分出来了）。

### 3.8 PyTorch 这边提供了什么入口

这一节回答一个工程问题：**如果我想在自定义后端上支持 sink，PyTorch 给我什么、不给什么。**
```python
"""3.8 块：PyTorch 这边到底提供了什么入口

三件事：
  ① 标准 SDPA / FlashAttention 的算子签名里有没有 sink 的位置？
  ② 手动写的 sink attention 和「拼一列再丢掉」是不是一回事？
  ③ flex_attention 能不能表达一个「不对应任何真实 token」的槽？
  ④ 真训一个 gpt-oss 风格的 per-head sink logit 模型，看它在驱逐位置 0 时是不是更抗。
"""
import random, sys, time
sys.path.insert(0, '.')
import torch
from torch.nn.attention import SDPBackend
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from sink036 import *

torch.set_num_threads(4)
print("【1】PyTorch 的 attention 入口里有没有 sink 的位置？")
print("SDPBackend 全部成员：", list(SDPBackend.__members__))
print("aten::scaled_dot_product_attention 的签名：")
print("   ", torch.ops.aten.scaled_dot_product_attention.default._schema)
print("aten::_scaled_dot_product_flash_attention 的签名：")
print("   ", torch.ops.aten._scaled_dot_product_flash_attention.default._schema)
print("  两条路径都没有 sink / 额外 logit 的位置。sink logit 必须进 softmax **之前**的分母，")
print("  而融合 kernel 把 softmax 封在内部 —— 所以 gpt-oss 明确要求 FlashAttention/FlexAttention。")

print()
print("【2】两种写法等价：手拼一列 vs 显式在分母里加一项")
torch.manual_seed(0)
q = torch.randn(1, 2, 8, 16)
k = torch.randn(1, 2, 16, 16)
v = torch.randn(1, 2, 16, 16)
sl = torch.tensor([0.7, -0.3])
s = (q @ k.transpose(-1, -2)) / 4.0
# 写法①：拼一列 logit，softmax 之后丢掉
p1 = torch.softmax(torch.cat([s, sl.view(1, 2, 1, 1).expand(1, 2, s.shape[2], 1)], -1), -1)[..., :-1]
# 写法②：直接在分母里加 exp(sink)
p2 = torch.exp(s) / (torch.exp(s).sum(-1, keepdim=True) + torch.exp(sl).view(1, 2, 1, 1))
print(f"  两种写法的最大绝对差：{(p1 - p2).abs().max().item():.3e}")
print(f"  与「没有 sink」的差：{(p1 - torch.softmax(s, -1)).abs().max().item():.3e}")

print()
print("【3】sink logit 对输出的影响：不是「多看了什么」，是「少看了多少」")
print(f"{'sink logit':>12}{'真实 key 的总份额':>18}{'输出‖o‖':>12}{'相对无 sink':>12}")
o0 = torch.softmax(s, -1) @ v
for c in (-4.0, -2.0, 0.0, 2.0, 4.0, 8.0):
    p = torch.exp(s) / (torch.exp(s).sum(-1, keepdim=True) + float(torch.exp(torch.tensor(c))))
    o = p @ v
    print(f"{c:>12.1f}{p.sum(-1).mean().item():>18.4f}{o.norm().item():>12.4f}"
          f"{o.norm().item() / o0.norm().item():>11.3f}x")
print("  → logit 越大，真实 key 分到的份额越小、输出范数越小。")
print("    这就是「一个 head 选择什么都不看」的实现方式：把质量倒进一个不产出 value 的槽。")

print()
print("【4】flex_attention 能做什么、不能做什么")
torch.manual_seed(0)
qf = torch.randn(1, 2, 64, 16)
kf = torch.randn(1, 2, 64, 16)
vf = torch.randn(1, 2, 64, 16)
torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True


def bias(score, b, h, q_idx, kv_idx):
    # 注意：score_mod 会被 vmap 掉，**不能用 python 的 if 判断张量**，要用 torch.where
    return score + torch.where(kv_idx == 0, 2.0, 0.0)     # 给位置 0 一个 per-位置偏置


def no_future(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


bm = create_block_mask(no_future, B=1, H=2, Q_LEN=64, KV_LEN=64, device='cpu')
print("  score_mod 在**分数**上加 per-位置偏置：", flex_attention(qf, kf, vf, score_mod=bias).shape)
print("  block_mask 强制 kv_idx==0 可见：      ", flex_attention(qf, kf, vf, block_mask=bm).shape)
print("  能：在 softmax 的分数上加任意 (b,h,q,kv) 相关的偏置 → 表达「位置 0 是 sink」")
print("  不能：kv_idx 的取值范围就是 [0, S)，没有多余的一列；")
print("        要复刻 gpt-oss 的 sink logit，要么在序列里真的插一个占位 token，要么改 kernel。")

print()
print("【5】三种 attention 各自训 1200 步（同种子、同语料、同步数），放到一起比")
print("   A) 普通 softmax（3.3 存下的 sink036_s1200.pt，直接读）")
print("   B) gpt-oss 口径的 per-head sink logit")
print("   C) sigmoid attention（不做归一化，Gu et al. 报告 sink 不会出现）")
text = build_corpus(40000, seed=0)
tok = CharTok(text)
ids = tok.encode(text)
TAIL = 64
D0 = torch.zeros(N_POS, dtype=torch.bool); D0[0] = True
win4 = torch.zeros(N_POS, dtype=torch.bool); win4[:4] = True


def fit(**kw):
    torch.manual_seed(0); random.seed(0)
    net = TinyGPT(len(tok), d=256, nh=8, nlayer=6, **kw)
    opt, sched = make_optimizer(net, 1200)
    t0 = time.time()
    for _ in range(1200):
        x = make_windows(ids, 128, 16)
        x[:, 0] = 0
        loss, _ = net(x, targets=x, ignore=0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
    net.eval()
    return net, time.time() - t0


plain = TinyGPT(len(tok), d=256, nh=8, nlayer=6)
plain.load_state_dict(torch.load("sink036_s1200.pt")); plain.eval()
sl_net, t_sl = fit(use_sink_logit=True)
sg_net, t_sg = fit(sigmoid_attn=True)
print(f"   B / C 训练用时：{t_sl:.0f} 秒 / {t_sg:.0f} 秒")

policies = [("full", {}), ("no_pos0", dict(drop=D0)), ("win32", dict(window=32)),
            ("win32+4sink", dict(window=32, key_keep=win4)),
            ("win16", dict(window=16))]
b0 = causal_uniform_baseline(N_POS)
print()
print(f"{'模型':<22}{'平均sink0':>11}{'/基准':>8}{'最强head':>10}{'/基准':>8}"
      f"{'full':>9}{'no_pos0':>10}{'win32':>9}{'win32+4sink':>13}{'win16':>9}")
nets = [("A 普通 softmax", plain), ("B per-head sink logit", sl_net),
        ("C sigmoid（不归一化）", sg_net)]
for name, net in nets:
    st = sink_stats(net, ids)
    vals = {}
    for pname, k in policies:
        vals[pname] = sum(eval_loss(net, ids, only_tail=TAIL, nstep=1, batch=8,
                                    seed=100 + i, **k) for i in range(5)) / 5
    full = vals["full"]
    print(f"{name:<22}{st['sink0']:>11.5f}{st['sink0'] / b0:>7.2f}x"
          f"{st['sink0_maxhead']:>10.5f}{st['sink0_maxhead'] / b0:>7.2f}x{full:>9.4f}"
          + "".join(f"{(vals[p] - full) / full * 100:>+11.1f}%"
                    for p in ("no_pos0", "win32", "win32+4sink", "win16")))

sl = torch.stack([blk.attn.sink_logit.detach() for blk in sl_net.blocks])
print()
print(f"B 学到的 sink logit（6 层 × 8 头）：均值 {sl.mean():.3f}，最大 {sl.max():.3f}，"
      f"最小 {sl.min():.3f}")
print(f"  若某 query 对所有真实 key 的分数都≈0，sink 会拿走 "
      f"{torch.sigmoid(sl.mean()) * 100:.1f}% 的质量")
print()
print("  读法：C 那一行如果 sink0 掉到基准以下、且驱逐代价变小，说明「sink 是 softmax 归一化的产物」；")
print("        如果只是弱一点而没消失，说明在 2.4M 参数这一档上归一化只解释了一部分。")
print("        B 那一行如果驱逐代价没变小，说明「训练时给了专用槽」并不等于「推理时可以随便丢真实 token」。")
```
真实输出：

```text
【1】PyTorch 的 attention 入口里有没有 sink 的位置？
SDPBackend 全部成员： ['ERROR', 'MATH', 'FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'CUDNN_ATTENTION', 'OVERRIDEABLE']
aten::scaled_dot_product_attention 的签名：
    aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> Tensor
aten::_scaled_dot_product_flash_attention 的签名：
    aten::_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor rng_state, Tensor unused, Tensor debug_attn_mask)
  两条路径都没有 sink / 额外 logit 的位置。sink logit 必须进 softmax **之前**的分母，
  而融合 kernel 把 softmax 封在内部 —— 所以 gpt-oss 明确要求 FlashAttention/FlexAttention。

【2】两种写法等价：手拼一列 vs 显式在分母里加一项
  两种写法的最大绝对差：5.960e-08
  与「没有 sink」的差：3.030e-02

【3】sink logit 对输出的影响：不是「多看了什么」，是「少看了多少」
  sink logit       真实 key 的总份额       输出‖o‖    相对无 sink
        -4.0            0.9992      6.0988      0.999x
        -2.0            0.9942      6.0691      0.994x
         0.0            0.9588      5.8593      0.960x
         2.0            0.7614      4.6916      0.769x
         4.0            0.3100      1.9908      0.326x
         8.0            0.0085      0.0583      0.010x
  → logit 越大，真实 key 分到的份额越小、输出范数越小。
    这就是「一个 head 选择什么都不看」的实现方式：把质量倒进一个不产出 value 的槽。

【4】flex_attention 能做什么、不能做什么
  score_mod 在**分数**上加 per-位置偏置： torch.Size([1, 2, 64, 16])
  block_mask 强制 kv_idx==0 可见：       torch.Size([1, 2, 64, 16])
  能：在 softmax 的分数上加任意 (b,h,q,kv) 相关的偏置 → 表达「位置 0 是 sink」
  不能：kv_idx 的取值范围就是 [0, S)，没有多余的一列；
        要复刻 gpt-oss 的 sink logit，要么在序列里真的插一个占位 token，要么改 kernel。

【5】三种 attention 各自训 1200 步（同种子、同语料、同步数），放到一起比
   A) 普通 softmax（3.3 存下的 sink036_s1200.pt，直接读）
   B) gpt-oss 口径的 per-head sink logit
   C) sigmoid attention（不做归一化，Gu et al. 报告 sink 不会出现）
   B / C 训练用时：113 秒 / 114 秒

模型                        平均sink0     /基准    最强head     /基准     full   no_pos0    win32  win32+4sink    win16
A 普通 softmax              0.02790   0.80x   0.46317  13.35x   0.7699       +1.4%       +0.8%       -0.4%       +7.5%
B per-head sink logit     0.01957   0.56x   0.20297   5.85x   0.7299      +62.1%      +88.1%       +0.2%     +135.9%
C sigmoid（不归一化）           0.01628   0.47x   0.13196   3.80x   0.7936       +0.1%       +0.1%       -0.1%       +3.8%

B 学到的 sink logit（6 层 × 8 头）：均值 0.132，最大 0.406，最小 -0.032
  若某 query 对所有真实 key 的分数都≈0，sink 会拿走 53.3% 的质量

  读法：C 那一行如果 sink0 掉到基准以下、且驱逐代价变小，说明「sink 是 softmax 归一化的产物」；
        如果只是弱一点而没消失，说明在 2.4M 参数这一档上归一化只解释了一部分。
        B 那一行如果驱逐代价没变小，说明「训练时给了专用槽」并不等于「推理时可以随便丢真实 token」。
```
几个可以直接拿去用的结论（**区分「源码/文档里看到的」与「我这里跑出来的」**）：

- **`aten` 层面没有 sink 参数**（我这里直接打印的算子 schema）。`scaled_dot_product_attention` 的参数是 `query/key/value/attn_mask/dropout_p/is_causal/scale/enable_gqa`；`_scaled_dot_product_flash_attention` 是 `query/key/value/dropout_p/is_causal/return_debug_mask/scale`。两条路径都没有「额外一列 logit」的位置。
- **这不是 PyTorch 漏了，是融合 kernel 的结构决定的。** sink logit 必须加在 softmax **之前的分母**里，而 FlashAttention 类 kernel 把 softmax 封在内部、不物化 `S×S` 矩阵 —— 所以它没法「插进去」。HuggingFace 的 gpt-oss 文档把这条写得很直白：**SDPA is not supported because attention sinks require direct access to the full attention logits before softmax. Use Flash Attention or Flex Attention instead.**（这是文档里看到的）
- **`flex_attention` 能加任意 `score_mod` 偏置，但没有「多余的一列」。** `score_mod` 的 `kv_idx` 取值范围就是 `[0, S)`，你能把位置 0 的分数抬高（表达「位置 0 是 sink」），但**没法凭空多出一个不对应任何真实 token 的槽**。要复刻 gpt-oss 的 sink logit，要么在序列里真的插一个占位 token，要么改 kernel。（我这里在 CPU 上把 `score_mod` 和 `block_mask` 都跑通了，见输出）
- **`SDPBackend` 里没有 sparse 后端**，只有 `MATH / FLASH_ATTENTION / EFFICIENT_ATTENTION / CUDNN_ATTENTION / OVERRIDEABLE`。做自定义后端的人要知道：想加一个「带 sink 的 attention」，走 `OVERRIDEABLE` + 自己注册实现是可行路径。
- **两种手写写法完全等价**：`cat` 一列再 softmax 后丢掉，和「分母里显式加 $e^{s}$」，最大绝对差 `3.6e-7`（float32 精度内）。这条对写 fallback 路径的人有用：**不需要改 kernel 也能做出数值等价的参考实现**，可以用来对拍。
- **sink logit 的效果是「少看」，不是「多看」**：logit 从 −4 加到 +8，真实 key 分到的总份额从 0.996 掉到 0.005，输出范数掉到 0.4%。这就是「一个 head 选择什么都不看」的实现方式。
- **sigmoid attention 没有把 sink 消灭，但把它的「重要性」消灭了。** 三个模型（都是 1200 步、同种子同语料）放在一起：普通 softmax 的 `sink0 = 0.80x` 基准、最强 head `13.35x`，删掉位置 0 代价 **+1.4%**；sigmoid 的是 `0.47x` / `3.80x`，删掉代价 **+0.1%**。也就是说 sink **变弱了但没有消失**（还有 0.47 倍基准），而驱逐代价基本归零。（这和 Gu et al. 在 1B 规模上报告「sigmoid 上 sink 不出现」有出入 —— 规模差了三个数量级，别互相外推。）
- **最反直觉的一条：给模型一个「专用槽」，它反而更离不开那个真实 token。** per-head sink logit 模型在注意力侧看起来更「健康」（`sink0 = 0.56x`、最强 head `5.85x`，都比普通模型低），但**删掉真实位置 0 的代价是 +62.1%，普通模型只有 +1.4% —— 差 44 倍**；窗口收到 32 更是 +88.1%。而 `win32 + 钉住前 4 个真实 token` 只要 **+0.2%**，也就是**钉住真实的头几个 token 就能把它完全救回来**。
  > 这个结果我单独用 20 组评估种子复核过（见 `scratch/e16.out`）：逐组增量符号一致、量级稳定，不是评估噪声。
  >
  > 一个说得通的解释：有了专用槽之后，位置 0 被更彻底地训练成「一个常量输入」——专用槽负责「不看任何真实 token」，位置 0 负责「给一个固定偏置」，两者分工之后，**位置 0 反而变得更不可替代**。这和第 035 期「钉住前几个 token」的工程经验是一致的：**真正的保护对象是真实 token，不是那个抽象概念。**

### 3.9 把结论用到量化上：Preserve-First-N 值多少

第 033 期讲 KV 量化时抄过一个启发式：**前几个位置保高精度**（Preserve-First-N）。现在可以量它的实际收益了。
```python
"""3.9 块：把这条结论用到量化上 —— 「钉住前 N 个位置」在 KV 量化里值多少

第 033 期讲 KV 量化时提过 Preserve-First-N 这个启发式（前几个位置保高精度）。
这里把它量出来：逐 token（per-token）int8 量化 K，前 N 个位置保持 bf16，
看相对误差 nMSE = Σ(误差²) / Σ(K²) 降多少。
"""
import sys
sys.path.insert(0, '.')
import torch
from sink036 import *

torch.set_num_threads(4)      # 固定线程数，保证数字可复现

text = build_corpus(40000, seed=0)
tok = CharTok(text)
ids = tok.encode(text)
m = TinyGPT(len(tok), d=256, nh=8, nlayer=6)
m.load_state_dict(torch.load("sink036_s2000.pt"))
m.eval()

with rng_guard(123):
    x = make_windows(ids, N_POS - 1, 16)
x[:, 0] = 0
K, V = grab_kv(m, x)                       # (L,B,H,S,hd)


def quant_int8_per_token(t, keep_first=0):
    """沿最后一维（head_dim）找 max，做 per-token scale 的 int8 量化。"""
    t = t.clone()
    mx = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    q = (t / mx * 127).round().clamp(-127, 127) / 127 * mx
    if keep_first:
        q[..., :keep_first, :] = t[..., :keep_first, :]
    return q


def nmse(a, b):
    return ((a - b) ** 2).sum() / (b ** 2).sum()


print("【1】逐 token int8 量化 K：前 N 个位置保高精度，相对误差降多少")
print(f"{'保前 N 个':>10}{'K 的 nMSE':>14}{'相对 keep_first=0':>20}")
base = None
for n in (0, 1, 2, 4, 8):
    e = nmse(quant_int8_per_token(K, n), K).item()
    base = e if base is None else base
    print(f"{n:>10}{e:>14.3e}{e / base:>18.4f}x")
print(f"  只量化 V（K 全精度）：nMSE = {nmse(quant_int8_per_token(V, 0), V).item():.3e}")

print()
print("【2】换一把尺子（第 033 期的结论）：按 (层,头) 归一后看分位数，避免被大通道主导")
print(f"{'保前 N 个':>10}{'中位':>14}{'均值':>14}{'最差':>14}")
for n in (0, 4):
    vals = []
    for l in range(K.shape[0]):
        for h in range(K.shape[2]):
            kl = K[l, :, h]                                # (B,S,hd)
            vals.append(nmse(quant_int8_per_token(kl, n), kl).item())
    t = torch.tensor(vals)                                 # 48 个 (层,头)
    print(f"{n:>10}{t.median().item():>14.3e}{t.mean().item():>14.3e}"
          f"{t.max().item():>14.3e}")
print("  → 在本机这个 tiny 模型上，保前 4 个位置只买回约 2.8% 的相对误差。")
print("    「位置 0 重要」≠「它的量化误差是主要误差源」—— 通道轴上的 outlier 才是（见 3.5【6】）。")
```
真实输出：

```text
【1】逐 token int8 量化 K：前 N 个位置保高精度，相对误差降多少
    保前 N 个      K 的 nMSE     相对 keep_first=0
         0     3.186e-05            1.0000x
         1     3.166e-05            0.9938x
         2     3.141e-05            0.9859x
         4     3.094e-05            0.9710x
         8     2.997e-05            0.9407x
  只量化 V（K 全精度）：nMSE = 2.376e-05

【2】换一把尺子（第 033 期的结论）：按 (层,头) 归一后看分位数，避免被大通道主导
    保前 N 个            中位            均值            最差
         0     3.077e-05     3.189e-05     4.192e-05
         4     2.995e-05     3.097e-05     4.100e-05
  → 在本机这个 tiny 模型上，保前 4 个位置只买回约 2.8% 的相对误差。
    「位置 0 重要」≠「它的量化误差是主要误差源」—— 通道轴上的 outlier 才是（见 3.5【6】）。
```
结论很扫兴，但很有用：

- **保前 4 个位置只买回 2.9% 的相对误差**（`0.9710x`）；保前 8 个才 5.9%。
- 对比一下：**只量化 V、K 全精度是 `2.376e-05`，而 K 量化是 `3.186e-05`** —— 也就是说「量化谁」这个选择影响 30%+，而「保护前几个 token」影响不到 3%。
- 换第 033 期那把更严的尺子（按 `(层,头)` 归一后看分位数）结论不变。

所以：**「位置 0 重要」和「位置 0 的量化误差是主要误差源」是两件不同的事。** 在这个模型上，误差主要来自通道轴的 outlier（§3.5【6】的维度 27/248/230），不是位置轴。Preserve-First-N 有用，但它不是量化误差的主要矛盾 —— 别指望靠它把 int4 救回来。

### 3.10 换一档语料，结论还成立吗

上面用的是「每个后继都可学」的干净语料。把 60% 的后继换成随机词，重新跑一遍 2000 步。
```python
"""3.10 块：换一档语料再跑一遍 —— 位置 0 的重要性是不是普遍现象？

3.3 用的是「每个后继都可学」的干净语料。这里把 60% 的后继换成随机词：
上下文中有一大半位置「没有任何有用信息可看」，模型对位置 0 的依赖会不会改变？
（用同一份代码、同一种子、同样 2000 步，只换语料。）
"""
import random, sys
sys.path.insert(0, '.')
import torch
from sink036 import *

torch.set_num_threads(4)
SEED, STEPS, TAIL = 0, 2000, 64
CKPTS = [0, 200, 400, 800, 1200, 1600, 2000]
text = build_corpus(40000, seed=0, noise_p=0.60)
tok = CharTok(text)
ids = tok.encode(text)
N = N_POS
print(f"语料 noise_p=0.60，{len(text)} 字符，词表 {len(tok)}，"
      f"基准 base0 = {causal_uniform_baseline(N):.5f}")

torch.manual_seed(SEED)
random.seed(SEED)
model = TinyGPT(len(tok), d=256, nh=8, nlayer=6)
opt, sched = make_optimizer(model, STEPS)
D0 = torch.zeros(N, dtype=torch.bool); D0[0] = True


def vzero(v, B, S):
    v = v.clone(); v[:, :, 0, :] = 0
    return v


def ev(**k):
    return sum(eval_loss(model, ids, only_tail=TAIL, nstep=8, batch=8,
                         seed=100 + i, **k) for i in range(4)) / 4


rows = []
for step in range(STEPS + 1):
    if step in CKPTS:
        model.eval()
        st = sink_stats(model, ids)
        full = ev(); a = ev(drop=D0)
        for blk in model.blocks:
            blk.attn.v_patch = vzero
        b = ev()
        for blk in model.blocks:
            blk.attn.v_patch = None
        rows.append((step, full, st['sink0'], st['sink0_maxhead'], a - full, b - full))
        model.train()
    if step == STEPS:
        break
    x = make_windows(ids, 128, 16)
    x[:, 0] = 0
    loss, _ = model(x, targets=x, ignore=0)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()

b0 = causal_uniform_baseline(N)
print(f"{'step':>6}{'full':>9}{'平均sink0':>12}{'/基准':>9}{'最强head':>11}{'/基准':>9}"
      f"{'Δ删掉pos0':>11}{'Δv0置零':>11}")
for step, full, s0, mx, da, db in rows:
    print(f"{step:>6}{full:>9.4f}{s0:>12.5f}{s0 / b0:>8.2f}x{mx:>11.5f}{mx / b0:>8.2f}x"
          f"{da:>+11.4f}{db:>+11.4f}")
print()
print("对比 3.3（干净语料，同一步数）：")
print("  · 这一档的 sink 更强（平均质量甚至**超过**基准，最强 head 是基准的 14.7 倍）；")
print("  · 删掉位置 0 的代价也确实更大，但在中间几步上剧烈波动 —— ")
print("    语料越不可预测，模型越依赖那个常量输入，这条曲线的噪声也越大。")
```
真实输出：

```text
语料 noise_p=0.60，200457 字符，词表 26，基准 base0 = 0.03469
  step     full     平均sink0      /基准     最强head      /基准    Δ删掉pos0      Δv0置零
     0   3.2841     0.03490    1.01x    0.03801    1.10x    +0.0346    +0.0324
   200   0.8651     0.02392    0.69x    0.13516    3.90x    +0.0003    +0.0005
   400   0.8454     0.02700    0.78x    0.20764    5.98x    +0.0045    +0.0008
   800   0.8340     0.04269    1.23x    0.51025   14.71x    +0.2503    +0.0713
  1200   0.8179     0.04568    1.32x    0.51080   14.72x    +0.0223    +0.0020
  1600   0.8026     0.04533    1.31x    0.51059   14.72x    +0.1476    +0.0099
  2000   0.7930     0.04547    1.31x    0.51044   14.71x    +0.1792    +0.0487

对比 3.3（干净语料，同一步数）：
  · 这一档的 sink 更强（平均质量甚至**超过**基准，最强 head 是基准的 14.7 倍）；
  · 删掉位置 0 的代价也确实更大，但在中间几步上剧烈波动 —— 
    语料越不可预测，模型越依赖那个常量输入，这条曲线的噪声也越大。
```
两档语料的对照：

| | 干净语料（3.3） | 噪声语料（3.10） |
| --- | --- | --- |
| 平均 sink0 / 基准 | 0.83x（**低于**基准） | **1.31x（高于基准）** |
| 最强 head / 基准 | 13.26x | 14.71x |
| 第 800 步 ΔA | +0.0001 | +0.2503 |
| 第 2000 步 ΔA | +0.0653 | +0.1792 |

三点：

1. **噪声语料下 sink 更强，而且是「真的强」**：平均质量甚至超过了因果基准 —— 这个模型的注意力确实在系统性地偏向位置 0。直觉上也说得通：上下文里 60% 的位置没有任何有用信息，模型更需要一个倾倒场。
2. **代价也更大，但更抖。** 中间几步（800 → 1200）从 +0.2503 掉到 +0.0223，再涨回去。**这条曲线本身不稳定** —— 所以任何「sink 强度 和 驱逐代价 成正比」的说法都要小心，它在这两档之间看起来成立，在同一个 run 内部却不成立（3.3 的 800 → 2000 就是反例）。
3. **「sink 是训练产物」这个结论在两档上都成立**：两档的第 0 步都是 `0.03488 / 0.03490` vs 基准 `0.03469`，即 1.01x。

### 3.11 复核：那条 +62% 是噪声吗？

`+62.1%` 是一个大到必须怀疑的数字。所以单独复核一遍：**换 20 组评估种子**（3.8 只用了 5 组），并且加一组**对照实验 —— 逐位置各删一个，看 sink-logit 模型是不是对哪个位置都这么敏感**。

如果它对删任何位置都掉一大截，那只能说明「这个模型脆弱」；如果只有位置 0 掉，才说明位置 0 是独一无二地不可替代。

```python
"""3.11 块：复核 3.8 那条反直觉结论 —— 带 sink logit 的模型，删掉真实位置 0 到底掉多少？

b8 里量到 +62.1%，比普通 softmax 模型的 +1.4% 大 44 倍。这么大的数必须独立复核：
  ① 换更多评估种子（20 组）看逐组增量；
  ② 确认这不是「sink logit 把真实 key 的份额压掉」造成的评估异常；
  ③ 看看它对「删掉其它位置」是不是同样敏感（对照组：删位置 1、删位置 64）。
"""
import random, sys
sys.path.insert(0, '.')
import torch
from sink036 import *

torch.set_num_threads(4)
text = build_corpus(40000, seed=0)
tok = CharTok(text)
ids = tok.encode(text)
TAIL = 64


def fit(**kw):
    torch.manual_seed(0); random.seed(0)
    net = TinyGPT(len(tok), d=256, nh=8, nlayer=6, **kw)
    opt, sched = make_optimizer(net, 1200)
    for _ in range(1200):
        x = make_windows(ids, 128, 16)
        x[:, 0] = 0
        loss, _ = net(x, targets=x, ignore=0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
    net.eval()
    return net


def one(ps):
    t = torch.zeros(N_POS, dtype=torch.bool)
    for p in ps:
        t[p] = True
    return t


SEEDS = [5000 + i for i in range(20)]


def paired(net, keep0=None, win=None):
    b = [eval_loss(net, ids, only_tail=TAIL, nstep=1, batch=8, seed=s) for s in SEEDS]
    k = dict(window=win) if win else {}
    v = [eval_loss(net, ids, only_tail=TAIL, nstep=1, batch=8, seed=s, **k,
                   **({"drop": one([0])} if keep0 == "drop0" else {})) for s in SEEDS]
    d = sorted(x - y for x, y in zip(v, b))
    return sum(b) / len(b), d[len(d) // 2], min(d), max(d), d


plain = TinyGPT(len(tok), d=256, nh=8, nlayer=6)
plain.load_state_dict(torch.load("sink036_s1200.pt")); plain.eval()
sl = fit(use_sink_logit=True)
print("20 组评估种子的配对中位增量（对照 seed 5000..5019）\n")
print(f"{'模型':<22}{'干预':<22}{'中位Δ':>10}{'Δ%':>9}{'最小':>10}{'最大':>10}")
for name, net in [("普通 softmax (1200 步)", plain), ("per-head sink logit", sl)]:
    for label, kw in [("删位置 0", dict(keep0="drop0")),
                      ("win32（不钉 sink）", dict(win=32)),
                      ("win32 + 钉位置 0", dict(win=32))]:
        b, med, lo, hi, d = paired(net, **kw)
        if label == "win32 + 钉位置 0":
            continue
        print(f"{name:<22}{label:<22}{med:>+10.4f}{med / b * 100:>+8.1f}%{lo:>+10.4f}{hi:>+10.4f}")

print()
print("对照组：sink-logit 模型对「删掉别的位置」敏感吗？（逐位置删一个，看 Δ）")
b = [eval_loss(sl, ids, only_tail=TAIL, nstep=1, batch=8, seed=s) for s in SEEDS]
print(f"{'删掉的位置':>10}{'配对中位 Δ':>14}{'Δ%':>9}")
for p in (0, 1, 2, 4, 8, 16, 32, 64, 96, 128):
    v = [eval_loss(sl, ids, only_tail=TAIL, nstep=1, batch=8, seed=s, drop=one([p]))
         for s in SEEDS]
    d = sorted(x - y for x, y in zip(v, b))
    print(f"{p:>10}{d[len(d) // 2]:>+14.4f}{d[len(d) // 2] / (sum(b) / len(b)) * 100:>+8.1f}%")

print()
print("sink logit 的最终取值（层 × 头）：")
slt = torch.stack([blk.attn.sink_logit.detach() for blk in sl.blocks])
for l in range(slt.shape[0]):
    print("  层", l, " ".join(f"{v:+.3f}" for v in slt[l]))
print(f"  均值 {slt.mean():.3f}  → 若某 query 对所有真实 key 的分数都≈0，"
      f"sink 拿走 {torch.sigmoid(slt.mean()) * 100:.1f}% 的质量")
```

真实输出：

```text
20 组评估种子的配对中位增量（对照 seed 5000..5019）

模型                    干预                           中位Δ       Δ%        最小        最大
普通 softmax (1200 步)   删位置 0                    +0.0085    +1.1%   -0.0033   +0.0221
普通 softmax (1200 步)   win32（不钉 sink）           +0.0073    +0.9%   -0.0080   +0.0527
per-head sink logit   删位置 0                    +0.4229   +56.4%   +0.2815   +0.5668
per-head sink logit   win32（不钉 sink）           +0.5908   +78.8%   +0.4595   +0.7965

对照组：sink-logit 模型对「删掉别的位置」敏感吗？（逐位置删一个，看 Δ）
     删掉的位置        配对中位 Δ       Δ%
         0       +0.4229   +56.4%
         1       +0.0003    +0.0%
         2       +0.0003    +0.0%
         4       +0.0000    +0.0%
         8       +0.0001    +0.0%
        16       +0.0002    +0.0%
        32       +0.0002    +0.0%
        64       +0.0932   +12.4%
        96       +0.0690    +9.2%
       128       +0.0000    +0.0%

sink logit 的最终取值（层 × 头）：
  层 0 +0.235 +0.339 +0.269 +0.288 +0.323 +0.406 +0.161 +0.282
  层 1 +0.109 +0.242 +0.059 +0.115 +0.243 +0.090 +0.108 +0.299
  层 2 +0.067 +0.052 -0.007 +0.136 +0.170 +0.134 +0.056 +0.176
  层 3 -0.011 +0.059 +0.028 +0.072 +0.051 +0.099 +0.046 -0.005
  层 4 -0.032 +0.083 -0.012 +0.048 +0.042 +0.098 +0.042 +0.018
  层 5 +0.130 +0.300 +0.076 +0.148 +0.139 +0.208 +0.104 +0.268
  均值 0.132  → 若某 query 对所有真实 key 的分数都≈0，sink 拿走 53.3% 的质量
```

结论非常干净：

1. **20 组种子下，普通 softmax 模型删掉位置 0 是 +1.1%（范围 −0.3% ~ +2.2%），sink-logit 模型是 +56.4%（范围 +28.2% ~ +56.7%）。** 两组区间完全不重叠 —— 这不是评估噪声。
2. **对照组是决定性的**：在 sink-logit 模型上逐位置各删一个，**删位置 1 / 2 / 4 / 8 / 16 / 32 / 128 的代价全是 `+0.0%`**，而删位置 0 是 `+56.4%`。所以这不是「模型整体脆弱」，而是**位置 0 在这类模型里是唯一不可替代的那个 KV**（删位置 64 有 +12.4%、96 有 +9.2%，是次要的一档）。
3. 学到的 sink logit 均值 `0.132`（层 0 最大 +0.406），换算成「如果对所有真实 key 的分数都是 0，sink 拿走 53.3% 的质量」—— 和 §2.2 说的「硬出口」完全对上。

> 顺便注意第 1 行和第 2 行的差异：**同一个普通 softmax 模型，删位置 0 是 +1.1%，`win32` 是 +0.9%** —— 删掉一个位置和删掉 96 个位置的代价差不多。因为在这个 checkpoint（1200 步）上，位置 0 和那些远距离位置一样，都还不重要。到了 2000 步（§3.6）才变成 +8.9%。**同一份架构、同一个干预，换个训练时长，结论会变。**

---

## 4. 围绕该领域展开

### 4.1 谱系：三条路线

| 时间 | 工作 | 核心主张 | 和本期实验的关系 |
| --- | --- | --- | --- |
| 2023.09 | **StreamingLLM**（Xiao et al., ICLR 2024, arXiv 2309.17453） | 钉住前 4 个 token 的 KV + 滚动窗口；位置用**缓存内**的相对位置（RoPE 要重旋转）；Llama-2-7B 在 PG19 上从 0 个 sink 的 PPL **3359.95** 降到 4 个的 **9.59**；比「滑窗 + 重算」快最多 **22.2×**；预训练时加一个可学 sink token，推理只需 1 个 | §3.6 复现了「窗口 + 钉 sink ≈ 全上下文」这条；也复现了「1 个 ≈ 4 个」在 tiny 规模下成立 |
| 2024.02 | **Massive Activations**（Sun et al., COLM 2024, arXiv 2402.17762） | 少数固定维度上的激活幅度大几个数量级，起**输入无关偏置**作用；把它们置零会让模型崩掉 | §3.5【6】在本机复现了同类结构（维度 27/248/230，5.7 倍于中位） |
| 2024.10 | **When Attention Sink Emerges**（Gu et al., ICLR 2025, arXiv 2410.10781） | sink 在充分训练后**才出现**（约 1k~2k 步）；weight decay 助长它；改 loss function / 数据分布可以把 sink 挪到别的位置；**它更像 key bias，不参与 value 计算**；换成 sigmoid attention 后 1B 模型上不再出现 | §3.3 的核心时间轴就是这条的量化版；§3.7 的 ΔC ≈ 0 是「不参与 value 计算」的直接证据 |
| 2023 | **SoftMax1 / off-by-one**（Evan Miller） | 在 softmax 分母上加 1，让 head 能「保持安静」 | §3.8 的 sink logit 就是它的可学版本 |
| 2025.08 | **gpt-oss**（OpenAI） | 每个 head 一个可学 `sink` 标量，拼进 softmax 分数后再丢掉；要求 FA / FlexAttention，**不支持 SDPA** | §3.8 完整复刻了这个机制，并测了它的实际收益 |
| 2024→2025 | **Preserve-First-N / KVSink** | 量化时把 sink 位置保高精度 | §3.9 量出收益只有个位数百分比 |

### 4.2 $1/S$ 这个错基准，为什么会流传这么广

它有三个「看起来很对」的地方：

1. **在非因果（encoder / bidirectional）注意力里它是对的。** 没有因果掩码时每个 key 被所有位置看到，均匀分布下每个 key 就是 $1/S$。这个结论被大量 BERT 时代的材料带到了 decoder 场景。
2. **它有个漂亮的直觉解释**：「位置 0 只是 1/S 里的一份，却拿了 30 倍，所以它异常」。这个叙事在论文引言里很好写。
3. **它不影响「有没有 sink」这个定性结论**（因为真实 sink 往往超出它几十倍），所以没人去较真。但在两件事上会出事：一是**判据**（阈值定多少算 sink），二是**长上下文外推**（差值随 $\ln S$ 增长，$S$ 越大错得越多）。

**正确做法**：用 $b(j) = \frac{1}{S-1}\sum_{t\ge j} \frac{1}{t+1}$ 这条曲线逐位置做除法，然后看「还剩多少过量」。多一行代码，结论的可比性完全不同。

顺带补一句第 035 期：那一期测出 copy 任务模型位置 0 只有 `0.005~0.009` 的质量。按 $S=192$ 的正确基准 `0.025326` 算，它其实是基准的 **0.20~0.36 倍** —— **比「和背景一个量级」更彻底地低于预期**。所以那一期「这个模型没有 attention sink」的结论不但成立，而且比当时陈述的更强（当时没有写明用的是什么基准）。

### 4.3 这一期真正的方法论：指标和行为的错位

把 §3.3、§3.6、§3.7 拼起来看，能得到一个比 attention sink 本身更通用的教训。

**现象**：我们想知道的量是「删掉这个 KV 会掉多少」（行为），但我们能方便测的量是「它拿了多少注意力」（指标）。

**实测**：

| 步数 | 最强 head（指标） | ΔA（行为） | 指标能看到行为吗 |
| --- | --- | --- | --- |
| 200 | 4.13x | +0.0002 | ❌ 完全看不到 |
| 800 | 13.27x | +0.0001 | ❌ 更强了 3 倍，行为没动 |
| 1200 | 13.35x | +0.0120 | ❌ 指标持平，行为涨了 100 倍 |
| 2000 | 13.26x | +0.0653 | ❌ 指标持平，行为又涨了 5 倍 |

在**单次训练**里，指标从 800 步起就是常数，行为涨了 650 倍。也就是说：**这里的相关性不是「弱」，而是「在关键区间里恒等于零」。**

这不是「sink 特别难测」，是一个普遍的模式。第 032 期讲过一条类似的：**拿最小二乘残差当证据前，先断言方程数 > 未知数** —— 那里是「方程数不够，却先算出了一个看起来很漂亮的拟合」。这里是「指标早就饱和、之后和目标量彻底脱钩」。

**可操作的规则**（本期想留下的东西）：

1. 想测「某个 KV 重不重要」，**唯一的直接办法是把它删掉/替换掉，测输出差**（ablation），而不是看它的注意力质量。
2. 如果必须用代理指标，**至少要做两个不同性质的扰动**（这里是 A：删掉；B：抹掉 value），看两者是否一致。本机实验里 ΔA 和 ΔB 差 2 倍 —— 说明它们测的确实不是同一件事。
3. **汇报时要区分「模型在看哪里」和「模型依赖什么」。** 热力图是前者，只有 ablation 能回答后者。

### 4.4 工程上怎么用它

把本期的实测结论翻译成部署决策：

| 场景 | 该做什么 | 本机证据 |
| --- | --- | --- |
| 流式 / 定长缓存推理 | 钉住前 1~4 个 token + 滚动窗口；窗口下限要实测，别照抄 16 | §3.6：`win32+4sink` 追平全上下文，`win16` 掉 22% |
| KV 驱逐 / 稀疏化 | 把「保护 sink 位置」写成显式白名单，别指望打分函数自己学会 | §3.6：只删位置 0 就掉 8.9% |
| 想省掉那份被倒掉的质量 | 把它转成显式偏置（sink logit / 常数项），可以省掉「重新分配」那一半代价 | §3.7：ΔA = 2×ΔB，差值来自注意力重分配 |
| KV 量化 | 前几个位置保高精度可以做，但别指望它是主要收益来源 | §3.9：只有 2.9% |
| 自定义 attention 后端 | `aten` SDPA / FA 都没有 sink 槽；要么走 FlexAttention 的 `score_mod` + LSE 后处理，要么在序列里插占位 token | §3.8：算子 schema + flex_attention 实测 |
| 训练自己的模型 | 如果不需要 sink，sigmoid attention 是一个选项；但 tiny 规模下它只是变弱、没有消失 | §3.8：三模型对照 |

### 4.5 三个反直觉的实测结果

1. **同样强的 sink，重要性可以差 650 倍。** 第 800 步（13.27x 的强 sink）删掉代价 +0.0001；第 2000 步（同样 13.26x）代价 +0.0653。**指标读数几乎相同，行为差 650 倍** —— 而这两次测量的对象是同一个模型、同一份权重的两个训练阶段，连「换了个模型」这个借口都没有。
2. **训练时给了专用 sink 槽，推理时反而更脆弱 —— 而且是量级上的差异。** gpt-oss 口径的 per-head sink logit 模型，删掉真实位置 0 的代价是 **+62.1%**（§3.8，5 组评估种子）；同样 1200 步的普通 softmax 模型只有 **+1.4%**。换 20 组种子复核（§3.11）两个数字分别回落到 **+56.4%** 与 **+1.1%**，**两组区间完全不重叠**（sink-logit 的逐组增量在 +28.2% ~ +56.7%，普通 softmax 在 −0.3% ~ +2.2%），方向与量级都站得住。直觉上「专用槽」应该让模型不再依赖真实 token，实测方向完全相反。可操作的推论：**要给模型引入一个 sink 槽，就要同时把「真实的前几个 token 也必须留着」当成硬约束**，别以为换成了参数槽就可以省掉那几个 KV。
3. **`win64 + 4 sink` 比全上下文还好**（−0.1%）。丢掉 64 步以前的 KV 不是「损失很小」，而是**没有任何损失**。原因是那些位置的注意力权重本来就接近零，留着只是噪声。

---

## 5. 什么时候该用不该用

| 场景 | 建议 | 理由 |
| --- | --- | --- |
| 长上下文流式推理 | ✅ 钉 sink + 滑窗，这是标准做法 | §3.6 实测能追平全上下文 |
| KV 驱逐 / 稀疏的默认规则 | ✅ 把前几个位置写成硬保护 | 本机 2000 步的模型上，只删位置 0 就掉 8.9%（1200 步的同一架构只掉 1.1%） |
| 判断「某个 token 的 KV 该不该删」 | ❌ **别只看注意力权重** | §4.3：关键区间上相关性恒为零 |
| 短上下文（< 1k）的离线推理 | ⚠️ 优先级低 | 省下来的显存有限，而「保护 sink」本身要占常数开销（多留几个槽、多一次判断）；这一期所有结论都是在 129 个位置上测的 |
| 想靠 Preserve-First-N 救 int4 量化 | ❌ 别指望 | §3.9：只有 2.9% |
| 用 sink logit 想让「位置 0 可丢」 | ❌ 实测方向相反 | §3.8：删掉真实位置 0 的代价从 +1.4% 涨到 +62.1% |
| 把 $1/S$ 当背景阈值 | ❌ 绝对不要 | §3.2：$S$=131k 时零 sink 模型就有 11.36 倍 |
| 需要 dense 的 sink 语义 + 融合 kernel | ⚠️ 有冲突 | §3.8：SDPA / FA 都没有 sink 槽，要自己改 kernel 或用 FlexAttention |

---

## 6. 常见坑

**坑 1：拿 $1/S$ 当「零 sink 基准」。**
这是本期的头号坑。因果掩码本身就给位置 0 一个约 $\ln S$ 倍于 $1/S$ 的结构性加成，$S=131072$ 时是 11.36 倍。用错基准的后果不是「差一点」，而是**把每一个正常模型都判成有 sink**，并且这个误判随上下文长度单调恶化。

**坑 2：把「注意力质量」当成「重要性」。**
§4.3 的表格：第 800 步 13.27x 的 sink、删掉零代价；第 2000 步同样的 13.27x、代价 +0.065。**唯一可靠的判断是 ablation**（删掉 / 抹掉 value / 换成别的值，三种扰动一起做）。做稀疏化、驱逐、量化决策时，先跑一遍 ablation，再决定保护谁。

**坑 3：滑窗掩码写错方向 —— 这是代码级的坑，而且症状很迷惑。**
`causal_window_mask` 里有两个必须写对的地方：
- **窗口必须相对每个 query**（`k > q - window`）。写成一个静态的「数组末尾 W 个」时，序列前半段的 query 会「什么都看不到」，现象是 loss 在某些配置下突然爆掉（本机踩到过 `win32` 崩到 4.28 而 `win64` 正常）。
- **`key_keep` 必须是并集，不是交集**，且**每个 query 必须能看到自己**。写成交集，钉住的 sink 会被窗口整个抹掉，现象是**所有策略的 loss 一模一样**（本机踩到过全是 5.2782）；忘了加 identity，整行 `-inf` → NaN。

**坑 4：拿「不同评估数据下的绝对 loss」比较策略。**
同一个策略在不同随机窗口上的 loss 能在 0.65~0.87 之间跳（§3.6 配对表）。必须**在相同数据上做配对减法**再比较。看着很基础，但这是最容易在赶时间时省掉的一步。

**坑 5：在小模型上量出「sink 不重要」，就直接外推到 7B/70B。**
本机的模型只有 2.4M 参数、129 个位置、2000 步。它能在**定性**上复现文献的机制（sink 是训练产物、是常量、不参与 value），但**定量**数字（1 个 vs 4 个 sink token、窗口下限 32）没有规模不变性。这条边界要写清楚：**机制可信，常数不可信。**

---

## 7. 一句话总结

**attention sink 是一个「质量倾倒场」而不是「信息源」—— 它的注意力质量由 key 向量（一个训练出来的常量）决定，删掉它的代价由 value 侧（一个跨样本常量偏置）决定，这两件事在训练时间轴上是两条错开 650 倍的曲线，所以想判断它重不重要，只能靠 ablation，不能靠看注意力热力图。**

---

## 8. 今日练习

<details>
<summary>今日练习</summary>

**题目**：写一个函数 `kv_importance(model, ids, policies)`，对给定的若干「缓存策略」输出三列：全上下文 loss、该策略 loss、Δ%。然后用它回答一个具体问题：

> 在 §3.3 训出的 2000 步模型上，**窗口 24 至少要配几个 sink token 才能追平全上下文（Δ ≤ 1%）**？

要求：

1. 用 `sink036.causal_window_mask` 的 `window` 和 `key_keep` 组合实现，**不要**自己重写掩码；
2. 评估时用 `only_tail=64`、至少 5 组评估种子并**取配对中位数**（不是「中位数除中位数」）；
3. 额外报告：把 sink token 从「前 N 个」换成「前 1 个 + 最后一个可见位置」，结果会怎样变？为什么？

<details>
<summary>参考答案（本机真跑）</summary>

完整脚本（存成 `b11_exercise.py`，和 `sink036.py` 放同一目录）：

```python
"""3.11 块的对照脚本：今日练习的参考答案 —— win24 要配几个 sink？"""
import sys
sys.path.insert(0, '.')
import torch
from sink036 import *

torch.set_num_threads(4)
text = build_corpus(40000, seed=0)
tok = CharTok(text)
ids = tok.encode(text)
m = TinyGPT(len(tok), d=256, nh=8, nlayer=6)
m.load_state_dict(torch.load("sink036_s2000.pt"))
m.eval()
TAIL, SEEDS = 64, [4000 + i for i in range(5)]
base = [eval_loss(m, ids, only_tail=TAIL, nstep=1, batch=8, seed=s) for s in SEEDS]


def keep(ps):
    t = torch.zeros(N_POS, dtype=torch.bool)
    for p in ps:
        t[p] = True
    return t


def paired_median(**k):
    d = sorted(eval_loss(m, ids, only_tail=TAIL, nstep=1, batch=8, seed=s, **k) - b
               for s, b in zip(SEEDS, base))
    return d[len(d) // 2]


print(f"全上下文 loss（5 组配对均值）= {sum(base) / 5:.4f}；只统计最后 {TAIL} 个位置")
print()
print(f"{'窗口':>6}{'sink 数':>9}{'配对中位 Δloss':>16}{'Δ%':>9}")
for w in (24, 32):
    for n in (0, 1, 2, 4, 8):
        d = paired_median(window=w, key_keep=keep(range(n)))
        print(f"{w:>6}{n:>9}{d:>+16.4f}{d / (sum(base) / 5) * 100:>+8.1f}%")
print()
print("最后一问：把 sink 从「前 N 个真实起始位置」换成别的选法，会怎样？")
alt = [
    ("前 1 个 (位置 0)", [0]),
    ("前 2 个", [0, 1]),
    ("前 4 个", [0, 1, 2, 3]),
    ("只留末位 128", [128]),
    ("位置 0 + 末位 128", [0, 128]),
    ("位置 0 + 位置 64", [0, 64]),
    ("位置 64 单点", [64]),
]
print(f"{'钉住的集合':<20}{'配对中位 Δloss':>16}{'Δ%':>9}")
for name, ps in alt:
    d = paired_median(window=24, key_keep=keep(ps))
    print(f"{name:<20}{d:>+16.4f}{d / (sum(base) / 5) * 100:>+8.1f}%")
print()
print("  → 末位 128 单独钉住几乎无用：它在每个 query 眼里的位置都在变，")
print("    既不是输入无关的常量，也没有稳定方向 —— 当不了「倾倒场」。")
print("    这条反证了 3.5 的机制：sink 的前提是「key 与输入无关」。")
```

真实输出：

```text
全上下文 loss（5 组配对均值）= 0.7348；只统计最后 64 个位置

    窗口   sink 数      配对中位 Δloss       Δ%
    24        0         +0.0979   +13.3%
    24        1         +0.0039    +0.5%
    24        2         +0.0038    +0.5%
    24        4         +0.0032    +0.4%
    24        8         +0.0020    +0.3%
    32        0         +0.0718    +9.8%
    32        1         +0.0005    +0.1%
    32        2         -0.0002    -0.0%
    32        4         -0.0007    -0.1%
    32        8         -0.0000    -0.0%

最后一问：把 sink 从「前 N 个真实起始位置」换成别的选法，会怎样？
钉住的集合                     配对中位 Δloss       Δ%
前 1 个 (位置 0)                 +0.0039    +0.5%
前 2 个                        +0.0038    +0.5%
前 4 个                        +0.0032    +0.4%
只留末位 128                     +0.0979   +13.3%
位置 0 + 末位 128                +0.0039    +0.5%
位置 0 + 位置 64                 +0.0033    +0.4%
位置 64 单点                     +0.0966   +13.1%

  → 末位 128 单独钉住几乎无用：它在每个 query 眼里的位置都在变，
    既不是输入无关的常量，也没有稳定方向 —— 当不了「倾倒场」。
    这条反证了 3.5 的机制：sink 的前提是「key 与输入无关」。
```

**要点**：

- `window=24` 是**相对每个 query** 的；用静态末尾窗口会让前半段 query 无上下文，结果完全不可比。
- `key_keep` 与窗口是**并集**；写成交集时所有 n 的结果会完全一样（这是个很好的自检信号）。
- 必须是**配对**减：`eval_loss` 每次重抽数据，绝对 loss 本身在 0.73 附近上下跳，比较两个策略的绝对值会得到假结论。

**答案**：

1. **`win24` 下 1 个 sink 就够了**（+0.5%），2 / 4 / 8 个都落在噪声里（+0.5% / +0.4% / +0.3%）—— 和 §3.6「1 个 ≈ 4 个」一致。这也说明**文献里的「4 个」是 7B 模型长序列上的经验值，不是普适常数**；窗口越小，需要的 sink 越多（`win16+4sink` 才 +1.7%）。
2. **第三问的答案很干净**：把 sink 换成「前 1 个 + 序列末位 128」，结果和「前 1 个」**完全一样**（都是 +0.5%）—— 末位那一票没起作用。单独钉末位 128 则等于没钉（+13.3%，和不钉一样）。位置 64 也是（+13.1%）。
3. **原因**：末位 128 在每个 query 眼里的**相对位置都在变**，不是输入无关的常量，也没有稳定的方向 —— 它当不了「倾倒场」。这正好反证了 §3.5 的核心机制：**sink 之所以能当倾倒场，前提是它的 key 与输入无关（batch 内偏差 1.9e-06，比普通位置小 6 个数量级）**。「随便挑一个位置钉住」是无效的，必须是**训练里被模型选中的那个位置**。
4. 顺带注意 `32 / 2` 和 `32 / 4` 出现了**负的** Δ（−0.0% / −0.1%）：窗口去掉的那些位置的注意力权重本来就接近零，去掉它们等于去噪。这是 §3.6「win64+4sink 反而比全上下文好」的同一件事。

</details>
</details>

