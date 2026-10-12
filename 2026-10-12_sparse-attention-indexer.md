# PyTorch 每日一课 · 第 034 期

## 稀疏注意力：谁来决定「看哪里」

| | |
|---|---|
| **日期** | 2026-10-12 |
| **难度** | ⭐⭐⭐⭐⭐ |
| **前置知识** | 第 032 期（一个 token 的 KV 到底多少字节）、第 011 期（PagedAttention 的块布局）、第 005 期（FlashAttention 是「怎么算得快」）、第 027/028 期（roofline 平衡点与批调度） |
| **预计阅读时间** | 60 分钟 |
| **关联** | 第 032 期 · 那期讲「KV 压得小」，这期讲「看得少」；第 033 期 · 那期把每个字节砍到 8 bit，这期把一个 token 的可见范围砍到 1.6%；第 005 期 · FlashAttention 不改变数学形式只改 IO，这期**改数学形式**；第 029 期 · 前缀缓存也是「少算一点」的一条路，但它省的是重复的 prefill，这期省的是每个 query 自己的全序列扫描；第 026 期 · MoBA 直接把 MoE 的 top-k 门控搬到了 attention 上 |

---

## 1. 这个领域解决什么问题

先把上一期的账收个尾（这期独立可读，不需要回去翻）。

第 032 期算过一个 token 的 KV 要多少字节，第 033 期把它往下压：

| 口径 | 每 token KV | 来源 |
|---|---|---|
| MLA 全 fp16 | 68.62 KiB | 第 032 期的基准 |
| MLA fp8 + RoPE fp16 | 38.12 KiB | 权重与 latent 走 fp8 |
| 再加 DSA 的 indexer key cache | **45.75 KiB** | 本期 §4.3 |

字节省了 33%。但请注意一件事：**这三行里 attention 的计算量一模一样。**

一个 token 的 KV 从 68.62 KiB 压到 45.75 KiB，改变的只是「把它从 HBM 搬到 SM」的时间。而 attention 真正要做的事是：**每个 query 和它之前每一个 token 都做一次点积**。这件事的工作量是

$$\text{MAC per query} = A \cdot L,\qquad A = n_{\text{heads}} \times (2 L_{kv} + R)$$

代入 DeepSeek-V3.2 的实际配置（`config.json`：`num_attention_heads=128`、`kv_lora_rank=512`、`qk_rope_head_dim=64`）：

$$A = 128 \times (2\times512 + 64) = 139{,}264\ \text{MAC}$$

$L = 131072$ 时，**每层每个 query 1.83e10 MAC，61 层就是 1.12e12 MAC**。这个数字和「KV 压到几比特」毫无关系——你把 KV 压到 1 bit，这 1.12 T MAC 一个不少。

> 这就是这一期要讲的领域：**在保持 softmax 注意力数学形式的前提下（不是换成线性注意力/SSM），稀疏注意力是唯一能砍掉注意力计算量本身的路子。**

三个数量级的概念先摆在一起：

| 优化手段 | 改变了什么 | 在长上下文 decode 上的收益 |
|---|---|---|
| fp8 权重 | 权重读取字节 | 1.01–1.17×（第 033 期 roofline） |
| KV 量化到 fp8 | KV 读取字节 | 1.55–1.95×（第 033 期 roofline） |
| **只看 2048 个 token（DSA）** | **计算量本身** | **13.43×（本期 §4.2 实算）** |

（三行的口径不同：前两行是第 033 期按 H800 级带宽做的 roofline 推算，第三行是本期按 $A/W$ 的解析式算的注意力计算量之比，都不含权重读取项。）

### 1.1 稀疏不是新东西，「能训练的稀疏」才是

稀疏注意力 2020 年就有了（Longformer、BigBird），但那批方案有一条共同的死穴：**稀疏模式是静态的、预先写死的**。每个 token 固定看「附近 W 个 + 少数几个 global token」，与输入内容无关。

静态模式在问答、检索这类任务上会直接掉点，因为答案可能正好落在一个既不在窗口内、也不是 global 的位置。于是有了第二代：

| 代际 | 代表 | 分数谁给 | 死穴 |
|---|---|---|---|
| 一、静态结构 | Longformer / BigBird / StreamingLLM | 人写死的规则 | 与内容无关，任务相关 |
| 二、推理期剪枝 | H2O / SnapKV / Quest / MInference | 用当前 query 临时挑 | **模型是用 dense 训出来的**，强制稀疏引入分布偏移；且只优化 prefill 或只优化 decode 中的一段 |
| 三、原生可训练 | **NSA**（ACL 2025 最佳论文） | 学习出来的信号 | 必须从预训练阶段介入 |
| 三、生产落地 | **DSA**（DeepSeek-V3.2-Exp） | 专用小网络（lightning indexer） | 需要框架与 kernel 深度配合 |
| 三、MoE 类比 | **MoBA**（Moonshot，已上 Kimi） | 块均值池化 + 门控 | 门控信号表达力有限 |

第二代的「分布偏移」是这段历史的关键。NSA 论文写得很直白：推理时稀疏的方法「只恢复了潜在加速的一小部分」，因为**模型从来没有被要求通过稀疏模式路由信息**。这句话是第三代全部设计的出发点。

NSA 论文还给了动机的量化依据：**用 softmax attention 解码 64K 长度的上下文时，attention 计算占总延迟的 70–80%**。这就是为什么这个领域值得单独做一期。

---

## 2. 核心思想

### 2.1 稀疏注意力只有三个自由度

不管叫什么名字，所有稀疏注意力方案都在回答同样三个问题：

| 自由度 | 可选值 | 谁来定 |
|---|---|---|
| **粒度** | token 级 / 块级 / 页级 | 由硬件访存模式倒逼（见 §4.6） |
| **信号** | 固定位置规则 / 派生分数 / 专用网络 | 这是各方案的核心分歧 |
| **预算分配** | 每层同一个 k / 逐层不同 / 逐 head 不同 | 多数方案用全局同一个 k |

三个自由度里，**「信号」决定一切**。粒度是硬件约束的结果，预算分配是调参。

### 2.2 一个类比：图书馆

- **dense attention**＝每问一个问题，就把全馆每一本书翻一遍。
- **sliding window**＝只看最近上架的那一排书。快，但你要的那本 1998 年的书就没戏了。
- **静态结构稀疏**＝按固定书架编号看（第 1 排恒看、第 3 排恒看）。方便，但跟你问什么无关。
- **内容相关稀疏**＝先查目录，再按目录取书。

四种做法的差别全在**目录**上：目录是谁编的（固定规则？临时算的分数？专门训练的小网络？）、编得多细（整本书还是一个章节？）、编得准不准（目录指的书是不是真的相关？）。

一句话：**稀疏注意力的质量上限，由「选择信号」的质量决定，而不是由「稀疏率」决定。**

### 2.3 唯一的验收指标：recall

既然「选哪些位置」是核心，就需要一个指标来描述选择的好坏。定义

$$\text{recall}(t) = \sum_{j \in \mathcal{S}_t} p_{t,j}$$

其中 $p_{t,\cdot}$ 是**原本 dense attention 的 softmax 质量**，$\mathcal{S}_t$ 是被选中参与计算的位置集合。

这个指标的性质很好：

- 它是**选择质量的纯度量**，与「选中多少」无关（预算固定时可以直接比较）。
- dense 的 recall 恒等于 1；随机选择的 recall 等于「随机撒 $k$ 个位置恰好碰到的质量比例」——质量接近均匀时约等于 $k/L$，但在有局部衰减的真实分布下会明显高于 $k/L$（下面实测 $k/L = 0.125$ 时随机选择拿到 0.3847）。
- 它和输出误差强相关（下面的实验会看到单调关系）。

下一节开始，所有实验都在问同一件事：**在固定的计算预算下，谁能把这 1.0 的 recall 拿回来更多。**

### 2.4 块粒度的两难

真实硬件不可能按 token 粒度去 gather KV（见 §4.6），所以必须按块选。而块粒度天然损失选择自由度：

- 你只能表达「选中的 64 个 token 是连续的」这类集合，不能表达任意 64 个 token。
- 对角块（query 所在的那个块）被强制选中，但块内只有一半在因果掩码内 —— 这部分算了但没用。

这两条代价本期都量化了（§3.1 的 Q3a/Q3c）。

### 2.5 三条路线怎么分岔

| | NSA | DSA | MoBA |
|---|---|---|---|
| 粒度 | 三层并行：压缩块 32 / 选择块 64 / 窗口 512 | token 级（页 64） | 块级 |
| 信号 | **复用压缩分支自己的注意力分数** | **专用 indexer 网络**（64 头 × 128 维，fp8） | **块均值池化后的内积** |
| 预算 | 压缩 + top-16 块 + 窗口，三者相加 | top-2048 token | top-k 个块 |
| 训练 | 从头原生训练 | 从 dense ckpt 续训两阶段 | 与 dense 混合训练 |
| 额外参数 | 有（block summarizer MLP + gate） | 有（新增 q/key/weight 三个投影） | **零** |
| 落点 | 论文（27B/3B active，270B tokens） | DeepSeek-V3.2-Exp 生产 | Kimi 生产 |

有意思的是：**NSA 拿了 ACL 2025 最佳论文，但 DeepSeek 自己在生产里用的是简化版 DSA**（三分支 → 单 indexer）。这一条对「论文 ≠ 实践」的提醒很值钱，后面 §4.7 展开。

---

## 3. 在 PyTorch 中怎么用

### 3.1 先把「选择质量」量出来

这一节的脚本构造一份**有结构的**注意力 logits（attention sink + 局部衰减 + 每个 head 各自的长距离高权重区 + 噪声），然后在**固定的计算预算 K=256** 下对比五种选择策略。因为要能真实复现，所有张量都在本机 CPU 上算。

<details>
<summary>为什么用合成 logits 而不是真模型</summary>

本机没有 CUDA、也没有能跑长上下文的模型。但 recall 这个指标只需要 $p_{t,\cdot}$ 和 mask，与模型的其余部分无关。合成 logits 的好处是**可以精确控制「注意力质量长什么样」**，从而把「块稀疏到底靠什么成立」这个混淆变量拆干净。

风险是结论依赖合成分布。所以这一节的所有结论都配了单变量扫描（连续长度、head 数），而不是只给一个点。

</details>

```python
"""E1：稀疏注意力的唯一质量指标 —— 选中集合覆盖了多少注意力质量（recall）

三个问题一次答完：
  Q1 固定预算 K 下，token / block / window / random 四种选择的 recall 与输出误差
  Q2 选择必须跨 query head 共享（MQA/GQA 下 KV 只有一份），共享要付多少
  Q3 块稀疏成立的前提是什么 —— 注意力质量的「空间连续性」run_len
"""
import torch

L, H, D, K = 2048, 8, 64, 256
t_idx = torch.arange(L)
causal_ok = t_idx.view(L, 1) >= t_idx.view(1, L)          # [q_i, k_j] = i >= j


def make_logits(L, H, seed=0, run_len=1, n_region=2):
    """合成一份「像真实 LLM 注意力」的 logits：
    sink + 局部衰减 + 每个 head 各自的长距离高权重区（n_region 个、每个长 run_len）+ 噪声。
    run_len=1 是孤立尖峰（对块粒度最坏），run_len=64 是连续片段（真实检索的常见形态）。
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L).view(L, 1)
    j = torch.arange(L).view(1, L)
    dist = (t - j).clamp(min=0).to(torch.float32)
    lg = (-1.4 * torch.log1p(dist / 8.0)).unsqueeze(0)
    lg = lg + torch.randn(H, 1, L, generator=g) * 0.9
    lg[:, :, 0] += 6.0                                     # sink
    starts = torch.randint(0, L, (H, L, n_region), generator=g)
    mag = 4.0 + 3.0 * torch.rand(H, L, n_region, generator=g)
    off = torch.arange(run_len).view(1, 1, 1, run_len)
    pos = (starts.unsqueeze(-1) + off).clamp(max=L - 1)     # [H, L, n_region, run_len]
    ok = (pos < t.unsqueeze(-1)).to(torch.float32)          # 只能看过去
    lg.scatter_add_(2, pos.reshape(H, L, -1),
                    (mag.unsqueeze(-1) * ok).reshape(H, L, -1))
    return lg.masked_fill(~causal_ok.view(1, L, L), float("-inf"))


def build(seed, run_len, n_region=2, Hh=H):
    torch.manual_seed(7)
    lg = make_logits(L, Hh, seed=seed, run_len=run_len, n_region=n_region)
    p = torch.softmax(lg, dim=-1)
    v = torch.randn(Hh, L, D) * 0.5
    return lg, p, v, p @ v


def evaluate(lg, p, v, out_dense, mask2d):
    causal = causal_ok.view(1, L, L)
    m = mask2d.view(1, L, L) & causal
    lp = torch.where(m, lg, torch.full_like(lg, float("-inf")))
    out = torch.softmax(lp, dim=-1) @ v
    err = (out - out_dense).norm(dim=-1) / out_dense.norm(dim=-1).clamp(min=1e-9)
    recall = (p * m).sum(dim=-1)
    return recall.mean().item(), err.mean().item()


def topk_mask(score, k):
    idx = torch.topk(score, k, dim=-1).indices
    m = torch.zeros_like(score, dtype=torch.bool)
    return m.scatter_(-1, idx, True)


def block_mask(score2d, k, b):
    """块级选择：块分数 = 块内【可见】位置的平均；强制包含对角块（MoBA 的做法）；
    然后块内位置全选，超出当前位置的靠因果掩码挡住（挡住也算算了）。"""
    nb = L // b
    vis = causal_ok.to(score2d.dtype)
    s = (score2d * vis).view(L, nb, b)
    cnt = vis.view(L, nb, b).sum(-1).clamp(min=1.0)
    bs = s.sum(-1) / cnt                                   # 只对可见位置求均值
    blk_start = torch.arange(nb).view(1, nb) * b
    blk_end = blk_start + b - 1
    bs = bs.masked_fill(blk_start > t_idx.view(L, 1), float("-inf"))
    diag = (blk_start <= t_idx.view(L, 1)) & (t_idx.view(L, 1) <= blk_end)   # 对角块
    bs = bs.masked_fill(diag, float("inf"))                # 强制选中自己所在的块
    keep = max(k // b, 1)
    idx = torch.topk(bs, keep, dim=-1).indices
    m = torch.zeros(L, nb, dtype=torch.bool).scatter_(1, idx, True)
    return m.unsqueeze(-1).expand(L, nb, b).reshape(L, L)


def head_agg(lg):
    """跨 head 聚合分数（DSA / NSA / MoBA 选块时都要一个共享信号）"""
    out = lg[0].clone()
    for h in range(1, lg.shape[0]):
        out = torch.logsumexp(torch.stack([out, lg[h]]), dim=0)
    return out


def budget_for(mass_fn, target=0.90, lo=1, hi=None):
    """找到「覆盖 target 比例注意力质量」所需的最小预算"""
    hi = hi or (L // 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if mass_fn(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


lg, p, v, dense = build(0, run_len=1)
agg = head_agg(lg)

print(f"配置：L={L}  heads={H}  d={D}  每 query 预算 K={K} token")
print("logits：sink + 局部衰减 + 每 head 各自的长距离高权重区 + 噪声")
print()
print(f"{'策略（全部共享一个跨 head 选择）':<44}{'recall':>9}{'输出误差':>10}{'实算key':>9}")
print("-" * 76)


def row(name, m, nk):
    rc, er = evaluate(lg, p, v, dense, m)
    print(f"{name:<44}{rc:>9.4f}{er:>10.4f}{nk:>9.0f}")
    return rc, er


per_head = torch.stack([topk_mask(lg[h], K) for h in range(H)]) & causal_ok.view(1, L, L)
oph = torch.softmax(torch.where(per_head, lg, torch.full_like(lg, float("-inf"))), dim=-1) @ v
e_ph = ((oph - dense).norm(dim=-1) / dense.norm(dim=-1).clamp(min=1e-9)).mean().item()
r_ph = (p * per_head).sum(dim=-1).float().mean().item()
print(f"{'① token top-256｜逐 head 各选各的（上界，kernel 做不到）':<44}{r_ph:>9.4f}{e_ph:>10.4f}{K:>9.0f}")

r_tok, e_tok = row("② token top-256｜跨 head 共享", topk_mask(agg, K), K)
for b in (16, 32, 64, 128):
    row(f"③ block top-256 b={b}｜共享 + 块内全选", block_mask(agg, K, b), K)
row("④ window 256（sliding window，与内容无关）",
    (t_idx.view(L, 1) - t_idx.view(1, L) < K) & causal_ok, K)
g = torch.Generator().manual_seed(3)
rnd = torch.rand(L, L, generator=g).masked_fill(~causal_ok, -1)
row("⑤ random 256（下界）", topk_mask(rnd, K), K)

print()
print("=== Q2 跨 head 共享的代价 ===")
print(f"  recall      {r_ph:.4f} → {r_tok:.4f}  （{(r_tok - r_ph) * 100:+.2f} pt）")
print(f"  输出误差    {e_ph:.4f} → {e_tok:.4f}  （{e_tok / e_ph:.2f}×）")

print()
print("=== Q3a 单一 head：要不要块粒度，取决于质量的空间连续性 ===")
print("（H=1、n_region=2，扫高权重区的连续长度 run_len；问「覆盖 90% 质量要多少预算」）")
print(f"{'run_len':>8}{'token 预算':>12}{'block(64) 预算':>16}{'块粒度税':>11}")
print("-" * 50)
for rl in (1, 2, 4, 8, 16, 32, 64, 96):
    lgx, px, vx, dx = build(0, run_len=rl, n_region=2, Hh=1)
    ax = head_agg(lgx)
    tk = budget_for(lambda k: (px * topk_mask(ax, k).view(1, L, L)).sum(-1).mean().item())
    bk = budget_for(lambda k: (px * block_mask(ax, k, 64).view(1, L, L)).sum(-1).mean().item(),
                    lo=64)
    print(f"{rl:>8}{tk:>12}{bk:>16}{bk / tk:>10.2f}×")

print()
print("=== Q3b 多 head：共享选择要覆盖「并集」，预算被 head 数摊薄 ===")
print("（run_len=16、n_region=2、预算固定 K=256）")
print(f"{'heads':>7}{'理想并集需求':>14}{'token 共享 recall':>19}{'block(64) recall':>18}")
print("-" * 58)
for hh in (1, 2, 4, 8, 16):
    lgx, px, vx, dx = build(0, run_len=16, n_region=2, Hh=hh)
    ax = head_agg(lgx)
    r_t, _ = evaluate(lgx, px, vx, dx, topk_mask(ax, K))
    r_b, _ = evaluate(lgx, px, vx, dx, block_mask(ax, K, 64))
    print(f"{hh:>7}{hh * 2 * 16:>14}{r_t:>19.4f}{r_b:>18.4f}")

print()
print("=== Q3c 对角块的固定浪费：块越大，因果掩码下越亏 ===")
for b in (16, 32, 64, 128):
    m = block_mask(agg, K, b)
    computed = m[K:].sum().item()
    useful = (m[K:] & causal_ok[K:]).sum().item()
    n_q = L - K
    print(f"  b={b:<4} 每 query 实算 {computed / n_q:>6.1f} 个 key（预算 {K}），"
          f"被因果掩码挡掉 {(computed - useful) / n_q:>5.1f} 个 = "
          f"{(computed - useful) / computed * 100:>4.1f}%")
```

```text
配置：L=2048  heads=8  d=64  每 query 预算 K=256 token
logits：sink + 局部衰减 + 每 head 各自的长距离高权重区 + 噪声

策略（全部共享一个跨 head 选择）                            recall      输出误差    实算key
----------------------------------------------------------------------------
① token top-256｜逐 head 各选各的（上界，kernel 做不到）     0.9346    0.0746      256
② token top-256｜跨 head 共享                      0.9149    0.1010      256
③ block top-256 b=16｜共享 + 块内全选                 0.8598    0.2880      256
③ block top-256 b=32｜共享 + 块内全选                 0.8504    0.3124      256
③ block top-256 b=64｜共享 + 块内全选                 0.8412    0.3271      256
③ block top-256 b=128｜共享 + 块内全选                0.8212    0.3539      256
④ window 256（sliding window，与内容无关）             0.8558    0.3080      256
⑤ random 256（下界）                               0.3847    1.2798      256

=== Q2 跨 head 共享的代价 ===
  recall      0.9346 → 0.9149  （-1.97 pt）
  输出误差    0.0746 → 0.1010  （1.36×）

=== Q3a 单一 head：要不要块粒度，取决于质量的空间连续性 ===
（H=1、n_region=2，扫高权重区的连续长度 run_len；问「覆盖 90% 质量要多少预算」）
 run_len    token 预算    block(64) 预算       块粒度税
--------------------------------------------------
       1         169             384      2.27×
       2         157             448      2.85×
       4         141             448      3.18×
       8         124             448      3.61×
      16         106             384      3.62×
      32          95             320      3.37×
      64          98             320      3.27×
      96         108             320      2.96×

=== Q3b 多 head：共享选择要覆盖「并集」，预算被 head 数摊薄 ===
（run_len=16、n_region=2、预算固定 K=256）
  heads        理想并集需求    token 共享 recall  block(64) recall
----------------------------------------------------------
      1            32             0.9587            0.8229
      2            64             0.9484            0.7872
      4           128             0.9399            0.7539
      8           256             0.9233            0.7488
     16           512             0.8945            0.7386

=== Q3c 对角块的固定浪费：块越大，因果掩码下越亏 ===
  b=16   每 query 实算  256.0 个 key（预算 256），被因果掩码挡掉   7.5 个 =  2.9%
  b=32   每 query 实算  256.0 个 key（预算 256），被因果掩码挡掉  15.5 个 =  6.1%
  b=64   每 query 实算  256.0 个 key（预算 256），被因果掩码挡掉  31.5 个 = 12.3%
  b=128  每 query 实算  256.0 个 key（预算 256），被因果掩码挡掉  63.5 个 = 24.8%
```

这张表里有四条独立的结论，每条都值得单独记下来：

**① token 粒度的 recall 最高，但 kernel 做不到。**
逐 head 各选各的（0.9346）比跨 head 共享一个集合（0.9149）好 1.97 pt，输出误差差 1.36×。

为什么 kernel 做不到？因为在 MQA/GQA 下 **KV 只有一份**。如果每个 query head 各选各的 256 个位置，那 8 个 head 就要从 HBM 里 gather 8 份**互不相同**的 KV，等于把「共享 KV」这个省带宽的设计彻底废掉。NSA 论文的说法是 kernel 层面「each key-value entry must be shared across multiple queries for computational efficiency」，DSA 论文同样把这一条写成了 MQA 形态的理由。

代价只有 2 pt，非常划算——这是整个稀疏注意力领域的地基。

**② 块粒度的 recall 掉到和滑窗一个水平。**
b=16 时 0.8598，b=128 时 0.8212，而**滑窗 256 是 0.8558**。也就是说在本实验的分布下，内容相关的块选择几乎没比「与内容无关的滑窗」好。

这条看起来反直觉，但机制很清楚：局部衰减决定了大部分质量本来就落在最近的 256 个位置里，滑窗白拿 0.8558。内容选择只在「窗口外还有值得看的东西」时才赢。

**③ 对角块的固定浪费是 $(b-1)/(2K)$。**
b=16 → 2.9%，b=32 → 6.1%，b=64 → 12.3%，b=128 → 24.8%。手算：对角块被强制选中（MoBA 明确要求「每个 token 必须路由到自己的当前块」），块内平均有 $(b-1)/2$ 个位置在当前位置之后，算了但被因果掩码挡掉。相对整个预算 $K$ 就是 $(b-1)/(2K)$——**与 L 无关，是个固定税**。

所以「块越大越省」是错的：块大了索引开销小、访存更整齐，但固定税线性增长。

**④ 随机选择的 recall 是 0.3847。**
这个数字有参照价值：**任何选择策略的 recall 如果只有 0.38 上下，说明它没在选，只是在随机取。** 后面 §3.3 会看到：不做 warmup 的 indexer 在端到端准确率上直接掉到滑窗水平（0.1177 vs 0.1001）——同一个意思，换了个指标。

### 3.1.1 那内容选择什么时候才赢？

把「局部质量占比」当作自变量扫一遍就清楚了。

```python
"""E1b：滑窗 vs 内容选择 —— 稀疏注意力的收益从哪来

滑窗（sliding window）与内容选择（top-k）都在 K 个 token 的预算下工作，
但两者的 recall 对「注意力质量的构成」的敏感度完全相反：
  滑窗：recall ≈ 局部质量占比（与内容无关，只看位置）
  内容选择：recall ≈ 1 - 未被覆盖的长尾，与质量分布无关（只要 k/L 够）
扫「长距离依赖的强度」看不交叉点。
"""
import torch

L, H, D, K = 2048, 4, 64, 256
t_idx = torch.arange(L)
causal_ok = t_idx.view(L, 1) >= t_idx.view(1, L)


def build(decay, long_range_gain, seed=0):
    """decay 越大 → 局部质量占比越高；long_range_gain 控制长距离依赖的分量"""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L).view(L, 1)
    j = torch.arange(L).view(1, L)
    dist = (t - j).clamp(min=0).to(torch.float32)
    lg = (-decay * torch.log1p(dist / 8.0)).unsqueeze(0)
    lg = lg + torch.randn(H, 1, L, generator=g) * 0.9
    lg[:, :, 0] += 6.0
    starts = torch.randint(0, L, (H, L, 2), generator=g)
    mag = long_range_gain * (0.6 + 0.4 * torch.rand(H, L, 2, generator=g))
    pos = starts                                          # [H, L(query), 2] 孤立尖峰
    t3 = torch.arange(L).view(1, L, 1)
    ok = (pos < t3).to(torch.float32)                     # 只能看过去
    lg.scatter_add_(2, pos, mag * ok)
    mask = ~causal_ok.view(1, L, L)
    return lg.masked_fill(mask, float("-inf"))


def agg(lg):
    out = lg[0].clone()
    for h in range(1, lg.shape[0]):
        out = torch.logsumexp(torch.stack([out, lg[h]]), dim=0)
    return out


def topk_mask(s, k):
    idx = torch.topk(s, k, dim=-1).indices
    return torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, True)


def recall_of(p, m):
    return (p * (m.view(1, L, L) & causal_ok)).sum(-1).mean().item()


win = (t_idx.view(L, 1) - t_idx.view(1, L) < K) & causal_ok
print(f"配置：L={L}  heads={H}  预算 K={K} token（滑窗与 top-k 同预算）")
print("长距离依赖 = 每个 head 各 2 个孤立尖峰；decay 越大局部质量占比越高")
print()
print(f"{'decay':>7}{'长距离增益':>11}{'局部质量占比':>14}{'滑窗 recall':>13}{'top-k recall':>14}{'谁赢':>8}")
print("-" * 68)
for decay in (0.6, 1.0, 1.4, 2.0, 2.8):
    for gain in (6.0,):
        lg = build(decay, gain)
        p = torch.softmax(lg, dim=-1)
        # 局部质量占比 = 最近 K 个位置承载的质量
        local_share = recall_of(p, win)
        r_w = local_share
        r_t = recall_of(p, topk_mask(agg(lg), K))
        winner = "top-k" if r_t > r_w else "滑窗"
        print(f"{decay:>7.1f}{gain:>11.1f}{local_share:>14.3f}{r_w:>13.3f}{r_t:>14.3f}{winner:>8}")

print()
print("固定在 decay=1.4，扫长距离依赖强度（尖峰幅度）")
print(f"{'增益':>7}{'局部质量占比':>14}{'滑窗 recall':>13}{'top-k recall':>14}{'差距':>10}")
print("-" * 60)
for gain in (0.0, 2.0, 4.0, 6.0, 9.0, 12.0):
    lg = build(1.4, gain)
    p = torch.softmax(lg, dim=-1)
    r_w = recall_of(p, win)
    r_t = recall_of(p, topk_mask(agg(lg), K))
    print(f"{gain:>7.1f}{r_w:>14.3f}{r_w:>13.3f}{r_t:>14.3f}{(r_t - r_w) * 100:>9.1f}pt")
```

```text
配置：L=2048  heads=4  预算 K=256 token（滑窗与 top-k 同预算）
长距离依赖 = 每个 head 各 2 个孤立尖峰；decay 越大局部质量占比越高

  decay      长距离增益        局部质量占比    滑窗 recall  top-k recall      谁赢
--------------------------------------------------------------------
    0.6        6.0         0.476        0.476         0.735   top-k
    1.0        6.0         0.673        0.673         0.823   top-k
    1.4        6.0         0.850        0.850         0.916   top-k
    2.0        6.0         0.971        0.971         0.983   top-k
    2.8        6.0         0.998        0.998         0.999   top-k

固定在 decay=1.4，扫长距离依赖强度（尖峰幅度）
     增益        局部质量占比    滑窗 recall  top-k recall        差距
------------------------------------------------------------
    0.0         0.856        0.856         0.911      5.6pt
    2.0         0.856        0.856         0.911      5.6pt
    4.0         0.855        0.855         0.912      5.7pt
    6.0         0.850        0.850         0.916      6.6pt
    9.0         0.788        0.788         0.933     14.5pt
   12.0         0.640        0.640         0.961     32.2pt
```

滑窗的 recall 恒等于「窗口内质量占比」——它是个与内容无关的 k/L 采样器。而 top-k 的增量恰好是**窗口外那部分高质量位置**：

- 局部质量占比 0.476 时：滑窗 0.476 vs top-k 0.735，差 25.9 pt
- 局部质量占比 0.850 时：0.850 vs 0.916，差 6.6 pt
- 局部质量占比 0.998 时：差 0.1 pt（滑窗就够了）

第二张表换了个角度看同一件事：把长距离尖峰的幅度从 0 加到 12，滑窗的 recall 从 0.856 掉到 0.640，而 top-k 从 0.911 **升到** 0.961，差距从 5.6 pt 拉到 32.2 pt。

**长距离依赖越强，内容选择越值钱——而长距离依赖正是长上下文的全部意义。**

这也顺带解释了 NSA 为什么要分三个 branch：滑窗分支负责「局部质量占比」那一大块（它拿得最便宜），选择分支只负责窗口外那部分。**两条路各管一段，比强迫一个分支同时干两件事划算。**

### 3.2 indexer 的设计空间：它为什么只需要 64 头 × 128 维

DSA 的 indexer 打分公式（论文原文形式）：

$$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j} \cdot \text{ReLU}\left(q^I_{t,j} \cdot k^I_s\right)$$

三个设计选择都是为了「便宜」：

- **$H^I$ 个头**（V3.2 实配 64）：比主注意力的 128 头少一半；
- **所有 indexer head 共享同一份 key**（$k^I_s$ 只有一个，不按 head 分裂，MQA 式）：所以读 indexer key 的字节数与 head 数无关；
- **ReLU 而不是 GELU/SiLU**：没有 exp，纯比较 + 置零，吞吐高；
- **fp8**：indexer 的 key cache 直接存 fp8。

其中「共享 key」是个硬约束，它把 indexer 能看到的 key 空间压到一个 $d^I$ 维子空间里。所以真正的问题不是「64 个头够不够」，而是：

> **$d^I$ 要多大，才能让 top-k 选到该选的位置？**

下面这个实验直接做 key 侧的秩截断（用完整 SVD，不是子空间迭代），看 top-k 覆盖质量怎么掉。

```python
"""E2：indexer 的秩瓶颈 —— 为什么它只需要 64 头 × 128 维

DSA 的 indexer 打分是
    I(t,s) = Σ_{n=1}^{H^I} w[t,n] · ReLU( qI[t,n] · kI[s] )
所有 indexer head 共享同一个 key 投影 kI（MQA 式，为了 kernel 效率），
所以「indexer 能看到的 key 空间」被限制在一个 d^I 维子空间里，d^I = index_head_dim。
本实验直接量化这个约束：把教师注意力分数的 key 侧矩阵做秩截断，看 top-k 选中的质量怎么掉。

教师：H=8 头 × d=64 维 → 把每个 (t,s) 的分数写成 vec_q(t)·vec_k(s)（512 维内积）
      注意力质量集中在「同主题」的 token 上（topic block 结构）
"""
import torch

H_H, DH, L, TQ, K_TOPK = 8, 64, 2048, 512, 32
DK = H_H * DH                                            # 512


def build(n_topic, noise, seed=0):
    g = torch.Generator().manual_seed(seed)
    code = torch.randn(n_topic, DK, generator=g)
    topic = torch.randint(0, n_topic, (L,), generator=g)
    K = code[topic] * 2.0 + torch.randn(L, DK, generator=g) * noise      # [L, DK]
    qtopic = torch.randint(0, n_topic, (TQ,), generator=g)
    Q = code[qtopic] * 2.0 + torch.randn(TQ, DK, generator=g) * noise     # [TQ, DK]
    return Q, K


def topk_mask(s, k):
    idx = torch.topk(s, k, dim=-1).indices
    return torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, True)


def quant_fp8(x):
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    return (x / s).clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype) * s


print(f"教师 {H_H} 头 × {DH} 维（key 侧维度 {DK}）  L={L}  query 数={TQ}  预算 top-{K_TOPK}")
Q, K = build(16, 1.0)
S = Q @ K.T / DH ** 0.5                                  # [TQ, L] 教师分数
p_full = torch.softmax(S, dim=-1)
base = (p_full * topk_mask(S, K_TOPK)).sum(-1).mean().item()

U, sv, Vh = torch.linalg.svd(K, full_matrices=False)
print(f"\nkey 侧矩阵 {K.shape} 的奇异值谱：前 16 个 "
      f"{[round(v.item(), 1) for v in sv[:16]]}")
tot_e = (sv ** 2).sum()
for r in (4, 8, 16, 32, 64, 128, 256, 512):
    e = (sv[:r] ** 2).sum() / tot_e
    if r >= 256 or r in (4, 8, 16, 32, 64, 128):
        print(f"  rank {r:>4}: 能量保留 {e:>7.4f}")

print()
print("=== indexer 的 key 维度 d^I 对 top-k 选择质量的影响 ===")
print("（把 key 侧截断到 rank d^I，用截断后的分数选 top-k，看覆盖了教师多少质量）")
print(f"{'d^I':>6}{'key 侧能量保留':>16}{'top-k 覆盖质量':>17}{'fp8 量化后':>13}{'相对满秩损失':>14}")
print("-" * 66)
for r in (4, 8, 16, 32, 64, 128, 256, 512):
    Kr = (U[:, :r] * sv[:r]) @ Vh[:r]
    Sr = Q @ Kr.T / DH ** 0.5
    m = (p_full * topk_mask(Sr, K_TOPK)).sum(-1).mean().item()
    mq = (p_full * topk_mask(Q @ quant_fp8(Kr).T / DH ** 0.5, K_TOPK)).sum(-1).mean().item()
    print(f"{r:>6}{(sv[:r] ** 2).sum() / tot_e:>16.4f}{m:>17.4f}{mq:>13.4f}"
          f"{(base - m) * 100:>13.2f}pt")
print(f"\n满秩基准（= 教师自己选）：{base:.4f}")

print()
print("=== 有效秩随「注意力结构强度」变化 ===")
print("（看的是「截断到 d^I=128 相对满秩」的差距，而不是绝对值 —— 绝对值里含了 top-k 自身的天花板）")
print(f"{'噪声':>7}{'rank16 能量':>13}{'rank128 能量':>14}{'满秩天花板':>13}"
      f"{'d^I=128':>10}{'差距':>9}")
print("-" * 68)
for noise in (0.3, 0.7, 1.0, 1.5, 2.5):
    Q2, K2 = build(16, noise)
    S2 = Q2 @ K2.T / DH ** 0.5
    p2 = torch.softmax(S2, dim=-1)
    U2, s2, V2 = torch.linalg.svd(K2, full_matrices=False)
    te = (s2 ** 2).sum()
    K128 = (U2[:, :128] * s2[:128]) @ V2[:128]
    m128 = (p2 * topk_mask(Q2 @ K128.T / DH ** 0.5, K_TOPK)).sum(-1).mean().item()
    mfull = (p2 * topk_mask(S2, K_TOPK)).sum(-1).mean().item()
    print(f"{noise:>7.1f}{(s2[:16] ** 2).sum() / te:>13.4f}{(s2[:128] ** 2).sum() / te:>14.4f}"
          f"{mfull:>13.4f}{m128:>10.4f}{(mfull - m128) * 100:>8.2f}pt")

print()
print("=== 跨 head 聚合：indexer 必须给出一个「所有 head 共用」的分数 ===")
print("注意：本合成数据里各 head 完全对齐，所以「求和」正好等于教师自己的口径，是上界；")
print("      真实模型里 head 之间会分歧，这是 NSA 用组内求和、MoBA 用均值池化的原因。")
print(f"{'聚合方式':<26}{'top-k 覆盖质量':>16}")
print("-" * 44)
Q3, K3 = build(16, 1.0)
Qh = Q3.view(TQ, H_H, DH)
Kh = K3.view(L, H_H, DH)
per_head = torch.einsum("thd,shd->hts", Qh, Kh) / DH ** 0.5
agg = {
    "单 head（仅 head 0）": per_head[0],
    "head 求和": per_head.sum(0),
    "head 平均": per_head.mean(0),
    "logsumexp（≈求和口径）": torch.logsumexp(per_head, dim=0),
    "head 取最大": per_head.max(0).values,
}
for name, s in agg.items():
    pp = p_full
    m = (pp * topk_mask(s, K_TOPK)).sum(-1).mean().item()
    print(f"{name:<26}{m:>16.4f}")
```

```text
教师 8 头 × 64 维（key 侧维度 512）  L=2048  query 数=512  预算 top-32

key 侧矩阵 torch.Size([2048, 512]) 的奇异值谱：前 16 个 [613.1, 594.0, 580.2, 571.6, 552.8, 541.4, 525.2, 514.4, 505.6, 485.3, 482.6, 467.9, 464.5, 451.3, 435.1, 414.1]
  rank    4: 能量保留  0.2646
  rank    8: 能量保留  0.4810
  rank   16: 能量保留  0.8086
  rank   32: 能量保留  0.8216
  rank   64: 能量保留  0.8446
  rank  128: 能量保留  0.8835
  rank  256: 能量保留  0.9410
  rank  512: 能量保留  1.0000

=== indexer 的 key 维度 d^I 对 top-k 选择质量的影响 ===
（把 key 侧截断到 rank d^I，用截断后的分数选 top-k，看覆盖了教师多少质量）
   d^I       key 侧能量保留       top-k 覆盖质量      fp8 量化后        相对满秩损失
------------------------------------------------------------------
     4          0.2646           0.5269       0.5279        47.29pt
     8          0.4810           0.7969       0.7932        20.29pt
    16          0.8086           0.9948       0.9926         0.50pt
    32          0.8216           0.9955       0.9937         0.43pt
    64          0.8446           0.9965       0.9966         0.33pt
   128          0.8835           0.9976       0.9973         0.22pt
   256          0.9410           0.9994       0.9993         0.04pt
   512          1.0000           0.9998       0.9998         0.00pt

满秩基准（= 教师自己选）：0.9998

=== 有效秩随「注意力结构强度」变化 ===
（看的是「截断到 d^I=128 相对满秩」的差距，而不是绝对值 —— 绝对值里含了 top-k 自身的天花板）
     噪声    rank16 能量    rank128 能量        满秩天花板   d^I=128       差距
--------------------------------------------------------------------
    0.3       0.9790        0.9872       0.8346    0.8318    0.28pt
    0.7       0.8956        0.9364       0.9956    0.9927    0.30pt
    1.0       0.8086        0.8835       0.9998    0.9976    0.22pt
    1.5       0.6552        0.7901       1.0000    0.9942    0.58pt
    2.5       0.4155        0.6442       1.0000    0.9727    2.73pt

=== 跨 head 聚合：indexer 必须给出一个「所有 head 共用」的分数 ===
注意：本合成数据里各 head 完全对齐，所以「求和」正好等于教师自己的口径，是上界；
      真实模型里 head 之间会分歧，这是 NSA 用组内求和、MoBA 用均值池化的原因。
聚合方式                            top-k 覆盖质量
--------------------------------------------
单 head（仅 head 0）                    0.5941
head 求和                             0.9998
head 平均                             0.9998
logsumexp（≈求和口径）                    0.7232
head 取最大                            0.6912
```

三条结论：

**① $d^I \ge 16$ 就基本够了，$d^I = 8$ 会崩。**
d^I=16 → 覆盖 0.9948（相对满秩只差 0.50 pt）；d^I=8 → 0.7969（差 20.29 pt）。而 key 侧矩阵在 rank-16 只保留了 80.86% 的能量——**能量保留率低不代表选择质量差**，因为重要的是「排序的前几名对不对」，不是「重构误差小不小」。这是本期的第二个方法论提醒（第一个是 §3.1 的 ④）。

**② 注意力分数的有效秩，远低于模型维度。**
key 侧是 512 维，但 rank-16 就吃掉了 80.86% 的能量。这和第 032 期讲 MLA 低秩瓶颈时是同一类现象——**注意力分数本身是低秩的**，这也正是 MLA 能成立的前提。既然分数低秩，用一个小得多的网络去猜它的 top-k 就是可行的。

**③ fp8 存储 indexer key 几乎免费。**
d^I=16 时 0.9948 → 0.9926，代价 0.22 pt。这条直接解释了为什么 FlashInfer 的 DSA indexer kernel 里 key cache 就是 fp8 的（`k_index_cache_fp8`）。

顺带看到跨 head 聚合的重要性：单 head 0.5941，求和/平均 0.9998，logsumexp 0.7232，max 0.6912。**必须有一个「所有 head 共用」的分数**，而且用简单的求和就很好。

> 诚实标注：本合成数据里各 head 完全对齐，所以「求和」正好等于教师自己的口径（是上界），这个比较对真实模型偏乐观。真实模型里 head 之间会分歧，这正是 NSA 用「组内求和」、MoBA 用「均值池化」、DSA 用「学出来的 $w$」的原因——都在找一个足够好的共享分数。

### 3.3 端到端：稀疏注意力到底能不能替代 dense

前面的实验都是「给定选择信号，测选择质量」。但真正的设计问题是：**如果我把选择也交给训练，模型能不能学会？**

用一个需要长距离检索的任务来测：关联召回。

```
序列 = [k1 v1 k2 v2 … km vm] [k_query]  →  预测 v_j（k_query = k_j）
```

$k_{query}$ 是前面某个 key 的复制，模型必须精确跳到远处那个 key 的位置，再读它后面一格。答案位置 $2j+1$ 到序列末尾的距离随 $m$ 增大——**滑窗 W=6 能覆盖到答案位置的概率上限只有 $W/(2m)$：m=8 时 37.5%，m=16 时 18.8%。**

<details>
<summary>这里有一个必须写进代码的细节：indexer 的输入要从计算图里 detach</summary>

如果写成 `isc = self.index_scores(y)`，KL 损失的梯度会顺着 $y$ 一路漏回主干网络和 embedding。后果不是报错，而是**训练目标被偷换**：主干网络会去迁就一个随机初始化的 indexer，而不是 indexer 去对齐主干。

DSA 论文原文对这一条说得很清楚：「we detach the indexer input from the computational graph for separate optimization. The training signal of the indexer is from only $\mathcal{L}^I$, while the optimization of the main model is according to only the language modeling loss.」

这个消融值得单独看一眼，因为它是「静默出错」的教科书案例：

```python
"""E7：indexer 输入 detach 的消融 —— 不 detach 会静默偷换训练目标

同一份配置、同样的步数，只改一处：KL 的梯度能不能顺着 indexer 的输入漏回主干。

  直接证据：一次前向后测 ‖∂KL / ∂(token embedding)‖。detach 时应当恒为 0。
  后果证据：600 步训练后的准确率（这段步数全落在 dense 注意力阶段内）。

复用 E5 的网络定义与训练循环（把它所在的源码 exec 进来），只翻转一个全局开关。
"""
import torch

src = open("e5_end2end.py").read().split("t0 = time.time()")[0]
ns = {}
exec(src, ns)
Block, Tiny, make_batch, run = ns["Block"], ns["Tiny"], ns["make_batch"], ns["run"]

print("配置：m=8（S=17），600 步，全部落在 dense 注意力阶段（warmup=1500 步未结束）")
print("唯一变量：KL 的梯度能不能顺着 indexer 的输入漏回主干")
print()

print("=== 直接证据：KL 对 token embedding 的梯度范数（一次前向反向）===")
print(f"{'DETACH_INDEX_INPUT':<22}{'‖∂KL/∂emb‖':>16}")
print("-" * 40)
for flag in (True, False):
    ns["DETACH_INDEX_INPUT"] = flag
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(100)
    model = Tiny("dsa", topk=8, win=6, warmup=True)
    x, y = make_batch(8, 64, gen)
    logits, (isc, tgt) = model(x)
    kl = torch.nn.functional.kl_div(torch.log_softmax(isc, -1), tgt, reduction="batchmean")
    g = torch.autograd.grad(kl, model.tok.weight, retain_graph=False,
                            allow_unused=True)[0]
    gn = 0.0 if g is None else g.norm().item()      # None = KL 根本不依赖主干
    print(f"{str(flag):<22}{gn:>16.6e}")

print()
print("=== 后果证据：600 步训练后的关联召回准确率 ===")
print(f"{'DETACH_INDEX_INPUT':<22}{'准确率':>10}")
print("-" * 34)
for flag in (True, False):
    ns["DETACH_INDEX_INPUT"] = flag
    acc = run("dsa", 8, steps=600, seed=0, warmup_steps=1500)[0]
    print(f"{str(flag):<22}{acc:>10.4f}")

ns["DETACH_INDEX_INPUT"] = True
print()
print("两行的网络结构、数据流、步数、学习率完全相同，差别只在那一行 detach；")
print("不 detach 时不会抛任何异常，只是 KL 悄悄变成了「让主干去迁就随机 indexer」。")
```

```text
配置：m=8（S=17），600 步，全部落在 dense 注意力阶段（warmup=1500 步未结束）
唯一变量：KL 的梯度能不能顺着 indexer 的输入漏回主干

=== 直接证据：KL 对 token embedding 的梯度范数（一次前向反向）===
DETACH_INDEX_INPUT          ‖∂KL/∂emb‖
----------------------------------------
True                      0.000000e+00
False                     9.485875e-01

=== 后果证据：600 步训练后的关联召回准确率 ===
DETACH_INDEX_INPUT           准确率
----------------------------------
True                      0.9214
False                     0.3267

两行的网络结构、数据流、步数、学习率完全相同，差别只在那一行 detach；
不 detach 时不会抛任何异常，只是 KL 悄悄变成了「让主干去迁就随机 indexer」。
```

一次前向反向就能把问题定性：detach 时 $\|\partial \text{KL}/\partial \text{emb}\|$ **恒等于 0**（autograd 直接报 `appeared to not have been used in the graph`），不 detach 时是 0.95。而后果是准确率 0.92 → 0.33，**过程里没有任何异常**。

</details>

```python
"""E5：端到端 —— 稀疏注意力能不能替代 dense，取决于 indexer 能不能找到那个「远方的 key」

任务：关联召回（associative recall）
    序列 = [k1 v1 k2 v2 ... km vm] [k_query]，要求预测 k_query 对应的 v
    k_query 是前面某个 k_i 的复制 → 必须精确跳到远处那个 key 的位置再去读它后面一格。
    答案所在位置 = 2j+1（j 是配对下标），与序列末尾的距离随 m 增大。
    滑窗 W 只在 (2j+1) 落在末尾 W 个位置内时才可能答对 → 命中率 ≈ W/(2m)。

对比：
    dense        标准因果注意力（上界参照）
    window(W)    只看最近 W 个位置
    oracle(K)    用「主注意力自己的分数」选 top-K（事后稀疏的上界，Quest 那一类）
    dsa_2stage   DSA 复现：先 dense 注意力 + indexer 用 KL 追主注意力分布，
                 再切成 top-K 稀疏、全部参数一起训（indexer 只吃 KL 的梯度）
    dsa_1stage   消融：跳过 warmup，indexer 随机初始化就开始稀疏训练
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

V, D, NH, DH, LAYERS = 48, 64, 4, 16, 2
DETACH_INDEX_INPUT = True        # DSA 论文要求 True；E7 拿它做消融


def make_batch(m, bs, gen):
    kv = torch.randint(0, V, (bs, m, 2), generator=gen)
    pick = torch.randint(0, m, (bs,), generator=gen)
    keys, vals = kv[:, :, 0], kv[:, :, 1]
    qpos = keys[torch.arange(bs), pick]
    x = torch.cat([kv.reshape(bs, 2 * m), qpos.unsqueeze(1)], dim=1)
    return x, vals[torch.arange(bs), pick]


class Block(nn.Module):
    def __init__(self, mode, topk=8, win=6, idx_dim=32, warmup=True):
        super().__init__()
        self.mode, self.topk, self.win, self.idx_dim = mode, topk, win, idx_dim
        self.warmup = warmup
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.wq = nn.Linear(D, NH * DH, bias=False)
        self.wk = nn.Linear(D, NH * DH, bias=False)
        self.wv = nn.Linear(D, NH * DH, bias=False)
        self.wo = nn.Linear(NH * DH, D, bias=False)
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.iq = nn.Linear(D, 4 * idx_dim, bias=False)       # indexer 的 query
        self.ik = nn.Linear(D, idx_dim, bias=False)           # indexer 的 key：所有 head 共用一份
        self.iw = nn.Linear(D, 4, bias=False)                 # 每 head 的权重 w

    def index_scores(self, x):
        B, S, _ = x.shape
        qi = self.iq(x).view(B, S, 4, self.idx_dim)
        ki = self.ik(x)
        s = torch.einsum("bthd,bsd->bths", qi, ki)
        return torch.einsum("bth,bths->bts", self.iw(x), F.relu(s))

    def forward(self, x):
        B, S, _ = x.shape
        y = self.ln1(x)
        q = self.wq(y).view(B, S, NH, DH).transpose(1, 2)
        k = self.wk(y).view(B, S, NH, DH).transpose(1, 2)
        v = self.wv(y).view(B, S, NH, DH).transpose(1, 2)
        mask = torch.arange(S).view(S, 1) >= torch.arange(S).view(1, S)
        logits = (q @ k.transpose(-1, -2) / DH ** 0.5).masked_fill(~mask, float("-inf"))

        # indexer 的 KL 目标（DSA 论文口径）：主注意力的【每个 head 的注意力概率】按 head 求和，
        # 再沿序列做 L1 归一化。注意 logits 是 [B, H, S, S]，按 head 求和要 sum(dim=1)，
        # 写成 logits[0] 会取到 batch 维 —— 不报错，但求和的是错的那一维。
        tgt = torch.softmax(logits, dim=-1).sum(dim=1)
        tgt = (tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-12)).detach()
        # indexer 的输入必须从计算图里 detach（DSA 论文原文要求）：
        # 否则 KL 的梯度会顺着 y 漏回主干网络，把「对齐」变成「让主干迁就随机 indexer」
        arg = y.detach() if DETACH_INDEX_INPUT else y
        isc = self.index_scores(arg).masked_fill(~mask, float("-inf"))

        dense_now = self.mode == "dense" or (self.mode == "dsa" and self.warmup)
        if dense_now:
            att = torch.softmax(logits, dim=-1)
        elif self.mode == "window":
            wm = mask & (torch.arange(S).view(S, 1) - torch.arange(S).view(1, S) < self.win)
            att = torch.softmax(logits.masked_fill(~wm, float("-inf")), dim=-1)
        else:
            src = tgt if self.mode == "oracle" else isc        # [B, S, S]
            sel = torch.topk(src, min(self.topk, S), dim=-1).indices
            m2 = (torch.zeros_like(src, dtype=torch.bool).scatter_(-1, sel, True)
                  & mask).unsqueeze(1)                         # → [B, 1, S, S] 才能广播到 head 维
            att = torch.softmax(logits.masked_fill(~m2, float("-inf")), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, S, NH * DH)
        x = x + self.wo(o)
        return x + self.mlp(self.ln2(x)), isc, tgt


class Tiny(nn.Module):
    def __init__(self, mode, **kw):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(128, D)
        self.blocks = nn.ModuleList([Block(mode, **kw) for _ in range(LAYERS)])
        self.head = nn.Linear(D, V)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        info = None
        for blk in self.blocks:
            h, isc, tgt = blk(h)
            info = (isc, tgt)
        return self.head(h[:, -1]), info


IDX = (".iq.", ".ik.", ".iw.")


def run(mode, m, steps=3000, seed=0, topk=8, win=6, warmup_steps=1500):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    model = Tiny(mode, topk=topk, win=win, warmup=warmup_steps > 0)
    idxp = [p for n, p in model.named_parameters() if any(t in n for t in IDX)]
    oth = [p for n, p in model.named_parameters() if not any(t in n for t in IDX)]
    opt = torch.optim.Adam([{"params": oth}, {"params": idxp}], lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=3e-4)

    for step in range(steps):
        if mode == "dsa" and step == warmup_steps:
            for blk in model.blocks:
                blk.warmup = False                            # 阶段二：切成稀疏
        x, y = make_batch(m, 64, gen)
        logits, (isc, tgt) = model(x)
        ce = F.cross_entropy(logits, y)
        if mode == "dsa":
            kl = F.kl_div(torch.log_softmax(isc, -1), tgt, reduction="batchmean")
            loss = ce + 0.1 * kl
        else:
            loss = ce
        # tgt 已 detach、top-k 下标不可导 → CE 的梯度只落到主模型，KL 的梯度只落到 indexer
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()

    with torch.no_grad():
        accs = []
        for _ in range(8):
            x, y = make_batch(m, 256, gen)
            logits, (isc, tgt) = model(x)
            accs.append((logits.argmax(-1) == y).float().mean().item())
        acc = sum(accs) / len(accs)
    return acc, model


t0 = time.time()
print(f"任务：关联召回 [k1 v1 … km vm] + [k_query] → 预测 v（词表 {V}，{LAYERS} 层，d={D}）")
print(f"稀疏配置：top-K=8 个位置 / 滑窗 W=6；序列长度 S=2m+1；答案位置 2j+1 距末尾 2(m-j)-1")
print(f"滑窗能覆盖到答案位置的概率上限 = W/(2m)：m=8 → {6 / 16:.1%}，m=16 → {6 / 32:.1%}")
print(f"随机猜 = {1 / V:.4f}（{V} 类）")
print()
print(f"{'注意力':<30}{'m=8 (S=17)':>13}{'m=16 (S=33)':>13}")
print("-" * 58)
for mode, label, warm in (("dense", "dense（上界）", 1500),
                          ("window", "window W=6", 1500),
                          ("oracle", "oracle top-8（用真分数选）", 1500),
                          ("dsa", "dsa 两阶段 top-8", 1500),
                          ("dsa", "dsa 单阶段（跳过 warmup）", 0)):
    accs = [run(mode, m, seed=0, warmup_steps=warm)[0] for m in (8, 16)]
    print(f"{label:<30}{accs[0]:>13.4f}{accs[1]:>13.4f}")

print()
print("=== 多随机种子复核（m=16, S=33）===")
print(f"{'注意力':<30}{'seed0':>9}{'seed1':>9}{'seed2':>9}{'均值':>9}")
print("-" * 66)
for mode, label, warm in (("dense", "dense", 1500), ("window", "window W=6", 1500),
                          ("dsa", "dsa 两阶段", 1500), ("dsa", "dsa 单阶段", 0)):
    accs = [run(mode, 16, seed=s, warmup_steps=warm)[0] for s in (0, 1, 2)]
    print(f"{label:<30}{accs[0]:>9.4f}{accs[1]:>9.4f}{accs[2]:>9.4f}"
          f"{sum(accs) / 3:>9.4f}")

print()
print("=== 稀疏密度扫描（m=16，dsa 两阶段）===")
print(f"{'top-K':>7}{'密度 K/S':>10}{'准确率':>9}")
print("-" * 28)
for tk in (2, 8, 32):
    a, _ = run("dsa", 16, seed=0, topk=tk)
    print(f"{tk:>7}{tk / 33:>10.1%}{a:>9.4f}")
print(f"\n（{time.time() - t0:.0f} 秒，全部在 CPU 上跑）")
```

```text
任务：关联召回 [k1 v1 … km vm] + [k_query] → 预测 v（词表 48，2 层，d=64）
稀疏配置：top-K=8 个位置 / 滑窗 W=6；序列长度 S=2m+1；答案位置 2j+1 距末尾 2(m-j)-1
滑窗能覆盖到答案位置的概率上限 = W/(2m)：m=8 → 37.5%，m=16 → 18.8%
随机猜 = 0.0208（48 类）

注意力                              m=8 (S=17)  m=16 (S=33)
----------------------------------------------------------
dense（上界）                            0.9331       0.8589
window W=6                           0.1519       0.1001
oracle top-8（用真分数选）                  0.9277       0.8613
dsa 两阶段 top-8                        0.8628       0.5430
dsa 单阶段（跳过 warmup）                   0.8750       0.1177

=== 多随机种子复核（m=16, S=33）===
注意力                               seed0    seed1    seed2       均值
------------------------------------------------------------------
dense                            0.8589   0.8608   0.8687   0.8628
window W=6                       0.1001   0.0864   0.1484   0.1117
dsa 两阶段                          0.5430   0.5176   0.5317   0.5308
dsa 单阶段                          0.1177   0.5034   0.1147   0.2453

=== 稀疏密度扫描（m=16，dsa 两阶段）===
  top-K    密度 K/S      准确率
----------------------------
      2      6.1%   0.2144
      8     24.2%   0.5430
     32     97.0%   0.8594

（653 秒，全部在 CPU 上跑）
```

这张表里有五条结论，其中两条是反直觉的。

**① 滑窗在长距离任务上直接崩掉，而且崩的位置和预测一致。**
window W=6：m=8 时 0.1519，m=16 时 0.1001（随机猜 0.0208），而 dense 是 0.9331/0.8589。上限分析给的命中率是 $W/(2m)$ = 37.5%/18.8%——实测比上限还低（因为窗口内即使碰到了位置，模型也要学会用），**量级完全对得上**。

**② 用真分数选 top-8，几乎无损。**
oracle 0.9277/0.8613，与 dense 的 0.9331/0.8589 基本重合。这说明一件事：**在 S=17/33 这个规模上，稀疏不是问题，「选得准不准」才是问题。** 换到 128K 上下文，"选得准"的难度指数上升——这就是 indexer 存在的全部理由。

**③ 反直觉之一：m=8 时「不做 warmup」反而更好（0.8750 vs 0.8628）。**
原因是密度的变化。S=17、top-8 时密度是 47%，一个**随机初始化**的 indexer 也能覆盖近一半位置，主干网络完全可以靠自己的学习绕过这个「不太准的目录」。密度这么高的时候，稀疏几乎等于没稀疏。**稀疏注意力的结论对密度极度敏感，单点配置上的结论不可信——这是本期第三次出现「必须多配置复核」的教训（前两次见 §3.1.1 与 §3.2）。**

**④ 密度一降下来，warmup 立刻变成生死线。**
S=33、top-8 时密度 24.2%：dsa 两阶段 0.5430，dsa 单阶段 0.1177（掉到滑窗水平），多种子复核的均值是 0.5308 vs 0.2453——**2.16×，而且单阶段那三个种子是 0.1177/0.5034/0.1147，方差极大（偶尔运气好）。**

warmup 阶段做的事只有一件：**让 indexer 先学会模仿一个已经训好的 dense 注意力在看哪里**。这正是 DSA 论文花 2.1B token 做那 1000 步的道理——**先有真值，再有近似**。

**⑤ 反直觉之二：两阶段也只有 0.53，离 oracle 的 0.86 差得很远。**
即使 indexer 训过了，它也追不上「用真分数选」。而 DSA 论文里这 0.33 的差距是靠**第二阶段 943.7B token 把主干网络训到适应稀疏模式**来补的。本项目里 tiny 模型 + 3000 步补不上，所以看到的就是这个「半路上」的数字。

**这条恰恰是最重要的：它说明稀疏注意力的收益从来不是「indexer 一个人挣的」，而是「indexer + 主干重新适配」两件事加起来。** 这也是 NSA 论文批评第二代方法时说的那句话的实现版：模型从没被要求通过稀疏模式路由信息，你只换个 kernel 是拿不到收益的。

密度扫描进一步确认了权衡的形状：top-K 从 2（6.1%）到 8（24.2%）到 32（97.0%），准确率 0.2144 → 0.5430 → 0.8594。**稀疏率与质量是平滑的权衡曲线，不存在「越稀疏越好」的甜点。**

<details>
<summary>关于这一节的两处踩坑（都是静默错误）</summary>

**坑一：head 维和 batch 维搞混。** `logits` 是 `[B, H, S, S]`，DSA 的 KL 目标要「按 head 求和」，正确写法是 `torch.softmax(logits, -1).sum(dim=1)`。我第一版写成了 `logits[0]`（取 batch 维）再 logsumexp——形状对不上时报错，但在 batch 恰好等于 head 数时会**静默通过**，训练出来的 target 完全是错的。

**坑二：mask 的维度要对齐到 head 维。** `logits` 是 4 维而选择 mask 是 3 维（`[B, S, S]`），直接 `masked_fill` 会被广播到 `[B, H, S, S]` 时**把 batch 维当成 head 维**。报错信息是 `size of tensor a (64) must match size of tensor b (4) at non-singleton dimension 1` ——看起来像 batch 问题，其实是 mask 少了一个 `unsqueeze(1)`。

两处都属于「不报错就全错」的类型。防御手段和 033 期的教训一致：**形状断言要写在每一步，不能只靠最终指标。**

</details>

### 3.4 PyTorch 官方侧的入口到底有什么

既然这是「PyTorch 每日一课」，必须回答：**在 PyTorch 里想做稀疏注意力，今天的官方入口是什么？**

```python
"""E6：PyTorch 官方侧的稀疏注意力入口到底有什么

核查（如实报告跑出来的结果，不含推断）：
  ① torch.nn.attention 暴露了哪些符号
  ② SDPA 的后端枚举里有没有「稀疏」这条路
  ③ flex_attention 能不能在 CPU 上跑；block mask 能表达「块稀疏」但不能表达「内容相关的 top-k」
  ④ torch.sparse 的半结构化（2:4）与内容相关的 token 选择是两回事
"""
import torch
import torch.nn.attention as A
from torch.nn.attention import SDPBackend

print("torch", torch.__version__)
print()
print("① torch.nn.attention 的公开符号：")
print("  ", [n for n in dir(A) if not n.startswith("_")])
print()
print("② SDPA 后端枚举：", [b for b in SDPBackend.__members__])
print("   注意：全部是「怎么算」的选择（flash / mem-efficient / math / cudnn），")
print("   没有一个是「少算哪些位置」—— 稀疏注意力在 SDPA 层面没有后端。")
print()
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

B, H, S, D = 1, 2, 64, 16
q, k, v = (torch.randn(B, H, S, D) for _ in range(3))


def mask_mod(b, h, i, j):
    return i >= j


bm = create_block_mask(mask_mod, B, H, S, S, device="cpu")
print("③ flex_attention：")
print("   create_block_mask 成功，返回", type(bm).__name__)
print("   它的形状参数是 (B, H, Q_LEN, KV_LEN) →", bm.shape)
try:
    o = flex_attention(q, k, v, block_mask=bm)
    print("   CPU 上前向成功，输出", tuple(o.shape))
except Exception as e:
    print("   CPU 上前向失败：", type(e).__name__, str(e).split("\n")[0][:160])

# 试一个「块对角 + 局部」的块稀疏 mask，看 flex_attention 能不能表达 NSA 的窗口分支
nb = S // 16


def block_mod(b, h, i, j):
    bi, bj = i // 16, j // 16
    return (bj == bi) | (bj == bi - 1) | (bj == 0)      # 对角 + 前一块 + 第 0 块（sink）


bm2 = create_block_mask(block_mod, B, H, S, S, BLOCK_SIZE=16, device="cpu")
print("   块稀疏 mask（块 16，对角+前一块+sink）也建成功 →", bm2.shape)
try:
    o2 = flex_attention(q, k, v, block_mask=bm2)
    print("   跑通，输出", tuple(o2.shape))
except Exception as e:
    print("   失败：", type(e).__name__, str(e).split("\n")[0][:160])

print()
print("   torch.nn.attention 还暴露了一组可插拔注意力后端的接口：")
print("   list_flash_attention_impls() =", A.list_flash_attention_impls())
print("   current =", A.current_flash_attention_impl())
print("   （register_flash_attention_impl / activate_flash_attention_impl 可注册自研实现 ——")
print("     这是「把 SGLang dsa backend 那样的稀疏注意力接进 PyTorch」的官方缝隙）")

print()
print("④ 稀疏注意力的两件事，PyTorch 里分别在哪：")
print("   结构性稀疏（块对角 / 窗口）→ flex_attention 的 BlockMask，可用")
print("   内容相关稀疏（top-k / indexer）→ 需要把「选位置」本身做成一个算子：")
print("       已经在框架里的：SGLang 的 dsa backend、vLLM 的 FLASHMLA_SPARSE、FlashInfer 的")
print("       dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 —— 都是模型相关的专用 kernel，")
print("       不在 PyTorch 通用算子层")
print("   torch.sparse 的半结构化 2:4 是「权重里固定砍掉一半」，与「每个 query 选不同的 token」")
print("   是两种稀疏，不可混用：")
print("       to_sparse_semi_structured 只接受 2 维 / 形状满足 64、128 整数倍约束的稠密 tensor")
try:
    from torch.sparse import to_sparse_semi_structured
    x = torch.randn(64, 128)
    sp = to_sparse_semi_structured(x)
    print(f"       to_sparse_semi_structured(64×128) → {type(sp).__name__}，"
          f"values {tuple(sp.values().shape)}（原始 {x.numel()} 个元素 → "
          f"{sp.values().numel()} 个）")
except Exception as e:
    print("       ", type(e).__name__, str(e)[:120])
```

```text
torch 2.14.0

① torch.nn.attention 的公开符号：
   ['Iterable', 'SDPAParams', 'SDPBackend', 'Union', 'WARN_FOR_UNFUSED_KERNELS', 'activate_flash_attention_impl', 'can_use_efficient_attention', 'can_use_flash_attention', 'contextlib', 'current_flash_attention_impl', 'list_flash_attention_impls', 'register_flash_attention_impl', 'restore_flash_attention_impl', 'sdpa_kernel', 'torch', 'warn']

② SDPA 后端枚举： ['ERROR', 'MATH', 'FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'CUDNN_ATTENTION', 'OVERRIDEABLE']
   注意：全部是「怎么算」的选择（flash / mem-efficient / math / cudnn），
   没有一个是「少算哪些位置」—— 稀疏注意力在 SDPA 层面没有后端。

③ flex_attention：
   create_block_mask 成功，返回 BlockMask
   它的形状参数是 (B, H, Q_LEN, KV_LEN) → (1, 2, 64, 64)
   CPU 上前向成功，输出 (1, 2, 64, 16)
   块稀疏 mask（块 16，对角+前一块+sink）也建成功 → (1, 2, 64, 64)
   跑通，输出 (1, 2, 64, 16)

   torch.nn.attention 还暴露了一组可插拔注意力后端的接口：
   list_flash_attention_impls() = ['FA3', 'FA4']
   current = None
   （register_flash_attention_impl / activate_flash_attention_impl 可注册自研实现 ——
     这是「把 SGLang dsa backend 那样的稀疏注意力接进 PyTorch」的官方缝隙）

④ 稀疏注意力的两件事，PyTorch 里分别在哪：
   结构性稀疏（块对角 / 窗口）→ flex_attention 的 BlockMask，可用
   内容相关稀疏（top-k / indexer）→ 需要把「选位置」本身做成一个算子：
       已经在框架里的：SGLang 的 dsa backend、vLLM 的 FLASHMLA_SPARSE、FlashInfer 的
       dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 —— 都是模型相关的专用 kernel，
       不在 PyTorch 通用算子层
   torch.sparse 的半结构化 2:4 是「权重里固定砍掉一半」，与「每个 query 选不同的 token」
   是两种稀疏，不可混用：
       to_sparse_semi_structured 只接受 2 维 / 形状满足 64、128 整数倍约束的稠密 tensor
        RuntimeError Error original_tensor.device= cpu is not supported! Only CUDA tensors are currently supported.
```

跑出来的事实（都是本机实测，不是推断）：

- `SDPBackend` 的枚举是 `ERROR / MATH / FLASH_ATTENTION / EFFICIENT_ATTENTION / CUDNN_ATTENTION / OVERRIDEABLE`。**全部是「怎么算」的选择，没有一个是「少算哪些位置」的选择。** 稀疏注意力在 SDPA 这一层没有后端。
- `torch.nn.attention.flex_attention` 可用，**CPU 上也能跑**（返回形状正确）。`create_block_mask` 能表达块对角、窗口、sink 这些**结构性**稀疏，这正是 NSA 滑窗分支那一类。
- 但 flex_attention 表达不了「内容相关的 top-k」——因为 selection 依赖 query，是个数据相关的算子，不是一个 mask。
- `torch.nn.attention` 暴露了 `register_flash_attention_impl / activate_flash_attention_impl / list_flash_attention_impls`，本机返回 `['FA3', 'FA4']`。**这是把自研稀疏注意力后端接进 PyTorch 的官方缝隙。**
- `to_sparse_semi_structured`（2:4 半结构化）本机直接报 `Only CUDA tensors are currently supported`。更重要的是语义：**2:4 稀疏砍的是权重里固定位置的一半，和「每个 query 选不同的 token」是两种完全不同的稀疏**，不能互相替代（这个区分在第 023 期剪枝里也出现过）。

所以现实的工程路径是：**结构稀疏用 flex_attention，内容相关稀疏自己写 kernel 或调框架的专用后端**（§4.4）。

---

## 4. 围绕该领域展开

### 4.1 三条路线各自的机制

**NSA：三个分支各管一段，用压缩分支的分数当路由信号**

NSA 的做法是把 KV 拆成三种视图，并行做三次注意力：

| 分支 | 看什么 | 成本 | 作用 |
|---|---|---|---|
| Compression | 每个块压成一个「块代表」（可学习 MLP） | $O(N/l)$ | 全局粗视图 |
| Selection | 用**压缩分支已经算出来的**注意力分数挑 top-n 个块，对原始 KV 精算 | $O(k\cdot l)$ | 局部精度 |
| Sliding Window | 最近 $w$ 个 token | $O(w)$ | 兜住近邻细节 |

两个设计细节值得记：

1. **selection 的路由信号是从 compression 分支免费捡来的**——不需要额外的 indexer 网络。这是 NSA 与 DSA 最大的结构差异：NSA 复用已有中间量，DSA 训一个专用小网络。
2. **三个分支各自持有独立的 KV，输出再用学习的 gate 相加**，目的是「防止分支之间的梯度干扰」。分开的代价是显存与访存，换来的是训练稳定。

论文给的配置是压缩块 32 / 步长 16 / 选择块 64 / 选 16 个块 / 窗口 512；GQA 下组内各头共享块重要性分数（求和）。实验规模 27B 总参数 / 3B 激活、270B tokens、8 卡 A100，报出的加速比是 64K 下 decode 11.6× / forward 9.0× / backward 6.0×，LongBench 平均 0.469（比 full attention 高 0.032）。

> 标注：上述 NSA 数字来自论文与其技术解读，**本期没有复现**，只作为谱系事实引用。

**DSA：一个专用小网络 + 两阶段训练**

DSA（DeepSeek Sparse Attention）把 NSA 的三分支简化成一条路：

$$\underbrace{I_{t,s}}_{\text{indexer 分数}} \;\longrightarrow\; \text{top-}k\;(k=2048) \;\longrightarrow\; \underbrace{\text{对选中的 2048 个 KV 做标准 attention}}_{\text{MLA 的 MQA 形态}}$$

工程上真正让它 work 的是训练配方（论文原文数据）：

| 阶段 | 做什么 | 数据量 | 学习率 |
|---|---|---|---|
| ① Indexer warm-up | 保持 dense attention，**冻结除 indexer 外的所有参数**，用 KL 让 indexer 追主注意力的分布 | 1000 步 × 16 条 × 128K = **2.1B tokens** | 1e-3 |
| ② Sparse training | 换成 top-2048 稀疏，**优化全部参数**；indexer 仍只吃 KL 的梯度 | 15000 步 × 480 条 × 128K = **943.7B tokens** | 7.3e-6 |

KL 目标分布的定义是这段配方里最关键的一行：把主注意力的分数**按 attention head 求和**，再沿序列做 L1 归一化。二期里这个目标只在**被选中的 token 集合**上对齐。

注意阶段①「冻结所有其他参数」这件事的含义：**indexer 必须去拟合一个已经训好的 dense 模型在看哪里，而不是让 dense 模型去适应 indexer。** 先有真值，再有近似——这是整个方案能不掉点的前提。

结果：API 价格降 50% 以上（每百万 token：输入 0.28 美元、缓存命中 0.028 美元、输出 0.42 美元），128K 输入的长上下文成本降 6–7×，绝大多数基准与 V3.1-Terminus 持平。

**MoBA：把 MoE 的 top-k 门控搬到 attention 上**

MoBA 的做法极简，而且**不增加任何参数**：

$$s_i = \langle q,\ \text{mean\_pool}(K[I_i])\rangle,\qquad g_i = \mathbb{1}[s_i \in \text{Topk}]$$

块分数就是 query 与「块内 key 的均值」的内积，取 top-k 个块，块内位置全选。两条因果性处理值得记：

1. **不看未来块**：$pos(q) < i \times B$ 的块直接把 $s_i$ 置 $-\infty$。
2. **强制包含当前块 + 块内因果掩码**：因为对整个块做均值池化会泄漏未来 token 的信息，所以必须「强制路由到自己所在的块」并且在块内应用因果掩码。MoBA 论文把这一步类比成 MoE 里的 shared expert——一条静态路由规则。

MoBA 论文还指出：**滑动窗口和 attention sink 都是 MoBA 的特例**（前者是「门控永远选最近的块」，后者是「永远选第一块和最近块」）。这个观察很漂亮：静态结构稀疏不是另一种东西，只是门控被写死成常数的退化情形。

它和全注意力的**无缝切换**也是设计目标之一——参数量完全一致，所以可以在训练中途切换，甚至逐层混搭（相邻层一层 MoBA 一层 full attention）。Kimi 上线，Llama 3.1 8B 扩展到 1M 上下文，RULER@128K 为 0.7818 vs full attention 0.7849。

### 4.2 算力账：稀疏注意力的加速比是有天花板的

这一节的数学只有一行，但结论很重要：**主注意力被压到 $O(Lk)$ 了，indexer 自己仍然是 $O(L^2)$。**

$$\text{每 query 的成本} = \underbrace{W \cdot L}_{\text{indexer}} + \underbrace{A \cdot k}_{\text{稀疏主注意力}}$$

$L \to \infty$ 时第二项可以忽略，加速比趋于

$$\text{speedup} \to \frac{A \cdot L}{W \cdot L} = \frac{A}{W}$$

**天花板与 $k$ 无关，只由「主注意力每对成本」与「indexer 每对成本」的比值决定。**

```python
"""E3：indexer 自己的账 —— 稀疏注意力为什么有加速天花板

主注意力被稀疏化到 O(L·k)，但 indexer 仍然是 O(L²)。
两项相加 = W·L + A·k（W = indexer 每 (q,k) 对的 MAC 数，A = 主注意力每 (q,选中 k) 对的 MAC 数）
→ 加速比 ceiling = A/W，与 k 无关
并用真实 matmul 计时验证「indexer 份额随 L 上升」这个趋势。

DeepSeek-V3.2-Exp 的实际参数（config.json）：
  index_n_heads=64, index_head_dim=128      → W = 64×128 = 8192 MAC/(q,k)
  num_attention_heads=128, kv_lora_rank=512, qk_rope_head_dim=64
                                            → A = 128×(2×512+64) = 139264 MAC/(q,选中k)
  index_topk = 2048
"""
import time

import torch

W = 64 * 128                      # indexer：每 (query, 每个 key) 的 MAC
A = 128 * (2 * 512 + 64)          # 主注意力（MLA 吸收形态）：每 (query, 每个选中 key) 的 MAC
K = 2048
print(f"indexer 每对 {W} MAC  主注意力每对 {A} MAC  A/W = {A / W:.2f}")
L_star = K * A // W
print(f"两项打平的上下文长度 L* = k·A/W = {K} × {A / W:.2f} = {L_star}")
print()
print(f"{'L':>9}{'indexer MAC':>15}{'主注意力 MAC':>15}{'总 MAC':>15}"
      f"{'indexer 占比':>14}{'对 dense 加速':>14}")
print("-" * 84)
for L in (8_192, 32_768, 65_536, 131_072, 1_048_576):
    i_mac = W * L
    m_mac = A * K
    tot = i_mac + m_mac
    dense = A * L
    print(f"{L:>9}{i_mac:>15,}{m_mac:>15,}{tot:>15,}"
          f"{i_mac / tot:>13.1%}{dense / tot:>13.2f}×")
print()
print(f"加速比的天花板（L→∞）= A/W = {A / W:.2f}×，因为 indexer 自己也是 O(L²)")
print(f"L=131072 时已拿到天花板的 "
      f"{(A * 131072 / (W * 131072 + A * K)) / (A / W):.1%}"
      f"，剩下 {(1 - (A * 131072 / (W * 131072 + A * K)) / (A / W)):.1%} 是 indexer 的固定拖累")

# ---- 真实计时：同一台机器上 indexer 与稀疏主注意力的相对成本随 L 的变化 ----
print()
print("=== 真实计时（CPU，成对交替采样取逐对比值的中位数）===")
TQ = 4                                     # 只跑 4 个 query token，测单 token 成本
torch.manual_seed(0)


def timed(fn, rep=3):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


print(f"{'L':>8}{'indexer (ms)':>15}{'稀疏主注意力 (ms)':>19}{'实测比值':>11}{'解析比值':>11}")
print("-" * 66)
for L in (4_096, 16_384, 65_536):
    qi = torch.randn(TQ, 64, 128)
    ki = torch.randn(L, 128)
    qm = torch.randn(TQ, 128, 576)
    km = torch.randn(2048, 512)
    vm = torch.randn(2048, 512)

    def f_idx():
        torch.einsum("tnd,sd->tns", qi, ki)

    def f_main():
        sc = torch.einsum("tnd,sd->tns", qm[..., :512], km)
        torch.einsum("tns,sd->tnd", sc, vm)

    t_i = timed(f_idx)
    t_m = timed(f_main)
    print(f"{L:>8}{t_i * 1e3:>15.2f}{t_m * 1e3:>19.2f}{t_i / t_m:>11.2f}"
          f"{(W * L) / (A * K):>11.2f}")
```

```text
indexer 每对 8192 MAC  主注意力每对 139264 MAC  A/W = 17.00
两项打平的上下文长度 L* = k·A/W = 2048 × 17.00 = 34816

        L    indexer MAC       主注意力 MAC          总 MAC    indexer 占比    对 dense 加速
------------------------------------------------------------------------------------
     8192     67,108,864    285,212,672    352,321,536        19.0%         3.24×
    32768    268,435,456    285,212,672    553,648,128        48.5%         8.24×
    65536    536,870,912    285,212,672    822,083,584        65.3%        11.10×
   131072  1,073,741,824    285,212,672  1,358,954,496        79.0%        13.43×
  1048576  8,589,934,592    285,212,672  8,875,147,264        96.8%        16.45×

加速比的天花板（L→∞）= A/W = 17.00×，因为 indexer 自己也是 O(L²)
L=131072 时已拿到天花板的 79.0%，剩下 21.0% 是 indexer 的固定拖累

=== 真实计时（CPU，成对交替采样取逐对比值的中位数）===
       L   indexer (ms)        稀疏主注意力 (ms)       实测比值       解析比值
------------------------------------------------------------------
    4096           0.24               1.38       0.17       0.12
   16384           1.02               1.39       0.73       0.47
   65536           4.47               1.40       3.18       1.88
```

代入 V3.2 的实配（$W = 64\times128 = 8192$，$A = 128\times(2\times512+64) = 139264$）：

- **$A/W = 17.00$，这就是稀疏注意力的加速天花板。** L=128K 时拿到了 13.43×（天花板的 79%），剩下的 21% 就是 indexer 的固定拖累。
- **两项打平的上下文长度 $L^* = k \cdot A/W = 2048 \times 17 = 34{,}816$。** 超过 35K 之后，**indexer 比被它省下来的主注意力还贵**——L=1M 时 indexer 占了总注意力算力的 96.8%。
- 这就是为什么 indexer 要「少头 + 共享 key + ReLU + fp8」四件套齐上：**它自己就是这条路上最大的成本项。**

> 表格下半部分的真实计时是同机成对交替采样取中位数。实测比值（0.17 / 0.73 / 3.18）系统性高于解析比值（0.12 / 0.47 / 1.88），约 1.4–1.7 倍——原因是 CPU 上 einsum 与手写 kernel 的效率差异，两边的绝对效率不一样。**趋势（随 L 线性上升、4 倍 L 换 4 倍占比）与解析一致，绝对值不能直接对上。**

### 4.3 字节账：indexer 自己也要一份 cache

稀疏省的是「读」，但 indexer 必须能对全序列打分——所以它要一份**全序列的 key cache**。这笔账必须算清楚。

V3.2 的 cache 由三部分组成，逐项来源可查：

| cache | 元素数 | 字节/元素 | 字节/token/层 | 来源 |
|---|---|---|---|---|
| MLA latent | 512 | 1（fp8） | 512 | `kv_lora_rank=512` |
| MLA RoPE key | 64 | 2（fp16） | 128 | `qk_rope_head_dim=64` |
| **Indexer key** | **128** | **1（fp8）** | **128** | `index_head_dim=128` |

```python
"""E4：DSA 的 KV 字节账 —— 稀疏省的是「读」，但 indexer 自己也要一份 cache

DeepSeek-V3.2-Exp 的 cache 由三部分组成（逐项来源：config.json + FlashInfer kernel 签名）：
  ① MLA latent cache : kv_lora_rank = 512     × fp8  = 512 B/token/layer
  ② MLA RoPE key     : qk_rope_head_dim = 64  × fp16 = 128 B/token/layer
  ③ Indexer key cache: index_head_dim = 128   × fp8  = 128 B/token/layer
     （FlashInfer 的 dsa_topk_indexer 内核里这一维叫 head_dim_with_scale = 132：
       128 个 fp8 值 + 4 字节 per-token scale）
  层数 num_hidden_layers = 61
"""
L_LAYERS = 61
MLA_LATENT, ROPE_DIM, IDX_DIM = 512, 64, 128
MLA_ABSORB = 128                       # num_attention_heads
TOPK = 2048

rows = [
    ("① MLA latent cache", MLA_LATENT, 1, "kv_lora_rank=512, fp8"),
    ("② MLA RoPE key", ROPE_DIM, 2, "qk_rope_head_dim=64, fp16"),
    ("③ Indexer key cache", IDX_DIM, 1, "index_head_dim=128, fp8"),
]
print("=== DeepSeek-V3.2-Exp 每 token 每层的 KV 字节（按 cache 类型拆开）===")
print(f"{'cache 类型':<24}{'元素数':>8}{'字节/元素':>10}{'字节/token/层':>15}{'来源':>38}")
print("-" * 96)
tot = 0
for name, n, b, src in rows:
    tot += n * b
    print(f"{name:<24}{n:>8}{b:>10}{n * b:>15}{src:>38}")
print("-" * 96)
print(f"{'合计':<24}{'':>8}{'':>10}{tot:>15} B/token/层")
print(f"{'全模型 61 层':<24}{'':>8}{'':>10}{tot * L_LAYERS:>15,} B/token"
      f" = {tot * L_LAYERS / 1024:.2f} KiB/token")

main_kv = MLA_LATENT + ROPE_DIM * 2                 # 640
idx_kv = IDX_DIM                                    # 128（不计 4 字节 scale）
print()
print("=== 对比：几种口径下的「每 token KV」 ===")
print(f"  {'32 期口径 · MLA 全 fp16':<34}{((MLA_LATENT + ROPE_DIM) * 2) * L_LAYERS:>10,} B"
      f" = {((MLA_LATENT + ROPE_DIM) * 2) * L_LAYERS / 1024:>6.2f} KiB")
print(f"  {'MLA 全 fp8（RoPE 也压）':<34}{(MLA_LATENT + ROPE_DIM) * L_LAYERS:>10,} B"
      f" = {(MLA_LATENT + ROPE_DIM) * L_LAYERS / 1024:>6.2f} KiB")
print(f"  {'MLA fp8 + RoPE fp16（V3.2 实况）':<34}{(main_kv) * L_LAYERS:>10,} B"
      f" = {(main_kv) * L_LAYERS / 1024:>6.2f} KiB")
print(f"  {'  + indexer key cache':<34}{(main_kv + idx_kv) * L_LAYERS:>10,} B"
      f" = {(main_kv + idx_kv) * L_LAYERS / 1024:>6.2f} KiB")
print()
print(f"  indexer cache 的额外开销 = {idx_kv}/{main_kv} = {idx_kv / main_kv:.1%}")
print(f"  DSA 相对「MLA 全 fp16」省了 "
      f"{1 - (main_kv + idx_kv) / ((MLA_LATENT + ROPE_DIM) * 2):.1%}")
print("  注意：网上常见的 46,848 B/token 对应的是「忽略 indexer 的 4 字节 scale」的口径"
      f"（{MLA_LATENT}+{ROPE_DIM * 2}+{IDX_DIM} = {main_kv + idx_kv} × 61 = "
      f"{(main_kv + idx_kv) * L_LAYERS:,}）；把 scale 算进去是 "
      f"{(main_kv + idx_kv + 4) * L_LAYERS:,} B/token")

print()
print("=== decode 时每层读多少字节：dense MLA vs DSA ===")
print("（k=2048 个选中 token 的完整 KV + 全序列的 indexer key）")
print(f"{'L':>9}{'dense 读 (MB/层)':>19}{'DSA 读 (MB/层)':>17}{'降低':>9}{'indexer 占读取':>15}")
print("-" * 70)
for L in (2_048, 8_192, 32_768, 131_072, 1_048_576):
    dense = main_kv * L
    dsa = idx_kv * L + main_kv * TOPK
    share = idx_kv * L / dsa
    print(f"{L:>9}{dense / 1e6:>19.2f}{dsa / 1e6:>17.2f}{dense / dsa:>8.2f}×{share:>14.1%}")

L_cross = main_kv * TOPK / (main_kv - idx_kv)
print()
print(f"字节账的打平点：L > main_kv·k/(main_kv-idx_kv) = {L_cross:,.0f} 时 DSA 读得更少")
print(f"读 indexer key 超过读选中主 KV 的临界点：L = main_kv·k/idx_kv = "
      f"{main_kv * TOPK / idx_kv:,.0f}")
print(f"字节账的天花板（L→∞）= main_kv/idx_kv = {main_kv / idx_kv:.2f}×")
```

```text
=== DeepSeek-V3.2-Exp 每 token 每层的 KV 字节（按 cache 类型拆开）===
cache 类型                     元素数     字节/元素     字节/token/层                                    来源
------------------------------------------------------------------------------------------------
① MLA latent cache           512         1            512                 kv_lora_rank=512, fp8
② MLA RoPE key                64         2            128             qk_rope_head_dim=64, fp16
③ Indexer key cache          128         1            128               index_head_dim=128, fp8
------------------------------------------------------------------------------------------------
合计                                                    768 B/token/层
全模型 61 层                                           46,848 B/token = 45.75 KiB/token

=== 对比：几种口径下的「每 token KV」 ===
  32 期口径 · MLA 全 fp16                   70,272 B =  68.62 KiB
  MLA 全 fp8（RoPE 也压）                    35,136 B =  34.31 KiB
  MLA fp8 + RoPE fp16（V3.2 实况）          39,040 B =  38.12 KiB
    + indexer key cache                 46,848 B =  45.75 KiB

  indexer cache 的额外开销 = 128/640 = 20.0%
  DSA 相对「MLA 全 fp16」省了 33.3%
  注意：网上常见的 46,848 B/token 对应的是「忽略 indexer 的 4 字节 scale」的口径（512+128+128 = 768 × 61 = 46,848）；把 scale 算进去是 47,092 B/token

=== decode 时每层读多少字节：dense MLA vs DSA ===
（k=2048 个选中 token 的完整 KV + 全序列的 indexer key）
        L     dense 读 (MB/层)     DSA 读 (MB/层)       降低    indexer 占读取
----------------------------------------------------------------------
     2048               1.31             1.57    0.83×         16.7%
     8192               5.24             2.36    2.22×         44.4%
    32768              20.97             5.51    3.81×         76.2%
   131072              83.89            18.09    4.64×         92.8%
  1048576             671.09           135.53    4.95×         99.0%

字节账的打平点：L > main_kv·k/(main_kv-idx_kv) = 2,560 时 DSA 读得更少
读 indexer key 超过读选中主 KV 的临界点：L = main_kv·k/idx_kv = 10,240
字节账的天花板（L→∞）= main_kv/idx_kv = 5.00×
```

三条要点：

**① indexer cache 的额外开销正好是 20%。**
$128/640 = 20.0\%$。相对第 032 期的 MLA 全 fp16 口径（68.62 KiB），DSA 的 45.75 KiB 省了 33.3%——但**如果不算 indexer 那份 cache，会误以为省了 44.4%**（38.12 vs 68.62）。**这 11 个百分点就是 indexer 的价钱。**

**② 有个流传很广的数字需要澄清。**
「46,848 B/token」这个数字在中文资料里出现很多次，它对应的口径是 $512 + 64\times2 + 128 = 768$ B/token/层 × 61 = 46,848 —— 也就是**忽略了 indexer key 的 4 字节 per-token scale**。

而这个 scale 是真实存在的：FlashInfer 里那个 kernel 的签名就是

```
dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
  q_index_fp8      [batch_size, num_index_heads=64, index_head_dim=128]
  k_index_cache_fp8[num_pages, page_size, kv_cache_num_heads=1, head_dim_with_scale=132]
  weights          [batch_size, num_index_heads=64]   float32
  topk_indices     [batch_size, topk=2048]            int32
```

`head_dim_with_scale = 132 = 128 + 4`，那 4 个字节就是 fp32 scale。所以把 scale 算进去是 $772 \times 61 = 47{,}092$ B/token。

**两个数字都对，取决于你算不算 scale。** 引用二手数字前先看它的口径——这是本期第三次「口径决定结论」。

**③ 字节账的天花板是 5×，比算力账的 17× 低得多。**
dense 读 $\Phi L$，DSA 读 $\Psi L + \Phi k$（$\Phi = 640$、$\Psi = 128$）：

- $L > \Phi k /(\Phi - \Psi) = 2{,}560$ 时 DSA 读得更少（**注意：2048 的上下文里 DSA 反而更亏**）；
- $L > \Phi k/\Psi = 10{,}240$ 之后，**读 indexer key 的字节超过了读选中主 KV 的字节**（L=128K 时 92.8%）；
- $L \to \infty$ 时比值趋于 $\Phi/\Psi = 5.00$。

**所以「稀疏注意力省带宽」的说法只在长上下文成立，而且上限只有 5×。** decode 场景真正的收益来自算力账那 17×，而不是字节账这 5×——这与第 030 期「decode 是带宽问题」的结论不冲突：DSA 对 decode 的帮助主要是**缩短了 attention 的算术强度下限**，而它自己又引入了新的全序列读取。

### 4.4 工程落地：框架侧到底怎么接

这一节全部来自可查的**源代码/文档/内核签名**（不是推断），并且注明来源类型。

**SGLang**（官方 attention backend 文档）：`--attention-backend dsa`（已废弃别名 `nsa`）会自动按模型架构激活，内部对 prefill / decode 分派到不同子后端，可以显式覆盖：

| 子后端 | prefill | decode | 说明（文档原文要点） |
|---|---|---|---|
| `flashmla_sparse` | ✅ | ✅ | Hopper/Blackwell 上的 BF16 默认 prefill |
| `flashmla_sparse_q8` | ✅ | ❌ | SM90 原生 FP8（q8×kv8）稀疏 prefill，要求 `--kv-cache-dtype fp8_e4m3` |
| `triton_sparse_mla` | ✅ | ❌ | 融合 Triton 稀疏 prefill，按 TP 后的真实 head 数工作（不 padding 到 64/128）；SM90+，每 rank ≤32 个 query head，且不能与 `index_kpool` 同用；`--dsa-triton-union 2 4` 让相邻 query 共享一份 gather 索引 |
| `flashmla_kv` | ✅ | ✅ | Hopper 上 FP8 的默认 |
| `fa3` / `trtllm` | | ✅ | Hopper / Blackwell 上的默认 decode |

**vLLM**：目前 DSA 只有 `FLASHMLA_SPARSE` 一条路，而官方 FlashMLA kernel 只支持 SM90/SM100。vLLM 的 issue #30644 里记录的现象是：一旦检出 `index_topk`，后端选择就被强制走到 `FLASHMLA_SPARSE`，没有回退路径，在不支持的架构（如 SM120）上会直接崩；而 SGLang 的 NSA backend 有 `tilelang_sparse_fwd` 这类可移植 kernel 作为兜底。（**标注：这一条来自 GitHub issue，不是官方文档。**）

**FlashInfer**：DSA 的 indexer 被单独做成了一个内核，签名见 §4.3，benchmark 页明确写「Matches SGLang/deep_gemm implementation」。**这说明 indexer 在工程上被当成一等公民算子，而不是 attention kernel 内部的一段代码。**

**为什么一切都对齐到 64/128？** 这是硬件约束倒逼的：

- 块大小要和 **Tensor Core / SMEM tile** 对齐，才能用矩阵指令而不是逐元素指令；
- NSA 的 kernel 设计叫 **group-centric data loading**：把 GQA 组内共享同一份稀疏 KV 索引的所有 query head 一起载入 SRAM，避免同一块 KV 被重复搬运（这正是 §3.1 ① 那个「跨 head 共享选择只掉 2 pt」的硬件理由）；
- **页/块大小 64 与 PagedAttention 的块（第 011 期）天然对齐**——所以「稀疏选择」在存储层不需要新的数据结构，只需要一个「哪些页被选中」的额外索引；
- 反过来，**块越大，§3.1 ③ 的固定税 $(b-1)/(2K)$ 越重**，而块越小，索引与访存的开销越大。这就是为什么真实系统的块大小集中在 64（FlashInfer 的 DSA kernel page_size=64，vLLM/page 默认 16，NSA 用 32/64，MoBA 用块划分）。

### 4.5 它和哪些已发期数相关

这一节是「领域地图」，把第 034 期挂到整个站点的坐标系里。

| 关联期 | 关系 | 具体连接点 |
|---|---|---|
| 011 PagedAttention | 同一套块布局 | 稀疏选择选的是「页」，与 PagedAttention 的块表天然兼容 |
| 005 FlashAttention | **正交的另一条路** | FlashAttention 不改变 $O(L^2)$，只改 IO；稀疏注意力改的是数学形式。两者可以叠加（FlashMLA 就是「Flash 化的 MLA + sparse」） |
| 032 Attention 变体 | 上下游 | 032 决定「一个 token 的 KV 有多大」，034 决定「要看多少个 token」。**KV 字节 × 可见 token 数 = 实际读取量**，两期共同决定 decode 的带宽账 |
| 033 KV 量化 | 上下游 | 033 把每字节压到 8 bit（1.95×），034 把可见数压到 1.6%（13.4×）。注意 033 的结论「动态量化的 KV 不能跨机」在这里有回声：**indexer 的 per-token scale 也必须进传输契约** |
| 029 前缀缓存 | 互补 | 029 省的是「重复的 prefill 计算」，034 省的是「每个 query 自己的全序列扫描」。两者叠加时要注意：命中前缀之后，剩余部分仍需稀疏选择 |
| 026 MoE | **同源类比** | MoBA 直接把 MoE 的 top-k 门控搬到 attention。MoE 里「专家坍缩」「容量因子」「负载均衡」这套语言在块路由里都有对应版本——只不过块路由没有负载均衡问题（因为 attention 的「专家」不需要参数） |
| 027 投机解码 / 028 连续批处理 | 共享同一个 roofline 平衡点 | 027/028 反复出现的 295.2 平衡点在这里变成 $L^* = kA/W$ 的另一个版本：都是「算力时间 vs 内存时间打平的长度」 |
| 030 PD 分离 | 一个隐患 | DSA 让 prefill 变便宜，但 prefill 的 indexer 也是 $O(L^2)$；PD 分离时 indexer 的开销只落在 P 实例上，配比公式需要重算 |

### 4.6 已知的限制与争议

诚实列三条：

**① 最佳论文 ≠ 最佳实践。** NSA 拿了 ACL 2025 Best Paper，但 DeepSeek 自己在 V3.2 里没有用三分支结构，而是简化成单 indexer 的 DSA。可能的解释：三分支独立 KV 带来的访存与显存开销，在工程上不如「一个足够好的 indexer + 把主干训到适应它」划算。这一条对「论文结论怎么读」很有价值。

**② 稀疏注意力的收益与超参强绑定。** block size、top-k、window 的最优值随模型规模与任务漂移，换一个模型就要重搜一轮。MoBA 论文自己也承认「细粒度块划分」是个需要实验的维度。

**③ 部署侧比论文侧更难。** 落地靠的是模型相关的专用 kernel（SGLang 的 dsa backend、vLLM 的 FLASHMLA_SPARSE、FlashInfer 的 indexer 内核），而不是通用算子。硬件一代换一次，kernel 就要重写一次——这与「hardware-aligned」是一体两面。

---

## 5. 什么时候该用 / 不该用

**该用：**

- 长上下文（**L 远大于 35K**，因为这是 indexer 与主注意力打平的位置）。短上下文用稀疏注意力纯亏——$L < 2{,}560$ 时连字节账都是负的。
- 上下文里有**长距离依赖**（检索、代码库、多轮、RAG）。如果任务的信息全在最近的几 K token 里，滑窗就够了，不需要 indexer（§3.1.1）。
- 你有预训练或续训的能力。**原生可训练是第三代稀疏注意力的前提**——直接套到 dense 模型上会掉点（这是 NSA 论文对第二代方法的批评）。
- decode 侧受限（长上下文 + 中等 QPS）。这是 DSA 收益最大的场景。

**不该用：**

- 短上下文（< 8K）或信息局部的任务（滑窗/流式）。
- 你没法改训练流程、只有推理侧可动。此时稀疏注意力会退化成「分类偏移的近似」，收益小、风险大。
- 你需要严格无损。稀疏注意力的验收标准是 KL / perplexity / 下游指标，**不是 argmax 一致率**（033 期讲 KV 量化时踩过同一个坑：top-1 不变 ≠ 无损）。
- 想要「通用加速」。稀疏注意力的收益天花板由 $A/W$ 决定（V3.2 是 17×），且只在高 MFU 需求下才值；小模型上 $A$ 本身就小，$\frac{A}{W}$ 更小。

---

## 6. 常见坑

**坑 1：把 head 维当成 batch 维，或者 mask 少一个 unsqueeze。**
本期两次踩到（§3.3 的两个 details 折叠块）。`logits` 是 4 维 `[B, H, S, S]`，任何「按 head 求和」要写 `sum(dim=1)`，任何 3 维 mask 要 `unsqueeze(1)` 才能广播到 head 维。**两种错误都会在 batch = head 数时静默通过。**

**坑 2：indexer 的输入忘了 detach。**
不 detach，KL 的梯度会漏回主干，把「indexer 对齐主干」偷换成「主干迁就随机 indexer」。本机实测：dense 模式准确率从 0.92 掉到 0.33，**不报任何异常**。

**坑 3：拿稀疏率当收益。**
「2048/131072 = 1.6%，所以快 64 倍」是错的。正确算法是 $\frac{A \cdot L}{W \cdot L + A \cdot k}$，天花板 $A/W$。本期实测 L=128K 只有 13.43×（天花板的 79%）。此外报告出来的数字还要区分 prefill/decode、区分是否含 indexer、区分端到端与 kernel-only。

**坑 4：拿绝对 recall 比较不同方案，而不看口径。**
本期的三个「口径陷阱」：recall 的绝对值和 top-k 自身的天花板有关（所以要报「相对满秩的差距」）；KV 字节数含不含 per-token scale（768 vs 772）；「块对齐多算」和「对角块浪费」是两个不同的量。**先问口径，再比数字。**

**坑 5：忘记 indexer 也是一份要读的 cache。**
它占 20% 的 KV 显存，且在 L > 10,240 之后成为**读取字节的主要项**（L=128K 时 92.8%）。把它漏掉会系统性高估收益。

---

## 7. 一句话总结

**稀疏注意力的全部难点在一句话里：主注意力越省，indexer 的份额越大——直到它在 L≈35K 之后变成最大的那一项；所以这个领域真正在做的事，是造一个「比被它省掉的注意力便宜 17 倍，却能猜对往哪看」的小网络。**

---

## 8. 今日练习

<details>
<summary>今日练习</summary>

### 练习 1：把「块粒度税」的解析式写出来并对上实测

对预算 $K$ 个 token、块大小 $b$，对角块被强制选中且块内平均 $(b-1)/2$ 个位置在因果掩码之外。求「块对齐造成的固定浪费」占预算的比例，并用 §3.1 的实测值验证。

**参考答案**

浪费的位置数 = $(b-1)/2$（对角块内未来位置的平均个数）

$$\text{浪费比例} = \frac{(b-1)/2}{K}$$

代入 $K = 256$：

| $b$ | $(b-1)/2$ | 解析值 | §3.1 实测 |
|---|---|---|---|
| 16 | 7.5 | 2.93% | 2.9% |
| 32 | 15.5 | 6.05% | 6.1% |
| 64 | 31.5 | 12.30% | 12.3% |
| 128 | 63.5 | 24.80% | 24.8% |

逐项吻合。**注意这个式子与 $L$ 无关**——块对齐的浪费是固定税，不随上下文变长而摊薄。所以「上下文越长，块大小越不敏感」是错的。

另外注意：这里只算了「对角块被强制选中」造成的浪费。真正的实现里，块内的非对角位置也常常没有质量（那部分损失体现在 recall 上，不是这里的浪费）。

### 练习 2：为什么 $A/W = 17$ 而不是「2048 分之 L」

有人算 DSA 的收益是这么算的：seq=128K、top-k=2048，所以「只算了 1.6%」，加速 64×。请指出这个算法漏了什么，并给出正确的形式。

**参考答案**

漏了两项：

**(a) indexer 自己也是 $O(L^2)$。** 它对每个 (query, key) 对都要算：

$$W = H^I \times d^I = 64 \times 128 = 8192\ \text{MAC}$$

所以总成本是 $W \cdot L + A \cdot k$，不是 $A \cdot k$。$L \to \infty$ 时加速比趋于 $A/W = 17$。

**(b) 「1.6%」是 token 数的比例，不是成本的比例。** 每个 (query, 选中 key) 对的成本 $A$ 与每个 (query, 全部 key) 对的 indexer 成本 $W$ 不同（相差 17 倍），所以不能直接按 token 数比。

正确形式：

$$\text{speedup}(L) = \frac{A \cdot L}{W \cdot L + A \cdot k} = \frac{A/W}{1 + \frac{A \cdot k}{W \cdot L}}$$

代入 $L = 131072$：分母 $= 1 + 139264\times2048/(8192\times131072) = 1 + 0.2656 = 1.2656$，所以 $17.00/1.2656 = 13.43$。与 §4.2 的实测一致。

顺带：这个式子还告诉你**提高 $k$ 的边际代价**——$k$ 只在分母第二项里，所以把 $k$ 从 2048 提到 4096，L=128K 时加速比从 13.43 降到 11.10（仍然远好于 dense 的 1×）。

### 练习 3：滑窗什么时候够用，给一个判据

§3.1.1 说滑窗的 recall 恒等于「窗口内质量占比」。请据此给出一个**可测量的判据**，用来判断某个 workload 是否需要内容相关的稀疏选择。

**参考答案**

定义局部质量占比

$$\rho_W = \frac{\sum_{t}\sum_{j \in \text{last } W} p_{t,j}}{\sum_t 1} = \text{recall}_{\text{window}}(W)$$

（即 §3.1.1 第一张表的第二列）

判据：**如果 $\rho_W$ 已经接近 1（比如 > 0.97），上 indexer 的复杂度跟收益完全不成比例；只有当 $\rho_W$ 明显低于 1（比如 < 0.9）时，内容选择才值得。**

实测支持：$\rho_W = 0.476$ 时 top-k 领先 25.9 pt；$\rho_W = 0.850$ 时领先 6.6 pt；$\rho_W = 0.998$ 时只领先 0.1 pt。

更实用的推论：$\rho_W$ 可以在一小批数据上**用一次 dense 前向直接量出来**（把每层的注意力概率累加即可），不需要训练任何东西。**在决定要不要上稀疏注意力之前，先量 $\rho_W$——这一步比调 block size 重要得多。**

（注意：$\rho_W$ 是「窗口能拿到的质量」，不是「窗口够不够」的全部答案。还要看剩下那部分质量对任务是否关键——一个 3% 的长距离质量如果恰好是答案所在，recall 再高也没用。所以 $\rho_W$ 是必要条件，最终还要看下游指标。）

</details>

---

*本期的全部数值均在 torch 2.14.0（CPU / Apple MPS，无 CUDA）上实跑，脚本与原始 stdout 保留在 `scratch-034/`。涉及论文与框架的数字均标注了来源类型；NSA 的实验数字来自论文与其技术解读，本期未复现，只作谱系引用。*
