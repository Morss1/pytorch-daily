# PyTorch 每日一课 · 第 029 期

## RadixAttention 与前缀缓存：把 KV Cache 变成一棵可以共享的树

> **日期**：2026-09-30　**难度**：⭐⭐⭐⭐　**预计阅读**：32 分钟
>
> **前置知识**：attention 的因果性；KV cache 的结构与显存占用；第 011 期 PagedAttention 的「块表」概念；第 027 期 roofline 口径（7B BF16 / H800 级 / 权重 14 GB / 3.35 TB/s / 989 TFLOPS / 平衡点 295）
>
> **关联**：第 011 期 · 它解决的是「显存怎么放」，本期解决「算过的东西怎么不重算」，两者共用同一套块表；第 028 期 · 那一期讲过调度器里其实没有 prefill/decode 阶段之分，只有 `num_computed_tokens` 追 `num_tokens_with_spec` —— 前缀命中就是把后者的起点从 0 改成命中长度；第 027 期 · 同一个 295 平衡点在这里决定「命中之后时间还能不能继续降」
>
> **一句话预告**：前缀缓存能让 prefill 的**算力**降到 6%，但**墙钟时间**降不到 6% —— 因为权重总得读一遍。

---

## 1. 结论先行

先摆一张在一个混合工作负载上实测的表（2,160 条请求、152 万 prompt token、四种共享模式混在一起，代码在第 4 节）：

| 问题 | 数字 |
|---|---|
| 流过缓存的 prompt token 总量 | 1,523,820 |
| 去重之后**真实存在的内容**只有 | 277,305 token（33.9 GiB KV） |
| 也就是说平均每个 token 被复用 | **5.50 次** |
| 前缀命中率（基数树，无限容量） | **93.63%** |
| prefill 算力降到基线的 | **6.34%**（15.8×） |
| 但同一批请求的 prefill **时间**只降到 | 15.8× 里的一部分 —— 看下面 |
| 一条 RAG 请求（1590 token prompt） | 算力 16.4×、时间只 12.2× |
| 树管理开销（纯 Python 实现）占无缓存 prefill 时间 | **0.31%** |
| 容量掉到去重总量的 1/5 才明显掉命中率；压力下 LPM 调度把重算 token | **÷1.91** |

三个反直觉的点，本文会逐个给出量化的来源：

1. **命中率就是「省下的算力比例」，这是一个精确等式**（不是近似、与模型大小无关）。但**时间是 `max(算力时间, 权重读取时间)`**，命中率超过一定值之后，时间被地板卡住不再下降。
2. **缓存占用的是「去重后的内容量」，不是「请求量」**。152 万 token 的请求量，去重后只有 27.7 万 token —— 这才是要预留的显存。
3. **缓存和 batch 争同一块显存**。SGLang 论文里有一句很直白的话：等请求多起来，系统会「驱逐掉所有缓存 token，换取更大的 batch size」。（这句话我在论文原文里核对到了，见 5.5 节。）

---

## 2. 这个领域解决什么问题

### 2.1 因果注意力给了一个免费的承诺

decoder-only 的 attention 里，第 $i$ 个位置的输出只依赖位置 $\le i$ 的输入。写成公式：

$$
K_{1:i} = W_k\, h_{1:i}, \qquad V_{1:i} = W_v\, h_{1:i}, \qquad o_i = \mathrm{softmax}\!\left(\frac{q_i K_{1:i}^\top}{\sqrt{d}}\right) V_{1:i}
$$

关键在于 $K_{1:i}$ 和 $V_{1:i}$ 这两个张量**只由前缀决定**，跟后面要问什么完全无关。所以：

> **只要两个请求的前缀逐 token 相同，它们前面的 KV 就是同一份张量，可以算一次、用无数次。**

这句话听起来像一个纯粹的工程技巧，但它有一个很强的性质：**它不改变输出**。第 4 节我会用一个真实的小模型跑出「偏差 = 0.000e+00」来证明这一点 —— 不是「近似相等」，是逐位相等。

### 2.2 现在的负载里，前缀重复到什么程度

2023 年之前，推理服务的主要形态是「一次性问答」，请求之间没有共享。而现在的主流形态是 **LM program**（SGLang 论文里的说法）：一个程序里有多次、带依赖的 LLM 调用。它天然产生大量共享前缀：

| 形态 | 共享什么 | 典型占比 |
|---|---|---|
| 多轮对话 | system prompt + 整个历史 | 第 8 轮时，本轮新增可能只占全部 prompt 的 10% |
| RAG | 同一篇长文档被反复提问 | 指令 + 文档 1530 token 完全重复 |
| few-shot / agent | 示例块、工具定义、思维链模板 | 全是重复的 |
| 并行采样 / 树搜索 | 同一 prompt 的分叉 | 100% 重复 |

**数字动机**：一条 4096 token 的 prompt，在 7B（BF16、H800 级、MFU 45%）上跑一次 prefill 要 **143.8 ms**。如果这段 prompt 里 96.9% 已经算过，剩下的算力时间只有 4.42 ms。这不是优化 10%、20%，是优化一个数量级。

### 2.3 为什么早期框架不做

因为要做对，需要同时解决四件事，缺一个就做不成：

1. **匹配粒度**：前缀边界落在哪？按块？按 token？
2. **数据结构**：多个请求的前缀互相交叉，怎么表示「A 和 B 共享前 1000 token，B 和 C 又共享另一个 800 token」这种多级共享？
3. **驱逐**：显存满了先扔谁？扔错了会让「还挂在树上的后代」变成永远命中不了的垃圾。
4. **并发**：正在跑的请求正在引用某段 KV，这段不能扔。

本节剩下的部分讲前两条，第 3 节讲后两条。

---

## 3. 核心思想

### 3.1 把 KV cache 看成文件系统的路径

这是理解 RadixAttention 最省力的类比：

- **每个 token 是路径上的一个字符**；
- **一段 KV 是一条路径**；
- **多个请求共享前缀 = 多条路径共享同一段路**；
- **缓存就是这棵路径树**，节点上挂着对应的 KV 页。

区别只在于：文件系统的路径是给人看的，而这里的「路径」长度动辄几千，且每个「字符」是 0~15 万的整数（token id）。

用「路径树」而不是「哈希表」的直接收益是**多级共享**。举一个具体的例子（第 4 节的代码会实际跑出这个结果）：

```
请求 1 插入： [A(10 token)] [100] [20..24]          共 16 token
请求 2 插入： [A(10 token)] [100] [30..34]          共 16 token  ← 与请求 1 共享 11 token
请求 3 插入： [A(10 token)] [101] [40..44]          共 16 token  ← 只与它们共享 10 token

插入完成后，树长这样（3 次插入、2 次分裂、6 个节点、27 个 token）：

              root
               │  A = 10 token
               ▼
        ┌─ A ──────────────┐
        │                   │ 101
   100  ▼                   ▼
   ┌─ [100] ─┐        [101, 40..44]
   │         │              (6)
  20..24   30..34
   (5)       (5)
```

**注意「分裂」这一步**：请求 1 刚进来时，树里只有一条 16 token 的边。请求 2 进来时匹配到第 11 个 token 处开始不一致 —— 于是把原来那条边从中间切开，前半段 11 token 变成新的父节点，后半段 5 token 变成它的子节点。这就是「基数树」（radix tree）相对普通 trie 的压缩：**边上挂的是任意长度的 token 序列，不是单个 token**。好处是节点数远小于 token 数；代价是匹配到一半可能要切一刀。

**一次真实的分裂长什么样**（第 4 节的校验输出）：

```
三次插入后：节点数 6，缓存 token 27，分裂 2 次
  请求 2 的前缀命中 = 16 token
  请求 3 的前缀命中 = 16 token
  一条全新请求的命中 = 0 token
```

### 3.2 为什么哈希表也能做，但会不一样

vLLM 选的是另一条路：**块哈希链**。它把 16 个 token 打包成一个块，块的 key 定义为

$$
\text{key}_b = H\big(\text{key}_{b-1},\ \text{tokens}[16b:16b+16]\big)
$$

把父块的哈希串进来，效果是**「前缀相同」在哈希层面就等价于「key 相同」**，于是查找退化成一次字典查询，不用走树。vLLM 的官方设计文档里画得很清楚：

```
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

两条路线的差别可以列成表（前一半是结构差异，后一半是工程后果）：

| | vLLM 块哈希链 | SGLang 基数树 |
|---|---|---|
| 匹配粒度 | 块（`BlockSize` 默认 16） | token（`page_size` 默认为 None，按模型自动选） |
| 数据结构 | dict：块 key → 物理块 | 树：节点 → 子节点字典 |
| 多级共享 | 靠链式哈希隐式表达 | 显式表达（分叉） |
| 命中边界 | 必须落在块边界，且**只缓存满块** | 可以落在任意 token |
| 驱逐顺序 | 空闲队列 LRU（同一请求的块按**倒序**入队，越靠后的越先被驱逐） | LRU **叶子优先** |
| 维护开销 | 每块一次哈希（默认 sha256） | 一次逐 token 比较，复杂度与命中长度线性 |
| 中间节点能否被驱逐 | 可以（但会留下够不到的「僵尸块」） | **不可以**，结构上禁止 |
| 命中是否要求前缀路径完整 | 要求（链断一块，后面全废） | 不要求（可以匹配到任意深度） |
| 超时/隔离 | `cache_salt` 混进第一个块的哈希 | `RadixKey` 的 `extra_key`：leading token 相同但 extra_key 不同的条目被刻意放在互不相交的子树 |

「块粒度 vs token 粒度」的实际差距有多大？第 4 节的代码在同一段公共前缀上做了对照：

```
同一段公共前缀，两种粒度各能命中多少 token：
  公共前缀     8 token → 树命中     8，块命中     0，少   8 token（100.0%）
  公共前缀    16 token → 树命中    16，块命中    16，少   0 token（  0.0%）
  公共前缀    24 token → 树命中    24，块命中    16，少   8 token（ 33.3%）
  公共前缀    31 token → 树命中    31，块命中    16，少  15 token（ 48.4%）
  公共前缀    48 token → 树命中    48，块命中    48，少   0 token（  0.0%）
  公共前缀   200 token → 树命中   200，块命中   192，少   8 token（  4.0%）
  公共前缀  1500 token → 树命中  1500，块命中  1488，少  12 token（  0.8%）
```

规律很清楚：**绝对损失恒为 $\text{命中长度} \bmod 16$，小于 16 个 token，与命中长度本身无关；相对损失则是 $\frac{15}{L}$ 量级。** 所以：

- 长前缀负载（RAG 的 1500 token 文档）：损失 < 1%，**块粒度完全够用**；
- 短前缀负载（只有 24 token 的公共 system prompt）：损失 33%，**块粒度直接废掉三分之一**。

这不是「谁更好」的问题，是「块大小该配多大」的问题。第 9 节的练习会把这条曲线完整跑一遍 —— 结论比直觉更有意思。

### 3.3 唯一的硬约束：只有叶子能被驱逐

这是树结构带来的一条**结构性约束**，也是我认为整个设计里最值得记住的一点。

考虑树里的内部节点 $u$，它有一段 KV，然后有子节点 $v$。$v$ 的 KV 是在 $u$ 的 KV 之后算出来的（$v$ 的每个 token 的 attention 都要看 $u$ 那一段）。所以：

> **扔了 $u$ 的 KV，$v$ 的 KV 就永远不可用了** —— 不是「命中率下降」，是物理上不可用，因为没法只拿着后半截去做 attention。

于是驱逐必须从叶子做起：扔一个叶子，然后看它的父节点是不是变成叶子了，是就继续候选。这就是论文里那句 "evicts the least recently used leaf first. By evicting leaves first, we enable the re-use of their common ancestors until those ancestors become leaves and are also evicted."（原文，我核对了 PDF。）

vLLM 那边的对应做法很巧妙：请求结束时，它把该请求的块**按倒序**追加到空闲队列尾部。倒序意味着「最后一块」（哈希了最多 token、复用可能性最低）排在更前面，于是**先被驱逐**。用一句不太严谨但直观的话说：两条路线各自从自己的结构出发，都收敛到了同一个原则 —— **优先驱逐「更深的、可复用性更差的」那一段**。

**这里有一个我踩到的真 bug，值得单说。** 写基数树的 `_split` 时，第一版我是这么写的：

```python
def _split(self, node, m):
    head = Node(node.key[:m], parent=node.parent)
    head.children[node.key[0]] = node
    self.leaves.discard(node)      # ← 这一行是错的
    self.leaves.add(head)          # ← 这一行也是错的
    return head
```

看起来像「原节点的叶子身份转移给了新父节点」。但这是错的，原因很直白：

- `head` 恰恰**有**一个子节点（就是原 `node`），所以 `head` 不是叶子，永远不该进驱逐候选；
- 原 `node` 的叶子身份**完全没变** —— 它还是没孩子。把它从 `leaves` 里摘掉，等于**永久取消它被驱逐的资格**。

后果不是崩溃，是**显存只增不减**：这棵树会慢慢攒下一堆谁也赶不走的节点。我加了 `audit()` 自检（断言 `leaves` 集合与真实叶子集合完全相等）之后才发现。这也是真实实现里必须有的不变量检查 —— 第 4 节的代码块里保留了这段自检。

### 3.4 引用计数：正在跑的请求把路径锁住

树上的每个节点带一个 `lock_ref`：

- 一条请求被调度时，从根到命中末端这条路**每个节点** `+1`；
- 请求结束，路径上 `-1`；
- `lock_ref > 0` 的节点**不可驱逐**。

没有这个计数器，正在跑 decode 的请求会在下一轮 eviction 里被抽掉脚下的 KV 页，输出直接变垃圾。

还有一个容易忽略的设计选择，论文原文：

> "Note that we do not preallocate a fixed-size memory pool as a cache. Instead, we let the cached tokens and the currently running requests share the same memory pool. Therefore, the system dynamically allocates memory for cache and running requests. When enough waiting requests run, the system will evict all cached tokens in favor of a larger batch size."

这段话说出了一件很反直觉的事：**缓存不是「免费的额外空间」，它是从 batch size 里抠出来的**。上下文长度相同时，多缓存 10 万个 token 的前缀，就意味着 decode 时少放 10 万个 token 的并发请求。第 5.5 节会给这个数字。

### 3.5 缓存驻留量 = 去重后的内容量

这是我在跑完模拟后回头才想明白的一个量。直觉上会觉得「152 万 token 的请求量，缓存得有 152 万 token 吧」，实际不是：

| 量 | 数值 |
|---|---|
| 流过缓存的 prompt token 总量 | 1,523,820 |
| 去重后的**真实内容**（无限容量下的峰值缓存） | **277,305** |
| 比值 | **5.50×** |

原因很简单：**第二个请求不新增内容**。同一篇文档被问 40 次，缓存里只有那一份。所以：

> **要预留的显存 = 去重后的内容量，不是请求量；命中率越高，这个比值越大。**

277,305 token 换算成显存（7B / 32 层 / GQA 8 / head_dim 128 / bf16，$2 \times 32 \times 8 \times 128 \times 2 = 128$ KiB/token）：

$$
277{,}305 \times 128\ \text{KiB} = 33.9\ \text{GiB}
$$

一张 80 GB 卡、留 60 GB 给 KV，容量上限是 491,520 token。也就是说 **这个负载的全部去重内容要吃掉 KV 池的 56%**。缓存容量的规划从一开始就不是「要不要设上限」的问题，是「上限设多少」的问题。

---

## 4. 在 PyTorch 中怎么用

### 4.1 先证明它不改输出

这是最该先跑的一步。下面这段是自包含的：搭一个 4 层的小 transformer，然后把同一个 96 token 的序列用两种方式算一遍 —— A 从头算全量，B 先算前 64 token 留下 KV、再只算后 32 token、把 KV 拼起来。

```python
"""文章代码块 1：验证「复用前缀 KV」与「从头算全量」逐位相等"""
import math

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

torch.manual_seed(0)
D, H, NL, VOCAB, SEQ, PREFIX = 64, 4, 4, 256, 96, 64
DH = D // H


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(D, D, bias=False) for _ in range(4))
        self.w1 = nn.Linear(D, 4 * D, bias=False)
        self.w3 = nn.Linear(D, 4 * D, bias=False)
        self.w2 = nn.Linear(4 * D, D, bias=False)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)

    def layer(self, blk, x, past):
        """past = (k, v)，形状 [B, H, Tp, DH]；返回 (新 x, 本层的 k, v)"""
        B, T, _ = x.shape
        ln = blk.ln1(x)
        q = blk.wq(ln).view(B, T, H, DH).transpose(1, 2)
        k = blk.wk(ln).view(B, T, H, DH).transpose(1, 2)
        v = blk.wv(ln).view(B, T, H, DH).transpose(1, 2)
        if past is not None:                       # ← 前缀缓存就发生在这一行
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        Tk = k.shape[2]
        pos_q = torch.arange(T).view(T, 1) + (Tk - T)   # query 的绝对位置 = 偏移 + 局部下标
        pos_k = torch.arange(Tk).view(1, Tk)
        att = (q @ k.transpose(-1, -2) / math.sqrt(DH)).masked_fill(pos_k > pos_q, float("-inf"))
        x = x + blk.wo((att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D))
        h = blk.w1(blk.ln2(x)) * torch.nn.functional.silu(blk.w3(blk.ln2(x)))
        return x + blk.w2(h), k, v

    def forward(self, idx, past=None):
        x, kvs = self.emb(idx), []
        for i, blk in enumerate(self.blocks):
            x, k, v = self.layer(blk, x, None if past is None else past[i])
            kvs.append((k, v))
        return self.lnf(x) @ self.head.weight.T, kvs


m = Tiny().eval()
tokens = torch.randint(0, VOCAB, (1, SEQ))

with torch.no_grad():
    logits_full, _ = m(tokens)                              # 走法 A：从头算全量
    _, past = m(tokens[:, :PREFIX])                         # 走法 B 第 1 趟：算前缀，留下 KV
    logits_hit, _ = m(tokens[:, PREFIX:], past=past)        # 走法 B 第 2 趟：只算后缀

diff = (logits_full[:, PREFIX:] - logits_hit).abs().max().item()
print(f"全量 prefill  vs  前缀复用后只算后缀：最大 logits 偏差 = {diff:.3e}")
print(f"（logits 平均绝对值 {logits_full.abs().mean().item():.4f}，bf16 的 eps 是 {2.0**-8:.3e}）")


def count(fn):
    c = FlopCounterMode(display=False)
    with c:
        fn()
    return c.get_total_flops()


with torch.no_grad():
    f_full = count(lambda: m(tokens))
    f_hit = count(lambda: m(tokens[:, PREFIX:], past=past))
print(f"\n单条请求：全量 {f_full:,} FLOPs → 命中后 {f_hit:,} FLOPs，"
      f"降到 {f_hit/f_full*100:.1f}%")
```

真实输出（managed venv 里的 torch 2.14.0，CPU）：

```text
全量 prefill  vs  前缀复用后只算后缀：最大 logits 偏差 = 0.000e+00
（logits 平均绝对值 0.4588，bf16 的 eps 是 3.906e-03）

单条请求：全量 62,914,560 FLOPs → 命中后 20,971,520 FLOPs，降到 33.3%
```

两件事值得停下来看一眼：

1. **偏差是 0.000e+00，不是 1e-7**。因为走法 B 根本不是在「近似」走法 A：`cat` 出来的 $K$ 张量和走法 A 里的 $K_{1:96}$ 是同一个张量的同一段内存布局，后面的 matmul 输入完全一致，浮点运算顺序也一致。**前缀缓存是一项不改变数值的优化** —— 这也是 vLLM 官方文档敢写 "prefix caching is almost a free lunch and won't change model outputs" 的原因。
2. **33.3% = 32/96，正好是「后缀占比」，也就是「1 − 命中率」。** 这不是巧合，下一节会给出它的一般形式。

### 4.2 基数树的完整实现

下面这段是自包含可跑的：匹配、分裂、叶子优先 LRU 驱逐、引用计数、结构自检，全在里面。为了效率我把 LRU 做成了**惰性删除堆**（`ts` 变了就压新条目，弹出时校验），否则每次驱逐扫一遍叶子集合在真实规模下会退化成 $O(n)$ 每次。

```python
"""文章代码块 2：一个能跑的基数树 KV 缓存（匹配 / 分裂 / 叶子优先 LRU）"""
import heapq
import random


class Node:
    __slots__ = ("key", "children", "lock_ref", "ts", "parent")

    def __init__(self, key, parent=None, ts=0):
        self.key = key            # 这条边上挂的 token 序列，长度可变（基数树相对 trie 的压缩）
        self.children = {}        # 首 token -> Node
        self.lock_ref = 0         # 有多少条正在跑的请求引用它
        self.ts = ts              # 最近访问时间戳，LRU 用
        self.parent = parent


class RadixCache:
    def __init__(self, capacity=None):
        self.root = Node([])
        self.capacity = capacity
        self.size = 0                 # 已缓存 token 数
        self.clock = 0
        self.leaves = set()           # 可驱逐候选：只有叶子能走
        self.heap = []
        self.splits = 0
        self.evicted = 0

    def _touch(self, node):
        node.ts = self.clock
        if node in self.leaves:
            heapq.heappush(self.heap, (node.ts, id(node), node))

    def match_prefix(self, tokens):
        """返回最长已缓存前缀的长度 —— token 粒度，可以落在任何位置"""
        self.clock += 1
        node, pos, trace = self.root, 0, [self.root]
        while pos < len(tokens):
            child = node.children.get(tokens[pos])
            if child is None:
                break
            k = child.key
            m = 0
            while m < len(k) and pos + m < len(tokens) and tokens[pos + m] == k[m]:
                m += 1
            if m < len(k):                      # 命中落在一段中间 → 分裂
                node = self._split(child, m)
                trace.append(node)
                pos += m
                break
            node, pos = child, pos + m
            trace.append(node)
        for n in trace:
            self._touch(n)
        return pos

    def _split(self, node, m):
        """把 node 的前 m 个 token 切出来当父节点，原 node 变成它的子节点。"""
        parent = node.parent
        head = Node(node.key[:m], parent=parent, ts=node.ts)
        parent.children[head.key[0]] = head
        node.key = node.key[m:]
        node.parent = head
        head.children[node.key[0]] = node
        # head 一定有一个子节点 → 它不是叶子；原 node 的叶子身份不变。
        # 这一行如果顺手写成「把原 node 从 leaves 里摘掉」，它就会永久失去被驱逐的资格，
        # 表现是显存只增不减 —— 一个很安静的泄漏。
        self.splits += 1
        return head

    def insert(self, tokens):
        self.clock += 1
        node, pos = self.root, 0
        while pos < len(tokens):
            child = node.children.get(tokens[pos])
            if child is None:
                new = Node(tokens[pos:], parent=node, ts=self.clock)
                node.children[tokens[pos]] = new
                self.leaves.discard(node)
                self.leaves.add(new)
                heapq.heappush(self.heap, (new.ts, id(new), new))
                self.size += len(new.key)
                break
            k, m = child.key, 0
            while m < len(k) and pos + m < len(tokens) and k[m] == tokens[pos + m]:
                m += 1
            if m < len(k):
                head = self._split(child, m)
                pos += m
                if pos < len(tokens):
                    new = Node(tokens[pos:], parent=head, ts=self.clock)
                    head.children[tokens[pos]] = new
                    self.leaves.add(new)
                    heapq.heappush(self.heap, (new.ts, id(new), new))
                    self.size += len(new.key)
                break
            node, pos = child, pos + m
        while self.capacity is not None and self.size > self.capacity and self._evict_one():
            pass

    def _evict_one(self):
        while self.heap:
            ts, _, node = self.heap[0]
            if node.ts != ts or node not in self.leaves or node.children or node.lock_ref:
                heapq.heappop(self.heap)      # 过期条目，惰性删除
                continue
            heapq.heappop(self.heap)
            self.size -= len(node.key)
            self.evicted += len(node.key)
            self.leaves.discard(node)
            p = node.parent
            del p.children[node.key[0]]
            if not p.children and p is not self.root:   # 父节点变成新叶子，进入候选
                self.leaves.add(p)
                heapq.heappush(self.heap, (p.ts, id(p), p))
            return True
        return False

    def zombies(self):
        """树结构不会产生僵尸节点：驱逐只能从叶子做起，父节点要么带着子树一起留着，要么等子树走光后自己变成叶子"""
        return 0

    def audit(self):
        """自检：leaves 集合必须与实际叶子完全一致"""
        real, stack, cnt = [], [self.root], 0
        while stack:
            n = stack.pop()
            cnt += 1
            if not n.children and n is not self.root:
                real.append(n)
            stack.extend(n.children.values())
        assert set(real) == self.leaves, (len(real), len(self.leaves))
        return cnt, len(real)


def brute_force(all_seqs, probe):
    best = 0
    for s in all_seqs:
        m = 0
        while m < min(len(s), len(probe)) and s[m] == probe[m]:
            m += 1
        best = max(best, m)
    return best


rnd = random.Random(7)
ok, total_splits = 0, 0
for _ in range(200):
    seqs, cache = [], RadixCache()
    for _ in range(rnd.randint(1, 12)):
        s = [rnd.randrange(40) for _ in range(rnd.randint(1, 30))]
        seqs.append(s)
        cache.insert(s)
        probe = [rnd.randrange(40) for _ in range(rnd.randint(1, 30))]
        assert cache.match_prefix(probe) == brute_force(seqs, probe)
        ok += 1
    cache.audit()
    total_splits += cache.splits
print(f"match_prefix 与暴力最长公共前缀对照 {ok} 次全部一致"
      f"（累计 {total_splits} 次节点分裂，leaves 集合与真实叶子完全吻合）")

# 看一次分裂：插入「示例块 / 分隔符 / 问题」的三条请求
cache = RadixCache()
A = list(range(10))                         # 假装是一段示例文本
cache.insert(A + [100] + list(range(20, 25)))     # 请求 1
cache.insert(A + [100] + list(range(30, 35)))     # 请求 2：共享 A+[100]
cache.insert(A + [101] + list(range(40, 45)))     # 请求 3：只共享 A
print(f"三次插入后：节点数 {cache.audit()[0]}，缓存 token {cache.size}，分裂 {cache.splits} 次")
print(f"  请求 2 的前缀命中 = {cache.match_prefix(A + [100] + list(range(30, 35)))} token")
print(f"  请求 3 的前缀命中 = {cache.match_prefix(A + [101] + list(range(40, 45)))} token")
print(f"  一条全新请求的命中 = {cache.match_prefix([999, 998, 997])} token")
```

真实输出：

```text
match_prefix 与暴力最长公共前缀对照 1242 次全部一致（累计 191 次节点分裂，leaves 集合与真实叶子完全吻合）
三次插入后：节点数 6，缓存 token 27，分裂 2 次
  请求 2 的前缀命中 = 16 token
  请求 3 的前缀命中 = 16 token
  一条全新请求的命中 = 0 token
```

**校验方法值得单独说一句**：这类「重排/索引」逻辑（匹配、分裂、驱逐）最讨厌的地方在于 **错了不报错，只出错值**。所以我的做法是拿一个 $O(n \cdot m)$ 的暴力最长公共前缀当正确答案，用 1242 组随机输入硬碰；再额外断言 `leaves` 集合与遍历出来的真实叶子集合**完全相同**（这是上面那个 bug 能暴露出来的唯一原因）。随机输入特意用小词表（`randrange(40)`）来制造大量巧合共享和中间分裂 —— 大词表下分裂几乎不发生，测不出问题。

### 4.3 块哈希链的对照实现

```python
"""文章代码块 3：vLLM 式的块哈希链缓存（对照实现）"""
from collections import OrderedDict

BLOCK = 16                       # vLLM CacheConfig.DEFAULT_BLOCK_SIZE = 16


class BlockHashCache:
    """块粒度、哈希链、只缓存满块、LRU 驱逐。

    key_b = (key_{b-1}, 本块 16 个 token)
    —— 把父块的哈希串进来，是为了让「前缀相同」在哈希层面就等价，
       代价是命中必须从第 0 块起连续，中间断一块后面全作废。
    """

    def __init__(self, capacity_tokens=None):
        self.capacity = capacity_tokens
        self.hashes = {}
        self.parent_of = {}
        self.free = OrderedDict()      # 头部 = 最先被驱逐
        self.size = 0
        self.evicted = 0

    def _keys(self, tokens):
        keys, parent = [], None
        for b in range(0, len(tokens) - BLOCK + 1, BLOCK):     # 不满的尾巴不进缓存
            k = (parent, tuple(tokens[b:b + BLOCK]))
            keys.append((k, parent))
            parent = k
        return keys

    def match_prefix(self, tokens):
        hit, parent = 0, None
        for b in range(0, len(tokens) - BLOCK + 1, BLOCK):
            k = (parent, tuple(tokens[b:b + BLOCK]))
            if k not in self.hashes:
                break                          # 链断 → 后面即使还在也不能用
            hit += BLOCK
            self.free.pop(k, None)
            self.free[k] = None                # 移到尾部 = 最近使用
            parent = k
        return hit

    def insert(self, tokens):
        for k, parent in self._keys(tokens):
            if k in self.hashes:
                continue
            self.hashes[k] = BLOCK
            self.parent_of[k] = parent
            self.free[k] = None
            self.size += BLOCK
        while self.capacity is not None and self.size > self.capacity and self.free:
            k, _ = self.free.popitem(last=False)
            self.hashes.pop(k, None)
            self.size -= BLOCK
            self.evicted += BLOCK

    def zombies(self):
        """父块被驱逐、自己还留着的块：占着显存，但永远不可能被命中"""
        return sum(BLOCK for k in self.hashes
                   if self.parent_of.get(k) is not None and self.parent_of[k] not in self.hashes)


# 第一条请求的 prompt 就是这段公共前缀；后续请求共享它、尾部各不相同
print("同一段公共前缀，两种粒度各能命中多少 token：")
for L in [8, 16, 24, 31, 48, 200, 1500]:
    prefix = list(range(L))
    tree, blk = RadixCache(), BlockHashCache()
    tree.insert(prefix)
    blk.insert(prefix)
    probe = prefix + [777, 778, 779]          # 后续请求：共享前缀 + 自己的尾巴
    a, b = tree.match_prefix(probe), blk.match_prefix(probe)
    print(f"  公共前缀 {L:>5} token → 树命中 {a:>5}，块命中 {b:>5}，"
          f"少 {a-b:>3} token（{(a-b)/a*100 if a else 100.0:>5.1f}%）")
```

（这段沿用上一段的 `RadixCache`。）真实输出：

```text
同一段公共前缀，两种粒度各能命中多少 token：
  公共前缀     8 token → 树命中     8，块命中     0，少   8 token（100.0%）
  公共前缀    16 token → 树命中    16，块命中    16，少   0 token（  0.0%）
  公共前缀    24 token → 树命中    24，块命中    16，少   8 token（ 33.3%）
  公共前缀    31 token → 树命中    31，块命中    16，少  15 token（ 48.4%）
  公共前缀    48 token → 树命中    48，块命中    48，少   0 token（  0.0%）
  公共前缀   200 token → 树命中   200，块命中   192，少   8 token（  4.0%）
  公共前缀  1500 token → 树命中  1500，块命中  1488，少  12 token（  0.8%）
```

顺便注意 `zombies()` 这个方法的存在意义：链式哈希的好处是查找 $O(1)$，代价是**它允许一部分块变得「够不到」** —— 父块被驱逐了，子块还留在字典里，占着显存，但任何请求来查都会在第一块就断链，永远命中不了。树结构天然不会有这个问题（3.3 节），代码里 `RadixCache.zombies()` 直接返回 0。这个差距在实测里有多大？看下一节。

### 4.4 在混合工作负载上对照

现在把两种缓存放进同一个 2,160 条请求的工作负载。四种共享模式混在一起，按到达顺序（同一会话的轮次保持因果先后）处理：

| 负载 | 规模 | 共享结构 |
|---|---|---|
| 多轮对话 | 120 会话 × 8 轮 | system(80) + persona(40)，每轮 user 36 + assistant 120 |
| RAG | 10 篇文档 × 1500 token，400 次提问 | 指令 30 + 文档 1500 + 问题 60 |
| few-shot | 300 次提问 | 605 token 公共示例块（5 个子示例各 120 token + 分隔符，制造多级共享） |
| 短前缀群 | 500 次短提问 | 只有 24 token 的公共 system prompt |

```python
"""文章代码块 4：在混合工作负载上对照（沿用上面两个缓存类）"""
import random
import time

RND = random.Random(20260930)
VOCAB = 100_000


class NullCache:
    size = 0
    evicted = 0

    def match_prefix(self, t):
        return 0

    def insert(self, t):
        pass

    def zombies(self):
        return 0


def hit_of(cache, prompt):
    r = cache.match_prefix(prompt)
    return r if isinstance(r, int) else r[0]


def build_workload():
    reqs = []
    sys_prompt = [RND.randrange(VOCAB) for _ in range(80)]
    for s in range(120):                                   # 多轮对话：120 会话 × 8 轮
        hist = sys_prompt + [RND.randrange(VOCAB) for _ in range(40)]
        for t in range(8):
            prompt = hist + [RND.randrange(VOCAB) for _ in range(36)]
            reply = [RND.randrange(VOCAB) for _ in range(120)]
            reqs.append((("chat", s, t), prompt, reply))
            hist = prompt + reply
    docs = [[RND.randrange(VOCAB) for _ in range(1500)] for _ in range(10)]
    instr = [RND.randrange(VOCAB) for _ in range(30)]
    for i in range(400):                                   # RAG：同一文档被反复提问
        reqs.append((("rag", i),
                     instr + docs[RND.randrange(10)] + [RND.randrange(VOCAB) for _ in range(60)],
                     [RND.randrange(VOCAB) for _ in range(80)]))
    shots = [d for _ in range(5) for d in ([RND.randrange(VOCAB) for _ in range(120)] + [1])]
    header = [RND.randrange(VOCAB) for _ in range(8)]
    for i in range(300):                                   # few-shot：605 token 公共示例块
        reqs.append((("few", i), shots + header + [RND.randrange(VOCAB) for _ in range(40)],
                     [RND.randrange(VOCAB) for _ in range(60)]))
    short_sys = [RND.randrange(VOCAB) for _ in range(24)]
    for i in range(500):                                   # 短前缀群：24 token 公共 system
        reqs.append((("short", i), short_sys + [RND.randrange(VOCAB) for _ in range(12)],
                     [RND.randrange(VOCAB) for _ in range(30)]))
    order = []
    for idx, (tag, _, _) in enumerate(reqs):
        if tag[0] == "chat":                       # 同一会话的轮次必须保持先后
            _, s, t = tag
            key = (s * 8 + t) * 60 + RND.randrange(60)
        else:
            key = RND.randrange(200_000)
        order.append((key, idx))
    order.sort()
    return [reqs[i] for _, i in order]


def simulate(reqs, cache, policy="fcfs", window=256, batch=32):
    waiting, per_req = list(range(len(reqs))), {}
    tot = hit = 0
    peak = 0
    t0 = time.perf_counter()
    while waiting:
        cand = waiting[:window]
        if policy == "lpm":            # 缓存感知：优先调度「已缓存前缀最长」的那些
            chosen = [i for _, _, i in sorted(
                ((hit_of(cache, reqs[i][1]), -k, i) for k, i in enumerate(cand)),
                reverse=True)[:batch]]
        else:
            chosen = cand[:batch]
        drop = set(chosen)
        waiting = [i for i in waiting if i not in drop]
        for i in chosen:
            _, prompt, reply = reqs[i]
            h = hit_of(cache, prompt)
            cache.insert(prompt + reply)     # 请求结束，prompt + 输出一起进缓存
            tot += len(prompt)
            hit += h
            per_req[i] = (len(prompt), h)
            peak = max(peak, cache.size)
    return dict(tot=tot, hit=hit, rate=hit / tot, peak=peak, per_req=per_req,
                mgmt_s=time.perf_counter() - t0, zombies=cache.zombies(),
                evicted=cache.evicted, n=len(reqs))


reqs = build_workload()
tot_prompt = sum(len(p) for _, p, _ in reqs)
KV_BYTES = 2 * 32 * 8 * 128 * 2      # 7B / 32 层 / GQA 8 / head_dim 128 / bf16
print(f"{len(reqs)} 条请求，prompt 合计 {tot_prompt:,} token；"
      f"7B 的 KV = {KV_BYTES/1024:.0f} KiB/token\n")

cases = [("无缓存（基线）", lambda: NullCache(), "fcfs"),
         ("树·无限·FCFS", lambda: RadixCache(), "fcfs"),
         ("树·无限·LPM", lambda: RadixCache(), "lpm"),
         ("块·无限·FCFS", lambda: BlockHashCache(), "fcfs"),
         ("树·60K·FCFS", lambda: RadixCache(60_000), "fcfs"),
         ("树·60K·LPM", lambda: RadixCache(60_000), "lpm"),
         ("树·15K·FCFS", lambda: RadixCache(15_000), "fcfs"),
         ("树·15K·LPM", lambda: RadixCache(15_000), "lpm"),
         ("块·15K·FCFS", lambda: BlockHashCache(15_000), "fcfs")]
print(f"{'配置':<14}{'命中率':>8}{'重算 token':>12}{'峰值缓存':>10}"
      f"{'驱逐 token':>12}{'僵尸':>7}{'管理耗时':>10}")
res = {}
for name, fac, pol in cases:
    r = simulate(reqs, fac(), policy=pol)
    res[name] = r
    print(f"{name:<14}{r['rate']*100:>7.2f}%{r['tot']-r['hit']:>12,}{r['peak']:>10,}"
          f"{r['evicted']:>12,}{r['zombies']:>7,}{r['mgmt_s']*1000:>9.0f} ms")

print(f"\n去重后的内容总量 = {res['树·无限·FCFS']['peak']:,} token "
      f"= {res['树·无限·FCFS']['peak']*KV_BYTES/1024**3:.1f} GiB；"
      f"平均每个 token 被复用 {tot_prompt/res['树·无限·FCFS']['peak']:.2f} 次")

print("\n分组命中率：")
print(f"  {'配置':<14}{'多轮对话':>10}{'RAG':>9}{'few-shot':>10}{'短前缀群':>10}")
for name in ["树·无限·FCFS", "块·无限·FCFS", "树·15K·FCFS", "树·15K·LPM"]:
    g = {}
    for i, (tag, prompt, _) in enumerate(reqs):
        a, b = g.setdefault(tag[0], [0, 0])
        g[tag[0]] = [a + len(prompt), b + res[name]["per_req"][i][1]]
    print(f"  {name:<14}" + "".join(f"{g[k][1]/g[k][0]*100:>9.1f}%" for k in
                                    ["chat", "rag", "few", "short"]))

N, NL, D = 6.74e9, 32, 4096
base = sum(2 * N * p + 4 * NL * D * p * p for p, _ in res["无缓存（基线）"]["per_req"].values())
print("\n按 7B（2N·T + 4Ld·T·S，T = 未命中 token 数）折算 prefill 算力：")
for name in ["树·无限·FCFS", "树·60K·FCFS", "树·15K·FCFS", "树·15K·LPM"]:
    fl = sum(2 * N * (p - h) + 4 * NL * D * (p - h) * p
             for p, h in res[name]["per_req"].values())
    print(f"  {name:<14}{fl:>10.4e} FLOPs = 基线的 {fl/base*100:>5.2f}%")
```

真实输出：

```text
2160 条请求，prompt 合计 1,523,820 token；7B 的 KV = 128 KiB/token

配置                 命中率    重算 token      峰值缓存    驱逐 token     僵尸      管理耗时
无缓存（基线）          0.00%   1,523,820         0           0      0        4 ms
树·无限·FCFS       93.63%      97,105   277,305           0      0      142 ms
树·无限·LPM        93.63%      97,105   277,305           0      0      261 ms
块·无限·FCFS       92.69%     111,420   279,760           0      0      646 ms
树·60K·FCFS      93.04%     106,105    60,000     226,356      0      145 ms
树·60K·LPM       93.53%      98,605    60,000     218,838      0      263 ms
树·15K·FCFS      78.29%     330,833    14,999     496,148      0      123 ms
树·15K·LPM       88.61%     173,630    15,000     340,176      0      233 ms
块·15K·FCFS      76.97%     350,988    14,992     486,992     16      637 ms

去重后的内容总量 = 277,305 token = 33.9 GiB；平均每个 token 被复用 5.50 次

分组命中率：
  配置                  多轮对话      RAG  few-shot      短前缀群
  树·无限·FCFS          94.1%     93.9%     93.6%     66.5%
  块·无限·FCFS          93.4%     93.2%     92.8%     44.4%
  树·15K·FCFS         94.1%     57.3%     92.9%     66.5%
  树·15K·LPM          94.1%     81.8%     93.6%     66.4%

按 7B（2N·T + 4Ld·T·S，T = 未命中 token 数）折算 prefill 算力：
  树·无限·FCFS     1.3591e+15 FLOPs = 基线的  6.34%
  树·60K·FCFS    1.4879e+15 FLOPs = 基线的  6.94%
  树·15K·FCFS    4.7040e+15 FLOPs = 基线的 21.93%
  树·15K·LPM     2.4544e+15 FLOPs = 基线的 11.44%
```

（表里最后一列是**墙钟时间**，只用于量级判断：同一台机器上重复跑会有 ±10% 波动，命中率、重算 token、峰值缓存、驱逐 token、僵尸这五列是确定的、逐次一致的。）

这张表里有五件值得逐条读的事：

**① 整体差距很小，分组差距很大。** 树 93.63% vs 块 92.69%，只差 0.94 个百分点。但拆开看分组：多轮对话只差 0.7pt、RAG 差 0.7pt、few-shot 差 0.8pt，而**短前缀群差了 22.1pt（66.5% vs 44.4%）**。这正好对上 3.2 节那张表的预测：$24 \bmod 16 = 8$，损失 8/24 = 33.3%。**「块粒度不好」这个说法只在短前缀负载上成立**，在长文档负载上完全可以忽略。

**② 僵尸块几乎没有。** 15K 容量下块缓存里有 16 token 僵尸 —— 也就是一个块、占那 14,992 token 缓存的 0.1%。原因在第 3.3 节提过：vLLM 把请求的块按倒序入空闲队列，父块天然比子块活得久。**「结构上允许」和「实际会发生」是两回事**，这条要老实说：我原本预期这里会是个大数字，实测推翻了这个预期。

**③ 容量掉到 1/5 才伤到命中率。** 277K（去重总量）→ 120K → 60K，命中率从 93.63% 只掉到 93.04%；到 30K 掉到 89.89%；15K 才崩到 78.29%。原因见 3.5 节：**驻留量是去重内容量，而 LRU 会优先留住热内容**。这个负载里有 10 篇文档（15K token）+ 605 token 示例块 + 若干活跃会话，它们构成的「热工作集」远小于 277K。

**④ 容量一紧张，调度策略的价值就出来了。** 15K 容量下：FCFS 重算 330,833 token，LPM 重算 173,630 token —— **÷1.91**。RAG 分组命中率 57.3% → 81.8%。机制很简单：LPM 优先调度「已缓存前缀最长」的请求，也就是把访问同一篇文档的请求攒在一起处理，避免 LRU 在文档之间来回抖动。**容量充足时 LPM 和 FCFS 完全一样（93.63%），容量不足时才是它的主场。**

**⑤ 树管理开销的上界。** 约 0.14 s（纯 Python 实现、2160 条请求、含所有匹配与插入）对应无缓存 prefill 折算时间 48.2 s（$2.1451\times10^{16} / (989\times10^{12}\times0.45)$），**占 0.3%**。论文在 ShareGPT 上测出的是「100 条请求总共 74.3 s，树操作只花 0.2 s，占比 0.3%」—— 同一量级。考虑到我的是 Python 而论文是 C++，这个 0.31% 可以当上界看。（顺便：块哈希实现是约 0.65 s，因为我用 tuple 做哈希键；真实实现里这是 xxhash/sha256 在 C++ 里的常数，不该拿我的 Python 实现下结论。）

---

## 5. 围绕该领域展开

### 5.1 它和 PagedAttention 是互补的两半

第 011 期讲过 PagedAttention：把 KV cache 切成固定大小的物理块，用「块表」把逻辑序列映射到物理块，解决的是**显存碎片**问题。前缀缓存解决的是**重复计算**问题。两者共用同一套块表，但是两个独立的层次：

```
请求 → 【前缀匹配】→ 已算过的 token 数 → 【块表分配】→ 物理块 → 【attention kernel】
         ↑ 本期                                  ↑ 第 011 期
```

**没有 PagedAttention，前缀缓存做不了**：前缀匹配的结果是一条「已命中的 token 前缀」，你要让新请求**指向**别人已经写好的那些物理块，而不是复制一份 KV。块表正是干这个的 —— vLLM 里叫 `ref_cnt` + 共享块，SGLang 里叫 `RadixCache` 返回的 `device_indices`（一段 KV 页索引）。这也是为什么两家都是在有了分页 KV 之后才把前缀缓存做扎实。

顺带一个容易忽略的约束：**共享的块是只读的**。两个请求共享前缀块，然后各自往里追加 token —— 追加必须写到新块上，不能原地改。这直接影响「一个请求能不能原地扩展它的最后一个块」，也是 vLLM v1 里块表被设计成 append-only 的原因之一。

### 5.2 命中率就是省下的算力比例 —— 一个精确等式

这一条我认为是整个话题里最漂亮的结果，而且它是精确的，不是近似。

prefill 的 FLOPs 拆成两项：

$$
\text{FLOPs}(T, S) = \underbrace{2NT}_{\text{线性项}} + \underbrace{4Ld \cdot T \cdot S}_{\text{注意力二次项}}
$$

其中 $N$ 是非 embedding 参数量，$L$ 是层数，$d$ 是 $d_{model}$，$T$ 是这一条请求实际要算的 query 数，$S$ 是 KV 全长（注意力里 query 仍然要跟全部 $S$ 个 key 做点积）。$N$ 的系数是 2 是因为每个权重每个 token 参与一次 MAC。

一次 cache miss：$T = S$。一次 cache hit：$T = S - P$（$P$ 是命中前缀）。两者相除：

$$
\frac{\text{FLOPs}(S-P, S)}{\text{FLOPs}(S, S)} = \frac{2N(S-P) + 4Ld(S-P)S}{2NS + 4LdS^2} = \frac{(S-P)(2N + 4LdS)}{S(2N + 4LdS)} = \frac{S-P}{S}
$$

**两项都带 $(S-P)$ 因子，正好约掉了。**

> 一次前缀命中省下的 prefill FLOPs 比例，**恒等于命中率** —— 与模型大小 $N$、层数 $L$、上下文长度 $S$ 全都无关。

第 4.1 节的输出里，命中 64/96 → 剩下的 FLOPs 是 33.3%，正好是 $(96-64)/96$。这不是巧合。

**但时间不是这个数。** 因为时间有下限：

$$
t_{\text{prefill}} = \max\left(\frac{\text{FLOPs}(T,S)}{\rho_{\text{eff}} \cdot \text{PEAK}},\ \frac{W_{\text{bytes}}}{\text{BW}}\right), \qquad \rho_{\text{eff}} = \text{MFU}
$$

权重读取那项与 $T$ 无关 —— **前向至少得把 14 GB 权重读一遍**。于是：

```python
"""文章代码块 5：命中率就是「算力省下的比例」——但时间不是"""
N, NL, D = 6.74e9, 32, 4096          # LLaMA-7B 量级
W_BYTES, PEAK, BW, MFU = 14e9, 989e12, 3.35e12, 0.45
LIN, ATT = 2 * N, 4 * NL * D         # 线性项系数 / 注意力二次项系数
FLOOR = W_BYTES / BW * 1e3           # 权重读一遍的下限，ms


def flops(S, T):
    """S = KV 全长；T = 这一条请求实际要算的 query 数（命中后只剩后缀）"""
    return LIN * T + ATT * T * S


def ttft(S, T):
    return max(flops(S, T) / (PEAK * MFU) * 1e3, FLOOR)


base = ttft(4096, 4096)
print(f"权重读一遍的带宽地板 = {FLOOR:.3f} ms；7B 全量 prefill 4096 token = {base:.1f} ms\n")
print(f"{'命中率':>8}{'后缀':>7}{'算力时间':>11}{'实际 TTFT':>11}{'降幅':>9}  瓶颈")
for hit_rate in [0, 0.5, 0.875, 0.969, 0.984, 0.992]:
    T = int(4096 * (1 - hit_rate))
    tc = flops(4096, T) / (PEAK * MFU) * 1e3
    t = max(tc, FLOOR)
    print(f"{hit_rate*100:>7.1f}%{T:>7}{tc:>9.2f}ms{t:>9.2f}ms{base/t:>8.2f}x  "
          f"{'算力' if tc > FLOOR else '带宽'}")

RHO = PEAK * MFU / BW
print(f"\n平衡算术强度 ρ = PEAK·MFU/BW = {RHO:.1f} FLOP/byte；"
      f"BF16 每读 1 字节权重对应 2/bytes = 1 FLOP/token，")
print(f"所以算力时间跌破地板发生在后缀 < ρ·bytes/2 = {RHO*2/2:.0f} token"
      f"（MFU=100% 时就是 {PEAK/BW*2/2:.0f} —— 027/028 期那个 295）")
print("命中率再往上加，TTFT 不再下降。")

# 一条具体的 RAG 请求：指令 30 + 文档 1500 + 问题 60 = 1590，命中 93.9%
S, T = 1590, 97
print(f"\n一条 RAG 请求（prompt {S} token，命中后只剩 {T} token 后缀）：")
print(f"  算力     {flops(S, S):.4e} → {flops(S, T):.4e} FLOPs，降到 "
      f"{flops(S, T)/flops(S, S)*100:.2f}%（省 {flops(S,S)/flops(S,T):.1f}x）")
print(f"  时间     {ttft(S, S):.2f} ms → {ttft(S, T):.2f} ms（只降 "
      f"{ttft(S,S)/ttft(S,T):.1f}x）")
print(f"  差额去哪了？命中后的算力时间只剩 {flops(S,T)/(PEAK*MFU)*1e3:.2f} ms，")
print(f"  已经低于权重读取的 {FLOOR:.2f} ms，于是被地板截住。")
```

真实输出：

```text
权重读一遍的带宽地板 = 4.179 ms；7B 全量 prefill 4096 token = 143.8 ms

     命中率     后缀       算力时间    实际 TTFT       降幅  瓶颈
    0.0%   4096   143.83ms   143.83ms    1.00x  算力
   50.0%   2048    71.91ms    71.91ms    2.00x  算力
   87.5%    512    17.98ms    17.98ms    8.00x  算力
   96.9%    126     4.42ms     4.42ms   32.51x  算力
   98.4%     65     2.28ms     4.18ms   34.42x  带宽
   99.2%     32     1.12ms     4.18ms   34.42x  带宽

平衡算术强度 ρ = PEAK·MFU/BW = 132.9 FLOP/byte；BF16 每读 1 字节权重对应 2/bytes = 1 FLOP/token，
所以算力时间跌破地板发生在后缀 < ρ·bytes/2 = 133 token（MFU=100% 时就是 295 —— 027/028 期那个 295）
命中率再往上加，TTFT 不再下降。

一条 RAG 请求（prompt 1590 token，命中后只剩 97 token 后缀）：
  算力     2.2759e+13 → 1.3884e+12 FLOPs，降到 6.10%（省 16.4x）
  时间     51.14 ms → 4.18 ms（只降 12.2x）
  差额去哪了？命中后的算力时间只剩 3.12 ms，
  已经低于权重读取的 4.18 ms，于是被地板截住。
```

**这张表要这么读**：

- 命中率 0 → 87.5% 这一段，**降幅就等于命中率**（2.00× 对应 50%，8.00× 对应 87.5%）—— 因为全程算力瓶颈，时间正比于 FLOPs；
- 过了 96.9% 之后，**命中率继续涨，TTFT 不动了**。后两行的降幅都是 34.42×，一模一样；
- 拐点在**后缀 < 133 token**（$T^* = \rho \cdot \text{bytes}/2$，$\rho = \text{PEAK}\cdot\text{MFU}/\text{BW}$）。MFU 取 100% 时 $T^* = 295$ —— **和第 027、028 期那个反复出现的 295 是同一个数**，只是那两期它管的是 decode 的「免费验证额度」，这里管的是 prefill 「命中之后还剩多少可降」。

这也解释了为什么超长上下文是前缀缓存的黄金场景：$S$ 越大，命中带来的**绝对**时间节省越大（143.8 ms → 4.18 ms 是 34×），而这个地板始终是 4.18 ms 不变。

### 5.3 和调度器的接合点：前缀缓存只是「起点不为 0」

第 028 期花了不少篇幅讲一件事：vLLM v1 的调度器里**没有 prefill 和 decode 的阶段之分**，只有「`num_computed_tokens` 追 `num_tokens_with_spec`」。前缀缓存落到这个模型里异常干净：

```
新请求进来 → match_prefix(prompt) = P
           → 初始 num_computed_tokens = P     ← 前缀缓存的全部贡献就这一行
           → 之后调度器照旧按预算推进
```

由此可以推出三件不那么直观的事：

1. **chunked prefill 天然兼容**：既然只有「已算 token 数」这一个状态量，命中的前缀就是「已经算完的部分」，切不切 chunk 无所谓。第 028 期提过 vLLM v1 里 chunked prefill「默认开启且优先调度 decode」—— 前缀缓存和它是正交的两件事。
2. **前缀缓存只优化 prefill，完全不碰 decode**。vLLM 功能文档的 Limits 一节写得很直接：APC 只减少处理 query 的时间（prefill），不减少生成新 token 的时间（decode）。
3. **抢占重算会打回原形**。第 028 期讲过 vLLM v1 的抢占只有 recompute、没有 swap，源码里 `_preempt_request` 做的是 `num_computed_tokens = 0`。注意这里要区分两件事：`num_computed_tokens` 归零是「这条请求要重算」，但它**下次被调度时仍然会先查一次前缀缓存**，如果它的块还在缓存里，重算的开销就很小。**前缀缓存把「抢占重算」从一次全量 prefill 变成了「命中后只算后缀」** —— 这是两个机制之间一个不显眼但很实在的协同。

### 5.4 缓存感知调度：LPM 与 DFS 序

第 4.4 节里 LPM 在 15K 容量下把重算 token 从 330,833 压到 173,630（÷1.91）。这个收益有理论解释，论文里给了定理（我核对了 PDF，是 Theorem 3.1）：

> 对一批请求，以**深度优先搜索顺序**遍历请求的基数树、并且缓存容量不小于最长请求长度时，可以达到最优命中率。**「最长共享前缀优先」（longest-shared-prefix-first）等价于 DFS 序。**

直觉很容易想通：把共享同一段前缀的请求**连续**处理。如果它们在时间上散开，LRU 会在中间把它们的前缀挤出去；连续处理则「算一次、连着用」。DFS 序遍历树，恰好就是把「共享祖先」的请求排在一起。

论文也诚实地说了一个代价（原文）："While greedy cache-aware scheduling can achieve high throughput, it can lead to starvation. We leave its integration with other fair scheduling methods as future work." —— **贪心的缓存感知调度会让「前缀不热」的请求饿死**，公平性要另外设计。这也解释了为什么 SGLang 当前的 `--schedule-policy` 默认不是 `lpm`（见 5.6）。

### 5.5 缓存和 batch 争同一块显存

第 3.4 节引了论文那段话，这里把它量化一下。上下文长度 $S$ 时，KV 占用是

$$
\text{KV}(S) = 2 \times L \times n_{\text{kv\_head}} \times d_{\text{head}} \times \text{bytes} \times S
$$

对 7B（$L=32$、$n_{\text{kv\_head}}=8$、$d_{\text{head}}=128$、bf16）：$128$ KiB/token。**缓存 10 万 token 的前缀，就是 12.5 GiB 的 KV 池。**

换算成「少放几条并发请求」：

| 缓存住的前缀 | 显存 | 等价于少放多少条 4096-token 的并发请求 |
|---|---|---|
| 15K token | 1.83 GiB | 3.7 条 |
| 60K token | 7.32 GiB | 15 条 |
| 120K token | 14.6 GiB | 30 条 |
| 277K token（本例去重总量） | 33.9 GiB | 69 条 |

**所以「前缀缓存要不要开」这个问题问错了。** 正着问应该是：**在我的负载上，缓存多留 1 GiB 换来的命中率提升，值不值得牺牲 2 条并发？** 这个问题只能靠实测的容量-命中率曲线回答（第 4.4 节那张表就是它的横切面）。vLLM 和 SGLang 都提供了 `--gpu-memory-utilization` / `--mem-fraction-static` 来切分权重与 KV 池，但没有哪个参数能替你回答这个问题。

### 5.6 源码 vs 文档：三处我觉得值得指出的地方

我在写这期时对着 SGLang 和 vLLM 的当前源码核了几件事。有三处二手资料和源码/文档不一致，值得记下来（**以下都标注来源类型**）：

**① `--schedule-policy` 的默认值**。网上（DeepWiki 的一篇分析）说默认是 `lpm`，另一篇博客和一本在线教材说默认是 `fcfs`。源码说的是后者 —— SGLang 主分支 `python/sglang/srt/arg_groups/fields/schedule.py`：

```python
schedule_policy: A[str, Arg(help="The scheduling policy of the requests.",
    choices=["lpm", "random", "fcfs", "dfs-weight", "lof", "priority",
             "routing-key", "hrrn", "shortest-prefill-first"])] = "fcfs"
```

**默认 `fcfs`。** 也就是说 SGLang 默认开着 RadixAttention，但**不会**为了命中率去重排等待队列。想启用缓存感知调度要显式传 `--schedule-policy lpm`。这和 5.4 节末尾那个「贪心调度会饿死」的坦白是一致的。策略枚举也分成了两类（同在 `schedule_policy.py`）：

```python
class CacheAwarePolicy(Enum):      # 感知树缓存
    LPM = "lpm"                    # 最长前缀匹配
    DFS_WEIGHT = "dfs-weight"      # 深度优先加权
    HRRN = "hrrn"                  # 最高响应比优先，带 token 老化
    SHORTEST_PREFILL_FIRST = "shortest-prefill-first"

class CacheAgnosticPolicy(Enum):   # 不感知树缓存
    FCFS = "fcfs"
    LOF = "lof"                    # 最长输出优先
    RANDOM = "random"
    ROUTING_KEY = "routing-key"
```

**② vLLM 的前缀缓存默认是开的**。功能文档（`features/automatic_prefix_caching`）还写着 "Set `enable_prefix_caching=True` in vLLM engine to enable APC"，而源码里 `vllm/config/cache.py` 是：

```python
enable_prefix_caching: bool = True
"""Whether to enable prefix caching."""
```

CLI 那侧 `--enable-prefix-caching` 的 argparse `default` 被设成 `None`（表示「用户没指定」），回落到 `CacheConfig` 的默认值。所以**有效默认是开启**，文档那句话是旧版本留下的。SGLang 那边同理：`--disable-radix-cache` 的默认是 `False`，即 radix cache 默认开启。

**③ 「块粒度 vs token 粒度」这个二分正在被两边一起抹掉。**
- vLLM 加了 `--prefix-match-unit`，源码注释（`config/cache.py`）说它是 "The finest token boundary (in tokens) a prefix-cache hit can land on"，可以**比物理块更细**（例如 32 vs 1024 的混合模型块），只要每个 KV cache group 的 `block_size` 能被它整除，「enabling cache hits at boundaries inside a physical block」。它只控制匹配粒度，不控制状态多久存一次；代码里它就等于 `hash_block_size`。此外哈希算法默认已从早期版本换成 `sha256`（`--prefix-caching-hash-algo` 还提供 `sha256_cbor` / `xxhash` / `xxhash_cbor`）。
- SGLang 加了 `--page-size`（`The number of tokens in a page.`，默认 `None` 按模型自动选），`RadixCache.match_prefix` 的 docstring 明确写了「如果 `page_size > 1`，匹配前 key 会被截断到 `page_size` 的整数倍」。

所以 3.2 节那张对照表，其实现代版本里更像是**两个可调参数的组合**，而不是两个流派的根本分歧。它俩真正的结构性差异只剩两条：**驱逐顺序**（树只能从叶子走 vs 块级 LRU 允许产生够不到的块）和**引用计数的作用域**（树是路径级、块是块级）。

还有一条 SGLang 侧的机制值得指出，因为它把「多租户隔离」做成了数据结构的一部分。`match_prefix` 的 docstring（`mem_cache/radix_cache.py`）：

> "The logical namespace for prefix matching is determined by both the token id sequence and the optional `extra_key` carried by `RadixKey`. Entries that share identical leading token ids but *different* `extra_key` values are intentionally kept disjoint and never share prefix nodes."

也就是说：**两条请求即使前缀逐 token 相同，只要 `extra_key` 不同，它们在树里就走不同的子树**。用途明说了三种：LoRA / adapter ID 隔离、采样 salt、以及「检索增强上下文」这类本来就不该共享的情况。这比「在哈希里混一个 salt」更彻底 —— 它是结构隔离，不只是 key 空间隔离。

### 5.7 顺带一个安全话题：命中率本身就是一条侧信道

这一条是前缀缓存的一个固有副作用，值得知道。命中与否只影响 TTFT，不影响输出内容，所以**攻击者不需要看到内容，只要计时就能判断「系统最近有没有处理过这段前缀」**。

```python
"""文章代码块 6：命中率是一条时序侧信道（用代码块 5 的模型）"""
N, NL, D = 6.74e9, 32, 4096
W_BYTES, PEAK, BW, MFU = 14e9, 989e12, 3.35e12, 0.45


def ttft(S, T):
    return max((2 * N * T + 4 * NL * D * T * S) / (PEAK * MFU), W_BYTES / BW) * 1e3


print("受害者反复查询一份私有文档，攻击者发一条前缀为该文档的探测请求。")
print("命中与否只影响 TTFT，不影响输出 —— 所以不需要看内容就能判断出来。\n")
print(f"{'探测长度':>10}{'命中 TTFT':>13}{'未命中 TTFT':>14}{'时间差':>12}{'比值':>9}")
for S in [512, 1024, 2000, 4096, 8192]:
    hit, miss = ttft(S, 2), ttft(S, S)     # 命中后只剩 2 个 token 的后缀要算
    print(f"{S:>10}{hit:>11.2f}ms{miss:>12.2f}ms{miss-hit:>10.1f}ms{miss/hit:>8.1f}x")
print("\n网络抖动是毫秒级，而这里的判据是几十到上百毫秒 —— 不需要精确计时也能分辨。")
```

真实输出：

```text
受害者反复查询一份私有文档，攻击者发一条前缀为该文档的探测请求。
命中与否只影响 TTFT，不影响输出 —— 所以不需要看内容就能判断出来。

      探测长度      命中 TTFT      未命中 TTFT         时间差       比值
       512       4.18ms       15.82ms      11.6ms     3.8x
      1024       4.18ms       32.25ms      28.1ms     7.7x
      2000       4.18ms       65.29ms      61.1ms    15.6x
      4096       4.18ms      143.83ms     139.6ms    34.4x
      8192       4.18ms      327.18ms     323.0ms    78.3x

网络抖动是毫秒级，而这里的判据是几十到上百毫秒 —— 不需要精确计时也能分辨。
```

注意最左列恒定在 4.18 ms —— 地板又把命中的 TTFT 压平了（5.2 节），所以**只要命中了，延迟看起来都一样**，这反而让判据更干净：看到 4.18 ms 就是命中，看到 65 ms 就是没命中。

两家的对策都已经做成了显式参数：
- **vLLM**：请求里带 `cache_salt`，它被注入第一个块的哈希，只有同 salt 的请求能复用。设计文档里点明了动机："This prevents timing-based attacks where an adversary could infer cached content by observing latency differences."
- **SGLang**：`RadixKey` 的 `extra_key`，效果是子树级隔离（5.6 ③）。

**这两个机制默认都是关的**（salt 得自己传）。多租户、或者同一台机器上跑不同客户的请求时，值得主动打开。

---

## 6. 什么时候该用 / 不该用

**该用**（判据是「前缀重复率」和「prompt 长度」两个维度）：

| 场景 | 为什么 |
|---|---|
| 多轮对话 / agent 循环 | 历史越长，本轮新增占比越低；命中率随轮次上升 |
| RAG，同一份文档反复提问 | 文档段 100% 共享 |
| few-shot / 固定工具定义 / 长 system prompt | 纯重复内容，收益最直接 |
| 并行采样、思维树、self-consistency | 分叉前 100% 共享 |
| 长上下文（≥ 8K） | 省下的绝对时间最大（$S$ 越大越值） |

**不该用 / 要关掉**：

- **prompt 之间几乎不共享**（每条都是独一无二的短请求）。命中率接近 0 时，缓存只占显存不产生收益 —— SGLang 的 `--disable-radix-cache`、vLLM 的 `--no-enable-prefix-caching` 就是为这个准备的。（不过开销本身很小：论文实测 0.3%，SGLang 也因此敢默认开启。）
- **decode 占绝对主导的负载**。前缀缓存一分钱 decode 都不省。用户要求很长的输出、输入很短时，TTFT 本来就不是瓶颈。
- **显存已经紧到在抢 batch 的场景**。见 5.5：缓存多的和 batch 大的是同一块显存。
- **多租户且没有做隔离**。见 5.7。
- **注意一个反直觉点**：命中率最高的配置**不一定**是端到端最优。第 5.2 节已经说明，命中率过了某个点之后 TTFT 就不再下降，但显存占用还在涨。极端情况下（比如把 chunked prefill 的预算全让给缓存），命中率看着漂亮，吞吐反而降。

---

## 7. 常见坑

**① 命中要求 token 序列逐位相同 —— tokenizer 一变，命中率归零。**
这是实践中最容易吃的一闷棍。同一个 system prompt，只要前后多一个空格、多一个换行、或者模板里 `<|user|>` 换成 `<|User|>`，就完全是另一条路径。更隐蔽的：多模态输入里图片被替换成一串占位 token，如果不把图片哈希塞进块的 key，`[IMG]` 位置相同但内容不同的两条请求会被误判为共享 —— vLLM 专门为此在哈希里加了 `extra hash`（图片哈希）就是干这个的。**排查命中率异常时，第一件事是打印 token id 序列比对，不是打印文本。**

**② 只缓存满块 ⇒ 最后那段尾巴永远进不了缓存。**
vLLM 的设计文档里有 Note：「We only cache full blocks.」所以一条长度不是 16 倍数的 prompt，尾巴 $(L \bmod 16)$ 个 token 永远不进缓存。**再加上命中从第 0 块起连续**这个约束，块粒度下的损失就是 3.2 节那个 $\text{命中长度} \bmod 16$。第 9 节的练习会看到它在短前缀负载下的实际伤害。

**③ 「显存只增不减」这类结构性 bug 不会报错。**
第 3.3 节那个 `_split` 的 bug 是我这期真的踩到的：节点分裂后原节点从 `leaves` 集合里消失，从此不可驱逐。**表现是显存缓慢上涨、命中率没有任何异常**。防御手段只有一条：给数据结构写不变量自检（我用的 `audit()` 断言 `leaves` 集合与遍历出的真实叶子集合完全相等），并且在**容量受限**的模式下长时间跑。容量无限时这个 bug 完全暴露不出来。

**④ 引用计数没算对，会静默产出垃圾输出。**
正在 decode 的请求脚下的 KV 被驱逐，输出不会报错，只会变得莫名其妙。要点：路径上**每个**节点都要 `+1`（不只是末端节点），请求正常结束和异常中断（客户端断连、`max_tokens` 截断、抢占）都要 `-1`。SGLang 论文里给了一个边界情况：如果没有预分配固定大小的池，缓存和运行中请求共享内存时，**系统可能在请求多起来时驱逐掉所有缓存 token 去换更大的 batch** —— 这个行为本身是有意的，但意味着「缓存占用」不是单调的，监控时别把它当泄漏。

**⑤ 别把「命中率」当成唯一的观测指标。**
vLLM 暴露的是两个计数器（`vllm:prefix_cache_queries` 和 `vllm:prefix_cache_hits`）而不是一个现成的命中率仪表 —— 它把时间窗口的选择权留给你。至少要看三个量：**命中率**（复用效果）、**去重后内容量**（该预留多少显存，见 3.5）、**驱逐量/驱逐速率**（在不在抖动）。只看第一个数，很容易得出「命中率 95%，很健康」的结论，而实际上驱逐速率已经高到每次都要重算一遍被驱逐的东西。

---

## 8. 一句话总结

**前缀缓存把 KV cache 从「每条请求的私有副本」变成了「一棵跨请求共享的树」，它省下的算力比例精确等于命中率，但省下的时间会在「权重读取」这道地板上停下来 —— 所以它真正改变的是长 prompt 的 TTFT，而它真正付出的代价，是从 batch size 里借走的那部分显存。**

---

## 9. 今日练习

**题目**：把第 4.3 节的 `BlockHashCache` 改成块大小可配，在同一个工作负载上跑块大小 = 4 / 8 / 16 / 32 / 64，回答三个问题：

1. 块越小，总命中率能涨多少？涨幅值不值得？
2. **短前缀群**（24 token 公共 system prompt）的命中率随块大小怎么变？为什么？
3. 代价项（哈希表条目数、块表长度、管理耗时）随块大小怎么变？

```python
"""文章代码块 7（今日练习参考答案）：块大小到底该怎么选"""
from collections import OrderedDict

from exp4_workload import build_workload


class BlockCache:
    def __init__(self, block, capacity_tokens=None):
        self.block = block
        self.capacity = capacity_tokens
        self.hashes = {}
        self.free = OrderedDict()
        self.size = 0
        self.total_entries = 0

    def _keys(self, tokens):
        keys, parent = [], None
        for b in range(0, len(tokens) - self.block + 1, self.block):
            k = (parent, tuple(tokens[b:b + self.block]))
            keys.append(k)
            parent = k
        return keys

    def match_prefix(self, tokens):
        hit, parent = 0, None
        for b in range(0, len(tokens) - self.block + 1, self.block):
            k = (parent, tuple(tokens[b:b + self.block]))
            if k not in self.hashes:
                break
            hit += self.block
            self.free.pop(k, None)
            self.free[k] = None
            parent = k
        return hit

    def insert(self, tokens):
        for k in self._keys(tokens):
            if k in self.hashes:
                continue
            self.hashes[k] = self.block
            self.free[k] = None
            self.size += self.block
            self.total_entries += 1
        while self.capacity is not None and self.size > self.capacity and self.free:
            k, _ = self.free.popitem(last=False)
            self.hashes.pop(k, None)
            self.size -= self.block

    def zombies(self):
        return 0


reqs = build_workload()
print(f"{'块大小':>8}{'总命中率':>10}{'短前缀群':>11}{'多轮对话':>10}"
      f"{'哈希条目数':>12}{'平均块表长':>12}{'管理耗时':>11}")
for bs in [4, 8, 16, 32, 64]:
    cache = BlockCache(bs)
    g = {"chat": [0, 0], "short": [0, 0]}
    tot = hit = 0
    blocks = 0
    import time
    t0 = time.perf_counter()
    for tag, prompt, reply in reqs:
        h = cache.match_prefix(prompt)
        cache.insert(prompt + reply)
        tot += len(prompt)
        hit += h
        if tag[0] in g:
            g[tag[0]][0] += len(prompt)
            g[tag[0]][1] += h
        blocks += (len(prompt) // bs) + 1        # 该请求块表里的条目数
    dt = time.perf_counter() - t0
    print(f"{bs:>8}{hit/tot*100:>9.2f}%{g['short'][1]/g['short'][0]*100:>10.1f}%"
          f"{g['chat'][1]/g['chat'][0]*100:>9.1f}%{len(cache.hashes):>12,}"
          f"{blocks/len(reqs):>12.1f}{dt*1000:>9.0f} ms")
```

<details><summary>今日练习（点开看参考答案与实测输出）</summary>

真实输出：

```text
     块大小      总命中率       短前缀群      多轮对话       哈希条目数       平均块表长       管理耗时
       4    93.56%      66.5%     94.1%      69,076       177.2     5673 ms
       8    93.35%      66.5%     93.9%      34,492        88.7     1708 ms
      16    92.69%      44.4%     93.4%      17,485        44.7      630 ms
      32    91.24%       0.0%     92.3%       9,191        22.6      243 ms
      64    88.79%       0.0%     90.0%       4,940        11.5      164 ms
```

**问题 1：总命中率能涨多少？**
从 16 降到 4，总命中率 **92.688% → 93.555%**，只涨 **0.87 个百分点**（重算 token 从 111,420 降到 98,204，省 13,216 token —— 相对 152 万 token 的请求量，省下 0.87%）。而代价是哈希条目数从 17,485 涨到 69,076（**3.95×**）、管理耗时从约 0.63 s 涨到约 5.7 s（**9×**）。**不划算。**

各块大小的精确重算量：

| 块大小 | 命中 token | 重算 token | 命中率（4 位小数） |
|---|---|---|---|
| 4 | 1,425,616 | 98,204 | 93.5554% |
| 8 | 1,422,464 | 101,356 | 93.3486% |
| 16 | 1,412,400 | 111,420 | 92.6881% |
| 32 | 1,390,368 | 133,452 | 91.2423% |
| 64 | 1,352,960 | 170,860 | 88.7874% |
 整体命中率是被长前缀负载（RAG / few-shot / 多轮对话）主导的，而这些负载在块大小 16 下的损失本来就不到 1%（3.2 节）。

**问题 2：短前缀群为什么会崩？**
看这一列：块 4 → 66.5%，块 8 → 66.5%，块 16 → **44.4%**，块 32/64 → **0.0%**。

- 公共前缀是 24 token。$24 \bmod 4 = 0$、$24 \bmod 8 = 0$、$24 \bmod 16 = 8$、$24 < 32$。
- 块 4 和 8 能完整命中这 24 token（命中率 24/36 = 66.7%，实测 66.5%，差值来自少数请求在别处撞上的巧合共享）。
- 块 16 只命中 16 token → 16/36 = 44.4%。
- **块 32 及以上直接是 0：24 个 token 连一个满块都凑不出来，整段公共 system prompt 在缓存里根本不存在。** 这比「损失 33%」严重得多 —— 是「完全没有」。短 prompt 负载配大块，前缀缓存等于没开。

**问题 3：代价怎么变？**
两列都有很清晰的趋势，且方向相反：

| 块大小 | 哈希条目数 | 平均块表长 | 管理耗时 | 短前缀群命中率 |
|---|---|---|---|---|
| 4 | 69,076 | 177.2 | 约 5.7 s | 66.5% |
| 8 | 34,492 | 88.7 | 约 1.7 s | 66.5% |
| 16 | 17,485 | 44.7 | 约 0.63 s | 44.4% |
| 32 | 9,191 | 22.6 | 约 0.24 s | 0.0% |
| 64 | 4,940 | 11.5 | 约 0.16 s | 0.0% |

块大小每翻一倍：条目数减半、块表长度减半、管理耗时降到约 1/3。

**结论：块大小 = 8 是这里的甜点。** 它拿到了块大小 4 的绝大部分命中率好处（短前缀群 66.5%，与块 4 完全相同；总命中率 93.349% 对 93.555%，只差 0.21pt，重算量差 3,152 token），但条目数只有块 4 的一半、管理耗时只有约 1/3.4。块大小 4 的额外收益（+0.21pt）远不能抵消它的代价。而块大小 ≥ 32 在短 prompt 负载上是灾难 —— 这也是为什么「块大小设大点省显存」这种想法在有短共享前缀的负载上会翻车。

**扩展思考**：真实的块大小选择还牵涉一个本文模型没覆盖的成本 —— **attention kernel 的分页开销**。块越小，块表越长，attention kernel 在页表上跳转的次数越多、访存越碎。这就是为什么 vLLM 提供 `--prefix-match-unit` 而不建议直接把 `--block-size` 调到 1：**匹配粒度**和**物理块大小**是两个独立的旋钮，前者影响命中率，后者影响 kernel 效率。5.6 ③ 提到的那条设计演进（匹配可以比物理块更细）正是为了解耦这两件事。

</details>

---

> **下一期预告方向**：PD 分离与 KV 传输链路（disaggregation）—— 把「前缀缓存的共享」推进到「跨机器的共享」，也就是 LMCache / NIXL / Mooncake 那条线在做的事。
>
> **本站**：[https://morss1.github.io/pytorch-daily/](https://morss1.github.io/pytorch-daily/)
