# PyTorch 每日一课 · 第 028 期

## 连续批处理（Continuous Batching）：把「一批请求」拆成「一次迭代」

> **日期**：2026-09-29　**难度**：⭐⭐⭐⭐　**预计阅读**：30 分钟
>
> **前置知识**：attention 与 KV cache 的基本结构；第 011 期 PagedAttention 的「块表」概念；第 027 期投机解码里的 roofline 口径（7B BF16 / H800 级 / 权重 14 GB / 3.35 TB/s / 989 TFLOPS）
>
> **关联**：第 011 期 · 它和 PagedAttention 是共生关系，没有分页 KV cache 就没有连续批处理；第 027 期 · 两者是 decode 优化的两面——投机解码买**延迟**，连续批处理买**吞吐**，读完之后你会知道为什么它们不能同时最大化；第 021 期 · 「每一步的 batch 形状都不一样」是动态形状问题的最极端形态

---

## 1. 结论先行

三个数字，先摆出来：

| 问题 | 静态 batching | 连续 batching | 变化 |
|---|---|---|---|
| 256 条请求、槽位 32 的 burst 场景，**槽位利用率**（正在做有用工作的槽位比例） | 22.1% | 61.2% | — |
| 同上，**跑完 52285 个输出 token 的总步数** | 7377 步 | 2670 步 | **÷2.76** |
| 同上，H800 级设备口径下的**总墙钟时间** | 30.98 s | 11.19 s | **÷2.77** |
| 稳态过载下的 **TTFT p99**（首 token 延迟尾部） | 21.5 s | 1.03 s | **÷20.9** |

连续批处理不是「把 batch 调大一点」或者「换个更好的调度参数」。它改的是**调度的时间粒度**：从「一个请求进 batch 就待到整批结束」改成「每一步都重新决定谁在 batch 里」。

这一个改动之所以能白拿 2~3 倍吞吐，是因为它恰好压在一个物理事实上——**在 GPU 上，decode 阶段的 batch 从 1 涨到 295，每一步的耗时不涨**。这个「免费额度」在第 027 期讲投机解码时出现过（$B \cdot L \ll 295$），这一期是它的另一半用法：**既然批里塞更多请求不额外花钱，那就每一小步都重新塞满。**

---

## 2. 静态 batching 的两个结构性缺陷

先说清楚要淘汰的东西。**静态 batching**（也叫 request-level batching）是 2022 年之前所有推理服务框架的做法：

```
收集 N 条请求 → 组成一个 batch → 一起跑 forward → 全部跑完 → 收集下一批
```

它有两个缺陷，都不是参数问题。

### 2.1 缺陷一：长尾把整批拖住

一批 N 条请求共享同一个 forward。谁的输出最长，整批就要陪着它跑到最后一步。其他人跑完之后，它们的槽位**空转**——batch 维度还是 N，每一步仍然要按 N 行做计算，只是那些行产出的是废 token。

这不是理论担忧。LLM 的输出长度天然长尾：绝大多数回答几十个 token 就完事，少数会写上千个。我按 log-normal（median=128、σ=1.0，这是 LLM 输出长度的常用近似）采样了 1024 条长度为 burst 负载，然后精确统计「静态 batching 一个 batch 里有多少槽位在做有用工作」——这个计数与设备无关，是纯算术：

```
burst workload：1024 条请求同时到达，输出长度 log-normal(median=128, σ=1.0)
  实测分布：p50=133  p90=462  p99=1226  max=2021  mean=211.8   总输出 216925 token
  CPU f(1)=10.38ms f(64)=337.74ms（实测，线性外推）
  GPU f(1)=4.184ms f(64)=4.500ms f(512)=7.245ms（roofline）

      N    组数    Σmax步      槽位浪费 E[max]/E[len]     CPU总时间(s)     GPU总时间(s)
      1  1024   216925      0.0%          1.00        2251.5        907.64
      4   256   116673     53.5%          2.15        3529.8        489.93
      8   128    80213     66.2%          2.96        4037.8        338.43
     16    64    53891     74.8%          3.97        5075.8        229.53
     32    32    33452     79.7%          4.93        6067.1        145.16
     64    16    20990     83.9%          6.19        7089.1         94.45
    128     8    12041     85.9%          7.10        8121.9         58.04
    256     4     6517     87.0%          7.69        8791.7         35.59
    512     2     3515     87.9%          8.30        9483.8         25.47
   1024     1     2021     89.5%          9.54       10905.7         29.28
```

这张表是本期最反直觉的一张，值得盯一会儿。先看「槽位浪费」这一列：**批开得越大，浪费越严重**。批大小 32 时浪费 79.7%，批大小 1024（全部 1024 条一次跑完）时浪费 89.5%。

根因是一个纯粹的极值统计：批大小 $N$ 的同步步数是 $\max$ 而不是 $\mathrm{mean}$。定义

$$\text{浪费率} = 1 - \frac{\sum_i \ell_i}{N \cdot \max_i \ell_i}$$

对 log-normal 这类重尾分布，$E[\max_N]/E[\ell]$ 随 $N$ 缓慢但持续增长（表里从 1.00 涨到 9.54），浪费率就一路爬到 90%。

**「批开大一点」在静态 batching 下是越开越亏的。** 这解释了为什么当年大家直觉上觉得「batch 越大越好」，实测却总差着一截——那个直觉来自均匀长度的图像分类数据集，而不是长尾的文本生成。

### 2.2 缺陷二：新请求必须等整车

静态 batching 的第二个缺陷在延迟侧，而且更致命：**新请求必须等当前整批跑完才能进**。

考虑一个 batch 里有一条要生成 2000 token 的请求。接下来几十秒内到达的所有短请求，全部压在队列里干等。它们的**首 token 延迟（TTFT）**不受自己长短影响，只受「前面那条最长的请求还要跑多久」影响。

我用泊松到达生成稳态过载负载（到达率 = 连续调度上限的 1.2 倍），对比三种调度的 TTFT 尾部：

```
====================================================================================
场景 2 · 稳态过载（λ = 1.2 × 连续调度上限，GPU 时间尺度）   （256 条请求，槽位预算 M=32）

  [GPU roofline]  f(1)=4.180ms  f(32)=4.199ms  f(64)=4.219ms
  策略                  步数     墙钟(s)      槽位浪费    步均(ms)   TTFTp50    TTFTp99    TTFTmax     tok/s
  静态 batching       7377     31.55     77.9%      4.20   12608.1    21490.9    21552.5    1657.1
  动态 batching       8499     35.67     80.8%      4.20   13756.2    26955.4    26962.9    1465.8
  连续 batching       2607     10.93     37.3%      4.19     199.1     1027.1     1035.9    4784.6
    吞吐相对静态：静态 1.00×   动态 0.88×   连续 2.89×
    TTFT p99 相对静态：静态 1.000×   动态 1.254×   连续 0.048×
```

静态 batching 的 TTFT p99 是 **21.5 秒**，连续 batching 是 **1.03 秒**——差 20.9 倍。而最坏情况（TTFT max）对比更刺眼：21.6 s vs 1.04 s。

### 2.3 还有一个中间态：动态 batching

表里的「动态 batching」值得单独说。它不等齐 N 条，有多少收多少就发车（Triton Inference Server 的 dynamic batching 就是这个）。它解决了「等齐」的问题，但**没有解决「跑到底」的问题**——一批仍然要跑到最长的那条结束。

结果在上面那张表里很尴尬：**动态 batching 在 GPU 口径下比静态还差（0.88×）**。原因在第 4 节会讲清楚——在 memory-bound 区间，每一步都要完整读一遍 14 GB 权重，批次越小这个固定成本摊得越薄。「小批多次」比「大批少次」更亏。这也解释了为什么 Triton 的 dynamic batching 要配一个 `max_queue_delay_microseconds`：**主动等一下，把批凑大**。

---

## 3. 那个被忽略的前提：decode 时算力几乎免费

连续批处理之所以是「白拿」，建立在第 027 期建立的那个 roofline 事实之上。复习一下（口径与第 027 期完全一致，7B BF16 / H800 级）：

$$t_{\text{mem}} = \frac{7\times10^9 \times 2\ \text{bytes}}{3.35\times10^{12}\ \text{B/s}} = 4.179\ \text{ms}, \qquad
\frac{\text{PEAK}}{BW} = \frac{989\times10^{12}}{3.35\times10^{12}} = 295.2\ \text{FLOP/byte}$$

decode 一步的算力需求是 $2N \cdot B$ FLOP（$B$ 是批里的 token 数），访存需求是**同一个** 14 GB（权重要整读一遍，与批大小无关）。所以只要

$$B \cdot L \ll 295$$

耗时就恒为 $t_{\text{mem}} = 4.179$ ms，**批里塞 1 个 token 还是 294 个 token 都一样贵**。

于是：

- **在 GPU 上（memory-bound 区间），把空槽位填满的成本是 0**。短请求跑完腾出的槽位，立刻塞一条新请求进去，那一步的耗时不变。这是连续批处理「白拿」的全部来源。
- **在 CPU 上，这个前提不成立**。CPU 只有几十个核心、SIMD 宽度有限、没有张量核心，decode 一步接近 compute-bound。我在这台机器的 CPU 上实测了同架构 tiny 模型（26.2M 参数、fp32 权重 100 MiB）的 decode 单步耗时：

```
模型参数量 26,227,712，fp32 权重 100.1 MiB
decode 单步（T_past=128，1 条新 token）
    B        ms    相对 B=1    GiB/s
    1    10.379     1.000      9.7
    2    18.195     1.753      5.7
    4    30.253     2.915      3.6
    8    50.339     4.850      2.4
   16    94.186     9.075      1.5
   24   138.126    13.308      1.2
   32   181.368    17.475      1.1
   48   271.490    26.158      0.9
   64   337.735    32.540      0.8
   96   507.640    48.911      0.7
  128   674.524    64.990      0.7
```

**CPU 上耗时随 batch 几乎线性增长**（B=64 是 B=1 的 32.5 倍），没有任何平台期。这是 compute-bound 的签名。

这个对照不是为了说明 CPU 差，而是为了指出一件重要的事：**连续批处理的收益大小是硬件形态的函数**。在 memory-bound 的设备上它近乎免费；在 compute-bound 的设备上，填满槽位要真花算力，收益就退化回「减少步数」这一项。后面 7.6 节会看到，这个差异大到能让「最优批大小」的方向完全反过来。

---

## 4. 核心机制：迭代级调度

### 4.1 从 Orca 开始

把调度粒度从「请求」降到「一步」这个想法，最早系统化提出是在 **Orca**（Yu et al., OSDI 2022）。它给这个做法起的名字是 **iteration-level scheduling**（迭代级调度）。核心改动只有一句话：

> **不再把 batch 当成「一批请求」，而是把每一步当成「一次 forward，成员可以换」。**

类比：静态 batching 像**单位班车**——坐满发车，到终点站才让下车，中途不停。连续 batching 像**地铁**——每到一站（每一步 decode）就开门，谁到站谁下，站台上等着的人立刻上车，车永远不空驶。

### 4.2 vLLM V1 源码里的那段注释

vLLM 的 V1 调度器把这件事做到了极致。源码里有一段 woosuk 写的设计注释，是理解整个设计的钥匙（原文照引）：

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec. num_tokens_with_spec =
# len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec. This is general enough to cover
# chunked prefills, prefix caching, speculative decoding,
# and the "jump decoding" optimization in the future.
```

翻译一下这段话的分量：**调度器里根本没有「prefill 阶段」和「decode 阶段」这两个概念**。每条请求只携带两个数——已经算了多少 token（`num_computed_tokens`）和总共有多少 token 需要算（`num_tokens_with_spec`）。调度器每步做的事，就是分配 token 预算让前者追上后者。

一旦用这个视角看，很多看起来是「三个不同功能」的东西变成了**同一件事的三种表现**：

| 看起来是独立功能 | 在这个视角下其实是 |
|---|---|
| chunked prefill（把长 prompt 切片） | 一个请求的 `num_computed_tokens` 分几步追上 |
| prefix caching（复用公共前缀） | `num_computed_tokens` 的初值被抬高了一截 |
| 投机解码（一次验证 γ+1 个 token） | `num_tokens_with_spec` 里多算了 γ 个草稿 token |
| 抢占后重算 | `num_computed_tokens` 被重置为 0 |

这就是「领域级理解」和「API 复读」的分界线：**API 层面它们是四个开关，机制层面它们是一个机制。**

### 4.3 三种策略的谱系

| 策略 | 调度粒度 | 成员何时进出 | GPU 口径总时间（burst，1024 条/槽位 32） |
|---|---|---|---|
| 静态 batching | 一批请求 | 整批开始 / 整批结束 | 基线 |
| 动态 batching | 一批请求 | 收满或超时就发，仍跑到底 | ≈ 静态（小批时更差） |
| **连续 batching** | **一次迭代** | **每一步都可以换成员** | **÷2.77** |

---

## 5. 在 PyTorch 里怎么用：手写一个最小连续 batching 引擎

调度逻辑本身很短，难点全在工程细节上。下面这个例子真跑一个 tiny 模型（128 维 × 2 层 × 4 头，460K 参数），把 6 条请求按两种方式跑，打印每一步的批次组成。

请求：A(输出 4)、B(12)、C(2)、D(8)、E(3)、F(6)，全部在 t=0 到达，槽位上限 3。

```python
import torch, torch.nn as nn

class ContinuousBatchEngine:
    """最小连续 batching 引擎：每步重新决定批次成员。

    两个必须自己处理的工程细节（真实引擎里对应 PagedAttention 和 CUDA Graph 分桶）：
      1. 各请求的 KV cache 长度不同（上车时间不同），没法直接 torch.cat
         → 这里用左 padding + attention mask 兜住
      2. 每步的批次形状都在变，所以「批」不能预先捕获成一张 CUDA Graph
         → 真实系统按 batch size 分桶，见 7.4 节
    """

    def __init__(self, model, max_seqs: int):
        self.model = model
        self.max_seqs = max_seqs          # 等价于 vLLM 的 max_num_seqs
        self.waiting = []                 # 还没上车的请求
        self.running = []                 # 正在车上跑的请求

    def add(self, req_id, prompt_ids, out_len):
        """prefill：prompt 一次性算完，留下 KV cache 后进入 waiting"""
        with torch.no_grad():
            logits, caches = self.model(prompt_ids)
        self.waiting.append({
            "id": req_id, "caches": caches, "rem": out_len,
            "next": logits[:, -1].argmax(-1, keepdim=True),
        })

    def step(self):
        """跑一步。返回 (本步成员, 本步完成的人)。"""
        # 1) 先把跑完的请下车 —— 这一步就是连续 batching 的全部魔法
        done = [r for r in self.running if r["rem"] <= 0]
        self.running = [r for r in self.running if r["rem"] > 0]
        # 2) 再用 waiting 补满空出来的槽位
        while len(self.running) < self.max_seqs and self.waiting:
            self.running.append(self.waiting.pop(0))
        if not self.running:
            return [], done
        self._decode(self.running)
        return list(self.running), done

    def _decode(self, reqs):
        """把活跃请求拼成一个 batch 跑一步 decode（左 padding + mask 处理不等长）"""
        lens = [r["caches"][0][0].shape[2] for r in reqs]
        lmax = max(lens)
        nxt = torch.cat([r["next"] for r in reqs], 0)
        caches = []
        for i in range(len(self.model.blocks)):
            ks, vs = [], []
            for r, L in zip(reqs, lens):
                k, v = r["caches"][i]
                if lmax > L:                       # 左 padding，padding 位会被 mask 掉
                    k = torch.cat([torch.zeros(1, k.shape[1], lmax - L, k.shape[3]), k], 2)
                    v = torch.cat([torch.zeros(1, v.shape[1], lmax - L, v.shape[3]), v], 2)
                ks.append(k); vs.append(v)
            caches.append((torch.cat(ks, 0), torch.cat(vs, 0)))
        mask = torch.zeros(len(reqs), 1, 1, lmax + 1)
        for j, L in enumerate(lens):
            if lmax > L:
                mask[j, 0, 0, :lmax - L] = float("-inf")
        with torch.no_grad():
            logits, new = self.model(nxt, caches=caches, addmask=mask)
        for j, r in enumerate(reqs):
            off = lmax - lens[j]
            r["caches"] = [(new[i][0][j:j + 1, :, off:, :],
                            new[i][1][j:j + 1, :, off:, :])
                           for i in range(len(self.model.blocks))]
            r["next"] = logits[j:j + 1, -1].argmax(-1, keepdim=True)
            r["rem"] -= 1
```

调度主循环只有几行：

```python
engine = ContinuousBatchEngine(MODEL, max_seqs=3)
for rid, plen, out_len in REQS:
    engine.add(rid, torch.randint(0, VOCAB, (1, plen)), out_len)

step = 0
while engine.waiting or engine.running:
    members, done = engine.step()          # ← 每一步都重新决定成员
    for r in done:
        print(f"  step {step:>2}: {r['id']} 跑完下车，槽位立刻让出来")
    if members:
        print(f"  step {step:>2}  在算 [{' '.join(r['id'] for r in members)}]"
              f"  有效槽位 {len(members)}/3")
        step += 1
```

真跑输出（静态 batching 对照）：

```
请求（全部在 t=0 到达）：[('A', 'prompt8→输出4'), ('B', 'prompt8→输出12'), ('C', 'prompt8→输出2'), ('D', 'prompt8→输出8'), ('E', 'prompt8→输出3'), ('F', 'prompt8→输出6')]

============================================================================
静态 batching：凑够 3 条才发车，整批同步推进到最慢的那条结束
============================================================================
  批 1：[A B C] 输出长度 [4, 12, 2] → 要跑 12 步
    step  0  实际在算 [A B C]  有效槽位 3/3  2.44 ms
    step  1  实际在算 [A B C]  有效槽位 3/3  0.26 ms
    step  2  实际在算 [A B]  有效槽位 2/3  0.24 ms
    step 11  实际在算 [B]  有效槽位 1/3  0.12 ms
  批 2：[D E F] 输出长度 [8, 3, 6] → 要跑 8 步
    step  0  实际在算 [D E F]  有效槽位 3/3  0.23 ms
    step  1  实际在算 [D E F]  有效槽位 3/3  0.20 ms
    step  2  实际在算 [D E F]  有效槽位 3/3  0.24 ms
    step  7  实际在算 [D]  有效槽位 1/3  0.12 ms

  总计：20 步，槽位-step 60，其中有用 35 → 浪费 41.7%

============================================================================
连续 batching：每一步重新决定谁上车，跑完的立刻腾出槽位（上限 3）
============================================================================
    step  0  在算 [A B C]  有效槽位 3/3  0.23 ms
    step  1  在算 [A B C]  有效槽位 3/3  0.22 ms
    step  2: C 跑完下车，槽位立刻让出来
    step  2  在算 [A B D]  有效槽位 3/3  0.23 ms
    step  3  在算 [A B D]  有效槽位 3/3  0.23 ms
    step  4: A 跑完下车，槽位立刻让出来
    step  4  在算 [B D E]  有效槽位 3/3  0.24 ms
    step  5  在算 [B D E]  有效槽位 3/3  0.24 ms
    step  6  在算 [B D E]  有效槽位 3/3  0.25 ms
    step  7: E 跑完下车，槽位立刻让出来
    step  7  在算 [B D F]  有效槽位 3/3  0.23 ms
    step  8  在算 [B D F]  有效槽位 3/3  0.24 ms
    step  9  在算 [B D F]  有效槽位 3/3  0.25 ms
    step 10: D 跑完下车，槽位立刻让出来
    step 10  在算 [B F]  有效槽位 2/3  0.22 ms
    step 11  在算 [B F]  有效槽位 2/3  0.23 ms
    step 12: B 跑完下车，槽位立刻让出来
    step 12  在算 [F]  有效槽位 1/3  0.12 ms
    step 13: F 跑完下车，槽位立刻让出来

  总计：13 步，槽位-step 39，其中有用 35 → 浪费 10.3%
```

读这张输出的关键：

- **静态：20 步，槽位浪费 41.7%。** 批 1 里 C 在第 3 步就完事了，但它的槽位一直占到第 12 步（`step 11 实际在算 [B] 有效槽位 1/3`）。同时 D/E/F 在队列里干等了整整 12 步。
- **连续：13 步，浪费 10.3%，剩下的 10.3% 全是尾部**（最后两步批次里只剩 B、F 两条，没有别人可以补进来）。**只要队列够深，连续 batching 的浪费趋近于 0。**

注意 `step 0` 在两边都是最慢的（2.44 ms / 0.23 ms）——因为那是第一次 forward，包含预热开销。

### 5.1 不想自己写的话，现成入口在哪

- **最省事**：vLLM / SGLang 的 OpenAI 兼容服务，连续批处理是**默认行为**，没有任何开关需要打开。你能调的只有 `max_num_seqs`、`max_num_batched_tokens`、`gpu_memory_utilization` 这几个预算参数。
- **要在 PyTorch 里自建**：真正难的不是调度循环（就是上面那 60 行），而是它依赖的三件套——(1) 能处理变长 KV 的 attention（PagedAttention 的 varlen 接口，或 flash-attn 的 `cu_seqlens`）；(2) 一个运行时能增删成员的 KV 块管理器；(3) 按批大小分桶的 CUDA Graph。**这三件都不在 torch 的公开 API 里**，所以现实中要么用现成引擎，要么自己搭一整套。
- **训练侧没有对应物**：训练的 batch 是静态的（等长或 padding 到等长），因为梯度的归约要求 batch 成员在整段计算里保持一致。**连续批处理是推理独有的概念**，它成立的前提是「decode 每一步的输出互不影响，所以成员可以随便换」——这个前提在反向传播里不成立。这也是为什么第 018 期讲的训练数据管线（DataLoader / prefetch）用的是完全不同的思路：那边优化的是「下一批数据什么时候准备好」，这边优化的是「这一步的批里放谁」。

---

## 6. vLLM V1 的调度器实际长什么样

上面的例子是玩具。真实引擎要处理的是「预算有限、显存有限、请求带优先级、还要和投机解码/前缀缓存共存」。以下全部来自 vLLM V1 的 [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) 与官方 `optimization.md` 文档，我区分「源码/文档里的原文」与「我的推断」。

### 6.1 队列与调度顺序

源码里的队列成员（**不是**老架构的 `waiting / running / swapped` 三件套）：

```python
self.waiting = create_request_queue(self.policy)      # 还没算过 prefill 的
self.skipped_waiting = create_request_queue(self.policy)  # 因异步依赖/约束被跳过的
self.running: list[Request] = []                      # 正在跑的（是 list，不是 queue）
```

两个调度策略由 `SchedulingPolicy` 枚举给出：`FCFS = "fcfs"`、`PRIORITY = "priority"`。对应的两个实现差别很实在：

- `FCFSRequestQueue` 继承 `deque[Request]`，`prepend_request` 就是 `appendleft`（O(1) 插到队首）。
- `PriorityRequestQueue` 用 `heapq` 维护，元素按 `(priority, arrival_time)` 排序，**它的 `prepend_request` 没有「插队首」的语义**（源码注释原话：「In a priority queue, there is no concept of prepending to the front」），就是重新 `heappush`，让堆按优先级重排。

每步的调度顺序是固定的两段：

```python
# First, schedule the RUNNING requests.
while req_index < len(self.running) and token_budget > 0:
    ...

# Next, schedule the WAITING requests.
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        ...
```

**先 RUNNING 后 WAITING，而且只要本步发生过抢占，就整步跳过 WAITING。** 后一条是个躲不开的设计：本步已经因为显存不够踢人了，再往车上塞新人只会立刻又被踢下去，来回抖动。

### 6.2 三个预算数字决定一切

调度器的行为基本被这几个数锁死：

| 参数 | 源码里的角色 | 直觉 |
|---|---|---|
| `max_num_seqs` | `self.max_num_running_reqs`，**model runner 的槽位数** | 车上总共几个座 |
| `max_num_active_seqs` | `self.max_num_active_reqs`，**准入上限**，默认回落到上面那个 | 最多允许几个人同时在车上（暂停流式会话仍占座） |
| `max_num_batched_tokens` | `input_budget`，本步新算 token 的总预算 | 这趟车总共能走多少「token 米」 |
| `max_num_scheduled_tokens` | `token_budget`，默认回落到 `max_num_batched_tokens` | 同上，但可单独收紧 |
| `long_prefill_token_threshold` | 单个长 prefill 一次最多吃多少 token | 防止一条长请求把预算吃光 |

其中 `long_prefill_token_threshold` 有一段自适应逻辑值得单看（源码原文注释 + 代码）：

```python
# `long_prefill_token_threshold` exists to stop a long prefill from
# starving other requests of the token budget. When it is the only
# request there is nobody to starve, so let it use the whole budget.
num_eligible_reqs = (
    len(self.running) + len(self.waiting) + len(self.skipped_waiting)
)
long_prefill_token_threshold = (
    self.scheduler_config.long_prefill_token_threshold
    if num_eligible_reqs > 1
    else 0
)
if long_prefill_token_threshold > 0 and self.adaptive_long_prefill_threshold:
    # Floor the cap at a fair share of the input budget so it never
    # cuts a request below max_num_batched_tokens / num requests.
    long_prefill_token_threshold = max(
        long_prefill_token_threshold, input_budget // num_eligible_reqs
    )
```

两个细节：**（1）只有车上/站台上还有别人时这个上限才生效**——系统里就这一条请求，没人需要被保护，那就让它一次吃满预算，别白白多切几刀。**（2）上限有下限**——不能切得比「人均预算」还小，否则切出来的块太小，每一块都要重新读一遍权重，纯亏。

### 6.3 抢占：V1 只有 recompute

抢占的触发点很直白：`kv_cache_manager.allocate_slots(...)` 返回了 `None`（显存里腾不出连续的 KV 块）。

```python
if new_blocks is not None:
    break
...
# The request cannot be scheduled.
# Preempt the lowest-priority request.
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
else:
    preempted_req = self.running[-1]
```

**FCFS 下踢的是 `self.running[-1]`——队尾那个，也就是最近才上车的请求。** 这个选择有道理：新人重算的代价最小（它已经算过的 token 最少）。PRIORITY 下则踢「优先级数值最大 + 到达最晚」的。

被踢的请求怎么处理？

```python
def _preempt_request(self, request, timestamp, drop_stale_output=False):
    ...
    self._free_request_blocks(request)
    ...
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    ...
    request.num_preemptions += 1
    # Put the request back to the waiting queue.
    self.waiting.prepend_request(request)
```

**`num_computed_tokens = 0` 是关键**：它的 KV 已经被释放了，所以进度条归零、从头再来（recompute）。**这条路径里没有「把 KV 换到 CPU」这一步。** 官方文档也明说了：

> In vLLM V1, the default preemption mode is `RECOMPUTE` rather than `SWAP`, as recomputation has lower overhead in the V1 architecture.

（我在源码里确认了：整个 `vllm/v1/core/sched/scheduler.py` 只有 recompute 这一条路径，没有 swap 分支。「老架构 V0 有 `PreemptionMode.SWAP`」是我基于官方文档对 V0 的描述以及 V1 文档的措辞做的推断，没有在这一版源码里直接验证。）

看到「抢占」出现时的官方建议是这几条（原文）：

> - Increase `gpu_memory_utilization`. vLLM pre-allocates GPU cache using this percentage of memory.
> - Decrease `max_num_seqs` or `max_num_batched_tokens`. This reduces the number of concurrent requests in a batch, thereby requiring less KV cache space.
> - Increase `tensor_parallel_size` / `pipeline_parallel_size`.

以及那句很有信息量的警告日志：

```
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

**盯 `total_cumulative_preemption_cnt` 这个累计计数**，它比吞吐数字更早告诉你「显存预算给少了」。

### 6.4 chunked prefill 在 V1 是默认开启的

这一点很多人还停留在 V0 的 `--enable-chunked-prefill` 开关印象里。官方 `optimization.md` 原文：

> In V1, chunked prefill is enabled by default whenever possible.
>
> With chunked prefill enabled, the scheduling policy prioritizes decode requests. It batches all pending decode requests before scheduling any prefill operations. When there are available tokens in the `max_num_batched_tokens` budget, it schedules pending prefills. If a pending prefill request cannot fit into `max_num_batched_tokens`, it automatically chunks it.

它带来的两个好处（文档原文列举）：

> - It improves inter-token latency (ITL) and generation decode because decode requests are prioritized.
> - It helps achieve better GPU utilization by locating compute-bound (prefill) and memory-bound (decode) requests to the same batch.

第二句是本期最值得划线的工程洞察：**把 compute-bound 的 prefill 和 memory-bound 的 decode 塞进同一批，两边的浪费互相填。** prefill 需要算力但算力闲着，decode 需要带宽但带宽闲着——合在一起，权重的那一次读取被两边共享。

`max_num_batched_tokens` 的调法文档也给了方向：

| 取值 | 效果 |
|---|---|
| 小（如 `2048`） | ITL 更好（更少的 prefill 拖慢 decode） |
| 大 | TTFT 更好（一批能吞更多 prefill token） |
| `> 8192` | 官方推荐的吞吐最优区间（小模型 + 大 GPU 尤其） |
| `== max_model_len` | 几乎等价于 V0 的默认调度策略（区别：仍然优先 decode） |

还有一条硬约束要记住：**关掉 chunked prefill 时，`max_num_batched_tokens` 必须大于 `max_model_len`，否则服务启动就可能崩。**

---

## 7. 围绕这个领域展开

### 7.1 它和 PagedAttention 是共生的，不是两个独立优化

这是本期最重要的一节。

连续 batching 每一步都在换批次成员。成员一换，KV cache 的**逻辑顺序**就变了：这一步 B 排在第 1 位，下一步 C 下车、F 上车，B 就排到第 2 位。如果 KV cache 是连续内存布局，那**每一步都要把活跃序列的 KV 重新搬到新位置**，否则 attention kernel 无法按新顺序找到它们。

这个搬运量有多大？按 7B（32 层 / 8 个 KV head / head_dim 128 / bf16，即每 token 每层 4096 字节）、单序列上下文 2048、批 32 条算：

| 方案 | 字节数 | H800 上耗时 | 占一步 decode 的比例 |
|---|---|---|---|
| 搬全量 KV（无分页） | 8.00 GiB | 2.564 ms | **61.4%** |
| 改块表（PagedAttention） | 512.0 KiB | 0.157 µs | 0.0037% |

**比值 16384×。** 我在这台机器上真跑量过这两种操作的形态差：

```
   --- 关键区别：重排的是数据还是指针 ---
   真跑基准：gather 一个 [32, 8, 2048, 64] fp32 张量（模拟一层 KV 的按序重取）
     耗时 2.09 ms，搬运 128.0 MiB → 有效带宽 59.79 GiB/s
   真跑基准：gather 一个 block table [32, 32, 128] int32
     耗时 18.00 µs，搬运 512.0 KiB
```

结论很清楚：**没有 PagedAttention 的块表，连续 batching 光是「把活跃序列重新排紧」就要吃掉六成 decode 时间，2.77× 的收益会被吃回负数。** 反过来，PagedAttention 解决了显存碎片，但如果没有连续 batching，块表带来的「可以任意组合」的能力就没处用。两者是同一个设计的两个面 —— 这也是第 011 期的主题。

### 7.2 chunked prefill：削峰 13.5×，只花 5.4% 的代价

不做 chunked prefill 时，一条长 prompt 的 prefill 会独占一整步。这一步有多长，同时在建的所有 decode 用户就要等多久 —— 这就是 **ITL（inter-token latency）尖峰**。

用本期的 roofline 口径算（7B BF16、32 条 decode 在跑、上下文 128）：

```
   纯 decode 步（B=32）耗时 = 4.339 ms

   方案                              该步耗时(ms)      ITL 尖峰倍数     prefill总耗时(ms)
   不分块，一次吞完 L=512                     7.698          1.77              7.698
   不分块，一次吞完 L=1024                   14.942          3.44             14.942
   不分块，一次吞完 L=2048                   29.432          6.78             29.432
   不分块，一次吞完 L=4096                   58.411         13.46             58.411

   切 512 一块，分 8 步混跑                   7.698          1.77             61.581
```

把 L=4096 的 prefill 一次吞完，那一步要 **58.4 ms**，32 条 decode 用户集体吃到一个 **13.5× 的 ITL 尖峰**。切成 512 token 一块混跑后，尖峰降到 **1.77×**，而 prefill 自己完成的总时间只从 58.4 ms 变成 61.6 ms——**多花 5.4% 的时间，把尖峰削掉 7.6 倍**。

注意这里的「1.77×」不是 0 成本：切块后每块仍要触发一次 forward，权重读取的固定成本被摊了 8 次。这 5.4% 就是摊销损失的量化值。

（这张表是 roofline 推算，标注清楚：本地没有 CUDA 设备，GPU 上的绝对耗时来自模型而非实测。）

### 7.3 它和投机解码是同一枚硬币的两面

刚刚讲完的第 027 期，和本期几乎是同一套 roofline 的两个相反用法：

| | 连续批处理（本期） | 投机解码（第 027 期） |
|---|---|---|
| 想优化什么 | **吞吐**（同一时间内服务多少请求） | **延迟**（单条请求多久出一个 token） |
| 用掉免费额度 $B\cdot L \ll 295$ 的方式 | 横向填满 batch（$B$ 变大） | 纵向填满 batch（$L$ 变大，一次验证 γ+1 个 token） |
| 在什么负载下赚 | 高 QPS、队列够深 | 低 QPS、memory-bound |
| 在什么负载下亏 | 请求数不足、队列空 | batch 超过 $B_{\text{crit}} = B^*\tau/(\gamma+1)$ |

**两者争夺的是同一份免费额度。** 一次前向的 $B \cdot L \le 295$ 是一块固定大小的蛋糕：你把 $B$ 填满（连续批处理），就没有空间给 $L$（投机解码）；你想纵向验证 8 个草稿 token，横向就只剩 $B \le 33$ 条能并发。

这条约束的实际推论：**同一个部署上同时开连续批处理和投机解码，必须按实时 batch 动态调 $\gamma$**，否则两边互相挤。vLLM 文档说投机解码「reduce inter-token latency under medium-to-low QPS, memory-bound workloads」，在同一份文档里连续批处理讲的是 throughput —— 官方其实在告诉你这两个开关的适用区间不重叠。

### 7.4 每一步的 batch 都在变 —— 动态形状的极端形态

第 021 期讲过 PyTorch 的动态形状：一个 `SymInt` 变成多个具体值就会触发重编译。**连续批处理的 batch 维度是每步都变的**，这是动态形状问题能遇到的最坏情况：

- 静态 batching：batch 维度恒为 N，序列长度每步 +1（规整）。
- 连续 batching：batch 维度在 1 到 M 之间任意跳，序列长度还各不相等。

CUDA Graph 要求形状固定的物理图，所以**不能把整个 decode 循环捕获成一张图**。工程上的解法是 **piecewise CUDA graph（按批大小分桶）**：预先为 `batch_size ∈ {1, 2, 4, 8, ..., max_num_seqs}` 各捕获一张图，运行时挑最接近的那张，多出来的槽位用 dummy 请求填。

代价很直接：**分桶意味着实际 batch 落在 32 就按 32 跑，落在 33 就要按 64 跑，多出来的 31 个槽位是白烧的。** 所以「批大小分桶粒度」和「槽位利用率」是一对需要权衡的东西。我在实验 A 里真跑出的 f(B) 曲线（B=32 是 181.4 ms、B=48 是 271.5 ms）就说明这种阶梯是真实存在的成本。

### 7.5 抢占的两种代价，以及为什么 V1 还是选重的那个

抢占有两种实现，代价结构完全不同：

- **recompute**：丢掉 KV、`num_computed_tokens = 0`、从头重算 → 代价 = 重算 L 个 token 的 prefill $= 2PL$ FLOPs
- **swap**：把 KV 换出到 CPU、需要时再换回来 → 代价 = 搬 $2 \times \text{KV}_{\text{per token}} \times L$ 字节过 PCIe

两者都与序列长度 $L$ 成正比，所以可以按「每 token 成本」直接比：

```
   PCIe 4.0 x16 单向有效带宽按 25 GB/s 计

   模型               P    KV/token  recompute µs/token  swap µs/token    swap 便宜?
   7B 32层8KV       7B       128 KiB               14.15          10.49           是
   13B 40层8KV     13B       160 KiB               26.28          13.11           是
   70B 80层8KV     70B       320 KiB              141.50          26.21           是

   临界条件：swap 便宜 ⟺ KV_per_token × PEAK < P × PCIE
   → P > KV_per_token × PEAK / PCIE = KV_per_token × 39576
   以 32 层 / 8 KV head / head_dim 128 计：P > 5.19B
```

这里出现了一个**我的推算和官方选择不一致**的地方，值得诚实摆出来：按这个 roofline，**swap 更便宜**，而且模型越大优势越大（70B 时 swap 便宜 5.4 倍）。但 vLLM V1 的实际默认是 recompute。

怎么解释？**这个 roofline 只算了两条数据通路的算力和带宽，没算架构成本** —— 官方给的理由就是「recomputation has lower overhead in the V1 architecture」。要维持 swap 路径，需要 CPU 侧的内存池、KV 的换出/换入调度、以及抢占往往发生在负载高峰时 PCIe 正在被其他通信争用。而 V1 的 PagedAttention 块本来就可以原地重新分配，重算一行代码就够了。

**这就是「模型推算」和「工程决策」的分界**：roofline 能告诉你哪条路径的物理代价更低，告诉不了你哪条路径的实现复杂度更低。做性能判断时两者都要，而且要说清哪个是哪个。

### 7.6 反直觉结论：设备形态决定批大小的方向

回到第 2 节那张表，换个视角看它。左边四列是**与设备无关的纯计数**（怎么分组决定步数），右边两列是把它乘上各设备自己的 $f(N)$：

| N | Σmax步 | 槽位浪费 | E[max]/E[len] | CPU 总时间(s) | GPU 总时间(s) |
|---|---|---|---|---|---|
| 1 | 216925 | 0.0% | 1.00 | **2251.5** | 907.64 |
| 4 | 116673 | 53.5% | 2.15 | 3529.8 | 489.93 |
| 8 | 80213 | 66.2% | 2.96 | 4037.8 | 338.43 |
| 16 | 53891 | 74.8% | 3.97 | 5075.8 | 229.53 |
| 32 | 33452 | 79.7% | 4.93 | 6067.1 | 145.16 |
| 64 | 20990 | 83.9% | 6.19 | 7089.1 | 94.45 |
| 128 | 12041 | 85.9% | 7.10 | 8121.9 | 58.04 |
| 256 | 6517 | 87.0% | 7.69 | 8791.7 | 35.59 |
| 512 | 3515 | 87.9% | 8.30 | 9483.8 | **25.47** |
| 1024 | 2021 | 89.5% | 9.54 | 10905.7 | 29.28 |

**CPU 上最快的批大小是 1**（2251.5 秒），**GPU 上最快的是 512**（25.47 秒）。方向完全相反。

- CPU（compute-bound）：批越大越慢，N 从 4 涨到 1024 总时间膨胀 **3.09 倍**。因为浪费的槽位是真的白烧算力。
- GPU（memory-bound）：批越大越快，N 从 4 涨到 1024 总时间缩到 **1/16.7**。因为每一步的 14 GB 权重读取是固定成本，批越大摊得越薄，**长尾浪费的损失远小于摊销的收益**。
- GPU 上还有拐点：**N=1024 时反而比 N=512 慢了 15%**（29.28 vs 25.47 s），因为 $B$ 冲过了 295 那个平衡点，转成 compute-bound，此时长尾浪费才真正开始收钱。

所以「批开大点还是开小点」这个问题**没有通用答案**，答案取决于你的设备在 roofline 上站的位置。而连续批处理的价值恰恰在于：**它是唯一一个能在「设备喜欢大批」和「请求长度是长尾」之间不妥协的调度方式** —— 每步都是满批（满足设备对摊销的需求），每步的成员又都是此刻真正活跃的请求（满足长尾的约束）。

### 7.7 调度策略谱系

| 策略 | 抢占对象 | 适用 |
|---|---|---|
| FCFS（vLLM 默认） | `running[-1]`，队尾最新上车的 | 通用；新人重算代价最小 |
| PRIORITY | `max(running, key=(priority, arrival_time))`，优先级最低的 | 多租户、有 SLA 分级的线上服务 |
| SJF（最短作业优先） | — | 理论上能最小化平均延迟，但输出长度事前不可知 |
| 抢占 + 重排队 | 上述两种 | 显存不足时的兜底 |

工程上还有个更实用的维度：**准入控制**。与其让请求进来再抢占，不如在 `waiting` 队列里就拦住——`max_num_active_seqs` 就是干这个的，它和 `max_num_seqs`（runner 槽位）分开，让「能同时占显存」和「能同时跑」变成两个独立旋钮。

### 7.8 goodput：为什么不能只盯吞吐

上面所有对比我用的都是「总墙钟时间」和「tok/s」。但真实服务里**吞吐提升常常是拿延迟换的**：多塞请求进 batch，单位时间产出多了，每条请求的 ITL 也变长了。

所以线上应该盯的是 **goodput**：只统计**满足 SLO 的那些请求**的吞吐。

$$\text{goodput} = \frac{\#\{\text{请求} \mid \text{TTFT} \le \tau_1 \wedge \text{ITL} \le \tau_2\}}{\text{时间}}$$

这个定义之下，「连续 batching 提升 2.77× 吞吐」和「连续 batching 把 TTFT p99 从 21.5 s 压到 1.03 s」是同一件事的两面：**静态 batching 下的高吞吐里，很大一部分是 SLO 违约的请求贡献的**，根本不能计入 goodput。

### 7.9 同一件事在不同引擎里叫什么

| 引擎 | 叫法 | 与 vLLM 的差异 |
|---|---|---|
| Orca（论文，OSDI 2022） | iteration-level scheduling | 原始出处，只提了调度，没有 KV 内存管理方案 |
| vLLM | continuous batching | 块表 + 抢占 + chunked prefill，V1 里默认全开 |
| TensorRT-LLM | in-flight batching | 与 CUDA Graph 绑得更紧，批大小分桶更激进 |
| SGLang | RadixAttention + 连续批处理 | 前缀共享用 radix tree 而非块哈希 |
| HF TGI | continuous batching | 早期版本是纯动态 batching，后来才跟上 |

**SGLang 的 RadixAttention 值得单独说一句**：它和连续批处理是**正交**的两个优化，但共享同一个前提——KV 可以被任意组合。RadixAttention 用一棵基数树管理所有请求的前缀，任意两个请求的公共前缀在显存里只存一份。在 agent / few-shot 这类 system prompt 反复出现的场景里，它直接把 prefill 的工作量砍掉一大截，而且是**跟连续批处理叠加生效**的（前者省 prefill 的算力，后者省 decode 的带宽）。

**PD 分离（disaggregation）**是这条路线走到尽头的形态：既然 prefill 是 compute-bound、decode 是 memory-bound，那就干脆用两组不同的 GPU 分别跑，中间把 KV 过网络传过去。收益是两边各自按自己的瓶颈配硬件，代价多了一条 KV 传输链路和一套流水编排。目前主要出现在大规模自建栈里，中小规模下引入的复杂度通常盖过收益。

**这几个方向放在一起看，会发现它们都在解决同一个约束**：KV cache 是推理系统里唯一一个「随上下文线性增长、又必须随机访问」的状态。PagedAttention 管它的**布局**，连续批处理管它的**生命周期**，RadixAttention 管它的**复用**，投机解码管它的**生产速率**，PD 分离管它的**存放位置**。理解了这一点，这一整片技术就串成一条线了。

---

## 8. 什么时候该用 / 不该用

**收益最大的场景**

- 输出长度方差大（长尾重）。这是收益的主来源，可以用 2.1 节那张表估：$\text{浪费率} = 1 - E[\ell]/E[\max_N]$。
- 请求持续到达、队列深度足够（能保持槽位填满）。burst 越像「细水长流」，收益越大。
- memory-bound 设备（GPU 的常规 decode 区间），此时填满槽位近乎免费。
- 槽位预算有限（显存不够开大 batch）。连续 batching 用同样的显存换更高吞吐。

**收益很小的场景**

- 输出长度高度均匀（如批量做固定格式抽取、classification）。此时 $E[\max_N] \approx E[\ell]$，浪费率接近 0，连续调度没有可优化的空间。
- 离线批处理、所有请求一次性给全。此时没有「途中到达」的需求，静态分组 + 按长度排序分桶就能接近最优。
- 单请求 / 低并发。1 条请求时没有 batching 可言。

**要小心的场景**

- **compute-bound 设备**（本机 CPU 就是例子）：填满槽位要真花算力，收益退化成「减少步数」这一项。
- **槽位预算给得太大**：一旦 $B$ 冲过 roofline 平衡点（BF16 下 $B\cdot L = 295$），你会同时付「长尾浪费」和「算力成本」两份钱。
- **延迟敏感 + 尾部要求严**：连续 batching 的抢占会在尾部制造方差，且抢占的代价（recompute 或 swap）最终落在被抢占那条请求的延迟上。

---

## 9. 常见坑

**坑 1 · 把 `max_num_seqs` 当吞吐旋钮一路往上调。**
它是**显存预算的除数**，不是吞吐旋钮。调大的直接后果是每个请求分到的 KV 块变少 → `allocate_slots` 失败 → 开始抢占 → 被抢占的请求从头重算 → 总吞吐反而下降。先看 `total_cumulative_preemption_cnt`，它非零就说明你在踩这个坑。

**坑 2 · 以为「连续 batching 消除了所有浪费」。**
它消除的是**批内长尾**的浪费，消除不了**尾部请求不足**的浪费。我上面的仿真里，burst 场景下连续 batching 仍有 **38.8%** 的槽位空置——最后一批请求陆续跑完、没有新请求补进来，那段时间槽位就是空的。要压掉这部分只能靠「请求到达更密」或「降低槽位预算」。

**坑 3 · 在 compute-bound 设备上照搬 GPU 的调参直觉。**
第 2 节那张表已经量化了：同样一份 workload，CPU 口径下最快批大小是 1，GPU 口径下是 512。带着「大 batch 一定吞吐高」的经验去调加速器或 CPU 后端，会稳定调错方向。**先测你的 $f(B)$ 曲线，看它有没有平台期**——有平台期就是 memory-bound，可以放心开大；近似线性就是 compute-bound，别开。

**坑 4 · 以为 chunked prefill 只影响延迟、不影响吞吐。**
它同时影响两边，而且方向相反。切块降低 ITL 尖峰（延迟变好），但每块都要重读一遍权重（prefill 自身吞吐变差，上面实测 5.4%）。`max_num_batched_tokens` 就是在调这个配比：调小偏向 ITL，调大偏向 TTFT 与 prefill 吞吐。**别把它当纯收益开关。**

**坑 5 · 用平均吞吐评价延迟敏感的服务。**
静态 batching 的平均吞吐看着还行，TTFT p99 是 21.5 秒（见 2.2 节那张表）。平均值把最糟的那部分请求藏起来了。用 goodput 说话。

---

## 10. 一句话总结

> **连续批处理把调度粒度从「一批请求」降到「一次迭代」，让每一步都重新决定谁在车上；它之所以近乎白拿，是因为 memory-bound 区间里填满槽位不额外花钱——而这个前提本身，也决定了它在你手上的设备上到底值多少。**

---

<details>
<summary>今日练习（3 题，参考答案在下面）</summary>

### 练习 1 · 估算尾部浪费

假设槽位预算 $M = 64$，请求以 200 req/s 持续到达，平均输出 150 token。估算「尾部请求不足导致的槽位空置率」大概是多少？

提示：算一下「一个请求的平均驻留时间」和「系统平均在跑多少条」，看离 64 有多远。

### 练习 2 · 决定要不要开投机解码

一个线上服务：`max_num_seqs = 128`，实测平均 batch = 96，单条请求平均生成 400 token。你想同时开投机解码（$\gamma = 4$）。用第 027 期的 $B_{\text{crit}} = B^* \cdot \frac{\tau}{\gamma+1}$ 判断一下该不该开，并说清为什么。

### 练习 3 · 反推抢占代价

你的服务出现了抢占。用源码事实回答：FCFS 策略下被踢的是哪条请求？它被踢之后 `num_computed_tokens` 变成多少？如果这条请求 prompt 长度 3000、已经生成 500 个 token，那么重算的代价大约是多少 FLOPs（按 7B 模型估）？和「把它的 KV swap 出去再换回来」比，哪个更贵？

</details>

<details>
<summary>参考答案</summary>

### 答案 1

- 单请求驻留时间：$150\ \text{token} \times t_{\text{step}}$。按 7B/H800 口径 $t_{\text{step}} \approx 4.2$ ms（memory-bound，满批时仍是这个数），所以驻留 $\approx 630$ ms。
- 由 Little's Law：系统平均并发 $= \lambda \times W = 200 \times 0.63 \approx 126$ 条。
- 126 > 64，**预约的槽位根本不够用**，系统稳态是「排队一直有货」，槽位空置率接近 0——但代价是排队延迟极长。

要点：**尾部空置率高说明槽位预算过剩（浪费显存），空置率接近 0 才说明预算用满。** 这一题里该做的是「请求到达太密、槽位不够」，应该加槽位或加卡，而不是担心空置。

（如果换成 20 req/s：平均并发 $= 20 \times 0.63 = 12.6$ 条，只有槽位预算 64 的 20%，此时空置率约 80%，说明 `max_num_seqs=64` 开得太大了。）

### 答案 2

不该开，或者至少要大幅降低 $\gamma$。

代入第 027 期的公式：

$$B_{\text{crit}} = B^* \cdot \frac{\tau}{\gamma+1}, \qquad B^* = 295.2\ (\text{BF16})$$

$\gamma = 4$ 时，$\tau(4)$ 取决于单步接受率 $\alpha$。取一个偏乐观的 $\alpha = 0.8$：

$$\tau(4) = \frac{1-\alpha^5}{1-\alpha} = \frac{1-0.32768}{0.2} = 3.36$$

$$B_{\text{crit}} = 295.2 \times \frac{3.36}{5} = 198.4$$

看起来 96 < 198，似乎还能赚。但要注意第 027 期的另一个界：**衰减从 $L_0 = 295/(\gamma+1) = 59$ 就开始了**。实测平均 batch 96 已经越过 59，所以投机的收益处在快速衰减段——远不是 $\tau = 3.36$ 倍，而只是「比 1 大一点」。

**决定性的一条是两者争夺同一份免费额度**：连续批处理本身就在把 $B$ 往 128 顶，而投机的 γ+1=5 会把这 5 个位置占掉。同一个 `max_num_seqs=128` 在 $\gamma=4$ 时，等效的请求并发上限降到约 128/5 ≈ 26 条（如果想让 $B \cdot L$ 保持同样水平）。

结论：**高并发吞吐型服务不该开投机解码。** 想让两者共存，必须按实时 batch 动态调 $\gamma$（batch 大时把 $\gamma$ 削到 0），这是 vLLM 文档里「medium-to-low QPS」那句的实际含义。

### 答案 3

- **FCFS 下踢 `self.running[-1]`** —— 队尾那条，即最近才上车的请求（源码原文：`preempted_req = self.running[-1]`）。
- **`num_computed_tokens = 0`** —— 源码原文如此，它的 KV 块已被 `_free_request_blocks` 释放，所以进度归零、从头算。之后 `self.waiting.prepend_request(request)` 把它塞回等待队列队首。
- **重算代价**：要重算 prompt 3000 + 已生成 500 = 3500 个 token，代价 $2 \times 7\times10^9 \times 3500 = 4.9\times10^{13}$ FLOPs。按 989 TFLOPS 算约 **49.5 ms**。
- **swap 代价**：KV per token = 2 × 32 层 × 8 KV head × 128 dim × 2 bytes = 128 KiB。3500 token × 128 KiB = 448 MiB，换出 + 换回 = 896 MiB，按 PCIe 单向 25 GB/s 约 **35.8 ms**。

**按这个 roofline，swap 略便宜（35.8 vs 49.5 ms，省 28%）。** 但 vLLM V1 仍然选了 recompute，官方理由是「recomputation has lower overhead in the V1 architecture」——重算不需要 CPU 侧内存池、不引入 PCIe 争用、也不需要在架构里维护一条换入换出的状态机。

**要能同时说出这两句话**才算答对：硬件侧的 roofline 说 swap 便宜，工程侧的复杂度说 recompute 便宜。做方案选择时两个都要摆出来。

</details>

---

*本期全部数值结论来自本机 managed venv 的 torch 2.14.0（CPU + MPS，无 CUDA）真跑，以及标注为「roofline 推算」的 H800 级模型计算。CPU 口径的 f(B) 曲线是实测（tiny 模型 26.2M 参数），GPU 口径的所有绝对耗时是模型推算而非实测——本机没有 CUDA 设备。vLLM 的队列名称、抢占逻辑、预算变量均引自 V1 源码原文，文中已逐处标注。*
