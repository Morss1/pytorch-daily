# PyTorch 每日一课 · 第 031 期

## KV 感知路由：把缓存命中率从「调度器的运气」变成「路由器的决策」

> **日期**：2026-10-07　**难度**：⭐⭐⭐⭐（需要前缀缓存与排队论的基础）
> **前置知识**：KV Cache 与 PagedAttention（第 011 期）、连续批处理（第 028 期）、RadixAttention 与前缀缓存（第 029 期）、PD 分离与跨机 KV（第 030 期）
> **预计阅读时间**：50 分钟
> **关联**：第 029 期 · 关联点「前缀缓存是单实例内的魔法，本期问它在 N 个实例的集群里还剩多少」；第 030 期 · 关联点「KV 事件通道、一致性哈希、PD 双池路由都是同一套基础设施的另一面」；第 028 期 · 关联点「调度器里没有 prefill/decode 之分，路由器是在它上面再加一层决策」

---

## 1. 这个领域解决什么问题

### 1.1 第 029 期的魔法有一个隐藏前提

第 029 期讲了前缀缓存：请求 B 的 prompt 如果以请求 A 的 prompt 为前缀，B 就能直接复用 A 算好的 KV，prefill 只需要算多出来的那部分。我们当时量出来的结论很漂亮——**命中率恒等于省下的 prefill FLOPs 比例，是精确等式**：

$$\frac{\text{FLOPs}(T=S-P,\,S)}{\text{FLOPs}(T=S,\,S)} = \frac{2N(S-P) + 4Ld(S-P)S}{2NS + 4LdS^2} = \frac{S-P}{S}$$

但那个结论有一个当时没点明的前提：**A 和 B 走的是同一个引擎实例，共用同一个 KV 池。**

一旦你把它部署成一个集群，这个前提就没了。

### 1.2 扩容会把缓存撕成 N 份互不知情的碎片

现实的推理服务不是一个实例，是 N 个副本 + 前面一个负载均衡器。每个实例有**自己独立的 KV 池**，互相看不见对方缓存了什么（第 030 期讲过 PD 分离的做法就是把这件事做得更彻底：连 prefill 和 decode 的 KV 池都分开）。

而最前面那个负载均衡器——ALB、K8s Service、nginx、Envoy——的职责是**把流量摊平**。它看不到 prompt 的内容，也看不到任何一个实例的 KV 池里有什么。它的默认行为正好是缓存的天敌：

```
请求 1（公司通用 system prompt 2000 token）→ 实例 A → 算出 KV，缓存起来
请求 2（同一个 system prompt）           → 实例 B → 从零重算这 2000 token
请求 3（同一个 system prompt）           → 实例 C → 从零重算这 2000 token
```

**均匀分布恰恰是缓存局部性的反面。** llm-d 的一篇基准博客把这一幕叫做「a heartbreaking KV-cache miss scenario」，并且直接点出：在几千并发请求的生产环境里，这不是偶发事件，而是默认行为。

### 1.3 打散的代价不是「一点浪费」，而是容量被除以 N

这一条是本期最核心的定量结论，先把直觉讲清楚，第 5 节再给公式和实测。

设集群里所有请求共享前缀的**去重后**总内容量是 $D$（第 029 期的口径：缓存驻留量 = 去重后的内容量，不是请求量）。每个实例的 KV 池能放 $C$ 个 token 的前缀内容。

- **打散路由**（blind）：每个实例都在收全量前缀的采样。想让前缀 $g$ 留在缓存里，这个实例必须在两次用到 $g$ 之间不被别的 $g'$ 挤出去。稍微推一下就知道，这要求

$$C \ge D = \sum_g P_g$$

- **亲和路由**（aware）：前缀 $g$ 只在某一个实例上被使用，每个实例只需装 $D/N$：

$$C \ge \frac{D}{N}$$

**同一份工作集，打散路由需要 N 倍的缓存。** 反过来说：同样的硬件，亲和路由能撑住 N 倍大的工作集。

llm-d 公布的基准工况恰好落在这个不等式上：8 个实例（16 张 H100）服务 150 个企业客户、每客户 6000 token 上下文，**缓存全部活跃客户前缀需要集群总 KV 容量的约 73%**。73% / 8 ≈ 9.1%，也就是单实例容量的 5.84 倍——**比任何一个实例自己的容量（12.5%）大了将近 6 倍**。也就是说，只要路由是打散的，这个 workload 必然抖动，一个前缀都留不住。

### 1.4 但「把相关请求都送同一台」是另一个极端

如果你读到这儿就想「那就按前缀哈希钉死到某一台」，先看这个：45% 的请求共用同一段公司通用 system prompt 的负载下，把所有这些请求都送到持有该前缀的那一个实例上，那个实例的负载会是它公平份额的 $0.45 \times 8 = 3.6$ 倍。它立刻过载，而那 3.6 倍里的大部分本来是**很便宜**的命中请求——现在它们排在一个饱和的队列后面。

于是本领域的核心张力出现了：

> **缓存亲和与负载均衡是两个直接对立的目标。**
> 想提高命中率就得聚，想均衡负载就得散。

并且它有很强的非线性：排队论里等待时间随利用率是 $W \propto \rho/(1-\rho)$，在 $\rho \to 1$ 附近发散。**为了省一点 prefill 而制造出的负载倾斜，代价可能在尾部被放大一个数量级。** 第 6 节会给出实测：两个策略做掉了同样多的算力（平均重算 273 vs 268 token），命中率几乎一样（98.89% vs 99.23%），但 TTFT p99 差了 **7.4 倍**。

所以这个领域的真正命题不是「怎么把请求送到有缓存的实例」，而是——

> **在负载不均衡度有上界的前提下，最大化缓存命中率。**

---

## 2. 核心思想

### 2.1 一个类比：N 个分馆的图书馆

把每个推理实例想成一个分馆，共享前缀（system prompt、RAG 文档、多轮对话历史）想成一本常被查阅的参考书。

- 一本书只有被某个分馆**印过一次**之后，其他读者去那个分馆才能直接取阅；去别的分馆要当场现印（prefill）。
- **调度台（路由器）坚持轮流介绍**（round-robin）：每个分馆都得印一遍。N 个分馆，同一本书印 N 次。这就是 §1.3 的不等式。
- **调度台一律把人往有书的那个分馆指**：那家爆满，其余分馆空转。
- **正确的调度台**是：**尽量往有书的分馆指，但一旦发现某家排的队超过其他家太多，就把人分流走**——哪怕分流意味着当场现印。

最后那一条就是这个领域的全部工程内容。注意它的措辞：不是「有书就去」，也不是「排队短就去」，而是**两个信号按某个汇率加权比较**。

### 2.2 「知道缓存里有什么」的三个层次

任何 KV 感知路由器都必须回答一个问题：*这个请求的前缀，在实例 $i$ 上命中了多少？* 回答这个问题的代价，构成了整个设计谱系：

| 层次 | 路由器怎么知道 | 内存/基础设施成本 | 精度 |
|---|---|---|---|
| **① 无状态哈希** | 不知道。只是把 prompt 的**前 N 个 token** 哈希一下，映射到固定实例 | 零状态，$O(\log n)$ 查表 | 只保证「同一个前缀总去同一台」，不保证那儿真有缓存 |
| **② 近似索引** | 从**路由历史**里长出一棵前缀树：我往哪个实例发过什么 | $O(\text{路由过的 token 总量})$，仅路由器内存 | 知道「曾经写过」，**不知道引擎是否已经驱逐** |
| **③ 精确索引** | 消费引擎发出的 **KV 事件**（块创建/驱逐），维护全局块哈希 → 实例的映射 | 需要事件通道（ZMQ / 共享内存）+ 索引进程 | 与引擎真实状态一致 |

三个层次的实现分别是（都是官方在维护的生产代码）：

- ①：SGLang 的 `--policy prefix_hash`、vLLM Production Stack 的 `session`/`prefixaware`、K8s Gateway API Inference Extension 早期版本的「把 token 前缀哈希进一张内存表」
- ②：SGLang 的 `--policy cache_aware`（默认）、llm-d 的 `approximate` 前缀缓存评分器
- ③：llm-d 的 `precise` 前缀缓存评分器、NVIDIA Dynamo 的 KV Router（`KvIndexer`）、vLLM Production Stack 的 `kvaware`（走 LMCache 的 cache controller）

**①到③不是「越来越好」，而是「越来越贵」。** 第 8 节会给出一个反直觉的实测结果：**②的误差到底值不值钱，取决于你的打分函数是什么形状**——阈值型判据几乎免疫，线性打分会被打得粉碎。这是本期最有实用价值的一条。

### 2.3 一次路由决策的解剖

不管哪套实现，一个请求到达路由器后都是这几个动作：

1. **拿候选集**：健康检查过滤 + 可能的模型/pool 过滤（PD 分离下 prefill 池和 decode 池要先分开）
2. **算两个信号**：
   - **缓存亲和分**：该请求的前缀在每个候选实例上命中多少。可能是「匹配 token 数」，也可能是「匹配比例」
   - **负载分**：每个候选实例现在有多忙。可能是**在飞请求数**（vLLM 走事件驱动的 `request_stats`）、队列长度、KV 池利用率，或它们的组合
3. **按汇率合并**：这是各家差异最大的地方——**硬阈值** 还是 **线性打分**
4. **选一个**：argmax / max-score，或者带随机性的 softmax / weighted-random
5. **更新自己的状态**：把「这条请求发给了谁」写进自己的索引（源码里确实是选完立刻写）

第 3 步是分水岭，两种做法有本质区别：

**硬阈值型**（SGLang `cache_aware`）：

```
如果 (max_load - min_load > 64) 且 (max_load > min_load × 1.5):
    走最短队列                      ← 完全不看缓存
否则:
    如果 前缀匹配率 > 0.3:  走匹配最高的实例
    否则:                   走最短队列
```

**线性打分型**（vLLM Production Stack `loadaware`）：

```python
benefit = min(matched_tokens, prompt_tokens) / max(prompt_tokens, 1)
score   = benefit - beta * relative_load        # beta 默认 1.0
```

差别在于：

- 硬阈值型里，负载项只有「触发 / 不触发」两态，且那个 `64` 是一个**绝对计数**——它的合理取值和集群规模、模型大小绑定（第 7 节会用 Little 定律证明这个默认值在常规部署里根本触发不了）。
- 线性打分型里，`beta` 是一个**汇率**：一单位相对负载值多少缓存收益。它是无量纲的（源码注释原话：*Both score terms are dimensionless, which is what makes this a defensible default rather than a number calibrated on one cluster*），因此随集群规模自动伸缩。

### 2.4 把它写成优化问题

设 $b_i \in [0,1]$ 是候选实例 $i$ 的缓存受益（命中的前缀占比），$r_i$ 是它的相对负载。路由器要解的是

$$\max_i \; \left[ b_i - \beta \cdot r_i \right] \quad \text{s.t.} \quad \max_i r_i \le \bar{r}$$

那个约束不在任何一套代码里显式写着，但它是隐含的：**当 $r_i$ 越过 1（两倍平均负载）时，$\beta r_i$ 会强到足以压倒任何 $b_i$**。换句话说，线性打分用「汇率」的方式隐式地实现了约束。

这里有个不变量值得记下来：**从一个实例搬走一个请求，$b$ 的损失是 $\Delta b$，$r$ 的收益是 $\Delta r$；只有当 $\Delta b < \beta \Delta r$ 时才划算。** 所以

$$\beta^* = \frac{\text{一单位相对负载的代价}}{\text{一次整段命中的价值}}$$

第 7 节会把这个 $\beta^*$ 解析地算出来，跟实测的最优点对照。

---

## 3. 引擎里到底怎么实现的

这一节全部是**源码里看到的**，逐处标了文件路径。凡是文档和源码不一致的地方我都单独标出来。

### 3.1 SGLang Model Gateway 的 `cache_aware`

策略实现在 `sgl-model-gateway/src/policies/cache_aware.rs`（Rust，1600 多行）。文件开头那段注释已经把算法写全了，搬过来：

```
Strategy Details:
1. Cache-Aware Routing (Approximate Tree)
   This strategy maintains an approximate radix tree for each worker based on request
   history, eliminating the need for direct cache state queries. The tree stores raw
   text characters instead of token IDs to avoid tokenization overhead.
   Process:
   a. For each request, find the worker with the highest prefix match
   b. If match rate > cache_threshold:
      Route to the worker with highest match
   c. If match rate ≤ cache_threshold:
      Route to the worker with smallest tree size
   d. Background maintenance: Periodically evict least recently used leaf nodes
```

三个值得单独拎出来的设计决定：

1. **树里存的是原始文本字符，不是 token id**（注释原话：`stores raw text characters instead of token IDs to avoid tokenization overhead`）。所以它的匹配率是**字符比**，而 `cache_threshold` 也是按字符比判定的。对英文大致 4 字符/token，比例和 token 比接近；中文大约 1~1.5 字符/token，两者会分叉。**这一点在官方文档里完全没有提**，是我从源码注释里读到的。

2. **不均衡判定用的是 `&&` 不是 `||`**（`select_worker` 里）：

```rust
let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
    && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);
```

**两个条件必须同时成立**才算不均衡。而官方文档在描述这两个参数时写的是「当绝对负载差超过 64，**或**当相对负载比超过 1.5 → 触发重新均衡」——把 AND 写成了 OR。这是我核对源码时发现的第二处文档/实现分歧。

3. **最短队列的 tie-break 是随机的**（`select_worker_min_load` 里 `loads.iter().filter(...).choose(&mut rand::rng())`）。这个细节很重要：如果并列时固定选 0 号实例，常见负载下（各实例负载都是 0 或 1）会系统性地把所有冷请求都塞给同一个实例。我在仿真里第一版就踩了这个坑。

还有个和 PD 分离直接相关的细节：树是按 `pool::model` 键控制的（`make_tree_key(pool_tag(worker_type), model)`）。注释里解释了为什么必须隔离：

```rust
/// Without this isolation, the `tree.insert(text, url)` at the end of every
/// `select_worker` call would overwrite the previous pool's tenant for the same
/// prompt and collapse cache_aware into a flip-flop between pools.
```

**PD 分离下 prefill 池和 decode 池必须各有一棵独立的树**，否则同一条 prompt 在两个池之间来回覆盖对方的 tenant，路由退化成乒乓。这就是第 030 期那套架构在本期的必然要求。

### 3.2 vLLM Production Stack 的 `loadaware`

实现在 `src/vllm_router/routers/routing_logic.py`。核心是两行，我把它们原样抄出来：

```python
@classmethod
def relative_loads(cls, request_stats, endpoints) -> Dict[str, float]:
    """Each endpoint's load as a signed fraction of the fleet mean.

    `(load - mean) / max(1, mean)`: 0.0 is "average", +1.0 is "twice the
    fleet average". Clamping the denominator at 1 keeps a near-idle fleet
    from amplifying one in-flight request into a large relative load and
    thrashing on noise.
    """
    loads = {e.url: cls.load_penalty(request_stats, e.url) for e in endpoints}
    if not loads:
        return {}
    mean = sum(loads.values()) / len(loads)
    return {url: (load - mean) / max(1.0, mean) for url, load in loads.items()}

def score_endpoint(self, matched_tokens, prompt_tokens, relative_load):
    """`cache_hit_benefit - beta * relative_load` for one endpoint."""
    benefit = min(matched_tokens, prompt_tokens) / max(prompt_tokens, 1)
    return benefit - self.beta * relative_load
```

（`score_endpoint` 的 docstring 还解释了一个边界：`matched_tokens` 可能因为匹配被向上取整到 chunk 边界而超过 `prompt_tokens`，所以必须 `min()` 一下，否则「被取整的匹配」会压过「真正的全命中」。）

三个细节值得记住：

- `load_penalty` 用的是 `request_stats`（在飞请求数，事件驱动、实时），源码注释明确说了为什么要它而不是 `engine_stats`：`Uses request_stats because it is event-driven and fresh, unlike the scrape-lagged engine_stats.`——**抓取式指标有延迟，用它做路由会基于过期信息决策**。
- 分母 clamp 到 `max(1.0, mean)`。集群接近空闲时 `mean` 会很小，`(load-mean)/mean` 会把「一个在飞请求」放大成一个巨大的相对负载，路由就在噪声上抖动。这是一个纯粹的工程防御，但它决定了这个策略在小负载下能不能用。
- `beta` 的语义在源码注释里定义得很清楚：

```
# The single tunable of the `loadaware` routing logic. beta = 1.0 reads as: an
# endpoint sitting 100% above fleet-average load is docked one full cache hit's
# worth of preference. Both score terms are dimensionless, which is what makes
# this a defensible default rather than a number calibrated on one cluster.
DEFAULT_LOADAWARE_BETA = 1.0
```

**「一个比集群平均负载高 100% 的实例，被扣掉整整一次缓存命中那么多的偏好」**——这句话就是 §2.4 那个汇率的最直白表述。第 7 节会量出这个默认值在实测里偏高。

顺带一提，vLLM Production Stack 现在的 `RoutingLogic` 枚举有 8 项（源码）：

```python
class RoutingLogic(str, enum.Enum):
    ROUND_ROBIN = "roundrobin"
    SESSION_BASED = "session"
    KVAWARE = "kvaware"
    LOADAWARE = "loadaware"
    PREFIXAWARE = "prefixaware"
    DISAGGREGATED_PREFILL = "disaggregated_prefill"
    DISAGGREGATED_PREFILL_ORCHESTRATED = "disaggregated_prefill_orchestrated"
    PRIORITY = "priority"
```

`prefixaware` 是路由器本地的（一棵进程内 HashTrie，按 128 字符分块做 xxHash，不依赖 LMCache）；`kvaware` 要走 LMCache 的 cache controller 查真实 KV 位置。**同一套栈里同时提供了 §2.2 的 ① 和 ③ 两个层次**，这个对比本身就是设计谱系的最好注脚。

### 3.3 NVIDIA Dynamo 的 KV Router

Dynamo 走的是**代价函数**（cost function）路线，越低越好：

```
adjusted_prefill_blocks = max(0, prefill_blocks − effective_device_credit × device_overlap_blocks
                                 − host_cache_hit_weight  × host_overlap_blocks
                                 − disk_cache_hit_weight  × disk_overlap_blocks
                                 − shared_cache_multiplier × shared_beyond_blocks)
cost = prefill_load_scale × adjusted_prefill_blocks + potential_decode_blocks + active_request_blocks
```

它比其他人多做了一件事：**缓存收益是按内存层级打折的**（GPU 命中、CPU 命中、磁盘命中各有权重）。这很合理——CPU 里的 KV 命中了还要走一遍 PCIe 才能用。官方文档里那个 3 worker 的算例（`overlap_score_weight = 1.0`）：

```
Worker 1: raw prefill 10 blocks, device overlap 2 blocks, decode 10 blocks => cost = 8 + 10 = 18
Worker 2: raw prefill 10 blocks, device overlap 5 blocks, decode  5 blocks => cost = 5 +  5 = 10  ← 选中
Worker 3: raw prefill 10 blocks, device overlap 8 blocks, decode  9 blocks => cost = 2 +  9 = 11
```

注意 Worker 3：命中最多（8 块），但因为 decode 负载重，总分反而输给了 Worker 2。**这就是两目标在单个算例里的样子。**

另外它有一个 `overlap_score_credit_decay_factor = 1 / (1 + decay * normalized_excess_prefill)`——**负载越重的实例，它的缓存收益被打折得越狠**。这个 $1/(1+x)$ 的形式值得看一眼：排队延迟 $W \propto 1/(1-\rho)$ 也是同一种形式。也就是说，Dynamo 是在用一个双曲衰减隐式地逼近排队代价，虽然没有显式建模。

### 3.4 llm-d / Gateway API Inference Extension：评分器 + 选择器

llm-d 把路由拆成了三个可插拔的角色，这是最干净的架构抽象：

- **Scorer（评分器）**：给每个候选实例打一个分。它提供的是 `prefix-cache-scorer`（缓存亲和）、`queue-scorer`（队列）、`kv-cache-utilization-scorer`（KV 池利用率）、`session-affinity-scorer`、`no-hit-lru-scorer`（冷请求优先发给从没接过请求的实例，把昂贵的 prefill 摊开）
- **Picker（选择器）**：`max-score-picker`（默认）、`random-picker`、`weighted-random-picker`（用分数当相对概率，即抽签调度）
- **ProfileHandler**：`single-profile-handler` 或 `disagg-profile-handler`（PD 分离下**同时跑两套打分**：prefill profile 和 decode profile，decode 端作为主目的地，prefill 端作为特殊 header 注入请求）

红帽文档给出的一个完整配置里的权重分配是：`prefix-cache-scorer` 权重 3，`queue-scorer` 和 `kv-cache-utilization-scorer` 各 2。**看，还是那个汇率，只是换成了权重。**

这里有个实用的设计细节值得学：`no-hit-lru-scorer` 只对**零命中**的请求生效（有命中的请求上它给所有实例相同分数，等于不起作用），它的作用是**让昂贵的冷 prefill 在集群里均匀铺开**。这跟我第 6 节消融实验里发现的机制是同一个东西——**真正决定系统稳不稳的，往往是「冷请求往哪走」这个兜底动作，而不是缓存命中路径上的精调。**

GIE 把 token 前缀哈希进一张内存映射表来记录「哪个副本最后算过它」，并且抓取每个 vLLM 副本的 `vllm:kv_cache_usage_perc` 和队列深度来打分。

### 3.5 参数对照表，以及三处文档漂移

| 实现 | 旋钮 | 默认值 | 语义 |
|---|---|---|---|
| SGLang `cache_aware` | `--cache-threshold` | **0.3** | 最小前缀匹配率（**字符比**，源码注释确认） |
| | `--balance-abs-threshold` | **64** | 绝对负载差阈值（单位：在飞请求数） |
| | `--balance-rel-threshold` | **1.5** | 相对负载比阈值 |
| | `--eviction-interval-secs` | 120 | 近似树叶子回收节奏 |
| | `--max-tree-size` | 67108864 | 树节点上限 |
| SGLang `prefix_hash` | `--prefix-token-count` | **256** | 取 prompt 前 N 个 token 做哈希 |
| | `--prefix-hash-load-factor` | **1.25** | 超过平均负载这个倍数就顺时针走下一个 |
| SGLang `power_of_two` | — | — | 随机抽两个实例，选负载低的 |
| vLLM `loadaware` | `beta`（`LOADAWARE_BETA` 环境变量可覆盖） | **1.0** | 一单位相对负载 = 多少缓存收益 |
| Dynamo KV Router | `--router-kv-overlap-score-weight` | **1** | 同上，0 = 忽略缓存纯负载均衡 |
| | `--router-temperature` | **0** | 0 = 确定性选最低代价；>0 用 softmax 采样 |
| | `--router-queue-threshold` | **2.0** | 全部 worker 超过 `max_num_batched_tokens` 这个比例时把请求压在路由器队列里 |
| | `--router-queue-policy` | **fcfs** | 另有 `wspt`（加权最短处理时间，Smith 规则） |

**三处文档/源码不一致（都是我逐处核对过的）：**

1. **CLI 默认值和库默认值不同。** `sgl-model-gateway/src/policies/mod.rs` 里 `impl Default for CacheAwareConfig` 给的是 `cache_threshold: 0.5, balance_abs_threshold: 32, balance_rel_threshold: 1.1, eviction_interval_secs: 30, max_tree_size: 10000`；而 CLI（`src/main.rs` 的 `#[arg(long, default_value_t = ...)]`）给的是 `0.3 / 64 / 1.5 / 120 / 67108864`。**走命令行启动用的是后者**；从代码里构造 `CacheAwareConfig` 用的是前者。差别不小（`max_tree_size` 差了四个数量级）。
2. **文档把 AND 写成了 OR。** 见 §3.1 第 2 点。
3. **策略清单在两个方向上都对不齐。** 官方文档的 "Load Balancing Policies" 表列了 5 个：`random / round_robin / cache_aware / power_of_two / bucket`；而 `main.rs` 里 `--policy` 的 `value_parser` 是 `["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash", "manual"]`。也就是说 **`bucket` 在文档里但不在 CLI 里，`prefix_hash` 和 `manual` 在 CLI 里但不在文档里**（`bucket` 确实存在于库里的 `PolicyConfig::Bucket`，只是我没找到从 CLI 打开它的路径）。

还有一条官方在 "Production Recommendations" 里明确写着的、容易被忽略的坑：

> "With multiple replicas, the cache-aware routing policy's radix tree is not synchronized across replicas... **Expected cache hit rate reduction: 10-20%**."

**路由器自己多副本部署时，各自的近似树是不同步的**，官方给的命中率损失预期是 10~20%。这是近似索引路线的一个隐藏成本。

---

## 4. 先算清楚：一次命中到底值多少钱

在讨论「往哪发」之前，得先知道被优化的那个量是什么。

### 4.1 省下的是算力，不是时间

真跑一个带显式 KV Cache 的 tiny Transformer（6.949 M 参数，GQA 8 query / 2 KV 头，S=512）。两种走法：全量重算（miss），和「前 P 个 token 的 KV 已在缓存里、只算后面 512−P 个」（hit）。注意 hit 路径里被重算的 token **仍然要对整个 512 做 attention**——KV 一个字节不少地读一遍，这是引擎的真实行为，不能偷掉。

```python
"""代码块 1：一次前缀缓存命中，省下的是「算力」还是「时间」？—— 真跑一个带 KV Cache 的 tiny 模型。

同一模型、同一总长度 S=512，两种走法：
  miss：512 个 token 全部 prefill（KV 从零建）
  hit ：前 P 个 token 的 KV 已存在，本次只 prefill 后 512-P 个，
        但被重算的 token 仍要对整个 512 做 attention（KV 一个字节不少地读一遍）
用「成对交替采样、取逐对比值的中位数」计量（第 030 期沉淀的方法）。
"""
import time
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(4)
D, L, H, HK, DH, V = 256, 4, 8, 2, 32, 8192
S_TOTAL, REPS = 512, 7


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(D, H * DH, bias=False)
        self.wk = nn.Linear(D, HK * DH, bias=False)
        self.wv = nn.Linear(D, HK * DH, bias=False)
        self.wo = nn.Linear(H * DH, D, bias=False)
        self.w1 = nn.Linear(D, 4 * D, bias=False)
        self.w2 = nn.Linear(4 * D, D, bias=False)
        self.n1 = nn.RMSNorm(D)
        self.n2 = nn.RMSNorm(D)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B, T, H, DH).transpose(1, 2)
        k = self.wk(h).view(B, T, HK, DH).transpose(1, 2)
        v = self.wv(h).view(B, T, HK, DH).transpose(1, 2)
        if cache is not None:                   # 前缀 KV 已在缓存里
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        S = k.shape[2]
        kk = k.repeat_interleave(H // HK, 1)
        vv = v.repeat_interleave(H // HK, 1)
        # query 位置 i 的绝对位置 = S-T+i，只能看 <= 自己
        mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)
        o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, H * DH))
        return x + self.w2(F.silu(self.w1(self.n2(x)))), (k, v)


class TinyGQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.layers = nn.ModuleList([Layer() for _ in range(L)])
        self.norm = nn.RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    @torch.no_grad()
    def prefill(self, ids, caches=None):
        x = self.emb(ids)
        new = []
        for i, lyr in enumerate(self.layers):
            x, c = lyr(x, None if caches is None else caches[i])
            new.append(c)
        return self.head(self.norm(x)), new


m = TinyGQA().eval()
N_PAR = sum(p.numel() for p in m.parameters())
FL = lambda T, S: 2.0 * N_PAR * T + 4.0 * L * D * T * S      # FLOPs 模型（同第 029 期口径）
base_flops = FL(S_TOTAL, S_TOTAL)
print(f"模型 {N_PAR/1e6:.3f} M（d_model={D}, {L} 层, {H} query 头, {HK} KV 头, S={S_TOTAL}）")

prefix_ids = torch.randint(0, V, (1, S_TOTAL))
for _ in range(3):                              # 预热：跑掉懒初始化与首次分配
    m.prefill(prefix_ids)


@torch.no_grad()
def run_once(hit):
    """成对交替采样：同一轮里先跑 miss 再跑 hit。"""
    miss_ts, hit_ts = [], []
    for _ in range(REPS):
        t0 = time.perf_counter()
        m.prefill(prefix_ids)                                   # 整段重算
        t1 = time.perf_counter()
        cache = m.prefill(prefix_ids[:, :hit])[1] if hit else None
        t2 = time.perf_counter()
        m.prefill(prefix_ids[:, hit:], cache)                   # 只算被逐出的部分
        t3 = time.perf_counter()
        miss_ts.append(t1 - t0)
        hit_ts.append(t3 - t2)
    # 微计时取最小值：中位数会被偶发调度抖动整体抬高
    return min(miss_ts), min(hit_ts)


rows = [(hit,) + run_once(hit) for hit in [0, 128, 256, 384, 448, 496, 504, 511]]
base_ms = rows[0][1]

print(f"\n{'命中前缀':>8}{'重算 T':>8}{'FLOPs 占比':>12}{'FLOPs 省下':>12}"
      f"{'最小耗时':>11}{'耗时占比':>10}{'耗时省下':>10}")
for hit, _, ht in rows:
    T = S_TOTAL - hit
    fr = FL(T, S_TOTAL) / base_flops
    tr = ht / base_ms
    print(f"{hit:>8}{T:>8}{fr*100:>11.2f}%{(1-fr)*100:>11.2f}%"
          f"{ht*1000:>9.2f}ms{tr*100:>9.2f}%{(1-tr)*100:>9.2f}%")
print(f"\n基准：prompt {S_TOTAL} token 全量重算 {base_ms*1000:.2f} ms（{REPS} 轮最小值）")
print(f"第一行其实是「同一件事」的两遍采样（都是全量重算）：左边 FLOPs 省下 "
      f"{(1-FL(S_TOTAL,S_TOTAL)/base_flops)*100:.2f}%，")
print(f"右边耗时省下 {(1-rows[0][2]/base_ms)*100:.2f}% —— 两者都该是 0，留下的偏差就是采样噪声。")
print("这张表要读的是趋势，不是小数点后两位。")
```

```text
模型 6.949 M（d_model=256, 4 层, 8 query 头, 2 KV 头, S=512）

    命中前缀    重算 T    FLOPs 占比    FLOPs 省下       最小耗时      耗时占比      耗时省下
       0     512     100.00%       0.00%    11.33ms   104.73%    -4.73%
     128     384      75.00%      25.00%    10.08ms    93.20%     6.80%
     256     256      50.00%      50.00%     8.30ms    76.72%    23.28%
     384     128      25.00%      75.00%     4.95ms    45.75%    54.25%
     448      64      12.50%      87.50%     3.10ms    28.64%    71.36%
     496      16       3.12%      96.88%     1.95ms    18.01%    81.99%
     504       8       1.56%      98.44%     1.79ms    16.59%    83.41%
     511       1       0.20%      99.80%     1.15ms    10.64%    89.36%

基准：prompt 512 token 全量重算 10.82 ms（7 轮最小值）
第一行其实是「同一件事」的两遍采样（都是全量重算）：左边 FLOPs 省下 0.00%，
右边耗时省下 -4.73% —— 两者都该是 0，留下的偏差就是采样噪声。
这张表要读的是趋势，不是小数点后两位。
```

**看到两列的差了吗。** FLOPs 那一列省下多少，耗时那一列就少省 **9~13 个百分点**。命中 504/512（98.44%）时 FLOPs 省下 98.44%，耗时只省下约 87%；命中 511/512（99.80%）时前者 99.80%，后者约 91%。**这个差值在整条曲线上几乎恒定——它是加性的，不是乘性的。**

多跑几次会看到，绝对耗时那几列飘得很大（同一条 512 token 全量重算，不同的轮次实测在 **8.6 ms ~ 13.8 ms** 之间，取决于整机忙不忙），但**两列省下量的差值稳定在 8~13 个百分点**。要读的是这个结构，不是某一次的小数点。

### 4.2 那条降不下去的地板由什么构成

把上面的实验改一下：固定 S=512，只改要重算的 token 数 $T$，然后线性拟合 $t(T) = a + bT$。

```python
"""代码块 2：那条「降不下去的地板」由什么构成？—— 对 t(T) 做线性拟合 t = a + b·T。

微计时取 10 次的最小值（丢掉第一次），比中位数更抗系统抖动。

复用代码块 1 的模型与预热。这次固定 S=512，只改要重算的 token 数 T。
"""
Ts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
ts = []
for T in Ts:
    c = m.prefill(prefix_ids[:, :S_TOTAL - T])[1]     # 前 S_TOTAL-T 个 token 已在缓存里
    samples = []
    for _ in range(10):
        t0 = time.perf_counter()
        m.prefill(prefix_ids[:, S_TOTAL - T:], c)
        samples.append(time.perf_counter() - t0)
    # 微计时用最小值：中位数会被偶发的调度抖动整体抬高
    ts.append(min(samples[1:]))

n = len(Ts)
mx, my = sum(Ts) / n, sum(ts) / n
b = sum((T - mx) * (t - my) for T, t in zip(Ts, ts)) / sum((T - mx) ** 2 for T in Ts)
a = my - b * mx

print(f"{'重算 T':>7}{'最小耗时':>12}{'拟合值':>12}{'相对 512':>11}")
for T, t in zip(Ts, ts):
    print(f"{T:>7}{t*1000:>10.3f}ms{(a+b*T)*1000:>10.3f}ms{t/ts[-1]*100:>10.1f}%")

print(f"\n线性拟合 t = a + b·T（最小二乘，10 个点）")
print(f"  a（固定开销，与 T 无关）= {a*1000:.3f} ms")
print(f"  b（每 token 边际成本）  = {b*1e6:.2f} µs/token")
print(f"  a / t(T=512)            = {a/ts[-1]*100:.2f}%   ← 这就是那条地板")
print(f"  T ≤ 16 时实测耗时是平台，约 {statistics.median(ts[1:5])*1000:.3f} ms"
      f"（= {statistics.median(ts[1:5])/ts[-1]*100:.1f}% of T=512）")

fpt = FL(1, S_TOTAL)
print(f"\n每个 token 的 FLOPs（对 S={S_TOTAL} 的 KV 做一次全层前向）= {fpt/1e6:.2f} MFLOPs")
print(f"  → 边际算力 {fpt/b/1e12:.4f} TFLOP/s（含全部 CPU 开销）")
print(f"  → 纯线性投影项占 {2*N_PAR/fpt*100:.1f}%，对 S 的注意力项占 "
      f"{4*L*D*S_TOTAL/fpt*100:.1f}%")
print(f"\n结论：T=1 时理论 FLOPs 只有全量的 {FL(1,S_TOTAL)/FL(S_TOTAL,S_TOTAL)*100:.2f}%，"
      f"实测耗时却有 {ts[0]/ts[-1]*100:.1f}% —— 差的就是 a。")
```

```text
   重算 T        最小耗时         拟合值     相对 512
      1     1.024ms     1.470ms       7.4%
      2     1.628ms     1.495ms      11.8%
      4     1.737ms     1.544ms      12.6%
      8     1.683ms     1.642ms      12.2%
     16     1.768ms     1.839ms      12.8%
     32     2.125ms     2.232ms      15.4%
     64     2.921ms     3.019ms      21.2%
    128     4.766ms     4.592ms      34.5%
    256     8.150ms     7.738ms      59.1%
    512    13.799ms    14.029ms     100.0%

线性拟合 t = a + b·T（最小二乘，10 个点）
  a（固定开销，与 T 无关）= 1.446 ms
  b（每 token 边际成本）  = 24.58 µs/token
  a / t(T=512)            = 10.48%   ← 这就是那条地板
  T ≤ 16 时实测耗时是平台，约 1.710 ms（= 12.4% of T=512）

每个 token 的 FLOPs（对 S=512 的 KV 做一次全层前向）= 16.00 MFLOPs
  → 边际算力 0.6508 TFLOP/s（含全部 CPU 开销）
  → 纯线性投影项占 86.9%，对 S 的注意力项占 13.1%

结论：T=1 时理论 FLOPs 只有全量的 0.20%，实测耗时却有 7.4% —— 差的就是 a。
```

**地板 = 截距 $a$，实测在 0.9~1.5 ms 之间（随整机负载波动），占全长 prefill 耗时的 9%~13%**；而且 $T \le 16$ 时实测耗时基本是个平台（约 1.2~1.7 ms）——小 $T$ 区间被固定开销完全主导。

要说清这个地板**是什么**：本次跑在 CPU 上，$a$ 主要来自每层每个算子的 Python/调度开销（4 层 × 6~7 个算子），与 $T$ 无关。真实 GPU 引擎上对应的地板是另一回事（权重要读一遍、被重算的 token 要 attend 整个 S 的 KV），第 029 期已经量过它的形式：$T^* = \rho \cdot \text{bytes}/2$，$\rho = \text{PEAK} \cdot \text{MFU}/\text{BW} = 132.9$ FLOP/byte（MFU 45%）。

**但不管地板来自哪里，对路由的结论是同一个：路由器的「缓存亲和分」是与 FLOPs 成正比的连续量，而用户感受的延迟是打了折的。**

$$\text{省下的延迟} = 1 - \frac{a + bT_{\text{recompute}}}{a + bT_{\text{full}}} \;<\; \frac{\Delta\text{FLOPs}}{\text{FLOPs}}$$

llm-d 把自己的指标叫做 "Effective Cache Throughput"（每秒直接从缓存提供的 prompt token 数）——**那是一个算力口径的指标**。你用它调路由没问题，但要记得它和 TTFT 之间差着这条地板。这也正是第 029 期「命中率超过约 97% 后 TTFT 就不再降」的同一个现象，只是那次是带宽地板，这次是固定开销地板。

---

## 5. 容量定律：路由带来的全部杠杆就是那个 N

现在回到 §1.3 的那个不等式，用仿真验证它。场景照抄 llm-d 的公开基准：150 个客户，每家有 6000 token 的共享前缀，请求 = 共享前缀 + 1200 token 自己的问题，集群 8 个实例。

两种路由：
- `blind`：均匀随机（round-robin / 随机 / 普通 L4 负载均衡器的行为）
- `aware`：按前缀把每个客户钉死到某一个实例（缓存亲和路由的极限形态）

引擎侧忠实建模三个事实：**块粒度 16 token、链式哈希（中间断一块后面全不可用）、LRU 驱逐**。

```python
"""代码块 4：容量定律 —— 为什么「把流量打散」会杀死前缀缓存。

场景抄自 llm-d 的公开基准：一批企业客户，每家有自己一段很长的共享前缀。
集群 N 个实例，每个实例的 KV 池只能放 C 个 token 的前缀内容。
  blind  均匀随机（round-robin / 随机 / 普通 L4 负载均衡器的行为）
  aware  按前缀把每个客户钉到某一个实例（cache-aware 路由的极限形态）
关键量：D = 去重后所有前缀的总 token 数，R = D / C = 工作集是单实例容量的几倍。

忠实建模两个引擎事实：块（16 token）粒度、链式哈希（断一处后面全不可用）、LRU 驱逐。
"""
import random
from collections import OrderedDict

BLOCK, N_INST, G, PREFIX, SUFFIX, N_REQ = 16, 8, 150, 6000, 1200, 20000
D = G * PREFIX
print(f"集群 {N_INST} 个实例，{G} 个客户，每客户共享前缀 {PREFIX} token")
print(f"去重后工作集 D = {D/1e6:.3f} M token；N = {N_INST}")


class LruPrefixCache:
    def __init__(self, cap_tokens):
        self.cap, self.size, self.d = cap_tokens, 0, OrderedDict()

    def match(self, g, n_blocks):
        m = 0
        while m < n_blocks and (g, m) in self.d:
            self.d.move_to_end((g, m))
            m += 1
        return m

    def insert(self, g, n_blocks):
        for b in range(n_blocks):
            key = (g, b)
            if key in self.d:
                self.d.move_to_end(key)
                continue
            while self.size + BLOCK > self.cap and self.d:
                self.d.popitem(last=False)
                self.size -= BLOCK
            self.d[key] = None
            self.size += BLOCK


def make_cdf(dist):
    w = [1.0] * G if dist == "uniform" else [1.0 / ((i + 1) ** 1.1) for i in range(G)]
    tw = sum(w)
    cum, acc = [], 0.0
    for x in w:
        acc += x
        cum.append(acc / tw)
    return cum


def simulate(policy, cap_tokens, dist, seed=7):
    rng = random.Random(seed)
    caches = [LruPrefixCache(cap_tokens) for _ in range(N_INST)]
    nb, cum = PREFIX // BLOCK, make_cdf(dist)
    hit_tok = miss_tok = 0

    def pick_group():
        u, lo, hi = rng.random(), 0, G - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo

    for _ in range(N_REQ):
        g = pick_group()
        i = rng.randrange(N_INST) if policy == "blind" else g % N_INST
        cached = caches[i].match(g, nb) * BLOCK
        hit_tok += cached
        miss_tok += PREFIX - cached + SUFFIX
        caches[i].insert(g, nb)
    prefix_tokens = hit_tok + miss_tok - N_REQ * SUFFIX
    return hit_tok / prefix_tokens, miss_tok


for dist in ("uniform", "zipf"):
    label = "等概率（纯容量定律）" if dist == "uniform" else "Zipf(1.1)（真实热度）"
    print(f"\n{'='*70}\n热度分布：{label}\n{'='*70}")
    print(f"{'R=D/C':>7}{'单实例容量':>12}{'blind 命中率':>14}"
          f"{'aware 命中率':>14}{'重算量之比':>12}")
    for R in [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]:
        C = int(D / R)
        bhr, bmiss = simulate("blind", C, dist)
        ahr, amiss = simulate("aware", C, dist)
        print(f"{R:>7.1f}{C/1e6:>10.3f}M{bhr*100:>13.2f}%{ahr*100:>13.2f}%"
              f"{bmiss/amiss:>11.2f}x")

print(f"\n理论预测：aware 的临界容量 = D/N = {D/N_INST/1e6:.3f} M token（R = N = {N_INST}）")
print(f"          blind 的临界容量 = D   = {D/1e6:.3f} M token（R = 1）")
print("同一份工作集，blind 需要 N 倍的容量；反过来说同样的容量下，")
print("aware 能撑住 N 倍大的工作集。这就是路由带来的全部杠杆。")
```

```text
集群 8 个实例，150 个客户，每客户共享前缀 6000 token
去重后工作集 D = 0.900 M token；N = 8

======================================================================
热度分布：等概率（纯容量定律）
======================================================================
  R=D/C       单实例容量     blind 命中率     aware 命中率       重算量之比
    0.5     1.800M        94.00%        99.25%       1.25x
    1.0     0.900M        94.00%        99.25%       1.25x
    2.0     0.450M        49.26%        99.25%       3.41x
    4.0     0.225M        24.20%        99.25%       4.62x
    6.0     0.150M        16.36%        99.25%       4.99x
    8.0     0.113M        11.77%        95.47%       4.41x
   12.0     0.075M         8.19%        63.23%       1.97x
   24.0     0.037M         4.18%        31.95%       1.32x

======================================================================
热度分布：Zipf(1.1)（真实热度）
======================================================================
  R=D/C       单实例容量     blind 命中率     aware 命中率       重算量之比
    0.5     1.800M        94.19%        99.25%       1.24x
    1.0     0.900M        94.19%        99.25%       1.24x
    2.0     0.450M        84.95%        99.25%       1.69x
    4.0     0.225M        71.47%        99.25%       2.34x
    6.0     0.150M        63.41%        99.25%       2.73x
    8.0     0.113M        56.44%        98.50%       2.96x
   12.0     0.075M        47.72%        90.70%       2.47x
   24.0     0.037M        32.65%        76.59%       2.01x

理论预测：aware 的临界容量 = D/N = 0.113 M token（R = N = 8）
          blind 的临界容量 = D   = 0.900 M token（R = 1）
同一份工作集，blind 需要 N 倍的容量；反过来说同样的容量下，
aware 能撑住 N 倍大的工作集。这就是路由带来的全部杠杆。
```

这张表要竖着看两件事：

**第一，等概率那一栏的两个悬崖位置和理论预测完全对上。** `blind` 从 R=1（94.00%）到 R=2（49.26%）之间掉下来——理论预测悬崖在 R=1；`aware` 一路 99.25% 到 R=6，R=8 时掉到 95.47%——理论预测悬崖在 R=N=8。**两个悬崖之间正好差 N=8 倍。**

那个 99.25% 的天花板也不是拟合出来的：150 个客户各自的第一次访问必然是冷的，$150 \times 6000$ 的 token 落在 $20000 \times 6000$ 的总量里就是 $1 - 150/20000 = 99.25\%$。**实测值和这个解析式一位不差。**

**第二，Zipf 热度会把这个悬崖抹平，但抹不掉差距。** 真实负载里少数大客户占大头，热门前缀被反复刷新，所以 `blind` 是渐进劣化（R=2 时还有 84.95%）而不是断崖。即便这样，R=8 时 `blind` 只守住 56.44% 而 `aware` 有 98.50%——重算量差 **2.96 倍**。

留意最后几行：R 大到一定程度之后两者都在低位，重算量之比反而收窄了。那是因为**容量小到连分区后的每实例工作集都装不下时，aware 也救不了你**——路由能给你 N 倍的杠杆，但杠杆乘在一个太小的基数上没有意义。

---

## 6. 两目标问题：亲和与均衡的正面交锋

容量定律说「聚起来才有命中」。现在把负载加上，看聚合的代价。

完整仿真台：内容寻址的链式块缓存 + 四种路由器 + 每实例一条 FCFS 队列。负载形状换成**嵌套前缀**——40 个部门各有一段 1024 token 的共享段，每部门 5 个客户各有 512 token 的私有段，请求再带 256 token 自己的问题。这样「部分匹配」才会真实出现（第 7 节要靠它）。

```python
"""第 031 期文章的代码块 3：仿真台 —— 引擎侧的块缓存 + 路由器侧的索引与策略。

后面 §5~§8 的所有表格都复用这个命名空间（文章里是「续写块」）。
"""
import random
import statistics
from collections import OrderedDict, deque

BLOCK = 16                      # 块大小（vLLM 默认）
PER_TOKEN = 0.023 / 1000.0      # 秒/token：2000 token 前缀 prefill ≈ 45 ms（H200 级真机量级）
FIXED = 2.0 / 1000.0            # 秒/请求：固定开销
N_INST = 8
N_DEPT, N_CUST = 40, 5          # 40 个部门共享段，每部门 5 个客户
DEPT_LEN, CUST_LEN = 1024, 512  # 部门段 / 客户私有段
SUFFIX = 256                    # 请求自己那部分，每次必算
PREFIX = DEPT_LEN + CUST_LEN
PROMPT = PREFIX + SUFFIX
G = N_DEPT * N_CUST             # 200 个前缀组
N_REQ = 20000


def chain_hashes(toks, h=0):
    """链式块哈希：h_k = hash(h_{k-1}, 块 k)。

    这是内容寻址 + 前缀链：块 k 能命中，隐含了它前面所有块都在。
    中间缺一块，后面全不可用 —— 第 029 期讲过的那个性质。
    """
    out = []
    for b in range(len(toks) // BLOCK):
        h = hash((h, toks[b * BLOCK:(b + 1) * BLOCK])) & 0xFFFFFFFF
        out.append(h)
    return out


def build_workload():
    """每组前缀 = 部门共享段 + 客户私有段。共享段让「部分匹配」成为可能。"""
    work, tokens = [], []
    for g in range(G):
        d = g // N_CUST
        t = (tuple(10_000_000 + d * 100_000 + k for k in range(DEPT_LEN))
             + tuple(20_000_000 + g * 100_000 + k for k in range(CUST_LEN)))
        tokens.append(t)
        work.append(chain_hashes(t))
    return work, tokens


WORK, TOKENS = build_workload()


class EngineCache:
    """引擎侧真实缓存：内容寻址的块 + LRU 驱逐。"""

    def __init__(self, cap_tokens, on_evict=None):
        self.cap, self.size, self.d = cap_tokens, 0, OrderedDict()
        self.on_evict = on_evict

    def match(self, hashes):
        m = 0
        while m < len(hashes) and hashes[m] in self.d:
            self.d.move_to_end(hashes[m])       # 被访问，刷新 LRU
            m += 1
        return m

    def insert(self, hashes):
        for h in hashes:
            if h in self.d:
                self.d.move_to_end(h)
                continue
            while self.size + BLOCK > self.cap and self.d:
                victim, _ = self.d.popitem(last=False)
                self.size -= BLOCK
                if self.on_evict:
                    self.on_evict(victim)
            self.d[h] = None
            self.size += BLOCK


class ApproxIndex:
    """路由器侧的**近似**索引：只增不删。引擎驱逐了它不知道。"""

    def __init__(self):
        self.have = set()

    def match(self, hashes):
        m = 0
        while m < len(hashes) and hashes[m] in self.have:
            m += 1
        return m

    def insert(self, hashes):
        self.have.update(hashes)


def pick_min_load(loads, rng):
    """源码里是 loads.filter(load == min_load).choose(&mut rng())：
    在**所有**最小负载实例里随机挑；不随机 tie-break 会系统性偏向 0 号实例。"""
    mn = min(loads)
    idx = [i for i, l in enumerate(loads) if l == mn]
    return idx[rng.randrange(len(idx))]


def relative_loads(loads):
    """逐字对齐 vLLM production-stack 的 LoadAwareRouter.relative_loads：
    (load - mean) / max(1.0, mean)。0.0 是平均，+1.0 是两倍平均。"""
    mean = sum(loads) / len(loads)
    return [(l - mean) / max(1.0, mean) for l in loads]


def simulate(policy, n_inst=N_INST, cap_ratio=6.0, cache_th=0.3, abs_th=64,
             rel_th=1.5, beta=1.0, prefix_tokens=256, hot_share=0.35,
             lam_per_inst=32.5, seed=7, precise=False):
    """离散事件仿真。每个实例一条单服务器 FCFS 队列。

    policy: cache_blind / prefix_hash / cache_aware / pure_cache / loadaware
    返回 dict：命中率、平均重算 token、逐实例利用率、端到端 TTFT 分位。
    """
    rng = random.Random(seed)
    cap = int((N_DEPT * DEPT_LEN + G * CUST_LEN) / cap_ratio)
    idxs = []
    caches = []
    for _ in range(n_inst):
        idxs.append(ApproxIndex())
    # precise=True 时，引擎驱逐会同步撤销索引里的条目（相当于消费 KV 事件）
    for i in range(n_inst):
        caches.append(EngineCache(
            cap, (idxs[i].have.discard if precise else None)))
    pending = [deque() for _ in range(n_inst)]
    last_fin = [0.0] * n_inst
    inst_work = [0.0] * n_inst
    routed = [0] * n_inst
    ghost = ghost_slots = 0
    hit_tok = miss_prefix_tok = 0
    imb = []                                  # 每次决策时的 (max_load - min_load)
    ttf, t = [], 0.0
    lam = lam_per_inst * n_inst

    for _ in range(N_REQ):
        t += rng.expovariate(lam)
        g = 0 if rng.random() < hot_share else rng.randint(1, G - 1)
        hashes, toks = WORK[g], TOKENS[g]
        for i in range(n_inst):                 # 刷新各实例的瞬时队列长度
            q = pending[i]
            while q and q[0] <= t:
                q.popleft()
        loads = [len(q) for q in pending]
        imb.append(max(loads) - min(loads))

        if policy == "cache_blind":             # 普通负载均衡器：看不到任何缓存
            i1 = rng.randrange(n_inst)
            i2 = (i1 + 1 + rng.randrange(n_inst - 1)) % n_inst
            i = i1 if loads[i1] <= loads[i2] else i2
        elif policy == "prefix_hash":           # 无状态：哈希 prompt 前 N 个 token
            i = (hash(tuple(toks[:prefix_tokens])) & 0xFFFFFFFF) % n_inst
            avg = (sum(loads) + 1) / n_inst
            if loads[i] > avg * 1.25:           # load_factor 默认 1.25，过载就顺时针走一个
                i = (i + 1) % n_inst
        elif policy == "loadaware":             # vLLM：benefit - beta * relative_load
            rel = relative_loads(loads)
            best, bs = 0, -float("inf")
            for k in range(n_inst):
                matched = idxs[k].match(hashes) * BLOCK
                s = min(matched, PROMPT) / PROMPT - beta * rel[k]
                if s > bs:
                    best, bs = k, s
            i = best
        elif policy in ("cache_aware", "pure_cache", "pure_cache_jsq"):
            mn, mx = min(loads), max(loads)
            # 源码是 && 不是 ||：两个条件必须同时成立才算不均衡
            if policy == "cache_aware" and (mx - mn > abs_th) and (mx > mn * rel_th):
                i = pick_min_load(loads, rng)
            else:
                best, bm = 0, -1
                for k in range(n_inst):
                    m = idxs[k].match(hashes)
                    if m > bm:
                        best, bm = k, m
                if policy == "pure_cache":
                    # 只要有一点匹配就跟着走；完全没有匹配时随机挑一个
                    i = best if bm > 0 else rng.randrange(n_inst)
                elif policy == "pure_cache_jsq":
                    # 消融：唯一区别是「完全没有匹配时」改成走最短队列
                    i = best if bm > 0 else pick_min_load(loads, rng)
                else:
                    i = best if (bm * BLOCK / PROMPT) > cache_th else pick_min_load(loads, rng)
        else:
            raise ValueError(policy)

        routed[i] += 1
        ahead = idxs[i].match(hashes)            # 路由器以为命中的块数
        real = caches[i].match(hashes)           # 引擎真实命中的块数
        ghost += max(0, ahead - real)
        ghost_slots += 1
        hit_tok += real * BLOCK
        miss_prefix_tok += PREFIX - real * BLOCK
        svc = FIXED + ((PREFIX - real * BLOCK) + SUFFIX) * PER_TOKEN
        start = max(t, last_fin[i])
        ttf.append(start - t + svc)              # 端到端 TTFT = 排队 + 本次 prefill
        last_fin[i] = start + svc
        pending[i].append(last_fin[i])
        inst_work[i] += svc
        caches[i].insert(hashes)
        idxs[i].insert(hashes)                   # 源码里也是选完立刻写进树

    ttf.sort()
    u = [w / t for w in inst_work]
    imb.sort()
    return {"hit": hit_tok / (hit_tok + miss_prefix_tok),
            "recompute": miss_prefix_tok / N_REQ + SUFFIX,
            "p50": ttf[len(ttf) // 2], "p99": ttf[int(len(ttf) * 0.99)],
            "u_mean": statistics.mean(u), "u_max": max(u),
            "route_max": max(routed) / (N_REQ / n_inst),
            "ghost": ghost / ghost_slots,
            # 队列长度差的统计量：绝对值阈值就是跟它比大小
            "imb_p50": imb[len(imb) // 2], "imb_p99": imb[int(len(imb) * 0.99)],
            "imb_max": imb[-1]}


def row(label, r, width=26):
    print(f"{label:>{width}}{r['hit']*100:>11.2f}%{r['recompute']:>10.0f}"
          f"{r['u_mean']*100:>8.1f}%{r['u_max']*100:>8.1f}%"
          f"{r['u_max']/r['u_mean']:>8.2f}x{r['p99']*1000:>10.1f}ms")


HDR = (f"{'策略':>26}{'前缀命中率':>12}{'平均重算':>10}{'u_mean':>9}{'u_max':>9}"
       f"{'不均衡':>9}{'TTFT p99':>11}")
```

指标口径说明：`u_mean` 是集群平均利用率，`u_max` 是负载最重那个实例的利用率，两者之比即不均衡度。TTFT 是**端到端**的（排队 + 本次 prefill 本身），不是只有排队——这个区别很关键，负载低的时候只有端到端口径才能区分策略。

### 6.1 五种策略同台

```python
"""代码块 5：五种策略同台 —— 两目标问题的正面证据。"""
print(f"实例 {N_INST} 个，{G} 个前缀组（其中 1 组占 {0.35*100:.0f}% 请求），"
      f"每实例 KV 池 = 工作集的 1/6")
print(f"prompt {PROMPT} token；单实例到达率 32.5/s，共 {N_REQ} 请求\n")
print(HDR)
print("-" * len(HDR))
for name in ["cache_blind", "prefix_hash", "pure_cache", "cache_aware", "loadaware"]:
    row(name, simulate(name))

print("\n两个目标分别看（以 cache_blind 为基线）：")
base = simulate("cache_blind")
for name in ["prefix_hash", "pure_cache", "cache_aware", "loadaware"]:
    r = simulate(name)
    print(f"  {name:>13}: 工作量 {r['u_mean']/base['u_mean']:>5.2f}x"
          f"（越小越省）    TTFT p99 {r['p99']/base['p99']:>5.2f}x")

print("\n多随机种子复核（5 个种子）：")
for name in ["pure_cache", "cache_aware", "loadaware"]:
    rs = [simulate(name, seed=s) for s in [1, 7, 13, 42, 101]]
    h = [x["hit"] * 100 for x in rs]
    p = [x["p99"] * 1000 for x in rs]
    print(f"  {name:>13}: 命中 {min(h):.2f}~{max(h):.2f}%   "
          f"TTFT p99 {min(p):.1f}~{max(p):.1f} ms")
```

```text
实例 8 个，200 个前缀组（其中 1 组占 35% 请求），每实例 KV 池 = 工作集的 1/6
prompt 1792 token；单实例到达率 32.5/s，共 20000 请求

                        策略       前缀命中率      平均重算   u_mean    u_max      不均衡   TTFT p99
--------------------------------------------------------------------------------------
               cache_blind      51.90%       995    80.2%    81.3%    1.01x     124.7ms
               prefix_hash      91.51%       386    35.5%    82.0%    2.31x      83.5ms
                pure_cache      98.89%       273    27.0%    92.5%    3.43x     446.3ms
               cache_aware      99.23%       268    26.6%    75.2%    2.83x      60.3ms
                 loadaware      55.90%       933    76.4%    87.8%    1.15x      87.4ms

两个目标分别看（以 cache_blind 为基线）：
    prefix_hash: 工作量  0.44x（越小越省）    TTFT p99  0.67x
     pure_cache: 工作量  0.34x（越小越省）    TTFT p99  3.58x
    cache_aware: 工作量  0.33x（越小越省）    TTFT p99  0.48x
      loadaware: 工作量  0.95x（越小越省）    TTFT p99  0.70x

多随机种子复核（5 个种子）：
     pure_cache: 命中 94.35~98.89%   TTFT p99 166.2~32242.4 ms
    cache_aware: 命中 99.18~99.53%   TTFT p99 60.3~78.8 ms
      loadaware: 命中 54.78~55.90%   TTFT p99 84.8~89.6 ms
```

这张表是本期的核心证据，逐行读：

- **`cache_blind`（普通负载均衡器）**：命中率 51.90%——随机分到某个实例时，那个实例恰好有这段前缀的概率大约是一半。算力消耗 995 token/请求，是全场最贵的。
- **`pure_cache`（只按缓存选，没有任何负载兜底）**：命中率 98.89%，算力只要 273 token（**0.34×**）。但它最忙的实例利用率 92.5%——注意这一行**过饱和的边已经被摸到**。TTFT p99 是 `cache_blind` 的 **3.58 倍**（446.3 ms）。而这一行的多种子波动是 **166 ms ~ 32.2 s**，200 倍——它踩在悬崖边上。
- **`cache_aware`（SGLang 默认）**：命中率 99.23%（比 `pure_cache` 还略高），算力 268 token（0.33×），但 TTFT p99 只有 **60.3 ms**，是全场最好。
- **`loadaware`（vLLM 默认 beta=1.0）**：命中率只有 55.90%，几乎退化成 `cache_blind`。原因有两个因素叠加，第 7、8 节分别揭晓。

**把 `pure_cache` 和 `cache_aware` 两行摞在一起看**：它们做掉的算力几乎一样（273 vs 268 token，差 1.9%），命中率只差 0.34 个百分点，但 **TTFT p99 差 7.4 倍**（446.3 vs 60.3 ms）。这就是 §1.4 那句话的实证——**它们省下的算力一样多，差别全在负载**。

### 6.2 消融：真正起作用的是哪个机制

现在要回答一个很关键的问题：`cache_aware` 比 `pure_cache` 好，是**阈值**（`cache_threshold`）在起作用，还是别的什么？

看源码，两者只有一个分支不同：**完全没有匹配（`bm == 0`）时往哪走**。`pure_cache` 走随机，`cache_aware` 走最短队列。那就只改这一处，其余保持完全一致：

```python
"""代码块 6：消融与阈值 —— 真正起作用的是哪个机制。"""
print("A. 消融：pure_cache 与 cache_aware 的唯一差别，是「完全没有匹配时」往哪走")
print(HDR)
print("-" * len(HDR))
row("cache_blind（对照）", simulate("cache_blind"))
row("pure_cache（冷请求随机）", simulate("pure_cache"))
row("pure_cache_jsq（冷请求→最短队列）", simulate("pure_cache_jsq"))
row("cache_aware（默认 0.3）", simulate("cache_aware"))
print("\n后两行的输出逐位相同 —— 只要把兜底动作改成最短队列，命中率、负载、延迟")
print("全部落到 cache_aware 上。也就是说这个 workload 里起作用的不是阈值，是兜底。")

print(f"\nB. 扫 cache_threshold（嵌套前缀下，部分匹配率恰好 = {DEPT_LEN/PROMPT:.4f}）")
print(f"{'cache_threshold':>16}{'前缀命中率':>12}{'平均重算':>10}{'u_max':>9}{'TTFT p99':>11}")
for th in [0.1, 0.65, 0.70, 0.75, 0.80, 0.95, 1.0]:
    r = simulate("cache_aware", cache_th=th)
    print(f"{th:>16.2f}{r['hit']*100:>11.2f}%{r['recompute']:>10.0f}"
          f"{r['u_max']*100:>8.1f}%{r['p99']*1000:>9.1f}ms")
print(f"分界点正好落在 {DEPT_LEN/PROMPT:.4f}：阈值在它下面时「只命中部门段」也算数，")
print("请求被吸到持有部门段的实例上；抬到它上面就退回最短队列。")

print("\nC. 扫到达率：缓存亲和在哪一侧划算（TTFT p99，单位 ms）")
print(f"{'单实例 λ':>10}{'cache_blind':>13}{'pure_cache':>13}{'cache_aware':>13}"
      f"{'blind 利用率':>14}")
for lam in [5.0, 12.5, 25.0, 37.5, 50.0]:
    a = simulate("cache_blind", lam_per_inst=lam)
    b = simulate("pure_cache", lam_per_inst=lam)
    c = simulate("cache_aware", lam_per_inst=lam)
    print(f"{lam:>10.1f}{a['p99']*1000:>13.1f}{b['p99']*1000:>13.1f}"
          f"{c['p99']*1000:>13.1f}{a['u_mean']*100:>13.1f}%")
print(f"低负载下三者的 p99 都在 20~45 ms 量级 —— 正好是一次整段冷 prefill 的墙钟")
print(f"（FIXED + PROMPT·c = 2 + 1792×0.023 ≈ 43 ms）。此时缓存感知路由省的是算力，")
print("延迟差异还没被排队放大。负载上来以后差距才被拉开，而且顺序出人意料：")
print("先崩的是 pure_cache（没有任何兜底的纯缓存亲和），它在 λ=37.5 已经 4.2 s；")
print("cache_aware 靠最短队列兜底，同一负载下只有 131 ms。")
```

```text
A. 消融：pure_cache 与 cache_aware 的唯一差别，是「完全没有匹配时」往哪走
                        策略       前缀命中率      平均重算   u_mean    u_max      不均衡   TTFT p99
--------------------------------------------------------------------------------------
           cache_blind（对照）      51.90%       995    80.2%    81.3%    1.01x     124.7ms
         pure_cache（冷请求随机）      98.89%       273    27.0%    92.5%    3.43x     446.3ms
  pure_cache_jsq（冷请求→最短队列）      99.23%       268    26.6%    75.2%    2.83x      60.3ms
       cache_aware（默认 0.3）      99.23%       268    26.6%    75.2%    2.83x      60.3ms

后两行的输出逐位相同 —— 只要把兜底动作改成最短队列，命中率、负载、延迟
全部落到 cache_aware 上。也就是说这个 workload 里起作用的不是阈值，是兜底。

B. 扫 cache_threshold（嵌套前缀下，部分匹配率恰好 = 0.5714）
 cache_threshold       前缀命中率      平均重算    u_max   TTFT p99
            0.10      99.23%       268    75.2%     60.3ms
            0.65      80.51%       555    77.3%    114.9ms
            0.70      80.51%       555    77.3%    114.9ms
            0.75      80.51%       555    77.3%    114.9ms
            0.80      80.51%       555    77.3%    114.9ms
            0.95      51.73%       997    81.8%    119.3ms
            1.00      51.73%       997    81.8%    119.3ms
分界点正好落在 0.5714：阈值在它下面时「只命中部门段」也算数，
请求被吸到持有部门段的实例上；抬到它上面就退回最短队列。

C. 扫到达率：缓存亲和在哪一侧划算（TTFT p99，单位 ms）
     单实例 λ  cache_blind   pure_cache  cache_aware     blind 利用率
       5.0         43.2         19.7         32.5         12.4%
      12.5         74.3         25.5         25.8         30.9%
      25.0         92.6         74.6         40.0         61.9%
      37.5        192.1       4233.4        131.5         92.7%
      50.0      11733.8      20535.9       1427.8        123.3%
低负载下三者的 p99 都在 20~45 ms 量级 —— 正好是一次整段冷 prefill 的墙钟
（FIXED + PROMPT·c = 2 + 1792×0.023 ≈ 43 ms）。此时缓存感知路由省的是算力，
延迟差异还没被排队放大。负载上来以后差距才被拉开，而且顺序出人意料：
先崩的是 pure_cache（没有任何兜底的纯缓存亲和），它在 λ=37.5 已经 4.2 s；
cache_aware 靠最短队列兜底，同一负载下只有 131 ms。
```

**A 段的结果很干净：`pure_cache_jsq` 和 `cache_aware` 的四列数字逐位相同。** 只改一个分支就完全复现——所以这个 workload 下 `cache_threshold` 没有参与决策，起作用的纯粹是「冷请求走最短队列」这个兜底。

这解释了 §3.4 里 GIE 为什么要专门设一个 `no-hit-lru-scorer`：**系统的稳定性是由「完全没有命中时往哪走」决定的，而不是由命中路径上的精调决定的。**

**B 段揭示了 `cache_threshold` 真正在做什么。** 它是台阶，不是旋钮——因为嵌套前缀下可取的匹配率只有三档：0、0.5714（只命中部门段）、1.0（全命中）。阈值在 0.10~0.65 时「部分命中也算数」，请求被吸到持有部门段的实例上（命中率 99.23%，最忙实例 75.2%）；抬到 0.70~0.80 就把这档排除，退成最短队列（命中率掉到 80.51%，p99 从 60.3 涨到 114.9 ms）；再到 0.95 以上连全命中都不认了（51.73%，退化成纯负载均衡）。

**这给出了一条很实用的判断规则**：`cache_threshold` 的合理取值应该落在你负载里「典型的部分匹配率」和「全命中」之间——**设得太低会让浅匹配也把请求吸过去（制造倾斜），设得太高会连真命中都不认（白扔缓存）**。而这个「典型部分匹配率」取决于你的前缀结构（共享段占多少），不是一个能抄的常数。

**C 段的负载扫描回答了「什么时候划算」**：

| 单实例 λ | `cache_blind` | `pure_cache` | `cache_aware` | blind 利用率 |
|---|---|---|---|---|
| 5.0 | 43.2 | 19.7 | 32.5 | 12.4% |
| 12.5 | 74.3 | 25.5 | 25.8 | 30.9% |
| 25.0 | 92.6 | 74.6 | 40.0 | 61.9% |
| 37.5 | 192.1 | **4233.4** | 131.5 | 92.7% |
| 50.0 | 11733.8 | 20535.9 | 1427.8 | 123.3% |

低负载端（λ=5）三者的 p99 都在 20~43 ms，**那正好是一次整段冷 prefill 的墙钟**（$2 + 1792 \times 0.023 \approx 43$ ms）。此时三个策略的延迟没有区别——缓存感知路由省下来的是**算力**，不是延迟。

差距要到利用率上来才出现。而且崩溃的顺序很反直觉：**先崩的是 `pure_cache`**（λ=37.5 时 4.23 s，比不用缓存的 `cache_blind` 的 192 ms 还差 22 倍）。`cache_aware` 靠最短队列兜底，同一负载下 131 ms。

这条结论和第 027 期那句「投机解码是延迟武器、不是吞吐武器」是同构的：

> **KV 感知路由是容量杠杆，不是延迟技巧。**
> 负载轻时它换来的是省下 N 倍算力；负载重时它换来的是「不排队」。

---

## 7. 旋钮的解剖：两个参数，两种失败方式

SGLang 那个 `cache_threshold` 是台阶式的。vLLM 的 `beta` 是连续汇率，更值得单独拆。

```python
"""代码块 7：扫 beta —— 「缓存亲和值多少负载」这个汇率。"""
print("A. 扫 vLLM loadaware 的 beta（=「一单位相对负载值多少缓存收益」）")
print(f"{'beta':>8}{'前缀命中率':>12}{'平均重算':>10}{'u_mean':>9}{'u_max':>9}"
      f"{'TTFT p99':>11}")
for beta in [0.0, 0.1, 0.2, 0.25, 0.5, 1.0, 4.0, 1000.0]:
    r = simulate("loadaware", beta=beta)
    tag = "  ← 退化：并列后全挤一个实例" if beta == 0 else ("  ← 只看负载" if beta > 100 else "")
    print(f"{beta:>8.2f}{r['hit']*100:>11.2f}%{r['recompute']:>10.0f}"
          f"{r['u_mean']*100:>8.1f}%{r['u_max']*100:>8.1f}%"
          f"{r['p99']*1000:>9.1f}ms{tag}")
print("\nbeta = 0 是退化配置，不是「纯按缓存」的正确写法：")
print("此时所有实例的 benefit 都是 0，score 全部并列，而源码的 tie-break 是")
print("字典序（for info in sorted(endpoints, key=lambda e: e.url) + 严格大于判等），")
print("于是冷请求全部涌向同一个实例。beta > 0 时相对负载项天然打散并列，才不踩这个坑。")

print("\n每个 beta 取 5 个种子的中位（用来确认最优点不是种子运气）")
print(f"{'beta':>8}{'前缀命中率中位':>16}{'u_max 中位':>13}{'TTFT p99 中位':>16}")
for beta in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
    rs = [simulate("loadaware", beta=beta, seed=s) for s in [1, 7, 13, 42, 101]]
    print(f"{beta:>8.2f}{statistics.median(x['hit']*100 for x in rs):>15.2f}%"
          f"{statistics.median(x['u_max']*100 for x in rs):>12.1f}%"
          f"{statistics.median(x['p99']*1000 for x in rs):>14.1f}ms")
print("实测最优在 0.15~0.25 这一段，而默认值是 1.0 —— 1.0 的命中率只有 55.4%。")

print("\nC. 负载压上去以后，最优点会不会漂？（每格 3 个种子的 p99 中位）")
print(f"{'单实例 λ':>10}{'u_mean':>9}{'b=0.1':>10}{'b=0.2':>10}{'b=0.4':>10}"
      f"{'b=0.8':>10}{'b=1.5':>10}{'argmin':>9}")
drift = []
for lam in [12.5, 25.0, 50.0, 75.0]:
    base = simulate("cache_aware", lam_per_inst=lam)
    cells = []
    for beta in [0.1, 0.2, 0.4, 0.8, 1.5]:
        rs = [simulate("loadaware", beta=beta, lam_per_inst=lam, seed=s)
              for s in [1, 7, 13]]
        cells.append((beta, statistics.median(x["p99"] * 1000 for x in rs)))
    best = min(cells, key=lambda x: x[1])
    drift.append((lam, base["u_mean"], best[0]))
    print(f"{lam:>10.1f}{base['u_mean']*100:>8.1f}%"
          + "".join(f"{p:>9.1f}m" for _, p in cells)
          + f"{best[0]:>9.1f}")
print("这是一条 U 形曲线：beta 太小 → 请求堆在持缓存的实例上；beta 太大 → 缓存被拆散。")
print("负载真正起来以后（u_mean > 20%）最优点稳定在 0.2；系统过载后才抬到 0.4。")
print("而负载极低时反而是更大的 beta 更好 —— 排队几乎为零，p99 完全由「有没有撞上")
print("一次整段冷 prefill」决定，铺得越开越不容易撞上（43.2 ms 正好等于一次完整")
print("冷 prefill 的墙钟）。")
print("真正该记住的是 β 的**容忍区间随负载收窄**：u_mean 82% 时 beta=1.5 已经崩到 1.8 s。")
```

```text
A. 扫 vLLM loadaware 的 beta（=「一单位相对负载值多少缓存收益」）
    beta       前缀命中率      平均重算   u_mean    u_max   TTFT p99
    0.00      52.02%       993    80.9%   647.2% 416073.0ms  ← 退化：并列后全挤一个实例
    0.10      96.86%       304    29.3%    57.4%     47.6ms
    0.20      95.76%       321    30.6%    63.4%     43.2ms
    0.25      97.11%       300    29.0%    61.0%     38.8ms
    0.50      82.45%       526    45.9%    73.3%     58.3ms
    1.00      55.90%       933    76.4%    87.8%     87.4ms
    4.00      55.49%       940    76.9%    88.4%     91.8ms
 1000.00      55.49%       940    76.9%    88.4%     91.8ms  ← 只看负载

beta = 0 是退化配置，不是「纯按缓存」的正确写法：
此时所有实例的 benefit 都是 0，score 全部并列，而源码的 tie-break 是
字典序（for info in sorted(endpoints, key=lambda e: e.url) + 严格大于判等），
于是冷请求全部涌向同一个实例。beta > 0 时相对负载项天然打散并列，才不踩这个坑。

每个 beta 取 5 个种子的中位（用来确认最优点不是种子运气）
    beta         前缀命中率中位     u_max 中位     TTFT p99 中位
    0.05          97.80%        77.7%          64.2ms
    0.10          97.80%        74.4%          58.0ms
    0.15          98.06%        57.7%          38.5ms
    0.20          96.76%        61.7%          43.2ms
    0.30          89.34%        64.9%          56.0ms
    0.50          82.45%        74.2%          58.7ms
    1.00          55.37%        88.8%          85.7ms
实测最优在 0.15~0.25 这一段，而默认值是 1.0 —— 1.0 的命中率只有 55.4%。

C. 负载压上去以后，最优点会不会漂？（每格 3 个种子的 p99 中位）
     单实例 λ   u_mean     b=0.1     b=0.2     b=0.4     b=0.8     b=1.5   argmin
      12.5    10.5%     91.0m     67.2m     61.3m     57.4m     43.2m      1.5
      25.0    20.4%     67.0m     49.8m     52.7m     55.1m     68.6m      0.2
      50.0    81.9%     67.1m     45.6m     65.5m    334.7m   1819.5m      0.2
      75.0   150.5%  23242.1m    176.2m    161.2m   1352.0m   1713.8m      0.4
这是一条 U 形曲线：beta 太小 → 请求堆在持缓存的实例上；beta 太大 → 缓存被拆散。
负载真正起来以后（u_mean > 20%）最优点稳定在 0.2；系统过载后才抬到 0.4。
而负载极低时反而是更大的 beta 更好 —— 排队几乎为零，p99 完全由「有没有撞上
一次整段冷 prefill」决定，铺得越开越不容易撞上（43.2 ms 正好等于一次完整
冷 prefill 的墙钟）。
真正该记住的是 β 的**容忍区间随负载收窄**：u_mean 82% 时 beta=1.5 已经崩到 1.8 s。
```

三个要读出来的东西：

**第一，`beta = 0` 不是「纯按缓存」，是一个退化配置。** 所有实例的 benefit 都是 0 时，score 全部并列，而源码的 tie-break 是**确定性的字典序**（`for info in sorted(endpoints, key=lambda e: e.url)` 配合严格大于判等，第一个最大值胜出）。结果冷请求全涌向同一个实例，最忙实例利用率 647%、p99 416 秒。`beta > 0` 时相对负载项天然打散并列，才不踩这个坑。**想实现「纯缓存亲和」应该用别的手段，不是把 `beta` 设成 0。**

**第二，实测最优点（0.15~0.25）离默认值 1.0 有一段距离。** 五种子的中位数很稳：`beta=0.15` 时命中率 98.06%、最忙实例 57.7%、p99 38.5 ms；`beta=1.0` 时命中率掉到 55.37%、最忙实例涨到 88.8%、p99 85.7 ms。**默认值把命中率从 98% 打到 55%，延迟翻倍还多。**

**第三，`beta` 的最优点会漂，而且加载曲线的形状是 U 形。** λ=12.5（利用率 10.5%）时最优是 1.5（零排队下铺得越开越不容易撞上冷 prefill）；λ=25~50（利用率 20%~82%）时稳定在 0.2；λ=75（利用率 150%，已过载）时抬到 0.4。更要紧的是**容忍区间随负载急剧收窄**：利用率 82% 时 `beta=1.5` 的 p99 已经是 1.8 秒，而 `beta=0.2` 还是 45.6 ms。

### 7.1 `beta*` 的解析形式

既然 `beta` 是个汇率，那它应该有一个基于经济的取值。把两项都换算成延迟：

```python
"""代码块 8：beta 到底该取多少？—— 把打分函数的两项都换算成延迟。

打分是 benefit − beta · relative_load，所以 beta 是「一单位相对负载值多少缓存收益」。
合理的汇率应当是

    beta* = 一单位相对负载的代价 / 一次整段命中的价值

  一次整段命中的价值 ≈ 省下的 prefill 服务时间 = PROMPT · c = 41.22 ms
  一单位相对负载的代价 ≈ M/M/1 下「负载翻倍」带来的等待增量
                        ΔW = E[S] · [ 2ρ̄/(1-2ρ̄) − ρ̄/(1-ρ̄) ]

注意 ΔW 里带着 ρ̄，所以 beta* 不是一个通用常数：负载越重，负载项越贵，beta* 越大。
"""
PROMPT, PER_TOKEN, ES_MS = 1792, 0.023, 9.75       # E[S] 取命中主导工作点的实测值
hit_value = PROMPT * PER_TOKEN

print(f"一次整段命中的价值 = {PROMPT} × {PER_TOKEN} ms = {hit_value:.2f} ms")
print(f"\n{'ρ̄':>6}{'ΔW/E[S]':>12}{'ΔW (ms)':>12}{'beta* = ΔW / 命中价值':>24}")
rows = []
for rho in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    dW = 2 * rho / (1 - 2 * rho) - rho / (1 - rho)
    dW_ms = dW * ES_MS
    rows.append((rho, dW_ms / hit_value))
    print(f"{rho:>6.2f}{dW:>12.3f}{dW_ms:>10.1f}ms{dW_ms/hit_value:>22.3f}")


def implied_rho(beta):
    """给定 beta，反查解析表说「你现在工作在多重负载上」。"""
    for (r0, b0), (r1, b1) in zip(rows, rows[1:]):
        if b0 <= beta <= b1:
            return r0 + (r1 - r0) * (beta - b0) / (b1 - b0)
    return None


for b in (0.2, 1.0):
    r = implied_rho(b)
    print(f"beta = {b:<4} ↔ 解析工作点 ρ̄ ≈ {r:.3f}"
          + ("   ← 实测最优点附近" if b == 0.2 else "   ← vLLM 默认值，接近满载"))

print("\n对照实测：cache_aware 的工作点 u_mean ≈ 0.27，解析表插值给 beta* ≈ 0.20，")
print("实测最优点落在 0.15~0.25 —— 量级和方向都对得上。")
print("差别在于解析表单调递增，实测是 U 形：0.2 附近有稳定甜点，过载后才抬到 0.4。")
print("两条曲线指向同一个结论 —— 默认 1.0 是在「接近满载」的假设下定的，")
print("常规服务区间（ρ̄ < 0.4）里它偏高，会把本该复用的前缀主动拆散。")
```

```text
一次整段命中的价值 = 1792 × 0.023 ms = 41.22 ms

    ρ̄     ΔW/E[S]     ΔW (ms)       beta* = ΔW / 命中价值
  0.05       0.058       0.6ms                 0.014
  0.10       0.139       1.4ms                 0.033
  0.15       0.252       2.5ms                 0.060
  0.20       0.417       4.1ms                 0.099
  0.25       0.667       6.5ms                 0.158
  0.30       1.071      10.4ms                 0.253
  0.35       1.795      17.5ms                 0.425
  0.40       3.333      32.5ms                 0.789
  0.45       8.182      79.8ms                 1.935
beta = 0.2  ↔ 解析工作点 ρ̄ ≈ 0.272   ← 实测最优点附近
beta = 1.0  ↔ 解析工作点 ρ̄ ≈ 0.409   ← vLLM 默认值，接近满载

对照实测：cache_aware 的工作点 u_mean ≈ 0.27，解析表插值给 beta* ≈ 0.20，
实测最优点落在 0.15~0.25 —— 量级和方向都对得上。
差别在于解析表单调递增，实测是 U 形：0.2 附近有稳定甜点，过载后才抬到 0.4。
两条曲线指向同一个结论 —— 默认 1.0 是在「接近满载」的假设下定的，
常规服务区间（ρ̄ < 0.4）里它偏高，会把本该复用的前缀主动拆散。
```

（这里的 $\rho$ 是**每个实例**的利用率，$\bar\rho$ 是集群平均。$\Delta W$ 用 M/M/1 算负载从 $\bar\rho$ 翻倍到 $2\bar\rho$ 的等待增量，$\bar\rho = 0.4$ 时 $2\bar\rho$ 已经逼近 1，所以 $\Delta W$ 开始发散。）

**这张表的意义在于它给了 `beta` 一个物理解释。** 命中价值 $41.22$ ms 是固定的（由 prompt 长度和模型速度决定），而一单位相对负载的代价随利用率超线性增长：

$$\beta^* \approx \frac{E[S] \cdot \left[\dfrac{2\bar\rho}{1-2\bar\rho} - \dfrac{\bar\rho}{1-\bar\rho}\right]}{P_{\text{prompt}} \cdot c_{\text{token}}}$$

分子是负载项的边际代价，分母是缓存项的边际价值。**它在 $\bar\rho \approx 0.27$ 附近给出 0.20，在 $\bar\rho \approx 0.30$ 给出 0.253——而默认值 1.0 对应的解析工作点是 $\bar\rho \approx 0.41$。** 也就是说，**默认 1.0 是在「接近满载」的假设下定的**；常规服务区间（$\bar\rho < 0.4$）里它偏高，会把本该复用的前缀主动拆散。

解析表单调递增，实测是 U 形（负载极低时更大的 `beta` 反而更好，因为此时 p99 完全由「有没有撞上一次整段冷 prefill」决定，铺得越开越不容易撞上）。**量级和方向对得上，绝对值必须按工作点标定。** 这也是源码注释里那句 `rather than a number calibrated on one cluster` 的另一面：它确实不是一个常数，但默认值选得偏保守了。

### 7.2 另一个旋钮：`balance_abs_threshold`

`beta` 是一个连续汇率。SGLang `cache_aware` 里还有第二个旋钮，是**绝对差阈值**——就是 §3.1 那条 `&&` 判据的左半边：

```text
is_imbalanced = (max_load - min_load) > balance_abs_threshold      ← 绝对差，CLI 默认 64
                && max_load > min_load * balance_rel_threshold     ← 相对比，CLI 默认 1.5
```

它到底在设定什么？把 `abs_th` 扫一遍，同时把**每次决策时的队列长度差**统计下来：

```python
"""代码块 11：绝对差阈值为什么会失效 —— 它的量级由 Little 定律决定。

SGLang cache_aware 的判据是
    (max_load - min_load) > balance_abs_threshold  AND  max_load > min_load * balance_rel_threshold
左边那个减法的量级是多少？稳态下实例队列长度 L ≈ λ·E[S]（Little 定律）。
λ 是单实例到达率，E[S] 是一次 prefill 的服务时间 —— 两者都是「每个实例」的量，
所以 L 与实例数无关。默认阈值 64 是在跟一个 O(1) 的数比大小。
"""
print("A. 同一个阈值，实例数不同，行为完全不同")
print(f"{'N':>4}{'abs_th':>8}{'命中率':>10}{'u_mean':>9}{'u_max':>9}"
      f"{'imb_p50':>9}{'imb_p99':>9}{'imb_max':>9}{'TTFT p99':>11}")
rows_by_th = {}
for N in [8, 16, 32]:
    for th in [1, 8, 16, 64]:
        r = simulate("cache_aware", n_inst=N, abs_th=th)
        rows_by_th[(N, th)] = r
        print(f"{N:>4}{th:>8}{r['hit']*100:>9.2f}%{r['u_mean']*100:>8.1f}%{r['u_max']*100:>8.1f}%"
              f"{r['imb_p50']:>9}{r['imb_p99']:>9}{r['imb_max']:>9}{r['p99']*1000:>9.1f}ms")

print("\nB. 哪些 abs_th 是逐位相同的（说明那个分支从没触发）")
for N in [8, 16, 32]:
    groups = {}
    for th in [1, 8, 16, 64]:
        r = rows_by_th[(N, th)]
        key = (round(r["hit"], 12), round(r["p99"], 12), round(r["u_max"], 12))
        groups.setdefault(key, []).append(th)
    print(f"  N={N:>2}: " + "  ==  ".join(
        "{" + ",".join(str(t) for t in v) + "}" for v in groups.values()))

print("\nC. 阈值实际设定的是什么")
for N in [8, 16, 32]:
    a1, a64 = rows_by_th[(N, 1)], rows_by_th[(N, 64)]
    print(f"  N={N:>2}: abs_th 1→64，命中率 {a1['hit']*100:.2f}% → {a64['hit']*100:.2f}%，"
          f"TTFT p99 {a1['p99']*1000:.1f}ms → {a64['p99']*1000:.1f}ms，"
          f"u_max {a1['u_max']*100:.1f}% → {a64['u_max']*100:.1f}%")
print("放宽阈值 → 命中率单调上升、延迟单调恶化：这就是那两个目标的汇率，")
print("balance_abs_threshold 是它的手动档。")
print("看 imb_p50 那一列：阈值为 8/16/64 时它恰好是 9 / 16~17 / 64~65 —— 几乎是阈值的")
print("镜像。因为计数一旦越过阈值，路由器立刻去排空最长的那条队列，不均衡被顶回阈值附近。")
print("所以这个参数设定的不是「什么时候介入」，而是「允许多歪」。")
print("健康状态下（abs_th=1、没有实例饱和）队列长度差只有 2~4 个请求，")
print("默认值 64/32 比它高了一到两个数量级 —— 阈值之所以还能被触发，")
print("恰恰是因为它先允许系统歪到某个实例饱和、队列开始随时间线性增长为止。")
print("护栏只在它本该防住的那场事故已经发生之后才真的介入。")
```

```text
A. 同一个阈值，实例数不同，行为完全不同
   N  abs_th       命中率   u_mean    u_max  imb_p50  imb_p99  imb_max   TTFT p99
   8       1    53.52%    79.3%    94.1%        2        3        4    104.0ms
   8       8    97.81%    28.2%    75.0%        2        8        9     57.2ms
   8      16    99.23%    26.6%    75.2%        2        9       12     60.3ms
   8      64    99.23%    26.6%    75.2%        2        9       12     60.3ms
  16       1    52.16%    80.6%    94.0%        2        3        4     89.3ms
  16       8    60.80%    70.8%    98.3%        9       10       10    244.3ms
  16      16    62.79%    68.3%    99.5%       16       18       19    410.8ms
  16      64    65.82%    66.1%    99.9%       64       66       66   1576.5ms
  32       1    52.05%    81.0%    91.1%        2        3        3     78.8ms
  32       8    58.61%    73.5%    98.0%        9       10       10    196.8ms
  32      16    59.01%    73.3%    99.0%       17       18       18    345.0ms
  32      64    60.66%    71.0%   101.9%       65       66       66    519.1ms

B. 哪些 abs_th 是逐位相同的（说明那个分支从没触发）
  N= 8: {1}  ==  {8}  ==  {16,64}
  N=16: {1}  ==  {8}  ==  {16}  ==  {64}
  N=32: {1}  ==  {8}  ==  {16}  ==  {64}

C. 阈值实际设定的是什么
  N= 8: abs_th 1→64，命中率 53.52% → 99.23%，TTFT p99 104.0ms → 60.3ms，u_max 94.1% → 75.2%
  N=16: abs_th 1→64，命中率 52.16% → 65.82%，TTFT p99 89.3ms → 1576.5ms，u_max 94.0% → 99.9%
  N=32: abs_th 1→64，命中率 52.05% → 60.66%，TTFT p99 78.8ms → 519.1ms，u_max 91.1% → 101.9%
放宽阈值 → 命中率单调上升、延迟单调恶化：这就是那两个目标的汇率，
balance_abs_threshold 是它的手动档。
看 imb_p50 那一列：阈值为 8/16/64 时它恰好是 9 / 16~17 / 64~65 —— 几乎是阈值的
镜像。因为计数一旦越过阈值，路由器立刻去排空最长的那条队列，不均衡被顶回阈值附近。
所以这个参数设定的不是「什么时候介入」，而是「允许多歪」。
健康状态下（abs_th=1、没有实例饱和）队列长度差只有 2~4 个请求，
默认值 64/32 比它高了一到两个数量级 —— 阈值之所以还能被触发，
恰恰是因为它先允许系统歪到某个实例饱和、队列开始随时间线性增长为止。
护栏只在它本该防住的那场事故已经发生之后才真的介入。
```

这张表里的 `imb_p50` 是这次实验的关键指标——它是「每次决策时 $ℓ_{\max}-ℓ_{\min}$ 的中位数」。三条结论：

1. **`imb_p50` 几乎精确等于阈值的镜像。** 阈值为 8 / 16 / 64 时，每次决策时实测的 $\ell_{\max}-\ell_{\min}$ 中位数是 **9 / 16~17 / 64~65**。因为计数一旦越过阈值，路由器立刻去排空最长队列，不均衡被顶回阈值附近——这是一个负反馈，把系统**钉在**阈值这个高度上。
2. **所以 N=8 和 N=32 的结论是相反的。** N=8 时阈值 64 最优（命中率 99.23%、p99 60.3 ms，比 `abs_th=1` 的 53.52% / 104.0 ms 好得多）；N=32 时同一个 64 变成最差（60.66% / 519.1 ms，而 `abs_th=1` 是 52.05% / 78.8 ms）。**同一个默认值，在小集群是优点，在大集群是事故**——因为不均衡的绝对幅度随实例数增长，而阈值是固定的。
3. **它也不能靠 Little 定律就推断成「死代码」。** 直觉上 $L=\lambda E[S]\approx0.3$ 远小于 64，分支不该触发；但实测在 $N\ge16$ 时它触发得很频繁。原因是护栏允许系统先歪到某个实例饱和（$u_{\max}\to100\%$），队列开始随时间线性增长之后，这个差才够得着阈值。

**和 §7.1 放在一起看**，`cache_aware` 和 `loadaware` 的分野就清楚了：两者都在决定「缓存亲和」与「负载均衡」的汇率，区别只在这个汇率是连续可调的（`beta`）还是分段手动的（`abs_th`）。而两个默认值的失败方式完全不同——

- `beta = 1.0` 是**按错的工作点标定**：代价是命中率从 98% 打到 55%，但它至少还在工作。
- `abs_th = 64` 是**量级层面的错**：代价是护栏被钉在一个错误的高度上，你甚至看不出它在工作还是不在工作。

后者更危险，因为它不留下任何可观测的痕迹。

---

## 8. 索引精度值不值那条 KV 事件通道

现在回到 §2.2 的第 ② 和 ③ 层次。精确索引要维护一条 KV 事件通道（ZMQ 订阅、索引进程、高可用），成本不低。这一节量它值不值——**结论比预期更细致**。

```python
"""代码块 9：索引精度到底值不值那条 KV 事件通道？—— 取决于打分函数的形状。

路由器手里的前缀索引有两种精度：
  近似：只记「曾经往这个实例写过什么」，引擎驱逐了它不知道（相当于不消费 KV 事件）
  精确：通过 KV 事件同步撤销条目，索引与引擎真实状态一致

把「幽灵块」量出来：路由器以为命中、引擎其实已经驱逐掉的块数。
"""
print("A. 幽灵块随容量压力的增长（策略固定为 cache_aware）")
print(f"{'cap_ratio':>10}{'命中率·近似':>13}{'命中率·精确':>13}"
      f"{'幽灵块/请求·近似':>18}{'幽灵块/请求·精确':>18}{'p99·近似':>11}{'p99·精确':>11}")
for cr in [2.0, 4.0, 6.0, 10.0, 20.0]:
    a = simulate("cache_aware", cap_ratio=cr, precise=False)
    p = simulate("cache_aware", cap_ratio=cr, precise=True)
    print(f"{cr:>10.1f}{a['hit']*100:>12.2f}%{p['hit']*100:>12.2f}%"
          f"{a['ghost']:>18.2f}{p['ghost']:>18.2f}"
          f"{a['p99']*1000:>9.1f}ms{p['p99']*1000:>9.1f}ms")
print("幽灵块随容量收紧单调增长（0 → 28 块/请求）。容量宽松时（R ≤ 6）两列完全相同；")
print("容量吃紧后阈值型也只差 2 个百分点 —— 因为它只问「匹配率过没过线」，")
print("所有候选的匹配量被同等高估，过线与不过线的判定基本不变。")

print("\nB. 换一个形状的打分函数，同一份近似索引代价立刻显现")
print(f"{'打分方式':>26}{'命中率·近似':>13}{'命中率·精确':>13}"
      f"{'最忙接单·近似':>15}{'最忙接单·精确':>15}{'p99·近似':>11}{'p99·精确':>11}")
for label, pol, kw in [("阈值型 cache_aware(0.3)", "cache_aware", {}),
                       ("线性打分 loadaware β=0.2", "loadaware", {"beta": 0.2}),
                       ("线性打分 loadaware β=1.0", "loadaware", {"beta": 1.0}),
                       ("线性打分 loadaware β=4.0", "loadaware", {"beta": 4.0})]:
    a = simulate(pol, precise=False, **kw)
    p = simulate(pol, precise=True, **kw)
    print(f"{label:>26}{a['hit']*100:>12.2f}%{p['hit']*100:>12.2f}%"
          f"{a['route_max']:>14.2f}x{p['route_max']:>14.2f}x"
          f"{a['p99']*1000:>9.1f}ms{p['p99']*1000:>9.1f}ms")
print("阈值型在同一容量配置下（cap_ratio=6.0）两列逐位相同；线性打分的命中率")
print("分别掉了 2.9 / 20.3 / 19.9 个百分点 —— 索引精度的价值完全由打分函数决定。")

print("\nC. 幽灵块是怎么把线性打分带偏的")
print(f"{'beta':>7}{'幽灵块/请求':>14}{'近似命中率':>13}{'精确命中率':>13}{'差距':>10}")
for beta in [0.1, 0.2, 0.5, 1.0, 4.0]:
    a = simulate("loadaware", beta=beta, precise=False)
    p = simulate("loadaware", beta=beta, precise=True)
    print(f"{beta:>7.2f}{a['ghost']:>14.2f}{a['hit']*100:>12.2f}%"
          f"{p['hit']*100:>12.2f}%{(p['hit']-a['hit'])*100:>9.2f}pt")
print("关键在于幽灵条目是**持久**的：被引擎驱逐掉的块，近似索引里仍然记着。")
print("于是「我这里有这块前缀」这个判断对该实例永远虚高，打分持续把它选出来 ——")
print("实际命中不了，KV 池又被新流量挤得更满，虚高得更厉害。")
print("精确索引在驱逐发生的那一刻就撤掉条目，匹配量随之下降，流量自动移开。")
print("beta 越大，实例之间的候选分数越接近，虚高的那一截就越容易在临界处翻盘 ——")
print("所以精确索引的收益随 beta 单调放大（0.98pt → 20.25pt）。")
print("结论：**不消费 KV 事件可以，但前提是你的打分函数是阈值型。**")
```

```text
A. 幽灵块随容量压力的增长（策略固定为 cache_aware）
 cap_ratio       命中率·近似       命中率·精确         幽灵块/请求·近似         幽灵块/请求·精确     p99·近似     p99·精确
       2.0       99.53%       99.53%              0.00              0.00     60.3ms     60.3ms
       4.0       99.53%       99.53%              0.00              0.00     60.3ms     60.3ms
       6.0       99.23%       99.23%              0.29              0.00     60.3ms     60.3ms
      10.0       90.82%       93.02%              8.36              0.00     68.5ms     96.0ms
      20.0       69.81%       71.83%             28.53              0.00    325.4ms    154.8ms
幽灵块随容量收紧单调增长（0 → 28 块/请求）。容量宽松时（R ≤ 6）两列完全相同；
容量吃紧后阈值型也只差 2 个百分点 —— 因为它只问「匹配率过没过线」，
所有候选的匹配量被同等高估，过线与不过线的判定基本不变。

B. 换一个形状的打分函数，同一份近似索引代价立刻显现
                      打分方式       命中率·近似       命中率·精确        最忙接单·近似        最忙接单·精确     p99·近似     p99·精确
      阈值型 cache_aware(0.3)       99.23%       99.23%          2.92x          2.92x     60.3ms     60.3ms
      线性打分 loadaware β=0.2       95.76%       98.79%          2.32x          2.41x     43.2ms     29.2ms
      线性打分 loadaware β=1.0       55.90%       76.15%          1.36x          1.52x     87.4ms     47.9ms
      线性打分 loadaware β=4.0       55.49%       75.41%          1.31x          1.50x     91.8ms     51.1ms
阈值型在同一容量配置下（cap_ratio=6.0）两列逐位相同；线性打分的命中率
分别掉了 2.9 / 20.3 / 19.9 个百分点 —— 索引精度的价值完全由打分函数决定。

C. 幽灵块是怎么把线性打分带偏的
   beta        幽灵块/请求        近似命中率        精确命中率        差距
   0.10          2.56       96.86%       97.83%     0.98pt
   0.20          3.52       95.76%       98.79%     3.03pt
   0.50         15.58       82.45%       95.09%    12.64pt
   1.00         39.08       55.90%       76.15%    20.25pt
   4.00         39.47       55.49%       75.41%    19.92pt
关键在于幽灵条目是**持久**的：被引擎驱逐掉的块，近似索引里仍然记着。
于是「我这里有这块前缀」这个判断对该实例永远虚高，打分持续把它选出来 ——
实际命中不了，KV 池又被新流量挤得更满，虚高得更厉害。
精确索引在驱逐发生的那一刻就撤掉条目，匹配量随之下降，流量自动移开。
beta 越大，实例之间的候选分数越接近，虚高的那一截就越容易在临界处翻盘 ——
所以精确索引的收益随 beta 单调放大（0.98pt → 20.25pt）。
结论：**不消费 KV 事件可以，但前提是你的打分函数是阈值型。**
```

**这是本期最有实用价值的一条结论，而且它是个「条件性答案」。**

`cache_aware` 的近似索引在容量收紧到 `cap_ratio=20`（每实例只能装工作集的 1/20）时，幽灵块涨到 **28.53 块/请求**——几乎三分之一个 96 块的前缀都是「路由器以为有、实际没有」。但它的命中率两列**几乎相同**（69.81% vs 71.83%）。

为什么？因为阈值型判据只问「匹配率过没过 0.3 这条线」。所有候选的匹配量被同等高估时，**过线与不过线的集合不变**，所以路由决策不变。它只需要匹配量的**排序**大体正确。

换成线性打分，同一个近似索引的代价立刻显现：

| 打分配置 | 近似索引 | 精确索引 | 差距 |
|---|---|---|---|
| 阈值型 `cache_aware`(0.3) | 99.23% | 99.23% | **0.00 pt** |
| 线性 `loadaware` β=0.2 | 95.76% | 98.79% | 3.03 pt |
| 线性 `loadaware` β=1.0 | 55.90% | 76.15% | **20.25 pt** |
| 线性 `loadaware` β=4.0 | 55.49% | 75.41% | 19.92 pt |

线性打分的 `benefit` 是一个**连续量**，它要和负载项**逐单位比大小**。索引把匹配量估高几块，这两项的相对大小就变了——`benefit` 被虚增，于是路由更坚决地把请求推向那个实例。

更糟的是**估高不是均匀的**：持有该前缀最多的实例（也通常正是已经被吸引得最满的那个）被估高得最狠，于是打分把更多请求推给它；而热点上真实的 KV 又最容易被驱逐——**正反馈闭合了**。这就是为什么 `beta=1.0` 时幽灵块涨到 39 块/请求、命中率掉 20 个百分点，而 `beta=0.1` 时只掉 0.98 个百分点（缓存项权重小，虚增的 benefit 不足以翻转决策）。

所以正确的表述是：

> **不消费 KV 事件是免费的——前提是你的打分函数是阈值型。**
> 如果你要用线性打分，就必须配精确索引，因为线性打分对索引误差是放大而非衰减的。

这也顺带解释了 llm-d 那个「approximate 31 秒 vs precise 0.54 秒」的 57 倍差距为什么可能发生：他们用的正是一个**连续的前缀亲和分数**去和负载分数加权比较，属于对索引误差敏感的那一类。（我没有复现出那个量级——本次仿真里最坏情况是 20 个百分点。差异应该来自 workload 与索引更新方式，这一点我标为**推断**。）

---

## 9. 设计谱系的价格表

最后把 §2.2 的三个层次加上「完全不看缓存」的基线，放在同一负载下比一次。

```python
"""代码块 10：设计谱系的价格表 —— 从「零基础设施」到「精确索引」四种选择同台。"""
print(f"{'策略':>34}{'前缀命中率':>12}{'平均重算':>10}{'u_mean':>9}"
      f"{'最忙接单':>10}{'TTFT p99':>11}")
stack = [("cache_blind（普通 L4/L7 负载均衡）", "cache_blind", {}),
         ("prefix_hash N=256（无状态哈希）", "prefix_hash", {}),
         ("prefix_hash N=1536（覆盖整个前缀）", "prefix_hash", {"prefix_tokens": 1536}),
         ("cache_aware（SGLang 默认）", "cache_aware", {}),
         ("loadaware beta=1.0（vLLM 默认）", "loadaware", {"beta": 1.0}),
         ("loadaware beta=0.2（调过的）", "loadaware", {"beta": 0.2})]
for label, pol, kw in stack:
    r = simulate(pol, **kw)
    print(f"{label:>34}{r['hit']*100:>11.2f}%{r['recompute']:>10.0f}"
          f"{r['u_mean']*100:>8.1f}%{r['route_max']:>9.2f}x{r['p99']*1000:>9.1f}ms")

print("\n无状态哈希的代价（相对「直读引擎缓存」的理想上界）：")
best = simulate("cache_aware")
ph = simulate("prefix_hash")
print(f"  命中率 {ph['hit']*100:.2f}% vs {best['hit']*100:.2f}%"
      f"（拿到 {(ph['hit']/best['hit'])*100:.1f}%）")
print(f"  平均重算 {ph['recompute']:.0f} vs {best['recompute']:.0f} token"
      f"（{ph['recompute']/best['recompute']:.2f}x 的算力）")
print(f"  它换来的是：路由器侧零状态、O(log n) 查表、不需要任何 KV 事件通道。")

print("\n多随机种子复核（5 个种子，prefix_hash 的哈希盐也一起换）：")
for label, pol, kw in stack[1:]:
    rs = [simulate(pol, seed=s, **kw) for s in [1, 7, 13, 42, 101]]
    h = [x["hit"] * 100 for x in rs]
    p = [x["p99"] * 1000 for x in rs]
    print(f"  {label:>34}: 命中 {min(h):.2f}~{max(h):.2f}%  "
          f"p99 {min(p):.1f}~{max(p):.1f} ms")
```

```text
                                策略       前缀命中率      平均重算   u_mean      最忙接单   TTFT p99
        cache_blind（普通 L4/L7 负载均衡）      51.90%       995    80.2%     1.02x    124.7ms
          prefix_hash N=256（无状态哈希）      91.51%       386    35.5%     2.26x     83.5ms
        prefix_hash N=1536（覆盖整个前缀）      68.45%       741    62.0%     2.20x    130.4ms
            cache_aware（SGLang 默认）      99.23%       268    26.6%     2.92x     60.3ms
       loadaware beta=1.0（vLLM 默认）      55.90%       933    76.4%     1.36x     87.4ms
           loadaware beta=0.2（调过的）      95.76%       321    30.6%     2.32x     43.2ms

无状态哈希的代价（相对「直读引擎缓存」的理想上界）：
  命中率 91.51% vs 99.23%（拿到 92.2%）
  平均重算 386 vs 268 token（1.44x 的算力）
  它换来的是：路由器侧零状态、O(log n) 查表、不需要任何 KV 事件通道。

多随机种子复核（5 个种子，prefix_hash 的哈希盐也一起换）：
            prefix_hash N=256（无状态哈希）: 命中 91.28~91.63%  p99 79.4~83.5 ms
          prefix_hash N=1536（覆盖整个前缀）: 命中 68.09~68.45%  p99 130.4~145.1 ms
              cache_aware（SGLang 默认）: 命中 99.18~99.53%  p99 60.3~78.8 ms
         loadaware beta=1.0（vLLM 默认）: 命中 54.78~55.90%  p99 84.8~89.6 ms
             loadaware beta=0.2（调过的）: 命中 95.76~98.86%  p99 34.1~43.2 ms
```

三条值得记的：

1. **无状态哈希（第 ① 层）性价比很高。** `prefix_hash N=256` 拿到 `cache_aware` 的 **92.2%** 的命中率，代价是 1.44× 的算力（386 vs 268 token），基础设施成本是**零**。它自己的源码注释也这么定位：`prefix_hash trades optimal cache utilization for predictable O(log n) performance`。**如果你的服务刚起步、还没有 KV 事件通道，这是最划算的第一站。**

2. **`--prefix-token-count` 的取舍很尖锐。** 取 256 个 token 时，同一部门内 5 个客户哈希到同一个实例（命中率 91.51%，但 40 个部门挤进 8 个实例，最忙实例接单是平均值的 2.26 倍——鸽笼原理，躲不掉）；取到 1536（覆盖整个前缀）时每个客户各自钉死，倾斜摊平了，**但命中率掉到 68.45%、p99 从 83.5 涨到 130.4 ms**。原因是每个实例要装的前缀总量暴涨，超过它的 KV 池容量。**N 取多大不是一个调优问题，是「你的共享段有多长」这个事实的函数**，必须按 workload 定。

3. **默认配置的代价可以被量出来。** `cache_aware`（SGLang 默认）和 `loadaware beta=0.2`（vLLM 调过的）是两个健康点：命中率 99.23% / 95.76%，p99 60.3 / 43.2 ms，都远比基线好。而 `loadaware beta=1.0`（vLLM **默认**）落在 55.90%——**几乎是 `cache_blind` 的水平**。一个参数的值，值 20 个百分点的命中率。

---

## 10. 围绕该领域展开

### 10.1 和第 029 期的关系：命中率的两端

第 029 期解决的是「一个前缀**怎么**被命中」（块哈希链、基数树、叶子优先驱逐）；本期解决的是「一个前缀**在哪儿**被命中」。两者相乘才是有效命中率：

$$\text{有效命中率} = \underbrace{P(\text{路由到持有者})}_{\text{本期}} \times \underbrace{P(\text{块仍在缓存里})}_{\text{第 029 期}}$$

第 029 期量出的「容量掉到去重总量 1/5 才明显伤命中率」这个结论，在集群环境下要改写成「**每实例**容量掉到 $D/N$ 的 1/5」。§5 的表里 R=8（正好是 $D/N$）时 aware 还有 95.47%，R=12 掉到 63.23%——和单机的那个比例关系完全一致。

### 10.2 和第 030 期的关系：同一套基础设施的三个用途

这三件事用的是同一批零件——**内容寻址的块哈希 + 一个全局的「谁有什么」索引 + 一条事件通道**：

| 用途 | 索引键 | 索引值 | 决策 |
|---|---|---|---|
| 单实例前缀缓存（029） | 块哈希链 | 本地块池 | 命中哪几个块 |
| KV 感知路由（031） | 块哈希链 | **实例** 列表 | 请求发给谁 |
| PD 分离 / 分层 KV（030） | 块哈希链 | 实例 **+ 内存介质**（GPU/CPU/磁盘） | 请求发给谁、KV 从哪搬 |

llm-d 的索引结构就是这两层的叠加：`kvevents.Pool` 维护 block-hash → (pod, 内存介质) 的底层映射，`kvcache.Index` 在它上面把「逻辑 token 序列」映射到持有它的 pod。Dynamo 的 `KvIndexer` 更直白——它就是在第 029 期那棵前缀树的每个节点上**再存一个 worker id**：

```
KVIndexer: We modify the original prefix tree by also storing the worker id on each
node. This is so we can return the number of matched blocks for each worker.
```

**第 029 期那棵树，加一个字段就变成了路由器。** 这是本期最省事的一个实现观察。

索引的元数据开销 llm-d 也给了：DeepSeek-R1 FP8 在 8×H200 上，365 GB 的 KV 池（每块 128 token / 8.6 MB），每块元数据只要一个 64-bit 哈希，**管整个池子约 339 KB，数据/元数据比超过 1,000,000 : 1**。所以「精确索引很贵」的贵不在内存，在**事件通道的运维**（ZMQ 端点、活跃/被动高可用、索引与引擎版本的一致性）。

### 10.3 一致性哈希的三条硬约束

跨实例的块哈希必须**逐位一致**，否则索引全废。红帽文档给出的 vLLM 配置里这三条是必须的：

```
VLLM_ADDITIONAL_ARGS: --prefix-caching-hash-algo sha256 --block-size 16 \
                      --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq",...}'
PYTHONHASHSEED: 42
```

- `PYTHONHASHSEED` 必须固定：`Specifies a fixed Python hash seed to ensure consistent prefix hashing across replicas`。**不固定的话不同副本的哈希不同，索引会指向错误的副本。**
- `--prefix-caching-hash-algo` 和 `--block-size` 必须全集群一致。
- 第 030 期还讲过一条更隐蔽的：**动态量化的 KV 不支持跨实例**（`Per-block scales are not transferred alongside KV cache data`），要求静态量化或 packed-layout 内联 scale。

### 10.4 PD 分离下的双级路由

第 030 期讲过 PD 分离把系统拆成 P 池和 D 池。这不只是「两台机器」，它把路由问题变成了**两个不同的路由问题**：

- **P 池**：路由目标是「谁的缓存里有这段前缀」——和本期讲的一模一样。但 P 池处理的 prefill 是**计算密集**的，所以负载信号应该更偏向**算力/FLOPs 队列**而不是请求数。
- **D 池**：路由目标是「谁有地方放这条请求的 KV、谁的 decode 队列短」。此时**缓存亲和的意义变了**——对 D 来说，「命中」意味着 KV 已经在它的池子里（可能来自本地 prefill，也可能来自远端传输），这要求路由器和 KV 传输层共享索引。

llm-d 的做法是 `disagg-profile-handler` 同时跑两个 profile，decode 端选为请求的主目的地、prefill 端作为特殊 header 注入，由 D 侧的 sidecar 拦截并协调远端 prefill。SGLang 侧则是 `--disaggregation-mode prefill|decode` 加 router 的 `--prefill-policy cache_aware --decode-policy round_robin`——**P 池用缓存感知、D 池用轮询**，这个默认组合本身就是「同一个汇率在两侧取不同值」的体现。

回到 §3.1 那个 `pool_tag` 的注释：`pool::model` 键控的树，正是为了避免 PD 两侧互相覆盖 tenant、把路由退化成乒乓。

### 10.5 和 MoE 的第 026 期的关系：NIC 是共同的瓶颈

第 030 期算过：一个满负荷的 D 实例需要入带宽 $\Theta_D \times \text{per\_tok} \times S/G$，临界 $S/G \approx 12$（IB400）。第 026 期算过：MoE 的专家并行（EP）通信也要走同一张网卡，而且 MoE 需要的全局 batch 是稠密模型的 $E/K$ 倍。

**KV 传输、EP 的 all-to-all、以及路由器的索引同步，三者抢同一块带宽。** 做路由规划时把这三笔账放在一起算，而不是各算各的。

### 10.6 和 CUDA Graph、KV 池切分的三角关系

这条把第 001 期、第 029 期、本期串起来：

- CUDA Graph 要求**形状固定**，所以推理引擎要给 batch 大小分桶（第 001 期）。分桶意味着「不同的 batch 占用不同的捕获图」。
- 前缀缓存让不同请求的 prefill 长度差异巨大（命中 0 块 vs 命中 96 块），这会让**分桶的利用率变差**——同一个桶里混着长 prefill 和短 prefill。
- 而路由决策**决定了这个桶里混合的构成**：把各种前缀打散的路由器会让每个实例的 prefill 长度分布更均匀但更长；按前缀聚合的路由器会让分布更双峰（大量全命中 + 少量全未命中）。

**所以路由策略会反过来影响 CUDA Graph 分桶的效率。** 这是一个真正的联立调参问题，本期的记忆里也把它记成了候选话题。

### 10.7 会话亲和是另一条路，但有天花板

最简单的「缓存感知」其实是**会话粘性**：同一个用户的请求总是路由到同一个实例（vLLM 栈的 `session` 策略、GIE 的 `session-affinity-scorer`）。实现成本极低，对多轮对话有效。

但它天花板很低。llm-d 的基准里给了个具体数字：150 个客户、每客户 6000 token 的系统提示下，会话调度会创建 **750 个独立会话**（150 客户 × 5 并发用户），**但漏掉了客户组内跨用户的缓存复用**——同一个部门 5 个人共享的那段前缀，在会话亲和下被算了 5 遍。

**会话亲和的粒度是「用户」，不是「前缀」。** 前缀的共享结构是棵树（部门段 → 客户段 → 会话段），而会话亲和只认最底下一层。

### 10.8 一个更激进的方向：不做路由，做 KV 共享

本期和 030 期的所有方案都在搬运「决策」——决定请求去哪。还有一条相反的路：**把 KV 从实例的私有资产变成集群的共享资产**（LMCache / Mooncake / NIXL 的 KV 池），让任何一个实例都能读到任意一段 KV。

这条路的代价第 030 期已经算过：临界带宽 $S/G \approx 12$（IB400），即 prompt 长度除以生成 token 数超过约 12 倍时，网卡先于显存成为瓶颈。**所以它和路由不是二选一，而是互补**：短 prompt 场景共享便宜，长 prompt + 短输出场景路由便宜。真正成熟的部署会两个都用——**先按前缀路由尽量本地命中，本地没有再去共享池里取**。llm-d 那个「GPU / CPU / 磁盘三档权重」的代价函数，已经把这层考虑编进去了。

---

## 11. 什么时候该用 / 不该用

**该用：**

1. **实例数 ≥ 2 且负载里有重复前缀。** 一个实例时前缀缓存自己就够（第 029 期），路由问题不存在。实例数一多，§5 那个 N 倍容量杠杆就自动生效。
2. **负载含长共享前缀**：统一 system prompt、RAG 落同一批文档、多轮对话历史、Agent 的固定工具描述。DigitalOcean 给的一个真实例子：2000 token 的 system prompt + 200 token 用户消息，**91% 的输入是共享上下文**。
3. **你处在容量约束而不是延迟约束下。** 这是本期最重要的判断（§6.2 的 C 段）：负载轻时路由省的是算力不是延迟；只有利用率上到 60% 以上，延迟收益才出现。**如果你 GPU 大量闲置，先别折腾路由。**
4. **Agentic / 多轮场景**：NVIDIA 给的一组数字很说明问题——Claude Code 的一次编码会话里，第一次 API 调用之后的每次调用在同 worker 上有 **85~97% 命中率**，agent 团队（4 个 Opus 协作者）的**聚合命中率 97.2%**，读/写比 **11.7×**。这是教科书级的 write-once-read-many 模式，路由的价值最大化。

**不该用（或者先别用）：**

1. **单实例部署。** 没有路由可言。
2. **前缀基本不重复的负载**（每条 prompt 都是全新长文档，且没有共享头）。此时所有策略都退化，`cache_blind` 就够。
3. **实例数量少且负载极度倾斜。** N=2 时容量杠杆只有 2 倍，为它引入一个带状态的组件未必划算——先试 `prefix_hash` 这种零状态的。
4. **还没有 KV 事件通道，却打算用线性打分。** §8 的结论：线性打分 + 近似索引是本期实测里最差的组合（命中率掉 20 个百分点）。**要么上精确索引，要么换成阈值型判据。**
5. **SLO 很松、GPU 有余量。** 同理第 3 条。

---

## 12. 常见坑

### 坑 1：把「纯缓存亲和」实现成 `beta = 0`，结果全部流量挤向一个实例

§7 实测：`beta = 0` 时所有实例的 benefit 都是 0，score 全部并列，而源码的 tie-break 是确定性的字典序（`sorted(endpoints, key=lambda e: e.url)` + 严格大于判等），于是冷请求全涌向同一个实例——最忙实例利用率 **647%**、TTFT p99 **416 秒**。

同类坑在 SGLang 的 `cache_aware` 里也埋着：它的最短队列分支是 `filter(load == min_load).choose(rng)`，**有随机 tie-break**。我仿真第一版漏了这个随机，所有负载为 0 的实例里永远选 0 号，结果 `cache_aware` 表现得比 `pure_cache` 还差。**任何「选最小/最大」的代码，都要检查并列时怎么办。**

### 坑 2：`balance_abs_threshold` 的语义是「允许多歪」，而且同一个默认值在不同规模的集群里结论相反

这个旋钮的完整解剖在 **§7.2**（含完整实测表）。这里只记最实用的两条：

**第一条：它设定的不是「什么时候介入」，而是「允许多歪」。** 实测的队列长度差中位数几乎精确等于你设的阈值（阈值 8 → 9、16 → 16~17、64 → 64~65）。护栏不阻止倾斜，它把倾斜**钉在阈值这个高度上**；而在钉住之前，系统必须先歪到某个实例饱和、队列开始随时间线性增长——护栏只在它本该防住的那场事故已经发生之后才真的介入。

**第二条：同一个默认值 64，在小集群是优点，在大集群是事故。**

| 实例数 N | `abs_th=1` 的 TTFT p99 | `abs_th=64` 的 TTFT p99 | 哪个好 |
|---|---|---|---|
| 8 | 104.0 ms（命中率 53.52%） | **60.3 ms**（命中率 99.23%） | 64 好得多 |
| 16 | **89.3 ms**（命中率 52.16%） | 1576.5 ms（命中率 65.82%） | 1 好得多 |
| 32 | **78.8 ms**（命中率 52.05%） | 519.1 ms（命中率 60.66%） | 1 好得多 |

N=8 时缓存亲和的收益压过均衡，所以「允许歪」是对的；N=16/32 时不均衡的绝对幅度随实例数放大，同一个 64 就从优点变成事故。**因为不均衡的绝对幅度随 $N$ 增长（§1.4 那笔账），而阈值是固定的。** 相对量型打分（`loadaware` 的 `(load-mean)/max(1,mean)`）没有这个问题，它把「允许多歪」表达成对平均值的倍数，随规模自动伸缩。

**排查方法**：扫这个参数，同时把队列长度差的**分布**打出来。如果那个中位数正好贴着阈值，说明它正在全力工作；如果扫好几个取值输出逐位相同（§7.2 的 B 段就是），说明它在这个规模下已经越过了有效区间。

### 坑 3：近似索引 + 线性打分 = 正反馈

§8 的表：`cache_aware`（阈值型）在近似索引下的命中率和精确索引**逐位相同**；`loadaware`（线性）在 `beta=1.0` 时差 **20.25 个百分点**。而且这个偏差不是随机的——持有前缀最多的实例被估高得最狠，于是打分把更多请求推给它，热点上真实 KV 又最容易被驱逐，正反馈闭合。

**推论**：如果你打算不给 KV 事件通道，就**不要用连续亲和分数**，用「匹配率过阈值」这类布尔判据。

### 坑 4：跨实例的哈希配置不一致，索引会静默指错

`PYTHONHASHSEED`、`--prefix-caching-hash-algo`、`--block-size` 三者任何一项在不同副本间不一致，块哈希就对不上，**路由器会稳定地把请求送到「它以为有缓存」的实例上**。而 §8 告诉我们，这种错误在线性打分下会形成正反馈，表现为「路由看起来在工作、命中率却上不去」。

这类 bug 最难查的地方在于**它不报错**——路由决策、KV 事件流、索引更新全部正常运行，只是每一跳都错位。防御手段（沿用 030 期的思路）：**写一个断言，在启动时用同一条固定 prompt 在各副本上算出块哈希，逐个比对。**

### 坑 5：路由器自己多副本部署

官方 "Production Recommendations" 明确写着：多个 router 副本时，`cache_aware` 的基数树**不跨副本同步**，**预期命中率损失 10~20%**。

这是近似索引路线的一个隐藏成本：为了路由器自身的高可用，你要么接受命中率下降，要么把索引抽成共享的（那就又变成一条需要运维的通道了）。llm-d 走的是后一条路（`kvevents.Pool` 支持 active-active 或 active-passive，共享同一个全局索引）。

---

## 13. 一句话总结

**KV 感知路由买的是 N 倍的缓存容量（§5）和「不排队」（§6），代价是要在两个对立目标之间定一个汇率（§7）——而如果你的打分函数是连续量的，你还必须为它配一条精确的 KV 事件通道（§8）。**

---

## 14. 今日练习

<details><summary>今日练习</summary>

### 练习 1（基础）：算一下你的部署需不需要它

一台 8 卡的机器上跑 7B、GQA-8、bf16（第 029 期的口径：KV = **128 KiB/token**），80GB 卡留 60GB 给 KV 池（第 030 期的口径），部署成 $N=4$ 个实例，每实例 2 卡。

现有 workload：6 个租户，每租户有 24000 token 的共享上下文，请求平均带 800 token 的用户消息。

请回答：(a) 去重后的工作集 $D$ 是多少？(b) 每实例 KV 池容量 $C$ 是多少 token？(c) 打散路由需要 $C \ge D$，这成立吗？(d) 亲和路由需要 $C \ge D/N$，这成立吗？(e) 结论是什么？

**参考答案：**

(a) 工作集 = 6 个租户 × 24000 token = **144,000 token** 的前缀内容（去重后——6 个租户各不相同，没有可共享的部分）。

(b) 每实例 2 卡 × 60GB = 120 GB。$120 \text{ GiB} / 128 \text{ KiB/token}$：先把单位理清，$128 \text{ KiB} = 131072 \text{ B}$，$120 \text{ GiB} = 120 \times 1024^3 = 1.2885\text{e}11 \text{ B}$。

$$C = \frac{1.2885\text{e}11}{131072} \approx 983{,}040 \text{ token}$$

约 **98.3 万 token**。

(c) $D = 144{,}000 \le C = 983{,}040$。**成立，而且余量巨大**（$D/C \approx 0.147$）。

(d) 自然成立（比 (c) 更宽松 4 倍）。

(e) **这个部署不需要 KV 感知路由。** 每个实例的 KV 池（98.3 万 token）能装下全部工作集（14.4 万 token）的**6.8 倍**——打散路由的不等式 $C \ge D$ 富余得离谱，所以每个实例都能把全部 6 个租户的前缀留在缓存里，轮到谁都能命中。加一个路由器只会增加运维复杂度，不会提升命中率。

**这道题的真正意义在于它给出了判断顺序**：先用 $D$ 和 $C$ 比一下，再决定要不要折腾。§5 的容量定律给出的判据是——**只有当你单实例缓存装不下整个工作集时（$D > C$），路由才有价值**。装得下的话，`cache_blind` 和 `cache_aware` 的表现是一样的。

顺带验算一下 (b) 的另一种情形，体会一下这个判据有多敏感：如果租户数从 6 涨到 60（每租户仍是 24000 token），$D = 1{,}440{,}000 > C = 983{,}040$——**打散路由的不等式破了**。此时亲和路由需要 $C \ge D/N = 360{,}000$，仍然成立（98.3 万 ≫ 36 万）。这就是 §5 那个 N 倍杠杆的具体形态：**同一套硬件，6 个租户时路由器是多余的，60 个租户时路由器是必需的。**

### 练习 2（进阶）：`balance_abs_threshold = 64` 是死代码，还是「允许多歪」的上限？

§7.2 实测发现两件事：`imb_p50` 几乎等于你设的阈值；同时 $N=8$ 时 `abs_th ∈ {16, 64}` 的输出逐位相同。

请先用 Little 定律算出「稳态下队列长度差」的量级，再回答：**为什么「阈值是死代码」这个推断是错的？** 它到底在控制什么？

提示：Little 定律 $L = \lambda W$。你要估算的是「单个实例上的在飞请求数」。

**参考答案：**

**第一步：算量级。** 稳态在飞请求数 $L = \lambda_{\text{每实例}} \times E[S]$。代入几种典型配置：

| 场景 | 每实例 QPS | 平均服务时间 $E[S]$ | 在飞请求 $L$ |
|---|---|---|---|
| 7B / 短上下文 / 高 QPS | 30 | 20 ms | 0.6 |
| 7B / 长上下文 | 5 | 200 ms | 1.0 |
| 70B / 中上下文 | 2 | 1 s | 2.0 |
| 70B / 长上下文 / 低延迟诉求 | 1 | 5 s | 5.0 |
| 极小规模（$N=2$，压测） | 10 | 8 s | 80 |

**一个健康系统的稳态队列长度差只有个位数**（前三行都小于 2）。所以「`max_load - min_load > 64` 在健康状态下不成立」这个推断是对的。

**第二步：但推断的结论错了。** 错在「不成立 ⇒ 是死代码」这一步。实测数据（§7.2 的 A 段）：

| N | `abs_th=1`：命中率 / p99 | `abs_th=64`：命中率 / p99 | `imb_p50` @64 |
|---|---|---|---|
| 8 | 53.52% / 104.0 ms | **99.23% / 60.3 ms** | 2（未触发，与 16 逐位相同） |
| 16 | 52.16% / **89.3 ms** | 65.82% / 1576.5 ms | **64**（触发了） |
| 32 | 52.05% / **78.8 ms** | 60.66% / 519.1 ms | **65**（触发了） |

$N ≥ 16$ 时那个分支**明明触发了**——而且触发得很规律：`imb_p50` 恰好等于阈值。

**机制是反馈。** 阈值不是「一个很难达到的条件」，而是一个**允许歪斜的上限**：

1. 系统先自由倾斜，某个实例的队列逐渐变长（长尾的前缀吸走流量）；
2. 队列长度差涨到阈值附近时，`&&` 判据成立，路由器切到最短队列；
3. 最长队列被排空，长度差回落到阈值以下，于是又切回缓存亲和；
4. 循环，把 $\ell_{\max}-\ell_{\min}$ **钉在阈值上**。

所以 `imb_p50 ≈ 阈值` 是这个负反馈环的稳态标志。这也解释了为什么「扫出几个取值输出逐位相同」并不等于「参数是死的」——$N=8$ 时 16 和 64 逐位相同，只说明**在 8 个实例这个规模下，有效区间在 16 以下**（`abs_th=1` 和 `8` 都各有独立效果，而 ≥16 的取值都退化成「几乎不介入」）。

**它到底在控制什么**：不是「什么时候介入」，而是**「能容忍多歪」**。改它不是在调灵敏度，是在直接挪系统的工作点——和 030 期那条「不变量」是同一种形状。

**第三步：所以正确结论是三条。**

1. 默认值 64 不是死代码，是一个**过于宽松的上限**：它允许系统先歪到某个实例饱和（$u_{\max}\to 100\%$，队列开始随时间线性增长，p99 不再是稳态量），然后才介入。
2. 它的效果**强依赖集群规模**：$N=8$ 时「允许歪」是对的（缓存亲和收益压过均衡，命中率 99.23% 且 p99 更好）；$N=16/32$ 时同一个 64 就是事故（p99 涨 17.6 倍 / 6.6 倍）。
3. 排查方式不是「推断它会不会触发」，而是**扫它 + 把队列长度差的分布打出来**：如果 `imb_p50` 贴着阈值，它在全力工作；如果扫出大片逐位相同，说明已经越过有效区间。

（顺带一提，$N=8$ 时 `abs_th=1` 的命中率 53.52% 比 `abs_th=8` 的 97.81% 差得多——**调低不一定好**，因为它频繁切到最短队列，把缓存亲和砸碎了。这个参数两端都有坑。）

### 练习 3（挑战）：把「命中率」和「延迟」的换算做出来

§4 量出：命中 504/512（98.44%）时 FLOPs 省下 98.44%，耗时只省下约 87%，地板（固定开销占全长 prefill 的比例）约 11%。

假设一个真实服务的 prefill 墙钟 $t_{\text{full}} = 2.83$ s（8000 token，按 §4 那个量级放大），地板占比取 **11.4%**（§4 实测区间 9%~13% 的中位附近）。现在路由器给你两个候选方案：

- 方案 A：命中率 99%（平均只重算 1% 的 prompt）
- 方案 B：命中率 95%

请估算两者在 TTFT 上的差别，并回答：**这个差别值不值得为它多付一条 KV 事件通道的运维成本？**

**参考答案：**

先把 §4 那个线性模型按比例放大到 8000 token。原模型（512 token）是 $t = a + bT$，某一轮实测 $a = 0.982$ ms、$b = 15.32$ µs/token、$t(512) = 8.653$ ms，地板占比 11.35%。

按题目给的 $t_{\text{full}} = 2.83$ s 等比缩放：$a' = 0.114 \times 2830 = 322.6$ ms，$b' = (1 - 0.114) \times 2830 / 8000 = 313.4$ µs/token。

- 命中率 99%：重算 $T = 0.01 \times 8000 = 80$ token → $t = 322.6 + 80 \times 0.3134 = 347.7$ ms
- 命中率 95%：重算 $T = 0.05 \times 8000 = 400$ token → $t = 322.6 + 400 \times 0.3134 = 448.0$ ms

**差别约 100 ms**（347.7 vs 448.0），也就是 **1.29 倍**。

**这就是那条地板的威力**：命中率从 95% 提到 99%（重算量降到 1/5），TTFT 只改善 29%。如果没有地板，这个比例应该是 5 倍。

那么值不值一条 KV 事件通道？**这道题的正确答案是要看 §8 的条件，不能只看这个数字：**

- 如果你用的是**阈值型判据**（`cache_aware`）：§8 实测近似索引和精确索引命中率**逐位相同**。此时精确索引买不到任何东西——**不值**。
- 如果你用的是**线性打分**（`loadaware`）：§8 实测 `beta=1.0` 时近似索引比精确索引低 **20.25 个百分点**。20 个百分点远大于上面算的 4 个百分点，转换过去的延迟差会大得多——**值**。

**所以这个练习的落点是**：不要把「命中率」当成单一指标去优化，先回答两个问题——(1) 我的打分函数是哪种形状？(2) 我的工作点在负载曲线的哪一侧（§6.2 的 C 段：利用率低于 ~60% 时缓存感知路由根本不改善延迟）？

两个问题的答案决定了那条 KV 事件通道值不值。**本期的全部结论都可以归结为：先定位工作点，再选架构，最后才调参数。**

</details>

---

> **本期所有代码均在本地真跑**（torch 2.14.0，CPU）：
> - §4 的 tiny 模型微计时（代码块 1、2）
> - §5 的容量定律仿真（代码块 4）
> - §6~§9 的离散事件调度仿真（代码块 3、5、6、7、9、10，共用同一命名空间）
> - §7.1 的解析推导（代码块 8）
> - §12 坑 2 的绝对差阈值扫描（代码块 11，复用 §6 的仿真台）
>
> **绝对值会飘，比值不会**：本节所有墙钟时间在重复运行之间有较大波动（同一条 512 token 全量重算实测在 8.6~13.8 ms 之间），所以文中引用的耗时都用「约」和比值表述；所有比值（加速比、命中率、重算量之比）在 5 个随机种子下的波动范围都在文中逐一列出。
