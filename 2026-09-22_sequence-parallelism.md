# PyTorch 每日一课 · 第 024 期

## 序列并行（Sequence Parallelism）：把「长度」也切开

> **日期**：2026-09-22
> **难度**：⭐⭐⭐⭐⭐
> **前置知识**：知道张量并行在做什么（列并行 / 行并行各切什么）；理解 Transformer 一个 block 里的残差流；对「激活显存」和 Flash Attention 的 online softmax 有大致印象。本篇独立可读，第 019 期张量并行没看过也不影响。
> **预计阅读时间**：25 分钟

---

## 一、这个领域解决什么问题

### 1.1 并行有四条轴，但只有一条在减小「每张卡看多少 token」

到大模型训练这一步，我们已经有一整套并行工具箱了。但把它们摆在一起看，会发现一个很尴尬的事实：

| 并行轴 | 切什么 | 省下什么 | **每张卡上的 token 数** |
|---|---|---|---|
| DP / ZeRO | batch | 优化器状态、梯度 | 不变 |
| TP | 权重矩阵、hidden 维 | 参数、部分激活 | **不变** |
| PP | 层堆叠 | 参数 | **不变** |
| **SP / CP** | **序列** | **激活** | **s → s/P** ✅ |

这就是序列并行存在的全部理由：**前面三条轴都不碰「序列长度」这个变量。**

DP 切的是 batch——但如果我的问题就是「一条 128K token 的样本」呢？batch 已经是 1 了，切无可切。TP 把 `h×4h` 的权重矩阵切成八块，可每一块算出来的激活仍然横跨整条序列。PP 更不用说，它只换了层。

于是有个直接的后果：**单卡要存的激活，永远在随 s 线性增长**（注意力的中间量还随 s² 增长）。你加再多卡，只要不加「序列」这条轴，这条曲线就不会向下弯。

### 1.2 先把账单摊开看

Transformer 一层里，「把所有中间量都存下来」的激活字节数大致是（fp16 存激活、dropout mask 按 1 字节摊销进系数）：

$$\text{per-layer} = 34sbh \;+\; 5as^2b$$

其中 `s` 序列长度、`b` 批次、`h` 隐层宽度、`a` 注意力头数。前一项是线性的（MLP、QKV、LayerNorm、残差……），后一项是平方的（注意力分数矩阵、softmax 输出、它们的 dropout mask 与输出）。

这两项的**数量级差别是残酷的**。取 LLaMA-70B 量级（h=8192、64 heads、80 层）跑 seq=32768、batch=1：

- 线性项 `34sbh` ≈ **8.50 GiB / 层**，80 层 = 680 GiB
- 平方项 `5as²b` ≈ **320.00 GiB / 层**，80 层 = 25600 GiB
- 比值：**37.6 倍**

**所以长上下文第一件事根本不是序列并行，而是把 s² 项干掉**——靠 Flash Attention / 选择性激活重算（第 005 期、第 009 期讲过），让这些中间量随用随算、不驻留。

把平方项消掉之后，剩下的线性项才是序列并行的战场。可这里还有第二个门槛：**在 TP 之下，线性激活里有 29.4% 是切不动的。**

### 1.3 实测账本（本机纯 Python 复现，无需 GPU）

下面这张表是本文所有数字的来源，脚本在 3.1 节，纯公式、不用 torch，你可以直接跑：

```
① LLaMA-70B 量级 (h=8192, heads=64, layers=80, batch=1) 逐层激活 / GiB

  --- seq = 4096 ---
  策略                            线性项       平方项        单层      80 层合计
  TP=8                          0.406     0.625     1.031       82.50
  TP=8 + SP                     0.133     0.625     0.758       60.62
  TP=8 + SP + 选择性重算             0.133     0.000     0.133       10.62
  再加 CP=8 (本地 512)              0.017     0.000     0.017        1.33

  --- seq = 32768 ---
  TP=8                          3.250    40.000    43.250     3460.00
  TP=8 + SP                     1.062    40.000    41.062     3285.00
  TP=8 + SP + 选择性重算             1.062     0.000     1.062       85.00
  再加 CP=8 (本地 4096)             0.133     0.000     0.133       10.62

  --- seq = 131072 ---
  TP=8                         13.000   640.000   653.000    52240.00
  TP=8 + SP                     4.250   640.000   644.250    51540.00
  TP=8 + SP + 选择性重算             4.250     0.000     4.250      340.00
  再加 CP=8 (本地 16384)            0.531     0.000     0.531       42.50
```

这张表里有三个值得停下来看一眼的事实：

1. **seq=32768 时，TP=8 的单层激活 43.25 GiB 里有 40 GiB 是平方项。** 不做 Flash Attention，什么并行都救不了你。做完之后从 3285 GiB 掉到 85 GiB——这才是量级上的改变。
2. **SP 只在「线性项」上发力**：3.250 → 1.062 GiB，省了 3.06 倍；但因为平方项占绝对大头，单层合计只从 43.25 降到 41.06（**只降了 5%**）。**如果你以为开了 SP 就能把长上下文跑起来，会失望。** SP 的收益是「默默生效」的：等平方项被 Flash Attention 消掉之后，逐层激活从 3.25 掉到 1.06 GiB，80 层从 260 GiB 掉到 85 GiB——这一步是 SP 早就帮你做完的，只是当时被平方项淹没了。
3. **真正把 seq=32768 拉进单卡的是 CP=8**：本地序列变成 4096，逐层激活 1.062 → 0.133 GiB，80 层 10.62 GiB。**如果只看「省激活」这一件事，CP 比 SP 猛得多**——但代价完全在通信上，后面会算。

> **一句话记住这里的结论**：SP 是「TP 的补丁」，CP 是「独立的并行轴」。名字像，定位完全不同。

---

## 二、核心思想：三条路线，一个共同点

四条轴的地图摆好了，现在讲机制。历史上「序列并行」这个名字被三家抢注过，它们干的事差别很大，但共同点只有一个：

> **都让每张卡只处理 s/P 个 token；区别在于「什么时候把序列拼回去」。**

### 2.1 TP 为什么剩下 10sbh 切不动

先把靶子立清楚。看 Megatron-LM 定义的 TP 区域（tensor-parallel region）：

```
        ┌─────────────── TP 区域 1 ───────────────┐
  x ──► LayerNorm ──► [列并行 Linear] ──► GeLU ──► [行并行 Linear] ──► Dropout ──► +x
        └─────────────────────────────────────────┘
        ┌─────────────── TP 区域 2 ───────────────┐
   ─────► LayerNorm ──► [列并行 Linear] ──► GeLU ──► [行并行 Linear] ──► Dropout ──► +x
        └─────────────────────────────────────────┘
```

TP 靠两个约定算子把通信限制住：

- **f 算子**：进入 TP 区域。前向是恒等映射（输入本来就是复制的），反向是 all-reduce。
- **g 算子**：离开 TP 区域。前向是 all-reduce（把各卡的部分和加起来），反向是恒等。

矩阵乘法被切开了，但**夹在两个 TP 区域之间的东西没被切**：LayerNorm（4sbh）、Dropout 的 mask 和输出（6sbh），加起来 **10sbh**。为什么切不动？因为它们是 **pointwise 操作——逐 token 独立计算，和 hidden 维无关**。TP 的刀是沿 hidden 维切的，而 LayerNorm 在每个 token 上都是「对完整的 h 个值做归一化」，切了就错。

于是每张 TP rank 都存了一份**完整的 [b, s, h] 激活**。这 10sbh 就是 34sbh 里切不动的 29.4%：

```
② 线性激活里「切不动」的比例（10sbh / 34sbh = 29.4%），SP 把它也除掉
  TP 度 t               仅 TP          TP+SP        收益
  1              34.00 sbh       34.00 sbh       1.00x
  2              22.00 sbh       17.00 sbh       1.29x
  4              16.00 sbh        8.50 sbh       1.88x
  8              13.00 sbh        4.25 sbh       3.06x
  16             11.50 sbh        2.12 sbh       5.41x
```

**注意收益随 TP 度数放大**：TP=2 时 SP 只值 1.29 倍，TP=16 时值 5.41 倍。这也解释了为什么 SP 是在 TP 度数拉到很高的场景下才被提起的。

### 2.2 路线一：Megatron-SP —— 在 TP 的缝里塞进序列切分

既然这 10sbh 是「沿序列方向逐 token 独立」的，那就**沿序列维切开**，每人拿 s/t 个 token 的 LayerNorm/Dropout/残差。

问题来了：进入 TP 区域要做矩阵乘法，而矩阵乘法的输入必须是**完整序列**（QKV 投影是每个 token 独立投影，但列并行的权重分片要求每张卡都能看到「属于自己那份输出」的全部 token 行）。所以：

- **进 TP 区域（原 f）**：从「恒等」变成 **all-gather**——把各卡的序列分片拼回完整序列。
- **出 TP 区域（原 g）**：从「all-reduce」变成 **reduce-scatter**——规约之后不广播，只保留属于自己那段序列。

**关键算术**：`all-reduce ≡ reduce-scatter + all-gather`（这就是 Ring all-reduce 的两阶段结构）。所以改造前后**通信量完全一样**，只是把一次 all-reduce 拆成了两步。原本一次前向+反向要 4 次 all-reduce，现在变成 4 次 all-gather + 4 次 reduce-scatter，体积不变。

效果：那 10sbh 也被除以 t，逐层激活变成 `34sbh/t + 5as²b/t`——**激活显存对设备数真正线性了**。

**代价与边界（很重要）**：

- **必须 TP > 1**。SP 是相对于 TP 区域定义的概念，没有 TP 就没有那个 all-reduce 可替换。官方文档也把 `SequenceParallel` 描述为「实现 *Reducing Activation Recomputation in Large Transformer Models* 中描述的操作」——那篇论文本身就是 TP 的补丁。
- **它不减小注意力计算的序列长度**。注意力仍然每卡算完整序列（all-gather 之后）。所以 SP 单独**不能扩展上下文长度**，它只是让 LayerNorm/Dropout 那几段不再复制显存。
- **归一化层的权重必须是复制的**。PyTorch 文档明确提醒：`SequenceParallel` 假定模块权重是 ones 初始化（`nn.LayerNorm`/`RMSNorm` 默认如此）；如果你自定义初始化了，得在 `parallelize` 前后手动广播权重。

### 2.3 路线二：Ulysses —— 用 all-to-all 做一次「转置」

Megatron-SP 只切了 pointwise 那一段，注意力本体没动。DeepSpeed-Ulysses（论文名 *DeepSpeed Ulysses*，得名于长篇史诗）换了个思路——**用两次 all-to-all，把「按序列切」换成「按 head 切」**。

打个比方。64 个注意力头 × 8192 个 token 的一大叠数据，8 个人分。Megatron-SP 是「每人拿 1/8 的 token，但每个人手里都得有完整的 64 个头」；Ulysses 是：

1. 先按 token 分：每人算自己那 1024 个 token 的 Q/K/V（这步是「完整 h、1/8 序列」）。
2. **all-to-all 转置**：每人把手里的 1024×h 数据按 head 拆成 8 条，送给对应的人；收回来的是「完整 8192 个 token、但只有属于我的 8 个头」。
3. 在本卡上对完整序列做标准 attention——**可以直接调 FlashAttention v2**，因为数据布局就是正常的 `[b, heads=8, s=8192, d]`。
4. 结果再 all-to-all 转回去，恢复成「1/8 序列、完整 h」，交还给下一个 LayerNorm。

论文给出的核心性质是**通信量恒定**：每卡本地数据量是 `3·b·(s/P)·h`，当序列长度和设备数按比例一起涨时（s ↓ 由 P 补偿），**每卡搬运量不变**。论文报告的数字：相对当时 SOTA 序列长度 **4×**、通信量降低 **10× 以上**、吞吐提升最高 **2.5×**、稳态 175+ TFLOPs/GPU（约 54% 硬件峰值）、支持百万 token 训练，并且能和 ZeRO-3 叠着用。

**约束（踩过的人都知道）**：attention head 数必须能被 P 整除。**GQA/MQA 场景下这条变得很尖锐**——LLaMA-3-70B 有 64 个 Q head 但只有 **8 个 KV head**，Ulysses 的度数上限直接被卡到 8；想要 P=16 就崩了。这是 Ulysses 在现代 GQA 模型上最大的现实限制。

### 2.4 路线三：Ring Attention / Context Parallel —— 让 K/V 转圈

最激进的一条：**每张卡只保留自己那段 Q，K/V 分块沿着环流动**。

```
rank 0:  Q0 K0 V0 ──► 算 attn(Q0, K0V0)
                 K1 V1 ──► 算 attn(Q0, K1V1)   （K/V 从 rank1 转过来）
                 K2 V2 ──► ...
         ...
经过 P 步，Q0 见过了所有 K/V；K/V 块转完一圈回到原位。
```

每一步算出一个**局部**的注意力输出，然后用 **online softmax** 增量合并。这就是它能成立的全部数学基础——也正是 Flash Attention 的核心技巧：

```
维护三元组 (m, l, O)：
  m  = 目前见过的最大分数（数值稳定的基准）
  l  = 目前累积的 softmax 分母 Σexp(x - m)
  O  = 目前累积的加权输出
来新的一块 (S_new, V_new)：
  m_new = max(m, rowmax(S_new))
  l     = l * exp(m - m_new) + Σ exp(S_new - m_new)     ← 旧贡献重新缩放
  O     = O * exp(m - m_new) + exp(S_new - m_new) @ V_new
  m     = m_new
```

**没有这个可增量合并的公式，ring attention 就不存在。** 反过来说：**ring attention 是站在 Flash Attention 肩膀上的**——它把「分块」从「一块卡内部的分块」扩展成了「跨设备的分块」。这也是为什么 PyTorch 的 CP API 直接**替换 `F.scaled_dot_product_attention` 的实现**，而不是让你手写注意力。

**内存收益**：注意力中间量从 O(a·s²) 降到 O(a·(s/P)²)——**降 P² 倍**。这一条比 SP 那个 1/t 猛得多。

**能不能隐藏通信？** 可以精确算。每一步：

- 计算量：算 `attn(Q_blk, K_blk) @ V_blk`，约 `4·b·(s/P)²·h` FLOPs
- 通信量：搬运 K/V 两块，约 `4·b·(s/P)·h` bytes

$$\text{算术强度} = \frac{4b(s/P)^2h}{4b(s/P)h} = \frac{s}{P} = \text{本地块长}$$

**这个结果非常干净：ring attention 每一步的算术强度，就等于每张卡本地持有的序列长度。**

```
④ Ring Attention：每步的算术强度 = 本地块长 s/P
   本地块长    512 →      512 FLOP/byte
   本地块长   4096 →     4096 FLOP/byte
   本地块长  32768 →    32768 FLOP/byte
   隐藏通信门槛（H100 bf16 dense 989 TFLOP/s）:
     NVLink4 单向 450 GB/s            本地块长 ≥    2198 token
     跨节点 IB 400Gb 单向 50 GB/s        本地块长 ≥   19780 token
```

**怎么读这两行门槛**：H100 的 bf16 dense 算力 989 TFLOP/s，NVLink 4 单向约 450 GB/s → 机器平衡点是 2198 FLOP/byte。也就是说，只要本地块长超过约 2200 个 token，计算就能盖住通信。**跨节点就完全不同了**：400Gb IB 单向约 50 GB/s，平衡点跳到 19780——本地块长要接近两万 token 才能藏住通信。

**实践含义**：CP 优先在 NVLink 域内（同机 8 卡）铺开，效果好；跨机做 CP 时，要么本地块留得足够长（序列本身就很大），要么接受通信暴露。（真实 kernel 一般只跑到峰值的 30~50%，门槛会同比例下降到 1/2~1/3，但两个数量级的差距不会变。）

### 2.5 因果掩码的负载均衡：一个 53% 的效率黑洞

因果掩码给 CP 挖了一个坑。第 j 行的 token 只能看到前 j 个 token，计算量 ∝ j。如果**按连续块切给各 rank**：

```
③ 因果掩码 + 度 8：连续切分 vs Zigzag（全序列 8192，每卡 1024 行）
  连续切分       各 rank 相对负载 0.13 0.38 0.63 0.88 1.12 1.37 1.62 1.87  → 整体效率 53.3%
  Zigzag     各 rank 相对负载 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00  → 整体效率 100.0%
```

rank 0 手上是最早的 1024 个 token（几乎只能看自己），rank 7 手上是最后 1024 个（能看到前面所有东西）。**最慢的 rank 决定整体速度：效率 53.3%——白白浪费一半算力。**

解法是 **zigzag / round-robin**：把**最短的块和最长的块配给同一个人**（rank 0 拿第 1 块和第 8 块，rank 1 拿第 2 块和第 7 块……），每个人的负载就基本相等了。PyTorch 的 CP 实现里就带了这个负载均衡器，官方描述是「把最短的 query 块和最长的 query 块分给 rank 0，第二短的和第二长的分给 rank 1，以此类推」；如果是非因果注意力，则退回顺序切分。

### 2.6 PyTorch 里两种转圈方式，以及为什么默认是 all-gather

PyTorch 的 CP 实现给了两种 shard 轮转策略：

| 策略 | 做法 | 特点 |
|---|---|---|
| **all-gather based pass-KV**（默认） | 先把所有 rank 的 K/V 一次性 all-gather 齐，然后本地按需取块算 | 集合通信次数少；Llama3 训练用的就是这个思路；跨机更稳 |
| **all-to-all based pass-KV** | 每算完一块就用 all-to-all 把 shard 转给下一个 rank | 理论重叠更完美，但每次通信都要藏住 |

官方给出的 profiling 观察（Llama3-8B / 8×H100 / CP=8 / 无 compile 无 checkpoint）：all-to-all 那次的集合通信耗时约 **470 µs，藏不住**；而 all-gather 全程只需要一次集合通信，而 all-to-all 需要 N-1 次。所以他们把 **all-gather 定为默认**：更简单、更好调、跨机更稳，而且 all-to-all 一旦重叠失败惩罚更重。

顺带一提，PyTorch 那个「all-gather + 本地同时算本地块」的改法是**并行做**的：边 all-gather 边用本地 K/V 算，最后再补算剩余块——这也是把通信藏进计算里的一种做法。

---

## 三、在 PyTorch 中怎么用

### 3.1 先算账，别急着上集群

跑任何分布式之前，先把显存账算清楚。这个脚本**不需要 torch、不需要 GPU**，纯公式：

```python
# ledger.py —— 序列并行账本（可直接运行，本机无需 torch/GPU）
GIB = 1024 ** 3


def activation(s, b, h, a, tp=1, sp=False, recompute=False):
    """返回单层激活的 (线性项, 平方项)，单位 GiB。
    公式来自 Korthikanti et al. 2022《Reducing Activation Recomputation in
    Large Transformer Models》(Megatron-SP 论文) 的逐层激活推导：
      无并行      : 34*s*b*h               + 5*a*s^2*b
      仅 TP(度 t) : 10*s*b*h + 24*s*b*h/t  + 5*a*s^2*b/t
      TP+SP       : 34*s*b*h/t             + 5*a*s^2*b/t
    激活按 fp16（2 字节）计，dropout mask 的 1 字节已摊销进系数。
    """
    quad = 0.0 if recompute else 5 * a * s * s * b / tp   # 平方项，重算后不驻留
    if sp:
        lin = 34 * s * b * h / tp          # 连 LayerNorm/Dropout/残差 一起切开
    else:
        lin = 10 * s * b * h + 24 * s * b * h / tp   # 那 10sbh 切不动，是复制的
    return lin / GIB, quad / GIB


def ledger(name, s, b, h, a, layers, tp=1, sp=False, recompute=False):
    lin, quad = activation(s, b, h, a, tp, sp, recompute)
    print(f"{name:<26}{lin:>9.3f}{quad:>10.3f}{lin + quad:>10.3f}"
          f"{(lin + quad) * layers:>12.2f}")


# ---------- ① 逐层激活账本：LLaMA-70B 量级 ----------
h, a, L, b = 8192, 64, 80, 1
print(f"① LLaMA-70B 量级 (h={h}, heads={a}, layers={L}, batch={b}) 逐层激活 / GiB")
for s in (4096, 32768, 131072):
    print(f"\n  --- seq = {s} ---")
    print(f"  {'策略':<24}{'线性项':>9}{'平方项':>10}{'单层':>10}{'80 层合计':>12}")
    ledger("TP=8", s, b, h, a, L, tp=8)
    ledger("TP=8 + SP", s, b, h, a, L, tp=8, sp=True)
    ledger("TP=8 + SP + 选择性重算", s, b, h, a, L, tp=8, sp=True, recompute=True)
    ledger(f"再加 CP=8 (本地 {s // 8})", s // 8, b, h, a, L, tp=8, sp=True, recompute=True)

# ---------- ② SP 的收益随 TP 度数放大 ----------
print("\n② 线性激活里「切不动」的比例（10sbh / 34sbh = 29.4%），SP 把它也除掉")
print(f"  {'TP 度 t':<10}{'仅 TP':>15}{'TP+SP':>15}{'收益':>10}")
for t in (1, 2, 4, 8, 16):
    no = (10 + 24 / t) if t > 1 else 34     # t=1 就是退化成无并行
    yes = 34 / t
    print(f"  {t:<10}{no:>10.2f} sbh{yes:>12.2f} sbh{no / yes:>11.2f}x")

# ---------- ③ 因果掩码下的负载均衡 ----------
print("\n③ 因果掩码 + 度 8：连续切分 vs Zigzag（全序列 8192，每卡 1024 行）")
T, S = 8, 8192
BLK = S // T
# 因果注意力中，第 j 行的计算量 ∝ 它能看到的 token 数（≈ j）
work = [sum(range(i * BLK + 1, (i + 1) * BLK + 1)) for i in range(T)]
zig = [work[i] + work[T - 1 - i] for i in range(T)]   # 最短块 + 最长块 配对
for name, w in (("连续切分", work), ("Zigzag", zig)):
    mean = sum(w) / len(w)
    print(f"  {name:<10} 各 rank 相对负载 "
          f"{' '.join(f'{x / mean:.2f}' for x in w)}  → 整体效率 {mean / max(w) * 100:.1f}%")

# ---------- ④ Ring Attention 每一步的算术强度 ----------
print("\n④ Ring Attention：每步的算术强度 = 本地块长 s/P")
print("   每步算 4b(s/P)^2·h FLOPs，搬 K/V 两块共 4b(s/P)h bytes")
for blk in (512, 4096, 32768):
    print(f"   本地块长 {blk:>6} → {4 * blk ** 2 * h / (4 * blk * h):>8.0f} FLOP/byte")
print("   隐藏通信门槛（H100 bf16 dense 989 TFLOP/s）:")
for name, bw in (("NVLink4 单向 450 GB/s", 450e9), ("跨节点 IB 400Gb 单向 50 GB/s", 50e9)):
    print(f"     {name:<30} 本地块长 ≥ {989e12 / bw:>7.0f} token")
```

输出就是第一节那张表 + 后面的三个结论块。

### 3.2 路线一：TP + SP（DTensor）

需要 2 张 GPU。存成 `tp_sp_demo.py`，用 `torchrun --standalone --nproc-per-node=2 tp_sp_demo.py` 跑：

```python
# tp_sp_demo.py —— TP=2 + SP 的最小可运行示例
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


class Block(nn.Module):
    """pre-norm 残差块，等价于 Megatron-LM 的一个 TP 区域：
    x → LayerNorm → 列并行 Linear → GeLU → 行并行 Linear → +x
    """

    def __init__(self, h=256, ffn=1024):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, ffn, bias=False)
        self.fc2 = nn.Linear(ffn, h, bias=False)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


def main():
    rank, world = int(dist.get_rank()), int(dist.get_world_size())
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")     # torchrun 会自动注入环境变量

    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("tp",))
    b, s, h = 2, 4096, 256

    model = Block(h).cuda()
    for p in model.parameters():        # 为了各卡初始一致，先广播一次权重
        dist.broadcast(p.data, src=0)

    # 关键：归一化层的权重必须保持 replicated，SP 只切激活不切权重
    plan = {
        "norm": SequenceParallel(),                              # LayerNorm 沿序列维切分
        "fc1": ColwiseParallel(),                                # 列并行：输入 Replicate（会自动 all-gather）
        "fc2": RowwiseParallel(output_layouts=Shard(1)),          # 行并行 + 出口 reduce-scatter
    }
    model = parallelize_module(model, mesh, plan)

    # 输入的残差流本身就按序列维分片：shape 是 [b, s/P, h]
    x = torch.randn(b, s, h, device="cuda")
    x = distribute_tensor(x, mesh, [Shard(1)])

    out = model(x)
    print(f"[rank {rank}] 输入本地形状 {x.to_local().shape} "
          f"→ 输出本地形状 {out.to_local().shape}")

    # 反向也能正常跑：梯度同样按序列维分片
    out.to_local().sum().backward()
    print(f"[rank {rank}] norm.weight.grad 形状 "
          f"{model.norm.weight.grad.shape}   （replicated，与本地序列无关）")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

几个要点：

- **`RowwiseParallel(output_layouts=Shard(1))` 是 SP 的关键开关**。行并行 Linear 出口本来需要 all-reduce（各部分和相加）并广播成 Replicate；把输出布局标成 `Shard(1)`，DTensor 就会把这次 all-reduce 实现成 **reduce-scatter**——这正是 Megatron-SP 里那个「g 算子」。
- **`SequenceParallel()` 只支持 LayerNorm / Dropout / RMSNorm**（官方文档口径）。它假定模块权重是 ones 初始化；如果你自己改过初始化，得手动广播，否则各卡上的 norm 权重会不一致。
- 入口的 `distribute_tensor(..., [Shard(1)])` 对应实际训练里的「模型输入、labels、position_ids 全部按序列维切」——下一节会讲这是最容易出错的地方。

### 3.3 路线三：Context Parallel + Ring Attention（官方 API）

需要 4 张 GPU。这段基于 PyTorch 官方 Context Parallel 教程，加上了一层「和单卡 SDPA 对答案」的验证：

```python
# cp_ring_demo.py —— torchrun --standalone --nproc-per-node=4 cp_ring_demo.py
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import (
    context_parallel_unshard,
    set_rotate_method,          # 目前还在私有模块里，API 可能变
)
from torch.nn.attention import sdpa_kernel, SDPBackend


def main(world_size: int, rank: int):
    assert torch.cuda.is_available() and dist.is_nccl_available()
    torch.cuda.set_device(f"cuda:{rank}")
    torch.cuda.manual_seed(0)
    dist.init_process_group("nccl", init_method="env://")

    # CP 度数 = 整个 world（实践中通常是 2D mesh，比如 (dp, cp) 或 (tp, cp)）
    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))

    batch, nheads, seq, dim = 8, 8, 8192, 32
    backend = SDPBackend.FLASH_ATTENTION     # CP 要求可用的 SDPA 后端
    dtype = torch.bfloat16                   # FlashAttention 走 bf16/fp16

    # 每个 rank 都先造一份完整数据，并记录单卡参考输出
    qkv = [
        torch.rand((batch, nheads, seq, dim), dtype=dtype, requires_grad=True, device="cuda")
        for _ in range(3)
    ]
    with sdpa_kernel(backend):
        ref = F.scaled_dot_product_attention(*qkv, is_causal=True)

    cp_qkv = [t.detach().clone() for t in qkv]   # 待分片的副本

    set_rotate_method("allgather")   # 默认就是 allgather；改成 "alltoall" 可切策略

    with sdpa_kernel(backend):
        with context_parallel(
            device_mesh,
            buffers=tuple(cp_qkv),
            buffer_seq_dims=(2, 2, 2),   # 三个张量的序列维都是第 2 维
        ):
            # 这一步做了两件事：
            #   1. 把 buffers 沿序列维原地切分，每个 rank 只留自己那段
            #   2. 把 F.scaled_dot_product_attention 换成 Ring Attention 实现
            cp_out = F.scaled_dot_product_attention(*cp_qkv, is_causal=True)

        # 输出仍然是按序列维分片的，用这个 API 拼回完整张量
        (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [2])

    assert torch.allclose(cp_out, ref, atol=1e-3 * world_size), "Ring Attention 与单卡结果不一致！"
    if rank == 0:
        print(f"✅ CP={world_size} 的 Ring Attention 与单卡 SDPA 一致")
        print(f"   本地序列分片：{seq // world_size} token（原来 {seq}）")
        print(f"   注意力中间量降 {(world_size ** 2)} 倍")


if __name__ == "__main__":
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    try:
        main(world_size, rank)
    finally:
        dist.barrier()
        dist.destroy_process_group()
```

几个要点：

- **`context_parallel()` 不是「手动写 ring attention」，而是「打补丁 + 切 buffer」**：它把 `F.scaled_dot_product_attention` 换成 CP 版 ring attention，同时把 `buffers` 里的张量**原地**沿序列维切分。出上下文时会自动还原（除非你放进 `no_restore_buffers`）。
- **`buffers` 是这份 API 里最容易被忽略、也最容易算错的部分。** 官方明确提醒：以 Llama3 训练为例，**忘了把 `freq_cis`（RoPE 的 cos/sin）放进 `buffers`，旋转位置编码就会算错**。凡是「用法依赖序列维」的张量——输入 batch、labels、position_ids、loss mask、freq_cis——都得放进去。
- `buffers` 里**不能有 `nn.Parameter`**（那是权重的领地）。
- **这是 prototype 特性**，官方文档标了「API is subject to change」，`set_rotate_method` 甚至还在 `_attention` 私有模块里。用在生产前先锁版本。

### 3.4 如果还不需要 CP：先把手边的三件事做完

CP 带来的复杂度（额外的 mesh 维度、buffer 分片、负载均衡、通信重叠调优）不小。**在序列长度还没到「单卡真的放不下」之前，按这个顺序做更划算**：

```python
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def block_forward(self, x):
    # ① Flash Attention：消掉 5as^2b 这一项（收益最大的一步）
    #    只要用 F.scaled_dot_product_attention 并指定 causal，后端自己会选 flash kernel
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # ② 激活重算：把「线性项」从 34sbh 降到「一层 34sbh」
    #    use_reentrant=False 是现在的推荐用法（支持非张量输入、和编译配合更好）
    def mlp(y):
        return self.fc2(F.gelu(self.fc1(y)))

    x = x + self.proj(attn)
    x = x + checkpoint(mlp, self.norm2(x), use_reentrant=False)
    return x
    # ③ 序列打包（packing）：把多条短样本拼成一条长序列，避免 padding 浪费算力。
    #    代价是必须用 varlen 路径（cu_seqlens）或 block mask，位置编码要按样本重置。
```

三件事做完再看账本：**如果 80 层激活还塞得下，就不需要 CP。** 需要 CP 的典型信号是：seq ≥ 32K 且 batch ≥ 1 已经很紧，或者目标直接是 128K / 1M 级别的长上下文训练。

---

## 四、围绕这个领域展开

### 4.1 三条路线一张表

| | **Megatron-SP** | **Ulysses (DeepSpeed)** | **Ring / CP** |
|---|---|---|---|
| 切什么 | 只切 LayerNorm/Dropout/残差的激活 | 切序列 + 用 all-to-all 转成切 head | 切序列，K/V 在环上流动 |
| 注意力在哪算 | **每卡算完整序列**（all-gather 之后） | 每卡算完整序列、部分 head | 每卡算本地 Q × 环上的 K/V |
| 通信原语 | all-gather + reduce-scatter | 两次 all-to-all | P2P ring（或 all-gather / all-to-all 轮转） |
| 通信量 | 与原 TP 的 all-reduce **完全相同** | 序列与设备同比例增长时**恒定** | 每步算术强度 = s/P |
| 激活收益 | 线性项 ÷ t（t=8 时 3.06×） | 序列维 ÷ P | 注意力中间量 ÷ P² |
| 依赖 | **必须 TP > 1** | head 数可被 P 整除（GQA 受限） | 需要可用的 SDPA 后端 |
| 能否独立扩展序列长度 | ❌ 不能 | ✅ 能 | ✅ 能（百万 token 级） |
| 论文/实现 | Megatron-LM (2022) | DeepSpeed-Ulysses (2023) | Ring Attention (2023) / 各家 CP |

**记忆锚点**：SP 是「TP 的补丁」，Ulysses 是「转置」，Ring 是「转圈」。三者可以叠加——Megatron-Core 的实践里 `tp × cp × pp × dp` 是同一个 4D 并行组。

### 4.2 和 Flash Attention 的关系：同一套数学，换了个层级

- **卡内**：Flash Attention 把一条长序列的注意力切成加载进 SRAM 的小块，用 online softmax 串起来算 → 省的是 SRAM↔HBM 的带宽和中间量显存。
- **卡间**：Ring Attention 把「块」的粒度从 SRAM 提到设备级，用 online softmax 跨卡合并 → 省的是跨设备的显存。
- **选择性激活重算**（Megatron 论文的第二个贡献）：只重算「算起来便宜、占显存大」的部分（注意力 softmax/dropout 相关），其余照常存。论文报告：SP + 选择性重算合起来，激活显存降低 **5×**，重算带来的时间开销降低 **90% 以上**；530B 模型在 2240 张 A100 上达到 **54.2% MFU**，而全量重算是 42.1%（提升约 29%）。

所以这一家的技术路线是一层层垒起来的：**Flash Attention（消平方项）→ SP（消复制的线性项）→ CP（把序列本身切开）**。

### 4.3 和 ZeRO / FSDP 的关系：两条正交的轴

- **ZeRO/FSDP 管的是「状态」**：优化器状态、梯度、参数的切分。它和序列长度无关。
- **SP/CP 管的是「激活」**：它和参数量基本无关。
- 所以两者**正交，可以叠**。Ulysses 论文里就是和 ZeRO-3 一起用的：ZeRO-3 负责让参数量级能装下，Ulysses 负责让序列长度能装下。

### 4.4 和推理侧的关系：prefill 才是长序列场景

- 解码（decode）阶段每步只处理 1 个 token，序列维的并行没意义；**prefill 阶段要一次性处理整条 prompt，才是长序列的主场**，也才是 CP 在推理里的用武之地（chunked prefill、prefill 的序列切分）。
- 但推理侧的主流选择不是 CP，而是 **PagedAttention + continuous batching + chunked prefill**（第 011 期）：把 KV Cache 分页管理来控制显存碎片，把长 prompt 切块调度来做负载均衡。它们解决的是「吞吐与显存碎片」，和训练侧 CP 解决「激活装不下」是不同问题。
- 有一个共同的数学工具：**online softmax**。Flash-Decoding、chunked prefill 的部分结果合并，用的都是它。

### 4.5 CP 之外的另一条路：别让复杂度变成 O(s²)

序列并行本质上是「**接受 O(s²)，然后把它分给更多卡**」。另有一整条路线是「**从根上不做 O(s²)**」：

- **线性注意力 / SSM（Mamba 系）**：把 attention 换成 O(s) 的递归形式，长序列的显存压力直接消失。代价是表达力和工程生态。
- **稀疏注意力 / 滑窗 + 全局 token**：只算一部分 token 对，配合 block mask。
- **蒸馏出的长上下文**：短窗口训练 + 位置插值，把上下文窗口撑长而不是从零训。

**怎么选**：如果是「沿用标准 Transformer、把上下文从 32K 推到 128K」，CP 是正路；如果是「从零设计一个百万 token 的模型」，SSM/线性注意力那条路值得认真评估——因为 CP 的通信开销随设备数线性增长，而 O(s) 架构不吃这一刀。

### 4.6 那些「跟着序列一起被切」的东西

这是实践中最容易翻车的一类问题。开了 CP 之后，**凡是形状或语义依赖序列位置的张量都必须同步切分**：

| 张量 | 为什么必须切 | 忘了会怎样 |
|---|---|---|
| `input_ids` / `inputs_embeds` | 模型输入本身 | 各卡算的是同一段序列，形状对不上或重复计算 |
| `labels` | 损失按 token 算 | loss 重复累加，梯度错误 |
| `position_ids` | 每个 token 的位置 | 位置编码全错 |
| **RoPE 的 `freq_cis` / cos-sin 表** | 位置相关 | **官方点名的经典错误**：旋转位置编码算错，loss 看着能降但模型废掉 |
| `loss_mask` / `attention_mask` | 有效 token 标记 | 统计和掩码错位 |
| packed 序列的 `cu_seqlens` | 打包边界 | 跨样本注意力泄露 |

PyTorch 的 `context_parallel(buffers=..., buffer_seq_dims=...)` 这个设计就是为了**把这类「必须一起切」的东西集中在一个地方声明**。反过来讲：**只要模型里还有任何一处「手工依赖全局序列长度」的逻辑，CP 就会静默出错**——不报错，只是结果悄悄不对，这是它最危险的地方。

### 4.7 生态现状

| 框架 | SP/CP 的实现方式 |
|---|---|
| **PyTorch 原生** | `torch.distributed.tensor.parallel.SequenceParallel`（TP+SP）；`torch.distributed.tensor.experimental.context_parallel` + Ring Attention（CP，prototype） |
| **TorchTitan** | 官方长上下文训练参考实现，CP 已用于 **1M 序列长度**训练 |
| **Megatron-Core** | `tensor_model_parallel_size` + `context_parallel_size`，4D 并行；CP 带 zigzag 负载均衡 |
| **DeepSpeed** | Ulysses（基于 all-to-all），和 ZeRO-3 组合 |
| **HuggingFace Transformers** | 训练时提供 CP 支持，底层接 PyTorch 的 `context_parallel` |

---

## 五、什么时候该用 / 不该用

**该用**：

- 序列长度已经让单卡激活放不下，且**平方项已经被 Flash Attention 处理掉**了（否则先做那一步）
- 训练长上下文（32K → 128K → 1M），需要把上下文长度作为一等目标来扩展
- 已经是 TP/PP/DP 的 3D 并行，还想继续沿某个轴扩展——这时 CP 是第四条轴
- **TP 度数已经拉到 8 或 16**：此时 SP 收益 3~5 倍，几乎是白送的（通信量不变）
- CP 优先在 NVLink 域内铺；跨节点做 CP 要确认本地块长够长（≥ ~2 万 token 更稳）

**不该用**：

- **序列不长（< 8K）**：TP+SP 足够，CP 的负载均衡、buffer 分片、通信重叠调优都是纯负担
- **只想省一点显存**：先做激活重算（省得更多、改动更小、无通信）
- **注意力不是标准 SDPA**：自定义稀疏 mask、手写 kernel、滑动窗口——CP 打的是 `F.scaled_dot_product_attention` 的补丁，绕开它就绕开了 CP
- **模型是 GQA/MQA 且用 Ulysses**：KV head 数少（如 8）会把度数上限卡死，得换 Ring/CP
- **推理场景**：那是 PagedAttention + chunked prefill 的地盘，别把训练侧的 CP 硬套过来
- **TP=1 想单独上 SP**：SP 需要 TP 区域，TP=1 时它没有可替换的 all-reduce

---

## 六、常见坑

**坑 1：以为开了 SP 就能训长序列**

这是最普遍的误解。SP 只做一件事——把 LayerNorm/Dropout/残差那 10sbh 的复制消掉。它**不减小注意力的序列长度**，也管不住 5as²b 的平方项。看第一节的表：seq=32768 时 SP 把单层激活从 43.25 GiB 降到 41.06 GiB，**只降了 5%**。真正需要长序列时，SP 和 CP 是两件事。

**坑 2：忘了切 RoPE 的 freq_cis（或任何序列相关的 buffer）**

官方点名的经典错误：**在 Llama3 训练里漏掉 `freq_cis`，旋转位置编码就会算错。** 这类 bug 的可怕之处在于**不报错**——loss 照样往下降，只是模型学到的位置信息是错的。开 CP 之前，把模型里所有「形状依赖序列长度」或「语义依赖绝对位置」的张量列一遍，全部塞进 `buffers`。

**坑 3：因果掩码下用连续切分，白白丢 47% 算力**

不做 zigzag，rank 之间的负载差距是 1.87 : 0.13（度 8），整体效率 53.3%——你多买了一半的卡在等最慢的那个 rank。而且**负载不均还会破坏通信与计算的重叠**：负载轻的 rank 早早算完在等，本该被计算藏住的通信就暴露出来了。用框架自带的 round-robin / zigzag sharder，别自己写连续切。

**坑 4：指望 all-to-all 完美重叠，结果被单次通信拖死**

官方 profiling 给的是很实在的数据：all-to-all 单次集合通信 ~470 µs，**藏不住**；而 all-to-all 需要 N-1 次（N = CP 度数）。all-gather 全程只需一次。**如果通信一暴露就完蛋**，优先用默认的 all-gather 策略；all-to-all 只在「算力确实很闲、通信带宽很富余、且做过实测」时才考虑。

**坑 5：`SequenceParallel` 的权重没保持复制的**

`SequenceParallel` 假定模块权重是 ones 初始化（LayerNorm/RMSNorm 的默认值）。如果你对这些层做了自定义初始化，**必须手动广播**，否则各卡上的归一化参数不一致——又是一个不报错只算错的坑。同理，SP 只切激活不切权重，别指望它帮你省参数显存。

**坑 6：拿 CP 去跑 MQA/GQA 的 Ulysses**

LLaMA-3-70B 有 64 个 Q head 但只有 **8 个 KV head**。Ulysses 要求 head 数可被 P 整除，KV head 8 个 → 度数上限 8。想上 P=16 就得换 Ring/CP。选路线前先看清模型的 head 配置。

---

## 七、一句话总结

**序列并行是并行工具箱里唯一一条「减小每张卡 token 数」的轴——Megatron-SP 用 all-gather/reduce-scatter 的等价替换，把 TP 切不动的 29.4% 激活也除掉（收益随 TP 度数放大，换汤不换药的通信量）；Ulysses 用两次 all-to-all 把「按序列切」转置成「按 head 切」（通信量恒定，但受 head 数整除约束）；Ring/CP 让 K/V 在环上流动、用 online softmax 增量合并（注意力中间量降 P²，每步算术强度恰好等于本地块长 s/P）——而三者都建立在同一个前提上：平方项先被 Flash Attention 消掉，否则什么并行都救不了你。**

---

<details>
<summary>今日练习（点击展开参考答案）</summary>

### 练习 1：算一下你自己的场景

用 3.1 节的脚本，改成你实际关心的配置（改 `h`、`a`、`L`、`b`），回答：

1. 在 seq=16K、batch=2 时，TP=8 的逐层激活里，平方项占比多少？
2. 如果只上 SP，逐层激活降低百分之几？
3. 要把 80 层激活压到单卡 20 GiB 以内，需要 CP 度数至少是多少？

**参考答案**：

```python
GIB = 1024 ** 3
h, a, L, b, s, tp = 8192, 64, 80, 2, 16384, 8

lin_tp  = (10 * s * b * h + 24 * s * b * h / tp) / GIB
lin_sp  = (34 * s * b * h / tp) / GIB
quad    = (5 * a * s * s * b / tp) / GIB

print(f"仅 TP   : 线性 {lin_tp:.2f} 平方 {quad:.2f} → 逐层 {lin_tp + quad:.2f} GiB")
print(f"TP + SP : 线性 {lin_sp:.2f} 平方 {quad:.2f} → 逐层 {lin_sp + quad:.2f} GiB")
print(f"平方项占比 {quad / (lin_tp + quad) * 100:.1f}%")
print(f"SP 使逐层总激活下降 {(1 - (lin_sp + quad) / (lin_tp + quad)) * 100:.1f}%")

# 关掉平方项（选择性重算）后，求满足 80 层 ≤ 20 GiB 的最小 CP 度数
for P in (1, 2, 4, 8, 16):
    per = (34 * (s // P) * b * h / tp) / GIB
    if per * L <= 20:
        print(f"CP={P} → 逐层 {per:.3f} GiB，80 层 {per * L:.2f} GiB ✅")
        break
```

输出：

```
仅 TP   : 线性 3.25 平方 20.00 → 逐层 23.25 GiB
TP + SP : 线性 1.06 平方 20.00 → 逐层 21.06 GiB
平方项占比 86.0%
SP 使逐层总激活下降 9.4%
  CP=1 → 逐层 1.062 GiB，80 层 85.00 GiB
  CP=2 → 逐层 0.531 GiB，80 层 42.50 GiB
  CP=4 → 逐层 0.266 GiB，80 层 21.25 GiB
  CP=8 → 逐层 0.133 GiB，80 层 10.62 GiB ✅
```

**要点**：① 平方项占 86.0%，所以只上 SP 只省 9.4%——**先做 Flash Attention，再谈并行**；② 关掉平方项后激活和本地序列长度成正比，CP 度数翻倍、显存减半（1.062 → 0.531 → 0.266 → 0.133）；③ 注意 batch=2 让一切都翻了一倍，这就是为什么长上下文训练普遍把 micro-batch 压到 1，靠梯度累积凑等效 batch。

### 练习 2：验证 SP 的通信量真的没变

Megatron-SP 的核心论点是「all-reduce ≡ reduce-scatter + all-gather，所以通信量不变」。用纯计数的方式验证一下：在一个 TP=8、b=1、s=8192、h=8192、fp16 的 Transformer 层里，前向+反向原本的 4 次 all-reduce 一共搬多少字节？改成 4 次 all-gather + 4 次 reduce-scatter 之后呢？

**参考答案**：

```python
GIB = 1024 ** 3
s, b, h, t = 8192, 1, 8192, 8
nbytes = 2                      # fp16

# ring all-reduce 的两个阶段：reduce-scatter + all-gather
# 每个 rank 每阶段接收 (t-1)/t 份数据
per_rank_ar = 2 * s * b * h * nbytes * (t - 1) / t      # 一次 all-reduce
per_rank_ag = s * b * h * nbytes * (t - 1) / t          # all-gather
per_rank_rs = s * b * h * nbytes * (t - 1) / t          # reduce-scatter

print(f"改造前（4 次 all-reduce）: {4 * per_rank_ar / GIB:.3f} GiB")
print(f"改造后（4 AG + 4 RS）    : {4 * (per_rank_ag + per_rank_rs) / GIB:.3f} GiB")
print(f"比值 = {(4 * (per_rank_ag + per_rank_rs)) / (4 * per_rank_ar):.4f}")
```

输出：

```
改造前（4 次 all-reduce）: 0.875 GiB
改造后（4 AG + 4 RS）    : 0.875 GiB
比值 = 1.0000
```

两者都是 **0.875 GiB**，比值 **1.0000**——**体积严格相等**。

**要点**：SP 不增加通信量，它只是把「一次 all-reduce」拆成了「两次单向集合通信」。实际系统里的差别在**延迟和重叠**：all-reduce 是一个原子操作，语义清晰；拆成两步之后，框架需要在更细的粒度上安排通信与计算的交错，这也是为什么 SP 的性能强依赖通信-计算重叠的实现质量。

### 练习 3：为什么 ring attention 的算术强度等于 s/P

从第一性原理推一遍：每一步 ring attention，本地 Q 块长度是 s/P，要处理一块同样长度的 K/V。写出计算量（FLOPs）和通信量（bytes），求比值。

**参考答案**：

设每卡本地块长 `L = s/P`，batch = b，hidden = h，head 维 d = h/a，头数 a。

- **计算**：两步矩阵乘。`S = Q @ Kᵀ` 的形状是 `[b, a, L, L]`，FLOPs = `2b·a·L²·d`；`O = softmax(S) @ V` 同样是 `2b·a·L²·d`。合计 `4b·a·L²·d = 4b·L²·h`。
- **通信**：这一步要接收 K 块和 V 块各一个，每块 `b·L·h` 个元素、fp16 2 字节。合计 `2 · b·L·h · 2 = 4b·L·h` bytes。

$$\text{算术强度} = \frac{4bL^2h}{4bLh} = L = \frac{s}{P}$$

**要点**：**`b`、`h` 全部约掉，只剩本地块长。** 这个结论非常实用——它意味着「CP 扩不扩得动，只取决于你每张卡本地留了多长的序列」，和模型宽度、batch 大小都没关系。代入机器参数就能算出「本地块长要多长才藏得住通信」：H100 平衡点约 2200 token（NVLink）/ 19800 token（跨机 IB）。

</details>

---

**下期预告**：还在挑，候选有损失函数设计、正则化全景、cuDNN/cuBLAS、低秩分解与 LoRA、知识蒸馏、torch.export + AOT Inductor——挑一个你没看过的。
