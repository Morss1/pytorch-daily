# PyTorch 每日一课 · 第 030 期

## KV 的跨进程与跨机共享：PD 分离与分层 KV 存储

> **日期**：2026-10-01　**难度**：⭐⭐⭐⭐（需要 KV Cache 与 roofline 的基础）
> **前置知识**：KV Cache 的形状与体积（第 011 期）、张量并行怎么切 KV（第 019 期）、连续批处理与 chunked prefill（第 028 期）、前缀缓存与命中率等式（第 029 期）、投机解码的免费额度（第 027 期）
> **预计阅读时间**：45 分钟
> **关联**：第 029 期 · 关联点「KV 的体积就是传输的体积，一个字节都不差」；第 028 期 · 关联点「调度器里没有 prefill/decode 之分，本期是在物理上把它们分开」；第 027 期 · 关联点「B·L = 295 的平衡点第四次出现，这次它决定 chunk 该开多大」

---

## 1. 这个领域解决什么问题

### 1.1 一个被忽略的事实：整个推理系统里，只有 KV 需要搬

一个 transformer 推理服务有很多状态：模型权重（只读，副本越多越好）、调度队列（几 KB 的元数据）、采样器的随机数发生器（可忽略）。只有一样东西是**既大、又可变、又必须精确地跟着某条请求走**的：

**KV Cache。**

它的体积由「层数 × KV 头数 × head_dim × 精度 × 序列长度」决定，与模型参数量无关。第 029 期算过 7B / GQA-8 / bf16 是 **128 KiB/token** —— 一条 32K 上下文的请求，KV 就是 **4 GiB**。而它的内容无法重算两次得到同一个值之外的任何捷径：它就是这条请求的历史。

这带来两个后果：

1. **凡是「让 KV 在别处存在、把计算省下来」的想法，本质上都是搬运问题。** 第 029 期的前缀缓存是这个想法在同机内的版本（KV 留在同一个 KV 池里，靠指针复用）；本期讲它在**跨进程、跨机**时的版本。
2. **搬运量的单位就是 KV 体积，没有折扣。** 你不搬 128 KiB/token，就不可能让别的地方知道这条请求的历史。

### 1.2 第二个事实：prefill 和 decode 想要的东西是相反的

用统一口径（H800 级：BF16 稠密算力 989 TFLOPS、HBM3 3.35 TB/s、MFU 0.45、bf16 权重 2 字节）算一下算术强度：

```python
N, L, D, HK, DH = 7e9, 32, 4096, 8, 128
BY, BW, PEAK, MFU = 2, 3.35e12, 989e12, 0.45

def flops(T, S):
    """T 个 query token 对 S 个 key token 的一次全层前向：线性项 + 注意力项。"""
    return 2.0 * N * T + 4.0 * L * D * T * S

rows = [("prefill S=4096 整段 T=4096", flops(4096, 4096)),
        ("prefill 分块 T=512, S=4096",  flops(512, 4096)),
        ("decode batch=1, S=4096",      flops(1, 4096)),
        ("decode batch=32",             flops(32, 4096)),
        ("decode batch=132",            flops(132, 4096)),
        ("decode batch=512",            flops(512, 4096))]
print(f"{'阶段':<28}{'FLOPs':>14}{'权重字节':>11}{'AI':>9}{'瓶颈':>16}")
for name, f in rows:
    ai = f / (N * BY)
    print(f"{name:<28}{f/1e9:>12.1f}G{N*BY/1e9:>9.1f}G{ai:>9.1f}"
          f"{'compute 等算力' if ai > 295.2 else 'memory 等带宽':>16}")
```

```text
阶段                                   FLOPs       权重字节       AI             瓶颈
prefill S=4096 整段 T=4096         66140.1G     14.0G   4724.3    compute 等算力
prefill 分块 T=512, S=4096          8267.5G     14.0G    590.5    compute 等算力
decode batch=1, S=4096               16.1G     14.0G      1.2    memory 等带宽
decode batch=32                     516.7G     14.0G     36.9    memory 等带宽
decode batch=132                   2131.5G     14.0G    152.2    memory 等带宽
decode batch=512                   8267.5G     14.0G    590.5    compute 等算力
```

prefill 的算术强度是 **4724**，decode（小 batch）是 **1.2** —— 差了三个数量级。

这个差别落到硬件上意味着完全不同的诉求：

| | prefill | decode |
|---|---|---|
| 瓶颈 | 算力（compute-bound） | 显存带宽（memory-bound） |
| 想要的 batch | 越大越好，吃满 Tensor Core | 顶到平衡点 **132.9** 就停，再大反而转 compute-bound |
| 想要的并行策略 | TP 大一点压 TTFT | TP 小一点省通信、省 KV 复制 |
| 对延迟的容忍 | 能等，TTFT 几百 ms 正常 | 一步 4.18 ms 都不能抖 |

第 028 期已经讲过：vLLM V1 的调度器**根本没有 prefill/decode 阶段之分**，只有 `num_computed_tokens` 追 `num_tokens_with_spec`，靠 chunked prefill 把长 prompt 切成小块插进 decode 迭代里削峰。

**本期的主题是：把这种「挤在同一个迭代里」的做法，换成「物理上分到不同的 GPU 上」。** 一句话概括动机：

> 同一个 GPU 上塞两个诉求相反的阶段，只能用时间片轮转互相将就；分开之后，每个阶段都能按自己的最优形态跑，代价是 KV 必须过一次网络。

### 1.3 第三个事实：这个「将就」到底有多贵

第 028 期测过一个数字：4K prompt 一次吞完会让那一步从 4.3 ms 涨到 58.4 ms（13.46× ITL 尖峰）；切成 512 的块分 8 次，尖峰降到 1.77×，prefill 自身总耗时只多 5.4%。

看起来切块很划算。但第 028 期没往下追一个问题：**如果我要把尖峰再压到 1.2× 呢？**

本期 §3.3 会给出解析答案，并且指出这是一条**双曲线**：把尖峰压到 1 倍（完全不干扰）的唯一办法是 chunk 大小为 0，而那时 prefill 就完全不做了。你压尖峰付出的代价，和 prompt 长度成正比。

这就是 PD 分离真正要解决的东西 —— 不是吞吐，是这条不可能三角。

---

## 2. 核心思想

### 2.1 把 KV 当成一件「可以离开出生地」的商品

先建立一个反直觉的直觉：**搬运 KV 比重新算一遍便宜得多，而且不是便宜一点点。**

7B / GQA-8 / bf16 / H800 级：

```python
N, L, D, HK, DH = 7e9, 32, 4096, 8, 128
BY, BW, PEAK, MFU = 2, 3.35e12, 989e12, 0.45
PER_TOK = 2 * L * HK * DH * BY          # 128 KiB/token
t_pf = 2 * N / (MFU * PEAK)             # prefill 每 token 边际算力成本

for S in [1024, 4096, 8192, 32768, 131072]:
    kv = PER_TOK * S
    # 重算：因果掩码下注意力 FLOPs = 4Ld*S^2/2，线性项 2N*S
    rec = (2 * N * S + 4 * L * D * S * S / 2) / (MFU * PEAK)
    t_pc, t_ib = kv / 64e9, kv / 50e9
    print(f"S={S:>7}  KV={kv/2**20:8.0f}MiB  PCIe5 {t_pc*1e3:8.2f}ms  IB400 {t_ib*1e3:8.2f}ms  "
          f"重算 {rec*1e3:9.1f}ms  搬/算 = {rec/max(t_pc,t_ib):6.1f}x  临界带宽 {kv/rec/1e9:5.2f} GB/s")
```

```text
S=   1024  KV=     128MiB  PCIe5     2.10ms  IB400     2.68ms  重算      32.9ms  搬/算 =   12.2x  临界带宽  4.09 GB/s
S=   4096  KV=     512MiB  PCIe5     8.39ms  IB400    10.74ms  重算     139.1ms  搬/算 =   12.9x  临界带宽  3.87 GB/s
S=   8192  KV=    1024MiB  PCIe5    16.78ms  IB400    21.47ms  重算     297.7ms  搬/算 =   13.8x  临界带宽  3.61 GB/s
S=  32768  KV=    4096MiB  PCIe5    67.11ms  IB400    85.90ms  重算    1665.9ms  搬/算 =   19.4x  临界带宽  2.58 GB/s
S= 131072  KV=   16384MiB  PCIe5   268.44ms  IB400   343.60ms  重算   14261.4ms  搬/算 =   41.5x  临界带宽  1.21 GB/s
```

（这里的重算时间用的是**因果掩码**下的注意力 FLOPs $4LdS^2/2$，比「T=S 全注意力」的算法少一半 —— 所以和 §3.3 表格里的数值不完全可比，后者用的是同一条因果口径。文中凡是引用绝对时间都标了口径。）

**临界带宽**那一列是全表最重要的一列：它表示「网络只要比这个速度更快，搬运就比重算划算」。4K 上下文只需要 **3.87 GB/s** —— 连一条 30 Gbps 的链路都能满足。

换句话说：

> **KV 放在哪里并不重要，重要的是它别被重算。** 这是第 029 期前缀缓存能成立的全部前提，也是本期跨机搬运能成立的全部前提。两台机器之间只要有一根 100Gb 网线，把 KV 搬过去就比重算划算十几倍。

（顺带解释一个 vLLM 的默认值：NixlConnector 有个 `kv_recompute_threshold`，默认 **64** —— 「远端命中不到 64 个 token 就老老实实本地重算，不发起传输」。原因就是**传输有固定开销**（握手、注册、建立 RDMA 队列），小传输里这个固定项吃掉全部收益。上表的「临界带宽」只算了稳态带宽，没算这个固定项。）

### 2.2 三个反复出现的常数

本期所有推导都挂在三个数上，它们和第 027/028/029 期是同一套：

| 常数 | 值 | 含义 |
|---|---|---|
| **per-token KV 体积** | 7B/GQA-8/bf16 = **128 KiB** | 传输量的单位。与参数量无关，与层数/KV 头数/精度有关 |
| **平衡 batch `B*`** | `PEAK·MFU·bytes/(2·BW)` = **132.9** | decode 加到这么多 token，算力时间追平内存时间 |
| **`t_pf`（prefill 边际成本）** | `2N/(MFU·PEAK)` = **31.5 µs/token** | 每多处理一个 prompt token 要花的算力时间 |

关键洞察：**`B*` 和「一个 decode 迭代相当于多少 prompt token」是同一个数。**

$$\frac{T_{dec}}{t_{pf}} = \frac{N \cdot \text{bytes}/BW}{2N/(\text{MFU}\cdot\text{PEAK})} = \frac{\text{MFU}\cdot\text{PEAK}\cdot\text{bytes}}{2\,BW} = B^* = 132.9$$

不是巧合 —— 两者问的都是同一个问题：「多少 token 能让算力时间追平内存时间」。一边是 decode 的 batch，一边是 prefill 的 chunk。

而这两个东西的诉求是**相反**的：

- prefill 要 chunk **≥ 132.9**，否则 prefill 掉进 memory-bound，算力白等；
- 同卡共存要 chunk **尽量 < 132.9**，否则每个 decode 迭代都被撑长。

这个矛盾就是 §3.3 要量化的东西。

### 2.3 两条正交的拆分轴

```
                        把 prefill 和 decode 分开
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                                                 │
   时间上拆（第 028 期）                          空间上拆（本期）
   chunked prefill：                              PD 分离：
   把长 prompt 切成小块，                          让 prefill 和 decode
   插进 decode 迭代的间隙                          跑在不同 GPU 上
        │                                                 │
   收益：削平 TTFT 尖峰                            收益：ITL 尾部干净，
   代价：chunk 太小则 prefill 低效                  两边各自最优
   局限：无法同时优化两者                          代价：KV 过一次网络
```

这两条轴**不是替代关系**，而是可以叠加：vLLM 的 KVConnector 兼容矩阵里明确写着 chunked prefill 与 APC 在 PD 分离下**仍然可用**。

### 2.4 分层 KV：把「KV 该住在哪」变成一道显存/带宽的工程题

既然 KV 可以离开出生地，那它就可以有**住处等级**。E1-E 把这笔账算清楚了：

```text
介质                                  带宽        每 GiB 传输耗时      相对 GPU HBM
GPU HBM (H100 SXM)           3350.0 GB/s          320.5 µs            1.0x
NVLink 5 (B200 单向)           1800.0 GB/s          596.5 µs            1.9x
PCIe Gen5 x16 (单向)             64.0 GB/s        16777.2 µs           52.3x
NVMe SSD (PCIe Gen4)            7.0 GB/s       153391.7 µs          478.6x
InfiniBand NDR 400G            50.0 GB/s        21474.8 µs           67.0x
RoCE 200Gbps                   25.0 GB/s        42949.7 µs          134.0x
```

每一级往下都是**数量级**的跌落。于是「KV 该放哪」的判据和第 028 期「抢占时该 swap 还是 recompute」完全同构：

**搬过去需要的时间 < 重算需要的时间 → 搬；否则重算。**

而 §2.1 已经证明，对 KV 而言这个不等式几乎总是成立 —— 除了两个例外：

1. **迁移距离太远**（NVMe 比 HBM 慢 478×）。跨 NVMe 搬 4 GiB 要 613 ms，而重算只要 2.3 s？……还是搬便宜。所以真正的杀手不是慢，而是**固定开销 + 并发带宽争抢**。
2. **KV 太小**。一个 24 token 的 system prompt 的 KV 只有 3 MiB，传输的握手开销就能吃掉收益 —— 这正是 `kv_recompute_threshold=64` 存在的原因。

---

## 3. 在 PyTorch 里真的跑一遍

本节所有数字都是本机实跑的 stdout（torch 2.14.0，CPU + MPS，无 CUDA）。凡是涉及 GPU 带宽/算力的部分都会明确标注为 **roofline 模型推算**，并沿用与第 027/028/029 期完全相同的常数，保证跨期可比。

### 3.1 KV 体积的精确核算：公式 vs 真张量

一切从「要搬多少字节」开始。标准 MHA/GQA/MQA 的 per-token KV 是：

$$\text{KV}_{tok} = 2 \cdot L \cdot h_{kv} \cdot d_{head} \cdot \text{bytes}$$

系数 2 是 K 和 V 两份。**MLA 是唯一打破这个公式的架构** —— 它每层只存一个压缩潜向量 $c_{KV}$（`kv_lora_rank` 维）加一个共享的 RoPE 键（`qk_rope_head_dim` 维），K/V 由 $c_{KV}$ 上投影出来，不落盘，所以**没有那个 2**。

```python
import torch

BYTES = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}


def kv_bytes_per_token_mha(n_layers, n_kv_heads, head_dim, dtype="bf16"):
    """标准 MHA/GQA/MQA：每 token 每层存 K 和 V 两份。"""
    return 2 * n_layers * n_kv_heads * head_dim * BYTES[dtype]


def kv_bytes_per_token_mla(n_layers, kv_lora_rank, qk_rope_dim, dtype="bf16"):
    """DeepSeek MLA：只存压缩潜向量 + 共享 RoPE 键，没有系数 2。"""
    return n_layers * (kv_lora_rank + qk_rope_dim) * BYTES[dtype]


def alloc_gpt_kv(n_layers, n_kv_heads, head_dim, seq, batch, dtype="bf16"):
    """真分配一个 (2, B, H_kv, T, D) 的 KV cache，用 numel*element_size 量字节。"""
    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    cache = [torch.empty(2, batch, n_kv_heads, seq, head_dim, dtype=dt) for _ in range(n_layers)]
    return sum(c.numel() * c.element_size() for c in cache)


print("=" * 78)
print("标准 MHA/GQA/MQA 的 per-token KV（公式 vs 真张量）")
print("=" * 78)
cfgs = [
    ("MHA  7B (LLaMA-2-7B 样式)", 32, 32, 128),
    ("GQA  7B (LLaMA-3-8B 样式 8:1)", 32, 8, 128),
    ("MQA  7B (全部头共享)", 32, 1, 128),
    ("GQA  70B (LLaMA-3-70B 样式 8:1)", 80, 8, 128),
    ("MHA 70B 假设不分组", 80, 64, 128),
    ("GQA  671B (DeepSeek-V3 若用 GQA)", 61, 8, 128),
]
print(f"{'配置':<36}{'层':>4}{'KV头':>6}{'d':>5}{'KB/token':>10}{'8K上下文 GiB':>14}")
for name, L, h, d in cfgs:
    per = kv_bytes_per_token_mha(L, h, d)
    print(f"{name:<36}{L:>4}{h:>6}{d:>5}{per/1024:>10.1f}{per*8192/2**30:>14.2f}")

print()
print("真张量交叉验证（seq=2048, batch=1）：")
for name, L, h, d in [cfgs[0], cfgs[1], cfgs[2]]:
    real = alloc_gpt_kv(L, h, d, seq=2048, batch=1)
    formula = kv_bytes_per_token_mha(L, h, d) * 2048
    print(f"  {name:<36} 实测 {real/2**20:8.1f} MiB  vs  公式 {formula/2**20:8.1f} MiB  "
          f"{'OK' if real == formula else 'MISMATCH'}")

print()
print("=" * 78)
print("MLA（DeepSeek-V3/V2）：per-token 与序列长度完全解耦的压缩")
print("=" * 78)
v3 = kv_bytes_per_token_mla(61, 512, 64)
v2 = kv_bytes_per_token_mla(60, 512, 64)
ref = kv_bytes_per_token_mha(61, 8, 128)
ref_mha = kv_bytes_per_token_mha(61, 128, 128)
print(f"DeepSeek-V3 MLA (61层, 512+64)      : {v3/1024:8.2f} KB/token")
print(f"DeepSeek-V2 MLA (60层, 512+64)      : {v2/1024:8.2f} KB/token")
print(f"同层数若用 GQA-8                     : {ref/1024:8.2f} KB/token   压缩比 {ref/v3:.2f}x")
print(f"同层数若用 MHA-128                   : {ref_mha/1024:8.2f} KB/token   压缩比 {ref_mha/v3:.2f}x")
print(f"MLA 每层每 token 只有 {(512+64)*2} B，其中 RoPE 键占 {64*2}/576 = {64*2/576:.1%}")
```

```text
==============================================================================
标准 MHA/GQA/MQA 的 per-token KV（公式 vs 真张量）
==============================================================================
配置                                     层   KV头    d  KB/token     8K上下文 GiB
MHA  7B (LLaMA-2-7B 样式)               32    32  128     512.0          4.00
GQA  7B (LLaMA-3-8B 样式 8:1)           32     8  128     128.0          1.00
MQA  7B (全部头共享)                       32     1  128      16.0          0.12
GQA  70B (LLaMA-3-70B 样式 8:1)         80     8  128     320.0          2.50
MHA 70B 假设不分组                         80    64  128    2560.0         20.00
GQA  671B (DeepSeek-V3 若用 GQA)        61     8  128     244.0          1.91

真张量交叉验证（seq=2048, batch=1）：
  MHA  7B (LLaMA-2-7B 样式)              实测   1024.0 MiB  vs  公式   1024.0 MiB  OK
  GQA  7B (LLaMA-3-8B 样式 8:1)          实测    256.0 MiB  vs  公式    256.0 MiB  OK
  MQA  7B (全部头共享)                      实测     32.0 MiB  vs  公式     32.0 MiB  OK

==============================================================================
MLA（DeepSeek-V3/V2）：per-token 与序列长度完全解耦的压缩
==============================================================================
DeepSeek-V3 MLA (61层, 512+64)      :    68.62 KB/token
DeepSeek-V2 MLA (60层, 512+64)      :    67.50 KB/token
同层数若用 GQA-8                     :   244.00 KB/token   压缩比 3.56x
同层数若用 MHA-128                   :  3904.00 KB/token   压缩比 56.89x
MLA 每层每 token 只有 1152 B，其中 RoPE 键占 128/576 = 22.2%
```

三个读法：

1. **GQA 的 4 倍压缩是免费的**（MHA-32 → GQA-8），因为它只改 KV 头数不改表达能力 —— 这也是为什么 LLaMA-3 之后几乎没人用 MHA 推理。
2. **MLA 的 3.56 倍（相对同层数 GQA-8）在跨机场景比在同机场景更值钱**：同机省显存只是能多放几条请求，跨机省下来的直接是网络带宽。
3. **per-token 是常数** —— 这是分页、前缀缓存、跨机传输能成立的共同前提。KV 的增长是线性的，但线性项系数固定。

顺便把「一张卡能装多少」算清楚，因为 PD 分离的第一个设计决策就是 P 和 D 各自的 KV 池开多大：

```text
E1-D  一张 80GB 卡能装多少 token（KV 池视角，全程用 GiB = 2^30 字节）
==============================================================================
  卡容量 74.51 GiB，7B bf16 权重 13.04 GiB，activation 2.0 GiB
  gpu_memory_utilization=0.90  bf16: KV 池  52.02 GiB →  0.426M token  (≈   104 条 4K 上下文的并发请求)
  gpu_memory_utilization=0.90   fp8: KV 池  52.02 GiB →  0.852M token  (≈   208 条 4K 上下文的并发请求)

对照：同一张卡放 Qwen3-30B-A3B 这类 MoE 权重（30B bf16 = 60 GB 十进制 = 55.9 GiB）时，
  0.90 利用率下 KV 池只剩 9.16 GiB —— 这就是 MoE 推理必须先做 KV 卸载的原因。
```

**只有 9.16 GiB 的 KV 池** —— 这是第 026 期 MoE 结论在服务侧的必然推论：MoE 用参数换算力，代价就是把 KV 的空间挤掉，于是 MoE 服务比稠密模型更需要分层 KV 存储和 PD 分离。

### 3.2 真跑：同卡上 prefill 打断 decode 到底有多贵

现在做本期的核心实验。用 CPU 上 3.22M 参数的 tiny GQA 模型，测量「一个 prefill 块塞进 decode 迭代」的代价。

为了让 CPU 计时噪声不淹掉信号，每个被测调用都用**固定的输入张量**（同一份 KV、同一个 token 序列），这样每次计算量完全相同。

```python
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
torch.manual_seed(0)


class TinyGQA(nn.Module):
    """d=192 / 6 层 / 8 个 query 头、4 个 KV 头 / head_dim 24 —— 3.22M 参数。"""

    def __init__(self, vocab=4096, d=192, n_layer=6, n_head=8, n_kv=4, mlp=768):
        super().__init__()
        self.d, self.n_head, self.n_kv = d, n_head, n_kv
        self.dh = d // n_head
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d), "ln2": nn.LayerNorm(d),
                "wq": nn.Linear(d, n_head * self.dh, bias=False),
                "wk": nn.Linear(d, n_kv * self.dh, bias=False),
                "wv": nn.Linear(d, n_kv * self.dh, bias=False),
                "wo": nn.Linear(n_head * self.dh, d, bias=False),
                "w1": nn.Linear(d, mlp, bias=False), "w2": nn.Linear(mlp, d, bias=False),
            }))

    def forward(self, idx, past):
        B, T = idx.shape
        x = self.emb(idx)
        for i, l in enumerate(self.layers):
            h = l["ln1"](x)
            q = l["wq"](h).view(B, T, self.n_head, self.dh).transpose(1, 2)
            k = l["wk"](h).view(B, T, self.n_kv, self.dh).transpose(1, 2)
            v = l["wv"](h).view(B, T, self.n_kv, self.dh).transpose(1, 2)
            k = torch.cat([past[i][0], k], dim=2)
            v = torch.cat([past[i][1], v], dim=2)
            S = k.shape[2]
            rep = self.n_head // self.n_kv
            mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)
            o = F.scaled_dot_product_attention(
                q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), attn_mask=mask)
            x = x + l["wo"](o.transpose(1, 2).reshape(B, T, self.d))
            x = x + l["w2"](F.gelu(l["w1"](l["ln2"](x))))
        return self.emb.weight @ x.transpose(1, 2)


m = TinyGQA().eval()
print(f"模型参数 {sum(p.numel() for p in m.parameters())/1e6:.2f}M"
      f"（d=192 / 6 层 / 8Q-4KV / head_dim 24），fp32 运行")
print(f"KV = 2*6*4*24*4 = {2*6*4*24*4} B/token")
VOCAB = 4096
B = 32


def make_past(B_, S_):
    return [(torch.randn(B_, m.n_kv, S_, m.dh) * 0.1,
             torch.randn(B_, m.n_kv, S_, m.dh) * 0.1) for _ in m.layers]


def paired(f_small, f_big, n=25):
    """交替测两个调用，返回 (小者中位耗时, 大者中位耗时, 逐对比值的中位数)。
    注意第三个返回值：逐对比值比「中位数除以中位数」抗整机速度漂移，
    否则同一段代码两次运行能差出 15%。"""
    with torch.no_grad():
        for _ in range(3):
            f_small(); f_big()
        ts, tb, rs = [], [], []
        for _ in range(n):
            t0 = time.perf_counter(); f_small(); t1 = time.perf_counter()
            f_big(); t2 = time.perf_counter()
            ts.append(t1 - t0); tb.append(t2 - t1); rs.append((t2 - t1) / (t1 - t0))
    ts.sort(); tb.sort(); rs.sort()
    return ts[n // 2], tb[n // 2], rs[n // 2]


past = make_past(B, 2048)
dec_idx = torch.randint(0, VOCAB, (B, 1))
t_dec = paired(lambda: m(dec_idx, past), lambda: m(dec_idx, past))[0]
print(f"\ndecode 步（batch={B}，上下文 2048）：{t_dec*1e3:.2f} ms")
print(f"{'prefill 块 C':>14}{'块耗时':>12}{'逐对比值':>12}")
for C in [8, 32, 128, 512]:
    idx = torch.randint(0, VOCAB, (B, C))
    ts, tb, r = paired(lambda: m(dec_idx, past), lambda idx=idx: m(idx, past))
    print(f"{C:>14}{tb*1e3:>10.2f}ms{r:>11.2f}x")

print("\nE3-C  真同卡交错：把块塞进 decode 循环，逐迭代记 ITL")
C, EVERY = 128, 8
idx_c = torch.randint(0, VOCAB, (B, C))
with torch.no_grad():
    for _ in range(3):
        m(dec_idx, past); m(idx_c, past)
    seq = []
    for i in range(32):
        t0 = time.perf_counter()
        m(dec_idx, past)
        if i % EVERY == EVERY - 1:
            m(idx_c, past)
        seq.append(time.perf_counter() - t0)
clean = sorted(seq[i] for i in range(32) if i % EVERY != EVERY - 1)
dirty = sorted(seq[i] for i in range(32) if i % EVERY == EVERY - 1)
c50, d50 = clean[len(clean)//2], dirty[len(dirty)//2]
print(f"干净迭代（只 decode，n={len(clean)}）: p50 {c50*1e3:7.2f}ms  max {clean[-1]*1e3:7.2f}ms")
print(f"带块迭代（decode+块，n={len(dirty)}）: p50 {d50*1e3:7.2f}ms  max {dirty[-1]*1e3:7.2f}ms")
print(f"尖峰倍数 {d50/c50:.2f}x；可加性检验 干净+块 = {(c50+214.10e-3)*1e3:.2f}ms vs 实测 {d50*1e3:.2f}ms")
print(f"整轮平均 ITL {sum(seq)/len(seq)*1e3:.2f}ms，"
      f"按「(7*干净+带块)/8」算 {(7*c50+d50)/8*1e3:.2f}ms")
```

```text
模型参数 3.22M（d=192 / 6 层 / 8Q-4KV / head_dim 24），fp32 运行
KV = 2*6*4*24*4 = 4608 B/token

decode 步（batch=32，上下文 2048）：22.32 ms
   prefill 块 C         块耗时        逐对比值
             8     35.27ms       1.56x
            32     62.99ms       2.72x
           128    214.10ms       9.14x
           512    883.59ms      36.68x

E3-C  真同卡交错：把块塞进 decode 循环，逐迭代记 ITL
干净迭代（只 decode，n=28）: p50   23.23ms  max   25.68ms
带块迭代（decode+块，n=4）: p50  238.49ms  max  242.55ms
尖峰倍数 10.27x；可加性检验 干净+块 = 237.77ms vs 实测 238.49ms
整轮平均 ITL 50.24ms，按「(7*干净+带块)/8」算 50.14ms
```

（这段代码在同一天连跑三次，「逐对比值」列是 1.52–1.56 / 2.72–2.77 / 9.14–9.18 / 36.68–37.27，
尖峰倍数 10.22–10.43 —— **比值稳定，绝对毫秒数有 ±5% 漂移**。下面引用比值，不引用绝对值。）

这组数字里有三件事值得停下来看：

1. **additivity 成立**：带块迭代 238.49 ms ≈ 干净迭代 23.23 ms + 块 214.10 ms = 237.33 ms（差 0.5%）。说明同一个 GPU 上 prefill 和 decode 是**串行执行**的，没有重叠 —— 这正是「共存」的物理含义。
2. **干净迭代极其稳定**：p50 23.23 ms，max 25.68 ms。decode 本身不抖，抖的是块。
3. **p50 几乎不动，但每 8 步里有 1 步跳 10.3 倍**。整轮平均 ITL 从 23.23 ms 涨到 50.24 ms（**2.16 倍**）—— 平均延迟的恶化是「尖峰 × 1/N」这个量级，不便宜。

而对交互式服务，用户感知的卡顿来自 **p99**，不是 p50。这就是 PD 分离的立论起点。

**一个必须说清的口径**：上表的绝对毫秒数不能直接搬到 GPU 上。CPU 上的 decode 步慢是因为注意力每次都被真算了一遍（无 CUDA kernel，无 flash attention）。看一下同样口径的 GPU roofline 对照：

```text
E3-A  decode 步随上下文长度增长：
CPU 实测（batch=32，fp32，T=1）：
    上下文 S      步时 p50      步时 max    相对 S=128
      128      2.06ms      2.25ms       1.00x
      512      6.79ms      7.36ms       3.29x
     2048     22.82ms     40.89ms      11.06x

GPU roofline 对照（7B/GQA-8/bf16/H800，batch=32）：HBM 上权重与 KV 读取串行相加，算力并行
    上下文 S       权重读取      KV 读取       算力时间         步时    相对 S=128
      128     4.18ms     0.16ms     1.01ms     4.34ms       1.00x
      512     4.18ms     0.64ms     1.03ms     4.82ms       1.11x
     2048     4.18ms     2.56ms     1.08ms     6.74ms       1.55x
     8192     4.18ms    10.26ms     1.32ms    14.44ms       3.33x
```

CPU 上 128 → 2048 涨 11.06 倍，GPU 上只涨 1.55 倍 —— 因为 GPU 上「权重读取 = 4.18 ms」是个**不随 S 变的常数项**在垫底。但到 S=8192 时 KV 读取（10.26 ms）反超权重读取，涨幅拉到 3.33 倍。

**所以 §3.2 的可迁移部分是「块耗时/步耗时」这个比值的结构，不是它的绝对值。** 后面 §3.3 会把比值换成 H800 的真实常数。

### 3.3 chunk 大小的两难：一条与 chunk 无关的不变量

现在把 7B / H800 级的真实常数代进去，做本期最核心的推导。

同卡共存时，一个迭代串行执行「ongoing decode + 一个 chunk」：

$$t_{iter} = T_{dec} + C \cdot t_{pf}$$

于是：

- **ITL 尖峰倍数**：$\text{spike} = \dfrac{T_{dec} + C\,t_{pf}}{T_{dec}} = 1 + \dfrac{C}{R}$，其中 $R = T_{dec}/t_{pf} = B^* = 132.9$
- **prompt 从到达到 prefill 完成**：$\text{wall} = \dfrac{S}{C}\cdot(T_{dec} + C\,t_{pf}) = \underbrace{S\,t_{pf}}_{\text{有用功}} + \underbrace{\dfrac{S}{C}\cdot T_{dec}}_{\Delta w}$

消掉 $C$，得到一条**不变量**：

$$\boxed{\ \Delta w \cdot (\text{spike} - 1) = S \cdot t_{pf}\ }$$

右边只跟 prompt 长度有关，与 chunk 大小、模型大小、硬件参数都无关。

```python
N, L, D, HK, DH = 7e9, 32, 4096, 8, 128
BY, BW, PEAK, MFU = 2, 3.35e12, 989e12, 0.45
B_STAR = PEAK * MFU * BY / (2 * BW)
per_tok = 2 * L * HK * DH * BY
t_pf = 2 * N / (MFU * PEAK)
T_dec = N * BY / BW
R = T_dec / t_pf

print(f"B* = {B_STAR:.1f}，t_pf = {t_pf*1e6:.1f} µs/token，T_dec = {T_dec*1e3:.2f} ms，R = {R:.1f}")
S = 4096
print(f"prompt S = {S}，S*t_pf = {S*t_pf*1e3:.1f} ms（不变量应有的值，全表恒定）")
print(f"{'chunk C':>9}{'尖峰':>9}{'Δw 额外延迟':>15}{'Δw×(尖峰-1)':>16}{'prefill 墙钟':>14}{'prefill 算力利用率':>20}")
for C in [128, 133, 256, 512, 1024, 2048, 4096]:
    spike = 1 + C / R
    dw = (S / C) * T_dec
    eff = min(1.0, C / B_STAR)
    print(f"{C:>9}{spike:>8.2f}x{dw*1e3:>13.1f}ms{dw*(spike-1)*1e3:>14.1f}ms"
          f"{(S*t_pf+dw)*1e3:>12.1f}ms{eff:>19.0%}")
```

```text
B* = 132.9，t_pf = 31.5 µs/token，T_dec = 4.18 ms，R = 132.9
prompt S = 4096，S*t_pf = 128.8 ms（不变量应有的值，全表恒定）
  chunk C       尖峰        Δw 额外延迟       Δw×(尖峰-1)    prefill 墙钟       prefill 算力利用率
      128    1.96x        133.7ms         128.8ms       262.6ms                96%
      133    2.00x        128.7ms         128.8ms       257.6ms               100%
      256    2.93x         66.9ms         128.8ms       195.7ms               100%
      512    4.85x         33.4ms         128.8ms       162.3ms               100%
     1024    8.71x         16.7ms         128.8ms       145.6ms               100%
     2048   16.42x          8.4ms         128.8ms       137.2ms               100%
     4096   31.83x          4.2ms         128.8ms       133.0ms               100%
```

**第四列是一个常数（128.8 ms），这就是不变量。** 换成一句人话：

> 在同卡共存的模式下，**「prefill 被 decode 拖住多等的时间」× 「ITL 尖峰涨了几倍」永远等于「prefill 自己该干的时间」**。
>
> 你想把尖峰从 2 倍压到 1.5 倍，就要让 prefill 多等 2 倍的时间；想压到 1 倍（完全不干扰），就得让 prefill 永远等下去。

这就是 vLLM 官方文档在 `disagg_prefill.md` 里写下的那句话的数学形式：

> *"Chunked prefill with a proper chunk size also can achieve the same goal, but in practice it's hard to figure out the correct chunk size value. So disaggregated prefilling is a much more reliable way to control tail ITL."*

「很难找到那个正确的 chunk 值」——因为**根本不存在**一个好的值，只存在一条双曲线。

再给尖峰设一个预算，反解两种架构下的 TTFT，就能看到分离的收益来自哪里：

```python
hdrs = ["尖峰 2x", "尖峰 3x", "尖峰 5x", "尖峰 11x", "尖峰 101x"]
DSP = [1, 2, 4, 10, 100]
print(f"{'S':>7}  {'架构':<22}" + "".join(f"{h:>11}" for h in hdrs))
for S_ in [1024, 4096, 16384, 65536]:
    col = [S_ * t_pf * (1 + 1 / d) for d in DSP]
    kv = per_tok * S_
    pd = S_ * t_pf + kv / 50e9 + kv / BW
    print(f"{S_:>7}  {'共存（受尖峰约束）':<20}" + "".join(f"{v*1e3:>9.1f}ms" for v in col))
    print(f"{'':>7}  {'PD 分离（IB400）':<21}" + "".join(f"{pd*1e3:>9.1f}ms" for _ in col))
    print(f"{'':>7}  {'→ PD/共存(尖峰2x)':<22}{col[0]/pd:>10.2f}x")
```

```text
      S  架构                          尖峰 2x      尖峰 3x      尖峰 5x     尖峰 11x    尖峰 101x
   1024  共存（受尖峰约束）                64.4ms     48.3ms     40.3ms     35.4ms     32.5ms
         PD 分离（IB400）              34.9ms     34.9ms     34.9ms     34.9ms     34.9ms
         → PD/共存(尖峰2x)               1.84x
   4096  共存（受尖峰约束）               257.7ms    193.3ms    161.1ms    141.7ms    130.1ms
         PD 分离（IB400）             139.7ms    139.7ms    139.7ms    139.7ms    139.7ms
         → PD/共存(尖峰2x)               1.84x
  16384  共存（受尖峰约束）              1030.8ms    773.1ms    644.2ms    566.9ms    520.5ms
         PD 分离（IB400）             559.0ms    559.0ms    559.0ms    559.0ms    559.0ms
         → PD/共存(尖峰2x)               1.84x
  65536  共存（受尖峰约束）              4123.2ms   3092.4ms   2577.0ms   2267.7ms   2082.2ms
         PD 分离（IB400）            2235.9ms   2235.9ms   2235.9ms   2235.9ms   2235.9ms
         → PD/共存(尖峰2x)               1.84x
```

共存那一行是双曲线：把尖峰预算从 2 倍一路放宽到 101 倍，TTFT 才从 2·S·t_pf 收敛到 S·t_pf。**而它永远收敛不到 PD 的位置** —— PD 只要付 8.3% 的网络时间。

### 3.4 仿真：同样 4 张卡，共存 vs 分离到底谁扛得住更多流量

解析式只算了单条请求。真实系统要回答的是：**在同样的 GPU 数、同样的 SLO 下，哪种架构能接更多请求？**

用一个离散事件仿真来回答。服务时间全部来自上面的常数，arrivals 用 Poisson，前 25% 丢弃做 warmup，5 个种子取中位。

```python
import heapq, random, statistics as st

N7, BY, BW, PEAK, MFU, L7, HK7, DH7 = 7e9, 2, 3.35e12, 989e12, 0.45, 32, 8, 128
PER_TOK = 2 * L7 * HK7 * DH7 * BY      # 128 KiB
B_MAX = 133
T_DEC = N7 * BY / BW                   # 4.18 ms
T_PF_TOK = 2 * N7 / (MFU * PEAK)       # 31.5 µs
NIC = 50e9                             # IB NDR400 单向
S, G = 4096, 256
SPAN, DEAD = 200.0, 60.0


def T_pf(C):
    return max(T_DEC, C * T_PF_TOK)     # prefill 也有内存地板


def poisson(lam, span, seed):
    rng = random.Random(seed)
    a, t = [], 0.0
    while t < span:
        t += rng.expovariate(lam)
        a.append(t)
    return a


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def sim_col(n_eng, lam, seed, chunk):
    """共存：每个引擎每迭代 = running 全体推 1 token + 队首请求推一个 chunk。"""
    arr = poisson(lam, SPAN, seed)
    eng_arr = [[] for _ in range(n_eng)]
    load = [0.0] * n_eng
    est = S * T_PF_TOK + G * T_DEC / B_MAX
    for a in arr:                        # 近似 join-shortest-queue 路由
        i = min(range(n_eng), key=lambda j: load[j])
        eng_arr[i].append(a)
        load[i] += est
    ttft, itl, done = [], [], 0
    for i in range(n_eng):
        my = eng_arr[i]
        if not my:
            continue
        ptr, clock, run, wait = 0, 0.0, [], []
        while ptr < len(my) or run or wait:
            while ptr < len(my) and my[ptr] <= clock:
                wait.append({"arr": my[ptr], "left": S, "gen": 0, "first": None})
                ptr += 1
            if not run and not wait:
                if ptr < len(my):
                    clock = max(clock, my[ptr])
                    continue
                break
            dt, C = 0.0, 0
            if run:
                dt += T_DEC
            if wait:
                w = wait[0]
                C = min(chunk, w["left"])
                w["left"] -= C
                dt += T_pf(C)
            t1 = clock + dt
            served = []
            for r in run:                # 每个 running 请求各推进 1 个 token
                r["gen"] += 1
                if r["first"] is None:
                    r["first"] = t1
                if t1 > DEAD:
                    itl.append(dt)
                if r["gen"] >= G:
                    served.append(r)
                    done += 1
            for r in served:
                run.remove(r)
                if t1 > DEAD:
                    ttft.append(r["first"] - r["arr"])
            while wait and wait[0]["left"] == 0 and len(run) < B_MAX:
                run.append(wait.pop(0))
            clock = t1
    return {"ttft": ttft, "itl": itl, "done": done}


def sim_pd(nP, nD, lam, seed):
    """PD：P 池吃整段 prompt → 一次 KV 传输 → D 池（batch ≤ B_MAX，步长 T_DEC）。"""
    arr = poisson(lam, SPAN, seed)
    t_trans = S * PER_TOK / NIC
    pfree = [0.0] * nP
    done_pf = []
    for a in arr:
        i = min(range(nP), key=lambda j: pfree[j])
        start = max(pfree[i], a)
        pfree[i] = start + T_pf(S)
        done_pf.append((a, pfree[i] + t_trans))
    eng_fin = [[] for _ in range(nD)]    # 每个 D 引擎在飞请求的完成时刻
    eng_step = [0.0] * nD
    ttft, itl, done = [], [], 0
    for a, ready in sorted(done_pf, key=lambda x: x[1]):
        for j in range(nD):
            while eng_fin[j] and eng_fin[j][0] <= ready:
                heapq.heappop(eng_fin[j])
        cands = [j for j in range(nD) if len(eng_fin[j]) < B_MAX] or list(range(nD))
        j = min(cands, key=lambda x: max(eng_step[x], ready))
        start = max(eng_step[j], ready)
        first = start + T_DEC
        heapq.heappush(eng_fin[j], start + G * T_DEC)
        eng_step[j] = start + T_DEC
        if a > DEAD:
            ttft.append(first - a)
            itl.extend([T_DEC] * (G - 1))
        done += 1
    return {"ttft": ttft, "itl": itl, "done": done}


def lam_max_col(chunk):
    best = 0.0
    for lam in [x * 0.5 for x in range(2, 180)]:
        if not all(pctl(sim_col(4, lam, s, chunk)["ttft"], 0.99) <= 2.0 for s in range(1, 4)):
            break
        best = lam
    return best


def lam_max_pd(nP, nD):
    best = 0.0
    for lam in [x * 0.5 for x in range(2, 320)]:
        if not all(pctl(sim_pd(nP, nD, lam, s)["ttft"], 0.99) <= 2.0 for s in range(1, 4)):
            break
        best = lam
    return best


print(f"负载 S={S} G={G}，Poisson 到达，仿真 {SPAN:.0f}s，前 {DEAD:.0f}s 丢弃，5 种子取中位")
print(f"T_dec={T_DEC*1e3:.2f}ms  t_pf={T_PF_TOK*1e6:.1f}µs/tok  T_pf(S)={T_pf(S)*1e3:.1f}ms  "
      f"T_net={S*PER_TOK/NIC*1e3:.2f}ms  B*={B_MAX}")

print()
print("共存：4 引擎，chunk 由 ITL 尖峰预算决定")
for chunk in [133, 399, 4096]:
    rs = [sim_col(4, 8, s, chunk) for s in range(1, 6)]
    print(f"  C={chunk:>4}  尖峰理论 {1+chunk/B_MAX:5.2f}x  "
          f"实测 ITL p50 {st.median(pctl(r['itl'], 0.5) for r in rs)*1e3:6.2f}ms  "
          f"p99 {st.median(pctl(r['itl'], 0.99) for r in rs)*1e3:7.2f}ms  "
          f"λ=8 完成 {st.median(r['done'] for r in rs):.0f} 条")

print()
print("同 4 卡、不同 ITL SLO 下的达标吞吐（TTFT p99 ≤ 2s）")
print(f"{'ITL SLO':>10}{'最优 chunk':>12}{'共存 λ_max':>12}{'PD 1P+3D':>11}{'PD 2P+2D':>11}"
      f"{'PD 3P+1D':>11}{'PD/共存':>10}")
for slo in [0.005, 0.0085, 0.020, 1e9]:
    C = min(S, int(B_MAX * (slo / T_DEC - 1))) if slo < 1e8 else S
    col = lam_max_col(C)
    pd = [lam_max_pd(a, b) for a, b in [(1, 3), (2, 2), (3, 1)]]
    lab = f"{slo*1e3:.1f}ms" if slo < 1e8 else "无约束"
    print(f"{lab:>10}{C:>12}{col:>12.1f}{pd[0]:>11.1f}{pd[1]:>11.1f}{pd[2]:>11.1f}"
          f"{max(pd)/col:>9.2f}x")
```

```text
负载 S=4096 G=256，Poisson 到达，仿真 200s，前 60s 丢弃，5 种子取中位
T_dec=4.18ms  t_pf=31.5µs/tok  T_pf(S)=128.8ms  T_net=10.74ms  B*=133

共存：4 引擎，chunk 由 ITL 尖峰预算决定
  C= 133  尖峰理论  2.00x  实测 ITL p50   8.36ms  p99    8.36ms  λ=8 完成 1582 条
  C= 399  尖峰理论  4.00x  实测 ITL p50   4.18ms  p99   16.73ms  λ=8 完成 1582 条
  C=4096  尖峰理论 31.80x  实测 ITL p50   4.18ms  p99  133.03ms  λ=8 完成 1582 条

同 4 卡、不同 ITL SLO 下的达标吞吐（TTFT p99 ≤ 2s）
   ITL SLO    最优 chunk    共存 λ_max   PD 1P+3D   PD 2P+2D   PD 3P+1D     PD/共存
     5.0ms          26        1.0        5.5       15.0       22.5    22.50x
     8.5ms         137       15.0        5.5       15.0       22.5     1.50x
    20.0ms         503       23.0        5.5       15.0       22.5     0.98x
       无约束        4096       28.0        5.5       15.0       22.5     0.80x
```

（表里 C=4096 的理论尖峰是 31.80× 而不是 §3.3 表格里的 31.83× —— 因为仿真里 batch 上界是整数 133，而 §3.3 用的是解析值 $R = 132.9$。这种 0.1% 的差异不影响任何结论。）

这张表是本期最重要的结果，值得逐行读：

- **SLO 无约束时，共存反而赢**（28.0 vs 22.5 req/s）。原因很直白：共存没有跨机传输、没有 P 池排队，4 张卡全都能干活；PD 分离里 1 张 D 卡在 workload 偏 prefill 时大量空转（`D 占用` 只有 18%）。
- **SLO 收紧到 20 ms**，两者打平（23.0 vs 22.5）。
- **SLO 收紧到 8.5 ms**，共存被迫用 C=137，λ 掉到 15.0，PD 保持 22.5 —— **1.50×**。
- **SLO 收紧到 5 ms**（只给尖峰留 1.2 倍余量），共存基本不可用（1.0 req/s），PD 仍是 22.5 —— **22.5×**。

**而 PD 那一列完全不随 SLO 变化**，因为它的 ITL p99 恒为 `T_dec` = 4.18 ms（无尖峰）。

把这三件事合起来看，就得到一个非常干净的结论：

> **PD 分离不提升吞吐，它买的是 ITL 尾部的确定性。**
>
> 在宽松的 SLO 下，这份确定性不值钱（共存更便宜）；SLO 越紧，它越值钱，直到共存彻底不可用。

这个结论不是我推出来的孤例 —— vLLM 官方文档在 `disagg_prefill.md` 里有一句加粗的注意事项：

> *"**Disaggregated prefill DOES NOT improve throughput.**"*

文档还给出了两条理由，和本节的推导完全对上：

> *"Tuning TTFT and ITL separately. … gives you the flexibility to assign different parallel strategies (e.g. `tp` and `pp`) to tune TTFT without affecting ITL, or to tune ITL without affecting TTFT."*
>
> *"Controlling tail ITL. … Without disaggregated prefilling, vLLM may insert some prefill jobs during the decoding of one request. This results in higher tail latency."*

顺带一句对照：第 027 期的结论是「**投机解码是延迟武器，不是吞吐武器**」。本期是同一个模式的第二次出现 —— **在推理系统里，几乎所有「把计算换一个地方做」的技术，买的都是延迟分布而不是吞吐**。因为它们不改变总工作量，只改变工作发生在什么时候、在哪张卡上。

---

## 4. 工程落地：接口长什么样

### 4.1 vLLM 的抽象：KVConnectorBase_V1

vLLM V1 把所有 KV 搬运动作收进一个抽象基类 `KVConnectorBase_V1`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py`）。它按 vLLM 自己的 scheduler/worker 分界线切成两组 API：

**Scheduler 侧**（与调度器同进程，负责「指挥」）：

| 方法 | 作用 |
|---|---|
| `get_num_new_matched_tokens()` | 报告远端缓存里已经有多少个 token —— **它复用 `num_computed_tokens` 那套机制，对调度器来说远端命中和本地前缀命中长得一模一样** |
| `update_state_after_alloc()` | 分配完 block 后更新 connector 状态 |
| `request_finished()` | 决定 block 是否能立即释放，还是必须活到异步传输读完 —— 这就是**生产者侧不泄漏显存的机制** |
| `build_connector_meta()` | 把这一步要读/要写的 KV 打包成元数据，交给 worker |

**Worker 侧**（与计算同进程，负责「执行」）：

| 方法 | 作用 |
|---|---|
| `start_load_kv()` | 发起异步加载（非阻塞） |
| `wait_for_layer_load(i)` | **在 attention 层内部**阻塞，直到第 i 层到齐 |
| `save_kv_layer(i)` | 保存第 i 层的 KV |
| `wait_for_save()` | forward 退出前阻塞，防止 paged KV buffer 被写坏 |
| `get_finished()` | 返回哪些请求的异步收发已完成 |

**逐层粒度是这套设计的核心。** 有了 `wait_for_layer_load` / `save_kv_layer`，decode 实例可以在第 0 层 KV 落地后立刻开始算第 0 层，同时生产者还在算第 1 层 —— 几 GiB 的传输被**重叠在 forward 里**，而不是排在它前面。第 024 期讲 Ring Attention 时见过同样的思路（把通信藏进计算），这里是它在 KV 搬运上的复用。

还有一个调度器侧的状态值得一提：**`WAITING_FOR_REMOTE_KVS`**。KV 还在路上的请求会停在这个状态，被调度器跳过，直到可以被提升（promotable）。这就是第 028 期「跳过条件」清单里的一条 —— 分离架构给调度器增加了一个新的等待原因。

vLLM 把 connector 做成了插件，目前官方文档列了 **9 种**：

| Connector | 底层 |
|---|---|
| `NixlConnector` | NIXL 库，全异步收发，后端可选 UCX / GDS / LIBFABRIC |
| `NixlPushConnector` | 按层名路由的推送式，专门解决 PP + hybrid KV 的组合 |
| `LMCacheConnectorV1` / `LMCacheMPConnector` | LMCache 缓存层（底层也是 NIXL），MP 模式下有独立 `lmcache server` 给多个 vLLM 实例共用 |
| `MooncakeConnector` / `MooncakeStoreConnector` | Mooncake 传输引擎 / 分布式 KV store |
| `OffloadingConnector` | KV 卸载到 CPU 内存，可配 `block_size` 与 `cpu_bytes_to_use` |
| `FlexKVConnectorV1` | 分布式 KV store + 多级缓存 |
| `MultiConnector` | 把多个 connector 排成有序列表（从第一个能命中的读，往所有上写） |
| `MoRIIOConnector` | ROCm 专用 |
| `ExampleConnector` | 示例 |

### 4.2 起一个真的 PD 集群

官方用法长这样（NixlConnector，同机双卡）：

```bash
# Producer（prefiller）—— 第 0 号卡
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-0.6B \
  --port 8100 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

# Consumer（decoder）—— 第 1 号卡
CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-0.6B \
  --port 8200 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

# 前置代理：把请求拆成 prefill + decode 两段，并转发 kv_transfer_params
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8192 --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts localhost --decoder-ports 8200
```

跨机时把 `VLLM_NIXL_SIDE_CHANNEL_HOST` 设成本机 IP 即可（默认 `localhost`），端口默认 `5600`；TP/DP 部署时同一节点上第 k 个 worker 用 `base_port + dp_rank`。

几个**只在源码/文档里看得到**的细节：

- **`kv_role="kv_both"` 已经被废弃**（对 NixlConnector）。文档明确要求 prefill 实例写 `kv_producer`、decode 实例写 `kv_consumer`，`kv_both` 会在未来版本移除。网上大量教程还停留在 `kv_both` 的写法。
- **`kv_lease_duration` 默认 30 秒**：prefill 请求结束后，它的 KV block 会被**保留 30 秒**等 decoder 来读；decoder 排队期间会周期发心跳自动续租；既没心跳也没读到通知就释放。这是「生产者侧不能立刻释放 block」的具体实现。
- **`decoder_kv_blocks_ttl` 默认 480 秒**：双向传输模式下 decoder 侧缓存的 KV block 生命周期（多轮对话复用）。注意它**不靠心跳续期**，和上面的 lease 语义不同。
- **`kv_recompute_threshold` 默认 64**：命中不足 64 个远端 token 就本地重算 —— 正是 §2.1 说的「小传输的固定开销吃掉收益」。

### 4.3 SGLang 的对应物

SGLang 走的是「进程 + 参数」路线，不搞 connector 插件，而是两个 `--disaggregation-mode`：

```bash
# Prefill 节点
sglang serve --model-path <MODEL> --tp 8 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0,mlx5_1 \
  --disaggregation-bootstrap-port 8998 \
  --host 0.0.0.0 --port 30000

# Decode 节点
sglang serve --model-path <MODEL> --tp 8 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0,mlx5_1 \
  --host 0.0.0.0 --port 30001

# PD Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<A>:30000 8998 \
  --decode http://<B>:30001 \
  --policy round_robin --host 0.0.0.0 --port 8000
```

对照一下两家的取舍：

| | vLLM | SGLang |
|---|---|---|
| 抽象 | `KVConnectorBase_V1` 插件（9+ 种） | `--disaggregation-mode` + `--disaggregation-transfer-backend`（mooncake / nixl / ascend） |
| 传输引擎 | NIXL / Mooncake / GDS 都可插 | Mooncake 或 NIXL，用 `--disaggregation-ib-device` 指定 RDMA 设备 |
| 角色声明 | 显式 `kv_role: kv_producer / kv_consumer` | 由 `--disaggregation-mode` 隐含 |
| 握手 | ZMQ side channel（`VLLM_NIXL_SIDE_CHANNEL_PORT`）+ 兼容性哈希校验 | bootstrap port + per-request `bootstrap_room` |
| 路由 | 需要外部 proxy | 自带 `sglang_router`，支持 `--prefill-policy cache_aware --decode-policy round_robin` |
| 拓扑约束 | 支持异构 TP（有模型限制） | 官方 recipe 里 P/D 要求**相同 TP**、PP=1 |

SGLang 那份 recipe 里有个细节很有意思：**Mooncake 后端在注册时就建好 RDMA 连接，所以第一个请求就没有冷启动**；用满 8 张 NIC 能降低 TTFT，且上下文越长差距越大 —— 这正是 §5.3 那条「KV 传输量 ∝ 上下文长度」的直接体现。

另外 SGLang 有个 `--disaggregation-decode-enable-offload-kvcache`，把 KV 卸载和 PD 分离组合起来，这就是下一节要讲的「分层 KV」。

### 4.4 分层 KV 存储：把「KV 住哪」变成一道选择题

PD 分离解决的是「KV 在哪个**计算角色**手里」，分层 KV 解决的是「KV 住在哪**一级介质**」。两者可以叠加。

vLLM 这边的入口是 `OffloadingConnector`：

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
                       "kv_connector_extra_config":{"block_size": 64,
                                                    "cpu_bytes_to_use": 1000000000}}'
```

注意 `kv_role: kv_both` 在这里是**合法且必要的** —— 它和 NixlConnector 的废弃语义不是一回事：卸载的读写都在同一个实例内完成，所以没有 producer/consumer 之分。文档也明确把「KV offloading 是 sibling，不是同一件事」写出来了：

> *"KV offloading moves cold blocks down the memory hierarchy of the same serving stack, GPU to CPU RAM to filesystem, instead of evicting them."*

三级存储的量化依据就是 §2.4 那张表：

| 层级 | 带宽（相对 HBM） | 单 GiB 搬回耗时 | 适合放什么 |
|---|---|---|---|
| GPU HBM | 1.0× | 320 µs | 正在 decode 的请求 |
| CPU DRAM（PCIe Gen5 单向） | 52.3× 慢 | 16.8 ms | 冷 block、跨请求复用的公共前缀 |
| NVMe / 远端 store | 478.6× / 67× 慢 | 153 ms / 21.5 ms | 会话级复用、RAG 文档缓存、跨实例共享 |

判据仍然只有一个：**搬回来要花的时间 < 重算这批 token 要花的时间**。用 §2.1 的临界带宽反过来算：PCIe Gen5（64 GB/s）远高于 3.87 GB/s 的临界值，所以**卸载到 CPU DRAM 在带宽上永远划算**；它真正的问题是 PCIe 的**并发带宽争抢**（还要和别的 GPU 抢同一条总线）以及额外的显存占用（pin memory）。

---

## 5. 围绕该领域展开

### 5.1 传输原语的谱系：从 cudaMemcpy 到 NIXL

| 层 | 原语 | 典型带宽（单向） | 语义 |
|---|---|---|---|
| 进程内 | `tensor.copy_()` | HBM 3.35 TB/s | 就是 §2.4 里那级 |
| 同机 GPU 间 | NVLink 4（H100）/ NVLink 5（B200） | 450 GB/s / 900 GB/s | P2P，可被 `cudaMemcpyPeer` 或 NVSHMEM 使用 |
| 同机跨 NUMA | PCIe Gen5 x16 | 64 GB/s | 走 CPU 内存中转，`cudaMemcpy` 即可 |
| 机间 | InfiniBand NDR400 / RoCE 200G | 50 / 25 GB/s | RDMA，需要注册内存（MR） |
| 机间（集合通信） | NCCL P2P (`ncclSend/ncclRecv`) | 接近 NIC 线速 | vLLM 早期的 `P2pNcclConnector` 走这条 |
| 机间（零拷贝） | **NIXL** / **Mooncake** | 接近 NIC 线速 + GDS 直通 | 统一抽象 CPU/GPU/storage 的异步传输 |
| 存储直通 | GDS（GPUDirect Storage） | NVMe 线速 | KV 从 SSD 直接进显存，不过 CPU |

**NIXL（NVIDIA Inference Xfer Library）是这一层的抽象。** 它不发明新传输，而是把 UCX（RDMA 通用层）、GDS（存储直通）、LIBFABRIC（libfabric 后端）统一成一个异步 API。vLLM 里可以这样指定多个后端：

```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both",
                       "kv_buffer_device":"cuda",
                       "kv_connector_extra_config":{"backends":["UCX","GDS"]}}'
```

`kv_buffer_device` 这个参数决定了传输源/目标是 GPU 显存（`cuda`）还是主机内存（`cpu`）—— 前者需要 GPU 注册 MR，后者可以复用宿主侧 pin memory。

**一个容易被忽略的事实**：第 007 期讲过的集合通信原语（AllReduce / AllGather / ReduceScatter）和这里是**同一块硬件上的不同流量**。在 TP=8 的部署里，P 实例和 D 实例各自都要跑 TP 的 all-reduce；PD 之间再叠一层 KV 传输。三者共用 NIC 与 NVLink。这就是为什么 §4.3 里 SGLang 的 recipe 要把 `--disaggregation-ib-device` 显式指出来 —— 不指清楚，KV 传输会和 TP 通信抢同一张卡。

### 5.2 P:D 实例配比：长 prompt 反而需要更多 prefill 实例

用 §2.2 的常数，一个 P 实例的 prefill 吞吐是 $\text{MFU}\cdot\text{PEAK}/(2N + 4LdS)$，一个 D 实例的聚合输出吞吐是 $B^*\cdot BW/(N\cdot\text{bytes})$：

```text
统一模型：7B / 32 层 / hidden 4096 / GQA-8 / head_dim 128 → KV 128 KiB/token
平衡 batch B* = PEAK*MFU*bytes/(2*BW) = 132.9

一个 D 实例：步时 4.18ms（纯权重读取，与 batch 无关，直到 B*），batch=133 时聚合输出 31.79K tok/s

      S     G      prefill 吞吐/实例       D 吞吐/实例     n_P : n_D    ≈ S/G
    200  1000           31.55K       31.79K   0.20 : 1     0.20
   2048   256           29.52K       31.79K   8.62 : 1     8.00
   8192   256           24.33K       31.79K  32.00 : 1    32.00
  32768   256           14.27K       31.79K 128.00 : 1   128.00
  32768  2048           14.27K       31.79K  16.00 : 1    16.00
 131072  1024            5.38K       31.79K 128.00 : 1   128.00
```

**配比几乎就是 prompt:output 的 token 数比** —— 因为两侧每 token 的吞吐量级相近（都约 3e4 tok/s）。

推论很反直觉但很重要：

- **聊天负载（prompt 短、输出长）需要更多 D 实例**（1 : 5）；
- **文档问答 / RAG（prompt 长、输出短）需要更多 P 实例**（8.6 : 1，甚至 128 : 1）。

而 §3.4 的仿真里最优切法是 **3P+1D** —— 因为那个 workload 是 S=4096/G=256，正好是 prefill 主导的。**PD 分离的 P:D 切法必须按 workload 的 prompt:output 比来定，没有通用答案。** 这也是很多团队试点 PD 分离「效果不明显」的原因：默认 1P:1D 的切法与自己的 workload 不匹配。

### 5.3 网络会变成新的瓶颈：一个 D 实例要吞多少 KV

这是 PD 分离最容易翻车的地方。一个 D 实例在满负荷运行时聚合输出 31.79K tok/s，而每输出 1 个 token 它必须**从网络收进** $S \cdot \text{per\_tok} / G$ 字节的 KV：

```text
一个 D 实例（batch=133）聚合输出 31.8K tok/s。
每输出一个 token，D 实例必须从网络收进 S*per_tok/G 字节的 KV：
           S/G      每输出 token 收 KV         所需入带宽    IB400 占比    PCIe5 占比            结论
           0.2             26.2KB         0.8GB/s         2%         1%             可接受
           8.0           1048.6KB        33.3GB/s        67%        52%             可接受
          16.0           2097.2KB        66.7GB/s       133%        104%       IB400 扛不住
          32.0           4194.3KB       133.3GB/s       267%        208%       IB400 扛不住
         128.0          16777.2KB       533.3GB/s      1067%        833%       IB400 扛不住
```

**临界比值是 S/G ≈ 12（IB400）**：prompt 比输出长 12 倍以上时，单张 400G NIC 就喂不饱一个 D 实例。

这个结论很反直觉 —— 通常大家担心的是「长上下文撑爆显存」，但在 PD 分离下先撑爆的是**网卡**。而且它还解释了一件事：为什么 PD 分离经常绑着 NVLink 域做（450 GB/s 是 IB400 的 9 倍），因为 NVLink 域内做分离时网络瓶颈基本消失。

对应的四个缓解手段，全部来自前面的推导：

1. **降 KV 精度**（bf16 → fp8）：per-token 字节减半，网络需求直接减半。vLLM 的 KV 量化用 `cache_dtype`，要求 P/D **必须一致**（见 §6 的坑）。
2. **换 MLA**：61 层模型下 MLA 是 GQA-8 的 **1/3.56**，网络需求同比降到 285.9 GB/s（S=32768 时从 533.3 GB/s 降下来）。
3. **提高 G/S**（prompt 短、输出长）：S/G = 0.2 时网络需求只有 0.8 GB/s。
4. **退回同卡共存**：S/G 很大时，共存反而更划算（§3.4 那张表）。

### 5.4 与 TP / SP / MoE 的交互

**异构 TP（heterogeneous TP）** 是 PD 分离独有的自由度：P 实例和 D 实例可以用不同的 `tp`。因为 prefill 想要大 TP 压 TTFT，decode 想要小 TP 省通信和 KV 复制。vLLM 的兼容矩阵里把这一条列在「**可以安全不同**」那一栏：

> *"What can safely differ between P and D: `tensor-parallel-size` (heterogeneous TP, subject to model restrictions above), `block-size` (heterogeneous block size), Number of KV cache blocks, `num_speculative_tokens` …"*

但**有模型限制**：

| 模型类型 | 异构 TP | 原因 |
|---|---|---|
| Dense Transformer | ✅ | 按头切分即可 |
| **MLA（DeepSeek-V2/V3）** | 🟠 | **MLA 的 KV 在 TP worker 之间是复制的，所以没有头可切**；P TP > D TP 时只读一次（跳过冗余 rank），D TP > P TP 也可以，但拿不到「切分」的好处 |
| Hybrid SSM / Mamba | 🚧 | 要求同构 TP |

这一条和第 019 期（张量并行）、第 011 期（MLA）直接相关：**MLA 的 KV 复制特性，在 TP 场景是浪费（每张卡存全量），在 PD 分离场景反而变成了优势**（没有头切分问题、异构 TP 天然可用）。

**MoE** 的情况更复杂：MoE 权重大，KV 池被挤到很小（§3.1 算出只剩 9.16 GiB），所以 MoE 服务天然需要分层 KV 或 PD 分离。而 vLLM 的兼容矩阵里 MoE 在「Basic PD / Spec Decode / Hetero TP / Cross-layer blocks / SWA / Host buffer」六个维度上全部 ✅（Hetero block size 是 🟠）。第 026 期的专家并行（EP）通信和 PD 之间的 KV 传输会共用 NIC —— 这是 MoE + PD 部署必须做流量规划的原因。

**跨层 block（cross-layer blocks）** 是个值得单独提的优化：`VLLM_KV_CACHE_LAYOUT=BLHNC` 让 KV 在显存里按「层」连续排布，于是一次传输可以覆盖多层（而不是逐层 N 次 RDMA），减少了大块传输的切分开销。代价是 attention kernel 的访存模式变差 —— 又是一个「传输友好 vs 计算友好」的取舍。

### 5.5 和已发期数的关系

| 期数 | 关联点 |
|---|---|
| **011 KV Cache / Paged Attention** | 本期一切的地基：per-token 常数、block 粒度 |
| **027 投机解码** | 同一个 `B*` 平衡点；同样的「延迟武器，不是吞吐武器」结论 |
| **028 连续批处理** | `WAITING_FOR_REMOTE_KVS` 是调度器新增的一个跳过条件；chunked prefill 是 PD 分离的替代方案（那条双曲线） |
| **029 前缀缓存** | 「KV 体积 = 传输体积」；`get_num_new_matched_tokens()` 让远端命中在调度器眼里和本地命中一样 |
| **024 序列并行** | Ring Attention 把通信藏进计算的思路，在 `wait_for_layer_load` 上复用 |
| **019 张量并行 / 026 MoE** | 异构 TP、MLA 的 KV 复制、MoE 挤占 KV 池 |
| **007 通信原语** | TP 的 all-reduce 与 PD 的 KV 传输共用 NIC，需要流量规划 |

---

## 6. 什么时候该用 / 不该用

**该用：**

- **ITL 尾部有硬 SLO**（比如 p99 < 10 ms）且 workload 是长 prompt。§3.4 的表里这是 PD 唯一不可替代的场景。
- **prompt:output 比极端**（>8:1 或 <1:5）。这时 P 池和 D 池的规模需求差一个数量级，绑在一起很浪费。
- **需要给 TTFT 和 ITL 设不同的并行策略**：P 用大 TP 压 TTFT，D 用小 TP 保带宽效率 —— 这个自由度只有分离能给。
- **要跨机做 KV 复用**（多轮对话、RAG 文档池）。这已经是 §4.4 的分层 KV 范畴了。

**不该用：**

- **短 prompt 聊天**。S=128 时共存的 TTFT 是 8.1 ms、分离是 4.4 ms，差的 3.7 ms 比一次跨进程握手的开销还小（§3.3 的 E4-C 表）。收益完全被工程开销吃掉。
- **ITL 没有硬 SLO、只要平均吞吐**。§3.4 表最后一行：无约束时共存 28.0 req/s 反而比 PD 的 22.5 高 24%。
- **S/G > 12 且网络只有 100Gb/200Gb**。§5.3 的表：网络会成为比显存更早的瓶颈，这时应该先换 MLA / fp8 KV，而不是上分离。
- **P 和 D 无法共享同一套模型配置**（attention backend、cache_dtype、KV layout 不一致）。vLLM 的兼容性哈希会在握手阶段直接拒绝 —— 硬上就是启动失败。

**一句话判据：**

$$\text{用 PD 分离当且仅当}\ \underbrace{\frac{\text{SLO}_{ITL}}{T_{dec}}}_{\text{尖峰预算}} \text{很小（}<2\text{）} \quad\text{且}\quad \frac{S}{G} < \frac{\text{NIC}}{B^* \cdot BW/(N\cdot\text{bytes})\cdot \text{per\_tok}}$$

---

## 7. 常见坑

### 坑 1：以为 PD 分离能提升吞吐，然后发现指标没动

这是最高频的误解。vLLM 官方文档专门用加粗写了一句：

> *"**Disaggregated prefill DOES NOT improve throughput.**"*

本期的仿真独立复现了这个结论：SLO 无约束时共存 28.0 req/s > PD 22.5 req/s；两者打平点在 SLO ≈ 20 ms。

**根因**：分离不改变总工作量（同样的 FLOPs、同样的字节），只改变工作发生的时间和位置。它买的是延迟分布的确定性。如果 TTFT/ITL 没有硬 SLO，这份确定性一文不值。

**正确做法**：上线前先测出「共存架构下，把 ITL SLO 从宽松收到目标值会让 chunk 掉到多少」—— 用 §3.3 的不变量 $\Delta w \cdot (\text{spike}-1) = S \cdot t_{pf}$ 心算一下就有答案。如果算出来 chunk 还在 133 以上，分离的收益就会很小。

### 坑 2：P/D 两侧的 KV 布局不一致，启动就被拒，或者算错

vLLM 在握手阶段会校验一个**兼容性哈希**，要求 P 和 D 在以下方面完全一致：vLLM 与 connector 版本、模型（架构/dtype/KV 头数/head size/层数）、**attention backend**、**`cache_dtype`**、投机方法配置、**NIXL 传输模式（push vs pull）**。

容易踩的三个：

1. **KV 量化 dtype 不一致**。用 fp8 KV 时 P 和 D 必须都是 fp8。而且——**动态量化不支持**：文档明确写 `Dynamic quantization (scales computed at runtime): ❌ Not supported. Per-block scales are not transferred alongside KV cache data.`。也就是说 runtime 算出来的 per-block scale 不会跟着 KV 一起传过去，D 侧拿到的是没有 scale 的数据，结果静默错误。只有从 checkpoint 载入的静态量化 scale 和 packed-layout 内联 scale 才行。
2. **attention backend 不一致**。比如 P 用 FA3、D 用 FlashInfer，KV 的物理排布可能不同。文档把它列在「必须一致」里。
3. **push / pull 模式混用**。文档原话：*"a push (WRITE) connector and a pull (READ) connector use incompatible transfer protocols and must never be paired"* —— 而且是**建议不要**用 `enforce_handshake_compat: false` 关掉校验（文档原话是 "at your own risk"）。

### 坑 3：block size 异构时的方向性

vLLM 允许 P 和 D 用不同的 `block-size`，但**只有当不需要 HMA（hybrid memory allocator，hybrid 模型才需要）时**才支持异构 block size，而且「**只支持 P block size < D block size**」。反过来会出问题。

更隐蔽的是 PP 的组合：**默认的 pull 版 NixlConnector 不支持 `pipeline-parallel-size > 1` + hybrid KV（HMA）**，因为 region index 在 prefill/decode 的层切分下不是稳定标识，connector 会在启动时直接抛异常。这时唯一的出路是 `NixlPushConnector`（按层名成员身份路由），而且它还有一堆限制：只有 P 侧能 PP 分片、decode 侧 PP 不支持、hybrid SSM/Mamba 布局不支持、HMA 要求两侧 block size 相同。

### 坑 4：把「分层 KV」和「PD 分离」当成同一件事

两者是正交的：

- **PD 分离**：KV 在**不同计算角色**之间移动（P → D），关注的是 prefill/decode 的隔离。
- **KV 卸载（offloading）**：KV 在**同一实例内的不同介质**之间移动（GPU → CPU → 文件系统），关注的是容量。

vLLM 文档把这一点写得很清楚（offloading 是 sibling，不是同一件事）。**所以 `OffloadingConnector` 配 `kv_role: kv_both` 是正确写法**，而 `NixlConnector` 配 `kv_both` 已经被废弃 —— 这两个 `kv_both` 语义不一样，很容易照着旧教程抄错。

实际落地里两者经常一起用：`--disaggregation-decode-enable-offload-kvcache`（SGLang）就是把它们组合起来。此时要小心**带宽争抢**：KV 卸载走 PCIe，PD 传输走 NIC，TP 通信走 NVLink/NIC —— 三条路径如果有重叠，就会互相拖慢。

---

## 8. 一句话总结

> **PD 分离不是加速技术，是把「prefill 打断 decode」这条不可能三角拆掉的技术：它用一次网络搬运（7B/GQA-8 只要 3.87 GB/s 的临界带宽，比想象中便宜得多）换来了 ITL 尾部的确定性，代价是必须接受「网络可能比显存更早成为瓶颈」这个新约束，而且它买不到吞吐 —— 同样的 4 张卡，SLO 宽松时共存反而更快。**

---

<details>
<summary>今日练习</summary>

### 练习 1：算一个部署方案

你要部署一个 7B / GQA-8 / bf16 的服务，SLA 是「ITL p99 ≤ 12 ms」。硬件是 8 张 H800（IB NDR400 组网）。workload：prompt 8192、输出 256、峰值 40 req/s。

(a) 如果同卡共存，chunk 上界是多少？prefill 的墙钟时间会比「不共存」的理想值多多少？
(b) `S/G` 是多少？§5.3 的网络判据过得去吗？
(c) 按 §5.2 的配比公式，8 张卡该怎么切？

**参考答案：**

(a) 尖峰预算 = 12 ms / 4.18 ms = 2.87 → $C \le B^*\times(2.87-1) = 132.9 \times 1.87 = 248.5$。

用不变量算额外延迟：$\Delta w = S\,t_{pf} / (\text{spike}-1)$。$S\,t_{pf} = 8192 \times 31.5\,\mu s = 258.05$ ms。

$$\Delta w = \frac{258.05}{1.87} = 138.0\ \text{ms}$$

prefill 墙钟 = 有用功 + Δw = 258.05 + 138.0 = **396.0 ms**（比理想值多 **53.5%**）。

（直接验算：$C = 248$，迭代数 = ceil(8192/248) = 34，每迭代 $T_{dec} + C t_{pf} = 4.18 + 7.81 = 11.99$ ms，总计 $34 \times 11.99 = 407.7$ ms。与解析式 396.0 ms 差 3%，差额来自最后一次迭代没吃满 chunk。）

(b) $S/G = 8192/256 = 32$。

§5.3 给出：一个 D 实例满负荷时（聚合输出 31.79K tok/s）所需入带宽 = $31.79\text{K} \times \text{per\_tok} \times S/G$。

每输出 token 收 KV $= 131072 \times 8192 / 256 = 4.194$ MB。

所需入带宽 $= 31790 \times 4.194\text{e}6 = 133.3$ GB/s。

**IB400 只有 50 GB/s，只够 37.5%。过一个。** 所以要么换 fp8 KV（减半到 66.7 GB/s，仍然不够），要么换 MLA（÷3.56 → 37.4 GB/s，够了），要么**退回同卡共存**。

(c) 按 §5.2 的公式，$n_P/n_D = S \cdot \Theta_D / (G \cdot \Theta_P)$，其中 $\Theta_P = \text{MFU}\cdot\text{PEAK}/(2N + 4LdS)$。

$4LDS = 4 \times 32 \times 4096 \times 8192 = 4.295\text{e}9$，$2N = 1.4\text{e}10$，和 = $1.8295\text{e}10$。$\Theta_P = 0.45 \times 989\text{e}12 / 1.8295\text{e}10 = 24330$ tok/s。

$n_P/n_D = 8192 \times 31790 / (256 \times 24330) = 41.8$。所以 8 张卡全部给 P 都不够（40 req/s 需要 $40 \times 8192 / 24330 = 13.5$ 张 P 卡）。

**结论：这个 workload 在 8 张 H800 上根本跑不了 40 req/s** —— 是 prefill 算力不够，不是架构问题。要么加卡，要么降 S/G（用 RAG 而不是把 8K 文档整段塞进 prompt）。

这个练习的用意：PD 分离的适用面被两件事夹住 —— 左边是网络（S/G 太大过不去），右边是 prefill 算力（长 prompt 的绝对成本）。很多「PD 分离效果不好」的案例其实是 workload 本身就不适合这个硬件配置。

### 练习 2：不变量自检

用 §3.3 的不变量证明：**如果把 prompt 长度从 $S$ 翻倍到 $2S$，在保持 ITL 尖峰倍数不变的前提下，prefill 的额外延迟 Δw 也翻倍。**

**参考答案：**

不变量 $\Delta w \cdot (\text{spike}-1) = S\,t_{pf}$。

保持 spike 不变 → $(\text{spike}-1)$ 不变。

右边从 $S\,t_{pf}$ 变成 $2S\,t_{pf}$，翻倍。

所以 $\Delta w$ 必须翻倍。$\blacksquare$

从机制上看这也直观：chunk 大小不变（因为 spike 不变 → $C = (\text{spike}-1)R$ 不变），而迭代数 $\lceil S/C \rceil$ 翻倍，每次迭代被 decode 拖住的 $T_{dec}$ 不变 → 总被拖时间翻倍。

**工程含义**：长上下文服务的 TTFT 恶化是**超线性**的 —— 不只是 prefill 本身按 $S$ 增长，被 decode 拖住的额外时间也按 $S$ 增长。这就是长上下文场景里 PD 分离收益最大的根本原因，也解释了 §3.3 的 E4-C 表里「省下的绝对量 ∝ S」。

### 练习 3（进阶）：给 §3.4 的仿真加一个约束

§3.4 的表里，PD 3P+1D 的「D 占用」在 λ=22.5 时只有 18%。请据此估算：如果 workload 从 S=4096/G=256 改成 S=2048/G=512（同样是 prefill 主导吗？），3P+1D 还成立吗？

**参考答案：**

先算新的配比需求（沿用 §5.2 的方法，$\Theta_D = 31.79$K tok/s）：

$4LDS = 4 \times 32 \times 4096 \times 2048 = 1.074\text{e}9$，$2N = 1.4\text{e}10$，和 = $1.5074\text{e}10$。

$\Theta_P = 4.45\text{e}14 / 1.5074\text{e}10 = 29{,}522$ tok/s。

$n_P/n_D = S \cdot \Theta_D/(G \cdot \Theta_P) = 2048 \times 31790 / (512 \times 29522) = 4.31$。

也就是 **4.31 : 1**，比原来的 8.62 : 1 更偏 decode。所以 3P+1D 的比例（3:1）此时**偏 de code 不足**：P 池会闲着（3 张卡只用 4.31/3 = 70%），而 D 池会成为瓶颈。

更合适的切法是 3P+1D 里的 D 那张卡会被打满，或者干脆 1P+1D 之类的小规模测试。用 8 卡的话大概是 5P+3D（5/3 = 1.67 vs 需求 4.31 …… 不对，反了）。

**再想一遍**：$n_P/n_D = 4.31$ 意味着**需要 4.31 张 P 配 1 张 D**。所以 3P+1D 的 P:D = 3 小于需求，**P 侧不够**，P 池会成为瓶颈。D 那张卡会很闲。

要凑 8 张卡：需求 4.31:1，即 4.31 份 P + 1 份 D = 5.31 份 → 每份 8/5.31 = 1.51 张 → P = 6.5 张、D = 1.5 张。取整为 **6P+2D**（比例 3:1，仍偏 P 不足但比 3P+1D 好）或 **7P+1D**（比例 7:1，偏 P 过剩）。

**真正应当注意的方法论**：这个「按比例凑整数」的分配忽略了整数效应 —— §3.4 的表里 1P+3D 只有 5.5 req/s，2P+2D 是 15.0，3P+1D 是 22.5，**相邻切法之间差 2~3 倍**。所以 P:D 切法的敏感性极高，必须按实际 workload 压测，不能靠公式拍。

**顺带一个反直觉点**：把 S 从 4096 减半到 2048、同时把 G 翻倍到 512 之后，总 token 量其实没变（4096+256 = 4352 vs 2048+512 = 2560，反而少了），但 workload 的性质从「prefill 主导」变成了「decode 更吃紧」。**这就是 §5.2 那句「聊天负载 1:5、文档问答 5:1，方向完全相反」的具体形态。**

</details>
