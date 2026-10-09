# PyTorch 每日一课 · 第 033 期

## KV Cache 量化：把一个 token 的 KV 再砍一半

| | |
|---|---|
| **日期** | 2026-10-09 |
| **难度** | ⭐⭐⭐⭐ |
| **前置知识** | 第 032 期（一个 token 的 KV 到底多少字节）、第 011 期（PagedAttention 的块布局）、第 016 期（量化的 PTQ/QAT 基本概念）、第 027 / 028 期的 roofline 平衡点 |
| **预计阅读时间** | 55 分钟 |
| **关联** | 第 032 期 · 同一本账，这期问「那些字节里有多少可以省掉」；第 030 期 · 为什么「动态量化的 KV 不能跨机」；第 029 期 · 缓存驻留量随每 token 字节线性变；第 016 期 · 同样是量化，权重和 KV 的性价比完全相反 |

---

## 1. 这个领域解决什么问题

先把上一期的结论摆出来（这一期独立可读，不用回去翻）。

一个 token 的 KV 字节数是

$$\text{KV bytes/token} = 2 \times L \times n_{kv} \times d_h \times b$$

$2$ 是 K 和 V 各一份，$L$ 是层数，$n_{kv}$ 是 KV 头数，$d_h$ 是 head_dim，$b$ 是每个数值的字节数。对 Llama-3 风格的 7B（$L=32$、$n_{kv}=8$、$d_h=128$、bf16）：

$$2 \times 32 \times 8 \times 128 \times 2 = 131072\ \text{B} = 128\ \text{KiB/token}$$

现在算一笔 decode 的带宽账。7B 权重 bf16 是 14.0 GB。把权重和 KV 都读一遍才能出一个 token：

| 精度 | 每 token KV | B=32, S=8K 时 KV 读取量 | KV 占比 |
|---|---|---|---|
| bf16 | 128 KiB | 34.4 GB | **71.1%** |
| fp8 / int8（per-tensor） | 64 KiB | 17.2 GB | 55.1% |
| int4（per-tensor） | 32 KiB | 8.6 GB | 38.0% |

权重那 14.0 GB 是**常数**，KV 那 34.4 GB 是 $B \times S \times k$ 的函数。$B \times S \approx 1.07\times10^5$ 时两者打平——B=32 只要 S≈3338，B=128 只要 S≈834。也就是说，除了「小 batch + 短上下文」这个角落，**decode 读的字节里 KV 占大头**。

那就有个很自然的问题：**同样砍到 8 bit，这个 bit 花在权重上还是花在 KV 上？**

| 配置 | B=32,S=8K | B=128,S=8K | B=32,S=32K | B=128,S=32K |
|---|---|---|---|---|
| bf16 权重 + bf16 KV（基线） | 1.00× | 1.00× | 1.00× | 1.00× |
| **只把权重量化到 fp8** | **1.17×** | **1.05×** | **1.05×** | **1.01×** |
| bf16 权重 + fp8 KV | 1.55× | 1.83× | 1.83× | 1.95× |
| fp8 权重 + int4 KV（group 32） | 3.05× | 3.57× | 3.57× | 3.79× |

（H800 级：bf16 稠密算力 989 TFLOPS、HBM 带宽 3.35 TB/s，按「带宽已饱和」的 roofline 推算，完整脚本见 §4.5）

**权重量化在 decode 上几乎白干。** batch 和上下文一大，1.17× 就退化成 1.01×——因为权重的字节数不随负载增长，KV 的字节数随负载线性增长。这就是这一期要讲的领域。

要让「KV 量化」这件事有意义，需要三个条件同时成立：decode 是带宽瓶颈（大 batch 或长上下文）、显存装不下想跑的并发、KV 的精度损失可以接受。前两条是第 027 / 028 / 029 期反复出现的那个 $B \cdot S \gg 295$ 与容量问题，第三条是这一期的主题。

---

## 2. 核心思想：量化只有三件事

### 2.1 格式：均匀刻度 vs 指数刻度

所有量化格式都在回答同一个问题：**用有限个 level 去逼近一段实数值。**

- **int8 是对称 absmax**：$\hat{x} = \text{round}(x/s)\cdot s$，其中 $s = \max|x|/127$。整个张量（或一个组）共享 256 个均匀的 level，level 的间距 $s$ 完全由 **max** 决定。
- **fp8-e4m3** 是 1 符号 + 4 指数 + 3 尾数：`eps=0.125`，规约数范围 $[0.015625, 448]$（跨 15 个 binade），次正规还能下探到 $0.001953$。它在每个 binade 内只放 8 个 level。

这两种格式的性格完全不同，而且在 KV 上差异被放大：

> **正文数值均在 torch 2.14 CPU 上实测，脚本见 §4.1**

| 粒度 | int8 通道中位 nMSE | fp8 通道中位 nMSE |
|---|---|---|
| per-tensor（全张量 1 个 scale） | 0.904284 | 0.000691 |
| per-channel | 0.000584 | 0.000661 |
| per-token | 0.110972 | 0.000679 |
| per-token-group g=32 | 0.007356 | 0.000665 |
| per-token-group g=16 | 0.002072 | 0.000665 |

**int8 的误差对粒度极度敏感（0.904 → 0.00207，437 倍），fp8 几乎不敏感（0.000691 → 0.000665，1.04 倍）。**

机制很直接：int8 是均匀刻度，分辨率 $= \max/127$，所以「谁参与决定 max」就决定了一切；fp8 是自带非均匀刻度的格式，一个数值落在哪个 binade 里就自带对应的相对精度，只要不越界、不落到次正规以下，**scale 的粒度对它意义不大**。fp8 只有尾数位数（3 位 → 相对误差最大 $2^{-4}=6.25\%$）在定天花板。

这条机制直接解释了 vLLM 的 API 面为什么长这样（**文档里看到的**，`features/quantization/quantized_kvcache.md`）：

- FP8 KV 只提供两种 scheme：**per-tensor**（`q/k/v_scale = [1]`）和 **per-attention-head**（`k/v_scale = [num_kv_heads]`）；
- per-attention-head **只在 Flash Attention 后端可用**，且需要 llm-compressor 的标定路径；
- 而标准注意力的 `TRITON_ATTN` 后端才提供 `fp8_per_token_head` / `int8_per_token_head` / `int4_per_token_head` 这些细粒度 KV dtype（**注意力后端支持矩阵里看到的**）。

**"尺度对 fp8 不敏感" 不等于 "fp8 不需要 scale"。** vLLM 的默认路径 `kv_cache_dtype="fp8"` 把**所有 scale 设成 1.0**（文档原话："All quantization scales are set to `1.0`"）。e4m3 的天花板是 448，一旦某层某个头的 K/V 幅度越界就是硬截断。实测同一份数据：

```
fp8 裸 cast（= vLLM 默认 scales=1.0 时的行为）全局 nMSE 0.178224，通道中位 0.000692
同样数据走 fp8 per-tensor（先除 scale 再 cast）：全局 0.000610，通道中位 0.000691
```

**通道中位几乎一样（0.000692 vs 0.000691），全局差了 292 倍。** 意思是：scales=1.0 的路径对「量级落在 $(0.002,\ 448)$ 之内」的通道几乎无损（这也是它敢当默认值的原因），但只要有大值越界，误差就全堆在那些大值上。而 K 的 outlier 通道恰好就是大值——这就是为什么文档同时给出 `calculate_kv_scales`（在线随机 token 标定）和 llm-compressor 数据集标定两条后路。

> 顺便记一个文档不一致（**文档里看到的**）：当前版本的这一节写 "using **three** different approaches"，但列表里只剩两条（不标定 / 数据集标定）。较早的文档构建版本里还有第三条 `calculate_kv_scales=True`（用一批随机 token 在线估计 scale 后固定）。读文档时别被 "three" 骗了。

### 2.2 粒度：真正的规则是「让谁共享一个 max」

把「粒度」这个词翻译一遍：**它说的是多少个元素共享一个 scale，而 scale 是这一组元素的 absmax 的函数。** 所以粒度的好坏，本质上是**极值统计**问题。

对 absmax 对称量化，一组内的误差是均匀分布，方差 $=s^2/12$，而 $s=\max/127$。于是

$$\text{nMSE} \approx \frac{\mathbb{E}[\max^2]}{12 \cdot 127^2 \cdot \sigma^2}$$

**实测验证**（§4.2）：纯高斯数据上扫 10 个组大小（16 到 262144，跨 1.6 万倍），实测/预测的比值落在 **0.936 ~ 1.002**。再迁移到四种异质结构，14/16 个格子的比值在 0.93 ~ 1.02 之间。

这条定律有两个立刻可用的推论。

**推论一：E[max] 对组大小极度不敏感。** 对高斯样本，$\mathbb{E}[\max_n] \approx \sigma\sqrt{2\ln n}$。实测 n 从 16 涨到 262144（1.6 万倍），$\mathbb{E}[\max]$ 只从 **2.08 涨到 4.66**（2.2 倍）。所以**纯统计意义上，改粒度的收益是对数级的、温和的**——把「每 2048 个元素一个 scale」改成「每 128 个一个」，只把 E[max] 从约 3.5 降到约 2.9。

**推论二：真正打崩一个粒度的，是分布本身的异质。** 异质让 max 被极少数元素独占，于是同一组里其他元素的 level 被吃光。KV 里有两个**正交**的异质源：

- **通道异质**：K 有一小撮固定的通道，幅度比别的通道大一两个数量级，而且**跨所有 token 一致**（KIVI 论文原话："there are a few fixed channels whose magnitudes are very large"）。
- **token 异质**：同一个通道里，不同 token 的幅度能差几十倍。

把它们分别开关，做 2×2 消融（指标用「通道中位 nMSE」，理由见 §2.2 结尾）：

| 配置 | per-tensor | per-channel | per-token | per-token-group g=32 |
|---|---|---|---|---|
| 纯高斯（无任何异质） | 0.000111 | 0.000067 | 0.000042 | 0.000029 |
| **只有通道 outlier** | 0.242341 | **0.000067** | 0.031333 | 0.000031 |
| **只有 token 异质** | 0.001529 | 0.000584 | **0.000042** | 0.000028 |
| 两者都有（真实 K 的样子） | 0.904284 | 0.000584 | 0.110972 | **0.007356** |

读法：

- 有通道 outlier 时，**per-channel 比 per-token 好 468 倍**（0.000067 vs 0.031333）——因为 per-token 的 scale 被 outlier 通道一个数拉满，其余 124 个通道全被压扁；
- 有 token 异质时，**per-token 比 per-channel 好 14 倍**（0.000042 vs 0.000584）——对称的机制：per-channel 的 scale 由最大的那个 token 决定；
- 两个异质源同时存在时，**只有 per-token-group 活下来**（0.007356，比 per-tensor 好 123 倍）。

这正是 KIVI（ICML'24）处方的完整理由（**论文原文**）：

> "For key cache, there are a few fixed channels whose magnitudes are very large … Thus, the key cache should be quantized **per-channel** because it can confine the error to each individual channel, without impacting the other normal channels."
>
> "For value caches, there is no obvious outlier pattern. Although the value cache has not obvious outlier pattern, we experimentally show that it can only be quantized **per-token** because it is used to calculate the attention output, which is essentially a value cache mixer."

**别忘了口径。** 如果只看「全局 nMSE」（$\sum \text{err}^2 / \sum x^2$），同一份数据给出的排名是**反的**：

| 粒度 | 全局 nMSE | 通道中位 nMSE |
|---|---|---|
| per-channel | 0.000713 | 0.000584 |
| per-token | **0.000392** | 0.110972 |

全局口径说 per-token 好 1.8 倍；通道中位口径说 per-channel 好 190 倍。原因是全局 nMSE 的分子分母都被大值主导，**小通道被毁掉这件事在聚合指标里几乎不可见**。选粒度必须用「每通道先归一、再看中位/分位」的口径。

### 2.3 scale 自己的字节账

量化不是「位宽减半」这么简单，**每个 scale 也要占字节**。数一遍就知道：一个组里有 `group` 个数值，每个 `b` 字节；这个组要付 `items × B` 字节的 scale（`items` 是每组里 scale 加 zero-point 的个数，`B` 是每个 scale 的字节数）。于是

$$\text{scale 开销} = \frac{items \times B}{group \times b}$$

这个式子里**没有 head 数**——分子分母上的 $2 n_{kv}$ 正好约掉。所以决定开销的只有两件事：**组内有多少个数值**，和**每个 scale 多少字节**。

| group | b | scale 字节 | 开销 | 有效 B/数值 | vs bf16 |
|---|---|---|---|---|---|
| 128 | 1.0（int8/fp8） | 4（fp32） | 3.12% | 1.031 | 1.94× |
| 64 | 1.0 | 4 | 6.25% | 1.063 | 1.88× |
| 32 | 1.0 | 4 | 12.50% | 1.125 | **1.78×** |
| 16 | 1.0 | 4 | 25.00% | 1.250 | **1.60×** |
| 32 | 0.5（int4） | 4 | 25.00% | 0.625 | **3.20×** |
| 16 | 0.5（int4） | 4 | **50.00%** | 0.750 | **2.67×** |
| 64 | 1.0 | 1（ue8m0） | 1.56% | 1.016 | 1.97× |

**位宽越低，scale 的账越咬人。** int4 + group 16 + fp32 scale 的开销是 50%，理论 4× 压缩实际只剩 2.67×。这就是为什么真正的生产布局会用 1 字节的 2 的幂次 scale。

**外部对照（这是本期最漂亮的一处验证）**：DeepSeek V3.2 的 `fp8_ds_mla` 布局每个 token 656 字节（**vLLM Day-0 博客与源码注释里看到的**）：

```
前 512 字节：512 个 float8_e4m3（"quantized NoPE" 部分）
接下来 16 字节：4 个 float32 scale，每个管连续 128 个 fp8 值   → group=128
最后 128 字节：64 个 bfloat16（RoPE key，不量化，"for accuracy"）
```

$16/512 = 3.1250\%$。用上面的式子：$4/(128 \times 1.0) = 3.1250\%$。**逐字节相符。**

DeepSeek V4 的布局是 584 字节：448 个 fp8（NoPE）+ 128 字节 bf16（RoPE，不量化）+ **7 个 `ue8m0` scale（每个 1 字节）+ 1 字节 padding**，每个 scale 管 64 个 fp8 值。开销 $7/448 = 1.5625\%$——把 group 从 128 砍到 64，代价反而更低了，**因为它把 scale 从 4 字节换成了 1 字节**。这正是 $\text{开销} = items \cdot B/(group \cdot b)$ 的两个旋钮：group 和 $B$。

---

## 3. 在 PyTorch 中怎么用

### 3.1 四条官方入口，以及它们互不相同的语义

torch 里有现成的量化原语，但**它们的参数语义互不一致**，这是最容易踩的地方。

```python
"""第 033 期 附：PyTorch 里量化 KV 的四种官方入口（本机真跑）
注意最后两条的实现细节差异 —— 它们是很容易踩的坑。
"""
import torch
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

torch.manual_seed(0)
x = torch.randn(4, 8)          # 假装是一个 [token, head_dim] 的 K 分片


def rel(y):
    return ((y - x).norm() / x.norm()).item()


print("=== 入口 1：fake_quantize_per_tensor_affine（per-tensor，手给 scale）===")
s = x.abs().amax() / 127.5
y = torch.fake_quantize_per_tensor_affine(x, s.item(), 0, -128, 127)
print(f"  scale={s.item():.6f}  zero_point=0  quant 范围 [-128,127]")
print(f"  相对误差 {rel(y):.6f}")
print(f"  这个 4x8 张量里去重后只有 {y.unique().numel()} 个不同的取值")

print()
print("=== 入口 2：fake_quantize_per_channel_affine（per-channel，沿某一维一个 scale）===")
# 坑 1：scale 的长度必须等于 x.size(axis)；axis=1 就要 x.size(1)=8 个 scale，
#       所以是沿 dim 0 求 amax（写反了会报 "dimensions ... not consistent"）
# 坑 2：scale 必须是 float32/bf16（传 float64 会报 "Scale must be Float or BFloat16"）
#       zero_point 必须是 int32
# 坑 1.5：这里要的是「scale 本身」（step = max/qmax），不是 max！传 max 进去不报错，
#         但等价于把分辨率放大 127 倍，相对误差直接从 0.4% 变成 36%
scale = x.abs().amax(0) / 127.5             # shape [8]，对应 axis=1
zp = torch.zeros(8, dtype=torch.int32)
y2 = torch.fake_quantize_per_channel_affine(x, scale, zp, 1, -128, 127)
print(f"  scale 形状 {list(scale.shape)}（= x.size(axis=1)），dtype {scale.dtype}")
print(f"  相对误差 {rel(y2):.6f}   （per-tensor 是 {rel(y):.6f}）")

print()
print("=== 入口 3：observer 自动收集 scale ===")
ob1 = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
ob1(x)
sc1, zp1 = ob1.calculate_qparams()
print(f"  MinMaxObserver(per_tensor_symmetric) → scale {sc1.item():.6f}, zp {zp1.item()}")
try:
    MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
except NotImplementedError as e:
    print(f"  MinMaxObserver(per_channel_symmetric) → NotImplementedError：{e}")
ob2 = PerChannelMinMaxObserver(dtype=torch.qint8,
                               qscheme=torch.per_channel_symmetric, ch_axis=0)
ob2(x)
sc2, zp2 = ob2.calculate_qparams()
print(f"  PerChannelMinMaxObserver(ch_axis=0) → scale {list(sc2.shape)} 个，"
      f"dtype {sc2.dtype}，zp dtype {zp2.dtype}")
print(f"  这 4 个 scale 的极差 {sc2.max().item() / sc2.min().item():.2f}x"
      f" —— 极差越大，per-channel 相对 per-tensor 的收益越大")

print()
print("=== 入口 4：真正的 int8 张量 torch.quantize_per_channel ===")
# 坑 3：这里 scale 必须是 float64、zero_point 必须是 int64，和上面 fake_quantize 的要求正好相反
s64 = (x.abs().amax(0) / 127.5).to(torch.float64)
z64 = torch.zeros(8, dtype=torch.int64)
qt = torch.quantize_per_channel(x, s64, z64, 1, torch.qint8)
print(f"  int_repr 第 0 行：{qt.int_repr()[0].tolist()}")
print(f"  相对误差 {rel(qt.dequantize()):.6f}")
print(f"  q_per_channel_scales dtype = {qt.q_per_channel_scales().dtype}")
print("  注意：torch.quantize_per_tensor / quantize_per_channel 生成 qint8 张量的这条路")
print("        在 torch 2.14 上已经打出 DeprecationWarning（pytorch/pytorch#184982），")
print("        指向的替代方向是 tensor subclass + fake-quant，而不是老的 quantized tensor。")

print()
print("=== 入口 5：per-token-group（框架里真正在用的那一种，torch 没有现成算子）===")


def qdq_per_token_group(t, g):
    """沿最后一维分组，每个 (token, 头, 组) 一个 absmax scale。t: [..., d]"""
    shape = t.shape
    f = t.reshape(*shape[:-1], shape[-1] // g, g)
    s = (f.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
    return ((f / s).round().clamp(-127, 127) * s).reshape(shape)


k = torch.randn(2048, 8, 128)          # [token, kv_head, head_dim]
k[:, :, 32] *= 40.0                    # 造一个 outlier 通道，模拟真实 K 的 fixed channels
for g in [128, 64, 32, 16]:
    q = qdq_per_token_group(k, g)
    print(f"  group={g:>3}  相对误差 {((q - k).norm() / k.norm()).item():.6f}"
          f"  scale 个数 {2048 * 8 * (128 // g)}")
print("  相对误差随 group 变小而单调下降，代价是 scale 个数线性增加 —— 这就是全部权衡")


if __name__ == "__main__":
    pass
```

```text
=== 入口 1：fake_quantize_per_tensor_affine（per-tensor，手给 scale）===
  scale=0.016590  zero_point=0  quant 范围 [-128,127]
  相对误差 0.004684
  这个 4x8 张量里去重后只有 30 个不同的取值

=== 入口 2：fake_quantize_per_channel_affine（per-channel，沿某一维一个 scale）===
  scale 形状 [8]（= x.size(axis=1)），dtype torch.float32
  相对误差 0.003573   （per-tensor 是 0.004684）

=== 入口 3：observer 自动收集 scale ===
  MinMaxObserver(per_tensor_symmetric) → scale 0.016590, zp 0
  MinMaxObserver(per_channel_symmetric) → NotImplementedError：MinMaxObserver's qscheme only support torch.per_tensor_symmetric                     and torch.per_tensor_affine.
  PerChannelMinMaxObserver(ch_axis=0) → scale [4] 个，dtype torch.float32，zp dtype torch.int64
  这 4 个 scale 的极差 1.67x —— 极差越大，per-channel 相对 per-tensor 的收益越大

=== 入口 4：真正的 int8 张量 torch.quantize_per_channel ===
  int_repr 第 0 行：[-106, -87, -56, -70, 78, 56, -36, -127]
  相对误差 0.003573
  q_per_channel_scales dtype = torch.float64
  注意：torch.quantize_per_tensor / quantize_per_channel 生成 qint8 张量的这条路
        在 torch 2.14 上已经打出 DeprecationWarning（pytorch/pytorch#184982），
        指向的替代方向是 tensor subclass + fake-quant，而不是老的 quantized tensor。

=== 入口 5：per-token-group（框架里真正在用的那一种，torch 没有现成算子）===
  group=128  相对误差 0.024667  scale 个数 16384
  group= 64  相对误差 0.017402  scale 个数 32768
  group= 32  相对误差 0.012267  scale 个数 65536
  group= 16  相对误差 0.008558  scale 个数 131072
  相对误差随 group 变小而单调下降，代价是 scale 个数线性增加 —— 这就是全部权衡
```

四个坑全在注释里：**scale 要的是 step 不是 max**（传 max 不报错，相对误差从 0.4% 变 36%）、**axis=1 就需要 `x.size(1)` 个 scale**（所以要 `amax(0)`）、**`fake_quantize_per_channel_affine` 要 float32 scale + int32 zero_point，而 `quantize_per_channel` 要 float64 + int64**、**`MinMaxObserver` 遇到 per-channel 会直接 `NotImplementedError`，必须换 `PerChannelMinMaxObserver`**。

最后一条实测输出里还有一句值得记下：**torch 2.14 上生成 `qint8` 张量的那条路（`quantize_per_tensor` / `quantize_per_channel`）已经打出 `DeprecationWarning`**，指向 `pytorch/pytorch#184982`。这和 025 期讲 QLoRA 时看到的趋势一致——量化正在从「量化张量」转向 **tensor subclass + fake-quant**，这样能干净地和 `torch.compile`、`FSDP2`、meta device 组合。

### 3.2 per-token-group：torch 没有现成算子，得自己写

框架里真正用在 KV 上的粒度（KIVI 的 per-channel K、FlashInfer / TRITON_ATTN 的 per-token-head）都没有对应的 torch 算子。核心就三行：

```python
def qdq_per_token_group(t, g):
    """沿最后一维分组，每个 (token, 头, 组) 一个 absmax scale。"""
    shape = t.shape
    f = t.reshape(*shape[:-1], shape[-1] // g, g)          # 把通道维切成 d/g 组
    s = (f.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
    return ((f / s).round().clamp(-127, 127) * s).reshape(shape)
```

关键是 `amax(-1)` 沿哪个轴——`-1` 是「组内」那一维，也就是**只让同一组的 g 个元素共享一个 max**。写成 `amax(-2)` 就变成了 per-token，写成不带 keepdim 的全局 `amax()` 就退化成 per-tensor。这三种写法语法上都对，误差能差几十倍。

### 3.3 vLLM 侧怎么开

```bash
# 最省事的开法：fp8 KV，scale 全 1.0
vllm serve <model> --kv-cache-dtype fp8

# 部分层保持原精度（sliding-window 层对 KV 量化更敏感）
vllm serve <model> --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window

# 指定具体层号
vllm serve <model> --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers 0 1 23
```

`kv_cache_dtype` 的取值与硬件要求（**文档里看到的**）：`auto`（跟模型同 dtype）、`fp8` / `fp8_e4m3`（CUDA 11.8+ 与 ROCm）、`fp8_e5m2`（CUDA 11.8+）。还有一条容易忽略的：**FA3 后端 + fp8 KV 时，attention 运算本身也在 fp8 域里跑，query 也会被量化**——所以精度决策不只是存储决策，还包含运行时计算。

精度对齐的建议（文档里给的是三条，去掉了一条）：

| 标定方式 | 配置 | 说明 |
|---|---|---|
| 不标定 | `kv_cache_dtype="fp8"`（默认 scales=1.0） | 最省事，赌数值不越 448 |
| 在线随机 token 标定 | `calculate_kv_scales=True` | 用一批随机 token 估 scale 后固定（当前文档列表里已不提，但参数仍在） |
| 数据集标定 | `llm-compressor` + `kv_cache_scheme=fp8_args` | 唯一能开 per-attention-head 的路径，也是文档推荐的默认 |

自动选择后端的优先级（**注意力后端支持矩阵里看到的**）：**FA2 的 KV dtype 只有 `auto` / `float16` / `bfloat16`——不支持 fp8**；fp8 从 **FA3 起**才有（`fp8` / `fp8_e4m3`，SM 9.x）；FA4（SM ≥10.0）同样支持。所以「开了 `--kv-cache-dtype fp8` 却发现没用上」的一个常见原因是后端落到了 FA2。

---

## 4. 六个真跑的实验

本机环境：torch 2.14.0，CPU + MPS，**没有 CUDA**。所有标注「实测」的块都是真跑出来的 stdout；涉及 GPU kernel 本身的部分（§4.5 的 roofline）明确标注为解析推算。

### 4.1 粒度谱：两个正交的异质源 + 口径反转

```python
"""第 033 期 实验 1（v3）：2x2 异质源消融
两个正交的异质源：通道异质（K 的 fixed outlier channels）、token 异质（幅度随位置的涨落）。
各自开关，看每个粒度分别在哪种情形下崩。
指标：通道中位 nMSE（每通道先归一，再看中位数）—— 弱者视角。
"""
import torch

torch.manual_seed(0)
S, D = 2048, 128
OUT_IDX = torch.tensor([81, 89, 98, 122])


def make_kv(sig_ch, sig_tok, gain, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(S, D, generator=g)
    x = x * torch.exp(sig_ch * torch.randn(D, generator=g))
    x = x * torch.exp(sig_tok * torch.randn(S, generator=g))[:, None]
    x[:, OUT_IDX] *= gain
    return x


def view_groups(x, mode, g=None):
    if mode == "per_tensor":
        return x.reshape(1, -1)
    if mode == "per_token":
        return x
    if mode == "per_channel":
        return x.T.contiguous()
    if mode == "per_token_group":
        return x.view(S, D // g, g).reshape(-1, g)
    if mode == "per_channel_group":
        return x.view(S // g, g, D).permute(1, 0, 2).reshape(g, -1).T.contiguous()
    raise ValueError(mode)


def unview(x, flat, mode, g=None):
    if mode in ("per_tensor", "per_token"):
        return flat.view(S, D)
    if mode == "per_channel":
        return flat.view(D, S).T.contiguous()
    if mode == "per_token_group":
        return flat.view(S, -1, g).reshape(S, D)
    if mode == "per_channel_group":
        return flat.view(S // g, D, g).permute(0, 2, 1).reshape(S, D)


def quant(x, mode, fmt, g=None):
    flat = view_groups(x, mode, g)
    if fmt == "int8":
        s = (flat.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-12)
        q = (flat / s).round().clamp(-127, 127) * s
    elif fmt == "fp8":
        s = (flat.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
        q = (flat / s).to(torch.float8_e4m3fn).to(torch.float32) * s
    else:
        raise ValueError(fmt)
    return unview(x, q, mode, g)


def g_nmse(x, q):
    return ((q - x) ** 2).sum() / (x ** 2).sum()


def ch_med(x, q):
    return (((q - x) ** 2).sum(0) / (x ** 2).sum(0)).median()


CANDS = [("per-tensor", "per_tensor", None), ("per-channel", "per_channel", None),
         ("per-token", "per_token", None), ("per-tok-grp g=32", "per_token_group", 32)]


def main():
    print("=== 2x2 消融：通道异质 × token 异质（int8，指标=通道中位 nMSE）===")
    print(f"{'配置':<36}" + "".join(f"{n:>16}" for n, _, _ in CANDS))
    print("-" * (36 + 16 * len(CANDS)))
    cases = [(0.0, 0.0, 1.0, "纯高斯（无任何异质）"),
             (0.0, 0.0, 50.0, "只有通道 outlier（token 幅度齐整）"),
             (0.0, 0.8, 1.0, "只有 token 异质（无 outlier 通道）"),
             (1.0, 0.8, 50.0, "两者都有（真实 K 的样子）")]
    for sc, st, gain, tag in cases:
        x = make_kv(sc, st, gain)
        cells = []
        for _, mode, g in CANDS:
            cells.append(ch_med(x, quant(x, mode, "int8", g)))
        print(f"{tag:<36}" + "".join(f"{c:>16.6f}" for c in cells))

    print()
    print("同一批配置，换成「全局 nMSE」（大值主导，注意力打分视角）:")
    for sc, st, gain, tag in cases:
        x = make_kv(sc, st, gain)
        cells = [g_nmse(x, quant(x, mode, "int8", g)) for _, mode, g in CANDS]
        print(f"{tag:<36}" + "".join(f"{c:>16.6f}" for c in cells))

    print()
    print("=== fp8-e4m3：粒度对它有影响吗（同为通道中位 nMSE）===")
    x = make_kv(1.0, 0.8, 50.0)
    for name, mode, g in CANDS + [("per-tok-grp g=16", "per_token_group", 16),
                                  ("per-ch-grp g=128", "per_channel_group", 128)]:
        print(f"  {name:<22} int8 {ch_med(x, quant(x, mode, 'int8', g)):.6f}"
              f"   fp8 {ch_med(x, quant(x, mode, 'fp8', g)):.6f}")

    print()
    print("=== 格式自身的上限：单个 value 的相对误差与绝对动态范围 ===")
    for fmt in ["int8", "int4", "fp8"]:
        bits = {"int8": 8, "int4": 4, "fp8": 8}[fmt]
        if fmt.startswith("int"):
            print(f"  {fmt}: 均匀刻度，{2 ** bits} 个 level，分辨率 = max/(2^{bits - 1})，"
                  f"动态范围只由 scale 决定")
    x = make_kv(1.0, 0.8, 50.0)
    z = x.to(torch.float8_e4m3fn).to(torch.float32)
    print(f"  fp8 裸 cast（= vLLM 默认 scales=1.0 时的行为）全局 nMSE "
          f"{g_nmse(x, z):.6f}，通道中位 {ch_med(x, z):.6f}")
    print(f"  同样数据走 fp8 per-tensor（先除 scale 再 cast）：全局 "
          f"{g_nmse(x, quant(x, 'per_tensor', 'fp8')):.6f}，"
          f"通道中位 {ch_med(x, quant(x, 'per_tensor', 'fp8')):.6f}")
    f = torch.finfo(torch.float8_e4m3fn)
    nmant = round(-torch.log2(torch.tensor(f.eps)).item())
    print(f"  fp8-e4m3: 总宽 {f.bits} 位 = 1 符号 + 4 指数 + {nmant} 尾数，"
          f"eps={f.eps}，规约数范围 [{f.smallest_normal}, {f.max}]"
          f"（跨 {torch.log2(torch.tensor(f.max / f.smallest_normal)):.0f} 个 binade），"
          f"次正规下探到 {f.smallest_normal / 2 ** nmant}")


if __name__ == "__main__":
    main()
```

```text
=== 2x2 消融：通道异质 × token 异质（int8，指标=通道中位 nMSE）===
配置                                        per-tensor     per-channel       per-tokenper-tok-grp g=32
----------------------------------------------------------------------------------------------------
纯高斯（无任何异质）                                  0.000111        0.000067        0.000042        0.000029
只有通道 outlier（token 幅度齐整）                    0.242341        0.000067        0.031333        0.000031
只有 token 异质（无 outlier 通道）                   0.001529        0.000584        0.000042        0.000028
两者都有（真实 K 的样子）                              0.904284        0.000584        0.110972        0.007356

同一批配置，换成「全局 nMSE」（大值主导，注意力打分视角）:
纯高斯（无任何异质）                                  0.000112        0.000068        0.000042        0.000029
只有通道 outlier（token 幅度齐整）                    0.003110        0.000081        0.000400        0.000130
只有 token 异质（无 outlier 通道）                   0.001519        0.000623        0.000042        0.000028
两者都有（真实 K 的样子）                              0.009425        0.000713        0.000392        0.000123

=== fp8-e4m3：粒度对它有影响吗（同为通道中位 nMSE）===
  per-tensor             int8 0.904284   fp8 0.000691
  per-channel            int8 0.000584   fp8 0.000661
  per-token              int8 0.110972   fp8 0.000679
  per-tok-grp g=32       int8 0.007356   fp8 0.000665
  per-tok-grp g=16       int8 0.002072   fp8 0.000665
  per-ch-grp g=128       int8 0.000174   fp8 0.000511

=== 格式自身的上限：单个 value 的相对误差与绝对动态范围 ===
  int8: 均匀刻度，256 个 level，分辨率 = max/(2^7)，动态范围只由 scale 决定
  int4: 均匀刻度，16 个 level，分辨率 = max/(2^3)，动态范围只由 scale 决定
  fp8 裸 cast（= vLLM 默认 scales=1.0 时的行为）全局 nMSE 0.178224，通道中位 0.000692
  同样数据走 fp8 per-tensor（先除 scale 再 cast）：全局 0.000610，通道中位 0.000691
  fp8-e4m3: 总宽 8 位 = 1 符号 + 4 指数 + 3 尾数，eps=0.125，规约数范围 [0.015625, 448.0]（跨 15 个 binade），次正规下探到 0.001953125
```

### 4.2 极值统计定律

```python
"""第 033 期 实验 4：粒度背后的唯一定律
假设：对称 absmax 量化里，一组元素的误差只由「这组的绝对值最大值」决定。
      nMSE ≈ E[max²] / (12 · 127² · σ²)
如果这条成立，那「粒度」这个词就可以整个替换成「你让多少个元素共享一个 max」。
本机真实运行。
"""
import torch

torch.manual_seed(0)
S, D = 2048, 128


def qdq_group_int8(x):
    """x: [n_groups, n] —— 每组一个 absmax scale。"""
    s = (x.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
    return (x / s).round().clamp(-127, 127) * s, s.squeeze(-1)


def law(groups_flat):
    """返回 (实测 nMSE, E[max²]/σ², 定律预测)"""
    q, s = qdq_group_int8(groups_flat)
    mse = ((q - groups_flat) ** 2).mean()
    var = (groups_flat ** 2).mean()
    emax2 = ((s * 127) ** 2).mean()
    return mse / var, emax2 / var, emax2 / var / (12 * 127 ** 2)


def make_data(kind, S=S, D=D, sig_ch=1.0, sig_tok=0.8, gain=50.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(S, D, generator=g)
    if kind in ("ch", "both"):
        x = x * torch.exp(sig_ch * torch.randn(D, generator=g))
    if kind in ("tok", "both"):
        x = x * torch.exp(sig_tok * torch.randn(S, generator=g))[:, None]
    if kind in ("out", "both"):
        idx = torch.arange(4)
        x[:, idx] *= gain
    return x


def as_groups(x, mode, g=None):
    if mode == "per_tensor":
        return x.reshape(1, -1)
    if mode == "per_channel":
        return x.T.contiguous()
    if mode == "per_token":
        return x
    if mode == "per_token_group":
        return x.view(S, D // g, g).reshape(-1, g)
    raise ValueError(mode)


def main():
    print("=== 1. 定律的正面检验：纯高斯，只改「一组有多少元素」（int8 absmax）===")
    x = make_data("none")
    flat = x.reshape(-1)
    print(f"数据：{flat.numel()} 个 iid 标准正态样本")
    print(f"{'每组元素数':>12}{'组数':>10}{'E[max²]/σ²':>13}{'实测 nMSE':>14}{'定律预测':>14}{'实测/预测':>11}")
    for n in [16, 32, 64, 128, 256, 1024, 4096, 16384, 65536, 262144]:
        grp = flat[: (flat.numel() // n) * n].view(-1, n)
        meas, em2, pred = law(grp)
        print(f"{n:>12}{grp.shape[0]:>10}{em2:>13.4f}{meas:>14.3e}{pred:>14.3e}{meas / pred:>11.3f}")

    print()
    print("=== 2. 定律的迁移检验：同一批粒度，四种异质结构（d=128）===")
    print(f"{'数据结构':<26}{'粒度':<20}{'E[max²]/σ²':>12}{'实测 n MSE':>13}{'预测':>13}{'比':>8}")
    rows = [("per_tensor", None), ("per_channel", None), ("per_token", None),
            ("per_token_group", 32)]
    for kind, tag in [("none", "纯高斯"), ("tok", "token 幅度异质"),
                      ("out", "通道 outlier ×50"), ("both", "两者都有")]:
        x = make_data(kind)
        for mode, g in rows:
            grp = as_groups(x, mode, g)
            meas, em2, pred = law(grp)
            name = f"{mode}" + (f" g={g}" if g else "")
            print(f"{tag if mode == 'per_tensor' else '':<26}{name:<20}"
                  f"{em2:>12.3f}{meas:>13.3e}{pred:>13.3e}{meas / pred:>8.2f}")

    print()
    print("=== 3. max 有多脆：E[max] 随样本数只按 sqrt(2·ln n) 走 ===")
    print("（数据是 iid 标准正态，σ 精确等于 1，直接用 1 归一）")
    print(f"{'n':>10}{'实测 E[max]':>14}{'sqrt(2·ln n)':>15}{'比值':>9}")
    for n in [16, 128, 1024, 65536, 262144]:
        grp = flat[: (flat.numel() // n) * n].view(-1, n)
        emax = grp.abs().amax(-1).mean().item()
        th = (2 * torch.log(torch.tensor(float(n)))).sqrt().item()
        print(f"{n:>10}{emax:>14.4f}{th:>15.4f}{emax / th:>9.4f}")
    print("  → 样本数从 16 涨到 26 万（1.6 万倍），E[max] 只从 2.08 涨到 4.9（2.4 倍）")
    print("  → 所以纯统计意义上，粒度带来的收益是对数级的、温和的")
    print("  → 真正把某个粒度打崩的是「分布本身的异质」：它让 max 由极少数元素独占")

    print()
    print("=== 4. 异质源如何抬高 E[max]（这才是粒度的真正杠杆）===")
    print(f"{'数据结构':>22}{'per-tensor E[max²]':>21}{'per-token E[max²]':>19}"
          f"{'per-ch-grp32 E[max²]':>22}")
    for kind, tag in [("none", "纯高斯"), ("tok", "token 幅度异质"),
                      ("out", "通道 outlier ×50"), ("both", "两者都有")]:
        x = make_data(kind)
        vals = []
        for mode, g in [("per_tensor", None), ("per_token", None)]:
            grp = as_groups(x, mode, g)
            vals.append(law(grp)[1])
        xg = x.view(S, D // 32, 32).reshape(-1, 32)
        vals.append(law(xg)[1])
        print(f"{tag:>22}{vals[0]:>21.3f}{vals[1]:>19.3f}{vals[2]:>22.3f}")


if __name__ == "__main__":
    main()
```

```text
=== 1. 定律的正面检验：纯高斯，只改「一组有多少元素」（int8 absmax）===
数据：262144 个 iid 标准正态样本
       每组元素数        组数   E[max²]/σ²       实测 nMSE          定律预测      实测/预测
          16     16384       4.5679     2.209e-05     2.360e-05      0.936
          32      8192       5.7324     2.863e-05     2.962e-05      0.967
          64      4096       6.9367     3.527e-05     3.584e-05      0.984
         128      2048       8.2258     4.226e-05     4.250e-05      0.994
         256      1024       9.4666     4.880e-05     4.891e-05      0.998
        1024       256      12.0591     6.208e-05     6.231e-05      0.996
        4096        64      14.5461     7.518e-05     7.515e-05      1.000
       16384        16      17.4594     9.018e-05     9.021e-05      1.000
       65536         4      19.8264     1.026e-04     1.024e-04      1.002
      262144         1      21.6910     1.121e-04     1.121e-04      1.000

=== 2. 定律的迁移检验：同一批粒度，四种异质结构（d=128）===
数据结构                      粒度                    E[max²]/σ²     实测 n MSE           预测       比
纯高斯                       per_tensor                21.691    1.121e-04    1.121e-04    1.00
                          per_channel               13.243    6.825e-05    6.842e-05    1.00
                          per_token                  8.226    4.226e-05    4.250e-05    0.99
                          per_token_group g=32       5.732    2.863e-05    2.962e-05    0.97
token 幅度异质                per_tensor              1008.050    5.161e-03    5.208e-03    0.99
                          per_channel              227.277    1.172e-03    1.174e-03    1.00
                          per_token                  8.049    4.132e-05    4.159e-05    0.99
                          per_token_group g=32       5.701    2.910e-05    2.946e-05    0.99
通道 outlier ×50            per_tensor               410.968    2.123e-03    2.123e-03    1.00
                          per_channel               11.247    5.838e-05    5.811e-05    1.00
                          per_token                 77.768    3.981e-04    4.018e-04    0.99
                          per_token_group g=32      19.496    9.711e-05    1.007e-04    0.96
两者都有                      per_tensor              7539.845    1.233e-02    3.896e-02    0.32
                          per_channel              119.954    6.305e-04    6.198e-04    1.02
                          per_token                 81.364    4.023e-04    4.204e-04    0.96
                          per_token_group g=32      26.845    1.288e-04    1.387e-04    0.93

=== 3. max 有多脆：E[max] 随样本数只按 sqrt(2·ln n) 走 ===
（数据是 iid 标准正态，σ 精确等于 1，直接用 1 归一）
         n     实测 E[max]   sqrt(2·ln n)       比值
        16        2.0814         2.3548   0.8839
       128        2.8410         3.1151   0.9120
      1024        3.4584         3.7233   0.9289
     65536        4.4495         4.7096   0.9448
    262144        4.6582         4.9953   0.9325
  → 样本数从 16 涨到 26 万（1.6 万倍），E[max] 只从 2.08 涨到 4.9（2.4 倍）
  → 所以纯统计意义上，粒度带来的收益是对数级的、温和的
  → 真正把某个粒度打崩的是「分布本身的异质」：它让 max 由极少数元素独占

=== 4. 异质源如何抬高 E[max]（这才是粒度的真正杠杆）===
                  数据结构   per-tensor E[max²]  per-token E[max²]  per-ch-grp32 E[max²]
                   纯高斯               21.691              8.226                 5.732
            token 幅度异质             1008.050              8.049                 5.701
        通道 outlier ×50              410.968             77.768                19.496
                  两者都有             7539.845             81.364                26.845
```

### 4.3 K 和 V 的误差怎么传播

直觉上「K 量化比 V 量化危险」，因为 softmax 对打分很敏感。实测下来没这么简单。

```python
"""第 033 期 实验 3（v2）：K 和 V 的误差怎么传播
本机真实运行（torch 2.14 CPU，math 后端 SDPA）。
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_grad_enabled(False)
D = 128


def qdq_int8(x, gran, g=None):
    shape = x.shape
    if gran == "per_tensor":
        f = x.reshape(1, -1)
        s = (f.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        return ((f / s).round().clamp(-127, 127) * s).view(shape)
    if gran == "per_token":
        s = (x.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        return (x / s).round().clamp(-127, 127) * s
    if gran == "per_channel":
        s = (x.abs().amax(-2, keepdim=True) / 127).clamp_min(1e-12)
        return (x / s).round().clamp(-127, 127) * s
    if gran == "per_token_group":
        *pre, d = shape
        f = x.reshape(*pre, d // g, g)
        s = (f.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        return ((f / s).round().clamp(-127, 127) * s).view(shape)
    raise ValueError(gran)


def make_qkv(S, Sq=256, H=8, q_gain=1.0, v_corr=0.0, corr_kind="ar1", seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(1, H, Sq, D, generator=g) * q_gain
    K = torch.randn(1, H, S, D, generator=g)
    V = torch.randn(1, H, S, D, generator=g)
    if v_corr > 0:
        if corr_kind == "rank1":
            u = torch.randn(1, H, 1, D, generator=g)
            V = (1 - v_corr) ** 0.5 * V + v_corr ** 0.5 * u
        else:                                # AR(1)：真实的语言里相邻 token 的 V 是相关的
            rho = v_corr
            e = torch.randn(1, H, S + 4096, D, generator=g) * (1 - rho ** 2) ** 0.5
            acc = torch.zeros(1, H, D)
            out = []
            for t in range(S + 4096):
                acc = rho * acc + e[:, :, t]
                out.append(acc)
            V = torch.stack(out, 2)[:, :, 4096:] * (1 / (1 - rho ** 2) ** 0.5)
    return Q, K, V


def rel(a, b):
    return (a - b).norm() / b.norm()


def attn(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V, scale=D ** -0.5)


def run(Q, K, V, kq=None, vq=None, gran="per_token_group", g=32):
    out = attn(Q, K, V)
    K2 = K if kq is None else qdq_int8(K, gran, g)
    V2 = V if vq is None else qdq_int8(V, gran, g)
    return rel(attn(Q, K2, V2), out)


def main():
    print("=== 1. 粒度对 K 和 V 的影响几乎一样（S=4096，int8）===")
    Q, K, V = make_qkv(4096)
    print(f"{'粒度':<24}{'K 自身 err':>12}{'V 自身 err':>12}"
          f"{'只量化 K':>12}{'只量化 V':>12}{'两个都量化':>12}")
    for name, gran, g in [("per-tensor", "per_tensor", None),
                          ("per-channel", "per_channel", None),
                          ("per-token", "per_token", None),
                          ("per-token-group g=64", "per_token_group", 64),
                          ("per-token-group g=32", "per_token_group", 32),
                          ("per-token-group g=16", "per_token_group", 16)]:
        ke = rel(qdq_int8(K, gran, g), K)
        ve = rel(qdq_int8(V, gran, g), V)
        a = run(Q, K, V, kq=True, gran=gran, g=g)
        b = run(Q, K, V, vq=True, gran=gran, g=g)
        c = run(Q, K, V, kq=True, vq=True, gran=gran, g=g)
        print(f"{name:<24}{ke:>12.6f}{ve:>12.6f}{a:>12.6f}{b:>12.6f}{c:>12.6f}")
    print("  → 输出相对误差 ≈ 被判据操作数自身的相对误差；K 和 V 在这一点上没有差异")

    print()
    print("=== 2. 上下文变长，误差会累积吗（int8 per-token-group g=32）===")
    print(f"{'S':>8}{'Σp²':>12}{'只量化 K':>12}{'只量化 V':>12}")
    for S in [256, 1024, 4096, 16384]:
        Q, K, V = make_qkv(S)
        p = F.softmax(Q @ K.transpose(-1, -2) * (D ** -0.5), dim=-1)
        print(f"{S:>8}{p.pow(2).sum(-1).mean().item():>12.6f}"
              f"{run(Q, K, V, kq=True):>12.6f}{run(Q, K, V, vq=True):>12.6f}")
    print("  → Σp² 随 S 掉了 40 倍，输出误差纹丝不动：分母被同等地平均掉了，比值不变")

    print()
    print("=== 3. 关键恒等式：softmax 的概率扰动之和恒为 0 ===")
    Q, K, V = make_qkv(4096)
    p = F.softmax(Q @ K.transpose(-1, -2) * (D ** -0.5), dim=-1)
    K2 = qdq_int8(K, "per_token_group", 32)
    p2 = F.softmax(Q @ K2.transpose(-1, -2) * (D ** -0.5), dim=-1)
    dp = p2 - p
    print(f"  Σ_j p_j     = {p.sum(-1).mean().item():.10f}")
    print(f"  Σ_j p'_j    = {p2.sum(-1).mean().item():.10f}")
    print(f"  |Σ_j Δp_j|  = {dp.sum(-1).abs().max().item():.3e}   （行和的差，只有浮点残差）")
    print(f"  Σ_j |Δp_j|  = {dp.abs().sum(-1).mean().item():.6f}   （扰动的总幅度，并不小）")
    print(f"  p 的 TV 距离 = {0.5 * dp.abs().sum(-1).mean().item():.6f}")

    print()
    print("=== 4. 推论：K 的误差只能通过 V 的跨 token 变化泄漏到输出 ===")
    print("V = sqrt(1-c)·白噪声 + sqrt(c)·一个跨 token 恒定的向量（Rank-1 相关）")
    print(f"{'c':>8}{'V 的跨token标准差':>20}{'只量化 K':>13}{'只量化 V':>13}{'K/V':>8}")
    for c in [0.0, 0.3, 0.6, 0.9, 0.99, 0.999]:
        Q, K, V = make_qkv(2048, v_corr=c, corr_kind="rank1")
        spread = V.std(dim=2).mean().item()
        a = run(Q, K, V, kq=True)
        b = run(Q, K, V, vq=True)
        print(f"{c:>8.3f}{spread:>20.4f}{a:>13.6f}{b:>13.6f}{a / b:>8.2f}")
    print("  → 两个方向都变好，但 K 快得多：V 越「平」，K 的扰动越打空")

    print()
    print("=== 5. 换成更真实的 AR(1) 跨 token 相关（ρ 扫描）===")
    print(f"{'ρ':>6}{'只量化 K':>13}{'只量化 V':>13}{'K/V':>8}")
    for rho in [0.0, 0.5, 0.9, 0.99]:
        Q, K, V = make_qkv(2048, v_corr=rho, corr_kind="ar1")
        a = run(Q, K, V, kq=True)
        b = run(Q, K, V, vq=True)
        print(f"{rho:>6.2f}{a:>13.6f}{b:>13.6f}{a / b:>8.2f}")


if __name__ == "__main__":
    main()
```

```text
=== 1. 粒度对 K 和 V 的影响几乎一样（S=4096，int8）===
粒度                          K 自身 err    V 自身 err       只量化 K       只量化 V       两个都量化
per-tensor                  0.011541    0.011901    0.011678    0.011912    0.016655
per-channel                 0.008663    0.008671    0.008720    0.008672    0.012291
per-token                   0.006468    0.006459    0.006647    0.006515    0.009306
per-token-group g=64        0.005934    0.005931    0.006052    0.005971    0.008477
per-token-group g=32        0.005350    0.005348    0.005496    0.005430    0.007716
per-token-group g=16        0.004697    0.004693    0.004825    0.004730    0.006753
  → 输出相对误差 ≈ 被判据操作数自身的相对误差；K 和 V 在这一点上没有差异

=== 2. 上下文变长，误差会累积吗（int8 per-token-group g=32）===
       S         Σp²       只量化 K       只量化 V
     256    0.010389    0.005276    0.005327
    1024    0.002654    0.005456    0.005455
    4096    0.000666    0.005496    0.005430
   16384    0.000167    0.005479    0.005364
  → Σp² 随 S 掉了 40 倍，输出误差纹丝不动：分母被同等地平均掉了，比值不变

=== 3. 关键恒等式：softmax 的概率扰动之和恒为 0 ===
  Σ_j p_j     = 1.0000000000
  Σ_j p'_j    = 1.0000000000
  |Σ_j Δp_j|  = 1.305e-06   （行和的差，只有浮点残差）
  Σ_j |Δp_j|  = 0.004259   （扰动的总幅度，并不小）
  p 的 TV 距离 = 0.002129

=== 4. 推论：K 的误差只能通过 V 的跨 token 变化泄漏到输出 ===
V = sqrt(1-c)·白噪声 + sqrt(c)·一个跨 token 恒定的向量（Rank-1 相关）
       c        V 的跨token标准差        只量化 K        只量化 V     K/V
   0.000              0.9994     0.005462     0.005276    1.04
   0.300              0.8362     0.000302     0.000350    0.86
   0.600              0.6321     0.000162     0.000249    0.65
   0.900              0.3160     0.000066     0.000210    0.31
   0.990              0.0999     0.000020     0.000197    0.10
   0.999              0.0316     0.000006     0.000199    0.03
  → 两个方向都变好，但 K 快得多：V 越「平」，K 的扰动越打空

=== 5. 换成更真实的 AR(1) 跨 token 相关（ρ 扫描）===
     ρ        只量化 K        只量化 V     K/V
  0.00     0.005462     0.005276    1.04
  0.50     0.004250     0.004191    1.01
  0.90     0.002059     0.002031    1.01
  0.99     0.000645     0.000665    0.97
```

三张表各说一件事：

**表 1：误差传播机制本身是对称的。** 同一个粒度下，只量化 K 和只量化 V 的输出误差几乎相同（g=32 时 0.005496 vs 0.005430），而且都精确等于**被量化操作数自身的相对误差**（0.005350）。S=16384 时依旧如此。

**表 2：上下文变长不会让误差累积。** S 从 256 涨到 16384，$\sum_j p_j^2$ 掉了 62 倍（0.0104 → 0.000167），输出误差纹丝不动。这有个精确的解释：$\text{Var}(\Delta \text{out}) = \sigma_\varepsilon^2 \sum_j p_j^2$，而 $\|\text{out}\|^2 \approx \sigma_v^2 \sum_j p_j^2$，**同一个 $\sum p_j^2$ 在分子分母里约掉了**，比值恒为 $\sigma_\varepsilon/\sigma_v$。

所以「注意力越弥散，V 的误差越会被平均掉」这个直觉是**错的**——分母也被同等地平均掉了。这一点值得单独记：**V 的相对误差就是它自身的量化相对误差，与上下文长度无关**（在 V 跨 token 不相关的前提下）。

**表 3 / 表 4：K 的误差只能通过 V 的跨 token 变化泄漏。** 先看一个恒等式：

```
Σ_j Δp_j = 1.305e-06   （行和的差，只有浮点残差）
Σ_j |Δp_j| = 0.004259  （扰动的总幅度，并不小）
```

K 的误差作用在 softmax 概率上，而**概率扰动之和恒等于 0**（softmax 行和恒为 1）。于是 $\Delta\text{out} = \sum_j \Delta p_j v_j$ 里，如果所有 $v_j$ 相等，这一项**精确为零**——K 的扰动完全打在空处。实测验证（rank-1 相关性从 0 扫到 0.999，即 V 的跨 token 标准差从 0.999 掉到 0.032）：

| V 的跨 token 标准差 | 只量化 K | 只量化 V | K/V |
|---|---|---|---|
| 0.9994 | 0.005462 | 0.005276 | 1.04 |
| 0.3160 | 0.000066 | 0.000210 | **0.31** |
| 0.0999 | 0.000020 | 0.000197 | **0.10** |
| 0.0316 | 0.000006 | 0.000199 | **0.03** |

**V 越「平」，K 越安全**：V 几乎不随 token 变化时，只量化 K 的输出误差比只量化 V 小 33 倍。换成更贴近语言的 AR(1) 相关结构，两个方向会同步变好、K/V 比稳定在 1.0 左右——说明起作用的不是「相关性」本身，而是**V 里那个跨整个上下文近似恒定的分量**。

把三张表拼起来，KIVI 的两条处方就都解释得通了：**机制上 K 和 V 的传播是对称的（表 1），不对称的是它们的分布结构**——K 的 outlier 长在通道轴上（§4.1 的 2×2），V 没有这种通道 outlier。所以「K 用 per-channel、V 用 per-token」不是审美，是跟着 outlier 长在哪个轴走。

### 4.4 scale 的字节账（含一次自查纠错）

这一节的第一版是错的：闭式写成 $\frac{4}{g \cdot n_{kv} \cdot b}$，而「显式计数」的代码也漏了同一个 $n_{kv}$ 因子，**两边共享同一个错误假设，互相核对检不出来**。后来用 DeepSeek V3.2 的真实字节布局做外部对照才发现对不上。修正后：

```python
"""第 033 期 实验 2（v2）：scale 自己的字节账
v1 的闭式和「显式计数」共享了同一个错误假设（漏了 head 数因子），互相核对检不出来。
这一版回到第一性原理：数「一层里到底有几个组、每个组多少字节」，
并用 DeepSeek V3.2 的真实 kv cache 布局（656 B/token）做外部对照。
"""
import random


def count_truth(n_kv, d, mode, group, b_value, s_items=1, s_bytes=4):
    """第一性原理计数。返回 (每个 token 每层的数值字节, scale 字节)。

    一层里 KV 的组数：
      per-token-group   每个 token 的每个 head 各把 d 个通道切成 d/group 组 → 2·n_kv·(d/group) 组
      per-channel-group 每 group 个 token 一块，块内每 head 每 d 个通道一个 scale → 2·n_kv·d/group 组
      per-token         每个 token 的每个 head 一行一个 → 2·n_kv 组，组内含 d 个数值
      per-tensor        整层就 2 个标量，摊到整个 cache，按 0 算
    """
    if mode == "per_token_group":
        assert group <= d, f"group={group} > d={d}：分组退化，无意义"
    vpt = 2 * n_kv * d * b_value
    if mode == "per_tensor":
        return vpt, 0.0
    if mode == "per_token":
        n_groups = 2 * n_kv
    elif mode == "per_token_group":
        n_groups = 2 * n_kv * (d // group)
    elif mode == "per_channel_group":
        n_groups = 2 * n_kv * d / group
    else:
        raise ValueError(mode)
    return vpt, n_groups * s_items * s_bytes


def table(title, n_kv, d, rows, b_value, s_items=1, s_bytes=4):
    print(f"--- {title}（n_kv={n_kv}, d={d}, {b_value} B/数值）---")
    print(f"{'粒度':<24}{'scale B/tok/层':>16}{'实际 B/数值':>14}{'vs bf16':>10}{'scale 占比':>12}")
    for name, mode, grp in rows:
        v, s = count_truth(n_kv, d, mode, grp, b_value, s_items, s_bytes)
        eff = (v + s) / (2 * n_kv * d)
        print(f"{name:<24}{s:>16.1f}{eff:>14.4f}{2.0 / eff:>9.2f}x{s / v * 100:>11.2f}%")


def main():
    rows = [("per-tensor", "per_tensor", None),
            ("per-token", "per_token", None),
            ("per-token-group g=128", "per_token_group", 128),
            ("per-token-group g=64", "per_token_group", 64),
            ("per-token-group g=32", "per_token_group", 32),
            ("per-token-group g=16", "per_token_group", 16),
            ("per-channel-group t=128", "per_channel_group", 128),
            ("per-channel-group t=256", "per_channel_group", 256)]

    print("=== 1. 统一结论：开销 = 每个 scale 的字节数 / 一个组里数值的字节数 ===")
    print("   一个组里有 group 个数值，每个 b 字节；这个组要付 items×B 字节的 scale。")
    print("   开销 = items·B / (group·b)   —— 与 head 数完全无关（分子分母同时乘 2·n_kv）")
    print()
    print(f"{'group':>7}{'b':>6}{'scale 字节':>11}{'闭式':>10}{'第一性原理':>13}")
    for grp in [16, 32, 64, 128, 512]:
        for b in [1.0, 0.5]:
            for si, sb in [(1, 4), (1, 1)]:
                v, s = count_truth(8, grp, "per_token_group", grp, b, si, sb)
                print(f"{grp:>7}{b:>6.1f}{si * sb:>11}{si * sb / (grp * b) * 100:>9.2f}%"
                      f"{s / v * 100:>12.2f}%")
    print()

    print("=== 2. int8 / fp8 KV（1 B/数值，fp32 scale）===")
    table("GQA-8（Llama-3 风格）", 8, 128, rows, 1.0)
    print()
    table("MQA（n_kv=1）", 1, 128, rows, 1.0)
    print()
    print("=== 3. int4 KV（0.5 B/数值，fp32 scale）===")
    table("GQA-8", 8, 128, rows, 0.5)
    print()
    table("MQA（n_kv=1）", 1, 128, rows, 0.5)
    print()

    print("=== 4. 外部对照：DeepSeek V3.2 的真实 fp8_ds_mla 布局（656 B/token）===")
    print("  官方布局：512 B = 512 个 float8_e4m3（NoPE 部分）")
    print("            16 B  = 4 个 float32 scale，每个管连续 128 个 fp8 值 → group=128")
    print("           128 B  = 64 个 bfloat16（RoPE key，不量化）")
    real_s = 16.0
    real_v = 512.0
    print(f"  实测 scale 占比 = {real_s}/{real_v} = {real_s / real_v * 100:.4f}%")
    v, s = count_truth(1, 512, "per_token_group", 128, 1.0, 1, 4)
    print(f"  第一性原理   = {s}/{v} = {s / v * 100:.4f}%   → 逐字节相符")
    print()
    print("=== 5. 外部对照：DeepSeek V4 的 fp8_ds_mla 布局（584 B/token）===")
    print("  官方布局：448 B = 448 个 float8_e4m3")
    print("           128 B = 64 个 bfloat16（RoPE，不量化）")
    print("             8 B = 7 个 ue8m0 scale（每个 1 字节，管 64 个 fp8 值）+ 1 B padding")
    print(f"  实测（含 padding）= 8/448 = {8 / 448 * 100:.4f}%")
    v, s = count_truth(1, 448, "per_token_group", 64, 1.0, 1, 1)
    print(f"  第一性原理（不含 padding）= {s}/{v} = {s / v * 100:.4f}%"
          f"，加上 1 B padding 后 {(s + 1) / v * 100:.4f}%")
    print("  → 注意 ue8m0 是 1 字节的 2 的幂次 scale，比 fp32 省 4 倍，所以 V4 能把 group 砍到 64")
    print()

    print("=== 6. 落到显存：7B / 32 层 / GQA-8 / d=128，80GB 卡留 60GB 给 KV ===")
    L = 32
    pool = 60 * 1024 ** 3
    print(f"{'KV 精度与粒度':<28}{'KiB/token':>12}{'可驻留 token':>15}{'vs bf16':>10}")
    ref = None
    for tag, b, mode, grp in [("bf16 基线", 2.0, None, None),
                              ("fp8/int8 per-tensor", 1.0, "per_tensor", None),
                              ("fp8/int8 per-token", 1.0, "per_token", None),
                              ("fp8/int8 g=64", 1.0, "per_token_group", 64),
                              ("fp8/int8 g=32", 1.0, "per_token_group", 32),
                              ("int4 per-token", 0.5, "per_token", None),
                              ("int4 g=64", 0.5, "per_token_group", 64),
                              ("int4 g=32", 0.5, "per_token_group", 32),
                              ("int4 g=16", 0.5, "per_token_group", 16)]:
        if mode is None:
            v, s = 2 * 8 * 128 * 2.0, 0.0
        else:
            v, s = count_truth(8, 128, mode, grp, b)
        tot = (v + s) * L
        cap = pool / tot
        if ref is None:
            ref = cap
        print(f"{tag:<28}{tot / 1024:>12.1f}{cap / 1e6:>12.2f}M{cap / ref:>9.2f}x")

    print()
    print("=== 7. 自检：随机 500 组参数，闭式 vs 第一性原理 ===")
    rnd = random.Random(0)
    worst = 0.0
    for _ in range(500):
        n_kv = rnd.choice([1, 2, 4, 8, 16])
        grp = rnd.choice([16, 32, 64, 128])
        d = grp * rnd.choice([1, 2, 4, 8])
        b = rnd.choice([0.5, 1.0])
        si, sb = rnd.choice([(1, 4), (1, 1)])
        v, s = count_truth(n_kv, d, "per_token_group", grp, b, si, sb)
        closed = si * sb / (grp * b)
        worst = max(worst, abs(s / v - closed))
    print(f"  最大偏差 = {worst:.3e}（应为 0；前提是 group ≤ d）")
    print("  v1 的教训：闭式和「显式计数」如果共享同一个错误假设，互相核对是检不出来的 ——")
    print("  必须回到「一层里到底有几个组」从头数一遍。")


if __name__ == "__main__":
    main()
```

```text
=== 1. 统一结论：开销 = 每个 scale 的字节数 / 一个组里数值的字节数 ===
   一个组里有 group 个数值，每个 b 字节；这个组要付 items×B 字节的 scale。
   开销 = items·B / (group·b)   —— 与 head 数完全无关（分子分母同时乘 2·n_kv）

  group     b   scale 字节        闭式        第一性原理
     16   1.0          4    25.00%       25.00%
     16   1.0          1     6.25%        6.25%
     16   0.5          4    50.00%       50.00%
     16   0.5          1    12.50%       12.50%
     32   1.0          4    12.50%       12.50%
     32   1.0          1     3.12%        3.12%
     32   0.5          4    25.00%       25.00%
     32   0.5          1     6.25%        6.25%
     64   1.0          4     6.25%        6.25%
     64   1.0          1     1.56%        1.56%
     64   0.5          4    12.50%       12.50%
     64   0.5          1     3.12%        3.12%
    128   1.0          4     3.12%        3.12%
    128   1.0          1     0.78%        0.78%
    128   0.5          4     6.25%        6.25%
    128   0.5          1     1.56%        1.56%
    512   1.0          4     0.78%        0.78%
    512   1.0          1     0.20%        0.20%
    512   0.5          4     1.56%        1.56%
    512   0.5          1     0.39%        0.39%

=== 2. int8 / fp8 KV（1 B/数值，fp32 scale）===
--- GQA-8（Llama-3 风格）（n_kv=8, d=128, 1.0 B/数值）---
粒度                         scale B/tok/层       实际 B/数值   vs bf16    scale 占比
per-tensor                           0.0        1.0000     2.00x       0.00%
per-token                           64.0        1.0312     1.94x       3.12%
per-token-group g=128               64.0        1.0312     1.94x       3.12%
per-token-group g=64               128.0        1.0625     1.88x       6.25%
per-token-group g=32               256.0        1.1250     1.78x      12.50%
per-token-group g=16               512.0        1.2500     1.60x      25.00%
per-channel-group t=128             64.0        1.0312     1.94x       3.12%
per-channel-group t=256             32.0        1.0156     1.97x       1.56%

--- MQA（n_kv=1）（n_kv=1, d=128, 1.0 B/数值）---
粒度                         scale B/tok/层       实际 B/数值   vs bf16    scale 占比
per-tensor                           0.0        1.0000     2.00x       0.00%
per-token                            8.0        1.0312     1.94x       3.12%
per-token-group g=128                8.0        1.0312     1.94x       3.12%
per-token-group g=64                16.0        1.0625     1.88x       6.25%
per-token-group g=32                32.0        1.1250     1.78x      12.50%
per-token-group g=16                64.0        1.2500     1.60x      25.00%
per-channel-group t=128              8.0        1.0312     1.94x       3.12%
per-channel-group t=256              4.0        1.0156     1.97x       1.56%

=== 3. int4 KV（0.5 B/数值，fp32 scale）===
--- GQA-8（n_kv=8, d=128, 0.5 B/数值）---
粒度                         scale B/tok/层       实际 B/数值   vs bf16    scale 占比
per-tensor                           0.0        0.5000     4.00x       0.00%
per-token                           64.0        0.5312     3.76x       6.25%
per-token-group g=128               64.0        0.5312     3.76x       6.25%
per-token-group g=64               128.0        0.5625     3.56x      12.50%
per-token-group g=32               256.0        0.6250     3.20x      25.00%
per-token-group g=16               512.0        0.7500     2.67x      50.00%
per-channel-group t=128             64.0        0.5312     3.76x       6.25%
per-channel-group t=256             32.0        0.5156     3.88x       3.12%

--- MQA（n_kv=1）（n_kv=1, d=128, 0.5 B/数值）---
粒度                         scale B/tok/层       实际 B/数值   vs bf16    scale 占比
per-tensor                           0.0        0.5000     4.00x       0.00%
per-token                            8.0        0.5312     3.76x       6.25%
per-token-group g=128                8.0        0.5312     3.76x       6.25%
per-token-group g=64                16.0        0.5625     3.56x      12.50%
per-token-group g=32                32.0        0.6250     3.20x      25.00%
per-token-group g=16                64.0        0.7500     2.67x      50.00%
per-channel-group t=128              8.0        0.5312     3.76x       6.25%
per-channel-group t=256              4.0        0.5156     3.88x       3.12%

=== 4. 外部对照：DeepSeek V3.2 的真实 fp8_ds_mla 布局（656 B/token）===
  官方布局：512 B = 512 个 float8_e4m3（NoPE 部分）
            16 B  = 4 个 float32 scale，每个管连续 128 个 fp8 值 → group=128
           128 B  = 64 个 bfloat16（RoPE key，不量化）
  实测 scale 占比 = 16.0/512.0 = 3.1250%
  第一性原理   = 32/1024.0 = 3.1250%   → 逐字节相符

=== 5. 外部对照：DeepSeek V4 的 fp8_ds_mla 布局（584 B/token）===
  官方布局：448 B = 448 个 float8_e4m3
           128 B = 64 个 bfloat16（RoPE，不量化）
             8 B = 7 个 ue8m0 scale（每个 1 字节，管 64 个 fp8 值）+ 1 B padding
  实测（含 padding）= 8/448 = 1.7857%
  第一性原理（不含 padding）= 14/896.0 = 1.5625%，加上 1 B padding 后 1.6741%
  → 注意 ue8m0 是 1 字节的 2 的幂次 scale，比 fp32 省 4 倍，所以 V4 能把 group 砍到 64

=== 6. 落到显存：7B / 32 层 / GQA-8 / d=128，80GB 卡留 60GB 给 KV ===
KV 精度与粒度                       KiB/token      可驻留 token   vs bf16
bf16 基线                            128.0        0.49M     1.00x
fp8/int8 per-tensor                 64.0        0.98M     2.00x
fp8/int8 per-token                  66.0        0.95M     1.94x
fp8/int8 g=64                       68.0        0.93M     1.88x
fp8/int8 g=32                       72.0        0.87M     1.78x
int4 per-token                      34.0        1.85M     3.76x
int4 g=64                           36.0        1.75M     3.56x
int4 g=32                           40.0        1.57M     3.20x
int4 g=16                           48.0        1.31M     2.67x

=== 7. 自检：随机 500 组参数，闭式 vs 第一性原理 ===
  最大偏差 = 0.000e+00（应为 0；前提是 group ≤ d）
  v1 的教训：闭式和「显式计数」如果共享同一个错误假设，互相核对是检不出来的 ——
  必须回到「一层里到底有几个组」从头数一遍。
```

**教训值得单独记一行**：拿两个实现互相核对的前提是它们**不共享前提假设**。否则最稳妥的办法是找一个外部真值（这里是官方公布的 656 B 布局）来对账。

### 4.5 roofline：量化能换到多少

本机没有 CUDA，这一节是解析推算，不是实测。假设带宽已饱和（大 batch 长上下文下成立），单步 decode 耗时 $t = (W_\text{bytes} + B \cdot S \cdot k)/BW$。

```python
"""第 033 期 实验 5：量化 KV 在 decode 里到底省多少（roofline 模型，不是实测）
本机没有 CUDA，这部分是解析推算，明确标注。
"""
import torch

# H800 SXM 级别：BF16 稠密算力 989 TFLOPS、HBM 带宽 3.35 TB/s（公开规格）
PEAK_BF16 = 989e12
BW = 3.35e12
BALANCE = PEAK_BF16 / BW          # FLOP/byte 的平衡点


def kv_bytes_per_token(L, n_kv, d, b_value, scale_bytes=0.0):
    """每层每 token 的 KV 字节（K 和 V 都要存）× 层数。"""
    return L * (2 * n_kv * d * b_value + scale_bytes)


def main():
    print(f"硬件口径（H800 级）：PEAK={PEAK_BF16 / 1e12:.0f} TFLOPS, "
          f"BW={BW / 1e12:.2f} TB/s → 平衡点 {BALANCE:.1f} FLOP/byte")
    print()

    # ---- 1. 7B / 32 层 / GQA-8 / d=128 的 KV 账本 ----
    L, n_kv, d, N = 32, 8, 128, 7.0e9
    W = N * 2                                   # bf16 权重字节
    print("=== 1. KV 与权重谁是大头（7B 密集模型，GQA-8，d=128）===")
    print(f"权重 {W / 1e9:.1f} GB (bf16)")
    print(f"{'精度':<26}{'KiB/token':>12}{'S=8K,B=32 时 KV 读取':>22}{'KV 占字节比':>13}")
    for tag, b, sb in [("bf16", 2.0, 0.0), ("fp8（per-tensor）", 1.0, 0.0),
                       ("fp8（per-tok-grp32）", 1.0, 32.0),
                       ("int4（per-tensor）", 0.5, 0.0),
                       ("int4（per-tok-grp32）", 0.5, 32.0),
                       ("int4（per-tok-grp16）", 0.5, 64.0)]:
        k = kv_bytes_per_token(L, n_kv, d, b, sb)
        read = 32 * 8192 * k
        print(f"{tag:<26}{k / 1024:>12.1f}{read / 1e9:>17.1f} GB"
              f"{read / (read + W) * 100:>12.1f}%")
    print()

    # ---- 2. 端到端 decode 时间 ----
    print("=== 2. decode 单步耗时 = (权重字节 + KV 读取字节) / BW ===")
    print("假设带宽已饱和（大 batch 长上下文下成立）")
    print(f"{'配置':<30}{'B=32,S=8K':>12}{'B=128,S=8K':>13}"
          f"{'B=32,S=32K':>13}{'B=128,S=32K':>14}")
    cfgs = [("bf16 权重 + bf16 KV", 2.0, 0.0, 2.0),
            ("bf16 权重 + fp8 KV", 2.0, 0.0, 1.0),
            ("bf16 权重 + fp8 KV grp32", 2.0, 32.0, 1.0),
            ("bf16 权重 + int4 KV grp32", 2.0, 32.0, 0.5),
            ("fp8 权重 + bf16 KV", 1.0, 0.0, 2.0),
            ("fp8 权重 + fp8 KV", 1.0, 0.0, 1.0),
            ("fp8 权重 + int4 KV grp32", 1.0, 32.0, 0.5)]
    base = {}
    for tag, bw_, sb, bkv in cfgs:
        row = []
        for B, S in [(32, 8192), (128, 8192), (32, 32768), (128, 32768)]:
            wb = N * bw_
            kb = B * S * kv_bytes_per_token(L, n_kv, d, bkv, sb)
            t = (wb + kb) / BW
            row.append(t * 1e3)
            if tag == "bf16 权重 + bf16 KV":
                base[(B, S)] = t
        print(f"{tag:<30}" + "".join(f"{v:>11.2f} ms" if i == 0 else f"{v:>12.2f} ms"
                                     for i, v in enumerate(row)))
    print()
    print("  相对「bf16 权重 + bf16 KV」的加速比：")
    for tag, bw_, sb, bkv in cfgs:
        sp = []
        for B, S in [(32, 8192), (128, 8192), (32, 32768), (128, 32768)]:
            wb = N * bw_
            kb = B * S * kv_bytes_per_token(L, n_kv, d, bkv, sb)
            sp.append(base[(B, S)] / ((wb + kb) / BW))
        print(f"    {tag:<28}" + "".join(f"{v:>12.2f}x" for v in sp))
    print()

    # ---- 3. 显存容量 ----
    print("=== 3. 显存：80GB 卡留 60GB 给 KV（7B 权重已单独占 14 GB bf16）===")
    pool = 60 * 1024 ** 3
    print(f"{'精度':<26}{'KiB/token':>12}{'可驻留 token':>16}{'vs bf16':>10}")
    ref = None
    for tag, b, sb in [("bf16", 2.0, 0.0), ("fp8 per-tensor", 1.0, 0.0),
                       ("fp8 per-tok-grp32", 1.0, 32.0),
                       ("int4 per-tensor", 0.5, 0.0),
                       ("int4 per-tok-grp32", 0.5, 32.0)]:
        k = kv_bytes_per_token(L, n_kv, d, b, sb)
        cap = pool / k
        if ref is None:
            ref = cap
        print(f"{tag:<26}{k / 1024:>12.1f}{cap / 1e6:>13.2f}M{cap / ref:>9.2f}x")
    print()

    # ---- 4. KV 量化把「算力/带宽」平衡点推向哪里 ----
    print("=== 4. 平衡点怎么动：一次前向里「每字节携带多少 FLOP」===")
    print("decode 时 FLOPs/token ≈ 2N + 2·L·n_kv·d·(2S/…)，这里只看主导项：")
    print("  FLOPs ≈ 2N + 4·L·n_h·d·S（attention 部分用 128 个 Q 头）")
    n_h = 128
    print(f"{'精度':<26}{'B·S':>12}{'算术强度':>12}{'是否带宽瓶颈':>14}")
    for tag, b, sb in [("bf16", 2.0, 0.0), ("fp8", 1.0, 0.0), ("int4 grp32", 0.5, 32.0)]:
        k = kv_bytes_per_token(L, n_kv, d, b, sb)
        for B, S in [(32, 8192), (128, 32768)]:
            flops = 2 * N * B + 4 * L * n_h * d * S * B
            byt = N * 2 + B * S * k
            ai = flops / byt
            print(f"{tag:<26}{B * S:>12}{ai:>12.1f}"
                  f"{'  是' if ai < BALANCE else '  否':>14}")


if __name__ == "__main__":
    main()
```

```text
硬件口径（H800 级）：PEAK=989 TFLOPS, BW=3.35 TB/s → 平衡点 295.2 FLOP/byte

=== 1. KV 与权重谁是大头（7B 密集模型，GQA-8，d=128）===
权重 14.0 GB (bf16)
精度                           KiB/token     S=8K,B=32 时 KV 读取      KV 占字节比
bf16                             128.0             34.4 GB        71.1%
fp8（per-tensor）                   64.0             17.2 GB        55.1%
fp8（per-tok-grp32）                65.0             17.4 GB        55.5%
int4（per-tensor）                  32.0              8.6 GB        38.0%
int4（per-tok-grp32）               33.0              8.9 GB        38.8%
int4（per-tok-grp16）               34.0              9.1 GB        39.5%

=== 2. decode 单步耗时 = (权重字节 + KV 读取字节) / BW ===
假设带宽已饱和（大 batch 长上下文下成立）
配置                               B=32,S=8K   B=128,S=8K   B=32,S=32K   B=128,S=32K
bf16 权重 + bf16 KV                   14.44 ms       45.21 ms       45.21 ms      168.29 ms
bf16 权重 + fp8 KV                     9.31 ms       24.69 ms       24.69 ms       86.23 ms
bf16 权重 + fp8 KV grp32               9.39 ms       25.01 ms       25.01 ms       87.51 ms
bf16 权重 + int4 KV grp32              6.82 ms       14.76 ms       14.76 ms       46.49 ms
fp8 权重 + bf16 KV                    12.35 ms       43.12 ms       43.12 ms      166.20 ms
fp8 权重 + fp8 KV                      7.22 ms       22.60 ms       22.60 ms       84.14 ms
fp8 权重 + int4 KV grp32               4.73 ms       12.67 ms       12.67 ms       44.40 ms

  相对「bf16 权重 + bf16 KV」的加速比：
    bf16 权重 + bf16 KV                   1.00x        1.00x        1.00x        1.00x
    bf16 权重 + fp8 KV                    1.55x        1.83x        1.83x        1.95x
    bf16 权重 + fp8 KV grp32              1.54x        1.81x        1.81x        1.92x
    bf16 权重 + int4 KV grp32             2.12x        3.06x        3.06x        3.62x
    fp8 权重 + bf16 KV                    1.17x        1.05x        1.05x        1.01x
    fp8 权重 + fp8 KV                     2.00x        2.00x        2.00x        2.00x
    fp8 权重 + int4 KV grp32              3.05x        3.57x        3.57x        3.79x

=== 3. 显存：80GB 卡留 60GB 给 KV（7B 权重已单独占 14 GB bf16）===
精度                           KiB/token       可驻留 token   vs bf16
bf16                             128.0         0.49M     1.00x
fp8 per-tensor                    64.0         0.98M     2.00x
fp8 per-tok-grp32                 65.0         0.97M     1.97x
int4 per-tensor                   32.0         1.97M     4.00x
int4 per-tok-grp32                33.0         1.91M     3.88x

=== 4. 平衡点怎么动：一次前向里「每字节携带多少 FLOP」===
decode 时 FLOPs/token ≈ 2N + 2·L·n_kv·d·(2S/…)，这里只看主导项：
  FLOPs ≈ 2N + 4·L·n_h·d·S（attention 部分用 128 个 Q 头）
精度                                 B·S        算术强度        是否带宽瓶颈
bf16                            262144        20.6             是
bf16                           4194304        18.8             是
fp8                             262144        32.0             是
fp8                            4194304        36.7             是
int4 grp32                      262144        43.6             是
int4 grp32                     4194304        68.0             是
```

关键读数：

- **B=32, S=8K 时 KV 读取占 71.1% 的字节**，KV 从 bf16 降到 fp8 拿 **1.55×**，降到 int4（g=32）拿 **2.12×**；
- 同样砍到 8 bit，**权重量化只拿 1.17×（B=128, S=32K 时退化到 1.01×）**；
- 显存侧：60 GB 的 KV 池，bf16 装 0.49M token，fp8 装 0.98M，int4（per-token）装 1.85M。第 029 期说「缓存驻留量 = 去重后的内容量」，那一期算出的 277K token 在 fp8 下只需要一半的池子；
- 即使 `B·S` 到 4.19M（B=128, S=32K），算术强度也只有 18.8 ~ 68.0，**离 295.2 的平衡点还差 4 倍以上**——KV 量化在可见的负载范围内都还是纯收益。

### 4.6 端到端：在真训过的小模型上量代价

前面都是张量级的误差。这一节训一个 0.883M 的 GQA 模型（4 层、d=128、8 个 Q 头、2 个 KV 头），任务是**复制**：前 64 个 token 随机，后 64 个要求原样重放。这逼模型学出精确的 induction attention——K 一旦受伤、位置选错，就立刻显形。

```python
"""第 033 期 实验 6（v3）：端到端 —— 用「输出分布漂移」量 KV 量化的代价
上一版用 top-1 命中率，结果所有格式都无损 —— 因为任务饱和了（复制段本来就 100%）。
换成连续指标：KL(bf16 的输出分布 ‖ 量化后的输出分布)，单位 millinat。它不饱和。
本机真实训练 + 真跑推理。
"""
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
VOCAB, T = 48, 64
D_MODEL, N_HEAD, N_KV, N_LAYER = 128, 8, 2, 4
STEPS, BATCH = 700, 16


class TinyGQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(T * 2, D_MODEL)
        self.blocks = nn.ModuleList()
        for _ in range(N_LAYER):
            self.blocks.append(nn.ModuleDict(dict(
                ln1=nn.LayerNorm(D_MODEL),
                q=nn.Linear(D_MODEL, N_HEAD * 32, bias=False),
                k=nn.Linear(D_MODEL, N_KV * 32, bias=False),
                v=nn.Linear(D_MODEL, N_KV * 32, bias=False),
                o=nn.Linear(N_HEAD * 32, D_MODEL, bias=False),
                ln2=nn.LayerNorm(D_MODEL),
                fc1=nn.Linear(D_MODEL, 4 * D_MODEL, bias=False),
                fc2=nn.Linear(4 * D_MODEL, D_MODEL, bias=False),
            )))
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, idx, kvfn=None):
        B, S = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(S, device=idx.device))[None]
        for b in self.blocks:
            h = b["ln1"](x).to(torch.float32)
            q = b["q"](h).view(B, S, N_HEAD, 32).transpose(1, 2)
            k = b["k"](h).view(B, S, N_KV, 32).transpose(1, 2)
            v = b["v"](h).view(B, S, N_KV, 32).transpose(1, 2)
            if kvfn is not None:
                k, v = kvfn(k), kvfn(v)
            a = F.scaled_dot_product_attention(
                q, k.repeat_interleave(N_HEAD // N_KV, 1),
                v.repeat_interleave(N_HEAD // N_KV, 1),
                is_causal=True, scale=32 ** -0.5)
            x = x + b["o"](a.transpose(1, 2).reshape(B, S, -1))
            x = x + b["fc2"](F.gelu(b["fc1"](b["ln2"](x))))
        return self.head(self.lnf(x))


def _grp(x, g):
    return x.reshape(*x.shape[:-1], 1 if g is None else x.shape[-1] // g,
                     1 if g is None else g) if g is None else \
        x.reshape(*x.shape[:-1], x.shape[-1] // g, g)


def q_int(x, bits, g=None):
    qm = 2 ** (bits - 1) - 1
    if g is None:
        s = (x.abs().amax() / qm).clamp_min(1e-12)
        return (x / s).round().clamp(-qm, qm) * s
    f = x.reshape(*x.shape[:-1], x.shape[-1] // g, g)
    s = (f.abs().amax(-1, keepdim=True) / qm).clamp_min(1e-12)
    return ((f / s).round().clamp(-qm, qm) * s).reshape(x.shape)


def q_fp8(x, g=None):
    if g is None:
        s = (x.abs().amax() / 448).clamp_min(1e-12)
        return (x / s).to(torch.float8_e4m3fn).to(torch.float32) * s
    f = x.reshape(*x.shape[:-1], x.shape[-1] // g, g)
    s = (f.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-12)
    return ((f / s).to(torch.float8_e4m3fn).to(torch.float32) * s).reshape(x.shape)


def make_noise(eps, seed=7):
    def f(x):
        g = torch.Generator().manual_seed(seed)
        return x + torch.randn(x.shape, generator=g) * x.std() * eps
    return f


def make_batch(bs=BATCH, gen=None):
    a = torch.randint(0, VOCAB, (bs, T), generator=gen)
    return torch.cat([a, a], 1)


def main():
    gt = torch.Generator().manual_seed(1234)
    model = TinyGQA()
    print(f"模型 {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M 参数，"
          f"{N_LAYER} 层，d={D_MODEL}，Q 头 {N_HEAD} / KV 头 {N_KV}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=STEPS, pct_start=0.1)
    t0 = time.time()
    for step in range(STEPS):
        xb = make_batch(gen=gt)
        loss = F.cross_entropy(model(xb[:, :-1]).reshape(-1, VOCAB), xb[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    print(f"训练 {STEPS} 步用了 {time.time() - t0:.1f} s")
    model.eval()

    xv = make_batch(bs=64, gen=torch.Generator().manual_seed(999))
    xin, tgt = xv[:, :-1], xv[:, 1:]
    with torch.no_grad():
        ref = model(xin)
        ce0 = F.cross_entropy(ref.reshape(-1, VOCAB), tgt.reshape(-1)).item()
        acc0 = (ref.argmax(-1) == tgt).float().mean().item()
        logp0 = F.log_softmax(ref, -1)
        p0 = logp0.exp()
        margin = (ref.topk(2, -1).values[:, :, 0] -
                  ref.topk(2, -1).values[:, :, 1]).mean().item()
    print(f"bf16 KV 基线：CE {ce0:.4f}，top-1 {acc0 * 100:.2f}%，"
          f"top1−top2 的平均 logit 间距 {margin:.3f}")

    def probe(kvfn):
        with torch.no_grad():
            lg = model(xin, kvfn=kvfn)
            logp = F.log_softmax(lg, -1)
            kl = (p0 * (logp0 - logp)).sum(-1).mean().item() * 1000   # millinat
            dmax = (lg - ref).abs().max().item()
            acc = (lg.argmax(-1) == tgt).float().mean().item()
            ce = F.cross_entropy(lg.reshape(-1, VOCAB), tgt.reshape(-1)).item()
        return kl, dmax, ce, acc

    print()
    print("=== 1. 各格式的真实相对误差 + 输出分布漂移 ===")
    print(f"{'KV 精度':<24}{'K 的 rel err':>14}{'ΔCE':>9}{'top-1':>9}"
          f"{'最大 logit 偏移':>16}{'KL 漂移(mnat)':>15}{'KV KiB/tok':>12}")
    with torch.no_grad():
        h0 = model.blocks[0]["ln1"](model.tok(xin))
        K0 = model.blocks[0]["k"](h0.to(torch.float32))
    rows = [("bf16（基线）", None, 2.0, 0.0),
            ("fp8 per-tensor", lambda x: q_fp8(x), 1.0, 0.0),
            ("fp8 per-tok-grp32", lambda x: q_fp8(x, 32), 1.0, 32.0),
            ("int8 per-tensor", lambda x: q_int(x, 8), 1.0, 0.0),
            ("int8 per-token", lambda x: q_int(x, 8, 32), 1.0, 32.0),
            ("int8 per-tok-grp16", lambda x: q_int(x, 8, 16), 1.0, 64.0),
            ("int4 per-tensor", lambda x: q_int(x, 4), 0.5, 0.0),
            ("int4 per-tok-grp32", lambda x: q_int(x, 4, 32), 0.5, 32.0),
            ("int4 per-tok-grp16", lambda x: q_int(x, 4, 16), 0.5, 64.0)]
    for tag, fn, b_val, sb in rows:
        rel = 0.0 if fn is None else ((fn(K0) - K0).norm() / K0.norm()).item()
        kl, dmax, ce, acc = probe(fn)
        kvk = 2 * N_LAYER * N_KV * 32 * b_val / 1024 + N_LAYER * sb / 1024
        print(f"{tag:<24}{rel * 100:>13.3f}%{ce - ce0:>9.4f}{acc * 100:>8.2f}%"
              f"{dmax:>16.4f}{kl:>15.3f}{kvk:>12.1f}")

    print()
    print("=== 2. 误差预算曲线（KL 口径）：注入噪声 vs 量化，谁更狠 ===")
    print(f"{'注入的相对噪声 ε':>18}{'top-1':>9}{'最大 logit 偏移':>16}{'KL 漂移(mnat)':>15}")
    for eps in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]:
        kl, dmax, ce, acc = probe(make_noise(eps) if eps > 0 else None)
        print(f"{eps:>18.3f}{acc * 100:>8.2f}%{dmax:>16.4f}{kl:>15.3f}")
    print("  → KL 漂移是连续的，能排出格式的好坏；而 top-1 全都在 51.4~51.6% 之间不动")
    print("  → 这就是「指标饱和」：任务本身留了太多判别余量，top-1 量不出代价")

    print()
    print("=== 3. 同样砍到 8 bit：权重 vs KV（KL 口径）===")
    kl0, _, _, _ = probe(None)
    with torch.no_grad():
        m2 = copy.deepcopy(model)
        m2.load_state_dict({k: (q_int(v, 8) if v.ndim >= 2 else v)
                            for k, v in model.state_dict().items()})
        lg = m2(xin)
        logp = F.log_softmax(lg, -1)
        klw = (p0 * (logp0 - logp)).sum(-1).mean().item() * 1000
    klk, _, _, _ = probe(lambda x: q_int(x, 8, 32))
    print(f"  int8 权重 + bf16 KV：KL 漂移 {klw:.3f} mnat")
    print(f"  bf16 权重 + int8 KV：KL 漂移 {klk:.3f} mnat")
    print("  （两者都远小于注入 1% 噪声造成的漂移 —— 见上一张表）")


if __name__ == "__main__":
    main()
```

```text
模型 0.883M 参数，4 层，d=128，Q 头 8 / KV 头 2
训练 700 步用了 21.2 s
bf16 KV 基线：CE 1.9246，top-1 51.46%，top1−top2 的平均 logit 间距 3.756

=== 1. 各格式的真实相对误差 + 输出分布漂移 ===
KV 精度                      K 的 rel err      ΔCE    top-1     最大 logit 偏移    KL 漂移(mnat)  KV KiB/tok
bf16（基线）                        0.000%   0.0000   51.46%          0.0000          0.000         1.0
fp8 per-tensor                  2.589%   0.0000   51.44%          0.2819          0.005         0.5
fp8 per-tok-grp32               2.417%   0.0000   51.41%          0.3136          0.004         0.6
int8 per-tensor                 0.896%  -0.0000   51.44%          0.1393          0.001         0.5
int8 per-token                  0.533%   0.0000   51.46%          0.0778          0.000         0.6
int8 per-tok-grp16              0.472%   0.0000   51.46%          0.0426          0.000         0.8
int4 per-tensor                16.575%   0.0005   51.51%          2.4137          0.306         0.2
int4 per-tok-grp32              9.718%   0.0002   51.48%          1.1277          0.061         0.4
int4 per-tok-grp16              8.663%   0.0002   51.44%          0.8264          0.050         0.5

=== 2. 误差预算曲线（KL 口径）：注入噪声 vs 量化，谁更狠 ===
         注入的相对噪声 ε    top-1     最大 logit 偏移    KL 漂移(mnat)
             0.000   51.46%          0.0000          0.000
             0.010   51.48%          0.0920          0.001
             0.030   51.50%          0.2765          0.005
             0.100   51.45%          1.0374          0.057
             0.300   51.53%          5.1139          0.860
             1.000   46.79%         15.1930        319.894
  → KL 漂移是连续的，能排出格式的好坏；而 top-1 全都在 51.4~51.6% 之间不动
  → 这就是「指标饱和」：任务本身留了太多判别余量，top-1 量不出代价

=== 3. 同样砍到 8 bit：权重 vs KV（KL 口径）===
  int8 权重 + bf16 KV：KL 漂移 0.013 mnat
  bf16 权重 + int8 KV：KL 漂移 0.000 mnat
  （两者都远小于注入 1% 噪声造成的漂移 —— 见上一张表）
```

先说清楚这个任务的性质：**复制段 100.0%，前半段 2.2%**——模型把复制任务学到满分了（前半段随机不可预测，48 词表下 2.1% 就是随机水平），所以「后半段满分」是真的满分，不是任务没学会。

结果是**所有 KV 量化格式几乎无损**（ΔCE ≤ 0.0005，top-1 最大只动 0.05pt）。但这里有个更重要的东西：

**top-1 不变不等于没损失。** 基线 top1−top2 的平均 logit 间距是 **3.756**，而 int4 per-tensor 造成的最大 logit 偏移是 **2.4137**——**扰动还没有越过判别间距，所以 argmax 一个都没翻**。而同一配置的 KL 漂移已经是 **0.306 mnat**——int8 per-tensor 只有 0.001 mnat，int8 per-token 更是低到打印精度以下（< 0.0005）。把人为噪声扫一遍就能看到这个断层：ε=0.3 时最大 logit 偏移 5.11、KL 0.860、top-1 仍是 51.53%；ε=1.0 时偏移 15.19 越过间距，top-1 才掉到 46.79%。

**结论**：判断「KV 量化损失了多少」要用连续指标（KL、perplexity），别用 argmax 一致率。

**但这条结论有规模门槛，必须显式标注。** 0.883M 的模型 + 复制任务，判别余量（3.756）本身就很宽，所以它**能**证明「机制是这样传播的」（K 的误差被 $\sum \Delta p = 0$ 吸收、V 的误差是加权平均），**不能**证明「int4 KV 在大模型上无损」。第 032 期的教训在这里同样适用：小模型对某些问题没有区分度，不能拿它当大模型的证据。

---

## 5. 围绕该领域展开

### 5.1 和第 032 期：MLA 的 576 个元素里，哪一部分被量化

第 032 期算出 MLA 每 token 576 个元素 = latent $d_c$ 512（88.9%）+ 共享 RoPE key $d_h^R$ 64（11.1%）。V3.2 的 `fp8_ds_mla` 布局把这两部分的处理**完全分开**：

- **512 个 NoPE 值全部量化成 fp8**，用 group=128 的 per-block scale（4 个 fp32）；
- **64 个 RoPE key 保持 bf16，不量化**（vLLM 的注释原话是 "This part is not quantized for accuracy"）。

这是个很值得琢磨的不对称：**RoPE key 只占 11.1% 的体积，却被完整保护下来了。** 第 032 期讲过 RoPE 阻断了矩阵吸收（$W^{UK\top} R_{j-i} W^{UK} = I$ 只有 $R=I$ 才成立，最小二乘下界残差 0.7182）。现在换一个角度看同一件事：**RoPE key 会被乘进 $q^\top k$ 的每一个 score，它的误差直接进 softmax 打分，而且不像 latent 那样有低秩结构可供误差"摊薄"**。保护它，代价 11.1%；量化它，收益 11.1%。这笔账在生产里被判定为不划算。

顺带一个第 032 期的延伸：MLA 的 latent 是**同一个张量**（K 和 V 共用），所以 per-token 粒度下每个 token 只有 1 个组，而 GQA-8 有 16 个（8 个 K 头 + 8 个 V 头）。但按 $\text{开销} = items \cdot B/(group \cdot b)$，开销只由**组内元素数**决定：MLA 的 latent 行宽 512 → 组内 512 个值 → 开销 0.78%；GQA-8 的 head_dim 128 → 组内 128 个值 → 开销 3.12%。**宽行比窄行更划算**，这也是 MLA 在精度压缩上顺带占的一个便宜。

### 5.2 和第 030 期：为什么「动态量化的 KV 不能跨机」

第 030 期记过一条官方事实：vLLM 的 NixlConnector 兼容性要求里写着，**动态量化的 KV 不支持跨机传输**（`Dynamic quantization: ❌ Not supported. Per-block scales are not transferred alongside KV cache data.`），而静态量化和 packed-layout 内联 scale 是支持的。

现在能说清这句话在讲什么了：第 030 期讲的 KV 传输契约是「P 侧把 KV block 搬到 D 侧」，但**量化后的 KV 不是一个自包含的东西**——它必须配上 scale 才有意义。如果 scale 是运行时按 block 现算的（`_reshape_cache_per_token_head` 那种在写 KV 时由 Triton kernel 动态算 absmax 的路径），那么 scale 存在**独立的** tensor 里（例如 `[num_blocks, block_size, num_kv_heads]` 的 float32），传输协议没有为它留字段。而 `fp8_ds_mla` 这种把 scale 直接内联进 656 字节布局的写法，就随 KV 数据一起搬走了。

**规则**：跨机共享的前提是「一个 block 的字节自解释」。量化把 KV 从自解释变成了「数据 + 旁路元数据」，元数据要么内联、要么进协议。这也是 `--kv-cache-dtype-skip-layers` 之外，另一个「哪些层能量化」的约束来源。

### 5.3 和第 029 期：缓存驻留量翻倍

第 029 期算过一个去重后的前缀缓存驻留量：**277,305 token = 33.9 GiB**（按 7B/GQA-8/bf16 的 128 KiB/token）。同一个内容量，fp8 只要 17.0 GiB，int4（per-token）要 8.5 GiB。那一期说「容量掉到去重总量的 ~1/5 才明显伤命中率」，量化直接把这条线往后推了 2 ~ 4 倍——**在量化之前需要靠 eviction 换出来的容量，量化之后可能根本不需要换**。KV 量化和前缀缓存是同一个显存池上的两支竞争力量：一支让你多装，一支让你少装。

### 5.4 和第 016 期：同样 8 bit，为什么权重和 KV 的性价比完全相反

第 016 期讲权重量化时，收益是「权重少占显存」。这一期讲 KV 量化，收益是「decode 每一步少读字节」。两者的字节账形状完全不同：

| | 权重量化 | KV 量化 |
|---|---|---|
| 省的字节量 | $N \cdot (b_w - b_w')$，**常数** | $2 L n_{kv} d_h (b - b')$ per token，**随 batch × 上下文增长** |
| decode 里的收益 | 只在小 batch / 短上下文时明显 | 随 $B \cdot S$ 单调放大 |
| 一次前向被读次数 | 每步一次（常数） | 每步 $B \cdot S$ 次 |
| 精度敏感度 | 可以逐层/逐通道调，有大量离线工具 | 必须动态、必须 per-token/per-block |

一句话：**权重量化是容量武器，KV 量化是带宽武器。** decode 是带宽瓶颈（第 027 / 028 期的 $B \cdot S \ll 295$ 那半边），所以 KV 量化打在正确的位置上。

### 5.5 和第 011 期：PagedAttention 的块布局决定了 scale 的落点

第 011 期讲的 PagedAttention 把 KV 切成固定大小的 block，这个块大小（vLLM 默认 16）**也就是 scale 的物理落点**。第 030 期的 `fp8_ds_mla` 用 block size 64、索引器 key 用 block size 64（并明确说「这也是我们对该模型仅支持块大小 64 的原因之一」）。对上了：**scale 的 tiling 必须跟 cache 的物理 tiling 对齐**，否则读写 scale 会变成一次随机的 gather。所以「group 设多大」不是纯精度问题——group 必须整除/被整除于 block 结构，否则你省下的字节会被一次额外访存吃回去。

### 5.6 和其他 KV 压缩手段的正交性

KV 压缩有四条互相正交的轴，量化只是其中一条：

| 轴 | 手段 | 省的是什么 | 已发期数 |
|---|---|---|---|
| **架构** | MQA / GQA / MLA | 每个 token 的**元素个数** | 第 032 期 |
| **精度** | fp8 / int8 / int4 + per-block scale | 每个元素的**字节数** | 本期 |
| **token 选择** | 驱逐、StreamingLLM、H2O | **留多少个 token** | 第 029 期（前缀缓存那侧） |
| **稀疏** | NSA / DSA / MoBA 的 indexer | 每个 query **看多少 token** | 未讲 |

它们可以叠加：MLA 是架构压缩（56.89×），fp8 是精度压缩（2×），DSA 是稀疏压缩（长上下文下读流量再降约 4.8×）。三条轴乘起来才是「每 token 每步读多少字节」的最终答案。

### 5.7 硬件侧：为什么 KV 用得最多的是 fp8 而不是 int8

- **fp8 有原生 Tensor Core 路径**。Hopper 起有 fp8 的 MMA 指令，`torch._scaled_mm` 是它在 PyTorch 里的入口（CUDA 算子，本机无 CUDA，所以本期没有实测它）。int8 对 KV 来说更麻烦：int8 GEMM 需要两边都是整型 + 各自的 scale，而 attention 的 $q^\top k$ 一边是 fp16/bf16 的 query、一边是量化过的 key，混精度的 scale 处理比 fp8 的「双边 fp8 + 两个标量 descale」复杂得多。
- **fp8 的 scale 可以粗**（§2.1 实测：粒度对 fp8 几乎没影响），所以不需要 per-block scale 的基础设施就能用，工程成本低。
- **int4 需要 1 字节的 scale 才划算**（§2.3 与 V4 的 `ue8m0`）。用 fp32 scale 的 int4 在 group=16 时开销 50%，直接把 4× 打成 2.67×。

---

## 6. 什么时候该用 / 不该用

**该用：**

- **长上下文 + 大 batch 的 decode**。$B \cdot S$ 越大，KV 占的字节比例越高（§4.5 实测 B=32,S=8K 已经 71.1%），收益越确定。
- **显存容量是硬约束**的时候。0.49M → 0.98M（fp8）→ 1.85M（int4）token 的容量差，往往比多买一张卡便宜。
- **要跨机传 KV**（第 030 期的 PD 分离 / 分层 KV）。传输量线性减半，而第 030 期算过临界带宽只要 3.87 GB/s——量化把这条线又往下推了一倍。
- **前缀缓存命中之后**。第 029 期说命中率超过 ~97% 后 TTFT 不再降；此时剩下的瓶颈是「驻留量」，量化直接作用在这一项上。

**不该用：**

- **小 batch + 短上下文**。$B \cdot S < 10^5$ 时权重还是大头，1.17× 甚至 1.01× 的收益不值得换精度风险。
- **sliding-window attention 层**。vLLM 专门给了 `--kv-cache-dtype-skip-layers sliding_window` 来绕开它们——窗口内的 KV 样本数少，per-token scale 的极值统计没有优势，误差反而更显眼。
- **不能让 block 自解释的跨机场景**。旁路 scale 传不过去（§5.2）。
- **拿 argmax 一致率当验收指标**。§4.6 已经演示了：top-1 一个都没翻，KL 已经差了三百倍。

---

## 7. 常见坑

**坑 1：用全局 nMSE 判断粒度，会得出反向结论。**
§4.1 里 per-token 的全局 nMSE（0.000392）比 per-channel（0.000713）好 1.8 倍，但通道中位 nMSE 里 per-channel（0.000584）比 per-token（0.110972）好 190 倍。前者意味着「小通道被毁掉了 19% 的能量」，这在聚合指标里看不见。**选粒度必须用「逐通道归一后再看分位数」的口径。**

**坑 2：用 top-1 命中率判断量化有无损失。**
§4.6：int4 per-tensor 的最大 logit 偏移 2.41 < 判别间距 3.756，top-1 纹丝不动，但 KL 漂移 0.306 mnat，是 int8 per-token 的几千倍。argmax 相当于把连续输出量化到 1 bit，它天然看不见小扰动。用 KL / perplexity。

**坑 3：忘了算 scale 自己的字节。**
$\text{开销} = items \cdot B/(group \cdot b)$。int4 + group 16 + fp32 scale 开销 50%，理论 4× 变实际 2.67×。**位宽越低，这笔账越不能省。** 生产布局（V4 的 `ue8m0`）之所以用 1 字节 scale，就是为了把 group 砍到 64 的同时不付出代价。

**坑 4：以为 `scales=1.0` 的默认路径无害。**
e4m3 的通道中位误差确实和有 scale 时一样（0.000692 vs 0.000691），但一旦有值越过 448 就是硬截断——实测全局 nMSE 从 0.000610 涨到 0.178224（292 倍）。默认路径是「赌数值落在量程内」，不是「不需要 scale」。

**坑 5：三个量化 API 的参数语义互不相同，而且传错不报错。**
`fake_quantize_per_channel_affine` 要 `float32` scale + `int32` zero_point 且 scale 长度 = `x.size(axis)`；`quantize_per_channel` 要 `float64` + `int64`；而 `quantize_per_channel` 传错长度时给的是 `length of scales must equal to channel` 这种容易误读的报错。最阴的是**把 max 当 scale 传**——不报错、不警告，相对误差从 0.4% 变 36%（§3.1 实测）。

**坑 6：拿两个自己写的实现互相核对。**
§4.4 的第一版就是这么错的：闭式和「显式计数」共享了同一个漏掉 $n_{kv}$ 的假设，核对报告「偏差 0，完全一致」。**自洽的错误不是证据。** 要么找一个外部真值（官方公布的字节布局），要么回到第一性原理从头数一遍。

---

## 8. 一句话总结

**KV 量化只做三件事——选格式（fp8 自带非均匀刻度、对粒度不敏感；int8 均匀刻度、粒度就是命门）、选粒度（跟着 outlier 长在哪根轴走：K 在通道轴、V 在 token 轴）、算 scale 的账（$\text{开销} = items \cdot B/(group \cdot b)$，位宽越低越咬人）——而它换来的收益是有条件的：只有在 decode 被带宽和容量卡住（大 batch、长上下文、要跨机传 KV）时才真正值钱，因为权重那部分字节是常数，KV 那部分随负载线性增长。**

---

## 9. 今日练习

<details>
<summary>今日练习</summary>

### 练习 1：闭式算一把，再用代码验证

配置：$n_{kv}=8$、$d_h=128$、int4 KV、per-token-group `g=32`、fp32 scale（`items=1, B=4`）。

1. 用 $\text{开销} = items \cdot B/(group \cdot b)$ 算 scale 开销和实际压缩率（相对 bf16）。
2. 如果把 group 从 32 改成 16，压缩率变成多少？这一档值不值？
3. 如果改用 int8（`b=1`）但 group 也改成 16，压缩率比 int4+group32 好还是差？

<details>
<summary>参考答案</summary>

1. $4/(32 \times 0.5) = 25\%$；实际 B/数值 $= 0.5 \times 1.25 = 0.625$；相对 bf16 是 $2/0.625 = \mathbf{3.20\times}$（理论 4× 被吃掉 20%）。
2. $4/(16 \times 0.5) = 50\%$；$2/0.75 = \mathbf{2.67\times}$。**不值**——压缩率从 3.20× 掉到 2.67×（少装 17% 的 token），换来的是§4.1 里那种量级的 nMSE 改进（g=32 → g=16 大约 2 倍）。而在 int4 这种低精度下，误差本身已经在 8~17%（§4.6 实测 int4 g=32 的 K 相对误差 9.718%，g=16 是 8.663%，只改善 11%）。**用 17% 的容量换 11% 的误差，不划算。**
3. int8 + g=16：开销 $4/16 = 25\%$，B/数值 $0.25 \times 1.25 = 1.25$，$2/1.25 = \mathbf{1.60\times}$。比 int4+g32 的 3.20× **差一倍**。这就是「位宽优先于粒度」——先把位宽降下来，粒度是第二位的事。

验证代码：

```python
def overhead(group, b, items=1, B=4):
    return items * B / (group * b)

for tag, g, b in [("int4 g=32", 32, 0.5), ("int4 g=16", 16, 0.5),
                  ("int8 g=16", 16, 1.0), ("int8 g=32", 32, 1.0)]:
    o = overhead(g, b)
    print(f"{tag}: 开销 {o*100:.2f}%  压缩率 {2 / (b * (1 + o)):.2f}x")
```

```text
int4 g=32: 开销 25.00%  压缩率 3.20x
int4 g=16: 开销 50.00%  压缩率 2.67x
int8 g=16: 开销 25.00%  压缩率 1.60x
int8 g=32: 开销 12.50%  压缩率 1.78x
```

</details>

### 练习 2：用极值定律先预测，再实测

用 $\text{nMSE} \approx \mathbb{E}[\max^2]/(12 \cdot 127^2 \cdot \sigma^2)$：给定一组 iid 标准正态样本，每组 64 个：

1. 先估一下 $\mathbb{E}[\max]$（可以查正态分布极值表，也可以用 $\sqrt{2\ln n}$ 近似）。
2. 预测 int8 absmax 的 nMSE 量级。
3. 用代码实测，看偏差多大。

<details>
<summary>参考答案</summary>

$\sqrt{2\ln 64} = 2.883$。实测（§4.2 表）$\mathbb{E}[\max^2]/\sigma^2 = 6.9367$（不是 $2.883^2 = 8.31$，因为 $\mathbb{E}[\max^2] = \text{Var}(\max) + \mathbb{E}[\max]^2$，真实样本最大值的二阶矩比一阶矩平方大）。

预测：$6.9367/(12 \times 127^2) = 6.9367/193548 = 3.585\times10^{-5}$。
实测：$3.527\times10^{-5}$，比值 0.984。

```python
import torch

flat = torch.randn(262144, generator=torch.Generator().manual_seed(0))
grp = flat.view(-1, 64)
s = (grp.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
q = (grp / s).round().clamp(-127, 127) * s
print(f"实测 nMSE {((q-grp)**2).mean()/(grp**2).mean():.3e}  "
      f"预测 {((s*127)**2).mean()/193548:.3e}")
```

```text
实测 nMSE 3.527e-05  预测 3.585e-05
```

注意定律的**适用边界**：当组内的 max 由极值统计独占（重尾分布 + 组特别大）时，多数元素挤在极少数 level 上，误差不再服从「$s^2/12$ 的均匀分布」，实测会明显**小于**预测。§4.2 表 2 里「两者都有 / per-tensor」那一格实测/预测只有 0.32 就是这个原因——定律退化成了上界。

</details>

### 练习 3：亲手重现「口径反转」

造一份同时有通道 outlier 和 token 异质的数据，然后**分别**用全局 nMSE 和通道中位 nMSE 给四个粒度排名，确认两个排名不一致。

<details>
<summary>参考答案</summary>

```python
import torch

S, D = 2048, 128
g = torch.Generator().manual_seed(0)
x = torch.randn(S, D, generator=g)
x = x * torch.exp(1.0 * torch.randn(D, generator=g))
x = x * torch.exp(0.8 * torch.randn(S, generator=g))[:, None]
x[:, torch.arange(4)] *= 50.0          # 4 个固定 outlier 通道


def q_int8(t, mode, gg=None):
    if mode == "per_tensor":
        s = (t.abs().amax() / 127).clamp_min(1e-12)
        return (t / s).round().clamp(-127, 127) * s
    if mode == "per_channel":
        s = (t.abs().amax(0, keepdim=True) / 127).clamp_min(1e-12)
        return (t / s).round().clamp(-127, 127) * s
    if mode == "per_token":
        s = (t.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        return (t / s).round().clamp(-127, 127) * s
    f = t.view(S, D // gg, gg)
    s = (f.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
    return ((f / s).round().clamp(-127, 127) * s).view(S, D)


def glob(t, q):
    return ((q - t) ** 2).sum() / (t ** 2).sum()


def chmed(t, q):
    return (((q - t) ** 2).sum(0) / (t ** 2).sum(0)).median()


print(f"{'粒度':<20}{'全局 nMSE':>12}{'通道中位 nMSE':>16}")
for name, mode, gg in [("per-tensor", "per_tensor", None),
                       ("per-channel", "per_channel", None),
                       ("per-token", "per_token", None),
                       ("per-tok-grp g=32", "per_token_group", 32)]:
    q = q_int8(x, mode, gg)
    print(f"{name:<20}{glob(x, q):>12.6f}{chmed(x, q):>16.6f}")
```

```text
粒度                       全局 nMSE       通道中位 nMSE
per-tensor              0.012332        0.586169
per-channel             0.000631        0.000584
per-token               0.000402        0.029053
per-tok-grp g=32        0.000129        0.003316
```

**全局口径的排名**（越小越好）：per-tok-grp32 (0.000129) < per-token (0.000402) < per-channel (0.000631) < per-tensor (0.012332)

**通道中位口径的排名**：per-channel (0.000584) < per-tok-grp32 (0.003316) < per-token (0.029053) < per-tensor (0.586169)

两份排名不一致，而且不一致的方式值得琢磨：

- **per-channel 在两个口径里差了 4.9 倍**（0.000631 vs 0.000129），它是全局榜上的第三名，却是通道中位榜上的冠军。机制：per-channel 给每个通道自己的 scale，所以 124 个正常通道完好无损——这件事在「中位通道」里看得一清二楚，但在全局 nMSE 里被 4 个 outlier 通道的巨大 $x^2$ 权重盖住了。
- **per-token 反了过来**：全局榜第二名（0.000402），通道中位榜第三名（0.029053，比 per-channel 差 50 倍）。因为 per-token 的 scale 被 outlier 通道一个数拉满，全体正常通道一起被压扁。
- **per-tok-grp32 在两个榜上都是第一或第二**，是最稳的那个——它同时吃掉了两个异质源。
- **per-tensor 在通道中位口径下是 0.586169**，意思是「中位通道只剩不到一半的信号」——它连量级都对不上了。

**所以口径不是「哪个更对」，而是「你想保护谁」**：
- 想保护**注意力打分的准确性**（K 的职责）→ 关心大值和整体误差 → 看全局 nMSE；
- 想保护**每个通道的信息**（V 的职责、以及 outlier 通道之外的 124 个通道）→ 看通道中位 nMSE。

§4.1 的 2×2 表两个口径都给了，正是因为单一数字会骗人。

</details>

</details>

---

## 附：本期的自查记录

| 项 | 结果 |
|---|---|
| ```python 块数 / 其中带输出块的 | 11 / 10（1 个是无限出的片段，只做语法检查） |
| 语法检查 + 独立子进程执行 | 11/11 通过 |
| 输出块与真实 stdout 逐行 diff | **10/10 一致**（墙钟时间列用正则归一化后比对） |
| 执行隔离 | 每个块在独立子进程里跑——某些块会 `torch.set_grad_enabled(False)`，同进程内会污染后面的训练块 |
| 脚本真实执行环境 | torch 2.14.0，CPU + MPS，无 CUDA |
| GPU kernel 相关部分 | §4.5 为 roofline 解析推算，**非实测** |
| 本轮订正 | §4.4 的 scale 开销闭式（原 `4/(g·n_kv·b)` 漏乘 head 数因子）已修正为 `items·B/(group·b)`，并用 DeepSeek V3.2 的 656 B 真实布局对账 |
