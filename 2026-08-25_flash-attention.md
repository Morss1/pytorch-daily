# PyTorch 每日一课 · 第 005 期

## Flash Attention：让注意力计算不再被内存拖后腿

| 项目 | 内容 |
|---|---|
| 日期 | 2026-08-25 |
| 难度 | ⭐⭐⭐⭐ |
| 前置知识 | Transformer 的 Self-Attention 机制、GPU 内存层次结构（HBM vs SRAM）、了解 `nn.MultiheadAttention` |
| 预计阅读时间 | 15 分钟 |

---

## 这个领域解决什么问题

Transformer 的 Self-Attention 有一个看似简单的公式：

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
```

但当序列长度 `N` 变大时（比如 4K、8K、甚至 100K），这个「简单」的计算背后藏着巨大的工程瓶颈。

### 瓶颈不在算力，而在「搬运数据」

想象你在一个超级工厂里做组装（GPU 计算核心），你的原材料都放在一个很远的仓库（HBM，显存主存）。标准 Attention 的做法是：

1. 先从仓库搬来 Q、K；
2. 在操作台上算出 `QK^T`，得到一张巨大的 `N×N` 得分表；
3. 把这张表运回仓库存起来；
4. 再搬回来做 softmax；
5. 再搬去和 V 做矩阵乘法；
6. 最终结果再运回仓库。

问题是：这张 `N×N` 的得分表可能大得离谱——当 `N = 8192` 时，光是这张表就要 **256 MB**（FP32）。而且每次读写 HBM 的速度（约 1.5 TB/s）远远跟不上计算核心吞吐需求。这就像是法拉利发动机被堵在了早高峰的乡间小路上。

> **Flash Attention 的核心洞察**：Attention 的瓶颈不是 FLOPs（浮点运算次数），而是 **HBM 读写带宽**。只要减少 HBM 访问，就能大幅提速。

---

## 核心思想：在 SRAM 上「边算边丢」

GPU 的内存是分层的：

| 层级 | 容量 | 带宽 | 类比 |
|---|---|---|---|
| HBM（高带宽显存）| 几十 GB | ~1.5 TB/s | 远郊大仓库 |
| SRAM（共享内存/缓存）| 每 SM 几十 KB ~ 几百 KB | ~10+ TB/s | 手边工作台 |

Flash Attention 的策略很简单：

> **不要把整张 `N×N` 的注意力矩阵写到 HBM 里。把它切成一小块一小块（tile），在 SRAM 上算完 softmax 和乘 V 后，只把最终结果写回 HBM。**

### 类比：流水线上的即时结账

想象超市收银台：

- **标准 Attention**：先把所有商品（Q、K）搬到后台，算总价（`QK^T`），把总价单贴墙上（写入 HBM），再拿这张单子去结账（softmax × V）。
- **Flash Attention**：商品从传送带直接过扫描仪，扫一件算一件，最后只打印一张小票（输出结果），不保留中间清单。

### 技术实现：Tiling + 在线 Softmax

标准的 softmax 需要看到一整行才能算出正确的分母（`sum(exp(x_i - max))`）。Flash Attention 用了一个技巧：**在线 softmax**——维护一个「running max」和「running sum」，每读入一块新 tile，就更新这两个统计量，逐步逼近整行的真实 softmax。

反向传播时，Flash Attention 选择**不保存中间注意力矩阵**，而是重新计算它。这听起来很浪费，但因为是在 SRAM 上重算，读写代价远低于从 HBM 加载一张巨大的 `N×N` 矩阵。最终结果是：**更少内存、更快速度**——而不是传统的「用时间换空间」。

---

## 在 PyTorch 中怎么用

从 PyTorch 2.0 开始，`torch.nn.functional.scaled_dot_product_attention`（简称 SDPA）已经内置了对 Flash Attention 的支持。你通常不需要手动安装任何库。

### 基础用法

```python
import torch
import torch.nn.functional as F

# 构造输入：batch=2, heads=8, seq_len=4096, head_dim=64
batch, n_heads, seq_len, head_dim = 2, 8, 4096, 64

torch.manual_seed(42)
q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

# ============================================================
# 方法 1：直接用 SDPA（推荐，PyTorch 会自动选择最优后端）
# ============================================================
with torch.cuda.amp.autocast(dtype=torch.float16):
    out = F.scaled_dot_product_attention(q, k, v)

print("SDPA 输出形状:", out.shape)  # [2, 8, 4096, 64]

# ============================================================
# 方法 2：显式查看当前启用了哪些后端
# ============================================================
print("Flash SDP 启用:", torch.backends.cuda.flash_sdp_enabled())      # True/False
print("Math SDP 启用: ", torch.backends.cuda.math_sdp_enabled())       # 兜底实现
print("Mem-Efficient SDP:", torch.backends.cuda.mem_efficient_sdp_enabled())

# ============================================================
# 方法 3：强制只用 Flash Attention（用于对比测试）
# ============================================================
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,          # 关闭纯数学实现
    enable_mem_efficient=False  # 关闭 memory-efficient 实现
):
    out_flash = F.scaled_dot_product_attention(q, k, v)

# ============================================================
# 方法 4：因果掩码（Causal Mask，用于 Decoder 的自回归生成）
# ============================================================
out_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 性能对比：感受 Flash Attention 的加速

```python
import torch
import torch.nn.functional as F
import time

def benchmark(name, fn, q, k, v, warmup=5, repeats=10):
    """简单的 CUDA 计时器"""
    # 热身
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    # 正式计时
    start = time.perf_counter()
    for _ in range(repeats):
        fn(q, k, v)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats * 1000
    print(f"{name}: {elapsed:.3f} ms/iter")
    return elapsed

# 构造长序列输入
batch, n_heads, seq_len, head_dim = 2, 12, 8192, 64
dtype = torch.float16
device = "cuda"

q = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)
k = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)
v = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)

# 强制走不同的后端
def run_flash(q, k, v):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        return F.scaled_dot_product_attention(q, k, v)

def run_math(q, k, v):
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        return F.scaled_dot_product_attention(q, k, v)

def run_memeff(q, k, v):
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
        return F.scaled_dot_product_attention(q, k, v)

benchmark("Flash Attention", run_flash, q, k, v)
benchmark("Math (标准实现)", run_math, q, k, v)
benchmark("Memory-Efficient", run_memeff, q, k, v)
```

在 Ampere 架构（A100、RTX 3090/4090 等）上，你通常会看到：

- **Flash Attention** 比标准实现快 **2~8 倍**（序列越长，优势越大）；
- 同时显存占用从 `O(N²)` 降到近似 `O(N)`。

### 使用 `flash-attn` 官方库（需要更细粒度控制时）

```python
# 需要先安装：pip install flash-attn --no-build-isolation
from flash_attn import flash_attn_func

# flash-attn 库直接暴露底层接口
out = flash_attn_func(q, k, v, causal=True)
# 返回形状: [batch, seq_len, n_heads, head_dim]
# 注意：它的输入布局是 [batch, seq_len, n_heads, head_dim]（不是 [batch, n_heads, seq_len, head_dim]）
```

---

## 围绕该领域展开

Flash Attention 不是孤立的算子优化，它是一整套「IO-Aware 算法设计」思想的体现。理解它，能帮你串起一大片相关概念：

### 1. 与 KV Cache 的关系

大模型推理时，Decoder 每一步只生成一个新 token，但 Q、K、V 中的 K 和 V 可以缓存下来复用。这就是 **KV Cache**。Flash Attention 在推理阶段配合 KV Cache 使用，能显著降低长文本生成的延迟。没有 Flash Attention，KV Cache 增长到一定长度后，单次 attention 计算会因为 HBM 瓶颈而急剧变慢。

### 2. 与 PagedAttention（vLLM）的关系

vLLM 提出的 **PagedAttention** 进一步解决了 KV Cache 的显存管理问题：不同请求的 KV Cache 被切成固定大小的「页」（page），像操作系统的虚拟内存一样按需分配。Flash Attention 解决了单个请求内部的 attention 计算效率，PagedAttention 解决了多请求并发时的 KV Cache 显存碎片和管理问题。**两者结合**，才构成了现代大模型推理服务（如 vLLM、TensorRT-LLM）的性能底座。

### 3. Flash Attention v1 vs v2

| 特性 | v1 | v2 |
|---|---|---|
| 并行维度 | 按 batch×head 并行 | 额外按序列维度并行，减少空闲 warp |
| 速度 | 快 | 更快（通常 v2 比 v1 快 1.5~2 倍）|
| 显存占用 | 低 | 更低 |
| PyTorch 内置 | SDPA 支持 | SDPA 支持（PyTorch 2.1+）|

核心改进：v2 把 attention 的「计算-通信」模式重新排布，让每个 warp（GPU 最小调度单元）更饱和，减少了线程空闲等待。

### 4. 与序列并行（Sequence Parallelism）的关系

当序列长度长到单卡都放不下时（比如 100K+），只靠 Flash Attention 也不够。**序列并行**把输入序列切成多段，分散到不同 GPU 上。Flash Attention 负责单卡内的 tile 计算，序列并行负责跨卡的序列拆分——两者层级互补。Meta 的 LLaMA 3 长文本训练就同时用到了这两者。

### 5. 与 Ring Attention 的关系

**Ring Attention** 是 Flash Attention 的进一步扩展：它把整个 KV Cache 组织成一个环形队列，每块 GPU 只持有一部分，计算时像传接力棒一样传递 K、V 块。这使得理论上可以处理**无限长**的序列（只受通信带宽限制，不受单卡显存限制）。Flash Attention 是 Ring Attention 单节点内的计算核心。

### 6. 与 Triton / CUDA Kernel 的关系

Flash Attention 的原始实现是手写 CUDA Kernel。后来社区也出现了基于 Triton（OpenAI 的 Python 级 GPU 编程语言）的实现。PyTorch 的 SDPA 底层会根据硬件自动 dispatch 到最合适的 kernel——可能是 CUDA 手写的，也可能是 Triton 生成的。理解 Flash Attention 有助于你明白：**为什么 `torch.compile` + SDPA 的组合在长序列上表现优异**——因为 compile 能融合周围的算子，SDPA 内部又用 Flash Attention 消除了 HBM 瓶颈。

### 7. 与稀疏注意力（Sparse Attention）的关系

Flash Attention 解决的是「标准稠密 attention 的内存带宽问题」，它不改变 `O(N²)` 的计算复杂度。如果序列真的长到 `N²` 都受不了，就需要 **稀疏注意力**（如 Sliding Window、Longformer、BigBird）来降低计算量到 `O(N)` 或 `O(N log N)`。两者的取舍：

- Flash Attention：保持全连接、精度无损、硬件友好；
- 稀疏 Attention：改变 attention 模式、可能损失精度、更长的序列上限。

### 8. 与量化（Quantization）的关系

KV Cache 量化（如 KV-Cache INT8/FP8）和 Flash Attention 是正交的优化方向：Flash Attention 减少内存访问量，量化减少每个数据占用的字节数。**两者叠加**，可以在保持速度的同时把 KV Cache 显存压到原来的 1/4 甚至更低。

---

## 什么时候该用 / 不该用

### ✅ 该用 Flash Attention 的时候

- **长序列训练**：当 `seq_len > 1024` 时，Flash Attention 的收益开始明显；`seq_len > 4096` 时，几乎是必选项；
- **大 batch 推理**：batch size 越大，HBM 访问压力越大，Flash Attention 的收益越显著；
- **显存紧张的场景**：因为它不保存中间 `N×N` 注意力矩阵，峰值显存占用大幅降低；
- ** causal（自回归）生成**：`is_causal=True` 时 Flash Attention 有专门的优化路径。

### ❌ 不该用或要小心的时候

- **短序列（`seq_len <= 512`）**：tile 切分的 overhead 可能抵消收益，甚至略慢；
- **需要提取注意力权重矩阵**：Flash Attention 不输出 `N×N` 的 attention score。如果你需要可视化注意力热力图、做 attention 分析，需要用标准实现或者 `flash-attn` 库的特定接口；
- **head_dim 不兼容**：Flash Attention 对 head_dim 有特定要求（通常是 64、128 等），如果你的模型用了奇怪的 head_dim（比如 80），SDPA 会自动 fallback 到其他实现；
- **需要确定性 backward + dropout**：Flash Attention 的反向在某些组合下可能不支持完全确定性（ determinism）。

---

## 常见坑

### 坑 1：以为用了 SDPA 就一定在跑 Flash Attention

SDPA 会自动选择后端。如果硬件不支持（非 Ampere 架构）、数据类型不对（比如 FP64）、head_dim 不匹配，它会默默 fallback 到 math 或 memory-efficient 实现。要确认，用上面的 `torch.backends.cuda.sdp_kernel` 显式控制，或者用 benchmark 验证速度。

### 坑 2：`is_causal=True` 和手动传入 `attn_mask` 的混淆

```python
# ❌ 不要这样做：手动构造一个巨大的下三角 mask
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)  # mask 可能占很多显存

# ✅ 正确做法：直接传 is_causal
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # 内部优化，不分配大 mask
```

手动传 `attn_mask` 时，如果 mask 是 `N×N` 的稠密矩阵，本身就违背了 Flash Attention 省内存的初衷。`is_causal=True` 是特殊路径，不会分配额外内存。

### 坑 3：推理时没注意到 KV Cache 和 Flash Attention 的 layout 差异

有些框架（如 HuggingFace）在调用 Flash Attention 时需要把 Q、K、V 转成特定的内存布局（比如 `[batch, seq, n_heads, head_dim]` 而非 `[batch, n_heads, seq, head_dim]`）。如果你的推理代码直接 copy 了训练时的 attention 逻辑，可能会遇到 shape 不匹配或性能下降。

### 坑 4：反向传播时的重计算被误认为是 bug

```python
out = F.scaled_dot_product_attention(q, k, v)
loss = out.sum()
loss.backward()
```

有人发现 backward 时显存占用和 forward 差不多，而不是像普通算子那样只增加一点梯度内存。这是因为 Flash Attention 的 backward 会**重计算 forward 的中间结果**（在线 softmax 的统计量），而不是从 HBM 加载。这是设计如此，不是泄漏。

---

## 一句话总结

> Flash Attention 不是让 Attention 算得更快，而是让 Attention 不再被「搬数据」拖后腿。它通过在 GPU SRAM 上 tile 化计算、在线 softmax、以及反向重计算，把 Attention 从 HBM 带宽瓶颈中解放出来——这是长序列 Transformer 训练与推理的必备基石。

---

## 今日练习

下面这段代码分别在「强制 Flash」和「强制 Math」两种模式下运行 SDPA。请在具有 Ampere 及以上架构的 GPU 上运行，并回答：当序列长度从 1024 增加到 8192 时，两种模式的耗时比例如何变化？为什么？

```python
import torch
import torch.nn.functional as F
import time

def measure(seq_len, repeats=20):
    b, h, d = 2, 8, 64
    q = torch.randn(b, h, seq_len, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, seq_len, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, seq_len, d, device="cuda", dtype=torch.float16)

    def run_flash():
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v)

    def run_math():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v)

    for fn, name in [(run_flash, "Flash"), (run_math, "Math")]:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / repeats * 1000
        print(f"  {name}: {ms:.2f} ms")

for sl in [1024, 2048, 4096, 8192]:
    print(f"seq_len={sl}")
    measure(sl)
```

<details>
<summary>参考答案</summary>

**预期现象**：随着序列长度增加，Math 模式的耗时增长接近 `O(N²)`，而 Flash Attention 的增长曲线明显更平缓。在 `N=8192` 时，Flash 通常比 Math 快 3~8 倍。

**原因**：

1. **Math 模式**需要把完整的 `N×N` 注意力得分矩阵写入 HBM，再读回来做 softmax 和乘 V。序列翻倍时，HBM 读写量近似翻 4 倍（`QK^T` 是 `N×N`，乘 V 也是 `N×N` 相关的规模），而 HBM 带宽是固定的，所以时间近似 `O(N²)`。

2. **Flash Attention** 在 SRAM 上逐 tile 计算，不写入中间 `N×N` 矩阵到 HBM。它的瓶颈变成了计算本身（FLOPs），而计算核心的吞吐远大于 HMB 带宽。因此增长更接近线性（虽然理论 FLOPs 仍然是 `O(N²)`，但常数因子和内存墙的问题被消除了）。

3. 在短序列（如 1024）时，Flash Attention 的 tile 切分 overhead 可能让优势不那么明显；但序列越长，节省的 HBM 访问越多，优势越压倒性。

</details>
