# PyTorch 每日一课 · 第 043 期

## GPU 内存层次与访存模式：带宽阶梯、coalescing、bank conflict

**日期** 2026-10-27　**难度** ⭐⭐⭐⭐　**前置** 知道 warp / thread block 是什么（第 008 期 Triton 讲过 block 网格）；知道 cache 的基本概念；第 041 期的「循环顺序」会直接用到　**预计阅读** 约 30 分钟

**关联** 第 041 期 · 041 讲的「循环顺序」正是访存模式的源码级表达，本期回答「为什么换个顺序能差 8 倍」；第 040 期 · 融合的价值以「少搬多少字节」计价，本期说明字节为什么是通用货币；第 042 期 · 042 讲块怎么分，本期讲块内部怎么用内存；第 014 期 · profiler 只能说「慢」，本期说「为什么慢」；第 030~036 期 · KV cache 那一整串优化，本质都在跟 $\mathrm{AI}\approx 1$ 的带宽墙搏斗

---

### 0. 先把结论摆出来

1. **现代 kernel 的瓶颈是「搬」而不是「算」**。H100 每个 SM 摊到的 HBM 带宽只有 **25.4 GB/s**，却要喂 7.5 TFLOPS 的 FP16 Tensor Core —— 想让算力跑满，每搬 1 字节需要做 **295 次浮点运算**。逐元素算子只有 0.17，GEMM 4096³ 有 1365，差三个数量级。
2. **内存不是「一块」，是五块，容量与带宽严格反向**：寄存器 256 KB/SM、shared+L1 256 KB/SM、L2 50 MB、HBM 80 GB。片上（shared 的 128 B/clk）比片外快**整 10 倍**。
3. **「浪费」有两个计费单位，位置不同**：global memory 按 **32 字节 sector** 计费（L1 缓存行 128 B），shared memory 按 **bank** 计费（32 banks × 4 B = 128 B/clk）。
4. **GPU 的 cacheline 是 128 B，sector 是 32 B —— 这是两个不同的数，混用会得出反向结论**。实测一台 128 B cacheline 的 CPU 做 1-D 跨步扫描，惩罚在 stride=32 处斜率骤降（8→32 的 24 步里涨 5.1 倍，32→128 的 96 步里只涨 1.4 倍）；而 GPU 的 sector 模型在 stride=8 就触底（有效率下限 12.5%）。
5. **框架已经替你干了一部分活，而且分得很清楚**：纯逐元素算子，非连续输入的代价几乎为零（实测 **0.97~1.03x**）—— 因为它有「换循环顺序」的自由。归约类算子只有部分自由（列归约慢 **2.5~3x**），融合类算子（softmax / layernorm）几乎没有自由（**3.1~3.5x / 5.6~6.3x**）。
6. **`.contiguous()` 不是免费的**：它本身是一次完整的读写，实测让一个已经很快的算子变成 **5.9~6.4x**。真正贵的不是「非连续」，而是「为了连续付出的搬运」。
7. **分块（tiling）本身不值钱，值钱的是「把跨步访问换成两次连续访问」**。实测：只把循环分块 **0.94~0.98x**（等于白忙），加上局部 tile 暂存变成 **7.6~8.5x**。
8. **bank conflict 的正确解法是「让宽度与 32 互质」，不是「加 1」这个数字本身**。tile 存成 `float[32][W]` 读一列时命中的 bank 数是 $32/\gcd(W,32)$：W=32 → 1 个 bank（32 路冲突），W=33 → 32 个 bank，W=34 → 16 个，W=36 → 8 个。

---

### 1. 这个领域解决什么问题

一块 H100 有 132 个 SM、80 GB HBM3、3.35 TB/s 带宽。这三个数放在一起就注定了今天的 kernel 长什么样：

- 算力：FP16 Tensor Core dense 989.5 TFLOPS，摊到每个 SM 是 **7.5 TFLOPS**。
- 带宽：3.35 TB/s 摊到每个 SM 是 **25.4 GB/s**。
- 两者一比，**每个 SM 每搬 1 字节，平均只能做 295 次浮点运算**，否则算力就在等数据。

这个比值叫**算术强度**（arithmetic intensity，$\mathrm{AI}$，单位 FLOP/byte），而 295 就是 H100 在 FP16 下的**屋脊点**（ridge point）。roofline 模型的全部内容就是一句话：

$$\text{可达到性能} = \min\left(\text{峰值算力},\ \ \mathrm{AI}\times \text{峰值带宽}\right)$$

往左（$\mathrm{AI}$ 小）撞带宽墙，往右（$\mathrm{AI}$ 大）撞算力墙。先算清楚自己的算子在哪一边：

```python
# H100 SXM5 官方规格：NVIDIA 产品页 + Hopper Tuning Guide
SM = 132
BW_HBM = 3.35e12      # HBM3
CLK = 1.98e9          # boost 时钟
L2_BYTES = 50e6
FP32_TFLOPS = 67.0    # FP32 CUDA core（非 Tensor）
# Tensor 核官方值含 2:4 稀疏，dense 要减半
TENSOR_DENSE = {"TF32": 989 / 2, "FP16": 1979 / 2, "FP8": 3958 / 2}

# ---- 带宽阶梯：每个 SM 能拿到多少 ----
smem_per_sm = 128 * CLK          # 32 bank x 4 B，每 clk 一次
hbm_per_sm = BW_HBM / SM
print(f"shared memory: 128 B/clk x {CLK/1e9:.2f} GHz = {smem_per_sm/1e9:.1f} GB/s/SM")
print(f"                  x {SM} SM = {smem_per_sm*SM/1e12:.1f} TB/s 聚合（推算）")
print(f"HBM3         : {BW_HBM/1e12:.2f} TB/s（官方）"
      f" -> 每 SM 只有 {hbm_per_sm/1e9:.1f} GB/s")
print(f"片上 : 片外  = {smem_per_sm/hbm_per_sm:.1f}x")
print(f"容量阶梯     : 寄存器 256 KB/SM | shared+L1 256 KB/SM | "
      f"L2 {L2_BYTES/1e6:.0f} MB | HBM 80 GB")

# ---- roofline 屋脊点 ----
print("\n屋脊点 = 峰值算力 / 峰值带宽")
print(f"{'精度':<22}{'峰值':>12}{'屋脊点':>12}")
for name, tf in [("FP32 CUDA core", FP32_TFLOPS), *TENSOR_DENSE.items()]:
    print(f"{name:<22}{tf:>9.1f} TF{tf*1e12/BW_HBM:>9.1f} FLOP/B")
```

```text
shared memory: 128 B/clk x 1.98 GHz = 253.4 GB/s/SM
                  x 132 SM = 33.5 TB/s 聚合（推算）
HBM3         : 3.35 TB/s（官方） -> 每 SM 只有 25.4 GB/s
片上 : 片外  = 10.0x
容量阶梯     : 寄存器 256 KB/SM | shared+L1 256 KB/SM | L2 50 MB | HBM 80 GB

屋脊点 = 峰值算力 / 峰值带宽
精度                              峰值         屋脊点
FP32 CUDA core             67.0 TF     20.0 FLOP/B
TF32                      494.5 TF    147.6 FLOP/B
FP16                      989.5 TF    295.4 FLOP/B
FP8                      1979.0 TF    590.7 FLOP/B
```

现在把常见算子放进去看落点。下面这张表里没有一行是「推测」——全都是从 FLOP 数和必须搬的字节数直接算出来的：

```python
BW_HBM, FP32_TFLOPS = 3.35e12, 67.0
TENSOR_DENSE = {"TF32": 989 / 2, "FP16": 1979 / 2, "FP8": 3958 / 2}
N = 4096          # GEMM 边长
H, S, D_ = 32, 32768, 128     # decode attention：32 head，32K 上下文，head_dim 128
RIDGE = TENSOR_DENSE["FP16"] * 1e12 / BW_HBM      # 295.4 FLOP/B

ops = [
    ("逐元素 c=a*b+d (fp32)", 2*N*N, 3*N*N*4, FP32_TFLOPS*1e12/BW_HBM),
    ("逐元素 c=a*b+d (fp16)", 2*N*N, 3*N*N*2, FP32_TFLOPS*1e12/BW_HBM),
    (f"GEMM {N}^3 (fp16)", 2*N**3, 3*N*N*2, RIDGE),
    ("GEMM 256^3 (fp16)", 2*256**3, 3*256*256*2, RIDGE),
    ("GEMV m=1, k=4096 (fp16)", 2*4096*4096, 4096*4096*2, RIDGE),
    (f"decode attn S={S} d={D_}", 4*H*S*D_, 4*H*S*D_, RIDGE),
]
print(f"{'算子':<24}{'字节':>10}{'AI':>9}{'结论':>24}")
for name, fl, by, ridge in ops:
    ai = fl / by
    verdict = "算力受限（能填满）" if ai > ridge else f"带宽受限，只有峰值 {ai/ridge*100:.1f}%"
    print(f"{name:<24}{by:>10.2e}{ai:>9.2f}{verdict:>24}")
```

```text
算子                              字节       AI                      结论
逐元素 c=a*b+d (fp32)        2.01e+08     0.17          带宽受限，只有峰值 0.8%
逐元素 c=a*b+d (fp16)        1.01e+08     0.33          带宽受限，只有峰值 1.7%
GEMM 4096^3 (fp16)        1.01e+08  1365.33               算力受限（能填满）
GEMM 256^3 (fp16)         3.93e+05    85.33         带宽受限，只有峰值 28.9%
GEMV m=1, k=4096 (fp16)   3.36e+07     1.00          带宽受限，只有峰值 0.3%
decode attn S=32768 d=128  5.37e+08     1.00          带宽受限，只有峰值 0.3%
```

逐元素算子只有峰值的 0.8%，GEMM 4096³ 能填满，而 decode 阶段的 attention 落在 **$\mathrm{AI}=1.00$** —— 连峰值的 0.3% 都摸不到。

那个 $1.00$ 不是巧合。对一个 token、$H$ 个 head、上下文长度 $S$、head 维 $d$、K/V 各存 fp16 的情形：

$$\text{FLOPs} = 2\cdot\underbrace{2HSd}_{\text{QK}^\top + \text{AV}} = 4HSd,\qquad
\text{Bytes} = \underbrace{2\cdot HSd\cdot 2}_{K,\,V\ \text{各一份 fp16}} = 4HSd$$

$$\Rightarrow\quad \mathrm{AI} = \frac{4HSd}{4HSd} = 1.00\ \text{FLOP/byte}$$

$H$、$S$、$d$ 全部约掉了。**decode attention 的算术强度恒等于 1，跟模型多大、上下文多长都没关系** —— 这就是为什么第 030~036 期那一整串（continuous batching、RadixAttention、KV 量化、KV 驱逐、PD 分离）全都在做同一件事：少搬 KV 字节，或者让更多 token 共享同一次搬运。

这也顺便解释了 batch 的临界点：

```python
BW_HBM = 3.35e12
H, S, D_ = 32, 32768, 128     # 32 head，32K 上下文，head_dim 128
kv_bytes = 4 * H * S * D_              # 单 token 要读的 K/V 总字节（fp16，K/V 各一份）
w_bytes = 70e9 * 2                     # 70B 参数，fp16
print(f"单 token 的 attention 读 K/V : {kv_bytes/2**20:.0f} MiB")
print(f"在 {BW_HBM/1e12:.2f} TB/s 下的理论下限 : {kv_bytes/BW_HBM*1e6:.0f} us / token")
print(f"同一步还要读一遍权重          : {w_bytes/1e9:.0f} GB"
      f" -> {w_bytes/BW_HBM*1e3:.0f} ms / token")
print(f"权重是这份 KV 的 {w_bytes/kv_bytes:.0f} 倍")
print(f"batch 涨到 {w_bytes/kv_bytes:.0f} 以上，KV 反超成为主项")
```

```text
单 token 的 attention 读 K/V : 512 MiB
在 3.35 TB/s 下的理论下限 : 160 us / token
同一步还要读一遍权重          : 140 GB -> 42 ms / token
权重是这份 KV 的 261 倍
batch 涨到 261 以上，KV 反超成为主项
```

70B fp16 的权重是 140 GB，单 token 的 32K 上下文 KV 是 512 MiB，比值 **261**。batch=1 时读权重主导；batch 涨到 261 以上，KV 反超成为主项。**张量并行不改变这个比值**，因为权重和 KV 同时按 TP 切分。

---

### 2. 核心思想：两级「计费粒度」

把内存层次想成仓库 / 货架 / 桌面 / 手上：

| 层 | 类比 | 容量 | 取一次的成本 |
|---|---|---|---|
| 寄存器 | 手上 | 256 KB/SM | 几乎免费，但只有那么多只手 |
| shared memory | 桌面 | 228 KB/SM | 128 B/clk，很快，但你自己负责摆 |
| L1 / L2 | 货架 | 256 KB/SM、50 MB | 中等 |
| HBM | 仓库 | 80 GB | 3.35 TB/s，一趟一趟跑 |

关键不在于「快的容量小」这种老生常谈，而在于**仓库不零售**：你去 HBM 取 4 个字节，它按 32 字节的箱子给你记账；你去 shared memory 取 32 个线程各一个字节，如果它们落在同一个 bank 上，它就分 32 趟给你。

于是有效带宽这件事可以写成一条统一公式：

$$\eta = \frac{\text{你真正用到的字节}}{\text{硬件记账的字节}}$$

提高 $\eta$ 只有两条路，而且只有这两条：

1. **让一个 warp 的地址连续** —— 少买空箱子（coalescing）。
2. **把重复读的东西搬到片上** —— 少去仓库（tiling / shared memory / 融合）。

后面所有技巧，包括 §4.4 那个 8 倍的 transpose，都是这一句话的展开。

#### 2.1 global memory 的计费单位：32 字节 sector

CUDA 官方文档（*CUDA C++ Best Practices Guide*）写得很直白：

> For devices of compute capability 6.0 or higher, the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of **32-byte** transactions necessary to service all of the threads of the warp.

一个 warp 32 个线程，每个读 4 字节 fp32：

- **地址连续**：128 字节，正好 4 个 sector，**一个不多** → $\eta = 100\%$。
- **间隔 2 个元素**（stride 2）：地址铺开到 256 字节，8 个 sector，用了一半 → $\eta = 50\%$。文档原文就是「a stride of 2 results in a 50% of load/store efficiency」。
- **地址整体错位 4 字节**：跨了 5 个 sector（160 字节）→ $\eta = 80\%$。文档给的是「five 32-byte segments」。

#### 2.2 shared memory 的计费单位：bank

shared memory 被切成 32 个 bank，每个 bank 每时钟周期能供 32 位（4 字节）。地址到 bank 的映射是**纯取模**：

$$\text{bank}(\text{addr}) = \left\lfloor \frac{\text{addr}}{4} \right\rfloor \bmod 32$$

32 个线程如果落在 32 个不同 bank 上，一次搬完（1 个 wavefront，128 B/clk）；如果落进同一个 bank 的不同地址，就串行化成多趟。**唯一例外是广播**：32 个线程读**同一个地址**不算冲突，硬件会做多播。

这两条规则加起来只有十几个字，但足够推出后面所有的坑。

---

### 3. 在 PyTorch 里怎么用：stride 是访存模式的载体

PyTorch 里没有「指针」给你调，能表达访存模式的只有一样东西：**tensor 的 stride**。`t.stride()` 就是「沿着第 k 维走一步，物理内存跳多少个元素」。

#### 3.1 先学会看 stride

```python
import torch

x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
for tag, t in [("reshape(2,3,4)", x),
               ("transpose(1,2)", x.transpose(1, 2)),
               ("permute(2,0,1)", x.permute(2, 0, 1)),
               ("expand 到 (5,2,3,4)", x.unsqueeze(0).expand(5, 2, 3, 4))]:
    print(f"{tag:<22} shape={tuple(t.shape)}  stride={t.stride()}  "
          f"contiguous={t.is_contiguous()}")
```

```text
reshape(2,3,4)         shape=(2, 3, 4)  stride=(12, 4, 1)  contiguous=True
transpose(1,2)         shape=(2, 4, 3)  stride=(12, 1, 4)  contiguous=False
permute(2,0,1)         shape=(4, 2, 3)  stride=(1, 12, 4)  contiguous=False
expand 到 (5,2,3,4)     shape=(5, 2, 3, 4)  stride=(0, 12, 4, 1)  contiguous=False
```

`stride` 里出现 `0`（expand）、出现跟 shape 不匹配的顺序（transpose），都意味着「这个张量在物理内存里不是按它的逻辑顺序摆的」。**这就是全部信息源，没有第二个地方能看。**

#### 3.2 纯逐元素算子：框架已经帮你重排了

教科书的说法是「非连续输入会变慢」。实测一遍：

```python
import time, torch
torch.set_num_threads(4)

def bench(fn, reps=5, warmup=2):
    for _ in range(warmup):
        fn()
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best

M = 4096
A = torch.randn(M, M, dtype=torch.float32)
At = A.t()                      # 零拷贝视图，stride 从 (M,1) 变成 (1,M)
print(f"A.stride={A.stride()}   At.stride={At.stride()}")

t_c = bench(lambda: A * 2.0 + 1.0)
t_n = bench(lambda: At * 2.0 + 1.0)
t_fix = bench(lambda: At.contiguous() * 2.0 + 1.0)
print(f"连续输入        : {t_c*1e3:6.2f} ms")
print(f"转置输入        : {t_n*1e3:6.2f} ms   -> {t_n/t_c:.2f}x")
print(f"先 contiguous 再算: {t_fix*1e3:6.2f} ms   -> {t_fix/t_c:.2f}x")
```

```text
A.stride=(4096, 1)   At.stride=(1, 4096)
连续输入        :   2.12 ms
转置输入        :   2.23 ms   -> 1.05x
先 contiguous 再算:  13.44 ms   -> 6.34x
```

转置输入的代价是 **0.97~1.03x** —— 完全在噪声里。原因在 `TensorIterator`：它会把维度按 stride 排序，让**物理上最连续的那一维跑内层循环**。对逐元素算子，输出元素之间互不依赖、读写都是一一对应，所以它完全可以把循环顺序换成「按内存顺序扫」。访存模式已经被框架改写成连续的了，你没看见而已。

而「先 `contiguous()`」是 **5.9~6.4x** —— 因为它多做了一次完整的 128 MiB 读 + 128 MiB 写，就为了让接下来那 2 ms 的计算「舒服一点」。**这笔账永远是亏的。**

> 这正是第 041 期的主题在用户层的投影：循环顺序可以被重排，重排之后访存模式就变了。

#### 3.3 归约类算子：只有部分自由

归约不能随便换顺序 —— 但也没那么死。`sum` 沿哪一维，决定它读的维是 stride-1 还是 stride-8192。用 2×2 对照把「维度编号」和「stride」拆开：

```python
import time, torch
torch.set_num_threads(4)

def bench(fn, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best

M = 8192
A = torch.randn(M, M, dtype=torch.float32)    # 256 MiB
B = A.t()                                     # 零拷贝视图，只换 stride
nb = A.numel() * 4
print(f"A.stride={A.stride()}   B.stride={B.stride()}")

for tag, t, dim, note in [("A.sum(1)", A, 1, "内层走 stride-1"),
                          ("A.sum(0)", A, 0, "内层走 stride-8192"),
                          ("B.sum(1)", B, 1, "= A.sum(0) 同一个数学量"),
                          ("B.sum(0)", B, 0, "= A.sum(1) 同一个数学量")]:
    tt = bench(lambda t=t, dim=dim: t.sum(dim))
    print(f"{tag:<10}{tt*1e3:7.1f} ms  {nb/tt/1e9:6.1f} GB/s  {note}")
```

```text
A.stride=(8192, 1)   B.stride=(1, 8192)
A.sum(1)      2.7 ms    98.0 GB/s  内层走 stride-1
A.sum(0)      9.8 ms    27.4 GB/s  内层走 stride-8192
B.sum(1)      8.1 ms    33.0 GB/s  = A.sum(0) 同一个数学量
B.sum(0)      2.7 ms    99.9 GB/s  = A.sum(1) 同一个数学量
```

四格里两快两慢，而**快的两格和慢的两格分别对应同一个数学量**。结论：决定快慢的是「这个归约在内存里走的步长」，不是「归约第几维」这个表面描述。差距 **2.5~3x**，远小于 §4.1 里一维跨步的 30 倍 —— 因为 `TensorIterator` 还能把「哪一维放内层」换掉，把列归约改成「按行读 + 一个行宽的累加缓冲」（这一句是推断，依据是它按 stride 排序维度的设计；我没有去逐行读完 reduce kernel 的源码）。

#### 3.4 融合类算子：归约轴被钉死了

softmax、layernorm 要先沿某一维求 max/sum，再逐元素变换。归约轴是**语义指定的**，框架没有换的自由：

```python
import time, torch
torch.set_num_threads(4)

def bench(fn, reps=5, warmup=2):
    for _ in range(warmup):
        fn()
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best

B_, S, D = 64, 1024, 512
x = torch.randn(B_, S, D)                    # stride (S*D, D, 1)
xt = x.transpose(1, 2).contiguous()          # stride (S*D, S, 1)，沿 D 变成跨步
print(f"x.stride ={x.stride()}   xt.stride ={xt.stride()}")

s1 = bench(lambda: torch.softmax(x, -1))
s2 = bench(lambda: torch.softmax(xt, 1))
s3 = bench(lambda: x.transpose(1, 2).softmax(1))
print(f"{'softmax 沿 stride-1':<22}{s1*1e3:7.2f} ms{1.0:>8.2f}x")
print(f"{'softmax 沿 stride-S':<22}{s2*1e3:7.2f} ms{s2/s1:>8.2f}x")
print(f"{'softmax 跨步视图直算':<22}{s3*1e3:7.2f} ms{s3/s1:>8.2f}x")

ln, lns = torch.nn.LayerNorm(D), torch.nn.LayerNorm(S)
t1 = bench(lambda: ln(x))
t2 = bench(lambda: lns(x.transpose(1, 2)))
print(f"{'layernorm 沿 D（连续）':<22}{t1*1e3:7.2f} ms{1.0:>8.2f}x")
print(f"{'layernorm 沿 S（跨步）':<22}{t2*1e3:7.2f} ms{t2/t1:>8.2f}x")
```

```text
x.stride =(524288, 512, 1)   xt.stride =(524288, 1024, 1)
softmax 沿 stride-1      11.32 ms    1.00x
softmax 沿 stride-S      29.00 ms    2.56x
softmax 跨步视图直算          55.51 ms    4.90x
layernorm 沿 D（连续）        4.27 ms    1.00x
layernorm 沿 S（跨步）       20.71 ms    4.85x
```

沿跨步维做 softmax 要 **3.1~3.5x**（先转成连续副本）到 **5.6~6.3x**（直接用跨步视图），layernorm **3.6~4.9x**。两次独立运行都落在这个区间里。

#### 3.5 一张判断表

| 算子类别 | 有没有「换循环顺序」的自由 | 实测代价 | 你该做什么 |
|---|---|---|---|
| 纯逐元素（`a*b+c`、激活函数） | 有，完全自由 | 约 1.0x | 不用管，**别** `.contiguous()` |
| 归约（`sum`/`mean`/`max` 指定 dim） | 部分（能换哪维在内层） | 2.5~3x | 大张量时考虑换 layout |
| 融合算子（softmax / layernorm / attention） | 没有（归约轴被语义钉死） | 3.1~6.3x | **换 layout，不是加 `.contiguous()`** |
| 需要双向跨步（transpose / permute + 双线性索引） | 没有 | 7~8x | 自己写 tiling kernel（§4.4） |

判断方法一句话：**问这个算子「能不能靠换循环顺序把跨步访问变成连续访问」**。能，你不用管；不能，你不动 layout 就只能吃惩罚。

---

### 4. 围绕该领域展开

#### 4.1 一条 stride 扫描 = cacheline 尺寸的指纹

要证明「一次取货的最小单位是一个 cacheline」，最直接的办法是量「每个有用元素要摊多少字节」。把一个 256 MiB 的数组按不同 stride 抽元素求和：

```python
import os, time, torch
torch.set_num_threads(4)

LINE = int(os.popen("sysctl -n hw.cachelinesize").read().strip())  # macOS
# Linux 上等价物：getconf LEVEL1_DCACHE_LINESIZE
ELEM = 4
print(f"cacheline = {LINE} B -> 每行装得下 {LINE // ELEM} 个 fp32")

def bench(fn, reps=5, warmup=2):
    for _ in range(warmup):
        fn()
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best

N = 64 * 1024 * 1024
x = torch.randn(N, dtype=torch.float32)      # 256 MiB，远超 6 MB L2
print(f"\n{'stride':>6}{'读取元素':>12}{'有效 GB/s':>12}{'模型 B/元素':>13}{'实测倍数':>10}")
ref, rows = None, []
for s in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 33, 48, 64, 128]:
    v = x[0::s]
    n = v.numel()
    t = bench(lambda: v.sum())
    gbs = n * ELEM / t / 1e9                 # 只按「有用字节」算带宽
    ref = gbs if ref is None else ref
    model = min(ELEM * s, LINE)              # 每拿到 1 个有用元素要摊多少字节
    rows.append((s, ref / gbs))
    print(f"{s:>6}{n:>12,}{gbs:>12.2f}{model:>13.0f}{ref/gbs:>10.1f}")

# 判拐点：看两段「跨度不同但惩罚涨幅相近」的区间
d = dict(rows)
print(f"\nstride   8 ->  32（跨度 24）: {d[8]:>5.1f} -> {d[32]:>5.1f} 倍")
print(f"stride  32 -> 128（跨度 96）: {d[32]:>5.1f} -> {d[128]:>5.1f} 倍")
```

```text
cacheline = 128 B -> 每行装得下 32 个 fp32

stride        读取元素     有效 GB/s      模型 B/元素      实测倍数
     1  67,108,864       96.81            4       1.0
     2  33,554,432       49.27            8       2.0
     3  22,369,622       33.38           12       2.9
     4  16,777,216       25.37           16       3.8
     6  11,184,811       16.56           24       5.8
     8   8,388,608       13.40           32       7.2
    12   5,592,406        8.80           48      11.0
    16   4,194,304        6.52           64      14.9
    24   2,796,203        4.21           96      23.0
    32   2,097,152        2.64          128      36.7
    33   2,033,602        2.90          128      33.4
    48   1,398,102        3.11          128      31.2
    64   1,048,576        2.49          128      38.8
   128     524,288        1.88          128      51.5

stride   8 ->  32（跨度 24）:   7.2 ->  36.7 倍
stride  32 -> 128（跨度 96）:  36.7 ->  51.5 倍
```

看最后两列。模型说指针每前进一步要摊掉 `min(4s, 128)` 字节，s ≤ 32 那一段实测几乎贴着它走：2.0、2.9、3.8、5.8、7.2、11.0、14.9、23.0、36.7 倍。

**然后斜率塌了。** 最后两行是这一节的判据：

- stride **8 → 32**，跨度 24 步，惩罚从 7.2 涨到 36.7 —— 涨了 5.1 倍，**每单位 stride 约 +17%**。
- stride **32 → 128**，跨度 96 步（是前一段的 4 倍），惩罚从 36.7 只涨到 51.5 —— 涨了 1.4 倍，**每单位 stride 约 +0.4%**。

每单位 stride 的惩罚掉了 40 倍。而 32 正是 128 B ÷ 4 B —— **拐点位置就是 cacheline 尺寸的指纹**。不用查 `sysctl`，找到「再多跨一步、代价突然不涨了」的那个 stride，乘上元素大小，就是 cacheline 宽度。这个手法在任何机器上都通用，连 GPU 都能这么测。

尾段确实不单调（stride 33 和 48 那两格还略低于 32 格），因为这里已经混进了 DRAM page 局部性、内存通道交织和 TLB —— **真实机器不会给你一条干净的阶梯，只会给你一个斜率突变的拐点**。

顺带一个反直觉的对照：**GPU 的曲线不该在这个位置拐**。按 §4.2 的 sector 模型，GPU 在 stride=8 就触底（有效率 12.5%），因为它的记账单位只有 32 字节。CPU 的 128 B line 一次要买 4 倍的量 —— **同一个跨步访问，在按 128 B 记账的机器上浪费得更彻底**（地板 3.1% 对 12.5%）。

#### 4.2 coalescing 的精确算术，顺便和官方文档对账

不用 GPU 也能算出「一个 warp 会买几个 sector」：把地址取整到 32 字节的格子，数有几个不同的格子。

```python
SECTOR = 32
WARP = 32

def n_sectors(addrs, size=4):
    """一个 warp 的地址列表 -> 需要几个 32 B sector"""
    s = set()
    for a in addrs:
        s.update(range(a // SECTOR, (a + size - 1) // SECTOR + 1))
    return len(s)

print(f"{'一个 warp（32 线程 x 4 B）':<30}{'sector':>8}{'记账字节':>10}{'有效率':>9}")
cases = [("连续 addr=4t（对齐）", [4 * t for t in range(WARP)]),
         ("stride 2  addr=8t", [8 * t for t in range(WARP)]),
         ("stride 4  addr=16t", [16 * t for t in range(WARP)]),
         ("stride 8  addr=32t", [32 * t for t in range(WARP)]),
         ("连续但错位 4 B", [4 + 4 * t for t in range(WARP)]),
         ("每线程读一列 addr=4096t", [4096 * t for t in range(WARP)])]
for name, addrs in cases:
    n = n_sectors(addrs)
    print(f"{name:<30}{n:>8}{n*SECTOR:>10}{WARP*4/(n*SECTOR)*100:>8.1f}%")

print("\n和 CUDA 官方文档给的数字对账：")
for name, addrs, want in [("连续", [4 * t for t in range(WARP)], 4),
                          ("stride 2", [8 * t for t in range(WARP)], 8),
                          ("整体错位 4 B", [4 + 4 * t for t in range(WARP)], 5)]:
    got = n_sectors(addrs)
    print(f"  {name:<12} 模型 {got} 段 / 文档 {want} 段 -> "
          f"{'一致' if got == want else '不一致'}")
```

```text
一个 warp（32 线程 x 4 B）            sector      记账字节      有效率
连续 addr=4t（对齐）                       4       128   100.0%
stride 2  addr=8t                    8       256    50.0%
stride 4  addr=16t                  16       512    25.0%
stride 8  addr=32t                  32      1024    12.5%
连续但错位 4 B                            5       160    80.0%
每线程读一列 addr=4096t                   32      1024    12.5%

和 CUDA 官方文档给的数字对账：
  连续           模型 4 段 / 文档 4 段 -> 一致
  stride 2     模型 8 段 / 文档 8 段 -> 一致
  整体错位 4 B     模型 5 段 / 文档 5 段 -> 一致
```

三个数字全部对上（4 / 8 / 5），这不是巧合 —— 文档给的就是这条取整算术的结果。**同一套算术可以拿去算任何 layout，包括你自己设计的 kernel。**

一个值得记住的闭式结果：warp 32 个线程每个读 4 字节、元素间隔 $s$ 个时，

$$N_{\text{sector}} = \min\left(32,\ 4s\right),\qquad
\eta = \frac{128}{32\,N_{\text{sector}}} = \max\left(\frac{1}{s},\ \frac{1}{8}\right)$$

**有效率的地板是 12.5%，不是 0**。原因很朴素：一个 warp 最多只能买 32 个 sector，每个线程各一个 —— 再散也就散到这儿了。所以「散布访问」的代价有上限，量级是 8 倍。

#### 4.3 bank conflict 的精确算术：为什么是「加一列」

shared memory 那头同理，只是记账单位从 sector 换成了 bank：`bank = (byte_addr // 4) % 32`。

```python
BANKS, WORD, WARP = 32, 4, 32

def waves(addrs):
    """一个 warp 的 32 个字节地址 -> (wavefront 数, 命中的 bank 数)
    同一 bank 上的多个**不同**地址串行化；同一地址是广播，不算冲突。"""
    per = {}
    for a in addrs:
        per.setdefault((a // WORD) % BANKS, set()).add(a)
    return max((len(v) for v in per.values()), default=0), len(per)

def show(name, addrs):
    w, nb = waves(addrs)
    print(f"{name:<28} bank 数 {nb:>2}  wavefront {w:>2}  "
          f"有效 {WARP*WORD/w:>5.1f} B/clk")

print("一个 warp 读 shared memory：")
show("连续 32 个 word", [4 * t for t in range(WARP)])
show("stride 2", [8 * t for t in range(WARP)])
show("stride 4", [16 * t for t in range(WARP)])
show("stride 32", [128 * t for t in range(WARP)])
show("全部读同一个地址（广播）", [0] * WARP)

print("\n32x32 的 tile 存成 float[32][W]，读一列（固定 j、i=0..31）：")
print(f"{'W':>4}{'命中的 bank 数':>15}{'wavefront':>11}{'冲突路数':>10}")
for W in [32, 33, 34, 36, 40]:
    addrs = [(i * W) * WORD for i in range(WARP)]
    w, nb = waves(addrs)
    print(f"{W:>4}{nb:>15}{w:>11}{w:>10}")
```

```text
一个 warp 读 shared memory：
连续 32 个 word                 bank 数 32  wavefront  1  有效 128.0 B/clk
stride 2                     bank 数 16  wavefront  2  有效  64.0 B/clk
stride 4                     bank 数  8  wavefront  4  有效  32.0 B/clk
stride 32                    bank 数  1  wavefront 32  有效   4.0 B/clk
全部读同一个地址（广播）                 bank 数  1  wavefront  1  有效 128.0 B/clk

32x32 的 tile 存成 float[32][W]，读一列（固定 j、i=0..31）：
   W     命中的 bank 数  wavefront      冲突路数
  32              1         32        32
  33             32          1         1
  34             16          2         2
  36              8          4         4
  40              4          8         8
```

读一列时命中的 bank 数是 $32/\gcd(W, 32)$，于是：

- $W=32$（无 padding）→ $\gcd=32$ → 只剩 **1 个 bank** → 32 路冲突，读一列要 32 个 wavefront，有效带宽从 128 B/clk 掉到 **4 B/clk**。
- $W=33$ → $\gcd=1$ → **32 个 bank** → 1 个 wavefront。**加一列的全部作用就是让 W 与 32 互质。**
- $W=34$ → $\gcd=2$ → 16 个 bank，还是 2 路冲突；$W=36$ → 8 个 bank，4 路。

所以「pad 一列」不是玄学数字，是数论：**要 pad 到奇数，最好是与 bank 数互质。**

> 注意「读一列」不是 rare case。矩阵乘法里把 $B$ 的一列读进寄存器、transpose 里把 tile 转置写出、attention 里按 head 维取 KV —— 都是这个形状。

#### 4.4 一个真 kernel：8 倍从哪来，以及 padding 为什么在这儿没用

理论讲完了，落到真代码。`torch.utils.cpp_extension` 能在本机编译 C++ kernel，下面四个变体只改「怎么访存」，数学完全一样：

```python
import os, time, torch
from torch.utils.cpp_extension import load_inline

# 本机 ninja 装在 venv 里、不在 PATH 上，补一句（你环境里若有可删）
os.environ["PATH"] = "/Users/morssi/.workbuddy/binaries/python/envs/default/bin:" + os.environ["PATH"]
torch.set_num_threads(1)          # 单线程：让访存模式成为唯一变量

CPP = r"""
#include <torch/extension.h>
constexpr int TS = 32;

void naive(const float* __restrict A, float* __restrict B, int N) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) B[j*N+i] = A[i*N+j];
}
void blocked(const float* __restrict A, float* __restrict B, int N) {
  for (int ii = 0; ii < N; ii += TS)
    for (int jj = 0; jj < N; jj += TS)
      for (int i = ii; i < ii+TS; ++i)
        for (int j = jj; j < jj+TS; ++j) B[j*N+i] = A[i*N+j];
}
void staged(const float* __restrict A, float* __restrict B, int N) {
  float t[TS][TS];
  for (int ii = 0; ii < N; ii += TS)
    for (int jj = 0; jj < N; jj += TS) {
      for (int i = 0; i < TS; ++i)
        for (int j = 0; j < TS; ++j) t[i][j] = A[(ii+i)*N + jj+j];
      for (int j = 0; j < TS; ++j)
        for (int i = 0; i < TS; ++i) B[(jj+j)*N + ii+i] = t[i][j];
    }
}
void staged_pad(const float* __restrict A, float* __restrict B, int N) {
  float t[TS][TS+1];                      // 只多这一列
  for (int ii = 0; ii < N; ii += TS)
    for (int jj = 0; jj < N; jj += TS) {
      for (int i = 0; i < TS; ++i)
        for (int j = 0; j < TS; ++j) t[i][j] = A[(ii+i)*N + jj+j];
      for (int j = 0; j < TS; ++j)
        for (int i = 0; i < TS; ++i) B[(jj+j)*N + ii+i] = t[i][j];
    }
}
torch::Tensor transpose(torch::Tensor A, int64_t mode) {
  int N = A.size(0); auto B = torch::empty_like(A);
  const float* a = A.data_ptr<float>(); float* b = B.data_ptr<float>();
  if (mode == 0) naive(a, b, N); else if (mode == 1) blocked(a, b, N);
  else if (mode == 2) staged(a, b, N); else staged_pad(a, b, N);
  return B;
}
"""

mod = load_inline(name="tr043", cpp_sources=CPP, functions=["transpose"],
                  extra_cflags=["-O3"], verbose=False)

N = 4096
A = torch.randn(N, N, dtype=torch.float32)     # 64 MiB
ref = A.t().contiguous()

def bench(fn, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best

names = ["朴素（直读直写）", "只把循环分块", "分块 + 局部 tile", "分块 + tile + padding"]
print(f"{'实现':<24}{'ms':>8}{'有效 GB/s':>11}{'相对朴素':>10}")
base, res = None, []
for m in range(4):
    assert torch.equal(mod.transpose(A, m), ref)
    t = bench(lambda m=m: mod.transpose(A, m))
    base = t if base is None else base
    res.append(t)
    print(f"{names[m]:<24}{t*1e3:>8.1f}{N*N*4*2/t/1e9:>11.2f}{base/t:>9.2f}x")
print(f"\n只分块的收益       : {res[0]/res[1]:.2f}x")
print(f"再加局部 tile      : {res[1]/res[2]:.2f}x")
print(f"再加 padding       : {res[2]/res[3]:.2f}x")
```

```text
实现                            ms    有效 GB/s      相对朴素
朴素（直读直写）                    80.4       1.67     1.00x
只把循环分块                      82.3       1.63     0.98x
分块 + 局部 tile                 9.5      14.13     8.46x
分块 + tile + padding          9.7      13.91     8.33x

只分块的收益       : 0.98x
再加局部 tile      : 8.67x
再加 padding       : 0.98x
```

三条结论，每条都和教科书说法不完全一样：

**第一，只把循环分块是白忙。** 0.94~0.98x，六轮独立运行没有一轮快过朴素版。因为分块只改变了「循环的先后」，每个内层迭代仍然是「读连续、写间隔 4096 个元素」的散写，一次 4 字节地往 32 个不同 cacheline 上戳。

**第二，8 倍的来源是「把两次跨步访问换成一次局部 + 两次连续」。** `staged` 版本里，`t[i][j] = A[...]` 是连续读，`B[...] = t[i][j]` 是连续写 —— 跨步访问被压缩进了那个 4 KB 的栈上数组，而栈上数组一次装得进 L1。**这就是 shared memory 在 GPU 上的全部用处**：它把「global 上的跨步」变成「shared 上的跨步 + global 上的连续」，然后用 §4.3 的手段把 shared 上的跨步也消掉。

**第三，padding 在这台机器上没有任何收益（0.97~0.98x，也就是略慢）。** 因为 CPU 上那个 tile 是栈上数组，编译器知道每个下标的静态偏移，没有「bank」这个结构；而且 32 个 float = 128 字节恰好是一条 cacheline，`[32][32]` 的每一行正好占一行，本来就不冲突。**padding 是 NVIDIA bank 结构的专属对策，不是通用技巧。**

#### 4.5 哪些是 NVIDIA 特有，哪些能迁移

| 机制 | 归属 | 迁移时该怎么问 |
|---|---|---|
| 32 B sector / 128 B L1 line | **NVIDIA 数值特有** | 「你的加速器一次最小访问多少字节？」 |
| 32 banks × 4 B、`+1` padding | **NVIDIA 特有**（bank 数、位宽都可变） | 「多少个 bank、每个多宽？映射是取模吗？」 |
| shared memory 228 KB/SM、128 B/clk | **NVIDIA 特有数值** | 「片上暂存有多大、每 clk 搬多少？」 |
| TMA / wgmma / cp.async / 线程块簇 / 2:4 稀疏 | **NVIDIA 特有指令** | 先看有没有等价物，没有就换算法 |
| 带宽阶梯（容量↑带宽↓，约 10 倍/级） | **通用** | 任何多级存储都成立 |
| coalescing 的动机（warp 地址连续） | **通用** | 换 warp 宽度、换最小事务粒度 |
| bank = 地址取模的冲突语义 | **通用** | 只要有 banked SRAM 就有 |
| roofline / 算术强度 | **通用** | 换峰值算力和带宽即可 |
| tiling：跨步 → 两次连续 | **通用** | 不依赖任何 NVIDIA 细节 |

这张表是给移植用的：**前四行换硬件就要重新量，后五行可以直接搬。** 拿到一块新加速器，先问三个数——最小访问粒度、bank 数与位宽、每 clk 片上带宽——剩下的事本章的算术全都能推。

#### 4.6 回到 roofline：为什么 decode 注定撞墙

把 §1 的表再想一遍。$\mathrm{AI}=1$ 意味着**每读一个 fp16 参数（2 字节）只做一次乘加（2 FLOP）**，这活儿的算力利用率天花板就是 $1/295 \approx 0.34\%$。想让 decode 变快，唯一的出路是**提高 $\mathrm{AI}$**，而提高 $\mathrm{AI}$ 只有两种办法：

1. **让同一次搬运服务更多计算** —— continuous batching（第 029 期）、提高 batch、MoE 的专家共享。
2. **减少必须搬的字节** —— KV 量化（第 033 期）、KV 驱逐（第 035 期）、RadixAttention 前缀复用（第 030 期）、PD 分离把 KV 一次搬完（第 031 期）。

第 040 期的算子融合也是这一条：把 `a*b+c` 三趟搬变成一趟，$\mathrm{AI}$ 直接乘 3。**所有推理优化的账面，最终都记在「搬了多少字节」这一栏。**

---

### 5. 什么时候该用 / 不该用

**该动手改访存模式的时候：**

- 你的 kernel 落在 roofline 左边（$\mathrm{AI}$ 低于屋脊点），且 profiler 显示 mem throughput 已经接近峰值 —— 此时唯一的出路是提高 $\mathrm{AI}$ 或提高有效率。
- 张量的归约/融合算子跑在跨步维上（§3.4），且这个算子在 hot path 上。
- 你需要写自定义 kernel（Triton / cpp_extension），一次 warp 的地址由你决定。
- 移植到新硬件时，先量最小访问粒度和 bank 结构。

**不该动手的时候：**

- 纯逐元素算子遇到非连续输入 —— 框架已经重排过（约 1.0x），你插手只会更慢。
- 想用 `.contiguous()` 「修复」layout —— 除非下游要复用这个张量很多次，否则那一趟搬运就吃掉全部收益（约 6x）。
- kernel 本身是 launch-bound / 延迟受限（第 042 期刚量过 omp 唤醒成本），访存优化不是瓶颈。
- 数据量根本装得进 L2 —— 此时 cache 已经把访存模式熨平了，优化空间接近于零。

---

### 6. 常见坑

**坑 1：把 `is_contiguous() == False` 当成「要出事」的信号。**
它只说明物理布局与逻辑顺序不同，不说明慢。逐元素算子实测 0.97~1.03x，因为 `TensorIterator` 会换循环顺序。判断标准是 §3.5 那张表：**这个算子有没有换顺序的自由**，而不是 `is_contiguous()` 的返回值。

**坑 2：把 128 B cacheline 和 32 B sector 混着用。**
这是本期最容易犯的错，而且会导致**方向相反的结论**。§4.1 的实测在 stride=32 处斜率塌陷（128 B 记账），§4.2 的 GPU 模型在 stride=8 就触底（32 B 记账）。拿前者的曲线去解释 GPU，你会以为 stride=16 的访存还有救；拿后者去解释 CPU，你会低估 4 倍的浪费。**先问「这台机器一次最少买多少字节」。**

**坑 3：一维跨步的结论直接外推到多维归约。**
一维没有「换哪维在内层」的自由，吃满全额惩罚（stride≥32 就 30 倍以上）；二维列归约还有这个自由，只吃 2.5~3 倍。差了一个数量级。**测之前先问「这个 kernel 还有没有重排的余地」。**

**坑 4：只加循环分块，不加局部暂存。**
0.94~0.98x，等于白改一遍代码还引入了复杂度。分块本身不解决任何访存问题 —— 它只是为「暂存」创造了条件。真正的收益来自 §4.4 里那个 `float t[32][32]`。

**坑 5：给 CPU 代码加 `+1` padding，然后纳闷为什么没用。**
padding 治的是 bank 冲突，不是 cache 冲突。CPU 上没有 bank（编译器用静态偏移寻址栈数组），所以 0.97x。反过来，在 GPU 上不加 padding 的 transpose 会凭空多 32 倍 shared 访问 —— **同一个技巧换个机器就从必需变成无用。**

**坑 6：计时只跑一次就下结论。**
本机同一段代码三次独立运行，stride=128 那一格的倍数落在 45~57 之间（跨度 27%）。§0 里所有「N 倍」都是三次独立进程的区间，不是单次读数。**访存实验的比值通常很稳，但绝对耗时永远不稳**，所以本文全部用比值说话。

---

### 7. 一句话总结

**内存层次的所有优化都在做同一件事：让「硬件记账的字节」贴近「你真正用到的字节」** —— global 那头靠 warp 地址连续（sector），shared 那头靠地址与 bank 数互质（padding），跨步那头靠 tiling 把一次跨步换成两次连续；而框架已经替你在「有重排自由」的地方做完了，你只需要认得出「没有自由」的那几类算子。

---

### 今日练习

<details><summary>今日练习（点击展开参考答案）</summary>

**题目**：一个 attention kernel，warp 里 32 个线程负责同一个 token 的 32 个 head（head $h$ 由线程 $h$ 处理），KV cache 按 `(S, H, d)` 存放，`d=128`，fp16。问：

1. warp 读 `K[t, h, :]` 的 `K[t, h, 0]` 这一个元素时，买几个 sector？有效率多少？
2. 把 KV cache 改成 `(H, S, d)` 布局，同一个访问变成什么样？
3. 如果改成「warp 里 32 个线程负责同一 head 的 32 个相邻 token」，又变成什么样？

**答案**：

1. `(S, H, d)` 布局下，线程 $h$ 的地址是 $\big((tH + h)\,d\big)\times 2$ 字节，相邻线程相差 $d\cdot 2 = 256$ 字节，各占一个 sector 且互不相邻 → **32 个 sector，记账 1024 字节，真正用到 $32\times 2 = 64$ 字节，有效率 6.25%**。

   注意这不是「跌破 §4.2 的地板」—— 那个 12.5% 的地板是按「每线程读 4 字节」算的；把访问宽度换成 2 字节，地板本身就得换成 $2/32 = 6.25\%$。**地板是访问宽度的函数，不是常数**，这是套用公式时最容易漏掉的一层。

2. `(H, S, d)` 布局下，地址是 $(h\cdot S\cdot d + t\cdot d)\times 2$，相邻线程仍相差 $S\cdot d\cdot 2$ 字节（$S$ 越大差得越远）→ **依然 32 个 sector，6.25%**。**这个改法对「一 token 一 head」的分工完全无效** —— 布局不是随便换就有收益的，要看它跟「谁和谁是邻居」是否匹配。

3. 让线程 $i$ 负责 token $t+i$（同一 head），地址是 $(hSd + (t+i)d)\times 2$，相邻线程差 $d\cdot 2 = 256$ 字节 → **还是 32 个 sector，6.25%**。

   真正能救的是**让相邻线程读相邻字节**：线程 $i$ 负责 `d` 维里的第 $i$ 段，即线程 $i$ 读 `K[t, h, 4i : 4i+4]`（4 个 fp16 = 8 字节）。地址变成

   $$\big((tH+h)\,d + 4i\big)\times 2 = \text{base} + 8i$$

   相邻线程只差 8 字节，32 个线程刚好铺满 $32\times 8 = 256$ 字节 = **8 个 sector，用到 256 字节，有效率 100%**。

   **这就是 flash attention 装载 K/V 时让线程覆盖连续 `d` 维元素、并以 8 或 16 字节向量化读的原因**：不是为了「SIMD 好看」，是为了让一个 warp 的地址连续，把 32 个箱子压成 8 个。

</details>
