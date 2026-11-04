# PyTorch 每日一课 · 第 049 期

## Split-K：多个 CTA 共用一个输出 tile

| | |
|---|---|
| **日期** | 2026-11-04 |
| **难度** | ⭐⭐⭐⭐ |
| **前置知识** | GEMM 的分块与算术强度（043 期）、Tensor Core 的 warpgroup 与寄存器预算（045/047 期）、epilogue 的写回路径（048 期） |
| **预计阅读** | 25 分钟 |
| **关联** | 第 048 期：一个 CTA 内部怎么收尾 → 本期问多个 CTA 共用一个 tile 时收尾变成什么样；第 046 期：DSMEM 是归约的第三种出口 |

---

## 一、这个领域解决什么问题

GEMM 的并行度全部来自 $M \times N$ 两个方向。$K$ 方向天生是串行的——你必须把 $K$ 项乘加全部累加完，才能得到一个输出元素。

这在前向的大矩阵乘里没问题：$M=N=4096$、tile 取 $128\times128$，能切出 $32 \times 32 = 1024$ 个 CTA，比 H100 的 132 个 SM 多得多，随便怎么排都能喂饱。

问题出在另一个形状：**$M \times N$ 很小，$K$ 很大**。

- decode 阶段的 attention：query 只有 1 个 token，$M=1$
- 单条样本的推理：batch size = 1
- LoRA 的 $B\cdot A$：rank 只有 8~64，中间那个维度就是 $K$
- 序列方向的线性投影：$M$ = 序列长度可以很大，但若按 head 拆开，每个 head 的 $N$ 只有 64/128

这些形状的 CTA 数可能比 SM 数还少，甚至只有 1 个。**SM 有 132 个，你只派出 1 个 CTA，剩下 131 个在空转。**

先在本机把这件事量出来。下面测的是方形 GEMM 在不同 $M=N$ 下的算力兑现情况：

```python
import time, torch

torch.set_num_threads(4)

def bench(fn, reps=9, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

print(f"torch {torch.__version__}, 线程数 {torch.get_num_threads()}")
print("\n=== 固定 K=4096，扫 M=N：CTA 数从 1 涨到 1024 ===")
print(f"{'M=N':>6} {'CTA(128²)':>10} {'ms':>9} {'GFLOPS':>9}")
for mn in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
    a = torch.randn(mn, 4096); b = torch.randn(4096, mn)
    t = bench(lambda: a @ b)
    print(f"{mn:>6} {((mn + 127)//128)**2:>10} {t*1e3:>9.3f} {2*mn*mn*4096/t/1e9:>9.0f}")
```

```text
torch 2.14.0, 线程数 4

=== 固定 K=4096，扫 M=N：CTA 数从 1 涨到 1024 ===
   M=N  CTA(128²)        ms    GFLOPS
    32          1     0.012       671
    64          1     0.036       937
   128          1     0.095      1409
   256          4     0.364      1475
   512         16     1.328      1617
  1024         64     4.500      1909
  2048        256    17.015      2019
  4096       1024    66.390      2070
```

同一台机器、同一份代码，从约 670 GFLOPS 到约 2070 GFLOPS，差 **3.1 倍**。这就是并行度不足的代价。

> 注意本机的局限：Apple M5 只有 10 个核，而且实测这台机器的 GEMM 根本不走多线程（把线程数从 1 调到 4，时间比值恒为 1.00，因为 macOS 的 BLAS 路径不受 `set_num_threads` 支配）。所以这里的 2.8 倍是「cache 与流水线效率」的差异，**不是**「132 个 SM 里用了几个」的差异。真 GPU 上的差距会激进得多——$M=N=128$ 在 H100 上只有 1 个 CTA，等于 132 个 SM 用了 1 个。这一点只能靠算术，后面第三节会给。

split-K 就是对这个问题的回答：**既然 $M\times N$ 切不出足够的块，那就把 $K$ 也切一刀。**

一个输出 tile 的 $K$ 次累加，分给 $S$ 个 CTA，每个算 $K/S$ 的一段，各自得到一份**部分和**（partial sum），最后把这 $S$ 份加起来。

代价立刻出现了：**输出不再是一个 tile，而是 $S$ 个 tile**，而且要多一次归约。这就是本期全部复杂度的来源。

---

## 二、核心思想：用归约换并行度

### 一个类比

假设你要核对一本 64 万笔的账。

- **不 split**：一个人从头核到尾。慢，但不需要跟别人对账。
- **split-K**：叫 64 个人，每人核 1 万笔，各自出一个小计，然后你把 64 个小计加起来。

第二种做法快，但多了两个新问题：**小计放哪**（workspace），以及**谁来把小计加起来**（归约）。

而归约方式的选择，直接决定了这件事划不划算。工程上有三种主流姿势：

| 姿势 | 做法 | 谁在用 |
|---|---|---|
| **atomicAdd** | 部分和直接原子加到最终输出上 | Triton 手写 kernel、早期 CUTLASS split-K SIMT 版本 |
| **两趟 kernel** | 部分和写 workspace，第二个 kernel 归约 | CUTLASS `GemmSplitKParallel` + `ReduceSplitK` |
| **单 kernel + 跨 CTA barrier** | 部分和写 workspace，barrier 后每个 CTA 各自重算一遍全量 | Inductor 的 cooperative reduction、cluster 内 DSMEM |

第三种最反直觉，但它是 PyTorch 自己的选择，代码在后文第五节。

### 关键的不对称

设单 tile 走完整个 $K$ 的耗时是 $t_K$，输出 tile 的字节数是 $M_t N_t b$（$b$ 是输出元素位宽），总带宽 $BW$。那么

- 计算侧：拆成 $S$ 份后是 $t_K / S$ —— **随 $S$ 反比下降**
- 归约侧：要读写 $S$ 份部分和，$\approx S \cdot M_t N_t b / BW$ —— **随 $S$ 线性上升**

一个 $\propto 1/S$，一个 $\propto S$，所以必有最优解，而且它不会在 $S$ 很大时出现。这是本期最重要的一句话：**split-K 不是「切得越细越好」，切到某个点之后每多切一份都是净亏。**

### 把最优 $S$ 解出来

$$T(S) = \underbrace{\frac{t_K}{S}}_{\text{算}} + \underbrace{\frac{S \cdot M_t N_t b}{BW}}_{\text{归约}}$$

对 $S$ 求导置零：

$$\frac{dT}{dS} = -\frac{t_K}{S^2} + \frac{M_t N_t b}{BW} = 0 \quad\Longrightarrow\quad S^* = \sqrt{\frac{t_K \cdot BW}{M_t N_t b}}$$

把 $t_K = \dfrac{2 M_t N_t K}{P_{sm}}$（$P_{sm}$ 是单个 SM 的算力）代进去，$M_t N_t$ 恰好约掉：

$$\boxed{\;S^* = \sqrt{\frac{2K \cdot BW}{P_{sm} \cdot b}}\;}$$

这个化简有两个值得停下来看的地方。

**第一，$S^*$ 与 tile 尺寸无关。** tile 放大一倍，$t_K$ 翻倍，但归约字节也翻倍，两者在 $T(S)$ 里同比例出现，"用 $S$ 份换时间"的汇率不变。所以最优切分数不动。这一点用代码验证：

**第二，加速比不是 $P$，是 $S^*/2$。**

$$\frac{T(1)}{T(S^*)} = \frac{t_K}{2\sqrt{t_K M_t N_t b/BW}} = \frac{1}{2}\sqrt{\frac{t_K BW}{M_t N_t b}} = \frac{S^*}{2}$$

也就是说，你把 132 个 SM 全部派上去，**split-K 只能拿到大约 $S^*/2$ 倍的加速**，剩下的被归约吃掉。

下面把三个真实 GPU 的参数代进去。规格取厂商公布的 dense 数字：A100 SXM 108 SM / 2.039 TB/s / FP16 dense 312 TFLOPS，H100 SXM 132 SM / 3.35 TB/s / 989.5 TFLOPS，B200 192 SM / 8 TB/s / FP16 dense 2250 TFLOPS（**这三个是文档级参数，本机无 GPU 无法实测核对**；下面的 $S^*$ 与加速比是由它们算出来的，不是测出来的）：

```python
import math

devs = {
    "A100 SXM": dict(P=108, bw=2.039e12, fp16=312e12),
    "H100 SXM": dict(P=132, bw=3.35e12,  fp16=989.5e12),
    "B200":     dict(P=192, bw=8.0e12,   fp16=2250e12),
}
K, bm, bn, b = 8192, 128, 128, 4   # fp32 输出

print(f"K={K}, tile={bm}x{bn}, 输出 {b}B")
print(f"{'GPU':>10} {'SM':>4} {'BW TB/s':>8} {'单SM TF':>8} {'t_K us':>8} {'S*':>6} {'T(S*)us':>8} {'加速比':>7} {'兑现率':>7}")
for name, d in devs.items():
    psm = d["fp16"] / d["P"]
    tK = 2 * bm * bn * K / psm
    sstar = math.sqrt(2 * K * d["bw"] / (psm * b))
    T = tK / sstar + sstar * bm * bn * b / d["bw"]
    print(f"{name:>10} {d['P']:>4} {d['bw']/1e12:>8.2f} {psm/1e12:>8.2f} "
          f"{tK*1e6:>8.2f} {sstar:>6.1f} {T*1e6:>8.2f} {tK/T:>7.1f} {tK/T/d['P']:>7.1%}")

print("\n=== S* 与 tile 尺寸无关：t_K 变了，S* 不动 ===")
d = devs["H100 SXM"]; psm = d["fp16"] / d["P"]
for (m, n) in [(64, 64), (128, 128), (128, 256), (256, 256)]:
    tK = 2 * m * n * K / psm
    print(f"  tile {m:>3}x{n:<3}: t_K={tK*1e6:>7.2f} us, S*={math.sqrt(tK*d['bw']/(m*n*b)):>6.2f}")

print("\n=== T(S) 的 U 形（H100, tile 128x128, K=8192）===")
psm = d["fp16"] / d["P"]; tK = 2 * bm * bn * K / psm
print(f"{'S':>4} {'计算 us':>9} {'归约 us':>9} {'合计 us':>9} {'vs S=1':>8}")
for S in [1, 4, 8, 16, 32, 42, 64, 128, 256]:
    tc = tK / S
    tr = S * bm * bn * b / d["bw"]
    print(f"{S:>4} {tc*1e6:>9.2f} {tr*1e6:>9.2f} {(tc+tr)*1e6:>9.2f} {tK/(tc+tr):>7.1f}x")
```

```text
K=8192, tile=128x128, 输出 4B
       GPU   SM  BW TB/s   单SM TF   t_K us     S*  T(S*)us     加速比     兑现率
  A100 SXM  108     2.04     2.89    92.92   53.8     3.46    26.9   24.9%
  H100 SXM  132     3.35     7.50    35.81   42.8     1.67    21.4   16.2%
      B200  192     8.00    11.72    22.91   52.9     0.87    26.4   13.8%

=== S* 与 tile 尺寸无关：t_K 变了，S* 不动 ===
  tile  64x64 : t_K=   8.95 us, S*= 42.78
  tile 128x128: t_K=  35.81 us, S*= 42.78
  tile 128x256: t_K=  71.62 us, S*= 42.78
  tile 256x256: t_K= 143.24 us, S*= 42.78

=== T(S) 的 U 形（H100, tile 128x128, K=8192）===
   S     计算 us     归约 us     合计 us   vs S=1
   1     35.81      0.02     35.83     1.0x
   4      8.95      0.08      9.03     4.0x
   8      4.48      0.16      4.63     7.7x
  16      2.24      0.31      2.55    14.0x
  32      1.12      0.63      1.75    20.5x
  42      0.85      0.82      1.67    21.4x
  64      0.56      1.25      1.81    19.8x
 128      0.28      2.50      2.78    12.9x
 256      0.14      5.01      5.15     7.0x
```

三件事值得逐条读。

**$S^*$ 几乎不随硬件代际变化。** A100 / H100 / B200 分别是 53.8 / 42.8 / 52.9。原因在公式里：$S^*$ 只依赖 $K$、每 SM 分到的算力、和总带宽。算力涨了 4 倍，带宽涨了 4 倍，比值不动，$S^*$ 就不动。

**但 SM 数从 108 涨到 192。** 所以「兑现率」一路下滑：24.9% → 16.2% → 13.8%。这句话的工程含义是：**硬件越新，窄形状越绝望**。你能靠 split-K 榨出来的并行度是个近似常数（20 倍上下），而理想并行度每代涨 30%~40%。买新卡不会让 $M=N=128$ 的 GEMM 变好。

**U 形的底部很浅。** $S$ 从 42 加到 64，时间只从 1.67 涨到 1.81 us（+8%）；但加到 256 就崩到 5.15 us，比 $S=8$ 还慢。所以选 $S$ 选的不是精确值，是「量级」——在 $S^*$ 的一半到两倍之间，怎么选都不太亏。

### 硬上限从哪来

$S^* = 42.8$ 这个数字有隐约的眼熟感。Inductor 里有个常量 `TRITON_MAX_RSPLIT = 64`，位置在 `torch/_inductor/runtime/hints.py:22`。它当然不是照着这个公式定的，但两者撞上了：

```python
import math

d = dict(P=132, bw=3.35e12, fp16=989.5e12)   # H100 SXM
psm = d["fp16"] / d["P"]
print("S* 只随 K 走，64 这个上限对应哪个 K：")
for k in [1024, 2048, 4096, 8192, 16384, 32768]:
    s = math.sqrt(2 * k * d["bw"] / (psm * 4))
    print(f"  K={k:>6}: S*={s:>6.2f}" + ("   <-- 到顶" if 60 < s < 68 else ""))

print("\n但第一道闸不是 S*，是并行度余量 P/C：")
print(f"{'M=N':>6} {'K':>7} {'C':>6} {'P/C':>7} {'S*':>7} {'实际取':>7}")
for mn, k in [(128, 8192), (256, 8192), (512, 4096), (1024, 4096), (4096, 4096)]:
    C = (mn // 128) ** 2
    cap = math.sqrt(2 * k * d["bw"] / (psm * 4))
    s = max(1, min(int(d["P"] / C), int(cap), 64))
    print(f"{mn:>6} {k:>7} {C:>6} {d['P']/C:>7.2f} {cap:>7.1f} {s if s > 1 else 0:>7}")
```

```text
S* 只随 K 走，64 这个上限对应哪个 K：
  K=  1024: S*= 15.13
  K=  2048: S*= 21.39
  K=  4096: S*= 30.25
  K=  8192: S*= 42.78
  K= 16384: S*= 60.51   <-- 到顶
  K= 32768: S*= 85.57

但第一道闸不是 S*，是并行度余量 P/C：
   M=N       K      C     P/C      S*     实际取
   128    8192      1  132.00    42.8      42
   256    8192      4   33.00    42.8      33
   512    4096     16    8.25    30.3       8
  1024    4096     64    2.06    30.3       2
  4096    4096   1024    0.13    30.3       0
```

`TRITON_MAX_RSPLIT = 64` 对应的正是 $K \approx 16\text{K}$。超过这个深度，公式说还该继续切，但实现不给。

最后一行是全文最该记住的判据：**$M=N=4096$、$K=4096$ 时 $C=1024 > P=132$，$P/C = 0.13$，split-K 一份都不该切。** 并行度早就够了，再切只是把输出写 $S$ 遍。

---

## 三、在 PyTorch 中怎么用

这里有个反直觉的事实：**在 PyTorch 里你没法"顺手加个 split-K"**，因为框架在三个层面各自已经做了决策，而且它不问你。

### 3.1 CPU 端：线程数就是 split 份数

ATen 的 CPU 归约内核里，split-K 是逐字写出来的。`torch/include/ATen/native/cpu/Reduce.h` 第 224~245 行：

```cpp
// 摘自 torch 2.14.0 的 ATen/native/cpu/Reduce.h:224-245
acc_t total_acc = init;
auto numel = sub_iter.numel();
if (numel < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ||
    at::in_parallel_region()) {
  total_acc = reduction_body(total_acc, 0, numel);       // 串行：一个累加器
} else {
  int max_threads = at::get_num_threads();
  std::vector<acc_t> buffer((unsigned)max_threads, init); // S 个独立累加器
  at::parallel_for(0, numel, internal::GRAIN_SIZE,
    [&](int64_t begin, int64_t end) {
      auto& acc = buffer[at::get_thread_num()];           // 每线程自己的部分和
      acc = reduction_body(acc, begin, end);
    }
  );
  for (const auto i : c10::irange(max_threads)) {
    total_acc = ops.combine(total_acc, buffer[i]);         // 顺序合并 S 份
  }
}
```

对照一下就能看出完整对应关系：

- `numel` 就是 $K$（归约轴长度）
- `parallel_for(0, numel, GRAIN_SIZE)` 就是按 $K$ 切块
- `std::vector<acc_t> buffer(max_threads, init)` 就是 workspace，每份一个槽
- `ops.combine(total_acc, buffer[i])` 顺序 merge，就是 ReduceSplitK 那个循环

所以 **CPU 上 $S$ = 线程数**，而且 `GRAIN_SIZE = 32768`（`ATen/TensorIterator.h:77`）是那道「值不值得切」的门槛：`numel < GRAIN_SIZE` 直接走串行，一份都不切。

这件事的直接后果是：**它改变数值结果**。因为浮点加法不满足结合律，$S$ 变了，累加顺序就变了。这不是 bug，是 split-K 的固有属性，而且**任何** split-K 实现都有——包括 GPU 上的 atomicAdd（顺序由硬件调度决定，连跑两次都可能不同）。

下面这份代码就是在观测框架自己的 split-K：只改线程数，不改数据、不改代码：

```python
import torch

print("=== 框架内部的 split-K：改线程数就改结果 ===")
torch.manual_seed(0)
N = 1 << 23
x = torch.randn(1, N)
ref = x.double().sum(1)
print(f"N = 2^23 = {N}, fp64 参考 = {ref.item():.6f}")
print(f"{'线程数':>6} {'与单线程结果的差':>18} {'与 fp64 的相对误差':>20}")
base = None
for nt in [1, 2, 4, 8]:
    torch.set_num_threads(nt)
    r = x.sum(1)
    if base is None:
        base = r.clone()
    d = (r.double() - ref).abs() / ref.abs()
    print(f"{nt:>6} {(r - base).abs().item():>18.3e} {d.item():>20.3e}")
```

```text
=== 框架内部的 split-K：改线程数就改结果 ===
N = 2^23 = 8388608, fp64 参考 = -165.773442
   线程数           与单线程结果的差         与 fp64 的相对误差
     1          0.000e+00            3.654e-06
     2          1.221e-04            2.918e-06
     4          8.545e-04            1.500e-06
     8          0.000e+00            3.654e-06
```

第一列是「同一份输入，只改 $S$，结果差多少」。$N=2^{23}$ 个 randn，绝对差 1.2e-04 ~ 8.5e-04（相对真值 165.77 是 1e-6 量级）。

工程上这意味着一件事：**训练/推理的 bitwise 可复现性会被 $S$ 破坏**。换一张卡（SM 数不同 → Inductor 的 `split` 不同）、换一个线程数（CPU）、甚至同一张卡换个启动顺序（atomicAdd），结果都会变。想复现就得上 `torch.use_deterministic_algorithms(True)`，代价是放弃所有会引入不确定性的 kernel。

第二列还有一件事值得注意：**`nt=8` 的结果和 `nt=1` 一模一样**（都是 3.654e-06，差值恰好 0）。这不是巧合，是这台机器的 OpenMP 线程上限就是 4（`omp_get_max_threads() = 4`），把 `set_num_threads` 设到 8 之后框架退回了串行路径。

**这条比数字本身更重要：$S$ 不是一个你能任意拧的旋钮。** 它由「线程池真正给了几个线程」「buffer 有几个槽」「grader 粒度能不能整除」这些实现细节共同决定，你能设的值和实际生效的值不是一回事。GPU 那边同理——Inductor 的 `split` 由 `multi_processor_count` 算出来，你也不知道硬件最后怎么排的。

第二列里误差也有下降迹象（3.65e-06 → 1.50e-06），但这可能是这组数据的偶然。专门用多随机种子、换成与数据量级无关的指标复核了一遍：

```python
import torch, statistics

# 指标：|误差| / (eps * Σ|x|)。这是浮点累加的标准误差界，归一化后与数据量级无关
print("=== split 份数对 fp32 累加精度的影响 ===")
N = 1 << 24
EPS = torch.finfo(torch.float32).eps
splits = [1, 2, 4, 8, 16]
acc = {s: [] for s in splits}
for seed in range(8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, generator=g)
    ref = x.double().sum().item()
    denom = EPS * x.abs().double().sum().item()
    for s in splits:
        torch.set_num_threads(s)
        acc[s].append(abs(x.sum().item() - ref) / denom)
print(f"N = 2^24, eps = {EPS:.3e}, 归一化误差 = |err| / (eps*Σ|x|)")
print(f"{'S':>4} {'中位数':>12} {'最小':>12} {'最大':>12}")
for s in splits:
    v = acc[s]
    print(f"{s:>4} {statistics.median(v):>12.3e} {min(v):>12.3e} {max(v):>12.3e}")
```

```text
=== split 份数对 fp32 累加精度的影响 ===
N = 2^24, eps = 1.192e-07, 归一化误差 = |err| / (eps*Σ|x|)
   S          中位数           最小           最大
   1    2.919e-04    2.263e-05    7.129e-04
   2    4.411e-04    5.163e-05    1.124e-03
   4    3.141e-04    9.063e-05    8.875e-04
   8    4.594e-04    7.462e-05    9.408e-04
  16    5.304e-04    2.620e-05    9.567e-04
```

这里用的指标是 $\dfrac{|\text{error}|}{\varepsilon \sum|x_i|}$ —— 分母就是浮点累加的标准误差界，所以这个比值衡量的是「你的累加树有多好」，和数据的绝对值、正负号都无关，不会像相对误差那样在被求和接近 0 时炸掉。

结论很干脆：**$S$ 从 1 到 16，比值在 2.9e-04 ~ 5.3e-04 之间随机摆动，没有系统性趋势。** 而且最小值（2.3e-05，出现在 $S=1$）和最大值（1.1e-03，出现在 $S=2$）都出现在相邻档位——说明波动来自随机种子，不来自 $S$。

换一组量级跨度更大的病态数据（每 1000 个塞一个 $10^4$ 的大值、其余 $\sim 10^{-3}$）也复核过，结论一致：没有趋势。

为什么没有趋势？因为**误差的主导因素不是 $S$**。累加误差主要取决于「当前累加值的量级」和「累加器位宽」。$S$ 变大让每份的累加长度变短（好），但份数变多又引入更多层合并（坏），两个效应同量级，互相抵消。真正把它压下去的是 fp32 累加器本身（$1.2\times10^{-7}$ 的机器 epsilon），以及框架内部那套向量化的多累加器结构——**$S$ 只是这套结构最外面的一层，改动它对总误差几乎无影响**。

所以这一节的结论是负面的，但很重要：**别拿精度当 split-K 的选型依据**。它既不是理由也不是借口。它唯一的确定性后果是「结果会变」，而不是「结果会变差」。

### 3.2 GPU 端：Inductor 的 cooperative reduction

PyTorch 在 GPU 上对 reduction 类算子有一套叫 **cooperative reduction** 的 split-K 实现，位置在 `torch/_inductor/codegen/triton.py`。它的做法和上面 CPU 那套、和 CUTLASS 那套都不一样，值得单独说。

**第一步，决定切不切。** `torch/_inductor/choices.py:420-439`：

```py
# 摘自 torch 2.14.0 的 _inductor/choices.py:420-439
xhint = V.graph.sizevars.optimization_hint(numel, fallback=2)
if xhint <= 8:
    threshold = 32768 * xhint
elif xhint <= 16:
    threshold = 2097152
else:
    return False                                  # 输出元素太多，不需要切
return V.graph.sizevars.statically_known_geq(reduction_numel, threshold)
```

注意 `xhint > 16` 时直接 `return False`。这就是 $P/C$ 那道闸的另一种写法：**输出元素（能并行的维度）超过 16 个，说明并行度够用，不切。** 和上一节表格里 $M=N=4096$ 一行说的是同一件事。

**第二步，切几份。** `torch/_inductor/heuristics/triton_codegen/reduction.py:551-555`：

```py
# 摘自 torch 2.14.0 的 _inductor/heuristics/triton_codegen/reduction.py:551-555
target = last_power_of_2(triton_meta["device"].multi_processor_count)
split = max(1, min((rnumel, target // xnumel, TRITON_MAX_RSPLIT)))
```

`target // xnumel` 就是 $\lfloor P/C \rfloor$，$P$ 向下取到 2 的幂。这和我们的 $S^* = \min(P/C, \ldots)$ 是同一个形状——只不过它用 `rnumel`（$K$ 的长度）当上限，而公式用的是归约成本推出来的 $S^*$。实现在这里选了更简单也更保守的那条。

**第三步，怎么归约。** 这是最新颖的部分。`codegen/triton.py:6094-6125` 的 `codegen_cooperative_reduction_peer_combine`：

```py
# 摘自 torch 2.14.0 的 _inductor/codegen/triton.py:6094-6125（注释原文）
"""
Generate code to save a [XBLOCK, RSPLIT] temporary workspace, where each thread
block writes a different column.  After the barrier, every thread block loads
the completed value so that it can compute the final value independently.
"""
tl.store(result_ws + (xindex * RSPLIT + rsplit_id), result, mask)
# ... barrier 之后 ...
peers = tl.load(result_ws + (xindex * rsplit_arange), rsplit_mask, ...)
```

它既不是 atomicAdd，也不是两趟 kernel，而是：**每个 CTA 把部分和写成 workspace 的一列，跨 CTA barrier 之后，每个 CTA 各自把整行 $S$ 个数读回来重算一遍。**

这个设计的好处是没有 master-slave 委派：barrier 之后所有 CTA 同时完成，不需要再造一个 kernel 或者指定一个 CTA 做归约。代价是同一份数据被读 $S$ 次（$S$ 个 CTA 各读一遍全量），总读字节是 $S^2 \cdot b$ 而不是 $S \cdot b$。

代价之所以能接受，是因为它只在 $S \le P/C$ 且输出很小的时候启用——$S$ 通常个位数到几十，而输出 tile 只有几百 KB。

**barrier 本身**在 `runtime/triton_helpers.py:959-990`，是个用单个 uint32 信号量实现的 sense-reversing barrier：最高位当 flag，低 31 位当计数器，program 0 先加 `0x80000000 - (arrive_count - 1)`，其余各加 1，累加到 `0x80000000` 时最高位翻转，所有人自旋到 flag 变化。

这个做法的隐含要求是**所有 CTA 必须同时驻留**（grid 必须能一波放下），否则先到的 CTA 会等到死。这就是为什么 `split` 必须 $\le P/C$——不是性能考虑，是正确性考虑。

**最后一个关键事实**：这个特性**默认是关的**。`torch/_inductor/config.py:2139-2141`：

```py
# 摘自 torch 2.14.0 的 _inductor/config.py:2139-2141
cooperative_reductions = (
    os.environ.get("TORCHINDUCTOR_COOPERATIVE_REDUCTIONS", "0") == "1"
)
```

而且 `choices.py` 里还有一道 `if not config.triton.cooperative_reductions or device.type == "cpu": return False`。所以本机（CPU + 没有 triton）既跑不了它，也验证不了它，上面所有关于 GPU 行为的描述都只到「读源码」这一级。要真跑得在 CUDA 机器上开这个环境变量，然后用 `TORCH_LOGS="output_code"` 看生成的 Triton kernel 里有没有 `x_grid_barrier`。

### 3.3 手写的时候怎么写

Triton 里的 split-K 是最短的。第二个 program 维就是 $S$，末尾一次 `tl.atomic_add`：

```py
# 不运行：本机没有 triton，也无法编译 GPU kernel
@triton.jit
def splitk_gemm(a_ptr, b_ptr, c_ptr, M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                SPLIT: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)                  # 第 3 维就是 split-K 的份
    # 每份只负责 K 的一段，向上取整对齐到 BLOCK_K
    k_per = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT) * BLOCK_K
    k_lo = pid_k * k_per
    k_hi = tl.minimum(k_lo + k_per, K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_lo, k_hi, BLOCK_K):
        a = tl.load(a_ptr + ..., mask=..., other=0.0)
        b = tl.load(b_ptr + ..., mask=..., other=0.0)
        acc += tl.dot(a, b)
    # 注意：epilogue 的 bias/activation 绝不能写在这里，见第四节
    tl.atomic_add(c_ptr + ..., acc)
```

调用方必须先 `c.zero_()`，因为 `atomic_add` 是累加不是赋值。这一句清零在后文会被算成一笔实实在在的开销。

CUTLASS 那边是两趟的经典实现。`include/cutlass/gemm/kernel/gemm_splitk_parallel.h` 里 K 迭代数直接均分：

```cpp
// 摘自 CUTLASS 上游 main 分支 gemm_splitk_parallel.h（Params 构造函数）
int full_gemm_k_iterations = problem_size.k() / Mma::Shape::kK;
int gemm_k_iterations = full_gemm_k_iterations / grid_tiled_shape.k();
gemm_k_size = gemm_k_iterations * Mma::Shape::kK;
```

`grid_tiled_shape.k()` 就是 $S$。部分和写 workspace，第二趟由 `cutlass/reduction/kernel/reduce_split_k.h` 收尾，它的核心循环只有一个细节值得学：

```cpp
// 摘自 CUTLASS 上游 main 分支 reduce_split_k.h，kPartitionsPerStage = 4
for (int k = 0; k < params.partitions; k += kPartitionsPerStage) {
  for (int i = 0; i < kPartitionsPerStage; ++i) {          // 一次装载 4 份
    if (k + i < params.partitions) {
      workspace_frag[i] = *reinterpret_cast<FragmentWorkspace const *>(workspace_ptr);
      workspace_ptr += params.partition_stride;
    }
  }
  for (int i = 0; i < kPartitionsPerStage; ++i) {          // 再 4 路累加
    if (k + i < params.partitions) {
      accumulator = reduction_op(accumulator, workspace_frag[i]);
    }
  }
}
```

一次装 4 份、攒够了再算，把 4 次独立的 workspace 读取流水重叠起来。这和 043 期「测内存带宽要 4 路独立累加器」是同一个道理：单个累加器是串行依赖链，4 路才能把延迟藏住。**归约 kernel 本身就是个访存受限的小 kernel，它遵循的规律和别的访存 kernel 没有区别。**

---

## 四、围绕该领域展开

split-K 是那种「改动很小、影响面很宽」的优化。它一旦打开，上下游一串东西都得跟着改。

### 4.1 epilogue 必须整个挪到归约之后（最重要）

048 期讲了 epilogue 的写回路径：累加器出 Tensor Core 之后过 SMEM、转 dtype、TMA store。那时假设的累加器是**最终值**。

split-K 之后它不再是最终值，它只是一份**部分和**。于是 epilogue 里所有非线性算子都失去交换性——$f(a+b) \neq f(a)+f(b)$。

这不是小误差，是量级错误。下面用真数据量一下（$a, b$ 各 2048×2048 标准正态，"正确"是归约后做激活，"错误"是每份各做一遍再相加）：

```python
import torch

print("=== epilogue 放在归约前 vs 归约后 ===")
torch.manual_seed(0)
M, N = 2048, 2048
a = torch.randn(M, N)
b = torch.randn(M, N)

acts = {
    "relu":    torch.relu,
    "gelu":    torch.nn.functional.gelu,
    "sigmoid": torch.sigmoid,
    "exp":     torch.exp,
    "x*x":     lambda t: t * t,
}
print(f"{'算子':>8} {'正确(归约后)':>14} {'错误(分别做)':>14} {'最大绝对误差':>14} {'均值偏差':>9}")
for name, f in acts.items():
    right = f(a + b)
    wrong = f(a) + f(b)
    rel = (wrong.mean() - right.mean()).abs() / right.abs().mean()
    print(f"{name:>8} {right.mean().item():>14.6f} {wrong.mean().item():>14.6f} "
          f"{(right-wrong).abs().max().item():>14.6f} {rel.item():>9.3f}")

print("\n=== relu 的偏差可以用解析式对上 ===")
import math
print(f"E[relu(a)+relu(b)] / E[relu(a+b)] 理论 = 2/sqrt(2) = {2/math.sqrt(2):.6f}")
r = torch.relu(a) + torch.relu(b)
s = torch.relu(a + b)
print(f"实测比值 = {r.mean().item()/s.mean().item():.6f}")
```

```text
=== epilogue 放在归约前 vs 归约后 ===
      算子        正确(归约后)        错误(分别做)         最大绝对误差      均值偏差
    relu       0.563990       0.797718       3.285918     0.414
    gelu       0.460450       0.564065       3.452722     0.184
 sigmoid       0.499992       0.999991       0.941285     1.000
     exp       2.718643       3.297555    1222.969604     0.213
     x*x       1.999233       1.999711      27.081877     0.000

=== relu 的偏差可以用解析式对上 ===
E[relu(a)+relu(b)] / E[relu(a+b)] 理论 = 2/sqrt(2) = 1.414214
实测比值 = 1.414419
```

逐条读：

- **sigmoid 的偏差恰好 1.000**。因为 $\sigma(a)+\sigma(b)$ 的均值约等于 1（两个 0.5），而 $\sigma(a+b)$ 约等于 0.5。这不是"略有偏差"，是**结果完全错**。
- **relu 系统性偏高 41%**，而且可以解析验证。$a,b \sim N(0,1)$ 时 $\text{relu}(a+b)$ 的自变量是 $N(0,2)$，$\sigma=\sqrt2$，所以 $E[\text{relu}(a+b)] = \sqrt2/\sqrt{2\pi} = 0.5642$；而 $2E[\text{relu}(a)] = 2/\sqrt{2\pi} = 0.7979$。**比值恰好 $\sqrt2 = 1.414214$，实测 1.414419**，差 0.015%。解析式对到小数点后四位，说明这不是数值噪声，是 Jensen 不等式的必然结果：relu 是凸函数，$E[f(X)+f(Y)] \ge E[f(X+Y)]$。实测的 0.563990 / 0.797718 也分别贴住理论值 0.5642 / 0.7979。
- **exp 的最大绝对误差 1223**。exp 长得快，$e^{a}+e^{b}$ 和 $e^{a+b}$ 差了几个数量级。
- **`x*x` 的均值偏差是 0.000，但最大绝对误差 27.08。** 这一行是最阴险的：$a^2 + b^2$ 和 $(a+b)^2$ 的均值只差 $E[a^2+b^2-(a+b)^2] = -2E[ab] = 0$，交叉项均值为零，所以**任何只看均值的检查都发现不了它**。但逐元素错了 $2ab$，量级和 $a^2$ 本身一样大。

最后一条给出一个很实用的判据：**打印 mean 不足以验证 epilogue 顺序，必须逐元素比**。而且它解释了为什么这类 bug 能活到生产——loss 看起来正常，梯度均值看起来正常，只有某些下游统计量会悄悄偏掉。

所以规则很硬：**split-K 打开时，epilogue 里的 bias、activation、甚至 `beta*C` 都必须延后。**

有意思的是 CUTLASS 把这个规则写进了代码结构：`reduce_split_k.h:226` 那一行 `output_op(accumulator, source_frag)` 在归约循环**之后**才执行，而 source（即 $C$）也是**只在归约完成后读一次**（`if (output_op.is_source_needed())` 才读）。如果每份都读一次 $C$，16 MB 的输出配 $S=64$ 就要多读 4 GB —— 这个算术在第五节会变成一条具体的判据。

顺带说一个可以省的：$\alpha$ 和 $\beta$ 这类**线性**系数是可以拆的，$\alpha(a+b) + \beta c = (\alpha a + \beta c) + \alpha b$，本机实测 max$|\Delta| = 1.9\times10^{-6}$（纯浮点舍入量级）。所以归约 kernel 里做 `alpha*acc + beta*C` 是安全的，不安全的是任何非线性。

### 4.2 它与 048 期（epilogue）的关系

048 期的判据 $K^* = \text{屋脊点} \times b/2$（$K$ 多深时 epilogue 的写回不再可忽略）回答的是「一个 tile 的累加器写回 HBM 要花多久」。split-K 把这个前提整个换掉了：**写回的对象从 HBM 变成了 workspace，写的次数从 1 变成 $S$。**

于是 048 期那个「写回地板只占 1%」的乐观结论在 split-K 下不再成立——不是因为它算错了，而是因为条件变了。写回字节乘了 $S$，把它从噪声变成了主项。这也是本期 $S^*$ 公式里那一项的来源：$S \cdot M_tN_tb / BW$ 就是 048 节 4 节的写回账，只不过要乘 $S$。

### 4.3 它与 046 期（cluster / DSMEM）的关系

046 期讲了 cluster 内的 DSMEM：同一个 cluster 的 CTA 可以直接读写彼此的 shared memory，跨 SM 走 DSMEM 是 181 cycle，比经 L2 少 32%（由这两个数反推，经 L2 约 266 cycle；数据出自 Hopper 微基准 arXiv:2501.12084）。

这是归约的**第四种**出口，也是唯一一种不经过 HBM 的：

| 出口 | 延迟量级 | 容量上限 | 适用 |
|---|---|---|---|
| atomicAdd 到 HBM | 高（每次都是一个 L2/HBM 原子操作，还带串行化） | 无 | 份数少、能容忍走一次 L2 |
| workspace + 第二趟 kernel | 一次 HBM 往返 + kernel launch | 显存 | 通用，最稳 |
| workspace + grid barrier（Inductor） | 一次 HBM 往返 + 自旋 | 显存，且 grid 必须一波放下 | 中等份数 |
| **DSMEM（cluster 内）** | **片上，~181 cycle** | **DSMEM 总容量（Hopper cluster=8 时 1816 KB）** | **$S$ 小、tile 小** |

DSMEM 的限制在容量那一列：Hopper 每个 SM 的 shared memory 是 228 KB，cluster 8 个也就 1816 KB。一个 $128\times128$ 的 fp32 输出 tile 是 64 KB，$S=8$ 就要 512 KB —— 装得下，但会挤压主循环的流水线 buffer（044 期算过 4 级流水占 192 KiB，只剩 35 KiB）。所以 DSMEM 归约只适合小 tile、小 $S$ 的场景。

### 4.4 它与 007 期（通信原语）的关系

上一节的四种出口，本质上是 007 期那套集合通信原语的**单卡内版本**：

| 分布式 | 单卡内 |
|---|---|
| ReduceScatter | 各 CTA 算各自 K 段，部分和留在原地 |
| AllReduce | 结果要发给所有 CTA（Inductor 的全冗余重算做的就是这个） |
| AllGather | 每个 CTA 读回全部 $S$ 份部分和 |
| 归约树 | 顺序 merge vs 树形 merge |

一个容易忽略的对应：**分布式训练里的 Ring AllReduce 之所以带宽最优，正是因为它没有中心节点；而 Inductor 的"每个 CTA 各自读全量重算"是它的极端反例——带宽 $O(S^2)$ 而不是 $O(S)$。** 只因为 $S$ 小、tile 小，这个平方项才没有暴露出来。

### 4.5 它与 040 期（算子融合）的关系

040 期讲的是「把相邻的 elementwise 融进 GEMM 的 epilogue，省一趟 HBM 往返」。在 split-K 下这条优化会被**部分撤销**：

- 线性算子（bias、scale）能留，因为可以拆到归约之后
- 非线性算子（gelu、relu、exp）必须移到归约 kernel 里

移出去之后，它又多了一趟 HBM 往返——归约 kernel 读完 workspace 算出结果，紧接着做激活，这倒是能融进同一个 kernel；但如果激活后面还跟着别的 elementwise（比如残差相加），融合链就断了。

所以「开不开 split-K」和「能不能融合」是一对矛盾。这解释了为什么 Inductor 把 cooperative reduction 设成默认关：它是给「形状实在太窄、不切就没法跑」的兜底，不是通用加速开关。

### 4.6 它是 persistent scheduling 的对偶

`torch/_inductor/kernel/mm_common.py:32-40` 里有一个函数，恰好是 split-K 的镜像：

```py
# 摘自 torch 2.14.0 的 _inductor/kernel/mm_common.py:32-40
@SymbolicGridFn
def persistent_mm_grid(M: int, N: int, meta: dict[str, Any], *, cdiv, min):
    """Defines the grid for persistent kernels."""
    return (
        min(meta["NUM_SMS"], cdiv(M, meta["BLOCK_M"]) * cdiv(N, meta["BLOCK_N"])),
        1, 1,
    )
```

- **persistent**：当 $C > P$（并行度过剩），把 grid 压到 `min(NUM_SMS, C)`，一个 CTA 连吃多个 tile。省的是调度开销和 cache 复用。
- **split-K**：当 $C < P$（并行度不足），把 $K$ 切开制造出 $C \times S$ 个 CTA。

两者是同一个问题的两个方向，判据都是 $P/C$ 这个比值。工程上应该先算它：**$P/C > 2$ 才考虑 split-K，$P/C < 0.5$ 才考虑 persistent，中间那段什么都别做。**

### 4.7 一条实测的代价曲线

前面所有分析都假设归约是一次干净的批量读取。这里把「归约本身要花多少」单独量出来，$M=N=512$、$K=8192$ 的 GEMM 作分母（$S=64$ 时 workspace 是 64 MiB，约为 M5 的 L2 四倍，确保它走 HBM 而不是 cache）：

```python
import time, torch

torch.set_num_threads(4)

def bench(fn, reps=25, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

M = N = 512; K = 8192
a = torch.randn(M, K); b = torch.randn(K, N)
t_gemm = bench(lambda: a @ b)
print(f"GEMM {M}x{N}x{K}: {t_gemm*1e3:.4f} ms, {2*M*N*K/t_gemm/1e9:.0f} GFLOPS")

print(f"\n{'S':>3} {'ws MiB':>7} {'清零 ms':>9} {'sum(0) ms':>10} {'顺序累加 ms':>12} {'顺序/sum0':>9}")
for S in [32, 64]:
    ws = torch.randn(S, M, N)
    t_zero = bench(lambda: ws.zero_())
    t_pass = bench(lambda: ws.sum(0))
    def seq():
        out = ws[0].clone()
        for s in range(1, S):
            out += ws[s]
        return out
    t_seq = bench(seq)
    print(f"{S:>3} {ws.numel()*4/2**20:>7.1f} {t_zero*1e3:>9.4f} {t_pass*1e3:>10.4f} "
          f"{t_seq*1e3:>12.4f} {t_seq/t_pass:>9.2f}")

print(f"\nS=64：归约一次 pass 占 GEMM 的 {t_pass/t_gemm:.2f} 倍")
print(f"S=64：清零 64 MiB 占 GEMM 的 {t_zero/t_gemm:.2f} 倍")
```

```text
GEMM 512x512x8192: 2.5725 ms, 1670 GFLOPS

  S  ws MiB     清零 ms  sum(0) ms      顺序累加 ms   顺序/sum0
 32    32.0    0.3426     0.4929       0.7119      1.44
 64    64.0    0.7012     1.3013       1.4556      1.12

S=64：归约一次 pass 占 GEMM 的 0.51 倍
S=64：清零 64 MiB 占 GEMM 的 0.27 倍
```

三个数字都值得看：

**归约占 GEMM 的 51%。** $S=64$ 时读 64 MiB workspace 花 1.30 ms，而 GEMM 本身 2.57 ms。这直接印证了 U 形表：$S=42$ 附近归约项和计算项打平，到 $S=64$ 归约已经是同一量级的主项。**这就是 $S^*$ 存在的物理原因，不是理论玩具。**

**顺序累加和一次 pass 差不多（1.12 倍 / 1.44 倍）。** 这一行值得停一下，因为它和直觉相反——CUTLASS 的 `ReduceSplitK` 是顺序累加的，看起来应该比一次 pass 的 `sum(0)` 吃亏。这里没吃亏，原因是 64 MiB 已经走 HBM，两者都带宽受限，`out += ws[s]` 那点 read-modify-write 的额外流量被 write-combining 吸收了。

**所以「怎么归约」只在 workspace 很小的时候才重要。** 同样这两档在 16 MiB（恰好等于 M5 的 L2）上重测，顺序累加会比一次 pass 慢好几倍——但那个尺寸的绝对值抖动到 ±50%，按纪律不引用具体倍数。定性结论是：**当归约的字节数超过 cache，讨论归约算法没有意义；只有当它落在 L2 里、时间被 kernel launch 而非带宽支配时，`kPartitionsPerStage` 这种参数才有价值。**

那 CUTLASS 为什么还要 `kPartitionsPerStage = 4`？因为它面对的场景正是后者：`ReduceSplitK` 的每个 partition 只有几百 KB，一个 CTA 读完所有 partition 也就几 MB，全在 L2 里。这种尺寸下，64 次串行的 workspace 访问每次都要等一次延迟，4 路预取就能把 4 次延迟叠成 1 次加 3 次重叠。**这和 043 期「测带宽要 4 路独立累加器」是同一个动作**，只不过那里藏的是访存延迟，这里藏的是归约依赖链。

**清零占 GEMM 的 27%。** atomicAdd 方案必须先 `zero_()`，这是纯加的成本：64 MiB 写一遍花 0.70 ms。注意这个比例会随输出规模恶化——按 048 期测的写带宽地板 87.6 GB/s，$M=N=1024$、$S=64$ 时 workspace 是 256 MiB，清零要 $256/87.6\times10^3 \approx 2.9$ ms，而那个形状的 GEMM 本身约 8.5 ms，**清零单独就吃掉 35%**。而 workspace 越大，你越不可能指望它落在 L2 里。

---

## 五、什么时候该用 / 不该用

把前面所有判据合成一张表。第一道闸永远是 $P/C$，不是 $S^*$：

| 条件 | 结论 |
|---|---|
| $P/C \le 1$（CTA 数已超 SM 数） | **别切**。纯亏：输出写 $S$ 遍，K 循环一点没变快 |
| $P/C > 1$ 且 $K$ 很大（$\gtrsim 4096$） | 可切，$S = \min(\lfloor P/C \rfloor, S^*, 64)$ |
| $P/C > 1$ 但 $K$ 小（$\lesssim 1024$） | 别切。$t_K$ 本来就短，归约的固定开销（barrier / launch）盖过收益 |
| 输出 tile 特别大（$M_tN_tb \gtrsim$ 几百 KB） | 谨慎。$S$ 份 workspace 会直接撞显存 |
| epilogue 里有非线性算子 | 可以切，但必须把激活移到归约之后，融合链会断 |
| 需要 bitwise 可复现 | 别切（atomicAdd 尤其，顺序由调度决定） |

**该用的典型场景**，都是「$M \times N$ 天然小、$K$ 天然大」：

- decode 阶段 attention 的 $QK^T$：$M = 1 \sim 8$（batch × head 的 query 数），$K$ = 上下文长度
- 单样本推理的所有 GEMM：$M = 1$
- 大词表的输出投影：$N$ = 词表大小，看起来大，但按 batch=1 拆完 $M=1$
- 低秩分解的 $A\cdot B$：中间维 rank 就是 $K$，通常 8~64

**不该用的**：训练前向的大 GEMM（$C \ge P$ 是常态）、任何 $K < 1024$ 的矩阵乘、以及一切已经在用 tensor parallel 拆过 $K$ 的场合（019 期：TP 已经把 $K$ 切成 $1/N$ 了，再切一刀等于切两遍）。

---

## 六、常见坑

**坑 1：以为 `atomicAdd` 版本不用管 workspace。**
它确实不需要 workspace，但它需要**清零输出**，而且是在每次调用前。$S$ 越大这个成本占比越高（$M=N=512$、$S=64$ 实测是 GEMM 的 27%；workspace 涨到 256 MiB 的外推值是 35%）。更麻烦的是清零本身是个独立 kernel launch，在 launch-bound 的小形状下这个 launch 可能比 GEMM 还贵。对策：把清零融进前一个 kernel（比如上层的 bias kernel 顺手写 0），或者改用 `beta=0` 的写法让归约 kernel 自己初始化。

**坑 2：把 epilogue 留在每个 split 里。**
这是最容易犯、后果最严重的错误。sigmoid 的结果会直接变成"1 而不是 0.5"，relu 系统性偏高 $\sqrt2$ 倍。而且它**不会报错、不会 NaN、loss 曲线看起来还能降**——只是永远差一截。判断依据：如果部分和里有负数而激活是 relu，你几乎必然踩到。必须用 4.1 节的对照代码验一遍。

**坑 3：`S` 拍脑袋取大。**
$S=256$ 在 H100 的 $M=N=128$ 形状上只有 7.0 倍加速，比 $S=8$ 的 7.7 倍还差。而 $S=42$ 是 21.4 倍。**U 形底部很浅但两侧很陡**，多切 6 倍直接损失 3 倍性能。取 $S^*$ 的一半到两倍都安全，超出这个范围就要重算。

**坑 4：忘了 `S` 会改变数值，然后怪模型训不出来。**
本机实测：只改线程数（等价于改 $S$），$2^{23}$ 个元素的和最多差 $8.5\times10^{-4}$。分布式训练里如果不同 rank 的 $S$ 不一致（GPU 型号不同、或某张卡被别的进程占了一半 SM），梯度就会静默地不一致。$S$ 必须是一个**跨 rank 一致**的常量，不能由 `multi_processor_count` 这种运行时值决定——而后者恰好是 Inductor 的做法（`last_power_of_2(multi_processor_count)`），所以异构集群上要显式固定它。

---

## 七、一句话总结

split-K 是**用归约带宽换并行度**的交易，而汇率是 $S^*/2$ 倍而不是 SM 数那么多——因为 $S^* = \sqrt{2K \cdot BW/({P_{sm}} b)}$ 与 tile 尺寸无关、与硬件代际基本无关，所以你永远只能拿回理想并行度的一小部分（A100 25%、H100 16%、B200 14%），且硬件越新拿回得越少。

---

<details>
<summary>今日练习</summary>

**题 1**：$M=N=256$、$K=16384$、fp16 输入、fp32 输出，跑在 H100 上。$P/C$ 是多少？按 $S^*$ 公式该切几份？实际取几份？预计加速比多少？

<details>
<summary>答案</summary>

tile 取 $128\times128$，则 $C = 2 \times 2 = 4$。$P/C = 132/4 = 33$。

$S^* = \sqrt{2 \cdot 16384 \cdot 3.35\times10^{12} / (7.495\times10^{12} \cdot 4)} = \sqrt{3661} = 60.5$。

实际取 $S = \min(33, 60.5, 64) = 33$ —— **卡在并行度余量上，不是卡在 $S^*$ 上**。

$T(33) = t_K/33 + 33 \cdot 65536/3.35\times10^{12}$，其中 $t_K = 2\cdot128\cdot128\cdot16384/7.495\times10^{12} = 71.6$ us。
所以 $T = 2.17 + 0.65 = 2.82$ us，加速 $71.6/2.82 = 25.4$ 倍（对比 $S=1$）。

关键点：$P/C = 33$ 比 $S^* = 60.5$ 更紧，所以这里的瓶颈是「没有更多 SM 可派」，不是「归约太贵」。

</details>

**题 2**：某个 split-K kernel 里，epilogue 是 `gelu(acc + bias)`，然后发现部分和里有正有负。为什么这个写法对小 $K$ 时"看起来没问题"，大 $S$ 时就崩了？

<details>
<summary>答案</summary>

因为 `gelu` 在 0 附近近似恒等，在负半轴快速衰减到 0。

$S=1$ 时 $acc$ 是完整的和，`gelu(acc+bias)` 就是正确答案，当然没问题。$S$ 变大后每份只是一段 $K$ 的部分和：**部分和的量级比总和小 $S$ 倍**，所以它更容易落在 0 附近甚至正负翻转。

- 落在正半轴：`gelu` 近似恒等，误差不大，所以"看起来能用"
- 落在负半轴：`gelu(x) ≈ 0`，这一份的贡献被**整个抹掉**，然后你再把 $S$ 份加起来，就只剩正的那几份

$S$ 越大，每份越接近 0，被 gelu 抹掉的比例越高。实测在标准正态上 relu 版本偏高 $\sqrt2$ 倍（bias 项可比），gelu 版本偏高 22%（0.564/0.461，都是从同一组实验里读出来的）。

教训：**判断"能不能提前做 epilogue"的唯一依据是算子的可拆性，不是"跑起来看着差不多"。** 线性算子（含 bias、scale、`beta*C`）可以拆；任何凸/凹/非单调算子（relu、gelu、exp、sigmoid）都不行，且误差随 $S$ 单调恶化。

</details>

**题 3**：你手上有一个 $M=1$、$N=4096$、$K=4096$ 的矩阵乘（单个 token 的输出投影）。用 CUDA Graph 把这一个 kernel 的 launch 开销消掉了，测得 8 us。按本期公式，这个形状用 split-K 值不值？如果换成 GEMV kernel 呢？

<details>
<summary>答案</summary>

先算 $C$：$M=1$ 时按 $128\times128$ tile 切，$C = 1 \times 32 = 32$ 个 CTA。$P/C = 132/32 = 4.1$。

$S^* = \sqrt{2\cdot4096\cdot3.35\times10^{12}/(7.495\times10^{12}\cdot4)} = 30.25$。

实际 $S = \min(4.1, 30.25, 64) = 4$。

$t_K = 2\cdot1\cdot128\cdot4096/7.495\times10^{12}$。注意这里 $M_t$ 实际只有 1 行有效，但 tile 是按 128 分配的——**这个形状用 GEMM tile 本身就是浪费**，有效算力只有 $1/128$。

所以老实回答：**不值，因为你选错了 kernel。** 这不是 GEMM 的问题，是 $M=1$ 应该用 GEMV。

GEMV 的 AI 恒等于 1（043 期算过：FLOPs = Bytes = 4HSd，H/S/d 全部约掉），它天生是访存受限，$M=1$ 时读一遍 B 矩阵就完事，根本不需要 Tensor Core，也不需要 split-K。$N=4096, K=4096$ 的 fp16 B 矩阵是 32 MiB，按 H100 的 3.35 TB/s 约 $10$ us —— 你测到的 8 us 已经接近这个地板了，**说明这个 kernel 在访存上已经到位，没有任何 split-K 的空间**。

结论：split-K 只在「$M\times N$ 能切出可观的 CTA 数、但不足以喂满 SM」那个窗口里有价值。$M=1$ 的极端形状已经掉出了这个窗口——它的解法是换 kernel，不是切 K。

</details>

</details>
