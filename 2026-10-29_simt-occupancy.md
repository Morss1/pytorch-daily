# PyTorch 每日一课 · 第 045 期

## SIMT 与 Occupancy：warp 调度、延迟隐藏、寄存器压力如何决定并行度

| 项目 | 内容 |
|---|---|
| 日期 | 2026-10-29 |
| 难度 | ⭐⭐⭐⭐⭐ |
| 前置知识 | 知道 warp 是 32 个线程；读过第 043 期（访存层次）和第 044 期（异步拷贝）会更快，但不依赖 |
| 预计阅读 | 35 分钟 |
| 关联 | 第 044 期 · 044 讲「用异步指令把等待填满」，本期讲「用更多 warp 把等待填满」，是同一条延迟隐藏战线的两条分支；第 042 期 · 042 是 CPU 上「谁来执行并行块」，本期是 GPU 上同一问题的完整答案 |

---

## 一、这个领域解决什么问题

先看一组数字。同一个 kernel，一个字符都没改逻辑，只是编译器多分给它 1 个寄存器：

| 寄存器/线程 | 每个 warp 实占 | 一个 SM 能挂几个 block | 活跃 warp | occupancy |
|---|---|---|---|---|
| 32 | 1024 | 8 | 64 | 100% |
| **33** | **1280** | **6** | **48** | **75%** |

从 32 到 33，只多了一个寄存器，能同时驻留的 warp 就从 64 掉到 48，**occupancy 直接砍掉四分之一**。

这不是段子。它能发生，是因为 GPU 的资源分配方式和 CPU 完全不同：CPU 上你开 8 个线程，操作系统给每个线程分配独立的栈，线程之间几乎不抢资源；GPU 上所有 warp 住在一个 SM 里，**共享同一份寄存器堆和同一块 shared memory**，谁能住进来、能住几个，是一道除法题。

而这道除法题的答案，直接决定了这个 kernel 有没有能力把显存喂饱。

所以这个领域要回答的问题是：

> 一个 kernel 在 SM 上到底能同时跑多少活？这个数由什么决定？它为什么能决定性能？又为什么有时候决定不了？

这个问题在 PyTorch 的每一层都会冒出来：你写 `torch.compile` 让 Inductor 生成 Triton kernel，它替你选了 `num_warps=4`；你手写 Triton kernel，也得填 `num_warps` 和 `num_stages`；你用 `ncu` 看指标，第一眼就是 "Achieved Occupancy 12.5%"。这些数字背后是同一套机制。

先给结论，后面逐条算给你看：

1. **occupancy 是手段，不是目的。** 真正决定带宽能不能跑满的量是「同一时刻有多少字节的请求在飞」，occupancy 只是供给这个量的一个渠道（另一个渠道是单 warp 的 MLP，第 044 期讲过）。
2. **occupancy 被三把锁卡死**：寄存器、shared memory、每 SM 的 block 数上限。三把锁任何一把合上，其他两把再富余也没用。
3. **锁不是连续的，是台阶。** 寄存器按 256 个/warp 分配，所以 32→33 这种「只多一个」的跳变会造成 25% 的悬崖。
4. **缺的量可以精确算出来，不用猜。** $\text{在飞字节} = \text{延迟} \times \text{带宽}$。H100 每个 SM 需要 **6.00 KB** 在飞，等价于 48 条 128 B 请求 —— 这就是 occupancy 该给到的下限。

---

## 二、核心思想：候机厅、座位与行李

先建立一个能一直用下去的类比。

一个 SM 是一间**候机厅**，有 **64 个座位**（warp 槽）。旅客是 **warp**（32 个人一组，永远同进同出）。每个人随身带的行李是**寄存器**，候机厅总共有 65536 件行李位；有人还托运了箱子放在**行李房**（shared memory），总共 228 KB。此外还有一条规矩：**同一个团最多占 32 个相连的座位**（每 SM 最多 32 个 block）。

于是"能坐下多少人"就是一道除法：

$$\text{能坐的团数} = \min\left(\left\lfloor\frac{65536}{\text{每人行李} \times 32 \times \text{团人数}}\right\rfloor,\ \left\lfloor\frac{228\,\text{KB}}{\text{每团托运}}\right\rfloor,\ 32,\ \left\lfloor\frac{64}{\text{团人数}}\right\rfloor\right)$$

关键在**行李是按「每 32 人一组」打包称重的**：一个 warp 的 32 个人，每人要 $r$ 件行李，那就是 $32r$ 件，但要向上取整到 256 件的整数倍。这就是那个悬崖的来源：$r=32$ 时 $32 \times 32 = 1024$，正好 4 个 256；$r=33$ 时 $32 \times 33 = 1056$，向上取整成 1280 —— **多一件行李，占位涨 25%**。

现在讲第二层，也是大部分人漏掉的一层。

**坐满不等于飞机准点。** 旅客坐下之后要做的事是「等行李转盘」。假设每个人都要等 4 分钟才拿到行李，那候机厅里坐 64 个人和坐 16 个人，**吞吐是一样的** —— 都是每 4 分钟走 16 个，因为瓶颈在转盘不在座位。只有当"同时等行李的人"多到能把转盘的传送带占满，座位数才开始起作用。

用排队论的话说，这是 Little 定律：

$$N = \lambda \times W$$

其中 $N$ 是系统里同时存在的请求数，$\lambda$ 是完成速率（吞吐），$W$ 是每个请求停留时间。反过来写成我们要的形式：

$$\text{需要的在飞请求数} = \text{目标吞吐} \times \text{单次延迟}$$

如果这个乘积要求的请求数**超过**了硬件能同时挂住的数量，带宽就跑不满；如果**低于**，那么再增加并发单元（occupancy）一分钱收益都没有。

第 044 期已经用 CPU 验证了这条定律的前半段（加并发能把有效延迟压下去）。这一期要做的是把它做成**一把可以量任何机器的尺子**，然后用它去审 occupancy。

---

## 三、SIMT：warp 是怎么被调度的

在算 occupancy 之前，得先知道"挂进去的 warp"到底在干什么。因为 occupancy 的全部意义，就是为了给调度器提供**可以随时换上去的候选人**。

### 3.1 一个 warp 是 32 条并行的车道，共用同一个 PC

CUDA 把 32 个线程编成一个 **warp**。这 32 个线程**共用一条指令流**：一个 PC、一次取指、一次译码，然后同一条指令在 32 条 lane 上各执行一遍。这就是 **SIMT**（Single Instruction, Multiple Threads）。

和 CPU 的 SIMD（比如 NEON / AVX）比，区别在**契约放在哪一层**：

- SIMD 把向量宽度写进**指令**里，程序员和编译器都得显式向量化；
- SIMT 把宽度藏在**硬件**里，你写的是普通标量代码，硬件负责把 32 个线程绑在一起跑。

这个差别的代价在分支上暴露出来。**一个 warp 只有一个 PC，所以 32 条 lane 必须走同一条路径。** 如果第 0~15 号线程走 `if` 分支、第 16~31 号走 `else`，硬件不会把 warp 拆开 —— 它会**先跑 `if` 分支（`else` 那半被掩码掉），再跑 `else` 分支（`if` 那半被掩码掉）**。两条路径串行，代价最多翻倍；如果 32 条 lane 走了 32 条不同路径，理论上就是 **32 倍**。

32 倍是个**上界**，不是常态（编译器对短分支会用谓词执行 if-conversion，根本不产生跳转）。但它揭示的规律是真的：**分支的代价取决于 warp 内部的一致性，而不是分支本身的复杂度。** 所以数据并行 kernel 里那句老话——"先排序再分支"——不是玄学：把 work 按分支条件分桶，让同一个 warp 的 32 条 lane 走同一条路，分支就从"翻倍"变成"免费"。

用 Triton 写的时候这个约束是隐式的：`tl.where` 是逐元素选择（无分支，两个分支的**计算**都会做），Python 层的 `if` 才会真的产生控制流。选哪个取决于哪一边更贵 —— 算一遍再选，还是两条路都算。

### 3.2 SM 里有 4 个调度器，每个每 cycle 发一条

一个 SM 不是一个大脑，是**四个**。它被切成 4 个 sub-partition，每个 sub-partition 有自己的 warp 池和一个 **warp scheduler**，每个 scheduler 每个时钟周期可以发射 **1 条**指令。

所以一个 SM 的理论发射上限是 **4 条指令/cycle**，和它的 64 个 warp 槽没有直接关系。64 个 warp 会被大致均匀地摊给 4 个 scheduler，**每个 scheduler 手上约 16 个**。

这就解释了第七节那张"活跃 warp"表为什么要盯着 64 这个数：

| 活跃 warp | 每 scheduler 可分到 | 状态 |
|---|---|---|
| 16 | 4 | 一旦有 4 个同时等内存，这个 scheduler 就空了 |
| 32 | 8 | 常见的中等配置 |
| 48 | 12 | 表 1 里 33 regs 那档 |
| 64 | 16 | 满 occupancy |

### 3.3 切换是免费的，这是整个设计的地基

CPU 切换线程要保存/恢复寄存器、换栈、可能掉 TLB —— 所以线程是"重"的，一次上下文切换几千个 cycle。GPU 敢让 64 个 warp 抢 4 个发射口，是因为**它根本不做切换**。

原因在寄存器堆的大小：65536 个 32 位寄存器 ÷ 2048 个线程 = **每线程 32 个**。这个数字不是巧合，而是设计目标 —— **寄存器堆被造得刚好能容纳所有在驻留线程的全部状态**，所以每个线程的寄存器永久驻留在物理寄存器堆里，谁也不覆盖谁。调度器把发射口从一个 warp 转给另一个 warp 时，**没有任何东西需要保存或恢复**。

这就是"warp 切换零开销"的物理来源，也是第二节那个除法的物理来源：**每线程 32 个寄存器，就是"满 occupancy"这个状态的定价。**

于是延迟隐藏的机制可以完整描述出来了：

1. warp A 发出一条 `LDG`（全局读），它立刻进入"未就绪"状态 —— 数据要几百个 cycle 才回来；
2. scheduler 不做任何等待，下一 cycle 直接从**就绪队列**里挑 warp B 发射；
3. 只要还有就绪的 warp，发射口就不会空。这就是"用并行度隐藏延迟"。

反过来，**如果 16 个 warp 全都在等内存，这个 scheduler 就真的空转了**。occupancy 的全部价值就在这里：它决定了"就绪队列"里平均有几个候选。

这也解释了第五节会看到的那个现象：加线程和加"每线程的独立请求数"能互相替代 —— 前者是**加候选人数**，后者是**让同一个人同时干几件事**，对 scheduler 来说效果一样。

---

## 四、先把两个常数测出来

尺子需要两个刻度：延迟 $L$ 和峰值带宽 $BW$。这两个数必须实测，因为它们决定了后面所有推算。

先测延迟。用 pointer chasing：把一堆槽位打乱串成单向环，沿着环一跳一跳走。下一跳的地址依赖上一跳的结果，硬件没有任何重叠的机会，测到的就是**裸延迟**。

```python
import os
import sys

# 本机 ninja 装在解释器旁边的 bin 下但不在 PATH 上，cpp_extension 找不到它
os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ["PATH"]

import numpy as np
import time
import torch
from torch.utils.cpp_extension import load_inline

src = r"""
#include <torch/extension.h>
#include <omp.h>

// 单链 pointer chasing：下一跳的地址依赖上一跳的值，硬件无法重叠
double chase(const at::Tensor& ring_t, int64_t start, int64_t steps) {
  const int64_t* ring = ring_t.data_ptr<int64_t>();
  int64_t idx = start;
  double t0 = omp_get_wtime();
  for (int64_t i = 0; i < steps; ++i) {
    idx = ring[idx];
  }
  double t1 = omp_get_wtime();
  return (t1 - t0) + 0.0 * (double)idx;   // 让最后一次访问不被优化掉
}
"""

m = load_inline(
    name="lat_probe",
    cpp_sources=src,
    functions=["chase"],
    extra_cflags=["-O3", "-Xpreprocessor", "-fopenmp"],
    extra_ldflags=["-lomp"],
    verbose=False,
)


def build_ring(nslot, seed=0):
    """构造一个随机置换的单向环：ring[i] 是环上下一个槽位的下标"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nslot).astype(np.int64)
    nxt = np.empty(nslot, dtype=np.int64)
    nxt[perm] = np.roll(perm, -1)
    return torch.from_numpy(nxt)


STEPS = 4_000_000
print("== 单链延迟阶梯（每档走满 4M 步，取 5 次最小）==")
print(f"{'工作集':>10} {'延迟':>9}")
for mb, label in [(0.5, "512KB"), (2, "2MB"), (8, "8MB"), (32, "32MB"), (512, "512MB")]:
    nslot = int(mb * 1024 * 1024) // 8
    ring = build_ring(nslot)
    best = min(m.chase(ring, 0, STEPS) for _ in range(5))
    print(f"{label:>10} {best / STEPS * 1e9:>7.2f}ns")

# 峰值带宽：大张量的顺序读
print()
print("== 顺序读带宽（torch.sum，128 MB）==")
for nt in (1, 4):
    torch.set_num_threads(nt)
    x = torch.randn(128 * 1024 * 1024 // 4)
    x.sum()
    t0 = time.perf_counter()
    for _ in range(5):
        x.sum()
    best = (time.perf_counter() - t0) / 5
    print(f"{nt} 线程: {best * 1e3:.2f}ms -> {x.numel() * 4 / best / 1e9:.1f} GB/s")
```

```text
== 单链延迟阶梯（每档走满 4M 步，取 5 次最小）==
       工作集        延迟
     512KB    2.48ns
       2MB    4.53ns
       8MB    7.01ns
      32MB   33.94ns
     512MB   83.11ns

== 顺序读带宽（torch.sum，128 MB）==
1 线程: 1.72ms -> 77.9 GB/s
4 线程: 1.65ms -> 81.4 GB/s
```

两个数字到手：**DRAM 延迟 83.11 ns**，**峰值带宽 81.4 GB/s**。

顺便注意第二个结果里的一个细节：**1 线程和 4 线程的带宽几乎一样**（77.9 vs 81.4）。这不是测错了。顺序访问时，硬件预取器会自动把一个线程的访问变成几十个在飞的请求 —— 也就是说，**光靠单线程的 MLP 就已经把这台机器的内存喂饱了，加线程在这里只换来 4.5%**。

这正好是本期第一个反直觉结论：并发单元（线程、warp）不是唯一的并发来源，**单单元内部的 MLP 也是**。

而 pointer chasing 那条路，单链只有 1 个请求在飞，所以只能跑到 1.5 GB/s（本期实测；第 044 期在另一个工作集上测到 1.4 GB/s）—— 同样一台机器，差 57 倍，唯一变量就是"同时在飞的请求数"。

---

## 五、核心实验：把并发拆成两个旋钮

现在做正事。我们把"同时在飞的请求数"拆成两个可独立控制的量：

- $T$：线程数，对应 GPU 上的 **occupancy**（挂多少个 warp）
- $M$：每线程的独立链数，对应一个 warp 内部能发多少条**互不依赖**的访存（MLP）

总在飞请求数 $k = T \times M$。

关键设计（不这么设计会得到假结论）：

1. **工作集固定 512 MB**，不随 $k$ 变化。否则扫并发度时每段的尺寸在变，小的那几档会滑进 L2，产出假拐点。
2. **用随机置换环而不是顺序流**。顺序流会让预取器自动提供 MLP，把 $M$ 这个旋钮短接。
3. **每条链一个独立游标**。如果让多个链共用一个累加器，测到的是标量依赖链的延迟，不是吞吐。
4. **总访问次数固定 4M 次**（正好覆盖 512 MB 一遍），这样每档之间可直接比吞吐。

```python
import os
import sys

os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ["PATH"]

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

src = r"""
#include <torch/extension.h>
#include <omp.h>

#define MAXM 64

// T 个线程，每线程 M 条互不依赖的链同时推进；返回总耗时（秒）
double sweep(const at::Tensor& ring_t, int64_t nslot, int64_t T, int64_t M,
             int64_t steps, int64_t salt) {
  const int64_t* ring = ring_t.data_ptr<int64_t>();
  omp_set_num_threads((int)T);
  double t0 = omp_get_wtime();
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int64_t idx[MAXM];
    // 每条链一个独立起点，彼此在环上隔开，避免抢同一条 cacheline
    for (int64_t j = 0; j < M; ++j) {
      idx[j] = ((int64_t)(tid * MAXM + j) * 2654435761LL + salt * 7919LL) % nslot;
    }
    for (int64_t s = 0; s < steps; ++s) {
      for (int64_t j = 0; j < M; ++j) {
        idx[j] = ring[idx[j]];   // M 条互不依赖 → M 个 miss 可以同时在飞
      }
    }
    int64_t acc = 0;
    for (int64_t j = 0; j < M; ++j) acc += idx[j];
    if (acc == -12345) printf("never\n");   // 防止整段被优化掉
  }
  return omp_get_wtime() - t0;
}
"""

m = load_inline(
    name="occ_sweep",
    cpp_sources=src,
    functions=["sweep"],
    extra_cflags=["-O3", "-Xpreprocessor", "-fopenmp"],
    extra_ldflags=["-lomp"],
    verbose=False,
)

NSLOT = 512 * 1024 * 1024 // 8      # 512 MB / 8 B = 67108864 个槽位
rng = np.random.default_rng(7)
perm = rng.permutation(NSLOT).astype(np.int64)
nxt = np.empty(NSLOT, dtype=np.int64)
nxt[perm] = np.roll(perm, -1)       # 随机置换 → 单向环
ring = torch.from_numpy(nxt)
del perm, nxt

TOTAL = 4_000_000                   # 每档总访问次数固定

print(f"{'线程T':>5} {'每线程链M':>9} {'在飞k':>7} {'耗时':>9} {'吞吐':>9} {'摊薄延迟':>9}")
for T in (1, 2, 4):
    for M in (1, 2, 4, 8, 16, 32, 64):
        k = T * M
        steps = TOTAL // k
        m.sweep(ring, NSLOT, T, M, 1000, 1)                     # 预热，不计时
        best = min(m.sweep(ring, NSLOT, T, M, steps, r + 2) for r in range(3))
        total = steps * k
        gbs = total * 128 / best / 1e9                          # 每次 miss 拉一条 128B
        print(f"{T:>5} {M:>9} {k:>7} {best*1e3:>8.1f}ms {gbs:>7.1f}GB/s {best/total*1e9:>7.2f}ns")
```

```text
  线程T     每线程链M     在飞k        耗时        吞吐      摊薄延迟
    1         1       1    336.7ms     1.5GB/s   84.17ns
    1         2       2    168.7ms     3.0GB/s   42.16ns
    1         4       4     93.6ms     5.5GB/s   23.40ns
    1         8       8     49.2ms    10.4GB/s   12.30ns
    1        16      16     25.8ms    19.9GB/s    6.44ns
    1        32      32     14.6ms    35.0GB/s    3.65ns
    1        64      64     12.3ms    41.7GB/s    3.07ns
    2         1       2    170.2ms     3.0GB/s   42.56ns
    2         2       4     86.1ms     5.9GB/s   21.53ns
    2         4       8     48.6ms    10.5GB/s   12.16ns
    2         8      16     25.6ms    20.0GB/s    6.41ns
    2        16      32     13.7ms    37.4GB/s    3.42ns
    2        32      64      8.3ms    62.0GB/s    2.07ns
    2        64     128      6.7ms    76.5GB/s    1.67ns
    4         1       4     87.4ms     5.9GB/s   21.85ns
    4         2       8     44.6ms    11.5GB/s   11.15ns
    4         4      16     25.2ms    20.4GB/s    6.29ns
    4         8      32     13.6ms    37.5GB/s    3.41ns
    4        16      64      7.7ms    66.5GB/s    1.93ns
    4        32     128      6.0ms    85.4GB/s    1.50ns
    4        64     256      5.9ms    86.7GB/s    1.48ns
```

这张表要横着读、也要竖着读。

**横着读（固定 T，加 M）**：一个线程加链数，吞吐几乎线性涨。$M=1 \to 2$ 是 1.5 → 3.0，$\to 4$ 是 5.5，$\to 8$ 是 10.4。每翻一倍链数，吞吐翻一倍 —— 一直翻到撞上带宽墙。

**竖着读（固定 M，加 T）**：加线程的效果和加链数**一模一样**。$T=4, M=1$ 是 5.9 GB/s，$T=1, M=4$ 是 5.5 GB/s；$T=2, M=2$ 是 5.9 GB/s。三个方案在飞请求数都是 4，吞吐都在 5.5~5.9 之间。

**这就是 occupancy 的本质：它只是一个提高 $k$ 的杠杆，和"每个 warp 内部多开几条在飞请求"完全等价。**

三处可以互相对照的等值线：

| 在飞 k | T=4·M=1 | T=2·M=2 | T=1·M=4 |
|---|---|---|---|
| 4 | 5.9 | 5.9 | 5.5 |
| 8 | 11.5 | 10.5 | 10.4 |

另一个维度同样重要 —— **饱和之后，再加并发就是白加**：

| k | 64 | 128 | 256 |
|---|---|---|---|
| 吞吐 | 66.5 GB/s | 85.4 GB/s | 86.7 GB/s |

$k$ 从 128 翻到 256，吞吐只涨了 1.3 GB/s（1.5%）。**多花一倍的并发，买到的是 1.5%。**

还有一处细节很关键：$T=1, M=64$ 只能到 **41.7 GB/s**，而 $T=4, M=16$（同样 $k=64$）能到 **66.5 GB/s**。说明并发单元**不能无限互替** —— 单线程的发射宽度是有限的，64 条链在它手上排不过来。这个上限在 GPU 上就是"每个 warp 的 MLP 受寄存器数限制"，后面会算到。

---

## 六、把尺子架到 H100 上

有了 $L$ 和 $BW$，直接算需要的在飞字节数。H100 的参数来自 CUDA 官方 compute-capabilities 表和 Hopper Tuning Guide：132 个 SM、HBM3 3.35 TB/s、每 SM 64 个 warp 槽 / 2048 线程 / 64K 个 32 位寄存器 / 228 KB shared memory / 每 block 最多 1024 线程、每 SM 最多 32 个 block。延迟取公开测量值 **479 cycle**（pointer chase 测得，按 1.98 GHz 折成 241.9 ns）。

> 延迟这个数字要说清楚来源：它是**别人在 H100 上测的**，不是我在这台 Mac 上测的。同一类测量还有 363 ns 的报道，两者差 33%。这个差异会线性放大到下面所有 GPU 侧的结论上 —— 所以那些数字应该读成"量级正确"，而不是"精确到小数点"。真正精确的是本机 CPU 那两行，因为它们是本期实测的。

```python
# 本机 CPU 的实测常数：取自前面两个实验（多次运行的典型值）
L_cpu, BW_cpu = 83.7e-9, 86.0e9

# H100 SXM 的官方参数（文档值，非本机实测）
SMS = 132
BW_hbm = 3.35e12
BW_sm = BW_hbm / SMS          # 每个 SM 摊到的 HBM 带宽
L_gpu = 479 / 1.98e9          # 479 cycle @ 1.98 GHz

print("== 两台机器的「延迟 × 带宽」==")
print(f"{'':>10} {'延迟':>9} {'带宽':>11} {'在飞字节':>9} {'128B 请求':>10}")
for name, L, BW in [("本机 CPU", L_cpu, BW_cpu), ("H100 每 SM", L_gpu, BW_sm)]:
    need = L * BW
    print(f"{name:>10} {L*1e9:>7.1f}ns {BW/1e9:>9.1f}GB/s {need/1024:>7.2f}KB {need/128:>10.1f}")

print()
print(f"H100 全卡: {L_gpu*1e9:.1f}ns x {BW_hbm/1e12:.2f}TB/s = "
      f"{L_gpu*BW_hbm/1e6:.2f} MB = {L_gpu*BW_hbm/128:.0f} 条 128B 请求")

print()
print("== 每个 warp 能开几条在飞，决定需要多少 occupancy ==")
need = L_gpu * BW_sm
print(f"{'每warp在飞':>10} {'需要warp':>9} {'occupancy':>10} {'每线程寄存器':>12}")
for mm in (1, 2, 4, 8, 16):
    warps = need / (mm * 128)
    regs = min(65536 // (warps * 32), 255)
    print(f"{mm:>10} {warps:>9.1f} {warps/64*100:>9.1f}% {regs:>12}")
```

```text
== 两台机器的「延迟 × 带宽」==
                  延迟          带宽      在飞字节    128B 请求
    本机 CPU    83.7ns      86.0GB/s    7.03KB       56.2
 H100 每 SM   241.9ns      25.4GB/s    6.00KB       48.0

H100 全卡: 241.9ns x 3.35TB/s = 0.81 MB = 6331 条 128B 请求

== 每个 warp 能开几条在飞，决定需要多少 occupancy ==
   每warp在飞    需要warp  occupancy       每线程寄存器
         1      48.0      74.9%         42.0
         2      24.0      37.5%         85.0
         4      12.0      18.7%        170.0
         8       6.0       9.4%          255
        16       3.0       4.7%          255
```

这张表是本期的核心，值得逐行解释。

**第一，两台机器的在飞字节数几乎一样（7.03 KB vs 6.00 KB）。** 一个是 4 个性能核的 CPU，一个是 132 个 SM 的 GPU，带宽差 39 倍，延迟差 2.9 倍，但 $L \times BW$ 落在同一个量级。这不是巧合 —— 这个乘积就是"一个执行单元要喂饱自己的带宽，必须同时握住多少数据"，它是**架构无关的**。换到任何加速器上都能用：量出它的 $L$ 和 $BW$，乘一下，就知道它需要多大的并发度。

**第二，H100 每个 SM 需要 48 条 128 B 请求在飞。** 而一个 SM 恰好有 64 个 warp 槽。如果每个 warp 只能开 1 条在飞请求，需要 **48 warp = 74.9% occupancy** 才能喂满显存。这个数字比"20% 就够"的江湖说法高得多。

**第三，但这个 74.9% 的前提是「每 warp 只开 1 条」。** 一旦编译器帮你展开循环、一次发 4 条独立 load，需要的 warp 就掉到 12 个 —— **18.7% occupancy 就够喂满 HBM**。

这就解释了第 044 期那个看起来矛盾的现象：Hopper 上的 CUTLASS GEMM 只有 12.5% occupancy，性能却接近峰值。**不是 occupancy 不重要了，而是它的份额被 MLP 和 TMA 顶掉了。**

**第四，寄存器预算是这道题真正的约束。** 想开 8 条在飞，需要 8 个目的寄存器 + 若干地址寄存器；而表里显示，要维持 8 条在飞所需的 6 个 warp，每线程只有 255 个寄存器的额度（已经到顶）。反过来说：**每线程 32 个寄存器是"满 occupancy + 少 MLP"这条路，每线程 168 个寄存器是"低 occupancy + 多 MLP"这条路。两条路都能喂满带宽，但它们是互斥的。**

这解释了一个流传很广的误解 —— "occupancy 要在 50% 以上"。准确的说法是：

$$\text{occupancy} \times \text{每 warp 的 MLP} \gtrsim \frac{L \times BW}{128\ \text{B}}$$

左边是两个可以互换的因子，右边是硬件定的常数。**只盯左边一个因子没有意义。**

---

## 七、三把锁：occupancy 是怎么被算出来的

上面讲的是"需要多少"。现在讲"能给多少"。

一个 SM 上能同时驻留多少 block，是四个上限取最小：

1. 寄存器堆：65536 个 32 位寄存器
2. shared memory：228 KB
3. warp 槽：64 个
4. block 数：32 个

其中寄存器按 **warp** 分配，分配单元是 **256**。所以一个 warp 的实际占用是：

$$\text{regs}_{\text{warp}} = 256 \times \left\lceil \frac{32 r}{256} \right\rceil$$

这一个向上取整，就是悬崖的来源。下面把三把锁分别扫一遍（全部按官方参数计算）。

```python
SM = dict(warps=64, threads=2048, regs=65536, smem=228 * 1024, blocks=32, warp_size=32)
REG_UNIT, SMEM_UNIT = 256, 128


def active_blocks(regs_per_thread, threads_per_block, smem_per_block=0):
    """一个 SM 上最多能挂几个 block：四把锁取最小"""
    per_warp = regs_per_thread * SM["warp_size"]
    per_warp = (per_warp + REG_UNIT - 1) // REG_UNIT * REG_UNIT   # 向上取整到 256
    nwarps = threads_per_block // SM["warp_size"]
    smem_block = (smem_per_block + SMEM_UNIT - 1) // SMEM_UNIT * SMEM_UNIT
    by_reg = SM["regs"] // (per_warp * nwarps)
    by_smem = SM["smem"] // smem_block if smem_block else SM["blocks"]
    by_warp = SM["warps"] // nwarps
    return min(by_reg, by_smem, by_warp, SM["blocks"]), (by_reg, by_smem, by_warp)


print("== 锁 1：寄存器（block = 256 线程 = 8 warp）==")
print(f"{'regs/线程':>9} {'每warp实占':>10} {'每block实占':>11} {'可挂block':>9} {'活跃warp':>8} {'occupancy':>9}")
for r in (24, 32, 33, 40, 41, 48, 64, 96, 168):
    nb, _ = active_blocks(r, 256)
    pw = (r * 32 + 255) // 256 * 256
    print(f"{r:>9} {pw:>10} {pw*8:>11} {nb:>9} {nb*8:>8} {nb*8/64*100:>8.1f}%")

print()
print("== 锁 2：block 大小（regs/线程 = 40 固定）==")
print(f"{'block线程':>9} {'warps/block':>11} {'可挂block':>9} {'活跃warp':>8} {'occupancy':>9}")
for bs in (32, 64, 128, 256, 512, 1024):
    nb, d = active_blocks(40, bs)
    w = min(nb * (bs // 32), 64)
    print(f"{bs:>9} {bs//32:>11} {nb:>9} {w:>8} {w/64*100:>8.1f}%")

print()
print("== 锁 3：shared memory（block = 256 线程，寄存器设得很低）==")
print(f"{'smem/block':>10} {'可挂block':>9} {'活跃warp':>8} {'occupancy':>9} {'卡在哪':>8}")
for kb in (0, 16, 32, 48, 64, 100, 200):
    nb, d = active_blocks(24, 256, kb * 1024)
    w = min(nb * 8, 64)
    lock = ["寄存器", "smem", "warp槽"][d.index(min(d))]
    print(f"{kb:>8}KB {nb:>9} {w:>8} {w/64*100:>8.1f}% {lock:>8}")
```

```text
== 锁 1：寄存器（block = 256 线程 = 8 warp）==
  regs/线程    每warp实占    每block实占   可挂block   活跃warp occupancy
       24        768        6144         8       64    100.0%
       32       1024        8192         8       64    100.0%
       33       1280       10240         6       48     75.0%
       40       1280       10240         6       48     75.0%
       41       1536       12288         5       40     62.5%
       48       1536       12288         5       40     62.5%
       64       2048       16384         4       32     50.0%
       96       3072       24576         2       16     25.0%
      168       5376       43008         1        8     12.5%

== 锁 2：block 大小（regs/线程 = 40 固定）==
  block线程 warps/block   可挂block   活跃warp occupancy
       32           1        32       32     50.0%
       64           2        25       50     78.1%
      128           4        12       48     75.0%
      256           8         6       48     75.0%
      512          16         3       48     75.0%
     1024          32         1       32     50.0%

== 锁 3：shared memory（block = 256 线程，寄存器设得很低）==
smem/block   可挂block   活跃warp occupancy      卡在哪
       0KB         8       64    100.0%    warp槽
      16KB         8       64    100.0%    warp槽
      32KB         7       56     87.5%     smem
      48KB         4       32     50.0%     smem
      64KB         3       24     37.5%     smem
     100KB         2       16     25.0%     smem
     200KB         1        8     12.5%     smem
```

三张表各有一句要记的话。

**表 1 是悬崖。** 32 → 33 是一个台阶（100% → 75%），40 → 41 是下一个（75% → 62.5%）。原因是 $32 \times 33 = 1056$ 向上取整到 1280，而 $32 \times 32 = 1024$ 正好落在 256 的倍数上。所以优化寄存器用量时，**目标是踩在 256 的整数倍上**（32、40、48、64），而不是"尽量小"。

这里有个口径要分清楚：如果只看寄存器，33 regs/线程允许 $65536 / 1280 = 51.2$ 个 warp。但 warp 是按 block 分配的，6 个 8-warp 的 block 只能给 48 个 warp。**51 是上界，48 是实际**。看文档和别人文章里的 occupancy 数字时，先确认它用的是哪个口径。

**表 2 是两头堵。** block 太小（32 线程）会被"每 SM 最多 32 个 block"这条规矩卡住，只能到 50%；block 太大（1024 线程）又会被寄存器卡住，同样 50%。中间那段 64~512 才是甜的。注意 64 线程那行给出 **78.1%** —— 比 128/256 的 75% 还高，因为 25 个 2-warp block 凑出 50 个 warp。这种细节只有算过才知道，靠直觉猜不出来。

**表 3 是"要了 shared memory 就得交税"。** 32 KB 是第一个转折点（87.5%），48 KB 掉到 50%，100 KB 只剩 25%。这条斜线就是第 044 期讲的"`num_stages` 加深 → smem 变多 → occupancy 被挤掉"的定量版本：`BM=128, BK=64, stages=5` 要 160 KB，一个 SM 只能放一个 block，occupancy 上限 12.5%。**它不是设计失误，是拿 occupancy 换流水线深度。**

---

## 八、还有一层：wave 量化

前三把锁决定"一个 SM 能装多少"，还有一个更粗的粒度决定"整个 GPU 装满几轮"。

GPU 的调度器把 block 分给 SM，分完一批（一个 wave）等它们跑完再发下一批。如果最后一波只填了 5% 的 SM，那 95% 的卡在最后一波里是闲着的。

```python
SMS, PER_SM = 132, 1        # 典型 GEMM：每 SM 只放得下 1 个 block

print(f"== wave 量化（{SMS} 个 SM，每 SM {PER_SM} 个 block）==")
print(f"{'总block数':>9} {'wave 数':>8} {'最后一波':>12} {'利用率':>8}")
for c in (64, 128, 132, 264, 512, 1024, 1056, 2048, 4096):
    waves = -(-c // (SMS * PER_SM))
    last = c - (waves - 1) * SMS * PER_SM
    print(f"{c:>9} {waves:>8} {last:>5}/{SMS:<6} {c/(waves*SMS*PER_SM)*100:>7.1f}%")
```

```text
== wave 量化（132 个 SM，每 SM 1 个 block）==
  总block数   wave 数         最后一波      利用率
       64        1    64/132       48.5%
      128        1   128/132       97.0%
      132        1   132/132      100.0%
      264        2   132/132      100.0%
      512        4   116/132       97.0%
     1024        8   100/132       97.0%
     1056        8   132/132      100.0%
     2048       16    68/132       97.0%
     4096       32     4/132       97.0%
```

这张表要分两段看。

**大 grid 段（≥128 个 block）**：利用率都在 97% 以上，wave 量化不是问题。哪怕最后一波只填 4 个 block，摊到 32 个 wave 里也只损失 3%。

**小 grid 段**：64 个 block 只有 **48.5%** 的利用率 —— 一半的 SM 从头到尾没接到活。这正是 decode 阶段小 batch 的病根：`M=1` 的 GEMM 切出来的 block 数远小于 SM 数，卡不是算得慢，是**根本没铺开**。这也解释了为什么 continuous batching（第 028 期）和 PD 分离（第 030 期）那么重要 —— 它们本质上都在做同一件事：**把 batch 凑大，让 grid 超过 SM 数**。

---

## 九、在 PyTorch 里怎么用

上面都是机制。落地到 PyTorch，occupancy 这件事几乎从不由你亲手设置，而是三层结构里被某个人替你决定。认清这三层，比记住任何一个 API 都有用。

### 9.1 第一层：硬件参数怎么读

Inductor 算 occupancy 用的就是这几个字段：

```python
import torch

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("SM 数:", p.multi_processor_count)
    print("每 SM 寄存器:", p.regs_per_multiprocessor)
    print("每 SM 最大线程:", p.max_threads_per_multi_processor)
    print("warp 大小:", p.warp_size)
    shared = 65536 // p.max_threads_per_multi_processor
    print(f"满 occupancy 下每线程寄存器预算: {shared}")
else:
    # 本机无 CUDA，用官方 H100 参数代入同样的公式
    print("本机无 CUDA，按 H100 参数计算：")
    print("满 occupancy 下每线程寄存器预算:", 65536 // 2048)
```

```text
本机无 CUDA，按 H100 参数计算：
满 occupancy 下每线程寄存器预算: 32
```

`65536 / 2048 = 32` —— 这个数字就是"满 occupancy 只能给每线程 32 个寄存器"的出处，也是表 1 里那个悬崖为什么踩在 32 上。

### 9.2 第二层：Triton 的 num_warps

写 Triton kernel 时你填的 `num_warps` 决定了一个 block 有多少个 warp，`num_stages` 决定它吃多少 smem。两者合起来决定一个 SM 能挂几个 block —— 就是表 2、表 3 那道题。

PyTorch 自己的选择逻辑写在一个很朴素的函数里（`torch/_inductor/runtime/triton_heuristics.py:3774`），它是纯函数，在 CPU 上就能调用，下面是真实输出：

```python
from torch._inductor.runtime.triton_heuristics import _num_warps

print("默认（min=2, max=8）：")
for w in (1, 2, 3, 4, 8, 16):
    print(f"  想要 {w:>2} warp -> 实际 {_num_warps(w)}")

print("寄存器密集型（持久化归约）：")
for w in (2, 4, 8, 16):
    print(f"  想要 {w:>2} warp -> 实际 {_num_warps(w, register_intensive=True)}")

print("wave64 的 AMD CDNA：")
for w in (4, 8, 16):
    print(f"  想要 {w:>2} warp -> 实际 {_num_warps(w, warp_size=64)}")
```

```text
默认（min=2, max=8）：
  想要  1 warp -> 实际 2
  想要  2 warp -> 实际 2
  想要  3 warp -> 实际 4
  想要  4 warp -> 实际 4
  想要  8 warp -> 实际 8
  想要 16 warp -> 实际 8
寄存器密集型（持久化归约）：
  想要  2 warp -> 实际 2
  想要  4 warp -> 实际 4
  想要  8 warp -> 实际 4
  想要 16 warp -> 实际 4
wave64 的 AMD CDNA：
  想要  4 warp -> 实际 4
  想要  8 warp -> 实际 4
  想要 16 warp -> 实际 4
```

三个信息：

1. **上限是 8，下限是 2，中间取 2 的幂。** 所以 Triton 世界里 occupancy 的调节范围天然被限制在每 block 2~8 warp（64~256 线程）。一个 SM 要挂满 64 warp，就得挂 8~32 个 block。
2. **`register_intensive=True` 直接把上限减半到 4。** 这是源码里唯一一处因为"寄存器压力"而主动放弃 occupancy 的地方，注释原话是 `# persistent reduction is register intensive`。
3. **wave64 的 AMD 把 warp 数砍半**，因为一个 warp 有 64 个 lane，线程总数才对得上。这是"warp 大小是可变量"的活证据 —— 第 008 期写 Triton kernel 时，`num_warps` 的物理含义会随硬件变。

### 9.3 第三层：Inductor 会为了 occupancy 主动降寄存器用量

这是最有意思的一处。`torch/_inductor/runtime/triton_heuristics.py` 里有个方法 `_iter_rblock_scale_candidates`，注释把整道题写了出来：

    每 SM 有 65536 个 32-bit 寄存器。要让 2048 个线程同时驻留，
    每线程不能用超过 65536 / 2048 = 32 个寄存器。
    在 occupancy 重要、而每线程寄存器又超标的场景下，
    就去把 R0_BLOCK 减半，从而把寄存器用量降下来。
    （注释里给了一个真实案例：latency 从 7.795ms 降到 4.883ms）

它的判定条件写得很清楚，分三步：

1. 读编译产物里的 `compiled_binary.n_regs`（每线程实际寄存器数）。拿不到就跳过。
2. 如果 `n_regs <= regs_per_multiprocessor // max_threads_per_multi_processor`（也就是 ≤32），跳过 —— **没超标，不用管**。
3. 算出 `max_blocks_per_sm = regs_per_multiprocessor // (nreg * warp_size * num_warps)`，如果总 block 数没超过 `max_blocks_per_sm × SM 数`，也跳过 —— **grid 本来就装得下，occupancy 低不影响**。

三条都过了，才去把最大的 `Rn_BLOCK` 减半，重新编译，重测。这个循环会一直下去。

这段代码值得抄下来，因为它是本期整套理论在生产环境里的完整闭环：**测量 → 算锁 → 判断 occupancy 是否真的是瓶颈 → 用"降寄存器"这把钥匙开锁 → 重测**。注意第 2、3 步那两个 `continue` —— 它明确承认了"occupancy 低不一定是问题"。

### 9.4 手写 kernel 时你手上的旋钮

| 旋钮 | 作用在哪 | 代价 |
|---|---|---|
| `num_warps`（Triton）/ block 大小（CUDA） | 每 block 的 warp 数 | 太大撞寄存器，太小撞 32-block 上限（表 2） |
| `num_stages`（Triton）/ 多缓冲级数 | 吃 smem | 表 3 那条斜线，直接挤掉 occupancy |
| `maxnreg`（Triton）/ `__launch_bounds__`（CUDA） | 硬性限制每线程寄存器 | 编译器被迫 spill，溢出到 local memory（走 L2/HBM） |
| 循环展开 / `tl.load` 的向量宽度 | 每 warp 的 MLP | 每开一条在飞就要几个目的寄存器 |

第四个是本期最重要的一个：**它不在 occupancy 这一侧，而在 MLP 这一侧，效果和 occupancy 等价**（第五节那三条等值线就是证据）。当你在低 occupancy 下看到带宽跑不满，第一反应不该是"加 warp"，而该是"让每个 warp 多开几条在飞"。

---

## 十、围绕这个领域展开

occupancy 不是一个孤立的知识点，它是好几条线的交汇处。

**和 044（异步拷贝）的关系：两条完全不同的填坑路线。** 044 讲的是"用一个不占寄存器的通道（`cp.async` / TMA）把请求送出去"，本期讲的是"用更多 warp 把请求送出去"。前者不消耗寄存器但需要 smem 和多级缓冲，后者消耗寄存器但不需要额外硬件。到了 Hopper，第一条路线赢了大半 —— 所以 CUTLASS 的 GEMM 敢只要 12.5% occupancy。

**和 043（访存层次）的关系：一个算"买多少"，一个算"等多久"。** 043 的算术强度决定你落在 roofline 的哪一侧；本期决定你在那一侧能不能贴住那条线。两个问题正交：算术强度低的 kernel 一定带宽受限，但带宽受限不等于能跑满带宽 —— 那要看并发够不够。

**和 042（CPU 并行）的关系：同一道题的两个答案。** 042 里 CPU 的答案是"绝对门槛 32768 + 线程池"，本期的答案是"$L \times BW$ 决定的在飞字节数"。CPU 上线程是重的（一次唤醒十几 µs），所以它靠粗粒度切块；GPU 上 warp 是免费的（切换零开销），所以它靠海量 warp。**同一个 Little 定律，两种成本结构，导出两种设计。**

**和 014（Profiling）的关系：指标的名字要先审。** ncu 报的 "Achieved Occupancy" 是"活跃 warp 数 / 最大 warp 数"，它只说"挂了多少"，不说"够不够"。同一个 12.5% 在一个 kernel 里是灾难，在另一个里是最优。判断方法就是本期的表 B：算出这个 kernel 的需要量，再比。

**和 037（零假设）的关系：occupancy 也有它的"零假设"。** 当有人告诉你"occupancy 提到 75% 之后性能涨了 10%"，先问：**这 10% 是 occupancy 带来的，还是改 block 大小时顺手改了别的东西？** 表 2 里改 block 大小会同时改变 warp 数、每 block 的 work 分配、以及 wave 量化。想干净地隔离，只能固定 block 大小去改 `maxnreg` —— 但那条路会引入 spill，所以还得同时盯住 `n_spills` 有没有变。**"只动一个变量"在 GPU 上很贵，这就是为什么 037 那套零假设纪律在这里格外重要。**

**和 001（CUDA Graph）、003（torch.compile）的关系：不同层级的"减少唤醒"。** 001 减少 kernel 之间的 launch 间隔，003 减少 Python 侧开销，本期减少 warp 之间的等待。三者都在跟"单元之间的空隙"较劲，只是粒度从毫秒降到微秒再降到纳秒。

**和 040（算子融合）的关系：融合会改变 occupancy 的账。** 把 5 个逐元素算子融成一个 kernel，寄存器用量会涨（5 个中间结果同时在寄存器里），可能就把 occupancy 从 100% 挤到 75%。**融合的收益在"少搬字节"，代价可能在"少挂 warp"**，这两笔账要用 040 的流量账和本期的表 B 一起算。

**和 032/033（KV Cache 量化、MLA）的关系：decode 阶段的 kernel 全是小 grid。** 第八节那张 wave 表说明，batch 小的时候 SM 根本铺不满，此时 occupancy 的讨论都失去了意义 —— 该做的是把 batch 凑大（028、030），而不是调 warp 数。

---

## 十一、什么时候该用 / 不该用

**该盯 occupancy 的信号：**

- 带宽受限的 kernel（逐元素、归约、memcpy 类），且实测带宽 < 峰值的 70%；
- ncu 里 `Achieved Occupancy` 明显低于 `Theoretical Occupancy`（说明有别的因素在拖，比如尾部、同步）；
- 寄存器用量或 smem 用量刚好卡在台阶上（33、41、49 这种数字），把编译器往回调一格就能上一个台阶；
- grid 足够大（远超 SM 数 × 每 SM block 数），wave 量化不是瓶颈。

**不该盯 occupancy 的信号：**

- **kernel 是计算受限的**。算力已经跑满，加 warp 只会多加竞争。第 042 期实测过 CPU 上的同一现象：`copy` 加线程只有 1.04~1.11×，`exp` 有 2.96~3.04× —— 位置决定收益。
- **grid 太小**。64 个 block 只填了 48.5% 的 SM，此时 occupancy 再高也是在给不到一半的卡提要求。
- **smem 本来就该吃满**。Hopper GEMM 的 160 KB 是故意的，用 occupancy 换流水线深度是正收益。
- **每 warp 的 MLP 已经很高**。第九节 `_iter_rblock_scale_candidates` 里那两个 `continue` 就是 PyTorch 自己在说"这里不用管"。

一句话的判据：**先算需要多少在飞字节，再看当前配置能给多少，两者差得远才动手。**

---

## 十二、常见坑

**坑 1：把 `Achieved Occupancy` 当成 KPI。**
它是"挂了多少"，不是"够不够"。同一个 12.5% 在 Hopper GEMM 上是精心设计的结果，在一个逐元素 kernel 上就是灾难。要判断，必须算出这个 kernel 需要多少（表 B），而不是跟别人的数字比大小。

**坑 2：以为寄存器用量"越小越好"，忽略了台阶。**
33 比 32 贵 25%，41 比 40 贵 20%。原因是 $32r$ 要向上取整到 256 的倍数，而 32 和 40 恰好踩在整数倍上。优化时应该把目标定在 32 / 40 / 48 / 64 这几个数上，而不是"能省一个是一个"。

**坑 3：加 `__launch_bounds__` / `maxnreg` 压寄存器，以为白赚。**
压下去的寄存器不会消失，它会变成 **spill** —— 溢出到 local memory，实际走 L2 甚至 HBM。你在 occupancy 上赚了 25%，可能在访存上赔了 50%。判断方法是看 `n_spills`，或者直接对比压之前和之后的耗时。**这一步必须实测，不能推**。

**坑 4：用顺序访问测并发度。**
本机实测：1 线程顺序读 77.9 GB/s，4 线程 81.4 GB/s —— 加三个线程只涨 4.5%。如果拿这个结论去说"CPU 的并发不重要"，就完全错了：换成随机访问（关掉预取器的自动 MLP）后，1 条链 1.5 GB/s、128 条链 85.4 GB/s，**差 57 倍**。测并发之前先确认预取器有没有替你干活。

**坑 5：扫并发度时让工作集跟着变。**
如果"加并发"的实现方式是"把数组切得更碎"，那每段的尺寸就在变。段小到滑进 L2 时，你会看到一个漂亮但假的拐点。本期的做法是总工作集固定 512 MB，只改链数。同理，**扫并发度时必须固定每段 ≥ 32 MB**，或者让总工作集远超最后一级缓存。

---

## 十三、一句话总结

**occupancy 是候机厅的上座率，真正决定吞吐的是同一时刻有多少人在等行李；这个数等于延迟乘带宽除以每次请求的字节数，硬件的三把锁（寄存器、smem、block 上限）决定你能给到多少，而"每 warp 多开几条在飞"和"多挂几个 warp"是同一枚硬币的两面。**

---

## 今日练习

<details><summary>今日练习</summary>

**题目**：某个 Triton kernel 编译出来每线程用 **64 个寄存器**，block 大小 **128 线程**，不吃 shared memory，grid 有 **4096 个 block**，跑在 H100 上（132 SM）。

1. 它的 theoretical occupancy 是多少？
2. 如果要回到 75%，每线程寄存器最多能用到多少？
3. 这个 kernel 值不值得为了 occupancy 去压寄存器？

**参考答案**（在下面这段代码里算，避免手推出错）：

```python
REG_UNIT = 256


def occ(regs_per_thread, threads_per_block, smem_per_block=0):
    per_warp = (regs_per_thread * 32 + REG_UNIT - 1) // REG_UNIT * REG_UNIT
    nwarps = threads_per_block // 32
    by_reg = 65536 // (per_warp * nwarps)
    by_smem = 228 * 1024 // smem_per_block if smem_per_block else 32
    by_warp = 64 // nwarps
    blocks = min(by_reg, by_smem, by_warp, 32)
    return blocks, min(blocks * nwarps, 64)


print("1) 当前配置（64 regs / 128 线程）：")
nb, w = occ(64, 128)
print(f"   可挂 {nb} 个 block，活跃 {w} warp，occupancy = {w/64*100:.1f}%")

print("2) 要回到 75%（48 warp）：")
for r in range(64, 20, -1):
    nb, w = occ(r, 128)
    if w >= 48:
        print(f"   每线程 <= {r} 个寄存器即可（此时 {nb} 个 block，{w} warp）")
        break

print("3) grid = 4096 个 block，SM 数 132：")
nb, w = occ(64, 128)
cap = nb * 132
print(f"   全部 SM 一轮能吞 {cap} 个 block，需要 {-(-4096//cap)} 个 wave")
print(f"   wave 利用率 = {4096 / (-(-4096//cap) * cap) * 100:.1f}%")
```

```text
1) 当前配置（64 regs / 128 线程）：
   可挂 8 个 block，活跃 32 warp，occupancy = 50.0%
2) 要回到 75%（48 warp）：
   每线程 <= 40 个寄存器即可（此时 12 个 block，48 warp）
3) grid = 4096 个 block，SM 数 132：
   全部 SM 一轮能吞 1056 个 block，需要 4 个 wave
   wave 利用率 = 97.0%
```

**结论**：50% occupancy，要回 75% 得把寄存器从 64 压到 40（压掉 37.5%，几乎肯定要 spill）。而这个 kernel 的 wave 利用率已经是 97%，grid 铺得很满 —— 也就是说，**它在时间维度上没有空隙可补**，压寄存器换 occupancy 大概率是亏的。第三问才是该先问的问题。

</details>
