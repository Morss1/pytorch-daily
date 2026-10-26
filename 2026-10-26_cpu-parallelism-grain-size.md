# PyTorch 每日一课 · 第 042 期

## CPU 上的并行：GRAIN_SIZE、`at::parallel_for` 与 omp 线程池

**日期** 2026-10-26　**难度** ⭐⭐⭐⭐　**前置** 第 041 期（知道融合体内部有「并行 P」这个自由度）、听说过 OpenMP 的 `#pragma omp parallel`　**预计阅读** 约 25 分钟

**关联** 第 041 期 · 041 讲了融合体「怎么切」，本期往下一层问「谁来切、切完谁去跑」；第 001 期（CUDA Graph 省的是 launch 开销，本期量的是另一头的唤醒开销）；第 003 期（torch.compile 生成的 kernel 是另一套并行实现）；第 040 期（融合改变了并行门槛前后的收益比）

---

### 0. 先把结论摆出来

上一期把 24 个逐元素算子融成 1 个 kernel，靠的是「一次搬完」。这一期问一个更基础的问题：**这 1 个 kernel 内部，是谁把 16M 个元素分给 4 个核的？**

1. **PyTorch 在 CPU 上有三套完全不同的并行实现**，而不是一套。eager 走 `at::parallel_for`（线程池 + `GRAIN_SIZE` 门槛），`torch.compile` 生成的 C++ kernel 走**裸 OpenMP pragma**（门槛换成「每线程 512 元素」，且 `num_threads` 编译期写死），Triton 那边根本没有「线程池」这个概念（只有 block 网格）。
2. **`at::parallel_for` 的门槛是 32768 个元素，与线程数无关**。实测 `copy_` 在 N ≤ 32768 时 4 线程和 1 线程**一样快**（比值 ≈ 1.00），N=32769 时 4 线程**慢近 5 倍**（比值 0.21）。门槛只决定「要不要并行」，不决定「切几块」。
3. **`grain_size` 同时干两件事**，这是最容易读错的地方：当**门槛**用一次（`numiter > grain_size`），再当**切块上限**用一次（`任务数 = min(线程数, ⌈N/grain⌉)`）。实测固定 N=1,000,000，grain 从 500000 改到 200001，任务数依次是 2 / 3 / 4，公式逐项对上。
4. **一次 omp 区域唤醒的净成本约十几 µs**，而刚过门槛的那 32769 个 fp32 元素本身搬运只需约 2 µs。**这就是「刚过门槛就并行」是净亏的原因** —— 不是并行没用，是这一步的活还不够付门票。
5. **门槛和盈亏平衡点差一个数量级以上**。实测交叉点落在 262144~524288，而门槛是 32768。**「过了门槛」只意味着进了并行分支，和「并行划算」是两件事。**
6. **加线程的收益由 roofline 位置决定，不由线程数决定**。同一台机器、同样 16M 元素：`copy_` 4 线程约 1.05×、`add` 约 1.10×（撞带宽墙）；`exp` **约 3.0×**（计算受限，线程几乎线性加）。

---

### 1. 这个领域解决什么问题

一个 16M 元素的 `x * 2 + 1`，在 4 核 CPU 上要 4 个核一起干。但「一起干」这句话要落地，得回答四个问题：

| 问题 | 谁回答 | 答错的后果 |
|---|---|---|
| 什么时候值得开并行？ | 门槛（`GRAIN_SIZE`） | 小数组上开并行，唤醒成本吃掉全部收益 |
| 切成几块？ | `grain_size` / `min_chunk_size` | 块太少 → 有核闲着；块太多 → 调度开销 |
| 谁来唤醒那几个核？ | omp 线程池 / 线程池实现 | 唤醒一次 ~15 µs，比搬 32K 元素还贵 |
| 每个核内部再怎么加速？ | 向量化（041 期） | 少了 4× 的宽度 |

这四个问题在 PyTorch 里的答案是**不统一**的：

- **eager**：`at::parallel_for(0, N, grain, f)`，门槛 32768、切块按线程数均分。
- **`torch.compile`（CPU 后端）**：生成的 C++ kernel 里直接写 `#pragma omp parallel num_threads(4)` + `#pragma omp for`，**没有 32768 那种绝对门槛**，只有「每线程 ≥ 512 元素」这一条相对门槛。
- **Triton（CUDA）**：没有线程池，`triton_config` 给的是 `XBLOCK` 和 `num_warps`，线程是 grid 的一部分。

这就是为什么「我把 `torch.set_num_threads` 调到 8 了为什么没变快」这个问题，答案往往不在你调的那个旋钮上 —— 而在**你正在跑的到底是哪一套实现**。

---

### 2. 核心思想：门槛、切块、唤醒，三件不同的事

#### 2.1 `at::parallel_for` 的三层结构

```
parallel_for(begin, end, grain_size, f)
   │
   ├─ ① 门槛判断（ATen/Parallel-inl.h）
   │     use_parallel = (numiter > grain_size) && (numiter > 1)
   │                    && !in_parallel_region() && (get_num_threads() > 1)
   │     不满足 → 当前线程直接把 [begin, end) 全干完，零开销
   │
   ├─ ② 切块（ATen/ParallelOpenMP.h::invoke_parallel）
   │     num_threads = min(omp_get_num_threads(), ⌈N / grain_size⌉)
   │     chunk_size  = ⌈N / num_threads⌉
   │     tid 拿 [tid*chunk, min(N, (tid+1)*chunk))
   │
   └─ ③ 唤醒 omp 区域（这一步的固定成本 ~15 µs）
```

**关键点：`grain_size` 是同一个数字，但在 ① 里它是「最少多少元素才值得开始」，在 ② 里它变成了「最多切成几块」。**

源码原文（`aten/src/ATen/Parallel-inl.h:22-25`，上游 main 分支）：

```cpp
const auto numiter = end - begin;
const bool use_parallel =
    (numiter > grain_size && numiter > 1 && !at::in_parallel_region() &&
     at::get_num_threads() > 1);
```

源码原文（`aten/src/ATen/ParallelOpenMP.h:30-37`）：

```cpp
int64_t num_threads = omp_get_num_threads();
if (grain_size > 0) {
  num_threads = std::min(num_threads, divup((end - begin), grain_size));
}
int64_t tid = omp_get_thread_num();
int64_t chunk_size = divup((end - begin), num_threads);
int64_t begin_tid = begin + tid * chunk_size;
```

注意 `divup(N, grain_size)` 是 **⌈N/grain⌉**，不是 ⌈N/(grain×threads)⌉。所以 grain_size 的语义更接近「**一次派活的粒度上限**」，不是「每个线程至少拿多少」。

#### 2.2 那个 32768 是哪来的

`GRAIN_SIZE` 在源码里是一个宏，定义在 `aten/src/ATen/ParallelCommon.cpp` 的匿名 namespace：

```cpp
constexpr int64_t GRAIN_SIZE = 32768;
```

（这段是**从上游源码读到的**；本机 pip 包只带 Python，C++ 侧走 GitHub raw 抓的 main 分支。）

为什么是 32768？和 041 期那个 `min_chunk_size = 512` 是同一个思路的两次应用：

- **512** 是「一个线程至少要有的活」，来自向量化的收益门槛（4 lane × 128 次迭代）。
- **32768** 是「整个算子至少要有的活」，等于 64 个线程 × 512，或者 4 个线程 × 8192。

两者不是同一个约束，但都指向同一件事：**每单位并行度必须有足够的元素数，否则调度成本会盖过计算收益**。

#### 2.3 三个后端的并行门槛都不一样

| 执行器 | 门槛条件 | 门槛值（默认） | 切块依据 |
|---|---|---|---|
| eager ATen | `N > GRAIN_SIZE` | **32768 元素**（绝对值） | 按线程数均分 |
| torch.compile CPU | `seq // threads ≥ min_chunk_size` | **每线程 512 元素**（相对值） | `decide_parallel_depth` 决定切几层 |
| Triton CUDA | 无显式门槛 | — | `XBLOCK × grid`，由 autotune 选 |

`torch.compile` 那条来自 `torch/_inductor/codegen/cpp.py:2767`（本机 pip 包里的 Python 源码，可直接读）：

```python
def decide_parallel_depth(self, max_parallel_depth, threads):
    ...
    for expr in ranges:
        hint = V.graph.sizevars.optimization_hint(expr, fallback=8192)
        if par >= 2 * threads or par == threads:
            break
        if seq // threads < config.cpp.min_chunk_size:
            # not enough work
            break
        depth += 1
        par *= hint
        seq /= hint
```

注意第一条 break：`if par >= 2 * threads or par == threads`。`par` 从 1 开始，所以 **T=1 时第一次循环就 break，depth=0，完全不生成 omp** —— 这就是 041 期「T=1 产物里一条 omp pragma 都没有」的源码依据。

而 T=4 时 `seq // 4 < 512` 才 break，即 **N < 2048 才不并行**。和 eager 的 32768 差了 16 倍。

---

### 3. 在 PyTorch 中怎么用：把三套实现摆在一起看

#### 3.1 先确认你在这台机器上的「线程池」是什么

```python
import os

import torch

# 想看到「默认」线程数，就必须先确认没被环境变量改过。
# 注意：OMP_NUM_THREADS 是在 import torch 时就固化的，设了再删也没用，得在启动前就没设。
print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', '(未设置)')}")
print(f"torch {torch.__version__}")
print(f"at::get_num_threads()      = {torch.get_num_threads()}")
print(f"at::get_num_interop_threads() = {torch.get_num_interop_threads()}")
print()
print(torch.__config__.parallel_info())
```

```text
OMP_NUM_THREADS = (未设置)
torch 2.14.0
at::get_num_threads()      = 4
at::get_num_interop_threads() = 10

ATen/Parallel:
	at::get_num_threads() : 4
	at::get_num_interop_threads() : 10
OpenMP 202011
	omp_get_max_threads() : 4
MKLDNN not found
std::thread::hardware_concurrency() : 10
Environment variables:
	OMP_NUM_THREADS : [not set]
ATen parallel backend: OpenMP
```

这里有个细节值得停一下：`hardware_concurrency() = 10`，但 `get_num_threads() = 4`。这台是 Apple Silicon，`sysctl -n hw.perflevel0.physicalcpu` 也是 4 —— **PyTorch 在 macOS/aarch64 上默认只让你用性能核**，源码 `ParallelCommon.cpp:116-127` 里显式读了 `hw.perflevel0.physicalcpu`：

```cpp
#if defined(__aarch64__) && defined(__APPLE__)
    // On Apple Silicon there are efficient and performance core
    // Restrict parallel algorithms to performance cores by default
    sysctlbyname("hw.perflevel0.physicalcpu", ...);
```

所以「为什么我 10 核机器只有 4 个线程」不是 bug，是设计。

#### 3.2 亲手看见门槛：固定 N，二分找开关

用 `copy_` 做探针最干净 —— 它是纯搬运，没有任何计算混进来。

```python
import time
import torch

src = torch.randn(16_000_000)
dst = torch.empty(16_000_000)


def solo(nt, N, warm=12, reps=80):
    """单一配置：min-of-reps。每个配置独立预热，避免线程池状态互相污染"""
    d, s = dst[:N], src[:N]
    torch.set_num_threads(nt)
    for _ in range(warm):
        d.copy_(s)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        d.copy_(s)
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


print(f"{'N':>9}{'1 线程 µs':>12}{'4 线程 µs':>12}{'比值':>8}")
for N in (8192, 16384, 32767, 32768, 32769, 65536, 131072, 262144, 1048576):
    t1, t4 = solo(1, N), solo(4, N)
    print(f"{N:>9}{t1:>12.2f}{t4:>12.2f}{t1 / t4:>8.2f}")
```

```text
        N     1 线程 µs     4 线程 µs      比值
     8192        0.79        0.75    1.06
    16384        1.17        1.12    1.04
    32767        2.58        2.58    1.00
    32768        2.62        2.58    1.02
    32769        2.62       12.38    0.21
    65536        5.00       13.21    0.38
   131072       10.75       13.79    0.78
   262144       20.17       23.21    0.87
  1048576       93.88       53.46    1.76
```

读法：

- **N ≤ 32768：比值恒为 1.00**（四格实测 1.06/1.04/1.00/1.02，都在噪声范围内），因为两个配置下 `use_parallel` 都是 false，跑的是同一段代码。
- **N = 32769：比值 0.21**，4 线程反而慢近 5 倍。多出来的那 9 µs 全是唤醒成本。
- **N = 262144：比值 0.87** —— 仍然亏。要到 1048576 才明确转正（1.76）。

注意 32769 这个点特别有意思：它只比 32768 多 **1 个元素**，耗时从 2.62 µs 跳到 12.38 µs。这是「门槛」最纯粹的样子 —— 不是缓慢退化，是断崖。

#### 3.3 直接用 C++ 侧看分片：grain 只决定切几块

官方公开的 `at::parallel_for` 头可以直接在 `torch.utils.cpp_extension` 里用。写十几行，让每个分片自己报告边界：

```python
import torch
from torch.utils.cpp_extension import load_inline

src = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <atomic>
#include <vector>

// 返回每个分片的 [begin, end, tid]
std::vector<std::vector<int64_t>> slices(int64_t n, int64_t grain) {
  // 先按最大分片数预分配，再按下标写 —— 多个线程并发 push_back 同一个 vector
  // 是数据竞争，会踩坏内部指针导致段错误（我第一次就是这么写崩的）
  const int64_t cap = (n + grain - 1) / grain;
  std::vector<std::vector<int64_t>> out(cap);
  std::atomic<int64_t> next{0};
  at::parallel_for(0, n, grain, [&](int64_t begin, int64_t end) {
    // 用原子计数器领一个独占下标，而不是拿 begin/grain 算 —— 末尾那个任务
    // 的 begin 不一定是 grain 的整数倍
    const int64_t idx = next.fetch_add(1);
    out[idx] = {begin, end, at::get_thread_num()};
  });
  out.resize(next.load());          // 实际任务数可能小于 cap
  std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) { return a[0] < b[0]; });
  return out;
}
"""

m = load_inline(
    name="pf_probe",
    cpp_sources=src,
    functions=["slices"],
    # Mac 上 clang 默认没有 -fopenmp，不显式打开的话 at::parallel_for 会静默退化成串行
    extra_cflags=["-O2", "-Xpreprocessor", "-fopenmp"],
    extra_ldflags=["-lomp"],
    verbose=False,
)

torch.set_num_threads(4)
N = 1_000_000
print(f"N = {N:,}，线程池 = 4")
print(f"{'grain':>9}{'任务数':>7}{'任务大小':>18}{'tid':>12}")
for g in (500_000, 400_000, 333_334, 327_768, 250_000, 200_001):
    r = sorted(m.slices(N, g))         # 按 begin 排序，输出与调度顺序无关
    sizes = sorted({e - b for b, e, _ in r})
    tids = sorted({t for _, _, t in r})
    want = min(4, -(-N // g))          # min(线程数, ⌈N/grain⌉)
    chunk = -(-N // want)              # ⌈N/任务数⌉
    print(f"{g:>9}{len(r):>7}{str(sizes):>18}{str(tids):>12}"
          f"  ⌈N/g⌉={want} → 每片 {chunk}")
```

```text
N = 1,000,000，线程池 = 4
    grain    任务数              任务大小         tid
   500000      2          [500000]      [0, 1]  ⌈N/g⌉=2 → 每片 500000
   400000      3  [333332, 333334]   [0, 1, 2]  ⌈N/g⌉=3 → 每片 333334
   333334      3  [333332, 333334]   [0, 1, 2]  ⌈N/g⌉=3 → 每片 333334
   327768      4          [250000][0, 1, 2, 3]  ⌈N/g⌉=4 → 每片 250000
   250000      4          [250000][0, 1, 2, 3]  ⌈N/g⌉=4 → 每片 250000
   200001      4          [250000][0, 1, 2, 3]  ⌈N/g⌉=4 → 每片 250000
```

三件事被钉死了：

1. **任务数 = `min(4, ⌈N/grain⌉)`**。grain 从 500000 降到 200001，任务数 2→3→4，然后**卡在 4 不动了**，因为线程池只有 4。grain 继续变小只是"允许"切得更细，没有第 5 个线程来接。
2. **grain 不影响每片的大小**，只影响能被切到多细。`grain=250000` 和 `grain=200001` 给出的分片完全一样。
3. **最后一片可能比 chunk 短**。N=1,000,000、3 个任务时，chunk = ⌈1000000/3⌉ = 333334，前两片各 333334，最后一片只剩 333332。这是 `min(end, chunk_size + begin_tid)` 干的事。

> **踩坑记录**：这段代码在 Mac 上第一次跑出来是「**任务数恒为 1**」，我一度以为是 `parallel_for` 根本不并行。真实原因是 clang 默认没有 `-fopenmp`，`ATen/Parallel.h` 里 `#if AT_PARALLEL_OPENMP` 那段头根本没被编译进来，于是 `parallel_for` 走进 `#else` 分支直接串行执行 —— **不报错、不警告**。加上 `-Xpreprocessor -fopenmp` 与 `-lomp` 之后才露出真面目。

#### 3.4 量一下门票钱：一次 omp 唤醒到底多贵

上面的分片探针能告诉我们「切了几块」，但说不出「唤醒要多少钱」。用一个串行/并行只差 1 个元素的对照来量：

```python
import statistics
import time

import torch
from torch.utils.cpp_extension import load_inline

src = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>

int64_t parallel_for_total(int64_t n, int64_t grain) {
  int64_t total = 0;
  at::parallel_for(0, n, grain, [&](int64_t b, int64_t e) { total += (e - b); });
  return total;
}
"""

m = load_inline(name="omp_cost", cpp_sources=src,
                functions=["parallel_for_total"],
                extra_cflags=["-O2", "-Xpreprocessor", "-fopenmp"],
                extra_ldflags=["-lomp"], verbose=False)

torch.set_num_threads(4)


def best_of(fn, reps=300, rounds=9):
    """返回 (min, 中位)。min 更抗抖动，中位更抗偶发长尾，两个都看"""
    fn()
    mins, meds = [], []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        mins.append((time.perf_counter() - t0) / reps * 1e6)
        meds.append((time.perf_counter() - t0) / reps * 1e6)
    return min(mins), statistics.median(meds)


# 32768 走串行分支，32769 走并行分支，只差 1 个元素
a1, m1 = best_of(lambda: m.parallel_for_total(32768, 32768))
a2, m2 = best_of(lambda: m.parallel_for_total(32769, 32768))
print(f"N=32768（串行分支） min {a1:6.2f} µs   中位 {m1:6.2f} µs")
print(f"N=32769（并行分支） min {a2:6.2f} µs   中位 {m2:6.2f} µs")
print(f"→ 一次 omp 区域唤醒净成本 ≈ {m2 - m1:.2f} µs（按中位）")
print(f"→ 这 32769 个 fp32 元素搬运需 ≈ {32769 * 8 / 130e3:.2f} µs（按 130 GB/s）")
```

```text
N=32768（串行分支） min   0.09 µs   中位   0.09 µs
N=32769（并行分支） min  15.88 µs   中位  17.34 µs
→ 一次 omp 区域唤醒净成本 ≈ 17.25 µs（按中位）
→ 这 32769 个 fp32 元素搬运需 ≈ 2.02 µs（按 130 GB/s）
```

**唤醒一次 omp 区域约 17 µs（上表这一次），而它要干的活只有 2 µs。门票比戏贵 8 倍多。**

这个数单次跑会飘：多个独立进程各跑一遍，净成本落在 **14~19 µs**，上面填的这次是 17.25 µs。所以量级要写成「十几 µs」，不要精确到小数点。注意串行分支那 0.09 µs 不是「搬运 32768 个元素的时间」，而是**整个 C++ 函数被优化成什么都不做**之后的空调用开销 —— 这正好说明这个块测的是「唤醒」，不是「搬运」。这也是为什么它必须和 32769 那一行相减：绝对值都不可信，只有差有意义。

结论是「刚过门槛就并行」在物理上就是亏的 —— 门槛设成 32768 只保证「进了并行分支不至于错得离谱」，不保证「一定变快」。

真实的交叉点在更后面。把 N 扫大（同样是 min-of-reps）：

```python
import time
import torch

SRC = torch.randn(64_000_000)
DST = torch.empty(64_000_000)


def solo(nt, N, warm=12, reps=80):
    d, s = DST[:N], SRC[:N]
    torch.set_num_threads(nt)
    for _ in range(warm):
        d.copy_(s)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        d.copy_(s)
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


print(f"{'N':>10}{'1 线程 µs':>12}{'4 线程 µs':>12}{'比值':>8}{'带宽 GB/s':>12}")
for N in (131072, 262144, 524288, 1048576, 4194304, 16777216):
    t1, t4 = solo(1, N), solo(4, N)
    bw = 2 * N * 4 / t4 / 1e3 if t4 else 0
    print(f"{N:>10}{t1:>12.2f}{t4:>12.2f}{t1 / t4:>8.2f}{bw:>12.1f}")
```

```text
         N     1 线程 µs     4 线程 µs      比值     带宽 GB/s
    131072       12.79       24.46    0.52        42.9
    262144       21.42       24.83    0.86        84.5
    524288       40.38       33.87    1.19       123.8
   1048576       81.46       53.75    1.52       156.1
   4194304      398.46      179.58    2.22       186.8
  16777216     1652.75     1084.25    1.52       123.8
```

**门槛 32768，但真正的交叉点在 262144 ~ 524288 之间 —— 门槛和盈亏平衡点差了 8~16 倍。**

交叉点这一格必须交叉验证，因为本机只有 4 个性能核，而 WorkBuddy 自己常占 2~3 个核（`ps aux` 实测 Electron 主进程 + 渲染进程合计 260% CPU），负载一动比值就会飘。跑 5 个独立进程，每轮给出逐格比值，五个轮次的范围是：

         N      逐轮比值范围（5 轮）
    131072     0.53 ~ 0.76        全部 < 1
    262144     0.61 ~ 0.91        全部 < 1
    524288     1.02 ~ 1.35        全部 > 1
   1048576     1.47 ~ 1.81
   4194304     1.99 ~ 2.51
  16777216     1.26 ~ 1.72

关键在 **262144 的上界 0.91 和 524288 的下界 1.02 之间没有重叠**。所以「交叉点落在两者之间」在负载波动下依然成立 —— 这个结论不依赖单次运气。

（`16777216` 那格 1.26~1.72 比 4M 的 1.99~2.51 低，是带宽从 181 GB/s 掉到 122 GB/s 导致的，见下段。）

结论只能写成**区间**：「交叉点落在 262144 ~ 524288 之间」。**不要记成一个具体数字** —— 这个数字在你自己机器上、甚至同一台机器不同负载下都会变。真正稳定的是三件事：门槛是 32768（硬编码常量）、唤醒成本是十几 µs 量级、交叉点比门槛高一个数量级以上。

原因还是那十来 µs 的唤醒成本：只有当「并行省下的时间」超过它，才真的赚。32769 个元素即使 4 线程完美加速也只省 `2.0 × 3/4 = 1.5 µs`，注定亏。

顺带能看出带宽墙在哪：`copy_` 的 4 线程带宽一路涨到 4M 处的约 190 GB/s，到 16M 反而掉到约 124 GB/s（N 太大，缓存和 TLB 都跟不上）。这个峰值就是这台机器的**多核可用带宽**，也是 4.3 节「`copy_` 加线程很快就撞墙」的物理依据。

顺带说一句这张表的读法风险：`1048576` 那一格的比值 1.52 和 `16777216` 那一格的 1.52 **恰好相等，但不是同一回事** —— 前者说明「百万级开始赚」（此时带宽才 156 GB/s，还没到峰），后者说明「涨到峰值后又被大 N 拖回来了」（带宽已从 187 掉到 124）。同一条曲线上的同两个数字，含义相反。所以正文里从不说「越大越并行越好」，只说「过了交叉点才赚，但赚多少由带宽定」。

---

### 4. 围绕该领域展开

#### 4.1 三个后端的并行实现对照（本期最该记住的表）

同一个 `x * 2 + 1`，16M fp32：

```python
import glob
import tempfile

import torch
import torch._dynamo as dynamo
import torch._inductor.config as icfg
import torch._inductor.runtime.triton_heuristics as th

x = torch.randn(16_000_000)

# ---------- A. eager ----------
torch.set_num_threads(4)
print("A. eager ATen")
print("   并行入口 = at::parallel_for，门槛 = GRAIN_SIZE(32768)")
print("   实测：N=32768 仍串行，N=32769 转并行")
print()

# ---------- B. torch.compile ----------
dynamo.reset()
d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
icfg.trace.enabled = True
icfg.trace.debug_dir = d
icfg.force_disable_caches = True
torch.compile(lambda v: (v * 2 + 1).relu(), dynamic=False)(x)
paths = glob.glob(f"{d}/**/output_code.py", recursive=True)
txt = open(paths[0]).read()
icfg.trace.enabled = False
icfg.force_disable_caches = False

print("B. torch.compile（CPU / C++ 后端）")
for ln in txt.split("\n"):
    if "pragma omp" in ln or "omp_get_thread_num" in ln:
        print("   " + ln.strip())
print(f"   config.cpp.threads={icfg.cpp.threads}  "
      f"min_chunk_size={icfg.cpp.min_chunk_size}  "
      f"dynamic_threads={icfg.cpp.dynamic_threads}")
print()

# ---------- C. Triton ----------
print("C. Triton（CUDA）—— 只有 block 网格，没有线程池")


class Cfg:
    def __init__(self, kw, num_warps, num_stages=1):
        self.kwargs, self.num_warps, self.num_stages = kw, num_warps, num_stages


th.Config = Cfg           # 本机没装 triton，塞个最小替身
numel = 16_000_000
bs = max(256, min(numel // 128, 1024))
for ew in (256, 64):
    c = th.triton_config({"x": numel}, bs, num_elements_per_warp=ew)
    xb, w = c.kwargs["XBLOCK"], c.num_warps
    print(f"   num_elements_per_warp={ew:>3}  XBLOCK={xb}  num_warps={w}"
          f"（{w * 32} 线程/block）  元素/线程={xb / (w * 32):.0f}")
```

```text
A. eager ATen
   并行入口 = at::parallel_for，门槛 = GRAIN_SIZE(32768)
   实测：N=32768 仍串行，N=32769 转并行

B. torch.compile（CPU / C++ 后端）
   #pragma omp parallel num_threads(4)
   int tid = omp_get_thread_num();
   #pragma omp for
   config.cpp.threads=-1  min_chunk_size=512  dynamic_threads=False

C. Triton（CUDA）—— 只有 block 网格，没有线程池
   num_elements_per_warp=256  XBLOCK=1024  num_warps=4（128 线程/block）  元素/线程=8
   num_elements_per_warp= 64  XBLOCK=1024  num_warps=8（256 线程/block）  元素/线程=4
```

三套实现的结构差异：

| | eager ATen | torch.compile CPU | Triton CUDA |
|---|---|---|---|
| 并行单元 | omp 线程池（`at::parallel_for`） | omp 线程池（裸 pragma） | block 网格 × warp |
| 门槛 | 32768 元素（绝对） | 每线程 512 元素（相对） | 无（autotune 选 XBLOCK） |
| 每线程元素数 | 由 ⌈N/线程数⌉ 决定，通常上万 | 同上 | **4~8** |
| 谁决定线程数 | `torch.set_num_threads` | 同左（编译期写死进 pragma） | `num_warps`（autotune） |
| 切多层 | 不支持（一次平铺） | **支持**（`decide_parallel_depth` 逐层切） | 支持（多维 grid） |

最反直觉的一行是「每线程元素数」：CPU 要**4 位数量级**才划算，GPU 只要**个位数**就够。这不是谁更先进，而是 GPU 的 block 网格是硬件调度器管的，没有「唤醒成本」这一项 —— 一个 block 起来了自己跑完就退出，无需 barrier。

#### 4.2 `config.cpp.dynamic_threads` 到底改了什么

同样是 `torch.compile`，这个开关把 `num_threads(4)` 从 pragma 里抹掉：

```python
import glob
import tempfile

import torch
import torch._dynamo as dynamo
import torch._inductor.config as icfg


def pragmas_of(dynamic_threads):
    dynamo.reset()
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    icfg.force_disable_caches = True
    icfg.cpp.dynamic_threads = dynamic_threads
    torch.compile(lambda v: (v * 2 + 1).relu(), dynamic=False)(torch.randn(4_000_000))
    txt = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    icfg.trace.enabled = False
    icfg.force_disable_caches = False
    icfg.cpp.dynamic_threads = False
    dynamo.reset()
    return [l.strip() for l in txt.split("\n") if "pragma omp" in l]


torch.set_num_threads(4)
print("dynamic_threads=False →", pragmas_of(False))
print("dynamic_threads=True  →", pragmas_of(True))
```

```text
dynamic_threads=False → ['#pragma omp parallel num_threads(4)', '#pragma omp for']
dynamic_threads=True  → ['#pragma omp parallel', '#pragma omp for']
```

源码依据在 `torch/_inductor/codegen/cpp.py:6202-6224`。这里还有个自动判定：**即使 `dynamic_threads=False`，只要线程数恰好等于 `os.cpu_count()`，也会自动改用动态版本**。

```python
if config.cpp.threads >= 1:
    use_dynamic = False                      # 用户显式设了，硬编码
elif threads == os.cpu_count():
    use_dynamic = True                       # 像是默认值，交给运行时
else:
    use_dynamic = False                      # 用户大概调过，硬编码
```

这解释了一个容易被当成 bug 的现象：同一份代码在两台机器上生成的 cpp 不一样，一台写死 `num_threads(4)`，另一台是裸 `#pragma omp parallel` —— **不是随机，是「当前线程数是否等于核数」这个条件在起作用**。这也意味着**生成的 kernel 是跟线程数绑定的，改 `set_num_threads` 会触发重新编译**（Dynamo 的 guard 里有 `GLOBAL_STATE changed: num_threads`）。

#### 4.3 加线程的收益：roofline 决定一切

这是本期最实用的结论。三个逐元素算子，同样 64 MB：

```python
import statistics
import time

import torch

N = 16_000_000
A = torch.randn(N)
B = torch.empty(N)


def one(fn, nt):
    torch.set_num_threads(nt)
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def paired(fn, nt_a, nt_b, rounds=40):
    """成对交替采样，取逐对比值的中位数 —— 比「中位数除中位数」抗漂移"""
    for _ in range(6):
        one(fn, nt_a)
        one(fn, nt_b)
    ra, rb, rr = [], [], []
    for i in range(rounds):
        if i % 2 == 0:
            x, y = one(fn, nt_a), one(fn, nt_b)
        else:
            y, x = one(fn, nt_b), one(fn, nt_a)
        ra.append(x)
        rb.append(y)
        rr.append(x / y)
    return min(ra), min(rb), statistics.median(rr)


cases = {
    "copy_": lambda: B.copy_(A),
    "add": lambda: torch.add(A, 1.0, out=B),
    "exp": lambda: torch.exp(A, out=B),
}

print(f"{'算子':<8}{'2 线程 ms':>10}{'1→2 比值':>10}"
      f"{'4 线程 ms':>10}{'1→4 比值':>10}")
for tag, fn in cases.items():
    _, t2, r12 = paired(fn, 1, 2)
    _, t4, r14 = paired(fn, 1, 4)
    print(f"{tag:<8}{t2 * 1e3:>10.2f}{r12:>10.2f}{t4 * 1e3:>10.2f}{r14:>10.2f}")
```

```text
算子         2 线程 ms    1→2 比值   4 线程 ms    1→4 比值
copy_         1.02      1.38      1.33      1.04
add           1.25      1.08      1.38      1.13
exp           6.79      1.95      4.07      2.97
```

多个独立进程各跑一遍后，`copy_` 1→4 线程落在 **1.04~1.11×**、`add` **1.07~1.17×**、`exp` **2.96~3.04×**。三者都不跨过「1 附近」和「3 附近」这两条线，结论稳定。

怎么读这张表：

- **`copy_` 和 `add` 在 2 线程后就撞墙**。它们纯粹受内存带宽限制，2 线程就把带宽推到了这台机器可用带宽的绝大部分，再加线程只是抢同一条总线。
- **`exp` 近似线性加速到 4 线程**（约 3.0× / 4 = 75% 并行效率）。`exp` 有除法、有超越函数，算术强度远高于 add，1 线程时 CPU 在算而不是在等内存，所以加核能真加速。

一句话判据：**算术强度高的算子该加线程，算术强度低的加了白加**。这和 041 期「排法决定贴带宽还是贴算力下界」是同一件事的另一面 —— 041 讲怎么排，本期讲派几个人。

#### 4.4 `at::in_parallel_region()`：防止嵌套爆炸

`parallel_for` 的第三个门槛条件是 `!at::in_parallel_region()`。它的实现在 OpenMP 后端就是一行：

```cpp
bool in_parallel_region() {
  return omp_in_parallel();
}
```

意思是：**如果当前已经在 omp 区域里，就别再开一层**。没有这条，任何「并行算子内部又调并行算子」的组合都会 fork 出 4×4=16 个线程。

这条也解释了 softmax 为什么是「并行在 outer 维、串行在 lastdim 维」（`SoftMaxKernel.cpp:70-92`）：

```cpp
parallel_for(0, outer_size, 0, [&](int64_t begin, int64_t end) {
  for (const auto i : c10::irange(begin, end)) {     // 每个线程串行走若干行
    scalar_t max_input = vec::reduce_all<scalar_t>(...);   // 行内归约：向量化，不再并行
    vec::map(...);
    scalar_t tmp_sum = vec::reduce_all<...>(...);
    ...
  }
});
```

外层 N 行并行，内层一行内部**只向量化不并行**。因为一行内部的归约要靠 `vec::reduce_all` 做横向加法，再并行就把累加顺序拆碎了 —— 这直接连到 041 期「排法改变浮点结果」的结论。

注意这个 `grain_size` 传的是 **0**，源码注释写得很清楚：`grain_size=0` 表示「不受门槛限制，一律并行」，并让 `invoke_parallel` 自己分配（`divup(N, 0)` 在 `grain_size > 0` 判断下不会执行，所以 `num_threads` 保持 `omp_get_num_threads()`，即全部线程都上）。

> 注意区分：`grain_size = 0` 的语义是「**没有门槛，永远并行**」，与 041 期 `decide_parallel_depth` 里 `depth = 0` 是「**不并行**」正好相反。同名不同义，读代码时容易搞混。

#### 4.5 与已发期数的连接

| 期数 | 关联点 |
|---|---|
| 041 期 | 融合体内部的「并行 P」怎么切 —— 本期讲清那个 P 的算法与门槛 |
| 040 期 | 融合把 24 个算子变 1 个，**并行门槛也就只付一次的 15 µs**，而不是 24 次 |
| 003 期 | `torch.compile` 的 CPU 后端用裸 pragma，门槛与 eager 差 16 倍 |
| 001 期 | CUDA Graph 省的是 launch 开销（CPU→GPU 的固定税）；本期量的是 CPU 内部的等价物：omp 唤醒税 |
| 014 期 | profiler 里看到的算子耗时已经含了唤醒成本，但**看不出**它 |
| 037 期 | 「加线程变快」这个判断的零假设是「1 线程耗时的 1/4」，实测 `copy_` 只有 1.10× |

040 那条最容易被忽略：融合的价值不只是省内存往返，**还顺带把 N 次唤醒税压成 1 次**。24 个算子各自 15 µs → 360 µs，融成一个之后只付 15 µs。在小张量上这一项甚至可能超过内存往返的收益。

---

### 5. 什么时候该用 / 不该用

**该加线程：**

- 元素数 ≫ 262144（经验交叉点，不是 32768）。- 算术强度高的算子：`exp`/`log`/`tanh`/`softmax`/`rsqrt`/卷积、GEMM。
- 一次调用算大量元素且**周围没有别的并行**（图里是串行的）。

**不该加（加了白加甚至更慢）：**

| 场景 | 原因 | 实测证据 |
|---|---|---|
| N < 131072 的逐元素算子 | 赚不回 15 µs 唤醒税 | N=65536 比值 0.38 |
| 纯搬运 / 逐元素低强度 | 2 线程即打满带宽 | `copy_` 1.37× → 1.10× |
| 已经在 omp 区域内部 | `in_parallel_region()` 会拦住，但代码里若有别的并行机制就会真嵌套 | — |
| 小 batch 推理（batch=1 的 decode） | 每个算子元素数太少，唤醒税占比极高 | 推论（与 028 期 decode 场景同源） |

**该改的其实是别的旋钮：**

- 小张量 → 降线程数（`torch.set_num_threads(1)`），把 15 µs 的门票彻底省掉。
- 多进程并行（`DataLoader` worker）比 intra-op 并行更划算，当你有独立样本要处理时。
- 融合（040 期）+ 向量化（041 期）比加线程的收益上限更高：前者是带宽账，后者是核数账。

---

### 6. 常见坑

**坑 1：`torch.set_num_threads` 不是「越多越好」，而且它会让 kernel 重编译**

`torch.compile` 生成的 kernel 把线程数写死在 `#pragma omp parallel num_threads(4)` 里。Dynamo 的 guard 里有 `GLOBAL_STATE changed: num_threads`，所以你**在 benchmark 循环里改线程数会触发反复重新编译**，测出来的数字里混进了编译时间。

```python
import torch

f = torch.compile(lambda v: v * 2, dynamic=False)
x = torch.randn(1000)

for nt in (1, 2, 4, 8):
    torch.set_num_threads(nt)
    f(x)   # 每一轮都可能重编译
```

正确做法：每个线程数配置**单独跑一个进程**，或者用 `torch._dynamo.reset()` 显式清缓存并预热足够多次。

**坑 2：`grain_size = 0` 和 `depth = 0` 语义相反**

- `at::parallel_for(0, N, 0, f)`：grain_size=0 表示**没有门槛，一律并行**。
- `decide_parallel_depth(...)` 返回 `depth=0`：表示**不并行**（T=1 时第一次循环就 break）。

两个 `0` 出现在同一个话题里，含义正好反着。读代码时先看上下文。

**坑 3：Mac 上 `torch.utils.cpp_extension` 编译的扩展里，`at::parallel_for` 默认是串行的**

clang 默认不支持 `-fopenmp`，而 `ATen/Parallel.h` 的并行实现在 `#if AT_PARALLEL_OPENMP` 里，未定义时 `parallel_for` 走 `#else` 直接串行执行。**不报错、不警告、`at::get_num_threads()` 还照样返回 4** —— 你会以为线程池在跑，其实一个线程都没有。

编译时要加：

```python
extra_cflags=["-Xpreprocessor", "-fopenmp"],
extra_ldflags=["-lomp"],
```

（`-lomp` 能链上 PyTorch 自带的 libomp；本机不装 homebrew libomp 也能跑通。）

**坑 4：`min_chunk_size` 门槛与 `grain_size` 门槛差 16 倍，别拿一个的结论套另一个**

| | eager | compile |
|---|---|---|
| 门槛 | N > 32768 | N / threads ≥ 512 |
| T=4 时 | 32768 | 2048 |

同样是「16K 元素的逐元素 op」，eager 完全串行，compiled 已经开了 4 线程。这会导致两个执行器在小张量上的相对性能出现非单调拐点（见 041 期「8192~65536 有一层 ~30 µs 固定平台」）。

**坑 5：`torch.compile` 生成的 kernel 里 `num_threads` 是编译期字面量，`set_num_threads` 改不动已经编译好的那份**

`trace.debug_dir` 抓 `output_code.py` 会看到这一行：

```cpp
#pragma omp parallel num_threads(4)
```

`4` 是生成代码那一刻的线程数，写死在字符串里。之后再 `torch.set_num_threads(1)`，这个 kernel 照样唤醒 4 个线程 —— 实测 1 线程下 `torch.compile` 版仍要约 17 µs，而 eager 的同表达式只要约 5 µs（因为 `at::parallel_for` 的 `use_parallel` 含 `at::get_num_threads() > 1`，T=1 直接串行）。

判据很简单：**改完线程数，去看生成的 C++ 是否变了。没变就是没生效。** 想让它是动态的，要么别动 `torch.set_num_threads`（保持等于 `os.cpu_count()`），要么显式设 `torch._inductor.config.cpp.dynamic_threads = True` —— 但改 config 前记得 `torch._dynamo.reset()`，否则 Dynamo 缓存会返回旧代码（这是上期踩过的坑）。

开关的实际效果，抓两次生成代码对比就一目了然：

```python
import glob
import os
import tempfile

import torch
from torch._inductor import config as icfg

icfg.force_disable_caches = True
icfg.trace.enabled = True


def pragma_line(dynamic):
    torch.set_num_threads(4)       # 固定成 4，否则生成的 num_threads 会随环境变量变
    icfg.cpp.dynamic_threads = dynamic
    torch._dynamo.reset()          # 换了 config 必须 reset，否则拿回旧代码
    d = tempfile.mkdtemp()
    icfg.trace.debug_dir = d
    fn = torch.compile(lambda v: v * 2 + 1, dynamic=False)
    fn(torch.randn(65536))
    for p in glob.glob(os.path.join(d, "**", "output_code.py"), recursive=True)[:1]:
        for ln in open(p).read().splitlines():
            if "#pragma omp parallel" in ln:
                return ln.strip()
    return "<没找到>"


print("dynamic_threads=True  →", pragma_line(True))
print("dynamic_threads=False →", pragma_line(False))
```

```text
dynamic_threads=True  → #pragma omp parallel
dynamic_threads=False → #pragma omp parallel num_threads(4)
```

顺带记一个同类的读数陷阱：`omp_get_num_threads()` 在**串行区恒返回 1**（不是 `omp_get_max_threads()`）。想在扩展里验证「并行是否生效」，必须把这个调用放在 `#pragma omp parallel` 内部；放在外面读到的 1 会被误读成「并行没生效」—— 这正是本期坑 3 里我排查了两小时的那个假象的第一层。

---

### 7. 一句话总结

**PyTorch 在 CPU 上的并行不是一套实现，是三套：eager 走「门槛 32768 元素 / 唤醒十几 µs / 按线程均分」的 `at::parallel_for`，`torch.compile` 生成的 kernel 走「阈值每线程 512 元素 / `num_threads` 编译期写死」的裸 omp，Triton 干脆没有线程池只有 block 网格 —— 而加线程能不能赚，取决于你的算子站在 roofline 的哪一侧、以及固定开销有没有被盖住，不取决于你调到了几。**

---

<details>
<summary>今日练习</summary>

**题目**：下面这段代码在你的机器上，`f` 的 4 线程相对 1 线程会快多少？先算再跑。

```python
import torch

torch.set_num_threads(4)
g = torch.compile(lambda v: v * 2 + 1, dynamic=False)
x = torch.randn(4096)   # 只有 4096 个元素
```

**提示**：注意 `torch.compile` 的门槛是 `seq // threads < min_chunk_size(512)`。

**参考答案**

`4096 // 4 = 1024 ≥ 512` → **不 break，会真的生成 `#pragma omp parallel`**。所以 4 线程下它确实开了 4 个线程。但是：

- 每个线程只分到 1024 个元素，fp32 就是 4 KB，读+写 8 KB。
- 按 130 GB/s 算，搬运 1024 个元素只需 **0.016 µs**。
- 而一次 omp 唤醒要 **13~18 µs**。

所以理论上应该慢几百倍 —— 实测确实**慢 4 倍左右**：

```python
import time
import torch

f = torch.compile(lambda v: v * 2 + 1, dynamic=False)


def bench(nt, N, reps=100):
    torch.set_num_threads(nt)
    x = torch.randn(N)
    for _ in range(15):
        f(x)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        f(x)
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


for N in (1024, 4096, 2048):
    t1, t4 = bench(1, N), bench(4, N)
    print(f"N={N:>6}  1 线程 {t1:7.2f} µs  4 线程 {t4:7.2f} µs  比值 {t1 / t4:.2f}")
```

```text
N=  1024  1 线程   10.62 µs  4 线程   10.83 µs  比值 0.98
N=  4096  1 线程   11.83 µs  4 线程   31.33 µs  比值 0.38
N=  2048  1 线程   11.50 µs  4 线程   46.29 µs  比值 0.25
```

**真正的考点在这里**：N=1024 的比值是 0.98（`1024 // 4 = 256 < 512` → break，**根本没生成 omp**，1 线程和 4 线程跑的是同一份代码，这点差异只是噪声）；N=2048 和 N=4096 一旦越过 512 门槛、真的唤醒 4 个线程，比值马上掉到 0.25~0.38。

也就是说：**门槛判断本身是二值的，但代价不是**。跨过门槛那一刻，你付出的不是「1/4 的时间」也不是「相等的时间」，而是**加上唤醒成本后的 2.6~4 倍**。这条曲线在 2048 附近有一个数倍的向下跳变，跳变的起因只是一行 `seq // threads < 512`。

顺带一个观察：三个 N 的 1 线程耗时几乎一样（10~11 µs），说明这点计算量确实全被 Python→C++ 边界吃掉了 —— 但 4 线程那条线**没有被吃掉**，因为唤醒 4 个线程要 30~40 µs，比边界开销还大一个量级，它自己就是主要成本。

所以这道题真正考的是**测量分辨率的两面**：小 N 上你测到的 10 µs 是解释器开销、不是 kernel 开销；但只要你把线程数开起来，唤醒成本就会从这层壳里戳出来。要测「并行有没有用」，要么把 N 扫到唤醒成本之上（≥ 10⁵），要么直接读生成的 C++ 源码。

验证一下（把 N 扫大，编译版才开始区分）：

```python
import time
import torch

g = torch.compile(lambda v: v * 2 + 1, dynamic=False)


def bench(nt, N, warm=15, reps=100):
    torch.set_num_threads(nt)
    x = torch.randn(N)
    for _ in range(warm):
        g(x)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        g(x)
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


print(f"{'N':>9}{'1 线程 µs':>12}{'4 线程 µs':>12}{'比值':>8}")
for N in (4096, 65536, 4194304):
    t1, t4 = bench(1, N), bench(4, N)
    print(f"{N:>9}{t1:>12.2f}{t4:>12.2f}{t1 / t4:>8.2f}")
```

```text
        N     1 线程 µs     4 线程 µs      比值
     4096       11.00       44.87    0.25
    65536       17.21       49.71    0.35
  4194304      397.00      251.38    1.58
```

`N=4096` 时 4 线程慢 4.1 倍，`N=65536` 时仍然慢 2.9 倍 —— 编译版的 2048 门槛让它在小 N 上就开了线程，而这点活根本不够付 30 µs 的门票。要到 4M 量级才赚回来（**1.58×**）。

再看 `N=65536`：eager 在 3.2 节里比值 0.38（慢 2.6 倍），编译版 0.35（慢 2.9 倍）。**两者几乎一样差，编译版略亏。**

这里有个容易想错的地方：编译版的门槛（2048）比 eager（32768）低一个数量级，很多人会顺推「编译版在小张量上更容易亏」。但「门槛低」只决定**要不要**进并行区，不决定**进去之后划不划算**。我一开始还猜「编译版生成的代码更薄、所以固定开销更小」，实测直接否掉了：

```python
import time
import torch

f = torch.compile(lambda v: v * 2 + 1, dynamic=False)
N = 65536
x = torch.randn(N)
y = torch.empty(N)


def bench(fn, reps=300, warm=30):
    for _ in range(warm):
        fn()
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


print(f"{'set_num_threads':>16}{'compile µs':>13}{'eager add µs':>15}")
for nt in (1, 2, 4, 8):
    torch.set_num_threads(nt)
    tc = bench(lambda: f(x))
    te = bench(lambda: torch.add(x, 1.0, out=y))
    print(f"{nt:>16}{tc:>13.2f}{te:>15.2f}")
```

```text
 set_num_threads   compile µs   eager add µs
               1        16.75           5.21
               2        30.17          11.38
               4        39.58          12.21
               8        63.04          22.50
```

**编译版在每一个线程数下都比 eager 慢 3~4 倍**，外壳薄的说法是错的。

（这一格的多线程行会飘：`set_num_threads(2)` 那格我四次跑在 31~45 µs 之间，因为 4 个性能核上要塞 2 个 omp 线程、剩下的被系统调度抢。所以只看两端：1 线程 16 µs、8 线程 65 µs，趋势是稳的、倍数区间是稳的，具体某一格别当精确值。）

原因在生成的源码里，`trace.debug_dir` 抓到的 `output_code.py` 第 40 行写着：

```cpp
    #pragma omp parallel num_threads(4)
```

`num_threads(4)` 是**编译期写死的字面量**，取自 `torch.compile` 那一刻的线程数。所以后面 `set_num_threads(1)` 根本改不动它 —— 上表 1 线程那一行约 17 µs 里，仍然结结实实地唤醒了 4 个 omp 线程。而 eager 的 `parallel_for` 里 `use_parallel` 含 `at::get_num_threads() > 1` 这一条，T=1 时直接走串行分支，一次 omp 都不唤醒。

那这 17 µs 里有几成是 omp？可以用「换线程数重新编译」来拆 —— 在 `torch.compile` **之前**设好线程数，生成的代码会不一样：

```python
import time

import torch

x = torch.randn(65536)


def compile_at(nt):
    torch.set_num_threads(nt)
    torch._dynamo.reset()          # 换线程数必须 reset，否则拿回旧图
    f = torch.compile(lambda v: v * 2 + 1, dynamic=False)
    f(x)
    return f


def best_of(fn, reps=300, warm=30):
    for _ in range(warm):
        fn()
    b = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b * 1e6


f1 = compile_at(1)
t1 = best_of(lambda: f1(x))
f4 = compile_at(4)
torch.set_num_threads(4)
t4 = best_of(lambda: f4(x))

print(f"compile(T=1 编，生成代码里无 omp)  {t1:6.2f} µs")
print(f"compile(T=4 编，生成代码里有 omp)  {t4:6.2f} µs")
print(f"→ 差值 ≈ 一次 omp 唤醒            {t4 - t1:6.2f} µs")
```

```text
compile(T=1 编，生成代码里无 omp)   18.54 µs
compile(T=4 编，生成代码里有 omp)   37.12 µs
→ 差值 ≈ 一次 omp 唤醒             18.58 µs
```

**T=1 编出来的代码里一条 `#pragma omp` 都没有**（`decide_parallel_depth` 第一次循环就 `par == threads` 而 break，正是第 2 节讲的那条），所以 18.54 µs 是纯粹的 **`torch.compile` Python 侧 wrapper 开销**，跟并行毫无关系。而 T=4 多出来的 18.58 µs，才是 omp 唤醒。

两个数几乎相等纯属巧合，但这一格给出的结论很重要：**在 65536 这个量级，编译版的耗时被「wrapper 18 µs + omp 18 µs」两笔固定成本均分，真正搬 256 KB 数据的时间（不到 1 µs）根本不是主要矛盾。** 所以讨论「并行划不划算」之前，先确认你要比的那部分是不是被固定开销盖住了 —— 这正是本期练习要考的点。

顺带一个反直觉的推论：**`torch.compile` 版在单线程场景下是「假的单线程」**。如果你先 `torch.compile` 再 `set_num_threads(1)`，生成的 kernel 照样 fork 4 个线程（上表 1 线程那一行就是这种情况）。要做真正的单线程对照实验，线程数必须在 `torch.compile` **之前**设好。

**这条也解释了 `dynamic_threads` 为什么存在**：它的作用就是把 `num_threads(4)` 换成裸 `#pragma omp parallel`，让运行时能改。`torch/_inductor/codegen/cpp.py:6209-6224` 的三段判定是：

```python
use_dynamic = False
if config.cpp.threads >= 1:
    use_dynamic = False              # 用户显式设了 config.cpp.threads，写死
elif threads == os.cpu_count():
    use_dynamic = True               # 线程数等于系统核数（多半是默认值），用动态
else:
    use_dynamic = False              # 线程数与系统不同（用户设过 set_num_threads），写死
if use_dynamic or config.cpp.dynamic_threads:
    code.writeline("#pragma omp parallel")
else:
    code.writeline(f"#pragma omp parallel num_threads({threads})")
```

读法很反直觉：**正好在你「什么都没设」的时候它是动态的；一旦你为了「省事」调了 `torch.set_num_threads(4)` 而不是系统的 10 核，它反而写死成 `num_threads(4)`。** 也就是说，想让编译版跟随运行时线程数，比想让编译版固定线程数要更难。

到 4M 这一格，两者就收敛了：编译版 1.54×，eager 在 1048576 上 1.57× —— 不是巧合，**都已经撞到内存带宽墙**（第 4.3 节的 roofline）。并行效率的上限由带宽定，不由门槛定。

</details>
