# PyTorch 每日一课 · 第 050 期

## Stream-K：split-K 与 persistent 之间的连续谱

| | |
|---|---|
| **日期** | 2026-11-05 |
| **难度** | ⭐⭐⭐⭐⭐ |
| **前置知识** | tile 与算术强度（043 期）、occupancy 与 wave（045 期）、epilogue 的写回路径（048 期）、split-K 的 $S^*$（049 期） |
| **预计阅读** | 30 分钟 |
| **关联** | 第 049 期：split-K 只能切整数份 → 本期问「切不整怎么办」；第 048 期：fixup 是 epilogue 的第五种出口 |

---

## 一、这个领域解决什么问题

先用一句话把结论摆出来，后面全部是它的展开：

> **GEMM 的并行度损失（wave quantization）有一个绝对上限——不到一个 tile 的时间。stream-K 就是把这不到一个 tile 的时间找回来的技术。代价是 $P-1$ 次 fixup，每次折合千分之一到一个百分之一 tile。所以它在大 GEMM 上只值 2.6%，在窄形状上值 49%。**

### 1.1 量化损失的绝对值有一个天花板

设 $M\times N$ 切出 $T$ 个输出 tile，机器有 $P$ 个执行单元（SM，或更准确地说 `ctas_per_wave`）。最朴素的调度——data-parallel，也是 persistent kernel 最常见的那种写法——把 tile 依次轮发给 CTA：

$$t_{DP} = \left\lceil \frac{T}{P} \right\rceil \cdot t_{tile}$$

利用率 $\eta = T / \left(P\lceil T/P\rceil\right)$，浪费掉的时间是

$$\Delta t = \left( \left\lceil \frac{T}{P} \right\rceil - \frac{T}{P} \right)\cdot t_{tile} = W(1-\eta)\cdot t_{tile}, \qquad W = \left\lceil T/P \right\rceil$$

注意括号里那一项。因为 $\lceil x \rceil - 1 < x \le \lceil x \rceil$，所以

$$0 \le W(1-\eta) < 1$$

**这就是本期的地基：不管 GEMM 多大，wave quantization 的绝对损失都不到一个 tile 的时间。**

这个恒等式的反面才是重点：

- $T=140, P=132$：损失 0.94 个 tile，总共 2 波 → 占 **47%**
- $T=4096, P=132$：损失 0.97 个 tile，总共 32 波 → 占 **3.0%**

绝对损失几乎一样（都不到一个 tile），相对损失差 16 倍。**所以问题从来不是「怎么彻底消除量化损失」——那部分有天花板，消除不了多少；问题是「在相对占比大的形状上，怎么把那不到一个 tile 抢回来」。**

### 1.2 先把两个物理前提量出来

stream-K 的全部推理建立在一个假设上：**一个 CTA 的耗时正比于它分到的 k-iter 数**。如果不成立（比如耗时随 $K$ 次线性），「均分 k-iter」就不等于「均分时间」，整个算法失去依据。

这条假设可以直接量：

```python
"""前提 (a)：单次 GEMM 的耗时是否线性于 K。
前提 (b)：多个执行单元能不能同时跑 GEMM（决定本机能不能测「多 CTA 并行」）。
"""
import subprocess
import sys
import time

import torch

torch.set_num_threads(1)


def bench(fn, reps=5, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


print(f"torch {torch.__version__}，单线程，tile 512x512")
print(f"\n(a) 耗时是否线性于 K：")
print(f"{'K':>7} {'ms':>9} {'GFLOPS':>9} {'每 4096 个 k 的 ms':>18}")
for K in [1024, 2048, 4096, 8192, 16384, 32768]:
    a = torch.randn(512, K)
    b = torch.randn(K, 512)
    t = bench(lambda: a @ b)
    print(f"{K:>7} {t*1e3:>9.3f} {2*512*512*K/t/1e9:>9.0f} {t*1e3/(K/4096):>18.3f}")

# 下面这段让 P 个独立进程同时跑同一个 GEMM，看总吞吐会不会涨
SRC = """
import sys, time
import torch
torch.set_num_threads(1)
a = torch.randn(512, 4096)
b = torch.randn(4096, 512)
c = torch.empty(512, 512)
for _ in range(150):
    torch.mm(a, b, out=c)          # 热身：把进程拉上性能核，并触发 page fault
print("ready", flush=True)
sys.stdin.readline()                # 等父进程统一放行
t0 = time.perf_counter()
for _ in range(300):
    torch.mm(a, b, out=c)
print("TIME", time.perf_counter() - t0, flush=True)
"""


def run(P):
    ps = [
        subprocess.Popen([sys.executable, "-c", SRC], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, bufsize=1)
        for _ in range(P)
    ]
    for p in ps:
        p.stdout.readline()         # 等 ready
    for p in ps:
        p.stdin.write("\n")
        p.stdin.flush()
    times = []
    for p in ps:
        for line in p.stdout:
            if line.startswith("TIME"):
                times.append(float(line.split()[1]))
        p.wait()
    return max(times)


print(f"\n(b) P 个独立进程同时跑同一个 GEMM（每进程 300 次 mm）：")
print(f"{'P':>3} {'墙钟 s':>9} {'每次 mm 的 ms':>14} {'总吞吐 GFLOPS':>14}")
for P in [1, 2, 4]:
    t = run(P)
    print(f"{P:>3} {t:>9.3f} {t/300*1e3:>14.2f} {P*300*2*512*512*4096/t/1e9:>14.0f}")
```

```text
torch 2.14.0，单线程，tile 512x512

(a) 耗时是否线性于 K：
      K        ms    GFLOPS    每 4096 个 k 的 ms
   1024     0.323      1663              1.292
   2048     0.619      1736              1.237
   4096     1.300      1652              1.300
   8192     2.609      1646              1.305
  16384     5.375      1598              1.344
  32768    10.975      1565              1.372

(b) P 个独立进程同时跑同一个 GEMM（每进程 300 次 mm）：
  P      墙钟 s     每次 mm 的 ms     总吞吐 GFLOPS
  1     0.388           1.29           1661
  2     0.767           2.56           1681
  4     1.581           5.27           1630
```

先看 (a)。每 4096 个 k-iter 的耗时从头到尾稳在 1.2~1.4 ms，**不随 K 单调**（最小值出现在 K=1024/2048，最大值出现在 K=32768），这是测量抖动不是系统性偏离。**线性假设成立**，$t \propto K$。

再看 (b)，这个结果很关键：4 个独立进程同时跑同一个 GEMM，单次耗时从 1.3 ms 涨到 5.3 ms——**正好 4 倍**。总吞吐钉在 1.6~1.7 TFLOPS 一动不动。

也就是说：**本机的矩阵单元是共享的，加进程不加算力。** 单进程单线程就已经吃满了这块芯片的矩阵吞吐。

这带来一个直接后果：**本机没法用 GEMM 做「多个 CTA 并行」的实验**，因为 P 个并发工作者的总吞吐恒等于 1 个。要测「尾波浪费了多少」，必须换一个能真正并行的负载。

下面用一段 C 写的纯 ALU 串行依赖链当「工作单元」——它不碰矩阵单元，能真正 4 路并行。**并行度是单独量的**：4 个进程各跑 2 秒纯计算，`RUSAGE_CHILDREN` 累计 CPU 时间 8.09 s、墙钟 2.04 s，比值 3.96。一个工作单元 = 固定次数的 FMA，对应「一个 k-iter 的计算量」。

### 1.3 在真机上把尾波量出来

P=4 个执行单元，每档重复 5~8 轮，两种调度方案：

- **方案 A（tile 分配）**：每个单元拿整块 tile，拿不到的闲置
- **方案 B（stream-K）**：把 $T$ 个 tile 的 k-iter 总量均分
- **方案 C**：方案 B 再加每轮末尾的 fixup 同步

```python
"""三个方案的真实墙钟：tile 分配 vs stream-K vs stream-K + fixup 同步。

工作单元用 C 现编的 ALU 依赖链（本机矩阵单元无法并行，见 1.2 的 (b)）。
多个执行单元用独立子进程，靠 stdin 做一轮一次的同步。
"""
import os
import subprocess
import tempfile

SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double burn(long iters) {
    double x = 1.000001, s = 0.0;
    for (long i = 0; i < iters; i++) { s = s * x + 1.0; s = s - 0.5; }  /* 串行依赖链 */
    return s;
}
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
int main(int argc, char **argv) {
    long warm  = atol(argv[1]);   /* 热身的迭代数 */
    long per   = atol(argv[2]);   /* 一个工作单元的迭代数 */
    int rounds = atoi(argv[3]);   /* 重复多少轮 */
    int units  = atoi(argv[4]);   /* 本轮分配给它几个工作单元 */
    int sync   = atoi(argv[5]);   /* 每轮末尾要不要等齐 */
    burn(warm);
    printf("ready\n"); fflush(stdout);
    getchar();
    double t0 = now(), acc = 0;
    for (int r = 0; r < rounds; r++) {
        for (int u = 0; u < units; u++) acc += burn(per);
        if (sync && r < rounds - 1) { printf("sync\n"); fflush(stdout); getchar(); }
    }
    printf("TIME %.6f\n", now() - t0);
    return acc > 1e300;
}
"""

WORK = tempfile.mkdtemp()
BIN = os.path.join(WORK, "burn")
with open(os.path.join(WORK, "burn.c"), "w") as f:
    f.write(SRC)
subprocess.run(["cc", "-O3", "-o", BIN, os.path.join(WORK, "burn.c")], check=True)

PER = 12_000_000   # 一个工作单元的迭代数，约 15 ms
WARM = 60_000_000  # 热身，把进程拉上性能核


def run(units_per_worker, rounds, sync=False):
    ps = [
        subprocess.Popen([BIN, str(WARM), str(PER), str(rounds), str(u), "1" if sync else "0"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        for u in units_per_worker
    ]
    for p in ps:                       # 等全部就绪
        p.stdout.readline()
    early = []
    for _ in range(rounds):            # 逐轮统一放行
        for p in ps:
            p.stdin.write("\n")
            p.stdin.flush()
        if sync:
            for p in ps:
                line = p.stdout.readline()   # 最后一轮这里读到的是 TIME
                if line.startswith("TIME"):
                    early.append(float(line.split()[1]))
    times = list(early)
    for p in ps:
        for line in p.stdout:
            if line.startswith("TIME"):
                times.append(float(line.split()[1]))
        p.wait()
    return max(times)


def dp_plan(T, P, kiter):
    """tile 分配：第 i 个单元拿 tile i, i+P, i+2P, ..."""
    u = [0] * P
    for t in range(T):
        u[t % P] += kiter
    return u


def sk_plan(T, P, kiter):
    """stream-K：把 T*kiter 个 k-iter 均分，余数给前几个（对应 CUTLASS 的 big_units）"""
    base, rem = divmod(T * kiter, P)
    return [base + (1 if i < rem else 0) for i in range(P)]


if __name__ == "__main__":
    P, KITER = 4, 4
    print(f"P={P} 个并发执行单元，每 tile 的 K 方向 = {KITER} 个 k-iter，"
          f"1 个 k-iter = {PER/1e6:.0f}M 次串行 FMA\n")
    for T, rounds in [(3, 8), (5, 5), (6, 5)]:
        a, b = dp_plan(T, P, KITER), sk_plan(T, P, KITER)
        ta = run(a, rounds)
        tb = run(b, rounds)
        tc = run(b, rounds, sync=True)
        ideal = max(a) / max(b)
        print(f"=== T={T} 个 tile（共 {T*KITER} 个 k-iter）===")
        print(f"  A tile 分配  每单元 units={a}  墙钟 {ta:>5.3f}s")
        print(f"  B stream-K   每单元 units={b}  墙钟 {tb:>5.3f}s  {ta/tb:>4.2f}x"
              f"（理论 {ideal:.2f}x，兑现 {(ta/tb)/ideal*100:>4.1f}%）")
        print(f"  C B+fixup同步                墙钟 {tc:>5.3f}s  {ta/tc:>4.2f}x")
        print()
```

```text
P=4 个并发执行单元，每 tile 的 K 方向 = 4 个 k-iter，1 个 k-iter = 12M 次串行 FMA

=== T=3 个 tile（共 12 个 k-iter）===
  A tile 分配  每单元 units=[4, 4, 4, 0]  墙钟 0.510s
  B stream-K   每单元 units=[3, 3, 3, 3]  墙钟 0.372s  1.37x（理论 1.33x，兑现 102.9%）
  C B+fixup同步                墙钟 0.374s  1.36x

=== T=5 个 tile（共 20 个 k-iter）===
  A tile 分配  每单元 units=[8, 4, 4, 4]  墙钟 0.622s
  B stream-K   每单元 units=[5, 5, 5, 5]  墙钟 0.393s  1.58x（理论 1.60x，兑现 98.9%）
  C B+fixup同步                墙钟 0.389s  1.60x

=== T=6 个 tile（共 24 个 k-iter）===
  A tile 分配  每单元 units=[8, 8, 4, 4]  墙钟 0.606s
  B stream-K   每单元 units=[6, 6, 6, 6]  墙钟 0.465s  1.31x（理论 1.33x，兑现 97.9%）
  C B+fixup同步                墙钟 0.466s  1.30x

```

三档数据，每一档都在说同一件事：

| T | 方案 A 的分配 | 方案 B 的分配 | 实测加速 | 理论 $1/\eta$ | 兑现率 |
|---|---|---|---|---|---|
| 3 | `[4,4,4,0]` | `[3,3,3,3]` | 约 1.35x | 1.33x | ≈100% |
| 5 | `[8,4,4,4]` | `[5,5,5,5]` | 约 1.57x | 1.60x | ≈98% |
| 6 | `[8,8,4,4]` | `[6,6,6,6]` | 约 1.31x | 1.33x | ≈98% |

（表中的加速比每次跑会抖 ±5% 左右，以代码块里的实际 stdout 为准。）

三点解读：

**第一，收益完全由「最长那个单元分到几个工作单元」决定，与算法细节无关。** T=3 时最长的分到 4 个，均分后是 3 个，$4/3=1.33$。就这么简单。

**第二，兑现率接近 100%。** 这说明「均分 k-iter」一旦成立，收益就是纯算术——不存在「实践中拿不到」的问题。stream-K 的难点全在**代价**那一侧（fixup），不在收益这一侧。

**第三，fixup 的同步开销小到测不出来。** 方案 C 每轮末尾加了一次全单元同步（等价于 fixup 的 barrier 成本），它的墙钟和方案 B 只差在小数点后第二位。

> **一个必须说明的差异**：T=5 那档实测约 1.57x 略低于理论 1.60x。原因是 CPU 上提前完成的进程会把核让出来，剩下的进程自动变快（尾波被部分自动补偿）；GPU 上 SM 是静态分区的，空转的 SM 不会加速剩余的 CTA。所以真 GPU 上这一档会更接近 1.60x。这是本机实验与 GPU 真实行为之间唯一的结构性差异，其余结论可以直接迁移。

### 1.4 049 期留的那个洞

049 期把 split-K 和 persistent 摆成了一个对偶，判据是 $P/C$（$P$ 是并发能力，$C$ 是 tile 数）：

> **$P/C > 2$ 才考虑 split-K，$P/C < 0.5$ 才考虑 persistent，中间那段什么都别做。**

这句话在「手上只有这两个工具」的前提下是对的：

- split-K 要求 $T$ 明显小于 $P$（否则切成 $S$ 份之后每份不到一个 k-tile，prologue 开销吃掉一切）
- persistent 要求 $T$ 明显大于 $P$（否则 grid 压缩没有意义）

但中间那段（$0.5 < C/P < 2$，也就是 tile 数是 SM 数的 0.5~2 倍）**恰恰是量化损失最严重的地方**：$T=140, P=132$ 时 $\eta=0.53$，有一半的 SM 在空转。

「什么都别做」= 接受 47% 的损失。

stream-K 是第三种工具，它的存在就是为了填这个洞。它不是 split-K 的改进版，而是**把两者统一起来的那个坐标系**。

---

## 二、核心思想：把工作单位从 tile 换成 k-iter

### 2.1 一个类比

049 期用的是「64 万笔账分给 64 个人」。这里换一个更能说明问题的版本。

假设你要核对 3 本书，每本 512 页，有 4 个人。

- **split-K 的做法**：按「本」分。3 本书给 4 个人——只能把每本对半撕开，得到 6 个半本，分给 4 个人。每人拿到 1.5 个半本，还是不均。
- **persistent 的做法**：一人一本，第 4 个人闲着。
- **stream-K 的做法**：不问「几本」，问「总共多少页」。$3\times512=1536$ 页，4 人各 384 页。切点必然落在某本书的中间——那就得**两个人交接**：一个人核完 1~384 页，另一个人从 385 页接着核，最后把两段小计加起来。

第三种做法工作量最均匀（各 384 页，误差 0），代价是**交接**。这个交接就是 fixup。

关键洞察：**split-K 的最小单位是「本」，stream-K 的最小单位是「页」。** 单位换细之后，「分不匀」问题消失了，但「交接」问题出现了。

### 2.2 连续谱

定义一个无量纲参数：

$$\lambda = \frac{T}{P}$$

「每个执行单元平均分到几个 tile」。三种调度不是三种技术，而是同一个算法的三个区间：

| $\lambda$ | 名称 | 一个单元分到 | 归约发生在 | 谁在用 |
|---|---|---|---|---|
| $\lambda < 1$ | split-K | 不到一个 tile 的 K | 跨 $S$ 个 CTA | 049 期、Inductor 的 cooperative reduction |
| $1 \le \lambda \lesssim 2$ | **stream-K** | 1~2 个 tile 的 K | 跨相邻的 unit | 本期 |
| $\lambda \gg 1$ | tile 分配 / persistent | 整数个 tile | 不归约 | 049 期的 persistent |

$\lambda < 1$ 时，一个单元连一个 tile 的 K 都吃不下 → 必然跨 tile 边界 → 每个 tile 都被切 → 这时它退化成 split-K。
$\lambda \gg 1$ 时，一个单元能连吃好几个完整 tile → 切点几乎总落在 tile 边界上 → 退化成 tile 分配。

**所以 stream-K 不是一种新算法，而是「按 k-iter 均分」这个规则在 $\lambda \approx 1$ 时的自然形态。** 这个视角的价值在于：你不需要判断「该用哪个」，只需要算 $\lambda$。

### 2.3 代价在 cut，不在 merge

fixup 要付出三样东西：

1. **一块 workspace**：每个被切开的 tile 要存一份 partial 累加器
2. **一个同步原语**：谁先写完、谁最后到、谁负责归约
3. **一次额外的读写**：partial 写进去、再读出来

但要注意一件事：**cut 的次数是「单元数 $-1$」，不是「tile 数」。**

$T=140$ 个 tile 分给 $P=132$ 个单元，切点有 131 个。如果认为「每个 tile 都要 fixup」，就是 140 次；实际上一次 cut 只影响它落到的那一个 tile。下一节会用代码验证：140 个 tile 确实全被切开了，但 cut 的总次数是 131，等于 unit 数减一。

这个「$P-1$ 而不是 $T$」是 stream-K 成本可控的根本原因——**它随机器规模走，不随问题规模走**。

---

## 三、在 PyTorch 中怎么用

### 3.1 现状：PyTorch 没有 stream-K

这是必须先说清的事实，免得你去翻 API：

| 层 | 有没有 stream-K | 证据 |
|---|---|---|
| CUTLASS | **有** | `KernelSchedule::StreamKScheduler` → `PersistentTileSchedulerSm90StreamK` / `Sm100StreamK` |
| Inductor | 没有 | 本机 torch 2.14.0 的 `torch/_inductor/` 下 grep `stream_k\|streamk\|stream-K` **零命中** |
| Triton 官方 tutorial | 没有 | `python/tutorials/09-persistent-matmul.py` 有 persistent + TMA + warp specialization，grep `stream` 只命中 CUDA stream 参数 |

所以现阶段 stream-K 是**CUTLASS 层的技术**：你写 `.cu` 调 CUTLASS 时有，你写 `torch.matmul` 时没有。

Inductor 手上有两个近亲，正好是这条谱的两端：

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

- `persistent_mm_grid` 把 grid 压到 `min(NUM_SMS, tiles)` —— $\lambda \gg 1$ 端
- `cooperative_reductions`（049 期讲过，默认关）按**整 tile** 切 K —— $\lambda < 1$ 端

中间那段（$\lambda \approx 1$），Inductor 目前没有对应机制。这就是为什么窄形状的 GEMM 在 `torch.compile` 下仍然会浪费一半 SM。

### 3.2 如果你要用，只能自己写

最小骨架长这样（Triton 风格，**本机无 GPU，这段未编译未运行**，只示意结构）：

```py
import triton
import triton.language as tl


@triton.jit
def streamk_matmul(a_ptr, b_ptr, c_ptr, ws_ptr, lock_ptr,
                   M, N, K, NUM_SMS: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

    # 整个问题的线性化迭代空间：总 k-iter 数 = ceil(M/BM)*ceil(N/BN)*ceil(K/BK)
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    k_iters = tl.cdiv(K, BLOCK_K)
    total = num_tiles_m * num_tiles_n * k_iters

    # 均分：前 (total % NUM_SMS) 个单元多拿一个，对应 CUTLASS 的 big_units
    per = total // NUM_SMS
    rem = total % NUM_SMS
    start = pid * per + tl.minimum(pid, rem)
    count = per + (1 if pid < rem else 0)
    end = start + count

    # 从后往前遍历自己覆盖的 tile（CUTLASS 的注释解释了为什么要反向）
    tile = (end - 1) // k_iters
    while tile >= start // k_iters:
        k_lo = tl.maximum(start - tile * k_iters, 0)
        k_hi = tl.minimum(end - tile * k_iters, k_iters)
        # ... 在 [k_lo, k_hi) 上做这个 tile 的累加，得到 acc ...

        # 是不是这个 tile 的最后一段？是就直接写 C，不是就写 workspace
        # 用 lock 的到达计数判断自己是不是最后一个
        # ... fixup: 最后一个到达的 peer 把 workspace 里的 partial 加起来，跑 epilogue ...

        tile -= 1
```

四个要点，每一处都对应 CUTLASS 里的一个具体设计：

1. **grid = `NUM_SMS`**，不是 `num_tiles`——这是个 persistent kernel
2. **切点是 `start` / `end` 两个整数**，不是 tile 索引——一个单元可以跨越 tile 边界
3. **反向遍历**（从 `end-1` 所在 tile 往 `start` 所在 tile 走）——因为「一个单元区间的起点」是上一个单元的终点，反向走让「需要被别人等的那部分」最后算，减少阻塞（`sm90_tile_scheduler_stream_k.hpp:924` 的注释原话是 *Stream-K work units will work backwards*）
4. **lock 是一个到达计数器**，不是自旋标志——`Deterministic` 模式下第 $i$ 个 peer 等 `lock == i`，实际形成一条串行链

### 3.3 没有旋钮

诚实地说：如果你在 `torch.matmul` 这一层，**没有任何 flag 能让它走 stream-K**。你的三个选择是：

1. 换 CUTLASS（改 kernel schedule 模板参数）
2. 自己写 Triton / CUDA
3. 认了——先算一下 $\lambda$ 和 $\eta$，如果量化损失本来就只有 3%，那确实不必折腾

第三条其实是最常被忽略的正确答案。下一节会给出算这笔账的方法。

---

## 四、围绕该领域展开

### 4.1 CUTLASS 的三条闸

CUTLASS 的启发式比想象中保守得多。下面这段逐行对应 `tile_scheduler_params.h:1068-1113` 的 `get_num_sk_tiles`：

```python
"""把 CUTLASS 的 stream-K 启发式翻成 Python，扫真实形状。

对应源码（NVIDIA/cutlass main）：
  include/cutlass/gemm/kernel/tile_scheduler_params.h
    :470         min_iters_per_sk_unit_ = 8u
    :1068-1113   get_num_sk_tiles()
    :1119-1140   get_num_sk_units()
"""
import math

P = 132          # H100 SXM5 的 SM 数
BM = BN = 128
BK = 16          # fp16 wgmma 的 k 步长
MIN_ITERS = 8


def gate(tiles, kiter, cluster=1):
    """返回 (sk_tiles, 哪条闸把它拦住了)"""
    full, total = tiles // P, math.ceil(tiles / P)
    if full == total:                      # 闸 1：没有量化损失
        return 0, "无量化损失"
    if kiter <= MIN_ITERS:                 # 闸 2：K 方向太短
        return 0, f"K 方向只有 {kiter} 个 k-tile"
    tail = tiles - full * P
    if 2 * tail >= P:                      # 闸 3：尾波超过半满
        return 0, f"尾波 {tail} 个 >= P/2"
    dp_tiles = (full - 1) * P if full > 1 else 0   # 前 full-1 波照旧走 data-parallel
    sk = tiles - dp_tiles                          # 只有最后两波交给 stream-K
    return (sk // cluster) * cluster, f"通过（前 {dp_tiles} 个 tile 走 DP）"


def sk_units(sk, kiter, cluster=1):
    return min(P, ((kiter * sk) // MIN_ITERS) // cluster * cluster)


print(f"P={P} SM，tile {BM}x{BN}，k-tile={BK}")
head = f"{'M':>6}{'N':>6}{'K':>6}{'tiles':>6}{'W':>3}{'eta':>7}{'sk_tiles':>9}{'sk_units':>9}  判定"
print(head)
print("-" * 76)
for M, N, K in [(1024, 1024, 8192), (1280, 1792, 8192), (1536, 3200, 8192),
                (2048, 2048, 8192), (4096, 4096, 8192), (8192, 8192, 8192),
                (1280, 1792, 128)]:
    tiles = math.ceil(M / BM) * math.ceil(N / BN)
    kiter = math.ceil(K / BK)
    waves = math.ceil(tiles / P)
    sk, why = gate(tiles, kiter)
    u = sk_units(sk, kiter) if sk else 0
    print(f"{M:>6}{N:>6}{K:>6}{tiles:>6}{waves:>3}{tiles/(waves*P):>7.3f}"
          f"{sk:>9}{u:>9}  {why}")

print("\n量化损失的锯齿结构（K=8192，扫 tile 数）：")
print(f"{'tiles':>6}{'W':>3}{'eta':>7}{'损失tile':>9}{'上限':>7}{'sk':>5}  判定")
for tiles in [128, 132, 133, 140, 200, 264, 265, 300, 396, 400]:
    waves = math.ceil(tiles / P)
    sk, why = gate(tiles, 512)
    eta = tiles / (waves * P)
    print(f"{tiles:>6}{waves:>3}{eta:>7.3f}{waves*(1-eta):>9.3f}{1/eta:>6.2f}x{sk:>5}  {why}")
```

```text
P=132 SM，tile 128x128，k-tile=16
     M     N     K tiles  W    eta sk_tiles sk_units  判定
----------------------------------------------------------------------------
  1024  1024  8192    64  1  0.485       64      132  通过（前 0 个 tile 走 DP）
  1280  1792  8192   140  2  0.530      140      132  通过（前 0 个 tile 走 DP）
  1536  3200  8192   300  3  0.758      168      132  通过（前 132 个 tile 走 DP）
  2048  2048  8192   256  2  0.970        0        0  尾波 124 个 >= P/2
  4096  4096  8192  1024  8  0.970        0        0  尾波 100 个 >= P/2
  8192  8192  8192  4096 32  0.970      136      132  通过（前 3960 个 tile 走 DP）
  1280  1792   128   140  2  0.530        0        0  K 方向只有 8 个 k-tile

量化损失的锯齿结构（K=8192，扫 tile 数）：
 tiles  W    eta   损失tile     上限   sk  判定
   128  1  0.970    0.030  1.03x    0  尾波 128 个 >= P/2
   132  1  1.000    0.000  1.00x    0  无量化损失
   133  2  0.504    0.992  1.98x  133  通过（前 0 个 tile 走 DP）
   140  2  0.530    0.939  1.89x  140  通过（前 0 个 tile 走 DP）
   200  2  0.758    0.485  1.32x    0  尾波 68 个 >= P/2
   264  2  1.000    0.000  1.00x    0  无量化损失
   265  3  0.669    0.992  1.49x  133  通过（前 132 个 tile 走 DP）
   300  3  0.758    0.727  1.32x  168  通过（前 132 个 tile 走 DP）
   396  3  1.000    0.000  1.00x    0  无量化损失
   400  4  0.758    0.970  1.32x  136  通过（前 264 个 tile 走 DP）
```

三条闸，一条比一条不直观：

**闸 1：`full_waves == total_waves` → 不做。** 没有量化损失时，引入 stream-K 只会白付 fixup。

**闸 2：`k_tiles_per_output_tile <= 8` → 不做。** 这是 `min_iters_per_sk_unit_ = 8`（`:470`）的另一面。K 方向只有 8 个 k-tile 时，切出来的每一份连 8 个 k-iter 都不到——pipeline 的 prologue 都填不满，切了纯亏。表里 `1280×1792, K=128` 就是被这条拦住的（$128/16 = 8$，正好卡在边界上）。

**闸 3：`2 * tail_tiles >= ctas_per_wave` → 不做。** 尾波超过半满就放弃。

这条最值得盯。看表里这两行：

```
  2048  2048  8192   256  2  0.970        0        0  尾波 124 个 >= P/2
  8192  8192  8192  4096 32  0.970      136      132  通过（前 3960 个 tile 走 DP）
```

**两行的 $\eta$ 完全一样（都是 0.970），判定却相反。** 因为 $\eta$ 相同不代表尾波相同：

- $2048^2$：256 个 tile，2 波，最后一波有 124 个 tile（占 94% 的 SM）
- $8192^2$：4096 个 tile，32 波，最后一波只有 4 个 tile（占 3% 的 SM）

判据看的是**尾波的绝对大小**（124 个 vs 4 个），不是利用率。这是一个很容易搞错的地方——如果你自己实现判断逻辑，用 $\eta < 0.9$ 这类阈值会在 $2048^2$ 上误触发。

再看锯齿：

```
   132  1  1.000    0.000  1.00x    0  无量化损失
   133  2  0.504    0.992  1.98x  133  通过
   200  2  0.758    0.485  1.32x    0  尾波 68 个 >= P/2
   264  2  1.000    0.000  1.00x    0  无量化损失
   265  3  0.669    0.992  1.49x  133  通过
```

$T=132$ 时完美整除，损失 0；多一个 tile（$T=133$）立刻掉到 $\eta=0.504$，损失接近一整波。这就是 wave quantization 的锯齿形状：**损失在 $T$ 略大于 $P$ 的整数倍时最大，在整除时为零。** 而 $T=200$ 时损失 0.485 个 tile，看着不少，但尾波 68 个 ≥ 66，CUTLASS 选择不做——因为它判断「引入 131 次 fixup 不如忍下这 0.485 个 tile」。

最后一个观察，也是这节最重要的：

**CUTLASS 从来不做「全盘 stream-K」。** 看 `1536×3200` 那一行——300 个 tile，只有后 168 个交给 stream-K，前 132 个照旧走 data-parallel。因为前 132 个正好是一整波，没有任何损失，给它们做 stream-K 只会让每个 tile 都被切开、白付 132 次 fixup。

这是「只在有损失的地方付代价」的极端版本：**损失在最后两波，所以只需要处理最后两波。**

### 4.2 切在哪：`min_iters = 8` 与碎片推让

均分 k-iter 时，切点会落在 tile 内部。如果某个单元恰好分到「某个 tile 的最后 3 个 k-iter」，它要把 A/B 的 pipeline prologue 重新走一遍，就为算 3 个 k-iter——纯亏。

CUTLASS 的处理是把这个碎片推给邻居。下面这段复现了那个逻辑并验证结果：

```python
"""切点分配 + 碎片调整，验证「无重无漏」。

对应 CUTLASS：
  sm90_tile_scheduler_stream_k.hpp:763-899  get_current_work_iter_start_possible_...
    —— 均分，然后把小于 8 个 k-tile 的碎片推给邻居
  sm90_tile_scheduler_stream_k.hpp:901-978  update_output_tile_id_and_work_tile_k()
    —— 一个 unit 的区间切成「若干完整 tile + 首尾两个 partial」
"""
K_PER_TILE = 512   # 一个输出 tile 在 K 方向有几个 k-tile（K=8192, k-tile=16）
MIN_ITERS = 8      # tile_scheduler_params.h:470


def assign(sk_tiles, sk_units, k_per_tile=K_PER_TILE):
    """把 sk_tiles * k_per_tile 个 k-iter 均分给 sk_units 个 unit，再修碎片。"""
    total = sk_tiles * k_per_tile
    per, rem = divmod(total, sk_units)
    starts, counts = [], []
    for u in range(sk_units):
        starts.append(per * u + min(u, rem))      # 前 rem 个多拿一个（big_units）
        counts.append(per + (1 if u < rem else 0))

    # 起点落在 tile 的前 8 个或后 8 个 k-tile → 把这段碎片推走 / 从邻居接管
    for u in range(1, sk_units):
        k_in_tile = starts[u] % k_per_tile
        if k_in_tile < MIN_ITERS:
            starts[u] -= k_in_tile
            counts[u] += k_in_tile
        elif k_in_tile > k_per_tile - MIN_ITERS:
            shift = k_per_tile - k_in_tile
            starts[u] += shift
            counts[u] -= shift

    # 终点同理
    for u in range(sk_units):
        k_in_tile = (starts[u] + counts[u]) % k_per_tile
        if 0 < k_in_tile < MIN_ITERS:
            counts[u] -= k_in_tile
        elif k_in_tile > k_per_tile - MIN_ITERS:
            counts[u] += k_per_tile - k_in_tile
    return starts, counts


def coverage(starts, counts, k_per_tile=K_PER_TILE):
    """每个 k-iter 被几个 unit 覆盖；每个 tile 被几个 unit 碰过。"""
    total = max(s + c for s, c in zip(starts, counts))
    hits = [0] * total
    for s, c in zip(starts, counts):
        for i in range(s, s + c):
            hits[i] += 1
    per_tile = [0] * (total // k_per_tile)
    for i, h in enumerate(hits):
        if h:
            per_tile[i // k_per_tile] += 1
    return hits, per_tile


if __name__ == "__main__":
    for sk_tiles, sk_units in [(140, 132), (168, 132), (64, 132)]:
        starts, counts = assign(sk_tiles, sk_units)
        total = sk_tiles * K_PER_TILE
        hits, per_tile = coverage(starts, counts)
        contiguous = all(starts[i] + counts[i] == starts[i + 1] for i in range(len(starts) - 1))
        print(f"=== sk_tiles={sk_tiles}, sk_units={sk_units}, 总 k-iter={total} ===")
        print(f"  每 unit 分到 k-iter：min={min(counts)} max={max(counts)} 合计={sum(counts)}")
        print(f"  无重无漏：{sum(counts) == total and starts[0] == 0 and contiguous}"
              f"（每个 k-iter 恰好被 1 个 unit 覆盖：{all(h == 1 for h in hits)}）")
        cut = [i for i, n in enumerate(per_tile) if n > 1]
        print(f"  被切开的 tile：{len(cut)}/{len(per_tile)} 个（它们的累加器要跨 unit 归约）")
        n_span = [(s + c - 1) // K_PER_TILE - s // K_PER_TILE + 1 for s, c in zip(starts, counts)]
        print(f"  每个 unit 跨的 tile 数：{min(n_span)}~{max(n_span)}")
        print("  前 5 个 unit 的区间：")
        for u in range(min(5, sk_units)):
            s, c = starts[u], counts[u]
            print(f"    unit {u:>3}: k-iter [{s:>6},{s+c:>6})  跨 tile {s//K_PER_TILE}~"
                  f"{(s+c-1)//K_PER_TILE}  起点在 tile 内第 {s % K_PER_TILE} 个 k-tile")
        print()
```

```text
=== sk_tiles=140, sk_units=132, 总 k-iter=71680 ===
  每 unit 分到 k-iter：min=540 max=546 合计=71680
  无重无漏：True（每个 k-iter 恰好被 1 个 unit 覆盖：True）
  被切开的 tile：140/140 个（它们的累加器要跨 unit 归约）
  每个 unit 跨的 tile 数：2~3
  前 5 个 unit 的区间：
    unit   0: k-iter [     0,   544)  跨 tile 0~1  起点在 tile 内第 0 个 k-tile
    unit   1: k-iter [   544,  1088)  跨 tile 1~2  起点在 tile 内第 32 个 k-tile
    unit   2: k-iter [  1088,  1632)  跨 tile 2~3  起点在 tile 内第 64 个 k-tile
    unit   3: k-iter [  1632,  2176)  跨 tile 3~4  起点在 tile 内第 96 个 k-tile
    unit   4: k-iter [  2176,  2719)  跨 tile 4~5  起点在 tile 内第 128 个 k-tile

=== sk_tiles=168, sk_units=132, 总 k-iter=86016 ===
  每 unit 分到 k-iter：min=644 max=658 合计=86016
  无重无漏：True（每个 k-iter 恰好被 1 个 unit 覆盖：True）
  被切开的 tile：168/168 个（它们的累加器要跨 unit 归约）
  每个 unit 跨的 tile 数：2~3
  前 5 个 unit 的区间：
    unit   0: k-iter [     0,   652)  跨 tile 0~1  起点在 tile 内第 0 个 k-tile
    unit   1: k-iter [   652,  1304)  跨 tile 1~2  起点在 tile 内第 140 个 k-tile
    unit   2: k-iter [  1304,  1956)  跨 tile 2~3  起点在 tile 内第 280 个 k-tile
    unit   3: k-iter [  1956,  2608)  跨 tile 3~5  起点在 tile 内第 420 个 k-tile
    unit   4: k-iter [  2608,  3260)  跨 tile 5~6  起点在 tile 内第 48 个 k-tile

=== sk_tiles=64, sk_units=132, 总 k-iter=32768 ===
  每 unit 分到 k-iter：min=248 max=249 合计=32768
  无重无漏：True（每个 k-iter 恰好被 1 个 unit 覆盖：True）
  被切开的 tile：64/64 个（它们的累加器要跨 unit 归约）
  每个 unit 跨的 tile 数：1~2
  前 5 个 unit 的区间：
    unit   0: k-iter [     0,   249)  跨 tile 0~0  起点在 tile 内第 0 个 k-tile
    unit   1: k-iter [   249,   498)  跨 tile 0~0  起点在 tile 内第 249 个 k-tile
    unit   2: k-iter [   498,   747)  跨 tile 0~1  起点在 tile 内第 498 个 k-tile
    unit   3: k-iter [   747,   996)  跨 tile 1~1  起点在 tile 内第 235 个 k-tile
    unit   4: k-iter [   996,  1245)  跨 tile 1~2  起点在 tile 内第 484 个 k-tile

```

三个结论：

**一，无重无漏。** 每个 k-iter 恰好被 1 个 unit 覆盖，区间首尾相接。这是 stream-K 正确性的最低要求，也是最容易写错的地方（off-by-one 会导致某个 k-iter 被算两次或漏算）。

**二，工作量差 6 个 k-iter（1.1%）。** 140 个 tile / 132 个 unit，均分后是 540~546——碎片调整造成的差异只有 6 个 k-iter，相对 542 是 1.1%。

这才是 stream-K 真正的收益来源：**量化损失没有消失，只是从 tile 级（差一整个 tile = 100%）降到了 k-iter 级（差 1.1%）。**

**三，140 个 tile 全部被切开。** 这不是 bug——132 个 unit 分 140 个 tile，每个 unit 平均要跨 1.06 个 tile 边界，几乎每个 tile 都被穿过。这正是「cut 次数 = unit 数 $-1$」的实证：cut 有 131 次，而 tile 有 140 个。**fixup 的数量由机器规模决定，不由问题规模决定。**

顺带看一个细节：`sk_tiles=64, sk_units=132` 那一组，每个 unit 只拿 248~249 个 k-iter，不到一个 tile 的一半。这就是 2.2 节说的 $\lambda < 1$ 情形——**同一个代码路径在不同 $\lambda$ 下自动变成 split-K**，不需要单独的算法分支。CUTLASS 只有一个例外：当 `sk_units % sk_tiles == 0` 时它会显式切到 `DecompositionMode::SplitK`（`tsp.h:911`），因为这时每个 tile 的 peer 数正好相等，可以走更省的路径。

### 4.3 fixup 的账：千分之一到一个百分之一

现在算最关键的账。一次 fixup 要做的事是：写出一份 partial 累加器，再读回来。字节数是 $2M_tN_tb$。

一个 tile 的计算时间是 $2M_tN_tK/P_{sm}$。

两者相除，$M_tN_t$ 全部约掉：

$$\boxed{\text{一次 fixup} = \frac{b}{K}\cdot\frac{P_{sm}}{BW}\ \text{个 tile 的时间}}$$

$\frac{P_{sm}}{BW}$ 是「单 SM 算力 ÷ 整机带宽」，H100 上是 $7.50/3.35 = 2.24$ FLOP/B（它等于整机屋脊点 295.4 除以 132 个 SM）。

```python
"""把 fixup 成本与量化损失换算到同一单位：「几个 tile 的时间」。

一次 fixup = 2*Mt*Nt*b 字节，耗时 2*Mt*Nt*b/BW
一个 tile  = 2*Mt*Nt*K FLOP，耗时 2*Mt*Nt*K/P_sm
相除 → f = (b/K) * (P_sm/BW)，Mt、Nt 全部约掉。
"""
import math

SM = 132              # H100 SXM5
BW = 3.35e12          # B/s
P_TOTAL = 989.5e12    # FLOP/s，FP16 dense
P_SM = P_TOTAL / SM
ROOFLINE_SM = P_SM / BW

BM = BN = 128
BK = 16
K = 8192
b = 4                 # fp32 输出

per_fixup = b / K * ROOFLINE_SM
t_tile = 2 * BM * BN * K / P_SM

print(f"H100 SXM5（文档级）：{SM} SM，HBM {BW/1e12:.2f} TB/s，FP16 dense {P_TOTAL/1e12:.1f} TFLOPS")
print(f"  整机屋脊点 = {P_TOTAL/BW:.1f} FLOP/B   单 SM = {ROOFLINE_SM:.2f} FLOP/B")
print(f"  tile {BM}x{BN}，K={K}，fp32 输出：一个 tile {t_tile*1e6:.1f} us")
print(f"  一次 fixup 折合 {per_fixup:.4f} 个 tile 的时间\n")


def sk_tiles_of(tiles, kiter):
    """与 4.1 同一套 CUTLASS 启发式"""
    waves = math.ceil(tiles / SM)
    full = tiles // SM
    if full == waves or kiter <= 8:
        return 0, waves
    if 2 * (tiles - full * SM) >= SM:
        return 0, waves
    return tiles - ((full - 1) * SM if full > 1 else 0), waves


print(f"{'M':>6}{'N':>7}{'tiles':>7}{'W':>4}{'eta':>7}{'损失tile':>10}{'上限':>7}"
      f"{'sk':>5}{'fixup':>7}{'成本':>7}{'净赚':>7}{'占原时间':>9}")
print("-" * 82)
for M, N in [(1024, 1024), (1280, 1792), (1536, 3200), (2048, 2048),
             (4096, 4096), (8192, 8192), (16384, 16384)]:
    tiles = math.ceil(M / BM) * math.ceil(N / BN)
    kiter = K // BK
    sk, waves = sk_tiles_of(tiles, kiter)
    eta = tiles / (waves * SM)
    gain = waves * (1 - eta)          # 单位：tile 时间，也就是量化损失
    if sk == 0:
        print(f"{M:>6}{N:>7}{tiles:>7}{waves:>4}{eta:>7.3f}{gain:>10.3f}{1/eta:>6.2f}x"
              f"{0:>5}{0:>7}{0:>7}{0:>7}{'启发式放弃':>11}")
        continue
    units = min(SM, (kiter * sk) // 8)
    fixups = max(units - 1, 0)
    cost = fixups * per_fixup
    net = gain - cost
    print(f"{M:>6}{N:>7}{tiles:>7}{waves:>4}{eta:>7.3f}{gain:>10.3f}{1/eta:>6.2f}x"
          f"{sk:>5}{fixups:>7}{cost:>7.3f}{net:>7.3f}{net/waves*100:>8.1f}%")

print("\n量化损失的锯齿结构（P=132，K=8192，扫 tile 数）：")
print(f"{'tiles':>7}{'W':>4}{'eta':>7}{'损失':>8}{'上限':>7}{'sk':>5}{'净赚':>7}{'加速':>7}")
for tiles in [128, 132, 133, 140, 160, 200, 264, 265, 280, 300, 396, 400]:
    sk, waves = sk_tiles_of(tiles, 512)
    eta = tiles / (waves * SM)
    gain = waves * (1 - eta)
    if sk == 0:
        print(f"{tiles:>7}{waves:>4}{eta:>7.3f}{gain:>8.3f}{1/eta:>6.2f}x{0:>5}{0:>7}{1.0:>6.2f}x")
        continue
    units = min(SM, (512 * sk) // 8)
    net = gain - max(units - 1, 0) * per_fixup
    print(f"{tiles:>7}{waves:>4}{eta:>7.3f}{gain:>8.3f}{1/eta:>6.2f}x{sk:>5}{net:>7.3f}"
          f"{waves/(waves-net):>6.2f}x")
```

```text
H100 SXM5（文档级）：132 SM，HBM 3.35 TB/s，FP16 dense 989.5 TFLOPS
  整机屋脊点 = 295.4 FLOP/B   单 SM = 2.24 FLOP/B
  tile 128x128，K=8192，fp32 输出：一个 tile 35.8 us
  一次 fixup 折合 0.0011 个 tile 的时间

     M      N  tiles   W    eta    损失tile     上限   sk  fixup     成本     净赚     占原时间
----------------------------------------------------------------------------------
  1024   1024     64   1  0.485     0.515  2.06x   64    131  0.143  0.372    37.2%
  1280   1792    140   2  0.530     0.939  1.89x  140    131  0.143  0.796    39.8%
  1536   3200    300   3  0.758     0.727  1.32x  168    131  0.143  0.584    19.5%
  2048   2048    256   2  0.970     0.061  1.03x    0      0      0      0      启发式放弃
  4096   4096   1024   8  0.970     0.242  1.03x    0      0      0      0      启发式放弃
  8192   8192   4096  32  0.970     0.970  1.03x  136    131  0.143  0.827     2.6%
 16384  16384  16384 125  0.993     0.879  1.01x  148    131  0.143  0.736     0.6%

量化损失的锯齿结构（P=132，K=8192，扫 tile 数）：
  tiles   W    eta      损失     上限   sk     净赚     加速
    128   1  0.970   0.030  1.03x    0      0  1.00x
    132   1  1.000   0.000  1.00x    0      0  1.00x
    133   2  0.504   0.992  1.98x  133  0.849  1.74x
    140   2  0.530   0.939  1.89x  140  0.796  1.66x
    160   2  0.606   0.788  1.65x  160  0.645  1.48x
    200   2  0.758   0.485  1.32x    0      0  1.00x
    264   2  1.000   0.000  1.00x    0      0  1.00x
    265   3  0.669   0.992  1.49x  133  0.849  1.39x
    280   3  0.707   0.879  1.41x  148  0.736  1.32x
    300   3  0.758   0.727  1.32x  168  0.584  1.24x
    396   3  1.000   0.000  1.00x    0      0  1.00x
    400   4  0.758   0.970  1.32x  136  0.827  1.26x
```

**一次 fixup = 0.0011 个 tile。** 也就是千分之一。131 次 fixup 加起来是 0.143 个 tile 的时间，而 $1280\times1792$ 的量化损失是 0.939 个 tile——**兑现 85%**。

表里最该盯的是这两行：

```
  1280  1792    140   2  0.530     0.939  1.89x  140    131  0.143  0.796    39.8%
  8192  8192   4096  32  0.970     0.970  1.03x  136    131  0.143  0.827     2.6%
```

**同一个技术，同样的绝对收益（0.796 vs 0.827 个 tile，都是约 30 μs），相对收益差 15 倍。** 因为 $8192^2$ 的 32 波里，这 0.8 个 tile 只占 2.6%。

这也是为什么上一节 CUTLASS 的启发式在这些形状上会直接放弃：$4096^2$ 和 $2048^2$ 的损失只有 0.24 / 0.06 个 tile，还不够付 fixup 的 0.143。

**一句话：stream-K 的绝对收益上限是一个 tile 的时间，与问题规模无关。** 所以它天生是「小问题、窄形状」的技术——问题越大，这一个 tile 越不重要。

顺便，fixup 的成本随 $K$ 下降（$f \propto 1/K$）：$K=1024$ 时是 0.0087 个 tile，$K=16384$ 时是 0.0005。因为 K 越长，一个 tile 的活越多，写一次累加器就显得越便宜。

还有一处设计细节值得注意：$1280\times1792$ 那档的 workspace 是 $131\times128\times128\times 4\text{B} \approx 8.2$ MiB（$8192^2$ 那档 $sk\_tiles=136$，8.5 MiB）。H100 的 L2 是 50 MB，**整个 workspace 装得进去**。这不是巧合——CUTLASS 的 `get_reduction_workspace_size` 结尾有一个 `round_up_to_l2_alignment`，它保证 partial 的读写走 L2 而不是穿到 HBM。如果 workspace 超出 L2，上面的账要按 HBM 带宽重算，成本会翻几倍。

### 4.4 确定性归约的代价

fixup 是把多个 partial 加起来。加法的顺序会改变 fp32 结果，所以 CUTLASS 给了两个模式：

- `ReductionMode::Deterministic`（**默认**）：第 $i$ 个 peer 等前一个写完再加，形成一条串行链，结果 bitwise 可复现
- `ReductionMode::NonDeterministic`：谁先到谁先加（atomicAdd），快，但每次结果可能不同

先量一下「顺序到底影响多少」。误差归一化到 $|\text{err}|/(\epsilon\sum|x|)$（049 期踩过的坑：用相对误差会在求和接近 0 时炸成噪声）：

```python
"""切点变了，fp32 累加顺序就变了 —— 影响有多大？

误差归一化到 |err| / (eps * sum|x|)：标准浮点累加误差界，
分母与数据量级和正负号都无关，跨 seed、跨切法可比。
"""
import torch

torch.manual_seed(0)
M = N = 256
K = 8192
eps = torch.finfo(torch.float32).eps

A = torch.randn(M, K)
B = torch.randn(K, N)
ref = A @ B


def seg_sum(parts):
    acc = torch.zeros(M, N)
    for p in parts:
        acc = acc + p
    return acc


def norm_err(x):
    return ((x - ref).abs() / (eps * (A.abs() @ B.abs()))).max().item()


print("按 stream-K 切法把 K 切成 s 段，再顺序累加：")
print(f"{'s':>4} {'max|err|':>12} {'归一化':>10}")
for s in [1, 2, 3, 4, 8, 16]:
    # 切点必须覆盖到 K 末尾：8192/3 除不尽，用 K//s 会丢掉尾部 2 列
    bounds = [i * K // s for i in range(s + 1)]
    parts = [A[:, bounds[i]:bounds[i + 1]] @ B[bounds[i]:bounds[i + 1], :] for i in range(s)]
    c = seg_sum(parts)
    print(f"{s:>4} {(c - ref).abs().max().item():>12.3e} {norm_err(c):>10.3e}")

print("\n同一组 3 个 partial，只改相加顺序（模拟 NonDeterministic 的到达顺序）：")
b3 = [i * K // 3 for i in range(4)]
parts = [A[:, b3[i]:b3[i + 1]] @ B[b3[i]:b3[i + 1], :] for i in range(3)]
fixed = seg_sum(parts)
print(f"  固定顺序 0,1,2        归一化 {norm_err(fixed):.3e}")
g = torch.Generator().manual_seed(7)
errs, diffs = [], []
for _ in range(20):
    perm = torch.randperm(3, generator=g).tolist()
    c = seg_sum([parts[i] for i in perm])
    errs.append(norm_err(c))
    diffs.append((c - fixed).abs().max().item())
print(f"  随机 20 种到达顺序    归一化 min={min(errs):.3e} max={max(errs):.3e}")
print(f"  不同顺序的结果与固定顺序的最大差 max|Δ| = {max(diffs):.3e}")
print(f"  参考值量级 max|ref| = {ref.abs().max().item():.3f}")

print("\n每 tile 切成 2 份，但切点位置不同（peer 数恒为 2）：")
print(f"{'切点':>8} {'归一化':>10}")
for cut in [64, 1024, 2048, 4096, 7168]:
    c = seg_sum([A[:, :cut] @ B[:cut, :], A[:, cut:] @ B[cut:, :]])
    print(f"{cut:>8} {norm_err(c):>10.3e}")
```

```text
按 stream-K 切法把 K 切成 s 段，再顺序累加：
   s     max|err|        归一化
   1    0.000e+00  0.000e+00
   2    1.129e-03  1.810e+00
   3    1.221e-03  1.958e+00
   4    1.404e-03  2.252e+00
   8    1.343e-03  2.154e+00
  16    1.221e-03  1.958e+00

同一组 3 个 partial，只改相加顺序（模拟 NonDeterministic 的到达顺序）：
  固定顺序 0,1,2        归一化 1.958e+00
  随机 20 种到达顺序    归一化 min=1.909e+00 max=1.958e+00
  不同顺序的结果与固定顺序的最大差 max|Δ| = 3.052e-05
  参考值量级 max|ref| = 445.243

每 tile 切成 2 份，但切点位置不同（peer 数恒为 2）：
      切点        归一化
      64  1.444e+00
    1024  2.399e+00
    2048  2.448e+00
    4096  1.810e+00
    7168  1.128e+00
```

三个结果：

**一，段数几乎不影响精度。** $s$ 从 2 到 16，归一化误差在 1.8~2.3 之间随机摆动，**没有趋势**。这和 049 期的结论一致（split-K 的份数不是精度变量），可以放心。

**二，相加顺序的影响是 2.5%**，绝对差 $3\times10^{-5}$（相对参考值 445 是 $7\times10^{-8}$）。这个量级意味着：**NonDeterministic 模式在数值上完全可用，代价只是「不可复现」而不是「不准」。**

**三，切点位置的影响（1.13~2.45，跨 2.2 倍）比份数的影响更大。** 反直觉但合理：切点决定了每一段内部连续加法的长度。切在 64 时前一段只有 64 项、后一段 8128 项，误差分布和对称切分完全不同。

这条的实际含义：**换一个 $\lambda$（哪怕 $\lambda$ 只从 1.06 变到 1.5）就会改变所有切点，从而改变 fp32 结果的最后几位。** 如果你的流水线里有「同输入必须同输出」的断言（比如回归测试），stream-K 的形状敏感性会让你很痛苦。

至于 Deterministic 模式的串行链有多长——`max_peers_per_tile(sk_units, sk_tiles)`（`tsp.h:531`）的定义是 `sk_units / sk_tiles + 2`：

- $1280\times1792$：$132/140 + 2 = 2$ → 串行链 2 跳
- $1024\times1024$：$132/64 + 2 = 4$ → 串行链 4 跳

**链长由「每个 tile 有几个 peer」决定，不是由 unit 数决定。** 所以「确定性归约很贵」的担心在这里不成立——链只有 2~4 跳。

（那个 `+2` 是给余数留的：一个 tile 的 K 区间可能被「前一个 unit 的尾巴」和「后一个 unit 的头」各切一刀，所以最多比均分值多 2 个 peer。）

### 4.5 它和已发各期的关系

这一节把所有线头接起来。

**与 049 期（split-K）：本期是它的连续化。** 049 用 $S^* = \sqrt{2K\cdot BW/(P_{sm}b)}$ 回答「切几份最划算」，前提是份数必须是整数、且每份是一个完整的 k-iter 段。stream-K 取消了这个前提，代价从「归约 $S$ 份 partial」变成「$P-1$ 次 fixup」。049 的结论「$S^*$ 与 tile 尺寸无关」在这里有个漂亮的对应：本节 4.3 的 $f = (b/K)(P_{sm}/BW)$ 同样不含 $M_tN_t$——**同一个约掉 $M_tN_t$ 的结构出现了两次，因为它俩都是「归约字节 ÷ 计算 FLOP」**。

**与 048 期（epilogue）：fixup 是 epilogue 的第五种出口。** 048 讲了四种（直接写、过 SMEM、`stmatrix`、TMA store），全都假设「累加器是最终的」。stream-K 打破了这个假设：累加器可能只是 partial，要先写 workspace、等 peer、再归约、最后才跑 epilogue。048 那个「$K^*$ = 屋脊点 × $b/2$」的判据（多大 K 之后写回成为瓶颈）在这里变成了另一种形式：$f = (b/K)(P_{sm}/BW)$。两者其实是同一个量的两种归一化。

**与 047 期（wgmma）：$P_{sm}$ 就是 wgmma 的吞吐。** 047 算出「一条 wgmma m64n256k16 占 138.5 cycle」，乘以每秒的指令数就是 $P_{sm}$。本期整个成本模型的分母都是它。

**与 046 期（cluster）：stream-K 的 unit 按 cluster 分组。** `sk_tiles = (sk_tiles / cluster_size) * cluster_size`（`tsp.h:1112`）——tile 数必须向下取整到 cluster 的倍数，否则最后一个 cluster 会分到不完整的工作。而且同 cluster 内的 unit 共享 K offset（`sm90_tile_scheduler_stream_k.hpp:780-790`），这样 TMA multicast 才能命中。

**与 045 期（occupancy）：$P$ 不是 SM 数。** 本期的 $P$ 一直是 `ctas_per_wave`——如果 occupancy 允许一个 SM 驻留 2 个 CTA，$P$ 就是 $2\times$ SM 数，$\lambda$ 直接减半。CUTLASS 里 `ctas_per_wave = grid.x * grid.y`（`tsp.h:854`），是真实算出来的，不是写死的 SM 数。**这是最容易算错的一个参数。**

**与 043 期（屋脊点）：$P_{sm}/BW$ 就是「单个 SM 的屋脊点」。** 整机屋脊点 295.4 / 132 = 2.24。本期所有成本公式里那个 2.24 就是它。

### 4.6 为什么只在最后两波启用

4.1 的表里已经看到：CUTLASS 只把最后两波交给 stream-K，前面完整的波照旧 data-parallel。这个设计有三重好处：

1. **fixup 次数从 $T$ 降到 $P$**。全盘 stream-K 时每个 tile 都被切开，fixup 次数 = tile 数；只切最后两波时，fixup 次数 = unit 数。
2. **前面几波没有同步、没有 workspace**。纯粹的 data-parallel 速度。
3. **stream-K 的 unit 排在 work 队列前面**（`sm90_tile_scheduler_stream_k.hpp:1027` 那个分支：`linear_idx >= sk_units_` 才走 data-parallel）。所以核函数一开始就跑 stream-K，fixup 的同步发生在**早期**，不会拖在最后。

第 3 点值得单独说。如果 stream-K 排在最后，所有 CTA 会在收尾时挤在一起等 barrier；排在最前面，等 barrier 的时候后面还有大量 DP tile 可以填——**把等待藏进后面的活里**。

配合 4.2 节那个「反向遍历 tile」的细节，整体是这样一个设计：一个 stream-K unit 从自己区间的**末尾**往**开头**走。这样「它的起点 partial」——也就是**上一个 unit 的终点 partial**——最后才算。而上一个 unit 恰恰需要等这个 partial 才能做归约，所以「先算不需要等人的部分，后算需要等人的部分」。

---

## 五、什么时候该用 / 不该用

### 一个可以直接抄的三步判据

1. 算 $\lambda = T/P$（$P$ 用 `ctas_per_wave`，不是 SM 数）和 $\eta = \lambda/\lceil\lambda\rceil$
2. 量化损失 $= \lceil\lambda\rceil(1-\eta)$ 个 tile 的时间
3. fixup 成本 $= (n_{fix}\cdot b/K)\cdot(P_{sm}/BW)$ 个 tile 的时间，$n_{fix}\approx P-1$

**第 2 项 > 1.5 倍第 3 项才值得。**

H100 + K=8192 + fp32 输出的常用值：一次 fixup = 0.0011 tile，131 次 = 0.143 tile。所以门槛大约是「损失 > 0.21 个 tile」，即 $\lceil\lambda\rceil(1-\eta) > 0.21$。

### 该用

- **窄形状**：$M\times N$ 只够切出 1~2 波 tile。LLM decode（$M=1$）、LoRA 的 $B\cdot A$、单样本推理、小 batch 的 attention QK^T
- **$\lambda$ 略大于整数**：$T = 133, 265, 400$ 这种「刚过一整波」的形状，$\eta$ 掉到 0.5~0.67，收益 1.3~1.9 倍
- **延迟敏感的推理**：$1280\times1792$ 上 49% 的加速是真实的、可兑现的
- **tile 大（128×256 以上）**：tile 越大，$t_{tile}$ 越长，量化损失的绝对值越大

### 不该用

- **$T \gg P$ 且尾波半满**：$4096^2$/$8192^2$ 上只值 2~3%，不如把精力放在 mainloop 上
- **$K$ 很短**（$K/BK \le 8$）：闸 2，切了每份连 prologue 都填不满
- **要求 bitwise 可复现且形状会变**：4.4 节说过，切点一变结果就变
- **训练的正向大 GEMM**：本来就有几千个 tile，$\eta$ 已经 0.97 以上
- **已经开了 split-K 的形状**（$\lambda < 0.5$）：两者会打架，这时该用 split-K 的 $S^*$ 而不是 stream-K

---

## 六、常见坑

**1. 用 $\eta$ 当判据，而不是尾波的绝对大小。**

$2048^2$ 和 $8192^2$ 的 $\eta$ 都是 0.970，前者尾波 124 个（94% 满，不该做），后者尾波 4 个（3% 满，该做）。用 `eta < 0.9` 这种阈值会把两个都判错。判据必须是 `2 * tail < ctas_per_wave`。

**2. 以为 fixup 的成本是「归约的 FLOP」。**

131 个 tile 的 partial 加起来是 $131\times128\times128\approx 2.1$M 次加法，按 989 TFLOPS 算是 2 纳秒——完全可以忽略。真正的成本是**字节往返**（17 MB）和**同步**。算账时要算字节，不要算 FLOP。

**3. 忘了 workspace 的 L2 对齐。**

CUTLASS 的 `get_reduction_workspace_size` 结尾是 `round_up_to_l2_alignment`。不对齐时，partial 的读写会跨 cacheline、穿到 HBM，4.3 节那笔 0.0011 tile 的账会翻两三倍。自己实现时很容易漏掉这一行。

**4. 在 $T = kP$ 上找收益。**

$\lambda$ 正好是整数时 $\eta = 1$，量化损失为 0，stream-K 一点收益都没有，却仍然付 $P-1$ 次 fixup。CUTLASS 的闸 1 拦住了这种情况；自己写循环时容易忘。

**5. 把 $P$ 当成 SM 数。**

$P$ 是 `ctas_per_wave`。occupancy 允许一个 SM 驻留 2 个 CTA 时（比如 tile 小、寄存器压力低），$P = 2\times$ SM 数，$\lambda$ 直接减半——本来该用 stream-K 的形状会变成该用 split-K。这个参数在 CUTLASS 里是 `grid.x * grid.y`，是算出来的。

**6. 用 `atomicAdd` 做归约却要求结果可复现。**

4.4 节测出来顺序变化的绝对差是 $3\times10^{-5}$，数值上完全可接受。但如果你的 CI 里有一条「同输入输出必须逐位相同」的断言，NonDeterministic 会让它随机失败。要么用 Deterministic，要么把断言放宽到 `allclose`。

---

## 七、一句话总结

**stream-K 不是「更聪明的 split-K」，而是把调度单位从 tile 降级到 k-iter 之后，split-K 和 persistent 自动成为同一个算法的两个端点（$\lambda<1$ 和 $\lambda\gg1$）。它的全部价值是抢回「不到一个 tile」的量化损失——那部分损失有天花板，所以绝对收益与问题规模无关；它的全部代价是 $P-1$ 次折合千分之一 tile 的 fixup。于是结论很干脆：窄形状上值 49%，$8192^2$ 上值 2.6%，判据就三步算术。**

---

## 八、今日练习

<details><summary>今日练习（点开看题）</summary>

**题 1**：H100（132 SM，HBM 3.35 TB/s，FP16 dense 989.5 TFLOPS），$M=2048, N=7168, K=4096$，tile $128\times256$，fp16 输入 / fp32 输出。

(a) 算 $T$、$\lambda$、$\eta$、量化损失（以 tile 时间为单位）。
(b) CUTLASS 的启发式会做 stream-K 吗？被哪条闸拦？
(c) 如果强行做，净收益是多少？

**题 2**：同一个 $M,N$，把 tile 换成 $128\times128$，答案会变吗？为什么？

**题 3**：为什么「一次 fixup 折合 0.0011 个 tile」这个数字与 $M_t$、$N_t$ 无关？用两句话解释。

</details>

<details><summary>参考答案</summary>

**题 1**

(a) $T = \lceil2048/128\rceil \times \lceil7168/256\rceil = 16\times28 = 448$。
$\lambda = 448/132 = 3.394$，$\lceil\lambda\rceil = 4$，$\eta = 448/528 = 0.848$。
量化损失 $= 4\times(1-0.848) = 0.606$ 个 tile。

(b) `full_waves = 448//132 = 3`，`total_waves = 4`，有量化损失 → 过闸 1。
`kiter = 4096/16 = 256 > 8` → 过闸 2。
`tail = 448 - 3\times132 = 52`，`2\times52 = 104 < 132` → **过闸 3**。
所以会做。`dp_tiles = (3-1)\times132 = 264`，`sk_tiles = 448-264 = 184`。

(c) fixup：`sk_units = min(132, 256\times184/8) = 132`，$n_{fix} = 131$。
一次 fixup $= (4/4096)\times2.24 = 0.00219$ 个 tile。
成本 $= 131\times0.00219 = 0.287$ 个 tile。
净收益 $= 0.606-0.287 = 0.319$ 个 tile，占原时间 $0.319/4 = 8.0\%$ → 加速约 **1.087x**。

注意 $K=4096$ 时 fixup 单价是 $K=8192$ 的两倍（$f\propto1/K$），所以兑现率只有 47%（0.319/0.606），比 4.3 节那个 85% 差很多。**K 越短，stream-K 越不划算。**

**题 2**

换成 $128\times128$ 后 $T = 16\times56 = 896$，$\lambda = 6.79$，$\lceil\lambda\rceil=7$，$\eta = 896/924 = 0.970$。
`tail = 896 - 6\times132 = 104`，`2\times104 = 208 \ge 132` → **被闸 3 拦住，不做**。

同一个 $M,N,K$，tile 尺寸一改，结论从「做」变成「不做」。原因是 tile 变小 → tile 数变多 → 尾波的绝对数量也变多（52 → 104）→ 超过半波。而且 tile 变小时 $t_{tile}$ 也变短，量化损失的绝对值缩小。**tile 尺寸是 stream-K 判据里最敏感的参数。**

**题 3**

因为 fixup 的字节数（$2M_tN_tb$）和一个 tile 的计算量（$2M_tN_tK$ FLOP）**含同一个因子 $M_tN_t$**，相除时直接约掉。

物理含义：一个 tile 越大，写它的累加器越贵，但算它也越久，两者同比例放大，比值只由「每 FLOP 要搬多少字节」决定，也就是 $b/K$（输出位宽 ÷ K 方向的计算深度）。

</details>
