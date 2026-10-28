# PyTorch 每日一课 · 第 044 期

## 异步拷贝与软件流水线：cp.async / TMA / warp specialization

| 项目 | 内容 |
|---|---|
| 日期 | 2026-10-28 |
| 难度 | ⭐⭐⭐⭐ |
| 前置知识 | 知道 shared memory 是什么；读过第 043 期（访存层次）会更快，但不依赖 |
| 预计阅读 | 30 分钟 |
| 关联 | 第 043 期 · 043 算的是「每次访存买多少字节」，本期算「等这次访存的时间怎么填满」 |

---

## 一、这个领域解决什么问题

第 043 期我们算过一笔账：H100 每个 SM 分到的 HBM 带宽约 25 GB/s，而 shared memory 能给它 253 GB/s，差 10 倍。结论很容易被记成「HBM 慢」。

但这句话漏了一半。HBM 的问题不是慢，是**慢得不一样**——它的**带宽**和**延迟**是两个独立的天花板，而绝大多数 kernel 卡在后者。

先在本机把这两个数字分开量出来。用一个经典的 pointer chasing：把一堆 128 B 的槽位打乱串成单向环，沿着环一跳一跳走。因为下一跳的地址依赖上一跳的结果，CPU 必须等数据真的回来才能继续——这时候测到的就是**真实延迟**，没有任何重叠可言。

```python
import os
import sys
from torch.utils.cpp_extension import load_inline

# 本机 ninja 装在解释器旁边的 bin 下、但不在 PATH 上，cpp_extension 找不到它
os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ["PATH"]

cpp = r"""
#include <torch/extension.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>

static constexpr int64_t SLOT = 128;   // 本机 cacheline = 128 B，一槽正好一行
static volatile int64_t g_sink = 0;

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// 把 [begin, begin+seg) 这些槽打乱后串成一个单向环，返回每槽的下一跳
static void fill_ring(int64_t* nxt, int64_t begin, int64_t seg, std::mt19937_64& rng) {
    std::vector<int64_t> idx(seg);
    for (int64_t i = 0; i < seg; i++) idx[i] = begin + i;
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int64_t i = 0; i < seg; i++) nxt[idx[i]] = idx[(i + 1) % seg];
}

// ---- 实验一：单链延迟阶梯 ----
// 单链 = 每次访问都必须等上一次返回，测到的就是该层存储的真实延迟
static std::vector<double> ladder(int64_t n_slots, int64_t total_steps, int reps, uint64_t seed) {
    std::vector<char> buf((size_t)n_slots * SLOT, 1);
    std::mt19937_64 rng(seed);
    std::vector<int64_t> idx(n_slots);
    for (int64_t i = 0; i < n_slots; i++) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int64_t i = 0; i < n_slots; i++)
        *reinterpret_cast<int64_t*>(buf.data() + idx[i] * SLOT) = idx[(i + 1) % n_slots];

    double best = 1e30;
    for (int rep = 0; rep < reps; rep++) {
        const char* p = buf.data() + idx[0] * SLOT;
        double t0 = now_s();
        for (int64_t i = 0; i < total_steps; i++) {
            int64_t k = *reinterpret_cast<const int64_t*>(p);
            p = buf.data() + k * SLOT;
        }
        double dt = now_s() - t0;
        g_sink = p - buf.data();
        if (dt < best) best = dt;
    }
    return {best, (double)total_steps};
}

// ---- 实验二：k 条独立链交错 ----
template <int K>
static void chase(char* base, int64_t steps, int64_t seg) {
    const char* p[K];
    for (int j = 0; j < K; j++) p[j] = base + (int64_t)(j) * seg * SLOT;
    for (int64_t i = 0; i < steps; i++) {
#pragma unroll
        for (int j = 0; j < K; j++) {
            int64_t k = *reinterpret_cast<const int64_t*>(p[j]);
            p[j] = base + k * SLOT;
        }
    }
    g_sink = p[0] - base;
}

static std::vector<double> mlp(int64_t n_slots, int ks, int reps, uint64_t seed) {
    std::vector<int64_t> nxt(n_slots);
    std::mt19937_64 rng(seed);
    int64_t seg = n_slots / ks;
    for (int64_t r = 0; r < ks; r++) fill_ring(nxt.data(), r * seg, seg, rng);
    std::vector<char> buf((size_t)n_slots * SLOT, 1);
    for (int64_t i = 0; i < n_slots; i++)
        *reinterpret_cast<int64_t*>(buf.data() + i * SLOT) = nxt[i];
    int64_t steps = seg;
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        switch (ks) {
            case 1: chase<1>(buf.data(), steps, seg); break;
            case 2: chase<2>(buf.data(), steps, seg); break;
            case 4: chase<4>(buf.data(), steps, seg); break;
            case 8: chase<8>(buf.data(), steps, seg); break;
            case 16: chase<16>(buf.data(), steps, seg); break;
            default: chase<32>(buf.data(), steps, seg); break;
        }
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return {(double)ks, (double)steps, best};
}

// ---- 顺序读基线：8 路累加，避免被加法依赖链卡住 ----
static double stream(int64_t n_slots, int reps) {
    std::vector<char> buf((size_t)n_slots * SLOT, 1);
    const int64_t* q = reinterpret_cast<const int64_t*>(buf.data());
    int64_t n = n_slots * (SLOT / 8);
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        int64_t a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
        double t0 = now_s();
        for (int64_t i = 0; i + 8 <= n; i += 8) {
            a0 += q[i];   a1 += q[i+1]; a2 += q[i+2]; a3 += q[i+3];
            a4 += q[i+4]; a5 += q[i+5]; a6 += q[i+6]; a7 += q[i+7];
        }
        double dt = now_s() - t0;
        g_sink = a0+a1+a2+a3+a4+a5+a6+a7;
        if (dt < best) best = dt;
    }
    return best;
}
"""

mod = load_inline(name="lat044", cpp_sources=cpp,
                  functions=["ladder", "mlp", "stream"],
                  extra_cflags=["-O3"], verbose=False)

TOTAL = 1 << 22
print("== 实验一：单链随机访问的延迟阶梯（每个尺寸都走满 4M 步）==")
print(f"{'working set':>12} {'min 耗时':>10} {'单次访问延迟':>13}")
for kb in (8, 64, 256, 1024, 2048, 4096, 8192, 32768, 131072, 524288):
    slots = kb * 1024 // 128
    t, steps = mod.ladder(slots, TOTAL, 3, 424242)
    label = f"{kb}KB" if kb < 1024 else f"{kb//1024}MB"
    print(f"{label:>12} {t*1e3:>9.2f}ms {t/steps*1e9:>12.2f}ns")

print("\n== 实验二：k 条独立链交错（工作集固定 1 GiB）==")
NSLOT = 1 << 23
print(f"{'并行链':>6} {'每链步数':>10} {'总 miss':>10} {'min 耗时':>11} "
      f"{'有效延迟':>9} {'吞吐':>10}")
for k in (1, 2, 4, 8, 16, 32):
    kk, steps, t = mod.mlp(NSLOT, k, 3, 20261028)
    total = kk * steps
    print(f"{int(kk):>6} {int(steps):>10} {int(total):>10} {t*1e3:>10.1f}ms "
          f"{t/total*1e9:>8.2f}ns {total*128/t/1e9:>7.1f}GB/s")

ts = mod.stream(NSLOT, 3)
print(f"\n顺序读基线（同 1 GiB，8 路累加）：{ts*1e3:.1f} ms → {NSLOT*128/ts/1e9:.1f} GB/s")
```

```text
== 实验一：单链随机访问的延迟阶梯（每个尺寸都走满 4M 步）==
 working set     min 耗时        单次访问延迟
         8KB      5.66ms         1.35ns
        64KB      5.64ms         1.35ns
       256KB     13.13ms         3.13ns
         1MB     16.04ms         3.82ns
         2MB     29.47ms         7.03ns
         4MB     29.52ms         7.04ns
         8MB     32.52ms         7.75ns
        32MB    249.34ms        59.45ns
       128MB    358.53ms        85.48ns
       512MB    367.59ms        87.64ns

== 实验二：k 条独立链交错（工作集固定 1 GiB）==
   并行链       每链步数     总 miss      min 耗时      有效延迟         吞吐
     1    8388608    8388608      745.4ms    88.86ns     1.4GB/s
     2    4194304    8388608      368.7ms    43.95ns     2.9GB/s
     4    2097152    8388608      182.2ms    21.72ns     5.9GB/s
     8    1048576    8388608       89.8ms    10.71ns    12.0GB/s
    16     524288    8388608       43.4ms     5.18ns    24.7GB/s
    32     262144    8388608       16.1ms     1.92ns    66.6GB/s

顺序读基线（同 1 GiB，8 路累加）：14.1 ms → 76.3 GB/s
```

> 本机是 Apple M5（4 性能核 + 6 能效核，cacheline 128 B），没有 NVIDIA GPU。这一节所有数字都是**本机真实测量**，不是引用别人的表格；涉及 H100 / B200 的部分我会单独标明是「官方参数」还是「按参数推算」。

第一张表是缓存阶梯：8 KB 到 64 KB 都是 1.35 ns（L1），1 MB 掉到 3.82 ns，8 MB 时 7.75 ns，到 32 MB 猛跳到 59.45 ns，512 MB 稳定在 87.64 ns（DRAM）。最慢和最快差 **64.9 倍**。

第二张表才是关键。同样的总访问次数（8.4M 次随机 miss），只是把「1 条依赖链」换成「k 条互不依赖的链交错推进」，有效延迟从 88.86 ns 一路降到 1.92 ns——**降了 46 倍**。

为什么？因为 1 条链时，同一时刻只有 1 个 miss 在飞，其余时间全在等；32 条链时，32 个 miss 同时在飞，等第一个回来的时候第 33 个已经发出去了。总耗时几乎只剩 `miss 数 × 延迟 / 并发度`。

这就是 **MLP（memory-level parallelism，访存级并行）**。GPU 上所有那些听起来很玄的机制——occupancy、多 warp、cp.async、多级缓冲——做的都是同一件事：**让足够多的请求同时在飞**。

但这条路有尽头，而且尽头来得比想象中早。16 条链时有效延迟 5.18 ns，按「延迟 ÷ 并发度」预测是 88.86 ÷ 16 = 5.55 ns，基本吻合；到 32 条链，预测是 2.78 ns，实测 1.92 ns——**比预测还快**，说明硬件自动做的重叠已经超过了我们显式给的 32 路。

真正该看的是吞吐那一列：32 条链时 66.6 GB/s，已经是同一台机器单线程顺序读（76.3 GB/s）的 87%。换个算法更直接——76.3 GB/s 意味着一条 128 B 的 cache line 摊到 1.68 ns，而实测每次 miss 用了 1.92 ns，只多出 14%。

**剩下的空间就这 14%。** 延迟侧已经被榨干，再往上加并发度，能买到的东西封顶就是这 14%。

这条分界线决定了后面所有内容的基调：**异步机制买的是延迟，不是带宽。** 它能让你在 1.4 GB/s 的窘境里爬出来，爬到接近带宽上限；但它一分钱带宽也变不出来。

---

## 二、核心思想：把「等」从关键路径上摘出去

用一个类比。厨房里两个动作：备菜（从冰箱拿、洗、切）和炒菜。如果只有一个厨师，流程必然是「拿菜 → 切 → 下锅 → 再拿菜 → 切 → 下锅」，每道菜都要等切完。这是**单链**。

正确的厨房是这样的：一个帮厨专门备菜，把切好的菜放在台面上（buffer）；厨师只从台面取，取到就炒。帮厨切第 3 道菜的时候，厨师在炒第 2 道。台面上要放得下几盘菜，这个「几盘」就是**流水线级数**（stage depth）。

三个必要条件，缺一不可：

1. **一块缓冲**：放得下还没被消费的成品（shared memory 里的 tile）
2. **一个能并行搬东西的单元**：帮厨（GPU 上是 cp.async 的拷贝引擎 / TMA 引擎，CPU 上就是另一个线程）
3. **一个交接协议**：帮厨怎么告诉厨师「这盘好了」，厨师怎么告诉帮厨「这盘空了」（GPU 上是 `mbarrier`，CPU 上是条件变量）

现在把这件事翻译成数字。设搬运一个 tile 要 $c$，计算一个 tile 要 $p$，共 $N$ 个 tile。

完全不重叠（同步）：

$$
T_{\text{sync}} = N(c + p)
$$

完美重叠（流水线填满后，搬和算同时进行，只有第一个 tile 的搬运露在关键路径上）：

$$
T_{\text{pipe}} \approx c + N \cdot \max(c, p)
$$

所以加速比的上限是：

$$
S_{\max} = \frac{N(c+p)}{c + N\max(c,p)} \xrightarrow{N \to \infty} \frac{c+p}{\max(c,p)}
$$

这个式子值得盯三秒钟。它说：

- $c \ll p$（计算主导，比如卷积）：$S_{\max} \to 1$。**流水线救不了你**，因为搬运本来就不在关键路径上。
- $c = p$（搬算相当，比如访存密集的 GEMM）：$S_{\max} \to 2$。这是流水的甜点区。
- $c \gg p$（搬运主导，比如逐元素算子）：$S_{\max} \to c/p$。收益很大，但这时你真正该做的是减少搬运量（第 040 期的算子融合），而不是加流水线。

记住这个式子，后面所有实测都要拿它当尺子。

---

## 三、NVIDIA 的三代机制

> 本节是**文档/源码级事实**（CUDA C++ Programming Guide、PTX ISA、CUTLASS），本机无 CUDA，无法实测。凡是 NVIDIA 特有的机制我会标 🔶，跨硬件通用的抽象标 ⚪。

### 3.1 第一代：`LDG` → 寄存器 → `STS`

最原始的做法：用普通的 global load 把数据读进寄存器，再写进 shared memory（`__shared__` 数组）。

```
LDG.E.128  R0, [R2]        // global → 寄存器
STS.128    [R4], R0        // 寄存器 → shared
BAR.SYNC                   // 等所有线程都写完
```

问题在于**数据必须在寄存器里中转**。而寄存器是 GPU 最稀缺的资源：一个 SM 只有 65536 个 32-bit 寄存器，分给每个线程通常 32~255 个。想吃 16 B（4 个寄存器）的数据，就得先腾出 4 个寄存器；想同时有 32 个请求在飞，就是 128 个寄存器——直接爆。

这就是为什么在 Ampere 之前，「加深流水线」和「提高 occupancy」天然冲突：**两者抢同一份寄存器**。

### 3.2 第二代：`cp.async`（Ampere, `sm_80`）🔶

`cp.async` 是一条指令完成 global → shared 的搬运，**不经过寄存器**：

```
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;
cp.async.commit_group;
...
cp.async.wait_group N;      // 只等除了最新 N 组以外的都完成
```

三个设计要点：

- **粒度是 4 / 8 / 16 字节**。16 B 那一档能走 `cg`（cache global，绕过 L1）直通 shared。
- **不占寄存器**。数据不进 register file，所以「同时有多少请求在飞」不再受寄存器数量限制——这是它最本质的收益，不是「更快」，而是「能开更多」。
- **`commit_group` / `wait_group` 是分组的**。`wait_group 3` 的意思是「我只要保证前 N-3 组完成了，最后 3 组还在飞也没关系」。这一条指令就是多级缓冲的硬件原语：级数 = 那个 `N`。

对照第 043 期的算术强度视角：一次 `LDG.32` 买 4 字节，一次 `cp.async` 买 16 字节。同样的 MLP（在飞的请求数）下，后者**带宽是前者的 4 倍**。这直接对应我们前面看到的「加并发只能买延迟，买不到带宽」——`cp.async` 换了个方向，它**提高每个请求的字节密度**。

### 3.3 第三代：TMA（Hopper, `sm_90`）🔶

TMA（Tensor Memory Accelerator）把搬运再推一步：**地址生成也从 SM 上挪走**。

```
// 主机侧：把多维 tile 的形状、stride、swizzle 编码成一个 descriptor
cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmem_ptr, dims, strides, ...);

// 设备侧：一条指令，由单个线程发起，硬件自己去搬
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem], [map, {x, y}], [mbar];
```

相比 `cp.async`，多了四件事：

1. **一次描述整个多维 tile**。`cp.async` 每个线程一次只能搬 16 B 的一维片段，一个 128×64 的 fp16 tile 需要上千条指令、上千次地址计算；TMA 是一两条指令。
2. **地址由硬件生成**。SM 侧不再付地址算术的指令开销，也不再需要那么多寄存器存坐标。
3. **可以配 zero-fill 处理边界**，越界自动补零，省掉手写 predicate。
4. **完成信号走 `mbarrier`**，而且带事务字节计数（`expect_tx`），同步粒度是「整个 tile 到齐」而不是「每个线程各自到齐」。

代价是：descriptor 要在主机侧建好（或 `cuTensorMapEncodeTiled` 一次），tile 的形状和 stride 得是编译期或启动期已知的——**动态形状是 TMA 的天敌**，这条和第 021 期（动态形状）直接相关。

### 3.4 三代对照

| | 谁发起 | 单次最大字节 | 占寄存器 | 地址谁算 | 完成信号 |
|---|---|---|---|---|---|
| `LDG` + `STS` | 每个线程 | 16 B | 占 | SM | `BAR.SYNC` |
| `cp.async` | 每个线程 | 16 B | **不占** | SM | `commit/wait_group` |
| TMA (`cp.async.bulk.tensor`) | **单个线程** | 整个 tile（KB 级） | **不占** | **硬件** | `mbarrier` |

一句话概括这条演进线：**每一步都在减少 SM 侧的指令数和寄存器占用，同时提高单个请求搬运的字节数**。

⚪ 而这三步背后的抽象完全通用：任何加速器做流水线，都要回答「谁来搬、一次搬多少、怎么通知算完了」。华为昇腾的 `DataCopy` + `TQue`、TPU 的 DMA 队列、CPU 上的 `memcpy` + 线程，都是这三个问题的不同答案。你换硬件的时候，可以忘掉 `cp.async` 这个名字，但别忘掉这三个问题。

---

## 四、在 PyTorch 中怎么用

### 4.1 第一层：Triton 的 `num_stages`

Triton 把整个软件流水线打包成一个参数。你写的循环里只有「load tile → 用它算」，Triton 的编译器负责把它变成 cp.async / TMA + 多缓冲：

```python
# 需 NVIDIA GPU（Ampere 及以上），本机 CPU 环境跑不了
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # num_stages=4 告诉编译器：这个循环做 4 级流水线
    for k in tl.range(0, K, BK, num_stages=4):
        a = tl.load(A + (pid_m * BM + tl.arange(0, BM))[:, None] * K
                    + (k + tl.arange(0, BK))[None, :])
        b = tl.load(B + (k + tl.arange(0, BK))[:, None] * N
                    + (pid_n * BN + tl.arange(0, BN))[None, :])
        acc += tl.dot(a, b)
    tl.store(C + ..., acc)
```

`num_stages=4` 这四个字背后就是 4 份 A/B tile 的 shared memory、4 组 `commit_group` 标记、以及一个 `wait_group 3` 的等待条件。你不写 buffer，编译器替你开。

### 4.2 第二层：Inductor 怎么决定 `num_stages`

PyTorch 2.x 的 `torch.compile` 走 Inductor 生成 Triton kernel。它对流水线的处理有一个很有意思的兜底逻辑，我读了本机 2.14.0 的源码：

第一，开关在 `torch/_inductor/config.py` 第 2332 行，属于 `config.triton`：

```python
dynamic_disable_pipelining = True
```

第二，`torch/_inductor/codegen/triton.py` 第 6985 行把它塞进 `inductor_meta`：

```python
# 摘自 codegen/triton.py，这个键被塞进传给后端的 inductor_meta
inductor_meta = {
    "dynamic_disable_pipelining": config.triton.dynamic_disable_pipelining,
    # 其余键略
}
```

第三，`torch/_inductor/runtime/triton_heuristics.py` 第 1085 行附近，是这个开关真正生效的地方：

```python
if (
    isinstance(exc, (OutOfResources, torch.cuda.OutOfMemoryError))
    and (
        config.num_stages > 1
        or config.kwargs.get("NUM_STAGES", 1) > 1
    )
    and self.inductor_meta.get("dynamic_disable_pipelining", True)
):
    self.launchers = [self.compile_by_disabling_pipelining(config)]
    return
```

而 `compile_by_disabling_pipelining` 干的事就是把 `cfg.num_stages` 设成 1 重编译一遍。

这段话翻译过来是：**如果 kernel 因为 shared memory 装不下而编译失败，而它本来开了多级流水线，那就把流水线关掉重来一次。**框架比谁都清楚「级数是要花 smem 买的」。

本机跑一下这个开关的默认值：

```python
import torch
from torch._inductor import config

print("torch", torch.__version__)
print("dynamic_disable_pipelining =", config.triton.dynamic_disable_pipelining)
```

```text
torch 2.14.0
dynamic_disable_pipelining = True
```

还有一个小细节值得记：`codegen/triton.py` 第 6656 行，Inductor 真正往生成的代码里写 `num_stages=2` 的场合只有一处，而且是 HIP（AMD）专用：

```python
# Conditionalize pipelining on HIP for Triton due to
# reports of numerical inaccuracies on older Triton
if torch.version.hip and get_triton_version() > (3, 2):
    num_stages = ", num_stages = 2"
else:
    num_stages = ""
```

在 CUDA 上它不显式指定，交给 Triton 自己的默认策略。这是「文档里看不到、只能读源码」的那类事实。

### 4.3 第三层：自己写一个多级缓冲

如果你不在 Triton 里，而是用 `torch.utils.cpp_extension` 直接写循环（或者写 CUDA kernel），流水线得自己搭。下面这个实验在 CPU 上做——它把「一个执行单元搬、另一个执行单元算」这件事完整复现了一遍：

```python
import os
import sys
from torch.utils.cpp_extension import load_inline

os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ["PATH"]

cpp = r"""
#include <torch/extension.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <vector>

static constexpr int64_t TILE = 65536;   // 每块 256 KB 的 float
static volatile double g_sink = 0;

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// LANES 路独立累加：既要躲开 add 的 4 cycle 延迟，也要给 SIMD 留出空间
template <int LANES>
static inline float reduce_tile(const float* b, int64_t n) {
    float acc[LANES];
#pragma unroll
    for (int j = 0; j < LANES; j++) acc[j] = 0.f;
    for (int64_t e = 0; e + LANES <= n; e += LANES) {
#pragma unroll
        for (int j = 0; j < LANES; j++) acc[j] += b[e + j];
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < LANES; j++) s += acc[j];
    return s;
}

// 串行基线：搬完再算，一条路走到黑
template <int LANES>
static double run_serial(int tiles, int reps) {
    std::vector<float> src((size_t)tiles * TILE, 0.5f);
    std::vector<float> slot(TILE);
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        float acc = 0.f;
        double t0 = now_s();
        for (int i = 0; i < tiles; i++) {
            std::memcpy(slot.data(), src.data() + (size_t)i * TILE, TILE * sizeof(float));
            acc += reduce_tile<LANES>(slot.data(), TILE);
        }
        double dt = now_s() - t0;
        g_sink = acc;
        if (dt < best) best = dt;
    }
    return best;
}

// 流水线：生产者线程搬，主线程算，环形 buffer 深度 D
template <int LANES>
static double run_pipe(int D, int tiles, int reps) {
    std::vector<float> src((size_t)tiles * TILE, 0.5f);
    std::vector<std::vector<float>> slots(D, std::vector<float>(TILE));
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        std::mutex m;
        std::condition_variable cv;
        int64_t produced = 0, consumed = 0;
        float acc = 0.f;
        double t0 = now_s();
        std::thread prod([&] {
            for (int64_t i = 0; i < tiles; i++) {
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return produced - consumed < D; });
                }
                std::memcpy(slots[i % D].data(), src.data() + i * TILE, TILE * sizeof(float));
                {
                    std::lock_guard<std::mutex> lk(m);
                    produced++;
                }
                cv.notify_all();
            }
        });
        for (int64_t i = 0; i < tiles; i++) {
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return produced > consumed; });
            }
            acc += reduce_tile<LANES>(slots[i % D].data(), TILE);
            {
                std::lock_guard<std::mutex> lk(m);
                consumed++;
            }
            cv.notify_all();
        }
        prod.join();
        double dt = now_s() - t0;
        g_sink = acc;
        if (dt < best) best = dt;
    }
    return best;
}

static double only_copy(int tiles, int reps) {
    std::vector<float> src((size_t)tiles * TILE, 0.5f);
    std::vector<float> slot(TILE);
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        for (int i = 0; i < tiles; i++)
            std::memcpy(slot.data(), src.data() + (size_t)i * TILE, TILE * sizeof(float));
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    g_sink = slot[7];
    return best;
}

template <int LANES>
static double only_compute(int tiles, int reps) {
    std::vector<float> slot(TILE, 0.5f);
    double best = 1e30;
    float acc = 0.f;
    for (int r = 0; r < reps; r++) {
        acc = 0.f;
        double t0 = now_s();
        for (int i = 0; i < tiles; i++) acc += reduce_tile<LANES>(slot.data(), TILE);
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    g_sink = acc;
    return best;
}

// pybind 绑不了模板，用非模板包装分发
static double w_serial(int L, int tiles, int reps) {
    return L == 8 ? run_serial<8>(tiles, reps) : run_serial<4>(tiles, reps);
}
static double w_pipe(int L, int D, int tiles, int reps) {
    return L == 8 ? run_pipe<8>(D, tiles, reps) : run_pipe<4>(D, tiles, reps);
}
static double w_compute(int L, int tiles, int reps) {
    return L == 8 ? only_compute<8>(tiles, reps) : only_compute<4>(tiles, reps);
}
"""

mod = load_inline(name="pipe044d", cpp_sources=cpp,
                  functions=["only_copy", "w_compute", "w_serial", "w_pipe"],
                  extra_cflags=["-O3"], verbose=False)

TILES, REPS = 128, 9
# 预热：首次调用要付 page fault 与线程创建的成本，不预热的话第一组数据会整体偏慢
mod.only_copy(TILES, 2)
mod.w_compute(4, TILES, 2)
mod.w_pipe(4, 2, TILES, 2)
t_copy = mod.only_copy(TILES, REPS)
print(f"纯搬运 32 MiB：{t_copy*1e3:.2f} ms → {TILES*65536*4/t_copy/1e9:.1f} GB/s")
print(f"\n{'并行度':>6} {'纯计算':>9} {'串行基线':>9} {'D=1':>8} {'D=2':>8} "
      f"{'D=4':>8} {'D=8':>8} {'实测加速':>9} {'理论上限':>9}")
for L in (4, 8):
    tc = mod.w_compute(L, TILES, REPS)
    ts = mod.w_serial(L, TILES, REPS)
    pipes = [mod.w_pipe(L, D, TILES, REPS) for D in (1, 2, 4, 8)]
    best = min(pipes)
    print(f"{L:>6} {tc*1e3:>8.2f}ms {ts*1e3:>8.2f}ms "
          + " ".join(f"{t*1e3:>7.2f}ms" for t in pipes)
          + f" {ts/best:>8.2f}x {(t_copy+tc)/max(t_copy,tc):>8.2f}x")
```

```text
纯搬运 32 MiB：0.44 ms → 76.0 GB/s

   并行度       纯计算      串行基线      D=1      D=2      D=4      D=8      实测加速      理论上限
     4     1.18ms     1.77ms    1.99ms    1.33ms    1.33ms    1.32ms     1.34x     1.37x
     8     0.67ms     1.11ms    1.44ms    1.15ms    0.75ms    0.73ms     1.53x     1.66x
```

这里 `LANES` 是计算侧的并行度，本质就是 roofline 里的**算术强度**旋钮：`LANES` 越大，同样字节数能干的计算越多，搬运占总时间的比例就越高。

四组数字，四个结论：

**（1）实测几乎贴住理论上限。** `LANES=4` 时算 ÷ 搬 = 1.18 ÷ 0.44 ≈ 2.7，理论上限 1.37x，实测 1.34x（达到 98%）。`LANES=8` 时算得更快了，搬算比拉到 1.5，上限升到 1.66x，实测 1.53x（92%）。第二节那个 $(c+p)/\max(c,p)$ 不是纸上的公式，它是能对得上的。

**（2）`D=1` 是纯亏。** `LANES=8` 时串行 1.11 ms，`D=1` 反而 1.44 ms——慢了 30%。因为 `D=1` 时生产者必须等消费者读完才能覆盖那唯一的 buffer，两者完全串行，还白付了 128 次条件变量的唤醒成本。**只开一格缓冲的多线程，是最糟糕的选择。**

**（3）`D=2` 常常还不够。** `LANES=8` 时 `D=2` 是 1.15 ms，几乎等于串行基线 1.11 ms——重叠了，但收益被吃光了。原因不是原理错了，是**交接本身有时延**：消费者算完 tile $i$、唤醒生产者、生产者写完、再唤醒消费者，这一来一回每次几微秒，128 个 tile 就是几百微秒。`D=4` 一跳到 0.75 ms，因为生产者从此不用等，始终领先 3 个 tile。

这件事在 GPU 上同样成立，只是数字不同：CPU 上条件变量唤醒要进内核、走调度，是微秒级；GPU 上用 `mbarrier` 自旋轮询，是几十纳秒级。但结论一样——**流水线级数不能只按「装得下」算，得按「盖得住交接延迟」算**。这也解释了为什么 CUTLASS / Triton 的默认 `num_stages` 是 3~5 而不是 2。对照第 042 期讲的 grain size：并行度不是越大越好，得匹配开销的量级。

**（4）达成率随搬算比上升而下降。** `LANES=4`（搬运占 27%）达成 98%，`LANES=8`（搬运占 40%）只达成 92%。差的这 6 个百分点就是**争抢**：CPU 上 `memcpy` 和 compute 抢同一份内存带宽，算得越猛，抢得越凶。GPU 上 `cp.async` 走的是 global→shared 的独立通路、`TMA` 更是完全由硬件引擎驱动，所以争抢远比 CPU 上小——但绝不会是零，因为最后都要过 HBM。

---

## 五、`num_stages` 到底能开几级：一笔 smem 账

第三节那句话得落到具体数字：流水线的每一级都要在 shared memory 里占一块 tile。开得多，装不下；开得少，盖不住延迟。

H100 SXM5 的官方参数是每 SM 228 KB shared memory、单个 block 上限 227 KB。一个 stage 要同时放 A、B 两个 tile：

```python
SMEM_PER_BLOCK_KB = 227   # H100 SXM5：单 block 能申请到的上限
DTYPE_BYTES = 2           # fp16 / bf16

print(f"{'BM':>5} {'BK':>4} {'stages':>7} {'每 stage':>9} {'smem/block':>11} "
      f"{'可放 block 数':>13} {'判定':>6}")
for BM, BK, BN in [(128, 64, 128), (128, 128, 128)]:
    per_stage_kb = (BM * BK + BK * BN) * DTYPE_BYTES / 1024
    for stages in (2, 3, 4, 5, 6, 7, 8):
        need = per_stage_kb * stages
        blocks = int(SMEM_PER_BLOCK_KB // need)
        ok = "可行" if blocks >= 1 else "超限"
        print(f"{BM:>5} {BK:>4} {stages:>7} {per_stage_kb:>7.0f}KB {need:>9.0f}KB "
              f"{blocks:>13} {ok:>6}")
    print()

for BM, BK, BN in [(128, 64, 128), (128, 128, 128), (64, 64, 64)]:
    per_stage_kb = (BM * BK + BK * BN) * DTYPE_BYTES / 1024
    max_stages = int(SMEM_PER_BLOCK_KB // per_stage_kb)
    print(f"BM={BM} BK={BK} BN={BN}: 每 stage {per_stage_kb:.0f} KB → "
          f"最多 {max_stages} 级（用满 {max_stages*per_stage_kb:.0f} KB）")
```

```text
   BM   BK  stages   每 stage  smem/block    可放 block 数     判定
  128   64       2      32KB        64KB             3     可行
  128   64       3      32KB        96KB             2     可行
  128   64       4      32KB       128KB             1     可行
  128   64       5      32KB       160KB             1     可行
  128   64       6      32KB       192KB             1     可行
  128   64       7      32KB       224KB             1     可行
  128   64       8      32KB       256KB             0     超限

  128  128       2      64KB       128KB             1     可行
  128  128       3      64KB       192KB             1     可行
  128  128       4      64KB       256KB             0     超限
  128  128       5      64KB       320KB             0     超限
  128  128       6      64KB       384KB             0     超限
  128  128       7      64KB       448KB             0     超限
  128  128       8      64KB       512KB             0     超限

BM=128 BK=64 BN=128: 每 stage 32 KB → 最多 7 级（用满 224 KB）
BM=128 BK=128 BN=128: 每 stage 64 KB → 最多 3 级（用满 192 KB）
BM=64 BK=64 BN=64: 每 stage 16 KB → 最多 14 级（用满 224 KB）
```

这张表解释了一堆看起来毫无道理的默认值：

- `BM=128, BK=64` 时 7 级是硬上限，第 8 级直接编译不过——这就是 4.2 节那段 `compile_by_disabling_pipelining` 兜底逻辑存在的原因。
- `BK` 从 64 翻到 128，每 stage 从 32 KB 涨到 64 KB，级数上限从 7 掉到 3。**所以"BK 开大一点"和"流水线开深一点"是直接的资源竞争关系**，Autotuner 才需要把这两个维度放在一起搜。
- 反向的一个：`BM=BK=BN=64` 时每 stage 只 16 KB，理论上能开 14 级——但 64×64 的 tile 算力利用率低，所以实际没人这么选。

---

## 六、围绕该领域展开

### 6.1 warp specialization：把角色分工写进 kernel 结构

`cp.async` / `TMA` 解决了「怎么异步搬」，但还剩一个问题：**谁来发这些指令**？

如果所有 warp 都既搬又算，那么每个 warp 在发完 `cp.async` 之后要等 `wait_group`，这段等待里它什么也算不了。Hopper 上的做法是**按角色切分 warp**：

- **producer warp**（通常 1 个）：只负责发 `cp.async.bulk.tensor`，发完就在 `mbarrier` 上等
- **consumer warp**（其余）：只负责在 `mbarrier` 上等数据到齐，到了就发 `wgmma` 算

这就是 CUTLASS 3.x 里 Hopper kernel 的标准结构。它和第 017 期（CUDA Streams）是同一个思想的两种尺度：Streams 是在**主机侧**把不同任务丢到不同队列让 GPU 并发；warp specialization 是在**设备侧**把不同职责丢到不同 warp 让它并发。

### 6.2 occupancy 的含义被改写了

传统 GPU 编程的第一性原理是：**用多 warp 切换来藏延迟**。所以 occupancy（每 SM 上活跃 warp 数 / 最大 warp 数）越高越好，低于 50% 就会被当成问题。

但在 Hopper 的 GEMM kernel 里，你会看到 occupancy 只有 12.5%（一个 block、8 个 warp、SM 最多容 64 个 warp），性能却接近峰值。上面那张 smem 表就是原因：`BM=128, BK=64, stages=5` 就要 160 KB，一个 SM 只装得下一个 block。

这能成立，是因为**藏延迟的手段换了**：

| 时代 | 藏延迟的主力 | occupancy 的作用 |
|---|---|---|
| Pascal / Volta | warp 切换 | 决定性 |
| Ampere | warp 切换 + `cp.async` 多缓冲 | 重要但可让位 |
| Hopper | `TMA` + `wgmma` 双异步 + 深流水 | **不再是主要手段** |

所以「occupancy 低就是有问题」这个直觉在新硬件上会误判。第 043 期讲的是「访存买多少字节」，这一期讲的是「等待怎么填」——两者都指向同一个变化：**SM 从"什么都自己干"变成了"指挥别人干"**。

### 6.3 与前几期的连接

- **第 041 期（融合体内部的循环顺序）**：那期讲一个循环体内部先访问谁；本期讲循环之间怎么交错。前者是空间局部性，后者是时间重叠。
- **第 042 期（CPU 并行与 grain size）**：那期的结论「粒度要匹配开销」在这里换了张脸——流水线级数要匹配交接延迟，`D=2` 不够就是这个道理。
- **第 043 期（访存层次）**：算的是每次访问的字节成本；本期算的是等这段时间怎么填。两期的账加起来才是完整的：`时间 = max(字节/带宽, 延迟/并发度)`。
- **第 040 期（算子融合）**：融合减少搬运总量，流水线隐藏搬运时间。**先做融合再谈流水线**——搬运量少一半，流水线的收益上限也跟着变。
- **第 005 期（Flash Attention）**：FA 之所以能比朴素 attention 快那么多，核心就是「分块 + 在线 softmax + 多级缓冲」，把 HBM 往返压到最少。它本质上是一个把流水线用到极致的 kernel。

### 6.4 一个容易漏掉的：`mbarrier` 为什么不用 `BAR.SYNC`

`__syncthreads()`（`BAR.SYNC`）是**全 block 屏障**：所有线程必须到齐。在 warp specialization 的结构里这行不通——producer warp 和 consumer warp 干的活不同，用全屏障会把 producer 卡在 consumer 的节奏上。

`mbarrier` 是**可编程的、带计数的、可以只让部分线程等**的屏障，而且支持 `expect_tx`（预先声明「我要收 N 字节」），TMA 搬完自动递减计数。所以它的语义是「等数据到齐」而不是「等人到齐」。

这个区别在 CPU 上就对应「条件变量」vs「屏障」。上面那个实验用的是条件变量，正是因为生产者和消费者不需要同时到齐。

---

## 七、什么时候该用 / 不该用

**该用：**

- kernel 的搬运时间和计算时间在同一个量级（$c/p$ 在 0.3~3 之间）。GEMM、attention、卷积都属于这一类。
- 有明确的 tile 结构，可以提前知道下一块要什么数据（`i+1` 的地址在算 `i` 时就能算出来）。
- shared memory 有富余，或者愿意为了流水线牺牲 occupancy。

**不该用：**

- **纯计算受限**（$c \ll p$）。搬运本来就不在关键路径上，加流水线只是白占 smem、白占寄存器。用第二节的公式一算就知道上限接近 1。
- **纯搬运受限**（$c \gg p$，例如逐元素算子）。这时候该做的是减少搬运量（融合、量化、更好的访存模式），不是隐藏搬运时间。流水线能把 `c` 藏进 `p` 里，但藏不进「带宽」这堵墙——第一节的实验已经说明：并发加到 32 条链，吞吐 66.6 GB/s 已经到顺序读基线 76.3 GB/s 的 87%，再加并发最多只剩 13%。
- **访存是纯随机的**（gather / scatter、稀疏索引）。流水线要求「提前知道下一块在哪」，随机访问没有这个结构，只能靠提高并发度（这又回到 MLP）。
- **动态形状严重**。TMA 的 descriptor 需要在启动前确定 tile 形状（第 021 期的老问题）。

---

## 八、常见坑

### 坑 1：把「延迟问题」当成「带宽问题」来治

这是最贵的一个。两者的处方完全相反：

- 延迟问题 → 加并发（多 warp、多级缓冲、更深的展开）
- 带宽问题 → 减字节（融合、量化、更好的 coalescing）

第一节那张 MLP 表就是分界线：从 1 条链到 16 条链，每翻一倍延迟减半（延迟侧）；从 16 条到 32 条，吞吐从 24.7 涨到 66.6 GB/s 之后就只剩 13% 的空间了（带宽侧）。**先测一下自己站在哪一侧，再决定往哪边使劲。**

### 坑 2：只看「平均延迟」不看「并发度」

我在做第一节实验时扫到了 64 条链，测出来 718 GB/s——比单线程顺序读的 76.3 GB/s 快将近 10 倍。这显然不可能，直接作废了那一档。

事后复盘，判据可以写得很硬：**任何一档的有效延迟如果低于本机 L1 的延迟，那一档一定测错了**。同一台机器上 L1 只要 1.35 ns，而 718 GB/s 意味着每次 miss 只要 0.18 ns——比 L1 还快 7 倍，物理上不成立。

这条和第 037 期（零假设基线）是同一个方法论：**给每一档结果配一个物理上界，越界即作废，不要试图解释它**。差一点就把一个错误数字写进结论里了。

### 坑 3：`num_stages` 无脑往上开

三个反噬：

1. **smem 装不下**，直接编译失败（触发 4.2 节的降级逻辑，你会看到 kernel 悄悄变成 `num_stages=1`，性能莫名其妙掉一半）。
2. **occupancy 被挤掉**。`stages=5` 用 160 KB，一个 SM 就只剩一个 block，原本靠多 block 掩盖的其他延迟（比如 epilogue 的写回）就暴露了。
3. **级数超过需求就是浪费**。第二节的公式里，`D` 只要够盖住交接延迟和一次搬运的时间就饱和了，再多不会更快。`D=4` 和 `D=8` 在实验里几乎一样（0.75 vs 0.73 ms）就是证明。

### 坑 4：缓冲开对了，同步写错了

流水线最难的不是开 buffer，是**同步**。三个典型错误：

- **生产者覆盖还没读完的 buffer**：忘了检查 `produced - consumed < D`，数据静默出错，不报错，只是结果偶尔不对。
- **用全屏障代替定向同步**：在 `__syncthreads()` 上等 TMA 完成，会把无关 warp 一起卡住，流水线退化成串行。
- **忘记 `expect_tx`**：`mbarrier` 的字节计数没设对，会永久挂起或提前放行——前者死锁，后者读脏数据。

第一条在我上面那段 C++ 里就是那行 `cv.wait(lk, [&] { return produced - consumed < D; })`。删掉它，双缓冲立刻变成「生产者跑飞、消费者读到半新半旧的数据」，而且**大概率不崩，只是结果错**。这类 bug 比崩溃难查一个数量级。

---

## 九、一句话总结

异步拷贝和软件流水线买的是**延迟的隐藏**，不是**带宽的增加**——它的收益上限被 $\frac{c+p}{\max(c,p)}$ 钉死，能被拿走多少取决于搬运和计算会不会抢同一份带宽，而能开几级流水线则由 shared memory 预算说了算。

---

## 今日练习

<details>
<summary>今日练习：算一算你自己的 kernel 该不该上流水线</summary>

**题目**

假设你在 H100 上写一个 fp16 的 GEMM kernel，`BM=BN=128`，`BK=64`。

1. 如果选 `num_stages=4`，一个 block 占多少 shared memory？一个 SM 能放几个 block？
2. 如果实测这个 kernel 的 compute 时间是 200 µs、搬运时间是 150 µs（都是单 block 口径、串行执行），上了完美流水线之后的理论总时间是多少？加速比是多少？
3. 如果把 `BK` 改成 128，`num_stages` 最多能开到几？这对第 2 问的答案有什么影响？

**参考答案**

**第 1 问**：每 stage = `(128×64 + 64×128) × 2 B = 32768 B = 32 KB`。4 级 = 128 KB。
H100 单 block 上限 227 KB，`227 // 128 = 1`，所以一个 SM 只能放 **1 个 block**。此时如果 block 是 256 线程（8 warps），occupancy 上限是 `8/64 = 12.5%`。

**第 2 问**：注意题目给的 200 µs 和 150 µs 是**整个 kernel 的总量**口径，和第二节那个 per-tile 口径的公式不同，要换算成总量形式：

$$
T_{\text{sync}} = c + p = 350\ \mu s,\qquad
T_{\text{pipe}} = \max(c,p) + \frac{\min(c,p)}{N}
$$

其中 $N$ 是 tile 数。$N$ 很大时第二项可忽略，所以 $T_{\text{pipe}} \approx \max(150, 200) = 200\ \mu s$，加速比

$$
S = \frac{350}{200} = 1.75\times
$$

和第二节的 $(c+p)/\max(c,p) = 350/200 = 1.75$ 是同一个数——两种口径推出同一个答案，可以拿来自查。

**第 3 问**：`BK=128` 时每 stage = `(128×128 + 128×128) × 2 = 65536 B = 64 KB`，`227 // 64 = 3`，最多 **3 级**。
影响有两面：

- 级数从 4 掉到 3，盖交接延迟的余量变小，可能盖不住 TMA 的往返延迟（所以在 Hopper 上手写 kernel 时，`BK=64/stages=5` 常常比 `BK=128/stages=3` 更快）。
- 但 `BK` 变大意味着每次搬的字节更多、内层循环更长，**单位数据的地址计算开销下降**，`c/p` 的比值会变。所以这两个参数必须联合搜，不能分开定——这就是 Triton / CUTLASS 的 Autotuner 为什么把 `(BM, BN, BK, num_stages, num_warps)` 放在一个搜索空间里。

**加分思考**：如果这个 kernel 的搬运时间不是 150 µs 而是 30 µs（$c \ll p$），加速比上限变成多少？还值得开 4 级流水线吗？
答：$S_{\max} = (30+200)/200 = 1.15$。为了 15% 的收益牺牲 128 KB smem 和 87.5% 的 occupancy，通常**不划算**——这时候更该做的是调整 tile 形状让计算更满，或者干脆接受 `num_stages=2`。

</details>
