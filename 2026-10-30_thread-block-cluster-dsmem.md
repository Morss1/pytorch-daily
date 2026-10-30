# PyTorch 每日一课 · 第 046 期

## 线程块簇与分布式共享内存：把 shared memory 从「一个 block 的私产」变成「一个 GPC 的公共池」

> 日期：2026-10-30 ｜ 难度：⭐⭐⭐⭐ ｜ 前置知识：CUDA 的 block / shared memory 模型、Hopper 的 TMA 与 mbarrier（第 044 期）、occupancy 的三把锁（第 045 期）｜ 预计阅读：28 分钟
>
> 关联：第 043 期（内存层次的台阶）· 第 044 期（异步拷贝与软件流水线）· 第 045 期（occupancy）。本期是这三期的合流点 —— 043 讲了「每次访问买多少字节」，044 讲了「等这次访问的时间怎么填」，045 讲了「多少个线程同时在场」。这三个问题都默认了一个前提：**并发的基本单位是 block**。Hopper 把这个前提松了一格。

---

## 一、这个领域解决什么问题

先把账摆出来。一个 block 想要高效地算一个 fp16 的 GEMM tile，它需要什么？

两样东西：**足够大的 tile**（提升算术强度），**足够快的暂存区**（把 A/B 的 tile 停在片上不让它掉回 HBM）。

这两件事在 Hopper 上直接打架。

**算术强度这一侧**：一个 block 拿着 $T \times T$ 的输出 tile，沿 $K$ 方向滚动，FLOPs 是 $2T^2K$，从 L2 拉进 SMEM 的字节是 $2 \times T \cdot K \times 2$（A、B 各一遍，fp16 两字节）。所以

$$
\text{AI}(T) = \frac{2T^2K}{4TK} = \frac{T}{2} \quad \text{FLOP/byte}
$$

**SMEM 这一侧**：单 stage 要 $2TK \times 2$ 字节，软件流水线要 3~4 级。$T=256, K=64$ 时单 stage 就是 64 KB，4 级 256 KB —— 而 H100 一个 block 最多只能申请 **227 KB**。

于是你被卡在中间：tile 想再大一点，SMEM 不放；SMEM 想多留几级流水，tile 只能变小。这就是 044 期最后落到的那句话的另一面 —— 流水线级数是被 SMEM 预算硬顶住的。

**在 cluster 出现之前，只有一个出口**：把装不下的部分放回 L2/global。代价是每次取用都要走一趟 L2，延迟从几十 cycle 跳到几百 cycle（本期第二节会把这条阶梯量出来）。

**Thread Block Cluster（线程块簇）** 是 Hopper（compute capability 9.0）在 grid 和 block 之间插进去的一层：一组最多 8 个 block，硬件**保证**它们同时驻留、且落在**同一个 GPC** 里。这个保证一旦成立，两件原本不可能的事就成立了：

1. 一个 block 可以 `mapa` 出邻居 block 的 shared memory 地址，直接读写它的 SMEM（**Distributed Shared Memory，DSMEM**）；
2. 一次 TMA 搬运可以把同一块数据**同时投递**到 cluster 内多个 block 的 SMEM（**multicast**）。

所以这个领域要回答的问题是：

> **当一块数据对 8 个 block 都有用、而每个 block 的 SMEM 都装不下它时，怎么让这 8 个 block 共享它，而不让它退回 L2？**

---

## 二、核心思想：把 8 个工位串成一个车间

用工厂打比方。

**原来的模型**（CUDA 从 2007 年到 2020 年一直是这个）：每个 block 是独立工位，工位有自己的工具台（SMEM，227 KB）和抽屉（寄存器）。工位之间不许说话 —— 编程模型明确规定 block 之间不能同步、不能保证同时活着。要交换东西，只能放到**中央仓库**（L2 / global memory），然后各自去取。中央仓库很大但很远。

**cluster 做的事**：把 8 个相邻工位指定成一个**车间**。车间有一个硬件承诺：这 8 个工位在同一个厂房（GPC）里，同时开工、同时收工。有了这个承诺，就可以在工位之间拉一条**传送带**（GPC 内部的 SM-to-SM 网络），把半成品直接递过去，不用进出中央仓库。

这个类比的三个要点，恰好对应后面要讲的三个代价：

| 类比 | 硬件事实 | 代价 |
|---|---|---|
| 车间必须整体占用 8 个工位 | cluster 的 block 必须在同一 GPC 同时驻留 | 调度不灵活，occupancy 变成「数车间」 |
| 传送带比中央仓库近，但比伸手够自己的工具台远 | DSMEM 181 cycle vs 本地 SMEM 29 cycle | 别把热数据放邻居那儿 |
| 传送带是车间内共享的一条 | SM-to-SM 网络带宽被 cluster 内所有 block 分摊 | cluster 越大，单 block 分到的越少 |

先记住一句话：**cluster 提供的不是「更快的通信」，而是一个「存在性承诺」**。所有能力（DSMEM、multicast、cluster 级 barrier）都是从这句承诺里长出来的。没有这个承诺，`map_shared_rank` 返回的指针随时可能指向一块已经被释放的 SMEM。

---

## 三、三档阶梯：把数字摆上桌

在讲机制之前，先把「本地 / 跨单位置 / 走中央仓库」这三档的差距量出来。这是整个话题的地基 —— 如果三档之间没有台阶，cluster 就没有存在意义。

本机（Apple M5）没有 CUDA，但「本地存储 → 片上共享 → 片外」这个阶梯结构是跨硬件通用的（这一点在第 043 期已经说过）。我用单线程 pointer chasing 把三档量出来：

```python
"""本机三档存储延迟：32 KB 落 L1 / 8 MB 落 L2 / 64 MB 越过 L2。

用 cc 现编一个几十行的小 C 程序，Python 只负责编译、运行、解析。
指针必须声明成 volatile，否则整条依赖链没有副作用，会被 -O3 整段删掉。
"""
import os
import subprocess
import tempfile

C_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* 单线程 pointer chasing：chain 是随机单环，每步 idx = chain[idx] */
static double chase(volatile int64_t* chain, int64_t steps, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        int64_t idx = chain[0];
        uint64_t t0 = now_ns();
        for (int64_t s = 0; s < steps; ++s) idx = chain[idx];
        uint64_t t1 = now_ns();
        double per = (double)(t1 - t0) / (double)steps;
        if (per < best) best = per;
        if (idx == -1) return -1.0;
    }
    return best;
}

int main(void) {
    int64_t sizes[] = {32 * 1024, 8 * 1024 * 1024, 64 * 1024 * 1024};
    const char* names[] = {"32 KB", "8 MB", "64 MB"};
    srand(0);
    for (int k = 0; k < 3; ++k) {
        int64_t n = sizes[k] / 8;
        int64_t* chain = malloc(n * sizeof(int64_t));
        for (int64_t i = 0; i < n; ++i) chain[i] = i;
        for (int64_t i = n - 1; i > 0; --i) {          /* Fisher-Yates 打乱 */
            int64_t j = rand() % (i + 1);
            int64_t t = chain[i]; chain[i] = chain[j]; chain[j] = t;
        }
        printf("CHASE %s %.2f\n", names[k], chase(chain, 2000000, 5));
        free(chain);
    }
    return 0;
}
"""

with tempfile.TemporaryDirectory() as d:
    c_path, exe = os.path.join(d, "lat.c"), os.path.join(d, "lat")
    with open(c_path, "w") as f:
        f.write(C_SRC)
    subprocess.run(["cc", "-O3", "-o", exe, c_path], check=True)
    out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout

print("工作集      每步延迟(ns)")
lat = {}
for line in out.splitlines():
    parts = line.split()
    if parts and parts[0] == "CHASE":
        lat[parts[1]] = float(parts[-1])          # 键是 "32" / "8" / "64"
        print(f"{' '.join(parts[1:-1]):>8s}  {float(parts[-1]):>12.2f}")

# 把比值也印出来 —— 绝对值每次都漂，比值才是能跨运行比的东西
l1, l2, dram = lat["32"], lat["8"], lat["64"]
print(f"\nL1  -> L2    {l2 / l1:>6.1f}x")
print(f"L2  -> DRAM  {dram / l2:>6.1f}x")
print(f"L1  -> DRAM  {dram / l1:>6.1f}x")
```

```text
工作集      每步延迟(ns)
   32 KB          1.60
    8 MB          6.83
   64 MB         49.73

L1  -> L2       4.3x
L2  -> DRAM     7.3x
L1  -> DRAM    31.1x
```

相邻两档的比值印在输出的最后三行：**每一档之间差 4~7 倍，跨完两层差三十倍左右**。绝对值在不同次运行之间会漂 ±30%（L1 那档最敏感，1.4~1.6 ns 之间跳），但比值稳定 —— 所以下面只比比值，不比绝对值。

现在把 Hopper 的对应数字放上来。这组数据来自亚利桑那州立大学那篇 Hopper 微基准论文（*Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis*, arXiv:2501.12084，Table III / Table V，CUDA 层面的直接测量，不是我的推算）：

| 档位 | 延迟 | 相对本地 SMEM | 谁来搬 |
|---|---|---|---|
| 本地 shared memory（`ld.shared`） | **29 cycle** | 1.0x | SMEM 直接寻址 |
| 本地 SMEM，但走 DSM 接口（`ld.shared::cluster` 指向自己） | **33 cycle** | 1.14x | 多付 4 cycle 的包封装开销 |
| 邻居 block 的 SMEM（cluster=2） | **181 cycle** | **6.2x** | GPC 内 SM-to-SM 网络 |
| 邻居 block 的 SMEM（cluster 2→16） | 184~213 cycle | 6.3~7.3x | 同上，随 cluster 变宽变慢 |
| 经 global memory 中转（store + load） | **956~1110 cycle** | 33~38x | L1 → L2 → L1 |
| 同上，论文引的官方说法 | — | ~7x 改善 | DSM 相对 global 的官方宣传值 |

把两条阶梯并排放：

- 本机：相邻档之间 4~7 倍（见上面输出的最后三行）
- Hopper：SMEM → DSMEM 是 **6.2 倍**，DSMEM → 走 global 是 **5.3~6.1 倍**

**形状几乎一样：每一档之间差 5 倍上下。** 这不是巧合，是「片上网络 / 一致性结构 / 片外存储」这三层的物理差距在两种完全不同的硬件上落在同一个量级。所以本机测出来的形状可以当作看 Hopper 的脚手架 —— 但**绝对数字不要迁移**，Hopper 的一档是几十到几百 cycle，本机是几到几十纳秒。

这条阶梯给出了 DSMEM 的定位，非常清楚：

> **DSMEM 是插在「本地 SMEM」和「L2」之间的一档。** 它比本地 SMEM 慢 6 倍（别把热数据放邻居那儿），但比走 L2 快 5~6 倍（所以值得放）。

还有一条容易漏掉的：**DSMEM 可以和 L2 访问同时进行**。NVIDIA 的 Hopper 调优指南明确写了这一点 —— 需要跨 SM 通信的应用可以同时吃 DSMEM 和 L2 两条通路的带宽。这不是「二选一」，是「两条路并联」。

---

## 四、一次硬件承诺：为什么是 GPC，为什么是 8

### 4.1 GPC 是硬件的物理分组

GPU 不是 132 个平等的 SM。它们被组织成若干 **GPC（Graphics Processing Unit Cluster，图形处理簇）**，一个 GPC 里的 SM 共享一些物理资源（纹理单元、部分 L1 通路、SM-to-SM 网络）。

GH100 满配是 **144 SM / 8 GPC**，也就是每 GPC 18 个 SM；H100 SXM5 屏蔽到 132 SM，按每 GPC 16~17 个推算（这个分布是推断，**NVIDIA 没有公开 H100 的逐 GPC SM 数**，但总量 132 与 8 个 GPC 是公开的）。

cluster 必须整体落在一个 GPC 内 —— 因为 SM-to-SM 那条网络是**GPC 内部**的走线，跨 GPC 就没有了。这直接给出尺寸上限：

| 尺寸 | 出处 | 备注 |
|---|---|---|
| **8** | 官方文档：最大**便携**（portable）cluster 尺寸 | 跨架构、跨代保证，代码可以放心写 8 |
| **16** | H100 允许的**非便携**尺寸 | 要显式设 `cudaFuncAttributeNonPortableClusterSizeAllowed` |
| 实际上限 | `cudaOccupancyMaxPotentialClusterSize()` 运行时查询 | 取决于 kernel 的 SMEM/寄存器占用，可能比 16 还小 |

**为什么 portable 只给 8？** 因为这个数值要保证在所有支持 cluster 的硬件上都能跑。GPC 的大小是逐型号变的，8 是保守的跨代下界。你要 16 就是在赌「我这块卡的 GPC 至少 16 个 SM」，赌输了 launch 直接失败。

### 4.2 cluster 尺寸怎么写

两种写法，都在 launch 层面：

```cpp
// 写法 1：编译期写死（__cluster_dims__ 属性）
__global__ void __cluster_dims__(8, 1, 1) my_kernel(...) { ... }

// 写法 2：launch 时给（更灵活，可以按 shape 动态决定）
cudaLaunchConfig_t cfg = {};
cfg.gridDim = grid; cfg.blockDim = block;
cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim.x = 8;   // 注意是三轴，cluster 可以是 2D/3D 的
attrs[0].val.clusterDim.y = 1;
attrs[0].val.clusterDim.z = 1;
cfg.attrs = attrs; cfg.numAttrs = 1;
cudaLaunchKernelEx(&cfg, my_kernel, args...);
```

`__cluster_dims__` 的写法来自 CUDA 的 Hopper 文档与 *The CUDA Handbook* 第 7.8 节；`cudaLaunchKernelEx` 的字段名来自 CUDA Runtime API 文档。**这两段代码在本机没有 nvcc，我没有编译过**，只做了 API 名称与语义的核对。

### 4.3 承诺的另一面：block 可能挤在同一个 SM 上

这一点很容易想错。cluster 保证的是「同一 GPC、同时驻留」，**没有**保证「每个 block 一个 SM」。

CUDA 提供 `cudaLaunchAttributeClusterSchedulingPolicyPreference` 控制调度偏好：

| 策略 | 行为 | 对 SMEM 池的影响 | 对 DSMEM 延迟的影响 |
|---|---|---|---|
| `Spread` | 尽量摊到 GPC 内不同 SM | 池子最大（8 个 SM 的 SMEM 之和） | 最慢（全部走 SM-to-SM） |
| `LoadBalancing` | 允许塞进同一 SM | 池子按实际占用的 SM 数算 | 更快的 block 对（同 SM 时退化成 33 cycle 那档） |

论文的实测里 LoadBalancing 在多数场景下更好，原因正是「同 SM 的 block 不用走 SM-to-SM 网络」。所以：

> **cluster 的 SMEM 池大小不是一个定值**，而是「cluster 内**实际占用的 SM 数** × 227 KB」。写得再大，调度把 8 个 block 塞进 4 个 SM，池子就是 4 × 227 KB，不是 8 × 227 KB。

这是一个典型的「文档没说死、要靠实测确认」的坑，后面会再提。

---

## 五、分布式共享内存：`mapa` 之后发生了什么

### 5.1 地址是怎么变得「跨 block」的

CUDA C++ 层面只有三行：

```cpp
namespace cg = cooperative_groups;
cg::cluster_group cluster = cg::this_cluster();   // 拿到本 cluster 的句柄

// 把我自己 SMEM 里的 smem_ptr，映射成 rank 号那个 block 的同一位置
float* peer = cluster.map_shared_rank(smem_ptr, rank);

cluster.sync();          // 第一次同步：保证对方已经写好
sum = smem[i] + peer[i]; // 直接读邻居
cluster.sync();          // 第二次同步：保证我读完之前对方不许退出
```

`map_shared_rank` 编译成 PTX 的 **`mapa`** 指令（具体形如 `mapa.shared::cluster.u32 %r, %r_addr, %rank;`）。它干的事是把一个「本地 SMEM 地址」重写成「cluster 地址空间里的地址」—— 高位塞进目标 block 的 rank。之后所有 `ld.shared::cluster` / `st.shared::cluster` 就都带着这个 rank 走 GPC 内的网络。

**这就是为什么必须有「存在性承诺」**：`peer` 这个指针的合法性完全依赖「rank 号那个 block 还活着」。CUDA 的规则是：**远程访问必须被两次 `cluster.sync()` 夹住** —— 前面那次保证对方已经把数据写进 SMEM 并分配好，后面那次保证在所有远程读结束之前对方不许退出（block 退出会释放它的 SMEM）。

漏掉第二次同步，就是一个「读到别人已经腾空的工位」的经典竞态。编译器不会报警，因为从 C++ 语义看这两个指针都合法。

### 5.2 反直觉的地方：DSMEM 的访问规则像 global，不像 smem

这是这个话题里最容易踩、也最有意思的一处。你的直觉会说：「我拿到的是 shared memory 的地址，那就该按 SMEM 的规则来 —— 避免 bank conflict、用交错访问。」

**错的。**

NVIDIA 的 Hopper 调优指南原话（中文版）：*为了获得访问分布式共享内存的最佳性能，应使用 CUDA C++ 最佳实践指南中「全局内存」一节描述的访问模式。具体来说，如果可能，对分布式共享内存的访问应**合并**并与 **32 字节段对齐**。应尽可能避免非单位步幅的访问模式。*

两份独立的第三方分析给了同样的结论：

- 微基准论文：DSMEM 的 load 是**以数据包形式传输的，和 global memory load 类似**，所以最优访问模式不是「避免 bank conflict 的交错」，而是「连续合并访问」。
- 另一份 Blackwell/Hopper 的指令级分析：**要让本地 SMEM 跑到 128 B/clk 的峰值，必须用不带 `::cluster` 修饰符的 `ld.shared`**；用 `ld.shared::cluster` 访问本地内存也拿不到峰值。

换句话说：**`map_shared_rank` 交给你的不是「别人的 SMEM」，而是一块「挂载在网络上的、按 global 规则计费的存储」**。同一篇论文的吞吐测量把这个折扣量化了：

| 场景 | 吞吐 | 相对本地 SMEM 理论峰值 |
|---|---|---|
| 本地 SMEM 理论峰值（1755 MHz × 128 B） | **225 GB/s** per SM | 100% |
| 本地 SMEM，不走 DSM 接口 | > 99.8% | ~225 GB/s |
| 本地 SMEM，走 DSM 接口 | **205 GB/s** | **80%** |
| 跨 SM DSMEM（cluster=2 峰值） | **3.28 TB/s**（全 cluster 合计） | — |
| 跨 SM DSMEM（cluster=4 峰值） | **2.78 TB/s** | 比 cluster=2 掉 15% |

最后两行值得单独说：**cluster 从 2 涨到 4，DSMEM 的总吞吐反而降了 15%**。原因是 SM-to-SM 网络是 GPC 内共享的一条通路，block 越多，抢得越凶。所以「cluster 越大越好」是错的 —— 它是一条会饱和的总线。

---

## 六、三笔账：cluster 到底买到了什么

下面三笔账全部是确定性算术（同样的输入永远同样的输出），代码和输出都可以逐位复现。它们分别回答：**容量能变多大**、**字节能省多少**、**调度要付多少**。

```python
"""cluster 的三笔账：容量 / 字节 / 调度。全部是确定性算术。"""

SM_H100 = 132                        # H100 SXM5 的 SM 数
SMEM_PER_SM = 227 * 1024             # 每 SM 可用 shared memory（单 block 上限同为 227 KB）

# ---------- 账 1：容量 ----------
print("=== 账 1：cluster 把共享内存池变大 ===")
print("cluster 尺寸   池大小(KB)")
for c in (1, 2, 4, 8, 16):
    print(f"{c:>12d}  {c * SMEM_PER_SM / 1024:>11.0f}")

a_kb = b_kb = 128 * 512 * 2 / 1024   # fp16，想要 128x512 的 A tile 和 B tile
need = a_kb + b_kb
print(f"\nfp16 GEMM 想要单 stage {a_kb:.0f} KB (A) + {b_kb:.0f} KB (B) = {need:.0f} KB")
for c in (1, 2, 4, 8):
    per_block = need / c
    ok = "装得下" if per_block <= SMEM_PER_SM / 1024 else "装不下"
    print(f"  cluster={c}: 每个 block 只需 {per_block:.0f} KB  -> {ok}")

# ---------- 账 2：字节 ----------
print("\n=== 账 2：L2->SMEM 的字节数怎么随 cluster 形状摊薄 ===")
M = N = K = 4096
tile = 128
blocks_m = blocks_n = M // tile
n_blocks = blocks_m * blocks_n
a_tile = tile * tile * 2 / 1024 / 1024          # MiB，fp16
base = n_blocks * 2 * a_tile
print(f"GEMM {M}x{N}x{K}, fp16, tile {tile}x{tile}: {blocks_m}x{blocks_n} = {n_blocks} 个 block")
print(f"不复用时 L2->SMEM 总共要搬 {base:.0f} MiB\n")
print("cluster (cm,cn)   A 摊薄   B 摊薄   总字节(MiB)   相对无 cluster")
for cm, cn in [(1, 1), (1, 2), (2, 2), (2, 4), (4, 4), (8, 1)]:
    total = n_blocks * a_tile / cn + n_blocks * a_tile / cm
    print(f"{cm:>10d},{cn:<3d}    {cn:>6d}x  {cm:>6d}x  {total:>12.0f}  {base / total:>13.2f}x")

# ---------- 账 3：调度 ----------
print("\n=== 账 3：cluster 尺寸把 SM 槽位与 wave 尾部钉成离散值 ===")
print("假设每 SM 只放 1 个 block（tile 吃满 SMEM）")
print("cluster   槽位  闲置SM   grid=128    grid=130    grid=1024")
for c in (1, 8):
    slots = (SM_H100 // c) * c
    row = []
    for g in (128, 130, 1024):
        waves = -(-g // slots)
        util = (g - (waves - 1) * slots) / slots
        row.append(f"{waves}波/{util*100:>3.0f}%")
    print(f"{c:>7d}  {slots:>5d}  {SM_H100 - slots:>5d}  " + "  ".join(f"{r:>9s}" for r in row))
```

```text
=== 账 1：cluster 把共享内存池变大 ===
cluster 尺寸   池大小(KB)
           1          227
           2          454
           4          908
           8         1816
          16         3632

fp16 GEMM 想要单 stage 128 KB (A) + 128 KB (B) = 256 KB
  cluster=1: 每个 block 只需 256 KB  -> 装不下
  cluster=2: 每个 block 只需 128 KB  -> 装得下
  cluster=4: 每个 block 只需 64 KB  -> 装得下
  cluster=8: 每个 block 只需 32 KB  -> 装得下

=== 账 2：L2->SMEM 的字节数怎么随 cluster 形状摊薄 ===
GEMM 4096x4096x4096, fp16, tile 128x128: 32x32 = 1024 个 block
不复用时 L2->SMEM 总共要搬 64 MiB

cluster (cm,cn)   A 摊薄   B 摊薄   总字节(MiB)   相对无 cluster
         1,1           1x       1x            64           1.00x
         1,2           2x       1x            48           1.33x
         2,2           2x       2x            32           2.00x
         2,4           4x       2x            24           2.67x
         4,4           4x       4x            16           4.00x
         8,1           1x       8x            36           1.78x

=== 账 3：cluster 尺寸把 SM 槽位与 wave 尾部钉成离散值 ===
假设每 SM 只放 1 个 block（tile 吃满 SMEM）
cluster   槽位  闲置SM   grid=128    grid=130    grid=1024
      1    132      0    1波/ 97%    1波/ 98%    8波/ 76%
      8    128      4    1波/100%    2波/  2%    8波/100%
```

三笔账各有一句话的结论：

**账 1（容量）**：cluster=2 就把「要 256 KB 单 stage」这个不可能变成了可能。这是 cluster 最朴素、也最常被引用的价值 —— **它把 SMEM 的上限从 227 KB 抬到 8 × 227 KB**（前提是 Spread 调度，见 4.3）。

**账 2（字节）**：cluster 是**二维**的（`(cm, cn)` 两轴），A tile 沿 `cn` 轴被复现，B tile 沿 `cm` 轴被复现。`(4, 4)` 的 16-block cluster 把 64 MiB 打到 16 MiB，**4 倍**。注意 `(8, 1)` 只有 1.78 倍 —— **cluster 写成一条线是最差的写法**，因为只有一维在摊薄。这条结论很少有人讲清楚。

**账 3（调度）**：这是唯一的负账，也是最容易被忽视的。`grid=128` 时 cluster=8 拿到完美的 100% 利用率；但 `grid=130` 时变成 **2 波，最后一波只有 2% 的槽位有活**。多出来的 2 个 block 把整个 kernel 拉长了一倍。

---

## 七、把字节账换算成时间

账 2 少搬了 48 MiB，这值多少时间？用第 044 期在本机实测的 MLP 吞吐上限（32 条独立访存链达到的 66.6 GB/s）做锚：

```python
"""把账 2 省下的字节换算成时间。

66.6 GB/s 是第 044 期在本机实测的「多链并发访存」吞吐上限，
用它当锚，算的是「同样的访存效率在 H100 上能省多少时间」的形似换算，
不是 H100 的实测值 —— H100 的 HBM 带宽是 3.35 TB/s，比这个大 50 倍。
"""
bw = 66.6e9                      # B/s
for mib in (64, 16):
    sec = mib * 1024 * 1024 / bw
    print(f"{mib:>3d} MiB / 66.6 GB/s = {sec*1e3:>5.2f} ms")

print(f"\n省下的时间：{(64-16)*1024*1024/bw*1e3:.2f} ms（少了 {64/16:.0f} 倍字节）")
```

```text
 64 MiB / 66.6 GB/s =  1.01 ms
 16 MiB / 66.6 GB/s =  0.25 ms

省下的时间：0.76 ms（少了 4 倍字节）
```

这个换算的意思不是「H100 上能省 0.76 ms」，而是：**在访存受限的场景里，字节数减少多少，时间就几乎等比例减少多少**。GEMM 的 $4096^3$ 在 H100 上算完大约 0.14 ms（989.5 TFLOPS 峰值打满的理想值，实际更慢），所以 L2→SMEM 这 1.01 ms 根本不是小数 —— 它说明**在这种 shape 下，SMEM 的填充流量本身就是瓶颈候选之一**，而 cluster 是直接砍这个流量。

这也解释了为什么 CUTLASS 在 Hopper 上把 cluster 当成 GEMM 的默认配置项，而不是「高级优化选项」。

---

## 八、在 PyTorch 里怎么用

分三层，从「看不见」到「自己写」。

### 8.1 第一层：eager 模式下你碰不到它

我把本机 torch 2.14.0 的 CUDA 头文件全搜了一遍（`torch/include/ATen/` 下 `map_shared_rank` / `cluster_group` / `this_cluster` / `__cluster_dims__` 四个符号）：

```python
import subprocess, glob, os, torch

inc = os.path.join(os.path.dirname(torch.__file__), "include")
aten = os.path.join(inc, "ATen")
# 先确认真的扫到了东西 —— 否则下面一排 0 是自己没找到目录造出来的假证据
print(f"扫了 {len(glob.glob(os.path.join(aten, '**', '*.h'), recursive=True))} 个 ATen 头文件")

pats = ["map_shared_rank", "cluster_group", "this_cluster", "__cluster_dims__"]
print("符号                 ATen 头文件里命中的文件数")
for p in pats:
    r = subprocess.run(["grep", "-rl", p, aten], capture_output=True, text=True)
    hits = [l for l in r.stdout.splitlines() if l.strip()]
    print(f"{p:20s} {len(hits)}")
```

```text
扫了 8273 个 ATen 头文件
符号                 ATen 头文件里命中的文件数
map_shared_rank      0
cluster_group        0
this_cluster         0
__cluster_dims__     0
```

**ATen 一个都没用。** 也就是说 `torch.matmul`、`torch.nn.functional.scaled_dot_product_attention`、`layer_norm` 这些 eager 算子，没有一个走 cluster —— 它们跑的还是 Ampere 时代的 block 模型（cuBLAS/cuDNN 内部是否用 cluster 是闭源的，但 PyTorch 自己的 kernel 里没有）。

这符合 cluster 的定位：它是写给**手写 kernel** 的，不是写给算子库消费者的。

### 8.2 第二层：Triton 的 `num_ctas`

Triton 把 cluster 尺寸暴露成一个编译选项 `num_ctas`（Hopper 上生效）。Triton 的 `CUDAOptions` 里默认 `num_ctas=1`、`cluster_dims=(1,1,1)`，`num_ctas > 1` 时编译器会发射 `.reqnctapercluster` 指令并启用 CGA（Cooperative Grid Array，就是 cluster 在 Triton/PTX 里的叫法）相关的 lowering。

```py
import torch
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(A, B, C, M, N, K, sa0, sb0, sc0,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    pid_m, pid_n = pid // grid_m, pid % grid_m
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(A + (pid_m * BM + tl.arange(0, BM))[:, None] * sa0
                      + (k * BK + tl.arange(0, BK))[None, :])
        b = tl.load(B + (k * BK + tl.arange(0, BK))[:, None] * sb0
                      + (pid_n * BN + tl.arange(0, BN))[None, :])
        acc += tl.dot(a, b)
    tl.store(C + (pid_m * BM + tl.arange(0, BM))[:, None] * sc0
               + (pid_n * BN + tl.arange(0, BN))[None, :], acc.to(C.dtype.element_ty))

# 关键就是 num_ctas：它把每 8 个 program（CTA）打包成一个 cluster。
# 只有 SM90+ 生效，num_ctas=1 就是完全关掉。
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
gemm_kernel[grid](A, B, C, M, N, K, A.stride(0), B.stride(0), C.stride(0),
                  BM=128, BN=256, BK=64,
                  num_warps=8, num_stages=3, num_ctas=8)
```

这段**没有在本机跑过** —— 本机装的 torch 2.14.0 里没有 triton（`import triton` 直接 `ModuleNotFoundError`），也没有 CUDA 设备，所以连 triton 源码我也核不了。`num_ctas` / `cluster_dims` 这两个名字来自 Triton 公开的 `CUDAOptions` 定义与官方 multi-CTA 教程（Triton 文档把 cluster 叫 CGA，上限写的是 16 个 CTA）。**这一节的证据等级是「文档级」，不是「源码级」也不是「实测级」。**

需要注意：Triton 直接把 cluster 尺寸给你了，但**它不保证你的 kernel 因此变快**。`num_ctas=8` 真正的收益需要 kernel 内部配合 —— 要么让 TMA 的 multicast 吃掉复用（Triton 在 Hopper 上对 `tl.load` 走 TMA 时会考虑 CGA layout），要么用 Gluon 那一层的 `cga_layout` 显式把 tile 切给多个 CTA。**光把 `num_ctas` 从 1 改成 8，很多时候只是让调度变差。**

### 8.3 第三层：Inductor 替你开了吗？没有

我把本机 torch 2.14.0 的 Inductor 源码关于 `num_ctas` 的所有出现都看了一遍，结论很干脆：

| 位置 | 干了什么 |
|---|---|
| `_inductor/runtime/triton_heuristics.py:3822` `def triton_config(...)` | 参数表里**没有** `num_ctas` |
| `_inductor/runtime/triton_heuristics.py:3245` | 生成 launcher 时从 `binary.num_ctas` / `binary.metadata.num_ctas` **读**出来当 `cta_args` |
| `_inductor/codegen/cpp_wrapper_gpu.py:896` | `num_ctas = params.get("config", {}).get("num_ctas", 1)` —— 只用来算 `global_scratch` 要乘几倍 |
| `_inductor/runtime/static_triton_launcher.py:149` | `if num_ctas != 1: raise ... "Static cuda launcher only supports num_ctas == 1"` |

也就是说：**Inductor 构造 autotune 配置的地方根本不提供 `num_ctas` 这个旋钮，它只在「别人给了」的时候负责把它传下去**。`torch.compile` 生成的 kernel（包括 matmul 模板）在 torch 2.14.0 上都是 `num_ctas=1`。

这是本期最实用的一条结论：

> **想吃到 cluster，`torch.compile` 帮不了你。你得自己写 Triton kernel（并显式 `num_ctas`）或者自己写 CUDA kernel。** 这不是 Inductor 疏漏，是因为 cluster 的收益强依赖 tile 形状与 grid 对齐，通用 autotune 很难替你做这个决策 —— 账 3 那个 `grid=130 / cluster=8 → 2% 利用率` 就是反例。

---

## 九、同一件事的另一种做法：cluster 上的通信原语

cluster 不光能共享 SMEM，还能**只在一个 cluster 内同步**。这把第 007 期讲的通信原语整个搬到了片上。

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// cluster 内的 ring allreduce：每个 block 只和自己前后两个邻居说话
__global__ void __cluster_dims__(8, 1, 1) ring_allreduce(float* g, int n) {
    cg::cluster_group cluster = cg::this_cluster();
    extern __shared__ float smem[];

    int rank = cluster.block_rank();
    int nblk = cluster.num_blocks();
    // 每个 block 把自己负责的 1/nblk 段读进 SMEM
    for (int i = threadIdx.x; i < n / nblk; i += blockDim.x)
        smem[i] = g[rank * (n / nblk) + i];
    cluster.sync();                       // 保证所有 block 的 SMEM 都就位

    int next = (rank + 1) % nblk;
    int prev = (rank - 1 + nblk) % nblk;
    float* to_next   = cluster.map_shared_rank(smem, next);
    float* from_prev = cluster.map_shared_rank(smem, prev);
    // 后续 reduce-scatter / all-gather 就在这里直接读写邻居的 SMEM，
    // 全程不落 global memory
    ...
    cluster.sync();                       // 收尾：读完之前谁也不许退出
}
```

这段是**示意**，我没有编译（本机无 nvcc），reduce-scatter/all-gather 的循环体略去了。它的价值在于说明**量级**：一个 8-block 的 ring allreduce，走 DSMEM 是 181 cycle 一档、3.28 TB/s 一档；走 global memory 中转是 1000 cycle 一档。同一份逻辑，差一个数量级。

这就是第 019 期（张量并行）和第 007 期（AllReduce）在**卡内**的翻版。区别在于：跨卡靠 NVLink + NCCL/IFCCL，卡内靠 GPC 的 SM-to-SM 网络。**两层的拓扑、带宽、同步语义都不一样，不能互相套用。**

---

## 十、围绕这个领域展开

这一节把 cluster 放回它在 PyTorch 生态里的位置，以及它和已发各期的连线。

### 10.1 与第 043 期（内存层次）的关系：多了一档

043 期画的是「寄存器 → SMEM → L1 → L2 → HBM」的带宽阶梯，结论是每降一档慢 5~10 倍。cluster 做的事，是在 **SMEM 和 L2 之间插进去一档**：

```text
29 cycle     SMEM（本地）          —— 自己的工具台
181 cycle    DSMEM（邻居的 SMEM）  —— 车间的传送带      ← cluster 新增的一档
~1000 cycle  global（经 L2 中转）  —— 中央仓库
```

同时它把「这层存储属于谁」的粒度从 **SM** 改成了 **GPC**。043 期讲的访存合并、32 B 段对齐这些规则，在这一档里**依然适用** —— 只是规则集换成了 global 的那一套（见 5.2）。

### 10.2 与第 044 期（异步拷贝）的关系：多了一个收货地址

044 期把 cp.async / TMA 讲成「一条生产者-消费者流水线」：TMA 引擎把 tile 从 HBM 搬到 SMEM，mbarrier 通知消费者，`num_stages` 决定几级缓冲。

cluster 给这条流水线加的不是「更快的搬运」，而是**一种新的目的地**。PTX 里那条指令长这样（名称按 PTX ISA 文档，我未在本机编译）：

```text
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    .multicast::cluster [dstSmem], [tensorMap, {x, y}], [mbar], ctamask;
```

注意 `.multicast::cluster` 和最后一个 `ctamask` 参数 —— 一个 bitmask，指定这次搬运的结果要投递到 cluster 里哪几个 block 的 SMEM。**一次 HBM/L2 读取，多个 SMEM 副本**。

CUDA 的分析文章给出了它的理想上限：**显式 TMA multicast 可以让 L2 流量降到「每字节 SMEM 数据只对应 $1/\text{cluster\_size}$ 的 L2 流量」**。这正好是账 2 里那个摊薄系数。

所以 044 期的 `num_stages` 旋钮，在 cluster 语境下多了一个上游约束：**stage 的 SMEM 预算由 cluster 内所有 block 分担**，这直接改变了「能开几级流水」的算术。044 期算过的「227 KB / 每 stage 32 KB → 7 级」在 cluster=8 时变成「1816 KB / 每 stage 32 KB → 56 级」（理论值，实际会被别的资源卡住）。

### 10.3 与第 045 期（SIMT 与 Occupancy）的关系：第四把锁

045 期讲了决定 occupancy 的三把锁：寄存器、SMEM、线程数。cluster 尺寸是**第四把**，而且它的锁法不一样：

- 前三把锁都是「每个 block 消耗多少资源」，消耗少就多放几个；
- cluster 锁的是「多少个 block 必须**同时**有位置」。当每 SM 只放 1 个 block（tile 吃满 227 KB SMEM）时，一个 cluster=8 的组需要 **8 个不同的 SM 同时空出来**。

总槽位数没变（128 个 block），但**可用性从「任意一个空位」变成「连续的 8 个空位」** —— 碎片化。这就是账 3 里 `grid=130` 那个灾难的来源：最后 2 个 block 凑不出一个完整的 cluster，只能单独占一波。

045 期的结论「occupancy 不是越高越好，它的作用是藏延迟」在这里依然成立，只是计数单位从 block 变成了 cluster 组。

### 10.4 与第 040 期（算子融合）的关系：融合不了的就搬近一点

040 期的融合是在**一个 kernel 内**把多个算子合成一段循环，省掉中间张量的 global 往返。cluster 解决的是融合解决不了的那部分：**当中间结果大到装不进一个 block 的 SMEM 时**，融合就断了。

cluster 给了一条中间路线：不用合成一个 kernel，让两个 block 各自算一段，边界数据通过 DSMEM 直接递过去，不落 global。这比「融合成一个巨大 kernel」灵活，比「落 global 再读回来」快 5~6 倍。

### 10.5 与第 038 期（SDPA 后端）的关系：一条没有用上的捷径

attention 的结构天生适配 cluster —— 同一个 head 的 Q tile 要配所有 K/V tile，`cluster` 里几个 block 各持一段 K/V，Q 通过 DSMEM 广播出去，能省掉大量重复的 SMEM 填充。

**但我要说清楚证据等级**：我没有查到 PyTorch 的 SDPA 三个后端（flash / mem-efficient / math）在 Hopper 上用 cluster 的公开证据；`torch/include/ATen/` 里四个 cluster 符号的命中数都是 0（8.1 节）。所以这一条是**「结构上天然适配」的推断，不是「已经这么做了」的事实**。FA3 在 Hopper 上的主要武器是 warp specialization + TMA + wgmma，这三个都不是 cluster。

---

## 十一、什么时候该用 / 不该用

**该用：**

| 场景 | 判据 |
|---|---|
| 单 block 的 SMEM 装不下想要的 tile | 账 1：cluster=2 就能把 256 KB 的单 stage 需求变成 128 KB |
| 一份数据被多个相邻 block 同时需要 | 账 2：复用度 ≥ 4 时收益开始显著 |
| 跨 block 的边界交换（stencil 的 halo、GEMM 的 panel、scan 的 partial） | 交换量小但同步频繁，DSMEM 的 181 cycle 比 1000 cycle 划算 |
| 已经有 TMA 流水线，想把 L2 流量再砍一刀 | 显式 multicast 能把 L2 流量降到 $1/\text{cluster\_size}$ |

**不该用：**

| 场景 | 原因 |
|---|---|
| **grid 不能被 cluster 尺寸整除，或 grid 很小** | 账 3：`grid=130 / cluster=8` 直接变 2 波，末波 2% 利用率 |
| **每个 block 的 SMEM 用量已经很大（occupancy 本来就 1）** | cluster 要求「连续 N 个 SM 同时空闲」，碎片化雪上加霜 |
| **跨 block 通信量很小、复用度低** | cluster 的固定成本（更贵的 barrier、调度不弹性）收不回来 |
| **只有一个消费者** | multicast 没意义，DSMEM 反而比本地 SMEM 慢 6 倍 |
| **消费级 Blackwell（如 RTX 50 系）** | cluster 被限制到「最多 1 个 block」，等于关掉（*The CUDA Handbook* 的说法） |
| **想靠 `torch.compile` 自动吃到** | 8.3 节：Inductor 在 torch 2.14.0 上不提供 `num_ctas` |

一句话判据：**「这份数据的复用度 ≥ cluster 尺寸，且 grid 能被 cluster 整除」两个条件同时成立才值得开。** 少一个都可能是负收益。

---

## 十二、常见坑

**坑 1：按 SMEM 的规则去访问 DSMEM。**
你的直觉是「避免 bank conflict、用交错访问」，但 DSMEM 的 load 是**按包传输的**，规则和 global memory 一样：要合并、要 32 B 段对齐、要避免非单位步幅。更要命的是 `ld.shared::cluster` 访问**本地** SMEM 也拿不到 128 B/clk 的峰值（实测 205 GB/s vs 225 GB/s，掉 20%）。所以：**本地数据用 `ld.shared`，只有当指针确实指向别的 block 时才用 `::cluster` 变体。**

**坑 2：漏掉第二次 `cluster.sync()`。**
远程指针的合法性依赖「目标 block 还活着」。block 退出时它的 SMEM 会被释放，此时别的 block 还在读 —— 结果未定义，而且**不会有任何报错**。规则是：所有远程访问必须被 `cluster.sync()` 从两边夹住，前面那次保「已就位」，后面那次保「别提前走」。

**坑 3：cluster 尺寸写死，grid 没对齐。**
账 3 的 `grid=130`：多出 2 个 block，kernel 时间翻倍。正确做法是把 grid 向上取整到 cluster 尺寸的整数倍，多出来的 block 直接 `return`（并且**不要**参与 `cluster.sync()`，否则死锁 —— 这是个更深的坑：cluster barrier 要求全 cluster 的所有线程都到达）。

**坑 4：以为 cluster 越大越快。**
论文实测：DSMEM 总吞吐 cluster=2 时 3.28 TB/s，cluster=4 掉到 2.78 TB/s（-15%）。SM-to-SM 网络是 GPC 内共享的一条通路，block 越多抢得越凶。另外 cluster 越大，`cluster.sync()` 越贵（等最慢的那个），本机测的同步域形状也是这样：

```python
"""同步成本随「同步域宽度」怎么长。

每个线程做的同步次数完全相同（2000 次），唯一变量是「一个同步域里有几个线程」。
注意：这组数字测的是 Python 线程的唤醒级联（受 GIL 主导），
绝对值没有硬件意义，能带走的只有**形状**：域从 2 宽涨到 16 宽，成本涨三倍多。
"""
import threading
import time


def worker(barrier, rounds, out, slot):
    for _ in range(200):
        barrier.wait()                       # 预热
    t0 = time.perf_counter_ns()
    for _ in range(rounds):
        barrier.wait()
    out[slot] = time.perf_counter_ns() - t0


def trial(n_total, n_groups, rounds=2000):
    width = n_total // n_groups
    out = [0] * n_total
    if width == 1:                           # 域宽 1，没有同步，只测空转
        def busy(slot):
            t0 = time.perf_counter_ns()
            for _ in range(rounds):
                pass
            out[slot] = time.perf_counter_ns() - t0
        ts = [threading.Thread(target=busy, args=(i,)) for i in range(n_total)]
    else:
        barriers = [threading.Barrier(width) for _ in range(n_groups)]
        ts = [threading.Thread(target=worker,
                               args=(barriers[i % n_groups], rounds, out, i))
              for i in range(n_total)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return max(out) / rounds                 # 同步是等所有人，看最慢者


print("总线程  组数  域宽  每次同步(us)")
res = {}
for n_total, n_groups in [(8, 8), (8, 4), (8, 2), (8, 1), (16, 1)]:
    ns = min(trial(n_total, n_groups) for _ in range(3))
    res[n_total // n_groups] = ns / 1000
    print(f"{n_total:>6d}  {n_groups:>4d}  {n_total // n_groups:>4d}  {ns / 1000:>11.1f}")

print(f"\n域宽 2 -> 16（8 倍）：成本 {res[2]:.1f} -> {res[16]:.1f} us，"
      f"涨 {res[16] / res[2]:.1f} 倍")
```

```text
总线程  组数  域宽  每次同步(us)
     8     8     1          0.0
     8     4     2         41.5
     8     2     4         63.4
     8     1     8         76.8
    16     1    16        155.4

域宽 2 -> 16（8 倍）：成本 41.5 -> 155.4 us，涨 3.7 倍
```

域宽从 2 涨到 16（宽度 8 倍），同步成本涨 **三倍多**（具体倍数印在输出的最后一行）。Hopper 上的 `cluster.sync()` 有同样的形状 —— 只是绝对值在几十到几百 cycle，而不是几十微秒。

**坑 5（工程细节）：用 non-portable 的 16 忘了开权限。**
`cudaFuncAttributeNonPortableClusterSizeAllowed` 不设，launch 直接失败。而且就算设了，也要先用 `cudaOccupancyMaxPotentialClusterSize()` 查一遍这块卡这个 kernel 到底能给多少 —— 返回 16 不代表你的 kernel 用得上 16（SMEM 占用可能把它压到 4）。

---

## 十三、一句话总结

> **cluster 是一句硬件承诺 ——「这 8 个 block 同时活着、都在同一个 GPC 里」；DSMEM 是这句承诺的兑现方式：把 8 个 SM 的 shared memory 接成一片，于是「装不下就退回 L2」多了一条 181 cycle 的中间路。它的收益全在「少搬字节」和「少等一趟 L2」，代价全在「调度不再弹性」—— 所以 cluster 尺寸不是一个大就好的旋钮，而是一个必须和 grid 对齐、和复用度匹配的离散档位。**

---

## 今日练习

<details><summary>今日练习：给定 shape 和 tile，判断该不该开 cluster</summary>

**题目**：你要在 H100（132 SM，227 KB SMEM/block）上跑 fp16 GEMM，$M=N=K=8192$。

1. 若 tile 固定 $128 \times 128$，grid 是多少个 block？开 `cluster=(2,4)` 时 L2→SMEM 的字节数从多少降到多少？
2. 这个 grid 能不能被 cluster 尺寸整除？会不会有 wave 尾部损失？
3. 若把 tile 放大到 $256 \times 128$（$K$ 方向块大小 64），单 block 单 stage 要多少 SMEM？3 级流水装得下吗？装不下的话，cluster 至少要多大？
4. 综合上面三条，你会怎么选？

**参考答案**：

```python
SM, SMEM = 132, 227 * 1024
M = N = K = 8192
fp16 = 2

# 1. tile 128x128
tile = 128
bm = bn = M // tile
nb = bm * bn
a_tile_mib = tile * tile * fp16 / 1024 / 1024
base = nb * 2 * a_tile_mib
cm, cn = 2, 4
multi = nb * a_tile_mib / cn + nb * a_tile_mib / cm
print(f"1) grid = {bm}x{bn} = {nb} block")
print(f"   无 cluster: {base:.0f} MiB -> cluster(2,4): {multi:.0f} MiB"
      f"（{base/multi:.2f}x）")

# 2. grid 能否被 cluster 整除（cluster=2*4=8 个 block 一组）
csize = cm * cn
print(f"2) cluster 尺寸 {csize}，{nb} % {csize} = {nb % csize}，"
      f"{'整除，无尾部损失' if nb % csize == 0 else '不整除，有尾部损失'}")
slots = (SM // csize) * csize
print(f"   槽位 {slots}，需要 {-(-nb // slots)} 波")

# 3. tile 256x128，BK=64
T, BK = 256, 64
stage_kb = (T * BK + T * BK) * fp16 / 1024
print(f"3) 单 stage = {stage_kb:.0f} KB，3 级 = {stage_kb*3:.0f} KB，"
      f"{'装得下' if stage_kb*3 <= SMEM/1024 else '装不下'}"
      f"（上限 {SMEM/1024:.0f} KB）")
for c in (2, 4, 8):
    print(f"   cluster={c}: 每 block 只需 {stage_kb*3/c:.0f} KB -> "
          f"{'装得下' if stage_kb*3/c <= SMEM/1024 else '装不下'}")
```

真实输出（本机跑过）：

```text
1) grid = 64x64 = 4096 block
   无 cluster: 256 MiB -> cluster(2,4): 96 MiB（2.67x）
2) cluster 尺寸 8，4096 % 8 = 0，整除，无尾部损失
   槽位 128，需要 32 波
3) 单 stage = 64 KB，3 级 = 192 KB，装得下（上限 227 KB）
   cluster=2: 每 block 只需 96 KB -> 装得下
   cluster=4: 每 block 只需 48 KB -> 装得下
   cluster=8: 每 block 只需 24 KB -> 装得下
```

**第 4 问的答案**：这个 shape 上我**不会**为了容量开 cluster —— 因为 $256 \times 128$ 的 3 级流水（192 KB）已经装得下一个 block 了。开 cluster 的理由只剩「字节数」（256 → 96 MiB，2.67 倍）。而 4096 个 block 能被 8 整除、槽位 128、正好 32 波，尾部成本为零，所以开是对的。

**关键判断顺序**：先看容量（够不够），再看字节（复用度够不够），最后看调度（grid 对齐）。**三步里任何一步是负的，就不要开** —— 账 3 那个 `grid=130` 的例子就是「容量和字节都赚，调度亏光」的典型。

</details>
