# PyTorch 每日一课 · 第 051 期

## $P$ 到底是什么 —— occupancy 如何决定 `ctas_per_wave`

**元信息**

| | |
|---|---|
| 日期 | 2026-11-09 |
| 难度 | ⭐⭐⭐⭐ |
| 前置知识 | 第 045 期（SIMT 与 occupancy）、第 043 期（内存层次）、第 047 期（Tensor Core 的寄存器预算） |
| 预计阅读 | 40 分钟 |
| 关联 | 第 050 期 · 那条「$P$ 会翻倍」的话没有下文，本期把它算完 |

---

## 一、这个领域解决什么问题

第 050 期讲 Stream-K 的时候，有一个参数从头用到尾：$P$，一个 wave 里能同时驻留多少个 CTA。文里有一句话，原文是这么写的：

> 如果 occupancy 允许一个 SM 驻留 2 个 CTA，$P$ 就是 $2\times$ SM 数，$\lambda$ 直接减半。

这句话是对的，但它停在半路。$\lambda$ 减半之后呢？总时间会不会跟着减半？如果会，那所有 GEMM 只要把 tile 缩小一半就能白赚一倍性能——显然不可能。如果不会，那 050 那一整套判据（$\lambda = T/P$、$\eta = \lambda/\lceil\lambda\rceil$、尾波该不该做 stream-K）到底在优化什么？

要回答这个，得先把三个问题拆开：

1. **一个 SM 到底能驻留几个 CTA？** 045 期讲的是「一个 SM 能挂多少个 warp」，那是同一个问题的另一面，但 045 从来没有把它和 tile 尺寸联系起来。
2. **我把 tile 缩小、让 P 翻倍，赚到了什么？** 这是 050 留下的空白。
3. **同一块 SM 上的两个 CTA，真的在并行吗？** 如果它们其实是抢同一批执行单元，那「P 翻倍」这件事的性质就完全变了。

这三个问题看上去只是调度细节。其实不是——第 049 期的 split-K 判据 $S^* = \sqrt{2K\cdot BW/(P_{sm}b)}$、第 050 期的 $\lambda$ 与尾波闸门、第 046 期 cluster 的「调度不再弹性」，全建在 $P$ 这个数上。**$P$ 取错，判据全部反过来。**

---

## 二、核心思想：$P$ 是一个商，不是一个旋钮

最常见的误解是把 occupancy 当成一个可以"调"的东西：tile 大了就调小一点，occupancy 就上去了。

更准确的画面是宿舍分床位。一间宿舍能住几个人？

$$\text{住几个人} = \min(\text{床位总数}, \text{衣柜数}, \text{门禁名额}, \text{房间上限})$$

你不能"调"这个数，你只能改**每个人带多少行李**，然后由上面这个 $\min$ 算出结果。GPU 上一个 SM 就是一间宿舍，CTA 是住进来的人，四条限制是：

| 限制 | H100 的值 | 谁在消耗它 |
|---|---|---|
| warp 槽位 | 64 个 warp / SM（= 2048 线程） | 每个 CTA 的线程数 |
| 寄存器池 | 65536 个 32-bit 寄存器 / SM | 每线程寄存器数 × 线程数 |
| shared memory | 228 KiB / SM（单 block 上限 227 KiB） | 每 stage 的操作数缓冲 × 流水线级数 |
| block 槽位 | 32 个 / SM | 纯粹的硬上限 |

于是

$$P = SM \times \left\lfloor \min\left(\frac{64}{w_{blk}},\ \frac{65536}{32\,w_{blk}\,r},\ \frac{227\text{K}}{s_{blk}},\ 32\right) \right\rfloor$$

其中 $w_{blk}$ 是每 CTA 的 warp 数，$r$ 是每线程寄存器数，$s_{blk}$ 是每 CTA 的 shared memory 字节数。

**关键的一点在这里**：GEMM 的 tile 尺寸一动，这三把锁**同时**变。tile 面积减半 → 累加器寄存器减半（锁 2 松了）；但每 CTA 的 warp 数也减半（锁 1 松了），而流水线的 stage 缓冲变小意味着你可以塞进更多 stage，于是 shared memory 的占用几乎不变（锁 3 没动）。

结果就是本节要证明的东西：**$P$ 会涨，但 $P$ 涨的原因是工作量被切得更碎了，不是机器变强了。** 这两件事在数学上完全抵消。下面把它算出来。

---

## 三、把账算出来

先明确假设，后面所有数字都从这几个假设推出来（不是实测，是确定性算术，可以逐位复现）：

- 硬件取 H100 SXM5：132 SM、每 SM 65536 寄存器、228 KiB 可配置 shared memory、64 个 warp 槽、32 个 block 槽。
- tile 用 Hopper 的惯例：一个 warpgroup（128 线程）负责 $M=64$，所以 $M_t=128$ 的 tile 用 256 线程。
- 累加器是 fp32，1 个累加器 = 1 个寄存器。非累加器开销（地址、循环变量、描述符、stage 指针）按累加器的 $\rho = 40\%$ 估。
- 寄存器按 8 个一组分配、shared memory 按 128 B 粒度分配（CUDA 的 occupancy calculator 就是这么算的），不按需分配。

### 3.1 先解释一个一直被当成"规定"的数字：$N \le 256$

CUTLASS 的 GEMM tile 里，$N_t$ 几乎从不超过 256。这看着像惯例，其实是被一个硬上限逼出来的：

```python
REG_SM = 65536          # 每 SM 的 32-bit 寄存器总数
REG_MAX = 255           # 每线程寄存器硬上限，ptxas 收不下更多
SM = 132                # H100 SXM5 的 SM 数
WARP_SLOT = 64          # 每 SM 的 warp 槽位（= 2048 线程）


def ru(a, g):
    """向上取整到 g 的倍数：寄存器和共享内存都按粒度分配，不按需分配"""
    return (a + g - 1) // g * g


print("== 每线程累加器数 = N/2（Mt=128、256 线程时），够不够用 255 个寄存器 ==")
print(f"{'N':>5} {'累加器/线程':>12} {'含开销寄存器':>12} {'合法':>6}")
for nt in (128, 192, 256, 288, 512):
    acc = nt // 2                     # accum = Mt*Nt/threads = Nt/2
    regs = ru(int(acc * 1.4), 8)
    print(f"{nt:>5} {acc:>12} {regs:>12} {'是' if regs <= REG_MAX else '否':>6}")
```

```text
== 每线程累加器数 = N/2（Mt=128、256 线程时），够不够用 255 个寄存器 ==
    N       累加器/线程       含开销寄存器     合法
  128           64           96      是
  192           96          136      是
  256          128          184      是
  288          144          208      是
  512          256          360      否
```

$N_t=512$ 时每线程要 360 个寄存器，物理上不可能。**这就是 $N \le 256$ 的来源：不是选出来的，是被 255 这个数按住的。**

同一套算术还能给出 **tile 面积**的上限。一个 CTA 自己必须先装进寄存器池（不是几条 CTA 分摊，是它单独就得装下），所以

$$2M_t \times \frac{(1+\rho)M_tN_t}{2M_t} = (1+\rho)M_tN_t \le 65536 \quad\Rightarrow\quad M_tN_t \le 46811$$

$128\times256 = 32768$ 装得下，$256\times256 = 65536$ 装不下。**这就是为什么 CUTLASS 的 tile 面积几乎封顶在 $128\times256$——它同时贴着两条墙：$N\le256$ 贴着每线程 255 个寄存器，面积 $\le 46811$ 贴着每 CTA 65536 个寄存器。**

### 3.2 一个 SM 能驻留几个 CTA

把四条锁都算出来，取 $\min$：

```python
REG_SM = 65536
SMEM_SM = 228 * 1024    # 每 SM 可配置 shared memory 上限（字节）
SMEM_RESERVED = 1024    # 驱动/运行时保留
WARP_SLOT = 64
BLOCK_SLOT = 32
SM = 132


def ru(a, g):
    return (a + g - 1) // g * g


def plan(mt, nt, kt=64, stages=4, rho=0.4, esize=2):
    """给一个 GEMM tile，算它在一个 SM 里能驻留几个 CTA。

    mt/nt/kt : 输出 tile 的 M/N 与 K 方向单 stage 深度
    stages   : 软件流水线级数（每级一份 (mt+nt)*kt 的操作数缓冲）
    rho      : 非累加器开销 / 累加器寄存器
    esize    : 操作数字节数（fp16 = 2）
    """
    threads = 2 * mt                  # 一个 warpgroup(128 线程) 负责 M=64
    warps = threads // 32
    accum = mt * nt // threads        # 每线程 fp32 累加器数 = 占用的寄存器数
    regs = ru(int(accum * (1 + rho)), 8)
    smem = (mt + nt) * kt * esize * stages

    b_reg = REG_SM // (warps * 32 * regs)
    b_smem = (SMEM_SM - SMEM_RESERVED) // smem
    b_warp = WARP_SLOT // warps
    b = min(b_reg, b_smem, b_warp, BLOCK_SLOT)
    lock = ["寄存器", "SMEM", "warp 槽", "block 上限"][
        [b_reg, b_smem, b_warp, BLOCK_SLOT].index(b)]
    return dict(mt=mt, nt=nt, threads=threads, warps=warps, accum=accum, regs=regs,
                smem=smem, b_reg=b_reg, b_smem=b_smem, b_warp=b_warp, b=b, lock=lock)


print("== 一个 SM 能驻留几个 CTA（K 方向单 stage 64，fp16 操作数，4 级流水）==")
print(f"{'tile':>9} {'线程':>4} {'累加器':>6} {'regs':>5} {'smem/KiB':>9} "
      f"{'卡在哪':>7} {'blocks':>7} {'P':>5} {'活跃warp':>8} {'occ%':>5}")
TILES = [(128, 256), (128, 128), (128, 64), (64, 64), (256, 128)]
for mt, nt in TILES:
    p = plan(mt, nt)
    print(f"{f'{mt}x{nt}':>9} {p['threads']:>4} {p['accum']:>6} {p['regs']:>5} "
          f"{p['smem']//1024:>9} {p['lock']:>7} {p['b']:>7} {p['b']*SM:>5} "
          f"{p['b']*p['warps']:>8} {p['b']*p['warps']/WARP_SLOT*100:>5.1f}")
```

```text
== 一个 SM 能驻留几个 CTA（K 方向单 stage 64，fp16 操作数，4 级流水）==
     tile   线程    累加器  regs  smem/KiB     卡在哪  blocks     P   活跃warp  occ%
  128x256  256    128   184       192     寄存器       1   132        8  12.5
  128x128  256     64    96       128    SMEM       1   132        8  12.5
   128x64  256     32    48        96    SMEM       2   264       16  25.0
    64x64  128     32    48        64    SMEM       3   396       12  18.8
  256x128  512     64    96       192     寄存器       1   132       16  25.0
```

三件事值得停下来看：

1. **$128\times256$ 的 occupancy 只有 12.5%**，卡在寄存器（184 regs × 256 线程 × 1 block = 47104 个寄存器，再放第二个就要 94208 > 65536）。这和 045 期、048 期看到的现象对得上：CUTLASS 的 Hopper GEMM 就是这个数，而它的性能接近峰值。**12.5% 的 occupancy 不是缺陷。**
2. **$128\times128$ 的 occupancy 还是 12.5%**——寄存器松了，但 4 级流水的 shared memory 从 192 KiB 降到 128 KiB，而 228 KiB 只装得下**一个** 128 KiB。**换了一把锁合上而已。**
3. 真正涨到 2 个 CTA 的是 $128\times64$，此时 SMEM 掉到 96 KiB。再看一眼算术强度会掉多少——这个代价我们后面算。

### 3.3 两条预算的解析式，$M_t$ 全部约掉

上面那张表可以化成两个闭式。寄存器那条：

$$b \le \frac{65536}{(1+\rho)\,M_tN_t}$$

推导只有一步：$b \cdot w_{blk}\cdot 32\cdot r \le 65536$，而 $w_{blk}\cdot 32 = 2M_t$、$r = (1+\rho)M_tN_t/(2M_t)$，两个 $2M_t$ 直接约掉，**$M_t$ 消失了**。

shared memory 那条同理，$K_t=64$、fp16 时：

$$b\cdot\text{stages}\cdot(M_t+N_t) \le \frac{227\text{KiB}}{64\times 2} = 1816$$

这两条式的物理含义完全不同：**寄存器那条管的是 tile 的"面积"，SMEM 那条管的是 tile 的"周长"**。面积决定你能同时算多少输出，周长决定你每个 stage 要搬多少操作数——而 $\text{面积}/\text{周长}$ 正好就是 043 期那个算术强度 $M_tN_t/2(M_t+N_t)$。

```python
REG_SM = 65536
SMEM_SM = 228 * 1024
SMEM_RESERVED = 1024
WARP_SLOT = 64
BLOCK_SLOT = 32
SM = 132


def ru(a, g):
    return (a + g - 1) // g * g


def plan(mt, nt, kt=64, stages=4, rho=0.4, esize=2):
    threads = 2 * mt
    warps = threads // 32
    accum = mt * nt // threads
    regs = ru(int(accum * (1 + rho)), 8)
    smem = (mt + nt) * kt * esize * stages
    b_reg = REG_SM // (warps * 32 * regs)
    b_smem = (SMEM_SM - SMEM_RESERVED) // smem
    b_warp = WARP_SLOT // warps
    b = min(b_reg, b_smem, b_warp, BLOCK_SLOT)
    lock = ["寄存器", "SMEM", "warp 槽", "block 上限"][
        [b_reg, b_smem, b_warp, BLOCK_SLOT].index(b)]
    return dict(mt=mt, nt=nt, b=b, lock=lock)


print("== 两条预算的解析式（Mt 全部约掉，只剩 tile 面积和周长）==")
print("  寄存器池：blocks <= 65536 / ((1+rho) * Mt * Nt)")
print("  SMEM 池 ：blocks * stages * (Mt + Nt) <= 227KiB / (Kt * esize) = 1816")
print(f"{'tile':>9} {'寄存器上界':>10} {'SMEM 上界':>9} {'实际 blocks':>11} {'卡在哪':>7}")
for mt, nt in [(128, 256), (128, 128), (128, 64), (64, 64), (256, 128)]:
    p = plan(mt, nt)
    print(f"{f'{mt}x{nt}':>9} {REG_SM/(1.4*mt*nt):>10.2f} "
          f"{1816/(4*(mt+nt)):>9.2f} {p['b']:>11} {p['lock']:>7}")
```

```text
== 两条预算的解析式（Mt 全部约掉，只剩 tile 面积和周长）==
  寄存器池：blocks <= 65536 / ((1+rho) * Mt * Nt)
  SMEM 池 ：blocks * stages * (Mt + Nt) <= 227KiB / (Kt * esize) = 1816
     tile      寄存器上界   SMEM 上界   实际 blocks     卡在哪
  128x256       1.43      1.18           1     寄存器
  128x128       2.86      1.77           1    SMEM
   128x64       5.71      2.36           2    SMEM
    64x64      11.43      3.55           3    SMEM
  256x128       1.43      1.18           1     寄存器
```

表里有个细节值得注意：$128\times256$ 的寄存器上界算出来是 **1.43**，SMEM 上界 **1.18**，实际取 1。而 $128\times128$ 的寄存器上界是 **2.86**，SMEM 上界只有 **1.77**——**寄存器先松手了，SMEM 顶住了**。这就是 3.2 节第 2 条现象的定量版本。

### 3.4 守恒律：$P$ 涨了，总时间一动不动

现在回答 050 留下的问题。设问题固定为 $8192^3$，每 CTA 的完工时间 $t_{tile} \propto M_tN_t\cdot b$（因为一个 SM 的算力被 $b$ 个 CTA 平分），总时间 $= \lceil T/P\rceil \cdot t_{tile}$：

```python
REG_SM = 65536
SMEM_SM = 228 * 1024
SMEM_RESERVED = 1024
WARP_SLOT = 64
BLOCK_SLOT = 32
SM = 132


def ru(a, g):
    return (a + g - 1) // g * g


def plan(mt, nt, kt=64, stages=4, rho=0.4, esize=2):
    threads = 2 * mt
    warps = threads // 32
    accum = mt * nt // threads
    regs = ru(int(accum * (1 + rho)), 8)
    smem = (mt + nt) * kt * esize * stages
    b = min(REG_SM // (warps * 32 * regs), (SMEM_SM - SMEM_RESERVED) // smem,
            WARP_SLOT // warps, BLOCK_SLOT)
    return dict(b=b)


M = N = 8192
TILES = [(128, 256, dict()), (128, 128, dict()), (128, 64, dict()),
         (64, 64, dict())]

print(f"== 守恒律：问题 {M}x{N}x8192，P 从 132 涨到 396，总时间一动不动 ==")
print(f"{'tile':>9} {'T':>6} {'P':>5} {'lambda':>7} {'waves':>6} {'t_tile':>8} {'总时间':>8}")
base = None
for mt, nt, kw in TILES:
    b = plan(mt, nt, **kw)['b']
    T = (M // mt) * (N // nt)
    P = b * SM
    t_tile = mt * nt * b / 32768          # 归一化：128x256 且 b=1 记为 1.000
    total = -(-T // P) * t_tile
    if base is None:
        base = total
    print(f"{f'{mt}x{nt}':>9} {T:>6} {P:>5} {T/P:>7.2f} {-(-T//P):>6} "
          f"{t_tile:>8.3f} {total/base:>8.4f}")

print()
print("== 尾波损失 =（ceil(lambda)-lambda) x t_tile，恒小于一个 t_tile ==")
print(f"{'情形':>24} {'P':>5} {'lambda':>7} {'W':>6} {'t_tile':>7} {'尾波损失':>9} {'占理想时间':>10}")
for name, mt, nt, kw in [("128x256, 4 stage", 128, 256, dict()),
                         ("128x128, 4 stage", 128, 128, dict()),
                         ("128x128, 3 stage rho=0.6", 128, 128, dict(stages=3, rho=0.6)),
                         ("128x64, 4 stage", 128, 64, dict())]:
    b = plan(mt, nt, **kw)['b']
    T = (M // mt) * (N // nt)
    P = b * SM
    lam = T / P
    W = -(-T // P) - lam
    t_tile = mt * nt * b / 32768
    print(f"{name:>24} {P:>5} {lam:>7.2f} {W:>6.3f} {t_tile:>7.3f} "
          f"{W*t_tile:>9.4f} {W/lam*100:>9.2f}%")
```

```text
== 守恒律：问题 8192x8192x8192，P 从 132 涨到 396，总时间一动不动 ==
     tile      T     P  lambda  waves   t_tile      总时间
  128x256   2048   132   15.52     16    1.000   1.0000
  128x128   4096   132   31.03     32    0.500   1.0000
   128x64   8192   264   31.03     32    0.500   1.0000
    64x64  16384   396   41.37     42    0.375   0.9844

== 尾波损失 =（ceil(lambda)-lambda) x t_tile，恒小于一个 t_tile ==
                      情形     P  lambda      W  t_tile      尾波损失      占理想时间
        128x256, 4 stage   132   15.52  0.485   1.000    0.4848      3.12%
        128x128, 4 stage   132   31.03  0.970   0.500    0.4848      3.12%
128x128, 3 stage rho=0.6   264   15.52  0.485   1.000    0.4848      3.12%
         128x64, 4 stage   264   31.03  0.970   0.500    0.4848      3.12%
```

这就是 050 那句话的完整答案，一句话：

$$\boxed{\text{只要 } P \text{ 和 } T \text{ 同比例上升，}\lambda \text{ 就不变，总时间也不变}}$$

从 $128\times256$ 一路走到 $64\times64$，$P$ 从 132 涨到 396（三倍），$T$ 从 2048 涨到 16384（八倍），**总时间一直是 1.0000**。$\lambda$ 在这四行里是 15.52 → 31.03 → 31.03 → 41.37，来回晃，但总时间一动不动。

那尾波呢？第二张表里四行的尾波损失完全相同——**0.4848，占理想时间的 3.12%**。这不是巧合：$T$ 翻倍让 $\lambda$ 翻倍，$\lambda$ 的小数部分就从 0.485 变成 0.970，而 $t_{tile}$ 同时减半，乘积不动。（这批 2 的幂形状下是精确相等的；换个非 2 的幂的形状会有一点出入，但不会改变"乘积大致不动"这个结论。）

物理原因在第四节给出（同一 SM 上的第二个 CTA 只能分走算力，不能增加算力），这里是它的算术影子：$t_{tile}\propto b$、$P \propto b$，两个 $b$ 在一次乘除里抵消。**「$P$ 翻倍」在这条路上是一次恒等变换。**

所以 050 的「$\lambda$ 直接减半就能少做量化损失」需要补一个前提：**$\lambda$ 减半必须是「$T$ 不变、$P$ 翻倍」那条路，而不是「$P$ 和 $T$ 一起翻倍」那条路。** 前者真省钱，后者是恒等变换。

### 3.5 那"$T$ 不变、$P$ 翻倍"这条路走得通吗

$T$ 不变意味着 tile 尺寸不变。要让 $b$ 从 1 变 2，寄存器和 SMEM 两条预算都得松开——而它们恰好被这个 tile 占住了：

```python
REG_SM = 65536
SMEM_SM = 228 * 1024
SMEM_RESERVED = 1024
WARP_SLOT = 64
BLOCK_SLOT = 32


def ru(a, g):
    return (a + g - 1) // g * g


def plan(mt, nt, kt=64, stages=4, rho=0.4, esize=2):
    threads = 2 * mt
    warps = threads // 32
    accum = mt * nt // threads
    regs = ru(int(accum * (1 + rho)), 8)
    smem = (mt + nt) * kt * esize * stages
    b_reg = REG_SM // (warps * 32 * regs)
    b_smem = (SMEM_SM - SMEM_RESERVED) // smem
    b = min(b_reg, b_smem, WARP_SLOT // warps, BLOCK_SLOT)
    return regs, smem, b_reg, b_smem, b


print("== 要 blocks=2，寄存器和 SMEM 必须同时松手 ==")
print(f"{'tile':>9} {'stages':>6} {'rho':>5} {'regs':>5} {'smem/KiB':>9} "
      f"{'reg 允许':>8} {'smem 允许':>9} {'blocks':>7}")
for mt, nt in ((128, 128), (128, 64)):
    for st in (3, 4):
        for rho in (0.6, 0.4):
            regs, smem, b_reg, b_smem, b = plan(mt, nt, stages=st, rho=rho)
            print(f"{f'{mt}x{nt}':>9} {st:>6} {rho:>5.1f} {regs:>5} "
                  f"{smem//1024:>9} {b_reg:>8} {b_smem:>9} {b:>7}")
```

```text
== 要 blocks=2，寄存器和 SMEM 必须同时松手 ==
     tile stages   rho  regs  smem/KiB   reg 允许   smem 允许  blocks
  128x128      3   0.6   104        96        2         2       2
  128x128      3   0.4    96        96        2         2       2
  128x128      4   0.6   104       128        2         1       1
  128x128      4   0.4    96       128        2         1       1
   128x64      3   0.6    56        72        4         3       3
   128x64      3   0.4    48        72        5         3       3
   128x64      4   0.6    56        96        4         2       2
   128x64      4   0.4    48        96        5         2       2
```

结论很干脆：$128\times128$ 在 4 级流水下，寄存器允许 2 个 CTA（$b_{reg}=2$——表里每线程 96 个寄存器，离 128 的门槛还有余量），**卡住的是 SMEM**：$b_{smem}=1$。把流水线从 4 级降到 3 级，$b_{smem}$ 才变成 2，这时两条都不紧，$b=2$。

寄存器那一栏的门槛是 **128 个/线程**（$65536/(2\times 8\times 32)$）。所以真正要付的代价只有一个：**少一级流水线**。

这个代价立刻兑现。第 044 期算过异步流水的收益上限是 $(c+p)/\max(c,p)$，级数从 4 降到 3，这个上界直接往下掉。**为了 occupancy 牺牲流水线深度，是把 044 期赚到的钱拿去买 050 期一个不存在的收益。**

### 3.6 什么时候 P 才真的可以很大

寄存器池和 SMEM 池只在"累加器吃满寄存器"的 kernel 上才是主要矛盾。访存受限的小 kernel 没有累加器，两条预算都退居二线，此时唯一咬人的是 warp 槽位和 block 上限：

```python
REG_SM = 65536
SMEM_SM = 228 * 1024
SMEM_RESERVED = 1024
WARP_SLOT = 64
BLOCK_SLOT = 32
SM = 132


def ru(a, g):
    return (a + g - 1) // g * g


print("== 没有累加器的小 kernel：P 可以到 SM 数的十几倍 ==")
print(f"{'每 block 线程':>12} {'regs/线程':>9} {'blocks/SM':>9} {'P':>6} {'occ%':>6} {'卡在哪':>7}")
for threads, regs in ((128, 40), (256, 40), (512, 32), (1024, 32), (1024, 64)):
    warps = threads // 32
    r = ru(regs, 8)
    b_reg = REG_SM // (warps * 32 * r)
    b_warp = WARP_SLOT // warps
    b = min(b_reg, b_warp, BLOCK_SLOT)
    lock = ["寄存器", "warp 槽", "block 上限"][[b_reg, b_warp, BLOCK_SLOT].index(b)]
    print(f"{threads:>12} {r:>9} {b:>9} {b*SM:>6} {b*warps/WARP_SLOT*100:>6.1f} {lock:>7}")
```

```text
== 没有累加器的小 kernel：P 可以到 SM 数的十几倍 ==
  每 block 线程   regs/线程 blocks/SM      P   occ%     卡在哪
         128        40        12   1584   75.0     寄存器
         256        40         6    792   75.0     寄存器
         512        32         4    528  100.0     寄存器
        1024        32         2    264  100.0     寄存器
        1024        64         1    132   50.0     寄存器
```

128 线程、40 个寄存器 → 一个 SM 能挂 12 个 CTA，$P = 1584$，是 SM 数的 12 倍。**所以「occupancy」这个词在 GEMM kernel 和 elementwise kernel 上指的根本不是同一件事**：前者是 12.5%，后者可以到 75%~100%，而两者都在各自的性能上限附近。这也是为什么 045 期那句「occupancy 是手段不是目的」必须被反复讲——它取决于你在看哪一类 kernel。

---

## 四、同一块 SM 上的两个 CTA，真的在并行吗

这是 050 完全没碰、而 wave 模型能不能成立又完全依赖它的问题。

答案是：**一半并行，一半不并行。**

- **真的分区**：寄存器（每个 CTA 分到固定份额）、shared memory（物理上切成互不重叠的段）、warp 槽位。这三样是空间分割，两个 CTA 各拿各的，互不干扰。
- **不分区，时间共享**：FP32 的 lane、Tensor Core、LSU（访存单元）、SFU。一个 SM 只有固定数量的执行单元，两个 CTA 的 warp 都往同一批单元上发指令，**由 warp scheduler 轮流发射**。

换句话说，第二个 CTA 带来的不是"更多的算力"，而是**调度器就绪队列里更多的候选**。这正是 045 期的结论——只是 045 讲的是 warp 粒度，这里是 CTA 粒度，而结论一模一样：occupancy 的价值是延迟隐藏，不是吞吐。

CPU 上可以把这件事的骨架量出来。本机是 Apple M5，4 个 P 核 + 6 个 E 核，没有 SMT（一个核同时只能跑一个硬件线程），所以"同一核上挂多个上下文"这件事在 CPU 上做不到——但"执行单元的数量是硬上界"和"延迟受限时空闲有多严重"这两件事能量。

先用一个纯 FMA 依赖链的负载（`cc -O3` 现编，不用 `cpp_extension`——后者第一次编译要五到十分钟），扫同时运行的工作单元数：

```python
"""执行单元数的硬上界：同一进程内加线程，总工作量固定。
负载用 cc -O3 现编的纯 FMA 依赖链（计算受限）。不用 GEMM：本机的矩阵单元
被所有进程共享，多进程同时跑总吞吐恒定，拿它做并行度实验会得出假结论。"""
import re
import subprocess
import tempfile
from pathlib import Path

SRC = r"""
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now_ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
typedef struct { long iters; double sink; } job_t;
static void run(job_t *j) {
  double s0 = 1.000001, s1 = 1.000002, s2 = 1.000003, s3 = 1.000004;
  const double x = 1.0000001;
  for (long i = 0; i < j->iters; i++) {
    s0 = s0 * x + 1.0; s1 = s1 * x + 1.0;
    s2 = s2 * x + 1.0; s3 = s3 * x + 1.0;
  }
  j->sink = s0 + s1 + s2 + s3;
}
static void *worker(void *p) { run((job_t *)p); return NULL; }
int main(int argc, char **argv) {
  int nth = atoi(argv[1]);
  long iters = atol(argv[2]), per = iters / nth;
  job_t *jobs = calloc(nth, sizeof(job_t));
  pthread_t *th = calloc(nth, sizeof(pthread_t));
  /* 热身：页错误和「调度器把线程迁到 P 核」都要时间，冷启动会慢一倍 */
  for (int t = 0; t < nth; t++) { jobs[t] = (job_t){per / 4, 0}; run(&jobs[t]); }
  for (int t = 0; t < nth; t++) jobs[t] = (job_t){per, 0};
  double t0 = now_ms();
  for (int t = 0; t < nth; t++) pthread_create(&th[t], NULL, worker, &jobs[t]);
  for (int t = 0; t < nth; t++) pthread_join(th[t], NULL);
  printf("TIME %.3f\n", now_ms() - t0);
  free(jobs); free(th);
  return 0;
}
"""

D = Path(tempfile.mkdtemp())
(D / "u.c").write_text(SRC)
BIN = str(D / "u")
subprocess.run(["cc", "-O3", "-pthread", "-o", BIN, str(D / "u.c")], check=True)

TOT = 200_000_000


def run(t, iters, reps=5):
    best = None
    for _ in range(reps):
        p = subprocess.run([BIN, str(t), str(iters)], capture_output=True, text=True)
        ms = float(re.search(r"TIME ([\d.]+)", p.stdout).group(1))
        best = ms if best is None else min(best, ms)
    return best


print("== 同时驻留的工作单元数 vs 吞吐（总工作量固定 200M 次迭代）==")
print(f"{'线程':>4} {'墙钟/ms':>9} {'吞吐/Giter/s':>13} {'相对 1 线程':>11}")
base = None
for t in (1, 2, 3, 4, 6, 8, 10, 12, 16):
    ms = run(t, TOT)
    thr = TOT / ms / 1e6
    base = thr if base is None else base
    print(f"{t:>4} {ms:>9.1f} {thr:>13.2f} {thr/base:>10.2f}x")
```

```text
== 同时驻留的工作单元数 vs 吞吐（总工作量固定 200M 次迭代）==
  线程     墙钟/ms    吞吐/Giter/s     相对 1 线程
   1     144.9          1.38       1.00x
   2      73.6          2.72       1.97x
   3      51.1          3.92       2.84x
   4      41.9          4.77       3.46x
   6      30.7          6.51       4.72x
   8      25.3          7.90       5.72x
  10      22.4          8.93       6.47x
  12      25.0          8.00       5.79x
  16      22.8          8.79       6.37x
```

曲线在 **10** 个单元处拐平，然后掉头。10 = 本机的物理核数（4P + 6E）。三点读数：

1. **上限就是执行单元数。** 12 个以上的线程只能抢 10 份算力，多出来的在空转还会互相干扰：10 个线程是约 6.5 倍，12 个掉下来，16 个也只回到 6.4 倍左右，**始终没有再涨上去**（抖动在 ±10% 量级，别抠小数）。
2. **10 个单元只给了约 6 倍吞吐，不是 10 倍。** 因为 M5 的 4 个 P 核和 6 个 E 核不等价——这一点和 GPU 不同，GPU 的 SM 是等价的。如果拿 CPU 类比 GPU，要记得这个折扣。
3. **超订不会带来收益。** 一个执行单元上挂超过一个工作单元，只有在"这个单元正在等"的时候才有用——而 CPU 没有 SMT，等的时候就是纯等。

第三点是关键。它意味着第二个 CTA 的价值**完全取决于第一个 CTA 会不会阻塞**：

```python
"""两个杠杆：单单元内的 ILP，和单元的总数。
mode 2 = 1 路 FMA 依赖链（ILP=1）；mode 3 = 2 路；mode 0 = 4 路
mode 1 = 4 路 FMA + 每次迭代一次依赖上一次结果的随机载入（延迟受限）
"""
import re
import subprocess
import tempfile
from pathlib import Path

SRC = r"""
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NSLOT (16L << 20)
static volatile int64_t *chain;
static double now_ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
/* 24 bit 上的双射（12+12 位 Feistel）：双射才能保证游走不会掉进短环。
   一开始用 xorshift 的低位当随机映射，期望环长只有 ~sqrt(2^26) ≈ 5000 个元素，
   40 KB 全在 L1 —— 量出来的「DRAM 延迟」只有 3 ns，是假信号。 */
static uint32_t feistel12(uint32_t x) {
  uint32_t L = x >> 12, R = x & 0xFFFu;
  for (int r = 0; r < 6; r++) {
    uint32_t f = ((R * 2654435761u + 0x9E3779B9u * (uint32_t)(r + 1)) >> 7) & 0xFFFu;
    uint32_t t = L ^ f; L = R; R = t;
  }
  return (L << 12) | R;
}
typedef struct { long iters; int mode; double sink; } job_t;
static void run(job_t *j) {
  double s0 = 1.000001, s1 = 1.000002, s2 = 1.000003, s3 = 1.000004;
  const double x = 1.0000001;
  int64_t idx = 0;
  if (j->mode == 2) {
    for (long i = 0; i < j->iters; i++) s0 = s0 * x + 1.0;
  } else if (j->mode == 3) {
    for (long i = 0; i < j->iters; i++) { s0 = s0 * x + 1.0; s1 = s1 * x + 1.0; }
  } else if (j->mode == 0) {
    for (long i = 0; i < j->iters; i++) {
      s0 = s0 * x + 1.0; s1 = s1 * x + 1.0;
      s2 = s2 * x + 1.0; s3 = s3 * x + 1.0;
    }
  } else {
    for (long i = 0; i < j->iters; i++) {
      s0 = s0 * x + 1.0; s1 = s1 * x + 1.0;
      s2 = s2 * x + 1.0; s3 = s3 * x + 1.0;
      idx = chain[idx];          /* 下一次的地址依赖这一次的载入结果 */
    }
    s0 += (double)(idx & 1023) * 1e-3;
  }
  j->sink = s0 + s1 + s2 + s3;
}
static void *worker(void *p) { run((job_t *)p); return NULL; }
int main(int argc, char **argv) {
  int mode = atoi(argv[1]), nth = atoi(argv[2]);
  long iters = atol(argv[3]), per = iters / nth;
  chain = malloc(sizeof(int64_t) * NSLOT);
  for (long i = 0; i < NSLOT; i++) chain[i] = (int64_t)feistel12((uint32_t)i);
  job_t *jobs = calloc(nth, sizeof(job_t));
  pthread_t *th = calloc(nth, sizeof(pthread_t));
  for (int t = 0; t < nth; t++) { jobs[t] = (job_t){per / 4, mode, 0}; run(&jobs[t]); }
  for (int t = 0; t < nth; t++) jobs[t] = (job_t){per, mode, 0};
  double t0 = now_ms();
  for (int t = 0; t < nth; t++) pthread_create(&th[t], NULL, worker, &jobs[t]);
  for (int t = 0; t < nth; t++) pthread_join(th[t], NULL);
  printf("TIME %.3f\n", now_ms() - t0);
  return 0;
}
"""

D = Path(tempfile.mkdtemp())
(D / "b.c").write_text(SRC)
BIN = str(D / "b")
subprocess.run(["cc", "-O3", "-pthread", "-o", BIN, str(D / "b.c")], check=True)
FMA = {2: 1, 3: 2, 0: 4, 1: 4}


def run(mode, threads, iters, reps=5):
    best = None
    for _ in range(reps):
        p = subprocess.run([BIN, str(mode), str(threads), str(iters)],
                           capture_output=True, text=True)
        ms = float(re.search(r"TIME ([\d.]+)", p.stdout).group(1))
        best = ms if best is None else min(best, ms)
    return best


IT = 40_000_000
print("== A. 同一个执行单元内，靠 ILP 能填多满（单线程）==")
print(f"{'ILP':>4} {'FMA/迭代':>9} {'ns/迭代':>9} {'GFMA/s':>9} {'相对 ILP=1':>11}")
base = None
for mode in (2, 3, 0):
    ms = run(mode, 1, IT)
    g = IT * FMA[mode] / ms / 1e6
    base = g if base is None else base
    print(f"{FMA[mode]:>4} {FMA[mode]:>9} {ms*1e6/IT:>9.3f} {g:>9.2f} {g/base:>10.2f}x")

print()
print("== B. 延迟受限时，一个执行单元有多少时间在干活 ==")
ns_c = run(0, 1, IT) * 1e6 / IT
ns_l = run(1, 1, 2_000_000) * 1e6 / 2_000_000
print(f"计算受限：{ns_c:.3f} ns/迭代，每迭代 4 条 FMA")
print(f"延迟受限：{ns_l:.3f} ns/迭代，其中算术只占 {ns_c:.3f} ns")
print(f"单线程的有效利用率上界 = {ns_c/ns_l*100:.1f}%")

print()
print("== C. 延迟受限时，只能靠「同时挂更多工作单元」==")
print(f"{'单元数':>6} {'ns/迭代(每链)':>13} {'吞吐/Miter/s':>13} {'相对 1':>8}")
base = None
for t in (1, 2, 4):
    ms = run(1, t, 2_000_000, reps=3)
    thr = 2_000_000 / ms / 1e3
    base = thr if base is None else base
    print(f"{t:>6} {ms*1e6/(2_000_000/t):>13.1f} {thr:>13.1f} {thr/base:>7.2f}x")
```

```text
== A. 同一个执行单元内，靠 ILP 能填多满（单线程）==
 ILP    FMA/迭代     ns/迭代    GFMA/s    相对 ILP=1
   1         1     0.755      1.32       1.00x
   2         2     0.756      2.64       2.00x
   4         4     0.726      5.51       4.16x

== B. 延迟受限时，一个执行单元有多少时间在干活 ==
计算受限：0.728 ns/迭代，每迭代 4 条 FMA
延迟受限：74.248 ns/迭代，其中算术只占 0.728 ns
单线程的有效利用率上界 = 1.0%

== C. 延迟受限时，只能靠「同时挂更多工作单元」==
   单元数     ns/迭代(每链)    吞吐/Miter/s     相对 1
     1          73.7          13.6    1.00x
     2          70.5          28.4    2.09x
     4          61.4          65.2    4.80x
```

三组读数：

- **同一个执行单元内，ILP 从 1 加到 4，吞吐涨约 4.2 倍。** 一条 FMA 的延迟是 4 个周期（0.755 ns ≈ 3.5 cycle @ 4.7 GHz），所以单条依赖链只能把 1/4 的流水线喂饱；开 4 条独立链就把这条流水线填满了。**这是"多开 ILP"这条路。**
- **延迟受限时单线程的利用率只有约 1%。** 一次串行随机载入 70~80 ns（和 044 期测到的 87.6 ns 同一量级），算术只占 0.73 ns。也就是说一个执行单元在等的时候，99% 的时间是闲的。
- **这时只能靠加工作单元：1 → 4 个单元拿到约 4.8 倍**（比线性还高一点，因为 4 条独立链的 TLB 页走也并行了）。

把这三条接到 GPU 上：**occupancy 就是"加工作单元"这条路，ILP 就是"每个单元多开几条在飞请求"这条路，两者是可以互换的。** 045 期给出了这条互换关系的定量形式：

$$\text{occupancy} \times \text{每 warp 的 MLP} \gtrsim \frac{L \times BW}{128\ \text{B}}$$

本期做的事只是把它从 warp 粒度搬到 CTA 粒度：**第二、第三个 CTA 的作用，是让 warp scheduler 的就绪队列变长，而不是让 SM 多出任何执行资源。** 所以 `P` 翻倍时，$t_{tile}$ 一定同步翻倍——这就是 3.4 节那条守恒律的物理原因。

---

## 五、PyTorch 里怎么用

### 5.1 Inductor 真的在用这个判据

`torch/_inductor/runtime/triton_heuristics.py` 里有一个函数叫 `_iter_rblock_scale_candidates`，它的注释把整件事说得比大多数文档都清楚（下面这段是上游 2.14.0 的源码，行号以本机安装版本为准）：

```python
# 摘自本机 torch 2.14.0 的 _inductor/runtime/triton_heuristics.py:929-958。
# 原文在一个 for 循环里用 continue 跳过，这里改写成函数 + return，逻辑一字未动。
def should_scale_rblock(compiled_binary, triton_config, device_prop, total_block, warp_size):
    nreg = getattr(compiled_binary, "n_regs", None)     # ptxas 报的实际寄存器用量
    # 每 SM 有 65536 个寄存器；要拿满 occupancy 要跑 2048 线程，
    # 所以每线程最多 65536/2048 = 32 个寄存器
    need = device_prop.regs_per_multiprocessor // device_prop.max_threads_per_multi_processor
    if nreg <= need:
        return False                                   # 已经够满，什么都不做
    nreg_per_block = nreg * warp_size * triton_config.num_warps
    max_blocks_per_sm = max(device_prop.regs_per_multiprocessor // nreg_per_block, 1)
    if total_block <= max_blocks_per_sm * device_prop.multi_processor_count:
        return False                                   # grid 一波装得下，不用付代价
    # 走到这里才动手：把最大的那个 R0_BLOCK 砍一半，用更少的寄存器换更多 block
    return True
```

这两个 `if` 就是本期的全部内容在生产代码里的样子：

1. 先看每线程寄存器有没有超过 $65536/2048 = 32$——没有就说明 occupancy 已经拉满，别动。
2. 算出 `max_blocks_per_sm`，乘上 SM 数得到 $P$。
3. **只有当 $T > P$ 时，才愿意牺牲 block 尺寸去换 occupancy。**

把第 2、3 步单独拿出来跑一遍，就能看到那条分界线：

```python
# 复现 Inductor 的 occupancy 判据（A100/H100 的公开规格）
REGS_SM = 65536          # regs_per_multiprocessor
MAX_THR_SM = 2048        # max_threads_per_multi_processor
SM = 132                 # multi_processor_count
WARP = 32
FREE_REGS = REGS_SM // MAX_THR_SM      # = 32，Inductor 的「够满」门槛


def verdict(nreg, num_warps, total_block):
    """返回 (max_blocks_per_sm, P, 是否要降 R0_BLOCK)"""
    if nreg <= FREE_REGS:
        return None, None, False       # 第一道闸：已经够满
    nb = max(REGS_SM // (nreg * WARP * num_warps), 1)
    P = nb * SM
    return nb, P, total_block > P      # 第二道闸：T > P 才动手


print(f"每 SM {REGS_SM} 个寄存器 / {MAX_THR_SM} 线程 -> 每线程 {FREE_REGS} 个以内算「够满」")
print()
print("== 判据的分界线就是 T = P ==")
for nreg, nw in ((232, 8), (168, 8), (40, 4)):
    nb, P, _ = verdict(nreg, nw, 1)
    print(f"nreg={nreg:>3} num_warps={nw:>2} -> max_blocks_per_sm={nb:>2}, P={P:>4}")
    for T in (128, 256, 1024, 4096, 16384):
        _, _, need = verdict(nreg, nw, T)
        print(f"    T={T:>6}  {'要降 R0_BLOCK' if need else '不动'}")
```

```text
每 SM 65536 个寄存器 / 2048 线程 -> 每线程 32 个以内算「够满」

== 判据的分界线就是 T = P ==
nreg=232 num_warps= 8 -> max_blocks_per_sm= 1, P= 132
    T=   128  不动
    T=   256  要降 R0_BLOCK
    T=  1024  要降 R0_BLOCK
    T=  4096  要降 R0_BLOCK
    T= 16384  要降 R0_BLOCK
nreg=168 num_warps= 8 -> max_blocks_per_sm= 1, P= 132
    T=   128  不动
    T=   256  要降 R0_BLOCK
    T=  1024  要降 R0_BLOCK
    T=  4096  要降 R0_BLOCK
    T= 16384  要降 R0_BLOCK
nreg= 40 num_warps= 4 -> max_blocks_per_sm=12, P=1584
    T=   128  不动
    T=   256  不动
    T=  1024  不动
    T=  4096  要降 R0_BLOCK
    T= 16384  要降 R0_BLOCK
```

注意 `nreg=40, num_warps=4` 那组：$P = 1584$，比 `nreg=232` 的 132 大了整整 12 倍。这正是 3.6 节那张表——同一个判据，在 GEMM kernel 上几乎永远触发不了，在 reduction kernel 上很容易触发。

### 5.2 Triton 侧的两个旋钮

Triton 不让你直接设 occupancy，只给你两个间接旋钮：

```py
# 本机没有 triton，这段未编译未运行，只示意参数位置
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m) / tl.load(y_ptr + off, mask=m), mask=m)


# num_warps 决定每 CTA 的 warp 数（w_blk），num_stages 决定 SMEM 里驻留几份缓冲
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024,
                                   num_warps=4,     # 4*32 = 128 线程
                                   num_stages=1)    # 逐元素 kernel 通常 1 级就够
```

这两个参数的物理意义正好对应 3.2 节的前两把锁：

- `num_warps` 直接改 $w_{blk}$：它同时影响 warp 槽位（$64/w_{blk}$）、寄存器的块级用量（$32\,w_{blk}\,r$）和 CTA 的线程数。
- `num_stages` 直接改 $s_{blk}$：每级一份 $(M_t+N_t)K_t$ 的缓冲，级数就是 SMEM 的倍数。

**所以调这两个参数是在移动四条锁，不是在直接设定 occupancy。** 大多数"triton autotune 调出来正好是 4 warps / 3 stages"的配置，本质上是这两条预算恰好都留了一点余量。

Inductor 给逐元素 kernel 定 `num_warps` 的逻辑很朴素——它不看寄存器，只看每个 warp 摊到多少个元素：

```python
# 复现 _inductor/runtime/triton_heuristics.py 的 num_warps 选择
# _num_warps(n, min_num_warps=1) = next_power_of_2(min(max(n, 1), 8))
# 然后 triton_config 里还有一句 num_warps = max(num_warps, 4)（前提是 BLOCK >= 128）
REGS_SM = 65536
WARP_SLOT = 64
SM = 132
NEPW = 256          # num_elements_per_warp 的默认值


def next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def ind_num_warps(xblock, nepw=NEPW):
    n = xblock // nepw
    nw = next_pow2(min(max(n, 1), 8))
    return max(nw, 4) if xblock >= 128 else nw


print(f"== XBLOCK -> num_warps -> 每 SM 能挂几个 CTA（假设 ptxas 报 40 个寄存器）==")
print(f"{'XBLOCK':>7} {'num_warps':>10} {'线程/block':>10} {'blocks/SM':>10} {'P':>6} {'occ%':>6}")
for xblock in (256, 512, 1024, 2048, 4096):
    nw = ind_num_warps(xblock)
    threads = nw * 32
    b = min(REGS_SM // (threads * 40), WARP_SLOT // nw, 32)
    print(f"{xblock:>7} {nw:>10} {threads:>10} {b:>10} {b*SM:>6} "
          f"{b*nw/WARP_SLOT*100:>6.1f}")
```

```text
== XBLOCK -> num_warps -> 每 SM 能挂几个 CTA（假设 ptxas 报 40 个寄存器）==
 XBLOCK  num_warps   线程/block  blocks/SM      P   occ%
    256          4        128         12   1584   75.0
    512          4        128         12   1584   75.0
   1024          4        128         12   1584   75.0
   2048          8        256          6    792   75.0
   4096          8        256          6    792   75.0
```

`num_warps` 在 4 和 8 之间就封顶了（`max_num_warps=8`），因为再往上加，一个 block 的寄存器需求就会把它自己卡死。于是逐元素 kernel 的 occupancy 上限由「每线程寄存器」决定，而不是由 warp 数决定。

### 5.3 CUTLASS 侧：$P$ 是算出来的，而且更常见的是"问"出来的

```cpp
// 摘自 CUTLASS 上游 main 分支：include/cutlass/gemm/kernel/tile_scheduler_params.h:853
// 前面第 845 行刚用 get_grid_shape() 权衡完 swizzle / cluster，得到 grid
uint64_t ctas_per_wave = grid.x * grid.y;      // 一个 wave 能同时驻留的 CTA 数
uint64_t ctas_per_wave_in_full_clusters = (ctas_per_wave / cluster_size) * cluster_size;
```

第二行就是 046 期那句「用了 cluster 调度就不再弹性」的算术形态：$\lambda$ 的分母被迫向下取整到 cluster 的整数倍，**尾波永远按整簇浪费**。

（050 期引的是 `tsp.h:854`，本文重新抓的 main 分支是 `:853`——差一行是上游改动，不是笔误。）

那 `grid` 又是从哪来的？往回翻到 `tile_scheduler_params.h:226`：

```cpp
// get_grid_shape() 的第一行就说明了一切
int const sm_count = hw_info.sm_count;
int const max_active_clusters = hw_info.max_active_clusters;   // 不是算的，是传进来的
```

occupancy 不是 CUTLASS 自己推的，它**问 CUDA 运行时**：

```cpp
// 摘自 include/cutlass/gemm/device/gemm_universal_adapter.h:271-293
static int maximum_active_blocks(int = -1) {
  int max_active_blocks = -1;
  int smem_size = GemmKernel::SharedStorageSize;
  // 动态 shared memory 超过 48 KB 时，必须先设属性，否则下面的查询会直接失败
  if (smem_size >= (48 << 10)) {
    cudaFuncSetAttribute(device_kernel<GemmKernel>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
  // 让运行时去算：只有它知道 ptxas 的真实寄存器数、SMEM 分配粒度和 block 上限
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, device_kernel<GemmKernel>,
      GemmKernel::MaxThreadsPerBlock, smem_size);
  return max_active_blocks;
}
```

这三段合起来正好是本文那张表的三个来源：

- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 返回的就是 3.2 节的最后一列（blocks/SM）。**它内部就是把寄存器、SMEM、warp 槽、block 上限四条锁取一次 $\min$。**
- `ctas_per_wave = grid.x * grid.y` 是把这个数乘上 SM 数、再被 cluster 截断之后的结果。
- 所以想在自己的 kernel 里拿到 $P$，**正确做法不是手算，是调这个 API**。手算唯一的价值是想清楚"换 tile 会撞哪把锁"——也就是本文第三节干的事。

（本机无 CUDA，这一段是源码级/文档级证据，未实测。）

## 六、围绕该领域展开

| 期号 | 那一期讲的 | 和本期的接口 |
|---|---|---|
| 043 | 内存层次、屋脊点、算术强度 | 3.3 节的两条预算最后归到 $M_tN_t/2(M_t+N_t)$，就是 043 的算术强度 |
| 044 | cp.async / TMA / 软件流水线 | `stages` 是 3.5 节里唯一能换来 occupancy 的筹码，而它正是 044 的全部内容 |
| 045 | SIMT 与 occupancy（warp 粒度） | 本期是同一个结论的 CTA 粒度版本；045 给的三把锁在这里变成四条 |
| 046 | 线程块簇与 DSMEM | cluster 给 occupancy 加了第五条约束：必须整簇驻留 |
| 047 | Tensor Core 与 wgmma 的寄存器预算 | 3.1 节的 $N\le256$ 就是 047 那个"每线程 4 个 reg"约定的另一面 |
| 048 | Epilogue 与累加器写回 | epilogue 的子块化（`epilogue_subtile`）会临时占用寄存器，直接压低 $b_{reg}$ |
| 049 | Split-K，$S^*$ 判据 | $S^*$ 是"用归约带宽换并行度"，本期是"用 tile 尺寸换并行度"，两者是同一笔交易的两个面 |
| 050 | Stream-K，$\lambda=T/P$ | 本期把 050 里那个"$P$ 会翻倍"的省略号补完了：**它不带来加速，只带来一个新的 $\lambda$** |

如果只记一条线，记这条：

$$\underbrace{\text{tile 尺寸}}_{\text{047/048}} \to \underbrace{\text{每线程资源}}_{\text{045}} \to \underbrace{b_{reg},\ b_{smem}}_{\text{本期}} \to \underbrace{P = SM\cdot b}_{\text{050}} \to \underbrace{\lambda}_{\text{049/050}}$$

**这条链上每一环都是除法和取整，没有任何一环是"调参"。** 你能动的只有最左边那个 tile 尺寸和流水线级数，剩下的全是它推出来的。

---

## 七、什么时候该用 / 不该用

**该关心 occupancy 的场合：**

1. **$T < P$**：grid 连一波都装不满。这时 occupancy 直接决定你有多少个执行单元在闲着。判据就是这个不等式，不要凭感觉。
2. **访存受限的小 kernel**（逐元素、layer norm、reduction）：3.6 节那张表，$P$ 能到 SM 数的 12 倍，occupancy 高就有更多请求在飞，直接换带宽。
3. **寄存器密集型 reduction 且 $T \gg P$**：这时砍 block 尺寸换 occupancy 是划算的——Inductor 的 `_iter_rblock_scale_candidates` 就是干这个的，注释里还带了一个实测收益（7.795 ms → 4.883 ms）。

**不该关心的场合：**

1. **compute-bound 的大 GEMM。** $128\times256$、12.5% occupancy、性能接近峰值——这不是特例，是常态。算力被 tile 的算术强度锁住了，多挂 CTA 只会让每个 CTA 变慢。
2. **$T$ 刚好落在 $P$ 的整数倍附近。** 尾波会被抹平（3.4 节第二张表里 $128\times128$ 的 $\lambda$ 恰好离整数只差 0.03），这时候优化 occupancy 没有任何可优化空间。
3. **为了 occupancy 降 `num_stages`。** 这是拿 044 的钱买 050 的账，3.5 节已经算过。

---

## 八、常见坑

**坑 1：把 occupancy 当 KPI。** 12.5% 听起来很糟，但它是 CUTLASS Hopper GEMM 的正常值。真正该看的量是「每 SM 每 cycle 吐了多少 FLOP」和「供数带宽够不够」——那是 043 和 044 的账本，不是 occupancy 的。

**坑 2：以为缩小 tile 能提高吞吐。** 3.4 节的表里，$P$ 从 132 涨到 264，总时间一动不动。$P$ 涨是因为工作量被切得更碎，不是因为机器变强。**唯一有效的收益是"尾波变得更细"，而 3.4 节第二张表显示连这个都不一定成立。**

**坑 3：拿 SM 数当 $P$。** 050 期专门强调过一次，本期给了它的完整算法：$P = SM \times \min(64/w_{blk},\ 65536/(32w_{blk}r),\ 228\text{K}/s_{blk},\ 32)$。差一个因子 2，$\lambda$ 就错一倍，$\eta$ 就错到判据翻转（050 期里 $2048^2$ 和 $8192^2$ 的判据就是靠 $\eta$ 精确到小数点后三位才分得开的）。

**坑 4：以为多挂 CTA 就多算力。** 寄存器、SMEM、warp 槽是空间分割的，但执行单元是时间共享的。所以第二个 CTA 让 SM 的总算力**一点都没变**，只是让调度器多了一个候选。这直接解释了为什么 $t_{tile}\propto b$——守恒。

**坑 5：用 `maxrregcount` 硬压寄存器换 occupancy。** 压寄存器会让编译器把中间值溢出到 local memory（走 L1，实际上是显存），代价通常是几个百分点的性能。只有当 $T > P$、且压下来真的能把 $b$ 抬上去时才划算——和 Inductor 那三行 `if` 是同一个判据。

---

## 九、一句话总结

$$\boxed{P = SM\times\min(\text{warp 槽},\ \text{寄存器池},\ \text{SMEM 池},\ \text{block 上限})}$$

**$P$ 是一个商，不是一个旋钮；把 tile 缩小能让它翻倍，但翻倍的代价是 $T$ 也翻倍，$\lambda$ 和总时间都不动。**

---

<details>
<summary>今日练习</summary>

### 练习 1：$P$ 取错会怎样

取 050 期的形状 $M=N=2048$、$K=2048$，tile 用 $128\times256$（$b=1$，$P=132$）和 $128\times64$（$b=2$，$P=264$）。分别算 $\lambda$、$\eta$，并按 050 期的闸门「$2\cdot\text{tail} \ge \text{ctas\_per\_wave}$ 就不做 stream-K」判断该不该做。

```python
SM = 132


def wave_stats(M, N, mt, nt, b):
    T = (M // mt) * (N // nt)
    P = b * SM
    lam = T / P
    waves = -(-T // P)
    tail = waves * P - T
    return T, P, lam, lam / waves, tail


print(f"{'tile':>9} {'T':>5} {'P':>5} {'lambda':>7} {'eta':>6} {'尾波空槽':>8} {'2*tail>=P?':>11}")
for mt, nt, b in ((128, 256, 1), (128, 64, 2)):
    T, P, lam, eta, tail = wave_stats(2048, 2048, mt, nt, b)
    print(f"{f'{mt}x{nt}':>9} {T:>5} {P:>5} {lam:>7.2f} {eta:>6.3f} {tail:>8} "
          f"{'不做' if 2*tail >= P else '做':>11}")
```

```text
     tile     T     P  lambda    eta     尾波空槽  2*tail>=P?
  128x256   128   132    0.97  0.970        4           做
   128x64   512   264    1.94  0.970       16           做
```

**要点**：两种 tile 的 $\lambda$ 差了一倍（0.97 → 1.94），但 $\eta$ **都是 0.970**，闸门结论也完全一样（都该做）。原因还是守恒律：$T$ 涨 4 倍、$P$ 涨 2 倍，$\lambda$ 跟着涨 2 倍，而 $\lceil\lambda\rceil$ 正好也从 1 变成 2，于是 $\eta=\lambda/\lceil\lambda\rceil$ 纹丝不动。

**这是一个有用的副产品**：050 期的 $\eta$ 判据对「换 tile 尺寸」这一族变换是**不变量**。所以那张判据表可以只按形状（几个 tile、几个执行单元）来列，完全不用管 tile 具体多大——这也解释了为什么 050 期正文里从头到尾没讨论过 tile 尺寸。

### 练习 2：给一个具体的 $b$ 反推 $\rho$

已知某 kernel：256 线程/CTA、$128\times128$ tile、3 级流水、fp16 操作数，ptxas 报 128 个寄存器/线程。问：SMEM 和寄存器哪条是紧的？$b$ 是多少？

```python
REG_SM = 65536
SMEM_SM = 228 * 1024
SMEM_RESERVED = 1024

threads, warps, regs, stages, kt, esize = 256, 8, 128, 3, 64, 2
mt, nt = 128, 128

b_reg = REG_SM // (warps * 32 * regs)
smem = (mt + nt) * kt * esize * stages
b_smem = (SMEM_SM - SMEM_RESERVED) // smem
tight = "两条打平" if b_reg == b_smem else ("寄存器" if b_reg < b_smem else "SMEM")
print(f"寄存器允许 {b_reg} 个 CTA（{warps*32*regs} 个寄存器/CTA）")
print(f"SMEM  {smem//1024} KiB/CTA，允许 {b_smem} 个")
print(f"-> b = {min(b_reg, b_smem)}，P = {min(b_reg, b_smem)*132}")
print(f"   紧的那条是：{tight}")
```

```text
寄存器允许 2 个 CTA（32768 个寄存器/CTA）
SMEM  96 KiB/CTA，允许 2 个
-> b = 2，P = 264
   紧的那条是：两条打平
```

**要点**：128 regs × 256 线程 = 32768，正好是寄存器池的一半，所以寄存器这条刚好允许 2 个；96 KiB × 2 = 192 KiB < 227 KiB，SMEM 也允许 2 个。**两条都不紧，才能拿到 2。** 只要把 stages 改回 4（128 KiB × 2 = 256 KiB > 227 KiB），立刻掉回 1——这就是 3.5 节那张表。

### 练习 3：为什么 CUTLASS 不把 tile 做成 $64\times512$

$64\times512$ 的面积和 $128\times256$ 相同，寄存器上界一样是 1.43。为什么没有这个形状？

```python
mt, nt = 64, 512
threads = 2 * mt
accum = mt * nt // threads
print(f"线程 {threads}，每线程累加器 {accum} 个")
print(f"含 40% 开销后需要 {int(accum*1.4)} 个寄存器，上限 255")
print(f"-> {'合法' if int(accum*1.4) <= 255 else '无法编译：超过每线程 255 个寄存器的硬上限'}")

# 为什么面积相同却差这么多：wgmma 的单条指令把 M 钉在 64（047 期）
print(f"\nwgmma 的 M 固定为 64，一个 warpgroup(128 线程) 负责一个 M=64 的块")
print(f"Mt={mt} 只能用 {mt//64*128} 个线程，Nt={nt} 全摊到这几个线程上")
print(f"同样面积换方向：Mt=128 用 {2*128} 线程，每线程只要 {128*256//256} 个累加器")
print("-> 把 M 钉死在 64，是 255 这个上限最终落在 N 身上的原因")
```

```text
线程 128，每线程累加器 256 个
含 40% 开销后需要 358 个寄存器，上限 255
-> 无法编译：超过每线程 255 个寄存器的硬上限

wgmma 的 M 固定为 64，一个 warpgroup(128 线程) 负责一个 M=64 的块
Mt=64 只能用 128 个线程，Nt=512 全摊到这几个线程上
同样面积换方向：Mt=128 用 256 线程，每线程只要 128 个累加器
-> 把 M 钉死在 64，是 255 这个上限最终落在 N 身上的原因
```

**要点**：$N_t=512$ 时每线程要 256 个累加器，光累加器就已经超过 255 的硬上限。根因是 wgmma 把 $M$ 钉在 64（047 期）：$M_t=64$ 只能用 128 个线程，$N_t=512$ 个累加器全摊到这 128 个线程上，就是每人 256 个。同样面积的 $128\times256$ 用 256 线程摊，每人只要 128 个。

**所以 tile 的形状空间是被两条完全不同的规矩夹住的：一条是 255 个寄存器的上限（它经过 warpgroup 的 M=64 约束，最终落在 $N$ 上），一条是 warpgroup 的粒度（它决定 $M$ 只能是 64 的倍数）。** 两条都不允许「用更长的 $N$ 换更短的 $M$」。

</details>

---

**下一期预告**：本期把 $P$ 算清楚了，但有一个东西一直在假设里没碰——「按 $1/b$ 平分 SM 的算力」这个假设，在 GEMM 这种 Tensor Core 满载的 kernel 上对不对？如果两个 CTA 的 wgmma 争同一组 Tensor Core，它们到底是各拿一半，还是有一个会被饿着？这牵到 warp scheduler 的发射策略，也就是 045 期挂着没做的「warp 调度器内部机制」。
