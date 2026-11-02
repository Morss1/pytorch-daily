# PyTorch 每日一课 · 第 047 期

## Tensor Core 与 wgmma：一次乘加放大 26 万倍之后，剩下的全是喂数

---

**日期**：2026-11-02　**难度**：⭐⭐⭐⭐⭐　**预计阅读**：25 分钟
**前置知识**：矩阵乘法的分块思想、共享内存（SMEM）是什么、GPU 的 warp 概念（不熟也能读，本文从动机重讲）
**关联**：第 043 期（屋脊点与算术强度）· 第 044 期（cp.async/TMA 把数据送进 SMEM）· 第 046 期（cluster 与 multicast）—— 那两期讲「怎么把数据搬进来」，本期讲「搬进来之后怎么被吃掉」

---

## 一、这个领域解决什么问题

### 1.1 先看一个我自己没预料到的数

我在一台 Apple M5 上量了 GEMM 的吞吐，单线程，纯 fp32：

```python
import time, torch
torch.set_num_threads(1)          # 单线程，排除多核干扰

def gflops(n, reps=15):
    a, b = torch.randn(n, n), torch.randn(n, n)
    for _ in range(3):             # warmup：page fault 和频率爬升都要先吃掉
        a @ b
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        a @ b
        ts.append(time.perf_counter() - t0)
    return 2 * n**3 / min(ts) / 1e9        # 取 min，避开调度抖动

GHZ, SIMD_FMA = 4.6, 16    # NEON: 每核每 cycle 最多 16 次 fp32 FMA（4 条 128-bit FMA 流水 × 4 lane）
print(f"{'n':>5} {'GFLOPS':>8} {'FMA/cycle':>10} {'SIMD 够吗':>10}")
for n in (128, 256, 512, 1024):
    g = gflops(n)
    fma = g * 1e9 / 2 / (GHZ * 1e9)        # 每个 FMA 算 2 FLOP
    print(f"{n:>5} {g:>8.1f} {fma:>10.1f} {'够' if fma <= SIMD_FMA else '不够':>10}")
```

```text
    n   GFLOPS  FMA/cycle    SIMD 够吗
  128    867.7       94.3         不够
  256   1433.0      155.8         不够
  512   1553.1      168.8         不够
 1024   2068.0      224.8         不够
```

1024³ 那一行：**单核每 cycle 做 224 次 fp32 乘加**（多次运行稳定在 220~226）。

这个数物理上不可能是 SIMD 给你的。一条 NEON FMA 指令吃 4 个 fp32，一个核每 cycle 最多发射 4 条 → 上限 16。**224 是上限的 14 倍**。这不是优化得好不好的问题，是这台机器的 CPU 里有一个我平时不关心的专用矩阵单元在干活（ARMv9 的 SME 提供了这个能力；Apple 没有公开 M5 的具体实现，我把它算作**推断**）。

> 尺度说明：`n=32`、`n=64` 这两档我没放进表里。三次独立运行分别在 52~72、322~419 GFLOPS 之间跳（±30%~40%），小尺寸下矩阵单元的流水填充和线程首次唤醒占了大头，取 min 也压不住。趋势是明确的 —— n=32 时只有 6~10 次 FMA/cycle，**连 SIMD 都没喂饱** —— 但具体数字我不引用。

这个「小矩阵上不去、大矩阵突然起飞」的形状，就是本期要讲的整个故事：

> **专用矩阵单元把「一条指令做 4 个乘加」变成「一条指令做几十万个乘加」。它只在形状对的时候工作，而且一旦工作起来，瓶颈立刻从「算」转到「喂」。**

### 1.2 GPU 上是同一件事，只是数字更大

NVIDIA 把 CPU 里的这个东西叫 Tensor Core。H100 SXM5 的官方规格：

| 单元 | 精度 | 吞吐 |
|---|---|---|
| CUDA Core（FFMA） | FP32 | 67 TFLOPS |
| Tensor Core | TF32（10 位尾数） | 494.7 TFLOPS |
| Tensor Core | FP16 / BF16 | 989.5 TFLOPS |
| Tensor Core | FP8 | 1979 TFLOPS |

132 个 SM、boost 1.98 GHz。算一下每个 SM 每 cycle 能吃多少：

$$\frac{989.5 \times 10^{12}}{132 \times 1.98 \times 10^{9}} \approx 3786\ \text{FLOP/cycle} = 1893\ \text{次 fp16 FMA/cycle}$$

对比 CPU 那两百多次。**同一个 SM 里，Tensor Core 每 cycle 吃 1893 个乘加，而一条普通 FFMA 指令只吃 1 个。**

### 1.3 于是瓶颈整体挪位了

一条 `wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16` 干多少活？

$$64 \times 256 \times 16 = 262144\ \text{次 FMA} = 524288\ \text{FLOP}$$

把它放到上面算出的 Tensor Core 吞吐上：

$$262144 \div 1893 \approx 138.5\ \text{cycle}$$

**一条指令，138 个 cycle 才做完一个。** 而它占用的发射槽是 1 个。

这就是本期的核心矛盾，也是 Hopper 之后所有 GEMM kernel 的设计出发点：

> **算力已经免费了。一条指令扔出去，138 个 cycle 之内你什么都不用做。你唯一要操心的是：这 138 个 cycle 里，下一批操作数能不能准时出现在 SMEM 里、以及累加器放不放得下。**

---

## 二、核心思想

### 2.1 三层约束：形状、位置、时序

Tensor Core 不是一个「更快的乘法器」，它是一个**形状被焊死的流水线**。想让一条指令吃下 26 万个乘加，操作数就必须按它规定的样子摆好。约束分三层：

**第一层：形状。** M 恒等于 64，N 从 8 到 256（步长 8），K 由输入位宽决定。这不是工程选择，是被一块固定的硅面积锁死的。CUTLASS 上游 `mma_traits_sm90_gmma.hpp` 里这一串 `Shape_MNK` 就是全部的合法形状：

```text
:490   Shape_MNK = Shape<_64,_8,_16>       f16,  N 最小值
:903   Shape_MNK = Shape<_64,_256,_16>     f16,  N 最大值
:2289  Shape_MNK = Shape<_64,_256,_8>      tf32, K 减半
:2341  Shape_MNK = Shape<_64,_8,_32>       fp8/int8, K 翻倍
```

M=64 这个数从哪来？`ThrID = Layout<_128>` —— 一个 warpgroup 是 4 个 warp × 32 lane = 128 线程。M 和线程数是绑定的：64 行的累加器要在 128 个线程之间均分。

**第二层：位置。** 操作数不能随便放。A 和 B 通常必须以「SMEM 描述符」的形式给出，也就是**它们得在共享内存里，且必须按 GMMA 能解读的布局摆放**。这是本期最硬核的部分，第 2.3 节展开。

**第三层：时序。** `wgmma` 是异步的，指令发出去就返回，你得靠 `commit_group` / `wait_group` 管生命周期。这一点和 044 期讲的 `cp.async` 是同一套哲学。

### 2.2 位宽如何直接决定一条指令的大小

这是我觉得整件事里最漂亮的一处因果链，四步全是算术。

A 操作数的一个 64×K 分块，要摊到 128 个线程的寄存器里。每个线程手上**固定是 4 个 32-bit 寄存器**（这是 GMMA 的接口约定）。那么这 4 个寄存器能装多少 K？

| 精度 | 每个元素占 | 4 个寄存器 = 128 bit 能装 | 每线元素数 | 于是 K = |
|---|---|---|---|---|
| fp16 / bf16 | 16 bit | 128 bit | $64K/128$ | **16** |
| tf32 | 32 bit | 128 bit | — | **8** |
| fp8 / int8 | 8 bit | 128 bit | — | **32** |

K 就是这么被位宽定死的。而 K 一确定，一条指令的 FMA 数就确定了：

$$\text{FMA}_{m64n256} = 64 \times 256 \times K$$

fp16 是 262144，tf32 直接腰斩到 131072，fp8 翻倍到 524288。**吞吐比也正好是 2:1:4**：989.5 / 494.7 / 1979 TFLOPS。

看到没？**位宽 → 寄存器能装的 K → 每条指令的 FMA → 吞吐**，这条链一点没浪费，也没一点含糊。硬件规格表上的 TFLOPS 不是标出来的，是这个乘法算出来的。

### 2.3 位置：64 位描述符，和它必须解决的 bank 冲突

GMMA 不接收指针。它接收一个 **64 位的描述符**，在里面编码「矩阵从 SMEM 的哪里开始、行怎么跨、用哪种 swizzle」。CUTLASS 的位段定义（`cute/arch/mma_sm90_desc.hpp:109-125`，上游 main 分支）：

```text
bit [0,14)   start_address        矩阵起始地址，单位 16 B（低 4 位不编码）
bit [16,30)  leading_byte_offset  leading 维字节偏移，单位 16 B
bit [32,46)  stride_byte_offset   stride 维字节偏移，单位 16 B
bit [49,52)  base_offset          只在 SWIZZLE_128B / 64B 下有效
bit [62,64)  layout_type          0=INTERLEAVE 1=SWIZZLE_128B 2=64B 3=32B
```

两个立刻能推出来的后果：

1. **地址的单位是 16 B，字段只有 14 bit** → 最大可寻址 $2^{14} \times 16\,\text{B} = 256\,\text{KB}$。刚好罩住 Hopper 的一个 SM 的 227 KB SMEM。不是巧合，是按这个上限设计的。
2. **地址必须 16 B 对齐**。对齐摆不好，描述符就编不出来。

现在讲那个最容易被跳过的问题：**为什么非要 swizzle？**

因为你把 tile 放进 SMEM 的那一刻，就已经把 bank 冲突写死了。假设 SMEM 里放 A 的一个 K-major 分块，每行 128 B（= 64 个 fp16）。128 B 恰好等于 32 个 bank × 4 B —— **行距 128 B 意味着每一行的起始 bank 都是同一个**。

于是「读同一列上的 8 个不同行」这个动作，8 个地址全落在同一组 bank 上，只能排队。

swizzle 干的事很小：把**每个 128 B 行内部的 8 个 16 B 段，按行号低 3 位旋转**。物理地址就一句：

$$\text{phys}(r, s) = r \times 128 + \big(s \oplus (r \bmod 8)\big) \times 16 + \text{inner}$$

XOR 是自逆的，所以这是个双射，不丢数据，只是换了个摆放顺序。下面这个脚本算它到底省了多少个周期 —— bank 模型是 32 bank × 4 B，每周期每 bank 服务 1 个 word：

```python
"""SWIZZLE_128B 到底省了几个周期。bank 模型：32 bank × 4 B，每周期每 bank 服务 1 个 word。"""
NBANK, WORDB, ROW_B, SEG_B = 32, 4, 128, 16

def addr(row, seg, swizzle):
    """K-major tile 的字节地址；seg 是行内第几个 16 B 段（0..7）。"""
    if swizzle:
        seg = seg ^ (row & 7)          # 核心：段号按行号低 3 位旋转
    return row * ROW_B + seg * SEG_B

def cycles(rows, nseg, swizzle):
    """warp 的 len(rows)*nseg 个 lane 各读一整段 16 B，返回最忙的那个 bank 要服务几个 word。"""
    per = {}
    for r in rows:
        for s in range(nseg):
            base = addr(r, s, swizzle)
            for w in range(SEG_B // WORDB):
                a = base + w * WORDB
                per.setdefault((a // WORDB) % NBANK, set()).add(a)
    return max(len(v) for v in per.values())

print(f"{'场景':>20} {'无 swizzle':>11} {'SWIZZLE_128B':>13} {'理想值':>7}")
for name, rows, nseg in (("8 行 × 1 段", list(range(8)), 1),
                         ("8 行 × 2 段", list(range(8)), 2),
                         ("8 行 × 4 段", list(range(8)), 4)):
    c0 = cycles(rows, nseg, False)
    c1 = cycles(rows, nseg, True)
    ideal = len(rows) * nseg * SEG_B // WORDB // NBANK
    print(f"{name:>20} {c0:>9} cyc {c1:>10} cyc {ideal:>7}")

print("\n8 行 × 1 段时每行落在哪些 bank：")
for sw in (False, True):
    banks = sorted({addr(r, 0, sw) // WORDB % NBANK for r in range(8)})
    print(f"  {'SWIZZLE_128B' if sw else 'no swizzle':>13}: {banks}")
```

```text
                  场景   无 swizzle  SWIZZLE_128B     理想值
           8 行 × 1 段         8 cyc          1 cyc       1
           8 行 × 2 段         8 cyc          2 cyc       2
           8 行 × 4 段         8 cyc          4 cyc       4

8 行 × 1 段时每行落在哪些 bank：
     no swizzle: [0]
   SWIZZLE_128B: [0, 4, 8, 12, 16, 20, 24, 28]
```

三行都成立：**无 swizzle 时不管读几段都是 8 个周期（因为全挤在 bank 0 那一组）；加了 swizzle 之后，周期数恰好等于理论下限**。第二张表是微观解释：不加 swizzle，8 行的首地址全在 bank 0；加完之后散成 `0,4,8,...,28` 八个。

关于 swizzle 我要诚实标注两件事：
- 上面的 `s ^ (r & 7)` 是 **16 B 段粒度**的等价描述，是理解这件事最清爽的形式。
- CUTLASS 实际用的是 CuTe 的 `Swizzle<3,4,3>`（`mma_traits_sm90_gmma.hpp:78`，128 B 对应 `Layout_MN_SW128_Atom_Bits`），它作用在 **16 bit 元素粒度**上，比上面的式子细一档（B=3 位 XOR、M=4 表示最低 4 bit 不动、S=3 表示 XOR 源在上方第 3 位）。两者是同一个 3 位 XOR 结构，只是粒度不同——GMMA 的 core matrix 以 16 bit 元素为最小交换单位，所以要用细的那一档。

### 2.4 一条 wgmma 要 138 个 cycle，而发射只要 1 个

把第二节的算术收一下。这段代码把「一条指令多大事、吃掉多少寄存器」全部算出来：

```python
"""一条 wgmma 的大小与寄存器预算。形状取自 CUTLASS 上游 mma_traits_sm90_gmma.hpp。
H100 SXM5 官方规格：132 SM、boost 1.98 GHz、FP16 Tensor dense 989.5 TFLOPS。"""
SM, CLK, FP16_TC, REGS_SM = 132, 1.98e9, 989.5e12, 65536
fma_per_clk = FP16_TC / SM / CLK / 2          # 每 SM 每 cycle 的 fp16 FMA 数
print(f"H100 每 SM 每 cycle: {fma_per_clk:.0f} 次 fp16 FMA")

print(f"\n{'形状':>14} {'FMA/指令':>10} {'占满 TC':>9} {'acc reg/线程':>12}")
for n in (8, 64, 128, 256):
    fma = 64 * n * 16
    acc = 64 * n / 128                        # fp32 累加器，128 个线程均分
    print(f"{f'm64n{n}k16':>14} {fma:>10} {fma/fma_per_clk:>8.1f}cyc {acc:>12.0f}")

print("\n== 4 个寄存器能装多少 K（这就是 K 的来源）==")
for name, bits, k in (("fp16", 16, 16), ("tf32", 32, 8), ("fp8", 8, 32)):
    elem = 64 * k // 128                      # A 是 64×K，摊到 128 个线程
    print(f"  {name:>5}: K = {k:>2}，每线程 {elem:>2} 个元素 = {elem*bits/32:.0f} 个 32-bit reg")

print("\n== 寄存器预算：一个 SM 有 65536 个 32-bit 寄存器 ==")
for n in (64, 128, 256):
    total = 64 * n / 128 + 32                 # 累加器 + 指针/循环变量等
    print(f"  m64n{n}k16: {total:>3.0f} reg/线程 → 每 SM 最多 {REGS_SM // (128*total):.0f} 个 warpgroup"
          f"（按上限 255 算则 {REGS_SM // (128*255):.0f} 个）")
```

```text
H100 每 SM 每 cycle: 1893 次 fp16 FMA

            形状     FMA/指令     占满 TC   acc reg/线程
      m64n8k16       8192      4.3cyc            4
     m64n64k16      65536     34.6cyc           32
    m64n128k16     131072     69.2cyc           64
    m64n256k16     262144    138.5cyc          128

== 4 个寄存器能装多少 K（这就是 K 的来源）==
   fp16: K = 16，每线程  8 个元素 = 4 个 32-bit reg
   tf32: K =  8，每线程  4 个元素 = 4 个 32-bit reg
    fp8: K = 32，每线程 16 个元素 = 4 个 32-bit reg

== 寄存器预算：一个 SM 有 65536 个 32-bit 寄存器 ==
  m64n64k16:  64 reg/线程 → 每 SM 最多 8 个 warpgroup（按上限 255 算则 2 个）
  m64n128k16:  96 reg/线程 → 每 SM 最多 5 个 warpgroup（按上限 255 算则 2 个）
  m64n256k16: 160 reg/线程 → 每 SM 最多 3 个 warpgroup（按上限 255 算则 2 个）
```

第二段是第 2.2 节那条因果链的算术验证：**fp16 / tf32 / fp8 的 A 分块分别是 64×16 / 64×8 / 64×32，摊到 128 个线程之后，恰好都是 4 个 32-bit 寄存器。** 一个不差，一个不多。这就是 K 只能取 16/8/32 的原因——不是设计者挑的数，是 4 个寄存器除以元素位宽除出来的。

真正的结论在第四段：

- **m64n256k16 的累加器要 128 个寄存器/线程。** 一个线程总共 255 个。也就是说一条指令的累加器就占掉一半的寄存器文件。
- 一个 warpgroup 128 线程，按 160 reg/线程算要 20480 个寄存器，而一个 SM 有 65536 个 → **最多同时存在 3 个 warpgroup**（这还是乐观估计，实际按 255 硬上限算是 2 个）。
- m64n128 松一档（96 reg/线程 → 5 个 warpgroup），m64n64 更松（8 个）。**N 越大算得越猛，但同时也把自己能并发的份数挤掉了** —— 这个取舍是 Hopper GEMM 调参的主战场。

这就是 Hopper GEMM kernel 的全部形状：**一个 SM 上跑 1 个 producer warpgroup（专职搬数，用 TMA）+ 2 个 consumer warpgroup（专职算）**。不是为了好看，是寄存器只够这么多。

对比 045 期讲的 occupancy：那里寄存器压力决定能塞几个 block；这里它决定**能不能塞第二个 warpgroup**。同一把锁，锁得更死。

---

## 三、在 PyTorch 里怎么碰得到它

### 3.1 大多数人碰不到，因为 `torch.matmul` 已经帮你选好了

PyTorch 的 `matmul` 走 cuBLAS（`aten::mm` → `at::cuda::blas::gemm`），cuBLAS 在 Hopper 上自动选 wgmma 的 kernel。你不用写一行 PTX。

但「自动选」意味着**你看不见它选了哪个精度**。这是本节唯一需要你记住的坑。

### 3.2 精度开关：卷积默认吃 TF32，matmul 默认不吃

TF32 的尾数是 10 位。它换来了 2 倍吞吐（494.7 vs 247.35... 实际是相对 FP32 CUDA Core 的 7.4 倍），代价是精度。误差到底多大？下面这段用一个自写的尾数截断模拟 Tensor Core 的「输入截断 + 高精度累加」：

```python
"""Tensor Core 的输入精度：尾数截断后，误差有多大，以及它随 K 怎么变。
Tensor Core 的行为：输入按该精度的尾数截断，乘加在 fp32 中累加 —— 这里精确复现这两步。"""
import numpy as np
rng = np.random.default_rng(0)

def trunc(x, keep):
    """把 fp32 尾数截到 keep 位（就近舍入）。keep=23 时原样返回。"""
    drop = 23 - keep
    if drop == 0:
        return x.astype(np.float32)
    u = x.astype(np.float32).view(np.uint32)
    half = np.uint32(1) << np.uint32(drop - 1)
    return ((u + half) & (np.uint32(0xFFFFFFFF) << np.uint32(drop))).view(np.float32)

M = N = 256
print(f"{'精度':>8} {'尾数位':>6} {'K=64':>10} {'K=512':>10} {'K=4096':>10}")
for name, bits in (("fp32", 23), ("tf32", 10), ("fp16", 10), ("bf16", 7)):
    row = []
    for K in (64, 512, 4096):
        A = rng.standard_normal((M, K))
        B = rng.standard_normal((K, N))
        ref = A @ B                                          # float64 参考
        got = trunc(A, bits).astype(np.float64) @ trunc(B, bits).astype(np.float64)
        row.append(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    print(f"{name:>8} {bits:>6} " + "".join(f"{v:>10.2e}" for v in row))

print("\n== fp16 的指数只有 5 位（最大 65504），这才是它和 tf32 的真正区别 ==")
for scale in (1.0, 1e3, 1e4, 1e5):
    x = (rng.standard_normal(100000) * scale).astype(np.float32)
    h = x.astype(np.float16)
    print(f"  scale={scale:>7.0e}  inf {np.mean(np.isinf(h))*100:5.2f}%"
          f"   下溢为 0 {np.mean((h == 0) & (x != 0))*100:5.2f}%")
```

```text
      精度    尾数位       K=64      K=512     K=4096
    fp32     23   3.60e-08  3.60e-08  3.55e-08
    tf32     10   2.95e-04  2.92e-04  2.94e-04
    fp16     10   2.98e-04  2.94e-04  2.94e-04
    bf16      7   2.36e-03  2.37e-03  2.36e-03

== fp16 的指数只有 5 位（最大 65504），这才是它和 tf32 的真正区别 ==
  scale=  1e+00  inf  0.00%   下溢为 0  0.00%
  scale=  1e+03  inf  0.00%   下溢为 0  0.00%
  scale=  1e+04  inf  0.00%   下溢为 0  0.00%
  scale=  1e+05  inf 51.38%   下溢为 0  0.00%
```

三个能直接用的结论：

1. **tf32 和 fp16 的误差一模一样**（2.9e-4），因为它们的尾数都是 10 位。差异全在指数范围：fp16 的 5 位指数在 scale=1e5 时炸掉一半（51.38% 变 inf）。**这就是 002 期讲 AMP 时为什么要 loss scaling 的定量来源。**
2. **bf16 差 8 倍**（7 位 vs 10 位尾数，理论上 $2^3 = 8$，实测 2.36e-3 / 2.95e-4 = 8.0，完全吻合）。
3. **误差完全不随 K 增长。** K 从 64 到 4096 涨 64 倍，误差一动不动。原因很简单：累加是 fp32 做的，累加误差被压到 3.6e-8 量级，地板完全由输入的尾数截断决定。**这就是「fp32 累加器」存在的全部理由** —— 没有它，K 一大误差就线性堆上去。

这条「误差与 K 无关」的性质非常好用：它意味着你可以放心地把 K 做得很大（Hopper 的 GEMM 主循环里 K 方向一次吃 64、128 个），精度不会因为 K 大而崩。

现在回到默认开关。PyTorch 2.14 在本机读出来是这样：

```python
import torch
print("matmul.allow_tf32       =", torch.backends.cuda.matmul.allow_tf32)
print("cudnn.allow_tf32        =", torch.backends.cudnn.allow_tf32)
print("matmul.fp32_precision   =", torch.backends.cuda.matmul.fp32_precision)
print("get_float32_matmul_precision() =", torch.get_float32_matmul_precision())
```

```text
matmul.allow_tf32       = False
cudnn.allow_tf32        = True
matmul.fp32_precision   = none
get_float32_matmul_precision() = highest
```

**`cudnn` 的 TF32 默认是开的，`matmul` 的默认是关的。**

所以同一个模型里，卷积层悄悄用着 10 位尾数，线性层老老实实用 fp32。你要是拿卷积的数值去 calibrate 量化的 scale，那 2.9e-4 的相对误差就是白送的。这是我觉得整个 PyTorch 里最不对称的一个默认值。

### 3.3 算术强度账本：为什么 tile 必须大

Tensor Core 的算力上去了，供数压力就跟着上去了。043 期讲过屋脊点，这里把 GEMM 的算术强度真算一遍。

下面这个脚本跑的是真的分块矩阵乘，但**统计的是真正从 global 读进寄存器的字节**（用 ndarray 切片的 `nbytes` 累加，不是估出来的）：

```python
"""分块如何改变「每个 FLOP 要搬多少字节」。真跑一遍 blocked GEMM 并统计实际读入字节。"""
import numpy as np
M = N = K = 512
A = np.ones((M, K), dtype=np.float32)
B = np.ones((K, N), dtype=np.float32)
BK = 64                                   # K 方向也分块，模拟 GEMM 主循环

def blocked(tm, tn):
    C = np.zeros((M, N), dtype=np.float32)
    gb = 0
    for i0 in range(0, M, tm):
        for j0 in range(0, N, tn):
            acc = np.zeros((tm, tn), dtype=np.float32)
            for k0 in range(0, K, BK):
                at = A[i0:i0 + tm, k0:k0 + BK]
                bt = B[k0:k0 + BK, j0:j0 + tn]
                gb += at.nbytes + bt.nbytes       # 真正读进来的字节
                acc += at @ bt
            C[i0:i0 + tm, j0:j0 + tn] = acc
    return gb, C

print(f"{'tile':>9} {'global 读':>10} {'AI(FLOP/B)':>11}")
for tm, tn in ((8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (128, 256)):
    gb, C = blocked(tm, tn)
    assert np.allclose(C, K)              # 结果必须对，否则账本是假的
    print(f"{f'{tm}x{tn}':>9} {gb/2**20:>7.1f} MiB {2*M*N*K/gb:>11.1f}")
```

```text
     tile   global 读  AI(FLOP/B)
      8x8   128.0 MiB         2.0
    16x16    64.0 MiB         4.0
    32x32    32.0 MiB         8.0
    64x64    16.0 MiB        16.0
  128x128     8.0 MiB        32.0
  128x256     6.0 MiB        42.7
```

算术强度的公式是

$$\text{AI} = \frac{t_m t_n}{2(t_m + t_n)}$$

`128×128` 给 32，`128×256` 给 42.7，脚本算出来的和公式一分不差。

现在把 043 期的屋脊点摆过来比：

| 精度 | H100 屋脊点（FLOP/B） | `128×128` tile 的 AI | 差多少 |
|---|---|---|---|
| FP32 CUDA Core | 20.0 | 32 | 够 |
| TF32 Tensor | 147.6 | 32 | 缺 4.6 倍 |
| FP16 Tensor | 295.4 | 32 | 缺 9.2 倍 |
| FP8 Tensor | 590.7 | 32 | 缺 18 倍 |

**这就是 Hopper GEMM 长那样的原因。**

- `128×128` tile 喂 FP32 完全够，喂 FP16 差 9 倍。所以要靠 **L2 缓存**再复用一轮（CTA tile 128×256 时，B 会被相邻 CTA 从 L2 读到，不进 DRAM）。
- 更彻底的办法是 046 期的 **TMA multicast**：一个 cluster 里的 CTA 读同一块 B，让 L2→SM 的流量除以 cluster size。这不是为了省 DRAM 带宽，是为了把「AI 缺口」补上。
- K 方向不能无限大。整个 tile 的 SMEM 占用是 $(t_m + t_n) \times B_K \times \text{elem}$，Hopper 只有 227 KB。

一条完整的因果链：**Tensor Core 把算力提高了 14 倍 → 屋脊点跟着提高了 14 倍 → tile 的 AI 一下不够了 → 逼出了 TMA、multicast、cluster、warp specialization 这一整套 Hopper 特性。**

---

## 四、围绕这个领域展开

### 4.1 它在整条链上的位置

```
HBM ──(TMA/cp.async, 044 期)──► SMEM ──(wgmma 描述符, 本期)──► Tensor Core ──► 累加器寄存器
       ▲                            ▲                                              │
       │                        (swizzle 布局, 本期 2.3)                          │
  cluster multicast                                          (epilogue, 040 期融合)
  (046 期)                                                          │
                                                                    ▼
                                                              写回 HBM
```

这张图里，本期只负责中间那一段：**SMEM 里的字节，怎么变成 Tensor Core 吃的操作数**。左边是 044/046 的地盘，右边是 040（epilogue 融合）的地盘。

### 4.2 和已发各期的关联点

- **043 期（内存层次与访存模式）**：那期给了屋脊点和 H100 的带宽/算力数字。本期把 GEMM 的 AI 真算出来，直接对上屋脊点，得到「缺 9.2 倍」这个具体缺口。**没有那期的屋脊点，本期的分块账本就没有刻度**。
- **044 期（异步拷贝与软件流水线）**：cp.async / TMA 解决的是「等数据的时间怎么填掉」，上限是 $(c+p)/\max(c,p)$。本期给出了「为什么 c 必须被彻底藏掉」的定量理由：一条 wgmma 要 138 个 cycle，而这 138 个 cycle 里 SMEM 必须已经在供应下一批。
- **046 期（线程块簇与 DSMEM）**：multicast 把 L2→SM 的流量除以 cluster size。本期 3.3 节的表说明这不是锦上添花 —— FP16 下 AI 缺口 9.2 倍，靠单 CTA 的 tile 补不上。
- **045 期（SIMT 与 Occupancy）**：那期讲寄存器压力决定 block 数。本期 2.4 节算出来 m64n256k16 的累加器要 128 reg/线程，一个 SM 只放得下 2~3 个 warpgroup —— 同样的锁，锁在更关键的位置。
- **005 期（Flash Attention）/ 038 期（SDPA 后端）**：Flash Attention 之所以快，一半功劳在 Tensor Core 上。它把 attention 拆成 $QK^\top$ 和 $PV$ 两个 64×N 的 GEMM，正好对上 GMMA 的 M=64。这也是为什么 flash 后端在 fp16/bf16 上收益巨大、在 fp32 上收益打折。
- **002 期（AMP）**：本期的精度表解释了 bf16 和 fp16 的取舍不是口味问题 —— 尾数差 3 位（8 倍误差），指数差 3 位（fp16 会溢出）。H100 上 bf16 和 fp16 的吞吐完全一样（都是 989.5），所以**除非你要极致精度，就该用 bf16**，省掉 loss scaling 那一层。
- **008 期（Triton）**：你写 `tl.dot(a, b)`，Triton 在 sm90 上就落成 wgmma，在 sm80 上落成 `mma.sync`。Triton 帮你选形状、算 swizzle、生成描述符 —— 但它不会告诉你这些约束存在。本期就是那些约束。

### 4.3 哪些是 NVIDIA 特有，哪些能迁移

| 概念 | 归属 | 迁移说明 |
|---|---|---|
| M=64 固定、N 8~256 步长 8 | **NVIDIA 特有**（Hopper wgmma） | Blackwell 的 `tcgen05.mma` 换了 M/N 的约束；AMD 的 MFMA 是 16×16 的 tile；昇腾的 Cube 是 16×16×16 |
| 64 位 SMEM 描述符 + 14 bit 地址字段 | **NVIDIA 特有** | 别的硬件用指针或不同的描述符格式 |
| `Swizzle<3,4,3>` 这个具体参数 | **NVIDIA 特有** | 但「用 XOR 打散 bank」的思想通用：AMD 用 `ds_bpermute` + padding，CPU 用 padding |
| 32 bank × 4 B、bank 冲突的动机 | **通用抽象** | 任何有 banked SMEM 的硬件都成立。本机 M5 的 cacheline 是 128 B，与 GPU 的 128 B 行同构 |
| 一条指令做 $t_m t_n k$ 次 FMA | **通用抽象** | 这是「矩阵单元」的定义，CPU 的 SME、昇腾的 Cube 都是这个模式 |
| 位宽 → K → 吞吐的因果链 | **通用抽象** | 位宽减半、K 翻倍、吞吐翻倍，在哪都一样 |
| 累加器占寄存器 → occupancy 受限 | **通用抽象** | 只是具体数字不同 |
| 屋脊点 / 算术强度 | **通用抽象** | 043 期已经说过一次，这里再用 |

### 4.4 一条容易被忽略的推论：为什么 decode 用不上 Tensor Core

043 期算过，decode 阶段单个 token 的 attention 算术强度**恒等于 1**，与 H、S、d 全部无关。Tensor Core 要 AI ≥ 295（FP16）才划算。

于是 decode 的 attention 只能落到 CUDA Core 上，或者靠 batch 把 AI 拉起来（continuous batching，028 期）。**这也是为什么 027~036 那十期 KV Cache 优化的方向全是「省字节」，而不是「提算力」** —— 那个工作负载根本不在 Tensor Core 的能力范围内。

反过来，prefill 阶段是纯 GEMM，AI 几百上千，Tensor Core 满负荷。**同一个模型，两个阶段一个吃 Tensor Core 一个吃带宽**，这就是 PD 分离（030 期）在硬件层面最根本的动机。

---

## 五、什么时候该用 / 不该用

**该用（也就是「该指望 Tensor Core」）**：

- 大 GEMM：M、N、K 都 ≥ 256。这时 AI 高，Tensor Core 是唯一出路。
- prefill、训练的前反向、MoE 的 expert GEMM（026 期）。
- 只要你能把 dtype 降到 fp16/bf16 且精度可接受。

**不该用 / 用了白用**：

- **K 很小**（decode 的 GEMV，K = hidden 但 M = 1）。M=1 连 GMMA 的 M=64 都填不满，硬件会直接退回到别的路径。你写 `@` 也不会变快。
- **算术强度本来就低**的逐元素 / 归约算子。Tensor Core 帮不上，040 期的融合才是解法。
- **精度敏感且无法补偿**的场景（大 K 的 fp32 累加链、需要 exact 复现的数值）。tf32 的 2.9e-4 在有些地方是要命的。
- **小 batch 的推理**：batch=1 时整个 forward 都是 GEMV，Tensor Core 全程闲着。

---

## 六、常见坑

### 坑 1：以为手工分块一定更快

我自己踩的。同样 512³ 的 fp32 GEMM，`cc -O3` 编出来：

```text
naive  i-k-j 三重循环  32.55 GFLOPS
手写 4×4 寄存器分块     22.25 GFLOPS   ← 更慢
```

反直觉，但原因很实在：`-O3` 已经把 naive 版的 `j` 循环向量化了，一次读 4~8 个连续的 `B[k][j]` 再广播 `a`，那本身就是一块 tile；而我手写的 4×4 版里 `A[i+1][k]`、`A[i+2][k]` 是跨行访问（stride = n），多出来的 4 次非连续加载把收益吃光了。

**分块的收益不是免费的，要配套数据重排（把 A 转成按 tile 连续存放）。** cuBLAS/CUTLASS 里那个 `ldmatrix` + swizzle + 描述符的整套机制，本质就是在为「分块之后访存还连续」付账。

### 坑 2：以为换了 dtype 就换了指令路径

本机 CPU 上测 fp32 / bf16 的 GEMM：fp32 有 2 TFLOPS 量级，**bf16 掉到个位数 GFLOPS**，差三个数量级。

不是 bf16 慢，是这台机器的 CPU **没有 bf16 的矩阵单元路径**，`aten::mm` 退化成了逐元素实现。GPU 上不会这么惨（H100 的 bf16 和 fp16 同速），但在别的硬件上（包括各种国产加速器）这是常态。

**判据**：换成低精度后先量一遍，别假设「bf16 应该比 fp32 快」。验收标准是「和该硬件的 Tensor Core 峰值比」，不是「和 fp32 比」。

### 坑 3：wgmma 的 commit/wait 组被用超

`wgmma` 是异步的，配对的语义是：

```text
wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 {...}, descA, descB;
wgmma.commit_group.sync.aligned;          // 把已发出的 wgmma 归成一组
wgmma.wait_group.sync.aligned N;          // 等到「最多还剩 N 组未完成」
```

**未完成的组数上限是 8。** 超了这个数，行为是未定义的（不是报错，是算错）。所以主循环里 `wait_group 1` 这种写法很常见 —— 留着 1 组在飞，重叠下一轮的搬运。

这一段我**没有在真机上验证**（本机无 CUDA），属于文档级事实，标注在此。

### 坑 4：描述符的地址是 16 B 单位，且只有 14 bit

`start_address` 字段是 14 bit、单位 16 B。两个后果：

- 地址必须 **16 B 对齐**，否则低 4 位丢掉、读到错位的数据（不报错）。
- 可寻址范围是 256 KB。Hopper 的 SMEM 是 227 KB，刚好；但如果你在一个 SMEM 窗口里做偏移（比如双缓冲的两块 tile 相隔很远），**偏移量本身也会溢出**。

这类「字段宽度不够」的 bug 最难查，因为它不崩、只是数字不对。写自己的描述符构造代码时，务必对着 `mma_sm90_desc.hpp` 的位段一行行核。

### 坑 5：swizzle 和你的写入布局没对上

swizzle 是**布局约定**，不是硬件自动帮你做的事。你用 TMA 搬数进 SMEM 时声明了 SWIZZLE_128B，那描述符里也必须写 `layout_type = 1`，两边不一致就读到乱序数据。

Triton / CUTLASS 帮你管这个一致性，但只要你手写一层，这就是最常见的错误来源。**检查方法**：把 SMEM 里的一块 tile dump 出来，和期望的逻辑布局逐元素比。

---

## 七、一句话总结

**Tensor Core 把一次乘加放大了 26 万倍，于是「算」这件事彻底退出了瓶颈列表 —— 剩下所有复杂度（swizzle、描述符、TMA、multicast、warp specialization）都是在为「准时喂上一个 64×N×16 的块」付账。**

---

## 八、今日练习

<details>
<summary>今日练习（点开看参考答案）</summary>

**题 1**：m64n128k16 和 m64n256k16 两条 wgmma，前者一条指令 131072 次 FMA、后者 262144 次。为什么实际 kernel 里 m64n256 不一定更快？

**答**：因为累加器。m64n256 的 fp32 累加器要 128 reg/线程，m64n128 只要 64。按 2.4 节的算术，m64n256 一个 SM 只放得下 2~3 个 warpgroup，m64n128 能放 8 个（乐观估计）。warpgroup 少了，就没有足够的独立工作流去填满 138 个 cycle 的延迟 —— 你会退化成「一条 wgmma 做完等一条」。所以 CUTLASS 的 Hopper GEMM 常常用 `128×256` 的 CTA tile 但拆成两个 `64×256` 的 wgmma 交替发射，或者干脆选 `64×128` 配更多 stage。

**题 2**：为什么 fp16 的 GEMM 在 K=4096 时误差不会比 K=64 大？（提示：本期的表）

**答**：因为累加是 fp32。误差来源分两部分：输入尾数截断（$2^{-11}$ 相对量级，与 K 无关）和累加舍入（$\sqrt{K} \cdot 2^{-24}$ 量级）。K=4096 时后者是 $64 \times 6\times10^{-8} \approx 4\times10^{-6}$，仍比前者的 $3\times10^{-4}$ 小两个数量级。所以误差地板由输入位宽决定，累加器只要「够宽」就行。**这也是为什么 Tensor Core 一定要配 fp32 累加器** —— 用 fp16 累加的话第二项会变成 $\sqrt{K}\cdot2^{-11}$，K=4096 时直接到 3e-2，比输入误差大 100 倍。

**题 3**：你要在 H100 上跑一个 M=8、N=4096、K=4096 的 GEMM（比如 batch=8 的推理最后一层）。Tensor Core 能帮上忙吗？怎么改？

**答**：直接算 AI：$\text{FLOPs} = 2 \times 8 \times 4096 \times 4096 \approx 2.7\times10^{8}$，字节数 $\approx (8\times4096 + 4096\times4096)\times 2 \approx 3.4\times10^{7}$，AI ≈ 8。FP16 的屋脊点是 295，**差 37 倍**。而且 M=8 连 GMMA 的 M=64 都填不满，硬件会退回到 `mma.sync`（m16n8k16）甚至更窄的路径。

改法只有一条：**增加 M**。要么把 batch 做大（continuous batching，028 期），要么把多个小 GEMM 拼成一个（比如把 MoE 里同一 expert 的多个 token 拼起来，026 期）。这是「小 batch 推理为什么喂不饱 GPU」的算术本质。

**题 4**：本机实测 n=32 的 GEMM 只有 6~10 FMA/cycle（三次独立运行的区间），n=1024 是 224。这个「拐点」在 GPU 上对应什么？

**答**：对应 tile 太小、Tensor Core 的固定开销摊不薄。n=32 时一次 GEMM 的总工作量太小，矩阵单元的启动/排空（pipeline fill/drain）占了大头。GPU 上同理：`m64n8k16` 那条指令只要 4.3 个 cycle 就做完了，但发出它、读描述符、等 fence 的开销远不止 4.3 个 cycle —— 所以 N=8 的形状几乎只在需要凑尾巴时用。**这就是为什么所有 GEMM kernel 的 tile 都往大了选**。

</details>

---

## 附：本期数字的出处分级

| 数字 | 级别 | 出处 |
|---|---|---|
| 单核 224 FMA/cycle（n=128 已超 SIMD 上限的 6 倍） | **本机实测** | 本文 1.1 节脚本，Apple M5，torch 2.14.0 |
| naive 32.55 / 手写分块 22.25 GFLOPS | **本机实测** | `cc -O3 -march=native`，512³ fp32 |
| tf32/fp16 2.9e-4、bf16 2.4e-3、与 K 无关 | **本机实测** | 本文 3.2 节脚本，numpy 模拟 |
| fp16 在 scale=1e5 时 51% 溢出 | **本机实测** | 同上 |
| AI 2.0 → 42.7、字节 128→6 MiB | **本机实测** | 本文 3.3 节脚本，真跑 blocked GEMM |
| 8 行 × 1 段 8 cyc → 1 cyc | **确定性算术** | 本文 2.3 节脚本，bank 模型 |
| 1893 FMA/SM/cycle、138.5 cycle/指令、128 reg | **官方规格 + 算术** | H100 SXM5 datasheet，132 SM @ 1.98 GHz |
| M=64/N≤256/K=16·8·32、SWIZZLE 枚举、位段 | **上游源码** | CUTLASS `mma_traits_sm90_gmma.hpp`、`mma_sm90_desc.hpp`（main 分支） |
| `cudnn` TF32 默认开、`matmul` 默认关 | **本机实测** | torch 2.14.0 属性读取 |
| wgmma commit/wait 组上限 8 | **文档级（未实测）** | PTX ISA / CUTLASS 用法，本机无 CUDA |
| Apple M5 CPU 侧用的是 SME | **推断** | 由 224 FMA/cycle 反推，Apple 未公开 |

> 说明：本机（Apple M5，torch 2.14.0）无 CUDA、无 triton、无 nvcc，所以所有 NVIDIA 侧的 kernel 行为**没有真机验证**，一律标为「文档级」或「官方规格 + 算术」。涉及 GMMA 形状与描述符位段的引用，来自 CUTLASS 上游 main 分支源码，与本机 torch 版本无关。
