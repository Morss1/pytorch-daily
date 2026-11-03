# PyTorch 每日一课 · 第 048 期

## Epilogue：累加器离开 Tensor Core 之后

> **元信息**
> 日期：2026-11-03 · 难度：⭐⭐⭐⭐ · 预计阅读：45 分钟
> 前置知识：`wgmma` 与累加器布局（047 期）、SMEM / 流水线（043 / 044 期）、
> 占用率与寄存器压力（045 期）、算子融合的动机（040 期）
> 关联：第 047 期 · 那期结尾停在「累加器落在寄存器里」，本期从那一刻接上；
> 同时把 040 的「融合少搬字节」、043 的屋脊点、045 的寄存器预算串到同一个账本上

---

## 一、这个领域解决什么问题

047 期算过一笔账：一条 `wgmma.m64n256k16` 指令做 262144 次 FMA，需要 138.5 个 cycle
才做完，但只占 **1 个 issue slot**。也就是说，从 Hopper 开始，「算」这件事在 issue
层面已经彻底退出了瓶颈列表 —— 你几乎不需要为「发指令」操心，只要保证操作数就位。

那么剩下的事情是什么？

一条 GEMM 的时间轴上，`wgmma` 结束的那一刻，**数据全部躺在寄存器里**。一个 128×256
的 CTA tile 有 32768 个 fp32，整个 CTA 256 个线程，**每个线程手里 128 个 fp32**。
接下来必须发生三件不愉快的事：

1. **形状对不上。** 寄存器里的布局是 MMA 指令规定的 8×8 交织（每线程 2 个相邻列、
   跨 8 行），HBM 想要的是「128 字节连续」。两边都不肯妥协。
2. **数量对不上。** 128 个寄存器占掉了单线程 255 个寄存器上限的一半。不把这 128 KiB
   腾出去，下一个 tile 的累加器就没地方放，流水线直接停。
3. **通路对不上。** 想把结果用 TMA store 一条指令甩出去？TMA 的源**必须在 SMEM 里**，
   寄存器不能直连。

从 MMA 指令完成，到数据真正落进 HBM，中间这一整段工序，就叫 **epilogue（尾声）**。
它不是「顺手加个 bias」，而是一个有独立寄存器预算、独立 SMEM 预算、独立流水线、
在 Hopper 上还占着**独立 warpgroup** 的完整阶段。

这一期讲的就是这一段：它为什么存在、账怎么算、PyTorch 生态里它长什么样、
以及什么时候它才是你该优化的那 80%。

---

## 二、核心思想：三笔账，一个判据

### 2.1 第一笔账：手里有多少东西

先不算任何东西，只是把数量摆出来。

```python
# 一个 CTA 的输出 tile，以及它离开 Tensor Core 时的样子
TM, TN = 128, 256                 # 典型 Hopper CTA tile
WARPGROUPS = 2                    # 128 行由 2 个 warpgroup 各管 64 行
THREADS = 128 * WARPGROUPS        # 一个 warpgroup = 128 线程

n_acc = TM * TN
print(f"CTA tile              {TM} x {TN}")
print(f"累加器                {n_acc} 个 fp32 = {n_acc*4/1024:.0f} KiB")
print(f"线程数                {THREADS}（{WARPGROUPS} 个 warpgroup）")
print(f"每线程                {n_acc//THREADS} 个 fp32 = {n_acc//THREADS} 个 32-bit 寄存器")
print(f"占 255 个上限的        {n_acc//THREADS/255*100:.0f}%")

# 这笔数据要搬出去，三种通路各需要几条指令
b32, b16 = n_acc * 4, n_acc * 2
warps = THREADS // 32
direct    = (n_acc // THREADS // 2) * warps   # 直接写：每线程一次只存 2 个相邻 fp32 = 8 B
staged32  = b32 // (32 * 16)                  # 过 SMEM，fp32，每线程 v4 = 16 B
staged16  = b16 // (32 * 16)                  # 过 SMEM，转 fp16 再 v4
stmatrix  = b16 // (4 * 8 * 8 * 2)            # stmatrix.m8n8.x4 = 512 B/条
tma       = b16 // (32 * 256 * 2)             # TMA store：每个 32x256 子块一条

print()
print(f"① 直接写 8 B/线程      {direct} 条 warp 指令")
print(f"② 过 SMEM，fp32 v4     {staged32} 条  （{direct/staged32:.1f}x 更少）")
print(f"③ 过 SMEM，转 fp16 v4  {staged16} 条  （{direct/staged16:.1f}x 更少）")
print(f"④ stmatrix.m8n8.x4     {stmatrix} 条")
print(f"⑤ TMA store            {tma} 条")
```

```text
CTA tile              128 x 256
累加器                32768 个 fp32 = 128 KiB
线程数                256（2 个 warpgroup）
每线程                128 个 fp32 = 128 个 32-bit 寄存器
占 255 个上限的        50%

① 直接写 8 B/线程      512 条 warp 指令
② 过 SMEM，fp32 v4     256 条  （2.0x 更少）
③ 过 SMEM，转 fp16 v4  128 条  （4.0x 更少）
④ stmatrix.m8n8.x4     128 条
⑤ TMA store            4 条
```

128 个寄存器、50% 的寄存器预算 —— 这就是为什么 epilogue 的第一要务不是「算得漂亮」，
而是**尽快把寄存器还回去**。

注意 ⑤ 那 4 条是按 32×256 的子块算的。子块越小、条数越多，但 SMEM 占用越低：
32×256 fp16 是 16 KiB，而 CUTLASS 的默认子块是 64×32（只有 4 KiB，代价是 16 个子块、
16 条 TMA store）。**子块尺寸是 SMEM 预算和指令数之间的一根滑杆** —— 第 4.4 节看
CUTLASS 怎么选它，第八节的练习会算它为什么必须是那个值。

### 2.2 第二笔账：直接写到底亏在哪

很多人以为「不过 SMEM 就得浪费字节」。**这个说法是错的**，而且错得很关键。

MMA 累加器的线程映射是 8×8 子块：32 个线程里 `t = 4r + c` 的线程持有第 `r` 行的
第 `2c` 和第 `2c+1` 列。所以一条 `st.global`（每线程 8 B）覆盖的是 **8 行 × 32 B**。
32 B 刚好是一个完整的 sector —— **字节一个都没浪费**。

亏的是另外两样东西：

| | 直接写 | 过 SMEM 后 v4 写 |
|---|---|---|
| 一条指令的字节数 | 256 B（8 行 × 32 B） | 512 B（连续） |
| 触及的 128 B 行 | 8 个，每个只填 25% | 4 个，每个填满 |
| 指令总数（fp32 tile） | 512 条 | 256 条 |
| 行事务总数 | 4096 | 1024 |

**行事务 4 倍、指令数 2 倍，字节数完全相同。** 这个区别在 043 期讲 sector 模型时
埋过伏笔：读侧的最小记账单位是 128 B 的 cacheline，写侧是 32 B 的 sector。
所以「写」天生比「读」宽容 —— 但也仅宽容到 sector 为止，行事务数是躲不掉的。

### 2.3 第三笔账：SMEM 装不下

第三个问题最实在：你想把整个 tile 暂存到 SMEM 再统一走 TMA，装得下吗？

H100 单个 block 的 SMEM 上限 227 KiB。A tile 128×64 fp16 = 16 KiB、B tile 64×256
fp16 = 32 KiB，每 stage 48 KiB。为了喂饱 044 期那条异步流水线，至少 4 级：

```
4 级流水        192 KiB
剩下            35 KiB
整 tile fp16 暂存 64 KiB   → 装不下
```

结论很硬：**epilogue 必须切成子块（subtile），一块一块地走**。这不是风格选择，
是 SMEM 预算逼出来的。下面的源码证据会看到 CUTLASS 正是这么做的。

### 2.4 判据：epilogue 什么时候才是瓶颈

这是本期最有用的一行公式。

epilogue 要写 $M \times N$ 个元素，每个 $b$ 字节，带宽 $BW$：

$$t_E = \frac{MN \cdot b}{BW}$$

GEMM 要做 $2MNK$ 次浮点运算，峰值算力 $P$：

$$t_C = \frac{2MNK}{P}$$

令两者相等，解出临界 $K$：

$$K^{*} = \frac{P}{BW} \cdot \frac{b}{2} = \text{屋脊点} \times \frac{b}{2}$$

**epilogue 的临界 K，就是 043 期那个屋脊点乘上「输出元素字节数除以 2」。** 两个口径
在同一张 roofline 图上自动闭合，这不是巧合 —— epilogue 的字节量正好等于「把输出完整
写一遍」，而屋脊点的定义就是「算力相对带宽的拐点」。

H100 FP16：屋脊点 295.4 FLOP/B，输出 fp32 时 $K^* = 590.7$，输出 fp16 时
$K^* = 295.4$。

**K 大于这个数，写一次输出的时间比算完整个 GEMM 还短 —— epilogue 可以忽略。
K 小于这个数，epilogue 就是你该先看的地方。** 而推理的 decode 阶段，K 常常只有
几千甚至几百，M 只有 1。

---

## 三、在 PyTorch 中怎么用

### 3.1 先看 Inductor 替你生成了什么

PyTorch 不会给你一个叫 `epilogue` 的 API。你能看到它的唯一地方是**编译产物**。
拿一个最朴素的式子试：`gelu(a @ b + bias)`。

```python
import time
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
torch.manual_seed(0)
M = N = 2048

def bench(fn, warm=3, reps=9):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)

print(f"{'K':>5} {'mm/ms':>8} {'GFLOPS':>8} {'eager/ms':>9} {'fused/ms':>9} "
      f"{'epilogue 占比':>13} {'max|Δ|':>10}")
for K in [64, 256, 1024, 4096]:
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    bias = torch.randn(N)

    def f():
        return F.gelu(a @ b + bias, approximate="none")

    t_mm = bench(lambda: a @ b)
    t_eager = bench(f)
    cf = torch.compile(f, dynamic=False)
    out = cf()                       # 触发编译
    t_fused = bench(cf)
    dmax = (out - f()).abs().max().item()

    print(f"{K:>5} {t_mm*1e3:>8.3f} {2*M*N*K/t_mm/1e9:>8.1f} {t_eager*1e3:>9.3f} "
          f"{t_fused*1e3:>9.3f} {(t_eager-t_mm)/t_eager*100:>12.1f}% {dmax:>10.1e}")
```

```text
    K    mm/ms   GFLOPS  eager/ms  fused/ms   epilogue 占比     max|Δ|
   64    0.497   1080.8     2.568     2.327         80.7%    1.5e-05
  256    1.240   1732.5     3.094     2.903         59.9%    3.1e-05
 1024    4.284   2005.0     6.138     6.159         30.2%    9.2e-05
 4096   17.666   1944.9    19.864    20.424         11.1%    3.0e-04
```

表里最后那列 `epilogue 占比` 的口径是「eager 时间减去 GEMM 本体时间，再除以 eager」，
它包含**多跑的两趟 HBM 加上 GELU 本身的运算**，所以比 2.4 节判据里那个「只写一次
输出」的地板大得多。两个口径不要混用：判据问的是「写回值不值」，这一列问的是
「不融合总共要付多少」。

三个可以直接读出来的事实：

**第一，K=64 时 epilogue 占了八成时间。** 这不是「顺手加个 bias」的量级。输出
2048×2048 fp32 是 16 MiB，按纯写带宽算至少 0.19 ms，而 GEMM 本体只要 0.5 ms。

**第二，融合只快了 10%（2.568 → 2.327 ms），不是 3 倍。** 想知道为什么，得看生成的代码。用
`TORCH_LOGS=output_code` 把产物打出来，核心三行是：

```text
buf0 = empty_strided_cpu((2048, 2048), (2048, 1), torch.float32)
extern_kernels.addmm(arg2_1, arg0_1, arg1_1, alpha=1, beta=1, out=buf0)
buf1 = buf0; del buf0  # reuse
cpp_fused_gelu_0(buf1)
```

注意 `addmm(..., alpha=1, beta=1, out=buf0)`：**bias 已经被折进 GEMM 的 `beta*C` 项了**，
这是 epilogue 融合最原始的一种形态 —— 把「加一个向量」变成「累加器的初值」。
但 `gelu` 没地方去，于是单独生成一个 `cpp_fused_gelu_0`，**原地读一遍、写一遍**。
eager 版在这 16 MiB 的输出上总共搬了 80 MiB（mm 写 16 + add 读16写16 + gelu 读16写16），
融合版是 48 MiB —— 字节账 **1.67 倍**，实测却只有 **1.10 倍**。

这个落差本身就是本期的重点：**融合省的是字节，但省下来的时间不一定能兑现**。
这里卡住的是 GELU 本身 —— 4.19M 个元素按 erf 的吞吐（5.2 Gelem/s，见 4.1 节）
要 0.8 ms，这笔计算量无论怎么融合都省不掉。字节省了 40%，时间只省了 10%，
因为瓶颈已经从「搬」移到了「算」。

记住这个区分：**带宽受限的 epilogue，融合立刻兑现；计算受限的 epilogue，
融合只兑现搬的那部分。**

**第三，数值不一致，而且 K 越大越不一致。** 上表最后一列 `max|Δ|` 随 K 单调增长
到 `3.0e-4`。原因是 Inductor 的 CPU 路径换了 GEMM 实现，fp32 累加顺序变了。
用 `torch.allclose` 的默认容差（`rtol=1e-5, atol=1e-8`）验收编译结果，K 大的时候
**会直接判失败**。这是实务上一定要知道的：编译前后数值不等价是常态，等价的只有
数学表达式。

### 3.2 想控制 epilogue，就得自己写 GEMM

PyTorch 生态里最直接的入口是 **Triton**（再往下是 CUTLASS，4.4 节）。
写法和 008 期那个 Triton 入门一致，区别只在最后 —— 而 epilogue 就是那个「最后」：

```py
@triton.jit
def matmul_kernel(A, B, Bias, C, M, N, K, sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  EPI_TILE_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)     # 累加器住在寄存器里

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A + offs_m[:, None] * sa_m + (k * BLOCK_K + tl.arange(0, BLOCK_K))[None, :] * sa_k)
        b = tl.load(B + (k * BLOCK_K + tl.arange(0, BLOCK_K))[:, None] * sb_k + offs_n[None, :] * sb_n)
        acc += tl.dot(a, b)                                  # 这一步结束，acc 就是「累加器」

    # ---- 从这里开始是 epilogue：acc 还在寄存器里，下面的写法决定它怎么落地 ----
    bias = tl.load(Bias + offs_n)                            # 只读一段，broadcast 到所有行
    acc = acc + bias[None, :]                                # 折进寄存器，不额外读写 HBM
    acc = acc * 0.5 * (1.0 + tl.erf(acc * 0.7071067811865476))  # GELU 也在寄存器里做完

    # 关键：按子块存，而不是一次性 tl.store 整块
    for i in tl.static_range(0, BLOCK_M, EPI_TILE_M):
        rows = offs_m[i:i + EPI_TILE_M]                                  # 子块的行坐标
        sub = acc[i:i + EPI_TILE_M, :].to(tl.float16)                    # 转 fp16，字节砍半
        tl.store(C + rows[:, None] * sc_m + offs_n[None, :] * sc_n, sub)
```

这段的要点全在注释里，但背后每一条都有账：

- `acc + bias` 和 GELU **都在寄存器里做**，一次 HBM 都不碰。这是 epilogue 融合的
  全部价值所在 —— 不是「少一个 kernel」，是「少一趟 HBM」。
- `.to(tl.float16)` 把落盘字节砍半。043 期量过：写带宽只看字节数，不看元素个数。
  同样的 GB/s，元素数翻倍时间不变 —— 所以输出精度直接决定 epilogue 的时间。
- 按 `EPI_TILE_M` 切子块存，是 2.3 节那笔 SMEM 预算的推论。子块化之后，编译器才有机会
  把这块落到 SMEM 再走 descriptor store（Triton 里是 `tl.make_tensor_descriptor` /
  `_experimental_descriptor_store`）。**这一层的实际生成结果本机无 GPU，未实测**；
  确定的是 `separate_epilogue_store` / `epilogue_subtile` 这两个旋钮在 Inductor 里
  确实作为 autotune 维度存在（4.4 节给行号）。

### 3.3 融合到底值多少：一个可以自己跑的对照

把「中间结果落不落 HBM」单独拎出来量。两条路径做同一件事，只是一个分两趟：

```python
import time
import torch

torch.set_num_threads(4)
N = 1 << 25                        # 33.6M fp32 = 128 MiB
a, b = torch.randn(N), torch.randn(N)
o = torch.empty(N)

def bench(fn, warm=3, reps=9):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)

def two_pass():                    # add 落盘，再读回来 clamp：中间结果过了一趟 HBM
    torch.add(a, b, out=o)
    torch.clamp(o, min=0.0, out=o)

def one_pass():
    return torch.clamp(a + b, min=0.0)

t2 = bench(two_pass)
cf = torch.compile(one_pass, dynamic=False)
y = cf()
t1 = bench(cf)

print(f"两趟（add 落盘 + clamp 读回） {t2*1e3:7.3f} ms")
print(f"一趟（编译后融合）           {t1*1e3:7.3f} ms")
print(f"收益 {t2/t1:.2f}x，结果一致 {torch.allclose(y, torch.clamp(a+b, min=0.0))}")
print(f"字节账：两趟 {3*128+2*128} MiB，一趟 {3*128} MiB，"
      f"比值 {(3*128+2*128)/(3*128):.2f}x")
```

```text
两趟（add 落盘 + clamp 读回）   5.385 ms
一趟（编译后融合）             3.444 ms
收益 1.56x，结果一致 True
字节账：两趟 640 MiB，一趟 384 MiB，比值 1.67x
```

这次实测和字节账吻合（1.56 vs 1.67），因为 `clamp` 是**纯带宽受限**的 —— 它只是
一个 max，算术部分免费。换个 `gelu` 进来就会像 3.1 节那样只兑现 10%。

所以定标要分两种情形：

| epilogue 里的运算 | 融合能兑现多少 | 为什么 |
|---|---|---|
| `+bias`、`relu`、`clamp`、cast | 接近字节账（1.5~1.7x） | 算术免费，纯搬 |
| `gelu`、`sigmoid`、`tanh` | 远低于字节账（~1.1x） | 卡在超越函数吞吐 |

**融合不是银弹，它是「把一笔本来不该付的搬运费退回来」—— 退多少，取决于你原本
是在等带宽还是在等 ALU。**

---

## 四、围绕该领域展开

先把这块拼图放回它周围的版图上：

- **047 期（wgmma）**：那期讲「数据进 SMEM 之后怎么被吃进 Tensor Core」，
  停在累加器落在寄存器。本期是它的下一句：那些累加器怎么走。
- **044 期（cp.async / TMA）**：全局 → SMEM。本期是反方向：SMEM → 全局，
  同一个 TMA 引擎，只是 `cp.async.bulk.tensor` 的 S2G 方向。
- **046 期（cluster / DSMEM）**：epilogue 也可以走 DSMEM，把结果聚到 cluster 的
  一个 CTA 再统一写 —— 那是「写事务数」的另一种解法。
- **045 期（occupancy）**：128 个累加器寄存器就是占用率的第三把锁；本期讲
  怎么尽早把这把锁解开。
- **043 期（内存层次）**：那期的屋脊点在这里直接变成判据里的临界 K。
- **040 期（算子融合）**：那期讲「为什么融合省字节」，本期讲融合发生在什么位置、
  由谁负责把那笔字节省下来。
- **008 期（Triton）**：写自定义 epilogue 的入口就是它。

### 4.1 epilogue 里放什么，代价可能差 10 倍

这是本期最反直觉的一节。同样是把一个逐元素运算塞进 epilogue，
选哪个函数，代价差一个数量级。

```python
import time
import torch

torch.set_num_threads(4)
N = 1 << 25                        # 33.6M fp32 = 128 MiB，远超 L2
a, b, o = torch.randn(N), torch.randn(N), torch.empty(N)

def bench(fn, warm=3, reps=9):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)

rows = [
    ("fill_    纯写",        lambda: o.fill_(1.0), 4),
    ("copy_    读1写1",      lambda: o.copy_(a), 8),
    ("relu     max",        lambda: torch.clamp(a, min=0.0, out=o), 8),
    ("add      读2写1",      lambda: torch.add(a, b, out=o), 12),
    ("bias+relu 分两趟",     lambda: torch.clamp(torch.add(a, b, out=o), min=0.0, out=o), 20),
    ("exp",                 lambda: torch.exp(a, out=o), 8),
    ("erf",                 lambda: torch.erf(a, out=o), 8),
    ("tanh",                lambda: torch.tanh(a, out=o), 8),
    ("gelu 近似=none(erf)",  lambda: torch.nn.functional.gelu(a, approximate="none", out=o), 8),
    ("gelu 近似=tanh",       lambda: torch.nn.functional.gelu(a, approximate="tanh", out=o), 8),
]
times = [(name, bench(fn), bpe) for name, fn, bpe in rows]
t_relu = dict((n, t) for n, t, _ in times)["relu     max"]      # 写带宽地板

print(f"{'算子':<22} {'耗时':>10} {'Gelem/s':>9} {'GB/s':>8} {'相对 relu':>10}")
print("-" * 64)
for name, t, bpe in times:
    print(f"{name:<22} {t*1e3:>9.3f}ms {N/t/1e9:>9.2f} {N*bpe/t/1e9:>8.1f} "
          f"{t/t_relu:>9.2f}x")
```

```text
算子                             耗时   Gelem/s     GB/s    相对 relu
----------------------------------------------------------------
fill_    纯写                1.538ms     21.82     87.3      0.74x
copy_    读1写1              2.069ms     16.22    129.8      1.00x
relu     max               2.072ms     16.19    129.6      1.00x
add      读2写1              3.379ms      9.93    119.2      1.63x
bias+relu 分两趟              5.399ms      6.22    124.3      2.61x
exp                        4.798ms      6.99     55.9      2.32x
erf                        5.039ms      6.66     53.3      2.43x
tanh                      18.095ms      1.85     14.8      8.73x
gelu 近似=none(erf)          6.456ms      5.20     41.6      3.12x
gelu 近似=tanh              20.995ms      1.60     12.8     10.13x
```

三行值得盯住：

- **`relu` 和 `copy_` 一样快。** 它已经跑在写带宽地板上 —— 也就是说，
  在 epilogue 里加一个 max，**完全免费**。
- **`bias+relu` 分两趟是 2.6 倍。** 仅仅因为中间结果落了一次 HBM。
- **`gelu 近似=tanh` 比 `近似=none`(erf) 慢 3 倍多。** 这条和 GPU 上的直觉完全相反，
  下面单独拆。

### 4.2 tanh 为什么慢：去源码里找答案

在 NVIDIA GPU 上，`tanh` 有硬件指令：`tanh.approx.f32`，sm_75 起支持，一条 MUFU
就出结果（**PTX ISA 文档级，本机无 CUDA 未实测**）。由此**推断**：GPU 上 GELU 的
tanh 近似版通常比 erf 版更便宜 —— 后者没有硬件指令，只能走多项式。
这一条是推论，不是我在 GPU 上量到的事实。

但在这台 Apple M5 上，实测完全反过来。原因不在数学，在实现：

```python
import time
import torch

torch.set_num_threads(4)
N = 1 << 23                        # 8.4M fp32 = 32 MiB
x = torch.randn(N) * 2
o = torch.empty_like(x)
buf = torch.empty_like(x)

def bench(fn, warm=3, reps=9):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)

def tanh_via_exp():
    # tanh(x) = 2/(1+exp(-2x)) - 1，四个独立算子，各自过一次 HBM
    torch.mul(x, -2.0, out=buf)
    torch.exp(buf, out=buf)
    buf.add_(1.0)
    torch.div(2.0, buf, out=buf)
    buf.sub_(1.0)
    return buf

t_torch = bench(lambda: torch.tanh(x, out=o))
t_mine = bench(tanh_via_exp)
err = (tanh_via_exp() - torch.tanh(x)).abs().max().item()

print(f"torch.tanh              {t_torch*1e3:7.3f} ms")
print(f"手写 exp 版（4 步）      {t_mine*1e3:7.3f} ms")
print(f"手写版快 {t_torch/t_mine:.2f}x，且它的 HBM 流量是 torch.tanh 的 4 倍")
print(f"手写版最大绝对误差       {err:.3e}")
```

```text
torch.tanh                4.210 ms
手写 exp 版（4 步）        2.987 ms
手写版快 1.41x，且它的 HBM 流量是 torch.tanh 的 4 倍
手写版最大绝对误差       1.788e-07
```

手写版多跑了 4 个 kernel、HBM 流量是 4 倍，**结果还快了 1.4 倍**。这不是数学问题，
是 `torch.tanh` 在这台机器上走了高精度实现。

源码里能看到 torch 确实对这两个函数做了不同取舍。`vec256_float.h` 的 **x86 分支**里，
`erf()` 是**手写的** Abramowitz-Stegun 7.1.26 五阶有理逼近（一次除法 + 4 个 FMA
+ 1 次 exp），而同一文件里的 `tanh()` 直接交给 Sleef 的 u10 版本：

```text
// aten/src/ATen/cpu/vec/vec256/vec256_float.h（上游 main 分支）
erf(x) = sign(x) * (1 - r * t * exp(-x*x))     // t = 1/(0.3275911*|x| + 1)，5 个常数
...
Vectorized<float> tanh() const {
  return Vectorized<float>(Sleef_tanhf8_u10(values));   // ≤1 ulp 的全精度版本
}
```

一个只有约 1.5e-7 绝对误差的近似式（换速度），一个 ≤1 ulp（换精度）。同一份源码里
两个相邻的函数做了相反的选择。

需要说清证据边界：**这段代码是 x86 分支，本机是 ARM，走的是 `AT_VEC_CUSTOM_MATH`
的另一份实现，我没有验证那一份怎么写。** 但 4.1 节那个 3.25 倍的实测差距独立成立 ——
它证明的是「同一台机器上 tanh 与 erf 的实现代价差一个数量级」，而不是「torch 全平台
都这么写」。

**这是本期最值得带走的一条：epilogue 里的代价由「谁实现了它」决定，不由公式长短决定。**
跨硬件迁移时，同一个算子在不同 backend 上的相对排名可能完全反转 —— 想当然会踩坑。

### 4.3 写回来的地板有多低

既然 epilogue 本质是写，那「写」本身有多快？这决定了整件事的上限。

```python
import time
import torch

torch.set_num_threads(4)
MB = 2 ** 20
N = 1 << 25                                  # 33.6M fp32 = 128 MiB
x, y, o = torch.randn(N), torch.randn(N), torch.empty(N)

def bench(fn, warm=3, reps=15):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)

def show(name, nbytes, t):
    print(f"{name:<26} {t*1e3:>8.3f} ms  {nbytes/t/1e9:>7.1f} GB/s")

print("== 128 MiB 缓冲 ==")
show("fill_ 纯写", 128 * MB, bench(lambda: o.fill_(1.0)))
show("copy_ 读1写1", 256 * MB, bench(lambda: o.copy_(x)))
show("add   读2写1", 384 * MB, bench(lambda: torch.add(x, y, out=o)))

# 写侧的最小记账单位：32 B sector（= 8 个 fp32）
print("\n== 纯写粒度扫描（64 MiB，每第 s 个元素写）==")
big = torch.empty(1 << 24)
for s in [1, 2, 4, 8, 16, 32]:
    t = bench(lambda s=s: big[::s].fill_(1.0))
    useful = (1 << 24) // s * 4
    sectors = ((1 << 24) // 8) if s <= 8 else ((1 << 24) // s)
    print(f"  stride={s:<3} {t*1e3:7.3f} ms  有效 {useful/t/1e9:6.1f} GB/s"
          f"  触及 sector {(sectors*32)/t/1e9:6.1f} GB/s")
```

```text
== 128 MiB 缓冲 ==
fill_ 纯写                      1.541 ms     87.1 GB/s
copy_ 读1写1                    2.071 ms    129.6 GB/s
add   读2写1                    3.357 ms    120.0 GB/s

== 纯写粒度扫描（64 MiB，每第 s 个元素写）==
  stride=1     0.734 ms  有效   91.5 GB/s  触及 sector   91.5 GB/s
  stride=2     0.893 ms  有效   37.6 GB/s  触及 sector   75.2 GB/s
  stride=4     0.953 ms  有效   17.6 GB/s  触及 sector   70.4 GB/s
  stride=8     0.980 ms  有效    8.6 GB/s  触及 sector   68.5 GB/s
  stride=16    0.977 ms  有效    4.3 GB/s  触及 sector   34.3 GB/s
  stride=32    0.984 ms  有效    2.1 GB/s  触及 sector   17.1 GB/s
```

两个结论：

- **纯写比拷贝的单方向带宽高 1.34 倍**（87.1 vs 129.6/2），因为拷贝要多读一份。
  写回这一趟，你只付一次写。
- **粒度扫描里，`stride=8` 时有用字节只剩 1/8，耗时却只从 0.734 ms 涨到 0.980 ms。**
  因为 8 个 fp32 刚好填满一个 32 B sector —— 触及的 sector 总量没变，
  带宽稳定在 68~92 GB/s。**写侧按 32 B 记账，读侧按 128 B 记账**（043 期）。
  这个不对称是 epilogue 优化的物理基础：只要你的 store 覆盖完整 sector，
  就不用担心「写不合并」。

### 4.4 Hopper 的 epilogue 长什么样：CUTLASS 源码

上面三小节讲的都是「为什么」。真去看工业实现，会发现 CUTLASS 把这套账做成了模板参数。
以下全部引**上游 main 分支**（本机没有 cutlass 头文件，走 `raw.githubusercontent.com` 抓取）：

`include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`（963 行）：

```text
:68    class EpilogueTile_,                       // (EPI_TILE_M, EPI_TILE_N)
:129   static_assert(size<0>(CtaTileMNK{}) % size<0>(shape(EpilogueTile{})) == 0, ...)
:130   static_assert(size<1>(CtaTileMNK{}) % size<1>(shape(EpilogueTile{})) == 0, ...)
:163   using SmemLayoutC = ... make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}), Int<StagesC>{})
:168   using SmemLayoutD = ... make_shape(..., Int<ReuseSmemC ? StagesC : StagesD>{})
:171   constexpr static bool support_smem_reuse = ... StagesD <= StagesC && ...
:261   TMA_C tma_load_c;    TMA_D tma_store_d;
:455   Tensor gC_epi = flat_divide(gC, EpilogueTile{});   // (EPI_M, EPI_N, EPI_M, EPI_N)
:714   bool issue_tma_store = (thread_idx / NumThreadsPerWarp) == 0;   // 只有 warp 0 发 TMA
:726   // We can delay issue of TMA store by one iteration to achieve better interleaving
:736   cutlass::arch::fence_view_async_shared();      // 保证 smem 写对 TMA 可见
```

逐条对上前面三笔账：

- **`:129-130` 两条 `static_assert`** 就是 2.3 节那笔 SMEM 预算的编译期表达：
  `EpilogueTile` 必须整除 CTA tile，不允许出现半块。你在运行时发现的「子块装不下」
  在 CUTLASS 里是编译错误。
- **`:163-171`** 说明 SMEM 是按 `EpilogueTile × Stages` 分配的，而且 `StagesD`
  只在「C 和 D 同宽同类型」时才允许复用 C 的缓冲区（`ReuseSmemC`）。
  **这是一个典型的「为了省 4 KiB 加了 3 行约束」的工程决策。**
- **`:714`** 一句话道破 TMA store 的本质：**整个 warpgroup 只有一个 warp 发指令**。
  剩下的 3 个 warp 在等 —— 这既是优点（一条指令搬 16 KiB，几乎不占 issue slot），
  也是缺点（要等它们等完）。
- **`:736`** 那个 `fence_view_async_shared()` 是最容易漏的一行：SMEM 写完到 TMA
  能读，中间必须有这道栅栏。044 期讲 mbarrier 时是反方向（TMA 写完 → 消费者读），
  这里是生产者自己发栅栏。

再看默认参数是怎么推出来的，`include/cutlass/epilogue/collective/builders/sm90_builder.inl`：

```text
:69    constexpr int EpiTiles = size(shape_div(take<0,2>(TileShapeMNK{}), EpilogueTileMN{}));
:75    constexpr int StagesD = cute::min(EpiTiles, 2);          // 最多 2 个 TMA store 在飞
:76    constexpr int StagesC = ReuseSmem ? max(min(EpiTiles,4), StagesD+1) : min(EpiTiles,4);
...
:121   // EpilogueTileAuto，warp-specialized schedule：
:122   constexpr int N_perf = (8-bit && CTA_N%64==0) ? 64 : 32;
:123   auto tile_m = cute::min(_64{}, size<0>(TileShape_MNK{}));   // M 方向封顶 64
:124   auto tile_n = cute::gcd(cute::min(Int<N_perf>{}, size<1>(TileShape_MNK{})), size<1>(TileShape_MNK{}));
...
:157   static_assert(M == 64 && sm90_is_warp_specialized_v<Schedule> ||
                       M == 128 && sm90_is_cooperative_v<Schedule>, "Unsupported tile shape");
:159   static_assert(N % 16 == 0, "Unsupported tile shape");
```

对一个 128×256 的 fp16 GEMM：`EpiTiles = (128/64) × (256/32) = 16`，
`StagesD = min(16,2) = 2`，子块 **64×32 fp16 = 4 KiB**，两个在飞 = 8 KiB。

对比 2.3 节算出来的「4 级流水只剩 35 KiB」——**CUTLASS 把 epilogue 的 SMEM 压到了 8 KiB**，
只为了给主流水线留够空间。这就是「子块尺寸是滑杆」的实际取值。

而 Inductor 的 Triton 模板把同一根滑杆做成了 autotune 维度。
`torch/_inductor/heuristics/template/triton.py`（本机 torch 2.14.0）：

```text
:162   epilogue_subtile: int = dataclasses.field(kw_only=True, default=1)   # 必须是 2 的幂
:168   separate_epilogue_store: bool = dataclasses.field(kw_only=True, default=False)
:417   BlackwellGPUGemmConfig(128, 256, 64, 4, 8, epilogue_subtile=2, warp_specialize=True, ...)
:2732  subtile = template_kwargs.get("EPILOGUE_SUBTILE", 1)
:2733  # each epilogue subtile is BLOCK_N // EPILOGUE_SUBTILE wide
:2734  if block_n // subtile < 32: return False            # 子块宽度下限 32 列
:2793  for EPILOGUE_SUBTILE in [1, 2, 4]:                  # 和 num_stages/num_warps 一起 sweep
```

两条独立实现路线在同一个数字上撞了：**CUTLASS 的 `EPI_TILE_N` 默认 32，
Inductor 剪掉 `block_n // subtile < 32` 的配置。** 这不是巧合 —— 32 列 fp16 是 64 B，
两个 sector；再窄下去，TMA 的 box 和 SMEM 的 swizzle 都开始不划算。

另外注意 `BlackwellGPUGemmConfig` 里 128×256 的配置用 `epilogue_subtile=2`，
也就是 epilogue 子块是 128×128 —— **比 Hopper 的 64×32 大 8 倍**。
原因是 Blackwell 的 SMEM 更宽裕、TMA 更成熟，可以一次性搬更多；
这也说明这根滑杆的最优值随架构变，没有普适答案。

---

## 五、什么时候该用 / 不该用

把 2.4 节那个判据算成表：

```python
# 判据：epilogue 时间 = GEMM 时间的那个 K
def k_star(peak_flops, bandwidth, out_bytes):
    return peak_flops * out_bytes / (2 * bandwidth)

print(f"{'平台':<26} {'屋脊点':>9} {'fp32 输出':>11} {'fp16 输出':>11}")
print("-" * 62)
for name, P, BW in [
    ("本机 M5 fp32 / 4 线程", 1940e9, 87.6e9),
    ("H100 SXM5 FP16 dense", 989.5e12, 3.35e12),
    ("A100 SXM FP16 dense", 312e12, 2.039e12),
]:
    print(f"{name:<26} {P/BW:>9.1f} {k_star(P,BW,4):>11.1f} {k_star(P,BW,2):>11.1f}")

print("\n交叉验证 043 期的屋脊点：H100 FP16 = 295.4 FLOP/B")
print(f"  295.4 x 4 B / 2 = {295.4*4/2:.1f}  ←  与上表 fp32 输出列一致")

# epilogue 的 issue 成本 vs MMA 的周期，用 047 期推出的一条 wgmma = 138.5 cycle
print("\n== 每 warpgroup 的时间轴（K 决定 MMA 占多少）==")
cycles_per_wgmma = 138.5
store_instr = 128 // 2          # 每线程 128 个累加器，一次存 2 个 = 64 条
for K in [64, 512, 4096]:
    n_mma = K // 16
    print(f"  K={K:<5} wgmma {n_mma:>3} 条 = {n_mma*cycles_per_wgmma:>7.0f} cycle；"
          f"epilogue 每线程 {store_instr} 条 store")
```

```text
平台                               屋脊点     fp32 输出     fp16 输出
--------------------------------------------------------------
本机 M5 fp32 / 4 线程               22.1        44.3        22.1
H100 SXM5 FP16 dense           295.4       590.7       295.4
A100 SXM FP16 dense            153.0       306.0       153.0

交叉验证 043 期的屋脊点：H100 FP16 = 295.4 FLOP/B
  295.4 x 4 B / 2 = 590.8  ←  与上表 fp32 输出列一致

== 每 warpgroup 的时间轴（K 决定 MMA 占多少）==
  K=64    wgmma   4 条 =     554 cycle；epilogue 每线程 64 条 store
  K=512   wgmma  32 条 =    4432 cycle；epilogue 每线程 64 条 store
  K=4096  wgmma 256 条 =   35456 cycle；epilogue 每线程 64 条 store
```

**该管 epilogue 的场景：**

- **GEMM 的 K 小于表里的临界值**，尤其是 decode 阶段的 `M=1`、`K` 只有几千的
  GEMV 型矩阵乘。这时 epilogue 是 30%~80% 的时间，任何融合都是净赚。
- **输出精度可以从 fp32 降到 fp16/bf16。** 判据里的 `b` 从 4 变 2，时间直接砍半，
  而且不需要改任何算法。
- **epilogue 里带着非线性、residual add、或者 cast。** 只要还在寄存器里做，
  就省掉一整趟 HBM。

**不该管的场景：**

- **K 远大于临界值。** K=4096 时光写一次输出只占 GEMM 的 1.1%，
  把工程预算花在这里纯属浪费。
- **GEMM 自己还没跑满峰值算力。** 本机 K=64 那档，写回地板 16 MiB ÷ 87.1 GB/s
  = 0.193 ms，除以 GEMM 的 0.497 ms 只有 39%；而解析式按峰值算力给的预测是 69%。
  差距来自那一档 GEMM 并没跑满（1081 vs 1945 GFLOPS），输入带宽在限制它，
  写回被部分掩盖了。
  **判据只在 GEMM 已经算力饱和时才成立。**
- **你用的是别人的库。** cuBLAS 的 GEMM 没有 epilogue 钩子，cuBLASLt 只支持有限几种。
  想融合，只能自己写 kernel（Triton / CUTLASS）或者让 Inductor 生成。

---

## 六、常见坑

### 坑 1：以为 epilogue 只是「顺手加个 bias」

实测 K=64、M=N=2048 时，`gelu(a@b+bias)` 里 epilogue 占 80.8%。低算术强度的
GEMM 里，收尾阶段完全可能比计算本身还贵。**先量，再判断该不该优化它。**

### 坑 2：在 epilogue 里无脑选 `approximate="tanh"`

GPU 上它是一条 MUFU，CPU 上它比 erf 版慢 3 倍。同理，`exp`、`sigmoid`、`erf`
在三家 backend 上的相对代价各不相同。**换硬件前先用一行 bench 量一遍，
不要靠另一台机器上的直觉。**

### 坑 3：以为 `torch.compile` 一定把 epilogue 融进 GEMM

本机（torch 2.14.0，CPU）实测：`mode="max-autotune"` 和默认模式生成的代码**完全一样**，
都是 `addmm(beta=1)` 加一个独立的原地 `cpp_fused_gelu_0`，只快了 10%。
GEMM 的 epilogue 能不能融合，取决于后端有没有提供「带 epilogue 的 GEMM」这个入口 ——
CPU 上的 `addmm` 只认 bias。GPU 上 Triton 模板有 `epilogue_subtile` 和
`separate_epilogue_store` 两个专门的旋钮，那是另一条路。

### 坑 4：用 `torch.allclose` 默认容差验收编译结果

fp32 的累加顺序一变，K=4096 时 `max|Δ|` 到 `3.0e-4`。默认 `atol=1e-8` 会判失败，
然后你去怀疑编译器有 bug —— 其实只是浮点加法不满足结合律。验收编译结果要用
**相对容差**（`rtol=1e-4` 起）或者比对统计量，不要逐元素比。

---

## 七、一句话总结

**epilogue 的复杂度全部来自一个不对称：Tensor Core 用 1 个 issue slot 吐出 128 个
寄存器，而 HBM 只接受 128 字节连续 —— 中间那段对齐工作，就是 epilogue。**

---

## 八、今日练习

<details>
<summary>今日练习</summary>

**题目**

H100 SXM5 上跑一个 `M = N = 4096`、`K = 128`、fp16 输入输出、epilogue 里带
`+bias` 和 GELU 的 GEMM。已知：

- FP16 dense 算力 989.5 TFLOPS，HBM 3.35 TB/s，SMEM 227 KiB / block
- 你打算用 128×256 的 CTA tile、4 级 A/B 流水

回答三个问题：

1. epilogue 是否是这个 kernel 的瓶颈？给出定量依据。
2. 如果要从 epilogue 下手，按收益从大到小排出三个动作。
3. 为什么 CUTLASS 的 `EpilogueTile` 在 warp-specialized schedule 下 M 方向被
   硬编码成 64、N 方向必须能被 16 整除？

**参考答案**

**1. 是不是瓶颈**

用判据：$K^* = \text{屋脊点} \times b / 2$，fp16 输出 $b = 2$，屋脊点
$989.5/3.35 = 295.4$ FLOP/B，所以 $K^* = 295.4$。

$K = 128 < 295.4$，**epilogue 主导**。

定量：GEMM 本体 $\frac{2 \times 4096^2 \times 128}{989.5 \times 10^{12}} = 4.34\ \mu s$；
epilogue 写 $4096^2 \times 2 = 32$ MiB，按 3.35 TB/s 需 $9.55\ \mu s$。
**写回比算还贵 2.2 倍**。这个 kernel 从头到尾就是在搬数据。

**2. 三个动作，按收益排序**

- **① 输出精度确认在 fp16**（如果上游允许 bf16 也一样）。判据里的 `b` 直接决定这一项，
  从 fp32 改 fp16 就是 2 倍。
- **② 保证 epilogue 完全在寄存器里完成，且用 TMA store 一次性甩出去。**
  关键是别让 bias / GELU 变成独立的 kernel —— 那就是 3 倍。用 Triton 模板的
  `epilogue_subtile` 或者手写 CUTLASS 的 `Sm90TmaWarpSpecialized`。
- **③ 把 epilogue 和下一个 tile 的 MMA 重叠。** Hopper 的做法是独立的 epilogue
  warpgroup，这也是 CUTLASS 里 `NumEpilogueWarpGroups` 存在的理由。
  收益取决于 epilogue 占的比例，这里是 2.2/(1+2.2) = 69%，理论上限很高。

注意排序逻辑：①②是「少搬字节」，③是「把搬的时间藏起来」。前者确定，后者依赖调度。

**3. 为什么是 64 和 16**

`EpilogueTile` 的约束在 `sm90_builder.inl` 里是两条 `static_assert`：

- **M 方向 = 64，硬编码。** 因为 warp-specialized schedule 下累加器由**一个
  warpgroup（128 线程）**持有，而一条 `wgmma.m64nNk16` 的 M 就固定是 64 ——
  子块的 M 必须等于单条 MMA 指令的 M，否则寄存器到 SMEM 的布局转换会跨 warpgroup，
  没法用一个 `stmatrix` 覆盖。cooperative schedule 用 128，因为那里两个 warpgroup
  协同持有一个 M=128 的累加器。
- **N 方向能被 16 整除。** 这是 swizzle atom 的粒度：SMEM 的 swizzle 布局
  （047 期那个 `Swizzle<3,4,3>`）最小单元是 8 个 16-bit 元素，而 TMA store 的
  box 要求 N 方向对齐到 16 个元素才能保证一趟连续搬运。

一个旁证：Inductor 的 Triton 模板走的是同一个下限。`triton.py` 里对
`block_n // subtile < 32` 的配置直接剪掉 —— **两个独立实现在「epilogue 子块
N 方向不能低于 32 列」这件事上收敛到同一个数**。这不是巧合，是 sector 和
swizzle 的物理约束。

</details>
