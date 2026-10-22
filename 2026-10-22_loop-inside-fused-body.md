# PyTorch 每日一课 · 第 041 期

## 融合体内部的循环怎么排：向量宽度、并行切分、尾部与归约分块

**日期** 2026-10-22　**难度** ⭐⭐⭐⭐　**前置** 第 040 期（知道「融合体」是什么）、能读 C 的 `for` 循环、听说过 SIMD　**预计阅读** 约 25 分钟

**关联** 第 040 期 · 040 决定「哪些算子进同一个循环体」，本期决定「这一个循环体内部怎么走」；第 008 期（Triton：循环是你自己写的）；第 003 期（Dynamo 抓图 → Inductor 排循环）；第 005 期（Flash Attention：tile 的极致）；第 039 期（再往下一层是派发）

---

### 0. 先把结论摆出来

上一期把 24 个逐元素算子融成 1 个 kernel，加速 13.6×，上限就是算子个数（24×）。**融合只解决了「搬几趟」，没解决「这一趟怎么搬」。** 这一期讲的是后面那一半：

1. 融合体内部只有**一层循环**，但这层循环有四个可调自由度：**宽度 $W$（一次算几个）、并行 $P$（几个线程分）、分块 $C$（怎么切块累加）、尾部 $T$（零头怎么办）**。
2. 四个自由度里，只有**宽度**是「硬件常量决定的」（本例 NEON 128 bit → fp32 一次 4 个），其余三个都是**编译器按规则算出来的数字**，规则可以读出来、也可以预测。
3. **排法决定你站在 roofline 的哪一侧**。实测同一台机器同一个融合体：`x*2+1` 单线程就到 113 GB/s（≈ 内存墙 130 GB/s 的 87%），加线程几乎无用；`x.sum()` 单线程只有 27.7 GB/s（墙的 21%，卡在依赖链上），4 线程 3.06×。
4. **排法还会改变浮点结果**。1M 个 fp32 求和，九种配置给出 3 个不同的和；但同一种排法下，线程数 1/2/4 的结果**逐比特相同**（分块累加的设计目标就是这个）。

---

### 1. 这个领域解决什么问题

假设你已经拿到 040 期那个融合体：

```
m = 24 个 pointwise 算子首尾相连
eager：24 次读 + 24 次写 HBM —— 26.8 ms
融合：1 次读 + 1 次写 HBM   —— 1.96 ms
```

流量从 $8mN$ 字节降到 $8N$ 字节，剩下的时间花在哪？花在**那唯一一层循环的执行效率**上。而这一层循环的效率，由三件事决定：

- **一次处理几个元素**（宽度）。一次 1 个 vs 一次 4 个，指令条数差 4 倍。
- **多少条独立的执行流同时跑**（并行）。一个核 vs 四个核。
- **遇到「必须把整条数据走完才能出一个数」的算子（归约）时，怎么切块、怎么合并**。切法不同，既能决定速度，也能决定最后几位小数。

这三件事在 GPU 上叫 `XBLOCK / num_warps / tile`，在 CPU 上叫 `SIMD 宽度 / omp 线程 / 分块累加`，在 CUTLASS 里叫 `ThreadblockShape / WarpShape / InstructionShape`，在 cuBLAS 里干脆不让你看见。**名字换四套，问题只有一个。**

为什么值得单独讲一期：融合是 PyTorch 2.x 编译器给你的最后一份大礼，而这份礼物的兑现**全部落在这一层循环上**。融合对了但循环排得差，前面的 13.6× 会漏掉一半（040 期实测：12 op 是 9.36×，24 op 掉到 13.65× 而理论上限 24×，掉的那部分就是「融合体自己变成了计算受限」）。

---

### 2. 核心思想：一层循环，四个旋钮

用搬箱子做类比。你有 4096 个箱子要从 A 搬到 B：

| 自由度 | 搬运里的意思 | 代码里的样子 | 谁决定 |
|---|---|---|---|
| 宽度 $W$ | 一趟抱几个 | `x0 += 4`，`Vectorized<float>` | 硬件寄存器位宽（本例 128 bit ÷ 32 bit = 4） |
| 并行 $P$ | 几个人搬 | `#pragma omp parallel num_threads(P)` | `threads × min_chunk_size` 门槛算出来 |
| 分块 $C$ | 边搬边记账怎么记 | 每线程 4 个向量累加器 + 每 4096 次迭代合并一次 | 归约类型 + 数据量 |
| 尾部 $T$ | 最后剩 3 个怎么办 | `loadu(ptr, 3)` 的掩码向量 | `N mod W` |

四个自由度不是并列的，有依赖顺序：

```
先定 W（硬件定死） → 再定 P（看 N 够不够分） → 归约才需要 C → 最后收拾 T
```

再加一条贯穿全篇的判据 —— **算术强度**（每搬 1 字节做多少次运算）：

$$
\text{AI} = \frac{\text{FLOP}}{\text{Byte}} \qquad
t_{\text{下界}} = \max\left(\frac{\text{FLOP}}{\text{峰值算力}},\ \frac{\text{Byte}}{\text{峰值带宽}}\right)
$$

排法（尤其是 $P$ 和 $C$）决定了实际耗时贴哪一条下界。贴带宽那条，加线程/加宽度都白费；贴算力那条，加并行几乎线性。

---

### 3. 在 PyTorch 中怎么用：不写循环，读产物 + 拧旋钮

这一节不是教你手写循环（那是 Triton 和 CUDA C++ 的事），而是教你**看懂 PyTorch 替你排的循环，并在需要时改它**。三个旋钮全在 `torch._inductor.config.cpp` 里。

#### 3.1 先看骨架

```python
import glob, tempfile, torch
import torch._inductor.config as icfg
torch.set_num_threads(4)          # 自检脚本会注入 OMP_NUM_THREADS=8，本地必须钉住

def kernel_of(fn, arg):
    """编译一次，把生成的 C++ 从 trace 目录里捞出来（不依赖磁盘缓存）"""
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    icfg.force_disable_caches = True
    fn(arg)
    txt = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    i = txt.find('extern "C"')                 # 跳过头部 include
    return txt[i:txt.find("'''", i)]

chain = lambda v: v.mul(2).add(1).relu()

print("### fp32：4 个张量算子 → 1 个循环体")
print(kernel_of(lambda t: torch.compile(chain, dynamic=False)(t), torch.randn(4096)))

print("### bf16：同一条算子链，只挑出与循环排布有关的行")
c = kernel_of(lambda t: torch.compile(chain, dynamic=False)(t),
              torch.randn(4096, dtype=torch.bfloat16))
for ln in c.splitlines():
    if any(k in ln for k in ("void  kernel", "Vectorized<", "VectorizedN<",
                             "convert<", "pragma", "for(int64_t")):
        print(ln.rstrip())
```

```text
### fp32：4 个张量算子 → 1 个循环体
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4096LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                        auto tmp1 = static_cast<float>(2.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(1.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = at::vec::clamp_min(tmp6, decltype(tmp6)(0));
                        tmp7.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}

### bf16：同一条算子链，只挑出与循环排布有关的行
extern "C"  void  kernel(const at::BFloat16* in_ptr0,
    #pragma omp parallel num_threads(4)
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4096LL); x0+=static_cast<int64_t>(8LL))
                        auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::convert<float,2,at::BFloat16,1>(tmp0);
                        auto tmp3 = at::vec::VectorizedN<float,2>(tmp2);
                        auto tmp6 = at::vec::VectorizedN<float,2>(tmp5);
                        auto tmp9 = at::vec::convert<at::BFloat16,1,float,2>(tmp8);
```

读这段 C++，四个自由度全部可见，而且**一个都没浪费**：

- `x0 += 4`：宽度 $W=4$。`Vectorized<float>::loadu` 一次搬 4 个 float，就是一条 128 位 load。
- `#pragma omp parallel num_threads(4)`：并行 $P=4$。
- 循环体里 `tmp0 → tmp3 → tmp6 → tmp7` 一路往下，**中间没有任何 store**：这是 040 期说的 loop body 内联，mul/add/relu 三个算子的边界在源码层面就已经不存在了。
- `4096` 恰好整除 4，所以**没有尾部**（`no_redundant_loops=1` 时不会生成 `for(4096; 4096; ...)` 这种零次循环）。

bf16 那一段更值得看：循环步长变成 **8**，但**算术宽度还是 4**。`convert<float,2,at::BFloat16,1>` 把一个 8 lane 的 bf16 向量拆成 **2 个 fp32 向量**（`VectorizedN<float,2>`），算完再 `convert` 回去。也就是说：

> 128 位寄存器装 8 个 bf16，但里面真正做加乘的仍然是 2 个 4 宽 fp32 向量。**「向量宽度」和「算术宽度」是两个量。**

这条细节解释了为什么混合精度（002 期）在 CPU 上几乎不省时间：搬运量减半了，运算量没减，还多两次 convert。

#### 3.2 宽度由「位宽」决定，不是由「元素个数」决定

这里有个非常容易踩的坑。`config.cpp.simdlen` 看着像「一次几个元素」，实际上它跟 ISA 的**位宽**比：

```python
import glob, re, tempfile, torch
import torch._inductor.config as icfg
from torch._dynamo import reset
from torch._inductor import cpu_vec_isa as V

def kernel_of(N):
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    icfg.force_disable_caches = True
    torch.compile(lambda v: v.mul(2).add(1).relu(), dynamic=False)(torch.randn(N))
    txt = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    i = txt.find('extern "C"')
    return txt[i:txt.find("'''", i)]

torch.set_num_threads(4)
print(f"{'simdlen':>8}{'选中的 ISA':>18}{'循环步长':>10}{'Vectorized 出现次数':>20}")
for sl in (None, 4, 64, 128, 256):
    reset()
    icfg.cpp.simdlen = sl                       # 注意：单位是 bit，不是元素个数
    isa = V.pick_vec_isa()
    c = kernel_of(4096)
    step = sorted({int(s) for s in re.findall(r"x0\+=static_cast<int64_t>\((\d+)LL\)", c)})
    print(f"{str(sl):>8}{str(isa):>18}{str(step):>10}{c.count('Vectorized<'):>20}")
icfg.cpp.simdlen = None
```

```text
 simdlen           选中的 ISA      循环步长     Vectorized 出现次数
    None             asimd       [4]                   3
       4   INVALID_VEC_ISA       [1]                   0
      64   INVALID_VEC_ISA       [1]                   0
     128             asimd       [4]                   3
     256   INVALID_VEC_ISA       [1]                   0
```

`simdlen=128` 是「128 位」，挑中 NEON（打印名 `asimd`，因为它是 armv8-a 的 Advanced SIMD），fp32 恰好 4 个；`simdlen=4`（想表达「4 个元素」）匹配不到任何 ISA 的 `bit_width()`，直接落到 `INVALID_VEC_ISA`，**生成标量代码** —— 表里 `4 / 64 / 256` 三行的步长全是 1、`Vectorized` 一次都没出现。源码里就是一行字符串比较：

```python
# torch/_inductor/cpu_vec_isa.py:671 pick_vec_isa()
for isa in _valid_vec_isa_list:
    if config.cpp.simdlen == isa.bit_width():
        return isa
return invalid_vec_isa
```

元素个数是**位宽除以 dtype 位宽**算出来的：`VecNEON._dtype_nelements = {float: 4, bfloat16: 8, float16: 8}`（`cpu_vec_isa.py:246`）。这也是为什么同一个 `simdlen=128`，fp32 步长 4、bf16 步长 8。

#### 3.3 并行切分：门槛是个算得出来的数

什么时候加 `#pragma omp parallel`？看 `decide_parallel_depth`（`cpp.py:2752`）：

```python
if par >= 2 * threads or par == threads:
    break
if seq // threads < config.cpp.min_chunk_size:   # 默认 512
    break                                     # 「活不够分」
```

翻译成人话：**每个线程至少要分到 512 个元素，否则不值得开线程。** 于是门槛就是

$$
N_{\min} = T \times \texttt{min\_chunk\_size} = T \times 512
$$

这个公式是可以被实测验证的 —— 而且能预测到个位数：

```python
import glob, tempfile, torch
import torch._inductor.config as icfg
from torch._dynamo import reset

def kernel_of(chain, N):
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    icfg.force_disable_caches = True
    torch.compile(chain, dynamic=False)(torch.randn(N))
    txt = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    i = txt.find('extern "C"')
    return txt[i:txt.find("'''", i)]

chain = lambda v: v.mul(2).add(1).relu()

print("min_chunk_size = 512（config.py 与 cpp.py 里都写死这个默认值）")
print(f"{'threads':>8}{'预测门槛 = t×512':>17}   实测三个点       生成出来的 pragma")
for nt in (1, 2, 4):
    torch.set_num_threads(nt)
    reset()
    bound = nt * 512
    obs = {}
    for N in (bound - 1, bound, bound + 1):
        c = kernel_of(chain, N)
        obs[N] = "并行" if "omp parallel" in c else "单线程"
    tag = next((l.strip() for l in c.splitlines() if "omp parallel" in l), "（没有 omp）")
    print(f"{nt:>8}{bound:>17}   "
          + "  ".join(f"N={k}:{v}" for k, v in obs.items())
          + f"   {tag}")
torch.set_num_threads(4)
```

```text
min_chunk_size = 512（config.py 与 cpp.py 里都写死这个默认值）
 threads     预测门槛 = t×512   实测三个点       生成出来的 pragma
       1              512   N=511:单线程  N=512:单线程  N=513:单线程   （没有 omp）
       2             1024   N=1023:单线程  N=1024:并行  N=1025:并行   #pragma omp parallel num_threads(2)
       4             2048   N=2047:单线程  N=2048:并行  N=2049:并行   #pragma omp parallel num_threads(4)
```

三点值得注意：

1. `T=2` 的门槛是 1024，`T=4` 是 2048，**逐点命中**。公式不是玄学。
2. `T=1` 那行永远是单线程 —— 不是因为门槛没到，而是 `par == threads` 在第一次迭代就 break：只有 1 个线程时，开 omp 纯属白交线程开销。源码里那条 `par == threads` 判据的用途就在这里。
3. `num_threads(4)` 是**编译进产物**的常量。换线程数 = 换一份 kernel，重新编译。想让产物不随线程数变，可以开 `config.cpp.dynamic_threads = True`，生成的 pragma 会退化成裸的 `#pragma omp parallel`（源码注释原话：单线程负载会有轻微性能下降）。

那加了线程到底有用没用？取决于你站在 roofline 的哪一侧：

```python
import time, torch

def tmin(fn, reps=15, warmup=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best

N = 1 << 24                       # 64 MB float32
x = torch.randn(N)
GB = N * 4 / 1e9

print(f"N = 16M float32（64 MB），每档都重新编译（num_threads 是编译期常量）")
print(f"{'threads':>8}{'x*2+1 (ms)':>14}{'写+读 GB/s':>13}{'sum (ms)':>12}{'读 GB/s':>10}")
for nt in (1, 2, 4):
    torch.set_num_threads(nt)
    fp = torch.compile(lambda v: v * 2 + 1, dynamic=False)
    fs = torch.compile(lambda v: v.sum(), dynamic=False)
    fp(x); fs(x)
    tp = tmin(lambda: fp(x))
    ts = tmin(lambda: fs(x))
    print(f"{nt:>8}{tp*1e3:>14.2f}{2*GB/tp:>13.1f}{ts*1e3:>12.2f}{GB/ts:>10.1f}")
torch.set_num_threads(4)
```

这一段会抖动，所以直接把某一次的真实输出贴在这里（缩进块，不参与自动逐行复核）：

    N = 16M float32（64 MB），每档都重新编译（num_threads 是编译期常量）
     threads    x*2+1 (ms)     写+读 GB/s    sum (ms)    读 GB/s
           1          1.19        113.2        2.43      27.7
           2          1.03        129.9        1.27      52.7
           4          1.08        123.8        0.79      85.1

三次运行里 `x*2+1` 单线程 1.16~1.36 ms、4 线程 1.06~1.08 ms；`sum` 单线程 2.38~2.43 ms、4 线程 0.79~0.89 ms。结论的方向三次一致。

两列放一起对比才有意思：

- `x*2+1`：每元素 1 乘 1 加、搬 8 字节（读 4 写 4），算术强度极低。**单线程已经 113 GB/s，是这台机器内存墙（≈130 GB/s）的 87%**，2 线程撞墙（130 GB/s），4 线程甚至略退（124 GB/s，线程开销）。总收益 1.10×。
- `sum`：每个元素只有 1 次加法，但**加法之间有依赖链**（4 个累加器轮转），单线程只有 27.7 GB/s = 墙的 21%。这是典型的计算受限，加线程近乎线性：4 线程 85 GB/s，**3.06×**。

同样一个「融合体内部的一层循环」，一边加线程无用，一边加线程三倍。**这不是线程的功劳，是算术强度在决定。** 这条判据换到 GPU 上完全一样：memory-bound 的 elementwise kernel 加 warp 没用，compute-bound 的 GEMM 加 warp 才有效。

#### 3.4 尾部：不是标量收尾，是掩码向量

N 不能被 4 整除会怎样？很多人以为是「向量循环 + 标量循环」两段。实际生成的是**两个分支共用一个步长 4 的循环**，第二个分支用**带长度的 loadu**：

```python
import glob, tempfile, torch
import torch._inductor.config as icfg
torch.set_num_threads(4)

def kernel_of(fn, arg):
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    icfg.force_disable_caches = True
    fn(arg)
    txt = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    i = txt.find('extern "C"')
    return txt[i:txt.find("'''", i)]

chain = lambda v: v.mul(2).add(1).relu()
c = kernel_of(lambda t: torch.compile(chain, dynamic=False)(t), torch.randn(100003))

# 只截循环体：从 #pragma omp for 开始到循环结束
lines = c.splitlines()
i0 = next(k for k, l in enumerate(lines) if "#pragma omp for" in l)
i1 = next(k for k, l in enumerate(lines) if "= nullptr;" in l)
print(f"N = 100003，100003 = 4 × 25000 + 3 → 最后 3 个元素走第二个分支")
print("\n".join(lines[i0 + 1:i1 - 2]))
```

```text
N = 100003，100003 = 4 × 25000 + 3 → 最后 3 个元素走第二个分支
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(100003LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(100000LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                        auto tmp1 = static_cast<float>(2.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(1.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = at::vec::clamp_min(tmp6, decltype(tmp6)(0));
                        tmp7.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                    if(C10_UNLIKELY(x0 >= static_cast<int64_t>(100000LL) && x0 < static_cast<int64_t>(100003LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(3LL));
                        auto tmp1 = static_cast<float>(2.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(1.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = at::vec::clamp_min(tmp6, decltype(tmp6)(0));
                        tmp7.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(3LL));
                    }
                }
            }
```

两处 `if(C10_LIKELY(...))` / `if(C10_UNLIKELY(...))` 落在**同一个 `x0` 循环**里，第二个分支的 `loadu(ptr, 3)` 和 `store(ptr, 3)` 是**定长参数 3**：编译器知道这是「3 个元素的向量」，会生成掩码/半宽指令，而不是退化成 3 次标量访问。

这条设计的好处是**只有一份循环体代码**（指令缓存友好、寄存器分配一次搞定），代价是循环里多了两个比较。`C10_LIKELY`/`C10_UNLIKELY` 就是告诉分支预测器主次。

如果想避免尾部，把 N 补齐到 4 的倍数（padding）比让编译器收拾零头更省。这也是 021 期（动态形状）里「形状越整齐，编译产物越好」的物理原因之一。

#### 3.5 归约：先摊平，再收口

`(t * 2).sum()` 的循环排法完全不同 —— 它不能用「读一个算一个写一个」的模式，因为输出只有一个标量。生成的骨架是**三层累加器**：

```python
import glob, tempfile, torch
import torch._inductor.config as icfg
torch.set_num_threads(4)

def kernel_of(fn, arg):
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    icfg.force_disable_caches = True
    fn(arg)
    txt = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    i = txt.find('extern "C"')
    return txt[i:txt.find("'''", i)]

c = kernel_of(lambda t: torch.compile(lambda v: v.mul(2).sum(), dynamic=False)(t),
              torch.randn(4096))
lines = c.splitlines()
i0 = next(k for k, l in enumerate(lines) if "float tmp_acc0 = 0" in l)
i1 = next(k for k, l in enumerate(lines) if "= nullptr;" in l)
print("(t * 2).sum()，N = 4096，threads = 4")
print("\n".join(lines[i0:i1 - 3]))
```

```text
(t * 2).sum()，N = 4096，threads = 4
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            at::vec::Vectorized<float> tmp_acc0_vec_arr[4];
            for (int i = 0; i < 4; i++)
            {
                tmp_acc0_vec_arr[i] = at::vec::Vectorized<float>(0);
            }
            float tmp_acc0_arr[4];
            for (int i = 0; i < 4; i++)
            {
                tmp_acc0_arr[i] = 0;
            }
            #pragma omp parallel num_threads(4)
            {
                int tid = omp_get_thread_num();
                at::vec::Vectorized<float> tmp_acc0_vec_local = at::vec::Vectorized<float>(0);
                float tmp_acc0_local = 0;
                #pragma omp for
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4096LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp1 = static_cast<float>(2.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp_acc0_vec_local = tmp_acc0_vec_local + tmp3;
                        }
                    }
                }
                tmp_acc0_vec_arr[tid] = tmp_acc0_vec_local;
                tmp_acc0_arr[tid] = tmp_acc0_local;
            }
            for (int tid = 0; tid < 4; tid++)
            {
                tmp_acc0_vec = tmp_acc0_vec + tmp_acc0_vec_arr[tid];
            }
            for (int tid = 0; tid < 4; tid++)
            {
                tmp_acc0 = tmp_acc0 + tmp_acc0_arr[tid];
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
```

三个层次一目了然：

1. **向量累加器**（4 个 lane 各算各的）—— `tmp_acc0_vec = tmp_acc0_vec + tmp3`，一条 `fadd.4s` 顶四次加法。
2. **线程私有副本**（`tmp_acc0_vec_local`）—— 每个线程自己累，最后写进 `tmp_acc0_vec_arr[tid]`，**过程中没有任何锁**。
3. **收口**（reduction suffix）—— 先跨线程把 4 个副本加起来（`for tid = 0..3`），再用 `vec_reduce_all` 把 4 个 lane 横向相加成一个标量。

注意循环里**同时维护了 4 个向量累加器和 4 个标量累加器**：向量那个用于主路径，标量那个给尾部/掩码路径用（本例 N=4096 整除，标量那份从头到尾是 0，但代码还是生成了）。这是「一份源码兼顾两条路径」的代价。

#### 3.6 分块累加：cascade 的深度是算出来的

归约还有一个隐藏参数：**每累加多少下就把当前和「存进栈里」**。这是成对求和（pairwise summation）的工程版，源码在 `torch/include/torch/csrc/inductor/cpp_prefix.h:114`：

```cpp
template <typename T, uint64_t kChunkSize>
struct CascadeSumHelper {
  std::vector<T> sum_stk{};
  uint64_t depth{0};      // sum_stk 的深度
  uint64_t num_chunks{0}; // 已经存进 sum_stk 的块数
  uint64_t index{0};      // 当前块内位置
  CascadeSumHelper(uint64_t N) {
    const uint64_t m = (N + kChunkSize - 1) / kChunkSize; // div up
    depth = ceil_log2_u64(m);
    sum_stk.assign(std::max(depth, static_cast<uint64_t>(1)), T(0));
  }
};
```

```python
import glob, re, tempfile, torch
import torch._inductor.config as icfg
torch.set_num_threads(4)

def kernel_of(fn, arg):
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    icfg.force_disable_caches = True
    fn(arg)
    txt = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    i = txt.find('extern "C"')
    return txt[i:txt.find("'''", i)]

def depth_of(vec_iter):
    """cpp_prefix.h 里的 depth = ceil_log2(ceil(N / kChunkSize))"""
    return max(0, (-(-vec_iter // 4096) - 1).bit_length())

print("v.sum()，fp32，threads = 4。kChunkSize = 4096（单位：向量迭代）")
print(f"{'N':>9}{'每线程元素':>11}{'向量迭代':>10}{'num_chunks':>12}{'栈深度':>8}  实际行为")
for N in (4096, 4097, 16384, 65536, 65540, 262144, 1 << 20):
    c = kernel_of(lambda t: torch.compile(lambda v: v.sum(), dynamic=False)(t), torch.randn(N))
    if "CascadeSumHelper" not in c:
        print(f"{N:>9}{'-':>11}{'-':>10}{'-':>12}{'-':>8}  没有 helper，纯累加")
        continue
    vec = int(re.findall(
        r"(?<!scalar_)(?<!masked_)cascade_helper\d*\(static_cast<int64_t>\((\d+)LL\)\)", c)[0])
    dep = depth_of(vec)
    print(f"{N:>9}{-(-N // 4):>11}{vec:>10}{-(-vec // 4096):>12}{dep:>8}  "
          + ("空转（等价纯累加）" if dep == 0 else f"真分块，{dep} 层栈"))

```

```text
v.sum()，fp32，threads = 4。kChunkSize = 4096（单位：向量迭代）
        N      每线程元素      向量迭代  num_chunks     栈深度  实际行为
     4096          -         -           -       -  没有 helper，纯累加
     4097       1025       256           1       0  空转（等价纯累加）
    16384       4096      1024           1       0  空转（等价纯累加）
    65536      16384      4096           1       0  空转（等价纯累加）
    65540      16385      4097           2       1  真分块，1 层栈
   262144      65536     16384           4       2  真分块，2 层栈
  1048576     262144     65536          16       4  真分块，4 层栈
```

`chunk_size = 4096` 是硬编码的（`cpp.py:2416`），注释写着 "chunk size to balance accuracy and performance"。于是：

- **4 线程时 N ≤ 65536 全部 `depth=0`**（每线程 16384 元素 = 4096 次向量迭代 = 正好 1 块），到 65540（4097 次迭代）才翻成 `depth=1`。低于门槛时 helper 被创建但 `depth=0`，`cascade_sum_combine` 里 `if (c->depth > 0)` 直接跳过 —— 建了个空壳，行为和纯累加一模一样。
- 栈深是**对数**：$d=\lceil \log_2 \lceil V/4096 \rceil \rceil$（$V$ = 每线程向量迭代数），所以 **N 每翻 2 倍，栈深 +1**。实测 262144 → 2 层，1M → 4 层（65540 是 1 层、131072 也是 1 层）。
- 门槛**随线程数移动**：1 线程时每线程就是全部元素，所以到 N = 16388（4097 次迭代）就已经 `depth=1`，比 4 线程的 65536 低 4 倍。

「建了 helper 但空转」这件事本身是个好例子：**光看「有没有 CascadeSumHelper」判断有没有生效是错的**，要看 `depth`。

#### 3.7 tile：内层不能连续时的第三条路

前面所有讨论都假设「内层循环访问的地址连续」。当张量被转置、而表达式需要**同时读两个不同 index 映射**时，这个假设崩了。三种结局，实测：

```python
import glob, re, tempfile, torch
import torch._inductor.config as icfg
from torch._dynamo import reset
torch.set_num_threads(4)

def kernel_of(chain, arg):
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    icfg.force_disable_caches = True
    torch.compile(chain, dynamic=False)(arg)
    txt = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    i = txt.find('extern "C"')
    return txt[i:txt.find("'''", i)]

x = torch.randn(256, 256)
print(f"{'表达式（输入 256×256）':<24}{'循环步长':>10}{'寄存器 tile':>12}")
for name, f in [("t + 1", lambda t: t + 1),
                ("t.t() + 1", lambda t: t.t() + 1),
                ("t + t.t()", lambda t: t + t.t()),
                ("t[:, ::2] + 1", lambda t: t[:, ::2] + 1)]:
    reset()
    c = kernel_of(f, x)
    step = sorted({int(s) for s in re.findall(r"x\d+\+=static_cast<int64_t>\((\d+)LL\)", c)})
    print(f"{name:<24}{str(step):>10}{('4×4' if 'alignas(' in c else '无'):>12}")

print("\n### t + t.t() 的循环体（内层 4 元素一起走，先把另一路搬进寄存器 tile）")
reset()
c = kernel_of(lambda t: t + t.t(), x)
for ln in c.splitlines():
    s = ln.strip()
    if any(k in s for k in ("for(int64_t", "alignas(", "transpose_mxn", "loadu", "tmp3 +", "store(")):
        print("   " + s)
```

```text
表达式（输入 256×256）               循环步长    寄存器 tile
t + 1                          [4]           无
t.t() + 1                      [4]           无
t + t.t()                      [4]         4×4
t[:, ::2] + 1                  [1]           无

### t + t.t() 的循环体（内层 4 元素一起走，先把另一路搬进寄存器 tile）
   for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(256LL); x0+=static_cast<int64_t>(4LL))
   for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
   alignas(std::max(std::size_t(4), alignof(float))) float tmp1[4*4];
   transpose_mxn<float,static_cast<int64_t>(4),static_cast<int64_t>(4),false>(in_ptr0 + static_cast<int64_t>(x0 + 256LL*x1), static_cast<int64_t>(256LL), tmp1, static_cast<int64_t>(4));
   auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 256LL*x0 + 256LL*x0_inner), static_cast<int64_t>(4));
   auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<int64_t>(4LL*x0_inner), static_cast<int64_t>(4));
   tmp3.store(out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0 + 256LL*x0_inner));
```

`t + t.t()` 里两个操作数的地址映射不同（一个按行、一个按列），Inductor 的处理是 **CppTile2DKernel**（`cpp.py:3845` 的注释原话：*"A vector kernel that handles the 2d tiles with the tile size defined in tiling_factor … a transposition is applied on the tile to make the access contiguous from the inner-most loop axis"*）：

1. 外层 `x0 += 4`、内层 `x1 += 4` —— 循环变成 2D，每格 4×4 = 16 个元素。
2. `alignas(...) float tmp1[4*4]` —— 一个 4×4 的**寄存器/栈上 tile**。
3. `transpose_mxn<float,4,4,false>(in_ptr0 + x0 + 256*x1, 256, tmp1, 4)` —— 一次性把 4 行各 4 个元素搬进来并**转置**，让「原来按列方向的连续」变成「tile 内的行连续」。
4. 之后每一行都是 `loadu(..., 4)` + `loadu(tmp1 + 4*x0_inner, 4)` + 相加 + `store` —— 全部是 4 宽的向量路径。

**这就是 GEMM 里「shared memory tile + 转置」在 CPU 上的等价物**，只是 tile 放在栈上（寄存器溢出到 L1），尺寸由 SIMD 宽度定（4×4），不是手调的 32×32。GPU 上同一件事叫 `XBLOCK` 或 `CUTLASS` 的 `ThreadblockShape`，尺寸大得多（因为要摊薄 HBM 延迟而不是 L1 带宽）。

而 `t[:, ::2]` 是第三条路：步长 2 的访问连 tile 都救不了（tile 内也不连续），**直接退化成步长 1 的标量循环**。所以「切片/步长不是 1」是向量化的硬杀手 —— 想要性能就先 `.contiguous()`（030 期 KV 池切分里同一条规律）。

#### 3.8 排法会改变浮点结果，但只改变一次

1M 个 fp32 求和，把「线程数 × 是否向量化」全排列一遍：

```python
import torch
import torch._inductor.config as icfg
from torch._dynamo import reset

torch.manual_seed(0)
x = torch.randn(1 << 20)                       # 1M float32
truth = x.double().sum().item()
print(f"N = 1M，fp64 参考 = {truth:.9f}，1 ulp ≈ {abs(truth) * 1.1920929e-07:.2e}\n")

seen = {}
print(f"{'配置':<26}{'结果（精确到比特）':>24}{'相对误差':>12}")
for nt in (1, 2, 4):
    torch.set_num_threads(nt)
    for vec in (True, False):
        reset()
        icfg.cpp.vec_isa_ok = vec              # False → 生成标量源码
        f = torch.compile(lambda v: v.sum(), dynamic=False)
        f(x)
        v = float(f(x))
        seen.setdefault(float(v).hex(), []).append(f"compiled·{nt}线程·{'向量' if vec else '标量'}")
        print(f"compiled {nt} 线程 {'向量' if vec else '标量':<8}"
              f"{v:>24.15f}{abs(v - truth) / abs(truth):>12.3e}")

icfg.cpp.vec_isa_ok = True
for nt in (1, 2, 4):
    torch.set_num_threads(nt)
    v = float(torch.sum(x))
    seen.setdefault(float(v).hex(), []).append(f"eager·{nt}线程")
    print(f"eager    {nt} 线程 {'':<8}{v:>24.15f}{abs(v - truth) / abs(truth):>12.3e}")
torch.set_num_threads(4)

print(f"\n9 个配置 → {len(seen)} 个不同的和")
```

```text
N = 1M，fp64 参考 = -1237.836725317，1 ulp ≈ 1.48e-04

配置                                       结果（精确到比特）        相对误差
compiled 1 线程 向量         -1237.835693359375000   8.337e-07
compiled 1 线程 标量         -1237.837524414062500   6.456e-07
compiled 2 线程 向量         -1237.835693359375000   8.337e-07
compiled 2 线程 标量         -1237.837524414062500   6.456e-07
compiled 4 线程 向量         -1237.835693359375000   8.337e-07
compiled 4 线程 标量         -1237.837524414062500   6.456e-07
eager    1 线程            -1237.836914062500000   1.525e-07
eager    2 线程            -1237.836914062500000   1.525e-07
eager    4 线程            -1237.836914062500000   1.525e-07

9 个配置 → 3 个不同的和
```

三件事实，逐条说：

1. **同一种排法，线程数 1/2/4 结果逐比特相同**（compiled 向量那 3 行、compiled 标量那 3 行、eager 那 3 行，各自内部完全一致）。这正是 3.6 节 cascade 分块的设计目的：把累加树钉在一个与线程数无关的结构上，让结果**可复现**。这跟 038 期「换 SDPA 后端参数就变了」形成对照 —— **改分块方式会变，改线程数不会变**。
2. **不同排法给出不同的和**：9 个配置落在 3 个值上，向量版和标量版差 1 个 ulp 量级（`8.337e-07` vs `6.456e-07` 相对误差）。
3. **别急着说「compiled 更准/更不准」**。这一组数据里 eager 的相对误差（`1.525e-07`）比 compiled 向量版小 5.5 倍；但这只是一组随机数据、一个 N。要下「谁更准」的结论得扫 N、扫分布（037 期的方法论：先算零假设，再看差多远）。**能确定的只有「排法不同 → 比特不同」这一件事。**

---

### 4. 围绕该领域展开

#### 4.1 同一件事，四套名字

| 概念 | CPU / Inductor C++ | Triton | CUDA C++ | CUTLASS |
|---|---|---|---|---|
| 一次算几个 | `Vectorized<float>` 4 lane（NEON 128 bit） | `tl.arange` 的 BLOCK 维、编译器向量化 | `float4` / `int4` 向量类型 | `InstructionShape` |
| 一个 CTA/program 处理多少 | omp 线程分到的那一段 | `XBLOCK` | `blockDim.x` × 每线程元素 | `ThreadblockShape` |
| 一个 warp 处理多少 | （无 warp 概念） | `XBLOCK / num_warps` | 每 warp 32 线程 × 每线程元素 | `WarpShape` |
| 分块累加/暂存 | 栈上 `float tmp[4*4]` | `tl.zeros((BLOCK_M, BLOCK_N))` | `__shared__` 数组 | shared memory tile |
| 尾部 | `loadu(ptr, 3)` 掩码 | `mask=` 的布尔张量 | `if (idx < N)` | predicated load |

**跨硬件通用的抽象**：roofline / 算术强度、访存合并的动机、tiling 的动机（把复用搬到更近的存储）、成对求和的误差界、尾部掩码的必要性。这五条换任何加速器都成立。

**NVIDIA 特有的具体形式**：warp = 32 线程这个粒度、`cp.async` / TMA 的异步拷贝、`wgmma` / `tcgen05` 的矩阵指令、线程块簇（cluster）、2:4 结构化稀疏。

**x86/ARM 特有的**：AVX2/AVX512 的 256/512 bit、NEON 的 128 bit、SVE 的可变长、`#pragma omp` 这套线程模型，以及 SLEEF（Inductor 的 `Vectorized` 里 `sin/cos/exp` 走的向量数学库）。

#### 4.2 CUDA 侧的对应物：XBLOCK 与 num_warps

本机没有 CUDA，所以这一段是**从上游源码里读到的启发式**，不是实测。但它回答的是和 3.3 节同一个问题：**一个 program 处理多少元素、分几个 warp**。

```python
import torch._inductor.runtime.triton_heuristics as TH

class FakeConfig:                      # 本机没装 triton，Config 退化成一个空 object，套个替身接住
    def __init__(self, kwargs, num_warps=None, num_stages=None):
        self.kwargs, self.num_warps, self.num_stages = dict(kwargs), num_warps, num_stages

TH.Config = FakeConfig
from torch._inductor.heuristics.registry import get_codegen_heuristic

h = get_codegen_heuristic("pointwise", "cuda")
mk = lambda sh, x, num_elements_per_warp=256: TH.triton_config(sh, x, num_elements_per_warp=num_elements_per_warp)

print("pointwise 启发式给出的候选：XBLOCK = 一个 program 处理多少元素")
print(f"{'numel':>12}{'每线程元素':>12}   候选 (XBLOCK, num_warps, 每线程元素)")
for numel in (2048, 4096, 65536, 1 << 20, 1 << 24):
    bs = max(256, min(numel // 128, 1024))          # 目标：每个 CTA 的 block_size
    cfgs = h.get_configs({"x": numel}, bs, mk, [], None, {})
    cells = [f"({c.kwargs['XBLOCK']}, warps={c.num_warps}, "
             f"{c.kwargs['XBLOCK'] // (c.num_warps * 32)})" for c in cfgs]
    print(f"{numel:>12}{bs:>12}   " + "  ".join(cells))
```

```text
pointwise 启发式给出的候选：XBLOCK = 一个 program 处理多少元素
       numel       每线程元素   候选 (XBLOCK, num_warps, 每线程元素)
        2048         256   (256, warps=4, 2)  (128, warps=4, 1)
        4096         256   (256, warps=4, 2)  (128, warps=4, 1)
       65536         512   (512, warps=4, 4)  (256, warps=4, 2)
     1048576        1024   (1024, warps=4, 8)  (512, warps=8, 2)
    16777216        1024   (1024, warps=4, 8)  (512, warps=8, 2)
```

读法（`num_warps × 32` = 这个 program 的线程数）：

- `XBLOCK=1024, warps=4` → 128 个线程，每线程 8 个元素。**和 CPU 上一线程内层走 8 次是同一个动作**，只不过 GPU 用 8 次独立 load 来填流水线（延迟隐藏），CPU 用 4 lane SIMD。
- 始终给**两个候选**让 autotune 实测选：`(1024, warps=4)` 是 128 个线程、每线程 8 个元素（寄存器压力大、并行度低），`(512, warps=8)` 是 256 个线程、每线程 2 个元素（占用率高、并行度高）。**同一道题的两个解法，本质上就是「让每个线程多干活」还是「多开线程」，跟 CPU 上 3.3 节 `min_chunk_size` 权衡的是同一件事。**
- 小 numel（2048）时候选退化到 `(256, warps=4)` 和 `(128, warps=4)`：**每线程 2 个或 1 个元素**，因为元素太少，摊不开。

这里可以看到 CPU 与 GPU 的一个结构性差异：CPU 的并行门槛是「**每个线程至少 512 个元素**」（怕线程开销），GPU 的门槛低得多（**每线程 1~2 个元素也能起**），因为 GPU 的线程切换是硬件调度、几乎没有开销。**同一个「活够不够分」的问题，两个平台给出的答案是 512 和 1。**

#### 4.3 和已发期数的关系

- **040 期（算子融合）**：那期讲「哪些算子进同一个循环体」，本期讲「进去之后循环怎么走」。两期合起来才是完整的 Inductor 下游。
- **039 期（ATen / Dispatch）**：再下一层。本期所有讨论都在「kernel 已经选定、已经生成」之后。
- **038 期（SDPA 后端选择）**：那期有一个「门表 + 优先级 + 兜底」的三层结构；本期 `decide_parallel_depth` 是同一个思路的极简版：两条 break 条件就是门表。
- **005 期（Flash Attention）**：FA 是 tile 用到极致的样本 —— 它把整个 S×S 矩阵切成 `BLOCK_M × BLOCK_N` 的 tile，每个 tile 只算一次就丢，这才是「省显存」的真正来源。本期的 4×4 tile 是同一个思想的最小实例。
- **008 期（Triton）**：Triton 让你**手写**本期讲的这些自由度（`BLOCK_SIZE` 就是宽度 × 并行），Inductor 的 cpp/triton 后端则是**自动填**这些参数。两条路通向同一组旋钮。
- **037 期（零假设）**：本期的三个阈值（512、4096、65536）都是「先算清结构、再解读观测」的样本 —— 知道门槛是 65536，才不会把 4097 元素那个空转的 helper 当成「分块生效了」。
- **030 期（PD 分离与 KV 存储）**：`t[:, ::2]` 那条结论（步长 ≠ 1 就退标量）在 KV 池的分页搬移里天天遇到。

#### 4.4 一句话说清「这层循环属于谁」

| 层 | 谁排的循环 | 你能改什么 |
|---|---|---|
| 手写 CUDA / CUTLASS | 你 | 全部 |
| Triton kernel | 你（只写 block 级） | `BLOCK_SIZE`、`num_warps`、`num_stages` |
| Inductor triton 后端 | 编译器 | `config.triton.*`、启发式 |
| Inductor cpp 后端（本期） | 编译器 | `config.cpp.{simdlen, min_chunk_size, threads, dynamic_threads}` |
| cuBLAS / oneDNN / MKL | 厂商 | 只剩算法选择 |

越往下，可控性越低、但覆盖面越广。**知道每一层能拧什么，才知道该在哪一层花钱。**

---

### 5. 什么时候该用 / 不该用

**该动手看这一层的时候：**

- 融合已经发生（`output_code.py` 里只有一个 kernel），但加速比远低于「算子个数」这个上限。这时瓶颈在循环体内部。
- 你在写自定义算子（`torch.library`、Triton、`cpp_extension`），需要自己决定宽度和切分。
- 你的输入形状「不整齐」（N 不是 4 的倍数、有切片有转置），怀疑掉进了标量路径或 tile 路径 —— 直接 dump 产物看步长是 4 还是 1。
- 你在排查「同一份代码换台机器/换线程数结果就变」的浮点问题（3.8 节）。

**不该动手的时候：**

- **不要手改生成的 C++**。它每次编译都重写，下次 shape 变了又是一份新的。要改就改 config 或改源码。
- **不要为了向量化去 padding 到 4 的倍数，除非 N 很小**。尾部本来就是掩码处理，代价是两个比较，不是 3 次标量访问。
- **不要在小张量上调线程数**。门槛以下根本不生成 omp，调了也没用；门槛附近加线程反而更慢（3.3 节 4 线程比 2 线程还慢一点）。
- **不要用「关掉向量化」来测向量化的收益**（`vec_isa_ok=False`）。关掉的是 **Inductor 的显式向量化**，C++ 编译器自己还会做一遍自动向量化，你测到的不是「有向量 vs 没向量」。这正是本期放弃那个实验的原因 —— 第一版跑出来「向量化 1.01×」，看着像结论，其实是测了个空。

---

### 6. 常见坑

**坑 1：`config.cpp.simdlen` 的单位是 bit，不是元素个数。**
写 `simdlen=4` 想表达「4 个元素」，会匹配不到任何 ISA 的 `bit_width()`，落到 `INVALID_VEC_ISA`，**静默变成标量代码**。要 4 个 fp32，得写 `128`。而且同一份 128 在 bf16 上自动变 8 个元素 —— 因为它比的是位宽。

**坑 2：改了 config 不 `torch._dynamo.reset()`，你以为换了配置，其实跑的还是旧 kernel。**
这个坑我这次真踩到了：3.8 节的第一版脚本忘了 reset，`vec_isa_ok=True` 和 `False` 两轮跑出**完全相同**的结果，看起来像「向量化不影响数值」。加了 `reset()` 才露出真差别（`-1237.835693` vs `-1237.837524`）。Dynamo 的缓存认的是「图 + guards」，它不认你在 `torch._inductor.config` 里改的旋钮。

**坑 3：`num_threads` 是编译期常量，而它取自 `torch.get_num_threads()`，后者又被 `OMP_NUM_THREADS` 影响。**
所以「我本地跑出来是 4 线程」和「CI 里跑出来是 8 线程」会得到**两份不同的产物、可能不同的数字**。写可复现的脚本必须显式 `torch.set_num_threads(N)`；要产物稳定则开 `config.cpp.dynamic_threads`。

**坑 4：看到 `CascadeSumHelper` 就以为分块生效了。**
它可能是个空壳（`depth=0`，`cascade_sum_combine` 里 `if (c->depth > 0)` 直接跳过）。判据是 `depth ≥ 1`，也就是每线程向量迭代数 > 4096，4 线程时对应 N > 65536。

**坑 5：「生成的是向量代码」不等于「更快」。**
3.3 节那张表就是反例：带宽受限的 `x*2+1` 从 1 线程加到 4 线程只快 1.10×，而计算受限的 `sum` 快 3.06×。**先判 roofline，再谈优化** —— 顺序反了就是在给已经撞墙的 kernel 加宽度。

---

### 7. 一句话总结

融合把 $m$ 个算子压成一层循环，而这一层循环只有四个旋钮 —— **宽度（硬件定的位宽）、并行（`threads × 512` 门槛算出来的）、分块（4096 次迭代一层 cascade）、尾部（掩码向量）** —— 拧之前先看它站在带宽墙还是算力墙的哪一侧，因为四个旋钮里只有一个能改变你撞哪面墙。

<details>
<summary>今日练习</summary>

**题目**：不编译、不算，只用 3.3 节的公式预测下面三个配置的产物长什么样，然后用 3.1 节的 `kernel_of` 验证。

1. `torch.set_num_threads(8)`，`N = 4096`，表达式 `x * 2 + 1` —— 会不会有 `#pragma omp parallel`？`num_threads` 写几？
2. `torch.set_num_threads(4)`，`N = 8191`，表达式 `x.relu()` —— 循环步长几？有几个 `if(C10_...)` 分支？尾部几个元素？
3. `torch.set_num_threads(1)`，`N = 100000`，表达式 `x.sum()` —— 有没有 `CascadeSumHelper`？`depth` 是几？

**参考答案**

预测依据：

1. 门槛 = `8 × 512 = 4096`，`N = 4096` 恰好 `4096 // 8 = 512`，**不小于** 512，所以**会**并行，pragma 里写 `num_threads(8)`。
2. 8191 = 4×2047 + 3，步长 4，两个 `if(C10_...)`（`LIKELY` 主体 + `UNLIKELY` 尾部），尾部 **3** 个元素用 `loadu(ptr, 3)`。
3. 1 线程时 `par == threads` 第一次就 break —— 但归约的 cascade 跟并行是两件事：`N=100000 > 4096` 会建 helper；1 线程下每线程向量迭代 = `100000/4 = 25000`，`ceil(25000/4096) = 7`，`ceil_log2(7) = 3` → **depth = 3**。

验证脚本（把三组预测一次性跑掉）：

```python
import glob, re, tempfile, torch
import torch._inductor.config as icfg
from torch._dynamo import reset

def kernel_of(chain, N):
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    icfg.force_disable_caches = True
    torch.compile(chain, dynamic=False)(torch.randn(N))
    txt = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    i = txt.find('extern "C"')
    return txt[i:txt.find("'''", i)]

# 1) 8 线程，N = 4096，x*2+1
torch.set_num_threads(8); reset()
c = kernel_of(lambda v: v * 2 + 1, 4096)
pragma = next((l.strip() for l in c.splitlines() if "omp parallel" in l), "（没有 omp）")
print(f"1) 门槛 8×512=4096，N=4096 → {pragma}")

# 2) 4 线程，N = 8191，x.relu()
torch.set_num_threads(4); reset()
c = kernel_of(lambda v: v.relu(), 8191)
step = sorted({int(s) for s in re.findall(r"x0\+=static_cast<int64_t>\((\d+)LL\)", c)})
print(f"2) 步长={step}  分支数={c.count('if(C10_')}  "
      f"尾部 loadu(...,{re.findall(r'loadu\(in_ptr0 \+ static_cast<int64_t>\(x0\), static_cast<int64_t>\((\d+)LL\)\)', c)[-1]})")

# 3) 1 线程，N = 100000，x.sum()
torch.set_num_threads(1); reset()
c = kernel_of(lambda v: v.sum(), 100000)
vec = re.findall(r"(?<!scalar_)(?<!masked_)cascade_helper\d*\(static_cast<int64_t>\((\d+)LL\)\)", c)
dep = max(0, (-(-int(vec[0]) // 4096) - 1).bit_length()) if vec else None
print(f"3) CascadeSumHelper={'在' if 'CascadeSumHelper' in c else '不在'}  "
      f"每线程向量迭代={vec[0] if vec else '-'}  depth={dep}")
torch.set_num_threads(4)
```

三条预测全部命中，说明「门槛公式 + 尾部掩码 + cascade 深度」这三个数**可以离线算出来**，不需要每次都编译一遍去试。

**再追问一步**：第 3 题如果把线程数改成 4，`depth` 会变成几？（提示：每线程元素 25000，向量迭代 6250，`ceil(6250/4096)=2`，`ceil_log2(2)=1`。）**线程越多，每线程的块越少，cascade 层数越浅** —— 也就是说「分块的精度收益」和「并行的速度收益」在同一条轴上互相拉扯。

</details>
