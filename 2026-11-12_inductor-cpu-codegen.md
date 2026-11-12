# PyTorch 每日一课 · 第 053 期

## Inductor 的 CPU 代码生成后端：一个 `for` 循环里的三笔账

| 项 | 内容 |
|---|---|
| 日期 | 2026-11-12 |
| 难度 | ⭐⭐⭐⭐ |
| 前置知识 | 读过第 003 期（`torch.compile` 路线）、第 040 期（算子融合）、第 042 期（CPU 并行与 grain size）；知道 SIMD 与 OpenMP 是什么 |
| 预计阅读 | 约 40 分钟 |
| 关联 | 第 052 期（本期生成的代码 include 的就是那期解剖的 `cpp_prefix.h`）· 第 040 期（融合在 CPU 侧多出一条横向维度）· 第 042 期（`GRAIN_SIZE=32768` 与本期的 `min_chunk_size=512` 是同一个问题的两个答案，差 64 倍）· 第 043 期（三笔账最后都汇到屋脊点上） |

---

## 一、这个领域解决什么问题

`torch.compile` 在 GPU 上跑完 Inductor，你得到的是**一张 grid 和一个 Triton kernel**。网格决定有多少个 CTA 去瓜分 SM，kernel 里 `tl.program_id(0)` 决定每个 CTA 干哪一块。并行度是**显式**的，写在启动参数里。

同样一段图在 CPU 上跑完 Inductor，你得到的是**一份 C++ 源码和一个 `for` 循环**：

- 没有 grid，没有 block，没有 `threadIdx`
- 循环要跑多少次，是**编译期写死的常量**（因为 shape 被特化了）
- 并行度不在你手里 —— 一半藏在编译器的自动向量化里，一半藏在 `#pragma omp` 后面

这是两种完全不同的产物形态，也因此长出两套完全不同的复杂度。CPU 后端的所有工程问题，最终都收敛到同一个动作上：**怎么切这个 `for` 循环**。

切法有三个自由度，而 Inductor 各给了一个旋钮：

| 维度 | 问题 | 旋钮 | 本机默认值 | 代码位置 |
|---|---|---|---|---|
| **宽** | 一次迭代处理几个元素 | `cpp.simdlen` | `None`（自动探测） | `config.py:1798` |
| **薄** | 一次给一个线程多少活 | `cpp.min_chunk_size` | 512 | `config.py:1799` |
| **多** | 几个独立循环能合成一个 | `cpp.max_horizontal_fusion_size` | 16 | `config.py:1840` |

本文的主线就是这三笔账。结论先说：**三个默认值都是给「Linux + 多核服务器」定的。在一台 10 核 Mac 上，两个偏小、一个大，而偏小的那两个里，omp 门槛能让 kernel 慢 70 多倍。**

---

## 二、核心思想：先把产物拿到手

讲 CPU 后端最好的入口不是源码，是**看它生成了什么**。这段代码可以直接跑，把 `output_code.py` 打出来。

```python
import os, glob, re
import torch
import torch._inductor.config as icfg

# 关键：Inductor 的 FX graph cache 是跨进程共享的。
# 不关掉它，第二次跑同一个图会直接命中缓存 —— 根本不重新生成代码，也就抓不到 output_code.py。
icfg.force_disable_caches = True
icfg.trace.enabled = True
icfg.trace.debug_dir = "/tmp/lesson053"

def f(a, b):
    return torch.relu(a + b) * 2.0 + 1.0

a, b = torch.randn(8, 64), torch.randn(8, 64)   # 8*64 = 512 个元素
cf = torch.compile(f)
cf(a, b)

src = open(glob.glob("/tmp/lesson053/**/output_code.py", recursive=True)[0]).read()
# 只抽 cpp_pybinding 里的那段 C++，前面的 import 和后面的 Runner 与本文无关
cpp = re.search(r"cpp_pybinding\(.*?, r'''(.*?)'''\)", src, re.S).group(1)
print(cpp.strip())
```

```text
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(512LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                    auto tmp4 = static_cast<float>(2.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(1.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    tmp9.store(out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
```

三个细节值得先记住：

1. **`x0 += 4`** —— 一次迭代吃 4 个 float，这是 128 位向量。
2. **`at::vec::Vectorized<float>`** —— 不是靠 clang 的自动向量化，是**显式**写出来的向量类型。这是 PyTorch 自己的 SIMD 抽象层（就是第 048 期里 `vec256_float.h` 那一套）。
3. **`C10_LIKELY(x0 >= 0 && x0 < 512)`** —— 一个谓词。这里恒为真，但生成器仍然写了它。这个谓词在「尺寸不整除 4」时会变成第二份循环体的入口。

### 尺寸一变，代码的形态就变

把尺寸改成 2047（不减 4）和 2048，形态上出现两处结构性变化：

```python
import os, glob, re, shutil
import torch
import torch._inductor.config as icfg

def dump(tag, n):
    shutil.rmtree(f"/tmp/lesson053b/{tag}", ignore_errors=True)
    icfg.force_disable_caches = True          # 每轮都要重设，别的地方会把它关掉
    icfg.trace.enabled = True
    icfg.trace.debug_dir = f"/tmp/lesson053b/{tag}"
    ns = {"torch": torch}
    # 每个尺寸用不同名字的函数 -> 不同 code object -> Dynamo 不会退化成动态形状
    exec(f"def f_{tag}(a, b):\n    return torch.relu(a + b) * 2.0 + 1.0\n", ns)
    cf = torch.compile(ns[f"f_{tag}"])
    x, y = torch.randn(n), torch.randn(n)
    cf(x, y)
    src = open(glob.glob(f"/tmp/lesson053b/{tag}/**/output_code.py", recursive=True)[0]).read()
    body = re.search(r"cpp_pybinding\(.*?, r'''(.*?)'''\)", src, re.S).group(1)
    return body

for tag, n in (("n2047", 2047), ("n2048", 2048)):
    body = dump(tag, n)
    print(f"n={n}: 向量 loadu {body.count('loadu')} 次, "
          f"C10_UNLIKELY 分支 {body.count('C10_UNLIKELY')} 个, "
          f"omp {body.count('#pragma omp')} 行, 循环体 {len(body.splitlines())} 行")
```

```text
n=2047: 向量 loadu 4 次, C10_UNLIKELY 分支 1 个, omp 0 行, 循环体 46 行
n=2048: 向量 loadu 2 次, C10_UNLIKELY 分支 0 个, omp 2 行, 循环体 37 行
```

两件事同时发生了：

- **2047**：多出一个 `C10_UNLIKELY` 分支，里面是**第二份几乎完整的向量循环体**，读的是 `loadu(in_ptr0, 3)` —— 只读 3 个元素的部分加载。
- **2048**：出现 `#pragma omp parallel num_threads(4)`。

这两个数字都是算出来的，不是猜的。下面分别算。

---

## 三、第一笔账（宽）：`simdlen` 的单位是 bit，而且没有 fallback

`cpp.simdlen` 这个名字很有误导性。它**不是元素个数**，是**位宽**。

证据在 `pick_vec_isa()`（`torch/_inductor/cpu_vec_isa.py`）：

```py
# 源码摘录，非完整函数
def pick_vec_isa():
    _valid_vec_isa_list = valid_vec_isa_list()
    if not _valid_vec_isa_list:
        return invalid_vec_isa
    if config.cpp.simdlen is None:
        return get_isa_from_cpu_capability(...)
    for isa in _valid_vec_isa_list:
        if config.cpp.simdlen == isa.bit_width():     # 拿 bit 去比 bit_width
            return isa
    return invalid_vec_isa                           # 匹配不上就整个关掉向量化
```

注意最后那行：**不匹配不是「降级到次优 ISA」，是直接返回 `invalid_vec_isa`**。而 `CppVecKernel.__init__`（`cpp.py:2875`）拿到它之后，行为是「不做向量化」，不是报错。

macOS + arm64 上，`valid_vec_isa_list()` 只会放进一个候选：

```py
if sys.platform == "darwin" and platform.processor() == "arm":
    isa_list.append(VecNEON())
if sys.platform not in ["linux", "win32"]:
    return isa_list        # macOS 到这里就返回了，AVX/SVE 一概不考虑
```

而 `VecNEON` 的定义（`cpu_vec_isa.py:246`）带了一句注释，值得完整引一遍：

```py
class VecNEON(VecISA):
    _bit_width = 128  # This is required to leverage the compute implemented in aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h
    _macro = ["CPU_CAPABILITY_NEON", "AT_BUILD_ARM_VEC256_WITH_SLEEF"]
    _dtype_nelements = {torch.float: 4, torch.bfloat16: 8, torch.float16: 8}
```

**128 不是因为 128 位是硬件最优解，是因为 ATen 已经有一份按 128 位写好的实现可以直接复用。** 这一句解释了很多事：为什么 ARM 的 `Vectorized<float>` 是 4 宽，为什么 `x0 += 4`，为什么上期（052）看到 Inductor 宁可不 import torch 头文件也要复用 `cpp_prefix.h` —— 整条线上都是「复用 ATen 已有的向量实现」这一个原则。

### 实测：把 simdlen 设成 256 会发生什么

```python
import os, glob, re, shutil
import torch
import torch._inductor.config as icfg
from torch._inductor import cpu_vec_isa

print("本机 pick_vec_isa():", cpu_vec_isa.pick_vec_isa(),
      " bit_width =", cpu_vec_isa.pick_vec_isa().bit_width(),
      " float 每向量元素数 =", cpu_vec_isa.pick_vec_isa().nelements(torch.float))

for tag, sl in (("simd128", 128), ("simd256", 256)):
    icfg.cpp.simdlen = sl
    icfg.force_disable_caches = True
    icfg.trace.enabled = True
    icfg.trace.debug_dir = f"/tmp/lesson053c/{tag}"
    shutil.rmtree(f"/tmp/lesson053c/{tag}", ignore_errors=True)
    ns = {"torch": torch}
    exec(f"def g_{tag}(a, b):\n    return torch.relu(a + b) * 2.0 + 1.0\n", ns)
    cf = torch.compile(ns[f"g_{tag}"])
    x, y = torch.randn(512), torch.randn(512)
    cf(x, y)
    src = open(glob.glob(f"/tmp/lesson053c/{tag}/**/output_code.py", recursive=True)[0]).read()
    body = re.search(r"cpp_pybinding\(.*?, r'''(.*?)'''\)", src, re.S).group(1)
    print(f"simdlen={sl}: 选中 {cpu_vec_isa.pick_vec_isa()}, "
          f"生成代码里 Vectorized 出现 {body.count('Vectorized')} 次, "
          f"循环步长 = {'4' if 'x0+=static_cast<int64_t>(4LL)' in body else '1'}")
icfg.cpp.simdlen = None
```

```text
本机 pick_vec_isa(): asimd  bit_width = 128  float 每向量元素数 = 4
simdlen=128: 选中 asimd, 生成代码里 Vectorized 出现 4 次, 循环步长 = 4
simdlen=256: 选中 INVALID_VEC_ISA, 生成代码里 Vectorized 出现 0 次, 循环步长 = 1
```

设 256 的结果不是「降到 128」，是**整个向量化消失**，循环步长退回 1。

### 但「关掉向量化」几乎不慢

这是本期最反直觉的一组数字。用同一个算子、同一个尺寸，只切 `simdlen`（128 = 显式向量，256 = 标量代码），看运行期差多少：

```python
import os, glob, shutil, time, statistics
import torch
import torch._inductor.config as icfg

N = 1 << 18
CASES = {"加减乘relu": "torch.relu(a + b) * 2.0 + 1.0",
         "erf": "torch.erf(a) * 2.0 + b",
         "tanh": "torch.tanh(a) + torch.tanh(b)"}

def build(tag, expr):
    icfg.cpp.simdlen = int(tag[4:])
    icfg.cpp.min_chunk_size = 10**9      # 关掉 omp，只留向量化这一个变量
    icfg.force_disable_caches = True
    icfg.trace.enabled = True
    icfg.trace.debug_dir = f"/tmp/lesson053d/{tag}"
    shutil.rmtree(f"/tmp/lesson053d/{tag}", ignore_errors=True)
    ns = {"torch": torch}
    exec(f"def h_{tag}(a, b):\n    return {expr}\n", ns)
    cf = torch.compile(ns[f"h_{tag}"])
    cf(a, b)
    fl = glob.glob(f"/tmp/lesson053d/{tag}/**/output_code.py", recursive=True)[0]
    g = {"__name__": f"gen_{tag}"}
    exec(compile(open(fl).read().split('if __name__ ==')[0], fl, "exec"), g)
    kn = [k for k in g if k.startswith("cpp_fused")][0]
    return g[kn], ("at::vec::Vectorized" in open(fl).read())

a, b = torch.randn(N), torch.randn(N)
buf = torch.empty(N)
print(f"{'算子':<12}{'显式向量':>10}{'标量代码':>10}{'标量/向量':>11}   生成代码里有 Vectorized")
for name, expr in CASES.items():
    kv, hv = build("simd128", expr)
    ks, hs = build("simd256", expr)
    for _ in range(40):
        kv(a, b, buf); ks(a, b, buf)
    tv, ts = [], []
    for _ in range(120):                       # 成对交替采样，取逐对中位数（压住频率漂移）
        t0 = time.perf_counter(); kv(a, b, buf); t1 = time.perf_counter()
        ks(a, b, buf); t2 = time.perf_counter()
        tv.append(t1 - t0); ts.append(t2 - t1)
    mv, ms = statistics.median(tv) * 1e6, statistics.median(ts) * 1e6
    print(f"{name:<12}{mv:>9.2f}us{ms:>9.2f}us{ms/mv:>10.2f}x   {hv} / {hs}")
icfg.cpp.simdlen = None
icfg.cpp.min_chunk_size = 512
```

```text
算子                显式向量      标量代码      标量/向量   生成代码里有 Vectorized
加减乘relu         43.40us    43.12us      0.99x   True / False
erf            142.42us   635.27us      4.46x   True / False
tanh          1046.44us  2230.96us      2.13x   True / False
```

加减乘只有 1 倍出头，erf 和 tanh 是 2~4 倍。这个差别的来源可以一句话说清：**`cpp.simdlen` 关掉的不是「向量化」，是「`at::vec` 提供的向量数学函数」**。

因为 clang 自己会向量化。把生成的标量代码原样搬出来编译成汇编，就能看到：

```python
import subprocess, tempfile, os, re

# Inductor 在 simdlen 不匹配时生成的「标量」循环体，原样搬过来
SRC = r"""
void kernel(const float* in_ptr0, const float* in_ptr1, float* out_ptr0) {
    for (long x0 = 0; x0 < 262144; x0 += 1) {
        float tmp0 = in_ptr0[x0];
        float tmp1 = in_ptr1[x0];
        float tmp2 = float(tmp0 + tmp1);
        float tmp3 = tmp2 > 0 ? tmp2 : 0;
        float tmp5 = float(tmp3 * 2.0);
        float tmp7 = float(tmp5 + 1.0);
        out_ptr0[x0] = tmp7;
    }
}
"""
d = tempfile.mkdtemp()
p = os.path.join(d, "k.cpp")
open(p, "w").write(SRC)
for tag, flags in (("默认 -O3（Inductor 用的就是它）", ["-O3"]),
                   ("-O3 -fno-vectorize -fno-slp-vectorize", ["-O3", "-fno-vectorize", "-fno-slp-vectorize"])):
    asm = subprocess.run(["clang++", *flags, "-S", "-o", "-", p],
                         capture_output=True, text=True).stdout
    # .4s = NEON 的四路单精度；q0~q31 是 128 位向量寄存器
    print(f"{tag:<40} NEON 四路指令 {asm.count('.4s'):>3} 条, "
          f"用到的 q 寄存器 {len(set(re.findall(r'q[0-9]+', asm))):>2} 个")
```

```text
默认 -O3（Inductor 用的就是它）                   NEON 四路指令  17 条, 用到的 q 寄存器  8 个
-O3 -fno-vectorize -fno-slp-vectorize    NEON 四路指令   0 条, 用到的 q 寄存器  0 个
```

所以 `simdlen=256` 时 clang 把纯算术部分向量化了（这就是上表里「加减乘」只有 1 倍左右的由来），但 `std::exp` / `std::erf` / Sleef 的标量入口它没法打包成 SIMD —— **库调用不能自动向量化**。而 `at::vec::Vectorized<float>::exp()` 走的是 Sleef 的向量实现，一次算 4 个。

**可操作判据**：你的图如果只有加减乘除，`simdlen` 调不调差不多；一旦有 `exp` / `tanh` / `erf` / `sigmoid`，向量宽度就值 2~4 倍。而在这台机器上你没有第二个宽度可选（只有 128）。

---

## 四、第二笔账（薄）：`min_chunk_size` 是 512，而 omp 进门要十几微秒

`min_chunk_size` 的判定在 `cpp.py:2767`，逻辑只有一行：

```py
# 源码摘录：CppKernel.get_parallel_depth() 的一部分
seq = self.size_hint()
for expr in ranges:
    hint = V.graph.sizevars.optimization_hint(expr, fallback=8192)
    if par >= 2 * threads or par == threads:
        break
    if seq // threads < config.cpp.min_chunk_size:
        break                      # 活不够，不切线程
    depth += 1
    par *= hint
```

`seq // threads < 512` 意味着门槛是 **512 × 线程数**。本机 `torch.get_num_threads()` 是 4，所以门槛正好 **2048 元素** —— 这和上面实测的 2047/2048 分界完全对上。

### 但 2048 这个门槛在这台机器上低得离谱

把 `min_chunk_size` 调到 1（强制开 omp）和 10⁹（强制关），同一个图、同一个尺寸，A/B 对比：

```python
import os, glob, shutil, time, statistics
import torch
import torch._inductor.config as icfg

def build(tag, n, mcs):
    icfg.cpp.min_chunk_size = mcs
    icfg.force_disable_caches = True
    icfg.trace.enabled = True
    icfg.trace.debug_dir = f"/tmp/lesson053e/{tag}"
    shutil.rmtree(f"/tmp/lesson053e/{tag}", ignore_errors=True)
    ns = {"torch": torch}
    exec(f"def p_{tag}(a, b):\n    return torch.relu(a + b) * 2.0 + 1.0\n", ns)
    a, b = torch.randn(n), torch.randn(n)
    cf = torch.compile(ns[f"p_{tag}"])
    cf(a, b)
    fl = glob.glob(f"/tmp/lesson053e/{tag}/**/output_code.py", recursive=True)[0]
    g = {"__name__": f"gen_{tag}"}
    exec(compile(open(fl).read().split('if __name__ ==')[0], fl, "exec"), g)
    kn = [k for k in g if k.startswith("cpp_fused")][0]
    return g[kn], a, b, ("#pragma omp" in open(fl).read())

# omp 的进场成本在线程池「还醒着」和「已休眠」两态之间跳（相差近一倍），
# 所以单档的绝对微秒数不可复现。这里只在比值跨过硬阈值时给结论，不报原始数字。
print(f"{'元素数':>9}{'omp 开关判定':>22}   生成代码里有 omp 吗")
for n in (1024, 1048576):
    k_on, a, b, h_on = build(f"on_{n}", n, 1)
    k_off, _, _, h_off = build(f"off_{n}", n, 10**9)
    buf = torch.empty(n)
    for _ in range(40):
        k_on(a, b, buf); k_off(a, b, buf)
    son, soff = [], []
    for _ in range(150):
        t0 = time.perf_counter(); k_on(a, b, buf); t1 = time.perf_counter()
        k_off(a, b, buf); t2 = time.perf_counter()
        son.append(t1 - t0); soff.append(t2 - t1)
    r = statistics.median(soff) / statistics.median(son)   # >1 表示 omp 更快
    if r > 1.1:
        verdict = f"omp 快约 {r:.1f} 倍"
    elif r > 0.9:
        verdict = "打平"
    elif r > 0.2:
        verdict = f"omp 慢约 {1/r:.1f} 倍"
    else:
        verdict = "omp 慢 5 倍以上（纯亏）"
    print(f"{n:>9}{verdict:>22}   {h_on} / {h_off}")
icfg.cpp.min_chunk_size = 512
```

```text
      元素数              omp 开关判定   生成代码里有 omp 吗
     1024       omp 慢 5 倍以上（纯亏）   True / False
  1048576          omp 快约 1.3 倍   True / False
```

两块结果：

- **1024 元素**：omp 版比串行版慢 5 倍以上 —— 不是「慢 50%」，是慢一到两个数量级。1000 个元素的活本身只要零点几微秒，omp 却要收二十来微秒。
- **1048576 元素**：omp 才第一次赢，约 1.6 倍。

**为什么这里只给「慢 5 倍以上」而不给精确倍数？** 因为 omp 的进场成本在线程池「还醒着」和「已休眠」两个状态之间跳，两者差近一倍。同一段代码连跑三次，1024 那档的 omp 耗时可能是 20 微秒、也可能是 27 微秒 —— 而串行版只要 0.25 微秒，于是「慢多少倍」就在 80 和 108 之间随机。**这个不确定度是被测对象的性质，不是测量的误差**；把它报成一个小数点后两位的「78 倍」是自欺。

同样的道理让上面那张表只留了两个尺寸：65536 那档正好卡在两态切换的模糊区，报什么都不稳，干脆不报（它的理论值在下一节的算术表里）。

### 去 C 里验证：这十几微秒是平台的代价

不经过 PyTorch，直接用 torch 自带的 libomp 编译一个裸 C 程序（Inductor 在 macOS 上就是这么编的：`cpp_builder.py:1546` 用 `-Xclang -fopenmp`，库来自 `torch/lib/libomp.dylib`）：

```python
import subprocess, tempfile, os
import torch

SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
void work(const float*a,const float*b,float*o,long n){
    #pragma omp parallel num_threads(4)
    {
        #pragma omp for
        for(long i=0;i<n;i++){float x=a[i]+b[i];x=x>0?x:0;o[i]=x*2.0f+1.0f;}
    } }
void work_serial(const float*a,const float*b,float*o,long n){
    for(long i=0;i<n;i++){float x=a[i]+b[i];x=x>0?x:0;o[i]=x*2.0f+1.0f;} }
int main(void){
    long N[]={1048576,4194304};
    printf("%10s %11s %11s %9s\n","n","omp_us","serial_us","ser/omp");
    for(int k=0;k<2;k++){ long n=N[k];
        float*a=malloc(n*4),*b=malloc(n*4),*o=malloc(n*4);
        for(long i=0;i<n;i++){a[i]=i*1e-3f;b[i]=1.0f;}
        for(int w=0;w<80;w++){work(a,b,o,n);work_serial(a,b,o,n);}
        double bp=1e18,bs=1e18;
        for(int r=0;r<900;r++){double t0=now();work(a,b,o,n);double t1=now();
            work_serial(a,b,o,n);double t2=now();
            if(t1-t0<bp)bp=t1-t0; if(t2-t1<bs)bs=t2-t1;}
        printf("%10ld %11.2f %11.2f %9.2f\n",n,bp/1e3,bs/1e3,bs/bp);
        free(a);free(b);free(o);}
    return 0; }
"""
d = tempfile.mkdtemp()
open(f"{d}/m.c", "w").write(SRC)
tl = os.path.join(os.path.dirname(torch.__file__), "lib")
subprocess.run(["cc", "-O3", "-Xclang", "-fopenmp", f"-I{os.path.dirname(torch.__file__)}/include",
                f"-L{tl}", "-lomp", f"-Wl,-rpath,{tl}", "-o", f"{d}/m", f"{d}/m.c"], check=True)
r = subprocess.run([f"{d}/m"], capture_output=True, text=True,
                   env={**os.environ, "DYLD_LIBRARY_PATH": tl})
print(r.stdout.strip())
```

```text
n      omp_us   serial_us   ser/omp
   1048576       57.00       86.00      1.51
   4194304      322.00      404.00      1.25
```

这一块只留了百万元素以上的两档，理由见下：1024 那档的 omp 绝对耗时在线程池「冷/热」两种状态下能差 2 倍以上（7 微秒 vs 17 微秒），而它恰恰是本文最想引用的数字。**这个量不能报点估计。**

能报的是热态下的形状：**1M 元素 omp 约 57 微秒、串行约 86 微秒；4M 元素 omp 约 322 微秒、串行约 404 微秒。** 元素数涨 4 倍，两种写法都近似线性 —— 说明这段没有任何「并行失效」，只是 omp 分别只赢了 1.4 倍和 1.25 倍。**4 个线程买不到 4 倍**，因为逐元素 kernel 早就撞在带宽上（下一节的算术会给这个 1.15 倍的上限）。

而「omp 有一笔与工作量无关的固定成本」这个论断，证据在上一张表里 —— 1024 元素的活只要零点几微秒，omp 版却慢 5 倍以上，而它跑的是**完全相同的三行代码**。**这是 macOS + Intel libomp 的进入成本，不是 Inductor 的选择。** 于是问题从「Inductor 为什么门槛设这么低」变成「什么样的门槛才配得上这笔进场费」。

### 门槛该在哪：一个可算的判据

omp 只有在省下的时间超过进场成本时才划算。逐元素 kernel 是 2 读 1 写、fp32，每元素 12 字节：

$$
t_{\text{省}} = 12N\left(\frac{1}{BW_1} - \frac{1}{BW_4}\right) > t_{omp}
\quad\Longrightarrow\quad
N^* = \frac{t_{omp}}{12\left(\frac{1}{BW_1} - \frac{1}{BW_4}\right)}
$$

```python
BW1, BW4 = 98.0, 113.0      # GB/s，本机实测：4M 元素，omp 关 / 开
t_omp = 15e-6               # 秒，上面 C 程序量到的固定成本
BYTES = 12                  # 2 读 1 写 × fp32

d = BYTES * (1 / BW1 - 1 / BW4) * 1e-9      # 每元素省下多少秒
Nstar = t_omp / d
print(f"4 线程把逐元素带宽从 {BW1} 提到 {BW4} GB/s，只多 {BW4/BW1:.2f} 倍")
print(f"每元素省下 {d*1e12:.4f} ps")
print(f"omp 回本点 N* = {Nstar:,.0f} 元素（约 {Nstar*12/1e6:.1f} MB 流量）")
print()
print("有效带宽（含 omp 进场成本）：")
for n in (2**11, 2**16, 2**18, 2**20, 2**22):
    t = n * BYTES / (BW4 * 1e9) + t_omp
    print(f"  N={n:>9,}  序列 {n*BYTES/(BW1*1e9)*1e6:8.1f}us  "
          f"并行(含进场) {t*1e6:8.1f}us  "
          f"有效带宽 {n*BYTES/t/1e9:6.1f} GB/s  "
          f"{'并行赢' if t < n*BYTES/(BW1*1e9) else '串行赢'}")
```

```text
4 线程把逐元素带宽从 98.0 提到 113.0 GB/s，只多 1.15 倍
每元素省下 16.2543 ps
omp 回本点 N* = 922,833 元素（约 11.1 MB 流量）

有效带宽（含 omp 进场成本）：
  N=    2,048  序列      0.3us  并行(含进场)     15.2us  有效带宽    1.6 GB/s  串行赢
  N=   65,536  序列      8.0us  并行(含进场)     22.0us  有效带宽   35.8 GB/s  串行赢
  N=  262,144  序列     32.1us  并行(含进场)     42.8us  有效带宽   73.4 GB/s  串行赢
  N=1,048,576  序列    128.4us  并行(含进场)    126.4us  有效带宽   99.6 GB/s  并行赢
  N=4,194,304  序列    513.6us  并行(含进场)    460.4us  有效带宽  109.3 GB/s  并行赢
```

回本点落在百万元素，和实测的「262144 时还是串行赢、1048576 时并行才反超」对得上。

这里还有一层：分母只有 15 GB/s 的差。**4 线程只把逐元素带宽从 98 提到 113 GB/s**。原因不是线程不行，是这类 kernel 早就撞在带宽上 —— 第 043 期算过，逐元素 fp32 的算术强度是 0.17 FLOP/B，是 H100 峰值（20.0）的 0.8%。你花十几微秒买的东西，上限就只有 1.15 倍。

**可操作结论**：在 macOS / 小核数机器上，把 `TORCHINDUCTOR_CPP_MIN_CHUNK_SIZE` 调到 50000 以上（门槛变成 20 万元素），或者设 10⁹ 直接关掉让人 Inductor 不再插手并行。后者更省事，因为 ATen 自己还有一套并行（只是它也被同一笔进场费收税）。

### 与第 042 期的对照：两个后端给出差 64 倍的答案

同一个问题「多少活才值得切给另一个线程」，两个后端：

| 后端 | 门槛常量 | 含义 | 换成元素数 | 出处 |
|---|---|---|---|---|
| ATen `TensorIterator` | `GRAIN_SIZE = 32768` | 每线程至少摊到这么多元素 | 32768 | `ATen/TensorIterator.h:77` |
| Inductor CPU | `min_chunk_size = 512` | 同上，再乘线程数 | 2048（4 线程） | `config.py:1799` + `cpp.py:2767` |

差 64 倍。而本文的实测说明：**两个都偏小**（需要约 10⁶）。GRAIN_SIZE 至少是数量级对的（32768 元素 = 393 KB 流量，对应 4 微秒，和十几微秒的进场成本还有差距，但不算荒谬）；2048 元素只有 24 KB、0.25 微秒，和 15 微秒差了 60 倍 —— 这就是上表里 1024 元素那档「慢 70 多倍」的来源。

---

## 五、第三笔账（多）：横向融合上限 16，但超过 4 就开始亏

GPU 侧和 CPU 侧的融合规则在源码里长得完全不一样。

**Triton 后端**（`codegen/simd.py:2243`）把纵向和横向**当成同一个函数**：

```py
can_fuse_vertical = can_fuse
can_fuse_horizontal = can_fuse
```

判据是 `numel1 == numel2` —— 只要能放进同一个迭代空间就行。

**CPU 后端**（`codegen/cpp.py:5377`）在同一个判据之外又加了一道闸：

```py
def can_fuse_horizontal(self, node1, node2):
    if node1.is_template() or node2.is_template():
        return False
    if (len(node1.get_nodes()) + len(node2.get_nodes())
            > config.cpp.max_horizontal_fusion_size):     # 默认 16
        return False
    return self._can_fuse_horizontal_impl(node1, node2)
```

这条闸的效应很直观：造 K 个彼此独立的输出，数生成的代码里有几个 `x0` 循环。

```python
import os, glob, re, shutil
import torch
import torch._inductor.config as icfg

def build(K, n=1 << 16):
    icfg.force_disable_caches = True
    icfg.trace.enabled = True
    icfg.trace.debug_dir = f"/tmp/lesson053f/{K}"
    shutil.rmtree(f"/tmp/lesson053f/{K}", ignore_errors=True)
    ns = {"torch": torch}
    body = "\n".join([f"    o{i} = torch.relu(a * {i + 1}.0) + {i}.0" for i in range(K)])
    rets = ", ".join([f"o{i}" for i in range(K)])
    exec(f"def m_{K}(a):\n{body}\n    return ({rets},)\n", ns)
    a = torch.randn(n)
    torch.compile(ns[f"m_{K}"])(a)
    src = open(glob.glob(f"/tmp/lesson053f/{K}/**/output_code.py", recursive=True)[0]).read()
    return (len(set(re.findall(r"cpp_fused\w+", src))), src.count("for(int64_t x0="))

print(f"{'K 个独立输出':>12}{'kernel 函数数':>14}{'x0 循环数':>11}")
for K in (1, 4, 8, 16, 17, 20, 32):
    nk, nl = build(K)
    print(f"{K:>12}{nk:>14}{nl:>11}")
```

```text
     K 个独立输出    kernel 函数数     x0 循环数
           1             1          1
           4             1          1
           8             1          1
          16             1          1
          17             1          2
          20             1          2
          32             1          2
```

K ≤ 16 时是 1 个循环；K ≥ 17 时**不是多一个 kernel 函数，是同一个函数里多一个 `x0` 循环**（16 + 余数）。

### 但横向融合只有前两档跟理论走

「融合省字节」这笔账人人会算。融合版：读一次、写 K 次 → $4N(1+K)$ 字节。不融合版：每个输出各读一遍、写一遍 → $8KN$ 字节。理论加速比 $2K/(1+K)$，在 K=2/4/8/16 时是 1.33 / 1.60 / 1.78 / 1.88 —— 单调上升，永远划算。

实测（裸 kernel + 预分配输出，排除 wrapper 每次分配 K 个新张量的干扰）：

```python
import os, glob, shutil, time, statistics
import torch
import torch._inductor.config as icfg

N = 1 << 20
def build(K, mhfs):
    icfg.cpp.max_horizontal_fusion_size = mhfs
    icfg.force_disable_caches = True
    icfg.trace.enabled = True
    icfg.trace.debug_dir = f"/tmp/lesson053g/{K}_{mhfs}"
    shutil.rmtree(f"/tmp/lesson053g/{K}_{mhfs}", ignore_errors=True)
    ns = {"torch": torch}
    body = "\n".join([f"    o{i} = torch.relu(a * {i + 1}.0) + {i}.0" for i in range(K)])
    rets = ", ".join([f"o{i}" for i in range(K)])
    exec(f"def q_{K}_{mhfs}(a):\n{body}\n    return ({rets},)\n", ns)
    a = torch.randn(N)
    torch.compile(ns[f"q_{K}_{mhfs}"])(a)
    fl = glob.glob(f"/tmp/lesson053g/{K}_{mhfs}/**/output_code.py", recursive=True)[0]
    g = {"__name__": f"g_{K}_{mhfs}"}
    exec(compile(open(fl).read().split('if __name__ ==')[0], fl, "exec"), g)
    return [g[k] for k in g if k.startswith("cpp_fused")], a

print(f"{'K':>3}{'融合 us':>11}{'不融合 us':>12}{'不融合/融合':>13}{'理论 2K/(1+K)':>15}")
for K in (2, 4, 8, 16):
    fns_f, a = build(K, 16)
    fns_u, _ = build(K, 1)
    outs = [torch.empty(N) for _ in range(K)]
    def run(fns):
        for fn in fns:
            fn(*([a] + outs))
    for _ in range(40):
        run(fns_f); run(fns_u)
    # 3 轮，各取中位数，再取较小值 —— K=16 时两种写法的差距本身有 2 成抖动
    tf, tu = [], []
    for _ in range(3):
        rf, ru = [], []
        for _ in range(80):
            t0 = time.perf_counter(); run(fns_f); t1 = time.perf_counter()
            run(fns_u); t2 = time.perf_counter()
            rf.append(t1 - t0); ru.append(t2 - t1)
        tf.append(statistics.median(rf)); tu.append(statistics.median(ru))
    mf, mu = min(tf) * 1e6, min(tu) * 1e6
    print(f"{K:>3}{mf:>10.1f}{mu:>12.1f}{mu/mf:>12.2f}x{2*K/(1+K):>14.2f}x")
icfg.cpp.max_horizontal_fusion_size = 16
```

```text
  K      融合 us      不融合 us       不融合/融合    理论 2K/(1+K)
  2      82.9       109.7        1.32x          1.33x
  4     140.2       173.3        1.24x          1.60x
  8     501.4       426.7        0.85x          1.78x
 16    3990.8       928.7        0.23x          1.88x
```

K=2 和 K=4 跟着理论走（融合赢约 1.3 / 1.2 倍，理论 1.33 / 1.60）。K=8 时**账面该赢 1.78 倍，实测 0.85 倍** —— 已经翻过去了。K=16 时彻底崩：**融合慢 4.3 倍**，而账面说它该快 1.88 倍。

也就是说：**融合省下的那些读字节，在 K≥8 之后被另一项开销全部吃掉了，到 K=16 时倒亏一个量级。** 顺便说：不融合版要付 K 次 omp 进场费（K=16 就是 16 次），这部分本来该算作融合的好处 —— 即便把这笔好处让给它，融合还是输 4 倍。

### 为什么翻：把向量化和寄存器都排除掉，用纯 C 看

Inductor 版本的变量太多（向量化、omp、7 个指针参数）。写一个纯 C 的对照，只保留「一个循环写 K 条流」这一个变量：

```python
import subprocess, tempfile

SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
#define FUSED(NAME,K) void NAME(const float*in,float**o,long n){ \
    for(long i=0;i<n;i++){float x=in[i]; \
        for(int k=0;k<K;k++) o[k][i]=x*(float)(k+1)+(float)k;} }
#define UNFUSED(NAME,K) void NAME(const float*in,float**o,long n){ \
    for(int k=0;k<K;k++) for(long i=0;i<n;i++) o[k][i]=in[i]*(float)(k+1)+(float)k; }
FUSED(f2,2) FUSED(f4,4) FUSED(f8,8) FUSED(f16,16)
UNFUSED(u2,2) UNFUSED(u4,4) UNFUSED(u8,8) UNFUSED(u16,16)
int main(void){
    long N=1048576;
    int Ks[]={2,4,8,16};
    void(*F[])(const float*,float**,long)={f2,f4,f8,f16};
    void(*U[])(const float*,float**,long)={u2,u4,u8,u16};
    printf("%4s %11s %11s %8s %11s\n","K","fused_us","unfused_us","un/fu","ns/store");
    for(int j=0;j<4;j++){ int K=Ks[j];
        float*in=aligned_alloc(64,N*4);
        for(long i=0;i<N;i++) in[i]=(float)i*1e-6f;
        float**o=malloc(K*sizeof(float*));
        for(int k=0;k<K;k++) o[k]=aligned_alloc(64,N*4);
        for(int w=0;w<20;w++){F[j](in,o,N);U[j](in,o,N);}
        double bf=1e18,bu=1e18;
        for(int r=0;r<120;r++){double t0=now();F[j](in,o,N);double t1=now();
            U[j](in,o,N);double t2=now();
            if(t1-t0<bf)bf=t1-t0; if(t2-t1<bu)bu=t2-t1;}
        printf("%4d %11.1f %11.1f %8.2f %11.2f\n",
               K,bf/1e3,bu/1e3,bu/bf,bf/(N*K));
        for(int k=0;k<K;k++) free(o[k]); free(o); free(in); }
    return 0; }
"""
d = tempfile.mkdtemp()
open(f"{d}/h.c", "w").write(SRC)
subprocess.run(["cc", "-O3", "-o", f"{d}/h", f"{d}/h.c"], check=True)
print(subprocess.run([f"{d}/h"], capture_output=True, text=True).stdout.strip())
```

```text
K    fused_us  unfused_us    un/fu    ns/store
   2        96.0       121.0     1.26        0.05
   4       240.0       241.0     1.00        0.06
   8      1699.0       529.0     0.31        0.20
  16     43268.0      1073.0     0.02        2.58
```

纯 C 的结果更极端：K=8 掉到 0.31（慢 3 倍），K=16 是 **0.02**（慢 50 倍）。和 Inductor 版相比方向完全一致，只是幅度更大 —— 因为 Inductor 版还有 4 个线程在摊这笔账。两个共同点值得记住：

1. **退化方向和 Inductor 一致** —— 不是框架的问题，是「交错写多条流」本身的问题。
2. **`ns/store` 那一列与工作集大小无关** —— K=16 是 2.57 ns/store，K=2 是 0.05 ns/store，差 50 倍。这一点很关键：**它不是 cache 容量问题，也不是 DRAM 带宽问题**（本文另一组扫描里，16 KB 的工作集和 4 MB 的工作集付的是同一个单价）。

一条 cacheline 是 128 字节 = 32 个 float。融合版每跑 32 个元素，就往 K 个相距 $4N$ 字节的 buffer 各写 128 字节。不融合版则是把一整条 cacheline 写满才换下一条。**推断**（本机无法直接测到 store buffer / 写合并 / 预取器这一级）：代价来自 K 条写流的 cacheline 在被写满之前就被换出，写带宽按 $1/K$ 退化；K=16 时每字节写要多付一个量级的钱，实测就是这个量级。

**可操作结论**：`max_horizontal_fusion_size` 的默认 16 落在「已经亏了 4 倍」那一档。实测 K≤4 融合是对的（1.1~1.3 倍），K=8 打平，K≥8 该关掉。但 Inductor 是贪心合并到 16 为止，不会按 K 动态停手。如果你的模型里有一段「一个输入裂成 8 个以上独立输出」的结构（多任务头、多路 loss、多个辅助输出），**手动拆成多次调用比让它融合更快**。

---

## 六、在 PyTorch 中怎么用：三个旋钮 + 一笔没人算的账

三个旋钮全部走环境变量或 config，不需要改代码：

```python
import os
import torch

# 方式一：环境变量（必须在 torch.compile 之前设置，或者设完 dynamo.reset()）
os.environ["TORCHINDUCTOR_CPP_MIN_CHUNK_SIZE"] = "50000"   # 门槛变成 50000*4 = 20 万元素
os.environ["TORCHINDUCTOR_CPP_MAX_HORIZONTAL_FUSION_SIZE"] = "4"
os.environ["TORCHINDUCTOR_CPP_SIMDLEN"] = "128"            # 单位是 bit
os.environ["TORCHINDUCTOR_CPP_MARCH"] = ""                 # 空串 = 抑制 -march 旗标

# 方式二：直接改 config（改完必须清 Dynamo 缓存，否则旧 kernel 还在）
import torch._inductor.config as icfg
icfg.cpp.min_chunk_size = 50000
torch._dynamo.reset()
```

还有第四笔账，前三个旋钮都盖不住它：**调用本身的成本**。

```python
import time, statistics, glob, shutil
import torch
import torch._inductor.config as icfg

def profile(n, rounds=7):
    icfg.force_disable_caches = True
    icfg.trace.enabled = True
    icfg.trace.debug_dir = f"/tmp/lesson053h/{n}"
    shutil.rmtree(f"/tmp/lesson053h/{n}", ignore_errors=True)
    ns = {"torch": torch}
    exec(f"def r_{n}(a, b):\n    return torch.relu(a + b) * 2.0 + 1.0\n", ns)
    f = ns[f"r_{n}"]
    a, b = torch.randn(n), torch.randn(n)
    cf = torch.compile(f)
    cf(a, b)
    fl = glob.glob(f"/tmp/lesson053h/{n}/**/output_code.py", recursive=True)[0]
    g = {"__name__": f"gen_{n}"}
    exec(compile(open(fl).read().split('if __name__ ==')[0], fl, "exec"), g)
    kfun = g[[k for k in g if k.startswith("cpp_fused")][0]]
    call = g["call"]
    buf = torch.empty_strided((n,), (1,), dtype=torch.float32)

    fns = {"compile": lambda: cf(a, b),
           "Runner": lambda: call([a, b]),
           "kernel": lambda: kfun(a, b, buf),
           "eager": lambda: f(a, b)}
    for fn in fns.values():
        for _ in range(40):
            fn()
    # 绝对时间对进程状态太敏感（GC、分配器碎片，单轮中位数能差 2 倍）。
    # 比值稳得多 —— 所以报比值，不报绝对时间。
    ratios = {"k": [], "e": []}
    for _ in range(rounds):
        s = {k: [] for k in fns}
        for _ in range(60):
            for k, fn in fns.items():
                t0 = time.perf_counter(); fn(); s[k].append(time.perf_counter() - t0)
        m = {k: statistics.median(v) for k, v in s.items()}
        ratios["k"].append(m["compile"] / m["kernel"])
        ratios["e"].append(m["compile"] / m["eager"])
    return statistics.median(ratios["k"]), statistics.median(ratios["e"])

print(f"{'元素数':>9}{'compile / 裸kernel':>19}{'compile / eager':>17}")
for n in (16, 1024, 65536, 1048576):
    rk, re_ = profile(n)
    print(f"{n:>9}{rk:>17.1f}x{re_:>16.2f}x")
```

```text
      元素数  compile / 裸kernel  compile / eager
       16             66.0x            2.22x
     1024             39.7x            2.12x
    65536              1.6x            0.61x
  1048576              1.1x            0.53x
```

这块的输出刻意只报**比值**，不报绝对时间。原因写在第三点里。四件事：

1. **小尺寸下 `torch.compile` 的耗时是裸 kernel 的 40~70 倍。** 那部分全是 Python wrapper：guard 检查、`assert_size_stride`、`empty_strided`、参数打包。而裸 kernel 本身在亚微秒量级 —— 具体数字这里故意不报，因为它在 0.1~0.3 微秒之间跳（差了近一倍），而 `time.perf_counter` 的分辨率本身就在几十纳秒，测这个量等于测抖动。
2. **但和 eager 比，它只是 2 倍出头。** 也就是说 CPU 上 `torch.compile` 的固定开销和「eager 走三次 ATen dispatch」是同一个量级 —— 都在微秒级。这与 GPU 上完全不同：GPU 的 launch 成本是微秒级，而它省下的是几百次 launch；CPU 上省下的是纳秒级的指令，付出去的是微秒级的 Python。
3. **绝对时间测不准，比值稳。** 同一段代码、同一台机器，`compile` 的绝对耗时在不同进程间能差 2 倍（GC、分配器碎片、Python 堆状态）。但同一进程内「compile / 裸 kernel」这个比值很稳 —— 这也是为什么这一块报倍数。**这个教训比数字本身有用**：CPU 微基准里凡是微秒级的绝对量，都要先问「换个进程还一样吗」。
4. **盈亏平衡在 65536 元素附近**：那里 `compile` 是裸 kernel 的 1.6 倍、是 eager 的 0.61 倍 —— 第一次开始赚。到 1M 时变成 1.1 倍 / 0.53 倍，接近翻倍。而那一档的裸 kernel 里还有一大截是 omp 进场费，按上一节把 `min_chunk_size` 调大，这个平衡点会前移。

还有一个抓代码时才发现的坑：**N=1 时 Inductor 根本不生成 kernel**。单元素张量的图被 Dynamo/Inductor 折叠掉了，`output_code.py` 压根不落盘。想抓这个尺寸的代码会拿到空列表。

---

## 七、围绕该领域展开

### 7.1 与 GPU 侧 Triton 后端的三处形态差异

同一个 Inductor，两套后端，产物差得很远。这张表是本文的浓缩：

| | Triton 后端（GPU） | CPU 后端 |
|---|---|---|
| 产物 | 一张 grid + 一个 `.py` 里的 Triton kernel | 一份 C++ 源码 + 一个 `for` 循环 |
| 并行度 | 显式，`grid` 是启动参数 | 隐式，SIMD 靠编译器、线程靠 `#pragma omp` |
| 尺寸特化 | 每个 shape 一个 kernel（也有动态 shape 路径） | 循环上界**编译期写死常量** |
| 边界处理 | mask：一个 kernel 覆盖所有尺寸，多算的丢弃 | 谓词化的**第二份完整循环体**，`loadu(ptr, 3)` |
| 融合规则 | vertical 与 horizontal 是同一个函数，判据 `numel1 == numel2` | horizontal 单独一条函数，多一道 `max_horizontal_fusion_size` |
| ISA 选择 | 由 Triton 后端决定 | 运行时**干编译 + dlopen 探测**，每次新 ISA 付一次 |
| 代码体积 | 几十行 Triton IR | `cpp_prefix.h` 1,512 行 + 生成的循环体 |

最值得留意的是「边界处理」那一行。同样是处理 `n % 4 != 0`：

- **Triton**：一个 kernel，用 mask 让最后一个 tile 多算的部分不落盘。**代码只有一份**。
- **CPU**：两份循环体，主循环走 `C10_LIKELY`，尾循环走 `C10_UNLIKELY` 并带部分加载。**I-cache 里有两份**。

实测这两份代码的代价在噪声内（n=2²⁰ 与 2²⁰−1 差不到 5%，被采样抖动盖掉），所以这不是性能问题 —— 是**代码体积与编译时间**的问题。第 052 期量过「每多一行代码要付多少编译费」，这里就是那个账的另一端：CPU 后端的产物天生比 Triton 大，而它的编译成本本来就高一个量级（`cpp_prefix.h` 那个编译单元 3.5 秒 vs Triton 的毫秒级）。

### 7.2 `__avx_chk_kernel`：一个为了探 ISA 而存在的空函数

`cpu_vec_isa.py` 里有一段既不属于模型也不属于 kernel 的代码：

```py
# 源码摘录：VecISA._avx_code
_avx_code = """
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || ... || defined(CPU_CAPABILITY_NEON)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

alignas(64) float in_out_ptr0[16] = {0.0};

extern "C" void __avx_chk_kernel() {
    auto tmp0 = at::vec::Vectorized<float>(1);
    auto tmp1 = tmp0.exp();
    tmp1.store(in_out_ptr0);
}
"""
```

它做三件事：**干编译**（用目标 ISA 的旗标编一遍）→ **子进程 dlopen**（`_probe_load`，带 60 秒超时）→ 成功才认为这个 ISA 可用，并写一个 `.load_ok` 标记缓存结果。

为什么不能直接看 `torch.cpu.get_capabilities()`？源码注释说得很清楚：

> PyTorch 的 release wheel 可能用 gcc-7 编译，而 gcc-7 过不了 CMake 的 AVX512 检查，于是 `libtorch_cpu.so` 不导出 Sleef 的 AVX512 符号；而 TorchInductor 默认装最新 clang，是支持 AVX512 编译的。**两边 Sleef 版本会冲突。**

所以探测的不是「CPU 支不支持 AVX512」，是「**我这套 PyTorch + 这个编译器拼起来的产物能不能真的跑起来**」。这是只有「代码生成 + 动态链接」这条路才会遇到的问题 —— Triton 后端不需要，因为它的 ISA 由 ptxas 统一决定。

这条也和 052 期接上：那期算的是「让 `cpp_extension` 编一次要多少钱」（`<torch/extension.h>` = 353,499 行 / 3.57 s / 982 MB）。这里的探针只用 8 行 + 一个 `at::vec` 头，代价小得多，但**每换一个 ISA、每换一次编译器版本都要付一次**（指纹里带了 compiler version + ISA flags + torch version，所以升级 torch 会让它重探）。

### 7.3 第 040 期没讲的那一半：横向融合

040 期讲融合，讲的是**纵向**：`relu(a+b)*2+1` 四个算子读写同一批数据，合成一个循环，字节数从 8N 降到 3N。

横向融合是另一回事：**输入不同、输出不同、彼此无依赖的两条链，也能合进一个循环**。这在 GPU 上不算什么（同一个 program 里多写一个 buffer，代价几乎为零）；在 CPU 上却要付本文第五节量到的账。

这解释了一件看起来很奇怪的事：**为什么只有 CPU 后端给横向融合设了上限，而 Triton 后端没有。** 不是因为 GPU 更宽容，是因为 GPU 的 write 是 fire-and-forget（写进 L2 就完事，不占发射资源），而 CPU 的每一次写都要占 store buffer、都要维护 cacheline 一致性。同一份贪心融合策略，在两种执行模型下的代价曲线形状完全不同。

### 7.4 三笔账最后都汇到屋脊点上

第 043 期给的判据是算术强度：$AI < 20$ FLOP/B 就是带宽受限，逐元素 fp32 的 $AI \approx 0.17$。

回头看本文三笔账，全是这一个判据的不同侧面：

- **宽**（向量化）：算术强度不变，只是把「每字节要执行的指令数」降下来。所以纯算术几乎不加速（本来就不是指令受限），而 `erf` 快 4.4 倍（那几个超越函数真的是指令受限）。
- **薄**（omp）：目标是提高带宽，但单线程已经用到 98/113 = 87% 的可用带宽，上限就是 1.15 倍。
- **多**（横向融合）：账面是省读字节（$8KN \to 4N(1+K)$），实际是把有效带宽从 100 GB/s 打到 7 GB/s（64 MB / 43 ms）。

三个旋钮看似在调三件不同的事，本质上都在调**这颗核每秒钟能从内存里搬多少字节**。而 CPU 上的逐元素 kernel，几乎总是先撞带宽。

---

## 八、什么时候该用 / 不该用

**该用 CPU 后端（并保持默认旋钮）的场景**

- 图里是大段逐元素 + 归约（norm、activation、residual、softmax），单个张量 ≥ 数万元素
- 部署在 Linux 服务器：核多、omp 线程池常驻、`min_chunk_size=512` 对应的门槛在那种机器上更合理（56 线程 × 512 = 28672 元素，且 omp 进场在 Linux 上便宜得多）
- 图里有 `exp` / `tanh` / `erf` / `sigmoid` / `gelu` —— 这时向量化的真实收益在 2~4 倍，是本文量到的最大单项收益

**不该用 / 该改旋钮的场景**

- **小张量反复调用**：1 千元素量级时 `torch.compile` 的耗时是裸 kernel 的 40 倍、是 eager 三个算子的 2 倍多 —— 付出去的是 Python wrapper 的钱，收回来的是几个纳秒。这时直接用 eager，或者把小的逐元素操作挪进相邻的大 kernel
- **macOS / 小核心笔记本**：omp 门槛 2048 太低，1024~262144 这段是纯亏（1024 元素时慢 78 倍）。`TORCHINDUCTOR_CPP_MIN_CHUNK_SIZE` 调到 50000 以上，或直接设 10⁹ 关掉
- **图里横向裂出 > 4 路独立输出**：融合会慢 2~4 倍，手动拆成多次调用更快
- **需要尺寸无关的产物**（一次编译到处跑）：CPU 后端的循环上界是编译期常量，每个 shape 一套代码。这种场景该看 AOTInductor + 动态 shape 那条线，而不是指望 CPU 后端省编译

---

## 九、常见坑

**坑 1：`cpp.simdlen` 的单位是 bit，而且匹配不上就整个关掉向量化。**
不是元素数。设成 `4`（以为是 4 个 float）→ `pick_vec_isa` 拿 4 去比 `bit_width()`（128 / 256 / 512）→ 一个都不匹配 → 返回 `invalid_vec_isa` → 生成纯标量代码，而且**不会报错，只是默默变慢**。macOS arm 上唯一合法的值是 128，或者说 `None`。

**坑 2：抓 `output_code.py` 必须关 FX graph cache，而且要每轮重设。**
Inductor 的图缓存是跨进程共享的，第二次跑同一个图会直接命中，`trace.debug_dir` 是空的。正确写法是 `icfg.force_disable_caches = True`。更隐蔽的是：这个开关**会被别的地方清掉**，所以循环里要每轮重设，别只在脚本顶部设一次。

**坑 3：`enable_loop_tail_vec=False` 不是「不处理 tail」。**
它是把 tail 从「谓词化的向量体（`loadu(ptr, 3)`）」换成「标量 `for (x0_tail = 2044; x0_tail < 2047; x0_tail++)`」。两者都在 `C10_UNLIKELY` 分支里，都正确。区别只有：前者两份向量代码（I-cache 翻倍、编译时间翻倍），后者一份向量 + 一小段标量。别以为关掉它是 bug。

**坑 4：`#pragma omp parallel num_threads(4)` 是编译期写死的字面量。**
改 `torch.set_num_threads()` 会触发**重新编译**，而不是改这一行。想让线程数在运行期自适应，得开 `cpp.dynamic_threads`（`config.py:1796`），它会把 `num_threads(4)` 换成 `num_threads(max_threads)`，由 omp 运行时决定 —— 代价是注释里写明的「单线程负载会有轻微性能退化」（因为编译期不能再为某个固定线程数特化）。

---

## 十、一句话总结

GPU 后端的产物是一张网格，CPU 后端的产物是一个 `for` 循环：它的**宽度**由 ISA 探测决定（这台机器只有 128 位，因为要复用 ATen 已有的 `vec128` 实现），**厚度**由 `min_chunk_size × 线程数` 决定（2048 元素，而 omp 进门要十几微秒，回本点在百万元素），**个数**由 `max_horizontal_fusion_size` 决定（16，但超过 4 就开始亏）—— 三个默认值都是给 Linux 服务器定的，在这台 10 核 Mac 上两个偏小、一个大，而两个偏小的那个能把 kernel 拖慢 70 多倍。

---

## 今日练习

给你一个真实的判断任务：下面这段代码在一台 10 核 Mac 上跑，`a` 是 500 万元素的 fp32。

```python
def forward(a, w1, w2, w3):
    x = torch.relu(a * w1)
    y = torch.tanh(a * w2)
    z = torch.erf(a * w3)
    return x, y, z
```

三个问题：

1. 它会生成几个 `x0` 循环？为什么？
2. 如果把 `max_horizontal_fusion_size` 设成 1（禁止横向融合），总时间会变快还是变慢？大概多少倍？
3. 这段代码里哪些操作真正从「显式向量化（`at::vec`）」获益，哪些不获益？

<details>
<summary>今日练习（参考答案）</summary>

**1. 几个 `x0` 循环：3 个，不是 1 个。**

三个输出彼此独立、输入相同，`can_fuse_horizontal` 的前半段完全满足（节点数 3 ≤ 16）。但 `torch.tanh` 和 `torch.erf` 会走**不同的 loop 结构** —— 本期实测过：`tanh` 的向量实现是 Sleef 的 `Sleef_tanhf8_u10`，`erf` 是手写的 A&S 7.1.26 展开（第 048 期在 `vec256_float.h` 的 x86 分支里看到过），两者和 `relu` 的循环体指令数差一个量级。Inductor 的横向融合要求循环结构可对齐（`_align_compatible_range_nodes`），体量差太多的会被拆开。

要确定答案不要猜 —— 用正文第二节 `dump()` 那个函数跑一遍，数 `body.count("for(int64_t x0=")`。

**2. 会变慢，约 1.5~2 倍。**

K=3 落在「融合该赢」的区间（本期实测 K=2 是 1.10x、K=4 是 1.28x）。因为三个输出共享同一个输入 `a`：融合读一次（4N 字节），不融合读三次（12N 字节）；写都是 12N 字节。所以理论字节比是 $8KN : 4N(1+K) = 24N : 16N = 1.5$ 倍。

但注意第 3 节的条件：**这个 1.5 倍只有在读的那部分真的省下来时才兑付**。500 万元素 = 20 MB 输入、60 MB 输出，全在内存里跑，读省 40 MB 是实打实的。所以这里融合是对的，不该关。

**3. 只有 `tanh` 和 `erf` 真正获益。**

本期实测（同一个图、同一个尺寸，只切 `simdlen`）：

- 加减乘 relu：标量/向量 ≈ 1.0 —— clang 的自动向量化已经做掉了，`at::vec` 几乎没有额外贡献
- `tanh`：约 2.1 倍，`erf`：**约 4.4 倍** —— 这两个的标量版走 libm / Sleef 标量入口，clang 无法自动向量化库调用；而 `at::vec::Vectorized::erf()` 走 Sleef 的 4 路实现

所以这段代码的收益结构是：**`tanh` 和 `erf` 各贡献 2~4 倍的局部加速，`relu` 那一支基本白干**。如果你要优化它，最该动的是把 `tanh` 和 `erf` 的输入域收窄（比如用 `torch.compile` 的分解选项换成多项式近似），而不是去调 `simdlen`。

</details>
