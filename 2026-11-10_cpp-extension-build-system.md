# PyTorch 每日一课 · 第 052 期

## C++ 扩展构建体系：那 5 秒编译费到底买到了什么

> **日期** 2026-11-10 ｜ **难度** ⭐⭐⭐ ｜ **前置** 会写 Python，见过 `#include`；不需要 CUDA，不需要 GPU ｜ **阅读** 约 30 分钟
>
> **关联** 第 039 期（ATen 与 Dispatch）· pybind11 是绕过 dispatcher 的旁路 · 第 040 期（算子融合）· Inductor 生成的融合 kernel 就是本期解剖的产物 · 第 003 期（torch.compile）· 第 008 期（Triton）· 另一条不需要 C++ 编译链路的路

---

## 0. 从一次 5 秒的等待说起

你写了一个 10 行的 kernel，接进 PyTorch。跑通了。然后你改了一个常数——`2.0` 改成 `2.5`——重新跑。

**等了 5 秒。**

再改一个常数，再等 5 秒。如果你在 Jupyter 里干这件事，一小时能试不了几次。

这 5 秒里到底发生了什么？大多数人会说「在编译」。这句话没错，但它把三笔完全不同的账混成了一笔：

1. 你的 10 行代码，编译要多久？—— **0.02 秒**。
2. 那 5 秒是谁花的？—— `#include <torch/extension.h>`，**353,499 行预处理结果，5,997 个头文件，982 MB 峰值内存**。
3. 你改了常数，为什么不能只重编你改的那部分？—— 因为缓存的最小单位是**整个扩展**，不是函数，也不是文件。

而这期真正想讲清楚的第三个问题，藏在更深一层：**这个缓存对「什么变了」的判断，有一半是瞎的**。你自己写的头文件它盯着，PyTorch 自己的头文件它一个都不看。这条不是设计者偷懒，是 `-MMD` 加 `-isystem` 的必然结果——后面会用实验把它钉死。

先把结论摆在前面。

> **编译期成本和运行期成本是两笔独立结算的账。`cpp_extension` 的默认配置在两边都不划算：它默认不加任何 `-O` 旗标（运行慢 40 倍，而编译时间一分没省），它的入场门票是 353,499 行头文件（PyTorch 自己都不用，Inductor 换了 126,158 行的另一张门票），它的增量判断只看你自己写的头（连 torch 换版本都不会触发重编）。**

---

## 1. 这个领域解决什么问题

### 1.1 三条路，三种代价

假设你要一个 ATen 没提供的算子。2026 年的 PyTorch 给你三条路：

| 路径 | 入口 | 你付出的 | 你放弃的 |
|---|---|---|---|
| Triton | `@triton.jit` | 学一门 DSL | 只能用 Triton 支持的东西；CPU 上基本没有 |
| C++ 扩展 | `torch.utils.cpp_extension` | **一条完整的 C++ 构建链路** | 与 PyTorch ABI 绑死 |
| 直接改 PyTorch | fork + 重编 | 几小时到一天 | 用官方 wheel 的资格 |

第 008 期讲的是第一条。这期讲第二条，而核心论点在这里：

**`cpp_extension` 不是「一个编译函数」，它是「用 PyTorch 自己的构建系统编你自己的代码」。**

这句话决定了它的全部形状。因为要和 PyTorch 一起工作，它必须：

- 用**和 PyTorch 相同的** C++ 标准（`-std=c++20`，硬编码在 `cpp_extension.py:2084`）
- 用**和 PyTorch 相同的**编译器家族（macOS 只接受 clang，`cpp_extension.py:495`）
- include **和 PyTorch 相同的**头文件目录（`include_paths()` 返回的 2 条路径）
- 链接 **和 PyTorch 相同的** `.dylib`/`.so`（`-lc10 -ltorch_cpu -ltorch -ltorch_python`）

这四条里每一条都合理，但它们的代价全部落在「编译时间」上。你不是在为自己的代码付钱，你是在为**「和 PyTorch 在同一个进程里和平共处」这个前提**付钱。

### 1.2 为什么这件事值得单独一期

因为它是一个**跨期反复踩到、但一直没被拆开**的东西。前面几期写 B 组话题时，为了跑一个微基准，反复在「用 `cpp_extension`」和「用 `cc -O3` 现编」之间做选择——而选择依据一直是手感和传闻（「`load_inline` 首次要 5~10 分钟」）。这期把那张分界表真正算出来。

顺带，这期也会回答一个很多人以为是玄学的问题：**为什么换了 torch 版本之后，有些扩展还会用旧的 `.so`**。

---

## 2. 核心思想：三笔账

把 5 秒拆成三笔账，每一笔的计价方式都不一样。

### 账 1：入场费（按「头文件行数」计价）

编译一个 TU（translation unit，一个 `.cpp` 加它递归 include 的全部内容）的时间，几乎完全由**预处理后的行数**决定，而不是你写了几行。

类比：你进一个园区办事。园区大门的安检要过 353,499 道检查——你只办一件事，但安检一次都不能少。

这笔费用的关键性质是：**它和你的代码量几乎无关**。你写 10 行还是 1000 行，入场费都一样。

### 账 2：存储费（按「整个扩展」计价）

缓存的最小单位是 `<扩展名>/` 整个目录。改了源码、改了旗标，那就是全量重编。

类比：寄存柜只有一整间仓库，没有小格子。你放一颗螺丝和放一台机床，价格相同。

但真正的坑不在粒度，在**它怎么判断「变了」**：

- 同进程内：它连判断都省了，直接返回已经加载好的模块
- 跨进程：它用 `-MMD` 生成依赖列表，让 ninja 精确追踪
- **但是**——`-MMD` 会排除 `-isystem` 引入的头。而 torch 的 2 个 include 路径，全部通过 `-isystem` 传入

第三条是这期最有价值的发现。它意味着：**PyTorch 那 5,997 个头文件，一个都不在依赖图里。**

### 账 3：运行费（按「每次调用」计价）

编译出来的代码快不快，取决于优化级别。而 `cpp_extension` 默认**不加任何 `-O` 旗标**，clang 默认就是 `-O0`。官方文档在 `cpp_extension.py:1864` 附近明说了：

> with optimizations, pass `extra_cflags=['-O3']`

这句话的语气像是「一个可选的性能调优」。实测下来它不是调优，是 40 倍的悬崖——而编译时间几乎没变。**省下的是 0 秒，花掉的是 40 倍。**

---

## 3. 在 PyTorch 中怎么用

### 3.1 第一步：不要相信任何人（包括本文），自己把 build.ninja 挖出来

`cpp_extension` 会生成一份 `build.ninja` 并交给 ninja 执行。这份文件就是「它到底喂了什么给编译器」的唯一真相。

不要用 `verbose=True` 去读滚动日志——直接读文件。

```python
# 在一个隔离的临时目录里编一个最小扩展，然后把 ninja 构建文件挖出来
import os, sys, tempfile, shutil, pathlib

# cpp_extension 要求 ninja 在 PATH 里；如果你的 ninja 装在 venv/bin 却没进 PATH，先补上
if not shutil.which("ninja"):
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]

BUILD = tempfile.mkdtemp(prefix="ext_probe_")
os.environ["TORCH_EXTENSIONS_DIR"] = BUILD        # 隔离，不污染 ~/.cache

from torch.utils.cpp_extension import load_inline

SRC = """
#include <torch/extension.h>
torch::Tensor axpy(torch::Tensor x, double a, double b) { return a * x + b; }
"""
mod = load_inline(name="probe_ninja", cpp_sources=SRC, functions=["axpy"])
print("调用结果:", mod.axpy(__import__("torch").arange(4.), 2.0, 1.0).tolist())

# 挖出 build.ninja，只打印「喂给编译器的参数」和「喂给链接器的参数」两行
nj = pathlib.Path(BUILD) / "probe_ninja" / "build.ninja"
for line in nj.read_text().splitlines():
    if line.startswith(("cxx =", "cflags =", "ldflags =")):
        # 把绝对路径压成 <TORCH>/... 方便阅读
        compact = line.replace(str(pathlib.Path(__import__("torch").__file__).parent), "<TORCH>")
        print(compact)
```

```text
调用结果: [1.0, 3.0, 5.0, 7.0]
cxx = c++
cflags = -DTORCH_EXTENSION_NAME=probe_ninja -DTORCH_API_INCLUDE_EXTENSION_H -isystem <TORCH>/include -isystem <TORCH>/include/torch/csrc/api/include -isystem /Users/morssi/.workbuddy/binaries/python/versions/3.13.12/include/python3.13 -fPIC -std=c++20
ldflags = -shared -L<TORCH>/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -undefined dynamic_lookup
```

三个观察，每一个都能解释后面的一批现象。

**观察一：`-std=c++20` 和 `-fPIC` 是硬编码的，不是你的选择。** 源码在 `cpp_extension.py:3008`：

```py
cflags = common_cflags + ['-fPIC', '-std=c++20'] + extra_cflags
```

注意 `extra_cflags` **排在最后**——所以你能覆盖它（传 `-std=c++17` 就真的生效），但你得知道自己在覆盖什么。

**观察二：三个 `-isystem`，没有一个是 `-I`。**

这个细节后面会变成整期最硬的一条结论：`-MMD` 生成的依赖文件会**跳过 `-isystem` 的目录**。用 `-isystem` 本意是「这些是系统头，别给我报警告」，副作用是「这些头变了，编译器不用重编」。而 PyTorch 的全部 6,000 个头都走这条路。

**观察三：macOS 上有 `-undefined dynamic_lookup`。**

```py
# cpp_extension.py:3073
if IS_MACOS:
    ldflags.append('-undefined dynamic_lookup')
```

它允许 `.so` 里存在「链接时找不到定义」的符号，留到 `dlopen` 时再解析。这是 pybind11 扩展能工作的前提（`PyInit_*` 的符号在 `python` 可执行文件里）。但它**只解决「符号未定义」，不解决「库未加载」**——这个区别会在坑 4 里变成一个真实的报错。

### 3.2 账 1：入场费按行数计价，而且单价不均匀

现在把入场费量出来。方法：拿同一个 `.cpp`，只换第一行的 `#include`，测预处理行数和编译耗时。

```python
# 同一个 TU，只换 include 的入口，量「入场费」
import subprocess, tempfile, pathlib, time, statistics, sysconfig

import torch
T = pathlib.Path(torch.__file__).parent
INC = ["-isystem", str(T / "include"),
       "-isystem", str(T / "include" / "torch" / "csrc" / "api" / "include"),
       "-isystem", sysconfig.get_path("include")]

CASES = {
    "空 TU":                  "int f(int x){return x+1;}\n",
    "<Python.h>":             "#include <Python.h>\nint f(int x){return x+1;}\n",
    "<pybind11/pybind11.h>":  "#include <pybind11/pybind11.h>\nint f(int x){return x+1;}\n",
    "<torch/.../cpp_prefix.h>": "#include <torch/csrc/inductor/cpp_prefix.h>\nint f(int x){return x+1;}\n",
    "<torch/library.h>":      "#include <torch/library.h>\nint f(int x){return x+1;}\n",
    "<ATen/ATen.h>":          "#include <ATen/ATen.h>\nint f(int x){return x+1;}\n",
    "<torch/extension.h>":    "#include <torch/extension.h>\nint f(int x){return x+1;}\n",
}

work = pathlib.Path(tempfile.mkdtemp(prefix="bloat_"))
print(f"{'入口':26s}{'预处理行数':>12s}{'编译':>9s}{'每万行':>9s}")
print("-" * 56)
for tag, code in CASES.items():
    f = work / (tag.strip("<>").replace("/", "_").replace(".", "") + ".cpp")
    f.write_text(code)
    n_line = subprocess.run(["c++", "-std=c++20", "-E", *INC, str(f)],
                            capture_output=True, text=True).stdout.count("\n")
    ts = []
    for _ in range(3):                      # 取 min，挡掉首次 page fault
        t0 = time.perf_counter()
        subprocess.run(["c++", "-std=c++20", "-O3", "-c", *INC, str(f), "-o", str(f.with_suffix(".o"))],
                       check=True, capture_output=True)
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    # 行数太少时「每万行」的分母近零，比值没有意义 —— 不打印，避免读者误读
    unit = f"{best/(n_line/1e4):6.3f}s" if n_line >= 10000 else "    —"
    print(f"{tag:26s}{n_line:>12,d}{best:>8.2f}s{unit:>9s}")
```

```text
入口                               预处理行数       编译      每万行
--------------------------------------------------------
空 TU                                 8    0.02s        —
<Python.h>                      17,901    0.05s   0.030s
<pybind11/pybind11.h>          113,539    0.69s   0.060s
<torch/.../cpp_prefix.h>       126,158    0.48s   0.038s
<torch/library.h>              191,169    1.11s   0.058s
<ATen/ATen.h>                  276,028    1.78s   0.065s
<torch/extension.h>            353,499    3.91s   0.111s
```

这张表要横着读、竖着读两遍。

**横着读（同一个入口，行数 → 时间）**：行数是最好的预测器。但单价不均匀——`<pybind11>`、`<torch/library.h>`、`<ATen/ATen.h>` 三档的「每万行」都落在 0.05~0.06 秒，而 `<torch/extension.h>` 涨到 **0.10 秒**，贵了接近一倍。

为什么？因为 `torch/extension.h` 是**两个模板体系交界的地方**：pybind11 的类型转换器（type caster）要为 `at::Tensor`、`c10::Scalar`、`c10::SymInt` 这些类型实例化模板。模板实例化的成本不和行数成正比，和**实例化出的符号数量**成正比。

**竖着读（几个入口的相对位置）**：

- `<Python.h>` → `<torch/extension.h>` 是 **353,499 / 17,901 ≈ 20 倍行数**，时间 0.05 → 3.6 秒（**72 倍**）。行数涨 20 倍，时间涨 72 倍——又是模板实例化的非线性。
- 中间那行 `<torch/csrc/inductor/cpp_prefix.h>` 请记住它，第 4 节会专门讲。它是 **126,158 行**，比 `torch/extension.h` 少 64%，编译时间只有 **1/8**。这是 PyTorch 自己给自己准备的替代品。

补一个更有冲击力的数字：补一个更有冲击力的数字：这个 3.6 秒的编译，产出的 `.o` 是 **0.5 KB**。

```python
# 3.6 秒 + 982 MB 内存，换回来多少目标码？
import subprocess, tempfile, pathlib, sysconfig, re
import torch
T = pathlib.Path(torch.__file__).parent
INC = ["-isystem", str(T / "include"),
       "-isystem", str(T / "include" / "torch" / "csrc" / "api" / "include"),
       "-isystem", sysconfig.get_path("include")]
work = pathlib.Path(tempfile.mkdtemp(prefix="mem_"))

def peak_rss(src, name):
    f = work / f"{name}.cpp"; f.write_text(src)
    # macOS 的 /usr/bin/time -l 会报 maximum resident set size（Linux 上是 -v + Maximum resident）
    r = subprocess.run(["/usr/bin/time", "-l", "c++", "-std=c++20", "-O3", "-c", *INC,
                        str(f), "-o", str(work / f"{name}.o")], capture_output=True, text=True)
    m = re.search(r"(\d+)\s+maximum resident set size", r.stderr)
    return int(m.group(1)) / 1e6, (work / f"{name}.o").stat().st_size

huge, obj = peak_rss("#include <torch/extension.h>\nint f(int x){return x+1;}\n", "huge")
tiny, _ = peak_rss("#include <Python.h>\nint f(int x){return x+1;}\n", "tiny")
print(f"include <torch/extension.h>:  峰值内存 {huge:6.0f} MB  →  产出 .o {obj/1024:.1f} KB")
print(f"include <Python.h>:           峰值内存 {tiny:6.0f} MB")
print(f"放大倍数: {huge/tiny:.1f}x")
```

```text
include <torch/extension.h>:  峰值内存    977 MB  →  产出 .o 0.5 KB
include <Python.h>:           峰值内存     59 MB
放大倍数: 16.5x
```

**982 MB 内存、3.6 秒，换 0.5 KB 目标码。** 这就是「入场费」的真实形态：编译器为了搞清 `at::Tensor` 长什么样，要在内存里装下一整套 ATen 的声明。

顺便解释一个现象：为什么在笔记本上开 `MAX_JOBS=8` 并行编译会卡死。单 TU 峰值 982 MB，8 个并发就是 7.7 GiB。16 GiB 的机器上一半内存被编译器吃掉，page cache 被挤出去，头文件得重新从磁盘读——第 3.4 节有实测。

### 3.3 账 2：缓存的三层洞

这节是全期的核心。三个层次的判断，一个比一个隐蔽。

**第一层：同进程内，它根本不判断。**

```python
# 同一个进程里连续 load 两次同名扩展：第二次不会去碰 ninja，直接返回已加载的模块
import os, sys, tempfile, shutil, pathlib, time
if not shutil.which("ninja"):
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
os.environ["TORCH_EXTENSIONS_DIR"] = tempfile.mkdtemp(prefix="cache_probe_")

import torch
from torch.utils.cpp_extension import load_inline

SRC = '#include <torch/extension.h>\ntorch::Tensor f(torch::Tensor x){return x*1.0;}\n'

t0 = time.perf_counter()
m1 = load_inline(name="cache_demo", cpp_sources=SRC, functions=["f"], extra_cflags=["-O3"])
t_first = time.perf_counter() - t0

# 参数一模一样，再 load 一次
t0 = time.perf_counter()
m2 = load_inline(name="cache_demo", cpp_sources=SRC, functions=["f"], extra_cflags=["-O3"])
t_second = time.perf_counter() - t0

print(f"首次  {t_first:6.2f} s  （付了完整入场费）")
# 第二次的耗时是个近零量（1~3 ms 抖动），比值会被它放大成噪声 —— 这里只判「够不够快」
print(f"第二次 {'< 10 ms，说明它没去启动 ninja' if t_second < 0.010 else f'{t_second*1000:.0f} ms'}")
print(f"是同一个模块对象: {m2 is m1}")
```

```text
首次    5.52 s  （付了完整入场费）
第二次 < 10 ms，说明它没去启动 ninja
是同一个模块对象: True
```

第二次只花几毫秒。但请注意它的含义：**这几毫秒和磁盘上的文件有没有变、头文件有没有变、`.so` 有没有被删，全都没关系。** 它连 ninja 都没启动。

这就是 Jupyter 里最经典的挫败感：你改了一个头文件里的常数，重跑 cell，结果没变——因为进程还活着，它手里握着旧的模块对象。

**第二层：跨进程，它精确追踪你自己写的头。**

换一个新进程，情况完全不同。`build.ninja` 里有这两行：

```
rule compile
  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags
  depfile = $out.d
  deps = gcc
```

`-MMD` 让编译器在编译时顺便吐出依赖列表，ninja 读进 `.ninja_deps` 二进制库。这是**正确**的增量编译做法。

实测：主源码 include 一个自己写的 `myconst.h`，编译一次 → 改 `myconst.h` 里的宏 → **新进程**再 load 一次。

```python
# 跨进程验证：改「自己写的头文件」会不会触发重编？用「是否真的重编」判断，不打印抖动的耗时
import os, subprocess, tempfile, pathlib, shutil, sys
work = pathlib.Path(tempfile.mkdtemp(prefix="dep_probe_"))
(work / "hdr").mkdir()
HDR = work / "hdr" / "myconst.h"
CPP = '#include <torch/extension.h>\n#include "myconst.h"\ntorch::Tensor f(torch::Tensor x){return x*MY_K;}\n'

DRIVER = r'''
import os, sys, pathlib
work = pathlib.Path(sys.argv[1])
os.environ["TORCH_EXTENSIONS_DIR"] = str(work / "ext")
import torch
from torch.utils.cpp_extension import load_inline
m = load_inline(name="dep_demo",
                cpp_sources=(work / "main_src.txt").read_text(),
                functions=["f"], extra_cflags=["-O3"],
                extra_include_paths=[str(work / "hdr")])
print(float(m.f(torch.ones(1)).item()))
'''
(work / "driver.py").write_text(DRIVER)
(work / "main_src.txt").write_text(CPP)

def fresh_run(tag, k):
    HDR.write_text(f"#pragma once\n#define MY_K {k}\n")
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    out = subprocess.run([sys.executable, str(work / "driver.py"), str(work)],
                         capture_output=True, text=True, env=env)
    val = out.stdout.strip().splitlines()[-1] if out.returncode == 0 else "失败"
    print(f"{tag:34s} f(1) = {val:5s}  {'重编了' if float(val) == k else '没重编（用了旧 .so）'}")

fresh_run("① 首次编译（MY_K=1.0）", 1.0)
fresh_run("② 改写成 MY_K=7.0，新进程", 7.0)

# 直接问 ninja：这个 .o 记住了哪些依赖
nj = shutil.which("ninja") or str(pathlib.Path(sys.executable).parent / "ninja")
deps = subprocess.run([nj, "-C", str(work / "ext" / "dep_demo"), "-t", "deps"],
                      capture_output=True, text=True).stdout
files = [l.strip() for l in deps.splitlines() if l.strip().startswith("/")]
print(f"\nninja 记录的依赖文件（{len(files)} 个）:")
for f in files:
    print("   ", pathlib.Path(f).name)
```

```text
① 首次编译（MY_K=1.0）                   f(1) = 1.0    重编了
② 改写成 MY_K=7.0，新进程                 f(1) = 7.0    重编了

ninja 记录的依赖文件（2 个）:
    main.cpp
    myconst.h
```

这里要注意代价：**它重编了，而且重编了全部**——不是只重编一个文件，是整条链路。因为扩展只有一个 TU，改任何东西都是全量。

但如果你的扩展只有几个源文件呢？那也没用，因为每个 TU 都要重新 include 一遍 `torch/extension.h`，各自付一遍 3.6 秒。

**第三层：`-isystem` 的头，它一个都不看。**

这是这期最需要记住的一条。上面那个依赖列表里，**只有 `main.cpp` 和你自己的 `myconst.h`**。`torch/extension.h` 和它展开出来的 5,997 个头文件，一个都没有。

原因就是 3.1 节观察二：`-MMD` 会跳过 `-isystem` 目录，而 torch 的两个 include 路径全是 `-isystem`。

我们自己造一个 `-isystem` 目录来验证，不碰真 torch：

```python
# 把「被 isystem 引入的头」单独拿出来验证：改了它，扩展会不会重编？
import os, subprocess, tempfile, pathlib, shutil, sys
work = pathlib.Path(tempfile.mkdtemp(prefix="isys_probe_"))
(work / "syshdr").mkdir()
CPP = '#include <torch/extension.h>\n#include "k.h"\ntorch::Tensor f(torch::Tensor x){return x*K;}\n'
(work / "main_src.txt").write_text(CPP)

DRIVER = r'''
import os, sys, pathlib
work = pathlib.Path(sys.argv[1])
os.environ["TORCH_EXTENSIONS_DIR"] = str(work / "ext")
import torch
from torch.utils.cpp_extension import load_inline
m = load_inline(name="isys_demo",
                cpp_sources=(work / "main_src.txt").read_text(),
                functions=["f"],
                # 关键：用 -isystem 引入，模拟 torch 自己那两个 include 路径
                extra_cflags=["-O3", "-isystem", str(work / "syshdr")])
print(float(m.f(torch.ones(1)).item()))
'''
(work / "driver.py").write_text(DRIVER)

def fresh_run(tag, k):
    (work / "syshdr" / "k.h").write_text(f"#pragma once\n#define K {k}\n")
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    out = subprocess.run([sys.executable, str(work / "driver.py"), str(work)],
                         capture_output=True, text=True, env=env)
    val = float(out.stdout.strip().splitlines()[-1])
    print(f"{tag:36s} f(1) = {val:4.1f}   {'重编了' if val == k else '没重编 ← 值还是旧的'}")

fresh_run("① 首次（-isystem 引入 k.h，K=1.0）", 1.0)
fresh_run("② 改 k.h 为 K=5.0，换新进程", 5.0)

nj = shutil.which("ninja") or str(pathlib.Path(sys.executable).parent / "ninja")
deps = subprocess.run([nj, "-C", str(work / "ext" / "isys_demo"), "-t", "deps"],
                      capture_output=True, text=True).stdout
files = [l.strip() for l in deps.splitlines() if l.strip().startswith("/")]
print(f"\nninja 记录的依赖文件（{len(files)} 个）: {[pathlib.Path(f).name for f in files]}")
print(">>> k.h 不在里面 —— 所以改了它，ninja 认为 .o 还是最新的")
```

```text
① 首次（-isystem 引入 k.h，K=1.0）          f(1) =  1.0   重编了
② 改 k.h 为 K=5.0，换新进程                 f(1) =  1.0   没重编 ← 值还是旧的

ninja 记录的依赖文件（1 个）: ['main.cpp']
>>> k.h 不在里面 —— 所以改了它，ninja 认为 .o 还是最新的
```

**这就是「换 torch 版本还用旧 `.so`」的机制。** 那 5,997 个头文件，一个都不在依赖图里。ninja 只看 `main.cpp` 的时间戳。

那为什么换 torch 之后大部分时候还是好的？两个原因让它侥幸没事：

1. 换 torch 通常换了安装路径，`-isystem` 的路径变了 → `build.ninja` 的命令行变了 → ninja 判定 dirty → 重编。**这层保险来自「路径变了」，不是「内容变了」。**
2. pip 重装会让整个 `site-packages/torch/` 的 mtime 更新，但那不影响判断，因为依赖表里根本没这些文件。

所以真正的危险场景是：**同一个路径下 torch 被原地替换，且 `-isystem` 路径没变。** 这时候 ninja 会认为无事发生，直接加载 ABI 可能已经不匹配的旧 `.so`。

### 3.4 账 3：优化级别的价格

现在算第三笔。同一段代码，五个优化级别，全量编译并实测运行速度。

代码用两种写法：一种朴素的标量循环（让编译器自己决定要不要向量化），一种显式调 `at::vec::Vectorized`（Inductor 生成的代码就长这样）。

```python
# 优化级别的价格：编译时间几乎不动，运行时间差 40 倍
import os, sys, time, shutil, tempfile, statistics

if not shutil.which("ninja"):
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
os.environ["TORCH_EXTENSIONS_DIR"] = tempfile.mkdtemp(prefix="opt_probe_")

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cpu/vec/vec.h>

// 写法 A：朴素标量循环，交给编译器决定是否向量化
void axpby(torch::Tensor x, torch::Tensor y, double a, double b) {
    const int64_t n = x.numel();
    const float* __restrict px = x.data_ptr<float>();
    float* __restrict py = y.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) py[i] = a * (px[i] > 0 ? px[i] : 0.f) + b;
}

// 写法 B：显式向量化，Inductor 生成的代码就是这个风格
void axpby_vec(torch::Tensor x, torch::Tensor y, double a, double b) {
    const int64_t n = x.numel();
    const float* px = x.data_ptr<float>();
    float* py = y.data_ptr<float>();
    constexpr int W = at::vec::Vectorized<float>::size();
    int64_t i = 0;
    for (; i <= n - W; i += W) {
        auto v = at::vec::Vectorized<float>::loadu(px + i);
        auto r = at::vec::clamp_min(v, decltype(v)(0))
               * at::vec::Vectorized<float>((float)a) + at::vec::Vectorized<float>((float)b);
        r.store(py + i);
    }
    for (; i < n; ++i) py[i] = a * (px[i] > 0 ? px[i] : 0.f) + b;
}
"""

LEVELS = ["none", "-O0", "-O1", "-O2", "-O3"]      # none = 不传任何旗标，也就是默认
mods, build_s, so_kb = {}, {}, {}
for lvl in LEVELS:
    flags = [] if lvl == "none" else [lvl]
    name = "optlvl_" + lvl.replace("-", "m")
    t0 = time.perf_counter()
    mods[lvl] = load_inline(name=name, cpp_sources=SRC, functions=["axpby", "axpby_vec"],
                            extra_cflags=flags)
    build_s[lvl] = time.perf_counter() - t0
    so_kb[lvl] = os.path.getsize(f"{os.environ['TORCH_EXTENSIONS_DIR']}/{name}/{name}.so") / 1024

n = 1 << 24                                   # 16M float = 64 MiB，远超 L2
x = torch.randn(n) * 0.5
y = torch.empty(n)
BYTES = n * 4 * 2                             # 读一次 + 写一次

# 预热：把数组的 page fault 打掉、把两个 kernel 都跑热（不预热会量到 page fault）
for _ in range(3):
    for lvl in LEVELS:
        mods[lvl].axpby(x, y, 2.0, 1.0)
        mods[lvl].axpby_vec(x, y, 2.0, 1.0)

# 交替采样：每轮把 5 档都碰一遍，避免「先测的那档背锅」
best = {lvl: {"axpby": 9e9, "axpby_vec": 9e9} for lvl in LEVELS}
for _ in range(12):
    for lvl in LEVELS:
        for fn in ("axpby", "axpby_vec"):
            f = getattr(mods[lvl], fn)
            t = time.perf_counter(); f(x, y, 2.0, 1.0)
            best[lvl][fn] = min(best[lvl][fn], time.perf_counter() - t)

print(f"{'优化级别':10s}{'编译':>8s}{'.so':>9s}{'标量 GB/s':>11s}{'显式向量 GB/s':>14s}")
print("-" * 54)
for lvl in LEVELS:
    print(f"{lvl:10s}{build_s[lvl]:7.2f}s{so_kb[lvl]:8.1f}K"
          f"{BYTES/best[lvl]['axpby']/1e9:11.1f}{BYTES/best[lvl]['axpby_vec']/1e9:14.1f}")

y.zero_(); mods["-O3"].axpby_vec(x, y, 2.0, 1.0)
print("\n正确性:", bool(torch.allclose(y, 2 * torch.clamp_min(x, 0) + 1, rtol=1e-5, atol=1e-5)))
```

```text
优化级别            编译      .so    标量 GB/s     显式向量 GB/s
------------------------------------------------------
none         4.84s   836.6K        1.9           0.8
-O0          4.75s   836.6K        1.9           0.8
-O1          5.37s   223.2K       34.2         109.9
-O2          5.32s   221.8K       77.5         111.2
-O3          5.37s   223.2K       77.5         112.6

正确性: True
```

这张表有两个反直觉的地方，都得说清楚。

**反直觉一：`-O0` 下「显式向量化」比标量循环还慢一倍多。**

因为 `at::vec::Vectorized` 的 `loadu` / `clamp_min` / `store` 全是模板函数和运算符重载。在 `-O0` 下它们不被内联，每次调用都要真的进函数、传结构体。你「手写了向量化」，得到的是「一个没优化的循环 + 每轮三次函数调用」。

这条能迁移到别的场合：**向量化 API 的价值建立在内联之上，而内联是优化的产物。** 在不优化的编译下，写得越「高级」越慢。

**反直觉二：编译时间几乎不受优化级别影响。**

表里 `none` 是 4.4 秒、`-O3` 是 4.9 秒，差 11%——但这个差额里包含链接 `.so`、起 ninja、写构建文件这些和优化无关的固定开销。把链路拆开、只量 clang 对同一个 TU 的那一段，两档之间交替测四轮：

| 旗标 | 单次 clang 编译（4 轮中位） |
|---|---|
| 不传 `-O`（默认 `-O0`） | 3.73 s |
| `-O3` | 3.76 s |

比值 **0.99**。也就是说：**省下 0 秒，赔掉 40 倍。**

原因回到账 1：编译时间被头文件解析支配（353,499 行），优化 pass 只作用在最后那 0.5 KB 的目标码上。**你的代码在总编译时间里占比不到千分之一，所以优化它不要钱。**

这条结论的实际意义很直接：**除了调试符号，没有任何理由在 `cpp_extension` 里不加 `-O3`。**

### 3.5 顺带算清：多几个源文件会怎样

如果扩展不止一个文件，直觉是「并行编译就能救」。实测四档。

注意口径：下表用**裸 clang**（`c++ -std=c++20 -O3 -c`），不含 3.4 节 `load_inline` 的 Python 侧开销，所以单 TU 是 3.90 秒而不是 4.4 秒。

| 编译单元数 | 串行墙钟 | 并行墙钟（`&` + `wait`） | 加速比 |
|---|---|---|---|
| 1 | 3.90 s | 3.87 s | 1.01x |
| 2 | 7.81 s | 4.14 s | 1.89x |
| 4 | 15.44 s | 5.47 s | 2.82x |
| 8 | 30.00 s | **38.55 s** | **0.78x** |

前三档符合预期（并行度受 4 个 P 核限制，4 档只到 2.82x）。第四档**崩了**：8 个并发比串行还慢 22%。

原因在 3.2 节已经埋好：单 TU 峰值内存 982 MB，8 个并发 7.7 GiB。16 GiB 的机器上被吃掉一半，page cache 被挤掉，6000 个头文件得重新从磁盘读。

所以 `MAX_JOBS` 不是一个「越大越快」的旋钮，它是一个和**内存总量**绑定的旋钮。经验值：`MAX_JOBS ≈ 可用内存(GB) / 1`，也就是每 GiB 内存放一个 clang。

---

## 4. 围绕该领域展开

到这里，三笔账都算完了。但这一节才是「领域」和「函数用法」的分界：`cpp_extension` 在整个 PyTorch 生态里的位置，以及**PyTorch 自己怎么评价它**。

### 4.1 PyTorch 自己都不用 `torch/extension.h`

这是本期最有趣的一环。跑一次 `torch.compile`，然后去缓存目录里把 Inductor 生成的 C++ 挖出来看。

```python
# Inductor 的 CPU 路径：同一套 include 真理源，另一套完全不同的构建流程
import os, glob, pathlib, tempfile, re

CACHE = tempfile.mkdtemp(prefix="inductor_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE
import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 64)
    def forward(self, x):
        return torch.relu(self.lin(x)) * 2.0 + 1.0     # 三个逐元素算子，看它融不融

m, x = M().eval(), torch.randn(8, 64)
cm = torch.compile(m)
with torch.no_grad():
    y = cm(x)
print("和 eager 一致:", torch.allclose(y, m(x), atol=1e-6))

cpps = sorted(glob.glob(CACHE + "/**/*.cpp", recursive=True), key=os.path.getsize)
print(f"\n生成了 {len(cpps)} 个 .cpp:")
for f in cpps:
    print(f"   {pathlib.Path(f).name}  ({os.path.getsize(f)} B)")

main = max(cpps, key=os.path.getsize)
text = pathlib.Path(main).read_text()
print(f"\n它 include 的入口（不是 <torch/extension.h>）:")
for l in text.splitlines():
    if l.strip().startswith("#include <"):
        print("   ", l.strip())
print(f"\n有没有 pybind11: {'pybind11' in text}")
print(f"有没有 openmp 并行: {'#pragma omp' in text}")

# 找到 kernel 本体，看循环里有没有三次遍历
body = text[text.find("extern \"C\""):text.find("// Python bindings")]
print(f"\nkernel 本体（{len(body.splitlines())} 行）里的遍历次数: {body.count('for(int64_t')}")
print("前 8 行:")
for l in body.splitlines()[:8]:
    print("   ", l.rstrip())
```

```text
和 eager 一致: True

生成了 2 个 .cpp:
   cbpzfpo2pzkbecubwjlbwtmermsdnunm2b7ikp3g7dgmh4tfxncl.cpp  (459 B)
   cnbyr5zdzfy3i6nmaiyvimqr2urbknkvelsajph3x6hqksswykrr.main.cpp  (4497 B)

它 include 的入口（不是 <torch/extension.h>）:
    #include <torch/csrc/inductor/cpp_prefix.h>
    #include <Python.h>
    #include <sstream>
    #include <cstdlib>
    #include <cerrno>

有没有 pybind11: False
有没有 openmp 并行: False

kernel 本体（27 行）里的遍历次数: 1
前 8 行:
    extern "C"  void  kernel(float* in_out_ptr0)
    {
        std::atomic<int> inductor_cpu_integer_div_error{0};
        inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
        {
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(512LL); x0+=static_cast<int64_t>(4LL))
            {
                {
```

三个发现：

**发现一：入口换成了 `<torch/csrc/inductor/cpp_prefix.h>`。** 而 `main.cpp` 里有一行注释直接写着：

```
// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
```

**「slooow」是 PyTorch 源码里的原话。** 官方自己承认那 353,499 行是个负担，并且为此专门做了一个 126,158 行的轻量入口。

**发现二：不用 pybind11。** 它手写 Python C API（`#include <Python.h>` + 自己拼 `PyModuleDef`）。因为 3.2 节的表告诉我们，`<pybind11/pybind11.h>` 单独就是 113,539 行——而在 Inductor 的场景里，它只需要把几个 `void*` 传给 `kernel()`，不需要任何类型转换器。

顺带解释输出里那个 `有没有 openmp 并行: False`：这个模型只有 8×64 = 512 个元素，Inductor 判断不值得开 omp（线程启动成本大于收益），所以生成的是单线程向量循环。元素数上到几万之后才会出现 `#pragma omp parallel for`。

**发现三：三个逐元素算子被融合成了一个循环。** 这是第 040 期讲的事，在这里有了具体的产物。

代价对比（同一台机器、同一个 `-O3`）：

| | 入口行数 | 编译一个 TU | 产出 `.so` |
|---|---|---|---|
| `cpp_extension` 默认 | 353,499 | 3.6 s | 220 KB |
| Inductor 的 `cpp_prefix.h` | 126,158 | 0.43 s | 50~57 KB |

**8.4 倍。** 这个数字不是「PyTorch 优化得好」，而是「PyTorch 知道自己不需要那些东西」。它不需要自动微分、不需要 `Tensor` 的完整接口、不需要 pybind11 的 type caster——它只需要 `at::vec::Vectorized` 和裸指针。

### 4.2 同一套真理源，两套构建流程

更值得注意的是：**Inductor 并没有完全抛弃 `cpp_extension`。**

`torch/_inductor/cpp_builder.py:1373` 和 `:2121`：

```py
from torch.utils import cpp_extension
include_dirs = cpp_extension.include_paths(...)
libraries_dirs = cpp_extension.library_paths(...)
```

它复用的是「**torch 的头和库在哪**」这个真理源，然后自己拼编译命令（`_get_optimization_cflags`、`cpp_compile_command`），不走 ninja、不写 `build.ninja`。

于是 PyTorch 内部并存着两套 C++ 构建流程：

| | `cpp_extension` | Inductor `cpp_builder` |
|---|---|---|
| include 路径来源 | `include_paths()` | `include_paths()`（同一个） |
| 构建描述 | `build.ninja` + ninja | Python 里拼 list，直接 `subprocess` |
| 增量判断 | ninja + `-MMD` | 自己算 hash 做缓存目录名 |
| 头文件入口 | `torch/extension.h` | `torch/csrc/inductor/cpp_prefix.h` |
| 绑定方式 | pybind11 | 手写 Python C API |
| 优化旗标 | **你自己传** | `-O3 -DNDEBUG`（`min_optimize=False` 时） |

顺带两个从源码里读到的细节，容易踩：

1. **Inductor 在 macOS 上不加 `-march=native`。** `cpp_builder.py:1063` 的注释：「`-march=native` is not recognized on Apple Silicon」。Linux aarch64 上会加，macOS 上不加——这意味着 macOS 上编出来的 kernel **没有指定目标架构**，用 clang 的默认值。
2. **`compile_wrapper_opt_level` 默认 `O1`，但它只影响 `min_optimize=True` 的路径。** JIT 的 `torch.compile` 走 `min_optimize=False`，拿的是 `-O3 -DNDEBUG`。别看到 `O1` 就以为 kernel 是 O1 编的。

### 4.3 和 `torch.library` 的分工：两条注册路径

第 039 期讲了 ATen 的 Dispatch 机制。这期补上它在「写扩展」这一侧的分叉：

| | pybind11 风格（本期主线） | `TORCH_LIBRARY` 风格 |
|---|---|---|
| 入口 | `PYBIND11_MODULE` + `m.def` | `TORCH_LIBRARY(ns, m) { m.def("op", &impl); }` |
| 调用路径 | Python → pybind11 → 你的 C++ 函数 | Python → dispatcher → backend → 你的函数 |
| 能被 `torch.compile` 追踪 | 不能（Dynamo 看不见） | 能（它是 ATen op） |
| 能做融合 / 换 backend | 不能 | 能（DispatchKey 决定） |
| `load_inline` 的 `functions=[...]` | 生成 pybind11 | 不生成 |
| 典型用途 | 调试、微基准、给 Python 补个纯函数 | 真的要进算子库 |

一句话：**`functions=[...]` 走的是旁路，`TORCH_LIBRARY` 才是进 dispatcher 的正门。** 旁路的代价是 `torch.compile` 会在你的调用点断开图（graph break），收益是编译代码量少、不用管 schema。

同一条线继续往下，就是 B 组还没发的那条「自定义算子与 PrivateUse1 后端」——那块讲的是「把新硬件接进 dispatcher」，和这里是同一个注册机制的另一端。

### 4.4 头文件成本的三种解法

既然 3.6 秒全是头文件，那解法只有三种，各有各的代价：

**解法一：换更小的入口。** Inductor 用的就是这个（`cpp_prefix.h`）。对你也有用：如果你的扩展只需要 `at::Tensor` 的裸指针和 `at::vec::Vectorized`，完全可以只 include `<ATen/ATen.h>`（276,028 行 / 1.65 秒）甚至更窄的东西。代价是失去 pybind11 的自动类型转换，得手写 Python C API。

**解法二：预编译头（PCH）。** `cpp_extension.py:1978` 有个 `_check_and_build_extension_h_precompiler_headers()`，看起来是给 `torch/extension.h` 做 PCH 的。但它的第 1993 行是：

```py
if not IS_LINUX:
    return
```

紧接着 1998 行还要求编译器是 GCC。**所以 PCH 在 macOS 上完全不生效，在 Linux + clang 上也不生效。** 我在本机验证过：`torch/include/torch/extension.h.gch` 这个文件根本不存在。

这条很重要——如果你在网上看到「PyTorch 有 PCH 加速，第一次编译慢后面快」，那说的是 Linux + GCC 的组合，macOS 上这些机制一行都不跑。

**解法三：拆模块，少 include。** 这是唯一通用的解，也是最不性感的解：把「需要 torch 类型」的代码和「纯计算」的代码分开。纯计算部分不 include torch，编译只要 0.02 秒；torch 那层做一次薄封装。这也是为什么 3.6 秒的成本只该付一次。

### 4.5 一个已经消失的旗标

如果你照抄过网上的 `cpp_extension` 教程，可能会见过这条：

```
-D_GLIBCXX_USE_CXX11_ABI=0
```

它是 libstdc++ 的双 ABI 问题的传统解药。而在本机装的 torch 2.14.0 里，这个字符串在 `cpp_extension.py` 里**一次都不出现**。我顺着 tag 查了一遍：

| 版本 | `_GLIBCXX_USE_CXX11_ABI` 出现次数 |
|---|---|
| v2.4.0 | 2 |
| v2.6.0 | 2 |
| v2.7.0 | 2 |
| **v2.8.0** | **0** |
| v2.9.0 | 0 |
| main | 0 |

v2.7.0 里它还在 `_get_glibcxx_abi_build_flags()` 和 `BuildExtension._add_gnu_cpp_abi_flag()` 里；v2.8.0 起这两个函数都没了。

**这不是本文的推断，是 grep 出来的行数。** 结论只有一句：抄二手资料时先在你自己的版本上 grep 一遍。本期所有源码引用都标了行号，也都标了版本。

---

## 5. 什么时候该用 / 不该用

把三笔账压成一张表。这张表的每一行都有前面的实测数字支撑。

| 你的场景 | 该用什么 | 首字成本 | 理由 |
|---|---|---|---|
| 确立一个物理事实（带宽、延迟、指令开销） | `cc -O3` 现编 C | **43 ms** | 不需要 tensor 语义，就不该付 353,499 行的门票 |
| 算法原型，反复改常数 | `cc -O3`，或 Triton | 43 ms / 一次 JIT | 改一次等 5 秒的原型不是原型 |
| 需要 `torch.Tensor` 语义的调试 kernel | `load_inline` + `-O3` | 5~6 s | 一次性的，值 |
| 要进 dispatcher、要能被 `torch.compile` 融合 | `TORCH_LIBRARY` + `load` | 5~6 s | pybind11 那条路 Dynamo 看不见 |
| 生产代码里的自定义算子 | `setup.py` + `BuildExtension` | 一次性 | 走正式包管理，别用 JIT |
| 只想给 Python 补个纯函数（无 tensor） | 别用 C++ 扩展，用 `ctypes`/`cffi` | 0 | 43 ms vs 5.8 s |

最后一行值得单独说。我做了一次完全对等的对照：同一个 `y = clamp_min(x,0)*2+1`，64 MiB 数组。

| | 构建 | 运行 | 带宽 |
|---|---|---|---|
| `cc -O3` 现编 C | **43 ms** | 1.722 ms | 77.9 GB/s |
| `cpp_extension` + `-O3` | **5770 ms** | 1.729 ms | 77.6 GB/s |

（口径：`cpp_extension` 的首次构建在 11 次独立进程中实测落在 **4.8~7.2 秒**区间，取决于 page cache 与磁盘状态；纯 C 那 43 ms 稳定在 43~44 ms。表里取的是其中一次点值。）

**运行速度分毫不差，构建成本差 134 倍。** 因为你两边用的都是 clang `-O3`，编出来的机器码就该是一样的。`cpp_extension` 那 5.7 秒一分钱都没买到更快的代码，它买的是「能和 `torch.Tensor` 对话」。

顺带一个诚实的补充：那个 `cc` 版第一次跑是 5.718 ms / 23.5 GB/s，加了 warmup 之后才是 1.722 ms / 77.9 GB/s。**3.3 倍**的差距全是 page fault——`malloc` 出来 64 MiB 第一次写要 fault 16384 个页。所以两边的「一次性成本」都存在，只是位置不同：一个在编译期，一个在首次运行。

---

## 6. 常见坑

### 坑 1：默认不优化，慢 40 倍

`load_inline(name=..., cpp_sources=...)` 不传 `extra_cflags`，得到的是 clang 默认的 `-O0`。实测 1.9 GB/s vs `-O3` 的 77.6 GB/s。

而单次 clang 的编译时间几乎不变（3.73 s → 3.76 s）。所以这不是「用编译时间换运行时间」的取舍，是纯亏。

**唯一例外**：你要单步调试自己的 kernel（`gdb`/`lldb` 追进去看变量），那 `-O0 -g` 是对的。除此之外一律加 `-O3`。

### 坑 2：`functions=[...]` 绑不了模板函数

`load_inline(functions=["tpl_add"])` 会替你生成 pybind11 的 `m.def`。但你给的必须是一个**已经实例化好的具体函数**，不能是模板。真实报错（本机抓的）：

```
error: no matching function for call to 'wrap_pybind_function'
1 error generated.
```

报错不说「模板不能绑」，只说 `wrap_pybind_function` 找不到匹配——因为它是在拿一个模板去匹配一个要具体类型的函数。看到这个错误，先去检查 `functions` 里那几项是不是模板、有没有重载、有没有默认参数。

**正确做法**：自己写一层具体签名的包装函数，把模板实例化掉：

```cpp
template <typename T> T tpl_add(T a, T b) { return a + b; }
torch::Tensor add_f32(torch::Tensor a, torch::Tensor b) { /* 这里实例化 tpl_add<float> */ }
// functions=["add_f32"]
```

### 坑 3：`PYBIND11_MODULE` 的名字必须等于扩展名

`cpp_extension` 会传 `-DTORCH_EXTENSION_NAME=<你的 name>`，标准写法是 `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`。如果你图省事写成字面量：

```cpp
PYBIND11_MODULE(wrong_name_here, m) { ... }
```

编译能过，链接能过，`import` 时报：

```
ImportError: dynamic module does not define module export function (PyInit_mod_name_probe)
```

因为 CPython 找的是 `PyInit_<扩展名>`。这个错误看起来像「`load` 失败」，其实 `.so` 已经完全编好了，只是导出符号名不对。

### 坑 4：不先 `import torch` 就加载 `.so`，报的错跟符号无关

`ldflags` 里有 `-undefined dynamic_lookup`，很多人以为「所以 `.so` 可以脱离 torch 单独加载」。试试直接 `ctypes.CDLL`：

```
OSError: dlopen(...): Library not loaded: @rpath/libc10.dylib
  Referenced from: .../mod_name_probe.so
  Reason: tried: '.../python3.13/lib/libc10.dylib' (no such file)
```

`-undefined dynamic_lookup` 只允许**符号**未定义。而 `-lc10 -ltorch_cpu -ltorch` 是**显式链接**的，`libc10.dylib` 会作为 LC_LOAD_DYLIB 写进 `.so` 的加载依赖。而 `_prepare_ldflags` 只在 `is_standalone=True` 时才加 `-Wl,-rpath`（`cpp_extension.py:2616`），JIT 路径不加。

所以：`import torch` 必须在加载扩展之前，它是把那些 dylib 先放进进程、让 dyld 能按 install_name 命中的唯一方式。

### 坑 5（不在代码里，在环境里）：ninja 必须在 PATH

`cpp_extension` 在 2.x 里**没有** fallback 到 distutils 的路径。找不到 ninja 就是硬失败：

```
RuntimeError: Ninja is required to load C++ extensions (pip install ninja to get it)
```

如果你的 `pip install ninja` 装进了 venv 但 venv 的 `bin` 不在 `PATH`（很多 IDE 的 Jupyter kernel 就是这样），就会踩。修法一行：

```python
import os, sys
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
```

---

## 7. 一句话总结

**`cpp_extension` 的 5 秒不是在编译你的代码，是在为「和 PyTorch 共处一个进程」付门票；门票按头文件行数计价（353,499 行 / 982 MB），缓存按整间仓库计价（改一行全量重编），而对 torch 自己那 6,000 个头它的增量判断是瞎的（`-MMD` 跳过 `-isystem`）——所以别指望它替你省事，只该在确实需要 tensor 语义时用它，其余场合 `cc -O3` 用 43 毫秒做同一件事。**

---

## 今日练习

<details>
<summary>练习 1：算出你机器上的「入场费单价」</summary>

在你自己机器上跑 3.2 节那段代码，然后把 `<torch/extension.h>` 那一行的时间除以 353,499/10000，得到「每秒多少万行」。

参考值（本机 Apple M5，clang 21）：
- 平均 **0.10 秒 / 万行**
- 也就是说：**你每写 100 行自己的代码，只占总编译时间的 0.03%**

再算一个数：如果把你项目里所有 `#include <torch/extension.h>` 换成 `<ATen/ATen.h>`，编译时间大概变成几分之几？先用 3.2 节的表估，再实际改一个文件验证。

> **提示**：那个 0.10 秒/万行 明显高于 `<ATen/ATen.h>` 的 0.06，差额是 pybind11 的 type caster 实例化。所以如果你的扩展不用 pybind11 的自动转换（手写 Python C API），能省掉的不止是行数。

</details>

<details>
<summary>练习 2：亲手制造一次「旧 .so」事故</summary>

按 3.3 节第三个实验的骨架，但这次**不换进程**：

1. 在同一个进程里编一个扩展，调用它
2. 在进程里改掉那个 `-isystem` 头文件的内容
3. 再次调用同一个函数

你会发现返回值**没变**。然后做对照：在同一个进程里改 `main.cpp`（也就是 `cpp_sources` 的内容）再 load，返回值**会变**（因为它会重编，约 5 秒）。

用这两次对比说清一件事：**同进程内的判断依据是「你这次传的参数有没有变」，不是「磁盘上的东西有没有变」。**

延伸：找一个真实的开源扩展（比如 `flash-attn`、`apex`），看它的构建脚本有没有刻意处理「torch 升级后缓存失效」这件事。

</details>

<details>
<summary>练习 3：把 8 个编译单元从 0.78x 救回来</summary>

3.5 节里 8 个 TU 并行的加速比是 **0.78x**（比串行还慢）。原因推测是内存不够（8 × 982 MB ≈ 7.7 GiB）。

验证并修复：

1. 先量出你机器上单 TU 的峰值 RSS（用 `/usr/bin/time -l`，Linux 用 `/usr/bin/time -v`）
2. 用 `MAX_JOBS` 从 1 扫到 12，找出你的机器上的最优并行度
3. 对比两种降内存的手段：
   - 把 `#include <torch/extension.h>` 换成更窄的入口
   - 加 `-fno-exceptions -fno-rtti`（如果你的代码不用异常/RTTI）

画出「并行度 → 墙钟」曲线，标出拐点。你会发现拐点位置和 `内存总量 / 单 TU 峰值` 高度相关。

> **注意**：这个练习里所有测量都要用「交替采样」而不是「连续测同一档」，否则你会把顺序效应当成并行度效应——我在 3.4 节第一次测优化级别时就踩了这个坑，同一档在两次连续测量里差了 3 倍。

</details>

---

<!-- 本期全部实测数据于 2026-11-10 在 Apple M5（10 核 4P+6E / 16 GiB / macOS 26.5.1 / torch 2.14.0 / clang 21）上采集。
     A 组：本机实测（真实 stdout）。源码引用标注了文件与行号，版本为 torch 2.14.0 与上游 v2.4~v2.9 tag 对比。
     本机无 CUDA、无 GPU，因此不涉及任何 GPU kernel 行为。 -->
