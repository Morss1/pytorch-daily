# PyTorch 每日一课 · 第 008 期

## Triton：在 PyTorch 里写自己的 GPU 内核

| 项目 | 内容 |
|---|---|
| 日期 | 2026-08-28 |
| 难度 | ⭐⭐⭐⭐ |
| 前置知识 | 基本的 PyTorch 张量操作、理解 GPU 是并行设备、对「算子」是什么有直觉 |
| 预计阅读时间 | 18 分钟 |

---

## 一、这个领域解决什么问题

在前几期里，我们已经被反复"剧透"了一些性能"魔法"：

- 第 005 期 Flash Attention：不把 `N×N` 注意力矩阵写回 HBM，改在 SRAM 上分块算
- 第 003 期 `torch.compile`：让 Inductor 自动生成"卷积 + 融合"的 GPU 代码
- 第 004 期 Caching Allocator：把零碎显存合并成大块再交付

这些优化的共同点是什么？**它们都最终落地为一段跑在 GPU 上的"小代码"，而这段代码不是 Python，是直接面向 CUDA 线程的程序**。Flash Attention 在论文里写的是 CUDA C++；Inductor 后端有一个武器叫 Triton；很多人想自己实现一个新算子（比如某种奇怪的激活函数、某种 fused loss），却只能对着「`torch._C` 怎么注册算子」望而却步。

**Triton 的目标就是把这道门打开**：让你用一种"看起来像 Python、实际上是 GPU 编程语言"的方言，直接写出可被 CUDA 性能级编译的 kernel。它是 PyTorch 团队官方主推的可编程 GPGPU 语言，连 Inductor 默认后端都用它。

学完这一期你会懂：

- 为什么 PyTorch 不能在 Python 层"自动"生成那种高速 kernel
- Triton 的"块级"思维和 CUDA 线程级有什么不同
- 怎么从零写一个 fused 算子（带 PyTorch baseline 对比）
- Triton 在整个 PyTorch 生态里扮演什么角色——它和 `torch.compile`、AOT Inductor、Flash Attention、定制 CUDA 扩展的关系

---

## 二、核心思想：把 GPU 当成"巨大的 SIMD 数组"而不是"线程大军"

### 2.1 经典的 CUDA 心智模型：每个线程都是一个"工人"

写 CUDA C++ 时，每一行代码都对应**一个线程**的执行：

```cpp
__global__ void add(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];   // 每个线程处理一个元素
}
```

这种模型的问题是：你要操心 warp 一致性、shared memory bank conflict、寄存器压力……光是写一个 `matmul`，就要花几百行。

### 2.2 Triton 的心智模型：我管"一整块"，编译器管线程

Triton 的核心抽象叫 **Block（块）**：你写的每一行代码，处理**一整块**数据（比如一个 128×128 的矩阵 tile），你描述的是"对这块数据做什么"，编译器自动把它映射到一堆 CUDA 线程上。

这有非常像**SIMD 编程**：

| | 写 CUDA C++ | 写 Triton |
|---|---|---|
| 你描述的粒度 | 每个线程做什么 | 每个 block 做什么 |
| 线程分配 | 程序员手动 | 编译器自动 |
| shared mem 设计 | 手工 bank 优化 | 编译器 `tl.swizzle` 决定 |
| 调试困难 | 极大 | 还可以 print |

> **一个生动的比喻**：
>
> CUDA 编程像是让你做包工头——每个工人（线程）你都得认识、分配任务、检查状态。
>
> Triton 编程像是让你做"项目经理"——你只关心把 100 人的团队分成 8 个 12 人小组，每组负责一个工位（block），组内怎么分工交给班组长（编译器）处理。

### 2.3 为什么 Triton 不只是"另一种写法"

关键在于**自动 block 调优**：同一个 Triton kernel，可以由编译器按 GPU 型号自动选择 block 大小、num_warps、num_stages——而 CUDA C++ 通常要靠程序员手写 `cudaOccupancyMaxPotentialBlockSize` 反复试。

---

## 三、在 PyTorch 中怎么用

### 3.1 安装与版本

Triton 自 PyTorch 2.0 之后已经作为核心依赖被装上：

```python
import torch
import triton
import triton.language as tl

print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
```

如果你的 PyTorch 是较新版本（2.2+），Triton 通常是 2.x；如果在 conda 默认通道里看到版本过旧，可以单独 `pip install triton --upgrade`。

### 3.2 第一个 kernel：向量加法（对照 PyTorch baseline）

写一个"向量加 + 缩放"的 Triton kernel，并与 PyTorch 对照速度。

```python
import torch
import triton
import triton.language as tl


@triton.jit                    # 用这个装饰器，Triton 就会 JIT 编译
def add_scale_kernel(
    x_ptr, y_ptr, out_ptr,    # 输入、输出指针
    n_elements,                # 张量总长度
    scale,                     # 缩放常数
    BLOCK_SIZE: tl.constexpr,  # 每个 block 处理的元素数（编译期常量）
):
    pid = tl.program_id(axis=0)             # 当前是第几个 block
    block_start = pid * BLOCK_SIZE          # 这个 block 负责的起始下标
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 这个 block 内的所有下标

    # 创建"越界掩码"——不要超出张量真实长度
    mask = offsets < n_elements

    # 加载（编译器会尽量放在 SMEM 里）
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 计算
    out = x + y * scale

    # 写回
    tl.store(out_ptr + offsets, out, mask=mask)


def add_scale(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    """封装一下，方便和 PyTorch 接口一致"""
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024                # 每个 block 处理 1024 个元素
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)  # 算出需要多少个 block
    add_scale_kernel[grid](x, y, out, n, scale, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ----- 测试 -----
if __name__ == "__main__":
    x = torch.randn(1024 * 1024, device="cuda")
    y = torch.randn(1024 * 1024, device="cuda")
    scale = 2.5

    # PyTorch baseline
    ref = x + y * scale

    # Triton 版本
    out = add_scale(x, y, scale)

    print("最大误差:", (out - ref).abs().max().item())  # 应该是 0
```

**几个关键点**：

- `tl.program_id(axis=0)` 类似 CUDA 的 `blockIdx.x`
- `tl.arange(0, BLOCK_SIZE)` 类似 CUDA 的 `threadIdx.x`，但**它是整块一起生成下标向量**
- `tl.load` / `tl.store` 自带掩码，相当于 CUDA 里手写 `if (i < n)`
- `BLOCK_SIZE` 必须标 `constexpr`，因为它是编译期参数

### 3.3 自动调优：让编译器选最优配置

Triton 最强大的武器之一——**AutoTuner**：

```python
import triton

# 准备多组候选配置
configs = [
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
]

@triton.autotune(configs=configs, key=['n_elements'])
@triton.jit
def add_scale_v2(x_ptr, y_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y * scale, mask=mask)


def add_scale_v2_call(x, y, scale):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_scale_v2[grid](x, y, out, n, scale)   # 注意：BLOCK_SIZE 已由 autotuner 注入
    return out
```

`key=['n_elements']` 表示"按张量长度缓存最优配置"。同一长度再调用时直接拿缓存结果，不重新测。

### 3.4 用 `triton.testing.do_bench` 对比性能

```python
import triton.testing

ms_triton = triton.testing.do_bench(
    lambda: add_scale_v2_call(x, y, scale)
)
ms_torch = triton.testing.do_bench(
    lambda: x + y * scale
)
print(f"Triton:  {ms_triton:.4f} ms")
print(f"PyTorch: {ms_torch:.4f} ms")
```

对这种简单算子，PyTorch 通常不慢，因为有 cuBLAS/cuDNN 加持；但**对复杂 fused 算子**，Triton 往往能轻松赢得 2~10 倍。

---

## 四、围绕该领域展开：Triton 在整个 PyTorch 生态里的位置

Triton 不是一个孤立工具，它处在 PyTorch 性能栈的**关键节点**。

### 4.1 `torch.compile` 的真正后台——Inductor ≈ Triton 编译器

第 003 期讲的 `torch.compile` 背后是 **TorchInductor**，它的工作流是：

```
Python 代码
  ↓ (Dynamo 抓图)
FX Graph (一系列 aten 调用)
  ↓ (Inductor 前端)
分块 + 融合决策
  ↓ (Inductor 后端 → Triton)
Triton kernel
  ↓ (Triton 编译器)
PTX / SASS (真正的 GPU 指令)
```

> 也就是说，**当你用 `torch.compile(model)` 时，那些 fused kernel 几乎都是 Triton 写的**。
> 懂得 Triton，你就能看懂 Inductor 在做什么；看不懂时也可以自己 debug 它的 intermediate。

### 4.2 Flash Attention 2/3 直接用 Triton

Tri Dao 团队后来公布的 FlashAttention 2/3 大量使用 Triton 重写，因为 Triton 让"非 GEMM 的复杂算子"也能享受到自动调优、tiling 等优化，**比手写 CUDA 效率高很多**。理解 Triton 后，你读 FA 的源码会觉得顺畅很多。

### 4.3 与 `torch.utils.cpp_extension`（CUDA 扩展）的关系

| 维度 | Triton | CUDA C++ 扩展 |
|---|---|---|
| 语言 | Python 方言 | C++/CUDA C++ |
| 编译难度 | 几乎无感知 | 需要 nvcc、setuptools |
| 性能上限 | ~90% 手工 CUDA | 理论最高 |
| 调试 | print 大法 | print + cuda-gdb |
| 适合场景 | 大多数 fused 自定义算子 | 极致性能、与现有 CUDA 库交叉 |

一句话：能用 Triton 解决就别拉 CUDA 扩展。除非你对极致延迟敏感（比如推理发动机）。

### 4.4 `torch.library` + Triton：把自定义 kernel 装进 PyTorch

如果你写了一个 Triton kernel 想被 PyTorch 原生支持（包括 autograd），可以走 `torch.library` 路线：

```python
from torch.library import custom_op, impl

@custom_op("my_ops::fused_gelu_add", mutates_args=())
def fused_gelu_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Triton kernel 调用
    return my_triton_helper(x, bias)

@impl("my_ops::fused_gelu_add", "CUDA")
def fused_gelu_add_cuda(x, bias):
    return my_triton_helper(x, bias)
```

注册之后，你的算子就能挂在 autograd 图上、被 `torch.compile` 进一步融合、被 Dynamo 追踪。**这相当于"民间 Triton 算子"和"PyTorch 一等公民算子"之间的桥梁**。

### 4.5 推理优化领域：AOT Inductor

`torch.export` + `AOT Inductor` 的流程最终会把模型"离线编译"成 Triton + C++ 的混合代码（生成 `output.so`），由 `torch._inductor.aot_compile` 喂给推理引擎。这是 **Triton 在生产部署上的主舞台**——它不是 ad-hoc kernel playground，而是 PyTorch 推理性能的关键支柱。

---

## 五、什么时候该用 / 不该用

### ✅ 该用

- **自定义 fused 算子**：比如 `(x * y + z).relu()` 这种简单链式操作，你想一次融合避免多次读 HBM
- **新研究的算子**：不想碰 nvcc，但又想跑出接近手写 CUDA 的速度
- **做研究**：paper 里新提出的 attention / MoE 路由，Triton 是最常用"快速验证工具"
- **想做 Inductor 之外的 fallback**：有时 `torch.compile` 生成的 kernel 不够好，你需要手写替换

### ❌ 不该用 / 谨慎用

- **已经有高度优化的 cuBLAS / cuDNN 实现**的算子（GEMM、Conv、norm）：不要重新发明轮子
- **极端延迟要求**（如交易所、高频推理）：可能仍需要手写 CUDA
- **跨硬件的可移植性**：Triton 主要面向 NVIDIA GPU，AMD（ROCm 早期）/ 华为 NPU 的支持有限
- **复杂的图算法 / BFS**：这类非密集张量计算，Triton 模型不擅长

---

## 六、常见坑

### 坑 1：忘记写 `constexpr`，block 变成 Python int

```
TypeError: constexpr_vars must be marked as such
```

Triton 要求所有影响 shape 的参数都是编译期常量。如果你不想每次重编，就把它放进 autotune 的 config 里。

### 坑 2：把"全 block"和"部分 block"混在一起时越界读取

如果 `n_elements` 不是 `BLOCK_SIZE` 整数倍，最后一个 block 必须用 mask，否则会读未初始化内存并产生随机结果。

```python
# 正确写法
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)   # 加 mask！
```

### 坑 3：以为 autotune 永远会"自动赢"

不是所有 kernel 都值得 autotune 多个配置——小算子一次调优耗时比多次运行还慢。常被忽视的：**第一次 autotune 本身就要 profiling，延迟要算在方案总成本里**。

### 坑 4：tune 后 `key` 设错导致缓存命中错误

`@triton.autotune(key=['n_elements'])` 中，如果 length 变化大但你只填了 key 一个维度，可能 cache 命中错误配置。**推荐同时把 `dtype`、`BLOCK_SIZE` 上限都写进 key**。

---

## 七、一句话总结

> **Triton 把"写 GPU kernel"的门槛从"懂 CUDA 编程模型"拉低到"懂 Python + 数组思维"，并因此成为 `torch.compile`、Flash Attention、定制算子和 AOT Inductor 共同的低层语言——它本身不是优化，而是一把让一切优化都更容易的钥匙。**

---

## 八、今日练习

**题目**：写一个 Triton kernel，把 `tanh(x)` 逐元素作用在张量上，并与 `torch.tanh(x)` 对比加速比。要求支持 `BLOCK_SIZE = {256, 1024, 4096}` 三种配置，并使用 autotune。

<details>
<summary>今日练习</summary>

```python
import torch
import triton
import triton.language as tl


configs = [
    triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
]


@triton.autotune(configs=configs, key=['n_elements'])
@triton.jit
def tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # tl.math.tanh 是 Triton 内置
    out = tl.math.tanh(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def fast_tanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    tanh_kernel[grid](x, out, n)
    return out


if __name__ == "__main__":
    x = torch.randn(8 * 1024 * 1024, device='cuda')
    ref = torch.tanh(x)
    out = fast_tanh(x)
    err = (out - ref).abs().max().item()
    ms_torch = triton.testing.do_bench(lambda: torch.tanh(x))
    ms_triton = triton.testing.do_bench(lambda: fast_tanh(x))
    print(f"max err: {err:.6e}")
    print(f"torch   : {ms_torch:.4f} ms")
    print(f"triton  : {ms_triton:.4f} ms")
    print(f"speedup : {ms_torch / ms_triton:.2f}x")
```

预期：误差量级 `1e-7` 左右，加速比大约 1.0~2.5（`torch.tanh` 本身已是高度优化 baseline，主要是想练手）。

> 进一步挑战：把 `tanh` 升级成 `tanh(x) + 0.1*x^2`，观察 Triton 自动调优选哪个配置；或者把 kernel 改成"非 masked"分支（只在长度已知是 256 倍数时使用）以观察性能差异。

</details>
