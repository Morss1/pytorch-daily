# PyTorch 每日一课 · 第 003 期

## torch.compile：PyTorch 2.x 的编译路线（Dynamo + Inductor）

> 日期：2026-08-20  
> 难度：⭐⭐⭐⭐  
> 前置知识：用 PyTorch 训练过模型；知道前向 / 反向 / 优化器三件套；本期与「CUDA Graph」「混合精度」关联极强，建议先读前两期  
> 预计阅读：14 分钟

---

### 这个领域解决什么问题

先看一段再正常不过的 PyTorch 训练代码：

```python
x = torch.randn(batch, dim, device="cuda")
for i in range(1000):
    x = torch.relu(x @ W1)
    x = torch.relu(x @ W2)
    x = x @ W3
    loss = (x - target).pow(2).mean()
    loss.backward()
```

在 H100 上你也许会惊讶地发现：**这段代码实际跑出几毫秒，但其中 70% 的时间花在了「算子被调度」这件事上，而不是「算子真正在做计算」这件事上**。

直觉上「matmul 才是重活」，但 GPU 的物理真相是：一个算子从 Python 调用到 GPU 真正执行，要经历 **Python 字节码解释 → ATen 派发 → 设备调用 → CUDA kernel launch → ……**。一张 H100 每秒能完成 30 万亿次 FP16 矩阵乘，但单个 CUDA kernel launch 的 CPU 端开销是 **几微秒**。当你的模型是大量小算子串联（小 batch、Transformer 解码阶段、循环层），GPU 真实算力利用率可能不到 30%。

`torch.compile` 这个领域就是为这件事而生的：**让用户一行不改（或几乎不改）的代码，自动获得「类手写融合 kernel」的性能**。它的全部价值可以浓缩成一句话——**把 Python 解释器从热路径上挪走**。

要做到这件事分两步，这就是为什么这个领域叫「Dynamo + Inductor」：

- **TorchDynamo（前端）**：拿到你的 Python 代码，做字节码级别的图捕获，把 `x @ W1` / `relu` / 这些零散的算子合成一张完整的 **FX Graph**
- **TorchInductor（后端）**：拿到这张图，针对每个硬件后端（GPU 用 Triton、CPU 用特殊的 codegen）生成高度融合的 kernel

而当你开启 `mode="reduce-overhead"` 时，它还会把整张图塞进 **CUDA Graph**（第一期讲过）一次执行，几乎完全消除 CPU 调度开销——三件事就这样咬合成了一个工具。

---

### 核心思想：从「解释执行」到「图执行」

PyTorch 的「eager mode」（默认模式）是逐算子立刻执行的——你写一行，CPU 立刻把它翻译成 CUDA 算子扔给 GPU。优点是灵活、好调试；缺点是每次翻译都要花钱。

`torch.compile` 的核心想法完全等于「**把急切的执行延后到「看清整张图」之后**」。类比三件事：

#### 1. 跟 JVM 的关系
Java 早期是解释执行的，后来有了 HotSpot：**边跑边统计热点（profile）**，发现某段代码是热点就把它 JIT 编译成本地机器码。`torch.compile` 的工作流非常像——第一遍执行时 Dynamo 会 trace 并编译你的函数，第二次起就直奔编译过的图。Warmup 完之后的代价就消失了。

#### 2. 跟 TensorFlow Graph Mode 的关系
TF1.x 也图执行，但代价巨大：你必须把所有控制流、数据相关分支都用 TF 的 API 重写一遍。`torch.compile` 高明在 **不动你的 Python 代码**——遇到 `if / for / 复杂数据依赖`，它会触发「**图中断（graph break）**」，把图切成几段分别编译。能编译就编译，不能就老老实实回 eager。

#### 3. 跟手写 Triton kernel 的关系
Inductor 的 GPU 后端是用 Triton 写出来的（下一期会讲 Triton）。它不是手写每个算子，而是把**整段连续计算**（比如 matmul + bias + relu 这串）融合成一个 Triton kernel，一次 launch 完成所有事，并且中间结果直接放在 GPU 的寄存器和共享内存里——**不来回读写 HBM**，带宽效率拉满。

#### 一句话描述它的本质

> **eager 是「每句话都立刻翻译成外文给对方听」；torch.compile 是「先把整段话听完整，翻译成一份完美的外文稿再一次性念出来」。**

---

### 在 PyTorch 中怎么用

#### 基础用法：一行改造训练循环

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10),
).cuda()

# 关键就这一句：把整个模型包起来
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for x, y in dataloader:
    optimizer.zero_grad()
    pred = model(x)             # 第一次跑得慢，第二次起飞
    loss = nn.functional.cross_entropy(pred, y)
    loss.backward()
    optimizer.step()
```

不需要改 `loss.backward()`、不需要改优化器、不需要改 dataloader——这就是它对用户的承诺。

#### 三种主要 mode

```python
# 1. 默认：最大化减少 Python/调度开销，按需编译
model_opt = torch.compile(model)

# 2. reduce-overhead：再叠加 CUDA Graph 消除 kernel launch 延迟
model_opt = torch.compile(model, mode="reduce-overhead")

# 3. max-autotune：每个融合 kernel 试多种 Triton 配置，挑最快的
#                   慢启动（第一次跑可能几分钟），稳定后跑得最猛
model_opt = torch.compile(model, mode="max-autotune")
```

#### 看到「图中断」发生在哪里

```python
# 打开 dynamo 的详细日志，看看哪里发生了 graph break
import torch._dynamo as dynamo
dynamo.config.log_level = logging.INFO

# 或者直接看日志：
import logging
logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
```

图中断最常见的元凶：

```python
def complicated(x):
    if x.sum() > 0:                  # ❌ 数据依赖的分支，必然 break
        return x.relu()
    else:
        return x * 2

data = torch.randn(8)
data_compiled = torch.compile(complicated)
data_compiled(data)  # 日志里会告诉你：GRAPH BREAK at <data-dependent branch>
```

```python
# 解决：把数据相关的判断挪出热路径
def clean(x):
    flag = (x.sum() > 0).item()      # 把 bool 提前成一个标量
    if flag:
        return x.relu()
    else:
        return x * 2
# 但 item() 会触发 CPU-GPU 同步——这又是一个坑
```

#### 调试：让它回退到 eager 看错误在哪

```python
# 全局禁用编译，先确认模型本身没问题，再开 compile 排查编译问题
torch._dynamo.config.suppress_errors = True
# 现在如果编译失败会 fallback 到 eager，但会有 WARNING
```

#### 看编译产物本身

```python
# 把 Dynamo 生成的 FX Graph 和 Inductor 生成的 Triton kernel 都打印出来
import os
os.environ["TORCH_LOGS"] = "output_code"

model_opt = torch.compile(model)
model_opt(data)
# 控制台会输出：
#   @triton.jit
#   def triton_red_fused__0(...):
#       ...
#     真实的 Triton 源代码
```

读完这段代码你就懂 Inductor 在做什么了——它是一段**自动融合的 GPU kernel**。

---

### 围绕 torch.compile 展开：你需要知道的相关领域

#### 1. FX Graph：Dynamo 的中间表示

Dynamo 捕获完字节码后会生成 [FX](https://pytorch.org/docs/stable/fx.html) Graph——一种类似 SSA 的 Python-IR。每个节点是 `call_function` / `call_method` / `placeholder` / `output`。Inductor 的输入就是它，**所有后端优化都在这一层发生**。

```
eager Python     →     FX Graph (Python IR)
   ↓
Inductor (后端 lowering)
   ↓
Triton kernel / CUDA kernel / CPU vec op
```

#### 2. AOTInductor：编译时而非运行时

`torch.compile` 默认是「即时编译」（JIT），第一次跑前向时才编译。`AOTInductor`（`torch._inductor.aot_compile`）让你**提前编译成独立的 .so 文件**，训练前加载、推理时零编译开销——是生产部署的关键拼图。它最终产物是一个 C++ 扩展，里面调 Triton/CUDA kernel，对延迟极敏感的场景必备。

#### 3. Dynamic Shapes 与 `mark_dynamic`

`torch.compile` 默认会**针对每个 shape 编译一份**新图。输入 batch size 一变就重编译，浪费时间。提前告诉它「某些维度是动态的」：

```python
torch._dynamo.mark_dynamic(x, 0)      # 第 0 维（通常是 batch）是动态的
torch._dynamo.mark_dynamic(x, 1)      # 第 1 维（序列长度）
model_opt = torch.compile(model)
# 现在无论你喂 1、8、64 还是 256 的 batch，都共享同一份编译产物
```

这跟 TensorRT/ONNX 的「dynamic_axes」是同一类思想。

#### 4. CUDA Graph 的内嵌使用

当你设 `mode="reduce-overhead"` 时，Inductor 会把**整张 FX 图**用 `torch.cuda.CUDAGraph` 录制一次。这就把 kernel launch 开销从「每算子几微秒」压到「整张图总共几十微秒」。

但**它跟你手写的 CUDA Graph 不同**：手写 CUDA Graph 要求输入 shape/指针固定，而 torch.compile 内嵌的版本会在第一次录制时观察占位符的指针/size，第二次执行时**自动复制数据到那个 buffer**（这就是 `cudagraph_trees` 管理的）。这个自动 buffer 复制的开销……有时反而把 CUDA Graph 的优势抵消了，**这是它最大的坑之一**（后面会讲）。

#### 5. Triton 后端：Inductor 的「笔」

Inductor 默认 GPU 后端是 Triton（下期会专门讲）。它不会为每个算子生成一个 kernel，而是把**一连串能被融化的算子合成一个 Triton 函数**。比如 `matmul → add bias → relu → add bias → dropout` 可能在 eager 下 launch 5 次、做 5 次 HBM 读写；在 Inductor 下变成一次 launch、中间结果全在 register/shared memory 里。

这条规律就是：**算子越碎、越大、GPU 利用率越低，compile 后收益越大**。Transformer 的 decode 阶段（每次只生成 1 个 token、算子极碎）是收益最大的经典场景，通常能提速 1.5–3 倍。

#### 6. 全图编译失败的 fallback 体系

`torch.compile` 是个**追求鲁棒性的编译器**——遇到它处理不了的代码模式不会直接崩，而是：

- **图中断（graph break）**：把图切两段，未编译段回 eager
- **重编译（recompilation）**：发现 shape 没匹配上时打回原图重编，每多一个不同 shape 就多一份代码
- **完全 fallback**：极端情况下整个模型回 eager，控制台会 WARNING

生产训练里**重编译是大忌**——每次重编译都触发额外几分钟。第一次跑前用 `torch._dynamo.config.cache_size_limit = 64` 把缓存调大、配合 profiler 监控重编译次数。

---

### 什么时候该用 / 不该用

**该用（绝大多数场景）：**

- Transformer / 大语言模型训练或推理，几乎全员受益
- 模型瓶颈在「小算子拼接」（attention、decode 阶段、RNN）
- 想做 `torch.export` / AOTInductor 来部署到生产环境
- 已有 eager 代码想零改造提速——一行 `torch.compile(...)` 就够

**谨慎或不该用：**

- **动态控制流非常多**的模型（带数据依赖分支、动态图结构）——图中断频繁，性能反而更差
- **显存极度紧张**：`torch.compile` 会保留激活图用于反向传播，可能让峰值显存抬升 10%–30%。遇到 OOM 优先靠 FSDP/gradient checkpointing，而不是 `torch.compile`
- **调试阶段**：第一次跑复杂模型时先关掉 compile，确认 eager 模式下逻辑正确
- **算子极重的模型**（比如一两次巨大 GEMM 撑爆显存）：compile 省不下多少时间，因为 compute bound

---

### 常见坑

1. **首次跑得特别慢就以为是 bug**
   ```python
   # ❌ 误判：第一次执行慢是因为它在编译
   model(x); model(x); model(x)
   # ↑ 前两次慢，第三次起飞。第一次的 wall time 不要算进 epoch 里
   ```

2. **`reduce-overhead` 模式下还自己在 dataset 里 `.pin_memory()` 拷贝数据**
   ```python
   # ❌ reduce-overhead 自动管理 cuda graph input buffer，
   #    你再手写数据搬运会失效甚至错乱
   #    解决：要么不用 reduce-overhead，要么完全依赖它自己的 input mutation
   ```

3. **每步迭代都改输入张量的 `data_ptr()`**
   ```python
   # ❌ torch.compile 的 CUDA Graph 模式会缓存指针，指针漂移就悄悄静默错误
   x = torch.randn(8, 16).cuda()
   for _ in range(100):
       x = torch.cat([x, torch.randn(1, 16).cuda()], dim=0)   # 指针每次都不一样
       out = model_opt(x)
   # ✅ 预先分配固定 buffer，复用同一份内存
   x = torch.zeros(max_len, 16).cuda()
   for _ in range(100):
       real_len = ...
       x[:real_len] = new_chunk
       out = model_opt(x)
   ```

4. **把 `model = torch.compile(model)` 加在已经 wrap 过的模型上**
   ```python
   # ❌ 嵌套 compile：日志里全乱、debug 噩梦
   model = torch.compile(model)
   model = torch.compile(model)   # 不要这做
   # ✅ 编译一次就够；想换 mode 就重新编
   ```

---

### 一句话总结

> **torch.compile 的本质是「让 Python 解释器从热路径上搬走」：Dynamo 把字节码变成 FX Graph，Inductor 把图变回 Triton 的融合 kernel，再叠加 CUDA Graph 把 launch 开销干掉——你只写一行代码，却让 GPU 用上了「编译器视角」的算力。**

---

### 今日练习

1. 拿一段你手头的 transformer 模型（哪怕 BERT-base 也行），分别用 `eager / torch.compile / mode="reduce-overhead" / mode="max-autotune"` 跑同一个 batch，对比吞吐——把四个数写在终端里
2. 故意写一个会触发图中断的函数（比如带 `if x.sum() > 0`），跑 `torch.compile`，打开 `TORCH_LOGS=+dynamo` 看日志里哪一行写了 `GRAPH BREAK`
3. 思考：为什么「图中断」之后的 eager 段会成为瓶颈？如果一张图被切成 N 段，性能瓶颈会出现在哪里？

<details>
<summary>练习 3 参考答案（点击展开）</summary>

图中断（graph break）发生在 Dynamo 遇到不能 trace 的 Python 操作——数据依赖的 if 分支、对象属性访问、第三方库的 C++ 调用等。中断点两侧被切成两段：被 trace 的部分走编译路径，剩下的部分回到 eager 模式继续执行。

性能瓶颈有三个来源：

1. **同步点增多**：break 处的 Python 控制流会引发 CPU-GPU 同步（因为需要 Python 决定下一步送哪个算子给 GPU），阻塞 GPU 流水线。CUDA Streams 的并发优势也因此丧失。

2. **图调度开销被部分抵消**：原本一次 launch 一整张图，速度很快；break 之后每段又得逐算子 launch，开销回升。

3. **数据复用机会消失**：eager 段里的中间张量会被 PyTorch 立刻物化、无法跨 break 边界优化。Inductor 原本能融合掉的算子现在不得不跨 break 边界反复读写 HBM，带宽变差。

工程上遇到图中断只能减少、不能彻底消除：把数据相关的判断从 Python 里剥出去（比如用 `.item()` 强迫同步拿标量、或把分支挪到 model 外面用权重硬编码子模型）、避免在热路径里调用外部 Python 库、给 Dynamo 加 `torch._dynamo.allow_in_graph(...)` 显式允许某些操作。

</details>

---

> 明天见。每天一个领域，不求全覆盖，只求真懂。
