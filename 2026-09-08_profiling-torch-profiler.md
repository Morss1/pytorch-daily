# PyTorch 每日一课 · 第 014 期

## Profiling：用 torch.profiler 找性能瓶颈

> 日期：2026-09-08
> 难度：⭐⭐⭐
> 前置知识：PyTorch 基本训练流程、对 GPU 训练有初步接触
> 预计阅读时间：12 分钟

---

## 一、这个领域解决什么问题

你有没有遇到过这样的场景——

训练一个模型，GPU 利用率忽高忽低，显卡风扇一会儿转一会儿停；或者换了个更大的 batch size，速度反而没提升；或者明明买了 8 张 A100，训练速度却只有单卡的 3 倍；又或者领导问你"这个推理服务为什么 P99 延迟 200ms"，你只能说"感觉是模型太大"。

**感觉不是证据。** 优化的第一步永远是回答这个问题：**时间到底花在哪了？**

这就是 Profiling（性能剖析）领域要解决的问题。它不是教你调参数，而是给你一双"透视眼"：

- 每一行代码在 CPU 和 GPU 上各花了多少毫秒？
- 是算子（kernel）本身慢，还是 GPU 在等待数据、等待 CPU 发指令？
- 显存是谁吃掉的？峰值出现在哪一步？
- 通信和计算有没有重叠？DataLoader 是不是瓶颈？

没有 Profiling 的优化是赌博，有 Profiling 的优化是手术。**这个领域的方法论是：先测量、再假设、再验证，而不是凭直觉乱改。**

## 二、核心思想：时间是花在哪的？

要理解 Profiler，先要理解一个训练迭代的时间都由什么构成：

```
一个 iteration = CPU 时间 + GPU 时间 + 等待时间
```

关键洞察在于：**CPU 和 GPU 是异步工作的**。你在 Python 里调用 `y = model(x)`，CPU 只是"排队"了一个任务就立刻返回了，GPU 在后台慢慢算。所以：

- **CPU 时间**：Python 开销、算子调度、kernel launch、数据预处理
- **GPU 时间**：真正跑 kernel 的时间
- **等待时间**：GPU 闲着等 CPU 发指令（CPU bound），或 CPU 闲着等 GPU 算完（GPU bound）

一个经典误区：看到 GPU 利用率 100% 就以为没问题。其实 `nvidia-smi` 的利用率只表示"有 kernel 在跑"，不表示 GPU 在满负荷干活——跑一个极小的 kernel 也算 100%。

Profiler 的核心就是把这些时间**归因到具体的事件**上，让你看到：

1. **算子级视图**：每个 kernel / 每个 PyTorch op 的耗时排名（"热点"）
2. **时间线视图（trace）**：CPU 线程和 GPU stream 在时间轴上的活动，一眼看出"空泡"（gap）
3. **聚合视图**：按算子类型分组，看 CPU 开销 vs GPU 计算的比例

一个好的类比：Profiler 就像医院的 CT 扫描。病人说"我浑身不舒服"（训练慢），CT 一扫，发现病灶在左肺一个小点（比如 80% 时间花在 `copy_` 这个 H2D 拷贝上）。对症下药，药到病除。

## 三、在 PyTorch 中怎么用

### 3.1 基础用法：profile 上下文管理器

```python
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

# 一个简单模型
model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
).cuda()

x = torch.randn(512, 1024, device="cuda")

# 先跑几个 iteration 预热（让 cudnn autotune、cache 等稳定下来）
for _ in range(5):
    model(x)

# 正式剖析：CPU 和 GPU 活动都记录
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    for _ in range(10):
        model(x)

# 按总耗时排序，打印前 15 个热点
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=15
))
```

输出表格长这样（节选）：

```
-----------------  ------------  ------------  ------------  ------------
             Name    Self CPU      Self CUDA     CUDA total     # of Calls
-----------------  ------------  ------------  ------------  ------------
    aten::linear         ...          ...            ...             20
    gemm_kernel          ...          ...         45.2%             20
        aten::relu       ...          ...            ...             10
  Memcpy HtoD (...)     ...          ...         12.8%             10
-----------------  ------------  ------------  ------------  ------------
```

关键列含义：
- **Self CUDA**：这个算子自己（不含子算子）在 GPU 上的时间
- **CUDA total**：含子算子的总 GPU 时间
- **# of Calls**：调用次数——次数异常多往往意味着 kernel 太碎

### 3.2 导出 Chrome Trace，用肉眼看时间线

表格只能告诉你"谁慢"，**trace 能告诉你"为什么慢"**——空泡、串行、锁等待全在里面：

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,   # 记录张量形状：排查动态形状带来的重编译
) as prof:
    model(x)

# 导出 Chrome Trace 文件
prof.export_chrome_trace("trace.json")
# 用 Chrome 打开 chrome://tracing，加载这个文件，即可看到可视化时间线
```

时间线上你会看到：CPU 线程一排排"打点"（launch kernel），GPU stream 一排排"色块"（执行 kernel）。**色块之间的大段空白 = GPU 在等待**，这就是你要消灭的东西。

### 3.3 剖析完整的训练循环（含 DataLoader 和反向传播）

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(torch.randn(1024, 64), torch.randn(1024, 1))
loader = DataLoader(dataset, batch_size=128, num_workers=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(   # 剖析节奏控制
        wait=1,   # 先跳过 1 步（不记录）
        warmup=1,  # 1 步预热（记录但结果不用于统计）
        active=3,  # 正式记录 3 步
        repeat=1,  # 重复一轮
    ),
    on_trace_ready=lambda p: p.export_chrome_trace(f"step_{p.step_num}.json"),
) as prof:
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x.cuda()), y.cuda())
        loss.backward()
        optimizer.step()
        prof.step()   # 告诉 profiler 一个 iteration 结束了
```

`schedule` 的意义：剖析本身有开销（可能 10%~30% 减速），跳过前几步可以避开 autotune 等一次性成本，让样本更真实。

### 3.4 显存剖析：揪出吃显存的元凶

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,          # 开启显存追踪
    with_stack=True,              # 记录分配处的 Python 调用栈
) as prof:
    out = model(torch.randn(256, 1024, device="cuda"))
    out.sum().backward()

print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage", row_limit=10
))
# 可以看到每个算子分配/释放了多少显存，
# 配合 with_stack=True 可以用 export_memory_timeline 找到具体代码行
```

### 3.5 新一代：TORCH_LOGS 与 kineto / HTA

PyTorch 2.x 生态里还有配套工具值得一提：

```python
# TensorBoard 插件：交互式查看 trace
from torch.profiler import tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler("./log"),
) as prof:
    for _ in range(3):
        model(x)

# 然后：tensorboard --logdir=./log
# 打开 TensorBoard 的 PyTorch Profiler 标签页，有火焰图和统计视图
```

多卡训练可以用 NVIDIA 的 **HTA (Holistic Trace Analysis)** 分析几十个 rank 的 trace，快速定位"最慢的那张卡在等谁"。

## 四、围绕该领域展开：它在 PyTorch 生态中的位置

### 4.1 整个工具栈的分层

```
你看到的：    loss.backward()
PyTorch 层：  aten::linear → at::cuda::blas::gemm
CUDA 层：     cublasGemmEx → kernel 在 SM 上执行
硬件层：      tensor core 在算 FMA
```

不同工具剖析不同层：

- **torch.profiler**（基于 Kineto）：覆盖 PyTorch 算子 + CUDA kernel + NVTX 区间，**日常首选**
- **NVIDIA Nsight Systems**：系统级 trace，含 NCCL、CUDA stream、内存拷贝的精确时序，剖析分布式/流水线重叠的终极武器；PyTorch 里用 `torch.cuda.nvtx.range_push("forward")` 打的标记可以直接在 Nsight 里看到
- **NVIDIA Nsight Compute**：单 kernel 深度剖析（寄存器占用、共享内存、occupancy），写 Triton kernel 调优时配合使用（呼应第 008 期）
- **py-spy / cProfile**：纯 Python 层，剖析 DataLoader 的 CPU 瓶颈时有用
- **CUDA Events / `torch.cuda.synchronize` 计时**：手动的、最轻量的定点测量

### 4.2 和 torch.compile 的关系

torch.compile（第 003 期）的加速效果必须有 Profiler 来验证：compile 前后各跑一次 profile，对比 kernel 数量（通常会大幅减少，因为算子融合了）和空泡长度。`record_shapes=True` 记录的形状信息，正是 Dynamo 用来做 dynamic shape guard 的输入——看到 trace 里大量 `guard` / graph break 相关条目，就知道动态形状在拖累编译。

### 4.3 和分布式训练的关系

多卡场景下 Profiler 的价值更大：8 卡训练慢，trace 一看发现 AllReduce 全程和反向传播**串行**执行——这就是该上梯度分桶/overlap 通信（DDP 内置，第 006 期）的信号。NCCL kernel（`nccl:all_reduce`）在 trace 里的耗时和位置，是判断通信效率的直接证据。

### 4.4 相关的开源生态

- **PyTorch Kineto**：profiler 的底层实现（libkineto），同时是 PyTorch 和 NVIDIA 工具链的桥梁
- **TensorBoard Profiler 插件**：可视化 trace 的标准前端
- **Holistic Trace Analyzer (HTA)**：Meta 开源的多卡 trace 分析器
- **torch.utils.benchmark**：不是 profiler，而是"计时器"——`Timer` 提供统计上稳健的微基准测试（自动同步、多次采样、ADCI），比手写 `time.time()` 靠谱得多

### 4.5 Profiling 的开销本身也是学问

Profiler 记录每个事件都有成本（栈展开、字符串拷贝、CUPTI 回调），所以：

- 永远只在**采样窗口**内开 profile，不要全程开着跑几天
- 剖析结果的**绝对耗时**会被抬高，关注**相对占比**更可靠
- 需要"无损"长时间观测时，用轻量方案：`torch.cuda.Event` 定点打点 + 定期采样 `nvidia-smi`

## 五、什么时候该用 / 不该用

**该用：**
- 任何优化动手之前——先建立 baseline，明确瓶颈类别（CPU bound / GPU bound / IO bound / 通信 bound）
- 训练吞吐不达标，怀疑 DataLoader 喂不饱 GPU
- torch.compile / 半精度 / 算子替换之后，验证优化是否真的生效
- 推理服务延迟高，需要定位是 prefill 慢还是 decode 慢、是计算还是内存搬运
- 训练中途 OOM，需要找到显存峰值发生在哪个算子

**不该用 / 谨慎用：**
- 问题原因已经明确（比如明知 batch size 太小 kernel 没吃饱），直接改，改完再 profile 验证
- 生产环境长时间挂着 profiler——开销可观，trace 文件也能到几个 GB
- 只看一个 iteration 就下结论——第一个 iteration 包含编译、缓存、autotune，样本严重失真
- 微基准测 kernel 速度用 profiler——那是 `torch.utils.benchmark.Timer` 的领地

## 六、常见坑

**坑 1：忘了 warmup 和 synchronize，测出假时间**

```python
import time
torch.cuda.synchronize()           # 记住：起点先同步
t0 = time.time()
for _ in range(100):
    model(x)
torch.cuda.synchronize()           # 终点再同步，否则 GPU 还没跑完就"计时结束"
print(time.time() - t0)
```

CUDA 是异步的，不同步的计时约等于只测了 CPU 排队的时间。同理，profiler 的 `schedule(wait=1, warmup=1)` 就是为了避开这类污染。

**坑 2：把 profiler 报的绝对时间当真理**

分析期间每个事件都要记录元数据，总时间普遍虚高 10%~30%。正确姿势是看**占比和排名**："H2D 拷贝占了 GPU 总时间的 40%" 是可信结论，"这个 kernel 花了 3.2ms" 不太可信。

**坑 3：trace 文件巨大、Chrome 打不开**

`active` 步数设太多、模型很大、开了 `with_stack=True`，trace 轻松上 GB。控制采样窗口（schedule）、只剖析 2~3 个 iteration、必要时只开 `ProfilerActivity.CUDA`。超大文件可以换用 Perfetto UI（ui.perfetto.dev）打开，比 chrome://tracing 能扛。

**坑 4：只看表格不看时间线**

表格告诉你 GEMM 占 60%，但时间线可能告诉你：剩下的 40% 里 GPU 有大段空白，因为 CPU 在一个 `for` 循环里逐元素地 launch 小 kernel（典型的 Python 开销瓶颈）。**结论完全不同：前者要换更快的 GEMM，后者要减少 Python 循环 / 上 torch.compile。** 表格找热点，时间线找空泡，两个都要看。

## 七、一句话总结

**优化之前先 profile——表格找热点，时间线找空泡，让数据告诉你时间花在了哪里，而不是让直觉带你赌博。**

## 今日练习

<details><summary>今日练习</summary>

**题目：** 写一个脚本，剖析下面这段"故意写慢"的训练循环，找出三个性能问题并修复它们。修复后重新 profile，对比 GPU 总时间的变化。

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512)
).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for step in range(20):
    optimizer.zero_grad()
    total = 0.0
    # 问题藏在这个循环里
    for i in range(64):
        x = torch.randn(32, 512, device="cuda")   # 每次都在 GPU 上生成数据
        y = model(x)
        loss = loss_fn(y, x)
        total = total + loss                      # 标量逐步累加，串起整条计算图
    total.backward()
    optimizer.step()
```

**参考答案：**

Profile 后你会发现三个典型问题：

1. **串行长计算图**：`total = total + loss` 把 64 份 loss 串成链，反向传播要沿链逐个回溯，autograd 图节点数是批处理写法的 64 倍。修复：`loss = loss_fn(model(x_all), y_all)` 一次算整个 batch。
2. **小 batch 打不满 GPU**：32×512 的 GEMM 对 GPU 来说太小，kernel launch 开销占比高。修复：合成一个大 batch 一次前向。
3. **数据在 GPU 上逐步生成**：`torch.randn(..., device="cuda")` 每次都要 launch 一个随机数 kernel（且 `torch.manual_seed` 不可控性差）。修复：一次生成全部数据，或数据放在 CPU 由 DataLoader 喂入。

修复后版本：

```python
x_all = torch.randn(64 * 32, 512, device="cuda")

for step in range(20):
    optimizer.zero_grad()
    loss = loss_fn(model(x_all), x_all)
    loss.backward()
    optimizer.step()
```

对比 profile 结果：GPU 总时间通常下降一个数量级，且 `randn` kernel 从 64 次降到 1 次，autograd 相关的 CPU 开销几乎消失。**这就是 profile → 假设 → 修复 → 再 profile 验证的完整闭环。**

</details>
