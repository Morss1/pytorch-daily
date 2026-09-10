# PyTorch 每日一课 · 第 017 期

## CUDA Streams：GPU 上的并发模型

> **日期**：2026-09-11
> **难度**：⭐⭐⭐⭐
> **前置知识**：GPU 基本执行模型、CUDA Kernel 的概念、PyTorch 基础用法
> **预计阅读时间**：12 分钟

---

## 一、这个领域解决什么问题？

先想一个日常场景：你在训练一个模型，每个 batch 要做三件事——

1. 把数据从 CPU（ pageable 内存）拷到 GPU
2. GPU 上跑 forward/backward
3. 偶尔往 GPU 拷一些小的控制参数

如果没有 streams，这三件事是**串行**的：GPU 跑 kernel 的时候，数据拷贝通道（DMA 引擎）闲着；拷数据的时候，SM 计算单元闲着。这就像一个厨房里，厨师切菜时烤箱必须停着——不是烤箱坏了，而是没人安排它们同时工作。

GPU 的硬件其实有**多套独立的执行单元**：

- **SM（流式多处理器）**：负责计算，成千上万个核心
- **拷贝引擎（Copy Engine）**：独立的 DMA 硬件，专职 CPU↔GPU、GPU↔GPU 的数据搬运

CUDA Streams 解决的问题是：**如何把任务分配到不同的硬件队列上，让拷贝和计算重叠（overlap）起来，把 GPU 的每个部件都喂饱。**

一句话：streams 是 GPU 世界的「并发调度模型」，类似于 CPU 世界的多线程——但粒度更粗、语义更简单。

## 二、核心思想：队列、依赖与水管理论

### 2.1 Stream 就是一个 FIFO 任务队列

一个 stream 是一列按**提交顺序**执行的任务（kernel launch、memcpy、event）。同一个 stream 内部严格串行——先来后到，绝不乱序。

想要并发？把没有依赖的任务放进**不同的 stream**。硬件调度器看到两个 stream 里的任务用的是不同资源（一个用拷贝引擎、一个用 SM），就可能让它们真正同时跑。

### 2.2 经典类比：水管工和电工

装修房子时，水管工在厨房干活，电工可以在卧室布线——互不干扰，这就是多 stream 并发。但如果电工要接的是厨房水管旁边的插座，两人就得协调先后——这就是**跨 stream 同步**（event）存在的意义。

### 2.3 三种典型的并发收益

```
无重叠（单 stream）:
  [ H2D 拷贝 ][  计算  ][ D2H 拷贝 ]
  ────────────────────────────────> 时间

计算与拷贝重叠（双 stream）:
  stream A: [H2D 拷贝][H2D 拷贝][H2D 拷贝]
  stream B:    [计算1]  [计算2]  [计算3]
  ────────────────────────────────> 时间
```

1. **拷贝-计算重叠**：最常见。第 N 块数据在拷贝时，GPU 在算第 N-1 块。
2. **kernel 间并发**：多个小 kernel（比如两路独立的分支网络）如果各自占不满 SM，可以塞进不同 stream 同时跑。
3. **CPU-GPU 异步**：CUDA 本身就是异步 launch 的——CPU 提交任务后立刻返回去干别的，这正是 stream 语义的自然结果。

### 2.4 默认 stream 的特殊性

不指定 stream 时，PyTorch 的操作都跑在**默认 stream** 上（传统默认 stream 有全局同步语义：它会等待其他所有 stream 完成，其他 stream 也要等它）。所以"什么都没配"其实就是"全串行"。想获得重叠，必须显式开辟非默认 stream。

## 三、在 PyTorch 中怎么用

### 3.1 基础：创建 stream 并在其中执行操作

```python
import torch

# 假设有一块 GPU
device = "cuda"

# 创建一个非默认 stream
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)
out = torch.empty(1024, 1024, device=device)

# with 语句块内的所有 CUDA 操作都会提交到 s1
with torch.cuda.stream(s1):
    tmp = a @ a          # 矩阵乘法提交到 s1

# 另一路独立的计算提交到 s2
with torch.cuda.stream(s2):
    tmp2 = b @ b         # 与 s1 中的乘法可能并发执行

# 等待两个 stream 都完成（跨 stream 读取结果前必须同步！）
torch.cuda.synchronize()
out.copy_(tmp + tmp2)
print(out.shape)
```

### 3.2 经典套路：数据预取与计算重叠

这是 streams 最实用的场景——**预取下一个 batch**：

```python
import torch

class PrefetchLoader:
    """在计算当前 batch 的同时，把下一个 batch 拷到 GPU"""
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()  # 专用拷贝 stream
        self.next_batch = None

    def _prefetch(self, batch):
        # 在拷贝 stream 上异步搬运数据
        with torch.cuda.stream(self.stream):
            # .to() 在非默认 stream 里是异步的，CPU 不会等它完成
            return tuple(x.to("cuda", non_blocking=True) for x in batch)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            # 确保上一轮的预取已经完成（记录事件并等待）
            if self.next_batch is not None:
                torch.cuda.current_stream().wait_stream(self.stream)
            else:
                self.next_batch = self._prefetch(batch)
                continue

            yield self.next_batch          # 返回已就绪的 batch
            self.next_batch = self._prefetch(batch)  # 预取下一轮

# 用法：PrefetchLoader(DataLoader(...)) —— DataLoader 侧再配 pin_memory=True 效果最佳
```

关键点：`non_blocking=True` 让 CPU 不阻塞在拷贝上，配合独立 stream，数据搬运与计算就真正并行了。

### 3.3 事件（Event）：跨 stream 的依赖管理

```python
import torch

s_copy = torch.cuda.Stream()   # 负责拷贝
s_compute = torch.cuda.Stream() # 负责计算

data = torch.randn(8192, 8192, pin_memory=True)  # 锁页内存是异步拷贝的前提！

with torch.cuda.stream(s_copy):
    gpu_data = data.to("cuda", non_blocking=True)
    done = torch.cuda.Event()       # 在 s_copy 上记录一个事件
    done.record()

# 计算流等待拷贝流：只有 done 触发后，s_compute 的任务才开始
with torch.cuda.stream(s_compute):
    torch.cuda.current_stream().wait_event(done)
    result = gpu_data @ gpu_data

torch.cuda.synchronize()
print(result.sum().item())
```

`record_stream` 是另一个重要 API：告诉 allocator「这块张量还在别的 stream 里被使用，先别回收显存」——它是很多诡异 bug 的解药（见常见坑第 2 条）。

## 四、围绕该领域展开

streams 不是孤立功能，它在 PyTorch 生态里和一堆概念咬合在一起：

**① 与 Caching Allocator 的关系**（第 004 期讲过显存管理）
显存分配器是**按 stream 记账**的。一个 block 被分配时记录了所属 stream；如果张量在另一个 stream 上使用而不调用 `record_stream`，allocator 可能提前回收这块显存，导致数据被悄悄覆盖。理解 streams 是理解显存 bug 的前提。

**② 与 DataLoader / pin_memory 的关系**
异步拷贝（`non_blocking=True`）只在**锁页内存（pinned memory）**上真正异步——pageable 内存必须先拷到临时锁页区，CPU 会被迫同步等待。所以 `DataLoader(..., pin_memory=True)` 是 streams 发挥作用的物质基础。PyTorch 官方 DataLoader 的 `prefetch_factor` 负责 host 侧预取，GPU 侧预取则要靠你自己的 stream 逻辑。

**③ 与 NCCL / DDP 的关系**
分布式训练里，梯度 AllReduce 走的是独立的 NCCL stream。DDP 的 `no_sync`、梯度桶通信之所以能与反向传播重叠（backward 还在算，通信已经开跑），靠的就是 communication stream 与 computation stream 的分离。你在单卡上写 stream 代码的习惯，直接迁移到多卡场景。

**④ 与 CUDA Graph 的关系**（第 001 期讲过）
CUDA Graph 捕获时**所有操作必须在非默认 stream 上**（PyTorch 内部会自动处理，用 `torch.cuda.graph()` 上下文时它自己建 side stream）。Graph 的本质是「把一整串 stream 操作录下来一次性重放」，它建立在 stream 的语义之上。

**⑤ 与 torch.compile / TorchInductor 的关系**
Inductor 生成的代码里有一个 `cuda_async_tasks` 实验特性，编译器会自动分析依赖、把独立的 kernel 拆到多 stream 里并发执行——也就是说，**手工 stream 调度正在被编译器自动化**。了解 streams 的原理，才能看懂这类优化在做什么。

**⑥ 更底层的视角：硬件队列**
stream 在驱动层对应硬件队列（HW queue），kernel launch 实际是往队列里塞命令。队列有数量上限（通常 32~1024 个），event 是插在队列里的栅栏。再往下还有 CUDA Graph 把「提交 N 个任务」压缩成「提交 1 个图」，减少 launch 开销——这三者是同一层抽象的演进。

## 五、什么时候该用 / 不该用

**该用：**
- 数据加载是瓶颈：CPU→GPU 拷贝时间占比高（用 profiler 看到 `Memcpy HtoD` 很长）
- 模型里有多个独立分支（如双塔结构、MoE 的专家计算），每个 kernel 都很小、单个吃不满 GPU
- 推理服务想对多个请求做 batch 级并发
- 需要在训练主循环外维护异步的日志/指标拷贝（D2H）

**不该用：**
- 你的 kernel 又大又满（一个 matmul 就能吃满所有 SM），多 stream 无利可图，反而增加调度开销
- 任务之间有强依赖链条（A 的输出是 B 的输入），拆 stream 只会徒增同步成本
- 没有先做 profiling——"感觉慢"不等于"需要并发"，盲改 stream 是过早优化
- 代码里到处是隐式的全局状态（默认 stream 语义），强行插入多 stream 会引入难以复现的竞态 bug

## 六、常见坑

**坑 1：忘写 `synchronize()` 或 `wait_event`，读到脏数据**
在 stream A 里写张量，立刻在 stream B（或 CPU 侧）读它——拷贝还没完成，读到的是未初始化的旧显存。**症状**：loss 偶尔变成 NaN 或 0，且无法复现。跨 stream 传递张量前，必须用 event 建立依赖。

**坑 2：allocator 跨 stream 回收显存**
张量在 stream A 分配，却被 stream B 使用。A 上的操作结束后，allocator 认为 block 空闲可以复用，而 B 还在读它——数据被覆盖。**解法**：在 stream B 里调用 `tensor.record_stream(torch.cuda.current_stream())`，或者干脆把张量移动到使用它的 stream 上再操作。

**坑 3：pageable 内存 + `non_blocking=True` 的错觉**
以为加了 `non_blocking=True` 就是异步拷贝。错—— pageable 内存上它仍然会同步（内部要先拷到暂存锁页区）。**解法**：DataLoader 加 `pin_memory=True`，或手动 `tensor.pin_memory()`。

**坑 4：在 stream 上下文里创建张量，作用域外使用**
```python
with torch.cuda.stream(s):
    x = torch.randn(N, device="cuda")  # x 的分配记账在 s 上
y = x * 2  # 这行在默认 stream 上执行，与 s 无同步 → 竞态！
```
PyTorch 对部分这类场景有自动保护（记录 stream 防回收），但语义上依然是未定义行为的雷区。安全做法：跨 stream 边界传张量时，显式 `wait_stream` / `synchronize`。

## 七、一句话总结

**CUDA Streams 是 GPU 的任务队列模型：同一队列严格串行、跨队列可并发；用好它的全部秘密只有一句话——让没有依赖的任务，走不同的硬件通道。**

## 八、今日练习

<details><summary>今日练习</summary>

**题目**：写一个双 stream 流水线，模拟「边拷贝边计算」：准备 4 个大张量，要求第 k 个张量的 H2D 拷贝与第 k-1 个张量的 `matmul` 重叠执行。用 `torch.cuda.Event` 计时，对比「全部串行」与「流水线」两种版本的总耗时。

**参考答案**：

```python
import torch

N = 4096
NUM = 4

def serial_version(datas):
    """串行：拷贝完一个算一个"""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for d in datas:
        g = d.to("cuda", non_blocking=True)
        _ = g @ g
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end)

def pipelined_version(datas):
    """流水线：拷贝 stream 与计算 stream 分离"""
    copy_s = torch.cuda.Stream()
    comp_s = torch.cuda.Stream()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()

    prev_event = None
    for d in datas:
        with torch.cuda.stream(copy_s):
            g = d.to("cuda", non_blocking=True)
            copied = torch.cuda.Event()
            copied.record()          # 记录“拷贝完成”事件
        with torch.cuda.stream(comp_s):
            # 计算流等待本次拷贝完成
            comp_s.wait_event(copied)
            _ = g @ g
            prev_event = torch.cuda.Event()
            prev_event.record()
        if prev_event is not None:
            # 拷贝流也无需等计算——两个流真正独立滚动
            pass
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end)

datas = [torch.randn(N, N, pin_memory=True) for _ in range(NUM)]
t1 = serial_version(datas)
t2 = pipelined_version(datas)
print(f"串行版本:   {t1:.2f} ms")
print(f"流水线版本: {t2:.2f} ms")   # 预期 t2 < t1，拷贝与计算重叠了
```

**思考延伸**：把 `pin_memory=True` 去掉再跑一次，流水线收益会大幅缩水甚至消失——去验证坑 3。再用 `torch.profiler` 的 timeline 视图（第 014 期）看两个 stream 的任务条是否真的在时间轴上错开重叠。

</details>

---

*下期预告：数据管线（DataLoader、prefetch、内存映射）——正好是本期预取思路的完整版。*
