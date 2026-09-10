# PyTorch 每日一课 · 第 004 期

## GPU 显存管理：Caching Allocator 的设计哲学

| 项目 | 内容 |
|---|---|
| 日期 | 2026-08-21 |
| 难度 | ⭐⭐⭐ |
| 前置知识 | PyTorch 张量基础、CUDA 设备概念、了解 `model.to('cuda')` |
| 预计阅读时间 | 12 分钟 |

---

## 这个领域解决什么问题

想象你开了一家自习室（GPU 显存），顾客（张量）来来往往：

- 有人只坐 5 分钟就走；
- 有人一坐一下午；
- 高峰期一下子来几十个人，每个人对座位大小要求还不一样。

如果每次来人都重新装修座位、每次走人都把桌椅砸了（`cudaMalloc` / `cudaFree`），成本会高得离谱。而且晚高峰可能根本抢不到新座位（Out Of Memory，OOM）。

PyTorch 的 **CUDA Memory Allocator（通常叫 Caching Allocator）** 本质上就是这家自习室的「座位管理系统」：

>  tensor 释放时不把显存真正还给 CUDA 驱动，而是先缓存起来，等下一个 tensor 需要时复用，从而避免频繁的 `cudaMalloc`/`cudaFree`。

它解决的不是「模型太大塞不下」的问题（那是 FSDP / 量化 / 梯度检查点要管的事），而是：

1. **减少分配开销**：频繁的 `cudaMalloc` 非常慢；
2. **避免显存碎片化**：大块连续显存更容易复用；
3. **提供可见的调试接口**：`torch.cuda.memory_stats()`、`reserved_memory`、`allocated_memory` 等；
4. **支撑高层优化**：CUDA Graph、AMP、Gradient Checkpointing 都依赖它。

---

## 核心思想：一块显存的两层账本

Caching Allocator 引入了两个关键概念，一定要分清楚：

| 概念 | 类比 | 含义 |
|---|---|---|
| **Allocated** | 真正被顾客占用的座位 | 当前活着的 tensor 占用的显存 |
| **Reserved** | 自习室已经租下来的总面积 | 已经向 CUDA 驱动申请、但不一定全被占用的显存 |
| **Reserved - Allocated** | 空着的备用座位 | 可被复用的缓存块 |

用 PyTorch 的术语：

- `torch.cuda.memory_allocated()` → 当前 tensor 实际占用；
- `torch.cuda.memory_reserved()` → Caching Allocator 向 CUDA 预申请的总量；
- 差值就是缓存池里的「待命」显存。

### 为什么这样设计？

CUDA 驱动的显存分配是**全局串行**的：所有进程、所有上下文都走同一个堆。频繁 `cudaMalloc` 不仅会触发同步，还会产生大量碎片。Caching Allocator 在 PyTorch 进程内部维护一个「私有缓存池」，把大块切成小块，需要时快速派发，释放时回收进池子，**只对驱动做少量大块申请**。

---

## 在 PyTorch 中怎么用

下面这段代码可直接运行（需要 GPU），它演示了显存从分配到释放、再到复用的过程。

```python
import torch

# 重置显存统计，确保数字干净
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

def log(prefix):
    """打印当前显存状态"""
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
    print(f"{prefix}: allocated={allocated:.2f} MB, reserved={reserved:.2f} MB")

log("初始")

# 1. 创建一个大张量：Allocator 会向 CUDA 申请一块显存
x = torch.randn(1024, 1024, 1024, device=device)  # 约 4 GB (float32)
log("创建 x 后")

# 2. 释放 x：allocated 下降，但 reserved 可能不变（进入缓存池）
del x
log("删除 x 后")

# 3. 再创建同样大小的 y：如果缓存池里有合适块，就不会触发新的 cudaMalloc
y = torch.randn(1024, 1024, 1024, device=device)
log("创建 y 后")

# 4. empty_cache()：把 reserved 中「未被使用」的部分真正还给 CUDA 驱动
#    注意：这会破坏缓存池，下一次分配可能变慢
del y
torch.cuda.empty_cache()
log("empty_cache 后")

# 5. 峰值显存：训练时最关心这个，它告诉你这一时刻最大占用是多少
peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
print(f"本次峰值 allocated: {peak:.2f} MB")
```

运行后你通常会看到：

- 删除 `x` 后 `allocated` 骤降，`reserved` 基本不变；
- 创建 `y` 时 `reserved` 几乎不变，因为复用了刚才释放的块；
- `empty_cache()` 后 `reserved` 也下降，但再分配时会更慢。

### 把显存分配可视化

```python
import torch

torch.cuda.reset_peak_memory_stats()

def allocate_and_free(size, n):
    """连续创建 n 个相同大小张量再删除，观察 reserved 的变化"""
    for i in range(n):
        t = torch.empty(size, device="cuda")
        # 保留最后一个，其余释放
        if i < n - 1:
            del t
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"size={size}, allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")

# 从小到大分配多种尺寸，模拟真实网络中不同层产生不同大小张量
allocate_and_free((1024, 1024), 5)      # 约 4MB * 5
allocate_and_free((2048, 2048), 5)      # 约 16MB * 5
allocate_and_free((4096, 4096), 5)        # 约 64MB * 5
```

这个脚本能帮你直观感受：Caching Allocator 会为不同尺寸维护多个「桶」（bucket），类似内存分配器中的 slab。小尺寸张量复用率高，超大尺寸可能每次都要重新向 CUDA 申请。

---

## 围绕该领域展开

显存管理不是孤立的 API，它是 PyTorch 性能栈的底座。理解了 Caching Allocator，就能串起一大片相关技术：

### 1. 与 CUDA Graph 的关系

CUDA Graph 要求显存地址在整个 graph 生命周期内稳定。Caching Allocator 可以把一段预留显存「钉住」，让 Graph 反复复用同一批分配，避免每次前向/反向都重新分配。上一期 CUDA Graph 本质上就是建立在这层稳定分配之上的加速手段。

### 2. 与 AMP / FP16 / BF16 的关系

混合精度训练减少的是 **allocated**（每个 tensor 占用的字节数变少），但 Caching Allocator 管理的是这些更小 tensor 的复用。两者一起作用：AMP 降需求，Caching Allocator 降分配开销。

### 3. 与 Gradient Checkpointing 的关系

Gradient Checkpointing 用时间换空间：前向时只保留检查点，反向时重算中间激活。它降低的是峰值 `allocated`，但会引入更多中间张量的创建/销毁，**更依赖 Caching Allocator 的高效复用**。如果缓存池碎片化严重，反而可能 OOM。

### 4. 与 Distributed Data Parallel (DDP) 的关系

DDP 每张卡上都有独立的 Caching Allocator 实例。`torch.nn.parallel.DistributedDataParallel` 的梯度桶（gradient buckets）需要稳定的显存块来 AllReduce。理解 allocator 有助于解释为什么 DDP 第一次前向后会有一段「热身期」——它在为 bucket 预分配缓存。

### 5. 与 `pin_memory` / DataLoader 的关系

DataLoader 的 `pin_memory=True` 把 CPU 页锁定内存里的数据异步拷贝到 GPU。拷贝完成后，GPU 端 tensor 由 Caching Allocator 接管。如果预取队列太长，Reserved 会升高，但 Allocated 不一定高。

### 6. 显存碎片与 OOM

这是最令人头疼的场景之一：

```
Reserved: 20 GB
Allocated: 12 GB
Free in reserved: 8 GB
```

但你要申请一个 6 GB 的连续块时却 OOM 了。为什么？因为那 8 GB 被切成了很多不连续的小块，**没有一段连续 6 GB 的可用空间**。这就是显存碎片化。`empty_cache()` 有时能缓解，但代价是后续分配变慢。

### 7. 环境变量调优

PyTorch 提供了一些环境变量来调节 Allocator 行为：

- `PYTORCH_CUDA_ALLOC_CONF`：最重要的一个，例如
  - `max_split_size_mb:512`：超过 512MB 的块不再被拆分成小桶，减少大模型碎片；
  - `expandable_segments:True`：用可扩展段（expandable segments）代替一次性大块，对动态形状更友好。

示例：

```bash
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True" python train.py
```

---

## 什么时候该用 / 不该用

### ✅ 该关注 Caching Allocator 的时候

- 训练时遇到 **OOM 但 reserved 明明还有剩余**（大概率是碎片问题）；
- 想要判断模型真正需要多少显存，而不是看 nvidia-smi 的「已占用」；
- 第一次迭代特别慢、后面变快，怀疑是 allocator 热身；
- 做性能分析（profiling）时，需要区分「计算慢」还是「分配慢」。

### ❌ 不要滥用

- 不要在每个 iteration 末尾都调用 `torch.cuda.empty_cache()`，这会清空缓存池，导致下一 iteration 重新 `cudaMalloc`，反而更慢；
- 不要把它当成「模型压缩」工具——它只能管理分配，不能让模型变小；
- 不要只看 `nvidia-smi` 判断显存压力，要看 `torch.cuda.memory_stats()` 里的 `allocated` 和 `reserved`。

---

## 常见坑

### 坑 1：把 `reserved` 当成「泄漏」

很多人发现训练结束后 `nvidia-smi` 还占着几 GB，以为是内存泄漏。其实多数情况下只是 Caching Allocator 把释放的 tensor 缓存起来了。只要 `allocated` 回到低位，就是正常的。

### 坑 2：频繁 `empty_cache()` 治百病

`empty_cache()` 只在**显存不足且确认后续不会再分配同样大小 tensor** 时有用。训练主循环里滥用它会让 allocator 失去意义。

### 坑 3：多进程里每个进程独立管理

用 `torch.multiprocessing` 或 DDP 时，每个 CUDA 上下文有自己的 allocator。父进程缓存的显存在子进程里不会复用，启动多进程前最好 `empty_cache()` 一次。

### 坑 4：动态形状导致缓存池爆炸

如果每个 batch 的序列长度、图像尺寸都不一样，allocator 需要维护很多不同尺寸的桶，reserved 会涨得很高。此时可以考虑：

- padding 到固定长度；
- 开启 `expandable_segments`；
- 或改用 torch.compile 做 shape 感知的 kernel 融合。

---

## 一句话总结

> PyTorch 的 Caching Allocator 不是为了让模型变小，而是让显存「借得快、还得省、能复用」；它是连接 CUDA 驱动与上层训练框架的「座位管理系统」，也是理解 OOM、碎片化和各种性能优化的基础。

---

## 今日练习

在你的 GPU 上运行下面的代码，然后回答：为什么 `del big` 之后 `allocated` 下降了，但 `reserved` 可能不变？什么情况下 `reserved` 也会下降？

```python
import torch

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

big = torch.randn(2048, 2048, 2048, device="cuda")  # 约 32 GB，请确认你的卡够用
print("allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("reserved: ", torch.cuda.memory_reserved() / 1024**3, "GB")

del big
print("\n删除 big 后:")
print("allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("reserved: ", torch.cuda.memory_reserved() / 1024**3, "GB")
```

<details>
<summary>参考答案</summary>

`del big` 只是让 Python 释放了对 tensor 的引用，tensor 不再被任何变量持有。Caching Allocator 不会立刻把这块显存还给 CUDA 驱动，而是把它放入内部缓存池，等待复用。因此 `allocated` 下降，`reserved` 保持不变。

`reserved` 下降的情况：
1. 调用 `torch.cuda.empty_cache()`，显式要求 allocator 把未使用的缓存还给驱动；
2. 长时间没有分配请求，allocator 在某些策略下可能主动归还（不保证）；
3. 进程退出，所有显存自然释放。

注意：如果之后马上创建同样大小的 tensor，`reserved` 不会变，因为直接从缓存池复用。
</details>
