# PyTorch 每日一课 · 第 006 期

## Distributed Data Parallel（DDP）：多卡训练的基石

> **日期**：2026-08-26  
> **难度**：⭐⭐⭐⭐（4/5）  
> **前置知识**：熟悉 `nn.Module` 和基本训练循环，了解 GPU 与 CPU 的区别  
> **预计阅读时间**：15 分钟

---

## 这个领域解决什么问题

你有一张 24GB 的 RTX 4090，但模型有 30GB；或者训练一个 batch 需要 16GB，你想跑 batch_size=64 来稳定收敛——单卡已经装不下了。

**Distributed Data Parallel（DDP）** 的核心目的就一句话：

> 把「一份模型 + 一份数据」复制到 N 张 GPU 上，每张卡只处理 1/N 的数据，梯度通过集体通信自动同步，让 N 张卡协同完成一次完整的参数更新。

换句话说，DDP 解决的是**「横向扩展」**问题：不是把模型塞进一张更大的卡里，而是让多张卡一起干活。

这和「模型并行」（把模型拆开放到不同卡上）是两条完全不同的路线。DDP 属于**数据并行**，是目前最常用、最成熟的多卡训练方案。

---

## 核心思想：用类比讲清楚

想象一个工厂里有 10 个工人（GPU），每个工人手里都有一份完全相同的操作手册（模型参数）。

现在来了 1000 个零件（数据）。主管（DDP 的 launcher）把零件分成 10 堆，每堆 100 个，分给每个工人。

每个工人独立加工自己的 100 个零件，算出「这批零件的加工误差」（本地梯度）。然后所有工人开一个**同步会议**（AllReduce），把各自的误差汇总取平均，得到一个**统一的修正方案**（同步后的平均梯度）。最后每个工人按照这个统一方案，各自更新自己手中的操作手册。

关键点：**每个工人的手册始终保持一致**。所以 DDP 要求每张卡的模型参数在初始化时完全相同，更新后也完全相同。

这个「同步会议」就是 DDP 的灵魂——它用一种叫 **Ring AllReduce** 的算法高效完成梯度聚合，不需要把梯度全部集中到某一张卡上。

---

## 在 PyTorch 中怎么用

### 1. 最简启动脚本

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def setup(rank, world_size):
    """初始化进程组：让当前进程知道自己是 N 张卡中的第几张。"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # backend='nccl' 是 NVIDIA GPU 上最高效的通信后端
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_training(rank, world_size):
    """每个进程都会执行这个函数，rank 是当前进程的编号（0 ~ world_size-1）。"""
    setup(rank, world_size)

    # 把模型放到对应的 GPU 上
    model = torch.nn.Linear(10, 1).to(rank)
    # 用 DDP 包装：PyTorch 会自动处理前向/反向和梯度同步
    ddp_model = DDP(model, device_ids=[rank])

    # 数据集：每个进程只看到自己的 1/world_size 数据
    dataset = torch.randn(1000, 10)  # 1000 条假数据
    labels = torch.randn(1000, 1)
    # DistributedSampler 负责把数据切成 world_size 份
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(list(zip(dataset, labels)), batch_size=32, sampler=sampler)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(3):
        sampler.set_epoch(epoch)  # 关键：每个 epoch 重新打乱采样顺序
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(rank)
            batch_y = batch_y.to(rank)

            optimizer.zero_grad()
            output = ddp_model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()          # DDP 在这里自动完成梯度 AllReduce
            optimizer.step()

    cleanup()


def main():
    world_size = torch.cuda.device_count()  # 自动检测有几张 GPU
    # 用 spawn 启动 world_size 个进程，每个进程对应一张卡
    mp.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
```

### 2. 命令行启动（推荐）

上面的 `mp.spawn` 写法适合单机多卡。更常见的做法是用 `torchrun`（PyTorch 1.9+）在命令行启动：

```bash
# 单机 4 卡
torchrun --standalone --nproc_per_node=4 train.py

# 多机多卡（2 台机器，每台 8 张卡）
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=12355 train.py
```

对应 `train.py` 的入口简化为：

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group("nccl")  # torchrun 会自动设置环境变量
    rank = dist.get_rank()           # 当前进程编号
    local_rank = int(os.environ["LOCAL_RANK"])  # 当前机器上的 GPU 编号

    model = MyModel().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    # ... 训练循环 ...

if __name__ == "__main__":
    main()
```

### 3. 核心 API 一览

| API / 概念 | 作用 |
|-----------|------|
|`dist.init_process_group()`|建立进程间的通信组，所有进程加入同一个「世界」|
|`DDP(model, device_ids=...)`|包装模型，自动拦截 `forward()` 和 `backward()` 做梯度同步|
|`DistributedSampler`|把数据集切成 N 份，保证每张卡数据不重叠|
|`sampler.set_epoch(epoch)`|每轮重新设置随机种子，保证不同 epoch 的切分不同（避免固定 bias）|
|`torchrun`|官方推荐的启动器，自动设置 `RANK`、`WORLD_SIZE` 等环境变量|

---

## 围绕 DDP 展开：相关概念与关联技术

### 1. AllReduce：DDP 底层的通信原语

DDP 的梯度同步不是「把梯度发到主卡再广播」，而是直接在进程间做 **Ring AllReduce**。

- **Reduce**：把所有进程的同一个梯度加起来。
- **AllReduce**：不仅加起来，还要把结果发回给每个进程。

Ring AllReduce 的巧妙之处在于：每张卡只需要和左右邻居通信，像传接力棒一样，每轮传递 2×(N−1) 次，通信量和卡数无关，只和数据量有关。这是 DDP 能 scale 到上百张卡的理论基础。

PyTorch 底层通过 `torch.distributed.all_reduce()` 暴露这个操作，DDP 在 `backward()` 时自动调用它。

### 2. Gradient Bucketing：隐藏的大杀器

DDP 不是等所有层的梯度都算完再一次性 AllReduce，而是把梯度**按桶分批**同步。当某个桶里的梯度全部就绪，就立刻发起 AllReduce，和后续层的反向传播**重叠执行**。

这叫做 **Gradient Bucketing**，是 DDP 性能远超早期 DataParallel（DP）的关键原因。DP 把所有梯度往主卡一扔，通信和计算串行，GPU 大量空等。

> 桶的大小可以通过 `DDP(bucket_cap_mb=...)` 调节，默认 25MB。通信带宽差的网络可以调小，带宽好的可以调大。

### 3. Process Group：不只是「世界」

`init_process_group` 创建的是默认的 **World Group**。但你可以创建子组，比如：

- **Pipeline Parallel** 中，同一条 pipeline stage 的卡组成一个 group 做 DDP。
- **Tensor Parallel** 中，行并行/列并行的卡各自组成 group。

```python
from torch.distributed.distributed_c10d import new_group
# 把 rank 0,1 和 rank 2,3 分成两个组
group01 = new_group([0, 1])
group23 = new_group([2, 3])
```

### 4. DDP vs DataParallel（DP）：别再混用

| 特性 | `nn.DataParallel` (DP) | `DistributedDataParallel` (DDP) |
|------|------------------------|--------------------------------|
| 进程数 | 单进程，多线程 | 多进程，每个进程一张卡 |
| 梯度同步 | 主卡收集再广播（串行） | Ring AllReduce（并行重叠） |
| 性能 | 2 张卡以上反而可能变慢 | 线性扩展，8 卡接近 8x |
| 推荐度 | ❌ 已废弃，不建议使用 | ✅ 官方唯一推荐 |

**简单记忆**：看到 `DataParallel` 就替换为 `DDP + torchrun`。

### 5. DDP 与 FSDP 的关系

DDP 每张卡都存一份完整的模型参数和优化器状态。当你有 70B 的模型，单卡放不下时，DDP 就无能为力了。

**FSDP（Fully Sharded Data Parallel）** 是 DDP 的进化版：
- 参数、梯度、优化器状态都被**分片**（shard）到所有卡上。
- 前向/反向时按需从其他卡 `all_gather` 参数。
- 本质上仍是数据并行，但通过分片大幅降低了单卡显存占用。

> DDP 是「每个工人有一本完整手册」，FSDP 是「每个工人只保管手册的 1/N 页，需要时互相借阅」。

### 6. DDP 与混合精度（AMP）

DDP 和 `torch.cuda.amp` 天然兼容。但注意 `GradScaler` 的 `unscale_` 和 `step()` 需要在每个进程上独立执行，DDP 只负责梯度同步，不碰优化器状态。

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = ddp_model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()   # DDP 同步的是 scaled 梯度
scaler.step(optimizer)
scaler.update()
```

---

## 什么时候该用 / 不该用

### ✅ 该用 DDP
- 你有 2 张及以上的 GPU，想加速训练。
- 单卡 batch_size 太小（比如只能跑 4），DDP 可以 N 卡一起跑到 32/64，提升统计稳定性。
- 数据量很大，单卡 epoch 时间太长。

### ❌ 不该用 DDP（或只用 DDP 不够）
- 单卡就能装下模型和数据——没必要引入分布式复杂度。
- 模型太大，单卡放不下参数。这时需要 **FSDP** 或 **模型并行**（Pipeline/Tensor Parallel）。
- 通信带宽极差（比如跨数据中心）。DDP 的 AllReduce 会受网络瓶颈拖累，效率暴跌。
- 你需要极致的显存优化。DDP 不做分片，考虑 FSDP + Gradient Checkpointing。

---

## 常见坑

### 坑 1：忘记 `sampler.set_epoch(epoch)`

如果不调用，每个 epoch 的采样顺序完全一样，随机性丧失，模型可能学不到数据分布的全貌，收敛变差。

```python
# ❌ 错误：忘记 set_epoch
for epoch in range(EPOCHS):
    for x, y in dataloader:  # 每个 epoch 顺序一样
        ...

# ✅ 正确
for epoch in range(EPOCHS):
    sampler.set_epoch(epoch)  # 重新设置随机种子
    for x, y in dataloader:
        ...
```

### 坑 2：在 DDP 模型外面直接访问 `model.parameters()`

`ddp_model = DDP(model)` 后，`ddp_model.module` 才是原始模型。保存 checkpoint 时要存 `ddp_model.module.state_dict()`，加载时也要先 `model.load_state_dict()` 再包 DDP。

```python
# ❌ 错误：保存了 DDP 包装层的 state_dict
torch.save(ddp_model.state_dict(), "ckpt.pt")
# 加载到非 DDP 模型时会报 key 不匹配（多了 module. 前缀）

# ✅ 正确
torch.save(ddp_model.module.state_dict(), "ckpt.pt")
model.load_state_dict(torch.load("ckpt.pt"))
ddp_model = DDP(model, device_ids=[rank])
```

### 坑 3：BatchNorm 在多卡上的统计偏差

DDP 每张卡只看到局部 batch，BatchNorm 的 running_mean/variance 在不同卡上可能不一致。PyTorch 的 `SyncBatchNorm` 可以解决这个问题：

```python
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
ddp_model = DDP(model, device_ids=[rank])
```

它会跨卡同步 BatchNorm 的均值和方差，但会引入额外通信开销。小 batch 时尤其需要开启。

### 坑 4：`find_unused_parameters=True` 的隐藏代价

如果模型有某些参数在某些 iteration 中不参与计算（比如动态图、条件分支），DDP 默认会报错。加上 `find_unused_parameters=True` 可以解决：

```python
ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
```

但这会**显著降低性能**，因为 DDP 需要遍历计算图找出哪些参数没用到。尽量让模型的前向路径固定，避免开启这个选项。

---

## 一句话总结

> DDP 不是把模型拆开，而是把数据拆开——每张卡跑同样的模型、不同的数据，梯度通过 Ring AllReduce 高效同步，让 N 张 GPU 像一张更大的 GPU 一样工作。

---

## 今日练习

**场景**：你有一个 4 层 MLP（输入 128 → 256 → 256 → 10），数据量 10000 条。请写出一段**能用 `torchrun --nproc_per_node=2` 启动**的完整脚本，要求：
1. 使用 `DistributedSampler` 切分数据；
2. 每个 epoch 调用 `set_epoch`；
3. 保存 checkpoint 时去掉 `module.` 前缀；
4. 只在 rank 0 进程打印 loss（避免日志刷屏）。

<details>
<summary>参考答案</summary>

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 模型
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[local_rank])

    # 数据
    X = torch.randn(10000, 128)
    Y = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(X, Y)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)
        for x, y in loader:
            x, y = x.to(local_rank), y.to(local_rank)
            optimizer.zero_grad()
            out = ddp_model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.4f}")

    # 保存：只让 rank 0 存，且去掉 module. 前缀
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "mlp_ckpt.pt")
        print("Checkpoint saved.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

**启动命令**：
```bash
torchrun --standalone --nproc_per_node=2 train.py
```

</details>
