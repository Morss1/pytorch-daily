# PyTorch 每日一课 · 第 018 期

## 数据管线：DataLoader、prefetch、内存映射

> **日期**：2026-09-14
> **难度**：⭐⭐⭐
> **前置知识**：会用 Dataset / DataLoader 的基础 API，了解 GPU 训练的基本流程
> **预计阅读时间**：12 分钟

---

## 一、这个领域解决什么问题

很多人入门 PyTorch 时有一个根深蒂固的错觉：**训练慢 = GPU 不行或模型太大**。于是拼命换卡、换混合精度、上分布式。但真实世界里，大量训练任务的速度瓶颈根本不在 GPU，而在**喂数据的管道**上——GPU 每秒能吞几万个 batch 的计算，而数据还在磁盘上躺着、还在 CPU 上排队等解码、还在单线程里慢悠悠地搬。

看一个典型的性能画像（用上一期讲的 torch.profiler 抓的）：

- GPU 计算一个 step 用时 50ms
- 但每 step 间隔了 120ms 的空白

那 120ms 里 GPU 在干什么？**什么都不干，在等数据**。这就是「GPU 饥饿」（GPU starvation）。你的百万显卡在空转，电费照烧，进度条照慢。

数据管线这个领域要回答的问题只有一个：**如何让数据供给的速度永远快于数据消费的速度**。它横跨了存储格式、文件系统、CPU 并行、内存管理、GPU 传输五个层面，是工程味最浓、也最容易被忽视的一块 PyTorch 技术。

---

## 二、核心思想：流水线与局部性

### 2.1 一个类比：餐厅后厨

把训练想象成一家餐厅：

- **GPU** 是主厨，炒一道菜（一个 step）只要 1 分钟
- **数据加载** 是备菜：洗菜、切菜、摆盘
- **DataLoader** 是传菜系统

如果主厨每炒完一道菜都要自己跑去洗菜切菜，那这家餐厅的出菜速度完全被备菜拖死。解决办法显而易见：**主厨炒菜的同时，其他人并行备菜**——这就是 **prefetch（预取）**：在 GPU 算第 N 个 batch 时，CPU 后台已经在准备第 N+1、N+2 个 batch。

进一步，如果洗菜工只有一个人（单进程加载），切菜再快也可能跟不上。**多雇几个洗菜工**——这就是 `num_workers` 多进程加载。

再进一步，如果食材仓库在城外，每次取货都要跑一趟（读磁盘），那不如把常用食材先搬到后厨货架上（内存），甚至直接预处理好半成品（**内存映射 / 预处理缓存**）。

### 2.2 一条数据的一生

一个样本从磁盘到 GPU，要经过这样一条流水线：

```
磁盘/对象存储
   │  ① 读取（I/O，最慢的一环，常以毫秒计）
   ▼
内存中的原始字节
   │  ② 解码（JPEG/PNG 解码成像素数组，CPU 密集）
   ▼
numpy / PIL 对象
   │  ③ 预处理（crop、resize、normalize、augmentation）
   ▼
torch.Tensor（CPU）
   │  ④ collate：拼 batch（stack 成 [B, C, H, W]）
   ▼
torch.Tensor（CPU，pin_memory）
   │  ⑤ H2D 拷贝：CPU 内存 → GPU 显存
   ▼
GPU 上的 batch ← 训练循环消费
```

数据管线的全部优化手段，本质上都是在问：**这条链上哪一环最慢？能不能把几环重叠起来（并行）？能不能把慢的一环整体绕开（缓存/换格式）？**

---

## 三、在 PyTorch 中怎么用

### 3.1 基础形态：Dataset + DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader

# Dataset 定义"单个样本怎么取"，DataLoader 定义"怎么批量喂"
class MyDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 这里通常是最贵的一步：磁盘读取 + 解码
        img = load_and_decode(self.paths[idx])        # 假设返回 [3, 224, 224] 张量
        return img, self.labels[idx]

loader = DataLoader(
    MyDataset(paths, labels),
    batch_size=64,
    shuffle=True,
    num_workers=4,      # 4 个子进程并行执行 __getitem__，主进程只负责收 batch
    pin_memory=True,    # 分配页锁定内存，加速 CPU→GPU 拷贝
    persistent_workers=True,  # 每个 epoch 不再杀掉重启 worker，省去反复 fork 的开销
    prefetch_factor=2,  # 每个 worker 预先准备 2 个 batch（默认值），饥饿时可调大
)
```

`num_workers` 是最重要旋钮。经验法则：**从 0（纯同步）开始测，逐步翻倍到 GPU 利用率不再上升为止**。常见落点在 CPU 核数的一半到全部之间。如果调到 8 个 worker GPU 还是饿，说明瓶颈可能根本不在并行度，而在 ① 读取或 ② 解码本身——这时候加 worker 没用，得换存储格式。

### 3.2 验证你的管线是否成为瓶颈

```python
import time

# 简单粗暴的吞吐测试：只迭代 DataLoader，不做任何计算
def benchmark_loader(loader, n_iter=50):
    it = iter(loader)
    next(it)  # 第一个 batch 含 worker 启动开销，不计入
    t0 = time.perf_counter()
    for _ in range(n_iter):
        next(it)
    dt = time.perf_counter() - t0
    print(f"数据管线吞吐: {n_iter * loader.batch_size / dt:.0f} 样本/秒")

benchmark_loader(loader)

# 再测你的训练 step 耗时（不含取数）。如果
#   step 耗时 ≈ 数据耗时 → 管线是瓶颈（GPU 在等数据）
#   step 耗时 >> 数据耗时 → 计算是瓶颈，管线健康
```

更精确的方法是上一期的 `torch.profiler`：看 timeline 上 GPU kernel 之间有没有大片空白，空白处 CPU 在跑什么（通常能看到 `dataloader` 相关线程在做解码）。

### 3.3 pin_memory + non_blocking：让 H2D 拷贝不再卡主循环

```python
device = "cuda"
for x, y in loader:
    # pin_memory=True 时，non_blocking=True 允许拷贝异步执行
    # CPU 不必等数据完全落到 GPU 就继续往下走（配合 CUDA Stream 重叠更彻底）
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    ...
```

`pin_memory` 的原理：普通 CPU 内存是可换页的，CUDA 驱动不能直接对其发起 DMA 拷贝，必须先拷到一个「页锁定」的中转区。`pin_memory=True` 让 DataLoader 直接把 batch 放进页锁定内存，省掉中转，拷贝带宽大幅提升。代价是页锁定内存不可换页、分配更贵，所以只对要上 GPU 的 batch 用。

### 3.4 内存映射：把「读文件」变成「读内存」

```python
import numpy as np

# 假设你把整个数据集预处理成了一块 [N, 3, 224, 224] 的 float16 数组存到磁盘
# np.memmap 不会把文件读进内存，只是建立 虚拟地址 ↔ 磁盘偏移 的映射
big = np.memmap("images.dat", dtype=np.float16, mode="r",
                shape=(1_000_000, 3, 224, 224))

class MemmapDataset(Dataset):
    def __getitem__(self, idx):
        # 这次"读磁盘"只发生在这里：操作系统按 4KB 页按需加载
        # 第一次访问某页是磁盘 I/O，之后该页驻留 page cache，接近内存速度
        x = torch.from_numpy(big[idx].astype(np.float32))
        return x, 0
```

这就是为什么把 JPEG 预转换成 **raw/npz/webdataset 格式**后训练往往快好几倍——不是数据变少了，而是把「随机读小文件 + JPEG 解码」这两件最慢的事，换成了「按需按页读大文件 + 零解码」。

---

## 四、围绕这个领域展开：全景地图

数据管线不是一个类，而是一个生态系统。理解它的邻居，才能真正定位问题。

### 4.1 张量级管线：免 DataLoader 的数据流

当数据小到能整个塞进显存（比如几十万条表格数据），`DataLoader` 的多进程反而是累赘——每次 worker 取数据都要跨进程传张量（走共享内存文件描述符）。此时直接手写取数更快：

```python
data_gpu = data.to("cuda")           # 一次性上 GPU
for i in range(0, len(data_gpu), bs):
    batch = data_gpu[i:i+bs]         # 纯显存切片，零拷贝零进程通信
```

这就是表格类任务（如 CTR 训练）几乎不用 DataLoader 的原因。**DataLoader 是为「数据放不进内存」的世界设计的**。

### 4.2 分布式管线：DistributedSampler

DDP 场景下（第 006 期），每张卡必须拿到互不重叠的数据分片，靠 `DistributedSampler` 实现：

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=64, sampler=sampler,
                    num_workers=4, pin_memory=True)

for epoch in range(epochs):
    sampler.set_epoch(epoch)  # 不设这个，每个 epoch 每张卡拿到的顺序都一样！
```

`set_epoch` 是分布式训练中最经典的「静默坑」之一：忘了设，shuffle 就形同虚设，模型收敛质量悄悄变差。

### 4.3 生态位：IterableDataset 与流式数据

Map-style Dataset（有 `__len__` + `__getitem__`）适合静态数据；但训练大模型时数据常常是**流**——从对象存储按 shard 读、甚至在线合成。此时用 `IterableDataset`：

```python
class StreamDataset(IterableDataset):
    def __iter__(self):
        for shard in download_shards():   # 按需下载分片，不占全量内存
            for sample in parse_shard(shard):
                yield transform(sample)
```

代价：没有长度、难以随机 shuffle（只能 shard 级 shuffle + buffer 内 shuffle）、每个 worker 要自己处理分片去重（`worker_init_fn` 里按 worker id 切分数据，否则每个 worker 都会产出全量数据）。webdataset、HuggingFace `datasets` 的 streaming 模式都走这条路。

### 4.4 与 torch.compile / CUDA Graph 的关系

第 001 期讲过 CUDA Graph 要求静态输入形状，第 003 期的 torch.compile 对动态形状会反复重编译。**数据管线是形状抖动的头号来源**：drop_last=False 时最后一个 batch 更小、变长文本没 pad 到统一长度，都会击穿静态形状假设。所以追求极致性能的训练管线都会 `drop_last=True` + 严格 pad 到固定长度——这不是为了数据正确性，是为了编译稳定性。

### 4.5 上游生态

- **WebDataset / tar 分片**：把上万个碎文件打包成顺序读的 tar，对象存储时代的事实标准
- **FFCV / DALI / RMM-DALI**：把解码和增强搬到 GPU 或独立流水线，专治图像解码瓶颈
- **MosaicML Streaming / MDS**：训练中边下边训，数据集格式自带分片索引
- **HuggingFace datasets**：Arrow 内存映射列式存储，天然契合本文 3.4 的思路

共同方向：**从「文件系统上的随机读」走向「顺序读大分片 + 内存映射 + 按需流式」**。

---

## 五、什么时候该用 / 不该用

**该动手优化数据管线的信号：**
- profiler 显示 GPU 利用率低、kernel 间有大片空白
- 加大 batch_size 吞吐不升反降
- CPU 占用常年打满而 GPU 占用低
- 数据是海量小文件（几 KB ~ 几百 KB 一个）格式

**不必过度优化的时候：**
- 数据在内存/显存放得下：直接张量切片，扔掉 DataLoader
- 训练本身极重（大模型 FSDP）：step 耗时几百毫秒，管线天然跟得上
- 只是跑实验原型：默认参数先跑通，profile 证明确实饿了再优化——过早优化管线是浪费时间

**优先级顺序（性价比从高到低）：**
1. `num_workers=4~8` + `pin_memory=True` + `persistent_workers=True`（一行代码的事）
2. 预处理一次、缓存到磁盘（转 raw/float16/分片格式），别在训练时反复解码
3. 小文件合并成分片 + 内存映射
4. 上专用库（DALI / FFCV / webdataset）
5. prefetch、双缓冲等手工优化（最后才做）

---

## 六、常见坑

**坑 1：num_workers=0 还在纳闷为什么慢**
`num_workers` 默认是 0——单进程同步加载，取完一个 batch 才能开始下一步计算，GPU 每 step 都要干等解码。这是最常见的「为什么我的 GPU 利用率只有 30%」的答案。

**坑 2：worker 里生成了随机数，每个 epoch 结果不同 / 或每个 worker 相同**
每个 worker 是独立进程，随机数状态需要单独管理。`__getitem__` 里做随机增强时，DataLoader 默认会为每个 worker 设置不同的 seed（基于 base_seed + worker_id），但你**自己在 `__init__` 里 `np.random.seed(42)` 是没用的**——那是父进程的状态，不会传给 worker。要用 `worker_init_fn` 或 `torch.initial_seed()` 正确派生。

**坑 3：worker 太多，内存爆了**
每个 worker 是完整复制的进程，会把 Dataset 对象（包括你挂在 `self` 上的大数据、缓存字典）整体 fork 一份。8 个 worker × 每个缓存 5GB = 40GB 内存蒸发。对策：大数据用内存映射（memmap 是共享页缓存的，fork 后不额外复制），小心 `__getitem__` 里的 `lru_cache`（每个 worker 各自缓存一份）。

**坑 4： IterableDataset 忘了按 worker 切分，epoch 长度凭空翻了 N 倍**
`num_workers=4` 时，每个 worker 都会完整迭代一遍 `__iter__`，数据集被重复读 4 遍。必须用 `get_worker_info()` 在 `__iter__` 内按 worker id 跳过不属于自己的样本。

---

## 七、一句话总结

**训练速度的上限由最慢的一环决定，而那一环几乎从来不是 GPU——把「读取、解码、增强、搬运」重叠成流水线，让 GPU 永远有活干，就是数据管线的全部。**

---

## 八、今日练习

**题目**：你接手一个图像分类任务，观测到：GPU 利用率 35%，每个 step 中 GPU 计算只占 40ms，但 step 总耗时 180ms；`nvidia-smi` 里 CPU 单核 100%。请给出至少三条按优先级排序的优化措施，并说明每条分别作用于数据流水线的哪个环节（①读取/②解码/③增强/④collate/⑤H2D）。

<details><summary>参考答案</summary>

诊断：GPU 空转 140ms/step，管线严重饥饿；CPU 单核打满提示解码或增强在单线程上跑。措施（按性价比排序）：

1. **`num_workers=8, pin_memory=True, persistent_workers=True``**——作用于 ②③④，多进程并行执行 `__getitem__` 里的解码与增强，是单核打满的直接解药，一行配置改动。
2. **预处理落盘缓存**——作用于 ①②：把 JPEG 解码 + resize 的结果一次性转成 float16/raw 格式的分片大文件，训练时跳过解码环节。若解码是主导（如大分辨率 PNG），这条收益最大，可与第 1 条叠加。
3. **小文件合并 + 内存映射**——作用于 ①：如果画像显示时间耗在文件系统随机读（海量小文件），把样本打包成顺序分片并用 `np.memmap` / webdataset 读取，让 OS page cache 发挥作用。
4. （可选进阶）**`prefetch_factor` 调大 / `non_blocking=True` H2D 拷贝**——作用于 ⑤，掩盖主机到设备传输延迟；只有前几条做完还有空闲时才值得。

验证方法：改一条测一次吞吐（本文 3.2 的 benchmark），直到「step 总耗时 ≈ GPU 计算耗时」，说明管线已不再是瓶颈。

</details>
