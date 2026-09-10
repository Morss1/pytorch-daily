# PyTorch 每日一课 · 第 007 期

## 通信原语：AllReduce、AllGather、ReduceScatter 到底在干什么

- **日期**：2026-08-27
- **难度**：⭐⭐⭐⭐
- **前置知识**：多 GPU 训练的基本概念（知道 DDP 是干嘛的即可）、张量的形状运算
- **预计阅读时间**：15 分钟

---

## 一、这个领域解决什么问题

上一期我们讲了 DDP：每张卡各自算梯度，然后「同步」一下，让所有卡的模型保持一致。

但你有没有想过，那个「同步」在 GPU 之间到底是怎么发生的？

多卡训练的本质困难只有一个：**数据分散在多张卡上，而计算却需要「全局」的信息**。

举一个最简单的例子——梯度平均：

- 卡 0 算出梯度 `g0`，卡 1 算出 `g1`，卡 2 算出 `g2`，卡 3 算出 `g3`
- 每张卡都需要拿到 `(g0+g1+g2+g3)/4` 才能更新参数
- 但 GPU 之间不能直接「读」对方的显存

所以我们需要一套**卡间搬运和合并数据的标准化操作**。这套操作就叫**集合通信原语（Collective Communication Primitives）**。

理解了这个领域，你就能看懂：

- DDP 的梯度同步到底卡在哪一步
- 为什么 FSDP / ZeRO 的通信量和 DDP 一样，显存却省那么多
- 为什么张量并行必须用 AllReduce / AllGather，而不能用别的
- 「通信瓶颈」这个词到底指什么

---

## 二、核心思想：快递网络的比喻

把多张 GPU 想象成 4 个分布在不同城市的仓库，它们之间只能通过快递网络互相寄包裹。

### 2.1 「Reduce」= 汇总到一家

**Reduce（规约）**：所有仓库把各自的货物寄到「总仓」（比如卡 0），由卡 0 把它们加起来（或求最大值等）。

结果：**只有卡 0** 拥有汇总结果，其他卡两手空空。

```
卡0: g0 ─┐
卡1: g1 ─┼─→  卡0 拿到 g0+g1+g2+g3
卡2: g2 ─┤    其他卡什么都没有
卡3: g3 ─┘
```

问题很明显：训练需要**每张卡**都拿到结果。于是有了 AllReduce。

### 2.2 「All」= 人人有份

**AllReduce（全局规约）**：在 Reduce 的基础上，把结果再分发回所有卡。执行完后，**每张卡都持有完全相同的汇总结果**。

```
执行前：卡i 各持有 gi
执行后：每张卡都持有 (g0+g1+g2+g3)
```

这就是 DDP 梯度同步用的原语。AllReduce 的价值在于：**通信完成后没有「多余的卡」**，大家拿到的结果一模一样，可以各自独立地更新参数。

### 2.3 「AllGather」= 拼图合体

**AllGather（全局收集）**：每张卡持有一块**碎片**（比如一个大张量的第 1/4 切片），AllGather 执行完后，每张卡都持有**完整的拼接结果**。

```
执行前：卡i 持有 [xi]        （各一块碎片）
执行后：每张卡都持有 [x0|x1|x2|x3]   （完整拼图）
```

### 2.4 「ReduceScatter」= 分蛋糕

**ReduceScatter（规约分发）**：Reduce 和 Scatter 的合体——先把所有卡的数据加起来（Reduce），然后**只把结果的第 i 块切片**发给卡 i。

```
执行前：卡i 持有 gi
执行后：卡i 只持有 (g0+g1+g2+g3) 的第 i 块切片
```

注意一个精妙的事实：

> **AllReduce = ReduceScatter + AllGather**

先用 ReduceScatter 让每张卡拿到求和结果的一块，再用 AllGather 把块拼回完整结果。这个分解正是 Ring 算法的实现方式，后面会讲。

### 2.5 三兄弟一句话对照

| 原语 | 执行前（每卡持有） | 执行后（每卡持有） | 通信量 |
|---|---|---|---|
| **AllReduce** | `gi` | `Σgj`（完整） | `2(N-1)/N · V` |
| **AllGather** | 碎片 `xi` | 完整拼图 `[x0..x3]` | `(N-1)/N · V` |
| **ReduceScatter** | `gi` | `Σgj` 的第 i 块 | `(N-1)/N · V` |

（N = 卡数，V = 每卡数据体积；可以看出 AllReduce 的通信量约是另外两者的 **2 倍**——这一点是理解 ZeRO 显存优化的钥匙。）

---

## 三、在 PyTorch 中怎么用

PyTorch 把这些原语封装在 `torch.distributed` 里。下面的代码用 **单机多进程（gloo 后端）** 就能跑，不需要真的有多张 GPU——用 CPU 模拟 4 个进程：

```python
# 保存为 demo_collectives.py，然后用 torchrun 启动：
# torchrun --nproc_per_node=4 demo_collectives.py
import os
import torch
import torch.distributed as dist

def main():
    # torchrun 会自动设置这些环境变量
    rank = int(os.environ["RANK"])          # 当前进程编号（0~3）
    world_size = int(os.environ["WORLD_SIZE"])  # 总进程数

    # 初始化进程组：所有进程互相「握手」
    dist.init_process_group(backend="gloo")  # CPU 用 gloo，GPU 用 nccl

    # 每个进程造一个自己的张量：值等于自己的 rank
    x = torch.tensor([float(rank)])
    print(f"[rank {rank}] Before AllReduce: {x.item()}")

    # ---- AllReduce：所有人求和，每卡拿到完整结果 ----
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    # 现在每张卡的 x 都是 0+1+2+3 = 6
    print(f"[rank {rank}] After  AllReduce: {x.item()}")
    assert x.item() == 6.0

    # ---- ReduceScatter：求和后只拿第 rank 块切片 ----
    # 输入必须能按 world_size 均分
    big = torch.arange(0, 8, dtype=torch.float32) + rank  # 8 个数，4 卡各分 2 个
    shard = torch.empty(2)
    dist.reduce_scatter_tensor(shard, big, op=dist.ReduceOp.SUM)
    # 求和结果理论上是 [6, 10, 14, 18]，rank 0 拿 [6,10]，rank 1 拿 [14,18]……
    print(f"[rank {rank}] ReduceScatter shard: {shard.tolist()}")

    # ---- AllGather：每卡的碎片拼成完整拼图 ----
    fragment = torch.tensor([float(rank)])  # 每卡一个碎片
    gathered = [torch.empty(1) for _ in range(world_size)]
    dist.all_gather(gathered, fragment)
    # 现在每张卡都持有 [0, 1, 2, 3]
    print(f"[rank {rank}] AllGather result: {[t.item() for t in gathered]}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

运行方式：

```bash
# 安装好 PyTorch 后执行（CPU 即可，gloo 后端不需要 GPU）
torchrun --nproc_per_node=4 demo_collectives.py
```

预期输出（顺序可能不同）：

```
[rank 0] Before AllReduce: 0.0
[rank 0] After  AllReduce: 6.0
[rank 0] ReduceScatter shard: [6.0, 10.0]
[rank 0] AllGather result: [0.0, 1.0, 2.0, 3.0]
...
```

**三个后端要知道**：

- `gloo`：CPU/调试用，慢但兼容性好
- `nccl`：NVIDIA GPU 的标配，为 NVLink/InfiniBand 深度优化，生产环境必用
- `mpi`：HPC 集群场景，需要自己编译 PyTorch

---

## 四、围绕这个领域展开：它在整个生态中的位置

这是本期最重要的部分。通信原语是**整个分布式训练生态的地基**，几乎所有上层框架都只是这几个原语的不同排列组合。

### 4.1 Ring AllReduce：为什么 AllReduce 可以「不经过中心节点」

最朴素的做法是「星型」：所有卡把数据发给卡 0，卡 0 加完再发回去。但这样卡 0 的带宽会成为瓶颈，卡越多越慢。

**Ring AllReduce** 的思路是把 N 张卡连成一个环，数据切成 N 块，分 N-1 轮传递：

1. **ReduceScatter 阶段**：每一轮，每张卡把自己的一块发给下一家，并累加收到的块。N-1 轮后，每张卡恰好持有一块「完整求和」的切片。
2. **AllGather 阶段**：再来 N-1 轮，把这些切片沿环传播，拼回完整结果。

好处：

- 每张卡只跟左右邻居通信，**带宽被均匀分摊**
- 总通信量 `2(N-1)/N · V` 与卡数几乎无关——卡从 4 张加到 64 张，每卡的通信负担几乎不变

NCCL 的核心就是高度优化的 Ring（以及双环、树形混合）实现。**当你听到「DDP 的梯度同步用 Ring AllReduce」时，指的就是这个。**

### 4.2 DDP：一次 AllReduce 搞定

DDP 把每个参数的梯度打包成 bucket，反向传播算完一个 bucket 就异步发起一次 **AllReduce**。每张卡最终拿到平均梯度，各自更新参数。

通信量 = 模型梯度总大小 × 2（每参数约 2 倍通信）。

### 4.3 ZeRO / FSDP：同样的通信量，省下显存

ZeRO 的洞察是：**梯度求和之后，每张卡其实不需要完整的平均梯度**——更新参数时每个 rank 只负责一部分参数就够了。

于是流程变成：

1. **ReduceScatter** 梯度 → 每卡只保留属于自己那块的求和结果（而不是完整梯度）
2. 用自己那块梯度更新自己负责的参数
3. 参数更新后，**AllGather** 参数 → 大家都拿到最新完整参数，继续前向

对比一下：

| | DDP | ZeRO-2 / FSDP |
|---|---|---|
| 梯度通信 | AllReduce（= RS + AG） | ReduceScatter |
| 参数通信 | 无（各存完整副本） | AllGather |
| 总通信量 | 2V | RS(V) + AG(V) = **2V** |

**通信量完全一样，但显存占用大幅下降**——这就是 FSDP 能训大模型而 DDP 不能的根本原因。所谓「免费的午餐」，只是把 AllReduce 拆开，把中间结果更聪明地分了家。

### 4.4 张量并行：把一个矩阵乘法拆到多卡

Megatron-LM 式张量并行的每层都要通信两次：

- 前向第一段后 **AllReduce**（把按列切分的部分结果加起来）
- 反向入口处 **AllReduce**（把梯度合并）

这解释了为什么**张量并行要求极快的卡间互联**（NVLink、单机内）：每层 Transformer 都要通信两次，走慢速网络会直接饿死计算。而数据并行（DDP）每步只通信一次，可以用跨机的 InfiniBand。

### 4.5 「计算与通信重叠」：藏起来的性能魔法

所有高性能框架都在干一件事：**把通信藏进计算的影子里**。

- DDP：反向传播还没算完，已经算完的梯度 bucket 就开始异步 AllReduce——通信和反向计算并行
- FSDP：前向传播某层时，就提前 AllGather 后面几层的参数（prefetch）
- `torch.compile` + 异步张量：编译期自动插入通信调度

如果你做过性能分析，会发现「GPU 利用率上不去」的多卡训练，八成时间不是花在计算，而是**每步结束都在等一个巨大的同步点**——这就是通信没有和计算重叠的典型症状。

### 4.6 关联概念速查

- **NCCL**：NVIDIA 的集合通信库，PyTorch GPU 分布式的默认引擎
- **NVLink / InfiniBand / RoCE**：物理层带宽决定了原语的天花板
- **Bucket（梯度桶）**：把小梯度合并成大块再通信，摊薄每次通信的固定开销
- **通信组（Process Group）**：`dist.new_group()` 可以只让一部分卡互相通信——张量并行 + 数据并行混用时靠它分组
- **Overlap / 异步通信**：`dist.all_reduce(..., async_op=True)` 返回 handle，可以稍后 wait

---

## 五、什么时候该用 / 不该用

**该直接关心这些原语的场景：**

- 写自己的分布式训练/推理流程（不想被 DDP 的黑盒限制）
- 多机训练通信慢，需要判断瓶颈在带宽还是在同步策略
- 实现模型并行、序列并行、专家路由等 DDP 覆盖不了的方案
- 阅读 Megatron-LM / DeepSpeed / vLLM 源码（它们满是裸调原语的代码）

**不需要直接关心的场景：**

- 单机单卡 / 单机多卡用 `DataParallel` 或 DDP + 单一模型——框架已经替你调好了
- 只是调参炼丹——知道「通信大概占了多少时间」即可，不必手写原语

一个经验法则：**当你开始手写 `dist.all_reduce` 时，你已经在写分布式系统，而不只是写训练脚本了。** 心里要有一张通信量账单。

---

## 六、常见坑

**坑 1：进程组没初始化就调原语**

直接跑一个调用了 `dist.all_reduce` 的脚本会报错。必须先用 `torchrun`（它会设好 `RANK`、`WORLD_SIZE` 等环境变量）并调用 `dist.init_process_group()`。用 `python demo.py` 直接跑是不行的。

**坑 2：CPU 上用了 NCCL 后端**

NCCL 只支持 GPU 张量。在 CPU/调试环境里要用 `backend="gloo"`。反过来，GPU 训练用 gloo 会慢一个数量级。条件写法：

```python
backend = "nccl" if torch.cuda.is_available() else "gloo"
dist.init_process_group(backend=backend)
```

**坑 3：ReduceScatter 的形状没对齐**

`reduce_scatter_tensor` 要求输入张量的第 0 维能被 `world_size` 整除。模型梯度经常不能整除（比如 97 个参数、4 张卡），所以 DDP/FSDP 内部都要做 bucket 打包和 padding。自己手写时忘了这一点，会得到一个莫名其妙的 shape 错误。

**坑 4：忘记同步就读结果（异步陷阱）**

`async_op=True` 的通信是异步的——发起后立刻返回，此时读结果是**未定义行为**。必须先 `handle.wait()`。更隐蔽的是，NCCL 是流异步的：即使不加 `async_op`，结果也要等当前 CUDA stream 上的操作排完队才可见，跨 stream 读取需要显式 `torch.cuda.synchronize()` 或事件同步。

---

## 七、一句话总结

> 多卡训练的一切上层设计——DDP、FSDP、张量并行——本质上都是在回答同一个问题：「**谁需要数据的哪一部分，用哪个原语、什么顺序搬过去最省**」。AllReduce 让人人有份，AllGather 拼图，ReduceScatter 分蛋糕，而 Ring 算法保证了搬家的成本几乎不随卡数增长。

---

## 八、今日练习

**题目**：假设你有一个 4 卡集群，每卡持有 100MB 的梯度。分别计算：

1. 用「星型」方案（全发给卡 0 再发回），卡 0 的收发总量是多少？
2. 用 Ring AllReduce，每张卡的收发总量是多少？
3. 如果把 DDP 换成 FSDP（ReduceScatter 梯度 + AllGather 参数），每步每卡通信量是多少？

<details><summary>参考答案</summary>

**1. 星型方案**：卡 0 要接收 4×100MB = 400MB（含自己那份其实只需收 3 份 300MB，视实现而定），求和后再发送 4×100MB = 400MB。卡 0 收发总量约 **700~800MB**，其他卡各 200MB。卡 0 成为瓶颈——卡数越多越失衡。

**2. Ring AllReduce**：总通信量公式为 `2(N-1)/N · V`，每卡承担其中 `2V` 的收发（发 2(N-1) 块 × 每块 V/N，收同样多）：

- 每卡发送：`(N-1) × V/N = 3 × 25MB = 75MB`（ReduceScatter 阶段）
- AllGather 阶段再发 75MB
- 每卡**收发各约 150MB**，与 4 卡时几乎持平；哪怕扩到 64 卡，每卡通信量依然是约 `2V = 200MB`——这就是 Ring 的可扩展性。

**3. FSDP 每卡通信量**：

- ReduceScatter 梯度：`(N-1)/N × V ≈ 75MB`
- AllGather 参数：`75MB`
- 合计约 **150MB/卡/步**——看起来只有 AllReduce（约 150MB）的一半？注意 AllReduce 每卡收发是 2 倍（150 发 + 150 收），而 FSDP 是 75 发 + 75 收。**传输的字节数 FSDP 与 DDP 相同**，区别在于 FSDP 中间不需要任何一张卡持有完整梯度，这就是显存省下来的地方。

</details>
