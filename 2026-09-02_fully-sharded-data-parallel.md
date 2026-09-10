# PyTorch 每日一课 · 第 010 期

## Fully Sharded Data Parallel (FSDP)：大模型的显存策略

> **日期**：2026-09-02
> **难度**：⭐⭐⭐⭐
> **前置知识**：DDP 基本概念（第 006 期）、通信原语 AllGather / ReduceScatter（第 007 期）、GPU 显存构成（第 004 期）
> **预计阅读时间**：15 分钟

---

## 一、这个领域解决什么问题

先算一笔账。假设你要训练一个 7B 参数的模型，用 Adam 优化器、混合精度训练（第 002 期），单卡需要多少显存？

| 项目 | 计算 | 大小 |
|---|---|---|
| 模型参数（BF16） | 7B × 2 字节 | 14 GB |
| 梯度（BF16） | 7B × 2 字节 | 14 GB |
| 优化器状态（FP32 参数副本 + 两个动量） | 7B × 12 字节 | 84 GB |
| **合计（还不算激活值）** | | **112 GB** |

一张 A100/H100 80GB 根本放不下。**问题不在计算量，而在"状态存放"**——模型还没开始跑第一步，显存就已经爆了。

DDP 解决不了这个问题：DDP 的设计是**每张卡都持有完整的模型副本**（Replicate），它只切了数据，不切模型。卡越多，冗余越多。

于是核心问题变成：**能不能把"参数、梯度、优化器状态"这些训练状态切成 N 份，分摊到 N 张卡上，用的时候再临时拼起来？** 这就是 FSDP（Fully Sharded Data Parallel）的全部出发点。

## 二、核心思想：碎片化 + 按需重组

用一个类比来理解。

想象一个 4 人团队要翻译一本 400 页的书：

- **DDP 的做法**：每人复印一本完整的书（4 本），各自翻译不同的章节，最后对一下笔记（梯度 AllReduce）。书越大，复印成本越高。
- **FSDP 的做法**：把书**拆成 4 份**，每人保管 100 页。翻译到某一章时，大家把自己手里的那 100 页**复印传阅**（AllGather 拼出完整章节），翻译完这一章立刻还回去，笔记也只记自己负责那 100 页的内容（ReduceScatter 切碎梯度）。

关键洞察：**任意时刻，你只需要"正在计算的那一层"是完整的，其余层都可以是碎片。** 而 Transformer 是逐层串行执行的——第 3 层算完才能算第 4 层——所以这个"按需重组"完全可行。

这就是 FSDP 的三段式节奏，对每一个参数块（parameter flat）：

1. **前向 AllGather**：进入某层前，各卡互相交换自己持有的碎片，拼出该层的完整参数 → 计算
2. **前向后释放**：算完这层，立刻把非本卡负责的参数碎片丢掉（free），显存回收
3. **反向 ReduceScatter**：反向传播到该层时再次 AllGather 拼参数，算出梯度后，把梯度 ReduceScatter——每张卡只保留自己那个碎片的梯度（已经求和好了），然后就地更新**自己的**优化器状态

注意最后一点：因为梯度已经按碎片切开，每张卡的优化器只需要为 1/N 的参数维护状态。**优化器状态这个最大的显存黑洞（上表 84GB）直接除以 N。**

### ZeRO 的三档策略

FSDP 的思想源头是微软 DeepSpeed 的 **ZeRO**（Zero Redundancy Optimizer）论文，它把训练状态切成三档：

- **ZeRO-1**：只切优化器状态 → 显存 ÷ N（大头）
- **ZeRO-2**：切优化器状态 + 梯度 → 再省一截
- **ZeRO-3 / FSDP 完全体**：切优化器状态 + 梯度 + **参数本身** → 显存可以做到约 1/N

PyTorch 官方的 FSDP 对应 ZeRO-3 的思路（参数也切）；而 `torchrun` 生态里常说的 "ZeRO-2" 配置在 DeepSpeed 里对应——梯度切、参数不切（省去了前向的 AllGather，通信量小，适合能装下的模型）。

**代价**：天下没有免费的显存。切参数意味着每一层前后都要 AllGather、反向都要 ReduceScatter——**通信量比 DDP 显著增加**，且和流水线/计算重叠做得好不好直接决定速度。FSDP 本质上是用通信带宽换显存容量。

## 三、在 PyTorch 中怎么用

以下代码单机多卡可用 `torchrun --nproc_per_node=4 fsdp_demo.py` 运行。

### 3.1 现代写法：FullyShardedDataParallelConfig + 自动包裹

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 或 size_based_auto_wrap_policy

# 1) 初始化进程组（torchrun 会注入环境变量）
torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(torch.distributed.get_rank())

model = nn.TransformerEncoder(...)

# 2) 混合精度策略：参数用 BF16 计算，梯度规约用 BF16（省带宽）
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# 3) FSDP 配置
cfg = dict(
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # = ZeRO-3；HYBRID_SHARD 见下文
    mixed_precision=mp_policy,
    auto_wrap_policy=ModuleWrapPolicy(nn.TransformerEncoderLayer),  # 按层包裹！
    cpu_offload=None,  # 需要时用 CPUOffload(offload_params=True) 把参数卸到 CPU
    limit_all_gathers=True,  # 防止多个 AllGather 排队堆积撑爆显存（务必开启）
    sync_module_states=True,  # 保证各卡初始权重一致（rank 0 广播）
)

model = FSDP(model, **cfg)

# 4) 之后的使用方式和普通模型一模一样
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# ... 标准 train loop：loss.backward(); optimizer.step(); optimizer.zero_grad()
```

### 3.2 关键点：auto_wrap_policy 决定了"切多细"

FSDP 不能把整个模型当成一个大块来切——那样前向开始前要一次性 AllGather **全部**参数，等于没省。`auto_wrap_policy` 的作用是把模型拆成多个小的 FSDP 单元，**每个单元独立地走"拼→算→释放"的循环**：

- `ModuleWrapPolicy(nn.TransformerEncoderLayer)`：每个 Transformer 层是一个单元——最常用、最符合直觉
- `size_based_auto_wrap_policy(min_num_params=1e6)`：按参数量自动切，适合不规则模型
- 也支持用 `@fsdp_wrap` 装饰器或 lambda 手工指定

包裹粒度是一个真实的权衡：**切得细 → 峰值显存低、但通信次数多；切得粗 → 通信合并、但峰值显存高**。

### 3.3 训练完保存/加载：分片状态字典

```python
from torch.distributed.checkpoint import save, load
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os

# 保存：每张卡各写一个分片文件，不会集中到一张卡上
state = {"model": model.state_dict()}
save(state, checkpoint_id="ckpt_dir")

# 加载：
state = {"model": model.state_dict()}
load(state, checkpoint_id="ckpt_dir")
model.load_state_dict(state["model"])
```

不要用普通的 `torch.save(model.state_dict())`——FSDP 模型的 state_dict 里是**碎片**，需要用官方分片保存 API（旧版是 `FSDP.state_dict_type()` 系列，新版推荐 `torch.distributed.checkpoint`）。

## 四、围绕该领域展开：FSDP 在生态中的位置

这是本文的重点。FSDP 不是孤立的 API，它是"大模型显存工程"谱系上的一个节点，往前后各看一步，你才能选对工具。

### 4.1 显存优化手段的"光谱"

按侵入性从低到高排列：

```
AMP 混合精度 ──> 梯度检查点 ──> ZeRO-1/2 ──> FSDP/ZeRO-3 ──> 张量并行 ──> 流水线并行 ──> 专家并行
 (省一半)      (激活换时间)   (切状态)     (状态全切)     (切单层计算)  (切层到不同卡)  (切 MoE 路由)
```

- **AMP（第 002 期）**：任何模型都该先开，零改造成本
- **梯度检查点（第 009 期）**：针对**激活值**显存；FSDP 针对**模型状态**显存——两者正交，可以叠加
- **FSDP**：针对模型状态；数据并行的底子保留（每卡仍然吃不同的 batch）
- **张量并行（TP）**：把**单层内部的矩阵乘法**切到多卡（如把权重矩阵按列切开）；需要卡间极低延迟（一般限单机 NVLink 内）
- **流水线并行（PP）**：把**不同的层**放到不同卡，像流水线一样传；吞吐高但有 bubble
- 大模型实战通常是 **FSDP + TP + PP 组合拳**（3D 并行），FSDP2 的出现正是为了让这种组合更好写

### 4.2 FSDP1 vs FSDP2

- **FSDP1**（`torch.distributed.fsdp`）：基于 `FlatParameter`——把一组参数拍平成一个大 tensor 管理的黑魔法，改写了模块的参数注册，导致复杂模型（共享参数、tied embedding、Meta 初始化）容易踩坑
- **FSDP2**（`torch.distributed.fsdp.fully_shard`，PyTorch 2.6+ 稳定）：基于 per-parameter sharding + DTensor，逐参数碎片化，对 torch.compile / DTensor / 更深的并行组合更友好。新项目优先选 FSDP2：

```python
from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy

for layer in model.layers:           # 先包内层
    fully_shard(layer)
fully_shard(model)                   # 再包外层（顺序很重要：由内向外）
```

### 4.3 周边的配套工具

- **`torch.distributed.checkpoint`（DCP）**：分片保存/加载，天然适配 FSDP，支持异构 shape 重排
- **`torch.distributed.tensor（DTensor）**：FSDP2 的底层表示，"带分片语义的张量"，也是 TP/SP 未来统一的语言
- **加速通信**：NCCL 的通信 overlaps 计算是 FSDP 性能的命脉；`device_mesh`（`torch.distributed.device_mesh`）用来描述卡的拓扑
- **CPU Offload / 磁盘 Offload**：`CPUOffloadPolicy` 把碎片参数放内存，甚至换到 NVMe——训练再大的模型时的最后手段
- **监控**：`torch.distributed.memtracer`、`memory_profiler`，以及注册 `FSDP` 的 reshape/优化器状态回调来查看真实峰值

### 4.4 与 DeepSpeed ZeRO 的关系

一句话：**FSDP ≈ PyTorch 原生版的 ZeRO-3**。DeepSpeed 提供了 ZeRO-1/2/3 三档旋钮 + 更多工程化功能（ZeRO-Offload、ZeRO-Infinity）；FSDP 后来居上，胜在与 PyTorch 生态（compile、DTensor、checkpoint）的无缝集成。选择上：想要精细控制和微软全家桶 → DeepSpeed；想要原生、简洁、和 torch 生态贴合 → FSDP。

## 五、什么时候该用 / 不该用

**该用 FSDP：**
- 模型状态（参数+梯度+优化器状态）单卡放不下，但单层计算单卡放得下
- 有多张卡和较好的卡间带宽（NVLink 最好，RDMA 万兆起步）
- 训练中小规模 LLM（1B–100B 量级）、做微调/继续预训练

**不该用：**
- 模型本来就装得下（<1B）→ DDP 更简单、通信更少、吞吐更高
- 单层就大过单卡显存（如超大 embedding、巨型 MLP）→ 先考虑张量并行
- 卡间只有慢速以太网 → FSDP 的 AllGather 会把训练拖死，考虑 PP/TP 或先升级网络
- **推理场景** → FSDP 是训练技术，推理请看 torch.export / vLLM 路线

## 六、常见坑（4 个）

1. **忘记 `sync_module_states=True`，各卡权重不一致**。随机初始化在每张卡上不同，不同步会导致"各卡在训练不同的模型"。从 checkpoint 加载则天然一致，可不开。
2. **不设 `auto_wrap_policy`，整个模型一个大单元**。结果第一次前向就要 AllGather 全部参数，显存峰值和不切一样高——白切了。
3. **关闭 `limit_all_gathers` 后 OOM**。反向传播时多个单元的 AllGather 请求可以排队重叠，若不限流，它们同时挂起会瞬间堆出几个单元的完整参数。默认开启，别手贱关。
4. **模块参数过多/Few 巨型层** + **保存用 `torch.save(model.state_dict())`**。前者导致某单元 AllGather 拼出的完整参数过大；后者保存的是碎片、普通加载必然出错。用 `torch.distributed.checkpoint` 的 save/load，且保存前 `model.eval()` 并确保没有进行中的预取。

## 七、一句话总结

**FSDP 用"把训练状态切碎、按需 AllGather 重组、ReduceScatter 归还"的方式，把模型状态显存从 O(模型) 降到 O(模型/卡数)——代价是更多通信，本质是用带宽换容量。**

## 今日练习

<details><summary>今日练习</summary>

**题目**：一个 2B 参数的模型用 AdamW + BF16 混合精度训练。请计算：(a) 单卡 DDP 需要多少模型状态显存？(b) 8 卡 FSDP（FULL_SHARD）下每卡的模型状态显存是多少？(c) 假设激活值每卡固定 6GB，8×40GB 的 A100 机器上 DDP 和 FSDP 各能不能训？

**参考答案**：

模型状态构成（BF16 混合精度 + AdamW）：
- 参数：2B × 2 字节 = 4 GB
- 梯度：2B × 2 字节 = 4 GB
- 优化器状态：FP32 参数副本 2B × 4 + 一阶动量 2B × 4 + 二阶动量 2B × 4 = 24 GB

(a) DDP 单卡：4 + 4 + 24 = **32 GB**（每张卡都要这么多）

(b) FSDP 8 卡：每卡只保存 1/8 的参数、梯度、优化器状态 = 32 / 8 = **4 GB**

(c) 40GB A100：
- DDP：32（状态）+ 6（激活）≈ 38 GB，勉强能塞但极紧，稍一波动就 OOM
- FSDP：4（状态）+ 6（激活）≈ 10 GB，每卡余量 30 GB，可以从容加大 batch 或开梯度检查点训练更长序列

这也解释了一个工程直觉：**FSDP 把省出来的显存，通常再投资给更大的 batch 或更长的上下文**，而不只是"能跑起来"。

</details>
