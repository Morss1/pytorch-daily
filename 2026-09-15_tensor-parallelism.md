# PyTorch 每日一课 · 第 019 期

## 张量并行 (Tensor Parallelism)：把一个矩阵乘法拆到多卡上

> **日期**：2026-09-15
> **难度**：⭐⭐⭐⭐
> **前置知识**：矩阵乘法分块、DataParallel/DDP 的基本概念（第 006 期）、通信原语 AllReduce/AllGather/ReduceScatter（第 007 期）
> **预计阅读时间**：15 分钟

---

## 一、这个领域解决什么问题

先回忆一下 DDP 的做法：每张卡都持有**完整的模型副本**，各自处理不同的数据，最后同步梯度。这叫**数据并行**——拆的是数据，模型没拆。

但问题来了：

- 模型大到**单卡放不下**怎么办？一个 70B 模型光 FP16 权重就要 140GB，单张 A100（80GB）根本塞不下。
- DDP 的显存冗余是巨大的：8 卡训练 70B 模型，等于把同样的 140GB 权重复制了 8 份，共 1.1TB 显存，其中绝大部分是浪费的重复。

FSDP/ZeRO（第 010 期）给出了一个答案：把参数、梯度、优化器状态**切碎分片**到各卡，用的时候临时聚合。但它的思路仍然是「训练时省钱」——每个瞬间每个参数还是完整地参与计算。

**张量并行 (TP) 走了一条更彻底的路：不是把模型当整体复制或分片，而是把单层内部的矩阵乘法本身拆开，让每张卡只算一个矩阵的一部分。**

一个直观的类比：

> **DDP 像开连锁店**：8 家分店（8 卡），每家都有完整的菜单（完整模型），各自服务自己的客人（不同数据）。
>
> **张量并行像一条流水线厨房**：一道菜（一层的矩阵乘法）被拆成 8 道工序，8 个厨师（8 卡）每人只负责一道工序，每道菜都要 8 个人合力完成。

关键差别：DDP 中卡与卡只在每步结束时通信一次；TP 中卡与卡在**每一层的前向和反向都要通信**。这决定了 TP 必须用在**高带宽互联**（NVLink）上才能发挥威力。

## 二、核心思想：怎么拆一个矩阵乘法

TP 的数学基础出奇地简单——就是**分块矩阵乘法**。以Transformer 层里两个最重要的运算为例（来自 Megatron-LM 的经典论文）。

### 2.1 拆法一：列切（按输出维度拆）

把权重矩阵 $W$ 沿列切成两半 $W = [W_1, W_2]$，输入 $X$ 不动：

$$XW = X[W_1, W_2] = [XW_1, XW_2]$$

- 卡 1 算 $Y_1 = XW_1$，卡 2 算 $Y_2 = XW_2$
- 输出 $Y = [Y_1, Y_2]$ **天然就是分片状态**，不需要立刻通信
- 这就是为什么 MLP 的第一个线性层这么拆——它的输出马上要进 GeLU，而 GeLU 是逐元素的，分片状态各自激活完全没问题

### 2.2 拆法二：行切（按输入维度拆）

把 $W$ 沿行切、输入 $X$ 沿列切：

$$XY = [X_1, X_2] \begin{bmatrix} W_1 \\ W_2 \end{bmatrix} = X_1W_1 + X_2W_2$$

- 每张卡算出一个**部分和**，最后需要一次 **AllReduce** 把部分和加起来
- MLP 的第二个线性层这么拆，正好接住第一个层留下的列分片输入，一步到位消掉分片

### 2.3 MLP 的完整拆法

Megatron 的 MLP 块：第一个线性层**列切**、第二个线性层**行切**，中间不需要任何通信，块结束只需**一次 AllReduce**。这是整个 TP 最优雅的地方：

```
        X (每卡完整)
         │
   ┌─────┴─────┐
   │  W1 列切   │   Y1 = XW1     Y2 = XW2   ← GeLU 可以各自做
   └─────┬─────┘
   GeLU(Y1)   GeLU(Y2)
         │
   ┌─────┴─────┐
   │  W2 行切   │   Z = Y1W2' + Y2W2' 的部分和
   └─────┬─────┘
         │
      AllReduce  ← 一次通信，得到完整 Z
```

### 2.4 注意力头怎么拆

自注意力更好拆：**按注意力头切**。8 个头、2 张卡，每卡管 4 个头。每个头有自己的 Q/K/V 投影矩阵（天然独立），输出投影矩阵按行切。头内部计算完全独立，最后 AllReduce 一次。

这也是为什么 **TP 并行度最好是注意力头数的约数**——比如 64 个头拆 8 卡很整齐，拆 48 卡就出现头被切开再拼接的尴尬。

### 2.5 通信代价：为什么 TP 只在机内用

一层 Transformer 前向需要 2 次 AllReduce（MLP 一次 + 注意力一次），反向还需要 2 次。AllReduce 的数据量与**隐藏维度 × 序列长度 × batch** 成正比，且发生在每一层。

对比带宽量级：
- **NVLink（机内）**：约 400-900 GB/s
- **InfiniBand（机间）**：约 50-400 GB/s

所以工程实践的定式是：**TP 只在单机内（最多 8 卡 NVLink 互联）使用，跨机用 FSDP/DP 叠加**。这也解释了 Megatron 的 3D 并行（TP × PP × DP）的设计。

## 三、在 PyTorch 中怎么用

### 3.1 PyTorch 原生 API：`torch.distributed.tensor.parallel`

PyTorch 2.x 开始把 Megatron 的 TP 思路收编为原生模块。最直接的两个原语是 `colwise_parallel` 和 `rowwise_parallel`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    colwise_parallel, rowwise_parallel, PairwiseParallel
)

# 第一步：初始化进程组和设备网格（假设 2 张卡，单机）
# 实际使用需用 torchrun 启动，这里展示逻辑
torch.distributed.init_process_group(backend="nccl")
mesh = init_device_mesh("cuda", (2,))  # 一维网格：TP size = 2

class TPMLP(nn.Module):
    """Megatron 风格的张量并行 MLP"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # 第一个线性层：列切（输出维度拆到两张卡）
        self.fc1 = colwise_parallel(nn.Linear(in_dim, hidden_dim))
        # 第二个线性层：行切（输入维度拆到两张卡）
        self.fc2 = rowwise_parallel(nn.Linear(hidden_dim, in_dim))

    def forward(self, x):
        # x 是完整的（每卡都有副本）
        y = self.fc1(x)          # 输出天然分片：每卡只有 hidden/2 列
        y = F.gelu(y)            # 逐元素操作，分片状态直接算，零通信
        z = self.fc2(y)          # 每卡得到部分和
        # ReduceScatter/AllReduce 在这里自动发生（PairwiseParallel 语义）
        return z

model = TPMLP(1024, 4096).cuda()
x = torch.randn(8, 1024, device="cuda")   # 完整输入
out = model(x)                            # 每卡得到完整输出
print(out.shape)                           # torch.Size([8, 1024])
```

`colwise_parallel` / `rowwise_parallel` 做了三件事：把权重按对应维度切到各卡、在前向时正确处理输入的分片状态、在需要时插入通信原语。你写模型代码时完全不用关心这些细节。

### 3.2 更省事的方式：`parallelize_module` 整模块套用

```python
from torch.distributed.tensor.parallel import parallelize_module

class MyMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

model = MyMLP(1024, 4096).cuda()

# 一行声明式并行化：指定每一层怎么切
tp_plan = {
    "fc1": colwise_parallel,
    "fc2": rowwise_parallel,
}
parallelize_module(model, mesh, tp_plan)
# 现在模型已经是张量并行的了，forward 用法不变
```

### 3.3 Attention 的按头切分

```python
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

class TPAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # QKV 合并投影：按头切 = 按输出列切
        self.qkv = ColwiseParallel(nn.Linear(dim, 3 * dim))
        # 输出投影：行切
        self.proj = RowwiseParallel(nn.Linear(dim, dim))

    def forward(self, x):
        b, s, d = x.shape
        qkv = self.qkv(x)  # 每卡只持有自己那部分头对应的 3*dim/tp 列
        # 按本卡分到的头做标准注意力
        local_heads = self.n_heads // 2  # tp=2 举例
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, s, local_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, local_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, local_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, s, -1)
        return self.proj(attn)  # 行切 → 部分和 → 自动 AllReduce
```

### 3.4 高层封装：torchtitan / Megatron-LM

生产级训练不会手搭 TP。Meta 的 **torchtitan**（Llama 3 训练参考实现）里只需一行配置：

```python
# torchtitan 的 job config
parallelize.plan = "tp"        # 或 "fsdp_tp"、"pp_tp" 等组合
parallelize.tensor_parallel_degree = 8
```

它的 `parallelize_llama()` 函数自动把 Llama 的每个 `wq/wk/wv`（列切）、`wo`（行切）、`w1/w3`（列切）、`w2`（行切）按 Megatron 惯例切好。

### 3.5 推理侧：vLLM 里的 TP

```bash
vllm serve meta-llama/Llama-3-70B --tensor-parallel-size 4
```

vLLM 把 TP 用在推理上，且和 KV Cache 分片天然配合（第 011 期讲过）：每张卡只存自己那部分注意力头的 KV Cache，显存压力也跟着除以并行度。**TP 是 vLLM 多卡部署的默认并行方式**。

## 四、围绕该领域展开：TP 在生态中的位置

### 4.1 并行方式光谱

把主流并行方式放在一个谱系上看：

| 并行方式 | 拆什么 | 通信频率 | 通信量级 | 典型场景 |
|---------|--------|---------|---------|---------|
| **DP/DDP** | 数据 | 每步 1 次 | 梯度（与模型同大） | 模型放得下单卡 |
| **FSDP/ZeRO** | 参数+梯度+优化器状态 | 每层 2 次 | 参数级 AllGather/Scatter | 大模型训练 |
| **TP** | 层内矩阵 | **每层前反向 4 次** | 激活级 AllReduce | 机内 NVLink、推理 |
| **PP** | 层（纵深） | 微批次间 | 激活（点对点） | 超大模型跨机 |
| **SP** | 序列维度 | 每层若干 | 激活 AllGather | 长序列训练 |

关键认知：**这些方式不是互斥的，而是正交的维度**。Megatron 的 3D 并行 = TP（机内）× PP（跨机组纵向）× DP（跨机组横向）。torchtitan 支持 4D 并行再加 SP。

### 4.2 TP 与 FSDP 的关系：训练时的互补

一个常见的实际选型问题：70B 模型训练用 TP 还是 FSDP？

- **FSDP 优势**：不需要 NVLink 也能跑、代码侵入小（`fully_shard` 包一下）、容错好。**劣势**：通信发生在参数聚合时，计算与通信难重叠；每步都要「聚合 → 计算 → 再散开」。
- **TP 优势**：计算过程本身就是并行的，通信是计算图的一部分；**劣势**：每层 4 次通信，跨机带宽撑不住。

实践经验：**中等规模（≤70B）单机 8 卡训练，FSDP 往往更简单够用；追求极致吞吐或做千亿级训练，机内 TP + 机间 PP 是主流**。

### 4.3 TP 与序列并行 (SP) 的配合

TP 的一个隐藏痛点：LayerNorm 和 Dropout 是在**完整激活**上做的，不参与切分，成为显存瓶颈（激活显存没省）。**序列并行**（Megatron-SP / DeepSpeed-Ulysses）把激活沿序列长度再切一刀，LayerNorm 每卡只算自己那段序列。

组合公式：TP 的两次 AllReduce 在反向时可以改写为 ReduceScatter（配合 SP）+ AllGather，与 FSDP 的通信原语对偶——这就是第 007 期讲的通信原语在这里的落地。

### 4.4 与 torch.distributed 生态的关系

PyTorch 的演进路线值得注意：

- **1.x 时代**：只有 `torch.nn.parallel.DistributedDataParallel`，TP 要靠 Megatron 自己魔改
- **2.0 `fully_shard`**（FSDP2）：基于 DTensor 重写
- **2.4+ DTensor**：把「分片张量」变成一等公民。TP/FSDP/SP 的 API 都建立在 DTensor 之上——`colwise_parallel` 切完的权重就是一个 `DTensor`，它自己知道自己是哪个 mesh 上的哪个分片，通信由 Placement 统一描述

**DTensor 是理解 PyTorch 分布式未来的钥匙**：并行策略从「框架黑盒」变成「张量属性」，用户可以自定义 sharding placement 组合出新的并行方案。

### 4.5 TP 在推理引擎中的变体

- **vLLM**：TP + PagedAttention，每卡的 KV Cache 天然分片
- **TensorRT-LLM**：支持 TP + 「节点内 NVLink 域」自动感知
- **SGLang**：TP + 数据并行 Attention（DP-Attention），对超长上下文场景把 TP 的激活通信换成 DP 的负载均衡——一个「反 TP」的有趣设计：长序列时 TP 的 AllReduce 通信量随序列长度暴涨，DP-Attention 反而更划算

### 4.6 硬件视角：为什么 NVLink 决定了 TP 的边界

TP 对通信的要求是「**低延迟、高带宽、高频次**」。一次 AllReduce 4ms 的延迟，乘以层数（Llama-70B 有 80 层 × 每层 4 次）× 步数，就是数小时的天壤之别。这背后是硬件拓扑的问题：

- NVSwitch 全互联：任意两卡间点对点带宽一致，TP 任意切法都高效
- PCIe：带宽低一个数量级，TP=8 时通信占比可能超过 40%
- 跨机 IB：除非万不得已（超大模型），不要把 TP 拉出机箱

国产加速器（如 IFWA 这类新硬件）落地 vLLM 时，TP 支持的优先级极高，而**卡间互联拓扑和集合通信库的延迟/带宽**直接决定 TP 的实际收益——这也是适配工作中 profiling（第 014 期）要重点测的指标。

## 五、什么时候该用 / 不该用

**该用 TP：**
- 推理部署大模型，单卡放不下（vLLM `--tensor-parallel-size`）
- 训练千亿级模型，机内 NVLink 互联
- 想压榨机内 8 卡的极限吞吐（TP 计算与通信重叠好）
- 层很「胖」（hidden dim 大），切分后每卡仍有足够的计算强度

**不该用 TP：**
- 模型单卡放得下 → DDP 更简单
- 跨机部署、带宽有限 → FSDP 或 PP
- 模型很小（层内矩阵小）→ 通信开销占比过高，得不偿失
- 卡数不是注意力头数的约数 → 头切不齐，效率损失
- 没有 NVLink 的 PCIe 机器上拉高 TP 度

## 六、常见坑

**坑 1：TP 度切过头了**
8 卡机器上对 7B 模型用 TP=8，每卡只分到 0.875B 参数，单卡计算量太小，通信占比飙到 50%+，速度反而比 TP=2 慢。**经验法则：每卡至少分到 1-2B 以上参数，TP 才划算**。7B 模型 TP=1 或 2 通常最优。

**坑 2：LayerNorm 和 Dropout 没切，激活显存爆了**
只做 TP 不做 SP，权重显存省了，但激活里 LayerNorm 的部分仍是每卡完整副本。长序列 + 大 batch 训练时这是常见 OOM 来源。Megatron 的解法是 TP+SP 联用。

**坑 3：TP 与 FSDP 套用时切分维度打架**
`fully_shard` 默认在 DP mesh 维分片，如果权重已经被 TP 切过，DTensor 需要正确处理嵌套 mesh（2D mesh: [DP, TP]）。用旧式 `colwise_parallel` + FSDP1 混搭会出各种诡异 shape 错误——**坚持用同一代 API（DTensor 系）**。

**坑 4：在普通以太网跨机拉 TP，性能悬崖**
TP 的通信频率是每层 4 次，跨机时每次 AllReduce 都吃满网络 RTT。看到「GPU 利用率 90% 但吞吐只有理论值 1/5」的经典症状，先查 TP 是不是跨机了。

## 七、一句话总结

> 张量并行把「拆数据」换成「拆矩阵本身」，用分块矩阵乘法的数学性质换取单层计算的多卡分担；它每层都要通信，因此被 NVLink 锁死在机内，但正是这种深度耦合让它成为大模型推理部署和 3D 并行训练中不可替代的一维。

## 今日练习

<details><summary>今日练习</summary>

**题目**：一个 Transformer 的 MLP 层，`fc1` 是 `(4096, 16384)` 的权重，`fc2` 是 `(16384, 4096)`，隐藏维度 4096。若用 TP=2（两张卡）按 Megatron 惯例切分，请回答：

1. 每张卡上 `fc1` 和 `fc2` 的权重形状各是多少？
2. 前向一次 MLP 块，最少需要几次集合通信？分别是什么原语？
3. 如果两层都改成列切，会发生什么？

**参考答案**：

1. `fc1` 列切：每卡 `(4096, 8192)`；`fc2` 行切：每卡 `(8192, 4096)`。
2. **一次**。`fc1` 列切后输出天然分片，GeLU 逐元素无需通信；`fc2` 行切后每卡得部分和，块尾一次 **AllReduce** 求和。（若要接 SP，可改写为 ReduceScatter。）
3. `fc2` 也列切的话，输出在「隐藏维度」上分片，但下游需要的是完整输出——要么补一次 **AllGather** 拼回完整激活（多一次通信），要么后续所有层都得在分片状态下重推导。Megatron 选择「列切→行切」配对，正是为了让分片状态在块内自然流转、块尾一次通信收尾——**切分方案的本质是用数学结构换通信次数**。

</details>
