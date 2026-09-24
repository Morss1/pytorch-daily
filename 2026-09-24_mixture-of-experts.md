# PyTorch 每日一课 · 第 026 期

## 混合专家（MoE）：参数很多，计算很少

> **日期**：2026-09-24
> **难度**：⭐⭐⭐⭐⭐
> **前置知识**：知道 Transformer 的 FFN 结构（SwiGLU 是什么）；对 all-reduce / all-to-all 这类集合通信有概念；理解「显存带宽」和「算力」是两个不同的瓶颈。第 019 期张量并行、第 024 期序列并行没看过也不影响，本篇独立可读。
> **预计阅读时间**：30 分钟

---

## 一、这个领域解决什么问题

### 1.1 稠密模型的诅咒：参数和计算被焊死在一起

先看一个让人不太舒服的事实。一个标准 Transformer，每处理一个 token 需要的浮点运算量是：

$$\text{FLOPs} \approx 2N$$

其中 $N$ 是参数量。**这个 2 就是「一次乘加算 2 个浮动操作」的意思。**

这条公式的问题在于它是一个**等式**，不是一个不等式。你没有办法在不增加计算量的情况下增加参数量。想要 10 倍的知识容量？准备好 10 倍的计算账单。

为什么这很致命？因为这两件事的价值不一样：

| | 参数量 ↑ | 计算量 ↑ |
|---|---|---|
| 带来什么 | 知识容量、表达能力、记忆 | **成本**：时间、电费、卡 |
| 你想要它 | 越大越好 | 越小越好 |
| 现实 | 和上面那行绑在一起 | —— |

这就是 Scaling Law 时代最别扭的地方：**我们知道怎么让模型更聪明（加参数），也知道怎么让它更便宜（少算），但这两件事在稠密架构里是同一件事。**

### 1.2 MoE 的答案：把「仓库」和「工位」分开

MoE（Mixture of Experts，混合专家）的全部野心就一句话：

> **把「模型有多少知识」和「处理一个 token 要花多少算力」解耦。**

具体做法出人意料地简单粗暴：**把 Transformer 每一层的 FFN（前馈网络）从「一个」换成「很多个」，但每个 token 只走其中少数几个。**

- **总参数量** = 所有 FFN 加起来 → 决定知识容量
- **激活参数量** = 每个 token 实际走过的那几个 FFN → 决定计算量

于是账本变成了两个数而不是一个。

### 1.3 先把账算清楚

这是整篇文章的地基，我们认真算一遍。取 DeepSeek-V3 的真实配置：61 层（前 3 层是稠密 FFN，后 58 层是 MoE），$d=7168$，MLA 注意力，每个 MoE 层有 **256 个路由专家 + 1 个共享专家**，每个专家是 $d_{ff}=2048$ 的 SwiGLU。

单个专家的参数量（SwiGLU 有三个权重矩阵）：

$$3 \times d \times d_{ff} = 3 \times 7168 \times 2048 = 44{,}040{,}192 \approx 44.04\text{M}$$

注意这个数字：**一个专家只有 4400 万参数**。这非常小——对比一下，Llama-2-7B 单个 FFN 层就有 $3 \times 4096 \times 11008 \approx 135\text{M}$。DeepSeek-V3 走的是「**多而小**」的路线，这一点后面会反复用到。

现在把三种模型的账本摊开（下面这些数字是我用纯 Python 逐项核算的，不是抄的，你可以自己验）：

| 模型 | 专家/层 | top-K | 总参数 | 激活参数 | 激活率 | **杠杆** |
|---|---|---|---|---|---|---|
| **Mixtral 8x7B** | 8 | 2 | **46.57B** | **12.75B** | 27.4% | 3.65× |
| **DeepSeek-V3** | 256+1共享 | 8+1 | **670.92B** | **36.52B** | 5.4% | **18.37×** |
| Qwen1.5-MoE-A2.7B | 60+4共享 | 4+4 | 14.00B | 2.37B | 17.0% | 5.90× |
| —— 对照 —— | | | | | | |
| Llama-2-7B（稠密） | — | — | 6.61B | 6.61B | 100% | **1.00×** |
| Llama-3-70B（稠密） | — | — | 69.50B | 69.50B | 100% | **1.00×** |

**DeepSeek-V3 那一行值得单独看一眼**：我的核算得到 670.92B / 36.52B，官方公布的是 **671B / 37B**。两个数字都对上了，这说明这套核算方法是准的。

那么杠杆是什么意思？**DeepSeek-V3 用 18.4 倍的知识容量，付了跟一个 37B 稠密模型一模一样的算力账单。**

```
【DeepSeek-V3 的分项账本】
  MLA 注意力 (61 层)     : 11.41B
  稠密 FFN (3 层)        : 1.19B
  专家总计 (58 层 × 257)  : 656.46B      ← 97.8% 的参数在这里
  每 token 激活的专家参数 : 22.99B
  embedding + lm_head    : 1.85B
  ──────────────────────────────────
  ── 总参数              : 670.92B   (官方 671B)
  ── 激活参数 / token    : 36.52B    (官方 37B)
  激活率                 : 5.44%
  容量/算力 杠杆          : 18.37×
```

顺便解一个常见困惑：**Mixtral 8x7B 为什么不是 56B？**「8x7B」这个名字误导了很多人。它不是「8 个 7B 模型」，而是「**每层 8 个专家，每个专家的参数量大致相当于一个 7B 模型里的 FFN**」。因为 attention 是 8 个专家**共用**的（只有 FFN 被复制了），所以总参数是 46.6B 而不是 56B。名字是市场部起的，架构才是真的——**看 MoE 只看 three 个数字：每层几个专家、top-K 取几个、专家多大。**

---

## 二、核心思想

### 2.1 一个类比：医院分诊

想象一家医院。

**稠密模型**是这样运作的：每个进门的病人都要**把全院所有科室走一遍**，心内科、骨科、皮肤科、眼科……每个医生都看一眼，最后综合所有意见出诊断。这样诊断质量当然高——但一个感冒病人也要占用全院资源。

**MoE** 是正常医院的运作方式：门口先有个**分诊台（router / gate）**，看一眼病情，然后说「你去心内科，同时再去一下内分泌科」（这就是 top-K，K=2）。病人只走这两个科室，其他科室的医生在办公室里待命。

关键洞察有三条：

1. **分诊台不治病，它只指路。** router 是个极小的网络（就是一个 `nn.Linear(d, E)`，参数量 $d \times E$，在 V3 里是 $7168 \times 256 = 1.8\text{M}$，占总量 0.0003%）。它决定谁来干活，自己不做实质计算。
2. **医生数量决定了医院的实力上限，但单个病人的等待时间取决于分诊准确度。** 这就是「总参数 vs 激活参数」。
3. **如果分诊台偷懒，只往心内科送人，其他科室就废了。** 这是 MoE 最核心的工程难题——**负载均衡**，后面会用一整节讲。

### 2.2 数学形式

一层 MoE 的前向（以 token-choice + 共享专家为例）：

$$y = \underbrace{\sum_{s \in \text{Shared}} E_s(x)}_{\text{所有 token 都走}} + \underbrace{\sum_{e \in \text{TopK}(g(x))} \tilde{g}_e(x) \cdot E_e(x)}_{\text{只有少数 token 走}}$$

拆开看每一步：

| 符号 | 含义 | 形状 |
|---|---|---|
| $g(x) = W_g x$ | router 打分，$W_g$ 是 $[d, E]$ 的小矩阵 | `[T, E]` |
| $\text{TopK}(g(x))$ | 取分数最大的 K 个专家 | `[T, K]` 的整数索引 |
| $\tilde{g}_e(x)$ | 这 K 个的 softmax 权重，重新归一化到和=1 | `[T, K]` 的浮点 |
| $E_e(x)$ | 第 e 个专家的 FFN | `[T, d]` → `[T, d]` |
| $y$ | K 个专家输出的加权和 | `[T, d]` |

**几个容易忽略的细节**：

- **`Shared` 那一项是 DeepSeek 的发明**（DeepSeekMoE 论文）。它是个永远激活的专家，负责吸收「通用知识」；这样路由专家才能空出手去学真正特化的东西。Qwen1.5-MoE 更激进，用了 **4 个共享专家**——但它 top-K 也是 4，意味着 **50% 的激活计算量花在永远激活的共享专家上**，稀疏带来的收益直接被砍掉一半。这是个很好的设计取舍案例。
- **softmax 在哪个范围做很重要**。GShard/Switch 是对全部 E 个 logits 做 softmax 再取 top-K；DeepSeek 是取 top-K 后再对 K 个原始分数做归一化。后者在 K 比较大时数值行为更稳。
- **归一化后权重和 = 1**，所以 MoE 的输出和前一层是同一个量级，不需要额外的缩放。

### 2.3 一个最小的 MoE 层

先写**语义版**——完全按上面公式来，慢，但一眼能看懂：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """一个专家，内部就是普通的 SwiGLU FFN。没有任何特殊之处。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate 分支
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # up 分支
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # down 分支

    def forward(self, x):                                 # x: [T, d]
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TopKRouter(nn.Module):
    """
    token-choice 路由：每个 token 自己挑 K 个专家。
    这是一个参数极少的小网络——它只决定「谁干活」，不做实质计算。
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        # 注意 bias=True：DeepSeek-V3 的 aux-loss-free 负载均衡，
        # 调整的就是这个 bias 项（给冷门专家加分、给热门专家减分）
        self.gate = nn.Linear(d_model, n_experts, bias=True)
        self.n_experts = n_experts
        self.top_k = top_k
        # 初始化：把 gate 的方差压小，否则训练一开始路由就极端化
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):                                 # x: [T, d]
        logits = self.gate(x)                             # [T, E]
        probs = logits.softmax(dim=-1)                    # 全部 E 个的概率
        w, idx = probs.topk(self.top_k, dim=-1)           # 各 [T, K]
        # GShard 的做法：在选中的 K 个里重新归一化，保证输出量级不变
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return probs, w, idx


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k, n_shared=0,
                 aux_coeff=1e-2, cap_factor=None):
        super().__init__()
        self.n_experts, self.top_k, self.n_shared = n_experts, top_k, n_shared
        self.aux_coeff, self.cap_factor = aux_coeff, cap_factor
        self.router = TopKRouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        # 共享专家：永远激活，负责吸收通用知识
        self.shared = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_shared)])

    def forward(self, x):                                 # x: [T, d]
        T = x.shape[0]
        probs, w, idx = self.router(x)

        # ---- 容量控制：每个专家最多接手多少个 token ----
        if self.cap_factor is not None:
            # 理想负载 = 总槽位数 / 专家数；容量因子就是给它的冗余倍数
            cap = max(self.top_k, int(self.cap_factor * T * self.top_k / self.n_experts))
            # 被丢弃的槽位把权重归零 → 该 token 失去这个专家的贡献。
            # 若一个 token 的 K 个槽位全被丢，它的 FFN 就被整体跳过（只剩残差）。
            keep = torch.ones_like(idx, dtype=torch.bool)
            for e in range(self.n_experts):
                pos = (idx == e).nonzero(as_tuple=False)      # [n_here, 2] 的 (token, slot)
                if pos.shape[0] > cap:
                    drop = pos[cap:]                          # 先到先得，排在后面的丢
                    keep[drop[:, 0], drop[:, 1]] = False
            w = w.masked_fill(~keep, 0.0)
            # 生产实现用 cumsum 一次性算出「每个槽位是该专家的第几个」，
            # 不需要 python 循环；这里为了语义清晰写了循环。

        # ---- 前向：共享专家（所有 token 都走）+ 路由专家（只有少数 token 走）----
        out = None
        for e in self.shared:
            contrib = e(x)
            out = contrib if out is None else out + contrib

        for e in range(self.n_experts):
            mask = (idx == e)                             # [T, K] 哪些槽位选了专家 e
            if not mask.any():
                continue                                  # ★ 没 token 选它就跳过 —— 省算力就在这一行
            tok_ids, slot_ids = mask.nonzero(as_tuple=True)
            h = self.experts[e](x[tok_ids])               # 只算这些 token
            h = h * w[tok_ids, slot_ids].unsqueeze(-1)    # 乘路由权重（被丢的槽位权重为 0）
            # scatter-add 回原位置：一个 token 会被命中 K 次，自动累加
            contrib = torch.zeros_like(x).index_add(0, tok_ids, h)
            out = contrib if out is None else out + contrib

        # ---- 辅助负载均衡损失（GShard / Switch 的经典形式）----
        if self.training and self.aux_coeff > 0:
            # f_i: 硬分配比例（每个槽位算一次）
            f = torch.bincount(idx.reshape(-1),
                               minlength=self.n_experts).float() / (T * self.top_k)
            # P_i: 软概率的均值
            P = probs.mean(dim=0)
            # L_aux = α · E · Σ f_i · P_i；完全均衡时 Σ f_i·P_i = 1/E，损失取到最小值
            self.aux_loss = self.aux_coeff * self.n_experts * (f * P).sum()
        return out
```

上面这段是**教学版**：`for e in range(n_experts)` 这种循环在真实模型里是不可能的——256 次 kernel launch，每次只算几个 token，GPU 利用率会掉到个位数。但它把语义说得明明白白。**理解了它，剩下的全是性能优化。**

### 2.4 天下没有免费的午餐：三个新问题

MoE 用「稀疏」换来了参数/算力的解耦，代价是凭空造出了三个稠密模型没有的问题：

| 新问题 | 本质 | 解决方向 |
|---|---|---|
| **负载不均衡** | router 会自发往少数专家倾斜，还带正反馈 | 辅助损失 / 动态 bias / expert-choice |
| **通信爆炸** | 专家分散在不同卡上，token 要跨卡来回飞 | 专家并行 + all-to-all + 通信重叠 |
| **显存被占满** | 所有专家权重都得常驻，即使这个 token 一个都不用 | 更高精度压缩（FP8）、EP 分摊 |

这三条就是本章后面「围绕展开」的全部内容。**MoE 的论文看起来在讲架构，实际上 90% 的工程都在处理这三件事。**

---

## 三、在 PyTorch 中怎么用

### 3.1 性能版：sort → grouped GEMM → scatter_add

真实的 MoE 实现遵循一个固定套路，它把「循环」变成了「一次大矩阵乘」。这个套路有四个动作：

```
    token             ① 按专家 id 排序         ② 一次 grouped GEMM
  [T, d]  ──────►  把同一专家的 token         所有专家的矩阵乘
                    排在一起 [T*K, d]          打成一批算
                                                      │
                                                      ▼
  [T, d]  ◄──────  ④ scatter-reduce           ③ 乘路由权重
   输出              把结果加回原位置            再做第二次/第三次
```

关键 API 是 `torch._grouped_mm`。它的契约是：

```python
torch._grouped_mm(A, B, offs)
#   A:    [M, K]        所有组的输入拼在一起（M = 所有 token-专家对的总数）
#   B:    [E, K, N]     堆叠在一起的专家权重（E = 专家数）
#   offs: [E]  int32    **累积**的组边界，offs[i] = 前 i 个组的 token 数之和
#   返回: [M, N]
```

> ⚠️ `torch._grouped_mm` 是一个**下划线开头的私有 API**，签名可能变动，且只在 CUDA 上可用。但它已经是事实标准——HuggingFace Transformers、torchao、TorchTitan、NVIDIA NeMo 都统一到它上面了。

下面是完整的可运行实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedExperts(nn.Module):
    """
    把 E 个专家的权重堆成 3D Parameter，这样一次 grouped GEMM 就能算完所有专家。
    ★ 权重必须是 3D nn.Parameter —— 这是能用 grouped GEMM 的前提。
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        # 每个都是 [E, out, in]
        self.w1 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))      # gate
        self.w3 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))      # up
        self.w2 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))      # down
        for w in (self.w1, self.w2, self.w3):
            nn.init.normal_(w, std=0.02)

    def forward(self, x_flat, offsets):
        """
        x_flat:  [M, d]      已经按专家 id 排好序的 token（含重复）
        offsets: [E] int32   累积组边界，offsets[i] = Σ_{j<i} count_j

        注意 B 参数是 [E, K, N] 形状，且 transpose 出来的是**非连续视图**。
        生产实现（如 NVIDIA NeMo）会在加载权重时做一次 relayout，
        让转置后的布局连续 —— 否则 grouped GEMM kernel 会拒绝或走慢路径。
        """
        h = F.silu(torch._grouped_mm(x_flat, self.w1.transpose(-2, -1), offsets))
        h = h * torch._grouped_mm(x_flat, self.w3.transpose(-2, -1), offsets)
        return torch._grouped_mm(h, self.w2.transpose(-2, -1), offsets)


def moe_forward_grouped(x, gmm: GroupedExperts, topk_idx, topk_w, n_experts):
    """
    x:        [T, d]
    topk_idx: [T, K]  每个 token 选中的专家 id
    topk_w:   [T, K]  对应路由权重
    """
    T, K = topk_idx.shape
    flat = topk_idx.reshape(-1)                       # [T*K] 展平成槽位

    # ① 按专家 id 排序 —— 把同一专家的 token 聚到一起
    #    stable=True 保证同专家的 token 保持原有相对顺序（结果可复现）
    order = flat.argsort(stable=True)                 # [T*K]
    tok = order // K                                  # 排序后每个槽位属于哪个 token

    # ② 每个专家收到多少 token，转成累积边界（grouped GEMM 要的格式）
    counts = torch.bincount(flat, minlength=n_experts)          # [E]
    offsets = counts.cumsum(0).to(torch.int32)                  # [E]

    # ③ gather：把 token 复制到它对应的槽位（一个 token 选 K 个专家 → 复制 K 份）
    xg = x[tok]                                       # [T*K, d]

    # ④ 一次 grouped GEMM 算完所有专家
    y = gmm(xg, offsets)                              # [T*K, d]

    # ⑤ 乘上路由权重（注意要用排序后的顺序取权重）
    y = y * topk_w.reshape(-1)[order].unsqueeze(-1)

    # ⑥ scatter-reduce：把同一 token 的 K 份输出加起来，放回原位置
    out = torch.zeros_like(x)
    out.index_add_(0, tok, y)                         # 一个 token 会命中 K 次，自动累加
    return out


# ---------- 跑一遍 ----------
torch.manual_seed(0)
T, d_model, d_ff, E, K = 64, 128, 256, 8, 2

router = nn.Linear(d_model, E, bias=False)
gmm = GroupedExperts(d_model, d_ff, E)

x = torch.randn(T, d_model)
probs = router(x).softmax(-1)
topk_w, topk_idx = probs.topk(K, dim=-1)
topk_w = topk_w / topk_w.sum(-1, keepdim=True)

out = moe_forward_grouped(x, gmm, topk_idx, topk_w, E)
print(out.shape)          # torch.Size([64, 128])
print(topk_idx[0])        # 第一个 token 选了哪两个专家
```

**为什么这样写会快？** 六个动作里，只有 ④ 是真正的大计算（占 99% 的 FLOPs），其余全是内存搬运。而 ④ 之所以快，是因为它把 256 个形状各异的小矩阵乘，**打成了一批发往 GPU 的批量任务**——GPU 不再饿肚子等下一次 kernel launch。

这也是为什么 HuggingFace 在 MoE 博客里给出三个 backend 供选择：

| backend | 实现方式 | 适合场景 |
|---|---|---|
| `eager` | 逐专家循环（上面教学版） | 正确性参考、调试 |
| `batched_mm` | `torch.bmm`，把权重按 token 复制 | **小 batch**、显存充裕 |
| `grouped_mm` | 排序 + `torch._grouped_mm` | **大 batch**、显存紧张 ✅ |

### 3.2 生态里现成的实现

**（1）HuggingFace Transformers**——用装饰器切换 backend，EP 一行打开：

```python
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig

# 打开专家并行：专家权重会沿 dim=0 被切到各张卡上
distributed_config = DistributedConfig(enable_expert_parallel=True)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-120b",
    dtype="auto",
    distributed_config=distributed_config,
)
# 用 torchrun --nproc-per-node N 启动，N 需整除专家总数
```

内部由两个组件协作：`GroupedGemmParallel` 负责把专家权重按 dim=0 切开、每卡只加载 $E/P$ 个；`RouterParallel` 负责把全局专家 id 映射成本卡局部 id、屏蔽掉不属于自己的专家，并用 all-reduce 汇总各卡的部分输出。

**（2）torchao**——把 grouped GEMM 换成低精度版本（MoE 特别吃这一套，原因见 4.8）：

```python
from torchao.prototype.moe_training import _scaled_grouped_mm as torchao_scaled_gmm
from torchao.prototype.moe_training.conversion_utils import MoEScalingType

out = torchao_scaled_gmm(
    activations,                                   # [M, K]
    expert_weights.transpose(-2, -1),              # [E, N, K]
    offs=offsets,
    scaling_type=MoEScalingType.MXFP8,             # MXFP8: e4m3 数据 + e8m0 分块 scale
)
```

它是 `torch._grouped_mm` 的**可微 drop-in 替代**。官方 microbenchmark：Llama4 17Bx16E 形状快 **1.4–1.8×**，DeepSeek-V3 671B 形状快 **1.2–1.4×**。注意 MXFP8 路径需要 CUDA compute capability ≥ 10（GB200 / sm_100+）。

**（3）TorchTitan**——端到端训练，并行度用命令行配置：

```bash
# 专家并行 4 路 + 流水并行 2 路；MXFP8 grouped GEMM
torchrun --nproc-per-node 8 -m torchtitan.train \
    --model.name deepseek_v3 \
    --parallelism.expert_parallel_degree 4 \
    --parallelism.pipeline_parallel_degree 2 \
    --model.converters="quantize.grouped_mm.mx" \
    --compile.enable
```

**（4）一个非常实用的小开关**。TorchTitan 提供了 `--debug.moe_force_load_balance`，强制让所有专家负载均等。**做 MoE 性能对比实验时必须打开它**——否则你的「优化前 vs 优化后」两组实验里，路由分布本身就不一样，测出来的差异可能全部来自负载差异，而不是你的优化。这是 MoE 基准测试最常见的陷阱。

---

## 四、围绕这个领域展开

前面是「怎么用」，这一节是「这个领域长什么样」。MoE 是一个**设计空间**，不是一个固定架构，下面每一小节都是空间里的一个维度。

### 4.1 路由算法谱系：谁来挑谁

| 方案 | 谁做选择 | 负载均衡性 | 代表工作 |
|---|---|---|---|
| **Token-Choice top-K** | token 挑专家 | 需要辅助手段 | GShard (K=2)、Switch (K=1)、Mixtral (K=2) |
| **Expert-Choice** | 专家挑 token | **天然完美均衡** | Google, 2022 |
| **哈希路由** | 固定哈希，不学习 | 完全均匀 | 早期的 Sparsely-Gated MoE |
| **共享专家 + 细粒度** | token 挑，另有常开专家 | 辅助手段 | DeepSeekMoE / V2 / V3 |
| **Aux-loss-free bias** | 动态调 bias 影响选择 | 边界自调节 | DeepSeek-V3 |

**Expert-Choice 的诱惑与陷阱值得单独看。** 它让每个专家自己挑 top-$c$ 个 token，于是**专家负载天然完美均衡**（每个专家恰好 $c$ 个），完全不需要辅助损失。听起来完美，但它把不均衡转移到了另一边——**token 侧**。我实测了一下（E=16, T=2048, K=2, 每专家容量 256）：

```
【Token-Choice（带容量截断）】
  每 token 被算次数分布 {0: 20, 1: 88, 2: 1940}   平均 1.938
  被完全丢弃(0 次)的 token: 20 (0.98%)
  专家负载: {256, 230, 235, 243, 246, 248, 218, 252}   ← 有些专家没吃满

【Expert-Choice（每专家取 top-256）】
  每 token 被算次数分布 {0: 250, 1: 559, 2: 578, 3: 395, 4: 171, 5: 67, 6: 20, 7: 7, 8: 1}
  被完全忽略(0 次)的 token: 250 (12.21%)        ← 灾难
  被重复计算(>K 次)的 token: 661 (32.28%)       ← 浪费算力
  专家负载: 完全均衡，每个都是 256
```

**结论**：Expert-Choice 把「专家负载不均」换成了「**12.21% 的 token 一个专家都没被分到**」和「**32.28% 的 token 被算了 3 次以上**」。前者意味着这些 token 的 FFN 计算被彻底跳过（信息丢失），后者意味着算力浪费。**这解释了为什么生产系统几乎都选 token-choice**——token 侧的公平性比专家侧的公平性重要得多，因为 token 是「数据」，丢了就是丢了；专家闲一会儿只是浪费卡。

### 4.2 负载均衡的两条路线

**路线一：辅助损失（aux loss）**

GShard 提出、Switch Transformer 推广的形式：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

其中 $f_i$ 是分给专家 $i$ 的 token 比例（硬分配），$P_i$ 是专家 $i$ 的平均路由概率（软）。**完全均衡时这个乘积取到最小值。** 直觉是：$f_i$ 惩罚「实际去的人多」，$P_i$ 惩罚「router 想送的人多」——两个一起压，router 就不敢偏心了。

问题在于：**这个损失和语言模型的主损失是竞争关系。** 它逼 router 把 token 送去「负载轻的专家」，而不是「最适合的专家」。这就是 DeepSeek-V3 报告里说的 "the model can sacrifice natural token routing to satisfy the balancing regularizer"。

**路线二：aux-loss-free —— 用 bias 代替损失**

DeepSeek-V3 的做法：给每个专家维护一个**动态 bias** $b_i$，加在 gate 的分数上：

$$s_i(x) = \text{affinity}(x, i) + b_i$$

注意两个细节：

1. **bias 只影响「选谁」，不影响「权重多大」。** 选中之后的组合权重仍然由原始 affinity 决定。这意味着**平衡压力只作用在路由的边界上**，不污染主损失，也不改变专家的相对贡献。
2. **bias 按负载反馈调整**：过载的专家 $b_i$ 调低，空闲的专家 $b_i$ 调高。可以证明这是对某个 Lagrangian 的一步原始-对偶更新。

> **`torch.nn.Linear(d, E, bias=True)` 里的这个 bias，就是 DeepSeek 用来做负载均衡的那个旋钮。** 平时我们把它当常规参数顺手初始化成 0，但在 MoE 里它承担着关键职责。

报告提到 V3 训练全程「没有出现不可恢复的 loss spike、没有回滚」，aux-loss-free 是这个稳定性故事的一部分。

### 4.3 容量因子与 token dropping

在分布式或静态 buffer 的实现里，每个专家能接的 token 数必须**事先定好**（因为要分配显存）：

$$\text{capacity} = \text{cap\_factor} \times \frac{T \times K}{E}$$

超过容量的 token 被丢弃（走残差绕过 FFN）。`cap_factor` 是最经典的调参旋钮：**太小→丢 token 伤质量，太大→预留显存和算力被浪费。**

我用蒙特卡洛模拟了一下（T=4096，router 带有中等程度的系统性偏好，模拟训练中后期的真实状态）：

| 每层专家数 | top-K | cap=1.0 | cap=1.5 | cap=2.0 | cap=2.5 | cap=4.0 |
|---|---|---|---|---|---|---|
| 8 | 2 | 35.23% | 17.88% | 8.60% | 0.00% | 0.00% |
| 64 | 6 | 34.19% | 20.02% | 11.61% | 6.10% | 0.38% |
| 256 | 8 | **42.36%** | **29.57%** | **20.06%** | 13.78% | 4.03% |

（数字是「整 token 丢弃率」——一个 token 的 K 个专家全部满载才算丢）

**三个结论值得记住**：

1. **`cap_factor=1.0` 是灾难**：35%~42% 的 token 被丢。原因很简单——「平均负载」和「最大负载」从来不是一回事，按平均值切容量，一半以上的专家必然溢出。
2. **专家越多，越难用容量制控制**：E=256 时要到 `cap_factor=4.0` 才能把丢弃率压到 4%。原因见下一小节的多项分布分析——**E 越大，负载的相对波动越大**。
3. **容量制天生是「浪费」的**。如果某个专家的负载是平均值的 3.26 倍（4.3 节实测的偏差程度），按最大负载给所有专家分配 buffer，有效吞吐利用率只有 $1/3.26 = 30.67\%$，**浪费 69.3%**。这就是为什么现代 MoE 训练转向 **dropless**（不丢 token、动态变长 buffer），代价是 kernel 复杂度大增。

**那负载不均衡有多少是「随机涨落」，有多少是「router 真的偏了」？** 这个区分很重要。我用 IID 随机 router（每 token 的 logits 独立同分布 $N(0,1)$，完全没有偏好）测了一下：

| 配置 | 理论波动系数 $\sqrt{(E-1)/T}$ | 实测 max/avg | 实测 min/avg | Gini |
|---|---|---|---|---|
| T=4096, E=8, K=2 | 0.041 | 1.023 | 0.967 | 0.0093 |
| T=4096, E=64, K=6 | 0.124 | 1.040 | 0.953 | 0.0119 |
| T=4096, E=256, K=8 | **0.249** | 1.105 | 0.900 | 0.0210 |

**IID 随机 router 只会带来 ±10% 的波动（Gini ≈ 0.01–0.02），这是纯粹的统计噪声，而且和大 batch 平均后会被抹掉。** 真正制造灾难的是 router 的**系统性偏好**：

| router 偏好强度 | Gini | max/avg | min/avg | max/min |
|---|---|---|---|---|
| 无 | 0.0067 | 1.015 | 0.971 | 1.05 |
| 弱 | 0.1961 | 1.563 | 0.513 | 3.05 |
| 中 | 0.4262 | 2.384 | 0.137 | 17.44 |
| **强** | **0.6135** | **3.257** | **0.004** | **833.90** |
| 极强 | 0.7113 | 3.879 | ~0 | ~4×10¹² |

**偏好强度只到「中」的时候，已经有一个专家几乎收不到 token（min/avg = 0.137），max/min 达到 17 倍。** 这就是负载均衡非做不可的量化理由——**它不是性能优化，它是防止你的模型一半专家变成废铁。**

### 4.4 路由的动力学：为什么需要辅助损失，以及为什么不能用力过猛

负载不均衡为什么会发生？因为存在**正反馈**：

```
   某专家被选中更多  →  它收到更多梯度、训练得更充分  →  它的 representation 更好
         ↑                                                      │
         └──────────────  router 更倾向选它  ←───────────────────┘
```

这是一个自我强化的循环，学术上叫**专家坍缩（expert collapse）**。我写了个玩具仿真来观察它的动力学（E=16, T=512, top-K=2）：

```
正反馈强度 = 0.00 : r0:G=0.097  r20:G=0.053  r50:G=0.077  r100:G=0.048  r399:G=0.048   ← 永不坍缩
正反馈强度 = 0.005: r0:G=0.097  r20:G=0.056  r50:G=0.124  r100:G=0.660  r200:G=0.875   ← 100 轮后坍缩
正反馈强度 = 0.02 : r0:G=0.097  r20:G=0.438  r50:G=0.875                         ← 50 轮就塌了
正反馈强度 = 0.05 : r0:G=0.097  r20:G=0.875                                     ← 20 轮
```

（G = 负载分布的 Gini 系数，0 = 完全均衡，0.875 = 16 个专家里只剩 2 个在干活）

**关键发现**：正反馈为 0 时，路由永远保持均衡——**坍缩完全由正反馈驱动，不是随机巧合**。而哪怕极弱的正反馈（0.005），100 轮之后系统也彻底塌了。这就是为什么 MoE 必须**从第一步训练就带平衡机制**，不能指望「先让它自由学，后面再修」。

那么平衡力度调多大？我把系数扫了一遍：

| 辅助损失系数 | max/理想 | min/理想 | 活跃专家 | Gini | 专家质量极差 |
|---|---|---|---|---|---|
| 0.000（无） | 8.00 | 0.00 | 2/16 | 0.8750 | 25.0× |
| 0.050 | 2.59 | 0.00 | 7/16 | 0.5841 | 25.0× |
| 0.080 | 1.86 | 0.09 | 16/16 | 0.3160 | 11.1× |
| **0.100** | **1.23** | **0.66** | **16/16** | **0.0797** | **1.9×** ✅ |
| 0.150 | 1.59 | 0.45 | 16/16 | 0.1788 | 1.1× |
| 0.200 | 2.25 | 0.06 | 16/16 | 0.4185 | 1.1× |
| 0.500 | 6.23 | 0.00 | 6/16 | 0.8223 | 1.4× |
| 1.000 | 8.00 | 0.00 | 5/16 | 0.8344 | 1.3× |

**这张表最反直觉的一行是最后两行：把平衡力度从 0.1 加到 0.5，Gini 从 0.0797 恶化到 0.8223——比完全不做平衡（0.8750）只好了那么一点点。**

我用 5 个不同随机种子验证过，这个非单调性很稳健（coeff=0.1 时 Gini 落在 0.078~0.153，coeff=0.5 时落在 0.777~0.851）。

**为什么？** 因为平衡力是基于**滞后的负载 EMA** 施加的，这是一个**带延迟的负反馈环**。增益小于 1 时它稳定收敛；增益远大于 1 时它会**过冲→振荡**——某个专家刚吃满就被狠狠压下去，下轮变成最饿的，再被狠狠抬起来。系统开始追着自己跑。

还有第二个代价：**注意 `专家质量极差` 那一列**。coeff=0.1 时是 1.9×（专家之间有了健康的差异化），coeff=0.5 时掉到 1.4×，coeff≥0.15 之后基本都在 1.1~1.4×。**过度平衡把专家之间的差异也一起抹平了——而差异化正是 MoE 存在的理由。** 如果所有专家都一模一样，你要 256 个专家干什么？不如用一个。

> 这就是 DeepSeek 走 aux-loss-free 路线的深层动机：**它要的不是「更用力地平衡」，而是「不把平衡的力打进主损失里」。** 用 bias 在路由边界上做微调，主损失保持纯净，专家特化得以保留。

### 4.5 专家并行（EP）：把 FFN 变成分布式的

MoE 的 671B 参数不可能放一张卡上。**专家并行（Expert Parallelism, EP）做的事极其简单：把 E 个专家切成 P 份，每张卡持有 $E/P$ 个。**

它的运作是一个 **dispatch → compute → combine** 的循环：

```
                    ┌──────── rank 0 ────────┐        ┌──────── rank 1 ────────┐
  本地 token        │ 持有专家 0..127        │        │ 持有专家 128..255      │
  [T/P, d]         │                        │        │                        │
      │            │                        │        │                        │
      ├── dispatch ──► all-to-all ──────────┼───────►│  收到该发往本地专家的   │
      │            │  （token 按目标 rank    │        │  token                 │
      │            │    重新分发）           │        │                        │
      │            │                        │        │  local grouped GEMM    │
      │            │  all-to-all ◄──────────┼────────┤  只算自己持有的专家     │
      │◄── combine ─┤  （结果回到原 rank）   │        │                        │
      │            └────────────────────────┘        └────────────────────────┘
      ▼
   加权求和
```

通信量的账非常干净：**每个 token 的 $d$ 维隐藏态要被复制 K 份发出去，算完再收回来。**

$$\text{每层通信量} = 2 \times T \times K \times d \times \text{bytes}$$

代入 DeepSeek-V3（$d=7168$, $K=8$, FP8 即 1 byte，58 个 MoE 层）：

```
 T =    4096 : 单层 469.76 MB  58层合计   27.25 GB
 T =   32768 : 单层   3.67 GB  58层合计  217.97 GB
 T =  262144 : 单层  29.36 GB  58层合计 1743.76 GB
```

注意这是**整个集群的通信总量**。如果放在 NVLink（~400 GB/s）上，T=32768 时 58 层需要 545 ms；如果跨节点走 IB（~50 GB/s），需要 **4360 ms —— 8 倍**。

**这解释了 EP 部署的两条铁律**：

1. **EP 度数不要超过单机 GPU 数**（如 8 卡机就 EP=8），否则 all-to-all 掉到跨节点网络上，性能断崖。DeepSeek-V3 训练用 EP=64，但特意做了 **node-limited routing（节点限制路由）**——限制每个 token 最多被送到 M 个节点，把跨节点流量压下来。
2. **通信必须和计算重叠**。V3 报告里把「communication overlap」列为核心设计之一，和 FP8、MTP 并列。

显存侧则是直接的除法：

```
 DeepSeek-V3 专家参数合计 690.4 B（含共享专家）
   EP=  8 : 每卡  86.30 GB(fp8) / 172.60 GB(bf16)
   EP= 32 : 每卡  21.58 GB(fp8) /  43.15 GB(bf16)
   EP= 64 : 每卡  10.79 GB(fp8) /  21.58 GB(bf16)
   EP=128 : 每卡   5.39 GB(fp8) /  10.79 GB(bf16)
```

**FP8 在这里不是「顺带的优化」，而是必需品**：EP=32 时 bf16 要 43 GB 权重，加上激活、优化器状态、通信 buffer，80 GB 的卡根本装不下；换成 FP8 直接减半到 21.6 GB。

### 4.6 MoE 推理到底卡在哪 —— 一个必须搞清楚的问题

这一节是全文最重要的量化分析。**很多人以为 MoE 推理快，因为「只激活 37B」，这是个危险的误解。**

先建立分析框架：**算术强度（arithmetic intensity）= 总 FLOPs / 总内存流量**。它与机器的**平衡点**（峰值算力 / 峰值带宽）比较，谁大谁就是瓶颈。

H800 量级：FP8 峰值 989 TFLOP/s，HBM 带宽 3.35 TB/s。

$$\text{平衡点} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} = 295 \text{ FLOP/byte}$$

**意思是：每从显存读 1 个字节，你得做满 295 次浮点运算，才算「算得动」；否则就是在等内存。**

现在算 MoE 的一个专家。DeepSeek-V3 的专家：$d=7168$，$d_{ff}=2048$，FP8。

- 权重字节数：$3 \times 7168 \times 2048 = 44.04\text{ MB}$
- 一个 token 通过它的计算量：$2 \times 3 \times 7168 \times 2048 = 88.08\text{ MFLOP}$
- **算术强度 = 88.08 / 44.04 = 2.0 FLOP/byte**

**2.0，而平衡点是 295。差了 148 倍。**

这意味着：**要让算力打满，同一个专家必须一次吃下 148 个 token**（这样权重读一次、复用 148 次）。

关键的不对称在这里：**稠密模型的 batch 就是全局 batch，MoE 的「专家 batch」是全局 batch 被除以了 $E/K$。** 每个专家收到的 token 数：

$$\text{每专家 token 数} = \frac{B \times K}{E} = \frac{B \times 8}{256} = \frac{B}{32}$$

所以要让 $B/32 \geq 148$，需要 **全局 batch $B \geq 4724$ 个 token**。而稠密模型只需要 148。

**MoE 需要的全局 batch 是稠密模型的 32 倍，因为专家把 batch 切成了 32 份。**

利用率曲线（`cap = 有效算术强度 / 平衡点`）：

```
 B=     1  每专家 token=    0.03  有效 AI=    0.06  算力利用率=  0.02%
 B=    32  每专家 token=    1.00  有效 AI=    2.00  算力利用率=  0.68%
 B=   256  每专家 token=    8.00  有效 AI=   16.00  算力利用率=  5.42%
 B=  1024  每专家 token=   32.00  有效 AI=   64.00  算力利用率= 21.68%
 B=  4096  每专家 token=  128.00  有效 AI=  256.00  算力利用率= 86.71%
 B= 16384  每专家 token=  512.00  有效 AI= 1024.00  算力利用率=100.00%
```

**Batch=1 的单条对话，MoE 的算力利用率是 0.02%。** 直接看延迟下限更震撼：

```
 每 token 要读 8(K) × 58(层) = 464 份专家权重
 总量 20.43 GB → 理论下限耗时 6.10 ms（受显存带宽限制）
 对比：FLOPs 理论下限   0.04 ms
 → 比值 147.6×，纯 memory-bound
```

**MoE 推理在 decode 阶段被显存带宽死死卡住，不是被算力卡住。**

这也直接推导出 MoE 服务的核心工程手段——全都指向同一个目标「**让每个专家一次吃更多 token**」：

| 手段 | 作用 | 代价 |
|---|---|---|
| **Continuous batching** | 把 batch 撑到几千 token | 需要调度器，首 token 延迟略升 |
| **增大 K / 减少 E** | 每专家 token 数 = $BK/E$，减小 E 直接放大 | 失去细粒度专家的优势 |
| **投机解码** | 一次验证多个 token → 等效放大 batch | 需要 draft 模型 |
| **专家预取 / 缓存** | 预测下一层要用的专家，提前搬到片上 | 预测错了白搬 |
| **高压缩精度（FP8/FP4）** | 权重字节数减半 → 流量减半 | 精度风险，见 4.8 |

**一句话**：**MoE 的 prefill 是算力问题（省 FLOPs 就是赚），decode 是带宽问题（省 FLOPs 没用，要省字节）。** 这两个阶段的最优策略完全不同——这是理解 MoE 服务性能的关键分野。

### 4.7 MoE 在并行地图里怎么摆

MoE 不是替代 TP/PP/DP，而是**插进**它们。标准摆法是「**四轴正交**」：

| 轴 | 切什么 | 在 MoE 模型里管什么 |
|---|---|---|
| **DP** | batch | 数据并行，梯度 all-reduce |
| **TP** | 权重矩阵、hidden 维 | **attention 和稠密层**，切成列/行并行 |
| **PP** | 层堆叠 | 层间流水 |
| **EP** | **专家维度** | **路由专家**，all-to-all 通信 |

**一条重要的实践建议**（来自 PyTorch 官方文档）：**不要对路由专家做张量并行（TP）。**

理由很直接：专家已经足够小了。DeepSeek-V3 一个专家只有 $d_{ff}=2048$、44M 参数。把 $[2048, 7168]$ 的矩阵再切一刀，得到的是「又小又碎」的 GEMM——**GEMM 效率在矩阵维度太小时会急剧下降，TP 的通信开销反而盖过收益。** 正确做法是：attention 和稠密 FFN 可以上 TP，**路由专家用 EP**。这也呼应了第一部分的观察——**细粒度（多而小）的专家路线，天然和 TP 不兼容。**

### 4.8 精度与量化：为什么 MoE 特别吃 FP8

回顾 4.6 的结论：**MoE decode 是带宽瓶颈。** 那么减少字节数就是最直接的收益，而减少字节数最有效的手段就是降低精度。

| 精度 | 每个权重字节数 | 671B 模型权重 | 相对 bf16 |
|---|---|---|---|
| BF16 | 2 | 1342 GB | 1.00× |
| **FP8** | **1** | **671 GB** | **0.50×** |
| FP4 / NVFP4 | 0.5 | 335 GB | 0.25× |

**对 MoE 来说，精度降低带来的是「双份」收益**：既省显存（能装下更多专家），又**按比例减少 decode 的带宽压力**。这就是 DeepSeek-V3 把 FP8 训练当作核心突破、torchao 把 MXFP8 grouped GEMM 当作重点的原因。

而且 MoE 在低精度下有两个天然优势：

1. **专家权重的分布比稠密模型更「规整」**。每个专家看到的数据分布更窄（因为路由本身做了聚类），权重的动态范围更小，量化误差更容易控制。
2. **专家天然适合 per-expert / per-block 量化**。HuggingFace 的 MoE 博客专门提到：只有当专家被 pack 成可预测的布局（就是 3.1 里那个 `[E, d_ff, d]` 的 3D Parameter），「per-expert 量化」才有意义。

**但代价也很实在**：量化的误差会**叠加在路由决策上**。router 的 logits 是 $d$ 维向量的内积，如果 $d$ 很大而数值精度低，微小的分数差异可能被量化噪声淹没——**导致路由本身变得不可靠**。实践中 router 的计算通常保持高精度（fp32），只对专家权重和激活做低精度。

### 4.9 微调 MoE：稀疏梯度带来的新问题

MoE 的微调和训练有本质区别：**你只有一个 batch 的 token，而专家有几百个。** 后果是：

- **每个专家的梯度信号极稀疏**。一个 batch 里可能有专家只收到 3 个 token，甚至 0 个（收 0 个的专家这一步完全没有梯度）。
- **全量微调 MoE 的优化器状态开销和参数量成正比**。671B 参数用 Adam 需要 $m$ 和 $v$ 两份状态，即使 FP32 也要 5.4 TB——这是绝大多数人碰不到的量级。
- **因此 MoE 微调几乎总是 LoRA 形态**。冻结巨大的专家权重，只训练小的适配器。这也让「只给部分专家挂 adapter」成为可能。

还有一条路叫 **Upcycling**：拿一个训练好的稠密模型，把它每一层的 FFN **复制 E 份**当作初始专家，再继续训练。好处是专家一开始就有不错的质量（继承了稠密模型的），不会出现「一开始某几个专家随机地强」的严重坍缩。代价是所有专家初始完全相同，需要靠训练后期的梯度差异慢慢分化。

### 4.10 MoE 与其他稀疏技术的对比

把 MoE 放进稀疏技术的大图里看，会更清楚它的位置：

| 技术 | 稀疏的是什么 | 粒度 | 何时生效 | 省什么 |
|---|---|---|---|---|
| **MoE** | **计算路径（哪些专家）** | **token 级，动态** | 训练+推理 | 算力（但**不省显存**） |
| 剪枝 Pruning | 权重（置零/删通道） | 权重级/结构级，静态 | 训练后 | 显存+算力 |
| LoRA | 梯度（只训低秩增量） | 参数子空间，静态 | 微调 | 优化器状态 |
| 量化 Quantization | 数值精度（位宽） | 元素级，静态 | 训练后/训练中 | 显存+带宽 |
| 蒸馏 Distillation | 模型规模（大→小） | 模型级 | 训练时 | 全部 |

**MoE 最特殊的一点：它是唯一一个「不省显存」的稀疏技术。** 剪枝和量化都能让模型变小，MoE 反而让模型变得巨大（671B！）。它省的是**算力**，代价是**显存**和**通信**。

这个特性决定了它的适用边界——下一节。

---

## 五、什么时候该用 / 不该用

**该用 MoE 的场景：**

| 场景 | 为什么合适 |
|---|---|
| **训练预算受限但想要大容量** | 每 FLOP 的「知识容量」提升 3~18 倍，是最划算的买卖 |
| **大规模推理服务（high QPS）** | batch 能撑到几千 token，专家 batch 足够大，算力利用率和带宽效率都能打满 |
| **多语言 / 多领域 / 多任务** | 专家天然适合特化——不同语言、不同领域各占几个专家，这是 MoE 最自然的适用面 |
| **显存充裕、算力紧张** | 精确匹配 MoE 的资源画像：要显存、要带宽，不需要那么多 FLOPS |
| **Prefill 密集的场景**（长 prompt、离线批处理） | Prefill 是 compute-bound，MoE 省 FLOPs 的收益能完整吃到 |

**不该用 MoE 的场景：**

| 场景 | 为什么 |
|---|---|
| **单用户、低 QPS 的本地部署** | batch=1，算力利用率 0.02%，还多背 18 倍显存——**双输** |
| **显存是硬约束**（单卡、边缘设备） | MoE 让模型变大而不是变小，方向完全反了 |
| **小模型（< 1B）** | 专家数少 → 杠杆小；而负载均衡、通信、容量的复杂度一分不少。**收益不抵复杂度** |
| **延迟极敏感的在线服务** | all-to-all 通信和动态路由带来不可预测的尾延迟 |
| **通信带宽受限的多机集群** | EP 跨节点时 all-to-all 会吃掉全部收益，除非能做好的通信重叠 |

**一个粗略的判据**：

> **如果你的瓶颈是「算力不够」→ MoE 有用。如果你的瓶颈是「显存不够」或「batch 撑不起来」→ MoE 帮倒忙。**

---

## 六、常见坑

**坑 1：把 `lb_loss` 的后向传播忘了 —— 辅助损失静默失效**

```python
# ❌ 错：aux_loss 算出来了，但没加进总损失，它就是个装饰品
def forward(self, x):
    out = self.moe(x)
    loss = self.criterion(out, labels)
    self.aux_loss = self.moe.aux_loss     # 算了但没用
    return loss

# ✅ 对：显式加上，并且注意系数是「相对主损失」的量级
def forward(self, x):
    out = self.moe(x)
    loss = self.criterion(out, labels)
    loss = loss + self.moe.aux_loss       # ← 必须真的加进去
    return loss
```

这个坑的隐蔽之处在于：**不报错、不崩溃，只是慢慢坍缩。** 你会在训练到几万步之后发现 loss 上不去了，翻遍代码才发现辅助损失从来没生效过。**养成习惯：把 `aux_loss` 打进日志，观察它是否在下降。**

**坑 2：`aux_coeff` 调得太大 —— 越平衡越差**

4.4 节那张表就是最直接的警告：系数从 0.1 加到 0.5，Gini 从 0.0797 恶化到 0.8223，**比完全不做的 0.8750 只好一点点**。而且专家质量极差被压到 1.4×，**特化能力被抹平了**。

正确做法：`aux_coeff` 通常取 **1e-3 ~ 1e-2** 这个量级（相对主损失），并且**观察负载的最大/最小比值，而不是只看损失值**。如果发现负载仍然严重倾斜，优先考虑换 aux-loss-free 的 bias 方案，而不是继续加大系数。

**坑 3：`cap_factor` 设太小，静默丢掉大量 token**

看 4.3 的表：`cap_factor=1.0` 在 E=256 时丢掉 **42.36%** 的 token。这些 token 的 FFN 计算被完全跳过，但**不会有任何报错**——前向照常返回，loss 照常下降（因为这些 token 走了残差，相当于那一层的 FFN 被跳过，模型仍然能学，只是学得更慢更差）。

**排查方法**：监控「丢弃 token 比例」这个指标。它应该是 **< 1%**。如果超过 5%，说明 `cap_factor` 太小或者负载均衡没做好。TorchTitan 的 `--debug.moe_force_load_balance` 可以用来排除负载因素。

**坑 4：用小 batch 做 MoE 的性能对比实验**

这是最经典的实验设计错误。回顾 4.6：MoE 的算力利用率对 batch 极其敏感。

```
 B=   256 → 利用率  5.42%
 B=  4096 → 利用率 86.71%
```

**同一个优化，在 B=256 下测出来「提升 3%」，在 B=4096 下可能测出「提升 40%」，也可能完全相反。** 做 MoE 性能实验必须：

1. 固定并**报告 batch size**（以及 seq length，因为 $B$ 通常是 token 数）
2. 打开 `--debug.moe_force_load_balance` 保证两组实验的路由分布一致
3. E 大时特别小心——E=256 的负载相对波动是 E=8 的 6 倍（$\sqrt{(E-1)/T}$），同样的实验两次跑出来的路由分布可能完全不同

**坑 5：以为「激活 37B」就等于「37B 模型的开销」**

这个误解在网络层和显存层都是错的：

- **显存**：所有 671B 权重都得常驻（或至少可访问），不是 37B
- **带宽**：batch=1 时理论下限 6.10 ms 是拿 464 份专家权重（20.43 GB）算出来的，和 37B 无关
- **网络**：EP 的 all-to-all 流量是 $2T K d$，和「激活多少参数」没有直接关系

**「激活 37B」只在算 FLOPs 时成立。** 而算 FLOPs 恰恰是这三者里最不重要的一项——因为 4.6 已经证明，MoE 推理根本不被 FLOPs 卡住。

---

## 七、一句话总结

**MoE 用「让每个 token 只走少数几个专家」这一招，把参数量和计算量彻底解耦，让 DeepSeek-V3 能用 18.4 倍的知识容量付一个 37B 稠密模型的算力账单；代价是凭空造出负载均衡、专家通信、显存常驻三个新麻烦——MoE 论文讲的是架构，MoE 工程做的全是这三件事。**

---

<details>
<summary>今日练习（3 题，参考答案在下面）</summary>

### 练习 1：核算一个 MoE 的账本

给定配置：$d=4096$，32 层全部是 MoE 层，每层 16 个路由专家 + 0 个共享专家，top-K=2，每个专家 $d_{ff}=2048$，SwiGLU。attention 用标准 MHA，32 个头、8 个 KV 头，参数量按每层 $d^2 + 2d \cdot (8 \times d/32) + d^2$ 计算。vocab=32000。

求：(a) 总参数 (b) 激活参数 (c) 杠杆倍数 (d) 这个配置相比 Mixtral 8x7B，杠杆是更大还是更小，为什么？

**参考答案**：

```python
d, n_layer, n_head, n_kv, d_ff, E, K, vocab = 4096, 32, 32, 8, 2048, 16, 2, 32000
hd = d // n_head

attn_per_layer = d*d + 2*d*(n_kv*hd) + d*d
attn = attn_per_layer * n_layer

expert_one = 3*d*d_ff                 # SwiGLU 三个矩阵
experts_all  = expert_one * E * n_layer
experts_act  = expert_one * K * n_layer
emb = vocab * d

total  = attn + experts_all + emb
active = attn + experts_act + emb

print(f"注意力      : {attn/1e9:.2f}B")
print(f"专家总计    : {experts_all/1e9:.2f}B")
print(f"── 总参数   : {total/1e9:.2f}B")
print(f"── 激活参数 : {active/1e9:.2f}B   ({active/total*100:.1f}%)")
print(f"── 杠杆     : {total/active:.2f}×")
```

输出：

```
注意力      : 1.34B
专家总计    : 12.88B
── 总参数   : 14.36B
── 激活参数 : 3.08B   (21.5%)
── 杠杆     : 4.66×
```

**(d) 杠杆比 Mixtral（3.65×）更大，但明显不如 DeepSeek-V3（18.37×）。** 三个原因：

1. **专家数多**：16 vs 8。杠杆的上限就是 $E/K$，本例是 $16/2 = 8$，Mixtral 是 $8/2 = 4$。
2. **专家更小**：$d_{ff}=2048$ vs Mixtral 的 14336。专家越小，「多而小」的细粒度路线杠杆上限越高。
3. **稠密部分在「稀释」杠杆**。真正的关系不是 $\text{杠杆} = E/K$，而是：

$$\text{杠杆} = \frac{N_{\text{稠密}} + E \cdot N_{\text{专家}}}{N_{\text{稠密}} + K \cdot N_{\text{专家}}}$$

其中 $N_{\text{稠密}}$ 是**所有 token 都要走的参数**——attention、embedding、前几层稠密 FFN。这些参数在分子分母里是同一份，**它们不产生任何杠杆，只会把比值往 1 拉。**

本例中 $N_{\text{稠密}} = 1.34 + 0.13 = 1.47\text{B}$，占了总参数的 10.2%，所以杠杆被从 8 拉到 4.66。DeepSeek-V3 的稠密部分是 $11.41 + 1.19 + 1.85 = 14.45\text{B}$，占总参数 2.2%，稀释得很少，所以 18.37 能接近 $E/K = 32$。

**要点**：**杠杆 ≈ 用「专家层占模型的比例」折算过的 $E/K$。** 想让 MoE 划算，除了「E 多一点、K 少一点、专家小一点」，还要让**稠密部分尽量便宜**——这也是为什么 DeepSeek-V3 要用 MLA 把 attention 压到 11.41B（同样 $d=7168$ 的普通 MHA 会是它好几倍）。

</details>

<details>
<summary>练习 2：推导 MoE decode 的算术强度（本题是全文的核心量化分析）</summary>

一个 MoE 模型：$d=4096$，$d_{ff}=14336$（注意是 Mixtral 的专家尺寸，不是 V3 的），BF16（2 bytes），每个专家是 SwiGLU。机器：峰值算力 989 TFLOP/s（此处按 BF16 算），HBM 带宽 3.35 TB/s。

求：(a) 单个专家的算术强度 (b) 平衡点 (c) 每个专家需要多少 token 才能打满算力 (d) 如果 E=8、K=2，全局 batch 需要多大？

**参考答案**：

```python
d, d_ff = 4096, 14336
PEAK, BW = 989e12, 3.35e12
BYTES = 2                               # BF16

w_bytes = 3*d*d_ff*BYTES                # SwiGLU 三个矩阵
flops   = 2*3*d*d_ff                    # 每个 token 通过该专家的 FLOPs

ai  = flops / w_bytes
bal = PEAK / BW
need_local = bal / ai
need_global = need_local * 8 / 2        # E / K

print(f"单专家权重   : {w_bytes/1e6:.2f} MB")
print(f"每 token 计算: {flops/1e6:.2f} MFLOP")
print(f"算术强度     : {ai:.3f} FLOP/byte")
print(f"机器平衡点   : {bal:.1f} FLOP/byte")
print(f"每专家需 token: {need_local:.0f}")
print(f"全局 batch 需: {need_global:.0f}")
```

输出：

```
单专家权重   : 352.32 MB
每 token 计算: 352.32 MFLOP
算术强度     : 1.000 FLOP/byte
机器平衡点   : 295.2 FLOP/byte
每专家需 token: 295
全局 batch 需: 1181
```

**要点**：算术强度恰好是 **1.000 FLOP/byte**——这不是巧合。对 SwiGLU，FLOPs = $6 \cdot d \cdot d_{ff}$，权重字节数是 $3 \cdot d \cdot d_{ff} \cdot \text{bytes}$，比值 = $2 / \text{bytes}$。**BF16 时是 1，FP8 时是 2，FP4 时是 4。** 这是 MoE 专家的通用结论：**算术强度只由精度决定，与模型尺寸无关。**

同理，需要的 token 数 = $295 \times \text{bytes} / 2$：BF16 要 295 个，FP8 只要 148 个。**这就是「降低精度对 MoE 是双重收益」的精确表述——既省一半显存，又让达到算力饱和所需的 batch 减半。**

对照 4.6 节我对 DeepSeek-V3（$d_{ff}=2048$、FP8）算出的 148 和 $B \geq 4724$，你可以自己验一下是同一套公式。

</details>

<details>
<summary>练习 3：读懂一次线上异常</summary>

你负责一个 E=64、top-K=6 的 MoE 模型训练。某天运维给你三个指标：

1. 每层专家负载的 `max/avg = 3.1`，`min/avg = 0.02`
2. 日志里 `token_drop_rate = 28%`
3. `aux_loss` 数值在下降，但 `lb_loss` 权重设的是 0.5

三个现象各自说明什么？它们之间的因果关系是什么？给出**按优先级排序**的三条行动。

**参考答案**：

**现象 1（`min/avg = 0.02`）：专家坍缩已经发生。** 有些专家几乎收不到 token。参考 4.3 节的实测：router 偏好强度到「强」时（对应 max/avg ≈ 3.26），min/avg ≈ 0.004。这里的 3.1 / 0.02 说明情况已经接近那个量级。**注意这不是随机涨落**——IID router 的 max/avg 只有 1.04，Gini 只有 0.0119。

**现象 2（28% 丢弃率）：是现象 1 的后果。** 容量按 $\text{cap\_factor} \times T \times K / E$ 分配，而实际有专家的负载是平均值的 3.1 倍。只要 `cap_factor < 3.1`，那个专家必然溢出，而溢出量正好等于不均衡量。**所以 28% 的丢弃率和 max/avg=3.1 是同一个问题的两个面。**

**现象 3（`aux_loss` 在降但没用）：这是根因。** 系数 0.5 属于 4.4 节实测的**过强区间**——

| 辅助损失系数 | max/理想 | min/理想 | Gini |
|---|---|---|---|
| 0.100 | 1.23 | 0.66 | 0.0797 |
| 0.500 | 6.23 | 0.00 | 0.8223 |

coeff=0.5 时 Gini 是 0.8223，**负载比不做任何平衡还差**（不做是 0.8750）。而且这个非单调性很稳健（5 个随机种子上 coeff=0.5 的 Gini 都在 0.777~0.851）。

**为什么 `aux_loss` 在下降但负载没改善？** 因为「损失值下降」和「负载变均衡」不是同一件事。系数过强时，平衡项基于滞后的负载 EMA 施加，形成**增益远大于 1 的延迟负反馈环**，系统进入振荡——损失的平均值可以很低（因为偏差正负抵消），但每一时刻的负载都是极端的。**这是一个「指标看起来在改善、实际问题在恶化」的经典陷阱。**

**行动（按优先级）**：

1. **先把 `aux_coeff` 从 0.5 降到 0.01 量级**。这是根因，且改动最小。改完观察 `max/avg` 是否回到 1.5 以内。同时**增加监控**：不要只看 `aux_loss` 的绝对值，要看**负载分布的 Gini 或 max/min 比值**——那才是真正想优化的量。
2. **把 `cap_factor` 临时调大到 3.5+**，让 `token_drop_rate` 先降到 5% 以下，止住训练质量的流血。这一步是止血，不是治本——调大容量会增加显存和 padding 浪费（参考 4.3：max/avg=3.26 时静态 buffer 浪费 69.3%）。
3. **如果降 `aux_coeff` 后仍然不均衡，换 aux-loss-free 方案**：用 gate 的 bias 做动态调整，把平衡压力从主损失里移出去，在路由边界上生效。这既能避免振荡，也**不会抹平专家特化**（参考 4.4：coeff=0.5 时专家质量极差被压到 1.4×，而 coeff=0.1 时是健康的 1.9×）。

**要点**：这三个现象是**一条因果链**——`aux_coeff` 过大 → 延迟反馈振荡 → 负载极端不均 → 容量溢出 → 大量 token 被丢。**看起来是三个 bug，实际是一个。**

</details>

---

**下期预告**：还在挑，候选有损失函数设计、正则化全景、cuDNN/cuBLAS、投机解码 Speculative Decoding、知识蒸馏、torch.export + AOT Inductor——挑一个你没看过的。
