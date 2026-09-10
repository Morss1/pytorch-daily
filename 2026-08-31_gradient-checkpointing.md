# PyTorch 每日一课 · 第 009 期

## 梯度检查点 (Gradient Checkpointing)：用时间换显存

> **日期**：2026-08-31
> **难度**：⭐⭐⭐
> **前置知识**：Autograd 基础（`backward()` 做了什么）、前向计算图的概念、GPU 显存的大致构成
> **预计阅读时间**：12 分钟

---

## 一、这个领域解决什么问题

先看一个每个深度学习从业者都撞过的墙：

```
RuntimeError: CUDA out of memory. Tried to allocate 2.34 GiB.
```

训练一个模型时，GPU 显存主要被四类东西占据：

| 占用项 | 大小量级 | 是否必须常驻 |
|---|---|---|
| 模型参数 | 参数量 × 4 字节（FP32） | 是 |
| 优化器状态 | 参数量 × 8 字节（Adam 的一阶+二阶矩） | 是 |
| 梯度 | 参数量 × 4 字节 | 是 |
| **中间激活值（activations）** | **随 batch、序列长度、网络深度线性增长** | **——这是可优化的部分** |

前三项只和模型大小有关，模型定了就定了。而第四项——**中间激活值**——才是真正的显存杀手，因为：

- 反向传播需要它。Autograd 计算 `dL/dW` 时要用到前向的中间结果（比如矩阵乘法的输入、激活函数的输出）；
- 它随 batch size、序列长度、网络深度**线性甚至平方级增长**。一个 70B 模型训 8K 上下文，激活值能吃掉几十上百 GB；
- 如果不做任何处理，PyTorch 默认**保留所有中间激活**直到 backward 结束。

梯度检查点（Gradient Checkpointing，也叫 Activation Checkpointing / Activation Recomputation）解决的就是：**能不能不把所有中间激活都存着？**

答案是：可以。因为计算是可重放的——**与其存结果，不如存「怎么算出这个结果」的配方**。

---

## 二、核心思想：用时间换显存

### 2.1 一个生活类比

想象你在做一道很长的菜（前向传播），有 20 道工序。正常情况下，你把每道工序的半成品都摆在桌上（存激活），最后写总结报告（反向传播）时要回头参考每个半成品。

桌子不够大怎么办？

**梯度检查点的做法**：只在第 0、5、10、15、20 道工序处留一份半成品（检查点），中间的扔掉。写报告时如果需要第 7 道工序的产物，就从第 5 道的半成品**重新做一遍第 6、7 道**（重计算）。

代价：多做了约一倍的前向计算。收益：桌上只摆 1/4 的东西。

### 2.2 时空权衡的数学

设网络有 $L$ 层，每层前向激活占 $a$ 显存、前向计算耗时 $t$。

- **不重计算**：显存 $O(L \cdot a)$，时间 $O(L \cdot t)$
- **全部重计算**（每层都是检查点）：显存 $O(a)$（只存一层），时间 $O(2L \cdot t)$（前向做两遍）
- **每 $\sqrt{L}$ 层设一个检查点**：显存 $O(\sqrt{L} \cdot a)$，时间 $O(L \cdot t + \sqrt{L} \cdot t) \approx O(L \cdot t)$

第三条是经典结论（Griewank & Walther, 2000）：合理选检查点位置，**显存从 $O(L)$ 降到 $O(\sqrt{L})$，而时间开销接近零**。这就是为什么它几乎是免费午餐——尤其是对深网络。

### 2.3 和 Autograd 的关系

PyTorch 的 Autograd 默认在**每个 op 执行时**就把输入存进计算图（`SavedTensor`）。梯度检查点本质上是一个 Autograd 级的技巧：

1. 前向时，把一段子图包在一个「不存中间结果」的上下文里跑，**只保留这段子图入口处的输入**（这就是检查点）；
2. 反向传播到这个检查点时，Autograd 发现中间激活没了，于是**用保存的入口输入重新执行一遍前向**，现算出中间激活；
3. 用现算的激活正常完成这段子图的梯度计算。

所以它不需要你改模型结构、不需要换优化器——只是改变了「中间结果什么时候被计算出来」这一件事。

---

## 三、在 PyTorch 中怎么用

### 3.1 最简单的方式：`torch.utils.checkpoint.checkpoint`

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    """一个普通的 Transformer 块，内部结构随便"""
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 注意力子层（带残差）
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        # FFN 子层（带残差）
        x = x + self.ffn(self.norm2(x))
        return x

class Model(nn.Module):
    def __init__(self, n_layers=12, d_model=512, use_checkpoint=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(n_layers)]
        )
        self.use_checkpoint = use_checkpoint  # 开关：是否启用梯度检查点

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint:
                # 关键一行：把整个块包进 checkpoint
                # 前向时只存 x（检查点），反向时重算块内部
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

# 对比显存占用
def measure_peak_memory(use_ckpt: bool) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = "cuda"
    model = Model(n_layers=16, use_checkpoint=use_ckpt).to(device)
    x = torch.randn(8, 2048, 512, device=device)  # batch=8, seq=2048

    out = model(x)
    loss = out.square().mean()
    loss.backward()  # 反向时若开了 checkpoint，会触发重计算

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    del model, x, out, loss
    return peak_mb

if __name__ == "__main__":
    normal = measure_peak_memory(False)
    ckpt = measure_peak_memory(True)
    print(f"不开检查点: {normal:.0f} MB")
    print(f"开检查点:   {ckpt:.0f} MB")
    # 典型输出（16 层、seq 2048）：开检查点能省 30%-60% 峰值显存
```

**关键参数 `use_reentrant=False`**：这是新版推荐路径（PyTorch 2.1+）。旧的 reentrant 模式用 `torch.autograd.Function` 的怪异方式实现，有「至少一个输入 requires_grad」等诸多限制；非重入模式由 Autograd 引擎原生支持，能正确处理嵌套、Dropout 随机性、no_grad 等边界情况。**新代码一律用 `use_reentrant=False`**。

### 3.2 自动包一层：`checkpoint_sequential`

```python
from torch.utils.checkpoint import checkpoint_sequential

# 对 nn.Sequential 风格的层堆叠，一行搞定「每 N 层设一个检查点」
# 相当于把 16 层分成 8 段，每段一个检查点
out = checkpoint_sequential(
    nn.Sequential(*[TransformerBlock() for _ in range(16)]),
    segments=8,          # 分成 8 段（即 8 个检查点）
    x,
    use_reentrant=False,
)
```

### 3.3 一行开启（生态级 API）

主流训练框架里它几乎都有一等公民接口，因为这是训练大模型的标准操作：

```python
# HuggingFace Transformers
model.gradient_checkpointing_enable()

# PyTorch Lightning
trainer = Trainer(gradient_clip_val=1.0)
model.configure_gradient_checkpointing()  # 或在 LightningModule 里实现

# Accelerate
model, optimizer, _, _ = accelerator.prepare(model, optimizer, None, None)
model.gradient_checkpointing_enable()

# torchtitan / Megatron 风格（每层选择性重计算）
# 在 layer 的 forward 里手动挑「贵但不占显存」的部分重算
```

---

## 四、围绕该领域展开

这是「领域」的核心部分——梯度检查点不是孤立的函数调用，它是**显存-时间权衡谱系**中的一个点，理解它的邻居才能真正理解它。

### 4.1 显存优化谱系：四个层次

```
显存不够时的选择（代价从低到高）：

1. AMP 混合精度        —— 激活和参数都减半，几乎无代价（BF16）
2. 梯度检查点          —— 激活 O(L)→O(√L)，多 ~30% 计算时间
3. FSDP / ZeRO         —— 参数/梯度/优化器状态切分到多卡
4. CPU Offload / 换页  —— 显存换主存，PCIe 带宽成为新瓶颈
```

它们**正交且可叠加**：真实大模型训练（如 LLaMA 级别）通常四招全用。梯度检查点负责的是「激活」这一项；FSDP/ZeRO 负责「参数+梯度+优化器状态」那三项。两者解决的是不同列的显存账。

### 4.2 前缀树与重计算策略：并非只有「全开」

「每层都重计算」是最粗暴的版本。实际工程里的演进方向是**选择性重计算（Selective Recomputation）**：

- **Megatron-LM 的发现**：Transformer 里 Flash Attention 那部分**计算贵、但用了 Flash Attention 后激活很省**；反过来，LayerNorm/Dropout 的输出**计算极其便宜、但激活很占**。
- 于是最优策略不是整层重算，而是：**便宜的部分（norm/dropout 输出）重算，贵的部分（attention 矩阵）存着**。
- 这就是 Megatron 论文里 "selective activation recomputation" 的由来——用 5% 的额外计算省掉 ~70% 的激活显存。

### 4.3 与分布式训练的化学反应

- **ZeRO-2 / ZeRO-3 + 激活检查点**：FSDP 的 `apply_activation_checkpointing` API 允许对包装后的模型按策略（如每层）施加检查点；
- **张量并行中的重计算**：序列并行（Sequence Parallelism，见后续期数）把 LayerNorm/Dropout 的激活按序列维度切分，和选择性重计算配合，把激活显存再压一个量级；
- **一个有趣的三角关系**：通信（第 007 期）想 overlap 计算（第 008 期），而重计算恰好提供了「可被 overlap 的计算」——有人把激活重计算放在通信等待期做，白嫖了这段空转时间。

### 4.4 推理端为什么不需要它

注意梯度检查点是**训练期专属**技术：推理没有 backward，激活用完即弃，根本不存在「保留中间激活」的问题。推理端的对应物是 **KV Cache**（激活的另一种缓存策略，方向相反——推理是主动选择「存」来换时间，训练重计算是主动选择「不存」来换空间）。同一块显存，训练和推理的最优策略完全相反，这是很漂亮的对称性。

### 4.5 与 PyTorch 机制底座的衔接

- 底层实现挂在 `torch.autograd.graph.saved_tensors_hooks` 上——一个通用的「拦截所有 SavedTensor」钩子。梯度检查点、DTensor 的激活分片、offload 到 CPU，都用它；
- `torch.compile`（第 003 期）和梯度检查点可以共存，但 Inductor 有自己的激活管理（`torch._inductor.config` 里有自动重计算相关配置），未来趋势是编译器自动决定哪些中间量重算——**手动 checkpoint 可能逐渐被编译器接管**；
- CUDA Graph（第 001 期）捕获静态图后，重计算路径也被捕获，两者兼容但要注意 `use_reentrant=False` 才能正确组合。

### 4.6 相关但不同的概念（防混淆）

| 概念 | 干什么的 | 和本期的关系 |
|---|---|---|
| Gradient Checkpointing | 丢弃激活，反向时重算 | 本期主角 |
| Gradient Accumulation | 攒多次小 batch 的梯度再 step | 省**激活显存的另一种方式**（缩小单次 batch），常一起用 |
| Offload (activation) | 把激活搬到 CPU 内存 | 空间换 PCIe 带宽，与重算互为替代 |
| `detach()` | 把张量从计算图切下来 | 彻底不要梯度，语义完全不同 |

---

## 五、什么时候该用 / 不该用

**该用：**

- 训练深网络（Transformers、UNet、ResNet-1000 之类）时 OOM，尤其是**激活占比高**的场景（长序列 NLP、大分辨率 CV）；
- 想在同样显存下**加大 batch 或序列长度**——经常比省下的 30% 计算时间更划算（大 batch 带来的吞吐收益可能超过重算开销）；
- 微调大模型（LoRA/全参）显存刚好处在「差一点」的边缘；
- 配合 FSDP 做大模型训练时（几乎是必选项）。

**不该用 / 慎用：**

- **瓶颈是计算而非显存**时：GPU 利用率本来就 100%，重算直接线性增加训练时间；
- **网络很浅**（几层的 MLP）：激活本来就不大，$O(\sqrt{L})$ 的收益微乎其微；
- **前向里有不可重放的副作用**：比如 forward 里更新 BN running stats、往磁盘写文件、改全局状态——重算一遍会执行两次副作用；
- **使用了自定义 autograd Function 且没实现好 double-backward**（需要二阶梯度时，旧 reentrant 模式直接不支持；`use_reentrant=False` 支持但自定义 Function 要写对）；
- Dropout：注意非重入模式会**保存 RNG 状态并在重算时恢复**，保证两次前向的 dropout mask 一致——这是它比旧模式正确的关键点之一，但也意味着重算不是「随便跑一遍」。

---

## 六、常见坑

**坑 1：Dropout 随机性不一致，梯度是错的**
旧 reentrant 模式下，如果 checkpoint 的段里有 Dropout 且没处理 RNG，重算时 dropout mask 和前向不同，梯度悄悄错误（不报错！）。非重入模式自动保存/恢复 RNG 状态解决了这个问题——**这是必须用 `use_reentrant=False` 的最重要理由**。

**坑 2：`torch.no_grad()` 里包了 checkpoint，反向直接报错**
梯度检查点靠「反向时重算」工作，前提是这段计算在反向时**能够**被重新执行并建图。如果在 no_grad 上下文里调用，或者传入的输入全部 `requires_grad=False`，反向时就没有图可建。报错信息类似 `none of the inputs have requires_grad=True`（reentrant 模式）。

**坑 3：以为显存会降到理论值，实际省得没那么多**
重计算本身也需要工作显存（重算时的临时激活），且检查点入口处的输入仍然要保存。另外如果你只对模型的一部分（比如每 4 层的 1 层）开了检查点，收益按比例缩水。建议：用 `torch.cuda.memory_allocated()` 分段打点，先搞清楚**你的显存到底花在哪**（激活 vs 参数 vs 优化器），别盲目开。

**坑 4：和 BN 的 running statistics 冲突**
BatchNorm 的 forward 会更新 running_mean/running_var（副作用）。开检查点后这段前向跑两遍，统计量被更新两次，验证指标会漂移。对策：换 LayerNorm，或用 `checkpoint(..., determine_rng_state...)` 之类的上下文把 BN 放在检查点段外，或干脆给 BN 加 freeze 逻辑。

---

## 七、一句话总结

> 梯度检查点把「存中间结果」改写成「存配方、按需重做」，用一次前向重放把激活显存从 $O(L)$ 压到 $O(\sqrt{L})$——它是训练期显存账本上「激活」那一列的标准答案，和 FSDP 管的另外三列（参数/梯度/优化器状态）拼起来才是完整的省显存图景。

---

## 八、今日练习

试着不跑代码先想答案，再验证。

<details><summary>今日练习</summary>

**练习 1**：一个 48 层的网络，如果每层设检查点（段=1 层），前向计算量变成多少？如果每 6 层设一个呢？

<details><summary>参考答案</summary>

- 每 1 层一个检查点：反向时每段都要重算 1 层，总计算 ≈ 前向 48 层 + 重算 48 层 = **2 倍**；
- 每 6 层一个检查点（8 段，每段 6 层）：每段反向时重算本段 6 层，重算总量 = 48 层，总计算仍是 **2 倍**？

不对——注意重算的是「段内全部层」，所以只要所有层都被检查点覆盖，重算量都是一整遍前向，**倍数和分段粒度无关**（都是 2×）。分段粒度影响的是**显存**：段越小，需要同时保留的激活越少（但检查点入口输入数量变多，且重放调度开销变大）。这就是为什么实践中常取「每层一个」或「每几个 transformer block 一个」——显存收益递减，调度开销上升。

**练习 2**：为什么 HuggingFace 的 `gradient_checkpointing_enable()` 默认只在训练模式下生效（`model.training` 为 True），eval 时自动跳过？

<details><summary>参考答案</summary>

因为推理/验证时 `torch.no_grad()` 下没有反向传播，激活用完即弃，本来就不用保留——重计算无利可图，纯属浪费。HF 的实现是在每个层的 forward 里判断 `if self.gradient_checkpointing and self.training:`，保证 eval 路径走普通前向。这也印证了第 4.4 节：这是训练期专属技术。

**练习 3（动手）**：把上面 3.1 的代码中 `n_layers` 改成 32、序列长度改成 4096，分别在 CPU（把 `.to(device)` 改掉、删掉 CUDA 统计，用 `torch.profiler` 或简单计时对比）或 GPU 上对比：开/关检查点的**训练一步总耗时**差异是多少？显存差异是多少？

<details><summary>参考答案</summary>

典型结果（A100-40G 量级）：峰值显存下降 40%-60%；单步耗时增加 20%-35%（不到 2 倍，因为重算可以和其他工作重叠、且只重算被检查点覆盖的部分，加上 backward 本身的通信/访存并不翻倍）。如果观察到接近 2 倍耗时，说明该场景瓶颈完全在计算上，可能不值得开满检查点，可考虑只对一半的层开启。
</details>
</details>
</details>

---

*下期预告：显存-时间权衡讲完了，接下来聊聊把「参数」这一列也切到多卡的 FSDP / ZeRO。*
