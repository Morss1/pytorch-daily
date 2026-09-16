# PyTorch 每日一课 · 第 020 期

## 学习率调度器：训练节奏的设计哲学

> **日期**：2026-09-16
> **难度**：⭐⭐⭐
> **前置知识**：SGD/Adam 基本原理、训练循环的写法（第 013 期有铺垫，但本篇独立可读）
> **预计阅读时间**：12 分钟

---

## 一、这个领域解决什么问题

假设你在训练一个模型。第一天你把学习率设成 0.1，loss 疯狂震荡不下降；你改成 0.001，loss 下降得很稳但慢得让人绝望。你隐约感觉到：**学习率不是一次性决定好的超参数，而是应该随训练进程变化的量。**

这就是学习率调度器（Learning Rate Scheduler）存在的理由。它回答的问题是：**在整个训练生命周期里，"步子该迈多大" 应该如何演化？**

直觉上有三个阶段：

1. **热身期（warmup）**：训练刚开始，参数是随机的，梯度方向噪声极大。这时候步子太大容易直接"崩"——loss 爆炸或陷入糟糕的区域。所以要先小心翼翼地小步走。
2. **探索期**：参数已经进入合理的区域，梯度可信了，此时保持较大的学习率，快速逼近最优点。
3. **精修期**：接近收敛时，大步子会让你在最优点附近"来回跳"却落不进去（想象在碗底蹦迪）。需要逐渐减小学习率，让参数精细地稳定下来。

一个著名的经验结论来自 Leslie Smith 的论文《Cyclical Learning Rates》和《Super-Convergence》：**训练的关键瓶颈往往不是优化器，而是学习率的节奏。** 同样的模型、同样的数据、同样的优化器，只是调度策略不同，最终精度可以差好几个点。

而 PyTorch 把这件事做成了一个独立模块：`torch.optim.lr_scheduler`——它和优化器解耦，负责"何时、如何"改学习率，让优化器只负责"怎么走"。

---

## 二、核心思想：几种经典的"节奏曲线"

### 1. 阶梯衰减（Step Decay）：最古典的方案

每过 N 个 epoch，学习率乘以 0.1。像下楼梯：

```
0.1 ──────┐
          └─ 0.01 ──────┐
                        └─ 0.001 ────
```

对应深度学习早期的经验："训练到平台期就除以 10"。AlexNet、ResNet 原论文都是这么训的。

### 2. 余弦退火（Cosine Annealing）：现代默认款

学习率按余弦曲线从最大值平滑降到接近零：

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

它的好处是**前期降得慢、中期降得快、末期又变慢**——正好匹配训练的三个阶段。而且曲线光滑，没有阶梯衰减那种"突变时刻选哪个"的纠结。ViT、LLaMA 等现代模型几乎清一色用它。

### 3. Warmup：给训练一个"起步缓冲"

在训练最初的若干步里，学习率从 0（或很小的值）**线性升**到目标值。为什么需要它？

- Adam/AdamW 的二阶动量估计在初期样本极少，方差巨大，第一步就可能迈出离谱的大步；
- 大模型（尤其是 Transformer）对初始大学习率极其敏感，直接上 1e-3 可能直接 loss 爆 NaN。

所以现代配方几乎是标配：**warmup + cosine decay**。比如 GPT-3 用了 375 步线性 warmup，然后余弦衰减到 10%。

### 4. One Cycle：一种"先冲后收"的激进策略

Leslie Smith 提出的 super-convergence 方案：学习率先从低升到最高点，再一路降到接近零；同时动量反向变化。整个过程像一个"冲浪"动作，可以在极少 epoch 内达到甚至超过常规训练的精度。`torch.optim.lr_scheduler.OneCycleLR` 就是官方实现。

---

## 三、在 PyTorch 中怎么用

### 基础用法：调度器要放在 `optimizer.step()` 之后

```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 阶梯衰减：每 10 个 epoch 学习率乘 0.5
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(30):
    train_one_epoch(model, optimizer)   # 内部会调用 optimizer.step()
    scheduler.step()                    # 注意：每个 epoch 结束后调用一次
    print(f"epoch {epoch}, lr = {optimizer.param_groups[0]['lr']:.4f}")
```

两个关键细节：

1. **`scheduler.step()` 必须在 `optimizer.step()` 之后调用**。顺序反了会跳过第一个学习率值，且新版 PyTorch 会直接报警告。
2. 学习率的真实值要通过 `optimizer.param_groups[0]['lr']` 查看——调度器修改的就是优化器内部的 `param_groups`。

### 现代配方：Warmup + Cosine（手写版，面试常考）

```python
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """线性 warmup + 余弦衰减，Transformer 训练的标配配方"""
    def lr_lambda(current_step):
        # 阶段一：warmup，学习率线性从 0 升到 1（乘上基准 lr）
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # 阶段二：余弦衰减到 0
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=100, total_steps=10000)

# 注意：Transformer 通常按 step（而不是 epoch）调度
for step, batch in enumerate(dataloader):
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    scheduler.step()          # 每个 batch 之后都调用
    optimizer.zero_grad()
```

### ReduceLROnPlateau：看 loss 脸色的"自适应"调度

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

for epoch in range(50):
    train_one_epoch(model, optimizer)
    val_loss = validate(model)
    # 注意：这个调度器传的是指标值，不是无参调用！
    scheduler.step(val_loss)
```

当验证 loss 连续 3 个 epoch 没有改善，学习率自动乘 0.5。适合没有明确训练时长的场景（比如微调）。

### 查看整条调度曲线（调试利器）

```python
# 把整个训练过程的学习率曲线画出来，确认调度符合预期
lrs = []
for _ in range(1000):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

import matplotlib.pyplot as plt
plt.plot(lrs)  # 应该看到先升后降的 warmup + cosine 曲线
```

---

## 四、围绕这个领域展开：它在整个生态中的位置

### 1. 和优化器的关系：分工与耦合

调度器不是优化器的一部分——它是**站在优化器之上的一个策略层**。但它和优化器有微妙的耦合：

- **Adam 系对学习率不敏感吗？** 不。虽然 Adam 有自适应缩放（梯度除以 √v），但学习率仍然是"整体步长"的旋钮，调度依然重要。AdamW 的默认 1e-3 只是小模型的经验值，LLM 训练常用峰值 3e-4 甚至更低。
- 有些优化器自带"调度内嵌"，如 **LAMB/Novograd** 论文里调度和优化器是一体设计的。

### 2. 调度器分类学：按"依据什么决策"

| 依据 | 代表 | 特点 |
|---|---|---|
| 按时间（epoch/step） | StepLR、CosineAnnealingLR、LinearLR | 确定性强，需要预估总训练步数 |
| 按指标（loss 停滞） | ReduceLROnPlateau | 自适应，但引入了额外超参（patience、threshold） |
| 按梯度状态 | RAdam、Lookahead 的内建机制 | 温和地解决 warmup 问题，属于优化器层 |

### 3. 大模型训练中的调度实践

- **LLaMA/GPT 系列**：warmup 2000 步左右 + 余弦衰减到峰值的 10%（不是 0！末期保留一点学习率让 loss 继续下降）。
- **Chinchilla/Muon 等新工作**开始质疑 cosine：因为 cosine 的"好"依赖于**你知道总训练步数**——想中途续训/延长训练时，曲线形状就不对了。WSD（Warmup-Stable-Decay）调度因此流行：长期保持恒定学习率，最后阶段快速衰减，天然支持"随时停止"。
- **微调场景**（LoRA 等）：常用很小的恒定学习率 + 短 warmup，或者 linear decay。

### 4. 周期性调度：不要小看"降了再升"

`CosineAnnealingWarmRestarts` 会让学习率周期性地升回去。听起来反直觉，但背后是"重启探索"的思想：陷入局部盆地后，升回学习率可能跳出该盆地，找到更深的那个。SGDR（Stochastic Gradient Descent with Warm Restarts）和快照集成（Snapshot Ensembles，在每个周期底部存一个模型做集成）都建立在这上面。

### 5. 相关但不同的东西

- **梯度裁剪（clip_grad_norm_）**：也影响"有效步长"，但作用是安全阀而非节奏控制，两者正交、常搭配使用。
- **学习率查找（LR Range Test）**：训练前让学习率从 1e-7 指数升到 1，找 loss 下降最陡的位置——用来选峰值学习率。fastai 的 `lr_find`、PyTorch Lightning 的 `lr_finder` 都是这个思路。
- **Param Groups**：优化器支持按参数组设置不同学习率（比如对 backbone 用 1e-4、对新增 head 用 1e-3）。调度器对所有组统一缩放（除非自定义 LambdaLR 分别处理）——迁移学习里这是必修技能。

---

## 五、什么时候该用 / 不该用

**该用：**

- 任何训练超过几个 epoch 的场景——固定学习率几乎总是次优的
- 从零训练大模型/Transformer：warmup 不是可选项，是保命项
- 微调：用低峰值 + 短 warmup 的温和调度
- 不知道训多久：ReduceLROnPlateau 或 WSD 调度

**不该用 / 谨慎：**

- **强化学习**中很多 on-policy 算法对学习率变化敏感，常用恒定学习率
- 训练极短（几十个 step 的 toy 实验），调度没意义
- 已在使用内建 warmup 机制的优化器（如 RAdam）时，可以缩短或去掉显式 warmup
- 使用 cosine 却不确定总步数——先定总步数再选调度，否则"末段精修"永远不会到来

---

## 六、常见坑

**坑 1：`scheduler.step()` 和 `optimizer.step()` 顺序颠倒**

正确顺序是先 `optimizer.step()` 再 `scheduler.step()`。反过来会导致第一个学习率值被跳过，且 PyTorch 2.x 会抛出 UserWarning。记住口诀："先用优化器走一步，再让调度器拨表"。

**坑 2：每个 epoch 调还是每个 step 调搞混**

`StepLR(step_size=10)` 默认按 epoch 计数；而 `OneCycleLR`、warmup+cosine 配方按 step 计数。混用的后果是学习率变化速度差了几百倍。原则：**调度器的"单位"要和你 `scheduler.step()` 的调用频率一致。** 用 `total_steps` 参数时传的是 step 数，不是 epoch 数乘错就行。

**坑 3：保存/恢复 checkpoint 时忘了调度器状态**

`scheduler.state_dict()` 和 `optimizer.state_dict()` 一样需要保存。只存优化器、恢复后调度器从零开始计数——学习率曲线直接错位，续训效果莫名变差。正确做法：

```python
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),   # 别忘了这一行
    'step': global_step,
}, 'ckpt.pt')
```

**坑 4：ReduceLROnPlateau 的调用方式不一样**

它是唯一一个 `scheduler.step(metric)` 需要传参的调度器。无参调用不会报错但逻辑全错——它不知道 loss 有没有改善。另外 `patience` 是"容忍多少个 epoch 无改善"，设成 0 会过于激进，学到一半学习率就被砍没了。

---

## 七、一句话总结

**学习率调度器是训练的"节奏指挥"：warmup 保护起步，大学习率负责探索，衰减负责精修——优化器决定怎么走，调度器决定何时快、何时慢。**

---

<details>
<summary>今日练习</summary>

**练习 1**：写一个调度器，前 50 步从 0 线性 warmup 到 0.1，之后线性衰减到 0，总共 500 步。用画图验证曲线形状。

参考答案：

```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

model = nn.Linear(4, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # lr 是峰值

warmup, total = 50, 500
scheduler = LambdaLR(optimizer, lambda t: min((t + 1) / warmup, (total - t) / (total - warmup)))

lrs = []
for step in range(total):
    lrs.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()

# 验证：lrs 应该是先线性升到 0.1，再线性降到接近 0 的三角形
assert abs(lrs[49] - 0.1) < 1e-6   # warmup 结束时达到峰值（近似）
assert lrs[-1] < 0.001              # 结束时接近 0
```

**练习 2**：思考题——你在微调一个 ResNet，backbone 用 lr=1e-4、新分类头用 lr=1e-3，想让两者按同一余弦曲线衰减。为什么直接套 `CosineAnnealingLR` 可能出问题？

参考答案：`CosineAnnealingLR` 会把每个 param group 的学习率都按同一条余弦曲线衰减，这个场景下其实**没问题**——因为它按比例缩放各组当前的 lr。真正会出问题的是 `StepLR` 之外那些"绝对值覆盖"型调度器（如直接设置 base_lr 的场景），以及 `LambdaLR` 里写了绝对值的 lambda。更值得注意的点是：如果两个组想用**不同形状**的曲线（比如 head 衰减得更快），就需要给 `LambdaLR` 传一个 lambda 列表，每个 group 一个函数。理解 param_groups 与调度器的交互，比记结论更重要。

</details>
