# PyTorch 每日一课 · 第 013 期

## 优化器家族：SGD、Adam、AdamW 的本质

> **日期**：2026-09-07
> **难度**：⭐⭐⭐
> **前置知识**：反向传播、梯度下降的基本概念、Autograd 基础
> **预计阅读时间**：15 分钟

---

## 一、这个领域解决什么问题

训练神经网络的本质，是在一个高维空间里寻找一组参数，让损失函数尽量小。反向传播只负责一件事：**告诉你每个参数往哪个方向走会让损失变大**（梯度指向损失上升最快的方向，所以往反方向走）。

但「知道方向」和「走得好」之间隔着巨大的鸿沟。想象你蒙着眼睛在一片连绵起伏的山地里找最低点，你手里只有一个指南针（梯度）。你会立刻遇到一堆现实问题：

- **步子迈多大？** 迈大了直接跨过谷底冲上对面山坡（发散），迈小了十万年走不到（收敛太慢）。
- **每个方向用同一个步子吗？** 有的维度陡峭、有的维度平缓，统一步长会导致平缓方向还没走几步，陡峭方向已经来回震荡。
- **路上全是小坑（局部极小值、鞍点）怎么办？** 深度网络的损失曲面几乎处处是鞍点，纯靠梯度容易卡住。
- **数据是有噪声的**：每个 batch 算出来的梯度只是真实梯度的有噪声估计，怎么让噪声互相抵消而不是推着你乱走？

优化器（Optimizer）就是回答这些问题的工程学。它决定了「拿到梯度之后，参数到底怎么更新」。选错优化器或配错超参，模型可能根本训不动——这不是玄学，背后有清晰的数学结构。

## 二、核心思想：从 SGD 到 AdamW 的演化逻辑

理解优化器家族最好的方式，不是背公式，而是把它看成一场**逐步打补丁的演化史**。每一代都是在解决上一代的痛点。

### 第一代：SGD —— 相信指南针

最朴素的更新规则：

```
参数 ← 参数 - 学习率 × 梯度
```

**动量（Momentum）补丁**：SGD 在噪声梯度下走的是一条抖动的折线路。动量的想法来自物理学：把参数更新看成一个有惯性的小球，之前的速度会延续下来。这样随机噪声因为方向不一致会互相抵消，而一致的方向会累积加速。用类比说：普通 SGD 是每次都重新决定走哪边的人，带动量的 SGD 是滚下山坡的球——坑坑洼洼的表面噪声推不动它，但它会沿着大趋势越滚越快。

### 第二代：自适应学习率 —— 让每个参数有自己的步子

SGD 全家（含动量）有个根本问题：**所有参数共用一个学习率**。但想象一个特征几乎从不出现（梯度常年接近零）的参数，和一个特征天天出现的参数——前者应该用大步子快速追上，后者要小心微调。

**AdaGrad** 的思路：给每个参数维护一个「历史梯度平方的累计和」，累积越多的参数，学习率除得越多。问题是一直累加，学习率最终衰減到零，训练提前「冻死」。

**RMSProp** 的补丁：把「累加」改成「指数移动平均（EMA）」，只看最近的梯度历史，旧的慢慢忘掉。

### 第三代：Adam —— 动量 + 自适应的合体

**Adam（2014）** 把两条线合起来：

- 一阶矩估计 `m`：梯度的 EMA —— 相当于 RMSProp 版的动量（**方向**的平滑）
- 二阶矩估计 `v`：梯度平方的 EMA —— 相当于 RMSProp（**步长**的自适应）
- 偏差修正：训练初期 EMA 从零启动，会严重低估真实值，Adam 用 `m̂ = m / (1 - β₁ᵗ)` 修正

```
m ← β₁·m + (1-β₁)·g          # 梯度的滑动平均（方向）
v ← β₂·v + (1-β₂)·g²         # 梯度平方的滑动平均（尺度）
参数 ← 参数 - lr · m̂ / (√v̂ + ε)   # 每个参数自己的有效学习率
```

直观理解：`m / √v` 大致是「梯度的平均幅度 / 梯度的典型幅度」，接近 ±1。所以 **Adam 天生对学习率不敏感**——不管梯度是 0.001 还是 100，更新量都被归一化到 lr 这个量级。这就是为什么 Adam 几乎开箱即用，成了深度学习默认优化器。

### 第四代：AdamW —— 修复一个挂错位置的正则项

**Weight Decay（权重衰减）**：每步把参数往 0 缩一点（`参数 ← 参数 × (1 - λ)`），防止权重无限膨胀、抑制过拟合。

问题来了：把 weight decay 实现「加到梯度上」（`g ← g + λ·参数`），在 SGD 里和直接缩参数**数学等价**；但在 Adam 里**不等价**！衰减项会被自适应缩放机制除以 `√v`，参数越大的维度反而衰减得越不正常，正则效果被扭曲。这被称为 **解耦权重衰减（decoupled weight decay）** 问题（Hinton 组 2017 年指出）。

**AdamW** 的修复简单粗暴：衰减不进梯度，直接在参数更新后再乘 `(1 - λ·lr)`。一行之差，让 weight decay 真正起作用。**今天你无脑选 AdamW 而不是 Adam，理由就在这里。**

## 三、在 PyTorch 中怎么用

PyTorch 把所有优化器统一在 `torch.optim` 下，接口完全一致：`zero_grad()` → `loss.backward()` → `step()`。

```python
import torch
import torch.nn as nn

# ---- 1. 一个玩具任务：拟合带噪声的正弦波 ----
torch.manual_seed(0)
x = torch.linspace(-3, 3, 500).unsqueeze(1)          # (500, 1)
y = torch.sin(x) * 1.5 + 0.1 * torch.randn_like(x)   # 目标带噪声

model = nn.Sequential(
    nn.Linear(1, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
)

# ---- 2. 三大优化器的标准写法 ----
opt_sgd   = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
opt_adam  = torch.optim.Adam(model.parameters(), lr=1e-3)
opt_adamw = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 实际训练时只用一个，这里以 AdamW 为例
optimizer = opt_adamw
criterion = nn.MSELoss()

# ---- 3. 标准训练循环（背下来，这是 PyTorch 的心跳） ----
for step in range(1000):
    optimizer.zero_grad()        # 清空上一步的梯度（梯度默认累加！）
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()              # 反向传播：计算所有参数的梯度
    optimizer.step()             # 优化器：根据梯度更新参数

    if step % 200 == 0:
        print(f"step {step:4d}  loss = {loss.item():.4f}")

# ---- 4. 进阶：参数组 —— 不同参数不同待遇 ----
# 典型场景：bias 和 norm 层不做 weight decay，只有权重矩阵做
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim <= 1 or name.endswith(".bias"):   # 1 维参数 = bias / LayerNorm
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},   # bias/norm 不衰减
], lr=1e-3)

# ---- 5. 学习率调度器：和优化器是一对 ----
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
for step in range(1000):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler.step()             # 注意：每个 step 后调用（旧版是每个 epoch）
```

## 四、围绕该领域展开：优化器在 PyTorch 生态中的位置

这是本文的重点。优化器不是孤立的 API，它处在好几个系统的交汇点上。

### 4.1 与 Autograd 的契约

优化器和 Autograd 之间的接口就是 `param.grad`。`loss.backward()` 把梯度写进每个参数的 `.grad` 属性，优化器的 `step()` 读它。理解这个契约，你能明白一堆现象：

- **为什么必须 `zero_grad()`**：Autograd 的设计是梯度累加（为了梯度累积等场景），不清零的话梯度会一直叠加。
- **为什么冻结参数要设 `requires_grad=False`**：不反传也就没有 `.grad`，优化器自然不会动它。
- **梯度裁剪发生在中间**：`loss.backward()` 之后、`optimizer.step()` 之前调用 `torch.nn.utils.clip_grad_norm_`，它直接修改 `.grad`，优化器无感知。

### 4.2 状态与显存：优化器也是显存大户

优化器是有**状态**的：Adam/AdamW 给每个参数额外存 `m` 和 `v` 两个同形状张量。算一笔账（以 FP16 混合精度训练 7B 模型为例）：

| 项目 | 倍数（相对参数量） |
|---|---|
| 参数（FP16） | 1× |
| 梯度（FP16） | 1× |
| 参数主副本（FP32） | 2× |
| Adam 的 m + v（FP32） | 4× |

这就是著名的「Adam 训练 = 约 16 字节/参数」，7B 模型光训练状态就要 100+ GB 显存——这正是 **FSDP/ZeRO 分片优化器状态**的直接动机（第 010 期讲过）。看懂这笔账，你就明白了为什么大模型时代人人都在和优化器状态搏斗，也理解了 8-bit Adam（把 m、v 量化到 int8）这类技术为什么有价值。

### 4.3 与分布式训练的纠缠

- **DDP 下的优化器**：梯度 AllReduce 发生在 `backward()` 内部（overlap 通信与计算），优化器看到的 `.grad` 已经是全卡平均后的梯度，`step()` 逻辑不变。
- **FSDP 下的优化器**：优化器状态本身被分片到各卡，`step()` 时临时 AllGather 参数——优化器被「劫持」了，但对用户代码透明。
- **主权重问题**：混合精度训练中优化器更新的是 FP32 主副本，而不是 FP16 参数本身，这是 `torch.cuda.amp` + GradScaler 联动的原因。

### 4.4 家族族谱与近亲

- **带动量 SGD** 至今仍是 CV 大规模训练（目标检测、分割、ResNet 系）的主流，泛化性常常好于 Adam 系。
- **Adam 系变体**：AdaBelief（用 v 和 m 的差做分母）、Lion（符号更新）、 Sophia（大模型专用、二阶信息）。它们都是对 m/v 结构的小改动——懂了 Adam，这些论文一天能读完三篇。
- **学习率调度器**（`torch.optim.lr_scheduler`）是优化器的孪生兄弟：Warmup + 余弦衰减是 Transformer 训练的标配。注意 AdamW + Warmup 是强绑定组合——Adam 的二阶矩估计在训练初期不稳定，warmup 就是给它缓冲时间。

## 五、什么时候该用 / 不该用

**用 SGD（+ momentum）当你：**
- 做 CV 任务、训练预算充足、追求最后 0.5% 的泛化精度
- 模型不深、损失曲面相对友好（ResNet 这类残差结构）
- 愿意认真调学习率和 schedule

**用 AdamW 当你：**
- 训练 Transformer / 大模型（工业界事实标准，没什么可犹豫的）
- 快速做实验、不想调参（Adam 对 lr 鲁棒，1e-3 / 3e-4 两档基本通吃）
- 有稀疏梯度场景（embedding、NLP）——自适应方法的优势区

**警惕：**
- **别用裸 Adam**：只要你想加 weight decay，就该用 AdamW。裸 Adam 里的 weight_decay 参数就是那个「挂错位置」的实现。
- **别在 BatchNorm/LayerNorm 参数和 bias 上做衰减**：这些参数本来就该保持原值，衰减它们纯属伤害。
- **Adam 不保证更优泛化**：很多 CV 任务上 AdamW 的最终精度仍差 SGD 一点，它是「快、稳、不挑食」，不是「处处最强」。

## 六、常见坑

**坑 1：忘记 `zero_grad()`，loss 越训越大**
梯度默认累加，不清零等于学习率被历史梯度不断放大。新版本可以写 `optimizer.zero_grad(set_to_none=True)`——把 `.grad` 直接置 None 而不是清成 0，还省一次内存写入，PyTorch 官方推荐。

**坑 2：优化器创建晚于参数替换 / 学习率改不动**
`optimizer = Adam(model.parameters())` 保存的是**参数对象的引用列表**。如果你之后 `model.fc = nn.Linear(...)` 替换了层，优化器还攥着旧参数——新层永远不更新。同理，直接改 `optimizer.param_groups[0]["lr"]` 是安全的改法，重新建优化器则要小心加载 state_dict。

**坑 3：保存 checkpoint 只存了模型权重**
`model.state_dict()` 不包含优化器状态。训练中断后恢复，如果只恢复模型不恢复优化器，Adam 的 m/v 清零，加上学习率调度器位置错误，恢复后的训练曲线会明显跳变。正确做法：模型、优化器、调度器、step 数一起存。

**坑 4：weight decay 当成正则万能钥匙乱开**
decay=0.01 只是 Transformer 的常见默认值，不是物理常数。decay 太大 + 训练太久，模型会被压向欠拟合；而且在 Adam 中设置 decay 的效果和 AdamW 中同数值完全不可比（前者被自适应缩放扭曲）。迁移配置时记得区分。

## 七、一句话总结

**优化器 = 「拿到梯度之后的一整套走法策略」：SGD 提供基准，动量解决噪声，自适应学习率解决尺度不一，AdamW 修复了正则项的挂错位置——而它的状态显存开销，正是大模型显存优化的主战场。**

## 八、今日练习

<details><summary>今日练习</summary>

**练习**：用同一个 MLP 拟合任务，分别用 SGD（无动量）、SGD+momentum、Adam、AdamW 四个优化器各训练 500 步，记录 loss 曲线。尝试回答：

1. 无动量 SGD 需要比 Adam 大多少倍的学习率才能达到相当收敛速度？
2. 把 Adam 的 weight_decay 和 AdamW 的 weight_decay 设成同一个值（如 0.1），哪个先崩？为什么？

**参考答案**：

```python
import torch
import torch.nn as nn

def train(opt_name, make_opt, steps=500):
    torch.manual_seed(0)
    x = torch.linspace(-3, 3, 500).unsqueeze(1)
    y = torch.sin(x) * 1.5 + 0.1 * torch.randn_like(x)
    model = nn.Sequential(
        nn.Linear(1, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )
    opt = make_opt(model.parameters())
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"{opt_name:16s}  初始 {losses[0]:.3f} → 最终 {losses[-1]:.4f}")
    return losses

train("SGD",        lambda p: torch.optim.SGD(p, lr=0.1))
train("SGD+moment", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9))
train("Adam",       lambda p: torch.optim.Adam(p, lr=1e-3))
train("AdamW",      lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.01))
```

**观察要点**：

1. SGD 通常需要 10~100 倍于 Adam 的学习率（0.1 vs 1e-3）才有相近速度——因为 Adam 的 `m/√v` 把更新量归一化到了 lr 量级，与梯度绝对大小无关。这正解释了「Adam 对学习率鲁棒、SGD 需要精调」。
2. 同为 weight_decay=0.1 时，Adam 更容易崩：衰减项混入梯度后被 `√v` 缩放，实际衰减强度在每个参数维度上被不可控地扭曲，等效的正则远超你设定的值；AdamW 的衰减是解耦的、可预测的。这也是 LAMB、AdamW 系论文反复强调的「耦合 decay 在自适应方法下等效强度随梯度尺度漂移」。

</details>

---

*下期预告话题池：Profiling、量化 PTQ/QAT、Hook 系统、CUDA Streams……*
