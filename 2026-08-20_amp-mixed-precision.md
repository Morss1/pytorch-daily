# PyTorch 每日一课 · 第 002 期

## 混合精度训练（AMP）：FP16 / BF16 / FP8 的取舍

> 日期：2026-08-20  
> 难度：⭐⭐⭐  
> 前置知识：知道模型训练的前向 / 反向传播流程、用过 `.to(device)`  
> 预计阅读：10 分钟

---

### 这个领域解决什么问题

先看一个扎心的账本。一个 7B 参数的模型，如果全部用 FP32（32 位浮点数）训练：

- 模型参数：7B × 4 字节 = **28 GB**
- 梯度：又一份 28 GB
- 优化器状态（Adam 的一阶 + 二阶动量）：再两份 28 GB

还没算激活值，显存就已经爆了。而 FP16 每个数字只占 2 字节，**直接省一半**；更关键的是，现代 GPU（如 A100/H100）的 Tensor Core 对 FP16/BF16 矩阵乘法的吞吐量是 FP32 的 **8~16 倍**。

那问题来了：**为什么不干脆把所有东西都换成 FP16？**

因为会训崩。FP16 的表示范围太小（最大约 65504，最小正常数约 6e-5），大梯度直接上溢变成 `inf`，小梯度直接下溢变成 0——尤其是反向传播中大量梯度小于 1e-5，全都会消失。

**混合精度训练（Automatic Mixed Precision，AMP）就是为了在这两者之间取得平衡而生的领域：用低精度换取速度和显存，同时用一整套工程手段保证数值稳定。**

---

### 核心思想：该精的地方精，该快的地方快

类比：你是一个公司的会计，日常记账（大量矩阵乘法）用「万元」为单位粗算就行——快；但最后进总账（权重更新）必须用「元」精确记录，否则每次四舍五入的误差积累起来，几个月后账就乱了。

混合精度的「混合」体现在三个层面：

**1. 算子层面：敏感的用 FP32，不敏感的用低精度**

- 矩阵乘法、卷积：误差会互相抵消，对精度不敏感 → **用 FP16/BF16**（还能吃 Tensor Core 加速）
- softmax、loss 计算、逐元素累加：误差会累积，对精度敏感 → **保持 FP32**

**2. 权重层面：Master Weights（主权重）**

模型权重用 FP16 算前向反向，但**优化器里永远保留一份 FP32 的主权重**。原因：`w = w - lr × grad`，当 `lr × grad` 比 `w` 小太多时（比如 w=1.0，更新量=1e-7），FP16 下这个更新会被直接舍入掉——`1.0 + 1e-7` 在 FP16 里还是 `1.0`。训练几万步，模型就「原地踏步」了。

**3. 梯度层面：Loss Scaling（损失缩放）**

反向传播从小到大地传递梯度，FP16 下小梯度容易下溢成 0。解决方案简单粗暴：先把 loss 乘一个大数（比如 65536），让所有梯度跟着整体放大，落回 FP16 的可表示范围；优化器更新前再除回来。

```
loss × 65536 → 反向传播（梯度整体放大 65536 倍）→ 梯度 ÷ 65536 → 更新 FP32 主权重
```

如果放大后溢出了？动态 loss scaling 会检测到 `inf/nan`，跳过这一步更新，把缩放系数减半重试。

**BF16 出现后，局面变了**：BF16 的指数位和 FP32 一样多（8 位），表示范围与 FP32 相同，基本不会上下溢。代价是尾数位少（只有 7 位），精度差一点——但对深度学习足够。所以 **Ampere 之后的卡 + BF16 = 可以不做 loss scaling**，整个流程简单了一大截。

---

### 在 PyTorch 中怎么用

#### 日常用法：autocast + GradScaler

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# GradScaler：负责动态 loss scaling（仅 FP16 需要，BF16 不需要）
scaler = torch.amp.GradScaler("cuda")

for step, (x, y) in enumerate(dataloader):
    x, y = x.cuda(), y.cuda()

    optimizer.zero_grad()

    # autocast：自动决定每个算子用什么精度，你不用管
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred = model(x)
        loss = criterion(pred, y)

    # 注意：scaler.scale(loss) 放大 loss，再反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)   # 内部会先 unscale，再检查 inf/nan，再 step
    scaler.update()          # 根据是否溢出，动态调整缩放系数

    if step % 100 == 0:
        print(f"step {step}, loss {loss.item():.4f}")
```

关键点：
- `autocast` 上下文**只包前向传播**。反向传播会自动使用前向时录制的精度，不需要（也不应该）再包一层
- `GradScaler` 只在 `dtype=torch.float16` 时需要；**BF16 下直接不用 scaler**

#### BF16 写法（更简单，推荐 A100/H100 用户）

```python
# BF16：范围和 FP32 一样，不需要 loss scaling
scaler = None  # 不需要！

for step, (x, y) in enumerate(dataloader):
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = model(x)
        loss = criterion(pred, y)

    loss.backward()      # 直接 backward
    optimizer.step()     # 直接 step
```

#### 不用 GPU 也能体验：CPU 上的 autocast

```python
# 在 Mac / CPU 上也能感受混合精度（Apple Silicon 对 BF16 有加速）
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    c = a @ b
    print(c.dtype)  # torch.bfloat16 —— matmul 在 BF16 下执行
```

#### 想看 autocast 到底对每个算子做了什么？

```python
# 打开 autocast 的详细日志，看每个算子被分派到什么精度
torch.autocast(device_type="cpu", dtype=torch.bfloat16).__enter__()
print(torch.autocast.get_autocast_cpu_list())
```

---

### 围绕混合精度展开：你需要知道的相关领域

#### 1. 数值格式家族：FP32 / TF32 / FP16 / BF16 / FP8

理解 AMP 的前提是理解这些格式的区别。一个浮点数 = 符号位 + 指数位（决定**范围**）+ 尾数位（决定**精度**）：

| 格式 | 指数位 | 尾数位 | 范围 | 典型用途 |
|------|--------|--------|------|----------|
| FP32 | 8 | 23 | ~1e38 | 传统训练基准 |
| TF32 | 8 | 10 | ~1e38 | A100 上 FP32 矩阵乘的「隐形加速」（默认开启，无需改代码） |
| FP16 | 5 | 10 | ~65504 | Tensor Core 加速，需要 loss scaling |
| BF16 | 8 | 7 | ~1e38 | **大模型训练主流**，不需要 loss scaling |
| FP8 (E4M3/E5M2) | 4/5 | 3/2 | 更小 | H100 时代的前沿，权重/激活/梯度分别用不同格式 |

规律：**指数位决定会不会溢出，尾数位决定算得准不准**。深度学习对「溢出」比对「不准」敏感得多，这就是 BF16 和 FP8 设计哲学的出发点。

#### 2. Tensor Core：混合精度为什么快

GPU 上专门做矩阵乘法的硬件单元。A100 的 Tensor Core 一个时钟周期就能完成一个 4×4 FP16 矩阵乘。**AMP 的速度收益几乎全部来自 Tensor Core**——所以如果你的模型瓶颈不在矩阵乘法（比如大量小算子、内存带宽受限），AMP 的加速会很有限。

#### 3. 与 torch.compile 的关系

`torch.compile` 生成的 kernel 会尊重 autocast 的 dtype 策略，且 Inductor 能进一步融合算子、减少低精度与 FP32 之间的转换开销。现代大模型训练的标配组合是：**BF16 autocast + torch.compile + FSDP**。

#### 4. 与显存策略的配合

AMP 主要省的是**参数、梯度、激活值**的显存（计算直接在低精度 buffer 上做）。但优化器里的 master weights 和 Adam 动量仍然是 FP32——这部分要靠 **FSDP / ZeRO** 分片省。这些技术是互补关系，不是替代关系。

#### 5. 溢出检测与调试

分布式训练中排查 NaN 的工具链：`torch.autograd.set_detect_anomaly(True)`（慢，但能定位到具体算子）、`torch.distributed.algorithms.ddp_comm_hooks` 里的通信后检查、以及简单地逐层打印梯度的 max/abs 值。混精训练训崩了，第一反应永远是：**先切 BF16 排除溢出问题，再查数据和学习率**。

---

### 什么时候该用 / 不该用

**该用（几乎总是）：**
- GPU 是 Ampere（A100/RTX 30 系）及之后 → 无脑 BF16
- 训练大模型、显存吃紧 → AMP 是最便宜的显存优化，一行代码
- 推理部署 → 低精度收益更大（还连上了量化生态）

**需要谨慎：**
- 强数值敏感的任务：GAN 训练、强化学习的 critic、涉及求逆矩阵的层——对精度误差敏感，混精可能不稳定
- 老卡（V100 及以前）：没有 BF16 硬件支持，只能 FP16 + scaler，更容易踩坑
- 模型极小且瓶颈在数据加载：AMP 换不来速度，反而多了 dtype 转换开销

---

### 常见坑

1. **BF16 还在用 GradScaler**
   ```python
   # ❌ 不需要：BF16 不会下溢，scaler 是给 FP16 用的
   scaler = torch.amp.GradScaler("cuda")  # dtype=bfloat16 时纯属多余
   # ✅ BF16 直接 loss.backward() + optimizer.step()
   ```

2. **autocast 包住了整个训练循环（包括反向和 step）**
   ```python
   # ❌ 错误：反向传播和优化器更新不需要、也不应该被 autocast 包住
   with torch.autocast(...):
       loss.backward()
       optimizer.step()
   # ✅ 正确：autocast 只包前向计算（loss 计算可以包含在内）
   ```

3. **unscale 之前就做梯度裁剪**
   ```python
   # ❌ 错误：此时梯度还是被放大过的，阈值完全失真
   scaler.scale(loss).backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 裁的是放大后的梯度
   scaler.step(optimizer)
   # ✅ 正确：先手动 unscale_
   scaler.unscale_(optimizer)
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   scaler.step(optimizer)
   ```

4. **在 autocast 区域里手动转换 dtype**
   ```python
   # ❌ 不要自己 cast，autocast 会为每个算子做最优决定
   with torch.autocast("cuda", dtype=torch.float16):
       x = x.half()          # 多此一举，还可能干扰 autocast 的决策
       out = model(x)
   ```

---

### 一句话总结

> **混合精度训练的本质是一场精度经济学：用 BF16/FP16 做不敏感的大规模矩阵乘法换取 Tensor Core 速度和一半显存，用 FP32 主权重 + loss scaling 守住数值稳定的底线。Ampere 之后，无脑 BF16 几乎总是对的。**

---

### 今日练习

1. 写一个两层 MLP，分别在 FP32 / FP16+scaler / BF16 下各训练 500 步，对比 loss 曲线和每步耗时（提示：`torch.cuda.Event` 计时）
2. 思考：为什么 `1.0 + 1e-7` 在 FP16 里等于 `1.0`，而这对于「用 FP16 权重直接做优化器更新」是致命的？
3. 查一下你的显卡（或云上租的卡）的架构，判断它支持 BF16 吗？FP8 呢？

<details>
<summary>练习 2 参考答案（点击展开）</summary>

FP16 的尾数只有 10 位，能表示的相邻两个数之间的间隔约为数值本身的 1/1024（约 1e-3 量级的相对精度）。当 w=1.0 时，比 1e-3 更小的增量都会被舍入掉，所以 `1.0 + 1e-7 == 1.0`。

如果直接用 FP16 权重做更新 `w ← w - lr × grad`：小学习率或小梯度时，每一步的更新量都可能被完全舍入为 0，模型参数永远不动——表面上 loss 曲线「正常下降几步然后卡住」，实际是参数冻结。这就是 AMP 必须维护 FP32 master weights 的根本原因：更新量在 FP32 下被完整保留，累积多步之后再转回 FP16 就不会丢失。

</details>

---

> 明天见。每天一个领域，不求全覆盖，只求真懂。
