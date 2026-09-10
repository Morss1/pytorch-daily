# PyTorch 每日一课 · 第 001 期

## CUDA Graph：消灭 kernel launch 开销的终极手段

> 日期：2026-08-19  
> 难度：⭐⭐⭐⭐  
> 前置知识：知道 CPU 和 GPU 是分开的、用过 `.cuda()`  
> 预计阅读：8 分钟

---

### 这个领域解决什么问题

先理解一个你可能没意识到的事实：

**每次你调用一个 PyTorch 操作，CPU 都要「派遣」一个指令给 GPU。**

```python
x = x.cuda()
y = torch.relu(x)        # CPU 告诉 GPU：做一个 relu
z = torch.matmul(y, w)   # CPU 又告诉 GPU：做一个 matmul
out = z + b              # CPU 又告诉 GPU：做一个加法
```

这个「派遣」过程叫 **kernel launch**。每次 launch 都有开销，大约 **5-10 微秒**。

对于训练大模型，每个 kernel 本身运行几毫秒，5μs 的 launch 开销可以忽略。

但对于**推理小模型**或**小 batch size**，每个 kernel 只运行几微秒，此时 launch 开销反而成了主要瓶颈——你花在「指挥 GPU 做什么」上的时间，比 GPU 实际计算的时间还长。

**CUDA Graph 就是为了消灭这个开销而生的。**

---

### 核心思想：录制一次，回放无数次

类比：你在用遥控器操控机器人。每动一步，你都要按一次按钮，机器人收到指令才动。

- **普通模式**：每次都按按钮（kernel launch）
- **CUDA Graph 模式**：你先录一段「按按钮的序列」，之后一键回放，机器人按录好的序列连续执行

具体来说三步：
1. **Warmup（预热）**：先跑一遍，让 GPU 分配好内存、构建好内部状态
2. **Capture（录制）**：再跑一遍，这次 GPU 不真正执行，而是「录制」所有操作
3. **Replay（回放）**：之后每次调用，直接回放录制好的序列，零 CPU 开销

---

### 在 PyTorch 中怎么用

#### 手动方式（理解原理）

```python
import torch

# 准备静态输入（内存地址必须固定，不能换）
static_input = torch.randn(1, 3, 224, 224, device="cuda")
static_weight = torch.randn(768, 3, 224, 224, device="cuda")

# Step 1: Warmup 预热，跑 3 遍让 GPU 分配好内存
for _ in range(3):
    out = torch.relu(static_input)
    out = torch.nn.functional.conv2d(out, static_weight)
torch.cuda.synchronize()

# Step 2: Capture 录制计算图
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = torch.relu(static_input)
    out = torch.nn.functional.conv2d(out, static_weight)

# Step 3: Replay 回放
new_data = torch.randn(1, 3, 224, 224, device="cuda")
static_input.copy_(new_data)  # 把新数据写进固定地址
g.replay()                     # 一键回放，零 launch 开销
result = out.clone()           # 取出结果
```

注意关键点：
- `static_input` 的**内存地址**在整个过程中不能变
- 新数据用 `copy_()` 写进去，而不是重新创建张量
- 录制时什么操作序列，回放时就是什么操作序列——**形状和操作必须固定**

#### 现代方式（推荐日常使用）

```python
# torch.cuda.make_graphed_callables：自动处理 warmup + capture
model = MyModel().cuda().eval()
graphed_model = torch.cuda.make_graphed_callables(
    model, sample_args=(static_input,)
)

# 之后像普通模型一样用
static_input.copy_(new_data)
result = graphed_model(static_input)
```

---

### 围绕 CUDA Graph 展开：你需要知道的相关领域

理解 CUDA Graph 不能只看它本身，还要看它周围关联了什么。

#### 1. CUDA Stream（流）

GPU 上的操作不是排一个队，而是可以排到多个「流」上。同一个流内按序执行，不同流之间可以并行。**CUDA Graph 本质上是在一个流上录制操作序列**。理解 Stream 是理解 Graph 的前提。

#### 2. torch.compile + CUDA Graph

在 PyTorch 2.x 中，`torch.compile` 可以自动管理 CUDA Graph：

```python
model = torch.compile(model, mode="reduce-overhead")
```

`mode="reduce-overhead"` 底层就是用 CUDA Graph 来减少 launch 开销。你不需要手动 warmup/capture，编译器帮你搞定。**这是未来的主流方式。**

#### 3. 静态形状 vs 动态形状

CUDA Graph 的硬限制：**录制时的操作序列和形状必须和回放时一致**。

- ✅ 固定 batch size 的推理 → 完美适用
- ❌ 变长序列（NLP）→ 需要特殊处理（padding 到固定长度）
- ❌ 训练（梯度形状可能变）→ 一般不推荐

这也是为什么 CUDA Graph 在**推理部署**中应用最多，训练中较少。

#### 4. 什么时候不该用 CUDA Graph

- **batch size 很大**：kernel 运行时间远大于 launch 开销，用了也没提升
- **模型有动态控制流**（if/else 依赖数据）：录制时走一条路，回放时数据变了会出错
- **CPU 是瓶颈**：如果你 CPU 端数据预处理就慢，Graph 帮不上忙

---

### 常见坑

1. **忘了 copy_ 而是直接赋新值**
   ```python
   # ❌ 错误：创建了新张量，地址变了
   static_input = new_data
   # ✅ 正确：把数据写进固定地址
   static_input.copy_(new_data)
   ```

2. **Warmup 不充分**：第一次 capture 前，如果 CUDA 还没分配好内存，录制的图会包含内存分配操作，回放时会出错。一般 warmup 3 次。

3. **录制图中有随机操作**：Dropout 在录制时产生随机数，回放时用的是同一组随机数——如果你的模型有随机性，需要特殊处理。

4. **多 GPU + Graph**：每张 GPU 上的图是独立的，DDP 训练中用 Graph 很复杂，一般不用。

---

### 一句话总结

> **CUDA Graph 把「CPU 反复指挥 GPU」变成「录制一次，一键回放」，在 kernel launch 开销 >> kernel 执行时间的场景（小 batch 推理）中有显著加速。代价是必须静态形状。**

---

### 今日练习

1. 写一个小模型，测量用 CUDA Graph 前后的推理延迟差异（提示：用 `torch.cuda.Event` 计时）
2. 思考：为什么 PyTorch 在训练中默认不自动使用 CUDA Graph？
3. 尝试用 `torch.compile(model, mode="reduce-overhead")` 体验自动 CUDA Graph

<details>
<summary>练习 2 参考答案（点击展开）</summary>

训练过程中：(1) 梯度形状可能因 batch size 变化而改变；(2) 存在优化器更新等 CPU 端逻辑；(3) 动态控制流（如带条件分支的模型）无法被录制。这些都与 CUDA Graph 的「静态录制」要求冲突。而推理部署时模型固定、输入形状固定，完美匹配。
</details>

---

> 明天见。每天一个领域，不求全覆盖，只求真懂。
