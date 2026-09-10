# PyTorch 每日一课 · 第 015 期

## Hook 系统：不改模型代码注入逻辑

> 📅 日期：2026-09-09
> 🎯 难度：⭐⭐⭐
> 📚 前置知识：nn.Module 基本用法、了解正向传播与反向传播流程（参考第 012 期 Autograd 引擎）
> ⏱️ 预计阅读时间：12 分钟

---

## 一、这个领域解决什么问题

先看一个真实场景：你接手了一个 50 层的 ResNet 训练代码，发现 loss 变成了 NaN。你想知道**到底是哪一层开始出问题**的——是第 30 层的激活值爆炸了？还是第 31 层的梯度消失了？

直觉的做法：打开模型源码，在每一层的 `forward()` 里加 `print`。但这就意味着：

1. **你要改模型源码**——如果模型来自第三方库（torchvision、HuggingFace），改源码意味着污染依赖，升级时全部丢失
2. **调试完还得改回去**——改来改去容易引入新 bug
3. **同一份逻辑想复用很难**——比如「记录每一层的 FLOPs」这个需求，换个模型就要重写一遍

这类需求的共同本质是：**想在训练/推理流程的固定位置"插一脚"，但不想动原有代码**。这就是 Hook（钩子）系统存在的意义——它是 PyTorch 官方提供的**观察点 + 注入点机制**，让你在不修改任何模型代码的前提下，拦截前向传播和反向传播的过程。

类似的机制在软件工程里到处都是：

- Git 的 pre-commit hook：提交前自动跑测试
- 浏览器的 DevTools 断点：不改网页代码就能观察执行
- Web 框架的中间件：不改业务逻辑就能加日志和鉴权

Hook 系统在 PyTorch 生态里的地位也是如此：**调试、可视化、正则化注入、模型分析，几乎所有"外围工具"都建立在它之上**。

---

## 二、核心思想：在流水线的传送带上开观察窗

把一次前向传播想象成工厂流水线：原料（输入张量）经过一站一站的工序（各个 Module），最后变成产品（输出）。反向传播则是流水线倒着走一遍（梯度从终点流回起点）。

PyTorch 在设计 `nn.Module` 时，在**每个站点前后**都预留了"窗口"：

```
前向传播：  输入 ──→ [Module] ──→ 输出
                   ↑      ↑
              forward    forward
              pre-hook   hook（拿到输入和输出）

反向传播：  grad_out ──→ [Module] ──→ grad_in
                      ↑        ↑
                 full_backward   tensor
                 hook            backward hook
```

具体来说，PyTorch 提供了四类 Hook，挂在两个层级上：

| Hook 类型 | 挂载层级 | 触发时机 | 能拿到什么 |
|---|---|---|---|
| `forward pre-hook` | Module 级 | 该 Module 的 forward **执行前** | 该层的输入 |
| `forward hook` | Module 级 | 该 Module 的 forward **执行后** | 该层的输入和输出 |
| `full backward hook` | Module 级 | 该层梯度计算**完成后** | 该层输入、输出的梯度 |
| `tensor backward hook` | Tensor 级 | 该张量的梯度**就绪时** | 该张量自己的梯度 |

两个关键设计决策值得体会：

**1. 注册而非修改。** `module.register_forward_hook(fn)` 把函数塞进这个 Module 内部的一个列表里，下次执行 forward 时 PyTorch 会自动遍历这个列表调用它们。模型代码一行不动，`state_dict` 一点不变， Hook 完全存在于模型的"账本"之外。

**2. Hook 的返回值可以篡改数据流。** 这是最容易被忽视的一点：forward pre-hook 如果返回一个新张量，**它会替换掉真正的输入**；forward hook 返回新张量则替换输出。这让 Hook 不只是"观察者"，还可以是"变形者"——后文讲的 dropout 注入、对抗样本生成都靠这个能力。

> ⚠️ 一个历史包袱要注意：`register_backward_hook` 在旧版本 PyTorch 中行为混乱（多个 Module 共享同一个 Tensor 时触发不可预测），从 1.8 起被废弃，替代品是 `register_full_backward_hook`。名字里的 "full" 就是在强调"这次语义是对的"。

---

## 三、在 PyTorch 中怎么用

### 3.1 最小示例：观察每一层的输出形状

```python
import torch
import torch.nn as nn

# 一个普通的玩具模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# 定义 hook 函数：签名是 (module, input, output)
def print_shape(module, inputs, output):
    # module 是触发 hook 的那个层本身，可以拿到它的名字和参数
    print(f"{module.__class__.__name__:>10}  输出形状: {tuple(output.shape)}")

# 给每一层都挂上 hook
handles = [layer.register_forward_hook(print_shape) for layer in model]

x = torch.randn(32, 784)   # 一个 batch 的"图片"
model(x)                    # 前向传播，hook 自动触发

# 用完记得摘掉，避免内存泄漏和重复触发
for h in handles:
    h.remove()
```

运行输出：

```
   Linear  输出形状: (32, 256)
    ReLU  输出形状: (32, 256)
   Linear  输出形状: (32, 10)
```

### 3.2 实战：找出梯度消失/爆炸的位置

这是 backward hook 最经典的用途——训练异常时定位问题层：

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64, 64), nn.Sigmoid(),   # 故意用 Sigmoid 制造梯度衰减
    nn.Linear(64, 64), nn.Sigmoid(),
    nn.Linear(64, 1),
)

# full backward hook 签名： (module, grad_input, grad_output)
# grad_output[0] 是"流出到下一层"的梯度（相对于本层输出）
# grad_input[0]  是"穿过本层之后"的梯度（相对于本层输入）
def check_grad(module, grad_input, grad_output):
    g_in = grad_input[0].abs().mean().item()
    g_out = grad_output[0].abs().mean().item()
    # 梯度穿过这一层后缩小的倍数，一目了然
    print(f"{module.__class__.__name__:>8}: 进入 {g_out:.3e} → 离开 {g_in:.3e}")

handles = [m.register_full_backward_hook(check_grad)
           for m in model if isinstance(m, nn.Linear)]

loss = model(torch.randn(8, 64)).sum()
loss.backward()   # 反向传播时 hook 逐层触发，从最后一层往前
```

你会清楚地看到 Sigmoid 层如何让梯度一层层缩水——这正是第 013 期讲优化器之前必须理解的现象。

### 3.3 进阶：用 forward pre-hook 篡改输入（对抗样本 / 测试注入）

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 2)

# pre-hook 签名：(module, input)，返回值会【替换】真正的输入
def add_noise(module, inputs):
    (x,) = inputs
    # 训练时给输入加高斯噪声，等价于一种数据增强/正则化
    if module.training:
        return (x + 0.1 * torch.randn_like(x),)
    return None  # 返回 None 表示不修改

model.register_forward_pre_hook(add_noise)

model.train()
out1 = model(torch.zeros(4, 10))   # 带噪声
model.eval()
out2 = model(torch.zeros(4, 10))   # 无噪声，和原模型完全一致
```

### 3.4 Tensor 级 hook：盯住某一个具体的值

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()

# tensor.register_hook 的回调只接收这个张量自己的梯度
x.register_hook(lambda grad: print(f"x 的梯度: {grad}"))

y.backward()
# 输出: x 的梯度: tensor([...])，即 2x
```

这个粒度最细，常用于检查某个中间损失、某组特殊参数（比如 LoRA 的增量矩阵）的梯度是否正常。

---

## 四、围绕该领域展开：Hook 在 PyTorch 生态中的位置

单看 API，Hook 只是几个 `register_*` 函数；但放眼整个生态，**PyTorch 的一大批核心功能都是围绕 Hook 机制构建的**。理解这一点，才算理解了这个"领域"。

### 4.1 与 nn.Module 的关系：`__call__` 里的秘密

为什么 Hook 能自动触发？答案藏在 `nn.Module.__call__` 里。当你写 `model(x)` 时（注意不是 `model.forward(x)`），实际执行的大致是：

```python
def __call__(self, *args, **kwargs):
    # 1. 依次调用所有 forward pre-hook（可能篡改输入）
    # 2. 调用 self.forward(*args, **kwargs)
    # 3. 依次调用所有 forward hook（可能篡改输出）
    # 4. 反向时通过 autograd graph 触发 backward hook
    ...
```

这也解释了一个经典面试题：**为什么必须用 `model(x)` 而不是 `model.forward(x)`？** 因为直接调 `forward()` 会绕过所有 Hook——包括 DDP 挂上去的梯度同步逻辑（第 006 期讲过 DDP 在 backward hook 里做 AllReduce 的 bucket 机制）。绕过它，多卡训练的梯度就不同步了。

### 4.2 与 Autograd 引擎的关系

Module 级 backward hook 底层是靠在计算图上插入自动微分节点实现的（复习第 012 期：Autograd 引擎沿着计算图反向调度节点）。Tensor 级 hook 则直接挂在 `grad_fn` 的执行路径上。所以：

- **Hook 是异步于 Python 主线程的**——GPU 上反向传播还在跑时，你的 hook 回调可能在稍后才被触发，所以分布式训练里 hook 里做通信同步要小心死锁
- **`no_grad` 上下文中 forward hook 照常触发**，但 backward hook 永远不会（没有计算图就没有反向）

### 4.3 生态中"寄生"在 Hook 上的功能

| 功能 | 用了什么 Hook | 在做什么 |
|---|---|---|
| **DDP 梯度桶通信** | backward hook | 梯度一就绪就触发 AllReduce（第 006 期） |
| **FSDP 参数搬运** | forward pre-hook | 前向前把分片参数 AllGather 回来（第 010 期） |
| **Gradient Checkpointing** | 计算图重算机制（hook 思想的亲戚） | 反向时重新执行被丢弃的前向（第 009 期） |
| **torchinfo / FLOPs 统计库** | forward hook | 遍历所有层，统计输入输出形状算 FLOPs |
| **特征可视化（Grad-CAM）** | forward + backward hook | 抓中间层激活和对应梯度，加权生成热力图 |
| **Weights & Biases / TensorBoard 的梯度直方图** | tensor hook | 训练中定期采样梯度分布 |
| **PNNX、torch.export 的追踪**（部分实现思路） | 执行路径观察 | 类似思想：不改代码观察执行流 |

看出规律了吗？**Hook 是"框架与外围世界之间的标准接口"**。你写模型时不用关心要不要支持可视化、要不要支持分布式——框架和工具通过 Hook 在你的模型外部"搭脚手架"。这是一种非常值得学习的架构思想：**好莱坞原则（Don't call us, we'll call you）**——控制流归框架，你只提供回调。

### 4.4 与 PyTorch 2.x 的关系：forward hook 会被 torch.compile 保留吗

一个现代常见坑：**`torch.compile` 默认会尽量不破坏 module 结构**，forward hook 基本可用；但如果触发了 graph break 优化路径，某些 hook 的触发顺序和次数可能与 eager 模式不同。做性能分析时（第 014 期），建议先用 eager 模式 + hook 定位问题，再用 compile 验证收益，不要混着来。

### 4.5 相关但不同的机制：`forward_hooks` 之外的注入手段

- **`__init__` 时包装子模块**（`nn.Module.__getattr__` 覆写）：比 Hook 更重，适合改变模型结构本身
- **`torch.func` 的函数式变换**（grad、vmap、functional_call）：不依赖 Hook，直接对"函数"做变换，是 JAX 风格的路线，PyTorch 2.x 重点发展方向
- **TorchDispatchMode / TorchFunctionMode**：更底层的拦截机制，能拦到每个 ATen 算子调用，是 functorch、FP8 训练这类深度改造的基础设施

一个粗略的分层：**TorchFunctionMode（拦算子）> Module Hook（拦层）> Tensor Hook（拦单个张量）**，侵入性递减，粒度递减。

---

## 五、什么时候该用 / 不该用

**该用 Hook 的场景：**

- ✅ 调试 NaN/Inf：定位梯度异常最早出现的层
- ✅ 提取中间层特征：做特征可视化、知识蒸馏的 soft target、Grad-CAM
- ✅ 训练监控：记录每层激活分布、梯度范数，画到 TensorBoard
- ✅ 无法改源码的场景：模型来自第三方库，或多人共享同一份模型代码
- ✅ 需要临时开关的逻辑：`handle.remove()` 一行就能干净撤掉

**不该用 Hook 的场景：**

- ❌ 把核心训练逻辑塞进 Hook：业务主逻辑应该写在 `training_step` 里，Hook 是旁路，可读性和可维护性都差
- ❌ 性能敏感的热路径上挂重计算：每个 step 都触发的 hook 里做 CPU-GPU 同步（`.item()`）会拖垮吞吐
- ❌ 需要改模型结构/参数形状的需求：该写自定义 Module 或用 `replace_module`，Hook 改不了"有哪些层"
- ❌ 与 `torch.compile` 深度耦合的复杂逻辑：hook + compile 的组合目前仍有边角案例，等价改写为显式代码更稳

---

## 六、常见坑

**坑 1：Hook 挂了不摘，训练越跑越慢 / 显存爆了**

Hook 的回调会持有你保存的张量引用。如果你在回调里 `features.append(output)` 做记录，每一步都在往一个无限增长的列表里塞 GPU 张量，显存迟早爆炸；而且 hook 列表本身越长，遍历越慢。**对策**：全局只保留必要步数的记录（环形缓冲），训练结束 `h.remove()`。

**坑 2：backward hook 的 `grad_input` 语义和你想的不一样**

`grad_input` 是"相对于该层**输入**的梯度"，即梯度已经**穿过了这一层**之后的结果，不是"进入这一层的梯度"。另外当一层有多个输入/输出时，`grad_input`/`grad_output` 是 tuple，且某些位置可能是 `None`（该输入不需要梯度）。**对策**：始终先打印结构确认，`grad_output[0]` 才是最接近"上游传来的梯度"的那个。

**坑 3：forward hook 里保存的输出张量，之后内容变了**

hook 给你的 `output` 张量可能是后续计算中被原地复用/修改的（尤其经过 ReLU、add_ 等操作）。想留档却拿到被污染的数据。**对策**：保存时显式克隆：`saved.append(output.detach().clone())`。

**坑 4：在 hook 回调里触发反向传播或修改 requires_grad 状态，导致死锁或图错误**

分布式训练中，在 backward hook 里再做通信（比如自己实现梯度平均）极易死锁——反向传播是异步调度的，此时再发起集合通信，不同 rank 的执行顺序无法保证一致。**对策**：优先用 DDP/FSDP 现成机制；必须自己实现时，用 reducer/延迟队列模式，不要在 hook 里同步等通信完成。

---

## 七、一句话总结

> Hook 系统是 PyTorch 在"不动你的代码"的前提下，向模型前向/反向流水线开的观察窗与注入点——它让你能观察、篡改、监控训练过程，而 DDP、FSDP、可视化工具等整个生态都寄生在这套机制之上；用它是为了旁路逻辑，永远别把主逻辑写进去。

---

## 八、今日练习

**题目：** 给定下面的小模型，请分别实现两个功能：
1. 用 forward hook 记录第一个 `nn.Linear` 层的输出，在每个 epoch 结束后打印这些输出范数的均值，然后**清空记录**（避免内存膨胀）
2. 用 full backward hook 检测：如果任何一层的 `grad_output` 中出现 NaN，立刻打印该层名字并停止记录

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
data = torch.randn(64, 32)
target = torch.randn(64, 1)
```

<details><summary>今日练习（参考答案）</summary>

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
data = torch.randn(64, 32)
target = torch.randn(64, 1)

# ---------- 功能 1：记录第一个 Linear 的输出范数 ----------
recorded_norms = []

def record_norm(module, inputs, output):
    # detach + 不 clone 也没关系：范数立即计算，不留张量引用
    recorded_norms.append(output.detach().norm().item())

# named_modules 拿到名字，方便定位"第一个 Linear"
hooks = []
for name, m in model.named_modules():
    if isinstance(m, nn.Linear) and "0" in name:
        hooks.append(m.register_forward_hook(record_norm))

# ---------- 功能 2：NaN 检测 ----------
nan_found = False

def check_nan(module, grad_input, grad_output):
    global nan_found
    if nan_found:
        return
    for g in grad_output:
        if g is not None and torch.isnan(g).any():
            print(f"⚠️ 层 {module.__class__.__name__} 出现 NaN 梯度，停止记录")
            nan_found = True

for m in model.modules():
    if isinstance(m, nn.Linear):
        hooks.append(m.register_full_backward_hook(check_nan))

# ---------- 模拟 3 个 epoch ----------
loss_fn = nn.MSELoss()
for epoch in range(3):
    out = model(data)
    loss = loss_fn(out, target)
    loss.backward()
    print(f"epoch {epoch}: 平均输出范数 = {sum(recorded_norms)/len(recorded_norms):.4f}")
    recorded_norms.clear()   # 关键：清空，防止内存膨胀
    model.zero_grad()

# 收尾：摘掉所有 hook
for h in hooks:
    h.remove()
```

**答案要点：**
1. **清空记录**是防坑 1 的关键——`recorded_norms.clear()` 每个 epoch 一次
2. hook 回调里只存 `.item()` 标量而不是张量，天然避免持有 GPU 显存引用
3. NaN 检测用标志位短路，一旦发现就不再重复扫描（大模型上 `torch.isnan().any()` 有开销）
4. `hooks` 列表统一管理，最后批量 `remove()`——这是工程上的好习惯

</details>

---

*下期预告：从「观察训练」回到「改造训练」——量化（Quantization）：PTQ vs QAT 的全景。*
