# PyTorch 每日一课 · 第 012 期

## Autograd 引擎：反向传播在底层是怎么跑的

> **日期**：2026-09-04
> **难度**：⭐⭐⭐⭐
> **前置知识**：会写 `loss.backward()`，了解基本微积分（链式法则）
> **预计阅读时间**：15 分钟

---

## 一、这个领域解决什么问题

每个 PyTorch 用户每天都在写 `loss.backward()`，但很少有人追问：**这一行代码按下回车之后，机器里到底发生了什么？**

这个问题背后是一个真实的工程难题：

- 一个现代神经网络可能有**上亿个参数**、**几千个算子**，反向传播需要按拓扑序依次计算几千个梯度；
- 这些计算的**依赖关系**是动态的——你的 Python 代码里有 `if`、有循环，每次前向的图都可能不一样；
- 梯度计算还必须**自动**完成——你不可能手写每个算子的求导公式并维护它们与forward的一致性。

所以 PyTorch 需要一套系统，能：

1. 在前向计算时**自动记录**"谁依赖谁"（建图）；
2. 在调用 `backward()` 时**自动调度**所有梯度计算（反向执行）；
3. 对每个算子提供配套的求导规则（可扩展）。

这套系统就是 **Autograd**。理解它，你就理解了 PyTorch "动态图" 这个卖点到底动在哪、为什么调试比静态图框架容易、为什么 `retain_graph=True` 这种参数存在、以及为什么有些操作会"断图"。

---

## 二、核心思想：两条规则 + 一张动态图

### 2.1 链式法则的"乘法版"：反向模式自动微分

多元微积分里的链式法则有两种展开方向：

- **雅可比-向量积（JVP，正向模式）**：从输入端往输出端推，一次传播一个方向扰动；
- **向量-雅可比积（VJP，反向模式）**：从输出端往输入端推，一次传播一个标量损失对**所有**参数的梯度。

神经网络训练的目标是"一个 loss 对上亿个参数的梯度"——输出是 1 个标量，输入是海量参数，**反向模式一次扫描全部搞定**，这正是 `backward()` 选择 VJP 的原因。

一个直观类比：把前向计算想象成一条**装配流水线**，原材料（输入）经过一道道工序（算子）变成产品（loss）。

- **正向模式**（JVP）：想知道"某个原材料价格涨 1 元，产品涨多少"，你得为**每种原材料各跑一遍**流水线；
- **反向模式**（VJP）：从成品倒着走一遍流水线，每经过一道工序就问"这道工序的成本放大了多少倍"，**一趟下来所有原材料的答案全有了**。

### 2.2 动态建图：图是前向计算的"副产品"

与 TensorFlow 1.x "先定义图、再喂数据"不同，PyTorch 的计算图是**执行 Python 代码时顺手搭出来的**：

- 每个参与训练的张量身上带一个 `grad_fn`（记录"我是哪个算子产出的"）；
- 每执行一个算子，就新建一个 `Function` 节点，把输入张量的 `grad_fn` 挂成自己的父节点；
- 于是 Python 跑完一遍，一张 DAG（有向无环图）就自然形成了——**你的 Python 控制流就是图的结构**，`if` 换个分支，图就换一张。

这就是"动态图"的本质：**图的生命周期 = 一次前向的生命周期**。默认情况下 `backward()` 跑完图就释放，下次前向重新搭。

### 2.3 执行引擎：拓扑排序 + 就绪队列

调用 `backward()` 后，C++ 侧的 autograd 引擎做的事：

1. 从 loss 节点出发，对整张图做**拓扑排序**（按依赖关系排出执行顺序）；
2. 维护一个**就绪队列**：某个节点的所有子节点（梯度都到齐了）才入队；
3. 多线程执行——**不同节点的梯度如果互相独立，可以在不同线程并行算**；
4. 每个节点的任务是：拿到下游传回的梯度（`grad_output`），调用该算子注册的 `backward` 公式，产出传给上游的梯度。

这背后有一套精妙的引用计数机制：每个节点记录"我还有几个梯度没收"，收到一个减一，减到零才真正执行——和 Python 的引用计数思想如出一辙。

---

## 三、在 PyTorch 中怎么用

### 3.1 观察图的存在

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 3
z = y.sum()

# grad_fn 就是"图"露出的把手——每个中间张量都记得自己怎么来的
print(y.grad_fn)  # <MulBackward0 object at 0x...>
print(z.grad_fn)  # <SumBackward0 object at 0x...>

# 顺着 grad_fn 往回走，就是完整的前向历史
node = z.grad_fn
while node is not None:
    print(type(node).__name__)  # SumBackward0 -> MulBackward0 -> AccumulateGrad
    node = node.next_functions[0][0] if node.next_functions else None
```

### 3.2 backward() 的基本形态

```python
z.backward()          # 从 z（标量）出发反向传播
print(x.grad)         # tensor([3., 3.])  —— dz/dx = 3
```

### 3.3 自定义算子：自己写求导规则

这是理解"每个算子都注册了 VJP 公式"的最好方式：

```python
import torch

class MySquare(torch.autograd.Function):
    """自定义 y = x^2，手动写反向公式"""

    @staticmethod
    def forward(ctx, x):
        # ctx 是上下文袋子，正向存的东西反向可以取出来
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors          # 取回正向保存的输入
        return grad_output * 2 * x        # d(x^2)/dx = 2x，乘上链式法则传来的梯度

# 用 apply 调用（不是直接调 forward）
x = torch.tensor([2.0], requires_grad=True)
y = MySquare.apply(x)
y.backward()
print(x.grad)  # tensor([4.])  —— 2*2=4，公式生效
```

### 3.4 控制图的常用开关

```python
# 1) 不追踪：推理/数据预处理时关掉，省内存省时间
with torch.no_grad():
    out = model(x)

# 2) 图已释放还想再 backward 一次
loss.backward(retain_graph=True)  # 保留图，可再跑一次

# 3) 中间变量的梯度也想看
y = model.mid_layer(x)
y.retain_grad()

# 4) 梯度累加（不 zero_grad 直接多次 backward）——模拟大 batch
# 常用于显存不够时分步累积梯度，本质是利用 grad 是"累加器"这个设计
```

---

## 四、围绕该领域展开：Autograd 在生态中的位置

这一节是重点——Autograd 不是孤岛，PyTorch 半个生态都建在它上面。

### 4.1 AccumulateGrad：`x.grad` 为什么是累加的

叶子张量（你直接创建、`requires_grad=True` 的张量）在图里挂着一个特殊的 `AccumulateGrad` 节点——反向传播到它这里时，不是"赋值"而是 **`+=`** 到 `.grad` 上。这就解释了：

- 为什么训练循环里要 `optimizer.zero_grad()`——不清零，上一个 batch 的梯度会混进来；
- 为什么**梯度累积**（gradient accumulation）不需要任何特殊 API——只要不清零，天然生效。

### 4.2 与 Hook 系统：往引擎里"埋点"

Autograd 提供了两类钩子，让你不改模型代码就能观察/修改梯度流动：

```python
# 张量级 hook：查看/修改某个参数收到的梯度
x.register_hook(lambda g: print(f"grad arrived: {g}") or g)

# 模块级 hook：监控整个模块的反向（第 014 期 Hook 系统的主角）
model.layer3.register_full_backward_hook(...)
```

梯度裁剪（`clip_grad_norm_`）、GradNorm、各种可视化和反梯度攻击，都是靠这套机制实现的。

### 4.3 与 torch.compile：图捕获的上下游关系

Dynamo 捕获的是** FX 图**（算子级别的数据流），但反向传播依然要靠 Autograd 的 `nested functions` 机制生成——AOTAutograd 的本质就是"提前把 backward 图也 traced 出来"，让 Inductor 能把前反向一起编译。**Autograd 是源头，compile 是它的下游加速器**。

### 4.4 与分布式：DDP 的梯度同步搭 Autograd 的便车

DDP 的经典实现（reducer bucket）就是在反向传播的 hook 里插入 AllReduce：**梯度一算完就异步通信**，通信和剩余的反向计算重叠。如果没有 Autograd 的 hook 机制，DDP 只能在整个 `backward()` 结束后才开始同步，效率大打折扣。

### 4.5 与推理引擎：export 时图必须"断梯度"

`torch.export` 导出的是纯前向图。所以导出前要把模型变成 `no_grad`/inference 模式，剥掉所有 `grad_fn` 关联——这也是为什么 `.eval()` + `torch.no_grad()` 是推理的标准姿势：**前者管模块行为（BN/Dropout），后者管图构建**，两者缺一不可。

### 4.6 不支持自动求导的操作

`inplace` 修改、直接改 `.data`、某些非可导操作（如 `torch.argmax`）会**悄悄断图**或让梯度路径失效。这是初学者 bug 的重灾区（见下文常见坑）。

### 4.7 一图看全家

```
 前向: x ──[算子A]── y ──[算子B]── z ──[算子C]── loss
        │              │              │
        │ (执行时建图)  │              │
        ▼              ▼              ▼
 图节点: FnA ←──────── FnB ←──────── FnC ←── backward(loss) 从这出发
        │              │
        └── grad_fn 链把前向历史完整记下来 ──┘

 下游生态:
   AccumulateGrad ──→ 优化器 / 梯度累积
   autograd hook  ──→ DDP 同步 / 梯度裁剪 / 可视化
   图捕获        ──→ AOTAutograd → Inductor 编译
   图剥离        ──→ torch.export / 推理引擎
```

---

## 五、什么时候该用 / 不该用

**该深入理解 Autograd 的场景：**

- 排查"梯度为 None / 梯度爆炸 / 梯度不流动"类 bug——90% 是图被意外切断；
- 写自定义算子（新激活函数、特殊损失）需要自定义求导；
- 实现梯度相关的训练技巧：梯度累积、梯度裁剪、对抗训练、influence function；
- 阅读 DDP/PEFT/对抗攻防等库的源码——它们全是 hook 重度用户。

**不需要 / 应该绕开的场景：**

- 纯推理服务：`torch.no_grad()` 或 `inference_mode` 全程关掉，能省掉建图的全部开销；
- 极致性能场景：动态图调度有 Python/C++ 边界开销，交给 `torch.compile` 去消除；
- 损失不可导的问题（组合优化、离散采样）：自动微分帮不了你，需要 RL 或 Straight-Through 之类的技巧。

---

## 六、常见坑

**坑 1：忘了 `zero_grad()`，loss 不降**

`AccumulateGrad` 是累加语义。多个 batch 的梯度加在一起，等于在用一个不断膨胀的"超级 batch"训练。症状是训练前期看起来还行，很快发散。

**坑 2：inplace 操作破坏图**

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
y += 1        # 💥 RuntimeError: a leaf Variable that requires grad
              # is being used in an in-place operation
# 更隐蔽的情况：中间张量被 inplace 改写，backward 时版本号对不上直接报错
```

Autograd 给每个张量带版本号（`_version`），inplace 修改会更新版本号，反向传播发现保存的输入和当前版本不一致就报错——这是保护，不是刁难。

**坑 3：用 `.data` 或 `detach()` 绕过检查，梯度悄悄错误**

`.data` 绕过 autograd 的版本追踪，**不报错但梯度可能是错的**——比如 inplace 改了 `.data`，反向照常跑，结果静默出错。正确做法是用 `with torch.no_grad():` 块，它至少会被追踪到。

**坑 4：对非标量直接 backward**

```python
out = model(x)          # shape [B, C]
out.backward()          # 💥 grad can be implicitly created only for scalar outputs
out.sum().backward()    # ✅ 或者传一个同 shape 的权重向量:
out.backward(torch.ones_like(out))  # ✅ 相当于对加权和求导
```

`backward(gradient=...)` 那个参数本质上就是 VJP 里的"向量"——理解了这一点，这个报错就永远不会再困扰你。

---

## 七、一句话总结

> Autograd = 前向时顺手搭一张动态 DAG + 每个算子预注册一条 VJP 公式 + 反向时按拓扑序多线程调度——你写的是 Python 控制流，它跑的是自动化的链式法则。

---

<details>
<summary>今日练习</summary>

**练习 1**：不使用 `torch.pow`，用 `torch.autograd.Function` 实现 `y = x³` 的前向和反向，验证 `x=2` 处梯度为 `12`。

**练习 2**：用 `register_hook` 找出下面网络中哪一层的梯度范数最大（梯度在哪一层"最猛"）：

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
```

**练习 3**：解释为什么下面代码第二次 `backward()` 会报错，以及两种修复方式分别适用于什么场景：

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()
y.backward()   # 💥
```

---

**参考答案**：

**练习 1**：

```python
import torch

class Cube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 3

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 3 * x ** 2   # d(x³)/dx = 3x²

x = torch.tensor([2.0], requires_grad=True)
Cube.apply(x).backward()
print(x.grad)  # tensor([12.])
```

**练习 2**：

```python
norms = {}
hooks = []
for name, linear in model.named_modules():
    if isinstance(linear, nn.Linear):
        # 给每层的权重挂 hook，记录反向时收到的梯度范数
        hooks.append(linear.weight.register_hook(
            lambda g, n=name: norms.update({n: g.norm().item()})
        ))

x = torch.randn(8, 64)
model(x).sum().backward()
print(norms)   # 输出各层梯度范数，比较即可
```

这其实就是梯度裁剪和深度网络诊断（判断是否梯度消失/爆炸）的最小实现。

**练习 3**：默认情况下 `backward()` 执行完会释放中间结果（缓冲区），图随之销毁，第二次调用自然没有图可跑。
- 修复 A：`y.backward(retain_graph=True)`——保留图，适合需要多次反向（如 RNN 训练技巧、计算高阶导数）的场景，代价是显存占用；
- 修复 B：重新前向建一张新图再 backward——适合"重算比保留更便宜"的常规训练场景（默认行为）。

</details>
