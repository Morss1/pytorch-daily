# PyTorch 每日一课 · 第 016 期

## 量化 (Quantization)：PTQ vs QAT 的全景

> **日期**：2026-09-10
> **难度**：⭐⭐⭐⭐
> **前置知识**：基本的张量操作、神经网络训练流程、对浮点数表示有初步概念
> **预计阅读时间**：15 分钟

---

## 一、这个领域解决什么问题

先看一个扎心的数字：你训练好一个 ResNet-50，模型权重是 FP32 的，**98 MB**。要部署到手机上？用户下载 App 时看到「需要下载 98 MB 模型」大概率直接卸载；要部署到边缘设备上？内存和算力都不够看；想跑得快点？FP32 的乘加运算在专用低精度硬件（如 NPU、Tensor Core INT8）上比 FP32 快 2~4 倍。

**量化的本质就一句话：用更少的比特表示原本用浮点数表示的数值，同时尽量不损失模型精度。**

这里有一个天然的矛盾：

- **比特越少，越快越省**：INT8 乘法器比 FP32 简单得多，存储和带宽直接除以 4；
- **比特越少，信息越少**：FP32 有约 7 位有效十进制数字，INT8 只有 256 个离散取值。

量化领域要解决的核心工程问题就是：**如何在这个矛盾中找到最佳平衡点**——用最少的精度损失换取最大的速度和体积收益。

量化的应用场景几乎是今天所有大模型落地的基础：

- 手机端 LLM（ llama.cpp 的 Q4/Q8 量化）；
- 云端推理服务的成本优化（INT8 推理吞吐翻倍）；
- 微调不动大模型时的 LoRA + 4bit 底座（QLoRA）。

---

## 二、核心思想：用「体重秤」和「钢琴键」理解量化

### 2.1 仿射量化（Affine Quantization）

最常用的量化方案叫**线性仿射量化**。它用一个公式把浮点数映射到整数：

```
q = round(x / scale) + zero_point
```

反量化（还原）时：

```
x ≈ (q - zero_point) * scale
```

类比一下：**把一根连续的尺子换成一架钢琴**。

- 浮点数是尺子上任意位置——你可以指在 3.14159 cm 处；
- INT8 是钢琴键——只有 88 个键（INT8 有 256 个），你只能「弹最接近的那个键」。

`scale`（缩放因子）就是「一格代表多长」，`zero_point`（零点偏移）解决「浮点 0 不一定对应整数 0」的问题。比如权重分布是 `[0.1, 0.2, ..., 3.0]`，全为正数，如果对称量化，负半边整数全浪费了——加个 zero_point 偏移，就能把整个整数区间用满。

### 2.2 对称 vs 非对称

- **对称量化**（`zero_point = 0`）：`scale = max(|x|) / 127`。简单，硬件友好（反量化少一次减法），适合权重（权重分布通常近似对称）。
- **非对称量化**：`scale = (max - min) / 255`，`zero_point = -round(min / scale)`。区间利用率高，适合激活值（ReLU 之后全为非负，分布天然不对称）。

### 2.3 Per-Tensor vs Per-Channel

如果一层权重的 512 个输出通道中，某个通道的数值范围特别大（离群通道，LLM 中极其常见），Per-Tensor（整个张量共用一个 scale）会把这个通道「绑架」——其他通道被迫用粗粒度。**Per-Channel**（每个输出通道一个 scale）是解药，也是今天 LLM 量化的标配。代价是反量化时需要按通道操作，硬件支持稍复杂。

### 2.4 两大流派：PTQ vs QAT

这是整个量化领域的「华山论剑」，也是本期的主角：

| 维度 | PTQ (训练后量化) | QAT (量化感知训练) |
|---|---|---|
| 做法 | 拿训好的 FP32 模型直接量化 | 训练/微调时就模拟量化 |
| 需要训练数据吗 | 只需少量校准数据（几百条） | 需要完整训练管线 |
| 成本 | 分钟级，零代码改动 | 小时~天级，要改训练代码 |
| 精度 | INT8 通常掉 0~2 个点 | 接近 FP32，甚至 INT4 可用 |
| 适用 | 快速验证、云端 INT8 部署 | 追求极致低位宽（INT4/INT2） |

**PTQ 的核心是「校准」（Calibration）**：不训练，只是拿几百个样本过一遍模型，统计激活值的分布（min/max 或直方图），据此确定 scale。统计方法本身是门学问：

- **Min-Max 校准**：直接取观测到的最大最小值。简单但对离群点敏感；
- **Percentile 校准**：截掉 0.1% 的极端值，让 scale 更贴合主体分布；
- **MSE / KL 散度校准**：搜索使量化误差（重建误差或分布距离）最小的截断点。TensorRT 的早期经典方法就是 KL 散度（也即 entropy 校准）。

**QAT 的核心是「伪量化」（Fake Quantize）**：训练时在计算图里插入「量化再立刻反量化」的节点：

```
x_fake = dequant(quant(x))
```

前向传播用 `x_fake`，于是**损失函数真实地「感知」到了量化误差**；反向传播时用直通估计器（STE, Straight-Through Estimator）——round() 不可导，就当它导数恒为 1，梯度直接穿过去。模型权重会在训练过程中**主动迁移到对量化更友好的区域**，这就是 QAT 精度更高的根本原因。

一个直观的理解：PTQ 是「考完试再改答案」（模型已定型，只能事后修补），QAT 是「考前就按考试格式练习」（模型在学习时就适应了低精度表达）。

---

## 三、在 PyTorch 中怎么用

PyTorch 的量化 API 经历了三代演进，现在主线是 **TorchAO**（PyTorch 官方的架构优化库），但经典的 `torch.ao.quantization` 流程仍是最适合理解原理的教材。我们从 PTQ 和 QAT 各看一个可运行示例。

### 3.1 动态量化（最简单的入门：一行代码）

```python
import torch
import torch.nn as nn

# 动态量化：权重提前量化为 INT8，激活值在推理时动态量化
# 只有 Linear / LSTM / RNN 这类「权重密集」层收益最大
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10),
)

quantized_model = torch.ao.quantization.quantize_dynamic(
    model,                    # 要量化的 FP32 模型
    {nn.Linear},              # 指定量化的层类型
    dtype=torch.qint8         # 目标数据类型
)

print(quantized_model)  # Linear 层已变成 DynamicQuantizedLinear
```

动态量化的激活值在推理时动态计算 scale，无需校准，是 LSTM 类模型部署的经典选择（当年 BERT 推理提速的标准操作）。

### 3.2 静态 PTQ（完整流程：融合 → 校准 → 转换）

```python
import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)

fp32_model = ToyModel().eval()

# ── 第 1 步：设置量化配置 ──────────────────────────────
fp32_model.qconfig = torch.ao.quantization.get_default_qconfig("x86")
# "x86" 后端用 fbgemm（服务器 CPU）；ARM 手机用 "qnnpack"

# ── 第 2 步：融合 Conv + BN + ReLU 为单个算子 ──────────
# 融合是关键：INT8 下分开做三次 rounding 误差会叠加
fused_model = torch.ao.quantization.fuse_modules(
    fp32_model, [["conv", "relu"]]
)

# ── 第 3 步：准备校准（在模型中插入 Observer）─────────
prepared = torch.ao.quantization.prepare(fused_model)

# ── 第 4 步：用校准数据跑前向，统计激活分布 ──────────
# 实际中用几百条真实数据，这里用随机数据演示
for _ in range(100):
    dummy = torch.randn(4, 3, 32, 32)
    prepared(dummy)

# ── 第 5 步：转换为真正的 INT8 模型 ──────────────────
int8_model = torch.ao.quantization.convert(prepared)

# 验证：输出与 FP32 模型接近但不完全相同
x = torch.randn(1, 3, 32, 32)
print(fp32_model(x))   # FP32 输出
print(int8_model(x))   # INT8 输出（有微小偏差）
```

### 3.3 QAT（训练时感知量化）

```python
import torch
import torch.nn as nn

model = ToyModel().eval()  # 沿用上面的 ToyModel
model.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")

# 第 1 步：插入伪量化节点（FakeQuantize）
qat_model = torch.ao.quantization.prepare_qat(model)

# 第 2 步：照常训练！损失函数会看到量化误差
optimizer = torch.optim.SGD(qat_model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for step in range(100):
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    loss = loss_fn(qat_model(x), y)
    optimizer.zero_grad()
    loss.backward()   # STE 让梯度穿过 round()
    optimizer.step()

# 第 3 步：训练完成后转为真量化模型
qat_model.eval()
int8_model = torch.ao.quantization.convert(qat_model)
```

### 3.4 现代 API：TorchAO 一瞥

今天做 LLM 量化，官方推荐 TorchAO，思想完全相同但更贴近大模型场景：

```python
# pip install torchao
from torchao.quantization import quantize_, int8_weight_only

model = get_llm()  # 你的 LLM
# 仅权重量化（W8A16）：权重 INT8，激活仍 FP16/FP4
# 这正是 QLoRA 底座用的思路
quantize_(model, int8_weight_only())
```

---

## 四、围绕该领域展开：量化在 PyTorch 生态中的位置

### 4.1 量化的「三轴坐标系」

理解任何一个量化方案，问三个问题就能定位：

1. **量化什么**？仅权重（W8A16）/ 仅激活 / 权重+激活（W8A8）？
   - 仅权重量化省显存、省带宽，计算仍是浮点——**显存带宽往往是 LLM 推理的真正瓶颈**，所以仅权重量化收益就很大；
   - W8A8 才能用上 INT8 矩阵乘指令，需要校准激活。
2. **什么时候量化**？训练后（PTQ）还是训练中（QAT）？
3. **多细的粒度**？Per-Tensor / Per-Channel / Per-Group（每 64 个数值一组 scale，GPTQ/AWQ 这类 4bit 方案的标配）。

### 4.2 与 LLM 时代的碰撞：离群值难题

Transformer 的激活值中有少数几个通道数值异常大（离群通道，magnitude 可达其他的 100 倍）。Per-Tensor INT8 量化下，这几根「柱子」撑大了 scale，其他所有通道的分辨率被压垮——这是 LLM INT8 量化掉点的头号原因。由此诞生了一批著名工作，它们全部是「轴坐标系」上不同位置的选择：

- **LLM.int8()**（bitsandbytes）：离群通道保留 FP16，其余 INT8，混合精度分治；
- **SmoothQuant**：把激活的离群难度「迁移」到权重上（数学上等价：除以 per-channel scale 再乘回去），让两边都变得好量化；
- **GPTQ**：基于二阶信息（Hessian）逐列求解量化误差最小化，4bit 权重量化的奠基者；
- **AWQ**：观察到「保护 1% 的显著权重」即可，用激活分布引导权重缩放。

### 4.3 量化 × 微调：QLoRA

QLoRA 把量化和参数高效微调组合成了经典搭配：**4bit 量化冻结的基座模型 + NF4 数据类型（信息论最优的 4bit 分布，正态分布的分位点量化）+ BF16 的 LoRA 适配器**。单卡微调 65B 模型因此成为可能。这是「量化不只是部署技术，还能改变训练经济学」的代表作。

### 4.4 量化 × 编译：torch.compile 与 Inductor

第 003 期讲过 torch.compile。今天 PyTorch 的方向是量化算子在 Inductor 中以 **Reference 模式**（用 ATen dequant + 线性运算的组合表达量化算子）参与编译，让编译器自己把量化模式融合（fold scale 进权重、fused QKV 等），而不是手写专用 kernel。也就是说，**量化正在从「独立后处理」变成编译流程的一部分**。

### 4.5 量化 × 硬件：后端（Backend）决定一切

PyTorch 量化代码里那个 `"x86"` / `"qnnpack"` 参数暴露了本质：**量化没有「通用正确」，只有「对特定硬件正确」**。fbgemm（x86 CPU）、qnnpack（ARM）、TensorRT（NVIDIA GPU）、苹果 ANE 各有各的算子支持和数值行为。这也是为什么 PyTorch 用 `QuantizedEngine` 抽象而不是直接暴露 kernel——同一份量化模型描述，落到不同后端生成不同的高效代码。

### 4.6 FP8 与「量化」的边界

严格说，FP8（e4m3/e5m2）是**浮点格式的缩减**而非整数量化——它有指数位，保留动态范围，只是尾数精度大幅降低。H100 之后 FP8 训练和推理的兴起说明：**数值精度领域不是「FP32 vs INT8」二选一，而是一个连续谱系**（FP32 → BF16 → FP8 → INT8 → INT4），每一档都是「范围 vs 精度 vs 硬件效率」的不同折中。

---

## 五、什么时候该用 / 不该用

**该用：**

- 模型要部署到手机/边缘设备，体积和延迟是硬约束；
- 云端推理服务边际成本高，INT8 直接翻倍吞吐（等于服务器费用减半）；
- 微调大模型但显存不够——QLoRA 式的 4bit 底座；
- 模型已经训好、不能重训，但需要快速压缩——PTQ 五分钟出结果。

**不该用（或先别用）：**

- **训练阶段**：量化是推理侧技术（FP8 训练是另一套体系），训练老老实实用 BF16；
- 模型本身就很小（几 MB 的 MobileNet 已经过 NAS 级压缩）；
- 精度处于毫厘之间的高风险场景（医疗、金融风控）且没做严谨的量化后评估——PTQ 的 1~2 个点掉点在某些临界样本上可能是灾难；
- 硬件后端不支持（查清楚目标设备的量化算子覆盖再动手，省得白干）；
- 瓶颈根本不在计算——先做 Profiling（第 014 期），确认是 compute-bound 再量化，否则是白忙。

---

## 六、常见坑

**坑 1：量化前忘了 `.eval()`**
QAT/PTQ 流程中模型含 BN 层时，`prepare` 后模型仍处于 train 模式会导致 BN 统计量持续更新，校准数据把 running stats 搞乱，量化结果莫名变差。**量化前永远 `model.eval()`**。

**坑 2：校准数据与真实数据分布不匹配**
用 ImageNet 图片校准一个将在医学影像上运行的模型——激活分布完全不同，scale 全偏。**校准数据的分布必须贴近真实推理数据**，数量倒在其次（几百条通常够用）。

**坑 3：Conv+BN+ReLU 不做算子融合**
分开执行三个算子意味着三次独立的取整误差，且错过硬件融合优化。`fuse_modules` 看起来像「可选优化」，实际上对 INT8 精度是**必需步骤**——这也是为什么 PyTorch 的静态量化教程永远以 fusion 开头。

**坑 4：以为量化模型和 FP32 输出完全一致**
量化是有损压缩，输出必然有偏差。评估量化模型时要用「量化模型自己跑完整个任务」的端到端指标（accuracy / BLEU / pass rate），而不是对比 logits 的最大相对误差。**在下游任务上验证，不要在中间张量上较劲。**

---

## 七、一句话总结

**量化是在「数值表示的比特数」和「模型精度」之间做工程交换的艺术：PTQ 用校准数据事后适配（便宜快速），QAT 用训练过程事前适应（精度上限高），而 LLM 时代的离群值问题，把这门手艺推向了 SmoothQuant、GPTQ、QLoRA 这样与训练和部署深度耦合的新形态。**

---

<details>
<summary>今日练习</summary>

**练习 1（动手）**：对下面的模型做动态量化，对比量化前后 `model(x)` 的输出差异和参数量：

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 1024),
)
```

**练习 2（思考）**：一个权重分布为 `N(0, 1)` 但混入 0.1% 的 `±50` 离群值的张量，Per-Tensor 对称 INT8 量化会损失多少分辨率？Per-Channel 能解决吗？如果离群值集中在**同一个通道**内部呢？

**练习 3（进阶）**：用 `torch.ao.quantization.get_default_qconfig` 换成 `"qnnpack"` 后端，重新跑 3.2 的静态量化流程，观察校准后各 Observer 记录的 scale 有无变化，思考为什么。

<details>
<summary>参考答案</summary>

**练习 1**：

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 1024),
).eval()

q_model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, torch.qint8)

x = torch.randn(8, 1024)
out_fp32 = model(x)
out_int8 = q_model(x)

print("最大输出偏差:", (out_fp32 - out_int8).abs().max().item())
# 典型结果在 1e-2 ~ 1e-1 量级——有损但可用

fp32_params = sum(p.numel() for p in model.parameters())
print(f"FP32 参数量: {fp32_params:,}（量化后权重存储约缩小 4 倍）")
```

**练习 2**：对称量化下 `scale = 50 / 127 ≈ 0.39`，而主体分布（N(0,1)）的数值间隔只有 0.39——即±1σ 范围内只有约 5 个量化台阶，大部分主体数值会塌缩到同一个整数上，分辨率损失巨大。Per-Channel **不能**解决——离群值和主体在同一个张量/通道内，Per-Channel 的 scale 是按通道划分的，通道内部还是同一个 scale。这正是 GPTQ/AWQ 使用 **Per-Group**（如 group_size=64）粒度的原因：粒度足够细，离群值被隔离在小组内，不会污染全局 scale。

**练习 3**：scale 通常会有细微变化，但多数情况下差别不大。原因：`get_default_qconfig` 的本质区别是选择不同的 Observer 实现（min-max 的搜索策略、是否对称等），qnnpack 与 fbgemm 的默认 observer 配置略有不同；此外随机校准数据本身也会引入波动。真正的差异在于**最终生成的算子内核**面向的指令集不同（x86 VNNI vs ARM NEON），这个差异在 CPU 上实测推理速度时才能体现。
</details>
</details>
