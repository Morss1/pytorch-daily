# PyTorch 每日一课 · 第 011 期

## KV Cache 与 Paged Attention：大模型推理加速的核心密码

> **日期**：2026-09-03
> **难度**：⭐⭐⭐⭐
> **前置知识**：Transformer 的 Self-Attention、PyTorch 张量操作基础、对显存有基本概念
> **预计阅读时间**：15 分钟

---

## 一、这个领域解决什么问题

假设你让一个大语言模型给你写一篇 1000 字的文章。模型不是一口气把 1000 字全吐出来的，而是一个 token 一个 token 地生成。每生成一个 token，都要做一次前向计算——而 Self-Attention 的天性决定了：**生成第 N 个 token 时，需要看到前面所有 N-1 个 token 的 Key 和 Value 向量**。

最朴素的做法是：每生成一个新 token，就把整个序列（旧的 + 新的）重新跑一遍 Attention。这意味着生成第 N 个 token 要重算 N-1 个旧 token 的 K/V，整个生成过程是 **O(N²) 的重复计算**。1000 个 token 的文章，前面那些 token 的 K/V 被白白重算了上千次。

第二个问题更致命：**显存**。假设一个 7B 模型、32 层 Transformer、每层 32 个注意力头、每个头的维度 128，序列长度 4096，KV 用 FP16 存储。算一下：

- 每个 token 每层需要存 K 和 V 各 128×32 = 4096 个数，共 2×4096 = 8192 个 fp16 数 = **16 KB**
- 32 层 → 每个 token 共 **512 KB**
- 4096 个 token → **2 GB**，仅仅是一个请求的 KV！

如果并发 10 个用户，20 GB 显存就没了——而模型权重才 14 GB。**KV Cache 的显存管理，直接决定了一个推理服务能同时服务多少人、能支持多长的上下文。**

这个领域要解决的就是两件事：
1. **避免重复计算** → KV Cache（缓存机制）
2. **高效管理缓存显存** → Paged Attention（分页机制）

---

## 二、核心思想：用类比讲清楚

### 2.1 KV Cache = 考试时允许翻笔记

想象你在做听写考试，老师一个字一个字地念。如果没有笔记本，每写一个新字你都要在脑子里把之前听过的所有字重新回忆一遍——这就是"无缓存"的自回归生成。

**KV Cache 就是允许你记笔记**：老师每念一个字（生成一个 token），你把它的"摘要"（Key 和 Value 向量）记在本子上。写下一个字时只需要看笔记本，不用重新听一遍。

用一张图对比两种方式：

```
无缓存（每步重算全部）：
  第1步: [Q1] × [K1]        → 1 次注意计算
  第2步: [Q2] × [K1,K2]      → 重新算 K1,K2
  第3步: [Q3] × [K1..K3]     → 重新算 K1,K2,K3
  总计算量: O(N²)

有 KV Cache（只算新增部分）：
  第1步: [Q1] × [K1]         算 K1,V1 → 存入 cache
  第2步: [Q2] × [K1,K2]      只算 K2,V2 → 追加进 cache
  第3步: [Q3] × [K1..K3]     只算 K3,V3 → 追加进 cache
  总计算量: O(N)
```

**为什么只缓存 K 和 V，不缓存 Q？** 因为生成第 N 个 token 时，Attention 的 Query 只有第 N 个 token 自己（每个新 token 只"询问"历史），而 Key 和 Value 需要全部历史 token 来"回答"。这正是因果掩码（causal mask）的结构决定的。

### 2.2 Paged Attention = 操作系统的虚拟内存

KV Cache 解决了计算问题，但传统实现有严重的显存浪费。传统做法是为每个请求**预分配一段连续显存**，按最大序列长度（比如 2048）预留。这带来两个问题：

1. **内部碎片**：用户实际只生成了 200 个 token，但预留了 2048 的空间，90% 浪费了。
2. **外部碎片**：连续大块分配/释放反复发生，显存被切得七零八落，大请求明明有总量够的空闲显存却放不进去。

这和早期操作系统的内存管理问题一模一样。操作系统的答案是**分页（Paging）**：物理内存切成固定大小的页框，程序的逻辑地址通过页表映射到离散的页框，按需分配。

**Paged Attention 把这套思想搬到了 GPU 上**：

- KV Cache 被切成固定大小的 **block**（通常 16 个 token 一块）
- 每个请求维护一张 **block table**：逻辑块号 → 物理 block 的地址
- 序列变长就再分配一个 block，不需要连续
- 多个请求如果共享相同的前缀（比如同一个 system prompt），物理 block 可以**共享**——写时复制（copy-on-write）

vLLM 的论文报告，传统实现的 KV Cache 实际利用率只有 20%~40%，而 Paged Attention 能做到 **96% 以上**，等效吞吐提升 2~4 倍。这就是为什么它成了几乎所有现代推理引擎的标配。

---

## 三、在 PyTorch 中怎么用

### 3.1 手写一个最小 KV Cache，看清它的本质

KV Cache 并不神秘，用 30 行 PyTorch 代码就能复现它的核心逻辑：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# 模型参数（一个"单层迷你 Transformer"的注意力部分）
d_model = 64       # 模型隐藏维度
n_heads = 4        # 注意力头数
d_head = d_model // n_heads  # 每个头的维度 = 16

# 投影矩阵（真实模型中是 nn.Linear）
W_q = torch.randn(d_model, d_model) * 0.1
W_k = torch.randn(d_model, d_model) * 0.1
W_v = torch.randn(d_model, d_model) * 0.1

class KVCache:
    """最小 KV Cache：就是两个沿序列维度不断拼接的张量"""
    def __init__(self, n_layers=1):
        self.k = [None]  # 每层一个缓存，形状 [batch, heads, seq, d_head]
        self.v = [None]

    def update(self, layer_idx, k_new, v_new):
        """新算出来的 k/v 追加进缓存，返回包含全部历史的完整 k/v"""
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            # 核心操作：沿 seq 维度拼接，旧数据不重算
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k_new], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v_new], dim=2)
        return self.k[layer_idx], self.v[layer_idx]

def attention(x, W, n_heads):
    """投影并拆成多头: [batch, seq, d_model] -> [batch, heads, seq, d_head]"""
    b, s, _ = x.shape
    out = x @ W
    return out.view(b, s, n_heads, d_head).transpose(1, 2)

# ============ 自回归生成：预填充 + 逐 token 解码 ============
cache = KVCache()
tokens = torch.randn(1, 5, d_model)   # 假装这是 prompt 的 5 个 token 的隐藏态

# 阶段一：prefill（预填充）—— 一次性处理整个 prompt，填充缓存
q = attention(tokens, W_q, n_heads)
k = attention(tokens, W_k, n_heads)
v = attention(tokens, W_v, n_heads)
k_all, v_all = cache.update(0, k, v)   # 缓存了 prompt 全部的 K/V

# 阶段二：decode（解码）—— 每步只算 1 个新 token
for step in range(3):
    new_token = torch.randn(1, 1, d_model)  # 新 token 的隐藏态
    q_new = attention(new_token, W_q, n_heads)          # 只算新 token 的 Q
    k_new = attention(new_token, W_k, n_heads)          # 只算新 token 的 K
    v_new = attention(new_token, W_v, n_heads)          # 只算新 token 的 V
    k_all, v_all = cache.update(0, k_new, v_new)        # 追加进缓存

    # Q 只查自己，K/V 查全部历史 —— 因果注意力的天然结构
    scores = q_new @ k_all.transpose(-2, -1) / (d_head ** 0.5)
    attn = F.softmax(scores, dim=-1)
    out = attn @ v_all   # [1, heads, 1, d_head]
    print(f"第 {step+1} 步: 注意力权重长度 = {attn.shape[-1]}（= 已见过的 token 数）")
```

跑一下你会发现：注意力权重的长度每步 +1，但**旧的 K/V 从未重算**——这就是 KV Cache 的全部秘密。HuggingFace 的 `generate()` 之所以快，底层就是在做这件事（`DynamicCache`）。

### 3.2 用 PyTorch 原生组件开关 KV Cache

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

inputs = tokenizer("KV Cache 的本质是", return_tensors="pt")

# use_cache=True（默认）：prefill 后缓存 KV，解码阶段只算新 token
out_fast = model.generate(**inputs, max_new_tokens=50, use_cache=True)

# use_cache=False：每步重算整个序列，长文本下会明显变慢
out_slow = model.generate(**inputs, max_new_tokens=50, use_cache=False)

print(tokenizer.decode(out_fast[0]))
```

### 3.3 估算一个模型的 KV Cache 显存

部署前必做的算术，值得背下来：

```python
def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, dtype_bytes=2):
    """计算单个请求的 KV Cache 显存（字节）
    
    n_kv_heads: 注意 KV 头数！用了 GQA 的模型 KV 头数远少于 Q 头数
    每层每个 token: K 和 V 各 n_kv_heads * d_head 个数
    """
    per_token_per_layer = 2 * n_kv_heads * d_head * dtype_bytes
    return per_token_per_layer * n_layers * seq_len

# Qwen2.5-7B：28 层，4 个 KV 头（GQA），d_head=128，FP16
b = kv_cache_bytes(n_layers=28, n_kv_heads=4, d_head=128, seq_len=4096)
print(f"{b / 1024**3:.2f} GB / 请求")   # 约 0.055 GB —— GQA 功不可没

# 对比没有 GQA 的模型（32 个 KV 头）：直接 ×8
```

---

## 四、围绕该领域展开：它在整个生态中的位置

KV Cache 与 Paged Attention 不是孤立的技术点，它连接了推理加速的半壁江山。理解这些关联，才算真正进入这个领域。

### 4.1 GQA / MQA：从源头缩小 KV Cache

- **MHA**（多头注意力）：32 个 Q 头配 32 个 KV 头 → KV Cache 最大
- **MQA**（Multi-Query Attention）：32 个 Q 头共享 **1 个** KV 头 → KV Cache 缩小 32 倍，但精度略降
- **GQA**（Grouped-Query Attention）：折中方案，比如 32 个 Q 头分 8 组，每组共享 1 个 KV 头 → 缩小 4 倍，精度几乎无损

这是 Llama 2/3、Qwen、Mistral 等主流模型的标配。**模型结构层面减少要缓存的东西，比缓存管理本身更根本。**

### 4.2 Flash Attention：与 KV Cache 的共生关系

Flash Attention（本系列第 005 期）解决的是"Attention 计算中的显存读写瓶颈"，它把 QKV 留在 SRAM 里分块计算，不落显存。推理时两者各管一段：

- **Prefill 阶段**（长 prompt 一次性算）：受益于 Flash Attention 的计算加速
- **Decode 阶段**（每步 1 个 token）：受益于 KV Cache 的免重算

Paged Attention 的 block 结构也借鉴了 Flash Attention 的分块思想——把不连续的 KV block 当作 tile 来加载，在 SRAM 里完成注意力计算。

### 4.3 Continuous Batching：吞吐的关键搭档

传统静态批处理要等一批请求里最长的生成完才能换下一批，短请求被迫陪跑。**Continuous Batching**（连续批处理）允许每个请求"生成完就走，新请求随时插队"。

它能成立的前提恰恰是 Paged Attention：因为 KV Cache 是按 block 离散分配的，新请求的缓存可以塞进任何空闲 block，不需要预留整块连续显存。**vLLM 的吞吐神话 = Paged Attention + Continuous Batching 两件套。**

### 4.4 前缀缓存（Prefix Caching）：系统 prompt 的免费午餐

聊天服务里，成千上万的请求共享同一段 system prompt（可能几千 token）。这段 prompt 的 KV Cache 对每个请求完全相同，为什么算 N 遍？

Paged Attention 的 block 共享机制天然支持：相同前缀的 KV block 直接复用（类似操作系统的页共享 + 写时复制），新请求命中前缀缓存时，prefill 几乎免费。这对多轮对话场景（每轮都要重算历史？不，历史就是前缀）是数量级的加速。

### 4.5 训练侧的对照：为什么训练时不用 KV Cache？

一个常见的困惑：既然 KV Cache 这么好，训练时为什么不用？因为训练时**所有 token 的 K/V 本来就要一次性算出来**（并行处理整个序列，没有"逐步生成"的过程），Attention 一次前向就完成了——缓存无处发力。**KV Cache 是"自回归生成"这个场景特有的优化**。而梯度检查点（第 009 期）反而是训练侧"显存不够"的解法，两者是镜像问题：一个为了推理快，一个为了训练省。

### 4.6 相关推理引擎版图

| 引擎 | KV Cache 策略 | 特色 |
|------|--------------|------|
| vLLM | Paged Attention（首创） | 高吞吐服务的事实标准 |
| TensorRT-LLM | 分页 + 量化 KV（FP8 KV Cache） | NVIDIA 官方，极致单卡性能 |
| SGLang | RadixAttention（前缀树缓存） | 多轮对话/结构化生成极快 |
| llama.cpp | 统一 KV 量化、mmap 权重 | 消费级硬件/端侧 |

---

## 五、什么时候该用 / 不该用

**该用（几乎总是）：**
- 一切 LLM 自回归生成场景，`use_cache=True` 是默认正确选项
- 部署推理服务、关心吞吐和并发数时，选带 Paged Attention 的引擎
- 多轮对话或固定 system prompt 重的业务，开启前缀缓存

**不该用 / 需要警惕：**
- **训练或全序列前向**（如给整个序列算 loss）：KV Cache 无用，直接一次前向
- **需要完整注意力权重做分析**（如可视化 attention map）：某些分页实现不返回中间权重
- **beam search 大宽度**：每条候选序列各有一份 KV Cache，beam=8 显存 ×8，需要专门的实现来复用共同前缀
- **显存极度受限的端侧**：考虑 KV 量化（FP8/INT4）或 StreamingLLM 这类"滑动窗口 + 注意力锚点"方案，丢弃远处的 KV

---

## 六、常见坑（4 个）

**坑 1：忘了 `use_cache` 和模型结构的关系**
用了 GQA 的模型，KV 头数 ≠ Q 头数。自己写推理代码时如果按 Q 头数分配 KV Cache，显存直接多算几倍，或者形状对不上报错。永远用 `model.config.num_key_value_heads` 查真实值。

**坑 2：把 KV Cache 显存算进"模型大小"**
"7B 模型 FP16 只要 14 GB，为什么 24 GB 卡跑不动长上下文 + 并发？"——因为 KV Cache 是随序列长度和并发数**线性增长**的额外开销。部署前务必用第 3.3 节的公式估算：`总显存 ≈ 权重 + KV Cache × 并发数 + 激活值`。

**坑 3：以为 `use_cache=False` 更省显存**
恰恰相反。没有 KV Cache 时，每步都对全序列做前向，激活值反而更大，而且更慢。`use_cache=False` 唯一的用途是训练或某些需要重算的场景。

**坑 4：torch.compile 与 KV Cache 的动态形状冲突**
解码阶段每生成一个 token，Cache 的 seq 维度都 +1——这是典型的动态形状，会让 Dynamo 反复触发 graph break 或重编译（见第 003 期）。推理引擎的做法是给 Cache 预分配到最大长度（用 padding + mask 维持静态形状），或干脆把 decode 路径交给 CUDA Graph（第 001 期）捕获。**KV Cache 的"动态增长"本质，正是推理场景对动态形状支持提出严苛要求的根源。**

---

## 七、一句话总结

**KV Cache 用"空间换时间"消除了自回归生成的 O(N²) 重复计算，Paged Attention 用操作系统的分页思想把缓存显存利用率从 20% 拉到 96%——两者合起来，构成了大模型推理引擎吞吐能力的地基。**

---

<details>
<summary>今日练习</summary>

**练习 1（估算题）**：你要部署一个模型，参数：40 层 Transformer，GQA 8 个 KV 头，d_head=128，权重 BF16 占 15 GB，GPU 显存 24 GB。服务需要支持并发 16 个请求，每个请求上下文最长 8192。请估算 KV Cache 需求，并判断是否可行；不可行的话给出两条改进路径。

**练习 2（代码题）**：修改第 3.1 节的 `KVCache` 类，加上 `max_seq_len` 参数，用 `torch.zeros` 预分配缓存、用指针 `seq_idx` 追踪写入位置（而不是每步 `torch.cat`）。思考：为什么真实推理引擎都这样做，而不是用 `torch.cat`？

---

**参考答案**

**练习 1**：
- 每 token 每层 KV = 2 × 8 × 128 × 2 字节 = 4096 字节
- 每请求：4096 × 40 层 × 8192 token = 4096 × 40 × 8192 = 1,342,177,280 字节 ≈ **1.25 GB**
- 16 并发 → **20 GB** KV Cache
- 总需求 ≈ 15（权重）+ 20（KV）+ 激活/开销 ≈ 36+ GB > 24 GB，**不可行**。
- 改进路径：
  1. **KV 量化**：BF16 → FP8 直接减半（10 GB），勉强可行；INT4 再减半
  2. **限制并发或上下文**：并发降到 8（KV 10 GB，总 ~26 GB 仍紧），或上下文限到 4096（KV 10 GB）——这就是工程上常见的"吞吐 vs 长度"权衡
  3. 换更大的卡或用张量并行把权重和 KV 分摊到多卡（也可接受）

**练习 2**：核心改动：

```python
class PagedKVCache:
    def __init__(self, max_seq_len, n_layers, n_kv_heads, d_head, dtype=torch.float16):
        # 一次性预分配最大长度，形状从此固定不变
        self.k = torch.zeros(1, n_layers, n_kv_heads, max_seq_len, d_head, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.seq_idx = 0  # 写入指针

    def update(self, layer_idx, k_new, v_new):
        s = k_new.shape[2]                    # 新增的 token 数
        # 原地写入，零拷贝、零分配
        self.k[:, layer_idx, :, self.seq_idx:self.seq_idx + s, :] = k_new
        self.v[:, layer_idx, :, self.seq_idx:self.seq_idx + s, :] = v_new
        self.seq_idx += s
        # 读取时返回 [0, seq_idx) 的视图（view），不产生拷贝
        return (self.k[:, layer_idx, :, :self.seq_idx],
                self.v[:, layer_idx, :, :self.seq_idx])
```

为什么真实引擎不用 `torch.cat`：① `cat` 每步都分配一块新显存并拷贝全部旧数据，产生 O(N²) 的总拷贝量和大量临时分配，拖垮 Caching Allocator（见第 004 期）；② 张量形状每步变化，无法被 CUDA Graph 捕获、无法被 `torch.compile` 静态优化；③ 预分配 + 指针的做法让读写地址完全静态，kernel 可以精确调度。**"预分配 + 原地写 + 视图读"是 GPU 编程中处理增长型缓冲区的通用范式。**

</details>
