# PyTorch 每日一课 · 第 040 期

## 算子融合：把中间结果留在寄存器里

| | |
|---|---|
| 日期 | 2026-10-20 |
| 难度 | ⭐⭐⭐⭐ |
| 前置知识 | 会用 `torch.compile`；知道 ATen 算子调用要过一次 dispatch（第 039 期）；有 roofline / 算术强度的直觉 |
| 预计阅读 | 25 分钟 |
| 关联 | 039（一个算子调用的入口税）、003（Dynamo 抓图 → Inductor 出 kernel）、005（FlashAttention 是融合的极致形态）、037（先算上限，再读实测） |

---

## 1. 这个领域解决什么问题

先摆一笔实测账。同一台机器（Apple Silicon，4 线程，64 MB 的 fp32 张量，16.7M 个元素）：

- 纯拷贝 `y.copy_(x)`：**1.04 ms**，合 **129 GB/s**
- eager 跑一个 `x + 1`：**1.12 ms**，合 **120 GB/s**

一个逐元素加法的算术量是 16.7M 次加法，对一颗现代 CPU 来说基本等于零。它花掉的 1.1 ms 全部用在「读 64 MB、写 64 MB」这件事上。

**这就是这一期全部动机的来源：这些 op 的耗时和它算多少无关，只和它搬多少有关。**

再往下推一步。eager 跑 24 个链式 pointwise，实测 27.0 ms ≈ 24 × 1.12 ms —— 完美线性。每个算子都独立地「完整读一遍、完整写一遍」，中间结果必须落回内存。

第 039 期量过，一次 `a + b` 从 Python 穿过 C++ 边界、查完派发表是 **0.36 µs**；而这里一次 64 MB 的往返是 **1.1 ms**。**差三个数量级**。

所以「融合是为了省 dispatch 开销」这句话不算错，但它不是主要收益。主要收益是那趟内存。融合真正干掉的是**中间张量这个实体**：图上的边不再是一个 `buf1`，而是一个寄存器里的临时变量。

## 2. 核心思想

车间流水线的类比。eager 的做法是：每道工序做完，把半成品装箱、运回仓库、下一道工序再从仓库取出来。融合的做法是：三道工序摆在同一张工作台上，半成品从一只手直接交到另一只手。

翻译成 IR：一个 fusion group 就是一个**循环体**。外面一层 `for` 扫过元素，里面按顺序把组里所有算子的活干完，只从内存读一次输入、只写一次输出。

设一条链上有 $m$ 个逐元素算子，张量有 $N$ 个 fp32 元素，一个缓冲区是 $4N$ 字节：

$$
\text{eager}: 2m \cdot 4N = 8mN \ \text{字节}
\qquad\text{vs}\qquad
\text{fused}: 2 \cdot 4N = 8N \ \text{字节}
$$

**加速比的上限就是 $m$。** 这个上限是后面所有结论的坐标系 —— 和第 037 期讲的一样，先算清楚「不融合时理论最好情况长什么样」，实测离它多远才有意义。

融合有三种形态，它们在同一次编译里都会出现：

| 形态 | 长什么样 | 例子 |
|---|---|---|
| 垂直（vertical） | producer 的输出直接被 consumer 的循环体消费 | `(x*2+1).relu()` |
| 水平（horizontal） | 两条**互不依赖**的分支并进同一个循环，一趟扫完各写各的 | `x.sum()` 与 `x.max()` 一起算 |
| epilogue / prologue | 归约或模板算子的前后各挂一串 pointwise | `(x*2).sum() / 3` |

## 3. 在 PyTorch 中怎么用

结论先行：**PyTorch 没有 `torch.fuse()`**。融合是 Inductor 在编译期自己做的决定，你能做的是三件事 —— 别打断图、别指望单算子、学会看产物。

顺便说一句 eager 世界里的「融合」：它不叫融合，叫**预融算子**。`addmm` 是 mul + add + mm 三条链写死成一个 ATen op；`baddbmm` 同理；`layer_norm` 把求均值、方差、归一化、affine 一整个包成一个 op；`scaled_dot_product_attention` 把整条 attention 融成一个 op（第 038 期讲的就是它内部怎么选实现）。这些都是有人手工把链焊死的结果。想自己造一条链的融合，路只有两条：`torch.compile`，或者自己写 Triton / C++ kernel（第 008 期）。

### 3.1 先看它到底生成了什么

```python
import glob, re, tempfile, torch
import torch._inductor.config as icfg
torch.set_num_threads(4)                    # 定死线程数，生成的代码里会写死 num_threads(4)

def gen_code(fn, *args):
    """编译一次，从 debug trace 里取回生成的 output_code.py（C++ kernel 就嵌在里面）"""
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    icfg.force_disable_caches = True        # 关掉磁盘缓存，保证每次都真的重编一遍
    torch.compile(fn, dynamic=False)(*args)
    return open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()

def show(s, w=100):
    s = s.strip()
    return s if len(s) <= w else s[:w - 1] + "…"

x = torch.randn(1 << 20)                    # 1M 个 float32 = 4 MB
code = gen_code(lambda t: (t * 2 + 1).relu().sigmoid(), x)

print("4 个 ATen 算子编译出的 kernel：")
for name in re.findall(r"^(\w+) = async_compile\.cpp_pybinding", code, re.M):
    print("  " + name)

src = re.search(r"r'''(.*?)'''", code, re.S).group(1)
keep = re.compile(r'^\s*(extern "C"|#pragma omp|for\(int64_t|auto tmp\d+ =|.*\.store\(|int tid)')
print(f"\n整个 kernel 共 {len(src.splitlines())} 行 C++，骨架如下：")
for line in src.splitlines():
    if keep.match(line):
        print("  " + show(line))
```

```text
4 个 ATen 算子编译出的 kernel：
  cpp_fused_add_mul_relu_sigmoid_0

整个 kernel 共 35 行 C++，骨架如下：
  extern "C"  void  kernel(const float* in_ptr0,
  #pragma omp parallel num_threads(4)
  int tid = omp_get_thread_num();
  #pragma omp for
  for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(1048576LL); x0+=static_cast<int64…
  auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64…
  auto tmp1 = static_cast<float>(2.0);
  auto tmp2 = at::vec::Vectorized<float>(tmp1);
  auto tmp3 = tmp0 * tmp2;
  auto tmp4 = static_cast<float>(1.0);
  auto tmp5 = at::vec::Vectorized<float>(tmp4);
  auto tmp6 = tmp3 + tmp5;
  auto tmp7 = at::vec::clamp_min(tmp6, decltype(tmp6)(0));
  auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
  tmp8.store(out_ptr0 + static_cast<int64_t>(x0));
```

四行代码读出来的信息比想象中多：

1. **4 个 ATen 算子 → 1 个 kernel**，而且 kernel 名字直接把融进去的算子列出来了：`cpp_fused_add_mul_relu_sigmoid_0`。这是最省事的观测手段 —— 数一眼名字就知道融了几个。
2. 整个 kernel **35 行 C++**，一个 OpenMP 并行区，循环步长是 4（这台机器是 ARM NEON，`Vectorized<float>` 一次 4 个 float）。
3. 循环体里 `tmp3 → tmp6 → tmp7 → tmp8` 就是 mul → add → relu → sigmoid，**中间一次 `store` 都没有**。
4. 唯一一次 `.store()` 在最后一行。整段代码只 load 一次、store 一次 —— 这就是第 2 节那个公式的物理实现。

一个容易上钩的细节：**名字里的算子不是按执行顺序排的，是去重后按字母序排的**。源码在 `torch/_inductor/utils.py` 的 `get_fused_kernel_name()`：

```python
sources = sorted(OrderedSet(sources))          # 排序 + 去重
name = "_".join(["fused"] + sources)
```

验证一下：`((t*2+1).relu().sigmoid()*3-1).relu()` 是 7 个算子（mul 和 relu 各出现两次），生成的名字是 `cpp_fused_add_mul_relu_sigmoid_sub_0` —— 只有 5 个名字，而且 add 排在最前面，尽管它是第二个执行的。所以别拿名字去数算子、更别拿它读执行顺序；要数就数循环体里的 `auto tmp`。

（顺带说清两个后端的命名差异，源码在 `codegen/cpp.py:6036` 与 `codegen/triton.py:8206`：CPU 是 `["cpp", fused_name, 序号]`，CUDA 是 `["triton", 三字母类别, fused_name, 序号]`，所以 CUDA 上长这样：`triton_poi_fused_add_mul_0`。）

### 3.2 融合分两个阶段，别混为一谈

很多人第一次打开 `ir_pre_fusion.txt` 会以为「融合前应该是一堆算子」，结果发现循环体里已经排了一串。原因是融合真的分两层：

```python
import glob, re, tempfile, torch
import torch._inductor.config as icfg
torch.set_num_threads(4)

d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
icfg.trace.enabled = True
icfg.trace.debug_dir = d
icfg.force_disable_caches = True

x = torch.randn(4096)
def f(t):
    a = t * 2                      # 分支 1：一条 pointwise 链
    b = t + 1                      # 分支 2：先加、再整体求和
    return a.relu().sigmoid(), b.sum()
torch.compile(f, dynamic=False)(x)

def deps(chunk, key):
    m = re.search(key + r" = \[(.*?)\]", chunk, re.S)
    return re.findall(r"MemoryDep\('(\w+)'", m.group(1)) if m else []

def nodes(fname):
    path = glob.glob(f"{d}/**/{fname}", recursive=True)[0]
    lines = open(path).read().splitlines()
    tops = []
    for i, ln in enumerate(lines):
        m = re.match(r"^(op\d+(?:_op\d+)*): (\w+)(\(.*)?$", ln)
        if not m:
            continue
        prev = next((lines[j].strip() for j in range(i - 1, -1, -1) if lines[j].strip()), "")
        if prev.endswith("="):
            continue                       # 这是 FusedSchedulerNode 里嵌着的子节点
        chunk = "\n".join(lines[i:i + 40])
        body = re.search(r"def body\(self, ops\):(.*?)(?=\n\S|\Z)", chunk, re.S)
        seq = []
        for op, arg in re.findall(r"ops\.(\w+)\(([^)]*)\)", body.group(1) if body else ""):
            if op == "reduction":
                seq.append("reduction(" + re.search(r"'(sum|max|min)'", arg).group(1) + ")")
            elif op not in {"load", "store", "constant", "get_index", "store_reduction"}:
                seq.append(op)
        n_sub = (m.group(3) or "").count(",") + 1 if "Fused" in m.group(2) else 0
        tops.append((m.group(1), m.group(2), n_sub,
                     deps(chunk, "(?<!un)met_dependencies"), deps(chunk, "writes"), seq))
    return tops

for label, fname in (("融合前", "ir_pre_fusion.txt"), ("融合后", "ir_post_fusion.txt")):
    ts = nodes(fname)
    print(f"{label}：{len(ts)} 个顶层节点")
    for name, kind, n_sub, reads, writes, seq in ts:
        extra = f"（{n_sub} 个子节点）" if n_sub else ""
        print(f"  {name}  {kind}{extra}  读 {reads}  写 {writes}")
        if seq and not n_sub:
            print(f"      循环体：{' → '.join(seq)}")
```

```text
融合前：2 个顶层节点
  op0  SchedulerNode  读 ['arg0_1']  写 ['buf0']
      循环体：mul → relu → sigmoid
  op1  SchedulerNode  读 ['arg0_1']  写 ['buf1']
      循环体：add → reduction(sum)
融合后：1 个顶层节点
  op0_op1  FusedSchedulerNode（2 个子节点）  读 ['arg0_1']  写 ['buf0', 'buf1']
```

这段输出里有两个必须分开的事实：

**阶段一，lowering 期的 loop body 内联。** 看「融合前」的 `op0`：它的循环体里已经写着 `mul → relu → sigmoid` 三条了。也就是说这条 pointwise 链在进入 scheduler **之前**就已经是同一个循环体，`a = t*2` 从来没有被单独物化成一个张量。

**阶段二，scheduler 期的 node 融合。** 看「融合后」：只剩 1 个 `FusedSchedulerNode`，含 2 个子节点，写 `buf0` 和 `buf1`。两个子节点都只读 `arg0_1`、互不依赖 —— 这是**水平融合**：一趟循环扫过去，顺手把两个分支的结果各写一份。

所以 `ir_pre_fusion.txt` 里的 "pre" 指的是「scheduler 融合之前」，不是「什么都没融之前」。这是这个文件名最容易读错的地方。

### 3.3 它为什么拒绝：把决策日志接出来

Inductor 的融合决策有一个专门的日志通道 `torch._inductor.scheduler` 的 `fusion` artifact，默认是关的。接出来之后，它会把「候选了谁」「拒了谁、为什么」「接受了谁」全打出来：

```python
import logging, re, torch
import torch._inductor.config as icfg
from torch._inductor import scheduler as sch
torch.set_num_threads(4)
icfg.force_disable_caches = True

class Cap(logging.Handler):
    """把 Inductor 的融合决策日志接出来"""
    def __init__(self):
        super().__init__()
        self.msgs = []
    def emit(self, rec):
        self.msgs.append(rec.getMessage())

cap = Cap()
sch.fusion_log.addHandler(cap)
sch.fusion_log.setLevel(logging.DEBUG)
sch.fusion_log.propagate = False

x = torch.randn(4096)
def f(t):
    a = torch.softmax(t, -1)              # softmax 会被拆成 4 个节点
    b = a.view(64, 64) @ a.view(64, 64)   # 矩阵乘是外部调用，不参与融合
    return b.sum() + a.relu()
torch.compile(f, dynamic=False)(x)

msgs = cap.msgs
first = next(i for i, m in enumerate(msgs) if m.startswith("fuse_nodes_once"))
cands = []
for m in msgs[first + 1:]:
    if not m.startswith("  "):
        break
    nm = re.search(r"name='(\w+)'", m)
    if not nm:
        continue
    kind = re.search(r"'\), (\w+)\(", m)
    org = re.search(r"origins=OrderedSet\(\[([^\]]*)\]\)", m)
    cands.append((nm.group(1), kind.group(1) if kind else "ExternKernel",
                  org.group(1) if org else "-"))

print(f"第 1 轮候选：{len(cands)} 个 SchedulerNode")
for name, kind, org in cands:
    print(f"  {name}  {kind:<10s} 来自 {org}")

seen, rej = set(), []
for m in msgs:
    if m.startswith("cannot fuse") and m not in seen:
        seen.add(m); rej.append(m)
print("\n被拒绝的配对（附原因）：")
for m in rej:
    print("  " + m)

print("\n被接受的融合：")
for m in msgs:
    if m.startswith("fusing "):
        print("  " + m)

last = [m for m in msgs if m.startswith("completed fusion round")][-1]
print("\n" + last)
```

```text
第 1 轮候选：7 个 SchedulerNode
  op0  Reduction  来自 amax
  op1  Pointwise  来自 exp, sub
  op2  Reduction  来自 sum_1
  op3  Pointwise  来自 div
  op4  ExternKernel 来自 -
  op5  Reduction  来自 sum_2
  op6  Pointwise  来自 add, relu

被拒绝的配对（附原因）：
  cannot fuse op1 with op3: intermediate nodes between node1 & node2
  cannot fuse op3 with op6: intermediate nodes between node1 & node2

被接受的融合：
  fusing op1 with op2

completed fusion round (2/10): fused 6 nodes into 6 nodes
```

三个信息：

1. **`origins=OrderedSet([...])` 是从 ATen 算子到 scheduler 节点的映射表**。`torch.softmax` 一个调用被拆成 4 个节点：`amax`（找最大值）→ `exp, sub`（减最大值再取指数）→ `sum_1`（分母）→ `div`（归一）。这就是「一个高层算子 = 一小段计算图」的直接证据。
2. **matrix multiply 是 `ExternKernel`**，它有节点但不参与融合 —— 它不是一个能塞进循环体的表达式，而是一次对外部库的调用。
3. **拒绝理由是 `intermediate nodes between node1 & node2`**：`op1`（exp/sub）想和 `op3`（div）融，中间隔着 `op2`（sum）和 `op4`（mm），不行。唯一被接受的是 `op1 + op2`：exp 紧跟着喂给 sum，纯粹的垂直融合。

这条日志的源码位置在 `torch/_inductor/scheduler.py`：`WhyNoFuse` 这个 dataclass 的 `__call__` 就是「记下原因并打到 fusion_log」。拒绝原因的完整清单有三十来条（`node2 is extern or nop`、`node1 and node2 uses different buf layouts`、`device mismatch`、`prologue fusion only supported for pointwise nodes`……），上面这两条只是最常撞的。**以上均为在本机 `torch 2.14.0` 安装包里读到的 Python 源码，不是文档。**

## 4. 围绕这个领域展开

### 4.1 谁被融、谁被挡：一张边界表

CPU 上有一件事和直觉相反：**「一个算子一个 kernel」几乎从来不是真的**。下面这张表是实测（同一个 4096 元素张量，逐个编译后数 `async_compile` 出来的 kernel 个数和外部调用）：

```python
import glob, re, tempfile, torch
import torch._inductor.config as icfg
torch.set_num_threads(4)
icfg.force_disable_caches = True

def compile_and_inspect(fn, *args):
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    torch.compile(fn, dynamic=False)(*args)
    code = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    kernels = re.findall(r"^(\w+) = async_compile\.(?:cpp_pybinding|triton)", code, re.M)
    extern = re.findall(r"(extern_kernels\.\w+|aten\.\w+\.\w+)\(", code)
    return kernels, sorted(set(extern))

t = torch.randn(4096)
cases = [
    ("t*2                      1 个 pointwise", lambda x: x * 2),
    ("(t*2+1).relu().sigmoid() 4 个 pointwise", lambda x: (x * 2 + 1).relu().sigmoid()),
    ("(t*2).sum()              pointwise→归一约", lambda x: (x * 2).sum()),
    ("(t.sum()/3).relu()       归一约→pointwise", lambda x: (x.sum() / 3).relu()),
    ("(t.sum(), t.max())       两个互不依赖的归约", lambda x: (x.sum(), x.max())),
    ("t.half()+1               中间换了 dtype", lambda x: x.half() + 1),
    ("t.view(64,64)@t.view(64,64)  外部 GEMM", lambda x: x.view(64, 64) @ x.view(64, 64)),
    ("(t*2)@t                  pointwise→外部 GEMM", lambda x: (x * 2).view(64, 64) @ x.view(64, 64)),
    ("softmax→mm→sum           被 GEMM 切成两段", lambda x: (torch.softmax(x, -1).view(64, 64) @ x.view(64, 64)).sum()),
    ("torch.sort(t)[0]", lambda x: torch.sort(x)[0]),
    ("t.cumsum(0)", lambda x: x.cumsum(0)),
    ("两个分支共享中间结果", lambda x: (lambda a: (a + 1, a.sum()))(x * 2)),
]
for name, fn in cases:
    ks, ext = compile_and_inspect(fn, t)
    print(f"{name}")
    show = ks if len(ks) <= 2 else ks[:2] + ['...']
    print(f"    融合体 {len(ks)} 个 {show if show else '—'}  外部调用 {ext if ext else '无'}")
```

```text
t*2                      1 个 pointwise
    融合体 1 个 ['cpp_fused_mul_0']  外部调用 无
(t*2+1).relu().sigmoid() 4 个 pointwise
    融合体 1 个 ['cpp_fused_add_mul_relu_sigmoid_0']  外部调用 无
(t*2).sum()              pointwise→归一约
    融合体 1 个 ['cpp_fused_mul_sum_0']  外部调用 无
(t.sum()/3).relu()       归一约→pointwise
    融合体 1 个 ['cpp_fused_div_relu_sum_0']  外部调用 无
(t.sum(), t.max())       两个互不依赖的归约
    融合体 1 个 ['cpp_fused_max_sum_0']  外部调用 无
t.half()+1               中间换了 dtype
    融合体 1 个 ['cpp_fused__to_copy_add_0']  外部调用 无
t.view(64,64)@t.view(64,64)  外部 GEMM
    融合体 0 个 —  外部调用 ['extern_kernels.mm']
(t*2)@t                  pointwise→外部 GEMM
    融合体 1 个 ['cpp_fused_mul_0']  外部调用 ['extern_kernels.mm']
softmax→mm→sum           被 GEMM 切成两段
    融合体 2 个 ['cpp_fused__softmax_0', 'cpp_fused_sum_1']  外部调用 ['extern_kernels.mm']
torch.sort(t)[0]
    融合体 0 个 —  外部调用 ['aten.sort.stable']
t.cumsum(0)
    融合体 0 个 —  外部调用 ['aten.cumsum.default']
两个分支共享中间结果
    融合体 1 个 ['cpp_fused_add_mul_sum_0']  外部调用 无
```

把这张表读成三条规则：

- **融不融和「两个算子是不是一类」没关系，和「能不能塞进同一个循环体」有关系。** 两个互不依赖的归约（`sum` 和 `max`）照样融成一个 kernel —— 水平融合，一趟扫完各算各的累加器。归约→pointwise（`(t.sum()/3).relu()`）也融，pointwise 变成归约的 epilogue（kernel 名 `cpp_fused_div_relu_sum_0`，`sum` 排在最后但它是先执行的）。连 `t.half()+1` 里的 dtype 转换都融进去了（kernel 名 `cpp_fused__to_copy_add_0`）。
- **真正切断图的是外部调用。** `mm` / `sort` / `cumsum` 不生成 C++ kernel，直接调 aten，于是前后两段没法共享循环。`softmax → mm → sum` 就是 2 个融合体夹着 1 次 `extern_kernels.mm`：GEMM 像一堵墙，把链劈成两半。
- **中间结果被多处使用不构成障碍。** 最后一行 `a = x*2` 同时喂给 `a+1` 和 `a.sum()`，仍然只有 1 个融合体 `cpp_fused_add_mul_sum_0` —— 它在同一趟循环里把 `a` 算一次，然后分发给两个消费者。

还有第三类，不是「不融」而是**根本没进编译器**：数据依赖形状的算子会让 Dynamo 直接断图。`torch.nonzero` 断在 gb0036（输出形状取决于输入数据），`x.sum().item()` 断在 gb0124（`capture_scalar_outputs=False` 时的 `Tensor.item()`）。断图之后前后两段各自编译、中间那段退回 eager —— **没有任何报错**。

### 4.2 决策在哪一行：三个钩子 + 两个阈值

融合不是一个「算法」，是一个「候选打分 +  legality 检查」的循环。源码里对得上号的几个点：

| 位置 | 作用 |
|---|---|
| `can_fuse_vertical(node1, node2)` | 两个节点能不能上下串起来 |
| `can_fuse_horizontal(node1, node2)` | 两个互不依赖的节点能不能并进一个循环 |
| `can_fuse_reduction_epilogue(node1, node2)` | 归约后面能不能挂 pointwise（默认返回 `False`，由后端覆写） |
| `choices.py` 里的 `max_fusion_size` | 融合体里 **scheduler node** 个数的上限，默认 64 |
| `config.score_fusion_memory_threshold` | 两个节点融了只能省不到 10 个内存操作就不融，默认 10 |

一个容易误读的地方：`max_fusion_size = 64` 限制的是**节点数**，不是算子数。我用 200 轮（600 个 pointwise）的链试过，kernel 依然只有一个 —— 因为长 pointwise 链在 lowering 阶段早就内联进同一个循环体了，压根没变成 200 个 scheduler node 等着被融合。

### 4.3 它和已发的几个领域怎么咬合

- **和第 039 期（ATen / Dispatch）**：039 讲的是一个算子调用怎么找到它的实现，代价是查表 + 一次 kernel launch；这一期讲的是「把算子的边界本身抹掉」。两者是互补的省法：dispatch 省的是入口税（0.36 µs 量级），融合省的是内存往返（1 ms 量级）。
- **和第 001 期（CUDA Graph）**：CUDA Graph 把 launch 的 CPU 开销摊掉，但**图里每个 kernel 还是各跑各的、各读写各的**。融合改的是 kernel 的数量和它们之间的数据流。真正的「图执行」是两者叠起来：融合减少节点，CUDA Graph 减少节点之间的调度。
- **和第 005 期（FlashAttention）**：FA 本质上就是一次极致的融合 —— 把 `QK^T → softmax → PV` 整条链塞进一个 kernel，中间的 $S \times S$ 分数矩阵永远不落地。第 038 期量过：math 后端要 545 MB 的地方，融合版只要 8.5 MB，**64×**。那是同一笔流量账在 attention 上的极端版本。
- **和第 014 期（Profiling）**：融合之后你在 profiler 里看到的算子数会**直接变少**。看 trace 时别把「算子少了」误读成「计算少了」—— 算术量一点没变，变的是内存流量。
- **和第 037 期（零假设基准）**：第 2 节那个「上限 $m$ 倍」就是这一期的零假设。实测 24 个算子只有 13.6× 而不是 24×，差的那部分必须解释，不能直接当噪声。

### 4.4 在 CUDA 上是什么样（标注：本机无 GPU，以下为源码/文档所见 + 推断）

这台机器没有 CUDA，上面所有 kernel 都是 C++/OpenMP 后端生成的。在 CUDA 上同一套融合决策输出的是 Triton kernel：一个融合体一个 `@triton.jit`，用 `xindex` 遍历、`tl.load/tl.store` 收尾，命名规则是 `triton_<三字母类别>_fused_<算子名>_<序号>`（第 4.1 节末尾给过源码位置）。**融合的决策逻辑（候选、打分、legality）是后端共享的，只有代码生成器换掉** —— 这是我从 `torch/_inductor` 源码里读到的结构，不是实测。

跨硬件通用的部分：流量账（$8mN$ vs $8N$）、roofline 判据、水平/垂直两种形态、外部调用切断图。NVIDIA 特有的部分：Triton 作为默认 codegen、`tma` / `wgmma` 这类在 prologue/epilogue 融合里才会用到的异步搬运指令。迁移到别的加速器时，前三样能用，后两样要换。

## 5. 收益的量级

上面都是「几个 kernel」，这一节是「快多少」。64 MB 张量、`(t*1.0001+0.5).relu()` 重复 k 轮（每轮 3 个 pointwise），成对交替采样取最小值：

```python
import time, torch
torch.set_num_threads(4)

N = 1 << 24                        # 16.7M 个 float32 = 64 MB，远大于末级缓存
x = torch.randn(N)
BUF = 4 * N                        # 一个 float32 缓冲区的字节数

def pair(fa, fb, rep=7):
    """成对交替采样：同一轮里先后各跑一次，逐对比值；取多轮里的最小值（最干净的一轮）"""
    for _ in range(2):
        fa(); fb()
    best = (float("inf"), float("inf"))
    for _ in range(rep):
        t0 = time.perf_counter(); fa(); t1 = time.perf_counter()
        t2 = time.perf_counter(); fb(); t3 = time.perf_counter()
        if (t1 - t0) + (t3 - t2) < sum(best):
            best = (t1 - t0, t3 - t2)
    return best[0] * 1e3, best[1] * 1e3

y = torch.empty_like(x)
t_copy, _ = pair(lambda: y.copy_(x), lambda: y.copy_(x))
print(f"纯拷贝参考  {t_copy:5.2f} ms   {2*BUF/1e9/(t_copy/1e3):5.1f} GB/s  ← 这台机器的内存带宽天花板")

cf1 = torch.compile(lambda t: t + 1, dynamic=False)
cf1(x)
a, b = pair(lambda: torch.add(x, 1), lambda: cf1(x))
print("单个算子（读 64 MB + 写 64 MB = 128 MB 流量）")
print(f"  eager     {a:5.2f} ms   {2*BUF/1e9/(a/1e3):5.1f} GB/s")
print(f"  compiled  {b:5.2f} ms   {2*BUF/1e9/(b/1e3):5.1f} GB/s   eager/compiled = {a/b:.2f}x")

def chain(t, k):
    for _ in range(k):
        t = (t * 1.0001 + 0.5).relu()          # 每轮 3 个 pointwise：mul / add / relu
    return t

print("\n链式融合（eager 每个算子都要把中间结果写回内存）")
print(f"{'算子数':>5s} {'eager':>8s} {'compiled':>9s} {'加速':>6s} {'理论上限':>7s} {'融合体带宽':>10s}")
for k in (1, 2, 4, 8):
    cf = torch.compile(chain, dynamic=False)
    cf(x, k)
    e, c = pair(lambda: chain(x, k), lambda: cf(x, k))
    print(f"{3*k:5d} {e:6.2f}ms {c:7.2f}ms {e/c:5.2f}x {3*k:6d}x"
          f" {2*BUF/1e9/(c/1e3):8.1f} GB/s")
```

    纯拷贝参考   1.04 ms   129.2 GB/s  ← 这台机器的内存带宽天花板
    单个算子（读 64 MB + 写 64 MB = 128 MB 流量）
      eager      1.12 ms   119.7 GB/s
      compiled   1.18 ms   113.7 GB/s   eager/compiled = 0.95x

    链式融合（eager 每个算子都要把中间结果写回内存）
      算子数    eager  compiled     加速    理论上限      融合体带宽
        3   3.22ms    1.18ms  2.74x      3x    114.2 GB/s
        6   6.85ms    1.19ms  5.75x      6x    112.7 GB/s
       12  13.39ms    1.43ms  9.36x     12x     93.8 GB/s
       24  26.80ms    1.96ms 13.65x     24x     68.4 GB/s

（耗时类数字每次跑都会抖，这一块不参与逐行复核。同一脚本连跑三次的范围：3 个算子 2.69~2.74×、6 个 5.19~5.75×、12 个 8.95~9.84×、24 个 13.60~13.87×；纯拷贝 129~131 GB/s。）

三件事从这张表里读出来：

**① 3~6 个算子时，实测几乎兑现了理论上限。** 3 个算子 2.74×（上限 3×），6 个 5.75×（上限 6×）。原因看最后一列：融合体的有效带宽 112~114 GB/s，和纯拷贝的 129 GB/s 只差 12% —— 说明这段时间**完全由内存决定**，加速比自然等于流量比。

**② 24 个算子时收益掉出一半。** 13.65× 对上限 24×。关键证据不是这个比值，而是**字节数没变**：12 个算子搬的字节和 24 个一模一样，时间却从 1.43 ms 涨到 1.96 ms（+37%）。这 37% 不可能来自内存，只能来自计算 —— 融合体太长，把自己从带宽受限推成了计算受限。同一列也能看出来：有效带宽从 93.8 掉到 68.4 GB/s。

**③ 单个算子编译后反而慢一点。** `eager/compiled = 0.95x`，也就是编译版慢 5%（三次运行 0.94~0.96）。融合对单算子没有收益 —— 本来就只有 1 个 kernel，没有「中间结果」可省，只剩下「生成的 kernel 比 aten 手写 kernel 略慢」这点代价。

## 6. 什么时候该用 / 不该用

| 场景 | 建议 | 原因 |
|---|---|---|
| 一长串逐元素算子（归一化、激活、逐元素缩放、cast 混在一起） | ✅ 典型受益者 | 收益上限是算子个数，实测能拿到 90% |
| 热点里出现「elementwise 雨」的模块（Transformer block 的残差 + norm + 逐元素） | ✅ | 省的是内存带宽，模型越大越值 |
| 归约后面跟一串小算子（loss 计算、softmax 的尾巴） | ✅ | epilogue 融合白拿 |
| GEMM / convolution 主导的模型 | ⚠️ 收益有限 | 外部调用切断融合，你只能融到它的前后各一小段 |
| 只有 1~2 个算子的函数 | ❌ | 没有中间结果可省，实测还慢 5% |
| 张量小于几万个元素 | ❌ 见下面第 6 张表 | 编译版有 ~30 µs 的固定下限 |
| 需要逐比特复现的训练 | ⚠️ 先读第 7 节第 1 条 | reduction 的顺序会变 |

小张量这一条值得单独量一次。同样 6 个算子的链，扫一遍张量大小：

```python
import time, torch
torch.set_num_threads(4)

def chain(t):
    for _ in range(2):
        t = (t * 1.0001 + 0.5).relu()      # 6 个 pointwise 算子
    return t

def bench(fn, *a, rep=25):
    for _ in range(3):
        fn(*a)
    best = float("inf")
    for _ in range(rep):
        t0 = time.perf_counter(); fn(*a); best = min(best, time.perf_counter() - t0)
    return best * 1e6                       # µs，取最小值以压低抖动

cf = torch.compile(chain, dynamic=False)
print(f"{'元素数':>9s} {'eager(µs)':>10s} {'compiled(µs)':>13s} {'eager/compiled':>15s}")
for N in (1 << 10, 1 << 13, 1 << 14, 1 << 16, 1 << 18, 1 << 22):
    x = torch.randn(N)
    cf(x)
    e, c = bench(chain, x), bench(cf, x)
    print(f"{N:9d} {e:10.1f} {c:13.1f} {e/c:14.2f}x")
```

    元素数  eager(µs)  compiled(µs)  eager/compiled
       1024        4.3           7.5           0.58x
       8192        6.6          30.5           0.22x
      16384       10.6          28.9           0.37x
      65536       80.5          34.3           2.35x
     262144      100.0          46.9           2.13x
    4194304     1351.8         254.7           5.31x

（同样会抖，两次运行交叉点位置一致：1.6 万以下 eager 赢，6.5 万以上编译版赢。）

compiled 那一列在 8192~65536 之间有个 **~30 µs 的平台**，它和元素个数无关 —— 这是 `torch.compile` 每次调用的固定开销（guard 求值 + wrapper 派发），不是 kernel 的时间。eager 那一列则随 N 线性增长。两条线在 1.6 万~6.5 万元素之间交叉。

所以「小张量别 compile」不是玄学，是一条可以算出来的线：**固定开销 30 µs ÷ 每元素收益**。

## 7. 常见坑

**① 融合会改变归约顺序，于是结果不再逐比特一致。**

```python
import torch
torch.set_num_threads(4)
torch.manual_seed(0)

N = 1 << 20
x = torch.randn(N)

def report(name, fn):
    e1 = fn(x)                                  # eager 第一次
    e2 = fn(x)                                  # eager 第二次（对照：验证本机逐比特可复现）
    c = torch.compile(fn, dynamic=False)(x)     # 编译版
    d = (e1 - c).abs().max().item()
    scale = e1.abs().max().item()
    print(f"{name:26s} eager两次相同={torch.equal(e1, e2)!s:5s} "
          f"编译后逐比特相同={torch.equal(e1, c)!s:5s} max|Δ|={d:.3e} 相对={d/scale:.2e}")

report("4 个 pointwise",  lambda t: (t * 2 + 1).relu().sigmoid())
report("(t*2).sum()",     lambda t: (t * 2).sum())
report("t.sum()",         lambda t: t.sum())
report("softmax(dim=-1)", lambda t: torch.softmax(t, -1))
```

```text
4 个 pointwise              eager两次相同=True  编译后逐比特相同=True  max|Δ|=0.000e+00 相对=0.00e+00
(t*2).sum()                eager两次相同=True  编译后逐比特相同=False max|Δ|=2.441e-03 相对=9.86e-07
t.sum()                    eager两次相同=True  编译后逐比特相同=False max|Δ|=1.221e-03 相对=9.86e-07
softmax(dim=-1)            eager两次相同=True  编译后逐比特相同=False max|Δ|=5.093e-11 相对=8.34e-07
```

结论很干净：**纯 pointwise 融合逐比特相同（`0.000e+00`），带归约的一个都不一样**。

原因不难理解：逐元素算子的融合只是把「在哪里算」换了个地方，每个元素的算术表达式一字不差；而归约融合会改变**累加顺序**（分块、分线程、树形合并的方式和 aten 的实现不同），浮点加法不满足结合律，于是最后几位就不一样了。第一列 `eager两次相同=True` 是这次实验的对照组 —— 它证明后面那些差异不是噪声。

这条坑和第 038 期的发现是同一形状的：换一次执行路径，`1e-6` 级的差会在训练里被放大（那一期实测 800 步放大 **170×**）。要逐比特复现，就别让 reduction 走融合路径，或者接受「可复现的随机性只在同一份编译产物内成立」。

**② graph break 会让融合静默失效。**

`torch.nonzero`、`t.sum().item()`、`if t.sum() > 0:` 这类东西会让 Dynamo 直接断图。断图之后你得到的是「两小段各自融合 + 中间一段 eager」，编译照样成功、结果照样对，**只是你以为融了的地方没融**。想查的话数 `torch._dynamo.utils.counters["graph_break"]`，或者直接看生成了几个 `output_code` 模块。

**③ 「编译成功」不等于「融合了」。**

第 4 节那张表里，`mm` / `sort` / `cumsum` 都是 0 个融合体 + 1 次外部调用。如果热点正好是这三类，`torch.compile` 只是在外面套了一层 guard，一分钱不省。判断方法很简单：看 kernel 名字，或者数 `async_compile` 的行数。

**④ 冷启动的编译时间不属于稳态。**

第 5、6 节表里的 compiled 那一列都是**编译之后**的稳态。第一次调用要走完 Dynamo 追踪 → Inductor lowering → 融合决策 → 代码生成 → C++ 编译，小函数几十毫秒、大函数几秒。而且 `dynamic=False`（默认）下**每个新 shape 都要重编一次**。任何 benchmark 都必须先 warmup 再计时，否则你量到的是编译器。

## 8. 一句话总结

融合不改变算术量，只改变**中间结果的落点**：把「每个算子读一遍写一遍」压成「整条链读一遍写一遍」，所以收益上限是算子个数，实测能拿到九成 —— 直到融合体长到撞上算力墙，或者图被一个外部调用切断。

## 今日练习

<details><summary>今日练习：让一个融合体裂成两个，并说清为什么</summary>

目标：拿一条 12 个算子的纯 pointwise 链，把「1 个融合体」变成「2 个融合体」，然后解释切点在哪。

提示两条路：

- **路 A（改图）**：在链中间插一个不能塞进循环体的东西。`t.cumsum(0)`、`t.sort()[0]`、`torch.nonzero(...)` 都可以，但它们的效果不同 —— 前两个是「外部调用切断」，第三个是「graph break」。哪个会得到 2 个融合体、哪个会得到 0 个？
- **路 B（改决策）**：把 `torch._inductor.config.max_fusion_size` 调到 1，看融合体的个数怎么变。注意这条限制的是 **scheduler node 个数**，不是算子个数 —— 所以你需要先想办法让这条链真的变成多个 node。

参考答案（路 A，外部调用切断）：

```python
import glob, re, tempfile, torch
import torch._inductor.config as icfg
torch.set_num_threads(4)
icfg.force_disable_caches = True

def count(fn, *args):
    d = tempfile.mkdtemp(prefix="ind", dir="/tmp")
    icfg.trace.enabled = True
    icfg.trace.debug_dir = d
    torch.compile(fn, dynamic=False)(*args)
    code = open(glob.glob(f"{d}/**/output_code.py", recursive=True)[0]).read()
    return re.findall(r"^(\w+) = async_compile\.cpp_pybinding", code, re.M)

x = torch.randn(4096)

def pure(t):                      # 12 个 pointwise，1 个融合体
    for _ in range(4):
        t = (t * 1.0001 + 0.5).relu()
    return t

def split(t):                     # 中间插一个 cumsum
    for _ in range(2):
        t = (t * 1.0001 + 0.5).relu()
    t = t.cumsum(0)               # ← 外部调用，图在这里断开
    for _ in range(2):
        t = (t * 1.0001 + 0.5).relu()
    return t

print("纯 pointwise：", count(pure, x))
print("中间插 cumsum：", count(split, x))
```

预期是 `['cpp_fused_add_mul_relu_0']` 和 `['cpp_fused_add_mul_relu_0', 'cpp_fused_add_mul_relu_1']` 两行。切点在 `cumsum` 上，因为它必须把整个张量扫完（前缀和天然有跨元素依赖）才能交出结果，没法被一个「逐元素独立」的循环体吸收，只能作为 `aten.cumsum` 的外部调用存在。前后两段因此各自成一个融合体。

如果你改成插 `torch.nonzero(t > 0)`，会看到融合体数量变成 0 甚至编译根本没走到 Inductor —— 那是 graph break，不是「不融合」，两者要分清。

</details>
