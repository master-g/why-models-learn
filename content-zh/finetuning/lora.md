---
title: "LoRA：用低秩增量限制参数更新"
tags: ["why-models-learn"]
---

LoRA（Low-Rank Adaptation，低秩适配）冻结预训练线性层的基础权重 $W_0$，只训练两个低秩因子 $A$ 和 $B$，再把增量写成 $\Delta W=(\alpha/r)BA$。它把一个 $d_{\mathrm{out}}\times d_{\mathrm{in}}$ 的全量更新限制在秩不超过 $r$ 的矩阵集合中。理解 LoRA 需要同时检查矩阵形状、缩放、初始化、目标模块、梯度路径和合并后的数值等价性；只报告 rank 或 adapter 文件大小，不能确认实际更新合同。

![LoRA 低秩增量示意图：冻结基础权重与输入保持主路径，输入同时经过 A、B 两个低秩因子，缩放后的增量与基础权重相加后得到部署矩阵](/assets/finetuning/svg/lora.1.svg)

## 先把全量更新改写成低秩更新

### 基础层和增量层的形状

设线性层的基础权重为：

$$
W_0\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\qquad
x\in\mathbb R^{d_{\mathrm{in}}},
\qquad
y_0=W_0x.
$$

全量微调会直接学习同样形状的 $\Delta W$。LoRA 把它分解成：

$$
\Delta W=BA,
\qquad
A\in\mathbb R^{r\times d_{\mathrm{in}}},
\qquad
B\in\mathbb R^{d_{\mathrm{out}}\times r}.
$$

于是：

$$
\operatorname{rank}(\Delta W)
\leq r,
\qquad
y=W_0x+\frac{\alpha}{r}BAx.
$$

矩阵乘法的轴必须先固定。$A$ 把 $d_{\mathrm{in}}$ 维输入压到 $r$ 维，$B$ 再把 $r$ 维结果映射回 $d_{\mathrm{out}}$ 维。若实现把 $A$、$B$ 的转置约定换了，参数量可能仍然看起来正确，但 forward 的输出轴已经改变。

### 参数量为什么从乘法变成加法

全量更新需要：

$$
P_{\mathrm{full}}=d_{\mathrm{out}}d_{\mathrm{in}}.
$$

LoRA 的 trainable 参数量为：

$$
P_{\mathrm{lora}}
=r d_{\mathrm{in}}+d_{\mathrm{out}}r
=r(d_{\mathrm{in}}+d_{\mathrm{out}}).
$$

当 $d_{\mathrm{in}}=d_{\mathrm{out}}=d$ 时，参数比率为：

$$
\rho
=\frac{P_{\mathrm{lora}}}{P_{\mathrm{full}}}
=\frac{2r}{d}.
$$

例如 $d=4096$、$r=8$ 时，单个方阵的比率为 $16/4096=0.390625\%$。这个比率只针对一个目标矩阵；如果把 attention 的四个投影和 MLP 的多个投影都设为 target module，应把每个矩阵的 $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ 相加。

|对象|形状|参数数量|训练状态|
|---|---|---:|---|
|基础权重 $W_0$|$d_{\mathrm{out}}\times d_{\mathrm{in}}$|$d_{\mathrm{out}}d_{\mathrm{in}}$|冻结但参与前向|
|全量增量 $\Delta W$|$d_{\mathrm{out}}\times d_{\mathrm{in}}$|$d_{\mathrm{out}}d_{\mathrm{in}}$|全部可训练|
|LoRA 因子 $A$|$r\times d_{\mathrm{in}}$|$rd_{\mathrm{in}}$|可训练|
|LoRA 因子 $B$|$d_{\mathrm{out}}\times r$|$d_{\mathrm{out}}r$|可训练|
|低秩增量 $BA$|$d_{\mathrm{out}}\times d_{\mathrm{in}}$|不单独保存|由 $A,B$ 计算|

### 一个二维输出的具体例子

取 $d_{\mathrm{in}}=3$、$d_{\mathrm{out}}=2$、$r=1$，并令：

$$
x=
\begin{bmatrix}
1\\2\\3
\end{bmatrix},
\qquad
A=
\begin{bmatrix}
0.2&-0.1&0.3
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
0.5\\-0.4
\end{bmatrix}.
$$

令 $\alpha=2$，因为 $r=1$，缩放为 $\alpha/r=2$。于是：

$$
\Delta W
=2BA
=
\begin{bmatrix}
0.2&-0.1&0.3\\
-0.16&0.08&-0.24
\end{bmatrix}.
$$

若基础矩阵是二维单位投影，基础输出、增量输出和合并输出分别为：

$$
y_0=
\begin{bmatrix}
1\\2
\end{bmatrix},
\qquad
\Delta y=\Delta W x=
\begin{bmatrix}
0.9\\-0.72
\end{bmatrix},
\qquad
y=
\begin{bmatrix}
1.9\\1.28
\end{bmatrix}.
$$

全量矩阵有 6 个参数，$A$ 和 $B$ 合计有 5 个参数。这个小例子的比率仍然是 83.333333%，因为维度太小；LoRA 的资源优势来自大模型投影中的 $r\ll d$，不能从低维玩具例子的比率外推。

## 缩放和初始化决定增量的起点

### $\alpha/r$ 是 forward 合同的一部分

常见 LoRA 前向写法为：

$$
y=W_0x+sBAx,
\qquad
s=\frac{\alpha}{r}.
$$

也可以把 $s$ 吸收进 $B$ 或在优化器外部处理，但训练、保存和合并必须使用同一个约定。若训练时用 $\alpha/r$，合并时误用 $\alpha$，增量会放大 $r$ 倍；若训练时已经把 scale 写进因子，推理时再次缩放，会得到重复缩放。

|变量|作用|需要记录的合同|
|---|---|---|
|rank $r$|限制增量的最大秩|每个 target module 的 rank|
|alpha $\alpha$|控制未归一化增量的乘数|数值、dtype、是否已吸收|
|scale $s$|实际进入 forward 的乘数|$\alpha/r$ 或其他实现约定|
|dropout|只作用于 adapter 输入的随机正则|概率、训练/推理开关|
|dtype|决定因子、累加和合并的精度|权重、因子、输出 dtype|

### 零增量初始化不是两边同时为零

常用初始化让 $A$ 使用小幅随机值、$B$ 全零。此时：

$$
B_0=0
\Longrightarrow
\Delta W_0=0
\Longrightarrow
y_0^{\mathrm{lora}}=W_0x.
$$

模型刚插入 LoRA 时，输出与基础模型一致。这个性质减少了适配模块对训练起点的扰动，但也造成第一步的梯度不对称。

令损失对输出的梯度为 $g=\partial\mathcal L/\partial y$。在行向量和列向量约定固定后，因子梯度可以写成：

$$
\frac{\partial\mathcal L}{\partial A}
=sB^{\mathsf T}g x^{\mathsf T},
\qquad
\frac{\partial\mathcal L}{\partial B}
=s g (Ax)^{\mathsf T}.
$$

当 $B=0$ 时：

$$
\frac{\partial\mathcal L}{\partial A}=0,
\qquad
\frac{\partial\mathcal L}{\partial B}
=s g (Ax)^{\mathsf T}.
$$

在前面的数字例子中，取 $g=[1,-2]$，$Ax=0.9$，所以首步得到：

$$
\frac{\partial\mathcal L}{\partial A}
=[0,0,0],
\qquad
\frac{\partial\mathcal L}{\partial B}
=[0.9,-1.8]^{\mathsf T}.
$$

这不是梯度断裂。$B$ 更新后，后续步骤中的 $\partial\mathcal L/\partial A$ 才会非零。审计时应分别打印两个因子的梯度范数，不能把初始化第一步的零梯度直接判定为 adapter 没有参与训练。

### 随机性必须放在增量路径

LoRA dropout 通常作用于进入 $A$ 的输入：

$$
y=W_0x+sB A\operatorname{Dropout}(x).
$$

它不应改变冻结基础路径。训练时若对基础路径和 adapter 路径使用不同的随机掩码，或者在合并前保留 dropout，merged 与 unmerged 的推理结果就不会相等。推理阶段应关闭 dropout，或者使用框架已经定义好的期望缩放。

## 目标模块决定低秩预算的落点

### attention 投影不是同一个功能

以 self-attention 为例，常见线性投影包括 $W_Q,W_K,W_V,W_O$。对它们全部插入相同 rank 的 LoRA，会把预算平均分配到查询、键、值和输出混合；只选 $W_Q,W_V$ 则减少参数，但更新空间也只覆盖两个路径。

|目标集合|主要改变的路径|资源结果|需要核对的现象|
|---|---|---|---|
|$W_Q,W_V$|查询选择和值读取|adapter 小|长距离选择或内容读取变化|
|$W_Q,W_K,W_V,W_O$|完整 attention 投影链|adapter 较大|注意力分布和输出混合都变化|
|MLP 上投影|中间特征扩展|增加通道变换自由度|任务词汇和组合特征变化|
|MLP 下投影|中间特征回写|影响残差写入|输出方向和残差幅度变化|
|attention + MLP|两类子层同时变化|参数和显存继续增加|更高容量也更难定位回归|
|embedding/LM head|词表输入或输出|参数量可能很大|新 token、词表和 tied weight 合同|

同一个名称在不同实现中可能对应 fused QKV、拆分的 Q/K/V 或量化包装层。target module 不能只按字符串猜测；应记录参数名、形状、是否共享存储和实际可训练数量。

### rank 预算必须按矩阵形状汇总

对多个目标矩阵 $\mathcal T$，LoRA 参数量为：

$$
P_{\mathrm{adapter}}
=\sum_{t\in\mathcal T}
r_t\bigl(d_{\mathrm{in},t}+d_{\mathrm{out},t}\bigr).
$$

不同层使用不同 rank 时，单个全局 rank 标签不足以复现实验。应保存每个 target module 的 $r_t$、alpha、dtype 和 factor shape。对 fused QKV，若权重形状为 $3d\times d$，直接作为一个目标矩阵的参数量是 $r(4d)$；拆成三个 $d\times d$ 矩阵时，若 rank 相同，总量也是 $3r(2d)=6rd$，两种写法并不相同。

|配置|单个 $d\times d$ 矩阵|四个 attention 矩阵|八个 attention/MLP 矩阵|
|---|---:|---:|---:|
|$d=4096,r=4$|32768|131072|262144|
|$d=4096,r=8$|65536|262144|524288|
|$d=4096,r=16$|131072|524288|1048576|
|$d=4096,r=32$|262144|1048576|2097152|

这张表只统计一个层的因子数量。若模型有 $L$ 层，应再乘以实际覆盖的层数，并把不同维度的 MLP 投影单独相加。

### 目标模块是能力范围，不是标签

LoRA 只在插入点产生增量。没有被选中的层仍使用 $W_0$，因此“rank-8 LoRA”不能单独描述可改变的函数族。至少要同时报告：

|字段|示例|缺失时的风险|
|---|---|---|
|target module|q_proj、v_proj|无法知道增量进入哪里|
|层范围|all layers 或指定层号|参数量和容量不确定|
|rank|8|低秩上限不确定|
|alpha|16|实际 scale 不确定|
|dropout|0.05|训练/推理路径不一致|
|bias policy|none、all 或 lora_only|隐藏的可训练参数|
|dtype|BF16 factor、FP32 master|显存和数值行为不确定|

## 低秩限制如何影响可表达的更新

### LoRA 学习的是一个受限子空间

任意矩阵 $\Delta W$ 都可以用奇异值分解表示为：

$$
\Delta W=U\Sigma V^{\mathsf T}.
$$

秩为 $r$ 的矩阵只能包含至多 $r$ 个非零奇异值。LoRA 的 $BA$ 因子直接把增量限制在这个集合中；训练并不会先求出全量微调的 $\Delta W$ 再做 SVD。若任务所需的变化需要许多独立方向，过小的 rank 会形成欠容量。

对固定参数预算，增加目标层的数量与增加单层 rank 是两种不同选择。前者让更多函数路径可变，后者让每个已选路径拥有更多独立方向。两者的 trainable 参数量可能相同，质量和回归位置仍可能不同。

|变化方式|保持不变的部分|新增自由度|适合检查的问题|
|---|---|---|---|
|rank 4 → rank 8|目标层不变|每个目标层更多方向|单层更新是否受 rank 限制|
|只调 Q/V → 调 Q/K/V/O|rank 不变|更多 attention 路径|选择与读取是否需要同时变化|
|后半层 → 全层|rank 和层内结构不变|更多深度位置|任务变化是否依赖早期表征|
|单任务 adapter → 多 adapter|基础权重不变|多个独立增量|路由、缓存和合并是否隔离|

### rank 不是越大越好

增大 rank 会增加可训练参数、optimizer state 和通信 payload，也可能让 adapter 更容易拟合训练数据。减小 rank 会降低资源，但可能使训练 loss 停在较高位置，或者只改变局部词汇而不能改变任务结构。比较 rank 时，应固定数据切分、有效监督 token、学习率搜索范围和 target module，报告 base、不同 rank 和 full 的同协议结果。

## 合并、未合并和多 adapter

### 合并前后应满足同一 forward

训练完成后，可以把增量写回基础权重：

$$
W_{\mathrm{merged}}
=W_0+sBA.
$$

未合并路径计算：

$$
y_{\mathrm{unmerged}}
=W_0x+sBAx.
$$

合并路径计算：

$$
y_{\mathrm{merged}}
=W_{\mathrm{merged}}x.
$$

在相同 dtype、同一 scale、无 dropout、相同 bias 和相同 kernel 语义下，两者应在浮点舍入误差范围内相同。合并是数值变换，不是只复制 adapter 文件；应保存合并前后的权重 hash、scale、dtype 和工具版本。

|部署形态|基础权重|adapter 因子|运行时计算|适用场景|
|---|---|---|---|---|
|unmerged|单独保存|单独加载|基础矩阵加低秩路径|多任务切换、保留可卸载 adapter|
|merged|已写入增量|可不再加载|单一矩阵路径|固定任务、减少运行时分支|
|多个 unmerged|同一基础权重|按请求路由|每个 adapter 产生自己的增量|多租户或任务路由|
|错误 merge|版本或 scale 不匹配|文件仍可读取|输出发生系统性偏移|需要立即回滚并重新校验|

### 多 adapter 需要隔离请求状态

同一个基础模型可以加载多个 LoRA adapter。服务端必须把 adapter ID、rank、scale、dtype 和请求 batch 绑定在一起。若 KV cache、prefix cache 或编译 kernel 仍复用上一个 adapter 的状态，输出会混入另一个增量。切换 adapter 时，应明确缓存是否失效；不能只检查模型对象上当前的 adapter 名称。

### base hash 是加载合同的一部分

adapter 只保存 $A$、$B$ 和配置，通常不包含完整基础权重。加载时至少检查：

1. 基础模型内容 hash 与训练时记录一致。
2. tokenizer、词表和 chat template 一致。
3. target module 的名字、shape 和共享权重关系一致。
4. rank、alpha、bias policy、dropout 和 dtype 一致。
5. merged/unmerged 模式与 checkpoint 标记一致。

如果基础 checkpoint 只改了 tokenizer 或 rotary 配置，adapter 文件仍可能成功加载，但输出语义已经不再具有可比性。

## 训练时要同时看两个因子

### 梯度路径的形状

把 $z=Ax$ 写成 rank 维的中间表示，可以得到：

$$
z\in\mathbb R^r,
\qquad
\Delta y=sBz,
\qquad
\frac{\partial\mathcal L}{\partial B}
=s\frac{\partial\mathcal L}{\partial y}z^{\mathsf T}.
$$

梯度对 $A$ 的更新依赖 $B$，梯度对 $B$ 的更新依赖 $Ax$。如果某个因子梯度长期为零，应区分三种原因：初始化阶段的预期不对称、输入或输出 mask 没有覆盖目标 token、以及参数实际没有进入 optimizer group。

### optimizer group 不能漏掉因子

冻结基础权重通常通过 requires-grad 或参数过滤实现，但这不等于 A、B 自动进入 optimizer。至少应打印：

|检查项|最低证据|
|---|---|
|可训练数量|按参数名求和并与公式相符|
|梯度|A、B 在有效 batch 后均有非零范数，允许首步 A 为零|
|optimizer group|A、B 都在 group，基础权重不在更新 group|
|学习率|记录 adapter 的 LR，不直接假设等于全量微调 LR|
|state|确认只为可训练参数创建 Adam moments|
|保存|checkpoint 中包含 A、B、配置和 base 标识|

### 学习率和 alpha 不能互相替代

alpha 改变 forward 中的增量幅度，学习率改变参数更新速度。把 alpha 提高一倍和把学习率提高一倍，虽然都可能增大训练早期的变化，但梯度、合并权重和最终 adapter 参数并不相同。实验记录应分开保存 alpha、实际 scale、学习率、warmup、weight decay 和 gradient clipping。

## 一个可运行的 LoRA 形状探针

下面的探针只使用 Python 标准库，核对二维 forward、缩放、合并输出和 $B=0$ 初始化时的梯度方向。它不实现反向传播框架，因此不能替代真实训练，但可以先排除转置、scale 和参数计数错误。

```python
from math import sqrt

x = [1.0, 2.0, 3.0]
A = [0.2, -0.1, 0.3]
B = [0.5, -0.4]
alpha = 2.0
rank = 1
scale = alpha / rank

delta = [[scale * b * a for a in A] for b in B]
base = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

def dot(row, vector):
    return sum(a * b for a, b in zip(row, vector))

y_base = [dot(row, x) for row in base]
y_delta = [dot(row, x) for row in delta]
y_merged = [a + b for a, b in zip(y_base, y_delta)]

g = [1.0, -2.0]
a_x = dot(A, x)
grad_a_when_b_zero = [0.0] * len(A)
grad_b_when_b_zero = [value * a_x for value in g]

fmt = lambda values: '[' + ', '.join(f'{value:.6f}' for value in values) + ']'
print('delta_W=', '[' + ', '.join(fmt(row) for row in delta) + ']')
print('base_output=', fmt(y_base))
print('delta_output=', fmt(y_delta))
print('merged_output=', fmt(y_merged))
print('A_x=', f'{a_x:.6f}')
print('grad_A_when_B_zero=', fmt(grad_a_when_b_zero))
print('grad_B_when_B_zero=', fmt(grad_b_when_b_zero))
print('full_matrix_parameters=', len(base) * len(base[0]))
print('lora_parameters=', len(A) + len(B))
print('trainable_ratio_percent=', f'{100 * (len(A) + len(B)) / (len(base) * len(base[0])):.6f}')
print('scaled_delta_frobenius=', f'{sqrt(sum(value * value for row in delta for value in row)):.6f}')
```

运行输出：

```text
delta_W= [[0.200000, -0.100000, 0.300000], [-0.160000, 0.080000, -0.240000]]
base_output= [1.000000, 2.000000]
delta_output= [0.900000, -0.720000]
merged_output= [1.900000, 1.280000]
A_x= 0.900000
grad_A_when_B_zero= [0.000000, 0.000000, 0.000000]
grad_B_when_B_zero= [0.900000, -1.800000]
full_matrix_parameters= 6
lora_parameters= 5
trainable_ratio_percent= 83.333333
scaled_delta_frobenius= 0.479166
```

探针输出说明 $A$ 先产生标量 $Ax=0.9$，再由 $B$ 生成二维增量 $[0.9,-0.72]$。缩放后的低秩矩阵与基础矩阵相加后，输出从 [1,2] 变为 [1.9,1.28]。$B$ 全零时，$\partial\mathcal L/\partial A$ 为零而 $\partial\mathcal L/\partial B=[0.9,-1.8]^{\mathsf T}$，符合梯度公式。玩具层的 trainable ratio 为 83.333333%，不能代表 4096 维投影的实际比例。

## 运行方法

将上一个 Python 代码块保存为 lora-shape-probe.py，再运行 python3 lora-shape-probe.py。修改 $d_{\mathrm{in}}$、$d_{\mathrm{out}}$、rank、target module 或 alpha 时，应同步检查因子 shape、参数数量、scale 和合并输出。

接入真实训练时，先枚举模块和参数名，再统计每层的 factor shape 与 trainable count。训练一个有效 batch 后分别记录 A、B 的梯度范数、optimizer state、有效监督 token 和 adapter checkpoint 大小。合并后用同一输入比较 merged 与 unmerged 的 logits 最大绝对差，并记录比较时的 dtype。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|LoRA 初始输出已偏离 base|B 是否按零增量初始化、scale 是否重复|同一输入比较插入前后 logits|
|adapter 没有学习|A、B 是否进入 optimizer、loss mask 是否有效|打印参数名、梯度范数和 group|
|合并后结果放大|训练和 merge 的 alpha/r 约定不同|记录 merged delta 与训练 forward|
|单个因子梯度长期为零|首步初始化、mask、冻结配置或 shape 错误|比较首步和多个有效 batch|
|参数量与 rank 不符|fused 模块、层范围或 bias policy 未计入|按模块 shape 重新求和|
|换基础模型仍能加载|加载器只检查文件字段，不检查 base hash|拒绝 hash 不匹配的 checkpoint|
|多个任务输出互相影响|adapter ID、batch 或 KV cache 复用|按请求记录 adapter 和 cache 状态|
|merged/unmerged 不一致|dropout、dtype、bias 或量化顺序不同|固定输入逐 token 比较 logits|
|rank 增大但质量不升|数据、有效 token 或目标层是瓶颈|保持训练协议，比较回归矩阵|
|adapter 文件很小但显存不足|冻结基础权重、激活或 workspace 占主导|分项测量运行时峰值|
|target module 匹配为空|实现使用 fused 或包装层名称|打印实际模块树与可训练参数|

### 最小审计表

|审计层|应保存|验收问题|
|---|---|---|
|base|内容 hash、tokenizer、template、位置配置|adapter 是否绑定同一个基础函数|
|module|参数名、shape、共享关系、层范围|LoRA 实际插入了哪些路径|
|factor|A/B shape、rank、alpha、dtype|低秩增量是否可复现|
|gradient|A/B 梯度范数、有效 token、mask|两个因子是否真正获得训练信号|
|optimizer|group、LR、weight decay、state|是否只为可训练参数维护 state|
|forward|base、unmerged、merged 的 logits 差值|合并是否保持数值合同|
|checkpoint|adapter、配置、base 标识、工具版本|能否在目标环境恢复|
|deployment|adapter ID、cache、batch、延迟|多 adapter 是否隔离|
|quality|base、不同 rank、full 的同协议结果|低秩限制造成了什么回归|

LoRA 的可训练参数少，来自 $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ 的因子化；它的能力边界来自 target module 和 rank，它的部署正确性来自 scale、dtype、base hash 与 merged/unmerged 一致性。实验报告至少要同时给出这些字段，不能只写“rank-8 LoRA”。

## 相关词条

- [全量微调与参数高效微调](../finetuning/full-vs-peft/)：比较全量更新与 PEFT 的参数空间、资源账和 checkpoint 合同。
- [监督微调](../finetuning/sft/)：说明示范数据如何产生 next-token 梯度。
- [指令数据](../finetuning/instruction-data/)：检查 LoRA 训练数据的结构、质量、覆盖和切分。
- [QLoRA](../finetuning/qlora/)：展开量化基础权重与 LoRA adapter 的组合。
- [Prompt Tuning](../finetuning/prompt-tuning/)：比较虚拟 token 参数与权重增量路径。
- [参数量](../transformer-components/parameter-count/)：核对矩阵、层和存储字节数。
- [优化器](../training-nn/optimizers/)：说明 A、B 的 optimizer state 与更新规则。
- [混合精度](../training-nn/mixed-precision/)：检查因子、基础权重和合并计算的 dtype。
- [灾难性遗忘](../finetuning/catastrophic-forgetting/)：评估 adapter 对基础能力和任务能力的影响。
- [推理](../inference/inference/)：比较 merged、unmerged 和多 adapter 的运行时路径。
