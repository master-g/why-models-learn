---
title: "残差流：Transformer 各层共同读写的表示通道"
tags: ["why-models-learn"]
---

残差流（residual stream）是 Transformer 中沿层传递、并由各个子层逐次加入增量的表示序列。对输入为 $x_0\in\mathbb R^{B\times T\times d_{\mathrm{model}}}$ 的网络，第 $l$ 个子层通常把一个同宽度增量 $\Delta_l$ 写回

$$
x_{l+1}=x_l+\Delta_l.
$$

因此 $x_l$ 同时是当前层的 hidden、后续子层的输入和所有历史增量的和。attention、FFN、归一化、残差加法和输出头各自承担不同职责：残差流负责传递和累积表示，子层负责读取当前流并产生修正。[残差连接](../cnn/residual-connections/)解释一般加法 shortcut；这篇把它放回 Transformer 的层轴、特征轴和诊断账本中。

本文先固定 residual stream 的 shape 和加法合同，再用一个三维数值例子分解 attention/FFN 更新，推导恒等 Jacobian 路径，说明特征坐标不等于固定语义槽位，最后处理 pre/post norm、padding、KV cache、混合精度、内存和核验协议。

![残差流沿层轴保持相同的 hidden 宽度，attention 和 FFN 从当前流读取并把增量写回相加节点](/assets/transformer-components/svg/residual-streams.1.svg)

## 先固定 residual stream 的 shape

### 三个轴的含义

设第 $l$ 个阶段的 residual stream 为

$$
x_l\in\mathbb R^{B\times T\times d},
$$

其中 $B$ 是 batch size，$T$ 是序列长度，$d=d_{\mathrm{model}}$ 是 residual stream 的特征宽度。固定 $b,t$ 后，

$$
x_{l,b,t,:}\in\mathbb R^d
$$

是一个 token 在第 $l$ 个阶段的表示。层索引 $l$ 改变表示内容，通常不改变这三个张量轴的 shape。

| 轴 | 含义 | 残差流中的合同 |
| --- | --- | --- |
| $B$ | 独立样本或序列 | 不因残差加法而互相混合 |
| $T$ | token 位置 | attention 可以按 mask 读取其他位置 |
| $d$ | feature width | 加法前必须与分支输出相同 |

attention 可以跨 token 读取，FFN 通常逐 token 处理，但两者都把结果写回同一个 $d$ 宽度的流。token 间是否耦合由子层的读取规则决定，不由 residual add 自动决定。

### 加法保持宽度

若第 $l$ 个子层产生

$$
\Delta_l\in\mathbb R^{B\times T\times d},
$$

则

$$
x_{l+1}=x_l+\Delta_l
\in\mathbb R^{B\times T\times d}.
$$

这是逐元素相加，不是拼接。拼接会把宽度变成 $2d$ 或更多，必须另行加入投影；它不是 residual stream 的默认更新。

### 多个子层可以连续写入

一个 Transformer block 常见的两次更新为

$$
\begin{aligned}
x_{l+\frac12}&=x_l+\Delta_{\mathrm{attn},l},\\
x_{l+1}&=x_{l+\frac12}+\Delta_{\mathrm{ffn},l}.
\end{aligned}
$$

第二个分支读取的是第一次更新后的 $x_{l+\frac12}$，除非架构明确采用并行分支。记录一个 block 时，必须区分并行写入和串行写入。

## 一个三维 stream 的数值例子

### 初始表示和 attention 更新

取一个 token 的三维 residual stream：

$$
x_0=(1,2,0).
$$

假设 attention 子层从当前表示读出增量

$$
\Delta_{\mathrm{attn}}
=(0.5,-0.5,1).
$$

相加得到

$$
x_{\frac12}
=x_0+\Delta_{\mathrm{attn}}
=(1.5,1.5,1).
$$

这里没有新建一条独立 hidden 通道。attention 的输出已经成为下一子层看到的 residual stream。

### FFN 更新和最终表示

假设 FFN 读取 $x_{\frac12}$ 后写入

$$
\Delta_{\mathrm{ffn}}
=(-0.25,0.75,0.5).
$$

最终 stream 为

$$
x_1
=x_{\frac12}+\Delta_{\mathrm{ffn}}
=(1.25,2.25,1.5).
$$

从初始表示到最终表示的总更新为

$$
x_1-x_0
=\Delta_{\mathrm{attn}}+\Delta_{\mathrm{ffn}}
=(0.25,0.25,1.5).
$$

### 逐坐标分解

| 坐标 | $x_0$ | $\Delta_{\mathrm{attn}}$ | $\Delta_{\mathrm{ffn}}$ | $x_1$ |
| ---: | ---: | ---: | ---: | ---: |
| 1 | $1$ | $0.5$ | $-0.25$ | $1.25$ |
| 2 | $2$ | $-0.5$ | $0.75$ | $2.25$ |
| 3 | $0$ | $1$ | $0.5$ | $1.5$ |

第 2 个坐标的 attention 更新和 FFN 更新方向相反，但 FFN 的正向修正更大，所以最终坐标从 $2$ 变成 $2.25$。只看最终 stream 无法知道这个坐标经历了哪些中间更新；保存各层增量才能恢复分解。

### 范数和残差比例

独立计算得到：

| 对象 | L2 范数 |
| --- | ---: |
| $\lVert x_0\rVert_2$ | $2.236067977500$ |
| $\lVert\Delta_{\mathrm{attn}}\rVert_2$ | $1.224744871392$ |
| $\lVert x_{\frac12}\rVert_2$ | $2.345207879912$ |
| $\lVert\Delta_{\mathrm{ffn}}\rVert_2$ | $0.935414346693$ |
| $\lVert x_1\rVert_2$ | $2.979093821953$ |

以当前 stream 为 shortcut 参照，两个残差比例为

$$
\rho_{\mathrm{attn}}
=\frac{\lVert\Delta_{\mathrm{attn}}\rVert_2}
{\lVert x_0\rVert_2}
=0.547722557505,
$$

$$
\rho_{\mathrm{ffn}}
=\frac{\lVert\Delta_{\mathrm{ffn}}\rVert_2}
{\lVert x_{\frac12}\rVert_2}
=0.398862017609.
$$

残差比例描述写入量相对当前流的尺度，不能单独解释增量的功能或方向。

## 跨层分解：当前 stream 是历史更新的和

### 展开递推

若

$$
x_{l+1}=x_l+\Delta_l,
$$

连续展开得到

$$
x_L=x_0+\sum_{l=0}^{L-1}\Delta_l.
$$

这个等式把第 $L$ 层表示分成初始表示和每个阶段的更新。它不要求 $\Delta_l$ 独立，也不要求更新方向正交；更新可以相互抵消、重复或改变后续子层的读取结果。

### 串行子层的展开

对一个 block 内的两次更新：

$$
\begin{aligned}
x_{l+\frac12}&=x_l+\Delta_{\mathrm{attn},l},\\
x_{l+1}&=x_l+\Delta_{\mathrm{attn},l}+\Delta_{\mathrm{ffn},l}.
\end{aligned}
$$

如果 FFN 的增量依赖 $x_{l+\frac12}$，则 $\Delta_{\mathrm{ffn},l}$ 本身已经包含 attention 更新对 FFN 读取的影响。把它称为“独立贡献”时，需要说明这是加法分解，不是因果独立分解。

### 并行分支的展开

有些 block 令两个分支都读取同一个 $x_l$：

$$
x_{l+1}
=x_l+F_{\mathrm{attn}}(x_l)+F_{\mathrm{ffn}}(x_l).
$$

此时两个增量在同一个加法节点合并。它与先 attention 再 FFN 的串行结构具有相同的 shape 合同，但 forward 依赖和 Jacobian 不同。

| 结构 | attention 读取 | FFN 读取 | 写回 |
| --- | --- | --- | --- |
| 串行 | $x_l$ | $x_l+\Delta_{\mathrm{attn},l}$ | 两次加法 |
| 并行 | $x_l$ | $x_l$ | 一次合并加法 |

审计模型图时，不能只统计每层有几个分支；要记录每个分支读取的是哪个 stream 版本。

## Jacobian：加法提供一条恒等路径

### 单次更新的局部导数

令子层写入

$$
\Delta_l=F_l(x_l).
$$

递推为

$$
x_{l+1}=x_l+F_l(x_l).
$$

设 $J_{F_l}$ 是 $F_l$ 在 $x_l$ 处的 Jacobian，则

$$
\begin{aligned}
dx_{l+1}
&=dx_l+J_{F_l}dx_l\\
&=(I+J_{F_l})dx_l.
\end{aligned}
$$

所以

$$
J_l=I+J_{F_l}.
$$

$I$ 来自 stream 的直接传递，$J_{F_l}$ 来自写入分支。若分支先经过 norm，则改为

$$
J_l=I+J_{F_l}J_{N_l}.
$$

这正是 pre-norm 的局部 Jacobian；post-norm 则把 $J_N$ 放在整个加法结果外侧，见 [Pre-norm 与 Post-norm](../transformer-components/pre-norm-vs-post-norm/)。

### 深层 Jacobian 是有序乘积

从第 $0$ 层到第 $L$ 层：

$$
\frac{\partial x_L}{\partial x_0}
=J_{L-1}J_{L-2}\cdots J_0.
$$

矩阵顺序不能交换。每层的恒等项使乘积包含多条直接传递路径，但分支 Jacobian 仍可能在某些方向上连续放大或抵消。

### 线性更新例子

若每层局部近似为

$$
F_l(x_l)=\alpha x_l,
$$

则

$$
x_{l+1}=(1+\alpha)x_l.
$$

取 $\alpha=0.1$，从 $x_0$ 开始，前三次更新的尺度因子为

$$
1,\qquad
1.1,\qquad
1.1^2=1.21,\qquad
1.1^3=1.331.
$$

这个例子同时说明两件事：恒等路径存在不等于范数不变；同向残差更新仍会沿层累积。

## Attention 如何读取和写入 residual stream

### Q、K、V 从某个 stream 版本产生

设 attention 读取的表示为 $h$，则

$$
\begin{aligned}
Q&=hW_Q,\\
K&=hW_K,\\
V&=hW_V.
\end{aligned}
$$

attention 输出可以抽象成

$$
\Delta_{\mathrm{attn}}
=\operatorname{Attn}(h).
$$

写回后

$$
h_{\mathrm{next}}
=h+\Delta_{\mathrm{attn}}.
$$

对于 pre-norm，$h=N(x_l)$ 而写回的 shortcut 是 $x_l$；对于 post-norm，attention 通常读取 $x_l$，加法后再对结果执行 $N$。读取版本和写回版本不是同一个变量时，必须在图上标注。

### attention 可以跨位置读，但按位置写回

对于序列表示，attention 输出为

$$
\Delta_{\mathrm{attn},b,t,:}
=\sum_{s=1}^{T}
A_{b,t,s}V_{b,s,:},
$$

其中 $A$ 受 causal 或 padding mask 约束。输出仍然是每个 query 位置一个 $d$ 维向量，写回

$$
x_{l+1,b,t,:}
=x_{l,b,t,:}+\Delta_{\mathrm{attn},b,t,:}.
$$

attention 的跨 token 读取不改变 residual stream 的最后一轴宽度，也不意味着所有 token 的 shortcut 被混合。混合发生在 $\Delta_{\mathrm{attn}}$，原始 $x_l$ 仍沿加法路径保留。

### value 的坐标必须回到 stream 宽度

多头 attention 内部可能使用 $H$ 个 head 和 $d_h$ 维 head 表示：

$$
d_{\mathrm{model}}=H d_h.
$$

拼接 heads 后通常经过 $W_O$ 投影回 $d_{\mathrm{model}}$：

$$
\Delta_{\mathrm{attn}}
=\operatorname{Concat}(O_1,\ldots,O_H)W_O.
$$

没有这个回投影时，分支输出 shape 可能与 residual stream 不一致。GQA 或 MQA 改变 K/V 的 head 组织，不改变写回前必须匹配 $d_{\mathrm{model}}$ 的合同。

## FFN 如何读取和写入 residual stream

### FFN 通常逐 token 读取

对一个 token 的 stream 向量 $h_t\in\mathbb R^d$，普通 FFN 可写为

$$
F_{\mathrm{ffn}}(h_t)
=W_2\phi(W_1h_t+b_1)+b_2.
$$

$W_1$ 可以把宽度扩展到 $d_{\mathrm{ffn}}$，$W_2$ 再映射回 $d$：

$$
\mathbb R^d
\xrightarrow{W_1}
\mathbb R^{d_{\mathrm{ffn}}}
\xrightarrow{\phi}
\mathbb R^{d_{\mathrm{ffn}}}
\xrightarrow{W_2}
\mathbb R^d.
$$

FFN 的中间宽度不是 residual stream 的宽度。只有 $F_{\mathrm{ffn}}(h_t)$ 写回时，才需要匹配 $d$。

### 门控 FFN 仍然是一个写入分支

SwiGLU 等门控形式可以抽象为

$$
F_{\mathrm{gate}}(h)
=W_{\mathrm{down}}\bigl(\phi(W_{\mathrm{gate}}h)
\odot W_{\mathrm{up}}h\bigr).
$$

无论内部有几个投影，最终输出仍是一个 $d$ 维增量。门控改变增量的方向和尺度，不改变 residual add 的加法合同。

### FFN 的输入版本

串行 block 中

$$
\Delta_{\mathrm{ffn},l}
=F_{\mathrm{ffn}}(N_2(x_l+\Delta_{\mathrm{attn},l}))
$$

或在 post-norm 结构中采用另一种 $N_2$ 放置。若把 FFN 错误地改为读取 $x_l$，模型会变成并行分支；输出 shape 仍可能正确，但函数已经改变。

## 特征坐标、基变换与表示解释

### 坐标不是固定语义槽位

residual stream 的单个坐标是表示空间中的一个坐标分量。若对所有层使用可逆基变换 $P$：

$$
\widetilde x_l=Px_l,
$$

则同一个抽象表示可以在新坐标下写成不同的分量。线性层需要相应重参数化，才能保持整体函数：

$$
\widetilde W=P W P^{-1}
$$

或根据行列向量约定使用对应的左右变换。单独查看某个坐标的数值，不能直接把它命名为一个固定概念。

### 加法分解依赖坐标但向量恒等式不变

在原坐标中

$$
x_L=x_0+\sum_l\Delta_l.
$$

变换到新坐标：

$$
\widetilde x_L
=P x_0+\sum_lP\Delta_l.
$$

更新的加法关系保持，但每个坐标的大小、稀疏性和符号可能改变。使用 residual stream 做特征归因时，需要记录基、归一化、投影和读出方向。

### 不能从范数判断写入语义

两个增量可以有相同的 L2 范数，但方向完全不同；一个增量也可以范数很小，却在某个读出向量上产生很大变化。至少同时记录：

$$
\lVert\Delta_l\rVert_2,
\qquad
\cos(\Delta_l,x_l),
\qquad
w^{\mathsf T}\Delta_l
$$

其中 $w$ 是明确给出的读出方向。没有读出方向时，只能报告几何尺度，不能报告特征含义。

### 数值例子的增量方向

前面的三维例子中

$$
\Delta_{\mathrm{attn}}^{\mathsf T}
\Delta_{\mathrm{ffn}}
=0.
$$

两个增量正交，attention 和 FFN 对 stream 的更新方向不同。这个正交关系只属于该例的选取，不是 Transformer 的结构定理。

## Pre-norm、Post-norm 与 stream 尺度

### Pre-norm 直接保存相加后的 stream

pre-norm 递推为

$$
x_{l+1}
=x_l+F_l(N_l(x_l)).
$$

$x_l$ 的主路径不经过 $N_l$，所以 residual stream 的均值、方差或均方可以随深度变化。这个变化需要用残差比例和逐层统计量测量，不能由 norm 的存在与否推断。

### Post-norm 归一化相加结果

post-norm 递推为

$$
x_{l+1}
=N_l(x_l+F_l(x_l)).
$$

每层输出经过 $N_l$，但 shortcut 的梯度和分支的梯度也一起经过 $J_N$。输出统计更受控不等于所有梯度方向都保持；具体方向由 LayerNorm 或 RMSNorm 的 Jacobian 决定。

### 两种结构的 stream 账本

| 项目 | pre-norm | post-norm |
| --- | --- | --- |
| shortcut 写回 | 原始 $x_l$ | 加法后再经过 $N_l$ |
| 分支输入 | $N_l(x_l)$ | $x_l$ |
| stream 输出统计 | 不由当前 block 自动固定 | 受当前 $N_l$ 约束 |
| 局部 Jacobian | $I+J_FJ_N$ | $J_N(I+J_F)$ |
| final norm | 常见但由架构决定 | 由架构决定 |

残差流是加法通道，pre/post 是归一化调用位置；两者不能互换名称。

## Padding、KV cache 和序列边界

### Padding 不会从 stream 中自动消失

padding token 仍然占有一个 residual stream 向量：

$$
x_{b,t,:}\in\mathbb R^d.
$$

attention 是否能读取它，需要 key padding mask；pooling 是否包含它，需要有效 token mask；loss 是否计算它，需要 loss mask。残差加法本身不会检查 token 是否有效。

### Padding 增量需要单独检查

若某层对 padding 产生增量 $\Delta_{\mathrm{pad}}$，则

$$
x_{\mathrm{pad},l+1}
=x_{\mathrm{pad},l}+\Delta_{\mathrm{pad}}.
$$

即使 attention 不能读 padding，FFN 仍可能处理它。要让 padding stream 保持特定值，必须显式执行 mask、清零或其他约束；不能假定残差结构自动保持零。

### KV cache 保存的是读取结果，不是整个 stream 历史

自回归解码第 $t$ 步通常只计算当前 hidden：

$$
x_{t,l}\in\mathbb R^d.
$$

过去 token 的 K/V cache 供 attention 读取。当前 residual stream 仍沿当前 token 的层轴递推：

$$
x_{t,l+1}
=x_{t,l}+\Delta_{t,l}.
$$

把过去 hidden 直接拼到当前 stream 后再做残差加法，会改变 shape 和归约集合。cache 的读路径与 residual stream 的写路径应分别记录。

### Packed sequence 需要位置边界

多个短序列可以 packed 到同一个时间轴，但 attention mask 必须阻止不同样本之间互读。残差加法对每个位置仍然执行；它不会理解 packed segment 的边界。若 segment id 影响归一化或位置编码，还要检查这些算子是否在同一边界上。

## 计算、精度与激活内存

### 一层 stream 的激活账本

取

$$
B=2,
\qquad
T=4096,
\qquad
d=4096.
$$

一个 residual stream 的元素数为

$$
BTd=33554432.
$$

如果以 FP16 保存，一个 stream 激活占

$$
33554432\times2
=67108864\ \mathrm{bytes}
=64\ \mathrm{MiB}.
$$

这只是一个层状态，不包含 attention 的 Q/K/V、FFN 中间激活、norm 统计量和临时 workspace。

### 32 层的保存成本

若 backward 或诊断保存 $L+1$ 个 stream 状态，取 $L=32$，则

$$
(L+1)\times64\ \mathrm{MiB}
=33\times64\ \mathrm{MiB}
=2112\ \mathrm{MiB}
=2.0625\ \mathrm{GiB}.
$$

checkpoint、activation recomputation 和只保存部分层会改变这笔账。报告残差流内存时必须说明保存的是所有层、选定层，还是只保存当前层。

### 残差加法的精度

若 shortcut 与分支输出分别为

$$
x_l,\qquad \Delta_l,
$$

低精度相加可能吞掉相对很小的 $\Delta_l$。可以比较三条路径：

1. FP32 参考加法；
2. 目标 dtype 中的直接加法；
3. kernel 融合后的实际加法。

若 $\lVert\Delta_l\rVert_2$ 很小，输出差异可能主要来自 residual add 的舍入，而不是 attention 或 FFN 的数值错误。

### 混合精度的边界

| 项目 | 需要固定 |
| --- | --- |
| stream 存储 dtype | hidden 写回格式 |
| branch 计算 dtype | attention/FFN 的内部格式 |
| norm 归约 dtype | LayerNorm/RMSNorm 的统计累加格式 |
| residual add dtype | shortcut 与增量相加的格式 |
| cache dtype | K/V 的保存格式 |
| 输出投影 dtype | residual stream 读出到 logits 的格式 |

同一模型的 stream 可能在不同算子之间多次转换 dtype。只报告输入和最终输出 dtype，不能复现中间加法。

## 读出、干预与诊断

### 读出是从 stream 到目标的另一个线性映射

若某个线性读出方向为 $w\in\mathbb R^d$，则

$$
s_l=w^{\mathsf T}x_l
$$

是第 $l$ 层的标量读出。相邻层的变化为

$$
s_{l+1}-s_l
=w^{\mathsf T}\Delta_l.
$$

这个等式把“某层写入了什么”转换为指定读出方向上的数值变化。$w$ 必须固定或明确学习来源；改变 $w$ 会改变解释。

### 干预必须区分替换和加入

给第 $l$ 层 stream 加一个向量 $v$：

$$
\widetilde x_l=x_l+v
$$

与把它替换成另一个向量

$$
\widetilde x_l=v
$$

不是同一个干预。前者保留原始 stream，后者删除历史信息。若干预发生在 pre-norm 分支之前、post-norm 加法之后或 final norm 之前，后续函数也不同。

### 诊断表

| 指标 | 公式 | 说明 |
| --- | --- | --- |
| stream 范数 | $\lVert x_l\rVert_2$ | 当前表示尺度 |
| 更新范数 | $\lVert\Delta_l\rVert_2$ | 子层写入量 |
| 残差比例 | $\lVert\Delta_l\rVert_2/(\lVert x_l\rVert_2+\delta)$ | 写入相对 shortcut 的尺度 |
| 更新余弦 | $\cos(\Delta_l,x_l)$ | 更新与当前 stream 的方向关系 |
| 读出变化 | $w^{\mathsf T}\Delta_l$ | 指定方向上的增量 |
| 跨层余弦 | $\cos(x_l,x_{l+1})$ | 相邻 stream 的方向变化 |

这些指标回答不同问题。单独报告范数不能说明方向，单独报告余弦不能说明绝对大小。

### Norm 统计和 stream 统计分开记录

pre-norm 中 $N(x_l)$ 可能具有受控的特征统计，但 $x_{l+1}$ 是加法结果。应分别记录：

$$
\operatorname{stats}(x_l),
\qquad
\operatorname{stats}(N(x_l)),
\qquad
\operatorname{stats}(x_{l+1}).
$$

把 $N(x_l)$ 的均值或均方当作 residual stream 的均值或均方，会把分支输入和加法输出混为一谈。

## 失效模式

### 把 residual stream 当作拼接通道

残差更新是 $x+\Delta$，不是 $\operatorname{concat}(x,\Delta)$。检查加法前后的最后一轴；宽度从 $d$ 变成 $2d$ 时，模型已经进入另一种架构。

### 忘记分支回投影

attention head 或 FFN 中间宽度可以与 $d$ 不同，但写回前必须回到 $d$。shape 错误通常会立即失败，错误的投影参数则可能保持 shape 正确但改变写入语义。

### 混淆串行和并行分支

如果 FFN 读取 $x_l+\Delta_{\mathrm{attn}}$，它是串行结构；如果 FFN 也读取 $x_l$，它是并行结构。最终加法式子可能看起来相似，但 Jacobian 和 attention 对 FFN 的影响不同。

### 只保存最终 stream

最终 $x_L$ 无法唯一恢复每层 $\Delta_l$。不同的增量序列可以拥有相同的总和。若任务需要层级归因，必须保存目标层的 stream 或增量。

### 从坐标数值推断固定语义

特征基变换会改变坐标分量，但不一定改变整体函数。未经基、参数和读出方向约束，不能把单个坐标直接称为一个概念。

### 把 pre-norm 的 norm 输出当作 stream

pre-norm 的 $N(x_l)$ 进入分支，真正写回的是 $x_l+F(N(x_l))$。日志若只保存 norm 输入或 norm 输出，不能替代 residual stream 的层状态。

### 忽略 padding 或 packed 边界

残差 add 不处理有效 token、segment 或 cache 边界。若 attention mask、pooling mask 或 loss mask 缺失，stream 中的无效位置仍可能被读取或聚合。

### 低估所有层激活

单层 64 MiB 的 stream 在 32 层保存时已达到 2.0625 GiB，还未计入 Q/K/V、FFN 和 workspace。内存预算必须写出层数、dtype 和保存策略。

### 用全局范数替代读出变化

一个更新可以几乎不改变 $\lVert x_l\rVert_2$，但显著改变 $w^{\mathsf T}x_l$；也可以改变范数而对指定读出方向影响很小。按任务选择读出方向。

## 一个可复算的核验协议

### 先检查 shape 与计算图

对每个残差加法节点记录：

| 检查项 | 通过条件 |
| --- | --- |
| shortcut shape | 与 stream 同为 $(B,T,d)$ |
| branch output | 已回投影到 $(B,T,d)$ |
| 写入方式 | 明确是加法，不是拼接 |
| branch input | 记录读取的是 $x_l$、$N(x_l)$ 或中间 stream |
| 分支关系 | 明确串行或并行 |
| mask | attention、pooling、loss 的边界分别记录 |
| norm | 记录 norm 输入、输出和加法位置 |
| cache | 只进入 attention 读路径，不改变当前 stream shape |

### 再核对三维数值分解

使用

$$
x_0=(1,2,0),
\quad
\Delta_{\mathrm{attn}}=(0.5,-0.5,1),
\quad
\Delta_{\mathrm{ffn}}=(-0.25,0.75,0.5),
$$

应得到

$$
x_{\frac12}=(1.5,1.5,1),
\qquad
x_1=(1.25,2.25,1.5),
\qquad
x_1-x_0=(0.25,0.25,1.5).
$$

残差比例应为

$$
\rho_{\mathrm{attn}}=0.547722557505,
\qquad
\rho_{\mathrm{ffn}}=0.398862017609.
$$

### 核对 Jacobian 和激活账本

若局部写入为 $F_l(x)=0.1x$，前三层的尺度因子应为

$$
1,\qquad1.1,\qquad1.21,\qquad1.331.
$$

若 $B=2,T=4096,d=4096$，FP16 单层 stream 应为 $64$ MiB；保存 32 层输入和最终输出共 33 个状态时，应为 $2.0625$ GiB，不包含其他激活。

### 最后检查读出与干预位置

对任何解释或干预报告，附带：

1. 使用的层索引；
2. stream 是 norm 前还是 norm 后；
3. 更新是加法还是替换；
4. 读出方向或 probe 参数；
5. attention mask、padding 和 cache 状态；
6. stream 与分支的 dtype。

这些信息缺失时，只能报告一个张量变化，不能把它归因于某个子层或概念。

## 相关词条

- [残差连接](../cnn/residual-connections/)
- [LayerNorm](../transformer-components/layernorm/)
- [RMSNorm](../transformer-components/rmsnorm/)
- [Pre-norm 与 Post-norm](../transformer-components/pre-norm-vs-post-norm/)
- [自注意力](../attention/self-attention/)
- [混合精度训练](../training-nn/mixed-precision/)
