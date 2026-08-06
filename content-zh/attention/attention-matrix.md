---
title: "Attention Matrix：权重如何成为一次矩阵乘法"
tags: ["why-models-learn"]
---

Attention matrix（注意力矩阵）是 query 对 key 的读取权重组成的矩阵。沿本文固定的行位置约定，第 $t$ 行对应第 $t$ 个 query，第 $i$ 列对应第 $i$ 个 key。矩阵先由缩放 score 和 mask 经过逐行 softmax 得到，再左乘 value：

$$
\begin{gathered}
S=\frac{QK^\mathsf T}{\sqrt{d_k}},\\
A=\operatorname{softmax}_{\mathrm{row}}(S+M),\\
C=AV.
\end{gathered}
$$

$A$ 的每一行是一个读取分布，$C$ 的每一行是该 query 从所有有效 value 读回的凸组合。attention matrix 不是只用于画热图的副产物；它是从 score 到 value 的线性读取算子。[自注意力](../attention/self-attention/)说明这些 query、key、value 如何从同一序列生成，[缩放点积注意力](../attention/scaled-dot-product/)说明 score 为什么要除以 $\sqrt{d_k}$，本篇集中处理矩阵 $A$ 的形状、方向和运算。

![query-key score 经过逐行 softmax 变成注意力矩阵，每一行再乘 value 得到对应位置的上下文输出](/assets/attention/svg/attention-matrix.1.svg)

## 先固定矩阵的方向

### 行是 query，列是 key

设 query 有 $T$ 个位置，key/value 有 $S$ 个位置。采用每个位置占一行的约定：

|对象|形状|行的含义|列的含义|
|---|---|---|---|
|$Q$|$T\times d_k$|query 位置 $t$|匹配维坐标|
|$K$|$S\times d_k$|key 位置 $i$|匹配维坐标|
|$QK^\mathsf T$|$T\times S$|第 $t$ 个 query|第 $i$ 个 key|
|$A$|$T\times S$|第 $t$ 个 query 的读取分布|第 $i$ 个 key 收到的读取权重|
|$V$|$S\times d_v$|第 $i$ 个 value|value 内容坐标|
|$C=AV$|$T\times d_v$|第 $t$ 个 query 的输出|输出内容坐标|

在 self-attention 中 $S=T$，所以 $A$ 是方阵；在 cross-attention 中 $T$ 与 $S$ 可以不同，$A$ 仍然是一个从 key/value 位置到 query 位置的 $T\times S$ 矩阵。[交叉注意力](../attention/cross-attention/)会处理两条序列的来源差异。

第 $t$ 行第 $i$ 列的元素为：

$$
A_{t,i}
=
\frac{
\exp\left(
\frac{\mathbf q_t\mathbf k_i^\mathsf T}{\sqrt{d_k}}
+M_{t,i}
\right)
}{
\sum_{j=1}^{S}
\exp\left(
\frac{\mathbf q_t\mathbf k_j^\mathsf T}{\sqrt{d_k}}
+M_{t,j}
\right)
}.
$$

分母沿 $i$ 所在的 key 轴求和。它不是沿 query 轴求和，所以一个 key 列的总权重不需要等于 1。

### 行随机而非列随机

如果每个 query 至少有一个有效 key，并且 mask 以 $-\infty$ 排除非法位置，则：

$$
A_{t,i}\ge 0,
\qquad
\sum_{i=1}^{S}A_{t,i}=1.
$$

用全 1 列向量表示，就是：

$$
A\mathbf 1_S=\mathbf 1_T.
$$

这类矩阵称为 row-stochastic matrix（行随机矩阵）。它的列和表示所有 query 合计分给某个 key 的读取质量：

$$
\mathbf 1_T^\mathsf T A
=
\left(
\sum_{t=1}^{T}A_{t,1},
\ldots,
\sum_{t=1}^{T}A_{t,S}
\right).
$$

列和可以大于 1，也可以小于 1；总和固定为 $T$：

$$
\sum_{t=1}^{T}\sum_{i=1}^{S}A_{t,i}=T.
$$

因此「这一列总权重较大」与「每一行都把它当作最高分 key」不是同一个判断。前者是跨 query 的聚合量，后者是每一行内部的相对排序。

### 位置约定可以改变，但必须全程一致

有些库把 token 表示为列向量，或者把 attention 权重写成 key 在行、query 在列。那时可能出现 $KQ^\mathsf T$、$A^\mathsf T V$ 或其他等价转置。记号本身不是错误，错误发生在以下对象的方向互相不匹配：

1. score 的行到底代表 query 还是 key；
2. softmax 沿哪条轴归一化；
3. value 的位置轴是否与 score 的 key 轴对齐；
4. 输出的位置轴是否与 query 轴对齐。

本文后面的所有公式都使用 query 行、key 列、$C=AV$。审计另一个实现时，先画出四个位置轴，再把它转换到这套约定比较。

## 一个三位置的 attention matrix

### 先给出 score 矩阵

取三个 query 和三个 key，假设缩放后的 score 为：

$$
S
=
\begin{bmatrix}
1&0&-1\\
0&0&0\\
-1&0&1
\end{bmatrix}.
$$

这个矩阵对称，表示位置 1 更匹配 key 1，位置 3 更匹配 key 3，位置 2 对三个 key 没有偏好。沿每一行 softmax：

$$
A
=
\begin{bmatrix}
0.665240956&0.244728471&0.090030573\\
0.333333333&0.333333333&0.333333333\\
0.090030573&0.244728471&0.665240956
\end{bmatrix}.
$$

每行的近似和为 1：

$$
A\mathbf 1_3
=
\begin{bmatrix}
1\\
1\\
1
\end{bmatrix}.
$$

但列和为：

$$
\mathbf 1_3^\mathsf T A
=
(1.088604862,\ 0.822790275,\ 1.088604862).
$$

第三个结果说明 $A$ 不是列随机矩阵。总列和为 $3$，与三行各自归一化一致。

### 对称 score 不保证对称 attention matrix

尽管 $S=S^\mathsf T$，矩阵 $A$ 仍不是对称矩阵：

$$
A_{1,2}=0.244728471,
\qquad
A_{2,1}=0.333333333.
$$

原因在于 softmax 的分母按行变化。位置 1 的竞争分数是 $(1,0,-1)$，位置 2 的竞争分数是 $(0,0,0)$；同一个 pair 的分子可以相同，分母却不同。只有对称 score 还不够，若要得到对称且行和为 1 的矩阵，还需要额外约束或专门的归一化。

### 矩阵乘 value

取每个 key 的二维 value：

$$
V
=
\begin{bmatrix}
10&0\\
20&5\\
40&10
\end{bmatrix}.
$$

矩阵乘法给出：

$$
C=AV
=
\begin{bmatrix}
15.148201906&2.123948087\\
23.333333333&5\\
32.404513384&7.876051913
\end{bmatrix}.
$$

第一行的第一列可以逐项展开：

$$
C_{1,1}
=
0.665240956\times10
+0.244728471\times20
+0.090030573\times40
=
15.148201906.
$$

第一行的第二列使用同一组读取权重，但读取 value 的第二个坐标：

$$
C_{1,2}
=
0.665240956\times0
+0.244728471\times5
+0.090030573\times10
=
2.123948087.
$$

矩阵乘法在这里不是抽象的批量技巧。$A$ 的每一行直接规定一个 query 如何混合 $V$ 的各行，$C$ 的同一行保留该 query 对所有 value 的同一组位置权重。

## $A$ 是 value 的混合算子

### 每一行输出都是凸组合

由于 $A_{t,i}\ge0$ 且行和为 1：

$$
\mathbf c_t
=
\sum_{i=1}^{S}A_{t,i}\mathbf v_i
$$

是 value 向量的凸组合。对每个 value 坐标 $r$，都有：

$$
\min_i V_{i,r}
\le
C_{t,r}
\le
\max_i V_{i,r}.
$$

这个范围约束只适用于没有额外输出投影、残差或非线性处理的 $C=AV$。Transformer 层还可能在 attention 输出后接输出投影、残差连接和归一化，所以最终 hidden state 不必位于原始 value 的坐标范围内。

### 常量 value 通过 attention 不变

如果所有 value 行都相同，记为 $\mathbf v^\mathsf T$，则：

$$
V
=
\mathbf 1_S\mathbf v^\mathsf T.
$$

利用 $A\mathbf 1_S=\mathbf 1_T$：

$$
AV
=
A\mathbf 1_S\mathbf v^\mathsf T
=
\mathbf 1_T\mathbf v^\mathsf T.
$$

每个 query 都得到同一个 value。attention matrix 可以重新分配位置权重，但不能从完全相同的 value 中制造位置差异。

### 列和描述跨 query 的总读取质量

矩阵乘法可以按列看：

$$
C_{:,r}
=
A V_{:,r}.
$$

固定某个 value 坐标 $r$ 时，$V_{:,r}$ 是长度为 $S$ 的列向量，$A$ 把它映射到长度为 $T$ 的 query 输出。某个 key 列的列和越大，表示该 key 的内容被更多 query 或更多总权重读取；它不说明该 key 的 value 必然更重要，因为 value 内容和下游损失仍然决定最终影响。

## mask 如何改变矩阵

### causal mask 只保留下三角

自回归 decoder 不能读取未来位置。三位置的 causal mask 可以写成：

$$
M_{\mathrm{causal}}
=
\begin{bmatrix}
0&-\infty&-\infty\\
0&0&-\infty\\
0&0&0
\end{bmatrix}.
$$

把它加到 score 后再逐行 softmax：

$$
A_{\mathrm{causal}}
=
\begin{bmatrix}
1&0&0\\
0.5&0.5&0\\
0.090030573&0.244728471&0.665240956
\end{bmatrix}.
$$

第一行只有当前位置有效，第二行在两个历史位置之间平均读取，第三行读取完整的三位置集合。每行仍然在自己的有效 key 集合内归一化，不能用全矩阵的总和判断 mask 是否正确。

用上一个二维 value 矩阵：

$$
A_{\mathrm{causal}}V
=
\begin{bmatrix}
10&0\\
15&2.5\\
32.404513384&7.876051913
\end{bmatrix}.
$$

第一个输出不含位置 2、3 的 value；第二个输出不含位置 3 的 value。这种依赖约束来自 mask，而不是来自 softmax 自身。

### padding mask 可以使每行的候选数不同

batch 中的序列长度不同时，padding key 需要从对应 query 行排除。设第一个样本只有前两个 key 有效，第二个样本三个 key 都有效，则 mask 的有效集合不同：

|query 样本|有效 key|该行权重和|
|---|---|---:|
|样本 1 的位置 1|$1,2$|1|
|样本 1 的位置 2|$1,2$|1|
|样本 2 的位置 1|$1,2,3$|1|
|样本 2 的位置 2|$1,2,3$|1|

padding query 本身是否需要计算输出，取决于上游模块的约定。常见做法是保留矩形张量、用 query mask 清零输出或在损失 reduction 时忽略 padding；这与只在 key 轴上 mask 是两个不同的操作。

### 全 mask 行没有合法的概率分布

如果某个 query 行的所有 key 都被设为 $-\infty$：

$$
\operatorname{softmax}(-\infty,\ldots,-\infty)
$$

没有定义，因为分母为 0。实现中可能出现 NaN，也可能用特殊分支返回全零。全零行不再满足行和为 1，所以后续代码必须知道它表示「无可读内容」，不能把它当成普通 attention matrix。空 memory、全 padding 候选和错误的 causal 边界都可能制造这种情况。

## 行 softmax 与其他归一化

### 归一化轴决定读取合同

对 score 矩阵 $S$，行 softmax 是：

$$
\operatorname{softmax}_{\mathrm{row}}(S)_{t,i}
=
\frac{\exp(S_{t,i})}
{\sum_{j=1}^{S}\exp(S_{t,j})}.
$$

列 softmax 则是：

$$
\operatorname{softmax}_{\mathrm{col}}(S)_{t,i}
=
\frac{\exp(S_{t,i})}
{\sum_{u=1}^{T}\exp(S_{u,i})}.
$$

前者让每个 query 在 key 集合中分配 1 单位读取质量；后者让每个 key 在 query 集合中分配 1 单位质量。它们回答不同问题，不能只通过转置矩阵形状来替换。

### 行 softmax 的对照例子

对前面的 score 矩阵，行 softmax 得到：

$$
A_{1,:}
=(0.665240956,\ 0.244728471,\ 0.090030573).
$$

如果沿列归一化，第一列的分母是：

$$
\exp(1)+\exp(0)+\exp(-1),
$$

于是列归一化后的第一列为：

$$
\left(
0.665240956,\ 0.244728471,\ 0.090030573
\right)^\mathsf T.
$$

因为这个例子的 $S$ 对称，列归一化矩阵恰好是行归一化矩阵的转置；但它的每一列和为 1，而每一行不一定为 1。对一般非对称 score，两个结果的数值和语义都会不同。

### row-stochastic 不等于 doubly stochastic

若矩阵同时满足行和与列和都为 1，它是 doubly stochastic matrix（双随机矩阵）。attention matrix 的行 softmax 只保证：

$$
A\mathbf 1_S=\mathbf 1_T.
$$

当 $T=S$ 时，仍不能推出：

$$
\mathbf 1_T^\mathsf T A=\mathbf 1_T^\mathsf T.
$$

前面的列和 $(1.088604862,0.822790275,1.088604862)$ 已经给出反例。不要把 attention matrix 当作普通的对称相似度矩阵或双随机传输矩阵。

## 反向传播中的矩阵路径

### $C=AV$ 的微分

对 value 读取写成：

$$
C=AV.
$$

同时改变 $A$ 和 $V$ 时：

$$
\mathrm dC
=
(\mathrm dA)V
+A(\mathrm dV).
$$

令上游梯度为 $G=\partial L/\partial C$，则矩阵微分给出：

$$
\frac{\partial L}{\partial A}
=
G V^\mathsf T,
\qquad
\frac{\partial L}{\partial V}
=
A^\mathsf T G.
$$

$A^\mathsf T$ 出现在 value 的梯度中，因为一个 value 被多个 query 行共同读取。前向时按行看 $A$，反向传给 value 时按列聚合这些读取路径。

### 一个全 1 上游梯度的核对

仍使用一维 value $V=(10,20,40)^\mathsf T$，令三个输出的上游梯度都为 1：

$$
G
=
\begin{bmatrix}
1\\
1\\
1
\end{bmatrix}.
$$

则：

$$
\frac{\partial L}{\partial V}
=
A^\mathsf T G
=
\begin{bmatrix}
1.088604862\\
0.822790275\\
1.088604862
\end{bmatrix}.
$$

这个结果正好等于 $A$ 的列和。多个 query 对同一个 value 的读取会沿 value 轴累加，不会因为每一行已经归一化就消失。

对 $A$ 的梯度是：

$$
\frac{\partial L}{\partial A}
=
G V^\mathsf T
=
\begin{bmatrix}
10&20&40\\
10&20&40\\
10&20&40
\end{bmatrix}.
$$

它还不是 score $S$ 的梯度。$A$ 经过逐行 softmax，继续反向时每一行都要乘对应的 softmax Jacobian；被 mask 的位置在理想的 $-\infty$ 表示下不应恢复梯度。

### softmax 的矩阵反向仍然逐行进行

对一行 $\alpha=\operatorname{softmax}(s)$，若上游为 $g=\partial L/\partial\alpha$，则：

$$
\frac{\partial L}{\partial s}
=
\alpha\odot
\left(
g-(g^\mathsf T\alpha)\mathbf 1
\right).
$$

这里的内积 $g^\mathsf T\alpha$ 是这一行的标量，$\odot$ 是逐分量乘法。矩阵实现会对每个 query 行独立应用这个公式。[softmax 函数](../neurons-and-activations/softmax/)解释其 Jacobian 的坐标耦合，[缩放点积注意力](../attention/scaled-dot-product/)解释缩放如何在进入这个 Jacobian 前改变 score 尺度。

## 批次、多头与逻辑矩阵

### 四维张量中仍有同一张二维矩阵

批次和多头实现通常使用：

|张量|形状|位置轴|
|---|---|---|
|$Q$|$B\times h\times T\times d_k$|query 轴 $T$|
|$K$|$B\times h\times S\times d_k$|key 轴 $S$|
|$S=QK^\mathsf T/\sqrt{d_k}$|$B\times h\times T\times S$|二维 attention matrix 的批次与 head 堆叠|
|$M$|$B\times1\times T\times S$|通常沿 head 广播|
|$A$|$B\times h\times T\times S$|每个 batch/head 独立逐行归一化|
|$V$|$B\times h\times S\times d_v$|key 轴与 $A$ 的列轴对齐|
|$C=AV$|$B\times h\times T\times d_v$|query 轴与 $A$ 的行轴对齐|

每个 $b,h$ 切片都是一张 $T\times S$ 矩阵。softmax 不能把 batch 或 head 维混进归一化分母，否则不同样本或不同 head 会互相竞争。

### 矩阵元素和资源

取 $B=4$、$h=8$、$T=S=1024$、$d_k=d_v=64$：

|量|计算|结果|
|---|---|---:|
|attention matrix 元素数|$B h T S$|33,554,432|
|QK 点积的维度项|$B h T S d_k$|2,147,483,648|
|AV 加权的维度项|$B h T S d_v$|2,147,483,648|
|FP16 $A$ 矩阵字节数|$B h T S\times2$|67,108,864 bytes = 64 MiB|
|每行 softmax 竞争数|$S$|1,024|

逻辑上需要 $T\times S$ 个位置对，即使 kernel 不把完整 $A$ 写入显存，这个矩阵仍然定义了每个 query 与每个 key 的交互。[注意力复杂度](../attention/attention-complexity/)讨论如何用稀疏、分块或线性近似改变这个资源账本。

### FlashAttention 改变存储方式，不改变矩阵语义

FlashAttention 之类的 kernel 会把 $QK^\mathsf T$、mask、row-wise softmax 和 $AV$ 分成 tile，在片上存储中完成部分计算，避免把完整 attention matrix 写入显存。对给定的 $Q/K/V/M$，如果数值误差在允许范围内，逻辑结果仍然是：

$$
C
=
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{QK^\mathsf T}{\sqrt{d_k}}
+M
\right)V.
$$

因此「内存中没有一个完整的 $A$ 张量」不等于「模型没有 attention matrix」；它表示 $A$ 作为中间算子被分块重算或消费。若需要保存热图，必须额外请求或重算权重，并明确这会增加存储和传输成本。

## attention matrix 与热图

### 热图显示的是行内读取分布

把 $A$ 画成热图时，横轴应标记 key 位置，纵轴应标记 query 位置。颜色表示 $A_{t,i}$，不是 raw score $S_{t,i}$，也不是 value 的大小。绘图前要说明：

1. 是否已经加入 mask；
2. 是否使用缩放后的 score；
3. 颜色范围是每行独立归一化还是全矩阵共享；
4. padding 行列是否被裁剪；
5. head 和 batch 是否已经选择或平均。

同一个矩阵在逐行归一化色阶和全局色阶下会产生不同视觉印象。视觉上颜色更深的格子只说明该行内权重更大；跨行比较还要看每行候选数和色阶。

### 权重不是单独的因果证明

$A_{t,i}$ 描述一次前向计算中 query $t$ 从 key $i$ 读取的权重。它不单独证明 key $i$ 是输出的因果原因，因为：

- value 内容可能接近零或与其他 value 抵消；
- score 路径和 value 路径都参与输出；
- 后续输出投影、残差和非线性会改变影响；
- 改变 token 还会同时改变 Q、K、V 和其他行的分布。

如果要研究干预影响，需要固定实验条件并比较遮蔽、替换或重跑后的输出变化。热图可以帮助定位读取路径，不能替代干预实验。

### 行熵和列聚合提供两种统计视角

行熵：

$$
\mathcal H_t
=
-\sum_{i=1}^{S}A_{t,i}\log A_{t,i}
$$

描述 query $t$ 的读取是否集中。列聚合：

$$
r_i
=
\sum_{t=1}^{T}A_{t,i}
$$

描述 key $i$ 被所有 query 合计读取多少。前面的三位置矩阵给出：

|统计量|位置 1|位置 2|位置 3|
|---|---:|---:|---:|
|行熵 $\mathcal H_t$|0.832395582|1.098612289|0.832395582|
|列聚合 $r_i$|1.088604862|0.822790275|1.088604862|

行熵与列聚合回答不同问题。平均它们之前，先确认统计轴和有效长度一致。

## 稀疏和近似矩阵的边界

### 稀疏 attention 仍可保留矩阵形式

如果每个 query 只允许读取一个候选集合 $\mathcal I_t$，则：

$$
A_{t,i}=0
\quad\text{for }i\notin\mathcal I_t,
\qquad
\sum_{i\in\mathcal I_t}A_{t,i}=1.
$$

矩阵仍然是 $T\times S$ 的逻辑对象，但非零模式受限。块稀疏、局部窗口和全局 token 都可以在这个框架内表达；改变的是 mask 的结构和非零元素数量。

### 低秩或线性 attention 可能不再显式形成 $A$

一些线性 attention 变体把 kernel feature map 或求和顺序重新组织为：

$$
\sum_{i=1}^{S}
\phi(\mathbf q_t)^\mathsf T
\phi(\mathbf k_i)\mathbf v_i^\mathsf T,
$$

从而避免显式构造每个 $t,i$ 的 $T\times S$ 矩阵。这类方法可以保留「query 根据 key 读取 value」的解释框架，但不一定产生与精确 row-wise softmax 相同的 $A$。比较近似方法时，要分别报告：

1. 是否存在显式 attention matrix；
2. 是否使用同一个 score 和 softmax；
3. 是否保持相同的 mask 语义；
4. 输出误差与资源节省如何变化。

「没有存储 attention matrix」和「使用了精确的 attention matrix」是实现层与数学层的两个不同陈述。

## 失效模式与审计方法

### 把 $QK^\mathsf T$ 写成 $KQ^\mathsf T$

如果 $Q$ 有 $T$ 行、$K$ 有 $S$ 行，正确的 score 形状是 $T\times S$。$KQ^\mathsf T$ 得到 $S\times T$，它把行列语义交换了。即使 self-attention 中 $T=S$、形状仍是方阵，也会把 query/key 角色和 softmax 轴弄反。

### softmax 沿错误轴归一化

检查：

$$
\max_t\left\lvert\sum_i A_{t,i}-1\right\rvert
$$

应接近数值误差。不要检查列和来确认普通 attention 的 row-wise softmax；列和的变化本身是允许的。

### value 位置轴错位

$A$ 的列轴必须与 $V$ 的行轴对应。把 padding 或 token 排序只应用在 $K$ 而没有同步应用在 $V$，会得到形状正确但语义错位的输出。用一个 value 行设置为明显的标记值，追踪它进入哪些 query 输出，可以发现这类错位。

### mask 广播到错误的 batch/head

检查 mask 的逻辑形状和广播结果。一个样本的 padding mask 不应覆盖另一个样本；一个 head 的稀疏模式也不应在无意中覆盖全部 head。打印最终参与 softmax 的 $M$，不要只打印输入 mask。

### 全 mask 行被静默处理

全 mask 行可能被 kernel 返回 NaN、全零或任意默认值。三种结果都需要在接口中定义。若选择全零，后续 residual、LayerNorm 和 loss reduction 也要知道该行不代表一个有效 token。

### 热图省略轴标签和色阶

没有 query/key 轴标签的热图无法判断对角线、因果三角或跨序列方向。没有说明色阶范围的热图无法比较不同 head、不同 batch 或不同层。保存可复现实验时同时保存 mask、token 索引、softmax 轴和色阶规则。

### 一份最小矩阵审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|score 形状|$T\times S$|Q/K 行数、transpose|
|行列语义|行是 query，列是 key|位置轴命名、热图标签|
|行归一化|有效 key 权重和为 1|softmax axis、全 mask 行|
|非法位置|权重为 0，梯度不泄漏|mask sentinel、广播|
|value 对齐|$A$ 的列轴对应 $V$ 的行轴|padding、排序、缓存|
|输出形状|$C$ 为 $T\times d_v$|矩阵乘法内维|
|列聚合|总和为 $T$，各列不必为 1|不要误判为列随机|
|资源|逻辑元素数为 $B h T S$|cross-attention 的 $T/S$|
|实现方式|是否物化 $A$ 有明确记录|FlashAttention、稀疏 kernel|

## 相关词条

- [自注意力](../attention/self-attention/)
- [缩放点积注意力](../attention/scaled-dot-product/)
- [Attention 作为检索](../attention/attention-as-retrieval/)
- [多头注意力](../attention/multi-head-attention/)
- [因果掩码](../attention/causal-masking/)
- [交叉注意力](../attention/cross-attention/)
- [注意力复杂度](../attention/attention-complexity/)
- [矩阵乘法](../linear-algebra/matrix-multiplication/)
- [softmax 函数](../neurons-and-activations/softmax/)
