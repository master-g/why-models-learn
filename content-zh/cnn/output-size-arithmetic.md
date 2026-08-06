---
title: "输出尺寸算术：逐层核对卷积空间形状"
tags: ["why-models-learn"]
---

输出尺寸算术是把卷积、池化或转置卷积的配置翻译成空间形状的过程。它回答的不是“这一层大概会缩小多少”，而是：窗口从哪些起点出发，最后一个完整窗口能否放下，输出索引对应原输入的哪个位置。只要把有效核尺寸、四侧 padding、stride 和取整规则逐项写出，卷积网络的空间 shape 就可以像账本一样逐层核对。

本文先只看一个空间轴，再推广到二维；随后把同一套窗口算术用于池化，并单独推导转置卷积的形状关系。全文使用深度学习库常见的不翻转滑动窗口记号。若底层算子采用数学卷积，核是否翻转会影响数值，但不会改变下面的空间尺寸公式。关于 padding、stride 和 dilation 各自如何改变坐标，参见[步幅、填充与膨胀](../cnn/stride-padding-dilation/)。

![输出尺寸算术的逐层账本：卷积、池化与膨胀卷积把空间形状从 32×32 推到 8×8，右侧给出转置卷积的反向尺寸关系](/assets/cnn/svg/output-size-arithmetic.1.svg)

## 先只算一个空间轴

设输入在某一空间轴上的长度为 $I$，核的离散长度为 $K$，dilation 为 $D$，stride 为 $S$，左、右 padding 分别为 $P_{\mathrm{left}}$ 和 $P_{\mathrm{right}}$。先不要把高度和宽度混在一个公式里；一个轴的边界条件已经包含了全部关键问题。

### 有效核尺寸

dilation 不会增加可学习权重的数量，却会拉开核内相邻权重读取的输入坐标。有效核尺寸是

$$
K_{\mathrm{eff}}
=D(K-1)+1.
$$

当 $D=1$ 时，$K_{\mathrm{eff}}=K$。当 $K=3,D=2$ 时，三个权重读取相对坐标 $0,2,4$，因此有效覆盖宽度是 $5$，而不是权重个数 $3$。

### Padded 长度与窗口起点

零 padding 后，数组在这个轴上的长度为

$$
I^{P}
=I+P_{\mathrm{left}}+P_{\mathrm{right}}.
$$

第 $r$ 个输出位置的窗口起点为

$$
q_r=rS.
$$

该窗口最后读取的坐标为

$$
q_r+K_{\mathrm{eff}}-1.
$$

它必须落在 padded 数组的最后一个位置 $I^{P}-1$ 以内。因此最后一个合法输出索引满足

$$
rS+K_{\mathrm{eff}}-1
\le I^{P}-1.
$$

解出可用的输出位置数，得到最常用的卷积或池化尺寸公式：

$$
O
=\left\lfloor
\frac{
I+P_{\mathrm{left}}+P_{\mathrm{right}}-K_{\mathrm{eff}}
}{S}
\right\rfloor+1.
$$

把 $K_{\mathrm{eff}}=D(K-1)+1$ 展开后，也可以写成

$$
O
=\left\lfloor
\frac{
I+P_{\mathrm{left}}+P_{\mathrm{right}}-D(K-1)-1
}{S}
\right\rfloor+1.
$$

每一项都对应一个可检查的事实：

- $I+P_{\mathrm{left}}+P_{\mathrm{right}}$ 是真正被滑动窗口访问的长度；
- $D(K-1)+1$ 是窗口跨度，不是参数个数；
- $S$ 是相邻窗口起点的间隔；
- floor 表示最后剩下的不完整窗口被丢弃。

如果分子小于零，说明连一个完整窗口都放不下。实现应把它当成非法 shape，而不是把负数输出长度静默改成零。

## 公式的三个基本特例

### Valid：只保留完全落在输入内的窗口

valid 不添加边界 padding，因此

$$
P_{\mathrm{left}}=P_{\mathrm{right}}=0,
$$

输出长度为

$$
O_{\mathrm{valid}}
=\left\lfloor
\frac{I-K_{\mathrm{eff}}}{S}
\right\rfloor+1.
$$

当 $S=1,D=1$ 时，这就是熟悉的 $I-K+1$。valid 的含义是窗口不能看到输入之外的坐标；它不等于“数值一定更可靠”，只是选择了更严格的窗口集合。

### Full：保留所有发生重叠的窗口

在 stride 为 $1$ 的一维数学卷积里，若希望保留从只碰到输入第一个元素到只碰到最后一个元素的所有窗口，可在两侧各添加 $K_{\mathrm{eff}}-1$ 个 padding：

$$
P_{\mathrm{left}}
=P_{\mathrm{right}}
=K_{\mathrm{eff}}-1.
$$

代入尺寸公式：

$$
O_{\mathrm{full}}
=I+K_{\mathrm{eff}}-1.
$$

full 不是所有深度学习库默认暴露的模式；它更适合用来解释离散卷积的边界集合和反向传播中的全尺寸中间结果。

### Same：先决定目标长度，再反推 padding

当 $S=1$ 且 $K_{\mathrm{eff}}$ 为奇数时，常见的 symmetric same padding 是

$$
P_{\mathrm{left}}
=P_{\mathrm{right}}
=\frac{K_{\mathrm{eff}}-1}{2},
$$

于是 $O=I$。但当 $S>1$、偶数核或输入长度不能整除 stride 时，same 不能只凭“左右各补一半”理解。常见语义是先规定

$$
O_{\mathrm{same}}
=\left\lceil\frac{I}{S}\right\rceil,
$$

再计算总 padding

$$
P_{\mathrm{total}}
=\max\bigl((O_{\mathrm{same}}-1)S+K_{\mathrm{eff}}-I,\ 0\bigr).
$$

最后把 $P_{\mathrm{total}}$ 分配给左右边界。若总数为奇数，许多实现把多出的一个放在右侧，但具体约定必须以算子文档和实际 shape 为准。same 的输出长度相同，不保证不同框架的输出相位相同。

## 用具体数字检查每一位

取一维输入长度 $I=7$、核长度 $K=3$、$D=1$、左右各补一个零、$S=2$。有效核尺寸和 padded 长度分别是

$$
K_{\mathrm{eff}}=1(3-1)+1=3,
\qquad
I^{P}=7+1+1=9.
$$

因此

$$
O
=\left\lfloor\frac{9-3}{2}\right\rfloor+1
=4.
$$

四个窗口起点是

$$
q_0=0,\qquad q_1=2,\qquad q_2=4,\qquad q_3=6.
$$

若 padded 输入写成

$$
[0,x_0,x_1,x_2,x_3,x_4,x_5,x_6,0],
$$

四个窗口分别覆盖

$$
[0,x_0,x_1],\quad
[x_1,x_2,x_3],\quad
[x_3,x_4,x_5],\quad
[x_5,x_6,0].
$$

最后一个窗口虽然触碰了右侧 padding，仍然是完整的三个位置，所以它合法；起点 $8$ 的窗口会超出 padded 数组，不能再生成一个输出。这个例子也说明了为什么只写“stride 为 $2$，长度大约减半”不够：精确答案来自最后一个合法窗口的位置。

如果把同一组参数改成 dilation $D=2$，有效核尺寸变成

$$
K_{\mathrm{eff}}=2(3-1)+1=5,
$$

输出长度变为

$$
O
=\left\lfloor\frac{9-5}{2}\right\rfloor+1
=3.
$$

权重个数没有变化，但窗口跨度增加了两格，因此尺寸发生了变化。

## 二维输出尺寸要沿两个轴分别计算

二维输入的高度和宽度为 $H,W$，核尺寸为 $K_h,K_w$。四侧 padding、两个方向的 stride 和 dilation 分别记为

$$
P_{\mathrm{top}},P_{\mathrm{bottom}},
P_{\mathrm{left}},P_{\mathrm{right}},
\qquad
S_h,S_w,
\qquad
D_h,D_w.
$$

两个方向的有效核尺寸是

$$
K_h^{\mathrm{eff}}
=D_h(K_h-1)+1,
\qquad
K_w^{\mathrm{eff}}
=D_w(K_w-1)+1.
$$

高度和宽度必须分别计算：

$$
H_{\mathrm{out}}
=\left\lfloor
\frac{
H+P_{\mathrm{top}}+P_{\mathrm{bottom}}-K_h^{\mathrm{eff}}
}{S_h}
\right\rfloor+1,
$$

$$
W_{\mathrm{out}}
=\left\lfloor
\frac{
W+P_{\mathrm{left}}+P_{\mathrm{right}}-K_w^{\mathrm{eff}}
}{S_w}
\right\rfloor+1.
$$

例如，设

$$
H=5,\quad W=6,\quad
(K_h,K_w)=(3,2),
\quad
(P_{\mathrm{top}},P_{\mathrm{bottom}})=(1,1),
\quad
(P_{\mathrm{left}},P_{\mathrm{right}})=(0,0),
\quad
S_h=S_w=2.
$$

因为 $K_h^{\mathrm{eff}}=3$、$K_w^{\mathrm{eff}}=2$，所以

$$
H_{\mathrm{out}}
=\left\lfloor\frac{5+1+1-3}{2}\right\rfloor+1
=3,
\qquad
W_{\mathrm{out}}
=\left\lfloor\frac{6-2}{2}\right\rfloor+1
=3.
$$

输出空间形状是 $3\times3$。高度有两侧 padding，宽度没有；把同一个 padding 数字同时代入两个方向会得到错误答案。

对于 batch、channel 等非空间维度，尺寸公式不负责改变它们。普通二维卷积的输出通常写成

$$
(N,C_{\mathrm{out}},H_{\mathrm{out}},W_{\mathrm{out}})
$$

或 channels-last 的

$$
(N,H_{\mathrm{out}},W_{\mathrm{out}},C_{\mathrm{out}}).
$$

这里的 $N$ 和通道数来自张量布局及卷积参数；只有 $H_{\mathrm{out}},W_{\mathrm{out}}$ 由窗口算术决定。

## 池化使用相同的窗口算术，但没有可学习核

最大池化和平均池化也要回答三个相同的问题：窗口从哪里开始、相邻起点隔多远、最后一个窗口是否完整。因此，在有效核尺寸为 $K_{\mathrm{eff}}$ 时，floor 模式仍然使用

$$
O_{\mathrm{pool}}
=\left\lfloor
\frac{
I+P_{\mathrm{left}}+P_{\mathrm{right}}-K_{\mathrm{eff}}
}{S}
\right\rfloor+1.
$$

池化和卷积的尺寸相同，不代表它们的边界数值语义相同。卷积窗口把输入与可学习权重相乘后求和；最大池化选择窗口最大值，平均池化还要明确 padding 是否参与除法。平均池化若把零 padding 也计入分母，边缘会被压低；若只按有效元素求平均，边缘的归一化又不同。

一些库提供 ceil mode，让只差一个不完整尾窗口的输入也产生输出。常见的抽象公式是

$$
O_{\mathrm{ceil}}
=\left\lceil
\frac{
I+P_{\mathrm{left}}+P_{\mathrm{right}}-K_{\mathrm{eff}}
}{S}
\right\rceil+1.
$$

但 ceil mode 往往还涉及尾窗口是否允许起点落入右侧 padding，以及是否额外补齐输入。不能只把 floor 换成 ceil 就假设数值行为完全一致；应同时检查输出 shape、最后一个窗口的起点和 padding 参与规则。

## Same 的目标长度和总 padding

把 same 当作一个目标约束更容易审计。给定输入长度 $I$ 和 stride $S$，先取目标长度

$$
O_{\mathrm{target}}
=\left\lceil\frac{I}{S}\right\rceil.
$$

然后让最后一个窗口恰好有机会覆盖到目标位置，所需总 padding 为

$$
P_{\mathrm{total}}
=\max\bigl(
(O_{\mathrm{target}}-1)S+K_{\mathrm{eff}}-I,\ 0
\bigr).
$$

将它拆成两侧：

$$
P_{\mathrm{left}}
=\left\lfloor\frac{P_{\mathrm{total}}}{2}\right\rfloor,
\qquad
P_{\mathrm{right}}
=P_{\mathrm{total}}-P_{\mathrm{left}}.
$$

这是一种常见的 top-left 对齐约定；某些实现会把额外 padding 放在左侧，或在不同维度采用不同规则。例外最容易出现在偶数核、stride 大于一和 dilation 大于一的组合中。

以 $I=7,S=2,K_{\mathrm{eff}}=3$ 为例：

$$
O_{\mathrm{target}}
=\left\lceil\frac{7}{2}\right\rceil
=4,
\qquad
P_{\mathrm{total}}
=(4-1)2+3-7
=2.
$$

左右各补一个正好得到前面的四个窗口。如果 $I=8,S=3,K_{\mathrm{eff}}=4$，则

$$
O_{\mathrm{target}}
=\left\lceil\frac{8}{3}\right\rceil
=3,
\qquad
P_{\mathrm{total}}
=(3-1)3+4-8
=2.
$$

总 padding 仍为 $2$，但如何分到两侧会决定输出相位。

## 多层网络要建立 shape ledger

只在脑中连续代入公式，很容易把上一层的输出长度、这一层的输入长度和某个 dilation 的有效核尺寸混在一起。更可靠的做法是为每一层保留一行 shape ledger，记录输入、有效核、padding、stride 和输出。

考虑一个二维网络，输入空间为 $32\times32$：

| 层 | 输入空间 | 核与参数 | 高度计算 | 输出空间 |
| --- | --- | --- | --- | --- |
| 输入 | — | — | — | $32\times32$ |
| 卷积 1 | $32\times32$ | $K=3,P=1,S=1,D=1$ | $(32+2-3)/1+1=32$ | $32\times32$ |
| 池化 | $32\times32$ | $K=2,P=0,S=2$ | $\lfloor(32-2)/2\rfloor+1=16$ | $16\times16$ |
| 卷积 2 | $16\times16$ | $K=3,P=1,S=2,D=1$ | $\lfloor(16+2-3)/2\rfloor+1=8$ | $8\times8$ |
| 卷积 3 | $8\times8$ | $K=3,P=2,S=1,D=2$ | $\lfloor(8+4-5)/1\rfloor+1=8$ | $8\times8$ |

最后一层虽然核仍然只有 $3\times3$，但 dilation 让有效核尺寸变成 $5\times5$；padding $2$ 抵消了这个跨度，所以 stride 为 $1$ 时空间尺寸保持为 $8\times8$。

shape ledger 至少应同时记录：

- 输入和输出的空间尺寸；
- 每个方向的 kernel、dilation、stride；
- 四侧 padding，而不是只写一个模糊的 padding；
- floor 或 ceil 模式；
- channel 数和数据布局；
- 若有残差相加、跳跃连接或拼接，记录参与操作的所有分支 shape。

残差连接要求相加两侧的空间 shape 和 channel shape 对齐；拼接通常只允许被拼接轴不同，其余轴必须相等。尺寸公式通过但分支 shape 不一致，仍然不能执行张量操作。

## 转置卷积使用相反方向的尺寸关系

转置卷积经常被描述成“把卷积放大”，但它的尺寸公式不是普通卷积公式倒过来解一个整数。对一个空间轴，输入长度为 $I$、kernel 为 $K$、dilation 为 $D$、stride 为 $S$、两侧 padding 为 $P_{\mathrm{left}},P_{\mathrm{right}}$，额外的 output padding 为 $O_{\mathrm{pad}}$ 时，常见形状关系是

$$
O_{\mathrm{trans}}
=(I-1)S
-P_{\mathrm{left}}
-P_{\mathrm{right}}
+D(K-1)
+O_{\mathrm{pad}}
+1.
$$

令

$$
K_{\mathrm{eff}}=D(K-1)+1,
$$

同一公式也可写成

$$
O_{\mathrm{trans}}
=(I-1)S
-P_{\mathrm{left}}
-P_{\mathrm{right}}
+K_{\mathrm{eff}}
+O_{\mathrm{pad}}.
$$

二维时，高度和宽度分别使用各自的参数：

$$
H_{\mathrm{out}}
=(H-1)S_h
-P_{\mathrm{top}}
-P_{\mathrm{bottom}}
+D_h(K_h-1)
+O_{\mathrm{pad},h}
+1,
$$

$$
W_{\mathrm{out}}
=(W-1)S_w
-P_{\mathrm{left}}
-P_{\mathrm{right}}
+D_w(K_w-1)
+O_{\mathrm{pad},w}
+1.
$$

例如，$I=3,K=3,D=1,S=2$，左右 padding 都为 $1$：

$$
O_{\mathrm{trans}}
=(3-1)2-1-1+(3-1)+O_{\mathrm{pad}}+1
=5+O_{\mathrm{pad}}.
$$

若允许 $O_{\mathrm{pad}}\in\{0,1\}$，输出长度可以是 $5$ 或 $6$。这正是 output padding 的用途：在多个输入 shape 都能对应相近输出关系时，补充一个离散的 shape 选择。它不是在输出右侧简单追加一列真实零，也不是用来替代普通 padding。

## 为什么转置卷积不是普通卷积的逆

把一次线性卷积写成矩阵乘法：

$$
y=Ax.
$$

对上游梯度或输入信号施加转置算子，得到

$$
z'=A^{\mathsf T}z.
$$

转置卷积实现的是这个线性算子转置后的乘法，因此它对应反向传播中把梯度从输出空间传播回输入空间的结构。它通常会扩大空间尺寸，但它不是求解 $A^{-1}$：

- $A$ 可能不是方阵，输入和输出维度本来就不同；
- 即使是方阵，$A^{\mathsf T}$ 也只有在特殊条件下才等于逆矩阵；
- stride 和边界重叠会让多个输入位置向同一输出位置累加，转置算子保留这种累加结构；
- output padding 只选择离散输出 shape，不会恢复被 stride 丢弃的全部信息。

因此，生成模型或解码器中的转置卷积仍需要通过 shape ledger 检查每一层的输出，以及通过数值实验检查重叠累加、棋盘格伪影和边界行为。

## 四侧 padding 和 output padding 的 shape 审计

同一个词 padding 在普通卷积和转置卷积公式中的符号方向不同。审计一层时，可以先把参数放进下面的对照表：

| 项目 | 普通卷积 | 转置卷积 | 审计问题 |
| --- | --- | --- | --- |
| 输入长度 | $I$ | $I$ | 这一层实际接收的 shape 是多少 |
| 有效核 | $D(K-1)+1$ | $D(K-1)+1$ | dilation 是否被计入跨度 |
| stride | 窗口起点间隔 | 输出位置间隔与重叠关系 | 是否把 $S$ 当成连续缩放比例 |
| padding | 加到输入长度分子 | 从转置输出关系中减去 | 左右两侧是否分别记录 |
| 取整 | 通常 floor 或 same 约定 | 通常由整数形状关系直接得到 | 框架是否另有 output size 参数 |
| output padding | 通常没有 | 选择额外的离散输出长度 | 是否误当成真实零 padding |

最小验收可以按这个顺序进行：

1. 写出输入空间尺寸和布局；
2. 展开有效核尺寸；
3. 分别写四侧 padding、两个方向 stride 和 dilation；
4. 计算最后一个合法窗口或转置后的首尾位置；
5. 将结果写回下一层的输入；
6. 检查分支相加、拼接和上采样目标是否一致；
7. 再看数值边界、相位和是否有不完整尾窗口。

## 常见失效模式

### 把 kernel size 当成有效核尺寸

当 dilation 大于一时，$K$ 是权重数量，$D(K-1)+1$ 才是空间覆盖范围。漏掉 dilation 会使输出尺寸和感受野同时算错。

### 只记录一个 padding 数字

对称 padding 可以简写为一个数字，但 same、偶数核、动态 shape 和转置卷积经常需要四侧值。只保留一个总数会丢失输出相位。

### 忘记 floor 或 ceil

当分子不能被 stride 整除时，最后一个不完整窗口是否保留决定了输出长度。floor 和 ceil mode 还可能改变尾窗口是否接触 padding。

### 把 stride 当成无损缩放

stride 只选择窗口起点；它不自动插值、不自动抗混叠，也不能恢复被跳过的窗口。需要下采样时，应把滤波和采样语义分开记录。

### 误把转置卷积套进普通卷积公式

普通卷积通常是在 padded 输入上寻找完整窗口；转置卷积是转置线性算子后的重叠累加。两者的 padding 项方向相反，output padding 也只属于转置关系。

### 只检查最终 shape

最终 shape 正确，不代表中间层对齐。残差相加可能出现半个像素的相位差，分支拼接可能在某层才暴露尺寸差，转置卷积还可能在重叠区域产生不均匀覆盖。

## 读输出尺寸公式的验收顺序

面对一个新的卷积或池化配置，可以把检查压缩成一行：

$$
\text{输入长度}
\longrightarrow
\text{有效核尺寸}
\longrightarrow
\text{四侧 padding 后长度}
\longrightarrow
\text{stride 下的最后合法起点}
\longrightarrow
\text{floor、ceil 或 same 目标}
\longrightarrow
\text{输出长度}.
$$

二维时把这条链沿高度和宽度各走一次；多层时把每一层的输出接到下一层的输入。对转置卷积则改用首尾位置关系，并明确 output padding 的离散选择。这样，输出尺寸不再是框架打印出来的黑箱数字，而是可以从坐标和边界逐项复算的结果。

## 相关词条

[步幅、填充与膨胀](../cnn/stride-padding-dilation/)

[二维卷积](../cnn/convolution-2d/)

[互相关与卷积](../cnn/cross-correlation-vs-convolution/)

[离散卷积](../cnn/discrete-convolution/)

[池化](../cnn/pooling/)

[卷积神经网络](../cnn/cnn/)

[残差连接](../cnn/residual-connections/)

[卷积作为 Toeplitz 结构](../cnn/convolution-as-toeplitz/)

[不变性与等变性](../cnn/invariance-and-equivariance/)
