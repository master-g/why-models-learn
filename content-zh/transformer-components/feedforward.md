---
title: "前馈网络：逐 token 的非线性特征变换"
tags: ["why-models-learn"]
---

前馈网络（feedforward network，FFN）是 Transformer 中对每个 token 的表示独立执行的非线性特征变换。对一个 token 的列向量 $h\in\mathbb R^{d_{\mathrm{model}}}$，最常见的两层形式为

$$
F(h)
=W_2\phi(W_1h+b_1)+b_2.
$$

$W_1$ 先把表示从 $d_{\mathrm{model}}$ 扩展到中间宽度 $d_{\mathrm{ffn}}$，$\phi$ 在每个中间坐标上施加非线性，$W_2$ 再把结果压回 $d_{\mathrm{model}}$。FFN 不负责 token 之间的信息读取；在 Transformer block 中，attention 先把上下文写入 [残差流](../transformer-components/residual-streams/)，FFN 再逐 token 改变当前表示。最终增量仍然要回到 residual stream 的宽度，写成

$$
x_{l+1}=x_l+F(h_l).
$$

本文先固定 FFN 的输入输出和矩阵 shape，再用一个二维例子逐项计算仿射层、ReLU、输出和局部 Jacobian，推导反向梯度，说明中间宽度、激活稀疏性、参数量、内存和数值精度之间的关系，最后给出 Transformer 集成和核验协议。

![前馈网络把一个 token 的表示扩展、逐坐标激活，再压回 residual stream 宽度](/assets/transformer-components/svg/feedforward.1.svg)

## 先固定 FFN 的 shape 合同

### 一个 token 的列向量约定

以下统一采用列向量。设

$$
h\in\mathbb R^d,
\qquad
d=d_{\mathrm{model}},
\qquad
d_{\mathrm{ffn}}=m.
$$

各中间量的 shape 如下：

| 符号 | shape | 作用 |
| --- | --- | --- |
| $h$ | $\mathbb R^d$ | 一个 token 的输入表示 |
| $W_1$ | $\mathbb R^{m\times d}$ | 把输入宽度扩展到 $m$ |
| $b_1$ | $\mathbb R^m$ | 第一层的逐中间坐标偏置 |
| $z$ | $\mathbb R^m$ | 激活函数的输入 |
| $a$ | $\mathbb R^m$ | 激活后的中间表示 |
| $W_2$ | $\mathbb R^{d\times m}$ | 把中间宽度压回 $d$ |
| $b_2$ | $\mathbb R^d$ | 输出层的逐特征偏置 |
| $y$ | $\mathbb R^d$ | 写回 residual stream 的分支输出 |

前向计算可以拆成三步：

$$
z=W_1h+b_1,
$$

$$
a=\phi(z),
$$

$$
y=W_2a+b_2.
$$

因此

$$
\mathbb R^d
\xrightarrow{\ W_1\ }
\mathbb R^m
\xrightarrow{\ \phi\ }
\mathbb R^m
\xrightarrow{\ W_2\ }
\mathbb R^d.
$$

中间宽度 $m$ 可以大于、等于或小于 $d$。Transformer 中通常取 $m>d$，但「扩展后再压回」是常见设计，不是加法合同本身的要求。

### 批量和序列的最后一轴

对于 batch 和序列，输入是

$$
H\in\mathbb R^{B\times T\times d}.
$$

FFN 对每个固定的 $(b,t)$ 独立执行：

$$
Y_{b,t,:}=F(H_{b,t,:}),
$$

所以

$$
Y\in\mathbb R^{B\times T\times d}.
$$

最后一轴保持为 $d$，但中间计算会出现

$$
Z,A\in\mathbb R^{B\times T\times m}.
$$

将前两个轴展平为 $N=BT$ 个 token 后，批量矩阵形式为

$$
X\in\mathbb R^{N\times d},
$$

$$
Z=XW_1^{\mathsf T}+\mathbf 1 b_1^{\mathsf T},
$$

$$
A=\phi(Z),
$$

$$
Y=AW_2^{\mathsf T}+\mathbf 1 b_2^{\mathsf T}.
$$

这里的 $X$ 是行向量堆叠，和前面的单 token 列向量公式只是转置约定不同。实现中常把 $B\times T$ 展平为 GEMM 的 batch 行，但不能因此把不同 token 的行混合到同一次特征变换中。

### FFN 不会自动改变 token 数量

FFN 只变换最后一轴：

$$
(B,T,d)
\longrightarrow
(B,T,m)
\longrightarrow
(B,T,d).
$$

它不会把 $(B,T,d)$ 变成 $(B,T',d)$，也不会在 token 轴上执行求和。若输出的 $T$ 发生变化，变化来自分词、池化、下采样或其他算子，不来自普通 FFN。

## 一个二维 FFN 的数值例子

### 固定输入和参数

取

$$
h=
\begin{pmatrix}
1\\
-2
\end{pmatrix},
\qquad
d=2,
\qquad
m=3.
$$

为了把 shape 和激活效果分开观察，先令两个偏置都为零：

$$
b_1=
\begin{pmatrix}
0\\
0\\
0
\end{pmatrix},
\qquad
b_2=
\begin{pmatrix}
0\\
0
\end{pmatrix}.
$$

第一层矩阵为

$$
W_1=
\begin{pmatrix}
1&0\\
0&1\\
1&-1
\end{pmatrix},
$$

第二层矩阵为

$$
W_2=
\begin{pmatrix}
1&0&0.5\\
0&1&-1
\end{pmatrix}.
$$

### 第一层产生三个中间坐标

矩阵乘法给出

$$
z=W_1h
=
\begin{pmatrix}
1\\
-2\\
3
\end{pmatrix}.
$$

三个中间坐标的来源分别是

$$
z_1=1\cdot1+0\cdot(-2)=1,
$$

$$
z_2=0\cdot1+1\cdot(-2)=-2,
$$

$$
z_3=1\cdot1+(-1)\cdot(-2)=3.
$$

这一步已经把二维输入投影到三维中间空间，但仍然是线性映射。

### ReLU 选择激活区域

取 $\phi(z)=\operatorname{ReLU}(z)$：

$$
\operatorname{ReLU}(u)=
\begin{cases}
u,&u>0,\\
0,&u\le 0.
\end{cases}
$$

逐坐标应用得到

$$
a=\operatorname{ReLU}(z)
=
\begin{pmatrix}
1\\
0\\
3
\end{pmatrix}.
$$

第 2 个中间坐标被置零，第 1、3 个坐标保持不变。激活函数没有混合中间坐标；坐标之间的线性混合发生在 $W_1$ 和 $W_2$。

### 第二层压回两个输出坐标

最后

$$
y=W_2a
=
\begin{pmatrix}
1&0&0.5\\
0&1&-1
\end{pmatrix}
\begin{pmatrix}
1\\
0\\
3
\end{pmatrix}
=
\begin{pmatrix}
2.5\\
-3
\end{pmatrix}.
$$

这个例子完整经过

$$
\begin{pmatrix}
1\\
-2
\end{pmatrix}
\longrightarrow
\begin{pmatrix}
1\\
-2\\
3
\end{pmatrix}
\longrightarrow
\begin{pmatrix}
1\\
0\\
3
\end{pmatrix}
\longrightarrow
\begin{pmatrix}
2.5\\
-3
\end{pmatrix}.
$$

输出回到二维空间，才可以和二维 residual stream 逐元素相加。中间宽度为 3 不会把 residual stream 永久变成三维。

### 数值账本

| 对象 | 数值 |
| --- | --- |
| $\lVert h\rVert_2$ | $2.236067977500$ |
| $\lVert z\rVert_2$ | $3.741657386774$ |
| $\lVert a\rVert_2$ | $3.162277660168$ |
| $\lVert y\rVert_2$ | $3.905124837953$ |
| 激活坐标 | 第 1、3 个为正，第 2 个为零 |

输出范数不等于输入范数。FFN 学习的是特征变换，不是一个保持长度的正交映射。

## 为什么需要非线性激活

### 两个线性层可以合并

如果把激活换成恒等映射 $\phi(z)=z$，则

$$
F(h)=W_2(W_1h+b_1)+b_2.
$$

展开得到

$$
F(h)
=(W_2W_1)h+(W_2b_1+b_2).
$$

这仍然只是一个仿射映射。中间宽度 $m$ 可以增加参数数量，但不会增加函数的非线性表达能力。

### 激活让不同区域使用不同线性映射

ReLU 把输入空间按 $z_j=0$ 分成多个区域。对一个固定激活模式 $D$：

$$
D=\operatorname{diag}
\bigl(\mathbf 1_{z_1>0},\ldots,\mathbf 1_{z_m>0}\bigr),
$$

$$
a=Dz,
$$

$$
F(h)=W_2D(W_1h+b_1)+b_2.
$$

在这个区域内部，FFN 仍是一个仿射映射；跨过某个激活边界后，$D$ 改变，使用的有效线性映射也随之改变。整个网络因此可以拼接多个局部线性区域。

二维例子在 $z=(1,-2,3)$ 处的激活矩阵为

$$
D=
\begin{pmatrix}
1&0&0\\
0&0&0\\
0&0&1
\end{pmatrix}.
$$

如果输入改变到使 $z_1$ 变成负数，第一条中间特征也会被关闭，输出函数就不再使用同一个局部矩阵。

### 线性宽度和非线性宽度承担不同职责

| 部分 | 主要职责 | 不能单独完成的事情 |
| --- | --- | --- |
| $W_1$ | 把输入投影到中间特征坐标 | 不能产生分段或平滑非线性 |
| $\phi$ | 选择、压缩或重标定中间坐标 | 不能跨坐标组合输入 |
| $W_2$ | 把中间特征组合回输出坐标 | 没有激活时只能与 $W_1$ 合并成一层 |

FFN 的非线性来自三者的组合。只增加 $m$ 而移除 $\phi$，不会得到同等的函数族。

## 激活函数和局部行为

### ReLU 的激活区域

ReLU 的导数在非零点为

$$
\operatorname{ReLU}'(u)=
\begin{cases}
1,&u>0,\\
0,&u<0.
\end{cases}
$$

$u=0$ 处不可导，具体实现会选择一个次梯度约定。一个中间坐标长期处于负区间时，它对当前 token 的局部梯度为零，可能形成 dead unit。

对于二维例子，$z_2=-2$，所以第 2 个中间坐标的当前导数为零。它并非永久失效；参数或输入变化把 $z_2$ 推过零点后，该坐标可以重新参与计算。

### GELU 和平滑门控

GELU 用高斯分布的累积分布函数平滑地缩放输入：

$$
\operatorname{GELU}(u)=u\Phi(u).
$$

常用近似为

$$
\operatorname{GELU}(u)
\approx
\frac12u
\left(
1+\tanh\left(
\sqrt{\frac2\pi}
\left(u+0.044715u^3\right)
\right)
\right).
$$

它不会像 ReLU 那样在负半轴上完全置零，但负值通常受到较小的连续权重。平滑激活改变的是局部导数，不改变 FFN 需要在最后一轴回到 $d$ 的 shape 合同。

### 常见激活的边界

| 激活 | 定义或特征 | 主要核验点 |
| --- | --- | --- |
| ReLU | $\max(0,u)$ | 负区间梯度为零，检查 dead unit |
| GELU | $u\Phi(u)$ | 负值保留但被平滑缩放，检查近似公式 |
| SiLU/Swish | $u\sigma(u)$ | 负区间仍有连续输出，检查 sigmoid 数值范围 |
| 门控激活 | 一个分支调制另一个分支 | 检查两条中间宽度和逐元素乘法 |

SwiGLU 把门控结构单独展开，见 [SwiGLU：门控 FFN](../transformer-components/swiglu-ffn/)。本篇的 shape、梯度和内存账本先以两层 FFN 为基准。

### 激活会改变统计量

即使 $z$ 的各坐标均值为零，ReLU 后的 $a$ 也通常具有非零均值：

$$
\mathbb E[\operatorname{ReLU}(z)]
\ne
\operatorname{ReLU}(\mathbb E[z]).
$$

因此不能只根据输入 $h$ 的均值和方差推断输出 $y$。需要分别记录 $z$、$a$ 和 $y$ 的统计量，并说明采用了哪一种激活。

## FFN 的局部 Jacobian

### 从链式法则展开

令

$$
z=W_1h+b_1,
\qquad
a=\phi(z),
\qquad
y=W_2a+b_2.
$$

设 $D_\phi(z)$ 是激活函数在 $z$ 处的逐坐标导数矩阵，则

$$
\frac{\partial z}{\partial h}=W_1,
$$

$$
\frac{\partial a}{\partial z}=D_\phi(z),
$$

$$
\frac{\partial y}{\partial a}=W_2.
$$

按从右到左的链式法则，

$$
J_F(h)
=\frac{\partial y}{\partial h}
=W_2D_\phi(z)W_1.
$$

ReLU 情况下 $D_\phi(z)$ 是由 0 和 1 构成的对角矩阵。GELU 或 SiLU 情况下，对角元素替换为对应的连续导数。

### 二维例子的 Jacobian

在 $z=(1,-2,3)$ 处，

$$
D=
\begin{pmatrix}
1&0&0\\
0&0&0\\
0&0&1
\end{pmatrix}.
$$

先计算

$$
DW_1=
\begin{pmatrix}
1&0\\
0&0\\
1&-1
\end{pmatrix}.
$$

于是

$$
J_F
=W_2DW_1
=
\begin{pmatrix}
1.5&-0.5\\
-1&1
\end{pmatrix}.
$$

取一个仍留在同一激活区域内的输入扰动

$$
\delta h=
\begin{pmatrix}
0.01\\
-0.02
\end{pmatrix}.
$$

三个 pre-activation 变为

$$
z+\delta z
=
\begin{pmatrix}
1.01\\
-2.02\\
3.03
\end{pmatrix},
$$

激活模式没有改变。局部线性关系给出

$$
J_F\delta h
=
\begin{pmatrix}
0.025\\
-0.03
\end{pmatrix}.
$$

在不跨越激活边界时，这个局部预测与直接重新执行前向计算一致。若扰动足够大使某个 $z_j$ 穿过零点，单个 Jacobian 只能描述边界一侧。

### 中间宽度和 Jacobian 的秩

由于

$$
J_F=W_2D_\phi W_1,
$$

有

$$
\operatorname{rank}(J_F)
\le
\min\bigl(d,m,k\bigr),
$$

其中 $k$ 是当前激活导数非零的中间坐标数量。即使 $m>d$，如果当前只有很少的中间单元参与，局部映射仍可能秩亏。

对 ReLU，$k$ 是当前 token 的正 pre-activation 数量。逐 token 的 active fraction 可以作为诊断信号，但它不能单独决定函数质量：激活单元的权重方向和输出组合也会影响 Jacobian。

### 与 residual Jacobian 的关系

如果 FFN 输入为 $N(x_l)$，写回为

$$
x_{l+1}=x_l+F\bigl(N(x_l)\bigr),
$$

则

$$
\frac{\partial x_{l+1}}{\partial x_l}
=I+J_FJ_N.
$$

这里 $J_F$ 是 FFN 的局部 Jacobian，外侧 $I$ 来自 residual stream 的 shortcut。FFN 本身的局部秩和整个 block 的梯度通路不是同一个对象；需要同时记录分支 Jacobian 和加法后的 Jacobian。pre-norm 与 post-norm 的完整边界见 [Pre-norm 与 Post-norm](../transformer-components/pre-norm-vs-post-norm/)。

## 反向传播：梯度如何穿过 FFN

### 单个 token 的反向公式

设损失对输出的上游梯度为

$$
g_y=\frac{\partial L}{\partial y}\in\mathbb R^d.
$$

从输出层开始：

$$
g_a=\frac{\partial L}{\partial a}
=W_2^{\mathsf T}g_y,
$$

$$
g_z=\frac{\partial L}{\partial z}
=g_a\odot\phi'(z),
$$

$$
g_h=\frac{\partial L}{\partial h}
=W_1^{\mathsf T}g_z.
$$

参数梯度为

$$
\frac{\partial L}{\partial W_2}
=g_ya^{\mathsf T},
\qquad
\frac{\partial L}{\partial b_2}=g_y,
$$

$$
\frac{\partial L}{\partial W_1}
=g_zh^{\mathsf T},
\qquad
\frac{\partial L}{\partial b_1}=g_z.
$$

每个外积都保留对应矩阵的 shape：

$$
g_ya^{\mathsf T}\in\mathbb R^{d\times m},
\qquad
g_zh^{\mathsf T}\in\mathbb R^{m\times d}.
$$

### 二维例子的梯度账本

取上游梯度

$$
g_y=
\begin{pmatrix}
1\\
2
\end{pmatrix}.
$$

先回传到激活前：

$$
g_a=W_2^{\mathsf T}g_y
=
\begin{pmatrix}
1\\
2\\
-1.5
\end{pmatrix},
$$

$$
g_z=g_a\odot(1,0,1)
=
\begin{pmatrix}
1\\
0\\
-1.5
\end{pmatrix}.
$$

再回传到输入：

$$
g_h=W_1^{\mathsf T}g_z
=
\begin{pmatrix}
-0.5\\
1.5
\end{pmatrix}.
$$

参数梯度为

$$
\frac{\partial L}{\partial W_2}
=
\begin{pmatrix}
1&0&3\\
2&0&6
\end{pmatrix},
$$

$$
\frac{\partial L}{\partial W_1}
=
\begin{pmatrix}
1&-2\\
0&0\\
-1.5&3
\end{pmatrix}.
$$

对应的偏置梯度为

$$
\frac{\partial L}{\partial b_2}
=
\begin{pmatrix}
1\\
2
\end{pmatrix},
\qquad
\frac{\partial L}{\partial b_1}
=
\begin{pmatrix}
1\\
0\\
-1.5
\end{pmatrix}.
$$

ReLU 的第 2 个坐标把 $g_{a,2}=2$ 截成 $g_{z,2}=0$。这体现了前向中的关闭和反向中的局部零梯度是同一个激活状态的两面。

### 序列中的梯度累加

把 $B\times T$ 个 token 编号为 $i=1,\ldots,N$，每个 token 有 $(h_i,a_i,g_{y,i},g_{z,i})$。共享参数的梯度是所有 token 外积的和：

$$
\frac{\partial L}{\partial W_2}
=\sum_{i=1}^{N}g_{y,i}a_i^{\mathsf T},
$$

$$
\frac{\partial L}{\partial W_1}
=\sum_{i=1}^{N}g_{z,i}h_i^{\mathsf T}.
$$

batch、位置和序列样本不会拥有独立的 $W_1,W_2$。参数共享意味着不同 token 的局部梯度在同一组参数上累加；它不意味着 token 的前向表示彼此混合。

如果损失使用 mean reduction，还要把上述和除以有效样本数或有效 token 数。padding 是否计入分母必须由损失 mask 明确规定，FFN 的局部梯度公式不会自动决定归约分母。

## 中间宽度、参数量与内存

### 两层 FFN 的参数量

忽略 bias 时，

$$
P_{\mathrm{no\ bias}}
=dm+md
=2dm.
$$

包含两个 bias 时，

$$
P_{\mathrm{bias}}
=2dm+m+d.
$$

取

$$
d=4096,
\qquad
m=11008,
$$

有

$$
dm=45\,088\,768,
$$

$$
P_{\mathrm{no\ bias}}
=90\,177\,536,
$$

$$
P_{\mathrm{bias}}
=90\,192\,640.
$$

这只是单层两矩阵 FFN 的参数量。门控 FFN 通常需要额外的上投影，不能直接套用两矩阵公式，见 [SwiGLU：门控 FFN](../transformer-components/swiglu-ffn/)。

### 典型宽度的比较

| $d$ | $m$ | 无 bias 参数量 | FP16 参数占用 |
| ---: | ---: | ---: | ---: |
| $4096$ | $4096$ | $33\,554\,432$ | $64$ MiB |
| $4096$ | $11008$ | $90\,177\,536$ | $172$ MiB |
| $4096$ | $16384$ | $134\,217\,728$ | $256$ MiB |

参数占用按每个参数 2 bytes 计算，不包含 FP32 master copy、优化器状态、量化 scale 和其他运行时数据。

### 中间激活通常比 residual stream 更宽

仍取

$$
B=2,
\qquad
T=4096,
\qquad
d=4096,
\qquad
m=11008.
$$

token 数量为

$$
N=BT=8192.
$$

一个中间激活 $A$ 的元素数为

$$
BTm
=2\times4096\times11008
=90\,177\,536.
$$

若以 FP16 保存，

$$
90\,177\,536\times2
=180\,355\,072\ \mathrm{bytes}
=172\ \mathrm{MiB}.
$$

同一配置下，一个 residual stream 的大小是 $64$ MiB。只保存 $A$ 仍然会占用约 2.69 倍的单层 stream；如果同时保存 $Z$ 和 $A$，中间激活约为 $344$ MiB。

### 参数和激活是两笔不同的账

| 对象 | 规模 | FP16 大小 | 生命周期 |
| --- | ---: | ---: | --- |
| $W_1,W_2$ | $90\,177\,536$ 参数 | $172$ MiB | 模型权重，跨 token 和 step 复用 |
| 一个 residual stream | $BTd=33\,554\,432$ 元素 | $64$ MiB | 层间传递 |
| 一个 $Z$ | $BTm=90\,177\,536$ 元素 | $172$ MiB | 前向和反向所需时段 |
| 一个 $A$ | $BTm=90\,177\,536$ 元素 | $172$ MiB | 前向和反向所需时段 |

推理时可以在写回之后释放 $Z,A$；训练反向若不重算，则需要保存或重新计算这些中间量。报告 FFN 内存时必须说明是权重、单个激活、所有层 checkpoint，还是包含优化器状态。

## 初始化和数值尺度

### 第一层的方差近似

假设输入坐标独立、均值为零，且 $W_1$ 的元素独立、均值为零、方差为 $\sigma_{W_1}^2$，则一个 pre-activation 坐标可近似写为

$$
\operatorname{Var}(z_j)
\approx
d\,\sigma_{W_1}^2\operatorname{Var}(h_k).
$$

如果希望 $z_j$ 的方差维持在目标尺度 $\sigma_z^2$，则第一层权重方差需要随 fan-in $d$ 调整：

$$
\sigma_{W_1}^2
\approx
\frac{\sigma_z^2}
{d\,\operatorname{Var}(h_k)}.
$$

这是独立同分布近似，不适用于强相关的实际激活。归一化、残差累积和参数相关性都会改变该估计。

### 第二层的输出方差

如果激活坐标的方差约为 $\sigma_a^2$，第二层权重元素方差为 $\sigma_{W_2}^2$，则一个输出坐标的粗略估计为

$$
\operatorname{Var}(y_r)
\approx
m\,\sigma_{W_2}^2\sigma_a^2.
$$

中间宽度 $m$ 增大后，第二层的 fan-in 也增大。只调整 $W_1$ 而不检查 $W_2$，不能保证写回 residual stream 的方差稳定。

### 激活统计会改变方差路径

ReLU 会把负值变成零，改变均值、方差和有效坐标数量。GELU、SiLU 等平滑激活保留负值，但也会改变尺度。初始化核验至少要记录：

| 位置 | 需要记录 |
| --- | --- |
| $h$ | 输入均值、方差和范数 |
| $z$ | pre-activation 均值、方差、极值 |
| $a$ | 激活后均值、方差、非零比例 |
| $y$ | 输出均值、方差、范数 |
| $x+F(h)$ | 写回后的 residual stream 统计 |

把 $z$ 的统计量直接当作 $a$ 的统计量，会遗漏激活函数的作用。

## FFN 与 Transformer block 的接口

### Pre-norm 中的 FFN

在 pre-norm block 中，FFN 通常读取归一化后的 stream：

$$
h_l=N_{\mathrm{ffn}}(x_{l+\frac12}),
$$

$$
\Delta_{\mathrm{ffn},l}=F_{\mathrm{ffn}}(h_l),
$$

$$
x_{l+1}=x_{l+\frac12}+\Delta_{\mathrm{ffn},l}.
$$

FFN 的输入宽度仍为 $d$，norm 只改变输入数值和统计量，不改变 $W_1$ 所需的最后一轴 shape。

### Post-norm 中的 FFN

在 post-norm 结构中，抽象形式可以写成

$$
\widetilde x_{l+1}
=x_{l+\frac12}+F_{\mathrm{ffn}}(x_{l+\frac12}),
$$

$$
x_{l+1}=N_{\mathrm{ffn}}(\widetilde x_{l+1}).
$$

FFN 读取哪个版本、norm 位于输入还是输出，决定了函数和 Jacobian。不能看到同样的 $W_1,W_2$ 就把两个结构当作同一个 FFN。

### 串行和并行 block

串行 block 中，FFN 读取 attention 写回后的 stream：

$$
x_{l+\frac12}
=x_l+F_{\mathrm{attn}}(N_{\mathrm{attn}}(x_l)),
$$

$$
x_{l+1}
=x_{l+\frac12}
+F_{\mathrm{ffn}}(N_{\mathrm{ffn}}(x_{l+\frac12})).
$$

并行 block 中，两个分支都读取同一个输入：

$$
x_{l+1}
=x_l
+F_{\mathrm{attn}}(N_{\mathrm{attn}}(x_l))
+F_{\mathrm{ffn}}(N_{\mathrm{ffn}}(x_l)).
$$

FFN 的局部 shape 在两种结构中都可以正确，但串行结构的 FFN 输入包含 attention 的更新，前向依赖和 Jacobian 不同。

### Attention 负责跨 token，FFN 负责逐 token

对序列中第 $t$ 个位置，FFN 的局部计算为

$$
y_t=F(h_t).
$$

没有显式的 $h_s$，其中 $s\ne t$。如果 $h_t$ 是 attention 的输出，$h_t$ 已经可能包含多个位置的 value 汇总，因此整个 block 的 $y_t$ 可以具有上下文信息；这不是 FFN 在当前调用中跨 token 读取的结果。

在 causal decoder 中，如果上一阶段的 $h_t$ 只依赖位置 $s\le t$，逐 token FFN 不会引入未来位置：

$$
h_t\ \text{依赖}\ \{x_s:s\le t\}
\Longrightarrow
F(h_t)\ \text{依赖}\ \{x_s:s\le t\}.
$$

### 每层通常有自己的 FFN 参数

Transformer 不同层通常使用不同的 $W_{1,l},W_{2,l}$：

$$
\Delta_l=F_l(h_l)
=W_{2,l}\phi(W_{1,l}h_l+b_{1,l})+b_{2,l}.
$$

同一层内的参数跨 batch 和位置共享，跨层通常不共享。将所有层错误地绑定到同一组 FFN 参数会改变模型的容量和函数。

## Bias、权重布局与实现合同

### Bias 可以存在，也可以被省略

完整形式是

$$
y=W_2\phi(W_1h+b_1)+b_2.
$$

某些 Transformer 实现省略 bias：

$$
y=W_2\phi(W_1h).
$$

省略 bias 会减少 $m+d$ 个参数，也会改变每个中间超平面的位置。载入 checkpoint 时，必须确认权重是否包含 bias；只按矩阵 shape 检查不能发现缺失 bias。

### 行向量实现的转置

单 token 列向量公式使用

$$
W_1\in\mathbb R^{m\times d},
\qquad
W_2\in\mathbb R^{d\times m}.
$$

如果实现把 token 堆成行向量 $X\in\mathbb R^{N\times d}$，对应写法是

$$
Z=XW_1^{\mathsf T},
\qquad
Y=\phi(Z)W_2^{\mathsf T}.
$$

某个框架把权重保存为 $[d,m]$ 或 $[m,d]$，只是布局约定；必须把乘法两侧的 shape、转置和 checkpoint 命名一起核对。

### 输出投影必须匹配 residual stream

若

$$
W_2\in\mathbb R^{d_{\mathrm{out}}\times m},
$$

则输出是 $\mathbb R^{d_{\mathrm{out}}}$。只有在

$$
d_{\mathrm{out}}=d_{\mathrm{model}}
$$

时，$y$ 才能直接写入同宽度 residual stream：

$$
x_{l+1}=x_l+y.
$$

如果 $d_{\mathrm{out}}\ne d_{\mathrm{model}}$，必须增加投影或改写加法接口。不能用广播把不同宽度“加起来”后继续解释为标准 residual update。

## Padding、Packed Sequence 与有效 token

### FFN 会看到 padding 的向量

padding token 也有

$$
h_{b,t,:}\in\mathbb R^d.
$$

普通 FFN 的逐位置公式仍会对它执行

$$
y_{b,t,:}=F(h_{b,t,:}).
$$

attention mask 不会自动阻止 FFN 计算 padding。若实现要跳过 padding，需要显式的 valid-token 索引、稀疏 kernel 或结果清零策略。

### Padding 的输出不能自动当作零

即使输入 padding 向量为零，如果 $b_1$ 或 $b_2$ 非零，

$$
F(0)=W_2\phi(b_1)+b_2
$$

也可能非零。若 padding stream 必须保持零，必须明确执行 mask 或清零，并说明它发生在 FFN 前、FFN 后还是 residual add 后。

### Packed sequence 的 FFN 边界简单，attention 边界复杂

把多个短序列 packed 到同一个时间轴后，FFN 仍然对每个位置独立执行。它不会读取相邻 packed segment，因此本身不会跨 segment 混合。

attention 仍必须使用 segment 边界 mask，防止不同样本互读。一个实现若在 FFN 中出现 token 轴上的卷积、池化或归约，就已经不再是本篇定义的普通逐 token FFN。

### Loss mask 和 FFN 计算 mask 是两件事

loss mask 可以只排除 padding 的损失：

$$
L
=\frac{\sum_{b,t}M_{b,t}\ell_{b,t}}
{\sum_{b,t}M_{b,t}},
\qquad
M_{b,t}\in\{0,1\}.
$$

这不会自动减少 FFN 的前向计算，也不会把 padding 的局部梯度从中间激活中删除。需要节省计算时，要在执行路径中单独实现有效 token 选择。

## 混合精度和中间激活

### FFN 内部可能有多次 dtype 转换

至少区分以下 dtype：

| 位置 | 需要固定 |
| --- | --- |
| 输入 stream | $h$ 的存储格式 |
| 权重 | $W_1,W_2$ 的存储格式 |
| GEMM 累加 | 点积部分和的累加格式 |
| 激活输入 | $z$ 的计算格式 |
| 激活输出 | $a$ 的存储或临时格式 |
| 输出 GEMM | $y$ 的累加和输出格式 |
| residual add | $x+y$ 的实际相加格式 |

只报告模型输入和最终 logits 的 dtype，不能复现 FFN 的舍入路径。

### 小增量可能在输出层被吞掉

如果 $W_2a$ 的某个坐标远小于 residual stream 的对应坐标，低精度输出或 residual add 可能无法保留它。需要分别比较：

1. 高精度参考的 $W_2a$；
2. 目标 dtype 的 FFN 输出；
3. 实际 kernel 的 fused output；
4. 写回 residual stream 后的结果。

差异出现在第 1 步和第 2 步之间时，问题在 FFN 计算或量化；差异只出现在第 3、4 步时，问题可能在融合 kernel 或 residual add。

### 激活保存和重算

训练反向需要 $h$、$z$ 或等价的激活状态。常见选择为：

| 策略 | 保存内容 | 影响 |
| --- | --- | --- |
| 保存 $z,a$ | 直接反向 | 激活内存高 |
| 保存 $h$，反向重算 | 少保存中间量 | 增加一次 FFN 前向 |
| activation checkpoint | 只保存边界 stream | 以计算换内存 |
| 推理即时释放 | 不保留反向状态 | 不能直接反向 |

报告 FFN 的 activation memory 时，要写明是否包含 $h,z,a$，以及是否跨层保存。

## 一个可复算的反向核验

### 核验前向

重新使用二维例子的参数：

$$
h=
\begin{pmatrix}
1\\
-2
\end{pmatrix},
\quad
W_1=
\begin{pmatrix}
1&0\\
0&1\\
1&-1
\end{pmatrix},
\quad
W_2=
\begin{pmatrix}
1&0&0.5\\
0&1&-1
\end{pmatrix}.
$$

应得到

$$
z=(1,-2,3)^{\mathsf T},
\qquad
a=(1,0,3)^{\mathsf T},
\qquad
y=(2.5,-3)^{\mathsf T}.
$$

### 核验 Jacobian

在不改变激活模式的扰动

$$
\delta h=(0.01,-0.02)^{\mathsf T}
$$

下，应得到

$$
J_F=
\begin{pmatrix}
1.5&-0.5\\
-1&1
\end{pmatrix},
\qquad
J_F\delta h=(0.025,-0.03)^{\mathsf T}.
$$

也可以用中心差分逐列估计：

$$
\frac{\partial F}{\partial h_k}
\approx
\frac{F(h+\varepsilon e_k)-F(h-\varepsilon e_k)}
{2\varepsilon}.
$$

选取足够小但不跨越 $z_j=0$ 的 $\varepsilon$，数值结果应接近解析 Jacobian。若扰动跨越激活边界，差分结果依赖两侧的激活模式，不能用单侧 Jacobian 直接比较。

### 核验反向

取

$$
g_y=(1,2)^{\mathsf T},
$$

应得到

$$
g_z=(1,0,-1.5)^{\mathsf T},
\qquad
g_h=(-0.5,1.5)^{\mathsf T}.
$$

再检查

$$
\frac{\partial L}{\partial W_2}
=
\begin{pmatrix}
1&0&3\\
2&0&6
\end{pmatrix},
\qquad
\frac{\partial L}{\partial W_1}
=
\begin{pmatrix}
1&-2\\
0&0\\
-1.5&3
\end{pmatrix}.
$$

第 2 行的 $W_1$ 梯度为零，原因是当前第 2 个 pre-activation 处于 ReLU 负区间。它不是因为第 2 个输入坐标没有参与 $W_1$；零梯度来自激活导数。

## 诊断指标和审计表

### 每层 FFN 的最小统计量

对 $N=BT$ 个 token 和 $m$ 个中间坐标，建议记录：

$$
\operatorname{active\_fraction}
=
\frac{1}{Nm}
\sum_{i=1}^{N}\sum_{j=1}^{m}
\mathbf 1_{z_{i,j}>0}
$$

（ReLU 情况），以及

$$
\operatorname{norm\_ratio}
=
\frac{\lVert F(h_i)\rVert_2}
{\lVert h_i\rVert_2+\delta}.
$$

还要记录

$$
\operatorname{mean}(z),
\quad
\operatorname{std}(z),
\quad
\operatorname{mean}(a),
\quad
\operatorname{std}(a),
\quad
\operatorname{mean}(y).
$$

这些量分别描述激活区域、写入尺度和统计偏移。没有激活函数类型和采样轴时，统计量无法复现。

### Shape 审计

| 检查项 | 通过条件 |
| --- | --- |
| 输入 | 最后一轴为 $d_{\mathrm{model}}$ |
| 第一层 | $W_1$ 的输入轴匹配 $d_{\mathrm{model}}$ |
| 中间量 | $z,a$ 的最后一轴为 $d_{\mathrm{ffn}}$ |
| 激活 | 逐坐标执行，不意外混合 token |
| 第二层 | $W_2$ 的输出轴回到 $d_{\mathrm{model}}$ |
| 残差写回 | $F(h)$ 与 shortcut shape 完全一致 |
| 参数共享 | 同层跨 batch/位置共享，跨层按架构确认 |
| bias | checkpoint 和实现对 bias 的约定一致 |

### 数值和计算图审计

| 检查项 | 证据 |
| --- | --- |
| 前向 | 独立计算 $z,a,y$ 并与实现输出比较 |
| 激活 | 记录激活函数、边界和非零比例 |
| Jacobian | 解析链式法则与局部有限差分一致 |
| 梯度 | $g_h$、两个权重矩阵和两个 bias 的 shape 一致 |
| mask | padding、loss 分母和实际跳过计算的路径分别记录 |
| dtype | GEMM 累加、激活、输出和 residual add 分别记录 |
| 内存 | 区分参数、stream、$z$、$a$ 和 checkpoint |
| 写回 | 记录 FFN 输出进入哪个 stream 版本 |

## 失效模式

### 把两层线性映射称为 FFN

如果删除 $\phi$，两层线性层可以合并成一层仿射映射。检查计算图中是否存在激活或门控，以及激活的输入是否确实来自 $W_1h+b_1$。

### 混淆 $d_{\mathrm{model}}$ 和 $d_{\mathrm{ffn}}$

$d_{\mathrm{model}}$ 是 residual stream 的宽度，$d_{\mathrm{ffn}}$ 是中间激活宽度。把中间宽度错误地写回 shortcut，会造成 shape 错误或隐式广播。

### 权重转置方向错误

行向量和列向量约定会改变矩阵乘法写法。只检查权重文件的两个整数维度，不能确认转置方向；要把乘法两侧的 shape、转置和 checkpoint 命名一起核对。

### 把 FFN 当作 token 混合器

普通 FFN 沿最后一轴逐 token 计算。出现跨位置求和、卷积、池化或 attention 权重时，计算图已经包含其他 token-mixing 算子。

### 忘记 attention 对 FFN 输入的影响

FFN 本身不跨 token，但它的输入可能来自 attention 写回后的 residual stream。把串行 block 的 FFN 错改为读取旧 stream，会保留 shape 却改变函数。

### padding 只 mask 了 loss

loss mask 不会自动跳过 FFN 的前向和中间激活。若 padding 的输出必须保持零，需要在计算路径中显式处理。

### ReLU dead unit 被误读成固定能力

当前 batch 的 active fraction 低，只说明这些输入落在负区间。要判断是否长期 dead，需要跨 batch、层和训练 step 统计 $z_j$ 的符号及其梯度。

### 只检查输出范数

FFN 可能保持输出范数，却改变读出方向；也可能增大范数但对任务方向影响很小。应同时记录 $F(h)$ 的范数、与 $h$ 的余弦和指定读出变化。

### 只报告参数量，不报告中间激活

中间激活的元素数是 $BTd_{\mathrm{ffn}}$。长序列和大中间宽度下，$Z,A$ 可能比单层 residual stream 更占内存。

### 混合精度差异被归因于激活函数

若参考 FP32 与实际输出不同，先按 GEMM、激活、输出投影和 residual add 分段比较。没有分段证据时，原因未查明。

## 一个可复用的核验协议

### 记录架构合同

1. 记录 $d_{\mathrm{model}}$、$d_{\mathrm{ffn}}$、层数和是否包含 bias。
2. 记录 $W_1,W_2$ 的布局、转置和 dtype。
3. 记录激活函数及其在边界点的导数约定。
4. 记录 FFN 读取的是 norm 前、norm 后还是 attention 写回后的 stream。
5. 记录 FFN 输出写回的 stream 版本和 residual add 的 dtype。

### 运行小矩阵探针

1. 使用二维例子计算 $z,a,y$。
2. 使用不跨激活边界的扰动检查局部 Jacobian。
3. 使用上游梯度检查 $g_h$、权重梯度和 bias 梯度。
4. 改变一个输入使激活模式跨过边界，再检查局部线性模型何时失效。

### 运行批量和内存探针

1. 以 $B=2,T=4096,d=4096,m=11008$ 核对参数量和中间激活。
2. 核对 FP16 与 FP32 的权重、$Z$、$A$ 和 residual stream 账本。
3. 对 padding 和 packed sequence 分别检查实际计算路径。
4. 对训练、checkpoint 和推理分别记录激活生命周期。

### 解释 FFN 写入

对每层至少附带：

1. 当前层和 token 位置；
2. FFN 输入是哪个 stream 版本；
3. 激活函数和 active fraction；
4. $\lVert F(h)\rVert_2/\lVert h\rVert_2$；
5. 指定读出方向上的 $w^{\mathsf T}F(h)$；
6. padding、mask、dtype 和 residual add 位置。

这些字段缺失时，只能确认张量经过了一个 FFN，不能把输出变化归因于某个特征或子层功能。

## 相关词条

- [残差流](../transformer-components/residual-streams/)
- [SwiGLU：门控 FFN](../transformer-components/swiglu-ffn/)
- [激活函数](../neurons-and-activations/activation-functions/)
- [参数量总账](../transformer-components/parameter-count/)
- [Pre-norm 与 Post-norm](../transformer-components/pre-norm-vs-post-norm/)
- [混合精度训练](../training-nn/mixed-precision/)
