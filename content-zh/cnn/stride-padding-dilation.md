---
title: "步幅、填充与膨胀：卷积窗口如何改变坐标"
tags: ["why-models-learn"]
---

padding、stride 和 dilation 都会改变卷积输出，但它们改动的是三件不同的事：padding 规定边界外的数值，stride 规定相邻输出窗口起点相隔多远，dilation 规定同一个核内部相邻权重在输入上相隔多远。把三者都笼统地叫作“缩放参数”，会同时误判输出尺寸、特征对齐、感受野和边界响应。

本文先在一维数组上固定坐标，再推广到二维张量。全文使用深度学习中常见的不翻转滑动窗口记号；若实现采用数学卷积，只需在空间核上额外翻转，padding、stride 和 dilation 的坐标推导不变。

![步幅、填充与膨胀的坐标对照：padding 扩展边界，stride 移动窗口起点，dilation 拉开核内采样点](/assets/cnn/svg/stride-padding-dilation.1.svg)

## 先把一个输出位置写成坐标

设一维输入为 $x[0],\ldots,x[N-1]$，核为 $w[0],\ldots,w[k-1]$。左、右 padding 分别为 $P_{\mathrm{left}}$ 和 $P_{\mathrm{right}}$，stride 为 $s$，dilation 为 $d$。

### Padding 定义输入坐标的延伸

零 padding 先把输入扩展为

$$
x^{P}[q]
=
\begin{cases}
x[q-P_{\mathrm{left}}],
&0\le q-P_{\mathrm{left}}<N,\\
0,
&\text{其他情况}.
\end{cases}
$$

原始输入位置 $u$ 在 padded 数组中的位置是 $u+P_{\mathrm{left}}$。因此，padding 不会移动原始像素的相对顺序，但会改变坐标原点以及窗口在边缘能看到的数值。

### Stride 定义窗口起点

输出索引 $r$ 对应的窗口起点是

$$
q_r=rs.
$$

当 $s=1$ 时，每个相邻起点都被访问；当 $s=2$ 时，窗口起点依次为 $0,2,4,\ldots$。stride 不是先把输入做插值或平均再卷积，而是直接跳过一部分窗口起点。

### Dilation 定义核内采样间距

窗口中的第 $a$ 个核权重读取 padded 输入坐标

$$
q_{r,a}=rs+ad,
\qquad
0\le a\le k-1.
$$

所以一个输出值为

$$
y[r]
=\sum_{a=0}^{k-1}
x^{P}[rs+ad]w[a].
$$

当 $d=1$ 时，核读取连续坐标；当 $d=2$ 时，核的相邻权重之间跳过一个输入位置。三个参数在这个式子中的位置不同：padding 已经写进 $x^P$，stride 乘在输出索引上，dilation 乘在核索引上。

## 有效核尺寸和输出长度

虽然 dilation 没有增加可学习权重的个数，但它扩大了核覆盖的坐标跨度。定义有效核尺寸

$$
k_{\mathrm{eff}}
=d(k-1)+1.
$$

第 $r$ 个输出窗口从 $rs$ 开始，最后一个被读取的坐标是

$$
rs+d(k-1)
=rs+k_{\mathrm{eff}}-1.
$$

该坐标不能超过 padded 数组最后一个位置 $N+P_{\mathrm{left}}+P_{\mathrm{right}}-1$。因此输出长度为

$$
N_{\mathrm{out}}
=\left\lfloor
\frac{
N+P_{\mathrm{left}}+P_{\mathrm{right}}-k_{\mathrm{eff}}
}{s}
\right\rfloor+1.
$$

这个公式的每一项都有坐标含义：

- $N+P_{\mathrm{left}}+P_{\mathrm{right}}$ 是 padded 长度；
- $k_{\mathrm{eff}}$ 是窗口真正覆盖的跨度，而不是权重个数 $k$；
- $s$ 是相邻窗口起点的间隔；
- floor 表示最后一个完整窗口之后，剩余不足一个 stride 的部分被丢弃。

若分子为负，说明有效核连一个完整位置也放不下；实现应拒绝这个 shape，而不是把输出长度解释成一个负数或静默裁剪。

### 二维输出尺寸

对高度和宽度分别使用参数 $P_{\mathrm{top}},P_{\mathrm{bottom}},P_{\mathrm{left}},P_{\mathrm{right}}$、$s_h,s_w$、$d_h,d_w$：

$$
k_h^{\mathrm{eff}}
=d_h(k_h-1)+1,
\qquad
k_w^{\mathrm{eff}}
=d_w(k_w-1)+1.
$$

二维输出尺寸为

$$
H_{\mathrm{out}}
=\left\lfloor
\frac{
H+P_{\mathrm{top}}+P_{\mathrm{bottom}}-k_h^{\mathrm{eff}}
}{s_h}
\right\rfloor+1,
$$

$$
W_{\mathrm{out}}
=\left\lfloor
\frac{
W+P_{\mathrm{left}}+P_{\mathrm{right}}-k_w^{\mathrm{eff}}
}{s_w}
\right\rfloor+1.
$$

高度和宽度不能只套一组参数。矩形核、不同轴的 stride、不同轴的 dilation 或左右不对称 padding，都可能让两个方向的输出行为不同。

## Padding：边界外的数值是什么

考虑输入

$$
x=[1,2,3],
\qquad
w=[1,0,-1].
$$

左右各补一个零后，padded 输入为 $[0,1,2,3,0]$。stride 和 dilation 都为 $1$ 时，三个输出窗口给出

$$
\begin{aligned}
y[0]&=0\times1+1\times0+2\times(-1)=-2,\\
y[1]&=1\times1+2\times0+3\times(-1)=-2,\\
y[2]&=2\times1+3\times0+0\times(-1)=2.
\end{aligned}
$$

如果改成 valid，不使用边界外零，只剩中心窗口，输出只有 $[-2]$。因此 same、valid 和 full 不只是输出数组的裁剪名称，它们隐含了不同的窗口集合或边界语义。

常见边界方式包括：

| padding | 边界外取值 | 写入的先验 | 可能的副作用 |
| --- | --- | --- | --- |
| zero | 取 0 | 背景或缺失值等于零 | 边缘可能出现人为响应 |
| reflect | 按边界镜像 | 信号在边缘较平滑 | 强边缘会被重复 |
| replicate | 复制最近值 | 边界值延续不变 | 平坦区域被拉长 |
| circular | 从另一侧回绕 | 数据有周期结构 | 图像两边产生非真实连接 |

卷积核本身不会告诉你该选哪一种 padding。选项来自数据的边界语义，必须在模型配置、论文公式或数据管线中单独记录。

## Stride：减少窗口起点，不是自动抗混叠

取 $N=7$、$k=3$、左右各补一个零、$s=2$、$d=1$。padded 长度是 $9$，窗口起点为

$$
q_0=0,
\qquad
q_1=2,
\qquad
q_2=4,
\qquad
q_3=6.
$$

因此

$$
N_{\mathrm{out}}
=\left\lfloor\frac{7+1+1-3}{2}\right\rfloor+1
=4.
$$

输出仍然是四个完整窗口的聚合结果，只是没有计算起点 $1,3,5$ 的窗口。stride 会降低空间采样密度和 MACs，但不自动消除输入中的高频成分；如果下采样前存在混叠风险，还需要明确的低通或抗混叠设计。

stride 大于 $1$ 时，输出索引与原输入坐标的映射也变得稀疏。若 padding 左侧为 $P_{\mathrm{left}}$，窗口左端点在未 padding 坐标中的位置是

$$
u_r=rs-P_{\mathrm{left}}.
$$

同一个输出索引在不同的左 padding 下可能对应不同的原始坐标。只比较输出 shape，无法确认特征是否空间对齐。

## Dilation：拉开核内的采样点

取 $k=3,d=2$，有效核尺寸为

$$
k_{\mathrm{eff}}=2(3-1)+1=5.
$$

三个可学习权重读取的相对坐标是

$$
0,\quad2,\quad4,
$$

中间的坐标 $1$ 和 $3$ 没有对应的核参数。若窗口起点是 $r s$，输出为

$$
y[r]
=x^{P}[rs]w[0]
+x^{P}[rs+2]w[1]
+x^{P}[rs+4]w[2].
$$

因此 dilation 扩大空间覆盖范围，但参数量仍然只有 $k$ 个权重。把有效核尺寸误写成 $k$，会让输出长度、same padding 和边缘窗口全部错一位。

例如 $N=9,k=3,d=2,s=1$，左右各补两个零：

$$
N_{\mathrm{out}}
=\left\lfloor\frac{9+2+2-5}{1}\right\rfloor+1
=9.
$$

这里的总 padding 由 $k_{\mathrm{eff}}=5$ 决定，而不是由三个可学习位置的数量 $k=3$ 决定。

## 三个参数组合时要保留完整坐标

一维窗口的完整采样坐标可以写成

$$
\{rs,\ rs+d,\ rs+2d,\ldots,rs+(k-1)d\}.
$$

在二维中，对输出位置 $(r,c)$ 和核位置 $(a,b)$，读取的 padded 输入坐标为

$$
\left(
r s_h+a d_h,\quad
c s_w+b d_w
\right).
$$

带输入通道 $i$、输出通道 $o$ 的互相关公式是

$$
Z_o[r,c]
=\beta_o
+\sum_i
\sum_{a=0}^{k_h-1}
\sum_{b=0}^{k_w-1}
X_i^{P}
\left[
r s_h+a d_h,\quad
c s_w+b d_w
\right]
K_{o,i,a,b}.
$$

严格数学卷积只把最后的核项换成

$$
K_{o,i,k_h-1-a,k_w-1-b}.
$$

padding、stride 和 dilation 的坐标合同在两种核方向约定下完全相同。先解决三者，再讨论互相关和卷积翻转，可以把两个问题分开验收。

## Same 不是一个足够完整的配置

很多接口用 same 表示“尽量保持预期的输出大小”，但当 stride 大于 $1$、核为偶数或 dilation 大于 $1$ 时，仍然需要指定目标输出和 padding 如何分配。

若希望一维输出满足

$$
N_{\mathrm{out}}=\left\lceil\frac{N}{s}\right\rceil,
$$

可以先计算所需的总 padding：

$$
P_{\mathrm{total}}
=\max\left(
(N_{\mathrm{out}}-1)s+k_{\mathrm{eff}}-N,\ 0
\right).
$$

再把它分给左右两侧：

$$
P_{\mathrm{left}}+P_{\mathrm{right}}
=P_{\mathrm{total}}.
$$

当 $P_{\mathrm{total}}$ 为奇数时，左边多一个还是右边多一个，会改变输出的空间相位。对二维张量，高度和宽度各自有一份 total padding 和分配规则。只写“same”而不写四侧数值，无法完整复现一个模型。

奇数核、dilation 为 $1$、stride 为 $1$ 时，常见的对称选择是

$$
P_{\mathrm{left}}=P_{\mathrm{right}}
=\frac{k-1}{2}.
$$

若 $k$ 为偶数或 dilation 使 $k_{\mathrm{eff}}$ 为偶数，这个对称分配可能不是整数，必须选择一个方向承接多出来的一个位置。

## 输出坐标的相位和感受野

一个层的输出中心不只由输出索引决定，还由左侧 padding 和有效核尺寸共同决定。窗口覆盖的未 padding 坐标范围为

$$
\left[
rs-P_{\mathrm{left}},
\ rs-P_{\mathrm{left}}+k_{\mathrm{eff}}-1
\right].
$$

若有效核尺寸为奇数，可以把几何中心写成

$$
u_{\mathrm{center}}
=rs-P_{\mathrm{left}}
+\frac{k_{\mathrm{eff}}-1}{2}.
$$

左右不对称 padding 会整体改变这个中心；stride 会改变相邻中心的间隔；dilation 会扩大单个窗口的覆盖范围。边界处的有效感受野还会包含 padding 值，因此“理论覆盖范围”和“真实输入像素数量”不一定相同。

连续多层时，令第 $l$ 层输出相邻点在原输入上的间距为 jump $j_l$，单个输出覆盖的理论感受野为 $r_l$。则

$$
j_l=j_{l-1}s_l,
\qquad
r_l=r_{l-1}
+\left(k_l^{\mathrm{eff}}-1\right)j_{l-1},
$$

初始为 $j_0=1,r_0=1$。例如第一层 $3\times3$、stride $2$，第二层 $3\times3$、dilation $2$、stride $1$：

$$
\begin{aligned}
j_1&=2,&r_1&=3,\\
j_2&=2,&r_2&=3+(5-1)\times2=11.
\end{aligned}
$$

第二层只有三个可学习位置，却能把理论感受野扩大到 $5\times5$ 的空间跨度；前一层的 stride 又把这四个新增间隔放大到原图上的八个单位。

## 反向传播仍然沿同一组坐标累加

令上游梯度为 $G[r]$，前向是一维窗口

$$
y[r]
=\sum_{a=0}^{k-1}
x^{P}[rs+ad]w[a].
$$

核参数梯度把所有输出位置使用过的输入值加起来：

$$
\frac{\partial L}{\partial w[a]}
=\sum_{r=0}^{N_{\mathrm{out}}-1}
G[r]x^{P}[rs+ad].
$$

对 padded 输入位置 $q$，输入梯度只接收满足

$$
q=rs+ad
$$

的窗口贡献：

$$
\frac{\partial L}{\partial x^{P}[q]}
=\sum_{r,a}
G[r]w[a]
\mathbf 1_{\{q=rs+ad\}}.
$$

stride 使可用的窗口起点变少，dilation 使每个窗口内部的命中位置变稀疏；二者都不改变“多个路径要累加”的原则。padding 区域的梯度在最后裁掉，只把原始输入范围内的部分传回数据。

如果前向使用了不对称 padding，输入梯度裁剪的左、右边界也必须跟着记录。只把梯度数组中心裁成原输入长度，可能会把输出相位错误地平移。

## 参数量不随 dilation 增加，但计算和内存会变

对固定的 $C_{\mathrm{in}},C_{\mathrm{out}},k_h,k_w$，普通卷积的参数量仍是

$$
P=C_{\mathrm{out}}C_{\mathrm{in}}k_hk_w+C_{\mathrm{out}}.
$$

stride、padding 和 dilation 不会新增核权重。每个输出位置仍然做 $C_{\mathrm{in}}k_hk_w$ 次乘加，因此单样本 MACs 近似为

$$
\mathrm{MACs}
=H_{\mathrm{out}}W_{\mathrm{out}}
C_{\mathrm{out}}C_{\mathrm{in}}k_hk_w.
$$

stride 增大通常减少 $H_{\mathrm{out}}W_{\mathrm{out}}$；padding 可能保持或增加输出位置；dilation 通常保持输出位置和乘加次数，却让访问地址更分散，可能改变缓存和硬件 kernel 的效率。参数量、MACs、访存和峰值激活不能用一个数字代替。

| 修改 | 参数量 | 输出位置数量 | 主要影响 |
| --- | --- | --- | --- |
| 增大 padding | 不变 | 可能增加或保持 | 边界语义、相位和计算量 |
| 增大 stride | 不变 | 通常减少 | 下采样密度、混叠风险和 MACs |
| 增大 dilation | 不变 | 由有效核尺寸重新决定 | 感受野、边界需要的 padding 和访存模式 |

## 常见失效模式

| 现象 | 可能原因 | 第一条证据 |
| --- | --- | --- |
| 输出长度差一位 | 用 $k$ 代替 $k_{\mathrm{eff}}$ 或 floor/ceil 混用 | 写出最后一个完整窗口的坐标 |
| same 输出左右错开 | total padding 分配方向不同 | 打印四侧 padding 和第一个窗口起点 |
| 边缘出现不对称响应 | 左右 padding 或边界函数不一致 | 手算最左、中心和最右窗口 |
| stride=2 的结果不稳定 | 下采样前没有考虑高频混叠 | 比较 stride=1 输出再隔点采样 |
| dilation 后只看到部分目标 | 核的采样点变稀疏或 padding 不够 | 列出 $rs+ad$ 的所有坐标 |
| 理论感受野算大了 | 没把前面层的 jump 乘进去 | 递推 $j_l,r_l$ 而不是直接相加 |
| 输入梯度少了一圈 | padding 梯度裁剪偏移或覆盖代替累加 | 追踪一个重叠输入坐标的所有路径 |
| 参数量被错误放大 | 把空洞位置当成新增权重 | 分开统计 $k$ 个参数和 $k_{\mathrm{eff}}$ 跨度 |
| shape 正确但特征错位 | 只比较高宽，没有比较空间相位 | 用脉冲输入检查输出峰值坐标 |
| 速度变化与理论 MACs 不符 | dilation 访存、布局或 kernel 实现不同 | 同时记录 MACs、内存和实际 kernel |

脉冲输入是检查空间相位的好工具：只在一个输入坐标放置非零值，逐项记录它在不同 padding、stride 和 dilation 下能影响哪些输出位置。它比观察一张自然图像更容易区分“响应变弱”和“响应坐标错位”。

## 读卷积空间参数的验收顺序

面对一层卷积或一个自定义算子，建议按下面的顺序：

1. 写出原始输入长度或高宽，以及核的权重个数；
2. 明确每个轴的左、右或上、下 padding；
3. 用 $rs+ad$ 写出一个输出位置读取的全部坐标；
4. 用 $k_{\mathrm{eff}}=d(k-1)+1$ 计算最后一个访问坐标；
5. 再用 floor 公式计算输出长度，并列出输出起点；
6. 分别检查 same 的目标长度和 padding 分配；
7. 递推多层 jump 与感受野；
8. 最后核对反向累加、MACs、访存和脉冲输入的空间相位。

只要完整保留“边界值、窗口起点、核内间距”这三个坐标事实，padding、stride 和 dilation 就不会再被混成一个模糊的缩放旋钮。

## 相关词条

- [二维卷积](../cnn/convolution-2d/)：把空间参数放入多通道张量、NCHW/NHWC 和分组结构。
- [互相关与卷积](../cnn/cross-correlation-vs-convolution/)：解释不翻转核与数学卷积的双轴方向差别。
- [离散卷积](../cnn/discrete-convolution/)：从有限支持和边界语义推导一维、二维卷积。
- [输出尺寸算术](../cnn/output-size-arithmetic/)：专门逐层核对卷积、池化和其他空间算子的 shape。
- [卷积神经网络](../cnn/cnn/)：把局部窗口、感受野和权值共享放进完整视觉架构。
- [池化](../cnn/pooling/)：比较固定下采样窗口与可学习卷积窗口的坐标行为。
- [不变性与等变性](../cnn/invariance-and-equivariance/)：连接空间平移、输出相位和特征响应。
- [卷积作为 Toeplitz 结构](../cnn/convolution-as-toeplitz/)：把 stride、padding 和共享参数写成矩阵结构。
- [向量化反向传播](../backpropagation/vectorized-backprop/)：把窗口梯度累加连接到批次矩阵运算。
