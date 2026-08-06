---
title: "二维卷积：从局部窗口到多通道张量"
tags: ["why-models-learn"]
---

二维卷积把一个小的空间核滑过图像或特征图，在每个位置读取局部窗口、逐项相乘并求和。一个输出通道会在所有位置复用同一组核参数，同时把所有输入通道的局部证据相加。深度学习框架通常实现的是不翻转核的二维互相关，但它仍沿用 convolution 这个名字；本文先固定张量坐标，再分别说明空间窗口、通道求和、输出尺寸、边界、分组和反向梯度。

前一篇[互相关与卷积](../cnn/cross-correlation-vs-convolution/)已经证明“不翻转”和“数学卷积”的差别来自两个空间轴的核翻转。这里的重点转向二维张量本身：一个数字到底对应哪个通道、哪个窗口坐标、哪个输出位置，以及这些坐标如何决定参数量和梯度累加。

![二维卷积示意图：输入特征图的局部窗口与二维核逐项相乘求和，经过通道聚合后写入输出特征图](/assets/cnn/svg/convolution-2d.1.svg)

## 先固定一个张量坐标合同

先采用 channels-first 记号，不包含 batch 轴。输入特征图写成

$$
X\in\mathbb R^{C_{\mathrm{in}}\times H\times W},
$$

其中 $i$ 是输入通道，$p$ 是高度坐标，$q$ 是宽度坐标。一个卷积层的核和偏置写成

$$
K\in
\mathbb R^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times k_h\times k_w},
\qquad
\beta\in\mathbb R^{C_{\mathrm{out}}}.
$$

输出为

$$
Z\in\mathbb R^{C_{\mathrm{out}}\times H_{\mathrm{out}}\times W_{\mathrm{out}}}.
$$

因此 $K_{o,i,a,b}$ 的四个位置分别表示：第 $o$ 个输出通道读取第 $i$ 个输入通道，并在窗口的高度偏移 $a$、宽度偏移 $b$ 使用一个权重。它不是一张独立的二维图，而是带有输入和输出通道轴的四维参数张量。

### Padding 先改变坐标，再让窗口移动

设上、下、左、右的 padding 分别为 $P_{\mathrm{top}}$、$P_{\mathrm{bottom}}$、$P_{\mathrm{left}}$、$P_{\mathrm{right}}$。零 padding 可以写成一个扩展后的数组：

$$
X_i^{P}[p,q]
=
\begin{cases}
X_i[p-P_{\mathrm{top}},q-P_{\mathrm{left}}],
&0\le p-P_{\mathrm{top}}<H,\quad
0\le q-P_{\mathrm{left}}<W,\\
0,
&\text{其他情况}.
\end{cases}
$$

这个定义把“访问边界外元素怎么办”提前固定下来。反射、复制和周期 padding 会使用不同的边界函数，但后面的窗口公式仍可以保持不变。

### 互相关形式的前向公式

设高度步幅和宽度步幅为 $s_h,s_w$，高度膨胀和宽度膨胀为 $d_h,d_w$。绝大多数深度学习卷积层的空间计算可写成

$$
Z_o[r,c]
=\beta_o
+\sum_{i=0}^{C_{\mathrm{in}}-1}
+\sum_{a=0}^{k_h-1}
+\sum_{b=0}^{k_w-1}
+X_i^{P}[r s_h+a d_h,\ c s_w+b d_w]
+K_{o,i,a,b}.
$$

公式中的 $a,b$ 是核内部坐标；$r,c$ 是输出坐标。每次把 $r$ 或 $c$ 加一，窗口在输入上移动的是 $s_h$ 或 $s_w$ 个位置；核的相邻元素在输入上相隔的是 $d_h$ 或 $d_w$ 个位置。

如果要使用严格的数学二维卷积，空间核需要同时翻转两个轴：

$$
K^{\mathrm{flip}}_{o,i,a,b}
=K_{o,i,k_h-1-a,k_w-1-b}.
$$

然后把前向公式中的 $K$ 换成 $K^{\mathrm{flip}}$。通道轴不参与空间翻转。两种约定的边界、步幅和张量形状可以相同，数值响应仍可能不同。

## 输出尺寸来自最后一个可访问坐标

先定义膨胀后的有效核尺寸：

$$
k_h^{\mathrm{eff}}
=d_h(k_h-1)+1,
\qquad
k_w^{\mathrm{eff}}
=d_w(k_w-1)+1.
$$

高度方向最后一次窗口访问的 padded 坐标不能超过 $H+P_{\mathrm{top}}+P_{\mathrm{bottom}}-1$。因此输出高度为

$$
H_{\mathrm{out}}
=\left\lfloor
\frac{
H+P_{\mathrm{top}}+P_{\mathrm{bottom}}-k_h^{\mathrm{eff}}
}{s_h}
\right\rfloor+1.
$$

宽度方向同理：

$$
W_{\mathrm{out}}
=\left\lfloor
\frac{
W+P_{\mathrm{left}}+P_{\mathrm{right}}-k_w^{\mathrm{eff}}
}{s_w}
\right\rfloor+1.
$$

更直接的验证方式是写出最后一个输出坐标：

$$
(H_{\mathrm{out}}-1)s_h+k_h^{\mathrm{eff}}
\le H+P_{\mathrm{top}}+P_{\mathrm{bottom}},
$$

并确认再增加一个输出位置就会越过 padded 高度。宽度方向使用同一个不等式替换对应符号。

例如 $H=5,W=6$，使用 $3\times2$ 的核、$s_h=s_w=2$、$d_h=d_w=1$，高度上下各补一个零、宽度不补零，则

$$
\begin{aligned}
H_{\mathrm{out}}
&=\left\lfloor\frac{5+1+1-3}{2}\right\rfloor+1
=3,\\
W_{\mathrm{out}}
&=\left\lfloor\frac{6-2}{2}\right\rfloor+1
=3.
\end{aligned}
$$

输出空间尺寸是 $3\times3$。步幅不是“先把输入缩小再做卷积”，而是改变输出坐标在 padded 输入上的采样间隔。

## 一个输出数字如何从窗口中产生

为了暂时去掉通道轴，取单通道输入和 $2\times2$ 核：

$$
X=
\begin{bmatrix}
1&2&0\\
3&4&1\\
2&0&5
\end{bmatrix},
\qquad
K=
\begin{bmatrix}
1&-1\\
2&0
\end{bmatrix}.
$$

使用不翻转的 valid 互相关，左上角输出为

$$
Z[0,0]
=1\times1+2\times(-1)+3\times2+4\times0
=5.
$$

向右移动一个位置，窗口变成

$$
\begin{bmatrix}
2&0\\
4&1
\end{bmatrix},
$$

于是

$$
Z[0,1]
=2\times1+0\times(-1)+4\times2+1\times0
=10.
$$

四个窗口全部计算后得到

$$
Z=
\begin{bmatrix}
5&10\\
3&3
\end{bmatrix}.
$$

这里的四个输出位置使用的是同一个 $K$。如果每个位置都有一份独立的 $2\times2$ 权重，就失去了卷积的权值共享；如果把窗口内的数只与同一坐标相乘而不求和，则变成了逐元素乘法而不是卷积。

## 多通道不是多做几次再随便堆叠

一个输出通道会把所有输入通道的窗口响应加在一起。对固定的 $o,r,c$：

$$
Z_o[r,c]
=\beta_o
+\sum_{i=0}^{C_{\mathrm{in}}-1}
+\left(
+\sum_{a=0}^{k_h-1}
+\sum_{b=0}^{k_w-1}
+X_i^{P}[r s_h+a d_h,c s_w+b d_w]
+K_{o,i,a,b}
+\right).
$$

例如，在某个像素位置有两个输入通道值 $3$ 和 $-1$，一个输出通道使用 $1\times1$ 核权重 $2$ 和 $4$，偏置为 $0.5$，则

$$
Z_o=3\times2+(-1)\times4+0.5=2.5.
$$

通道求和发生在每个输出空间位置。不同输出通道使用不同的 $K_{o,:,:,:}$，所以会得到多张特征图；同一个输出通道的核则跨空间位置共享。

### 参数量

带偏置的普通二维卷积参数量为

$$
\#\mathrm{params}
=C_{\mathrm{out}}C_{\mathrm{in}}k_hk_w+C_{\mathrm{out}}.
$$

例如 $C_{\mathrm{in}}=64$、$C_{\mathrm{out}}=128$、$k_h=k_w=3$ 时

$$
128\times64\times3\times3+128
=73728+128
=73856.
$$

参数量与 $H,W$ 无关；输入变大时，复用同一组参数的输出位置和计算量增加，但模型参数不随图像面积增长。

## 1×1 卷积是在每个像素做一次通道线性层

当 $k_h=k_w=1$、stride 和 dilation 都为 $1$ 时，空间窗口只包含当前位置：

$$
Z_o[r,c]
=\beta_o
+\sum_{i=0}^{C_{\mathrm{in}}-1}
+X_i[r,c]K_{o,i,0,0}.
$$

把当前位置的通道向量写成 $x_{r,c}\in\mathbb R^{C_{\mathrm{in}}}$，就可以写成

$$
z_{r,c}
=K_{1\times1}x_{r,c}+\beta.
$$

因此 1×1 卷积不读取邻近像素，却能改变通道数、混合语义并作为瓶颈层。它的参数量是 $C_{\mathrm{out}}C_{\mathrm{in}}+C_{\mathrm{out}}$，空间位置之间仍共享同一矩阵。

## 分组和深度卷积改变通道连接图

普通卷积中每个输出通道都能读取每个输入通道。分组卷积把通道划成 $G$ 组，每个输出组只连接对应的输入组。要求 $C_{\mathrm{in}}$ 和 $C_{\mathrm{out}}$ 都能被 $G$ 整除，每个输出通道看到 $C_{\mathrm{in}}/G$ 个输入通道。

带偏置的分组卷积参数量为

$$
\#\mathrm{params}_{\mathrm{grouped}}
=C_{\mathrm{out}}
\frac{C_{\mathrm{in}}}{G}
k_hk_w+C_{\mathrm{out}}.
$$

当 $G=1$ 时恢复普通卷积。当 $G=C_{\mathrm{in}}=C_{\mathrm{out}}$ 时，每个通道单独做空间卷积，这是一组 multiplier 为 $1$ 的 depthwise convolution：

$$
\#\mathrm{params}_{\mathrm{depthwise}}
=C_{\mathrm{in}}k_hk_w+C_{\mathrm{in}}.
$$

它几乎不做跨通道混合，通常需要后接 1×1 pointwise convolution 才能重新组合通道。若 depthwise multiplier 为 $m$，输出通道数为 $mC_{\mathrm{in}}$，空间核参数变为 $mC_{\mathrm{in}}k_hk_w$。

深度可分离卷积把普通卷积拆成：

1. 每个输入通道独立做 depthwise 空间卷积；
2. 用 1×1 卷积在通道轴上混合。

若输出通道为 $C_{\mathrm{out}}$，带偏置的参数量约为

$$
C_{\mathrm{in}}k_hk_w+C_{\mathrm{in}}
+C_{\mathrm{in}}C_{\mathrm{out}}+C_{\mathrm{out}}.
$$

计算和参数减少并不保证函数完全等价：拆分中间通常会插入激活、归一化或不同的数值近似，结构变化本身就是归纳偏置。

## Padding、stride 和 dilation 改变的不是同一件事

三个参数都能改变输出，但它们修改的坐标机制不同：

| 参数 | 直接改变 | 不应误解成 |
| --- | --- | --- |
| padding | 输入边界外的取值和输出覆盖范围 | 把真实像素凭空补出来 |
| stride | 相邻输出窗口起点之间的间隔 | 先缩小输入再用同一个核 |
| dilation | 核内部相邻权重在输入上的间距 | 增加可学习参数数量 |

对单个层，有效核尺寸是 $d(k-1)+1$，但参数个数仍然是 $k$ 个位置。dilation 扩大一个输出所能看到的局部跨度；stride 则减少输出位置的采样密度。两者同时使用时，输出尺寸公式必须先计算有效核尺寸。

连续多层的感受野还要考虑前面层的步幅。令第 $l$ 层之前相邻特征点在原图上的间距为 $j_{l-1}$，第 $l$ 层的有效核尺寸为 $k_l^{\mathrm{eff}}$，则

$$
j_l=j_{l-1}s_l,
\qquad
r_l=r_{l-1}+(k_l^{\mathrm{eff}}-1)j_{l-1},
$$

初始为 $j_0=1,r_0=1$。不能只把每层的核尺寸相加；前面 stride 会放大后续层每一个偏移对应的原图距离。

## Padding 的数值语义必须单独记录

边界外取零只是最常见的选择，不是二维卷积的唯一选择：

| 边界方式 | 边界外如何取值 | 适合的假设 | 风险 |
| --- | --- | --- | --- |
| zero | 取 0 | 背景值确实接近零 | 边缘可能出现一圈人为响应 |
| reflect | 镜像已有像素 | 边界延续较平滑 | 强边缘会被重复 |
| replicate | 复制最近像素 | 边界值应保持不变 | 平坦边缘可能被人为拉长 |
| circular | 按周期回绕 | 数据本身具有周期性 | 图像左右或上下产生非真实连接 |

same 也不是只由一个字母决定的唯一规则。stride 为 $1$、奇数核时两侧对称 padding 通常自然；偶数核或 stride 大于 $1$ 时，上下左右多出的一个位置如何分配会改变输出坐标。复现一个模型时，要记录每一侧的 padding，而不只记录 same。

## Channels-first 和 channels-last 只是布局，不是数学新算子

同一个批次可以采用两种常见内存布局：

| 布局 | 输入形状 | 空间坐标 | 常见混淆 |
| --- | --- | --- | --- |
| NCHW | $B\times C\times H\times W$ | 最后两个轴 | 把 batch 或 channel 当成高度 |
| NHWC | $B\times H\times W\times C$ | 中间两个轴 | 在转置后忘记更新权重布局 |

从 NCHW 改成 NHWC 只是在不改变数值语义的前提下重新排列轴。真正需要核对的是：卷积实现取哪两个轴作为空间窗口、权重张量如何排列、输出是否在同一布局下返回。batch 轴 $B$ 表示独立样本，不应与通道求和混在一起。

一个实用的形状账可以按下面顺序写：

1. 输入：$B,C_{\mathrm{in}},H,W$ 或 $B,H,W,C_{\mathrm{in}}$；
2. 权重：$C_{\mathrm{out}},C_{\mathrm{in}},k_h,k_w$，或库文档规定的变体；
3. padding、stride、dilation 的每一侧和每一个轴；
4. 输出空间尺寸 $H_{\mathrm{out}},W_{\mathrm{out}}$；
5. 输出通道数 $C_{\mathrm{out}}$ 和 bias 的广播轴。

## 反向传播是窗口贡献的累加

令上游梯度为

$$
G_o[r,c]=\frac{\partial L}{\partial Z_o[r,c]}.
$$

对卷积核的梯度来自所有输出位置使用过的输入窗口：

$$
\frac{\partial L}{\partial K_{o,i,a,b}}
=\sum_{r=0}^{H_{\mathrm{out}}-1}
\sum_{c=0}^{W_{\mathrm{out}}-1}
G_o[r,c]
X_i^{P}[r s_h+a d_h,\ c s_w+b d_w].
$$

偏置在所有空间位置累加：

$$
\frac{\partial L}{\partial \beta_o}
=\sum_{r=0}^{H_{\mathrm{out}}-1}
\sum_{c=0}^{W_{\mathrm{out}}-1}
G_o[r,c].
$$

输入方向更容易漏掉“多个窗口共享一个输入像素”这一点。对 padded 输入的某个坐标 $u,v$，只要存在

$$
u=r s_h+a d_h,
\qquad
v=c s_w+b d_w,
$$

它就会收到对应输出位置和核权重的贡献：

$$
\frac{\partial L}{\partial X_i^{P}[u,v]}
=\sum_{o,r,c,a,b}
G_o[r,c]K_{o,i,a,b}
\mathbf 1_{\{u=r s_h+a d_h,\ v=c s_w+b d_w\}}.
$$

实现时必须累加而不是覆盖。stride、dilation 和边界会改变哪些位置相遇，但不会改变“所有有效路径都要相加”的原则。padding 后的梯度还要裁回原始 $H\times W$ 区域，padding 本身不是需要更新的参数。

## 参数量和计算量要分开

普通卷积的参数量只由通道数、核大小和 bias 决定：

$$
P=C_{\mathrm{out}}C_{\mathrm{in}}k_hk_w+C_{\mathrm{out}}.
$$

单个样本的乘加数量近似为

$$
\mathrm{MACs}
=H_{\mathrm{out}}W_{\mathrm{out}}
C_{\mathrm{out}}C_{\mathrm{in}}k_hk_w.
$$

因此增加输入分辨率通常不增加参数量，却会增加激活内存和 MACs。分组、depthwise 与 stride 可以降低计算，但可能改变通道交互、空间采样或信息保留。

| 结构 | 每个输出位置的通道连接 | 参数量（含 bias） | 主要取舍 |
| --- | --- | --- | --- |
| 普通卷积 | 每个输出通道连接所有输入通道 | $C_{\mathrm{out}}C_{\mathrm{in}}k_hk_w+C_{\mathrm{out}}$ | 表达力强，成本最高 |
| 分组卷积 | 每组内连接输入通道 | $C_{\mathrm{out}}(C_{\mathrm{in}}/G)k_hk_w+C_{\mathrm{out}}$ | 降低成本，组间不直接交流 |
| depthwise | 每个输入通道独立 | $C_{\mathrm{in}}k_hk_w+C_{\mathrm{in}}$ | 空间滤波便宜，通道混合弱 |
| 1×1 卷积 | 同一像素内混合通道 | $C_{\mathrm{out}}C_{\mathrm{in}}+C_{\mathrm{out}}$ | 改通道数，不扩展空间邻域 |

只比较参数量会漏掉输出空间大小、内存访问、临时 im2col 缓冲区和硬件 kernel 的影响。结构审计要同时记录参数、MACs、峰值激活和布局。

## 常见失效模式

| 现象 | 可能原因 | 第一条证据 |
| --- | --- | --- |
| 输出高宽少一列或多一行 | 有效核尺寸、padding 或 floor 约定错 | 写出最后一个合法 padded 坐标 |
| 结果整体左右或上下翻转 | 互相关与数学卷积约定不同 | 用非对称二维核追踪四个角 |
| 只有边缘出现异常 | zero、reflect、replicate 或 circular 语义混用 | 单独比较中心窗口和边界窗口 |
| 通道数正确但数值不对 | 漏掉输入通道求和或权重轴顺序错 | 固定一个像素手算所有通道贡献 |
| 1×1 层改变了邻域 | 把 pointwise 误当成普通空间核 | 检查 $k_h=k_w=1$ 的窗口坐标 |
| 分组层结果像少了信息 | group 划分与通道排列不一致 | 打印每组输入/输出通道集合 |
| dilation 后输出尺寸错误 | 仍使用 $k$ 而不是 $d(k-1)+1$ | 计算最后一个实际访问坐标 |
| stride 后梯度有空洞或覆盖 | 输入梯度没有按所有窗口累加 | 对一个输入像素列出贡献路径 |
| NCHW/NHWC 互换后形状仍像对 | 只看 shape 没看空间轴与通道轴 | 用每个轴不同的长度或数值模式测试 |
| 训练前向正确、反向不稳定 | bias 归约或 padding 裁回漏项 | 对小张量做逐参数中心差分 |

其中“形状正确”是最容易造成假安全感的一项。只要 $H,W,C$ 恰好相等，轴交换甚至可能完全不改变 shape；测试张量必须让每个轴使用不同的长度或不同的数值模式。

## 读二维卷积实现的验收顺序

面对论文、框架层或自写 kernel，按以下顺序核对：

1. 先写清输入、权重、偏置和输出的轴顺序；
2. 用一个不等长且各轴数值不同的张量标记坐标；
3. 写出一个输出位置的完整通道和窗口求和；
4. 明确这是互相关还是两个空间轴都翻转的数学卷积；
5. 分别记录四侧 padding、stride、dilation 和输出尺寸；
6. 再核对 groups、depthwise、1×1 与通道连接图；
7. 用同一个上游梯度检查 kernel、bias 和 input 的累加；
8. 最后比较参数量、MACs、布局转换和部署内核。

只要一个输出数字能够按“输入通道 → 窗口坐标 → 核参数 → bias”逐项复算，二维卷积的 shape、方向和反向传播通常都能被局部证据锁定。

## 相关词条

- [卷积神经网络](../cnn/cnn/)：从局部连接、权值共享和空间归纳偏置概览卷积网络。
- [互相关与卷积](../cnn/cross-correlation-vs-convolution/)：核翻转、API 命名和二维双轴方向的索引对照。
- [离散卷积](../cnn/discrete-convolution/)：一维和二维离散卷积的支持集、边界与代数性质。
- [步幅、填充与膨胀](../cnn/stride-padding-dilation/)：专门展开三个空间采样参数及其输出尺寸。
- [输出尺寸算术](../cnn/output-size-arithmetic/)：把卷积、池化和多层空间尺寸逐层算清。
- [池化](../cnn/pooling/)：比较固定聚合窗口与可学习卷积窗口的下采样差异。
- [卷积作为 Toeplitz 结构](../cnn/convolution-as-toeplitz/)：把共享窗口写成稀疏结构矩阵。
- [向量化反向传播](../backpropagation/vectorized-backprop/)：把局部梯度累加连接到批次矩阵公式。
- [不变性与等变性](../cnn/invariance-and-equivariance/)：连接空间变换和特征图响应。
- [为什么需要卷积](../cnn/why-convolution/)：解释二维局部性与跨位置共享为何是有用先验。
