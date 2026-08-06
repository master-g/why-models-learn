---
title: "卷积作为 Toeplitz 结构：把共享窗口写成稀疏矩阵"
tags: ["why-models-learn"]
---

卷积可以写成一个稀疏结构矩阵乘以输入向量。矩阵中的零表示局部连接，沿对角线重复的权重表示跨位置共享，padding 和 stride 则改变矩阵的边界行与行采样。把卷积写成 Toeplitz 或 block Toeplitz 结构，不是为了真的构造一个巨大矩阵，而是为了同时看清前向、参数共享、反向传播和实现优化。

本文先从一维 valid 互相关写出带状 Toeplitz 矩阵，再加入 padding、stride、dilation 和二维 block Toeplitz with Toeplitz blocks 结构。随后把多通道卷积写成块矩阵，说明 im2col 为什么等价于稀疏窗口抽取，最后用矩阵转置推导输入梯度、核梯度和转置卷积。深度学习库通常实现互相关而不是预先翻转数学卷积；本文先采用这个常见窗口方向，并在相应位置说明翻转如何改变矩阵。

![卷积作为 Toeplitz 结构：一维共享核形成带状矩阵，二维形成 block Toeplitz with Toeplitz blocks，反向传播使用同一结构的转置](/assets/cnn/svg/convolution-as-toeplitz.1.svg)

## 一维 valid 卷积就是带状 Toeplitz 矩阵

设输入为

$$
x=
\begin{bmatrix}
x_0\\x_1\\x_2\\x_3\\x_4
\end{bmatrix},
$$

长度为 $K=3$ 的窗口权重为 $w_0,w_1,w_2$。采用深度学习库常见的不翻转互相关：

$$
y_r
=\sum_{a=0}^{2}w_ax_{r+a},
\qquad
r=0,1,2.
$$

三个输出逐项展开：

$$
\begin{aligned}
y_0&=w_0x_0+w_1x_1+w_2x_2,\\
y_1&=w_0x_1+w_1x_2+w_2x_3,\\
y_2&=w_0x_2+w_1x_3+w_2x_4.
\end{aligned}
$$

把输入和输出排成向量：

$$
\begin{bmatrix}
y_0\\y_1\\y_2
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
w_0&w_1&w_2&0&0\\
0&w_0&w_1&w_2&0\\
0&0&w_0&w_1&w_2
\end{bmatrix}
}_{T_w}
\begin{bmatrix}
x_0\\x_1\\x_2\\x_3\\x_4
\end{bmatrix}.
$$

于是

$$
y=T_wx.
$$

矩阵沿每条从左上到右下的对角线使用相同元素：

$$
(T_w)_{r,j}
=
\begin{cases}
w_{j-r},&0\le j-r<K,\\
0,&\text{其他情况}.
\end{cases}
$$

这就是 Toeplitz 的核心结构：矩阵元素只依赖列索引与行索引之差。由于只有主带及其附近的 $K-1$ 条带非零，它同时是带状矩阵和稀疏矩阵。

### 矩阵中的两个先验

上面的矩阵把 CNN 的两个结构先验变成了可见的代数事实：

| 矩阵结构 | 卷积语义 | 换成全连接后的结果 |
| --- | --- | --- |
| 大量固定零 | 每个输出只看局部窗口 | 允许任意输入位置连接 |
| 每条对角线重复 $w_a$ | 同一个权重跨位置共享 | 每个输入输出对拥有独立权重 |
| 带状非零区域 | 局部性和有限感受野 | 单层可以拥有全局交互 |
| 行只是窗口平移 | 平移后使用同一个规则 | 位置可以学习不同规则 |

局部连接和权值共享是两个独立开关。只把矩阵做成带状而不给不同窗口复用权重，会得到局部但不共享的算子；只复用权重但允许每一行连接所有输入，则得到全局共享的特殊线性层。Toeplitz 结构同时表达了这两条约束。

## 数值例子：矩阵乘法和滑动窗口相同

取

$$
x=[1,2,3,4,5]^{\mathsf T},
\qquad
w=[2,-1,1].
$$

对应矩阵为

$$
T_w=
\begin{bmatrix}
2&-1&1&0&0\\
0&2&-1&1&0\\
0&0&2&-1&1
\end{bmatrix}.
$$

矩阵乘法给出

$$
y=T_wx
=
\begin{bmatrix}
3\\5\\7
\end{bmatrix}.
$$

逐窗口计算得到同一结果：

$$
[2\cdot1-1\cdot2+1\cdot3,\quad
2\cdot2-1\cdot3+1\cdot4,\quad
2\cdot3-1\cdot4+1\cdot5]
=[3,5,7].
$$

矩阵只是把“窗口向右移动”改写成“下一行沿对角线复制同一权重”。它没有改变算子，只是让连接关系可以被线性代数的工具直接观察。

若采用数学卷积，把核先翻转为

$$
\widetilde w=[w_2,w_1,w_0],
$$

矩阵每一行就会使用 $\widetilde w$ 的顺序。Toeplitz 的稀疏和共享结构仍然存在，变化的是非零对角线分别对应哪个核索引。

## Padding 改变边界行

输入长度为 $4$，核仍为 $w_0,w_1,w_2$，左右各补一个零，stride 为 $1$。padded 输入是

$$
x^P=[0,x_0,x_1,x_2,x_3,0].
$$

输出长度为 $4$，矩阵写成

$$
\begin{bmatrix}
y_0\\y_1\\y_2\\y_3
\end{bmatrix}
=
\begin{bmatrix}
w_1&w_2&0&0\\
w_0&w_1&w_2&0\\
0&w_0&w_1&w_2\\
0&0&w_0&w_1
\end{bmatrix}
\begin{bmatrix}
x_0\\x_1\\x_2\\x_3
\end{bmatrix}.
$$

第一行和最后一行少看一个真实输入位置，因为另一个位置来自 padding。中间行仍然包含完整的三个权重。零并没有被作为输入向量的一列永久存储，而是被消去后写成了边界行中的零系数。

对称 same padding 在这个奇数核例子中仍然形成规则的 Toeplitz 带状矩阵，但边界行的非零带被截断。换成 reflect 或 replicate padding，边界行可能出现重复的输入系数；换成 circular padding，边界行会出现首尾输入列，矩阵趋向 circulant 结构。

这说明“矩阵是 Toeplitz”还不够完整。必须同时注明：

- 输入是否有限支持或周期延拓；
- 数学卷积还是不翻转互相关；
- padding 值和四侧分配；
- 输出行的数量；
- 边界行是否来自截断、镜像、复制或回绕。

## Stride 是选择矩阵行

先构造 stride 为 $1$ 的完整窗口矩阵 $T_w^{(1)}$，再只保留每隔 $S$ 行的输出，可以写成

$$
y=R_ST_w^{(1)}x.
$$

其中 $R_S$ 是行选择矩阵。例如 stride 为 $2$ 时：

$$
R_2=
\begin{bmatrix}
1&0&0&\cdots\\
0&0&1&\cdots\\
0&0&0&0&1&\cdots
\end{bmatrix}.
$$

它不改变每个保留窗口内部的权重，只减少输出位置。于是 stride 的两种等价理解是：

1. 窗口起点每次移动 $S$ 格；
2. 先写出 stride 为 $1$ 的 Toeplitz 矩阵，再抽取部分行。

第二种写法对反向传播很有用。输入梯度先经过 $T_w^{(1)\mathsf T}$ 的散射，再由 $R_S^{\mathsf T}$ 把上游梯度放回被保留的行；没有被 stride 采样到的窗口没有对应的上游梯度。

输出尺寸的 floor 或 ceil 规则决定 $R_S$ 选择多少行。若最后一个窗口不完整，是否通过 padding 补齐会改变矩阵的最后一行，不能只在行选择矩阵外部处理。

## Dilation 在带状矩阵中留下空洞

长度为 $K=3$、dilation 为 $D=2$ 的窗口读取

$$
x_r,\qquad x_{r+2},\qquad x_{r+4}.
$$

对应的行形状是

$$
\begin{bmatrix}
w_0&0&w_1&0&w_2&0&\cdots\\
0&w_0&0&w_1&0&w_2&\cdots\\
\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\ddots
\end{bmatrix}.
$$

非零对角线之间出现空列，参数数量仍是 $3$，有效核尺寸却是

$$
K_{\mathrm{eff}}
=D(K-1)+1
=5.
$$

因此 dilation 同时改变两件事：矩阵非零带之间的间距，以及输出尺寸公式中的有效窗口跨度。若只看非零元素个数，会低估感受野和边界约束。

## 二维卷积形成 block Toeplitz with Toeplitz blocks

把一个 $3\times3$ 输入按行展开：

$$
\operatorname{vec}(X)
=
[x_{0,0},x_{0,1},x_{0,2},
x_{1,0},x_{1,1},x_{1,2},
x_{2,0},x_{2,1},x_{2,2}]^{\mathsf T}.
$$

使用 $2\times2$ 核

$$
K=
\begin{bmatrix}
a&b\\c&d
\end{bmatrix},
$$

采用 valid 互相关，输出空间为 $2\times2$。按行展开输出，四个窗口对应的矩阵是

$$
\begin{bmatrix}
y_{0,0}\\y_{0,1}\\y_{1,0}\\y_{1,1}
\end{bmatrix}
=
\begin{bmatrix}
a&b&0&c&d&0&0&0&0\\
0&a&b&0&c&d&0&0&0\\
0&0&0&a&b&0&c&d&0\\
0&0&0&0&a&b&0&c&d
\end{bmatrix}
\operatorname{vec}(X).
$$

每一行是一个二维局部窗口；向右移动时，非零列在同一个行块内右移；向下移动时，整个行块向后移动一行宽度。水平结构是 Toeplitz，垂直方向又以 Toeplitz 方式组织这些块，所以称为 block Toeplitz with Toeplitz blocks，简称 BTTB。

如果使用相同的空间参数、零 padding 和多个输出通道，二维结构仍然由共享核的重复非零块构成。padding 只会在边界块中删除或替换列，stride 则选择二维行块中的部分输出行。

### 通道让标量 Toeplitz 变成块矩阵

对 channels-first 输入，先把每个输入 channel 的空间向量串接：

$$
x=
\begin{bmatrix}
x^{(0)}\\
x^{(1)}\\
\vdots\\
x^{(C_{\mathrm{in}}-1)}
\end{bmatrix}.
$$

输出 channel $o$ 的空间结果是

$$
y^{(o)}
=
\sum_{c=0}^{C_{\mathrm{in}}-1}
T_{W_{o,c}}x^{(c)}
+b_o\boldsymbol1.
$$

把所有输出 channel 叠起来：

$$
y=
\underbrace{
\begin{bmatrix}
T_{W_{0,0}}&T_{W_{0,1}}&\cdots&T_{W_{0,C_{\mathrm{in}}-1}}\\
T_{W_{1,0}}&T_{W_{1,1}}&\cdots&T_{W_{1,C_{\mathrm{in}}-1}}\\
\vdots&\vdots&\ddots&\vdots\\
T_{W_{C_{\mathrm{out}}-1,0}}&
T_{W_{C_{\mathrm{out}}-1,1}}&
\cdots&
T_{W_{C_{\mathrm{out}}-1,C_{\mathrm{in}}-1}}
\end{bmatrix}
}_{\mathcal T_W}
x+\text{bias}.
$$

每个块 $T_{W_{o,c}}$ 都是一个空间 Toeplitz 或 BTTB 矩阵。通道混合发生在块之间，空间局部性发生在块内部。

分组卷积把不属于同一组的块置为零；depthwise 卷积只保留通道对应的对角块；$1\times1$ 卷积的空间块退化为逐位置的 channel mixing。矩阵结构直接展示了三种算子在“空间连接”和“通道连接”上的差异。

## im2col 是显式窗口抽取，不是另一种卷积

显式 Toeplitz 矩阵包含大量零，直接存储通常浪费内存。im2col 的思路是先把每个输出位置需要的局部 patch 排成一行：

$$
X_{\mathrm{col}}
=P x,
$$

其中 $P$ 是由 one-hot 选择行组成的稀疏 patch 抽取矩阵。对单个输出 channel：

$$
y
=X_{\mathrm{col}}w
=(Px)w.
$$

更严格地，把每一行 patch 与核向量做内积，可以写成

$$
y
=\left(I_O\otimes w^{\mathsf T}\right)Px,
$$

其中 $O$ 是输出位置数，$\otimes$ 是 Kronecker 积。这个乘积的结果就是隐式 Toeplitz 矩阵：

$$
T_w
=\left(I_O\otimes w^{\mathsf T}\right)P.
$$

实际实现通常不真的形成 $T_w$，而是：

1. 用索引或专门 kernel 抽取 patch；
2. 把 patch 矩阵交给 GEMM；
3. 将结果 reshape 回输出 channel 和空间布局。

im2col 让硬件更容易使用高效矩阵乘法，但会显式保存重复的输入元素。一个输入像素可能出现在多个 patch 中，临时缓冲区的大小大约与输出位置数乘窗口大小成正比。

### 多通道 im2col

若每个 patch 连接所有输入 channel，单个 patch 的长度为

$$
C_{\mathrm{in}}K_hK_w.
$$

把所有 patch 作为行，权重展平为

$$
W_{\mathrm{col}}
\in
\mathbb R^{C_{\mathrm{out}}\times
(C_{\mathrm{in}}K_hK_w)}.
$$

则

$$
Y_{\mathrm{col}}
=X_{\mathrm{col}}W_{\mathrm{col}}^{\mathsf T}.
$$

分组和 depthwise 卷积可以通过拆分 patch 或 block-diagonal 权重矩阵实现。它们节省乘加，不会改变“局部 patch 抽取加线性组合”的基本结构。

## 前向、输入梯度和核梯度都来自同一矩阵

把单输出通道的前向写成

$$
y=T_wx+b\boldsymbol1.
$$

给定上游梯度 $g=\partial L/\partial y$：

### 输入梯度是矩阵转置的散射

$$
\frac{\partial L}{\partial x}
=T_w^{\mathsf T}g.
$$

矩阵转置把每个输出位置的梯度沿原窗口连接散射回输入；多个窗口同时覆盖一个输入位置时，转置乘法会自动相加。它不是“只把梯度传给最后一个窗口”，而是把所有非零连接的贡献累积起来。

### 偏置梯度是空间归约

若同一个 bias $b$ 用于所有输出位置：

$$
\frac{\partial L}{\partial b}
=\boldsymbol1^{\mathsf T}g
=\sum_{r=0}^{O-1}g_r.
$$

多 batch 和多空间位置时，偏置梯度还要沿 batch 和空间轴归约。矩阵视角能防止只沿最后一个轴求和的错误。

### 核梯度沿重复对角线累加

第 $a$ 个权重出现在 $T_w$ 的多条相同对角线上：

$$
\frac{\partial L}{\partial w_a}
=\sum_{r=0}^{O-1}g_rx_{r+a}
$$

在 padding、stride 和 dilation 存在时，$x_{r+a}$ 要替换成该窗口实际读取的输入坐标。这个式子说明权值共享为何在反向传播中表现为跨空间位置的梯度求和。

## 数值核对：矩阵转置和窗口累加一致

沿用

$$
x=[1,2,3,4,5]^{\mathsf T},
\qquad
w=[2,-1,1],
\qquad
g=[1,2,3]^{\mathsf T}.
$$

前向仍是

$$
y=[3,5,7]^{\mathsf T}.
$$

输入梯度：

$$
T_w^{\mathsf T}g
=
\begin{bmatrix}
2&0&0\\
-1&2&0\\
1&-1&2\\
0&1&-1\\
0&0&1
\end{bmatrix}
\begin{bmatrix}1\\2\\3\end{bmatrix}
=
\begin{bmatrix}2\\3\\5\\-1\\3\end{bmatrix}.
$$

核梯度按每个窗口的输入累加：

$$
\frac{\partial L}{\partial w_0}
=1\cdot1+2\cdot2+3\cdot3=14,
$$

$$
\frac{\partial L}{\partial w_1}
=1\cdot2+2\cdot3+3\cdot4=20,
\qquad
\frac{\partial L}{\partial w_2}
=1\cdot3+2\cdot4+3\cdot5=26.
$$

所以

$$
\frac{\partial L}{\partial w}
=[14,20,26].
$$

同一组数字可以用滑动窗口反向累加得到完全相同的结果。矩阵转置不是额外的近似，而是局部反向传播的精确线性代数表达。

## 转置矩阵不是普通卷积的逆

如果前向是

$$
y=T_wx,
$$

反向使用的是

$$
T_w^{\mathsf T}g,
$$

而不是

$$
T_w^{-1}g.
$$

两者差异有明确原因：

- $T_w$ 通常不是方阵，valid 卷积会减少空间长度；
- 即使矩阵是方阵，也不保证可逆；
- 转置保留了原连接图的方向反转和重叠累加；
- stride 丢掉的输出行不会被转置自动恢复；
- padding 行中的边界语义会影响转置后的输出位置。

把 $T_w^{\mathsf T}$ 组织成一个空间算子，就是转置卷积常见的矩阵来源。它可以扩大空间尺寸，但不是从低分辨率输入恢复原始高分辨率信号的逆函数。output padding 只选择离散输出 shape，不会补回被前向压缩掉的信息。

## 循环边界把 Toeplitz 变成 circulant

若输入采用周期边界，长度为 $N$ 的一维卷积矩阵不再在边缘截断，而是把右侧越界列绕回左侧。以核 $w_0,w_1,w_2$ 和长度 $N=5$ 为例，矩阵为

$$
C_w=
\begin{bmatrix}
w_0&w_1&w_2&0&0\\
0&w_0&w_1&w_2&0\\
0&0&w_0&w_1&w_2\\
w_2&0&0&w_0&w_1\\
w_1&w_2&0&0&w_0
\end{bmatrix}.
$$

每一行是上一行的循环移位，这类矩阵称为 circulant。离散 Fourier 基可以同时对角化所有 circulant 矩阵：

$$
F C_w F^{-1}
=\operatorname{diag}(\widehat w).
$$

于是循环卷积在频域变成逐频率相乘：

$$
\widehat y[\ell]
=\widehat w[\ell]\widehat x[\ell].
$$

有限 zero padding 的 Toeplitz 矩阵通常不能直接被同一个 DFT 完全对角化，需要嵌入更大的 circulant 矩阵并处理边界。FFT 卷积的补零长度和边界约定因此不是性能实现的旁枝，而是数值结果的一部分。

## 参数量、稀疏度和实际计算

设一维输入长度为 $I$、输出位置数为 $O$、核长为 $K$：

| 表示方式 | 存储规模 | 乘加量 | 主要问题 |
| --- | --- | --- | --- |
| 显式 dense 矩阵 | $O I$ | $O I$ | 忽略局部零结构，存储浪费 |
| 显式 Toeplitz | 约 $O K$ 个非零 | $O K$ | 仍需管理稀疏访问和边界 |
| 直接滑窗 | $K$ 个核参数 | $O K$ | 访存和向量化决定实际速度 |
| im2col 加 GEMM | patch 缓冲约 $O K$ | $O K$ | 重复 patch 带来额外内存 |
| FFT 或频域方法 | 与变换长度有关 | 适合大核或长信号 | padding、复数和变换开销 |

参数量是 $K$，不是 $O K$；同一个核被 $O$ 个输出位置反复使用。计算量却随输出位置数增长。二维多通道时，非零连接数大致为

$$
O_HW C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w,
$$

而参数量大致为

$$
C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w
+C_{\mathrm{out}}.
$$

这正是卷积可以参数高效但计算仍昂贵的矩阵解释：结构矩阵很稀疏，非零模式却在许多空间位置重复出现。

## 常见失效模式

| 现象 | 可能原因 | 第一条证据 |
| --- | --- | --- |
| 矩阵输出和滑窗输出相反 | 把互相关误当成翻转后的数学卷积 | 用非对称核逐行展开 |
| 输出少一行或多一行 | padding、stride、floor 或最后窗口错 | 记录 Toeplitz 的行数和最后一行 |
| 边界响应异常 | 边界行没有使用真实 padding 语义 | 单独打印第一行和最后一行 |
| dilation 结果跨度不对 | 只按非零权重个数计算 | 检查非零对角线之间的空洞 |
| 多通道结果 shape 正确但数值错 | block 顺序或 channel layout 错 | 明确每个 $T_{W_{o,c}}$ 所在块 |
| 分组卷积仍发生跨组混合 | 本应为零的块没有清除 | 检查块矩阵的非对角组块 |
| 输入梯度少了重叠贡献 | 把转置当作逐位置复制而非 scatter-add | 对一个输入位置求所有覆盖行 |
| 核梯度没有跨空间累加 | 把共享权重当成每行独立参数 | 检查同一对角线的梯度总和 |
| im2col 很快但显存暴涨 | 重复保存所有局部 patch | 记录 patch 缓冲区峰值 |
| FFT 与直接卷积边界不一致 | 补零长度或周期边界不同 | 用短非对称信号比较完整输出 |
| 把转置卷积当作逆 | 忽略矩阵非方阵和信息丢失 | 比较 $T_w^{\mathsf T}$ 与逆矩阵的定义 |

## 读卷积矩阵的验收顺序

面对一个新的卷积实现，可以按下面顺序建立结构证据：

1. 先固定互相关或数学卷积的核方向；
2. 写出输入、输出和按 channel 或空间布局展开的向量顺序；
3. 用一个小输入显式列出第一行、中心行和最后一行；
4. 检查非零列是否对应局部窗口，重复对角线是否对应共享权重；
5. 将 padding 写进边界行，将 stride 写成行选择，将 dilation 写成空洞对角线；
6. 二维时检查 block Toeplitz 的行块和块内 Toeplitz 结构；
7. 多通道时列出输出通道到输入通道的块矩阵，确认 group/depthwise 的零块；
8. 用 im2col 或直接滑窗独立核对前向；
9. 用矩阵转置独立核对输入梯度和重叠累加，用共享对角线核对核梯度；
10. 最后再比较直接、GEMM、稀疏或 FFT 实现的性能。

Toeplitz 结构把“局部、共享、可反向传播”放进同一个矩阵对象。它既解释了卷积为何比全连接更节省参数，也解释了为什么 stride、padding、通道布局和边界语义会在前向与反向同时留下痕迹。实现可以不显式构造这个矩阵，但审计时仍然可以把它作为最清楚的参考模型。

## 相关词条

[二维卷积](../cnn/convolution-2d/)

[互相关与卷积](../cnn/cross-correlation-vs-convolution/)

[离散卷积](../cnn/discrete-convolution/)

[步幅、填充与膨胀](../cnn/stride-padding-dilation/)

[输出尺寸算术](../cnn/output-size-arithmetic/)

[向量化反向传播](../backpropagation/vectorized-backprop/)

[不变性与等变性](../cnn/invariance-and-equivariance/)

[为什么需要卷积](../cnn/why-convolution/)
