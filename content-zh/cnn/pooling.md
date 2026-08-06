---
title: "池化：固定局部聚合如何下采样"
tags: ["why-models-learn"]
---

池化是对局部窗口施加固定聚合规则的空间算子，常见规则是取最大值或平均值。它没有卷积核那样的可学习权重，却仍然会改变输出尺寸、保留的信息、梯度流向、平移响应和后续层的感受野。把池化只理解成“把图像缩小”，会漏掉窗口边界、padding、取整、通道独立性和下采样混叠等真正决定行为的细节。

本文先在一维数组上定义窗口和聚合，再用二维数值例子核对 max pooling、average pooling 和 sum pooling。随后讨论 padding、ceil mode、反向传播、全局平均池化、与 stride 卷积的差异、下采样的相位与混叠风险，最后把池化放回多层网络的 shape ledger 和感受野账本中。输出尺寸的通用推导见[输出尺寸算术](../cnn/output-size-arithmetic/)，padding、stride 和 dilation 的坐标定义见[步幅、填充与膨胀](../cnn/stride-padding-dilation/)。

![池化的固定局部聚合：同一个二维窗口分别产生最大值和平均值，stride 让输出空间从四乘四变成两乘二](/assets/cnn/svg/pooling.1.svg)

## 池化先固定窗口，再固定聚合规则

设一维输入为 $x[0],\ldots,x[I-1]$，窗口长度为 $K$，stride 为 $S$，dilation 为 $D$，左、右 padding 为 $P_{\mathrm{left}},P_{\mathrm{right}}$。有效窗口尺寸为

$$
K_{\mathrm{eff}}
=D(K-1)+1.
$$

把输入按约定扩展成 padded 数组 $x^P$ 后，第 $r$ 个输出位置的窗口读取坐标是

$$
q_{r,a}
=rS+aD,
\qquad
0\le a<K.
$$

也可以把这个窗口写成一个坐标集合：

$$
\mathcal N_r
=\left\{
rS-P_{\mathrm{left}}+aD
\mathrel{\mid}
0\le a<K
\right\}.
$$

这个集合只描述窗口覆盖的原始坐标；越过输入边界的坐标如何取值，要由 padding 规则另外决定。池化的特殊之处在于，窗口确定之后不再与可学习权重相乘，而是调用一个固定聚合函数 $\phi$：

$$
y[r]
=\phi\left(x^P[rS],x^P[rS+D],\ldots,x^P[rS+D(K-1)]\right).
$$

因此，池化仍然继承卷积的空间尺寸算术：

$$
O
=\left\lfloor
\frac{
I+P_{\mathrm{left}}+P_{\mathrm{right}}-K_{\mathrm{eff}}
}{S}
\right\rfloor+1.
$$

它不继承卷积的参数量和通道混合：常见池化在每个通道内独立处理，同一个固定规则用于所有空间位置和所有通道。

## 三种基本聚合

### Max pooling：保留窗口中的最大响应

最大池化定义为

$$
y_{\mathrm{max}}[r]
=\max_{0\le a<K}x^P[rS+aD].
$$

它回答的问题是“这个局部窗口里是否出现了足够强的响应”。在边缘、纹理或局部检测器输出上，最大池化可以保留最强证据，即使证据只落在窗口中的一个位置。

最大池化不是求平均，也不是只选择窗口中心。窗口内其他位置的数值仍然决定最大值是否改变，但在最大值确定之后，它们不会直接贡献前向输出。

### Average pooling：保留局部平均

不考虑 padding 参与规则时，平均池化为

$$
y_{\mathrm{avg}}[r]
=\frac{1}{K}
\sum_{a=0}^{K-1}x^P[rS+aD].
$$

平均池化保留的是局部总量的平均水平，通常比最大池化更平滑，也更容易把尖锐响应摊薄。它可以看作一个固定的 box filter 加上 stride，但 box filter 不是所有信号上的充分抗混叠低通滤波器。

当窗口触碰 padding 时，分母不能默认写成 $K$。如果只对真实输入元素求平均，设窗口中有效元素的索引集合为 $\mathcal V_r$，则

$$
y_{\mathrm{avg,exclude}}[r]
=\frac{1}{\lvert\mathcal V_r\rvert}
\sum_{a\in\mathcal V_r}x[rS-P_{\mathrm{left}}+aD].
$$

如果把 padding 数值也当作窗口元素，分母则保持为窗口大小。两种规则在中心窗口相同，在边界窗口可能明显不同。

### Sum pooling：保留局部总量

求和池化定义为

$$
y_{\mathrm{sum}}[r]
=\sum_{a=0}^{K-1}x^P[rS+aD].
$$

它不做除法，所以窗口大小会直接改变输出的尺度。对计数、局部能量或需要保留总量的特征，sum pooling 可能比 average pooling 合适；对不同窗口大小之间需要可比幅度的特征，则要明确归一化策略。

## 一维数字例子：同一窗口产生三种不同证据

取输入

$$
x=[1,3,2,5,4,0,6],
$$

窗口长度 $K=3$，stride $S=2$，左右各补一个零，dilation $D=1$。padded 输入是

$$
x^P=[0,1,3,2,5,4,0,6,0].
$$

有效核尺寸为 $3$，输出长度是

$$
O
=\left\lfloor\frac{7+1+1-3}{2}\right\rfloor+1
=4.
$$

四个窗口从起点 $0,2,4,6$ 开始：

$$
[0,1,3],\qquad
[3,2,5],\qquad
[5,4,0],\qquad
[0,6,0].
$$

逐个聚合：

$$
y_{\mathrm{max}}
= [3,5,5,6],
$$

$$
y_{\mathrm{sum}}
= [4,10,9,6],
$$

若平均池化把 padding 计入分母：

$$
y_{\mathrm{avg,include}}
=\left[
\frac43,\frac{10}{3},3,2
\right].
$$

若平均池化只按真实输入元素归一化，左右边界的分母分别是 $2$ 和 $1$：

$$
y_{\mathrm{avg,exclude}}
=\left[
2,\frac{10}{3},3,6
\right].
$$

同样的窗口和同样的输出长度，已经产生了四种不同的数值语义。尤其是最后一个窗口，include-pad 的平均值为 $2$，exclude-pad 的平均值为 $6$；如果不记录分母规则，模型复现时即使 shape 完全一致，边界数值也会不同。

## 二维池化要沿两个空间轴展开

设输入是单通道的 $4\times4$ 矩阵：

$$
X=
\begin{bmatrix}
1&4&2&0\\
3&2&5&1\\
0&6&1&2\\
4&3&2&7
\end{bmatrix}.
$$

使用 $2\times2$ 窗口、两个方向 stride 都为 $2$，不添加 padding。高度和宽度分别为

$$
H_{\mathrm{out}}
=\left\lfloor\frac{4-2}{2}\right\rfloor+1
=2,
\qquad
W_{\mathrm{out}}
=\left\lfloor\frac{4-2}{2}\right\rfloor+1
=2.
$$

四个窗口是

$$
\begin{bmatrix}1&4\\3&2\end{bmatrix},
\quad
\begin{bmatrix}2&0\\5&1\end{bmatrix},
\quad
\begin{bmatrix}0&6\\4&3\end{bmatrix},
\quad
\begin{bmatrix}1&2\\2&7\end{bmatrix}.
$$

最大池化、平均池化和求和池化分别得到

$$
Y_{\mathrm{max}}
=
\begin{bmatrix}
4&5\\
6&7
\end{bmatrix},
\qquad
Y_{\mathrm{avg}}
=
\begin{bmatrix}
2.5&2\\
3.25&3
\end{bmatrix},
$$

$$
Y_{\mathrm{sum}}
=
\begin{bmatrix}
10&8\\
13&12
\end{bmatrix}.
$$

二维池化通常对每个 channel 独立进行。若输入 shape 是 $(N,C,H,W)$，$N$ 和 $C$ 不因空间池化改变，输出通常是

$$
(N,C,H_{\mathrm{out}},W_{\mathrm{out}}).
$$

池化跨空间窗口聚合，但不默认在 channel 轴上取最大值或平均值。把 channel 也放入池化窗口，会改变特征语义和输出 channel 数，必须作为另一个明确的算子合同记录。

## Padding 决定边界看见什么

### Max pooling 不能随便用零填充

对卷积来说，零 padding 是常见默认；对最大池化，如果真实特征可能为负，零会变成一个人为的强响应。例如窗口中的真实值为

$$
[-3,-2],
$$

如果右侧补零，最大值变成 $0$，而不是输入窗口的最大值 $-2$。因此 max pooling 的“虚拟边界”常使用负无穷语义，或只允许完整窗口：

$$
\max(-3,-2,-\infty)=-2.
$$

具体框架可能用最小可表示数近似负无穷，也可能限制 max pooling 的 padding 选项。审计时要问的不是“有没有 padding”，而是“padding 值能否赢得这个聚合”。

### Average pooling 要记录分母

平均池化至少有两种边界合同：

| 规则 | 分子 | 分母 | 边界效果 |
| --- | --- | --- | --- |
| include padding | 把 padding 值加入求和 | 固定窗口元素数 | zero padding 会把边缘平均值压向零 |
| exclude padding | 只累加真实输入 | 有效元素数量 | 边缘保持真实值尺度，但每个位置分母不同 |
| valid | 不让窗口接触边界 | 固定窗口元素数 | 丢弃边缘输出，shape 更小 |
| reflect / replicate | 用镜像或最近值扩展 | 固定窗口元素数 | 边缘不是零，但会引入重复或镜像结构 |

在图像分类中，边缘差异有时只影响少量位置；在密集预测、计数和全局平均汇聚中，边界像素的权重会累计到最终输出，不能把它当成无关细节。

### Padding 和输出相位一起决定对齐

左侧 padding 会改变输出索引映射回原输入的坐标。窗口起点为 $rS$ 时，未 padding 坐标的左端点可写为

$$
u_r=rS-P_{\mathrm{left}}.
$$

两个池化层即使输出 shape 相同，只要左、右 padding 分配不同，输出特征对应的输入中心就可能错开。残差相加、跳跃连接和上采样拼接对这种相位差尤其敏感。

## Floor、ceil mode 和最后一个窗口

floor 模式只保留完整窗口：

$$
O_{\mathrm{floor}}
=\left\lfloor
\frac{
I+P_{\mathrm{left}}+P_{\mathrm{right}}-K_{\mathrm{eff}}
}{S}
\right\rfloor+1.
$$

某些池化实现提供 ceil mode，常见抽象形式是

$$
O_{\mathrm{ceil}}
=\left\lceil
\frac{
I+P_{\mathrm{left}}+P_{\mathrm{right}}-K_{\mathrm{eff}}
}{S}
\right\rceil+1.
$$

但 ceil mode 不只是把一个字符 floor 换成 ceil。若最后一个窗口的起点已经落入右侧 padding，框架还要决定：

- 窗口是否允许使用这个起点；
- 需要补多少右侧边界值；
- average pooling 的分母是否包含这些边界值；
- max pooling 的边界值是否可能赢得最大值；
- 输出 shape 与下一层的对齐是否仍符合预期。

以 $I=5,K=2,S=2$、无 padding 为例，floor 输出长度为

$$
\left\lfloor\frac{5-2}{2}\right\rfloor+1=2,
$$

窗口起点是 $0,2$，位置 $4$ 剩下一个孤立元素。ceil 目标会倾向于保留第三个输出，但这个输出究竟看到什么，要由尾部窗口和 padding 约定决定。shape ledger 应同时写出最后一个窗口的坐标，不能只写输出长度 $3$。

## Max pooling 的反向传播把梯度送到赢家

设某个窗口内最大值唯一，argmax 位置为

$$
a^\star_r
=\mathop{\arg\max}_{0\le a<K}
x^P[rS+aD].
$$

最大池化输出对窗口输入的导数为

$$
\frac{\partial y_{\mathrm{max}}[r]}
{\partial x^P[rS+aD]}
=
\begin{cases}
1,&a=a^\star_r,\\
0,&a\ne a^\star_r.
\end{cases}
$$

如果上游梯度为 $g[r]=\partial L/\partial y[r]$，则唯一最大值的输入梯度为

$$
\frac{\partial L}
{\partial x^P[rS+aD]}
=g[r]\mathbf 1\{a=a^\star_r\}.
$$

多个输出窗口可能重叠，所以同一个输入位置可能是多个窗口的赢家，最终梯度要把这些贡献相加。stride 大于一减少窗口重叠，但不保证梯度稀疏模式简单；padding 区域的梯度通常不回写到真实输入。

### 相同最大值是次梯度和实现合同

若窗口是

$$
[2,2,1],
$$

最大值在两个位置同时出现，普通导数不存在唯一值。可行的次梯度可以把上游梯度分给两个位置，实际库也可能固定选择第一次出现的位置，或由底层 reduction 的 tie-breaking 决定。前向输出相同，不代表反向梯度相同。

因此，梯度检查 max pooling 时不能只用全不同的随机浮点数，还应专门测试并记录 tie 行为。训练复现、导出到另一后端和自定义 kernel 都可能在这个边界分叉。

## Average 和 sum pooling 的反向传播是密集分配

没有 padding、窗口大小固定时，average pooling 的每个窗口输入都收到相同份额：

$$
\frac{\partial y_{\mathrm{avg}}[r]}
{\partial x[rS+aD]}
=\frac1K.
$$

因此

$$
\frac{\partial L}
{\partial x[rS+aD]}
=\frac{g[r]}{K}.
$$

如果多个窗口重叠，输入位置的总梯度仍是所有相关窗口贡献之和：

$$
\frac{\partial L}{\partial x[i]}
=\sum_{r:\ i\in\mathcal N_r}
\frac{g[r]}{K}.
$$

sum pooling 只去掉除数：

$$
\frac{\partial L}
{\partial x[rS+aD]}
=g[r].
$$

边界 exclude-pad 的 average pooling 需要把 $K$ 换成每个窗口自己的 $\lvert\mathcal V_r\rvert$。如果实现把 padding 计入分母，边缘梯度会按固定窗口大小缩放；如果排除 padding，边缘真实元素会获得更大的单元素梯度。两者都可以是合理合同，但必须和前向分母一致。

## 全局平均池化把空间变成一个统计量

对每个 channel 做全局平均池化，输入空间为 $H\times W$，输出一个标量：

$$
y_c
=\frac{1}{HW}
\sum_{h=0}^{H-1}\sum_{w=0}^{W-1}X_{c,h,w}.
$$

这会把每个 channel 从一张空间特征图变成一个全局统计量。若后面接线性分类头，参数量可以从

$$
C_{\mathrm{out}}\cdot C\cdot H\cdot W+C_{\mathrm{out}}
$$

降到

$$
C_{\mathrm{out}}\cdot C+C_{\mathrm{out}}.
$$

代价是精确位置被压掉：两个空间分布不同但总平均相同的特征图，经过全局平均池化后会得到同一个向量。它适合“某个语义是否存在”比“语义在哪里”更重要的输出头，不适合直接承担需要像素级定位的任务。

全局最大池化也可以生成每个 channel 的一个标量，但它只保留最强位置，梯度更加稀疏、对异常峰值更加敏感。全局 average 和 global max 的差异不是仅仅换一个 reduce 函数，而是对空间证据提出了不同的统计假设。

## Lp pooling 和固定聚合的统一视角

最大、平均和求和不是唯一的固定聚合。一个常见的 Lp pooling 形式是

$$
y_p[r]
=\left(
\frac1K\sum_{a=0}^{K-1}
\lvert x^P[rS+aD]\rvert^p
\right)^{1/p}.
$$

当 $p$ 增大时，结果逐渐偏向窗口中较大的幅度；$p$ 取特定值可以获得介于平均与最大之间的响应。若 $p$ 也是可学习的，算子就不再是完全固定的 pooling，应额外记录参数约束、梯度和数值稳定性。

从统一视角看，池化选择了一个局部统计量：

| 聚合 | 局部统计量 | 梯度形状 | 常见作用 |
| --- | --- | --- | --- |
| max | 极值 | 主要送到赢家 | 保留最强局部证据 |
| average | 一阶均值 | 窗口内近似均匀 | 平滑并汇总局部水平 |
| sum | 局部总量 | 窗口内相同幅度 | 保留能量或计数尺度 |
| Lp | 幂平均 | 向大幅度位置倾斜 | 在均值和极值之间调节 |
| global average | 全空间均值 | 全空间密集 | 降参数并汇总通道语义 |

## 池化和 stride 卷积并不等价

两者都可以让空间输出变小，但它们写入的先验不同：

| 项目 | 池化 | stride 卷积 |
| --- | --- | --- |
| 参数 | 通常没有可学习参数 | 有 kernel 和 bias |
| 通道关系 | 常见形式逐通道独立 | 可以混合输入通道并改变输出通道 |
| 局部规则 | max、mean、sum 等固定聚合 | 学习局部加权组合和非线性前的响应 |
| 梯度 | max 稀疏，average/sum 按固定比例分配 | 按学习到的权重分配 |
| 下采样语义 | 先验明确，容易解释 | 由参数学习，表达能力更强 |
| 参数/MACs | 空间窗口只做固定比较或加法 | 需要随通道数增长的乘加 |
| 失效方式 | tie、边界、相位、信息丢失 | 过拟合、边界、相位、混叠和参数退化 |

如果输入通道为 $C_{\mathrm{in}}$，输出通道为 $C_{\mathrm{out}}$，卷积核为 $K_h\times K_w$，stride 卷积的参数量大致是

$$
C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w+C_{\mathrm{out}}.
$$

同样空间窗口上的逐通道池化参数量为零。池化因此不能凭“没有参数”被判定为无效；它是在用固定归纳偏置换取更少的自由度和更明确的空间统计。

更重要的差异是 channel mixing。普通池化把每个 channel 的空间证据压缩成同一个 channel 的输出；stride 卷积可以在下采样的同时组合颜色、边缘、纹理等不同通道。若只比较输出 $H\times W$，会掩盖两种算子的表示能力差异。

## 下采样、相位与混叠

### Stride 会丢掉窗口起点

取 $K=2,S=2$ 的 max pooling：

$$
x=[0,10,0,0]
\longrightarrow
y=[10,0].
$$

把峰值向右移动一个输入位置：

$$
x'=[0,0,10,0]
\longrightarrow
y'=[0,10].
$$

同一个强响应只移动了一格，输出位置却从第一个池化单元跳到第二个池化单元。这不是严格的“平移后输出只平移同样的离散距离”，而是 stride 网格与输入相位的相互作用。边界和奇偶尺寸会让这种差异更加明显。

### Average 有低通倾向，max 没有

对交替信号

$$
x=[1,-1,1,-1,1,-1],
$$

长度为 $2$ 的 average pooling 在不重叠窗口上给出

$$
\left[
\frac{1+(-1)}2,
\frac{1+(-1)}2,
\frac{1+(-1)}2
\right]
=[0,0,0].
$$

它确实消除了这个特定的高频模式。相同窗口的 max pooling 则给出

$$
[1,1,1].
$$

所以 max pooling 不是低通滤波器，不能因为它也有 stride 就称为抗混叠下采样。average pooling 也只是固定 box filter，在一般信号、二维图像和不同频率上并不等于理想低通。

若部署任务对高频纹理和小位移敏感，应把下采样拆成“滤波”和“抽取”两个可审计步骤，或使用明确的 anti-aliased pooling。实验至少应比较原图、半像素或一格平移后的输出，并检查高频输入是否被错误地折叠到低频响应。

## 池化改变感受野步距

设进入第一层前的感受野为 $r_0=1$、输入间距为 $j_0=1$。对有效核尺寸 $K_{\mathrm{eff}}$、stride $S$ 的层，常用递推是

$$
r_l
=r_{l-1}+(K_{\mathrm{eff}}-1)j_{l-1},
\qquad
j_l=j_{l-1}S.
$$

先做 $K=2,S=2$ 的池化：

$$
r_1=1+(2-1)\cdot1=2,
\qquad
j_1=2.
$$

再接 $K=3,S=1$ 的卷积：

$$
r_2=2+(3-1)\cdot2=6,
\qquad
j_2=2.
$$

第二层卷积的核仍然只有三格，但每个输入特征点已经代表原图上间距为 $2$ 的位置，所以它在原图上的理论感受野是六格。池化不仅改变 feature map 的宽高，也改变了后续层在原输入坐标上的采样步距。

这解释了为什么“只看最后一层 kernel size”会低估深层空间范围；同样也解释了为什么过早的 stride/pooling 可能让细小目标在进入深层前就被压掉。

## 池化不是可逆的上采样前身

池化通常是多对一映射。以 $2\times2$ average pooling 为例，很多不同的窗口都可能有相同平均值：

$$
\frac{1+3+5+7}{4}
=\frac{2+2+6+6}{4}
=4.
$$

仅凭输出 $4$ 无法恢复原来的四个输入。max pooling 额外保存 argmax 位置时，可以构造一种 unpooling，把最大响应放回记录的位置，但其他位置仍然未知，未记录的数值不能由池化输出恢复。

因此，decoder 中的 nearest/bilinear interpolation、转置卷积、sub-pixel rearrangement 和 pooling indices unpooling 是不同的上采样合同。把“pool 后放大”当作逆运算，会错误地期待被丢失的空间细节自动回来。

## 常见失效模式

| 现象 | 可能原因 | 最小核对 |
| --- | --- | --- |
| 输出长度差一格 | floor/ceil、stride 或四侧 padding 记错 | 写出最后一个窗口起点和有效核尺寸 |
| 全是负值时边缘突然变成零 | max pooling 使用 zero padding | 用负输入手算边界窗口，检查是否应为负无穷 |
| average pooling 边缘比中心小 | include-pad 把零计入分母 | 同时记录分子和分母，比较 exclude-pad |
| 梯度只落在少数像素 | max pooling 的 argmax 结构 | 记录每个窗口的 winner 和重叠累加 |
| 相同输入平移后输出相差很大 | stride 网格相位或边界不对齐 | 做一格平移实验，比较特征位置和数值 |
| 下采样后小目标消失 | 池化过早、窗口太大或理论感受野跳跃过快 | 记录每层 $r_l,j_l$ 与 feature map 尺寸 |
| 换后端后训练轨迹改变 | max tie、平均分母或 ceil mode 语义不同 | 用重复最大值和边界窗口做一致性测试 |
| decoder 无法恢复细节 | 池化是多对一且未保存足够信息 | 区分插值、转置卷积和带 indices 的 unpooling |
| 参数量看似为零却精度下降 | 固定聚合丢掉了位置或通道关系 | 与 stride 卷积比较通道混合、相位和任务需求 |

## 池化层的验收顺序

读到一个池化层时，可以按下面顺序建立最小可复现合同：

1. 写清输入布局、batch、channel 和两个空间尺寸；
2. 分别记录 kernel、dilation、stride、四侧 padding 与 floor/ceil 模式；
3. 用有效核尺寸计算每个空间轴的输出长度；
4. 列出一个中心窗口和一个边界窗口，确认 padding 值；
5. 明确是 max、average、sum、Lp 还是 global pooling；
6. 对 average 记录 padding 是否进入分母；
7. 对 max 记录 argmax、tie-breaking 和反向梯度累加；
8. 用一格平移和高频交替输入检查相位、混叠和局部不变性；
9. 把输出 shape、感受野 $r_l$ 和步距 $j_l$ 接入下一层 shape ledger；
10. 若有残差、拼接或 decoder，检查空间中心和 channel 合同。

池化层的“固定”只意味着规则不由训练学习，不意味着它没有设计选择。窗口大小、stride、padding、聚合统计量、分母和边界值共同构成了这个算子的行为；只有这些项目都被记录，池化才是可复现的架构组件。

## 相关词条

[输出尺寸算术](../cnn/output-size-arithmetic/)

[步幅、填充与膨胀](../cnn/stride-padding-dilation/)

[卷积神经网络](../cnn/cnn/)

[二维卷积](../cnn/convolution-2d/)

[为什么需要卷积](../cnn/why-convolution/)

[不变性与等变性](../cnn/invariance-and-equivariance/)

[卷积作为 Toeplitz 结构](../cnn/convolution-as-toeplitz/)

[残差连接](../cnn/residual-connections/)
