---
title: "反向传播：沿计算图把梯度累加回来"
tags: ["why-models-learn"]
---

反向传播是对计算图执行反向模式自动微分的算法：先按依赖顺序计算每个节点的值，再从标量损失出发，沿相反方向把局部导数相乘，并把汇合到同一个变量的梯度相加。它负责计算“参数改变会怎样影响损失”，不负责选择学习率或更新参数。本文从局部运算规则、分支累加和一个可手算的神经元开始，推导向量化矩阵形式，再说明反向模式的效率、梯度检查和最容易把公式写错的边界。

![计算图先从输入向右前向计算损失，再从损失向左反向累加梯度](/assets/backpropagation/svg/backpropagation.1.svg)

## 反向传播不是优化器

把一次训练迭代拆成三个不同动作：

1. **前向计算**：给定当前参数和输入，得到中间值、预测和损失；
2. **反向传播**：根据同一批前向值计算每个参数的损失梯度；
3. **参数更新**：优化器用梯度、学习率和状态改变参数。

如果参数记为 $\boldsymbol\theta$，损失为

$$
L=L(\boldsymbol\theta;\boldsymbol x,\boldsymbol y),
$$

反向传播产生的是

$$
\nabla_{\boldsymbol\theta}L.
$$

最简单的梯度下降更新另写成

$$
\boldsymbol\theta_{\mathrm{new}}
=\boldsymbol\theta-\eta\nabla_{\boldsymbol\theta}L.
$$

反向传播不决定 $\eta$，也不决定是否使用 momentum、Adam、权重衰减或梯度裁剪。把“反向传播”和“梯度下降”混成同一个步骤，会在诊断训练问题时失去边界：梯度数值错是计算图问题，步长太大是优化协议问题。

## 计算图把复合函数拆成局部运算

一个神经网络的前向表达式可以很长，但它通常由加法、乘法、线性变换、激活和损失等基本运算组成。把每个中间结果作为节点：

$$
x
\longrightarrow
z=wx+b
\longrightarrow
a=\tanh(z)
\longrightarrow
L=\frac12(a-y)^2.
$$

这是一张有向无环图。箭头表示前向依赖；它不表示梯度方向。反向传播做的事是先保存前向节点值，再沿反向拓扑序处理每条边。

为了简化记号，假设最终损失是标量，并定义节点 $v$ 的伴随量

$$
\bar v
:=\frac{\partial L}{\partial v}.
$$

于是 $\bar L=1$。伴随量不是节点本身的值：$v$ 是前向数值，$\bar v$ 是损失对它的敏感度。一个节点可以在前向时只计算一次，却在反向时把来自多个后继的敏感度加起来。

## 三条局部规则组成反向传播

下面所有规则都使用累加符号 $\mathrel{+}=$：如果一个变量通过多条路径影响损失，每条路径的贡献都必须加进去。

| 前向局部运算 | 反向贡献 |
| --- | --- |
| $v=u+r$ | $\bar u\mathrel{+}=\bar v,\quad \bar r\mathrel{+}=\bar v$ |
| $v=ur$ | $\bar u\mathrel{+}=r\bar v,\quad \bar r\mathrel{+}=u\bar v$ |
| $v=\phi(u)$ | $\bar u\mathrel{+}=\phi'(u)\bar v$ |
| $\boldsymbol v=A\boldsymbol u$ | $\bar{\boldsymbol u}\mathrel{+}=A^{\mathsf T}\bar{\boldsymbol v},\quad \bar A\mathrel{+}=\bar{\boldsymbol v}\boldsymbol u^{\mathsf T}$ |

这些规则都只是链式法则。对向量函数 $\boldsymbol v=f(\boldsymbol u)$，若 Jacobian 的行对应输出、列对应输入：

$$
J_{v,u}
=\frac{\partial\boldsymbol v}{\partial\boldsymbol u},
$$

则

$$
\bar{\boldsymbol u}
=J_{v,u}^{\mathsf T}\bar{\boldsymbol v}.
$$

反向传播使用的是向量–Jacobian 积或其等价的转置 Jacobian 乘向量，而不是显式构造整个 Jacobian。对大网络来说，避免存储一个巨大的稠密矩阵是它高效的关键。

### 为什么分支处必须相加

令一个变量被两个节点共同使用：

$$
s=x+x,
\qquad
L=s^2.
$$

若 $x=3$，前向值是 $s=6$、$L=36$。从损失开始：

$$
\bar L=1,
\qquad
\bar s=2s=12.
$$

节点 $s=x+x$ 有两条从 $x$ 出发的边。每条边贡献 $\bar s=12$，所以

$$
\bar x=12+12=24.
$$

直接求导 $L=(2x)^2=4x^2$ 也得到 $L'(3)=24$。如果实现只把第二条路径写入 $\bar x$，结果会少一半；如果把两条路径相乘，又会得到完全不同的量。计算图的 fan-out 复制敏感度，fan-in 汇合敏感度。

## 一个神经元的完整手算

考虑一个带 tanh 激活的标量神经元：

$$
z=wx+b,
\qquad
a=\tanh(z),
\qquad
L=\frac12(a-y)^2.
$$

取

$$
x=2,
\qquad
w=0.5,
\qquad
b=-0.5,
\qquad
y=0.25.
$$

### 前向阶段

逐个节点计算：

$$
z=0.5\times2-0.5=0.5,
$$

$$
a=\tanh(0.5)\approx0.462117157260,
$$

$$
L=\frac12(0.462117157260-0.25)^2
\approx0.022496844202.
$$

### 反向阶段

从损失往回：

$$
\bar L=1.
$$

平方损失节点给出

$$
\bar a
=\frac{\partial L}{\partial a}
=a-y
\approx0.212117157260.
$$

因为

$$
\frac{\mathrm d}{\mathrm dz}\tanh(z)
=1-\tanh^2(z)
=1-a^2,
$$

所以

$$
\bar z
=\bar a(1-a^2)
\approx0.166819057450.
$$

最后穿过仿射节点：

$$
\bar w=\bar z\,x
\approx0.333638114901,
\qquad
\bar b=\bar z
\approx0.166819057450,
\qquad
\bar x=\bar z\,w
\approx0.083409528725.
$$

每一个数字都可以追溯到一条局部边：

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
=(a-y)(1-a^2)x.
$$

反向传播只是把这条长乘积拆成三个小乘积，并把中间结果复用给 $w$、$b$ 和 $x$ 的梯度。

## 矩阵形式：一层反向传播的形状

设一个 batch 有 $n$ 个样本，输入维度为 $d$，隐藏单元数为 $h$。采用每个样本一行的布局：

$$
X\in\mathbb R^{n\times d},
\qquad
W\in\mathbb R^{h\times d},
\qquad
b\in\mathbb R^h.
$$

仿射层和逐分量激活为

$$
Z=XW^{\mathsf T}+\boldsymbol1 b^{\mathsf T},
\qquad
A=\phi(Z).
$$

若上游已经给出

$$
G_A=\frac{\partial L}{\partial A},
$$

则逐元素激活先用局部导数筛选：

$$
G_Z=G_A\odot\phi'(Z).
$$

矩阵乘法的反向结果是

$$
\begin{aligned}
G_X&=G_ZW,\\
G_W&=G_Z^{\mathsf T}X,\\
g_b&=G_Z^{\mathsf T}\boldsymbol1.
\end{aligned}
$$

形状检查如下：

| 量 | 形状 | 来源 |
| --- | --- | --- |
| $Z,A,G_A,G_Z$ | $n\times h$ | 每个样本、每个隐藏单元 |
| $G_X$ | $n\times d$ | $G_ZW$ |
| $G_W$ | $h\times d$ | $G_Z^{\mathsf T}X$ |
| $g_b$ | $h$ | 沿 batch 轴求和 |

如果损失是 batch 平均，$1/n$ 应该出现在 $G_A$ 或对应的 reduction 中；如果损失是 batch 求和，偏置梯度会随 $n$ 增长。两种约定都可以，但前向 loss、梯度和学习率必须使用同一约定。

### 一个小 batch 的数值核对

取

$$
X=
\begin{pmatrix}
1&2\\
-1&0.5
\end{pmatrix},
\quad
W=
\begin{pmatrix}
0.4&-0.2\\
0.1&0.3
\end{pmatrix},
\quad
b=
\begin{pmatrix}
0.2\\-0.1
\end{pmatrix}.
$$

令损失为每个样本平方误差的平均：

$$
L=\frac1{2n}\lVert Z-Y\rVert_F^2,
\qquad
Y=
\begin{pmatrix}
0.1&0.5\\
-0.4&0.2
\end{pmatrix}.
$$

前向结果和对 $Z$ 的梯度是

$$
Z=
\begin{pmatrix}
0.2&0.6\\
-0.3&-0.05
\end{pmatrix},
\qquad
G_Z=\frac{Z-Y}{n}
=
\begin{pmatrix}
0.05&0.05\\
0.05&-0.125
\end{pmatrix}.
$$

代入矩阵反向规则：

$$
G_W=
\begin{pmatrix}
0&0.125\\
0.175&0.0375
\end{pmatrix},
\qquad
g_b=
\begin{pmatrix}
0.1\\-0.075
\end{pmatrix},
$$

$$
G_X=
\begin{pmatrix}
0.025&0.005\\
0.0075&-0.0475
\end{pmatrix},
\qquad
L=0.023125.
$$

这里的 $0$ 是浮点计算中约为 $-1.4\times10^{-17}$ 的项，按展示精度写成零。最重要的不是这组小数字本身，而是每个转置和 batch 归约都能用形状解释。

## 激活、归一化和共享节点的边界

### 逐分量激活与 softmax 不同

对逐分量激活：

$$
G_Z=G_A\odot\phi'(Z).
$$

ReLU 的 $\phi'(Z)$ 是活动掩码；sigmoid、tanh、GELU 则提供逐坐标的局部导数。[逐元素导数](../calculus/elementwise-derivatives/)解释了为什么这些 Jacobian 是对角结构。

Softmax 的输出坐标共享分母，Jacobian 是

$$
J_{\mathrm{softmax}}
=\operatorname{diag}(\boldsymbol p)
-\boldsymbol p\boldsymbol p^{\mathsf T}.
$$

因此不能把它当作独立坐标逐个乘导数；与交叉熵配对后才简化为 $p-y$。[Softmax](../neurons-and-activations/softmax/)和[广播与归约求导](../calculus/broadcast-and-reduction-derivatives/)分别展开这个跨坐标边界。

### 一个变量被复用时，梯度只初始化一次、贡献多次

残差连接、共享 embedding、权重绑定和计算图中的显式复用都会让同一变量有多个后继。反向时应：

1. 给每个需要求梯度的节点准备一个累加槽；
2. 每处理一条后继边，就把局部 VJP 加进槽里；
3. 等所有后继都处理完，再沿前驱边继续传播。

不能因为某条路径先到，就覆盖掉之前的贡献。实现中常见的加法节点和 scatter-add，都是这个规则的不同外观。

### 不可导点需要约定

ReLU 在 $0$ 处没有经典导数，框架会选择一个次梯度约定；max、绝对值和分段损失也有类似边界。连续分布下恰好命中边界可能很少，但手工构造、量化值和 mask 仍可能命中。梯度检查时应避开边界或明确比较实现采用的约定。

## 反向模式为什么适合神经网络

设函数有 $d_{\mathrm{in}}$ 个输入方向和一个标量损失输出。前向模式自动微分传播一个方向导数：

$$
\delta\boldsymbol v
=J_{v,u}\delta\boldsymbol u.
$$

反向模式传播一个输出敏感度：

$$
\bar{\boldsymbol u}
=J_{v,u}^{\mathsf T}\bar{\boldsymbol v}.
$$

神经网络训练通常是“参数很多，损失一个标量”。反向模式可以在一次反向扫描中同时得到所有参数的梯度；如果用前向模式逐个覆盖参数方向，代价可能随参数数量增长。反过来，如果输入方向很少而输出很多，前向模式可能更合适。

反向模式的代价不是零：

* 前向时要保存激活、mask、归一化统计等反向需要的值；
* 保存这些中间值会占用显存；
* 训练时的计算通常接近一次前向加一次反向，而不是只做一次前向；
* 用 checkpoint 重算可以省内存，但会增加前向计算。

所以“反向传播高效”是相对于输出为标量、参数维度很大的训练目标而言，不是对所有 Jacobian 都便宜。

## 梯度检查：用有限差分验证反向结果

反向传播是链式乘法的长组合，转置、广播、归约或 mask 任一处写错，都可能让最终训练看起来只是“不收敛”。对小模型可以用中心有限差分核对某个参数 $\theta_k$：

$$
g_{\mathrm{fd},k}
=\frac{L(\theta_k+h)-L(\theta_k-h)}{2h}.
$$

把解析梯度 $g_{\mathrm{bp},k}$ 与有限差分比较：

$$
\operatorname{relerr}_k
=\frac{\lvert g_{\mathrm{bp},k}-g_{\mathrm{fd},k}\rvert}
{\max\left(1,\lvert g_{\mathrm{bp},k}\rvert,
\lvert g_{\mathrm{fd},k}\rvert\right)}.
$$

对前面标量神经元，以 $h=10^{-5}$ 分别扰动 $x,w,b$：

| 变量 | 反向传播 | 中心差分 | 误差量级 |
| --- | ---: | ---: | ---: |
| $x$ | $0.083409528725$ | $0.083409528721$ | $10^{-12}$ |
| $w$ | $0.333638114901$ | $0.333638114655$ | $10^{-10}$ |
| $b$ | $0.166819057450$ | $0.166819057421$ | $10^{-10}$ |

有限差分不是第二个自动微分实现，它也有截断误差和舍入误差。步长过大时不是局部导数，过小时两个接近的 loss 相减会丢精度。梯度检查应使用小网络、确定性前向、较高精度和多个步长；不要把随机 dropout、训练态 batch normalization 或 ReLU 零点直接塞进最小例子。

## 一次训练迭代的可审计顺序

对于一个 batch，可以把反向传播记录成下面的时间线：

| 阶段 | 使用的对象 | 输出 |
| --- | --- | --- |
| 前向 | 当前参数、输入、固定随机状态 | 中间激活、预测、loss |
| 初始化 | 标量 loss | $\bar L=1$ |
| 反向 | 缓存的前向值和局部导数 | 各节点伴随量、参数梯度 |
| 梯度审计 | 梯度范数、有限性、抽样坐标 | 是否需要停止或定位 |
| 优化更新 | 梯度、学习率、优化器状态 | 新参数 |

更新应该在整张图的梯度都计算完之后进行。若在计算另一层梯度时提前改变参数，就会把不同参数版本混入同一次链式法则，既不是对旧参数的梯度，也不是对新参数的完整梯度。

实现级别还要检查：

* 是否误用 detach、stop-gradient 或 no-grad 截断了路径；
* 是否在需要反向的中间张量上做了破坏版本的原地修改；
* loss 是 sum 还是 mean，batch 与 mask 的归约轴是否一致；
* 参数梯度是否在每个 batch 开始前清零，或明确采用累积梯度；
* 训练和推理的 dropout、batch normalization、有状态层和混合精度缩放是否符合协议。

## 失效模式

**把反向传播当成参数更新。** 反向传播只产生梯度；学习率、动量、权重衰减和更新时机属于优化器。

**覆盖而不是累加梯度。** 一个变量被多条路径使用时，所有后继贡献都必须相加；残差、共享权重和分支网络尤其容易漏掉一条路径。

**忘记矩阵转置。** 对 $\boldsymbol v=A\boldsymbol u$，输入梯度是 $A^{\mathsf T}\bar{\boldsymbol v}$；先写清每个矩阵的形状，再决定转置。

**混用 sum 与 mean reduction。** loss 数值看似正常，但梯度随 batch 大小改变；应把归约因子明确放进公式和实现。

**用更新后的参数继续反向。** 一次反向传播必须对应一次固定前向图；先更新一部分参数会破坏这条对应关系。

**把 softmax 当逐分量函数。** 共享分母带来非对角 Jacobian；应使用完整 Jacobian 或 logits 交叉熵的稳定梯度。

**在不可导边界上要求唯一答案。** ReLU、max 和绝对值需要次梯度约定；梯度检查应避开边界或按框架约定核对。

**把有限差分当成万能真值。** 差分步长、浮点精度、随机算子和有状态层都可能让检查失真；先做确定性最小图。

**误用 detach 或原地操作。** 前向结果仍可能正确，但计算图已经失去一条反向边；要同时检查值与梯度。

**只看最终 loss。** loss 下降不能证明每层梯度都正确；还要记录非有限值、范数、方向抽样和小图有限差分。

## 相关词条

- [链式法则](../calculus/chain-rule/)：从局部线性近似推导反向传播的乘法骨架。
- [逐元素导数](../calculus/elementwise-derivatives/)：解释逐分量激活的对角 Jacobian。
- [广播与归约求导](../calculus/broadcast-and-reduction-derivatives/)：处理 batch、共享参数和归约轴的梯度累加。
- [自动微分](../calculus/automatic-differentiation/)：比较前向模式 JVP、反向模式 VJP 与计算图实现。
- [什么是神经元](../neurons-and-activations/what-is-a-neuron/)：从仿射变换和激活函数定义单个神经元。
- [激活函数](../neurons-and-activations/activation-functions/)：比较反向路径中出现的局部导数。
- [Softmax](../neurons-and-activations/softmax/)：作为跨坐标耦合 Jacobian 的反向传播边界例子。
- [梯度检查](../backpropagation/gradient-checking/)：用有限差分系统验证复杂计算图。
- [单神经元反向传播](../backpropagation/backprop-single-neuron/)：把本篇的局部规则扩展为完整手算实例。
- [向量化反向传播](../backpropagation/vectorized-backprop/)：进一步整理 batch 矩阵形式和高效实现。
