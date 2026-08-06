---
title: "ReLU：正侧不饱和，负侧有死区"
tags: ["why-models-learn"]
---

ReLU（rectified linear unit，修正线性单元）是逐分量函数 $\operatorname{ReLU}(z)=\max(0,z)$：负的预激活被截成 0，正的预激活原样通过。它没有 sigmoid 和 tanh 那样的正侧饱和，所以深层网络在激活为正时更容易传递梯度；代价是负侧的导数为 0，某个神经元可能长期输出 0，成为死亡 ReLU。本文从分段定义和局部导数出发，推导 ReLU 的梯度、稀疏性和 He 初始化条件，再说明偏置、学习率以及 Leaky ReLU 等变体怎样改变这个权衡。

![ReLU 在负侧为零、正侧为直线，导数是从零跳到一的门](/assets/neurons-and-activations/svg/relu.1.svg)

## ReLU 是一条带硬门的折线

ReLU 的定义是

$$
\operatorname{ReLU}(z)=\max(0,z)
=
\begin{cases}
0,&z<0,\\
z,&z\geq0.
\end{cases}
$$

它把输入轴分成两个区域。$z<0$ 时，输出不再携带输入的数值；$z>0$ 时，输出就是输入本身。转折点在 $(0,0)$，左侧水平，右侧斜率为 1。

用三个值代入就能看到这条分段规则：

$$
\operatorname{ReLU}(-2)=0,
\qquad
\operatorname{ReLU}(0)=0,
\qquad
\operatorname{ReLU}(3)=3.
$$

这里的“修正”不是把负数换成它的绝对值。绝对值会把 $-2$ 变成 $2$，ReLU 则把它变成 0；它保留正方向，丢弃负方向。

ReLU 还有一个对分析有用的齐次性质。对任意 $c\geq0$，

$$
\operatorname{ReLU}(cz)=c\operatorname{ReLU}(z).
$$

证明只需要分两种情况。如果 $z\geq0$，两边都是 $cz$；如果 $z<0$，因为 $c\geq0$，$cz$ 仍然非正，两边都是 0。这个性质说明正侧的整体尺度可以被提出，但平移不行：

$$
\operatorname{ReLU}(z+c)\neq \operatorname{ReLU}(z)+c
$$

一般不成立。偏置会移动折点，而不是简单地给输出加常数。

## 导数在零点之外是一扇二值门

当 $z<0$ 时 ReLU 是常数 0，所以导数为 0；当 $z>0$ 时 ReLU 是直线 $z$，所以导数为 1：

$$
\operatorname{ReLU}'(z)=
\begin{cases}
0,&z<0,\\
1,&z>0.
\end{cases}
$$

在 $z=0$ 处，左导数是 0，右导数是 1，二者不相等。因此普通意义下的导数不存在。深度学习实现会选一个次梯度约定，例如

$$
g(0)\in[0,1],
$$

实际内核常固定为 0，也有实现固定为 1。对连续分布的随机预激活来说，恰好落在一个精确点 $z=0$ 的概率通常为 0，所以这个约定不是训练中最常见的差异；手工构造的边界输入和量化计算则仍然要记录它。

把导数写成门变量更直观。定义

$$
m(z)=\mathbf 1\{z>0\},
$$

在不碰到零点时，$\operatorname{ReLU}'(z)=m(z)$。反向传播不是让所有信号通过，而是先看前向时神经元是否处于正侧，再用这个二值门决定梯度是否通过。

## 一个神经元的前向与反向

设一个标量神经元先做仿射变换

$$
z=w^{\mathsf T}x+b,
\qquad
a=\operatorname{ReLU}(z).
$$

假设损失对激活的梯度是 $\partial L/\partial a$。链式法则给出

$$
\frac{\partial L}{\partial z}
=\frac{\partial L}{\partial a}\operatorname{ReLU}'(z),
$$

以及

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial z}x,
\qquad
\frac{\partial L}{\partial b}
=\frac{\partial L}{\partial z}.
$$

取

$$
x=\begin{bmatrix}2\\-1\end{bmatrix},
\qquad
w=\begin{bmatrix}0.5\\-1\end{bmatrix},
\qquad
b=-0.25,
$$

则

$$
z=0.5\cdot2+(-1)\cdot(-1)-0.25=1.75,
\qquad
a=1.75.
$$

若目标为 $y=1$，使用 $L=\tfrac12(a-y)^2$，则

$$
\frac{\partial L}{\partial a}=a-y=0.75.
$$

因为 $z>0$，ReLU 的导数为 1，所以

$$
\frac{\partial L}{\partial z}=0.75,
\qquad
\frac{\partial L}{\partial w}
=0.75\begin{bmatrix}2\\-1\end{bmatrix}
=\begin{bmatrix}1.5\\-0.75\end{bmatrix},
\qquad
\frac{\partial L}{\partial b}=0.75.
$$

如果只把偏置改成 $b=-2.5$，那么 $z=-0.5$、$a=0$。此时 $\partial L/\partial a=-1$，但

$$
\frac{\partial L}{\partial z}=(-1)\cdot0=0,
\qquad
\frac{\partial L}{\partial w}=0,
\qquad
\frac{\partial L}{\partial b}=0.
$$

损失仍然非零，梯度却无法从这个样本穿过 ReLU 回到 $w$ 和 $b$。这正是“负侧有死区”的局部含义。

## 对向量逐分量应用时，梯度是一张活动掩码

隐藏层通常同时处理一组神经元：

$$
z=Wx+b,
\qquad
h=\operatorname{ReLU}(z),
$$

其中 ReLU 对 $z$ 的每个分量分别作用。设

$$
D=\operatorname{diag}\bigl(\mathbf1\{z_1>0\},\ldots,\mathbf1\{z_m>0\}\bigr).
$$

在没有分量恰好为零时，激活层的 Jacobian 就是 $D$。若上游梯度为 $g_h$，则

$$
g_z=Dg_h.
$$

例如

$$
z=\begin{bmatrix}-1\\0.5\\2\\-0.2\end{bmatrix},
\qquad
g_h=\begin{bmatrix}3\\-2\\0.4\\5\end{bmatrix},
$$

活动掩码为

$$
D=\operatorname{diag}(0,1,1,0),
$$

因此

$$
g_z=Dg_h
=\begin{bmatrix}0\\-2\\0.4\\0\end{bmatrix}.
$$

第二、第三个神经元把梯度传回去，第一、第四个神经元在这个样本上完全屏蔽梯度。对前一层权重，若 $z=Wx+b$，则

$$
\frac{\partial L}{\partial W}=g_zx^{\mathsf T},
\qquad
\frac{\partial L}{\partial b}=g_z.
$$

所以 ReLU 的反向传播可以看成先乘一张 0/1 掩码，再按普通线性层的规则传播。

## 正侧不饱和，负侧却会丢信息

sigmoid 和 tanh 的导数在两端都趋近于 0。ReLU 的正侧不同：只要 $z>0$，

$$
\operatorname{ReLU}'(z)=1.
$$

无论 $z=1$ 还是 $z=100$，局部导数都是 1。对于一条始终停留在正侧的标量路径，ReLU 本身不会像饱和 sigmoid 那样逐层乘上很小的因子。

负侧的代价同样明确：$z<0$ 时输出被压成 0，且局部导数也为 0。一个负的预激活同时丢失了数值和一阶梯度。不能把“导数不饱和”理解成“信息永远不丢”；它只说明正侧的局部斜率不会随幅值变小。

把几个激活放在同一张表里：

| 激活 | 负侧输出 | 正侧输出 | 远端导数 | 主要边界 |
| --- | --- | --- | --- | --- |
| sigmoid | 接近 0 | 接近 1 | 两端趋近 0 | 两侧都饱和，输出不零中心 |
| tanh | 接近 -1 | 接近 1 | 两端趋近 0 | 两侧都饱和，输出零中心 |
| ReLU | 0 | 等于 $z$ | 正侧为 1 | 负侧导数为 0 |
| Leaky ReLU | 等于 $\alpha z$ | 等于 $z$ | 负侧为 $\alpha$ | 仍是折线，负侧保留小梯度 |
| softplus | 接近 0 | 接近 $z$ | 等于 sigmoid | 平滑但正侧渐近于 1 |

因此 ReLU 的优点和缺点来自同一条定义：正侧的斜率固定为 1，负侧的斜率固定为 0。

## 稀疏激活不是同义于死亡神经元

对称零均值的预激活分布常使约一半分量落在负侧。若 $Z$ 连续且关于 0 对称，

$$
\Pr[Z>0]=\Pr[Z<0]=\frac12.
$$

于是单个样本的 ReLU 输出常包含不少精确的 0。这种稀疏性可以降低后续矩阵乘法中实际参与的分量，也可能让表示更容易区分“某个特征是否出现”。

但“某个样本上为 0”和“对所有样本都为 0”是两件事。定义神经元在数据集 $\mathcal D$ 上的活动率为

$$
\rho=\frac1{\lvert\mathcal D\rvert}
\sum_{x\in\mathcal D}\mathbf1\{w^{\mathsf T}x+b>0\}.
$$

若 $\rho\approx0.5$，可能只是正常的稀疏激活；若 $\rho$ 在不同批次都接近 0，且权重梯度也接近 0，才需要怀疑神经元已经死亡。诊断时要同时记录输出非零比例和参数梯度，不要只看某一批次的零值数量。

偏置会直接移动活动率。若 $Z_0\sim\mathcal N(0,1)$，令 $Z=Z_0+b$，则

$$
\Pr[Z>0]=\Pr[Z_0>-b]=\Phi(b),
$$

其中 $\Phi$ 是标准正态分布的累积分布函数。数值上，$b=-1$ 时活动率约为 $0.158655$，$b=0$ 时为 $0.5$，$b=1$ 时约为 $0.841345$。同样的权重方差，只要偏置移动一单位，活动率就会显著改变。

## He 初始化保持的是二阶矩

ReLU 的初始化推导要区分均值、方差和二阶矩。设输入分量独立、均值为 0、方差为 $v$，一层有 $n$ 个输入，权重独立且

$$
\mathbb E[w_i]=0,
\qquad
\operatorname{Var}(w_i)=q.
$$

忽略偏置时，预激活

$$
Z=\sum_{i=1}^{n}w_iX_i
$$

的均值为 0，方差为

$$
\operatorname{Var}(Z)=nqv.
$$

若 $Z$ 近似零均值对称分布，则 ReLU 的正侧只保留一半质量。对零均值高斯变量，设 $Z\sim\mathcal N(0,\sigma^2)$，直接积分可以得到

$$
\mathbb E[\operatorname{ReLU}(Z)]
=\frac{\sigma}{\sqrt{2\pi}},
$$

以及

$$
\mathbb E[\operatorname{ReLU}(Z)^2]
=\frac{\sigma^2}{2}.
$$

因为二阶矩是输出平方的期望，代入 $\sigma^2=nqv$：

$$
\mathbb E[A^2]=\frac{nqv}{2},
\qquad A=\operatorname{ReLU}(Z).
$$

若希望每层的输出二阶矩大致保持为 $v$，就令

$$
\frac{nqv}{2}\approx v
\quad\Longrightarrow\quad
q\approx\frac2n.
$$

这就是 ReLU 语境下 He 初始化的核心方差条件：权重方差取输入宽度倒数的两倍。它保持的是 $\mathbb E[A^2]$，不是说 ReLU 输出均值为 0，也不是说输出方差精确等于输入方差。

事实上，

$$
\operatorname{Var}(A)
=\mathbb E[A^2]-\mathbb E[A]^2
=\sigma^2\left(\frac12-\frac1{2\pi}\right).
$$

在 $n=4$、$v=1$ 的示例中，比较两种权重方差：

| 权重方差 $q$ | $\operatorname{Var}(Z)=nq$ | $\mathbb E[A^2]$ | $\mathbb E[A]$ | $\operatorname{Var}(A)$ |
| ---: | ---: | ---: | ---: | ---: |
| $1/4$ | $1$ | $0.5$ | $0.398942$ | $0.340845$ |
| $1/2$ | $2$ | $1$ | $0.564190$ | $0.681690$ |

第二行是 $q=2/n$ 的 He 条件，输出二阶矩回到 1，但均值变成正数。后续层接收到的输入若不再中心化，活动率和方差分析就不能只靠这一条独立高斯近似判断。

## 固定活动模式时，网络是一段仿射函数

考虑一层

$$
h(x)=\operatorname{ReLU}(Wx+b).
$$

只要输入沿着一小段路径移动而不让任何预激活穿过 0，活动掩码 $D$ 就不变。此时

$$
h(x)=D(Wx+b),
$$

它对 $x$ 是仿射函数，局部 Jacobian 为

$$
J_h(x)=DW.
$$

当某个预激活穿过 0，$D$ 才会改变，局部斜率随之切换。这让 ReLU 网络成为由许多仿射片段拼接而成的函数；深度和宽度增加的是活动模式及其组合，而不是让每个片段内部变成高阶曲线。严格的片段数量和深度分离留给 [分段线性 ReLU 理论](../approximation-theory/piecewise-linear-relu/)，这里先用一个一维例子看清机制。

考虑下面这个一维组合：

$$
f(x)=\operatorname{ReLU}(x)-\operatorname{ReLU}(x-1).
$$

按两个折点 $0$ 和 $1$ 分段：

| 输入区域 | $\operatorname{ReLU}(x)$ | $\operatorname{ReLU}(x-1)$ | $f(x)$ |
| --- | ---: | ---: | ---: |
| $x\leq0$ | $0$ | $0$ | $0$ |
| $0<x<1$ | $x$ | $0$ | $x$ |
| $x\geq1$ | $x$ | $x-1$ | $1$ |

两条 ReLU 的线性斜率在折点处切换，组合后得到一个先上升、再保持不变的函数。非线性来自活动模式的切换，不来自每个正侧片段内部的曲率。

## 死 ReLU 是怎样形成的

先看单个样本。若当前 $z<0$，则 $a=0$ 且 $\partial a/\partial z=0$。对任何上游梯度 $g_a$，

$$
g_z=g_a\frac{\partial a}{\partial z}=0.
$$

于是这个样本对该神经元的权重和偏置都不给更新。如果一个神经元对训练集中的所有样本都满足 $w^{\mathsf T}x+b<0$，那么它在每个样本上的梯度都为 0；在普通 ReLU 下，它会停在当前参数上。

一个小例子：取 $x\in\{0,1,2\}$、$w=0.2$、$b=-0.5$，三个预激活是

$$
(-0.5,-0.3,-0.1).
$$

三者都在负侧，因此输出向量是 $(0,0,0)$，无论上游损失梯度是什么，来自这三条样本的 $\partial L/\partial w$ 和 $\partial L/\partial b$ 都是 0。若只把偏置改成 $b=0.1$，预激活变为 $(0.1,0.3,0.5)$，三个样本都重新进入正侧。

“学习率过大”会让神经元一步跨过折点并落到全负区域；负偏置、输入尺度变化和批次组成也会造成同样的结果。反过来，若数据中某类样本长期把预激活推到负侧，神经元可能只在少数批次活动，训练曲线看似仍在下降，但该单元已经不再贡献有效梯度。下一篇 [死亡 ReLU 与 Leaky ReLU](../neurons-and-activations/dead-relu-and-leaky/) 会把活动率监控、恢复策略和负侧变体单独展开。

## Leaky ReLU 保留一条负侧斜率

Leaky ReLU 用一个小正数 $\alpha$ 替换负侧的零斜率：

$$
\operatorname{LReLU}_\alpha(z)=
\begin{cases}
\alpha z,&z<0,\\
z,&z\geq0.
\end{cases}
$$

因此在 $z<0$ 时

$$
\frac{\mathrm d}{\mathrm dz}\operatorname{LReLU}_\alpha(z)=\alpha.
$$

若 $\alpha=0.01$ 且 $z=-2$，输出是 $-0.02$，导数是 $0.01$。这条梯度很小，却不再严格为 0；即便神经元暂时落到负侧，参数仍可能被数据推回正侧。

PReLU 把 $\alpha$ 也作为可学习参数。对负侧样本，

$$
\frac{\partial}{\partial\alpha}\operatorname{PReLU}_\alpha(z)=z,
\qquad z<0.
$$

它增加了一个自由度，也增加了过拟合、参数约束和跨设备实现的一致性问题。选择 Leaky ReLU 或 PReLU 时，要记录负侧斜率是否固定、是否参与权重衰减，以及导出模型是否保留了同一个参数。

softplus 是另一种思路：

$$
\operatorname{softplus}(z)=\log(1+e^z).
$$

它处处可导，并且

$$
\frac{\mathrm d}{\mathrm dz}\operatorname{softplus}(z)
=\frac{e^z}{1+e^z}
=\sigma(z).
$$

负侧导数不为 0，正侧导数逐渐接近 1；代价是它没有 ReLU 的精确零输出，也会在大负值处产生很小的 sigmoid 梯度。若只需要正侧线性和稀疏零值，ReLU 更直接；若需要平滑的一阶导数，softplus 才有明确理由。

在 $z=-2,0,2$ 上比较几个函数：

| $z$ | ReLU | Leaky ReLU（$\alpha=0.01$） | softplus | sigmoid 导数 |
| ---: | ---: | ---: | ---: | ---: |
| $-2$ | $0$ | $-0.02$ | $0.126928$ | $0.104994$ |
| $0$ | $0$ | $0$ | $0.693147$ | $0.25$ |
| $2$ | $2$ | $2$ | $2.126928$ | $0.104994$ |

最后一列不是 softplus 的输出，而是它的导数，用来提醒“平滑”不等于“梯度始终大”。

## ReLU 不适合所有输出头

ReLU 作为隐藏层激活时，输出为 0 往往是可接受的；作为最终输出头时，非负约束则必须和任务语义相配。

| 任务目标 | 常见输出头 | 为什么 |
| --- | --- | --- |
| 实数回归，可正可负 | identity | 不人为截断负预测 |
| 非负回归，如计数或尺度 | softplus 或 ReLU | 保证输出非负，softplus 通常更平滑 |
| 二分类概率 | sigmoid | 输出位于 $(0,1)$，可配 BCE |
| 多分类概率 | softmax | 各类概率和为 1 |
| 多标签概率 | 独立 sigmoid | 每个标签有自己的概率 |

把 ReLU 放在二分类 logits 前面会把所有负分数折成 0，损失无法区分原本不同的负证据；把它放在可正可负的回归头前面会直接禁止负预测。最后一层的激活要由输出空间和损失函数决定，不能因为隐藏层常用 ReLU 就一路复制到输出端。

## 数值实现要注意折点和溢出边界

ReLU 的前向计算本身简单，但仍有几个实现边界。

第一，零点约定要前后一致。训练框架的 backward 可能把 $z=0$ 的梯度设为 0；自定义算子若设为 1，梯度检查在精确零点附近会得到不同结果。测试时要分别覆盖负值、正值和零值。

第二，ReLU 通常不会像指数函数那样溢出，因为它只复制正输入；但上游仿射层仍可能产生 `inf` 或 `nan`。若 $z=+\infty$，输出仍是 $+\infty$；若 $z=\mathrm{nan}$，比较运算和输出行为要以具体框架为准，不能把 ReLU 当成 NaN 清理器。

第三，混合精度会把很小的负侧变体斜率或梯度舍入掉。普通 ReLU 的负侧本来就是 0；Leaky ReLU 的 $\alpha$ 若太小，低精度下可能失去与普通 ReLU 的实际差异。

## 失效模式

**把稀疏当成死亡。** 单个 batch 中输出为 0 只说明这些样本在负侧；要按神经元跨批次统计活动率，并同时查看权重和偏置梯度。

**把正侧不饱和当成全路径不衰减。** 只要某一层的预激活落到负侧，ReLU 门就会把梯度置零；多个线性层的权重本身也可能产生很小的奇异方向。诊断要沿完整路径记录门状态和 Jacobian，而不是只看激活函数公式。

**把 He 初始化当成方差万能保证。** $q=2/n$ 的推导依赖独立、零均值、近似对称的输入，并且保持的是二阶矩。偏置、相关输入、归一化层、残差连接和训练后的权重都会改变这些假设。

**在输出头盲用 ReLU。** 先写出目标的允许范围，再决定是否需要非负约束；分类概率和可正可负回归通常不应该由 ReLU 直接承担。

**忽略零点次梯度。** 量化输入、裁剪操作和人工构造的边界样本可能反复命中 $z=0$。实现文档、梯度检查和部署端算子必须使用同一个约定。

## 一个可复用的 ReLU 核验协议

遇到 ReLU 隐藏层或自定义实现，可以依次检查：

1. 写出实际形式，确认是 $\max(0,z)$，还是带斜率、偏置或输出缩放的变体。
2. 用 $z<0,z=0,z>0$ 三个输入检查前向值，并记录零点的次梯度约定。
3. 对一个正侧和一个负侧样本各做一次链式法则，确认 $g_z=g_a\mathbf1\{z>0\}$。
4. 按神经元和按批次统计活动率、输出均值、输出二阶矩以及 $w,b$ 的梯度范数。
5. 若采用 He 初始化，先核对输入宽度、输入方差和权重方差，再区分二阶矩保持与方差保持。
6. 检查学习率、偏置初始化、输入标准化和数据切分是否把大量预激活推到负侧。
7. 确认输出头与目标空间匹配，确认导出端、混合精度端和训练端的零点行为一致。

这份检查把三个问题分开：ReLU 公式是否实现正确，梯度是否在活动掩码下传播，训练分布是否让神经元长期没有机会进入正侧。三者不能用同一个“loss 在下降”指标替代。

## 相关词条

- [激活函数](../neurons-and-activations/activation-functions/)：比较激活函数的范围、导数和输出头职责。
- [阶跃与 sigmoid](../neurons-and-activations/heaviside-and-sigmoid/)：从不可导的硬阈值过渡到平滑门控。
- [tanh](../neurons-and-activations/tanh/)：对比零中心但两侧饱和的激活。
- [死亡 ReLU 与 Leaky ReLU](../neurons-and-activations/dead-relu-and-leaky/)：展开活动率诊断、恢复方法和负侧变体。
- [GELU](../neurons-and-activations/gelu/)：用平滑概率门替代硬折点。
- [Swish 与 SwiGLU](../neurons-and-activations/swish-and-swiglu/)：比较带 sigmoid 门控的平滑激活和门控结构。
- [饱和与梯度消失](../neurons-and-activations/saturation-and-vanishing/)：把激活导数的局部效应放到深层链式乘积中。
- [逐分量导数](../calculus/elementwise-derivatives/)：从 Jacobian 和逐分量链式法则统一看激活层反向传播。
