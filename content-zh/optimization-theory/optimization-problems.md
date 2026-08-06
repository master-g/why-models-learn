---
title: "优化问题：在允许的选择里找一个最好的"
tags: ["why-models-learn"]
---

**优化问题**是在一组允许的选择中，寻找让目标函数最小或最大的元素。最小化形式写成

$$
\min_{\boldsymbol x\in\mathcal C} f(\boldsymbol x)
$$

其中 $\boldsymbol x$ 是决策变量，$f$ 是目标函数，$\mathcal C$ 是可行集。机器学习训练通常也是这个结构：$\boldsymbol x$ 换成参数 $\boldsymbol\theta$，目标换成数据上的平均损失，正则化项和约束则改变可行集或目标本身。本篇先建立这些对象的边界，区分最优值、最优解和下确界，再连接经验风险、惩罚方法、随机目标和神经网络训练；凸性、局部最小值、最优性条件和具体算法留给后续词条。

## 一个优化问题由什么组成

把一个问题写完整，至少要回答四件事：

| 组成部分 | 符号 | 要回答的问题 | 机器学习例子 |
| --- | --- | --- | --- |
| 决策变量 | $\boldsymbol x$ | 哪些量可以改变 | 权重 $\boldsymbol\theta$ |
| 目标函数 | $f(\boldsymbol x)$ | 什么叫更好 | 平均交叉熵 |
| 可行集 | $\mathcal C$ | 哪些选择被允许 | 参数范围或约束 |
| 优化方向 | $\min$ 或 $\max$ | 是降低代价还是提高收益 | 最小化损失 |

无约束问题只是把可行集取成整个空间：

$$
\min_{\boldsymbol x\in\mathbb R^d}f(\boldsymbol x)
$$

有约束问题则把限制写成集合：

$$
\min_{\boldsymbol x}f(\boldsymbol x)
\quad\text{subject to}\quad
\boldsymbol x\in\mathcal C
$$

例如

$$
\mathcal C
=\left\{
\boldsymbol x:
g_i(\boldsymbol x)\leq0,\ i=1,\ldots,m,\quad
h_j(\boldsymbol x)=0,\ j=1,\ldots,r
\right\}
$$

其中 $g_i$ 是不等式约束，$h_j$ 是等式约束。只有满足所有约束的点才属于可行集，目标函数在不可行点上数值再低，也不能当作这个问题的答案。

## 最优解、最优值和 argmin

若某个 $\boldsymbol x^\star\in\mathcal C$ 满足

$$
f(\boldsymbol x^\star)
\leq f(\boldsymbol x)
\qquad
\text{for all }\boldsymbol x\in\mathcal C
$$

就称 $\boldsymbol x^\star$ 是全局最小解，最小值是

$$
f^\star=f(\boldsymbol x^\star)
$$

解和解的值不是同一个对象：

$$
\operatorname*{arg\,min}_{\boldsymbol x\in\mathcal C}f(\boldsymbol x)
$$

表示达到最小值的所有点组成的集合，而

$$
\min_{\boldsymbol x\in\mathcal C}f(\boldsymbol x)
$$

表示最小的那个函数值。如果最优解不唯一，argmin 可以有多个元素，但最优值仍然是一个数。

最大化可以改写成最小化负函数：

$$
\max_{\boldsymbol x\in\mathcal C}r(\boldsymbol x)
=\min_{\boldsymbol x\in\mathcal C}-r(\boldsymbol x)
$$

因此很多算法只需要实现最小化；改变符号不会改变最优点，只会改变目标值的符号。

## 一个最简单的无约束例子

考虑一维目标

$$
f(x)=(x-3)^2+1
$$

平方项永远非负，且只有 $x=3$ 时为 0，所以

$$
\operatorname*{arg\,min}_{x\in\mathbb R}f(x)=\{3\},
\qquad
\min_xf(x)=1
$$

用导数也能确认：

$$
f'(x)=2(x-3),
\qquad
f'(3)=0
$$

取三个点直接算：

$$
f(2)=2,
\qquad
f(3)=1,
\qquad
f(4)=2
$$

这里的「最优」来自目标函数的定义。如果把目标改成离 $x=0$ 更近，答案就会改变；优化算法只负责寻找指定目标的优点，不负责判断目标是否代表了我们真正想要的东西。

### 约束会改变答案

把同一个目标限制为

$$
\min_x(x-3)^2
\quad\text{subject to}\quad
x\leq2
$$

无约束最优点 $x=3$ 不可行。可行点中离 3 最近的是边界 $x=2$，所以

$$
x^\star=2,
\qquad
f(x^\star)=1
$$

这两个问题的目标函数相同，但可行集不同，因此答案不同。约束不是算法运行时的附加说明，而是问题定义的一部分。

### 等式约束的几何例子

考虑把原点到直线 $x_1+x_2=1$ 的距离平方最小化：

$$
\min_{x_1,x_2}
x_1^2+x_2^2
\quad\text{subject to}\quad
x_1+x_2=1
$$

由约束可令 $x_2=1-x_1$，目标变成

$$
\begin{aligned}
f(x_1)
&=x_1^2+(1-x_1)^2\\
&=2\left(x_1-\frac12\right)^2+\frac12
\end{aligned}
$$

所以

$$
\boldsymbol x^\star
=\left(\frac12,\frac12\right),
\qquad
f(\boldsymbol x^\star)=\frac12
$$

这就是正交投影的优化说法：在可行直线上寻找离原点最近的点。后续的约束优化会用拉格朗日乘子把这种几何条件写成方程。

## 最小值不一定被某个点达到

「向下有界」不等于「存在最小解」。考虑

$$
f(x)=e^x,
\qquad
x\in\mathbb R
$$

它始终大于 0，但当 $x\to-\infty$ 时趋近 0。因此

$$
\inf_{x\in\mathbb R}e^x=0
$$

但不存在满足 $e^{x^\star}=0$ 的有限 $x^\star$，所以

$$
\operatorname*{arg\,min}_{x\in\mathbb R}e^x=\varnothing
$$

这里应该写下确界 $\inf$，不能写成被某个点达到的 $\min$。另一个相反的边界是目标没有有限下界：

$$
f(x)=x
\quad\Longrightarrow\quad
\inf_{x\in\mathbb R}f(x)=-\infty
$$

常见的存在性保证来自 Weierstrass 定理：非空紧可行集上的连续函数一定能达到最小值。直觉是可行集没有跑向无穷远，也没有漏掉极限点；连续性则阻止目标在极限处突然跳走。机器学习参数空间经常不紧，训练目标也可能有平坦方向或未达到的极限，所以「优化器跑了很多步」不能证明全局最优解存在。

## 机器学习中的目标函数

监督学习给定样本

$$
\mathcal D=\{(\boldsymbol x_i,y_i)\}_{i=1}^n
$$

和带参数的预测器 $f_{\boldsymbol\theta}$。单样本损失记为

$$
\ell\left(f_{\boldsymbol\theta}(\boldsymbol x_i),y_i\right)
$$

最常见的经验风险目标是

$$
\min_{\boldsymbol\theta}
J_{\mathrm{emp}}(\boldsymbol\theta)
=\frac1n\sum_{i=1}^n
\ell\left(f_{\boldsymbol\theta}(\boldsymbol x_i),y_i\right)
$$

它的含义是：在训练样本上选一组参数，让平均损失尽量低。经验风险不是抽象的「模型好坏」，而是针对一组样本、一个损失函数和一个参数化定义出来的数。

例如回归使用平方损失：

$$
\ell(\widehat y,y)=(\widehat y-y)^2
$$

得到

$$
J_{\mathrm{MSE}}(\boldsymbol\theta)
=\frac1n\sum_{i=1}^n
\left(f_{\boldsymbol\theta}(\boldsymbol x_i)-y_i\right)^2
$$

分类使用交叉熵：

$$
J_{\mathrm{CE}}(\boldsymbol\theta)
=-\frac1n\sum_{i=1}^n
\log q_{\boldsymbol\theta}(y_i\mid\boldsymbol x_i)
$$

两个模型在 MSE 上的优劣，不必与它们在准确率、校准误差或交叉熵上的优劣一致，因为它们解决的是不同的优化问题。

### 正则化是目标的一部分

如果在经验风险上加入参数惩罚：

$$
J_\lambda(\boldsymbol\theta)
=J_{\mathrm{emp}}(\boldsymbol\theta)
 +\lambda R(\boldsymbol\theta)
$$

那么 $\lambda$ 和 $R$ 改变了最优解。它不是在原问题上「顺手把解修得更漂亮」，而是定义了一个新目标。

一维例子：

$$
J_\lambda(\theta)
=(\theta-3)^2+\lambda\theta^2
$$

对 $\theta$ 求导并令其为 0：

$$
\begin{aligned}
J_\lambda'(\theta)
&=2(\theta-3)+2\lambda\theta\\
&=2(1+\lambda)\theta-6=0
\end{aligned}
$$

所以

$$
\theta^\star=\frac{3}{1+\lambda}
$$

当 $\lambda=0$ 时 $\theta^\star=3$；当 $\lambda=1$ 时 $\theta^\star=1.5$。正则化把解从拟合数据的方向拉向 0，代价是训练目标的最小值也不再是原始平方误差的最小值。

### 经验风险和总体风险

如果数据来自分布 $P$，真正关心的可能是总体风险：

$$
J_{\mathrm{pop}}(\boldsymbol\theta)
=\mathbb E_{(\boldsymbol X,Y)\sim P}
\left[
\ell\left(f_{\boldsymbol\theta}(\boldsymbol X),Y\right)
\right]
$$

训练时只能用有限样本近似它：

$$
J_{\mathrm{emp}}(\boldsymbol\theta)
\approx J_{\mathrm{pop}}(\boldsymbol\theta)
$$

优化经验风险和得到低总体风险是两个不同问题。前者是数值优化，后者还涉及抽样误差、分布偏移和模型的泛化能力。把训练 loss 降低并不能单独证明测试集会变好。

## 可行集的几种来源

可行集可以来自数学结构、资源限制或业务规则。

### 盒约束

对每个参数指定范围：

$$
\mathcal C
=\{\boldsymbol\theta:
\ell_j\leq\theta_j\leq u_j,\ j=1,\ldots,d\}
$$

这表达了参数不能超出边界。它和训练后把参数裁剪回范围不同：裁剪是某种算法步骤，盒约束是算法试图解决的问题；若算法没有正确处理边界，裁剪不一定对应原问题的最优解。

### 线性约束

资源分配常写成

$$
\boldsymbol A\boldsymbol x\leq\boldsymbol b,
\qquad
\boldsymbol C\boldsymbol x=\boldsymbol d
$$

例如 $x_1,x_2\geq0$ 且 $x_1+x_2\leq1$ 的可行集是第一象限中的三角形。目标函数告诉我们在三角形中偏向哪一边，约束决定哪些方向根本不能走。

### 离散可行集

如果 $\boldsymbol x$ 是排列、路径、子集或整数向量，问题仍然是优化问题：

$$
\min_{\boldsymbol x\in\{0,1\}^d}f(\boldsymbol x)
$$

但连续导数可能不存在，不能直接把所有实数空间里的方法套上去。组合优化的可行点数量可以随 $d$ 指数增长，问题难点来自搜索空间，不只是目标函数是否光滑。

## 硬约束和惩罚项不是一回事

硬约束要求解永远在可行集内：

$$
\min_xf(x)
\quad\text{subject to}\quad
g(x)\leq0
$$

惩罚方法把违反约束的程度加到目标里：

$$
\min_x f(x)+\lambda\,[\max(0,g(x))]^2
$$

有限的 $\lambda$ 通常只是在违反约束时增加代价，不保证最优点严格可行。以

$$
\min_x(x-3)^2
\quad\text{subject to}\quad
x\leq2
$$

为例，用 $\lambda=1$ 的平方惩罚替代约束：

$$
\widetilde f(x)
=(x-3)^2+[\max(0,x-2)]^2
$$

在违反约束的区域 $x>2$ 内：

$$
\widetilde f(x)
=(x-3)^2+(x-2)^2
$$

其导数为

$$
\widetilde f'(x)=2(x-3)+2(x-2)
$$

零点是 $x=2.5$，它确实违反了 $x\leq2$。惩罚后的目标在 $x=2.5$ 处取值 $0.5$，而硬约束问题的最优点是 $x=2$。增大 $\lambda$ 可以把惩罚解推近边界，但「近」与「满足约束」仍是两种说法；精确约束需要专门的可行算法或约束优化方法。

## 目标的尺度会影响算法，但不一定影响最优点

把目标整体乘以正数 $c$：

$$
\widetilde f(\boldsymbol x)=c\,f(\boldsymbol x),
\qquad
c>0
$$

不会改变 argmin，因为所有候选点的大小顺序保持不变：

$$
\operatorname*{arg\,min}\widetilde f
=\operatorname*{arg\,min}f
$$

但梯度变成

$$
\nabla\widetilde f=c\nabla f
$$

固定学习率的梯度法会因此改变步长效果。把 MSE 从平均值改成总和、把 loss 从 nats 改成 bits、给正则化项换单位，都可能不改变某些理论最优点，却会改变梯度大小和超参数含义。

更危险的是只缩放目标的一部分：

$$
f(\boldsymbol x)+\lambda R(\boldsymbol x)
$$

中，改变数据项的单位而不调整 $\lambda$ 会改变两部分的相对权重，从而改变 argmin。正则化系数不是脱离损失单位的通用常数。

## 随机优化问题

当目标本身包含随机样本或噪声时，问题常写成期望：

$$
\min_{\boldsymbol\theta}
J(\boldsymbol\theta)
=\mathbb E_{\boldsymbol\xi}
\left[
F(\boldsymbol\theta;\boldsymbol\xi)
\right]
$$

有限样本近似是

$$
\widehat J_n(\boldsymbol\theta)
=\frac1n\sum_{i=1}^n
F(\boldsymbol\theta;\boldsymbol\xi_i)
$$

小批次方法在第 $k$ 步只用一部分样本估计梯度：

$$
\widehat{\boldsymbol g}_k
=\frac1{|B_k|}
\sum_{i\in B_k}
\nabla_{\boldsymbol\theta}
F(\boldsymbol\theta_k;\boldsymbol\xi_i)
$$

所以一次更新看到的是随机梯度，不是总体目标的精确梯度。随机性是算法获得可扩展性的来源，也会带来梯度方差、学习率和收敛判定的问题；后续的 SGD 理论会单独处理这些误差。

需要区分三种对象：

- 总体目标 $J$ 是希望解决的数学问题。
- 经验目标 $\widehat J_n$ 是手上数据定义出的近似问题。
- 小批次梯度是计算经验目标梯度的一个带噪估计。

把三者混称为「训练 loss」会掩盖优化误差和统计误差分别来自哪里。

## 神经网络中的优化问题

对神经网络，决策变量是所有层的参数：

$$
\boldsymbol\theta
=\{\boldsymbol W_1,\boldsymbol b_1,\ldots,
\boldsymbol W_L,\boldsymbol b_L\}
$$

一个带权重衰减的训练目标可以写成

$$
\min_{\boldsymbol\theta}
\frac1n\sum_{i=1}^n
\ell\left(f_{\boldsymbol\theta}(\boldsymbol x_i),y_i\right)
 +\lambda\sum_{\ell=1}^L
\|\boldsymbol W_\ell\|_F^2
$$

这里至少有四个层次：

1. 数据和标签定义了经验风险。
2. 损失函数定义了「错」的代价。
3. 正则化和约束定义了额外偏好。
4. 优化算法只是在计算资源和精度限制下寻找一个低目标值的参数。

神经网络目标通常是非凸的，可能有多个临界点、平坦方向和参数对称性。即使训练得到低 loss，也需要分开询问：它是不是全局最小值，是否泛化，是否满足约束，是否对应用户真正关心的指标。这些问题分别属于后续优化和学习理论词条，不应被一次训练曲线代替。

![优化问题是在可行集内寻找最低等高线](/assets/optimization-theory/svg/optimization-problems.1.svg)

## 优化器到底在保证什么

一个优化器通常提供的是迭代序列

$$
\boldsymbol x_0,\boldsymbol x_1,\boldsymbol x_2,\ldots
$$

以及某种停止条件，例如

$$
\|\nabla f(\boldsymbol x_k)\|\leq\varepsilon
$$

或

$$
|f(\boldsymbol x_{k+1})-f(\boldsymbol x_k)|
\leq\varepsilon
$$

这些是算法的近似判据，不自动等于

$$
\boldsymbol x_k\in\operatorname*{arg\,min}f
$$

梯度很小可能意味着已经接近最优点，也可能来自鞍点、平坦区域、数值精度或一个尺度不合适的参数化。要把停止条件升级成最优性结论，需要目标的凸性、光滑性、约束资格或其他假设。后续的「一阶最优性条件」、「二阶最优性条件」和「梯度下降理论」会分别说明这些条件。

## 失效模式

**把最优值和最优解混为一谈。** $f^\star$ 是一个数，argmin 是一个点集；报告训练 loss 不能代替报告参数、约束状态或预测行为。

**只看目标，不看可行集。** 不可行点即使目标更低，也不是约束问题的候选解。

**把 infimum 写成 min。** 目标可以无限接近下界而永远达不到；$e^x$ 在 $\mathbb R$ 上就是这个例子。

**把惩罚项当成硬约束。** 有限惩罚只增加违规代价，不保证解满足原约束。

**忘记正则化改变了问题。** 加上 $\lambda R(\theta)$ 后，最优解通常不再是原始经验风险的最优解；$\lambda$ 还依赖损失单位。

**把经验风险的最优当成总体风险的最优。** 训练集 loss 是有限样本上的数值，不包含泛化保证。

**把随机梯度当成精确梯度。** 小批次梯度是估计量，单步上升不一定代表总体目标变差，长期收敛需要额外假设。

**把梯度小当成全局最优。** 非凸目标中，梯度小可能来自鞍点、平坦区域或数值误差；必须结合目标结构解释。

## 相关词条

- [凸集与凸函数](../optimization-theory/convex-sets-and-functions/)：说明一类重要目标何时能把局部结论升级为全局结论。
- [局部与全局最小值](../optimization-theory/local-and-global-minima/)：区分邻域内最优和整个可行集内最优。
- [一阶最优性条件](../optimization-theory/first-order-optimality/)：解释梯度为零、边界和约束下的必要条件。
- [二阶最优性条件](../optimization-theory/second-order-optimality/)：用 Hessian 的曲率区分局部极小、极大和鞍点。
- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：分析迭代步长、光滑性和收敛速度。
- [约束优化](../optimization-theory/constrained-optimization/)：系统处理可行集、拉格朗日乘子和约束算法。
- [最小二乘是投影](../linear-models/least-squares-as-projection/)：把平方残差优化和正交投影连接起来。
- [cross-entropy](../information-theory/cross-entropy/)：分类训练中常用的条件概率目标。
- [maximum-likelihood](../probability/maximum-likelihood/)：说明最大似然如何转成负对数似然最小化。
