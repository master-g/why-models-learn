---
title: "鞍点：梯度为零但方向有升有降"
tags: ["why-models-learn"]
---

鞍点是一个驻点附近同时存在上升方向和下降方向的点。对最小化问题来说，它不是局部最小值，却可能让梯度暂时很小；对 min-max 问题来说，同样的几何结构反而正是要寻找的解。Hessian 的正负特征值可以识别严格鞍点，梯度下降在负曲率方向会远离它，随机扰动也因此通常能把轨迹从鞍点邻域推开。本文先从二维曲面和线性动力学说明鞍点的局部结构，再区分退化鞍点、损失景观中的鞍点以及博弈问题里的目标鞍点。

![鞍点沿不同方向一升一降](/assets/optimization-theory/svg/saddle-points.1.svg)

## 局部最低点不是唯一的驻点

驻点只满足

$$
\nabla F(\boldsymbol\theta^\star)=\boldsymbol0
$$

这条方程没有说明附近的函数值是向上、向下，还是不同方向混合。对一个小位移 $\boldsymbol\delta$，二阶展开为

$$
F(\boldsymbol\theta^\star+\boldsymbol\delta)
\approx
F(\boldsymbol\theta^\star)
+\frac12\boldsymbol\delta^\mathsf TH(\boldsymbol\theta^\star)\boldsymbol\delta
$$

如果二次型在某个方向为正、另一个方向为负，点就是鞍点。

最简单的例子是

$$
F(x,y)=\frac12(x^2-y^2)
$$

原点处

$$
\nabla F(0,0)=\boldsymbol0,
\qquad
H=
\begin{bmatrix}
1&0\\
0&-1
\end{bmatrix}
$$

沿两个坐标轴看：

| 位移 | 函数值变化 | 方向 |
| --- | --- | --- |
| $(t,0)$ | $t^2/2$ | 离开原点时上升 |
| $(0,t)$ | $-t^2/2$ | 离开原点时下降 |
| $(0.1,0)$ | $0.005$ | 正曲率方向 |
| $(0,0.1)$ | $-0.005$ | 负曲率方向 |

所以原点既不是局部最小，也不是局部最大。它像马鞍的座面：一个方向抬高，另一个方向压低。

## Hessian 如何识别严格鞍点

设 $H$ 是驻点处的对称 Hessian，令其特征值为

$$
\lambda_1,\ldots,\lambda_d
$$

把位移写到正交特征基中，二次项变为

$$
\frac12\boldsymbol\delta^\mathsf TH\boldsymbol\delta
=\frac12\sum_{i=1}^{d}\lambda_i z_i^2
$$

因此：

| Hessian 特征值 | 局部二次模型 |
| --- | --- |
| 全部正 | 严格局部最小的二阶证据 |
| 全部负 | 严格局部最大的二阶证据 |
| 有正有负 | 严格鞍点 |
| 有零特征值 | 二阶测试不充分，需要更高阶项 |

严格鞍点的条件是至少有一个正特征值和一个负特征值。负特征值对应的特征向量 $\boldsymbol v$ 满足

$$
\boldsymbol v^\mathsf TH\boldsymbol v
=\lambda\lVert\boldsymbol v\rVert^2<0
$$

沿 $\boldsymbol v$ 移动会降低局部二次模型。它不是「梯度很大」的方向，恰恰可能在驻点处梯度为零；曲率提供了离开它的信号。

## 梯度下降在负曲率方向会放大误差

对

$$
F(x,y)=\frac12(x^2-y^2)
$$

梯度为

$$
\nabla F(x,y)=(x,-y)^\mathsf T
$$

梯度下降更新：

$$
x_{k+1}=(1-\eta)x_k,
\qquad
y_{k+1}=(1+\eta)y_k
$$

当 $0<\eta<2$ 时，正曲率方向 $x$ 的误差衰减；但只要 $\eta>0$，负曲率方向 $y$ 的放大因子 $1+\eta$ 就大于 $1$。如果初始点刚好在 $y=0$ 的稳定直线上，算法可以一直留在这条直线上；只要有非零的 $y$ 分量，就会逐渐离开原点。

### 数值轨迹

取 $\eta=0.2$、$(x_0,y_0)=(1,0.1)$，三步更新为

| 步骤 | $x_k$ | $y_k$ | $F(x_k,y_k)$ |
| --- | --- | --- | --- |
| 0 | $1$ | $0.1$ | $0.495$ |
| 1 | $0.8$ | $0.12$ | $0.3128$ |
| 2 | $0.64$ | $0.144$ | $0.194432$ |
| 3 | $0.512$ | $0.1728$ | $0.116142$ |

这里函数值仍然下降，因为正曲率方向的能量起初更大；但 $y$ 坐标没有向零收敛，而是在增长。再继续迭代，负曲率方向最终会主导函数值。

从线性代数看，梯度下降在 Hessian 特征方向上的误差因子是

$$
1-\eta\lambda_i
$$

当 $\lambda_i>0$ 时可以选择 $\eta$ 让因子模小于 $1$；当 $\lambda_i<0$ 时

$$
1-\eta\lambda_i=1+\eta\lvert\lambda_i\rvert>1
$$

所以负曲率方向天然是不稳定方向。这解释了为什么小扰动常常足以让优化轨迹离开严格鞍点。

## 严格鞍点与退化鞍点

严格鞍点的 Hessian 已经有负特征值，二阶信息能直接指出逃逸方向。但有些鞍点的 Hessian 没有负特征值，二阶测试会停在这里。

看

$$
F(x,y)=x^2-y^4
$$

原点处

$$
\nabla F(0,0)=\boldsymbol0,
\qquad
H(0,0)=
\begin{bmatrix}
2&0\\
0&0
\end{bmatrix}
$$

Hessian 的特征值是 $2$ 和 $0$，没有负特征值。但沿两条路径：

$$
F(t,0)=t^2>0,
\qquad
F(0,t)=-t^4<0
\quad (t\neq0)
$$

仍然同时有上升和下降方向，所以原点是退化鞍点。数值上，$t=0.1$ 时两个值分别是 $0.01$ 和 $-0.0001$。高阶项在零曲率方向上承担了分类任务。

这类点和平台不能混为一谈。平台可能在一小片区域内函数变化都很小，但不一定有明确的正负方向；退化鞍点则必须存在两条趋近该点、函数值一正一负的路径。

## 为什么随机梯度常常能离开鞍点

若严格鞍点的稳定子空间维数小于参数总维数，刚好落在稳定子空间上的初始点集合通常很薄。随机初始化或小批次噪声给参数一个非零的负曲率分量，随后该分量按

$$
(1+\eta\lvert\lambda_-\rvert)^k
$$

放大，轨迹离开鞍点邻域。

这不是「噪声越大越好」。噪声过大时，正曲率方向也会被反复踢出谷底；学习率过大时，逃离鞍点后可能在高曲率方向发散。有效的逃逸需要在负曲率方向能积累位移，同时在正曲率方向保持稳定。

在满足光滑性、步长和噪声条件的严格鞍点分析中，常见策略包括：

- 在梯度范数很小但 Hessian 最小特征值为负时加入小扰动；
- 用 Hessian-向量积寻找负曲率方向；
- 使用信赖域或三次正则化限制离开步的长度；
- 用随机梯度的自然噪声打破稳定子空间。

如果 Hessian 没有负特征值而只是零特征值，严格鞍点的结论不能直接套用，必须检查更高阶项或使用其他几何信息。

## 最小化里的坏点与 min-max 里的目标

在普通最小化中，鞍点是要避开的驻点；但在 min-max 问题

$$
\min_{\boldsymbol x}\max_{\boldsymbol y}F(\boldsymbol x,\boldsymbol y)
$$

中，目标点要求对 $\boldsymbol x$ 是局部最小、对 $\boldsymbol y$ 是局部最大。理想的鞍点满足

$$
F(\boldsymbol x^\star,\boldsymbol y)
\leq
F(\boldsymbol x^\star,\boldsymbol y^\star)
\leq
F(\boldsymbol x,\boldsymbol y^\star)
$$

对邻域内的所有 $\boldsymbol x,\boldsymbol y$ 成立。

考虑

$$
F(x,y)=\frac12x^2+xy-\frac12y^2
$$

在 $(0,0)$ 处，对 $x$ 固定 $y=0$，有

$$
F(x,0)=\frac12x^2\geq0
$$

对 $y$ 固定 $x=0$，有

$$
F(0,y)=-\frac12y^2\leq0
$$

因此 $(0,0)$ 是联合变量意义下的鞍点，却是这个 min-max 问题希望找到的点。判断一个鞍点好不好，必须先说明每个变量的优化方向。

### 下降—上升动力学

对 $x$ 做下降、对 $y$ 做上升：

$$
x_{k+1}=x_k-\eta(x_k+y_k)
$$

$$
y_{k+1}=y_k+\eta(x_k-y_k)
$$

写成矩阵：

$$
\begin{bmatrix}
x_{k+1}\\
y_{k+1}
\end{bmatrix}
=
\begin{bmatrix}
1-\eta&-\eta\\
\eta&1-\eta
\end{bmatrix}
\begin{bmatrix}
x_k\\
y_k
\end{bmatrix}
$$

特征值是

$$
1-\eta+\mathrm i\eta,
\qquad
1-\eta-\mathrm i\eta
$$

其模为

$$
\sqrt{(1-\eta)^2+\eta^2}
$$

取 $\eta=0.4$，模约为 $0.7211$，小于 $1$。从 $(1,0)$ 出发：

| 步骤 | $x_k$ | $y_k$ | $\lVert(x_k,y_k)\rVert$ |
| --- | --- | --- | --- |
| 0 | $1$ | $0$ | $1$ |
| 1 | $0.6$ | $0.4$ | $0.7211$ |
| 2 | $0.2$ | $0.48$ | $0.52$ |
| 3 | $-0.072$ | $0.368$ | $0.3750$ |

这里坐标会旋转，但半径衰减，最终趋近 min-max 鞍点。若把同一个函数错误地当作联合最小化目标，$y$ 方向的负曲率会被误判为“需要逃离”；若把普通最小化的方法直接用于 min-max，又会得到不同的动力学。

作为对照，纯双线性函数

$$
F(x,y)=xy
$$

的下降—上升矩阵是

$$
\begin{bmatrix}
1&-\eta\\
\eta&1
\end{bmatrix}
$$

特征值模为 $\sqrt{1+\eta^2}>1$，轨迹会旋转并发散，而不是收敛。min-max 中还需要阻尼、额外曲率或外推等机制，不能只看到“有鞍点”就预测算法会自动到达它。

## 诊断一个梯度很小的点

训练日志里看到 $\lVert\nabla F\rVert$ 很小，只能说明一阶变化小。要区分最小点、最大点、鞍点和平台，至少检查：

| 检查量 | 最小点的信号 | 鞍点的信号 |
| --- | --- | --- |
| 梯度范数 | 小 | 也可能小 |
| 最小 Hessian 特征值 | 非负，严格时为正 | 严格鞍点时为负 |
| 随机方向扰动 | 函数值通常上升 | 有些方向上升，有些下降 |
| 梯度迭代轨迹 | 进入点后趋于稳定 | 负曲率分量被放大 |
| min-max 角色 | 不适用 | 可能正是目标解 |

大规模模型通常不能直接求完整 Hessian，但可以用 Hessian-向量积、Lanczos 迭代或沿随机方向的二阶差分，估计是否存在明显负曲率。一次随机方向没有发现下降，不代表所有方向都没有负曲率；它只是一个低成本抽样。

## 常见失效模式

**把梯度为零当成最小值。** 驻点方程只是一阶必要条件。先看 Hessian 的符号，遇到零特征值再看高阶项。

**把所有负特征值都当成错误。** 普通最小化希望避开负曲率，但 min-max 的最大化变量本来就需要负曲率。算法方向决定了同一个 Hessian 的含义。

**用过大的噪声逃逸。** 扰动确实能打破稳定子空间，但也会破坏正曲率方向的收敛。要同时观察负曲率逃逸和目标窗口趋势。

**忽略退化鞍点。** Hessian 半正定不代表局部最小；$x^2-y^4$ 的零特征值正好隐藏了下降方向。

**把旋转当成收敛。** 双线性 min-max 的轨迹可能绕着原点转圈甚至增大半径。记录范数、原始/对偶残差和窗口半径，不要只看某一个坐标。

**用最小化的稳定性条件分析 min-max。** 下降—上升更新的谱半径由一个非对称矩阵决定，不能把普通梯度下降的 $1-\eta\lambda$ 直接套到全部变量。

## 相关词条

- [局部与全局最小值](../optimization-theory/local-and-global-minima/)：区分鞍点、局部谷底和边界最优点。
- [二阶最优性条件](../optimization-theory/second-order-optimality/)：用 Hessian 二次型分类驻点。
- [Hessian 矩阵](../calculus/hessian/)：负曲率和特征方向的来源。
- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：正曲率方向的稳定步长分析。
- [二阶方法](../optimization-theory/second-order-methods/)：信赖域和负曲率处理。
- [约束优化](../optimization-theory/constrained-optimization/)：约束问题中的驻点和可行方向。
- [对偶性](../optimization-theory/duality/)：原变量与乘子形成的鞍点结构。
- [损失景观](../optimization-theory/loss-landscapes/)：高维训练目标中的非凸几何。
- [曲率与条件数](../optimization-theory/curvature-and-conditioning/)：不同方向的尺度和稳定性。
- [动量理论](../optimization-theory/momentum-theory/)：带历史状态的非一阶动力学。
