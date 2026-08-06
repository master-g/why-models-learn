---
title: "对偶性：用约束构造目标值的另一条路径"
tags: ["why-models-learn"]
---

对偶性把一个约束优化问题称为原问题，再用拉格朗日乘子构造一个对偶问题。原问题直接寻找可行参数和最小目标值，对偶问题寻找一个尽可能紧的目标值下界。弱对偶性保证这个下界不会超过原问题的最优值；在凸性和适当约束条件下，两者可以相等，这叫强对偶性。对偶变量不只是记号，它们还表示约束边界放松一点时最优值会怎样变化。本文从一维例子推导拉格朗日对偶函数，再连接 KKT、对偶上升、二次问题和 Fenchel 共轭。

## 原问题和对偶问题

先写一个带不等式约束的最小化问题：

$$
p^\star
=\inf_{\boldsymbol x}
f(\boldsymbol x)
\quad\text{subject to}\quad
\boldsymbol g(\boldsymbol x)\leq\boldsymbol0
$$

对每个不等式引入乘子 $\boldsymbol\lambda\geq\boldsymbol0$，拉格朗日函数是

$$
\mathcal L(\boldsymbol x,\boldsymbol\lambda)
=f(\boldsymbol x)
+\boldsymbol\lambda^\mathsf T\boldsymbol g(\boldsymbol x)
$$

固定 $\boldsymbol\lambda$，对 $\boldsymbol x$ 取全空间下确界：

$$
q(\boldsymbol\lambda)
=\inf_{\boldsymbol x}
\mathcal L(\boldsymbol x,\boldsymbol\lambda)
$$

$q$ 称为对偶函数。然后最大化它：

$$
d^\star
=\sup_{\boldsymbol\lambda\geq\boldsymbol0}
q(\boldsymbol\lambda)
$$

原问题在可行域中最小化 $f$，对偶问题在乘子非负的区域中最大化 $q$。这两个问题的变量甚至可以属于不同空间：一个是模型参数，一个是约束的价格或敏感度。

等式约束的乘子不受符号限制。若

$$
\boldsymbol c(\boldsymbol x)=\boldsymbol0
$$

则

$$
\mathcal L(\boldsymbol x,\boldsymbol\lambda)
=f(\boldsymbol x)
+\boldsymbol\lambda^\mathsf T\boldsymbol c(\boldsymbol x),
\qquad
\boldsymbol\lambda\in\mathbb R^m
$$

等式可以从正反两个方向违反，所以乘子不能提前限制为非负。

![原问题与对偶问题从两侧逼近同一个值](/assets/optimization-theory/svg/duality.1.svg)

## 弱对偶：每个合法乘子都给出下界

取任意可行点 $\boldsymbol x$ 和任意 $\boldsymbol\lambda\geq\boldsymbol0$。因为

$$
\boldsymbol g(\boldsymbol x)\leq\boldsymbol0
$$

所以

$$
\boldsymbol\lambda^\mathsf T\boldsymbol g(\boldsymbol x)\leq0
$$

从而

$$
\mathcal L(\boldsymbol x,\boldsymbol\lambda)
\leq f(\boldsymbol x)
$$

而对偶函数是对所有 $\boldsymbol x$ 取下确界：

$$
q(\boldsymbol\lambda)
=\inf_{\boldsymbol y}\mathcal L(\boldsymbol y,\boldsymbol\lambda)
\leq\mathcal L(\boldsymbol x,\boldsymbol\lambda)
\leq f(\boldsymbol x)
$$

这对每个可行 $\boldsymbol x$ 和每个合法乘子都成立，因此

$$
d^\star\leq p^\star
$$

这就是弱对偶性。对偶问题给出的数值是原问题最优值的下界，不需要先知道原问题的最优解。

定义对偶间隙

$$
\operatorname{gap}=p^\star-d^\star\geq0
$$

间隙为零时，两个问题的最优值相同；间隙很大时，对偶下界对原问题的估计仍然松。

## 一维不等式例子

考虑

$$
\min_x\frac12x^2
\quad\text{subject to}\quad
x\geq1
$$

把约束写成

$$
g(x)=1-x\leq0
$$

原问题的最优点是 $x^\star=1$，最优值为

$$
p^\star=\frac12
$$

拉格朗日函数为

$$
\mathcal L(x,\lambda)
=\frac12x^2+\lambda(1-x),
\qquad
\lambda\geq0
$$

固定 $\lambda$，对 $x$ 求导：

$$
\frac{\partial\mathcal L}{\partial x}=x-\lambda
$$

当 $x=\lambda$ 时达到关于 $x$ 的最小值，代回得到

$$
q(\lambda)
=\frac12\lambda^2+\lambda(1-\lambda)
=\lambda-\frac12\lambda^2
$$

它是一个开口向下的抛物线。不同乘子给出：

| $\lambda$ | $q(\lambda)$ | 与 $p^\star=0.5$ 的关系 |
| --- | --- | --- |
| $0$ | $0$ | 下界较松 |
| $0.5$ | $0.375$ | 仍低于最优值 |
| $1$ | $0.5$ | 与原问题相等 |
| $1.5$ | $0.375$ | 超过峰值后下降 |

对偶问题是

$$
\max_{\lambda\geq0}
\lambda-\frac12\lambda^2
$$

最优乘子为 $\lambda^\star=1$，对偶值为 $d^\star=0.5$，所以这里

$$
d^\star=p^\star,
\qquad
\operatorname{gap}=0
$$

这个例子还显示了乘子的含义：约束 $x\geq1$ 若向外放松一点，最优目标可以下降；在边界处，乘子 $1$ 是这种边际变化的尺度。

## 等式乘子可以是负数

考虑

$$
\min_x\frac12x^2
\quad\text{subject to}\quad
x=2
$$

拉格朗日函数为

$$
\mathcal L(x,\lambda)
=\frac12x^2+\lambda(x-2)
$$

对 $x$ 取下确界时

$$
x+\lambda=0,
\qquad
x=-\lambda
$$

代回：

$$
q(\lambda)
=-\frac12\lambda^2-2\lambda
$$

它在

$$
q'(\lambda)=-\lambda-2=0
$$

处达到最大值，所以

$$
\lambda^\star=-2,
\qquad
q(\lambda^\star)=2
$$

原问题的唯一可行点是 $x=2$，目标值也是 $2$。如果错误地把等式乘子限制为 $\lambda\geq0$，会把真正的对偶最优点排除掉。

## 强对偶：什么时候下界能贴住原值

弱对偶只给出

$$
d^\star\leq p^\star
$$

强对偶则要求等号成立。一个常见的充分条件是：

- $f$ 是凸函数；
- 每个 $g_i$ 是凸函数；
- 等式约束是仿射的；
- 存在一个点严格满足全部不等式，即 $g_i(\boldsymbol x)<0$。

最后一条是 Slater 条件。它排除了可行域只有尖角、没有内部点等一部分退化情况。它是强对偶的充分条件，不是每个强对偶问题都必须满足的必要条件。

当强对偶成立并且最优解达到时，原解和对偶解通常满足 KKT 条件：

$$
\nabla f(\boldsymbol x^\star)
+\sum_i\lambda_i^\star\nabla g_i(\boldsymbol x^\star)
=\boldsymbol0
$$

$$
\boldsymbol g(\boldsymbol x^\star)\leq\boldsymbol0,
\qquad
\boldsymbol\lambda^\star\geq\boldsymbol0
$$

$$
\lambda_i^\star g_i(\boldsymbol x^\star)=0
$$

互补松弛说的是：不活动的约束 $g_i<0$ 必须有 $\lambda_i=0$；乘子非零的约束必须正好卡在边界。KKT 在这里不是一组孤立的方程，而是原问题和对偶问题在同一个鞍点处相遇的条件。

## 二次问题的对偶

考虑正定二次原问题：

$$
\min_{\boldsymbol x}
\frac12\boldsymbol x^\mathsf TQ\boldsymbol x
-\boldsymbol b^\mathsf T\boldsymbol x
\quad\text{subject to}\quad
A\boldsymbol x=\boldsymbol d
$$

其拉格朗日函数为

$$
\mathcal L(\boldsymbol x,\boldsymbol\lambda)
=\frac12\boldsymbol x^\mathsf TQ\boldsymbol x
-\boldsymbol b^\mathsf T\boldsymbol x
+\boldsymbol\lambda^\mathsf T(A\boldsymbol x-\boldsymbol d)
$$

把与 $\boldsymbol x$ 有关的线性项合并：

$$
\mathcal L
=\frac12\boldsymbol x^\mathsf TQ\boldsymbol x
-\left(\boldsymbol b-A^\mathsf T\boldsymbol\lambda\right)^\mathsf T\boldsymbol x
-\boldsymbol\lambda^\mathsf T\boldsymbol d
$$

关于 $\boldsymbol x$ 的最小点满足

$$
Q\boldsymbol x
=\boldsymbol b-A^\mathsf T\boldsymbol\lambda
$$

所以

$$
\boldsymbol x(\boldsymbol\lambda)
=Q^{-1}
(\boldsymbol b-A^\mathsf T\boldsymbol\lambda)
$$

代回得到对偶函数：

$$
q(\boldsymbol\lambda)
=-\frac12
(\boldsymbol b-A^\mathsf T\boldsymbol\lambda)^\mathsf T
Q^{-1}
(\boldsymbol b-A^\mathsf T\boldsymbol\lambda)
-\boldsymbol\lambda^\mathsf T\boldsymbol d
$$

等式乘子没有符号限制。最大化这个凹二次函数，就等价于求原问题的约束最优点；在数值上，也可以先解对偶的较小系统，再恢复 $\boldsymbol x(\boldsymbol\lambda)$。

### 和 KKT 系统对照

取上一节约束优化中的例子：

$$
Q=I,
\qquad
\boldsymbol b=(2,0)^\mathsf T,
\qquad
A=\begin{bmatrix}1&1\end{bmatrix},
\qquad
d=1
$$

二次对偶函数为

$$
q(\lambda)
=-\frac12\left((2-\lambda)^2+\lambda^2\right)-\lambda
$$

它在 $\lambda=0.5$ 处达到最大值：

$$
q(0.5)=-1.75
$$

由恢复公式

$$
\boldsymbol x(0.5)
=(2-0.5,-0.5)^\mathsf T
=(1.5,-0.5)^\mathsf T
$$

这正是 KKT 块系统给出的原解，原值和对偶值完全相等。

## 对偶上升：在乘子空间爬坡

对偶函数 $q(\boldsymbol\lambda)$ 是凹函数，即使它不可微，也可以用次梯度做最大化。若

$$
\boldsymbol x(\boldsymbol\lambda)
\in\arg\min_{\boldsymbol x}
\mathcal L(\boldsymbol x,\boldsymbol\lambda)
$$

则约束残差

$$
\boldsymbol g(\boldsymbol x(\boldsymbol\lambda))
$$

是 $q$ 关于乘子的一个次梯度。对偶上升更新为

$$
\boldsymbol\lambda_{k+1}
=\left[
\boldsymbol\lambda_k
+\eta_\lambda
\boldsymbol g(\boldsymbol x_k)
\right]_+
$$

其中 $[\cdot]_+$ 表示逐分量取最大值 $0$。如果当前点违反约束，$g_i>0$，相应乘子增大；如果约束有余量，乘子会减小但不会掉到负数。

### 一维数值迭代

对 $x\geq1$ 的例子，给定 $\lambda$ 时原变量子问题的解是 $x(\lambda)=\lambda$。取 $\eta_\lambda=0.5$、$\lambda_0=0$：

$$
\lambda_{k+1}
=\left[\lambda_k+0.5(1-\lambda_k)\right]_+
$$

前三轮为

| 步骤 | $x(\lambda_k)$ | 新乘子 $\lambda_{k+1}$ | $q(\lambda_{k+1})$ | 约束残差 $1-\lambda_{k+1}$ |
| --- | --- | --- | --- | --- |
| 1 | $0$ | $0.5$ | $0.375$ | $0.5$ |
| 2 | $0.5$ | $0.75$ | $0.46875$ | $0.25$ |
| 3 | $0.75$ | $0.875$ | $0.492188$ | $0.125$ |

乘子从下方接近 $1$，对偶值从下方接近 $0.5$。这和增广拉格朗日中的乘子更新有相同的残差反馈结构，但这里直接把对偶函数当作要最大化的目标。

## Fenchel 共轭：不必显式写约束的对偶

对凸函数 $f$，Fenchel 共轭定义为

$$
f^\star(\boldsymbol y)
=\sup_{\boldsymbol x}
\left(
\boldsymbol y^\mathsf T\boldsymbol x-f(\boldsymbol x)
\right)
$$

由定义，对任意 $\boldsymbol x,\boldsymbol y$ 都有 Fenchel–Young 不等式：

$$
f(\boldsymbol x)+f^\star(\boldsymbol y)
\geq
\boldsymbol y^\mathsf T\boldsymbol x
$$

等号成立的条件是 $\boldsymbol y\in\partial f(\boldsymbol x)$。因此共轭把「函数的斜率」作为新变量，把原函数值和线性项之间的关系反过来表达。

### 二次函数的共轭

令

$$
f(x)=\frac12x^2
$$

共轭中的上确界是

$$
\sup_x\left(yx-\frac12x^2\right)
$$

驻点满足 $x=y$，代回得到

$$
f^\star(y)=\frac12y^2
$$

例如 $y=3$ 时 $f^\star(3)=4.5$，而

$$
f(3)+f^\star(3)=4.5+4.5
\geq3\cdot3=9
$$

这里恰好取等号，因为 $y=f'(3)$。

对复合目标

$$
\min_{\boldsymbol x}
f(\boldsymbol x)+h(A\boldsymbol x)
$$

用

$$
h(\boldsymbol z)
=\sup_{\boldsymbol y}
\left(
\boldsymbol y^\mathsf T\boldsymbol z-h^\star(\boldsymbol y)
\right)
$$

替换 $h(A\boldsymbol x)$，交换在满足条件时可交换的 min 和 sup，可以得到一个典型 Fenchel 对偶：

$$
\max_{\boldsymbol y}
-f^\star(-A^\mathsf T\boldsymbol y)
-h^\star(\boldsymbol y)
$$

这条形式在稀疏正则化、最小二乘、分类损失和 ADMM 的推导中反复出现。它与拉格朗日对偶使用不同的记号，但共同动作都是把难处理的原变量关系转换成另一个空间里的上界或下界问题。

## 对偶方法的数值边界

| 现象 | 原变量视角 | 对偶变量视角 |
| --- | --- | --- |
| 约束被违反 | 可行性残差变大 | 对应乘子沿残差方向上升 |
| 约束有余量 | 还有移动空间 | 不等式乘子趋向零 |
| 原问题病态 | 子问题难解 | 对偶函数可能很平或尖 |
| 强对偶成立 | 原值和下界相等 | 对偶间隙为零 |
| 强对偶不成立 | 原解仍可能存在 | 下界与原值之间留下间隙 |

实际求解时应同时报告原目标、约束残差、对偶目标和对偶间隙。只看到对偶目标上升，不能证明原变量已经可行；只看到原目标下降，也不能证明乘子找到了正确的价格。

## 常见失效模式

**把最大化对偶函数写成最小化。** 原问题是最小化，拉格朗日对偶函数是凹函数，对偶问题应最大化。符号写反会让乘子沿着相反方向移动。

**等式乘子错误地截断为非负。** 等式可以从两边违反，乘子必须允许正负；只有不等式乘子才使用非负投影。

**把弱对偶当成强对偶。** 非凸问题、约束资格失败或最优值不达到时，对偶只能提供下界，不能直接恢复原解。

**对偶函数的下确界是负无穷。** 某些乘子使 $\mathcal L(\boldsymbol x,\boldsymbol\lambda)$ 沿某个方向无限下降，此时 $q(\boldsymbol\lambda)=-\infty$，该乘子不能作为有效的对偶候选。

**只监控一个间隙。** 原变量不满足约束时，直接把 $f(\boldsymbol x)-q(\boldsymbol\lambda)$ 当作可信间隙会误导判断；应把可行性残差单独列出。

**乘子尺度和原变量尺度不匹配。** 约束单位差异很大时，对偶上升的一个统一步长会让某些乘子过冲、另一些几乎不动。缩放约束或使用预条件更新。

**子问题只求到很粗。** 对偶次梯度依赖 $\arg\min_{\boldsymbol x}\mathcal L$。如果原变量子问题没有解好，乘子看到的是带偏残差，对偶上升可能停在错误的位置。

## 相关词条

- [优化问题](../optimization-theory/optimization-problems/)：原问题中的变量、目标和可行集。
- [约束优化](../optimization-theory/constrained-optimization/)：投影、罚函数、增广拉格朗日和 ADMM。
- [一阶最优性条件](../optimization-theory/first-order-optimality/)：乘子、KKT 和互补松弛。
- [凸集与凸函数](../optimization-theory/convex-sets-and-functions/)：强对偶和 Slater 条件所需的凸性。
- [二阶方法](../optimization-theory/second-order-methods/)：二次原问题与 KKT 线性系统的曲率处理。
- [鞍点](../optimization-theory/saddle-points/)：原变量和对偶变量共同形成的鞍点结构。
- [正交投影](../linear-algebra/orthogonal-projections/)：约束集合和拉格朗日乘子的几何联系。
- [Hessian 矩阵](../calculus/hessian/)：二次原问题、罚函数和对偶子问题中的曲率。
- [损失函数](../training-nn/loss-functions/)：机器学习中的原目标如何带上约束或正则项。
