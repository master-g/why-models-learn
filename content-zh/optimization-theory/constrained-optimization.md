---
title: "约束优化：每一步都留在可行域"
tags: ["why-models-learn"]
---

约束优化不是把梯度下降的终点再检查一遍，而是要求每一步更新都同时处理目标函数和可行域。投影方法把越界点拉回集合，罚函数把违反约束的代价加进目标，增广拉格朗日用乘子记录约束残差，障碍方法则从可行域内部逼近边界。等式约束还可以直接组成 KKT 线性系统，变量分裂后又能用 ADMM 交替处理不同部分。本文从可行域的几何定义出发，推导这些方法的更新式和数值例子，再区分它们的适用条件与失效模式。

## 先把问题写成可行域

一般约束问题写成

$$
\min_{\boldsymbol x} f(\boldsymbol x)
\quad\text{subject to}\quad
c(\boldsymbol x)=\boldsymbol0,
\qquad
\boldsymbol g(\boldsymbol x)\leq\boldsymbol0
$$

其中 $c$ 表示等式约束，$\boldsymbol g$ 表示逐分量不等式约束。可行域是

$$
\mathcal C
=\left\{
\boldsymbol x:
c(\boldsymbol x)=\boldsymbol0,\quad
\boldsymbol g(\boldsymbol x)\leq\boldsymbol0
\right\}
$$

目标函数只在 $\mathcal C$ 内比较。若无约束梯度的一步

$$
\boldsymbol z_{k+1}
=\boldsymbol x_k-\eta\nabla f(\boldsymbol x_k)
$$

跑出了 $\mathcal C$，它就不是原问题的合法候选点。约束方法的核心区别在于：它要么修改这一步，要么修改目标，或者把违反约束的部分作为另一个子问题处理。

![约束优化每步先移动，再拉回可行域](/assets/optimization-theory/svg/constrained-optimization.1.svg)

## 投影梯度：先走，再投影

当可行域 $\mathcal C$ 是闭凸集时，投影定义为离一个点最近的可行点：

$$
\Pi_{\mathcal C}(\boldsymbol z)
=\arg\min_{\boldsymbol y\in\mathcal C}
\frac12\lVert\boldsymbol y-\boldsymbol z\rVert^2
$$

投影梯度更新是

$$
\boldsymbol x_{k+1}
=\Pi_{\mathcal C}
\left(
\boldsymbol x_k-\eta\nabla f(\boldsymbol x_k)
\right)
$$

无约束时 $\Pi_{\mathcal C}$ 就是恒等映射，恢复普通梯度下降。盒约束

$$
\ell_i\leq x_i\leq u_i
$$

的投影逐坐标截断：

$$
\Pi_{\mathcal C}(z_i)
=\min\left(u_i,\max(\ell_i,z_i)\right)
$$

因此盒约束的投影便宜；一般多面体、半正定锥或组合集合的投影可能需要另一个优化子问题。

### 非负约束的数值例子

考虑

$$
f(x,y)
=\frac12\left((x-3)^2+(y+1)^2\right),
\qquad
x\geq0,\quad y\geq0
$$

无约束最小点是 $(3,-1)$，但它违反 $y\geq0$，约束最小点应落在边界 $(3,0)$。从 $(x_0,y_0)=(0,0)$ 出发，取 $\eta=0.5$：

第一步梯度为

$$
\nabla f(0,0)=(-3,1)
$$

原始梯度步为

$$
(0,0)-0.5(-3,1)=(1.5,-0.5)
$$

把第二坐标投影到非负半轴后

$$
(x_1,y_1)=(1.5,0)
$$

第二步的梯度为 $(-1.5,1)$，所以

$$
(x_1,y_1)-0.5(-1.5,1)
=(2.25,-0.5)
\longrightarrow
(x_2,y_2)=(2.25,0)
$$

对应函数值为

$$
f(x_0,y_0)=5,
\qquad
f(x_1,y_1)=1.625,
\qquad
f(x_2,y_2)=0.78125
$$

投影没有把 $y$ 方向的梯度抹掉；它只把越过边界的结果截回 $y=0$。在边界上继续更新时，只有指向可行域内部的方向可以真正移动。

### 仿射集合的正交投影

对满行秩等式集合

$$
\mathcal A=\{\boldsymbol x:A\boldsymbol x=\boldsymbol d\}
$$

若 $AA^\mathsf T$ 可逆，点 $\boldsymbol z$ 到 $\mathcal A$ 的正交投影为

$$
\Pi_{\mathcal A}(\boldsymbol z)
=\boldsymbol z
-A^\mathsf T(AA^\mathsf T)^{-1}
(A\boldsymbol z-\boldsymbol d)
$$

例如 $A=[1\ \ 1]$、$\boldsymbol d=2$、$\boldsymbol z=(3,0)^\mathsf T$：

$$
A\boldsymbol z-\boldsymbol d=1,
\qquad
(AA^\mathsf T)^{-1}=\frac12
$$

所以

$$
\Pi_{\mathcal A}(\boldsymbol z)
=(3,0)^\mathsf T
-\begin{bmatrix}1\\1\end{bmatrix}\frac12
=(2.5,-0.5)^\mathsf T
$$

两坐标之和正好回到 $2$。这个公式是等式约束投影的几何版本；它也会在 KKT 线性系统和分裂方法中出现。

## 等式约束二次问题与 KKT 系统

对等式约束二次问题

$$
\min_{\boldsymbol x}
\frac12\boldsymbol x^\mathsf TQ\boldsymbol x
-\boldsymbol b^\mathsf T\boldsymbol x
\quad\text{subject to}\quad
A\boldsymbol x=\boldsymbol d
$$

构造拉格朗日函数

$$
\mathcal L(\boldsymbol x,\boldsymbol\lambda)
=\frac12\boldsymbol x^\mathsf TQ\boldsymbol x
-\boldsymbol b^\mathsf T\boldsymbol x
+\boldsymbol\lambda^\mathsf T(A\boldsymbol x-\boldsymbol d)
$$

对 $\boldsymbol x$ 和 $\boldsymbol\lambda$ 分别求导：

$$
Q\boldsymbol x-\boldsymbol b+A^\mathsf T\boldsymbol\lambda=\boldsymbol0,
\qquad
A\boldsymbol x=\boldsymbol d
$$

合并为块线性系统：

$$
\begin{bmatrix}
Q&A^\mathsf T\\
A&0
\end{bmatrix}
\begin{bmatrix}
\boldsymbol x\\
\boldsymbol\lambda
\end{bmatrix}
=
\begin{bmatrix}
\boldsymbol b\\
\boldsymbol d
\end{bmatrix}
$$

这个矩阵通常是不定的，即使 $Q$ 正定，右下角的零块也会破坏直接 Cholesky 的形式。它求的是同时满足驻点和可行性的候选解，不应把乘子块当成额外的参数坐标。

### 二维数值例子

取

$$
Q=I,
\qquad
\boldsymbol b=(2,0)^\mathsf T,
\qquad
A=\begin{bmatrix}1&1\end{bmatrix},
\qquad
d=1
$$

方程为

$$
x_1+\lambda=2,
\qquad
x_2+\lambda=0,
\qquad
x_1+x_2=1
$$

由前两式

$$
x_1=2-\lambda,
\qquad
x_2=-\lambda
$$

代入约束得到

$$
2-2\lambda=1,
\qquad
\lambda=0.5
$$

因此

$$
\boldsymbol x^\star=(1.5,-0.5)^\mathsf T
$$

约束确实满足 $1.5-0.5=1$，目标值为

$$
\frac12(1.5^2+(-0.5)^2)-2(1.5)
=-1.75
$$

无约束解是 $(2,0)$，等式把它沿法向量方向移到直线 $x_1+x_2=1$ 上。

## 罚函数：用目标代价换取可行性

等式约束的二次罚函数是

$$
F_\rho(\boldsymbol x)
=f(\boldsymbol x)
+\frac{\rho}{2}\lVert c(\boldsymbol x)\rVert^2
$$

不等式约束可以惩罚正部：

$$
F_\rho(\boldsymbol x)
=f(\boldsymbol x)
+\frac{\rho}{2}
\sum_i\max(0,g_i(\boldsymbol x))^2
$$

$\rho$ 越大，违反约束越贵；但有限 $\rho$ 通常只给出近似可行点。让 $\rho$ 趋于无穷可以逼近约束，却会把无约束子问题变得病态。

### 同一个二次问题

仍取

$$
f(x_1,x_2)
=\frac12\left((x_1-2)^2+x_2^2\right),
\qquad
c(x_1,x_2)=x_1+x_2-1
$$

对于给定 $\rho$，罚函数的一阶条件可以直接解出

$$
s_\rho=x_1+x_2-1=\frac1{1+2\rho}
$$

以及

$$
x_1=2-\frac{\rho}{1+2\rho},
\qquad
x_2=-\frac{\rho}{1+2\rho}
$$

数值上：

| $\rho$ | 解 $(x_1,x_2)$ | 约束残差 $c(x)$ | 罚函数值 |
| --- | --- | --- | --- |
| $1$ | $(1.6667,-0.3333)$ | $0.3333$ | $0.166667$ |
| $9$ | $(1.5263,-0.4737)$ | $0.05263$ | $0.236842$ |

残差变小了，但罚函数值不必单调变小，因为目标函数和约束代价在交换。这个例子中罚函数 Hessian 为

$$
I+\rho
\begin{bmatrix}
1&1\\
1&1
\end{bmatrix}
$$

它的特征值是 $1$ 和 $1+2\rho$。$\rho=9$ 时两者相差 $19$ 倍；在更高维、更大 $\rho$ 时，子问题会越来越难解。

## 增广拉格朗日：让乘子记录残差

增广拉格朗日同时保留乘子项和二次罚项：

$$
\mathcal L_\rho(\boldsymbol x,\boldsymbol\lambda)
=f(\boldsymbol x)
+\boldsymbol\lambda^\mathsf Tc(\boldsymbol x)
+\frac{\rho}{2}\lVert c(\boldsymbol x)\rVert^2
$$

典型迭代分两步：

$$
\boldsymbol x_{k+1}
\approx
\arg\min_{\boldsymbol x}
\mathcal L_\rho(\boldsymbol x,\boldsymbol\lambda_k)
$$

$$
\boldsymbol\lambda_{k+1}
=\boldsymbol\lambda_k
+\rho c(\boldsymbol x_{k+1})
$$

罚函数把违反约束的代价永远压进目标，增广拉格朗日则让 $\boldsymbol\lambda_k$ 累积残差的方向，通常不需要把 $\rho$ 推到极大。

### 残差如何衰减

对上面的二次问题取 $\rho=1$、$\lambda_0=0$，每次精确解 $\boldsymbol x$ 子问题。令

$$
s_k=x_{1,k}+x_{2,k}
$$

可得到

$$
s_{k+1}=\frac{4-2\lambda_k}{3},
\qquad
\lambda_{k+1}=\lambda_k+s_{k+1}-1
$$

前三步的残差是

| 步骤 | $s_k$ | $s_k-1$ | $\lambda_k$ |
| --- | --- | --- | --- |
| 1 | $1.333333$ | $0.333333$ | $0.333333$ |
| 2 | $1.111111$ | $0.111111$ | $0.444444$ |
| 3 | $1.037037$ | $0.037037$ | $0.481481$ |

约束残差依次约为 $1/3$、$1/9$、$1/27$。乘子逐渐接近等式约束的最优乘子 $0.5$，而不是靠把 $\rho$ 一次调成很大的数来强行修正。

## 障碍方法：从可行域内部接近边界

对于不等式 $g_i(\boldsymbol x)\leq0$，对数障碍在严格可行域内定义：

$$
\phi_\mu(\boldsymbol x)
=f(\boldsymbol x)
-\mu\sum_i\log\left(-g_i(\boldsymbol x)\right),
\qquad
\mu>0
$$

如果某个 $g_i(\boldsymbol x)$ 接近 $0$，负对数项会变大，把迭代点推回内部。逐渐减小 $\mu$，中心路径会向原问题的边界最优点靠近。障碍方法要求初始点严格可行；越界点不能直接代入对数。

### 一维数值例子

考虑

$$
\min_{x>0}\frac12(x-2)^2
$$

加入障碍后

$$
\phi_\mu(x)
=\frac12(x-2)^2-\mu\log x
$$

一阶条件为

$$
x-2-\frac{\mu}{x}=0
$$

乘以 $x$：

$$
x^2-2x-\mu=0
$$

取正根：

$$
x_\mu=1+\sqrt{1+\mu}
$$

当 $\mu=1$ 时

$$
x_\mu=1+\sqrt2\approx2.4142
$$

它在边界 $x=0$ 的另一侧不会出现；当 $\mu$ 逐渐趋近 $0$ 时，$x_\mu$ 趋近原问题的最优点 $2$。障碍参数太大时解会偏离真实边界解，太小时内部线性系统又可能变得病态。

## ADMM：把一个约束拆成两个子问题

当目标可以分成两个部分，考虑

$$
\min_{\boldsymbol x,\boldsymbol z}
f(\boldsymbol x)+h(\boldsymbol z)
\quad\text{subject to}\quad
A\boldsymbol x+B\boldsymbol z=\boldsymbol c
$$

ADMM 使用缩放乘子 $\boldsymbol u$，交替更新：

$$
\boldsymbol x_{k+1}
=\arg\min_{\boldsymbol x}
\left(
f(\boldsymbol x)
+\frac{\rho}{2}
\lVert A\boldsymbol x+B\boldsymbol z_k-\boldsymbol c+\boldsymbol u_k\rVert^2
\right)
$$

$$
\boldsymbol z_{k+1}
=\arg\min_{\boldsymbol z}
\left(
h(\boldsymbol z)
+\frac{\rho}{2}
\lVert A\boldsymbol x_{k+1}+B\boldsymbol z-\boldsymbol c+\boldsymbol u_k\rVert^2
\right)
$$

$$
\boldsymbol u_{k+1}
=\boldsymbol u_k
+A\boldsymbol x_{k+1}
+B\boldsymbol z_{k+1}
-\boldsymbol c
$$

$\boldsymbol x$ 和 $\boldsymbol z$ 的子问题可以使用不同的专门求解器。代价是每轮要解两个子问题，且原始残差与对偶残差需要共同监控。

### 一维一致性例子

取

$$
\min_{x,z}
\frac12(x-2)^2+\frac12(z+1)^2
\quad\text{subject to}\quad
x=z
$$

约束把两个变量合成一个，真实解为

$$
x^\star=z^\star=0.5
$$

取 $\rho=1$、$z_0=0$、$u_0=0$，更新式化为

$$
x_{k+1}=\frac{2+z_k-u_k}{2},
\qquad
z_{k+1}=\frac{-1+x_{k+1}+u_k}{2},
\qquad
u_{k+1}=u_k+x_{k+1}-z_{k+1}
$$

前三轮为

| 步骤 | $x_k$ | $z_k$ | $u_k$ | 原始残差 $x_k-z_k$ |
| --- | --- | --- | --- | --- |
| 1 | $1$ | $0$ | $1$ | $1$ |
| 2 | $0.5$ | $0.25$ | $1.25$ | $0.25$ |
| 3 | $0.5$ | $0.375$ | $1.375$ | $0.125$ |

$x$ 和 $z$ 逐渐一致，目标共同的最优值为 $0.5$。ADMM 的实际收敛速度还取决于 $\rho$ 和两个子问题的条件数；它不是把任意约束问题自动变成三行闭式公式。

## 不等式约束的活动集合

在解附近，有些不等式严格满足，有些不等式恰好卡在边界。令

$$
\mathcal A(\boldsymbol x)
=\{i:g_i(\boldsymbol x)=0\}
$$

表示活动约束。活动集方法先猜哪些约束会成为等式，把它们和原目标组成等式约束子问题；如果新解违反了未活动约束，加入该约束；如果某个活动约束的乘子符号不对，移除它。

这个过程的几何含义是：在边界上只沿切向方向移动，法向分量由乘子抵消。它需要判断和更新活动集合，适合约束数量可控、解具有稳定活动模式的问题。

投影、罚函数、障碍和活动集对应不同的可行性策略：

| 方法 | 迭代点是否保持可行 | 约束如何进入更新 | 主要代价 |
| --- | --- | --- | --- |
| 投影梯度 | 对凸集合通常是 | 每步计算投影 | 投影可能昂贵 |
| 罚函数 | 通常否 | 违反约束增加目标代价 | 大 $\rho$ 导致病态 |
| 增广拉格朗日 | 逐步趋于 | 乘子加二次残差 | 需要解子问题并更新乘子 |
| 障碍方法 | 严格保持内部可行 | 对边界加无限大代价 | 需要严格可行初始点 |
| 活动集 | 取决于当前猜测 | 把活动不等式当等式 | 活动集合可能反复变化 |

## 约束方法的失效模式

**可行域为空。** 如果等式和不等式互相矛盾，投影不存在，罚函数也只能不断增加残差。先做可行性检查，不能把优化器的失败都归因于学习率。

**投影不是一个便宜的函数。** 对盒约束投影只是截断，对一般凸集投影本身可能需要迭代求解；对非凸集合，最近点甚至可能不唯一。

**罚参数只会越来越大。** 增大 $\rho$ 确实通常会减小约束残差，但也会放大 Hessian 的条件数。应监控残差和子问题的数值稳定性，而不是只追求更大的 $\rho$。

**障碍点越过边界。** 对数只在严格可行域内定义。线搜索也必须检查候选点仍然满足 $g_i(\boldsymbol x)<0$，否则目标函数已经没有定义。

**乘子符号或更新方向写反。** 增广拉格朗日和 ADMM 对残差符号敏感。交换 $c-Ax$ 与 $Ax-c$ 时，乘子更新必须一起改变；只改一处会让残差发散。

**KKT 只是候选条件。** 非凸问题的 KKT 点可能是局部最大、鞍点或不满足全局最优的驻点。还要检查二阶性质、可行邻域和目标值。

**子问题没有解准。** 罚函数、增广拉格朗日和 ADMM 都把原问题换成一串子问题。内层过早停止会让外层看到的梯度和残差不可靠，内层过度求精又会浪费计算。

## 相关词条

- [优化问题](../optimization-theory/optimization-problems/)：决策变量、目标函数和可行集的统一定义。
- [一阶最优性条件](../optimization-theory/first-order-optimality/)：KKT、互补松弛和乘子条件。
- [二阶最优性条件](../optimization-theory/second-order-optimality/)：约束切空间和临界锥上的曲率。
- [二阶方法](../optimization-theory/second-order-methods/)：Newton、阻尼和信赖域如何求局部模型步。
- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：投影梯度所继承的一阶下降分析。
- [凸集与凸函数](../optimization-theory/convex-sets-and-functions/)：投影唯一性和凸问题全局性质的基础。
- [正交投影](../linear-algebra/orthogonal-projections/)：等式集合投影的线性代数公式。
- [Hessian 矩阵](../calculus/hessian/)：罚函数、增广项和二阶子问题中的曲率。
- [对偶性](../optimization-theory/duality/)：乘子、对偶目标和约束问题的另一种视角。
- [鞍点](../optimization-theory/saddle-points/)：约束驻点和非凸二次模型中的另一类候选点。
