---
title: "一阶最优性条件：没有可行下降方向"
tags: ["why-models-learn"]
---

**一阶最优性条件**把「当前位置不能再下降」写成梯度、可行方向或拉格朗日乘子之间的关系。无约束内部最优点满足梯度为零；约束边界上的最优点则只要求所有可行方向都不能下降，梯度可以指向不可行区域。对不等式约束，这个关系展开为 KKT 条件；对凸问题，满足这些条件的点还是全局最小点。本篇从方向导数开始，逐步处理等式约束、不等式约束、约束资格条件、凸充分性和不可微的次梯度。

## 无约束时，梯度为零只是第一道门

考虑

$$
\min_{\boldsymbol x\in\mathbb R^d}f(\boldsymbol x)
$$

如果 $f$ 在内部点 $\boldsymbol x^\star$ 可微，并且 $\boldsymbol x^\star$ 是局部最小点，那么任取方向 $\boldsymbol v$，一维函数

$$
\phi(t)
=f(\boldsymbol x^\star+t\boldsymbol v)
$$

都在 $t=0$ 处有局部最小。因此

$$
\phi'(0)
=\nabla f(\boldsymbol x^\star)^\mathsf T\boldsymbol v
=0
\qquad
\text{for all }\boldsymbol v
$$

只有零向量能与所有方向正交，于是

$$
\nabla f(\boldsymbol x^\star)=\boldsymbol0
$$

这说明梯度为零是内部局部最小的必要条件。它不是充分条件：$f(x)=x^3$ 在 $0$ 处导数为零，却既不是局部最小也不是局部最大；$f(x,y)=x^2-y^2$ 的原点是鞍点。

### 一个一维例子

令

$$
f(x)=\frac12(x-3)^2+2
$$

则

$$
f'(x)=x-3
$$

一阶条件给出 $x^\star=3$，代回得到

$$
f(3)=2
$$

因为这个函数的二阶导数为 $1>0$，这个驻点是严格全局最小点。如果只保留 $f'(x^\star)=0$ 而不检查曲率或凸性，结论还没有完成。

对可微凸函数，无约束的一阶条件会变成充分条件。若

$$
\nabla f(\boldsymbol x^\star)=\boldsymbol0
$$

且 $f$ 凸，则支撑超平面不等式给出

$$
f(\boldsymbol y)
\geq
f(\boldsymbol x^\star)
+\nabla f(\boldsymbol x^\star)^\mathsf T
(\boldsymbol y-\boldsymbol x^\star)
=f(\boldsymbol x^\star)
$$

对每个 $\boldsymbol y$ 成立，所以 $\boldsymbol x^\star$ 是全局最小点。非凸问题没有这个升级。

## 用方向导数描述约束边界

设 $\mathcal D$ 是可行集，$\boldsymbol x\in\mathcal D$，向量 $\boldsymbol d$ 是从该点出发的可行方向，如果存在某个 $\delta>0$，使

$$
\boldsymbol x+t\boldsymbol d\in\mathcal D
\qquad
\text{for all }0\leq t\leq\delta
$$

方向导数是

$$
D f(\boldsymbol x;\boldsymbol d)
=\lim_{t\to0^+}
\frac{f(\boldsymbol x+t\boldsymbol d)-f(\boldsymbol x)}{t}
$$

当 $f$ 可微时：

$$
D f(\boldsymbol x;\boldsymbol d)
=\nabla f(\boldsymbol x)^\mathsf T\boldsymbol d
$$

如果 $\boldsymbol x^\star$ 是局部最小点，任何可行方向都不能在一阶上下降：

$$
D f(\boldsymbol x^\star;\boldsymbol d)\geq0
\qquad
\text{for every feasible direction }\boldsymbol d
$$

若某个可行方向的方向导数严格小于零，就可以取足够小的 $t$ 让目标下降，当前位置不可能是局部最小。

### 一维边界例子

考虑

$$
\min_{x\in[0,2]}f(x),
\qquad
f(x)=x
$$

解是左端点 $x^\star=0$，但

$$
f'(0)=1\neq0
$$

从 $0$ 出发的可行方向只能满足 $d\geq0$，所以

$$
D f(0;d)=d\geq0
$$

没有可行下降方向。向左走会下降，但向左不属于这个问题的可行移动。边界最优点的正确一阶描述不是「梯度必须为零」，而是「梯度与每个可行方向的内积都不为负」。

## 等式约束：梯度落在约束法向空间

先考虑等式约束：

$$
\min_{\boldsymbol x}f(\boldsymbol x)
\qquad
\text{subject to}\qquad
\boldsymbol h(\boldsymbol x)=\boldsymbol0
$$

其中 $\boldsymbol h:\mathbb R^d\to\mathbb R^m$ 可微。若在 $\boldsymbol x^\star$ 附近移动 $\boldsymbol d$ 仍留在约束曲面上，一阶展开要求

$$
\boldsymbol h(\boldsymbol x^\star+t\boldsymbol d)
=\boldsymbol h(\boldsymbol x^\star)
+tJ_{\boldsymbol h}(\boldsymbol x^\star)\boldsymbol d
+o(t)
$$

所以可行方向满足

$$
J_{\boldsymbol h}(\boldsymbol x^\star)\boldsymbol d
=\boldsymbol0
$$

局部最优要求 $\nabla f(\boldsymbol x^\star)^\mathsf T\boldsymbol d=0$ 或至少不能为负，所有这样的切向方向都不能提供下降。在线性独立等正则条件下，这等价于存在乘子 $\boldsymbol\nu$：

$$
\nabla f(\boldsymbol x^\star)
+J_{\boldsymbol h}(\boldsymbol x^\star)^\mathsf T
\boldsymbol\nu
=\boldsymbol0
$$

这就是等式约束的一阶拉格朗日条件。它说目标梯度被约束面的法向量抵消；沿着约束面本身的切向方向，目标没有一阶下降。

### 拉格朗日函数

定义

$$
\mathcal L(\boldsymbol x,\boldsymbol\nu)
=f(\boldsymbol x)
+\boldsymbol\nu^\mathsf T\boldsymbol h(\boldsymbol x)
$$

对 $\boldsymbol x$ 求梯度：

$$
\nabla_{\boldsymbol x}\mathcal L
=\nabla f(\boldsymbol x)
+J_{\boldsymbol h}(\boldsymbol x)^\mathsf T\boldsymbol\nu
$$

等式约束的一阶方程就是

$$
\nabla_{\boldsymbol x}\mathcal L(\boldsymbol x^\star,\boldsymbol\nu^\star)
=\boldsymbol0,
\qquad
\boldsymbol h(\boldsymbol x^\star)=\boldsymbol0
$$

乘子不是新增的模型参数，也不是要被数据学习的权重；它是把约束法向方向的梯度分量记账的辅助变量。

### 数字例子：投影到一条直线

考虑把原点到直线 $x_1+x_2=1$ 的距离平方最小化：

$$
\min_{x_1,x_2}x_1^2+x_2^2
\qquad
\text{subject to}\qquad
x_1+x_2-1=0
$$

拉格朗日函数为

$$
\mathcal L
=x_1^2+x_2^2
+\nu(x_1+x_2-1)
$$

一阶方程：

$$
\frac{\partial\mathcal L}{\partial x_1}
=2x_1+\nu=0,
\qquad
\frac{\partial\mathcal L}{\partial x_2}
=2x_2+\nu=0
$$

两式相减得到 $x_1=x_2$，代回约束：

$$
\boldsymbol x^\star
=\left(\frac12,\frac12\right),
\qquad
\nu^\star=-1
$$

目标值是

$$
f(\boldsymbol x^\star)
=\frac14+\frac14
=\frac12
$$

几何上，最近点到原点的连线 $(1/2,1/2)$ 垂直于直线 $x_1+x_2=1$；代数上的乘子方程只是把这个垂直关系写成梯度与法向量平行。

## 不等式约束：KKT 把边界情况列全

把问题写成

$$
\begin{aligned}
\min_{\boldsymbol x}\quad& f(\boldsymbol x)\\
\text{subject to}\quad&
g_i(\boldsymbol x)\leq0,\quad i=1,\ldots,m\\
&h_j(\boldsymbol x)=0,\quad j=1,\ldots,r
\end{aligned}
$$

采用这个符号约定，拉格朗日函数是

$$
\mathcal L(\boldsymbol x,\boldsymbol\lambda,\boldsymbol\nu)
=f(\boldsymbol x)
+\sum_{i=1}^m\lambda_i g_i(\boldsymbol x)
+\sum_{j=1}^r\nu_j h_j(\boldsymbol x)
$$

KKT 条件包含四组关系。

### 原始可行性

候选点必须满足原问题的约束：

$$
g_i(\boldsymbol x^\star)\leq0,
\qquad
h_j(\boldsymbol x^\star)=0
$$

不满足约束的低目标点不是这个问题的候选解。

### 对偶可行性

不等式约束的乘子必须非负：

$$
\lambda_i^\star\geq0
$$

等式乘子 $\nu_j$ 没有正负限制，因为等式两侧都可以阻止移动。

### 一阶驻点

对原变量的梯度为零：

$$
\nabla f(\boldsymbol x^\star)
+\sum_{i=1}^m\lambda_i^\star
\nabla g_i(\boldsymbol x^\star)
+\sum_{j=1}^r\nu_j^\star
\nabla h_j(\boldsymbol x^\star)
=\boldsymbol0
$$

这表示目标梯度可以由活动约束的法向量抵消。

### 互补松弛

每个不等式约束都满足

$$
\lambda_i^\star g_i(\boldsymbol x^\star)=0
$$

如果约束不活动，即 $g_i(\boldsymbol x^\star)<0$，则必须有 $\lambda_i^\star=0$。如果乘子为正，则约束一定活动，即 $g_i(\boldsymbol x^\star)=0$。二者不能同时在一阶账本里产生非零贡献。

把四组条件合在一起：

$$
\begin{cases}
g_i(\boldsymbol x^\star)\leq0,\quad h_j(\boldsymbol x^\star)=0\\
\lambda_i^\star\geq0\\
\nabla_{\boldsymbol x}\mathcal L(\boldsymbol x^\star,\boldsymbol\lambda^\star,\boldsymbol\nu^\star)=\boldsymbol0\\
\lambda_i^\star g_i(\boldsymbol x^\star)=0
\end{cases}
$$

### 数字例子：上界约束激活

考虑

$$
\min_x(x-3)^2
\qquad
\text{subject to}\qquad
x\leq2
$$

写成 $g(x)=x-2\leq0$，拉格朗日函数为

$$
\mathcal L(x,\lambda)
=(x-3)^2+\lambda(x-2)
$$

无约束最小点 $x=3$ 不可行，所以最优点在边界 $x^\star=2$。驻点条件：

$$
\frac{\partial\mathcal L}{\partial x}
=2(x-3)+\lambda=0
$$

代入 $x=2$：

$$
-2+\lambda^\star=0,
\qquad
\lambda^\star=2
$$

此时

$$
g(2)=0,
\qquad
\lambda^\star g(2)=0,
\qquad
f(2)=1
$$

所有 KKT 条件都成立。梯度 $f'(2)=-2$ 并不为零，但它被正的约束法向乘子 $2\nabla g=2$ 恰好抵消。

## 约束资格条件不是装饰

从局部最优推出乘子方程，需要约束表面在该点具有足够正常的几何结构。常见的线性独立约束资格条件要求活动不等式约束的梯度和等式约束的梯度线性独立。它排除多个约束在同一点重复描述同一个法向方向的退化情况。

一个简单的退化例子是

$$
\min_x x
\qquad
\text{subject to}\qquad
x^2\leq0
$$

可行集只有 $x=0$，所以 $x=0$ 是全局最小点。但在 $0$ 处

$$
g(0)=0,
\qquad
g'(0)=0
$$

KKT 驻点条件会变成

$$
1+\lambda g'(0)=1=0
$$

无论 $\lambda\geq0$ 取什么值都不可能成立。最优点确实存在，失败的是把约束边界用普通非退化法向乘子表示的资格条件。KKT 必要性不是只看「有一个不等式」就自动成立。

## 什么时候 KKT 还是全局充分条件

对一般非凸问题，KKT 点通常只是候选点，不能凭它排除局部最大或鞍点。凸问题中可以得到更强结论：假设

- $f$ 是凸函数；
- 每个 $g_i$ 是凸函数；
- 每个 $h_j$ 是仿射函数；
- $\boldsymbol x^\star,\boldsymbol\lambda^\star,\boldsymbol\nu^\star$ 满足 KKT。

对任意可行点 $\boldsymbol y$，由于 $\lambda_i^\star\geq0$ 且 $g_i(\boldsymbol y)\leq0$：

$$
\mathcal L(\boldsymbol y,\boldsymbol\lambda^\star,\boldsymbol\nu^\star)
\leq f(\boldsymbol y)
$$

拉格朗日函数关于 $\boldsymbol x$ 仍是凸的，而 KKT 驻点条件给出

$$
\mathcal L(\boldsymbol y,\boldsymbol\lambda^\star,\boldsymbol\nu^\star)
\geq
\mathcal L(\boldsymbol x^\star,\boldsymbol\lambda^\star,\boldsymbol\nu^\star)
$$

互补松弛和等式可行性又给出

$$
\mathcal L(\boldsymbol x^\star,\boldsymbol\lambda^\star,\boldsymbol\nu^\star)
=f(\boldsymbol x^\star)
$$

合并三式：

$$
f(\boldsymbol y)
\geq
\mathcal L(\boldsymbol y,\boldsymbol\lambda^\star,\boldsymbol\nu^\star)
\geq
f(\boldsymbol x^\star)
$$

所以 $\boldsymbol x^\star$ 是全局最小点。这里要分清方向：一般光滑非凸问题中，KKT 常是必要条件；凸问题中，KKT 条件可以成为充分条件。约束资格条件主要服务于「最优点是否一定能产生乘子」这一必要性问题，不是每次使用凸充分性都要额外假设同一个条件。

## 不可微时，次梯度替代普通梯度

凸函数在尖点没有唯一梯度，但仍然可以用支撑下界。向量 $\boldsymbol s$ 是 $f$ 在 $\boldsymbol x$ 处的次梯度，如果

$$
f(\boldsymbol y)
\geq
f(\boldsymbol x)
+\boldsymbol s^\mathsf T(\boldsymbol y-\boldsymbol x)
\qquad
\text{for all }\boldsymbol y
$$

所有次梯度组成次微分集合 $\partial f(\boldsymbol x)$。对凸函数：

$$
\boldsymbol x^\star
\text{ 是全局最小点}
\quad\Longleftrightarrow\quad
\boldsymbol0\in\partial f(\boldsymbol x^\star)
$$

### 绝对值在原点的次梯度

令

$$
f(x)=|x|
$$

在 $x=0$，任意 $s\in[-1,1]$ 都满足

$$
|y|
\geq sy
\qquad
\text{for all }y
$$

因此

$$
\partial|x|\big|_{x=0}=[-1,1]
$$

其中包含 $0$，所以原点是全局最小点。普通导数在原点不存在，但一阶最优性仍然可以写成集合包含关系。选择 $s=0$、$s=1$ 或其他区间内的值是算法在尖点的次梯度约定，不是把经典导数变成了唯一数值。

有凸可行集 $\mathcal D$ 时，约束版本可以用法锥写成

$$
N_{\mathcal D}(\boldsymbol x)
=\left\{\boldsymbol v:
\boldsymbol v^\mathsf T(\boldsymbol y-\boldsymbol x)\leq0,
\ \text{for all }\boldsymbol y\in\mathcal D\right\}
$$

对于凸可微目标：

$$
\boldsymbol x^\star
\text{ 是全局最小点}
\quad\Longleftrightarrow\quad
\boldsymbol0\in
\nabla f(\boldsymbol x^\star)
+N_{\mathcal D}(\boldsymbol x^\star)
$$

无约束时法锥只有零向量，退化回 $\nabla f=\boldsymbol0$。上界集合 $\mathcal D=(-\infty,2]$ 在 $x=2$ 处的法锥是

$$
N_{\mathcal D}(2)=[0,\infty)
$$

边界例子中 $\nabla f(2)=-2$，因此 $-\nabla f(2)=2$ 落在这个法锥里，正好表达「下降方向指向不可行侧」。

## 乘子还记录约束的边际代价

拉格朗日乘子不仅是求方程的记号，也可以表示放松约束时最优值的一阶变化。继续看

$$
p(u)
=\min_x(x-3)^2
\qquad
\text{subject to}\qquad
x\leq u
$$

在 $u=2$ 附近，约束仍然激活，最优点是 $x=u$，因此

$$
p(u)=(u-3)^2
$$

直接求导：

$$
p'(2)=2(2-3)=-2
$$

采用 $g(x,u)=x-u\leq0$ 的符号，$u=2$ 时的 KKT 乘子为 $\lambda^\star=2$，所以

$$
p'(2)=-\lambda^\star
$$

乘子为正，表示把上界放松一小点会降低最小目标；负号来自约束写成 $x-u\leq0$。这个灵敏度解释依赖最优解和约束结构在附近稳定，不能把任意训练过程中的梯度都叫作「约束价格」。

![一阶条件表示没有可行下降方向](/assets/optimization-theory/svg/first-order-optimality.1.svg)

## 失效模式

**把梯度为零当成充分条件。** 对非凸函数，驻点可能是鞍点或局部最大；凸性或额外二阶信息才提供升级。

**忽略可行方向。** 约束边界可以有非零梯度；要检查的是所有可行方向的一阶变化。

**把乘子符号写反。** 采用 $g_i\leq0$ 时不等式乘子要求 $\lambda_i\geq0$；若改写为 $g_i\geq0$，符号约定也要一起改。

**漏写互补松弛。** 不活动的约束不能凭空产生正乘子；$\lambda_i g_i=0$ 是区分活动约束和松约束的条件。

**把 KKT 当成无条件定理。** 必要性需要约束资格条件或其他正则假设；退化约束可能有最优点却不存在 KKT 乘子。

**把 KKT 点当成全局解。** 非凸问题的 KKT 条件通常只是候选条件；凸目标、凸不等式和仿射等式才给出常用的全局充分性。

**把等式乘子也限制为非负。** 等式可以从两侧阻止移动，乘子没有非负限制；只有 $g_i\leq0$ 的不等式乘子需要 $\lambda_i\geq0$。

**把不可微点硬套普通梯度。** $|x|$、max 和 ReLU 的尖点需要次梯度、方向导数或明确的实现约定。

**把乘子灵敏度解释成训练梯度。** 乘子描述约束右侧变化下的最优值局部变化，前提是相应的正则性和稳定性成立。

**把一阶条件当成存在性证明。** 解可能不存在、下确界可能不被达到；一阶方程只能分析已经满足相应可微和约束条件的候选点。

## 相关词条

- [优化问题](../optimization-theory/optimization-problems/)：明确决策变量、目标函数和可行集。
- [局部与全局最小值](../optimization-theory/local-and-global-minima/)：区分局部比较、全局比较、驻点和鞍点。
- [凸集与凸函数](../optimization-theory/convex-sets-and-functions/)：说明凸性如何把一阶候选条件升级为全局结论。
- [梯度](../calculus/gradient/)：解释方向导数和最陡上升方向。
- [Hessian 矩阵](../calculus/hessian/)：处理一阶条件之后的二阶曲率判据。
- [二阶最优性条件](../optimization-theory/second-order-optimality/)：继续处理正定、退化和约束二阶条件。
- [约束优化](../optimization-theory/constrained-optimization/)：系统展开可行域、乘子和约束算法。
- [对偶性](../optimization-theory/duality/)：从拉格朗日函数继续推导原问题与对偶问题的关系。
- [子空间](../linear-algebra/subspaces/)：理解等式约束切空间与法向空间的线性代数背景。
- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：分析一阶更新、步长和收敛保证。
