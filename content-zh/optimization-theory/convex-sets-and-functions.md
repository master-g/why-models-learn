---
title: "凸集与凸函数：把局部比较升级为全局比较"
tags: ["why-models-learn"]
---

**凸集**是包含任意两点连线段的集合，**凸函数**则是在定义域为凸集时，线段中间的函数值不高于两端函数值的线性插值。两种凸性分别约束「允许走到哪里」和「目标值沿着直线怎么变化」。对优化问题来说，这个区分很重要：凸目标在凸可行域上的局部最小值就是全局最小值，而平方损失、逻辑回归损失等常见目标的凸性也因此可以被严格分析。本篇从线段定义开始，推导凸包、Jensen 不等式、上图集、Hessian 判据和一阶支撑超平面，再说明这些结论在哪些机器学习参数化中会失效。

## 凸集先规定允许怎么走

设 $\mathcal C\subseteq\mathbb R^d$。取 $\boldsymbol x,\boldsymbol y\in\mathcal C$，以及 $0\leq\lambda\leq1$。两点的凸组合是

$$
\boldsymbol z_\lambda
=\lambda\boldsymbol x+(1-\lambda)\boldsymbol y
$$

如果所有这样的 $\boldsymbol z_\lambda$ 都仍在 $\mathcal C$ 中，就称 $\mathcal C$ 是凸集：

$$
\forall\boldsymbol x,\boldsymbol y\in\mathcal C,\quad
\forall\lambda\in[0,1],\quad
\lambda\boldsymbol x+(1-\lambda)\boldsymbol y\in\mathcal C
$$

$\lambda=1$ 时取到 $\boldsymbol x$，$\lambda=0$ 时取到 $\boldsymbol y$，中间的 $\lambda$ 扫过整条线段。特别地，$\lambda=1/2$ 是中点：

$$
\boldsymbol m=\frac{\boldsymbol x+\boldsymbol y}{2}
$$

所以检查中点是检查凸性的一个直观入口，但只检查一对点的中点远远不够。定义要求集合中的每一对点和每一个权重都通过检查。

### 一个单位圆盘的数字检查

考虑二维单位圆盘

$$
\mathcal D
=\left\{\boldsymbol u\in\mathbb R^2:
\|\boldsymbol u\|_2\leq1\right\}
$$

取边界上的两点

$$
\boldsymbol x=(1,0),
\qquad
\boldsymbol y=(0,1)
$$

中点是

$$
\boldsymbol m
=\frac{\boldsymbol x+\boldsymbol y}{2}
=\left(\frac12,\frac12\right)
$$

它的平方范数为

$$
\|\boldsymbol m\|_2^2
=\left(\frac12\right)^2+\left(\frac12\right)^2
=\frac12\leq1
$$

因此这个中点仍在圆盘里。更一般地，范数满足三角不等式和绝对齐次性：

$$
\begin{aligned}
\|\lambda\boldsymbol x+(1-\lambda)\boldsymbol y\|_2
&\leq
\|\lambda\boldsymbol x\|_2
+\|(1-\lambda)\boldsymbol y\|_2\\
&=\lambda\|\boldsymbol x\|_2
+(1-\lambda)\|\boldsymbol y\|_2\\
&\leq1
\end{aligned}
$$

所以整条线段都在圆盘里，而不只是这个特定中点。

## 常见的凸集和不凸集合

下面这些集合的共同点是，取两个成员做凸组合后，定义它们的约束仍然成立。

| 集合 | 形式 | 为什么是凸的 |
| --- | --- | --- |
| 区间 | $[a,b]$ | 两个数的加权平均仍落在两个端点之间 |
| 球 | $\{\boldsymbol x:\|\boldsymbol x-\boldsymbol c\|_2\leq r\}$ | 范数的三角不等式保留半径约束 |
| 半空间 | $\{\boldsymbol x:\boldsymbol a^\mathsf T\boldsymbol x\leq b\}$ | 线性函数作用于凸组合仍是同样的加权平均 |
| 仿射子空间 | $\{\boldsymbol x:A\boldsymbol x=\boldsymbol b\}$ | $A(\lambda\boldsymbol x+(1-\lambda)\boldsymbol y)=\boldsymbol b$ |
| 概率单纯形 | $\{\boldsymbol p:p_i\geq0,\ \sum_i p_i=1\}$ | 非负性和总和为 1 都被凸组合保留 |

半空间的证明只用线性性。若 $\boldsymbol x,\boldsymbol y$ 都满足约束，则

$$
\begin{aligned}
\boldsymbol a^\mathsf T
[\lambda\boldsymbol x+(1-\lambda)\boldsymbol y]
&=\lambda\boldsymbol a^\mathsf T\boldsymbol x
+(1-\lambda)\boldsymbol a^\mathsf T\boldsymbol y\\
&\leq\lambda b+(1-\lambda)b=b
\end{aligned}
$$

这也是线性模型约束容易处理的原因之一：它们不会把线段切成碎片。

有些形状看起来平滑，却不是凸集。圆周

$$
\mathcal S
=\left\{\boldsymbol x\in\mathbb R^2:
\|\boldsymbol x\|_2=1\right\}
$$

就不是凸的。$(1,0)$ 和 $(0,1)$ 都在圆周上，但它们的中点范数是 $\sqrt{1/2}<1$，中点落在圆盘内部而不在圆周上。两个互不相交的区间也不是凸集：

$$
\mathcal C=[-2,-1]\cup[1,2]
$$

取 $x=-1$ 和 $y=1$，中点 $0$ 不属于 $\mathcal C$。凸性关心的是连线段是否完整，不是边界是否光滑。

## 凸集的运算

凸集可以通过几种常见运算组合起来。

### 交集仍然凸

若每个 $\mathcal C_k$ 都是凸集，则

$$
\mathcal C=\bigcap_{k=1}^m\mathcal C_k
$$

也是凸集。因为 $\boldsymbol x,\boldsymbol y\in\mathcal C$ 意味着它们同时属于每个 $\mathcal C_k$；对每个 $k$，凸组合都属于 $\mathcal C_k$，所以属于所有集合的交集。多个线性不等式和等式约束的可行集正是这个结构。

### 仿射映射保留凸性

令 $\mathcal C$ 凸，$T(\boldsymbol x)=A\boldsymbol x+\boldsymbol b$。它的像

$$
T(\mathcal C)=
\left\{A\boldsymbol x+\boldsymbol b:
\boldsymbol x\in\mathcal C\right\}
$$

仍是凸集。对 $\boldsymbol u=T(\boldsymbol x)$ 和 $\boldsymbol v=T(\boldsymbol y)$：

$$
\begin{aligned}
\lambda\boldsymbol u+(1-\lambda)\boldsymbol v
&=\lambda(A\boldsymbol x+\boldsymbol b)
+(1-\lambda)(A\boldsymbol y+\boldsymbol b)\\
&=A[\lambda\boldsymbol x+(1-\lambda)\boldsymbol y]+\boldsymbol b
\end{aligned}
$$

括号中的点属于 $\mathcal C$，所以右侧属于 $T(\mathcal C)$。仿射映射也保留凸性约束的原像：若 $\mathcal K$ 凸，则

$$
\{\boldsymbol x:A\boldsymbol x+\boldsymbol b\in\mathcal K\}
$$

是凸集。

### 凸包填上所有凸组合

任意集合 $\mathcal S$ 的凸包定义为

$$
\operatorname{conv}(\mathcal S)
=\left\{
\sum_{i=1}^k\alpha_i\boldsymbol x_i:
\boldsymbol x_i\in\mathcal S,\quad
\alpha_i\geq0,\quad
\sum_{i=1}^k\alpha_i=1
\right\}
$$

它包含用 $\mathcal S$ 中有限个点做出的全部凸组合。因为权重仍可再次组合，凸包本身是凸集；它也是包含 $\mathcal S$ 的最小凸集。两个点的凸包是线段，三个不共线点的凸包是实心三角形。这里的「实心」很关键：只保留三角形边界并不凸，因为边上两点的连线可能穿过内部，而边界本身没有包含内部点。

## 凸函数用函数值比较线段

设定义域 $\mathcal D$ 是凸集，函数 $f:\mathcal D\to\mathbb R$ 满足

$$
f\left(\lambda\boldsymbol x+(1-\lambda)\boldsymbol y\right)
\leq
\lambda f(\boldsymbol x)+(1-\lambda)f(\boldsymbol y)
$$

对所有 $\boldsymbol x,\boldsymbol y\in\mathcal D$ 和 $\lambda\in[0,1]$ 成立，就称 $f$ 是凸函数。右侧是两端函数值的线性插值；图像上的点不能高于连接两端图像点的弦。

注意定义有两个对象：$\mathcal D$ 必须是凸的，函数值也必须满足上式。函数本身的表达式和定义域缺一不可。

### 用平方函数把不等式算出来

对 $f(x)=x^2$，任取 $x,y\in\mathbb R$：

$$
\begin{aligned}
&\lambda x^2+(1-\lambda)y^2
-[\lambda x+(1-\lambda)y]^2\\
&=\lambda(1-\lambda)(x-y)^2\\
&\geq0
\end{aligned}
$$

移项就是凸函数定义。取 $x=-1$、$y=3$、$\lambda=1/2$：

$$
\lambda x+(1-\lambda)y=1
$$

于是

$$
f(1)=1,
\qquad
\frac{f(-1)+f(3)}2
=\frac{1+9}{2}=5
$$

确实有 $1\leq5$。这个数字例子表达的是整条抛物线向上弯，而不是「取两个点碰巧通过了检查」。

### 立方函数展示定义域的重要性

$f(x)=x^3$ 在整个 $\mathbb R$ 上不是凸函数。取 $x=-2$、$y=1$、$\lambda=1/2$：

$$
\lambda x+(1-\lambda)y=-\frac12
$$

左侧函数值是

$$
f\left(-\frac12\right)=-\frac18
$$

右侧平均值是

$$
\frac{f(-2)+f(1)}2
=\frac{-8+1}{2}
=-\frac72
$$

凸性要求 $-1/8\leq-7/2$，但这个不等式不成立。原因也能从二阶导数看到：

$$
f''(x)=6x
$$

它在负半轴为负、正半轴为正。因此 $x^3$ 在 $[0,\infty)$ 上是凸的，在 $(-\infty,0]$ 上是凹的，但不能把这两个结论拼成「在整个实数轴上凸」。

## 上图集把函数凸性变成集合凸性

函数 $f$ 的上图集是

$$
\operatorname{epi}(f)
=\left\{(\boldsymbol x,t):
\boldsymbol x\in\mathcal D,\quad
t\geq f(\boldsymbol x)\right\}
$$

它收集定义域内图像上方的所有点。一个函数在凸定义域上凸，当且仅当它的上图集是凸集。

先假设 $f$ 凸。取上图集中的两个点 $(\boldsymbol x,s)$ 和 $(\boldsymbol y,t)$，因此 $s\geq f(\boldsymbol x)$、$t\geq f(\boldsymbol y)$。它们的凸组合是

$$
(\boldsymbol z,r)
=\left(
\lambda\boldsymbol x+(1-\lambda)\boldsymbol y,
\lambda s+(1-\lambda)t
\right)
$$

第一坐标属于 $\mathcal D$，第二坐标满足

$$
\begin{aligned}
r
&\geq
\lambda f(\boldsymbol x)+(1-\lambda)f(\boldsymbol y)\\
&\geq
f\left(\lambda\boldsymbol x+(1-\lambda)\boldsymbol y\right)
=f(\boldsymbol z)
\end{aligned}
$$

所以 $(\boldsymbol z,r)$ 仍在上图集里。反过来，如果上图集凸，取图像上的两个点 $(\boldsymbol x,f(\boldsymbol x))$ 和 $(\boldsymbol y,f(\boldsymbol y))$，它们的凸组合仍在上图集，第二坐标就必须不低于第一坐标处的函数值，恰好得到凸函数不等式。

![凸函数的弦在图像上方](/assets/optimization-theory/svg/convex-sets-and-functions.1.svg)

这个等价关系在优化里很实用：检查函数是否凸，可以改看一个更高一维的集合是否凸；几何上，所有「图像上方」的点没有被连线段穿出去。

## Jensen 不等式：两点比较的有限平均版

两点定义可以递推到任意有限个点。若 $\boldsymbol x_1,\ldots,\boldsymbol x_k\in\mathcal D$，权重满足

$$
\alpha_i\geq0,
\qquad
\sum_{i=1}^k\alpha_i=1
$$

则

$$
f\left(\sum_{i=1}^k\alpha_i\boldsymbol x_i\right)
\leq
\sum_{i=1}^k\alpha_i f(\boldsymbol x_i)
$$

这就是有限形式的 Jensen 不等式。左侧先平均输入再应用函数，右侧先应用函数再平均输出。凸函数的顺序给出左侧不超过右侧；凹函数则反向。

### 具体的随机变量例子

令随机变量 $X$ 以相同概率取 $0$ 和 $2$，取凸函数 $f(x)=x^2$。期望输入是

$$
\mathbb E[X]
=\frac12\cdot0+\frac12\cdot2=1
$$

先平均再平方：

$$
f(\mathbb E[X])=f(1)=1
$$

先平方再平均：

$$
\mathbb E[f(X)]
=\frac12f(0)+\frac12f(2)
=\frac12\cdot0+\frac12\cdot4=2
$$

所以

$$
f(\mathbb E[X])=1
\leq
2=\mathbb E[f(X)]
$$

差值 $1$ 是这个分布在平方函数下的 Jensen gap。它不是抽样误差，而是「先平均」与「先应用非线性函数」之间的固定差异；如果 $f$ 是仿射函数，这个差异恒为 0。

在概率论里写成

$$
f(\mathbb E[X])
\leq
\mathbb E[f(X)]
$$

需要 $\mathbb E[X]$ 和 $\mathbb E[f(X)]$ 存在，并且随机变量的取值落在函数的凸定义域内。不能因为某一次有限样本平均满足某个式子，就跳过这些定义域和可积性条件。

## 常见凸函数从哪里来

### 仿射函数

若

$$
f(\boldsymbol x)=\boldsymbol a^\mathsf T\boldsymbol x+b
$$

则

$$
\begin{aligned}
f(\lambda\boldsymbol x+(1-\lambda)\boldsymbol y)
&=\lambda f(\boldsymbol x)+(1-\lambda)f(\boldsymbol y)
\end{aligned}
$$

等号成立，所以仿射函数既凸又凹。它没有弯曲，图像上的弦正好落在图像上。

### 范数和平方范数

任意范数都满足

$$
\|\lambda\boldsymbol x+(1-\lambda)\boldsymbol y\|
\leq
\lambda\|\boldsymbol x\|+(1-\lambda)\|\boldsymbol y\|
$$

所以范数是凸函数。平方二范数也凸：

$$
f(\boldsymbol x)=\|\boldsymbol x\|_2^2
$$

其二阶差值可以直接写成

$$
\begin{aligned}
&\lambda\|\boldsymbol x\|_2^2
+(1-\lambda)\|\boldsymbol y\|_2^2
-\|\lambda\boldsymbol x+(1-\lambda)\boldsymbol y\|_2^2\\
&=\lambda(1-\lambda)\|\boldsymbol x-\boldsymbol y\|_2^2
\geq0
\end{aligned}
$$

### 指数函数

$f(x)=e^x$ 在整个实数轴上凸，因为

$$
f''(x)=e^x>0
$$

它不是强凸函数：虽然每个点的曲率为正，但当 $x\to-\infty$ 时，$e^x$ 趋近于 0，不存在统一的正数 $\mu$ 使所有点的曲率都至少为 $\mu$。

### 仿射函数的最大值

若每个

$$
\ell_i(\boldsymbol x)
=\boldsymbol a_i^\mathsf T\boldsymbol x+b_i
$$

都是仿射函数，则

$$
f(\boldsymbol x)=\max_i\ell_i(\boldsymbol x)
$$

是凸函数。对任意 $\boldsymbol x,\boldsymbol y$：

$$
\begin{aligned}
f(\lambda\boldsymbol x+(1-\lambda)\boldsymbol y)
&=\max_i\left[
\lambda\ell_i(\boldsymbol x)
+(1-\lambda)\ell_i(\boldsymbol y)\right]\\
&\leq
\lambda\max_i\ell_i(\boldsymbol x)
+(1-\lambda)\max_i\ell_i(\boldsymbol y)
\end{aligned}
$$

最大值函数可以有尖角。不可微不代表不凸；后续算法可能使用次梯度而不是普通梯度。

### log-sum-exp

机器学习常见的

$$
L(\boldsymbol z)
=\log\sum_{j=1}^K e^{z_j}
$$

也是凸函数。令

$$
p_j=\frac{e^{z_j}}{\sum_{k=1}^Ke^{z_k}}
$$

则梯度是 $\nabla L=\boldsymbol p$，Hessian 为

$$
\nabla^2L
=\operatorname{diag}(\boldsymbol p)
-\boldsymbol p\boldsymbol p^\mathsf T
$$

对任意向量 $\boldsymbol v$：

$$
\begin{aligned}
\boldsymbol v^\mathsf T\nabla^2L\boldsymbol v
&=\sum_jp_jv_j^2
-\left(\sum_jp_jv_j\right)^2\\
&=\operatorname{Var}_{J\sim\boldsymbol p}(v_J)\\
&\geq0
\end{aligned}
$$

所以 Hessian 半正定。softmax 交叉熵是 $L(\boldsymbol z)-z_y$，减去一个仿射函数不改变凸性。

## 二次函数把凸性写成半正定矩阵

考虑二次函数

$$
f(\boldsymbol x)
=\frac12\boldsymbol x^\mathsf TQ\boldsymbol x
+\boldsymbol b^\mathsf T\boldsymbol x+c
$$

只有 $Q$ 的对称部分影响二次项。令

$$
Q_{\mathrm s}
=\frac{Q+Q^\mathsf T}{2}
$$

则

$$
\boldsymbol x^\mathsf TQ\boldsymbol x
=\boldsymbol x^\mathsf TQ_{\mathrm s}\boldsymbol x
$$

因为反对称部分的二次型恒为 0。二次函数的 Hessian 是 $Q_{\mathrm s}$，因此

$$
f\text{ 凸}
\quad\Longleftrightarrow\quad
Q_{\mathrm s}\succeq0
$$

这里 $\succeq0$ 表示对所有 $\boldsymbol v$ 都有 $\boldsymbol v^\mathsf TQ_{\mathrm s}\boldsymbol v\geq0$。如果存在 $\mu>0$ 使

$$
Q_{\mathrm s}\succeq\mu I
$$

则 $f$ 是 $\mu$-强凸的。

### 数字矩阵例子

取

$$
Q=
\begin{bmatrix}
2&0\\
0&4
\end{bmatrix},
\qquad
f(\boldsymbol x)=\frac12\boldsymbol x^\mathsf TQ\boldsymbol x
$$

它的特征值是 $2$ 和 $4$，所以

$$
Q\succeq2I
$$

函数在 $\boldsymbol x=(1,1)$ 处的值为

$$
f(1,1)
=\frac12(2+4)=3
$$

对任意 $\boldsymbol v=(v_1,v_2)$：

$$
\boldsymbol v^\mathsf TQ\boldsymbol v
=2v_1^2+4v_2^2
\geq2(v_1^2+v_2^2)
=2\|\boldsymbol v\|_2^2
$$

这同时给出了凸性和强凸性的数值证据。最小特征值控制最平的方向；即使另一方向曲率更大，强凸参数也不能超过最小曲率。

### 一个不凸的鞍形二次函数

取

$$
Q=
\begin{bmatrix}
1&0\\
0&-1
\end{bmatrix},
\qquad
f(x_1,x_2)=\frac12(x_1^2-x_2^2)
$$

沿 $x_1$ 方向向上弯，沿 $x_2$ 方向向下弯。取 $\boldsymbol x=(0,-1)$、$\boldsymbol y=(0,1)$：

$$
f(\boldsymbol x)=f(\boldsymbol y)=-\frac12,
\qquad
f\left(\frac{\boldsymbol x+\boldsymbol y}{2}\right)=f(0,0)=0
$$

凸性会要求 $0\leq-1/2$，因此失败。矩阵有一个负特征值，就存在一条方向使二次函数的线段比较反过来。

## 严格凸、强凸和最优解唯一性

普通凸性允许线段上处处相等。例如仿射函数满足等号。若对任意不同的 $\boldsymbol x,\boldsymbol y$ 和 $0<\lambda<1$ 都有严格不等式

$$
f(\lambda\boldsymbol x+(1-\lambda)\boldsymbol y)
<
\lambda f(\boldsymbol x)+(1-\lambda)f(\boldsymbol y)
$$

则称 $f$ 严格凸。严格凸函数在凸集上的最小解至多一个：如果两个不同点都是最小解，它们的中间点会有更小的函数值，矛盾。

强凸给出比严格不等式更具体的余量。若 $f$ 是 $\mu$-强凸，则

$$
\begin{aligned}
f(\lambda\boldsymbol x+(1-\lambda)\boldsymbol y)
&\leq
\lambda f(\boldsymbol x)+(1-\lambda)f(\boldsymbol y)\\
&\quad-\frac\mu2\lambda(1-\lambda)
\|\boldsymbol x-\boldsymbol y\|_2^2
\end{aligned}
$$

其中 $\mu>0$。余量随着两点距离增大而增大，表示函数不只是「向上弯」，还有统一的最小曲率。强凸一定严格凸；严格凸不一定强凸，例如 $e^x$ 在整个实数轴上严格凸但没有统一正的二阶曲率下界。

对 $f(x)=x^2$，Hessian 是 $2$，所以可以取 $\mu=2$。令 $x=-1$、$y=1$、$\lambda=1/2$：

$$
f(0)=0
=\frac12f(-1)+\frac12f(1)
-\frac{2}{2}\cdot\frac12\cdot\frac12\cdot4
$$

这里等号成立，说明强凸不等式的二次余量可以被精确达到。

## 凸性怎样把局部最小值升级成全局最小值

设 $\mathcal D$ 是凸集，$f$ 在 $\mathcal D$ 上凸，$\boldsymbol x^\star$ 是相对于 $\mathcal D$ 的局部最小点。假设存在另一个可行点 $\boldsymbol y$ 满足

$$
f(\boldsymbol y)<f(\boldsymbol x^\star)
$$

对任意 $0<t<1$，线段上的点

$$
\boldsymbol z_t
=(1-t)\boldsymbol x^\star+t\boldsymbol y
$$

仍属于 $\mathcal D$。凸性给出

$$
\begin{aligned}
f(\boldsymbol z_t)
&\leq
(1-t)f(\boldsymbol x^\star)+tf(\boldsymbol y)\\
&< (1-t)f(\boldsymbol x^\star)+tf(\boldsymbol x^\star)\\
&=f(\boldsymbol x^\star)
\end{aligned}
$$

当 $t$ 足够小时，$\boldsymbol z_t$ 任意接近 $\boldsymbol x^\star$，却比它的函数值更低，这与局部最小矛盾。因此不存在这样的 $\boldsymbol y$，$\boldsymbol x^\star$ 必是全局最小点。

证明中两个条件都被用到了：凸函数提供线段上的上界，凸定义域保证这条线段是可行的。如果目标凸但约束集合不凸，绕开局部点的低处可能被可行域的空洞挡住，结论不能直接套用。

## 一阶支撑超平面

若 $f$ 在凸开集上可微，凸性等价于对任意 $\boldsymbol x,\boldsymbol y$：

$$
f(\boldsymbol y)
\geq
f(\boldsymbol x)
+\nabla f(\boldsymbol x)^\mathsf T
(\boldsymbol y-\boldsymbol x)
$$

右侧是图像在 $\boldsymbol x$ 处的切平面值。凸函数的图像不会掉到自己的切平面下方，切平面是一个全局下界。

可以沿线段证明这个式子。定义一维函数

$$
\phi(t)=f(\boldsymbol x+t(\boldsymbol y-\boldsymbol x)),
\qquad
t\in[0,1]
$$

$\phi$ 是一维凸函数。凸函数在起点的右导数不超过从起点到终点的割线斜率：

$$
\phi'(0^+)\leq\phi(1)-\phi(0)
$$

由于

$$
\phi'(0^+)
=\nabla f(\boldsymbol x)^\mathsf T(\boldsymbol y-\boldsymbol x)
$$

就得到支撑超平面不等式。

若无约束最小点 $\boldsymbol x^\star$ 位于可微定义域内部，必要条件是

$$
\nabla f(\boldsymbol x^\star)=\boldsymbol0
$$

对凸函数，这个条件还足以推出全局最小：

$$
f(\boldsymbol y)
\geq
f(\boldsymbol x^\star)
+\boldsymbol0^\mathsf T(\boldsymbol y-\boldsymbol x^\star)
=f(\boldsymbol x^\star)
$$

这不是说「梯度为零在所有函数上都够用」；它依赖凸性。非凸函数的梯度零点也可能是局部最大值或鞍点。

## Hessian 判据和它的适用范围

若 $f$ 在凸开集上二阶连续可微，则

$$
f\text{ 凸}
\quad\Longleftrightarrow\quad
\nabla^2f(\boldsymbol x)\succeq0
\quad\text{对所有 }\boldsymbol x
$$

必要性来自沿任意方向 $\boldsymbol v$ 的一维限制：

$$
g(t)=f(\boldsymbol x+t\boldsymbol v)
$$

如果 $f$ 凸，$g$ 也凸，所以

$$
g''(0)
=\boldsymbol v^\mathsf T\nabla^2f(\boldsymbol x)\boldsymbol v
\geq0
$$

充分性则是把一维二阶导数非负沿任意线段积分两次，恢复凸函数的不等式。

这个判据有三个边界：

1. 只看一个点的 Hessian 不能证明全局凸性。
2. 不可微函数没有普通 Hessian，仍可能是凸的，例如 $|x|$。
3. 定义域需要是凸集；Hessian 只描述函数在每个局部方向的弯曲，不能填补定义域里的断裂。

## 凸性在机器学习目标里的位置

### 最小二乘对线性参数是凸的

把偏置并入特征，令 $\tilde{\boldsymbol x}_i$ 是扩展后的输入，参数为 $\boldsymbol\theta$。平方损失可以写成

$$
L(\boldsymbol\theta)
=\frac1{2n}
\sum_{i=1}^n
(\boldsymbol\theta^\mathsf T\tilde{\boldsymbol x}_i-y_i)^2
$$

它的 Hessian 是

$$
\nabla^2L(\boldsymbol\theta)
=\frac1n
\sum_{i=1}^n
\tilde{\boldsymbol x}_i\tilde{\boldsymbol x}_i^\mathsf T
$$

对任意 $\boldsymbol v$：

$$
\boldsymbol v^\mathsf T\nabla^2L\boldsymbol v
=\frac1n\sum_{i=1}^n
(\boldsymbol v^\mathsf T\tilde{\boldsymbol x}_i)^2
\geq0
$$

所以它对线性参数是凸的。若样本特征矩阵满列秩，Hessian 正定，最小解唯一；若存在未被数据区分的参数方向，Hessian 只有半正定，可能有多个同样好的解。[最小二乘是投影](../linear-models/least-squares-as-projection/)会从几何角度展开这个平方目标。

### 逻辑回归对参数是凸的

二分类的单样本损失，令 $z=\boldsymbol\theta^\mathsf T\tilde{\boldsymbol x}$：

$$
\ell(z,y)
=\log(1+e^z)-yz,
\qquad
y\in\{0,1\}
$$

关于 logit 的二阶导数是

$$
\frac{\partial^2\ell}{\partial z^2}
=\sigma(z)[1-\sigma(z)]
\geq0
$$

$z$ 是参数的仿射函数，仿射复合保留凸性；样本损失的非负加权平均也保留凸性。因此逻辑回归在参数空间中是凸优化问题。[逻辑回归](../linear-models/logistic-regression/)会继续处理概率解释、分离和正则化。

### softmax 交叉熵只在 logits 或最后一层保持这个保证

对 logits $\boldsymbol z$ 和真实类别 $y$：

$$
\ell(\boldsymbol z,y)
=\log\sum_j e^{z_j}-z_y
$$

上一节已经看到第一项 log-sum-exp 凸，第二项是仿射函数，所以该损失对 $\boldsymbol z$ 凸。如果 logits 对最后一层参数是仿射函数，那么固定前面表示时，最后一层的训练仍是凸的。

但深层网络的 logits 通常是

$$
\boldsymbol z
=W_L\phi_{L-1}
\left(
W_{L-1}\phi_{L-2}(\cdots)
\right)
$$

多个权重通过乘法和非线性组合在一起。即便损失对 logits 凸，也不能把凸性自动传回所有层的权重。一个最小反例是

$$
g(a,b)=(ab-1)^2
$$

它把两个参数相乘后再平方，整体不是凸函数。ReLU 作为关于输入的函数可以是凸的，但「输入」和「权重」同时变化时，网络训练问题仍然通常非凸。

## 复合、加和和加权时不要漏掉条件

若 $f_i$ 都是凸函数，且 $\alpha_i\geq0$，则

$$
f(\boldsymbol x)
=\sum_i\alpha_i f_i(\boldsymbol x)
$$

是凸函数；加一个仿射函数也不会改变凸性。仿射变换后的 $f(A\boldsymbol x+\boldsymbol b)$ 保持凸性，因为任意线段经过仿射变换仍是线段。

更一般的复合函数需要检查单调性。例如外层 $h$ 凸且不减、内层 $g$ 凸时，$h\circ g$ 才能按常见规则保持凸性；若外层的单调性方向相反，或者两个非线性参数互相相乘，就不能只凭「每个部分看起来凸」下结论。

这是机器学习里常见的边界：固定表示后最后一层损失可能凸，端到端训练却未必凸；对每个参数块分别凸，也不等于对所有参数联合凸。

## 失效模式

**只检查一个中点。** 一个点满足中点不等式只能说明这个局部样例通过；凸性要覆盖所有点对和所有 $\lambda\in[0,1]$。

**把圆周当成圆盘。** 圆周边界不包含弦的内部，圆盘才是凸集。平滑、连通或没有尖角都不是凸性的定义。

**把凸集和凸函数混为一谈。** 集合检查点的线段是否仍可行，函数检查函数值是否低于弦；前者说允许走哪里，后者说目标沿路怎么变。

**把函数表达式和定义域拆开。** $x^3$ 在非负半轴上凸，在整个实数轴上不凸；定义域是定理假设的一部分。

**只看一个 Hessian。** 某个点半正定不能推出全局凸；需要在整个凸开定义域上检查，且不可微函数要使用别的判据。

**忘记先对称化二次矩阵。** 二次型只看 $(Q+Q^\mathsf T)/2$；直接把一个非对称写法当作 Hessian 会把不影响函数的反对称部分混进判据。

**把严格凸和强凸当成同义词。** 强凸给统一的二次余量，严格凸只要求每条非退化线段有严格不等式；前者更强。

**把 logits 的凸性传给整张网络。** softmax 交叉熵对 logits 凸，不代表经过多层权重乘法后的端到端目标凸。

**把 Jensen 的权重当普通系数。** 必须有 $\alpha_i\geq0$ 且 $\sum_i\alpha_i=1$；任意线性组合不代表平均。

**忘记全局结论依赖凸定义域。** 「局部最小就是全局最小」使用了从局部点通往任意候选点的整条可行线段；非凸约束集合会破坏这一步。

## 相关词条

- [优化问题](../optimization-theory/optimization-problems/)：先把目标函数、决策变量和可行集写清楚。
- [线性组合与张成](../linear-algebra/linear-combinations-and-span/)：凸组合是系数非负且和为 1 的特殊线性组合。
- [仿射空间与映射](../linear-algebra/affine-spaces-and-maps/)：解释仿射集合、仿射变换和凸性保留。
- [二次型](../linear-algebra/quadratic-forms/)：展开二次函数、矩阵和曲率的对应关系。
- [特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/)：用最小特征值读取二次函数最平的方向。
- [谱定理](../linear-algebra/spectral-theorem/)：说明对称 Hessian 如何被正交对角化。
- [局部与全局最小值](../optimization-theory/local-and-global-minima/)：区分邻域内结论和整个可行域内结论。
- [一阶最优性条件](../optimization-theory/first-order-optimality/)：处理梯度为零、边界和约束下的必要条件。
- [二阶最优性条件](../optimization-theory/second-order-optimality/)：用 Hessian 分析局部曲率。
- [最小二乘是投影](../linear-models/least-squares-as-projection/)：把平方损失的凸二次结构转成正交投影。
- [逻辑回归](../linear-models/logistic-regression/)：展开逻辑损失为什么对线性参数凸。
- [softmax](../neurons-and-activations/softmax/)：处理 logits、归一化概率和多分类损失。
