---
title: "局部与全局最小值：梯度停在哪，不等于找到哪里"
tags: ["why-models-learn"]
---

在最小化问题中，**全局最小点**要和整个可行域的所有点比较，**局部最小点**只要在它附近没有更低的可行点。全局最小点一定是局部最小点，但反过来不成立；一个梯度为零的点甚至可能是局部最大值或鞍点。本篇把比较范围、边界点、驻点、Hessian 二阶判据和凸性条件分开，再用一个有两个深度不同的谷底的多项式说明非凸优化为什么会停在局部而不是全局。

## 全局和局部的比较范围不同

考虑最小化

$$
\min_{\boldsymbol x\in\mathcal D}f(\boldsymbol x)
$$

如果 $\boldsymbol x^\star\in\mathcal D$ 满足

$$
f(\boldsymbol x^\star)
\leq f(\boldsymbol y)
\qquad
\text{for all }\boldsymbol y\in\mathcal D
$$

那么 $\boldsymbol x^\star$ 是全局最小点，$f(\boldsymbol x^\star)$ 是全局最小值。它比较整个可行域，不管另一个点离它近还是远。

以 $\boldsymbol x^\star$ 为中心、半径 $\varepsilon$ 的邻域是

$$
B(\boldsymbol x^\star,\varepsilon)
=\left\{\boldsymbol x:
\|\boldsymbol x-\boldsymbol x^\star\|_2<\varepsilon\right\}
$$

如果存在某个 $\varepsilon>0$，使

$$
f(\boldsymbol x^\star)
\leq f(\boldsymbol y)
\qquad
\text{for all }\boldsymbol y\in
\mathcal D\cap B(\boldsymbol x^\star,\varepsilon)
$$

那么 $\boldsymbol x^\star$ 是相对于可行域 $\mathcal D$ 的局部最小点。这里的「附近」必须与 $\mathcal D$ 取交集：如果 $\boldsymbol x^\star$ 在边界上，就只比较仍然可行的一侧。

严格局部最小点把不等式收紧为

$$
f(\boldsymbol x^\star)
<
f(\boldsymbol y)
\qquad
\text{for all }\boldsymbol y\in
\mathcal D\cap B(\boldsymbol x^\star,\varepsilon),
\ \boldsymbol y\neq\boldsymbol x^\star
$$

类似地，最大值问题可以对 $-f$ 使用同样的定义。术语「局部最小值」有时指函数值，有时指达到它的点；本篇用「点」指参数，用「值」指对应的数。

全局最小点自动满足局部定义，因为全局比较包含了邻域比较。但全局最小点不一定唯一，局部最小点也不一定是全局的：

$$
\operatorname*{arg\,min}_{\boldsymbol x\in\mathcal D}f(\boldsymbol x)
$$

可能包含多个点，而某个局部最小点可能根本不属于这个集合。

## 凸抛物线和平台

先看没有局部与全局差别的简单例子：

$$
f(x)=(x-2)^2+1
$$

因为平方项非负，且只在 $x=2$ 时为零：

$$
f(2)=1,
\qquad
f(x)>1\quad\text{for all }x\neq2
$$

所以 $x=2$ 同时是严格局部最小点和全局最小点。函数是凸的，之后会证明：在凸定义域上，凸函数的每一个局部最小点都具有这个性质。

不要求严格时，最小点可以是一整段。设

$$
f(x)=
\begin{cases}
0,&-1\leq x\leq1\\
(x-1)^2,&x>1\\
(x+1)^2,&x<-1
\end{cases}
$$

那么区间 $[-1,1]$ 中的每一点都有函数值 0，都是全局最小点，但没有一个是严格最小点。报告「找到一个最小值」时，不能自动把它翻译成「参数唯一」。

## 局部最小却不是全局最小

要看清非凸情形，构造一个有两个谷底的多项式：

$$
f(x)
=5x^6+24x^5+30x^4-40x^3-75x^2
$$

它的导数可以因式分解为

$$
f'(x)
=30x(x^2-1)(x^2+4x+5)
$$

最后一个因子是

$$
x^2+4x+5=(x+2)^2+1>0
$$

所以全部驻点只有 $x=-1,0,1$。导数符号按区间变化如下：

| 区间 | $x$ | $x^2-1$ | $f'(x)$ 的符号 |
| --- | --- | --- | --- |
| $(-\infty,-1)$ | 负 | 正 | 负 |
| $(-1,0)$ | 负 | 负 | 正 |
| $(0,1)$ | 正 | 负 | 负 |
| $(1,\infty)$ | 正 | 正 | 正 |

函数先降后升，所以 $x=-1$ 是一个局部谷底；随后先升后降，$x=0$ 是局部峰顶；最后再次先降后升，$x=1$ 是另一个局部谷底。

直接代入三个驻点：

$$
f(-1)=-24,
\qquad
f(0)=0,
\qquad
f(1)=-56
$$

同时

$$
\lim_{|x|\to\infty}f(x)=+\infty
$$

导数没有其他零点，符号表覆盖了整条实数轴。因此 $x=-1$ 是局部最小点但不是全局最小点，$x=1$ 才是全局最小点。数值上，两个谷底相差 32 个目标单位；「已经向下走了」不说明走到了更深的那个谷底。

二阶导数也能确认三个点的局部形状：

$$
f''(x)
=150x^4+480x^3+360x^2-240x-150
$$

代入：

$$
f''(-1)=120>0,
\qquad
f''(0)=-150<0,
\qquad
f''(1)=600>0
$$

正、负、正分别对应局部最小、局部最大、局部最小。但二阶导数只说明局部曲率，不负责比较两个谷底谁更低；全局比较还需要看函数在整个定义域上的值。

![局部谷底不一定是全局谷底](/assets/optimization-theory/svg/local-and-global-minima.1.svg)

## 边界点不需要零梯度

梯度为零是无约束、可微、内部局部极值的必要条件。它不是所有最优点的必要条件。考虑

$$
\min_{x\in[0,2]}f(x),
\qquad
f(x)=x
$$

最优点是左端点 $x^\star=0$：

$$
f(0)=0,
\qquad
f(x)>0\quad\text{for all }x\in(0,2]
$$

但

$$
f'(0)=1\neq0
$$

原因是 $x=0$ 左侧的点不在可行域内。对任意可行的小方向 $d>0$：

$$
\frac{f(0+td)-f(0)}{t}=d>0
\qquad
t>0
$$

所有可行方向都让目标上升，没有可行下降方向，所以端点仍然是最小点。若机械地只寻找 $f'(x)=0$，会漏掉这个答案。

在多维约束中，边界最优点的梯度可以指向不可行区域；正确的局部判据要比较可行方向，等式/不等式约束则需要一阶最优性条件和 KKT 结构。[一阶最优性条件](../optimization-theory/first-order-optimality/)会把这个边界版本单独写清楚。

## 驻点不等于极小点

对可微无约束函数，满足

$$
\nabla f(\boldsymbol x^\star)=\boldsymbol0
$$

的点称为驻点。若 $\boldsymbol x^\star$ 是内部局部最小点，那么梯度必须为零：沿任意方向 $\boldsymbol v$ 的一维限制

$$
g(t)=f(\boldsymbol x^\star+t\boldsymbol v)
$$

在 $t=0$ 处有局部最小，因此 $g'(0)=0$，也就是

$$
\nabla f(\boldsymbol x^\star)^\mathsf T\boldsymbol v=0
\qquad
\text{for all }\boldsymbol v
$$

从而梯度为零。这是必要条件，不是充分条件。

### 立方函数的水平拐点

令

$$
f(x)=x^3
$$

则

$$
f'(0)=0,
\qquad
f''(0)=0
$$

但任意 $\varepsilon>0$ 内都有负的 $x$ 和正的 $x$。在左侧 $f(x)<0=f(0)$，在右侧 $f(x)>0=f(0)$，所以原点既不是局部最小，也不是局部最大。它是函数变平后改变弯曲方向的水平拐点。

### 鞍点同时有上升和下降方向

二维函数

$$
f(x,y)=x^2-y^2
$$

在原点的梯度为零：

$$
\nabla f(0,0)=
\begin{bmatrix}
0\\
0
\end{bmatrix}
$$

沿 $x$ 轴：

$$
f(t,0)=t^2>0
\qquad
t\neq0
$$

沿 $y$ 轴：

$$
f(0,t)=-t^2<0
\qquad
t\neq0
$$

所以原点附近既有比 $f(0,0)=0$ 高的值，也有更低的值，不能称为极小或极大。这样的驻点称为鞍点；曲面在不同方向的弯曲符号不同。

### Hessian 为零时还没有结论

函数

$$
f(x)=x^4
$$

在原点满足

$$
f'(0)=0,
\qquad
f''(0)=0
$$

但 $x^4\geq0$，所以原点是严格的全局最小点。把它换成 $x^3$，同样的两个导数值却不再是极小点。二阶信息为零时，必须回到函数定义或继续检查更高阶项，不能把「Hessian 不提供信息」误读成「没有最小值」。

## 二阶判据到底能保证什么

若 $f$ 在内部点 $\boldsymbol x^\star$ 附近二阶连续可微，Taylor 展开为

$$
f(\boldsymbol x^\star+\boldsymbol h)
=f(\boldsymbol x^\star)
+\nabla f(\boldsymbol x^\star)^\mathsf T\boldsymbol h
+\frac12\boldsymbol h^\mathsf T
H(\boldsymbol x^\star)\boldsymbol h
+o(\|\boldsymbol h\|_2^2)
$$

在驻点处，一阶项消失：

$$
f(\boldsymbol x^\star+\boldsymbol h)-f(\boldsymbol x^\star)
=\frac12\boldsymbol h^\mathsf T
H(\boldsymbol x^\star)\boldsymbol h
+o(\|\boldsymbol h\|_2^2)
$$

于是有下面的判据：

| 条件 | 能推出的局部结论 |
| --- | --- |
| $H$ 正定 | 严格局部最小点 |
| $H$ 负定 | 严格局部最大点 |
| $H$ 不定 | 鞍点 |
| $H$ 半正定但奇异 | 二阶测试无结论 |
| $H$ 半负定但奇异 | 二阶测试无结论 |

「正定」意味着存在某个 $c>0$，对所有非零方向 $\boldsymbol h$：

$$
\boldsymbol h^\mathsf TH\boldsymbol h
\geq c\|\boldsymbol h\|_2^2
$$

二次项在所有方向都严格为正，足以压过余项。负定的说法对 $-f$ 相同。不定矩阵存在 $\boldsymbol u,\boldsymbol v$ 使二次型分别为正和负，因此附近同时出现更高和更低的函数值。

半正定只说明二次项没有负方向，可能仍有平坦方向。$x^4$ 和 $x^3$ 都说明了为什么奇异 Hessian 不能单独给出极值结论。[二阶最优性条件](../optimization-theory/second-order-optimality/)会在更完整的约束和退化情形下继续整理这张表；[Hessian 矩阵](../calculus/hessian/)则从方向二阶导数开始推导同一判据。

## 凸性让局部结论覆盖全局

如果 $\mathcal D$ 是凸集，$f$ 在 $\mathcal D$ 上是凸函数，那么任意局部最小点都是全局最小点。证明只用一条线段。

设 $\boldsymbol x^\star$ 是局部最小点，假设存在可行点 $\boldsymbol y$ 使

$$
f(\boldsymbol y)<f(\boldsymbol x^\star)
$$

由于 $\mathcal D$ 凸，对任意 $0<t<1$：

$$
\boldsymbol z_t
=(1-t)\boldsymbol x^\star+t\boldsymbol y
\in\mathcal D
$$

凸性给出

$$
\begin{aligned}
f(\boldsymbol z_t)
&\leq
(1-t)f(\boldsymbol x^\star)+tf(\boldsymbol y)\\
&<
(1-t)f(\boldsymbol x^\star)+tf(\boldsymbol x^\star)\\
&=f(\boldsymbol x^\star)
\end{aligned}
$$

当 $t$ 足够小时，$\boldsymbol z_t$ 落在 $\boldsymbol x^\star$ 的任意局部邻域内，却有更低函数值，矛盾。因此不存在这样的 $\boldsymbol y$。

这个定理没有保证最小点一定存在。例如 $f(x)=e^x$ 在 $\mathbb R$ 上凸，函数值可以无限接近 0，但没有哪个有限 $x$ 达到 0。定理说的是「如果找到了局部最小点，它就是全局的」，不是「任何凸函数都能找到一个最小点」。

严格凸函数的全局最小点至多一个；强凸函数还提供统一的二次间隔。非严格凸函数可能有一整段同样好的点。凸性消除了局部非全局谷底，但不自动解决存在性、数值精度或算法速度问题。[凸集与凸函数](../optimization-theory/convex-sets-and-functions/)会从线段不等式、上图集和 Hessian 判据建立这些结论。

如果目标是凸的但可行域不凸，结论会失效。取

$$
\mathcal D=[-3,-2]\cup[1,3],
\qquad
f(x)=x^2
$$

在相对于 $\mathcal D$ 的邻域里，$x=-2$ 是局部最小点，因为它左侧可行点的平方都大于 4；但 $x=1$ 的函数值为 1，比 $f(-2)=4$ 更低。目标函数虽然凸，断开的可行域却没有提供连接两个分支的线段。

## 算法看到的是局部信息

梯度下降的一步写成

$$
\boldsymbol x_{k+1}
=\boldsymbol x_k-\eta\nabla f(\boldsymbol x_k)
$$

它使用当前位置的梯度和步长 $\eta$。当梯度变小或满足停止阈值时，算法得到的是一个数值上的近似驻点：

$$
\|\nabla f(\boldsymbol x_k)\|_2\leq\varepsilon
$$

这个条件本身不能区分全局最小、局部最小、鞍点、平坦拐点和数值误差。还需要目标的凸性、Hessian、函数值比较或其他结构来解释结果。

从一个初始点出发，迭代可能进入某个局部最小点的吸引盆。这里的「吸引盆」不是目标函数单独定义的集合，而是相对于某个具体更新规则、步长和停止标准来说，最终会落到同一个区域的初始点集合。换优化器、步长或随机噪声，吸引盆的边界都可能变化。

随机重启可以增加找到较深谷底的机会，噪声也可能帮助迭代越过较浅的障碍，但它们都不是全局最优证明。若需要全局保证，必须给出额外的可验证假设或使用有明确全局性质的算法；一条训练曲线只能报告这次运行到达了哪里。

## 机器学习里三种「好」不能混为一谈

神经网络训练常写成

$$
\min_{\boldsymbol\theta}
L(\boldsymbol\theta)
=\frac1n\sum_{i=1}^n
\ell\bigl(f_{\boldsymbol\theta}(\boldsymbol x_i),y_i\bigr)
$$

这里至少有三个不同问题。

**参数空间中的局部/全局。** $\boldsymbol\theta^\star$ 可能只是当前参数邻域内的局部最小点。网络整体通常非凸，不能凭梯度小就宣称它是全局最小点。

**训练目标的数值。** 即使找到了训练集上的全局最小点，也只是让这批样本的经验风险最小；它不自动让总体风险、验证集误差或用户关心的指标最小。

**函数而不是参数。** 不同参数可以表示同一个函数。例如两层线性网络

$$
f_{W_2,W_1}(\boldsymbol x)
=W_2W_1\boldsymbol x
$$

对任意可逆矩阵 $P$：

$$
(W_2P^{-1})(PW_1)\boldsymbol x
=W_2W_1\boldsymbol x
$$

因此参数空间中可能有一条或一片等价解，而它们的预测完全相同。隐藏单元置换、ReLU 的尺度重参数化也会产生类似现象。报告唯一参数解之前，要先问参数是否可识别。

最后一层固定表示后的线性回归或逻辑回归可以是凸问题；端到端更新所有层时，权重之间的复合通常把问题变成非凸问题。即使全局训练最小值存在，训练算法也不一定能从当前初始化找到它。

## 失效模式

**把梯度为零当成最小值。** 驻点可能是局部最大、鞍点或水平拐点；先检查局部函数值、方向或二阶结构。

**把半正定 Hessian 当成充分条件。** 半正定只排除了二阶负方向；奇异方向需要更高阶项或直接比较函数值。

**忘记边界。** 约束边界的最优点可以有非零梯度；必须只沿可行方向比较。

**把局部最低当成全局最低。** 非凸函数可以有多个深度不同的谷底；多次初始化得到不同目标值正是需要解释的证据。

**忽略定义域。** 凸目标在非凸可行域上仍可能出现局部而非全局的相对最小点。

**把训练 loss 当成任务最优。** 训练集上的经验风险、验证集表现、总体风险和业务指标不是同一个函数。

**把参数唯一当成函数唯一。** 网络对隐藏单元置换、尺度变化和矩阵分解可能不变；不同参数点不一定意味着不同模型行为。

**把停止阈值当成定理。** 小梯度是数值停止条件，不自动是全局最优性证书；还要记录容差、约束状态和目标结构。

**把随机重启当成证明。** 重启提高经验上找到深谷的概率，但有限次尝试不能证明没有更低的谷底。

**把「没有找到」写成「不存在」。** 一个优化器没访问到更低点，只说明搜索轨迹有限；不能由一次运行断言全局最小值不存在或已经达到。

## 相关词条

- [优化问题](../optimization-theory/optimization-problems/)：区分目标值、最优点、可行集和下确界。
- [凸集与凸函数](../optimization-theory/convex-sets-and-functions/)：说明凸目标在凸可行域上为何没有局部非全局最小点。
- [一阶最优性条件](../optimization-theory/first-order-optimality/)：处理内部驻点、边界和约束方向。
- [二阶最优性条件](../optimization-theory/second-order-optimality/)：系统整理 Hessian 正定、负定、不定和退化情形。
- [梯度](../calculus/gradient/)：解释梯度为零的必要条件与方向导数。
- [Hessian 矩阵](../calculus/hessian/)：从二阶方向导数推导局部曲率判据。
- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：分析步长、光滑性、收敛和停止误差。
- [损失景观](../optimization-theory/loss-landscapes/)：从高维几何和鞍点视角讨论神经网络目标。
- [经验风险最小化](../learning-framework/empirical-risk-minimization/)：区分训练样本上的最优和总体风险。
- [模型选择](../evaluation-and-generalization/model-selection/)：比较验证指标、泛化和模型复杂度。
