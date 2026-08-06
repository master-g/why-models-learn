---
title: "二阶方法：用局部曲率改写更新"
tags: ["why-models-learn"]
---

二阶方法不只看目标函数当前往哪边上升，还用 Hessian 描述附近的曲率，把局部目标近似成一个二次函数，再求这个二次模型的最优步。Newton 方法直接解 Hessian 线性系统，阻尼和信赖域方法限制这一步的风险，拟 Newton 方法则用梯度变化逐步近似曲率。曲率信息能让二次目标一步到位，却也带来矩阵存储、线性求解、奇异 Hessian 和负曲率等问题。本文从局部模型推导 Newton 步，再说明它为什么需要线搜索或信赖域，以及 BFGS、L-BFGS 和 Hessian-向量积怎样把二阶信息压缩到可计算的形式。

## 局部二次模型

在参数 $\boldsymbol\theta_k$ 处记

$$
\boldsymbol g_k=\nabla F(\boldsymbol\theta_k),
\qquad
H_k=\nabla^2F(\boldsymbol\theta_k)
$$

对一个小位移 $\boldsymbol p$ 做 Taylor 展开：

$$
F(\boldsymbol\theta_k+\boldsymbol p)
\approx
F(\boldsymbol\theta_k)
\boldsymbol g_k^\mathsf T\boldsymbol p
+\frac12\boldsymbol p^\mathsf TH_k\boldsymbol p
$$

把右侧称为局部二次模型：

$$
m_k(\boldsymbol p)
=F(\boldsymbol\theta_k)
+\boldsymbol g_k^\mathsf T\boldsymbol p
+\frac12\boldsymbol p^\mathsf TH_k\boldsymbol p
$$

对 $\boldsymbol p$ 求梯度：

$$
\nabla_{\boldsymbol p}m_k(\boldsymbol p)
=\boldsymbol g_k+H_k\boldsymbol p
$$

让模型的一阶导数为零，就得到线性系统

$$
H_k\boldsymbol p_k=-\boldsymbol g_k
$$

如果 $H_k$ 可逆，Newton 步是

$$
\boldsymbol p_k^{\mathrm N}
=-H_k^{-1}\boldsymbol g_k
$$

然后更新

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k+\alpha_k\boldsymbol p_k^{\mathrm N}
$$

这里特意保留 $\alpha_k$。理想的局部区域内可以取 $\alpha_k=1$，离解较远时则通常需要线搜索或阻尼。实际实现也不会真的计算 $H_k^{-1}$，而是直接解 $H_k\boldsymbol p=-\boldsymbol g_k$；求逆既浪费计算，也会放大数值误差。

![二阶方法先拟合局部二次模型，再解线性系统](/assets/optimization-theory/svg/second-order-methods.1.svg)

## 二次目标上 Newton 一步到位

考虑带中心 $\boldsymbol a$ 的正定二次目标：

$$
F(\boldsymbol\theta)
=\frac12(\boldsymbol\theta-\boldsymbol a)^\mathsf T
Q(\boldsymbol\theta-\boldsymbol a),
\qquad
Q\succ0
$$

梯度和 Hessian 分别是

$$
\boldsymbol g
=Q(\boldsymbol\theta-\boldsymbol a),
\qquad
H=Q
$$

Newton 线性系统为

$$
Q\boldsymbol p
=-Q(\boldsymbol\theta-\boldsymbol a)
$$

因为 $Q$ 可逆，直接得到

$$
\boldsymbol p
=-(\boldsymbol\theta-\boldsymbol a)
$$

于是

$$
\boldsymbol\theta+\boldsymbol p
=\boldsymbol a
$$

这不是「通常很快」的说法，而是二次模型和真实目标完全相同，所以一步就达到最小点。

### 二维数值例子

取

$$
Q=
\begin{bmatrix}
1&0\\
0&9
\end{bmatrix},
\qquad
\boldsymbol a=\boldsymbol0,
\qquad
\boldsymbol\theta_0=(1,1)^\mathsf T
$$

则

$$
\boldsymbol g_0
=Q\boldsymbol\theta_0
=(1,9)^\mathsf T
$$

Newton 系统是

$$
\begin{bmatrix}
1&0\\
0&9
\end{bmatrix}
\boldsymbol p_0
=
-\begin{bmatrix}
1\\
9
\end{bmatrix}
$$

所以

$$
\boldsymbol p_0=(-1,-1)^\mathsf T,
\qquad
\boldsymbol\theta_1=(0,0)^\mathsf T
$$

初始函数值为

$$
F(\boldsymbol\theta_0)
=\frac12(1+9)=5
$$

更新后的函数值为 $F(\boldsymbol\theta_1)=0$。普通梯度下降则要选一个同时服从两个特征值的学习率：

$$
\boldsymbol\theta_{k+1}
=(I-\eta Q)\boldsymbol\theta_k
$$

$\eta$ 太大时会被特征值 $9$ 的方向限制，$\eta$ 太小时特征值 $1$ 的方向缓慢。Newton 步把这两个方向分别除以各自的曲率，所以不需要用一个标量学习率折中。

## 非二次目标上的局部加速

Newton 的强项来自局部模型准确，而不是来自一个对任意目标都成立的恒等式。如果最优点 $\boldsymbol\theta^\star$ 附近的 Hessian 连续且可逆，并且 Hessian 的变化足够平滑，完整 Newton 步在足够近时可以达到二次收敛：

$$
\|\boldsymbol\theta_{k+1}-\boldsymbol\theta^\star\|
\leq
C\|\boldsymbol\theta_k-\boldsymbol\theta^\star\|^2
$$

误差平方意味着，误差从 $10^{-2}$ 变到下一步约 $10^{-4}$，再下一步约 $10^{-8}$，前提是已经进入该局部区域。离解较远时，Taylor 余项可能和二次项同样大，完整 Newton 步就可能走到模型不可信的地方。

### 四次函数的反例

看一维目标

$$
f(x)=\frac14x^4
$$

其导数和二阶导数为

$$
f'(x)=x^3,
\qquad
f''(x)=3x^2
$$

对 $x\neq0$，Newton 更新是

$$
x_{k+1}
=x_k-\frac{x_k^3}{3x_k^2}
=\frac23x_k
$$

从 $x_0=3$ 出发，数值轨迹为

| 步骤 | $x_k$ | $f(x_k)$ |
| --- | --- | --- |
| 0 | $3$ | $20.25$ |
| 1 | $2$ | $4$ |
| 2 | $1.3333$ | $0.790123$ |
| 3 | $0.8889$ | $0.156074$ |

误差每次只乘 $2/3$，是线性收敛，不是二次收敛。原因在于最优点 $x^\star=0$ 处 Hessian 也为零，Newton 局部二次收敛需要的非奇异条件不成立。即使目标函数很光滑，也不能只看到「用了二阶导数」就断言会二次收敛。

## 线搜索让 Newton 步先证明自己

如果 Newton 方向满足

$$
\boldsymbol g_k^\mathsf T\boldsymbol p_k^{\mathrm N}<0
$$

它是当前点的下降方向。线搜索从 $\alpha=1$ 开始，逐步缩小 $\alpha$，直到满足 Armijo 条件：

$$
F(\boldsymbol\theta_k+\alpha\boldsymbol p_k)
\leq
F(\boldsymbol\theta_k)
+c\alpha\boldsymbol g_k^\mathsf T\boldsymbol p_k,
\qquad
0<c<1
$$

右侧是当前梯度对函数下降的线性预测，条件要求实际下降至少达到预测的一定比例。这样做的结果是：局部模型足够准时保留完整 Newton 步，模型不准时用较短的步留在可信区域。

### 阻尼 Newton

另一种做法是在 Hessian 上加正则：

$$
(H_k+\lambda_k I)\boldsymbol p_k
=-\boldsymbol g_k,
\qquad
\lambda_k\geq0
$$

当 $\lambda_k$ 足够大，使 $H_k+\lambda_kI$ 正定时

$$
\boldsymbol g_k^\mathsf T\boldsymbol p_k
=-\boldsymbol g_k^\mathsf T
(H_k+\lambda_kI)^{-1}\boldsymbol g_k
<0
$$

所以这一步一定是下降方向。远离解时取较大的 $\lambda_k$，接近解时逐渐减小它，就能在梯度法的保守性和 Newton 的快速局部收敛之间移动。

仍以 $f(x)=x^4/4$ 为例，在 $x=1$ 处 $f'(1)=1$、$f''(1)=3$。取 $\lambda=0.1$：

$$
(3+0.1)p=-1,
\qquad
p=-0.3225806452
$$

于是新点为

$$
x_1=1+p=0.6774193548
$$

没有阻尼时这一步是 $p=-1/3$。差别不大，但在 Hessian 接近奇异或不定时，阻尼会显著改变方向和长度。

## 不定 Hessian 不能直接当作最小化方向

Newton 方程只是在寻找二次模型的驻点。若 Hessian 不正定，那个驻点可能是最大点或鞍点。

考虑

$$
F(x,y)=\frac12(x^2-y^2)
$$

在 $(1,1)$ 处

$$
\boldsymbol g=(1,-1)^\mathsf T,
\qquad
H=
\begin{bmatrix}
1&0\\
0&-1
\end{bmatrix}
$$

解 Newton 方程得到

$$
\boldsymbol p
=-H^{-1}\boldsymbol g
=(-1,-1)^\mathsf T
$$

一步会到 $(0,0)$。但 $(0,0)$ 是鞍点，初始和更新后的函数值都为 $0$，Newton 方程并没有告诉我们「这是要找的局部最小值」。

实际算法需要检查 Hessian 的正定性，或显式处理负曲率：

- 对 $H+\lambda I$ 增加足够阻尼，使最小特征值变为正；
- 用 Cholesky 分解检测正定性，失败时改用带修正的分解；
- 在信赖域内寻找不会沿负曲率无限下降的步；
- 对非凸问题保留负曲率方向，把它作为逃离鞍点的候选方向。

Hessian 奇异时也不能直接求逆。四次函数在 $x=0$ 的 Hessian 为 $0$，就是最简单的例子。阻尼、伪逆或信赖域可以给出一个有界子问题，但它们代表的是修改后的方法，不再是未经处理的 Newton 步。

## 信赖域：先限制步长，再比较模型

信赖域方法不先假定完整 Newton 步可靠，而是只允许

$$
\|\boldsymbol p\|\leq\Delta_k
$$

然后求或近似求解

$$
\min_{\|\boldsymbol p\|\leq\Delta_k}
m_k(\boldsymbol p)
$$

得到候选步后，比较真实下降和模型预测下降：

$$
\rho_k
=
\frac{
F(\boldsymbol\theta_k)-F(\boldsymbol\theta_k+\boldsymbol p_k)
}{
m_k(\boldsymbol0)-m_k(\boldsymbol p_k)
}
$$

$\rho_k$ 接近 $1$ 表示模型预测得好；$\rho_k$ 很小或为负，表示模型在这一步不可信。算法可以据此接受或拒绝候选步，并缩小或扩大 $\Delta_k$。

### 一维数值例子

取

$$
f(x)=5x^2,
\qquad
x_0=1,
\qquad
f'(x_0)=10,
\qquad
f''(x_0)=10
$$

无约束 Newton 步为 $p=-1$。如果信赖半径为 $\Delta=0.25$，先取可行候选步 $p=-0.25$：

$$
f(x_0)=5,
\qquad
f(x_0+p)=f(0.75)=2.8125
$$

这个目标本身就是二次函数，所以模型预测值也为 $2.8125$，从而

$$
\rho
=\frac{5-2.8125}{5-2.8125}
=1
$$

模型可靠时可以扩大信赖半径；如果真实下降远小于预测，就先缩小半径，而不是盲目执行完整 Newton 步。

## 拟 Newton：用梯度变化补曲率

显式形成 Hessian 的成本很高，但相邻两点的梯度差包含局部曲率信息。定义

$$
\boldsymbol s_k
=\boldsymbol\theta_{k+1}-\boldsymbol\theta_k,
\qquad
\boldsymbol y_k
=\boldsymbol g_{k+1}-\boldsymbol g_k
$$

如果目标在这段位移附近近似二次，则

$$
\boldsymbol y_k
\approx H\boldsymbol s_k
$$

因此可以要求 Hessian 近似 $B_{k+1}$ 满足割线条件

$$
B_{k+1}\boldsymbol s_k=\boldsymbol y_k
$$

拟 Newton 方法不重新计算完整 Hessian，而是让近似矩阵满足不断累积的割线信息。BFGS 通常直接维护逆 Hessian 近似 $M_k\approx H_k^{-1}$。令

$$
\rho_k=\frac1{\boldsymbol y_k^\mathsf T\boldsymbol s_k}
$$

标准 BFGS 逆更新为

$$
M_{k+1}
=
(I-\rho_k\boldsymbol s_k\boldsymbol y_k^\mathsf T)
M_k
(I-\rho_k\boldsymbol y_k\boldsymbol s_k^\mathsf T)
+\rho_k\boldsymbol s_k\boldsymbol s_k^\mathsf T
$$

只要

$$
\boldsymbol y_k^\mathsf T\boldsymbol s_k>0
$$

且 $M_k$ 正定，更新通常保持正定，这样 $-\!M_k\boldsymbol g_k$ 是下降方向。

### 一步 BFGS 数值例子

取

$$
M_0=I,
\qquad
\boldsymbol s=(-0.1,-0.1)^\mathsf T,
\qquad
\boldsymbol y=(-0.1,-0.9)^\mathsf T
$$

有

$$
\boldsymbol y^\mathsf T\boldsymbol s
=0.01+0.09=0.1,
\qquad
\rho=10
$$

把它们代入标准更新式，得到

$$
M_1=
\begin{bmatrix}
1.72&-0.08\\
-0.08&0.12
\end{bmatrix}
$$

它不等于真实 Hessian 的逆，却已经把一次「位移—梯度变化」关系编码进去。BFGS 需要存储一个 $d\times d$ 矩阵；L-BFGS 只保存最近 $m$ 对 $(\boldsymbol s_k,\boldsymbol y_k)$，用两层循环计算 $M_k\boldsymbol g_k$，所以存储从 $O(d^2)$ 降到 $O(md)$。

如果 $\boldsymbol y_k^\mathsf T\boldsymbol s_k\leq0$，可能是目标非凸、线搜索不满足曲率条件，或梯度噪声太大。实现通常跳过这次更新、修正 $\boldsymbol y_k$，或重新加入阻尼，而不是让 $\rho_k$ 直接变成异常的大数。

## Hessian-向量积与不完全 Newton

二阶方法不一定需要把 Hessian 的所有元素都存下来。给定方向 $\boldsymbol v$：

$$
H(\boldsymbol\theta)\boldsymbol v
=
\left.
\frac{\mathrm d}{\mathrm d\epsilon}
\nabla F(\boldsymbol\theta+\epsilon\boldsymbol v)
\right|_{\epsilon=0}
$$

自动微分可以直接计算这个 Hessian-向量积。若 $H$ 正定，可以用共轭梯度法近似求解

$$
H\boldsymbol p=-\boldsymbol g
$$

而不显式形成 $H$。不完全 Newton 只要求线性系统残差达到某个相对精度：

$$
\|H\boldsymbol p+\boldsymbol g\|
\leq
\zeta_k\|\boldsymbol g\|,
\qquad
0\leq\zeta_k<1
$$

早期离解较远时允许粗略求解，后期再提高线性求解精度。这样把「每一步都精确解一个巨大系统」改成了根据当前误差分配计算量。

| 方法 | 主要曲率信息 | 典型存储 | 主要代价 |
| --- | --- | --- | --- |
| Newton | 显式 Hessian 与线性系统 | $O(d^2)$ | 形成和分解矩阵 |
| Hessian-向量积 | 方向上的曲率作用 | 可接近 $O(d)$ | 多次 HVP 和迭代求解 |
| BFGS | 全矩阵拟合的割线信息 | $O(d^2)$ | 矩阵更新和乘法 |
| L-BFGS | 最近若干割线对 | $O(md)$ | 两层循环应用近似逆 |

在神经网络中，参数维数 $d$ 很大，完整 Hessian 的平方级存储通常不可接受。Hessian-向量积、阻尼和 L-BFGS 因而比教科书上的显式 Newton 更实际，但每个选项仍有自己的时间和噪声成本。

## 随机梯度下的曲率信息

用小批次估计梯度时，Hessian 也常由小批次估计：

$$
\widehat{\boldsymbol g}_k
=\boldsymbol g_k+\boldsymbol\xi_k,
\qquad
\widehat H_k
=H_k+E_k
$$

$\boldsymbol\xi_k$ 和 $E_k$ 都会随批次变化。线性系统对矩阵误差敏感，尤其当 $H_k$ 接近奇异时，微小的 $E_k$ 也可能大幅改变步长。负曲率在非凸神经网络中不是异常事件，Cholesky 失败并不意味着代码必然错误。

常见应对方式包括增大批次、给 Hessian 加阻尼、限制信赖半径、用截断共轭梯度避免沿不稳定方向走太远，以及只在局部子问题上使用二阶信息。它们都在交换单步精度、计算成本和噪声敏感性。

## 方法选择和失效模式

| 情况 | 更合适的起点 | 需要提防 |
| --- | --- | --- |
| 参数很多、梯度便宜 | SGD 或自适应一阶方法 | 收敛速度受条件数影响 |
| 小到中等规模、Hessian 可解 | 阻尼 Newton | 矩阵正定性和分解成本 |
| Hessian 可做 HVP、不能存全矩阵 | 不完全 Newton | 线性系统迭代次数 |
| 梯度可靠、希望少存曲率 | L-BFGS | 割线条件被噪声破坏 |
| 目标明显非凸 | 信赖域或带负曲率处理 | Newton 驻点可能是鞍点 |

**离局部区域太远。** Taylor 模型只在附近可信，完整 Newton 步可能穿过高曲率区域。先做线搜索或缩小信赖域。

**Hessian 不定。** Newton 方程求到的是模型驻点，不是模型最小点。检查正定性，或使用阻尼、信赖域和负曲率方向。

**Hessian 奇异或条件数很大。** 直接求逆会放大舍入误差和梯度噪声。改解线性系统，加阻尼，或使用适当的预条件器。

**线性系统解得不够准确。** 迭代过早停止会让方向不再下降，迭代过久又把大部分时间花在线性代数上。让残差阈值随优化阶段变化。

**把拟 Newton 矩阵当成真实曲率。** BFGS 的矩阵只满足有限个割线条件，它是对历史方向的模型，不是当前点完整 Hessian 的身份替代。

**只比较迭代次数。** Newton 一步可能比 SGD 一步贵很多。应同时比较函数/梯度评估次数、线性求解时间、内存和最终精度。

## 相关词条

- [二阶最优性条件](../optimization-theory/second-order-optimality/)：正定、半正定和不定 Hessian 如何决定局部形状。
- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：一阶更新的下降引理和条件数限制。
- [自适应学习率](../optimization-theory/adaptive-learning-rates/)：用梯度历史构造对角预条件器。
- [动量理论](../optimization-theory/momentum-theory/)：不显式使用 Hessian 的历史方向方法。
- [Hessian 矩阵](../calculus/hessian/)：二阶导数组成的曲率矩阵。
- Hessian-向量积：只计算 Hessian 作用在方向上的结果。
- [曲率与条件数](../optimization-theory/curvature-and-conditioning/)：病态方向如何限制一阶和二阶更新。
- [二阶最优性条件](../optimization-theory/second-order-optimality/)：说明信赖域和负曲率处理面对的局部曲率。
- [约束优化](../optimization-theory/constrained-optimization/)：把二阶模型扩展到可行域和约束线性系统。
- [损失景观](../optimization-theory/loss-landscapes/)：非凸目标中的鞍点和负曲率。
- 拟 Newton 方法：用割线信息构造 BFGS 类算法。
