---
title: "曲率与条件数：梯度下降为什么会被最陡方向拖慢"
tags: ["why-models-learn"]
---

曲率是目标函数沿某个方向的二阶变化率，条件数则把相关方向中最大的曲率和最小的正曲率放在一起比较。曲率谱决定局部像碗、脊还是鞍点；条件数决定同一个固定步长要面对多大的时间尺度差异。对梯度下降来说，最陡方向限制稳定步长，最平缓方向决定剩下的误差要消失多久。本文从方向曲率和 Hessian 谱出发，推导二次目标的稳定区间与最佳固定步长，再说明坐标缩放、预条件化、白化和非二次景观中的条件数该怎样解释。

![曲率谱把陡峭方向、平缓方向和预条件化放在一起](/assets/optimization-theory/svg/curvature-and-conditioning.1.svg)

## 曲率先问方向

一元函数在 $x$ 附近的二阶展开是

$$
f(x+h)
\approx
f(x)+f'(x)h+\frac12f''(x)h^2
$$

其中 $f''(x)$ 才是曲率；$f'(x)$ 仍然是一阶斜率。多元函数沿单位方向 $\boldsymbol u$ 走，定义一元剖面

$$
\phi(t)=f(\boldsymbol x+t\boldsymbol u),
\qquad
\lVert\boldsymbol u\rVert_2=1
$$

则

$$
\phi'(0)=\nabla f(\boldsymbol x)^\mathsf T\boldsymbol u,
\qquad
\phi''(0)=\boldsymbol u^\mathsf T
H(\boldsymbol x)\boldsymbol u
$$

第二个量称为方向曲率。它是一个标量，不是 Hessian 的某一个坐标元素；交叉偏导也会通过二次型影响方向曲率。

若 $f$ 二阶连续可微，Hessian 是对称矩阵。对它做谱分解：

$$
H=Q\Lambda Q^\mathsf T,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_d)
$$

把方向写成特征向量的线性组合 $\boldsymbol u=\sum_i z_i\boldsymbol q_i$，且 $\sum_i z_i^2=1$，则

$$
\boldsymbol u^\mathsf T H\boldsymbol u
=\sum_{i=1}^{d}\lambda_i z_i^2
$$

方向曲率是各个特征值按平方权重的平均。因此，特征值告诉我们局部可能遇到的曲率范围；特征向量告诉我们这些曲率分别朝向哪里。

| 特征值情况 | 局部几何 | 对最小化的含义 |
| --- | --- | --- |
| 全部为正 | 向上弯的碗 | 局部有正定曲率 |
| 有正有负 | 鞍形 | 存在上升和下降方向 |
| 有零但无负值 | 平台或等价方向 | 二阶信息不足以保证严格最小 |
| 全部为负 | 向下弯的盖子 | 对最小化是局部最大证据 |
| 正负混合且接近零 | 狭窄又近乎平坦 | 数值估计和步长更敏感 |

在驻点 $\nabla f(\boldsymbol x^\star)=\boldsymbol0$ 附近，Taylor 展开中的一阶项消失：

$$
f(\boldsymbol x^\star+\boldsymbol\delta)
\approx
f(\boldsymbol x^\star)
+\frac12\boldsymbol\delta^\mathsf T
H(\boldsymbol x^\star)\boldsymbol\delta
$$

这时曲率谱主导局部形状；若梯度还不为零，则不能只看 Hessian 的正负来判断当前点是不是最小。

## 最大和最小曲率分别限制什么

先看正定二次目标

$$
f(\boldsymbol\theta)
=\frac12(\boldsymbol\theta-\boldsymbol\theta^\star)^\mathsf T
Q(\boldsymbol\theta-\boldsymbol\theta^\star)
$$

其中 $Q$ 对称正定。令误差 $\boldsymbol e_k=\boldsymbol\theta_k-\boldsymbol\theta^\star$，梯度下降变成

$$
\boldsymbol e_{k+1}
=(I-\eta Q)\boldsymbol e_k
$$

在特征方向 $\boldsymbol q_i$ 上，每一步只做一个标量缩放：

$$
e_{i,k+1}
=(1-\eta\lambda_i)e_{i,k}
$$

要让每个正特征方向都衰减，需要

$$
\lvert1-\eta\lambda_i\rvert<1
\quad\text{对所有 }i
$$

这等价于

$$
0<\eta<\frac2{L},
\qquad
L=\lambda_{\max}(Q)
$$

所以最大曲率 $L$ 先决定稳定窗口的上界。最小正曲率 $\mu=\lambda_{\min}(Q)$ 则决定最慢方向的衰减速度。若 $0<\mu\le L$，条件数定义为

$$
\kappa=\frac{L}{\mu}
$$

下面取 $Q=\operatorname{diag}(1,100)$。两个方向的更新因子分别为 $1-\eta$ 和 $1-100\eta$：

| 学习率 $\eta$ | 平缓方向因子 | 陡峭方向因子 | 结果 |
| --- | --- | --- | --- |
| $0.01$ | $0.99$ | $0$ | 稳定，陡峭方向一步消失 |
| $2/101\approx0.019802$ | $0.980198$ | $-0.980198$ | 稳定但陡峭方向交替变号 |
| $0.021$ | $0.979$ | $-1.1$ | 陡峭方向发散 |

第二行的步长已经接近稳定上界 $2/100=0.02$，所以陡峭方向虽然幅度不再增长，符号却每轮翻转。第一行没有振荡，但在平缓方向上每一步只保留 99％ 的误差。一个数字同时服务两个方向，条件数为 100 的代价就在这里出现。

## 最佳固定步长由两端曲率折中

稳定不等于快。对误差范数的最坏收缩因子，应该考虑所有特征方向：

$$
\rho(\eta)
=\max_{\lambda\in[\mu,L]}\lvert1-\eta\lambda\rvert
$$

在正定二次问题中，区间两端足以决定这个最大值：

$$
\rho(\eta)
=\max\bigl\{\lvert1-\eta\mu\rvert,
\lvert1-\eta L\rvert\bigr\}
$$

最佳固定步长让最慢方向和最陡方向的误差因子大小相同、符号相反：

$$
1-\eta^\star\mu
=-\bigl(1-\eta^\star L\bigr)
$$

解得

$$
\eta^\star=\frac2{L+\mu},
\qquad
\rho^\star
=\frac{L-\mu}{L+\mu}
=\frac{\kappa-1}{\kappa+1}
$$

这不是凭经验调出来的学习率，而是对固定步长、正定二次目标和最坏特征方向的一个精确折中。

| 曲率范围 $(\mu,L)$ | 条件数 $\kappa$ | 最佳步长 $\eta^\star$ | 最坏收缩 $\rho^\star$ |
| --- | --- | --- | --- |
| $(1,1)$ | $1$ | $1$ | $0$ |
| $(1,9)$ | $9$ | $0.2$ | $0.8$ |
| $(1,100)$ | $100$ | $2/101\approx0.019802$ | $99/101\approx0.980198$ |

当 $\kappa=1$ 时，所有方向曲率相同，最佳步长一步消除误差；当 $\kappa=100$ 时，即使已经选到最佳固定步长，最坏方向每轮仍保留约 98％ 的误差。达到误差比例 $\varepsilon$ 所需的轮数满足

$$
(\rho^\star)^k\le\varepsilon
\quad\Longrightarrow\quad
k\ge\frac{\log(1/\varepsilon)}{\log(1/\rho^\star)}
$$

当 $\kappa$ 很大时，$\log(1/\rho^\star)$ 约为 $2/(\kappa+1)$，所以迭代次数的主要尺度与 $\kappa$ 成正比。

## 等值线的轴比是条件数的平方根

对二次目标

$$
f(\boldsymbol x)=\frac12\boldsymbol x^\mathsf TQ\boldsymbol x
$$

等值面 $f(\boldsymbol x)=c$ 在二维特征坐标中满足

$$
\lambda_1 z_1^2+\lambda_2 z_2^2=2c
$$

沿第 $i$ 个特征方向的半轴长度为

$$
r_i=\sqrt{\frac{2c}{\lambda_i}}
$$

因此在两个正曲率方向之间，等值线长短轴之比是

$$
\frac{r_{\max}}{r_{\min}}
=\sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}
=\sqrt{\kappa}
$$

$Q=\operatorname{diag}(1,100)$ 的等值线轴比是 10，而条件数是 100。图上看起来只是「细长十倍」，梯度下降的最坏迭代尺度却由 100 这一比值控制；几何长宽比和算法时间尺度不是同一个数字。

## 预条件化是在换一把尺子

可以给梯度先做一个对称正定的尺度变换：

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k
-\eta M\nabla f(\boldsymbol\theta_k),
\qquad
M\succ0
$$

对二次目标，令 $\boldsymbol z=M^{-1/2}\boldsymbol e$，更新在新坐标中由

$$
\widetilde Q=M^{1/2}QM^{1/2}
$$

控制。预条件器的目标不是让原始 Hessian 消失，而是让 $\widetilde Q$ 的特征值更集中。

对 $Q=\operatorname{diag}(1,100)$，取

$$
M=\operatorname{diag}\left(1,\frac1{100}\right)
$$

就有

$$
MQ=I,
\qquad
\widetilde Q=I
$$

预条件后的两个方向使用同一个更新因子。以 $\eta=0.5$、初始误差 $(1,1)$ 为例：

| 步骤 | 误差 $\boldsymbol e_k$ | $f(\boldsymbol e_k)$ |
| --- | --- | --- |
| $0$ | $(1,1)$ | $50.5$ |
| $1$ | $(0.5,0.5)$ | $12.625$ |
| $2$ | $(0.25,0.25)$ | $3.15625$ |
| $3$ | $(0.125,0.125)$ | $0.7890625$ |

这里参数的两个坐标都按 $0.5$ 缩小，目标值也按 $0.25$ 缩小。原始梯度下降若用同样的 $\eta=0.5$ 会在 $y$ 方向直接发散，因为它的更新因子是 $1-50=-49$；预条件器把这个方向的梯度缩小了 100 倍。

对角预条件器只能处理坐标尺度差异。若曲率的主方向发生旋转，交叉项仍然存在。设

$$
R=\frac1{\sqrt2}
\begin{bmatrix}
1&-1\\
1&1
\end{bmatrix},
\qquad
Q=R
\begin{bmatrix}
1&0\\
0&100
\end{bmatrix}
R^\mathsf T
=
\begin{bmatrix}
50.5&-49.5\\
-49.5&50.5
\end{bmatrix}
$$

它的特征值仍然是 $1$ 和 $100$，但曲率方向是 $(1,1)$ 与 $(1,-1)$，不是坐标轴。逐坐标缩放不能完全消除这类相关性；要做完整白化，需要估计旋转方向和尺度。

## 梯度流把条件数变成时间常数

把离散梯度下降的步长趋近于零，可以得到梯度流：

$$
\frac{\mathrm d\boldsymbol\theta}{\mathrm dt}
=-\nabla f(\boldsymbol\theta)
$$

在二次目标的特征方向上，误差满足

$$
\frac{\mathrm de_i(t)}{\mathrm dt}
=-\lambda_i e_i(t)
$$

解为

$$
e_i(t)=e_i(0)e^{-\lambda_i t}
$$

曲率为 $\lambda_i$ 的方向的时间常数是 $1/\lambda_i$。对 $(1,100)$ 两个方向，平缓方向的时间常数是 1，陡峭方向是 $0.01$；陡峭方向早已接近零时，平缓方向仍然保留主要误差。离散法把指数衰减换成因子 $1-\eta\lambda_i$，所以步长过大时会把连续时间的衰减变成离散振荡甚至发散。

这也解释了为什么损失曲线有时开头下降很快，后面却拖着长尾：快方向先被清掉，剩下的是小特征值对应的慢模态。只看前几轮的函数值下降，不能估计最终到达精度所需的时间。

## 数据尺度会把条件数带进训练

在线性最小二乘中，目标可以写成

$$
f(\boldsymbol w)
=\frac1{2n}\lVert X\boldsymbol w-\boldsymbol y\rVert_2^2
$$

其 Hessian 是

$$
H=\frac1nX^\mathsf TX
$$

因此输入特征的尺度和相关性直接进入优化曲率。若两个标准化特征的协方差矩阵为

$$
C=
\begin{bmatrix}
1&0.9\\
0.9&1
\end{bmatrix}
$$

它的特征值是 $1.9$ 和 $0.1$，条件数为 19。两个特征都各自有单位方差，却仍然因为高度相关而产生狭长谷。

白化用 $C^{-1/2}$ 变换特征：

$$
\boldsymbol z=C^{-1/2}\boldsymbol x
$$

理想情况下，新特征的协方差为

$$
\mathbb E[\boldsymbol z\boldsymbol z^\mathsf T]
=C^{-1/2}CC^{-1/2}
=I
$$

特征方向被去相关，曲率尺度更接近。实际数据中的协方差是估计量，白化也可能放大估计误差；它不是无条件的修复。按列标准化只能消除单坐标的单位差异，不能消除相关特征带来的旋转。

| 条件数来源 | 典型表现 | 可尝试的处理 |
| --- | --- | --- |
| 特征单位不同 | 某坐标曲率远大于其他坐标 | 标准化或按尺度预条件 |
| 特征高度相关 | Hessian 有明显交叉项 | 白化、旋转或更完整的预条件器 |
| 模型参数冗余 | Hessian 有零或很小特征值 | 重新参数化、正则化或接受非唯一解 |
| 非凸区域 | Hessian 同时有正负特征值 | 分开报告负曲率，不能只报条件数 |

## 非二次函数要使用局部条件数

对一般损失，Hessian 会随着参数变化。若在某个点的相关曲率全为正，可以定义局部量

$$
\mu_{\mathrm{loc}}(\boldsymbol\theta)
=\lambda_{\min}\bigl(H(\boldsymbol\theta)\bigr),
\qquad
L_{\mathrm{loc}}(\boldsymbol\theta)
=\lambda_{\max}\bigl(H(\boldsymbol\theta)\bigr)
$$

以及

$$
\kappa_{\mathrm{loc}}(\boldsymbol\theta)
=\frac{L_{\mathrm{loc}}(\boldsymbol\theta)}
{\mu_{\mathrm{loc}}(\boldsymbol\theta)}
$$

这三个量只描述当前点附近的二阶近似。若最小特征值接近 0，局部条件数会变得很大；若出现负特征值，则它已经不是正定谷，不能把负数直接放进条件数的分母。此时应报告最小特征值、最大特征值和负特征值数量，先说明局部几何。

在一个较大的邻域上，若能找到统一界

$$
\mu I\preceq H(\boldsymbol\theta)\preceq LI
$$

并且 $\mu>0$，才可以把 $\kappa=L/\mu$ 当作这个邻域的条件数上界。只在一个训练终点测得的 Hessian，不足以证明整条训练轨迹都满足同一个收敛率。

神经网络中还要区分全量 Hessian、小批次 Hessian、Gauss–Newton 矩阵和 Fisher 信息。它们都可以提供某种曲率近似，但来源和正负性不同。一个近似矩阵条件数良好，不代表原始损失在所有方向上都良好；应先写清楚测量的是哪个矩阵。

## 失效模式

**把负特征值塞进条件数。** 条件数的这个比值要求正定或至少在指定子空间上有正的曲率。遇到鞍点，应报告负曲率，而不是把它取绝对值后继续解释成收敛速度。

**只用最大特征值调步长。** $L$ 能给出稳定上界，却不能告诉你慢方向要多久。还要报告 $\mu$ 或条件数，才能解释长尾。

**把等值线轴比当作迭代次数。** 轴比是 $\sqrt\kappa$，最佳固定步长的最坏收缩率由 $(\kappa-1)/(\kappa+1)$ 决定，二者相关但不相等。

**把对角缩放当成完整白化。** 对角方法只能改变每个坐标的单位，不能去掉旋转后的交叉项。遇到强相关特征，要检查特征向量而不只是看对角元素。

**把局部条件数当成全局定理。** 非二次损失的 Hessian 会沿轨迹变化；必须说明估计点、数据集、批次和谱估计方法。

**忽略接近零的特征值和有限精度。** 当曲率接近数值噪声时，正负号和条件数都可能不稳定。应给出容差、估计误差和是否做了对称化。

**只报告参数空间曲率。** 重参数化能改变 Hessian 数字；训练集曲率也不等于验证集或总体风险曲率。比较模型时要固定坐标、损失归一化和扰动规则。

## 相关词条

- [二次型](../linear-algebra/quadratic-forms/)：把曲率写成矩阵二次型。
- [谱定理](../linear-algebra/spectral-theorem/)：解释对称 Hessian 的正交谱分解。
- [特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/)：把曲率和主方向对应起来。
- [Hessian](../calculus/hessian/)：从二阶偏导构造局部曲率矩阵。
- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：推导学习率稳定范围和收敛界。
- [自适应学习率](../optimization-theory/adaptive-learning-rates/)：讨论逐坐标预条件与其边界。
- [二阶方法](../optimization-theory/second-order-methods/)：处理局部二次模型、阻尼和曲率近似。
- [损失景观](../optimization-theory/loss-landscapes/)：从高维图形角度观察谷、平坦方向和障碍。
- [白化](../linear-algebra/eigendecomposition/)：继续讨论协方差去相关与尺度变换。
*** End Patch
