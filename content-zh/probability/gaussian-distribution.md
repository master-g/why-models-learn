---
title: "高斯分布：钟形密度与标准化坐标"
tags: ["why-models-learn"]
---

**高斯分布**用两个参数描述一条对称的钟形密度：均值 $\mu$ 决定中心位置，标准差 $\sigma$ 决定横向尺度。写作 $X\sim\mathcal N(\mu,\sigma^2)$，其中 $\sigma>0$，第二个参数是方差而不是标准差。本篇从密度的归一化常数和标准正态变量开始，计算覆盖概率与尾概率，再说明仿射变换、独立高斯变量求和、多元高斯和机器学习中的平方误差为何会出现。

## 高斯密度的形状

一维高斯分布的概率密度是

$$
f_X(x)
=\frac1{\sigma\sqrt{2\pi}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$$

其中

$$
X\sim\mathcal N(\mu,\sigma^2),\qquad
\sigma>0
$$

指数中的平方距离

$$
\frac{(x-\mu)^2}{\sigma^2}
$$

决定了离开中心多少个标准差。距离中心相同的两点密度相同：

$$
f_X(\mu-r)=f_X(\mu+r)
$$

所以 $\mu$ 是对称中心，也是密度的峰值位置。高斯分布的中位数、众数和均值都等于 μ，但这三个概念在一般分布中并不必然相同。

指数会随 $|x-\mu|$ 增大而快速衰减，却不会在有限位置突然变成 0。高斯变量的支持集是整个实数轴；「大于几倍标准差」只是尾部事件很小，不是数学上的不可能。

## 为什么常数是 $\sqrt{2\pi}$

密度前面的常数必须让总面积等于 1。关键积分是

$$
I=\int_{-\infty}^{\infty}e^{-x^2/2}\,dx
$$

直接求 I 不容易，但平方后可以把两个一维积分写成平面上的面积：

$$
I^2
=\iint_{\mathbb R^2}
e^{-(x^2+y^2)/2}\,dx\,dy
$$

换成极坐标 $x=r\cos\theta$、$y=r\sin\theta$，面积元素是 $r\,dr\,d\theta$：

$$
\begin{aligned}
I^2
&=\int_0^{2\pi}\int_0^\infty
e^{-r^2/2}r\,dr\,d\theta\\
&=2\pi\left[-e^{-r^2/2}\right]_0^\infty\\
&=2\pi
\end{aligned}
$$

所以 $I=\sqrt{2\pi}$。平移 $x\mapsto x-\mu$ 不改变总面积，缩放 $x\mapsto(x-\mu)/\sigma$ 会把横向长度放大 σ；因此需要除以 σ：

$$
\int_{-\infty}^{\infty}
\frac1{\sigma\sqrt{2\pi}}
e^{-(x-\mu)^2/(2\sigma^2)}\,dx
=1
$$

归一化常数不是为了让曲线看起来像钟，而是为了让密度真正定义一个概率分布。

## 标准化：把任意高斯变成 Z

令

$$
Z=\frac{X-\mu}{\sigma}
$$

则

$$
Z\sim\mathcal N(0,1)
$$

Z 叫标准正态变量。反过来

$$
X=\mu+\sigma Z
$$

所以一次高斯观测可以拆成两个部分：

1. 先从固定的标准钟形分布抽取一个无单位的 z；
2. 把它乘以 σ，再平移 μ。

标准化分数

$$
z=\frac{x-\mu}{\sigma}
$$

告诉我们观测 x 离均值有多少个标准差。$z=0$ 在中心，$z=2$ 在均值右侧两个标准差处，$z=-1$ 在左侧一个标准差处。

标准正态的 CDF 记作

$$
\Phi(z)=P(Z\leq z)
$$

一般没有初等函数形式，通常通过数值表、误差函数或数值库计算。任意高斯变量的 CDF 可由标准化得到：

$$
F_X(x)
=P(X\leq x)
=\Phi\left(\frac{x-\mu}{\sigma}\right)
$$

因此区间概率和尾概率都可以转成 Φ 的差或补集。

## 68–95–99.7 经验法则

标准正态的对称性给出

$$
P(|Z|\leq a)=\Phi(a)-\Phi(-a)
=2\Phi(a)-1
$$

几个常用覆盖率为

| 范围 | 标准化事件 | 概率 |
| --- | --- | ---: |
| 均值 ± 1σ | $\lvert Z\rvert\leq1$ | $0.682689$ |
| 均值 ± 2σ | $\lvert Z\rvert\leq2$ | $0.954500$ |
| 均值 ± 3σ | $\lvert Z\rvert\leq3$ | $0.997300$ |

这就是 68–95–99.7 规则。它是高斯分布的精确数值近似，不是任意分布都满足的概率公理。

若

$$
X\sim\mathcal N(10,2^2)
$$

则 $8\leq X\leq12$ 等价于 $|Z|\leq1$：

$$
P(8\leq X\leq12)
=P(-1\leq Z\leq1)
\approx0.682689
$$

而 $X\geq14$ 对应 $Z\geq2$：

$$
P(X\geq14)
=P(Z\geq2)
=1-\Phi(2)
\approx0.0227501
$$

标准差的单位和数值都会随变量改变，z 分数则无量纲，适合比较不同尺度的观测。

![高斯密度的标准差覆盖区间与钟形曲线](/assets/probability/svg/gaussian-distribution.1.svg)

## 仿射变换如何改变参数

若

$$
Y=aX+b
$$

且 $a\neq0$，则 Y 仍然是高斯分布：

$$
Y\sim\mathcal N(a\mu+b,\ a^2\sigma^2)
$$

均值的变换直接是

$$
\mathbb E[Y]=a\mu+b
$$

方差中的 $b$ 消失，因为平移不改变离散程度；a 会被平方，因为方差衡量的是平方距离：

$$
\operatorname{Var}(Y)=a^2\operatorname{Var}(X)=a^2\sigma^2
$$

例如 $X\sim\mathcal N(0,1)$ 时，$Y=3X+2$ 服从

$$
Y\sim\mathcal N(2,9)
$$

标准差从 1 变成 3，而不是变成 9；9 是方差。

若 a<0，分布仍然高斯，钟形曲线只是在中心 b 附近左右翻转；由于原分布对称，翻转不会改变形状。一般非线性变换则未必保留高斯形状。

## 独立高斯变量相加仍是高斯

设 X、Y 独立：

$$
X\sim\mathcal N(\mu_X,\sigma_X^2),\qquad
Y\sim\mathcal N(\mu_Y,\sigma_Y^2)
$$

则

$$
X+Y\sim
\mathcal N\left(
\mu_X+\mu_Y,\,
\sigma_X^2+\sigma_Y^2
\right)
$$

均值相加来自线性性。方差相加需要独立性：

$$
\operatorname{Var}(X+Y)
=\operatorname{Var}(X)+\operatorname{Var}(Y)
 +2\operatorname{Cov}(X,Y)
$$

独立时协方差为 0。

一个更短的推导使用特征函数。高斯变量 X 的特征函数为

$$
\varphi_X(t)
=\mathbb E[e^{itX}]
=\exp\left(i\mu_Xt-\frac12\sigma_X^2t^2\right)
$$

独立变量的和把特征函数相乘：

$$
\begin{aligned}
\varphi_{X+Y}(t)
&=\varphi_X(t)\varphi_Y(t)\\
&=\exp\left(
i(\mu_X+\mu_Y)t
-\frac12(\sigma_X^2+\sigma_Y^2)t^2
\right)
\end{aligned}
$$

这正是均值和方差相加后的高斯特征函数。

取

$$
X\sim\mathcal N(2,1),\qquad
Y\sim\mathcal N(3,4)
$$

则

$$
X+Y\sim\mathcal N(5,5)
$$

注意第二个参数 5 是方差。差 $X-Y$ 也服从高斯分布：

$$
X-Y\sim\mathcal N(-1,5)
$$

因为减法只改变均值符号，独立方差仍然相加。

若变量不独立，和仍可能是高斯，但方差要保留协方差项。若两个变量各自高斯却存在非线性依赖，它们的和甚至不一定高斯；「边缘都高斯」不足以推出「任意组合都高斯」。

## 多元高斯：用协方差矩阵控制椭圆

令 $\mathbf X\in\mathbb R^d$，均值向量为 $\boldsymbol\mu$，协方差矩阵为 Σ。若 Σ 正定，多元高斯密度为

$$
f_{\mathbf X}(\mathbf x)
=\frac1{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(
-\frac12
(\mathbf x-\boldsymbol\mu)^{\mathsf T}
\Sigma^{-1}
(\mathbf x-\boldsymbol\mu)
\right)
$$

指数里的二次型

$$
d_M^2
=(\mathbf x-\boldsymbol\mu)^{\mathsf T}
\Sigma^{-1}
(\mathbf x-\boldsymbol\mu)
$$

叫 Mahalanobis 距离的平方。密度相同的点满足 $d_M^2$ 相同，在二维里形成椭圆而不是普通圆。

若

$$
\Sigma=
\begin{bmatrix}
4&2\\
2&3
\end{bmatrix}
$$

则

$$
|\Sigma|=4\cdot3-2\cdot2=8
$$

非对角元素 2 表示两个坐标有正的共同变化趋势。若 Σ 是对角矩阵，坐标之间在多元高斯模型中独立；非对角元素会把圆形等密度线旋转并拉成椭圆。

多元高斯的线性变换仍然是高斯：

$$
\mathbf Y=A\mathbf X+\mathbf b
$$

满足

$$
\boldsymbol\mu_Y=A\boldsymbol\mu_X+\mathbf b,\qquad
\Sigma_Y=A\Sigma_XA^{\mathsf T}
$$

这个公式把线性层、协方差传播、PCA 和白化连到同一个矩阵表达。若 Σ 只有半正定而不是正定，分布可能落在低维仿射子空间上，不能直接使用普通的 $\Sigma^{-1}$ 和 d 维 Lebesgue 密度。

## 高斯噪声为什么导出平方误差

假设观测 y 围绕模型预测 $\mu_\theta(x)$ 产生高斯噪声：

$$
Y\mid X=x
\sim\mathcal N(\mu_\theta(x),\sigma^2)
$$

单个观测的负对数密度是

$$
\begin{aligned}
-\log f(y\mid x)
&=\frac{(y-\mu_\theta(x))^2}{2\sigma^2}
+\log\sigma+\frac12\log(2\pi)
\end{aligned}
$$

若 σ 对所有样本固定，后两项与模型参数无关，最小化负对数似然等价于最小化平方误差：

$$
\sum_i(y_i-\mu_\theta(x_i))^2
$$

所以 MSE 不是凭空选择的代数形式，它对应一个同方差高斯噪声假设。这个假设也带来边界：如果误差有重尾、明显偏斜或方差随输入改变，固定方差的高斯 NLL 可能不是合适的模型。

若模型同时预测 $\sigma_\theta(x)$，就不能删掉 $\log\sigma$：

$$
-\log f
=\frac{(y-\mu_\theta(x))^2}{2\sigma_\theta(x)^2}
+\log\sigma_\theta(x)
+\frac12\log(2\pi)
$$

第一项鼓励解释观测，第二项防止模型仅把不确定性无限放大。均值和不确定性因此需要一起校准，而不只是给出一个点预测。

## 标准化、异常值与模型假设

z 分数常被用来比较异常程度：

$$
z_i=\frac{x_i-\mu}{\sigma}
$$

在高斯模型下，$|z_i|>3$ 的概率约为 $0.0027$。但这只能在 μ、σ 合理且样本确实近似高斯时作为提示。重尾分布会产生比高斯更多的极端值；有限样本中 μ、σ 也是估计量，会增加额外不确定性。

高斯分布的对称性也可能不适合正值、计数或严重右偏数据。等待时间可用指数分布，计数可用泊松或负二项模型，类别则用分类分布。选择分布时先看支持集和误差机制，再看钟形曲线是否方便。

## 失效模式

**把 $\mathcal N(\mu,\sigma^2)$ 的第二个参数当标准差。**若 σ=3，写法是 $\mathcal N(\mu,9)$；9 是方差，3 才是标准差。

**把一个标准差覆盖 68% 当作普适定理。**68–95–99.7 规则依赖高斯形状。对偏斜或重尾数据，均值 ± 2σ 的覆盖率可能完全不同。

**把密度值当成观测点概率。**高斯在均值处的密度可以大于 1，仍不违反概率；精确点概率是 0，区间概率是密度积分。

**忽略标准化中的尺度。**$x-\mu$ 只是带单位的偏差，除以 σ 后才是无量纲 z 分数。不同变量之间直接比较原始偏差没有意义。

**不独立时仍把方差直接相加。**和的方差还包含 $2\operatorname{Cov}(X,Y)$。只有协方差为 0 时才能删掉这一项。

**从边缘高斯推出联合高斯。**每个坐标单独呈钟形，不代表坐标组合的联合结构是多元高斯；依赖关系仍需要建模。

**用固定方差高斯解释重尾误差。**极端残差会让平方误差和高斯 NLL 过度敏感。应检查残差分布，必要时改用更重尾或更符合数据机制的模型。

**把高斯噪声假设藏在 MSE 后面。**平方误差对应同方差高斯似然。若要预测异方差不确定性，必须保留方差项和它的正则化作用。

## 相关词条

- [连续分布](../probability/continuous-distributions/)：提供密度、CDF 和积分框架。
- [随机变量](../probability/random-variables/)：定义分布、观测值和随机向量。
- [联合分布](../probability/joint-distributions/)：继续讨论多个变量的联合结构。
- [方差与协方差](../probability/variance-and-covariance/)：推导高斯参数中的离散程度和相关性。
- [协方差矩阵](../probability/covariance-matrix/)：展开多元高斯的矩阵参数。
- [中心极限定理](../probability/central-limit-theorem/)：解释高斯为何在和与平均中频繁出现。
- [极大似然](../probability/maximum-likelihood/)：系统推导高斯似然的参数估计。
