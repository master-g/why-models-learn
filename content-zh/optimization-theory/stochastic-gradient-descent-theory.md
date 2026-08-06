---
title: "随机梯度下降理论：小批次噪声与期望收敛"
tags: ["why-models-learn"]
---

随机梯度下降把完整数据集上的梯度换成随机样本或小批次的平均梯度。只要抽样方式使估计量无偏，平均方向仍然指向真实梯度，但单步更新不再保证训练目标下降；方差、批次大小和学习率共同决定每一步的波动以及固定学习率下的噪声地板。本篇从经验风险的抽样形式化出发，推导期望下降、不收敛到零的误差项、非凸驻点率和强凸情形的期望误差，再说明这些定理和实际训练之间的边界。

## 从经验风险得到随机梯度

有 $N$ 个训练样本时，把每个样本的损失写成 $\ell_i(\boldsymbol\theta)$，完整经验风险是

$$
F(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
\ell_i(\boldsymbol\theta)
$$

它的真实梯度为

$$
\nabla F(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
\nabla\ell_i(\boldsymbol\theta)
$$

完整梯度需要遍历全部 $N$ 个样本。随机梯度下降在第 $k$ 步抽取一个大小为 $B$ 的小批次 $\mathcal B_k$，并计算

$$
\boldsymbol g_k
=\frac1B\sum_{i\in\mathcal B_k}
\nabla\ell_i(\boldsymbol\theta_k)
$$

为便于先看清数学结构，假设每次抽样独立且是有放回的均匀抽样。给定当前参数 $\boldsymbol\theta_k$，每个样本梯度的期望是

$$
\mathbb E[\boldsymbol g_i\mid\boldsymbol\theta_k]
=\frac1N\sum_{j=1}^N
\nabla\ell_j(\boldsymbol\theta_k)
=\nabla F(\boldsymbol\theta_k)
$$

独立平均之后仍然无偏：

$$
\mathbb E[\boldsymbol g_k\mid\boldsymbol\theta_k]
=\nabla F(\boldsymbol\theta_k)
$$

随机梯度下降的更新式是

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-\eta_k\boldsymbol g_k
$$

和完整梯度下降相比，改变的是梯度来源，不是更新方向的定义。区别在于 $\boldsymbol g_k$ 每次都可能不同。

### 方差随批次大小下降

令单样本梯度的噪声为

$$
\boldsymbol\xi_i
=\boldsymbol g_i-\nabla F(\boldsymbol\theta)
$$

则

$$
\mathbb E[\boldsymbol\xi_i\mid\boldsymbol\theta]
=\boldsymbol0
$$

假设单样本噪声的二阶大小满足

$$
\mathbb E\left[
\|\boldsymbol\xi_i\|_2^2
\mid\boldsymbol\theta
\right]
\leq\sigma^2
$$

小批次梯度的噪声是

$$
\boldsymbol\xi_{\mathcal B}
=\boldsymbol g_{\mathcal B}
-\nabla F(\boldsymbol\theta)
=\frac1B\sum_{j=1}^B\boldsymbol\xi_j
$$

独立性让交叉项的期望为零，因此

$$
\mathbb E\left[
\|\boldsymbol\xi_{\mathcal B}\|_2^2
\mid\boldsymbol\theta
\right]
=\frac1B
\mathbb E\left[
\|\boldsymbol\xi_i\|_2^2
\mid\boldsymbol\theta
\right]
\leq\frac{\sigma^2}{B}
$$

批次从 $B=1$ 增大到 $B=4$，方差缩小为原来的四分之一，而不是四分之一的标准差。标准差只缩小为原来的 $1/2$。

例如，某个参数点的四个等可能单样本梯度值是 $1,3,5,7$。它们的平均值为

$$
\bar g=\frac{1+3+5+7}{4}=4
$$

相对平均值的方差是

$$
\frac{(-3)^2+(-1)^2+1^2+3^2}{4}=5
$$

若独立抽取 $B=4$ 个样本并平均，平均梯度的方差是 $5/4=1.25$。抽样平均使方向更稳，但一次批次仍可能偏离 $4$。

## 单步不一定下降

设完整目标 $F$ 的梯度是 $L$-Lipschitz 连续的。记

$$
\boldsymbol g=\nabla F(\boldsymbol\theta),
\qquad
\widehat{\boldsymbol g}
=\boldsymbol g+\boldsymbol\xi,
\qquad
\mathbb E[\boldsymbol\xi\mid\boldsymbol\theta]=\boldsymbol0
$$

随机更新为

$$
\boldsymbol\theta^+
=\boldsymbol\theta-\eta\widehat{\boldsymbol g}
$$

对每一个已经抽到的 $\widehat{\boldsymbol g}$，下降引理给出

$$
\begin{aligned}
F(\boldsymbol\theta^+)
&\leq
F(\boldsymbol\theta)
-\eta\boldsymbol g^\mathsf T\widehat{\boldsymbol g}
+\frac{L\eta^2}{2}
\|\widehat{\boldsymbol g}\|_2^2
\end{aligned}
$$

现在对抽样取条件期望。无偏性给出

$$
\mathbb E[
\boldsymbol g^\mathsf T\widehat{\boldsymbol g}
\mid\boldsymbol\theta
]
=\|\boldsymbol g\|_2^2
$$

而

$$
\begin{aligned}
\mathbb E[
\|\widehat{\boldsymbol g}\|_2^2
\mid\boldsymbol\theta
]
&=
\|\boldsymbol g\|_2^2
+\mathbb E[
\|\boldsymbol\xi\|_2^2
\mid\boldsymbol\theta
]\\
&\leq
\|\boldsymbol g\|_2^2+\frac{\sigma^2}{B}
\end{aligned}
$$

所以期望的一步上界是

$$
\mathbb E[
F(\boldsymbol\theta^+)
\mid\boldsymbol\theta
]
\leq
F(\boldsymbol\theta)
-\eta\left(1-\frac{L\eta}{2}\right)
\|\nabla F(\boldsymbol\theta)\|_2^2
+\frac{L\eta^2\sigma^2}{2B}
$$

当 $0<\eta\leq1/L$ 时，可以简化为

$$
\mathbb E[
F(\boldsymbol\theta^+)
\mid\boldsymbol\theta
]
\leq
F(\boldsymbol\theta)
-\frac\eta2
\|\nabla F(\boldsymbol\theta)\|_2^2
+\frac{L\eta^2\sigma^2}{2B}
$$

前一项是梯度带来的下降，后一项是随机梯度带来的噪声代价。完整梯度相当于 $\sigma^2=0$，才会退回确定性下降引理。

### 数值例子：噪声把一次下降抬高

取一维目标

$$
F(\theta)=\frac12\theta^2,
\qquad
\theta_0=2,
\qquad
\eta=0.5
$$

这里 $L=1$，完整梯度为 $2$。假设单步噪声 $\xi$ 以相同概率取 $1$ 或 $-1$，于是

$$
\widehat g=2+\xi
\in\{1,3\}
$$

随机更新给出

$$
\theta_1
=2-0.5\widehat g
\in\{1.5,0.5\}
$$

两种结果的函数值分别为

$$
F(1.5)=1.125,
\qquad
F(0.5)=0.125
$$

期望函数值是

$$
\mathbb E[F(\theta_1)]
=\frac{1.125+0.125}{2}
=0.625
$$

如果用完整梯度，更新得到 $\theta_1=1$，函数值是 $F(1)=0.5$。随机性没有改变平均梯度，却把期望函数值抬高了 $0.125$。

把 $B=4$ 个独立噪声平均后，平均噪声方差从 $1$ 降为 $1/4$。这时

$$
\mathbb E[F(\theta_1)]
=0.5+\frac{0.5^2}{2}\cdot\frac14
=0.53125
$$

批次变大后更接近完整梯度结果，但每一步要读取四倍样本。

## 非凸目标的期望驻点保证

设 $F$ 是 $L$-光滑函数，有有限下界

$$
F_\star=\inf_{\boldsymbol\theta}F(\boldsymbol\theta)>-\infty
$$

并假设每个小批次梯度满足无偏性与

$$
\mathbb E[
\|\boldsymbol\xi_k\|_2^2
\mid\boldsymbol\theta_k
]
\leq\frac{\sigma^2}{B}
$$

取固定步长 $\eta\leq1/L$。从期望下降式移项：

$$
\frac\eta2
\mathbb E[
\|\nabla F(\boldsymbol\theta_k)\|_2^2
]
\leq
\mathbb E[
F(\boldsymbol\theta_k)-F(\boldsymbol\theta_{k+1})
]
+\frac{L\eta^2\sigma^2}{2B}
$$

对 $k=0,\ldots,T-1$ 求和，目标函数项望远镜相消：

$$
\frac\eta2
\sum_{k=0}^{T-1}
\mathbb E[
\|\nabla F(\boldsymbol\theta_k)\|_2^2
]
\leq
F(\boldsymbol\theta_0)-F_\star
+\frac{LT\eta^2\sigma^2}{2B}
$$

记初始间隙为

$$
\Delta=F(\boldsymbol\theta_0)-F_\star
$$

就得到平均梯度平方界

$$
\frac1T\sum_{k=0}^{T-1}
\mathbb E[
\|\nabla F(\boldsymbol\theta_k)\|_2^2
]
\leq
\frac{2\Delta}{\eta T}
+\frac{L\eta\sigma^2}{B}
$$

右侧有两个相反的趋势：

- $\eta$ 太小，第一项 $\dfrac{2\Delta}{\eta T}$ 很大，进展太慢；
- $\eta$ 太大，第二项 $\dfrac{L\eta\sigma^2}{B}$ 很大，噪声地板变高；
- $B$ 变大只直接压低第二项，代价是每次更新读取更多样本。

至少有一个迭代点满足

$$
\min_{0\leq k<T}
\mathbb E[
\|\nabla F(\boldsymbol\theta_k)\|_2^2
]
\leq
\frac{2\Delta}{\eta T}
+\frac{L\eta\sigma^2}{B}
$$

这里保证的是期望意义下的驻点，不是某一次运行必然找到的点。若目标非凸，梯度接近零仍可能是鞍点或局部最大点。

### 让两项平衡

暂时忽略 $\eta\leq1/L$ 的上限，把

$$
\frac{2\Delta}{\eta T}
+\frac{L\eta\sigma^2}{B}
$$

视为关于 $\eta$ 的函数。两项相等时的步长是

$$
\eta_{\mathrm{bal}}
=\sqrt{\frac{2\Delta B}{L\sigma^2T}}
$$

如果它没有超过 $1/L$，代回可得

$$
\frac1T\sum_{k=0}^{T-1}
\mathbb E[
\|\nabla F(\boldsymbol\theta_k)\|_2^2
]
\leq
2\sqrt{\frac{2L\Delta\sigma^2}{BT}}
$$

这显示了 $T$、$B$ 与方差之间的关系。它不是说把批次无限增大就可以免费加速：固定总样本访问量时，批次变大也会减少更新次数。

例如令 $L=1$、$\Delta=2$、$\sigma^2=1$、$B=4$、$\eta=0.5$、$T=100$，固定步长界为

$$
\frac{2\Delta}{\eta T}
+\frac{L\eta\sigma^2}{B}
=0.08+0.125
=0.205
$$

这是梯度平方的期望平均上界，不能直接当作函数值误差或实际训练曲线的预测值。

## 凸目标下的函数值平均界

非凸界只使用光滑性和下界。若 $F$ 还凸，梯度可以和函数值差连接起来。设 $\boldsymbol\theta^\star$ 是全局最小点，令

$$
\boldsymbol r_k
=\boldsymbol\theta_k-\boldsymbol\theta^\star
$$

更新后距离的条件期望为

$$
\begin{aligned}
\mathbb E[
\|\boldsymbol r_{k+1}\|_2^2
\mid\boldsymbol\theta_k
]
&=
\|\boldsymbol r_k\|_2^2
-2\eta
\nabla F(\boldsymbol\theta_k)^\mathsf T\boldsymbol r_k\\
&\quad+\eta^2
\left(
\|\nabla F(\boldsymbol\theta_k)\|_2^2
+\frac{\sigma^2}{B}
\right)
\end{aligned}
$$

凸性给出

$$
\nabla F(\boldsymbol\theta_k)^\mathsf T\boldsymbol r_k
\geq
F(\boldsymbol\theta_k)-F(\boldsymbol\theta^\star)
$$

而凸且 $L$-光滑的函数满足

$$
\|\nabla F(\boldsymbol\theta_k)\|_2^2
\leq
2L\bigl(
F(\boldsymbol\theta_k)-F(\boldsymbol\theta^\star)
\bigr)
$$

因此当 $\eta\leq1/(2L)$ 时：

$$
\begin{aligned}
\mathbb E[
\|\boldsymbol r_{k+1}\|_2^2
\mid\boldsymbol\theta_k
]
&\leq
\|\boldsymbol r_k\|_2^2
-\eta\bigl(
F(\boldsymbol\theta_k)-F(\boldsymbol\theta^\star)
\bigr)\\
&\quad+\frac{\eta^2\sigma^2}{B}
\end{aligned}
$$

累加后得到函数值差的平均界：

$$
\frac1T\sum_{k=0}^{T-1}
\mathbb E[
F(\boldsymbol\theta_k)-F(\boldsymbol\theta^\star)
]
\leq
\frac{\|\boldsymbol\theta_0-\boldsymbol\theta^\star\|_2^2}{\eta T}
+\frac{\eta\sigma^2}{B}
$$

它和非凸驻点界有同样的形状：第一项随迭代次数下降，第二项是固定步长留下的噪声地板。凸性把「梯度平方变小」升级为「函数值差的平均变小」，但仍然没有让每一次随机更新都下降。

## 强凸目标只能收敛到噪声邻域

现在假设 $F$ 是 $\mu$-强凸且 $L$-光滑，$\boldsymbol\theta^\star$ 是其唯一最小点。记

$$
\boldsymbol r_k
=\boldsymbol\theta_k-\boldsymbol\theta^\star
$$

无偏随机梯度的误差写成

$$
\widehat{\boldsymbol g}_k
=\nabla F(\boldsymbol\theta_k)+\boldsymbol\xi_k
$$

展开平方并取条件期望：

$$
\begin{aligned}
\mathbb E[
\|\boldsymbol r_{k+1}\|_2^2
\mid\boldsymbol\theta_k
]
&=
\|\boldsymbol r_k\|_2^2
-2\eta
\boldsymbol r_k^\mathsf T
\nabla F(\boldsymbol\theta_k)\\
&\quad+\eta^2
\|\nabla F(\boldsymbol\theta_k)\|_2^2
+\frac{\eta^2\sigma^2}{B}
\end{aligned}
$$

强凸性给出梯度的强单调性：

$$
\boldsymbol r_k^\mathsf T
\nabla F(\boldsymbol\theta_k)
\geq
\mu\|\boldsymbol r_k\|_2^2
$$

而光滑性与 $\nabla F(\boldsymbol\theta^\star)=\boldsymbol0$ 给出

$$
\|\nabla F(\boldsymbol\theta_k)\|_2
\leq
L\|\boldsymbol r_k\|_2
$$

代入后：

$$
\mathbb E[
\|\boldsymbol r_{k+1}\|_2^2
\mid\boldsymbol\theta_k
]
\leq
\left(1-2\mu\eta+L^2\eta^2\right)
\|\boldsymbol r_k\|_2^2
+\frac{\eta^2\sigma^2}{B}
$$

若

$$
0<\eta\leq\frac{\mu}{L^2}
$$

则

$$
\mathbb E[
\|\boldsymbol r_{k+1}\|_2^2
]
\leq
(1-\mu\eta)
\mathbb E[
\|\boldsymbol r_k\|_2^2
]
+\frac{\eta^2\sigma^2}{B}
$$

递推得到

$$
\mathbb E[
\|\boldsymbol r_k\|_2^2
]
\leq
(1-\mu\eta)^k
\|\boldsymbol r_0\|_2^2
+\frac{\eta\sigma^2}{\mu B}
$$

第一项是初始误差逐步消失，第二项是固定学习率留下的噪声地板。若 $\eta$ 减小，地板下降；若 $B$ 增大，地板也下降。

例如 $\mu=L=1$、$\sigma^2=1$、$B=4$、$\eta=0.2$、$\theta_0=2$ 时，上式给出

$$
\mathbb E[\theta_k^2]
\leq
0.8^k\cdot4+0.05
$$

在 $k=10$ 时右侧为

$$
0.8^{10}\cdot4+0.05
=0.4794967296
$$

这个数是递推上界，不是带有具体随机种子的一次训练轨迹。

## 递减学习率让噪声地板下降

固定 $\eta$ 的理论图像是：确定性误差下降到足够小后，噪声项与下降项同量级，参数在最优点附近持续抖动。要让误差继续缩小，可以使用递减学习率 $\eta_k$。

随机逼近中常见的 Robbins–Monro 条件是

$$
\sum_{k=0}^{\infty}\eta_k=\infty,
\qquad
\sum_{k=0}^{\infty}\eta_k^2<\infty
$$

第一条防止总步长有限、还没有走到目标附近就停止；第二条让累积的方差贡献有限。典型例子是

$$
\eta_k=\frac{\eta_0}{k+1}
$$

它的步长总和发散，而平方和收敛。在适当的无偏性、方差和目标正则性条件下，这类调度可以把强凸问题的期望误差继续压向零。

实际深度学习常用分段衰减、余弦衰减或预热后衰减。它们不一定逐项满足最简单的 Robbins–Monro 证明，但分析时仍然要区分两件事：固定学习率允许一个稳定噪声尺度，递减学习率改变了这个尺度，也改变了后期的适应速度。

## 批次大小是方差与更新频率的交换

把单步成本近似看成与 $B$ 成正比，批次大小会产生一个具体交换：

| 批次大小 | 单步梯度方差 | 每步读取样本 | 常见后果 |
| --- | --- | --- | --- |
| $B=1$ | $\sigma^2$ | 1 | 更新频繁，方向波动大 |
| $1<B\ll N$ | $\sigma^2/B$ | $B$ | 方向更稳，仍保留较多更新 |
| $B=N$ | 近似为 0 | 全部样本 | 接近完整梯度，单步成本高 |

如果总共访问相同数量的训练样本，大批次不一定产生同样多的参数更新。大批次降低了每一步的随机误差，却可能让优化器在相同样本预算下更少地修正参数。另一方面，硬件对矩阵批处理友好，增大 $B$ 可能提高吞吐量；理论中的方差下降和系统中的并行效率需要同时测量。

批次从有放回抽样改成一个 epoch 内无放回遍历时，批次之间不再完全独立，简单的 $\sigma^2/B$ 公式不再是精确描述。它仍然常作为分析基线，但不能把基线假设写成所有数据加载器都满足的事实。

## 训练目标和理论假设的边界

随机梯度下降理论通常需要以下条件：

- 给定参数时，随机梯度对目标梯度无偏；
- 梯度噪声的二阶矩有界，或至少能被某个随参数变化的量控制；
- 目标函数有下界，并且梯度满足光滑性条件；
- 约束、裁剪、延迟更新和数据增强没有悄悄改变被分析的目标。

实际训练中，每一项都可能失效：

- 梯度裁剪把估计量截断，通常会引入偏差；
- 动量使用历史随机梯度，状态不再只由当前参数决定；
- 分布式训练的延迟梯度对应旧参数；
- dropout、数据增强和正则化可能让每次更新的目标都带有额外随机性；
- 混合精度和梯度累积改变了实际的数值噪声和更新频率。

因此，期望收敛界是对指定随机过程的结论，不是对一份训练日志的自动认证。看到训练损失偶尔上升并不一定是实现错误；反过来，平均损失下降也不能证明每个样本的梯度都无偏或最终模型一定泛化。

![随机梯度的平均方向与噪声地板](/assets/optimization-theory/svg/stochastic-gradient-descent-theory.1.svg)

## 失效模式

**把无偏理解成每一步都正确。** 无偏只说条件期望等于真实梯度，单次小批次仍然可能偏向相反方向。

**把批次增大四倍理解成噪声标准差也缩小四倍。** 独立平均让方差缩小为四分之一，标准差只缩小为二分之一。

**把随机损失单步上升当成下降引理失败。** 确定性引理适用于给定方向，随机版本控制的是条件期望，并允许单步样本落在上界之外。

**忽略固定学习率的噪声地板。** 强凸问题的确定性误差可以趋于零，但有界方差和固定 $\eta$ 时，期望平方距离通常只收敛到与 $\eta\sigma^2/(\mu B)$ 同阶的邻域。

**把非凸驻点界写成全局最小值保证。** 非凸分析给的是梯度范数或其期望变小；鞍点、局部最大点和不同局部谷底仍需要额外结构来区分。

**只比较每步损失，不比较样本预算。** $B$ 变大后每一步可能更平滑，但同样的读取样本数对应更少的更新次数；吞吐量和优化进度必须分别报告。

**把递减学习率的条件省略。** 步长衰减太快会让总移动距离有限，衰减太慢又会让噪声平方和无法控制；需要同时检查两条级数条件或使用与目标假设匹配的定理。

**把理论方差当成真实训练中的固定常数。** 样本梯度方差可能随参数、类别不平衡和数据增强策略变化，单个全局 $\sigma^2$ 往往只是粗略上界。

## 相关词条

- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：先理解确定性梯度下降的下降引理、步长边界和收敛率。
- [梯度](../calculus/gradient/)：解释每个样本梯度如何组成当前参数的更新方向。
- [期望](../probability/expectation/)：处理无偏随机梯度和条件期望下的单步分析。
- [方差与协方差](../probability/variance-and-covariance/)：量化小批次平均如何降低梯度噪声。
- [优化问题](../optimization-theory/optimization-problems/)：定义经验风险、目标函数、可行集与最小值。
- [凸集与凸函数](../optimization-theory/convex-sets-and-functions/)：提供凸性和强凸性带来的全局函数值结构。
- [局部与全局最小值](../optimization-theory/local-and-global-minima/)：说明驻点、局部谷底和全局最小点的区别。
- [动量方法理论](../optimization-theory/momentum-theory/)：研究把历史随机梯度带入更新后对噪声和振荡的影响。
- [自适应学习率](../optimization-theory/adaptive-learning-rates/)：分析按坐标调节随机梯度步长的方法。
- [梯度裁剪](../training-nn/gradient-clipping/)：说明截断梯度为什么可能改变无偏性和实际优化目标。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：连接优化噪声、训练损失和泛化表现的边界。
