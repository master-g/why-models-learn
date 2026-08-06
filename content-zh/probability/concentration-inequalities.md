---
title: "集中不等式：不用完整分布也能控制尾部"
tags: ["why-models-learn"]
---

**集中不等式**用少量条件给出随机变量偏离中心的概率上界。它不需要先把完整分布写出来，也不把有限样本概率近似成正态曲线。最基本的目标是控制

$$
\mathbb P\left(
|X-\mathbb E[X]|\geq t
\right)
$$

随阈值 $t$ 如何下降。Markov 只使用非负变量的一阶矩，Chebyshev 使用方差，Chernoff 把矩母函数纳入优化，Hoeffding 使用有界区间，Bernstein 进一步利用方差和有界增量。本篇推导这些界，比较它们与大数定律、中心极限定理的区别，再把样本复杂度、并集界和经验风险联系起来。

## 尾部概率和集中不是同一个数字

设随机变量 $X$ 的中心为 $\mu$。常见的三种事件是

$$
\mathbb P(X-\mu\geq t)
$$

$$
\mathbb P(X-\mu\leq-t)
$$

以及双侧事件

$$
\mathbb P(|X-\mu|\geq t)
$$

上界的形式通常是

$$
\mathbb P(|X-\mu|\geq t)\leq B(t)
$$

它说真实概率不超过 $B(t)$，不说两者相等。若推导得到 $B(t)>1$，可以把它截成 $\min\{1,B(t)\}$；概率本身不会超过 1。一个有用的集中界应当随 $t$ 增大而下降，或随独立样本数 $n$ 增大而下降。

集中不等式通常控制有限样本的尾部，条件写在上界里。它可以只依赖均值，也可以依赖方差、支持区间或矩母函数。知道的信息越多，界通常越紧，但使用的前提也越强。

## Markov 不等式：从一个非负量开始

若 $Y\geq0$ 且 $\mathbb E[Y]<\infty$，对任意 $a>0$：

$$
\mathbb P(Y\geq a)
\leq
\frac{\mathbb E[Y]}{a}
$$

证明只需看指标函数。逐点都有

$$
\mathbf 1_{\{Y\geq a\}}
\leq
\frac Ya
$$

因为 $Y\geq a$ 的地方右侧至少为 1，不满足事件的地方左侧为 0。两边取期望：

$$
\mathbb P(Y\geq a)
=\mathbb E\left[
\mathbf 1_{\{Y\geq a\}}
\right]
\leq
\mathbb E\left[\frac Ya\right]
=\frac{\mathbb E[Y]}a
$$

### 数字例子

公平六面骰记为 $X$，它非负且 $\mathbb E[X]=3.5$。Markov 不等式给出

$$
\mathbb P(X\geq6)
\leq
\frac{3.5}{6}
=\frac7{12}
\approx0.5833
$$

实际概率是 $\mathbb P(X\geq6)=1/6\approx0.1667$，所以这个界很松。Markov 只知道期望，不知道骰子只能取 1 到 6，也不知道概率质量如何分布。它的价值在于条件极少，而且可以对任何非负随机量使用。

如果目标变量可以变成非负形式，就能得到矩界的家族。对任意 $r>0$：

$$
\mathbb P(|X|\geq t)
=\mathbb P(|X|^r\geq t^r)
\leq
\frac{\mathbb E[|X|^r]}{t^r}
$$

取 $r=1$ 是对 $|X|$ 使用 Markov，取 $r=2$ 会得到 Chebyshev 类型的界。高阶矩有限时，尾部上界可以随 $t^{-r}$ 更快下降。

## Chebyshev 不等式：把偏离平方

令

$$
Y=(X-\mu)^2,
\qquad
\mu=\mathbb E[X]
$$

事件 $|X-\mu|\geq t$ 等价于 $Y\geq t^2$。对 $Y$ 使用 Markov：

$$
\begin{aligned}
\mathbb P(|X-\mu|\geq t)
&=\mathbb P\left((X-\mu)^2\geq t^2\right)\\
&\leq
\frac{\mathbb E[(X-\mu)^2]}{t^2}\\
&=
\frac{\operatorname{Var}(X)}{t^2}
\end{aligned}
$$

这就是 Chebyshev 不等式。它需要有限方差，但不需要知道分布是高斯、均匀还是离散的。对独立同分布样本平均：

$$
\operatorname{Var}(\bar X_n)
=\frac{\sigma^2}{n}
$$

因此

$$
\mathbb P(|\bar X_n-\mu|\geq t)
\leq
\frac{\sigma^2}{nt^2}
$$

这个 $1/n$ 尾界正是弱大数定律的一条直接证明路径。它也说明了为什么有限方差是一个关键分界：没有方差，就不能沿用这个平方矩证明。

### 和 Markov 比一比

对同一个随机变量，Markov 可以应用于 $|X-\mu|$：

$$
\mathbb P(|X-\mu|\geq t)
\leq
\frac{\mathbb E[|X-\mu|]}{t}
$$

而 Chebyshev 使用平方：

$$
\mathbb P(|X-\mu|\geq t)
\leq
\frac{\mathbb E[(X-\mu)^2]}{t^2}
$$

哪个更紧取决于分布和阈值。Chebyshev 的优点不是对每个数字都更小，而是只需要一个容易计算的方差，并且自然给出二次衰减。

### 样本平均的骰子界

公平骰子有

$$
\mu=3.5,
\qquad
\sigma^2=\frac{35}{12}
$$

对 $n=100$ 次平均，取 $t=0.5$：

$$
\begin{aligned}
\mathbb P(|\bar X_{100}-3.5|\geq0.5)
&\leq
\frac{35/12}{100(0.5)^2}\\
&=\frac7{60}\\
&\approx0.1167
\end{aligned}
$$

和单个骰子的 Markov 界相比，这个界已经利用了样本平均的方差缩小。它仍然可能比精确概率宽松，因为所有只知道方差的分布都必须被同一个公式覆盖。

## Chernoff 界：用矩母函数搜索最佳指数

平方只提供二阶信息。若知道指数矩，可以把尾部事件变成 Markov 可以处理的非负变量。对任意 $\lambda>0$：

$$
\{X\geq t\}
=\{\exp(\lambda X)\geq\exp(\lambda t)\}
$$

于是

$$
\mathbb P(X\geq t)
\leq
\exp(-\lambda t)
\mathbb E[\exp(\lambda X)]
$$

令矩母函数的对数为

$$
K_X(\lambda)
=\log\mathbb E[\exp(\lambda X)]
$$

得到 Chernoff 形式：

$$
\mathbb P(X\geq t)
\leq
\inf_{\lambda>0}
\exp\left(
K_X(\lambda)-\lambda t
\right)
$$

优化 $\lambda$ 是关键。固定一个 $\lambda$ 会得到一个合法但可能不紧的界；对所有正的 $\lambda$ 取下确界，才能使用最好的指数尺度。下尾使用 $\lambda<0$：

$$
\mathbb P(X\leq t)
\leq
\inf_{\lambda<0}
\exp\left(
K_X(\lambda)-\lambda t
\right)
$$

若要控制双侧事件，可以分别处理上下尾再相加，这一步会带来一个因子 2 或两个不同的指数项。

### 伯努利和的相对熵界

令 $S_n$ 是 $n$ 个独立 Bernoulli$(p)$ 变量之和。对 $q>p$，优化 Chernoff 参数可以得到

$$
\mathbb P\left(\frac{S_n}{n}\geq q\right)
\leq
\exp\left(
-nD(q\Vert p)
\right)
$$

其中

$$
D(q\Vert p)
=q\log\frac qp
+(1-q)\log\frac{1-q}{1-p}
$$

是两个伯努利分布之间的 KL 散度。它不对称，但当 $q=p$ 时为 0；$q$ 离 $p$ 越远，指数衰减越快。

取 $p=0.5$、$q=0.6$、$n=100$：

$$
D(0.6\Vert0.5)
\approx0.02014
$$

因此

$$
\mathbb P\left(\frac{S_{100}}{100}\geq0.6\right)
\leq
\exp(-100\times0.02014)
\approx0.1335
$$

这个界是有限样本保证，不需要把二项分布的每一项概率逐个相加。它与 CLT 的正态近似回答不同问题：Chernoff 给安全上界，CLT 给中心区域的近似概率。

## Hoeffding 界：只知道每个样本的区间

设 $X_1,\ldots,X_n$ 独立，且

$$
a_i\leq X_i\leq b_i
$$

令 $Y_i=X_i-\mathbb E[X_i]$。Hoeffding 引理给出

$$
\mathbb E[\exp(\lambda Y_i)]
\leq
\exp\left(
\frac{\lambda^2(b_i-a_i)^2}{8}
\right)
$$

独立性让矩母函数相乘：

$$
\begin{aligned}
\mathbb E\left[
\exp\left(\lambda\sum_iY_i\right)
\right]
&=\prod_i\mathbb E[\exp(\lambda Y_i)]\\
&\leq
\exp\left(
\frac{\lambda^2}{8}
\sum_i(b_i-a_i)^2
\right)
\end{aligned}
$$

对和 $S_n=\sum_iX_i$ 使用 Markov 指数技巧：

$$
\begin{aligned}
\mathbb P(S_n-\mathbb E[S_n]\geq t)
&\leq
\exp\left(
-\lambda t
+\frac{\lambda^2}{8}
\sum_i(b_i-a_i)^2
\right)
\end{aligned}
$$

令

$$
\lambda
=\frac{4t}{\sum_i(b_i-a_i)^2}
$$

使右侧最小，得到

$$
\mathbb P(S_n-\mathbb E[S_n]\geq t)
\leq
\exp\left(
-\frac{2t^2}{\sum_i(b_i-a_i)^2}
\right)
$$

双侧版本是

$$
\mathbb P(|S_n-\mathbb E[S_n]|\geq t)
\leq
2\exp\left(
-\frac{2t^2}{\sum_i(b_i-a_i)^2}
\right)
$$

如果所有样本都落在同一个区间 $[a,b]$，对平均值 $\bar X_n$ 令 $t= n\varepsilon$：

$$
\mathbb P(|\bar X_n-\mathbb E[\bar X_n]|\geq\varepsilon)
\leq
2\exp\left(
-\frac{2n\varepsilon^2}{(b-a)^2}
\right)
$$

### 伯努利频率的数字界

Bernoulli 变量落在 $[0,1]$。对 $n=100$、$\varepsilon=0.2$：

$$
\mathbb P(|\bar X_{100}-p|\geq0.2)
\leq
2\exp(-2\times100\times0.2^2)
=2\exp(-8)
\approx0.0006709
$$

这个数字依赖有界支持，不依赖 $p$ 的具体值。若变量可能取任意大的异常值，不能直接把 Hoeffding 套上去。

## Bernstein 和 Bennett：把方差也放回指数界

Hoeffding 用区间长度控制最坏情况，但在变量集中在区间中部时，它没有利用实际方差。设 $Y_i=X_i-\mathbb E[X_i]$ 独立，满足

$$
Y_i\leq M,
\qquad
\sum_i\operatorname{Var}(Y_i)\leq v
$$

Bernstein 不等式给出

$$
\mathbb P\left(\sum_iY_i\geq t\right)
\leq
\exp\left(
-\frac{t^2}{2(v+Mt/3)}
\right)
$$

当 $t$ 较小时，分母主要是 $2v$，尾部近似 $\exp(-t^2/(2v))$；当 $t$ 很大时，$Mt/3$ 主导，衰减转向 $\exp(-3t/(2M))$。它同时记录了局部方差和极端增量。

Bennett 界把指数函数优化得更细：

$$
\mathbb P\left(\sum_iY_i\geq t\right)
\leq
\exp\left(
-\frac v{M^2}h\left(\frac{Mt}{v}\right)
\right)
$$

其中

$$
h(u)=(1+u)\log(1+u)-u
$$

对 $u$ 较小，$h(u)\approx u^2/2$，会回到高斯形状；Bernstein 可以看作用更简单的二次有理式下界替代 $h$。

### Bernoulli 的比较

令 $X_i\sim\operatorname{Bernoulli}(0.5)$。此时

$$
\operatorname{Var}(X_i)=0.25,
\qquad
|X_i-\mathbb E[X_i]|\leq0.5
$$

对 $n=100$、$\varepsilon=0.1$ 的上尾，$t=n\varepsilon=10$，可以取

$$
v=25,
\qquad
M=0.5
$$

于是 Bernstein 界为

$$
\exp\left(
-\frac{10^2}{2(25+0.5\times10/3)}
\right)
\approx0.1534
$$

它是单侧界。这个例子中 Hoeffding 的单侧界为 $\exp(-2)\approx0.1353$，反而略小；没有一种只看形式的界在所有阈值都占优。Bernstein 的优势在于当实际方差远小于支持区间给出的最坏方差时，$v$ 会把这个事实反映出来。

## Sub-Gaussian：把尾部形状写进一个参数

随机变量 $X$ 若满足对所有 $\lambda\in\mathbb R$：

$$
\mathbb E\left[
\exp\left(\lambda(X-\mathbb E[X])\right)
\right]
\leq
\exp\left(\frac{\sigma^2\lambda^2}{2}\right)
$$

就称它是参数 $\sigma^2$ 的 sub-Gaussian 变量。对右尾使用 Chernoff 界：

$$
\mathbb P(X-\mathbb E[X]\geq t)
\leq
\inf_{\lambda>0}
\exp\left(
-\lambda t+\frac{\sigma^2\lambda^2}{2}
\right)
$$

最优参数为 $\lambda=t/\sigma^2$，所以

$$
\mathbb P(X-\mathbb E[X]\geq t)
\leq
\exp\left(-\frac{t^2}{2\sigma^2}\right)
$$

双侧版本是

$$
\mathbb P(|X-\mathbb E[X]|\geq t)
\leq
2\exp\left(-\frac{t^2}{2\sigma^2}\right)
$$

独立 sub-Gaussian 变量相加时，参数相加：

$$
\sum_iX_i
\text{ 的 sub-Gaussian 参数为 }
\sum_i\sigma_i^2
$$

因此同方差样本平均的参数变成 $\sigma^2/n$，尾界中的有效尺度是 $1/\sqrt n$。高斯变量达到这个 MGF 界的等号；有界变量由 Hoeffding 引理可知也是 sub-Gaussian，其中 $X\in[a,b]$ 可以取

$$
\sigma^2=\frac{(b-a)^2}{4}
$$

这就是 Hoeffding 平均值界的另一种写法。

## 应该选择哪一个界

| 已知条件 | 常用不等式 | 两侧样本平均的典型形状 |
| --- | --- | --- |
| $Y\geq0$，只有 $\mathbb E[Y]$ | Markov | $\mathbb E[Y]/a$ |
| 只有有限方差 | Chebyshev | $\sigma^2/(nt^2)$ |
| 矩母函数可计算 | Chernoff | 优化后的指数界 |
| 样本独立且有界 | Hoeffding | $2\exp(-cn t^2)$ |
| 独立、有界增量和方差 | Bernstein | $\exp[-t^2/(v+Mt)]$ |
| MGF 被高斯函数控制 | Sub-Gaussian | $2\exp[-t^2/(2\sigma^2)]$ |

这张表不是「越往下越好」的排名。Markov 的前提最少，Hoeffding 对有界数据简单可靠，Bernstein 在方差很小时更有信息，Chernoff 需要能控制 MGF，Sub-Gaussian 是一种尾部性质而不是单独的推导步骤。实际使用时先列出数据真的满足的条件，再选能给出所需尺度的界。

## 从单个变量到同时控制许多事件

若有事件 $E_1,\ldots,E_m$，并集界不要求它们独立：

$$
\mathbb P\left(\bigcup_{j=1}^mE_j\right)
\leq
\sum_{j=1}^m\mathbb P(E_j)
$$

若每个事件的失败概率不超过 $\delta/m$，则所有事件同时成立的概率至少为 $1-\delta$。代价是一个 $\log m$ 或 $m$ 相关的复杂度项，取决于你如何把单事件上界反解成样本数。

### Hoeffding 的样本复杂度

对 $X_i\in[a,b]$，要求

$$
\mathbb P(|\bar X_n-\mu|\geq\varepsilon)
\leq\delta
$$

由 Hoeffding 界，只要

$$
n
\geq
\frac{(b-a)^2}{2\varepsilon^2}
\log\frac2\delta
$$

就足够。对 $[0,1]$ 中的变量，要求误差 $\varepsilon=0.05$、失败概率 $\delta=0.05$：

$$
n
\geq
\frac1{2(0.05)^2}\log\frac2{0.05}
=200\log40
\approx737.8
$$

所以取 $n=738$。这是一个保证达到目标的充分样本数，不是说 737 个样本一定失败，也不是说 738 个样本一定误差小于 0.05。

### 有限假设集合的经验风险

设假设集合 $\mathcal H$ 有限，每个样本损失落在 $[0,1]$。对固定的 $h\in\mathcal H$，Hoeffding 控制经验风险 $\widehat R(h)$ 与真实风险 $R(h)$ 的差：

$$
\mathbb P\left(
|\widehat R(h)-R(h)|\geq\varepsilon
\right)
\leq
2\exp(-2n\varepsilon^2)
$$

对所有 $h$ 使用并集界：

$$
\mathbb P\left(
\sup_{h\in\mathcal H}
|\widehat R(h)-R(h)|
\geq\varepsilon
\right)
\leq
2|\mathcal H|\exp(-2n\varepsilon^2)
$$

令右侧不超过 $\delta$：

$$
\varepsilon
\geq
\sqrt{
\frac{\log(2|\mathcal H|/\delta)}{2n}
}
$$

例如 $|\mathcal H|=1000$、$n=10000$、$\delta=0.05$ 时：

$$
\sqrt{
\frac{\log(40000)}{20000}
}
\approx0.0230
$$

这只是有限集合和有界损失下的教学版泛化界。神经网络的假设集合通常不能直接枚举，训练过程还会让模型选择依赖数据；此时要换用参数范数、覆盖数、Rademacher 复杂度或稳定性等工具，不能把有限集合公式原样套上去。

## 和大数定律、中心极限定理的关系

| 工具 | 结论性质 | 是否需要完整分布 | 是否给有限样本上界 |
| --- | --- | --- | --- |
| 大数定律 | 渐近收敛 | 通常只需矩和独立性等条件 | 不直接给具体误差概率 |
| 中心极限定理 | 渐近分布近似 | 需要方差和极限条件 | 给近似值，不是保证上界 |
| 集中不等式 | 尾部概率上界 | 依不等式而定 | 是，通常不依赖大样本极限 |

对相同的样本平均，大数定律说

$$
\bar X_n\xrightarrow{P}\mu
$$

中心极限定理说

$$
\frac{\sqrt n(\bar X_n-\mu)}{\sigma}
\xrightarrow{d}
\mathcal N(0,1)
$$

Hoeffding 或 Chebyshev 则直接给出某个有限 $n$ 和某个 $\varepsilon$ 的上界。它们可以互相补充：需要置信区间形状时使用 CLT，需要可证明的最坏情况保证时使用集中界，需要只说明长期收敛时使用大数定律。

![集中不等式从信息条件到尾部上界](/assets/probability/svg/concentration-inequalities.1.svg)

## 失效模式

**把上界当精确概率**：$\exp(-2n\varepsilon^2)$ 是保证，不是观测到的尾部概率。界很松时只能说明「不会超过它」，不能反推真实概率接近它。

**忽略前提**：Hoeffding 需要独立和有界，Bernstein 需要独立且能控制增量与方差。数据有长尾或强相关时，先换成适用的工具，不能只因为公式好看就套用。

**忘记双侧因子和并集代价**：从单侧推双侧通常要加两个尾部；同时控制 $m$ 个事件还要使用并集界。漏掉这些因子会把置信度写得过高。

**让 Chernoff 参数跑出 MGF 定义域**：$\mathbb E[\exp(\lambda X)]$ 可能只在一段 $\lambda$ 区间有限。优化时必须限制在这个定义域内；重尾分布的 MGF 甚至可能在任意正数都发散。

**只按样本方差替换理论方差**：Bernstein、Sub-Gaussian 等理论界中的方差代理必须有可信的确定性控制。把一组样本算出的方差直接当作无条件上界，会额外引入估计误差。

**把相关样本当独立样本**：时间序列、同一用户的重复记录、同一图像的增强副本都可能相关。独立乘积是 Hoeffding MGF 推导的关键一步；相关性会改变有效样本数，不能仍然使用 $n$ 的指数速度。

**误差尺度和损失尺度不一致**：集中界控制的是指定随机变量的平均。如果损失没有落在 $[0,1]$，不能直接使用对应的有界损失公式；先做缩放、截断，或改用次指数尾界。

## 相关词条

- [大数定律](../probability/law-of-large-numbers/)：集中界如何给出样本平均的有限样本控制。
- [中心极限定理](../probability/central-limit-theorem/)：尾部上界和渐近正态近似的区别。
- [期望](../probability/expectation/)：Markov、样本平均和风险定义的中心量。
- [方差与协方差](../probability/variance-and-covariance/)：Chebyshev 和 Bernstein 使用的波动量。
- [最大似然](../probability/maximum-likelihood/)：经验风险与负对数似然的统计解释。
- [指数族](../probability/exponential-family/)：Chernoff MGF、KL 和输出分布的统一结构。
- [KL 散度](../information-theory/kl-divergence/)：伯努利 Chernoff 界中的相对熵项。
