---
title: "KL 散度：两个概率模型相差多少"
tags: ["why-models-learn"]
---

**KL 散度**（Kullback–Leibler divergence）衡量用模型分布 $q$ 代替真实分布 $p$ 时增加的平均对数代价：

$$
D_{\mathrm{KL}}(p\Vert q)
=\sum_xp(x)\log\frac{p(x)}{q(x)}
$$

它也叫相对熵。KL 散度非负，且 $p=q$ 时为 0，但它不是距离：一般不对称，也不满足三角不等式。方向 $p\Vert q$ 表示由 $p$ 产生结果、用 $q$ 来编码。本篇从对数似然比证明非负性，再推导链式法则、数据处理不等式和 Pinsker 界，最后连接交叉熵、最大似然、指数族和变分推断。

## KL 是平均对数似然比

对单个结果 $x$：

$$
\log\frac{p(x)}{q(x)}
$$

是模型 $p$ 相对于模型 $q$ 对该结果的支持优势。按 $p$ 的概率加权：

$$
\begin{aligned}
D_{\mathrm{KL}}(p\Vert q)
&=\mathbb E_{X\sim p}
\left[
\log\frac{p(X)}{q(X)}
\right]\\
&=\mathbb E_p[\log p(X)]
-\mathbb E_p[\log q(X)]
\end{aligned}
$$

第一项是 $p$ 自己的平均对数概率，第二项是用 $q$ 预测 $p$ 产生的数据时的平均对数概率。KL 把「模型错了多少」变成了一个可累加的平均代价。

若使用 bit，KL 的单位是 bit；使用自然对数，单位是 nat。改变底数只会乘上固定换算因子，不会改变哪个模型更近。

## 非负性来自 Jensen 不等式

假设 $p(x)>0$ 的地方都有 $q(x)>0$，令

$$
R(X)=\frac{q(X)}{p(X)}
$$

在 $X\sim p$ 下：

$$
\mathbb E_p[R(X)]
=\sum_xp(x)\frac{q(x)}{p(x)}
=\sum_xq(x)
=1
$$

因为 $\log$ 是凹函数，Jensen 不等式给出

$$
\mathbb E_p[\log R(X)]
\leq
\log\mathbb E_p[R(X)]
=\log1=0
$$

两边乘以负号：

$$
\begin{aligned}
D_{\mathrm{KL}}(p\Vert q)
&=-\mathbb E_p[\log R(X)]\\
&\geq0
\end{aligned}
$$

等号成立需要 $R(X)$ 在 $p$ 的支持集上几乎处处为常数。由于它的 $p$ 期望为 1，这个常数只能是 1，因此 $p=q$。

另一种逐点证明使用

$$
\ln u\leq u-1
$$

令 $u=q(x)/p(x)$，即可得到同样的结果。Jensen 版本更直接地显示了「先取平均再取对数」与「先取对数再平均」之间的差别。

## 数字例子：方向会改变结果

令两个 Bernoulli 分布的正类概率分别为

$$
p=0.8,
\qquad
q=0.5
$$

从 $p$ 产生数据、用 $q$ 编码：

$$
\begin{aligned}
D_{\mathrm{KL}}(p\Vert q)
&=0.8\ln\frac{0.8}{0.5}
+0.2\ln\frac{0.2}{0.5}\\
&\approx0.1927\ \mathrm{nats}
\end{aligned}
$$

反过来：

$$
\begin{aligned}
D_{\mathrm{KL}}(q\Vert p)
&=0.5\ln\frac{0.5}{0.8}
+0.5\ln\frac{0.5}{0.2}\\
&\approx0.2231\ \mathrm{nats}
\end{aligned}
$$

数值不同。第一种方向更重视 $p$ 认为常见的结果，第二种方向更重视 $q$ 认为常见的结果。写公式时必须先说明谁产生数据、谁提供模型。

## 支持集不匹配会产生无穷散度

若存在某个 $x$ 满足

$$
p(x)>0,
\qquad
q(x)=0
$$

那么

$$
D_{\mathrm{KL}}(p\Vert q)=+\infty
$$

因为真实数据有正概率落到 $x$，但模型 $q$ 认为它不可能发生，编码代价无穷大。反过来，$q(x)>0$ 而 $p(x)=0$ 不会直接给 $p\Vert q$ 增加项，因为该结果在 $p$ 下从不出现。

这解释了为什么分类模型、语言模型和概率密度估计需要注意支持集。平滑可以避免训练样本落入零概率，但平滑也改变了模型本身，不是数学上的免费补丁。

## KL、熵和交叉熵的三角关系

交叉熵是

$$
H(p,q)=-\sum_xp(x)\log q(x)
$$

真实熵是

$$
H(p)=-\sum_xp(x)\log p(x)
$$

相减：

$$
\begin{aligned}
H(p,q)-H(p)
&=-\sum_xp(x)\log q(x)
+\sum_xp(x)\log p(x)\\
&=\sum_xp(x)\log\frac{p(x)}{q(x)}\\
&=D_{\mathrm{KL}}(p\Vert q)
\end{aligned}
$$

所以

$$
H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)
$$

这个等式同时说明两件事：

1. 交叉熵不可能小于真实熵；
2. 最小化交叉熵等价于在真实分布固定时最小化 KL。

分类交叉熵、语言模型 NLL 和最大似然，都是在不同数据表示下最小化这个额外代价。

## KL 的链式法则

对联合分布 $p(x,y)$ 和 $q(x,y)$：

$$
p(x,y)=p(x)p(y\mid x),
\qquad
q(x,y)=q(x)q(y\mid x)
$$

因此

$$
\begin{aligned}
\log\frac{p(x,y)}{q(x,y)}
&=\log\frac{p(x)}{q(x)}
+\log\frac{p(y\mid x)}{q(y\mid x)}
\end{aligned}
$$

按 $p(x,y)$ 取期望：

$$
\begin{aligned}
D_{\mathrm{KL}}(p(x,y)\Vert q(x,y))
&=D_{\mathrm{KL}}(p(x)\Vert q(x))\\
&\quad+
\mathbb E_{x\sim p}
\left[
D_{\mathrm{KL}}(p(y\mid x)\Vert q(y\mid x))
\right]
\end{aligned}
$$

联合模型的差异分成两部分：边际 $X$ 的差异，加上在真实 $X$ 上平均的条件 $Y$ 差异。由于条件 KL 非负：

$$
D_{\mathrm{KL}}(p(x,y)\Vert q(x,y))
\geq
D_{\mathrm{KL}}(p(x)\Vert q(x))
$$

如果两个模型对 $X$ 的边际完全一致，联合 KL 就只剩下条件预测器之间的平均差异。

### 一个完全依赖的例子

令 $X$ 是公平二元变量，真实模型中 $Y=X$，而近似模型让 $Y$ 在给定 $X$ 后仍公平独立。两者的 $X$ 边际相同，所以

$$
D_{\mathrm{KL}}(p_X\Vert q_X)=0
$$

但每个 $x$ 下，真实条件分布是确定的，近似条件分布为 $(1/2,1/2)$：

$$
D_{\mathrm{KL}}(p(Y\mid X=x)\Vert q(Y\mid X=x))
=\ln2
$$

因此联合 KL 是 $\ln2\approx0.6931$ nat。所有差异都来自「是否利用 $X$ 预测 $Y$」。

## 数据处理不等式：丢信息不会增加可区分性

令 $T$ 是对样本的确定性或随机性变换，例如把细粒度类别合并成大类。变换后分布记为 $p_T$ 和 $q_T$。数据处理不等式说：

$$
D_{\mathrm{KL}}(p_T\Vert q_T)
\leq
D_{\mathrm{KL}}(p\Vert q)
$$

直觉是：如果先把结果压缩或加噪，再比较两个模型，就无法比原始结果保留更多区分信息。对于确定性映射 $T(x)$，合并多个结果会把它们的概率相加；对数的凹性正好保证合并后的差异不增。

例如原始二元分布 $p=(0.8,0.2)$、$q=(0.5,0.5)$ 的 KL 是

$$
D_{\mathrm{KL}}(p\Vert q)
\approx0.1927\ \mathrm{nats}
$$

若把两个结果都映射成同一个标签，变换后的两个分布都变成 $(1)$，KL 变为 0。变换没有创造新的区分能力，只是把差异抹掉了。

## Pinsker 不等式把 KL 转成概率差

定义总变差距离：

$$
\operatorname{TV}(p,q)
=\frac12\sum_x|p(x)-q(x)|
$$

Pinsker 不等式给出

$$
\operatorname{TV}(p,q)
\leq
\sqrt{\frac12D_{\mathrm{KL}}(p\Vert q)}
$$

这里的 KL 使用自然对数。它把对数平均差异转成普通概率质量差异：KL 小，任何事件的概率差都不会太大。反过来不能简单用总变差精确决定 KL，因为 $q$ 可能在某些位置非常接近 0。

对上面的 Bernoulli$(0.8)$ 和 Bernoulli$(0.5)$：

$$
\operatorname{TV}(p,q)=|0.8-0.5|=0.3
$$

而 Pinsker 上界是

$$
\sqrt{\frac{0.1927}{2}}
\approx0.3104
$$

界略大于实际差异，符合「上界而非等式」的性质。

## 指数族中的 KL 是 Bregman 差

指数族自然参数为 $\eta$，对数配分函数为 $A(\eta)$。此前有

$$
\nabla A(\eta)=\mathbb E_\eta[T(X)]
$$

把两个指数族密度相除：

$$
\log\frac{p_\eta(x)}{p_{\eta'}(x)}
=(\eta-\eta')^{\mathsf T}T(x)
-A(\eta)+A(\eta')
$$

按 $p_\eta$ 取期望：

$$
\begin{aligned}
D_{\mathrm{KL}}(p_\eta\Vert p_{\eta'})
&=A(\eta')-A(\eta)\\
&\quad-
\nabla A(\eta)^{\mathsf T}(\eta'-\eta)
\end{aligned}
$$

右侧是凸函数 $A$ 在 $\eta$ 处的 Bregman 差。KL 的非负性在这里变成了对数配分函数的凸性；Fisher 信息则是这个凸函数的 Hessian。

## MLE 是经验分布到模型族的 KL 投影

给定样本 $x_1,\ldots,x_n$，经验分布记为 $\widehat p_n$。经验交叉熵是

$$
H(\widehat p_n,q_\theta)
=-\frac1n\sum_{i=1}^n\log q_\theta(x_i)
$$

分解为

$$
H(\widehat p_n,q_\theta)
=H(\widehat p_n)
+D_{\mathrm{KL}}(\widehat p_n\Vert q_\theta)
$$

$H(\widehat p_n)$ 与参数 $\theta$ 无关，所以

$$
\arg\min_\theta
H(\widehat p_n,q_\theta)
=\arg\min_\theta
D_{\mathrm{KL}}(\widehat p_n\Vert q_\theta)
$$

这就是 MLE 的几何解释：在允许的模型族中，选择一个相对于经验分布 KL 最小的模型。若真实分布不在模型族内，最优点仍然存在时，它是一个投影，不代表模型已经等于现实。

例如 10 次 Bernoulli 观测中 7 次成功，经验概率为 $\widehat p=0.7$。若只允许模型使用 $q=0.5$，经验分布到模型的 KL 是

$$
D_{\mathrm{KL}}(\operatorname{Bern}(0.7)
\Vert\operatorname{Bern}(0.5))
\approx0.0823\ \mathrm{nats}
$$

若允许 $q$ 在 $(0,1)$ 中自由选择，MLE 取 $q=0.7$，KL 降为 0。

## 变分推断为什么选择 KL 方向

在隐变量模型中，目标后验是 $p(z\mid x)$，近似分布记为 $q(z)$。变分推断常最小化

$$
D_{\mathrm{KL}}(q(z)\Vert p(z\mid x))
$$

用贝叶斯公式展开：

$$
\begin{aligned}
D_{\mathrm{KL}}(q\Vert p(z\mid x))
&=\mathbb E_q
\left[
\log\frac{q(z)}{p(z\mid x)}
\right]\\
&=\log p(x)
-\mathbb E_q
\left[
\log\frac{p(x,z)}{q(z)}
\right]
\end{aligned}
$$

因为 $\log p(x)$ 与 $q$ 无关，最小化 KL 等价于最大化 ELBO：

$$
\mathcal L(q)
=\mathbb E_q
\left[
\log p(x,z)-\log q(z)
\right]
$$

这里使用的是 $q\Vert p$，不是训练分类器时常见的 $p\Vert q$。方向不同会产生不同的近似偏好；不能只看到「KL 小」就省略方向。

![KL 散度是平均对数概率比](/assets/information-theory/svg/kl-divergence.1.svg)

## 失效模式

**把 KL 当作距离**：$D_{\mathrm{KL}}(p\Vert q)$ 与 $D_{\mathrm{KL}}(q\Vert p)$ 通常不同，也不满足三角不等式。需要对称量时要明确选择 Jensen–Shannon 等其他定义。

**漏写方向**：前向 KL 和反向 KL 的加权对象不同。交叉熵训练通常出现 $D_{\mathrm{KL}}(p\Vert q)$，变分推断常出现 $D_{\mathrm{KL}}(q\Vert p)$。

**忽略零概率**：若真实分布支持集上有 $q=0$，前向 KL 为无穷；若优化的是反向 KL，$q$ 避开 $p=0$ 的位置也同样重要。

**混用 bit 和 nat**：Pinsker 的常数写法默认自然对数。若 KL 用 bit，要先乘 $\ln2$ 再代入 nat 版本。

**把小 KL 当成所有任务都近似相同**：KL 控制的是分布的平均对数比，某些低概率区域仍可能有很大的局部比值。决策任务若重视尾部事件，还要单独检查这些区域。

**把数据处理理解成模型变好**：数据处理不等式说变换后更难区分，不代表模型预测更准确。压缩可能只是丢掉了本来有用的信息。

**把经验 KL 当成真实 KL**：用有限样本计算的是经验分布和模型的差异，真实分布未知时还有统计估计误差。训练集 KL 很小不等于部署分布下 KL 也小。

## 相关词条

- [交叉熵](../information-theory/cross-entropy/)：KL 是交叉熵减去真实熵的额外代价。
- [熵](../information-theory/entropy/)：KL 分解中的真实平均不确定性。
- [信息量与惊奇度](../information-theory/information-and-surprise/)：对数概率比和编码代价的单次解释。
- [指数族](../probability/exponential-family/)：KL 作为对数配分函数 Bregman 差的来源。
- [最大似然](../probability/maximum-likelihood/)：经验分布到模型族的 KL 投影。
- [最大后验](../probability/maximum-a-posteriori/)：加入先验后对模型目标的收缩。
- [变量变换](../probability/change-of-variables/)：连续分布 KL 中密度和参考测度的变换。
