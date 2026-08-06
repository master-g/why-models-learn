---
title: "最大后验：如何把先验知识并入参数估计"
tags: ["why-models-learn"]
---

**最大后验估计**（MAP）在观测数据的似然之外加入参数先验，选择后验密度最高的参数。由贝叶斯公式：

$$
p(\theta\mid D)
=\frac{p(D\mid\theta)\pi(\theta)}{p(D)}
$$

其中 $D$ 是数据，$\pi(\theta)$ 是先验，$p(D)$ 是证据。因此

$$
\widehat\theta_{\mathrm{MAP}}
=\arg\max_\theta p(\theta\mid D)
$$

由贝叶斯公式，最大化后验也等价于最大化似然与先验的乘积：

$$
\widehat\theta_{\mathrm{MAP}}
=\arg\max_\theta
\left[
p(D\mid\theta)\pi(\theta)
\right]
$$

取对数后：

$$
\widehat\theta_{\mathrm{MAP}}
=\arg\max_\theta
\left[
\log p(D\mid\theta)+\log\pi(\theta)
\right]
$$

MLE 只最大化似然，MAP 还让参数落在先验偏好的区域。本篇先拆开 MLE、MAP、后验均值和完整后验，再用 Beta–Bernoulli 与高斯共轭模型推导闭式结果，最后连接 L2 正则化、后验预测、Laplace 近似和参数化陷阱。

## MAP 只取后验的一座峰

固定数据 $D$ 后，后验是参数空间上的一个概率密度。MAP 是它的众数：

$$
\widehat\theta_{\mathrm{MAP}}
=\operatorname{mode}\bigl(p(\theta\mid D)\bigr)
$$

它和几个相邻概念不同：

| 方法 | 优化或计算对象 | 是否保留完整不确定性 | 先验是否进入 |
| --- | --- | --- | --- |
| MLE | 最大化 $p(D\mid\theta)$ | 否 | 否 |
| MAP | 最大化 $p(\theta\mid D)$ | 否，只保留峰值 | 是 |
| 后验均值 | 计算 $\mathbb E[\theta\mid D]$ | 否，只保留均值 | 是 |
| 后验分布 | 保留 $p(\theta\mid D)$ | 是 | 是 |

对称高斯后验中，均值、中位数和众数可能相同；偏斜或多峰后验中，它们可能完全不同。一个 MAP 点不能回答「参数还有多大不确定性」，更不能替代后验预测积分。

证据 $p(D)$ 对固定数据和固定模型族不依赖 $\theta$，所以求 MAP 时可以丢掉：

$$
\widehat\theta_{\mathrm{MAP}}
=\arg\max_\theta
\left[
\log p(D\mid\theta)+\log\pi(\theta)
\right]
$$

但在比较不同模型、计算后验概率或做模型证据时，$p(D)$ 不能被忘记。优化时可省略归一化常数，不等于概率论里它不存在。

## Beta–Bernoulli：先验如何移动成功率

令观测中有 $s$ 次成功、$f$ 次失败，$n=s+f$。Bernoulli 似然为

$$
p(D\mid p)
\propto
p^s(1-p)^f
$$

给成功率一个 Beta 先验：

$$
\pi(p)
=\frac1{B(a,b)}
p^{a-1}(1-p)^{b-1},
\qquad0<p<1
$$

后验正比于似然乘先验：

$$
\begin{aligned}
p(p\mid D)
&\propto
p^s(1-p)^f\,
p^{a-1}(1-p)^{b-1}\\
&\propto
p^{a+s-1}(1-p)^{b+f-1}
\end{aligned}
$$

所以

$$
p\mid D\sim\operatorname{Beta}(a+s,b+f)
$$

这是共轭的意思：更新前后属于同一个分布族，只需要把成功和失败计数加到先验形状参数上。

若后验两个形状参数都大于 1，Beta 分布的 MAP 是

$$
\widehat p_{\mathrm{MAP}}
=\frac{a+s-1}{a+b+n-2}
$$

后验均值则是

$$
\mathbb E[p\mid D]
=\frac{a+s}{a+b+n}
$$

MLE 为

$$
\widehat p_{\mathrm{MLE}}=\frac sn
$$

三者的分母和减法不同，不能互换。

### 数字例子

观察 10 次试验，其中 7 次成功。MLE 是

$$
\widehat p_{\mathrm{MLE}}=\frac7{10}=0.7
$$

若先验为 $\operatorname{Beta}(2,2)$，它在 0.5 附近更集中。后验为

$$
p\mid D\sim\operatorname{Beta}(9,5)
$$

于是

$$
\widehat p_{\mathrm{MAP}}
=\frac{9-1}{9+5-2}
=\frac8{12}
=\frac23
\approx0.6667
$$

后验均值是

$$
\mathbb E[p\mid D]=\frac9{14}\approx0.6429
$$

先验把点估计从样本频率 0.7 向 0.5 拉回。数据量继续增加时，固定强度的先验相对于数据的影响会减弱；当 $n$ 很大时，MAP、MLE 和后验均值通常更接近。

如果选择 $\operatorname{Beta}(1,1)$ 均匀先验，后验为 $\operatorname{Beta}(8,4)$，其 MAP 恰好为

$$
\frac{8-1}{8+4-2}=\frac7{10}
$$

均匀先验在这个例子中不移动 MAP，但仍然会改变后验分布和后验均值。

## 先验不是凭空加分，而是明确的建模假设

Beta 先验可以用伪计数解释：$\operatorname{Beta}(a,b)$ 像是先观察了 $a-1$ 次成功和 $b-1$ 次失败，再看到真实数据。这个解释有助于理解收缩，但不能把任意先验都机械当作实际历史样本。

先验有三个作用：

1. 在数据少时提供可解释的偏好；
2. 在边界或不可识别问题中提供稳定化；
3. 把领域知识写进可计算的概率模型。

先验也有代价。过于集中的先验会压制真实但不常见的参数；先验和似然的支持集没有交集时，后验可能不存在；不恰当的非正规先验可能导致后验不可归一化。使用 MAP 前要说明先验的尺度、来源和敏感性。

## Dirichlet–分类：对类别概率做平滑

分类概率向量满足

$$
\boldsymbol p=(p_1,\ldots,p_K),
\qquad
\sum_{k=1}^Kp_k=1
$$

Dirichlet 先验为

$$
\pi(\boldsymbol p)
\propto
\prod_{k=1}^Kp_k^{\alpha_k-1}
$$

若数据中的类别计数为 $n_1,\ldots,n_K$，后验仍为 Dirichlet：

$$
\boldsymbol p\mid D
\sim
\operatorname{Dirichlet}
(\alpha_1+n_1,\ldots,\alpha_K+n_K)
$$

当所有后验形状参数大于 1 时，MAP 的第 $k$ 个分量是

$$
\widehat p_{k,\mathrm{MAP}}
=\frac{\alpha_k+n_k-1}
{\sum_{j=1}^K(\alpha_j+n_j)-K}
$$

均匀的 $\alpha_k=1$ 先验会让 MAP 回到分类频率，只要后验没有落在边界；取 $\alpha_k>1$ 会把零计数类别从 0 拉开，避免后续对数似然出现 $\log0$。这是一种平滑，不是声称每个类别已经真实观察过。

## 高斯共轭先验：数据平均和先验平均加权

设观测方差 $\sigma^2$ 已知：

$$
X_i\mid\mu\sim\mathcal N(\mu,\sigma^2)
$$

给均值一个高斯先验：

$$
\mu\sim\mathcal N(\mu_0,\tau^2)
$$

忽略与 $\mu$ 无关的常数，负对数后验是

$$
\begin{aligned}
-\log p(\mu\mid D)
&=
\frac1{2\sigma^2}
\sum_{i=1}^n(x_i-\mu)^2
+\frac1{2\tau^2}(\mu-\mu_0)^2
+C
\end{aligned}
$$

对 $\mu$ 求导：

$$
\frac{\partial}{\partial\mu}
[-\log p(\mu\mid D)]
=
\frac n{\sigma^2}(\mu-\bar x)
+\frac1{\tau^2}(\mu-\mu_0)
$$

令其为 0：

$$
\widehat\mu_{\mathrm{MAP}}
=
\frac{
\frac n{\sigma^2}\bar x
+\frac1{\tau^2}\mu_0
}{
\frac n{\sigma^2}+\frac1{\tau^2}
}
$$

它是两个均值的精度加权平均。数据精度是 $n/\sigma^2$，先验精度是 $1/\tau^2$；方差越小，信息越强，权重越大。

例如 $n=4,\bar x=3,\sigma^2=1$，先验 $\mu_0=0,\tau^2=4$：

$$
\widehat\mu_{\mathrm{MAP}}
=\frac{4\cdot3+0.25\cdot0}{4+0.25}
=\frac{12}{4.25}
\approx2.8235
$$

MLE 是 $\bar x=3$。后验仍是高斯，后验方差为

$$
\operatorname{Var}(\mu\mid D)
=\frac1{4+0.25}
=\frac4{17}
\approx0.2353
$$

这里 MAP 和后验均值相同，是因为后验对称；这不是 MAP 普遍等于后验均值的定理。

## L2 正则化是高斯先验的 MAP 形式

对线性回归或神经网络权重 $w$，若观测噪声为高斯：

$$
y_i\mid x_i,w
\sim
\mathcal N(f_w(x_i),\sigma_y^2)
$$

给权重独立零均值高斯先验：

$$
w\sim\mathcal N(0,\tau^2I)
$$

负对数后验去掉常数后：

$$
\frac1{2\sigma_y^2}
\sum_{i=1}^n
(y_i-f_w(x_i))^2
+\frac1{2\tau^2}\|w\|_2^2
$$

乘以 $2\sigma_y^2$，等价于最小化

$$
\sum_{i=1}^n
(y_i-f_w(x_i))^2
+\lambda\|w\|_2^2,
\qquad
\lambda=\frac{\sigma_y^2}{\tau^2}
$$

因此 L2 惩罚可以解释为 MAP 的负对数先验。更强的先验（更小的 $\tau^2$）对应更大的惩罚系数，倾向于把权重拉向 0。

实际深度学习代码里常把损失写成样本平均而不是样本总和，此时为了保持相同的 MAP 解释，$\lambda$ 是否随 $n$ 缩放要明确说明。很多框架的 weight decay 还可能与优化器更新规则、动量和自适应预条件器分离，不能只看一个参数名就断言它等于严格的贝叶斯 MAP。

## MAP 不等于完整的后验预测

用 MAP 做点预测是

$$
p(y_\star\mid x_\star,\widehat\theta_{\mathrm{MAP}})
$$

完整贝叶斯后验预测则对所有参数加权：

$$
p(y_\star\mid x_\star,D)
=\int
p(y_\star\mid x_\star,\theta)
p(\theta\mid D)\,d\theta
$$

如果后验很窄，两者可能接近；如果后验偏斜、多峰或模型输出对参数高度非线性，代入 MAP 会丢失尾部和多峰信息。MAP 点也不能直接给出预测区间，必须使用后验采样、数值积分或近似推断。

参数不确定性和观测噪声是两件事。即使参数已经知道，$p(y_\star\mid x_\star,\theta)$ 仍可能有不可约噪声；MAP 只选了一组参数，更不会自动消除这个噪声。

## Laplace 近似：在 MAP 附近用一个高斯

若后验在 MAP 附近是单峰且足够光滑，把负对数后验在 $\widehat\theta_{\mathrm{MAP}}$ 处作二阶 Taylor 展开：

$$
-\log p(\theta\mid D)
\approx
-\log p(\widehat\theta_{\mathrm{MAP}}\mid D)
+\frac12
(\theta-\widehat\theta_{\mathrm{MAP}})^\mathsf T
H
(\theta-\widehat\theta_{\mathrm{MAP}})
$$

其中

$$
H=
\nabla_\theta^2
[-\log p(\theta\mid D)]
\bigg|_{\theta=\widehat\theta_{\mathrm{MAP}}}
$$

于是

$$
p(\theta\mid D)
\approx
\mathcal N\left(
\widehat\theta_{\mathrm{MAP}},H^{-1}
\right)
$$

这把 MAP 的曲率转成参数协方差近似。若 $H$ 奇异、后验多峰、边界截断明显或神经网络存在大量对称等价方向，$H^{-1}$ 可能不存在或严重误导。Laplace 近似是围绕 MAP 的局部近似，不是后验本身。

## MAP 的参数化陷阱

MLE 在一一对应的参数变换下保持最优点映射；MAP 一般没有同样的不变性，因为概率密度本身会随坐标 Jacobian 改变。

设 $\eta=g(\theta)$，后验密度满足

$$
p_\eta(\eta\mid D)
=p_\theta(g^{-1}(\eta)\mid D)
\left|\det J_{g^{-1}}(\eta)\right|
$$

变换后的密度多了 Jacobian 因子，众数可能移动。特别地，正参数 $\theta>0$ 变成 $\eta=\log\theta$ 后：

$$
p_\eta(\eta\mid D)
=p_\theta(e^\eta\mid D)e^\eta
$$

因此「先在 $\theta$ 上求 MAP 再取 log」和「先在 $\eta$ 上定义密度再求 MAP」可能不同。选择参数化时，要说明先验密度是相对于哪个测度定义的。

![似然与先验共同形成后验峰](/assets/probability/svg/maximum-a-posteriori.1.svg)

## 失效模式

**把 MAP 当成后验分布。**MAP 只给一个峰值，不能提供参数尾部、多峰和预测不确定性。

**把 MAP 当成 MLE。**先验的对数项会移动峰值，数据少时尤其明显；只有先验近似常数时才可能接近 MLE。

**把后验均值和 MAP 混用。**偏斜 Beta 后验中，众数和均值不同；对称高斯相同只是特殊情况。

**认为先验只是无害的工程常数。**先验尺度、支持集和参数化会改变后验，必须做敏感性分析。

**把 L2 系数直接等同于先验精度。**损失是按样本求和还是求平均、噪声方差如何缩放，都会改变正则系数和先验的对应关系。

**忽略边界和不适当先验。**后验可能在边界达到峰值，或者根本无法归一化；内部梯度为零不是普适条件。

**在多峰后验上使用单点 Laplace。**一个局部高斯只能看到一个峰，可能完全遗漏其他同样重要的参数区域。

**把 MAP 点预测当成后验预测。**代入一组参数没有积分掉参数不确定性，尤其不能直接产生可信区间。

## 相关词条

- [最大似然](../probability/maximum-likelihood/)：比较不含先验的 MLE 与 MAP 的似然部分。
- [贝叶斯定理](../probability/bayes-theorem/)：定义先验、似然、证据和后验的乘法关系。
- [期望](../probability/expectation/)：解释后验均值、后验预测和风险平均。
- [方差与协方差](../probability/variance-and-covariance/)：理解后验方差、参数协方差和不确定性。
- [变量变换](../probability/change-of-variables/)：说明 MAP 密度在参数化变换下为何会改变。
- [高斯分布](../probability/gaussian-distribution/)：推导高斯共轭先验和 Laplace 近似。
- [交叉熵](../information-theory/cross-entropy/)：连接条件似然、NLL 和分类训练。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：讨论先验惩罚、泛化和模型复杂度。
