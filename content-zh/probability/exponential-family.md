---
title: "指数族：把概率模型写成统一的可计算形式"
tags: ["why-models-learn"]
---

**指数族**是一组可以写成「充分统计量与自然参数做内积，再减去对数配分函数」的概率分布：

$$
p_\eta(x)
=h(x)\exp\left(
\eta^{\mathsf T}T(x)-A(\eta)
\right)
$$

其中 $T(x)$ 是充分统计量，$\eta$ 是自然参数，$A(\eta)$ 是对数配分函数，$h(x)$ 是底测度项。伯努利、分类、泊松、指数和固定方差的高斯分布都属于指数族。这个写法把归一化、矩、似然、充分统计量、共轭先验和广义线性模型放进同一套代数中。本篇从归一化推导 $A$ 的作用，再推导它的一阶、二阶导数，最后连接 MLE、正则化、KL 散度和机器学习中的输出层。

## 四个部件分别做什么

先把参数写成自然参数 $\eta$，把观测写成统计量 $T(x)$。连续变量用积分，离散变量用求和；统一记成相对于底测度 $\mu$ 的积分：

$$
A(\eta)
=\log\int h(x)
\exp\left(\eta^{\mathsf T}T(x)\right)
\mathrm d\mu(x)
$$

把这个定义代回密度：

$$
\begin{aligned}
\int p_\eta(x)\,\mathrm d\mu(x)
&=\int h(x)
\exp\left(\eta^{\mathsf T}T(x)-A(\eta)\right)
\mathrm d\mu(x)\\
&=\exp\left(-A(\eta)\right)
\int h(x)\exp\left(\eta^{\mathsf T}T(x)\right)
\mathrm d\mu(x)\\
&=1
\end{aligned}
$$

所以 $A(\eta)$ 不是随手补上的常数，而是把总质量调回 1 所必需的对数归一化因子。四个部件的职责可以这样读：

| 部件 | 作用 | 例子 |
| --- | --- | --- |
| $T(x)$ | 从观测中提取与参数交互的数值 | 伯努利中的 $x$，分类中的独热向量 |
| $\eta$ | 与统计量配对的自然参数 | 伯努利的 $\log\frac{p}{1-p}$ |
| $A(\eta)$ | 保证分布归一化，并编码矩 | 伯努利的 $\log(1+\mathrm e^\eta)$ |
| $h(x)$ | 不随自然参数变化的底测度项 | 泊松中的 $1/x!$ |

这里默认支持集不随参数改变，且 $A(\eta)$ 在所用参数区域内有限。Uniform$(0,\theta)$ 的支持集会随 $\theta$ 改变，它不是这一类正则指数族；把所有分布都强行塞进这个形式，会丢掉边界导数和支持集变化带来的问题。

## 对数配分函数的导数就是矩

指数族最有用的事实来自对 $A$ 求导。先看一维统计量 $T(x)$：

$$
A(\eta)
=\log\int h(x)\exp\left(\eta T(x)\right)\,\mathrm d\mu(x)
$$

设

$$
Z(\eta)
=\int h(x)\exp\left(\eta T(x)\right)\,\mathrm d\mu(x)
$$

则 $A(\eta)=\log Z(\eta)$。对 $\eta$ 求导：

$$
\begin{aligned}
A'(\eta)
&=\frac{Z'(\eta)}{Z(\eta)}\\
&=\frac{\int T(x)h(x)\exp\left(\eta T(x)\right)\,\mathrm d\mu(x)}
{\int h(x)\exp\left(\eta T(x)\right)\,\mathrm d\mu(x)}\\
&=\mathbb E_\eta[T(X)]
\end{aligned}
$$

再求一次导数：

$$
\begin{aligned}
A''(\eta)
&=\mathbb E_\eta[T(X)^2]
-\mathbb E_\eta[T(X)]^2\\
&=\operatorname{Var}_\eta(T(X))
\geq0
\end{aligned}
$$

多维版本是

$$
\nabla A(\eta)
=\mathbb E_\eta[T(X)]
$$

以及

$$
\nabla^2 A(\eta)
=\operatorname{Cov}_\eta(T(X))
$$

因此 $A$ 是凸函数。若统计量的某个线性组合在该分布下还有波动，对应方向的 Hessian 就是正的；若两个不同参数给出完全相同的统计量分布，则会出现零曲率和不可识别方向。指数族把「归一化」和「矩」连接起来，原因只是对指数函数连续求导。

## 伯努利例子：sigmoid 是归一化后的自然参数

伯努利分布满足

$$
p(x\mid p)=p^x(1-p)^{1-x},
\qquad x\in\{0,1\}
$$

令自然参数为 logit：

$$
\eta=\log\frac{p}{1-p}
$$

反解得到

$$
p=\frac{\exp(\eta)}{1+\exp(\eta)}
$$

于是

$$
p(x\mid\eta)
=\frac{\exp(\eta x)}{1+\exp(\eta)}
=\exp\left(
\eta x-\log(1+\exp(\eta))
\right)
$$

所以

$$
T(x)=x,
\qquad
A(\eta)=\log(1+\exp(\eta)),
\qquad
h(x)=1
$$

由导数公式：

$$
A'(\eta)=p,
\qquad
A''(\eta)=p(1-p)
$$

当 $p=0.8$ 时，$\eta=\log4\approx1.3863$，$A(\eta)=\log5\approx1.6094$。此时

$$
A'(\eta)=0.8,
\qquad
A''(\eta)=0.8(1-0.8)=0.16
$$

第一项是成功率，第二项是伯努利方差。神经网络最后一层使用 sigmoid，不只是把任意实数压到 $(0,1)$；它也正好把伯努利分布的自然参数变回均值参数。

## 似然变成充分统计量的线性函数

对 $n$ 个条件独立样本，指数族联合密度为

$$
\begin{aligned}
p_\eta(x_{1:n})
&=\prod_{i=1}^n
h(x_i)\exp\left(
\eta^{\mathsf T}T(x_i)-A(\eta)
\right)\\
&=\left(\prod_{i=1}^nh(x_i)\right)
\exp\left(
\eta^{\mathsf T}\sum_{i=1}^nT(x_i)
-nA(\eta)
\right)
\end{aligned}
$$

对数似然因此是

$$
\ell(\eta)
=\eta^{\mathsf T}\sum_{i=1}^nT(x_i)
-nA(\eta)
+\sum_{i=1}^n\log h(x_i)
$$

最后一项与 $\eta$ 无关。根据因子分解定理，样本对自然参数的相关信息被

$$
S=\sum_{i=1}^nT(x_i)
$$

保留下来，这个 $S$ 就是充分统计量。若 MLE 位于自然参数空间内部，令梯度为 0：

$$
\nabla\ell(\eta)
=S-n\nabla A(\eta)
=0
$$

所以 MLE 满足矩匹配：

$$
\nabla A(\widehat\eta)
=\frac Sn
=\overline T
$$

伯努利 10 次试验中 7 次成功时，$\overline T=0.7$。因此

$$
\widehat p=0.7,
\qquad
\widehat\eta
=\log\frac{0.7}{0.3}
=\log\frac73
\approx0.8473
$$

把 MLE 写成自然参数时，优化目标的导数就是「观测统计量减去模型期望统计量」。训练过程会把这两者推向相等，而不是直接凭空寻找一个神秘的最优数字。

## 常见分布都可以放进同一张表

下面只列出最小的自然参数表示。多参数高斯和分类分布还需要注意参数约束与不可识别性。

| 分布 | $T(x)$ | 自然参数 $\eta$ | $A(\eta)$ |
| --- | --- | --- | --- |
| 伯努利 | $x$ | $\log\frac p{1-p}$ | $\log(1+\exp\eta)$ |
| 分类 | 独热向量 $e_x$ | 类别 logit | $\log\sum_k\exp\eta_k$ |
| 泊松 | $x$ | $\log\lambda$ | $\exp\eta$ |
| 固定方差高斯 | $x$ | $\mu/\sigma^2$ | $\sigma^2\eta^2/2$ |
| 指数 | $x$ | $-\lambda<0$ | $-\log(-\eta)$ |

### 泊松分布

泊松分布的概率质量为

$$
p(x\mid\lambda)
=\frac{\lambda^x\exp(-\lambda)}{x!},
\qquad x=0,1,2,\ldots
$$

令 $\eta=\log\lambda$，就得到

$$
p(x\mid\eta)
=\frac1{x!}
\exp\left(
\eta x-\exp(\eta)
\right)
$$

所以 $T(x)=x$，$A(\eta)=\exp(\eta)$。于是

$$
A'(\eta)=\exp(\eta)=\lambda,
\qquad
A''(\eta)=\lambda
$$

均值和方差相等是泊松分布的结果，不是指数族的普遍结论；普遍结论只有「一阶导数给均值，二阶导数给方差」。

如果观测是 $1,2,3$，样本均值为 2，泊松 MLE 满足

$$
\widehat\lambda=2,
\qquad
\widehat\eta=\log2\approx0.6931
$$

### 固定方差的高斯分布

若 $\sigma^2$ 已知，只把均值 $\mu$ 当作参数：

$$
p(x\mid\mu)
=\frac1{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$$

令 $\eta=\mu/\sigma^2$，整理关于 $x$ 的项：

$$
p(x\mid\eta)
=h(x)\exp\left(
\eta x-\frac{\sigma^2\eta^2}{2}
\right)
$$

其中

$$
h(x)
=\frac1{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{x^2}{2\sigma^2}\right)
$$

因此 $A(\eta)=\sigma^2\eta^2/2$，并且

$$
A'(\eta)=\sigma^2\eta=\mu,
\qquad
A''(\eta)=\sigma^2
$$

固定方差高斯的曲率是常数，所以均值 MLE 就是样本均值。若把方差也当作未知参数，仍可写成二维指数族，但自然统计量会变成 $(x,x^2)$，自然参数还必须满足负二次项的约束。

### 指数分布

速率参数为 $\lambda>0$ 的指数密度是

$$
p(x\mid\lambda)
=\lambda\exp(-\lambda x),
\qquad x\geq0
$$

取 $\eta=-\lambda<0$：

$$
p(x\mid\eta)
=\exp\left(
\eta x+\log(-\eta)
\right)
=\exp\left(
\eta x-[-\log(-\eta)]
\right)
$$

所以自然参数空间不是整个实数轴，而是 $\eta<0$，且 $A(\eta)=-\log(-\eta)$。遗漏这个域约束，会把一个合法速率变成负数。

## 共轭先验也有统一写法

指数族的似然只通过 $S=\sum_iT(x_i)$ 和样本数 $n$ 依赖自然参数，因此可以构造形状相同的先验：

$$
\pi(\eta\mid\boldsymbol\chi,\nu)
\propto
\exp\left(
\eta^{\mathsf T}\boldsymbol\chi
-\nu A(\eta)
\right)
$$

把似然和先验相乘：

$$
\begin{aligned}
p(\eta\mid x_{1:n})
&\propto
\exp\left(
\eta^{\mathsf T}S-nA(\eta)
\right)
\exp\left(
\eta^{\mathsf T}\boldsymbol\chi-\nu A(\eta)
\right)\\
&\propto
\exp\left(
\eta^{\mathsf T}(\boldsymbol\chi+S)
-(\nu+n)A(\eta)
\right)
\end{aligned}
$$

后验更新只需要

$$
\boldsymbol\chi_{\mathrm{post}}
=\boldsymbol\chi+S,
\qquad
\nu_{\mathrm{post}}=\nu+n
$$

伯努利情形中，Beta$(a,b)$ 先验对应

$$
\boldsymbol\chi=a-1,
\qquad
\nu=a+b-2
$$

观察 $s$ 次成功和 $f$ 次失败后：

$$
\boldsymbol\chi_{\mathrm{post}}=a+s-1,
\qquad
\nu_{\mathrm{post}}=a+b+s+f-2
$$

这正好对应 Beta$(a+s,b+f)$。MAP 篇中的 Beta–Bernoulli 和 Dirichlet–分类推导，是这个统一更新式在具体分布上的展开。

## 凸性、Fisher 信息和优化

由

$$
\nabla^2A(\eta)
=\operatorname{Cov}_\eta(T(X))
\succeq0
$$

可知负对数似然的 Hessian 是

$$
\nabla^2[-\ell(\eta)]
=n\nabla^2A(\eta)
\succeq0
$$

在自然参数内部，负对数似然是凸函数；若协方差矩阵正定，它还是严格凸的，内部最优点至多一个。Fisher 信息正好是同一个矩阵：

$$
\mathcal I_n(\eta)
=n\nabla^2A(\eta)
=n\operatorname{Cov}_\eta(T(X))
$$

这解释了为什么 Fisher 信息会衡量参数的可辨识程度：统计量波动越丰富，曲率越大，似然峰越尖；统计量几乎不变时，某个方向的曲率接近 0，数据就难以区分相近参数。

在有输入 $z_i$ 的广义线性模型中，若自然参数为 $\eta_i=z_i^{\mathsf T}w$，负对数似然的 Hessian 是

$$
\nabla_w^2[-\ell(w)]
=\sum_i
\nabla^2A(\eta_i)z_iz_i^{\mathsf T}
\succeq0
$$

逻辑回归的交叉熵因此是凸的；神经网络把 $z_i$ 本身也变成参数的非线性函数后，整体训练问题就不再由指数族的凸性保证。

## 指数族和机器学习输出层

给定网络输出 $z$，把它当作自然参数，就能得到对应分布：

| 预测任务 | 分布 | 网络输出与自然参数的关系 | 常见逆链接 |
| --- | --- | --- | --- |
| 二分类 | 伯努利 | $\eta=z$ | $p=\operatorname{sigmoid}(z)$ |
| 多分类 | 分类 | $\eta_k=z_k$ | $\operatorname{softmax}(z)$ |
| 计数 | 泊松 | $\eta=z$ | $\lambda=\exp(z)$ |
| 连续回归 | 固定方差高斯 | $\eta=z/\sigma^2$ | $\mu=z$ |

训练时最小化负对数似然。二分类的每个样本损失是

$$
-\log p(y\mid z)
=-yz+\log(1+\exp z)
$$

其中 $y\in\{0,1\}$。对 $z$ 求导：

$$
\frac{\partial}{\partial z}
\left[-yz+A(z)\right]
=-y+A'(z)
=p-y
$$

所以误差项「预测概率减标签」并不是单独设计的口诀，而是对数配分函数导数等于均值的直接结果。

## 指数族中的 KL 散度是 $A$ 的 Bregman 差

同一个底测度下，两个自然参数 $\eta$ 和 $\eta'$ 对应的分布满足

$$
\begin{aligned}
\operatorname{KL}(p_\eta\Vert p_{\eta'})
&=\mathbb E_\eta
\left[
(\eta-\eta')^{\mathsf T}T(X)
-A(\eta)+A(\eta')
\right]\\
&=A(\eta')-A(\eta)
-\nabla A(\eta)^{\mathsf T}(\eta'-\eta)
\end{aligned}
$$

右侧正是凸函数 $A$ 的 Bregman 差，因此非负。以伯努利为例，真实分布成功率为 $p=0.8$，近似分布为 $q=0.5$ 时：

$$
\operatorname{KL}(p\Vert q)
=0.8\log\frac{0.8}{0.5}
+0.2\log\frac{0.2}{0.5}
\approx0.1927
$$

交叉熵可以分解为

$$
H(p,q)=H(p)+\operatorname{KL}(p\Vert q)
$$

当真实分布 $p$ 固定时，最小化交叉熵和最小化 KL 散度选出同一个 $q$。这给出了分类交叉熵与概率模型拟合之间的精确联系。

![指数族的统一结构](/assets/probability/svg/exponential-family.1.svg)

## 失效模式

**支持集随参数变化**：Uniform$(0,\theta)$ 这类模型的积分边界依赖参数，常规的「对数似然求导等于期望匹配」可能失效。先检查支持集是否固定，再套指数族公式。

**自然参数有边界**：指数分布要求 $\eta<0$，高斯自然参数要求二次项保持可积。优化器若在无约束实数域里搜索，必须用重参数化或显式约束。

**参数化不可识别**：分类分布的所有自然参数同时加同一个常数，softmax 概率不变。可以固定一个基准 logit，或让优化器在冗余方向上自行处理；解释 Hessian 时不能把这个零方向误判成数据没有信息。

**充分统计量不等于现实充分**：$S=\sum_iT(x_i)$ 对指定指数族模型是充分统计量，但若真实数据来自混合分布、存在离群点或有时间依赖，压缩后可能无法表达模型失配。充分性是相对于模型族说的。

**边界 MLE**：分类中某类计数为 0 时，MLE 可能位于概率单纯形边界；逻辑回归完全可分时，权重可能不断增大而没有有限最优点。此时需要先验、正则化或重新检查数据设计。

**把每个损失都叫交叉熵**：只有在指定输出分布和对数似然后，损失才有概率解释。平方损失对应固定方差高斯的负对数似然，泊松输出对应的损失则包含 $\exp(z)$ 项，不能只看一个「预测减标签」的梯度就混淆模型。

## 相关词条

- [期望](../probability/expectation/)：指数族对数配分函数的一阶导数给出期望。
- [方差与协方差](../probability/variance-and-covariance/)：二阶导数是统计量的协方差矩阵。
- [连续分布](../probability/continuous-distributions/)：连续密度与积分归一化的基础。
- [离散分布](../probability/discrete-distributions/)：伯努利、分类和泊松等离散模型的共同背景。
- [最大似然](../probability/maximum-likelihood/)：指数族 MLE 的矩匹配方程。
- [最大后验](../probability/maximum-a-posteriori/)：指数族共轭先验和正则化的具体用法。
- [交叉熵](../information-theory/cross-entropy/)：负对数似然与分类损失的联系。
- [KL 散度](../information-theory/kl-divergence/)：指数族中由对数配分函数生成的 Bregman 差。
