---
title: "最大似然：让模型最能解释观测数据"
tags: ["why-models-learn"]
---

**最大似然估计**把已经观察到的数据固定下来，把模型的参数当作变量，选择使这些数据最可能出现的参数。若观测为 $x$，参数为 $\theta$，似然函数是

$$
L(\theta;x)=p_\theta(x)
$$

最大似然估计量为

$$
\widehat\theta_{\mathrm{MLE}}
=\arg\max_\theta L(\theta;x)
$$

独立同分布样本 $x_1,\ldots,x_n$ 下：

$$
L(\theta;x_{1:n})
=\prod_{i=1}^np_\theta(x_i)
$$

最大似然不是说模型已经为真，也不是把参数当成随机变量；它是一种用数据选择参数的准则。本篇先区分概率、似然和后验，再推导伯努利、分类、高斯和指数模型的闭式 MLE，接着解释 score、Fisher 信息、渐近正态和 KL 投影，最后连接 MAP、正则化、交叉熵和神经网络训练。

## 概率、似然和后验的方向不同

同一个数值 $p_\theta(x)$ 可以有三种不同的阅读方式，关键在于「固定谁，改变谁」：

| 名称 | 固定对象 | 变化对象 | 是否需要对变化对象归一化 |
| --- | --- | --- | --- |
| 概率或密度 | 参数 $\theta$ | 数据 $x$ | 对所有 $x$ 归一化 |
| 似然 $L(\theta;x)$ | 观测 $x$ | 参数 $\theta$ | 不要求对 $\theta$ 归一化 |
| 后验 $p(\theta\mid x)$ | 观测 $x$ | 参数 $\theta$ | 对 $\theta$ 归一化 |
| 证据 $p(x)$ | 模型和先验 | 观测 $x$ | 是后验的归一化常数 |

例如 Bernoulli 模型中

$$
p_\theta(x)=\theta^x(1-\theta)^{1-x},
\qquad x\in\{0,1\}
$$

当 $\theta$ 固定、让 $x$ 取 0 和 1 时，两项概率和为 1；当观测固定为 $x=1$、让 $\theta$ 在 $[0,1]$ 变化时，函数 $\theta$ 不需要对参数积分为 1。把似然曲线当成参数的概率分布，会错误地引入一个并不存在的归一化解释。

贝叶斯后验则多了一项先验：

$$
p(\theta\mid x)
=\frac{p_\theta(x)\pi(\theta)}{p(x)}
$$

最大似然只最大化 $p_\theta(x)$；MAP 还最大化先验 $\pi(\theta)$。两者可以给出不同的参数，不能只因为目标函数里都有一个对数就把它们混为一谈。

## 独立样本让似然变成乘积

若给定 $\theta$ 后观测条件独立：

$$
p_\theta(x_1,\ldots,x_n)
=\prod_{i=1}^np_\theta(x_i)
$$

因此联合似然是

$$
L_n(\theta)
=\prod_{i=1}^np_\theta(x_i)
$$

乘积可以表达「每一个观测都要被模型解释」，但直接相乘容易数值下溢。由于对数是严格递增函数，最大化乘积等价于最大化对数似然：

$$
\ell_n(\theta)
=\log L_n(\theta)
=\sum_{i=1}^n\log p_\theta(x_i)
$$

所以

$$
\widehat\theta_{\mathrm{MLE}}
=\arg\max_\theta\ell_n(\theta)
$$

也等价于最小化负对数似然：

$$
\widehat\theta_{\mathrm{MLE}}
=\arg\min_\theta
\left[
-\frac1n\sum_{i=1}^n\log p_\theta(x_i)
\right]
$$

除以固定的 $n$ 不改变最优参数，但把目标变成每个样本的平均损失，便于比较不同批次和不同数据集。机器学习中的 NLL、交叉熵和许多训练损失，都可以从这一行的对数乘积得到。

## Bernoulli 模型：MLE 是样本频率

令 $X_i\sim\operatorname{Bernoulli}(p)$，观测值为 $x_i\in\{0,1\}$。成功总数为

$$
s=\sum_{i=1}^nx_i
$$

联合似然：

$$
\begin{aligned}
L(p)
&=\prod_{i=1}^np^{x_i}(1-p)^{1-x_i}\\
&=p^s(1-p)^{n-s}
\end{aligned}
$$

对数似然：

$$
\ell(p)=s\log p+(n-s)\log(1-p)
$$

在 $0<p<1$ 内求导：

$$
\frac{d\ell}{dp}
=\frac sp-\frac{n-s}{1-p}
$$

令导数为 0：

$$
\frac sp=\frac{n-s}{1-p}
\quad\Longrightarrow\quad
s(1-p)=p(n-s)
\quad\Longrightarrow\quad
\widehat p=\frac sn
$$

二阶导数为

$$
\frac{d^2\ell}{dp^2}
=-\frac{s}{p^2}-\frac{n-s}{(1-p)^2}<0
$$

只要 $0<s<n$，这个驻点就是唯一的最大值。若所有观测都是 1，MLE 位于边界 $\widehat p=1$；若全部都是 0，MLE 位于 $\widehat p=0$。此时不能机械要求内部导数等于 0，边界本身就是最优解。

十次试验中观察到 7 次成功时：

$$
\widehat p=\frac7{10}=0.7
$$

它也是样本频率。大数定律说明频率在合适条件下趋近真实 $p$；最大似然解释了为什么对 Bernoulli 观测，频率正好是使数据似然最大的参数。

## 分类分布：MLE 是各类别频率

若每次观测属于 $K$ 个类别之一，参数为

$$
\boldsymbol p=(p_1,\ldots,p_K),
\qquad
p_k\geq0,\quad
\sum_{k=1}^Kp_k=1
$$

记第 $k$ 类出现次数为 $n_k$，则

$$
\ell(\boldsymbol p)
=\sum_{k=1}^Kn_k\log p_k
$$

用拉格朗日乘子处理归一化约束：

$$
\mathcal L(\boldsymbol p,\lambda)
=\sum_{k=1}^Kn_k\log p_k
+\lambda\left(\sum_{k=1}^Kp_k-1\right)
$$

对每个 $p_k$ 求导：

$$
\frac{\partial\mathcal L}{\partial p_k}
=\frac{n_k}{p_k}+\lambda=0
\quad\Longrightarrow\quad
p_k=-\frac{n_k}{\lambda}
$$

把所有类别相加并使用 $\sum_kp_k=1$：

$$
1=-\frac1\lambda\sum_{k=1}^Kn_k
=-\frac n\lambda
\quad\Longrightarrow\quad
\widehat p_k=\frac{n_k}{n}
$$

因此分类 MLE 仍然是频率。若三类计数为 $(3,2,1)$，则

$$
\widehat{\boldsymbol p}
=\left(\frac12,\frac13,\frac16\right)
$$

零计数类别的 MLE 是 0。真实概率可能不是 0，只是有限样本尚未观察到它；若后续需要避免零概率，常使用平滑先验或正则化，那已经不再是纯最大似然。

## 高斯模型：均值用 n，方差的 MLE 也用 n

设观测独立来自

$$
X_i\sim\mathcal N(\mu,\sigma^2)
$$

联合对数似然：

$$
\ell(\mu,\sigma^2)
=-\frac n2\log(2\pi)
-\frac n2\log\sigma^2
-\frac1{2\sigma^2}
\sum_{i=1}^n(x_i-\mu)^2
$$

对 $\mu$ 求导：

$$
\frac{\partial\ell}{\partial\mu}
=\frac1{\sigma^2}\sum_{i=1}^n(x_i-\mu)
$$

设为 0：

$$
\sum_{i=1}^n(x_i-\widehat\mu)=0
\quad\Longrightarrow\quad
\widehat\mu=\bar x
$$

再令 $v=\sigma^2$，对 $v$ 求导：

$$
\frac{\partial\ell}{\partial v}
=-\frac n{2v}
+\frac1{2v^2}\sum_{i=1}^n(x_i-\mu)^2
$$

代入 $\mu=\bar x$ 并令导数为 0：

$$
\widehat\sigma_{\mathrm{MLE}}^2
=\frac1n\sum_{i=1}^n(x_i-\bar x)^2
$$

注意分母是 $n$，不是无偏样本方差使用的 $n-1$。MLE 的目标是最大化已观察数据的密度；无偏性是另一个准则。有限样本下：

$$
\mathbb E\left[
\widehat\sigma_{\mathrm{MLE}}^2
\right]
=\frac{n-1}{n}\sigma^2
$$

因此高斯方差的 MLE 有向下偏差，但随着 $n$ 增大，比例 $(n-1)/n$ 趋近 1。

对数据 $(1,2,4,5)$：

$$
\bar x=3,\qquad
\sum_{i=1}^4(x_i-\bar x)^2
=4+1+1+4=10
$$

所以

$$
\widehat\sigma_{\mathrm{MLE}}^2=\frac{10}{4}=2.5
$$

而无偏样本方差是 $10/3\approx3.333$。两者都可能是有用的量，但回答的问题不同。

## 几个常见模型的 MLE

把前面的推导放在一起：

| 模型 | 观测的充分统计量 | 最大似然估计 | 直觉 |
| --- | --- | --- | --- |
| Bernoulli($p$) | 成功数 $s$ | $\widehat p=s/n$ | 成功频率 |
| 分类($\boldsymbol p$) | 各类计数 $n_k$ | $\widehat p_k=n_k/n$ | 各类频率 |
| 高斯($\mu,\sigma^2$) | 样本和平方偏差 | $\widehat\mu=\bar x,\ \widehat\sigma^2=\mathrm{SSE}/n$ | 平均位置与平均平方误差 |
| 指数($\lambda$) | 样本和 $\sum x_i$ | $\widehat\lambda=n/\sum x_i=1/\bar x$ | 反向等待时间 |

指数模型的密度为

$$
f(x\mid\lambda)=\lambda e^{-\lambda x},
\qquad x\geq0
$$

对数似然

$$
\ell(\lambda)
=n\log\lambda-\lambda\sum_{i=1}^nx_i
$$

求导即可得到

$$
\frac{d\ell}{d\lambda}
=\frac n\lambda-\sum_{i=1}^nx_i=0
\quad\Longrightarrow\quad
\widehat\lambda=\frac n{\sum_i x_i}
$$

数据越大意味着平均等待时间越长，MLE 就把速率估得越低。这里的参数方向由模型密度决定，不能凭「参数越大模型越复杂」之类的直觉猜。

## Score 和曲率告诉我们估计是否稳定

对数似然的一阶导数叫 score：

$$
U_n(\theta)
=\frac{\partial\ell_n(\theta)}{\partial\theta}
$$

内部最大值通常满足

$$
U_n(\widehat\theta)=0
$$

二阶导数描述曲线在峰值附近有多尖。曲率很大时，参数稍微偏离峰值，似然就明显下降；曲率很小时，一整段参数都能近似解释数据，估计不稳定。

在满足正则条件时，单个观测的 Fisher 信息为

$$
\mathcal I_1(\theta)
=\mathbb E_\theta[U_1(\theta)^2]
=-\mathbb E_\theta\left[
\frac{\partial^2\log p_\theta(X)}{\partial\theta^2}
\right]
$$

独立样本的信息相加：

$$
\mathcal I_n(\theta)=n\mathcal I_1(\theta)
$$

经典渐近理论给出

$$
\sqrt n(\widehat\theta-\theta_0)
\xrightarrow{d}
\mathcal N\left(
0,\mathcal I_1(\theta_0)^{-1}
\right)
$$

等价地，大样本下

$$
\widehat\theta
\approx
\mathcal N\left(
\theta_0,\frac1{n\mathcal I_1(\theta_0)}
\right)
$$

信息越大，估计方差越小。对 Bernoulli 参数：

$$
\mathcal I_1(p)=\frac1{p(1-p)}
$$

所以 MLE 的渐近方差是

$$
\frac1{n\mathcal I_1(p)}
=\frac{p(1-p)}n
$$

这正好与样本频率的精确方差一致。对 $p=0.3,n=100$，标准误为

$$
\sqrt{\frac{0.3\cdot0.7}{100}}
\approx0.0458
$$

Fisher 信息是局部曲率的期望，不是每一个有限样本的精确置信度。边界解、非正则模型、参数不可识别或样本不独立时，正态近似可能不成立。

## MLE 和 KL 散度：大样本为何会找到真实模型

假设真实数据分布是 $p^\star(x)$，候选模型密度是 $q_\theta(x)$。总体平均对数似然为

$$
\mathbb E_{p^\star}[\log q_\theta(X)]
=\int p^\star(x)\log q_\theta(x)\,dx
$$

KL 散度定义为

$$
\operatorname{KL}(p^\star\Vert q_\theta)
=\int p^\star(x)
\log\frac{p^\star(x)}{q_\theta(x)}\,dx
$$

展开：

$$
\begin{aligned}
\operatorname{KL}(p^\star\Vert q_\theta)
&=\int p^\star(x)\log p^\star(x)\,dx
-\int p^\star(x)\log q_\theta(x)\,dx
\end{aligned}
$$

第一项与 $\theta$ 无关，所以最大化总体对数似然等价于最小化

$$
\operatorname{KL}(p^\star\Vert q_\theta)
$$

有限样本的平均对数似然是总体期望的样本平均：

$$
\frac1n\ell_n(\theta)
=\frac1n\sum_{i=1}^n\log q_\theta(x_i)
$$

在适当条件下，大数定律让它趋近总体期望。这就是 MLE 的两层解释：样本层面选择使观测最可能的参数，分布层面选择在模型族中最接近真实分布的参数。

若真实分布不在候选模型族中，MLE 仍可能收敛，但收敛到的是 KL 意义下的最佳近似，不是真实生成机制本身。模型错配时，标准误和似然比检验也需要更谨慎解释。

## MLE 的变换不变性

若 $\widehat\theta$ 最大化 $L(\theta;x)$，且 $\eta=g(\theta)$ 是一一对应的重新参数化，那么

$$
\widehat\eta=g(\widehat\theta)
$$

就是 $\eta$ 的 MLE。因为重新标记参数不会改变每个模型给数据的似然值。

例如若高斯标准差的 MLE 是 $\widehat\sigma^2$，那么标准差的 MLE 是

$$
\widehat\sigma=\sqrt{\widehat\sigma^2}
$$

这不是先对标准差重新优化得到的巧合，而是似然最大值在一一对应变换下被保留。若 $g$ 不是一一对应，需要比较所有映射到同一个 $\eta$ 的参数，不能无条件套用一个分支。

变换不变性不等于估计量无偏性保持。即使 $\widehat\theta$ 无偏，非线性变换 $g(\widehat\theta)$ 通常也不满足

$$
\mathbb E[g(\widehat\theta)]
=g(\theta)
$$

MLE 的定义是最大化似然，不是同时满足所有统计性质。

## MAP、正则化和条件似然

MAP 估计最大化后验：

$$
\widehat\theta_{\mathrm{MAP}}
=\arg\max_\theta
\left[
\log L(\theta;x)+\log\pi(\theta)
\right]
$$

若先验是均值为 0 的高斯分布：

$$
\pi(\theta)\propto
\exp\left(-\frac\lambda2\|\theta\|_2^2\right)
$$

则 MAP 等价于在负对数似然上加入 L2 惩罚：

$$
\arg\min_\theta
\left[
-\ell(\theta)+\frac\lambda2\|\theta\|_2^2
\right]
$$

因此「最大似然」「MAP」「正则化经验风险」可能有相似的优化形式，但它们对参数的解释不同：MLE 不放先验，MAP 把惩罚解释为先验或偏好，纯正则化也可以只是工程上的稳定化选择。

监督学习中，输入 $x_i$ 被视为给定条件，模型最大化标签的条件似然：

$$
L(\theta)
=\prod_{i=1}^n
p_\theta(y_i\mid x_i)
$$

负对数似然为

$$
-\ell(\theta)
=-\sum_{i=1}^n\log p_\theta(y_i\mid x_i)
$$

对 softmax logits $z_i$，单个正确类别 $y_i$ 的损失是

$$
-\log p_\theta(y_i\mid x_i)
=-\log
\frac{e^{z_{i,y_i}}}{\sum_{k=1}^Ke^{z_{i,k}}}
$$

这就是分类交叉熵的概率解释。训练时最小化的是有限样本条件 NLL，不自动保证测试分布上的 NLL 最小；抽样代表性、正则化和独立测试仍然是单独的问题。

## 数值优化和非理想似然

很多模型没有闭式 MLE，需要用梯度、Newton 法或其他优化方法近似求解。实际计算有几个优先级：

1. 在 log-domain 中计算乘积，避免概率连乘下溢；
2. 对 softmax 使用 log-sum-exp 稳定形式；
3. 检查梯度、曲率和参数边界；
4. 用多个初始化或全局结构检查局部极值；
5. 记录停止条件和最终 NLL，而不是只看优化器返回成功。

混合模型是典型的非凸例子：标签交换会造成多个等价峰，某些成分可能塌缩到单个观测附近，EM 或梯度优化可能落在不同局部最大值。参数不可识别时，不同参数可以产生完全相同的分布，单独报告某个参数值没有意义。

即使优化找到了全局最大值，似然也可能在参数边界发散。例如完全可分的逻辑回归中，某些权重不断变大可以持续提高训练似然，却没有有限的 MLE；正则化或重新定义模型是解决方案，不是把优化器再运行几次。

![似然峰值与对数似然求和](/assets/probability/svg/maximum-likelihood.1.svg)

## 失效模式

**把似然当成参数的概率。**固定数据后，似然是参数函数，不自动对参数归一化；要讨论参数后验，必须说明先验和证据。

**把每个概率直接相乘。**长序列的乘积很快下溢，应在 log-domain 中累加对数概率。

**把最大似然说成真实参数必然正确。**有限样本会有估计误差，模型错配时 MLE 可能只是 KL 意义下的最佳近似。

**把高斯方差 MLE 的分母写成 n−1。**$n$ 是最大似然解，$n-1$ 是无偏样本方差的修正；它们回答不同问题。

**忽略边界和零计数。**Bernoulli 全成功、分类零计数和指数速率等都可能产生边界或极端估计，内部一阶导数不一定适用。

**把 MAP 当成 MLE。**正则项或先验会改变目标函数，去掉它们后最优点通常会变化。

**用小样本渐近正态区间做精确承诺。**Fisher 信息和 Wald 区间依赖正则条件，偏斜、边界、弱识别和重尾都会使近似变差。

**忽略样本独立性和部署分布。**训练 NLL 是抽样数据上的似然，重复用户、时间漂移和标签泄漏会改变它对应的总体。

## 相关词条

- [概率空间](../probability/probability-spaces/)：说明概率测度和观测事件的基础。
- [随机变量](../probability/random-variables/)：定义模型中的随机观测和函数。
- [期望](../probability/expectation/)：提供总体对数似然和风险的平均解释。
- [方差与协方差](../probability/variance-and-covariance/)：连接 MLE 的方差、Fisher 信息和估计不确定性。
- [抽样](../probability/sampling/)：说明似然估计的样本从谁、以什么机制产生。
- [中心极限定理](../probability/central-limit-theorem/)：给出 MLE 渐近正态近似所依赖的极限语言。
- [贝叶斯定理](../probability/bayes-theorem/)：解释先验、后验和证据与似然的关系。
- [交叉熵](../information-theory/cross-entropy/)：展开分类 NLL 与交叉熵损失的训练形式。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：讨论训练似然、泛化和惩罚项的张力。
