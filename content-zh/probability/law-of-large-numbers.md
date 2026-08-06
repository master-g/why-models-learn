---
title: "大数定律：样本平均何时靠近总体期望"
tags: ["why-models-learn"]
---

**大数定律**说明，在合适的抽样条件下，样本平均会随着样本数量增加而靠近总体期望。若 $X_1,X_2,\ldots$ 是来自同一分布的样本，令

$$
\bar X_n=\frac1n\sum_{i=1}^nX_i
$$

那么大数定律讨论的是

$$
\bar X_n\longrightarrow\mu=\mathbb E[X]
$$

这里的「靠近」有不同强度：弱大数定律说偏离固定阈值的概率趋于 0，强大数定律说几乎每条随机样本路径最终都收敛。本篇先在独立同分布和有限方差的条件下用方差与 Chebyshev 不等式推导弱收敛，再解释强收敛、收敛速度、有限样本界、抽样依赖和机器学习中的经验风险。

## 样本平均在平均意义上没有偏差

设 X 的期望为

$$
\mu=\mathbb E[X]
$$

对 n 个同分布样本

$$
X_1,\ldots,X_n
$$

定义样本平均

$$
\bar X_n=\frac{X_1+\cdots+X_n}{n}
$$

期望的线性性立即给出

$$
\begin{aligned}
\mathbb E[\bar X_n]
&=\mathbb E\left[\frac1n\sum_{i=1}^nX_i\right]\\
&=\frac1n\sum_{i=1}^n\mathbb E[X_i]\\
&=\frac1n\cdot n\mu\\
&=\mu
\end{aligned}
$$

所以样本平均对总体均值是无偏的，而且任意 n 都成立。这个等式只说重复做很多组大小为 n 的实验后，各组平均值的平均等于 μ；它还没有说某一组的 $\bar X_n$ 一定很接近 μ。接近程度要看样本平均自身的方差。

## 独立性让样本平均的方差缩小

先假设 $X_1,\ldots,X_n$ 相互独立，且每个样本的方差都是

$$
\operatorname{Var}(X_i)=\sigma^2<\infty
$$

由独立样本的协方差为 0：

$$
\begin{aligned}
\operatorname{Var}(\bar X_n)
&=\operatorname{Var}\left(\frac1n\sum_{i=1}^nX_i\right)\\
&=\frac1{n^2}\sum_{i=1}^n\operatorname{Var}(X_i)\\
&=\frac1{n^2}\cdot n\sigma^2\\
&=\frac{\sigma^2}{n}
\end{aligned}
$$

样本平均的标准差是

$$
\operatorname{SD}(\bar X_n)
=\frac{\sigma}{\sqrt n}
$$

它通常称为均值的标准误。样本数增加 4 倍，标准误只减半；要把典型波动减小到原来的十分之一，通常要增加约 100 倍样本。

这里发生了两个不同的事实：

1. $\mathbb E[\bar X_n]=\mu$ 表示平均值没有系统偏移；
2. $\operatorname{Var}(\bar X_n)=\sigma^2/n$ 表示不同组平均值的散布变窄。

大数定律依赖第二件事把概率质量集中到 μ 附近。无偏本身不能替代集中。

## Chebyshev 不等式给出弱大数定律

对任意有有限方差的随机变量 Z，Chebyshev 不等式给出

$$
P\bigl(|Z-\mathbb E[Z]|\geq\varepsilon\bigr)
\leq
\frac{\operatorname{Var}(Z)}{\varepsilon^2},
\qquad\varepsilon>0
$$

令 $Z=\bar X_n$，代入上一节的期望和方差：

$$
\begin{aligned}
P\bigl(|\bar X_n-\mu|\geq\varepsilon\bigr)
&\leq
\frac{\operatorname{Var}(\bar X_n)}{\varepsilon^2}\\
&=\frac{\sigma^2}{n\varepsilon^2}
\end{aligned}
$$

当 ε 固定、n 趋于无穷时，

$$
\frac{\sigma^2}{n\varepsilon^2}\longrightarrow0
$$

所以

$$
P\bigl(|\bar X_n-\mu|\geq\varepsilon\bigr)
\longrightarrow0
$$

这就是一个常见的弱大数定律表述：

$$
\bar X_n\xrightarrow{P}\mu
$$

符号 $\xrightarrow{P}$ 读作「依概率收敛」。它不要求每次实验的平均值都单调靠近 μ，也不要求从某个 n 开始永远在 ε 邻域内；它要求在 n 很大时，仍然偏离 ε 或更多的概率可以任意小。

Chebyshev 证明使用了有限方差，是一个方便而不是最弱的条件。独立同分布且只有有限期望时，仍有经典的弱大数定律；但没有有限方差，就不能沿用上面的 $\sigma^2/n$ 计算和 Chebyshev 界。

## 一个公平骰子的数字界

公平六面骰的期望和方差是

$$
\mu=3.5,\qquad
\sigma^2=\frac{35}{12}
$$

掷 n 次并求平均：

$$
\mathbb E[\bar X_n]=3.5,\qquad
\operatorname{Var}(\bar X_n)=\frac{35}{12n}
$$

当 $n=100$ 时，标准误为

$$
\sqrt{\frac{35}{12\cdot100}}
\approx0.1708
$$

取 $\varepsilon=0.5$，Chebyshev 给出

$$
\begin{aligned}
P(|\bar X_{100}-3.5|\geq0.5)
&\leq
\frac{35/12}{100\cdot0.5^2}\\
&=\frac{35}{300}\\
&\approx0.1167
\end{aligned}
$$

这是一个保证上界，不是实际概率的精确值。因为 Chebyshev 只使用均值和方差，界通常会比知道完整分布后的精确计算宽松。它仍然说明：在固定误差阈值下，右侧随 n 按 $1/n$ 下降。

同一个骰子如果掷 $10\,000$ 次，标准误约为

$$
\sqrt{\frac{35}{12\cdot10000}}
\approx0.0171
$$

样本平均通常会比 100 次时更稳定，但某一次实验仍可能出现一段偏离均值较大的结果。定律控制概率，不会替每次有限实验消除随机性。

![样本平均随着样本数增加收窄到总体均值附近](/assets/probability/svg/law-of-large-numbers.1.svg)

## 弱大数定律和强大数定律

弱大数定律的说法是

$$
\forall\varepsilon>0,\quad
P(|\bar X_n-\mu|>\varepsilon)\to0
$$

它把每个 n 看成一个新的随机实验，只关注第 n 个样本平均落在邻域外的概率。

强大数定律写成

$$
\bar X_n\xrightarrow{\text{a.s.}}\mu
$$

其中 a.s. 是 almost surely，中文通常译为「几乎处处」或「几乎必然」。它说的是整条无限样本序列：

$$
P\left(
\lim_{n\to\infty}\bar X_n=\mu
\right)=1
$$

概率为 1 的事件允许存在一个概率为 0 的例外集合。强大数定律不是说逻辑上每一条序列都收敛，也不是说存在一个统一的有限 n 让所有未来平均值都落在指定误差内。

在独立同分布且 $\mathbb E[|X|]<\infty$ 时，强大数定律成立：

$$
\bar X_n\xrightarrow{\text{a.s.}}\mathbb E[X]
$$

有限方差的条件足以推出它，但不是强大数定律的最一般条件。弱收敛和强收敛都指向同一个总体均值，差别在于控制单个 n 的概率，还是控制整条样本路径的最终行为。

## 样本平均不是每次都单调靠近

考虑只取 0 和 1 的伯努利变量，$P(X=1)=p$。若 n 次样本中成功次数为

$$
S_n=X_1+\cdots+X_n
$$

则

$$
\bar X_n=\frac{S_n}{n}
$$

是成功频率。新样本可能让频率向 μ 靠近，也可能暂时把它推远：

$$
\bar X_{10}=0.6,\qquad
\bar X_{11}=\frac7{11}\approx0.636
$$

即使真实 $p=0.5$，有限前缀也可以交替出现偏高和偏低。大数定律不说「每一步都改善」，而说随着前缀变长，偏离给定阈值的概率和最终不收敛的样本路径都受到控制。

对伯努利样本：

$$
\mathbb E[\bar X_n]=p,\qquad
\operatorname{Var}(\bar X_n)=\frac{p(1-p)}{n}
$$

取 $p=0.3$、$n=100$：

$$
\operatorname{SD}(\bar X_{100})
=\sqrt{\frac{0.3\cdot0.7}{100}}
\approx0.0458
$$

样本频率通常以 $1/\sqrt n$ 的尺度波动。想把频率估计得更精细，必须增加观测量，而不是只重复查看同一批观测。

## 函数平均：经验风险和蒙特卡洛

大数定律可以作用于任何期望存在的函数 $g(X)$。令

$$
Y_i=g(X_i)
$$

若 $Y_i$ 仍然独立同分布且 $\mathbb E[|g(X)|]<\infty$，则

$$
\frac1n\sum_{i=1}^ng(X_i)
\longrightarrow
\mathbb E[g(X)]
$$

这给出蒙特卡洛估计：

$$
\widehat{\mathbb E}[g(X)]
=\frac1n\sum_{i=1}^ng(X_i)
$$

它的目标不是估计 $\mathbb E[X]$ 的某个替代值，而是直接用样本平均逼近函数期望。比如对随机输入的损失 $\ell_\theta(X,Y)$：

$$
R(\theta)
=\mathbb E[\ell_\theta(X,Y)]
$$

经验风险是

$$
\widehat R_n(\theta)
=\frac1n\sum_{i=1}^n
\ell_\theta(x_i,y_i)
$$

在样本来自总体分布且满足相应独立性条件时，经验风险会趋向期望风险。这里的“趋向”是统计保证，不等于当前训练集的平均损失已经精确等于总体风险。

mini-batch 梯度也有相同的平均结构。若单样本梯度为

$$
G_i=\nabla_\theta\ell_\theta(X_i,Y_i)
$$

批次平均为

$$
\widehat G_B=\frac1B\sum_{i=1}^BG_i
$$

其期望在无偏抽样时等于总体梯度的期望：

$$
\mathbb E[\widehat G_B]
=\mathbb E[G_1]
$$

当批次增大且样本之间近似独立时，梯度噪声的方差会下降。批次内重复同一用户、同一视频相邻帧或经过泄漏的数据，不符合简单的独立样本模型；“batch 更大”不一定带来与独立样本相同的方差缩减。

## 相关样本时，方差不一定按 $1/n$ 下降

对一般样本平均，完整方差公式是

$$
\operatorname{Var}(\bar X_n)
=\frac1{n^2}
\left(
\sum_{i=1}^n\operatorname{Var}(X_i)
+2\sum_{i<j}\operatorname{Cov}(X_i,X_j)
\right)
$$

独立性把所有交叉协方差置为 0，才得到 $\sigma^2/n$。若每一对样本都有相同的正协方差 c，且方差都是 $\sigma^2$：

$$
\operatorname{Var}(\bar X_n)
=\frac{\sigma^2}{n}
+\frac{n-1}{n}c
$$

当 n 很大时，第一项趋于 0，但第二项趋于 c。继续增加样本数量也不能消除由共同来源产生的波动。

例如同一用户的 1000 条行为记录可能有很强的共同偏好。把它们当成 1000 个独立用户，会低估均值估计的标准误；更合理的分析要以用户为抽样单位，或显式建模组内相关。

负协方差有时能加速平均的稳定，但不能把“有相关性”统一当成好或坏。关键是协方差项如何随 n 变化，以及抽样单位是否与问题中的总体一致。

## 重尾和期望不存在的边界

大数定律首先需要一个要收敛到的总体期望。若 $\mathbb E[|X|]$ 不存在，写出

$$
\bar X_n\longrightarrow\mathbb E[X]
$$

本身就没有目标。前面方差有限的证明还额外需要

$$
\mathbb E[X^2]<\infty
$$

以便使用 $\operatorname{Var}(\bar X_n)=\sigma^2/n$。

重尾数据中，极端观测出现的频率可能很低，但每次出现对总和的影响很大。样本平均仍可能在特定分布条件下收敛，但有限样本的波动远大于高斯直觉；如果均值不存在，样本平均也没有理由稳定到一个固定常数。

截断、稳健均值和分位数可以改变估计目标或减弱极端值影响，但它们不是无条件修复。使用任何“平均值会稳定”的说法前，要先说明总体量存在、抽样单位和尾部条件。

## 失效模式

**把大数定律当成小样本保证。**它描述 n 趋于无穷时的极限，不能直接把 n=10 的平均值宣布为可靠。

**把无偏当成已经接近。**$\mathbb E[\bar X_n]=\mu$ 对每个 n 成立，但单次平均仍可能远离 μ；还要看方差或尾部界。

**认为样本平均会逐步单调靠近。**加入一个新样本可能让平均值暂时远离总体均值，定律只约束极限概率或几乎必然的最终行为。

**忘记独立性在方差推导中的位置。**$\sigma^2/n$ 来自交叉协方差为零；相关样本要保留所有协方差项。

**把 Chebyshev 上界当精确概率。**它只使用均值和方差，通常比利用完整分布得到的概率更宽松。

**混用弱收敛和强收敛。**依概率收敛控制每个 n 的偏离概率，几乎处处收敛控制整条无限样本路径；两者的量词不同。

**只看样本数量，不看抽样单位。**同一用户、同一设备或同一视频片段产生的多条记录可能不是多个独立总体样本。

**对不存在的期望谈收敛。**重尾分布可能没有有限均值；先检查 $\mathbb E[|X|]$，再选择均值、截断均值或其他统计量。

**把大数定律和中心极限定理混为一谈。**大数定律告诉平均值趋向哪里；中心极限定理描述缩放后的平均值如何近似分布，精度和问题不同。

## 相关词条

- [期望](../probability/expectation/)：定义总体期望、函数期望和经验风险。
- [方差与协方差](../probability/variance-and-covariance/)：推导样本平均的方差和相关样本的交叉项。
- [独立性](../probability/independence/)：说明独立抽样为何让协方差项消失。
- [随机变量](../probability/random-variables/)：提供样本和指标变量的概率对象。
- [中心极限定理](../probability/central-limit-theorem/)：在大数定律之后描述均值的极限分布。
- [抽样](../probability/sampling/)：讨论抽样机制、估计目标和样本代表性。
- [极大似然](../probability/maximum-likelihood/)：使用经验对数似然逼近期望风险。
