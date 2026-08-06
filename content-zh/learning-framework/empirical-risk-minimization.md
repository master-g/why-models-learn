---
title: "经验风险最小化：训练如何在样本上选择规则"
tags: ["why-models-learn"]
---

经验风险最小化把「训练」写成一个明确的选择问题：在假设空间中找到使样本平均损失尽可能小的规则。它是监督学习、最大似然、最小二乘和许多正则化方法的共同骨架，但不是一个自动解决优化问题的算法。经验风险只看有限样本，最小化得再好也不等于总体风险最小；学习率、批次、正则化、参数化和停止时机都会决定我们是否真的接近了这个目标。本文先区分经验风险和总体风险，再推导常数 MSE、最小二乘和 Bernoulli 最大似然，说明正则化与小批次如何改变实际选择，最后把统计误差、优化误差和泛化误差拆开。

![经验风险最小化从样本损失平均到候选规则选择](/assets/learning-framework/svg/empirical-risk-minimization.1.svg)

## 经验风险是样本上的平均损失

给定带标签样本

$$
S=\{(x_i,y_i)\}_{i=1}^{n}
\sim P^n
$$

和损失函数 $\ell(h(x),y)$，规则 $h$ 的经验风险是

$$
\widehat R_S(h)
=\frac1n\sum_{i=1}^{n}
\ell\bigl(h(x_i),y_i\bigr).
$$

它只回答一个问题：这条规则在已经看到的 $n$ 个样本上平均犯了多大的错。真正关心的总体风险是

$$
R(h)
=\mathbb E_{(X,Y)\sim P}
\left[\ell\bigl(h(X),Y\bigr)\right].
$$

两者的差不是一个固定的模型属性，而会随样本集变化：

$$
\operatorname{gap}_S(h)
=R(h)-\widehat R_S(h).
$$

| 量 | 计算方式 | 依赖什么 | 训练时能否直接知道 |
| --- | --- | --- | --- |
| 经验风险 | 样本损失的平均 | 当前数据集与规则 | 能 |
| 总体风险 | 目标分布下的期望损失 | 未知分布与规则 | 通常不能 |
| 泛化间隙 | 总体风险减经验风险 | 样本、规则与分布 | 通常不能直接知道 |
| 验证风险 | 独立验证样本上的平均损失 | 验证样本与规则 | 能估计 |

即使每个样本都独立同分布，$\widehat R_S(h)$ 仍然是随机量。换一批样本，经验风险函数和它的最优规则都可能改变。经验风险最小化的泛化前提不是“训练集足够大”一句话，而是样本对部署分布有代表性、候选空间受到控制，且没有把验证或测试信息反复泄漏回选择过程。

## ERM 把拟合定义成一次选择

经验风险最小化的形式是

$$
\widehat h_{\mathrm{ERM}}
\in
\operatorname*{arg\,min}_{h\in\mathcal H}
\widehat R_S(h).
$$

这里写 $\in$ 而不是 $=$，因为可能存在多个经验风险相同的最小点。算法需要额外的平局规则，例如选择更小范数、更平滑、参数更简单或初始化路径到达的那个解。这个选择规则本身就是归纳偏置的一部分。

在有限候选空间中，ERM 可以被理解为逐个计算损失并选择最小者；在连续参数空间中，通常只能通过优化算法寻找近似解。若实际输出 $\widetilde h$ 满足

$$
\widehat R_S(\widetilde h)
\le
\inf_{h\in\mathcal H}\widehat R_S(h)
+\varepsilon_{\mathrm{opt}},
$$

就称它在经验目标上达到 $\varepsilon_{\mathrm{opt}}$ 近似。这个误差只说明离训练集上的最优差多少，不说明总体风险，也不说明最优规则本身属于合适的假设空间。

ERM 还依赖损失函数。相同输入和标签，MSE、MAE、交叉熵或代价加权损失会产生不同的经验目标；“训练损失下降”只有在损失和部署目标一致时才有直接意义。

## 平方损失把 ERM 变成均值或最小二乘

先看最简单的常数假设空间 $h_c(x)=c$。平方损失下：

$$
\widehat R(c)
=\frac1n\sum_{i=1}^{n}(c-y_i)^2.
$$

对 $c$ 求导：

$$
\frac{\mathrm d\widehat R}{\mathrm dc}
=\frac2n\sum_{i=1}^{n}(c-y_i).
$$

令导数为零：

$$
\begin{aligned}
0
&=\sum_i(c-y_i)
=nc-\sum_i y_i,\\
\widehat c_{\mathrm{ERM}}
&=\frac1n\sum_i y_i
=\bar y.
\end{aligned}
$$

因此在常数空间里，MSE 的 ERM 是样本均值。标签 $(1,3,5,7)$ 的均值是 4，经验 MSE 为

$$
\widehat R(4)
=\frac{(4-1)^2+(4-3)^2+(4-5)^2+(4-7)^2}{4}
=5.
$$

把假设空间扩大到线性函数 $h_\theta(x)=\theta_0+\theta_1x$，就得到最小二乘。矩阵记号下，在 $X^\mathsf TX$ 可逆时：

$$
\widehat{\boldsymbol\theta}
=\left(X^\mathsf TX\right)^{-1}X^\mathsf Ty.
$$

这不是另一种学习哲学，而是特定假设空间和平方损失下 ERM 可以写出闭式解的情形。若特征共线、样本不足或模型加入非线性，闭式公式可能不存在或不再代表我们真正想优化的对象。

## 最大似然是负对数损失下的 ERM

如果模型给出条件概率 $p_\theta(y\mid x)$，可以使用负对数似然损失：

$$
\ell_\theta(x,y)
=-\log p_\theta(y\mid x).
$$

经验风险变为

$$
\widehat R_{\mathrm{nll}}(\theta)
=-\frac1n\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i).
$$

最小化它等价于最大化样本条件似然：

$$
\begin{aligned}
\widehat\theta_{\mathrm{MLE}}
&\in\operatorname*{arg\,max}_\theta
\prod_{i=1}^{n}p_\theta(y_i\mid x_i)\\
&=\operatorname*{arg\,max}_\theta
\sum_{i=1}^{n}\log p_\theta(y_i\mid x_i).
\end{aligned}
$$

例如没有输入特征，只有四次 Bernoulli 观测 $(1,1,0,1)$，模型参数是正类概率 $p$。负对数似然 ERM 选择成功比例：

$$
\widehat p
=\frac{1+1+0+1}{4}
=0.75.
$$

所以最大似然不是与 ERM 无关的另一套机制：它是在选定概率模型和对数损失后得到的经验风险最小化。若概率模型不适合数据生成过程，似然最优也只是在错误的候选空间中最优。

## 正则化把选择偏好写进目标

当假设空间较大或样本较少时，常把复杂度惩罚加入经验风险：

$$
\widehat h_\lambda
\in
\operatorname*{arg\,min}_{h\in\mathcal H}
\left[
\widehat R_S(h)+\lambda\Omega(h)
\right],
\qquad
\lambda\ge0.
$$

$\Omega$ 不是数据损失，而是对规则复杂度、参数范数、稀疏性或平滑性的额外偏好。它会提高训练目标中复杂规则的代价，因此可能牺牲一些训练拟合换取更稳定的选择。

用一个标量看收缩效果。设未正则化目标是

$$
J(\theta)=(\theta-3)^2
$$

加入 $\lambda\theta^2$ 且 $\lambda=1$：

$$
\begin{aligned}
J_\lambda(\theta)
&=(\theta-3)^2+\theta^2,\\
0
&=\frac12J_\lambda'(\theta)
=(\theta-3)+\theta,\\
\widehat\theta_\lambda
&=\frac32=1.5.
\end{aligned}
$$

未正则化最优点是 $\theta=3$，正则化把它向零拉近。这个例子不是说零附近总是正确，而是显示惩罚项如何把选择偏好写进同一个优化目标。增大 $\lambda$ 会进一步收缩，但过强时可能带来表达不足。

惩罚形式和硬约束形式常有关联：

$$
\begin{aligned}
\text{惩罚形式}\quad
&\min_h\ \widehat R_S(h)+\lambda\Omega(h),\\
\text{约束形式}\quad
&\min_h\ \widehat R_S(h)
\quad\text{subject to}\quad
\Omega(h)\le B.
\end{aligned}
$$

在凸性、连续性和参数范围等条件满足时，可以通过乘子对应某些解；一般情况下两种形式的路径和最优点不能无条件视为相同。

## 小批次只是在估计同一个经验梯度

对参数化模型 $h_\theta$，完整经验风险的梯度是

$$
\nabla_\theta\widehat R_S(\theta)
=\frac1n\sum_{i=1}^{n}
\nabla_\theta\ell\bigl(h_\theta(x_i),y_i\bigr).
$$

小批次 $B$ 使用

$$
\widehat R_B(\theta)
=\frac1{|B|}\sum_{i\in B}
\ell\bigl(h_\theta(x_i),y_i\bigr),
$$

并用它做更新：

$$
\theta_{k+1}
=\theta_k-\eta\nabla_\theta\widehat R_B(\theta_k).
$$

若 $B$ 是合适的均匀抽样，批次梯度的期望接近完整经验梯度；单个批次仍然有噪声。小批次节省内存并增加更新频率，却没有改变“完整训练目标是什么”这一问题。

例如 $\widehat R(\theta)=(\theta-3)^2$，从 $\theta_0=0$ 出发，学习率 $\eta=0.1$：

$$
\theta_1
=0-0.1\cdot2(0-3)
=0.6.
$$

一步更新只是向 ERM 目标移动了一段距离。没有收敛、步长过大或梯度估计偏差时，训练输出可以明显高于经验最优。

## 权重和损失改变 ERM 看到的任务

不同样本的错误代价可以写成加权经验风险：

$$
\widehat R_w(h)
=\frac1n\sum_{i=1}^{n}
w_i\ell\bigl(h(x_i),y_i\bigr),
\qquad
w_i\ge0.
$$

改变权重会改变训练目标，不会创造新的样本信息。类别不平衡时，少数类权重可以让错误更受重视；如果少数类覆盖的状态太窄，权重不能替代新的观测。

损失还决定 ERM 选择的统计量。回归 MSE 倾向条件均值，MAE 倾向条件中位数；概率预测的负对数似然要求模型给正确事件合理概率，而不是只输出一个类别。部署目标若是排序、风险估计或尾部控制，必须选择与该目标相称的损失和验证指标。

这解释了为什么“把训练损失降到最低”不是脱离语境的目标。先确定动作、错误代价和部署分布，再确定经验风险怎样近似它。

## 统计误差和优化误差要分开

令 $h_{\mathcal H}^\star$ 是假设空间内的总体风险最优规则，$\widetilde h$ 是实际算法输出。可以把总差距写成：

$$
\begin{aligned}
R(\widetilde h)-R^\star
={}&
\bigl[R(\widetilde h)-\widehat R_S(\widetilde h)\bigr]\\
&+\bigl[\widehat R_S(\widetilde h)
-\widehat R_S(h_{\mathcal H}^\star)\bigr]\\
&+\bigl[\widehat R_S(h_{\mathcal H}^\star)
-R(h_{\mathcal H}^\star)\bigr]\\
&+\bigl[R(h_{\mathcal H}^\star)-R^\star\bigr].
\end{aligned}
$$

四项分别对应实际输出的泛化偏差、相对经验最优的优化误差、空间内最优规则的样本估计波动，以及假设空间相对所有规则的表达误差。这个分解是帮助定位问题的账本，不是说每项都能被独立精确测量。

训练集损失很高，可能是优化没完成，也可能是假设空间表达不足；训练集损失很低而验证损失很高，可能是估计/泛化问题；训练和验证都低但部署失败，则要检查部署分布或损失错配。只看一条 loss 曲线无法区分这些原因。

## 失效模式

**把 ERM 当成总体风险最小化。** ERM 只优化样本平均损失。泛化还需要代表性数据、合适空间和独立验证。

**忽略多个经验最小点。** 同样的训练损失可能对应不同的未见预测。参数化、初始化、正则化和优化路径会影响最终选择。

**把正则化惩罚当成数据损失。** 惩罚项表达选择偏好，不是新的标签证据。它的强度应由训练外证据选择。

**把小批次噪声当成新目标。** 小批次是完整经验梯度的估计，不应因为某一步损失下降就断言总体风险改善。

**把最大似然当成模型正确的证明。** 似然只是在给定概率模型和样本下的 ERM。模型错设、分布偏移和相关样本仍会破坏解释。

**把权重当成信息增广。** 权重改变错误代价，不会补上少数类没有覆盖的输入区域。

**把优化失败归咎于泛化。** 学习率、梯度、停止时机和数值稳定性先要证明训练目标真的接近了；否则还没有进入统计比较阶段。

## 相关词条

- [假设空间](../learning-framework/hypothesis-spaces/)：说明 ERM 在哪些候选规则中选择。
- [监督学习](../learning-framework/supervised-learning/)：对比损失、条件 Bayes 规则和标签噪声。
- [优化问题](../optimization-theory/optimization-problems/)：理解目标函数、可行选择和最小化形式。
- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：分析如何近似求解 ERM 的连续参数目标。
- [随机梯度下降理论](../optimization-theory/stochastic-gradient-descent-theory/)：解释小批次梯度的方差和收敛边界。
- [最大似然](../probability/maximum-likelihood/)：展开负对数似然作为经验风险的特例。
- [最大后验估计](../probability/maximum-a-posteriori/)：比较似然目标与先验正则化的关系。
- [交叉熵](../information-theory/cross-entropy/)：讨论概率预测中常用的负对数损失。
- [学习是什么](../learning-framework/what-is-learning/)：回看表达、估计和优化误差的总分解。
