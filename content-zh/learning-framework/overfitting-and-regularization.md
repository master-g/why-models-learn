---
title: "过拟合与正则化：训练误差为什么不能单独当证据"
tags: ["why-models-learn"]
---

过拟合是模型把训练样本里的偶然细节也当成规律，因而训练误差很低、未见数据上的风险却变高；正则化则是在拟合样本之外加入对复杂度、参数规模、训练时间或不变性的偏好。它们讨论的不是“模型够不够大”这一条规则，而是有限数据、候选空间和选择过程怎样共同决定泛化。本文先用同训练误差而不同外推的插值例子说明过拟合，再给出复杂度项的概率直觉，推导显式正则化，最后说明早停、优化器和参数化怎样产生隐式正则化，以及验证选择为什么也会被过拟合。

![训练误差下降而验证误差先降后升，正则化把选择点留在泛化较好的区域](/assets/learning-framework/svg/overfitting-and-regularization.1.svg)

## 训练误差和总体风险回答不同问题

给定训练样本 $S=\{(x_i,y_i)\}_{i=1}^{n}$ 和损失函数 $\ell$，经验风险是

$$
\widehat R_S(h)
=\frac1n\sum_{i=1}^{n}
\ell\bigl(h(x_i),y_i\bigr).
$$

它只检查已经看过的样本。若部署时的输入输出服从分布 $P$，真正关心的总体风险是

$$
R_P(h)
=\mathbb E_{(X,Y)\sim P}
\left[\ell\bigl(h(X),Y\bigr)\right].
$$

泛化间隙可以写成

$$
G_S(h)
=R_P(h)-\widehat R_S(h).
$$

训练过程通常能直接降低 $\widehat R_S$，却不能直接观察 $R_P$。这就产生了四个容易混淆的量：

| 量 | 它使用的证据 | 它适合回答什么 | 主要风险 |
| --- | --- | --- | --- |
| 训练风险 | 拟合过的样本 | 参数是否在适应训练目标 | 记住噪声也能很低 |
| 验证风险 | 选择阶段保留的样本 | 哪个模型、超参数或停止时机较好 | 比较次数太多会过拟合验证集 |
| 测试风险 | 方案冻结后的新样本 | 最终评估协议下的表现 | 反复查看后就不再独立 |
| 总体风险 | 真实部署分布 | 长期未见样本上的期望表现 | 通常无法直接计算 |

因此，“训练误差为零”只说明训练样本被完全拟合。它既没有证明目标规律被学到，也没有证明部署分布和训练分布一致。[训练、验证与测试集](../learning-framework/train-validation-test/) 负责规定这些证据如何分工。

## 过拟合来自对样本偶然性的适应

看一组已经知道标签的点：

$$
(x_i,y_i)\in
\{(-1,1),(0,0),(1,1),(2,4)\}.
$$

一个简单规则是

$$
h_{\mathrm{simple}}(x)=x^2.
$$

另一个更高阶的规则在训练点上加入一个恰好为零的乘积：

$$
h_{\mathrm{high}}(x)
=x^2+(x+1)x(x-1)(x-2).
$$

因为乘积项在 $x=-1,0,1,2$ 都为零，所以两条规则在训练集上完全一致：

$$
\widehat R_S(h_{\mathrm{simple}})
=\widehat R_S(h_{\mathrm{high}})
=0
$$

但如果真实规律是 $y=x^2$，在未见点 $x=3$ 处

$$
\begin{aligned}
h_{\mathrm{simple}}(3)&=9,\\
h_{\mathrm{high}}(3)&=9+(4)(3)(2)(1)=33.
\end{aligned}
$$

此时简单规则的平方误差是 $0$，高阶规则的平方误差是

$$
(33-9)^2=576.
$$

这个例子不是说所有高阶模型都会失败，而是说明训练样本没有唯一决定外推规则。高阶规则多使用了一个训练集之外没有证据支持的自由度；如果噪声恰好改变几个训练标签，它也可以通过增加局部弯折来把噪声一起插值进去。[假设空间](../learning-framework/hypothesis-spaces/) 讨论的候选规则集合，正是这里“还有多少种同样符合训练数据的规则”的正式表达。

## 复杂度项为什么会进入泛化判断

先看一个有限假设空间的简化结论。若损失是 0–1 损失，$\mathcal H$ 是在数据之前固定的有限集合，则对任意 $\varepsilon>0$，Hoeffding 不等式和并合界给出

$$
\Pr\left(
\exists h\in\mathcal H:
\left\lvert R_P(h)-\widehat R_S(h)\right\rvert>\varepsilon
\right)
\le
2\lvert\mathcal H\rvert
\exp(-2n\varepsilon^2).
$$

把右侧控制在 $\delta$ 以内，可以得到一个直观的复杂度项：

$$
\left\lvert R_P(h)-\widehat R_S(h)\right\rvert
\lesssim
\sqrt{
\frac{\log\bigl(2\lvert\mathcal H\rvert/\delta\bigr)}
{2n}
}.
$$

固定样本数 $n$ 和失败概率 $\delta$ 时，候选集合越大，统一控制所有候选的代价越大。以 $n=1000$、$\delta=0.05$ 为例：

| 候选空间规模 | $\log(2\lvert\mathcal H\rvert/\delta)$ | 复杂度项上界 |
| --- | ---: | ---: |
| $\lvert\mathcal H\rvert=10$ | 5.9915 | 约 0.0547 |
| $\lvert\mathcal H\rvert=10^6$ | 17.5044 | 约 0.0936 |

这个界是方向性的教学模型，不是深度网络的完整泛化理论。连续参数空间、算法选择和数据依赖需要用 VC 维、增长函数、Rademacher 复杂度或算法稳定性等更细的工具。但它揭示了同一件事：训练误差下降时，候选空间也可能在变大，模型因此有更多机会利用样本噪声。模型容量、双下降和过参数化会在 [模型容量](../learning-framework/model-capacity/) 与 [双下降](../learning-framework/double-descent/) 中单独展开。

## 显式正则化改变了训练目标

最直接的正则化是在经验风险之外加入复杂度惩罚：

$$
J_\lambda(h)
=\widehat R_S(h)
+\lambda\Omega(h),
\qquad
\lambda\ge0.
$$

等价的约束视角是

$$
\min_{h\in\mathcal H}\widehat R_S(h)
\quad
\text{subject to}
\quad
\Omega(h)\le c.
$$

惩罚形式用一个连续的 $\lambda$ 表示在拟合数据和偏好简单之间如何折中，约束形式直接规定复杂度上限。在合适的凸性和正则条件下，两种形式可以通过拉格朗日乘子联系；在非凸问题中，不应把每个 $\lambda$ 和某个 $c$ 机械地视为一一对应。

最小二乘里的 L2 惩罚可以用一个标量完全算清。假设数据项希望参数接近 $3$：

$$
J_\lambda(\theta)
=(\theta-3)^2+\lambda\theta^2.
$$

求导并令其为零：

$$
2(\theta-3)+2\lambda\theta=0,
\qquad
\theta_\lambda=\frac{3}{1+\lambda}.
$$

不同 $\lambda$ 的结果是：

| $\lambda$ | $\theta_\lambda$ | 数据拟合项 $(\theta-3)^2$ | 惩罚项 $\lambda\theta^2$ | 总目标 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 0 | 0 | 0 |
| 1 | 1.5 | 2.25 | 2.25 | 4.50 |
| 3 | 0.75 | 5.0625 | 1.6875 | 6.75 |

正则化没有让训练目标更小；它故意接受一部分训练误差，换取更小的参数和可能更稳定的未见表现。L2 惩罚 $\lVert\theta\rVert_2^2$ 连续地收缩所有坐标，L1 惩罚 $\lVert\theta\rVert_1$ 在零点有折角，可能把部分坐标直接压到零。对于

$$
J(\theta)=(\theta-a)^2+\lambda\lvert\theta\rvert,
$$

它的最小点可以写成软阈值：

$$
\theta^\star
=\operatorname{sign}(a)
\max\left(\lvert a\rvert-\frac{\lambda}{2},0\right).
$$

例如 $a=3$、$\lambda=4$ 时 $\theta^\star=1$；当 $\lambda\ge6$ 时，最优点被压到 $\theta^\star=0$。[经验风险最小化](../learning-framework/empirical-risk-minimization/) 讲的是样本损失怎样选择规则；正则化是在这个选择问题中明确加入第二个偏好。

## 训练曲线显示了选择点

正则化不只意味着在目标函数里写一个范数。早停把“优化到第几步”也当作选择变量。假设一次训练记录如下：

| 训练轮次 | 训练风险 | 验证风险 | 解释 |
| ---: | ---: | ---: | --- |
| 0 | 0.90 | 0.95 | 尚未学到主要结构 |
| 1 | 0.55 | 0.60 | 两者一起下降 |
| 2 | 0.25 | 0.31 | 仍在获得可迁移规律 |
| 3 | 0.10 | 0.24 | 验证风险最低 |
| 4 | 0.04 | 0.29 | 开始适应训练细节 |
| 5 | 0.01 | 0.37 | 训练误差更低但泛化更差 |

如果只看训练风险，会选择第 5 轮；如果验证集未被其他选择污染，应选择

$$
t^\star
\in
\operatorname*{arg\,min}_{t}
\widehat R_{\mathrm{val}}\bigl(h_t\bigr).
$$

第 3 轮的模型不是“训练得不够好”，而是在当前证据下训练目标和泛化目标的折中更好。早停的强度由优化步数、学习率和初始化共同决定，换一个优化器或学习率，完全相同的轮数不再表示相同的正则化强度。

## 优化过程也会产生隐式正则化

即使目标里没有显式的 $\lambda\Omega$，算法仍可能偏好某类解。先看一个只有一个参数的梯度下降：

$$
J(\theta)=\frac12(\theta-3)^2,
\qquad
\theta_{t+1}
=\theta_t-\eta(\theta_t-3).
$$

从 $\theta_0=0$ 出发且 $0<\eta<1$ 时，递推式解为

$$
\theta_t
=3\left[1-(1-\eta)^t\right].
$$

若 $\eta=0.5$，前几步是

| $t$ | 0 | 1 | 2 | 3 |
| ---: | ---: | ---: | ---: | ---: |
| $\theta_t$ | 0 | 1.5 | 2.25 | 2.625 |

早停让参数暂时保持在较小的幅度。这个标量例子没有多个全局最小点，但它说明优化时间本身可以控制解离初始化有多远。

在欠定线性回归中，隐式选择更明显。考虑

$$
J(\theta)
=\frac12\lVert X\theta-y\rVert_2^2,
\qquad
\theta_{t+1}
=\theta_t-\eta X^{\mathsf T}(X\theta_t-y).
$$

若 $X\theta=y$ 有多个解，从零初始化开始的全批次梯度下降只沿着 $X^{\mathsf T}$ 的列空间移动；在合适步长和可收敛条件下，它会收敛到最小 L2 范数的插值解：

$$
\theta_\mathrm{GD}
=X^{+}y
=\operatorname*{arg\,min}_{X\theta=y}
\lVert\theta\rVert_2.
$$

这不是数据额外告诉模型的事实，而是“零初始化 + 梯度方向 + 欧氏参数化”共同施加的偏好。改变初始化、参数化、优化器、批次噪声或网络结构，可能改变隐式偏置。[隐式正则化](../evaluation-and-generalization/implicit-regularization/) 会进一步讨论这种算法选择与显式惩罚并不等价的情形。

## 不同正则化手段偏好不同的解

| 手段 | 主要改变什么 | 常见偏好 | 需要警惕什么 |
| --- | --- | --- | --- |
| L2 惩罚或 weight decay | 参数目标 | 小而分散的权重 | 惩罚强度过大导致欠拟合 |
| L1 惩罚 | 参数目标的几何形状 | 稀疏权重 | 相关特征之间选择不稳定 |
| 早停 | 优化时间 | 离初始化较近的解 | 验证集选择次数会累积 |
| Dropout 或参数噪声 | 每次更新看到的模型 | 对局部扰动更稳定的规则 | 推理和训练的缩放约定必须一致 |
| 数据增强 | 训练样本与不变性 | 对指定变换保持一致 | 错误增强会改掉任务标签 |
| 权重共享或较小架构 | 假设空间 | 结构上更受限的函数 | 约束可能排除真正规律 |

“正则化”真正的共同点是缩小了可被训练证据支持的选择范围，或让模型偏向对扰动不敏感的规则；它不要求所有方法都写成一个范数。数据增强若把同一对象的旋转、裁剪视为同一标签，就是在加入不变性假设；权重共享则把许多位置的预测绑定到同一组参数。

## 正则化强度也会被验证集过拟合

正则化超参数不是凭空确定的。通常先在训练集上得到一族候选

$$
\widehat h_\lambda
=\mathcal A_\lambda(D_{\mathrm{train}}),
\qquad
\widehat\lambda
\in
\operatorname*{arg\,min}_{\lambda\in\Lambda}
\widehat R_{\mathrm{val}}
\bigl(\widehat h_\lambda\bigr).
$$

然后只在方案冻结后报告测试风险：

$$
\widehat R_{\mathrm{test}}
\bigl(\widehat h_{\widehat\lambda}\bigr).
$$

如果只试了两个 $\lambda$，验证风险的偶然波动已经会影响选择；如果试了几百个 $\lambda$、数据增强、随机种子和训练轮数，验证集也会成为被反复适应的样本。此时“选出的 λ 在验证集上最好”不是独立证据。可以扩大验证集、减少探索次数、使用嵌套交叉验证，或在方案冻结后保留一份新鲜测试集。[训练、验证与测试集](../learning-framework/train-validation-test/) 中的“验证集会被消耗”同样适用于正则化强度。

## 失效模式

**把训练误差最低当成目标。** 训练风险还在下降，不代表总体风险也在下降。应同时画训练与验证曲线，并把测试集留到最后。

**把大惩罚当成普遍安全。** $\lambda$ 太大时，模型会把真实结构也视为复杂度，出现系统性偏差。正则化是在偏差和方差之间移动，不是单调越大越好。

**只正则化参数，不检查数据边界。** 全量标准化、重复实体跨分片和未来字段泄漏，不能靠 L1 或 L2 惩罚修复。预处理和切分规则必须先遵守 [训练、验证与测试集](../learning-framework/train-validation-test/) 的边界。

**把早停轮数当成固定超参数。** 学习率、批次大小、初始化和优化器改变后，同一个轮数对应不同的有效正则化强度。

**把隐式正则化当成定理。** 某个优化器在某种参数化下偏好小范数解，不等于换成另一种参数化后仍然偏好同一个函数。必须说明数据、初始化、步长和收敛条件。

**把验证集试验当成免费诊断。** 选择模型、损失、阈值、增强、随机种子和正则化强度都在消耗验证证据。探索性结果和最终确认应分开记录。

**把指标错配当成过拟合。** 如果训练目标是平均平方误差，而部署代价主要来自少数类漏报，降低训练损失并不保证解决真正的问题；这首先是目标定义问题，其次才是复杂度问题。

## 相关词条

- [训练、验证与测试集](../learning-framework/train-validation-test/)：规定训练、验证和测试证据的职责边界。
- [经验风险最小化](../learning-framework/empirical-risk-minimization/)：说明正则化如何改变样本上的选择目标。
- [假设空间](../learning-framework/hypothesis-spaces/)：解释同一训练数据为何能留下多个候选规则。
- [过拟合与欠拟合](../learning-framework/overfitting-and-underfitting/)：比较模型太灵活与太受限的表现。
- [偏差—方差权衡](../learning-framework/bias-variance-tradeoff/)：分解正则化带来的系统误差和估计波动。
- [模型容量](../learning-framework/model-capacity/)：刻画候选空间能表达多少函数。
- [隐式正则化](../evaluation-and-generalization/implicit-regularization/)：深入分析优化器和参数化选择的解。
- [双下降](../learning-framework/double-descent/)：讨论容量继续增加后风险曲线为何可能再次下降。
- [损失景观](../optimization-theory/loss-landscapes/)：观察参数空间中的谷、平坦方向和障碍。
- [岭回归与 Lasso](../linear-models/ridge-and-lasso/)：把 L2 和 L1 正则化放进线性模型。
