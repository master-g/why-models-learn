---
title: "逻辑回归：用概率模型拟合二分类"
tags: ["why-models-learn"]
---

逻辑回归用线性函数表示对数几率，再用 sigmoid 函数把它变成位于 0 和 1 之间的概率。它训练的是 Bernoulli 条件概率模型，目标通常是交叉熵或负对数似然；最后把概率变成类别，仍需要单独指定阈值。这个区分很重要：逻辑回归不是先做最小二乘再把输出截断，而是直接让“概率预测”和“观测标签”在同一个似然目标中对齐。

![逻辑回归把线性分数映射为概率再进行阈值决策](/assets/linear-models/svg/logistic-regression.1.svg)

## 概率输出与硬分类是两层

对二分类标签 $y\in\{0,1\}$，给定输入特征 $x\in\mathbb R^p$，先计算线性分数

$$
z=\beta_0+x^{\mathsf T}\beta.
$$

再通过 sigmoid 函数

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

得到正类概率

$$
p(x)=\Pr(Y=1\mid x)=\sigma(z).
$$

由于 $\sigma(z)\in(0,1)$，它可以被解释为概率，但模型并没有在这一步做硬分类。给定阈值 $\tau\in(0,1)$ 后，才定义决策

$$
\widehat y_\tau=
\begin{cases}
1,&p(x)\ge\tau,\\
0,&p(x)<\tau.
\end{cases}
$$

常见的 $\tau=1/2$ 只对应一个特定代价对称的决策规则，不是逻辑回归的数学定义。模型可以输出 $0.73$，而不同业务在误报和漏报代价不同的情况下，可能选择不同阈值。

| 层次 | 输入 | 输出 | 由什么决定 |
| --- | --- | --- | --- |
| 线性分数 | $x$ | $z$ | 特征和系数 |
| 概率模型 | $z$ | $p=\sigma(z)$ | sigmoid 与训练参数 |
| 分类决策 | $p$ | $0$ 或 $1$ | 阈值与错误代价 |
| 评估 | 标签、概率或类别 | 指标 | 评估协议与业务目标 |

把四层混成一层，会误以为改变分类阈值就等于重新训练模型，或误以为准确率足以评价概率是否可信。

## Sigmoid 只改变坐标，不改变排序

sigmoid 严格单调递增：

$$
\sigma'(z)=\sigma(z)\bigl(1-\sigma(z)\bigr)>0.
$$

因此两个样本的线性分数谁更大，概率排序就不会改变。它把整个实数轴压到 $(0,1)$：

$$
\lim_{z\to-\infty}\sigma(z)=0,
\qquad
\sigma(0)=\frac12,
\qquad
\lim_{z\to+\infty}\sigma(z)=1.
$$

导数在 $p=1/2$ 附近最大，最大值为 $1/4$；分数绝对值很大时，sigmoid 进入饱和区，概率已经接近 0 或 1，局部变化变小。实现时应使用稳定的 log-sigmoid 或 softplus 形式计算损失，避免直接计算极大或极小的指数。

## Bernoulli 似然给出交叉熵

给定 $n$ 个独立样本 $(x_i,y_i)$，令

$$
p_i=\sigma(\beta_0+x_i^{\mathsf T}\beta).
$$

单个标签的条件概率可以统一写成

$$
\Pr(Y_i=y_i\mid x_i)
=p_i^{y_i}(1-p_i)^{1-y_i}.
$$

全体样本的似然是

$$
\mathcal L(\beta_0,\beta)
=\prod_{i=1}^{n}p_i^{y_i}(1-p_i)^{1-y_i}.
$$

最大化乘积容易数值下溢，因此取对数并改为最小化负对数似然。平均二元交叉熵为

$$
\operatorname{BCE}
=-\frac1n\sum_{i=1}^{n}
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right].
$$

如果模型给一个正样本很小的概率，$\log p_i$ 会产生很大的负值；如果给一个负样本很大的概率，$\log(1-p_i)$ 也会产生同样的惩罚。这让概率预测比只看最终类别的 0–1 损失获得更细的训练信号。

把 $z$ 代回去，可以得到数值更稳定的单样本损失：

$$
\ell(z,y)
=\log\bigl(1+e^z\bigr)-yz
=\operatorname{softplus}(z)-yz.
$$

计算时应根据 $z$ 的正负选择稳定的 softplus 实现，而不是直接让 $e^z$ 在极端分数处溢出。

## 梯度等于预测误差乘特征

对单个样本，先对线性分数求导：

$$
\frac{\partial\ell}{\partial z}
=\sigma(z)-y
=p-y.
$$

因为 $z=\beta_0+x^{\mathsf T}\beta$，所以

$$
\nabla_{\beta_0,\beta}\ell
=(p-y)
\begin{pmatrix}
1\\x
\end{pmatrix}.
$$

把带截距特征写成 $\widetilde x_i=(1,x_i^{\mathsf T})^{\mathsf T}$，设计矩阵写成 $\widetilde X$，则全体样本的梯度是

$$
\nabla\operatorname{BCE}
=\frac1n\widetilde X^{\mathsf T}(p-y).
$$

这和神经网络最后一层常见的“预测减标签，再乘输入特征”结构相同。不同之处在于逻辑回归的分数对参数是线性的，交叉熵加 sigmoid 形成一个凸的二分类目标。

Hessian 为

$$
\nabla^2\operatorname{BCE}
=\frac1n\widetilde X^{\mathsf T}W\widetilde X,
\qquad
W=\operatorname{diag}\bigl(p_i(1-p_i)\bigr).
$$

因为 $0<p_i(1-p_i)\le1/4$，$W$ 是半正定的，所以 Hessian 半正定，未加正则化的逻辑回归损失是凸函数。凸性保证局部最小点也是全局最小点，但不保证有限解存在，也不保证特征矩阵满秩。

## 两个样本的一次梯度更新

取一维带截距数据：

$$
\widetilde X=
\begin{pmatrix}
1&0\\
1&1
\end{pmatrix},
\qquad
y=
\begin{pmatrix}
0\\1
\end{pmatrix},
\qquad
\theta=
\begin{pmatrix}
\beta_0\\\beta_1
\end{pmatrix}
=
\begin{pmatrix}
0\\0
\end{pmatrix}.
$$

初始两个分数都是 0，所以

$$
p=
\begin{pmatrix}
1/2\\1/2
\end{pmatrix},
\qquad
\widetilde X^{\mathsf T}(p-y)
=
\begin{pmatrix}
0\\-1/2
\end{pmatrix}.
$$

若先使用学习率 $\eta=1$，用未除以样本数的损失做一步梯度下降：

$$
\theta_{\mathrm{new}}
=\theta-\eta\widetilde X^{\mathsf T}(p-y)
=
\begin{pmatrix}
0\\1/2
\end{pmatrix}.
$$

更新后的概率是

$$
p_{\mathrm{new}}
=
\begin{pmatrix}
1/2\\
\sigma(1/2)
\end{pmatrix}
\approx
\begin{pmatrix}
0.5\\0.622459
\end{pmatrix}.
$$

初始平均交叉熵是 $\log2\approx0.693147$。更新后为

$$
\operatorname{BCE}_{\mathrm{new}}
=\frac12
\left[
\log2+\log\left(1+e^{-1/2}\right)
\right]
\approx0.583612.
$$

损失下降说明这一步沿着训练目标的下降方向移动，但它不能单独证明学习率合适、模型已收敛或未来样本表现会变好。

## Newton 和 IRLS 使用曲率

梯度下降只使用一阶信息。逻辑回归的 Hessian 具有

$$
H=\widetilde X^{\mathsf T}W\widetilde X
$$

的形式，因此 Newton 更新可以写成

$$
\theta_{\mathrm{new}}
=\theta-
\left(\widetilde X^{\mathsf T}W\widetilde X\right)^{-1}
\widetilde X^{\mathsf T}(p-y).
$$

在每一步把权重 $W$ 固定后，这也可以理解为解一个加权最小二乘问题，称为迭代重加权最小二乘。实际实现通常不显式求逆，而是解线性系统；如果 Hessian 病态或 Newton 步过大，还需要阻尼、线搜索或信赖域。

凸性使优化比一般神经网络容易分析，但以下情况仍会造成数值困难：

- 特征列共线使 Hessian 近似奇异；
- 样本可被某个超平面完全分开时，参数会不断增大；
- 极端类别不平衡让大部分梯度来自同一类别；
- 极大分数让 sigmoid 饱和，直接指数计算发生上溢或下溢。

## 对数几率让系数有可解释单位

由

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}
$$

可以得到几率与对数几率：

$$
\frac{p}{1-p}=e^z,
\qquad
\log\frac{p}{1-p}=z=\beta_0+x^{\mathsf T}\beta.
$$

因此逻辑回归的线性假设不是“概率随特征线性变化”，而是“概率的对数几率随特征线性变化”。在其他特征固定时，$x_j$ 增加一单位会让几率乘上

$$
e^{\beta_j}.
$$

例如 $\beta_j=\log2$ 表示几率乘 2；这不是概率增加固定的 2 倍，因为概率和几率之间是非线性转换。系数的这种解释仍然是条件相关解释，不能仅凭回归模型就推出因果效应。

## 决策边界是一个超平面

使用阈值 $\tau$ 时：

$$
\sigma(z)\ge\tau
\quad\Longleftrightarrow\quad
z\ge\log\frac{\tau}{1-\tau}.
$$

当 $\tau=1/2$ 时，边界是

$$
\beta_0+x^{\mathsf T}\beta=0.
$$

当 $\tau=0.8$ 时，线性分数必须至少达到

$$
\log\frac{0.8}{0.2}=\log4\approx1.386294.
$$

改变阈值会平移决策边界，但不会改变已经训练好的概率排序。若正类漏报代价高，可以降低阈值；若误报代价高，可以提高阈值。阈值选择必须在验证数据上按照部署代价确定，不能用测试集反复试到最满意。

## 逻辑回归与普通最小二乘不是一回事

对二元标签直接做 OLS 有三个结构问题：

1. 线性预测可以小于 0 或大于 1，不能直接解释为概率；
2. 平方损失对应条件均值，而分类决策还需要概率与代价的匹配；
3. Bernoulli 观测的方差随均值变化，等方差高斯噪声并不是自然模型。

逻辑回归使用 Bernoulli 似然，保证输出落在概率范围内，并让每个错误概率按对数损失受到惩罚。它也不是“比 OLS 更复杂就一定更好”：如果任务确实是连续数值预测，应该回到回归损失和回归模型。

## 完全分离会让无正则化解逃向无穷

考虑前面的两点数据。取一族参数

$$
\beta_0=-\frac t2,
\qquad
\beta_1=t,
\qquad
t>0.
$$

两个样本的分数分别是 $-t/2$ 和 $t/2$。当 $t$ 越来越大时，第一个样本的预测概率趋近 0，第二个样本的预测概率趋近 1，交叉熵趋近 0，但任何有限的 $t$ 都还不是零损失：

$$
\operatorname{BCE}(t)
=\log\left(1+e^{-t/2}\right)
\longrightarrow0.
$$

于是最大似然估计没有有限的唯一最优参数，优化器会继续把系数推大，虽然分类结果早已不再改变。L2 正则化会给参数增长增加代价，L1 也会改变最优性条件；它们不仅控制泛化，也为分离数据提供有限解的偏好。

分离不是“优化器太差”的证据。应先检查标签、时间切分和特征泄漏，再决定是否使用正则化、改变模型或采用带先验的概率模型。

## 类别不平衡、阈值和校准要分开

类别不平衡时，全部预测为多数类可能得到很高准确率，却完全没有识别少数类的能力。可以在训练目标中使用类别权重：

$$
\operatorname{BCE}_{w}
=-\frac1n\sum_i w_{y_i}
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right].
$$

权重改变训练时哪类错误更重要；阈值改变部署时如何把概率转成行动；校准改变的是概率数值是否与长期频率一致。这三件事不能互相替代。

| 关注点 | 典型问题 | 常用证据 |
| --- | --- | --- |
| 分类错误 | 漏报和误报哪个更贵 | 混淆矩阵、召回率、特异度、精确率 |
| 排序能力 | 正例是否排在负例前面 | ROC AUC、PR AUC、分位数召回 |
| 概率质量 | 预测 0.8 的样本是否约 80% 为正 | 可靠性图、校准曲线、Brier 分数 |
| 训练目标 | 是否按部署代价加权 | 加权交叉熵、代价矩阵 |
| 阈值策略 | 哪个分数触发行动 | 验证集上的代价曲线 |

准确率、F1、AUC 和交叉熵回答的是不同问题。报告指标时必须同时说明概率还是硬标签、阈值是多少、正类定义是什么，以及评估切分是否模拟了部署时点。

## 正则化与训练流程

带 L2 正则化的逻辑回归目标为

$$
J(\theta)
=\operatorname{BCE}(\theta)
+\frac{\lambda}{2}\lVert\beta\rVert_2^2,
$$

这里不惩罚截距。梯度和 Hessian 变为

$$
\nabla_\beta J
=\nabla_\beta\operatorname{BCE}+\lambda\beta,
\qquad
\nabla^2_\beta J
=\nabla^2_\beta\operatorname{BCE}+\lambda I.
$$

$\lambda>0$ 可以改善病态和分离问题，但也会引入偏差。L1 正则化可以产生稀疏系数，和 [岭回归与 Lasso](../linear-models/ridge-and-lasso/) 的回归情形共享相同的“惩罚改变解偏好”思想；逻辑损失下通常没有最小二乘那样简单的软阈值闭式解，需要坐标下降、近端方法或其他凸优化算法。

一个可复用的训练流程是：

1. 先定义正类、预测时点、误报和漏报的代价；
2. 按时间、实体或分层规则切分训练、验证和测试数据；
3. 只用训练集拟合缺失处理、标准化和类别权重；
4. 在训练集优化交叉熵，记录收敛、梯度和正则化强度；
5. 用验证集选择 $\lambda$、阈值和是否校准；
6. 冻结全部选择后，在独立测试集同时报告概率、排序和阈值指标；
7. 在时间外、分布外和少数类子群上检查性能与校准是否保持。

如果最终用途是概率决策，应把校准和阈值策略作为模型的一部分保存，而不是只保存一个默认的 $0.5$。

## 失效模式

**把 sigmoid 输出直接当成可靠概率。** 输出位于 0 和 1 之间不等于已经校准；需要独立验证集检查概率与频率的对应关系。

**把阈值 0.5 当成自然法则。** 它只在代价和先验满足特定对称条件时合理，实际阈值应由部署损失确定。

**用 OLS 代替 Bernoulli 似然却继续解释为概率。** OLS 可能产生越界输出，也没有使用二元标签的自然方差结构。

**看到分离就盲目增加训练轮数。** 无正则化参数发散可能正是似然没有有限最优解；先查泄漏和标签，再选择正则化或带先验方法。

**只看准确率。** 类别不平衡、阈值变化和概率失准都会让单一准确率掩盖真正风险。

**把高 AUC 当成高质量概率。** AUC 只评价排序，概率校准要单独用可靠性图、Brier 分数或对数损失检查。

**把系数的几率比当成因果效应。** $e^{\beta_j}$ 是条件模型中的几率乘数，不会自动消除混杂、反向因果和选择偏差。

**在测试集上选择正则化、阈值和校准器。** 这会把最终评估反馈到模型流程，测试分数失去独立性。

## 相关词条

- [监督学习](../learning-framework/supervised-learning/)：定义分类标签、条件风险和训练外评估。
- [岭回归与 Lasso](../linear-models/ridge-and-lasso/)：比较 L1/L2 惩罚、收缩、稀疏和验证选择。
- [Softmax 回归](../linear-models/softmax-regression/)：把二分类概率模型推广到多个类别。
- [线性回归](../linear-models/linear-regression/)：对照连续标签下的最小二乘和条件均值。
- [训练、验证与测试集](../learning-framework/train-validation-test/)：规定阈值、正则化和校准的选择边界。
- [分类指标](../evaluation-and-generalization/classification-metrics/)：系统展开混淆矩阵、排序、概率与阈值指标。
- [最大似然估计](../probability/maximum-likelihood/)：从似然和对数似然的统一视角理解参数拟合。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：连接显式惩罚、分离和泛化风险。
