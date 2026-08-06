---
title: "损失函数：把预测误差写成可优化的目标"
tags: ["why-models-learn"]
---

损失函数把一次预测和真实结果之间的差异写成一个可以计算、比较和求导的数。它不仅回答“这次错了多少”，还通过误差的形状规定了模型更愿意修正哪一类错误：平方损失偏爱条件均值并放大大错，绝对损失偏爱条件中位数，交叉熵则要求模型为真实结果分配概率。训练时优化的是经过样本归约的损失目标，部署时报告的却可能是准确率、MAE、覆盖率或业务代价；这三者不能混为一谈。

![损失函数把残差和分类间隔变成不同的优化地形：大残差可被平方放大，也可被 Huber 截平；分类间隔超过一以后 hinge 损失归零，而 logistic 损失仍然奖励更有把握的预测](/assets/training-nn/svg/loss-functions.1.svg)

## 损失函数先回答什么是“错”

设输入为 $\boldsymbol x$，真实结果为 $y$，模型参数为 $\boldsymbol\theta$，预测为

$$
\widehat y=f_{\boldsymbol\theta}(\boldsymbol x).
$$

逐样本损失是

$$
\ell(\widehat y,y)\geq0.
$$

它把预测空间中的一个点和真实结果之间的关系映射为一个非负数。模型在数据集

$$
\mathcal D=\{(\boldsymbol x_i,y_i)\}_{i=1}^{n}
$$

上的经验目标通常写成

$$
J(\boldsymbol\theta)
=\frac1n\sum_{i=1}^{n}
\ell\bigl(f_{\boldsymbol\theta}(\boldsymbol x_i),y_i\bigr).
$$

这个定义看起来只是把误差加起来，但真正的建模选择都藏在 $\ell$ 的形状里。若两个预测的绝对误差分别是 $1$ 和 $3$，MAE 把它们看成 $1$ 和 $3$，平方损失则把它们看成 $1$ 和 $9$。前一种目标更抗单个异常点，后一种目标更愿意为大错投入梯度。

损失函数还必须和输出的含义匹配。连续数值的预测可以用一个实数表示；二分类的预测可以是类别名、正类概率或带符号的分数；多分类的输出通常是一个和为 $1$ 的概率向量或未归一化的 logits。对概率向量使用平方误差并非绝对错误，但它和负对数似然对置信度的惩罚不同；对离散类别直接使用“预测对不对”又无法为大多数分类器提供平滑梯度。

## 损失、目标、指标和代价不是同一个对象

下面四个名字经常出现在同一个训练脚本里，却扮演不同角色：

| 对象 | 作用 | 是否一定用于反向传播 |
| --- | --- | --- |
| 逐样本损失 | 衡量一个输出和真实结果的差异 | 不一定，先要归约 |
| 训练目标 | 对损失、权重、掩码和正则项做组合 | 通常是 |
| 评估指标 | 在冻结模型上汇总表现，便于比较和解释 | 通常不是 |
| 部署代价 | 把错误映射为人工、金钱、风险或延迟 | 通常不是 |

例如，训练可以最小化 token 级交叉熵，验证时报告 perplexity，产品上线后却按拒绝一个真实用户的成本来决定阈值。交叉熵、perplexity 和业务代价之间有联系，但它们的数值不能互相替代。若只优化一个容易计算的损失，却从不检查真正关心的指标，模型可能在训练曲线上进步而在产品目标上退化。

正则化项也不等同于逐样本损失。一个常见目标是

$$
J(\boldsymbol\theta)
=\frac1n\sum_{i=1}^{n}\ell_i
+\lambda R(\boldsymbol\theta),
$$

其中第一项由数据误差产生，$R$ 惩罚参数或函数的复杂度。权重衰减、稀疏惩罚和数据损失的单位、尺度以及梯度来源都不同；调节 $\lambda$ 时不能只看总目标而不拆开两项。

## 归约决定一次更新看见多大的损失

一批数据得到逐样本损失

$$
\boldsymbol\ell=(\ell_1,\ldots,\ell_n).
$$

最常见的三种归约是

$$
\begin{aligned}
L_{\mathrm{sum}}
&=\sum_{i=1}^{n}\ell_i,\\
L_{\mathrm{mean}}
&=\frac1n\sum_{i=1}^{n}\ell_i,\\
L_{\mathrm{weighted}}
&=\frac{\sum_{i=1}^{n}w_i\ell_i}
{\sum_{i=1}^{n}w_i}.
\end{aligned}
$$

sum 保留了批大小的信息：批量扩大一倍，目标和梯度通常也扩大一倍。mean 让相同数据分布下的梯度尺度大致不随批大小改变。加权平均则让不同样本、类别或时间位置拥有不同的重要性，但分母究竟是样本数、权重和还是有效 token 数，必须写进协议。

掩码是加权归约的一个特例。对变长序列，令 $m_i\in\{0,1\}$ 表示位置是否有效，则

$$
L_{\mathrm{mask}}
=\frac{\sum_i m_i\ell_i}{\sum_i m_i}.
$$

若错误地除以整个张量的长度，padding 比例变化就会改变有效 token 的梯度尺度。若把分母写成固定的 batch 大小，不同长度样本之间还会产生额外的样本权重。调试时应同时打印逐位置损失、掩码和归约分母，而不是只打印一个标量。

归约还会影响梯度：

$$
\frac{\partial L_{\mathrm{sum}}}{\partial\boldsymbol\theta}
=\sum_i\frac{\partial\ell_i}{\partial\boldsymbol\theta},
\qquad
\frac{\partial L_{\mathrm{mean}}}{\partial\boldsymbol\theta}
=\frac1n\sum_i\frac{\partial\ell_i}{\partial\boldsymbol\theta}.
$$

因此改动 sum 和 mean 不是纯粹的日志格式改动。若学习率、梯度裁剪阈值、正则项和混合精度缩放没有一起重新检查，训练动态会改变，即使每个样本的损失公式完全没变。

## 条件风险：损失函数决定模型想预测什么

损失的统计含义可以从固定输入 $\boldsymbol x$ 开始看。假设模型在这个输入上输出一个决策 $a$，它面对的不是一个确定标签，而是条件分布 $Y\mid X=\boldsymbol x$。条件风险为

$$
\mathcal R(a\mid\boldsymbol x)
=\mathbb E\bigl[\ell(a,Y)\mid X=\boldsymbol x\bigr].
$$

理想的预测是

$$
a^\star(\boldsymbol x)
=\operatorname*{arg\,min}_{a}
\mathcal R(a\mid\boldsymbol x).
$$

这个 $a^\star$ 叫作该损失下的 Bayes 决策。它说明了“训练得足够好”究竟意味着什么：不是所有损失都会把模型推向同一个真实量。

### 0–1 损失选择条件众数

分类标签 $y$ 属于有限类别集合时，0–1 损失为

$$
\ell_{0\text{-}1}(a,y)
=
\begin{cases}
0,&a=y,\\
1,&a\neq y.
\end{cases}
$$

固定 $X=\boldsymbol x$ 后，预测类别 $a$ 的风险是

$$
\mathcal R(a\mid\boldsymbol x)
=1-\mathbb P(Y=a\mid X=\boldsymbol x).
$$

所以最优类别是条件概率最大的类别，也就是条件众数：

$$
a^\star(\boldsymbol x)
=\operatorname*{arg\,max}_{c}
\mathbb P(Y=c\mid X=\boldsymbol x).
$$

0–1 损失很符合最终准确率，却在类别边界处不连续，也不能告诉一个错误预测“错得有多自信”。例如把真实类别概率从 $0.49$ 预测成 $0.01$，和预测成 $0.40$ 都是一次错误；对准确率而言二者没有区别。

### 平方损失选择条件均值

令预测是实数 $a$，平方损失为

$$
\ell_2(a,y)=(a-y)^2.
$$

记条件均值为

$$
\mu(\boldsymbol x)
=\mathbb E[Y\mid X=\boldsymbol x].
$$

把 $a-y$ 写成 $(a-\mu)+(\mu-y)$，并取条件期望：

$$
\begin{aligned}
\mathbb E[(a-Y)^2\mid X=\boldsymbol x]
&=\mathbb E[
((a-\mu)+(\mu-Y))^2
\mid X=\boldsymbol x]\\
&=(a-\mu)^2
+2(a-\mu)\mathbb E[\mu-Y\mid X=\boldsymbol x]\\
&\quad+\mathbb E[(\mu-Y)^2\mid X=\boldsymbol x]\\
&=(a-\mu)^2+\operatorname{Var}(Y\mid X=\boldsymbol x).
\end{aligned}
$$

最后一项与 $a$ 无关，第一项在 $a=\mu$ 时达到最小。因此平方损失不是抽象地“惩罚大误差”，它还明确规定了点预测要逼近条件均值。当条件分布有两个相距很远的峰时，均值可能落在两个峰之间、却并不代表一个常见样本。

### 绝对损失选择条件中位数

绝对损失为

$$
\ell_1(a,y)=\lvert a-y\rvert.
$$

在条件分布没有刚好落在 $a$ 上的质量时，对 $a$ 求导得到

$$
\frac{\mathrm d}{\mathrm da}
\mathbb E[\lvert a-Y\rvert\mid X=\boldsymbol x]
=\mathbb P(Y<a\mid\boldsymbol x)
-\mathbb P(Y>a\mid\boldsymbol x).
$$

导数从负变正的位置满足两侧概率各至少为 $1/2$，因此是条件中位数。离散分布或有重复值时，绝对损失的最小点可能是一个区间；这就是绝对值在零点使用次梯度的统计版本。

均值和中位数的区别可以用一个极端样本看出：结果为 $0,0,0,10$ 时，均值是 $2.5$，中位数是 $0$。如果十这个值是可靠但罕见的结果，平方损失会把预测拉向 $2.5$；如果它更像偶发异常，绝对损失会更接近主体。

## 回归损失：残差形状就是错误代价

下面统一用预测减真实值的残差

$$
r_i=\widehat y_i-y_i.
$$

对单个预测 $\widehat y$ 求导时，残差对预测的导数为 $1$，所以可以直接读出损失曲线的斜率。

### 平方损失在远处越来越陡

常用的半平方损失为

$$
\ell_{\mathrm{sq}}(r)=\frac12r^2,
\qquad
\frac{\mathrm d\ell_{\mathrm{sq}}}{\mathrm dr}=r.
$$

使用 $\frac12$ 只是让导数少一个常数 $2$；如果写 MSE，则通常使用 $r^2$ 再做批平均。平方损失的梯度随残差线性增长：一个离群点不仅损失大，还会用更大的梯度拉动参数。这个性质在噪声近似高斯、希望估计条件均值时很有用，在标签污染或长尾目标上却可能让单个样本主导更新。

### 绝对损失在远处保持固定斜率

绝对损失为

$$
\ell_{\mathrm{abs}}(r)=\lvert r\rvert.
$$

当 $r\neq0$ 时

$$
\frac{\mathrm d\ell_{\mathrm{abs}}}{\mathrm dr}
=\operatorname{sign}(r).
$$

它不会因为错误变大而继续增加斜率，所以对异常点更稳健；代价是在 $r=0$ 处不可微，优化器需要选择一个次梯度，且靠近零时曲面没有平方损失那么平滑。不可微不等于不能优化，但需要理解实现如何处理折点。

### Huber 在近处平方、远处线性

给定阈值 $\delta>0$，Huber 损失定义为

$$
\ell_\delta(r)
=
\begin{cases}
\frac12r^2,&\lvert r\rvert\leq\delta,\\
\delta\left(\lvert r\rvert-\frac12\delta\right),&\lvert r\rvert>\delta.
\end{cases}
$$

它的导数为

$$
\frac{\mathrm d\ell_\delta}{\mathrm dr}
=
\begin{cases}
r,&\lvert r\rvert\leq\delta,\\
\delta\operatorname{sign}(r),&\lvert r\rvert>\delta.
\end{cases}
$$

在 $r=\delta$ 两侧，函数值都是 $\frac12\delta^2$，导数都是 $\delta$；在 $r=-\delta$ 处同理。因此 Huber 损失在一阶上是连续的，却把远处梯度截到 $\pm\delta$。它不是“自动删除异常点”：异常样本仍然贡献线性损失，只是不再以平方速度增加影响。

选择 $\delta$ 等于选择一个残差尺度。若目标单位改变而 $\delta$ 不变，平方区和线性区的位置就变了；常见做法是先按训练数据的尺度标准化残差，再在验证协议中选择阈值。不能看完整测试集的残差分布后再回头设置 $\delta$。

### 分位数损失允许错误代价不对称

若低估和高估的代价不一样，条件均值和条件中位数都未必是目标。定义

$$
\rho_\tau(r)
=\begin{cases}
(1-\tau)r,&r\geq0,\\
-\tau r,&r<0,
\end{cases}
\qquad 0<\tau<1.
$$

当 $r=\widehat y-y>0$ 时，模型高估，斜率是 $1-\tau$；当 $r<0$ 时，模型低估，斜率是 $-\tau$。$\tau=0.9$ 会让高估的正侧斜率为 $0.1$，低估的负侧斜率为 $-0.9$，因此最优点倾向于条件分布的 90% 分位数。后续的 [回归损失](../training-nn/regression-losses/) 会单独讨论 Huber、分位数和预测区间。

## 概率损失：错误的不是数值距离，而是概率承诺

当模型输出概率时，损失需要评价“真实结果发生时，模型给它分了多少概率”。对于离散标签分布 $\boldsymbol y$ 和模型概率 $\boldsymbol q$，交叉熵或负对数似然为

$$
\ell_{\mathrm{NLL}}(\boldsymbol q,\boldsymbol y)
=-\sum_{k=1}^{K}y_k\log q_k.
$$

one-hot 标签的真实类别为 $c$ 时，它退化为

$$
\ell_{\mathrm{NLL}}=-\log q_c.
$$

真实类别概率为 $0.9$ 时损失约为 $0.105$，为 $0.01$ 时损失约为 $4.605$。后一个错误不只是在分类上错了一次，还表明模型对错误答案极有把握，所以受到很大惩罚。$q_c=0$ 时负对数似然发散；训练实现应从 logits 使用稳定公式，不能先把概率截断后再把截断值误当作原始目标。

### 二分类和多分类的 logits 形式

二分类可以输出一个 logit $z$，正类概率是

$$
q=\sigma(z)=\frac1{1+\exp(-z)}.
$$

对标签 $y\in\{0,1\}$，二元交叉熵是

$$
\ell_{\mathrm{BCE}}(z,y)
=-y\log\sigma(z)-(1-y)\log(1-\sigma(z)).
$$

直接先算 $\sigma(z)$ 再取对数，在 $z$ 很大或很小时可能发生下溢、上溢或 $\log(0)$。等价的稳定表达式是

$$
\ell_{\mathrm{BCE}}(z,y)
=\max(z,0)-zy+\log\bigl(1+\exp(-\lvert z\rvert)\bigr).
$$

多分类 logits 为 $\boldsymbol z=(z_1,\ldots,z_K)$，真实类别为 $c$。令

$$
\operatorname{LSE}(\boldsymbol z)
=\log\sum_{k=1}^{K}\exp(z_k),
$$

则 softmax 交叉熵可以写成

$$
\ell_{\mathrm{CE}}(\boldsymbol z,c)
=\operatorname{LSE}(\boldsymbol z)-z_c.
$$

数值稳定的 log-sum-exp 会先取 $m=\max_k z_k$：

$$
\operatorname{LSE}(\boldsymbol z)
=m+\log\sum_{k=1}^{K}\exp(z_k-m).
$$

减去同一个最大值不会改变 softmax 的概率，却避免了无意义的巨大指数。对 logits 的梯度更能说明交叉熵在训练什么：

$$
\frac{\partial\ell_{\mathrm{CE}}}{\partial z_k}
=q_k-\mathbf 1[k=c].
$$

它会把真实类别的概率向上推，把其他类别的概率向下推，而不仅仅是在预测错时才提供一个方向。

### Brier 损失有界但仍评价概率

多分类 Brier 损失可以写成

$$
\ell_{\mathrm{Brier}}(\boldsymbol q,\boldsymbol y)
=\sum_{k=1}^{K}(q_k-y_k)^2.
$$

二分类时就是 $(q-y)^2$，取值有界。它对概率的距离是平方距离，通常比交叉熵更不容易被单个极端错误主导；交叉熵则会持续强烈惩罚把真实结果概率压到接近零的预测。二者都可以作为概率评分规则，但优化地形、尺度和对过度自信的敏感性不同。

如果只报告由概率阈值产生的准确率，就看不到 $0.51$ 和 $0.99$ 这两个概率承诺之间的差别。评估概率模型时，应同时检查 NLL 或 Brier、校准曲线以及不同阈值下的决策代价。关于交叉熵、熵和 KL 的展开可见 [交叉熵](../information-theory/cross-entropy/)。

## 分类替代损失：为不可导指标提供训练信号

二分类分数为 $s(\boldsymbol x)$，标签用 $y\in\{-1,+1\}$ 表示，定义分类间隔

$$
m=ys.
$$

$m>0$ 表示分类方向正确，$m<0$ 表示方向错误。准确率对应的 0–1 损失是

$$
\ell_{0\text{-}1}(m)=\mathbf1[m\leq0].
$$

它只关心间隔的正负，不关心正确预测有多大的余量。训练时常用平滑或分段的替代损失。

### Hinge 损失只要求间隔达到一

支持向量机常用

$$
\ell_{\mathrm{hinge}}(m)=\max(0,1-m).
$$

当 $m\geq1$ 时损失为零；当 $m<1$ 时，错误样本和“虽然正确但不够自信”的样本都会得到梯度。它强调建立一个安全间隔，但超过一以后不再奖励更大的概率置信度。

### Logistic 损失持续奖励更大间隔

Logistic 损失为

$$
\ell_{\mathrm{logistic}}(m)
=\log(1+\exp(-m)).
$$

其导数是

$$
\frac{\mathrm d\ell_{\mathrm{logistic}}}{\mathrm dm}
=-\frac1{1+\exp(m)}.
$$

当间隔为负时，梯度较大；当间隔为正且越来越大时，梯度逐渐接近零但不会在有限间隔处突然截断。它与二元交叉熵有等价的概率解释，因此通常比 hinge 更直接地保留置信度信息。

替代损失也不是准确率的数学同义词。一个模型可能交叉熵更低但准确率相同，也可能准确率更高却概率严重失准。训练目标和最终决策之间还隔着阈值、类别代价以及校准步骤。

## 不对称代价决定部署阈值

假设模型给出正类概率 $p$，把样本判为正类的假阳性代价是 $C_{\mathrm{FP}}$，判为负类的假阴性代价是 $C_{\mathrm{FN}}$。判正的条件期望代价为

$$
C_{\mathrm{FP}}(1-p),
$$

判负的条件期望代价为

$$
C_{\mathrm{FN}}p.
$$

选择正类当且仅当

$$
C_{\mathrm{FP}}(1-p)
<C_{\mathrm{FN}}p,
$$

也就是

$$
p>\frac{C_{\mathrm{FP}}}
{C_{\mathrm{FP}}+C_{\mathrm{FN}}}.
$$

所以 $0.5$ 不是普遍正确的阈值：当漏报远比误报昂贵时，阈值应下降。交叉熵负责学习概率或对数几率，部署策略再把概率映射成动作；把阈值写死在训练损失内部，会让模型难以适应代价变化。

## 不同任务的损失选择

| 输出和任务 | 常见逐样本损失 | 主要被优化的对象 | 需要额外检查 |
| --- | --- | --- | --- |
| 连续点预测 | MSE、MAE、Huber | 条件均值、中位数或稳健点预测 | 异常点、目标尺度、残差分布 |
| 条件分位数 | pinball / quantile loss | 指定分位数 | 覆盖率、区间宽度、上下界顺序 |
| 单标签多分类 | softmax 交叉熵 | 类别概率分布 | 校准、长尾类别、标签噪声 |
| 多标签分类 | 每个标签的 BCE | 多个独立 Bernoulli 概率 | 阈值、标签共现、稀疏正例 |
| 二分类间隔 | hinge、logistic | 正负类间隔或对数几率 | 准确率、AUC、代价敏感阈值 |
| 序列预测 | token NLL 加 mask | 有效位置的平均对数似然 | padding、长度偏置、perplexity |
| 排序 | pairwise hinge 或 logistic | 正样本相对负样本的顺序 | top-k、曝光偏差、分组评估 |

表中的“常见”不表示唯一正确。选择前要先写清楚标签是什么、模型输出是什么、错误代价是否对称、是否需要概率以及评估分母如何定义。后续 [分类损失](../training-nn/classification-losses/)、[回归损失](../training-nn/regression-losses/) 和排序损失专题会分别展开这些任务。

## 读懂一个损失函数的五个性质

看到一个新损失时，可以按下面五个问题拆开，而不是先背它的名字：

| 性质 | 要问的问题 | 典型影响 |
| --- | --- | --- |
| 统计目标 | 理想决策是均值、中位数、分位数还是完整分布 | 决定预测的含义 |
| 斜率和曲率 | 大错时梯度变大、固定还是饱和 | 决定异常点和更新稳定性 |
| 光滑性 | 是否在折点或边界不可导 | 影响次梯度、二阶方法和数值检查 |
| 概率语义 | 输出是否必须归一、是否是 calibrated probability | 决定 logits、阈值和报告方式 |
| 尺度和归约 | 每样本、每 token、每像素如何加权 | 决定梯度尺度和样本贡献 |

还要检查损失是否有界、是否对预测与真实值交换对称、是否允许一个标签得到零概率，以及它在目标单位变化后是否需要重新标定。一个公式只要换了归约分母，就可能变成另一个训练协议。

## 一个小例子：同一组错误可以得到不同的梯度

取两个回归样本

$$
\boldsymbol y=(2,2),
\qquad
\widehat{\boldsymbol y}=(1.5,5.5),
\qquad
\boldsymbol r=(-0.5,3.5).
$$

若对样本取平均，MAE、MSE 以及 $\delta=1$ 的 Huber 损失分别为

$$
\operatorname{MAE}=2,
\qquad
\operatorname{MSE}=6.25,
\qquad
L_{\mathrm{Huber}}=1.5625.
$$

相对于两个预测值的梯度为

$$
\nabla_{\widehat{\boldsymbol y}}\operatorname{MAE}
=(-0.5,0.5),
$$

$$
\nabla_{\widehat{\boldsymbol y}}\operatorname{MSE}
=(-0.5,3.5),
$$

$$
\nabla_{\widehat{\boldsymbol y}}L_{\mathrm{Huber}}
=(-0.25,0.5).
$$

第二个样本的残差是第一个的七倍。MAE 只让它贡献一个同样大小的正向斜率，MSE 让它贡献七倍斜率，Huber 则在 $\delta=1$ 后把贡献截为平均后的 $0.5$。损失值告诉我们整体目标差多少，梯度才告诉优化器本轮往哪里走。

对概率预测，若目标为 $(1,0)$、模型概率为 $(0.8,0.2)$，二分类 NLL 为

$$
-\log(0.8)\approx0.223143551314,
$$

Brier 平均损失为

$$
\frac{(0.8-1)^2+(0.2-0)^2}{2}=0.04.
$$

这两个数的单位和尺度不同，不能直接说哪个“更小所以更好”。只有在同一损失、同一标签集合和同一归约下比较数值，才有明确意义。

## 运行方法

下面的脚本只使用 Python 标准库，计算本文的回归、概率、归约、间隔和 logits 例子。它没有调用机器学习框架，因此输出用于核对定义和手算数字，不替代真实训练时对 dtype、设备和框架实现的测试。

```python
import math

y = [2.0, 2.0]
prediction = [1.5, 5.5]
residual = [p - t for p, t in zip(prediction, y)]
n = len(y)

mae = sum(abs(r) for r in residual) / n
mse = sum(r * r for r in residual) / n
delta = 1.0
huber = sum(
    0.5 * r * r if abs(r) <= delta else delta * (abs(r) - 0.5 * delta)
    for r in residual
) / n

mae_grad = [
    (1.0 if r > 0 else -1.0 if r < 0 else 0.0) / n
    for r in residual
]
mse_grad = [2.0 * r / n for r in residual]
huber_grad = [
    max(-delta, min(delta, r)) / n
    for r in residual
]

probability = [0.8, 0.2]
target = [1.0, 0.0]
nll = -sum(t * math.log(p) for p, t in zip(probability, target))
brier_mean = sum((p - t) ** 2 for p, t in zip(probability, target)) / 2

losses = [0.2, 1.0, 3.0]
weights = [1.0, 2.0, 1.0]
weighted_mean = sum(w * loss for w, loss in zip(weights, losses)) / sum(weights)

margins = [0.4, 1.2]
hinge = [max(0.0, 1.0 - m) for m in margins]
logistic = [math.log1p(math.exp(-m)) for m in margins]
logistic_grad = [-1.0 / (1.0 + math.exp(m)) for m in margins]

true_probability = 0.8
reported = [0.6, 0.8, 0.95]
expected_log_loss = [
    -true_probability * math.log(p)
    - (1.0 - true_probability) * math.log(1.0 - p)
    for p in reported
]

logits = [2.0, 0.0]
true_class = 0
largest = max(logits)
logsumexp = largest + math.log(
    sum(math.exp(z - largest) for z in logits)
)
softmax = [
    math.exp(z - logsumexp)
    for z in logits
]
logit_loss = logsumexp - logits[true_class]

print(f"regression MAE={mae:.12f} MSE={mse:.12f} Huber={huber:.12f}")
print("regression gradients", mae_grad, mse_grad, huber_grad)
print(f"probability NLL={nll:.12f} Brier-mean={brier_mean:.12f}")
print(
    f"reduction sum={sum(losses):.12f} mean={sum(losses) / len(losses):.12f} "
    f"weighted-mean={weighted_mean:.12f}"
)
print("margins", margins)
print("hinge", [round(value, 12) for value in hinge])
print("logistic", [round(value, 12) for value in logistic])
print("logistic-grad", [round(value, 12) for value in logistic_grad])
print(
    "proper-scoring",
    [round(value, 12) for value in expected_log_loss],
)
print(
    f"logits loss={logit_loss:.12f} "
    f"probabilities={[round(value, 12) for value in softmax]}"
)
```

运行输出：

```text
regression MAE=2.000000000000 MSE=6.250000000000 Huber=1.562500000000
regression gradients [-0.5, 0.5] [-0.5, 3.5] [-0.25, 0.5]
probability NLL=0.223143551314 Brier-mean=0.040000000000
reduction sum=4.200000000000 mean=1.400000000000 weighted-mean=1.300000000000
margins [0.4, 1.2]
hinge [0.6, 0.0]
logistic [0.5130152524, 0.263282467338]
logistic-grad [-0.401312339888, -0.231475216501]
proper-scoring [0.591918645388, 0.500402423538, 0.640181090221]
logits loss=0.126928011043 probabilities=[0.880797077978, 0.119202922022]
```

这里的 weighted-mean 用权重和作为分母：$(0.2\times1+1.0\times2+3.0\times1)/(1+2+1)=1.3$。若把同一批损失除以样本数，数值会不同，含义也会不同。logits loss 直接使用 log-sum-exp，避免先计算一个可能下溢的 softmax 再取对数。

## 常见失效模式

### 用评估指标直接替代训练损失

准确率、F1、top-k 命中和离散业务动作通常包含阈值或排序，不能为每个参数点提供稳定的局部方向。它们适合做冻结模型后的报告，不代表一定适合直接反向传播。可以用交叉熵或其他替代损失训练，再在验证集上选择阈值并报告最终指标。

### 只改了归约，没有重标定训练

把 mean 改成 sum、把有效 token 平均改成全张量平均，都会改变梯度尺度。随后若仍使用原学习率、梯度裁剪阈值和正则系数，比较就不再是同一训练协议。日志至少记录 batch 大小、有效元素数、权重和以及损失归约。

### 在概率空间中制造数值问题

先做 softmax 再取 $\log$，或先做 sigmoid 再计算两个对数，在极端 logits 下会出现零概率和无穷损失。训练接口优先使用 logits 版本的 BCE 或交叉熵；概率只在展示、校准和决策阶段读取。若为了报告做概率截断，应明确它改变了报告公式，而不是声称修复了训练目标。

### 忽略标签、掩码和权重的定义

0 标签、padding 位置、缺失标签和“负类”并不总是同一件事。把未知标签当成负类会系统性改变目标，把 padding 当作真实 token 会让长度成为隐形权重，把类别权重写在损失外却忘了调整分母也会改变梯度尺度。逐样本损失和有效掩码必须能被单独导出。

### 把损失的低值当成概率校准

交叉熵低说明平均对数代价低，不保证每个置信度区间都与真实频率一致；Brier 低也不自动说明阈值策略正确。应在独立验证集上画可靠性图、统计分箱频率，并检查时间切分或群组切分后的漂移。校准、阈值和业务代价是训练目标之外的后处理问题。

### 只在平均数上做诊断

平均损失下降可能掩盖一个类别、一个用户群或一段时间窗口的退化。保存逐样本或逐分组损失的分布，按标签、难度、长度和数据来源切片，再比较相同归约后的指标。若线上代价由尾部事件主导，还要报告分位数、最大损失或风险约束，而不是只看均值。

## 核验协议

实现一个新损失时，至少做五层核对：

1. 用两个或三个手算样本核对损失值、符号、边界和极端输入；
2. 对无闭式梯度的折点分别检查左右导数、次梯度约定和零值处理；
3. 用中心差分或方向导数检查参数梯度，并固定随机种子、掩码和归约；
4. 用大正、大负和全相等 logits 检查稳定实现与概率实现是否在可表示范围内一致；
5. 把逐样本值、有效计数、权重和、归约损失以及最终评估指标一起记录。

如果损失被用于不同任务或不同尺度，还要分别核对单位、标签编码、类别权重、阈值和部署代价。只有数学式、代码实现和评估协议三者指向同一个对象，损失曲线才有可解释性。

## 相关词条

- [回归损失](../training-nn/regression-losses/)
- [分类损失](../training-nn/classification-losses/)
- [交叉熵](../information-theory/cross-entropy/)
- [经验风险最小化](../learning-framework/empirical-risk-minimization/)
- [回归评估指标](../evaluation-and-generalization/regression-metrics/)
- [分类评估指标](../evaluation-and-generalization/classification-metrics/)
- [梯度下降](../training-nn/gradient-descent/)
