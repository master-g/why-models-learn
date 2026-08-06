---
title: "间隔与支持向量机：让分类边界留出安全距离"
tags: ["why-models-learn"]
---

支持向量机不满足于把训练样本分到边界两侧，而是在可行的线性分界中选择几何间隔最大的那个。硬间隔要求每个样本都离边界至少一个规范化单位；软间隔允许违反约束，并用 hinge loss 和参数 $C$ 交换间隔宽度与训练错误。支持向量是约束最紧的样本，它们决定了最终边界。

![支持向量机在两类样本之间寻找最大间隔和支持向量](/assets/linear-models/svg/margins-and-svm.1.svg)

## 分数、函数 margin 与几何 margin

仍然使用二分类标签 $y\in\{-1,+1\}$ 和仿射分数

$$
f(x)=w^{\mathsf T}x+b.
$$

预测只看符号：

$$
\widehat y=\operatorname{sign}\bigl(f(x)\bigr).
$$

带符号的函数 margin 为

$$
m_i=y_i f(x_i)
=y_i\left(w^{\mathsf T}x_i+b\right).
$$

当 $m_i>0$ 时，样本在正确一侧；$m_i<0$ 时，样本被分错。这个数仍然依赖参数的尺度：把 $(w,b)$ 同时乘以正数，边界不变，但所有函数 margin 都被同样放大。

边界到点的垂直距离则由 $w$ 的长度归一化。对一个点 $x_0$，它到超平面 $w^{\mathsf T}x+b=0$ 的距离是

$$
d(x_0,\mathcal H)
=\frac{\lvert w^{\mathsf T}x_0+b\rvert}{\lVert w\rVert_2}.
$$

带上标签以后，几何 margin 为

$$
\rho_i
=\frac{y_i\left(w^{\mathsf T}x_i+b\right)}{\lVert w\rVert_2}
=\frac{m_i}{\lVert w\rVert_2}.
$$

所以函数 margin 适合写约束，几何 margin 才能比较不同参数尺度下的真实距离。SVM 会先固定函数 margin 的尺度，再最大化对应的几何间隔。

## 为什么可以把约束规范化为 1

如果一组参数把所有样本分开，那么对任意 $c>0$，$(cw,cb)$ 给出同一条边界。假设最小函数 margin 为正数 $m_{\min}$，选择 $c=1/m_{\min}$ 后，就得到一组等价参数，使

$$
y_i\left(w^{\mathsf T}x_i+b\right)\ge1.
$$

在这个规范化下，离边界最近的两侧平行面是

$$
w^{\mathsf T}x+b=+1,
\qquad
w^{\mathsf T}x+b=-1.
$$

它们到中心边界的距离各为 $1/\lVert w\rVert_2$，两侧之间的总间隔为

$$
\operatorname{width}
=\frac{2}{\lVert w\rVert_2}.
$$

因此最大化间隔等价于最小化 $w$ 的平方长度。平方形式更容易优化，也避免了平方根：

$$
\min_{w,b}\ \frac12\lVert w\rVert_2^2
\quad\text{subject to}\quad
y_i\left(w^{\mathsf T}x_i+b\right)\ge1,
\quad i=1,\ldots,n.
$$

这就是线性 hard-margin SVM 的原始问题。目标函数让边界尽可能宽，约束保证每个训练点位于正确的规范化一侧。

## 一个硬间隔的精确例子

只取两个点：

| 样本 | $x$ | $y$ |
| --- | --- | --- |
| $A$ | $(1,1)$ | $+1$ |
| $B$ | $(-1,-1)$ | $-1$ |

考虑参数

$$
w=\left(\frac12,\frac12\right)^{\mathsf T},
\qquad
b=0.
$$

两个点的函数 margin 都是

$$
m_A=m_B=1.
$$

它们同时落在两条规范化边界上，因此都是支持向量。参数长度为

$$
\lVert w\rVert_2=\frac1{\sqrt2}.
$$

中心边界是 $x_1+x_2=0$，单侧几何 margin 为

$$
\frac1{\lVert w\rVert_2}=\sqrt2,
$$

两侧之间的总间隔为 $2\sqrt2$。如果把 $w$ 再缩小，约束会被违反；如果把 $w$ 放大，边界不变但间隔变窄，所以这个规范化解已经达到最大间隔。

这个例子也提醒我们，$b$ 的作用是平移边界。把不带偏置的线性函数误当成过原点的直线，会错过很多合理的分类边界。

## 支持向量为什么重要

对 hard-margin 问题，约束可以分成三类：

| 样本状态 | 函数 margin | 对应约束 | 是否通常为支持向量 |
| --- | ---: | --- | --- |
| 正好贴在边界 | $m_i=1$ | 约束取等号 | 是 |
| 正确且离边界较远 | $m_i>1$ | 约束有余量 | 否 |
| 训练不可行 | $m_i<1$ | hard-margin 无法接受 | 不适用 |

支持向量不是“所有被模型看过的样本”的别名，而是对偶系数为正、约束足够紧的样本。在线性 hard-margin 情形，如果删掉一个远离边界的点，最优边界通常不变；删掉支持向量，则边界可能旋转或平移。

这和感知机的更新规则不同。感知机只在 $m_i\le0$ 时更新，正确但 margin 很小的点不会继续影响它；SVM 在 $0<m_i<1$ 时仍然通过 hinge loss 施加压力，把安全距离纳入目标。

## Soft-margin SVM 接受少量违反

真实数据常常含有噪声、标签错误或重叠类别，硬间隔约束可能根本无解。引入松弛变量 $\xi_i\ge0$：

$$
y_i\left(w^{\mathsf T}x_i+b\right)\ge1-\xi_i.
$$

$\xi_i$ 的大小表示约束违反了多少：

- $\xi_i=0$：样本在规范化 margin 外或正好位于 margin 上；
- $0<\xi_i<1$：样本仍在正确一侧，但落入 margin；
- $\xi_i\ge1$：样本在边界上或已经分错。

软间隔原始问题为

$$
\min_{w,b,\xi}
\frac12\lVert w\rVert_2^2
+C\sum_{i=1}^{n}\xi_i
$$

满足

$$
y_i\left(w^{\mathsf T}x_i+b\right)\ge1-\xi_i,
\qquad
\xi_i\ge0.
$$

$C$ 控制违反约束的代价。$C$ 大时，优化器更愿意增大参数、缩小间隔来减少训练违反；$C$ 小时，优化器更愿意保留宽间隔、接受更多违反。它不是“训练轮数”，也不是可以脱离数据尺度直接比较的通用质量分数。

## Hinge loss 把松弛变量消掉

对固定的 $w,b$，满足约束所需的最小松弛量是

$$
\xi_i
=\max\bigl(0,1-y_i f(x_i)\bigr).
$$

代回原始目标，得到无显式松弛变量的形式：

$$
\min_{w,b}
\frac12\lVert w\rVert_2^2
+C\sum_{i=1}^{n}
\max\bigl(0,1-y_i f(x_i)\bigr).
$$

右侧的分段函数就是 hinge loss：

$$
\ell_{\mathrm{hinge}}(m)
=\max(0,1-m).
$$

它在 $m\ge1$ 时为零，在 $m<1$ 时线性增加。这个目标把两个偏好放在一起：第一项希望 $w$ 小、边界宽；第二项希望样本的带符号 margin 至少达到 1。

对一个样本，三种状态的 hinge loss 为

$$
\begin{array}{c|c|c}
\text{状态}&m&\ell_{\mathrm{hinge}}(m)\\
\text{远离正确侧}&m\ge1&0\\
\text{正确但在间隔内}&0<m<1&1-m\\
\text{边界上或分错}&m\le0&1-m
\end{array}
$$

感知机损失只在错误侧提供信号；hinge loss 还会推动 margin 不足的正确样本。逻辑回归则用平滑的对数损失持续区分不同概率，即便样本已经在正确侧很远，也不会立刻得到完全为零的损失。

## 一个带异常点的软间隔账本

考虑一维样本：

| 样本 | $x$ | $y$ |
| --- | ---: | ---: |
| $A$ | $-2$ | $-1$ |
| $B$ | $-1$ | $-1$ |
| $C$ | $1$ | $+1$ |
| $D$ | $2$ | $+1$ |
| $E$ | $0$ | $-1$ |

前四个点由 $f(x)=x$ 完美分开，最后一个点 $E$ 位于边界上且标签为负。对 $w=1,b=0$：

$$
m_A=2,\quad
m_B=1,\quad
m_C=1,\quad
m_D=2,\quad
m_E=0.
$$

对应的 hinge loss 是

$$
(0,0,0,0,1).
$$

因此这个候选解的 soft-margin 目标值为

$$
\frac12w^2+C\sum_i\ell_{\mathrm{hinge}}(m_i)
=\frac12+C.
$$

这不是说它对所有 $C$ 都是最优解，而是把 $C$ 的作用算出来：异常点每违反一个单位，就给目标增加 $C$；参数平方项则把扩大斜率的代价记录为 $w^2/2$。实际最优解会在更宽的间隔与更少的违反之间比较这些项。

## KKT 条件把支持向量筛出来

为了理解对偶问题，给 hard-margin 约束写成

$$
g_i(w,b)=1-y_i\left(w^{\mathsf T}x_i+b\right)\le0.
$$

给每个约束一个乘子 $\alpha_i\ge0$，拉格朗日函数为

$$
\mathcal L
=\frac12\lVert w\rVert_2^2
-\sum_i\alpha_i
\left[
y_i\left(w^{\mathsf T}x_i+b\right)-1
\right].
$$

对 $w$ 和 $b$ 求驻点：

$$
\begin{aligned}
\nabla_w\mathcal L=0
&\Longrightarrow
w=\sum_i\alpha_i y_i x_i,\\
\frac{\partial\mathcal L}{\partial b}=0
&\Longrightarrow
\sum_i\alpha_i y_i=0.
\end{aligned}
$$

互补松弛条件是

$$
\alpha_i
\left[
y_i\left(w^{\mathsf T}x_i+b\right)-1
\right]=0.
$$

因此，如果某个点的 margin 严格大于 1，方括号内为正，必须有 $\alpha_i=0$；只有约束取等号的点才可能有正的 $\alpha_i$。这就是支持向量的优化解释。

把驻点代回去，hard-margin 的对偶问题只依赖样本内积：

$$
\max_{\alpha}
\sum_i\alpha_i
-\frac12\sum_{i,j}
\alpha_i\alpha_j y_i y_j x_i^{\mathsf T}x_j
$$

满足

$$
\alpha_i\ge0,
\qquad
\sum_i\alpha_i y_i=0.
$$

前面的两个点中，两个约束都活跃；由对称性 $\alpha_A=\alpha_B$，驻点关系给出

$$
\alpha_A=\alpha_B=\frac14.
$$

于是

$$
w
=\frac14(+1)(1,1)
+\frac14(-1)(-1,-1)
=\left(\frac12,\frac12\right).
$$

对偶形式说明了一个重要事实：如果只通过内积就能求解，训练时不必显式构造高维特征向量。

## Kernel 把内积换成相似度

假设存在一个特征映射 $\phi(x)$，把输入送到更高维空间。在这个空间训练线性 SVM 时，对偶目标只出现

$$
\phi(x_i)^{\mathsf T}\phi(x_j).
$$

如果一个核函数直接给出这个内积：

$$
K(x_i,x_j)
=\phi(x_i)^{\mathsf T}\phi(x_j),
$$

就可以把对偶目标中的每个内积替换成 $K(x_i,x_j)$，而不用写出 $\phi(x)$。常见例子包括

$$
K_{\mathrm{poly}}(x,x')
=\left(x^{\mathsf T}x'+c\right)^d
$$

和

$$
K_{\mathrm{rbf}}(x,x')
=\exp\left(-\gamma\lVert x-x'\\rVert_2^2\right).
$$

线性 SVM 在原始特征空间找超平面；核 SVM 在隐式特征空间找超平面，映回原始空间后可能是曲线边界。代价是预测通常要和许多支持向量计算核值，支持向量数量会直接影响推理成本。

核函数不能随意拼成任意相似度。要得到凸的标准 SVM 对偶问题，核矩阵需要满足对称和半正定等条件；否则优化问题的几何性质会改变。

## SVM 分数仍然不是概率

SVM 输出的是决策分数 $f(x)$ 和符号，不是满足概率约束的 $p(y\mid x)$。分数的绝对大小受特征尺度、$C$、正则化和核参数影响；即使两个样本的分数分别为 2 和 1，也不能直接说前者的正确概率是后者的两倍。

如果部署需要概率，可以在独立验证数据上拟合校准器，例如用 sigmoid 形式把 SVM 分数映射到概率，或使用其他校准方法。校准器必须成为评估流程的一部分，不能把未经校准的分数直接送进期望代价计算。

| 输出 | 能直接回答的问题 | 不能直接回答的问题 |
| --- | --- | --- |
| 符号 | 预测哪一类 | 这个预测的概率是多少 |
| 分数大小 | 样本在边界哪一侧、排序如何 | 错误概率或业务风险 |
| 几何 margin | 离当前边界有多远 | 分布外样本一定可靠吗 |
| 校准概率 | 长期频率和期望风险 | 边界是否是最大间隔得到的 |

## 训练与选择流程

1. 明确标签编码、错误代价和预测时点，检查是否存在标签冲突；
2. 在训练集内拟合标准化参数，尤其要注意线性模型和 RBF 核对尺度的敏感性；
3. 先用线性 hard-margin 或 soft-margin 作为基线，记录支持向量数量和验证 margin；
4. 在验证集上选择 $C$、核函数、$\gamma$、多项式次数和类别权重；
5. 用混淆矩阵、每类召回率、决策分数排序和校准证据评估，不只看训练准确率；
6. 冻结全部选择后，在独立测试集报告指标，并保留支持向量、标准化和校准器；
7. 检查训练时间、支持向量数量和单样本预测成本是否符合部署约束；
8. 对时间外、分布外和少数类样本重复检查 margin 与错误类型。

标准化、$C$ 和核参数必须一起考虑。改变特征单位会改变内积、函数 margin 和正则化相对强度；因此“沿用上一个数据集的 $C$”通常没有可比性。

## 失效模式

**把函数 margin 当成几何距离。** 不除以 $\lVert w\rVert_2$ 时，正倍数缩放参数会改变 margin 数值却不改变边界；比较间隔必须使用规范化约束或几何 margin。

**在不可分数据上强行使用 hard-margin。** 标签冲突、异常点和类别重叠会让约束无解；先换 soft-margin，再检查异常来源和评估代价。

**把 $C$ 当成越大越好。** $C$ 大会重罚训练违反，可能换来很窄的间隔和对噪声的适应；必须用验证集选择。

**不做特征缩放就比较内积和核值。** 大尺度特征会支配线性分数和 RBF 距离，导致 $C$、$\gamma$ 的解释一起漂移。

**把所有正确样本都称为支持向量。** 支持向量对应紧约束或对偶系数为正；离边界很远的正确样本通常不直接决定解。

**把 SVM 分数当成概率。** 分数只给出边界侧和排序，概率需要独立校准，校准后还要检查分布偏移。

**忽略核矩阵和推理成本。** 不满足半正定条件的相似度可能破坏凸性；支持向量过多时，预测每个样本都要付出较高的核计算成本。

**在测试集调 $C$、核参数或校准器。** 这会把测试反馈带入模型选择，最终指标不再代表冻结流程的泛化表现。

## 相关词条

- [经典感知机](../linear-models/perceptron-classic/)：对照“分对即可”的错误驱动更新和可分收敛。
- [逻辑回归](../linear-models/logistic-regression/)：比较平滑概率损失、对数几率和校准。
- [Softmax 回归](../linear-models/softmax-regression/)：比较多分类概率输出和竞争式交叉熵。
- [核技巧](../linear-models/kernel-trick/)：展开隐式特征映射、核矩阵和核方法。
- [分类指标](../evaluation-and-generalization/classification-metrics/)：检查硬标签、排序、margin 和概率评估。
- [训练、验证与测试集](../learning-framework/train-validation-test/)：规定 $C$、核参数和校准器的选择边界。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：理解宽间隔和显式惩罚的泛化取舍。
- [仿射空间与映射](../linear-algebra/affine-spaces-and-maps/)：理解偏置和分类超平面的几何结构。
