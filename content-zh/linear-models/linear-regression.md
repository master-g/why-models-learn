---
title: "线性回归：用特征的线性组合拟合数值关系"
tags: ["why-models-learn"]
---

线性回归用特征的线性组合预测一个数值标签。它的“线性”指对参数线性，而不要求输入和输出在原始坐标上只能是一条直线；加入平方项、交互项或一组固定基函数后，模型仍然可以对参数保持线性。本文从设计矩阵和最小二乘目标出发，推导正规方程、投影与残差，再说明系数的统计解释、特征编码、诊断方法、正则化和欠定系统的边界。

![线性回归从数据点、残差到矩阵正规方程](/assets/linear-models/svg/linear-regression.1.svg)

## “线性”是对参数线性

先看一个带截距的一元模型：

$$
\widehat y_i=\beta_0+\beta_1x_i.
$$

它对参数 $\beta_0,\beta_1$ 是线性的。若加入平方项，模型变成

$$
\widehat y_i=\beta_0+\beta_1x_i+\beta_2x_i^2,
$$

对参数仍然线性，虽然它对输入 $x_i$ 已经是曲线。更一般地，先把输入经过固定特征变换 $\phi$：

$$
\phi(x)=
\begin{pmatrix}
1\\
\phi_1(x)\\
\vdots\\
\phi_p(x)
\end{pmatrix},
\qquad
\widehat y=\phi(x)^{\mathsf T}\beta.
$$

只要 $\phi$ 不依赖待估参数 $\beta$，最小二乘仍然是在参数向量上解一个二次问题。神经网络若同时学习特征变换和最后的权重，整体就不再是这个简单的线性回归问题；固定最后一层特征时，最后一层仍可以是线性回归。

给定 $n$ 个样本和 $p$ 个非截距特征，把每个样本的特征行排成设计矩阵：

$$
X=
\begin{pmatrix}
1&x_{11}&\cdots&x_{1p}\\
1&x_{21}&\cdots&x_{2p}\\
\vdots&\vdots&&\vdots\\
1&x_{n1}&\cdots&x_{np}
\end{pmatrix}
\in\mathbb R^{n\times(p+1)},
\qquad
\beta=
\begin{pmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_p
\end{pmatrix}.
$$

标签向量和预测向量分别是

$$
y=
\begin{pmatrix}
y_1\\\vdots\\y_n
\end{pmatrix},
\qquad
\widehat y=X\beta.
$$

这些对象的职责不同：

| 对象 | 数学形式 | 作用 |
| --- | --- | --- |
| 样本特征 | $x_i$ | 一个输入样本的原始或变换后特征 |
| 设计矩阵 | $X$ | 把所有样本的特征行集中起来 |
| 参数 | $\beta$ | 每个特征在模型中的线性权重 |
| 标签 | $y$ | 训练时要拟合的数值目标 |
| 预测 | $\widehat y=X\beta$ | 把同一组参数作用到全部样本 |

截距并不是“最后再加一个常数”这么简单。把全为 1 的列放进 $X$，意味着模型允许预测平面整体平移；去掉这列则强制当所有特征为 0 时预测也为 0。是否包含截距是建模假设的一部分。

## 最小二乘把拟合写成距离最小

线性回归最常用的训练目标是残差平方和：

$$
\operatorname{RSS}(\beta)
=
\lVert y-X\beta\rVert_2^2
=
\sum_{i=1}^{n}(y_i-\widehat y_i)^2.
$$

除以样本数得到训练均方误差：

$$
\operatorname{MSE}(\beta)
=
\frac{1}{n}\operatorname{RSS}(\beta).
$$

平方让正负残差不会相互抵消，也让大残差受到更大惩罚。它同时把问题变成一个可用线性代数直接分析的凸二次目标。令残差为

$$
r(\beta)=y-X\beta.
$$

把目标展开：

$$
\begin{aligned}
\operatorname{RSS}(\beta)
&=(y-X\beta)^{\mathsf T}(y-X\beta)\\
&=y^{\mathsf T}y-2\beta^{\mathsf T}X^{\mathsf T}y
+\beta^{\mathsf T}X^{\mathsf T}X\beta.
\end{aligned}
$$

对参数求梯度：

$$
\nabla_\beta\operatorname{RSS}(\beta)
=-2X^{\mathsf T}y+2X^{\mathsf T}X\beta.
$$

令梯度为零得到正规方程：

$$
X^{\mathsf T}X\widehat\beta=X^{\mathsf T}y.
$$

如果 $X$ 的列线性无关，$X^{\mathsf T}X$ 可逆，解为

$$
\widehat\beta
=
\left(X^{\mathsf T}X\right)^{-1}X^{\mathsf T}y.
$$

这个公式不是说实际工程都应该显式计算矩阵逆。它是唯一解存在时的数学表达；数值实现通常使用 QR、SVD 或迭代优化来避免直接求逆的稳定性和成本问题。

### 三个样本的完整计算

取一个带截距的一元回归：

$$
X=
\begin{pmatrix}
1&0\\
1&1\\
1&2
\end{pmatrix},
\qquad
y=
\begin{pmatrix}
1\\3\\7
\end{pmatrix}.
$$

正规方程中的两个矩阵是

$$
X^{\mathsf T}X=
\begin{pmatrix}
3&3\\
3&5
\end{pmatrix},
\qquad
X^{\mathsf T}y=
\begin{pmatrix}
11\\17
\end{pmatrix}.
$$

因为

$$
\left(X^{\mathsf T}X\right)^{-1}
=
\frac16
\begin{pmatrix}
5&-3\\
-3&3
\end{pmatrix},
$$

所以

$$
\widehat\beta
=
\frac16
\begin{pmatrix}
5&-3\\
-3&3
\end{pmatrix}
\begin{pmatrix}
11\\17
\end{pmatrix}
=
\begin{pmatrix}
2/3\\3
\end{pmatrix}.
$$

拟合值为

$$
\widehat y=
\begin{pmatrix}
2/3\\
11/3\\
20/3
\end{pmatrix},
\qquad
r=y-\widehat y=
\begin{pmatrix}
1/3\\
-2/3\\
1/3
\end{pmatrix}.
$$

残差平方和与均方误差分别是

$$
\operatorname{RSS}
=
\left(\frac13\right)^2
+\left(-\frac23\right)^2
+\left(\frac13\right)^2
=\frac23,
\qquad
\operatorname{MSE}=\frac{\operatorname{RSS}}{3}=\frac29.
$$

直线没有穿过最后一个标签，因为平方损失要求它同时折中三个点。若加入更高次特征，训练误差可以继续降低；这会改变假设空间和泛化风险，而不是说明当前线性解算错了。

## 正规方程也是一个正交条件

正规方程可以重新写成

$$
X^{\mathsf T}(y-X\widehat\beta)=0,
$$

也就是

$$
X^{\mathsf T}r=0.
$$

$X$ 的列张成一个子空间 $\operatorname{col}(X)$。预测向量 $X\widehat\beta$ 落在这个子空间里，而残差 $r$ 与子空间的每一列都正交。因此最小二乘做的事情是：在所有可能的线性预测向量中，找出离标签向量 $y$ 最近的那个。

这和[正交投影](../linear-algebra/orthogonal-projections/) 中的最近点定理是同一个结构。若 $X$ 列满秩，投影矩阵或帽子矩阵为

$$
H=
X\left(X^{\mathsf T}X\right)^{-1}X^{\mathsf T},
\qquad
\widehat y=Hy.
$$

它满足

$$
H^2=H,
\qquad
H^{\mathsf T}=H,
$$

分别表示投影两次不再改变结果，以及投影和欧氏内积相容。残差生成矩阵是

$$
I-H,
\qquad
r=(I-H)y.
$$

所以

$$
Hr=0,
$$

残差完全落在设计矩阵列空间的正交补中。

对上面的三点数据，帽子矩阵是

$$
H=
\begin{pmatrix}
5/6&1/3&-1/6\\
1/3&1/3&1/3\\
-1/6&1/3&5/6
\end{pmatrix}.
$$

对角线元素

$$
h_{11}=5/6,\qquad h_{22}=1/3,\qquad h_{33}=5/6
$$

称为杠杆值。第一个和第三个输入在一元带截距设计中离特征均值更远，因此它们对拟合线的几何约束更强。高杠杆不等于异常值：它只说明输入位置特殊；还要结合残差大小判断它是否真正影响了拟合。

## 截距、中心化与系数公式

一元带截距回归可以直接用样本均值写出系数。令

$$
\bar x=\frac1n\sum_{i=1}^{n}x_i,
\qquad
\bar y=\frac1n\sum_{i=1}^{n}y_i.
$$

对

$$
\operatorname{RSS}(\beta_0,\beta_1)
=\sum_i\left(y_i-\beta_0-\beta_1x_i\right)^2
$$

分别对 $\beta_0$ 和 $\beta_1$ 求导。截距方程给出残差和为零：

$$
\sum_i\left(y_i-\widehat\beta_0-\widehat\beta_1x_i\right)=0.
$$

因此

$$
\widehat\beta_0=\bar y-\widehat\beta_1\bar x.
$$

把它代回斜率方程，得到

$$
\widehat\beta_1
=
\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}
\sum_i(x_i-\bar x)^2,
\qquad
\widehat\beta_0=\bar y-\widehat\beta_1\bar x.
$$

分子是未归一化协方差，分母是输入的未归一化方差。对前面的数据，$\bar x=1$、$\bar y=11/3$，所以

$$
\widehat\beta_1=3,
\qquad
\widehat\beta_0=\frac{11}{3}-3=\frac23.
$$

中心化特征 $\widetilde x_i=x_i-\bar x$ 后，斜率不变，截距坐标变成 $\bar y$。这可以改善数值条件，也让截距代表“平均输入处的预测”而不是输入为 0 时的预测。若特征的 0 点本身没有实际意义，直接解释原始截距通常会误导。

系数的单位也必须一起看。若 $x_j$ 的单位变成原来的 100 倍，保持同一预测需要让 $\beta_j$ 缩小 100 倍。系数绝对值不能脱离特征尺度比较；比较重要性时可以先标准化，或报告对一个有实际意义的输入变化量的预测变化。

## 统计模型解释了系数何时可信

最小二乘只是一个优化规则。若要把系数解释成总体规律，通常还要写出数据生成假设：

$$
Y=X\beta^\star+\varepsilon.
$$

把 $X$ 视为已给定的设计矩阵，如果满足

$$
\mathbb E[\varepsilon\mid X]=0,
$$

则

$$
\mathbb E[\widehat\beta\mid X]=\beta^\star.
$$

这表示估计器在重复抽取标签噪声时平均不偏。若噪声条件协方差为

$$
\operatorname{Cov}(\varepsilon\mid X)=\sigma^2I,
$$

则参数协方差为

$$
\operatorname{Cov}(\widehat\beta\mid X)
=
\sigma^2\left(X^{\mathsf T}X\right)^{-1}.
$$

输入列越接近线性相关，$X^{\mathsf T}X$ 越接近奇异，估计方差就越大。若噪声还服从高斯分布，系数的标准误可以进一步用于 t 区间和假设检验；高斯性不是最小二乘解存在的必要条件，而是精确小样本推断的额外条件。

对一个新的输入向量 $x_0$，要记得它也包含截距坐标，例如 $x_0=(1,1)^{\mathsf T}$。平均响应预测的方差是

$$
\operatorname{Var}\bigl(x_0^{\mathsf T}\widehat\beta\mid X\bigr)
=
\sigma^2x_0^{\mathsf T}
\left(X^{\mathsf T}X\right)^{-1}
x_0.
$$

如果要预测一个带新噪声的未来标签，还要加上 $\sigma^2$：

$$
\operatorname{Var}\bigl(Y_0-x_0^{\mathsf T}\widehat\beta\mid X\bigr)
=
\sigma^2\left[
1+x_0^{\mathsf T}
\left(X^{\mathsf T}X\right)^{-1}
x_0
\right].
$$

对三点设计、$x_0=(1,1)^{\mathsf T}$ 和 $\sigma^2=1/4$，帽子矩阵中间的杠杆值是 $1/3$，所以平均响应的预测方差是 $1/12$，未来观测的预测方差是 $1/3$。后者更大，是因为它还包含一份新的观测噪声。

## 特征工程仍然可以保持线性回归

线性回归的输入列不必是原始变量：

| 特征形式 | 模型例子 | 系数的条件含义 |
| --- | --- | --- |
| 连续变量 | $\beta_0+\beta_1x$ | 其他列固定时 $x$ 增加一单位的预测变化 |
| 多项式列 | $\beta_0+\beta_1x+\beta_2x^2$ | 曲率由固定的 $x$ 与 $x^2$ 共同决定 |
| 交互列 | $\beta_0+\beta_1x+\beta_2z+\beta_3xz$ | $x$ 的作用随 $z$ 改变 |
| one-hot 列 | $\beta_0+\sum_c\beta_c\mathbf 1\{g=c\}$ | 类别相对参考类别的平均差异 |
| 时间或空间基函数 | $\sum_j\beta_j\phi_j(t)$ | 在选定基函数空间中的线性组合 |

多项式特征会扩大列空间，交互项会允许条件效应，one-hot 编码会把分类变量转成数值列。它们也可能引入共线性和过拟合，所以“能加一列”不等于“应该加一列”。特征变换必须在训练集内拟合，不能用测试集的均值、方差或类别信息提前改变设计矩阵。

分类变量若包含 $k$ 个类别，通常保留 $k-1$ 个指示列并把剩下一类作为参考；若同时保留截距和全部 $k$ 列，列之间会出现精确线性关系。这个冗余不是模型表达能力增加，而是参数表示不唯一。

## 诊断要看残差如何排列

训练 MSE 是一个汇总数字，不能告诉我们模型在哪些输入区域失效。至少应把残差

$$
r_i=y_i-\widehat y_i
$$

按输入、预测值、时间或业务分组画出来，并检查以下模式：

| 残差现象 | 可能原因 | 下一步检查 |
| --- | --- | --- |
| 随输入呈弯曲或周期模式 | 线性特征空间缺少结构 | 加基函数、交互或非线性模型，并用验证集比较 |
| 预测值越大散布越宽 | 异方差 | 变换目标、加权最小二乘或使用稳健标准误 |
| 少数点残差极大 | 离群标签、测量错误或漏掉变量 | 查数据来源，不要先盲删 |
| 某些点杠杆值很高 | 输入远离中心或设计稀疏 | 计算影响度，检查部署是否会遇到这类输入 |
| 系数在重复切分中大幅改变 | 共线性、样本不足或分布偏移 | 看相关列、奇异值、条件数和分组稳定性 |
| 训练误差很低但时间外误差变高 | 泄漏或任务关系变化 | 按时间切分并审计特征生成时点 |

带截距的 OLS 残差和为零、残差与每一列特征正交，这是拟合目标带来的代数事实，不是模型正确的证据。残差图没有模式，也只能说明选定的诊断没有发现明显问题，不能证明因果关系或未来分布稳定。

共线性尤其容易被误读。若两列特征几乎可以互相线性预测，模型的整体预测可能仍然稳定，但两个系数会在重复样本或轻微正则化下大幅交换。此时“单个系数的方向”不是可靠的独立贡献解释；可以合并变量、改变基底或只解释可识别的组合。

## 奇异、欠定与正则化

若 $X$ 的列不满秩，$X^{\mathsf T}X$ 不可逆，普通公式不能直接使用。最小二乘解可能不唯一，但预测向量 $X\widehat\beta$ 仍然可以唯一；[伪逆](../linear-algebra/pseudoinverse/) 选择其中范数最小的解：

$$
\widehat\beta_{\mathrm{pinv}}=X^+y.
$$

当参数数超过样本数时，若系统可插值，零训练误差解可能有很多个。选择哪一个会影响未见输入的预测，[双下降](../learning-framework/double-descent/) 已经展示了插值阈值附近的风险变化。

岭回归通过加入平方范数惩罚稳定相关列：

$$
\widehat\beta_{\mathrm{ridge}}
=
\operatorname*{arg\,min}_{\beta}
\left[
\lVert y-X\beta\rVert_2^2
+\lambda\lVert\beta\rVert_2^2
\right],
\qquad
\lambda>0.
$$

它的解是

$$
\widehat\beta_{\mathrm{ridge}}
=
\left(X^{\mathsf T}X+\lambda I\right)^{-1}X^{\mathsf T}y.
$$

实际带截距时，通常不惩罚截距，把 $I$ 换成对角矩阵 $D=\operatorname{diag}(0,1,\ldots,1)$。岭回归用一点偏差换取更小的估计方差，$\lambda$ 应由训练内交叉验证或验证集选择，而不是看测试集挑选。L1 惩罚产生稀疏解的几何和算法不同，留给[岭回归与 Lasso](../linear-models/ridge-and-lasso/) 展开。

一个不带截距的数字例子是

$$
X=
\begin{pmatrix}
1\\2
\end{pmatrix},
\qquad
y=
\begin{pmatrix}
1\\3
\end{pmatrix},
\qquad
\lambda=1.
$$

此时 $X^{\mathsf T}X=5$、$X^{\mathsf T}y=7$。无惩罚解为 $7/5=1.4$，岭解为

$$
\widehat\beta_{\mathrm{ridge}}
=\frac{7}{5+1}=\frac76\approx1.1667.
$$

惩罚把参数从训练标签要求的方向拉回 0；它没有凭空增加数据，只是改变了在多个拟合取舍之间选择哪一个。

## 一次可复用的线性回归检查

面对一个新的回归任务，可以按下面顺序建立证据：

1. 先定义标签的时间点、单位和损失，确认特征在预测时确实可见；
2. 明确是否包含截距，记录类别编码、缺失值处理和所有特征变换；
3. 检查设计矩阵的列数、秩、奇异值和尺度，先发现精确冗余与严重共线性；
4. 在训练集上拟合 OLS，并保存系数、预测、残差和训练风险；
5. 用残差图、杠杆值、分组误差和时间切分检查模型假设；
6. 用验证集比较特征、岭惩罚和其他模型，报告不确定性而不只报告最低 MSE；
7. 方案冻结后只用独立测试集评估，并保留一组分布外或时间外样本作为压力测试。[训练、验证与测试集](../learning-framework/train-validation-test/) 规定了选择与最终证据的边界。

如果预测目标是因果效应或政策干预，线性回归的相关性拟合还不够。需要额外的处理分配、混杂和稳定性假设；一个拟合良好的线性模型仍可能只是在训练分布中利用相关特征。

## 失效模式

**把线性理解成只能拟合直线。** 只要对参数线性，平方项、交互项和固定基函数都可以纳入模型；真正要检查的是特征空间是否覆盖目标结构。

**看到正规方程就直接求逆。** 公式用于推导，数值实现应根据条件数选择 QR、SVD、岭回归或迭代方法。

**把系数当成因果效应。** 无混杂、无反向因果和测量正确等条件不会因为 OLS 算出一个数字就自动成立。

**忽略截距和单位。** 去掉截距会改变可行的预测子空间，改变单位会按比例改变系数；跨特征比较原始系数通常没有意义。

**把低训练 MSE 当成模型正确。** 高次特征、泄漏和离群点都可能降低训练误差。需要独立验证、残差诊断和时间外评估。

**把高杠杆点直接删掉。** 高杠杆表示输入位置特殊，不等于数据错误。要分别检查输入是否真实、标签是否可靠以及该区域是否属于部署分布。

**把共线性下的单个系数当成稳定结论。** 整体预测可以稳定，单个系数仍然会在特征之间重新分配；应报告组合效应、区间和正则化敏感性。

**用测试集选择特征或正则化。** 反复比较测试 MSE 会让最终分数参与学习流程。应在测试前冻结特征、编码、$\lambda$ 和停止规则。

## 相关词条

- [监督学习](../learning-framework/supervised-learning/)：说明回归任务如何由标签、损失和条件 Bayes 规则定义。
- [正交投影](../linear-algebra/orthogonal-projections/)：从最近点和正交残差解释最小二乘的几何。
- [最小二乘即投影](../linear-models/least-squares-as-projection/)：进一步展开回归、投影矩阵与误差分解的对应。
- [伪逆](../linear-algebra/pseudoinverse/)：处理秩亏、欠定和最小范数最小二乘解。
- [双下降](../learning-framework/double-descent/)：讨论参数数跨过样本约束后插值风险的非单调变化。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：比较特征扩张、早停和显式惩罚对泛化的影响。
- [训练、验证与测试集](../learning-framework/train-validation-test/)：规定特征、模型和正则化选择的评估边界。
