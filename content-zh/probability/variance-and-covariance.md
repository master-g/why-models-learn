---
title: "方差与协方差：量化波动和共同变化"
tags: ["why-models-learn"]
---

**方差**衡量随机变量围绕均值的平方偏离，**协方差**衡量两个变量是否倾向于同向或反向偏离。对 X、Y：

$$
\operatorname{Var}(X)
=\mathbb E\bigl[(X-\mathbb E[X])^2\bigr]
$$

$$
\operatorname{Cov}(X,Y)
=\mathbb E\bigl[(X-\mathbb E[X])(Y-\mathbb E[Y])\bigr]
$$

方差总是非负，协方差可以为正、为负或为零。方差的平方根是标准差，它与原变量同单位；协方差的单位则是 X 与 Y 单位的乘积。本篇从平方偏离的定义出发，推导计算公式、仿射变换、和的方差、相关系数和全方差定律，再说明样本估计、协方差矩阵及其在机器学习中的用途。

## 方差：平均平方偏离

令

$$
\mu=\mathbb E[X]
$$

方差定义为

$$
\operatorname{Var}(X)
=\mathbb E[(X-\mu)^2]
$$

离散变量写成

$$
\operatorname{Var}(X)
=\sum_x(x-\mu)^2p_X(x)
$$

连续变量写成

$$
\operatorname{Var}(X)
=\int_{-\infty}^{\infty}(x-\mu)^2f_X(x)\,dx
$$

先减去均值，是为了把「偏离中心」和「整体位置」分开；再平方，是为了让左右偏离都贡献非负的距离，并让较大的偏离受到更大惩罚。最后按概率加权，得到整个分布的平均平方距离。

公平六面骰的均值是

$$
\mu=\mathbb E[X]=\frac72
$$

所以

$$
\begin{aligned}
\operatorname{Var}(X)
&=\frac16\sum_{k=1}^6\left(k-\frac72\right)^2\\
&=\frac16\left(
\frac{25}{4}+\frac94+\frac14+\frac14+\frac94+\frac{25}{4}
\right)\\
&=\frac{35}{12}
\end{aligned}
$$

偏离均值较远的 1 和 6，各自贡献 $25/4$；靠近均值的 3 和 4，各自只贡献 $1/4$。方差的单位是「点数平方」，还不是可以直接和点数比较的尺度。

方差为零的条件很严格：

$$
\operatorname{Var}(X)=0
\quad\Longleftrightarrow\quad
P(X=\mu)=1
$$

因为平方项是非负数，非零概率的任何偏离都会让平均值变为正。反过来，如果 X 几乎必然等于一个常数，所有偏离都是 0，方差就是 0。

## 计算公式：$\mathbb E[X^2]$ 减去均值平方

展开平方：

$$
\begin{aligned}
\operatorname{Var}(X)
&=\mathbb E[(X-\mu)^2]\\
&=\mathbb E[X^2-2\mu X+\mu^2]\\
&=\mathbb E[X^2]-2\mu\mathbb E[X]+\mu^2\\
&=\mathbb E[X^2]-2\mu^2+\mu^2\\
&=\mathbb E[X^2]-\mu^2
\end{aligned}
$$

因此

$$
\boxed{\operatorname{Var}(X)=\mathbb E[X^2]-\bigl(\mathbb E[X]\bigr)^2}
$$

骰子平方的期望已经可以直接计算：

$$
\mathbb E[X^2]
=\frac{1^2+2^2+3^2+4^2+5^2+6^2}{6}
=\frac{91}{6}
$$

代回

$$
\operatorname{Var}(X)
=\frac{91}{6}-\left(\frac72\right)^2
=\frac{182-147}{12}
=\frac{35}{12}
$$

这个公式在计算上更方便，但解释上不应忘记它来自「围绕均值的平方距离」。如果先把 $X^2$ 的平均和均值平方分别算出来再相减，两个大数接近时可能发生数值消减；统计软件通常会使用更稳定的中心化算法。

## 标准差：回到原来的单位

标准差定义为方差的非负平方根：

$$
\sigma_X=\sqrt{\operatorname{Var}(X)}
$$

骰子的标准差是

$$
\sigma_X=\sqrt{\frac{35}{12}}\approx1.708
$$

方差是平方单位，适合代数展开；标准差和 X 同单位，适合解释「通常偏离均值多远」。这两个量包含同一份波动信息，不能把方差 35/12 直接说成「平均偏离 35/12 个点」。

标准差不等于平均绝对偏差：

$$
\mathbb E[|X-\mu|]
\neq
\sqrt{\mathbb E[(X-\mu)^2]}
$$

前者使用一次方，后者使用平方后再开根。平方会更重地强调少数极端值，因此同一分布的标准差通常会受到尾部样本的明显影响。

## 仿射变换：平移不改变方差，缩放平方改变方差

令

$$
Y=aX+b
$$

由期望的线性性，

$$
\mathbb E[Y]=a\mathbb E[X]+b=a\mu+b
$$

中心化后

$$
\begin{aligned}
Y-\mathbb E[Y]
&=aX+b-(a\mu+b)\\
&=a(X-\mu)
\end{aligned}
$$

所以

$$
\begin{aligned}
\operatorname{Var}(Y)
&=\mathbb E\left[(a(X-\mu))^2\right]\\
&=a^2\operatorname{Var}(X)
\end{aligned}
$$

平移 b 不影响偏离均值的距离；缩放 a 会把每个偏离乘以 a，平方距离因此乘以 $a^2$。例如若骰子点数换算成摄氏温度

$$
T=10X-20
$$

则

$$
\mathbb E[T]=10\cdot3.5-20=15,\qquad
\operatorname{Var}(T)=10^2\cdot\frac{35}{12}=\frac{875}{3}
$$

标准差变为原来的 10 倍，而方差变为原来的 100 倍。若 $a<0$，分布方向翻转，但方差仍使用 $a^2$，不会因为左右反向而变成负数。

## 协方差：两个变量是否一起偏离

令

$$
\mu_X=\mathbb E[X],\qquad
\mu_Y=\mathbb E[Y]
$$

协方差定义为

$$
\operatorname{Cov}(X,Y)
=\mathbb E[(X-\mu_X)(Y-\mu_Y)]
$$

离散联合分布写成

$$
\operatorname{Cov}(X,Y)
=\sum_{x,y}(x-\mu_X)(y-\mu_Y)p_{X,Y}(x,y)
$$

每个样本点贡献两个偏离的乘积：

- X、Y 同时高于均值或同时低于均值，乘积为正；
- 一个高于均值、另一个低于均值，乘积为负；
- 任一变量正好在均值处，乘积为 0。

把定义展开，可以得到计算公式：

$$
\begin{aligned}
\operatorname{Cov}(X,Y)
&=\mathbb E[XY-\mu_YX-\mu_XY+\mu_X\mu_Y]\\
&=\mathbb E[XY]-\mu_Y\mathbb E[X]-\mu_X\mathbb E[Y]+\mu_X\mu_Y\\
&=\mathbb E[XY]-\mu_X\mu_Y
\end{aligned}
$$

因此

$$
\boxed{\operatorname{Cov}(X,Y)
=\mathbb E[XY]-\mathbb E[X]\mathbb E[Y]}
$$

交换两个变量不会改变乘积：

$$
\operatorname{Cov}(X,Y)=\operatorname{Cov}(Y,X)
$$

把 X 和自己配对就得到方差：

$$
\operatorname{Cov}(X,X)
=\mathbb E[(X-\mu_X)^2]
=\operatorname{Var}(X)
$$

协方差的正负只是共同线性变化的方向，不代表它一定大或小。还要结合两个变量各自的尺度，才能进行跨数据集比较。

![围绕均值的方差与两个变量共同偏离产生的协方差](/assets/probability/svg/variance-and-covariance.1.svg)

## 独立性、零协方差和相关性

若 X、Y 独立且期望存在，则乘积期望分解：

$$
\mathbb E[XY]=\mathbb E[X]\mathbb E[Y]
$$

代入协方差计算公式：

$$
\operatorname{Cov}(X,Y)=0
$$

所以独立是协方差为零的充分条件。但反方向一般不成立。令 X 均匀取 $-1,0,1$，Y=X²，则

$$
\operatorname{Cov}(X,Y)=0
$$

但 Y 完全由 X 决定，二者不独立。协方差只检查一个乘积的平均；独立性要求所有事件组合都能分解。

为了消除单位影响，定义相关系数

$$
\rho_{X,Y}
=\frac{\operatorname{Cov}(X,Y)}
{\sigma_X\sigma_Y}
$$

要求 $\sigma_X>0$ 且 $\sigma_Y>0$。由 Cauchy–Schwarz 不等式：

$$
-1\leq\rho_{X,Y}\leq1
$$

$\rho>0$ 表示线性同向趋势，$\rho<0$ 表示线性反向趋势，$\rho=0$ 表示没有线性相关。它仍然不能检测所有非线性关系；X 和 X² 的例子就是相关系数为 0 但变量有确定关系。

当 $|\rho|=1$ 时，平方可积的非退化变量之间存在几乎处处的仿射关系：

$$
Y=aX+b
$$

其中 $a>0$ 对应 $\rho=1$，$a<0$ 对应 $\rho=-1$。这比「散点图看起来接近一条线」更强，因为它是 Cauchy–Schwarz 取等的条件。

## 和与差的方差：协方差决定额外项

先中心化：

$$
(X+Y)-(\mu_X+\mu_Y)
=(X-\mu_X)+(Y-\mu_Y)
$$

平方并取期望：

$$
\begin{aligned}
\operatorname{Var}(X+Y)
&=\mathbb E[(X-\mu_X)^2]
 +\mathbb E[(Y-\mu_Y)^2]\\
&\quad+2\mathbb E[(X-\mu_X)(Y-\mu_Y)]\\
&=\operatorname{Var}(X)+\operatorname{Var}(Y)
 +2\operatorname{Cov}(X,Y)
\end{aligned}
$$

同理

$$
\operatorname{Var}(X-Y)
=\operatorname{Var}(X)+\operatorname{Var}(Y)
-2\operatorname{Cov}(X,Y)
$$

正协方差会让和的波动变大，因为两个变量更常同时偏向同一侧；负协方差会抵消一部分波动。只有在协方差为 0 时，和的方差才是两个方差的简单相加。

若 X、Y 是两个独立公平骰子：

$$
\operatorname{Var}(X+Y)
=\frac{35}{12}+\frac{35}{12}
=\frac{35}{6}
$$

如果两个骰子每次都掷出相同点数，即 Y=X，则

$$
\operatorname{Var}(X+Y)
=\operatorname{Var}(2X)
=4\operatorname{Var}(X)
=\frac{35}{3}
$$

共同变化把独立情形的和方差从 $35/6$ 提高到 $35/3$。若 Y=-X，则和恒为 0，负协方差把两个方差完全抵消。

对有限个变量：

$$
\operatorname{Var}\left(\sum_{i=1}^nX_i\right)
=\sum_{i=1}^n\operatorname{Var}(X_i)
 +2\sum_{1\leq i<j\leq n}\operatorname{Cov}(X_i,X_j)
$$

这条式子说明，计算总噪声时不能只把每个分量的方差相加，还要检查分量之间是否共同漂移。

## 全方差定律：组内波动加组间波动

给定 Y 后，X 有条件均值和条件方差：

$$
m(Y)=\mathbb E[X\mid Y],\qquad
v(Y)=\operatorname{Var}(X\mid Y)
$$

全方差定律是

$$
\boxed{
\operatorname{Var}(X)
=\mathbb E[\operatorname{Var}(X\mid Y)]
 +\operatorname{Var}(\mathbb E[X\mid Y])
}
$$

它把总体波动拆成两部分：每个条件组内部的平均波动，以及各组条件均值之间的波动。

推导从

$$
X-\mathbb E[X]
=\bigl(X-\mathbb E[X\mid Y]\bigr)
 +\bigl(\mathbb E[X\mid Y]-\mathbb E[X]\bigr)
$$

开始。平方后有三个部分：

$$
\begin{aligned}
(X-\mathbb E[X])^2
&=(X-\mathbb E[X\mid Y])^2\\
&\quad+(\mathbb E[X\mid Y]-\mathbb E[X])^2\\
&\quad+2(X-\mathbb E[X\mid Y])
(\mathbb E[X\mid Y]-\mathbb E[X])
\end{aligned}
$$

对两边取期望。第一项用全期望定律变成条件方差的期望；第二项正是条件均值的方差；交叉项为 0，因为给定 Y 后

$$
\mathbb E[X-\mathbb E[X\mid Y]\mid Y]=0
$$

所以得到上面的分解。

例子中，Y 以相等概率取 0 或 2，给定 Y 后令

$$
X=Y+\varepsilon,\qquad
P(\varepsilon=-1)=P(\varepsilon=1)=\frac12
$$

并令 $\varepsilon$ 与 Y 独立。每个组内 X 只在两个相邻值之间变化：

$$
\operatorname{Var}(X\mid Y)=1
$$

所以组内方差的平均是 1。条件均值为

$$
\mathbb E[X\mid Y]=Y
$$

而 Y 在 0 和 2 之间等概率变化，均值为 1、方差为 1。因此

$$
\operatorname{Var}(X)=1+1=2
$$

直接列出 X 的四个等概率结果 $-1,1,1,3$，围绕均值 1 的平方偏离平均值也确实是

$$
\frac{4+0+0+4}{4}=2
$$

全方差定律在分层抽样、混合人群、不同设备的测量误差和模型不确定性分析中都提供了相同的记账方式。

## 样本方差：从有限数据估计总体方差

总体随机变量 X 的均值 μ 已知时，n 个观测 $x_1,\ldots,x_n$ 的经验二阶中心矩是

$$
\frac1n\sum_{i=1}^n(x_i-\mu)^2
$$

实际通常用样本均值

$$
\bar x=\frac1n\sum_{i=1}^nx_i
$$

替代未知的 μ。直接除以 n 的版本叫有偏的样本方差：

$$
s_n^2=\frac1n\sum_{i=1}^n(x_i-\bar x)^2
$$

若目标是估计总体方差，而且观测是来自同一分布的独立样本，常用 Bessel 修正：

$$
s^2=\frac1{n-1}\sum_{i=1}^n(x_i-\bar x)^2
$$

n−1 而不是 n 的原因是，样本均值已经从数据中估计了一个中心约束：

$$
\sum_{i=1}^n(x_i-\bar x)=0
$$

只有 n−1 个偏离可以自由决定最后一个偏离。更严格地，若 X_i i.i.d. 且总体方差有限，则

$$
\mathbb E[s^2]=\operatorname{Var}(X)
$$

而除以 n 的版本平均会低估总体方差：

$$
\mathbb E[s_n^2]=\frac{n-1}{n}\operatorname{Var}(X)
$$

数据 $1,3,5$ 的样本均值是 3，平方偏离为 4、0、4。因此

$$
s_n^2=\frac{4+0+4}{3}=\frac83,\qquad
s^2=\frac{4+0+4}{2}=4
$$

二者不是谁「永远正确」的问题，而是分母对应不同目标：描述这组三个数据本身，还是用它们估计产生数据的总体方差。

## 协方差矩阵与机器学习

对随机向量

$$
\mathbf X=(X_1,\ldots,X_d)^{\mathsf T}
$$

协方差矩阵定义为

$$
\Sigma_{ij}
=\operatorname{Cov}(X_i,X_j)
$$

写成矩阵形式：

$$
\Sigma
=\mathbb E\left[
(\mathbf X-\boldsymbol\mu)
(\mathbf X-\boldsymbol\mu)^{\mathsf T}
\right]
$$

对角线是各坐标方差，非对角线是坐标之间的协方差。对任意向量 a：

$$
\mathbf a^{\mathsf T}\Sigma\mathbf a
=\operatorname{Var}(\mathbf a^{\mathsf T}\mathbf X)
\geq0
$$

因此 Σ 是正半定矩阵。协方差矩阵的完整结构、特征分解和白化会在下一篇专门展开；这里先看它如何进入模型。

### 特征尺度和标准化

若一列特征的单位从米改成厘米，数值乘以 100，方差会乘以 $100^2$，协方差也会按两个坐标的缩放因子变化。直接用平方损失或欧氏距离时，尺度更大的列可能占据主要贡献。

标准化通常写成

$$
Z=\frac{X-\mu_X}{\sigma_X}
$$

此时

$$
\mathbb E[Z]=0,\qquad
\operatorname{Var}(Z)=1
$$

这是把位置和尺度拆开，不会自动消除不同特征之间的协方差。要进一步去相关，需要使用协方差矩阵的结构，例如 PCA 或白化。

### 预测误差的波动

对预测误差

$$
E=Y-\hat Y
$$

期望描述平均偏差，方差描述误差在样本之间的波动。平均误差为 0 不代表每个样本都预测得好：正误差和负误差可能相互抵消，而平方误差和方差会保留波动大小。训练时报告均值、标准差和分位数，通常比只报告一个平均损失更能暴露长尾或分组差异。

对多个输出分量，误差协方差还能显示哪些输出会一起错。若只看每一列的方差而忽略协方差，可能漏掉系统性的联合偏移；这也是多元回归、概率预测和不确定性估计需要协方差矩阵的原因。

## 失效模式

**把方差当平均绝对偏差。**方差先平方，标准差再开根；它对大偏离的惩罚强于绝对偏差，不能按同一含义解读。

**忘记标准差和方差的单位不同。**方差的单位是原单位的平方，标准差才回到原变量单位。

**认为方差或协方差可以为负。**方差是平方的期望，永远非负；协方差的正负表示共同偏离方向，可以为负。

**把零协方差当成独立。**零协方差只排除了一个二阶线性关系，X 与 X² 的例子仍然有确定的非线性依赖。

**把独立样本的方差公式用于相关样本。**$\operatorname{Var}(\sum_iX_i)$ 还包含协方差项；时间序列、重复用户和同一视频帧可能使这些项很大。

**把协方差和相关系数混为一谈。**协方差受单位和尺度影响，相关系数经过标准差归一化，范围在 −1 到 1。

**把全方差定律只算组内或只算组间。**总体方差同时包含组内平均方差和条件均值的方差，漏掉任何一项都会低估波动。

**不说明样本方差的分母。**除以 n 描述当前数据的经验二阶中心矩；除以 n−1 是在 i.i.d. 条件下对总体方差的无偏估计。

**把均值为零当成误差没有波动。**正负误差可能抵消，必须同时查看方差、标准差或平方损失。

## 相关词条

- [期望](../probability/expectation/)：提供均值、乘积平均和条件平均的定义。
- [随机变量](../probability/random-variables/)：定义方差与协方差所作用的数值对象。
- [独立性](../probability/independence/)：说明独立如何推出零协方差，以及为什么反向不成立。
- [联合分布](../probability/joint-distributions/)：提供计算协方差所需的联合 PMF/PDF。
- [边缘与条件分布](../probability/marginal-and-conditional/)：定义条件期望和条件方差。
- [高斯分布](../probability/gaussian-distribution/)：展示方差、协方差和标准化在高斯模型中的作用。
- [协方差矩阵](../probability/covariance-matrix/)：展开随机向量的协方差矩阵、正半定性和白化。
- [大数定律](../probability/law-of-large-numbers/)：解释样本均值和样本矩如何逼近总体量。
