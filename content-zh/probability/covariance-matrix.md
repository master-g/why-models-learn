---
title: "协方差矩阵：用一个矩阵记录多维波动"
tags: ["why-models-learn"]
---

**协方差矩阵**把随机向量每一对坐标的协方差排成矩阵。对随机向量 $\mathbf X\in\mathbb R^d$，均值向量为 $\boldsymbol\mu$：

$$
\Sigma
=\mathbb E\left[
(\mathbf X-\boldsymbol\mu)
(\mathbf X-\boldsymbol\mu)^{\mathsf T}
\right]
$$

矩阵的第 i 行第 j 列是 $\operatorname{Cov}(X_i,X_j)$。对角线记录各坐标自己的方差，非对角线记录坐标一起偏离的方向和大小。协方差矩阵还满足对称、半正定等结构，因此可以用特征分解找出主方向，用白化把这些方向旋转并缩放成互不相关的单位方差坐标。本篇从外积展开开始，推导这些性质，再连接样本估计、PCA、白化和多元高斯模型。

## 从随机向量的外积得到矩阵

二维随机向量写成

$$
\mathbf X=
\begin{pmatrix}
X_1\\
X_2
\end{pmatrix},
\qquad
\boldsymbol\mu=
\begin{pmatrix}
\mu_1\\
\mu_2
\end{pmatrix}
$$

中心化向量是

$$
\mathbf X-\boldsymbol\mu
=
\begin{pmatrix}
X_1-\mu_1\\
X_2-\mu_2
\end{pmatrix}
$$

它与自己的转置做外积：

$$
(\mathbf X-\boldsymbol\mu)
(\mathbf X-\boldsymbol\mu)^{\mathsf T}
=
\begin{pmatrix}
(X_1-\mu_1)^2
&
(X_1-\mu_1)(X_2-\mu_2)
\\
(X_2-\mu_2)(X_1-\mu_1)
&
(X_2-\mu_2)^2
\end{pmatrix}
$$

逐元素取期望：

$$
\Sigma=
\begin{pmatrix}
\operatorname{Var}(X_1)&\operatorname{Cov}(X_1,X_2)\\
\operatorname{Cov}(X_2,X_1)&\operatorname{Var}(X_2)
\end{pmatrix}
$$

由于协方差对称，

$$
\operatorname{Cov}(X_1,X_2)
=\operatorname{Cov}(X_2,X_1)
$$

所以 $\Sigma=\Sigma^{\mathsf T}$。矩阵不是把两个方差随便放在对角线上；每个非对角元素都来自同一批样本中两个坐标的共同变化。

对 d 维向量，统一写作

$$
\Sigma_{ij}
=\operatorname{Cov}(X_i,X_j)
$$

其中

$$
\Sigma_{ii}=\operatorname{Var}(X_i)
$$

这让「每一列特征的波动」和「两列特征的配对关系」在同一对象中出现。

## 一个二维例子：从矩阵读出方差和协方差

考虑

$$
\Sigma=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
$$

它表示

$$
\operatorname{Var}(X_1)=3,\qquad
\operatorname{Var}(X_2)=3,\qquad
\operatorname{Cov}(X_1,X_2)=1
$$

两个坐标的波动尺度相同，协方差为正表示它们倾向于一起高于或一起低于各自均值。相关系数为

$$
\rho_{X_1,X_2}
=\frac{1}{\sqrt3\sqrt3}
=\frac13
$$

它是无量纲的；协方差 1 则带有两个坐标单位的乘积。

若两个坐标完全不共同变化，矩阵会是对角矩阵：

$$
\Sigma_{\text{uncorrelated}}
=
\begin{pmatrix}
3&0\\
0&3
\end{pmatrix}
$$

对角矩阵表示零协方差，不自动表示概率独立。只有在额外的联合高斯条件下，零协方差才会推出独立。

协方差矩阵的非对角元素可以为负，只要整体仍满足半正定。例如

$$
\begin{pmatrix}
3&-1\\
-1&3
\end{pmatrix}
$$

表示两个坐标倾向于反向变化。判断合法性不能逐个检查矩阵元素是否非负，而要检查所有方向上的二次型。

![协方差椭圆的主方向与白化后的单位圆](/assets/probability/svg/covariance-matrix.1.svg)

## 半正定：每个线性投影的方差都不能为负

取任意固定向量 $\mathbf a\in\mathbb R^d$。投影随机变量是

$$
Z=\mathbf a^{\mathsf T}\mathbf X
$$

它的均值为

$$
\mathbb E[Z]
=\mathbf a^{\mathsf T}\boldsymbol\mu
$$

中心化投影为

$$
Z-\mathbb E[Z]
=\mathbf a^{\mathsf T}(\mathbf X-\boldsymbol\mu)
$$

因此

$$
\begin{aligned}
\operatorname{Var}(Z)
&=\mathbb E\left[
\mathbf a^{\mathsf T}(\mathbf X-\boldsymbol\mu)
(\mathbf X-\boldsymbol\mu)^{\mathsf T}\mathbf a
\right]\\
&=\mathbf a^{\mathsf T}\Sigma\mathbf a
\end{aligned}
$$

方差不能为负，所以

$$
\mathbf a^{\mathsf T}\Sigma\mathbf a\geq0
\qquad\text{对所有 }\mathbf a
$$

这就是半正定性。协方差矩阵必须对称半正定；若对某个非零方向方差为 0，说明 $\mathbf a^{\mathsf T}\mathbf X$ 几乎必然是常数，数据在该方向上没有随机变化。

二维例子中取 $\mathbf a=(1,1)^{\mathsf T}$：

$$
\begin{aligned}
\operatorname{Var}(X_1+X_2)
&=
\begin{pmatrix}1&1\end{pmatrix}
\begin{pmatrix}3&1\\1&3\end{pmatrix}
\begin{pmatrix}1\\1\end{pmatrix}\\
&=8
\end{aligned}
$$

取 $\mathbf a=(1,-1)^{\mathsf T}$：

$$
\operatorname{Var}(X_1-X_2)=4
$$

正协方差让和的波动增大，让差的波动减小。矩阵中的四个数字共同决定每个投影方向的方差，不能只看对角线。

## 正定、奇异和可逆的区别

若对所有非零 $\mathbf a$ 都有

$$
\mathbf a^{\mathsf T}\Sigma\mathbf a>0
$$

则 Σ 是正定的。此时没有任何非零线性组合是常数，所有特征值都为正，矩阵可逆。

若只满足大于等于 0，Σ 可能是奇异的。例如令

$$
X_2=2X_1
$$

则

$$
\begin{pmatrix}
1&-2
\end{pmatrix}
\begin{pmatrix}
X_1\\
X_2
\end{pmatrix}
=0
$$

这个方向没有随机变化，协方差矩阵至少有一个零特征值。二元协方差矩阵写成

$$
\Sigma=
\begin{pmatrix}
\sigma_1^2&\rho\sigma_1\sigma_2\\
\rho\sigma_1\sigma_2&\sigma_2^2
\end{pmatrix}
$$

其行列式为

$$
|\Sigma|
=\sigma_1^2\sigma_2^2(1-\rho^2)
$$

所以当 $|\rho|=1$ 时行列式为 0，两个变量存在完美的仿射关系，二维随机云退化到一条直线。

正定性是多元高斯密度使用 $\Sigma^{-1}$ 和 $|\Sigma|^{-1/2}$ 的条件。奇异高斯仍可以作为低维子空间上的分布讨论，但不能直接套用普通的全维密度公式。

## 特征分解：找到方差最大的方向

实对称协方差矩阵可以正交特征分解：

$$
\Sigma=Q\Lambda Q^{\mathsf T}
$$

其中 Q 的列是正交归一特征向量，Λ 是非负特征值组成的对角矩阵。对单位特征向量 $\mathbf q_i$：

$$
\operatorname{Var}(\mathbf q_i^{\mathsf T}\mathbf X)
=\mathbf q_i^{\mathsf T}\Sigma\mathbf q_i
=\lambda_i
$$

特征值就是沿对应主方向的方差。对

$$
\Sigma=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
$$

取

$$
\mathbf q_1=\frac1{\sqrt2}
\begin{pmatrix}1\\1\end{pmatrix},
\qquad
\mathbf q_2=\frac1{\sqrt2}
\begin{pmatrix}1\\-1\end{pmatrix}
$$

则

$$
\Sigma\mathbf q_1=4\mathbf q_1,
\qquad
\Sigma\mathbf q_2=2\mathbf q_2
$$

所以沿 $(1,1)$ 方向的方差是 4，沿 $(1,-1)$ 方向的方差是 2。原坐标中的正协方差，经过旋转后变成两个互不相关的主坐标。

所有特征值的和是迹：

$$
\operatorname{tr}(\Sigma)
=\sum_{i=1}^d\lambda_i
=\sum_{i=1}^d\operatorname{Var}(X_i)
$$

它是各坐标方差的总和，也等于中心化向量平方长度的期望：

$$
\operatorname{tr}(\Sigma)
=\mathbb E[\|\mathbf X-\boldsymbol\mu\|_2^2]
$$

PCA 按特征值从大到小排序，保留前 k 个特征向量，相当于保留总方差中变化最大的 k 个正交方向。它不是先看原坐标的某几列，而是在所有单位方向中寻找投影方差最大的方向。

## 相关矩阵：去掉各坐标的单位

令

$$
D=\operatorname{diag}(\sigma_1,\ldots,\sigma_d)
$$

其中 $\sigma_i=\sqrt{\Sigma_{ii}}$。若所有标准差都非零，相关矩阵是

$$
R=D^{-1}\Sigma D^{-1}
$$

它的元素为

$$
R_{ij}
=\frac{\Sigma_{ij}}{\sigma_i\sigma_j}
=\rho_{X_i,X_j}
$$

对角线恒为 1，非对角线位于 −1 与 1 之间。相关矩阵适合比较不同单位的特征；协方差矩阵保留了原始尺度，适合计算实际投影方差和噪声量。

单位变换会改变 Σ，却不改变 R。若 $Y_i=a_iX_i+b_i$ 且所有 $a_i>0$，相关系数保持不变；若某个 $a_i<0$，对应相关系数符号会翻转。

## 白化：旋转后把每个方向缩放到单位方差

先中心化，再使用特征分解：

$$
\mathbf z
=\Lambda^{-1/2}Q^{\mathsf T}
(\mathbf X-\boldsymbol\mu)
$$

其中

$$
\Lambda^{-1/2}
=\operatorname{diag}
\left(\frac1{\sqrt{\lambda_1}},\ldots,
\frac1{\sqrt{\lambda_d}}\right)
$$

当所有特征值都为正时：

$$
\begin{aligned}
\operatorname{Cov}(\mathbf z)
&=\Lambda^{-1/2}Q^{\mathsf T}
\Sigma Q\Lambda^{-1/2}\\
&=\Lambda^{-1/2}Q^{\mathsf T}
Q\Lambda Q^{\mathsf T}Q\Lambda^{-1/2}\\
&=I
\end{aligned}
$$

第一步 $Q^{\mathsf T}$ 把坐标转到主方向，第二步 $\Lambda^{-1/2}$ 把第 i 个方向除以自己的标准差。白化后的坐标均值为 0、协方差为单位矩阵，因此各坐标不相关且方差为 1。

若存在零特征值，不能直接取倒数。常见的稳定版本使用

$$
\mathbf z_\varepsilon
=(\Lambda+\varepsilon I)^{-1/2}
Q^{\mathsf T}(\mathbf X-\boldsymbol\mu)
$$

这会牺牲一点「严格单位方差」，换取不放大零空间或近零空间中的噪声。PCA 截断则直接删除小特征值方向；白化和降维的目标不同，不能把两步混成一个操作。

## 样本协方差：从数据矩阵直接计算

把 n 个 d 维观测按行堆成数据矩阵：

$$
X=
\begin{pmatrix}
\mathbf x_1^{\mathsf T}\\
\vdots\\
\mathbf x_n^{\mathsf T}
\end{pmatrix}
$$

每列的样本均值组成 $\bar{\mathbf x}$。中心化矩阵是

$$
X_c=X-\mathbf1\bar{\mathbf x}^{\mathsf T}
$$

常见的总体式经验协方差和无偏样本协方差分别为

$$
\widehat\Sigma_n=\frac1nX_c^{\mathsf T}X_c
$$

$$
\widehat\Sigma=\frac1{n-1}X_c^{\mathsf T}X_c
$$

后一式在 i.i.d. 抽样和总体协方差存在时对 Σ 无偏。分母选择和标量样本方差相同，不能只因为矩阵写法变了就忽略 Bessel 修正。

取三个二维观测：

$$
\mathbf x_1=
\begin{pmatrix}1\\2\end{pmatrix},
\qquad
\mathbf x_2=
\begin{pmatrix}3\\4\end{pmatrix},
\qquad
\mathbf x_3=
\begin{pmatrix}5\\6\end{pmatrix}
$$

均值为

$$
\bar{\mathbf x}
=
\begin{pmatrix}3\\4\end{pmatrix}
$$

中心化矩阵按行是

$$
X_c=
\begin{pmatrix}
-2&-2\\
0&0\\
2&2
\end{pmatrix}
$$

所以

$$
X_c^{\mathsf T}X_c
=
\begin{pmatrix}
8&8\\
8&8
\end{pmatrix}
$$

无偏样本协方差为

$$
\widehat\Sigma
=\frac12
\begin{pmatrix}
8&8\\
8&8
\end{pmatrix}
=
\begin{pmatrix}
4&4\\
4&4
\end{pmatrix}
$$

两个坐标在样本中完全同向，矩阵的特征值是 8 和 0。样本协方差的秩受到中心化约束限制：

$$
\operatorname{rank}(X_c^{\mathsf T}X_c)
\leq\min(d,n-1)
$$

当特征维数 d 大于样本数 n 时，样本协方差必然奇异；即使 d 不大，只要数据落在较低维子空间中也会奇异。直接求逆前必须检查特征值、秩和条件数。

## 多元高斯和二次型

若

$$
\mathbf X\sim\mathcal N(\boldsymbol\mu,\Sigma)
$$

且 Σ 正定，密度为

$$
f(\mathbf x)
=\frac1{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(
-\frac12
(\mathbf x-\boldsymbol\mu)^{\mathsf T}
\Sigma^{-1}
(\mathbf x-\boldsymbol\mu)
\right)
$$

二次型

$$
(\mathbf x-\boldsymbol\mu)^{\mathsf T}
\Sigma^{-1}
(\mathbf x-\boldsymbol\mu)
$$

按协方差的方向和尺度测量偏离，也就是 Mahalanobis 距离的平方。沿方差大的方向，同样的欧氏距离会被惩罚得更轻；沿方差小的方向，偏离会被惩罚得更重。

负对数密度忽略常数后是

$$
\frac12
(\mathbf x-\boldsymbol\mu)^{\mathsf T}
\Sigma^{-1}
(\mathbf x-\boldsymbol\mu)
+
\frac12\log|\Sigma|
$$

第一项衡量当前样本相对椭圆形等密度面的距离，第二项是归一化体积项。只保留二次型而丢掉 $\log|\Sigma|$，模型可能通过把 Σ 变得很小来虚假地提高训练点密度；概率模型必须同时保留尺度的归一化项。

## 机器学习中的协方差结构

### PCA 与降维

中心化数据的样本协方差特征值表示各主方向的经验方差。保留最大的 k 个特征向量，可以在平方重构误差意义下得到最佳 k 维线性子空间；被丢弃的特征值之和是丢掉的总方差。

如果第一主方向的特征值远大于其余方向，数据的变化主要集中在一条狭长的子空间。若特征值相近，任意旋转后的方向可能都有相近方差，主方向对样本扰动会更敏感。

### 标准化和白化

只对每一列做 z 分数标准化，会把协方差矩阵的对角线变成 1，但非对角线仍然保留：

$$
\operatorname{diag}(R)=(1,\ldots,1)
$$

白化进一步旋转并缩放，目标是完整的单位矩阵。它可以改善某些优化问题的尺度，但会放大近零特征值方向的噪声，实际实现需要截断或 $\varepsilon$ 正则。

### 高斯不确定性

概率回归、卡尔曼滤波和许多生成模型用均值向量与协方差矩阵描述多维不确定性。只预测每个坐标的标准差而忽略协方差，会把椭圆不确定性错误地当成轴对齐的圆角矩形；两个输出可能一起偏高或一起偏低，这种结构需要非对角元素表达。

协方差矩阵参数数量是

$$
\frac{d(d+1)}2
$$

因为它是对称矩阵。d 较大时，直接学习所有协方差元素会有参数量和正定性约束问题，工程上常使用对角协方差、低秩加对角结构或 Cholesky 因子化。

## 失效模式

**把协方差矩阵的非对角元素都当成必须非负。**单个协方差可以为负；合法性由所有二次型非负决定，不由逐元素符号决定。

**把对角线当标准差。**$\Sigma_{ii}$ 是方差，标准差是 $\sqrt{\Sigma_{ii}}$；直接把对角线当尺度会差一个平方根。

**把协方差矩阵当任意对称矩阵。**对称还不够，样本误差或手工填数可能让矩阵出现负特征值，无法作为方差矩阵。

**在奇异矩阵上直接求逆。**线性依赖、样本太少或近零特征值都会让逆不稳定。先看秩、特征值和条件数，再决定截断或加正则。

**混用 n 与 n−1。**除以 n 是经验二阶中心矩，除以 n−1 是 i.i.d. 条件下的无偏样本协方差；报告结果时必须说明分母。

**把标准化当成白化。**z 分数只调整每列的均值和方差，不会消除非对角协方差；白化还需要旋转到主坐标。

**把 PCA 的降维和白化的缩放混为一谈。**PCA 可能删除小方差方向，白化则会尝试把保留方向缩放到单位方差；小特征值在白化中反而是数值风险。

**忘记多元高斯的 $\log|\Sigma|$。**二次型和归一化体积是同一个密度公式的两部分，省略后模型可能偏爱退化协方差。

**样本数不足仍估计全协方差。**d 维数据至少需要足够多的独立观测覆盖各方向；当 $n-1<d$ 时，中心化样本协方差必然秩亏。

## 相关词条

- [方差与协方差](../probability/variance-and-covariance/)：提供矩阵元素的标量定义、全方差和相关系数。
- [期望](../probability/expectation/)：提供均值、外积期望和二次型期望。
- [随机变量](../probability/random-variables/)：定义随机向量的坐标和观测值。
- [联合分布](../probability/joint-distributions/)：描述多维坐标的联合概率结构。
- [特征分解](../linear-algebra/eigendecomposition/)：计算协方差矩阵的主方向和特征值。
- [谱定理](../linear-algebra/spectral-theorem/)：保证实对称协方差矩阵的正交特征分解。
- [迹](../linear-algebra/trace/)：把特征值和各坐标方差加成总方差。
- [SVD](../linear-algebra/svd/)：从中心化数据矩阵稳定计算主方向。
- [高斯分布](../probability/gaussian-distribution/)：把协方差矩阵放进多元高斯密度。
- [二次型](../linear-algebra/quadratic-forms/)：解释 Mahalanobis 距离和高斯指数中的矩阵表达。
