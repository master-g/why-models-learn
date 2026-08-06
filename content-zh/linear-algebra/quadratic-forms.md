---
title: "二次型:矩阵如何给方向打分"
tags: ["why-models-learn"]
---

**二次型**（quadratic form）是把同一个向量放在矩阵两侧的标量函数：

$$
q_A(\mathbf{x})=\mathbf{x}^{\mathsf T}A\mathbf{x}
$$

它把一个方向和一个长度组合成一个数：这个数可以表示能量、曲率、方差或距离的平方。二次型不一定是范数，也不一定非负；它的正负和等值线形状由矩阵的对称部分及其特征值决定。本篇先把这个表达式展开，再用谱坐标分类椭圆、抛物面和鞍面，最后连接 Rayleigh 商、PCA、Hessian 和 Mahalanobis 距离。

## 从二维展开开始

取对称矩阵

$$
A=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix},
\qquad
\mathbf{x}=
\begin{pmatrix}
x\\
y
\end{pmatrix}
$$

那么

$$
\begin{aligned}
q_A(\mathbf{x})
&=
\begin{pmatrix}x&y\end{pmatrix}
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
\begin{pmatrix}x\\y\end{pmatrix}\\
&=3x^2+xy+xy+3y^2\\
&=3x^2+2xy+3y^2
\end{aligned}
$$

交叉项会出现两次：左上到右下的乘法给出 $xy$，右下到左上的乘法再给出一个 $xy$。因此矩阵中一个对称的非对角元素 $a_{12}=a_{21}=1$，在多项式里对应的交叉项系数是 $2$。

算几个数字：

$$
q_A
\begin{pmatrix}1\\1\end{pmatrix}
=8,
\qquad
q_A
\begin{pmatrix}1\\-1\end{pmatrix}
=4,
\qquad
q_A
\begin{pmatrix}2\\1\end{pmatrix}
=19
$$

相同的矩阵可以给不同方向不同分数。第一、第二个向量长度平方都为 $2$，但沿 $(1,1)$ 的单位方向，二次型值为 $8/2=4$；沿 $(1,-1)$ 的单位方向，二次型值为 $4/2=2$。

## 只有对称部分真正参与

任意方阵 $A$ 都可以拆成

$$
A=S+K,
\qquad
S=\frac{A+A^{\mathsf T}}2,
\qquad
K=\frac{A-A^{\mathsf T}}2
$$

其中 $S^{\mathsf T}=S$，$K^{\mathsf T}=-K$。对反对称部分：

$$
\begin{aligned}
\mathbf{x}^{\mathsf T}K\mathbf{x}
&=\left(\mathbf{x}^{\mathsf T}K\mathbf{x}\right)^{\mathsf T}\\
&=\mathbf{x}^{\mathsf T}K^{\mathsf T}\mathbf{x}\\
&=-\mathbf{x}^{\mathsf T}K\mathbf{x}
\end{aligned}
$$

这个标量只能等于 $0$，所以

$$
\mathbf{x}^{\mathsf T}A\mathbf{x}
=\mathbf{x}^{\mathsf T}S\mathbf{x}
$$

例如

$$
K=
\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix}
$$

不是零矩阵，但对所有 $\mathbf{x}$ 都有 $\mathbf{x}^{\mathsf T}K\mathbf{x}=0$。因此研究二次型时，可以直接把 $A$ 换成 $S$；矩阵的反对称部分会影响双线性表达式 $\mathbf{x}^{\mathsf T}A\mathbf{y}$，但不会影响把 $\mathbf{x}$ 放在两边的二次型。

若 $A$ 对称，二次型还决定了对应的对称双线性形式：

$$
\mathbf{x}^{\mathsf T}A\mathbf{y}
=\frac12\left(
q_A(\mathbf{x}+\mathbf{y})
-q_A(\mathbf{x})
-q_A(\mathbf{y})
\right)
$$

这叫极化恒等式。知道所有 $\mathbf{x}^{\mathsf T}A\mathbf{x}$ 的值，就能恢复对称矩阵 $A$；反对称部分仍然不可见。

## 谱坐标把二次型化成平方和

对称矩阵可以使用 [谱定理](../linear-algebra/spectral-theorem/)：

$$
A=Q\Lambda Q^{\mathsf T},
\qquad
Q^{\mathsf T}Q=I
$$

令

$$
\mathbf{z}=Q^{\mathsf T}\mathbf{x}
$$

这是把 $\mathbf{x}$ 读成正交特征坐标。代入二次型：

$$
\begin{aligned}
q_A(\mathbf{x})
&=\mathbf{x}^{\mathsf T}Q\Lambda Q^{\mathsf T}\mathbf{x}\\
&=\mathbf{z}^{\mathsf T}\Lambda\mathbf{z}\\
&=\sum_{i=1}^n\lambda_i z_i^2
\end{aligned}
$$

所有交叉项在特征坐标中消失，每个方向只留下一个 $\lambda_i z_i^2$。前面的例子有

$$
\mathbf{q}_1=\frac1{\sqrt2}
\begin{pmatrix}1\\1\end{pmatrix},
\qquad
\mathbf{q}_2=\frac1{\sqrt2}
\begin{pmatrix}1\\-1\end{pmatrix},
\qquad
\lambda_1=4,\quad\lambda_2=2
$$

所以

$$
q_A(\mathbf{x})=4z_1^2+2z_2^2
$$

原坐标里的交叉项没有消失，只是被正交换基吸收到了 $z_1$、$z_2$ 的定义里。

## 正定、半正定和不定

对称二次型的分类完全由特征值决定：

| 特征值 | 二次型分类 | $q_A(\mathbf{x})=1$ 的典型形状 |
| --- | --- | --- |
| 全部 $\lambda_i>0$ | 正定 | 椭圆或椭球 |
| 全部 $\lambda_i\ge0$ 且有零值 | 半正定 | 椭圆柱或平坦方向 |
| 有正有负 | 不定 | 双曲线或鞍面 |
| 全部 $\lambda_i<0$ | 负定 | $q=1$ 无实点，$q=-1$ 为椭圆 |

**正定**表示 $\mathbf{x}\ne\mathbf{0}$ 时 $q_A(\mathbf{x})>0$。**半正定**允许某些非零方向得到 $0$，但不允许负值。**不定**表示可以找到两个方向，一个让二次型为正，一个让它为负。

前面的 $A$ 有特征值 $4$、$2$，所以是正定。它的单位等值线

$$
4z_1^2+2z_2^2=1
$$

是椭圆：沿第一特征方向的半轴长度为 $1/2$，沿第二特征方向的半轴长度为 $1/\sqrt2$。特征值越大，同一单位二次型下允许的长度越短。

对比

$$
B=
\begin{pmatrix}
1&0\\
0&-1
\end{pmatrix},
\qquad
q_B(x,y)=x^2-y^2
$$

它在 $(1,0)$ 方向给 $1$，在 $(0,1)$ 方向给 $-1$，等值线 $x^2-y^2=1$ 是双曲线；原点附近的曲面有一条向上、一条向下的弯曲方向，所以叫鞍面。

![二次型的两种等值线：左侧正定二次型沿两个特征方向形成椭圆，右侧不定二次型的正等值线是双曲线](/assets/linear-algebra/svg/quadratic-forms.1.svg)

## Rayleigh 商：单位球面上寻找最大方向

二次型的绝对值会随向量长度平方增长。为了只比较方向，除以长度平方：

$$
R_A(\mathbf{x})
=\frac{\mathbf{x}^{\mathsf T}A\mathbf{x}}
{\mathbf{x}^{\mathsf T}\mathbf{x}},
\qquad
\mathbf{x}\ne\mathbf{0}
$$

这叫 Rayleigh 商。对称谱分解下，令 $\mathbf{z}=Q^{\mathsf T}\mathbf{x}$：

$$
R_A(\mathbf{x})
=\frac{\sum_i\lambda_i z_i^2}{\sum_i z_i^2}
=\sum_i\lambda_i
\frac{z_i^2}{\sum_j z_j^2}
$$

分数

$$
w_i=\frac{z_i^2}{\sum_jz_j^2}
$$

非负且总和为 $1$，所以 Rayleigh 商是特征值的加权平均：

$$
\lambda_{\min}
\le R_A(\mathbf{x})
\le\lambda_{\max}
$$

等号分别在 $\mathbf{x}$ 落入最小或最大特征值的特征空间时取得。对

$$
A=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
$$

有

$$
R_A
\begin{pmatrix}1\\1\end{pmatrix}
=4,
\qquad
R_A
\begin{pmatrix}1\\-1\end{pmatrix}
=2,
\qquad
R_A
\begin{pmatrix}2\\1\end{pmatrix}
=\frac{19}{5}
$$

$19/5=3.8$ 落在 $2$ 和 $4$ 之间，正是两个特征方向贡献的加权平均。

## 约束优化为什么会产生特征值

在单位球面上最大化二次型：

$$
\max_{\mathbf{x}}\ \mathbf{x}^{\mathsf T}A\mathbf{x}
\quad\text{subject to}\quad
\mathbf{x}^{\mathsf T}\mathbf{x}=1
$$

用拉格朗日乘子 $\mu$：

$$
\mathcal{L}(\mathbf{x},\mu)
=\mathbf{x}^{\mathsf T}A\mathbf{x}
-\mu(\mathbf{x}^{\mathsf T}\mathbf{x}-1)
$$

对称 $A$ 下，梯度为

$$
\nabla_{\mathbf{x}}\mathcal{L}
=2A\mathbf{x}-2\mu\mathbf{x}
=\mathbf{0}
$$

于是

$$
A\mathbf{x}=\mu\mathbf{x}
$$

约束优化的驻点就是单位特征向量，目标值就是对应特征值。最大值为 $\lambda_{\max}$，最小值为 $\lambda_{\min}$。PCA 选择最大方差方向，正是这个优化问题应用在协方差矩阵上的结果。

## 梯度和 Hessian：二次型是最简单的曲率模型

对一般方阵 $A$：

$$
\nabla_{\mathbf{x}}\left(\mathbf{x}^{\mathsf T}A\mathbf{x}\right)
=(A+A^{\mathsf T})\mathbf{x}
=2S\mathbf{x}
$$

若 $A$ 对称，则

$$
\nabla q_A(\mathbf{x})=2A\mathbf{x},
\qquad
\nabla^2q_A(\mathbf{x})=2A
$$

Hessian 不随 $\mathbf{x}$ 改变，所以二次型的曲率在每个位置都一样。更常见的带线性项形式是

$$
f(\mathbf{x})
=\frac12\mathbf{x}^{\mathsf T}H\mathbf{x}
-\mathbf{b}^{\mathsf T}\mathbf{x}
+c
$$

当 $H=H^{\mathsf T}$ 时：

$$
\nabla f(\mathbf{x})=H\mathbf{x}-\mathbf{b},
\qquad
\nabla^2f(\mathbf{x})=H
$$

如果 $H$ 正定，唯一驻点

$$
\mathbf{x}^*=H^{-1}\mathbf{b}
$$

是全局最小值；如果 $H$ 有正有负，驻点是鞍点；如果有零特征值，某些方向没有二阶曲率，需要更高阶项或额外约束。

## Mahalanobis 距离也是二次型

协方差矩阵 $\Sigma$ 正定时，定义

$$
d_\Sigma(\mathbf{x},\boldsymbol{\mu})^2
=(\mathbf{x}-\boldsymbol{\mu})^{\mathsf T}
\Sigma^{-1}
(\mathbf{x}-\boldsymbol{\mu})
$$

令 $\Sigma=Q\Lambda Q^{\mathsf T}$，并写

$$
\mathbf{z}=Q^{\mathsf T}(\mathbf{x}-\boldsymbol{\mu})
$$

则

$$
d_\Sigma(\mathbf{x},\boldsymbol{\mu})^2
=\sum_i\frac{z_i^2}{\lambda_i}
$$

协方差大的方向允许更大的位移，除以较大的 $\lambda_i$ 后惩罚较轻；协方差小的方向稍微偏离就会得到较大的距离。白化

$$
\widetilde{\mathbf{z}}=\Lambda^{-1/2}Q^{\mathsf T}
(\mathbf{x}-\boldsymbol{\mu})
$$

正是把这个二次型变成普通欧氏长度平方：

$$
d_\Sigma^2=\|\widetilde{\mathbf{z}}\|_2^2
$$

所以 Mahalanobis 距离不是另造一把神秘的尺子，而是先换到协方差主轴，再按每条轴的方差重新缩放。

## 机器学习里的二次型

**PCA 与协方差。** $\mathbf{v}^{\mathsf T}\Sigma\mathbf{v}$ 给单位方向 $\mathbf{v}$ 上的方差；最大化它得到第一主成分，依次在正交补上最大化得到后续成分。

**Hessian 与优化。** 训练损失在某个点附近常用

$$
L(\mathbf{x}+\Delta)
\approx
L(\mathbf{x})
+\nabla L(\mathbf{x})^{\mathsf T}\Delta
+\frac12\Delta^{\mathsf T}H\Delta
$$

二次项决定局部弯曲。Hessian 特征值跨度很大时，等值线细长，梯度下降会在不同方向上使用不同有效步长；负特征值表示当前点不是局部极小值。

**正则化。** 权重衰减是 $\|w\|_2^2=w^{\mathsf T}w$ 这个最简单的正定二次型。更一般的 $\mathbf{w}^{\mathsf T}R\mathbf{w}$ 可以让不同参数方向受到不同惩罚，但要保证 $R$ 对称半正定，才能避免某些方向奖励无限增大。

**二次评分与双线性评分。** 同一个向量两侧的 $\mathbf{x}^{\mathsf T}A\mathbf{x}$ 只看 $A$ 的对称部分；两个不同向量的 $\mathbf{q}^{\mathsf T}W\mathbf{k}$ 是双线性形式，$W$ 的反对称部分不再自动消失。不要因为两者都写成「向量—矩阵—向量」就把它们当成同一种函数。

## 容易混淆的地方

**把二次型当成范数。** 二次型是标量函数，不自动满足齐次性的绝对值形式、三角不等式或零点唯一性；不定矩阵会给出负值。

**把非对称矩阵的每个元素都当成可见。** $\mathbf{x}^{\mathsf T}A\mathbf{x}$ 只看 $(A+A^{\mathsf T})/2$；要研究反对称部分，必须使用两个不同向量的双线性表达式。

**把半正定当成元素非负。** 对称矩阵的半正定性由特征值或 $\mathbf{x}^{\mathsf T}A\mathbf{x}$ 判断，非对角元素可以是负数。

**把 $q(\mathbf{x})=1$ 和 $q(\mathbf{x})\le1$ 混为一谈。** 前者是等值线或等值面，后者是填充区域；在半正定或不定情形，形状和是否有界还要单独检查。

**把 Rayleigh 商用于零向量。** 分母 $\mathbf{x}^{\mathsf T}\mathbf{x}$ 必须非零；它比较的是方向，不定义在 $\mathbf{x}=\mathbf{0}$。

**把所有特征值结论推广到非对称矩阵。** $\lambda_{\min}\le R_A(\mathbf{x})\le\lambda_{\max}$ 的实数区间结论依赖 $A$ 对称；一般矩阵的二次型先取对称部分。

**把 Mahalanobis 距离的逆矩阵随便求。** $\Sigma$ 必须至少在使用的子空间上可逆；小特征值会放大噪声，实践中常用正则化或截断。

## 相关词条

- [谱定理](../linear-algebra/spectral-theorem/)：用正交特征基把二次型化成平方和。
- [特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/)：Rayleigh 商极值和主方向。
- [特征分解](../linear-algebra/eigendecomposition/)：矩阵函数与谱坐标计算。
- [内积](../linear-algebra/inner-products/)：正定二次型与内积诱导范数。
- [范数](../linear-algebra/norms/)：二次型和范数的边界。
- [协方差矩阵](../probability/covariance-matrix/)：PCA 与 Mahalanobis 距离的矩阵来源。
- [正交投影](../linear-algebra/orthogonal-projections/)：约束子空间中的二次型最优化。
- [SVD](../linear-algebra/svd/)：非对称矩阵的谱量与低秩近似。
