---
title: "奇异值分解:把任意矩阵拆成旋转、缩放与旋转"
tags: ["why-models-learn"]
---

对任意实矩阵 $A\in\mathbb{R}^{m\times n}$，都可以找到两个正交矩阵和一个非负对角形矩阵，使得

$$
A=U\Sigma V^{\mathsf T}
$$

这叫**奇异值分解**（singular value decomposition，SVD）。它把一个可能是长方形、非对称、甚至秩亏的线性映射拆成三个动作：先用 $V^{\mathsf T}$ 换到输入的正交主方向，再用 $\Sigma$ 沿这些方向分别缩放，最后用 $U$ 把结果放回输出空间。本篇从 $A^{\mathsf T}A$ 的特征分解推导 SVD，算一个完整例子，再说明外积展开、几何意义、伪逆和低秩近似。

## 先看三个阶段的形状

矩阵 $A$ 把 $\mathbb{R}^n$ 中的向量送到 $\mathbb{R}^m$。完整 SVD 中各因子的形状如下：

| 因子 | 形状 | 作用 |
| --- | --- | --- |
| $V^{\mathsf T}$ | $n\times n$ | 在输入空间中换一组正交坐标 |
| $\Sigma$ | $m\times n$ | 沿对应方向乘以非负奇异值，并处理维度差异 |
| $U$ | $m\times m$ | 在输出空间中换回一组正交坐标 |

所以对一个向量 $\mathbf{x}$，实际计算顺序是

$$
\mathbf{x}
\xrightarrow{\ V^{\mathsf T}\ }
V^{\mathsf T}\mathbf{x}
\xrightarrow{\ \Sigma\ }
\Sigma V^{\mathsf T}\mathbf{x}
\xrightarrow{\ U\ }
U\Sigma V^{\mathsf T}\mathbf{x}=A\mathbf{x}
$$

$U$ 和 $V$ 是正交矩阵，满足

$$
U^{\mathsf T}U=I_m,
\qquad
V^{\mathsf T}V=I_n
$$

它们只改变坐标方向，不改变长度。真正改变长度的是 $\Sigma$：输入方向 $\mathbf{v}_i$ 经过 $A$ 后变成输出方向 $\mathbf{u}_i$，长度乘上 $\sigma_i$。这里的「旋转」也包括反射，因为正交矩阵的行列式可以是 $1$ 或 $-1$。

## SVD 的标准形式

把奇异值按从大到小排列：

$$
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0，
\qquad r=\operatorname{rank}(A)
$$

完整 SVD 写成

$$
A=U\Sigma V^{\mathsf T}，
\qquad
U\in\mathbb{R}^{m\times m}，
\quad
V\in\mathbb{R}^{n\times n}，
\quad
\Sigma\in\mathbb{R}^{m\times n}
$$

其中 $\Sigma$ 的主对角线上是 $\sigma_1,\dots,\sigma_r$，其余位置为 $0$。如果 $m=n$，它就是熟悉的方阵对角矩阵；如果 $m\ne n$，对角线之外多出的行或列也填 $0$。

只保留非零方向，可以写成紧 SVD：

$$
A=U_r\Sigma_rV_r^{\mathsf T}
$$

其中

$$
U_r\in\mathbb{R}^{m\times r}，
\qquad
\Sigma_r=\operatorname{diag}(\sigma_1,\dots,\sigma_r)\in\mathbb{R}^{r\times r}，
\qquad
V_r\in\mathbb{R}^{n\times r}
$$

紧 SVD 省掉了零空间和输出空间中与像空间正交的那些基向量。做低秩存储、最小二乘和伪逆时，通常直接使用它。

## 为什么从 $A^{\mathsf T}A$ 开始

先观察

$$
(A^{\mathsf T}A)^{\mathsf T}=A^{\mathsf T}A
$$

所以 $A^{\mathsf T}A$ 是对称矩阵。对任意 $\mathbf{x}$，

$$
\mathbf{x}^{\mathsf T}A^{\mathsf T}A\mathbf{x}
=(A\mathbf{x})^{\mathsf T}(A\mathbf{x})
=\lVert A\mathbf{x}\rVert^2
\ge0
$$

它还是半正定矩阵。根据 [谱定理](../linear-algebra/spectral-theorem/)，存在一组正交归一的向量 $\mathbf{v}_1,\dots,\mathbf{v}_n$，以及非负特征值 $\lambda_i$，满足

$$
A^{\mathsf T}A\mathbf{v}_i=\lambda_i\mathbf{v}_i，
\qquad
\mathbf{v}_i^{\mathsf T}\mathbf{v}_j=\delta_{ij}
$$

定义

$$
\sigma_i=\sqrt{\lambda_i}
$$

这些 $\sigma_i$ 就是 $A$ 的奇异值。为什么要开平方？因为

$$
\lVert A\mathbf{v}_i\rVert^2
=\mathbf{v}_i^{\mathsf T}A^{\mathsf T}A\mathbf{v}_i
=\lambda_i
=\sigma_i^2
$$

它们是 $A$ 沿某个单位输入方向放大的长度，而不是长度的平方。

当 $\sigma_i>0$ 时，定义输出方向

$$
\mathbf{u}_i=\frac{A\mathbf{v}_i}{\sigma_i}
$$

它确实是单位向量，因为

$$
\lVert\mathbf{u}_i\rVert^2
=\frac{\mathbf{v}_i^{\mathsf T}A^{\mathsf T}A\mathbf{v}_i}{\sigma_i^2}
=1
$$

不同的正奇异值对应的输出方向也正交：

$$
\mathbf{u}_i^{\mathsf T}\mathbf{u}_j
=\frac{\mathbf{v}_i^{\mathsf T}A^{\mathsf T}A\mathbf{v}_j}{\sigma_i\sigma_j}
=\frac{\lambda_j\mathbf{v}_i^{\mathsf T}\mathbf{v}_j}{\sigma_i\sigma_j}
=0
\qquad(i\ne j)
$$

定义马上给出两条对称关系：

$$
A\mathbf{v}_i=\sigma_i\mathbf{u}_i，
\qquad
A^{\mathsf T}\mathbf{u}_i
=\frac{A^{\mathsf T}A\mathbf{v}_i}{\sigma_i}
=\sigma_i\mathbf{v}_i
$$

如果 $\sigma_i=0$，则

$$
\lVert A\mathbf{v}_i\rVert^2=0
\quad\Longrightarrow\quad
A\mathbf{v}_i=\mathbf{0}
$$

这个方向属于 $\ker(A)$。把所有非零 $\mathbf{u}_i$ 补成输出空间的一组正交归一基，把所有 $\mathbf{v}_i$ 排成 $V$，就得到 $U$、$\Sigma$ 和

$$
A\mathbf{v}_i=\sigma_i\mathbf{u}_i
$$

对每个方向都成立。因此 $A=U\Sigma V^{\mathsf T}$。这也解释了 SVD 为什么不要求 $A$ 方阵或可逆：零奇异值只表示某些方向被压到零。

## 一个完整的二维分解

取一个非对称矩阵

$$
A=
\begin{pmatrix}
0&4\\
3&0
\end{pmatrix}
$$

先算两个乘积：

$$
A^{\mathsf T}A=
\begin{pmatrix}
9&0\\
0&16
\end{pmatrix}
，
\qquad
AA^{\mathsf T}=
\begin{pmatrix}
16&0\\
0&9
\end{pmatrix}
$$

$A^{\mathsf T}A$ 的最大特征值是 $16$，对应输入方向

$$
\mathbf{v}_1=
\begin{pmatrix}0\\1\end{pmatrix}，
\qquad
\mathbf{v}_2=
\begin{pmatrix}1\\0\end{pmatrix}
$$

因此

$$
\sigma_1=4，
\qquad
\sigma_2=3，
\qquad
V=
\begin{pmatrix}
0&1\\
1&0
\end{pmatrix}
$$

把输入方向送过去：

$$
A\mathbf{v}_1=
\begin{pmatrix}4\\0\end{pmatrix}
=4
\begin{pmatrix}1\\0\end{pmatrix}
，
\qquad
A\mathbf{v}_2=
\begin{pmatrix}0\\3\end{pmatrix}
=3
\begin{pmatrix}0\\1\end{pmatrix}
$$

可取

$$
U=
\begin{pmatrix}1&0\\0&1\end{pmatrix}，
\qquad
\Sigma=
\begin{pmatrix}4&0\\0&3\end{pmatrix}
$$

于是

$$
A=U\Sigma V^{\mathsf T}
=
\begin{pmatrix}1&0\\0&1\end{pmatrix}
\begin{pmatrix}4&0\\0&3\end{pmatrix}
\begin{pmatrix}0&1\\1&0\end{pmatrix}
=
\begin{pmatrix}0&4\\3&0\end{pmatrix}
$$

![奇异值分解的三段作用：先沿输入奇异向量换坐标，再按奇异值缩放，最后沿输出奇异向量换回坐标](/assets/linear-algebra/svg/svd.1.svg)

用一个向量检查三个阶段。令

$$
\mathbf{x}=\begin{pmatrix}1\\2\end{pmatrix}
$$

则

$$
V^{\mathsf T}\mathbf{x}
=\begin{pmatrix}2\\1\end{pmatrix}
，
\qquad
\Sigma V^{\mathsf T}\mathbf{x}
=\begin{pmatrix}8\\3\end{pmatrix}
，
\qquad
A\mathbf{x}=\begin{pmatrix}8\\3\end{pmatrix}
$$

原向量长度平方是 $1^2+2^2=5$，输出长度平方是 $8^2+3^2=73$。这不是一个统一缩放：输入在 $\mathbf{v}_1$ 方向的分量乘 $4$，在 $\mathbf{v}_2$ 方向的分量乘 $3$。

## 外积展开：每个方向贡献一块秩一矩阵

把 $U\Sigma V^{\mathsf T}$ 按列乘开，可以得到

$$
A=\sum_{i=1}^r\sigma_i\mathbf{u}_i\mathbf{v}_i^{\mathsf T}
$$

其中 $\mathbf{u}_i\mathbf{v}_i^{\mathsf T}$ 是一个秩为 $1$ 的矩阵。它先用 $\mathbf{v}_i^{\mathsf T}$ 从输入中取出第 $i$ 个方向的坐标，再沿 $\mathbf{u}_i$ 输出；$\sigma_i$ 是这条通道的增益。

看一个秩亏例子：

$$
B=
\begin{pmatrix}
3&0\\
4&0
\end{pmatrix}
$$

有

$$
B^{\mathsf T}B=
\begin{pmatrix}25&0\\0&0\end{pmatrix}
$$

所以奇异值是 $5$、$0$。取

$$
\mathbf{u}_1=\frac15\begin{pmatrix}3\\4\end{pmatrix}，
\qquad
\mathbf{u}_2=\frac15\begin{pmatrix}-4\\3\end{pmatrix}，
\qquad
\mathbf{v}_1=\begin{pmatrix}1\\0\end{pmatrix}，
\qquad
\mathbf{v}_2=\begin{pmatrix}0\\1\end{pmatrix}
$$

于是

$$
B=5\mathbf{u}_1\mathbf{v}_1^{\mathsf T}
=5
\begin{pmatrix}3/5\\4/5\end{pmatrix}
\begin{pmatrix}1&0\end{pmatrix}
=
\begin{pmatrix}3&0\\4&0\end{pmatrix}
$$

$\mathbf{v}_2$ 被送到零向量，因为它位于核中。虽然左侧仍然需要第二个正交方向 $\mathbf{u}_2$ 才能组成完整的 $U$，这个方向的通道增益已经是 $0$。

## 奇异值描述长度、面积和条件数

把输入向量按右奇异向量展开：

$$
\mathbf{x}=\sum_i z_i\mathbf{v}_i，
\qquad
z_i=\mathbf{v}_i^{\mathsf T}\mathbf{x}
$$

由于 $V$ 正交，输入长度平方是 $\sum_i z_i^2$。应用 $A$ 后，

$$
A\mathbf{x}=\sum_i\sigma_i z_i\mathbf{u}_i
$$

而 $U$ 的列也正交，所以

$$
\lVert A\mathbf{x}\rVert^2
=\sum_i\sigma_i^2z_i^2
$$

每个输入主方向独立地贡献一个平方项。单位球面经过 $A$ 后变成椭球；$\sigma_i$ 是相应半轴长度。若某个 $\sigma_i=0$，椭球在该方向塌成更低维的平面或线段。

从上式还可以直接读出最大长度放大倍数：

$$
\frac{\lVert A\mathbf{x}\rVert^2}{\lVert\mathbf{x}\rVert^2}
=\frac{\sum_i\sigma_i^2z_i^2}{\sum_i z_i^2}
\le\sigma_1^2
$$

因此

$$
\lVert A\rVert_2
=\max_{\mathbf{x}\ne\mathbf{0}}
\frac{\lVert A\mathbf{x}\rVert}{\lVert\mathbf{x}\rVert}
=\sigma_1
$$

当 $A$ 是可逆方阵时，最小奇异值也不为零，并且二范数条件数为

$$
\kappa_2(A)=\frac{\sigma_1}{\sigma_n}
$$

如果 $\sigma_n$ 很小，某个方向会被严重压缩；反向求解时，这个方向上的微小误差会被放大。矩阵范数与条件数的统一整理见 [矩阵范数](../linear-algebra/matrix-norms/)，这里先保留奇异值这个最直接的几何解释。

## 它和特征分解什么时候相同

奇异值分解使用 $A^{\mathsf T}A$，特征分解直接研究 $A$。两者在对称矩阵上有紧密关系，但不是同一个问题。

若 $A$ 是对称正半定矩阵，谱定理给出

$$
A=Q\Lambda Q^{\mathsf T}，
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n)，
\qquad
\lambda_i\ge0
$$

此时可以取

$$
U=V=Q，
\qquad
\Sigma=\Lambda
$$

因为 $A$ 的特征值本身就是非负奇异值。若 $A$ 是对称但不定，特征值可能为负，而奇异值仍然必须非负。写成

$$
A=Q\Lambda Q^{\mathsf T}，
\qquad
\Sigma=\operatorname{diag}(|\lambda_1|,\dots,|\lambda_n|)
$$

对每个负特征值把对应的左奇异向量取反，就能得到一个 SVD。例如可以令 $V=Q$，$U=Q\operatorname{diag}(\operatorname{sign}\lambda_i)$，零特征值处的符号任取。

一般非对称矩阵甚至可能没有实特征向量。旋转矩阵

$$
R=
\begin{pmatrix}0&-1\\1&0\end{pmatrix}
$$

在实数域没有特征值，但

$$
R^{\mathsf T}R=I
$$

所以它的两个奇异值都是 $1$。SVD 仍然可以准确描述它：没有长度缩放，只有方向旋转。[特征分解](../linear-algebra/eigendecomposition/)处理的是「是否存在不变方向」，SVD 处理的是「输入正交方向被放大多少以及被送到哪些输出正交方向」。

## 伪逆和低秩近似从哪里来

只要保留非零奇异值，紧 SVD 就给出伪逆：

$$
A^+=V_r\Sigma_r^{-1}U_r^{\mathsf T}
$$

其中

$$
\Sigma_r^{-1}
=\operatorname{diag}
\left(\frac1{\sigma_1},\dots,\frac1{\sigma_r}\right)
$$

它沿每个非零通道反向缩放，再把输出方向读回输入方向。零奇异值不被倒数，因为核中的信息已经丢失。方阵可逆时，所有奇异值都非零，公式退化为

$$
A^{-1}=V\Sigma^{-1}U^{\mathsf T}
$$

伪逆的最小二乘解释见 [伪逆](../linear-algebra/pseudoinverse/)。

外积展开还允许只保留最大的 $k$ 个通道：

$$
A_k=\sum_{i=1}^k\sigma_i\mathbf{u}_i\mathbf{v}_i^{\mathsf T}
$$

这会把矩阵表示成最多秩为 $k$ 的近似。保留大的奇异值，是保留主要方向上的作用；丢掉小的奇异值，则丢掉对应的弱通道。[低秩近似](../linear-algebra/low-rank-approximation/)篇会证明这种截断在常用误差度量下为什么最优，并计算存储量和误差如何变化。

## 在机器学习中的读法

对一个线性层 $\mathbf{y}=W\mathbf{x}$，SVD 把参数矩阵的作用拆成三句话：

1. $V^{\mathsf T}$ 找出输入表示中的正交主方向。
2. $\Sigma$ 给每个主方向一个增益 $\sigma_i$。
3. $U$ 把这些通道组合成输出表示。

如果最大奇异值很大，某个方向上的扰动可能被放大；如果连续线性层的某些奇异值很小，信息可能逐层衰减。多层矩阵的奇异值不等于各层奇异值逐项相乘，但范数满足

$$
\lVert W_L\cdots W_2W_1\rVert_2
\le
\prod_{\ell=1}^L\lVert W_\ell\rVert_2
$$

所以控制最大奇异值是分析信号放大的一个入口。

对中心化数据矩阵 $X\in\mathbb{R}^{N\times d}$ 做紧 SVD：

$$
X=U_r\Sigma_rV_r^{\mathsf T}
$$

右奇异向量是特征方向，因为

$$
X^{\mathsf T}X
=V_r\Sigma_r^2V_r^{\mathsf T}
$$

协方差矩阵 $X^{\mathsf T}X/(N-1)$ 的方差沿第 $i$ 个主方向为

$$
\frac{\sigma_i^2}{N-1}
$$

这就是 PCA 中用 SVD 代替直接形成协方差矩阵的原因：右奇异向量直接给主方向，奇异值平方给未归一化方差。对称正半定矩阵的 SVD 与谱分解在这里重合，具体推导见 [谱定理](../linear-algebra/spectral-theorem/)。

低秩参数化、压缩和 LoRA 都会保留少数方向的作用；SVD 提供了观察这些方向、排序它们并测量被丢弃能量的坐标系。

## 容易混淆的地方

**把特征值当成奇异值。** 奇异值是 $A^{\mathsf T}A$ 的特征值开平方，始终非负；$A$ 的负特征值不能直接写进 $\Sigma$。

**忽略矩阵形状。** $A$ 是 $m\times n$ 时，$V$ 作用在输入的 $n$ 维空间，$U$ 作用在输出的 $m$ 维空间。把 $U$、$V$ 的方向对调，乘积通常连形状都不成立。

**把 $A^{\mathsf T}A$ 的零特征值当作普通方向。** 它对应 $A\mathbf{v}=\mathbf{0}$，是核方向；做伪逆时不能取 $1/0$。

**以为所有方向都统一缩放。** 只有当所有非零奇异值相同时才是统一尺度；一般矩阵沿不同右奇异方向有不同增益。

**把矩阵元素大当成奇异值大。** 奇异值是整体作用的量，必须由 $A^{\mathsf T}A$ 或分解得到；一个单独的大元素不决定最大放大方向。

**把截断 SVD 当作随便删列。** $A_k$ 删除的是小奇异值对应的外积通道，不是原矩阵中任意几列或几行。

## 相关词条

- [特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/)：解释不变方向和特征值方程。
- [特征分解](../linear-algebra/eigendecomposition/)：把有特征基的方阵写成 $P\Lambda P^{-1}$。
- [谱定理](../linear-algebra/spectral-theorem/)：说明实对称矩阵为什么能使用正交特征基。
- [正交归一基](../linear-algebra/orthonormal-basis/)：构成 $U$、$V$ 的方向基。
- [正交矩阵与旋转](../linear-algebra/orthogonal-matrices-and-rotations/)：解释 $U$、$V$ 为什么保长度。
- [秩](../linear-algebra/rank/)：数非零奇异值，得到像空间维数。
- [伪逆](../linear-algebra/pseudoinverse/)：用非零奇异值构造最小二乘解。
- [低秩近似](../linear-algebra/low-rank-approximation/)：研究截断 SVD 的误差和存储。
- [矩阵范数](../linear-algebra/matrix-norms/)：展开奇异值与算子范数、Frobenius 范数的关系。
