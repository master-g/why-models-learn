---
title: "正交归一基:把内积变成坐标读数"
tags: ["why-models-learn"]
---

一组**正交归一基**(orthonormal basis)既是一组基，又满足不同基向量两两正交、每个基向量长度为 $1$：

$$
\langle \mathbf{q}_i,\mathbf{q}_j\rangle=\delta_{ij}
=\begin{cases}
1,&i=j\\
0,&i\ne j
\end{cases}
$$

它的特别之处不是定义更漂亮，而是计算突然变短：任意向量在这组基下的第 $j$ 个坐标，直接就是它和 $mathbf{q}_j$ 的内积。[基](../linear-algebra/basis/) 篇只要求「张成且无关」，[坐标](../linear-algebra/coordinates/) 篇因此需要解方程；正交归一基把解方程变成逐个做内积。本篇解释这个公式、用 Gram–Schmidt 过程构造这样的基，再说明它为什么同时适合几何分解与神经网络中的表示计算。

## 两个条件合在一起

先区分三个相近的说法。向量组**正交**，是不同向量之间的内积为零；向量组**归一**，是每个向量的范数为 $1$；两者同时成立，才是正交归一。

在 $mathbb{R}^2$ 中取

$$
\mathbf{q}_1=\left(\frac35,\frac45\right),\qquad
\mathbf{q}_2=\left(-\frac45,\frac35\right)
$$

它们满足

$$
\langle\mathbf{q}_1,\mathbf{q}_2\rangle
=-\frac{12}{25}+\frac{12}{25}=0,
\qquad
\|\mathbf{q}_1\|^2=\|\mathbf{q}_2\|^2
=\frac9{25}+\frac{16}{25}=1
$$

所以它们是正交归一的。又因为两个非零正交向量必然线性无关，而 $mathbb{R}^2$ 中两个无关向量构成一组基，它们组成 $mathbb{R}^2$ 的正交归一基。

更一般地，正交归一向量组自动线性无关。若

$$
c_1\mathbf{q}_1+\cdots+c_k\mathbf{q}_k=\mathbf{0}
$$

两边与任意一个 $mathbf{q}_j$ 做内积，就得到

$$
0
=\left\langle\sum_{i=1}^k c_i\mathbf{q}_i,\mathbf{q}_j\right\rangle
=\sum_{i=1}^k c_i\langle\mathbf{q}_i,\mathbf{q}_j\rangle
=c_j
$$

所以所有 $c_j$ 都是零。注意：$k$ 个正交归一向量总是构成它们张成空间的基，但若 $k<n$，它们只是 $mathbb{R}^n$ 某个子空间的基，还不是整个 $mathbb{R}^n$ 的基。

## 坐标为什么可以直接读出来

设 $\mathbf{q}_1,\dots,\mathbf{q}_n$ 是 $V$ 的一组正交归一基。任意向量 $\mathbf{x}$ 都能唯一写成

$$
\mathbf{x}=c_1\mathbf{q}_1+\cdots+c_n\mathbf{q}_n
$$

和第 $j$ 个基向量做内积：

$$
\begin{aligned}
\langle\mathbf{x},\mathbf{q}_j\rangle
&=\left\langle\sum_{i=1}^n c_i\mathbf{q}_i,\mathbf{q}_j\right\rangle\\
&=\sum_{i=1}^n c_i\langle\mathbf{q}_i,\mathbf{q}_j\rangle\\
&=c_j
\end{aligned}
$$

因此坐标公式是

$$
[\mathbf{x}]_Q
=\begin{pmatrix}
\langle\mathbf{x},\mathbf{q}_1\rangle\\
\vdots\\
\langle\mathbf{x},\mathbf{q}_n\rangle
\end{pmatrix}
$$

这里的下标 $Q$ 表示正交归一基 $Q=(\mathbf{q}_1,\dots,\mathbf{q}_n)$，不是说坐标变成了另一种向量。普通基下要解 $B\mathbf{c}=\mathbf{x}$；正交归一基下，每个 $c_j$ 都由一次内积给出。原因就在于其他基向量的交叉项全部为零，自己的项恰好留下 $1$。

仍用上面的基，取

$$
\mathbf{x}=(1,2)
$$

则两个坐标分别是

$$
c_1=\langle\mathbf{x},\mathbf{q}_1\rangle
=\frac35+\frac85=\frac{11}{5},
\qquad
c_2=\langle\mathbf{x},\mathbf{q}_2\rangle
=-\frac45+\frac65=\frac25
$$

回代检查：

$$
\frac{11}{5}\mathbf{q}_1+\frac25\mathbf{q}_2
=\left(\frac{33}{25}-\frac8{25},\frac{44}{25}+\frac6{25}\right)
=(1,2)
$$

同一向量在标准基下的坐标是 $(1,2)$，在这组旋转过的正交归一基下是 $(11/5,2/5)$。向量没有动，参照方向换了；但这次换算不需要解二元方程，只要做两次内积。

![Gram–Schmidt 过程把原向量变成正交归一基：先取 q₁，再从 v₂ 中减掉 q₁ 方向，最后归一化得到 q₂](/assets/linear-algebra/svg/orthonormal-basis.1.svg)

## Parseval 公式：长度也变成坐标平方和

直接把坐标公式代回内积，可以得到一整组同时成立的公式。设

$$
\mathbf{x}=\sum_{i=1}^n c_i\mathbf{q}_i,
\qquad
\mathbf{y}=\sum_{i=1}^n d_i\mathbf{q}_i
$$

则

$$
\begin{aligned}
\langle\mathbf{x},\mathbf{y}\rangle
&=\left\langle\sum_i c_i\mathbf{q}_i,\sum_j d_j\mathbf{q}_j\right\rangle\\
&=\sum_i\sum_j c_i d_j\langle\mathbf{q}_i,\mathbf{q}_j\rangle\\
&=\sum_{i=1}^n c_i d_i
\end{aligned}
$$

把 $\mathbf{y}$ 换成 $\mathbf{x}$，得到 Parseval 公式：

$$
\|\mathbf{x}\|^2=\sum_{i=1}^n c_i^2
$$

它是勾股定理从两个方向推广到任意多个正交方向的结果。上面的数值例子中，

$$
\|\mathbf{x}\|^2=1^2+2^2=5,
\qquad
\left(\frac{11}{5}\right)^2+\left(\frac25\right)^2
=\frac{121+4}{25}=5
$$

所以在正交归一基下，向量的总能量就是坐标能量的和，没有被交叉项藏起来。类似地，两个向量的相似程度也可以在坐标侧直接计算：

$$
\langle\mathbf{x},\mathbf{y}\rangle
=[\mathbf{x}]_Q^{\mathsf T}[\mathbf{y}]_Q
$$

这不是把几何事实换成近似值，而是同一个内积在另一组基下的精确表达。

## Gram–Schmidt：从任意基造出正交归一基

正交归一基很方便，但不能只凭愿望得到。给定线性无关向量 $\mathbf{v}_1,\dots,\mathbf{v}_k$，Gram–Schmidt 过程逐个消掉新向量在旧方向上的分量。

第一步先取

$$
\mathbf{u}_1=\mathbf{v}_1,
\qquad
\mathbf{q}_1=\frac{\mathbf{u}_1}{\|\mathbf{u}_1\|}
$$

之后第 $j$ 步先减去已经得到的方向：

$$
\mathbf{u}_j
=\mathbf{v}_j-sum_{i=1}^{j-1}
\langle\mathbf{v}_j,\mathbf{q}_i\rangle\mathbf{q}_i,
\qquad
\mathbf{q}_j=\frac{\mathbf{u}_j}{\|\mathbf{u}_j\|}
$$

为什么这样就正交？对任意 $\ell<j$，

$$
\begin{aligned}
\langle\mathbf{u}_j,\mathbf{q}_\ell\rangle
&=\langle\mathbf{v}_j,\mathbf{q}_\ell\rangle
-\sum_{i=1}^{j-1}
\langle\mathbf{v}_j,\mathbf{q}_i\rangle
\langle\mathbf{q}_i,\mathbf{q}_\ell\rangle\\
&=\langle\mathbf{v}_j,\mathbf{q}_\ell\rangle
-\langle\mathbf{v}_j,\mathbf{q}_\ell\rangle=0
\end{aligned}
$$

归一化只改变长度，不改变零内积。并且 $\mathbf{u}_j$ 不会变成零：如果它是零，就说明 $\mathbf{v}_j$ 可以由前面的 $\mathbf{q}_1,\dots,\mathbf{q}_{j-1}$ 组合出来，也就落在前面向量的张成里，与原向量组线性无关矛盾。

### 一个二维的完整计算

取

$$
\mathbf{v}_1=(1,1),\qquad \mathbf{v}_2=(1,0)
$$

第一步是

$$
\mathbf{u}_1=(1,1),
\qquad
\|\mathbf{u}_1\|=\sqrt2,
\qquad
\mathbf{q}_1=\left(\frac1{\sqrt2},\frac1{\sqrt2}\right)
$$

第二个向量在 $\mathbf{q}_1$ 方向上的系数为

$$
\langle\mathbf{v}_2,\mathbf{q}_1\rangle=\frac1{\sqrt2}
$$

所以去掉这部分后

$$
\mathbf{u}_2
=(1,0)-\frac1{\sqrt2}\left(\frac1{\sqrt2},\frac1{\sqrt2}\right)
=\left(\frac12,-\frac12\right)
$$

它的长度是 $1/\sqrt2$，于是

$$
\mathbf{q}_2
=\frac{\mathbf{u}_2}{\|\mathbf{u}_2\|}
=\left(\frac1{\sqrt2},-\frac1{\sqrt2}\right)
$$

最后检查

$$
\|\mathbf{q}_1\|=\|\mathbf{q}_2\|=1,
\qquad
\langle\mathbf{q}_1,\mathbf{q}_2\rangle
=\frac12-\frac12=0
$$

因此一组看起来斜着的原始向量，经过「减投影、再归一化」就变成了可直接读坐标的正交归一基。输入向量的顺序会影响得到的具体基，但不会改变它们最终张成的子空间。

## 子空间里的正交分解

设 $W$ 是由正交归一向量 $\mathbf{q}_1,\dots,\mathbf{q}_k$ 张成的子空间。对任意 $\mathbf{x}$，定义

$$
\mathbf{p}
=\sum_{i=1}^k\langle\mathbf{x},\mathbf{q}_i\rangle\mathbf{q}_i,
\qquad
\mathbf{r}=\mathbf{x}-\mathbf{p}
$$

对每个基向量 $\mathbf{q}_j$，

$$
\begin{aligned}
\langle\mathbf{r},\mathbf{q}_j\rangle
&=\langle\mathbf{x},\mathbf{q}_j\rangle
-\sum_{i=1}^k\langle\mathbf{x},\mathbf{q}_i\rangle
\langle\mathbf{q}_i,\mathbf{q}_j\rangle\\
&=\langle\mathbf{x},\mathbf{q}_j\rangle
-\langle\mathbf{x},\mathbf{q}_j\rangle=0
\end{aligned}
$$

所以 $\mathbf{p}\in W$，而 $\mathbf{r}\perp W$，并且

$$
\mathbf{x}=\mathbf{p}+\mathbf{r}
$$

这就是沿子空间方向和垂直子空间方向的正交分解。它还有唯一性：假如 $\mathbf{w}\in W$ 且 $\mathbf{x}-\mathbf{w}\perp W$，那么 $\mathbf{p}-\mathbf{w}\in W$，同时

$$
\langle\mathbf{p}-\mathbf{w},\mathbf{p}-\mathbf{w}\rangle=0
$$

只能推出 $\mathbf{p}=\mathbf{w}$。因此，正交归一基给出的 $\mathbf{p}$ 是 $W$ 中唯一留下正交残差的向量；[正交投影](../linear-algebra/orthogonal-projections/) 篇将把这个事实展开成最近点与最小二乘的统一语言。

## 矩阵视角：转置就是反向读数

把正交归一基向量作为列排成矩阵

$$
Q=\begin{pmatrix}\mathbf{q}_1&\cdots&\mathbf{q}_n\end{pmatrix}
$$

列之间的正交归一条件合起来就是

$$
Q^{\mathsf T}Q=I
$$

矩阵 $Q$ 把基下坐标送回标准坐标：$\mathbf{x}=Q[\mathbf{x}]_Q$；转置 $Q^{\mathsf T}$ 则把标准坐标读回这组基：

$$
[\mathbf{x}]_Q=Q^{\mathsf T}\mathbf{x}
$$

当 $Q$ 是方阵时，$Q^{\mathsf T}Q=I$ 说明 $Q^{-1}=Q^{\mathsf T}$。因此求逆不需要消元，反向变换就是转置。对任意坐标向量 $\mathbf{c}$，

$$
\|Q\mathbf{c}\|^2
=\mathbf{c}^{\mathsf T}Q^{\mathsf T}Q\mathbf{c}
=\|\mathbf{c}\|^2
$$

也就是说，正交归一基只是旋转或反射了参照方向，不会把长度拉长或压短；[正交矩阵与旋转](../linear-algebra/orthogonal-matrices-and-rotations/) 篇会专门讨论这种保持内积的线性映射。

如果 $Q$ 是 $n\times k$ 的长方形矩阵，列仍然可以正交归一，此时只能保证 $Q^{\mathsf T}Q=I_k$。一般不能把 $QQ^{\mathsf T}$ 也当成 $I_n$；它表示把向量留下在这 $k$ 个方向上的部分，正是上一节子空间分解的矩阵写法。

## 神经网络为什么在意正交

**表示的能量可以逐方向记账。** 隐藏状态若在正交归一方向上展开，$\|\mathbf{x}\|^2=\sum_i c_i^2$，每个坐标贡献的大小不会和别的坐标产生交叉项。做特征分析、压缩或监控激活尺度时，这个账本比相关坐标混在一起更容易解释。

**PCA 的方向是正交的。** PCA 把数据投到一组按方差排序的主方向上；在欧氏内积下，这些方向可以取成正交归一基。投到第一个方向的系数就是与第一主方向的内积，剩余方向依次记录未被前面方向解释的部分。PCA 的方向是统计结构的结果，不是随便找一组单位向量。

**QR 分解把相关列换成正交列。** 对列线性无关的矩阵 $A$，QR 分解写成

$$
A=QR
$$

其中 $Q$ 的列正交归一，$R$ 是上三角矩阵。Gram–Schmidt 给出这件事的几何构造；数值计算中常用更稳定的 QR 算法。先把方向正交化，再在 $R$ 中记录原始列需要多少份各方向，能避免直接拿一组高度相关的列反复解方程。

**余弦相似度依赖归一化但不自动产生正交。** 把两个 embedding 各自除以长度，点积就等于余弦相似度；这只保证每个向量长度为 $1$，并不保证不同 embedding 彼此正交。正交归一是「坐标轴之间不重叠」，归一化是「每个向量自身长度统一」，两者解决的是不同问题。

## 失效模式与常见误区

**正交不等于正交归一。** $(2,0)$ 与 $(0,3)$ 的内积是零，但长度分别为 $2$ 和 $3$，不能直接用 $\langle\mathbf{x},\mathbf{q}_j\rangle$ 读坐标。只有单位向量才有 $\langle\mathbf{q}_j,\mathbf{q}_j\rangle=1$；普通正交基的系数要除以对应的 $\|\mathbf{q}_j\|^2$。

**正交归一向量组不一定是整个空间的基。** 在 $\mathbb{R}^3$ 中，只有 $\mathbf{e}_1,\mathbf{e}_2$ 时，它们是 $xy$ 平面的正交归一基，却不能表示带非零 $z$ 分量的向量。判断「基」时必须同时报清楚它的目标空间。

**把 $Q^{\mathsf T}Q=I$ 推成 $QQ^{\mathsf T}=I$。** 对方阵这两式都成立；对列数少于行数的长方形 $Q$，前者是列正交归一，后者通常只是到子空间的投影，不能覆盖整个空间。

**把正交当成统计独立。** 两个随机变量协方差为零，是一种二阶线性关系；除非额外满足联合高斯等条件，它不等于概率意义上的独立。线性代数里的正交也只由选定的内积决定，换一个加权内积，原先的点积零未必仍代表正交。

**以为 Gram–Schmidt 的结果唯一。** 输入顺序或每一步的符号选择会改变得到的正交归一基；唯一的是它们张成的子空间。浮点计算中，经典 Gram–Schmidt 还可能逐渐丢失正交性，工程实现通常选修正 Gram–Schmidt 或 Householder QR。

## 相关词条

- [角度与正交](../linear-algebra/angles-and-orthogonality/)：内积如何定义夹角、正交与投影方向
- [内积](../linear-algebra/inner-products/)：正交归一条件所依赖的运算公理
- [基](../linear-algebra/basis/)：张成与无关，正交归一基的两层外壳
- [坐标](../linear-algebra/coordinates/)：一般基下的配方与坐标换算
- [正交投影](../linear-algebra/orthogonal-projections/)：正交分解、最近点与最小二乘
- [正交矩阵与旋转](../linear-algebra/orthogonal-matrices-and-rotations/)：$Q^{\mathsf T}Q=I$ 的几何变换
