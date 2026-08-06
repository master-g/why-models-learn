---
title: "正交投影:最近点、最小二乘与正规方程"
tags: ["why-models-learn"]
---

给定内积空间中的子空间 $W$ 和向量 $\mathbf{x}$，**正交投影**是 $W$ 中唯一一个向量 $\mathbf{p}$，使得残差 $\mathbf{x}-\mathbf{p}$ 与 $W$ 中每个向量都正交：

$$
\mathbf{p}=\operatorname{proj}_W\mathbf{x}\in W,
\qquad
\mathbf{x}-\mathbf{p}\perp W
$$

它把一个不一定落在 $W$ 里的向量拆成「子空间内的部分」和「垂直子空间的部分」。[角度与正交](../linear-algebra/angles-and-orthogonality/) 篇先在一条直线上推出了投影系数，[正交归一基](../linear-algebra/orthonormal-basis/) 篇又给出了一组正交方向的坐标公式；本篇把这两个结果合成一个线性算子，说明为什么它就是最近点，以及为什么线性回归的最小二乘解其实是在做投影。

## 最近点为什么由正交条件刻画

先看定义带来的几何结论。设 $\mathbf{p}\in W$ 且 $\mathbf{r}=\mathbf{x}-\mathbf{p}$ 与 $W$ 正交。对任意另一个 $\mathbf{w}\in W$，把从 $\mathbf{w}$ 指向 $\mathbf{x}$ 的向量拆开：

$$
\mathbf{x}-\mathbf{w}
=\underbrace{\mathbf{r}}_{\mathbf{x}-\mathbf{p}}
+\underbrace{(\mathbf{p}-\mathbf{w})}_{\in W}
$$

两项正交，所以勾股关系给出

$$
\|\mathbf{x}-\mathbf{w}\|^2
=\|\mathbf{r}\|^2+\|\mathbf{p}-\mathbf{w}\|^2
\geq\|\mathbf{r}\|^2
=\|\mathbf{x}-\mathbf{p}\|^2
$$

等号只有在 $\mathbf{w}=\mathbf{p}$ 时成立。因此，正交条件不只是画图时的一条直角标记：它严格保证 $\mathbf{p}$ 是 $W$ 中离 $\mathbf{x}$ 最近的点，而且最近点是唯一的。

### 先看一条直线

取子空间

$$
W=\operatorname{span}\{(1,1)\},
\qquad
\mathbf{x}=(2,1)
$$

$W$ 中的点都可以写成 $\mathbf{w}=t(1,1)$。距离平方是

$$
\begin{aligned}
\|\mathbf{x}-t(1,1)\|^2
&=(2-t)^2+(1-t)^2\\
&=2\left(t-\frac32\right)^2+\frac12
\end{aligned}
$$

最小值在 $t=3/2$ 处取得，所以

$$
\mathbf{p}=\left(\frac32,\frac32\right),
\qquad
\mathbf{r}=\mathbf{x}-\mathbf{p}=\left(\frac12,-\frac12\right)
$$

确实有

$$
\langle\mathbf{r},(1,1)\rangle
=\frac12-\frac12=0
$$

这个小例子把两个说法接上了：最小化距离平方得到的点，正好是残差垂直于直线的点。对高维子空间，直线上的一个标量 $t$ 会变成一组基系数，逻辑不变。

## 正交归一基给出的投影公式

设 $\mathbf{q}_1,\dots,\mathbf{q}_k$ 是子空间 $W$ 的一组正交归一基。[正交归一基](../linear-algebra/orthonormal-basis/) 篇已经证明，向量沿每个方向的系数就是内积，所以定义

$$
\operatorname{proj}_W\mathbf{x}
=\sum_{i=1}^k\langle\mathbf{x},\mathbf{q}_i\rangle\mathbf{q}_i
$$

令 $\mathbf{p}$ 表示右侧，$\mathbf{r}=\mathbf{x}-\mathbf{p}$。对任意 $j$，

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

残差和 $W$ 的每个基向量都正交，也就和 $W$ 中每个向量正交；上一节的最近点论证随即适用。

把基向量作为列排成 $Q=(\mathbf{q}_1,\dots,\mathbf{q}_k)$，有

$$
Q^{\mathsf T}Q=I_k,
\qquad
\operatorname{proj}_W\mathbf{x}=QQ^{\mathsf T}\mathbf{x}
$$

因此正交投影矩阵是

$$
P_W=QQ^{\mathsf T}
$$

![正交投影的几何分解与一般列基投影矩阵：左侧是子空间内的最近点与垂直残差；右侧是正规方程与投影算子](/assets/linear-algebra/svg/orthogonal-projections.1.svg)

例如 $W=\operatorname{span}\{(1,1)\}$ 的单位基是 $\mathbf{q}=(1,1)/\sqrt2$，于是

$$
P_W=\mathbf{q}\mathbf{q}^{\mathsf T}
=\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix}
$$

作用在 $\mathbf{x}=(2,1)$ 上：

$$
P_W\mathbf{x}
=\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix}
\begin{pmatrix}2\\1\end{pmatrix}
=\begin{pmatrix}3/2\\3/2\end{pmatrix}
$$

和配方法得到的最近点完全一致。

## 投影矩阵的四个性质

正交投影不是任意一个把向量「压扁」的矩阵。它的几何定义在代数上留下四个清晰的指纹：

| 性质 | 公式 | 含义 |
| --- | --- | --- |
| 幂等 | $P_W^2=P_W$ | 已经在 $W$ 里的向量再投影一次不变 |
| 对称 | $P_W^{\mathsf T}=P_W$ | 投影方向与内积相容 |
| 像 | $\operatorname{im}P_W=W$ | 输出恰好落在目标子空间 |
| 核 | $\ker P_W=W^\perp$ | 被完全压成零的正是垂直方向 |

前两条可以直接从 $P_W=QQ^{\mathsf T}$ 验证：

$$
P_W^2
=QQ^{\mathsf T}QQ^{\mathsf T}
=Q(Q^{\mathsf T}Q)Q^{\mathsf T}
=QQ^{\mathsf T}=P_W
$$

以及

$$
P_W^{\mathsf T}=(QQ^{\mathsf T})^{\mathsf T}=QQ^{\mathsf T}=P_W
$$

幂等性回答「再做一次会怎样」，对称性回答「投影不会凭空改变内积的交换关系」。此外，$\mathbf{x}$ 的正交分解可以写成

$$
\mathbf{x}=P_W\mathbf{x}+(I-P_W)\mathbf{x}
$$

其中第一项在 $W$ 中，第二项在 $W^\perp$ 中。由于两项正交，

$$
\|\mathbf{x}\|^2
=\|P_W\mathbf{x}\|^2+\|(I-P_W)\mathbf{x}\|^2
$$

所以投影不会增加长度：$\|P_W\mathbf{x}\|\leq\|\mathbf{x}\|$。被丢掉的长度不是神秘的「误差」，而是向量在 $W^\perp$ 中的那一部分。

## 一般列基：从投影到正规方程

正交归一基很特殊。若用任意线性无关的列向量组成矩阵

$$
A=\begin{pmatrix}\mathbf{a}_1&\cdots&\mathbf{a}_k\end{pmatrix},
\qquad
W=\operatorname{col}(A)
$$

列向量可能既不单位，也互相不正交。此时不能把 $A^{\mathsf T}\mathbf{x}$ 直接当作系数：它只是在每个原始列方向上做了内积，而这些方向彼此会互相干扰。

设投影点是 $\mathbf{p}=A\mathbf{c}$，残差是 $\mathbf{r}=\mathbf{x}-A\mathbf{c}$。残差要与 $W$ 正交，等价于与 $A$ 的每一列正交：

$$
A^{\mathsf T}\mathbf{r}=\mathbf{0}
$$

代入 $\mathbf{r}$，得到**正规方程**(normal equations)：

$$
A^{\mathsf T}A\mathbf{c}=A^{\mathsf T}\mathbf{x}
$$

因为 $A$ 的列线性无关，对任意非零 $\mathbf{z}$ 都有

$$
\mathbf{z}^{\mathsf T}A^{\mathsf T}A\mathbf{z}
=\|A\mathbf{z}\|^2>0
$$

所以 $A^{\mathsf T}A$ 可逆，系数与投影分别是

$$
\mathbf{c}=(A^{\mathsf T}A)^{-1}A^{\mathsf T}\mathbf{x},
\qquad
\operatorname{proj}_W\mathbf{x}
=A(A^{\mathsf T}A)^{-1}A^{\mathsf T}\mathbf{x}
$$

于是一般列基的投影矩阵为

$$
P_A=A(A^{\mathsf T}A)^{-1}A^{\mathsf T}
$$

如果 $A$ 的列本来就正交归一，$A^{\mathsf T}A=I$，这个公式才退化成 $P_A=AA^{\mathsf T}$。

### 三维中的数值核对

取

$$
A=\begin{pmatrix}
1&1\\
0&1\\
1&0
\end{pmatrix},
\qquad
\mathbf{x}=\begin{pmatrix}1\\2\\3\end{pmatrix}
$$

$A$ 的两列分别是 $(1,0,1)$ 和 $(1,1,0)$，内积为 $1$，所以它们不是正交基。先算正规方程的两边：

$$
A^{\mathsf T}A
=\begin{pmatrix}2&1\\1&2\end{pmatrix},
\qquad
A^{\mathsf T}\mathbf{x}
=\begin{pmatrix}4\\3\end{pmatrix}
$$

由于

$$
(A^{\mathsf T}A)^{-1}
=\frac13\begin{pmatrix}2&-1\\-1&2\end{pmatrix}
$$

得到

$$
\mathbf{c}
=\frac13\begin{pmatrix}2&-1\\-1&2\end{pmatrix}
\begin{pmatrix}4\\3\end{pmatrix}
=\begin{pmatrix}5/3\\2/3\end{pmatrix}
$$

因此

$$
\mathbf{p}=A\mathbf{c}
=\begin{pmatrix}7/3\\2/3\\5/3\end{pmatrix},
\qquad
\mathbf{r}=\mathbf{x}-\mathbf{p}
=\begin{pmatrix}-4/3\\4/3\\4/3\end{pmatrix}
$$

最后检查残差确实垂直于两列：

$$
A^{\mathsf T}\mathbf{r}
=\begin{pmatrix}
1&0&1\\1&1&0
\end{pmatrix}
\begin{pmatrix}-4/3\\4/3\\4/3\end{pmatrix}
=\begin{pmatrix}0\\0\end{pmatrix}
$$

能量也按正交分解守恒：

$$
\|\mathbf{x}\|^2=14,
\qquad
\|\mathbf{p}\|^2=\frac{26}{3},
\qquad
\|\mathbf{r}\|^2=\frac{16}{3},
\qquad
14=\frac{26}{3}+\frac{16}{3}
$$

## 最小二乘其实是在投影

把 $\mathbf{x}$ 改名为观测值 $\mathbf{y}$，把 $A$ 看成特征矩阵，问题就变成：找一个系数向量 $\mathbf{c}$，使预测 $A\mathbf{c}$ 尽量靠近 $\mathbf{y}$。形式化写作

$$
\min_{\mathbf{c}}\|\mathbf{y}-A\mathbf{c}\|^2
$$

所有可能的预测值 $A\mathbf{c}$ 恰好组成列空间 $\operatorname{col}(A)$，所以最优预测不是任意点，而是 $\mathbf{y}$ 到这个列空间的正交投影：

$$
\widehat{\mathbf{y}}
=P_A\mathbf{y},
\qquad
\mathbf{e}=\mathbf{y}-\widehat{\mathbf{y}},
\qquad
A^{\mathsf T}\mathbf{e}=\mathbf{0}
$$

「误差与每个特征列正交」就是正规方程的几何含义。数据无法由特征列张成的方向解释的部分，会留在残差里；增加特征列，就是扩大可投影的子空间，训练误差不会因此增加，但泛化表现还要另外评估。

这也解释了为什么实践中常把正规方程当作分析公式，而不直接计算 $(A^{\mathsf T}A)^{-1}$：当列接近线性相关时，$A^{\mathsf T}A$ 会把条件数平方，数值误差可能被放大。QR 分解直接把列空间换成正交归一方向，SVD 则还能识别秩亏方向；两者都比「显式求逆再相乘」更稳健。

## 神经网络里的投影与残差

**线性回归是最直接的例子。** 模型输出被限制在特征列空间内，最小二乘训练选择其中离标签最近的点。偏置项若存在，就相当于在 $A$ 中加入一列全为 $1$ 的特征；这改变了允许的子空间，而不只是给最后结果加一个常数。

**降维和重构都在丢弃正交分量。** 选取前 $k$ 个主方向组成 $Q_k$，重构可以写成 $Q_kQ_k^{\mathsf T}\mathbf{x}$。保留下来的能量是 $\|Q_kQ_k^{\mathsf T}\mathbf{x}\|^2$，误差能量是正交补中的 $\|(I-Q_kQ_k^{\mathsf T})\mathbf{x}\|^2$。这就是「压缩后还剩多少信息」的几何账本。

**投影不是任意线性层。** 一般权重矩阵 $W$ 不满足 $W^2=W$ 或 $W^{\mathsf T}=W$，所以 $W\mathbf{x}$ 通常不是到某个子空间的正交投影。残差连接中的「残差」也只是命名相似，只有当残差与目标子空间正交时，才具有本篇的最近点与能量分解性质。

## 失效模式与常见误区

**把 $A^{\mathsf T}\mathbf{x}$ 当成任意列基的坐标。** 只有列正交归一时才成立。一般列基必须解 $A^{\mathsf T}A\mathbf{c}=A^{\mathsf T}\mathbf{x}$；漏掉 Gram 矩阵 $A^{\mathsf T}A$，就是把互相重叠的特征方向假装成独立方向。

**把 $AA^{\mathsf T}$ 当成任何投影矩阵。** $AA^{\mathsf T}$ 是列正交归一时的公式；一般矩阵要用 $A(A^{\mathsf T}A)^{-1}A^{\mathsf T}$，而且前提是列满秩。列相关时逆不存在，应转向 QR、SVD 或 [伪逆](../linear-algebra/pseudoinverse/)。

**把投影误差当成算错了。** 当 $\mathbf{x}$ 不在 $W$ 中，$\mathbf{x}-P_W\mathbf{x}$ 本来就不为零；它是无法由 $W$ 表示的正交分量。最小二乘追求的是允许空间里的最近点，不保证零误差。

**把幂等误读成恒等。** $P^2=P$ 只表示第二次投影没有新变化，不表示 $P=I$。只有目标子空间是整个空间时，投影才对所有向量都不改变。

**忘记子空间与仿射集的区别。** 本篇的 $W$ 必须经过原点。若目标是平移后的仿射集 $\mathbf{a}+W$，应先对 $\mathbf{x}-\mathbf{a}$ 做到方向子空间 $W$ 的投影，再加回 $\mathbf{a}$；[仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/) 篇解释了为什么基点可以变化而方向不变。

**把正交性当成坐标轴的外观。** 正交由所选内积定义，不要求子空间贴着 $x$ 轴或某几个特征列。换成加权内积后，投影公式中的转置和正交条件也要相应改变，不能未经检查地沿用欧氏点积。

## 相关词条

- [角度与正交](../linear-algebra/angles-and-orthogonality/)：内积、夹角与直线投影的起点
- [正交归一基](../linear-algebra/orthonormal-basis/)：用正交方向读坐标并写出 $QQ^{\mathsf T}$
- [线性方程组](../linear-algebra/linear-systems/)：正规方程与可解性背后的消元语言
- [最小二乘就是投影](../linear-models/least-squares-as-projection/)：把本篇几何推广到线性模型
- [伪逆](../linear-algebra/pseudoinverse/)：列相关或非方阵情形下的广义解
- [仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/)：平移后的目标集与方向子空间
- [正交矩阵与旋转](../linear-algebra/orthogonal-matrices-and-rotations/)：保持内积的特殊线性变换
