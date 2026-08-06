---
title: "正交矩阵与旋转:保持内积的线性变换"
tags: ["why-models-learn"]
---

一个实方阵 $Q$ 称为**正交矩阵**(orthogonal matrix)，如果

$$
Q^{\mathsf T}Q=I
$$

方阵条件使它同时满足 $QQ^{\mathsf T}=I$，所以 $Q^{-1}=Q^{\mathsf T}$。这条代数等式的几何含义是：$Q$ 可以旋转或反射整个空间，但不改变向量之间的内积、长度、距离和夹角。[正交归一基](../linear-algebra/orthonormal-basis/) 篇从「一组好基」的角度研究了列向量，本篇把这些列组合成一个变换，回答「整个空间一起换方向时，哪些事实保持不变」。二维旋转是最直观的例子，反射则提醒我们：保持长度的不止旋转。

## 列向量已经把条件写完了

把 $Q$ 的列写成

$$
Q=\begin{pmatrix}\mathbf{q}_1&\cdots&\mathbf{q}_n\end{pmatrix}
$$

矩阵乘法的第 $(i,j)$ 个元素是

$$
(Q^{\mathsf T}Q)_{ij}
=\mathbf{q}_i^{\mathsf T}\mathbf{q}_j
=\langle\mathbf{q}_i,\mathbf{q}_j\rangle
$$

因此 $Q^{\mathsf T}Q=I$ 正好等价于：列向量两两正交，每列长度为 $1$。若列数等于空间维数，它们就是整个空间的一组正交归一基；因为 $Q$ 是方阵，它们不仅张成空间，还给出一个可逆变换。方阵的转置也满足

$$
QQ^{\mathsf T}=I
$$

所以行向量同样组成一组正交归一基。列是在说「标准基向量被送到哪里」，行是在说「输出如何读回输入坐标」。[线性映射](../linear-algebra/linear-maps/) 篇所说的「基向量的去向决定整个映射」，在这里又多了一层限制：这些去向必须保持正交归一。

沿用前一篇的基，取

$$
Q=\frac15\begin{pmatrix}3&-4\\4&3\end{pmatrix}
$$

它的两列是

$$
\mathbf{q}_1=\left(\frac35,\frac45\right),\qquad
\mathbf{q}_2=\left(-\frac45,\frac35\right)
$$

正交归一条件已经保证 $Q^{\mathsf T}Q=I$。对向量 $\mathbf{x}=(1,2)$，转置给出它在这组基下的坐标：

$$
Q^{\mathsf T}\mathbf{x}
=\begin{pmatrix}3/5&4/5\\-4/5&3/5\end{pmatrix}
\begin{pmatrix}1\\2\end{pmatrix}
=\begin{pmatrix}11/5\\2/5\end{pmatrix}
$$

再乘回去：

$$
Q\begin{pmatrix}11/5\\2/5\end{pmatrix}
=\begin{pmatrix}1\\2\end{pmatrix}
$$

这里同一个 $Q$ 有两种读法：作为「坐标矩阵」，它把正交归一基下的坐标送回标准坐标；作为「线性变换」，它把每个标准坐标向量旋转到对应的列向量。数学对象相同，问题语境不同。

## 为什么内积、长度和角度都不变

对任意 $\mathbf{x},\mathbf{y}$，直接把 $Q^{\mathsf T}Q=I$ 插入内积：

$$
\begin{aligned}
\langle Q\mathbf{x},Q\mathbf{y}\rangle
&=(Q\mathbf{x})^{\mathsf T}(Q\mathbf{y})\\
&=\mathbf{x}^{\mathsf T}Q^{\mathsf T}Q\mathbf{y}\\
&=\mathbf{x}^{\mathsf T}\mathbf{y}\\
&=\langle\mathbf{x},\mathbf{y}\rangle
\end{aligned}
$$

令 $\mathbf{y}=\mathbf{x}$，得到长度不变：

$$
\|Q\mathbf{x}\|^2=\|\mathbf{x}\|^2
$$

对差向量使用同一个结论，得到距离不变：

$$
\|Q\mathbf{x}-Q\mathbf{y}\|
=\|Q(\mathbf{x}-\mathbf{y})\|
=\|\mathbf{x}-\mathbf{y}\|
$$

角度的余弦是内积除以两个长度；三者都保持，所以非零向量的夹角也保持。正交矩阵可以改变「箭头指向坐标轴的哪一侧」，但不能改变两支箭头之间的几何关系。

### 一个 90° 旋转的逐项核对

取逆时针 90° 旋转矩阵

$$
R_{90}=\begin{pmatrix}0&-1\\1&0\end{pmatrix}
$$

以及

$$
\mathbf{x}=(3,4),\qquad \mathbf{y}=(1,-2)
$$

变换后是

$$
R_{90}\mathbf{x}=(-4,3),\qquad
R_{90}\mathbf{y}=(2,1)
$$

把几个不变量列在一起：

| 量 | 变换前 | 变换后 |
| --- | --- | --- |
| $\langle\mathbf{x},\mathbf{y}\rangle$ | $3\cdot1+4\cdot(-2)=-5$ | $(-4)\cdot2+3\cdot1=-5$ |
| $\|\mathbf{x}\|^2$ | $3^2+4^2=25$ | $(-4)^2+3^2=25$ |
| $\|\mathbf{x}-\mathbf{y}\|^2$ | $2^2+6^2=40$ | $(-6)^2+2^2=40$ |

坐标分量明显变了，内积账本却逐项相同。这是「旋转改变方向但不改变几何」的数值版本。

## 二维旋转矩阵

任意角度 $\theta$ 的逆时针旋转写成

$$
R_\theta
=\begin{pmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{pmatrix}
$$

它的两列长度分别是

$$
\cos^2\theta+\sin^2\theta=1
$$

两列的内积是

$$
\cos\theta(-\sin\theta)+\sin\theta\cos\theta=0
$$

所以 $R_\theta^{\mathsf T}R_\theta=I$。转置和逆分别对应反向旋转：

$$
R_\theta^{\mathsf T}=R_{-\theta}=R_\theta^{-1}
$$

连续旋转的复合角度相加。把三角加法公式写进矩阵乘法，就得到

$$
R_\alpha R_\beta=R_{\alpha+\beta}
$$

这条等式不是把角度「凭直觉相加」，而是两个正交线性映射逐次作用后的矩阵结果。$R_0=I$ 是不动变换，$R_\pi=-I$ 把每个向量反向但仍保持所有长度和夹角。

![二维正交变换的对照：左侧 90° 旋转把 x 送到 R₉₀x；右侧反射把向量翻到 x 轴另一侧，两者都保持长度](/assets/linear-algebra/svg/orthogonal-matrices-and-rotations.1.svg)

## 反射也是正交变换

关于 $x$ 轴的反射矩阵是

$$
F=\begin{pmatrix}1&0\\0&-1\end{pmatrix}
$$

它满足 $F^{\mathsf T}F=I$，作用是

$$
F(x,y)=(x,-y)
$$

因此反射同样保持内积、长度和距离；区别在于它会翻转空间的方向。用后续 [行列式](../linear-algebra/determinant/) 篇的语言，二维中

$$
\det R_\theta=1,\qquad \det F=-1
$$

$\det=1$ 的正交变换保持方向，二维里就是旋转；$\det=-1$ 的正交变换包含一次反射。任意一条过原点、与 $x$ 轴成 $\varphi$ 角的反射，都可以写成

$$
F_\varphi=R_\varphi F R_{-\varphi}
$$

先把反射轴转到 $x$ 轴，做一次 $F$，再转回去。这个分解把「斜着的镜面」还原成最简单的坐标轴反射。

更高维时，正交矩阵仍然满足 $\det Q=\pm1$：它可以由若干个旋转平面和反射组合而成。这里不展开行列式如何计算，先保留它的几何作用：绝对值为 $1$ 表示体积尺度不变，符号记录方向是否翻转。

## 投影结构会跟着一起搬动

正交变换不只保持单个向量的长度，还保持子空间之间的关系。设 $W$ 是子空间，$QW=\{Q\mathbf{w}:\mathbf{w}\in W\}$ 是它变换后的子空间。若 $\mathbf{w}\in W$、$\mathbf{z}\in W^\perp$，则

$$
\langle Q\mathbf{w},Q\mathbf{z}\rangle
=\langle\mathbf{w},\mathbf{z}\rangle=0
$$

所以

$$
Q(W^\perp)=(QW)^\perp
$$

上一条等式也意味着投影与正交变换相容：

$$
Q\operatorname{proj}_W\mathbf{x}
=\operatorname{proj}_{QW}(Q\mathbf{x})
$$

用投影矩阵写，就是

$$
P_{QW}=QP_WQ^{\mathsf T}
$$

验证方式很直接：$P_W\mathbf{x}$ 是 $W$ 内的分量，乘 $Q$ 后落入 $QW$；$\mathbf{x}-P_W\mathbf{x}$ 的正交性由 $Q$ 保留，所以 $QP_W\mathbf{x}$ 正是 $Q\mathbf{x}$ 的正交投影。换句话说，先投影再旋转，和先旋转再投影到旋转后的子空间，结果相同。

## 神经网络为什么会用正交约束

**保留表示几何。** 若一层权重 $Q$ 满足 $Q^{\mathsf T}Q=I$，这一层不会单独改变所有样本的两两距离和余弦相似度。它可以重新组织特征方向，却不会因为矩阵本身而把某些方向放大、把另一些方向压扁。

**稳定连续的线性传播。** 连续乘多个正交矩阵仍是正交矩阵：$Q_1Q_2$ 的转置乘积仍为单位阵。因此只由这些矩阵组成的线性链不会因为重复相乘而产生范数爆炸或消失。实际网络还有非线性、偏置、归一化和残差分支，正交约束是其中一项几何控制，不是对整条网络训练行为的保证。

**对齐两个表示空间。** 两个 embedding 集合可能只差一个整体旋转：样本之间的距离和夹角都一样，但坐标轴方向不同。寻找正交矩阵 $Q$ 使 $XQ$ 尽量靠近 $Y$，就是 Procrustes 对齐的基本形式；它改变参照方向而不破坏 $X$ 内部的几何关系。

**表达能力也受到限制。** 普通线性层可以拉伸、压缩、剪切和投影；正交矩阵只允许保持欧氏几何的旋转与反射。若任务需要改变尺度或丢弃方向，单独使用正交矩阵不够，通常还要配合对角缩放、低秩项或非线性。

## 失效模式与常见误区

**把正交矩阵等同于旋转。** 反射矩阵也满足 $Q^{\mathsf T}Q=I$；看行列式符号或检查是否翻转方向，才能区分旋转与含反射的变换。

**把 $Q^{\mathsf T}\mathbf{x}$ 和 $Q\mathbf{x}$ 当成同一件事。** 若 $Q$ 的列是新基，$Q^{\mathsf T}\mathbf{x}$ 是把标准坐标读成新基坐标；$Q\mathbf{x}$ 则是把 $\mathbf{x}$ 当作标准坐标向量送过一个正交变换。一个是换读数，一个是作用在几何向量上，必须看清语境。

**把长方形的列正交当成可逆的正交矩阵。** $n\times k$ 矩阵可以有正交归一列，此时 $Q^{\mathsf T}Q=I_k$，但 $QQ^{\mathsf T}$ 一般是投影矩阵而不是 $I_n$；它表示只覆盖 $k$ 维列空间。方阵条件不能省略。

**忘记内积是背景条件。** 本篇的正交矩阵针对标准欧氏内积。若内积写成 $\langle\mathbf{x},\mathbf{y}\rangle_M=\mathbf{x}^{\mathsf T}M\mathbf{y}$，保持它的变换条件是

$$
Q^{\mathsf T}MQ=M
$$

不一定是普通的 $Q^{\mathsf T}Q=I$。换了几何尺子，保持几何的矩阵条件也要跟着换。

**把正交线性变换和刚体运动混为一谈。** $Q\mathbf{x}+\mathbf{b}$ 也保持两点距离，但当 $\mathbf{b}\ne\mathbf{0}$ 时不再是线性映射，因为原点被移走了；它属于仿射等距变换。[仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/) 篇处理平移项。

**把浮点结果当成精确单位阵。** 理论上 $Q^{\mathsf T}Q=I$，数值计算中通常只会得到接近单位阵的结果。重复乘法、梯度更新或低精度存储会积累偏离，工程实现需要重新正交化或采用专门的参数化。

## 相关词条

- [内积](../linear-algebra/inner-products/)：正交矩阵保持的基本运算
- [正交归一基](../linear-algebra/orthonormal-basis/)：矩阵列满足的几何条件
- [正交投影](../linear-algebra/orthogonal-projections/)：投影在正交变换下的相容性
- [线性映射](../linear-algebra/linear-maps/)：矩阵作为映射的坐标表达
- [换基](../linear-algebra/change-of-basis/)：同一向量在不同参照系下的读数
- [行列式](../linear-algebra/determinant/)：用行列式符号区分方向保持与方向翻转
- [仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/)：加上平移后的等距变换
