---
title: "矩阵的逆:定义、判据与求法"
tags: ["why-models-learn"]
---

$n \times n$ 矩阵 $A$ 称为**可逆**(invertible)，如果存在矩阵 $B$ 使

$$
AB = BA = I
$$

这样的 $B$ 若存在则唯一(两个逆 $B_1$、$B_2$ 满足 $B_1 = B_1 A B_2 = B_2$)，记作 $A^{-1}$，叫 $A$ 的**逆矩阵**；不可逆的方阵叫**奇异**(singular)矩阵。逆是 undo 的矩阵形态：$A$ 把 $\mathbf{v}$ 送到 $A\mathbf{v}$，$A^{-1}$ 把它原样送回来。[秩零定理](../linear-algebra/rank-nullity/) 篇的方阵推论已经把门槛算清：可逆当且仅当满秩，当且仅当核为零——单射、满射、双射三件事在方阵上同时成立或同时垮掉。

## 判据花名册

方阵 $A$($n \times n$)可逆，与下列每一条等价：

- $\text{秩}(A) = n$(满秩，[秩](../linear-algebra/rank/) 篇)；
- $\ker A = \{\mathbf{0}\}$(核平凡，[核与像](../linear-algebra/kernel-and-image/) 篇)；
- $A\mathbf{x} = \mathbf{b}$ 对每个 $\mathbf{b}$ 有唯一解([线性方程组](../linear-algebra/linear-systems/) 篇)；
- $A$ 的列构成 $\mathbb{R}^n$ 的基([基](../linear-algebra/basis/) 篇)；
- 消元得到 $n$ 个主元([高斯消元](../linear-algebra/gaussian-elimination/) 篇)。

每一条都是已学过的概念换一身衣服。秩零定理保证它们同真同假：不满秩时核里有整条直线，无解与无穷多解一起到来。

## 2×2 公式与行列式的影子

$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ 的逆有闭式：

$$
A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

分母 $ad - bc$ 是 $A$ 的行列式([行列式](../linear-algebra/determinant/) 篇详谈)，它为零正是矩阵奇异的信号。验证 [秩零定理](../linear-algebra/rank-nullity/) 篇的 $B = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$：$\det B = 3 \cdot 2 - 1 \cdot 1 = 5$，

$$
B^{-1} = \frac{1}{5} \begin{pmatrix} 2 & -1 \\ -1 & 3 \end{pmatrix}, \qquad B^{-1}B = I \;(\text{实算确认})
$$

undo 演示：$B^{-1}(4, 6) = \left(\frac{2}{5}, \frac{14}{5}\right)$，再乘 $B$ 得 $(4, 6)$ 原样返回。

## 代数性质：反序那一条最容易错

- $(A^{-1})^{-1} = A$(undo 的 undo)；
- $(AB)^{-1} = B^{-1}A^{-1}$(**反序**：先穿的后脱——先被 $B$ 作用，要最后才解 $B$)；
- $(A^T)^{-1} = (A^{-1})^T$。

反序用数字按一遍。$A = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}$，$A^{-1} = \begin{pmatrix} 1/2 & -1/2 \\ 0 & 1 \end{pmatrix}$；$AB = \begin{pmatrix} 7 & 4 \\ 1 & 2 \end{pmatrix}$，$\det = 10$。实算：$(AB)^{-1} = \frac{1}{10}\begin{pmatrix} 2 & -4 \\ -1 & 7 \end{pmatrix}$，而 $B^{-1}A^{-1}$ 给出同一个结果；顺序写成 $A^{-1}B^{-1} = \frac{1}{10}\begin{pmatrix} 3 & -4 \\ -2 & 6 \end{pmatrix}$ 就不是了。

## 求法：扩增矩阵的消元

$A^{-1}$ 的第 $j$ 列满足 $A\mathbf{x} = \mathbf{e}_j$——求逆就是同时解 $n$ 个方程组，系数矩阵都是 $A$。[高斯消元](../linear-algebra/gaussian-elimination/) 篇的手法照抄：把 $n$ 个右端并排摆上，一次消元全部解决：

$$
[\,A \mid I\,] \;\xrightarrow{\text{消元}}\; [\,I \mid A^{-1}\,]
$$

左半场化成单位阵的时刻，右半场自动变成逆。$B$ 走一遍：

$$
\left[\begin{array}{cc|cc} 3 & 1 & 1 & 0 \\ 1 & 2 & 0 & 1 \end{array}\right] \to \left[\begin{array}{cc|cc} 1 & 0 & 2/5 & -1/5 \\ 0 & 1 & -1/5 & 3/5 \end{array}\right]
$$

右半场 $\frac{1}{5}\begin{pmatrix} 2 & -1 \\ -1 & 3 \end{pmatrix}$，与 2×2 公式一致。消元过程也顺手判了可逆性：左半场化不出单位阵(主元不够 $n$ 个)即奇异。

![高斯-若尔当求逆：左，[B | I]；右，同一轮消元后 [I | B⁻¹]。左手单位阵，右手逆矩阵](/assets/linear-algebra/svg/matrix-inverse.1.svg)

## 神经网络里的逆：概念天天用，矩阵从不算

逆在深度学习里是分析语言而非计算工具。最小二乘的正规方程 $\hat{\mathbf{w}} = (X^TX)^{-1}X^T\mathbf{y}$ 写着逆，实现时没人真求逆——显式求 $A^{-1}$ 再乘，比直接消元解方程更慢、数值误差更大；实务走 QR 或 SVD 分解([最小二乘](../linear-models/least-squares-as-projection/)、[SVD](../linear-algebra/svd/) 篇)。

概念层面逆无处不在：[换基](../linear-algebra/change-of-basis/) 的过渡矩阵互逆、[线性映射](../linear-algebra/linear-maps/) 的相似变换 $B^{-1}SB$、权重衰减可看作给 $X^TX$ 加 $\lambda I$ 让它「更可逆」。读论文见到逆，心里换算成「解一个方程组」就不会被符号唬住。

## 失效模式与常见误区

**以为方阵都可逆。** 奇异矩阵遍地都是：秩亏即不可逆，$W = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}$ 的核里有整条直线，undo 无从谈起。

**$(AB)^{-1}$ 写成 $A^{-1}B^{-1}$。** 顺序必须反过来，数字小节有反例。记忆：先穿的后脱。

**以为 $A^{-1}$ 是逐元素取倒数。** 逆是矩阵方程 $AB = I$ 的解，与逐元素运算无关。

**实务中显式求逆再乘。** 又慢又不稳。见到 $A^{-1}\mathbf{b}$，读作「解 $A\mathbf{x} = \mathbf{b}$」，用消元或矩阵分解。

## 相关词条

- [秩零定理](../linear-algebra/rank-nullity/)：可逆判据同真同假的保证
- [高斯消元](../linear-algebra/gaussian-elimination/)：求逆的算法本体
- [矩阵乘法](../linear-algebra/matrix-multiplication/)：逆的代数结构(结合律、单位元)
- [线性方程组](../linear-algebra/linear-systems/)：$A^{-1}\mathbf{b}$ 的概念意义
- [秩](../linear-algebra/rank/)：满秩判据
- [换基](../linear-algebra/change-of-basis/)：过渡矩阵天然互逆
- [线性映射](../linear-algebra/linear-maps/)：逆映射与相似变换
- [行列式](../linear-algebra/determinant/)：奇异性的数值信号
