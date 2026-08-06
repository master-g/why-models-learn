---
title: "伪逆:秩亏矩阵也能给出最小二乘解"
tags: ["why-models-learn"]
---

对任意实矩阵 $A\in\mathbb{R}^{m\times n}$，都存在一个唯一的矩阵 $A^+\in\mathbb{R}^{n\times m}$，叫**Moore–Penrose 伪逆**（pseudoinverse）。给定输出向量 $\mathbf{b}$，$A^+\mathbf{b}$ 返回一个最小二乘解；如果解不唯一，它还选择其中长度最小的那个。方阵可逆时，伪逆才退化成普通逆矩阵；长方形、秩亏或方程不相容时，伪逆仍然有定义。本篇从四个 Penrose 条件和 SVD 公式开始，再把它解释为两个正交投影，最后连接最小二乘、核空间和数值稳定性。

## 普通逆为什么不够

普通逆矩阵要求 $A$ 是方阵并且满秩：

$$
A^{-1}A=AA^{-1}=I
$$

但机器学习里的矩阵经常不是这种形状。$A$ 可能有更多样本而不是更多特征，可能有重复列导致秩亏，也可能把一个 $\mathbf{b}$ 送不到精确的像空间。此时三个问题会同时出现：

- 方程 $A\mathbf{x}=\mathbf{b}$ 可能无解；
- 即使有解，核中的方向也会让解不唯一；
- 直接写 $(A^{\mathsf T}A)^{-1}A^{\mathsf T}$ 可能因为 $A^{\mathsf T}A$ 奇异或病态而失败。

伪逆把问题改成两个明确的选择：先找离 $\mathbf{b}$ 最近的可达向量，再在所有达到这个最近距离的参数中取最短的那个。

## 四个 Penrose 条件

$A^+$ 由下面四个条件唯一确定：

$$
\begin{aligned}
AA^+A&=A，
&A^+AA^+&=A^+，\\
(AA^+)^{\mathsf T}&=AA^+，
&(A^+A)^{\mathsf T}&=A^+A
\end{aligned}
$$

前两条要求来回作用时不再改变 $A$ 或 $A^+$ 的有效部分，后两条要求 $AA^+$ 和 $A^+A$ 是对称投影，而不是任意的幂等变换。

形状先核对清楚：

$$
A:m\times n，
\qquad
A^+:n\times m，
\qquad
AA^+:m\times m，
\qquad
A^+A:n\times n
$$

当 $A$ 是可逆的 $n\times n$ 方阵时，取 $A^+=A^{-1}$，四个条件都退化为普通逆的性质。伪逆比普通逆多覆盖的部分，正是形状不方、秩不满和解不唯一。

## SVD 直接给出伪逆

设 $A$ 的紧 SVD 为

$$
A=U_r\Sigma_rV_r^{\mathsf T}，
\qquad
\Sigma_r=\operatorname{diag}(\sigma_1,\dots,\sigma_r)，
\qquad
\sigma_i>0
$$

把每个非零奇异值取倒数，再把输入输出方向交换：

$$
A^+=V_r\Sigma_r^{-1}U_r^{\mathsf T}，
\qquad
\Sigma_r^{-1}
=\operatorname{diag}
\left(\frac1{\sigma_1},\dots,\frac1{\sigma_r}\right)
$$

如果用完整 SVD，$\Sigma$ 的零位置仍然对应 $\Sigma^+$ 的零位置：

$$
\sigma_i\longmapsto
\begin{cases}
1/\sigma_i,&\sigma_i>0\\
0,&\sigma_i=0
\end{cases}
$$

零不能取倒数。它代表 $A$ 已经把输入方向压进核，输出中没有信息可以让我们恢复原坐标。

把公式代回四个条件就能看清它为什么成立。完整 SVD 下，

$$
AA^+
=U\Sigma\Sigma^+U^{\mathsf T}，
\qquad
A^+A
=V\Sigma^+\Sigma V^{\mathsf T}
$$

$\Sigma\Sigma^+$ 是输出空间中前 $r$ 个方向的对角投影，$\Sigma^+\Sigma$ 是输入空间中前 $r$ 个方向的对角投影，因此它们都是对称矩阵。再利用

$$
\Sigma\Sigma^+\Sigma=\Sigma，
\qquad
\Sigma^+\Sigma\Sigma^+=\Sigma^+
$$

就得到四个 Penrose 条件。

## 两个满秩特例

SVD 公式在两个满秩情形下会化成熟悉的闭式。

### 满列秩：超定系统

若 $A\in\mathbb{R}^{m\times n}$，$m\ge n$ 且列线性无关，那么 $A^{\mathsf T}A$ 可逆：

$$
A^+=(A^{\mathsf T}A)^{-1}A^{\mathsf T}
$$

因为 $A=U_r\Sigma_rV_r^{\mathsf T}$ 时

$$
A^{\mathsf T}A=V_r\Sigma_r^2V_r^{\mathsf T}，
\qquad
(A^{\mathsf T}A)^{-1}A^{\mathsf T}
=V_r\Sigma_r^{-1}U_r^{\mathsf T}
$$

它为通常的「样本多、方程多」的最小二乘系统选出唯一参数。

### 满行秩：欠定系统

若 $A\in\mathbb{R}^{m\times n}$，$m\le n$ 且行线性无关，那么 $AA^{\mathsf T}$ 可逆：

$$
A^+=A^{\mathsf T}(AA^{\mathsf T})^{-1}
$$

这时 $A\mathbf{x}=\mathbf{b}$ 对每个 $\mathbf{b}$ 都有解，但通常有无穷多个解；伪逆选择长度最小的那一个。

例如

$$
A=\begin{pmatrix}1&2\end{pmatrix}，
\qquad
A^+=\frac15\begin{pmatrix}1\\2\end{pmatrix}
$$

对 $b=3$，

$$
A^+b=\begin{pmatrix}3/5\\6/5\end{pmatrix}，
\qquad
A A^+b=3
$$

所有精确解都满足 $x_1+2x_2=3$。伪逆给出的向量平行于 $(1,2)$，它与核方向 $(2,-1)$ 正交，因此长度最小。

## 伪逆其实是两个投影

SVD 让两个乘积的含义变得直接：

$$
P_{\operatorname{col}(A)}=AA^+，
\qquad
P_{\operatorname{row}(A)}=A^+A
$$

前者是 $\mathbb{R}^m$ 中到 $A$ 的列空间的正交投影，后者是 $\mathbb{R}^n$ 中到 $A$ 的行空间的正交投影。[正交投影](../linear-algebra/orthogonal-projections/) 篇证明了正交投影的三个指纹：对称、幂等、输出落在目标子空间。这里从 SVD 也能逐项看出：

$$
(AA^+)^2=AA^+，
\qquad
(A^+A)^2=A^+A
$$

以及

$$
\operatorname{im}(AA^+)=\operatorname{col}(A)，
\qquad
\operatorname{im}(A^+A)=\operatorname{row}(A)
$$

对任意 $\mathbf{b}$，$AA^+\mathbf{b}$ 是 $\mathbf{b}$ 在列空间中最近的点，残差满足

$$
\mathbf{r}=\mathbf{b}-AA^+\mathbf{b}
\perp\operatorname{col}(A)
$$

所以 $A^+\mathbf{b}$ 是把这个最近点写回参数空间的一个最短坐标。

## 最小二乘与最小范数

考虑任意形状的方程

$$
\min_{\mathbf{x}}\lVert A\mathbf{x}-\mathbf{b}\rVert^2
$$

设 $\hat{\mathbf{x}}$ 是一个最小二乘解。对任意方向 $\mathbf{h}$，令

$$
f(t)=\lVert A(\hat{\mathbf{x}}+t\mathbf{h})-\mathbf{b}\rVert^2
$$

在 $t=0$ 处取得最小值，所以 $f'(0)=0$。展开导数：

$$
f'(0)=2\mathbf{h}^{\mathsf T}A^{\mathsf T}(A\hat{\mathbf{x}}-\mathbf{b})
$$

对所有 $\mathbf{h}$ 都为零，等价于正规方程

$$
A^{\mathsf T}(A\hat{\mathbf{x}}-\mathbf{b})=\mathbf{0}
$$

这只说明残差与列空间正交，并不保证 $\hat{\mathbf{x}}$ 唯一。因为若 $\mathbf{z}\in\ker(A)$，

$$
A(\hat{\mathbf{x}}+\mathbf{z})=A\hat{\mathbf{x}}
$$

所有这些参数都有同样的残差。伪逆挑出

$$
\mathbf{x}^+=A^+\mathbf{b}
$$

这个最小二乘解中的最小范数者：

$$
\mathbf{x}^+\perp\ker(A)，
\qquad
\lVert\mathbf{x}^+\rVert
\le\lVert\hat{\mathbf{x}}\rVert
$$

证明只用正交分解。[秩零定理](../linear-algebra/rank-nullity/) 和基本正交关系给出

$$
\operatorname{row}(A)=\ker(A)^\perp
$$

而 SVD 公式把 $A^+\mathbf{b}$ 放在行空间中。任意同样好的解都可写成 $\mathbf{x}^++\mathbf{z}$，其中 $\mathbf{z}\in\ker(A)$，于是

$$
\lVert\mathbf{x}^++\mathbf{z}\rVert^2
=\lVert\mathbf{x}^+\rVert^2+\lVert\mathbf{z}\rVert^2
\ge\lVert\mathbf{x}^+\rVert^2
$$

等号只在 $\mathbf{z}=\mathbf{0}$ 时成立，所以最小范数解唯一。

## 两个数字例子

### 满列秩：把正交投影的系数写成伪逆

取 [正交投影](../linear-algebra/orthogonal-projections/) 篇的矩阵

$$
A=\begin{pmatrix}
1&1\\
0&1\\
1&0
\end{pmatrix}，
\qquad
\mathbf{b}=\begin{pmatrix}1\\2\\3\end{pmatrix}
$$

它满列秩，且

$$
A^{\mathsf T}A=\begin{pmatrix}2&1\\1&2\end{pmatrix}，
\qquad
(A^{\mathsf T}A)^{-1}
=\frac13\begin{pmatrix}2&-1\\-1&2\end{pmatrix}
$$

所以

$$
A^+=\frac13\begin{pmatrix}
1&-1&2\\
1&2&-1
\end{pmatrix}
$$

作用于 $\mathbf{b}$：

$$
\mathbf{x}^+=A^+\mathbf{b}
=\begin{pmatrix}5/3\\2/3\end{pmatrix}，
\qquad
A\mathbf{x}^+=\begin{pmatrix}7/3\\2/3\\5/3\end{pmatrix}
$$

残差是

$$
\mathbf{r}=\mathbf{b}-A\mathbf{x}^+
=\begin{pmatrix}-4/3\\4/3\\4/3\end{pmatrix}
$$

两列正交检查为

$$
A^{\mathsf T}\mathbf{r}
=\begin{pmatrix}0\\0\end{pmatrix}
$$

这说明伪逆把 $\mathbf{b}$ 投影到了列空间；这里因为列满秩，参数解本身也唯一。

### 秩亏：无穷多参数中选最短者

令

$$
B=\begin{pmatrix}1&1\\1&1\end{pmatrix}
$$

它只有一个非零奇异值。取

$$
\mathbf{u}_1=\mathbf{v}_1=\frac1{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix}，
\qquad
\sigma_1=2
$$

则

$$
B^+=\frac1{\sigma_1}\mathbf{v}_1\mathbf{u}_1^{\mathsf T}
=\frac14\begin{pmatrix}1&1\\1&1\end{pmatrix}
$$

对不在列空间中的 $\mathbf{b}=(1,0)^{\mathsf T}$，

$$
\mathbf{x}^+=B^+\mathbf{b}
=\begin{pmatrix}1/4\\1/4\end{pmatrix}，
\qquad
B\mathbf{x}^+=\begin{pmatrix}1/2\\1/2\end{pmatrix}
$$

残差

$$
\mathbf{r}=\begin{pmatrix}1/2\\-1/2\end{pmatrix}
$$

正好垂直于列空间 $\operatorname{span}\{(1,1)\}$。

为什么还有无穷多个同样好的参数？令 $\mathbf{x}=(x_1,x_2)^{\mathsf T}$，则

$$
B\mathbf{x}=\begin{pmatrix}s\\s\end{pmatrix}，
\qquad
s=x_1+x_2
$$

距离平方为

$$
\left\|B\mathbf{x}-\mathbf{b}\right\|^2
=(s-1)^2+s^2
=2\left(s-\frac12\right)^2+\frac12
$$

只要 $x_1+x_2=1/2$，就得到同样的最小残差；这些解沿着核方向 $(1,-1)$ 组成一条直线。伪逆选择 $x_1=x_2=1/4$，因为它与核方向正交且长度最小。

![伪逆的两个投影：输出向量先投影到列空间，参数空间再取垂直于核方向的最小范数解](/assets/linear-algebra/svg/pseudoinverse.1.svg)

## 伪逆不是数值上的无条件安全

SVD 公式中每个通道都乘 $1/\sigma_i$。如果某个奇异值很小，原本很小的观测误差也会被大幅放大：

$$
\sigma_i\downarrow0
\quad\Longrightarrow\quad
\frac1{\sigma_i}\uparrow\infty
$$

因此实务上通常不先形成普通逆，也不盲目把所有非零数都倒过来。两种常见处理是：

**截断伪逆。** 设阈值为 $\tau$，只对 $\sigma_i>\tau$ 的通道取倒数：

$$
\sigma_i^+
=\begin{cases}
1/\sigma_i,&\sigma_i>\tau\\
0,&\sigma_i\le\tau
\end{cases}
$$

这等于把弱通道当作不可可靠的方向，和 [低秩近似](../linear-algebra/low-rank-approximation/) 的截断相连。

**岭正则化。** 对最小二乘加入参数惩罚：

$$
\mathbf{x}_\lambda
=\arg\min_{\mathbf{x}}
\left(\lVert A\mathbf{x}-\mathbf{b}\rVert^2
 +\lambda\lVert\mathbf{x}\rVert^2\right)
$$

其中 $\lambda>0$，解为

$$
\mathbf{x}_\lambda
=(A^{\mathsf T}A+\lambda I)^{-1}A^{\mathsf T}\mathbf{b}
$$

在奇异方向上，它不是使用 $1/\sigma_i$，而是使用

$$
\frac{\sigma_i}{\sigma_i^2+\lambda}
$$

这个系数在 $\sigma_i$ 很小时不会爆炸，但它也不再是精确伪逆。[矩阵范数](../linear-algebra/matrix-norms/) 篇会进一步说明小奇异值与条件数如何控制误差放大。

## 在线性回归和神经网络中的读法

对中心化的设计矩阵 $X$ 和标签 $\mathbf{y}$，最小二乘的最小范数解可以统一写成

$$
\hat{\mathbf{w}}=X^+\mathbf{y}
$$

如果特征列独立，这就是 $(X^{\mathsf T}X)^{-1}X^{\mathsf T}\mathbf{y}$；如果特征重复或特征数超过样本数，伪逆仍然给出一个明确解，并且选择参数范数最小的那一个。

当一个神经网络层的参数存在核方向时，不同参数可以产生相同的输出。伪逆选择的是这些等价参数中离原点最近的代表；这不等于训练一定会找到它，但它是一个可计算、可比较的基准。线性层的秩、核和像分别说明它能保留多少方向、丢掉哪些方向，以及哪些输出可达。

## 容易混淆的地方

**把伪逆当成逐元素倒数。** $A^+$ 由矩阵的奇异方向和奇异值共同决定，不能对 $A$ 的每个元素取倒数。

**看到 $A^+A=I$ 就当作普通逆。** 只有列满秩时，$A^+A=I_n$；一般情形它是到行空间的投影。类似地，$AA^+=I_m$ 只有行满秩时成立。

**把最小二乘解当作精确解。** $A^+\mathbf{b}$ 只有在 $\mathbf{b}\in\operatorname{col}(A)$ 时才满足 $A\mathbf{x}=\mathbf{b}$；否则只能保证残差最小。

**忘记秩亏时的解不唯一。** 伪逆给的是最小范数解，其他解可沿 $\ker(A)$ 移动而保持同样的预测和残差。

**直接用正规方程处理病态矩阵。** 形成 $A^{\mathsf T}A$ 会把条件数平方，实际计算通常使用 QR 或 SVD；显式形成逆也不是必要步骤。

**对极小奇异值照取倒数。** 先确认这些方向是否由数据可靠支持，必要时使用截断伪逆或岭正则化。

## 相关词条

- [奇异值分解](../linear-algebra/svd/)：从奇异方向和奇异值构造伪逆。
- [正交投影](../linear-algebra/orthogonal-projections/)：解释 $AA^+$ 与最近点的几何意义。
- [最小二乘投影](../linear-models/least-squares-as-projection/)：展开正规方程和残差正交。
- [矩阵的逆](../linear-algebra/matrix-inverse/)：伪逆在可逆方阵上的特例。
- [秩零定理](../linear-algebra/rank-nullity/)：说明核方向导致的非唯一性。
- [核与像](../linear-algebra/kernel-and-image/)：区分可达输出和不可恢复输入。
- [低秩近似](../linear-algebra/low-rank-approximation/)：截断弱奇异通道与伪逆正则化的联系。
- [矩阵范数](../linear-algebra/matrix-norms/)：分析小奇异值造成的误差放大。
