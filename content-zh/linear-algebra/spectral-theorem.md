---
title: "谱定理:实对称矩阵的正交特征分解"
tags: ["why-models-learn"]
---

**谱定理**说：每个实对称矩阵都有一组正交归一的实特征向量作为基。于是对 $A=A^{\mathsf T}$，总能写成

$$
A=Q\Lambda Q^{\mathsf T},
\qquad
Q^{\mathsf T}Q=I
$$

其中 $Q$ 的列是特征向量，$\Lambda$ 是实特征值组成的对角矩阵。一般实矩阵可能没有实特征方向，或者特征向量不够多；对称性把这两个问题同时排除。本篇先证明定理，再把它用于二次型、协方差、PCA、白化和 Hessian 的曲率分析。

## 定理的完整表述

对 $n\times n$ 实矩阵 $A$，如果 $A=A^{\mathsf T}$，那么以下三项都成立；第 2、3 项是同一个正交特征分解事实的两种说法：

1. $A$ 的全部特征值都是实数；
2. $A$ 有 $n$ 个两两正交、长度为 $1$ 的实特征向量；
3. 存在正交矩阵 $Q$ 和实对角矩阵 $\Lambda$，使 $A=Q\Lambda Q^{\mathsf T}$。

如果第 3 项成立，那么 $A^{\mathsf T}=Q\Lambda Q^{\mathsf T}=A$；但「特征值全为实数」单独并不能推出对称性。

这里的重点是「正交归一」：不是随便找一组能对角化的特征向量，而是可以把特征坐标系选成不改变长度、角度和距离的坐标系。[正交矩阵与旋转](../linear-algebra/orthogonal-matrices-and-rotations/) 篇已经证明了 $Q^{\mathsf T}Q=I$ 的几何含义。

## 一个二维例子先看结果

取对称矩阵

$$
A=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
$$

两条特征方向是

$$
\mathbf{q}_1=\frac1{\sqrt2}
\begin{pmatrix}
1\\
1
\end{pmatrix},
\qquad
\mathbf{q}_2=\frac1{\sqrt2}
\begin{pmatrix}
1\\
-1
\end{pmatrix}
$$

直接作用：

$$
A\mathbf{q}_1=4\mathbf{q}_1,
\qquad
A\mathbf{q}_2=2\mathbf{q}_2
$$

两列正交归一，因此

$$
Q=
\frac1{\sqrt2}
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix},
\qquad
\Lambda=
\begin{pmatrix}
4&0\\
0&2
\end{pmatrix},
\qquad
Q^{\mathsf T}Q=I
$$

重构：

$$
Q\Lambda Q^{\mathsf T}
=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
=A
$$

这套坐标变换没有一般特征分解中的斜基问题：$Q^{-1}=Q^{\mathsf T}$，读坐标和换回去都只是正交变换。

![谱定理的二维图景：对称矩阵沿两条正交主方向分别缩放，矩阵写成 QΛQᵀ，右侧椭圆的长短轴就是特征方向](/assets/linear-algebra/svg/spectral-theorem.1.svg)

## 证明第一步：特征值必须是实数

行列式的根在复数范围内总是存在，所以先把实对称矩阵看成复矩阵。取一个复特征对

$$
A\mathbf{z}=\lambda\mathbf{z},
\qquad
\mathbf{z}\in\mathbb{C}^n,
\qquad
\mathbf{z}\ne\mathbf{0}
$$

实对称矩阵满足 $A^*=A$，这里 $*$ 表示共轭转置。于是

$$
\mathbf{z}^*A\mathbf{z}
=(A\mathbf{z})^*\mathbf{z}
$$

把 $A\mathbf{z}=\lambda\mathbf{z}$ 代入两边：

$$
\lambda\,\mathbf{z}^*\mathbf{z}
=\overline{\lambda}\,\mathbf{z}^*\mathbf{z}
$$

因为 $\mathbf{z}^*\mathbf{z}=\sum_i|z_i|^2>0$，只能有

$$
\lambda=\overline{\lambda}
$$

所以 $\lambda$ 是实数。这个计算的核心是对称性：$\mathbf{z}^*A\mathbf{z}$ 必须等于自己的共轭。一般实矩阵没有 $A^*=A$，这个论证就断了，90° 旋转的复特征值正是反例。

还要说明特征向量可以取实数。把 $\mathbf{z}$ 写成

$$
\mathbf{z}=\mathbf{u}+\mathrm{i}\mathbf{w},
\qquad
\mathbf{u},\mathbf{w}\in\mathbb{R}^n
$$

因为 $A$ 和 $\lambda$ 都是实的，

$$
A\mathbf{u}+\mathrm{i}A\mathbf{w}
=\lambda\mathbf{u}+\mathrm{i}\lambda\mathbf{w}
$$

比较实部和虚部：

$$
A\mathbf{u}=\lambda\mathbf{u},
\qquad
A\mathbf{w}=\lambda\mathbf{w}
$$

$\mathbf{z}\ne\mathbf{0}$ 意味着 $\mathbf{u}$、$\mathbf{w}$ 至少有一个非零，选那个非零向量即可得到实特征向量。

## 证明第二步：不同特征值的方向正交

设

$$
A\mathbf{u}=\lambda\mathbf{u},
\qquad
A\mathbf{v}=\mu\mathbf{v},
\qquad
A=A^{\mathsf T}
$$

利用对称性，$\mathbf{u}^{\mathsf T}A\mathbf{v}$ 可以从两边读取：

$$
\begin{aligned}
\mathbf{u}^{\mathsf T}A\mathbf{v}
&=\mathbf{u}^{\mathsf T}(\mu\mathbf{v})
=\mu\,\mathbf{u}^{\mathsf T}\mathbf{v},\\
\mathbf{u}^{\mathsf T}A\mathbf{v}
&=(A\mathbf{u})^{\mathsf T}\mathbf{v}
=(\lambda\mathbf{u})^{\mathsf T}\mathbf{v}
=\lambda\,\mathbf{u}^{\mathsf T}\mathbf{v}
\end{aligned}
$$

相减得到

$$
(\lambda-\mu)\mathbf{u}^{\mathsf T}\mathbf{v}=0
$$

若 $\lambda\ne\mu$，就必须有 $\mathbf{u}^{\mathsf T}\mathbf{v}=0$。所以不同特征值的特征空间天然正交。

如果特征值重复，可以在同一个特征空间里做正交化而不离开它：若 $A\mathbf{u}=\lambda\mathbf{u}$、$A\mathbf{v}=\lambda\mathbf{v}$，那么任何线性组合也满足 $A(c\mathbf{u}+d\mathbf{v})=\lambda(c\mathbf{u}+d\mathbf{v})$。在这个空间里使用 Gram–Schmidt，就能得到正交归一的特征向量。

## 证明第三步：正交补保持不变

先选一个单位特征向量 $\mathbf{q}_1$，满足

$$
A\mathbf{q}_1=\lambda_1\mathbf{q}_1,
\qquad
\|\mathbf{q}_1\|_2=1
$$

考虑它的正交补

$$
W=\{\mathbf{w}\in\mathbb{R}^n:\mathbf{q}_1^{\mathsf T}\mathbf{w}=0\}
$$

任取 $\mathbf{w}\in W$，计算 $A\mathbf{w}$ 与 $\mathbf{q}_1$ 的内积：

$$
\begin{aligned}
\mathbf{q}_1^{\mathsf T}(A\mathbf{w})
&=(A\mathbf{q}_1)^{\mathsf T}\mathbf{w}\\
&=(\lambda_1\mathbf{q}_1)^{\mathsf T}\mathbf{w}
=\lambda_1\mathbf{q}_1^{\mathsf T}\mathbf{w}
=0
\end{aligned}
$$

因此 $A\mathbf{w}\in W$：$A$ 把正交补映回正交补。限制在 $W$ 上的变换仍然是对称的，而 $W$ 的维数从 $n$ 降为 $n-1$。

对 $n$ 做归纳。$n=1$ 时结论显然；假设 $(n-1)$ 维的实对称变换有正交归一特征基，那么把这个结论应用到 $W$，就得到 $\mathbf{q}_2,\dots,\mathbf{q}_n$。它们都与 $\mathbf{q}_1$ 正交，彼此也正交，合在一起就是全空间的正交归一特征基。

令

$$
Q=
\begin{pmatrix}
|&|&&|\\
\mathbf{q}_1&\mathbf{q}_2&\cdots&\mathbf{q}_n\\
|&|&&|
\end{pmatrix},
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n)
$$

逐列作用给出

$$
AQ=Q\Lambda
$$

由于 $Q$ 正交，$Q^{-1}=Q^{\mathsf T}$，于是

$$
A=Q\Lambda Q^{\mathsf T}
$$

这完成了谱定理的证明。证明中真正起作用的不是某个二次公式，而是两件事：对称性让正交补保持不变，归纳让特征方向一条条填满整个空间。

## 谱展开：每条方向各自贡献一项

设 $Q$ 的列为 $\mathbf{q}_1,\dots,\mathbf{q}_n$。因为

$$
Q\Lambda Q^{\mathsf T}
=\sum_{i=1}^n\lambda_i\mathbf{q}_i\mathbf{q}_i^{\mathsf T}
$$

所以对任意 $\mathbf{x}$：

$$
A\mathbf{x}
=\sum_{i=1}^n\lambda_i\mathbf{q}_i\mathbf{q}_i^{\mathsf T}\mathbf{x}
$$

$\mathbf{q}_i^{\mathsf T}\mathbf{x}$ 是 $\mathbf{x}$ 在第 $i$ 个正交方向上的坐标。先投影到这条方向，再乘 $\lambda_i$，最后把所有方向相加。每一项的投影矩阵

$$
\Pi_i=\mathbf{q}_i\mathbf{q}_i^{\mathsf T}
$$

满足

$$
\Pi_i^{\mathsf T}=\Pi_i,
\qquad
\Pi_i^2=\Pi_i,
\qquad
\Pi_i\Pi_j=0\quad(i\ne j)
$$

因此对称矩阵的谱投影同时是正交投影；一般特征分解中的 $P E_iP^{-1}$ 只有幂等性，不一定对称。

同一套坐标还给出

$$
A^k=Q\Lambda^kQ^{\mathsf T},
\qquad
f(A)=Qf(\Lambda)Q^{\mathsf T}
$$

其中 $f(\Lambda)$ 逐个作用在对角线特征值上。由于 $Q$ 和 $Q^{\mathsf T}$ 保持长度，谱展开把矩阵运算真正拆成互不干扰的方向。

## 对称矩阵的正定性和平方根

对任意 $\mathbf{x}$，令 $\mathbf{z}=Q^{\mathsf T}\mathbf{x}$。则

$$
\begin{aligned}
\mathbf{x}^{\mathsf T}A\mathbf{x}
&=\mathbf{x}^{\mathsf T}Q\Lambda Q^{\mathsf T}\mathbf{x}\\
&=\mathbf{z}^{\mathsf T}\Lambda\mathbf{z}
=\sum_i\lambda_i z_i^2
\end{aligned}
$$

因此：

- $A$ 半正定，当且仅当所有 $\lambda_i\ge0$；
- $A$ 正定，当且仅当所有 $\lambda_i>0$；
- $A$ 有负特征值，当且仅当存在方向使二次型为负；
- $A$ 可逆，当且仅当没有特征值为 $0$。

这里的半正定和正定会在 [二次型](../linear-algebra/quadratic-forms/) 篇中系统定义；谱定理先把它们化成对角线上的逐项判断。

如果 $A$ 半正定，可以定义对称平方根：

$$
A^{1/2}=Q\Lambda^{1/2}Q^{\mathsf T},
\qquad
\Lambda^{1/2}=\operatorname{diag}(\sqrt{\lambda_1},\dots,\sqrt{\lambda_n})
$$

因为

$$
A^{1/2}A^{1/2}
=Q\Lambda^{1/2}Q^{\mathsf T}Q\Lambda^{1/2}Q^{\mathsf T}
=Q\Lambda Q^{\mathsf T}
=A
$$

同理，正定矩阵的逆为

$$
A^{-1}=Q\Lambda^{-1}Q^{\mathsf T}
$$

小特征值对应的 $\lambda_i^{-1}$ 很大，这也是协方差逆矩阵和白化对噪声敏感的来源。

## 对协方差和 PCA 的解释

协方差矩阵 $\Sigma$ 满足 $\Sigma=\Sigma^{\mathsf T}$ 且半正定，所以谱定理直接给出

$$
\Sigma=Q\Lambda Q^{\mathsf T},
\qquad
\lambda_i\ge0
$$

对前面的

$$
\Sigma=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
$$

有

$$
\lambda_1=4,
\qquad
\lambda_2=2,
\qquad
\operatorname{tr}(\Sigma)=4+2=6
$$

第一主方向 $\mathbf{q}_1=(1,1)^{\mathsf T}/\sqrt2$ 的方差为 $4$，第二主方向的方差为 $2$。因此只保留第一列的 PCA 近似为

$$
\Sigma_1
=4\mathbf{q}_1\mathbf{q}_1^{\mathsf T}
=
\begin{pmatrix}
2&2\\
2&2
\end{pmatrix}
$$

总方差中被保留的比例是

$$
\frac{\lambda_1}{\lambda_1+\lambda_2}
=\frac46
=\frac23
$$

如果把样本写成 $x$，主坐标是

$$
z=Q^{\mathsf T}x
$$

白化则进一步做

$$
\widetilde z=\Lambda^{-1/2}Q^{\mathsf T}x
$$

把每个主方向的方差缩放到 $1$。舍弃小特征值是压缩；除以小特征值是放大噪声，两个操作不能混为一谈。

## 一般矩阵与对称矩阵的差别

| 性质 | 一般实矩阵 | 实对称矩阵 |
| --- | --- | --- |
| 特征值是否全为实数 | 不保证 | 保证 |
| 是否有完整实特征基 | 可能没有 | 一定有 |
| 特征向量是否可选正交归一 | 可能不行 | 可以 |
| 对角化形式 | $P\Lambda P^{-1}$（若可对角化） | $Q\Lambda Q^{\mathsf T}$（总能做到） |
| 谱投影是否正交 | 不保证对称 | 是正交投影 |

例如

$$
\begin{pmatrix}
2&1\\
0&3
\end{pmatrix}
$$

虽然能对角化，但两个特征方向不垂直；剪切矩阵没有完整特征基；90° 旋转在实数范围内没有特征方向。它们都不能享受谱定理的全部结论。

## 机器学习里的谱定理

**PCA 与降维。** 协方差矩阵的正交特征分解让主坐标互不相关，按特征值排序后截断即可控制保留的方差。正交性保证坐标变换本身不改变样本的总欧氏长度。

**白化。** $Q^{\mathsf T}$ 去掉相关性，$\Lambda^{-1/2}$ 调整每个方向的尺度。协方差特征值太小的方向会让白化放大测量误差，通常需要加小的正则项。

**Hessian 与曲率。** 在一个极小值附近，损失的 Hessian 是实对称矩阵。它的特征向量是局部曲率方向，特征值是沿这些方向的二阶曲率：正值表示向上弯，负值表示向下弯，接近零表示平坦或未约束方向。二次型的正式展开在下一篇处理。

**对称权重和相似度。** 某些核矩阵、Gram 矩阵和对称相似度矩阵可以用谱分解读出主方向；但普通神经网络权重一般不对称，不能把谱定理的正交结论自动套上去。

## 容易混淆的地方

**把对称矩阵的结论推广给所有矩阵。** 「特征值全实」「特征向量正交」「一定能正交对角化」都需要对称性或复数域的 Hermitian 条件；一般矩阵没有这些保证。

**只验证 $A=Q\Lambda Q^{\mathsf T}$，不验证 $Q^{\mathsf T}Q=I$。** 任意可逆 $P$ 都可能给出 $P\Lambda P^{-1}$，但只有正交 $Q$ 才是谱定理的结论。

**把半正定误读成所有元素非负。** $A$ 半正定检查的是 $\mathbf{x}^{\mathsf T}A\mathbf{x}\ge0$，等价条件是特征值非负；非对角元素可以是负数。

**把特征值全正当成数值计算永远稳定。** 最小特征值很小时，逆和白化会放大误差；还要看条件数和输入噪声。

**把 PCA 的丢弃和白化的除法混为一谈。** 小特征值方向在降维中可能被删掉，在白化中却会被放大；目标不同，操作相反。

**忽略内积和标量域。** 本篇的正交、转置和实特征基都以 $\mathbb{R}^n$ 的标准内积为准；换成复数要使用共轭转置，换成别的内积也要重新说明对称性。

## 相关词条

- [特征分解](../linear-algebra/eigendecomposition/)：一般可对角化矩阵的 $P\Lambda P^{-1}$。
- [特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/)：谱定理使用的根和方向。
- [特征多项式](../linear-algebra/characteristic-polynomial/)：特征值作为多项式根的代数来源。
- [正交矩阵与旋转](../linear-algebra/orthogonal-matrices-and-rotations/)：$Q^{\mathsf T}Q=I$ 的几何含义。
- [正交归一基](../linear-algebra/orthonormal-basis/)：正交特征基的坐标基础。
- [二次型](../linear-algebra/quadratic-forms/)：正定性与曲率的代数表达。
- [协方差矩阵](../probability/covariance-matrix/)：PCA 和白化的输入。
- [SVD](../linear-algebra/svd/)：一般矩阵的谱量和低秩近似。
