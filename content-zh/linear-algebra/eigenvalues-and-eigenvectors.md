---
title: "特征值与特征向量:线性变换中的不变方向"
tags: ["why-models-learn"]
---

对方阵 $A$，如果一个非零向量 $\mathbf{v}$ 经过变换后只被整体缩放，没有改变所在直线，那么它满足

$$
A\mathbf{v}=\lambda\mathbf{v}
$$

这里的 $\mathbf{v}$ 叫**特征向量**（eigenvector），缩放因子 $\lambda$ 叫**特征值**（eigenvalue）。一般向量会被旋转、剪切和拉伸，特征向量所在的方向却保持为自己；$\lambda$ 的大小和正负记录了这条方向被怎样改变。本篇从这个方程出发，说明如何求特征值和特征向量、什么时候能把矩阵化为对角矩阵，以及这些数为什么会控制 PCA 和迭代网络的增长。

## 先解一个二维例子

取

$$
A=
\begin{pmatrix}
2&1\\
0&3
\end{pmatrix}
$$

特征方程先把未知的向量换成未知的标量。由

$$
A\mathbf{v}=\lambda\mathbf{v}
\quad\Longleftrightarrow\quad
(A-\lambda I)\mathbf{v}=\mathbf{0}
$$

要有非零解，$A-\lambda I$ 必须把某个方向压到零，因此它必须是奇异矩阵：

$$
\det(A-\lambda I)=0
$$

在这个例子中

$$
\det
\begin{pmatrix}
2-\lambda&1\\
0&3-\lambda
\end{pmatrix}
=(2-\lambda)(3-\lambda)
$$

所以两个特征值是

$$
\lambda_1=2,
\qquad
\lambda_2=3
$$

现在分别代回去求向量。

当 $\lambda_1=2$ 时，

$$
A-2I=
\begin{pmatrix}
0&1\\
0&1
\end{pmatrix}
$$

方程要求第二个分量为零，所以

$$
\mathbf{v}_1=
\begin{pmatrix}
1\\
0
\end{pmatrix}
$$

是一个特征向量。直接核对：

$$
A\mathbf{v}_1=
\begin{pmatrix}
2\\
0
\end{pmatrix}
=2\mathbf{v}_1
$$

当 $\lambda_2=3$ 时，

$$
A-3I=
\begin{pmatrix}
-1&1\\
0&0
\end{pmatrix}
$$

方程要求 $y=x$，可以取

$$
\mathbf{v}_2=
\begin{pmatrix}
1\\
1
\end{pmatrix},
\qquad
A\mathbf{v}_2=
\begin{pmatrix}
3\\
3
\end{pmatrix}
=3\mathbf{v}_2
$$

![特征方向的几何含义：矩阵 A 把两条特殊方向上的向量分别放大 2 倍和 3 倍，方向线保持不变](/assets/linear-algebra/svg/eigenvalues-and-eigenvectors.1.svg)

## 特征向量是一条线，特征值是这条线的缩放因子

若 $\mathbf{v}$ 对应 $\lambda$，任意非零标量 $c$ 都满足

$$
A(c\mathbf{v})=cA\mathbf{v}=c\lambda\mathbf{v}
=\lambda(c\mathbf{v})
$$

所以特征向量通常不是一个单独的箭头，而是一整条特征方向。所有对应 $\lambda$ 的向量连同零向量组成

$$
E_\lambda=\ker(A-\lambda I)
$$

叫作 $\lambda$ 的**特征空间**（eigenspace）。定义特征向量时排除零向量，是因为零向量满足 $A\mathbf{0}=\lambda\mathbf{0}$ 对所有 $\lambda$ 都成立；若不排除，它就无法区分不同特征值。

特征值的不同取值对应不同几何行为：

| 特征值条件 | 对特征方向的作用 | 一维例子 |
| --- | --- | --- |
| $\lambda>1$ | 同方向拉长 | $v\mapsto 3v$ |
| $0<\lambda<1$ | 同方向压缩 | $v\mapsto \frac12v$ |
| $\lambda=1$ | 每个向量保持不动 | $v\mapsto v$ |
| $\lambda=0$ | 压到零向量 | $v\mapsto 0$ |
| $\lambda<0$ | 反向并按 $\lvert\lambda\rvert$ 缩放 | $v\mapsto -2v$ |

例如上面的 $E_2=\operatorname{span}\{(1,0)\}$，$E_3=\operatorname{span}\{(1,1)\}$。矩阵会把一般向量的方向改变，但会把这两条线分别映回自己。

## 一般向量为什么可以拆成特征方向

把两个特征向量排成列：

$$
P=
\begin{pmatrix}
|&|\\
\mathbf{v}_1&\mathbf{v}_2\\
|&|
\end{pmatrix}
=
\begin{pmatrix}
1&1\\
0&1
\end{pmatrix},
\qquad
D=
\begin{pmatrix}
2&0\\
0&3
\end{pmatrix}
$$

矩阵乘法按列作用，因此

$$
AP
=\begin{pmatrix}
A\mathbf{v}_1&A\mathbf{v}_2
\end{pmatrix}
=\begin{pmatrix}
\lambda_1\mathbf{v}_1&\lambda_2\mathbf{v}_2
\end{pmatrix}
=PD
$$

$P$ 的行列式为 $1$，所以它可逆，进而

$$
A=PDP^{-1}
$$

这叫**对角化**（diagonalization）：不是把原矩阵的数字随意抹成对角线，而是换到由特征向量组成的坐标系后，变换变成各坐标独立缩放。

取一个一般向量

$$
\mathbf{x}=
\begin{pmatrix}
3\\
1
\end{pmatrix}
=2\mathbf{v}_1+\mathbf{v}_2
=P
\begin{pmatrix}
2\\
1
\end{pmatrix}
$$

在特征坐标中，$A$ 只需分别乘 $2$ 和 $3$：

$$
\begin{aligned}
A\mathbf{x}
&=PDP^{-1}\mathbf{x}
=P
\begin{pmatrix}
2&0\\
0&3
\end{pmatrix}
\begin{pmatrix}
2\\
1
\end{pmatrix}\\
&=P
\begin{pmatrix}
4\\
3
\end{pmatrix}
=
\begin{pmatrix}
7\\
3
\end{pmatrix}
\end{aligned}
$$

直接相乘也得到 $A(3,1)^{\mathsf T}=(7,3)^{\mathsf T}$。换基没有改变结果，只是把「先混合坐标再观察」改写成「沿特征方向分别缩放」。

## 迭代：特征值控制增长和衰减

对角矩阵的幂很容易计算：

$$
D^k=
\begin{pmatrix}
2^k&0\\
0&3^k
\end{pmatrix}
$$

于是

$$
A^k=PD^kP^{-1}
$$

继续用 $\mathbf{x}=(3,1)^{\mathsf T}=2\mathbf{v}_1+\mathbf{v}_2$：

$$
\begin{aligned}
A^2\mathbf{x}
&=P
\begin{pmatrix}
2^2\cdot2\\
3^2\cdot1
\end{pmatrix}
=P
\begin{pmatrix}
8\\
9
\end{pmatrix}
=
\begin{pmatrix}
17\\
9
\end{pmatrix}
\end{aligned}
$$

因为 $3^k$ 比 $2^k$ 增长更快，只要初始向量在 $\mathbf{v}_2$ 方向上有非零分量，反复应用 $A$ 后，这个方向会越来越占主导。若所有特征值的绝对值都小于 $1$，对应方向会衰减；若有绝对值大于 $1$ 的分量，就会放大。$\lvert\lambda\rvert=1$ 时还要看是否有旋转、负号或不可对角化造成的额外行为，不能只凭一个绝对值下结论。

这条计算解释了线性递推和深层网络中的两种数值问题：特征值模长太小会让信号反复变换后消失，太大会让信号变大。真正的梯度传播由 Jacobian 的乘积控制，特征值是有用的局部诊断，但不是所有矩阵乘积行为的完整描述。

## 迹、行列式与特征值

[迹](../linear-algebra/trace/) 和 [行列式](../linear-algebra/determinant/) 提供了不用重新求全部特征向量的两个总量。若特征值按代数重数列为 $\lambda_1,\dots,\lambda_n$，则

$$
\sum_{i=1}^{n}\lambda_i=\operatorname{tr}(A),
\qquad
\prod_{i=1}^{n}\lambda_i=\det(A)
$$

对本篇的矩阵：

$$
\lambda_1+\lambda_2=2+3=5=\operatorname{tr}(A),
\qquad
\lambda_1\lambda_2=2\cdot3=6=\det(A)
$$

这是 [迹](../linear-algebra/trace/) 篇二维展开的另一种读法；特征多项式的系统定义、系数和根的关系放在 [特征多项式](../linear-algebra/characteristic-polynomial/) 篇。注意，迹和行列式只告诉你特征值的总和与乘积，不能单独告诉你每个特征方向长什么样。

## 不同特征值为什么给出独立方向

如果 $\lambda_1\ne\lambda_2$，对应的特征向量 $\mathbf{v}_1$、$\mathbf{v}_2$ 不可能线性相关。假设它们相关，就存在不全为零的 $c_1$、$c_2$ 使

$$
c_1\mathbf{v}_1+c_2\mathbf{v}_2=\mathbf{0}
$$

两边施加 $A$：

$$
c_1\lambda_1\mathbf{v}_1+c_2\lambda_2\mathbf{v}_2=\mathbf{0}
$$

再减去原等式乘 $\lambda_1$，得到

$$
c_2(\lambda_2-\lambda_1)\mathbf{v}_2=\mathbf{0}
$$

因为 $\lambda_2-\lambda_1\ne0$ 且 $\mathbf{v}_2\ne\mathbf{0}$，所以 $c_2=0$；代回原式得到 $c_1=0$，矛盾。这个论证可以逐步推广：不同特征值对应的特征向量组一定线性无关。

因此 $n\times n$ 矩阵如果有 $n$ 个互不相同的特征值，就一定有一组特征基，可以对角化。反过来，有重复特征值也不代表不能对角化，关键在于是否仍能找到足够多的线性无关特征向量。

## 重复特征值：能否对角化要看特征向量数量

单位矩阵 $I_2$ 只有一个特征值 $\lambda=1$，但每个非零向量都是特征向量，因此有足够多的方向，已经是对角矩阵。

剪切矩阵则不同：

$$
S=
\begin{pmatrix}
1&1\\
0&1
\end{pmatrix}
$$

它的特征多项式是 $(1-\lambda)^2$，唯一的特征值是 $1$。但

$$
(S-I)\mathbf{v}
=
\begin{pmatrix}
0&1\\
0&0
\end{pmatrix}
\begin{pmatrix}
x\\
y
\end{pmatrix}
=\mathbf{0}
\quad\Longrightarrow\quad
y=0
$$

所以它的特征空间只有 $\operatorname{span}\{(1,0)\}$，只能找到一个独立特征向量，不能组成二维特征基，也就不能写成 $PDP^{-1}$。它仍然有特征值，但「沿一组独立方向分别缩放」的坐标解释不够用；剪切造成的偏移会在迭代中留下来。

这里要区分两个计数：

- **代数重数**：特征值作为特征多项式的根重复了几次；
- **几何重数**：对应特征空间的维数，也就是能找到多少个独立特征向量。

可对角化需要每个特征值的几何重数加起来达到空间维数；重复根本身不是问题，缺少特征方向才是问题。

## 实对称矩阵与 PCA

实对称矩阵有额外的好性质。若

$$
A=A^{\mathsf T},
\qquad
A\mathbf{u}=\lambda\mathbf{u},
\qquad
A\mathbf{v}=\mu\mathbf{v}
$$

则对称性给出

$$
\mathbf{u}^{\mathsf T}A\mathbf{v}
=(A\mathbf{u})^{\mathsf T}\mathbf{v}
\quad\Longrightarrow\quad
\mu\mathbf{u}^{\mathsf T}\mathbf{v}
=\lambda\mathbf{u}^{\mathsf T}\mathbf{v}
$$

因此

$$
(\lambda-\mu)\mathbf{u}^{\mathsf T}\mathbf{v}=0
$$

不同特征值的特征向量必然正交。更完整的谱定理还会保证实对称矩阵存在正交归一的特征基，使

$$
A=Q\Lambda Q^{\mathsf T}
$$

其中 $Q$ 的列是正交归一特征向量，$\Lambda$ 是特征值组成的对角矩阵。[谱定理](../linear-algebra/spectral-theorem/) 篇会完整说明这个结论；这里先用它理解 PCA。

设协方差矩阵

$$
\Sigma=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
$$

它的两个特征方向来自 $(1,1)$ 和 $(1,-1)$：

$$
\Sigma
\begin{pmatrix}
1\\
1
\end{pmatrix}
=4
\begin{pmatrix}
1\\
1
\end{pmatrix},
\qquad
\Sigma
\begin{pmatrix}
1\\
-1
\end{pmatrix}
=2
\begin{pmatrix}
1\\
-1
\end{pmatrix}
$$

归一化后，沿第一条方向的方差是 $4$，沿第二条方向的方差是 $2$。PCA 选择最大特征值对应的方向作为第一主成分，因为对单位向量 $\mathbf{v}$：

$$
\operatorname{Var}(\mathbf{v}^{\mathsf T}x)
=\mathbf{v}^{\mathsf T}\Sigma\mathbf{v}
$$

而当 $\Sigma\mathbf{v}=\lambda\mathbf{v}$ 且 $\|\mathbf{v}\|_2=1$ 时，

$$
\mathbf{v}^{\mathsf T}\Sigma\mathbf{v}
=\lambda\mathbf{v}^{\mathsf T}\mathbf{v}
=\lambda
$$

总方差是两个特征值之和 $4+2=6=\operatorname{tr}(\Sigma)$。所以「主方向」「该方向上的方差」「协方差矩阵的特征值」是同一计算的三个表述。

## 实矩阵不一定有实特征方向

在实数范围内，特征方程可能没有实根。90° 旋转

$$
R=
\begin{pmatrix}
0&-1\\
1&0
\end{pmatrix}
$$

满足

$$
\det(R-\lambda I)=\lambda^2+1
$$

它的根是 $\lambda=\mathrm{i}$ 和 $\lambda=-\mathrm{i}$，所以没有实特征值，也没有实特征向量。平面上任何非零实向量都会转向 90°，没有一条实直线映回自己；如果把标量域扩展到复数，复特征方向才出现。

这不是求解失败，而是域的选择带来的事实。PCA 使用实对称协方差矩阵，因而不会遇到这个问题；一般实矩阵做谱分析时，则必须说明是在 $\mathbb{R}$ 还是 $\mathbb{C}$ 中讨论。

## 机器学习里怎样使用这些数

**PCA 与表示压缩。** 协方差特征值按大小排序，就是不同正交方向上的方差排序。保留前 $k$ 个特征向量，相当于保留变化最大的 $k$ 个方向；被丢弃方向的特征值之和是丢失的方差量。

**线性层的反复作用。** 若某个表示沿特征方向分解为 $\sum_i c_i\mathbf{v}_i$，重复应用线性层后变为 $\sum_i c_i\lambda_i^k\mathbf{v}_i$。这给出信号放大、衰减和方向偏置的第一近似，也解释了为什么深层线性链需要关注谱半径。

**梯度传播的诊断。** 一层网络在某点附近由 Jacobian 近似；多层梯度是多个 Jacobian 的乘积。若相关方向的缩放因子持续小于 $1$，梯度可能消失；持续大于 $1$，梯度可能爆炸。特征值可以提示风险，但非对角化矩阵、不同层的特征方向不对齐以及非正规矩阵都可能产生特征值无法单独预测的瞬态增长。

**低秩表示。** 特征空间告诉我们哪些方向被保留、压缩或消失。$\lambda=0$ 的特征方向落入核；很多接近零的特征值意味着某些方向数值上几乎被压扁。实际低秩判断还要结合奇异值和条件数，不能只看一两个特征值。

## 容易混淆的地方

**把零向量当特征向量。** 零向量对所有 $\lambda$ 都满足方程，因而定义中必须要求 $\mathbf{v}\ne\mathbf{0}$。

**把特征值当成矩阵对角线元素。** 对角矩阵或三角矩阵的特征值确实等于对角线元素，但一般矩阵不行。比如

$$
\begin{pmatrix}
1&2\\
3&4
\end{pmatrix}
$$

的特征值是 $\frac{5+\sqrt{33}}2$ 与 $\frac{5-\sqrt{33}}2$，不是 $1$ 和 $4$。

**重复特征值一定不能对角化。** 单位矩阵就是反例；要查的是每个特征空间的维数，而不是只看根有没有重复。

**有特征值就一定能写成 $PDP^{-1}$。** 任何方阵在复数范围内都有特征值，但不一定有足够多的独立特征向量；剪切矩阵正是二维反例。

**只看特征值就判断所有迭代行为。** 可对角化且特征方向稳定时，$\lambda_i^k$ 给出清楚的增长率；不可对角化或非正规时还可能出现多项式因子和短期放大。特征值是谱信息，不是完整的动态轨迹。

**忘记标量域。** 90° 旋转在 $\mathbb{R}$ 中没有特征向量，在 $\mathbb{C}$ 中有复特征值。写结论时要说明讨论的是实数还是复数。

## 相关词条

- [迹](../linear-algebra/trace/)：特征值之和与主对角线总和相等。
- [行列式](../linear-algebra/determinant/)：特征值之积与有向体积因子相等。
- [特征多项式](../linear-algebra/characteristic-polynomial/)：系统构造特征方程并读取根的重数。
- [特征分解](../linear-algebra/eigendecomposition/)：把可对角化矩阵写成特征基下的乘积。
- [谱定理](../linear-algebra/spectral-theorem/)：实对称矩阵的正交特征基。
- [二次型](../linear-algebra/quadratic-forms/)：Rayleigh 商与主方向优化的代数形式。
- [正交归一基](../linear-algebra/orthonormal-basis/)：PCA 和谱分解使用的坐标基础。
- [协方差矩阵](../probability/covariance-matrix/)：PCA 的对称矩阵来源。
- [线性映射](../linear-algebra/linear-maps/)：矩阵作为变换，特征向量描述其不变方向。
