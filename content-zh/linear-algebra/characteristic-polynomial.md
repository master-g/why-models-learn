---
title: "特征多项式:把矩阵的谱信息编码进一个多项式"
tags: ["why-models-learn"]
---

对 $n\times n$ 方阵 $A$，定义

$$
p_A(t)=\det(tI-A)
$$

这就是 $A$ 的**特征多项式**（characteristic polynomial）。它是一个首项系数为 $1$、次数为 $n$ 的多项式；它的根正好是特征值，根重复的次数是特征值的代数重数。矩阵的 $n^2$ 个元素被压缩成了 $n+1$ 个多项式系数，但最高次项、次高次项和常数项分别携带了维数、迹和行列式等结构信息。本篇固定这个符号约定，展开系数、根的重数、相似不变量和用多项式约束矩阵幂的方法。

## 从特征方程到特征多项式

特征向量方程是

$$
A\mathbf{v}=\lambda\mathbf{v},
\qquad
\mathbf{v}\ne\mathbf{0}
$$

移项后

$$
(A-\lambda I)\mathbf{v}=\mathbf{0}
$$

要有非零解，$A-\lambda I$ 必须不可逆，所以

$$
\det(A-\lambda I)=0
$$

本篇把未知标量写作 $t$，并选择

$$
p_A(t)=\det(tI-A)
$$

作为标准写法。对于 $n\times n$ 矩阵，

$$
\det(A-tI)=(-1)^n p_A(t)
$$

两者的根完全相同，但当 $n$ 为奇数时，两个多项式整体差一个负号。求根时这个负号不会改变答案，比较系数或写 Cayley–Hamilton 等式时却必须从头到尾使用同一种约定。

对

$$
A=
\begin{pmatrix}
2&1\\
0&3
\end{pmatrix}
$$

直接计算：

$$
\begin{aligned}
p_A(t)
&=\det
\begin{pmatrix}
t-2&-1\\
0&t-3
\end{pmatrix}\\
&=(t-2)(t-3)
=t^2-5t+6
\end{aligned}
$$

因此根是 $2$ 和 $3$，与 [特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/) 篇逐个解出的特征值一致。

![特征多项式把矩阵压缩成多项式：二维矩阵先得到 p_A(t)=t²−5t+6，再从因式读取根 2 和 3](/assets/linear-algebra/svg/characteristic-polynomial.1.svg)

## 二阶矩阵：三个系数各自表示什么

设

$$
A=
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix}
$$

则

$$
\begin{aligned}
p_A(t)
&=\det
\begin{pmatrix}
t-a&-b\\
-c&t-d
\end{pmatrix}\\
&=(t-a)(t-d)-bc\\
&=t^2-(a+d)t+(ad-bc)\\
&=t^2-\operatorname{tr}(A)t+\det(A)
\end{aligned}
$$

这个短式同时连接了三件事：

1. $t^2$ 的系数是 $1$，对应矩阵所在空间的维数为 $2$；
2. $t$ 的系数是 $-\operatorname{tr}(A)$，根的和是 $\operatorname{tr}(A)$；
3. 常数项是 $\det(A)$，根的积是 $\det(A)$。

取

$$
B=
\begin{pmatrix}
1&2\\
3&4
\end{pmatrix}
$$

有 $\operatorname{tr}(B)=5$、$\det(B)=-2$，所以

$$
p_B(t)=t^2-5t-2
$$

解二次方程：

$$
t=\frac{5\pm\sqrt{25+8}}2
=\frac{5\pm\sqrt{33}}2
$$

两个根的和仍为 $5$，乘积为 $-2$。这说明「矩阵对角线上的数字」通常不是特征值；这里对角线是 $1、4$，特征值却是 $\frac{5+\sqrt{33}}2$ 和 $\frac{5-\sqrt{33}}2$。

## 三阶矩阵：中间系数来自主子式

对

$$
A=
\begin{pmatrix}
a&b&c\\
d&e&f\\
g&h&i
\end{pmatrix}
$$

展开 $\det(tI-A)$，得到

$$
\begin{aligned}
p_A(t)
={}&t^3-(a+e+i)t^2\\
&+(ae-bd+ai-cg+ei-fh)t-\det(A)
\end{aligned}
$$

中间的系数可以按三个主 $2\times2$ 子式分组：

$$
\underbrace{\det\begin{pmatrix}a&b\\d&e\end{pmatrix}}_{ae-bd}
+\underbrace{\det\begin{pmatrix}a&c\\g&i\end{pmatrix}}_{ai-cg}
+\underbrace{\det\begin{pmatrix}e&f\\h&i\end{pmatrix}}_{ei-fh}
$$

所以三阶的结构是

$$
p_A(t)
=t^3-\operatorname{tr}(A)t^2
+(\text{三个主二阶子式之和})t-\det(A)
$$

取一个具体矩阵：

$$
C=
\begin{pmatrix}
1&2&0\\
0&1&1\\
2&0&1
\end{pmatrix}
$$

它的迹为 $3$，三个主二阶子式都是 $1$，行列式为 $5$，因此

$$
p_C(t)=t^3-3t^2+3t-5=(t-1)^3-4
$$

不必先把三次方程的根写成复杂的根式，也能从多项式读出：三个根的和为 $3$，三个根的乘积为 $5$。高维矩阵的中间系数继续由不同大小的主子式组合而成；全部系数的系统展开可以交给行列式的排列公式或余子式计算，不需要把每个根都手算出来。

## 根的重数：代数重数与几何重数

如果

$$
p_A(t)=(t-2)^2(t-3)
$$

那么 $2$ 是二重根，代数重数为 $2$；$3$ 是一重根，代数重数为 $1$。代数重数统计的是根在多项式里出现了几次，不直接告诉我们能找到多少个独立特征向量。

看两个二阶矩阵：

$$
D=
\begin{pmatrix}
2&0\\
0&2
\end{pmatrix},
\qquad
S=
\begin{pmatrix}
2&1\\
0&2
\end{pmatrix}
$$

两者都有

$$
p_D(t)=p_S(t)=(t-2)^2
$$

但 $D$ 的特征空间是整个 $\mathbb{R}^2$，几何重数为 $2$；对 $S$，

$$
(S-2I)\mathbf{v}
=\begin{pmatrix}0&1\\0&0\end{pmatrix}
\begin{pmatrix}x\\y\end{pmatrix}
=\mathbf{0}
\quad\Longrightarrow\quad y=0
$$

特征空间只有 $\operatorname{span}\{(1,0)\}$，几何重数为 $1$。两种重数满足

$$
1\le\text{几何重数}\le\text{代数重数}
$$

可对角化要求每个根的几何重数等于代数重数。特征多项式能告诉你根和代数重数，却必须再解零空间才能知道几何重数；这就是为什么「知道所有特征值」还不足以决定对角化。

## 相似变换不改变特征多项式

若 $A'=P^{-1}AP$，其中 $P$ 可逆，那么

$$
\begin{aligned}
p_{A'}(t)
&=\det(tI-P^{-1}AP)\\
&=\det\left(P^{-1}(tI-A)P\right)\\
&=\det(P^{-1})\det(tI-A)\det(P)\\
&=p_A(t)
\end{aligned}
$$

中间的等式利用了 $tI=P^{-1}(tI)P$，以及行列式的乘法性。于是换基会改变矩阵元素和特征向量的坐标，却不会改变特征多项式、特征值及其代数重数。[换基](../linear-algebra/change-of-basis/) 改的是表示，不是映射本身。

这个不变量比「看矩阵对角线」可靠得多。相似矩阵的对角线可能完全不同，但它们的特征多项式相同；迹和行列式正是这个多项式的两个系数读数。

## Cayley–Hamilton：矩阵满足自己的特征多项式

**Cayley–Hamilton 定理**说：把特征多项式里的标量变量 $t$ 换成矩阵 $A$，并把常数换成常数乘单位矩阵，就得到零矩阵：

$$
p_A(A)=0
$$

对二阶矩阵，这句话具体是

$$
A^2-\operatorname{tr}(A)A+\det(A)I=0
$$

仍用

$$
A=
\begin{pmatrix}
2&1\\
0&3
\end{pmatrix}
$$

有

$$
A^2=
\begin{pmatrix}
4&5\\
0&9
\end{pmatrix},
\qquad
5A=
\begin{pmatrix}
10&5\\
0&15
\end{pmatrix},
\qquad
6I=
\begin{pmatrix}
6&0\\
0&6
\end{pmatrix}
$$

逐项相减：

$$
A^2-5A+6I
=
\begin{pmatrix}
4&5\\
0&9
\end{pmatrix}
-
\begin{pmatrix}
10&5\\
0&15
\end{pmatrix}
+
\begin{pmatrix}
6&0\\
0&6
\end{pmatrix}
=
\begin{pmatrix}
0&0\\
0&0
\end{pmatrix}
$$

因此所有更高次的幂都可以降阶。例如

$$
A^2=5A-6I,
\qquad
A^3=5A^2-6A
=19A-30I
$$

定理的用处不是把一个漂亮的恒等式再背一遍，而是给矩阵幂提供有限阶递推：$n\times n$ 矩阵的 $n$ 次幂及以上，都能用 $I、A、\dots,A^{n-1}$ 的线性组合表示。在线性递推、离散动力系统和某些矩阵函数的计算里，这个降阶很有用。

这里的 $p_A(A)$ 是**矩阵多项式**：$t^2-5t+6$ 被解释为 $A^2-5A+6I$。它不是把 $A$ 放进 $\det(tI-A)$ 后得到 $\det(A-A)$；后者是把矩阵当成标量代入的错误读法。

## 特征多项式与机器学习

**PCA 的谱信息。** 协方差矩阵 $\Sigma$ 的特征多项式把主方向和方差编码成根。因为 $\Sigma$ 是半正定矩阵，所有实特征值非负；特征值之和是 $\operatorname{tr}(\Sigma)$，表示总方差，特征值的排序决定 PCA 保留哪些方向。[特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/) 篇处理了方向的几何含义，本篇补上「这些方向如何作为多项式的根出现」。

**线性递推的降阶。** 如果一个状态按 $\mathbf{x}_{k+1}=A\mathbf{x}_k$ 更新，Cayley–Hamilton 定理能把 $A^n$ 及以上的作用改写成前 $n$ 次幂的组合。它给出关于状态序列的有限阶线性递推；在分析稳定性时，递推的根仍然是特征值。

**Jacobian 的局部谱。** 神经网络在某个输入附近的 Jacobian $J$ 也有特征多项式。其根描述局部不变方向的伸缩因子，根的模长可以提示信号和梯度的放大或衰减。实际数值计算通常采用 QR 等稳定的特征值算法，而不是先展开高阶行列式再用通用多项式求根；高阶系数容易发生消去，重复根附近还会对舍入误差敏感。

**结构检查。** 对称协方差、正交变换和某些受约束的权重矩阵拥有额外的根结构。比如正交矩阵的特征值模长为 $1$；协方差矩阵的特征值为非负实数。多项式可以快速检查候选结果是否违反这些结构，但不能代替对特征向量和矩阵本身的检查。

## 容易混淆的地方

**混用两个符号约定。** $p_A(t)=\det(tI-A)$ 和 $\det(A-tI)$ 的根相同，奇数维时整体符号不同。系数比较时先选定一个约定，再从定义一路展开。

**把根的重数当成特征向量个数。** $(t-2)^2$ 只说明代数重数为 $2$；几何重数要通过 $\ker(A-2I)$ 的维数计算，剪切矩阵说明两者可以不同。

**把矩阵变量代入得行列式。** $p_A(A)$ 是矩阵多项式，常数项要写成常数乘 $I$；它不是 $\det(A-A)$。

**把特征多项式当成矩阵的完整指纹。** 相似矩阵一定有相同特征多项式，但同一个多项式也可能对应不相似的矩阵。$D$ 和 $S$ 都有 $(t-2)^2$，对角化行为却不同。

**把高阶展开当成数值算法。** 纸笔推导适合看系数结构，浮点数下直接展开行列式和求多项式根可能放大误差；生产代码应使用专门的谱算法，并检查残差 $\|A\mathbf{v}-\lambda\mathbf{v}\|$。

**忘记实数域的边界。** 实矩阵的特征多项式可以有复根。讨论 PCA 时特征值来自实对称协方差矩阵；讨论一般矩阵时，要说明是否允许复数根。

## 相关词条

- [特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/)：多项式的根对应不变方向。
- [迹](../linear-algebra/trace/)：次高次项的系数与特征值总和。
- [行列式](../linear-algebra/determinant/)：常数项与特征值乘积。
- [特征分解](../linear-algebra/eigendecomposition/)：从根和零空间构造特征基。
- [谱定理](../linear-algebra/spectral-theorem/)：实对称矩阵的正交谱分解。
- [换基](../linear-algebra/change-of-basis/)：相似变换为什么保持特征多项式。
- [二次型](../linear-algebra/quadratic-forms/)：协方差特征值与方向方差的优化。
- [协方差矩阵](../probability/covariance-matrix/)：PCA 的谱结构来源。
