---
title: "矩阵范数:给线性映射量长度"
tags: ["why-models-learn"]
---

矩阵范数给矩阵一个非负实数，用来描述它的大小。不同范数回答的问题不同：诱导范数问一个线性映射最多把向量放大多少，Frobenius 范数把所有元素当作一个整体来量，谱范数只看最大的奇异值，核范数则把所有奇异值加起来。本篇从范数公理和诱导范数出发，推导常用矩阵范数的公式，再用同一个具体矩阵核对它们的读法，最后连接矩阵乘法、误差传播、条件数和机器学习中的正则化。

## 矩阵范数要满足什么

先回忆向量范数。对向量 $\mathbf{x}$，范数 $\lVert\mathbf{x}\rVert$ 是它的长度，但不一定只能用欧氏长度。矩阵范数也要满足四条公理：

$$
\begin{aligned}
\lVert A\rVert&\ge 0
&&\text{且}\quad \lVert A\rVert=0\Longleftrightarrow A=0\\
\lVert cA\rVert&=|c|\lVert A\rVert\\
\lVert A+B\rVert&\le \lVert A\rVert+\lVert B\rVert\\
\lVert AB\rVert&\le \lVert A\rVert\lVert B\rVert
\end{aligned}
$$

前三条是普通范数的非负性、齐次性和三角不等式。最后一条叫**次乘性**：先做 $B$ 再做 $A$，整体的放大能力不会超过两次放大能力相乘。机器学习里反复相乘的权重矩阵正需要这个上界。

矩阵范数有两种常见来源。若把矩阵的元素逐个相加或平方后相加，它是**元素范数**；若把矩阵看成线性映射，比较输入输出向量的长度，它是**诱导范数**。同一个下标不代表同一种读法：矩阵的 $1$ 范数不是把全部元素绝对值相加，而是最大的列绝对值和。

## 诱导范数：看最坏长度放大

给定向量的 $p$ 范数，矩阵诱导 $p$ 范数定义为单位球上最大的放大倍数：

$$
\lVert A\rVert_p
=\sup_{\mathbf{x}\ne\mathbf{0}}
\frac{\lVert A\mathbf{x}\rVert_p}{\lVert\mathbf{x}\rVert_p}
=\max_{\lVert\mathbf{x}\rVert_p=1}
\lVert A\mathbf{x}\rVert_p
$$

定义立刻给出一个可用的误差上界：

$$
\lVert A\mathbf{x}\rVert_p
\le
\lVert A\rVert_p\lVert\mathbf{x}\rVert_p
$$

把它连续用两次，就得到次乘性：

$$
\lVert AB\mathbf{x}\rVert_p
\le
\lVert A\rVert_p\lVert B\mathbf{x}\rVert_p
\le
\lVert A\rVert_p\lVert B\rVert_p\lVert\mathbf{x}\rVert_p
$$

对所有单位向量取最大值，便有 $\lVert AB\rVert_p\le\lVert A\rVert_p\lVert B\rVert_p$。

### 矩阵 $1$ 范数：最大的列和

设 $A=(a_{ij})\in\mathbb{R}^{m\times n}$。对任意 $\mathbf{x}$，

$$
\begin{aligned}
\lVert A\mathbf{x}\rVert_1
&=\sum_i\left|\sum_j a_{ij}x_j\right|\\
&\le\sum_i\sum_j|a_{ij}||x_j|\\
&=\sum_j\left(\sum_i|a_{ij}|\right)|x_j|\\
&\le\left(\max_j\sum_i|a_{ij}|\right)\lVert\mathbf{x}\rVert_1
\end{aligned}
$$

取某一列绝对值和最大的标准基向量 $\mathbf{e}_j$，上面的上界恰好取到。因此

$$
\lVert A\rVert_1
=\max_j\sum_i|a_{ij}|
$$

它回答的是：输入坐标的总绝对量，经过 $A$ 后最多放大多少。

### 矩阵 $\infty$ 范数：最大的行和

对 $\infty$ 范数，单个输出坐标由一行控制：

$$
\begin{aligned}
\lVert A\mathbf{x}\rVert_\infty
&=\max_i\left|\sum_j a_{ij}x_j\right|\\
&\le\max_i\sum_j|a_{ij}||x_j|\\
&\le\left(\max_i\sum_j|a_{ij}|\right)\lVert\mathbf{x}\rVert_\infty
\end{aligned}
$$

选取绝对值和最大的那一行，并令输入各坐标取相应符号，等号可以取到。所以

$$
\lVert A\rVert_\infty
=\max_i\sum_j|a_{ij}|
$$

转置会交换行和列：

$$
\lVert A\rVert_1=\lVert A^{\mathsf T}\rVert_\infty
$$

## Frobenius 范数：把元素全部加起来

Frobenius 范数把矩阵摊平成一个长向量后取欧氏长度：

$$
\lVert A\rVert_{\mathrm F}
=\sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}a_{ij}^2}
=\sqrt{\operatorname{tr}(A^{\mathsf T}A)}
$$

迹公式只是把元素平方和换一种写法。若 $A$ 的每一列是 $\mathbf{a}_j$，那么

$$
\lVert A\rVert_{\mathrm F}^2
=\sum_j\lVert\mathbf{a}_j\rVert_2^2
$$

它也满足次乘性。对 $C=AB$ 的每个元素使用 Cauchy–Schwarz 不等式：

$$
\begin{aligned}
c_{ij}^2
&=\left(\sum_k a_{ik}b_{kj}\right)^2\\
&\le\left(\sum_k a_{ik}^2\right)
\left(\sum_k b_{kj}^2\right)
\end{aligned}
$$

对 $i,j$ 求和并分离两组下标：

$$
\begin{aligned}
\lVert AB\rVert_{\mathrm F}^2
&\le\sum_{i,j}
\left(\sum_k a_{ik}^2\right)
\left(\sum_k b_{kj}^2\right)\\
&=\left(\sum_{i,k}a_{ik}^2\right)
\left(\sum_{k,j}b_{kj}^2\right)\\
&=\lVert A\rVert_{\mathrm F}^2
\lVert B\rVert_{\mathrm F}^2
\end{aligned}
$$

所以 $\lVert AB\rVert_{\mathrm F}\le\lVert A\rVert_{\mathrm F}\lVert B\rVert_{\mathrm F}$。它会把每个元素都计入，但不直接告诉我们哪个输入方向最危险。

## 谱范数就是最大奇异值

谱范数也叫算子 $2$ 范数，是欧氏长度下的诱导范数。写出紧 SVD：

$$
A=U_r\Sigma_rV_r^{\mathsf T}
$$

其中 $\Sigma_r=\operatorname{diag}(\sigma_1,\dots,\sigma_r)$，并按 $\sigma_1\ge\cdots\ge\sigma_r>0$ 排序。令 $\mathbf{z}=V_r^{\mathsf T}\mathbf{x}$。由于 $U_r$ 和 $V_r$ 的列正交，

$$
\begin{aligned}
\lVert A\mathbf{x}\rVert_2^2
&=\lVert U_r\Sigma_rV_r^{\mathsf T}\mathbf{x}\rVert_2^2\\
&=\sum_{i=1}^{r}\sigma_i^2z_i^2\\
&\le\sigma_1^2\sum_{i=1}^{r}z_i^2\\
&\le\sigma_1^2\lVert\mathbf{x}\rVert_2^2
\end{aligned}
$$

沿着最大右奇异向量 $\mathbf{v}_1$ 取输入时等号成立，因此

$$
\lVert A\rVert_2=\sigma_1
$$

同一组奇异值还给出

$$
\lVert A\rVert_{\mathrm F}^2
=\sum_{i=1}^{r}\sigma_i^2
$$

于是谱范数与 Frobenius 范数之间有

$$
\lVert A\rVert_2
\le\lVert A\rVert_{\mathrm F}
\le\sqrt{r}\lVert A\rVert_2
$$

左边只看最大的奇异方向，右边把最多 $r$ 个奇异方向的能量全部算入。

## 核对一个矩阵的五把尺子

取

$$
A=\begin{pmatrix}
3&4\\
0&0
\end{pmatrix}
$$

它只有一行非零，所以可以直接算出：

$$
\lVert A\rVert_1=4
\qquad
\lVert A\rVert_\infty=7
\qquad
\lVert A\rVert_{\max}=4
$$

对谱范数，任取 $\mathbf{x}=(x_1,x_2)^{\mathsf T}$：

$$
\lVert A\mathbf{x}\rVert_2
=|3x_1+4x_2|
\le\sqrt{3^2+4^2}\lVert\mathbf{x}\rVert_2
=5\lVert\mathbf{x}\rVert_2
$$

当 $\mathbf{x}=(3,4)^{\mathsf T}/5$ 时等号成立，所以 $\lVert A\rVert_2=5$。Frobenius 范数也是 $5$，而核范数是非零奇异值之和，同样是 $5$。

| 范数 | 读法 | 本例结果 |
| --- | --- | --- |
| $\lVert A\rVert_1$ | 最大列绝对值和 | $4$ |
| $\lVert A\rVert_\infty$ | 最大行绝对值和 | $7$ |
| $\lVert A\rVert_2$ | 最大长度放大 | $5$ |
| $\lVert A\rVert_{\mathrm F}$ | 全部元素平方和开方 | $5$ |
| $\lVert A\rVert_*$ | 奇异值之和 | $5$ |
| $\lVert A\rVert_{\max}$ | 最大元素绝对值 | $4$ |

这里的 $\lVert A\rVert_*$ 是**核范数**，不是矩阵的迹 $\operatorname{tr}(A)$。对半正定矩阵二者相等，但一般矩阵不能混用。

![矩阵范数的五把尺子](/assets/linear-algebra/svg/matrix-norms.1.svg)

## 乘法和误差的统一控制

取

$$
B=\begin{pmatrix}
1&0\\
1&0
\end{pmatrix}
\qquad
AB=\begin{pmatrix}
7&0\\
0&0
\end{pmatrix}
$$

对每种范数分别算一次：

| 范数 | $\lVert A\rVert$ | $\lVert B\rVert$ | $\lVert AB\rVert$ | 乘积上界 |
| --- | ---: | ---: | ---: | ---: |
| $1$ | $4$ | $2$ | $7$ | $8$ |
| $\infty$ | $7$ | $1$ | $7$ | $7$ |
| $2$ | $5$ | $\sqrt{2}$ | $7$ | $5\sqrt{2}$ |
| $\mathrm F$ | $5$ | $\sqrt{2}$ | $7$ | $5\sqrt{2}$ |

例如在 $1$ 范数下，$7\le4\times2=8$；在 $\infty$ 范数下，$7=7\times1$。次乘性是上界，不要求每次乘法都达到它。

对多层线性映射，谱范数给出最直接的长度控制：

$$
\lVert W_L\cdots W_2W_1\mathbf{x}\rVert_2
\le
\left(\prod_{\ell=1}^{L}\lVert W_\ell\rVert_2\right)
\lVert\mathbf{x}\rVert_2
$$

如果某一层的权重扰动为 $\Delta W$，那么它对同一个输入造成的输出变化满足

$$
\lVert\Delta W\mathbf{x}\rVert_2
\le
\lVert\Delta W\rVert_2\lVert\mathbf{x}\rVert_2
$$

这说明范数可以把“参数变了一点”翻译成“输出最多变多少”，但上界可能比实际变化大。

## 条件数：小输入误差会被放大多少

可逆矩阵的条件数定义为

$$
\kappa(A)=\lVert A\rVert\lVert A^{-1}\rVert
$$

因为 $I=AA^{-1}$ 且范数次乘性，所以 $\kappa(A)\ge1$。在 $2$ 范数下，若 $A$ 可逆，

$$
\kappa_2(A)
=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
$$

最大的奇异值表示最强放大，最小的奇异值表示最弱但仍可逆的方向。二者相差很大时，求逆会把某些方向上的输入误差放大很多。比如

$$
D=\begin{pmatrix}
10&0\\
0&1
\end{pmatrix}
\qquad
\kappa_2(D)=\frac{10}{1}=10
$$

若矩阵有零奇异值，普通逆不存在；对秩为 $r$ 的矩形矩阵，可以在非零奇异方向上讨论有效条件数 $\sigma_1/\sigma_r$，但它不替代对核空间的处理。

## 范数之间的换算边界

不同范数的数值不能直接比较大小，维度会进入换算界。设

$$
\lVert A\rVert_{\max}=\max_{i,j}|a_{ij}|
$$

那么

$$
\lVert A\rVert_{\max}
\le\lVert A\rVert_2
\le\lVert A\rVert_{\mathrm F}
\le\sqrt{mn}\lVert A\rVert_{\max}
$$

第一条来自 $|a_{ij}|=|\mathbf{e}_i^{\mathsf T}A\mathbf{e}_j|\le\lVert A\rVert_2$，最后一条只是 $mn$ 个元素每个都不超过最大值。

谱范数还可以由行列和控制：

$$
\lVert A\rVert_2
\le\sqrt{\lVert A\rVert_1\lVert A\rVert_\infty}
$$

证明从 $A^{\mathsf T}A$ 开始：

$$
\begin{aligned}
\lVert A\rVert_2^2
&=\rho(A^{\mathsf T}A)\\
&\le\lVert A^{\mathsf T}A\rVert_1\\
&\le\lVert A^{\mathsf T}\rVert_1\lVert A\rVert_1\\
&=\lVert A\rVert_\infty\lVert A\rVert_1
\end{aligned}
$$

其中 $\rho$ 是谱半径。两边开平方就得到结论。

## 在机器学习中的读法

**看一层最坏的长度放大**：用谱范数 $\lVert W\rVert_2$。它控制线性层的 Lipschitz 常数；多层网络可以用各层谱范数的乘积给出整体上界。

**看参数整体大小**：用 Frobenius 范数。它等于全部参数平方和开方，因此平方后的 Frobenius 范数常出现在权重衰减和二次正则化中。

**希望矩阵变得低秩**：用核范数。它是奇异值的总和，通常比直接约束秩更容易优化；[低秩近似](../linear-algebra/low-rank-approximation/) 篇说明了奇异值截断与核范数之间的关系边界。

**担心逆问题不稳定**：看条件数和最小奇异值。[伪逆](../linear-algebra/pseudoinverse/) 中对零奇异值不取倒数，正是因为小奇异值会把噪声放大。

## 容易混淆的地方

- **矩阵 $1$ 范数不是元素 $1$ 范数**：矩阵 $\lVert A\rVert_1$ 取最大列和；把所有 $|a_{ij}|$ 相加是另一个元素级量。
- **谱范数不是 Frobenius 范数**：前者只看最大奇异值，后者把所有奇异值平方后相加；只有在至多一个奇异值非零时二者相等。
- **矩阵 $2$ 范数不是把矩阵摊平后的欧氏长度**：前者是最大长度放大，后者正是 Frobenius 范数。
- **次乘性不是等式**：矩阵乘法可能只激活部分方向，$\lVert AB\rVert$ 可以严格小于 $\lVert A\rVert\lVert B\rVert$。
- **条件数不是误差本身**：它是误差放大风险的尺度；实际误差还取决于输入扰动、算法和舍入。
- **矩阵奇异时不能硬套 $\lVert A^{-1}\rVert$**：应改用伪逆、截断 SVD 或正则化，并说明处理了哪些方向。

## 相关词条

- [向量](../linear-algebra/vectors/)
- [矩阵乘法](../linear-algebra/matrix-multiplication/)
- [奇异值分解](../linear-algebra/svd/)
- [低秩近似](../linear-algebra/low-rank-approximation/)
- [伪逆](../linear-algebra/pseudoinverse/)
- [正交矩阵与旋转](../linear-algebra/orthogonal-matrices-and-rotations/)
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)
