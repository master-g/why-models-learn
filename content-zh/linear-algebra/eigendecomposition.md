---
title: "特征分解:把线性变换拆成基、缩放与换基"
tags: ["why-models-learn"]
---

如果 $n\times n$ 矩阵 $A$ 有 $n$ 个线性无关的特征向量，就可以把这些向量排成可逆矩阵 $P$，把特征值排成对角矩阵 $\Lambda$，得到

$$
A=P\Lambda P^{-1}
$$

这叫**特征分解**（eigendecomposition），也叫可对角化。它不是把原矩阵的非对角元素直接删掉，而是换到一组特殊坐标后，变换沿每个坐标轴独立缩放。本篇把上一页的特征值和特征向量组织成可计算的分解，说明它如何简化矩阵幂、矩阵函数、投影和 PCA；对称矩阵为什么一定有更好的正交分解，留给 [谱定理](../linear-algebra/spectral-theorem/) 篇。

## 从特征向量列出分解

设

$$
A\mathbf{v}_i=\lambda_i\mathbf{v}_i,
\qquad
i=1,\dots,n
$$

并且这些 $\mathbf{v}_i$ 线性无关。把它们作为列排成

$$
P=
\begin{pmatrix}
|&|&&|\\
\mathbf{v}_1&\mathbf{v}_2&\cdots&\mathbf{v}_n\\
|&|&&|
\end{pmatrix},
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n)
$$

矩阵乘法逐列作用，所以

$$
\begin{aligned}
AP
&=\begin{pmatrix}
A\mathbf{v}_1&A\mathbf{v}_2&\cdots&A\mathbf{v}_n
\end{pmatrix}\\
&=\begin{pmatrix}
\lambda_1\mathbf{v}_1&\lambda_2\mathbf{v}_2&\cdots&\lambda_n\mathbf{v}_n
\end{pmatrix}\\
&=P\Lambda
\end{aligned}
$$

线性无关意味着 $P$ 可逆，于是右乘 $P^{-1}$：

$$
A=P\Lambda P^{-1}
$$

反过来，如果一个矩阵能写成这个形式，$P$ 的第 $i$ 列就自动满足

$$
A\mathbf{v}_i
=P\Lambda P^{-1}\mathbf{v}_i
=P\Lambda\mathbf{e}_i
=\lambda_i\mathbf{v}_i
$$

所以「有一组特征基」和「能写成 $P\Lambda P^{-1}$」是同一件事。[特征多项式](../linear-algebra/characteristic-polynomial/)给出根和代数重数，但是否有足够多的独立特征向量，还要检查特征空间。

## 一个完整的二维分解

沿用

$$
A=
\begin{pmatrix}
2&1\\
0&3
\end{pmatrix}
$$

它的两个特征向量可以取为

$$
\mathbf{v}_1=
\begin{pmatrix}
1\\
0
\end{pmatrix},
\qquad
\mathbf{v}_2=
\begin{pmatrix}
1\\
1
\end{pmatrix}
$$

对应特征值 $2、3$，所以

$$
P=
\begin{pmatrix}
1&1\\
0&1
\end{pmatrix},
\qquad
\Lambda=
\begin{pmatrix}
2&0\\
0&3
\end{pmatrix},
\qquad
P^{-1}=
\begin{pmatrix}
1&-1\\
0&1
\end{pmatrix}
$$

逐步乘回去：

$$
P\Lambda
=
\begin{pmatrix}
2&3\\
0&3
\end{pmatrix},
\qquad
P\Lambda P^{-1}
=
\begin{pmatrix}
2&3\\
0&3
\end{pmatrix}
\begin{pmatrix}
1&-1\\
0&1
\end{pmatrix}
=
\begin{pmatrix}
2&1\\
0&3
\end{pmatrix}
=A
$$

![特征分解的计算路径：一般矩阵 A 换到特征基后成为对角矩阵 Λ，特征坐标中只需分别乘 2 和 3，再换回原坐标](/assets/linear-algebra/svg/eigendecomposition.1.svg)

这里的 $P$ 不是正交矩阵：它的两列 $(1,0)$、$(1,1)$ 不垂直。因此 $P^{-1}$ 不等于 $P^{\mathsf T}$；一般特征分解的换基会改变长度和角度，只有对称矩阵的正交特征分解才免费保留这些几何量。

## 特征坐标里，复杂运算变成逐坐标运算

给定原坐标 $\mathbf{x}$，先换到特征坐标

$$
\mathbf{c}=P^{-1}\mathbf{x}
$$

再用 $\Lambda$ 缩放，最后换回去：

$$
A\mathbf{x}=P\Lambda\mathbf{c}
=P\Lambda P^{-1}\mathbf{x}
$$

取

$$
\mathbf{x}=
\begin{pmatrix}
3\\
1
\end{pmatrix}
$$

则

$$
\mathbf{c}=P^{-1}\mathbf{x}
=
\begin{pmatrix}
1&-1\\
0&1
\end{pmatrix}
\begin{pmatrix}
3\\
1
\end{pmatrix}
=
\begin{pmatrix}
2\\
1
\end{pmatrix}
$$

这句话就是 $\mathbf{x}=2\mathbf{v}_1+\mathbf{v}_2$。在特征坐标中，各种操作如下：

| 原空间中的操作 | 特征坐标中的操作 | 适用条件 |
| --- | --- | --- |
| $A\mathbf{x}$ | $c_i\mapsto\lambda_i c_i$ | 已有特征分解 |
| $A^k\mathbf{x}$ | $c_i\mapsto\lambda_i^k c_i$ | $k$ 为非负整数 |
| $A^{-1}\mathbf{x}$ | $c_i\mapsto c_i/\lambda_i$ | 所有 $\lambda_i\ne0$ |
| $f(A)\mathbf{x}$ | $c_i\mapsto f(\lambda_i)c_i$ | $f$ 在各特征值处有定义 |

第一行给出

$$
A\mathbf{x}
=P
\begin{pmatrix}
2&0\\
0&3
\end{pmatrix}
\begin{pmatrix}
2\\
1
\end{pmatrix}
=
\begin{pmatrix}
7\\
3
\end{pmatrix}
$$

第四行是多项式和指数函数等矩阵函数的统一记号：若 $f(t)=t^2$，就得到 $f(A)=A^2$；若 $f(t)=e^t$，则

$$
e^A=P
\begin{pmatrix}
e^{\lambda_1}&&0\\
&\ddots&\\
0&&e^{\lambda_n}
\end{pmatrix}
P^{-1}
$$

矩阵函数的难点被集中到标量函数在每个特征值上的取值。

## 矩阵幂与迭代增长

因为中间的对角矩阵相乘仍然对角：

$$
\begin{aligned}
A^k
&=(P\Lambda P^{-1})(P\Lambda P^{-1})\cdots(P\Lambda P^{-1})\\
&=P\Lambda^kP^{-1},\\
\Lambda^k
&=\operatorname{diag}(\lambda_1^k,\dots,\lambda_n^k)
\end{aligned}
$$

上面的二维例子中

$$
\Lambda^2=
\begin{pmatrix}
4&0\\
0&9
\end{pmatrix},
\qquad
A^2=P\Lambda^2P^{-1}
=
\begin{pmatrix}
4&5\\
0&9
\end{pmatrix}
$$

对 $\mathbf{x}=(3,1)^{\mathsf T}$，特征坐标是 $(2,1)^{\mathsf T}$，所以

$$
A^2\mathbf{x}
=P
\begin{pmatrix}
2^2\cdot2\\
3^2\cdot1
\end{pmatrix}
=
\begin{pmatrix}
17\\
9
\end{pmatrix}
$$

若某个特征坐标 $c_i$ 不为零，$\lvert\lambda_i\rvert$ 最大的方向通常会在长时间迭代中占主导。$\lvert\lambda_i\rvert<1$ 的成分衰减，$\lvert\lambda_i\rvert>1$ 的成分放大，负特征值还会逐次翻转方向。这里的「通常」很重要：重复根、不可对角化和不同方向之间的数值不对齐都可能带来额外的短期行为。

## 谱投影：把每个特征方向单独取出来

若特征值互异，可以在特征坐标中选出第 $i$ 个坐标，再换回原空间。令 $E_i$ 是只有第 $i$ 个对角元素为 $1$ 的矩阵，定义

$$
\Pi_i=PE_iP^{-1}
$$

它是对应特征空间的**谱投影**（spectral projector），满足

$$
\Pi_i^2=\Pi_i,
\qquad
\Pi_i\Pi_j=0\quad(i\ne j),
\qquad
\sum_i\Pi_i=I
$$

并且

$$
A=\sum_i\lambda_i\Pi_i,
\qquad
f(A)=\sum_i f(\lambda_i)\Pi_i
$$

对本篇的二维矩阵：

$$
\Pi_1=
\begin{pmatrix}
1&-1\\
0&0
\end{pmatrix},
\qquad
\Pi_2=
\begin{pmatrix}
0&1\\
0&1
\end{pmatrix}
$$

直接核对：

$$
\Pi_1+\Pi_2=I,
\qquad
2\Pi_1+3\Pi_2
=
\begin{pmatrix}
2&1\\
0&3
\end{pmatrix}
=A
$$

这两个投影一般不是正交投影：$\Pi_1^{\mathsf T}\ne\Pi_1$。它们沿着另一个特征空间的方向把向量送到目标特征空间；[正交投影](../linear-algebra/orthogonal-projections/)则要求残差垂直，并满足对称性。两者都满足幂等性，几何含义却不同。

## 对称矩阵的正交特征分解

如果 $A=A^{\mathsf T}$，不同特征值的特征向量彼此正交；在实数范围内，谱定理进一步保证可以选出正交归一特征基。于是 $P$ 可以换名为 $Q$，并满足

$$
Q^{\mathsf T}Q=I,
\qquad
Q^{-1}=Q^{\mathsf T},
\qquad
A=Q\Lambda Q^{\mathsf T}
$$

这比一般的 $P\Lambda P^{-1}$ 更稳定也更容易解释：$Q^{\mathsf T}$ 只是读取正交坐标，$\Lambda$ 分别缩放，$Q$ 再把坐标放回原空间。完整的存在性和正交性证明放在 [谱定理](../linear-algebra/spectral-theorem/)。

取协方差矩阵

$$
\Sigma=
\begin{pmatrix}
3&1\\
1&3
\end{pmatrix}
$$

两条正交特征方向可以取为

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

对应特征值为 $4、2$，所以

$$
Q=\frac1{\sqrt2}
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
\Sigma=Q\Lambda Q^{\mathsf T}
$$

对任意中心化样本 $x$，$z=Q^{\mathsf T}x$ 是主坐标，$z_1$ 的方差为 $4$，$z_2$ 的方差为 $2$。保留第一列就是 PCA 的一维近似；对应的低秩矩阵为

$$
\Sigma_1=4\mathbf{q}_1\mathbf{q}_1^{\mathsf T}
=
\begin{pmatrix}
2&2\\
2&2
\end{pmatrix}
$$

它保留最大方差方向，丢掉第二方向的方差 $2$。若还要把两个方向缩放到单位方差，白化坐标写成

$$
\widetilde z=\Lambda^{-1/2}Q^{\mathsf T}x
$$

这一步要求被保留的特征值非零；很小的特征值会放大噪声，所以实践中常设置阈值。

## 一般分解的代价与限制

**特征基不一定正交。** 上面的 $P$ 两列夹角不是直角，特征坐标的一个单位变化在原空间里可能很长。$P^{-1}$ 的数值条件数会把输入误差放大；即使每个 $\lambda_i$ 看起来稳定，分解后的计算也可能不稳定。

**有特征值不等于可分解。** 剪切矩阵

$$
S=
\begin{pmatrix}
1&1\\
0&1
\end{pmatrix}
$$

只有一个独立特征方向，不能构造可逆的 $P$。这时要使用 Jordan 形式或 Schur 分解等更一般的工具；本篇只讨论存在完整特征基的情形。

**实矩阵的特征值可能是复数。** 90° 旋转在 $\mathbb{R}$ 中没有实特征方向，因此不能在实数范围内写成实的 $P\Lambda P^{-1}$；在复数范围内可以讨论复特征分解，但坐标也随之变成复数。

**相同分解形式不代表唯一的 $P$。** 可以重新排列特征向量和特征值，只要两者同步排列；也可以把某个特征向量乘非零常数，同时把 $P^{-1}$ 中对应的行反向缩放。若特征值重复，还能在对应特征空间内更换一组基。矩阵 $A$ 不变，坐标表示变了。

## 机器学习里怎样使用特征分解

**PCA。** 协方差矩阵的正交特征分解把相关坐标旋转到互不相关的主坐标，再按特征值排序。截断 $\Sigma=Q\Lambda Q^{\mathsf T}$ 的小特征值，就是在方差预算下丢掉变化较小的方向。

**白化与预条件。** $\Lambda^{-1/2}Q^{\mathsf T}$ 同时完成去相关和尺度归一。它能改善某些优化问题的条件数，但小特征值处的除法会放大测量噪声，实际实现会加上正则项或下限。

**线性层的谱诊断。** 若某一层近似可对角化，权重或 Jacobian 在特征坐标中的每个分量有自己的增长率。谱半径可以给迭代稳定性的第一判断；但非正交 $P$ 和层间方向不对齐意味着不能只看 $\lvert\lambda_{\max}\rvert$。

**低秩近似的边界。** 特征分解适合对称矩阵，权重矩阵通常不对称；一般矩阵的低秩近似应看奇异值分解，而不是把特征值绝对值直接当奇异值。[SVD](../linear-algebra/svd/) 会处理这个区别。

## 容易混淆的地方

**把换基方向写反。** 若 $P$ 的列是特征向量，正确关系是 $AP=P\Lambda$，所以 $A=P\Lambda P^{-1}$。看到一个候选分解，先乘回或检查 $AP=P\Lambda$。

**默认 $P^{-1}=P^{\mathsf T}$。** 只有 $P$ 正交时才成立；一般特征向量既不单位，也不互相垂直。

**把 $\Lambda$ 当成原坐标下的矩阵。** $\Lambda$ 只在特征坐标中描述 $A$。原坐标里的向量必须先算 $c=P^{-1}x$，做完逐坐标缩放后再乘 $P$。

**把特征值排序却不重排特征向量。** 调整 $\Lambda$ 的列顺序时，$P$ 的列必须同步调整，否则 $AP=P\Lambda$ 立即失效。

**把特征分解当成所有矩阵都有的操作。** 根的存在不保证特征向量数量足够；重复根的几何重数不足时，必须换用更一般的分解。

**把特征值当奇异值。** 对称半正定矩阵上两者关系简单；一般权重矩阵上，特征值可以是负数或复数，奇异值却总是非负实数。低秩和稳定性诊断要先确认使用的是哪一种谱量。

## 相关词条

- [特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/)：特征分解所使用的根和方向。
- [特征多项式](../linear-algebra/characteristic-polynomial/)：从行列式得到根及其代数重数。
- [谱定理](../linear-algebra/spectral-theorem/)：实对称矩阵正交特征分解的存在性。
- [正交投影](../linear-algebra/orthogonal-projections/)：与谱投影的几何差别。
- [正交归一基](../linear-algebra/orthonormal-basis/)：正交特征坐标的基础。
- [协方差矩阵](../probability/covariance-matrix/)：PCA 特征分解的输入。
- [二次型](../linear-algebra/quadratic-forms/)：主方向与 Rayleigh 商的优化。
- [SVD](../linear-algebra/svd/)：一般矩阵的稳定低秩分解。
