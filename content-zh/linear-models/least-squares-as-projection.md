---
title: "最小二乘即投影：用正交分解理解拟合"
tags: ["why-models-learn"]
---

最小二乘不是一条只属于统计学的求参公式，而是一个几何问题：给定矩阵 $A$ 和目标向量 $b$，所有可表达的向量 $Ax$ 组成列空间 $\operatorname{col}(A)$，最小二乘要找的是这个子空间中离 $b$ 最近的点。参数 $\widehat x$ 只是描述这个最近点的一组坐标；真正唯一的几何对象是投影向量 $A\widehat x$。这个视角能同时解释正规方程、残差正交、投影矩阵、QR/SVD 求法和秩亏时的非唯一性。

![最小二乘把目标向量投影到特征列空间](/assets/linear-models/svg/least-squares-as-projection.1.svg)

## 最小二乘先问可表达的向量有哪些

设 $A\in\mathbb R^{m\times n}$，$b\in\mathbb R^m$。线性组合

$$
Ax=x_1a_1+\cdots+x_na_n
$$

由 $A$ 的列 $a_1,\ldots,a_n$ 生成，所以全部可表达的向量正好是

$$
\operatorname{col}(A)
=\{Ax:x\in\mathbb R^n\}.
$$

如果 $b$ 恰好在这个列空间里，就能找到 $x$ 使 $Ax=b$。如果 $b$ 不在列空间里，方程没有精确解，但仍然可以在所有可表达向量中选一个离它最近的：

$$
\widehat x\in
\operatorname*{arg\,min}_{x\in\mathbb R^n}
\lVert b-Ax\rVert_2^2.
$$

记

$$
p=A\widehat x,
\qquad
r=b-p=b-A\widehat x.
$$

这里 $p$ 是拟合向量，$r$ 是残差。最小二乘的三个对象职责不同：

| 对象 | 所在空间 | 作用 |
| --- | --- | --- |
| $b$ | $\mathbb R^m$ | 想要表达的目标向量 |
| $A$ 的列 | $\mathbb R^m$ | 提供可表达的方向 |
| $x$ | $\mathbb R^n$ | 组合这些方向的坐标 |
| $p=Ax$ | $\operatorname{col}(A)$ | 实际生成的拟合向量 |
| $r=b-p$ | $\mathbb R^m$ | 目标与拟合之间的差 |

因此“求最小二乘系数”和“求投影向量”是两个相关但不完全相同的问题。系数可能不唯一，投影向量仍可能唯一。

## 最近点的特征是残差正交

先不使用矩阵，设 $V$ 是内积空间中的有限维子空间，$b$ 是任意向量，$p\in V$。如果 $p$ 是 $b$ 在 $V$ 中的最近点，那么残差

$$
r=b-p
$$

必须与 $V$ 中每一个方向正交。证明时任取 $v\in V$，沿这条方向移动仍然留在 $V$：

$$
\begin{aligned}
f(t)
&=\lVert b-(p+tv)\rVert_2^2\\
&=\lVert r-tv\rVert_2^2\\
&=\lVert r\rVert_2^2-2t\langle r,v\rangle+t^2\lVert v\rVert_2^2.
\end{aligned}
$$

$t=0$ 是最小点，所以一阶导数必须为零：

$$
f'(0)=-2\langle r,v\rangle=0.
$$

由于 $v$ 是 $V$ 中任意方向，得到

$$
r\perp V.
$$

反过来也成立。如果 $p\in V$ 且 $r=b-p$ 与 $V$ 正交，对任意 $z\in V$ 都有 $p-z\in V$，所以

$$
\begin{aligned}
\lVert b-z\rVert_2^2
&=\lVert r+(p-z)\rVert_2^2\\
&=\lVert r\rVert_2^2+\lVert p-z\rVert_2^2\\
&\ge \lVert r\rVert_2^2.
\end{aligned}
$$

等号只在 $z=p$ 时成立。于是最近点定理可以压缩成一句话：

$$
p=P_Vb
\quad\Longleftrightarrow\quad
p\in V,\ \ b-p\perp V.
$$

投影的定义是“落在子空间里”，正交条件则是“再沿子空间移动也不能更近”。两者合在一起才唯一确定最近点。

## 正规方程就是正交条件的坐标写法

回到 $V=\operatorname{col}(A)$。最小二乘残差 $r=b-A\widehat x$ 与 $A$ 的每一列正交，因此

$$
A^{\mathsf T}(b-A\widehat x)=0.
$$

移项得到正规方程：

$$
A^{\mathsf T}A\widehat x=A^{\mathsf T}b.
$$

这不是一条凭空出现的代数技巧。$A^{\mathsf T}r$ 的第 $j$ 个分量就是第 $j$ 列与残差的内积；它为零正好表示残差垂直于每个生成方向。

若 $A$ 的列线性无关，$A^{\mathsf T}A$ 可逆，坐标唯一：

$$
\widehat x
=\left(A^{\mathsf T}A\right)^{-1}A^{\mathsf T}b.
$$

但即使列相关导致坐标不唯一，最小二乘拟合向量仍然是列空间中的唯一最近点。此时不能把“正规方程没有唯一参数解”误读成“投影不存在”。

## 投影矩阵把目标一次送到最近点

当 $A$ 列满秩时，把系数公式代回 $A\widehat x$：

$$
p
=A\left(A^{\mathsf T}A\right)^{-1}A^{\mathsf T}b
=P_Ab,
$$

其中

$$
P_A
=A\left(A^{\mathsf T}A\right)^{-1}A^{\mathsf T}.
$$

$P_A$ 只依赖列空间，不依赖列空间使用哪一组满秩基。它满足

$$
P_A^{\mathsf T}=P_A,
\qquad
P_A^2=P_A.
$$

对称性说明它是正交投影，幂等性说明已经投影过的向量再投影不会改变：

$$
P_A(P_Ab)=P_Ab.
$$

残差也有自己的投影矩阵：

$$
r=(I-P_A)b,
\qquad
(I-P_A)^2=I-P_A.
$$

由于 $P_A(I-P_A)=0$，拟合部分和残差部分互相正交。对任意目标向量都有勾股分解：

$$
\lVert b\rVert_2^2
=\lVert P_Ab\rVert_2^2+\lVert(I-P_A)b\rVert_2^2.
$$

### 一个二维直线投影

取直线 $V=\operatorname{span}\{u\}$，其中

$$
u=
\begin{pmatrix}
1\\2
\end{pmatrix},
\qquad
b=
\begin{pmatrix}
4\\1
\end{pmatrix}.
$$

直线上的向量都写成 $cu$。令残差与 $u$ 正交：

$$
u^{\mathsf T}(b-\widehat cu)=0,
$$

得到

$$
\widehat c
=\frac{u^{\mathsf T}b}{u^{\mathsf T}u}
=\frac{6}{5}.
$$

所以投影和残差分别是

$$
p=\widehat cu
=\begin{pmatrix}6/5\\12/5\end{pmatrix},
\qquad
r=b-p
=\begin{pmatrix}14/5\\-7/5\end{pmatrix}.
$$

直接检查：

$$
u^{\mathsf T}r
=\frac{14}{5}-\frac{14}{5}=0.
$$

长度也满足

$$
\lVert b\rVert_2^2=17,
\qquad
\lVert p\rVert_2^2=\frac{36}{5},
\qquad
\lVert r\rVert_2^2=\frac{49}{5},
\qquad
17=\frac{36}{5}+\frac{49}{5}.
$$

这个例子里 $b$ 不在直线 $V$ 上，所以没有 $cu=b$ 的精确解；$\widehat c=6/5$ 给出的不是“真实坐标”，而是最近点 $p$ 在这条直线基向量下的坐标。

## 非正交列也能投影

实际设计矩阵的列通常既不单位化，也不互相正交。取

$$
A=
\begin{pmatrix}
1&0\\
0&1\\
1&1
\end{pmatrix},
\qquad
b=
\begin{pmatrix}
1\\2\\0
\end{pmatrix}.
$$

两列分别是 $(1,0,1)^{\mathsf T}$ 与 $(0,1,1)^{\mathsf T}$，它们不正交，因为内积为 $1$。先算正规方程的两边：

$$
A^{\mathsf T}A
=
\begin{pmatrix}
2&1\\
1&2
\end{pmatrix},
\qquad
A^{\mathsf T}b
=
\begin{pmatrix}
1\\2
\end{pmatrix}.
$$

其逆矩阵是

$$
\left(A^{\mathsf T}A\right)^{-1}
=\frac13
\begin{pmatrix}
2&-1\\
-1&2
\end{pmatrix},
$$

所以

$$
\widehat x
=\left(A^{\mathsf T}A\right)^{-1}A^{\mathsf T}b
=\begin{pmatrix}0\\1\end{pmatrix}.
$$

拟合和残差为

$$
p=A\widehat x
=\begin{pmatrix}0\\1\\1\end{pmatrix},
\qquad
r=b-p
=\begin{pmatrix}1\\1\\-1\end{pmatrix}.
$$

两列分别与残差正交：

$$
A^{\mathsf T}r
=\begin{pmatrix}0\\0\end{pmatrix}.
$$

这里 $\lVert b\rVert_2^2=5$、$\lVert p\rVert_2^2=2$、$\lVert r\rVert_2^2=3$，仍然满足勾股分解。注意不能因为公式里出现 $A^{\mathsf T}$ 就把 $AA^{\mathsf T}b$ 当成投影；本例中

$$
AA^{\mathsf T}b
=\begin{pmatrix}1\\2\\3\end{pmatrix}
\ne
\begin{pmatrix}0\\1\\1\end{pmatrix}
=p.
$$

只有列已经正交归一时，$AA^{\mathsf T}$ 才直接是投影矩阵。

## 正交归一基让投影变成逐方向相加

如果 $Q\in\mathbb R^{m\times r}$ 的列是列空间的一组正交归一基，那么

$$
Q^{\mathsf T}Q=I_r,
\qquad
\operatorname{col}(Q)=\operatorname{col}(A).
$$

目标在第 $j$ 个方向上的坐标是 $\langle q_j,b\rangle$，所以投影可以逐项写成

$$
P_{\operatorname{col}(A)}b
=QQ^{\mathsf T}b
=\sum_{j=1}^{r}q_j\langle q_j,b\rangle.
$$

这解释了正交投影最直观的图像：保留目标在允许方向上的分量，丢掉正交补方向上的分量。若 $Q$ 不是正交归一的，直接写 $QQ^{\mathsf T}$ 会重复计算方向的尺度，必须先正交化，或使用带 $(A^{\mathsf T}A)^{-1}$ 的一般公式。

## QR 把稳定的投影坐标交给三角系统

对列满秩矩阵做 QR 分解：

$$
A=QR,
\qquad
Q^{\mathsf T}Q=I,
$$

其中 $Q$ 的列正交归一，$R$ 是可逆上三角矩阵。因为 $A$ 和 $Q$ 的列空间相同，

$$
A\widehat x
=Q(R\widehat x).
$$

要让 $A\widehat x$ 成为 $b$ 在列空间上的投影，只需要让 $R\widehat x$ 等于 $b$ 在 $Q$ 的方向坐标：

$$
R\widehat x=Q^{\mathsf T}b.
$$

实际计算时用回代解三角系统，不需要形成 $(A^{\mathsf T}A)^{-1}$。正规方程虽然推导简短，却把条件数大致平方：

$$
\kappa_2(A^{\mathsf T}A)=\kappa_2(A)^2.
$$

如果 $A$ 本来就病态，平方会放大舍入误差；QR 通常更稳。奇异值分解则把每个可识别方向单独列出来：

$$
A=U\Sigma V^{\mathsf T},
\qquad
A^+=V\Sigma^+U^{\mathsf T},
\qquad
P_{\operatorname{col}(A)}=AA^+=U_rU_r^{\mathsf T}.
$$

这里 $U_r$ 只保留非零奇异值对应的左奇异向量。小奇异值方向是否保留，可以根据噪声和截断阈值决定；这正是 SVD 既能求解又能诊断秩的原因。

| 路径 | 几何上做什么 | 数值上的注意 |
| --- | --- | --- |
| 正规方程 | 把残差正交条件写成坐标方程 | 计算短，但会平方条件数 |
| QR 分解 | 先构造列空间的正交归一坐标 | 通常比正规方程稳定 |
| SVD | 按奇异方向分解可识别与近零方向 | 最能诊断秩，也最贵 |
| 伪逆 | 在秩亏或欠定时选最小范数坐标 | 需要明确解的选择偏好 |

## 勾股分解给出误差比较

设 $p=P_Vb$，对 $V$ 中任意候选向量 $z$，都有

$$
\lVert b-z\rVert_2^2
=\lVert b-p\rVert_2^2+\lVert p-z\rVert_2^2.
$$

第一项是所有候选向量都无法消除的正交残差，第二项是候选点没有落到最近点造成的额外误差。这条式子比“最小二乘误差最小”更有信息：

1. 改变参数但仍让 $Az=p$，不会改变训练拟合和残差；
2. 只要 $Az\ne p$，就会额外增加 $\lVert p-Az\rVert_2^2$；
3. 若扩大列空间，新的方向可能减少不可消除的正交残差，但也可能让模型更容易适应噪声。

在线性回归中，$A$ 是设计矩阵，$b$ 是标签，$p$ 是训练集预测，$r$ 是回归残差。[线性回归](../linear-models/linear-regression/) 里的 RSS 正是

$$
\operatorname{RSS}
=\lVert b-p\rVert_2^2.
$$

所以回归的正规方程、残差和为零、残差与每个特征列正交，都可以统一还原为同一个投影定理，而不需要把每条结论当成独立口诀。

## 加权最小二乘只是换了一把尺子

如果不同样本的误差重要性不同，或观测噪声的方差不一样，可以选对称正定矩阵 $W$，最小化

$$
\lVert b-Ax\rVert_W^2
=(b-Ax)^{\mathsf T}W(b-Ax).
$$

这等价于在加权内积

$$
\langle u,v\rangle_W=u^{\mathsf T}Wv
$$

下做投影。加权正规方程是

$$
A^{\mathsf T}WA\widehat x=A^{\mathsf T}Wb.
$$

若 $A^{\mathsf T}WA$ 可逆，加权拟合向量可以写为

$$
p_W
=A\left(A^{\mathsf T}WA\right)^{-1}A^{\mathsf T}Wb.
$$

它通常不再是普通欧氏内积下的正交投影，但满足

$$
(b-p_W)^{\mathsf T}Wv=0
\qquad
\text{对所有 }v\in\operatorname{col}(A).
$$

因此加权最小二乘没有推翻投影解释，只是改变了“垂直”和“最近”的度量。若权重来自噪声方差估计，权重本身也应只用训练信息估计。

## 秩亏时投影唯一而坐标可以不唯一

考虑列重复或列相关的矩阵。若存在非零向量 $z$ 满足 $Az=0$，那么任何一组系数 $x$ 都可以改成

$$
x'=x+z,
\qquad
Ax'=Ax.
$$

因此多个参数向量会生成同一个拟合向量。只要列空间固定，最近点 $p$ 仍然唯一；不唯一的是“用哪套坐标表示这个点”。

伪逆给出一个常用选择：

$$
\widehat x_{\mathrm{pinv}}=A^+b.
$$

它在所有最小二乘坐标中选择欧氏范数最小的一个。这个选择不是投影定理强迫的，而是额外的解偏好；在欠定系统、过参数化线性模型和最小范数插值中尤其重要。若问题还要求稀疏、非负或满足业务约束，则应明确换成相应的优化问题，而不能把伪逆解当成唯一自然答案。

## 一次可复用的投影检查

面对一个最小二乘问题，可以按下面顺序检查：

1. 明确目标向量 $b$、可调整坐标 $x$ 和设计矩阵 $A$ 的形状；
2. 写出列空间 $\operatorname{col}(A)$，确认哪些方向实际可以表达；
3. 先问精确方程 $Ax=b$ 是否可解，再决定是否需要近似；
4. 验证候选残差 $r=b-Ax$ 是否满足 $A^{\mathsf T}r=0$，或在加权情形检查 $A^{\mathsf T}Wr=0$；
5. 用勾股分解检查 RSS 是否等于正交残差平方；
6. 检查列秩、奇异值和条件数，决定用正规方程、QR、SVD 还是伪逆；
7. 把参数唯一性与拟合向量唯一性分开报告，并说明任何最小范数、稀疏或正则化偏好。

## 失效模式

**把最小二乘当成一定能解出 $Ax=b$。** 目标不在列空间时，最小二乘只寻找最近的可表达向量，不会制造精确解。

**把参数坐标当成投影本身。** 列相关时坐标可以有多个，$A\widehat x$ 仍是唯一的最近点；报告时要区分参数与预测。

**对非正交列直接使用 $AA^{\mathsf T}$.** 只有列正交归一时这个乘积才是投影矩阵；一般情况需要 $(A^{\mathsf T}A)^{-1}$、QR 或 SVD。

**看到正规方程就显式求逆。** 正规方程适合解释和小规模手算，数值实现通常应使用 QR、SVD 或稳定的迭代方法。

**把残差小当成模型正确。** 投影只说明在选定列空间里最近，不说明列空间合理、数据没有泄漏或部署分布不会改变。

**忘记投影依赖内积。** 加权最小二乘改变了距离和正交的定义；欧氏投影与加权投影可能不是同一个点。

## 相关词条

- [正交投影](../linear-algebra/orthogonal-projections/)：从子空间最近点与正交补的角度系统说明投影算子。
- [线性回归](../linear-models/linear-regression/)：把列空间投影解释用于带截距的数值预测和残差诊断。
- [正交归一基](../linear-algebra/orthonormal-basis/)：说明为什么正交归一坐标能让投影逐方向相加。
- [伪逆](../linear-algebra/pseudoinverse/)：处理秩亏、欠定和最小范数最小二乘坐标。
- [奇异值分解](../linear-algebra/svd/)：用奇异方向诊断秩、条件数和投影。
- [矩阵范数](../linear-algebra/matrix-norms/)：补充条件数与算子放大对数值稳定性的影响。
- [岭回归与 Lasso](../linear-models/ridge-and-lasso/)：继续讨论显式惩罚如何改变投影和参数选择。
