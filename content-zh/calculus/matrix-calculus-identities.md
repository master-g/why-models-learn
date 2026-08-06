---
title: "矩阵微积分恒等式:先写微分，再读梯度"
tags: ["why-models-learn"]
---

**矩阵微积分恒等式**是一组把矩阵表达式的微小变化写成线性项的规则。关键不是背一张“矩阵求导表”，而是固定一个标量函数对矩阵的梯度约定：

$$
\mathrm df
=
\operatorname{tr}
\left(
(\nabla_X f)^{\mathsf T}\mathrm dX
\right)
$$

先用乘积法则、转置、逆矩阵和迹的规则展开 $\mathrm df$，再把它整理成上面的形状，就能读出 $\nabla_X f$。转置从哪里来、对称矩阵能不能把两项合并、矩阵乘法的左右顺序怎样影响结果，都在这个整理过程中变得可检查。本篇从基本规则推到二次型、Frobenius 残差、逆矩阵和 log-det，再把结果放回线性层和最小二乘。

## 先固定矩阵梯度的约定

令 $X\in\mathbb R^{m\times n}$，$f(X)$ 是一个标量。矩阵扰动 $\mathrm dX$ 与 $X$ 形状相同，标量函数的一阶变化是一个关于 $\mathrm dX$ 的线性函数。用 Frobenius 内积表示：

$$
\langle A,B\rangle_F
=
\operatorname{tr}(A^{\mathsf T}B)
=
\sum_{i=1}^{m}\sum_{j=1}^{n}A_{ij}B_{ij}
$$

矩阵梯度 $\nabla_Xf$ 定义为满足

$$
\mathrm df
=
\langle \nabla_Xf,\mathrm dX\rangle_F
=
\operatorname{tr}
\left(
(\nabla_Xf)^{\mathsf T}\mathrm dX
\right)
$$

的那个 $m\times n$ 矩阵。它和 $X$ 同形状，不是把矩阵自动摊成一列后忘记坐标对应关系。

这一定义给出一个立即可用的读法：

$$
\mathrm df=\operatorname{tr}(C\,\mathrm dX)
\quad\Longrightarrow\quad
\nabla_Xf=C^{\mathsf T}
$$

因为定义中的矩阵出现在转置之后。比如

$$
\mathrm df=\operatorname{tr}(A^{\mathsf T}\mathrm dX)
\quad\Longrightarrow\quad
\nabla_Xf=A
$$

而

$$
\mathrm df=\operatorname{tr}(A\,\mathrm dX)
\quad\Longrightarrow\quad
\nabla_Xf=A^{\mathsf T}
$$

这两个式子只差一个转置，却对应两种不同的表达式。[内积](../linear-algebra/inner-products/) 篇中的 Frobenius 内积正好提供了这里的配对。

## 四条基本微分规则

矩阵微分沿用标量微分的线性性：

$$
\mathrm d(A+B)=\mathrm dA+\mathrm dB
\qquad
\mathrm d(cA)=c\,\mathrm dA
$$

矩阵乘法使用乘积法则，但顺序不能交换：

$$
\mathrm d(AB)
=
(\mathrm dA)B+A(\mathrm dB)
$$

转置和迹分别满足

$$
\mathrm d(A^{\mathsf T})
=(\mathrm dA)^{\mathsf T}
\qquad
\mathrm d\operatorname{tr}(A)
=\operatorname{tr}(\mathrm dA)
$$

迹可以循环移动因子：

$$
\operatorname{tr}(ABC)
=
\operatorname{tr}(BCA)
=
\operatorname{tr}(CAB)
$$

但循环移动不等于任意交换。一般没有

$$
\operatorname{tr}(ABC)=\operatorname{tr}(ACB)
$$

矩阵乘积的形状要先能相乘，循环移动也必须保持每个乘积合法。写推导时可以把含 $\mathrm dX$ 的因子循环到最后，再与

$$
\operatorname{tr}\left((\nabla_Xf)^{\mathsf T}\mathrm dX\right)
$$

逐项比较。

## 线性迹表达式

最简单的标量表达式是

$$
f(X)=\operatorname{tr}(A^{\mathsf T}X)
$$

由线性性

$$
\mathrm df
=
\operatorname{tr}(A^{\mathsf T}\mathrm dX)
$$

所以

$$
\nabla_Xf=A
$$

它只是把矩阵内积展开成了一个标量。若写成

$$
f(X)=\operatorname{tr}(AX)
$$

则

$$
\nabla_Xf=A^{\mathsf T}
$$

这不是“迹把转置吃掉了”。迹只把方阵变成标量，梯度约定仍然要求比较 $\mathrm dX$ 前面的转置。

一个常用特例是 Frobenius 平方距离：

$$
f(X)
=
\frac12\|X-C\|_F^2
=
\frac12\operatorname{tr}\left((X-C)^{\mathsf T}(X-C)\right)
$$

令 $R=X-C$，则

$$
\begin{aligned}
\mathrm df
&=\langle R,\mathrm dR\rangle_F\\
&=\operatorname{tr}(R^{\mathsf T}\mathrm dX)
\end{aligned}
$$

因此

$$
\nabla_Xf=X-C
$$

每个元素的梯度就是对应残差。矩阵写法没有改变逐元素平方损失的事实，只是把所有坐标放进同一个 Frobenius 内积。

## 逆矩阵的微分

设 $X$ 可逆。恒等式

$$
XX^{-1}=I
$$

两边微分：

$$
(\mathrm dX)X^{-1}
+
X\,\mathrm d(X^{-1})
=
0
$$

左乘 $X^{-1}$，得到

$$
\mathrm d(X^{-1})
=
-X^{-1}(\mathrm dX)X^{-1}
$$

中间的 $\mathrm dX$ 不能被当成普通标量移到最右边。若左右再乘上别的矩阵，两个 $X^{-1}$ 仍然留在它的两侧。

这个规则本身也可以当作矩阵链式法则的局部版本。比如 $Y=X^{-1}C$，则

$$
\mathrm dY
=
-X^{-1}(\mathrm dX)X^{-1}C
$$

如果后面还有一个标量损失，就要继续把这三个矩阵按迹的循环规则排成 $\mathrm dX$ 的梯度形状。

## 二次型：为什么对称部分出现两次

对向量 $\boldsymbol x$ 和固定矩阵 $A$，考虑

$$
f(\boldsymbol x)
=
\boldsymbol x^{\mathsf T}A\boldsymbol x
$$

乘积法则给出

$$
\mathrm df
=
(\mathrm d\boldsymbol x)^{\mathsf T}A\boldsymbol x
+
\boldsymbol x^{\mathsf T}A\,\mathrm d\boldsymbol x
$$

把两项都改写成列向量梯度与 $\mathrm d\boldsymbol x$ 的内积：

$$
\begin{aligned}
(\mathrm d\boldsymbol x)^{\mathsf T}A\boldsymbol x
&=(A\boldsymbol x)^{\mathsf T}\mathrm d\boldsymbol x\\
\boldsymbol x^{\mathsf T}A\,\mathrm d\boldsymbol x
&=(A^{\mathsf T}\boldsymbol x)^{\mathsf T}\mathrm d\boldsymbol x
\end{aligned}
$$

于是

$$
\nabla_{\boldsymbol x}f
=
(A+A^{\mathsf T})\boldsymbol x
$$

如果 $A$ 对称，就可以写成

$$
\nabla_{\boldsymbol x}f=2A\boldsymbol x
\qquad
\nabla_{\boldsymbol x}^2f=2A
$$

如果 $A$ 不对称，二次型仍然只看它的对称部分：

$$
\boldsymbol x^{\mathsf T}A\boldsymbol x
=
\boldsymbol x^{\mathsf T}
\left(\frac{A+A^{\mathsf T}}2\right)
\boldsymbol x
$$

反对称部分在这个标量中抵消，所以 Hessian 必须是对称矩阵。[二次型](../linear-algebra/quadratic-forms/) 篇从几何和谱分解角度解释了同一个事实。

取

$$
A=
\begin{pmatrix}
2&1\\
0&3
\end{pmatrix}
\qquad
\boldsymbol x=
\begin{pmatrix}
1\\2
\end{pmatrix}
$$

先算

$$
A\boldsymbol x=
\begin{pmatrix}
4\\6
\end{pmatrix}
\qquad
f(\boldsymbol x)=16
$$

而

$$
(A+A^{\mathsf T})\boldsymbol x
=
\begin{pmatrix}
4&1\\
1&6
\end{pmatrix}
\begin{pmatrix}
1\\2
\end{pmatrix}
=
\begin{pmatrix}
6\\13
\end{pmatrix}
$$

不能直接把 $A\boldsymbol x$ 当成梯度 $(4,6)^{\mathsf T}$，因为 $\boldsymbol x$ 同时出现在左侧和右侧，乘积法则会产生两项。

## 矩阵二次型

向量二次型的矩阵版本是

$$
f(X)
=
\operatorname{tr}(X^{\mathsf T}AX)
$$

这里 $X\in\mathbb R^{p\times q}$，$A\in\mathbb R^{p\times p}$。微分为

$$
\mathrm df
=
\operatorname{tr}\left((\mathrm dX)^{\mathsf T}AX\right)
+
\operatorname{tr}\left(X^{\mathsf T}A\,\mathrm dX\right)
$$

分别循环并整理：

$$
\mathrm df
=
\operatorname{tr}
\left(
\left(AX+A^{\mathsf T}X\right)^{\mathsf T}\mathrm dX
\right)
$$

因此

$$
\nabla_Xf
=
(A+A^{\mathsf T})X
$$

特别地，

$$
\nabla_X\frac12\operatorname{tr}(X^{\mathsf T}AX)
=
\frac12(A+A^{\mathsf T})X
$$

当 $A$ 对称时，这就是 $AX$。另一个最常见的特例是

$$
\frac12\|X\|_F^2
=
\frac12\operatorname{tr}(X^{\mathsf T}X)
\qquad
\nabla_X\frac12\|X\|_F^2=X
$$

这正是矩阵权重衰减的一阶梯度。把它写成逐元素形式，就是每个权重乘以 $1$；把它写成迹，能直接接上更复杂的矩阵表达式。

还可以保留左右两个方阵：

$$
f(X)
=
\operatorname{tr}(X^{\mathsf T}AXB)
$$

其中 $A$ 和 $B$ 的形状分别与 $X$ 的行、列相容。两处 $X$ 各微分一次：

$$
\nabla_Xf
=
AXB+A^{\mathsf T}XB^{\mathsf T}
$$

若 $A$、$B$ 都是单位矩阵，就退化为 $2X$。这条式子也显示了为什么不能从标量规则“把矩阵顺序随便排”：左侧因子和右侧因子会在梯度中分别转置。

## 线性残差的梯度

机器学习和最小二乘中反复出现

$$
F(X)
=
\frac12\|AX-B\|_F^2
$$

令残差

$$
R=AX-B
$$

则

$$
\mathrm dR=A\,\mathrm dX
$$

用平方范数的微分：

$$
\begin{aligned}
\mathrm dF
&=\langle R,\mathrm dR\rangle_F\\
&=\operatorname{tr}\left(R^{\mathsf T}A\,\mathrm dX\right)\\
&=\operatorname{tr}\left((A^{\mathsf T}R)^{\mathsf T}\mathrm dX\right)
\end{aligned}
$$

所以

$$
\nabla_XF
=
A^{\mathsf T}(AX-B)
$$

转置 $A^{\mathsf T}$ 把输出残差拉回 $X$ 的输入空间，这和 [向量链式法则](../calculus/vector-chain-rule/) 中“上游敏感度乘局部映射的转置”是同一条规则。

如果矩阵乘法在右侧：

$$
F(X)
=
\frac12\|XA-B\|_F^2
\qquad
R=XA-B
$$

则

$$
\nabla_XF=RA^{\mathsf T}
$$

左右方向不能混写：

| 残差 | 残差形状 | 对 $X$ 的梯度 |
| --- | --- | --- |
| $AX-B$ | 与 $A$ 的行数和 $B$ 相同 | $A^{\mathsf T}(AX-B)$ |
| $XA-B$ | 与 $A$ 的列数和 $B$ 相同 | $(XA-B)A^{\mathsf T}$ |
| $X-C$ | 与 $X$ 相同 | $X-C$ |

取

$$
A=
\begin{pmatrix}
1&2\\
0&1
\end{pmatrix}
\qquad
X=
\begin{pmatrix}
1&0\\
2&1
\end{pmatrix}
\qquad
B=
\begin{pmatrix}
1&1\\
1&0
\end{pmatrix}
$$

得到

$$
AX=
\begin{pmatrix}
5&2\\
2&1
\end{pmatrix}
\qquad
R=
\begin{pmatrix}
4&1\\
1&1
\end{pmatrix}
$$

因此

$$
F(X)=9.5
\qquad
\nabla_XF
=
A^{\mathsf T}R
=
\begin{pmatrix}
4&1\\
9&3
\end{pmatrix}
$$

梯度的第二行出现 $9$，正是 $A^{\mathsf T}$ 把两行残差按正确方向混合后的结果。

## 几个迹恒等式

把含 $\mathrm dX$ 的项循环到最后，可以得到一组可直接复用的结果：

| 标量表达式 $f(X)$ | 微分整理后的梯度 $\nabla_Xf$ |
| --- | --- |
| $\operatorname{tr}(A^{\mathsf T}X)$ | $A$ |
| $\operatorname{tr}(AX)$ | $A^{\mathsf T}$ |
| $\operatorname{tr}(AXB)$ | $A^{\mathsf T}B^{\mathsf T}$ |
| $\operatorname{tr}(X^{\mathsf T}AX)$ | $(A+A^{\mathsf T})X$ |
| $\frac12\operatorname{tr}(X^{\mathsf T}AX)$ | $\frac12(A+A^{\mathsf T})X$ |
| $\frac12\operatorname{tr}(X^{\mathsf T}X)$ | $X$ |

例如

$$
\mathrm d\operatorname{tr}(AXB)
=
\operatorname{tr}(A\,\mathrm dX\,B)
=
\operatorname{tr}(BA\,\mathrm dX)
$$

所以梯度是 $(BA)^{\mathsf T}=A^{\mathsf T}B^{\mathsf T}$。表格中的每一行都只是同一个动作：乘积法则展开，迹循环，和梯度定义比较。[迹](../linear-algebra/trace/) 篇说明了循环性质为什么成立，也提醒了它不是任意交换律。

## 行列式与 log-det

逆矩阵微分可以继续推出行列式的微分。对可逆的 $X$，Jacobi 公式是

$$
\mathrm d\det X
=
\det(X)\operatorname{tr}(X^{-1}\mathrm dX)
$$

如果 $X$ 是对称正定矩阵，可以取对数：

$$
\mathrm d\log\det X
=
\operatorname{tr}(X^{-1}\mathrm dX)
$$

按梯度约定读出

$$
\nabla_X\det X
=
\det(X)X^{-\mathsf T}
$$

以及

$$
\nabla_X\log\det X
=
X^{-\mathsf T}
$$

这里的 $X^{-\mathsf T}$ 表示 $(X^{-1})^{\mathsf T}$。log-det 的定义域需要满足行列式为正；在实数优化里，最常用的是对称正定矩阵。接近奇异时，$X^{-1}$ 的元素可能很大，梯度也会变得敏感。

取

$$
X=
\begin{pmatrix}
2&0\\
0&3
\end{pmatrix}
$$

则

$$
\det X=6
\qquad
\nabla_X\log\det X
=
\begin{pmatrix}
1/2&0\\
0&1/3
\end{pmatrix}
$$

而

$$
\nabla_X\det X
=
\begin{pmatrix}
3&0\\
0&2
\end{pmatrix}
$$

行列式的梯度包含 $\det X$ 的整体缩放；log-det 把这个缩放除掉，只留下逆矩阵转置。

## 矩阵链式法则的形状版本

设中间矩阵由

$$
Z=AXB+C
$$

给出，$f(Z)$ 是标量，已知上游矩阵梯度

$$
G_Z=\nabla_Zf
$$

则

$$
\mathrm dZ=A\,\mathrm dX\,B
$$

从

$$
\begin{aligned}
\mathrm df
&=\operatorname{tr}(G_Z^{\mathsf T}A\,\mathrm dX\,B)\\
&=\operatorname{tr}\left((A^{\mathsf T}G_ZB^{\mathsf T})^{\mathsf T}\mathrm dX\right)
\end{aligned}
$$

得到

$$
\nabla_Xf
=
A^{\mathsf T}G_ZB^{\mathsf T}
$$

对应的几个形状变体是

| 前向变换 | 反向变换 |
| --- | --- |
| $Z=AX$ | $G_X=A^{\mathsf T}G_Z$ |
| $Z=XB$ | $G_X=G_ZB^{\mathsf T}$ |
| $Z=AXB$ | $G_X=A^{\mathsf T}G_ZB^{\mathsf T}$ |
| $Z=X^{\mathsf T}$ | $G_X=G_Z^{\mathsf T}$ |

例如全连接层的前向式 $Z=XW+\boldsymbol b$ 包含一个矩阵乘法和一个广播加法。矩阵部分把上游梯度右乘 $W^{\mathsf T}$，偏置部分沿批次轴求和，正好分别对应 [广播与归约导数](../calculus/broadcast-and-reduction-derivatives/) 和本节的矩阵链式规则。

![矩阵微分从标量表达式整理到梯度，并在残差平方中沿转置方向回传](/assets/calculus/svg/matrix-calculus-identities.1.svg)

## 用有限差分核对转置和系数

继续使用

$$
F(X)=\frac12\|AX-B\|_F^2
$$

的具体矩阵。解析梯度为

$$
\nabla_XF=
\begin{pmatrix}
4&1\\
9&3
\end{pmatrix}
$$

对每个矩阵元素 $X_{ij}$ 使用中心差分：

$$
\widehat G_{ij}(h)
=
\frac{
F(X+hE_{ij})-F(X-hE_{ij})
}{2h}
$$

其中 $E_{ij}$ 只有第 $(i,j)$ 个位置为 $1$。取 $h=10^{-4}$，得到

| 位置 | 解析梯度 | 中心差分 |
| --- | --- | --- |
| $X_{11}$ | $4$ | $4.000000$ |
| $X_{12}$ | $1$ | $1.000000$ |
| $X_{21}$ | $9$ | $9.000000$ |
| $X_{22}$ | $3$ | $3.000000$ |

再对 $X=\operatorname{diag}(2,3)$ 的两个对角元素检查 log-det：

| 位置 | 解析梯度 | 中心差分 |
| --- | --- | --- |
| $X_{11}$ | $1/2$ | $0.500000$ |
| $X_{22}$ | $1/3$ | $0.333333$ |

有限差分同时检查了数值系数和转置方向。若把 $A^{\mathsf T}R$ 错写成 $AR$，或把右乘残差的规则套到左乘残差，至少有一个矩阵位置会对不上。[数值微分](../calculus/numerical-differentiation/) 篇展开步长选择和中心差分的误差来源。

## 常见失效模式

- **先套记忆表再看梯度约定。** 先把 $\mathrm df$ 化成 $\operatorname{tr}(G^{\mathsf T}\mathrm dX)$，再读 $G$。
- **把矩阵乘法当成交换乘法。** $A\,\mathrm dX\,B$ 循环进迹时可以移动，不能把 $A$ 和 $B$ 任意换位。
- **看到对称例子就删掉一项。** $2A\boldsymbol x$ 只在 $A=A^{\mathsf T}$ 时成立；一般二次型的梯度是 $(A+A^{\mathsf T})\boldsymbol x$。
- **忽略残差所在的一侧。** $AX-B$ 的梯度是 $A^{\mathsf T}R$，$XA-B$ 的梯度是 $RA^{\mathsf T}$。
- **把逆矩阵当成逐元素倒数。** $\mathrm d(X^{-1})$ 的两个 $X^{-1}$ 分处 $\mathrm dX$ 两侧。
- **在奇异点使用 log-det 公式。** $X^{-1}$ 不存在时公式失效；接近奇异也会造成很大的梯度。
- **把有限差分当成定义。** 它是对解析推导的数值检查，步长太大或太小都可能掩盖错误。

## 相关词条

- [内积](../linear-algebra/inner-products/)：Frobenius 内积提供矩阵梯度的配对。
- [迹](../linear-algebra/trace/)：循环性质支撑迹表达式的整理。
- [二次型](../linear-algebra/quadratic-forms/)：解释二次型只看矩阵的对称部分。
- [逆矩阵](../linear-algebra/matrix-inverse/)：逆矩阵的代数定义与可逆性条件。
- [行列式](../linear-algebra/determinant/)：行列式和体积缩放、可逆性的关系。
- [矩阵乘法](../linear-algebra/matrix-multiplication/)：所有微分乘积规则的形状基础。
- [向量链式法则](../calculus/vector-chain-rule/)：把矩阵局部映射按顺序复合。
- [广播与归约导数](../calculus/broadcast-and-reduction-derivatives/)：偏置和批次梯度的复制、求和。
- [数值微分](../calculus/numerical-differentiation/)：用有限差分检查矩阵梯度。
