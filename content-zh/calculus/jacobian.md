---
title: "雅可比矩阵:向量函数的局部线性变换"
tags: ["why-models-learn"]
---

对于向量值函数 $F:\mathbb R^n\to\mathbb R^m$，**雅可比矩阵**把每个输出分量对每个输入坐标的偏导排成一个 $m\times n$ 矩阵：

$$
J_F(\boldsymbol a)
=
\left[
\frac{\partial F_i}{\partial x_j}(\boldsymbol a)
\right]_{m\times n}
$$

在 $\boldsymbol a$ 附近，它把输入小位移 $\boldsymbol h$ 映射为输出的一阶变化：

$$
F(\boldsymbol a+\boldsymbol h)
\approx
F(\boldsymbol a)+J_F(\boldsymbol a)\boldsymbol h
$$

矩阵的每一列对应一个输入坐标方向，每一行对应一个输出分量。标量输出时只有一行，雅可比矩阵就是梯度的转置；多个输出时，单个梯度向量已经不够记录完整的局部变化。本篇从这个行列结构出发，推导矩阵链式法则、局部面积缩放和机器学习层的线性化。

## 先看矩阵的形状

写成分量形式：

$$
F(\boldsymbol x)=
\begin{pmatrix}
F_1(x_1,\ldots,x_n)\\
\vdots\\
F_m(x_1,\ldots,x_n)
\end{pmatrix}
$$

则

$$
J_F(\boldsymbol x)
=
\begin{pmatrix}
\dfrac{\partial F_1}{\partial x_1}&\cdots&\dfrac{\partial F_1}{\partial x_n}\\
\vdots&\ddots&\vdots\\
\dfrac{\partial F_m}{\partial x_1}&\cdots&\dfrac{\partial F_m}{\partial x_n}
\end{pmatrix}
$$

有 $m$ 个输出，所以有 $m$ 行；有 $n$ 个输入，所以有 $n$ 列。第 $j$ 列是只沿第 $j$ 个输入坐标移动时的输出变化：

$$
J_F(\boldsymbol a)\boldsymbol e_j
=
\begin{pmatrix}
\dfrac{\partial F_1}{\partial x_j}(\boldsymbol a)\\
\vdots\\
\dfrac{\partial F_m}{\partial x_j}(\boldsymbol a)
\end{pmatrix}
$$

因此，雅可比矩阵不是“把所有数字随便放进表格”，而是一个从输入扰动空间到输出扰动空间的线性映射。它就是 [全导数](../calculus/total-derivative/) 在标准坐标下的矩阵表示。

## 一个二元到二元的例子

取

$$
F(x,y)=
\begin{pmatrix}
F_1(x,y)\\
F_2(x,y)
\end{pmatrix}
=
\begin{pmatrix}
x^2+3y\\
xy
\end{pmatrix}
$$

逐个计算四个偏导：

$$
\frac{\partial F_1}{\partial x}=2x
\qquad
\frac{\partial F_1}{\partial y}=3
$$

$$
\frac{\partial F_2}{\partial x}=y
\qquad
\frac{\partial F_2}{\partial y}=x
$$

所以在 $(1,2)$，

$$
F(1,2)=
\begin{pmatrix}
7\\2
\end{pmatrix}
\qquad
J_F(1,2)=
\begin{pmatrix}
2&3\\
2&1
\end{pmatrix}
$$

第一列 $(2,2)^{\mathsf T}$ 是沿 $x$ 方向的输出变化，第二列 $(3,1)^{\mathsf T}$ 是沿 $y$ 方向的输出变化。对位移 $\boldsymbol h=(h,k)$，

$$
J_F(1,2)
\begin{pmatrix}
h\\k
\end{pmatrix}
=
\begin{pmatrix}
2h+3k\\
2h+k
\end{pmatrix}
$$

取 $(h,k)=(0.2,-0.1)$，线性预测的输出变化为

$$
\begin{pmatrix}
2&3\\2&1
\end{pmatrix}
\begin{pmatrix}
0.2\\-0.1
\end{pmatrix}
=
\begin{pmatrix}
0.1\\0.3
\end{pmatrix}
$$

实际计算

$$
F(1.2,1.9)-F(1,2)
=
\begin{pmatrix}
7.14\\2.28
\end{pmatrix}
-
\begin{pmatrix}
7\\2
\end{pmatrix}
=
\begin{pmatrix}
0.14\\0.28
\end{pmatrix}
$$

差异 $(0.04,-0.02)^{\mathsf T}$ 是二阶余项，不是矩阵乘法错误。把位移缩小后，这个余项相对于位移长度会消失。

## 标量输出时退化为梯度

若 $m=1$，$F$ 就是标量函数 $f:\mathbb R^n\to\mathbb R$。雅可比只有一行：

$$
J_f(\boldsymbol a)
=
\begin{pmatrix}
\dfrac{\partial f}{\partial x_1}(\boldsymbol a)&\cdots&
\dfrac{\partial f}{\partial x_n}(\boldsymbol a)
\end{pmatrix}
$$

在标准欧氏内积下，梯度写成列向量：

$$
\nabla f(\boldsymbol a)
=
\begin{pmatrix}
\dfrac{\partial f}{\partial x_1}(\boldsymbol a)\\
\vdots\\
\dfrac{\partial f}{\partial x_n}(\boldsymbol a)
\end{pmatrix}
=
J_f(\boldsymbol a)^{\mathsf T}
$$

例如

$$
f(x,y)=x^2+3xy+y^2
$$

在 $(1,2)$，

$$
J_f(1,2)=
\begin{pmatrix}
8&7
\end{pmatrix}
\qquad
\nabla f(1,2)=
\begin{pmatrix}
8\\7
\end{pmatrix}
$$

把一行当作梯度，或把多输出函数错误地压成一列，都会让后续矩阵乘法的形状失去意义。[梯度](../calculus/gradient/) 适合描述一个标量输出对多个输入的敏感度；Jacobian 记录多个输出同时怎样响应输入。

## 矩阵链式法则

设

$$
G:\mathbb R^p\to\mathbb R^n
\qquad
F:\mathbb R^n\to\mathbb R^m
$$

先由 $G$ 把输入扰动变成中间扰动，再由 $F$ 把中间扰动变成输出扰动。因此

$$
J_{F\circ G}(\boldsymbol a)
=
J_F(G(\boldsymbol a))J_G(\boldsymbol a)
$$

形状也强制这个顺序：

$$
(m\times n)(n\times p)=m\times p
$$

取前面的

$$
F(x,y)=
\begin{pmatrix}
x^2+3y\\xy
\end{pmatrix}
\qquad
G(u,v)=
\begin{pmatrix}
u+v\\2u-v
\end{pmatrix}
$$

在 $(u,v)=(1,0)$，$G(1,0)=(1,2)$，并且

$$
J_G=
\begin{pmatrix}
1&1\\
2&-1
\end{pmatrix}
$$

所以

$$
\begin{aligned}
J_F(1,2)J_G(1,0)
&=
\begin{pmatrix}
2&3\\2&1
\end{pmatrix}
\begin{pmatrix}
1&1\\2&-1
\end{pmatrix}\\
&=
\begin{pmatrix}
8&-1\\4&1
\end{pmatrix}
\end{aligned}
$$

直接展开复合函数：

$$
(F\circ G)(u,v)
=
\begin{pmatrix}
(u+v)^2+3(2u-v)\\
(u+v)(2u-v)
\end{pmatrix}
$$

在 $(1,0)$ 对 $u,v$ 求偏导，也得到

$$
J_{F\circ G}(1,0)
=
\begin{pmatrix}
8&-1\\4&1
\end{pmatrix}
$$

这就是 [向量链式法则](../calculus/vector-chain-rule/) 的矩阵版本：右边的矩阵先作用，左边的矩阵后作用。

## Jacobian 行列式和局部面积

只有输入输出维度相同，Jacobian 才是方阵，才可以讨论行列式。先看线性变换

$$
T(x,y)=
\begin{pmatrix}
2&1\\
1&3
\end{pmatrix}
\begin{pmatrix}
x\\y
\end{pmatrix}
$$

它把单位正方形的两条边

$$
\boldsymbol e_1\longmapsto(2,1)
\qquad
\boldsymbol e_2\longmapsto(1,3)
$$

变成一个平行四边形。面积是

$$
\left|\det
\begin{pmatrix}
2&1\\1&3
\end{pmatrix}\right|
=|6-1|
=5
$$

对可微的二维函数 $F:\mathbb R^2\to\mathbb R^2$，在一点 $\boldsymbol a$ 附近，一个足够小的区域会被 $J_F(\boldsymbol a)$ 近似；局部面积缩放因子近似为

$$
\left|\det J_F(\boldsymbol a)\right|
$$

行列式为零时，线性近似把二维面积压到一条线或更低维方向；非零时，线性近似在二维中可逆。非零行列式只说明局部线性模型可逆，不保证原函数在远处也一一对应。

当输入输出维度不同，没有一个普通的方阵行列式可以直接使用。这时要看秩、列空间或奇异值，不能把一个长方形 Jacobian 强行套进方阵公式。

![输入扰动经过 Jacobian 矩阵变成输出扰动](/assets/calculus/svg/jacobian.1.svg)

## 机器学习层就是 Jacobian 的连续组合

对一个仿射层

$$
\boldsymbol y=W\boldsymbol x+\boldsymbol b
$$

对输入 $\boldsymbol x$ 的 Jacobian 不依赖输入点：

$$
J_{\boldsymbol y}(\boldsymbol x)=W
$$

因为

$$
\boldsymbol y(\boldsymbol x+\boldsymbol h)-\boldsymbol y(\boldsymbol x)
=W\boldsymbol h
$$

这里没有余项，仿射层本身就是精确的线性变化加平移。取

$$
W=
\begin{pmatrix}
2&-1\\
1&3
\end{pmatrix}
\qquad
\boldsymbol h=
\begin{pmatrix}
0.1\\0.2
\end{pmatrix}
$$

输出变化为

$$
W\boldsymbol h
=
\begin{pmatrix}
0\\0.7
\end{pmatrix}
$$

加入非线性激活后，整体 Jacobian 由各层 Jacobian 按链式法则相乘。某些层的矩阵在连续乘积中让向量长度快速缩小或放大，梯度消失和爆炸可以从这个局部线性乘积的角度理解；具体的反向传播词条会把参数梯度也纳入同一套形状规则。

## 用中心差分检查矩阵的每一列

回到

$$
F(x,y)=
\begin{pmatrix}
x^2+3y\\xy
\end{pmatrix}
$$

在 $(1,2)$，第一列可以用只改变 $x$ 的中心差分得到：

$$
\frac{F(1+h,2)-F(1-h,2)}{2h}
=
\begin{pmatrix}
2\\2
\end{pmatrix}
$$

第二列可以用只改变 $y$ 的中心差分得到：

$$
\frac{F(1,2+h)-F(1,2-h)}{2h}
=
\begin{pmatrix}
3\\1
\end{pmatrix}
$$

因为当前函数对每个坐标的最高次数是二次，中心差分在这里恰好消掉误差。实际程序中用有限步长时，仍要在截断误差和浮点舍入误差之间选择合适的 $h$：

| 步长 $h$ | 第 $1$ 列中心差分 | 第 $2$ 列中心差分 |
| --- | --- | --- |
| $0.1$ | $(2,2)^{\mathsf T}$ | $(3,1)^{\mathsf T}$ |
| $0.01$ | $(2,2)^{\mathsf T}$ | $(3,1)^{\mathsf T}$ |
| $0.001$ | $(2,2)^{\mathsf T}$ | $(3,1)^{\mathsf T}$ |

检查完整 Jacobian 时，逐列做这个测试比只检查某个标量输出更容易发现行列方向写反的问题。

## 常见失效模式

- **把行列方向写反。** 本篇采用“行是输出、列是输入”的约定。若代码库采用列向量，$J\boldsymbol h$ 的形状应为输出维度；看到转置时要先核对约定，而不是凭记忆改符号。
- **把标量梯度当成一般 Jacobian。** 梯度只有在输出是一个标量时才足够；向量输出需要每个输出分量的一行偏导。
- **有偏导矩阵不等于已有全导数。** 每个偏导存在只能提供候选矩阵，还要满足同一个线性映射的余项相对位移消失。偏导存在但不连续的反例仍可能不可微。
- **把 Jacobian 乘法当成有限变化的精确答案。** 非线性函数中，$J_F(\boldsymbol a)\boldsymbol h$ 是一阶项；位移不够小时，二阶余项会明显。
- **把行列式用于长方形矩阵。** 只有输入输出维度相同才有普通的 Jacobian 行列式；不同维度应检查秩或奇异值。
- **忽略链式法则的顺序。** 复合 $F\circ G$ 先经过 $G$，所以矩阵乘积是 $J_FJ_G$，形状也会阻止反向相乘。

## 相关词条

- [全导数](../calculus/total-derivative/)：把 Jacobian 解释为一般向量函数的整体线性映射。
- [偏导数](../calculus/partial-derivatives/)：提供 Jacobian 的每个矩阵元素。
- [梯度](../calculus/gradient/)：标量输出 Jacobian 的转置向量表示。
- [向量链式法则](../calculus/vector-chain-rule/)：解释复合函数的 Jacobian 矩阵乘法。
- [链式法则](../calculus/chain-rule/)：从一元复合函数开始的变化率乘积。
- [秩](../linear-algebra/rank/)：判断 Jacobian 局部能保留多少独立方向。
- [行列式](../linear-algebra/determinant/)：解释方阵 Jacobian 的局部面积和方向缩放。
- [Hessian 矩阵](../calculus/hessian/)：记录标量函数的二阶偏导。
