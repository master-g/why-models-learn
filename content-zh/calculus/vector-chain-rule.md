---
title: "向量链式法则:按形状相乘的导数"
tags: ["why-models-learn"]
---

当一个向量函数的输出再被另一个函数接收时，**向量链式法则**把两层的局部线性映射按先后顺序复合。若 $G:\mathbb R^p\to\mathbb R^n$、$F:\mathbb R^n\to\mathbb R^m$ 都可微，则

$$
D(F\circ G)(\boldsymbol a)
=
DF(G(\boldsymbol a))\circ DG(\boldsymbol a)
$$

在标准坐标下，这就是

$$
J_{F\circ G}(\boldsymbol a)
=
J_F(G(\boldsymbol a))J_G(\boldsymbol a)
$$

右边的 $J_G$ 先把输入扰动送到中间空间，左边的 $J_F$ 再把中间扰动送到输出空间。矩阵的形状不仅记账，还强制了乘法顺序。本篇从线性映射和分量公式推导这个规则，再用标量输出、三层组合、前向与反向敏感度说明它为什么是反向传播的数学骨架。

## 先把两个局部线性映射接起来

在输入点 $\boldsymbol a$ 施加小位移 $\boldsymbol h$。第一层给出

$$
G(\boldsymbol a+\boldsymbol h)
\approx
G(\boldsymbol a)+DG(\boldsymbol a)[\boldsymbol h]
$$

记中间位移为

$$
\boldsymbol k=DG(\boldsymbol a)[\boldsymbol h]
$$

再把 $\boldsymbol k$ 传给第二层：

$$
F(G(\boldsymbol a)+\boldsymbol k)
\approx
F(G(\boldsymbol a))+DF(G(\boldsymbol a))[\boldsymbol k]
$$

代回 $\boldsymbol k$：

$$
F(G(\boldsymbol a+\boldsymbol h))
\approx
F(G(\boldsymbol a))
+DF(G(\boldsymbol a))
\left[
DG(\boldsymbol a)[\boldsymbol h]
\right]
$$

复合两个线性映射，正好得到

$$
D(F\circ G)(\boldsymbol a)[\boldsymbol h]
=
DF(G(\boldsymbol a))
\left[
DG(\boldsymbol a)[\boldsymbol h]
\right]
$$

这不是把两个函数值相乘。先经过 $G$，所以先作用 $DG$；只有在中间扰动产生后，才能作用 $DF$。

## 分量公式为什么是矩阵乘法

设

$$
G(\boldsymbol x)=
\begin{pmatrix}
G_1(\boldsymbol x)\\
\vdots\\
G_n(\boldsymbol x)
\end{pmatrix}
\qquad
F(\boldsymbol y)=
\begin{pmatrix}
F_1(\boldsymbol y)\\
\vdots\\
F_m(\boldsymbol y)
\end{pmatrix}
$$

对复合函数的第 $i$ 个输出关于第 $j$ 个输入求偏导。中间有 $n$ 个坐标，每个坐标都贡献一项：

$$
\frac{\partial(F_i\circ G)}{\partial x_j}
=
\sum_{k=1}^{n}
\frac{\partial F_i}{\partial y_k}(G(\boldsymbol x))
\frac{\partial G_k}{\partial x_j}(\boldsymbol x)
$$

而矩阵乘积的第 $(i,j)$ 个元素正是

$$
\left(J_F(G(\boldsymbol x))J_G(\boldsymbol x)\right)_{ij}
=
\sum_{k=1}^{n}
\left(J_F\right)_{ik}
\left(J_G\right)_{kj}
$$

所以求和公式和矩阵公式是同一件事的两种写法。外层 Jacobian 的列数必须等于内层 Jacobian 的行数，因为它们都在中间的 $\mathbb R^n$ 中。

形状可以先写在函数箭头上：

| 映射 | 输入空间 | 输出空间 | Jacobian 形状 |
| --- | --- | --- | --- |
| $G$ | $\mathbb R^p$ | $\mathbb R^n$ | $n\times p$ |
| $F$ | $\mathbb R^n$ | $\mathbb R^m$ | $m\times n$ |
| $F\circ G$ | $\mathbb R^p$ | $\mathbb R^m$ | $m\times p$ |

最终的行数是最终输出维度，列数是最初输入维度。

## 一个向量复合例子

取

$$
G(u,v)=
\begin{pmatrix}
u+v\\
u-v
\end{pmatrix}
\qquad
F(x,y)=
\begin{pmatrix}
x^2+y\\
xy
\end{pmatrix}
$$

在 $(u,v)=(1,0)$，中间点为 $G(1,0)=(1,1)$。两层的 Jacobian 是

$$
J_G(1,0)=
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}
$$

以及

$$
J_F(1,1)=
\begin{pmatrix}
2&1\\
1&1
\end{pmatrix}
$$

相乘：

$$
\begin{aligned}
J_F(1,1)J_G(1,0)
&=
\begin{pmatrix}
2&1\\1&1
\end{pmatrix}
\begin{pmatrix}
1&1\\1&-1
\end{pmatrix}\\
&=
\begin{pmatrix}
3&1\\2&0
\end{pmatrix}
\end{aligned}
$$

直接展开复合函数：

$$
(F\circ G)(u,v)
=
\begin{pmatrix}
(u+v)^2+(u-v)\\
(u+v)(u-v)
\end{pmatrix}
=
\begin{pmatrix}
(u+v)^2+u-v\\
u^2-v^2
\end{pmatrix}
$$

在 $(1,0)$ 求偏导，也得到

$$
J_{F\circ G}(1,0)
=
\begin{pmatrix}
3&1\\2&0
\end{pmatrix}
$$

对输入位移 $(0.2,-0.1)$，先算中间变化：

$$
J_G
\begin{pmatrix}
0.2\\-0.1
\end{pmatrix}
=
\begin{pmatrix}
0.1\\0.3
\end{pmatrix}
$$

再算输出变化：

$$
J_F
\begin{pmatrix}
0.1\\0.3
\end{pmatrix}
=
\begin{pmatrix}
0.5\\0.4
\end{pmatrix}
$$

一次性相乘得到的结果相同：

$$
\begin{pmatrix}
3&1\\2&0
\end{pmatrix}
\begin{pmatrix}
0.2\\-0.1
\end{pmatrix}
=
\begin{pmatrix}
0.5\\0.4
\end{pmatrix}
$$

这里的每一步都是局部一阶变化；原函数在有限位移下还可能产生二阶余项。

## 标量输出：转置把敏感度拉回输入

如果外层函数 $\phi:\mathbb R^n\to\mathbb R$ 只有一个标量输出，$J_\phi$ 是一行。若内层 $G:\mathbb R^p\to\mathbb R^n$，则

$$
J_{\phi\circ G}(\boldsymbol a)
=
J_\phi(G(\boldsymbol a))J_G(\boldsymbol a)
$$

把两边转置成列向量，就得到常用的梯度形式：

$$
\nabla_{\boldsymbol a}(\phi\circ G)
=
J_G(\boldsymbol a)^{\mathsf T}
\nabla\phi(G(\boldsymbol a))
$$

注意顺序反过来了：行向量从左向右乘，列梯度从右向左乘，并且每一层 Jacobian 变成转置。

取

$$
G(u,v)=
\begin{pmatrix}
u+v\\u-v
\end{pmatrix}
\qquad
\phi(x,y)=x^2+3y
$$

在 $(1,0)$，$G(1,0)=(1,1)$，

$$
\nabla\phi(1,1)=
\begin{pmatrix}
2\\3
\end{pmatrix}
\qquad
J_G(1,0)=
\begin{pmatrix}
1&1\\1&-1
\end{pmatrix}
$$

所以

$$
\nabla(\phi\circ G)(1,0)
=
\begin{pmatrix}
1&1\\1&-1
\end{pmatrix}
\begin{pmatrix}
2\\3
\end{pmatrix}
=
\begin{pmatrix}
5\\-1
\end{pmatrix}
$$

直接展开：

$$
(\phi\circ G)(u,v)
=(u+v)^2+3(u-v)
$$

在 $(1,0)$ 的偏导确实是 $(5,-1)$。标量损失的参数梯度之所以可以从输出端一层一层拉回，就是这个转置矩阵乘法的连续应用。

## 一条标量路径上的三层链式法则

链式法则不要求每一层都是二维到二维。取一条标量输入经过向量中间层再得到标量输出：

$$
G(t)=
\begin{pmatrix}
t^2\\e^t
\end{pmatrix}
\qquad
\phi(x,y)=xy+y^2
$$

在 $t=0$，

$$
G(0)=
\begin{pmatrix}
0\\1
\end{pmatrix}
\qquad
G'(0)=
\begin{pmatrix}
0\\1
\end{pmatrix}
$$

外层梯度为

$$
\nabla\phi(0,1)=
\begin{pmatrix}
1\\2
\end{pmatrix}
$$

于是

$$
\frac{\mathrm d}{\mathrm dt}\phi(G(t))\bigg|_{t=0}
=
\nabla\phi(G(0))^{\mathsf T}G'(0)
=
(1,2)
\begin{pmatrix}
0\\1
\end{pmatrix}
=2
$$

直接代入得到

$$
\phi(G(t))=t^2e^t+e^{2t}
$$

在 $t=0$ 求导也是 $2$。中间表示可以有多个坐标，链式法则只要求每一层的输入输出维度相接。

![向量链式法则中的前向矩阵乘法与反向敏感度](/assets/calculus/svg/vector-chain-rule.1.svg)

## 前向与反向：同一乘积的两个方向

设局部链为

$$
\boldsymbol x
\xrightarrow{\;G\;}
\boldsymbol y
\xrightarrow{\;F\;}
\boldsymbol z
$$

前向传播一个输入方向 $\delta\boldsymbol x$：

$$
\delta\boldsymbol y
=J_G\delta\boldsymbol x
\qquad
\delta\boldsymbol z
=J_F\delta\boldsymbol y
=J_FJ_G\delta\boldsymbol x
$$

它回答“某个输入方向的微小变化会怎样到达输出”。如果只关心一个方向，这样逐层推进很自然。

反向传播从输出端给出一个行向量敏感度 $\boldsymbol\lambda_z^{\mathsf T}$：

$$
\boldsymbol\lambda_y^{\mathsf T}
=\boldsymbol\lambda_z^{\mathsf T}J_F
\qquad
\boldsymbol\lambda_x^{\mathsf T}
=\boldsymbol\lambda_y^{\mathsf T}J_G
=\boldsymbol\lambda_z^{\mathsf T}J_FJ_G
$$

它回答“输出的一个微小加权变化，对每层输入的敏感度是多少”。当最终输出是一个标量时，$\boldsymbol\lambda_z^{\mathsf T}$ 只有一行；反向传播可以把这个行向量逐层拉回许多参数，而不必为每个参数单独推送一条输入方向。

两种方向没有两套不同的数学：前向是矩阵从右向左作用在列向量上，反向是行向量从左向右乘同一个矩阵乘积。

## 机器学习中的层间形状

若参数到损失的局部链是

$$
\boldsymbol w\in\mathbb R^p
\xrightarrow{\;G\;}
\boldsymbol y\in\mathbb R^n
\xrightarrow{\;F\;}
\boldsymbol z\in\mathbb R^m
\xrightarrow{\;\ell\;}
L\in\mathbb R
$$

对应 Jacobian 形状为

$$
J_G:n\times p
\qquad
J_F:m\times n
\qquad
J_\ell:1\times m
$$

损失对参数的行向量导数是

$$
J_L
=
J_\ell J_F J_G
$$

转成列梯度则是

$$
\nabla_{\boldsymbol w}L
=
J_G^{\mathsf T}J_F^{\mathsf T}\nabla_{\boldsymbol z}\ell
$$

每个转置都对应“把敏感度拉回上一层”。形状相乘：

$$
(1\times m)(m\times n)(n\times p)=1\times p
$$

结果正好是参数个数对应的一行。后续计算图词条会把标量、向量和矩阵节点的局部导数继续细化；这里先保留最重要的顺序和形状。

## 用有限差分核对链式乘积

对前面的向量复合

$$
H(u,v)=(F\circ G)(u,v)
=
\begin{pmatrix}
(u+v)^2+u-v\\
u^2-v^2
\end{pmatrix}
$$

在 $(1,0)$，理论 Jacobian 是

$$
J_H=
\begin{pmatrix}
3&1\\2&0
\end{pmatrix}
$$

逐列做中心差分：

$$
\frac{H(1+h,0)-H(1-h,0)}{2h}
=
\begin{pmatrix}
3\\2
\end{pmatrix}
$$

$$
\frac{H(1,h)-H(1,-h)}{2h}
=
\begin{pmatrix}
1\\0
\end{pmatrix}
$$

在这个二次复合例子中，$h=0.1$、$0.01$ 和 $0.001$ 都会得到同样的两列。实际非线性网络中，有限差分只能在某个步长范围内近似 Jacobian；步长太大暴露高阶项，太小则受到浮点舍入影响。

## 常见失效模式

- **把复合顺序写反。** $F\circ G$ 先经过 $G$，所以是 $J_FJ_G$；反向写成 $J_GJ_F$ 往往会立刻违反形状。
- **把标量梯度的顺序套到行向量上。** 行导数向右乘，列梯度要用转置后向左乘；先固定行列约定，再写公式。
- **把中间坐标漏掉。** 分量公式中对 $k$ 的求和，表示每一个中间坐标都传递一条局部变化路径。
- **把局部乘积当成有限变化。** Jacobian 乘积给一阶项；每层非线性都会带来余项，有限步长不能直接用线性预测替代。
- **只传播数值不传播形状。** 前向方向是列向量，反向敏感度通常是行向量；每层的输入输出维度必须在乘法前后对齐。
- **把一个输出方向误当成所有输出。** 反向的行向量可以代表一个输出加权组合；如果要保留全部输出变化，就需要完整的 Jacobian，而不是只传一条敏感度。

## 相关词条

- [链式法则](../calculus/chain-rule/)：从一元复合函数开始的导数乘积。
- [全导数](../calculus/total-derivative/)：用线性映射定义多变量的一阶变化。
- [雅可比矩阵](../calculus/jacobian/)：记录每一层的局部线性映射。
- [梯度](../calculus/gradient/)：标量输出 Jacobian 的转置表示。
- [偏导数](../calculus/partial-derivatives/)：提供链式公式中的单项偏导。
- [计算图](../backpropagation/computational-graphs/)：把多层函数组合表示为图结构。
- [单神经元反向传播](../backpropagation/backprop-single-neuron/)：在具体神经元上展开反向敏感度。
- [向量化反向传播](../backpropagation/vectorized-backprop/)：把链式乘积写成矩阵形式。
