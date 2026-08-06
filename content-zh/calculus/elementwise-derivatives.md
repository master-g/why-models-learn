---
title: "逐分量导数:标量导数如何变成对角 Jacobian"
tags: ["why-models-learn"]
---

若一个向量函数的第 $i$ 个输出只依赖第 $i$ 个输入，这个函数就是**逐分量函数**。对

$$
\Phi(\boldsymbol x)
=
\begin{pmatrix}
\phi(x_1)\\
\vdots\\
\phi(x_n)
\end{pmatrix}
$$

它的 Jacobian 是对角矩阵：

$$
J_\Phi(\boldsymbol x)
=
\operatorname{diag}
\left(
\phi'(x_1),\ldots,\phi'(x_n)
\right)
$$

每个坐标都使用同一个标量导数规则，但不会把别的坐标混进来。于是矩阵乘法可以写成 [向量](../linear-algebra/vectors/) 的逐分量乘法：

$$
J_\Phi(\boldsymbol x)\boldsymbol v
=
\phi'(\boldsymbol x)\odot\boldsymbol v
$$

这里 $\odot$ 是 Hadamard 积，也就是 [向量](../linear-algebra/vectors/) 篇中的逐分量乘法。本篇从对角 Jacobian 的来源出发，连接激活函数、可分离损失和反向敏感度，再说明 ReLU、softmax 和广播操作为什么不能都套用同一个公式。

## 为什么 Jacobian 只有对角线

设

$$
\Phi(\boldsymbol x)=
\begin{pmatrix}
\phi(x_1)\\
\phi(x_2)\\
\vdots\\
\phi(x_n)
\end{pmatrix}
$$

第 $i$ 个输出是 $\Phi_i(\boldsymbol x)=\phi(x_i)$。对第 $j$ 个输入求偏导：

$$
\frac{\partial\Phi_i}{\partial x_j}
=
\begin{cases}
\phi'(x_i),&i=j\\
0,&i\ne j
\end{cases}
$$

当 $i\ne j$ 时，改变 $x_j$ 不会改变 $\Phi_i$，所以非对角元素为零；当 $i=j$ 时，才使用普通的一元导数。于是

$$
J_\Phi(\boldsymbol x)
=
\begin{pmatrix}
\phi'(x_1)&0&\cdots&0\\
0&\phi'(x_2)&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&\phi'(x_n)
\end{pmatrix}
$$

对角矩阵作用在 $\boldsymbol v=(v_1,\ldots,v_n)^{\mathsf T}$ 上：

$$
J_\Phi(\boldsymbol x)\boldsymbol v
=
\begin{pmatrix}
\phi'(x_1)v_1\\
\vdots\\
\phi'(x_n)v_n
\end{pmatrix}
=
\phi'(\boldsymbol x)\odot\boldsymbol v
$$

这条式子把矩阵形状和逐分量实现同时写了出来：前者适合证明和链式法则，后者适合理解每个坐标的局部变化。

## 常见逐分量函数

只要标量函数在相应点可导，就可以把它的导数放到 Jacobian 对角线上：

| 标量函数 $\phi(x)$ | 标量导数 $\phi'(x)$ | 逐分量 Jacobian 的对角线 |
| --- | --- | --- |
| $x$ | $1$ | $(1,\ldots,1)$ |
| $x^2$ | $2x$ | $(2x_1,\ldots,2x_n)$ |
| $e^x$ | $e^x$ | $(e^{x_1},\ldots,e^{x_n})$ |
| $\tanh x$ | $1-\tanh^2x$ | $(1-\tanh^2x_i)_i$ |
| $\sigma(x)$ | $\sigma(x)(1-\sigma(x))$ | $(\sigma(x_i)(1-\sigma(x_i)))_i$ |
| $\operatorname{ReLU}(x)$ | $0$ 或 $1$ | 按坐标取 $0$ 或 $1$ |

ReLU 在 $x=0$ 处没有经典导数，因此它的对角 Jacobian 只在每个坐标都避开零时成立。实际优化中可以选一个次梯度约定，但那是额外约定，不是经典导数突然存在。

## 一个 tanh 向量的数值例子

取

$$
\boldsymbol z=
\begin{pmatrix}
0\\1
\end{pmatrix}
\qquad
\Phi(\boldsymbol z)=
\begin{pmatrix}
\tanh0\\\tanh1
\end{pmatrix}
\approx
\begin{pmatrix}
0\\0.761594
\end{pmatrix}
$$

对角 Jacobian 为

$$
J_\Phi(\boldsymbol z)
=
\begin{pmatrix}
1&0\\
0&1-\tanh^2 1
\end{pmatrix}
\approx
\begin{pmatrix}
1&0\\
0&0.419974
\end{pmatrix}
$$

取输入扰动

$$
\boldsymbol\delta=
\begin{pmatrix}
0.01\\-0.01
\end{pmatrix}
$$

线性预测是

$$
J_\Phi(\boldsymbol z)\boldsymbol\delta
\approx
\begin{pmatrix}
0.01\\-0.004199743
\end{pmatrix}
$$

直接计算的输出变化为

$$
\Phi(\boldsymbol z+\boldsymbol\delta)-\Phi(\boldsymbol z)
\approx
\begin{pmatrix}
0.009999667\\-0.004231832
\end{pmatrix}
$$

差异约为

$$
\begin{pmatrix}
-0.000000333\\-0.000032088
\end{pmatrix}
$$

这是逐坐标的高阶余项。两个坐标互不串扰，但每个坐标自己的非线性仍会产生二阶及更高阶误差。

## 前向和反向都能写成逐分量乘法

如果上游给出输出扰动 $\delta\boldsymbol y$，逐分量层的前向变化是

$$
\delta\boldsymbol y
=
J_\Phi(\boldsymbol x)\delta\boldsymbol x
=
\phi'(\boldsymbol x)\odot\delta\boldsymbol x
$$

如果上游给出损失对输出的行向量敏感度 $\boldsymbol g_y^{\mathsf T}$，反向链式法则给出

$$
\boldsymbol g_x^{\mathsf T}
=
\boldsymbol g_y^{\mathsf T}J_\Phi(\boldsymbol x)
$$

因为对角矩阵转置后不变，所以写成列向量也只是逐分量相乘：

$$
\boldsymbol g_x
=
\boldsymbol g_y\odot\phi'(\boldsymbol x)
$$

以 $\tanh$ 为例，如果某个坐标处 $\tanh'(x_i)$ 很小，该坐标的上游敏感度就会被压小；如果导数接近 $1$，敏感度的一阶尺度基本保留。逐分量结构让这个变化可以按坐标独立检查。

![逐分量层的坐标通道和反向导数门](/assets/calculus/svg/elementwise-derivatives.1.svg)

## 线性层后接激活层

神经网络中常见的组合是

$$
\boldsymbol z=W\boldsymbol x+\boldsymbol b
\qquad
\boldsymbol y=\Phi(\boldsymbol z)
$$

线性层的 Jacobian 是 $W$，逐分量激活层的 Jacobian 是对角矩阵，所以

$$
J_{\boldsymbol y}(\boldsymbol x)
=
J_\Phi(\boldsymbol z)W
=
\operatorname{diag}(\phi'(\boldsymbol z))W
$$

一个输入扰动先被 $W$ 混合到每个 pre-activation 坐标，再由每个激活导数独立缩放。反向时，如果上游梯度是 $\boldsymbol g_y$：

$$
\boldsymbol g_z
=
\boldsymbol g_y\odot\phi'(\boldsymbol z)
\qquad
\boldsymbol g_x
=
W^{\mathsf T}\boldsymbol g_z
$$

先逐分量乘激活导数，再乘线性层转置；顺序不能交换，因为 $W$ 通常不是对角矩阵。

## 可分离标量函数的梯度和 Hessian

逐分量结构也会出现在标量函数的求和中。若

$$
L(\boldsymbol x)
=
\sum_{i=1}^{n}\ell(x_i)
$$

每一项只依赖自己的坐标，则

$$
\nabla L(\boldsymbol x)
=
\begin{pmatrix}
\ell'(x_1)\\
\vdots\\
\ell'(x_n)
\end{pmatrix}
$$

而 Hessian 的非对角元素全为零：

$$
H_L(\boldsymbol x)
=
\operatorname{diag}
\left(
\ell''(x_1),\ldots,\ell''(x_n)
\right)
$$

最简单的平方损失

$$
L(\boldsymbol x)
=\frac12\|\boldsymbol x-\boldsymbol y\|_2^2
=\frac12\sum_i(x_i-y_i)^2
$$

满足

$$
\nabla L=\boldsymbol x-\boldsymbol y
\qquad
H_L=I
$$

每个坐标有相同的曲率 $1$，坐标之间没有交叉项。如果损失中出现 $(x_1-x_2)^2$ 或其他共享参数的项，Hessian 就会出现非对角元素，不能再把坐标完全分开。

## 逐分量函数的复合

若先后应用两个逐分量函数

$$
\Phi(\boldsymbol x)=
\begin{pmatrix}
\phi(x_1)\\\vdots\\\phi(x_n)
\end{pmatrix}
\qquad
\Psi(\boldsymbol y)=
\begin{pmatrix}
\psi(y_1)\\\vdots\\\psi(y_n)
\end{pmatrix}
$$

则

$$
J_{\Psi\circ\Phi}(\boldsymbol x)
=
\operatorname{diag}(\psi'(\phi(x_i)))
\operatorname{diag}(\phi'(x_i))
$$

仍然是对角矩阵，等价于逐坐标相乘：

$$
\frac{\mathrm d}{\mathrm dx_i}\psi(\phi(x_i))
=
\psi'(\phi(x_i))\phi'(x_i)
$$

一旦中间夹入矩阵 $W$，坐标就会被混合，整体 Jacobian 变成

$$
\operatorname{diag}(\psi'(\boldsymbol z_2))
W_2
\operatorname{diag}(\phi'(\boldsymbol z_1))
W_1
$$

对角矩阵和一般矩阵的交替乘积，就是深层网络中局部导数的基本形状。

## 哪些函数不是逐分量的

有些表达式包含逐分量函数，却不能因此把整个 Jacobian 当成对角矩阵。典型例子是 softmax：

$$
s_i(\boldsymbol z)
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

分子中的指数是逐分量的，但分母同时依赖所有坐标。求导得到

$$
\frac{\partial s_i}{\partial z_j}
=
s_i(\delta_{ij}-s_j)
$$

因此 Jacobian 是

$$
J_s(\boldsymbol z)
=
\operatorname{diag}(\boldsymbol s)
-\boldsymbol s\boldsymbol s^{\mathsf T}
$$

通常有非零的非对角元素。范数、归一化、最大值和带共享统计量的操作也会让一个输出依赖多个输入，不能套用逐分量公式。后续广播与归约词条会专门处理坐标复制、求和和维度变化。

## 用有限差分检查对角结构

对 $\Phi(\boldsymbol z)=(\tanh z_1,\tanh z_2)$，在 $\boldsymbol z=(0,1)$，第 $j$ 列可以由中心差分得到：

$$
\frac{\Phi(\boldsymbol z+h\boldsymbol e_j)-\Phi(\boldsymbol z-h\boldsymbol e_j)}{2h}
$$

当 $j=1$ 时，第二个输出坐标不变，所以第二个分量应接近零；当 $j=2$ 时，第一个分量应接近零。取 $h=0.001$，数值结果约为

$$
J_\Phi(\boldsymbol z)
\approx
\begin{pmatrix}
1&0\\
0&0.419974
\end{pmatrix}
$$

有限差分既检查对角位置上的标量导数，也检查不该出现的交叉敏感度。若非对角位置明显不为零，先确认函数是否真的逐分量，再检查坐标索引和广播规则。

## 常见失效模式

- **把逐分量函数写成满矩阵。** 非对角偏导因为输出不依赖其他坐标而为零；先写依赖关系，再填 Jacobian。
- **把逐分量激活和线性层交换。** $\operatorname{diag}(\phi')W$ 通常不等于 $W\operatorname{diag}(\phi')$，矩阵会先后混合不同坐标。
- **忽略 ReLU 的零点。** ReLU 在 $0$ 没有经典导数，使用 $0$ 或 $1$ 的值是次梯度约定。
- **看到指数就认定是逐分量。** softmax 的分母、归一化和归约会把坐标耦合起来。
- **把向量逐分量乘法和点积混为一谈。** $\odot$ 保留向量形状，点积则把两个向量缩成一个标量。
- **只验证对角元素。** 非对角元素应接近零也同样重要，它能发现隐藏的坐标依赖和索引错误。

## 相关词条

- [导数](../calculus/derivatives/)：逐分量层使用的标量导数。
- [雅可比矩阵](../calculus/jacobian/)：把逐分量导数组成对角矩阵。
- [向量链式法则](../calculus/vector-chain-rule/)：组合线性层与逐分量层的 Jacobian。
- [梯度](../calculus/gradient/)：可分离标量函数的一阶敏感度。
- [Hessian 矩阵](../calculus/hessian/)：可分离损失的对角二阶曲率。
- [向量](../linear-algebra/vectors/)：Hadamard 积与逐分量运算的基础。
- [广播与归约的导数](../calculus/broadcast-and-reduction-derivatives/)：处理复制坐标和求和坐标的导数。
- [Softmax](../neurons-and-activations/softmax/)：指数逐分量但归一化跨坐标耦合的例子。
