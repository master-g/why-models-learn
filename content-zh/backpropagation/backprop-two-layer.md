---
title: "两层网络的反向传播：隐藏层梯度如何传回来"
tags: ["why-models-learn"]
---

两层网络的反向传播，是把单个神经元的误差信号再穿过一层参数和一层激活：输出层先从损失得到自己的 logit 梯度，隐藏层接收这个梯度乘以输出权重，再乘自己的激活导数，最后分别形成两层权重、两个偏置和输入的梯度。本文把“两层”固定为一个隐藏层加一个输出层，用一个带 ReLU 隐藏层的二元分类器逐项手算，再推广到不同输出头、batch 矩阵和参数更新。

![两层网络的前向值与反向梯度：输出层误差信号经过输出权重和隐藏层活动掩码传回第一层](/assets/backpropagation/svg/backprop-two-layer.1.svg)

## “两层”先把计数约定说清楚

这里把每次仿射变换算作一层。网络有一个隐藏层和一个输出层：

$$
\boldsymbol x
\longrightarrow
\boldsymbol z^{(1)}
\longrightarrow
\boldsymbol h
\longrightarrow
z^{(2)}
\longrightarrow
\hat y
\longrightarrow
L.
$$

有些资料把“输入到隐藏层”也称为第一层，因此会说这是一个两层网络；有些资料只数隐藏层而称它为“一隐藏层网络”。数字名称不如把方程写出来可靠。本文的两组可学习仿射参数是

$$
\begin{aligned}
\boldsymbol z^{(1)}
&=W^{(1)}\boldsymbol x+\boldsymbol b^{(1)},\\
\boldsymbol h
&=\phi\bigl(\boldsymbol z^{(1)}\bigr),\\
z^{(2)}
&=W^{(2)}\boldsymbol h+b^{(2)},\\
\hat y
&=\psi\bigl(z^{(2)}\bigr).
\end{aligned}
$$

设输入维度为 $d$，隐藏宽度为 $m$，输出先取标量。于是

$$
W^{(1)}\in\mathbb R^{m\times d},
\qquad
\boldsymbol b^{(1)}\in\mathbb R^m,
\qquad
W^{(2)}\in\mathbb R^{1\times m},
\qquad
b^{(2)}\in\mathbb R.
$$

反向阶段用两层误差信号表示敏感度：

$$
\boldsymbol\delta^{(1)}
:=\nabla_{\boldsymbol z^{(1)}}L,
\qquad
\delta^{(2)}
:=\frac{\partial L}{\partial z^{(2)}}.
$$

第一层的误差信号是向量，第二层在标量输出例子中是标量。它们都不是前向的 $\boldsymbol h$ 或 $z^{(2)}$，而是损失对这些节点的导数。

## 先算输出层的误差信号

无论隐藏层有多宽，反向传播都从损失开始。若输出层的上游导数为

$$
g^{(2)}
:=\frac{\partial L}{\partial \hat y},
$$

则

$$
\delta^{(2)}
=g^{(2)}\psi'\bigl(z^{(2)}\bigr).
$$

输出头和损失的组合决定这个标量或向量，隐藏层的反向规则并不替它选择损失：

| 输出头与损失 | 输出层传回的误差信号 | 适用边界 |
| --- | --- | --- |
| $\hat y=z^{(2)}$，$L=\frac12(\hat y-y)^2$ | $\delta^{(2)}=\hat y-y$ | 无界标量回归 |
| $\hat y=\sigma(z^{(2)})$，$L=\operatorname{BCE}(\hat y,y)$ | $\delta^{(2)}=\hat y-y$ | 二元分类 |
| $\hat{\boldsymbol y}=\operatorname{softmax}(\boldsymbol z^{(2)})$，交叉熵 | $\boldsymbol\delta^{(2)}=\hat{\boldsymbol y}-\boldsymbol y$ | 互斥多分类 |

后二者都出现“预测减目标”，但一个是标量，一个是类别向量。不能因为符号相似，就把多分类的 softmax Jacobian 当成若干个互相独立的 sigmoid。

## 一个带 ReLU 隐藏层的完整例子

取输入维度 $d=2$、隐藏宽度 $m=2$：

$$
\boldsymbol x=
\begin{bmatrix}
1\\
2
\end{bmatrix},
\qquad
W^{(1)}=
\begin{bmatrix}
1&0\\
0.5&-0.25
\end{bmatrix},
\qquad
\boldsymbol b^{(1)}=
\begin{bmatrix}
0\\
-0.25
\end{bmatrix}.
$$

隐藏层使用 ReLU，输出层是 sigmoid 加 BCE：

$$
W^{(2)}=
\begin{bmatrix}
0.5&-1
\end{bmatrix},
\qquad
b^{(2)}=0.2,
\qquad
y=1.
$$

这个例子把第二个隐藏单元放在负侧，便于观察活动掩码的作用。

### 前向阶段

第一层预激活为

$$
\begin{aligned}
\boldsymbol z^{(1)}
&=W^{(1)}\boldsymbol x+\boldsymbol b^{(1)}\\
&=
\begin{bmatrix}
1&0\\
0.5&-0.25
\end{bmatrix}
\begin{bmatrix}
1\\
2
\end{bmatrix}
+
\begin{bmatrix}
0\\
-0.25
\end{bmatrix}\\
&=
\begin{bmatrix}
1\\
-0.25
\end{bmatrix}.
\end{aligned}
$$

ReLU 输出是

$$
\boldsymbol h
=\operatorname{ReLU}\bigl(\boldsymbol z^{(1)}\bigr)
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
$$

输出 logit 为

$$
z^{(2)}
=W^{(2)}\boldsymbol h+b^{(2)}
=0.5\times1+(-1)\times0+0.2
=0.7.
$$

概率和损失为

$$
\hat y=\sigma(0.7)\approx0.668187772168,
$$

$$
L
=\log\bigl(1+\exp(0.7)\bigr)-0.7
\approx0.403186048885.
$$

到这里仍然只是一次前向计算。为了让反向阶段知道 ReLU 的局部导数，可以缓存活动掩码

$$
\boldsymbol m^{(1)}
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\boldsymbol m^{(1)}_i=
1_{\{z^{(1)}_i>0\}}.
$$

本例没有遇到 $z^{(1)}_i=0$ 的不可导边界，因此不需要使用 ReLU 零点的次梯度约定。

### 输出层反向

sigmoid 加 BCE 给出

$$
\delta^{(2)}
=\frac{\partial L}{\partial z^{(2)}}
=\hat y-y
\approx0.668187772168-1
=-0.331812227832.
$$

它表示把输出 logit 增大一点会降低当前损失。输出仿射节点对隐藏表示的局部导数是 $W^{(2)}$，所以

$$
\nabla_{\boldsymbol h}L
=\bigl(W^{(2)}\bigr)^{\mathsf T}\delta^{(2)}
=
\begin{bmatrix}
0.5\\
-1
\end{bmatrix}
(-0.331812227832)
=
\begin{bmatrix}
-0.165906113916\\
0.331812227832
\end{bmatrix}.
$$

第二个分量为正并不意味着第二个隐藏单元一定能收到正梯度。它还要穿过第二个隐藏单元自己的 ReLU 边。

### 穿过隐藏层激活

ReLU 的导数由活动掩码表示：

$$
\frac{\partial h_i}{\partial z^{(1)}_i}
=m^{(1)}_i.
$$

逐分量乘法得到第一层预激活的误差信号：

$$
\begin{aligned}
\boldsymbol\delta^{(1)}
&=\nabla_{\boldsymbol z^{(1)}}L\\
&=\nabla_{\boldsymbol h}L\odot\boldsymbol m^{(1)}\\
&=
\begin{bmatrix}
-0.165906113916\\
0.331812227832
\end{bmatrix}
\odot
\begin{bmatrix}
1\\
0
\end{bmatrix}\\
&=
\begin{bmatrix}
-0.165906113916\\
0
\end{bmatrix}.
\end{aligned}
$$

第二个隐藏单元的上游敏感度不是零，真正把它变成零的是 ReLU 的局部导数。这个顺序很重要：先乘输出权重得到 $\nabla_{\boldsymbol h}L$，再乘当前隐藏层的激活导数。

### 两层参数和输入的梯度

输出层的两个参数梯度是

$$
\nabla_{W^{(2)}}L
=\delta^{(2)}\boldsymbol h^{\mathsf T}
=
\begin{bmatrix}
-0.331812227832&0
\end{bmatrix},
\qquad
\frac{\partial L}{\partial b^{(2)}}
=\delta^{(2)}
=-0.331812227832.
$$

第一层仿射节点的局部导数把 $\boldsymbol\delta^{(1)}$ 和输入组成外积：

$$
\nabla_{W^{(1)}}L
=\boldsymbol\delta^{(1)}\boldsymbol x^{\mathsf T}
=
\begin{bmatrix}
-0.165906113916\\
0
\end{bmatrix}
\begin{bmatrix}
1&2
\end{bmatrix}
=
\begin{bmatrix}
-0.165906113916&-0.331812227832\\
0&0
\end{bmatrix}.
$$

偏置和输入的梯度为

$$
\nabla_{\boldsymbol b^{(1)}}L
=\boldsymbol\delta^{(1)}
=
\begin{bmatrix}
-0.165906113916\\
0
\end{bmatrix},
$$

$$
\nabla_{\boldsymbol x}L
=\bigl(W^{(1)}\bigr)^{\mathsf T}\boldsymbol\delta^{(1)}
=
\begin{bmatrix}
1&0.5\\
0&-0.25
\end{bmatrix}
\begin{bmatrix}
-0.165906113916\\
0
\end{bmatrix}
=
\begin{bmatrix}
-0.165906113916\\
0
\end{bmatrix}.
$$

这一组结果的形状是：

| 对象 | 形状 | 本例数值 |
| --- | --- | --- |
| $\delta^{(2)}$ | $1\times1$ | $-0.331812227832$ |
| $\boldsymbol\delta^{(1)}$ | $2\times1$ | $(-0.165906113916,0)$ |
| $\nabla_{W^{(2)}}L$ | $1\times2$ | $(-0.331812227832,0)$ |
| $\nabla_{W^{(1)}}L$ | $2\times2$ | 第一行非零，第二行全零 |
| $\nabla_{\boldsymbol b^{(1)}}L$ | $2\times1$ | $(-0.165906113916,0)$ |
| $\nabla_{\boldsymbol x}L$ | $2\times1$ | $(-0.165906113916,0)$ |

第二个输出权重的梯度为零，是因为第二个隐藏输出 $h_2=0$；第二个隐藏单元第一层参数的梯度为零，是因为它的活动掩码为零。这两个零值发生在不同的局部边上，调试时不能只看到“第二列梯度为零”就把原因都归成同一个问题。

## 把隐藏层梯度展开成路径乘积

对第一层某个权重 $W^{(1)}_{ij}$，从损失到它的路径依次经过输出仿射节点、隐藏激活和第一层仿射节点：

$$
\begin{aligned}
\frac{\partial L}{\partial W^{(1)}_{ij}}
&=
\frac{\partial L}{\partial z^{(2)}}
\frac{\partial z^{(2)}}{\partial h_i}
\frac{\partial h_i}{\partial z^{(1)}_i}
\frac{\partial z^{(1)}_i}{\partial W^{(1)}_{ij}}\\
&=\delta^{(2)}
W^{(2)}_{1i}
\phi'\bigl(z^{(1)}_i\bigr)
x_j.
\end{aligned}
$$

如果把中间的三项依次算好，就是

$$
\boldsymbol\delta^{(1)}
=\bigl(W^{(2)}\bigr)^{\mathsf T}\delta^{(2)}
\odot\phi'\bigl(\boldsymbol z^{(1)}\bigr),
\qquad
\nabla_{W^{(1)}}L
=\boldsymbol\delta^{(1)}\boldsymbol x^{\mathsf T}.
$$

这里有一个容易写反的顺序：

$$
\text{输出敏感度}
\longrightarrow
\text{乘 }W^{(2)\mathsf T}
\longrightarrow
\text{乘隐藏层局部导数}
\longrightarrow
\text{乘输入或累加到偏置}.
$$

不能用 $W^{(1)}$ 去生成隐藏层误差信号。$W^{(1)}$ 描述输入如何影响隐藏预激活，$W^{(2)}$ 描述隐藏表示如何影响输出；反向传播沿相反方向走，因此隐藏节点接收的是后一层的转置权重。

如果一个隐藏单元的输出同时连接到多个输出单元，$\nabla_{\boldsymbol h}L$ 会包含所有输出路径的和：

$$
\nabla_{\boldsymbol h}L
=\sum_{k}
\bigl(W^{(2)}_{k,:}\bigr)^{\mathsf T}
\delta^{(2)}_k.
$$

标量输出例子只有一项，向量输出时不能漏掉其他输出坐标的贡献。

## 输出头变化，隐藏层公式不变

设隐藏层仍然是

$$
\boldsymbol h=\phi\bigl(W^{(1)}\boldsymbol x+\boldsymbol b^{(1)}\bigr),
$$

只替换输出层。得到输出层误差信号后，隐藏层始终使用

$$
\boldsymbol\delta^{(1)}
=
\left(\bigl(W^{(2)}\bigr)^{\mathsf T}\boldsymbol\delta^{(2)}\right)
\odot
\phi'\bigl(\boldsymbol z^{(1)}\bigr).
$$

| 输出任务 | 输出层误差信号 | 第一层误差信号 |
| --- | --- | --- |
| 标量回归 | $\delta^{(2)}=\hat y-y$ | $\bigl(W^{(2)}\bigr)^{\mathsf T}\delta^{(2)}\odot\phi'(\boldsymbol z^{(1)})$ |
| 二元分类 | $\delta^{(2)}=\sigma(z^{(2)})-y$ | 同上 |
| $K$ 类分类 | $\boldsymbol\delta^{(2)}=\boldsymbol p-\boldsymbol y$ | $\bigl(W^{(2)}\bigr)^{\mathsf T}\boldsymbol\delta^{(2)}\odot\phi'(\boldsymbol z^{(1)})$ |

如果输出层是线性回归头，隐藏层仍可以用 ReLU、tanh 或其他激活；如果输出层是 softmax，$W^{(2)}$ 的形状变成 $K\times m$，而第一层梯度的形状仍然是 $m\times d$。

输出损失的 reduction 也只改变输出层误差信号的尺度。例如 batch mean 让每个样本的 $\delta^{(2)}_i$ 先除以 $n$，再沿层传播；不能传播完以后只给第一层除一次。

## batch 形式：每一层都要保留样本轴

把 $n$ 个样本按行堆叠，输入和中间量的形状为

$$
X\in\mathbb R^{n\times d},
\qquad
W^{(1)}\in\mathbb R^{m\times d},
\qquad
H\in\mathbb R^{n\times m}.
$$

采用行向量 batch 约定：

$$
\begin{aligned}
Z^{(1)}
&=X\bigl(W^{(1)}\bigr)^{\mathsf T}
+\boldsymbol1\bigl(\boldsymbol b^{(1)}\bigr)^{\mathsf T},\\
H
&=\phi\bigl(Z^{(1)}\bigr),\\
Z^{(2)}
&=H\bigl(W^{(2)}\bigr)^{\mathsf T}
+\boldsymbol1\bigl(\boldsymbol b^{(2)}\bigr)^{\mathsf T}.
\end{aligned}
$$

令 $G_{Z^{(2)}}$ 表示损失对输出预激活的梯度，并让它已经包含 sum 或 mean reduction 的尺度。反向矩阵公式是

$$
\begin{aligned}
G_H
&=G_{Z^{(2)}}W^{(2)},\\
G_{W^{(2)}}
&=G_{Z^{(2)}}^{\mathsf T}H,\\
\boldsymbol g_{b^{(2)}}
&=\bigl(G_{Z^{(2)}}\bigr)^{\mathsf T}\boldsymbol1,\\
G_{Z^{(1)}}
&=G_H\odot\phi'\bigl(Z^{(1)}\bigr),\\
G_X
&=G_{Z^{(1)}}W^{(1)},\\
G_{W^{(1)}}
&=G_{Z^{(1)}}^{\mathsf T}X,\\
\boldsymbol g_{b^{(1)}}
&=\bigl(G_{Z^{(1)}}\bigr)^{\mathsf T}\boldsymbol1.
\end{aligned}
$$

每个偏置梯度都沿样本轴求和。第一层的 $G_{W^{(1)}}$ 是隐藏误差信号与输入的矩阵乘法；它不是把每个样本的单样本梯度随意平均，而是先保持特征配对再聚合。

### 两样本 batch 的数值核对

沿用刚才的参数，取

$$
X=
\begin{bmatrix}
1&2\\
2&-1
\end{bmatrix},
\qquad
\boldsymbol y=
\begin{bmatrix}
1\\
0
\end{bmatrix},
$$

损失采用 mean。两行前向量为

$$
Z^{(1)}
=
\begin{bmatrix}
1&-0.25\\
2&1
\end{bmatrix},
\qquad
H=
\begin{bmatrix}
1&0\\
2&1
\end{bmatrix},
\qquad
\boldsymbol z^{(2)}
=
\begin{bmatrix}
0.7\\
0.2
\end{bmatrix}.
$$

概率、逐样本损失和平均损失为

$$
\boldsymbol{\hat y}
\approx
\begin{bmatrix}
0.668187772168\\
0.549833997312
\end{bmatrix},
\qquad
\boldsymbol\ell
\approx
\begin{bmatrix}
0.403186048885\\
0.798138869382
\end{bmatrix},
$$

$$
L_{\mathrm{mean}}
\approx0.600662459134.
$$

把 mean 的 $1/2$ 放进输出层误差信号：

$$
G_{Z^{(2)}}
\approx
\begin{bmatrix}
-0.165906113916\\
0.274916998656
\end{bmatrix}.
$$

逐层反向得到

$$
G_{W^{(2)}}
\approx
\begin{bmatrix}
0.383927883397&0.274916998656
\end{bmatrix},
\qquad
\boldsymbol g_{b^{(2)}}\approx0.109010884740,
$$

$$
G_{W^{(1)}}
\approx
\begin{bmatrix}
0.191963941698&-0.303364613244\\
-0.549833997312&0.274916998656
\end{bmatrix},
$$

$$
\boldsymbol g_{b^{(1)}}
\approx
\begin{bmatrix}
0.054505442370\\
-0.274916998656
\end{bmatrix}.
$$

第一行样本的第二隐藏单元仍然在 ReLU 负侧，因此该位置的 $G_{Z^{(1)}}$ 为零；第二行的第二隐藏单元进入正侧，得到非零的第二行参数梯度。一个 batch 内同一个神经元可以对不同样本使用不同活动掩码，不能把一个全 batch 的单一 0/1 状态缓存下来代替逐样本掩码。

## 更新一次，只使用同一张图的全部梯度

回到单样本例子，取 $\eta=0.1$，对两层参数同时做梯度下降：

$$
\Theta_{\mathrm{new}}
=\Theta-\eta\nabla_\Theta L,
\qquad
\Theta=
\left\{
W^{(1)},\boldsymbol b^{(1)},W^{(2)},b^{(2)}
\right\}.
$$

更新后的主要参数为

$$
W^{(2)}_{\mathrm{new}}
\approx
\begin{bmatrix}
0.533181222783&-1
\end{bmatrix},
\qquad
b^{(2)}_{\mathrm{new}}\approx0.233181222783,
$$

$$
W^{(1)}_{\mathrm{new}}
\approx
\begin{bmatrix}
1.016590611392&0.033181222783\\
0.5&-0.25
\end{bmatrix},
\qquad
\boldsymbol b^{(1)}_{\mathrm{new}}
\approx
\begin{bmatrix}
0.016590611392\\
-0.25
\end{bmatrix}.
$$

重新执行前向得到

$$
z^{(2)}_{\mathrm{new}}\approx0.819437260377,
\qquad
\hat y_{\mathrm{new}}\approx0.694116873050,
\qquad
L_{\mathrm{new}}\approx0.365114927682.
$$

当前步使这一个样本的损失从 $0.403186048885$ 降到 $0.365114927682$。这不是“两层网络必然下降”的定理；它依赖当前梯度、学习率和样本。只更新第二层而不更新第一层，或者在第一层更新后才计算第二层梯度，都会得到另一张计算图的混合结果。

## 反向阶段需要缓存每一层的什么

一次单样本前向至少要为反向保存：

| 缓存 | 用途 |
| --- | --- |
| $\boldsymbol x$ | 形成 $\nabla_{W^{(1)}}L=\boldsymbol\delta^{(1)}\boldsymbol x^{\mathsf T}$ |
| $\boldsymbol z^{(1)}$ 或活动掩码 | 计算 $\phi'(\boldsymbol z^{(1)})$ |
| $\boldsymbol h$ | 形成 $\nabla_{W^{(2)}}L=\delta^{(2)}\boldsymbol h^{\mathsf T}$ |
| $z^{(2)}$ 或 $\hat y$ | 计算输出头的局部导数与损失 |
| $\boldsymbol y$ | 计算输出误差信号 |

在 ReLU 例子中只缓存活动掩码就足以求导，但在需要输出值、调试或重新计算时仍可能保留 $\boldsymbol z^{(1)}$。缓存不是参数本身；它属于本次输入和旧参数的执行记录。

正确的时序是

$$
\text{前向缓存所有中间值}
\longrightarrow
\text{算 }\delta^{(2)}
\longrightarrow
\text{反向到 }\boldsymbol\delta^{(1)}
\longrightarrow
\text{计算两层全部梯度}
\longrightarrow
\text{统一更新参数}.
$$

反向完成前不能清掉隐藏表示，也不能先更新 $W^{(2)}$ 再用新权重计算 $\boldsymbol\delta^{(1)}$。否则输出层和隐藏层的梯度不再对应同一组前向值。

## 用有限差分检查每一层

对任意一个参数坐标 $\theta$，中心差分仍然是

$$
\frac{\partial L}{\partial\theta}
\approx
\frac{L(\theta+\varepsilon)-L(\theta-\varepsilon)}{2\varepsilon}.
$$

本例取 $\varepsilon=10^{-5}$，每次只改变一个坐标并从原始参数重新执行完整的两层前向。解析梯度与有限差分为：

| 坐标 | 解析梯度 | 中心差分 |
| --- | ---: | ---: |
| $W^{(1)}_{11}$ | $-0.165906113916$ | $-0.165906113908$ |
| $W^{(1)}_{12}$ | $-0.331812227832$ | $-0.331812227833$ |
| $W^{(1)}_{21}$ | $0$ | $0$ |
| $W^{(1)}_{22}$ | $0$ | $0$ |
| $b^{(1)}_1$ | $-0.165906113916$ | $-0.165906113908$ |
| $b^{(1)}_2$ | $0$ | $0$ |
| $W^{(2)}_1$ | $-0.331812227832$ | $-0.331812227833$ |
| $W^{(2)}_2$ | $0$ | $0$ |
| $b^{(2)}$ | $-0.331812227832$ | $-0.331812227833$ |

这九个坐标的最大绝对差约为 $7.82\times10^{-12}$。检查 ReLU 网络时要避开 $z^{(1)}_i=0$，否则解析梯度取的是某个次梯度，而中心差分跨过了分段边界，二者不必相等。

分层检查比只看最终损失更有用：先核对 $\boldsymbol z^{(1)}$ 和 $\boldsymbol h$，再核对 $z^{(2)}$、$\delta^{(2)}$，然后核对 $\boldsymbol\delta^{(1)}$、两层权重梯度和输入梯度。最终 loss 恰好一致，不能证明中间的转置、广播和活动掩码都正确。

## 失效模式：隐藏层多了一条容易写错的边

**把输出层的梯度直接当成隐藏层梯度。** $\delta^{(2)}$ 位于输出预激活节点，隐藏层还要乘 $\bigl(W^{(2)}\bigr)^{\mathsf T}$ 和激活导数。

**把下一层权重的方向写反。** 列向量约定下，隐藏梯度是 $\bigl(W^{(2)}\bigr)^{\mathsf T}\delta^{(2)}$；行向量 batch 约定下，等价写法变成 $G_{Z^{(2)}}W^{(2)}$。先检查形状再决定转置。

**使用了当前层权重而不是下一层权重。** $W^{(1)}$ 用于把输入梯度拉回 $\boldsymbol x$，$W^{(2)}$ 用于把输出敏感度拉回 $\boldsymbol h$。把二者互换通常会在宽度和输入维度不同的时候直接暴露形状错误。

**忘了逐分量激活导数。** ReLU 的活动掩码、tanh 的 $1-h^2$ 和 sigmoid 的 $a(1-a)$ 都属于隐藏层局部边。只有输出头的特定损失组合发生因子消去，不能把这个简化套到隐藏层。

**把一个样本的掩码广播给整个 batch。** $Z^{(1)}$ 的每个样本、每个隐藏坐标都有自己的活动状态。广播偏置的反向是沿样本轴求和，广播掩码不是沿样本轴取一个代表值。

**把两个零梯度混成一个原因。** 隐藏输出为零会使下一层权重的梯度为零；隐藏预激活落在 ReLU 负侧会使上一层参数梯度为零。调试记录要指出零发生在哪条局部边。

**只更新一层。** 如果优化器漏掉第一层参数，输出仍可能变化，但隐藏表示不会按计算出的第一层梯度移动。检查参数集合、梯度形状和 optimizer 参数组的成员关系。

**在反向中原地覆盖前向值。** 用新激活替换旧缓存，或者先更新输出权重再算隐藏误差，会把同一次训练迭代拆成互不一致的局部图。

## 相关词条

- [单个神经元的反向传播](../backpropagation/backprop-single-neuron/)：先在一条仿射加激活链上固定误差信号、参数梯度和有限差分。
- [反向传播](../backpropagation/backpropagation/)：给出伴随量、局部 VJP 和梯度累加的通用规则。
- [计算图上的链式法则](../backpropagation/chain-rule-on-graphs/)：解释两层之间的路径乘积和共享路径求和。
- [前向计算](../backpropagation/forward-pass/)：先核对每一层的前向缓存、输出头和损失。
- [计算图](../backpropagation/computational-graphs/)：把每层运算和依赖关系展开成可审计的 DAG。
- [向量化反向传播](../backpropagation/vectorized-backprop/)：进一步整理多层 batch 的矩阵乘法和归约。
- [梯度检查](../backpropagation/gradient-checking/)：系统处理多参数、分支和非光滑点的数值核验。
- [ReLU](../neurons-and-activations/relu/)：展开活动掩码、死亡单元和零点次梯度。
- [阶跃与 sigmoid](../neurons-and-activations/heaviside-and-sigmoid/)：解释二元输出头的概率、logit 和 BCE。
