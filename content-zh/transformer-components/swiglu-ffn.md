---
title: "SwiGLU：门控 FFN 如何选择性写入特征"
tags: ["why-models-learn"]
---

SwiGLU 是一种门控前馈网络。它把同一个 token 的输入投影成两条中间分支：一条产生门值，一条产生待调制的值；门分支经过 SiLU 后与值分支逐坐标相乘，最后由下投影回到 residual stream 的宽度。对列向量 $h\in\mathbb R^d$，带 bias 的形式为

$$
g=W_gh+b_g,
$$

$$
u=W_uh+b_u,
$$

$$
a=\operatorname{SiLU}(g)\odot u,
$$

$$
y=W_da+b_d.
$$

$W_g$ 是 gate projection，$W_u$ 是 up projection，$W_d$ 是 down projection。门控乘法产生输入相关的特征选择；它不是把两个向量拼接，也不是 attention 的 token 间读取。Transformer block 把 $y$ 作为 FFN 分支增量写入 [残差流](../transformer-components/residual-streams/)，所以最后一轴必须回到 $d=d_{\mathrm{model}}$。

本文先固定 gate、up、down 三个投影的 shape，再用一个二维数值例子计算门值、SiLU、逐坐标乘法和输出，推导局部 Jacobian 与反向梯度，比较普通 FFN 和 SwiGLU 的参数预算，最后处理 pre/post norm、padding、混合精度、融合 kernel 和核验协议。

![SwiGLU 从同一个 token 产生 gate 与 value 两条中间分支，经过 SiLU 门控乘法后下投影回 residual stream](/assets/transformer-components/svg/swiglu-ffn.1.svg)

## 先固定 SwiGLU 的 shape 合同

### 一个 token 的列向量约定

以下统一采用列向量。设

$$
h\in\mathbb R^d,
\qquad
d=d_{\mathrm{model}},
\qquad
m=d_{\mathrm{ffn}}.
$$

各量的 shape 为：

| 符号 | shape | 作用 |
| --- | --- | --- |
| $h$ | $\mathbb R^d$ | 一个 token 的输入表示 |
| $W_g$ | $\mathbb R^{m\times d}$ | 产生 gate pre-activation |
| $b_g$ | $\mathbb R^m$ | gate 分支的偏置 |
| $g$ | $\mathbb R^m$ | SiLU 的输入 |
| $W_u$ | $\mathbb R^{m\times d}$ | 产生 value pre-activation |
| $b_u$ | $\mathbb R^m$ | value 分支的偏置 |
| $u$ | $\mathbb R^m$ | 被门调制的中间值 |
| $a$ | $\mathbb R^m$ | 逐坐标门控乘积 |
| $W_d$ | $\mathbb R^{d\times m}$ | 把门控结果压回 stream 宽度 |
| $b_d$ | $\mathbb R^d$ | 输出 bias |
| $y$ | $\mathbb R^d$ | 写回 residual stream 的分支输出 |

完整前向可以写成

$$
\mathbb R^d
\xrightarrow{\ W_g,\ W_u\ }
\mathbb R^m\times\mathbb R^m
\xrightarrow{\ \operatorname{SiLU}(g)\odot u\ }
\mathbb R^m
\xrightarrow{\ W_d\ }
\mathbb R^d.
$$

两条上投影都从同一个 $h$ 读取，但它们拥有不同的参数。只有在逐坐标乘法之后，两个分支才合并为一个 $m$ 维中间表示。

### 批量和序列的最后一轴

对于

$$
H\in\mathbb R^{B\times T\times d},
$$

SwiGLU 固定 $(b,t)$ 后独立计算：

$$
G_{b,t,:}=W_gH_{b,t,:}+b_g,
$$

$$
U_{b,t,:}=W_uH_{b,t,:}+b_u,
$$

$$
A_{b,t,:}=\operatorname{SiLU}(G_{b,t,:})\odot U_{b,t,:},
$$

$$
Y_{b,t,:}=W_dA_{b,t,:}+b_d.
$$

所以

$$
Y\in\mathbb R^{B\times T\times d},
\qquad
G,U,A\in\mathbb R^{B\times T\times m}.
$$

把 $B\times T$ 个 token 展平为 $N=BT$ 行向量后，若权重仍按列向量约定保存，批量形式为

$$
G=XW_g^{\mathsf T}+\mathbf 1b_g^{\mathsf T},
$$

$$
U=XW_u^{\mathsf T}+\mathbf 1b_u^{\mathsf T},
$$

$$
A=\operatorname{SiLU}(G)\odot U,
$$

$$
Y=AW_d^{\mathsf T}+\mathbf 1b_d^{\mathsf T}.
$$

这里的逐元素乘法只发生在同一 token 的最后一轴，不在 $N$ 行之间做乘法。

### 门分支和 value 分支的命名

本篇固定以下名称：

| 名称 | 记号 | 说明 |
| --- | --- | --- |
| gate 分支 | $g$ | 经过 SiLU 后决定每个中间坐标的调制系数 |
| value 分支 | $u$ | 被 gate 逐坐标缩放的中间特征 |
| 门控结果 | $a$ | $\operatorname{SiLU}(g)\odot u$ |
| down 分支 | $W_d$ | 把门控结果组合回 $d$ 维输出 |

这里的 value 分支只是门控 FFN 的命名，不是 attention 中从 $V$ 矩阵读取的 value。两者都可以被逐坐标组合，但 shape、读取范围和参数职责不同。

## 一个二维 SwiGLU 的数值例子

### 固定输入和三个投影

取

$$
h=
\begin{pmatrix}
1\\
0
\end{pmatrix},
\qquad
d=2,
\qquad
m=2.
$$

为突出门控乘法，先令三个 bias 都为零：

$$
b_g=b_u=b_d=0.
$$

取 gate projection、up projection 和 down projection：

$$
W_g=
\begin{pmatrix}
0&1\\
1&1
\end{pmatrix},
\qquad
W_u=
\begin{pmatrix}
2&1\\
-1&1
\end{pmatrix},
$$

$$
W_d=
\begin{pmatrix}
1&0.5\\
-0.5&1
\end{pmatrix}.
$$

这些矩阵都遵守

$$
W_g,W_u\in\mathbb R^{2\times2},
\qquad
W_d\in\mathbb R^{2\times2}.
$$

### 两条上投影产生不同的中间值

gate 分支为

$$
g=W_gh
=
\begin{pmatrix}
0&1\\
1&1
\end{pmatrix}
\begin{pmatrix}
1\\
0
\end{pmatrix}
=
\begin{pmatrix}
0\\
1
\end{pmatrix}.
$$

value 分支为

$$
u=W_uh
=
\begin{pmatrix}
2&1\\
-1&1
\end{pmatrix}
\begin{pmatrix}
1\\
0
\end{pmatrix}
=
\begin{pmatrix}
2\\
-1
\end{pmatrix}.
$$

两个分支读取相同的 $h$，但得到的数值和职责不同。SwiGLU 不能把 $g$ 和 $u$ 当作两份重复的 hidden。

### SiLU 产生门值

定义 sigmoid 和 SiLU：

$$
\sigma(v)=\frac{1}{1+e^{-v}},
$$

$$
\operatorname{SiLU}(v)=v\sigma(v).
$$

对 $g=(0,1)^{\mathsf T}$，有

$$
\sigma(0)=0.5,
\qquad
\operatorname{SiLU}(0)=0,
$$

以及

$$
\sigma(1)=0.731058578630,
\qquad
\operatorname{SiLU}(1)=0.731058578630.
$$

因此

$$
s=\operatorname{SiLU}(g)
=
\begin{pmatrix}
0\\
0.731058578630
\end{pmatrix}.
$$

SiLU 的门值不受限于 $[0,1]$。正输入时它可以大于 1，负输入时可以为负；sigmoid 只参与计算，不是最终门值。

### 逐坐标门控和下投影

门控乘积为

$$
a=s\odot u
=
\begin{pmatrix}
0\\
0.731058578630
\end{pmatrix}
\odot
\begin{pmatrix}
2\\
-1
\end{pmatrix}
=
\begin{pmatrix}
0\\
-0.731058578630
\end{pmatrix}.
$$

最后

$$
y=W_da
=
\begin{pmatrix}
1&0.5\\
-0.5&1
\end{pmatrix}
\begin{pmatrix}
0\\
-0.731058578630
\end{pmatrix}
=
\begin{pmatrix}
-0.365529289315\\
-0.731058578630
\end{pmatrix}.
$$

第 1 个 gate 值为零，所以第 1 个 value 坐标在本例中被完全关闭。第 2 个 gate 值为正，但 value 为负，因此门控结果仍然为负。门控决定的是乘法系数，不是简单的二值保留掩码。

### 数值账本

| 对象 | 数值 |
| --- | --- |
| $\lVert h\rVert_2$ | $1.414213562373$ |
| $\lVert g\rVert_2$ | $1$ |
| $\lVert u\rVert_2$ | $2.236067977500$ |
| $\lVert \operatorname{SiLU}(g)\rVert_2$ | $0.731058578630$ |
| $\lVert a\rVert_2$ | $0.731058578630$ |
| $\lVert y\rVert_2$ | $0.817348338676$ |

门控结果的范数由 gate 和 value 的共同尺度决定，不能只用 gate 的范数预测。

## GLU 家族和普通 FFN 的差异

### 普通 FFN 只有一条上投影

两层普通 FFN 可以写成

$$
F(h)=W_2\phi(W_1h+b_1)+b_2.
$$

它先生成一个中间向量，再逐坐标激活，最后下投影。SwiGLU 把第一步拆成两条投影：

$$
g=W_gh+b_g,
\qquad
u=W_uh+b_u,
$$

再用

$$
a=\phi(g)\odot u.
$$

因此 gate 分支可以按当前输入改变 value 分支的有效尺度。这个乘法相当于输入相关的特征组合，不能被一个固定的两层线性矩阵直接替代。

### GLU 的统一形式

广义 GLU 形式为

$$
\operatorname{GLU}_{\phi}(h)
=\phi(W_gh+b_g)\odot(W_uh+b_u).
$$

不同变体只替换 $\phi$：

| 变体 | $\phi$ | 门控特点 |
| --- | --- | --- |
| GLU | $\sigma$ | 门值通常在 $[0,1]$ |
| ReGLU | ReLU | 负 gate 完全关闭 |
| GEGLU | GELU | 平滑的 GELU 门控 |
| SwiGLU | SiLU | 平滑、可为负且可超过 1 |

变体的名称只描述门函数，不改变两条上投影和一个下投影的基本 shape。

### 门控乘法不是拼接

拼接会产生

$$
\operatorname{concat}(g,u)\in\mathbb R^{2m}.
$$

SwiGLU 产生的是

$$
\operatorname{SiLU}(g)\odot u\in\mathbb R^m.
$$

拼接保留两条分支作为更宽的向量，乘法则把 gate 的数值作用直接写入 value 坐标。两者的输出宽度、参数合同和 Jacobian 都不同。

### 门控乘法不是 attention

attention 可以按 query 位置读取其他 token 的 value：

$$
\Delta_{t}=\sum_s A_{t,s}V_s.
$$

SwiGLU 的乘法是

$$
a_t=\operatorname{SiLU}(g_t)\odot u_t.
$$

它没有 $s$ 这个位置求和，也没有 attention matrix。上下文已经由前面的 attention 写入 $h_t$ 后，SwiGLU 才对 $h_t$ 做逐 token 的条件特征变换。

## SiLU 门值的性质

### 导数

由

$$
\operatorname{SiLU}(v)=v\sigma(v)
$$

得到

$$
\operatorname{SiLU}'(v)
=\sigma(v)+v\sigma(v)(1-\sigma(v)).
$$

也可以写成

$$
\operatorname{SiLU}'(v)
=\sigma(v)\bigl(1+v(1-\sigma(v))\bigr).
$$

在 $v=0$ 处，

$$
\operatorname{SiLU}'(0)=0.5.
$$

在数值例子的 $v=1$ 处，

$$
\operatorname{SiLU}'(1)
=0.927670511871.
$$

SiLU 在零点处仍有非零导数，和 ReLU 的负侧零导数不同。

### 负门值和过量门值

当 $v<0$ 时，$\operatorname{SiLU}(v)$ 可以为负；当 $v>0$ 且足够大时，$\operatorname{SiLU}(v)$ 接近 $v$，可以大于 1。门值的语义是连续调制系数，而不是概率。

| gate pre-activation | SiLU 门值 | value 为正时的效果 |
| ---: | ---: | --- |
| 大负值 | 接近 0 的负值 | 小幅反向写入 |
| 0 | $0$ | 完全关闭 |
| 小正值 | 小正值 | 弱正向写入 |
| 大正值 | 接近输入值 | 放大 value |

如果只记录 sigmoid 而不记录 SiLU 输出，会把真实门控幅度错误地限制在 $[0,1]$。

### Gate 和 value 相关时不能直接相乘期望

若随机变量 $g$ 和 $u$ 由同一个 $h$ 产生，通常相关：

$$
g=W_gh,
\qquad
u=W_uh.
$$

因此

$$
\mathbb E[\operatorname{SiLU}(g)\odot u]
\ne
\mathbb E[\operatorname{SiLU}(g)]
\odot
\mathbb E[u]
$$

一般不成立。用独立变量假设估计激活方差时，必须标记这是近似，不是模型恒等式。

## SwiGLU 的局部 Jacobian

### 先写三个微分

令

$$
s=\operatorname{SiLU}(g),
\qquad
s'=\operatorname{SiLU}'(g).
$$

因为

$$
dg=W_g\,dh,
\qquad
du=W_u\,dh,
$$

逐坐标乘法的微分为

$$
da
=\operatorname{diag}(u\odot s')\,dg
+\operatorname{diag}(s)\,du.
$$

代入两条上投影：

$$
da
=
\left[
\operatorname{diag}(u\odot s')W_g
+\operatorname{diag}(s)W_u
\right]dh.
$$

### 完整 Jacobian

下投影给出

$$
dy=W_d\,da.
$$

所以 SwiGLU 的局部 Jacobian 为

$$
J_{\mathrm{SwiGLU}}
=W_d
\left[
\operatorname{diag}(u\odot s')W_g
+\operatorname{diag}(s)W_u
\right].
$$

这里有两条输入路径：

1. $W_g$ 先改变 gate，再通过 $u\odot s'$ 调节输出；
2. $W_u$ 直接改变 value，再乘以当前 gate $s$。

当某个 gate $s_j$ 接近零时，value 分支通过该坐标的局部路径会变小；当 $u_j$ 接近零时，gate 分支通过该坐标的局部路径会变小。两条路径的有效性取决于同一个 token 的 gate 和 value。

### 二维例子的 Jacobian

数值例子中

$$
s=(0,0.731058578630)^{\mathsf T},
\qquad
s'=(0.5,0.927670511871)^{\mathsf T},
\qquad
u=(2,-1)^{\mathsf T}.
$$

代入得到

$$
J_{\mathrm{SwiGLU}}
=
\begin{pmatrix}
-0.829364545251&0.901694033379\\
-1.658729090501&-0.696611933241
\end{pmatrix}.
$$

取

$$
\delta h=
\begin{pmatrix}
0.01\\
-0.02
\end{pmatrix}.
$$

局部 Jacobian 预测的输出变化为

$$
J_{\mathrm{SwiGLU}}\delta h
=
\begin{pmatrix}
-0.026327526120\\
-0.002655052237
\end{pmatrix}.
$$

直接重新计算非线性前向，得到

$$
\Delta y_{\mathrm{direct}}
=
\begin{pmatrix}
-0.025996194561\\
-0.002492372456
\end{pmatrix}.
$$

两者接近但不完全相同，差异来自 SiLU 在该扰动区间内的曲率。局部 Jacobian 是一阶近似，不是非线性函数的全局恒等式。

### Jacobian 的秩和门控状态

由于

$$
J_{\mathrm{SwiGLU}}
=W_d(C_g+C_u),
$$

其中

$$
C_g=\operatorname{diag}(u\odot s')W_g,
\qquad
C_u=\operatorname{diag}(s)W_u,
$$

有

$$
\operatorname{rank}(J_{\mathrm{SwiGLU}})
\le
\min(d,m).
$$

如果某些 $s_j$ 和 $u_js'_j$ 同时很小，对应中间坐标对两个输入路径都贡献很小。门控稀疏性会改变局部 Jacobian，但不自动意味着整体函数低秩；要判断低秩，需要计算或估计整个矩阵。

## 反向传播：梯度如何经过 gate 和 value

### 先回传到门控乘积

设上游梯度为

$$
g_y=\frac{\partial L}{\partial y}\in\mathbb R^d.
$$

下投影的反向为

$$
g_a=\frac{\partial L}{\partial a}
=W_d^{\mathsf T}g_y.
$$

因为

$$
a=s\odot u,
$$

分别有

$$
g_s=g_a\odot u,
$$

$$
g_u=g_a\odot s.
$$

再穿过 SiLU：

$$
g_g=g_s\odot\operatorname{SiLU}'(g).
$$

最后合并两条输入路径：

$$
g_h=W_g^{\mathsf T}g_g+W_u^{\mathsf T}g_u.
$$

带 bias 时，参数梯度为

$$
\frac{\partial L}{\partial W_d}=g_ya^{\mathsf T},
\qquad
\frac{\partial L}{\partial b_d}=g_y,
$$

$$
\frac{\partial L}{\partial W_g}=g_gh^{\mathsf T},
\qquad
\frac{\partial L}{\partial b_g}=g_g,
$$

$$
\frac{\partial L}{\partial W_u}=g_uh^{\mathsf T},
\qquad
\frac{\partial L}{\partial b_u}=g_u.
$$

门值小会减小 $g_u$，value 小会减小 $g_s$；前向中的门控也会在反向中改变两条分支收到的梯度。

### 二维例子的反向账本

对数值例子取

$$
g_y=
\begin{pmatrix}
1\\
2
\end{pmatrix}.
$$

下投影反向为

$$
g_a=W_d^{\mathsf T}g_y
=
\begin{pmatrix}
0\\
2.5
\end{pmatrix}.
$$

乘法的两条梯度为

$$
g_s=g_a\odot u
=
\begin{pmatrix}
0\\
-2.5
\end{pmatrix},
$$

$$
g_u=g_a\odot s
=
\begin{pmatrix}
0\\
1.827646446575
\end{pmatrix}.
$$

SiLU 反向给出

$$
g_g=g_s\odot s'
=
\begin{pmatrix}
0\\
-2.319176279679
\end{pmatrix}.
$$

最后

$$
g_h=W_g^{\mathsf T}g_g+W_u^{\mathsf T}g_u
=
\begin{pmatrix}
-4.146822726254\\
-0.491529833104
\end{pmatrix}.
$$

本例的参数梯度为

$$
\frac{\partial L}{\partial W_d}
=
\begin{pmatrix}
0&-0.731058578630\\
0&-1.462117157260
\end{pmatrix},
$$

$$
\frac{\partial L}{\partial W_g}
=
\begin{pmatrix}
0&0\\
-2.319176279679&0
\end{pmatrix},
\qquad
\frac{\partial L}{\partial W_u}
=
\begin{pmatrix}
0&0\\
1.827646446575&0
\end{pmatrix}.
$$

三个 bias 的梯度分别是 $g_g$、$g_u$ 和 $g_y$。第 1 个 gate 坐标的 $g_s$ 为零，是因为本例的 $g_{a,1}=0$，而不是因为 SiLU 在零点没有导数；SiLU 在零点的导数为 $0.5$。

### 序列中的参数梯度累加

把 $B\times T$ 个 token 编号为 $i=1,\ldots,N$，每个 token 有独立的局部中间量，但共享同一层的参数。于是

$$
\frac{\partial L}{\partial W_d}
=\sum_{i=1}^{N}g_{y,i}a_i^{\mathsf T},
$$

$$
\frac{\partial L}{\partial W_g}
=\sum_{i=1}^{N}g_{g,i}h_i^{\mathsf T},
$$

$$
\frac{\partial L}{\partial W_u}
=\sum_{i=1}^{N}g_{u,i}h_i^{\mathsf T}.
$$

参数梯度累加不等于 token 前向混合。token 之间的梯度相加发生在参数更新账本中，SwiGLU 的一次前向仍只读取当前 token 的 $h_i$。

## 参数量、计算量与宽度预算

### 三矩阵参数量

忽略 bias 时，

$$
P_{\mathrm{SwiGLU,no\ bias}}
=md+md+dm
=3dm.
$$

包含三个 bias 时，

$$
P_{\mathrm{SwiGLU,bias}}
=3dm+2m+d.
$$

三个矩阵的职责不能用普通 FFN 的 $2dm$ 公式代替。

### 4096/11008 配置

取

$$
d=4096,
\qquad
m=11008.
$$

有

$$
dm=45\,088\,768,
$$

$$
P_{\mathrm{SwiGLU,no\ bias}}
=135\,266\,304,
$$

$$
P_{\mathrm{SwiGLU,bias}}
=135\,292\,416.
$$

如果以 FP16 保存无 bias 权重，参数占用为

$$
135\,266\,304\times2
=270\,532\,608\ \mathrm{bytes}
=258\ \mathrm{MiB}.
$$

### 与普通 FFN 的公平比较

如果普通 FFN 和 SwiGLU 使用相同的中间宽度 $m$，参数量比为

$$
\frac{3dm}{2dm}=1.5.
$$

但常见比较不是固定相同的 $m$。普通 FFN 若取 $m=4d$，无 bias 参数量为

$$
2d(4d)=8d^2.
$$

令 SwiGLU 与它参数量相同，需要

$$
3dm=8d^2,
$$

即

$$
m=\frac83d.
$$

当 $d=4096$ 时，

$$
\frac83d=10922.666\ldots.
$$

实际实现会把 $m$ 调整为硬件友好的整数。$m=11008$ 时，SwiGLU 的无 bias 参数量为 $135\,266\,304$，普通 $4d=16384$ FFN 为 $134\,217\,728$，两者比值为

$$
\frac{135\,266\,304}{134\,217\,728}
=1.0078125.
$$

因此必须说明比较的是相同中间宽度，还是大致相同参数预算。只写「SwiGLU 有三个矩阵」不能判断实际模型的资源差异。

### MAC 账本

每个 token 需要两次上投影和一次下投影，矩阵乘法的 MAC 数为

$$
C_{\mathrm{MAC/token}}
=3dm.
$$

对 $d=4096,m=11008$：

$$
C_{\mathrm{MAC/token}}
=135\,266\,304.
$$

取

$$
B=2,
\qquad
T=4096,
\qquad
N=8192,
$$

三次投影合计

$$
N\cdot3dm
=1\,108\,101\,562\,368
$$

个 MAC。这个账本不包含 SiLU、逐元素乘法、bias、读写和 kernel workspace；若报告 FLOPs，还要说明一次 MAC 是否按两个 FLOPs 计算。

## 中间激活和融合实现

### 未融合时的激活数量

仍取 $B=2,T=4096,d=4096,m=11008$。以下每个中间数组都有

$$
BTm
=90\,177\,536
$$

个元素。FP16 下单个数组为

$$
90\,177\,536\times2
=172\ \mathrm{MiB}.
$$

可能出现的数组包括：

| 数组 | 含义 | FP16 大小 |
| --- | --- | ---: |
| $G$ | gate pre-activation | $172$ MiB |
| $U$ | value pre-activation | $172$ MiB |
| $\operatorname{SiLU}(G)$ | gate 输出 | $172$ MiB |
| $A$ | 门控乘积 | $172$ MiB |

如果实现同时物化四个数组，理论上仅这部分就达到

$$
4\times172=688\ \mathrm{MiB}.
$$

实际 kernel 可以复用 buffer、在线完成 SiLU 和乘法、或在反向时重算，因此 688 MiB 是特定保存策略下的账本，不是所有实现的固定占用。

### Fused SwiGLU 的边界

融合 kernel 可能把以下操作合在一个执行路径：

1. 读取 $h$ 和 gate/up 权重；
2. 计算 $G$ 和 $U$；
3. 对 $G$ 计算稳定的 SiLU；
4. 逐坐标计算 $\operatorname{SiLU}(G)\odot U$；
5. 乘以 $W_d^{\mathsf T}$ 并写出 $Y$。

融合减少中间数组的 HBM 往返，但不改变数学函数。核验时要分别比较未融合参考和 fused 输出，不能把融合后的中间量缺失误认为门控没有发生。

### 反向的保存策略

训练反向可能需要 $G,U$ 或它们的等价信息。常见策略为：

| 策略 | 前向保存 | 反向影响 |
| --- | --- | --- |
| 全保存 | $G,U,\operatorname{SiLU}(G),A$ | 读取多，重算少 |
| 保存 $G,U$ | 重算门控乘积 | 节省一个数组 |
| 只保存 $h$ | 反向重算三次投影 | 激活少，计算多 |
| checkpoint | 保存层边界 stream | 以多次前向换内存 |

报告内存时要区分静态权重、临时数组、反向保存量和 kernel workspace。

## SwiGLU 与 residual stream 的接口

### Pre-norm 的串行写回

在 pre-norm block 中，SwiGLU 通常读取 attention 写回后的归一化 stream：

$$
h_l=N_{\mathrm{ffn}}(x_{l+\frac12}),
$$

$$
\Delta_{\mathrm{SwiGLU},l}
=W_{d,l}
\left[
\operatorname{SiLU}(W_{g,l}h_l+b_{g,l})
\odot
(W_{u,l}h_l+b_{u,l})
\right]
b_{d,l},
$$

$$
x_{l+1}=x_{l+\frac12}+\Delta_{\mathrm{SwiGLU},l}.
$$

门控内部是 $m$ 维，写回增量是 $d$ 维。不能把 $a$ 直接加到 $x_{l+\frac12}$。

### Post-norm 的归一化位置

post-norm 中可以抽象为

$$
\widetilde x_{l+1}
=x_{l+\frac12}+\operatorname{SwiGLU}(x_{l+\frac12}),
$$

$$
x_{l+1}=N_{\mathrm{ffn}}(\widetilde x_{l+1}).
$$

归一化位置会改变 SwiGLU 的输入统计和最终 Jacobian。相同的 gate/up/down 权重放在不同 norm 结构中，不代表得到相同输出。

### 并行分支

并行 block 可能令 attention 和 SwiGLU 都读取 $x_l$：

$$
x_{l+1}
=x_l
+F_{\mathrm{attn}}(N_{\mathrm{attn}}(x_l))
+\operatorname{SwiGLU}(N_{\mathrm{ffn}}(x_l)).
$$

此时 SwiGLU 不读取 attention 当步的增量。审计计算图时必须记录 gate/up 两个投影的输入是哪个 stream 版本。

### 每层参数通常独立

不同层通常有

$$
W_{g,l},W_{u,l},W_{d,l},
$$

而同一层的三矩阵跨 batch 和位置共享。把 gate projection、up projection 或 down projection 误共享到所有层，会改变模型的参数量和层间功能分工。

## Padding、Packed Sequence 和 mask

### Padding 仍会经过三条投影

padding 位置也拥有

$$
h_{b,t,:}\in\mathbb R^d.
$$

普通 SwiGLU 会计算

$$
g_{b,t,:}=W_gh_{b,t,:}+b_g,
$$

$$
u_{b,t,:}=W_uh_{b,t,:}+b_u,
$$

再形成 $a_{b,t,:}$ 和 $y_{b,t,:}$。attention mask 不会自动跳过 gate/up/down 三次矩阵乘法。

### 零输入不一定产生零输出

即使 padding 的 $h=0$，只要 bias 存在，

$$
g=b_g,
\qquad
u=b_u,
\qquad
y=W_d\bigl(\operatorname{SiLU}(b_g)\odot b_u\bigr)+b_d
$$

就可能非零。要让 padding stream 保持零，必须明确执行计算 mask、输出 mask 或 residual add 后清零。

### Packed sequence 的 FFN 不跨位置

多个短序列 packed 到同一时间轴时，SwiGLU 的三条投影都只读取当前行。它不会因为相邻位置属于不同 segment 就自动互读。

但 attention 的 segment mask 仍必须正确。若某个实现把 token 维拼接后又在 SwiGLU 中执行跨行归约，它已经不再是标准 token-wise SwiGLU。

### Loss mask 不等于计算 mask

只在损失中使用

$$
M_{b,t}\in\{0,1\}
$$

可以排除 padding 的 loss，但不会减少 SwiGLU 的 gate/up/down 计算。要减少 padding 的计算量，需要在 kernel 输入或 token 索引层面显式过滤。

## 精度、稳定性和量化

### SiLU 需要稳定的 sigmoid

直接计算

$$
\sigma(v)=\frac{1}{1+e^{-v}}
$$

在大负值时可能产生指数溢出。稳定实现会按 $v$ 的符号选择等价表达式，保持有限值，再计算 $v\sigma(v)$。

输入极端时的边界为

$$
\lim_{v\to+\infty}\operatorname{SiLU}(v)=+\infty,
$$

$$
\lim_{v\to-\infty}\operatorname{SiLU}(v)=0^-.
$$

数值实现需要区分数学极限和有限 dtype 下的实际范围。

### 低精度会同时影响 gate 和 value

SwiGLU 的乘法误差来自至少三处：

1. gate projection 的 $G$ 舍入；
2. SiLU 的 sigmoid、乘法和输出舍入；
3. value projection 的 $U$ 舍入；
4. $S\odot U$ 的乘法舍入；
5. down projection 的累加舍入。

只比较最终 $Y$ 无法定位误差。应分别对比 FP32 参考、目标 dtype、fused kernel 的 $G,U,S,A,Y$。

### 小 gate 可能放大相对误差

当 $s_j$ 接近零时，绝对输出可能很小，但相对误差会很大。此时要同时报告绝对误差和相对误差：

$$
\operatorname{abs\_err}
=\lVert y_{\mathrm{impl}}-y_{\mathrm{ref}}\rVert_2,
$$

$$
\operatorname{rel\_err}
=\frac{\operatorname{abs\_err}}
{\lVert y_{\mathrm{ref}}\rVert_2+\delta}.
$$

如果 reference 本身接近零，只看 relative error 会夸大数值差异。

### 量化不能只量化一个矩阵

gate/up/down 三个矩阵的分布可能不同。量化审计至少分别记录：

| 对象 | 需要核对 |
| --- | --- |
| $W_g$ | scale、zero point、输出 $G$ 的范围 |
| $W_u$ | scale、zero point、输出 $U$ 的范围 |
| $W_d$ | scale、zero point、输出 $Y$ 的范围 |
| SiLU | sigmoid 和乘法的计算 dtype |
| gate product | $S\odot U$ 是否在更高精度中计算 |

把三组权重共用一个未经验证的 scale，可能让 gate 分布或 value 分布出现非对称误差。

## Gate 统计和功能诊断

### 记录四组统计量

每层建议分别记录

$$
\operatorname{stats}(g),
\qquad
\operatorname{stats}(s),
\qquad
\operatorname{stats}(u),
\qquad
\operatorname{stats}(a).
$$

其中 $s=\operatorname{SiLU}(g)$，$a=s\odot u$。至少包含均值、标准差、极值、分位数和有限性。

只记录 $a$ 无法区分以下情况：

1. gate 接近零；
2. value 接近零；
3. gate 和 value 符号抵消；
4. down projection 在多个坐标上抵消。

### Gate 强度不是功能解释

一个 gate 值大，只说明对应 value 坐标的乘法系数大。它不说明该坐标对应某个固定语义，也不说明最终输出一定增大。down projection 会把多个门控坐标重新组合成 $d$ 维输出。

要报告某个方向上的写入，需要给出读出向量 $w$：

$$
s_{\mathrm{readout}}
=w^{\mathsf T}y
=w^{\mathsf T}W_da.
$$

改变 $w$ 会改变对同一个门控结果的解释。

### Gate 和 value 的联合统计

可以记录

$$
\operatorname{corr}(s_j,u_j),
$$

以及

$$
\operatorname{mean}(s_j u_j),
\qquad
\operatorname{mean}(\lvert s_j u_j\rvert).
$$

相关系数只能描述线性相关，不能替代联合分布。对功能诊断，应同时保存 gate、value 和 down projection 的读出结果。

### 与普通 FFN 的对照实验

在相同输入、相同输出宽度和近似参数预算下，至少比较：

| 对照 | 需要固定 |
| --- | --- |
| 普通 FFN | $m$、激活函数、bias、权重初始化 |
| GLU 变体 | gate 函数、两条上投影、down projection |
| SwiGLU | SiLU 精确或近似实现 |
| 训练比较 | batch、mask、优化器和学习率 |
| 运行比较 | dtype、kernel、checkpoint 和随机状态 |

如果只改变激活函数而同时改变矩阵数量和中间宽度，无法把结果差异归因于门控本身。

## 失效模式

### 把 gate 和 value 误用同一矩阵

SwiGLU 需要独立的 $W_g$ 和 $W_u$。如果两者共享同一矩阵，模型退化为

$$
\operatorname{SiLU}(Wh)\odot(Wh),
$$

函数族和参数量都发生变化。

### 把 SiLU gate 当作 sigmoid 概率

SiLU 输出可以为负，也可以大于 1。把它强行截断到 $[0,1]$ 会改变模型函数。

### 把门控乘法写成拼接

$\operatorname{concat}(s,u)$ 的宽度是 $2m$，而 $s\odot u$ 的宽度是 $m$。两者不能互换，也不能依赖广播隐藏 shape 错误。

### 把三矩阵参数量写成两矩阵

无 bias 的 SwiGLU 参数量是 $3dm$，不是 $2dm$。比较模型时要同时写出 $m$ 是否相同，以及是否按参数预算调整了中间宽度。

### 忽略 down projection

门控结果 $a$ 仍然是 $m$ 维。它必须经过 $W_d$ 回到 $d_{\mathrm{model}}$ 后才能和 residual stream 相加。

### 把门控乘法当作 token mixing

SwiGLU 的 gate、value 和逐坐标乘法都固定在当前 token。跨 token 的依赖来自输入 stream 的来源，例如此前的 attention，不来自门控本身。

### 只 mask loss，不 mask padding 计算

loss mask 不会自动减少 gate/up/down 的矩阵乘法。padding 的 bias 还可能产生非零输出，需要显式清零或跳过计算。

### 把 fused kernel 的中间量缺失当作错误

融合实现可能不物化 $G,U,S,A$。应比较最终输出和分段 reference，不能要求 fused kernel 暴露与未融合实现相同的临时 buffer。

### 只检查 gate 的均值

均值接近零可能来自正负值抵消，也可能来自大量接近零的 gate。还要记录分位数、极值、value 和门控乘积。

### 把数值误差归因于 SiLU

先分段比较 gate projection、sigmoid、SiLU、value projection、门控乘法和 down projection。没有分段证据时，原因未查明。

## 一个可复用的核验协议

### 记录结构和权重

1. 记录 $d_{\mathrm{model}}$、$d_{\mathrm{ffn}}$、gate/up/down 的矩阵 shape。
2. 记录是否包含 $b_g,b_u,b_d$。
3. 记录 checkpoint 名称与 $W_g,W_u,W_d$ 的对应关系。
4. 记录 SiLU 是精确实现还是近似实现。
5. 记录 FFN 读取的是 norm 前、norm 后还是 attention 写回后的 stream。

### 运行小矩阵前向

1. 使用二维例子独立计算 $g,u,s,a,y$。
2. 比较 gate、value 和门控结果的每个坐标。
3. 记录 SiLU 的 sigmoid、门值和 down projection 输出。
4. 检查 $y$ 是否回到 $d_{\mathrm{model}}$。

### 运行 Jacobian 和反向

1. 用 $J=W_d[\operatorname{diag}(u\odot s')W_g+\operatorname{diag}(s)W_u]$ 计算解析 Jacobian。
2. 用不跨越数值异常区间的中心差分检查输入 Jacobian。
3. 用上游梯度检查 $g_a,g_s,g_u,g_g,g_h$。
4. 检查 gate、up、down 三组权重梯度和 bias 梯度的 shape。

### 运行资源和运行时检查

1. 以 $d=4096,m=11008$ 核对三矩阵参数量和 MAC。
2. 以 $B=2,T=4096$ 核对 $G,U,S,A$ 的激活内存。
3. 对未融合和 fused kernel 分别记录临时数组和 workspace。
4. 对 FP16、BF16 和 FP32 参考分别记录 SiLU、门控乘法和 down projection 的误差。
5. 对 padding、packed sequence 和 loss mask 分别检查实际计算路径。

### 解释 SwiGLU 写入

对每层至少附带：

1. 当前层和 token 位置；
2. gate、value、门控结果和 down output；
3. gate 的 SiLU 统计量；
4. $\lVert y\rVert_2/\lVert h\rVert_2$ 和指定读出变化；
5. 参数预算、中间宽度和实际 dtype；
6. mask、fused kernel、checkpoint 与 residual add 位置。

这些字段缺失时，只能确认一个门控 FFN 产生了输出，不能把输出变化归因于 gate 选择、value 特征或某个语义方向。

## 相关词条

- [前馈网络](../transformer-components/feedforward/)
- [残差流](../transformer-components/residual-streams/)
- [Swish 与 SwiGLU](../neurons-and-activations/swish-and-swiglu/)
- [激活函数](../neurons-and-activations/activation-functions/)
- [参数量总账](../transformer-components/parameter-count/)
- [Pre-norm 与 Post-norm](../transformer-components/pre-norm-vs-post-norm/)
- [混合精度训练](../training-nn/mixed-precision/)
