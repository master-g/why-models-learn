---
title: "广播与归约导数:复制、求和与梯度累加"
tags: ["why-models-learn"]
---

**广播**把一个较小形状的输入重复到更大的张量中，**归约**则沿某些坐标轴把多个数合成较小的结果。广播的反向导数要把各个副本收到的梯度相加，求和归约的反向导数要把上游梯度沿被消掉的轴复制回去。它们看起来像形状操作，实际上决定了梯度到底要加几次、除不除以坐标数量。本篇把向量、矩阵和批次张量中的复制与合并写成明确的线性映射，再连接偏置、平均损失、范数和自动微分。

## 先把三种形状分开

逐分量函数、广播和归约都可能出现在一行代码里，但它们的导数结构不同。以矩阵 $X\in\mathbb R^{m\times d}$ 和向量 $\boldsymbol b\in\mathbb R^d$ 为例：

| 操作 | 输入与输出 | 一个输出依赖哪些输入 | 反向时的主要动作 |
| --- | --- | --- | --- |
| 逐分量 $Y=\Phi(X)$ | $m\times d\to m\times d$ | 只依赖同一坐标 | 逐分量乘 $\phi'$ |
| 广播加法 $Y=X+\boldsymbol b$ | $X:m\times d,\ \boldsymbol b:d\to Y:m\times d$ | $Y_{ij}$ 依赖 $X_{ij}$ 和同一个 $b_j$ | 对 $b_j$ 收集所有行的梯度 |
| 列归约 $r_j=\sum_iX_{ij}$ | $m\times d\to d$ | 一个 $r_j$ 依赖整列 | 把 $g_{r_j}$ 复制到整列 |

逐分量导数只处理“每个坐标各走各的”情况，见 [逐分量导数](../calculus/elementwise-derivatives/)。广播引入了共享：一个 $b_j$ 被多个输出共同使用。归约引入了合并：一个输出接收多个输入的贡献。只看输入和输出的数值形状，不写清楚共享关系，很容易把求和误写成取一个副本，或把平均误写成求和。

## 广播其实是复制映射

先把向量 $\boldsymbol b=(b_1,\ldots,b_d)^{\mathsf T}$ 广播到 $m$ 行。定义复制映射

$$
\mathcal B_m(\boldsymbol b)
=
\boldsymbol 1_m\boldsymbol b^{\mathsf T}
\in\mathbb R^{m\times d}
$$

其中 $\boldsymbol 1_m$ 是每个分量都为 $1$ 的 $m$ 维列向量。因此

$$
\bigl[\mathcal B_m(\boldsymbol b)\bigr]_{ij}=b_j
$$

同一个 $b_j$ 会在第 $j$ 列出现 $m$ 次。对输入扰动 $\mathrm d\boldsymbol b$，输出扰动是

$$
\mathrm dB
=
\mathcal B_m(\mathrm d\boldsymbol b)
=
\boldsymbol 1_m(\mathrm d\boldsymbol b)^{\mathsf T}
$$

例如

$$
\boldsymbol b=
\begin{pmatrix}
2\\-1
\end{pmatrix}
\qquad
\mathcal B_3(\boldsymbol b)
=
\begin{pmatrix}
2&-1\\
2&-1\\
2&-1
\end{pmatrix}
$$

如果 $\mathrm d\boldsymbol b=(0.1,-0.2)^{\mathsf T}$，三行都会收到同一个扰动：

$$
\mathrm dB
=
\begin{pmatrix}
0.1&-0.2\\
0.1&-0.2\\
0.1&-0.2
\end{pmatrix}
$$

这不是把三个独立向量拼在一起。三个行向量共享同一个 $\mathrm d\boldsymbol b$，所以反向时不能把三行当成三个独立参数。

## 广播的反向导数要把副本相加

设某个损失 $L$ 通过 $B=\mathcal B_m(\boldsymbol b)$ 依赖 $\boldsymbol b$。用 Frobenius 内积写输出矩阵的上游梯度 $G_B$：

$$
\langle A,C\rangle_F
=
\sum_{i=1}^{m}\sum_{j=1}^{d}A_{ij}C_{ij}
$$

反向微分满足

$$
\begin{aligned}
\mathrm dL
&=\langle G_B,\mathrm dB\rangle_F\\
&=\sum_{i=1}^{m}\sum_{j=1}^{d}
G_{B,ij}\,\mathrm d b_j\\
&=\sum_{j=1}^{d}
\left(\sum_{i=1}^{m}G_{B,ij}\right)\mathrm d b_j
\end{aligned}
$$

因此

$$
\frac{\partial L}{\partial b_j}
=
\sum_{i=1}^{m}\frac{\partial L}{\partial B_{ij}}
$$

用矩阵记号就是按行求和：

$$
\nabla_{\boldsymbol b}L
=
\sum_{i=1}^{m}G_{B,i:}
$$

取

$$
G_B=
\begin{pmatrix}
1&2\\
3&4\\
-1&5
\end{pmatrix}
$$

则

$$
\nabla_{\boldsymbol b}L
=
\begin{pmatrix}
1+3-1\\
2+4+5
\end{pmatrix}
=
\begin{pmatrix}
3\\11
\end{pmatrix}
$$

每一行都贡献了自己的敏感度。若只取最后一行，会得到 $( -1,5)^{\mathsf T}$；若把三行梯度平均，则又引入了一个不属于复制映射的 $1/m$。

![广播在前向复制输入，反向沿共享坐标求和梯度](/assets/calculus/svg/broadcast-and-reduction-derivatives.1.svg)

## 加法层中的广播

神经网络的偏置就是最常见的广播输入。对

$$
Y=X+\mathcal B_m(\boldsymbol b)
$$

有

$$
\mathrm dY
=
\mathrm dX+\mathcal B_m(\mathrm d\boldsymbol b)
$$

如果损失对 $Y$ 的梯度是 $G_Y$，则

$$
G_X=G_Y
\qquad
\nabla_{\boldsymbol b}L
=
\sum_{i=1}^{m}G_{Y,i:}
$$

矩阵 $X$ 的梯度保持原形状；偏置的梯度则把所有使用它的行合并。这个形状关系不依赖损失具体长什么样，只依赖加法的广播规则。

取

$$
X=
\begin{pmatrix}
1&2\\
3&4
\end{pmatrix}
\qquad
\boldsymbol b=
\begin{pmatrix}
0.5\\-1
\end{pmatrix}
\qquad
T=
\begin{pmatrix}
0&1\\
2&2
\end{pmatrix}
$$

令

$$
Y=X+\mathcal B_2(\boldsymbol b)
=
\begin{pmatrix}
1.5&1\\
3.5&3
\end{pmatrix}
\qquad
E=Y-T=
\begin{pmatrix}
1.5&0\\
1.5&1
\end{pmatrix}
$$

对于平方损失

$$
L(\boldsymbol b)
=
\frac12\|Y-T\|_F^2
=
2.75
$$

上游梯度是 $G_Y=E$，所以

$$
\nabla_XL
=
\begin{pmatrix}
1.5&0\\
1.5&1
\end{pmatrix}
\qquad
\nabla_{\boldsymbol b}L
=
\begin{pmatrix}
1.5+1.5\\
0+1
\end{pmatrix}
=
\begin{pmatrix}
3\\1
\end{pmatrix}
$$

同一个偏置坐标 $b_1$ 影响两行，因此它收到 $1.5+1.5$；$b_2$ 的两次贡献是 $0$ 和 $1$。把偏置梯度保留为 $2\times2$ 的矩阵，会把共享参数误当成了两个参数。

## 求和归约的正向导数

列求和是广播的线性对偶。定义

$$
\mathcal R_m(X)_j
=
\sum_{i=1}^{m}X_{ij}
\qquad
\mathcal R_m:\mathbb R^{m\times d}\to\mathbb R^d
$$

对扰动逐项求和：

$$
\mathrm dr_j
=
\sum_{i=1}^{m}\mathrm dX_{ij}
$$

因此

$$
\mathrm dr
=
\mathcal R_m(\mathrm dX)
$$

例如

$$
X=
\begin{pmatrix}
1&2\\
3&4\\
-1&5
\end{pmatrix}
\qquad
\mathcal R_3(X)
=
\begin{pmatrix}
3\\11
\end{pmatrix}
$$

取

$$
\mathrm dX=
\begin{pmatrix}
0.1&-0.2\\
0&0.3\\
-0.1&0.4
\end{pmatrix}
$$

就有

$$
\mathrm dr
=
\begin{pmatrix}
0.1+0-0.1\\
-0.2+0.3+0.4
\end{pmatrix}
=
\begin{pmatrix}
0\\0.5
\end{pmatrix}
$$

求和归约是线性映射，所以它的导数在每一点都相同。每个输出坐标只收集对应列的输入坐标；如果把所有矩阵元素求成一个标量，则这个标量会收集所有坐标。

平均归约只多一个常数：

$$
\mathcal M_m(X)_j
=
\frac1m\sum_{i=1}^{m}X_{ij}
=
\frac1m\mathcal R_m(X)_j
$$

所以

$$
\mathrm d\mathcal M_m(X)
=
\frac1m\mathcal R_m(\mathrm dX)
$$

在上面的 $X$ 上，列平均为

$$
\mathcal M_3(X)
=
\begin{pmatrix}
1\\11/3
\end{pmatrix}
$$

平均把每个坐标的前向贡献除以 $m$，反向也要把上游梯度除以 $m$。不能因为两种操作都叫“归约”就把 sum 和 mean 混用。

## 归约的反向规则是沿被消掉的轴复制

设 $r=\mathcal R_m(X)$，上游梯度为 $\boldsymbol g_r$。由

$$
\begin{aligned}
\mathrm dL
&=\boldsymbol g_r^{\mathsf T}\mathrm dr\\
&=\sum_{j=1}^{d}g_{r_j}
\sum_{i=1}^{m}\mathrm dX_{ij}\\
&=\sum_{i=1}^{m}\sum_{j=1}^{d}
g_{r_j}\,\mathrm dX_{ij}
\end{aligned}
$$

可以读出

$$
\frac{\partial L}{\partial X_{ij}}
=
g_{r_j}
$$

即

$$
G_X
=
\mathcal B_m(\boldsymbol g_r)
=
\boldsymbol 1_m\boldsymbol g_r^{\mathsf T}
$$

如果 $\boldsymbol g_r=(2,-1)^{\mathsf T}$，那么

$$
G_X=
\begin{pmatrix}
2&-1\\
2&-1\\
2&-1
\end{pmatrix}
$$

归约在正向合并了三行，反向就让同一个上游分量回到三行。这个“复制”不是额外的学习参数，而是链式法则对同一个输出依赖多个输入的直接展开。

对平均归约 $r=\mathcal M_m(X)$，则

$$
G_X
=
\frac1m\mathcal B_m(\boldsymbol g_r)
$$

当 $m=3$ 且 $\boldsymbol g_r=(2,-1)^{\mathsf T}$ 时，每一行都是 $(2/3,-1/3)$。前向先除以三，反向不能忘记同一个比例。

## 广播和求和互为伴随

在 Frobenius 内积和向量内积下，广播与求和满足

$$
\left\langle
\mathcal B_m(\boldsymbol v),X
\right\rangle_F
=
\left\langle
\boldsymbol v,\mathcal R_m(X)
\right\rangle
$$

因为左边是

$$
\sum_{i=1}^{m}\sum_{j=1}^{d}v_jX_{ij}
=
\sum_{j=1}^{d}v_j\left(\sum_{i=1}^{m}X_{ij}\right)
$$

所以

$$
\mathcal B_m^{*}=\mathcal R_m
\qquad
\mathcal R_m^{*}=\mathcal B_m
$$

这里的上标 $*$ 表示相对于内积的伴随映射，不是逐项取倒数。反向模式微分正是在计算这种伴随：先得到输出空间的敏感度，再沿局部线性映射的伴随回传到输入空间。[向量链式法则](../calculus/vector-chain-rule/) 给出一般矩阵链式法则；广播和归约只是其中形状有重复或合并的特殊线性映射。

## 批次和多轴张量中的广播

广播规则可以用“对齐形状”来描述。把较短的形状从右侧对齐，某个输入轴如果缺失，或者长度为 $1$，就会在该轴上重复。反向时，凡是输入中被重复的轴，都要对输出梯度求和。

| 共享输入形状 | 输出形状 | 反向对输出梯度求和的轴 |
| --- | --- | --- |
| $(D)$ | $(B,T,D)$ | $B,T$ |
| $(1,D)$ | $(B,T,D)$ | $B,T$ |
| $(B,1)$ | $(B,T)$ | $T$ |
| 标量 | $(B,T,D)$ | $B,T,D$ |

例如神经网络中的偏置 $\boldsymbol b\in\mathbb R^D$ 加到批次和序列上：

$$
Z_{n,t,k}
=
U_{n,t,k}+b_k
$$

一个偏置坐标 $b_k$ 被所有 $B$ 个样本、所有 $T$ 个时间位置共同使用。因此

$$
\frac{\partial L}{\partial b_k}
=
\sum_{n=1}^{B}\sum_{t=1}^{T}
\frac{\partial L}{\partial Z_{n,t,k}}
$$

对 $\boldsymbol b$ 的梯度最终仍然只有 $D$ 个数，但每个数可能累加 $B\times T$ 个上游贡献。形状检查能提前发现一个常见错误：把 $(D)$ 的梯度误保留成 $(B,T,D)$，或只沿一个共享轴求和。

如果输入的形状是 $(B,1)$，它在时间轴上广播，而不是在批次轴上广播。对于每个固定的 $n$，只有 $T$ 个位置共享同一个参数 $c_n$，所以反向只沿 $T$ 求和：

$$
\frac{\partial L}{\partial c_n}
=
\sum_{t=1}^{T}
\frac{\partial L}{\partial Y_{n,t}}
$$

不能只记“反向要 sum”，还要问“沿哪一根轴 sum”。

## 平均损失中的缩放不要重复

批次训练经常把样本损失取平均。设

$$
y_i=x_i+b
\qquad
r_i=y_i-t_i
$$

平方损失可以写成两种约定：

$$
L_{\mathrm{sum}}
=
\frac12\sum_{i=1}^{m}r_i^2
\qquad
L_{\mathrm{mean}}
=
\frac1{2m}\sum_{i=1}^{m}r_i^2
$$

因为 $b$ 被每个样本广播使用，

$$
\frac{\partial L_{\mathrm{sum}}}{\partial b}
=
\sum_{i=1}^{m}r_i
\qquad
\frac{\partial L_{\mathrm{mean}}}{\partial b}
=
\frac1m\sum_{i=1}^{m}r_i
$$

取

$$
\boldsymbol x=(1,2,4)
\qquad
b=0.5
\qquad
\boldsymbol t=(0,1,3)
$$

得到

$$
\boldsymbol r=(1.5,1.5,1.5)
$$

两种约定的数值不同：

| 损失约定 | 损失值 | 对 $b$ 的导数 |
| --- | --- | --- |
| $L_{\mathrm{sum}}$ | $3.375$ | $4.5$ |
| $L_{\mathrm{mean}}$ | $1.125$ | $1.5$ |

如果上游损失已经含有 $1/m$，广播反向只负责把各样本的梯度相加，不要在“还原偏置形状”时再除一次。反过来，如果损失是 sum，框架也不应凭空替你改成 mean。

## 不是所有缩维都是简单归约

求和和平均是线性归约，max、范数和归一化还要看它们怎样依赖输入。

对列最大值

$$
r_j=\max_iX_{ij}
$$

如果最大值位置唯一，梯度只回到该位置：

$$
\frac{\partial r_j}{\partial X_{ij}}
=
\begin{cases}
1,&i=\operatorname*{arg\,max}_qX_{qj}\\
0,&\text{其他位置}
\end{cases}
$$

有并列最大值时，经典导数可能不存在，具体实现需要选择一个次梯度或规定如何分配梯度。它不是把上游梯度复制到整列。

平方范数可以拆成逐分量平方再求和：

$$
\|\boldsymbol x\|_2^2
=
\sum_i x_i^2
\qquad
\nabla_{\boldsymbol x}\|\boldsymbol x\|_2^2
=
2\boldsymbol x
$$

而

$$
\nabla_{\boldsymbol x}\|\boldsymbol x\|_2
=
\frac{\boldsymbol x}{\|\boldsymbol x\|_2}
\qquad
\boldsymbol x\ne\boldsymbol0
$$

在零向量处，范数不可导。平方范数的求导可以按“逐分量加归约”拆开，普通范数则还要处理平方根和零点边界。[范数](../linear-algebra/norms/) 词条给出这些量的几何读法。

Softmax 也会先逐分量取指数，但再除以共享的总和；它的一个输出依赖所有输入坐标，Jacobian 通常是稠密的。广播规则只能用于确实被复制的轴，不能把所有“形状变化”都当成广播。

## 变形和转置不会产生梯度副本

reshape、转置和轴置换改变坐标的排列或视图，但没有把一个输入数值用于多个输出位置。若 $Y$ 只是 $X$ 的转置：

$$
Y_{ij}=X_{ji}
$$

则反向梯度也是转置：

$$
G_X=G_Y^{\mathsf T}
$$

reshape 的反向操作是按原来的形状 reshape 回去。只有当一个输入坐标对应多个输出坐标时，反向才需要求和；只有当多个输入坐标共同影响一个输出坐标时，反向才需要把上游分量送回多个位置。先画出坐标对应关系，再决定是重排、复制还是归约。

## 用有限差分检查复制与累加

继续使用广播加法的例子，令

$$
L(\boldsymbol b)
=
\frac12\left\|X+\mathcal B_2(\boldsymbol b)-T\right\|_F^2
$$

在 $\boldsymbol b=(0.5,-1)^{\mathsf T}$，解析梯度是 $(3,1)^{\mathsf T}$。对第 $j$ 个偏置坐标使用中心差分：

$$
\widehat g_j(h)
=
\frac{L(\boldsymbol b+h\boldsymbol e_j)-L(\boldsymbol b-h\boldsymbol e_j)}{2h}
$$

取 $h=10^{-4}$，得到

| 扰动方向 | 解析梯度 | 中心差分 |
| --- | --- | --- |
| $\boldsymbol e_1$ | $3$ | $3.000000$ |
| $\boldsymbol e_2$ | $1$ | $1.000000$ |

这个检查同时验证了两件事：$b_1$ 的两行贡献确实相加成 $3$，$b_2$ 的两行贡献确实相加成 $1$。如果实现只留下某一行，或错误地做了平均，有限差分会直接暴露比例不对。对数值更复杂的归约，还要改变 $h$ 检查舍入误差与截断误差的平衡，方法见 [数值微分](../calculus/numerical-differentiation/)。

## 常见失效模式

- **只保留一个副本的梯度。** 先列出一个共享输入被哪些输出位置使用，再沿这些位置求和。
- **把 sum 和 mean 混成同一条规则。** mean 的前向有 $1/m$，反向也必须有 $1/m$。
- **沿错轴还原形状。** 按从右侧对齐后的形状标出长度为 $1$ 或缺失的轴，只对这些轴求和。
- **平均损失重复除批次大小。** 先确认上游梯度是否已经包含样本平均因子，再做广播反向。
- **把 max 当成求和。** 唯一最大值只接收一份梯度，并列最大值还涉及次梯度约定。
- **把 reshape 当成广播。** 重排坐标不复制数值，反向只需执行逆排列或恢复原形状。
- **忽略共享统计量。** softmax、归一化和范数中有些部分会让不同坐标互相依赖，不能只按逐分量导数处理。

## 相关词条

- [逐分量导数](../calculus/elementwise-derivatives/)：同形状逐分量函数的对角 Jacobian。
- [雅可比矩阵](../calculus/jacobian/)：把复制和归约看成局部线性映射。
- [向量链式法则](../calculus/vector-chain-rule/)：组合这些映射并按形状排列矩阵乘法。
- [梯度](../calculus/gradient/)：把广播参数收到的多处敏感度合成为一个向量。
- [全导数](../calculus/total-derivative/)：用线性映射统一描述微小扰动。
- [范数](../linear-algebra/norms/)：平方和归约与零点不可导边界。
- [自动微分](../calculus/automatic-differentiation/)：在计算图中实现局部导数及其伴随。
- [数值微分](../calculus/numerical-differentiation/)：用有限差分检查广播和归约的实现。
