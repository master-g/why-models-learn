---
title: "矩阵乘法:行乘列、列组合与映射复合"
tags: ["why-models-learn"]
---

矩阵乘法是定义在两个矩阵上的运算：$A \in \mathbb{R}^{m \times n}$ 与 $B \in \mathbb{R}^{n \times p}$ 的乘积 $C = AB$ 是一个 $m \times p$ 矩阵，其第 $i$ 行第 $j$ 列元 $c_{ij}$ 等于 $A$ 的第 $i$ 行与 $B$ 的第 $j$ 列对应分量相乘再求和。它是矩阵从「装数的表格」变成「能做事的映射」的关键一步——[矩阵](../linear-algebra/matrices/) 篇许下的三个承诺(列的线性组合、映射的复合、内维相等)都要在本篇兑现。神经网络里每一层的前向传播都是一次矩阵乘法，它是全库使用频率最高的运算，没有之一。本篇给出定义与三把理解钥匙(行乘列、列视角、映射复合)，用真实数字验证运算律——结合律成立、交换律失效、转置顺序反转——最后落回神经网络语境。

## 定义：行乘列

先说形状纪律:**左侧矩阵的列数必须等于右侧矩阵的行数**。$A$ 是 $m \times n$、$B$ 是 $n \times p$ 时乘法才有定义，乘积的形状取「外维」：$(m \times n)(n \times p) \to m \times p$——内维相消，外维成型。

定义：乘积 $C = AB$ 的第 $i$ 行第 $j$ 列元，是 $A$ 的第 $i$ 行与 $B$ 的第 $j$ 列(都是 $n$ 个数)逐分量相乘再求和：

$$
c_{ij} = a_{i1} b_{1j} + a_{i2} b_{2j} + \cdots + a_{in} b_{nj} = \sum_{k=1}^{n} a_{ik} b_{kj}
$$

取一组具体的：$A$ 沿用 [矩阵](../linear-algebra/matrices/) 篇的 $2 \times 3$ 矩阵，$B$ 取 $3 \times 2$：

$$
A = \begin{pmatrix} 2 & 0 & -1 \\ 1 & 3 & 4 \end{pmatrix}, \qquad
B = \begin{pmatrix} 1 & 0 \\ 2 & 1 \\ -1 & 3 \end{pmatrix}
$$

形状检查：$(2 \times 3)(3 \times 2)$，内维 $3 = 3$ 合法，乘积是 $2 \times 2$。逐元算两个：

$$
c_{11} = (2, 0, -1) \cdot (1, 2, -1) = 2 \times 1 + 0 \times 2 + (-1) \times (-1) = 3
$$

$$
c_{22} = (1, 3, 4) \cdot (0, 1, 3) = 1 \times 0 + 3 \times 1 + 4 \times 3 = 15
$$

四个元都算完(剩下两个留给读者验证)：

$$
AB = \begin{pmatrix} 3 & -3 \\ 3 & 15 \end{pmatrix}
$$

这个定义初看别扭：为什么不直接对应位置相乘？那是 Hadamard 积(见 [向量](../linear-algebra/vectors/) 篇)，在数学里几乎是配角。行乘列才是主角，因为它恰好是「映射复合」所需要的运算——钥匙三会看到这并非巧合，而是被迫如此。

## 钥匙一：矩阵乘向量，是列的线性组合

最有用的特例是 $p = 1$：$B$ 退化为列向量 $\mathbf{x}$($n \times 1$ 矩阵)，乘积 $A\mathbf{x}$ 是 $m \times 1$，仍是一个向量。把行乘列的定义按列重新分组：

$$
A\mathbf{x} = \begin{pmatrix} a_{11} x_1 + a_{12} x_2 + \cdots + a_{1n} x_n \\ \vdots \\ a_{m1} x_1 + a_{m2} x_2 + \cdots + a_{mn} x_n \end{pmatrix} = x_1 \begin{pmatrix} a_{11} \\ \vdots \\ a_{m1} \end{pmatrix} + x_2 \begin{pmatrix} a_{12} \\ \vdots \\ a_{m2} \end{pmatrix} + \cdots + x_n \begin{pmatrix} a_{1n} \\ \vdots \\ a_{mn} \end{pmatrix}
$$

右端正是 $A$ 的各列以 $\mathbf{x}$ 的分量为系数的线性组合：

$$
A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n
$$

拿前两篇的老朋友当例子：$A$ 的列取 $\mathbf{u} = (3, 1)$、$\mathbf{v} = (1, 2)$、$\mathbf{x} = (2, 1)$：

$$
A\mathbf{x} = 2 \begin{pmatrix} 3 \\ 1 \end{pmatrix} + 1 \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 6 \\ 2 \end{pmatrix} + \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 7 \\ 4 \end{pmatrix}
$$

用行乘列验证：第一行 $(3, 1) \cdot (2, 1) = 7$，第二行 $(1, 2) \cdot (2, 1) = 4$。两种算法同一个答案。

![矩阵乘向量是列的线性组合：列 a1、a2 各按 x 的分量缩放后首尾相接，合向量 (7, 4) 就是 Ax](/assets/linear-algebra/svg/matrix-multiplication.1.svg)

上一篇视角三的承诺在此兑现：$A \mathbf{e}_j$ 是各列以 $\mathbf{e}_j$ 为系数的组合，而 $\mathbf{e}_j$ 只有第 $j$ 个分量是 1、其余都是 0，组合里只剩下第 $j$ 列——**$A \mathbf{e}_j$ 就是 $A$ 的第 $j$ 列**。「矩阵的第 $j$ 列是第 $j$ 个基向量的去向」，一行乘法即证。

这把钥匙还改变了「映射」的读法：$A\mathbf{x}$ 不再是 $m$ 次孤立的行乘列，而是「输入向量的每个分量对 $A$ 的对应列投了一票，各列加权求和得到输出」。列是基向量的去向，输出是所有去向按坐标的加权叠加——这是线性性最直接的样子,[线性组合与张成](../linear-algebra/linear-combinations-and-span/) 会把这个想法铺开。

## 钥匙二：矩阵乘法按列进行

把钥匙一逐列套用。$B$ 按列拆开写 $B = (\mathbf{b}_1 \; \mathbf{b}_2 \; \cdots \; \mathbf{b}_p)$，则：

$$
AB = \begin{pmatrix} A\mathbf{b}_1 & A\mathbf{b}_2 & \cdots & A\mathbf{b}_p \end{pmatrix}
$$

**乘积的第 $j$ 列，等于 $A$ 乘 $B$ 的第 $j$ 列。** 证明只有一行：$(AB)$ 第 $j$ 列的第 $i$ 元是 $\sum_k a_{ik} b_{kj}$，这正是 $A\mathbf{b}_j$ 的第 $i$ 元。

所以整个矩阵乘法是「$A$ 逐列变换 $B$」：$B$ 的每一列被 $A$ 映射一次，像向量并排站着，拼成 $AB$。列视角从向量升格到了矩阵。

单位矩阵的行为由此一目了然。$I_n$ 的第 $j$ 列是 $\mathbf{e}_j$，所以 $I_n \mathbf{x} = x_1 \mathbf{e}_1 + \cdots + x_n \mathbf{e}_n = \mathbf{x}$——上一篇预告的 $I_n \mathbf{v} = \mathbf{v}$ 得证，恒等变换名副其实。进而 $I_m A = A = A I_n$：单位矩阵是乘法的单位元(注意两侧的 $I$ 形状可以不同，$A$ 是 $m \times n$ 时左边站着 $I_m$、右边站着 $I_n$)。

## 钥匙三：乘积是映射的复合

现在回答定义为什么长这样。设 $B$ 把 $\mathbf{x}$ 映到 $\mathbf{y} = B\mathbf{x}$，$A$ 把 $\mathbf{y}$ 映到 $\mathbf{z} = A\mathbf{y}$。复合映射 $\mathbf{x} \mapsto \mathbf{z}$ 的矩阵是什么？把 $\mathbf{y}$ 的每个分量 $y_k = \sum_l b_{kl} x_l$ 代入 $\mathbf{z}$ 的定义：

$$
z_i = \sum_{k} a_{ik} y_k = \sum_{k} a_{ik} \left( \sum_{l} b_{kl} x_l \right) = \sum_{l} \left( \sum_{k} a_{ik} b_{kl} \right) x_l
$$

复合映射的矩阵，第 $i$ 行第 $l$ 列元正是 $\sum_k a_{ik} b_{kl}$——行乘列。**矩阵乘法的定义不是发明，是推导出来的：它是把「先做 $B$、再做 $A$」写成一个矩阵的唯一方式。** 注意顺序：$AB$ 作用在 $\mathbf{x}$ 上是 $A(B\mathbf{x})$，先 $B$ 后 $A$——矩阵从右往左读，与函数复合记号 $f \circ g$ 一致。

复合视角立刻白送一条运算律。映射复合显然有结合律——无论怎么分组,「依次施加」这件事不变——矩阵乘法因此继承：

$$
(AB)C = A(BC)
$$

用前面的 $A$、$B$，再取交换矩阵 $C = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$，两种括号真算一遍：

$$
(AB)C = A(BC) = \begin{pmatrix} -3 & 3 \\ 15 & 3 \end{pmatrix}
$$

结果相同。$C$ 的作用是交换两列：$(AB)C$ 就是把 $AB$ 的两列对调，与 $A(BC)$ 殊途同归。结合律在工程上价值极大：连乘 $A_1 A_2 \cdots A_k$ 可以任意加括号，选计算量最小的加法是数值计算的基本功——但括号只许挪动，顺序不许更换，那是下一节的事。

## 交换律失效

结合律在，交换律不在:**一般 $AB \neq BA$**。失效分三个层次，一个比一个深。

**层次一：形状都可能不答应。** $AB$ 有定义要求 $A$ 的列数等于 $B$ 的行数；$BA$ 有定义要求 $B$ 的列数等于 $A$ 的行数。两个条件互相独立：$2 \times 3$ 与 $3 \times 4$ 摆在一起，$AB$ 合法(得 $2 \times 4$)，$BA$ 直接无定义。就算两个方向都合法，形状也未必相同——前面的 $A$($2 \times 3$)与 $B$($3 \times 2$)：$AB$ 是 $2 \times 2$，$BA$ 却是 $3 \times 3$：

$$
BA = \begin{pmatrix} 2 & 0 & -1 \\ 5 & 3 & 2 \\ 1 & 9 & 13 \end{pmatrix}
$$

连形状都不同，谈不上相等。

**层次二：同形方阵也不交换。** 取上一篇的剪切 $S$，再配一个横轴拉伸矩阵 $D$：

$$
S = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \qquad D = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}
$$

$$
SD = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}, \qquad DS = \begin{pmatrix} 2 & 2 \\ 0 & 1 \end{pmatrix}
$$

两个乘积形状相同，数值不同。

**层次三：几何上本来就不该交换。** 矩阵乘法是映射的复合，而「先剪后拉」与「先拉后剪」是两种操作。看单位正方形在两种顺序下的命运：

![交换律失效：单位正方形先剪切再拉伸(左)与先拉伸再剪切(右)，终点形状不同——AB 与 BA 不是一回事](/assets/linear-algebra/svg/matrix-multiplication.2.svg)

左图先剪切(虚线是中间态)再横拉，正方形的一个角被推到 $(4, 1)$；右图先横拉再剪切，同一个角只到 $(3, 1)$。顺序一换，结果就换——复合映射的顺序就是乘法的顺序。

交换只在特例成立：与单位矩阵($IA = AI$)、与纯量矩阵 $cI$(各轴等比缩放，与任何同形方阵可交换)、以及同一矩阵的幂之间($A^2 A = A A^2$)。默认假设永远是：不可交换。

## 转置规律：顺序反转

上一篇预告的乘法转置规律，现在兑现并证明：

$$
(AB)^T = B^T A^T
$$

注意顺序反过来了。证明只有一行：$(AB)^T$ 的第 $(i, j)$ 元是 $(AB)$ 的第 $(j, i)$ 元，即「$A$ 的第 $j$ 行与 $B$ 的第 $i$ 列逐分量相乘再求和」；而 $B^T A^T$ 的第 $(i, j)$ 元是「$B^T$ 的第 $i$ 行与 $A^T$ 的第 $j$ 列逐分量相乘再求和」，也就是「$B$ 的第 $i$ 列与 $A$ 的第 $j$ 行」——同样的两列数逐对相乘再求和，加法交换律保证相等。用前面的数字验证：

$$
(AB)^T = \begin{pmatrix} 3 & 3 \\ -3 & 15 \end{pmatrix} = B^T A^T
$$

顺序为什么**必须**反？形状说了算。$A^T$ 是 $n \times m$、$B^T$ 是 $p \times n$：若写 $A^T B^T$，内维是 $m$ 对 $p$，一般不相等，乘法无定义；写 $B^T A^T$，内维是 $n$ 对 $n$，永远合法。转置把每个矩阵的内外维整个颠倒，想恢复「内维相等」只能反序相乘。这条规律之容易写错，上一篇预告它时就曾把公式写反(发现后已修正)——写完 $(AB)^T$ 先数一遍形状，是最便宜的自检。

$\mathbf{x}^T$ 的记号也在此正式登场(回收 [向量](../linear-algebra/vectors/) 篇的伏笔)：列向量 $\mathbf{x}$ 是 $n \times 1$ 矩阵，它的转置 $\mathbf{x}^T$ 是 $1 \times n$ 行向量。于是向量篇预告的两种「乘法」各就各位:**行向量乘列向量得一个数**，$\mathbf{x}^T \mathbf{y} = \sum_i x_i y_i$，这就是点积($1 \times 1$ 矩阵就是一个数);**列向量乘行向量得一个矩阵**，$\mathbf{x} \mathbf{y}^T$ 是 $n \times n$ 矩阵，第 $(i, j)$ 元是 $x_i y_j$，这就是外积。同一个 $\mathbf{x}$，横竖之别在乘法里就是形状之别。

## 神经网络里的矩阵乘法

上一篇说矩阵在神经网络里扮演数据批、权重、表示批三个角色，把它们串起来的正是矩阵乘法。

**全连接层：$\mathbf{y} = W\mathbf{x} + \mathbf{b}$。** 784 维输入经 $128 \times 784$ 的权重矩阵 $W$ 变为 128 维输出。列视角给这个运算一个语义：输出 = $W$ 的各列按输入分量加权——第 $j$ 个输入神经元对全部输出神经元的「发言权」，由 $W$ 的第 $j$ 列整体给出。「学习」就是调整这 784 列，即 $784 \times 128 = 100352$ 个分量(承接 [矩阵](../linear-algebra/matrices/) 篇的参数量账)。

**批量就是矩阵乘矩阵。** $N$ 个样本排成 $X \in \mathbb{R}^{N \times d}$(行是样本)，一层的作用是 $XW$(权重按 $d_{\text{in}} \times d_{\text{out}}$ 摆放)，输出仍是 $N$ 行——每个样本独立过层，一次矩阵乘法全部完成。这正是 GPU 的并行能力被吃满的原因，也是深度学习工程几乎全部算力的去向。

**注意力：$QK^T$。** 查询矩阵 $Q$ 与键矩阵 $K$ 都是 $N \times d$，$QK^T$ 是 $N \times N$——第 $(i, j)$ 元是第 $i$ 个查询与第 $j$ 个键的点积，即「词 $i$ 对词 $j$ 的相关程度」。转置在这里不是理论点缀，是每天都在写的形状操作，见 [注意力矩阵](../attention/attention-matrix/)。

## 失效模式与常见误区

**默认交换律。** 矩阵乘法最常见的错误是顺手把 $AB$ 当 $BA$。代码里顺序写反未必报错——两个矩阵都是方阵时形状照样合法，结果静默出错。复合视角是最好的疫苗：先问「先做什么、后做什么」，顺序由语义决定，不由书写习惯决定。

**内维不匹配被广播静默「修复」。** 三篇接力警告：写下任何乘法先标形状，$(m \times n)(n \times p) \to m \times p$，内维不等就该停笔。数学上非法的形状，代码库里可能用广播给你一个不报错但错误的答案。

**行向量列向量不分。** $\mathbf{x}^T A$ 与 $A \mathbf{x}$ 是完全不同的对象：前者是 $(1 \times n)(n \times p)$，得行向量；后者是 $(m \times n)(n \times 1)$，得列向量。漏写一个转置，形状链全错。

**转置忘反转。** 把 $(AB)^T$ 顺手写成 $A^T B^T$ 是手推时的头号笔误(上一篇真犯过)。自检法：标出每个矩阵的形状，$A^T B^T$ 的内维对不上时，错误立刻现形。

**把矩阵乘法当逐分量乘。** Hadamard 积(逐分量，见 [向量](../linear-algebra/vectors/) 篇)与矩阵乘法是两个运算，记号相近而地位悬殊。代码里 `*` 与 `@` 之分即源于此，混用不保证报错。

## 相关词条

- [矩阵](../linear-algebra/matrices/)：乘法让它的视角三(线性映射)真正兑现
- [向量](../linear-algebra/vectors/)：行/列向量是矩阵的特例，点积与外积在此闭环
- [内积](../linear-algebra/inner-products/):「行乘列」里那个乘再求和的正式身份
- [线性组合与张成](../linear-algebra/linear-combinations-and-span/)：列视角的展开——$A\mathbf{x}$ 永远落在列的张成里
- [线性映射](../linear-algebra/linear-maps/)：复合视角的公理化
- [逆矩阵](../linear-algebra/matrix-inverse/)：乘法的「撤销」，只对方阵有定义
- [秩](../linear-algebra/rank/)：乘积的秩不超过任一因子的秩，列视角可证
- [注意力矩阵](../attention/attention-matrix/)：$QK^T$ 的实战
- [MNIST 训练循环](../training-nn/mnist-mlp-training-loop/)：全连接层矩阵乘法的实战
