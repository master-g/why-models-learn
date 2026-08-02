---
title: "线性组合与张成:系数、集合与「能不能拼出」"
tags: ["why-models-learn"]
---

线性组合(linear combination)是加法与数乘的有限次使用：给定 $k$ 个向量 $\mathbf{v}_1, \dots, \mathbf{v}_k$ 与 $k$ 个实数 $c_1, \dots, c_k$，和 $c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k$ 就是这组向量的一个线性组合，$c_i$ 叫组合的**系数**。全部线性组合构成的集合，叫这组向量的**张成**(span)，记作 $\operatorname{span}\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$。这两个词前面几篇已经反复使用：矩阵乘向量是列的线性组合([矩阵乘法](../linear-algebra/matrix-multiplication/))，解方程组是求拼出 $\mathbf{b}$ 的配方([线性方程组](../linear-algebra/linear-systems/))，秩量的是列张成的丰富程度([秩](../linear-algebra/rank/))，张成集被证明是包含集合的最小子空间([子空间](../linear-algebra/subspaces/))。本篇把定义、几何与判定一次说清；配套的下一个问题——同一个目标，配方什么时候唯一——由 [线性无关](../linear-algebra/linear-independence/) 篇回答。

## 线性组合：加法与数乘的有限次使用

沿用本库的例向量 $\mathbf{u} = (3, 1)$、$\mathbf{v} = (1, 2)$。配上不同的系数，就得到不同的线性组合：

$$
2\mathbf{u} + \mathbf{v} = (7, 4), \qquad -\mathbf{u} + 2\mathbf{v} = (-1, 3), \qquad 0.5\mathbf{u} - 1.5\mathbf{v} = (0, -2.5), \qquad 0\mathbf{u} + 0\mathbf{v} = (0, 0)
$$

关于定义的边界，三点：

- 只有一个向量时，线性组合就是它的倍数；
- 系数取任意实数，向量个数 $k$ 有限但任意多；
- 「有限次」写在定义里：[向量空间](../linear-algebra/vector-spaces/) 的十条公理只提供有限次加法，无限和(级数)需要极限，不在线性组合的范围内。

## 张成：全部组合的集合

把系数所有可能的取法都算上，得到的集合就是张成。它能有多大，取决于组内的冗余，三种情况一次看清。

**一个向量：张成是一条直线。** $\operatorname{span}\{\mathbf{u}\} = \{t\mathbf{u} \mid t \in \mathbb{R}\}$，即平面上过原点、方向为 $\mathbf{u}$ 的那条线。

**两个不平行的向量：张成是整个平面。** 断言：任意 $\mathbf{b} = (b_1, b_2)$ 都在 $\operatorname{span}\{\mathbf{u}, \mathbf{v}\}$ 里。要配出 $\mathbf{b}$，系数 $a$、$c$ 须满足 $3a + c = b_1$ 与 $a + 2c = b_2$；按 [高斯消元](../linear-algebra/gaussian-elimination/) 消去 $a$(第二式乘 $3$ 减第一式)，得 $5c = 3b_2 - b_1$，于是：

$$
c = \frac{3b_2 - b_1}{5}, \qquad a = \frac{2b_1 - b_2}{5}
$$

对任何 $(b_1, b_2)$ 都有解，没有例外。代三个目标验证：$\mathbf{b} = (1, 1)$ 得 $(a, c) = (0.2, 0.4)$，而 $0.2\mathbf{u} + 0.4\mathbf{v} = (0.6 + 0.4,\; 0.2 + 0.8) = (1, 1)$；$\mathbf{b} = (4, 6)$ 得 $(0.4, 2.8)$；$\mathbf{b} = (7, 4)$ 得 $(2, 1)$。几何上这不意外：$\mathbf{u}$ 与 $\mathbf{v}$ 指向不同方向，沿两个方向各走一段，平面上没有到不了的点。

**两个平行的向量：张成退回一条直线。** 取 $\mathbf{w} = (6, 2) = 2\mathbf{u}$，任何组合 $a\mathbf{u} + c\mathbf{w} = (a + 2c)\mathbf{u}$ 仍落在 $\mathbf{u}$ 的那条线上，所以 $\operatorname{span}\{\mathbf{u}, \mathbf{w}\} = \operatorname{span}\{\mathbf{u}\}$。目标 $(7, 4)$ 在线外：任何组合的纵坐标恒为横坐标的三分之一，而 $4 \neq 7/3$。$\mathbf{w}$ 没能扩大张成——它自己已经在 $\mathbf{u}$ 的张成里，这正是「冗余」的直觉含义。

更高维同理：$\mathbb{R}^3$ 里 $\operatorname{span}\{\mathbf{e}_1, \mathbf{e}_2\}$ 是 $xy$ 平面，$(2, 3, 0)$ 在其中($2\mathbf{e}_1 + 3\mathbf{e}_2$)，而 $(0, 0, 1)$ 与 $(2, 3, 0.001)$ 不在。张成可以不填满所在的空间。

![左：单个向量 u 的张成是过原点的一条直线(虚线)；右：不平行的 u、v 张成整个平面，格点是整数系数的组合](/assets/linear-algebra/svg/linear-combinations-and-span.1.svg)

## 张成是子空间，而且是最小的那个

用 [子空间](../linear-algebra/subspaces/) 篇的三条判据检查 $\operatorname{span}(S)$：零向量在(系数全取零)；两个组合相加，系数对应相加，仍是组合；组合乘一个数，系数同乘，仍是组合。三条全过，所以**张成总是子空间**。

它还要更强一点。任何包含 $S$ 的子空间 $W$，既然对加法与数乘封闭，就必须包含 $S$ 的一切线性组合，即 $W \supseteq \operatorname{span}(S)$。所以 $\operatorname{span}(S)$ 是所有包含 $S$ 的子空间的交，是**包含 $S$ 的最小子空间**。这给了张成一个操作性的读法：它是把任意集合补齐成子空间的标准手续——只加不得不加的东西(全部组合)，别的什么都不加。

## 张成与方程组、秩的关系

前两篇的核心事实，用「张成」重说一遍，都变得更短。

$A\mathbf{x} = \mathbf{b}$ 有解，当且仅当 $\mathbf{b}$ 落在 $A$ 的列的张成里。这是 [线性方程组](../linear-algebra/linear-systems/) 篇的列视角，几乎只是换了个词：解就是配方，有配方就是 $\mathbf{b}$ 在列张成里。所以判定「$\mathbf{b}$ 在不在张成里」从来不需要逐点检查——它是「一个方程组有没有解」的问题，消元一次给出答案。

列张成的大小由 [秩](../linear-algebra/rank/) 度量：秩是列里线性无关的最大个数，也就是张成空间的维数(维数的正式定义在 [维数](../linear-algebra/dimension/) 篇)。上面两例：$\{\mathbf{u}, \mathbf{w}\}$ 张成一条直线，秩是 $1$；不平行的 $\{\mathbf{u}, \mathbf{v}\}$ 张成平面，秩是 $2$。

张成回答「能不能拼出」，不回答「有几种拼法」。把冗余向量拉进来，配方立刻变多：$\{\mathbf{u}, \mathbf{v}, \mathbf{w}\}$ 拼 $(7, 4)$，有 $(2, 1, 0)$、$(0, 1, 1)$、$(1, 1, 0.5)$ 等无穷多份配方。配方什么时候唯一，是 [线性无关](../linear-algebra/linear-independence/) 的定义与判定。

## 仿射组合与凸组合

给系数加约束，得到两种常用变体。

**仿射组合：系数和为 $1$。** 两个向量的仿射组合 $(1 - t)\mathbf{u} + t\mathbf{v}$ 扫过 $\mathbf{u}$、$\mathbf{v}$ 终点之间的整条直线：$t = 0.5$ 是中点 $0.5\mathbf{u} + 0.5\mathbf{v} = (2, 1.5)$，$t = 0.25$ 是 $(2.5, 1.25)$。这条直线一般不过原点——对仿射组合封闭的集合叫仿射集，比子空间少一个「含原点」，那是 [仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/) 的主题。

**凸组合：系数和为 $1$ 且全部非负。** 几何上，两个向量的凸组合扫出两点间的线段，三个向量扫出三角形，一般情形扫出凸包。插值是凸组合的标准用法：$t \in [0, 1]$ 时 $(1 - t)\mathbf{u} + t\mathbf{v}$ 正好走遍线段。

## 神经网络里的张成与凸组合

全连接层 $\mathbf{y} = W\mathbf{x}$ 的输出永远落在 $W$ 的列张成(列空间)里，这是 [秩](../linear-algebra/rank/) 篇「$784 \times 10$ 瓶颈」的张成说法：输入再丰富，输出也被限制在列张成的那个子空间里。

注意力的输出是凸组合。注意力权重经 softmax 归一化，和为 $1$ 且非负；输出是各 value 向量以权重为系数的凸组合。看一组具体数字：value 取 $(1, 0)$、$(0, 1)$、$(1, 1)$，权重 $(0.7, 0.2, 0.1)$，则：

$$
\text{输出} = 0.7 \times (1, 0) + 0.2 \times (0, 1) + 0.1 \times (1, 1) = (0.8, 0.3)
$$

即三个 value 构成的三角形内部的一点。「读哪几个词、各读多少」，在数学上就是选一个凸组合。注意力分数矩阵见 [注意力矩阵](../attention/attention-matrix/)。

权重插值与模型汤也是组合。同一架构的两组权重做插值 $\theta_t = (1 - t)\theta_0 + t\theta_1$，是参数空间里的仿射组合；多组微调权重取平均，是凸组合。组合的合法性由 [子空间](../linear-algebra/subspaces/) 篇保证：参数空间就是普通 $\mathbb{R}^n$，对加法与数乘封闭。

## 失效模式与常见误区

**把张成当成那几个向量本身。** $\{\mathbf{u}, \mathbf{v}\}$ 是两个向量，$\operatorname{span}\{\mathbf{u}, \mathbf{v}\}$ 是整个平面，无限多个点。张成指全部组合的集合，不是原集合的别名。

**以为两个向量一定张成平面。** 平行就退回直线；$\mathbb{R}^3$ 里三个向量也可能只张成平面。张成多大，取决于组内有没有冗余，正式度量是秩。

**逐点检查「在不在张成里」。** 张成是无穷集合，逐点试不完；判成员资格归结为解一个方程组(目标向量 = 各列的组合)，消元一次了结。

**把无限和当线性组合。** 定义只允许有限次加法；$e^x = 1 + x + x^2/2 + \cdots$ 这类无限和不是线性组合，它属于有极限概念的结构，不是十条公理的世界。

## 相关词条

- [向量](../linear-algebra/vectors/)：加法与数乘的定义，线性组合的全部原料
- [矩阵乘法](../linear-algebra/matrix-multiplication/)：$A\mathbf{x}$ 就是列的线性组合，本篇概念的主要用法
- [线性方程组](../linear-algebra/linear-systems/)：解 = 拼出 $\mathbf{b}$ 的配方，可解性 = $\mathbf{b}$ 在列张成里
- [线性无关](../linear-algebra/linear-independence/)：下一篇，配方什么时候唯一
- [基](../linear-algebra/basis/)：既张成又无关的向量组
- [秩](../linear-algebra/rank/)：列张成的维数
- [子空间](../linear-algebra/subspaces/)：张成是最小子空间的证明与三条判据
- [仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/)：仿射组合的展开
- [注意力矩阵](../attention/attention-matrix/)：凸组合在 Transformer 里的形态
