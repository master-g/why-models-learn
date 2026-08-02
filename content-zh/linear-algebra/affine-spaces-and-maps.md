---
title: "仿射空间与映射:平移过的线性"
tags: ["why-models-learn"]
---

**仿射子空间**(affine subspace)是子空间的平移：取线性子空间 $U$ 和任一向量 $\mathbf{p}$，集合

$$
M = \mathbf{p} + U = \{\mathbf{p} + \mathbf{u} \mid \mathbf{u} \in U\}
$$

就是仿射子空间。**仿射映射**(affine map)是线性映射加平移：$T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$。两个定义各只比线性版多一个「$+$」——不过原点，是仿射与线性的全部分别。神经网络的全连接层 $\mathbf{x} \mapsto W\mathbf{x} + \mathbf{b}$ 正是仿射映射，本篇是线性代数章的收尾，也是全连接层数学身份的落户处。

## 仿射子空间：平移过的子空间

$M = \mathbf{p} + U$ 里，**方向子空间** $U$ 由 $M$ 唯一决定($U = \{\mathbf{q}_1 - \mathbf{q}_2 \mid \mathbf{q}_1, \mathbf{q}_2 \in M\}$，任取两点作差)，基点 $\mathbf{p}$ 却可以是 $M$ 中任一点。演示：直线 $M = (1, 0) + \operatorname{span}\{(1, 2)\}$，点 $(1, 0)$、$(2, 2)$、$(0, -2)$ 都在 $M$ 上，任取一个当基点表示同一条线——它们的差 $(1, 2)$ 都在 $U$ 里。$M$ 的维数定义为 $U$ 的维数。

仿射子空间与方程组互为表里。[线性方程组](../linear-algebra/linear-systems/) 篇的通解结构说：$A\mathbf{x} = \mathbf{b}$ 的解集 $=$ 特解 $+$ 核——这就是「解集是仿射子空间」。反过来，任何仿射子空间都是某方程组的解集($U$ 用齐次方程刻画，$\mathbf{p}$ 提供右端)。同一条直线两种写法：$2x - y = 2$ 的解集，正是 $(1, 0) + \operatorname{span}\{(1, 2)\}$——$(1, 0)$ 满足方程，方向 $2x - y = 0$ 的核恰是 $\operatorname{span}\{(1, 2)\}$。

## 仿射组合：系数和为 1

[线性组合与张成](../linear-algebra/linear-combinations-and-span/) 篇已定义：**仿射组合**要求系数和为 $1$。这个约束的几何含义此刻可以说透：系数和为 $1$ 的组合不随原点平移而变——$\sum \lambda_i \mathbf{q}_i$($\sum \lambda_i = 1$)在换原点时结果跟着平移，指的还是同一个点；和不为 $1$ 的组合则依赖原点在哪，对仿射子空间没有内在意义。仿射子空间对仿射组合封闭：$M$ 里任取两点，整条连线都在 $M$ 里；含 $(1, 0)$ 与 $(2, 2)$ 的中点 $\left(\frac{3}{2}, 1\right)$ 也在上节的直线上。

## 仿射映射：线性加平移

$T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ 当 $\mathbf{b} \neq \mathbf{0}$ 时**不是**线性映射——[线性映射](../linear-algebra/linear-maps/) 篇的判据 $T(\mathbf{0}) = \mathbf{0}$ 直接出局。例：$A = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$，$\mathbf{b} = (1, 1)$，则 $T(0, 0) = (1, 1) \neq (0, 0)$。

仿射映射的特征性质是**保仿射组合**：$\sum \lambda_i = 1$ 时

$$
T\Big(\sum_i \lambda_i \mathbf{q}_i\Big) = \sum_i \lambda_i T(\mathbf{q}_i)
$$

验证：$\mathbf{p} = (1, 0)$、$\mathbf{q} = (2, 2)$ 的中点 $\left(\frac{3}{2}, 1\right)$ 映到 $(4, 2)$，恰是 $T(\mathbf{p}) = (3, 1)$ 与 $T(\mathbf{q}) = (5, 3)$ 的中点。中点映中点，直线映直线——仿射映射是「保持平直」的映射中最一般的一类。

## 复合公式：两层仿射仍是一层

仿射映射复合仍是仿射映射，系数有显式公式。先作用 $(C, \mathbf{d})$ 再作用 $(A, \mathbf{b})$：

$$
\mathbf{x} \xrightarrow{(C, \mathbf{d})} C\mathbf{x} + \mathbf{d} \xrightarrow{(A, \mathbf{b})} A(C\mathbf{x} + \mathbf{d}) + \mathbf{b} = (AC)\mathbf{x} + (A\mathbf{d} + \mathbf{b})
$$

复合结果是 $(AC,\; A\mathbf{d} + \mathbf{b})$。用 $A = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$、$\mathbf{b} = (1, 1)$、$C = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$、$\mathbf{d} = (0, 2)$ 在 $\mathbf{x} = (1, 1)$ 上验：逐步得 $(2, 3) \to (5, 4)$；公式给 $AC = \begin{pmatrix} 2 & 2 \\ 0 & 1 \end{pmatrix}$、$A\mathbf{d} + \mathbf{b} = (1, 3)$，同样得 $(5, 4)$。

这条公式是 [为什么需要非线性](../neurons-and-activations/why-non-linearity/) 篇的数学内核：两层全连接 $W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)$，没有激活函数，再深的网络也塌缩成一个仿射映射。深度要兑换成表达力，必须在层间打断仿射封闭性。

![左：仿射子空间 M = p + U——子空间 U(灰虚线)平移到基点 p 处(珊瑚)；右：两层仿射映射无激活时复合成一层，深度塌缩](/assets/linear-algebra/svg/affine-spaces-and-maps.1.svg)

## 神经网络里的仿射

全连接层的标准写法 $\mathbf{x} \mapsto W\mathbf{x} + \mathbf{b}$ 就是仿射映射([什么是神经元](../neurons-and-activations/what-is-a-neuron/) 篇的「先加权和再加偏置」)。偏置 $\mathbf{b}$ 的作用用本篇语言说：让映射不过原点，决策边界可以是空间中超平面的任意位置，而不必穿过原点。

分类边界本身是仿射对象：单层分类器 $W\mathbf{x} + \mathbf{b} = \mathbf{0}$ 的零等值面是一个仿射超平面——某个仿射子空间，写成方程组解集的形式正是本篇第二节的表里关系。线性可分的几何，就是两类点能否被一个仿射超平面隔开。

## 失效模式与常见误区

**以为仿射是线性的别称。** 线性映射必须过原点($T(\mathbf{0}) = \mathbf{0}$)，仿射映射可以不平移也可以平移；$\mathbf{b} = \mathbf{0}$ 时仿射退化为线性，线性是仿射的特例。

**仿射组合忘了系数和为 1。** 和不为 1 的组合随原点选取而漂移，不是几何对象。看到「凸组合」再收紧一步：和为 1 且非负。

**以为基点 $\mathbf{p}$ 唯一。** 方向子空间 $U$ 唯一，基点可以换——$M$ 中任一点都行。写 $M = \mathbf{p} + U$ 时，$\mathbf{p}$ 是代表不是特权。

**指望堆叠仿射层加深表达力。** 复合公式宣判：无数激活时，$n$ 层仿射 $=$ 一层仿射，权重 $W_n \cdots W_1$、偏置递归累加。表达力的来源是非线性打断，不是层数本身。

## 相关词条

- [线性映射](../linear-algebra/linear-maps/)：仿射映射 = 线性 + 平移；$T(\mathbf{0}) \neq \mathbf{0}$ 的出局判据
- [线性组合与张成](../linear-algebra/linear-combinations-and-span/)：仿射组合的出处
- [线性方程组](../linear-algebra/linear-systems/)：解集 = 特解 + 核 = 仿射子空间
- [核与像](../linear-algebra/kernel-and-image/)：通解里的「核」正是方向子空间
- [子空间](../linear-algebra/subspaces/)：仿射子空间平移回去的那个家
- [什么是神经元](../neurons-and-activations/what-is-a-neuron/)：全连接层 = 仿射映射 + 激活
- [为什么需要非线性](../neurons-and-activations/why-non-linearity/)：复合公式 = 塌缩的数学内核
