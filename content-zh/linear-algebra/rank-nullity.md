---
title: "秩零定理:核与像的维数之和"
tags: ["why-models-learn"]
---

**秩零定理**(rank-nullity theorem)：设 $T: V \to W$ 是线性映射，$V$ 有限维，则

$$
\dim \ker T + \dim \operatorname{im} T = \dim V
$$

[核与像](../linear-algebra/kernel-and-image/) 篇陈述了这个账本并用三个例子验过(主实例 $1 + 2 = 3$、求导 $1 + 2 = 3$、投影 $1 + 1 = 2$)，本篇给出证明。证明本身是一次构造：把定义域的基劈成两半--一半在核里(丢失的维数)，一半的像恰好构成像的基(显形的维数)。矩阵形态下，等式读作「主元列数 $+$ 自由变量数 $=$ 总列数」，消元即可读出；抽象的看点在于它不用消元、对任何有限维空间成立。

## 证明：把基劈成两半

设 $\dim V = n$。分三步，每步用到一件已有的工具。

**第一步：取核的基。** $\ker T$ 是 $V$ 的子空间([核与像](../linear-algebra/kernel-and-image/) 篇)，有限维，取一组基 $\mathbf{k}_1, \dots, \mathbf{k}_r$。这 $r$ 个向量无关，记 $r = \dim \ker T$。

**第二步：扩成 $V$ 的基。** 无关集可以不断扩充：往 $\{\mathbf{k}_1, \dots, \mathbf{k}_r\}$ 里逐个添加保持无关的向量，加不动为止；[基](../linear-algebra/basis/) 篇的「极大无关集即基」保证停下来的时刻手里就是 $V$ 的一组基：

$$
\mathbf{k}_1, \dots, \mathbf{k}_r,\; \mathbf{b}_1, \dots, \mathbf{b}_s
$$

其中 $r + s = n$(基的元素个数是维数)。

**第三步：$T(\mathbf{b}_1), \dots, T(\mathbf{b}_s)$ 是像的基。** 两条性质分别一行：

- **张成像**：任取 $T(\mathbf{v})$，把 $\mathbf{v} = \sum \alpha_i \mathbf{k}_i + \sum \beta_j \mathbf{b}_j$ 代入，$T(\mathbf{k}_i) = \mathbf{0}$ 把第一项全消掉，剩 $T(\mathbf{v}) = \sum \beta_j T(\mathbf{b}_j)$——任何像都是这 $s$ 个向量的组合；
- **无关**：设 $\sum \beta_j T(\mathbf{b}_j) = \mathbf{0}$，即 $T(\sum \beta_j \mathbf{b}_j) = \mathbf{0}$，所以 $\sum \beta_j \mathbf{b}_j \in \ker T$，可写成核基的组合 $\sum \alpha_i \mathbf{k}_i$；移项得 $\sum \beta_j \mathbf{b}_j - \sum \alpha_i \mathbf{k}_i = \mathbf{0}$——但这是 $V$ 的一组基的线性组合，系数只能全为零，$\beta$ 全零。

于是 $\dim \operatorname{im} T = s$。$r + s = n$ 即所证。$\square$

证明的构造在主实例上走一遍。$A = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}$ 的核基取 $\mathbf{k}_1 = (-1, -1, 1)$；扩基添 $\mathbf{e}_1$、$\mathbf{e}_2$——无关(第三分量先强制系数为零)，张成可由公式验证：$(x, y, z) = z\mathbf{k}_1 + (x + z)\mathbf{e}_1 + (y + z)\mathbf{e}_2$，代 $(5, 7, -2)$ 等三组数字均成立。像侧：$A\mathbf{e}_1 = (1, 0, 1)$、$A\mathbf{e}_2 = (0, 1, 1)$，正是 [核与像](../linear-algebra/kernel-and-image/) 篇算出的像的基；$1 + 2 = 3$，与定理一致。

求导 $D: P_2 \to P_1$ 同样服帖：核基 $\{1\}$，扩成 $\{1, x, x^2\}$，像侧 $D(x) = 1$、$D(x^2) = 2x$ 张成 $P_1$；$1 + 2 = 3$。

![维数账本：左，V 的基劈成两半——核基 k₁(dim 1，灰)与扩充 b₁，b₂(dim 2)；右上，核基被 T 压没；右下，扩充的像 T(b₁)、T(b₂) 恰好构成像的基(珊瑚，dim 2)](/assets/linear-algebra/svg/rank-nullity.1.svg)

## 消元视角：列的点名

矩阵 $A$($m \times n$)消元后，每一列是主元列或自由列，二者必居其一。主元列数 $=$ [秩](../linear-algebra/rank/) $=$ 像的维数；自由列数 $=$ 齐次解的自由变量个数 $=$ 核的维数([高斯消元](../linear-algebra/gaussian-elimination/) 篇)。于是矩阵形态的秩零定理几乎不需要证：

$$
\text{秩} + \text{核维数} = n \;(=\text{列数})
$$

消元给的是计算，基劈半给的是理解：前者告诉你怎么数，后者告诉你为什么非这么数不可——每一维要么在核里被压没，要么在像里显形，没有第三种去处。

## 方阵推论：单射与满射在此会合

秩零定理最锋利的推论出在方阵($V \to V$，或等维空间之间)：此时 $\dim \ker T = 0$ 与 $\dim \operatorname{im} T = n$ 同真同假，所以

$$
\text{单射} \iff \text{满射} \iff \text{双射}
$$

三者要么全成立，要么全不成立。验证两例：$B = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$ 秩 $2$，核为零(单射)、像为全平面(满射)、可逆；$W = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}$ 秩 $1$，核是直线(非单射)、像也是直线(非满射)、不可逆。注意这个等价是方阵特权：长方形映射可以只占一头，嵌入 $\mathbb{R}^2 \to \mathbb{R}^3$ 单射而不满射——秩零定理不管这个，因为定义域与值域维数不同。可逆性的正式处理在 [矩阵的逆](../linear-algebra/matrix-inverse/) 篇。

## 神经网络里的账本

秩零定理把层的账算到了最后一维。$784 \times 10$ 的全连接层：输入 $784$ 维，像的维数(秩)至多为 $10$，于是至少 $774$ 个输入方向躺在核里——无论训练得多好，这些方向上的差异在这一层之后不可恢复。压缩不是训练失败，是秩的预算决定了多少维能显形。

过参数化网络的核则给出另一种读法：极宽层的权重空间里，许多不同的参数配置给出同一个函数(彼此之差落在「函数不变」的方向上)。参数冗余与泛化的关系是开放研究问题，但冗余的住所是核——这句是确定的。

## 失效模式与常见误区

**以为定理说「核 $+$ 像 $= V$」。** 核住在定义域 $V$，像住在值域 $W$，两个子空间分属两地，谈不上相加。定理说的是**维数**之和：两个数，一个等式。

**忘记有限维假设。** 证明的第一步是取核的基，依赖 $V$ 有限维。无限维空间有对应理论，但等式形状与证明都不能照搬。

**以为单射与满射永远等价。** 等价只在「定义域与值域同维」时成立(方阵或等维)。长方形映射可以单而不满(嵌入)、满而不单(求导)，秩零定理本身对任意形状都成立，等价性不成立。

**以为核的基有更便宜的找法。** 没有：实务上就是解齐次方程组 $A\mathbf{x} = \mathbf{0}$，消元，读自由变量。定理保证的是答案的维数对，不保证不算。

## 相关词条

- [核与像](../linear-algebra/kernel-and-image/)：定理的陈述、例子与两个角色的分工
- [秩](../linear-algebra/rank/)：像的维数；「核维数 $= n - $ 秩」的另一半
- [基](../linear-algebra/basis/)：扩基的合法性(极大无关集即基)
- [维数](../linear-algebra/dimension/)：等式里的三个数
- [高斯消元](../linear-algebra/gaussian-elimination/)：主元列与自由列的点名
- [矩阵的逆](../linear-algebra/matrix-inverse/)：方阵推论引向可逆性，下一篇
- [线性方程组](../linear-algebra/linear-systems/)：通解 $=$ 特解 $+$ 核，核的维数决定解集大小
