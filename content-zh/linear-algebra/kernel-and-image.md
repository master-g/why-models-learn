---
title: "核与像:非唯一性与可达性"
tags: ["why-models-learn"]
---

线性映射 $T: V \to W$ 自带两个子空间。**核**(kernel)是被映射到零的全体：

$$
\ker T = \{\mathbf{v} \in V \mid T(\mathbf{v}) = \mathbf{0}\}
$$

**像**(image)是值域里被够着的全体：

$$
\operatorname{im} T = \{T(\mathbf{v}) \mid \mathbf{v} \in V\}
$$

注意住址：核住在**定义域** $V$ 里，像住在**值域** $W$ 里。[子空间](../linear-algebra/subspaces/) 篇见过的两个最重要的子空间——零空间与列空间——现在正名：矩阵 $A$ 的零空间就是映射 $\mathbf{x} \mapsto A\mathbf{x}$ 的核，列空间就是它的像。核度量**非唯一性**(核越大，同一个输出对应越多输入)，像度量**可达性**(像越大，越多目标有解)；两者的维数之和恒等于定义域维数，那是 [秩零定理](../linear-algebra/rank-nullity/) 的内容。

## 定义与子空间验证

先用 [子空间](../linear-algebra/subspaces/) 篇的三条判据把两个集合的「子空间」身份坐实。

核：零向量在($T(\mathbf{0}) = \mathbf{0}$)；$\mathbf{u}, \mathbf{v}$ 在核里，则 $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}) = \mathbf{0}$，加封闭；$T(\lambda\mathbf{u}) = \lambda T(\mathbf{u}) = \mathbf{0}$，数乘封闭。三条全过——线性的两条性质正好就是判据的两条。

像：零向量在($T(\mathbf{0})$)；$T(\mathbf{u}) + T(\mathbf{v}) = T(\mathbf{u} + \mathbf{v})$ 仍是某个输入的像；$\lambda T(\mathbf{u}) = T(\lambda\mathbf{u})$ 同理。也全过。

矩阵语言下，求核就是解齐次方程组 $A\mathbf{x} = \mathbf{0}$([高斯消元](../linear-algebra/gaussian-elimination/))，求像就是取列的[张成](../linear-algebra/linear-combinations-and-span/)。

## 核管非唯一性

不同输入给出相同输出，当且仅当它们的差在核里：

$$
T(\mathbf{u}) = T(\mathbf{v}) \iff T(\mathbf{u} - \mathbf{v}) = \mathbf{0} \iff \mathbf{u} - \mathbf{v} \in \ker T
$$

一行等价，信息量很大：输出重复的模式，完全由核决定。核只有零向量时，重复不可能发生——$T$ 是**单射**(injective)，不同输入必有不同输出。

用 [秩](../linear-algebra/rank/) 篇的主实例 $A = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}$ 演示。解 $A\mathbf{x} = \mathbf{0}$：$x_1 + x_3 = 0$、$x_2 + x_3 = 0$，得 $\ker A = \operatorname{span}\{(-1, -1, 1)\}$，一条直线，维数 $1$。于是非唯一性随处可见：$\mathbf{u} = (1, 1, 1)$ 与 $\mathbf{v} = (2, 2, 0)$ 的像都是 $(2, 2, 4)$——它们的差 $(-1, -1, 1)$ 正在核里。反过来，嵌入 $\mathbf{x} \mapsto (\mathbf{x}, 0)$($\mathbb{R}^2 \to \mathbb{R}^3$)的核只有零向量，它是单射。

## 像管可达性

方程 $T(\mathbf{x}) = \mathbf{b}$ 有解，当且仅当 $\mathbf{b}$ 在像里——这几乎是像的定义复述，但它把 [线性方程组](../linear-algebra/linear-systems/) 篇的可解性判据($\mathbf{b}$ 在列张成里)提升到了任何线性映射。像占满整个值域($\operatorname{im} T = W$)时，$T$ 是**满射**(surjective)，每个目标都有解。

继续用主实例。第三列 $=$ 第一列 $+$ 第二列，所以像由前两列张成：$\operatorname{im} A = \operatorname{span}\{(1, 0, 1), (0, 1, 1)\}$，$\mathbb{R}^3$ 里的一个平面，维数 $2$。$\mathbf{b} = (1, 1, 2)$ 在像里(它就是第三列)；$\mathbf{b} = (1, 1, 0)$ 不在——消元给出 $2 = 0$ 的矛盾。无解不是方程出了错，是目标落在了像外。

一个满射的例子：求导 $D: P_2 \to P_1$。任何 $b + cx$ 都是 $bx + \frac{c}{2}x^2$ 的导数，值域 $P_1$ 全部可达；而核是常数多项式($D(3) = 0$)，维数 $1$。满射但远非单射——不同的原函数相差一个常数，差恰好在核里，微积分里「$+C$」的线性代数解释。

## 维数账本

两个子空间的维数，加起来总是定义域的维数：

$$
\dim \ker T + \dim \operatorname{im} T = \dim V
$$

三个已验证的账：主实例 $A$，$1 + 2 = 3$；求导 $D: P_2 \to P_1$，$1 + 2 = 3$；投影到 $x$ 轴 $P(x, y) = (x, 0)$，核是 $y$ 轴、像是 $x$ 轴，$1 + 1 = 2$。直觉一句话：定义域的每一维，要么被压进核里(丢失)，要么在像里显形(保留)，没有第三种去处。等式的证明在 [秩零定理](../linear-algebra/rank-nullity/) 篇；像的维数就是 [秩](../linear-algebra/rank/)，所以等式的矩阵形态是 $\text{核维数} = n - \text{秩}$。

![左：投影 P(x，y)=(x，0) 的核是 y 轴(珊瑚)，两点 (2,1) 与 (2,3) 差在核里，映到同一点 (2,0)；右：像是 x 轴(珊瑚)，点 b=(1,2) 在像外，Px=b 无解](/assets/linear-algebra/svg/kernel-and-image.1.svg)

## 神经网络里的核与像

秩亏层的核方向是信息丢失的方向。[秩](../linear-algebra/rank/) 篇的 $W = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}$ 以 $(1, -1)$ 为核：两个输入若只差这个方向的倍数，经过这层之后再也无法区分——无论后面多少层，这个差异都追不回来。压缩的发生地是核，压缩不掉的差异才进像。

像的另一面是输出范围：[秩](../linear-algebra/rank/) 篇的 $784 \times 10$ 瓶颈，用本篇语言是「全连接层的输出活在像(列空间)里，像的维数是秩」。层输出够不着的位置，不是训练不到位，是像之外。

## 失效模式与常见误区

**核与像的住址搞反。** 核在定义域，像在值域：矩阵 $A$($m \times n$)的核是 $\mathbb{R}^n$ 的子空间，像是 $\mathbb{R}^m$ 的子空间。说「$\mathbf{b}$ 在核里」通常是口误——$\mathbf{b}$ 是值域的点，归像管。

**把无解当方程错误。** $A\mathbf{x} = \mathbf{b}$ 无解，说的是 $\mathbf{b} \notin \operatorname{im} A$，一个几何事实，不是数值计算的失败。

**以为核为零就万事大吉。** 核为零只保证单射(不同输入可区分)，不保证满射：嵌入 $\mathbb{R}^2 \to \mathbb{R}^3$ 没有核，平面上外的点照样够不着。单射与满射是两件独立的事。

**混淆两个名字。** 零空间(null space)就是核，列空间(column space)就是像——同一物两个名字，文献里混用，心里要有对照表。

## 相关词条

- [线性映射](../linear-algebra/linear-maps/)：核与像的主人
- [秩零定理](../linear-algebra/rank-nullity/)：维数账本的证明，下一篇
- [秩](../linear-algebra/rank/)：像的维数；主实例的出处
- [子空间](../linear-algebra/subspaces/)：零空间与列空间的正名处
- [线性方程组](../linear-algebra/linear-systems/)：可解性 = 目标在像里；通解 = 特解 + 核
- [高斯消元](../linear-algebra/gaussian-elimination/)：求核 = 解齐次方程组
- [维数](../linear-algebra/dimension/)：账本里的三个数
