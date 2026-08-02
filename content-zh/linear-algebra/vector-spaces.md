---
title: "向量空间:加法、数乘与十条公理"
tags: ["why-models-learn"]
---

向量空间(vector space)是一个集合配上两种运算——加法与数乘——并满足十条公理的结构。[向量](../linear-algebra/vectors/) 篇把向量当作具体对象(数字列表、平面箭头)，本篇问一个反方向的问题：什么样的东西也可以叫向量？答案不看元素长什么样，只看运算的行为：满足这十条公理，数字列表是向量，矩阵是向量，函数是向量，本篇还会验算一个更不像向量的例子。这个抽象是后续内容的地基：子空间、张成、线性无关、基、维数都立在它上面；机器学习把 embedding、权重张量、概率分布当向量处理，同样是因为它们各自满足这十条公理。

## 三个熟悉的例子

**$\mathbb{R}^n$。** $n$ 元实数组，加法逐分量，数乘逐分量，零向量是原点。十条公理就是从它身上抽象出来的，逐条检查都显然成立。

**$\mathbb{R}^{m \times n}$。** $m \times n$ 矩阵全体，配 [矩阵](../linear-algebra/matrices/) 篇的逐元素加法与数乘，同样十条全过：两个 $2 \times 2$ 矩阵相加还是 $2 \times 2$，乘 $3$ 还是 $2 \times 2$，零矩阵充当零向量。「向量」也可以长得像一张表格。

**次数不超过 2 的多项式。** $p(x) = a + bx + cx^2$，加法是系数相加，数乘是系数放大，三个系数 $(a, b, c)$ 与一个三维向量一一对应。把次数限制拿掉、把多项式换成闭区间上的连续函数，结论仍然成立：函数也可以当向量。

## 公理：从群到向量空间

十条公理分两层，记 $V$ 为集合，$\lambda, \psi \in \mathbb{R}$ 为标量，$\mathbf{x}, \mathbf{y}, \mathbf{z} \in V$ 为元素。

前五条只管加法：

1. 封闭：$\mathbf{x} + \mathbf{y}$ 仍在 $V$ 里；
2. 结合：$(\mathbf{x} + \mathbf{y}) + \mathbf{z} = \mathbf{x} + (\mathbf{y} + \mathbf{z})$；
3. 零元：存在一个 $\mathbf{0}$，使 $\mathbf{x} + \mathbf{0} = \mathbf{x}$；
4. 逆元：每个 $\mathbf{x}$ 都有 $-\mathbf{x}$，使 $\mathbf{x} + (-\mathbf{x}) = \mathbf{0}$；
5. 交换：$\mathbf{x} + \mathbf{y} = \mathbf{y} + \mathbf{x}$。

前四条叫群(group)，加上第五条交换律叫 Abel 群。

后五条管数乘，以及数乘与加法的配合：

6. 封闭：$\lambda \mathbf{x}$ 仍在 $V$ 里；
7. 分配律一(数乘对向量加法)：$\lambda(\mathbf{x} + \mathbf{y}) = \lambda\mathbf{x} + \lambda\mathbf{y}$；
8. 分配律二(数乘对标量加法)：$(\lambda + \psi)\mathbf{x} = \lambda\mathbf{x} + \psi\mathbf{x}$；
9. 数乘结合：$\lambda(\psi\mathbf{x}) = (\lambda\psi)\mathbf{x}$；
10. 单位：$1\mathbf{x} = \mathbf{x}$。

这个清单值得逐条读一遍：后面所有定理([秩](../linear-algebra/rank/) 的性质、秩零定理、维数公式)都只用这十条证出来，证一遍，对一切向量空间同时成立。

## 一个不像向量空间的向量空间

取 $V$ 为正实数集 $\mathbb{R}^+$，定义两种运算：

$$
\mathbf{x} \oplus \mathbf{y} \;:=\; \mathbf{x}\mathbf{y} \qquad (\text{「加法」就是普通乘法})
$$

$$
\lambda \odot \mathbf{x} \;:=\; \mathbf{x}^{\lambda} \qquad (\text{「数乘」就是乘方})
$$

它真的是向量空间，十条全过。「零向量」是 $1$(因为 $\mathbf{x} \oplus 1 = \mathbf{x}$)；「逆元」是倒数($\mathbf{x} \oplus \frac{1}{\mathbf{x}} = 1$)。用 $\mathbf{x} = 3$、$\mathbf{y} = 5$、$\lambda = 2$、$\psi = 3$ 把几条运算律逐条验算：

$$
2 \odot (3 \oplus 5) \;=\; 15^2 \;=\; 225 \;=\; 9 \cdot 25 \;=\; (2 \odot 3) \oplus (2 \odot 5),
$$

$$
(2 + 3) \odot 3 \;=\; 3^5 \;=\; 243 \;=\; 9 \cdot 27 \;=\; (2 \odot 3) \oplus (3 \odot 3),
$$

$$
(2 \cdot 3) \odot 3 \;=\; 3^6 \;=\; 729 \;=\; (3^3)^2 \;=\; 2 \odot (3 \odot 3), \qquad 1 \odot 3 = 3.
$$

同一个集合，运算一换，答案就变。十条公理约束的是运算的行为，不是元素的样子，所以在这个结构里，说「$3$ 是一个向量」没有任何问题。

## 机器学习为什么需要这个抽象

神经网络的参数全体构成向量空间：随机梯度下降的每一步是「参数 + 更新量」，多模型权重的平均与插值也是加法与数乘的组合。这些操作合法，是因为参数空间对加法和数乘封闭。

词向量之间谈相似、谈算术，前提是 embedding 落在一个向量空间里；「相似」具体怎么量，由 [内积](../linear-algebra/inner-products/) 篇处理。

一个网络 $f_\theta$ 也可以看成函数空间中的一个向量，训练就是在这个空间里从一点走到另一点；核方法与高斯过程直接在函数向量空间里工作。

## 失效模式与常见误区

**「向量 = 数字列表」。** 只认数组，会在核方法、函数空间、矩阵空间里迷路：$\mathbb{R}^+$ 例子里「向量」是正实数、「数乘」是乘方，函数空间里「向量」是函数。

**忘记查封闭性。** 第一象限 $V = \{(x_1, x_2) \mid x_1 \ge 0,\; x_2 \ge 0\}$：加法封闭——$(2, 1) + (1, 2) = (3, 3)$ 仍在象限内；但数乘不封闭——$-1 \cdot (2, 1) = (-2, -1)$ 逃出去了。整数集 $\mathbb{Z}$ 也一样：$0.5 \times 3 = 1.5 \notin \mathbb{Z}$。配通常运算时，封闭性几乎总是出事的地方。

![左：平面内 u、v 相加和缩放，结果全在面内(封闭)；右：第一象限里 u 在内，−u 逃逸出界(数乘不封闭)](/assets/linear-algebra/svg/vector-spaces.1.svg)

**不过原点的直线不是向量空间。** 直线 $x_1 + x_2 = 2$ 上取 $\mathbf{p} = (2, 0)$、$\mathbf{q} = (0, 2)$，二者都在线上，但 $\mathbf{p} + \mathbf{q} = (2, 2)$ 跑出直线，零向量 $(0, 0)$ 也不在线上。它不是坏掉了，而是有另一个名字：仿射集，即「平移过的向量子空间」，见 [仿射空间与映射](../linear-algebra/affine-spaces-and-maps/)。由此得到一个常用推论：向量子空间必须过原点，[子空间](../linear-algebra/subspaces/) 篇会直接用到。

**「几乎满足」不算数。** 次数 $\le 2$ 的多项式对加法与数乘封闭，是向量空间；但「乘 $x$」立刻把次数顶出去。向量空间只管加法与数乘两种运算，别的运算(乘法、复合)需要更强的结构，不能顺手假定。

**先查零向量。** 由公理能推出 $0\mathbf{x} = \mathbf{0}$ 恒成立，所以任何集合想当向量空间，先问自己含不含零元：不含，直接出局，其余九条不用查了。

## 相关词条

- [向量](../linear-algebra/vectors/)：具体对象篇，本篇把它公理化；
- [矩阵](../linear-algebra/matrices/)：全体 $m \times n$ 矩阵本身构成向量空间；
- [范数](../linear-algebra/norms/)：给向量空间加「长度」，[内积](../linear-algebra/inner-products/) 篇再加「夹角」；
- [秩](../linear-algebra/rank/)：列张成空间的维数，「空间」「维数」的定义源自本篇；
- [子空间](../linear-algebra/subspaces/)、[线性组合与张成](../linear-algebra/linear-combinations-and-span/)、[基](../linear-algebra/basis/)、[维数](../linear-algebra/dimension/)、[仿射空间与映射](../linear-algebra/affine-spaces-and-maps/)：依次在本篇的定义上展开。
