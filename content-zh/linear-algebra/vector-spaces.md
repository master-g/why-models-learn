---
title: "向量空间:不看长相,看行为"
tags: ["why-models-learn"]
---

向量空间(vector space)是一个集合配上两种运算——「加法」与「数乘」——并满足十条公理的结构。定义一句话就说完了，真正要紧的是它的读法:[向量](../linear-algebra/vectors/) 篇把向量当作具体对象(数字列表、平面箭头)，本篇问一个反方向的问题：什么样的东西**配**叫向量？答案是：不看长相，看行为。过了十条公理这道门槛，数字列表是向量，矩阵是向量，函数是向量，甚至连「正实数」也能是向量。这个抽象是后续一切的地基：子空间、张成、线性无关、基、维数，全部建立在「向量空间」三个字上；而机器学习把 embedding、权重张量、概率分布一律当向量对待，合法性正来自这里。

## 先验收三个老朋友

**$\mathbb{R}^n$，原型。** $n$ 元实数组，加法逐分量，数乘逐分量，零向量是原点。十条公理逐条显然——它就是公理的模特，公理从它身上抽象出来。

**$\mathbb{R}^{m \times n}$，矩阵空间。** $m \times n$ 矩阵全体，配 [矩阵](../linear-algebra/matrices/) 篇的逐元素加法与数乘，同样十条全过：两个 $2 \times 2$ 矩阵相加还是 $2 \times 2$，乘 $3$ 还是 $2 \times 2$，零矩阵当零向量。所以「向量」可以长得像一张表格。

**次数不超过 2 的多项式。** $p(x) = a + bx + cx^2$，加法就是系数相加，数乘就是系数放大。三个系数 $(a, b, c)$ 与一个三维向量一一对应。更猛的还在后面：闭区间上的连续函数全体也是向量空间——函数即向量，微积分与线性代数在这里握手。

## 公理：从群到向量空间

十条公理分两层，记 $V$ 为集合，$\lambda, \psi \in \mathbb{R}$ 为标量，$\mathbf{x}, \mathbf{y}, \mathbf{z} \in V$ 为元素。

**第一层：加法构成 Abel 群(5 条)。**

1. 封闭：$\mathbf{x} + \mathbf{y}$ 仍在 $V$ 里；
2. 结合：$(\mathbf{x} + \mathbf{y}) + \mathbf{z} = \mathbf{x} + (\mathbf{y} + \mathbf{z})$；
3. 零元：存在一个 $\mathbf{0}$，使 $\mathbf{x} + \mathbf{0} = \mathbf{x}$；
4. 逆元：每个 $\mathbf{x}$ 都有 $-\mathbf{x}$，使 $\mathbf{x} + (-\mathbf{x}) = \mathbf{0}$；
5. 交换：$\mathbf{x} + \mathbf{y} = \mathbf{y} + \mathbf{x}$。

前四条叫群(group)，加上第五条交换律叫 Abel 群。

**第二层：数乘与加法相处融洽(5 条)。**

6. 封闭：$\lambda \mathbf{x}$ 仍在 $V$ 里；
7. 分配律一(数乘对向量加法)：$\lambda(\mathbf{x} + \mathbf{y}) = \lambda\mathbf{x} + \lambda\mathbf{y}$；
8. 分配律二(数乘对标量加法)：$(\lambda + \psi)\mathbf{x} = \lambda\mathbf{x} + \psi\mathbf{x}$；
9. 数乘结合：$\lambda(\psi\mathbf{x}) = (\lambda\psi)\mathbf{x}$；
10. 单位：$1\mathbf{x} = \mathbf{x}$。

这个清单值得逐条过目，因为后面所有定理——[秩](../linear-algebra/rank/) 的性质、秩零定理、维数公式——都只用这十条证出来。证一遍，对一切向量空间同时成立：这是抽象的全部意义。

## 一个不像向量空间的向量空间

取 $V$ 为正实数集 $\mathbb{R}^+$，定义两种怪运算：

$$
\mathbf{x} \oplus \mathbf{y} \;:=\; \mathbf{x}\mathbf{y} \qquad (\text{「加法」就是普通乘法})
$$

$$
\lambda \odot \mathbf{x} \;:=\; \mathbf{x}^{\lambda} \qquad (\text{「数乘」就是乘方})
$$

它真的是向量空间，十条全过。「零向量」是 $1$(因为 $\mathbf{x} \oplus 1 = \mathbf{x}$)；「逆元」是倒数($\mathbf{x} \oplus \frac{1}{\mathbf{x}} = 1$)。用 $\mathbf{x} = 3$、$\mathbf{y} = 5$、$\lambda = 2$、$\psi = 3$ 把关键的运算律各点一遍名：

$$
2 \odot (3 \oplus 5) \;=\; 15^2 \;=\; 225 \;=\; 9 \cdot 25 \;=\; (2 \odot 3) \oplus (2 \odot 5),
$$

$$
(2 + 3) \odot 3 \;=\; 3^5 \;=\; 243 \;=\; 9 \cdot 27 \;=\; (2 \odot 3) \oplus (3 \odot 3),
$$

$$
(2 \cdot 3) \odot 3 \;=\; 3^6 \;=\; 729 \;=\; (3^3)^2 \;=\; 2 \odot (3 \odot 3), \qquad 1 \odot 3 = 3.
$$

这个例子是本篇的眼：同一个集合，运算一换，身份就变。公理约束的是**运算的行为**，不是元素的长相——「$3$ 是个向量」这句话，在这个结构里完全合法。

## 机器学习为什么需要这个抽象

**权重是向量。** 神经网络的参数全体构成向量空间：随机梯度下降每一步是「参数 + 更新量」，多模型权重平均与插值，合法正因参数空间对加法和数乘封闭。

**embedding 是向量。** 词向量之间谈相似、谈算术，前提是 embedding 落在一个向量空间里([内积](../linear-algebra/inner-products/) 篇接管「相似」怎么量)。

**模型是函数空间里的一个点。** 一个网络 $f_\theta$ 是函数空间中的一个向量，训练就是在这个空间里从一点走到另一点；核方法与高斯过程干脆直接在函数向量空间里工作。

## 失效模式与常见误区

**「向量 = 数字列表」的心智模型(本篇要拆除的第一堵墙)。** 上面的 $\mathbb{R}^+$ 例子里，「向量」是正实数，「数乘」是乘方；函数空间里「向量」是函数。只认数组，会在核方法、函数空间、矩阵空间里集体迷路。

**忘记查封闭性(头号挂科点)。** 第一象限 $V = \{(x_1, x_2) \mid x_1 \ge 0,\; x_2 \ge 0\}$：加法封闭——$(2, 1) + (1, 2) = (3, 3)$ 仍在象限内；但数乘不封闭——$-1 \cdot (2, 1) = (-2, -1)$ 逃出去了。它不是向量空间。整数集 $\mathbb{Z}$ 同一死法：$0.5 \times 3 = 1.5 \notin \mathbb{Z}$。配通常运算时，封闭性几乎总是出事的地方。

![左：平面内 u、v 相加和缩放，结果全在面内(封闭)；右：第一象限里 u 在内，−u 逃逸出界(数乘不封闭)](/assets/linear-algebra/svg/vector-spaces.1.svg)

**不过原点的直线不是向量空间。** 直线 $x_1 + x_2 = 2$ 上取 $\mathbf{p} = (2, 0)$、$\mathbf{q} = (0, 2)$，二者都在线上，但 $\mathbf{p} + \mathbf{q} = (2, 2)$ 跑出直线；更要命的是零向量 $(0, 0)$ 根本不在线上。它叫仿射集，是「平移过的向量子空间」，正式处理见 [仿射空间与映射](../linear-algebra/affine-spaces-and-maps/)。由此记住一个推论：向量子空间必须过原点——这条将在 [子空间](../linear-algebra/subspaces/) 篇直接用。

**「几乎满足」不算数。** 次数 $\le 2$ 的多项式对加法与数乘封闭，是向量空间；但「乘 $x$」立刻把次数顶出去。向量空间只管加法与数乘两种运算，别的运算(乘法、复合)需要更强的结构，别顺手假定。

**零向量躲不掉。** 由公理能推出 $0\mathbf{x} = \mathbf{0}$ 恒成立，所以任何集合想当向量空间，先问自己含不含零元——不含，直接出局，其余九条不用查了。

## 相关词条

- [向量](../linear-algebra/vectors/)：具体对象篇，本篇把它公理化——「向量」的定义从此从长相换成行为；
- [矩阵](../linear-algebra/matrices/)：全体 $m \times n$ 矩阵本身构成向量空间，矩阵只是其中一员；
- [范数](../linear-algebra/norms/)：给向量空间加装「长度」；[内积](../linear-algebra/inner-products/) 篇再装「夹角」；
- [秩](../linear-algebra/rank/)：矩阵的列张成的空间的维数，「空间」「维数」的正式定义都源自本篇的公理；
- [子空间](../linear-algebra/subspaces/)、[线性组合与张成](../linear-algebra/linear-combinations-and-span/)、[基](../linear-algebra/basis/)、[维数](../linear-algebra/dimension/)、[仿射空间与映射](../linear-algebra/affine-spaces-and-maps/)：本篇章主线，依次在本篇地基上展开。
