---
title: "内积:角度的三条公理"
tags: ["why-models-learn"]
---

**内积**(inner product)是向量空间上给每对向量配一个数的函数 $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$：满足三条公理——对称性、对第一变元线性、正定性——才算内积。[范数](../linear-algebra/norms/) 篇把长度公理化，但光有长度量不了「两个向量夹多少度」；内积把角度背后的运算(点积)公理化。本篇谈代数：哪类函数够格、内积怎么生出范数、以及连接两者的不等式——**Cauchy–Schwarz 不等式**，它是整个内积节最常用的工具；角度的几何本身(余弦、垂直)在 [角度与正交](../linear-algebra/angles-and-orthogonality/) 篇。神经元的加权和、注意力的分数、embedding 的相似度，都是内积。

## 三条公理

函数 $\langle \cdot, \cdot \rangle$ 是内积，当且仅当对所有 $\mathbf{x}, \mathbf{y}, \mathbf{z} \in V$ 与标量 $a, b$ 满足三条。三条各配一个「缺了会怎样」。

**对称性**：$\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle$。这条管「内积不分先后」。反例：$f(\mathbf{x}, \mathbf{y}) = x_1 y_1 + x_2 y_2 + x_1 y_2$ 给 $f((1, 0), (0, 1)) = 1$，交换后 $f((0, 1), (1, 0)) = 0$——同一个「夹角」两种读数，这样的函数谈不了角度。

**对第一变元线性**：$\langle a\mathbf{x} + b\mathbf{z}, \mathbf{y} \rangle = a\langle \mathbf{x}, \mathbf{y} \rangle + b\langle \mathbf{z}, \mathbf{y} \rangle$。配合对称性，对第二变元也线性，合称双线性。反例：$f(\mathbf{x}, \mathbf{y}) = (x_1 y_1 + x_2 y_2)^2$ 把缩放翻成平方：$f(2\mathbf{x}, \mathbf{y}) = 4f(\mathbf{x}, \mathbf{y}) \neq 2f(\mathbf{x}, \mathbf{y})$，齐次性出局。

**正定性**：$\langle \mathbf{x}, \mathbf{x} \rangle \geq 0$，且 $\langle \mathbf{x}, \mathbf{x} \rangle = 0$ 当且仅当 $\mathbf{x} = \mathbf{0}$。反例：$\langle \mathbf{x}, \mathbf{y} \rangle = x_1 y_1$ 只量第一坐标，$\langle (0, 5), (0, 5) \rangle = 0$ 而 $(0, 5) \neq \mathbf{0}$——与范数篇的正定性反例同源：长度和角度都不能漏看方向。

## 点积：Rⁿ 的原型

$\mathbb{R}^n$ 上的**点积**(dot product)是

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{n} x_i y_i
$$

三条公理逐条对：求和换序不变(对称)；求和对 $a\mathbf{x} + b\mathbf{z}$ 拆开(线性)；$\sum_i x_i^2$ 是平方和，非负且为零当且仅当逐项为零(正定)。数字：$\langle (3, 4), (1, 2) \rangle = 3 \times 1 + 4 \times 2 = 11$。配上内积的向量空间叫**内积空间**；$\mathbb{R}^n$ 配点积是全部内积空间的原型——[坐标](../linear-algebra/coordinates/) 篇说任何 $n$ 维空间配上坐标就是 $\mathbb{R}^n$，内积节的定理都在这个原型上演示。

## 内积诱导范数

[范数](../linear-algebra/norms/) 篇说内积是范数的另一个重要来源，现在交付：任何内积都诱导一个范数

$$
\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}
$$

三条范数公理里，正定性直接继承内积的正定性；齐次性一行：$\|c\mathbf{x}\| = \sqrt{c^2 \langle \mathbf{x}, \mathbf{x} \rangle} = |c| \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$；三角不等式需要 Cauchy–Schwarz 不等式，下节证完再补。点积诱导的正是欧几里得范数：$\sqrt{\langle (3, 4), (3, 4) \rangle} = \sqrt{25} = 5$，即范数篇的 $\|(3, 4)\|_2 = 5$。范数篇从公理出发定义 $p$ 范数，这里看到 $p = 2$ 那一把有内积出身——这个出身会换来别的范数没有的工具。

## Cauchy–Schwarz 不等式

**Cauchy–Schwarz 不等式**：对内积空间的任意 $\mathbf{x}, \mathbf{y}$，

$$
|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \, \|\mathbf{y}\|
$$

等号当且仅当 $\mathbf{x}, \mathbf{y}$ 共线(一个是另一个的倍数，含零向量)。

证明用一个二次函数。对任意实数 $t$，由正定性 $\langle \mathbf{x} + t\mathbf{y}, \mathbf{x} + t\mathbf{y} \rangle \geq 0$，按双线性展开：

$$
f(t) = \|\mathbf{y}\|^2 t^2 + 2\langle \mathbf{x}, \mathbf{y} \rangle \, t + \|\mathbf{x}\|^2 \geq 0 \quad (\forall t \in \mathbb{R})
$$

设 $\mathbf{y} \neq \mathbf{0}$，$f(t)$ 是开口向上的抛物线；恒非负当且仅当判别式不大于零：

$$
\Delta = 4\langle \mathbf{x}, \mathbf{y} \rangle^2 - 4\|\mathbf{x}\|^2 \|\mathbf{y}\|^2 \leq 0
$$

整理即得不等式。等号成立当且仅当判别式为零，即 $f(t)$ 有根——存在 $t$ 使 $\mathbf{x} + t\mathbf{y} = \mathbf{0}$，正是共线。($\mathbf{y} = \mathbf{0}$ 时不等式两边都是零，自动成立。)

用 $\mathbf{x} = (3, 4)$、$\mathbf{y} = (1, 2)$ 把证明的数字摆出来：$f(t) = 5t^2 + 22t + 25$，判别式 $\Delta = 484 - 500 = -16 < 0$，顶点在 $t^* = -2.2$，最小值 $f(t^*) = 0.8 > 0$——抛物线确实不触轴。不等式本身：$|\langle \mathbf{x}, \mathbf{y} \rangle| = 11 \leq 5 \times \sqrt{5} \approx 11.18$。换共线的 $(3, 4)$ 与 $(6, 8)$：$\langle \mathbf{x}, \mathbf{y} \rangle = 18 + 32 = 50$，$\|\mathbf{x}\| \|\mathbf{y}\| = 5 \times 10 = 50$，等号成立。

![左：Cauchy–Schwarz 证明中的抛物线 f(t)=5t²+22t+25 恒在 t 轴上方(判别式 −16<0)，顶点 t*=−2.2 处取最小值 0.8；右：两组柱图对比 |⟨x，y⟩| 与 ‖x‖·‖y‖——不共线时 11<11.18，共线时 50=50 取等号](/assets/linear-algebra/svg/inner-products.1.svg)

Cauchy–Schwarz 的两个直接交付。

**$p = 2$ 三角不等式**。范数篇把它留给了本篇，现在一行补齐：

$$
\|\mathbf{x} + \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + 2\langle \mathbf{x}, \mathbf{y} \rangle + \|\mathbf{y}\|^2 \leq \|\mathbf{x}\|^2 + 2\|\mathbf{x}\| \|\mathbf{y}\| + \|\mathbf{y}\|^2 = (\|\mathbf{x}\| + \|\mathbf{y}\|)^2
$$

开方即得。数字与范数篇同一个例子：$\|(4, 6)\|_2 = 2\sqrt{13} \approx 7.21 \leq 5 + \sqrt{5} \approx 7.24$。诱导范数的三角不等式由此齐全——$\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$ 确实是范数。

**$\|\mathbf{x}\|_1 \leq \sqrt{n} \, \|\mathbf{x}\|_2$**。范数篇等价链里没证的那一条，用 Cauchy–Schwarz 作用于 $|\mathbf{x}| = (|x_1|, \dots, |x_n|)$ 与 $\mathbf{1} = (1, \dots, 1)$：

$$
\|\mathbf{x}\|_1 = \langle |\mathbf{x}|, \mathbf{1} \rangle \leq \||\mathbf{x}|\|_2 \, \|\mathbf{1}\|_2 = \sqrt{n} \, \|\mathbf{x}\|_2
$$

数字：$(3, 4)$ 的 $\|\cdot\|_1 = 7 \leq \sqrt{2} \times 5 \approx 7.07$。

## 不止点积：别的内积

点积是原型，不是全部。三个例子，判据都是三条公理。

**加权内积**：$\mathbb{R}^2$ 上 $\langle \mathbf{x}, \mathbf{y} \rangle = 2x_1 y_1 + 3x_2 y_2$。抽查三公理：对称——$(1, 2)$ 与 $(3, 1)$ 两个方向都算得 $12$；线性——$\langle 2\mathbf{u} + \mathbf{v}, \mathbf{w} \rangle = 15 = 2\langle \mathbf{u}, \mathbf{w} \rangle + \langle \mathbf{v}, \mathbf{w} \rangle$(取 $\mathbf{u} = (1, 2)$、$\mathbf{v} = (3, 1)$、$\mathbf{w} = (0, 1)$)；正定——$\langle (1, 2), (1, 2) \rangle = 2 + 12 = 14 > 0$。权重不同，「哪个方向更费长度」就不同：这把尺子量 $(1, 0)$ 得 $\sqrt{2}$，量 $(0, 1)$ 得 $\sqrt{3}$。

**Frobenius 内积**：矩阵空间上 $\langle A, B \rangle = \operatorname{tr}(A^\top B) = \sum_{i, j} a_{ij} b_{ij}$——把矩阵按 [坐标](../linear-algebra/coordinates/) 篇的读法摊成向量，就是点积。数字：$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$、$B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$，$\operatorname{tr}(A^\top B) = 5$，逐分量乘积和 $1 \times 0 + 2 \times 1 + 3 \times 1 + 4 \times 0 = 5$，相等。Cauchy–Schwarz 照用：$5 \leq \sqrt{30} \times \sqrt{2} \approx 7.75$。它诱导的范数 $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$ 是 [矩阵范数](../linear-algebra/matrix-norms/) 篇的成员；神经网络的权重衰减 $\|W\|_F^2$ 就是这个内积配自己。

**积分内积**：多项式空间 $P_2$ 上 $\langle p, q \rangle = \int_0^1 p(x) q(x) \, dx$。三条公理逐条对：积分换序不变；积分对线性组合拆开；$\int_0^1 p^2 \, dx \geq 0$，且连续函数平方的积分为零只当 $p \equiv 0$。数字：$\langle x, x^2 \rangle = \int_0^1 x^3 \, dx = \frac{1}{4}$；$\|x\| = \sqrt{\int_0^1 x^2 \, dx} = \frac{1}{\sqrt{3}} \approx 0.577$；Cauchy–Schwarz：$\frac{1}{4} \leq \frac{1}{\sqrt{3}} \times \frac{1}{\sqrt{5}} \approx 0.258$。

## 哪些范数来自内积：平行四边形法则

内积诱导的范数必满足**平行四边形法则**：

$$
\|\mathbf{x} + \mathbf{y}\|^2 + \|\mathbf{x} - \mathbf{y}\|^2 = 2\big(\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2\big)
$$

把左端两个平方都按内积展开，$\pm 2\langle \mathbf{x}, \mathbf{y} \rangle$ 相消即得。这条法则反过来当判据用：不满足它的范数，不可能由任何内积诱导。$\mathbf{x} = (1, 0)$、$\mathbf{y} = (0, 1)$ 试两把尺子：$\|\cdot\|_2$ 得 $2 + 2 = 4 = 2 \times (1 + 1)$，成立；$\|\cdot\|_1$ 得 $4 + 4 = 8 \neq 2 \times (1 + 1) = 4$，出局——曼哈顿范数没有内积出身，$p = \infty$ 同理。Cauchy–Schwarz 只对内积范数成立，原因在这里：它的两边都是内积语言。

## 神经网络里的内积

神经元的第一步是点积：[什么是神经元](../neurons-and-activations/what-is-a-neuron/) 篇的加权和 $\mathbf{w} \cdot \mathbf{x}$ 就是内积，加权内积的角度看，权重 $\mathbf{w}$ 选的是「哪个输入方向更算数」。

注意力的分数是内积：query 与 key 的点积 $\mathbf{q} \cdot \mathbf{k}$ 量两个向量的「对齐程度」，再除以 $\sqrt{d}$ 用范数把分数压回温和区间——$\mathbf{q} = (1, 2)$、$\mathbf{k} = (3, 1)$ 时分数 $5$，缩放后 $5 / \sqrt{2} \approx 3.54$。[自注意力](../attention/self-attention/) 篇展开。

embedding 的相似度是内积：词向量训练得让语义相近的词内积大，归一化之后就是余弦相似度，[余弦相似度](../text-representation/cosine-similarity/) 与 [embedding 几何](../text-representation/embedding-geometry/) 篇展开。权重矩阵的衰减项 $\|W\|_F^2$ 是 Frobenius 内积 $\langle W, W \rangle$，上节已见。

## 失效模式与常见误区

**把内积结果当向量。** $\langle \mathbf{x}, \mathbf{y} \rangle$ 是个数。$\langle \mathbf{x}, \mathbf{y} \rangle \, \mathbf{z}$ 是数乘，而 $\mathbf{x} \, \langle \mathbf{y}, \mathbf{z} \rangle$ 通常与它不等——内积没有结合律，挪动括号前先看清哪个因子是数。

**把 Cauchy–Schwarz 往非内积范数上套。** 不等式右边 $\|\mathbf{x}\| \|\mathbf{y}\|$ 指的是内积诱导范数；$\|\cdot\|_1$、$\|\cdot\|_\infty$ 没有内积出身(平行四边形法则已证)，同名形式直接套用是错位。

**以为内积就是点积。** 加权、Frobenius、积分都是内积；判据是三条公理，不是长相。反过来，长得像乘法的 $x_1 y_1 + x_2 y_2 + x_1 y_2$ 因不对称出局。

**复数空间忘共轭。** $\mathbb{C}^n$ 上内积是 $\sum_i x_i \overline{y_i}$，对称性变成共轭对称 $\langle \mathbf{x}, \mathbf{y} \rangle = \overline{\langle \mathbf{y}, \mathbf{x} \rangle}$。本书记号全在实数域；遇到复数场合，这一条要换。

## 相关词条

- [范数](../linear-algebra/norms/)：内积诱导范数 $\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$ 的公理侧；$p = 2$ 三角不等式与 $\sqrt{n}$ 上界本篇补齐
- [角度与正交](../linear-algebra/angles-and-orthogonality/)：内积的几何——余弦、垂直、投影
- [长度与距离](../linear-algebra/lengths-and-distances/)：诱导范数再诱导度量
- [向量空间](../linear-algebra/vector-spaces/)：内积空间 = 向量空间 + 内积
- [坐标](../linear-algebra/coordinates/)：任何 $n$ 维空间配上坐标就是 $\mathbb{R}^n$，内积定理都在原型上演示
- [矩阵范数](../linear-algebra/matrix-norms/)：Frobenius 范数的归宿
- [什么是神经元](../neurons-and-activations/what-is-a-neuron/)：加权和 = 点积
- [自注意力](../attention/self-attention/)：注意力分数 = 内积加缩放
- [余弦相似度](../text-representation/cosine-similarity/)：归一化的内积
