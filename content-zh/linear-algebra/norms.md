---
title: "范数:长度的三条公理"
tags: ["why-models-learn"]
---

**范数**(norm)是向量空间上给每个向量配一个长度值的函数 $\|\cdot\|$：满足三条公理——正定性、齐次性、三角不等式——才算范数。线性代数章此前只谈代数：加法、数乘、张成、基、线性映射，说不了「这个向量有多大」；范数不回答长度从哪来，先固定长度必须遵守的规则，再看哪些函数够格。最常用的一族是 $p$ 范数，神经网络的损失函数、正则化、梯度监控，用的都是这一族里的成员；范数另一个重要来源(内积)在 [内积](../linear-algebra/inner-products/) 篇，本篇只从三公理出发。

## 三条公理

函数 $\|\cdot\|: V \to \mathbb{R}$ 是范数，当且仅当对所有 $\mathbf{x}, \mathbf{y} \in V$ 与所有标量 $c$ 满足三条。三条各管一件事，逐条配「缺了会怎样」。

**正定性**：$\|\mathbf{x}\| \geq 0$，且 $\|\mathbf{x}\| = 0$ 当且仅当 $\mathbf{x} = \mathbf{0}$。这条管「零向量是唯一没有长度的向量」。反例：$f(\mathbf{x}) = |x_1|$ 只量第一个坐标，$f(0, 5) = 0$——$(0, 5)$ 不是零向量，却被判成长度零。正定性排除的就是这种「看不见某个方向」的尺子。

**齐次性**：$\|c\mathbf{x}\| = |c| \cdot \|\mathbf{x}\|$。这条管「向量缩放，长度同倍缩放」。注意右端是 $|c|$ 不是 $c$：$\|-2\mathbf{x}\| = 2\|\mathbf{x}\|$，反向不改变长度。反例：$f(\mathbf{x}) = x_1^2 + x_2^2$ 把 $\mathbf{x} = (3, 4)$ 映到 $25$，而 $f(2\mathbf{x}) = f(6, 8) = 100 = 4 \times 25$，不是 $2 \times 25$——向量加倍，「长度」翻四倍，这样的函数不是范数(它恰是后面常用范数的平方，本篇还会遇到)。

**三角不等式**：$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$。这条管「长度不能绕过直边」：从原点到 $\mathbf{x} + \mathbf{y}$ 的直边，不长于经 $\mathbf{x}$、$\mathbf{y}$ 两段拼接的折线。数字感：$\mathbf{x} = (3, 4)$、$\mathbf{y} = (1, 2)$ 时

$$
\|\mathbf{x} + \mathbf{y}\|_2 = \|(4, 6)\|_2 = \sqrt{52} = 2\sqrt{13} \approx 7.21 \;\leq\; \|\mathbf{x}\|_2 + \|\mathbf{y}\|_2 = 5 + \sqrt{5} \approx 7.24
$$

## $p$ 范数：一个公式，一族尺子

对实数 $p \geq 1$，$p$ 范数定义为

$$
\|\mathbf{x}\|_p = \Big(\sum_{i=1}^{n} |x_i|^p\Big)^{1/p}
$$

一个公式生成一族尺子，常用三把：

- $p = 2$：**欧几里得范数**，日常说的长度，$\|(3, 4)\|_2 = \sqrt{9 + 16} = 5$；
- $p = 1$：**曼哈顿范数**，坐标绝对值直接相加，$\|(3, 4)\|_1 = 3 + 4 = 7$——出租车在方格街上开不出对角线，只能横竖走；
- $p \to \infty$：**最大模范数**，定义成极限 $\|\mathbf{x}\|_\infty = \lim_{p \to \infty} \|\mathbf{x}\|_p$，结果恰是最大坐标的绝对值 $\max_i |x_i|$，$\|(3, 4)\|_\infty = 4$。

极限为什么收敛到最大坐标：以 $\mathbf{x} = (3, 4)$ 为例提出最大项，

$$
\|\mathbf{x}\|_p = (3^p + 4^p)^{1/p} = 4\Big(1 + \big(\tfrac{3}{4}\big)^p\Big)^{1/p}
$$

$p$ 增大时 $(3/4)^p \to 0$，括号趋于 $1$，而 $1^{1/p} = 1$，整个式子趋于 $4$。实跑数字把收敛过程摆出来：

$$
\begin{array}{c|ccccccc}
p & 1 & 2 & 3 & 4 & 10 & 20 & 50 \\ \hline
\|\mathbf{x}\|_p & 7 & 5 & 4.498 & 4.285 & 4.022 & 4.001 & 4.000
\end{array}
$$

同一个向量，三把尺子三个数：$(3, 4)$ 是 $7$、$5$、$4$；$(1, 2, 2)$ 是 $\|\cdot\|_1 = 5$、$\|\cdot\|_2 = 3$、$\|\cdot\|_\infty = 2$。所以「这个向量多长」是个病句，得说「在哪个范数下多长」。

$p$ 为什么不能小于 $1$：此时三角不等式失效。取 $\mathbf{x} = (1, 0)$、$\mathbf{y} = (0, 1)$，按公式 $\|\mathbf{x}\|_{1/2} = 1$、$\|\mathbf{y}\|_{1/2} = 1$，但

$$
\|\mathbf{x} + \mathbf{y}\|_{1/2} = \|(1, 1)\|_{1/2} = (1 + 1)^2 = 4 \;>\; 1 + 1 = \|\mathbf{x}\|_{1/2} + \|\mathbf{y}\|_{1/2}
$$

直边比折线还长，$p < 1$ 的「范数」不是范数。

三角不等式的证明按 $p$ 分工：$p = 1$ 时逐项用 $|a + b| \leq |a| + |b|$ 相加即得；$p = \infty$ 时 $\max_i |x_i + y_i| \leq \max_i (|x_i| + |y_i|) \leq \max_i |x_i| + \max_j |y_j|$，也是两行；$1 < p < \infty$ 的一般情形(Minkowski 不等式)需要 Hölder 不等式，其中 $p = 2$ 的特例随 Cauchy–Schwarz 不等式免费得到——这两件工具都在 [内积](../linear-algebra/inner-products/) 篇。

## 单位球：范数的形状

**单位球**是集合 $\{\mathbf{x} : \|\mathbf{x}\| = 1\}$，即「长度恰为 $1$ 的向量们」。二维平面上，三把尺子的单位球是三种形状：$p = 2$ 是圆($x^2 + y^2 = 1$)；$p = 1$ 是菱形($|x| + |y| = 1$，顶点在 $(\pm 1, 0)$、$(0, \pm 1)$)；$p = \infty$ 是方块($\max(|x|, |y|) = 1$，边贴在 $x = \pm 1$、$y = \pm 1$)。三个形状层层包含：菱形 $\subset$ 圆 $\subset$ 方块——范数越小，单位球越大。

![二维平面上三种范数的单位球：p=2 是圆(墨)、p=1 是菱形(珊瑚)、p=∞ 是方块(灰虚线)，菱形 ⊂ 圆 ⊂ 方块；右：向量 (3, 4) 在三种范数下的长度 7、5、4](/assets/linear-algebra/svg/norms.1.svg)

包含关系背后是范数之间的等价不等式。对 $n$ 维空间：

$$
\|\mathbf{x}\|_\infty \;\leq\; \|\mathbf{x}\|_2 \;\leq\; \sqrt{n}\,\|\mathbf{x}\|_\infty, \qquad
\|\mathbf{x}\|_2 \;\leq\; \|\mathbf{x}\|_1 \;\leq\; \sqrt{n}\,\|\mathbf{x}\|_2
$$

左边四条里，三条各一行话：$\|\mathbf{x}\|_\infty = \max_i |x_i| \leq \sqrt{\sum_i x_i^2}$；$\sum_i x_i^2 \leq n \max_i x_i^2$，开方得右半；$\|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1$ 把 $(\sum_i |x_i|)^2$ 展开即见。剩下 $\|\mathbf{x}\|_1 \leq \sqrt{n}\,\|\mathbf{x}\|_2$ 一条要用 Cauchy–Schwarz，留给 [内积](../linear-algebra/inner-products/) 篇。数字验证($n = 2$，$\mathbf{x} = (3, 4)$)：

$$
4 \leq 5 \leq \sqrt{2} \cdot 4 \approx 5.66, \qquad 5 \leq 7 \leq \sqrt{2} \cdot 5 \approx 7.07
$$

三维的 $(1, 2, 2)$：$2 \leq 3 \leq 2\sqrt{3} \approx 3.46$，$3 \leq 5 \leq 3\sqrt{3} \approx 5.20$。

更一般地，**有限维空间上任意两个范数都等价**：存在常数 $c, C > 0$ 使 $c\|\mathbf{x}\|_a \leq \|\mathbf{x}\|_b \leq C\|\mathbf{x}\|_a$(证明需要一点分析，此处只引结论)。等价的直接后果：「序列是否收敛」「集合是否开闭」这类定性问题，与选哪把尺子无关——迭代算法的停机判据用哪个范数都行。但「只差一个常数因子」放到正则化上却不是细节：$p = 1$ 与 $p = 2$ 当惩罚项的效果定性不同，本篇倒数第二节展开。

## 从长度到距离与极限

有了长度，距离立刻有定义：$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$。度量的三条性质恰好由范数三公理换来：正定性给「距离非负，为零当且仅当重合」；对称性 $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$ 来自齐次性取 $c = -1$，即 $\|\mathbf{x} - \mathbf{y}\| = \|-(\mathbf{y} - \mathbf{x})\| = \|\mathbf{y} - \mathbf{x}\|$；度量的三角不等式就是范数的三角不等式换个写法。数字：$(3, 4)$ 到 $(1, 2)$ 的欧几里得距离是 $\|(2, 2)\|_2 = 2\sqrt{2} \approx 2.83$。

有了距离，极限就能说：$\mathbf{x}_k \to \mathbf{x}$ 当且仅当 $\|\mathbf{x}_k - \mathbf{x}\| \to 0$。迭代算法的停机判据 $\|\mathbf{x}_{k+1} - \mathbf{x}_k\| < \varepsilon$ 就是这么一句话。度量的展开在 [长度与距离](../linear-algebra/lengths-and-distances/) 篇。

## 神经网络里的范数

损失函数是预测误差的范数。回归的均方误差 $\mathrm{MSE} = \frac{1}{n}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$：注意平方之后的 $\|\cdot\|_2^2$ 不满足齐次性(第一节的反例)，不是范数——但平方是单调变换，最小化 $\|\cdot\|_2^2$ 与最小化 $\|\cdot\|_2$ 的最优点完全相同，优化照用。[损失函数](../training-nn/loss-functions/) 篇展开各任务的范数选择。

正则化是对权重加范数惩罚。取 $\mathbf{w} = (3, 4)$ 演示：L1 惩罚 $\|\mathbf{w}\|_1 = 7$；L2 惩罚常用 $\frac{1}{2}\|\mathbf{w}\|_2^2 = 12.5$。两者的梯度形态不同：L2 的梯度是 $\mathbf{w}$ 本身，按比例把权重往零收缩；L1 的梯度是 $\operatorname{sign}(\mathbf{w}) = (1, 1)$，每个坐标受恒定的力。结合单位球的形状看，L1 的菱形顶点在坐标轴上——惩罚项的等值面先碰到顶点，解就被推到「某些坐标恰为零」的位置，这是 L1 产生稀疏解的几何来源。[过拟合与正则化](../learning-framework/overfitting-and-regularization/) 与 [岭回归与 Lasso](../linear-models/ridge-and-lasso/) 篇展开。

梯度本身也用范数监控：$\|\nabla L\|$ 突然变大，说明参数更新过猛，[梯度裁剪](../training-nn/gradient-clipping/) 直接按范数把梯度缩回去；$\|\nabla L\|$ 长期贴近零，则是消失的信号。[梯度下降](../training-nn/gradient-descent/) 篇的收敛判据，就是上节「从长度到极限」的用法。

权重矩阵 $W$ 的「长度」是另一个话题：矩阵范数有自己的一族定义，[矩阵范数](../linear-algebra/matrix-norms/) 篇处理。

## 失效模式与常见误区

**把范数平方当范数。** $\|\mathbf{x}\|_2^2$ 违反齐次性：$\|2\mathbf{x}\|_2^2 = 4\|\mathbf{x}\|_2^2 \neq 2\|\mathbf{x}\|_2^2$。MSE 用平方是因为平方单调、不改最优点，不是因为平方后还是范数。

**齐次性忘了绝对值。** 写出 $\|-2\mathbf{x}\| = -2\|\mathbf{x}\|$ 该立刻警觉：长度不能为负。正确的是 $\|-2\mathbf{x}\| = 2\|\mathbf{x}\|$。

**报长度不报范数。** 「$(3, 4)$ 的长度」可以是 $7$、$5$、$4$。论文和代码里出现「误差 $\varepsilon = 0.5$」时，先确认是哪个范数、带不带平方。

**以为 $p$ 越小尺子越精细。** $p < 1$ 直接丢掉三角不等式($\|(1, 1)\|_{1/2} = 4 > 2$)，不是范数。常听到的「$\ell_0$ 范数」(非零元个数)也不是范数：$\|2\mathbf{x}\|_0 = \|\mathbf{x}\|_0$，齐次性就过不了。ML 里照用，是因为数非零元有用，不是因为它合法。

**以为范数等价就能随便换。** 等价保住的是收敛性这类定性结论；正则化选 $p = 1$ 还是 $p = 2$，一个把解推向稀疏、一个按比例收缩，等价不等式管不着这个分别。

## 相关词条

- [内积](../linear-algebra/inner-products/)：范数最重要的来源 $\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$；Cauchy–Schwarz 与 $p = 2$ 三角不等式的证法
- [长度与距离](../linear-algebra/lengths-and-distances/)：范数诱导度量的展开
- [向量空间](../linear-algebra/vector-spaces/)：范数定义在向量空间上——先有代数结构，再谈长度
- [矩阵范数](../linear-algebra/matrix-norms/)：矩阵的一族范数
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：范数惩罚的工程动机
- [岭回归与 Lasso](../linear-models/ridge-and-lasso/)：L1/L2 正则的正式处理
- [梯度下降](../training-nn/gradient-descent/)：停机判据与收敛的范数表述
- [损失函数](../training-nn/loss-functions/)：损失 = 误差范数的选择
