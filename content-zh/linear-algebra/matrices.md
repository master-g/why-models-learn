---
title: "矩阵:表格、向量堆叠与线性映射"
tags: ["why-models-learn"]
---

矩阵是把 $m \times n$ 个数按 $m$ 行、$n$ 列排成矩形得到的数学对象，记作 $A \in \mathbb{R}^{m \times n}$，第 $i$ 行第 $j$ 列交叉处的数记作 $a_{ij}$。如果说向量是神经网络里一切数据的最终形态，矩阵就是一切**批量**数据与一切**变换**的最终形态：一批图片是一个矩阵，一层网络的权重是一个矩阵，注意力分数也是一个矩阵。本篇建立理解矩阵的三个视角——表格、向量堆叠、线性映射——定义它的加法、数乘与转置，并认识几类有特殊结构的矩阵。矩阵真正的威力在乘法里，那是下一篇 [matrix-multiplication](../linear-algebra/matrix-multiplication/) 的事；本篇负责把乘法之前的全部地基打牢。

## 记号与形状

矩阵的一般写法是带上全部下标：

$$
A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix} \in \mathbb{R}^{m \times n}
$$

下标**行先列后**：$a_{ij}$ 是第 $i$ 行、第 $j$ 列交叉处的那个数。取一个具体的：

$$
A = \begin{pmatrix} 2 & 0 & -1 \\ 1 & 3 & 4 \end{pmatrix}
$$

则 $A \in \mathbb{R}^{2 \times 3}$，$a_{11} = 2$，$a_{13} = -1$，$a_{23} = 4$。

$m \times n$ 叫矩阵的**形状**(shape)。形状是矩阵的类型签名：每个矩阵运算能不能做，先看形状答不答应——加法要求两个矩阵同形(本篇)，乘法要求左侧的列数等于右侧的行数(下一篇)。动笔之前先给每个矩阵标好形状，是线性代数最值得养成的习惯，没有之一。

行数等于列数的矩阵叫**方阵**($n \times n$)。方阵享有特殊地位：行列式、逆矩阵、特征值这些概念都只对方阵有定义，分别见 [determinant](../linear-algebra/determinant/)、[matrix-inverse](../linear-algebra/matrix-inverse/)、[eigenvalues-and-eigenvectors](../linear-algebra/eigenvalues-and-eigenvectors/)。

最后明确一件上一篇已经埋下的事：向量是矩阵的特例。$n$ 维列向量就是 $n \times 1$ 矩阵，行向量就是 $1 \times n$ 矩阵。所以本篇的所有定义都与 [vectors](../linear-algebra/vectors/) 兼容——向量的加法和数乘规则会原封不动地搬过来。

## 理解矩阵的三个视角

**视角一：矩阵是表格。** 程序员的直觉完全正确：矩阵就是二维数组，就是电子表格。这个视角下矩阵是装数据的容器，而它最重要的装法有固定约定：**行是样本，列是特征**。MNIST 的六万张训练图片，每张拉平成 784 维向量(见 [vectors](../linear-algebra/vectors/))，六万个向量按行排开，就是一个 $60000 \times 784$ 的矩阵 $X$——整个训练集成为一个数学对象，$x_{ij}$ 是第 $i$ 张图片的第 $j$ 个像素。表格视角负责数据的一侧：数据批、协方差矩阵([covariance-matrix](../probability/covariance-matrix/))、注意力分数矩阵([attention-matrix](../attention/attention-matrix/))，首先都是表格。

**视角二：矩阵是向量的堆叠。** 换一个切法：把矩阵按列拆开。$m \times n$ 矩阵是 $n$ 个 $m$ 维列向量并排站着：

$$
A = \begin{pmatrix} | & | & & | \\ \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_n \\ | & | & & | \end{pmatrix}
$$

也可以按行看成 $m$ 个 $n$ 维行向量叠起来。拿上一篇的两个向量当例子：$\mathbf{u} = (3, 1)$、$\mathbf{v} = (1, 2)$ 并排一站，就是一个 $2 \times 2$ 矩阵：

$$
\begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}
$$

![矩阵是向量的堆叠：u 与 v 两个向量并排站着，就是一个 2×2 矩阵的两列](/assets/linear-algebra/svg/matrices.1.svg)

这个视角为什么重要？因为矩阵的很多深层性质，其实是它的**列向量组**的性质：列向量张成什么空间([linear-combinations-and-span](../linear-algebra/linear-combinations-and-span/))、列之间是否线性无关([linear-independence](../linear-algebra/linear-independence/))、所谓秩就是列向量组的秩([rank](../linear-algebra/rank/))。更实际的好处要等下一篇：矩阵乘向量，结果恰好是列向量的线性组合——列视角是理解乘法的钥匙。

**视角三：矩阵是线性映射。** 前两个视角里矩阵都是被动的：装数据，或者被拆开。第三个视角把矩阵看成主动的：**它是一个变换，吃进去一个向量，吐出一个向量**。具体怎么吃(矩阵乘法的定义)留给下一篇，这里先建立几何预期。看一个具体的 $2 \times 2$ 矩阵：

$$
S = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
$$

它作用在平面上的效果是一个**剪切**(shear)：每个点 $(x, y)$ 被挪到 $(x + y, y)$——横坐标加上纵坐标的量，纵坐标不动。越靠上的点被推得越远，方格纸被扭成斜格，正方形格子变成平行四边形。追踪两个标准基向量：$\mathbf{e}_1 = (1, 0)$ 原地不动，$\mathbf{e}_2 = (0, 1)$ 被推到 $(1, 1)$。

![矩阵作为线性映射：剪切把方格扭成斜格，e1 不动、e2 被推到 (1, 1)，而两者的去向恰好是矩阵的两列](/assets/linear-algebra/svg/matrices.2.svg)

现在注意一个不是巧合的事实：$\mathbf{e}_1$ 的去向 $(1, 0)$ 是 $S$ 的第 1 列，$\mathbf{e}_2$ 的去向 $(1, 1)$ 是 $S$ 的第 2 列。**矩阵的第 $j$ 列，就是第 $j$ 个标准基向量被变换后的去向。** 下一篇定义了乘法之后，这是一行就能验证的事，但它意味深长：想知道一个矩阵「做什么」，不必盯着 $mn$ 个数字看，看它的列就够了——基向量的去向定了，整个变换就定了。

这类变换叫「线性」映射，因为它保持向量的两种基本运算：先加再变换等于先变换再加，先缩放再变换等于先变换再缩放。几何后果是：直线变换后还是直线，原点永远不动(所以平移不是线性映射，它属于仿射，见 [affine-spaces-and-maps](../linear-algebra/affine-spaces-and-maps/))。「线性」的公理化展开在 [linear-maps](../linear-algebra/linear-maps/)。

三个视角各有分工：数据一侧用表格，做乘法、谈秩用向量堆叠，理解模型行为用线性映射。权重矩阵三者全占：它是存参数的表格，是一组列向量，更是「这一层对表示施加的变换」。

## 加法与数乘

定义照搬向量：同形矩阵相加，对应位置的数分别相加；一个数乘矩阵，乘到每个位置上：

$$
\begin{pmatrix} 1 & 2 \\ 0 & -1 \end{pmatrix} + \begin{pmatrix} 3 & 0 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 4 & 2 \\ 1 & 0 \end{pmatrix}, \qquad 2 \begin{pmatrix} 1 & 2 \\ 0 & -1 \end{pmatrix} = \begin{pmatrix} 2 & 4 \\ 0 & -2 \end{pmatrix}
$$

前提照旧：加法只在**同形**矩阵之间有定义——形状不同，位置对不齐。

性质也照旧免费继承：交换律、结合律、零矩阵(分量全为 0，记作 $O$，是加法的单位元)、加法逆元，逐分量展开即可验证，不再赘述。值得点破的是另一面：**全体 $m \times n$ 矩阵在加法与数乘下的行为与向量一模一样**，所以到了 [vector-spaces](../linear-algebra/vector-spaces/)，矩阵会正式获得向量身份。这不是纯形式主义：训练神经网络时，权重矩阵就是被当作一个高维向量来搜索的，梯度下降对它的全部参数同步微调，与在 $\mathbb{R}^{100352}$ 里移动一个点没有区别(这个数的来历见下文「神经网络里的矩阵」)。

## 转置

把矩阵沿主对角线「翻过来」：第 $i$ 行变成第 $i$ 列，第 $j$ 列变成第 $j$ 行。得到的矩阵叫 $A$ 的**转置**，记作 $A^T$：

$$
A = \begin{pmatrix} 2 & 0 & -1 \\ 1 & 3 & 4 \end{pmatrix} \quad\Longrightarrow\quad A^T = \begin{pmatrix} 2 & 1 \\ 0 & 3 \\ -1 & 4 \end{pmatrix}
$$

用下标说：$A^T$ 的 $(i, j)$ 元是 $A$ 的 $(j, i)$ 元，形状随之从 $m \times n$ 变成 $n \times m$。上面例子里 $a_{13} = -1$，转置后它出现在第 3 行第 1 列。

三条性质逐分量可验：

1. $(A^T)^T = A$——翻两次就回来了。
2. $(A + B)^T = A^T + B^T$——先加再翻等于先翻再加。
3. $(cA)^T = cA^T$——数乘与转置互不干扰。

乘法的转置规律 $(AB)^T = A^T B^T$ 留给下一篇——注意顺序反过来了，到时候会看清为什么。

**对称矩阵。** 满足 $A^T = A$ 的矩阵叫对称矩阵。它必须是方阵，且关于主对角线镜像对称：

$$
\begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}
$$

对称矩阵在机器学习里反复出现：数据的协方差矩阵天然对称([covariance-matrix](../probability/covariance-matrix/))，注意力分数矩阵也常常对称或接近对称([attention-matrix](../attention/attention-matrix/))。对称不只是漂亮——实对称矩阵必能被一组正交的坐标轴对角化，这是 [eigenvalues-and-eigenvectors](../linear-algebra/eigenvalues-and-eigenvectors/) 里的核心好消息之一。

顺带一提，转置记号闭环了上一篇的行/列问题：列向量 $\mathbf{x}$ 横着写就是 $\mathbf{x}^T$。

## 几类特殊结构的矩阵

**零矩阵** $O$：分量全为 0。它是加法的单位元($A + O = A$)；映射视角下，它把所有向量压到原点。

**单位矩阵** $I_n$：主对角线为 1、其余全为 0 的 $n \times n$ 方阵。它是矩阵世界的「1」：下一篇会验证 $I_n \mathbf{v} = \mathbf{v}$，即恒等变换——每个向量原样通过。逆矩阵、正交矩阵都以它为参照系定义。

**对角矩阵**：主对角线以外全为 0，记作 $\mathrm{diag}(d_1, \dots, d_n)$。映射视角下它做的事最简单：第 $i$ 个坐标轴独立伸缩 $d_i$ 倍，各轴互不串扰。对角矩阵是最容易分析的变换，矩阵理论的相当大一部分——特征值与对角化——就是在回答「一个一般的矩阵，在多大程度上能化归为对角矩阵」。

**三角矩阵**：主对角线以下全为 0 的叫上三角，以上全为 0 的叫下三角。它是高斯消元的终点形态：解线性方程组，就是把一般方阵化成三角再回代，见 [linear-systems](../linear-algebra/linear-systems/) 与 [gaussian-elimination](../linear-algebra/gaussian-elimination/)。

## 神经网络里的矩阵

上一篇说向量在神经网络里同时是数据、参数、中间状态；矩阵是同一组角色升了一维。

**角色一：数据批。** 深度学习几乎从不一次处理一个向量，而是一次处理一批：$N$ 个样本排成 $X \in \mathbb{R}^{N \times d}$。MNIST 训练集是 $60000 \times 784$，一个 32 张图片的小批量是 $32 \times 784$。批矩阵是 GPU 并行的基本单位——一次矩阵运算处理整批数据，比逐个向量循环快几个数量级。

**角色二：权重。** 一个把 784 维表示变成 128 维表示的全连接层，其变换由权重矩阵 $W$ 承载；「学习」的物理内容，就是调整 $W$ 的 $784 \times 128 = 100352$ 个分量。大模型的参数规模都是这么数出来的：每层的参数量，约等于各权重矩阵的分量数之和。

**角色三：中间表示的批。** 一批数据流进网络，每过一层，表示批矩阵就被变换一次，形状也可能改变，层层接力直到输出。Transformer 里连「词与词的相关程度」本身也是一个矩阵——注意力分数矩阵，见 [attention-matrix](../attention/attention-matrix/)。

## 失效模式与常见误区

**形状不匹配是头号错误源。** 加法要求同形，乘法要求内维相等(下一篇)。数学上非法的形状，代码里却可能被广播机制静默「修复」——不报错，只让结果悄悄变错。养成条件反射：写下任何矩阵运算之前，先给每个矩阵标注形状；运算一步，检查一步。

**表格视角的陷阱：忘了矩阵是映射。** 只把矩阵当容器，就看不懂权重矩阵在「做什么」。两个矩阵数字上只差一个分量，行为可以天差地别：把 $\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ 的右下角改成 0，恒等变换就变成了把整个平面压扁到 $x$ 轴的投影。判断一个矩阵，要看它对向量做的事，而不是看它数字的长相。

**下标方向别搞反。** $a_{ij}$ 是第 $i$ 行第 $j$ 列，行先列后。数学下标从 1 开始、代码下标一般从 0 开始，换算差一格，后面全错。

**转置不是逆。** 转置只是重排下标，几乎没有成本；逆矩阵(存在时)是「把变换撤销」，两者是完全不同的东西，只在正交矩阵这类特例身上重合(见 [orthogonal-matrices-and-rotations](../linear-algebra/orthogonal-matrices-and-rotations/))。看到 $A^T$ 不要自动脑补 $A^{-1}$。

## 相关词条

- [matrix-multiplication](../linear-algebra/matrix-multiplication/)：下一篇，矩阵真正变成线性映射的地方，列视角在那里开花结果
- [linear-maps](../linear-algebra/linear-maps/)：视角三的公理化——「线性」到底保持什么
- [linear-systems](../linear-algebra/linear-systems/)：矩阵的历史发源问题——解方程组
- [vector-spaces](../linear-algebra/vector-spaces/)：矩阵作为向量的正式身份
- [rank](../linear-algebra/rank/)：列向量视角的深化，矩阵「有效自由度」的度量
- [covariance-matrix](../probability/covariance-matrix/)：对称矩阵在统计里的核心角色
- [attention-matrix](../attention/attention-matrix/)：Transformer 里的矩阵
- [mnist-mlp-training-loop](../training-nn/mnist-mlp-training-loop/)：数据批与权重矩阵的实战
