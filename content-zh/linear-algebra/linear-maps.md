---
title: "线性映射:组合保持与矩阵表示"
tags: ["why-models-learn"]
---

两个向量空间之间的映射 $T: V \to W$ 是**线性映射**(linear map)，如果它保持线性组合：

$$
T(c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k) = c_1T(\mathbf{v}_1) + \cdots + c_kT(\mathbf{v}_k)
$$

等价地，$T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ 且 $T(\lambda\mathbf{v}) = \lambda T(\mathbf{v})$(组合展开即得等价)。一个立刻的推论：$T(\mathbf{0}) = \mathbf{0}$，因为 $T(\mathbf{0}) = T(0 \cdot \mathbf{v}) = 0 \cdot T(\mathbf{v}) = \mathbf{0}$。[矩阵](../linear-algebra/matrices/) 篇把矩阵介绍为「线性映射」时欠下的定义，本篇偿还：选定基之后，线性映射就是矩阵，矩阵就是线性映射的[坐标](../linear-algebra/coordinates/)；[换基](../linear-algebra/change-of-basis/) 篇提到的共轭公式也在此落地。

## 线性，还是不线性

判一个映射线不线性，查组合保持；查组合保持，先查 $T(\mathbf{0}) = \mathbf{0}$ 这个必要体检(不通过即出局，通过还要再查加法与数乘)。

**线性的例子。** 平面上的旋转、拉伸、剪切、投影；「转置」($A \mapsto A^T$，$(A + B)^T = A^T + B^T$)；以及一个不住在数组里的例子：求导。$D: P_2 \to P_1$，$D(a + bx + cx^2) = b + 2cx$——和的导数是导数的和，常数倍可以提出，组合保持成立。「求导是线性映射」不是比喻，是定义逐条核对后的结论，本篇后面给它排矩阵。

**非线性的例子。** 平移 $T(\mathbf{v}) = \mathbf{v} + \mathbf{a}$($\mathbf{a} \neq \mathbf{0}$)：$T(\mathbf{0}) = \mathbf{a} \neq \mathbf{0}$，体检出局——注意它的图像明明是一条直线，「线性」管的却不是图像直不直，是组合保不保。逐分量平方、取绝对值、max 也都出局。神经网络最爱的 ReLU 同样不行：$\text{ReLU}(-1) + \text{ReLU}(1) = 0 + 1 = 1$，而 $\text{ReLU}(-1 + 1) = \text{ReLU}(0) = 0$，加法不保。激活函数必须非线性是 [为什么需要非线性](../neurons-and-activations/why-non-linearity/) 的主题；这里记下的是另一半事实：线性层($\mathbf{y} = W\mathbf{x}$)是线性映射，加偏置($\mathbf{y} = W\mathbf{x} + \mathbf{b}$)严格说是仿射映射，归宿在 [仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/)。

## 矩阵是线性映射的坐标

线性映射有一个决定性的性质：**它由基向量上的取值唯一确定**。任取 $\mathbf{v} = x_1\mathbf{b}_1 + \cdots + x_n\mathbf{b}_n$，组合保持给出 $T(\mathbf{v}) = x_1T(\mathbf{b}_1) + \cdots + x_nT(\mathbf{b}_n)$——知道 $T$ 在 $n$ 个基向量上的去向，就知道它在所有向量上的去向。指定的自由度恰好是 $n$ 个向量、每个 $m$ 个坐标，共 $mn$ 个数([维数](../linear-algebra/dimension/) 篇的矩阵空间维数在此回收：Hom($V, W$) 的维数是 $mn$)。

把这 $mn$ 个数排起来就是矩阵。选定 $V$ 的基 $B$、$W$ 的基 $C$：

$$
[T]_{C \leftarrow B} = \begin{pmatrix} [T(\mathbf{b}_1)]_C & \cdots & [T(\mathbf{b}_n)]_C \end{pmatrix}
$$

**第 $j$ 列是第 $j$ 个基向量去向的坐标**。[矩阵](../linear-algebra/matrices/) 篇的「第 $j$ 列 = $\mathbf{e}_j$ 的去向」是它在标准基下的样子，这里是一般形式。于是对任何 $\mathbf{v}$：

$$
[T(\mathbf{v})]_C = [T]_{C \leftarrow B}\, [\mathbf{v}]_B
$$

映射的计算变成了矩阵乘法——坐标把「映射」也变成了数组运算，与 [坐标](../linear-algebra/coordinates/) 篇把「向量」变成数组是同一次胜利。

**例一(熟悉的)。** 剪切 $S = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$：$T(\mathbf{e}_1) = (1, 0)$、$T(\mathbf{e}_2) = (1, 1)$，读数排成列，正是 $S$ 自己。

**例二(不住在数组里的)。** 求导 $D: P_2 \to P_1$，基取 $(1, x, x^2)$ 与 $(1, x)$：$D(1) = 0$ 读作 $(0, 0)$，$D(x) = 1$ 读作 $(1, 0)$，$D(x^2) = 2x$ 读作 $(0, 2)$，于是

$$
[D] = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 2 \end{pmatrix}
$$

验证：[坐标](../linear-algebra/coordinates/) 篇的多项式 $p = 2 - x + 0.5x^2$(坐标 $(2, -1, 0.5)$)被作用后得 $\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 2 \end{pmatrix} \begin{pmatrix} 2 \\ -1 \\ 0.5 \end{pmatrix} = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$，即 $Dp = -1 + x$——与直接求导一致。微积分的一个基本运算，在线性代数眼里是一个 $2 \times 3$ 矩阵。

![求导是线性映射：左，多项式 p=2−x+0.5x²；中，它的矩阵 [D](2×3，第 j 列是第 j 个基元素导数的坐标)；右，像 Dp=−1+x 是一条直线](/assets/linear-algebra/svg/linear-maps.1.svg)

## 复合是乘法

两个线性映射复合 $S \circ T$ 仍线性(组合保持可以传递)，其矩阵是矩阵的乘积：

$$
[S \circ T] = [S] \cdot [T]
$$

这不是约定，是被迫：[矩阵乘法](../linear-algebra/matrix-multiplication/) 篇见过，$AB$ 的第 $j$ 列是 $A$ 乘 $B$ 的第 $j$ 列——先经 $T$ 送走基向量、再经 $S$ 送往下一程，正是复合映射里基向量的去向。矩阵乘法那个初看别扭的「行乘列」定义，在映射复合下是唯一正确的选择。

这也给 [为什么需要非线性](../neurons-and-activations/why-non-linearity/) 篇的坍缩实验补上收尾的一句话：两层不带激活的网络 $W_2W_1\mathbf{x}$ 仍是一个线性映射，所以多层等于一层；深度要换来表达力，非线性一次都不能少。

## 同一映射，不同基：共轭

矩阵是映射在基下的坐标，换基当然换矩阵。[换基](../linear-algebra/change-of-basis/) 篇的过渡矩阵给出换算规则：若 $B \to B'$、$C \to C'$，则

$$
[T]_{C' \leftarrow B'} = P_{C' \leftarrow C}\, [T]_{C \leftarrow B}\, P_{B \leftarrow B'}
$$

最常用的是 $V \to V$ 的自映射、两侧同基的情形：$A' = P^{-1}AP$。相差一个共轭的两个矩阵叫**相似**(similar)——同一映射在不同基下的两张照片。

用剪切验证。$S = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$，换到基 $B = (\mathbf{u}, \mathbf{v})$($\mathbf{u} = (3, 1)$、$\mathbf{v} = (1, 2)$)：

$$
S' = B^{-1}SB = \frac{1}{5}\begin{pmatrix} 2 & -1 \\ -1 & 3 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 1.4 & 0.8 \\ -0.2 & 0.6 \end{pmatrix}
$$

照片全变了，但有些数没变：$\operatorname{tr} S' = 1.4 + 0.6 = 2 = \operatorname{tr} S$，$\det S' = 1.4 \times 0.6 - 0.8 \times (-0.2) = 1 = \det S$。迹与行列式在共轭下不变——它们是属于映射本身的数值特征，不属于任何一组基。行列式的正式处理在 [行列式](../linear-algebra/determinant/) 篇。

## 神经网络里的线性映射

深度学习的参数，几乎全部住在线性映射里：embedding 是词表空间到表示空间的线性映射([坐标](../linear-algebra/coordinates/) 篇：$\mathbf{e}_i$ 乘矩阵 = 取列)，每个全连接层、注意力的 $W_Q, W_K, W_V, W_O$、输出头，都是。网络的设计，很大程度上是设计一串线性映射的形状(宽度、秩、共享方式)，再在线性映射之间安插固定的非线性。

微积分那边，线性映射还会回来：多元函数在一点的最佳线性逼近是雅可比矩阵，梯度下降沿着线性近似走。那是后话；眼下记住对应关系：$V \to W$ 的线性映射，选定基后，就是 $m \times n$ 矩阵——列是基向量去向的读数。

## 失效模式与常见误区

**以为图像直就是线性。** $T(\mathbf{v}) = 2\mathbf{v} + \mathbf{1}$ 的图像是直线，但 $T(\mathbf{0}) \neq \mathbf{0}$，不是线性映射。「线性」要求组合保持，过原点是必要的体检，直线不直线是仿射的事。

**以为线性映射可以随意指定。** 基向量上的取值可以任意指定($mn$ 个自由度)，一旦指定，其余向量的去向全部被组合保持锁死。想让 $T$ 在某个非基向量上「例外一下」，做不到。

**把矩阵当映射本身。** 矩阵是坐标，映射是物：换基矩阵全变($S \to S'$ 面目全非)，映射未动。性质分两类：挂在矩阵上的(具体元素)随基而变，属于映射的(秩、迹、行列式)与基无关。

**以为常用运算都线性。** ReLU、绝对值、max、平方都不保加法。网络里真正线性的只有矩阵乘法那几步；非线性是另外花钱买的，而且它必须存在。

## 相关词条

- [矩阵](../linear-algebra/matrices/)：第三视角的定义落地
- [矩阵乘法](../linear-algebra/matrix-multiplication/)：复合视角——乘法定义被迫如此
- [换基](../linear-algebra/change-of-basis/)：共轭公式的工具来源
- [坐标](../linear-algebra/coordinates/)：向量变数组之后，映射也变数组
- [核与像](../linear-algebra/kernel-and-image/)：线性映射的两个基本子空间，下一篇
- [向量空间](../linear-algebra/vector-spaces/)：定义域与值域
- [为什么需要非线性](../neurons-and-activations/why-non-linearity/)：线性堆叠必然坍缩
- [仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/)：偏置项的归宿
