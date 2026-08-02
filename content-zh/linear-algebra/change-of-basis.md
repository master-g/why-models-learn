---
title: "换基:过渡矩阵与坐标换算"
tags: ["why-models-learn"]
---

同一个向量 $\mathbf{v}$，在两组[基](../linear-algebra/basis/) $B$、$C$ 下各有一份[坐标](../linear-algebra/coordinates/)。换基(change of basis)回答它们之间的换算：**从 $B$ 坐标到 $C$ 坐标，乘一个矩阵**：

$$
[\mathbf{v}]_C = P_{C \leftarrow B}\, [\mathbf{v}]_B
$$

这个矩阵 $P_{C \leftarrow B}$ 叫**过渡矩阵**(transition matrix)，它的第 $j$ 列是 $[\mathbf{b}_j]_C$——旧基的第 $j$ 个向量用新基读。公式的推导只有一行，难的是方向纪律：谁读谁、乘在哪一侧，本篇用一套算到底的数字把它钉死。换基是[线性映射](../linear-algebra/linear-maps/)篇「同一映射在不同基下的矩阵」的前置计算，也是 PCA、注意力投影这些工程动作的共同代数。

## 推导：一行

设 $[\mathbf{v}]_B = (x_1, \dots, x_n)$，即 $\mathbf{v} = x_1\mathbf{b}_1 + \cdots + x_n\mathbf{b}_n$。两边取 $[\cdot]_C$，坐标映射线性([坐标](../linear-algebra/coordinates/) 篇)，得

$$
[\mathbf{v}]_C = x_1[\mathbf{b}_1]_C + \cdots + x_n[\mathbf{b}_n]_C
$$

右边正是「以 $[\mathbf{b}_j]_C$ 为列的矩阵」乘「列向量 $(x_1, \dots, x_n)$」——[矩阵乘法](../linear-algebra/matrix-multiplication/) 的列视角。所以 $P_{C \leftarrow B}$ 的构造与公式同时成立：

$$
P_{C \leftarrow B} = \begin{pmatrix} [\mathbf{b}_1]_C & \cdots & [\mathbf{b}_n]_C \end{pmatrix}, \qquad [\mathbf{v}]_C = P_{C \leftarrow B}\, [\mathbf{v}]_B
$$

记法 $C \leftarrow B$ 按数据流向读：把 $B$ 坐标送进，把 $C$ 坐标送出；箭头指向目的地。

## 方向纪律：谁读谁

构造规则一句话：**列是「用新基读旧基」**。$P_{C \leftarrow B}$ 的第 $j$ 列是旧基向量 $\mathbf{b}_j$ 在新基 $C$ 下的坐标。这常常与直觉相反——把 $B$ 坐标换算到 $C$ 坐标的矩阵，要用 $C$ 去读 $B$ 的向量。

两个立刻的推论。第一，反方向换算用逆矩阵：$P_{B \leftarrow C} = (P_{C \leftarrow B})^{-1}$，两组都是基保证过渡矩阵满秩可逆([秩](../linear-algebra/rank/) 篇：满秩 $\iff$ 可逆)。第二，换基可以链式进行：经过第三组基 $D$ 中转，$P_{D \leftarrow B} = P_{D \leftarrow C}\, P_{C \leftarrow B}$——先换到 $C$ 再换到 $D$，与一步到位相同。

## 一套数字算到底

三组基全部来自前几篇：标准基 $E$；$B = (\mathbf{u}, \mathbf{v})$，$\mathbf{u} = (3, 1)$、$\mathbf{v} = (1, 2)$；$C' = (\mathbf{p}, \mathbf{q})$，$\mathbf{p} = (1, 1)$、$\mathbf{q} = (1, -1)$。试验向量 $\mathbf{b} = (4, 6)$，其坐标已知：$[\mathbf{b}]_B = (0.4, 2.8)$，$[\mathbf{b}]_{C'} = (5, -1)$。

**与标准基互换：最不费脑的换基。** $P_{E \leftarrow B}$ 的列是 $\mathbf{u}$、$\mathbf{v}$ 用标准基读——就是它们自己，所以过渡矩阵是基矩阵 $\begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$。验证：$\begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 0.4 \\ 2.8 \end{pmatrix} = \begin{pmatrix} 4 \\ 6 \end{pmatrix}$。反方向用逆矩阵，而 $2 \times 2$ 的逆可以直接写：

$$
P_{B \leftarrow E} = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}^{-1} = \frac{1}{5}\begin{pmatrix} 2 & -1 \\ -1 & 3 \end{pmatrix}
$$

乘回验证是单位阵；作用于 $(4, 6)$ 得 $\frac{1}{5}(2 \cdot 4 - 6,\; -4 + 3 \cdot 6) = (0.4, 2.8)$。注意这个逆矩阵逐元读出「$a = (2b_1 - b_2)/5$、$c = (3b_2 - b_1)/5$」——正是 [线性组合与张成](../linear-algebra/linear-combinations-and-span/) 篇消元推出的通解公式。通解公式不是巧合，它就是换基矩阵的逆。

**两组非标准基之间。** 构造 $P_{C' \leftarrow B}$：用 $C'$ 读 $B$ 的两个向量。解 $a + c = 3$、$a - c = 1$ 得 $[\mathbf{u}]_{C'} = (2, 1)$；解 $a + c = 1$、$a - c = 2$ 得 $[\mathbf{v}]_{C'} = (1.5, -0.5)$。于是

$$
P_{C' \leftarrow B} = \begin{pmatrix} 2 & 1.5 \\ 1 & -0.5 \end{pmatrix}, \qquad \begin{pmatrix} 2 & 1.5 \\ 1 & -0.5 \end{pmatrix} \begin{pmatrix} 0.4 \\ 2.8 \end{pmatrix} = \begin{pmatrix} 0.8 + 4.2 \\ 0.4 - 1.4 \end{pmatrix} = \begin{pmatrix} 5 \\ -1 \end{pmatrix} = [\mathbf{b}]_{C'}
$$

与 [基](../linear-algebra/basis/) 篇直接求解的结果一致。链式验证：经标准基中转，$P_{C' \leftarrow B} = P_{C' \leftarrow E}\, P_{E \leftarrow B}$，即 $Q^{-1}$ 乘基矩阵($Q = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$，$Q^{-1} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$)，乘出来正是 $\begin{pmatrix} 2 & 1.5 \\ 1 & -0.5 \end{pmatrix}$。反方向：$P^{-1} = \begin{pmatrix} 0.2 & 0.6 \\ 0.4 & -0.8 \end{pmatrix}$，作用于 $(5, -1)$ 回到 $(0.4, 2.8)$。

![殊途同归：向量 b 在上，两种读法在下——先读 B 再乘过渡矩阵 P，与直接读 C 得到同一个答案](/assets/linear-algebra/svg/change-of-basis.1.svg)

## 换基之后，什么变了，什么没变

向量本身没变，变的是读数——[基](../linear-algebra/basis/) 篇的结论在执行层的面貌。几何事实(共线、张成、无关)对所有基同真，因为它们只用加法与数乘陈述；数值事实(分量大小、坐标正负)随基而变。

同一个线性映射，在不同基下有不同的矩阵；[线性映射](../linear-algebra/linear-maps/) 篇将给出它们的关系：共轭 $A' = P^{-1}AP$。映射没变，矩阵变了——与向量情形的「向量没变，坐标变了」是同一条原理的两个层面。

## 神经网络里的换基

PCA 是一次换基：把数据中心化后乘一个正交矩阵，坐标转到方差最大的方向上，前几个新坐标就承载了大部分信息。注意力里的 $W_Q$、$W_K$ 也是换基动作：同一批向量分别读进查询空间与键空间，再在新坐标下做内积比较。

正交基让换基几乎免费：正交矩阵的逆是转置($Q^{-1} = Q^T$)，不用消元、不用求逆，一次矩阵乘法搞定。这是标准正交基在计算上受宠的直接原因，内积与正交的篇章([正交投影](../linear-algebra/orthogonal-projections/))展开。

## 失效模式与常见误区

**方向记反。** $P_{C \leftarrow B}$ 的列是「用 $C$ 读 $B$」，不是「用 $B$ 读 $C$」。拿不准时做两件事：看记法箭头指向(目的地在左)，或拿一个已知向量试乘，对不上就是反了。

**乘错一侧。** 坐标是列向量，过渡矩阵乘在左边：$[\mathbf{v}]_C = P[\mathbf{v}]_B$。写成 $[\mathbf{v}]_B P$ 形状都不对(方阵时形状侥幸对，答案错)。

**以为换基变了向量。** 换基只换读数；向量、几何关系、张成与无关性都不动。觉得「乘以 $P$ 把向量转走了」，是把坐标当成了向量本身。

**忘记过渡矩阵必可逆。** 两组都是基，$P$ 必满秩。算出来不可逆，说明有一组「基」名不副实——回 [基](../linear-algebra/basis/) 篇查两个条件。

## 相关词条

- [坐标](../linear-algebra/coordinates/)：换基的对象与工具($[\cdot]_B$ 记法、坐标映射线性)
- [基](../linear-algebra/basis/)：过渡矩阵可逆的原因；同一向量的不同读数
- [矩阵乘法](../linear-algebra/matrix-multiplication/)：列视角——$P\mathbf{x}$ 是 $P$ 各列的组合，推导就靠它
- [线性映射](../linear-algebra/linear-maps/)：同一映射在不同基下的矩阵，共轭关系
- [秩](../linear-algebra/rank/)：满秩 $\iff$ 可逆
- [线性组合与张成](../linear-algebra/linear-combinations-and-span/)：通解公式 = 换基矩阵的逆
- [正交投影](../linear-algebra/orthogonal-projections/)：正交基下换基免费的原因
