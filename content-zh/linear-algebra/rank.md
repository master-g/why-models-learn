---
title: "秩:列秩、行秩与主元个数"
tags: ["why-models-learn"]
---

秩(rank)是矩阵的一个非负整数：它的列里，线性无关的最多有几个。这个定义一句话就说完了，但它还有两个等价的说法——「行里线性无关的最大个数」与「[高斯消元](../linear-algebra/gaussian-elimination/) 终点阶梯形里主元的个数」，三者总是相等。消元篇说过，主元个数不因消元路径而变，「它就是矩阵的秩」；[线性方程组](../linear-algebra/linear-systems/) 篇的列视角说过，「列向量组能拼出的目标向量的丰富程度，就是秩」。本篇先用一个矩阵把三种数法各看一遍，再证明它们永远给出同一个数(这个证明顺带解释了主元个数为什么路径无关)，然后列出秩的常用性质--其中一条把解的三种情形统一成一句判据--最后看秩在神经网络里的三个角色：瓶颈、信息丢失与低秩适配。

## 一个矩阵，三种数法

看一个 $3 \times 3$ 的矩阵：

$$
A = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}
$$

先数无关的列。三列 $\mathbf{c}_1 = (1, 0, 1)$、$\mathbf{c}_2 = (0, 1, 1)$、$\mathbf{c}_3 = (1, 1, 2)$ 并不「独立」：第三列恰是前两列之和，

$$
\mathbf{c}_1 + \mathbf{c}_2 = (1 + 0,\; 0 + 1,\; 1 + 1) = (1, 1, 2) = \mathbf{c}_3
$$

所以 $\mathbf{c}_3$ 是冗余——有它没它，三列能拼出的向量集合不变。而 $\mathbf{c}_1$ 与 $\mathbf{c}_2$ 互不为倍数，谁也无法替代谁。极大线性无关组含 2 个向量，**列秩**(column rank)是 2(线性无关与张成的严格定义见 [线性无关](../linear-algebra/linear-independence/) 与 [线性组合与张成](../linear-algebra/linear-combinations-and-span/)，本篇只用直观)。

再数无关的行。把矩阵横着看：三行 $\mathbf{r}_1 = (1, 0, 1)$、$\mathbf{r}_2 = (0, 1, 1)$、$\mathbf{r}_3 = (1, 1, 2)$。同样有 $\mathbf{r}_1 + \mathbf{r}_2 = \mathbf{r}_3$(这个矩阵对称，行列关系互为镜像)，**行秩**(row rank)也是 2。

最后数主元。对 $A$ 做高斯消元：$R_3 \leftarrow R_3 - R_1 - R_2$，第三行直接归零：

$$
\begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}
\;\xrightarrow{R_3 \leftarrow R_3 - R_1 - R_2}\;
\begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix}
$$

阶梯形有两个主元(两个 $1$，都在对角线上)，主元个数还是 2。

![左：向量 c₁、c₂ 张成平面，c₃ = c₁ + c₂ 落在平面内(秩 2)；右：三个向量共线，互为倍数(秩 1)](/assets/linear-algebra/svg/rank.1.svg)

三个 2 不是巧合，也不是对称矩阵的专利。列视角在几何上说的是 $A$ 的列张成一个怎样的空间：上例中三列躺在同一平面内($\mathbf{c}_3$ 是平行四边形的对角线)，张成的是二维平面；若三列共线，张成的只是一条直线。列张成的空间有名字，叫**列空间**(column space，正式处理见 [核与像](../linear-algebra/kernel-and-image/))；列空间的维数，正是列秩。

## 行秩为什么等于列秩

三种数法给出同一个数，是线性代数里最不显然的基本事实之一。「无关的列」与「无关的行」甚至不在同一个空间里(列在 $\mathbb{R}^m$ 里，行在 $\mathbb{R}^n$ 里)，直接比较无从下手。但它们都与第三个东西相等：**消元终点的主元个数**。证明只需两条引理，工具在消元篇已经备好。

**引理一：行变换不改变列之间的线性关系。** 列之间的一种线性关系，就是一组系数 $\mathbf{x}$ 使 $A\mathbf{x} = \mathbf{0}$(比如上例 $\mathbf{c}_1 + \mathbf{c}_2 - \mathbf{c}_3 = \mathbf{0}$，系数 $(1, 1, -1)$)。行变换保持方程组的解集(消元篇：三种变换都可逆，解集被一一对应地搬运)，所以 $A\mathbf{x} = \mathbf{0}$ 与变换后的 $A'\mathbf{x} = \mathbf{0}$ 有完全相同的解--**列的线性关系原封不动，连系数都一样**。上例验证：阶梯形里 $\mathbf{c}_3' = (1, 1, 0)$，仍有 $\mathbf{c}_1' + \mathbf{c}_2' = (1, 1, 0) = \mathbf{c}_3'$，系数还是 $(1, 1)$。列间关系不变，极大无关组的大小当然不变：**列秩 = 阶梯形的列秩**。而阶梯形的列秩可以直接读出：主元列彼此无关--最右的主元列在自己的主元行处非零，而更靠左的主元列在该行全是零(阶梯形的定义：主元以下全零)，于是一个零组合从右往左逐个逼出系数为零；每个非主元列的非零分量只出现在主元行上(再往下全是零行)，从最下面的主元行起回代，就能用主元列把它拼出。所以阶梯形的列秩 = 主元个数 $r$。

**引理二：行变换不改变行张成的集合。** 三种行变换产出的新行都是旧行的线性组合(张成集不扩大)；变换可逆，旧行也都是新行的组合(不缩小)。于是行张成的集合在整个消元过程中不变，**行秩 = 阶梯形的行秩**。阶梯形的行秩同样可以直接读出：$r$ 个非零行，主元位置互不相同--每一行在自己的主元列处非零，而其余非零行在该列全是零，谁也无法由别的行拼出。所以行秩 = 非零行数 = $r$。

两条引理合起来：

$$
\text{列秩} \;=\; \text{主元个数} \;=\; \text{行秩}
$$

这个等式还回答了消元篇留下的另一个问题：**主元个数与消元路径无关**。行秩与列秩的定义根本不涉及消元，而它们都等于主元个数--换任何路径消元，主元个数都被这两个路径无关的量夹住，只能是同一个值。从此这个数有了名字：**秩**，记作 $\operatorname{rank}(A)$；「行秩 = 列秩」的另一常见写法是 $\operatorname{rank}(A) = \operatorname{rank}(A^T)$。

## 秩的性质

秩的常用性质可以列成一张清单，每条附一句话理由。

**上界：$\operatorname{rank}(A) \le \min(m, n)$。** 列秩不能超过列数 $n$，行秩不能超过行数 $m$。取到上界的矩阵叫**满秩**(full rank)，够不到上界的叫**秩亏**(rank deficient)。

**转置不变：$\operatorname{rank}(A^T) = \operatorname{rank}(A)$。** 就是「列秩 = 行秩」换个说法。

**可解性判据：$A\mathbf{x} = \mathbf{b}$ 有解，当且仅当 $\operatorname{rank}(A) = \operatorname{rank}([A \mid \mathbf{b}])$。** 有解的意思是 $\mathbf{b}$ 能被 $A$ 的列拼出([线性方程组](../linear-algebra/linear-systems/) 篇的列视角)，也就是把 $\mathbf{b}$ 挂进列里不产生新的无关列。这一条把三种解的情形统一了：相交例($x_1 + x_2 = 4$、$x_1 - x_2 = 2$)两侧的秩都是 2，有解；平行例($x + y = 2$、$2x + 2y = 6$)$\operatorname{rank}(A) = 1$ 而增广后变成 2--多出的那个主元，就来自矛盾行 $[0 \; 0 \mid 2]$ 里的 $\mathbf{b}$ 列；重合例($2x + 2y = 4$)两侧的秩都是 1。

**解空间的维数：$A\mathbf{x} = \mathbf{0}$ 的解空间是 $n - \operatorname{rank}(A)$ 维。** 消元终点里，主元占住的列对应被锁定的未知数，没有主元的列是自由变量，每个自由变量贡献一维。重合例里 $n = 2$、秩 1、自由变量 1 个，解集正是那条一维的直线 $x + y = 2$(严格说它不过原点，是与解空间平行的一维仿射空间，见 [仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/))。这个公式是 [秩零定理](../linear-algebra/rank-nullity/) 的矩阵形式，完整展开(核与像)在那一篇。

**乘积收缩：$\operatorname{rank}(AB) \le \min(\operatorname{rank}(A), \operatorname{rank}(B))$。** $AB$ 的每一列都是 $A$ 各列的线性组合([矩阵乘法](../linear-algebra/matrix-multiplication/) 篇的列视角)，所以 $AB$ 的列空间不会超出 $A$ 的列空间，秩只缩不扩；对 $B^T A^T$ 用同一论证得另一半。等号并不保证：$\operatorname{diag}(1, 0) \cdot \operatorname{diag}(0, 1) = 0$，两个秩 1 乘出秩 0——$B$ 的列空间整个掉进了 $A$ 的核里。

**和收缩：$\operatorname{rank}(A + B) \le \operatorname{rank}(A) + \operatorname{rank}(B)$。** $A + B$ 的每列是 $A$ 列与 $B$ 列之和，张不出两边合起来的范围。同样可以严格小于：$\operatorname{diag}(1, 0) + \operatorname{diag}(-1, 0) = 0$，秩 $1 + 1$ 加出秩 0。

**方阵可逆 ⟺ 满秩。** $n \times n$ 的 $A$ 可逆，当且仅当 $\operatorname{rank}(A) = n$：消元到底、每列一个主元，当且仅当 $A\mathbf{x} = \mathbf{e}_j$ 对每个标准基向量都有解——那些解拼起来就是逆矩阵的列。[逆矩阵](../linear-algebra/matrix-inverse/) 篇展开。

## 神经网络里的秩

全连接层是一个瓶颈：MNIST 分类器的权重 $W \in \mathbb{R}^{784 \times 10}$ 满足 $\operatorname{rank}(W) \le 10$：无论 784 维的输入多么丰富，经过 $W$ 之后都活在至多 10 维的空间里。层的秩，是它输出方向丰富程度的上限。

线性堆叠时，秩只缩不扩：由乘积收缩，$\operatorname{rank}(W_2 W_1) \le \min(\operatorname{rank}(W_1), \operatorname{rank}(W_2))$：纯线性网络每加一层，信息的「通道」只可能更窄，而且这个通道从第一层就被封了顶。这是 [为什么需要非线性](../neurons-and-activations/why-non-linearity/) 的又一条理由：非线性不受秩的束缚——那一篇里 $(0, 1)$ 与 $(1, 0)$ 先被折叠到同一点、再被分开；折叠正是秩 $1$ 映射做的事，而分开是任何线性映射都做不到的。

秩亏就是信息丢失。极端例子：$W = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}$ 的秩是 1。输入 $(3, 1)$ 与 $(5, -1)$ 被映到同一个输出 $(4, 8)$--它们的差 $(2, -2)$ 落在 $W$ 的核里，被整个抹掉。秩亏的层把不同的输入压成同一输出，这份差异不可逆地消失，下游任何层都救不回来。(核的正式处理见 [核与像](../linear-algebra/kernel-and-image/)。)

低秩结构则是可以利用的冗余：微调大模型时，权重的改变量 $\Delta W$ 经验上接近低秩：真正改变的「方向」远少于矩阵的尺寸。于是 [LoRA](../finetuning/lora/) 用 $\Delta W = BA$($B \in \mathbb{R}^{d \times r}$、$A \in \mathbb{R}^{r \times d}$，乘积的秩至多为 $r$)替代全量微调：$d = 4096$、$r = 8$ 时，待训参数从 $4096^2 = 16{,}777{,}216$ 降到 $2 \times 4096 \times 8 = 65{,}536$，只剩 0.39%。为什么巨大的矩阵里真正起作用的常常只有少数方向，这是 [奇异值分解](../linear-algebra/svd/) 与 [低秩近似](../linear-algebra/low-rank-approximation/) 的主题。

## 失效模式与常见误区

**在浮点世界里问「秩是几」。** 数学上两个向量要么平行要么不平行，没有中间态；数值上全是中间态。取 $\mathbf{u} = (1, 1)$、$\mathbf{v} = (1, 1 + 10^{-9})$：它们线性无关，秩是 2；但这对列组成的矩阵，两个奇异值约为 $2.0$ 与 $5 \times 10^{-10}$，小者只有大者的 $2.5 \times 10^{-10}$——任何含噪声的数据、任何迭代算法，都无法区分这个矩阵与某个秩 1 矩阵。所以工程问题不是「秩是几」，而是「**多小算零**」：给定阈值，秩才有意义，而度量大小的尺子正是奇异值(numpy 的默认容差约为 $8.9 \times 10^{-16}$，只管舍入误差，所以这个例子它仍判 2；数据本身的噪声得自己定阈值)。数值秩(numerical rank)的工具在 [奇异值分解](../linear-algebra/svd/)。

**数非零元就当秩。** $\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$ 四个元素全非零，秩却是 1。秩定义在「线性无关」上，不在「非零」上；不消元(或等价的推理)，肉眼读不出。

**以为 $\operatorname{rank}(AB) = \min(\operatorname{rank}(A), \operatorname{rank}(B))$。** 不等式只有 $\le$：乘积的秩可以严格小于两者，对角例子里两个秩 1 乘出秩 0。乘积保持秩需要额外条件(比如左因子列满秩)，那是例外而非常态。

**在大矩阵上手算秩。** 秩是理论工具：它把「解是否存在」「信息的维数」变成一句话判据，但绝不意味着该在百万维矩阵上靠消元把它数出来($n^3$ 的墙见 [高斯消元](../linear-algebra/gaussian-elimination/) 篇)。数值世界估计秩，走奇异值分解。

## 相关词条

- [高斯消元](../linear-algebra/gaussian-elimination/)：主元个数就是秩；「主元个数路径无关」在本篇证成
- [线性方程组](../linear-algebra/linear-systems/)：三种解的情形被 $\operatorname{rank}(A)$ 与 $\operatorname{rank}([A \mid \mathbf{b}])$ 统一
- [矩阵乘法](../linear-algebra/matrix-multiplication/)：列视角与乘积的秩；$AB$ 的列是 $A$ 列的组合
- [线性无关](../linear-algebra/linear-independence/)：极大线性无关组的严格定义
- [线性组合与张成](../linear-algebra/linear-combinations-and-span/)：列张成的集合就是列空间
- [秩零定理](../linear-algebra/rank-nullity/)：$n - \operatorname{rank}(A)$ 的完整展开
- [核与像](../linear-algebra/kernel-and-image/)：秩亏层抹掉的信息由核刻画；列空间的正式名字是像
- [逆矩阵](../linear-algebra/matrix-inverse/)：方阵满秩 ⟺ 可逆
- [仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/)：非齐次方程的解集形状
- [奇异值分解](../linear-algebra/svd/)：数值秩的尺子
- [低秩近似](../linear-algebra/low-rank-approximation/)：低秩结构为什么普遍存在
- [LoRA](../finetuning/lora/)：低秩假设的工程应用
- [为什么需要非线性](../neurons-and-activations/why-non-linearity/)：线性堆叠秩只缩不扩，非线性才逃得掉
