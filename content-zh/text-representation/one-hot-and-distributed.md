---
title: "One-hot 与分布式表示：从离散 ID 到可学习几何"
tags: ["why-models-learn"]
---

one-hot 表示把词表中的一个离散类别写成一个坐标恰好为 1、其余坐标为 0 的向量。若词表大小为 $V$，第 $i$ 个 token 的 one-hot 向量是 $\mathbf e_i\in\mathbb R^V$。它只表达“这是第几个类别”，不表达类别之间的相似性。

分布式表示则把一个对象的信息分散到多个坐标中，用一个通常维度较低的连续向量表示它。对 token 来说，embedding 矩阵 $E\in\mathbb R^{V\times d}$ 把 one-hot 查表为

$$
\mathbf e_i^\mathsf T E=E_{i,:}\in\mathbb R^d.
$$

这个线性代数等式连接了三件事：

1. one-hot 是离散词表上的标准基；
2. embedding 是标准基经过一个可学习线性映射后的向量；
3. 分布式向量的几何关系来自训练得到的 $E$，不是来自 token ID 的整数大小。

![one-hot 是词表标准基，经 embedding 矩阵查表后变成低维分布式向量，并在几何空间中产生可学习的关系](/assets/text-representation/svg/one-hot-and-distributed.1.svg)

## 先分清 ID、one-hot 和 embedding

### 三个对象的接口不同

| 对象 | 形状 | 主要含义 | 是否有可解释的连续距离 |
| --- | --- | --- | --- |
| token 字符串 | 离散符号 | 词表定义的建模单位 | 没有 |
| token ID | 一个整数 | 词表中的索引 | 没有 |
| one-hot | $\mathbb R^V$ 中的稀疏向量 | 选择一个词表坐标 | 所有不同类别距离相同 |
| embedding | $\mathbb R^d$ 中的连续向量 | 可学习的分布式表示 | 可以由内积、余弦或距离定义 |

token ID 只是数组下标。若 cat 的 ID 是 3、dog 的 ID 是 4，不代表 cat 比 ID 为 20 的 fish 更接近 dog。把 ID 直接当作一个连续数送进线性层，会人为引入不存在的顺序关系。

### one-hot 是标准基，不是语义坐标

词表有 $V$ 个条目时，标准基满足

$$
\mathbf e_i
=
(0,\ldots,0,1,0,\ldots,0),
\qquad
\mathbf e_i\in\mathbb R^V,
$$

其中第 $i$ 个坐标为 1。它的内积为

$$
\mathbf e_i^\mathsf T\mathbf e_j
=
\delta_{ij}
=
\begin{cases}
1,&i=j,\\
0,&i\ne j.
\end{cases}
$$

这只表示“同一个类别得到 1，不同类别得到 0”。它不是一个由语言经验预先排列好的语义空间。

## One-hot 的几何性质

### 所有不同 token 等距

每个 one-hot 向量的范数都是 1：

$$
\lVert\mathbf e_i\rVert_2=1.
$$

当 $i\ne j$ 时，

$$
\begin{aligned}
\lVert\mathbf e_i-\mathbf e_j\rVert_2^2
&=
\lVert\mathbf e_i\rVert_2^2
+\lVert\mathbf e_j\rVert_2^2
-2\mathbf e_i^\mathsf T\mathbf e_j\\
&=1+1-0=2,
\end{aligned}
$$

所以所有不同类别之间的欧氏距离都是

$$
\lVert\mathbf e_i-\mathbf e_j\rVert_2=\sqrt2.
$$

它们之间的余弦相似度都是 0。one-hot 能可靠地区分类别，却不能表达“猫比汽车更接近狗”这样的 graded similarity。

### ID 的重排不会改变 one-hot 结构

设 $\pi$ 是一个词表 ID 的置换矩阵。把 cat 从 ID 3 换到 ID 10，会把 one-hot 从 $\mathbf e_3$ 换成 $\mathbf e_{10}$，但所有不同 token 仍然正交、等距。只要同时重排 embedding 的行，模型的函数可以保持不变：

$$
E'=P_\pi E,
\qquad
\mathbf e_{\pi(i)}^\mathsf T E'
=
\mathbf e_i^\mathsf T E.
$$

这说明 ID 编号本身没有语义。真正重要的是 tokenizer 的 token 到 ID 映射和 checkpoint 中 embedding 行的绑定必须一致。

## One-hot 如何变成 embedding

### 代数形式：选择矩阵的一行

令词表大小为 $V$、embedding 宽度为 $d$：

$$
E=
\begin{bmatrix}
E_{0,:}\\
E_{1,:}\\
\vdots\\
E_{V-1,:}
\end{bmatrix}
\in\mathbb R^{V\times d}.
$$

第 $i$ 个 token 的 one-hot 是 $\mathbf e_i^\mathsf T\in\mathbb R^{1\times V}$。相乘时只有第 $i$ 行被选中：

$$
\mathbf e_i^\mathsf T E
=
\begin{bmatrix}
0&\cdots&1&\cdots&0
\end{bmatrix}
E
=
E_{i,:}.
$$

所以 embedding lookup 不是一个与矩阵乘法不同的魔法操作；它是利用 one-hot 稀疏结构实现的行选择。显式构造 $V$ 维 one-hot 再做完整矩阵乘法会浪费计算，但数学结果相同。

### 一个四 token 的 toy 词表

假定词表是：

| token | ID | one-hot 标准基 |
| --- | ---: | --- |
| cat | 0 | $\mathbf e_0$ |
| dog | 1 | $\mathbf e_1$ |
| fish | 2 | $\mathbf e_2$ |
| UNK | 3 | $\mathbf e_3$ |

对应的 one-hot 向量是

$$
\mathbf e_{\text{cat}}=(1,0,0,0),
\quad
\mathbf e_{\text{dog}}=(0,1,0,0),
\quad
\mathbf e_{\text{fish}}=(0,0,1,0).
$$

设 embedding 矩阵为

$$
E=
\begin{bmatrix}
1&0\\
0.8&0.6\\
-1&0\\
0&1
\end{bmatrix}.
$$

那么

$$
\mathbf e_{\text{dog}}^\mathsf T E
=
(0,1,0,0)
\begin{bmatrix}
1&0\\
0.8&0.6\\
-1&0\\
0&1
\end{bmatrix}
=(0.8,0.6).
$$

one-hot 只负责选择第二行；$(0.8,0.6)$ 的数值和几何方向来自 $E$。

### 稀疏查表与完整乘法

批量 token 的 one-hot 矩阵记为 $X\in\mathbb R^{B\times V}$，则

$$
Z=XE\in\mathbb R^{B\times d}.
$$

每一行 $X_{b,:}$ 只有一个 1，因此 $Z$ 只是从 $E$ 选择 $B$ 行。显式 dense matmul 需要处理大量 0；实际实现直接用 ID 做 gather：

| 表达方式 | 逻辑形状 | 计算重点 | 常见实现 |
| --- | --- | --- | --- |
| one-hot 乘矩阵 | $B\times V$ 乘 $V\times d$ | $Vd$ 个位置中大部分为 0 | 教学推导 |
| sparse lookup | ID 列表到行 | 只读取需要的 $B$ 行 | embedding gather |
| tied output | embedding 与输出投影共享 | 同一 $E$ 服务输入和输出 | 语言模型常见选项 |

数学等价不代表内存和带宽成本相同。理解 one-hot 乘法有助于推导梯度，工程上通常不显式创建 one-hot。

## 分布式表示到底“分布”了什么

### 信息分布在多个坐标

在 one-hot 中，一个 token 的身份集中在一个坐标；在分布式向量

$$
\mathbf z=(z_1,z_2,\ldots,z_d)
$$

中，多个坐标共同参与表示。两个 token 可以共享某些方向，也可以在不同方向上有不同强度。一个坐标通常不等于一个固定语义，而是多个潜在因素的混合。

对 toy embedding：

$$
\mathbf z_{\text{cat}}=(1,0),
\qquad
\mathbf z_{\text{dog}}=(0.8,0.6),
\qquad
\mathbf z_{\text{fish}}=(-1,0).
$$

它们的点积和余弦可以产生有等级的关系：

$$
\mathbf z_{\text{cat}}^\mathsf T\mathbf z_{\text{dog}}=0.8,
\qquad
\cos(\mathbf z_{\text{cat}},\mathbf z_{\text{dog}})=0.8,
$$

而

$$
\mathbf z_{\text{cat}}^\mathsf T\mathbf z_{\text{fish}}=-1.
$$

cat 与 dog 的欧氏距离是

$$
\lVert\mathbf z_{\text{cat}}-\mathbf z_{\text{dog}}\rVert_2
=
\sqrt{(1-0.8)^2+(0-0.6)^2}
=\sqrt{0.4}
\approx0.632455532.
$$

这些关系不是因为 cat、dog 的 ID 相邻，而是因为我们给出的 $E$ 已经包含了某种几何结构。在真实模型中，这个结构由训练目标、数据和优化共同塑造。

### 分布式不等于自动有意义

随机初始化的 dense 向量也会有距离和余弦，但那不代表已经学到语言语义。要说 embedding 的方向有意义，至少需要说明：

1. 它由什么训练目标更新；
2. 比较的是哪个层、哪个 checkpoint；
3. 使用内积、余弦还是欧氏距离；
4. 是否做了归一化、中心化或其他变换；
5. 结论是否在新词、新域和不同上下文中稳定。

分布式表示提供了可学习的容量，不直接提供解释。

## Embedding 的梯度为什么是稀疏的

### 单个 token 只更新一行

设一个 token 的输入是 $\mathbf e_i^\mathsf T$，输出 embedding 为

$$
\mathbf z=\mathbf e_i^\mathsf T E.
$$

若上游梯度为

$$
\mathbf g=\frac{\partial\mathcal L}{\partial\mathbf z},
$$

则 embedding 矩阵的梯度是

$$
\frac{\partial\mathcal L}{\partial E}
=
\mathbf e_i\mathbf g^\mathsf T.
$$

因为 $\mathbf e_i$ 只有第 $i$ 个位置非零，所以

$$
\frac{\partial\mathcal L}{\partial E_{j,:}}
=
\begin{cases}
\mathbf g^\mathsf T,&j=i,\\
\mathbf 0^\mathsf T,&j\ne i.
\end{cases}
$$

一次 token lookup 不会直接更新其他词的 embedding 行。batch 中重复出现的 token 会把对应行的梯度累加；padding 行通常通过 mask 或冻结协议避免参与有效损失。

### 频率会影响更新次数

高频 token 在一个 batch 或整个训练周期中被 lookup 的次数更多，因此对应 embedding 行收到的更新机会更多。低频 token 可能很少被直接更新，subword、byte fallback 和共享上下文路径会影响它们如何获得统计信号。

这不是 one-hot 的缺陷，而是离散词表参数化的直接后果。词表设计和采样策略会影响不同 embedding 行的优化频率。

## One-hot 作为分类目标

### 输入 one-hot 与输出标签是同一个形式的两种用途

one-hot 不只可以表示输入 token，也可以表示分类目标。若正确类别是 $k$，目标分布是

$$
\mathbf y=\mathbf e_k\in\mathbb R^C,
$$

模型输出概率为 $\mathbf p$，交叉熵是

$$
\mathcal L
=
-\sum_{c=1}^{C}y_c\log p_c
=
-\log p_k.
$$

此时 one-hot 表示“监督目标选择第 $k$ 类”，而不是一个输入 embedding。输入侧的 one-hot 乘 $E$ 得到 dense vector；输出侧的 one-hot 与 $\log\mathbf p$ 做内积得到损失。

### label smoothing 不再是严格 one-hot

若用均匀分布 $\mathbf u=(1/C,\ldots,1/C)$ 做标签平滑：

$$
\mathbf y^{(\varepsilon)}
=
(1-\varepsilon)\mathbf e_k
+\varepsilon\mathbf u,
$$

目标的每个类别都可能有非零质量。它不再是严格的 one-hot，但仍然是类别分布。实现和评估时要明确目标是 hard one-hot、soft label 还是 multi-hot。

## One-hot、multi-hot 和序列矩阵

### 多标签不是 one-hot

one-hot 满足

$$
\mathbf x\in\{0,1\}^V,
\qquad
\sum_{i=1}^{V}x_i=1.
$$

如果一个样本可以同时属于多个独立标签，则可能是 multi-hot：

$$
\mathbf m\in\{0,1\}^V,
\qquad
\sum_{i=1}^{V}m_i>1.
$$

把 multi-hot 误称为 one-hot 会错误地暗示类别互斥。

### 一段文本是多个 one-hot 的序列

token 序列长度为 $L$ 时，可把每个位置的 one-hot 堆成

$$
X=
\begin{bmatrix}
\mathbf e_{x_1}^\mathsf T\\
\mathbf e_{x_2}^\mathsf T\\
\vdots\\
\mathbf e_{x_L}^\mathsf T
\end{bmatrix}
\in\mathbb R^{L\times V}.
$$

经过 embedding 后：

$$
Z=XE\in\mathbb R^{L\times d}.
$$

第 $\ell$ 行是 token $x_\ell$ 的 embedding。padding 会增加额外的 one-hot 行，但有效注意力和损失仍需由 mask 控制。

### multi-hot 的线性聚合不等于顺序表示

若把一袋 token 表示为 multi-hot 向量 $\mathbf m$，再计算

$$
\mathbf m^\mathsf T E
=
\sum_{i:m_i=1}E_{i,:},
$$

得到的是无序的加和。它可以形成 bag-of-embeddings，但无法区分 dog bites man 与 man bites dog。序列顺序需要位置索引、位置编码或循环/注意力结构额外提供。

## 表示空间中的可逆性和信息损失

### one-hot 到 embedding 通常不是可逆的

当 $d<V$ 时，矩阵 $E\in\mathbb R^{V\times d}$ 把 $V$ 个 one-hot 基向量映射到低维空间。若不同 token 的行恰好相同，就会立即丢失区分它们所需的信息；即使行不同，低维表示也不保证存在一个通用逆映射能恢复原始 ID。

one-hot 到 embedding 的映射是

$$
\phi:\{1,\ldots,V\}\longrightarrow\mathbb R^d,
\qquad
\phi(i)=E_{i,:}.
$$

模型不一定需要从 embedding 精确恢复 ID；它需要的是保留对任务有用的可分性和组合结构。压缩是代价，也是分布式表示能泛化的来源之一。

### 可分性取决于下游读出

若有一个线性读出矩阵 $W\in\mathbb R^{d\times V}$，输出 logits 为

$$
\boldsymbol\ell
=
\mathbf zW+\mathbf b.
$$

若 $E$ 的不同 token 行在 $W$ 下能产生不同 logits，模型仍然可以区分它们。若两行在所有后续计算中都相同，系统就无法依靠这条路径区分对应 token。

因此“embedding 维度小于词表”不等于“每个 token 都无法区分”；真正的区分能力取决于行结构、下游层和训练任务。

## 词表规模与表示成本

### one-hot 的坐标成本与 embedding 的参数成本

one-hot 的概念维度是 $V$，但它只有一个非零元素。embedding 的参数量是

$$
N_E=Vd.
$$

当 $V=50000,d=768$ 时，

$$
N_E
=50000\times768
=38400000.
$$

one-hot 本身不需要把所有坐标存成 dense 数组；真正昂贵的是可学习的 $V\times d$ 参数，以及训练中读写这些行的带宽。

| 设计选择 | 表示维度 | 主要成本 | 表达能力 |
| --- | ---: | --- | --- |
| one-hot | $V$ | 稀疏索引与分类坐标 | 区分身份，无相似结构 |
| 小 embedding | $d\ll V$ | $Vd$ 个参数 | 低维可学习几何 |
| 大 embedding | 更大的 $d$ | 参数和带宽增加 | 更高表示容量，未必更好 |
| multi-hot 聚合 | $V$ 的稀疏选择 | 聚合选中行 | 可表示集合，丢顺序 |

词表扩大、embedding 宽度增加、输出层是否绑定，都应放进同一个参数账本。

## 失效模式：看似只是换一种编码

### 1. 把 token ID 当作连续特征

ID 3 和 ID 4 的差为 1，并不代表它们比 ID 3 和 ID 20 更相似。应先 one-hot 或直接 lookup embedding，再交给模型。

### 2. 以为 one-hot 自带语义距离

所有不同 one-hot 向量的余弦相似度都是 0。语义或统计关系来自后续学习到的 $E$，不是来自标准基的坐标。

### 3. 词表重排但不重排 embedding

如果 tokenizer 把 token 到 ID 的映射重排，而 checkpoint 的 $E$ 行不跟着重排，模型会读取错误的向量；参数形状可能完全正常，错误却只表现为性能下降。

### 4. 把 embedding 行当成固定语义轴

坐标可以被旋转、缩放或与其他方向混合。两个 checkpoint 即使完成同一任务，单个坐标轴也不必逐维对齐。

### 5. 把随机 dense vector 当作已学语义

随机向量也有距离。只有结合训练目标、数据和稳定性检验，才能讨论 embedding 几何是否承载有用关系。

### 6. 把 multi-hot 当作 one-hot

多个 1 表示多个同时激活的标签或集合成员。若任务类别本应互斥，multi-hot 目标会改变损失与概率解释。

### 7. 用显式 dense one-hot 乘矩阵造成浪费

数学推导可以写 $XE$，实现通常直接用 ID gather。把 $V$ 维稀疏向量物化成 dense 数组，会增加内存和带宽而不增加表达能力。

### 8. 忽略序列位置

把多个 one-hot 或 embedding 直接求和会丢失顺序。语言模型需要位置编码、循环状态或注意力中的位置机制补回顺序信息。

### 9. 把输入 embedding 与输出标签混为一谈

输入 one-hot 选择 $E$ 的一行；输出 one-hot 参与交叉熵目标。两者形式相似，但梯度路径和语义不同。

## 最小审计协议

检查一个 token 表示接口时，可以沿下面的顺序复核：

1. 列出词表、token ID 和 special token 的固定顺序；
2. 选三个不同 token，验证 one-hot 的范数、内积和距离；
3. 用一个明确的 $E$ 手算 one-hot lookup 是否等于目标行；
4. 比较显式 $XE$ 与直接 gather 的数值结果；
5. 对一个单 token 损失检查 embedding 梯度是否只写入对应行；
6. 确认 padding、UNK 和 mask 的行是否按协议更新；
7. 区分 one-hot、soft label、multi-hot 和序列 one-hot 矩阵；
8. 用 ID 重排实验验证 tokenizer 与 embedding 是否同时绑定；
9. 若解释 embedding 几何，记录距离定义、归一化、层和 checkpoint；
10. 检查词表规模、embedding 宽度、输出层绑定和显存账本。

一个最小的数值契约是：

$$
\begin{gathered}
\mathbf e_{\text{dog}}=(0,1,0,0),\\
\mathbf e_{\text{dog}}^\mathsf T E=(0.8,0.6),\\
\lVert\mathbf e_{\text{cat}}-\mathbf e_{\text{dog}}\rVert_2=\sqrt2,\\
\cos(\mathbf z_{\text{cat}},\mathbf z_{\text{dog}})=0.8.
\end{gathered}
$$

它同时检查了离散身份、查表行、one-hot 几何和分布式空间的区别。

## 结语

one-hot 是一个清晰但无语义几何的离散接口：它把 token 变成词表标准基，所有不同类别正交且等距。embedding 则把这些标准基映射到较低维的连续空间，让训练可以通过共享方向、内积和组合结构学习分布式表示。

这一步的关键不是把一个整数“变成更复杂的数字”，而是把离散选择接入一个可学习的线性代数接口。ID 只负责索引，one-hot 负责选择，embedding 负责承载可训练几何；三者只在 tokenizer 与 checkpoint 的映射严格一致时才构成同一个模型输入。

## 相关词条

[Tokenization](../text-representation/tokenization/)

[WordPiece 与 SentencePiece](../text-representation/wordpiece-and-sentencepiece/)

[BPE 分词](../text-representation/tokenization-bpe/)

[嵌入](../text-representation/embeddings/)

[嵌入几何](../text-representation/embedding-geometry/)

[余弦相似度](../text-representation/cosine-similarity/)

[词表大小权衡](../text-representation/vocabulary-size-tradeoffs/)

[标签平滑](../training-nn/label-smoothing/)
