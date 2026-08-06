---
title: "Embeddings：让 token 在可学习空间中获得几何"
tags: ["why-models-learn"]
---

Embedding 是把离散对象映射到连续向量空间的可学习表示。对词表大小为 $V$ 的 tokenizer，最常见的 token embedding 是一个矩阵

$$
E_{\text{tok}}\in\mathbb R^{V\times d},
$$

第 $i$ 行是 token ID 为 $i$ 的向量。给定 token 序列 $(x_1,\ldots,x_L)$，模型取出

$$
\mathbf z_\ell=E_{\text{tok}}[x_\ell]\in\mathbb R^d.
$$

one-hot 词条说明了这一步在代数上等价于 $\mathbf e_{x_\ell}^{\mathsf T}E_{\text{tok}}$；这篇继续追问：矩阵的行为什么会包含可用的关系，它通过什么目标学习，什么时候它只是静态查表，什么时候会被上下文层改造成 contextual representation。

Embedding 的几何不是 token ID 自带的属性。它由训练数据、目标函数、优化过程、正则化、词表和 checkpoint 共同决定。相同 token 在不同模型中的向量不能直接比较，除非先说明坐标系、层、版本和对齐方式。

![token ID 经过 embedding 矩阵查表进入训练目标，得到静态向量；上下文层再把它们变成依赖句子的 hidden state](/assets/text-representation/svg/embeddings.1.svg)

## Embedding 矩阵是什么

### 行是 token，列是可学习坐标

设词表为

$$
\mathcal V=\{t_0,t_1,\ldots,t_{V-1}\}.
$$

embedding 矩阵可以写成

$$
E=
\begin{bmatrix}
\mathbf v_0^\mathsf T\\
\mathbf v_1^\mathsf T\\
\vdots\\
\mathbf v_{V-1}^\mathsf T
\end{bmatrix}
\in\mathbb R^{V\times d},
\qquad
\mathbf v_i\in\mathbb R^d.
$$

这里的第 $i$ 行是 token $t_i$ 的输入表示。列坐标通常没有预先命名的语义；一个维度可能同时承载多个统计因素，多个维度也可能共同表示一个因素。

| 对象 | 形状 | 作用 |
| --- | --- | --- |
| vocabulary | $V$ 个 token | 定义可以被查找的离散条目 |
| token ID | 一个整数 $i$ | 选择 embedding 的第 $i$ 行 |
| embedding matrix | $V\times d$ | 保存所有 token 的可学习向量 |
| lookup result | 一个 $d$ 维向量 | 作为后续网络的输入 |

只要 token 到 ID 的映射和矩阵行顺序一致，ID 的数值大小就没有几何含义。

### lookup 是稀疏线性变换

若 token $t_i$ 的 one-hot 向量为 $\mathbf e_i$，则

$$
\mathbf e_i^\mathsf T E
=
\mathbf v_i^\mathsf T.
$$

对长度为 $L$ 的序列，把 one-hot 行堆成 $X\in\mathbb R^{L\times V}$：

$$
Z=XE\in\mathbb R^{L\times d}.
$$

实际实现不需要物化 $X$，直接用 ID gather $E$ 的行即可。矩阵乘法是推导视角，gather 是计算视角。

## Embedding 如何被学习

### 监督学习只要求表示有用

在一个监督任务中，embedding 通常和下游网络一起优化：

$$
\theta
=
(E,\theta_{\text{rest}}),
\qquad
\min_\theta
\mathbb E_{(x,y)}
\big[
\mathcal L(f_\theta(x),y)
\big].
$$

只要最终损失下降，某个 token 的向量就会被梯度推向能帮助任务预测的位置。这个位置不一定对应人类词典中的语义，也不一定能跨任务复用。

### 分布式假设：相似上下文产生相似向量

在无监督或自监督语言表示中，常见直觉是：出现在相似上下文中的 token 可以共享表示。这个想法不是“相似 token 必须相邻”的硬约束，而是训练目标通过共同预测任务间接产生的统计压力。

词向量可能编码：

1. 词义或主题；
2. 句法角色和形态；
3. 词频与领域；
4. 任务标签相关的偏差；
5. 训练数据中的社会或事实关联。

因此 embedding 近邻是模型目标下的统计关系，不自动等于同义词关系。

## Skip-gram：用中心 token 预测上下文

### 正样本目标

设中心 token 为 $w$，上下文 token 为 $c$，中心向量为 $\mathbf v_w$，上下文输出向量为 $\mathbf u_c$。一个二分类式正样本得分是

$$
s(w,c)=\mathbf v_w^\mathsf T\mathbf u_c.
$$

希望正样本的 sigmoid 接近 1：

$$
\mathcal L_{+}(w,c)
=
-\log\sigma\big(\mathbf v_w^\mathsf T\mathbf u_c\big).
$$

这里通常需要两套矩阵：

| 矩阵 | 行的含义 | 训练角色 |
| --- | --- | --- |
| $V_{\text{in}}$ | 中心 token 向量 $\mathbf v_w$ | 作为条件输入 |
| $U_{\text{out}}$ | 上下文 token 向量 $\mathbf u_c$ | 被预测的目标 |

训练完成后，使用哪一套矩阵、两套矩阵是否平均或拼接，都是模型定义的一部分。

### 负采样

用完整 softmax 计算所有词表目标的代价可能很高。负采样给一个正 pair 配 $K$ 个负 token $n_1,\ldots,n_K$：

$$
\mathcal L_{\text{NS}}
=
-\log\sigma(\mathbf v_w^\mathsf T\mathbf u_c)
-\sum_{k=1}^{K}
\log\sigma(-\mathbf v_w^\mathsf T\mathbf u_{n_k}).
$$

正 pair 被推向较大的点积，负 pair 被推向较小的点积。这个目标塑造的是“区分真实上下文与采样噪声”的几何，而不是直接拟合一个人类标注的距离矩阵。

### 一个数值例子

设

$$
\mathbf v_{\text{cat}}=(1,0),
\quad
\mathbf u_{\text{dog}}=(0.8,0.2),
\quad
\mathbf u_{\text{car}}=(-0.2,0.6).
$$

dog 是正上下文，car 是一个负样本。两个得分是

$$
s_{+}=0.8,
\qquad
s_{-}=-0.2.
$$

负采样损失为

$$
\mathcal L_{\text{NS}}
=
-\log\sigma(0.8)
-\log\sigma(0.2)
\approx
0.969239535.
$$

对应的 score 梯度为

$$
\frac{\partial\mathcal L}{\partial s_+}
=\sigma(0.8)-1
\approx-0.310025517,
\qquad
\frac{\partial\mathcal L}{\partial s_-}
=\sigma(-0.2)
\approx0.450166003.
$$

梯度会增大正 pair 的得分，并降低负 pair 的得分。向量如何移动还取决于点积中的另一侧向量和其他 batch 样本。

## CBOW：用上下文预测中心 token

### 平均上下文向量

CBOW 把多个上下文 token 的向量聚合后预测中心词。对上下文集合 $C(w)$：

$$
\mathbf h_C
=
\frac{1}{\lvert C(w)\rvert}
\sum_{c\in C(w)}\mathbf v_c.
$$

随后用 $\mathbf h_C$ 预测中心 token。平均操作计算简单，但会丢掉上下文 token 的顺序和重复位置；具体实现可以用加权或其他聚合改进。

### Skip-gram 与 CBOW 的统计方向

| 方法 | 条件 | 预测目标 | 常见取舍 |
| --- | --- | --- | --- |
| Skip-gram | 一个中心 token | 多个上下文 token | 低频词可以获得较多局部监督，计算样本数多 |
| CBOW | 多个上下文 token | 一个中心 token | 训练更快，聚合会丢部分局部信息 |

两者都可以产生静态 embedding，但训练出的坐标不必一致。模型名字不能单独决定向量的语义。

## 从共现计数到低秩表示

### 计数矩阵是另一种表示

若统计词与上下文的共现次数，可以得到

$$
C\in\mathbb R_{\ge0}^{V\times V},
\qquad
C_{ij}=\text{token }t_i\text{ 与上下文 }t_j\text{ 的共现计数}.
$$

计数向量的维度可能等于词表大小，且非常稀疏。embedding 用低维矩阵近似这类关系：

$$
C\approx V_{\text{in}}U_{\text{out}}^\mathsf T.
$$

这不是说所有 embedding 训练都等价于一次 SVD，而是说明 embedding 可以被理解为一种低秩、可优化、带目标函数的共现结构压缩。

### 低秩带来共享与压缩

若 $d\ll V$，两个 token 的向量可以在相同方向上共享统计信号。新 token 不需要为每个上下文单独存一列计数，只要在低维空间中组合已有方向。

代价是信息损失。两个在任务上不同的 token 可能被压到相近位置；模型是否能区分它们，还要看上下文层和下游读出。

## 静态 embedding 与 contextual representation

### 静态 embedding 只看 token ID

静态 embedding 是一个函数

$$
\phi_{\text{static}}(t_i)=E_{i,:}.
$$

同一个 token 无论出现在什么句子、什么位置，查表结果相同。它可以表达跨句复用的词表统计，但无法单独区分一词多义：

$$
\phi_{\text{static}}(\text{bank})
\quad\text{在不同上下文中不变}.
$$

### 上下文表示依赖整段输入

经过上下文层 $F$ 后，第 $\ell$ 个位置的 hidden state 可以写成

$$
\mathbf h_\ell
=
F\big(
\mathbf z_1,\ldots,\mathbf z_L;
\text{position},\text{mask}
\big)_\ell.
$$

此时两个相同 token 的 $\mathbf h_\ell$ 可以因为邻近词、位置、因果 mask 和层状态不同而不同。Embedding 是上下文计算的起点，不等于最终 hidden state。

| 表示 | 依赖因素 | 同一 token 是否必然相同 | 典型用途 |
| --- | --- | --- | --- |
| token embedding | token ID | 是 | 输入查表 |
| position embedding | 位置 ID | 对同一位置相同 | 注入顺序 |
| contextual hidden state | token、上下文、位置、mask、层 | 否 | 预测、分类、检索 |
| pooled representation | 多个 hidden state 的聚合 | 否 | 句子或文档级任务 |

把上下文层输出误称为“词向量表中的那一行”，会混淆静态参数与运行时状态。

## 多张 embedding 表如何相加

### token、position 和 type embedding

一些架构将多个同维度表相加：

$$
\mathbf h_\ell^{(0)}
=
E_{\text{tok}}[x_\ell]
+E_{\text{pos}}[\ell]
+E_{\text{type}}[r_\ell].
$$

其中

$$
E_{\text{tok}}\in\mathbb R^{V\times d},
\quad
E_{\text{pos}}\in\mathbb R^{L_{\max}\times d},
\quad
E_{\text{type}}\in\mathbb R^{R\times d}.
$$

相加要求三个向量维度相同。每一张表承担的索引不同：

| 表 | 索引 | 表达的信息 |
| --- | --- | --- |
| token embedding | token ID | 离散词表单位 |
| position embedding | 位置 $0,\ldots,L_{\max}-1$ | 序列顺序 |
| type/segment embedding | segment ID | 输入段或角色 |

并非所有 Transformer 都使用三张可学习表。有的架构用固定位置编码、旋转位置机制或其他方式；不能从“输入是 embedding”推断具体位置方案。

### 相加会混合来源

输入层的和把 token、位置和 segment 信息送进同一向量：

$$
\mathbf h_\ell^{(0)}
\in\mathbb R^d.
$$

后续层通常不能仅靠某个坐标轴把三种来源完美分离。分析表示时，应同时保存各表参数和组合顺序，而不是只导出相加后的结果。

## 输入与输出 embedding 是否共享

### tied weights

语言模型可以用输入 embedding 矩阵的转置作为输出投影：

$$
\boldsymbol\ell_t
=
\mathbf h_tE_{\text{tok}}^\mathsf T+\mathbf b.
$$

这样 token 输入与输出词表共享参数。若不共享，则另有

$$
W_{\text{out}}\in\mathbb R^{d\times V}.
$$

| 方案 | 参数结构 | 影响 |
| --- | --- | --- |
| untied | 输入 $E_{\text{tok}}$ 与输出 $W_{\text{out}}$ 独立 | 参数更多，输入输出几何可分开 |
| tied | $W_{\text{out}}=E_{\text{tok}}^\mathsf T$ | 参数更少，输入与输出共享词表方向 |

权重绑定不是“embedding 天然应该转置输出”的数学必然，而是一个架构选择。词表变化时，两侧接口都要一起检查。

## Embedding 的几何为什么不唯一

### 正交变换保留内积

设 $Q\in\mathbb R^{d\times d}$ 满足

$$
Q^\mathsf TQ=I.
$$

把所有 embedding 行右乘 $Q$：

$$
E'=EQ.
$$

对任意两行 $\mathbf v_i,\mathbf v_j$：

$$
(\mathbf v_iQ)(\mathbf v_jQ)^\mathsf T
=
\mathbf v_iQQ^\mathsf T\mathbf v_j^\mathsf T
=
\mathbf v_i\mathbf v_j^\mathsf T.
$$

因此旋转、反射等正交变换可以保留内积和距离。两个模型完成同一目标时，坐标轴和单个坐标值不必对齐；直接逐维比较可能没有意义。

### 几何比较需要先固定口径

比较两个 embedding 空间时，需要说明：

1. 使用输入表、输出表还是某一层 hidden state；
2. token 词表是否相同，ID 是否对应；
3. 使用内积、余弦还是欧氏距离；
4. 是否做中心化、归一化或 Procrustes 对齐；
5. 是否只比较共享 token，如何处理 OOV；
6. checkpoint、训练数据和 tokenizer 是否一致。

embedding-geometry 词条会继续讨论这些距离和变换；这里先固定一个边界：向量坐标的绝对方向不是模型语义的唯一证据。

## 频率、稀疏梯度和低频 token

### 一次 lookup 只更新被选中的行

设第 $i$ 行查表结果为

$$
\mathbf z=\mathbf e_i^\mathsf T E,
$$

上游梯度为 $\mathbf g$，则

$$
\frac{\partial\mathcal L}{\partial E}
=
\mathbf e_i\mathbf g^\mathsf T.
$$

只有第 $i$ 行直接收到这次梯度。高频 token 会获得更多直接更新；低频 token 可能通过 subword、共享上下文和下游任务得到间接统计信号，但不能假定它们的训练覆盖相同。

### 频率不是质量

高频 token 的向量通常拥有更多样本支持，但也可能编码领域频率、功能词模式或数据偏差。低频 token 的向量方差可能更大，也可能因为与常见片段共享结构而表现稳定。需要用独立任务或固定 probe 评估，而不是用出现次数直接推断表示质量。

## OOV、subword 和 embedding 行

### 词表覆盖决定可学习粒度

整词 embedding 对已登录词只需一次 lookup，但新词可能变成 UNK。subword 或 byte tokenizer 让一个词拆成多个已知 token，每个 token 查自己的行，再交给上下文层组合：

$$
\text{字符串}
\longrightarrow
(t_1,\ldots,t_k)
\longrightarrow
\big(E[t_1],\ldots,E[t_k]\big).
$$

这会改变序列长度和梯度分配。新词没有单独一行，不等于模型无法处理它；模型可能复用其子词行，但组合质量依赖上下文层和训练覆盖。

| tokenizer 单位 | 词表行语义 | 新词处理 | 主要影响 |
| --- | --- | --- | --- |
| 整词 | 词形整体 | UNK 或扩词表 | 序列短，覆盖脆弱 |
| 子词 | 可复用片段 | 多行 lookup | 复用与长度折中 |
| byte | byte 或 byte merge | 任意输入可回退 | 覆盖强，长度可能增加 |

Embedding 质量不能脱离 tokenizer 讨论。

## 词表大小与参数账本

### 参数量随 $Vd$ 增长

若词表大小为 $V$、宽度为 $d$：

$$
N_E=Vd.
$$

例如 $V=50000,d=768$：

$$
N_E=50000\times768=38400000.
$$

若输入输出权重不共享，还要额外考虑输出投影 $d\times V$。增大词表可以减少 token 长度，却会增加参数、初始化、checkpoint 大小和显存带宽。

### 序列长度也产生成本

若 tokenizer 使序列长度从 $L$ 变为 $2L$，attention score map 的元素数从 $L^2$ 变为

$$
(2L)^2=4L^2.
$$

词表、embedding 宽度和上下文长度应放进同一系统账本。只比较 embedding 参数或只比较 token 数都会遗漏一半成本。

## 失效模式：向量存在不等于接口正确

### 1. 只保存 embedding 矩阵，遗漏 tokenizer

矩阵第 42 行只有在 token ID 映射也一致时才有意义。缺少 normalization、词表或 special-token 协议，无法从原始字符串可靠地复现 lookup。

### 2. 混用行向量和列向量约定

有的实现写 $\mathbf e_i^\mathsf T E$，有的实现写 $E\mathbf e_i$。两者可以等价，但矩阵形状、转置和输出投影必须同时检查。

### 3. 把静态 embedding 当作 contextual state

同一个 token 的查表行固定不变；上下文层输出会随句子和位置变化。两者的采样、缓存和可解释性都不同。

### 4. 直接逐维比较两个 checkpoint

正交旋转就足以改变每个坐标，同时保留所有内积。比较前需要共享 token、距离口径和对齐方式。

### 5. 只看 cosine，不看范数和频率

余弦丢弃长度信息。某些任务中范数也携带频率或置信度；只看角度可能隐藏实际差异。

### 6. 假定 input 与 output embedding 一定共享

权重绑定是可选架构。读取 checkpoint 时必须确认是否存在独立 $W_{\text{out}}$。

### 7. 把近邻当作因果解释

embedding 中的近邻说明训练目标下的统计相似性，不能单独证明模型在某次预测中使用了某个概念或因果关系。

### 8. 忽略低频行与 padding 行

低频 token 的更新次数、padding 的 mask、UNK 的聚合方式都会影响行的统计。审计时要按 token 频次分桶查看，而不是只看平均指标。

## 最小审计协议

检查 embedding 接口时，可以固定下面的证据：

1. 记录 tokenizer、词表、ID 顺序和 checkpoint 指纹；
2. 随机抽取一个 token，手算 one-hot lookup 与矩阵行；
3. 对一个 batch 检查输入序列的行索引、padding 和 mask；
4. 用一个正 pair 和一个负 pair 复算训练目标或下游 loss；
5. 检查梯度是否只更新被查找的行；
6. 标出 token、position、type 和 output 表的形状；
7. 确认权重是否 tied；
8. 分开报告静态行向量和上下文层 hidden state；
9. 用共享 token、明确距离和对齐方法做几何比较；
10. 把词表参数、序列长度、显存和吞吐放入同一账本。

一个可回放的 toy 契约是：

$$
\begin{gathered}
\mathbf v_{\text{cat}}=(1,0),
\quad
\mathbf u_{\text{dog}}=(0.8,0.2),
\quad
\mathbf u_{\text{car}}=(-0.2,0.6),\\
s_+=0.8,
\quad
s_-=-0.2,
\quad
\mathcal L_{\text{NS}}\approx0.969239535.
\end{gathered}
$$

它能检查向量方向、正负样本符号、sigmoid 目标和损失数值是否按同一约定计算。

## 结语

Embedding 是离散词表进入连续模型的可学习接口。lookup 只选择一行；训练目标决定这些行如何共享统计信号；上下文层再把静态 token 表示改造成依赖句子、位置和 mask 的运行时状态。

Embedding 的价值来自可学习几何，但几何需要完整协议才能解释：tokenizer 决定行索引，训练目标决定更新方向，词表和 checkpoint 决定参数绑定，距离和对齐决定比较口径。脱离这些条件，单个坐标、近邻或 ID 数字都不足以说明模型学到了什么。

## 相关词条

[One-hot 与分布式表示](../text-representation/one-hot-and-distributed/)

[Tokenization](../text-representation/tokenization/)

[BPE 分词](../text-representation/tokenization-bpe/)

[WordPiece 与 SentencePiece](../text-representation/wordpiece-and-sentencepiece/)

[嵌入几何](../text-representation/embedding-geometry/)

[余弦相似度](../text-representation/cosine-similarity/)

[词表大小权衡](../text-representation/vocabulary-size-tradeoffs/)

[自注意力](../attention/self-attention/)
