---
title: "Embedding 几何：长度、方向与邻域如何被读出"
tags: ["why-models-learn"]
---

Embedding 矩阵的每一行可以看成连续空间中的一个点。模型训练不会给这个空间预先附上一套人类命名的坐标轴；训练数据、目标函数、参数化方式和优化过程共同决定哪些 token 会接近、哪些方向会分开。因此，读取 embedding 几何时必须同时说明对象、层、坐标系、距离和归一化方式。

这一篇建立一个最小的几何接口：长度回答向量有多大，点积回答两个向量的方向与尺度如何共同作用，欧氏距离回答两个点相隔多远，余弦相似度回答两条非零方向有多一致。之后再把这些局部量组织成邻域、Gram 矩阵和距离矩阵，并说明坐标变换、各向异性与 contextual 表示会如何改变解释。

![Embedding 行向量作为几何点，长度和方向共同决定点积、距离与邻域；坐标旋转可以保留关系但改变坐标值](/assets/text-representation/svg/embedding-geometry.1.svg)

## 先固定可比较对象

### token 行、token 出现位置和 hidden state 不是同一个对象

静态 token embedding 是词表中的一行。若词表大小为 $V$、embedding 宽度为 $d$，则

$$
E_{\text{tok}}\in\mathbb R^{V\times d},
\qquad
\mathbf v_t=E_{\text{tok}}[t]\in\mathbb R^d.
$$

这里的 $t$ 是 token ID。相同 token 在不同句子中查到同一行。经过上下文层后，第 $\ell$ 层、位置 $j$ 的 hidden state 可以写成

$$
\mathbf h_{j}^{(\ell)}
=F_\ell\big(
\mathbf z_{1:L},
\mathbf p_{1:L},
\mathbf m_{1:L}
\big)_j,
$$

其中 $\mathbf z$ 是输入 embedding，$\mathbf p$ 可以包含位置表示，$\mathbf m$ 表示 mask 或其他结构约束。同一个 token 在不同上下文中的 $\mathbf h_j^{(\ell)}$ 通常不同。

| 比较对象 | 一个对象对应什么 | 同一 token 是否固定 | 典型几何问题 |
| --- | --- | --- | --- |
| token embedding 行 | 词表中的一个 ID | 是 | 哪些词表条目在训练目标下接近 |
| token 出现位置 | 某个句子中的一次 token | 由静态行决定 | 输入表示如何进入上下文计算 |
| contextual hidden state | 某层某位置的运行时向量 | 否 | 同一 token 如何随上下文改变 |
| 句子或文档向量 | 多个位置的聚合结果 | 取决于聚合 | 哪些样本在选定任务下接近 |

如果把静态行、不同层的 hidden state 和句子平均向量放入同一个最近邻表，结果无法说明单一机制。几何审计的第一步是先选定对象。

### 坐标、checkpoint 和 tokenization 必须同时固定

一个可复现的比较至少需要记录以下字段：

| 字段 | 需要固定的内容 | 未固定时的影响 |
| --- | --- | --- |
| tokenizer | 词表、normalization、special token 和 ID 顺序 | 同一个字符串可能对应不同的行 |
| checkpoint | 参数版本、训练步数和来源 | 同一 token 的坐标会改变 |
| 表示位置 | 输入 embedding、输出 embedding、层号和 pooling | 比较的对象不再相同 |
| 坐标约定 | 行向量或列向量、是否转置 | lookup 和距离公式可能读错 |
| metric | 点积、欧氏距离、余弦或其他度量 | 近邻排序可能改变 |
| preprocessing | 是否中心化、归一化、去公共方向 | 长度和方向的贡献会改变 |

同一组向量在固定坐标系下可以被复算。两个 checkpoint 即使来自相同架构，也不能因为维度相同就直接逐坐标比较。

## 把向量读成几何量

### 长度只描述尺度

对 $\mathbf x\in\mathbb R^d$，欧氏长度是

$$
\lVert\mathbf x\rVert_2
=
\sqrt{\sum_{k=1}^{d}x_k^2}.
$$

长度较大的行可能在某个训练目标中承担更强的 logit 或梯度作用，也可能只反映频率、正则化和参数化差异。长度本身不等于语义强度，也不能单独证明某个 token 更重要。

零向量没有方向。后文使用余弦相似度时，必须先检查分母对应的两个向量是否非零。

### 点积同时包含尺度和方向

两个行向量的点积为

$$
\mathbf x\mathbf y^{\mathsf T}
=
\sum_{k=1}^{d}x_ky_k.
$$

当两个向量都非零时，它也可以写成

$$
\mathbf x\mathbf y^{\mathsf T}
=
\lVert\mathbf x\rVert_2
\lVert\mathbf y\rVert_2
\cos\theta,
$$

其中 $\theta$ 是两个向量的夹角。这个分解说明：点积变大，可能来自方向更一致，也可能来自任一向量长度变大。

在 Skip-gram、输出投影或 attention score 中，点积的尺度通常属于目标函数的一部分。把点积直接当作纯方向相似度，会丢掉长度贡献。

### 欧氏距离描述点的位置差

两个向量作为空间中的点时，它们的欧氏距离是

$$
d_2(\mathbf x,\mathbf y)
=
\lVert\mathbf x-\mathbf y\rVert_2.
$$

平方距离展开为

$$
\begin{aligned}
d_2^2(\mathbf x,\mathbf y)
&=
\lVert\mathbf x\rVert_2^2
+\lVert\mathbf y\rVert_2^2
-2\mathbf x\mathbf y^{\mathsf T}.
\end{aligned}
$$

因此，欧氏距离也同时受长度和方向影响。只有在所有向量都已经归一化时，它才与余弦相似度存在简单的单调关系。

| 几何量 | 主要保留的信息 | 对统一缩放的响应 | 适合回答的问题 |
| --- | --- | --- | --- |
| $\lVert\mathbf x\rVert_2$ | 单个向量的尺度 | 线性改变 | 这行的参数幅度是多少 |
| $\mathbf x\mathbf y^{\mathsf T}$ | 尺度与方向的乘积 | 按两个缩放因子改变 | 两个向量的 score 有多大 |
| $d_2(\mathbf x,\mathbf y)$ | 两个点的绝对位置差 | 会改变 | 两个点在当前坐标度量下相隔多远 |
| $\cos(\mathbf x,\mathbf y)$ | 非零向量的方向 | 正缩放不变 | 两个方向有多一致 |

## 一个四点 toy 空间

### 先看长度、点积和距离的分工

设四个 token 的二维 embedding 为

$$
\mathbf v_{\text{cat}}=(1,0.5),
\quad
\mathbf v_{\text{dog}}=(0.8,0.6),
\quad
\mathbf v_{\text{car}}=(-0.2,0.6),
\quad
\mathbf v_{\text{fish}}=(0,-1).
$$

它们的长度如下：

| token | 向量 | $\lVert\mathbf v\rVert_2$ | 解释 |
| --- | --- | ---: | --- |
| cat | $(1,0.5)$ | $1.118034$ | 尺度大于 1 |
| dog | $(0.8,0.6)$ | $1$ | 作为单位向量 |
| car | $(-0.2,0.6)$ | $0.632456$ | 尺度较小 |
| fish | $(0,-1)$ | $1$ | 与 dog 等长 |

cat 和 dog 的点积为 $1.1$，同时距离只有 $\sqrt{0.05}\approx0.223607$。cat 和 car 的点积只有 $0.1$，距离却为 $\sqrt{1.45}\approx1.204159$。这两个比较分别说明点积的尺度与方向贡献不能脱离距离解释。

### 同一批点可以产生不同的近邻排序

下面只列出三组 pair：

| pair | 点积 | 平方欧氏距离 | 欧氏距离 | 余弦相似度 |
| --- | ---: | ---: | ---: | ---: |
| cat, dog | $1.1$ | $0.05$ | $0.223607$ | $0.983870$ |
| cat, car | $0.1$ | $1.45$ | $1.204159$ | $0.141421$ |
| dog, car | $0.2$ | $1$ | $1$ | $0.316228 |

在这个 toy 中，cat 的最近欧氏邻居和最高余弦邻居都是 dog。可是如果给 car 乘一个更大的正标量，点积排序可能改变，而余弦排序不会改变。指标选择不是展示层的小细节；它决定“近邻”这个结论。

### 归一化把距离变成方向比较

对非零向量定义单位化版本

$$
\widehat{\mathbf x}
=
\frac{\mathbf x}{\lVert\mathbf x\rVert_2}.
$$

两个单位向量的平方距离为

$$
\begin{aligned}
\lVert\widehat{\mathbf x}-\widehat{\mathbf y}\rVert_2^2
&=
\lVert\widehat{\mathbf x}\rVert_2^2
+\lVert\widehat{\mathbf y}\rVert_2^2
-2\widehat{\mathbf x}\widehat{\mathbf y}^{\mathsf T}\\
&=
2-2\cos(\mathbf x,\mathbf y).
\end{aligned}
$$

所以，在先单位化的前提下，最小欧氏距离和最大余弦相似度给出相同的排序。没有单位化时，这个等价关系不成立。

## 从 pairwise 量到邻域

### 最近邻是一个条件结果

给定一组候选向量 $\{\mathbf v_i\}_{i=1}^{n}$，对查询向量 $\mathbf v_q$，欧氏距离下的 top-$k$ 邻域可以写成

$$
N_k(q)
=
\operatorname{arg\,topk}_{i\ne q}
\big(-d_2(\mathbf v_q,\mathbf v_i)\big).
$$

如果换成余弦相似度，排序对象变为 $\cos(\mathbf v_q,\mathbf v_i)$。同一批向量可以有两套不同的邻域。

近邻只表示在指定数据集、表示层和指标下关系较近。它不能单独推出同义、因果、可替换性或模型在某次生成中实际使用了这条关系。

### 局部结构和全局结构要分开

| 观察层级 | 计算方式 | 能说明什么 | 不能直接说明什么 |
| --- | --- | --- | --- |
| 单个 pair | 一个点积、距离或余弦 | 两个向量在当前口径下的关系 | 整个词表的结构 |
| top-$k$ 邻域 | 对一个查询排序 | 局部近邻 | 近邻的语义原因 |
| 聚类或连通图 | 汇总多点关系 | 局部群组或连通性 | 坐标轴的人类含义 |
| 全局统计 | 范数、平均余弦、谱或密度 | 空间整体形状 | 单个样本的因果解释 |

一个近邻如果在重采样、不同指标和不同层中都稳定，确认程度更高。只在一次排序中出现的邻居应视为待核查观察。

## 用 Gram 矩阵组织几何

### 点积矩阵保存所有两两内积

把 $n$ 个行向量堆成 $X\in\mathbb R^{n\times d}$，Gram 矩阵为

$$
G=XX^{\mathsf T}\in\mathbb R^{n\times n},
\qquad
G_{ij}=\mathbf x_i\mathbf x_j^{\mathsf T}.
$$

对角线是各向量长度的平方：

$$
G_{ii}=\lVert\mathbf x_i\rVert_2^2.
$$

如果只需要点积或余弦，$G$ 可以作为统一的中间结果。余弦矩阵还需要用对角线的长度做归一化。

### 距离矩阵可以由 Gram 矩阵恢复

平方距离矩阵满足

$$
D^{(2)}_{ij}
=
G_{ii}+G_{jj}-2G_{ij}.
$$

这条关系把“点的位置差”和“点积的尺度与方向”连接起来。它也给出一个实现核对：先直接计算 pairwise distance，再用 Gram 矩阵恢复，两者应在浮点误差范围内一致。

### 矩阵形状是审计证据

| 中间量 | 形状 | 每个元素的含义 | 常见错误 |
| --- | --- | --- | --- |
| embedding 子矩阵 $X$ | $n\times d$ | 一个选定 token 的一行 | 把 token 数和宽度互换 |
| Gram 矩阵 $G$ | $n\times n$ | 一对 token 的点积 | 忘记转置或混入 batch 轴 |
| 距离矩阵 $D$ | $n\times n$ | 一对 token 的距离 | 把平方距离当作距离 |
| top-$k$ 索引 | $n\times k$ | 每个查询的候选位置 | 包含自身或越过 mask |

在大词表上不需要一次性物化完整 $V\times V$ 矩阵。按查询 batch 分块计算，或者先用 ANN 产生候选，再精确重排，都必须保留候选范围和距离口径。

## 坐标变换、对齐和可比性

### 正交变换会改变坐标，但保留几何关系

设行向量矩阵 $X$ 右乘一个正交矩阵 $Q$：

$$
X'=XQ,
\qquad
Q Q^{\mathsf T}=I.
$$

对任意两行 $\mathbf x,\mathbf y$：

$$
(\mathbf xQ)(\mathbf yQ)^{\mathsf T}
=
\mathbf xQQ^{\mathsf T}\mathbf y^{\mathsf T}
=
\mathbf x\mathbf y^{\mathsf T}.
$$

因此，正交旋转或反射可以改变每个坐标值，却保留点积、长度、欧氏距离和余弦。单个坐标的大小不能作为跨坐标系的语义证据。

### 一般线性变换会改变度量

若右乘一般矩阵 $A$，则

$$
\lVert(\mathbf x-\mathbf y)A\rVert_2^2
=
(\mathbf x-\mathbf y)
AA^{\mathsf T}
(\mathbf x-\mathbf y)^{\mathsf T}.
$$

此时隐含的度量矩阵是 $AA^{\mathsf T}$。不同方向被不同程度拉伸后，距离与近邻可能改变。若比较两个未经对齐的 checkpoint，不能假定它们只相差一个正交变换；训练和参数化可能引入更一般的差异。

### 词表 ID 置换和坐标旋转是两种不同变化

词表 ID 置换作用在行：

$$
E'=PE,
$$

其中 $P$ 是置换矩阵。只要 token 与矩阵行同步重排，token 之间的对应几何关系不变。坐标变换作用在列：

$$
E'=EQ.
$$

前者改变“哪一行对应哪个 token”，后者改变“同一行使用哪套坐标”。审计时必须分别检查。

## 几何不是天然各向同性

### 公共方向会压缩有效差异

如果很多向量都带有相近的公共分量 $\mathbf c$，可以写成

$$
\mathbf v_i=\mathbf s_i+\mathbf c,
$$

其中 $\mathbf s_i$ 是个体差异。即使 $\mathbf s_i$ 之间差别明显，公共方向也可能让任意两行的点积偏大，导致平均余弦偏高。

用候选集合的均值

$$
\boldsymbol\mu
=
\frac{1}{n}\sum_{i=1}^{n}\mathbf v_i
$$

做中心化：

$$
\widetilde{\mathbf v}_i
=
\mathbf v_i-\boldsymbol\mu.
$$

中心化会改变距离、点积和近邻。它不是一个无条件正确的修复，而是一种需要与原始几何并列报告的变换。

### 一个可报告的各向异性指标

对 $n$ 个非零向量，可以报告平均 pairwise cosine：

$$
A
=
\frac{2}{n(n-1)}
\sum_{1\le i<j\le n}
\cos(\mathbf v_i,\mathbf v_j).
$$

$A$ 较高可能表示公共方向较强，也可能来自候选集合的真实同质性。只有在固定采样协议并与随机或分层基线比较时，才能把它当作空间统计证据。

### 高维随机基线不能被误读为语义信号

对均匀分布在单位球面的独立随机向量，余弦的期望为 0，维度较高时其方差量级约为 $1/d$：

$$
\mathbb E[\cos(\mathbf x,\mathbf y)]=0,
\qquad
\operatorname{Var}[\cos(\mathbf x,\mathbf y)]
\approx\frac{1}{d}.
$$

因此高维空间中许多无关 pair 的余弦会集中在 0 附近。一个略高于 0 的数值需要和随机、频率分层或标签置换基线比较，不能单独当作稳定语义关系。

## Hubness、频率与上下文

### 一个点成为很多查询的邻居

在 top-$k$ 邻域中，定义 token $i$ 的 hub count 为

$$
h_k(i)
=
\sum_{q=1}^{n}
\mathbf 1\big[i\in N_k(q)\big].
$$

如果少数 token 的 $h_k(i)$ 远高于其余 token，它们可能是 hub。hubness 可能与维度、范数、各向异性、频率或指标有关。它是邻域分布的观察，不是某个单一机制的证明。

### 频率相关性需要单独检查

对 token 频次 $f_i$ 和向量长度，可以报告

$$
\rho_{\text{norm}}
=
\operatorname{corr}
\big(
\log(1+f_i),
\lVert\mathbf v_i\rVert_2
\big).
$$

相关性只说明两个观测量共同变化，不能自动说明频率造成了长度变化。至少应按频率分桶，报告每个桶的样本数、长度分布、近邻稳定性和 OOV 或 special-token 处理。

### Contextual 表示把一个 token 变成一组点

静态 embedding 是

$$
\mathbf v_t=E_{\text{tok}}[t].
$$

同一个 token 在不同上下文 $c$、不同位置 $j$ 和不同层 $\ell$ 的表示可以写成

$$
\mathbf h_{t,c,j}^{(\ell)}.
$$

如果要研究词义随上下文变化，不能把所有出现位置先平均成一行再声称看到了多义性。应保留 occurrence 级别，或者明确平均、池化和采样条件。

| 表示 | 样本单位 | 几何稳定性 | 适合的结论 |
| --- | --- | --- | --- |
| 静态 embedding | token ID | 在 checkpoint 内固定 | 训练目标塑造了词表行之间的关系 |
| contextual state | token occurrence 和层 | 随上下文、位置和层变化 | 某层如何编码局部语境 |
| occurrence 平均 | token 与采样协议 | 依赖样本分布 | 选定语料上的平均状态 |
| sentence pooling | 句子与 pooling 规则 | 依赖长度和内容 | 句子级任务中的表示关系 |

同一 token 的静态行和 contextual state 可以同时分析，但必须分开命名、分开采样、分开报告。

## 几何审计协议

### 先固定输入和表示

执行近邻或相似度分析前，记录：

1. tokenizer 的版本、词表和 special token；
2. checkpoint 的来源、训练步数和参数指纹；
3. 表示来自输入表、输出表还是某个 hidden layer；
4. token、occurrence、句子还是文档作为样本单位；
5. 行向量或列向量的形状约定；
6. padding、mask、OOV 和 subword 的处理；
7. 是否中心化、单位化或去除公共方向。

这些字段缺一项，后续的数值仍然可以计算，但不能作为可复现的比较结果。

### 再并列计算多种指标

对同一批向量至少报告：

| 报告项 | 原始向量 | 单位化向量 | 中心化后单位化 |
| --- | --- | --- | --- |
| 长度分布 | 记录均值、分位数和极值 | 应接近 1 | 重新记录 |
| 点积 | 记录尺度贡献 | 等于余弦 | 可能改变 |
| 欧氏距离 | 记录绝对位置差 | 与余弦单调对应 | 反映新坐标 |
| top-$k$ 邻域 | 记录候选和排序 | 记录重排 | 记录重排 |
| hub count | 记录集中程度 | 记录指标效应 | 记录预处理效应 |

不要只把一份最符合预期的近邻表放进结论。原始、单位化和中心化结果的差异本身就是几何证据。

### 用基线和重采样核对稳定性

一个最小的稳定性检查包含：

1. 从相同频率桶抽取多个候选子集；
2. 在同一候选集上比较不同 metric；
3. 计算 top-$k$ 集合的交并比；
4. 与随机向量或标签置换后的结果比较；
5. 对高频、低频、special token 分层报告；
6. 对 contextual state 固定层、位置和上下文采样；
7. 保存查询向量、候选 ID、预处理和随机种子。

如果近邻只在某个 metric、某个频率桶或某次采样中出现，应把结论写成条件性观察。

### 一个可回放的 toy 检查

对前面的四个向量，独立计算得到：

$$
\begin{aligned}
\lVert\mathbf v_{\text{cat}}\rVert_2&=\sqrt{1.25}\approx1.118034,\\
\mathbf v_{\text{cat}}\mathbf v_{\text{dog}}^{\mathsf T}&=1.1,\\
d_2^2(\mathbf v_{\text{cat}},\mathbf v_{\text{dog}})&=0.05,\\
\cos(\mathbf v_{\text{cat}},\mathbf v_{\text{dog}})&\approx0.983870.
\end{aligned}
$$

同时，dog 与 car 的欧氏距离为 $1$，而 cat 与 car 的欧氏距离约为 $1.204159$。如果实现输出了不同的顺序，应先检查向量行、转置、归一化和平方距离约定，再解释模型。

## 失效模式：近邻表不等于几何解释

### 1. 把 token ID 的差当作距离

ID 只选择 embedding 行。应使用行向量之间的距离或相似度，不要使用两个整数 ID 的差。

### 2. 把点积当作方向相似度

点积包含长度。应同时报告范数，或明确说明已经单位化。

### 3. 在零向量上计算余弦

零向量没有方向。应在计算前检查范数，并记录过滤或平滑规则。

### 4. 混用平方距离和距离

平方距离可以用于排序，但数值单位不同。表头和结论必须明确是 $d_2$ 还是 $d_2^2$。

### 5. 把中心化当作默认修复

中心化会改变原始空间。应把原始与中心化结果并列，说明公共方向判断的基线。

### 6. 直接比较两个 checkpoint 的坐标

正交变换就可以改变每个坐标而保留全部内积。应先共享 token，再确认对齐、metric 和层。

### 7. 把静态行和 contextual state 混在一张表

静态行按 token 固定，contextual state 按 occurrence 变化。应分开定义样本单位和池化规则。

### 8. 用一次 top-$k$ 结果证明语义或因果关系

近邻是统计观察。应加入频率分层、重采样、标签或随机基线，并把因果解释留给干预实验。

## 结语

Embedding 几何从四个接口开始：长度保留尺度，点积结合尺度与方向，欧氏距离比较点的位置，余弦只比较非零向量的方向。归一化、中心化和坐标变换会改变其中一部分量，因此“近邻”只有在对象、metric、预处理和 checkpoint 都固定后才有明确含义。

训练得到的空间可以保存统计关系，也可以带有公共方向、频率效应和 hubness。静态 embedding 是词表行；contextual state 是由上下文、位置和层共同决定的一组运行时点。完整的几何结论需要同时提供数值、基线、样本协议和失败边界。

## 相关词条

[Embeddings](../text-representation/embeddings/)

[One-hot 与分布式表示](../text-representation/one-hot-and-distributed/)

[余弦相似度](../text-representation/cosine-similarity/)

[长度与距离](../linear-algebra/lengths-and-distances/)

[角度与正交](../linear-algebra/angles-and-orthogonality/)

[词表大小权衡](../text-representation/vocabulary-size-tradeoffs/)

[Tokenization](../text-representation/tokenization/)
