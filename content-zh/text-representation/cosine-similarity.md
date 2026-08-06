---
title: "余弦相似度：只比较方向时究竟保留了什么"
tags: ["why-models-learn"]
---

余弦相似度把两个非零向量的点积除以它们的长度乘积。结果只保留方向关系，范围为 $[-1,1]$：同向为 $1$，正交为 $0$，反向为 $-1$。它会丢弃长度信息，因此适合回答“两个表示的方向是否一致”，不适合替代所有 score、距离或概率。

这条定义看似简单，却包含几个必须核对的条件：两个向量不能是零向量；正缩放不改变结果，负缩放会翻转结果；只有在先单位化时，余弦排序才与欧氏距离排序一致；$1-\cos$ 在一般非零向量上也不是严格的距离，因为正缩放后的两个不同点会得到零差异。

本篇从公式、角度和单位球面开始，推导矩阵批量计算、排序、梯度和温度缩放，再说明高维空间、中心化、频率、contextual state 与阈值选择会怎样影响解释。

![两个向量的夹角决定余弦相似度；单位化后余弦与欧氏距离可以互相换算，但原始点积还包含长度](/assets/text-representation/svg/cosine-similarity.1.svg)

## 定义与几何含义

### 余弦是归一化点积

对非零向量 $\mathbf x,\mathbf y\in\mathbb R^d$，余弦相似度定义为

$$
\cos(\mathbf x,\mathbf y)
=
\frac{\mathbf x\mathbf y^{\mathsf T}}
{\lVert\mathbf x\rVert_2\lVert\mathbf y\rVert_2}.
$$

由 Cauchy–Schwarz 不等式，

$$
\left\lvert\mathbf x\mathbf y^{\mathsf T}\right\rvert
\le
\lVert\mathbf x\rVert_2\lVert\mathbf y\rVert_2,
$$

所以

$$
-1\le
\cos(\mathbf x,\mathbf y)
\le1.
$$

余弦相似度是一个 score。它不是把两个向量映射到概率，也不自动表示语义真值。

### 余弦就是夹角的余弦

若 $\theta$ 是两个非零向量的夹角，则

$$
\cos(\mathbf x,\mathbf y)=\cos\theta.
$$

因此：

| 余弦值 | 夹角 | 几何关系 |
| ---: | ---: | --- |
| $1$ | $0^\circ$ | 同向 |
| $0$ | $90^\circ$ | 正交 |
| $-1$ | $180^\circ$ | 反向 |
| 接近 $1$ | 接近 $0^\circ$ | 方向接近 |
| 接近 $0$ | 接近 $90^\circ$ | 方向近似无关，仍需基线 |

“接近 $0$”只表示在当前坐标和样本集合中方向接近正交。高维随机向量也会产生大量接近 $0$ 的 pair，不能把它自动读成无关系的语言结论。

### 零向量没有余弦方向

当 $\mathbf x=\mathbf 0$ 或 $\mathbf y=\mathbf 0$ 时，分母为 0，余弦未定义：

$$
\lVert\mathbf x\rVert_2\lVert\mathbf y\rVert_2=0
\quad\Longrightarrow\quad
\cos(\mathbf x,\mathbf y)\text{ 未定义}.
$$

工程实现有时会用 $\varepsilon$ 防止除零：

$$
\frac{\mathbf x\mathbf y^{\mathsf T}}
{\max(\lVert\mathbf x\rVert_2,\varepsilon)
\max(\lVert\mathbf y\rVert_2,\varepsilon)}.
$$

这不是原始余弦的同一个函数。报告时要区分：过滤零向量、返回缺失值、使用平滑分母，还是把零向量当作一个特殊类别。

## 缩放不变量与反例

### 正缩放不改变余弦

对 $\alpha>0,\beta>0$：

$$
\cos(\alpha\mathbf x,\beta\mathbf y)
=
\frac{\alpha\beta\mathbf x\mathbf y^{\mathsf T}}
{\alpha\lVert\mathbf x\rVert_2\beta\lVert\mathbf y\rVert_2}
=
\cos(\mathbf x,\mathbf y).
$$

所以余弦不会看到同一方向上的长度差异。若长度携带训练目标中的置信度、频率或 logit 尺度，余弦会主动丢弃这部分信息。

### 负缩放会翻转方向

若只有一个缩放因子为负：

$$
\cos(-\alpha\mathbf x,\beta\mathbf y)
=
-\cos(\mathbf x,\mathbf y),
\qquad
\alpha>0,\ \beta>0.
$$

这说明“缩放不变”必须限定为正缩放。把任意线性变换都称作余弦不变量是不准确的。

### 一个最小数值例子

设

$$
\mathbf a=(3,4),
\quad
\mathbf b=(4,3),
\quad
\mathbf c=(-3,-4),
\quad
\mathbf z=(0,0).
$$

长度和点积为

| 向量或 pair | 长度或点积 | 余弦 |
| --- | ---: | ---: |
| $\mathbf a$ | $\lVert\mathbf a\rVert_2=5$ | — |
| $\mathbf b$ | $\lVert\mathbf b\rVert_2=5$ | — |
| $\mathbf a,\mathbf b$ | $\mathbf a\mathbf b^{\mathsf T}=24$ | $0.96$ |
| $\mathbf a,\mathbf c$ | $\mathbf a\mathbf c^{\mathsf T}=-25$ | $-1$ |
| $\mathbf a,\mathbf z$ | 分母为 0 | 未定义 |

$\mathbf a$ 和 $\mathbf b$ 的欧氏距离为 $\sqrt2$，但余弦为 $0.96$。$\mathbf c$ 与 $\mathbf a$ 长度相同且方向完全相反，因此余弦为 $-1$。$\mathbf z$ 没有方向，不能被静默当作余弦为 0。

## 单位球面上的距离关系

### 先单位化再计算

对任意非零向量定义单位化版本

$$
\widehat{\mathbf x}
=
\frac{\mathbf x}{\lVert\mathbf x\rVert_2},
\qquad
\lVert\widehat{\mathbf x}\rVert_2=1.
$$

单位化后的点积就是余弦：

$$
\widehat{\mathbf x}\widehat{\mathbf y}^{\mathsf T}
=
\cos(\mathbf x,\mathbf y).
$$

一批向量逐行单位化后，尺度被移出 pairwise score。这个操作改变了问题：它保留方向，删除长度。

### 余弦与单位化欧氏距离互相换算

对两个单位向量：

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

因此

$$
\cos(\mathbf x,\mathbf y)
=
1-\frac12
\lVert\widehat{\mathbf x}-\widehat{\mathbf y}\rVert_2^2.
$$

在单位球面上，最大余弦等价于最小欧氏距离。没有单位化时，右侧应使用单位化向量，不能直接把原始距离代入。

### 角距离和 cosine distance 不是同一个对象

常见的两个变换是

$$
d_{\text{cos}}(\mathbf x,\mathbf y)
=
1-\cos(\mathbf x,\mathbf y),
\qquad
d_{\angle}(\mathbf x,\mathbf y)
=
\arccos\big(\cos(\mathbf x,\mathbf y)\big).
$$

$d_{\angle}$ 是单位球面上的角距离。$d_{\text{cos}}$ 适合排序和损失，但在一般非零向量集合上不是严格 metric：$\mathbf y=2\mathbf x$ 时两个不同点的 $d_{\text{cos}}$ 仍为 0。命名时应使用“余弦差异”或“余弦距离 score”，除非已经说明了空间和公理范围。

## 点积、欧氏距离与余弦的排序差异

### 长度会让点积改变排序

设查询向量为

$$
\mathbf q=(1,0),
\qquad
\mathbf r=(10,5),
\qquad
\mathbf s=(1,0.1).
$$

点积和余弦分别为

| 候选 | 点积 $\mathbf q\mathbf x^{\mathsf T}$ | 候选长度 | 余弦 |
| --- | ---: | ---: | ---: |
| $\mathbf r$ | $10$ | $\sqrt{125}\approx11.180340$ | $0.894427$ |
| $\mathbf s$ | $1$ | $\sqrt{1.01}\approx1.004988$ | $0.995037$ |

点积选择 $\mathbf r$，余弦选择 $\mathbf s$。前者优先考虑 score 的尺度，后者优先考虑方向。不存在脱离任务的“正确指标”；应根据训练目标或检索目的选择。

### 什么时候欧氏距离与余弦排序相同

同一个查询和一组候选都先单位化时：

$$
d_2^2(\widehat{\mathbf q},\widehat{\mathbf x})
=
2-2\cos(\mathbf q,\mathbf x).
$$

右侧对余弦是严格递减函数，所以两个排序相同。若候选长度不同而没有单位化，欧氏距离会同时看到位置和长度，排序可能与余弦不同。

| 条件 | 点积排序 | 欧氏距离排序 | 余弦排序 |
| --- | --- | --- | --- |
| 原始向量，长度不同 | 受长度与方向影响 | 受位置与长度影响 | 只受方向影响 |
| 查询与候选都单位化 | 等于余弦排序 | 等于余弦排序的反向顺序 | 与单位化距离等价 |
| 有零向量 | 点积仍可计算 | 距离仍可计算 | 需要单独处理 |
| 目标使用未归一化 logits | 可能保留置信度尺度 | 通常不是训练 score | 可能不匹配目标 |

### top-k 结果必须携带 metric

对候选集合 $\{\mathbf x_i\}_{i=1}^{n}$，余弦 top-k 可以写成

$$
N_k^{\cos}(q)
=
\operatorname{arg\,topk}_{i\ne q}
\cos(\mathbf q,\mathbf x_i).
$$

如果检索系统先用内积索引、再用余弦重排，两个阶段的候选范围必须记录。初始候选没有覆盖真正的余弦邻居时，后续重排不能修复召回损失。

## 矩阵实现与形状核对

### 两个矩阵单位化后一次计算

设查询矩阵 $X\in\mathbb R^{m\times d}$，候选矩阵 $Y\in\mathbb R^{n\times d}$。逐行单位化：

$$
\widehat X_{i,:}
=
\frac{X_{i,:}}{\lVert X_{i,:}\rVert_2},
\qquad
\widehat Y_{j,:}
=
\frac{Y_{j,:}}{\lVert Y_{j,:}\rVert_2}.
$$

余弦矩阵为

$$
C
=
\widehat X\widehat Y^{\mathsf T}
\in\mathbb R^{m\times n},
\qquad
C_{ij}
=
\cos(X_{i,:},Y_{j,:}).
$$

矩阵乘法的形状给出最直接的检查：$m\times d$ 乘 $d\times n$ 必须得到 $m\times n$。如果结果是 $n\times m$，可能只是转置；如果结果维度包含 batch 或 sequence 轴，则需要重新确认样本单位。

| 中间量 | 形状 | 检查 |
| --- | --- | --- |
| 查询矩阵 $X$ | $m\times d$ | 行数是查询数 |
| 候选矩阵 $Y$ | $n\times d$ | 行数是候选数 |
| 行范数 | $m\times1$ 或 $n\times1$ | 每行只有一个尺度 |
| 单位化矩阵 $\widehat X,\widehat Y$ | 与输入相同 | 非零行范数应为 1 |
| cosine matrix $C$ | $m\times n$ | 每个元素对应一个 query-candidate pair |

### 同一矩阵的 cosine matrix 有对称性

当 $X=Y$ 且所有行都非零时：

$$
C=\widehat X\widehat X^{\mathsf T}.
$$

因此

$$
C=C^{\mathsf T},
\qquad
C_{ii}=1.
$$

数值实现中对称性只会受到浮点误差影响。若对角线明显不是 1，先检查是否包含零行、是否按错误轴归一化，或是否把向量列当成样本。

### 分块计算不改变公式

词表很大时，可以将候选矩阵切成块：

$$
Y=
\begin{bmatrix}
Y^{(1)}\\
Y^{(2)}\\
\vdots\\
Y^{(r)}
\end{bmatrix},
\qquad
C=
\begin{bmatrix}
\widehat X\widehat Y^{(1)\mathsf T}
&
\cdots
&
\widehat X\widehat Y^{(r)\mathsf T}
\end{bmatrix}.
$$

每个块的归一化必须使用它自己的候选行范数，但所有块必须共享同一个 checkpoint、层、metric 和 token 映射。分块只改变内存计划，不应改变结果。

## 余弦在训练目标中的作用

### 归一化会改变梯度方向

对固定非零 $\mathbf y$，令

$$
c(\mathbf x)
=
\frac{\mathbf x\mathbf y^{\mathsf T}}
{\lVert\mathbf x\rVert_2\lVert\mathbf y\rVert_2}.
$$

对 $\mathbf x$ 求梯度：

$$
\nabla_{\mathbf x}c
=
\frac{\mathbf y}{\lVert\mathbf x\rVert_2\lVert\mathbf y\rVert_2}
-
\frac{c(\mathbf x)}{\lVert\mathbf x\rVert_2^2}\mathbf x.
$$

第一项把 $\mathbf x$ 拉向 $\mathbf y$，第二项抵消纯粹的径向变化。梯度不是简单地把两个向量相减。

当 $\lVert\mathbf x\rVert_2=\lVert\mathbf y\rVert_2=1$ 时：

$$
\nabla_{\mathbf x}c
=
\mathbf y-c(\mathbf x)\mathbf x.
$$

它与 $\mathbf x$ 正交：

$$
\mathbf x
\big(\mathbf y-c(\mathbf x)\mathbf x\big)^{\mathsf T}
=
c(\mathbf x)-c(\mathbf x)=0.
$$

这说明单位球面上的余弦梯度只推动切向方向，不直接增加 $\mathbf x$ 的长度。

### 温度把余弦变成 logits

在对比学习或检索式分类中，常见 score 是

$$
s_{ij}
=
\tau\cos(\mathbf x_i,\mathbf y_j),
\qquad
\tau>0.
$$

对同一批 pair，$\tau$ 不改变余弦排序，却改变 softmax 分布的尖锐程度。若

$$
p_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{k=1}^{n}\exp(s_{ik})},
$$

则较大的 $\tau$ 会放大 pair 之间的 logit 差异，较小的 $\tau$ 会让分布更平。温度不是余弦本身的尺度；报告相似度时应分开记录 $\cos$ 和 $\tau\cos$。

### 余弦损失不能替代任务指标

训练目标可能最大化正 pair 的余弦、压低负 pair 的余弦，但下游任务还可能依赖范数、偏置、位置或上下文。一个 pair 的 cosine 提高，不等于准确率必然提高。

| 层级 | 计算量 | 适合的验证 |
| --- | --- | --- |
| 表示层 | cosine、范数、角距离 | 检查几何是否按目标移动 |
| 检索层 | top-k、召回、排序 | 检查候选是否可找回 |
| 任务层 | accuracy、F1、loss 或生成指标 | 检查表示变化是否转化为任务收益 |

三个层级需要分别记录。不要用表示层的余弦变化替代任务层的评测。

## 高维、公共方向与阈值

### 高维随机基线

若 $\mathbf x,\mathbf y$ 独立且均匀分布在 $d$ 维单位球面上，理论基线为

$$
\mathbb E[\cos(\mathbf x,\mathbf y)]=0,
\qquad
\operatorname{Var}[\cos(\mathbf x,\mathbf y)]
\approx\frac{1}{d}.
$$

维度越高，随机 pair 的 cosine 通常越集中在 0 附近。实际 embedding 可能存在公共方向，使平均 cosine 偏离这个基线。基线只用于校准数量级，不替代真实语料上的负样本分布。

### 中心化后再单位化是另一种相似度

候选集合均值为

$$
\boldsymbol\mu
=
\frac{1}{n}\sum_{i=1}^{n}\mathbf v_i.
$$

中心化并单位化：

$$
\overline{\mathbf v}_i
=
\frac{\mathbf v_i-\boldsymbol\mu}
{\lVert\mathbf v_i-\boldsymbol\mu\rVert_2}.
$$

中心化后的余弦是

$$
\cos_{\text{centered}}(i,j)
=
\overline{\mathbf v}_i\overline{\mathbf v}_j^{\mathsf T}.
$$

它研究的是“相对于候选均值的方向”，不是原始 embedding 的方向。两种结果都可以有用，但必须在标签中区分。

### 固定阈值不具有跨模型通用性

阈值 $\tau_{\cos}$ 把 pair 分为“相似”与“不相似”：

$$
\operatorname{similar}(x,y)
=
\mathbf 1\big[
\cos(\mathbf x,\mathbf y)\ge\tau_{\cos}
\big].
$$

阈值的含义取决于正负 pair 的分布、领域、token 频率、表示层、归一化和误报漏报代价。一个模型上的 $0.8$ 不能直接复制到另一个 checkpoint。

选择阈值时至少报告：

| 条件 | 需要说明的内容 |
| --- | --- |
| 正样本 | 标签来源、粒度和去重规则 |
| 负样本 | 随机、频率匹配、难负样本或标签置换 |
| 表示 | checkpoint、层、token 或 occurrence |
| 预处理 | 原始、单位化、中心化或其他变换 |
| 目标 | precision、recall、F1、AUROC 或检索收益 |

没有这些字段，阈值只是一个不可迁移的数值。

## Contextual 表示中的余弦

### 同一 token 的不同出现是不同 pair

静态 embedding 对 token $t$ 只有一行：

$$
\mathbf v_t=E_{\text{tok}}[t].
$$

contextual 模型在上下文 $c$、位置 $j$、层 $\ell$ 输出

$$
\mathbf h_{t,c,j}^{(\ell)}.
$$

比较同一 token 的语境变化时，应计算 occurrence 之间的 cosine，并固定层和池化。比较静态行与 contextual state 时，先说明是否处于相同维度和同一坐标接口。

### 余弦高不等于句子可替换

两个 occurrence 的 cosine 高，表示选定层的向量方向接近。它不能单独证明：

1. 两个 token 在所有句法环境中可替换；
2. 模型在生成时使用了同一个特征；
3. 两个句子的事实、情感或因果关系相同；
4. 表示上的近邻能在任务层获得相同预测。

这些结论需要加入上下文控制、干预或任务评测。

## 审计协议：把一个余弦数值变成证据

### 先做输入和数值检查

对每批向量执行：

1. 确认 dtype、设备和 shape；
2. 统计零行、近零行和长度分位数；
3. 固定单位化轴；
4. 用一个手算 pair 复核点积、范数和余弦；
5. 检查结果是否落在 $[-1,1]$；
6. 对同一矩阵检查对角线是否接近 1；
7. 用分块与非分块结果做误差比较。

近零向量的相似度会对浮点误差和分母规则敏感，应单独列出。

### 再做几何和任务检查

一个可复现的报告至少包含：

| 证据 | 最小记录 |
| --- | --- |
| pair 数值 | 两个向量 ID、长度、点积、余弦和是否零行 |
| 排序 | query、候选集、metric、top-k 和 tie 规则 |
| 预处理 | 是否单位化、中心化、去公共方向 |
| 稳定性 | 重采样后的重合率、频率分层和随机基线 |
| 训练关系 | temperature、正负样本、loss 与梯度口径 |
| 任务关系 | 检索或下游评测是否同步改善 |

### 一个可回放的 toy 契约

对 $\mathbf a=(3,4)$、$\mathbf b=(4,3)$：

$$
\lVert\mathbf a\rVert_2
=
\lVert\mathbf b\rVert_2
=5,
\qquad
\mathbf a\mathbf b^{\mathsf T}=24,
\qquad
\cos(\mathbf a,\mathbf b)=\frac{24}{25}=0.96.
$$

对查询 $\mathbf q=(1,0)$、候选 $\mathbf r=(10,5)$ 和 $\mathbf s=(1,0.1)$：

$$
\cos(\mathbf q,\mathbf r)
=\frac{10}{\sqrt{125}}
\approx0.894427,
\qquad
\cos(\mathbf q,\mathbf s)
=\frac{1}{\sqrt{1.01}}
\approx0.995037.
$$

点积排序为 $\mathbf r$ 在前，余弦排序为 $\mathbf s$ 在前。实现若产生相同排序，应检查是否意外单位化、是否用错候选向量或是否实际计算了不同指标。

## 失效模式：归一化后的分数不等于完整表示

### 1. 对零向量静默返回 0

零向量的余弦未定义。应报告过滤、缺失或平滑分母规则。

### 2. 把正缩放和任意缩放混为一谈

正缩放保持余弦，负缩放翻转方向。应明确缩放因子的符号。

### 3. 把 $1-\cos$ 当作所有空间上的 metric

在一般非零向量上，正缩放会让不同点的 cosine distance 为 0。需要严格 metric 时使用单位球面角距离或说明公理范围。

### 4. 只比较 cosine，不检查长度

余弦删除尺度。训练目标或下游 score 可能仍然依赖范数，应并列报告长度和点积。

### 5. 未说明单位化轴

对 embedding 矩阵逐行单位化和对 feature 列单位化得到不同空间。应记录样本轴、feature 轴和广播规则。

### 6. 直接复用跨 checkpoint 阈值

表示层、tokenizer、频率和公共方向改变后，cosine 分布会改变。应在目标 checkpoint 上重新校准。

### 7. 把中心化后的 cosine 当作原始 cosine

中心化改变均值和 pairwise 关系。应在指标名称中标出 centered 或其他预处理。

### 8. 用 cosine 近邻代替任务证据

方向接近只说明一个几何观察。应同时提供检索、分类、生成或干预层证据。

## 结语

余弦相似度是归一化点积。它保留方向，删除长度；在单位球面上，它与欧氏距离可以互相换算；在原始 embedding 上，它与点积和欧氏距离回答的是不同问题。零向量、负缩放、中心化、公共方向和维度基线都必须进入审计协议。

一个可靠的 cosine 结论至少绑定到对象、checkpoint、层、tokenization、预处理、候选集和任务指标。数值高低只是中间观察，稳定的近邻和可迁移的阈值还需要基线、重采样和下游验证。

## 相关词条

[Embedding 几何](../text-representation/embedding-geometry/)

[Embeddings](../text-representation/embeddings/)

[One-hot 与分布式表示](../text-representation/one-hot-and-distributed/)

[长度与距离](../linear-algebra/lengths-and-distances/)

[角度与正交](../linear-algebra/angles-and-orthogonality/)

[词表大小权衡](../text-representation/vocabulary-size-tradeoffs/)

[Tokenization](../text-representation/tokenization/)
