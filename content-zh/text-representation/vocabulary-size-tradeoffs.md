---
title: "词表大小权衡：覆盖率、序列长度与计算成本如何交换"
tags: ["why-models-learn"]
---

词表大小 $V$ 决定 tokenizer 可以直接输出多少种 token。增大 $V$ 通常能把常见片段合并得更长，降低序列长度和碎片化；同时会增加 embedding 参数、输出词表投影和每个位置的 softmax 候选数。减小 $V$ 会降低词表相关参数，却可能让相同文本变成更长的 token 序列，增加 attention、padding 和上下文窗口压力。

因此词表大小不是一个孤立的超参数。它同时改变离散接口、参数账本、序列长度分布、低频覆盖、跨语言分配和推理吞吐。比较两个 tokenizer 时，必须把 $V$、token 数、embedding 宽度、输出头、批次形状和任务指标放在同一份证据里。

![三种分词粒度把同一文本切成不同数量的 token；词表越大通常序列越短，但词表参数和输出候选越多](/assets/text-representation/svg/vocabulary-size-tradeoffs.1.svg)

## 词表大小决定什么

### $V$ 是离散接口的宽度

词表写成

$$
\mathcal V=\{t_0,t_1,\ldots,t_{V-1}\}.
$$

tokenizer 把字符串映射为 ID 序列：

$$
\operatorname{encode}(x)
=
(x_1,x_2,\ldots,x_L),
\qquad
x_\ell\in\{0,1,\ldots,V-1\}.
$$

这里 $V$ 是可以直接查表的 token 数，$L$ 是某个输入被切分后的序列长度。增大 $V$ 不保证每个输入的 $L$ 都变小；实际变化取决于训练语料、merge 规则、预分词、normalization 和语言分布。

### 词表条目不是字符数或词数

一个 token 可以是完整单词、词片段、字符、byte、special token 或带边界标记的组合。相同的 $V$ 可以对应不同的 token 粒度。需要单独记录：

| 量 | 含义 | 受什么影响 |
| --- | --- | --- |
| $V$ | 词表条目数量 | tokenizer 训练预算和 special token 预留 |
| $L$ | 一条输入的 token 数 | 文本、语言、切分规则和 byte fallback |
| $d$ | embedding 宽度 | 模型架构和表示容量 |
| $F=L/W$ | 每个词的平均 token 数 | 词表覆盖与形态结构 |
| coverage | 输入片段被直接或回退覆盖的比例 | 词表、Unicode 和 OOV 协议 |

$V$ 大不等于语义空间维度大。embedding 维度是 $d$；词表大小主要增加行数和输出类别数。

## 一个 toy 输入的三种切分

### 粒度改变序列长度

对句子 “cats are unhappy”，假定空格也作为字符 token，比较三种切分：

| 粒度 | token 序列 | $L$ | 主要特征 |
| --- | --- | ---: | --- |
| word | cats / are / unhappy | $3$ | 序列短，未登录新词处理压力大 |
| subword | cats / are / un / happy | $4$ | 复用片段，能组合较少见词形 |
| character | c / a / t / s / space / a / r / e / space / u / n / h / a / p / p / y | $16$ | 覆盖细，序列长，组合步数多 |

这个例子不比较实际 tokenizer 的质量，只固定同一字符串展示 $L$ 如何随粒度改变。对 self-attention，token 数从 4 变为 16 时，成对位置数从

$$
4^2=16
\quad\text{变为}\quad
16^2=256.
$$

如果只看词表参数而不看序列长度，会漏掉平方级的交互成本。

### token 数会改变上下文预算

设上下文窗口上限为 $T$，每个自然词平均占用 $F$ 个 token，则粗略可容纳的词数为

$$
W_{\text{window}}
\approx
\frac{T}{F}.
$$

这个近似忽略 special token、padding、语言混合和长尾输入。$F$ 从 1.2 增到 2.0 时，同一个 $T$ 能容纳的自然词数会下降约 40%。更长序列还会增加截断概率。

## 词表参数账本

### 输入 embedding 随 $Vd$ 线性增长

输入 embedding 矩阵的参数量为

$$
N_{\text{in}}
=
Vd.
$$

当 $d=768$ 时：

| 词表大小 $V$ | 输入 embedding 参数 | 相对 32k |
| ---: | ---: | ---: |
| $32{,}000$ | $24{,}576{,}000$ | $1$ |
| $64{,}000$ | $49{,}152{,}000$ | $2$ |
| $128{,}000$ | $98{,}304{,}000$ | $4$ |

这些参数还会影响 checkpoint 大小、初始化时间和 embedding 行的内存带宽。它们不直接说明表示质量。

### 输出头可能再增加一份 $Vd$

如果模型用隐藏宽度 $d$ 产生 $V$ 个输出 logits，输出投影参数量近似为

$$
N_{\text{out}}
=
dV.
$$

输入和输出权重不共享时，词表相关矩阵合计约为

$$
N_{\text{vocab}}
\approx
2Vd.
$$

如果使用 tied weights，把输入矩阵转置用于输出投影，则可减少一份矩阵；输出 bias、额外投影和分片布局仍需单独计入。

| 输出架构 | 词表矩阵 | $V=128{,}000,d=768$ 时的矩阵参数 |
| --- | --- | ---: |
| untied | $E_{\text{in}}$ 与 $W_{\text{out}}$ 两份 | $196{,}608{,}000$ |
| tied | 一份矩阵复用 | $98{,}304{,}000$ |
| tied 加额外投影 | 共享 embedding 但增加 $d\times d$ 或其他头 | 至少一份 $Vd$，再加投影 |

“词表翻倍”对 tied 和 untied 架构的绝对影响不同。比较模型时要先确认权重是否共享。

### 完整 softmax 每个位置都看到 $V$ 个候选

对批次大小 $B$、序列长度 $L$ 的输出，完整词表 logits 的元素数量为

$$
N_{\text{logit}}
=
BLV.
$$

例如 $B=8,L=256$：

| $V$ | 每批 logits 元素数 $BLV$ | 相对 $V=32k$ |
| ---: | ---: | ---: |
| $32{,}000$ | $65{,}536{,}000$ | $1$ |
| $128{,}000$ | $262{,}144{,}000$ | $4$ |

这只是输出元素数量，不等于实际 wall-clock 时间。kernel 融合、并行度、显存带宽和量化都会改变常数，但 $V$ 的线性依赖仍然存在。

## 词表大小与序列计算

### attention 看到的是 $L$，不是字符数

以单层 self-attention 的 score map 为例，位置 pair 数为

$$
N_{\text{attn}}
=
BL^2.
$$

如果把每个位置的向量宽度也纳入一次主要乘法量，粗略项可以写成

$$
C_{\text{attn}}
\propto
BL^2d.
$$

因此把平均序列长度从 $L$ 降到 $L/2$，该项约降为原来的四分之一。词表增大带来的 $Vd$ 增长是线性的，序列缩短带来的 attention 节省可能是平方级的；最终收益取决于它们的实际变化幅度。

### padding 会放大长序列代价

一个 batch 以最长序列 $L_{\max}$ 对齐时，实际处理的位置数约为

$$
N_{\text{pad}}
=
B L_{\max}.
$$

真实 token 利用率可以写成

$$
U_{\text{token}}
=
\frac{\sum_{b=1}^{B}L_b}{B L_{\max}}.
$$

若长尾样本把 $L_{\max}$ 拉高，较小词表造成的碎片化会同时降低 $U_{\text{token}}$。按长度分桶、动态 batch 或 packing 可以缓解 padding，但不能消除真实 token 数和 attention 交互数。

| 成本来源 | 主要变量 | 词表变大通常怎样影响 |
| --- | --- | --- |
| embedding lookup | $Vd$ 的参数和被访问的行 | 参数增加，单次 lookup 仍按被选行读取 |
| 输出 softmax | $BLV$ | 词表候选增加 |
| attention | $BL^2$ 或 $BL^2d$ | 若 $L$ 下降，可能降低 |
| padding | $BL_{\max}$ | 取决于长度分布和 batching |
| context window | $T/F$ 能覆盖的自然词数 | token 更长时通常增加覆盖 |

不能用一个“每 token 成本”代表所有项。embedding、输出头和 attention 对 $V$、$L$ 的依赖不同。

## 词表大与词表小的表示差异

### 大词表减少常见片段的组合步数

更大的词表可以直接保存高频词、常见词缀或跨词片段，带来：

1. 更短的平均 token 序列；
2. 更少的重复组合步骤；
3. 更低的常见文本 padding；
4. 更少的 subword 边界需要由上下文层重建；
5. 更大的直接 lookup 单元。

这些优点依赖训练语料。低频条目如果很少出现，增加一个独立 token 可能只增加参数而没有足够更新。

### 小词表提高复用，但增加组合负担

更小的词表通常让字符或短 subword 在更多词形中复用，带来：

1. 较低的 $Vd$ 和 $BLV$；
2. 更好的稀有词片段覆盖；
3. 更长的序列和更多位置；
4. 更高的 padding 或截断压力；
5. 更大的上下文层组合负担。

复用不是免费共享。模型需要在更多位置上组合片段，且一个片段的统计意义可能依赖上下文。

### 词表条目频率呈长尾

把 token 按训练出现次数分桶，可以比较：

| 频率桶 | 需要观察的量 | 可能的风险 |
| --- | --- | --- |
| 高频 | token 数、平均长度、loss 贡献 | 大 token 占用大量词表预算 |
| 中频 | 组合收益和参数更新次数 | 合并收益可能依赖领域 |
| 低频 | 行更新次数、梯度范数和近邻稳定性 | 条目存在但估计不稳定 |
| 零频或验证独有 | OOV、fallback 和 decode | 训练词表覆盖不代表泛化覆盖 |

词表大小选择应看新增条目的边际收益，而不是只看总覆盖率。

## OOV、byte fallback 与多语言

### 大词表不能消除所有 OOV

训练词表来自有限语料。新实体、拼写变体、代码、emoji 和新语言仍可能不在词表中。若 tokenizer 只有 UNK，未知片段会被压成同一个 token；若使用 subword 或 byte fallback，通常可以保留更多输入信息，但序列可能变长。

对输入集合 $\mathcal D$，可以记录：

$$
r_{\text{UNK}}
=
\frac{\text{UNK token 数}}
{\text{全部 token 数}},
\qquad
r_{\text{byte}}
=
\frac{\text{byte fallback token 数}}
{\text{全部 token 数}}.
$$

这两个比例要按语言、领域和输入类型分层。总平均可能掩盖某个脚本的完全不同结果。

### 共享多语言词表存在分配竞争

固定 $V$ 的多语言 tokenizer 需要在语言、脚本、符号和代码之间分配条目。高资源语言的高频片段可能占据大量 merge，低资源语言因此产生更长序列。

比较多语言词表时至少报告：

| 分层 | 指标 |
| --- | --- |
| 语言或脚本 | 平均 $L$、P95 $L$、每词 token 数 |
| 输入类型 | 自然语言、代码、数字、URL、emoji |
| 覆盖协议 | UNK、byte fallback、字符回退或报错 |
| 成本 | $Vd$、输出候选数、batch 利用率 |
| 任务结果 | 语言级 loss、检索或下游指标 |

单一总体平均不能代表共享词表对每种语言的成本。

## 词表大小的选择方法

### 先定义候选集合和固定账本

可以选择一组候选规模：

$$
\mathcal S_V
=
\{V_1,V_2,\ldots,V_m\}.
$$

对每个候选，记录：

1. tokenizer 训练语料和预分词规则；
2. special token 预留和 byte fallback 协议；
3. 词表大小、embedding 宽度和是否 tied；
4. 训练集与验证集的 token 长度分布；
5. $r_{\text{UNK}}$、$r_{\text{byte}}$ 和每词 token 数；
6. $Vd$、$BLV$、$BL^2d$ 和 padding 利用率；
7. 相同训练预算下的 loss、吞吐和任务指标。

词表改变后，训练样本的 token 序列也改变。不能把一个 tokenizer 的训练步数、token 数和另一个 tokenizer 直接当成相同数据量。

### 用边际收益而非固定经验值

把候选规模按成本和收益排序，观察

$$
\Delta \operatorname{benefit}
=
\frac{\operatorname{metric}(V_{j+1})
-\operatorname{metric}(V_j)}
{V_{j+1}-V_j}.
$$

也可以写一个只用于审计的加权账本：

$$
J(V)
=
\lambda_1 Vd
\;+\;
\lambda_2\mathbb E[BLV]
\;+\;
\lambda_3\mathbb E[BL^2d]
\;+\;
\lambda_4\operatorname{fragmentation}(V).
$$

$J$ 的权重取决于部署目标。训练成本、端到端延迟、内存上限和语言覆盖会给出不同最优点。这个式子用于暴露权衡，不是一个脱离硬件和任务的通用定律。

### 训练与评测必须保持公平

比较词表候选时，至少固定：

| 维度 | 公平比较要求 |
| --- | --- |
| 数据 | 相同原始文本、切分协议和数据顺序 |
| 模型 | 相同层数、宽度、优化器和参数更新预算，或明确调整 |
| 训练量 | 同时报告 step、原始字符数、词数和 token 数 |
| 评测 | 相同原始输入、相同截断规则和相同任务指标 |
| 资源 | GPU 时间、显存、吞吐和 checkpoint 大小 |

如果某个 tokenizer 让每条样本变短，固定 step 不等于固定看到的原始文本量；如果固定 token 数，样本数和更新频率又会改变。结论必须说明采用哪一种公平口径。

## 一个参数与长度的 toy 账本

设 $d=768$，候选词表为 $32k$ 和 $128k$，对应验证集平均序列长度分别为 256 和 192。这里只把它作为计算示例，不声称这两个长度一定来自这两个词表。

| 候选 | $V$ | 平均 $L$ | 输入参数 $Vd$ | attention 位置 pair $L^2$ |
| --- | ---: | ---: | ---: | ---: |
| A | $32{,}000$ | $256$ | $24{,}576{,}000$ | $65{,}536$ |
| B | $128{,}000$ | $192$ | $98{,}304{,}000$ | $36{,}864$ |

从 A 到 B：

$$
\frac{128000}{32000}=4,
\qquad
\frac{192^2}{256^2}
=0.5625.
$$

输入 embedding 参数增加为 4 倍，单层 attention 的位置 pair 约降为 56.25%。输出 softmax 候选数也增加为 4 倍。只有把这些项与实际 batch、kernel、显存和任务收益合并，才能判断 B 是否值得。

## 失效模式：词表规模不是单一质量指标

### 1. 只按词表大小判断 tokenizer 好坏

$V$ 只说明条目数量。应同时报告平均长度、P95 长度、覆盖、分片率、参数和任务指标。

### 2. 只看平均 token 数

平均值会隐藏长尾输入、低资源语言和代码。应报告分位数、按语言或领域分层的长度和截断率。

### 3. 把更大的词表当作更好的语义表示

更大的 token 可能减少组合步骤，也可能成为低频、难更新的独立行。表示质量需要训练和任务证据。

### 4. 忽略输出 softmax

只计算输入 embedding 的 $Vd$ 会漏掉 untied 输出矩阵和每个位置的 $V$ 个 logits。

### 5. 忽略 attention 的平方项

只比较每 token 的 lookup 或 softmax 代价，会遗漏 $L^2$ 的位置交互和 padding 代价。

### 6. 用一个语言的词表替代多语言评估

共享词表可能把预算集中到高资源语言。应按语言、脚本和输入类型报告长度与 fallback。

### 7. 把固定 step 当作固定训练数据量

不同 tokenizer 产生不同 token 数。应同时记录原始文本量、token 量、step 和参数更新次数。

### 8. 把 byte fallback 当作免费覆盖

byte fallback 减少未知输入失败，但可能显著增加序列长度。应同时统计 fallback 比例和长度尾部。

## 最小审计协议

审计一个词表候选时，固定以下证据：

1. 保存 tokenizer、词表、normalization 和 special-token 配置；
2. 对同一原始文本输出 token、ID 和可逆 decode；
3. 按语言、领域和输入类型统计 $L$、P50、P95、截断率和每词 token 数；
4. 计算 $Vd$、是否存在第二份输出矩阵和 $BLV$；
5. 用 batch 长度计算 padding 利用率和 $BL^2$；
6. 统计 UNK、byte fallback、低频 token 和 special token；
7. 在相同公平口径下报告训练 loss、吞吐、显存、延迟和任务指标；
8. 对候选词表重采样验证长度和近邻结论是否稳定；
9. 记录 tokenizer 版本与 checkpoint 的绑定关系。

一个可回放的 toy 检查是：

$$
\begin{aligned}
N_{\text{in}}(32000,768)&=24576000,\\
N_{\text{in}}(128000,768)&=98304000,\\
256^2&=65536,\\
192^2&=36864.
\end{aligned}
$$

它只验证账本算术。它不能替代真实语料上的 token 长度分布、训练曲线或任务评测。

## 结语

词表大小在三组接口之间交换成本：更大的 $V$ 增加词表矩阵和输出候选，更小的 $V$ 通常增加序列长度、组合步骤和 attention 交互。subword、byte fallback、tied weights、padding 和多语言分配会改变这组交换的具体位置。

没有一个脱离数据、模型、硬件和任务的最佳 $V$。可靠选择来自同一原始输入上的候选 tokenizer 对照：报告覆盖和碎片化，报告 $Vd$ 与 $BLV$，报告 $BL^2d$ 与 padding，再把这些成本和下游收益放在同一份审计里。

## 相关词条

[Tokenization](../text-representation/tokenization/)

[BPE 分词](../text-representation/tokenization-bpe/)

[WordPiece 与 SentencePiece](../text-representation/wordpiece-and-sentencepiece/)

[Embeddings](../text-representation/embeddings/)

[Embedding 几何](../text-representation/embedding-geometry/)

[余弦相似度](../text-representation/cosine-similarity/)

[Attention 作为检索](../attention/attention-as-retrieval/)

[自注意力](../attention/self-attention/)
