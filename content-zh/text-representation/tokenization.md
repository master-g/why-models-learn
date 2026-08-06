---
title: "Tokenization：把文本变成模型可处理的离散序列"
tags: ["why-models-learn"]
---

Tokenization（分词或标记化）是把原始文本转换为有限词表中的 token 序列，再把每个 token 映射为整数 ID 的过程。模型通常不直接接收字符串，而是接收类似

$$
\begin{gathered}
(x_1,x_2,\ldots,x_L),\\
x_\ell\in\{0,1,\ldots,V-1\}
\end{gathered}
$$

的离散序列；$L$ 是 token 长度，$V$ 是词表大小。之后 embedding 才把 ID 查成连续向量。

因此 tokenization 不是一个只负责「按空格切开」的预处理小步骤，而是模型接口的一部分。它决定：

1. 什么被当成一个可复用单位；
2. 未登录词能否被拆成已知片段；
3. 同一段文本会占用多少上下文位置；
4. 空格、标点、大小写、Unicode 和特殊 token 如何保留；
5. 训练、推理、评估和部署是否使用完全相同的词表与规则。

这篇建立一条可审计的 pipeline：

$$
\begin{gathered}
\text{原始文本}
\longrightarrow
\text{Unicode/规范化}\\
\longrightarrow
\text{预分词}
\longrightarrow
\text{token 序列}\\
\longrightarrow
\text{词表 ID}
\longrightarrow
\text{embedding}.
\end{gathered}
$$

![Tokenization 把原始文本依次变成规范化字符串、token 片段、词表 ID 和 embedding 输入](/assets/text-representation/svg/tokenization.1.svg)

## 先区分四个对象

### 字符串不是 token 序列

原始输入可能来自字节流、Unicode 字符串或已经分好词的文本。字符串里的「字符」也不是总能直接等同于用户看到的一个符号：

1. Unicode code point 是编码层单位；
2. grapheme cluster 更接近用户看到的一个字形；
3. UTF-8 byte 是实际编码传输的字节；
4. token 是由 tokenizer 词表定义的离散建模单位。

例如带组合音标的字符，视觉上可能是一个字形，Unicode code point 却可能有两个；byte-level tokenizer 看到的单位又不同。若规范化发生在训练后、推理前，token 边界可能改变，模型接收到的序列就不再是原来训练过的接口。

### token 不是 embedding

token 是离散符号或其整数 ID：

$$
\text{token} \longrightarrow \text{integer ID}.
$$

embedding 是可学习的连续查表：

$$
E\in\mathbb R^{V\times d},
\qquad
e_\ell=E[x_\ell]\in\mathbb R^d.
$$

ID 只是词表中的索引。ID 为 42 不表示比 ID 为 7 更「大」、更接近或更有语义；连续几何关系来自 embedding 参数，而不是整数编号本身。

### tokenizer 不是词表文件的全部

完整 tokenizer 通常至少包含：

| 组件 | 作用 | 漂移后果 |
| --- | --- | --- |
| normalization | 处理 Unicode、大小写或空白 | 同一文本可能产生不同字节或字符 |
| pre-tokenization | 识别空格、标点、数字或脚本边界 | merge 的候选边界改变 |
| model | word、字符、BPE、WordPiece 等切分算法 | token 数量与 OOV 行为改变 |
| vocabulary | token 到 ID 的映射 | embedding 行和 checkpoint 不再对应 |
| special tokens | BOS、EOS、PAD、UNK 等协议符号 | shift、mask、停止条件改变 |
| post-processing | 拼接模板、类型 ID、padding、截断 | batch shape 和模型输入改变 |

只复制一个词表文件而漏掉 normalization、merge ranks 或 special-token 模板，通常不能复现原 tokenizer。

## 最小 pipeline：从字符串到 ID

### 1. 原始文本和规范化

把输入记为字符串 $r$。规范化函数可以写成

$$
r'=N(r).
$$

可能的操作包括 Unicode NFC/NFKC、大小写折叠、空白统一、控制字符处理或脚本特定的规则。规范化必须是有明确协议的模型输入变换，不应在没有评估的情况下随意「清洗」文本。

例如视觉上相同的字符可能有两种 Unicode 表示：

$$
\text{é}
\qquad\text{和}\qquad
\text{e}+\text{combining acute accent}.
$$

若 tokenizer 先做 NFC，它们可以归一到同一形式；若 tokenizer 保留 code point 差异，它们可能进入不同切分路径。文本审计应记录实际输入的 Unicode 形式，而不只记录屏幕上的样子。

### 2. 预分词

预分词把规范化字符串划分成较粗的候选片段：

$$
r'\longrightarrow(p_1,p_2,\ldots,p_K).
$$

英文实现可能把空格、标点、数字和字母串分开；中文实现可能按字符、词典或脚本边界处理；byte-level 实现也可能把空格编码进相邻 token。预分词不是最终 tokenization，但它决定后续算法允许在哪里合并。

### 3. 词表模型产生 token

tokenizer model 把每个候选片段变成 token 序列：

$$
(p_1,\ldots,p_K)
\longrightarrow
(t_1,t_2,\ldots,t_L).
$$

token 可以是完整词、字符、字节、子词或带空格标记的片段。token 的边界不一定与语言学上的词边界一致，目标是让有限词表在覆盖率、序列长度和统计复用之间取得平衡。

### 4. 词表查整数 ID

给定词表函数

$$
\operatorname{id}: \mathcal V\longrightarrow\{0,\ldots,V-1\},
$$

得到

$$
(t_1,\ldots,t_L)
\longrightarrow
(x_1,\ldots,x_L),
\qquad
x_\ell=\operatorname{id}(t_\ell).
$$

之后按任务模板添加特殊 token、padding、截断或 segment/type ID，才形成真正送进模型的输入张量。

## 词级、字符级、字节级和子词级

### Word-level：语义单位大，OOV 直接

词级 tokenizer 可能把

$$
\text{I like cats.}
\longrightarrow
[\text{I},\text{like},\text{cats},\text{.}]
$$

每个常见词都能作为一个复用单位，序列短，解释也直观。但词表必须覆盖大量词形、拼写、专名和新词。若词表只有「cat」而没有「cats」，最简单的处理是

$$
\text{cats}\longrightarrow[\text{UNK}],
$$

整个词的信息被一个 unknown 符号替代。不同形态、拼写错误和新实体会迅速放大 OOV。

### Character-level：覆盖广，序列长

字符级 tokenizer 可以把同一句话拆成更细的序列：

$$
\text{cats}\longrightarrow[\text{c},\text{a},\text{t},\text{s}].
$$

字符词表小、几乎没有未知词，但一个语义单位需要多个时间步。对 RNN 来说，路径变长；对 self-attention 来说，序列长度会直接放大成对计算。

### Byte-level：覆盖任意字节，边界不直观

byte-level tokenizer 的基础词表可以覆盖 UTF-8 中的每个 byte，因此理论上不需要为每个 Unicode 字符预先建词表。代价是非 ASCII 文本可能被多个 byte token 表示，token 边界与人类可读字符更不一致。

byte fallback 可以降低 unknown，但不代表序列一定短，也不代表一个 token 对应一个字符。调试时应打印 token 字符串、原始 byte 和 decode 后文本，不能只看 token 数。

### Subword-level：在覆盖与长度之间折中

子词 tokenizer 让常见片段成为一个 token，让少见词由多个已知片段组合。一个玩具词表可以把

$$
\text{unhappiness}
\longrightarrow
[\text{un},\text{happi},\text{ness}],
$$

而新词仍可以通过更小片段表示。这里的切分只用于说明机制，不代表任何具体训练语料的真实词表。

子词的核心优势是组合性：

$$
\text{词表覆盖的片段}
\longrightarrow
\text{更多可组合的词形}.
$$

代价是一个词可能被拆成多个 token，形态边界不一定稳定，跨语言 token 长度也可能差异很大。

## 一个可手算的 toy tokenizer

为了把「切分」和「查表」分开，构造一个只包含少数规则的 toy tokenizer。词表如下：

| token | ID | 备注 |
| --- | ---: | --- |
| 〈PAD〉 | 0 | batch 补齐 |
| 〈UNK〉 | 1 | 无法表示的片段 |
| low | 2 | 常见片段 |
| er | 3 | 后缀片段 |
| s | 4 | 复数片段 |
| cat | 5 | 完整词 |

定义三个固定规则：

1. 「lower」优先匹配「low」与「er」；
2. 「cats」优先匹配「cat」与「s」；
3. 其他无法覆盖的片段变成 〈UNK〉。

则

$$
\begin{aligned}
\text{lower}&\longrightarrow[\text{low},\text{er}]
\longrightarrow[2,3],\\
\text{cats}&\longrightarrow[\text{cat},\text{s}]
\longrightarrow[5,4].
\end{aligned}
$$

这里的 ID 顺序只是词表文件的约定。交换「cat」和「s」的 ID 不会改变 tokenization 语义，只会要求 embedding 矩阵的对应行一起交换。

对 batch

$$
[\text{lower},\text{cats}]
\longrightarrow
[[2,3],[5,4]]
$$

若要求长度为 3，padding 后为

$$
X=
\begin{bmatrix}
2&3&0\\
5&4&0
\end{bmatrix},
\qquad
M=
\begin{bmatrix}
1&1&0\\
1&1&0
\end{bmatrix}.
$$

$M$ 是有效 token mask；它不属于 token ID 本身，却是后续 loss、attention 或 decoder state 更新的重要输入协议。

## BPE：用合并规则逐步形成子词

Byte Pair Encoding（BPE）在 tokenizer 中通常不是直接套用压缩文件的 BPE，而是把最常见的相邻 token pair 记录成 merge rank。初始序列可以从字符或 byte 开始：

$$
[\text{l},\text{o},\text{w},\text{e},\text{r}].
$$

若 merge 规则按优先级包含

$$
(\text{l},\text{o})\to\text{lo},
\qquad
(\text{lo},\text{w})\to\text{low},
\qquad
(\text{e},\text{r})\to\text{er},
$$

依次合并后：

$$
[\text{l},\text{o},\text{w},\text{e},\text{r}]
\to
[\text{lo},\text{w},\text{e},\text{r}]
\to
[\text{low},\text{e},\text{r}]
\to
[\text{low},\text{er}].
$$

真实 BPE tokenizer 还要明确：

1. 初始 alphabet 是字符、byte 还是带空格标记；
2. pair 的优先级如何表示；
3. 并列候选如何处理；
4. 词边界和空格是否进入 token 字符串；
5. 词表 ID 如何分配；
6. decode 时如何恢复原始文本。

因此只知道「使用 BPE」还不足以复现 token 序列；必须拿到完整 tokenizer 配置和版本。

## WordPiece 与 unigram 的接口差异

不同子词算法都在有限词表和序列长度之间折中，但目标函数和解码规则不同：

| 方法 | 典型机制 | 切分决策 | 常见边界 |
| --- | --- | --- | --- |
| BPE | 按 merge rank 合并相邻片段 | 由 merge 顺序驱动 | 需要完整 merge 表 |
| WordPiece | 选择高分片段组合 | 常见实现用最大匹配或词表评分 | 未覆盖时可能回退 UNK |
| Unigram | 给候选片段分配概率 | 从多个切分中选择整体得分高者 | 需要概率/候选规则 |
| byte fallback | byte 作为最终覆盖层 | 无法匹配时拆到 byte | 覆盖高但序列可能变长 |

「子词」不是一个单一算法名。文章、checkpoint 和 tokenizer 必须绑定在一起，不能看到同一个模型家族名就假定所有版本的切分一致。

## 特殊 token：模型协议的一部分

特殊 token 不是普通文本词，但会进入模型的离散序列：

| token | 作用 | 常见边界 |
| --- | --- | --- |
| BOS | decoder 序列开始 | 通常作为第一步输入，不是第一个要预测的内容 |
| EOS | 序列结束 | 训练 target 是否包含它，决定推理何时停止 |
| PAD | batch 对齐 | 必须配合 attention/state/loss mask |
| UNK | 未知片段 | 可能丢失局部字符信息 |
| SEP | 片段或句子边界 | 多段输入的模板协议 |
| CLS | 聚合或分类位置 | 只有模型训练过该位置时才有约定语义 |

对 decoder target，常见 shift 是

$$
\begin{aligned}
\text{decoder input}&=[\text{BOS},y_1,y_2,\ldots,y_{T-1}],\\
\text{target}&=[y_1,y_2,\ldots,y_{T-1},\text{EOS}].
\end{aligned}
$$

如果把 EOS、PAD 或 BOS 错一位，loss 可能仍然能计算，生成却会出现不停止、首 token 错位或 padding 被当作答案的行为。tokenizer 的 special-token ID 必须与模型 forward 的 shift、mask 和 stopping condition 一起核对。

## Token 长度是计算预算

tokenizer 选择会直接改变序列长度 $L$。对 self-attention，成对交互数量大致为

$$
L^2.
$$

把同一内容从 $L=128$ 拆成 $256$ 个 token，长度变为 2 倍，但成对位置数变为 4 倍：

| token 长度 | $L^2$ 个位置对 | 相对 $128$ |
| ---: | ---: | ---: |
| 128 | $16384$ | 1 倍 |
| 256 | $65536$ | 4 倍 |
| 512 | $262144$ | 16 倍 |

因此「token 更细」有覆盖和局部组合的好处，却可能消耗更多上下文和 attention 计算；「token 更粗」减少长度，却可能扩大词表、增加 OOV 或让少见词变成一个不可解释的整体。

### 长度不只是平均值

一个 tokenizer 的平均 token/character 比例不够描述部署成本，还要检查：

1. 不同语言和脚本的分位数；
2. URL、代码、数字和 emoji 的极端长度；
3. 长文截断发生在什么 token 边界；
4. 特殊 token 和模板带来的固定开销；
5. 同一语义内容在不同版本的 token 数变化。

上下文窗口通常按 token 计数，而用户感知的字、词或字符数量可能完全不同。长度预算必须以实际 tokenizer 输出为准。

## 可逆性：decode 能否恢复原文本

一个健康的 tokenizer 通常需要满足

$$
\operatorname{decode}(\operatorname{encode}(r))
\approx r,
$$

其中「约等于」允许协议明确的规范化变化，例如统一换行或 Unicode 形式。若 tokenization 把信息丢成 UNK，decode 就不可能恢复原始片段。

应分别测试：

| 输入类别 | 需要检查 |
| --- | --- |
| 普通空格 | 前后空格是否保留、合并或成为 token |
| 换行和 tab | 是否被统一、删除或编码 |
| Unicode 组合字符 | encode/decode 是否保持规范化约定 |
| 中文和混合脚本 | 字符、词和 byte 边界是否稳定 |
| emoji | 多 code point grapheme 是否被安全处理 |
| URL/代码 | 标点、斜杠、下划线和大小写是否保留 |
| 〈UNK〉 | unknown 出现时是否能接受不可逆损失 |

encode → decode → encode 也可以检查规范化后是否稳定：

$$
\operatorname{encode}(\operatorname{decode}(\operatorname{encode}(r)))
=
\operatorname{encode}(r)
$$

这个等式不是所有自定义后处理都自动满足，但若不满足，就应把差异写进协议，而不是把它当成随机误差。

## Tokenizer 与模型 checkpoint 必须绑定

embedding 矩阵的行与词表 ID 一一对应：

$$
E[x_\ell]\quad\text{读取 ID }x_\ell\text{ 对应的那一行}.
$$

如果 tokenizer A 把「cat」编成 42，而 tokenizer B 把同一个 token 编成 7，却把 B 的 ID 直接送给 A 训练的 checkpoint，模型会读取完全不同的 embedding 行。即使词表大小相同，模型也可能彻底失配。

必须一起固定：

1. tokenizer 文件和版本；
2. normalization 与 pre-tokenization 配置；
3. vocab 与 merge/probability 文件；
4. special-token 字符串和 ID；
5. truncation、padding、left/right padding；
6. encode/decode 的后处理；
7. 模型 checkpoint、embedding 行和输出头的词表绑定。

部署日志至少应保存原始文本哈希、tokenizer 版本、token ID 列表、有效长度和 mask。只保存 decode 后文本，无法诊断 token 边界和 ID 漂移。

## 训练和推理的边界

### tokenizer 不学习语义

tokenizer 可以统计频率、学习 merge 或片段概率，但 token ID 本身不携带语义。语义表示主要由 embedding 和后续网络在训练中学习：

$$
\text{text}
\xrightarrow{\text{tokenizer}}
\text{IDs}
\xrightarrow{\text{embedding}}
\text{vectors}
\xrightarrow{\text{network}}
\text{contextual states}.
$$

把 token 数值直接当作连续输入，会引入虚假的大小关系。例如 ID 42 不应该被模型理解成比 ID 7 更接近 ID 41；正确做法是 embedding lookup 或等价的 one-hot 矩阵乘法。

### tokenization 参与训练目标

语言模型的 next-token loss 计算的是 tokenizer token 的条件概率：

$$
\mathcal L
=-\sum_{\ell=1}^{L}
\log p(x_\ell\mid x_{<\ell}).
$$

因此 token 粒度改变了「一个错误」的单位、序列长度、每个样本的 loss 权重和 perplexity 的可比性。不同 tokenizer 的 token-level perplexity 不能直接当作相同指标比较，至少要说明 token 定义和归一化方式。

### 边界切分影响生成

若一个名字被拆成多个 token，模型生成它时必须连续预测多个位置；若一个罕见短语被当成 UNK，模型可能无法复制它；若空格被编码在 token 前缀中，少一个空格就会走到不同词表路径。tokenization 错误通常会表现为概率链、复制能力或输出格式问题，而不只是「分词看起来不漂亮」。

## 失效模式：文本相同，模型输入却不同

### 训练与推理 tokenizer 不同

训练使用 lowercase + BPE，推理使用原大小写 + WordPiece，会同时改变 token 边界、ID、长度和 special-token 位置。不要只比较 decode 文本；比较每个 token 字符串和 ID。

### Unicode 规范化不一致

同一视觉字符采用不同 code point 序列时，预分词和 byte fallback 可能产生不同结果。固定 normalization，并用组合字符、非拉丁脚本和 emoji 做测试。

### 词表相同，merge 顺序不同

BPE vocab 文件相同但 merge rank 不同，最终切分也会不同。必须保存 merge 文件和排序规则。

### unknown 把局部事实整块抹掉

Word-level 〈UNK〉 可能把人名、数字或拼写变体整个替换。检查 unknown rate、按数据类别分桶，并确定 byte/subword fallback 是否符合任务需求。

### special token 重复或缺失

模板层可能自动添加 BOS/EOS，而调用方又手动添加一次；也可能把 PAD 当成 EOS。检查最终 ID 序列，而不是只看 tokenizer API 的高层参数。

### padding 方向错

left padding 与 right padding 会改变 decoder 的位置、causal mask 和最后一个有效 token。生成模型通常还有 cache 起始位置约定，不能只靠 shape 猜测。

### 截断切断关键边界

按 token 截断可能把一个原始词拆在窗口边界，或丢掉 EOS、代码括号和结构标记。记录 truncation side、保留的 special tokens 和实际 decode 结果。

### 只看平均 token 长度

平均长度稳定，不代表长尾 URL、代码、中文或 emoji 不会超过窗口。应报告分位数和极端样本，尤其是生产输入的真实分布。

## 最小审计协议

面对一个 tokenizer，可以按以下顺序留下证据：

1. **固定输入字节。** 记录原始 UTF-8、Unicode 规范化形式和可见字符串。
2. **逐层打印。** 同时打印 normalization、pre-tokenization、token 字符串、ID、special tokens、mask 和 decode。
3. **核对词表绑定。** 确认 vocab、merge/probability 文件、ID、embedding 行和 checkpoint 版本一致。
4. **做覆盖测试。** 覆盖普通词、未知词、数字、URL、代码、中文、混合脚本、组合字符和 emoji。
5. **做可逆性测试。** 检查 encode/decode，明确 UNK 和规范化造成的允许差异。
6. **做长度账本。** 记录 token 长度、special-token 开销、padding、truncation 和 attention 预算。
7. **做 batch 边界。** 用不同有效长度核对 PAD、mask、left/right padding 和 loss 分母。
8. **做版本回归。** 同一固定样本在训练、推理、CI 和部署 tokenizer 上逐 ID 比较。
9. **做任务级测试。** 检查复制人名、数字、空格、换行和结构化输出，而不只测总体 token 数。
10. **报告 unknown rate。** 按语言、数据源和输入类型分桶，防止平均值掩盖局部失败。

一个最小日志表可以是

| 字段 | 示例 | 目的 |
| --- | --- | --- |
| raw input hash | SHA-256 | 固定输入字节 |
| normalization | NFC | 固定 Unicode 约定 |
| token strings | [low, er] | 核对边界 |
| token IDs | [2, 3] | 核对词表映射 |
| special tokens | BOS/EOS/PAD | 核对模型模板 |
| valid length | 2 of 4 | 核对 mask 与 loss |
| decode result | lower | 核对可逆性 |
| tokenizer version | name + revision | 排除版本漂移 |

## 结语

Tokenization 把连续、复杂、可能无限多样的字符串世界投影到一个有限的离散接口。它必须在覆盖率、序列长度、词表大小、可逆性和统计复用之间做选择。词级、字符级、byte-level 和子词级没有脱离任务的绝对赢家；它们改变的是模型看到的时间轴与错误单位。

最重要的边界有三条：

1. token 是离散 ID，不是 embedding，也不是有意义的整数坐标；
2. tokenizer 是模型 checkpoint 的一部分，规则、词表、special token 和版本必须绑定；
3. token 长度是计算预算，OOV、Unicode、padding 和 truncation 都会改变真实输入。

掌握这条 pipeline 后，BPE、WordPiece、SentencePiece 和 embedding 才能分别讨论：前者决定如何切，后者决定离散 token 如何进入连续表示空间。

## 相关词条

[BPE 分词](../text-representation/tokenization-bpe/)

[WordPiece 与 SentencePiece](../text-representation/wordpiece-and-sentencepiece/)

[One-hot 与分布式表示](../text-representation/one-hot-and-distributed/)

[嵌入](../text-representation/embeddings/)

[嵌入几何](../text-representation/embedding-geometry/)

[序列到序列](../rnn-lstm/seq2seq/)

[注意力复杂度](../attention/attention-complexity/)

[自注意力](../attention/self-attention/)

[交叉注意力](../attention/cross-attention/)
