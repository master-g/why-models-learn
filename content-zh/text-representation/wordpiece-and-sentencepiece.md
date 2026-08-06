---
title: "WordPiece 与 SentencePiece：从词片段到概率分词"
tags: ["why-models-learn"]
---

WordPiece 与 SentencePiece 都可以把词和字符之间的片段变成模型可处理的 token，但它们不是同一层级的名字。WordPiece 通常指一种子词分词模型和与之配套的训练、编码约定；SentencePiece 是一个可以直接处理原始句子的 tokenizer 工具包和模型格式，里面既可以训练 BPE，也可以训练 Unigram。Unigram 又是另一种以 token 概率和候选分词为中心的模型。

所以，准确的对照不是“BPE 对 SentencePiece”，而是：

1. **训练阶段**BPE 按当前语料中的 pair 频率逐轮合并；
2. **编码阶段**WordPiece 用似然相关的 pair 得分或目标改进来学习词片段，编码时常用最长匹配的贪心规则；
3. **输入协议**SentencePiece 提供原始字符串处理、空格协议和模型文件，模型类型可以是 BPE 或 Unigram；
4. **概率模型**Unigram 从一个较大的候选词表出发，用概率和动态规划选择或采样分词。

这篇把四个名字放回各自层级，用同一个 toy corpus 手算 WordPiece-style pair score，再用原始空格和 Unigram 概率展示为什么 tokenizer 不能只看一个算法名称。

![WordPiece 用归一化 pair 得分选择片段，SentencePiece 保留原始空格并可用 BPE 或 Unigram，最终都映射到 token ID](/assets/text-representation/svg/wordpiece-and-sentencepiece.1.svg)

## 先把三个名字放在正确层级

### BPE、WordPiece、SentencePiece 和 Unigram

| 名称 | 所处层级 | 训练信号 | 编码时的核心问题 |
| --- | --- | --- | --- |
| BPE | 子词 merge 模型 | 相邻 pair 的频率 | 固定 rank 中哪些 pair 先合并 |
| WordPiece | 子词模型与词表约定 | 似然相关得分或目标改进 | 当前前缀能否用一串合法 token 覆盖 |
| SentencePiece | tokenizer 工具包与模型格式 | 取决于配置的 BPE 或 Unigram | 原始句子的空格、Unicode 和分词如何统一 |
| Unigram | 概率式子词模型 | token 概率与句子似然 | 多种分词路径中哪一条概率最高或应被采样 |

BPE 与 WordPiece 可以共享“从字符或 byte 组成子词”的外形，但训练准则不同。SentencePiece 可以承载 BPE，因此看到一个 SentencePiece 模型文件时，还要继续检查 model type，不能只凭工具名称判断内部算法。

### 训练目标和推理算法是两条轴

一个 tokenizer 至少有两种行为需要分别记录：

| 行为 | 训练时的例子 | 推理时的例子 |
| --- | --- | --- |
| 如何产生或保留词表 | BPE 反复 merge，WordPiece 计算 pair score，Unigram 删除候选 | 不重新改变词表 |
| 如何切一个新字符串 | 统计 pair 或最大化句子概率 | 固定词表上的 greedy、Viterbi 或 sampling |

如果把训练时的 score 当成推理时的 score，或者把 SentencePiece 的工具名当成 Unigram 的算法名，就会把两个独立的协议混在一起。

## WordPiece：用相对得分挑 pair

### 一个可手算的 WordPiece-style score

WordPiece 的历史训练描述常用下面形式的 pair 得分来衡量“这个 pair 的共现是否超过两个单独符号的基准”：

$$
\operatorname{score}(a,b)
=
\frac{f(a,b)}{f(a)f(b)},
$$

其中 $f(a,b)$ 是相邻 pair 的语料频次，$f(a)$ 与 $f(b)$ 是两个符号在当前语料中的总频次。这个式子不是所有现代实现的完整训练目标，但它足以展示 WordPiece-style 选择与 BPE 频率选择为何会走向不同路径。

仍使用 BPE 词条中的 toy corpus：

| 词形 | 频次 | 初始序列 |
| --- | ---: | --- |
| 「low」 | 3 | 「l o w 〈/w〉」 |
| 「lower」 | 1 | 「l o w e r 〈/w〉」 |
| 「lowest」 | 1 | 「l o w e s t 〈/w〉」 |

所有词尾边界合计出现 5 次，l、o、w 也各出现 5 次，e 出现 2 次，r、s、t 各出现 1 次。由此得到部分 pair 分数：

| pair | pair 频次 | 符号频次 | score |
| --- | ---: | --- | ---: |
| (l,o) | 5 | 5 × 5 | 0.20 |
| (o,w) | 5 | 5 × 5 | 0.20 |
| (w,e) | 2 | 5 × 2 | 0.20 |
| (e,r) | 1 | 2 × 1 | 0.50 |
| (e,s) | 1 | 2 × 1 | 0.50 |
| (s,t) | 1 | 1 × 1 | 1.00 |

在这个 toy 统计里，BPE 会在 (l,o) 与 (o,w) 的频率 5 中按 tie-break 选一条；WordPiece-style score 则会优先选择 (s,t)，因为 s 和 t 各自都很少见，但它们一旦相邻就形成了非常集中的 pair：

$$
\operatorname{score}(s,t)
=
\frac{1}{1\times1}
=1
>
\frac{5}{5\times5}
=0.2
=
\operatorname{score}(l,o).
$$

于是第一条 WordPiece-style 规则是

$$
\text{s}+\text{t}
\longrightarrow
\text{st}.
$$

这个结果不是说 st 一定比 lo 更有语言意义，而是说明归一化得分对“组成部分本身是否常见”的惩罚不同。BPE 奖励绝对重复次数，WordPiece-style score 更关注 pair 相对于边缘频率的集中程度。

### score 不是完整的概率模型

如果把当前词表记为 $\mathcal V$，一个更高层的目标可以写成语料似然：

$$
\mathcal L(\mathcal V)
=
\sum_{s\in\mathcal D}
\log P_{\mathcal V}(s).
$$

实际 WordPiece trainer 可能用 pair score 作为高效近似、候选排序或词表增长准则，也可能在实现中加入最大词表、字符覆盖率、未知词约束和特殊 token 规则。阅读一个具体 tokenizer 时，应查它的训练器与模型文件语义，不要把上面的 toy score 当作所有库的字节级规范。

### 合并后统计对象会改变

如果选择 (s,t)→st，原来相邻的 (e,s) 与 (s,t) 会被 (e,st) 取代，(st,〈/w〉) 也成为新的候选。和 BPE 一样，每轮之后的统计必须针对新序列更新；不同的是“选哪一对”的分数不再只是 pair 的绝对频率。

## WordPiece 编码：训练 score 与推理 greedy

### continuation marker 表示词内延续

许多 WordPiece 词表用「##」前缀表示一个 token 只能出现在词的延续位置。例如在一个人为设定的词表中：

$$
\text{unhappiness}
\longrightarrow
[\text{un},\operatorname{cont}(\text{happi}),\operatorname{cont}(\text{ness})].
$$

这里的「##」不是两个普通字符的 merge 结果，而是词表和编码器约定的 continuation marker。词首的 un 与词内的 ##happi 不是同一个 token；如果把前缀去掉，ID 也不应自动相同。

### 最长匹配优先不是全局最优的保证

典型 WordPiece 编码器会从一个预分词片段的左端开始，寻找词表中能匹配当前前缀的最长 token；后续位置要求匹配带 continuation marker 的 token。一个简化流程是：

1. 在当前位置尝试最长词首 token；
2. 若成功，移动到剩余字符串；
3. 对剩余部分只接受合法的 continuation token；
4. 直到覆盖整个片段，否则返回 unknown。

设片段是 unhappiness，若词表同时有 un、unhappy、##happi 和 ##ness，最长匹配可能先取 unhappy；如果 ##ness 不能接上，则这个贪心路径可能失败，即使另一条较短的前缀路径可以完整覆盖。具体 tokenizer 是否回退、是否使用更复杂的搜索，要以实现为准。

### unknown 的粒度是接口选择

在许多 BERT 风格 WordPiece 配置中，只要一个预分词片段不能被合法 token 序列完整覆盖，整个片段会变成「[UNK]」，而不是把已经找到的前缀留下。于是一个字符只差异的拼写错误可能让整个词从多个 token 变成一个 unknown token。

如果词表增加字符级回退或 byte fallback，OOV 行为会改变：

| 回退策略 | 未知片段结果 | 信息保留 | 长度代价 |
| --- | --- | --- | --- |
| 整词 [UNK] | 一个 unknown token | 低 | 短 |
| 字符回退 | 若干字符 token | 中等 | 中到高 |
| byte 回退 | 若干 UTF-8 byte token | 高 | 可能很高 |

不能只看 WordPiece 这个名字判断 OOV 行为；必须检查词表、unknown 配置和编码器实现。

## SentencePiece：直接在原始句子上学习

### 空格被变成可学习的符号

SentencePiece 的重要设计是可以绕开外部的空格预分词器，直接把原始句子作为输入。常见协议先把空格替换为 U+2581 字符「▁」，再让模型在这个符号序列上学习。于是

$$
\text{Hello world}
\longrightarrow
\text{▁Hello▁world}.
$$

若最终词表没有把整段合并，输出可以表示为

$$
[\text{▁Hello},\text{▁world}].
$$

「▁」是 whitespace marker，不是普通的装饰空格。它让 token 本身携带词首信息，decode 时再按照 SentencePiece 的协议恢复普通空格。首词也可能带有这个标记，因此首词和后续词的 token 形式不必相同。

### raw sentence 并不等于不做预处理

SentencePiece 不依赖外部按空格切词，并不表示它完全不做 normalization。模型配置可能指定 Unicode normalization、字符覆盖率、用户定义符号、控制符号、空白处理和 byte fallback。要复现一个模型，必须同时保存：

1. 原始输入到规范化字符串的变换；
2. whitespace marker 的协议；
3. model type 与模型文件；
4. special token 和用户定义符号；
5. vocabulary 到 ID 的映射。

同一段可见文本若经过不同 NFKC、大小写或空白策略，SentencePiece 看到的符号序列仍会不同。

### SentencePiece 可以承载 BPE

SentencePiece 常见的模型类型包括 BPE 和 Unigram。若配置为 BPE，它仍然是固定 merge 规则的模型，只是输入边界和模型文件由 SentencePiece 规范处理；若配置为 Unigram，则训练和编码转为概率式候选分词。

| SentencePiece model type | 训练方式 | 编码方式 | 典型特征 |
| --- | --- | --- | --- |
| BPE | 学习有顺序的 merge | 按 merge 规则合并 | 与 BPE 词条的核心思想相同 |
| Unigram | 学习 token 概率并删除候选 | Viterbi 或 sampling | 一个字符串可保留多条候选路径 |

因此“这个模型用 SentencePiece”还不是完整答案；还要问“它的 model type 是 BPE 还是 Unigram”。

## Unigram：从候选词表中删除

### 从大词表开始

BPE 是从小 alphabet 向上增加 merge；Unigram 通常从一个较大的 seed vocabulary 开始，再删除对语料似然贡献较小的 token。每个 token $t$ 有一个概率 $p(t)$，对字符串 $s$ 的所有合法分词路径求和：

$$
P(s)
=
\sum_{z\in\operatorname{Seg}(s)}
\prod_{t\in z}p(t).
$$

训练可以用 EM 一类方法估计 token 概率，再用损失增量或似然影响删除候选，直到达到目标词表大小。它学习的是“哪些候选 token 组成的概率模型能解释语料”，不是“下一次应该合并哪个 pair”。

### 用 abab 手算 Viterbi

设 toy vocabulary 和概率如下：

| token | 概率 |
| --- | ---: |
| a | 0.40 |
| b | 0.30 |
| ab | 0.25 |
| aba | 0.05 |

概率总和为 1。字符串 abab 的合法分词路径及其乘积为：

| 分词路径 | 概率乘积 |
| --- | ---: |
| a b a b | 0.40 × 0.30 × 0.40 × 0.30 = 0.0144 |
| a b ab | 0.40 × 0.30 × 0.25 = 0.0300 |
| a ab b | 0.40 × 0.25 × 0.30 = 0.0300 |
| ab a b | 0.25 × 0.40 × 0.30 = 0.0300 |
| ab ab | 0.25 × 0.25 = 0.0625 |
| aba b | 0.05 × 0.30 = 0.0150 |

所以全部路径的字符串概率是

$$
P(\text{abab})
=
0.0144+0.03+0.03+0.03+0.0625+0.015
=0.1819.
$$

如果推理采用 Viterbi 最优路径，则选择乘积最大的 ab ab，因为

$$
0.0625
>
0.03
>
0.015
>
0.0144.
$$

实际计算通常使用对数概率，避免长序列乘积下溢：

$$
z^\star
=
\operatorname*{arg\,max}_{z\in\operatorname{Seg}(s)}
\sum_{t\in z}\log p(t).
$$

### 动态规划而不是只取最长 token

对字符位置 $j$ 定义到达该位置的最佳对数概率：

$$
\begin{gathered}
\operatorname{dp}[0]=0,\\
\operatorname{dp}[j]
=
\max_{\substack{0\le i<j\\s[i:j]\in\mathcal V}}
\left(
\operatorname{dp}[i]+\log p(s[i:j])
\right).
\end{gathered}
$$

这条递推会比较所有能覆盖当前位置的候选 token。最长 token 可能因为概率很低而不是最优路径；很短的 token 组合也可能胜出。若开启 sampling，系统不是每次都取同一条 Viterbi 路径，而是从后验或近似分布中采样，给同一字符串保留多种合理 segmentations。

## 同一输入上的三种子词模型

### 训练信号不同，切分结果也可能不同

以带词边界的 lowest 为例，三个模型可能给出：

| 模型 | 可能的输出 | 决定因素 |
| --- | --- | --- |
| BPE | low、e、s、t、边界 | merge rank 是否已经生成 low |
| WordPiece | low、##est | 词表中是否存在合法 continuation token，greedy 能否完整覆盖 |
| SentencePiece BPE | ▁low、est 或更长片段 | raw whitespace marker 与 BPE merge |
| SentencePiece Unigram | ▁low、est 或 ▁lowest | 所有候选路径的 token 概率 |

这些输出都可能是正确的 tokenizer 结果；模型参数只对其中一种 token-ID 接口训练过，不能直接互换。

### WordPiece 的 continuation 与 SentencePiece 的 whitespace 是不同信息

WordPiece 的「##」回答“这个 token 是否位于词内延续位置”；SentencePiece 的「▁」回答“这个 token 是否携带一个词首空格”。前者常附在 continuation token 的字符串表示上，后者常直接作为 token 内的字符。两者都在编码层携带边界，但方向不同：

| 标记 | 所在位置 | 主要含义 | decode 时的任务 |
| --- | --- | --- | --- |
| ## | WordPiece continuation token 的前缀 | 不是新词的开始 | 去掉标记并拼接词内片段 |
| ▁ | SentencePiece token 的词首符号 | 前面有一个空格或句首边界 | 把标记还原为空格或边界 |
| 〈/w〉 | 某些 BPE 词尾协议 | 词在这里结束 | 去掉边界并恢复词间关系 |

如果把 ## 当作普通字符合并，或把 ▁ 当作可随意删除的空格，round-trip 就会改变。

## 空格、规范化和 Unicode

### 预分词边界是模型的一部分

WordPiece 通常先依赖 whitespace 或 punctuation pre-tokenizer，再对每个片段执行 continuation 规则；SentencePiece 可以直接把空格编码进符号流；BPE 既可以使用词尾边界，也可以使用 byte-level 的前置空格标记。三种方案对同一个输入的第一步就可能不同：

| 输入协议 | 中间表示示意 | 可见 token 可能包含 |
| --- | --- | --- |
| 先按空格切词 | hello、world | 词首 token 与 continuation token |
| 词尾边界 | hello〈/w〉、world〈/w〉 | 词尾边界与跨词阻断 |
| SentencePiece raw | ▁hello▁world | 词首空格 marker |
| byte-level | UTF-8 byte 与前置空格协议 | byte 片段与 byte merge |

因此，token 长度统计必须注明 tokenizer 的边界协议，不能只写“使用子词分词”。

### Unicode normalization 可能在分词前改变字符串

设 normalization 为 $N$，编码器输入为

$$
s'=N(s).
$$

若 é 的预组合和 e 加组合重音在 $N$ 下被合并，它们可能共享 token；若一个配置保留 code point 差异，BPE pair、WordPiece coverage 或 Unigram 候选都会不同。SentencePiece 的 raw input 解决的是“由 tokenizer 统一处理空格和片段”，不等于绕开 Unicode normalization。

## 词表、ID 与 checkpoint 绑定

### token 字符串不是 token ID

WordPiece 的 un、##happi，SentencePiece 的 ▁Hello，或 BPE 的 low 都只是词表中的字符串。模型接收的是

$$
(t_1,\ldots,t_L)
\longrightarrow
(x_1,\ldots,x_L),
\qquad
x_i=\operatorname{id}(t_i).
$$

如果同一个 token 在两个 vocabulary 中的 ID 不同，embedding 查表行也不同。ID 的整数大小不代表 token 的语义距离；它只是 checkpoint 约定的索引。

### 词表变化会改变参数接口

embedding 矩阵 $E\in\mathbb R^{V\times d}$ 的参数量为

$$
N_E=Vd.
$$

增加一个 WordPiece continuation token、一个 SentencePiece whitespace 变体或一个 byte fallback token，都会增加词表行；若输出层与输入 embedding 绑定，输出接口也会变化。更危险的是保持 $V$ 不变却重排 ID：形状不变，但每一行的语义已经错位。

部署时应把下面的组合视为一个版本化接口：

$$
\Theta_{\text{tok}}
=
(N,P,M,\mathcal V,S,T),
$$

其中 $N$ 是 normalization，$P$ 是 pre-tokenization 或 raw-input 处理，$M$ 是 BPE、WordPiece 或 Unigram model，$\mathcal V$ 是 vocabulary，$S$ 是 special-token 协议，$T$ 是 padding、truncation 和模板后处理。只要其中一项变化，就应重新跑固定样本的 token、ID、mask 和 decode golden test。

## 长度与概率的权衡

### 更短的序列不等于更好的分词

更大的子词 token 可能让序列长度下降，attention 的 score map 规模从 $L^2$ 下降；但更大的词表会增加 embedding 和输出层参数。一个 tokenizer 的目标不是单独最小化 token 数，而是在以下量之间取平衡：

| 选择 | 可能的收益 | 可能的代价 |
| --- | --- | --- |
| 增加常见长 token | 常见文本更短 | 新域覆盖不足，词表参数增加 |
| 保留更多短 token | 组合能力和回退路径更强 | 序列更长，padding 与 attention 成本增加 |
| Unigram 保留多条候选 | 能表达分词不确定性 | 训练、解码和复现更复杂 |
| WordPiece continuation 约束 | 词内边界明确 | 片段无法完整覆盖时可能整词 [UNK] |

如果一个输入的 token 长度从 $L$ 变为 $2L$，简化的注意力 map 元素从 $L^2$ 变为

$$
(2L)^2=4L^2.
$$

tokenizer 的边界选择会因此进入模型的显存和吞吐账本，而不只是文本清洗细节。

### Unigram 的概率还提供了不确定性

BPE 和典型 WordPiece greedy 编码对同一个固定配置通常给出一条确定路径；Unigram 可以比较多条合法路径，并在需要时采样。这个能力可以用于 subword regularization，但也意味着：

1. 训练中若启用 sampling，同一个字符串可能有多个 token 序列；
2. 评估和部署必须明确使用 Viterbi 还是 sampling；
3. cache key、padding 和日志不能假设 tokenization 永远唯一；
4. 采样随机性不应被误判为模型参数或输入发生变化。

## 失效模式：名称相似，协议却不兼容

### 1. 把 SentencePiece 当成单一算法

SentencePiece 可能承载 BPE 或 Unigram。只记录“使用 SentencePiece”不够，必须记录 model type、模型文件、normalizer 和 vocabulary。

### 2. 把 WordPiece score 当成编码规则

训练时的 pair score 用于学习或排序候选；推理时常用 greedy longest-match 和 continuation 约束。加载 tokenizer 时不能重新按当前输入的 score 选择 pair。

### 3. 把 ##、▁ 和 〈/w〉 当成同一种边界

三者分别表示 continuation、词首空格和词尾边界。删除或移动一个标记会改变 token 字符串、decode 和词表 ID。

### 4. 用最长匹配替换 Unigram Viterbi

Unigram 的最优路径取决于 token 概率。最长 token 可能概率很低；只实现 longest-match 会得到另一个模型。

### 5. 只复制词表，遗漏 normalizer 或 model file

相同 token 字符串并不保证相同输入。Unicode、空格和 BPE merge rank 或 Unigram 概率缺一项，最终序列就可能漂移。

### 6. 允许 WordPiece 片段不完整地回退

如果参考实现对不可完整覆盖的预分词片段输出整词 [UNK]，而复现实现保留已匹配前缀再逐字符回退，两者的 token 长度和语义输入都不同。

### 7. 把 Unigram 的采样结果当作确定性 golden

sampling 的随机种子、temperature 或 n-best 约束会改变分词路径。评估 golden 应锁定 Viterbi 或同时记录随机配置。

### 8. 只更新 tokenizer，不更新 checkpoint

新增 continuation token、whitespace token 或 byte token 会改变 ID 空间。tokenizer 通过测试不意味着旧模型理解新的 embedding 行。

### 9. 看到相同 token 数就认为兼容

两个 tokenizer 可能都输出长度 8，但 token 字符串、ID、special-token shift 和 attention mask 完全不同。兼容性检查必须逐 token 对照，而不是只比较长度。

## 最小审计协议

审计一个 WordPiece 或 SentencePiece 实现时，可以固定以下证据链：

1. 记录 tokenizer 类型、版本、model type 和完整配置；
2. 用带前导空格、连续空格、标点、换行、非 ASCII 字符和未知片段的样本；
3. 保存 normalization 与 raw/pre-tokenization 的中间文本；
4. 输出边界标记，包括 ##、▁ 或词尾边界；
5. WordPiece 重放最长匹配和 unknown 规则；
6. SentencePiece BPE 重放 merge rank，SentencePiece Unigram 重放候选 token 的 DP；
7. 逐 token 查 vocabulary，核对 ID 与 special-token 位置；
8. 比较 encode、decode、padding、truncation 和 attention mask；
9. 对 Unigram 明确 Viterbi、sampling、随机种子与 n-best 配置；
10. 保存 tokenization 配置和 checkpoint 的版本指纹。

一个小型 golden 表可以同时覆盖三种语义：

| 测试输入 | 需要锁定的证据 | 失败时先查 |
| --- | --- | --- |
| lower | BPE rank、WordPiece continuation、Unigram 候选 | model type 与 vocabulary |
| Hello world | 空格 marker、decode 空格恢复 | raw-input 与 normalizer |
| 罕见 Unicode | normalization、byte fallback、UNK | coverage 与 fallback |
| abab | Unigram 各路径概率与 Viterbi | token 概率和 DP |

对于 Unigram toy 例子，至少应复算

$$
\operatorname*{arg\,max}
\left\{
0.0144,\,
0.03,\,
0.03,\,
0.03,\,
0.0625,\,
0.015
\right\}
=0.0625,
$$

并确认获胜路径是 [ab,ab]，而不是因为实现默认的最长匹配才得到它。

## 结语

WordPiece、SentencePiece 和 Unigram 的共同目标是用有限词表覆盖开放文本，但它们把“什么是好 token”写进了不同的统计协议。WordPiece-style score 关注 pair 相对边缘频率的集中程度，典型编码器还要遵守 continuation marker 和完整覆盖；SentencePiece 把空格与原始字符串处理纳入 tokenizer，并可选择 BPE 或 Unigram；Unigram 则把分词看成概率路径，在候选分词之间做 Viterbi 或 sampling。

真正需要绑定 checkpoint 的不是一个算法名称，而是完整的 tokenizer 接口：normalization、边界、model type、merge 或概率、词表、ID、special token 和后处理。只有把这些证据逐层固定，才能解释同一字符串为什么在不同模型里变成不同 token，也才能在更换 tokenizer 时知道模型究竟被改变了哪一部分。

## 相关词条

[Tokenization](../text-representation/tokenization/)

[BPE 分词](../text-representation/tokenization-bpe/)

[One-hot 与分布式表示](../text-representation/one-hot-and-distributed/)

[嵌入](../text-representation/embeddings/)

[嵌入几何](../text-representation/embedding-geometry/)

[词表大小权衡](../text-representation/vocabulary-size-tradeoffs/)

[自注意力](../attention/self-attention/)
