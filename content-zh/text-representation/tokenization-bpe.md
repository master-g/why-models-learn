---
title: "BPE 分词：用频繁合并学习子词词表"
tags: ["why-models-learn"]
---

BPE（byte pair encoding）在分词里不是把一段文字简单压缩成更短的字符串，而是从一个很小的初始符号集合出发，反复统计相邻符号对，把最常见的一对合并成新 token，并把每次合并记录成有顺序的规则。训练完成后，推理时不再重新统计频率，而是按这份固定的合并规则处理新输入。

因此 BPE 有两个必须分开的对象：

1. **训练阶段**学习哪些相邻符号应该成为可复用的子词；
2. **编码阶段**把新字符串按已经学习好的 merge list 确定性地切成 token。

这篇用一个可以手算的 toy corpus 展开这两个阶段，再把 token、merge rank、词表 ID、byte-level 变体和上下文成本接起来。全文中的边界标记「〈/w〉」只是协议符号，表示一个词的结尾；它不是输入文本本身。

![BPE 从带词边界的训练语料统计 pair，学习有顺序的 merge，再把新词编码为 token 和 ID](/assets/text-representation/svg/tokenization-bpe.1.svg)

## BPE 的两个阶段不能混为一谈

### 训练阶段：从频率得到规则

设训练语料中出现了若干个预分词片段。对每个片段，先把它展开成初始符号序列。初始符号可以是字符，也可以是 byte；在这篇的 toy 例子里使用字符，并在每个词末尾加上「〈/w〉」。

对相邻符号对 $(a,b)$ 统计加权频率：

$$
c(a,b)=
\sum_{w\in\mathcal D}
f(w)
\sum_i
\mathbf 1\big[(s_i,s_{i+1})=(a,b)\big],
$$

其中 $\mathcal D$ 是训练片段集合，$f(w)$ 是片段 $w$ 的出现次数，$s_i$ 是当前符号序列中的第 $i$ 个符号。选出频率最大的 pair：

$$
(a^\star,b^\star)
\in
\operatorname*{arg\,max}_{(a,b)}c(a,b),
$$

然后把所有当前序列中的相邻 $a^\star,b^\star$ 替换为一个新符号 $v^\star=a^\star b^\star$。重新计数，再进行下一轮。

每轮的产物不是只有一个新 token，还包括它在规则列表中的顺序：

$$
\mathcal M=
\big[
(a_1,b_1)\to v_1,\,
(a_2,b_2)\to v_2,\,
\ldots,\,
(a_R,b_R)\to v_R
\big].
$$

这个顺序就是 merge rank。它表示规则之间的优先级，不等于 token 在词表里的整数 ID。

### 编码阶段：用固定规则处理新输入

推理时给定训练后固定的 $\mathcal M$。新输入先经过相同的规范化和预分词，再展开成初始符号。随后只在当前序列中寻找已经存在于 $\mathcal M$ 的 pair；如果多个 pair 都可合并，优先使用 rank 更小的规则：

$$
(a_i,b_i)
\prec
(a_j,b_j)
\quad\Longleftrightarrow\quad
i<j.
$$

合并后，序列长度可能下降，新的相邻 pair 可能因此出现。重复这个过程直到没有可用规则。编码时不应该重新查看新输入的 pair 频率，也不应该把新发现的高频 pair 临时加入模型。

| 阶段 | 输入 | 主要操作 | 输出 |
| --- | --- | --- | --- |
| 训练 | 带频次的语料片段 | 统计当前 pair，选择最高频 pair 并合并 | merge list、词表候选、规则顺序 |
| 编码 | 新字符串与固定 tokenizer 配置 | 按 rank 逐步应用已知 merge | token 序列 |
| ID 映射 | token 序列与固定 vocabulary | 查 token 到整数的映射 | 模型输入 ID |

如果把训练和编码混在一起，同一个字符串在不同请求中可能因为 batch 内容不同而得到不同 token；这就不再是一个稳定的模型接口。

## 为什么要保留词边界

### 没有边界时，跨词合并会改变协议

假设语料中有「low」和「er」，又有另一个词恰好产生了相邻的「w」「e」。如果训练序列直接拼接为

$$
\text{l}\ \text{o}\ \text{w}\ \text{e}\ \text{r},
$$

统计器无法仅凭相邻字符知道「w」和「e」是否来自同一个词。某次 merge 可能学到跨词片段，编码结果就依赖预分词阶段是否保留空格。

常见的做法是在词尾加入一个边界符号：

$$
\text{low}
\longrightarrow
[\text{l},\text{o},\text{w},\langle/\text{w}\rangle].
$$

在本文的 Markdown 正文里写作「〈/w〉」，是为了避免它被解释成 HTML 标签；数学含义相同。这样「w」与词尾的边界可以成为一个可学习的 pair，但它不会和下一个词的首字符相邻。

### 空格也可以是显式协议

不同 tokenizer 不一定使用词尾标记。有些 byte-level 方案把空格编码成特殊 byte 或可见的空格标记，并让下一个词的首 token 带上这个标记。下表中的符号只是说明协议差异：

| 边界协议 | 片段示意 | 可能学习到的 token | 风险 |
| --- | --- | --- | --- |
| 词尾标记 | 「hello〈/w〉 world〈/w〉」 | 「hello〈/w〉」 | 需要在 decode 时正确移除边界 |
| 前置空格标记 | 「hello Ġworld」 | 「Ġworld」或更长片段 | 首词与后续词的 token 形式不同 |
| 原始空格 | 「hello world」 | 空格可能单独成 token | 规范化空格时容易改变序列 |
| 不保留边界 | 「helloworld」 | 允许跨词片段 | 词边界信息可能不可恢复 |

边界符号不是装饰。它决定哪些 pair 有资格在训练中反复出现，也决定 decode 是否有机会恢复词间空格。复现 tokenizer 时必须把边界协议和 merge list 一起保存。

## Toy corpus：手算前几次 merge

### 训练语料

现在只用三个词形：

| 词形 | 频次 | 初始序列 |
| --- | ---: | --- |
| 「low」 | 3 | 「l o w 〈/w〉」 |
| 「lower」 | 1 | 「l o w e r 〈/w〉」 |
| 「lowest」 | 1 | 「l o w e s t 〈/w〉」 |

频次表示训练语料中该片段被看到多少次；它会直接乘到 pair 的贡献上。我们暂时不加入特殊 token，也不把词表 ID 混进统计过程。

### 第 1 次：先合并 l o

初始 pair 的加权计数如下：

| pair | 计数来源 | 总频次 |
| --- | --- | ---: |
| (l,o) | 「low」的 3 次 + 「lower」的 1 次 + 「lowest」的 1 次 | 5 |
| (o,w) | 三种词形都出现 | 5 |
| (w,〈/w〉) | 「low」的 3 次 | 3 |
| (w,e) | 「lower」与「lowest」各 1 次 | 2 |
| (e,r) | 「lower」的 1 次 | 1 |
| (r,〈/w〉) | 「lower」的 1 次 | 1 |
| (e,s) | 「lowest」的 1 次 | 1 |
| (s,t) | 「lowest」的 1 次 | 1 |
| (t,〈/w〉) | 「lowest」的 1 次 | 1 |

(l,o) 与 (o,w) 都是最高频 pair。真实实现必须规定 tie-break；为了让手算结果确定，假定同频时按训练扫描顺序选择先出现的 (l,o)。于是第 1 条规则是

$$
\text{l}+\text{o}
\longrightarrow
\text{lo}.
$$

合并后，三个词形变成：

| 词形 | 第 1 轮后的序列 |
| --- | --- |
| 「low」 | 「lo w 〈/w〉」 |
| 「lower」 | 「lo w e r 〈/w〉」 |
| 「lowest」 | 「lo w e s t 〈/w〉」 |

注意：「lo」是一个新符号，但它还没有 ID。它首先只是 merge 过程中的新词表候选。

### 第 2 次：合并 lo w

现在 (lo,w) 在三种词形中各出现一次，并按频次加权得到 5。它成为下一条规则：

$$
\text{lo}+\text{w}
\longrightarrow
\text{low}.
$$

序列更新为：

| 词形 | 第 2 轮后的序列 |
| --- | --- |
| 「low」 | 「low 〈/w〉」 |
| 「lower」 | 「low e r 〈/w〉」 |
| 「lowest」 | 「low e s t 〈/w〉」 |

到这里，「low」这个片段在三个词形中共享。它既可以在单独的「low」里使用，也可以作为「lower」和「lowest」的共同前缀。

### 第 3 次：合并 low 与词尾边界

当前 (low,〈/w〉) 只来自「low」，但「low」出现了 3 次，因此计数为 3。其他候选 pair 的计数都不超过 1，于是得到：

$$
\text{low}+\langle/\text{w}\rangle
\longrightarrow
\text{low}\langle/\text{w}\rangle.
$$

第 3 轮后的序列为：

| 词形 | 第 3 轮后的序列 |
| --- | --- |
| 「low」 | 「low〈/w〉」 |
| 「lower」 | 「low e r 〈/w〉」 |
| 「lowest」 | 「low e s t 〈/w〉」 |

对应的前三条 merge list 是：

| rank | pair | 新 token | 训练时的加权频次 |
| ---: | --- | --- | ---: |
| 1 | (l,o) | 「lo」 | 5 |
| 2 | (lo,w) | 「low」 | 5 |
| 3 | (low,〈/w〉) | 「low〈/w〉」 | 3 |

第 1 条和第 2 条的频次相同，但 rank 不同。这个差异在编码阶段很重要：rank 是规则的先后次序，不能只保存每条规则的频次而丢掉顺序。

## 新输入如何按 merge rank 编码

### 编码 lower

新词「lower」初始展开为

$$
[\text{l},\text{o},\text{w},\text{e},\text{r},\langle/\text{w}\rangle].
$$

当前可用的第 1 条规则是 (l,o)，所以先得到

$$
[\text{lo},\text{w},\text{e},\text{r},\langle/\text{w}\rangle].
$$

接着第 2 条规则 (lo,w) 可用，得到

$$
[\text{low},\text{e},\text{r},\langle/\text{w}\rangle].
$$

第 3 条规则要求相邻的「low」和边界符号，但它们之间还有「e」「r」，因此不能应用。最终编码是

$$
\operatorname{BPE}_{\mathcal M}(\text{lower})
=
[\text{low},\text{e},\text{r},\langle/\text{w}\rangle].
$$

这里的结果说明：「low」已经成为共享子词，但并不意味着所有以「low」开头的词都会被合并成一个完整 token。

### 编码 low

「low」的初始序列为

$$
[\text{l},\text{o},\text{w},\langle/\text{w}\rangle].
$$

先应用 rank 1，再应用 rank 2，得到 [low,〈/w〉]；此时 rank 3 也可应用。因此最终 token 是

$$
\operatorname{BPE}_{\mathcal M}(\text{low})
=
[\text{low}\langle/\text{w}\rangle].
$$

这与「lower」的前缀「low」不是同一个 token。一个带词尾边界的完整 token 和一个还要继续接字符的前缀 token，承担不同的复用角色。

### 编码 lowest

同样地，

$$
[\text{l},\text{o},\text{w},\text{e},\text{s},\text{t},\langle/\text{w}\rangle]
\longrightarrow
[\text{low},\text{e},\text{s},\text{t},\langle/\text{w}\rangle].
$$

因为「low」后面紧接「e」，rank 3 不能应用，最终为

$$
\operatorname{BPE}_{\mathcal M}(\text{lowest})
=
[\text{low},\text{e},\text{s},\text{t},\langle/\text{w}\rangle].
$$

三种输入的对比如下：

| 输入 | 初始长度 | 最终 token 序列 | token 数 |
| --- | ---: | --- | ---: |
| 「low」 | 4 | [low〈/w〉] | 1 |
| 「lower」 | 6 | [low,e,r,〈/w〉] | 4 |
| 「lowest」 | 8 | [low,e,s,t,〈/w〉] | 5 |

边界符号是否单独计为一个 token，取决于实际 tokenizer 是否把它和前面的片段合并。这里的计数只用于说明 toy merge 结果，不应直接当作某个生产 tokenizer 的长度规则。

### 重叠 pair 与确定性

考虑符号序列 [a,a,a]，规则中有 (a,a)→aa。两个相邻位置都匹配同一条规则，但它们重叠。实现需要规定扫描策略，例如从左到右只合并不重叠的 pair：

$$
[a,a,a]
\longrightarrow
[aa,a],
$$

而不是同时把两个位置都替换。真实 BPE 编码器通常把当前序列维护成可更新的结构，并在每次合并后更新受影响的邻居；无论具体实现如何，左右 tie-break 和重叠处理都必须与训练或参考实现一致。

## BPE 词表、merge list 与 ID

### 三个文件承担三种职责

一个可运行的 BPE tokenizer 至少需要区分：

| 对象 | 例子 | 回答的问题 |
| --- | --- | --- |
| 初始 alphabet | 「l」「o」「w」「e」「r」「〈/w〉」 | 输入最开始能拆成哪些符号 |
| merge list | (l,o)→lo、(lo,w)→low | 哪些 pair 可以合并，以及先后顺序是什么 |
| vocabulary | low→6、e→4 | 每个最终 token 映射到哪个 ID |

训练时生成的「lo」「low」等候选要进入最终 vocabulary，才能在编码后查 ID。special token 也要占用词表中的固定位置；是否从 0 开始只是实现约定。

### toy vocabulary 的 ID 映射

假定为了展示，词表中相关条目是：

| token | ID | 来源 |
| --- | ---: | --- |
| 「〈/w〉」 | 0 | 边界协议 |
| 「l」 | 1 | 初始 alphabet |
| 「o」 | 2 | 初始 alphabet |
| 「w」 | 3 | 初始 alphabet |
| 「e」 | 4 | 初始 alphabet |
| 「r」 | 5 | 初始 alphabet |
| 「low」 | 6 | rank 2 merge |
| 「low〈/w〉」 | 7 | rank 3 merge |

注意这个表是人为安排的：「o」的 ID 是 2，「low」的 ID 是 6，并不表示 rank 2 的新 token 必然得到 ID 2。实际实现可能先放 special token，再放 alphabet，最后按内部顺序加入 merge token。

于是「lower」的 token 到 ID 变换是

$$
[\text{low},\text{e},\text{r},\langle/\text{w}\rangle]
\longrightarrow
[6,4,5,0].
$$

模型最终接收的是 ID 序列和由它派生的 mask，而不是 merge rank。若只复制 [6,4,5,0] 却没有复制 token 到 ID 的完整映射，其他字符串就无法可靠编码。

## BPE 训练的统计与停止条件

### 频率是当前状态下的频率

第 1 轮合并后，原始 pair 已经被替换，下一轮统计的对象不再是原始字符序列。例如 (l,o) 合并为「lo」后，下一轮寻找的是 (lo,w)，不是把第 1 轮的计数原封不动地继续相加。

如果只维护 pair 频率而不更新邻居，会出现陈旧计数。合并一个位置后，真正可能改变计数的 pair 只在合并点附近；高效实现会增量更新这些邻居，但最终结果必须等价于从当前序列重新计数。

### 什么时候停止

常见停止条件有三种：

| 停止方式 | 约束 | 影响 |
| --- | --- | --- |
| 固定 merge 次数 | 进行预设的 $R$ 轮 | 词表大小容易规划，频率阈值不固定 |
| 目标词表大小 | 达到 $V_{\text{target}}$ | 需要计算初始 alphabet、special token 与 merge token 的预算 |
| 最小 pair 频率 | 最高频 pair 低于阈值 | 低频 merge 不再值得增加词表 |

忽略冲突和去重时，可以用近似式理解词表预算：

$$
V\approx A+S+R,
$$

其中 $A$ 是初始 alphabet 大小，$S$ 是 special token 数，$R$ 是实际保留的 merge 数。生产 tokenizer 还可能把未使用候选剔除，或把多个边界形式分别计入词表，所以这是预算直觉，不是文件格式的严格等式。

### 频次加权会改变 merge

如果把 toy corpus 的频次都改成 1，「(l,o)」和「(o,w)」的计数都变为 3，「(w,〈/w〉)」变为 1，前两轮仍需要 tie-break。若再加入大量「walk」，「(w,a)」的计数可能超过其他 pair，训练路径就会改变。BPE 学的是语料分布中的重复邻接结构，不是一个脱离语料的语言学规则。

因此训练记录应至少保存：

1. 训练语料版本和预分词配置；
2. 初始 alphabet 与边界协议；
3. pair 计数的权重定义；
4. tie-break 规则；
5. merge 次数或目标词表大小；
6. 最终 merge list 与 vocabulary。

## Byte-level BPE：把初始 alphabet 换成 byte

### 为什么从 byte 出发

字符级初始 alphabet 可能遇到未见脚本、罕见符号或 Unicode 组合形式。byte-level BPE 先把文本编码为 UTF-8 bytes，再在 byte 序列上学习 merge。UTF-8 使用有限的 byte alphabet，因此理论上可以为任意输入提供回退路径，而不必为每个 Unicode code point 都建立初始 token。

抽象地写：

$$
\text{字符串}
\xrightarrow{\text{UTF-8}}
(b_1,b_2,\ldots,b_n)
\xrightarrow{\text{BPE merges}}
(t_1,t_2,\ldots,t_L).
$$

byte-level 的覆盖性很强，但 byte 不是用户感知的字符。一个汉字、emoji 或组合字形可能需要多个 byte；若语料里没有学到有用的跨 byte merge，序列长度会明显增加。

| 初始单位 | 覆盖性 | 单位与人类直觉的距离 | 常见代价 |
| --- | --- | --- | --- |
| Unicode code point | 受词表覆盖限制 | 中等 | 罕见 code point 需要 UNK 或扩词表 |
| 字符 | 受定义与语言影响 | 较近 | 组合字符与多脚本规则复杂 |
| UTF-8 byte | 任意有效 UTF-8 输入可拆 | 较远 | 非拉丁文本可能占更多 token |
| 子词 | 依赖训练语料 | 常较近 | 新域、新脚本会暴露 merge 覆盖缺口 |

byte fallback 并不保证长度最优。它保证的是“还能表示”，不是“每个字符只占一个 token”。

### byte-level 也需要边界和空格协议

把字符换成 byte 不会自动解决空格、规范化和特殊 token 问题。下列变换仍可能改变最终 token：

$$
\text{NFC}
\neq
\text{NFKC},
\qquad
\text{保留空格}
\neq
\text{折叠空格}.
$$

如果训练时把前导空格编码进 byte-level token，推理时删除前导空格，后续 pair 就可能全部错位。byte-level 只改变初始 alphabet，不会替 tokenizer 定义完整输入协议。

## 空格、标点和边界协议

### 标点可以成为高频独立单位

BPE 的 pair 统计会把标点当作普通符号，除非 pre-tokenizer 先把它隔离。例如语料中频繁出现「word,」，可能学习到带标点的片段；如果「word」与「,」总是被拆开，则它们可能分别进入词表。两种方案都可能有效，关键是训练和推理必须相同。

| 预分词策略 | 「hello,」的候选 | 可能的复用 | 需要检查的地方 |
| --- | --- | --- | --- |
| 标点隔离 | 「hello」「,」 | 词与标点跨句复用 | 空格和标点的重组 |
| 原样保留 | 「hello,」 | 常见搭配可合并 | 罕见词后缀的 OOV 风险 |
| byte-level 连续 | UTF-8 byte 流 | 任意输入有回退 | 标点前后的空格标记 |

### decode 是编码的另一半

如果最终 token 只保存了字母片段，却没有保存边界或空格信息，decode 可能把两个不同字符串都还原成同一结果。例如「hello world」与「helloworld」的 token 序列若完全相同，编码就不是可逆的。实际 tokenizer 可能接受有限不可逆性，但这必须是明确取舍，而不是排查时才发现。

最小的 round-trip 检查是：

$$
D(E(s))=s
$$

至少应在包含前导空格、连续空格、换行、标点、非 ASCII 字符和特殊 token 字符串的样本上检查。若系统的 decode 会做规范化，等式应改成与协议一致的规范化等价关系，而不是假装逐字符相等。

## 长度、词表和语义边界的交换

### merge 越多不等于一定越好

增加 merge 规则通常会让常见片段变长、token 数下降，但会同时增加词表行数和 embedding 参数。若词表大小为 $V$，embedding 宽度为 $d$，仅 embedding 矩阵的参数量就是

$$
N_E=Vd.
$$

当 $d=4096$ 时，词表每增加 10000 行，就增加

$$
10000\times4096=40960000
$$

个 embedding 参数；输出投影与词表绑定时，相关参数和显存成本也可能一起增长。

另一方面，序列长度 $L$ 会影响注意力的二次项。一个简化的 self-attention score map 大小是

$$
L^2.
$$

因此 tokenizer 在词表参数与上下文长度之间做的是系统级权衡：

| 更大的 merge 词表 | 更小的 merge 词表 |
| --- | --- |
| 常见词形 token 更长，序列可能更短 | 词表与 embedding 较小 |
| embedding 和输出层可能更大 | 罕见词形需要更多 token |
| 新域可能遇到更多未复用 token | 长序列增加 padding 与 attention 成本 |
| token 边界更依赖训练语料 | 字符或 byte 回退更常见 |

### token 边界不是语言学真理

「unhappiness」可能被切成「un」「happiness」，也可能被切成「un」「happy」「ness」，还可能出现 byte 片段。哪一种切分更好，取决于训练语料、目标任务、词表预算和模型是否能在后续层重新组合它们。不能把一个 tokenizer 的 token 边界直接当成模型内部已经承诺的语素分析。

## 版本漂移：同一文本不再是同一 token

tokenizer 是 checkpoint 的一部分。只更新 merge list 而保留旧 embedding，会造成 token 到向量的错配；只更新 vocabulary 而不更新 merge list，则编码仍可能产生旧 token；只更新 normalization，则相同字符可能在更早阶段分流。

可以把输入接口写成配置的组合：

$$
\Theta_{\text{tok}}=
(N,P,A,\mathcal M,\mathcal V,S,T),
$$

其中 $N$ 是 normalization，$P$ 是 pre-tokenization，$A$ 是初始 alphabet，$\mathcal M$ 是 merge list，$\mathcal V$ 是 vocabulary，$S$ 是 special-token 协议，$T$ 是后处理模板。只有 $\Theta_{\text{tok}}$ 全部一致，才有理由期待

$$
E_{\Theta_{\text{tok}}}(s)
=
E_{\Theta'_{\text{tok}}}(s).
$$

部署审计可以选一组固定样本，保存每个阶段的中间结果：

| 阶段 | 应保存的证据 | 漂移症状 |
| --- | --- | --- |
| normalization | 规范化前后字符串或 code point | 视觉相同文本分流 |
| pre-tokenization | 候选片段与边界 | pair 候选集合改变 |
| BPE model | token 字符串序列与 merge rank | token 数或边界改变 |
| vocabulary | token、ID、词表版本 | embedding 行错位 |
| post-processing | BOS/EOS、padding、mask、模板 | batch 输入 shape 改变 |

## 失效模式：merge 看似确定，边界仍可能错

### 1. 把训练时的高频 pair 当成推理时的动态决策

推理不能因为当前请求中「a b」出现很多次就临时合并它。否则同一个字符串在不同 batch 中得到不同 ID 序列，缓存、评估和生成结果都不稳定。

### 2. 把 rank 大小方向写反

有的文件把越早的规则记为更小 rank，有的实现把优先级分数按相反方向存储。若排序方向反了，重叠或链式 pair 的结果就会改变。加载后应对手算样本做 golden test，而不是只检查文件能否解析。

### 3. 只复制 vocabulary，遗漏 merge list

词表告诉系统某个完整 token 是否存在，却不告诉系统从初始符号如何到达它。没有 merge list，「lower」可能永远拆成字符，即使词表中已经有「low」。

### 4. 只复制 merge list，遗漏 token 到 ID

merge list 产生的是 token 字符串；模型输入需要整数 ID。ID 顺序变化会让相同 token 查到不同 embedding 行，结果通常不会立刻在 tokenizer 层报错。

### 5. 统计 pair 时忽略词频或边界

把「low」的三次出现误当一次，或把词尾边界删掉，都会改变前三个 merge。训练语料摘要必须包含频次定义和边界协议。

### 6. 选择 pair 后没有更新邻居

缓存的 pair 计数若没有在合并点附近失效，就会在后续轮次选择不存在的 pair。可以用慢速全量重算对照增量实现，至少在小语料上逐轮比较。

### 7. 认为 token 一定是可解释的语义片段

BPE 目标是统计复用和长度权衡。一个 token 可能是词根、空格加词首、标点搭配、UTF-8 byte 片段或纯粹的高频字符邻接。解释模型行为时，应把 tokenization 证据与 embedding、上下文计算分开。

### 8. 新域只增加 merge，不重新绑定 checkpoint

在新领域临时加入专有名词 merge，会改变 token 到 ID 的空间。除非同步训练或扩展对应 embedding，并明确版本协议，否则“更短”并不等于“模型能理解”。

## 最小审计协议

面对一个声称使用 BPE 的实现，可以按下面顺序复核：

1. 取一个含空格、标点、非 ASCII 字符、未知片段和特殊 token 字面量的固定样本；
2. 保存 normalization 和 pre-tokenization 的中间结果；
3. 输出初始 alphabet 展开结果，包括 byte 或边界标记；
4. 读取 merge list，确认 rank 的排序方向；
5. 手动或用参考实现重放前三到五次 merge；
6. 对每个输出 token 查 vocabulary，核对 ID；
7. 检查 decode(encode(s)) 与协议允许的规范化是否一致；
8. 对长度、padding、truncation 和 attention mask 做一次端到端检查；
9. 记录 tokenizer 配置、词表、merge list 与 checkpoint 的版本指纹。

一个小而完整的 golden test 可以写成：

$$
\begin{gathered}
\text{lower}
\longrightarrow
[\text{low},\text{e},\text{r},\langle/\text{w}\rangle]\\
\longrightarrow
[6,4,5,0],
\qquad
\text{decode}\big([6,4,5,0]\big)=\text{lower}.
\end{gathered}
$$

它不能证明整个 tokenizer 正确，但能同时锁定 boundary、rank、merge、vocabulary 和 decode 的基本契约。

## 结语

BPE 的核心不是“把常见字符拼起来”，而是一个有状态、带版本、需要确定性回放的词表学习过程。训练阶段在语料统计上选择 merge，编码阶段按固定 rank 应用 merge，最后才通过 vocabulary 把 token 转成模型使用的 ID。边界符号、空格协议、byte fallback、tie-break、词频和 checkpoint 绑定，任何一个遗漏都会改变模型实际看到的序列。

理解这条链，就能解释为什么两个都叫 BPE 的 tokenizer 仍然可能在同一句文本上产生完全不同的 token：它们学习的语料、初始 alphabet、边界协议、merge list 或 ID 映射并不相同。

## 相关词条

[Tokenization](../text-representation/tokenization/)

[WordPiece 与 SentencePiece](../text-representation/wordpiece-and-sentencepiece/)

[One-hot 与分布式表示](../text-representation/one-hot-and-distributed/)

[嵌入](../text-representation/embeddings/)

[嵌入几何](../text-representation/embedding-geometry/)

[词表大小权衡](../text-representation/vocabulary-size-tradeoffs/)

[自注意力](../attention/self-attention/)

[注意力复杂度](../attention/attention-complexity/)
