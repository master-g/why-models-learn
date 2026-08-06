---
title: "训练数据：从原始文档到可审计语料"
tags: ["why-models-learn"]
---

训练数据不是一个下载完成的文件，而是一条可以追踪的变换链：原始文档经过来源登记、格式解析、质量过滤、精确和近似去重、数据切分、tokenizer 编码与 shard 排列，才成为模型实际消费的 token。每一步都可能改变文档数量、有效 token 数、来源比例和评估污染风险。训练数据词条固定这条数据链和审计字段；[预训练](../pretraining/pretraining/)词条进一步处理 token budget、global batch、optimizer 和 checkpoint。

本文先区分原始记录、规范化文档、训练样本和有效 target token，再处理 provenance、license、隐私、清洗、去重和 train/validation/test 切分。随后用 manifest、内容哈希、shard、token 统计和版本差异描述可复现数据集，最后核对来源配比、近似重复和常见数据泄漏。

![训练数据从原始文档经过来源登记、规范化、过滤去重、切分和 tokenizer 统计，形成带 manifest 的可审计语料](/assets/pretraining/svg/training-data.1.svg)

## 一个训练样本有多个身份

### 原始记录、文档和 token 不是同一个单位

同一份内容在数据管线中会被表示多次。把这些层级混成“样本”，会让去重、切分和 token 统计失去明确对象：

|层级|典型字段|它回答的问题|
| --- | --- | --- |
|原始记录|来源 URL、抓取时间、原始字节、来源 ID|内容从哪里来，能否回溯|
|解析文档|纯文本、标题、段落、文档 ID|哪些文本进入规范化|
|规范化文档|canonical text、规范化哈希、过滤标记|哪些表面差异被合并|
|训练样本|segment、split、position、document ID|哪些 token 被放进哪条输入序列|
|有效 target token|label、loss mask、来源、token 位置|哪些事件真正进入损失分母|

一个文档可以被截成多条训练样本，一条 packed sequence 可以包含多个文档，一个训练样本也可以因为 padding 和 boundary mask 只产生部分有效 target token。文档数、样本数和有效 token 数需要分别记录。

### 数据变换要有明确输入和输出

令原始数据集合为 $\mathcal R_0$。第 $k$ 个数据处理步骤是一个带配置的变换：

$$
\mathcal R_{k+1}
=
F_k\left(
\mathcal R_k;\,
\gamma_k
\right),
$$

其中 $\gamma_k$ 包含代码版本、阈值、词表、随机种子或外部资源版本。最终训练集合不是一个只由内容决定的集合：

$$
\mathcal D
=
F_{K-1}\circ\cdots\circ F_1\circ F_0
\left(
\mathcal R_0
\right).
$$

如果只保存最终 token shard 而不保存每个 $F_k$ 和 $\gamma_k$，就无法回答“某个来源为什么减少”“某个验证样本何时被排除”或“这次版本变化来自哪一步”。

### 文档 ID 和内容哈希承担不同责任

文档 ID 是来源身份，内容哈希是内容版本。相同内容来自两个 URL 时，两个来源 ID 仍然需要保留；相同来源更新内容时，内容哈希应变化。可以用以下组合区分身份和版本：

$$
\begin{aligned}
\mathrm{record\_key}
&=
\left(
\mathrm{source\_id},
\mathrm{retrieved\_at}
\right),\\
\mathrm{content\_hash}
&=
H\left(
\mathrm{canonicalize}(\mathrm{raw\_content})
\right).
\end{aligned}
$$

source ID 用于 provenance、license 和删除追踪；content hash 用于精确去重、版本比较和缓存命中。只保存其中一个字段，会把来源合并或把同一内容的更新误判为重复。

## 来源、许可和隐私字段

### provenance 不是附加说明

训练数据至少应在 manifest 中记录：

|字段|示例内容|后续用途|
| --- | --- | --- |
|source ID|站点、仓库、文档库或内部数据源的稳定 ID|来源统计和删除定位|
|locator|URL、文件路径、仓库 commit 或对象键|回到原始内容|
|retrieved at|采集或导入时间|版本和时效分析|
|license class|允许的内部分类标签|进入、排除或人工复核|
|language/script|语言和脚本识别结果|来源配比和质量分项|
|content hash|规范化内容哈希|精确去重和版本 diff|
|transform version|解析、清洗、去重配置版本|重建同一派生记录|

这些字段未必进入模型输入，但会决定哪些数据被保留、怎样切分以及异常结果能否解释。把它们丢弃后再从最终 shard 猜来源，通常无法恢复原始关系。

### 许可检查要与文本清洗分开

许可状态、来源合同和文本质量是不同字段。文本看起来有用，不代表它满足当前数据集的进入条件；许可字段完整，也不代表文本没有乱码、重复或污染。数据管线应先把进入规则写成可检查的状态，再把合格记录交给内容过滤。

|判断|记录什么|不要替代成什么|
| --- | --- | --- |
|是否允许处理|license class、来源规则、人工决定编号|文本质量分数|
|是否允许训练|训练用途状态、排除原因|是否成功 tokenize|
|是否需要人工复核|review flag、复核结果|自动分类器置信度|
|是否需要删除|删除请求 ID、影响的版本|从最新 shard 直接擦除|

这是数据治理的操作字段，不是对具体法域的法律意见。需要遵守的许可或删除规则应由项目责任人和合规流程确认，并把确认结果写进版本记录。

### 隐私字段不能被过滤器静默吞掉

如果数据包含个人信息或内部敏感字段，应显式记录处理状态：

$$
\mathrm{privacy\_state}
\in
\{
\mathrm{unknown},
\mathrm{review},
\mathrm{redacted},
\mathrm{excluded}
\}.
$$

redaction 需要保存规则版本和受影响记录数；excluded 需要保存排除原因和可追踪的来源 ID。不要在日志中打印原始文本来证明过滤器工作；日志应使用记录 ID、哈希、计数和脱敏后的片段。

### 删除要沿派生链传播

一个来源记录可能已经出现在规范化文档、去重簇、split manifest 和 token shard 中。收到删除或排除决定后，至少要找到所有下游表示：

$$
\mathrm{raw\ record}
\rightarrow
\mathrm{canonical\ document}
\rightarrow
\mathrm{training\ segment}
\rightarrow
\mathrm{token\ shard}.
$$

只从最新 shard 中删除一段字节，不能证明所有旧版本、缓存和评估副本都不再引用它。删除操作应产生新的数据版本，并记录旧版本的状态与不可继续使用的原因。

## 清洗改变的是数据分布

### 解析失败先于质量评分

原始输入需要先经过可重复的解码和解析：

1. 记录原始字节和解码编码；
2. 处理格式损坏、空文档和无法解析的容器；
3. 分离正文、标题、代码块、表格或 metadata；
4. 保存解析失败原因和原始记录 ID；
5. 对成功解析的文本执行后续规范化。

解析失败不是低质量文本的同义词。把所有失败都静默丢弃，会让某些格式、语言或来源的保留率下降，却没有任何审计字段可以说明原因。

### 规范化要避免合并不同内容

常见的表面规范化包括 Unicode 规范、换行统一、空白压缩、HTML boilerplate 清除和大小写策略。规范化函数必须写清楚它保留哪些语义：

$$
\mathrm{canonicalize}(d)
=
N_{\mathrm{unicode}}
\circ
N_{\mathrm{newline}}
\circ
N_{\mathrm{boilerplate}}
(d).
$$

过弱的规范化会让同一内容的空白差异躲过去；过强的规范化会把代码大小写、数学符号、语言文字或结构化字段合并。规范化前后的长度、哈希和变换版本都应可查询。

### 过滤器的顺序会改变结果

设 $A$ 是语言过滤通过的集合，$B$ 是长度过滤通过的集合。一般来说：

$$
A\cap B=B\cap A,
$$

但实际过滤器往往依赖文本变换、来源统计或前一步产生的字段。例如先去 boilerplate 再做长度过滤，与先按原始长度过滤可能保留不同文档。若过滤器使用全局阈值或来源配额，执行顺序还会改变阈值本身。

每一步至少要记录：

|审计量|含义|
| --- | --- |
|input records|进入这一步的记录数|
|output records|通过这一步的记录数|
|removed records|被排除的记录数和原因计数|
|input tokens|输入记录 tokenize 后的 token 数|
|output tokens|输出记录 tokenize 后的 token 数|
|retention rate|output 除以 input 的记录或 token 比例|

记录数和 token 数要分开。一个过滤器可能只排除少量长文档，却显著改变 token 预算；也可能排除大量短文档，对实际计算量影响很小。

### 过滤阈值改变覆盖范围

质量过滤通常包含语言、长度、异常重复字符、乱码比例、代码或标记比例、空白比例和来源特定规则。阈值不应只写“清洗完成”，而应写成配置：

$$
\gamma_{\mathrm{filter}}
=
\left(
\tau_{\mathrm{length}},
\tau_{\mathrm{script}},
\tau_{\mathrm{repeat}},
\tau_{\mathrm{quality}},
\ldots
\right).
$$

阈值越严格，可能减少乱码和模板噪声，也可能减少低资源语言、短文、代码片段或数学内容。过滤结果应按来源、语言、长度和文档类型分项报告；总保留率无法说明覆盖范围。

## 精确去重与近似去重

### 精确去重先定义比较文本

对规范化文本 $d$ 计算内容哈希：

$$
h(d)
=
H\left(
\mathrm{canonicalize}(d)
\right).
$$

相同哈希可以作为精确重复候选。保留一个 canonical record 后，其余记录不能简单丢弃来源信息；应把它们作为 aliases 或 duplicate references 保存，以便解释来源覆盖和删除传播。

精确去重可以在不同粒度执行：

|粒度|能发现什么|可能漏掉什么|
| --- | --- | --- |
|整文档|完整重复文件或网页|只重复一个段落的长文档|
|段落|模板段落、页脚和复制片段|段落顺序变化或改写|
|token window|训练窗口之间的相同片段|不同 tokenizer 或近似改写|
|source group|同一来源的多次快照|跨来源复制内容|

去重粒度不同，数据集含义也不同。整文档去重不能替代评估集污染检查；训练数据中的重复段落仍可能让某个 benchmark 片段被反复看到。

### 近似重复用 shingle 定义相似性

把文档拆成长度为 $k$ 的连续 shingle 集合：

$$
S_k(d)
=
\left\{
(w_i,\ldots,w_{i+k-1})
\mid
0\leq i\leq \lvert d\rvert-k
\right\}.
$$

两个文档的 Jaccard 相似度为

$$
J(A,B)
=
\frac{\lvert A\cap B\rvert}
{\lvert A\cup B\rvert}.
$$

若 $J(A,B)$ 超过预设阈值，可以把它们放入同一个近似重复候选簇。阈值不是自然常数：短文本、代码、模板和不同语言需要不同验证方式。MinHash、LSH 或其他索引适合缩小候选对，不应被当成已经完成的最终判定。

### 候选、簇和保留记录要分开

近似去重至少有三种状态：

|状态|含义|下一步|
| --- | --- | --- |
|candidate pair|索引认为两个记录可能相似|计算精确 shingle 或其他证据|
|duplicate cluster|规则或人工审查确认同一内容族|选择 canonical record，保存成员映射|
|kept record|进入下游训练的代表记录|写入 manifest 和 dedup version|

把 candidate pair 直接当作 duplicate，会把相似但互补的文档误删。把 duplicate cluster 只保存为一个数量，则会丢失成员关系，后续无法检查某个验证文档是否与训练簇重叠。

### 一个 Jaccard 数字例子

设两个文档的 3-shingle 集合为

$$
\begin{aligned}
A&=\{a,b,c,d,e\},\\
B&=\{b,c,d,e,f\}.
\end{aligned}
$$

它们的交集有 4 个 shingle，并集有 6 个：

$$
J(A,B)
=
\frac46
=
0.666666666667.
$$

若项目阈值为 $0.8$，这对记录不会自动进入 duplicate cluster；若阈值为 $0.6$，它们会成为候选。这个数字只描述所选 shingle 和阈值下的表面重叠，不证明两段内容的来源、语义或许可状态相同。

## 切分必须先于训练结果

### split 的单位要与相关性单位一致

随机按行切分会把同一来源、同一用户、同一网页快照或同一文档簇分到不同 split。若这些记录高度相关，验证 loss 可能低于真正的部署泛化风险。切分单位应按相关性来源决定：

|相关性来源|建议切分单位|避免的情况|
| --- | --- | --- |
|同一原始文档|document ID|窗口跨 train 和 validation|
|同一网页或仓库|source group、repository 或 URL group|模板和版本泄漏|
|同一用户或会话|user、conversation 或 account group|用户内容跨 split|
|近似重复簇|duplicate cluster|改写版本跨 split|
|时间变化的数据|时间区间|未来内容进入过去的训练集|

切分不是只给每条记录写一个字符串。要保存 group ID、split 规则版本、随机种子和最终计数，才能重建相同的边界。

### train、validation、test 的作用不同

|split|用于什么|哪些信息不能回流|
| --- | --- | --- |
|train|更新模型参数|test 标签和最终部署结果|
|validation|选 checkpoint、超参数和停止时机|不能在结果确定后反复改规则|
|test|最终一次性报告|不能参与训练或模型选择|

若验证集在看过训练曲线后被重挑，validation 已经变成选择过程的一部分。若 test 内容出现在训练语料，即使模型没有直接复制答案，最终分数也需要标记 contamination 风险。

### contamination 是重叠证据，不是记忆证明

数据污染审计可以报告不同证据层级：

$$
\mathrm{overlap\ evidence}
\in
\{
\mathrm{exact},
\mathrm{near\ duplicate},
\mathrm{n\text{-}gram},
\mathrm{source\ related},
\mathrm{semantic\ candidate}
\}.
$$

exact overlap 说明规范化内容相同；near duplicate 和 n-gram overlap 说明表面片段重合；source related 说明来源相关；semantic candidate 需要进一步人工或模型审查。任何一层都不自动证明模型在评估时调用了记忆，但它会改变结果的解释和确认程度。

### 评估集要在训练前冻结

冻结一个评估集至少需要：

1. 保存原始记录 ID 和内容哈希；
2. 先做评估集内部去重；
3. 用同一规范化规则与训练候选比对；
4. 从训练候选移除 exact 和确认的 near duplicate；
5. 保存 overlap report、阈值和工具版本；
6. 冻结 split manifest 后再生成训练 shard。

如果先生成训练数据、训练模型，再根据结果选择验证文档，数据处理和模型选择已经相互耦合。后续指标无法再被当作完全独立的验证证据。

## manifest、shard 和版本

### manifest 是派生数据集的索引

一个可审计 manifest 可以把每条记录压缩成如下字段：

|字段|示例|是否应参与版本比较|
| --- | --- | --- |
|record ID|稳定的 canonical record ID|是|
|source group|来源或相关性组|是|
|content hash|规范化内容哈希|是|
|split|train、validation、test|是|
|token count|tokenizer 版本下的有效长度|是|
|filter flags|过滤器通过和排除原因|是|
|dedup cluster|精确或近似重复簇 ID|是|
|shard ID、offset|token 在哪个 shard 的位置|是|
|transform version|解析、过滤、去重和 tokenize 版本|是|

manifest 不需要保存全文，但必须能通过 record ID 或 hash 回到受控的原始或规范化存储。shard 是高吞吐的存储布局；manifest 是解释 shard 内容的索引。只有 shard 没有 manifest，无法按来源、split 或过滤原因重算统计。

### shard 需要内容地址和顺序信息

训练 loader 通常按 shard 流式读取。为了复现一次训练，需要知道：

$$
\mathrm{sample\ identity}
=
(\mathrm{dataset\ version},\ \mathrm{shard\ id},\ \mathrm{offset}).
$$

还要记录 shard 文件哈希、样本顺序、shuffle seed、epoch 或 token 游标。随机打乱不等于不可复现；不可复现来自没有保存产生顺序的配置和状态。

### 版本差异要说明变化类型

两个数据版本之间的 diff 至少区分：

|变化类型|例子|对训练解释的影响|
| --- | --- | --- |
|additive|新增一个来源或时间区间|有效 token 和覆盖范围增加|
|removal|删除一组来源或重复簇|分布和污染风险下降或改变|
|transform|规范化、tokenizer 或过滤规则改变|相同文档的 token 事件可能改变|
|repartition|split、shard 或顺序改变|训练/评估边界或复现路径改变|
|metadata-only|补充来源字段但不改内容|统计可解释性提高，目标可能不变|

只把版本号从 v1 改成 v2，不说明 diff，不能支持结果比较。即使 token 总数恰好不变，来源、长度、重复率和 split 也可能已经变化。

## 统计实际被模型看到的分布

### 文档比例和 token 比例是两个指标

设来源 $j$ 的有效 target token 数为 $T_j$，总有效 token 数为 $T$：

$$
s_j
=
\frac{T_j}{T},
\qquad
T=\sum_jT_j.
$$

文档比例可以写成 $N_j/\sum_kN_k$，但它不等于 $s_j$。tokenizer、平均文档长度、截断、padding、重复采样和 loss mask 都会改变有效 token 份额。

### 一个按文档采样的数字例子

来源 A 有 80 条文档，每条平均 100 个有效 token；来源 B 有 20 条文档，每条平均 400 个有效 token。文档比例和 token 比例如下：

|来源|文档数|平均 token 数|有效 token 数|文档比例|token 比例|
| --- | ---: | ---: | ---: | ---: | ---: |
|A|80|100|8,000|0.8|0.5|
|B|20|400|8,000|0.2|0.5|
|总计|100|—|16,000|1.0|1.0|

按文档抽样时，配置看起来是 $0.8/0.2$；按有效 token 统计时，实际 mixture 是 $0.5/0.5$。如果训练目标想让 token 份额为 $0.8/0.2$，就必须改变抽样单位、来源权重或每条记录的采样次数。

### 统计要按切分和来源交叉展开

总 token 数不能说明数据覆盖。至少应按以下维度交叉统计：

|维度|建议指标|
| --- | --- |
|source|文档数、有效 token、平均长度、重复率|
|language/script|文档数、token 份额、过滤保留率|
|document type|正文、代码、数学、对话或模板的份额|
|length bucket|长度分位数、截断比例、padding 比例|
|quality/filter|各过滤原因的输入和输出计数|
|split|train、validation、test 的文档和 token 数|
|dedup|exact duplicate、near duplicate 和簇大小分布|

如果只报告总体 token 数，低资源来源被高频来源覆盖时不会在总数中显现。训练结果的来源分项指标需要与这张数据 profile 一起读取。

### tokenization 统计要在最终版本上重算

字符数、字节数和 token 数是三种不同单位。数据版本、tokenizer 版本或 special token 规则变化后，应重新计算：

$$
\begin{aligned}
\bar \ell_{\mathrm{token}}
&=
\frac1N\sum_{n=1}^{N}\ell_n,\\
\rho_{\mathrm{truncate}}
&=
\frac{
\#\{\text{被截断的文档}\}
}{N},\\
\rho_{\mathrm{empty}}
&=
\frac{
\#\{\text{tokenize 后没有有效 target 的文档}\}
}{N}.
\end{aligned}
$$

平均长度和截断比例能够揭示“token 数增加但长程内容被切掉”的情况。[分词](../text-representation/tokenization/)词条处理 token 单位；这里关心的是 tokenizer 版本在最终数据集上的实际统计。

## 数据质量是多维度记录

### 一个总分不能代替质量 profile

训练数据质量至少包含不同方向：

|方向|可以检查什么|错误解释|
| --- | --- | --- |
|可解析性|解码、格式和文本完整性|把解析成功当成内容可靠|
|重复度|exact、near duplicate、模板比例|把低重复度当成高覆盖|
|覆盖范围|语言、来源、文档类型、长度|把总 token 数当成多样性|
|目标相关性|内容是否支持训练目标|把通用质量分数当成任务质量|
|时间和新鲜度|发布时间、快照和版本|把最新抓取时间当成事实正确|
|隐私与许可|状态、排除原因、删除追踪|把没有标记当成允许训练|
|评估隔离|train 与 eval overlap|把低训练 loss 当成无污染|

质量分数可以用于排序或采样，但应保留原始分项和阈值。一个总分会把低资源语言、代码、数学和长文档的差异压缩成一个数，无法说明哪一类内容被排除。

### 过滤器要做小样本人工复核

自动过滤器适合批量计数，不适合单独证明语义保留。每次改变阈值后，应抽取：

1. 通过样本；
2. 被排除样本；
3. 接近阈值的边界样本；
4. 各来源和语言的样本；
5. 近似重复簇中的 canonical 与 removed 样本。

复核结果要记录配置版本和样本 ID，不要只保存“人工看过”。边界样本能够暴露过滤器把表格、代码、公式或低资源语言误判为噪声的情况。

### synthetic data 需要单独标记来源

如果训练数据包含合成文本、模型改写或自动标签，manifest 应把生成模型、checkpoint、提示版本和生成时间作为来源字段。合成数据不能因为格式一致就与原始文档合并统计：

|来源类别|需要额外记录|
| --- | --- |
|原始文档|原始 source ID、采集时间和许可状态|
|规则变换|变换规则、输入记录和输出 hash|
|模型生成|生成模型、checkpoint、prompt 和采样配置|
|人工修改|修改者或流程 ID、修改范围和审查状态|

合成数据的内容可能重复训练模型的已有分布，也可能把错误复制到更多记录。来源分项和生成链保留后，才能把它与原始数据的覆盖和污染风险分开分析。

## 独立数值核对

下面用一个小型数据管线重算保留率、近似重复相似度和来源 token mixture。所有数字只依赖整数计数和自然对数之外的基础运算，不依赖数据处理框架。

### 过滤流水线的保留率

假设各阶段计数如下：

|阶段|文档数|有效 token 数|相对原始文档保留率|相对原始 token 保留率|
| --- | ---: | ---: | ---: | ---: |
|原始导入|1,000|1,000,000|1.000000|1.000000|
|解析和空文档过滤|960|960,000|0.960000|0.960000|
|语言与格式过滤|900|900,000|0.900000|0.900000|
|精确去重|840|840,000|0.840000|0.840000|
|近似去重|780|760,000|0.780000|0.760000|

最终文档保留率和 token 保留率分别为

$$
\rho_{\mathrm{doc}}
=
\frac{780}{1000}
=
0.780000,
\qquad
\rho_{\mathrm{token}}
=
\frac{760000}{1000000}
=
0.760000.
$$

近似去重只额外减少 60 条文档，却减少 80,000 个有效 token。若只报告“删除了 220 条文档”，会低估它对 token 预算和来源分布的影响。

### Jaccard 候选和来源混合

前面的 shingle 例子给出

$$
J(A,B)=0.666666666667.
$$

同一数据集的来源 A、B 有效 token 均为 8,000，因此

$$
s_A
=
\frac{8000}{16000}
=
0.500000,
\qquad
s_B
=
\frac{8000}{16000}
=
0.500000.
$$

独立脚本的输出应同时包含 retention_doc=0.780000、retention_token=0.760000、jaccard=0.666666666667 和 mixture=[0.5,0.5]。这些结果核对的是计数关系；它们不替代实际数据的来源、许可和污染审计。

## 失效模式：数据看起来可用但无法解释

### 只保存最终 token shard

**现象：**模型可以继续训练，但无法回答某个 token 来自哪个来源、哪个过滤版本或哪个原始文档。

**检查：**manifest 是否能从 shard offset 回到 record ID、content hash、split、来源和 transform version；是否保存 shard hash。

### 在切分之后才去重

**现象：**train 和 validation 的 exact 或 near duplicate overlap 很高，验证 loss 低于独立来源。

**检查：**冻结 evaluation manifest，使用相同 canonicalize 和 dedup 配置对训练候选比对，再生成最终 split。

### 用原始哈希做精确去重

**现象：**同一网页只有空白、HTML 包装或 Unicode 表面差异，却被保留成多份记录。

**检查：**明确 hash 的输入是 raw bytes 还是 canonical text，并分别报告两种重复率。

### 规范化过强

**现象：**代码大小写、数学符号、段落边界或语言字符被错误合并，训练覆盖范围下降。

**检查：**抽样比较规范化前后文本和 hash；对代码、数学、表格和多语言来源使用独立规则或保留原始字段。

### 把近似候选直接当重复

**现象：**标题相似、模板相同或短文本共享常见 shingle 的记录大量消失。

**检查：**区分 candidate pair、duplicate cluster 和 kept record；记录阈值、shingle 长度和人工复核结果。

### 按行随机切分高相关数据

**现象：**同一文档、用户、仓库或重复簇跨越多个 split。

**检查：**按 source group、document ID、user 或 duplicate cluster 切分，并报告 group overlap。

### 只看文档配比

**现象：**配置写成来源 A/B 为 $0.8/0.2$，实际有效 token 却接近 $0.5/0.5$。

**检查：**按 tokenizer、loss mask 和截断规则累计来源有效 target token，单独报告文档比例和 token 比例。

### 只看总保留率

**现象：**总体保留率稳定，但某种语言、文档类型或长度区间几乎被过滤掉。

**检查：**按 source、language、length bucket、document type 和 filter reason 展开 input/output 计数。

### 清洗规则没有版本

**现象：**同一数据集名称重新生成后，文档数、token 数或顺序变化，却无法说明代码和阈值差异。

**检查：**记录 transform version、配置 hash、代码 commit、随机种子、tokenizer hash 和依赖版本。

### 删除只作用于最新版本

**现象：**最新 shard 不再包含某记录，但旧 cache、旧 manifest 或训练 checkpoint 仍然引用它。

**检查：**沿 raw record 到 canonical document、segment、shard 和 checkpoint 元数据的引用链执行删除影响分析。

### 合成数据没有来源标签

**现象：**生成文本与原始文档混在一起，训练结果无法分开解释，模型错误可能被重复放大。

**检查：**记录生成模型、checkpoint、prompt、采样配置、生成时间和人工审查状态。

## 可复用的数据审计协议

审计一个训练数据集时，按以下顺序记录：

1. 冻结原始记录的 source ID、locator、时间、许可和隐私状态；
2. 保存原始字节或受控引用、解析结果和失败原因；
3. 固定 canonicalize、tokenizer、special token 和 transform version；
4. 为每条规范化文档计算 content hash 和稳定 record ID；
5. 执行语言、格式、长度和来源规则过滤，并记录逐步计数；
6. 先做 exact dedup，再做 near-duplicate candidate、cluster 和保留决策；
7. 冻结 validation/test manifest，按 group 和 duplicate cluster 检查 overlap；
8. 生成带 split、来源、token count、filter flags 和 dedup cluster 的 manifest；
9. 生成带 hash、offset、顺序和 seed 的 token shard；
10. 按来源、语言、文档类型、长度和 split 统计有效 token mixture；
11. 对边界过滤样本、重复簇和合成数据做抽样复核；
12. 保存版本 diff、删除影响、审计报告和可重建命令。

数据集通过这些检查后，仍然不等于“质量已经足够”。它只说明数据来源、变换、切分、统计和残余风险有记录。模型训练还要在固定的 token budget、目标函数和验证协议上进行；[下一词最大似然](../pretraining/next-token-as-mle/)负责解释有效 target token 如何进入概率目标。

## 相关词条

- [预训练](../pretraining/pretraining/)：把数据集接入 tokenizer、packing、token budget、optimizer 和 checkpoint。
- [下一词最大似然](../pretraining/next-token-as-mle/)：说明有效 target token 如何组成序列似然和 NLL。
- [分词](../text-representation/tokenization/)：固定文本到 token ID 的编码单位和词表版本。
- [训练、验证与测试集](../learning-framework/train-validation-test/)：区分 split、选择过程和最终评估。
- [分布偏移](../evaluation-and-generalization/distribution-shift/)：解释过滤和来源配比变化如何影响部署分布。
- [数据增强](../evaluation-and-generalization/data-augmentation/)：区分有标签变换、原始来源和新增训练样本。
- [经验风险最小化](../learning-framework/empirical-risk-minimization/)：把加权数据集上的平均损失放回有限样本选择问题。
- [最大似然](../probability/maximum-likelihood/)：区分观测概率、似然函数和参数选择。
