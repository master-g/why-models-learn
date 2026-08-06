---
title: "指令数据：把任务、约束和答案写成可验证的示范"
tags: ["why-models-learn"]
---

指令数据是把任务意图、输入上下文、约束条件和目标答案组织成训练记录的数据。它为监督微调提供条件与 target，但“像一条指令”不等于“适合训练”：答案的正确性、任务覆盖、格式合同、来源、去重、切分和评测边界都需要独立保存。高质量指令数据让模型学习任务映射和交互协议；模板重复、错误答案或训练—评测泄漏会让 loss 下降而任务能力不变。

![指令数据示意图：原始任务经过结构化、验证、去重和覆盖统计后按来源与任务族切分，再进入监督微调与固定协议评测](/assets/finetuning/svg/instruction-data.1.svg)

## 一条指令数据记录包含什么

### 指令、输入和答案承担不同职责

一条单轮记录可以抽象成：

$$
r_i=(I_i,X_i,C_i,Y_i,H_i),
$$

其中 $I_i$ 是任务指令，$X_i$ 是用户输入，$C_i$ 是额外上下文或约束，$Y_i$ 是目标答案，$H_i$ 是来源、质量、切分和版本等元数据。并非每条记录都有单独的 $X_i$ 或 $C_i$；字段缺失和空字符串需要区分。

|字段|回答的问题|示例|
|---|---|---|
|instruction|要完成什么任务|把段落压缩成三句话|
|input|任务作用于什么对象|一段产品说明|
|context|需要读取的外部条件|术语表、文档、工具状态|
|constraint|答案必须满足什么规则|只输出 JSON，不超过 80 字|
|output|什么结果算作示范答案|三句摘要或合法 JSON|
|metadata|记录从哪里来、如何切分|source、版本、task family|

把所有文字都放进一个 prompt 字段仍然可以训练，但会丢失结构信息。结构信息决定 mask、模板、验证器和切分键；它不是只供标注界面显示的附属字段。

### 空输入和空答案要明确编码

有些任务只需要 instruction，例如“把下面的数字排序”；有些任务的 input 为空，但 output 仍应存在。空 input 不等于缺少输入，空 output 也不等于合法的拒答。数据 schema 应记录字段是否存在、是否为空、是否经过截断，并在导出前执行约束：

$$
\operatorname{valid}(r_i)
=
\mathbf 1[\operatorname{instruction}(r_i)\neq\varnothing]
\mathbf 1[\operatorname{output}(r_i)\neq\varnothing].
$$

实际校验通常还要加入任务特定条件。例如 JSON 任务需要解析成功，分类任务需要标签在词表内，代码任务需要通过语法或执行检查。

### 元数据要跟记录一起移动

|元数据|用途|不保存时的影响|
|---|---|---|
|record ID|追踪修改、去重和审计|无法定位问题样本|
|source|区分人工、转换或合成来源|无法解释质量与权重|
|source version|固定原始数据版本|无法复现导出结果|
|task family|统计任务覆盖和分组切分|相邻模板可能跨 split|
|language/domain|控制部署分布|混合比例不可解释|
|difficulty|分层采样和评测|难度变化无法归因|
|validator result|保存规则检查结果|错误答案可能进入训练|
|split|train、valid、test 或 holdout|无法核对泄漏|

record ID 不能只等于导出文件中的行号。过滤、排序和 packing 都会改变行号，稳定 ID 应由原始来源和版本派生或显式保存。

## 把记录渲染成模型看到的序列

### chat template 规定角色边界

对话记录可以写成消息列表：

$$
M_i=[(r_1,c_1),(r_2,c_2),\ldots,(r_n,c_n)],
$$

其中 $r_j$ 是 role，$c_j$ 是 content。chat template 把消息列表映射到模型实际输入：

$$
s_i=\tau(M_i).
$$

模板决定 role token、换行、separator、assistant 起始位置和 EOS。相同的内容经过不同模板后，token 序列和 loss mask 都可能不同。训练使用一种模板、推理使用另一种模板，会把数据质量问题表现为格式失败。

### 单轮记录和多轮记录不能混用统计口径

单轮指令通常包含 user instruction 和 assistant output。多轮记录还包含历史 assistant、tool call、tool result 和当前 user。构造数据时要先决定：

- 历史 assistant 是否作为条件；
- 历史 assistant 是否再次计入 target；
- tool call 的结构化参数是否计入；
- tool result 是外部条件，还是要训练模型复述的目标；
- 当前轮次是否有独立的拒答、澄清或调用工具答案。

如果把历史 assistant 当作当前 target，loss 会混合不同时间点的任务。若把 tool result 当成 assistant output，模型会学习错误的角色归属。

### 输出协议应比自然语言描述更精确

结构化输出需要把协议写入 record：

|输出类型|需要固定|验证方式|
|---|---|---|
|自由文本|语言、长度、段落和引用规则|规则检查或人工抽查|
|分类标签|标签集合和多标签语义|集合成员检查|
|JSON|schema、必填字段、类型|解析与 schema 验证|
|代码|语言、依赖、入口和安全限制|语法检查或沙箱运行|
|tool call|函数名、参数 schema、调用顺序|结构和状态验证|
|拒答|触发条件、拒答范围、替代帮助|安全规则和任务评测|

“输出一个 JSON”不是完整合同。没有 schema 时，额外解释、字段类型和缺失字段都无法定义为正确或错误。

## 来源和构造方式

### 人工记录提供判断，转换记录提供覆盖

常见来源包括：

|来源|优势|需要核对|
|---|---|---|
|人工编写|意图、边界和答案质量可直接判断|成本、标注一致性、个人偏好|
|已有任务转换|任务结构清楚、规模容易扩展|转换模板、答案保真度、重复率|
|文档生成|覆盖领域术语和真实上下文|事实核对、版权与隐私、文档泄漏|
|程序生成|格式和标签容易验证|模板单一、自然分布偏差|
|模型生成|可生成改写、难例和解释|生成错误、模式复制、验证成本|

来源标签不能替代答案验证。合成记录可以加入训练，但需要保留生成器版本、生成 prompt、过滤规则和人工或程序验证结果。

### 转换数据要保存原始任务

把一篇文档转换成问答时，至少保存：

- 原始文档 ID 和版本；
- 证据片段的起止位置；
- 生成问题和答案的工具版本；
- 答案是否由原文支持；
- 是否经过人工修改；
- 同一文档产生的记录集合。

否则当某个答案错误时，只能删除导出的文本，无法修复同源记录，也无法判断评测集是否含有同一证据。

### 生成难例不能只增加表面噪声

有效难例改变任务约束、证据位置、推理步骤或冲突条件。无效难例只替换几个同义词，仍然与原记录共享同一模板和答案路径。难例标签应说明变化来自：

|难度来源|示例|验证重点|
|---|---|---|
|输入长度|更长文档或多段上下文|截断和证据定位|
|约束数量|同时要求格式、长度和语言|约束是否全部满足|
|证据距离|答案信息分散在多个段落|跨段组合|
|干扰项|加入相似但错误的事实|证据选择|
|状态变化|工具返回错误或缺少字段|错误处理和重试|
|输出结构|嵌套字段或多步调用|schema 和状态机|

难度提高后，validator 也要同步升级。只把答案变长不能证明任务更难。

## 质量验证是数据管线的一部分

### 先检查硬约束，再检查语义质量

硬约束适合自动执行：

- required field 存在；
- 字符编码和 Unicode normalization 合法；
- token 长度不超过边界；
- role、EOS 和 special token 位置合法；
- JSON、YAML、代码或标签可解析；
- 禁止的隐私字段和凭证模式不存在；
- instruction 与 output 不为空。

语义质量需要任务验证器、交叉核对或人工抽查：

- 答案是否回答了原任务；
- 计算、事实和引用是否正确；
- 约束是否都满足；
- 答案是否把输入中的错误当成事实；
- 拒答或工具调用是否符合策略。

硬校验通过只能说明记录可被处理，不能说明答案正确。

### 质量标签要可解释

可以把单条记录的质量向量写为：

$$
q_i=(q_{\mathrm{correct}},q_{\mathrm{relevant}},q_{\mathrm{complete}},q_{\mathrm{format}},q_{\mathrm{safe}}).
$$

如果把多个维度压成一个分数，应保存每个分量和阈值。总分相同的两条记录，可能分别是“事实正确但格式错误”和“格式正确但事实错误”，处理方式不同。

过滤规则也要版本化：

$$
\mathcal D_{\mathrm{kept}}^{(v)}
=
\{r_i\in\mathcal D:\operatorname{valid}_v(r_i)=1\}.
$$

换一版 validator，保留下来的样本集合可能改变。报告中只写“清洗后有 N 条”不能说明清洗做了什么。

### 答案验证器要符合任务类型

|任务|强验证|弱验证|
|---|---|---|
|数学和计算|独立计算、单元测试、数值容差|只比较表面字符串|
|分类|标签集合、混淆矩阵和边界样本|只检查标签非空|
|摘要|事实覆盖、长度、禁止新增信息|只检查字数|
|代码|语法、依赖、沙箱行为和测试|只检查代码块存在|
|JSON|解析、schema、字段类型和约束|只检查首尾花括号|
|工具调用|函数、参数、状态和返回处理|只检查函数名|

验证器也会有误报和漏报。应保存失败样本、人工复核结果和 validator 版本，不应把自动通过当作最终质量证明。

## 去重和防止评测泄漏

### 去重先做规范化

对文本做 Unicode normalization、大小写处理、空白折叠和可选标点归一化后，得到规范化字符串：

$$
n(s)=\operatorname{normalize}(s).
$$

完全重复可以按 $n(s)$ 的 hash 分组。规范化过强会把本应不同的数字、变量名或代码合并；规范化过弱又会保留大量模板重复。数字、URL、代码标识符和语言脚本需要分别处理。

### 近重复需要保存判定依据

对 token n-gram 集合 $G(a)$ 和 $G(b)$，Jaccard 相似度可以写成：

$$
J(a,b)=\frac{\lvert G(a)\cap G(b)\rvert}{\lvert G(a)\cup G(b)\rvert}.
$$

使用阈值合并近重复时，需要记录 tokenizer、n-gram 大小、阈值和保留规则。embedding 相似度还依赖 encoder checkpoint 和 pooling 方式，不同工具的结果不能混写成一个去重指标。

### 训练集和评测集要用同一套指纹

去重范围至少覆盖：

- 同一任务的 instruction；
- instruction 加 input；
- input 中的证据片段；
- output；
- 完整模板化序列；
- 文档、会话和用户分组。

只对 instruction 去重，会漏掉“问题不同但答案相同”的记录；只对完整字符串去重，又可能漏掉模板变体和同一证据改写。评测集应在进入训练前参与泄漏扫描，但不应把评测答案作为训练过滤器的公开输入。

## 覆盖和数据混合

### 用覆盖矩阵发现缺口

设任务族为 $f$，语言或领域为 $l$，输出格式为 $g$。记录覆盖矩阵：

$$
C_{f,l,g}=\mathbf 1[\exists r_i:(f_i,l_i,g_i)=(f,l,g)].
$$

矩阵只能说明某个组合是否出现，不能说明数量是否足够。还要保存每个 cell 的样本数、有效样本数、平均答案长度和质量分布。

### 样本数量和有效 token 数都要统计

对任务族 $k$，样本数量为 $N_k$，有效监督 token 数为 $T_k$。样本份额和 token 份额分别是：

$$
\rho_k=\frac{N_k}{\sum_jN_j},
\qquad
\tau_k=\frac{T_k}{\sum_jT_j}.
$$

当答案长度差异很大时，$\rho_k$ 和 $\tau_k$ 会明显不同。SFT 使用 token mean 时，$\tau_k$ 更接近实际梯度份额；使用 example mean 时，$\rho_k$ 更接近样本权重，但仍受每条样本内部 mask 影响。

### 采样目标与原始分布要分开

原始数据的任务分布不一定等于训练想要的分布。若目标采样比例为 $\pi_k$，应保存从原始计数到采样器的映射：

|记录项|含义|
|---|---|
|raw count|过滤前记录数量|
|kept count|过滤后记录数量|
|supervised tokens|真正进入 loss 的 token 数|
|target share|计划采样份额|
|observed share|实际运行中的样本或 token 份额|
|repeat count|过采样或重复次数|

observed share 偏离 target share 时，可能是过滤率、长度、packing 或分布式 sampler 造成的。只保存配置中的 target share 不足以解释训练实际消费的数据。

## 分组切分与数据卡

### 随机按行切分会放大模板记忆

应优先按产生共同信息的单位分组：

|分组键|保护对象|适用情况|
|---|---|---|
|document ID|同一文档的不同改写|文档问答和摘要|
|task family|同一任务模板|任务泛化评测|
|conversation ID|同一会话的所有轮次|多轮对话|
|source organization|同一组织的写作习惯|企业或领域数据|
|time/version|未来版本的信息|持续更新数据|

若一个 task family 同时出现在 train、valid、test，指标测量的是同族泛化；若按 task family 分开，指标测量的是跨族迁移。两种指标都可以有意义，但名称和切分边界必须写清楚。

### 数据卡需要能回答六个问题

每个数据版本至少应回答：

1. 记录来自哪些来源，许可和隐私边界是什么；
2. schema、template、tokenizer 和 validator 的版本是什么；
3. 过滤、去重和泄漏检查删除了多少记录；
4. 任务、语言、领域、格式、难度和长度如何分布；
5. 哪些字段进入 SFT loss，归约分母是什么；
6. train、valid、test 如何按来源和任务分组。

数据卡不是宣传摘要，而是让训练和评测结果能够回到原始记录的索引。

## 一个可运行的数据审计探针

下面的探针构造 8 条小记录，检查规范化后的重复 instruction、任务族/语言/格式/来源覆盖，并按任务族做互斥的 train、valid、test 分组。它不判断答案事实正确性，只演示结构审计。

```python
import re
from collections import Counter

records = [
    {"family": "qa", "language": "zh", "format": "text", "source": "human", "instruction": "回答问题"},
    {"family": "qa", "language": "zh", "format": "json", "source": "human", "instruction": "回答问题并给出 JSON"},
    {"family": "summarization", "language": "zh", "format": "text", "source": "human", "instruction": "总结文本"},
    {"family": "summarization", "language": "en", "format": "text", "source": "synthetic", "instruction": "Summarize text"},
    {"family": "classification", "language": "zh", "format": "label", "source": "human", "instruction": "判断类别"},
    {"family": "classification", "language": "zh", "format": "label", "source": "synthetic", "instruction": "判断类别"},
    {"family": "tool", "language": "zh", "format": "json", "source": "human", "instruction": "调用工具"},
    {"family": "tool", "language": "zh", "format": "json", "source": "synthetic", "instruction": "调用工具"},
]


def normalize(value):
    return re.sub(r"\s+", " ", value.strip().lower())


normalized = [normalize(record["instruction"]) for record in records]
duplicates = Counter(normalized)
print("records=", len(records))
print("unique_normalized_instructions=", len(duplicates))
print(
    "duplicate_groups=",
    sum(count > 1 for count in duplicates.values()),
)
for field in ("family", "language", "format", "source"):
    counts = Counter(record[field] for record in records)
    values = ",".join(
        f"{key}:{counts[key]}" for key in sorted(counts)
    )
    print(field + "_counts=", values)

cells = {
    (record["family"], record["language"], record["format"])
    for record in records
}
print("coverage_cells=", len(cells))

train_families = {"qa", "summarization"}
valid_families = {"classification"}
test_families = {"tool"}
for name, families in (
    ("train", train_families),
    ("valid", valid_families),
    ("test", test_families),
):
    count = sum(record["family"] in families for record in records)
    print("group_split_" + name + "=", count)
```

运行输出：

```text
records= 8
unique_normalized_instructions= 6
duplicate_groups= 2
family_counts= classification:2,qa:2,summarization:2,tool:2
language_counts= en:1,zh:7
format_counts= json:3,label:2,text:3
source_counts= human:5,synthetic:3
coverage_cells= 6
group_split_train= 4
group_split_valid= 2
group_split_test= 2
```

探针发现 8 条记录中只有 6 个规范化 instruction，存在 2 个重复组；任务族数量表面上均为 2 条，但语言分布为 zh 7 条、en 1 条，格式分布也不均衡。按任务族分组后，train、valid、test 各有 4、2、2 条记录，且任务族互斥。这个结果只说明结构覆盖，不说明答案质量。

## 运行方法

将上一个 Python 代码块保存为 instruction-data-audit.py，再运行 python3 instruction-data-audit.py。它只使用 Python 标准库。接入真实数据时，应把记录加载、validator 结果、去重指纹、tokenizer 长度和 split 规则加入同一份审计输出。

不要用这个小探针替代语义审核。真实数据还需要 JSON/schema 解析、代码测试、事实核对、隐私扫描、近重复检测和评测集泄漏检查；探针只核对计数和分组算术。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|loss 下降但任务正确率不升|答案错误、模板重复或 task family 缺口|抽查 target、看覆盖矩阵和分组指标|
|模型输出固定句式|instruction 模板重复、合成来源占比过高|规范化去重、按 source 分层统计|
|JSON 经常无法解析|schema 未保存或 validator 只检查首尾字符|解析并执行 schema 校验|
|模型复述输入而不完成任务|instruction 与 output 边界混乱|检查 role、target span 和答案 rubric|
|多轮回复角色错乱|历史 assistant、tool result 或 separator 标注错误|逐轮渲染 template 并打印 role token|
|困难集指标虚高|train/test 共享文档、模板或会话|按 document/task/conversation 做 group split|
|某个语言几乎没有有效样本|原始语言分布、过滤率或 tokenizer 截断|同时统计 raw、kept 和 supervised token|
|工具调用参数错误|函数 schema、状态和错误返回没有示范|运行结构验证和状态转移测试|
|拒答过宽或过窄|正例、边界例和替代帮助不平衡|按风险类别和边界条件分层评测|
|合成记录数量很大但能力无变化|验证器弱、重复率高或答案只改变表面词|保存 generator 版本并检查近重复|

### 最小审计表

|审计层|应保存|验收问题|
|---|---|---|
|schema|字段定义、空值规则、record ID|记录能否稳定解析|
|来源|source、版本、许可、生成器|答案和风险能否追溯|
|内容|instruction、input、context、constraint、output|任务条件和 target 是否完整|
|模板|role、separator、EOS、chat template|训练和部署格式是否一致|
|质量|validator、人工复核、质量分量|通过记录是否真的可用|
|去重|规范化、hash、n-gram 或 embedding 配置|近重复是否影响权重|
|分布|task、language、domain、format、difficulty、length|覆盖是否和部署目标一致|
|切分|document、task、conversation、time 分组|评测是否泄漏|
|训练接口|tokenizer、mask、截断、packing、采样|SFT 实际消费什么数据|
|评测|holdout、prompt、decode、scorer|指标是否能回到数据合同|

指令数据的单位不是一行字符串，而是一条带有任务条件、答案、来源和验证状态的记录。只有在 schema、质量、覆盖、去重、切分和 SFT loss 口径同时固定后，训练曲线才有可解释的对象。

## 相关词条

- [监督微调](../finetuning/sft/)：把 instruction、input、output 和 mask 转成 next-token 参数更新。
- [训练数据](../pretraining/training-data/)：讨论来源、版本、分布、过滤和切分的训练数据合同。
- [分词](../text-representation/tokenization/)：核对文本、special token、token ID 和长度。
- [因果语言建模](../transformer-architectures/causal-language-modeling/)：说明 shift、causal mask 和自回归训练前向。
- [困惑度评估](../pretraining/evaluation-perplexity/)：固定 target mask、窗口和 perplexity 归约。
- [数据增强](../evaluation-and-generalization/data-augmentation/)：比较改写、变换和新任务记录的分布变化。
- [灾难性遗忘](../finetuning/catastrophic-forgetting/)：评估指令数据训练后原有能力的变化。
