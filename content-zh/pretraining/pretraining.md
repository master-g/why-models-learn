---
title: "预训练：从 token 预算到可恢复的训练系统"
tags: ["why-models-learn"]
---

预训练是先在大规模、通常没有人工逐题标注的语料上训练基础模型，再把得到的参数用于评估、继续预训练、监督微调或对齐。对 decoder-only 语言模型，常见目标是 next-token prediction；但“预训练”还包含 tokenizer、数据清洗、采样比例、序列 packing、token 预算、优化器状态、checkpoint 和验证协议。目标函数只是其中一层，训练系统的其余合同同样决定模型实际学到了什么。

![预训练流水线：语料经过 tokenizer 和 packing 形成 token batch，进入 next-token 模型，再由 loss、optimizer、checkpoint 和 evaluation 组成可恢复训练循环](/assets/pretraining/svg/pretraining.1.svg)

## 预训练的边界

### 预训练不是一个单独的模型层

模型层只接收 token ID 和 attention mask，并返回 hidden 或 logits。预训练还要定义：

|边界|要回答的问题|
|---|---|
|语料|哪些文档进入训练，哪些文档排除|
|tokenizer|文本如何变成 token，特殊 token 如何处理|
|序列|token 如何切成窗口，文档边界是否可见|
|目标|哪些位置产生标签，哪些位置进入 loss|
|采样|不同数据源出现的概率是多少|
|更新|global batch、梯度累积、optimizer 和学习率如何配置|
|状态|checkpoint 是否能恢复参数、优化器和数据位置|
|评估|验证 token、指标分母和版本是否固定|

如果只报告“模型用 next-token 训练”，仍然不知道训练 token 来自哪里、每个数据源被看了多少次、padding 是否进入分母、断点恢复是否改变了数据顺序。

### 预训练、继续预训练和微调

三种阶段都可能使用交叉熵，但数据规模、更新预算和参数范围不同：

|阶段|主要输入|参数状态|目标|
|---|---|---|---|
|预训练|大规模混合语料|从随机或通用初始化开始|学习一般 token 条件分布和表示|
|继续预训练|新的领域或语言语料|从已有 checkpoint 继续|改变或扩展数据分布覆盖|
|监督微调|输入—输出样本和人工或规则标签|通常使用较少数据和较短预算|调整任务或对话行为|

阶段名称本身不决定目标函数。需要记录初始化 checkpoint、语料、token budget、学习率和冻结参数，才能判断一次运行属于哪种训练。

## 从文档到训练 batch

### 先保存文档边界和来源元数据

一条原始记录至少要有内容和来源标识：

$$
\text{record}
=
(\text{text},\ \text{source},\ \text{license},\ \text{language},\ \text{timestamp}).
$$

来源元数据不一定直接进入模型输入，但它决定去重、过滤、审计和验证切分。清洗后如果只保留 token 数而丢掉原始文档 ID，后续无法解释重复数据、污染评估集或某一数据源的异常 loss。

### tokenizer 决定训练 token 的单位

对文本 $d$，tokenizer 产生：

$$
\operatorname{tokenize}(d)
=
(x_0,x_1,\ldots,x_{n_d-1}),
\qquad
x_i\in\{0,\ldots,V-1\}.
$$

这里 $n_d$ 是文档的 token 长度，不等于字符数、字节数或词数。相同文档经过不同 tokenizer 可能得到不同 token budget、不同截断位置和不同 attention 计算量。[分词](../text-representation/tokenization/)与 [BPE 分词](../text-representation/tokenization-bpe/)词条处理 tokenizer 规则；预训练流程要把 tokenizer 版本和词表 checkpoint 一起固定。

### 文档过滤改变实际分布

常见过滤和去重步骤包括：

1. 删除格式损坏、空文本和超短记录；
2. 检查语言、编码、脚本比例和异常重复字符；
3. 对完全重复或近似重复文档去重；
4. 移除验证集、测试集和明确的 benchmark 内容；
5. 记录每一步保留的文档数、字符数和 token 数。

过滤不是只减少数据量。若代码文档、数学文本或低资源语言更容易被过滤，最终 token mixture 会改变；报告中应同时给出过滤前后各来源的 token 数。

## 序列切分与 packing

### causal batch 的输入和标签

设上下文长度为 $S$，一个完整训练样本的 token 为：

$$
(x_0,x_1,\ldots,x_S).
$$

decoder 输入和 next-token 标签右移一位：

$$
\mathbf X
=
(x_0,x_1,\ldots,x_{S-1}),
\qquad
\mathbf Y
=
(x_1,x_2,\ldots,x_S).
$$

模型在位置 $i$ 读取 $x_i$，预测 $x_{i+1}$。因此一个长度为 $S+1$ 的 token 片段有 $S$ 个候选目标；如果最后一个位置是 EOS，它是否进入 loss 由目标协议决定。

### 三种文档边界合同

文档切成长度为 $S$ 的序列时，边界可以有不同处理：

|合同|下一个位置能否读取前一文档|优点|风险|
|---|---|---|---|
|独立窗口|不能，窗口首位重置上下文|边界清晰|短文会浪费上下文|
|EOS 串接|可以读取 EOS 前的历史|token 利用率高|文档间出现人工条件关系|
|block-diagonal packing|同一 batch 内不同文档互不可见|利用矩形 batch 又保持隔离|需要额外 attention mask|

三者都可以使用 next-token loss，但产生的条件分布不同。若代码只是把文档拼成一个长 token 流，实际合同通常是 EOS 串接，而不是独立文档建模。

### padding 不应自动进入 loss

变长样本 padding 到相同长度后，输入可以保持矩形：

$$
\mathbf X\in\{0,\ldots,V-1\}^{B\times S},
$$

但需要 loss mask $m_{b,i}\in\{0,1\}$ 标记有效目标。mean loss 应写成：

$$
L
=
\frac{
\displaystyle\sum_{b=1}^{B}\sum_{i=0}^{S-1}
m_{b,i}\ell_{b,i}
}{
\displaystyle\sum_{b=1}^{B}\sum_{i=0}^{S-1}
m_{b,i}
}.
$$

分母是有效 target token 数，不是固定的 $BS$。如果 padding 比例随 batch 变化，固定除以 $BS$ 会让同一个 token 的梯度尺度随无效位置数量变化。

### packing 的两个独立轴

packing 同时影响输入布局和 attention 可见性：

1. token packing 决定哪些文档被放进同一个长度窗口；
2. attention mask 决定窗口中哪些位置可以互相读取；
3. loss mask 决定哪些位置进入目标损失；
4. document ID 决定统计、去重和评估污染追踪。

把多个文档放入同一行并不自动产生 block-diagonal mask。要判断边界是否隔离，必须直接检查 attention score 的可见集合。

## token budget 和 global batch

### 用有效 token 计算训练步数

设每个设备的 micro-batch 为 $b$，设备数为 $R$，梯度累积步数为 $a$，每个样本的序列长度为 $S$，且没有 padding：

$$
B_{\mathrm{global}}
=
bRa,
\qquad
T_{\mathrm{update}}
=
bRaS.
$$

其中 $T_{\mathrm{update}}$ 是一次 optimizer update 消耗的有效 target token 数。若存在 padding，应把 $S$ 替换为每个 micro-batch 的有效 target 数并求和。

给定 token budget $T_{\mathrm{budget}}$，完整更新步数为：

$$
U
=
\left\lfloor
\frac{T_{\mathrm{budget}}}{T_{\mathrm{update}}}
\right\rfloor.
$$

余数 token 的处理必须记录：丢弃、缩小最后一个 batch、延长训练，或者在下一轮继续消费。否则“训练了 1B token”可能只表示 dataloader 产出了 1B token，而不是 optimizer 实际更新使用了 1B 个有效标签。

### 梯度累积改变的是更新频率

如果每个 micro-batch 的平均梯度为 $\mathbf g_j$，累积 $a$ 个 micro-batch 后再更新：

$$
\mathbf g_{\mathrm{update}}
=
\frac{1}{a}
\sum_{j=1}^{a}\mathbf g_j.
$$

理想情况下，梯度累积接近更大的 global batch；但 dropout、动态 padding、样本顺序、梯度裁剪和非线性 optimizer state 会使两者不完全等价。报告中要同时给出 micro-batch、设备数、累积步数和有效 token 数。

### token budget 不等于样本数量

同一篇文档可以产生不同数量的 token。固定文档数会让英文、中文、代码和数学内容占据不同的训练计算量；固定 token budget 才能直接比较实际目标位置数量。两者都可以报告，但不能互相替代。

## 数据混合与采样概率

### 从数据源权重到 token 份额

设有 $M$ 个数据源，非负权重为 $w_1,\ldots,w_M$。按权重采样时：

$$
\pi_i
=
\frac{w_i}{\displaystyle\sum_{j=1}^{M}w_j},
\qquad
\sum_{i=1}^{M}\pi_i=1.
$$

在总训练 token 为 $T$ 时，数据源 $i$ 的期望 token 数为：

$$
\mathbb E[T_i]
=
\pi_iT.
$$

这里的 $\pi_i$ 是采样概率，不一定等于原始语料在清洗后自然占比。通过提高低资源来源的 $\pi_i$，可以让它在训练中被重复采样；通过降低高重复来源的 $\pi_i$，可以限制它消耗 token budget。

### 一个三来源的数值例子

设数据源 A、B、C 的采样权重为：

$$
\mathbf w=(0.5,0.3,0.2).
$$

权重已经归一化，所以 $\boldsymbol\pi=\mathbf w$。若每次 update 使用 4096 个有效 token，运行 1000 次 update：

$$
T_{\mathrm{total}}
=
1000\cdot4096
=
4{,}096{,}000.
$$

期望来源 token 数为：

$$
\begin{aligned}
\mathbb E[T_A]&=0.5\cdot4{,}096{,}000=2{,}048{,}000,\\
\mathbb E[T_B]&=0.3\cdot4{,}096{,}000=1{,}228{,}800,\\
\mathbb E[T_C]&=0.2\cdot4{,}096{,}000=819{,}200.
\end{aligned}
$$

实际整数采样会围绕期望值波动。应记录累计样本数和累计 token 数，而不是只记录配置中的权重。

### 原始占比和训练占比是两笔账

假设清洗后的语料有：

|来源|原始 token 数|原始占比|训练采样概率|
|---|---:|---:|---:|
|A|5,000,000|0.5|0.5|
|B|3,000,000|0.3|0.3|
|C|2,000,000|0.2|0.2|

如果把 C 的采样概率调到 0.4，并把 A、B 各调整为 0.4、0.2，那么训练中 C 的 token 份额变为原始占比的两倍。它可能在训练期间被重复看到，甚至比原始数据量更早耗尽。数据 loader 必须明确是有放回采样、循环采样还是按 epoch 重新构造混合。

### 采样单位会改变含义

可以按文档、按 token、按 shard 或按 batch 采样：

|采样单位|概率含义|常见偏差|
|---|---|---|
|文档|每篇文档机会相同|长文档贡献更多 token|
|token|每个 token 来源比例接近目标|需要跨文档切分和计数|
|shard|每个 shard 被抽取的机会相同|shard 大小不一致会偏移份额|
|batch|每个 batch 选择一个或多个来源|来源边界和梯度方差改变|

“数据混合比例为 30%”只有在采样单位和 token 计数口径明确时才可复核。

## loss、更新和训练状态

### token loss 经过多个归约层

设模型对第 $i$ 个 target token 的负对数似然为 $\ell_i$，一条数据样本的 loss、一个 global batch 的 loss 和一个 update 的 loss 可能有不同归约：

$$
L_{\mathrm{batch}}
=
\frac{\sum_i m_i\ell_i}{\sum_i m_i},
\qquad
L_{\mathrm{update}}
=
\frac{\sum_{j=1}^{a}n_jL_j}{\sum_{j=1}^{a}n_j}.
$$

其中 $n_j$ 是第 $j$ 个 micro-batch 的有效 token 数。若每个 micro-batch 的有效 token 数不同，简单平均 $a$ 个 batch loss 不等于按 token 加权的 global mean。

### optimizer state 属于 checkpoint 合同

一次可恢复的训练 checkpoint 至少包含：

|状态|作用|
|---|---|
|model parameters|恢复前向和反向的参数|
|optimizer moments|恢复 Adam 等 optimizer 的历史统计|
|learning-rate scheduler|恢复当前学习率和 warmup/decay 位置|
|gradient scaler|低精度训练时恢复 scale 和 overflow 状态|
|RNG states|恢复 dropout、采样和数据顺序的随机状态|
|dataloader cursor|恢复 shard、document、token 或 batch 位置|
|tokens seen / update step|恢复预算和日志计数|

只保存 model parameters 可以进行推理，但不一定能无缝继续训练。optimizer state 丢失时，后续更新路径已经改变；dataloader cursor 丢失时，数据顺序和数据混合也可能改变。

### 参数、梯度和状态的三个时刻

一个 update 的可审计顺序为：

1. 从 checkpoint 状态取出参数、optimizer 和数据游标；
2. 读取 micro-batch，得到有效 target mask；
3. 前向计算 logits 和 token loss；
4. 反向累积梯度并执行有限性检查；
5. 按 global batch 归约和梯度裁剪合同更新 optimizer；
6. 写回参数、optimizer state、随机状态和 token 计数；
7. 在固定验证集上记录 loss、perplexity 和数据来源分项结果。

如果 checkpoint 写在参数更新前，日志中的 step 和保存的参数可能相差一轮；恢复协议应明确保存点位于 update 前还是 update 后。

## 验证集、污染和评估口径

### 验证集必须在训练输入之前隔离

预训练验证集不是训练数据的随机剩余部分。需要先按 document ID、近似内容和时间范围划分，再执行训练采样。否则去重只在训练内部进行，验证内容的近似副本仍然可能进入训练。

至少要记录：

|证据|需要固定的内容|
|---|---|
|文档集合|ID、来源、版本、切分时间|
|去重|exact hash、近似 hash、阈值|
|tokenizer|词表、normalizer、special token|
|mask|padding、EOS、prompt 或评估目标位置|
|指标|sum/mean 归约、token 分母、perplexity 定义|

### 训练 loss 不能单独代表能力

训练 loss 下降说明当前训练目标在当前数据分布上的平均 NLL 下降。它不直接说明下游任务、长上下文、代码、低资源语言或事实性表现。评估应至少保留总 loss 和按数据来源、长度、语言或任务的分项 loss。

### contamination 需要内容级检查

如果 benchmark 的题目或答案出现在训练语料中，评估结果可能反映记忆而不是泛化。简单按 URL 过滤不够，因为同一文本可能被转载、改写或切分。需要使用内容 hash、n-gram overlap 或其他可复核的匹配规则，并记录阈值和处理版本。

## 预训练的资源账本

### token 量、更新量和序列长度

给定 micro-batch、设备数、梯度累积和上下文长度，先算有效 token：

$$
T_{\mathrm{update}}
=
bRaS
$$

再算更新次数：

$$
U
=
\left\lfloor
\frac{T_{\mathrm{budget}}}{bRaS}
\right\rfloor.
$$

如果使用 packing、变长样本或 padding mask，应把每个 update 的有效 target 数写入日志，不要用名义的 $bRaS$ 代替。

### 计算量还依赖模型配置

固定 token budget 不会固定总计算量。模型宽度、层数、attention 实现、MoE active expert 数、序列长度和 padding 都会改变每个 token 的计算。对 decoder-only Transformer，至少要分开记录：

|账本|示例字段|
|---|---|
|模型|参数量、层数、宽度、head 数、FFN 宽度|
|数据|有效 token、上下文长度、padding 比例、来源份额|
|训练|global batch、update 数、学习率、optimizer|
|系统|设备数、通信、吞吐、峰值显存|
|质量|训练 loss、验证 loss、分项 loss、perplexity|

同样的 token budget 可以在不同模型上消耗不同 GPU·小时。计算最优和 scaling law 需要更细的比较口径，本文只固定训练系统的计数方式。

## 独立数值核对

下面的数值由独立标准库脚本计算，再与正文的 token budget 和 data mixture 公式逐项比对。脚本没有使用深度学习框架。

|核对项|输出|
|---|---|
|$b=4,R=1,a=1,S=1024$ 的 global batch|4|
|一次 update 的有效 token|4096|
|$T_{\mathrm{budget}}=1{,}000{,}000$ 时完整 update 数|244|
|244 次 update 的实际 token|999,424|
|预算余数|576|
|1000 次 update 的总 token|4,096,000|
|混合比例 $(0.5,0.3,0.2)$ 的期望来源 token|2,048,000 / 1,228,800 / 819,200|
|$L=(2+1+3)/(1+1+2)$|1.5|

最后一行用三个 token loss 和三个 mask 验证按有效 token 归约：分子为 6，分母为 4。它与直接平均三个样本或固定除以序列长度不同。

## 失效模式

### 把字符数当 token budget

不同语言、代码、数学和 tokenizer 版本会产生不同 token 数。字符数相同的两个语料来源可能消耗不同计算量，也可能覆盖不同数量的目标位置。训练日志应记录 tokenizer 版本和有效 token 计数。

### 文档边界在 packing 中被静默跨越

EOS 串接、独立窗口和 block-diagonal packing 产生不同条件分布。只看到输入是一行 $(B,S)$ 不能判断模型是否能从前一文档读取 token；需要检查 EOS、document ID 和 attention mask。

### padding 进入 loss 分母

固定除以 $BS$ 会在短样本 batch 中降低真实 token 的梯度尺度。使用有效 target mask 和有效 token 分母，并在日志中同时记录名义长度与有效长度。

### 数据混合比例只存在配置文件

配置里的权重不等于实际 token 份额。动态采样、shard 大小、文档长度、重复采样和过滤会改变实际分布。应按来源累计 document 数和 token 数，检查它们是否接近目标概率。

### 只保存模型参数

参数 checkpoint 可以支持推理，但不能证明训练可恢复。optimizer moments、scheduler、RNG、dataloader cursor 和 token count 缺失时，恢复后的更新路径和数据路径都可能改变。

### 验证集在训练后才确定

观察训练结果后再挑选验证集会把评估口径与结果绑定。验证文档、去重规则和 contamination 检查应在训练输入生成之前固定并版本化。

### 只看总验证 loss

某一高频来源的 loss 改善可能掩盖低资源来源、代码或长序列性能下降。保留来源、长度和语言的分项指标，并报告分母和采样数量。

### 用 token budget 推断训练质量

更多 token 只说明模型看到更多训练目标位置。数据质量、重复率、分布、模型容量、优化稳定性和评估污染同样影响结果；token 数不能单独证明能力提升。

## 预训练运行审计

拿到一条预训练运行记录时，可以按以下顺序核对：

1. 固定代码、模型配置、tokenizer、数据版本和 special token；
2. 记录清洗、去重、过滤前后的文档数与 token 数；
3. 固定训练/验证/测试切分，并检查内容级 overlap；
4. 记录每个数据源的采样单位、权重、实际 token 份额和重复次数；
5. 核对序列长度、EOS、padding、document boundary 和 attention mask；
6. 计算每个 micro-batch、global batch 和 update 的有效 target token；
7. 区分 token budget、update 数、名义 epoch 和实际 optimizer 消耗；
8. 保存参数、optimizer、scheduler、RNG、dataloader 和 token 计数；
9. 分开记录训练 loss、验证 loss、perplexity 和来源分项；
10. 记录吞吐、峰值显存、设备数、通信时间和实际 GPU·小时；
11. 在恢复训练后比较参数、数据游标、学习率和下一批 token；
12. 把任何数据清洗或评估规则的版本写入 checkpoint 元数据。

预训练的结果由数据分布、token 计数、目标 mask、更新状态和评估协议共同产生。把这些边界保留下来，下一篇才能进一步讨论 next-token objective 为什么等价于语言模型数据上的最大似然。

## 相关词条

[因果语言建模](../transformer-architectures/causal-language-modeling/)

[下一词预测与最大似然](../pretraining/next-token-as-mle/)

[分词](../text-representation/tokenization/)

[BPE 分词](../text-representation/tokenization-bpe/)

[最大似然](../probability/maximum-likelihood/)

[困惑度评估](../pretraining/evaluation-perplexity/)

[混合精度训练](../training-nn/mixed-precision/)
