---
title: "评估困惑度：把语言模型分数变成可复核协议"
tags: ["why-models-learn"]
---

困惑度评估不是把一个 loss 张量取指数，而是固定模型、tokenizer、评估文本、target mask、上下文窗口、reduction、数值精度和聚合规则后，计算有效 target token 的平均负对数似然。评估结果属于这套协议；只报告一个 PPL 数字，无法说明模型究竟给哪些 token 打了分、使用了多长的上下文，或不同运行是否测量了同一个对象。

![困惑度评估示意图：文本经过 tokenizer 和上下文窗口切分，只对不重复的 target 区域计分，汇总有效 token 的 NLL 后得到 PPL](/assets/pretraining/svg/evaluation-perplexity.1.svg)

## 先定义评估对象

一次可复核的困惑度评估可以写成协议元组：

$$
\mathcal E
=
\left(
M,\mathcal T,D,
\operatorname{Tok},
\operatorname{Mask},
L_{\mathrm{ctx}},
s,
\rho,
\pi
\right).
$$

其中 $M$ 是模型 checkpoint，$\mathcal T$ 是 tokenizer 及其版本，$D$ 是评估文本和切分清单，$\operatorname{Mask}$ 是 target 选择规则，$L_{\mathrm{ctx}}$ 是上下文长度，$s$ 是窗口 stride，$\rho$ 是 loss reduction，$\pi$ 是精度、batch、分布式和评估态等运行协议。PPL 是 $\mathcal E$ 的函数，不是脱离评估协议的模型固有属性。

通用的 [困惑度](../information-theory/perplexity/)词条已经推导了平均 NLL、熵、交叉熵和 KL 散度。本文把这些定义落实到 decoder-only 语言模型的评估循环，重点检查以下边界：

|边界|需要固定的对象|变化后会改变什么|
| --- | --- | --- |
|文本|原始 bytes、Unicode 规范化、文档边界和 split manifest|模型看到的 token 序列和数据分布|
|tokenizer|词表、merge、normalization、special token 和版本|token 数、条件事件和 PPL 单位|
|输入标签|右移规则、BOS、EOS、PAD、completion mask|哪些位置进入 NLL 分母|
|上下文|最大长度、窗口、stride、文档 reset 和 cache|每个 target 位置能读取的前缀|
|reduction|按 token、按序列、按 batch 或按文档聚合|平均值和不同长度样本的权重|
|运行态|dropout、dtype、logits 处理、batch padding、rank|数值结果、吞吐和可重复性|
|报告|PPL、NLL、有效 token、区间和污染证据|结果能否被比较和解释|

如果这些字段没有一起记录，“模型 A 的 PPL 小于模型 B”只能作为一个未定义比较。[训练数据](../pretraining/training-data/)说明如何保存来源、切分和版本；评估协议还需要把最终消费的 token 规则记录下来。

## 从有效 target token 聚合 PPL

### 先把每个 target 位置写成一条事件

对第 $i$ 个 target 位置，模型给真实 token $y_i$ 的条件概率为

$$
q_i
=
p_M
\left(
y_i
  \middle\vert
c_i
\right),
$$

其中 $c_i$ 是该位置实际可见的上下文。它的负对数似然为

$$
\ell_i=-\ln q_i.
$$

用 mask $m_i\in\{0,1\}$ 选择进入评估的 target，定义有效 token 数和总 NLL：

$$
Q
=
\sum_i m_i,
\qquad
S
=
\sum_i m_i\ell_i.
$$

使用自然对数时，token-level 平均 NLL 和困惑度为

$$
\ell_{\mathrm{tok}}
=
\frac{S}{Q},
\qquad
\operatorname{PPL}_{\mathrm{tok}}
=
\exp\left(\ell_{\mathrm{tok}}\right).
$$

只要 $Q>0$，这个 PPL 就是有效 target token 上逆概率的几何平均：

$$
\operatorname{PPL}_{\mathrm{tok}}
=
\left(
\prod_{i:m_i=1}
\frac1{q_i}
\right)^{1/Q}.
$$

评估器真正需要保存的是 $S$ 和 $Q$，而不是先对每个 batch 取 PPL 再求平均。指数函数应在所有目标事件汇总后只应用一次。

### 变长文档会产生两种平均

设第 $d$ 个文档有 $Q_d$ 个有效 target，累计 NLL 为 $S_d$。token-level 聚合是

$$
\ell_{\mathrm{tok}}
=
\frac{\sum_d S_d}
{\sum_d Q_d}.
$$

sequence-level 聚合则先让每条文档等权：

$$
\ell_{\mathrm{seq}}
=
\frac1D
\sum_{d=1}^{D}
\frac{S_d}{Q_d}.
$$

两者只有在每条文档有效 token 数相同，或文档损失恰好满足特殊关系时才相等。它们回答的问题不同：

|聚合|一条长文档的影响|适合回答的问题|
| --- | --- | --- |
|token-level mean|按有效 token 数加权|模型对随机抽取的有效 token 的平均概率代价|
|sequence-level mean|每条文档等权|模型对随机抽取的文档的平均 per-token 代价|
|batch-level mean|取决于 batch reduction|实现中间统计，默认不适合最终报告|
|source-level mean|每个来源或 group 等权|不同来源的平均表现和来源差异|

### 一个变长文档的数字例子

有两条文档，短文档包含两个有效 target，损失为 $0.2$ 和 $0.6$；长文档包含四个有效 target，损失为 $1.2$、$1.4$、$1.6$ 和 $1.8$：

|文档|有效 token 数 $Q_d$|总 NLL $S_d$|文档平均 NLL|
| --- | ---: | ---: | ---: |
|short|$2$|$0.8$|$0.4$|
|long|$4$|$6.0$|$1.5$|

全体共有 $Q=6$ 个有效 token 和 $S=6.8$ 的 NLL，因此

$$
\ell_{\mathrm{tok}}
=
\frac{6.8}{6}
=
1.133333333333,
\qquad
\operatorname{PPL}_{\mathrm{tok}}
=
3.105992572342.
$$

若让两条文档等权：

$$
\ell_{\mathrm{seq}}
=
\frac{0.4+1.5}{2}
=
0.95,
\qquad
\operatorname{PPL}_{\mathrm{seq}}
=
2.585709659316.
$$

短文档的 token 较少但损失较低。sequence-level mean 给它与长文档相同的权重，因此 PPL 更低。报告中必须写明使用哪一个聚合；不能把较低的 sequence-level 数字当成 token-level 结果。

## target mask 决定模型到底被评估什么

### 右移输入和标签

因果语言模型使用输入和标签的右移对齐：

$$
X_i=x_i,
\qquad
Y_i=x_{i+1}.
$$

位置 $i$ 的 logits 只能读取 $x_{\leq i}$，但 loss 评估 $x_{i+1}$。[因果语言建模](../transformer-architectures/causal-language-modeling/)展开了这一输入—标签合同。评估时需要保存：

1. 原始 token 序列；
2. 送入模型的 input_ids；
3. 目标 labels；
4. causal mask；
5. target loss mask；
6. 右移后每个位置对应的原文 span。

这五种 mask 或索引的职责不同。causal mask 控制可见前缀，padding mask 排除填充位置，target loss mask 决定分母，document boundary mask 防止跨文档读取，completion mask 可以只评估回答区域。把它们压缩成一个布尔数组后再也无法解释某个 token 为什么被计分。

### BOS 和 EOS 不是默认细节

若序列使用 BOS：

$$
x_0=\langle\mathrm{bos}\rangle,
$$

第一条有效事件可以评估模型在 BOS 条件下预测 $x_1$ 的概率。若把 EOS 作为最后一个标签：

$$
Y_{T-1}=\langle\mathrm{eos}\rangle,
$$

它也会贡献一个有效 target。是否包含 BOS 条件事件和 EOS 标签都应固定，因为它们会改变 $Q$ 和边界 token 的损失。

|协议选择|可以怎样做|比较时要保持什么|
| --- | --- | --- |
|BOS|每条文档前加入一个固定 BOS|tokenizer、位置和首个 target 规则|
|EOS|文档末加入 EOS 并计入 target|EOS 是否训练过、是否进入分母|
|文档拼接|文档间加入 EOS 或独立 reset|跨文档上下文是否允许|
|PAD|只用于矩形 batch|PAD 不进入 target loss|
|空文档|跳过或判定数据合同错误|不能让 $Q=0$ 静默进入平均|

### 只评估 completion 还是评估全序列

对 instruction 数据，一个样本常写成

$$
\text{prompt}\;\Vert\;\text{response}.
$$

若目标是比较模型生成 response 的条件似然，可以令 prompt 位置的 loss mask 为零，只计 response token：

$$
m_i
=
\begin{cases}
0,&i\in\text{prompt positions},\\
1,&i\in\text{response positions}.
\end{cases}
$$

若目标是评估整个模板序列，则 prompt、模板标记和 response 都可以进入分母。两者的 PPL 没有直接可替换关系：

|评估目标|target mask|结果主要反映|
| --- | --- | --- |
|原始语料建模|文档中的有效 next token|基础模型对语料分布的拟合|
|prompt 条件 response|只保留 response|模型对给定指令后的答案概率|
|完整 chat 模板|保留模板和 response|模型对模板格式与答案共同的似然|
|指定 span|只保留标注 span|局部片段的条件概率|

如果模板 token 没有被训练目标使用，却被放入评估分母，PPL 会混入格式学习和特殊 token 的代价。评估脚本应输出 prompt token 数、response token 数和总有效 token 数。

## 上下文窗口和 stride 会改变条件概率

### 模型实际只能读取有限前缀

理论上的自回归分解可以使用全部历史，但模型的 context window 有上限 $L_{\mathrm{ctx}}$。对位置 $i$，实际上下文通常是

$$
c_i
=
x_{\max(0,i-L_{\mathrm{ctx}}):i}.
$$

当文档长度超过窗口时，评估器要决定如何构造窗口。最简单的无重叠切块会在每个块的开头重置上下文；它计算的是真实文档在多个截断前缀下的条件概率。滑动窗口则保留左侧上下文，并只对窗口中新增的 target 位置计分。

窗口 $j$ 可以表示为半开区间：

$$
W_j
=
[a_j,b_j),
\qquad
T_j
=
[u_j,v_j),
\qquad
T_j\subseteq W_j.
$$

其中 $W_j$ 是送入模型的完整窗口，$T_j$ 是本窗口真正计分的 target 区域。要求不同窗口的 target 区域不重叠：

$$
T_j\cap T_k=\varnothing
\qquad
(j\neq k).
$$

窗口之间可以共享左侧 context，但共享的 token 不能被重复计入分母。

### 一个 stride 账本

设 token 序列长度为 $12$，窗口长度为 $6$，使用 stride $4$，采用左侧 overlap 但只计分新增 target：

|窗口|输入区间 $W_j$|target 区间 $T_j$|计分 token 数|
| --- | --- | --- | ---: |
|$0$|$[0,6)$|$[1,6)$|$5$|
|$1$|$[4,10)$|$[6,10)$|$4$|
|$2$|$[8,12)$|$[10,12)$|$2$|

三个窗口一共计分 $11$ 个位置，target 区间互不重叠。位置 $4$ 和 $5$ 在后一个窗口再次作为 context 出现，但没有再次进入 target loss。若把每个窗口全部位置都计分，overlap 会重复增加容易预测的 token 或困难 token，使 PPL 依赖窗口切法。

### 第一个窗口的左边界

滑动窗口仍然需要规定第一个 target 的条件：

1. 用 BOS 作为位置 $0$ 的前缀；
2. 把第一个 token 当作不可评估位置；
3. 使用已有的 prompt 或文档前缀；
4. 对第一个窗口保留更长的左上下文。

不同选择改变最开始的若干事件。长文档中影响可能较小，短样本或 completion-only 评估中影响会变大。报告应写出首个 target 的 offset 和是否包含 EOS。

### 文档边界不能被 stride 隐式跨越

多个文档被 packing 到同一个序列时，窗口 overlap 可能让一个文档的 token 成为另一个文档的 context。若训练和评估合同不允许跨文档读取，应在 EOS 后重置可见上下文，或使用 block-diagonal attention mask。只把多个文档拼接后按位置切窗，会产生看不见的上下文泄漏。

## tokenizer 决定 PPL 的单位

### token-level PPL 不是跨 tokenizer 的通用单位

相同文本使用两个 tokenizer 时，token 数和边界可能不同。模型 A 用较大的 token 合并两个字符，模型 B 把它们拆成两个 token；即使两个模型对字节串给出的总概率相近，逐 token 平均 NLL 也会不同。token-level PPL 只能在 tokenizer、normalization、special token 和 target mask 一致时直接比较。

需要同时报告：

|字段|用途|
| --- | --- |
|原始文本 bytes|确认评估输入相同|
|规范化版本 hash|确认 Unicode 和换行处理相同|
|tokenizer 名称与版本|确定 token 边界与词表|
|有效 token 数|解释 NLL 分母和吞吐|
|每 byte 或每字符 NLL|跨 tokenizer 时提供补充尺度|
|BOS/EOS/PAD 规则|解释边界 token 是否计分|

[Tokenization](../text-representation/tokenization/)说明 tokenizer 是模型接口的一部分。PPL 结果中的“每 token”必须带上这个接口的版本。

### 用 bits per byte 做补充尺度

若一批文档对应 $B$ 个原始 bytes，累计自然对数 NLL 为 $S$，可以计算每 byte 的 bit 数：

$$
\operatorname{BPB}
=
\frac{S}
{B\ln 2}.
$$

BPB 把损失分母换成原始 bytes，适合在文本编码保持一致时比较不同 tokenizer；它仍然需要固定 Unicode、空白、规范化和文档边界。若模型输入是有损预处理后的文本，BPB 只描述预处理后的 bytes，不等于原始文件的编码代价。

PPL 与 BPB 不能相互当作同一个指标：

$$
\operatorname{PPL}_{\mathrm{tok}}
=
\exp\left(\frac{S}{Q}\right),
\qquad
\operatorname{BPB}
=
\frac{S}{B\ln 2}.
$$

它们只共享累计 NLL $S$，分母不同，回答的问题也不同。

## 评估态、padding 和 logits 处理

### 评估态必须冻结随机层

评估时通常关闭 dropout，并使用模型的 evaluation mode。对 BatchNorm 等带运行统计的组件，训练态和评估态会读取不同的统计量；对 decoder-only Transformer，dropout 也会让同一文本的 PPL 在不同运行间变化。

至少记录：

- checkpoint hash 和加载 dtype；
- evaluation mode 是否生效；
- dropout、随机采样和 temperature 是否关闭；
- attention、padding、document boundary mask；
- KV cache 是否只用于 context，不改变 target 位置；
- batch size、padding side 和 dynamic batching。

PPL 评估应使用 teacher forcing：每个 target 位置读取真实前缀，而不是把模型自己的采样 token 作为后续输入。teacher forcing 的 PPL 与 free-running generation 的错误累积不是同一个指标。

### padding 只改变存储形状

把变长序列补齐到 batch 最大长度后，模型输出 shape 可能是 $(B,T,V)$，但只有 mask 为 1 的位置进入：

$$
S
=
\sum_{b=1}^{B}
\sum_{i=1}^{T}
m_{b,i}
\left(
-\log q_{b,i}
\right),
\qquad
Q
=
\sum_{b=1}^{B}
\sum_{i=1}^{T}
m_{b,i}.
$$

不要直接对 batch 的平均 loss 再求平均，除非每个 batch 的有效 token 数相同。变长数据中最后一个 batch、packing 比例和 completion 长度都会破坏这个条件。

### 从 logits 直接计算 log probability

评估器最好使用稳定的 log-softmax 或交叉熵内核，从 logits 直接取真实标签的 log probability：

$$
\log q_{b,i}
=
z_{b,i,y_{b,i}}
-
\operatorname{logsumexp}
\left(
z_{b,i,:}
\right).
$$

先把 logits 转成低精度 probability，再取 log 可能引入下溢。累计 $S$ 和 $Q$ 时，至少让累计器使用 float64 或明确的 float32 误差预算；模型前向可以使用混合精度，但最终 NLL 的归约精度要单独报告。[训练稳定性](../pretraining/training-stability/)说明有限性检查、loss reduction 和混合精度顺序。

## 分布式评估只汇总充分统计量

设第 $r$ 个 rank 处理了累计 NLL $S_r$ 和有效 token 数 $Q_r$。正确的 global 聚合为

$$
S_{\mathrm{global}}
=
\sum_{r=1}^{R}S_r,
\qquad
Q_{\mathrm{global}}
=
\sum_{r=1}^{R}Q_r,
$$

$$
\ell_{\mathrm{global}}
=
\frac{S_{\mathrm{global}}}
{Q_{\mathrm{global}}},
\qquad
\operatorname{PPL}_{\mathrm{global}}
=
\exp\left(
\ell_{\mathrm{global}}
\right).
$$

不要先在每个 rank 计算 PPL，再做算术平均：

$$
\frac1R
\sum_{r=1}^{R}
\exp\left(\frac{S_r}{Q_r}\right)
\neq
\exp\left(
\frac{\sum_r S_r}{\sum_r Q_r}
\right).
$$

如果每个 rank 还要报告 source-level 或 document-level 指标，应传递文档 ID、group ID 或足够的 per-document 统计，而不是只传一个 rank mean。

### 一个分布式数字例子

rank 0 处理 $2$ 个有效 token，累计 NLL 为 $1.7$；rank 1 处理 $4$ 个有效 token，累计 NLL 为 $5.1$。汇总后：

$$
S_{\mathrm{global}}=6.8,
\qquad
Q_{\mathrm{global}}=6,
$$

因此

$$
\ell_{\mathrm{global}}
=
1.133333333333,
\qquad
\operatorname{PPL}_{\mathrm{global}}
=
3.105992572342.
$$

两个 rank 的局部 PPL 分别是 $\exp(0.85)$ 和 $\exp(1.275)$，它们的平均值不是 global PPL。有效 token 数不同的 rank 必须按 $Q_r$ 加权。

[分布式训练](../pretraining/distributed-training/)展开了 rank group、reduce 和有效 token 加权。评估阶段虽然不更新参数，仍然需要明确通信 group、样本分配、重复样本和聚合顺序。

## PPL 的不确定性和模型比较

### 指数化会放大 NLL 差异

两个运行的 NLL 差值为 $\Delta\ell$ 时，PPL 比值为

$$
\frac{\operatorname{PPL}_A}
{\operatorname{PPL}_B}
=
\exp\left(
\ell_A-\ell_B
\right).
$$

因此应优先报告 NLL 差值和置信区间，再把区间端点指数化。直接比较四舍五入后的 PPL，可能把很小的 NLL 改善写成看起来更大的比例改善。

### token 不是独立同分布的样本

同一文档中的 token 共享主题、格式和上下文。把 $Q$ 个 token 当成 $Q$ 个独立样本计算普通标准误，会低估不确定性。可选的统计单位包括：

|单位|适合的比较|主要风险|
| --- | --- | --- |
|token|描述整体 token 加权 NLL|忽略文档内相关性|
|document|文档级平均 loss 和 bootstrap|长短文档权重需要明确|
|source/group|来源、用户、仓库或时间组|组数少时区间很宽|
|paired document|同一文档上比较两个模型的差值|要求两个模型评分事件对齐|

对模型 A 和 B，若每条文档都能得到成对的平均 NLL，可以定义

$$
\delta_d
=
\ell_{A,d}
-
\ell_{B,d}.
$$

对 $\delta_d$ 做 document-level bootstrap，比从两个独立 token 样本分别估计区间更能保留配对关系。最终报告应写明 bootstrap 单位、重复次数、随机 seed 和区间类型。

### PPL 低不等于下游能力高

PPL 测量的是指定文本分布上的 teacher-forced next-token likelihood。它不直接测量：

- 生成答案的事实正确性；
- 指令遵循和格式遵循；
- 多步推理；
- 安全拒答；
- 工具调用；
- 长上下文任务；
- 采样温度下的输出质量。

一个模型可以在通用网页语料上有较低 PPL，却在目标领域术语、指令格式或事实问题上表现更差。PPL 适合做语言建模目标的诊断和受控比较；它需要与独立的 task evaluation 一起解释。

## 评估数据、污染和版本

### 评估集必须在比较前冻结

一个可审计的评估集至少保存：

|字段|记录内容|
| --- | --- |
|sample ID|每个文档或样本的稳定标识|
|raw hash|原始 bytes 的内容 hash|
|canonical hash|规范化后文本的内容 hash|
|source/group|来源、用户、仓库、时间或重复簇|
|split|评估集合和版本|
|tokenizer version|tokenizer、special token 和 normalization|
|mask rule|全序列、completion、span 或其他 target 规则|
|window rule|max context、stride、overlap 和 boundary|
|license/status|可使用、删除、限制或人工复核状态|

[训练、验证与测试集](../learning-framework/train-validation-test/)强调方案冻结后再使用测试证据。困惑度评估也要冻结样本、版本和报告字段；否则不断修改过滤器和 mask 后选择最低 PPL，会把评估集变成调参集。

### 污染检查是结果解释的一部分

如果评估文本或近似重复内容出现在训练数据中，模型可能只是在复现训练记忆。需要至少记录：

1. 训练语料版本和评估集版本；
2. exact hash 是否重叠；
3. near-duplicate 的 shingle 或相似度规则；
4. 删除、隔离或保留的样本 ID；
5. 污染样本是否单独报告；
6. PPL 是否在清洁子集上重新计算。

污染证据不能单独证明模型“记住了”某条文本，但它会降低 PPL 作为泛化证据的解释范围。若不能完全排除污染，应把结果标成受限解释，而不是和干净评估集混为一个均值。

### 版本差异要拆成来源、token 和代码三层

PPL 变化可能来自三类改变：

|变化层|例子|应怎样隔离|
| --- | --- | --- |
|数据层|新增来源、过滤器、去重和文本规范化|固定 tokenizer 和评估代码重跑|
|token 层|词表、merge、BOS/EOS 和 padding 规则变化|在同一文本上比较 token 统计和 BPB|
|模型层|checkpoint、精度、batch 和上下文窗口变化|固定数据协议，记录 logits 与运行状态|

只给出“新版本 PPL 下降 0.2”不能说明是哪一层贡献了变化。

## 标准库评估探针

下面的代码使用每条文档的 per-token NLL 作为输入，计算 token-level mean、sequence-level mean、PPL、分布式充分统计量和 stride 的 target 计数。它不加载模型，因此用于核对聚合合同；真实评估还需要从 logits 生成这些 loss，并保存 tokenizer、mask 和窗口 manifest。

```python
from math import exp


document_losses = {
    "short": [0.2, 0.6],
    "long": [1.2, 1.4, 1.6, 1.8],
}

total_nll = sum(
    sum(losses) for losses in document_losses.values()
)
effective_tokens = sum(
    len(losses) for losses in document_losses.values()
)
token_mean_nll = total_nll / effective_tokens
token_ppl = exp(token_mean_nll)

document_mean_nll = sum(
    sum(losses) / len(losses)
    for losses in document_losses.values()
) / len(document_losses)
document_ppl = exp(document_mean_nll)

per_document_ppl = {
    name: exp(sum(losses) / len(losses))
    for name, losses in document_losses.items()
}

print(
    f"total_nll={total_nll:.12f} "
    f"tokens={effective_tokens}"
)
print(
    f"token_mean_nll={token_mean_nll:.12f} "
    f"token_ppl={token_ppl:.12f}"
)
print(
    f"sequence_mean_nll={document_mean_nll:.12f} "
    f"sequence_ppl={document_ppl:.12f}"
)
print(
    "per_doc_ppl="
    + str({name: round(value, 12)
           for name, value in per_document_ppl.items()})
)

rank_statistics = [(1.7, 2), (5.1, 4)]
global_nll = sum(nll for nll, _ in rank_statistics)
global_tokens = sum(
    tokens for _, tokens in rank_statistics
)
global_mean_nll = global_nll / global_tokens
print(
    f"distributed_mean_nll={global_mean_nll:.12f} "
    f"distributed_ppl={exp(global_mean_nll):.12f}"
)

windows = [
    (0, 6, 1, 6),
    (4, 10, 6, 10),
    (8, 12, 10, 12),
]
scored_positions = sum(
    target_end - target_start
    for _, _, target_start, target_end in windows
)
print(
    f"window_count={len(windows)} "
    f"scored_positions={scored_positions}"
)
```

输出为：

```text
total_nll=6.800000000000 tokens=6
token_mean_nll=1.133333333333 token_ppl=3.105992572342
sequence_mean_nll=0.950000000000 sequence_ppl=2.585709659316
per_doc_ppl={'short': 1.491824697641, 'long': 4.481689070338}
distributed_mean_nll=1.133333333333 distributed_ppl=3.105992572342
window_count=3 scored_positions=11
```

输出显示了三个实现边界。token-level 聚合使用 $6$ 个有效 token，sequence-level 聚合给两条文档相同权重；分布式结果先求 $S$ 和 $Q$ 再指数化；三个重叠窗口只对不重复的 target 区间计分，得到 $11$ 个有效位置。

## 运行方法

把上一节代码保存为 evaluation_perplexity_probe.py，然后运行：

```bash
python3 evaluation_perplexity_probe.py
```

接入真实模型时，把 document_losses 替换为模型 logits 经过稳定 log-softmax 后的真实标签 NLL。不要把每个 batch 的 PPL 直接平均；保存每条文档的 $S_d$、$Q_d$、source/group、窗口区间和 mask 版本，再按最终协议聚合。

## 失效模式

### 把 batch PPL 直接平均

不同 batch 的有效 token 数通常不同。先指数化再平均会产生与全局 token-level PPL 不同的结果。保存累计 NLL 和有效 token 数，在全量评估结束后再指数化。

### 把 sequence-level 结果当成 token-level 结果

短文档和长文档被赋予不同权重。两种平均都可以合理，但必须用明确名称报告，并保留每条文档的有效 token 数。

### 重叠窗口重复计分

滑动窗口复用左侧 context 是合理的，复用 target loss 不是。为每个窗口保存输入区间和 target 区间，验证所有 target 区间不重叠。

### 让 prompt token 进入 completion PPL

instruction 评估中，prompt 通常是条件而不是被评估的输出。若 prompt 也进入分母，结果会混入 prompt 格式和模板 token 的 loss。报告 prompt、response 和总有效 token 数。

### 把 PAD 当作真实 token

padding 只为 batch 形状服务。若 labels 中的 PAD 没有被 mask，模型会被要求预测填充符号，PPL 可能下降或上升，但结果不再对应原始文本分布。

### 训练态和评估态混用

dropout、BatchNorm 运行统计、采样和随机数据增强会让 PPL 随运行改变。固定 evaluation mode、随机 seed 和模型前向开关，并记录实际状态。

### 在低精度 probability 上取 log

先 softmax 再取 log 可能把小概率变成零。使用 logits 的稳定 log-softmax，累计器使用足够的精度，并在出现非有限值时定位首个 batch 和 token。

### 不同 tokenizer 直接比较 PPL

PPL 的分母是 token 数。词表、normalization 或 special token 改变后，每个 token 的语义单位也改变。报告 BPB 或每 byte NLL 作为补充，并说明原始文本 bytes。

### 只比较最后一个 checkpoint

最后 checkpoint 的 PPL 可能受训练预算、早停、学习率和数据游标影响。若比较训练方案，应冻结评估协议，按相同有效 token、compute 或 checkpoint 规则报告曲线。

### 忽略污染和重复文档

训练集和评估集的 exact 或 near duplicate 会让 PPL 的泛化解释变弱。保存 hash、相似度规则和清洁子集结果，不要只报告一个混合均值。

## 困惑度评估审计清单

|范围|确认项|证据|
| --- | --- | --- |
|模型|checkpoint、参数 dtype、evaluation mode 和版本固定|checkpoint hash、配置快照|
|文本|raw/canonical hash、来源、split 和文档边界冻结|manifest、版本差异|
|tokenizer|词表、merge、normalization、BOS/EOS/PAD 规则固定|tokenizer manifest、token 计数|
|标签|右移、causal mask、target mask 和 completion 规则一致|样本级 input/label/mask|
|窗口|context length、stride、overlap、首个 target 和 reset 规则明确|window manifest、target 区间|
|归约|保存累计 NLL、有效 token 和文档统计|$S$、$Q$、$S_d$、$Q_d$|
|数值|从 logits 稳定取 log probability，累计器精度可见|dtype、finite 标志、误差预算|
|分布式|rank 只汇总充分统计量，样本没有重复|rank 分配、global $S$ 与 $Q$|
|不确定性|文档或 group 级区间、bootstrap 单位和 seed 报告|差值、区间、重复次数|
|解释|污染、下游任务和 PPL 局限单独说明|清洁子集、task eval、限制项|

当这些字段同时存在时，PPL 才能作为一项可复核的语言建模证据。它仍然只回答指定文本、指定上下文和指定 target mask 下的平均概率代价。

## 相关词条

- [困惑度](../information-theory/perplexity/)
- [下一词最大似然](../pretraining/next-token-as-mle/)
- [因果语言建模](../transformer-architectures/causal-language-modeling/)
- [Tokenization](../text-representation/tokenization/)
- [训练数据](../pretraining/training-data/)
- [训练、验证与测试集](../learning-framework/train-validation-test/)
- [分布式训练](../pretraining/distributed-training/)
- [训练稳定性](../pretraining/training-stability/)
