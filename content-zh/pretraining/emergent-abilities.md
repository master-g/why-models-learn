---
title: "能力涌现：连续缩放如何产生突然的评测拐点"
tags: ["why-models-learn"]
---

能力涌现是一个关于评测曲线的操作性描述：某项任务在较小模型上未达到预设成功标准，在较大模型上达到该标准。这个描述把模型、训练规模、任务样本、提示词、解码规则、指标和阈值绑在一起；它可以反映真实的能力积累，也可能由离散指标、有限样本、提示方式或任务组合放大成突然的拐点。研究涌现时应先固定评测协议，再区分潜在能力曲线、可观察行为和报告阈值。

![能力涌现示意图：潜在成功率随模型规模平滑上升，任务组合和离散阈值把连续曲线报告成突然拐点](/assets/pretraining/svg/emergent-abilities.1.svg)

## “涌现”先写成操作性事件

设模型家族中的规模变量为 $N$，任务评测协议为 $\mathcal E$，观察到的指标为 $A(N;\mathcal E)$，成功阈值为 $\tau$。一次报告中的“能力出现”可以写成：

$$
A(N_{\mathrm{small}};\mathcal E)<\tau,
\qquad
A(N_{\mathrm{large}};\mathcal E)\geq\tau.
$$

这只描述了两个规模点和一个阈值之间的关系。它还没有确定：

- 模型内部是否出现了新的表示结构；
- 潜在成功概率是否在某个规模发生不连续变化；
- 更大的模型是否只是让已有行为更可靠；
- 评测指标是否把连续的概率压成了 0/1；
- 提示、解码、样本量和污染是否改变了观察结果。

[Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) 使用“较小模型没有、较大模型有”的操作性定义讨论这类现象。后续工作 [Are Emergent Abilities of Large Language Models a Mirage?](https://proceedings.neurips.cc/paper_files/paper/2023/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf) 说明，指标的非线性和报告方式可以制造突然拐点。两类证据承担不同职责：前者定义要观察的行为，后者提醒我们检查测量变换。

## 四个对象要分开

### 潜在成功概率

假设模型对一个固定样本产生正确答案的概率是 $p(N)$。它可以随规模平滑变化：

$$
p(N)
=
\frac1{1+\exp\left[-a\left(\log_2N-b\right)\right]},
\qquad
a>0.
$$

这里 $a$ 控制曲线斜率，$b$ 控制中心位置。这个函数只是一种可解释的示例；它说明“看起来突然”不需要潜在函数真的不连续。

### 观察到的行为

模型在有限样本上运行一次，得到正确数、错误数或连续分数。若评测有 $n$ 个样本，正确数可以写成

$$
K\sim\operatorname{Binomial}
\left(
n,p(N)
\right),
\qquad
\widehat p=\frac Kn.
$$

$\widehat p$ 会受到抽样噪声影响。模型之间的真实差异小于这个误差范围时，曲线上的跳跃可能来自抽样。

### 报告指标

同一批输出可以经过 exact match、数值容差、人工评分、judge、pass@k、majority vote 或阈值化处理。指标选择改变了从输出到分数的映射：

|层|示例|引入的判断|
| --- | --- | --- |
|输出|文本、数字、轨迹、工具调用|哪些结果算作模型答案|
|解析|正则、标准化、容差、代码执行|怎样从答案提取结果|
|评分|exact match、部分得分、人工等级|错误与部分正确的权重|
|聚合|mean、majority vote、pass@k|多个样本怎样合成一次结果|
|阈值|达到 0.5、达到满分或通过测试|何时报告“具备能力”|

### 评测协议

把模型家族、规模、数据、任务、提示、解码和指标写成协议元组：

$$
\mathcal E
=
\left(
\mathcal M,
N,
D_{\mathrm{train}},
D_{\mathrm{eval}},
P,
G,
S,
R,
\tau
\right).
$$

$\mathcal M$ 是模型家族，$D_{\mathrm{train}}$ 和 $D_{\mathrm{eval}}$ 是训练与评估数据，$P$ 是 prompt，$G$ 是生成或解码规则，$S$ 是评分函数，$R$ 是重复与聚合规则，$\tau$ 是报告阈值。能力曲线应写成 $A(N;\mathcal E)$，而不是只写成 $A(N)$。

## 连续曲线如何变成“突然出现”

### 先看一个平滑的潜变量

取

$$
p(N)
=
\frac1{1+\exp\left[-\left(\log_2N-3\right)\right]}.
$$

它在 $\log_2N$ 轴上平滑增加。规模从 $N=4$ 增加到 $N=8$ 时，$p$ 从约 $0.268941$ 增加到 $0.5$；从 $N=16$ 增加到 $N=32$ 时，$p$ 从约 $0.731059$ 增加到 $0.880797$。每个相邻规模点都发生了有限变化。

如果报告规则是

$$
A_{\mathrm{hard}}(N)
=
\mathbf 1\left[
p(N)\geq0.5
\right],
$$

那么曲线会在 $N=8$ 处从 0 变成 1。这个跳变属于报告函数，潜变量仍然连续。

### 任务组合会增大斜率

有些任务要求连续完成 $k$ 个子步骤。若每个步骤在近似独立条件下成功概率为 $p(N)$，全部完成的概率为

$$
p_{\mathrm{task}}(N)
=
p(N)^k.
$$

当 $0<p(N)<1$ 时，$p(N)^k$ 仍然连续；$k$ 增大后，低成功率区域被进一步压低，曲线在接近高成功率时才快速上升。任务组合可以产生明显拐点，不能仅凭拐点推断模型内部发生相变。

|基础成功率 $p$|单步骤|四步骤全对 $p^4$|
| ---: | ---: | ---: |
|$0.50$|$0.500000$|$0.062500$|
|$0.60$|$0.600000$|$0.129600$|
|$0.70$|$0.700000$|$0.240100$|
|$0.80$|$0.800000$|$0.409600$|
|$0.90$|$0.900000$|$0.656100$|

若最终报告只保留“全步骤通过”或“失败”，前面的部分进展会被隐藏。增加部分得分、逐步骤正确率和中间状态检查，可以观察曲线在组合前的形状。

### exact match 会丢掉部分进展

设模型输出的正确性连续提高，但答案中存在格式差异。exact match 只返回：

$$
S_{\mathrm{exact}}(y,\widehat y)
=
\begin{cases}
1,&\operatorname{normalize}(\widehat y)
=\operatorname{normalize}(y),\\
0,&\text{otherwise}.
\end{cases}
$$

如果一个答案只差单位、括号或中间步骤，exact match 仍然给 0。可以额外记录：

- 数值误差和单位是否通过；
- 子问题通过数；
- 最终答案与中间步骤；
- 代码测试通过数；
- 置信度与校准误差；
- 不同解析器的分数。

不同指标在相同规模点上可能给出不同的“出现位置”。报告能力时应先写原始评分，再写阈值化版本。

## 模型规模是一条实验轴，不是完整原因

### 参数量需要和其他资源一起记录

[缩放定律](../pretraining/scaling-laws/)中，模型参数量、有效训练 token 和训练计算量共同决定训练条件。能力实验至少记录：

|资源轴|需要记录的量|可能混淆的因素|
| --- | --- | --- |
|模型|总参数、active 参数、层数、宽度、词表|架构、初始化、参数共享|
|数据|有效 token、来源混合、重复和质量|数据覆盖、污染、领域比例|
|计算|FLOP、GPU·小时、训练步、实际 token|batch、通信、重算和训练时长|
|优化|optimizer、学习率、warmup、精度|训练是否充分、稳定性和 checkpoint|
|模型家族|同一基座、同一 tokenizer、同一目标|不同预训练配方与数据过滤|
|后训练|监督数据、RL、偏好优化、模板|能力变化属于预训练还是后训练|

只改变参数量而改变了 tokenizer、数据混合或训练 token，观察到的差异无法归因于参数量单轴。跨家族比较时，应先说明这是产品级比较还是缩放曲线。

### 训练是否充分会改变曲线位置

小模型和大模型如果使用不同的 token-to-parameter ratio，较小模型可能欠训练，较大模型可能接近计算最优。此时能力曲线混合了模型容量与训练充分度。需要按有效 token 或 compute 对齐：

$$
A
=
A
\left(
N,D,C,\mathcal E
\right).
$$

在固定 $N$ 的训练扫描中增加 token，可以检查能力是否因继续训练而提升；在固定 compute 的扫描中改变 $N$ 和 $D$，可以检查模型规模与数据分配的作用。一个只按参数排序的表格不足以支持机制结论。

### 下游拐点不等于训练 loss 拐点

训练 loss 和验证 NLL 通常是连续的平均量。下游任务可能使用 exact match、阈值、有限样本或多步骤组合，因此它可以表现出更陡的变化。这个关系允许三种结果：

1. loss 下降而任务分数平滑上升；
2. loss 下降而任务分数在阈值附近快速上升；
3. loss 下降而任务分数几乎不变，直到提示、数据或解码规则改变。

不能从下游拐点反推出 loss 在同一点发生了相同拐点。[困惑度评估](../pretraining/evaluation-perplexity/)提供了语言建模分数的可复核协议；涌现评估需要继续记录 task-level 评分。

## 评测噪声可以制造或隐藏拐点

### 有限样本的置信区间

对 $n$ 个独立近似样本，观察成功率 $\widehat p$ 的近似标准误为

$$
\operatorname{SE}(\widehat p)
=
\sqrt{
\frac{\widehat p(1-\widehat p)}{n}
}.
$$

当 $\widehat p=0.62$、$n=100$ 时，正态近似的 95% 区间为

$$
0.62
\pm
1.96
\sqrt{\frac{0.62(1-0.62)}{100}}
=
[0.524864,\;0.715136].
$$

两个模型的点估计相差 0.03 时，区间可能高度重叠。此时应增加样本、使用配对差值或报告“差异低于当前可分辨范围”，不应直接标记为能力出现。

二项比例接近 0 或 1 时，Wilson 区间或精确区间通常比正态近似更稳。区间方法要写进报告，避免不同工作使用不同区间后比较宽度。

### 同一任务的方差来源

|方差来源|控制方法|必须保留的证据|
| --- | --- | --- |
|样本抽样|冻结样本或重复抽样|sample ID、seed、版本|
|模型随机性|固定 seed、重复 checkpoint 或多次生成|随机状态、重复次数|
|prompt|预注册模板、测试多个模板|原始 prompt、模板版本|
|解码|固定 greedy、temperature、top-k、top-p|解码参数、生成数|
|解析器|固定 normalize 和执行环境|解析规则、版本、失败样本|
|评分器|人工、程序或 judge 重复评估|评分 rubric、judge 配置|
|数据污染|hash、near-duplicate、清洁子集|污染清单、排除规则|

如果模型只在一个 prompt 和一个 seed 上跨过阈值，结果属于单协议观察。稳健的报告需要说明该阈值是否在 prompt、seed、样本和解析规则变化后仍然存在。

### pass@k 和 majority vote 不是同一个分数

若一次任务生成 $k$ 个样本，pass@k 统计至少一个样本通过；majority vote 选择出现次数最多的答案。它们使用不同的聚合函数：

$$
A_{\mathrm{pass}@k}
=
\mathbf 1
\left[
\exists j\in\{1,\ldots,k\}:S(y_j)=1
\right],
$$

$$
A_{\mathrm{vote}}
=
S
\left(
\operatorname{mode}
(y_1,\ldots,y_k)
\right).
$$

增加 $k$ 会改变 pass@k 的上限，也会改变成本和随机性。把 pass@k 的拐点解释成单次 greedy 能力，需要额外证据。评测报告应同时写每题生成数、温度、随机 seed 和聚合规则。

## 平滑指标与离散指标应同时保留

### 能力曲线至少保存三层数值

对同一批输出，建议保存：

1. 连续分数，例如概率、log probability、数值误差或子步骤比例；
2. 离散分数，例如 exact match、是否通过测试或是否达到阈值；
3. 原始输出和失败类别。

连续分数可以帮助判断模型是否逐步接近成功边界；离散分数回答预设任务是否通过；原始输出用于检查解析器和错误类型。只保留第三层的 0/1 结果，无法判断是模型能力、任务组合还是评分规则造成的跳变。

### 阈值要预先声明

若阈值 $\tau$ 在看完曲线后选择，阈值本身参与了结果选择。应在评测前固定：

- 成功定义；
- 部分得分规则；
- 数值容差；
- 通过比例；
- 置信区间；
- 报告“出现”的最小连续规模区间；
- 多重比较和任务筛选规则。

如果任务数量很多，总会有一些任务在噪声中看起来像拐点。预注册任务集合、统一报告所有任务和纠正多重比较，可以降低事后挑选的影响。

### 能力边界需要一个区间

在有限模型网格上，不能精确定位连续阈值。若 $N=8$ 时未通过、$N=16$ 时通过，只能报告观察到的边界：

$$
N_{\mathrm{crit}}
\in
(8,16].
$$

如果在 $N=12$、$14$ 处继续测量，可以缩小观察区间；如果模型家族、数据或训练预算在这些点变化，区间缩小也不代表因果结论更强。

## “真实相变”和“测量拐点”需要不同证据

### 观察到拐点时先检查测量层

按以下顺序排查：

1. 检查 task score 是否从连续原始量阈值化而来。
2. 检查样本量和区间是否能分辨相邻规模。
3. 检查 prompt、解码和解析器是否保持一致。
4. 检查任务是否要求多个子步骤同时成功。
5. 检查模型是否在同一家族、同一 tokenizer 和同一训练预算下。
6. 检查训练数据污染和后训练配方。
7. 检查 loss、token-level likelihood 或部分得分是否同步改变。
8. 重复多个 checkpoint 和 seed，报告所有观察到的曲线。

若连续指标平滑、离散指标陡峭，测量变换足以解释至少一部分现象。若多个独立指标、不同 prompt、不同样本和不同评分方式在相近规模处同时改变，关于能力结构的解释才获得更强证据，但仍需区分相关与机制。

### 用对照实验拆出因素

一个最小对照矩阵可以固定任务与模型家族，只改变一项：

|对照|固定|改变|可以检验什么|
| --- | --- | --- | --- |
|模型规模扫描|数据、tokenizer、训练目标、评测|参数量或宽度|规模与行为曲线|
|继续训练扫描|模型结构和初始 checkpoint|有效 token 或训练 step|训练充分度|
|prompt 扫描|模型、样本、评分|模板和 few-shot 数|提示敏感性|
|解码扫描|模型、prompt、样本|temperature、k、vote|生成随机性与聚合|
|指标扫描|模型、输出、样本|连续分数与 exact match|测量变换|
|清洁集对照|模型与评测代码|污染样本与清洁子集|记忆与泛化边界|

每个对照都需要固定未改变的字段。只做一个大模型和一个小模型的示例对比，无法建立曲线，也无法区分单点异常。

## 一个标准库探针

下面的代码构造一个平滑的潜在成功率、一个需要四个子步骤全对的任务概率、阈值化结果和有限样本标准误。它只演示测量机制，不代表任何真实模型或真实任务的能力曲线。

```python
from math import exp, log2, sqrt


def latent_success(model_size):
    log_size = log2(model_size)
    return 1.0 / (1.0 + exp(-(log_size - 3.0)))


for model_size in (1, 2, 4, 8, 16, 32):
    latent = latent_success(model_size)
    four_step = latent ** 4
    thresholded = int(four_step >= 0.5)
    standard_error = sqrt(
        four_step * (1.0 - four_step) / 100.0
    )
    print(
        f"N={model_size:2d} "
        f"latent={latent:.6f} "
        f"four_step={four_step:.6f} "
        f"thresholded={thresholded} "
        f"se100={standard_error:.6f}"
    )

observed = 0.62
sample_size = 100
standard_error = sqrt(
    observed * (1.0 - observed) / sample_size
)
lower = observed - 1.96 * standard_error
upper = observed + 1.96 * standard_error
print(
    f"proportion={observed:.2f} "
    f"n={sample_size} "
    f"approx_95ci=({lower:.6f},{upper:.6f})"
)

for score in (0.48, 0.52):
    print(
        f"score={score:.2f} "
        f"pass_threshold={int(score >= 0.5)}"
    )
```

输出为：

```text
N= 1 latent=0.047426 four_step=0.000005 thresholded=0 se100=0.000225
N= 2 latent=0.119203 four_step=0.000202 thresholded=0 se100=0.001421
N= 4 latent=0.268941 four_step=0.005232 thresholded=0 se100=0.007214
N= 8 latent=0.500000 four_step=0.062500 thresholded=0 se100=0.024206
N=16 latent=0.731059 four_step=0.285633 thresholded=0 se100=0.045172
N=32 latent=0.880797 four_step=0.601871 thresholded=1 se100=0.048951
proportion=0.62 n=100 approx_95ci=(0.524864,0.715136)
score=0.48 pass_threshold=0
score=0.52 pass_threshold=1
```

潜在成功率在每个规模点都连续增加。四步骤全对的概率把中间进展压低，阈值化报告又把 $0.48$ 和 $0.52$ 变成两个类别。样本量为 $100$、观察比例为 $0.62$ 时，近似 95% 区间为 $[0.524864,0.715136]$；点估计需要和这个区间一起解释。

## 运行方法

把上一节代码保存为 emergent_abilities_probe.py，然后运行：

```bash
python3 emergent_abilities_probe.py
```

接入真实评测时，把 latent_success 替换为实际模型输出的连续分数或逐题结果，并保存模型规模、训练 token、prompt、解码、样本 ID、评分函数和 seed。先输出连续分数，再生成阈值化报告。不要只保留“出现/未出现”的最终标签。

## 失效模式

### 把单个大模型的高分称为涌现

一次大模型高分只说明一个协议下的一个点。没有同家族的小模型、相邻规模、重复运行和不确定性，无法判断曲线形状。至少需要一个规模网格和固定评测协议。

### 把 exact match 跳变当作内部相变

exact match 会丢掉部分正确和格式接近的答案。先检查连续分数、子步骤和原始输出，再讨论能力边界。离散结果的拐点可以来自评分函数。

### 事后选择任务和阈值

在许多任务中挑出曲线最陡的一条，会增加选择偏差。预先固定任务集合和阈值，报告全体任务，并说明多重比较方法。

### 把参数量当作唯一规模

参数量变化经常伴随训练 token、数据混合、tokenizer、优化器或后训练变化。把这些资源分别记录，使用受控模型家族进行归因。

### 忽略训练是否完成

大模型可能消耗了更多有效 token，或小模型在相同 compute 下更接近欠训练。按 token、compute 和 checkpoint 对齐，再解释能力曲线。

### 用单一 prompt 代表指令能力

模板、few-shot 示例、角色消息和输出格式都会改变结果。多模板复测，并把 prompt 当成评测协议的一部分。

### 用 pass@k 代表单次生成

pass@k 允许多次尝试，通常高于单次 greedy。报告生成次数、采样参数和聚合规则，保持指标含义稳定。

### 把 PPL 直接当下游能力

PPL 是 teacher-forced next-token likelihood。它能诊断语言建模目标，却不能覆盖事实、推理、工具使用和安全行为。保留独立任务指标和错误分类。

### 忽略污染证据

训练与评估的重叠会让高分更容易出现。保存 exact/near duplicate 检查、清洁子集和版本信息，分别报告受污染与未污染结果。

### 把评测器错误归因于模型

解析器、judge、数值容差和执行环境可能制造分数变化。保留失败样本、解析日志和评分器版本，先复核评测器。

## 能力涌现审计清单

|范围|确认项|证据|
| --- | --- | --- |
|定义|操作性成功标准、阈值和任务集合预先固定|评测协议、预注册记录|
|规模|模型家族、参数量、active 参数、训练 token 和 compute 可比|模型与训练 manifest|
|数据|评估集、来源、重复和污染状态明确|sample ID、hash、清洁子集|
|提示|prompt、few-shot、模板和上下文长度固定或成组比较|prompt 文件、版本 hash|
|解码|temperature、top-k、top-p、生成次数和 vote 规则明确|generation config、seed|
|指标|连续分数、离散分数、解析器和阈值同时保存|原始输出、评分日志|
|统计|样本量、重复次数、区间和 bootstrap 单位报告|seed、CI、group|
|曲线|相邻规模点足够、未通过与通过边界有误差范围|scale grid、边界区间|
|对照|loss、PPL、子步骤和下游指标并列|连续曲线、task metrics|
|解释|测量拐点、任务组合、训练充分度和机制假设分开|限制项、对照实验|

只有在评测协议、连续指标和统计证据同时可见时，才能把“能力涌现”作为一个可复核的行为现象。关于模型内部机制的结论需要额外的干预、表征或因果证据。

## 相关词条

- [缩放定律](../pretraining/scaling-laws/)
- [困惑度评估](../pretraining/evaluation-perplexity/)
- [预训练](../pretraining/pretraining/)
- [模型容量](../learning-framework/model-capacity/)
- [双下降](../learning-framework/double-descent/)
- [训练数据](../pretraining/training-data/)
- [训练、验证与测试集](../learning-framework/train-validation-test/)
