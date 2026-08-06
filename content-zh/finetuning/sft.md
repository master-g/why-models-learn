---
title: "监督微调：把示范数据变成可审计的 next-token 更新"
tags: ["why-models-learn"]
---

监督微调（supervised fine-tuning，SFT）从一个已有的预训练 checkpoint 出发，用带有输入与目标答案的示范数据更新模型参数，使模型在指定的任务、格式和对话协议下提高目标 token 的条件概率。SFT 的训练信号来自数据中给出的 target，它不等价于偏好比较、奖励建模或强化学习；效果由 checkpoint、tokenizer、chat template、loss mask、数据分布、优化器和评测生成协议共同决定。

![监督微调示意图：带角色和答案的示范经过 chat template、tokenizer、shift 与 loss mask，只在目标 token 上计算 next-token loss，再更新模型参数](/assets/finetuning/svg/sft.1.svg)

## 从预训练目标切换到示范目标

### SFT 仍然预测下一个 token

预训练通常把一段文本的许多位置都作为 next-token 目标。SFT 也使用同一个自回归分解，但数据项包含条件输入 $x_i$ 和目标序列 $y_i$：

$$
p_\theta(y_i\mid x_i)=\prod_{t=1}^{T_i}p_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

这里的 $x_i$ 可以是用户问题、系统约束、对话历史或任务输入，$y_i$ 是示范答案。模型在训练时看到正确的目标前缀，这仍是 teacher forcing；推理时却要使用自己的前缀，二者的边界需要单独评测。

设目标序列的第 $t$ 个位置是否计入损失由 $m_{i,t}\in\{0,1\}$ 指定，assistant-only SFT 的单条损失为：

$$
\mathcal L_i(\theta)
=
-\frac{\sum_{t=1}^{T_i}m_{i,t}\log p_\theta(y_{i,t}\mid x_i,y_{i,<t})}{\sum_{t=1}^{T_i}m_{i,t}}.
$$

当 $m_{i,t}=0$ 时，该位置不贡献 loss，也不贡献对应的梯度。mask 的语义不是“模型看不见这个 token”；模型仍可能把 prompt 作为条件读入，只是 prompt token 不作为监督目标。

### 输入、标签和 mask 要同时检查

以一个简单的问答项为例：

|序列部分|示例 token|作为输入|作为目标|loss mask|
|---|---|---|---|---:|
|system|你是计算助手|是|否|0|
|user|计算 2 加 3|是|否|0|
|assistant|5|是|是|1|
|EOS|结束标记|是|可选|0 或 1|

实际实现通常把完整序列右移一位。设模板化后的 token 序列为 $s_0,s_1,\ldots,s_T$，输入和标签分别是：

$$
u_t=s_t,\qquad v_t=s_{t+1}.
$$

模型从 $u_t$ 产生 logits，交叉熵比较 $v_t$。因此 mask 必须跟随标签位置移动，不能只按照原始消息字符串的下标切分。一个常见错误是把 assistant 起始 token 的位置偏移一位，导致第一个答案 token 没有监督，或把最后一个输入 token 当作答案。

### 三种监督口径

|口径|prompt token|assistant token|适用场景|主要风险|
|---|---|---|---|---|
|full-sequence loss|计入|计入|需要继续建模完整格式的简单数据|模型可能把容量用在复述 prompt|
|assistant-only loss|不计入|计入|指令跟随和对话 SFT 的常用口径|mask 错误时可能出现空监督|
|last-turn-only loss|历史轮次不计入|只计入最后答案|多轮数据只训练当前响应|早期 assistant 行为不会得到直接更新|

这三种口径都可以训练出参数更新，但 loss 数值不能直接横向比较。报告结果时要保存 mask 规则、有效 target token 数和归约方式。

## 数据项是训练合同

### chat template 不是展示层格式

对话数据至少要记录 role、content、顺序和终止标记。一个抽象数据项可以写成：

$$
d_i=(M_i,\tau_i,\operatorname{Tok}_i,\mu_i,w_i,s_i),
$$

其中 $M_i$ 是消息列表，$\tau_i$ 是 chat template，$\operatorname{Tok}_i$ 是 tokenizer 版本，$\mu_i$ 是 token-level loss mask，$w_i$ 是样本权重，$s_i$ 是来源和切分信息。训练前把消息渲染成字符串，训练时再由 tokenizer 产生整数 token；评测时必须使用同一套渲染规则。

### 模板字段需要版本化

|字段|需要保存的内容|缺失时无法确认|
|---|---|---|
|roles|system、user、assistant、tool 等角色及顺序|目标答案边界|
|separators|角色标记、换行、特殊 token|token 数和条件前缀|
|generation prompt|是否在末尾追加 assistant 起始标记|推理起点|
|EOS|答案结束标记和停止规则|训练目标是否结束|
|tokenizer|词表、normalization、special token ID|token 对齐|
|loss mask|每个 target token 的监督状态|梯度来源|
|source/split|来源、版本、train/valid/test|数据泄漏和复现|

只把“用户文本”和“答案文本”拼接起来，不保存分隔符和 special token，后续无法判断模型是在学习任务，还是在学习一段不稳定的字符串格式。generation prompt 也不是无关的 UI 字段；它决定模型开始生成时所处的条件前缀。

### 多轮对话要先决定监督范围

设消息序列为：

$$
[\mathrm{system},\mathrm{user}_1,\mathrm{assistant}_1,\mathrm{user}_2,\mathrm{assistant}_2].
$$

可以对 assistant_1 和 assistant_2 都计算 loss，也可以只训练最后一个 assistant 响应。前者保留多轮行为的监督信号，后者更接近“给定完整历史，生成当前答案”的目标。无论选择哪一种，都要在数据统计中报告：

- 每条样本有多少 assistant turn；
- 哪些 turn 进入 loss；
- tool call、tool result 和最终答案是否分别计分；
- system 消息是否只作为条件；
- assistant 的 EOS 是否计入。

如果 tool result 被误标为 assistant target，模型会把环境返回内容当成自己的生成行为。如果 assistant 标签跨越了 role separator，模板边界与推理边界就不一致。

### EOS 决定答案何时结束

SFT 的目标序列通常需要一个结束标记。训练中不加入 EOS，模型可能在答案内容结束后继续生成；训练中把多个样本的下一个 prompt 当作同一个答案的延续，模型又可能学到错误的跨样本格式。EOS 是否计入 assistant-only loss，要和部署端的 stop token、最大生成长度一起测试。

## loss mask 决定梯度来自哪里

### 交叉熵梯度直接落在目标分布上

对于一个 target token，设模型 logits 为 $z_{t,k}$，softmax 概率为 $p_{t,k}$，真实 token 的类别为 $y_t$。单 token 交叉熵对 logits 的导数为：

$$
\frac{\partial\ell_t}{\partial z_{t,k}}
=
p_{t,k}-\mathbf 1[k=y_t].
$$

例如 $p_t=(0.7,0.2,0.1)$，真实类别为第 1 类时，梯度为 $(0.7,-0.8,0.1)$。提高真实类别的 logit 会降低 loss，降低其他类别的相对概率也会降低 loss。若这个 token 的样本权重为 $0.5$，三项梯度同时乘以 $0.5$，不会只缩放真实类别。

### 空 mask 与错误 mask 都要显式报错

assistant-only 训练至少应检查每个 batch 的有效监督 token 数：

$$
N_{\mathrm{sup}}=\sum_{i,t}m_{i,t}.
$$

如果 $N_{\mathrm{sup}}=0$，实现可能返回数值为零的 loss，或者在除法时产生 NaN。两种结果都不能当作“这一批没有错误”。训练循环应在 forward 后记录 prompt token、assistant target token、EOS token 和 ignored label 的计数，并在计数为零或异常时停止或标记 batch。

mask 还可能覆盖了错误的 role。只检查 loss 是否下降不够，因为模型可以在大量 prompt token 上获得稳定梯度，而 assistant token 根本没有训练。

### 归约方式改变样本权重

对不同长度的响应，至少有两种常见归约：

$$
\mathcal L_{\mathrm{token}}
=
-\frac{\sum_{i,t}m_{i,t}\ell_{i,t}}{\sum_{i,t}m_{i,t}},
$$

$$
\mathcal L_{\mathrm{example}}
=
\frac{1}{N}\sum_i
\left(
-\frac{\sum_t m_{i,t}\ell_{i,t}}{\sum_t m_{i,t}}
\right).
$$

token mean 让每个有效 token 获得近似相同的权重，长响应自然贡献更多梯度；example mean 让每条样本先求自己的平均，再让每条样本获得相近权重。数据长度分布改变时，这两个 loss 的数值和优化轨迹也会改变。

如果不同任务的响应长度差异很大，混合数据时要说明使用 token mean 还是 example mean。否则提高长答案比例，可能只是改变了长答案 token 的梯度权重。

## padding、packing 与 batch

### padding mask 和 loss mask 不是同一个 mask

padding mask 控制 attention 是否读取无效位置；loss mask 控制某个标签位置是否产生监督。对一个右侧 padding 的 batch：

|mask|作用对象|回答的问题|
|---|---|---|
|attention mask|query-key 连接|当前位置能读取哪些 token|
|causal mask|时间方向|能否读取未来 token|
|loss mask|target label|该位置是否产生梯度|
|document mask|样本边界|是否允许跨样本读取|

一个 padding token 可以同时满足“不能被 attention 读取”和“不能进入 loss”，但两者仍是两个独立条件。只设置 ignore_index 不会自动阻止 attention 读取 padding；只设置 padding mask 也不会让 prompt token 从 loss 中消失。

### packing 改变了可见上下文

为了提高 GPU 利用率，可以把多个短样本放进同一个固定长度序列：

$$
[A\ \mathrm{EOS}]\Vert[B\ \mathrm{EOS}]\Vert[C\ \mathrm{EOS}].
$$

如果 A、B、C 原本是独立样本，attention mask 需要阻止 B 读取 A 的内容，或至少把 document boundary 作为显式条件。否则 B 的答案可能读取 A 的答案，训练 loss 下降却没有对应的独立样本语义。

packing 还有三个需要分开保存的量：

- 物理序列长度：送入 kernel 的 padded 或 packed 长度；
- 逻辑样本边界：每个 token 属于哪条样本；
- 监督 token 集合：哪些标签进入 loss。

只保存 packed token 数，无法恢复样本级 loss、长度分布和错误归因。

### 有效 batch 要按 token 统计

若每个设备的 micro-batch 有 $b_\mu$ 条样本，梯度累积步数为 $a$，数据并行 world size 为 $W$，固定长度下的有效样本数可以写成：

$$
B_{\mathrm{eff}}=b_\mu aW.
$$

如果样本长度不同，更有用的量是一次 optimizer update 中的有效监督 token：

$$
N_{\mathrm{update}}=\sum_{i\in\mathrm{update}}\sum_t m_{i,t}.
$$

它决定了梯度噪声、吞吐和每个 epoch 的 update 数。把 padding token 当成有效 token 会高估 batch size；只按样本数报告又掩盖了响应长度变化。

## SFT 训练会改变什么

### checkpoint 是起点，不是数据的替代品

设预训练参数为 $\theta_0$。SFT 从 $\theta_0$ 开始，按示范 loss 计算梯度并更新：

$$
\theta_{u+1}
=
\theta_u-\eta_u\nabla_\theta\mathcal L_{\mathrm{batch}}(\theta_u).
$$

更新后的参数会提高训练分布中目标答案的概率。它可能改善指令格式、任务行为、输出风格和工具协议，也可能降低原有分布上的 perplexity 或知识覆盖。SFT 没有一个独立开关可以保证“只学会格式，不改变知识”；数据、学习率、训练 token、参数范围和回归集共同决定变化。

### 全量更新与参数高效更新是不同实验

全量 SFT 更新 checkpoint 的大多数或全部参数。参数高效方法只更新 adapter 或低秩参数，基础权重保持冻结。两种方案的可训练参数量、optimizer state、显存、合并方式和部署 checkpoint 都不同。本文只固定 SFT 的监督目标；full fine-tuning 与 PEFT 的参数账在后续词条中单独展开。

### 数据混合比例需要写入实验配置

设数据集分成 $K$ 个来源，来源 $k$ 的采样比例为 $\pi_k$，来源内平均损失为 $\mathcal L_k$，混合目标可以写成：

$$
\mathcal L_{\mathrm{mix}}
=
\sum_{k=1}^{K}\pi_k\mathcal L_k,
\qquad
\sum_{k=1}^{K}\pi_k=1.
$$

实际的 token mean 还受每个来源的响应长度、过滤率、重复率和 packing 方式影响。配置中至少保存：

|变量|需要记录|不记录的影响|
|---|---|---|
|source ratio|样本或 token 的采样比例|无法复现能力权重|
|response length|分位数、截断率和 EOS 率|无法解释 token mean|
|filter rate|去重、质量和安全过滤数量|训练集版本不清楚|
|task coverage|任务、语言、领域和格式分布|回归结果无法归因|
|sampling seed|数据顺序与随机种子|小数据集的波动无法重现|

“混合了更多任务”不是可复核的训练设置。需要说明按样本采样、按 token 采样，还是先分桶后设定每桶预算。

### 训练超参数必须和 token 预算一起看

|训练项|需要固定|为什么|
|---|---|---|
|learning rate|峰值、warmup、scheduler、最小值|决定参数移动尺度|
|optimizer|AdamW 等类型、betas、epsilon、weight decay|决定梯度累积和正则化|
|batch|micro-batch、gradient accumulation、world size|决定 update 的统计量|
|sequence length|最大长度、截断和 packing|决定可见上下文与 token 数|
|precision|FP32、BF16、FP16、loss scaling|决定数值范围和溢出边界|
|steps|optimizer update 数、epoch、有效 token|决定训练预算|
|checkpoint|保存间隔、best 规则、恢复状态|决定回滚和比较对象|
|evaluation|固定切分、生成参数、停止条件|决定指标是否可比|

同一个 epoch 在不同截断率和响应长度下，不代表同一个训练 token 预算。报告 SFT 结果时，应同时给出有效监督 token、update 数和最终 checkpoint。

## 数据质量与切分

### 正确性先于表面格式

一条看起来像指令数据的记录，至少要检查：

- 输入是否明确，答案是否真的回答输入；
- 答案中的事实、计算和代码是否可验证；
- role、separator、EOS 是否符合部署 template；
- 是否包含隐藏的提示词、答案泄漏或评测集内容；
- 是否重复，或只是对同一模板做了微小改写；
- 是否包含超出部署策略的工具调用、隐私字段或无效格式。

格式正确的错误答案也会被 SFT 当作目标分布。loss 下降只能说明模型更接近这组 target，不能说明 target 本身正确。

### train、valid、test 要按信息来源切分

随机按行切分容易把同一问题、同一模板、同一文档或同一用户会话分到不同集合。更稳妥的切分键包括：

|切分键|要避免的泄漏|
|---|---|
|source document|同一文档的段落同时出现在 train 和 test|
|task family|同一模板只改数字就进入不同 split|
|conversation ID|同一会话的历史与答案跨 split|
|user or organization|同一用户模式跨 split|
|time|未来数据混入过去版本的训练|

如果数据必须按 token 随机切分，应至少做 n-gram 或指纹去重，再报告近重复比例。

### 评测集要覆盖训练合同之外的边界

验证集可以估计训练分布内的 loss，测试集还需要覆盖：

- 没见过的指令表达；
- 不同输入长度和多轮深度；
- 结构化输出、工具调用和错误输入；
- 多语言或目标部署领域；
- 不能只靠模板匹配的问题；
- 长答案、短答案和必须提前 EOS 的任务。

如果测试集只复用了训练模板，SFT 可能退化为格式识别，与任务迁移能力无关。

## 评测要沿用训练合同

### loss 比较必须固定 mask 和归约

同一个 checkpoint 可以得到不同的 validation loss：

- 对 prompt 和 assistant 都计分，loss 会包含条件文本；
- 只对 assistant 计分，loss 只反映目标答案；
- 用 token mean，长答案贡献更多 token；
- 用 example mean，每条样本的平均损失权重接近；
- 把 EOS 计入或排除，会改变短答案的平均值。

因此评测报告应包含 mask 规则、有效 token 数、归约方式、tokenizer、chat template、截断策略和 checkpoint hash。

### 生成指标需要固定运行协议

对话模型的任务指标还受 decoding 影响。至少固定：

|协议项|固定内容|
|---|---|
|prompt|system、user、历史轮次和 generation prompt|
|length|max new tokens、输入截断和 stop token|
|decoding|greedy、temperature、top-p、seed|
|format|JSON schema、代码语言、引用和工具接口|
|scoring|exact match、程序执行、规则检查或人工标准|

不能把 greedy 的 exact match 与 sampling 的格式通过率放进同一条曲线。不能只报告生成文本长度而不报告截断和 EOS 情况。

### SFT 前后要做回归矩阵

|维度|基线|SFT 后需要观察|
|---|---|---|
|训练分布|base checkpoint|目标 loss 是否下降|
|原始语言建模|固定 perplexity 集|知识和局部语言能力是否回退|
|任务行为|固定 task suite|正确率、格式和工具成功率|
|长度|短、中、长输入|截断、长答案和停止边界|
|角色|system、user、assistant、tool|模板和 role 规则是否稳定|
|安全边界|拒答和不应执行的输入|是否产生错误遵循|
|鲁棒性|改写、噪声、未见模板|是否只记住表面格式|

长训练时间或更低训练 loss 不能单独作为保留 checkpoint 的理由。应保存基线、候选 checkpoint、数据版本和相同协议下的回归结果。

## 一个可运行的 SFT loss probe

下面的探针只使用 Python 标准库。它比较两条响应的 token mean、整个 batch 的 token mean 和 example mean，并核对 assistant-only mask、单 token softmax 梯度与有效 batch 规模。

```python
import math

examples = [
    [0.6, 0.4],
    [0.8, 0.5, 0.25],
]
loss_sums = [-sum(math.log(probability) for probability in row) for row in examples]
loss_means = [
    loss / len(row) for loss, row in zip(loss_sums, examples)
]

print("example_1_loss_sum=", f"{loss_sums[0]:.9f}")
print("example_1_token_mean=", f"{loss_means[0]:.9f}")
print("example_2_loss_sum=", f"{loss_sums[1]:.9f}")
print("example_2_token_mean=", f"{loss_means[1]:.9f}")
print(
    "batch_token_mean=",
    f"{sum(loss_sums) / sum(map(len, examples)):.9f}",
)
print("example_mean=", f"{sum(loss_means) / len(loss_means):.9f}")

labels = [-100, -100, 1, 2, -100, 3]
mask = [int(label != -100) for label in labels]
print("sequence_length=", len(labels))
print("prompt_loss_tokens=", len(labels) - sum(mask))
print("assistant_loss_tokens=", sum(mask))

probabilities = [0.7, 0.2, 0.1]
target = 1
gradient = [
    probability - int(index == target)
    for index, probability in enumerate(probabilities)
]
weighted_gradient = [0.5 * value for value in gradient]
print(
    "one_token_grad=",
    "[" + ", ".join(f"{value:.1f}" for value in gradient) + "]",
)
print(
    "weighted_grad=",
    "[" + ", ".join(f"{value:.2f}" for value in weighted_gradient) + "]",
)

microbatch = 2
accumulation = 4
world_size = 2
sequence_length = 1024
print("effective_examples=", microbatch * accumulation * world_size)
print(
    "padded_tokens_per_update=",
    microbatch * accumulation * world_size * sequence_length,
)
```

运行输出：

```text
example_1_loss_sum= 1.427116356
example_1_token_mean= 0.713558178
example_2_loss_sum= 2.302585093
example_2_token_mean= 0.767528364
batch_token_mean= 0.745940290
example_mean= 0.740543271
sequence_length= 6
prompt_loss_tokens= 3
assistant_loss_tokens= 3
one_token_grad= [0.7, -0.8, 0.1]
weighted_grad= [0.35, -0.40, 0.05]
effective_examples= 16
padded_tokens_per_update= 16384
```

两条响应共有 5 个 target token，因此 batch token mean 是总 loss 除以 5。example mean 先对两条响应分别求平均，再对两个样本求平均；二者相差 0.005397019。mask 示例中 6 个标签有 3 个进入 loss，另外 3 个位置被当作 prompt 或 padding。梯度示例直接对应 softmax cross-entropy 的 $p-\operatorname{onehot}(y)$ 公式。

## 运行方法

将上一个 Python 代码块保存为 sft-loss-probe.py，再运行 python3 sft-loss-probe.py。它只依赖 Python 标准库，不需要模型权重或网络。修改概率、mask 或 batch 参数后，应同步更新运行输出，并重新检查归约分母。

把这个探针扩展到真实训练时，还需要增加 tokenizer、chat template、shift、padding、packed document boundary 和框架 loss 的逐 token 对照。标准库探针只能核对算术，不能证明 GPU kernel、混合精度或数据 loader 的行为。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|loss 下降但模型不回答|assistant mask 为空或 role 边界错|打印每批各 role 的 supervised token 数|
|回答首 token 经常缺失|shift 后 mask 偏移一位|逐位置比较 input、label、mask|
|输出重复 prompt|prompt 也进入 loss，或 generation prompt 不一致|对 prompt/assistant 分别计算 loss|
|训练结束后不停止|EOS 未进入目标或 stop token 不匹配|统计 EOS 监督率与生成终止率|
|packing 后结果漂移|样本跨 boundary 读取|检查 document mask 和边界位置|
|短答案指标下降|token mean 让长答案占更大权重|比较 token mean 与 example mean|
|验证 loss 很低但测试很差|近重复、模板泄漏或切分键错误|按文档、任务和会话去重|
|工具调用格式错误|tool role、参数 schema 或 target span 不一致|按 role 和字段做结构校验|
|SFT 后原能力回退|学习率、token 预算或混合比例过大|运行 base/SFT 回归矩阵|
|不同运行结果不一致|数据顺序、seed、template 或 checkpoint 状态变化|保存配置、数据版本和 optimizer state|

### 最小审计表

|审计层|应保存|验收问题|
|---|---|---|
|数据|原始记录、过滤规则、去重指纹、版本|训练集是否可恢复|
|模板|chat template、special token、generation prompt|训练和推理是否使用同一格式|
|token|tokenizer、input、label、shift、长度|每个 target 是否对应正确前缀|
|mask|role mask、padding mask、causal mask、document mask|梯度和可见范围是否分开核对|
|loss|token/example mean、权重、EOS 规则|报告数值是否可比较|
|训练|checkpoint、optimizer、LR、有效 token、seed|参数更新是否可复现|
|评测|split、prompt、decode、stop、scorer|SFT 前后是否使用同一协议|
|回归|能力、格式、安全、长度、鲁棒性|是否出现局部改善与其他回退|

SFT 的结论至少应写成“在某个 checkpoint、数据版本、模板、mask、token 预算和生成协议下，某组任务指标发生了某个变化”。只给一个训练 loss 或一个示例回答，无法定位变化来自数据、参数更新还是运行协议。

## 相关词条

- [预训练](../pretraining/pretraining/)：说明 SFT 所继承的 next-token checkpoint 和训练分布。
- [Next-token MLE](../pretraining/next-token-as-mle/)：推导自回归目标与交叉熵的概率含义。
- [因果语言建模](../transformer-architectures/causal-language-modeling/)：说明 decoder-only 模型的 shift、causal mask 和训练前向。
- [分词](../text-representation/tokenization/)：检查字符串、special token、token ID 和 tokenizer 版本。
- [训练数据](../pretraining/training-data/)：展开来源、质量、去重、切分和数据分布。
- [指令数据](../finetuning/instruction-data/)：继续讨论任务格式、示范构造与数据混合。
- [全量微调与参数高效微调](../finetuning/full-vs-peft/)：比较可训练参数、显存和 checkpoint 合并。
- [困惑度评估](../pretraining/evaluation-perplexity/)：固定 masked target、窗口和 perplexity 归约。
- [灾难性遗忘](../finetuning/catastrophic-forgetting/)：评估 SFT 后原有能力和分布的回退。
