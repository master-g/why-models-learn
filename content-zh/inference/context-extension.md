---
title: "上下文扩展：把训练长度外的 token 放进位置合同"
tags: ["why-models-learn"]
---

上下文扩展把模型原本训练或配置的长度 $T_{\mathrm{train}}$ 延伸到更大的推理或训练长度 $T_{\mathrm{target}}$。它需要修改位置坐标、RoPE 频率、训练数据或运行时规则，使模型在新位置上仍能读取相对关系。把服务端最大 token 数改大只改变了入口限制；位置外推、局部距离压缩、KV cache、长序列训练和质量回归仍需要单独验证。

![上下文扩展示意图：原训练位置区间经过位置映射或频率缩放后覆盖更长推理区间，并保留局部与远距离关系](/assets/inference/svg/context-extension.1.svg)

## 先固定四个长度

### 训练长度、目标长度和窗口上限不同

设模型训练时常见的最大长度为 $T_{\mathrm{train}}$，计划扩展到 $T_{\mathrm{target}}$，服务运行时允许的最大长度为 $T_{\max}$。至少要记录：

|长度|含义|决定什么|
|---|---|---|
|$T_{\mathrm{train}}$|原训练数据和位置机制主要覆盖的长度|模型已见的位置分布|
|$T_{\mathrm{target}}$|扩展方案希望支持的长度|位置映射、训练或校准范围|
|$T_{\max}$|checkpoint、runtime 和显存允许的上限|请求是否可以进入|
|effective length|固定任务和阈值下仍可靠的长度|实际能力边界|

理想情况下：

$$
T_{\mathrm{train}}
\leq
T_{\mathrm{target}}
\leq
T_{\max}.
$$

$T_{\max}$ 可以大于 $T_{\mathrm{target}}$，但未经过验证的区间不应被报告为已支持。effective length 由长度、位置、任务和运行模式共同决定。

### 扩展因子要写进配置

定义扩展因子：

$$
r=\frac{T_{\mathrm{target}}}{T_{\mathrm{train}}}.
$$

当 $r=8$ 时，目标位置范围是原范围的 8 倍。这个数字还不能说明算法，因为还要知道：

- position ID 是否重映射；
- RoPE 的 base 或频率是否改变；
- scale 是否按请求动态计算；
- 是否继续训练长序列；
- attention mask 和 cache offset 是否改变；
- short-context 回归是否通过。

同一个 $r$ 可以对应不同位置合同。配置必须保存实际映射函数和频率参数。

### 输出预算也要重新分配

扩展 prompt 后，可留给生成的 token 数可能减少：

$$
T_{\mathrm{prompt}}+T_{\mathrm{new}}\leq T_{\max}.
$$

如果服务只把 prompt 上限扩大，却继续使用原来的 max new tokens，实际请求可能超窗。若把 max new tokens 缩小，质量和延迟比较又会发生变化。长 prompt、输出预算、stop sequence 和 timeout 必须同时冻结。

## 位置外推与位置插值

### 直接外推把新位置原样送入模型

直接外推令：

$$
f_{\mathrm{direct}}(p)=p.
$$

原来训练到 $T_{\mathrm{train}}-1$ 的模型，在新位置 $p\geq T_{\mathrm{train}}$ 继续计算位置编码。它保留原始位置间距，但会访问训练中没有覆盖的坐标。

这种方案的风险取决于位置机制。learned absolute embedding 可能没有新位置的向量；RoPE 会产生训练区间之外的相位；相对位置 bias 可能进入未覆盖的距离桶。直接外推不是一个统一算法，必须结合位置机制描述。

### 线性插值把新坐标压回训练区间

线性位置插值把目标位置 $p$ 映射到：

$$
f_{\mathrm{linear}}(p)
=
p\frac{T_{\mathrm{train}}}{T_{\mathrm{target}}}
=
\frac{p}{r}.
$$

若 $p$ 的范围为 $0$ 到 $T_{\mathrm{target}}-1$，映射后的最大坐标小于 $T_{\mathrm{train}}$。模型可以继续读取原训练位置范围内的 embedding 或 RoPE 相位。

代价是相对距离也被压缩：

$$
\Delta p'
=
\frac{\Delta p}{r}.
$$

在 $r=8$ 时，原来相距 32 token 的局部关系在映射坐标中只相距 4；原来相距 4,096 token 的关系映射为 512。模型的局部模式和远距离模式都需要重新解释。

### 线性插值的数值例子

当 $T_{\mathrm{train}}=4096$、$T_{\mathrm{target}}=32768$ 时，$r=8$：

|原始目标位置 $p$|线性映射 $f_{\mathrm{linear}}(p)$|直接外推坐标|
|---:|---:|---:|
|0|0.000000|0|
|2,048|256.000000|2,048|
|4,095|511.875000|4,095|
|8,192|1,024.000000|8,192|
|16,384|2,048.000000|16,384|
|32,767|4,095.875000|32,767|

线性插值保持坐标落在训练范围，但改变了每个相对距离。短上下文回归需要验证：原来 2,048 token 的距离是否被压缩后仍能表达，原来接近的 token 是否因量化或频率变化发生排序差异。

## RoPE scaling 先写出相位合同

### 原始 RoPE 的频率

对 head dimension $d_h$ 的第 $i$ 个二维坐标对，原始频率可以写成：

$$
\omega_i
=
\mathrm{base}^{-2i/d_h}.
$$

位置 $p$ 的旋转角度为：

$$
\theta_i(p)=p\omega_i.
$$

小 $i$ 对应较高频率，大 $i$ 对应较低频率。不同频率共同编码局部和远距离关系，改变任意一组频率都会改变 query-key 的相位差。

### 位置映射与频率缩放可以等价表示

如果采用线性映射 $p'=p/r$，则：

$$
\theta_i'(p)
=
p'\omega_i
=
p\frac{\omega_i}{r}.
$$

因此也可以保持原始位置 $p$，把频率改为：

$$
\omega_i'=\frac{\omega_i}{r}.
$$

这两种写法在数学上表达同一类相位缩放，工程实现可能选择 position index mapping 或 frequency table scaling。验收时要读取 kernel 实际使用的坐标和频率，不能只看配置字段名称。

### 改 base 会改变整组频率

另一种写法是使用新的 base：

$$
\omega_i'
=
\mathrm{base}'^{-2i/d_h}.
$$

base' 的选择决定不同频率被改变的程度。若所有频率按同一比例变化，局部距离和远距离都会受到相同的坐标压缩；若按频率分段或非线性变化，则不同 head、不同维度的相对位置信号会发生不同变化。

配置至少应包含：

|字段|说明|缺失时的影响|
|---|---|---|
|原始 base|checkpoint 使用的 RoPE base|无法复原原相位|
|目标 base 或频率表|扩展后实际使用的值|无法复现角度|
|scale factor|位置或频率缩放因子|无法复原目标坐标|
|低高频规则|是否分段、平滑或保持部分频率|无法解释局部差异|
|max position|运行时允许的目标长度|无法判断是否超出映射|
|attention factor|是否对 attention score 做额外缩放|logits 和概率可能变化|

### 高频和低频的折中

高频分量变化快，适合表达局部距离，也更容易在长位置积累相位变化。低频分量变化慢，适合表达较远的相对关系，也可能在扩展后保留更稳定的全局结构。

常见的扩展策略会对不同频率使用不同处理，例如：

- 只对部分频率缩放；
- 对缩放区间做平滑过渡；
- 保留局部高频，拉伸低频；
- 先使用位置插值，再用长序列数据继续训练；
- 对 attention logits 使用额外的校准因子。

这些策略的具体函数必须写入配置。只说“使用 RoPE scaling”无法确定局部距离、相位和 logits 的变化。

## 训练外推与继续训练

### 只改推理配置的边界

只改位置映射、base 或 max position，不改变模型权重，属于推理时上下文扩展。它的优点是成本低、checkpoint 不变；风险是模型没有在新位置分布上更新过参数。

此时至少要验证：

|验证层|测试内容|
|---|---|
|算子层|position ID、RoPE angle、mask、cache offset|
|短上下文|原有长度的 logits、token、任务分数|
|长上下文|目标长度各位置桶的检索和生成|
|系统层|prefill、decode、KV cache、峰值显存|
|数值层|不同 dtype、kernel、batch 和 request order|

推理配置没有改变模型的语言知识，也没有自动增加模型能稳定使用的上下文范围。

### 继续预训练改变了权重

继续预训练在更长 token 序列上更新模型，让权重接触新的位置分布和长距离依赖。训练合同至少包含：

- 长序列样本的来源和比例；
- packed sequence 与文档边界；
- 位置机制和扩展映射；
- loss mask 与有效 token 分母；
- batch、梯度累积和 optimizer state；
- 学习率、训练 token 和 checkpoint；
- 短上下文回归集与长上下文评测集。

长序列训练的 token 数增加，不代表有效长上下文能力按同一比例增加。训练数据可能包含大量局部依赖，模型仍然缺少跨段证据组合。

### 长序列训练需要控制采样分布

|采样变量|短序列偏置|长序列合同|
|---|---|---|
|长度|大多数样本接近短窗口|显式覆盖目标长度分桶|
|证据位置|答案接近末尾|覆盖开头、中间、结尾|
|文档结构|单段或随机拼接|保留标题、段落和文档边界|
|依赖距离|局部 token 依赖|加入跨段、跨文档和多跳样本|
|语言与领域|高频语料占主导|按目标部署分布配比|
|loss mask|padding 或重复 token 进入分母|只对有效 target 计分|

若训练数据只把短文档随机拼接到长窗口，模型可能学到位置长度，却没有学到文档内的长距离关系。数据构造和 loss mask 必须分别审计。

### 短上下文回归不能省略

扩展方案可能改善长位置，却改变短位置的相位或注意力分布。回归集至少包括：

|回归项|目的|
|---|---|
|原始长度 perplexity|检查局部语言建模|
|短 prompt 逐 token logits|检查位置与算子合同|
|格式和工具调用|检查 schema、stop 和结构化输出|
|短文档检索|检查原有定位能力|
|greedy 与 sampling|检查选择器边界|
|KV cache 对齐|检查 prefill/decode 一致性|

长上下文分数提高但短上下文回退时，结论应报告为能力迁移，而不是单一性能增加。

## 动态扩展规则影响请求和 cache

### 动态 factor 必须有确定输入

有些运行时按当前请求长度选择 scale factor。设请求有效长度为 $T$，实际规则为：

$$
r(T)=g(T,T_{\mathrm{train}},T_{\max}).
$$

如果同一请求的 prefill 使用一个 $r(T)$，后续 decode 又因为长度增长选择另一个 factor，历史 K/V 的相位合同可能不再一致。动态规则必须在请求开始时冻结，或者明确支持随长度变化的 cache 重映射。

### batch 中的请求必须共享可解释的合同

连续 batching 可能把不同长度、不同扩展因子和不同 position rule 的请求放进同一 kernel。可行方案包括：

- 按扩展因子分桶；
- 按位置规则分离 batch；
- 把 factor 作为显式 batch 轴；
- 在进入 kernel 前完成相位表选择；
- 禁止把不同合同的 cache 混用。

请求重排不能改变自己的 position ID、cache offset 或 scale。动态 batch 只改变调度顺序，不应改变位置语义。

### prefix cache 需要包含扩展配置

相同 token 前缀只有在以下字段一致时才能复用：

|字段|需要一致|
|---|---|
|checkpoint|权重和位置机制版本|
|tokenizer/template|token ID 和模板格式|
|position rule|linear、frequency、piecewise 或 direct|
|factor|$r$ 或动态规则的最终值|
|mask|padding、causal、document boundary|
|dtype/kernel|RoPE、attention 和 cache 计算路径|
|cache layout|page、block、scale 和 logical length|

只用原始 token hash 作为 prefix cache key，会把不同扩展合同的状态错误复用。

## 扩展长度的资源账仍然成立

### 位置扩展不会消除 KV cache

上下文扩展改变可用位置范围，KV cache 仍按历史 token 数增长。以 GQA-8、FP16、$L=32$、$B=1$、$D_{kv}=1024$ 为例，扩展到 32,768 token 时，主 KV payload 为：

$$
M_{\mathrm{KV}}
=
2\times32\times1\times32768\times1024\times2
=
4{,}294{,}967{,}296\ \text{bytes}.
$$

位置映射没有减少 cache 元素。量化、分页、滑动窗口和检索式读取属于独立的资源方案，需要单独报告。

### 长 prefill 的二次项仍然存在

即使位置通过插值回到训练范围，dense prefill 的逻辑 attention 交互仍按目标 token 数计算：

$$
\operatorname{MAC}_{\mathrm{attn}}
\approx
2BT_{\mathrm{target}}^2D.
$$

FlashAttention 可以控制中间 score 的物理峰值，不能把 dense 交互自动变成线性。若实际使用 local 或 sparse pattern，必须把可读 block 图和实际 MAC 单独列出。

### 扩展收益需要三组证据

|证据|应该比较|结论边界|
|---|---|---|
|位置|原始与扩展后的 angle、相对距离、position ID|位置合同是否改变|
|质量|短/长长度、位置、任务、生成协议|模型是否仍能利用新范围|
|系统|TTFT、ITL、KV bytes、峰值显存、吞吐|资源成本是否可接受|

只有位置表和 max position 时，只能说明入口允许更长 token。只有长任务分数时，还不能解释短上下文回归和系统成本。

## 运行方法

下面的探针比较 $T_{\mathrm{train}}=4096$ 到 $T_{\mathrm{target}}=32768$ 的线性位置插值与直接外推，并计算三个 RoPE 频率的角度。它不模拟完整 attention，只核对位置合同和长度账。

```python
train_length = 4096
target_length = 32768
ratio = target_length / train_length
positions = [0, 2048, 4095, 8192, 16384, 32767]

print("train_length=", train_length)
print("target_length=", target_length)
print("extension_ratio=", f"{ratio:.1f}")
for position in positions:
    print(
        "position=",
        position,
        "linear_mapped=",
        f"{position / ratio:.6f}",
        "direct_extrapolation=",
        position,
    )

base = 10000.0
dimension = 128
for index in (0, 16, 63):
    omega = base ** (-2 * index / dimension)
    direct_angle = 32767 * omega
    linear_angle = (32767 / ratio) * omega
    print(
        "frequency_index=",
        index,
        "omega=",
        f"{omega:.12f}",
        "direct_angle=",
        f"{direct_angle:.6f}",
        "linear_angle=",
        f"{linear_angle:.6f}",
    )

print("local_distance_32_linear=", f"{32 / ratio:.6f}")
print("global_distance_4096_linear=", f"{4096 / ratio:.6f}")
print("kv_cache_bytes_target_fp16=", 2 * 32 * 1 * target_length * 1024 * 2)
```

运行输出：

```text
train_length= 4096
target_length= 32768
extension_ratio= 8.0
position= 0 linear_mapped= 0.000000 direct_extrapolation= 0
position= 2048 linear_mapped= 256.000000 direct_extrapolation= 2048
position= 4095 linear_mapped= 511.875000 direct_extrapolation= 4095
position= 8192 linear_mapped= 1024.000000 direct_extrapolation= 8192
position= 16384 linear_mapped= 2048.000000 direct_extrapolation= 16384
position= 32767 linear_mapped= 4095.875000 direct_extrapolation= 32767
frequency_index= 0 omega= 1.000000000000 direct_angle= 32767.000000 linear_angle= 4095.875000
frequency_index= 16 omega= 0.100000000000 direct_angle= 3276.700000 linear_angle= 409.587500
frequency_index= 63 omega= 0.000115478198 direct_angle= 3.783874 linear_angle= 0.472984
local_distance_32_linear= 4.000000
global_distance_4096_linear= 512.000000
kv_cache_bytes_target_fp16= 4294967296
```

这个输出只验证位置和 payload 公式。它不能证明任何扩展方案已经保持任务质量；质量仍需使用固定 checkpoint、tokenizer、位置桶、任务和生成协议独立测量。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|max position 增大但模型不稳定|仍在使用 direct extrapolation 或 position table 未扩展|打印最终 position 和 angle|
|长位置检索下降|相位外推、局部距离压缩或中间位置竞争|按长度和位置桶配对|
|短 prompt 结果改变|扩展 factor 全局影响了短位置|逐 token logits 回归|
|decode 结果与 prefill 不同|prefill/decode 使用不同 factor 或 cache offset|比较同一前缀的 logits|
|动态 batch 结果漂移|请求被放入不同扩展合同的 kernel|记录 batch factor、position rule|
|prefix cache 命中后答案异常|cache key 缺少扩展配置|清空 cache 与命中路径对比|
|长训练 loss 下降但任务不升|数据缺少远距离依赖或评测错位|检查依赖距离和任务位置|
|显存随长度耗尽|KV cache、workspace 或 batch 没有重算|按阶段记录 bytes|
|局部任务下降|高频相位、短距离分辨率或 attention factor 改变|固定短距离 probe|

### 最小上下文扩展审计表

|检查项|应保存|验收问题|
|---|---|---|
|长度|$T_{\mathrm{train}}$、$T_{\mathrm{target}}$、$T_{\max}$、effective length|目标范围是否有证据|
|位置|position ID、映射函数、RoPE base、频率表|kernel 使用了什么坐标|
|距离|局部与远距离 probe|相对距离如何改变|
|训练|长序列数据、长度分布、loss mask、token 预算|权重是否接触新分布|
|运行|dtype、batch、factor、attention、cache|请求合同是否一致|
|质量|短/长 PPL、检索、位置、多跳、生成|能力变化发生在哪里|
|资源|TTFT、ITL、KV bytes、workspace、吞吐|扩展成本是多少|
|缓存|checkpoint、tokenizer、模板、position rule、factor|prefix cache 是否安全|
|失败|超窗、OOM、fallback、重试、数值异常|边界是否显式报告|

上下文扩展的核心是重新定义位置坐标与训练分布之间的关系。线性插值把新位置压回旧范围，频率缩放改变 RoPE 相位，继续训练改变权重适应的长度分布；三者都不能单独推出 effective context。最终结论必须同时给出位置合同、短长质量曲线、cache/attention 资源和失败边界。

## 相关词条

- [长上下文](../inference/long-context/)：按长度、证据位置、干扰项和任务评测 effective context。
- [RoPE](../transformer-components/rope/)：推导旋转位置编码的频率、相位和相对距离。
- [ALiBi](../transformer-components/alibi/)：比较按距离加入 score 偏置的位置机制。
- [推理数学](../inference/inference-math/)：核对长 prefill、decode、attention 和 KV cache 资源账。
- [KV cache](../inference/kv-cache/)：说明扩展长度后历史 K/V 的 shape、分页和 bytes。
- [推理](../inference/inference/)：固定 checkpoint、batch、停止条件和端到端协议。
- [预训练](../pretraining/pretraining/)：说明训练 token、数据分布和 next-token 目标。
- [训练数据](../pretraining/training-data/)：检查长序列样本、文档边界、语言和领域配比。
