---
title: "Prompt Tuning：用软提示条件化冻结模型"
tags: ["why-models-learn"]
---

Prompt Tuning 把一串可训练的连续向量放在输入 embedding 前面，冻结基础模型参数，只更新这些虚拟 token。它不是把一段中文提示词重新 tokenization，也不是只改变推理时的字符串；训练对象是 $P\in\mathbb R^{m\times D}$，前向时它和真实 token embedding 一起进入 Transformer。prompt 长度、位置、mask、模板、KV cache 和任务切换都属于运行合同。

![Prompt Tuning 示意图：可训练的虚拟 token 向量与真实 token embedding 拼接后进入冻结 Transformer，任务 prompt 只保存一小组参数](/assets/finetuning/svg/prompt-tuning.1.svg)

## 软提示和文本提示不是同一个对象

### 文本提示先经过 tokenizer

硬提示是字符串。给定字符串 $p_{\mathrm{text}}$ 和输入 $x$，tokenizer 产生离散 ID：

$$
\operatorname{tokenize}(p_{\mathrm{text}}\mathbin{\|}x)
=
(u_1,\ldots,u_m,x_1,\ldots,x_T).
$$

embedding 矩阵 $E\in\mathbb R^{V\times D}$ 再把每个 ID 映射为向量。硬提示的可选空间受词表和 tokenizer 限制，字符串长度也会改变 token 数量。

软提示直接学习：

$$
P=
\begin{bmatrix}
p_1\\
\vdots\\
p_m
\end{bmatrix}
\in\mathbb R^{m\times D},
\qquad
p_i\in\mathbb R^D.
$$

这些向量不要求等于词表中的某一行，也不需要通过 tokenizer。它们通常不可逆地翻译成自然语言；可审计对象是 prompt tensor 的 shape、dtype、参数值和基础模型标识。

|对象|产生方式|可训练对象|主要约束|
|---|---|---|---|
|hard prompt|字符串加 tokenizer|无或模型参数|词表、tokenizer、文本顺序|
|soft prompt|直接创建连续向量|$mD$ 个参数|维度、位置、dtype|
|prefix tuning|学习各层 prefix K/V|约 $2Lmh_{kv}d_h$ 个参数|层数、head 和 cache|
|LoRA|学习矩阵低秩增量|因子 $A,B$|target module、rank、scale|

### 软提示不是 embedding 表中的新词

把 $P$ 误写回词表会改变共享 embedding、词表大小和 checkpoint 合同。标准 Prompt Tuning 只在指定模型实例的输入路径前拼接 $P$，不增加 vocabulary ID，也不要求生成阶段输出这些向量对应的 token。

## 输入级 Prompt Tuning 的形状合同

### 拼接后的序列轴

真实输入 embedding 为：

$$
X=E(x)\in\mathbb R^{T\times D}.
$$

输入级软提示把它放在前面：

$$
H_0
=
\operatorname{concat}_{\mathrm{seq}}(P,X)
\in\mathbb R^{(m+T)\times D}.
$$

批量后：

$$
H_0\in\mathbb R^{B\times(m+T)\times D}.
$$

每个真实 token 的 hidden 维度仍然是 $D$，变化发生在 sequence 轴。若实现把 prompt 拼到 feature 轴，或者把 $P$ 当成 batch 轴广播，参数量可能仍为 $mD$，但 attention 的输入合同已经错误。

### 位置编号会整体后移

如果 prompt 放在真实 token 前面，常见位置编号为：

$$
\operatorname{pos}(p_i)=i,
\qquad
\operatorname{pos}(x_t)=m+t.
$$

绝对位置 embedding、RoPE 或 ALiBi 都会看到这个偏移。对于 decoder-only 模型，生成 token 的位置还要接在 $m+T$ 之后。训练时省略 prompt 的位置偏移，推理时再插入 prompt，会造成训练和部署的 position contract 不一致。

|位置项|没有 soft prompt|有 $m$ 个 soft token|
|---|---|---|
|第一个真实 token|0 或模板起点|$m$ 或模板起点加 $m$|
|真实序列长度|$T$|$m+T$|
|第一个生成 token|$T$|$m+T$|
|KV cache 起点|真实 token 0|prompt token 0|

不同模型的 position convention 可能从 0 或特殊 token 开始。表中的相对关系稳定，具体起点必须从模型实现和模板中确认。

### mask 要把 prompt 当作条件

对 decoder-only causal mask，prompt token 可以按因果顺序读取前面的 prompt，真实 token 可以读取它之前的 prompt 和真实 token。对于 encoder-only 模型，prompt 和真实 token 通常都在双向 attention 区域内。

prompt token 不应该自动变成监督 target。若训练目标是对真实 assistant token 计算交叉熵，loss mask 应把虚拟 token 的位置标为 0：

$$
\ell_t
=
\begin{cases}
-\log p_\theta(y_t\mid H_0),&m\leq t<m+T_{\mathrm{target}},\\
0,&t<m.
\end{cases}
$$

实现时要同时保存 input position、attention mask、loss mask 和 label shift。只检查输入长度，不能确认 prompt 没有进入损失分母。

## 参数量很小，但序列开销真实存在

### 输入级 prompt 的参数量

输入级 Prompt Tuning 只保存：

$$
P_{\mathrm{prompt}}=mD.
$$

如果基础模型有 $P$ 个参数，prompt 的参数比率为：

$$
\rho_{\mathrm{prompt}}
=\frac{mD}{P}.
$$

以 $m=32,D=4096,P=7\times10^9$ 为例，prompt 只有 131072 个参数，比率为约 0.0018739%。这描述了 checkpoint 和 optimizer 的规模，不代表 prompt 没有运行时成本。

### 深层 prompt 和 prefix tuning 的参数账不同

如果每一层都有一个 $m\times D$ 的深层 prompt，参数量为：

$$
P_{\mathrm{deep}}=LmD.
$$

如果 prefix tuning 直接为每层保存 K、V：

$$
P_{\mathrm{prefix}}
=2Lmh_{kv}d_h.
$$

由于 $D=h d_h$，而 $h_{kv}$ 可能小于 $h$，两种参数量不能只按 prompt token 数比较。输入级 prompt 的向量先经过每层投影，prefix tuning 的 K/V 则直接注入 attention 读取路径。

|方案|训练参数|插入位置|是否增加输入 token|
|---|---:|---|---|
|input soft prompt|$mD$|输入 embedding 前|是|
|deep prompt|$LmD$|多层 hidden 接口|通常不改变原始 token 数|
|prefix tuning|$2Lmh_{kv}d_h$|每层 K/V|不一定改变 token 输入|
|LoRA|$\sum r(d_{\mathrm{in}}+d_{\mathrm{out}})$|目标线性层|否|

### attention 和 cache 要按新长度计算

输入级 prompt 把 prefill 长度从 $T$ 改为 $T'=T+m$。在 dense self-attention 的主项下：

$$
\frac{C_{\mathrm{prompt}}}{C_{\mathrm{base}}}
\approx
\frac{(T+m)^2}{T^2}.
$$

当 $m$ 远小于 $T$ 时，增幅近似为 $1+2m/T$；当输入很短时，同样数量的 prompt token 会产生更大的比例开销。

对 decoder KV cache，prompt token 需要占用：

$$
M_{\mathrm{prompt\ KV}}
=2Lmh_{kv}d_h b_{\mathrm{kv}},
$$

其中前面的 2 表示 K 和 V，$b_{\mathrm{kv}}$ 是每个值的字节数。生成时这些 prompt KV 通常可以和真实 token KV 一样被读取。

|变化|公式|审计字段|
|---|---|---|
|prefill 长度|$T'=T+m$|prompt 是否计入 position 和 mask|
|attention 主项|$(T+m)^2$|batch、长度和实现 kernel|
|prompt KV|$2Lmh_{kv}d_hb_{\mathrm{kv}}$|层数、KV head、dtype|
|prompt train state|$mD(b_w+b_g+b_o)$|optimizer、dtype、分页|

## 初始化和训练信号

### 初始化方式决定起点

常见初始化有两类：

1. 从词表 embedding 中选择一段真实 token 的向量作为起点。
2. 按小方差随机分布直接初始化连续向量。

词表初始化提供一个已存在的尺度参考，但会把起点绑定到 tokenizer 和 embedding checkpoint。随机初始化不要求语义 token，但需要确认均值、方差和基础 hidden 的尺度相容。

初始化的目标不是让 prompt 具有可解释的文字含义，而是让插入 prompt 后的前向和梯度处在可训练范围。应记录初始化来源、随机种子、dtype 和是否做归一化。

### 梯度只沿真实损失回到 prompt

令 Transformer 参数为 $\theta_0$，冻结 $\theta_0$，模型输出为：

$$
Y=f_{\theta_0}([P;X]).
$$

损失对 prompt 的梯度为：

$$
\frac{\partial\mathcal L}{\partial P}
=
\frac{\partial\mathcal L}{\partial Y}
\frac{\partial Y}{\partial [P;X]}
\frac{\partial [P;X]}{\partial P}.
$$

最后一个 Jacobian 只是把 prompt 位置选出来；可训练信号来自 attention、MLP、残差和输出 head 对这些位置的依赖。若 loss 只保留真实 target 位置，prompt 仍能通过上下文路径获得梯度。

应确认：

|检查项|最低证据|
|---|---|
|prompt 参数|数量为 $mD$ 或声明的深层/prefix 参数量|
|基础参数|requires-grad 关闭且不在 optimizer group|
|prompt 梯度|有效 batch 后有非零范数|
|loss mask|虚拟 token 不在 target 分母|
|position|训练、验证和推理使用同一偏移|
|dtype|prompt、embedding、hidden 和 optimizer state 可复现|

### prompt 长度是容量和开销的共同变量

增大 $m$ 会增加可训练维度，也会增加 attention、KV cache 和输入位置占用。减小 $m$ 会降低资源，但可能限制任务条件。比较 prompt length 时，应固定基础 checkpoint、初始化方式、数据切分、有效 token、学习率和模板。

## 多任务 prompt 和缓存

### prompt checkpoint 不能脱离 base 使用

一个 prompt checkpoint 通常只包含：

$$
\operatorname{checkpoint}
=
\{P,\;m,\;D,\;\operatorname{dtype},\;\operatorname{base\ hash},\;\operatorname{template}\}.
$$

它不包含基础 Transformer 权重。加载时至少确认：

1. base model 内容 hash 一致。
2. tokenizer、embedding、位置机制和 chat template 一致。
3. prompt length、dtype、初始化元数据和目标任务一致。
4. input position、attention mask 和 loss mask 合同一致。

如果基础模型版本只更新了词表或 embedding，prompt tensor 仍可能 shape 相同，但语义已经变化。

### 多 prompt 可以共享基础模型

服务端可以在同一个 frozen base 上切换多个任务 prompt。切换时要绑定 prompt ID、base hash、模板、batch 和 cache 状态。若多个请求共享相同 prompt、相同 base 和相同前缀，可以复用 prefix cache；更换 prompt 后，旧 prompt 的 KV cache 不能继续作为新请求的前缀。

|场景|可复用对象|必须失效的对象|
|---|---|---|
|同一 prompt、同一 base|prompt KV、编译图、模板结果|请求级 token 状态|
|换 prompt、同一 base|基础权重、量化权重|prompt KV 和依赖它的 cache|
|同 prompt、换 base|prompt 文件本身可读|所有 hidden、KV 和编译状态|
|同 base、多个 prompt batch|基础权重和 kernel|不同 prompt 的位置和 cache|

### prompt 不会自动解决任务冲突

不同任务各自保存 prompt 可以隔离参数，但它们仍共享基础模型、tokenizer、运行时资源和评测协议。一个任务 prompt 的高分不能证明另一个任务 prompt 的质量。应按 prompt ID 建立任务回归表，并保留未适配 base 作为对照。

## 一个可运行的 Prompt Tuning 资源探针

下面的探针使用 Python 标准库，核对输入级 prompt、深层 prompt、序列长度、attention 主项、prompt KV cache 和 prompt Adam 训练状态。它不实现 Transformer，也不测量 kernel workspace。

```python
D = 4
m = 2
T = 3
L = 2
h_kv = 2
d_h = 2
bytes_per_value = 2

prompt_parameters = m * D
deep_prompt_parameters = L * m * D
sequence_length = T + m
attention_ratio = (sequence_length / T) ** 2
prompt_kv_bytes = 2 * L * m * h_kv * d_h * bytes_per_value
prompt_train_state_bytes = prompt_parameters * (2 + 2 + 8)
full_parameters = 7_000_000_000

print('prompt_parameters=', prompt_parameters)
print('deep_prompt_parameters=', deep_prompt_parameters)
print('sequence_length=', sequence_length)
print('attention_ratio=', f'{attention_ratio:.6f}')
print('prompt_kv_bytes=', prompt_kv_bytes)
print('prompt_train_state_bytes=', prompt_train_state_bytes)
print('full_matrix_parameters=', full_parameters)
print('prompt_ratio_percent=', f'{100 * prompt_parameters / full_parameters:.9f}')
print('prefix_kv_values=', 2 * L * m * h_kv * d_h)
```

运行输出：

```text
prompt_parameters= 8
deep_prompt_parameters= 16
sequence_length= 5
attention_ratio= 2.777778
prompt_kv_bytes= 64
prompt_train_state_bytes= 96
full_matrix_parameters= 7000000000
prompt_ratio_percent= 0.000000114
prefix_kv_values= 32
```

在这个小例子中，输入级 prompt 有 8 个参数，深层 prompt 有 16 个参数。真实 token 长度从 3 变为 5，dense attention 主项变为 25/9，即 2.777778 倍。2 层、2 个 KV head、每头维度 2、FP16 cache 下，2 个 prompt token 的 K/V cache 为 64 bytes。若权重、梯度和 Adam state 分别占 2、2、8 bytes，8 个 prompt 参数的训练状态为 96 bytes。相对 7B 基础参数，输入级 prompt 的参数比率为 0.000000114%。

这些数字只核对 shape 和单位。真实模型还会加入 BOS、system message、模板 separator、padding、RoPE offset、activation 和 kernel workspace。

## 运行方法

将上一个 Python 代码块保存为 prompt-tuning-ledger.py，再运行 python3 prompt-tuning-ledger.py。修改 prompt length、hidden size、层数、KV head 或 dtype 后，应同步检查输入长度、attention 主项、cache 和训练状态。

接入实际训练时，先打印 prompt 参数 shape 和 optimizer group，再用固定 batch 检查 prompt 梯度、loss mask、位置编号和 base 回归。部署前比较硬提示、base、soft prompt 和不同 prompt length 的同协议结果。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|soft prompt 插入后输出大幅变化|初始化尺度、位置偏移或模板重复|比较插入前后的固定 logits|
|prompt 没有学习|参数未进 optimizer、loss mask 为空或梯度被截断|打印参数名、梯度范数和有效 token|
|训练有效、推理无效|推理漏掉 prompt 或 position 不同|记录最终输入 shape 和位置编号|
|不同 tokenizer 无法复用 prompt|embedding、词表或模板变化|校验 base hash 和 tokenizer|
|prompt 长度增大后速度下降|attention 长度和 KV cache|分别测 prefill、decode 和 cache bytes|
|prompt 进入 loss 分母|label shift 或 mask 拼接错误|打印每个位置的 label/mask|
|多个任务相互影响|prompt ID、batch 或 cache 混用|按请求记录 prompt 与 cache 状态|
|深层 prompt 参数量不符|层数、插入层和 factor shape 漏计|枚举每层 trainable tensor|
|prefix tuning 被误称为 input prompt|实际注入了各层 K/V|检查 forward hook 和 cache shape|
|prompt 高分但 base 能力回退未查|只测 adapted task|加入 base 和原能力回归矩阵|
|checkpoint 可加载但结果偏移|base、template 或 position config 不同|拒绝 hash 或合同不一致的加载|

### 最小审计表

|审计层|应保存|验收问题|
|---|---|---|
|base|内容 hash、tokenizer、embedding、position config|prompt 是否绑定正确基础函数|
|prompt|shape、length、dtype、初始化和参数 hash|保存的 tensor 能否复现|
|position|prompt/真实 token 的 position ID|训练和推理偏移是否一致|
|mask|attention mask、loss mask、label shift|prompt 是否只作为条件|
|optimizer|group、LR、state、梯度范数|只有 prompt 进入更新吗|
|cache|prompt KV、cache prefix、batch 和 prompt ID|切换任务时是否失效|
|quality|base、hard prompt、soft prompt、不同长度|提升来自哪里|
|deployment|模板、stop、decode、prompt 路由|运行时能否复现训练输入|

Prompt Tuning 的 checkpoint 很小，因为它只保存虚拟 token 向量；运行成本没有消失，因为 prompt 仍然占用位置、attention 和 KV cache。结论必须同时给出 prompt length、hidden size、position/mask、基础模型标识和运行时长度。

## 相关词条

- [LoRA](../finetuning/lora/)：比较权重低秩增量与输入级连续 prompt。
- [QLoRA](../finetuning/qlora/)：说明量化基础权重与 adapter-only 训练状态。
- [监督微调](../finetuning/sft/)：固定 target shift、loss mask 和有效 token。
- [指令数据](../finetuning/instruction-data/)：审计 prompt 训练所用的任务、约束和答案数据。
- [Tokenization](../text-representation/tokenization/)：说明硬提示如何经过 tokenizer 变成离散 token。
- [Embedding](../text-representation/embeddings/)：说明 token ID 如何映射为 $D$ 维向量。
- [KV Cache](../inference/kv-cache/)：核对 prompt token 在生成阶段的缓存和位置。
- [推理](../inference/inference/)：比较训练 prompt、部署 prompt 和多任务路由。
- [灾难性遗忘](../finetuning/catastrophic-forgetting/)：保留基础模型和原任务回归。
