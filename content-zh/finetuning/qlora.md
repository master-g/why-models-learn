---
title: "QLoRA：量化基础权重后训练低秩适配器"
tags: ["why-models-learn"]
---

QLoRA 把 4-bit 量化的基础模型与 LoRA adapter 组合起来：基础权重以量化索引和量化常数保存，前向时按计算 dtype 反量化，训练时只更新低秩因子。它减少基础权重和训练状态的存储，却没有消除激活、workspace、通信和 adapter optimizer state。理解 QLoRA 需要分开记录量化存储、反量化计算、adapter 梯度、二次量化和分页 optimizer；“4-bit”只描述其中一个存储字段。

![QLoRA 示意图：冻结的基础权重先保存为 4-bit 索引和量化常数，前向时反量化到计算 dtype，另一条路径用 LoRA 因子产生增量，二者相加后输出](/assets/finetuning/svg/qlora.1.svg)

## 把 QLoRA 写成两条前向路径

### 量化基础权重不等于 4-bit 矩阵计算

设基础线性层为 $W_0\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$。量化器保存索引 $q$ 和量化常数 $s$，得到近似权重：

$$
\widehat W_0=\operatorname{dequant}(q,s).
$$

如果采用按组对称量化，一个组内的数值可以写成：

$$
q_i
=\operatorname{clip}\left(
\operatorname{round}\frac{w_i}{s_g},
q_{\min},q_{\max}
\right),
\qquad
\widehat w_i=s_g q_i.
$$

这里 $g$ 表示 $i$ 所属的 group。$q_i$ 只占 4 bit，但 $s_g$ 需要额外存储。量化误差为：

$$
e_i=\widehat w_i-w_i.
$$

因此，4-bit storage 仍然对应一个反量化后的近似浮点矩阵，不能把它直接理解成每个 kernel 都用 4-bit 累加。

### LoRA 只改变另一条路径

QLoRA 的训练前向可以写成：

$$
y
=\widehat W_0x
+s_{\mathrm{lora}}BAx,
\qquad
s_{\mathrm{lora}}=\frac{\alpha}{r}.
$$

基础权重 $\widehat W_0$ 被冻结，$A$ 和 $B$ 是可训练参数。反向传播只把有效梯度送入 adapter 的 optimizer；基础量化索引和量化常数参与前向，但不进入更新集合。

|组成|保存形态|计算形态|是否训练|
|---|---|---|---|
|基础权重 $W_0$|4-bit index、group scale|BF16/FP16 或实现指定 dtype|否|
|量化常数|FP32 或二次量化后的低 bit 值|反量化时读取|否|
|LoRA 因子 $A,B$|通常 FP16/BF16|低秩矩阵乘法|是|
|LoRA optimizer state|FP32 moments 或分页 state|更新 A、B|是|
|激活与 workspace|由 batch、长度和 kernel 决定|计算中间结果|否，但占峰值显存|

### 4-bit 只回答了一个问题

报告 QLoRA 实验时，至少把以下字段拆开：

|字段|示例|没有该字段时无法确认|
|---|---|---|
|quant type|NF4|4-bit codebook 和误差分布|
|group size|64|scale 数量和存储账|
|scale dtype|FP32 或二次量化|量化常数的额外字节|
|compute dtype|BF16|矩阵乘法和累加的数值路径|
|double quant|开启、nested group 256|scale 是否再次量化|
|LoRA rank/alpha|8/16|adapter 更新空间和实际 scale|
|target module|Q/K/V/O 或指定集合|哪些层可以变化|
|optimizer|paged AdamW 等|state 分配和峰值行为|
|sequence length|固定值或分布|activation 和临时 buffer|

只写“4-bit QLoRA”不能复现显存或结果。

## NF4 保存的不是均匀整数网格

### 均匀 INT4 和 NF4 的区别

均匀 INT4 把区间等分为有限的整数格点。NF4（NormalFloat4）把 4-bit index 映射到一组针对近似正态权重分布设计的非均匀 codepoint。它的索引数量仍然是：

$$
N_{\mathrm{code}}=2^4=16.
$$

对一个 group，量化过程可以抽象成：

$$
q_i
=\arg\min_{k\in\{0,\ldots,15\}}
\left\lvert \frac{w_i}{s_g}-c_k\right\rvert,
\qquad
\widehat w_i=s_g c_{q_i},
$$

其中 $c_k$ 是 NF4 codebook。codepoint 的具体实现、零点处理和 scale 定义由量化库决定，不能用“4-bit”反推出全部细节。

|量化类型|索引|代表值|更适合检查|
|---|---|---|---|
|均匀 INT4|整数格点|等间距|范围、饱和和端点误差|
|对称 INT4|带符号整数|零点固定在中心|正负分布和 absmax scale|
|非对称 INT4|整数加 zero point|区间按数据平移|偏置分布和零点|
|NF4|16 个非均匀 codepoint|按近似正态分布设计|权重主体区域的量化误差|

### group size 改变误差和 metadata

每组共享一个 scale。group 越大，scale 的 metadata 越少，但组内动态范围更难由一个常数覆盖；group 越小，近似通常更细，scale 数量和读取开销增加。

|group size|每个参数的 scale 负担|局部适应性|资源影响|
|---:|---:|---|---|
|32|较高|较强|scale 数量多|
|64|中等|中等|常见折中|
|128|较低|较弱|scale 数量少|
|256|更低|更弱|异常值更容易放大量化误差|

对同一个模型比较 NF4、group size 或 scale dtype 时，应固定 checkpoint、LoRA 配置、数据和计算 dtype。否则质量差异不能归因于量化方案。

### 量化误差会进入 logits

基础路径实际计算的是：

$$
\widehat W_0x
=W_0x+ex.
$$

即使 adapter 初始增量为零，量化模型的 logits 也可能已经不同于原始 BF16 模型。应保存 base BF16、quantized base 和 QLoRA checkpoint 的同一输入回归结果，测量 logits 最大绝对差、top-k 重合率和任务分数。

## 二次量化压缩量化常数

### scale 也可以按组量化

普通 group quantization 需要为每个参数组保存一个 scale。二次量化把这些 scale 组成第二个向量，再按更大的 nested group 量化：

$$
\widehat s_g
=\operatorname{dequant}(u,t),
\qquad
\widehat w_i
=\widehat s_g c_{q_i}.
$$

第一层的参数 index 使用 4 bit，第一层 scale 仍可以用 8 bit 或其他低 bit 表示，第二层的 metadata 需要单独保存。二次量化减少的是 scale payload，不会减少主权重 index 已经占用的 4 bit。

### 量化层级必须写进存储账

令模型参数量为 $P$，第一层 group size 为 $G$，scale 使用 $b_s$ bytes，二次量化的 scale payload 使用 $b_{s2}$ bytes，nested group size 为 $G_2$。忽略 header 与 padding，单次量化基础权重近似为：

$$
M_{\mathrm{single}}
\approx
\frac{Pb_q}{8}
+\left\lceil\frac{P}{G}\right\rceil b_s.
$$

二次量化基础权重近似为：

$$
M_{\mathrm{double}}
\approx
\frac{Pb_q}{8}
+\left\lceil\frac{P}{G}\right\rceil b_{s2}
+\left\lceil
\frac{\lceil P/G\rceil}{G_2}
\right\rceil b_s.
$$

这两个式子没有包含 kernel 对齐、tensor padding、量化 header、临时反量化 buffer 和 allocator 碎片。它们用于检查数量级，不能代替 runtime peak。

|账本项|单次量化|二次量化|说明|
|---|---:|---:|---|
|主权重 index|$Pb_q/8$|$Pb_q/8$|4-bit 时两者相同|
|第一层 scale|每组 $b_s$ bytes|每组 $b_{s2}$ bytes|二次量化只改变这一项|
|第二层 metadata|无|每 nested group $b_s$ bytes|仍然需要保存|
|基础权重总量|主权重加 scale|主权重加两层 scale|不含 runtime buffer|
|adapter state|单独统计|单独统计|不能从基础权重账中扣除|

### 二次量化不是误差免费

第二层量化会让 scale 本身产生误差：

$$
\widehat w_i-w_i
=
\widehat s_g c_{q_i}-w_i
=
(\widehat s_g-s_g)c_{q_i}
+(s_gc_{q_i}-w_i).
$$

第一项来自 scale 的二次量化，第二项来自主权重 index 的近似。二次量化的节省应与量化误差、任务质量和反量化 kernel 一起报告。

## 计算 dtype 和梯度边界

### 保存 dtype、计算 dtype 和 optimizer dtype 分开

同一个 QLoRA 运行至少有三类 dtype：

|dtype 角色|典型选择|作用|
|---|---|---|
|基础权重存储|4-bit index、8/32-bit scale|节省 checkpoint 和基础权重显存|
|矩阵计算|BF16 或 FP16|反量化后的 GEMM 输入|
|LoRA 因子|BF16/FP16，可能有 FP32 master|计算 adapter 增量和保存更新|
|累加器|硬件或 kernel 指定|降低点积累加误差|
|optimizer state|FP32 moments|保存 A、B 的更新状态|

若只把 storage dtype 写成 4-bit，就无法判断 GEMM、累加和 optimizer 的数值误差。

### 基础权重冻结时不需要基础梯度

把前向拆成：

$$
y=y_{\mathrm{base}}+y_{\mathrm{adapter}},
\qquad
y_{\mathrm{base}}=\widehat W_0x,
\qquad
y_{\mathrm{adapter}}=s_{\mathrm{lora}}BAx.
$$

若基础权重冻结，反向只需保留对 $A$、$B$ 的梯度：

$$
\frac{\partial\mathcal L}{\partial A}
=s_{\mathrm{lora}}B^{\mathsf T}
\frac{\partial\mathcal L}{\partial y}x^{\mathsf T},
\qquad
\frac{\partial\mathcal L}{\partial B}
=s_{\mathrm{lora}}
\frac{\partial\mathcal L}{\partial y}(Ax)^{\mathsf T}.
$$

QLoRA 不是把量化索引当作可训练整数参数。基础量化路径提供前向值，optimizer 只维护 adapter 的 state。若日志显示基础量化权重进入 optimizer，应先检查实际训练方案是否已经变成全量或部分量化权重微调。

### loss mask 仍然决定 adapter 学什么

量化只改变基础路径的数值精度，不改变 SFT 的 target shift、assistant-only mask、padding mask 或有效 token 分母。QLoRA loss 下降时，仍需要报告：

|字段|为什么保留|
|---|---|
|监督 token 数|确认 loss 分母没有把 padding 算进去|
|prompt/assistant mask|区分条件 token 和目标 token|
|quantized base 回归|确认 adapter 不是在补偿错误模板|
|base 与 QLoRA 对照|确认质量提升来自训练而不是评测变化|
|有效 token 预算|比较不同 rank、group size 和量化类型|

## paged optimizer 处理峰值，而不是减少状态

### optimizer state 仍然属于 adapter

对 adapter 参数量 $P_a$，若权重、梯度和 Adam state 分别占 $b_w$、$b_g$、$b_o$ bytes：

$$
M_{\mathrm{adapter\ state}}
\approx
P_a(b_w+b_g+b_o).
$$

QLoRA 不会因为基础权重使用 4-bit 就把 adapter optimizer state 变成 4-bit。paged optimizer 可以把一部分 state 放到分页的主存，并在需要时搬运，目的是降低 GPU 峰值；它可能增加 page movement、同步和延迟。

### 长序列会把激活重新推到前台

当基础权重已经压缩后，activation、temporary buffer、attention workspace、梯度 accumulation 和数据搬运可能成为主要峰值。尤其是序列长度增加时，量化基础权重的固定节省不会抵消所有与 token 数相关的内存。

|峰值来源|是否由 4-bit 直接消除|审计方法|
|---|---|---|
|基础权重 index|部分降低|分别记录 index、scale 和 padding|
|LoRA optimizer state|否|统计每个 optimizer group 的 state|
|attention activation|否|按 sequence length 测峰值|
|临时反量化 buffer|否|记录 kernel workspace 和生命周期|
|gradient accumulation|否|比较 micro-batch 与 accumulation steps|
|paged state 搬运|否|记录 GPU/CPU page resident 和传输时间|

## 一个可运行的 QLoRA 资源账本

下面的探针使用 Python 标准库比较 7B 模型的 4-bit 主权重、单次量化 scale、二次量化 scale、2,000 万参数 adapter state 和全量 BF16 训练状态。它不模拟真实 NF4 codebook、kernel padding、activation 或 paged transfer。

```python
from math import ceil

P = 7_000_000_000
bits = 4
group_size = 64
scale_bytes = 4
nested_group_size = 256
adapter_parameters = 20_000_000
weight_bytes = 2
grad_bytes = 2
adam_state_bytes = 8

quantized_payload = P * bits // 8
groups = ceil(P / group_size)
scales_fp32 = groups * scale_bytes
nested_scale_payload = groups
nested_scale_metadata = ceil(groups / nested_group_size) * scale_bytes
single_base = quantized_payload + scales_fp32
double_base = quantized_payload + nested_scale_payload + nested_scale_metadata
adapter_train_state = adapter_parameters * (weight_bytes + grad_bytes + adam_state_bytes)
full_train_state = P * (weight_bytes + grad_bytes + adam_state_bytes)
qlora_state = double_base + adapter_train_state

gib = lambda value: value / 2**30
print('quantized_payload_gib=', f'{gib(quantized_payload):.3f}')
print('groups=', groups)
print('single_quant_scales_gib=', f'{gib(scales_fp32):.3f}')
print('single_quant_base_gib=', f'{gib(single_base):.3f}')
print('double_quant_scale_payload_gib=', f'{gib(nested_scale_payload):.3f}')
print('double_quant_scale_metadata_gib=', f'{gib(nested_scale_metadata):.3f}')
print('double_quant_base_gib=', f'{gib(double_base):.3f}')
print('adapter_train_state_gib=', f'{gib(adapter_train_state):.3f}')
print('qlora_train_state_gib=', f'{gib(qlora_state):.3f}')
print('full_train_state_gib=', f'{gib(full_train_state):.3f}')
print('full_vs_qlora_reduction=', f'{full_train_state / qlora_state:.3f}')
print('double_quant_saving_mib=', f'{(single_base - double_base) / 2**20:.3f}')
print('nf4_codepoints=', 16)
```

运行输出：

```text
quantized_payload_gib= 3.260
groups= 109375000
single_quant_scales_gib= 0.407
single_quant_base_gib= 3.667
double_quant_scale_payload_gib= 0.102
double_quant_scale_metadata_gib= 0.002
double_quant_base_gib= 3.363
adapter_train_state_gib= 0.224
qlora_train_state_gib= 3.587
full_train_state_gib= 78.231
full_vs_qlora_reduction= 21.812
double_quant_saving_mib= 311.295
nf4_codepoints= 16
```

在这个账本中，4-bit 主权重 payload 为 3.260 GiB。第一次量化的 FP32 group scale 使基础权重达到 3.667 GiB；二次量化把 scale payload 和第二层 metadata 合并后，基础权重约为 3.363 GiB。2,000 万个 adapter 参数的 BF16 权重、梯度和 Adam state 约占 0.224 GiB，合计 QLoRA train state 为 3.587 GiB。全量 BF16 权重、梯度和 Adam state 约占 78.231 GiB，纸面账本的比率为 21.812 倍。

这些数字没有把 activation、workspace、kernel padding、通信 buffer、paged state 和 allocator 碎片算进去。它们用于检查 bit、group、scale 和 GiB 换算，部署前仍需测量设备峰值。

## QLoRA 仍然继承 LoRA 的容量边界

### 量化不改变 target module

QLoRA 只是改变基础权重的存储和计算路径，LoRA 的 rank、alpha、target module、bias policy 和层范围仍决定可训练函数族。相同 4-bit 基础模型搭配不同 target module，可能得到不同的 adapter 参数量和任务能力。

|QLoRA 配置|基础权重|可训练路径|比较重点|
|---|---|---|---|
|Q/V rank-8|4-bit|查询和值投影|adapter 小、attention 路径有限|
|Q/K/V/O rank-8|4-bit|完整 attention 投影|选择、读取和输出混合同时变化|
|attention + MLP rank-8|4-bit|attention 与特征变换|容量增加、回归定位更难|
|相同 target、rank-16|4-bit|每个目标矩阵更多方向|adapter state 和通信增加|
|相同 target、不同 group size|4-bit|LoRA 路径不变|隔离量化误差和 metadata 差异|

### rank、量化误差和数据质量要分开

如果 QLoRA 分数低，原因可能来自三个层面：

1. 量化基础路径与 BF16 基座的输出已经偏移。
2. LoRA rank 或 target module 不足以表达任务更新。
3. 数据、mask、学习率、有效 token 或评测合同有问题。

应建立至少包含 BF16 base、4-bit base、不同 rank 的 QLoRA 和 full/PEFT 对照的回归矩阵。只增加 rank 不能证明问题来自容量，只更换量化类型也不能证明问题来自量化。

## 合并和部署边界

### 未合并路径最容易保持量化合同

QLoRA 的未合并前向是：

$$
y_{\mathrm{unmerged}}
=\operatorname{dequant}(q,s)x
+s_{\mathrm{lora}}BAx.
$$

如果要合并，先得到浮点近似权重：

$$
W_{\mathrm{float}}
=\operatorname{dequant}(q,s)
+s_{\mathrm{lora}}BA.
$$

随后可以保存为 BF16/FP16 merged checkpoint，也可以重新量化得到新的 $q'$、$s'$。直接把 $A$、$B$ 的浮点值写进 4-bit index 不是矩阵相加；重新量化会引入新的误差，必须重新测量 logits 和任务质量。

|部署形态|基础权重|LoRA|主要风险|
|---|---|---|---|
|QLoRA unmerged|4-bit index 加 scale|独立加载|需要额外低秩路径和 adapter 路由|
|dequantized merged|BF16/FP16|已写回|基础权重显存恢复到浮点规模|
|requantized merged|重新量化的 4-bit|已写回后重编码|二次量化误差和 scale 合同改变|
|多个 unmerged|同一量化基础|按请求切换|cache、batch 和 adapter ID 混用|

### 合并前后要比较同一输入

对一批固定输入，至少比较：

$$
\delta_{\mathrm{logit}}
=
\max_j
\left\lvert
\ell_{\mathrm{unmerged},j}
-\ell_{\mathrm{merged},j}
\right\rvert.
$$

同时记录 tokenizer、chat template、quant config、LoRA config、compute dtype 和 merge 工具版本。对 requantized merged 结果，允许的误差应由任务和评测协议给出，不能直接套用未量化合并的浮点舍入阈值。

## 运行方法

将上一个 Python 代码块保存为 qlora-ledger.py，再运行 python3 qlora-ledger.py。修改参数量、group size、scale dtype、nested group、adapter 参数量或 optimizer state 后，应同步检查正文表格。

接入实际训练时，先记录量化配置和基础模型回归，再记录每个 target module 的 LoRA 参数、梯度、optimizer state、paged resident bytes、activation peak 和通信时间。训练结束后分别保存 unmerged、dequantized merged 或 requantized merged 的输出对照。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|显存仍然不足|activation、workspace、adapter state|按组件采集 peak memory|
|4-bit 模型加载后质量突然下降|quant type、group、scale 和 kernel|和 BF16 base 比较固定 logits|
|adapter 没有学习|A/B、loss mask、target module|打印 trainable count 和梯度范数|
|二次量化没有节省|scale、metadata、padding|分别统计两层 scale 文件字节|
|不同库结果不一致|NF4、zero point、compute dtype|保存 quant config 和版本|
|合并后输出偏移|requant 或 alpha/r 重复|比较三种 merge 输出|
|长序列速度下降|paged 搬运、workspace|测量 page movement、kernel 时间|
|换 adapter 后输出混入上一个任务|KV/prefix cache、batch|按请求记录 adapter ID 和 cache 生命周期|
|训练 loss 降但任务分数不升|量化误差、rank、数据、评测|运行 base/quant/QLoRA 回归矩阵|
|报告写了 4-bit 但无法复现|完整 quant config 缺失|保存完整 quant config|
|基础量化权重进入 optimizer|冻结配置或训练方案错误|枚举 optimizer 参数名和 state|

### 最小审计表

|审计层|应保存|验收问题|
|---|---|---|
|base|原始内容 hash、tokenizer、template|量化前后是否来自同一个 checkpoint|
|quantization|type、bits、group、scale、zero point、nested config|每个量化字段能否重建|
|compute|dequant kernel、compute dtype、accumulator|4-bit storage 如何参与矩阵计算|
|adapter|rank、alpha、target、factor shape、dtype|实际更新了哪些路径|
|optimizer|group、state dtype、paging、resident bytes|state 是否只属于 adapter|
|memory|index、scale、adapter、activation、workspace、通信|峰值显存是否完整|
|forward|BF16 base、quantized base、unmerged、merged logits|误差出现在哪个边界|
|quality|任务、回归、有效 token、切分|量化和训练质量能否分开比较|
|deployment|cache、batch、adapter ID、merge 产物|运行时状态是否隔离|

QLoRA 的资源节省来自冻结基础权重的低 bit 存储和 adapter-only optimizer state；它的数值边界来自反量化、scale、compute dtype 和重新量化。报告必须把这些字段与 LoRA 的 rank、target module 和有效 token 同时保存。

## 相关词条

- [LoRA](../finetuning/lora/)：展开低秩因子、alpha/r、初始化、target module 和合并。
- [量化](../inference/quantization/)：说明 scale、zero point、group、误差和 activation/KV cache 量化边界。
- [全量微调与参数高效微调](../finetuning/full-vs-peft/)：比较完整训练状态与 adapter-only 状态。
- [混合精度](../training-nn/mixed-precision/)：区分存储、计算、累加和 optimizer dtype。
- [监督微调](../finetuning/sft/)：固定 target shift、loss mask 和有效 token 分母。
- [指令数据](../finetuning/instruction-data/)：审计训练示范的质量、覆盖、去重和切分。
- [长上下文](../inference/long-context/)：说明序列长度对 activation、attention 和缓存的影响。
- [推理](../inference/inference/)：比较量化、反量化、merged 和 unmerged 运行时路径。
- [灾难性遗忘](../finetuning/catastrophic-forgetting/)：检查 adapter 训练后的基础能力回归。
