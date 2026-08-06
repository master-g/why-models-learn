---
title: "推理数学：把 prefill、decode 和 KV cache 写成资源账"
tags: ["why-models-learn"]
---

推理数学把一次生成拆成可核对的资源项：prefill 处理完整 prompt，attention 交互随序列长度平方增长；decode 每步只处理一个新 query，但需要读取长度为 $P$ 的历史 K/V；FFN、投影和 LM head 提供另外的矩阵乘法成本；KV cache 按层、请求、历史长度和 K/V head 数占用内存。只有先固定 MAC/FLOP 口径、GQA 形状、数据类型、batch 和长度，TTFT、token latency、显存与吞吐的比较才有意义。

![推理数学示意图：prefill 处理完整序列，decode 处理单个新 token 并读取历史 KV cache，底部汇总主要资源项](/assets/inference/svg/inference-math.1.svg)

## 先固定形状和计数口径

### 模型形状

设模型有 $L$ 层，主宽度为 $D$，attention query head 数为 $h$，每个 head 的维度为：

$$
d_h=\frac{D}{h}.
$$

GQA 或 MQA 使用 $h_{kv}$ 个 K/V head，K/V 投影宽度为：

$$
D_{kv}=h_{kv}d_h.
$$

MHA 是 $h_{kv}=h$ 的特例；MQA 是 $h_{kv}=1$。设 batch 中有 $B$ 条序列，prefill 的有效 token 长度为 $T$，decode 时每条请求已经拥有 $P$ 个可读历史 token。

设 FFN 中间宽度为 $D_{\mathrm{ff}}$，词表大小为 $V$。这些符号必须与实际 tokenizer、padding、packing 和 batch 定义对应，不能把字符数、token 数和最大 context length 混用。

### MAC 与 FLOP

本文把一次乘法和一次加法组成的 multiply-accumulate 记为一个 MAC：

$$
\operatorname{FLOPs}=2\operatorname{MACs}.
$$

有些硬件文档把一个 fused multiply-add 记为一个操作，有些文档把乘法和加法分开计数。报告必须说明采用 MAC 还是 FLOP，并在换算时保持同一口径。

下面的复杂度只统计主要矩阵乘法和 attention 交互：

- 不计 bias 加法、LayerNorm/RMSNorm、激活函数和 softmax 的低阶项；
- 不把 kernel launch、通信、padding 浪费和 cache miss 自动算入理论 MAC；
- 不把权重读取、K/V 读取和 workspace 内存当成 FLOP；
- 需要比较延迟时，再单独计算内存和调度下界。

## Prefill 的计算量

### GQA 投影和输出投影

在 GQA 中，Q、K、V 和 attention output 投影的 MAC 近似为：

$$
C_{\mathrm{proj}}^{\mathrm{pre}}
=
BT
\left(
D^2+DD_{kv}+DD_{kv}+D^2
\right)
=
BT(2D^2+2DD_{kv}).
$$

其中两项 $D^2$ 分别对应 Q 投影和输出投影，两项 $DD_{kv}$ 对应 K、V 投影。当 $D_{kv}=D$ 时：

$$
C_{\mathrm{proj}}^{\mathrm{pre}}
=
4BTD^2.
$$

GQA 降低 K/V 投影和 K/V cache 的宽度，但不降低 Q 投影和 output projection 的宽度。只把完整的 $4BTD^2$ 公式用于 GQA，会高估投影成本。

### Prefill attention 交互

Prefill 中每个 query 要和同一序列的 $T$ 个 key 做交互。QKᵀ 和 attention value 两次矩阵乘法分别为：

$$
C_{QK}^{\mathrm{pre}}
=
BhT^2d_h
=
BT^2D,
$$

$$
C_{AV}^{\mathrm{pre}}
=
BhT^2d_h
=
BT^2D.
$$

因此 attention 交互合计为：

$$
C_{\mathrm{attn}}^{\mathrm{pre}}
=
2BT^2D.
$$

这就是 prefill 的长度平方项。FlashAttention 可以降低中间 attention matrix 的读写和峰值内存，但不改变需要计算的主要 QKᵀ 与 AV 交互数量。

### Prefill FFN

标准两矩阵 FFN 的 token-wise MAC 为：

$$
C_{\mathrm{ffn}}^{\mathrm{pre}}
=
2BTDD_{\mathrm{ff}}.
$$

SwiGLU 等 gated FFN 通常有三条投影矩阵，因此主投影项近似为：

$$
C_{\mathrm{gated\ ffn}}^{\mathrm{pre}}
=
3BTDD_{\mathrm{ff}}.
$$

实际实现可能使用不同的中间宽度或把 gate/up 矩阵融合，但资源账本仍要列出矩阵数量和每个 shape。

### Prefill LM head

如果把最后 hidden 映射到词表 logits，LM head 的 MAC 近似为：

$$
C_{\mathrm{lm}}^{\mathrm{pre}}
=
BTDV.
$$

训练或需要每个位置 logits 的评估路径可能真的计算全部 $T$ 个位置。纯生成服务有时只需要最后一个位置的 logits，或在 prompt 处理阶段使用专门的输出策略；报告要说明是否计算完整 LM head。Tied embedding 共享参数，不会让这次矩阵乘法自动消失。

### Prefill 单层和全模型

在 GQA、标准 FFN、完整 LM head 的近似下，单层 block MAC 为：

$$
C_{\mathrm{block}}^{\mathrm{pre}}
=
BT(2D^2+2DD_{kv})
+
2BT^2D
+
2BTDD_{\mathrm{ff}}.
$$

全模型的 Transformer block MAC 为：

$$
C_{\mathrm{model}}^{\mathrm{pre}}
=
L C_{\mathrm{block}}^{\mathrm{pre}}.
$$

LM head 是否乘以 $L$ 取决于它是独立输出层，不能把它重复计入每一层。

## Decode 的单 token 计算量

### Decode 投影不再乘以完整长度

KV cache 已保存历史 K/V，decode 每次只为一个新 token 计算 Q、K、V 和 output projection：

$$
C_{\mathrm{proj}}^{\mathrm{dec}}
=
B(2D^2+2DD_{kv}).
$$

它与当前历史长度 $P$ 无关。历史长度的影响进入 attention 读取和 KV cache 内存。

### Decode attention 读取历史

新 query 对每个历史 key 做 QKᵀ，并对对应 value 做 AV：

$$
C_{QK}^{\mathrm{dec}}
=
BhPd_h
=
BPD,
$$

$$
C_{AV}^{\mathrm{dec}}
=
BhPd_h
=
BPD.
$$

因此单个 decode token 的 attention 交互为：

$$
C_{\mathrm{attn}}^{\mathrm{dec}}
=
2BPD.
$$

这是一项随历史长度线性增长的计算。KV cache 把原本每一步重新计算历史 K/V 的成本移除，但不能消除新 query 读取历史的 QKᵀ 和 AV 交互。

### Decode FFN 和 LM head

标准 FFN 的单 token MAC 为：

$$
C_{\mathrm{ffn}}^{\mathrm{dec}}
=
2BDD_{\mathrm{ff}}.
$$

完整词表 LM head 的单 token MAC 为：

$$
C_{\mathrm{lm}}^{\mathrm{dec}}
=
BDV.
$$

单层 block 的 decode MAC 为：

$$
C_{\mathrm{block}}^{\mathrm{dec}}(P)
=
B(2D^2+2DD_{kv})
+
2BPD
+
2BDD_{\mathrm{ff}}.
$$

全模型单 token MAC 为：

$$
C_{\mathrm{model}}^{\mathrm{dec}}(P)
=
L C_{\mathrm{block}}^{\mathrm{dec}}(P).
$$

如果使用 grouped-query attention，attention 交互中的 $D$ 仍由 query 总宽度决定；GQA 的主要节省来自 K/V 投影宽度、cache 宽度和内存访问，而不是把 QKᵀ 的 query 维度替换成 $D_{kv}$。

## 一个 GQA-8 数值账本

取：

$$
B=2,\quad
L=32,\quad
T=P=4096,\quad
D=4096,\quad
h=32,\quad
h_{kv}=8,\quad
D_{\mathrm{ff}}=11008,\quad
V=32000.
$$

因此：

$$
d_h=128,
\qquad
D_{kv}=1024.
$$

### Prefill 账本

|项目|单层或单次 MAC|解释|
|---|---:|---|
|Q/K/V/O 投影|343,597,383,680|$BT(2D^2+2DD_{kv})$|
|QKᵀ 与 AV|274,877,906,944|$2BT^2D$|
|标准 FFN|738,734,374,912|$2BTDD_{\mathrm{ff}}$|
|Transformer 单层合计|1,357,209,665,536|前三项相加|
|32 层 Transformer|43,430,709,297,152|单层乘以 $L$|
|prefill LM head|1,073,741,824,000|$BTDV$，若计算全部位置|

Attention 交互占单层 block 的比例不是固定常数。它随 $T$ 增长，而投影和 FFN 只随 $T$ 线性增长。短 prompt 可能由 FFN 和 projection 主导，长 prompt 才会显著暴露 $T^2$ attention 项。

### Decode 账本

设每条请求已经有 $P=4096$ 个可读历史 token：

|项目|每个 decode token MAC|解释|
|---|---:|---|
|Q/K/V/O 投影|83,886,080|$B(2D^2+2DD_{kv})$|
|QKᵀ 与 AV|67,108,864|$2BPD$|
|标准 FFN|180,355,072|$2BDD_{\mathrm{ff}}$|
|Transformer 单层合计|331,350,016|前三项相加|
|32 层 Transformer|10,603,200,512|单层乘以 $L$|
|decode LM head|262,144,000|$BDV$，若计算完整词表|

当 $P$ 继续增长时，decode attention 项线性增加；FFN、投影和完整 LM head 的单 token MAC 不随 $P$ 变化。若服务只返回一个 batch 中部分请求的 next token，还要把 active request 数与有效 batch 重新代入公式。

## KV cache 的内存账

### 每层 K/V 的 shape

每条序列、每一层保存：

$$
K,V\in\mathbb R^{P\times h_{kv}\times d_h}.
$$

如果每个元素占 $s$ bytes，全部层和 batch 的 KV cache 字节数为：

$$
M_{\mathrm{KV}}
=
2LBPh_{kv}d_hs
=
2LBPD_{kv}s.
$$

前面的 2 表示 K 和 V 两份缓存。它不表示 FP16 的两个 bytes；FP16 的元素字节数由 $s=2$ 单独提供。

若有 beam width $W$，并且每条 beam 独立保存 target history：

$$
M_{\mathrm{KV}}^{\mathrm{beam}}
=
2LBWPh_{kv}d_hs.
$$

Paged KV cache 可以减少碎片和空占，但逻辑 token 数、K/V head 数、元素字节数和所有权仍决定有效数据量。

### 数值例子的 cache 大小

对上面的 GQA-8、FP16、$B=2$、$P=4096$：

$$
M_{\mathrm{KV}}
=
2\times32\times2\times4096\times8\times128\times2
=
1,073,741,824\ \text{bytes}.
$$

这等于：

$$
1024\ \text{MiB}
$$

或每条请求约 $512$ MiB。若错误地把 $h_{kv}=32$ 当成 MHA head 数，账面 cache 会变成 4 倍，即每条请求约 $2$ GiB。

### 影响 cache 的四个轴

|轴|增加一个单位的影响|验证问题|
|---|---|---|
|层数 $L$|cache 线性增加|是否包含所有 decoder layer|
|batch $B$|cache 线性增加|active request 是否含 padding|
|历史长度 $P$|cache 线性增加|logical length 还是 allocated length|
|K/V head $h_{kv}$|cache 线性增加|MHA、GQA、MQA 的实际 head 数|
|元素字节数 $s$|cache 线性增加|FP16、BF16、FP8 或量化格式|

KV cache 不属于 trainable parameter。把模型权重显存和 KV cache 显存相加时，要单独标记静态、动态、临时和共享内存。

## 权重、激活和 workspace

### 权重内存

若模型有 $N_{\mathrm{param}}$ 个参数，每个权重元素平均占 $s_w$ bytes，权重主存储近似为：

$$
M_{\mathrm{weights}}
=
N_{\mathrm{param}}s_w
+
M_{\mathrm{scale}}
+
M_{\mathrm{metadata}}.
$$

量化后主权重字节数下降，但 scale、zero point、group index 和反量化 workspace 仍需要计入。Embedding 和 tied LM head 是否共享参数会改变 $N_{\mathrm{param}}$，但不会自动消除运行时输出投影。

### Activation 与 workspace

Prefill 需要同时处理 $B\times T$ 个位置，activation 和 workspace 峰值通常随 $BTD$ 或 attention tile 规模增加。Decode 只有新 token 的 activation，但 attention kernel 仍需读取历史 K/V。一个准确账本至少分开：

- hidden、Q、K、V 和 intermediate activation；
- attention score 或 tile workspace；
- logits 和 sampler buffer；
- NCCL 或 tensor parallel 通信 buffer；
- KV cache；
- allocator 保留但未使用的物理 block。

不要用参数量乘一个 dtype 字节数来估计完整推理显存。该公式只覆盖 weights，不覆盖动态状态。

## 从 MAC 到延迟的下界

### Roofline 形式

设一次操作需要 $F$ FLOPs，硬件峰值算力为 $R_{\mathrm{compute}}$ FLOPs/s，必须从显存层级移动 $M$ bytes，带宽为 $R_{\mathrm{memory}}$ bytes/s。忽略并行效率和调度开销时：

$$
t
\ge
\max
\left(
\frac{F}{R_{\mathrm{compute}}},
\frac{M}{R_{\mathrm{memory}}}
\right).
$$

算术强度为：

$$
I=\frac{F}{M}.
$$

当 $I$ 较低时，增加理论 FLOP 峰值未必降低延迟；当 $I$ 较高时，计算吞吐更可能成为瓶颈。Decode 的单 token attention 需要读取历史 K/V，常见工作点比 prefill 更偏向 memory-bound，但实际结论仍取决于 kernel、cache reuse、batch 和量化格式。

### 理论下界不是实测延迟

实测时间还包括：

|额外项|来源|
|---|---|
|queue time|请求等待 batch 或 GPU slot|
|kernel launch|多个小 decode kernel 的启动|
|communication|tensor parallel、pipeline parallel 和 KV 交换|
|padding|batch 中无效 token 的计算和读取|
|allocator|paged block 申请、回收和重排|
|sampling|top-k/top-p、RNG、tokenizer 和 stop matcher|
|host-device sync|日志、流式返回和控制流同步|

报告应把理论 MAC、带宽下界和实际 wall-clock 分成不同列。一次 benchmark 只给 token/s，不足以解释瓶颈。

## TTFT、ITL 和端到端长度

### Prefill 主要决定首 token

端到端首 token 延迟可以粗略拆为：

$$
\mathrm{TTFT}
=
t_{\mathrm{queue}}
+
t_{\mathrm{prefill}}
+
t_{\mathrm{first\ decode}}
+
t_{\mathrm{sampling}}.
$$

Prefill 的 $T^2$ attention、长 prompt 的 token 数、prefix cache 命中率和 batch 排队会显著影响 TTFT。Prefix cache 命中可以减少需要重新处理的 prompt，但 cache 的 position、mask、tokenizer 和模型版本必须一致。

### Decode 主要决定后续 token 间隔

相邻输出 token 的间隔可写成：

$$
\mathrm{ITL}
\approx
t_{\mathrm{decode}}(P,B)
+
t_{\mathrm{schedule}}
+
t_{\mathrm{sampling}}
+
t_{\mathrm{stream}}.
$$

其中 $P$ 随生成历史增加。若 batch 中请求长度不同，$B$ 和每条请求的 $P_b$ 需要逐步记录。总生成时间近似为：

$$
t_{\mathrm{e2e}}
=
\mathrm{TTFT}
+
\sum_{j=1}^{N_{\mathrm{new}}}\mathrm{ITL}_j.
$$

公式中的时间是事件分解，不是把所有项独立相加的精确 kernel 模型。要报告具体数值，必须说明测量点和是否包含网络、tokenizer 与流式输出。

## Batch、padding 和 chunked prefill

### 变长 batch 的有效长度

如果 batch 中第 $b$ 条请求的 prompt 长度为 $T_b$，逐条计算理论 prefill attention 交互：

$$
C_{\mathrm{attn}}^{\mathrm{pre}}
=
2D\sum_{b=1}^{B}T_b^2.
$$

把所有请求 padding 到 $T_{\max}$ 后，dense kernel 可能实际执行：

$$
C_{\mathrm{padded}}
=
2BDT_{\max}^2.
$$

两者之差是 padding 产生的理论浪费。Packed sequence、varlen attention 和 block-diagonal mask 可以减少无效交互，但实现需要保持每条序列的 cu-seqlens、position 和 cache offset 一致。

### Continuous batching 的 decode 账

Decode 阶段每条 active request 有自己的历史长度 $P_b$，attention 交互近似为：

$$
C_{\mathrm{attn}}^{\mathrm{dec}}
=
2D\sum_{b=1}^{B_{\mathrm{active}}}P_b.
$$

增加一个长上下文请求会提高整个 decode batch 的 K/V 读取量。请求完成后释放其 slot 可以降低后续成本，但 allocator 和 block table 更新也会产生调度开销。

### Chunked prefill 交换峰值与调度

把长 prompt 切成大小为 $C$ 的 chunk，可以把单次 activation 峰值从 $T$ 降低到 $C$，但需要多次 kernel 和 cache 写入。Chunk 的 attention 位置仍要看到已经处理的历史，计算量不能简单当作多个独立短 prompt 相加。

一个 chunked prefill 实现要记录：

- chunk token 范围；
- 进入 attention 的逻辑历史长度；
- position ID 和 causal mask；
- 每个 chunk 写入的 KV cache 区间；
- 被 decode 请求插入时的 batch slot；
- chunk 间的 queue 和 synchronization 时间。

## 数值精度和量化

### MAC 不随 dtype 改变，实际成本会变

理论 MAC 由 shape 决定，FP16、BF16、FP8 或 INT8 不改变矩阵乘法的数学项数。但 dtype 会改变：

- 权重和 activation 的 bytes；
- KV cache 字节数；
- 可用带宽和 cache residency；
- dequantize、scale 和 requantize 成本；
- 累计误差与 logits 复现；
- kernel 的实际吞吐。

因此“量化后 FLOPs 不变”不等于“延迟不变”。需要单独测量权重读取、KV 读取、反量化和输出质量。

### Cache 量化要标记存储与计算格式

如果 K/V 以低比特存储、以高精度参与 attention，应分别记录：

$$
M_{\mathrm{KV,stored}}
=
2LBPh_{kv}d_hs_{\mathrm{stored}},
$$

以及 kernel 内部的 compute dtype。scale 可以按 token、head、channel、group 或 block 保存；scale 的数量和共享方式影响额外内存。

### 低精度累计会改变边界

softmax、attention reduction、logit sampling 和 cache dequantization 都有归约顺序。接近 top-k 边界、EOS 概率或 stop 阈值的 token 可能因 dtype、归约和硬件 kernel 改变。数值核验需要固定输入、dtype、归约路径和容差，不要把理论公式当作 bitwise 证据。

## 运行方法

下面的标准库探针按 GQA-8 计算 prefill、decode、LM head 和 KV cache 的账本。它只计算整数 shape 公式，不执行神经网络。

```python
B, L, T, D, h = 2, 32, 4096, 4096, 32
D_ff, h_kv, bytes_per_element, V = 11008, 8, 2, 32000

d_h = D // h
D_kv = h_kv * d_h

prefill_proj = B * T * (2 * D * D + 2 * D * D_kv)
prefill_attn = 2 * B * T * T * D
prefill_ffn = 2 * B * T * D * D_ff
prefill_layer = prefill_proj + prefill_attn + prefill_ffn

decode_proj = B * (2 * D * D + 2 * D * D_kv)
decode_attn = 2 * B * T * D
decode_ffn = 2 * B * D * D_ff
decode_layer = decode_proj + decode_attn + decode_ffn

kv_bytes = 2 * L * B * T * D_kv * bytes_per_element

print("head_dim=", d_h)
print("kv_width=", D_kv)
print("prefill_proj=", prefill_proj)
print("prefill_attn=", prefill_attn)
print("prefill_ffn=", prefill_ffn)
print("prefill_layer_total=", prefill_layer)
print("prefill_model=", L * prefill_layer)
print("prefill_lm=", B * T * D * V)
print("decode_proj=", decode_proj)
print("decode_attn=", decode_attn)
print("decode_ffn=", decode_ffn)
print("decode_layer_total=", decode_layer)
print("decode_model=", L * decode_layer)
print("decode_lm=", B * D * V)
print("kv_bytes=", kv_bytes)
```

运行输出为：

```text
head_dim= 128
kv_width= 1024
prefill_proj= 343597383680
prefill_attn= 274877906944
prefill_ffn= 738734374912
prefill_layer_total= 1357209665536
prefill_model= 43430709297152
prefill_lm= 1073741824000
decode_proj= 83886080
decode_attn= 67108864
decode_ffn= 180355072
decode_layer_total= 331350016
decode_model= 10603200512
decode_lm= 262144000
kv_bytes= 1073741824
```

## 失效模式和审计方法

### 把 prefill 和 decode 用同一公式

Prefill attention 有 $T^2$ 交互；decode attention 读取历史 $P$，每步是 $P$ 的线性项。把 decode 误写成完整序列平方，或把 prefill 误写成常数，会让吞吐和延迟预测失真。审计要分别报告阶段、有效长度和 active batch。

### 把 MHA head 数套到 GQA cache

KV cache 使用 $h_{kv}$，不是 query head 数 $h$。GQA-8、MHA 和 MQA 的 cache 账本分别不同。检查模型 config、K/V tensor shape 和实际 kernel，而不是只看模型名称。

### 把 MAC 和 FLOP 混用

同一张表中同时出现 MAC、FMA、FLOP 而没有换算，会产生 2 倍差异。报告开头固定计数口径，并在硬件峰值和理论算量之间使用同一单位。

### 忽略 LM head

词表很大时，$BDV$ 或 $BTDV$ 可能成为显著项。只统计 Transformer block 会低估输出 logits 的计算和带宽。报告要说明是否只计算最后位置。

### 把理论 MAC 当作实际延迟

Kernel 融合、带宽、通信、padding、cache hit、调度和流式返回都会改变 wall-clock。理论账本用于解释趋势，实测 trace 用于确认瓶颈。

### 用 allocated length 代替 logical length

Paged cache 可能为请求预留比当前有效历史更大的物理 block。decode attention 使用 logical $P$，内存账本还要报告 allocated bytes 和碎片。两个数不能混为上下文长度。

### 漏掉 batch 与 beam 因子

KV cache 和大多数矩阵乘法都随 active batch 增加。Beam search 可能把序列数乘以 beam width；如果 prefix sharing 或 copy-on-write 改变物理占用，需要分别报告逻辑和物理账。

### 忽略 padding 和 chunk 调度

变长 batch 的 $T_{\max}$ 公式可能包含大量无效 token。Chunked prefill 降低峰值 activation，却增加调度和同步。需要保存每个请求的有效长度和 chunk 事件。

### 量化只减权重 bytes

KV cache、scale、metadata、dequantization workspace 和计算 dtype 仍占资源。量化前后同时记录 weight bytes、KV bytes、workspace、MAC、实测带宽和输出误差。

### 一份最小推理数学审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|形状|$L,B,T,P,D,h,h_{kv},d_h,D_{\mathrm{ff}},V$ 固定|model config、tokenizer、active batch|
|计数|MAC 与 FLOP 的换算固定|FMA 口径、硬件文档|
|prefill|投影、$T^2$ attention、FFN、LM head 分开|padding、packed sequence|
|decode|投影、$P$ attention、FFN、LM head 分开|KV cache、logical length|
|GQA|K/V 使用 $D_{kv}=h_{kv}d_h$|MHA/GQA/MQA config|
|cache|K、V、层、batch、长度、dtype 都计入|page、block、allocated length|
|memory|weights、KV、activation、workspace 分开|量化 scale、allocator|
|roofline|FLOPs/算力与 bytes/带宽分别计算|峰值定义、实测利用率|
|latency|TTFT、ITL、queue、sampling、streaming 分开|测量点、warmup|
|batch|有效长度、padding、active slot 记录|continuous batching|
|数值|dtype、归约、softmax、量化误差固定|边界 logits、NaN|
|复现|保存 shape、配置、版本和 trace|只保存 token/s|

推理数学的作用是把“模型很大所以很慢”拆成可检查的资源关系。Prefill 主要暴露序列长度和 $T^2$ attention；decode 主要暴露历史 K/V 读取、active batch 和 cache 字节数；GQA 通过减少 K/V 宽度降低内存；LM head、量化、padding、通信和调度决定理论账本如何映射到实际延迟。

## 相关词条

- [推理](../inference/inference/)：固定 prefill/decode、batch、停止条件和端到端协议。
- [KV cache](../inference/kv-cache/)：展开 K/V shape、追加、分页布局和 cache 字节数。
- [参数量](../transformer-components/parameter-count/)：区分权重参数、激活、optimizer state 和运行时 cache。
- [GQA 与 MQA](../attention/gqa-and-mqa/)：解释 query head 与 K/V head 的形状和内存差异。
- [注意力复杂度](../attention/attention-complexity/)：继续拆分 attention 交互、内存和稀疏路径。
- [FlashAttention](../attention/flash-attention/)：说明 tile 化如何降低中间矩阵读写和峰值内存。
- [量化](../inference/quantization/)：说明低比特权重、K/V、scale 和反量化成本。
