---
title: "推理：冻结参数后如何把上下文变成输出"
tags: ["why-models-learn"]
---

推理是加载冻结的模型参数和运行协议，对一段输入上下文执行前向计算并产生输出 token 或任务结果的过程。语言模型推理包含 tokenization、prefill、逐 token decode、KV cache、停止条件、batch 调度和输出后处理；它不更新参数，也不使用训练期的反向传播。一次推理结果同时受模型 checkpoint、tokenizer、prompt、上下文长度、解码配置、精度和服务调度影响。

![推理示意图：输入文本经过 tokenizer 和 prefill，KV cache 进入逐 token decode，输出经过停止条件和后处理返回结果；下方记录延迟与显存账本](/assets/inference/svg/inference.1.svg)

## 推理协议要先冻结

一次可复核的推理请求可以写成：

$$
\mathcal I
=
\left(
M,
\operatorname{Tok},
x_{1:T},
G,
L_{\mathrm{ctx}},
D,
P,
R
\right).
$$

其中 $M$ 是冻结的模型 checkpoint，$\operatorname{Tok}$ 是 tokenizer，$x_{1:T}$ 是输入 token，$G$ 是生成配置，$L_{\mathrm{ctx}}$ 是上下文窗口，$D$ 是设备和 dtype，$P$ 是并发与 batch 调度，$R$ 是停止、解码和后处理规则。输出应写成

$$
y_{1:K}
=
\operatorname{Generate}
\left(
M,
\operatorname{Tok}(x_{1:T});
G,
L_{\mathrm{ctx}},
D,
P,
R
\right).
$$

模型名称相同而 $\mathcal I$ 不同，输出文本、延迟和显存都可能不同。推理报告至少保存：

|范围|固定字段|为什么需要|
| --- | --- | --- |
|模型|checkpoint hash、架构、参数 dtype、量化配置|确认使用了同一组参数|
|输入|原始文本 hash、tokenizer、模板、prompt token 数|确认上下文相同|
|生成|max new tokens、temperature、top-k、top-p、seed|确认输出规则相同|
|上下文|最大长度、截断方向、EOS、stop sequence|确认可见前缀和结束规则|
|运行|设备、batch、并发、prefill/decode kernel|解释速度和内存|
|输出|生成 token、停止原因、后处理文本|区分模型输出与渲染结果|

[Decoder-only Transformer](../transformer-architectures/decoder-only/)说明模型结构和 causal mask；本文关注冻结参数后的执行生命周期。后续 kv-cache、greedy-decoding、temperature-sampling 和 top-k-top-p 词条分别展开缓存和解码规则。

## 一个生成请求经历两段前向

### Prefill 处理整个输入上下文

输入 prompt token 的 shape 通常为

$$
X_{\mathrm{prompt}}
\in
\mathbb N^{B\times T_{\mathrm p}},
$$

其中 $B$ 是 batch size，$T_{\mathrm p}$ 是 prompt 长度。prefill 一次性计算这段上下文的 hidden states、logits 和每层的 key/value：

$$
\left(
H_{\mathrm p},
\{K_\ell,V_\ell\}_{\ell=1}^{L}
\right)
=
\operatorname{Forward}
\left(
M,
X_{\mathrm{prompt}}
\right).
$$

prompt 中每个位置都可以参与 causal attention；如果只需要生成最后一个 token，可以只读取最后位置的 logits。此前位置产生的 $K$ 和 $V$ 留在 cache 中，后续 decode 不需要重新计算这些投影。

prefill 的主要成本随 prompt 长度增加。对单层 self-attention，attention score 的交互项近似与 $T_{\mathrm p}^2$ 成正比；线性投影和 FFN 还会带来与 $T_{\mathrm p}$ 成正比的项。prompt 越长，首 token 延迟通常越高，但同一次 prefill 可以并行处理多个位置。

### Decode 每次只追加一个 token

模型从最后一个位置的 logits 选择或采样一个 token $y_1$。下一次 decode 输入这个新 token：

$$
y_j
\longrightarrow
\operatorname{Forward}
\left(
M,
y_j,
\{K_\ell,V_\ell\}
\right)
\longrightarrow
p(y_{j+1}\mid x_{1:T_{\mathrm p}},y_{1:j}).
$$

当前 query 的 shape 近似为

$$
Q_{\mathrm{new}}
\in
\mathbb R^{B\times H_q\times1\times d_h}.
$$

cache 中的 key/value 长度随生成增长：

$$
K_\ell,V_\ell
\in
\mathbb R^{B\times H_{\mathrm{kv}}\times(T_{\mathrm p}+j)\times d_h}.
$$

每次 decode 只为新位置计算 Q、K、V 和 FFN，但新 query 仍要读取全部历史 K/V。单步 attention 的读取量随上下文长度线性增长；生成 token 越多，单步延迟和 KV cache 越大。

### 两段之间的边界

prefill 输出的最后位置 logits 可以直接决定第一个生成 token，也可以把一个显式 BOS 或 decoder start token 作为第一步 decode 输入。实现要固定以下约定：

1. prompt 最后一个 token 是否已经写入 cache；
2. 第一个生成 token 使用 prefill logits 还是额外 decode；
3. 新 token 在采样前还是采样后追加到 cache；
4. EOS 是否写入最终输出；
5. stop sequence 是否包含在返回文本中。

这些约定会改变一次请求的 decode 次数和计时。日志中应分别记录 prompt token 数、生成 token 数、prefill forward 次数和 decode forward 次数。

## KV cache 记录历史 K/V

### cache 保存什么

对第 $\ell$ 层，KV cache 保存每个历史位置的 key 和 value：

$$
\operatorname{Cache}_\ell
=
\left(
K_\ell,V_\ell
\right).
$$

当前 query $Q_\ell$ 读取 cache 后计算：

$$
A_\ell
=
\operatorname{softmax}
\left(
\frac{Q_\ell K_\ell^\mathsf T}
{\sqrt{d_h}}
\right),
\qquad
O_\ell
=
A_\ell V_\ell.
$$

decode 时 query 长度为 1，K/V 长度是已有上下文长度。cache 把“重新计算历史 projection”变成“读取历史 K/V”，所以它降低了每一步的重复计算；它没有消除 attention 对历史 token 的读取。

### cache 的内存公式

若模型有 $L$ 层、每层 $H_{\mathrm{kv}}$ 个 K/V head、每个 head 的宽度为 $d_h$，batch 为 $B$，当前缓存长度为 $T$，每个标量占 $b$ bytes，则 K 和 V 的 cache 内存为

$$
S_{\mathrm{KV}}
=
B
L
2
H_{\mathrm{kv}}
T
d_h
b.
$$

因果 decoder 使用 MHA 时，$H_{\mathrm{kv}}=H_q$；GQA 和 MQA 可以使用更少的 K/V head，从而减少 cache。[参数量与资源账本](../transformer-components/parameter-count/)说明参数、激活和运行时状态的区别；KV cache 属于每个请求随长度增长的运行时状态。

### cache 的生命周期

|时刻|cache 状态|需要核对|
| --- | --- | --- |
|请求开始|空|请求是否复用 prefix cache|
|prefill 结束|包含 prompt 的 K/V|长度、位置 offset、batch 映射|
|每次 decode 后|追加当前 token 的 K/V|追加顺序、EOS 和 mask|
|请求结束|释放或回收到池|请求 ID 与内存归还|
|请求合并/拆分|迁移到新的 batch slot|cache 与 sequence ID 对齐|
|beam 或多样本扩展|复制或重排 cache|分支索引和显存增长|

cache 复用要求 prefix 完全相同，包括 tokenizer、模板、位置编码、attention mask 和 dtype。只比较字符串前缀而忽略 special token，会把不同的状态当成同一 cache。

## 解码从 logits 产生 token

### logits 先变成条件分布

最后位置的 logits 为

$$
z_t\in\mathbb R^V.
$$

温度为 $T_{\mathrm{temp}}>0$ 时，条件概率可以写成

$$
p_i
=
\frac{
\exp\left(z_{t,i}/T_{\mathrm{temp}}\right)
}{
\sum_{j=1}^{V}
\exp\left(z_{t,j}/T_{\mathrm{temp}}\right)
}.
$$

实际实现应使用稳定 log-softmax，先减去最大 logit。温度、top-k、top-p、repetition penalty 和随机 seed 都会改变从 logits 到 token 的映射；它们属于解码协议，不属于模型参数。

### greedy、sampling 和 beam 是不同执行规则

推理主循环可以统一写成：

$$
y_{t+1}
=
\operatorname{Select}
\left(
p_\theta(\cdot\mid x_{\leq t});
G
\right).
$$

Select 可以选择最大概率 token、按分布采样、保留多个候选序列或调用外部约束。不同规则需要不同的状态和成本：

|规则|下一 token 的选择|额外状态|
| --- | --- | --- |
|greedy|取最大 logit 或概率|单条序列|
|temperature sampling|按温度调整后的分布采样|随机数状态|
|top-k/top-p|先截断候选，再采样|候选集合和归一化|
|beam search|保留多个累计 log probability 最高的序列|beam cache、分支分数|
|majority vote|生成多个完整答案后投票|多条序列和聚合器|

本文只需要固定 Select 的接口。各策略的概率变化、候选集合和分支排序留给后续词条；推理层仍要记录它们对延迟、显存和停止条件的影响。

### 生成分数和 PPL 分数使用不同前缀

[困惑度评估](../pretraining/evaluation-perplexity/)使用 teacher forcing，把真实历史 token 作为条件，计算每个 target 的 log probability。生成推理使用模型自己刚刚选择的 token 作为后续前缀。生成过程中一个错误 token 会改变之后的 context；因此生成长度、成功率和 PPL 不能互换。

## 停止条件是输出协议的一部分

生成循环在每次选择 token 后检查停止条件：

$$
\operatorname{stop}(y_{1:t})
=
\operatorname{EOS}(y_t)
\lor
\operatorname{StopSeq}(y_{1:t})
\lor
(t\geq K_{\max})
\lor
\operatorname{Timeout}.
$$

常见停止条件包括：

|条件|检查对象|返回时要记录|
| --- | --- | --- |
|EOS|最后一个 token ID|EOS 是否包含在 token 输出中|
|stop sequence|已生成文本的尾部|匹配字符串是否从结果中删除|
|max new tokens|生成 token 计数|达到上限还是自然结束|
|context limit|prompt 加生成长度|截断、拒绝还是提前结束|
|timeout|队列、prefill、decode 总时间|超时阶段和已生成 token|
|cancellation|客户端或调度器信号|请求是否释放 cache|
|格式约束|JSON、代码或工具调用状态|解析失败和恢复策略|

如果 stop sequence 在 UTF-8、token 或规范化文本的边界上定义不同，模型可能生成同一字符串而停止时刻不同。记录原始 token 序列和后处理文本，才能区分模型生成与服务层截断。

## 延迟要按阶段测量

设请求排队时间为 $T_{\mathrm q}$，prefill 时间为 $T_{\mathrm p}$，第一个 decode 时间为 $T_{\mathrm d,1}$，后续 decode 时间为 $T_{\mathrm d,j}$。首 token 延迟为

$$
\operatorname{TTFT}
=
T_{\mathrm q}
+
T_{\mathrm p}
+
T_{\mathrm d,1}.
$$

生成 $K$ 个 token 的端到端时间为

$$
T_{\mathrm{e2e}}
=
T_{\mathrm q}
+
T_{\mathrm p}
+
\sum_{j=1}^{K}
T_{\mathrm d,j}.
$$

如果 decode 单步时间近似固定为 $T_{\mathrm d}$：

$$
T_{\mathrm{e2e}}
\approx
T_{\mathrm q}
+
T_{\mathrm p}
+
K T_{\mathrm d}.
$$

常见指标应分开报告：

|指标|公式或定义|主要受什么影响|
| --- | --- | --- |
|TTFT|队列 + prefill + 首次 decode|prompt 长度、batch、prefill kernel|
|inter-token latency|相邻输出 token 的时间间隔|KV cache、上下文长度、decode batch|
|time per output token|decode 阶段总时间除生成 token 数|并发、采样、通信和内存带宽|
|prefill throughput|prompt token 数除 prefill 时间|矩阵乘利用率、输入长度|
|decode throughput|生成 token 数除 decode 时间|cache 读取、batch 和上下文|
|端到端延迟|从请求进入到输出返回|队列、停止、后处理和网络|
|显存峰值|权重、cache、workspace 的最大占用|并发、长度、dtype、量化|

不能用吞吐替代单请求延迟。增加 batch 可能提高设备利用率，却让短请求排队；连续 batching 可以提高总体吞吐，但每个请求的 TTFT 和 inter-token latency 需要单独记录。

## batch 和调度改变实际执行

### padding batch

把长度不同的 prompt 补齐到同一长度 $T_{\max}$ 时，理论计算位置为 $B T_{\max}$，有效位置为

$$
T_{\mathrm{valid}}
=
\sum_{b=1}^{B}T_b.
$$

padding waste 可以定义为

$$
W_{\mathrm{pad}}
=
1-
\frac{T_{\mathrm{valid}}}
{B T_{\max}}.
$$

长度差异大时，静态 batch 会浪费计算和显存。attention mask 可以阻止 PAD 进入结果，但不能自动消除所有 kernel 的存储和调度开销。

### continuous batching

服务可以让新请求加入正在运行的 decode batch，也可以在请求结束后释放 slot。调度器要维护：

1. request ID 与 batch slot 的映射；
2. 每个请求的 prompt offset 和生成长度；
3. cache 的物理位置与逻辑位置；
4. 每个请求的停止状态；
5. batch 重排时的 sequence index；
6. 每个请求的队列和取消时间。

batch slot 改变时，cache 迁移和 logits 对齐必须同时发生。只更新 token 数而不更新 cache index，会让一个请求读取另一个请求的历史。

### prefix cache

多个请求共享相同前缀时，可以复用 prefill 产生的 cache。缓存键应包含：

$$
\operatorname{key}
=
\operatorname{Hash}
\left(
\operatorname{Tok},
\operatorname{template},
x_{1:T_{\mathrm{prefix}}},
\operatorname{mask},
\operatorname{position}
\right).
$$

命中 prefix cache 可以降低 TTFT，但会引入 cache 命中率、过期策略、租户隔离和内存回收问题。服务日志应区分 cold prefill、prefix hit 和部分命中。

## 推理显存有四本账

一次请求的峰值显存可以粗略拆成：

$$
S_{\mathrm{peak}}
\approx
S_{\mathrm{weights}}
+
S_{\mathrm{KV}}
+
S_{\mathrm{activation}}
+
S_{\mathrm{workspace}}.
$$

|账本|随什么增长|典型控制手段|
| --- | --- | --- |
|权重|模型参数数量和存储 dtype|量化、分片、CPU offload|
|KV cache|并发请求、层数、K/V head、上下文长度|GQA/MQA、分页 cache、长度上限|
|激活|batch、prefill 长度、临时输出|kernel、分块、释放中间值|
|workspace|算子实现、attention kernel、量化 kernel|选择 kernel、限制 batch、复用 buffer|

参数内存是模型级固定成本，KV cache 是请求级动态成本。多个短请求可能比一个长请求占用更多 cache；并发是推理系统的核心资源变量。

### GQA 和 MQA 影响 cache

若 query head 数为 $H_q$，K/V head 数为 $H_{\mathrm{kv}}$：

$$
S_{\mathrm{KV}}
\propto
H_{\mathrm{kv}}.
$$

MHA 取 $H_{\mathrm{kv}}=H_q$，GQA 取较小的 K/V head 数，MQA 取一个 K/V head。减少 K/V head 会降低 cache 内存和带宽，但模型结构、质量和 kernel 支持也会改变。不能只比较 cache 数字而忽略模型家族。

## 数值和运行态要保持一致

### 推理要使用 evaluation mode

推理一般关闭 dropout，并把模型设为 evaluation mode。若模型包含运行统计、随机采样或外部工具状态，必须明确这些状态是否冻结。训练期的 grad、optimizer state、loss scaler 和 backward 不应进入普通推理路径。

### 低精度需要记录边界

混合精度或量化推理会改变 logits 的舍入、激活范围和内存带宽。[混合精度训练](../training-nn/mixed-precision/)讨论训练期的 dtype 和 loss scaling；推理至少记录：

- 权重 dtype 和量化格式；
- activation 与累加器 dtype；
- logits 是否转成 float32 做 softmax；
- KV cache dtype；
- dequantize 的位置；
- 非有限值和异常 token 的处理。

如果 logits 只用于 greedy，比较最大值时的舍入和 tie-breaking 也可能改变 token。需要固定 tie-breaking 规则和随机 seed。

### 重复请求要分清确定性层级

|层级|可复现条件|仍可能改变的因素|
| --- | --- | --- |
|同一进程|checkpoint、输入、seed、kernel 固定|异步执行和非确定性 kernel|
|同一设备|软件、驱动、硬件固定|并发调度、低精度归约|
|跨设备|算子和通信合同一致|硬件内核、量化和浮点舍入|
|跨服务|请求顺序、batch、cache、超时一致|队列和连续 batching|

“固定 seed”只约束使用随机数的路径，不能自动保证不同 batch 或不同 kernel 的 bitwise 一致。质量评测和延迟基准要分别声明要求统计一致还是逐 token 一致。

## 一个可复算的推理账本

设 $B=2$、prompt 长度为 $4$、生成 $3$ 个 token，模型有 $L=2$ 层、$H_{\mathrm{kv}}=4$、$d_h=8$，K/V 使用 FP16，每个标量占 $2$ bytes。按“第 $j$ 次 decode 读取长度 $4+j$ 的历史 cache”计数：

|量|数值|
| --- | ---: |
|decode context lengths|$[4,5,6]$|
|每步 attention score terms|$[512,640,768]$|
|累计 score terms|$1920$|
|最终 cache 长度|$7$|
|最终 KV cache|$3584$ bytes = $3.500000$ KiB|

每步 score terms 使用

$$
B L H_{\mathrm{kv}} d_h T_j
$$

计算。最终 cache 使用 K、V 两份张量：

$$
S_{\mathrm{KV}}
=
B L 2 H_{\mathrm{kv}}
(T_{\mathrm p}+K)
d_h b
=
3584\ \mathrm{bytes}.
$$

如果队列时间为 $12$ ms、prefill 为 $8$ ms、每次 decode 为 $3$ ms，则：

$$
\operatorname{TTFT}
=
12+8+3
=
23\ \mathrm{ms},
$$

$$
T_{\mathrm{e2e}}
=
12+8+3\times3
=
29\ \mathrm{ms}.
$$

prefill throughput 为 $4/0.008=500$ token/s，decode throughput 为 $1/0.003=333.333$ token/s。这个账本只计算 attention 交互和 KV 状态，不包含线性层、FFN、采样和网络传输；实际 benchmark 要把这些阶段单独计时。

## 一个标准库探针

下面的代码只使用 Python 标准库，核对 decode context 长度、KV cache 内存、attention score 项、TTFT、端到端时间和阶段吞吐。它不执行真实模型前向，因此输出用于检查账本和日志字段。

```python
batch_size = 2
prompt_tokens = 4
new_tokens = 3
layers = 2
kv_heads = 4
head_dim = 8
bytes_per_value = 2

contexts = [
    prompt_tokens + index
    for index in range(new_tokens)
]
score_terms = [
    batch_size * layers * kv_heads * head_dim * length
    for length in contexts
]
cache_length = prompt_tokens + new_tokens
kv_cache_bytes = (
    batch_size
    * layers
    * 2
    * kv_heads
    * cache_length
    * head_dim
    * bytes_per_value
)

print(f"contexts={contexts}")
print(
    f"decode_score_terms={score_terms} "
    f"total_score_terms={sum(score_terms)}"
)
print(
    f"final_kv_cache_bytes={kv_cache_bytes} "
    f"final_kv_cache_kib={kv_cache_bytes / 1024:.6f}"
)

queue_seconds = 0.012
prefill_seconds = 0.008
decode_seconds = 0.003
ttft = (
    queue_seconds
    + prefill_seconds
    + decode_seconds
)
end_to_end = (
    queue_seconds
    + prefill_seconds
    + new_tokens * decode_seconds
)
print(
    f"ttft_ms={ttft * 1000:.3f} "
    f"end_to_end_ms={end_to_end * 1000:.3f}"
)
print(
    f"prefill_tokens_per_s={prompt_tokens / prefill_seconds:.3f} "
    f"decode_tokens_per_s={1 / decode_seconds:.3f}"
)
```

输出为：

```text
contexts=[4, 5, 6]
decode_score_terms=[512, 640, 768] total_score_terms=1920
final_kv_cache_bytes=3584 final_kv_cache_kib=3.500000
ttft_ms=23.000 end_to_end_ms=29.000
prefill_tokens_per_s=500.000 decode_tokens_per_s=333.333
```

输出中的 cache 使用了 K 和 V 两份张量。score terms 随每次 decode 读取的历史长度增加。TTFT 包含队列、prefill 和首次 decode；端到端时间还包含后续两个 decode。

## 运行方法

把上一节的代码保存为 inference_probe.py，然后运行：

```bash
python3 inference_probe.py
```

真实服务接入时，保留同名日志字段，并增加 request ID、model hash、tokenizer version、cache hit、batch slot、停止原因和错误阶段。不要把后处理后的文本长度当作生成 token 数，也不要把 prefill throughput 当作单步 decode throughput。

## 失效模式

### 把 prefill 和 decode 当成同一个阶段

prefill 处理整个 prompt，decode 每次只处理新 token。两者的并行度、attention 形状、延迟和显存访问不同。分别记录 TTFT、inter-token latency 和阶段吞吐。

### 不记录 tokenizer 和模板

同一字符串经过不同 tokenizer 或 chat template 后，prompt token 数和可见上下文不同。保存原始文本 hash、tokenizer 版本、模板和 input_ids。

### KV cache 的 batch 映射错误

连续 batching、请求结束和 slot 重排会改变 cache 的物理位置。若 request ID、sequence ID 和 cache index 不同步，一个请求会读取另一个请求的历史。保存每次重排的映射和 cache 长度。

### 只计算权重显存

长上下文和高并发下，KV cache 可以成为主要内存项。按请求数、prompt 长度、生成长度、K/V head 和 dtype 估算峰值，而不是只看 checkpoint 大小。

### stop sequence 处理不一致

服务端可能把 stop sequence 从返回文本中删除，或者把 EOS 保留为 token。记录原始 token、停止原因、匹配位置和最终文本。

### 用生成 PPL 解释自由运行质量

PPL 使用真实前缀，生成使用模型前缀。错误会在生成中传播，两个指标需要分开报告。用 [困惑度评估](../pretraining/evaluation-perplexity/)的协议得到 teacher-forced 分数，再使用任务指标评价生成结果。

### 固定 seed 却忽略 batch 和 kernel

随机 seed 不能覆盖连续 batching、非确定性归约、量化舍入和请求顺序。基准测试要固定并发和调度，质量测试要报告统计重复。

### 低精度 logits 直接取概率

低精度 softmax 可能把尾部概率下溢为零，极端 logits 还会产生非有限值。使用稳定 log-softmax 或明确的 logits 精度，记录非有限检查。

### prefix cache 跨协议复用

不同模板、位置 offset、mask 或 tokenizer 版本不能共用 cache。缓存键要覆盖完整前缀协议，命中和失效要可审计。

## 推理审计清单

|范围|确认项|证据|
| --- | --- | --- |
|模型|checkpoint、架构、参数 dtype 和量化版本固定|hash、配置快照|
|输入|原文、tokenizer、模板、input_ids 和 prompt 长度固定|输入 manifest|
|prefill|输入 shape、首次 logits、cache 长度和 TTFT 可见|阶段 trace|
|decode|每步 token、context length、logits 和 cache offset 可见|token trace|
|停止|EOS、stop sequence、max tokens、timeout 和取消语义固定|停止原因|
|解码|temperature、top-k/top-p、beam、seed 和 tie-breaking 固定|generation config|
|batch|request ID、slot、padding、连续 batching 和重排可见|scheduler trace|
|内存|权重、KV、激活、workspace 分账|峰值显存与 cache ledger|
|性能|TTFT、inter-token latency、prefill/decode throughput 分开|benchmark report|
|数值|evaluation mode、dtype、logits 精度和 finite 检查固定|runtime config、异常日志|
|输出|原始 token、后处理文本和错误状态同时保存|response record|

推理结果是模型参数、输入协议、解码规则和服务调度共同产生的执行记录。只有把这些字段拆开，质量、延迟和显存的变化才有可比较的解释。

## 相关词条

- [Decoder-only Transformer](../transformer-architectures/decoder-only/)
- [因果语言建模](../transformer-architectures/causal-language-modeling/)
- [Tokenization](../text-representation/tokenization/)
- [困惑度评估](../pretraining/evaluation-perplexity/)
- [参数量与资源账本](../transformer-components/parameter-count/)
- [训练稳定性](../pretraining/training-stability/)
- [混合精度训练](../training-nn/mixed-precision/)
- [分布式训练](../pretraining/distributed-training/)
