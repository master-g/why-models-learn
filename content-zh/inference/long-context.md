---
title: "长上下文：当 token 数超过短序列假设"
tags: ["why-models-learn"]
---

长上下文描述模型在较长 token 序列上继续完成读取、推理或生成的能力。上下文窗口是模型和运行时允许的最大 token 数，effective context 是在固定任务和评测协议下仍能可靠利用的信息范围。两者需要分开报告：窗口可以扩大，远距离信息的可用性、位置分布、注意力成本、KV cache 和评测质量仍可能变化。

![长上下文示意图：token 序列包含局部窗口、远距离检索位置和末端生成位置，注意力与 KV cache 随长度增长](/assets/inference/svg/long-context.1.svg)

## 先固定上下文长度合同

### 窗口上限与有效上下文不是同一个量

设输入 prompt 有 $T_{\mathrm{prompt}}$ 个 token，计划生成 $T_{\mathrm{new}}$ 个 token，系统允许的最大窗口为 $T_{\max}$。最基本的合同是：

$$
T_{\mathrm{prompt}}+T_{\mathrm{new}}\leq T_{\max}.
$$

若服务把系统消息、工具 schema、检索片段、对话历史和用户输入全部拼接到同一序列，则每一部分都要计入 $T_{\mathrm{prompt}}$。字符数、词数、消息条数和 token 数不能互换。

effective context 还要附带任务条件。例如：

|概念|定义|需要附带的条件|
|---|---|---|
|context window|模型和运行时允许处理的最大 token 数|tokenizer、position、显存和服务上限|
|effective context|在指定任务上仍保持目标质量的 token 范围|指标、阈值、位置和干扰项|
|retrieval distance|查询与证据 token 之间的位置距离|方向、相对位置、文档边界|
|usable generation budget|留给新输出的 token 数|prompt 长度、停止条件、max new tokens|

一个 128k 窗口的模型可能在 64k 位置仍能完成格式约束，却无法稳定找到位于中间的单个证据。有效上下文必须通过位置和长度分桶评测得到，不能由窗口上限直接推导。

### tokenization 先于长度比较

长上下文实验必须冻结：

|字段|示例|影响|
|---|---|---|
|tokenizer|词表、normalizer、特殊 token|同一文本的 token 数|
|模板|system、user、assistant、工具消息格式|固定开销和边界位置|
|文档分隔符|换行、标题、标记 token|检索和段落边界|
|截断方向|保留前缀、保留后缀或双侧裁剪|证据是否被移除|
|padding|左 padding、右 padding、packed sequence|位置 ID 和有效 mask|
|输出预算|max new tokens、EOS、stop sequence|输入可使用的最大长度|

测量有效上下文前，应保存 token ID 序列和每个片段的起止位置。只保存原始字符串无法解释 tokenizer、模板和截断造成的长度差异。

### 预算应该逐项列出

|片段|token 预算|是否可裁剪|审计字段|
|---|---:|---|---|
|system prompt|固定或版本化|通常不裁剪|模板 hash、版本|
|工具定义|随工具数量变化|可按工具选择|schema、顺序|
|对话历史|随轮数增长|可摘要或淘汰|消息 ID、保留规则|
|检索文档|随召回数量增长|可重排或压缩|doc ID、位置、分数|
|用户输入|当前请求|通常保留|tokenizer、原文 hash|
|生成预算|max new tokens|服务约束|EOS、stop、超时|

预算不足时的处理需要显式记录。静默截掉最早消息会改变任务输入，也会让长度比较失去可比性。

## 长度如何改变计算与内存

### Dense attention 的交互项随长度平方增长

对 batch $B$、有效长度 $T$、主宽度 $D$ 的 dense attention，query-key 和 attention-value 的主要 MAC 项可以写成：

$$
\operatorname{MAC}_{\mathrm{attn}}
\approx
2BT^2D.
$$

把长度从 $T$ 增加到 $cT$，这一项变成：

$$
\operatorname{MAC}_{\mathrm{attn}}(cT)
=
c^2\operatorname{MAC}_{\mathrm{attn}}(T).
$$

FFN 和投影通常随 token 数线性增长，而 dense attention 的交互项会成为长 prefill 的主要长度敏感项。FlashAttention 可以避免显式保存完整 score matrix，仍然需要完成相同的逻辑交互，内存峰值和实际带宽路径会改变。

### KV cache 对长度线性增长

对 $L$ 层、batch $B$、历史长度 $P$、K/V 宽度 $D_{kv}$、每个元素 $s$ bytes 的 cache，主 payload 为：

$$
M_{\mathrm{KV}}
\approx
2LBPD_{kv}s.
$$

若使用 GQA，$D_{kv}=h_{kv}d_h$；MHA、GQA 和 MQA 的 query width 可以相同，KV cache width 不同。

以 $L=32$、$B=1$、$D=4096$、$h=32$、$h_{kv}=8$、$d_h=128$、FP16 为例，$D_{kv}=1024$。长度账本为：

|有效 token 数|attention 主要 MAC|FP16 KV cache 主 payload|
|---:|---:|---:|
|4,096|137,438,953,472|536,870,912 bytes|
|32,768|8,796,093,022,208|4,294,967,296 bytes|
|131,072|140,737,488,355,328|17,179,869,184 bytes|

从 4,096 增加到 32,768，dense attention 主要项增加 64 倍，KV cache 增加 8 倍；增加到 131,072 时，对应倍率为 1,024 倍和 32 倍。实际显存还要加入权重、activation、workspace、page 对齐和 scale metadata。

### Prefill、decode 和长上下文的瓶颈不同

|阶段|长度相关项|常见资源压力|测量字段|
|---|---|---|---|
|长 prompt prefill|$T^2$ attention、$T$ 个投影和 FFN|计算、workspace、带宽|TTFT、FLOPs、峰值显存|
|单请求 decode|每步读取 $P$ 长度的 KV|KV 带宽、cache locality|ITL、读取 bytes、batch|
|多请求 decode|每条请求的 $P_b$ 不同|padding、slot、调度|active batch、有效 token|
|chunked prefill|多个短块累积到同一前缀|同步、调度、cache 写入|chunk size、queue、峰值|

长上下文的 prefill 变慢，不能用 decode 的 token/s 直接解释。decode 变慢时，需要先检查历史 KV 的读取和 batch 形状，再检查生成器和网络。

## 位置编码决定远距离关系如何表达

### 位置是输入合同的一部分

模型需要知道 token 的顺序和相对距离。位置机制至少决定：

- 位置 ID 从哪里开始；
- padding 是否占用位置；
- packed sequence 是否在文档边界重置；
- cache offset 如何进入下一次 decode；
- 训练长度和推理长度的关系；
- 超出训练位置范围时如何处理。

相同 token 内容在不同位置产生的 hidden 可能不同。位置 ID、mask 和 cache offset 必须和 token 序列一起保存。

### 绝对位置编码有直接的长度边界

learned absolute position embedding 通常为每个位置保存一个向量表。若训练表只覆盖到 $T_{\mathrm{train}}$，推理位置超过该范围时需要扩表、插值或其他规则；直接访问不存在的位置没有定义。

sinusoidal encoding 用固定频率生成位置向量，外推形式更直接，但模型是否学会在更远位置使用这些频率仍需要评测。位置表达的形式可外推，不代表任务行为自动外推。

### RoPE 把位置放进 Q/K 的相对相位

RoPE 对每一对坐标使用旋转角度。用位置 $p$、维度索引 $i$、base 和模型维度 $d$ 表示时，可以写成：

$$
\theta_i(p)
=
p\cdot\omega_i,
\qquad
\omega_i
=
\mathrm{base}^{-2i/d}.
$$

query 和 key 分别在自己的位置旋转后再点积。旋转矩阵的组合使相对距离进入点积，但相位随位置增加也会继续变化。长位置下需要关注角度分布、低频与高频分量、训练长度和数值精度。

context-extension 词条会专门讨论 RoPE scaling、插值和外推策略。本篇只固定位置合同与评测边界，不把某一种扩展方法当作长上下文能力本身。

### ALiBi 用距离偏置改变 score

对 causal attention 的 query 位置 $p$ 和可见 key 位置 $r$，一种距离偏置可以写成：

$$
\operatorname{score}_{p,r}
=
\frac{q_p^\mathsf{T}k_r}{\sqrt{d_h}}
-
m_h(p-r),
\qquad
r\leq p.
$$

斜率 $m_h$ 按 head 选择。距离越远，偏置项通常越大，模型会获得明确的距离归纳偏置。实际有效范围仍取决于训练分布、任务证据位置和注意力竞争。

### 位置机制的比较

|机制|距离如何进入模型|长位置风险|评测重点|
|---|---|---|---|
|learned absolute|按位置查表相加|表长度和训练边界|超表边界、插值|
|sinusoidal|固定频率向量|频率组合与任务使用|长度外推、相位|
|RoPE|旋转 Q/K、相对相位|高频相位和训练外推|相对距离、长位置 logits|
|ALiBi|沿距离加入 score bias|距离偏置与内容竞争|远近证据、head 分工|
|相对位置 bias|按相对桶或距离查表|桶范围和边界|距离桶、跨段关系|

比较位置机制时要保持模型大小、训练 token、tokenizer、prompt 模板、评测样本和生成协议一致。只改变最大长度配置，不能说明位置机制的完整差异。

## 长上下文能力还取决于信息分布

### 远距离读取有三个独立问题

长序列中，模型需要同时处理：

1. 找到证据所在的位置。
2. 区分证据与相似干扰项。
3. 把证据与当前问题组合起来。

这三个步骤分别对应定位、选择和组合。单个 needle retrieval 可以检查定位，但无法覆盖多证据推理和冲突信息处理。

### Lost in the middle 要按位置测量

同一证据放在开头、中间和结尾，任务难度可能不同。位置实验应把文档长度固定，只改变证据位置：

|位置桶|证据位置|观察|
|---|---:|---|
|开头|0% 附近|前缀注意力、系统消息竞争|
|前中段|25% 附近|局部和远距离混合|
|中间|50% 附近|长距离竞争和信息衰减|
|后中段|75% 附近|靠近生成位置的读取|
|结尾|100% 附近|短距离优势、prompt 尾部格式|

每个位置至少使用相同数量的负样本、干扰项和答案格式。只报告平均准确率会掩盖位置曲线。

### 证据密度和干扰项需要独立控制

|变量|低值|高值|要保持不变的部分|
|---|---|---|---|
|文档长度|短上下文|长上下文|tokenizer、模板、答案|
|证据数量|单证据|多证据|问题和证据关系|
|干扰项数量|无相似文本|多个相似文本|正确证据内容|
|证据距离|相邻|跨段或跨文档|任务规则|
|组合深度|单跳|多跳|输出格式和评分|
|重复程度|唯一描述|多次相似描述|正确标签和位置桶|

长上下文评测应保存每个证据的 token 起止位置、文档 ID、干扰项类型和答案依赖关系。否则错误无法定位到长度、位置还是推理组合。

## 上下文组织可以改变可用范围

### 全局 attention 不是唯一组织方式

当完整 dense attention 的成本过高，可以设计局部、全局或分层读取路径：

|组织方式|读取范围|优势|新增合同|
|---|---|---|---|
|dense attention|所有可见位置|表达直接、兼容性高|$T^2$ 交互和长 cache|
|sliding window|最近 $W$ 个位置|局部成本受控|远距离信息需要桥接|
|global tokens|局部窗口加少量全局位置|保留跨段通道|全局 token 选择和预算|
|block sparse|按 block 图连接|可利用结构稀疏|稀疏模式、kernel 和边界|
|层级摘要|局部块先摘要、上层读取摘要|压缩长文档|摘要误差、更新时机|
|检索式读取|先选候选片段再计算|减少无关 token|召回率、重排、位置信息|

局部注意力减少交互数量，但会改变信息可达路径。若两个远距离位置没有共同的全局 token、摘要或检索桥接，模型不能依赖不存在的直接读取边。

### 位置与摘要会产生新的状态

摘要、压缩和检索都会引入额外状态：

|状态|必须保存|失配时的结果|
|---|---|---|
|chunk 边界|原始 token 范围、文档 ID|摘要无法回指原文|
|摘要版本|模型、prompt、生成配置|相同文档产生不同摘要|
|retrieval index|embedding、分块、版本|召回结果不可复现|
|全局 token|选择规则和位置|不同长度使用不同桥接|
|KV cache|逻辑长度、物理 block、scale|读取错位或显存泄漏|

长上下文系统的答案可能来自原始 token、摘要、检索片段和 cache 的组合。评测需要区分这些路径，不能只比较最终文本。

### Chunking 要保持位置和边界语义

把长 prompt 切成 chunk 可以降低单次峰值，但 chunk 之间如何连接决定了模型能否看到完整上下文：

- 直接重置位置会丢失全局顺序；
- 直接拼接 chunk 需要维护连续 position ID；
- 只保留摘要会丢失细节；
- 只保留 KV cache 需要保证 tokenizer、mask 和模型版本一致；
- chunk 之间的同步会影响 TTFT 和队列延迟。

chunk size、overlap、位置偏移、摘要 prompt、cache ownership 和失败重试都属于推理协议。

## 长上下文评测要把长度与能力拆开

### 长度桶和位置桶必须同时存在

最小实验矩阵至少包含：

|维度|建议取值|目的|
|---|---|---|
|总 token 长度|4k、8k、16k、32k、64k 或实际窗口|观察长度曲线|
|证据位置|开头、前中、中间、后中、结尾|观察位置曲线|
|干扰项|0、相似、冲突、跨文档|观察选择能力|
|任务类型|检索、分类、单跳、多跳、生成|区分读取与组合|
|输出预算|固定或按窗口比例|避免生成长度混淆|
|运行模式|dense、window、retrieval、cache|比较组织方式|

若资源有限，应优先保留所有长度的中心位置和所有位置桶的短/中/长三个长度，而不是只测一个最长配置。

### Perplexity 要按位置聚合

对自回归模型，长序列中的每个 target token 都有条件概率。若只报告整段平均 NLL，靠近文档尾部的 token 数量可能掩盖前中段的退化。可以按 token 位置或距离分桶：

$$
\operatorname{NLL}(r)
=
-\frac{1}{\lvert S_r\rvert}
\sum_{i\in S_r}\log p(y_i\mid y_{<i}),
$$

其中 $S_r$ 是位置桶 $r$ 中的有效 target 集合。分母、mask、文档边界和 stride 必须固定。

### Retrieval probe 不能替代真实任务

needle-in-a-haystack 可以快速画长度—位置热图，但它通常具有：

- 证据表述唯一；
- 答案短；
- 干扰项与真实文档不同；
- 评分接近 exact match；
- 任务只需要读取，不需要组合。

因此还需要加入自然文档、相似证据、冲突证据、表格、代码、跨段引用和多跳问题。每类任务都应保存原文、token 位置和评分规则。

### 评测结果的最小报告

|报告项|必须包含|
|---|---|
|模型|checkpoint hash、架构、训练长度、位置机制|
|输入|tokenizer、模板、长度、片段边界、证据位置|
|运行|device、dtype、batch、cache、attention 模式|
|生成|temperature、top-k/top-p、stop、max new tokens|
|质量|准确率、NLL/PPL、位置曲线、置信区间|
|资源|TTFT、ITL、峰值显存、KV bytes、吞吐|
|失败|超窗、截断、OOM、fallback、超时和重试|

长上下文能力的结论必须附带上下文长度、证据位置和运行模式。没有这些字段，跨模型或跨版本比较的含义不稳定。

## 运行方法

下面的探针固定 GQA-8、FP16、$L=32$、$B=1$、$D=4096$，比较 4k、32k 和 128k token。它只计算主 attention MAC 与 KV payload，不把 workspace、scale、page 对齐和权重混入 cache 数字。

```python
L = 32
D = 4096
Dkv = 1024
B = 1

for tokens in (4096, 32768, 131072):
    attention_macs = 2 * B * tokens * tokens * D
    kv_bytes = 2 * L * B * tokens * Dkv * 2
    print(
        "tokens=",
        tokens,
        "attention_macs=",
        attention_macs,
        "kv_bytes=",
        kv_bytes,
        "kv_gib=",
        f"{kv_bytes / 2**30:.3f}",
    )

print("attention_ratio_32768_vs_4096=", (32768 // 4096) ** 2)
print("attention_ratio_131072_vs_4096=", (131072 // 4096) ** 2)
print("kv_ratio_32768_vs_4096=", 32768 // 4096)
print("kv_ratio_131072_vs_4096=", 131072 // 4096)
```

运行输出：

```text
tokens= 4096 attention_macs= 137438953472 kv_bytes= 536870912 kv_gib= 0.500
tokens= 32768 attention_macs= 8796093022208 kv_bytes= 4294967296 kv_gib= 4.000
tokens= 131072 attention_macs= 140737488355328 kv_bytes= 17179869184 kv_gib= 16.000
attention_ratio_32768_vs_4096= 64
attention_ratio_131072_vs_4096= 1024
kv_ratio_32768_vs_4096= 8
kv_ratio_131072_vs_4096= 32
```

输出与公式一致：长度乘 8 时，dense attention 主要项乘 64，KV payload 乘 8；长度乘 32 时，分别乘 1,024 和 32。实际长上下文运行还要测量 padding、cache metadata、workspace、带宽和 kernel 路径。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|配置允许长窗口但加载失败|position table、RoPE 参数、服务 max length|打印最终 position 与模型配置|
|长文档前后段正常、中间段下降|证据位置、干扰项和 attention 竞争|按位置桶重复相同任务|
|长度增加后 TTFT 非线性上升|dense attention、workspace、padding|记录有效 token、MAC 和 kernel|
|decode ITL 随历史增长|KV bytes、cache locality、active batch|按 $P$ 分桶记录读取和 ITL|
|OOM 只在长 prompt 出现|KV cache、score materialization、activation|分阶段记录峰值显存|
|chunked 结果与整段结果不同|位置 offset、摘要、mask、边界|保存逐 chunk token 和 state|
|检索命中但回答错误|证据组合、冲突项、生成协议|分别评测定位和多跳推理|
|长上下文只在一种语言有效|tokenization、校准和训练分布|按语言和 token 数配对|
|不同 batch 得到不同答案|动态 padding、RNG、cache slot|固定 request-level 状态和排序|

### 最小长上下文审计表

|检查项|应保存|验收问题|
|---|---|---|
|长度|prompt、generation、window、有效 token|是否触及窗口和输出预算|
|位置|position ID、证据起止、cache offset|位置是否连续且可复现|
|mask|padding、causal、document boundary|模型实际能读取哪些 token|
|组织|dense、window、sparse、retrieval、summary|远距离信息走哪条路径|
|cache|逻辑长度、物理 block、dtype、bytes|cache 是否与序列同步|
|计算|attention MAC、workspace、带宽、kernel|长度成本是否符合理论|
|质量|位置曲线、长度曲线、NLL、任务分数|有效上下文到哪里|
|失败|截断、OOM、fallback、超时、重试|失败是否被静默处理|
|复现|checkpoint、tokenizer、模板、版本、seed|另一台设备能否重跑|

长上下文的核心问题是信息是否仍能被模型在正确位置、正确 mask 和可承受资源内读取。窗口上限只定义可处理的 token 数；effective context 还需要位置评测、干扰控制、任务组合和系统资源证据。后续 context-extension 将在此合同上讨论如何改变位置范围与训练—推理长度关系。

## 相关词条

- [推理数学](../inference/inference-math/)：拆分长 prefill、decode、attention、KV cache 和显存账。
- [KV cache](../inference/kv-cache/)：展开历史 K/V 的 shape、追加、分页和读取成本。
- [注意力复杂度](../attention/attention-complexity/)：比较 dense、稀疏和局部 attention 的计算与内存增长。
- [RoPE](../transformer-components/rope/)：说明旋转位置编码的相位、频率和相对位置关系。
- [ALiBi](../transformer-components/alibi/)：说明按距离加入 attention score 偏置的位置机制。
- [FlashAttention](../attention/flash-attention/)：说明 tile 化如何避免显式保存完整 attention matrix。
- [稀疏注意力](../attention/sparse-attention/)：比较局部、全局和稀疏可达路径。
- [上下文扩展](../inference/context-extension/)：讨论训练长度之外的位置扩展和 RoPE scaling。
- [推理](../inference/inference/)：固定端到端请求、batch、停止条件和实际延迟。
