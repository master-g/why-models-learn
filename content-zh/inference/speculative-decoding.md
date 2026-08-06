---
title: "投机解码：用草稿模型减少目标模型调用"
tags: ["why-models-learn"]
---

投机解码让一个较小的 draft model 先连续提出多个 token，再让目标模型一次前向验证这些 token。被接受的前缀直接提交；第一个被拒绝的位置从目标分布相对于草稿分布的残差中重采样。只要两个分布在同一 token 空间上、接受概率和残差分布实现正确，输出序列的边际分布仍然是目标模型分布。投机解码改变的是目标模型调用方式，不是把 draft model 的输出直接当成目标模型输出。

![投机解码示意图：草稿模型提出多个 token，目标模型批量验证，接受前缀并在首个拒绝位置从残差分布重采样](/assets/inference/svg/speculative-decoding.1.svg)

## 两个模型与一个目标分布

### Draft model 负责提出候选

设当前已提交的前缀为 $s$。目标模型在位置 $i$ 的 next-token 分布为：

$$
p_i(v)
=
P_{\mathrm{target}}(v\mid s,y_{<i}).
$$

Draft model 使用分布：

$$
q_i(v)
=
P_{\mathrm{draft}}(v\mid s,y_{<i}).
$$

Draft model 自回归提出 $\gamma$ 个 token：

$$
y_1,\ldots,y_\gamma,
\qquad
y_i\sim q_i.
$$

这里的 $y_{<i}$ 是本轮已经提出的草稿 token 前缀。目标模型不需要逐个 decode 这些 token，而是可以在一次验证前向中同时计算多个位置的 logits。

目标模型是最终分布的权威。Draft model 只影响候选顺序、接受率和运行成本，不直接决定输出概率。

### Token 空间必须兼容

标准投机解码要求两个模型能够对同一个 token ID 序列计算条件分布。至少需要固定：

|兼容项|要求|不满足时的结果|
|---|---|---|
|tokenizer|词表、token ID 和分词边界兼容|不能直接比较 $p_i(v)$ 与 $q_i(v)$|
|位置协议|position ID、RoPE 或其他位置规则一致|验证位置与提议位置不一致|
|mask|causal、padding、语法和 bad token 规则可比较|接受率和目标分布定义改变|
|EOS/stop|结束 token 和停止规则有明确合同|会提交错误长度或越过停止点|
|score 处理|temperature、top-k/top-p 等规则固定|比较的是不同的目标分布|
|数值类型|概率或 log probability 的计算精度有记录|边界接受结果可能漂移|

如果 draft model 使用不同 tokenizer，可以采用重新对齐、词片段提议或其他变体，但那不是本文的同 token ID 算法。不能把不同词表的 token 字符串直接按位置比较。

### 目标分布要包含最终处理规则

若服务的最终采样使用 temperature、repetition penalty、top-k 或 top-p，目标分布应定义为这些规则全部处理后的分布：

$$
p_i
=
\operatorname{SampleDistribution}
\left(
\operatorname{Processor}_{\mathrm{target}}(z_i)
\right).
$$

Draft 分布 $q_i$ 可以使用不同模型和不同近似，但两者必须仍然是在同一候选空间上定义的有效分布。若验证时使用未处理的 target logits，而真正输出时再应用 top-p，算法证明对应的目标分布就不再是实际服务分布。

## 一轮提议与验证

### Draft 一次提出 $\gamma$ 个 token

从已提交前缀开始，draft model 自回归运行最多 $\gamma$ 步：

$$
s
\rightarrow
s\,y_1
\rightarrow
s\,y_1y_2
\rightarrow\cdots\rightarrow
s\,y_1\cdots y_\gamma.
$$

每个位置保存：

- draft token $y_i$；
- draft probability $q_i(y_i)$ 或 log probability；
- draft cache 的新 K/V；
- EOS、stop 和约束状态；
- request slot 与 position offset。

Draft model 可以在一轮内一次生成一段，但 token 之间仍然依赖前一个草稿 token。目标模型随后读取完整的候选段，在一次前向中产生 $p_1,\ldots,p_\gamma$ 以及继续位置的目标分布。

### 接受概率是概率比

在位置 $i$，对 draft 提出的 token $y_i$ 定义：

$$
a_i
=
\min
\left(
1,
\frac{p_i(y_i)}{q_i(y_i)}
\right).
$$

使用均匀随机数：

$$
u_i\sim\operatorname{Uniform}[0,1).
$$

若 $u_i<a_i$，接受 $y_i$。若拒绝，停止检查后续草稿 token，并在这个位置使用残差分布选择替代 token。

如果 $p_i(y_i)\ge q_i(y_i)$，则 $a_i=1$，这个 token 必然接受。如果目标模型给出的概率低于 draft model，接受概率才小于 1。

实现应在 log probability 上计算：

$$
\log a_i
=
\min
\left(
0,
\log p_i(y_i)-\log q_i(y_i)
\right).
$$

这样可以避免先把极小概率转换成下溢为 0 的浮点数。若 $q_i(y_i)=0$，draft 不应提出该 token；如果数据路径出现该状态，服务应记录分布支持集错误，而不是执行无穷比值。

### 首次拒绝位置使用残差分布

定义：

$$
d_i(v)
=
\max\left(p_i(v)-q_i(v),0\right).
$$

残差质量为：

$$
Z_i
=
\sum_{v=0}^{V-1}d_i(v).
$$

如果 $Z_i>0$，拒绝后的修正分布为：

$$
r_i(v)
=
\frac{d_i(v)}{Z_i}.
$$

从 $r_i$ 中采样一个 token，提交它并丢弃 $y_i,\ldots,y_\gamma$ 的剩余草稿尾部。目标分布中高于 draft 分布的部分由残差补回，目标和草稿相同的质量不需要再次补偿。

如果 $Z_i=0$，说明 $p_i(v)\le q_i(v)$ 对所有 token 都成立；在两个都归一化的分布下，这只能发生在 $p_i=q_i$。此时不存在拒绝事件，服务应直接把该位置视为必然接受，而不是除以 0。

### 一轮输出的两条路径

|事件|输出|缓存处理|
|---|---|---|
|第 1 个位置拒绝|从 $r_1$ 采样一个 token|丢弃全部草稿尾部，重算修正 token 的目标 K/V|
|第 $i$ 个位置拒绝|提交 $y_1,\ldots,y_{i-1}$ 和一个修正 token|保留已接受前缀，回滚位置 $i$ 之后的草稿状态|
|全部 $\gamma$ 个位置接受|提交全部草稿 token，再从目标分布取一个额外 token|提交草稿段，继续处理目标额外 token|
|遇到 EOS 或 stop|按停止合同结束或截断返回|同步结束 draft、target 和 request state|

“全部接受”时的额外目标 token 是标准算法的一部分。它让每轮在没有拒绝时继续推进，而不是把目标模型最后一个验证 logits 丢弃。某些实现省略这一步，但那是吞吐不同的变体，必须单独标记。

## 为什么边际分布仍然是目标分布

### 一步算法的质量分解

先看一个位置，省略下标 $i$。Draft 先按 $q(v)$ 提出 token。该 token 被接受的联合质量为：

$$
q(v)\min\left(1,\frac{p(v)}{q(v)}\right)
=
\min(p(v),q(v)).
$$

拒绝事件的总质量为：

$$
1-\sum_v\min(p(v),q(v)).
$$

而残差质量满足：

$$
\sum_v\max(p(v)-q(v),0)
=
1-\sum_v\min(p(v),q(v)).
$$

因此残差分布的归一化分母正好是拒绝概率。

### 接受质量加残差质量

算法最终输出 token $v$ 的质量为：

$$
\min(p(v),q(v))
+
\max(p(v)-q(v),0).
$$

对每个 token 都有：

$$
\min(p(v),q(v))
+
\max(p(v)-q(v),0)
=
p(v).
$$

所以一步算法的输出分布与目标分布 $p$ 完全相同。这里的“完全相同”指理想精度和正确实现下的边际分布相同，不表示固定 seed 下每次都会产生相同 token。

### 多步验证用最早拒绝位置切断

如果前 $i-1$ 个 draft token 全部被接受，第 $i$ 个位置的前缀正好是目标模型会看到的已提交前缀。第 $i$ 个位置可以使用上面的单步分解。第 $i$ 个位置之后的草稿 token 被丢弃，不会被错误地当作已接受历史。

如果全部 $\gamma$ 个 token 接受，目标模型在下一位置直接采样，分布为 $p_{\gamma+1}$。因此一轮可以理解成多次条件正确的单步转移串联。只要 cache、历史、position 和 stop 状态与接受前缀一致，整个生成序列的边际分布保持为目标模型分布。

## 接受率与吞吐

### 接受率等于分布重叠

单个位置的平均接受率为：

$$
A_i
=
\sum_v q_i(v)
\min
\left(
1,\frac{p_i(v)}{q_i(v)}
\right)
=
\sum_v\min(p_i(v),q_i(v)).
$$

它是两个分布的重叠质量。$A_i=1$ 时两者完全相同；$A_i$ 越低，越多草稿 token 在目标验证时被拒绝。

接受率不是 draft model 的单独质量指标。相同的 draft perplexity 可能在不同 prompt、温度、top-p、格式约束和 token 粒度下产生不同接受率。

### 一轮期望输出长度

设一轮提出 $\gamma$ 个 token，第 $i$ 个位置的接受概率为 $A_i$，并且全部接受后再输出一个目标 token。每一轮输出 token 数的期望为：

$$
\mathbb E[L]
=
1+A_1+A_1A_2+\cdots+\prod_{i=1}^{\gamma}A_i.
$$

如果每个位置的接受率近似为常数 $A$：

$$
\mathbb E[L]
=
\sum_{j=0}^{\gamma}A^j.
$$

第一项来自首个位置的修正 token 或全接受时的额外目标 token；后续乘积表示前缀全部接受后才能继续提交更多草稿 token。

### 速度不只由接受率决定

一次目标验证前向可以处理多个位置，但收益还受以下成本影响：

|成本|增加的工作|收益边界|
|---|---|---|
|draft forward|每轮最多生成 $\gamma$ 个 token|draft 必须明显便宜|
|target verify|一次处理草稿段|验证段不能超过显存和带宽|
|rollback/recompute|拒绝后修正 token 需要恢复 cache|拒绝频繁时收益降低|
|sampling|每个位置可能消耗接受 RNG 和修正 RNG|复现协议更复杂|
|batch scheduling|不同请求在不同位置拒绝|需要 ragged 或动态 batch|

如果 draft model 太慢、接受率太低或目标验证受到内存带宽限制，投机解码可能比普通 decode 更慢。部署结论需要测量 TTFT、每轮输出长度、target calls、draft calls、回滚次数和端到端 token latency。

## KV cache、回滚和提交

### Draft cache 与 target cache 分开

两套模型各自拥有一套 KV cache：

|状态|Draft cache|Target cache|
|---|---|---|
|已提交前缀|稳定保留|稳定保留|
|本轮草稿段|临时追加 $\gamma$ 个位置|验证前向产生临时分支|
|前缀接受后|保留接受位置，删除尾部|保留接受位置，修正拒绝位置|
|拒绝后|回滚未提交 token|用修正 token 重算目标 K/V|
|下一轮|从同一提交前缀继续|从同一提交前缀继续|

不能把 target 验证前向产生的全部 K/V 直接标记为 committed。被拒绝位置以及其后的 K/V 对最终序列无效。

### 首次拒绝后必须重算修正 token

假设 $y_i$ 被拒绝，残差采样得到 $c_i\ne y_i$。Target model 在验证前向中为输入 $y_i$ 计算的 K/V 不能直接代表 $c_i$。正确状态转移为：

1. 保留 $y_1,\ldots,y_{i-1}$ 的 target K/V；
2. 删除位置 $i$ 及之后的临时 target K/V；
3. 把修正 token $c_i$ 写入逻辑序列；
4. 用 target model 对 $c_i$ 执行一次 decode，产生合法的 K/V；
5. 从新的提交状态开始下一轮。

如果实现跳过第 4 步，下一次 query 会读取错误的历史 K/V。输出可能看似正常，但已经不再对应目标模型。

### 全接受时也要更新一致位置

全接受后，草稿 token 都成为最终序列。Target cache 可以保留验证前向中对应这些 token 的 K/V，但必须确认：

- target 输入位置与最终 position ID 一致；
- padding 和 causal mask 一致；
- target cache 的逻辑长度增加 $\gamma$；
- 额外目标 token 的 K/V 已写入；
- draft cache 与 target cache 都指向同一提交序列。

不同实现可能把验证 cache 写入临时 buffer，再按接受长度提交；也可能使用 copy-on-write block。两种布局都需要用 cache length、slot 和 token history 做断言。

### Paged cache 需要原子提交

使用 paged KV cache 时，本轮草稿段可能占用新的物理 blocks。拒绝后不能只减少逻辑长度，还要处理：

- 未提交 block 是否归还；
- block table 是否回滚；
- accepted prefix 是否保持同一物理地址；
- target 和 draft 的 block ownership 是否分开；
- batch 重排时临时 block 是否随 request 移动。

若回滚只更新了逻辑长度而没有清理 block table，下一轮可能读取旧的 rejected K/V。[KV cache](../inference/kv-cache/)的逻辑长度和物理 block 映射都属于提交协议。

## EOS、stop 和约束

### Draft EOS 不是最终 EOS

Draft model 可能提前提出 EOS。服务需要按合同决定：

- 把 EOS 作为普通候选交给 target 验证；
- draft EOS 后停止继续提议，但让 target 采样该位置；
- 直接结束本轮并验证 EOS；
- 对最小生成长度先 mask EOS。

只有最终提交的 EOS 才能结束目标序列。被拒绝的 draft EOS 不应改变 target 的 stop 状态。

### Stop sequence 需要在提交后检查

Stop sequence 可能跨越本轮多个 token。推荐在接受前缀和修正 token 提交后检查：

1. 把 token ID 追加到逻辑序列；
2. 按 tokenizer 解码或增量状态机更新 stop matcher；
3. 记录匹配起点和是否从返回文本移除；
4. 清理 draft/target 临时 cache；
5. 不再发起下一轮提议。

如果在验证前向产生的未提交草稿尾部上检查 stop sequence，可能把 rejected token 错误地算入最终文本。

### 约束会改变两个分布

若语法状态为 $g_i$，目标与 draft 都要根据各自的状态产生有效分布：

$$
p_i(v)=P_{\mathrm{target}}(v\mid s,y_{<i},g_i),
\qquad
q_i(v)=P_{\mathrm{draft}}(v\mid s,y_{<i},g_i).
$$

如果 draft 与 target 使用不同的约束 mask，残差分布仍可以在同一词表上修正，但接受率和候选支持集会变化。若某个 token 在 target 中允许、在 draft 中被永久 mask，draft 无法提出它，只有拒绝时的残差采样可以恢复该目标质量。实现需要验证 $q_i$ 是否是有效归一化分布，并处理 $q_i(v)=0$。

## 数值稳定与随机状态

### 在 log space 计算接受

当 $p_i(y_i)$ 和 $q_i(y_i)$ 很小时，直接相除可能下溢或溢出。使用：

$$
\log a_i
=
\min
\left(
0,
\log p_i(y_i)-\log q_i(y_i)
\right).
$$

再与 $\log u_i$ 比较：

$$
\log u_i<\log a_i
\Longrightarrow
\text{接受 }y_i.
$$

残差分布需要计算 $\max(p-q,0)$。可以在较高精度下把 log probability 转换为概率，或使用稳定的 log-difference-exp。无论采用哪种实现，都要检查：

$$
\sum_v p_i(v)\approx1,
\qquad
\sum_v q_i(v)\approx1.
$$

### 接受和修正需要记录 RNG 消耗

一轮可能消耗：

- 每个验证位置一个 acceptance uniform；
- 第一个拒绝位置一个 residual sample uniform；
- 全接受时一个 target sample uniform；
- 其他 top-k/top-p 或约束 sampler 的随机数。

接受位置改变后续消耗次数。要复现序列，必须固定 RNG scope、算法、调用顺序、候选 token 顺序和 rejection path。只保存 seed 不足以复现 draft/target 混合流水线。

### 接受率统计不能替代分布核验

一个高接受率只说明 draft 与 target 的分布重叠较大，不证明残差采样实现正确。需要同时检查：

- 接受率是否接近 $\sum_v\min(p_i(v),q_i(v))$；
- 拒绝后的修正分布是否归一化；
- 一步输出频率是否接近 target $p_i$；
- 全接受额外 token 是否来自 target；
- rejection path 后的 cache 和 history 是否正确。

## 与其他解码规则的边界

|规则|主要模型|输出方式|是否保持 target 边际分布|
|---|---|---|---|
|普通 target sampling|target model|每步调用 target 并采样|是|
|speculative decoding|draft 提议、target 验证|接受草稿或从残差修正|在合同满足时是|
|greedy draft|draft model|用 draft argmax 提议|需要把 proposal 视为确定提议，不能直接套 q 比率|
|beam search|一个或多个模型|保留累计分数最高的多个分支|不是同一随机采样合同|
|top-k/top-p|当前分布过滤器|删除候选后重新归一化|改变定义中的 target 分布，需在 p/q 中一致记录|

如果 draft 使用 greedy 提议，$q_i(y_i)$ 不是一个普通的随机 proposal 分布。可以使用确定性 proposal 的专门修正算法，但不能把 greedy token 当作从未记录的 q 分布抽样后直接套接受率公式。

若 target 分布启用了 temperature 或 top-p，目标的 $p_i$ 必须是处理后的分布。Draft 的 $q_i$ 可以采用同样处理，也可以不同，但验证和残差必须使用实际的 $p_i$ 与 $q_i$。

## 运行方法

下面的标准库探针用一个三 token 分布核对接受比率、重叠质量、残差修正和边际分布。它还计算三个验证位置在不同接受率下的每轮期望输出长度。

```python
from math import prod


p = [0.5, 0.3, 0.2]
q = [0.4, 0.4, 0.2]

ratios = [
    min(1.0, target / draft)
    for target, draft in zip(p, q)
]
overlap = sum(
    min(target, draft)
    for target, draft in zip(p, q)
)
residual = [
    max(target - draft, 0.0)
    for target, draft in zip(p, q)
]
residual_mass = sum(residual)
correction = [
    value / residual_mass
    for value in residual
]
accepted_mass = [
    min(target, draft)
    for target, draft in zip(p, q)
]
marginal = [
    accepted + residual_mass * value
    for accepted, value in zip(accepted_mass, correction)
]

print("ratios=", [round(value, 6) for value in ratios])
print("overlap=", round(overlap, 6))
print("residual=", [round(value, 6) for value in residual])
print("correction=", [round(value, 6) for value in correction])
print("marginal=", [round(value, 6) for value in marginal])

acceptance = [0.9, 0.75, 0.6]
expected_emitted = sum(
    prod(acceptance[:index])
    for index in range(len(acceptance) + 1)
)
print("expected_emitted=", round(expected_emitted, 6))
```

运行输出为：

```text
ratios= [1.0, 0.75, 1.0]
overlap= 0.9
residual= [0.1, 0.0, 0.0]
correction= [1.0, 0.0, 0.0]
marginal= [0.5, 0.3, 0.2]
expected_emitted= 2.98
```

这里的 accepted mass 为 $\min(p,q)=(0.4,0.3,0.2)$，拒绝质量为 $0.1$，残差修正全部落在 token 0，因此最终边际分布恢复为 $(0.5,0.3,0.2)$。期望输出长度 $2.98$ 只描述给定三个接受率和“全接受时追加一个目标 token”的理想化一轮，不等同于线上吞吐。

## 失效模式和审计方法

### 直接提交 draft token

Draft model 只是 proposal。即使它通常很准，也不能跳过 target 验证，否则输出分布变成 q 而不是 p。审计要记录每个草稿 token 的 target probability、draft probability、接受随机数和最终提交原因。

### 接受比率使用错误方向

接受概率是 $\min(1,p/q)$。写成 $\min(1,q/p)$ 会优待目标概率更低的 draft token，破坏边际分布。用 $p>q$、$p=q$ 和 $p<q$ 三种单 token 输入分别测试。

### 拒绝后从完整 target 分布采样

拒绝位置不能直接从 $p$ 重新采样并声称使用了标准残差修正。完整 target 分布会重复包含已经通过接受事件分配的质量；标准修正使用 $\max(p-q,0)$ 后归一化的残差。

### 忽略 $q=0$

Draft 不能提出 q 概率为 0 的 token。若概率在低精度下下溢为 0，ratio 和残差实现需要明确支持集规则。记录 q 的最小非零概率和零概率 token 数。

### 首次拒绝后继续接受尾部

$y_i$ 拒绝后，$y_{i+1},\ldots,y_\gamma$ 的条件前缀包含未提交 token，不能继续提交。必须丢弃尾部并从修正 token 的新状态开始下一轮。

### 全接受时丢弃额外目标 token

全接受后若不从 target 的下一位置取额外 token，会少执行一个可提交的目标转移。若产品有意使用不追加的变体，应单独报告算法和吞吐口径。

### 回滚逻辑长度但不回滚 K/V

Rejected K/V 仍在 cache 中时，下一次 target query 会读取错误历史。使用可识别 token 的 prompt 检查逻辑长度、物理 block、block table 和下一个 position 的 attention 输入。

### Draft 与 target 的 tokenizer 不一致

不同 token ID 不能直接做 p/q 比率。即使两个字符串看起来相同，token 边界也可能不同。先固定 tokenizer 合同，或切换到明确处理异构 tokenization 的算法。

### 停止状态包含未提交 token

在草稿尾部上检查 EOS 或 stop sequence，会把 rejected token 当成最终输出。只在 accepted prefix 或 residual correction 提交后更新停止状态。

### 只看接受率

高接受率不能证明 residual sampler、cache rollback 或 batch slot 正确。把一步边际频率、接受质量、残差质量和 target 概率一起核验。

### 一份最小投机解码审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|模型角色|draft 只提议，target 决定最终分布|是否直接提交 draft|
|token 空间|词表、ID、位置和 mask 合同一致|tokenizer、RoPE、padding|
|proposal|保存每个 $q_i(y_i)$ 和 draft cache|概率下溢、历史错位|
|verify|一次 target 前向得到各位置 $p_i$|位置 shift、causal mask|
|接受|使用 $\min(1,p_i/q_i)$|比率方向、log space|
|残差|使用 $\max(p_i-q_i,0)$ 并重新归一化|误用完整 p、质量为 0|
|首次拒绝|拒绝后丢弃后续草稿 token|错误继续提交尾部|
|全接受|追加一个 target token，若采用标准算法|额外 token、EOS|
|cache|accepted prefix、修正 token、物理 block 同步|rollback、block table|
|stop|只对已提交序列更新 EOS 和 stop|未提交草稿尾部|
|RNG|接受、残差和额外采样的消耗顺序固定|seed、scope、rejection path|
|分布|一步输出边际与 target p 一致|只看文本或接受率|
|吞吐|记录 draft/target calls、接受长度和回滚次数|只看单次 latency|

投机解码的核心不是“让小模型替大模型回答”，而是把小模型当作 proposal，再用目标分布的接受—拒绝分解补回它漏掉的概率质量。速度来自一次目标验证处理多个候选位置；正确性来自首次拒绝截断和残差重采样。任何 tokenizer、score、mask、cache、stop 或 RNG 合同缺失，都会把吞吐优化变成另一种生成算法。

## 相关词条

- [推理](../inference/inference/)：固定生成请求、prefill/decode、停止条件和性能账本。
- [KV cache](../inference/kv-cache/)：说明 draft/target 临时 cache、回滚和物理 block 提交。
- [温度采样](../inference/temperature-sampling/)：说明目标分布经过 temperature 后如何抽样。
- [Top-k 与 Top-p](../inference/top-k-top-p/)：说明候选过滤如何改变 target 与 draft 的分布合同。
- [贪心解码](../inference/greedy-decoding/)：区分确定性 proposal 与随机 proposal。
- [束搜索](../inference/beam-search/)：区分累计分数搜索与边际分布保持的投机采样。
- [Decoder-Only](../transformer-architectures/decoder-only/)：说明自回归 next-token logits 和 position/cache 合同。
