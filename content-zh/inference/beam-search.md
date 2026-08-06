---
title: "束搜索：保留多个候选以近似序列最优"
tags: ["why-models-learn"]
---

束搜索在每个生成位置保留固定数量的候选序列，用累计 log probability 对候选扩展、排序和剪枝。束宽为 1 时它退化为贪心解码；束宽大于 1 时，它延迟部分局部选择，并同步重排每个候选的 KV cache。它仍然是有限宽度的近似搜索，不能保证找到所有可能序列中的全局最优。本篇固定候选分数、扩展与 top-k、长度归一化、EOS、cache 重排、batch 以及停止判据。

![束搜索示意图：一个前缀扩展为多个候选，按累计 log probability 保留两个分支，并用父索引重排对应的 KV cache](/assets/inference/svg/beam-search.1.svg)

## 先把束搜索的对象分开

### 每条 beam 是一条条件序列

设输入上下文为 $x$，当前生成长度为 $t$，词表大小为 $V$，束宽为 $K$。第 $k$ 条候选的序列和累计分数记为：

$$
y_{1:t}^{(k)},
\qquad
S_t^{(k)}.
$$

模型对第 $k$ 条候选的下一个 token 输出 logits：

$$
z_t^{(k)}
=
f_\theta\left(x,y_{<t}^{(k)}\right)
\in\mathbb R^V.
$$

束搜索先把 logits 转成 log probability：

$$
\ell_t^{(k)}(v)
=
\log p_\theta\left(v\mid x,y_{<t}^{(k)}\right).
$$

扩展 token $v$ 后，新候选的累计分数为：

$$
S_{t+1}^{(k,v)}
=
S_t^{(k)}
+
\ell_t^{(k)}(v).
$$

每次扩展有 $K V$ 个候选。算法只保留其中分数最高的 $K$ 个活跃或完成候选，其他分支不再进入后续前向。

### 为什么使用 log probability

自回归序列概率是条件概率的乘积：

$$
P(y_{1:T}\mid x)
=
\prod_{t=1}^{T}
p_\theta\left(y_t\mid x,y_{<t}\right).
$$

直接相乘会随长度快速下溢。取对数后：

$$
S_T
=
\log P(y_{1:T}\mid x)
=
\sum_{t=1}^{T}
\ell_t(y_t).
$$

对数把乘法变成加法，便于候选扩展和排序。由于 $\log$ 是单调函数，同一长度的候选按概率排序与按累计 log probability 排序相同。不同长度的候选还要处理长度偏置，不能只比较裸的 $S_T$。

### beam width=1 与 greedy 的关系

初始只有一个活跃候选，假设所有候选使用相同的当前长度、相同的 processor 和相同的 tie-breaking。当 $K=1$ 时，上一时刻的累计分数对当前所有 token 是同一个常数，因此：

$$
\operatorname*{arg\,max}_v
\left(
S_t+\ell_t(v)
\right)
=
\operatorname*{arg\,max}_v
\ell_t(v).
$$

所以 $K=1$ 的束搜索与 [贪心解码](../inference/greedy-decoding/)选择相同的 token。差异从 $K>1$ 开始：束搜索保留多个 parent，下一步可以从当前分数较低但后续概率较高的分支继续扩展。

## 一次扩展如何产生下一组 beam

### 先对每个 beam 计算 log-softmax

对一个活跃候选，稳定 log-softmax 可以写成：

$$
\ell_t(v)
=
z_t(v)
-
\operatorname{logsumexp}_{j}
z_t(j).
$$

令当前活跃 beam 数为 $K_{\mathrm{alive}}$。将所有候选的扩展分数排列成矩阵：

$$
C_t^{(k,v)}
=
S_t^{(k)}
+
\ell_t^{(k)}(v),
\qquad
C_t\in\mathbb R^{K_{\mathrm{alive}}\times V}.
$$

然后把矩阵展平为 $K_{\mathrm{alive}}V$ 个候选，选择分数最高的 $K$ 个。候选编号 $r$ 与 parent 和 token 的映射为：

$$
\operatorname{parent}(r)
=
\left\lfloor\frac{r}{V}\right\rfloor,
\qquad
\operatorname{token}(r)
=
r\bmod V.
$$

新的序列先复制 parent 的 token 前缀，再追加 token(r)。新的累计分数就是对应的 $C_t^{(k,v)}$。

### 一个两步、束宽为 2 的例子

第一步的概率为：

|token|第一步概率|第一步 log probability|
|---|---:|---:|
|A|0.60|$\log 0.60$|
|B|0.40|$\log 0.40$|
|C|0.05|$\log 0.05$|

若 $K=2$，第一步保留 A 和 B。第二步的条件概率设置为：

|parent|A|B|C|
|---|---:|---:|---:|
|A|0.50|0.30|0.20|
|B|0.90|0.05|0.05|

第二步的四个主要扩展概率为：

- A→A：$0.60\times0.50=0.30$；
- A→B：$0.60\times0.30=0.18$；
- B→A：$0.40\times0.90=0.36$；
- B→B：$0.40\times0.05=0.02$。

束宽为 2 时保留 B→A 和 A→A。贪心在第一步选择 A，只能继续得到 A→A；束搜索保留了 B，因而访问了累计概率更高的 B→A 分支。它没有访问 C，因为 C 在第一步已经被剪枝。

### top-k 是对展平后的候选做

束搜索每一步的 top-k 不是对每个 parent 各取一个 token，再从这些 token 中随便选。正确流程是：

1. 对每条活跃 beam 计算完整词表 log probability；
2. 加上该 beam 的累计分数；
3. 将 parent 与 token 的所有组合放到同一个候选集合；
4. 对组合分数做全局 top-k；
5. 从 top-k 索引恢复 parent 和 token；
6. 同步复制或重排序列状态。

如果每个 parent 只保留一个 token，算法就接近对每条 beam 单独 greedy，可能丢掉一个 parent 的第二名候选，而它与其他 parent 的第一名候选相比本来更高。

## 剪枝让束搜索成为近似算法

### 完整搜索的分支数呈指数增长

若每一步都有 $V$ 个候选，长度为 $T$ 的完整搜索树最多有：

$$
V^T
$$

条序列。束搜索每步只保留 $K$ 条活跃序列，下一步最多扩展为 $KV$ 个候选，再剪回 $K$ 条。忽略模型前向后，候选排序的规模从指数树变为每步的有限候选集合。

束宽越大，保留的历史分支越多，搜索覆盖更广；代价是：

- 同一位置需要计算更多候选 beam 的 logits；
- KV cache 的有效 batch 通常乘以 $K$；
- top-k、parent index 和 cache reorder 的工作增加；
- 完成 beam 与活跃 beam 的管理更复杂；
- 物理显存和带宽预算上升。

### 被剪枝的分支不会重新出现

假设某个 parent 在第 $t$ 步的候选没有进入 top-k，它的 cache、累计分数和 token 历史就不再参与第 $t+1$ 步。即使它在未来有更高的条件概率，束搜索也不会发现这条路径。

因此，束搜索的结果只能表述为“在当前束宽、分数和停止协议下选出的候选”。不能把 beam output 称为全局序列最优，除非使用了保留全部分支的穷举算法或有可证明的上界剪枝。

### beam width 的边界

|束宽|行为|主要资源|
|---:|---|---|
|1|退化为 greedy|一份 active cache|
|2 或更大|保留多个历史分支|cache、logits 和排序近似乘束宽|
|接近词表规模|第一步覆盖更多候选，后续仍需剪枝|top-k 与显存增加|
|保留全部树|穷举搜索|随生成长度指数增长|

束宽变化本身也会改变输出。比较两个模型时，必须固定 beam width、长度分数、EOS 处理、约束和 tokenizer；否则输出差异不能归因于 checkpoint。

## 长度分数决定候选排序的方向

### 裸累计 log probability 偏好短序列

每个 token 的 log probability 通常不大于 0。序列越长，继续相加后裸分数通常越低。若一条候选已经产生 EOS，另一条候选还在继续生成，直接比较两者的 $S_T$ 会倾向短序列。

一种简单的长度归一化是：

$$
S_{\mathrm{norm}}(y_{1:T})
=
\frac{S_T}{T^\alpha},
\qquad
\alpha\ge0.
$$

$\alpha=0$ 时恢复裸累计分数；$\alpha>0$ 时减轻长度带来的分数下降。实际实现也可能使用包含常数偏移的长度惩罚：

$$
S_{\mathrm{lp}}(y_{1:T})
=
\frac{S_T}
{\left(\frac{c+T}{c+1}\right)^\alpha},
\qquad
c>0.
$$

公式不同会产生不同排序。报告中要写出具体公式和 $\alpha$，不能只写“开启 length penalty”。

### 长度归一化会改变剪枝时机

如果使用最终长度归一化，当前活跃候选的未来长度尚未确定。实现可能：

- 用当前长度的归一化分数剪枝；
- 用未归一化累计分数剪枝，结束后再归一化；
- 对活跃和完成候选分别使用上界；
- 使用某种 length penalty 的增量近似。

这些选择不等价。尤其在一个短候选已经结束、另一个长候选仍活跃时，停止条件必须说明比较的是哪一种分数。

### PPL、平均 log probability 和 beam score 不是一项指标

困惑度评估使用真实 target token 的 token-level log probability；beam score 是搜索过程中模型对候选自身 token 的累计分数。长度归一化后的 beam score 也不等于 PPL，因为：

- 候选 token 来自模型搜索，不是固定 reference；
- 生成长度由 EOS 和停止规则决定；
- 归一化指数可能不同；
- beam 只观察被保留的候选。

[困惑度评估](../pretraining/evaluation-perplexity/)应继续使用独立的 target mask 和 token 统计，不要把 beam 的最终分数当作评估困惑度。

## EOS、完成候选与停止条件

### active beam 和 finished beam 分开

束搜索至少需要两组状态：

|状态|含义|是否继续模型前向|
|---|---|---|
|active beam|尚未触发 EOS 或 stop|是|
|finished beam|已经达到结束条件|否|
|invalid candidate|被 mask、NaN 或约束拒绝|否|
|cancelled request|被外部取消或超时|否|

EOS 候选可以进入 finished 集合，不应继续把新的 token 接在 EOS 后面。实现可以保留多个 finished beam，再按最终长度分数选出返回结果。

### “所有 beam 都结束”不是唯一停止规则

常见停止条件包括：

1. active beam 数变为 0；
2. 已有足够多的 finished beam，活跃 beam 的理论上界无法超过最优 finished；
3. 达到 max new tokens；
4. 达到 context limit；
5. stop sequence 或格式约束完成；
6. 请求取消或超时。

第 2 条需要一个可靠的上界。若实现没有证明使用的上界，提前停止只是启发式，不能称为已经找到全局最优。工程日志要记录停止原因、active 数、finished 数和停止时各自的 score。

### EOS 是否计入长度

EOS 计数规则会影响长度归一化。至少有三种口径：

|口径|EOS 是否计入|影响|
|---|---|---|
|model steps|通常计入|统计实际前向和 cache 写入|
|generated content|通常不计入|统计返回内容长度|
|score length|按 length penalty 协议|影响 beam 排序|

如果某个实现把 EOS 从返回文本删除，却在分数分母中排除 EOS，另一个实现包含 EOS，两者的 beam score 不能直接比较。

### stop sequence 可能发生在 token 边界之外

stop sequence 可以由字符串后处理匹配，也可以由 token ID 状态机匹配。UTF-8、空格合并和特殊 token 处理可能让同一文本在两种规则下有不同的停止位置。beam 内部应保留原始 token 序列和停止原因，最后再决定是否从返回文本中移除 stop sequence。

## KV cache 必须跟着 parent 一起重排

### beam 维通常并入 batch

对 batch size 为 $B$、束宽为 $K$ 的 decoder-only 模型，target self-attention 的 cache 逻辑第一轴通常是 $BK$：

$$
K_{\ell}^{\mathrm{cache}}
\in
\mathbb R^{(BK)\times h_{kv}\times L\times d_h},
\qquad
V_{\ell}^{\mathrm{cache}}
\in
\mathbb R^{(BK)\times h_{kv}\times L\times d_h}.
$$

每个 decode step 选择新的 parent index。若第 $r$ 个新 beam 来源于旧的第 $q$ 个 beam：

$$
\begin{aligned}
K_{\ell}^{\mathrm{next}}[r]&=
K_{\ell}^{\mathrm{old}}[q],\\
V_{\ell}^{\mathrm{next}}[r]&=
V_{\ell}^{\mathrm{old}}[q].
\end{aligned}
$$

这项操作要对每一层、K 和 V 同时执行。当前 token、累计 score、sequence ID、position offset 和 cache slot 也必须使用同一个 parent permutation。

### 只重排 logits 会产生跨分支历史

一个可运行但错误的实现可能完成以下步骤：

1. 计算所有 beam 的 logits；
2. 选择 top-k 的 parent 和 token；
3. 更新 token 序列和累计 score；
4. 保留原 cache 的物理顺序。

下一步模型会使用新 token 和旧 parent 的不匹配状态。输出仍然有正确 shape，错误只在下一步的条件上下文中出现。审计要用两个具有不同历史标记的 beam，强制发生 parent 交换，再检查每层 K/V 的可识别值是否一起交换。

### cache 字节数随 beam 增长

由 [KV cache](../inference/kv-cache/)的公式，固定长度和 dtype 时：

$$
S_{\mathrm{KV,beam}}
=
2L_{\mathrm{layer}}BKT h_{kv}d_hb.
$$

如果 prefix 物理 block 可以共享，实际分配量可能小于立即复制 $K$ 份的上界；一旦某个 beam 追加不同 token，copy-on-write 会为分歧位置分配新的 block。报告应分别写逻辑 beam bytes、共享 block bytes、已分配 physical bytes 和引用计数。

### beam reorder 与 paged cache

分页布局不一定需要搬移 K/V 内容。可以只重排每个 beam 的 block table 和 sequence metadata：

|布局|parent 变化时的主要操作|风险|
|---|---|---|
|contiguous|gather 每层 K/V 或交换 batch slot|大块内存复制|
|paged|重排 block table、length 和 sequence ID|旧 table 或引用计数残留|
|prefix shared|保留共享只读 block，分歧时复制|写入覆盖共享块|
|quantized paged|同步 K/V block 与 scale block|scale 对错物理块|

无论采用哪种布局，逻辑 token 顺序和物理寻址都必须保持一致。

## encoder-decoder 的 source cache 可以共享

### target self cache 需要分支，source cache 通常不需要

encoder-decoder 模型中，source encoder 输出对同一输入通常相同。decoder 的每个 beam 有自己的 target prefix，因此 target self-attention cache 需要按 beam 分支；cross-attention 的 static source K/V 可以在 beam 间共享，或者只复制引用：

$$
\begin{aligned}
K_{\ell}^{\mathrm{src}}
&\in
\mathbb R^{B\times h_{kv}\times S\times d_h},\\
K_{\ell}^{\mathrm{target,cache}}
&\in
\mathbb R^{(BK)\times h_{kv}\times T\times d_h}.
\end{aligned}
$$

source cache 共享减少重复 source 投影和显存；它不能让不同 source 输入共享同一 source cache。source 的 checkpoint、tokenizer、encoder mask 和 adapter 必须相同。[Encoder-Decoder Transformer](../transformer-architectures/encoder-decoder/)固定了 source length、target length 和 cross-attention 的轴合同。

### source 与 target 的 reorder 规则不同

beam reorder 时：

- target self cache 按 parent beam 重排；
- static source cache 可以保持 batch 级引用；
- 若实现复制 source cache 到 $BK$，复制必须与 beam 映射一致；
- source padding mask 不应被 target beam 的长度替换；
- target position offset 只沿 target 序列推进。

把 source K/V 误当作 target 历史，或者把 source 的固定长度 $S$ 乘到 target 的 beam length 中，会得到显存和 mask 都错误的结果。

## 约束束搜索和多样性边界

### 约束可以在每个 beam 上分别应用

对第 $k$ 条 beam，允许集合可以依赖该 beam 的历史：

$$
\mathcal V_t^{(k)}
=
G\left(y_{<t}^{(k)},\text{constraint state}\right).
$$

屏蔽后的候选分数为：

$$
C_t^{(k,v)}
=
\begin{cases}
S_t^{(k)}+\ell_t^{(k)}(v),&v\in\mathcal V_t^{(k)},\\
-\infty,&v\notin\mathcal V_t^{(k)}.
\end{cases}
$$

如果不同 beam 处在不同的语法状态，不能共用一个词表 mask。约束状态也要随着 parent index 一起重排。

### diverse beam 不是普通 beam 的参数别名

普通 beam 让所有候选竞争同一个累计分数。多样性束搜索可能对相同 token、相同 group 或相似前缀加入惩罚，再进行剪枝。它改变了候选分数，不再等于简单的 top-k cumulative log probability。报告必须写出 diversity penalty、group 数、每组宽度和排序顺序。

### beam 与 sampling 可以组合，但语义改变

beam search 可以和 temperature、top-k 或 top-p 组合，但此时候选集合不是完整词表，分数也不再是原始模型的完整条件概率。若先采样再做 beam，随机性进入 parent 分支；若先 beam 再采样，beam 的排序和分支定义不同。[温度采样](../inference/temperature-sampling/)与后续 [Top-k 与 Top-p](../inference/top-k-top-p/)词条分别固定采样路径；本文的数值只针对确定性 beam。

## 确定性和数值边界

### 固定 tie-breaking

beam 每一步要对 $KV$ 个展平候选排序。分数相同时，需要固定：

- parent beam 顺序；
- token ID 顺序；
- 词表排序；
- finished 与 active 的优先级；
- length penalty 的计算时机。

可以把候选的排序键写为：

$$
\left(
-\operatorname{score},
\operatorname{parent},
\operatorname{token}
\right).
$$

这里的负号表示分数降序。具体键可以不同，但必须在 CPU、GPU 和离线复现脚本中一致。

### log-softmax 与 top-k 的数值顺序

应先使用稳定 log-softmax 或等价的 log-sum-exp，再加累计 score。对每条 beam 先减去最大 logit，可以避免指数溢出：

$$
\operatorname{logsumexp}(z)
=
m+\log\sum_j\exp(z_j-m),
\qquad
m=\max_j z_j.
$$

如果只在 top-k 后计算 log probability，则要确认 top-k 的排序基于什么分数；先截断再归一化与对完整词表归一化后再选择，概率和累计 score 可能不同。

### 重复运行的差异需要按层定位

beam 的最终文本差异可能来自：

|层级|检查对象|
|---|---|
|输入|tokenizer、模板、source mask|
|模型|checkpoint、adapter、量化权重|
|执行|prefill、KV cache、position offset|
|分数|log-softmax、length penalty、constraint|
|排序|top-k、parent/token tie-breaking|
|状态|finished beam、block table、cache reorder|
|后处理|EOS、stop sequence、文本解码|

只比较第一条返回文本，无法知道差异来自 score、parent 还是 cache。至少要记录每步保留 beam 的 token 序列、累计 score、parent index、token ID 和停止状态。

## 运行方法

下面的标准库探针用固定的两步条件概率运行束宽为 2 的搜索。它同时核对累计 log probability、top-k 剪枝、parent index 和 cache 标记的重排。

```python
from math import exp, log


def expand(beams, next_probabilities, width):
    candidates = []
    for parent_index, (tokens, score) in enumerate(beams):
        for token, probability in next_probabilities[parent_index].items():
            candidates.append(
                (
                    score + log(probability),
                    parent_index,
                    tokens + (token,),
                )
            )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = candidates[:width]
    return [(tokens, score) for score, _, tokens in selected], selected


beam_width = 2
first_probabilities = [
    {"A": 0.60, "B": 0.40, "C": 0.05},
]
beams = [((), 0.0)]
beams, first_selected = expand(
    beams,
    first_probabilities,
    beam_width,
)
print(
    "step=1",
    [(tokens, round(score, 6)) for tokens, score in beams],
)

second_probabilities = [
    {"A": 0.50, "B": 0.30, "C": 0.20},
    {"A": 0.90, "B": 0.05, "C": 0.05},
]
beams, second_selected = expand(
    beams,
    second_probabilities,
    beam_width,
)
print(
    "step=2",
    [(tokens, round(score, 6)) for tokens, score in beams],
)
best_tokens, best_score = beams[0]
print("best=", best_tokens, "probability=", round(exp(best_score), 2))

old_cache = ["cache-A", "cache-B"]
parent_indices = [item[1] for item in second_selected]
print("parent_indices=", parent_indices)
print("cache_reorder=", [old_cache[index] for index in parent_indices])
```

运行输出为：

```text
step=1 [(('A',), -0.510826), (('B',), -0.916291)]
step=2 [(('B', 'A'), -1.021651), (('A', 'A'), -1.203973)]
best= ('B', 'A') probability= 0.36
parent_indices= [1, 0]
cache_reorder= ['cache-B', 'cache-A']
```

第一步保留 A 和 B。第二步把两个 parent 的全部扩展放在同一个候选集合中，B→A 的累计概率 $0.36$ 高于 A→A 的 $0.30$，因此它排在第一位。parent index 从 $[0,1]$ 变成 $[1,0]$，每层的 cache 也必须按这个顺序重排。

## 失效模式和审计方法

### 每个 parent 各取一个 token

这样会丢掉某个 parent 的第二名候选。正确的 top-k 对象是所有 parent-token 组合的累计分数，不是每个 parent 的局部最大值。用一个 parent 的第二名分数高于另一个 parent 的第一名分数的例子测试实现。

### 把 logits 相加而不是 log probability 相加

不同 beam 的 logits 没有共同的归一化常数，不能直接比较跨 beam 的原始 logits。先对每条 beam 做稳定 log-softmax，再加已有累计 log probability。若使用特殊分数，必须明确它不再是模型序列概率。

### 只重排 token，不重排 KV cache

输出 shape 仍然正确，下一步的上下文却来自错误 parent。用两个具有不同首 token 的 beam，强制 parent permutation 为非恒等排列，再逐层检查 K/V。

### 只重排 K，不重排 V

K 决定 score，V 决定 context。只交换一份会得到“看似正确的注意力权重、错误的读取内容”。K、V、累计分数、token 序列和 cache slot 使用同一个 parent index。

### 过早丢弃 finished beam

已完成候选不能继续追加 token，但也不能在还没有比较最终分数前随意删除。维护 active 和 finished 两组，记录每组的长度、score、停止原因和排序规则。

### 长度归一化时机不一致

有的实现用裸 score 剪枝，结束后归一化；有的实现每步使用 normalized score；有的实现使用上界提前停止。不同算法的输出不能仅以“都开了 length penalty”描述。

### 把 EOS 当作普通 token 继续扩展

EOS 之后继续计算下一步，会让返回序列、cache 长度和 score length 不一致。选择 EOS 后立即转入 finished 状态，除非协议明确规定它是普通控制 token。

### beam width 变化却复用旧 cache 形状

$K$ 从 1 改为 4 后，有效 batch、cache slot、block table 和 logits shape 都要改变。只修改 top-k 数量而不扩展或共享 cache，会产生索引越界或跨请求读取。

### source cache 和 target cache 混用

encoder-decoder 的 source K/V 可以按输入共享，target self K/V 必须按 beam 分支。把两者使用同一 parent gather 会误改 source 的 batch 映射或 target 的 position offset。

### 把 beam output 当作全局最优

有限束宽会剪枝。只有完整搜索或有可靠上界的搜索才能提供全局最优保证。报告中写“束宽为 K 的近似候选”，并给出被剪枝的候选范围和停止条件。

### 一份最小 beam 审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|beam width|$K$、active 数和有效 batch 固定|配置、slot 分配|
|分数|每步用稳定 log probability 加累计 score|log-softmax、归一化常数|
|扩展|候选数为 active beams × vocabulary|局部 top-1、mask|
|剪枝|对 parent-token 组合做全局 top-k|flatten、排序键|
|parent index|从候选编号正确恢复 parent 和 token|整除、取模、batch offset|
|tie-breaking|score 相同的候选顺序固定|parent/token ID、设备归约|
|长度分数|公式、指数和剪枝时机固定|EOS、active/finished 比较|
|EOS|finished 不再扩展且计数口径固定|停止原因、返回 token|
|KV cache|每层 K/V 按 parent permutation 同步重排|gather、cache slot|
|paged cache|block table、length、引用计数同步|copy-on-write、free list|
|source cache|相同 source 可共享，target cache 按 beam 分支|encoder/decoder 轴|
|确定性|输入、dtype、kernel、排序和后处理固定|近似 tie、量化|
|报告|保留每步 beam、score、parent 和停止状态|只保存最终文本|

束搜索的核心不是“多取几个 token”，而是维护多条带有历史分数和运行时状态的候选序列。每个候选的 token、累计 score、约束状态、position offset 和 KV cache 都必须由同一个 parent index 推进；任何一项不同步，下一步就不再代表同一条序列。

## 相关词条

- [推理](../inference/inference/)：固定生成请求、停止条件、batch 和性能协议。
- [贪心解码](../inference/greedy-decoding/)：说明束宽为 1 时的逐步 argmax 基线。
- [KV cache](../inference/kv-cache/)：说明 beam 维度、parent reorder 和分页 cache 的运行时状态。
- [因果语言建模](../transformer-architectures/causal-language-modeling/)：推导自回归条件概率与 next-token 目标。
- [Encoder-Decoder Transformer](../transformer-architectures/encoder-decoder/)：区分可共享的 source cache 与按 beam 分支的 target cache。
- [困惑度评估](../pretraining/evaluation-perplexity/)：区分 teacher-forcing 指标与生成候选分数。
- [温度采样](../inference/temperature-sampling/)：说明随机采样与确定性 beam 的组合边界。
- [Top-k 与 Top-p](../inference/top-k-top-p/)：说明候选截断如何改变可扩展的 token 集合。
