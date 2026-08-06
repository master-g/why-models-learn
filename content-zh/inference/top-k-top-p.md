---
title: "Top-k 与 Top-p：截断候选分布后再采样"
tags: ["why-models-learn"]
---

Top-k 与 Top-p 都先从 next-token 分布中删除一部分候选，再从剩余候选重新归一化后的分布中采样。Top-k 固定最多保留 $k$ 个 token；Top-p 又称 nucleus sampling，按概率从高到低累加，保留第一个使累计概率达到阈值 $\tau$ 的 token。两者改变候选集合和采样概率，不直接改变模型已经计算出的 logits。[温度采样](../inference/temperature-sampling/)定义正温度、稳定 softmax 和 CDF 抽样；本篇继续固定候选截断的边界、顺序、归一化和运行时状态。

![Top-k 与 Top-p 示意图：同一组概率经过固定数量截断和累计概率截断后形成不同候选集合](/assets/inference/svg/top-k-top-p.1.svg)

## 先固定截断前的分布

### logits、温度和概率

设当前请求的 processed logits 为：

$$
z=(z_0,\ldots,z_{V-1})\in\mathbb R^V.
$$

如果启用正温度 $T$，先计算：

$$
a_v=\frac{z_v}{T}.
$$

稳定 softmax 使用：

$$
m=\max_j a_j,
\qquad
p(v)=\frac{\exp(a_v-m)}
{\sum_{j=0}^{V-1}\exp(a_j-m)}.
$$

这里的 $p(v)$ 是截断前的完整词表概率。截断器不能把 raw logits 当成概率直接相加；它需要明确输入是 logits、log probability 还是已经归一化的 probability。[Softmax](../neurons-and-activations/softmax/)给出了概率单纯形和 log-sum-exp 的完整推导。

正温度保持 logits 排序：

$$
z_a>z_b
\Longrightarrow
\frac{z_a}{T}>\frac{z_b}{T}
\Longrightarrow
p(a)>p(b).
$$

因此正温度下 top-k 的排序通常不变，但 top-p 的累计概率会改变。温度越低，前几个 token 承担的概率质量越多，达到同一 $\tau$ 所需的候选数量通常越少。

### 截断前必须明确 mask

如果语法约束、bad words、未满足的格式状态或最小生成长度禁止某些 token，应先构造允许集合 $\mathcal A$：

$$
z^{\mathrm{mask}}_v
=
\begin{cases}
z_v,&v\in\mathcal A,\\
-\infty,&v\notin\mathcal A.
\end{cases}
$$

然后在有效候选上重新计算温度和 softmax：

$$
p_{\mathcal A}(v)
=
\operatorname{softmax}
\left(
z^{\mathrm{mask}}/T
\right)_v.
$$

如果先在完整词表上得到概率，再删除禁止 token 而不重新归一化，剩余质量小于 1。这个向量不能直接交给 categorical sampler。若 $\mathcal A$ 为空，服务应报告约束失败或执行明确的回退策略，不能让截断器静默选择 token 0。

### 处理顺序是算法的一部分

一次可复核的生成请求至少要记录以下路径：

$$
\text{raw logits}
\rightarrow
\text{logit processors}
\rightarrow
\text{constraint mask}
\rightarrow
\text{temperature}
\rightarrow
\text{top-k/top-p}
\rightarrow
\text{renormalization}
\rightarrow
\text{categorical sample}.
$$

实际实现可以采用不同顺序，但配置和日志必须说明顺序。repetition penalty、temperature、top-k、top-p 和 mask 含有非线性或集合操作，不能仅凭参数名推断它们是否可交换。

## Top-k：固定候选数量

### Top-k 先按分数排序

给定 $k$，令 $\operatorname{rank}(v)$ 表示按分数从高到低排序后的位置。Top-k 候选集合为：

$$
\mathcal C_k
=
\left\{
v:\operatorname{rank}(v)<k
\right\}.
$$

实际实现通常把 $k$ 约束在：

$$
1\le k\le V.
$$

当配置为 $k=1$ 时，候选集合只含当前最高分 token，重新归一化后的概率为 1，结果退化为 greedy 的单步选择。当 $k\ge V$ 时，top-k 不删除候选，结果等同于只做温度和 softmax。

Top-k 的候选数由配置决定，不随当前分布的尾部形状变化。一个尖锐分布和一个平坦分布都可以保留相同的 $k$ 个 token。

### 截断后必须重新归一化

定义 top-k 的候选质量：

$$
M_k=\sum_{j\in\mathcal C_k}p(j).
$$

采样分布不是把原概率向量截断后原样使用，而是：

$$
q_k(v)
=
\frac{p(v)\mathbf 1[v\in\mathcal C_k]}
{M_k}.
$$

因此：

$$
\sum_{v=0}^{V-1}q_k(v)=1.
$$

一个 token 在截断前的概率为 $0.2$，并不意味着它在 top-k 后仍然以 $0.2$ 的概率出现。它的概率要除以剩余候选质量 $M_k$。如果 top-k 保留的候选原本只占总质量的 $0.4$，每个保留 token 的相对概率会放大为原来的 $2.5$ 倍。

### tie-breaking 需要固定

多个 token 的 logits 相同时，Top-k 在边界处可能有多个合法集合。实现必须固定平局规则，例如：

- 先按 logit 降序，再按 token ID 升序；
- 使用稳定排序，保留原词表顺序；
- 使用硬件 top-k kernel 的确定性 tie-breaking；
- 记录边界处相同分数的 token 数量。

如果第 $k$ 名与第 $k+1$ 名相同，而不同设备采用不同规则，候选集合就会改变。即使后续 RNG、温度和 seed 完全相同，输出序列也会不同。

### Top-k 的边界配置

|配置|推荐语义|需要记录的边界|
|---|---|---|
|$k=1$|保留一个候选，等同于 greedy|是否仍调用 sampler、是否消耗 RNG|
|$1<k<V$|保留固定数量候选|第 $k$ 名 tie-breaking、排序 dtype|
|$k=V$|不做候选删除|是否仍执行一次 gather 和 renormalization|
|$k>V$|截断到词表大小，或拒绝配置|配置校验行为|
|$k\le0$|拒绝配置，或使用明确默认值|不能生成空候选集合|

不能把 $k=0$ 自动解释为“不做 top-k”，除非接口明确规定该语义。不同服务对 $k=0$ 的解释可能分别代表禁用、保留空集或回退到 greedy。

## Top-p：按累计概率保留自适应前缀

### 先排序再累计

Top-p 使用阈值 $\tau$，其中：

$$
0<\tau\le1.
$$

把 token 按截断前概率降序排列为：

$$
p(v_1)\ge p(v_2)\ge\cdots\ge p(v_V).
$$

令最小前缀长度为：

$$
m(\tau)
=
\min
\left\{
m:
\sum_{i=1}^{m}p(v_i)\ge\tau
\right\}.
$$

Top-p 候选集合是：

$$
\mathcal C_\tau
=
\{v_1,\ldots,v_{m(\tau)}\}.
$$

它保留刚好足以覆盖阈值的最小前缀。最后加入的 token 可能只贡献很小的概率，但只要前一个前缀还未达到 $\tau$，它就必须加入。

### crossing token 不能删除

设排序后的概率为：

$$
p=(0.50,0.30,0.15,0.05).
$$

当 $\tau=0.70$ 时，前两个 token 的累计概率为 $0.80$，因此候选集合为前两个 token。当 $\tau=0.65$ 时，第一个 token 的累计概率只有 $0.50$，第二个 token 仍然必须加入，候选集合仍为前两个 token。

如果实现先找到累计概率超过阈值的位置，再把 crossing token 删除，候选质量会低于阈值，而且算法与 top-p 的定义不一致。审计时应保存排序后的概率、累计和、阈值以及最终包含的 crossing token。

### Top-p 的候选数量随分布变化

Top-p 的候选数量不是固定常数。若一个 token 的概率已经接近 1，较小的候选集合就能达到 $\tau$；若概率接近均匀，则需要更多 token。

|分布形状|相同 $\tau$ 下的候选数量|截断结果|
|---|---:|---|
|高分 token 集中|较少|尾部删除更彻底|
|概率质量平滑衰减|中等|候选数随局部 logits 变化|
|接近均匀|较多|可能接近整个词表|
|所有 logits 相同|约 $\lceil\tau V\rceil$|温度不再改变候选比例|

Top-p 直接依赖概率质量，因此与温度、mask 和其他 logit processor 的顺序相关。它不像 top-k 那样只依赖排序。

### Top-p 的边界配置

|配置|推荐语义|需要记录的边界|
|---|---|---|
|$\tau$ 接近 0|至少保留一个最高分 token|是否设置 min tokens to keep|
|$0<\tau<1$|保留累计质量达到阈值的最小前缀|crossing token 是否包含|
|$\tau=1$|保留完整词表，受浮点累计影响|是否强制包含最后一个 token|
|$\tau\le0$|拒绝配置或回退到明确默认值|不能产生空集合|
|$\tau>1$|拒绝配置|不能等待永远不会达到的阈值|

在有限精度下，累计概率可能略小于 1。实现可以在遍历到最后一个 token 时强制加入该 token，但应把这作为边界协议记录，而不是让不同 kernel 自行决定。

## 截断后的采样分布

### 用候选质量重新归一化

Top-p 与 top-k 都可以写成同一个形式。给定候选集合 $\mathcal C$，定义：

$$
M_{\mathcal C}
=
\sum_{j\in\mathcal C}p(j).
$$

截断后的分布为：

$$
q_{\mathcal C}(v)
=
\frac{p(v)\mathbf 1[v\in\mathcal C]}
{M_{\mathcal C}}.
$$

然后使用：

$$
y_t\sim\operatorname{Categorical}(q_{\mathcal C}).
$$

采样器的 CDF 只应在候选 token 上累计。若候选 token 的原始顺序与排序顺序不同，系统必须记录 token ID 到 CDF 数组位置的映射。不能把排序后的概率写回词表时错位。

### 一个固定 logits 的数值账本

取：

$$
z=(2,1,0,-1),
\qquad
T=1.
$$

稳定 softmax 得到：

|token|原始概率|累计概率，按概率降序|top-k=2 后概率|
|---:|---:|---:|---:|
|0|0.643914|0.643914|0.731059|
|1|0.236883|0.880797|0.268941|
|2|0.087144|0.967941|不保留|
|3|0.032059|1.000000|不保留|

Top-k=2 的候选质量为：

$$
M_2=0.643914+0.236883=0.880797.
$$

因此 token 0 的新概率约为：

$$
q_2(0)
=
\frac{0.643914}{0.880797}
\approx0.731059.
$$

对同一分布应用不同 top-p 阈值：

|阈值 $\tau$|累计达到阈值的最小前缀|候选 token|候选原始质量|
|---:|---:|---|---:|
|0.60|1|0|0.643914|
|0.70|2|0、1|0.880797|
|0.90|3|0、1、2|0.967941|
|1.00|4|0、1、2、3|1.000000|

$ \tau=0.90 $ 时 token 2 是 crossing token，不能因为它的单独概率较小而删除。截断后的实际采样概率仍要除以候选原始质量。

## 多个过滤器的顺序

### Temperature 与 Top-k

在正温度、没有其他 processor 且排序没有数值变化时：

$$
\operatorname{TopK}(z)
=
\operatorname{TopK}(z/T).
$$

因此先温度再 top-k 与先 top-k 再温度通常得到相同候选集合。但两条路径仍可能产生不同实现结果：

- temperature 前后使用了不同精度；
- top-k 在量化 logits 上执行；
- tie-breaking 依赖中间张量的排序；
- top-k 后重新计算概率的归一化范围不同；
- temperature 只应用于候选，而不是完整词表。

工程上应固定一个顺序，并在日志中同时记录 top-k 边界 logit、候选 ID 和重归一化后的概率。

### Temperature 与 Top-p

Top-p 依赖累计概率。温度改变概率比例：

$$
\frac{p_T(a)}{p_T(b)}
=
\exp\left(\frac{z_a-z_b}{T}\right).
$$

因此相同 $\tau$ 下，温度改变 $m(\tau)$ 和候选集合。低温度通常减少候选数量，高温度通常增加候选数量；如果 logits 中存在相同分数、mask 或数值下溢，实际边界还需要以运行时结果为准。

常见路径是先温度再 top-p：

$$
z
\rightarrow
z/T
\rightarrow
\operatorname{softmax}
\rightarrow
\operatorname{TopP}_\tau
\rightarrow
q_\tau.
$$

如果实现先对未缩放 logits 计算 top-p，再对保留候选缩放温度，得到的是另一种算法，不能只通过配置名称判断两者等价。

### Top-k 与 Top-p 不可默认交换

Top-k 与 Top-p 都产生按分数排序的前缀，但 Top-p 的阈值可以在第一次截断后的重新归一化分布上重新计算，因而顺序可能改变结果。

令原始概率为：

$$
p=(0.50,0.20,0.15,0.10,0.05).
$$

考虑 top-k=3 和 top-p=0.80：

- 先 top-p：前三个 token 的原始质量为 $0.85$，候选为 $\{0,1,2\}$；
- 再 top-k：前三个仍全部保留，最终候选为 $\{0,1,2\}$；
- 先 top-k：保留前三个并重新归一化为 $(0.588235,0.235294,0.176471)$；
- 再 top-p：前两个归一化概率之和为 $0.823529$，最终只保留 $\{0,1\}$。

|处理顺序|重新计算 top-p 的分布|最终候选|
|---|---|---|
|top-p=0.80 → top-k=3|原始分布|0、1、2|
|top-k=3 → top-p=0.80|top-k 后的归一化分布|0、1|

如果系统声明的是同时启用 top-k 和 top-p，仍需说明是先过滤哪一个、每一步是否重新归一化、阈值在什么分布上计算。配置表中的两个数字不能替代算法协议。

### Repetition penalty 与候选截断

Repetition penalty 修改历史 token 的分数。它可能改变排序，所以应在 top-k/top-p 之前执行：

$$
z^{\mathrm{rep}}
=
\operatorname{RepetitionPenalty}(z,\text{history}).
$$

如果先 top-k 再施加 repetition penalty，原本被保留的候选可能降到边界以下，原本被删除的候选却无法重新进入集合。这样的实现也可以成立，但必须把它当成不同的解码规则测试。

## 采样、EOS 和约束状态

### EOS 是否进入候选集合

EOS 是一个普通的词表 token，但服务可以按生成状态对它施加约束：

- 生成最小 token 数之前屏蔽 EOS；
- 允许 EOS 进入 top-k/top-p，再按采样结果结束；
- 把 EOS 保留为额外的必选候选；
- stop sequence 与 EOS 分开检测。

这些规则会改变候选质量和后续长度。不能在采样完成后才删除 EOS，却仍把删除前的概率当成输出分布。

### 最小候选数量

有些服务设置 min tokens to keep，避免低温度或极尖锐分布只剩一个候选。此时 Top-p 的候选集合可以写成：

$$
\mathcal C_{\tau,r}
=
\{v_1,\ldots,v_{\max(m(\tau),r)}\},
$$

其中 $r$ 是最少保留数量。这个规则可能使实际累计质量超过 $\tau$，因此日志要同时记录阈值、最少数量和最终候选质量。

### 约束状态必须与采样步同步

语法约束、JSON 状态、工具调用状态或 bad words 过滤器都依赖已经生成的 token。每一步应按相同 request state 构造 mask，再执行 top-k/top-p。采样出的 token 若写入了错误的请求状态，下一步会使用错误候选集合，即使模型前向和 KV cache 没有报错，序列也已经失去语义一致性。

## 运行成本和 batch 实现

### 词表已经计算，过滤仍有成本

完整模型前向通常已经产生 $V$ 个 logits。候选过滤不会消除这次输出投影，但会增加排序、选择、重排和采样的工作。

|操作|典型计算|运行时关注点|
|---|---|---|
|top-k|选择前 $k$ 个分数，复杂度可接近 $O(V\log k)$|并行选择、边界 tie、临时索引|
|top-p|按概率排序并扫描累计和，常见为 $O(V\log V)$|完整排序、累计精度、早停|
|renormalization|对候选质量求和并除法|候选 gather、概率 dtype、空集合|
|CDF sample|在候选累计和上找随机数区间|RNG 消耗、边界和 token 映射|
|batch 组合|把不同候选数的结果填入张量|padding、ragged kernel、request slot|

某些实现会用近似选择、局部排序、固定 top-k 上限或融合 kernel 降低开销。这些优化必须用候选集合、候选质量和分布统计与参考实现对照，不能只比较最终文本。

### Batch 中每条请求可以有不同候选数

Top-k 在同一 batch 内通常有固定 $k$，但 top-p 的候选数随每条请求的 logits 变化。实现可以：

- 把每条请求的候选填充到 batch 内最大长度；
- 使用 ragged 候选数组和 offsets；
- 先用固定 top-k 近似，再在局部候选上做 top-p；
- 在一个 fused kernel 中完成排序、累计、重归一化和采样。

第二种路径节省 padding，但要求 offsets、token ID 和概率数组保持同一顺序。批处理重排不能把 A 请求的候选概率交给 B 请求的采样器。

### 过滤与 KV cache 是两份状态

Top-k/top-p 只决定当前一步返回哪个 token。KV cache 保存该 token 进入模型后的历史 K/V。采样步骤应形成明确的状态转移：

1. 读取当前请求的 logits 和约束状态；
2. 计算候选集合和重归一化分布；
3. 消耗约定的 RNG 并选择 token ID；
4. 把 token 追加到该请求的逻辑序列；
5. 用新的 position 执行 decode 并写入同一请求的 K/V cache；
6. 更新 EOS、stop、候选统计和生成计数。

如果第 3 步选择的 token ID 与第 5 步写入的 request slot 不一致，后续输出会把过滤器错误伪装成模型行为。[KV cache](../inference/kv-cache/)中的 cache slot、position offset 和 batch reorder 需要与候选状态一起审计。

## 与其他解码规则的边界

|规则|候选操作|是否随机|是否保留未选分支|
|---|---|---|---|
|greedy|保留最高分 token|不需要|不保留|
|temperature sampling|完整分布上 categorical sample|需要|不保留|
|top-k sampling|固定 $k$ 个候选后重新归一化再 sample|需要|不保留|
|top-p sampling|达到 $\tau$ 的最小概率前缀后重新归一化再 sample|需要|不保留|
|beam search|扩展多个 parent 并按累计分数剪枝|通常不需要|保留固定数量|

Top-k/top-p 是单路径 sampling 的候选过滤器，不是 beam search 的替代品。[贪心解码](../inference/greedy-decoding/)只描述每步 argmax；[束搜索](../inference/beam-search/)需要维护多个 parent、累计 score 和 cache reorder。[温度采样](../inference/temperature-sampling/)描述分布变平或变尖，本篇描述从该分布删除候选。

## 运行方法

下面的标准库探针用固定 logits 实现稳定 softmax、top-k、top-p 和截断后重新归一化。它不运行神经网络，因此只验证候选集合、crossing token 和概率账本。

```python
from math import exp


def softmax(logits):
    offset = max(logits)
    weights = [exp(value - offset) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def renormalize(probabilities, ids):
    total = sum(probabilities[index] for index in ids)
    return [(index, probabilities[index] / total) for index in ids]


def top_k(probabilities, k):
    count = max(1, min(k, len(probabilities)))
    ids = sorted(
        range(len(probabilities)),
        key=lambda index: (-probabilities[index], index),
    )[:count]
    return renormalize(probabilities, ids)


def top_p(probabilities, threshold):
    ids = []
    cumulative = 0.0
    ordered = sorted(
        range(len(probabilities)),
        key=lambda index: (-probabilities[index], index),
    )
    for index in ordered:
        ids.append(index)
        cumulative += probabilities[index]
        if cumulative >= threshold:
            break
    return renormalize(probabilities, ids)


logits = [2.0, 1.0, 0.0, -1.0]
probabilities = softmax(logits)
print("probabilities=", [round(value, 6) for value in probabilities])

for k in (1, 2, 3):
    result = [
        (index, round(value, 6))
        for index, value in top_k(probabilities, k)
    ]
    print("top_k=", k, "result=", result)

for threshold in (0.6, 0.7, 0.9, 1.0):
    result = [
        (index, round(value, 6))
        for index, value in top_p(probabilities, threshold)
    ]
    print("top_p=", threshold, "result=", result)
```

运行输出为：

```text
probabilities= [0.643914, 0.236883, 0.087144, 0.032059]
top_k= 1 result= [(0, 1.0)]
top_k= 2 result= [(0, 0.731059), (1, 0.268941)]
top_k= 3 result= [(0, 0.665241), (1, 0.244728), (2, 0.090031)]
top_p= 0.6 result= [(0, 1.0)]
top_p= 0.7 result= [(0, 0.731059), (1, 0.268941)]
top_p= 0.9 result= [(0, 0.665241), (1, 0.244728), (2, 0.090031)]
top_p= 1.0 result= [(0, 0.643914), (1, 0.236883), (2, 0.087144), (3, 0.032059)]
```

Top-p=0.90 包含 token 2，因为前两个 token 的累计概率只有 0.880797；这正是 crossing token 规则。Top-k=2 与 top-p=0.70 在这组 logits 上产生相同候选，但它们的配置语义不同：前者固定数量，后者固定累计质量。改变 logits、温度或其他 processor 后，两者会产生不同集合。

## 失效模式和审计方法

### 把 top-p 当作 top-k

Top-k 的 $k$ 是候选数量；Top-p 的 $\tau$ 是累计概率阈值。把一个服务的 top-k 配置直接映射成另一个服务的 top-p 数值，会改变候选数量和尾部质量。跨服务比较应记录每步候选数，而不是只比较配置字段。

### 删除 crossing token

如果循环在累计和达到阈值前后顺序错误，可能把最后一个必要 token 删除。审计时重放一个累计概率恰好跨越阈值的向量，检查最终质量是否至少达到 $\tau$，并检查 crossing token 是否存在。

### 截断后没有重新归一化

完整概率向量删除尾部后总和小于 1。CDF 最后一个边界也小于 1，落在剩余区间的随机数会触发未定义回退。审计时检查候选概率总和是否在容差内等于 1。

### Top-k 边界 tie 不稳定

不同排序、dtype 或硬件 kernel 可能在边界相同分数处选择不同 token。用人工构造相同 logits 的词表测试 tie-breaking，并保存边界 token ID。

### top-p 累计精度不足

低精度累计概率可能在阈值附近提前或延后一个 token。参考实现使用明确的累计 dtype；生产 kernel 应在相同输入上比较候选集合、累计质量和重归一化概率。

### 过滤器顺序没有记录

temperature、repetition penalty、mask、top-k、top-p 和 renormalization 的顺序会改变候选集合。日志至少保存每一步的 processor 名称、输入分数摘要、候选数、候选质量和最终 token。

### 约束把所有候选都屏蔽

空候选集合不是普通的低概率事件，而是约束状态或配置失败。服务应返回结构化错误或使用经过批准的回退规则，并保留失败时的 request state。

### batch 候选错位

在 ragged 或 padded batch 中，token ID、概率、offset 和 request slot 任何一个错位都会把一个请求的候选交给另一个请求。使用带有不同 prompt、不同候选数和不同 EOS 时刻的交叉 batch 测试。

### 只用最终文本判断优化正确

近似 top-k、fused top-p 和硬件采样 kernel 可能偶尔给出相同文本，但候选集合已不同。验证应比较候选 ID、候选质量、归一化概率、RNG 输入和最终 token，而不只是字符串。

### 一份最小候选截断审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|输入分数|明确 raw logits、processed logits 或 log probability|processor 顺序、温度、dtype|
|mask|禁止 token 在 softmax 前处理，允许集合非空|EOS、格式约束、bad words|
|top-k|$k$ 在明确范围内，边界 tie-breaking 固定|排序稳定性、token ID 顺序|
|top-p|$\tau$ 在明确范围内，最小前缀包含 crossing token|累计顺序、累计 dtype|
|候选质量|截断后显式记录 $M_{\mathcal C}$|概率求和、下溢|
|重新归一化|候选概率和在容差内等于 1|gather/scatter、空集合|
|CDF|候选顺序、边界和随机数协议固定|$<$ 与 $\le$、RNG 消耗|
|组合顺序|temperature、top-k、top-p 的顺序固定|配置解析、重复归一化|
|EOS/stop|EOS、stop sequence、max tokens 分开记录|最小生成长度、回退|
|batch|request slot、候选 offset、token ID 同步|ragged padding、重排|
|KV cache|采样 token 写入正确 request 和 position|cache slot、offset、append|
|复现|保留候选摘要、候选质量、随机状态和 token ID|只保存最终文本|

Top-k 和 Top-p 都是对 next-token 分布的有损候选压缩。Top-k 固定候选数量，适合把计算和随机性限制在明确大小内；Top-p 固定累计概率，适合让候选数量随分布尖锐程度变化。两者都必须明确输入分布、crossing token、重新归一化、过滤顺序、tie-breaking、RNG 和请求状态，才能把一次生成解释为可复核的采样过程。

## 相关词条

- [温度采样](../inference/temperature-sampling/)：说明温度如何改变完整 categorical 分布，再连接到候选截断。
- [贪心解码](../inference/greedy-decoding/)：说明 top-k=1 如何退化为单步 argmax。
- [束搜索](../inference/beam-search/)：对比保留多个累计分数 parent 的序列搜索。
- [重复惩罚](../inference/repetition-penalty/)：说明历史 token 的分数变换如何影响排序和候选集合。
- [推理](../inference/inference/)：固定生成配置、停止条件、batch 和性能协议。
- [KV cache](../inference/kv-cache/)：说明过滤后 token 如何进入下一次 decode 和历史 K/V。
- [Softmax](../neurons-and-activations/softmax/)：推导 logits 到概率单纯形的稳定归一化。
- [采样](../probability/sampling/)：提供离散分布、CDF 和随机抽样的基础定义。
