---
title: "贪心解码：每一步选择当前最高分 token"
tags: ["why-models-learn"]
---

贪心解码在每个生成位置从当前 logits 中选择分数最高的一个 token，并把它追加到下一步的上下文。它不维护多个候选序列，也不从概率分布随机采样；在 checkpoint、tokenizer、prompt、logit 变换、tie-breaking、精度和停止规则固定时，输出路径是确定的。本篇推导单步 argmax、局部选择与全局序列概率的差异，再处理温度、约束、EOS、重复请求、batch 和 KV cache 接口。

![贪心解码示意图：logits 经由固定的 tie-breaking 和约束后选择一个 token，再将该 token 追加到下一步；右侧对比局部最高分路径与全局序列概率](/assets/inference/svg/greedy-decoding.1.svg)

## 先固定“贪心”选择的输入

### logits 是词表上的一组分数

设词表大小为 $V$，在已经生成的前缀 $y_{<t}$ 和输入上下文 $x$ 条件下，模型输出最后位置的 logits：

$$
z_t
=
f_\theta(x,y_{<t})\in\mathbb R^V.
$$

第 $v$ 个词表项的 logit 是 $z_t(v)$。经过 softmax 后得到条件概率：

$$
p_\theta(v\mid x,y_{<t})
=
\frac{\exp z_t(v)}
{\sum_{j=0}^{V-1}\exp z_t(j)}.
$$

贪心解码的核心选择是：

$$
y_t
=
\operatorname*{arg\,max}_{v\in\{0,\ldots,V-1\}}
z_t(v).
$$

如果所有 logit 都是有限数，softmax 是严格单调的，因此：

$$
\operatorname*{arg\,max}_v z_t(v)
=
\operatorname*{arg\,max}_v p_\theta(v\mid x,y_{<t}).
$$

这意味着实现不需要先计算完整 softmax 再找最大概率。直接在 logits 上比较通常更稳定，也减少一次指数和归一化；如果后续需要概率日志或温度采样，才按对应协议计算 log-softmax 或 softmax。

### 选择前的 logit 变换属于协议

实际服务通常不会把模型原始 logits 直接交给 argmax。可以把选择过程写成：

$$
\begin{aligned}
z_t^{\mathrm{raw}}&=f_\theta(x,y_{<t}),\\
z_t^{\mathrm{select}}&=
F\left(
z_t^{\mathrm{raw}},
y_{<t},
\text{generation config}
\right),\\
y_t&=\operatorname{Greedy}\left(z_t^{\mathrm{select}}\right).
\end{aligned}
$$

$F$ 可以包含：

- 对禁止 token 设置 $-\infty$；
- 对 EOS、PAD 或控制 token 应用状态条件；
- 对已出现的 token 应用 repetition penalty；
- 对工具调用、JSON 或语法约束屏蔽不合法候选；
- 对不同温度或其他 processor 修改分数；
- 对词表映射、bad words 和自定义禁止集合做处理。

因此“greedy 输出”必须说明 argmax 作用在 raw logits 还是 processed logits 上。后续 [温度采样](../inference/temperature-sampling/)、[Top-k 与 Top-p](../inference/top-k-top-p/) 和 [重复惩罚](../inference/repetition-penalty/)分别展开具体变换；本文只固定它们进入选择器的接口。

## argmax 的数学细节

### 最大值集合可能不止一个

若多个 token 具有相同最大 logit，最大值集合为：

$$
A(z)
=
\left\{
v\in\{0,\ldots,V-1\}
\mathrel{\colon}
z(v)=\max_j z(j)
\right\}.
$$

argmax 在数学上可能返回集合，而程序需要返回一个 token ID。一个常见确定性规则是选择最小 token ID：

$$
\operatorname{Greedy}_{\min}(z)
=
\min A(z).
$$

也可以选择词表顺序中的第一个项，或者使用显式优先级表。规则本身不重要，规则必须固定。GPU reduction、并行归约、低精度舍入和不同库的 tie-breaking 若不一致，即使 logits 数值看起来相同，也可能选择不同 token。

用严格大于号维护当前最早索引，可以实现“相同分数保留较小 ID”：

|比较情况|更新当前 token|
|---|---|
|新分数大于当前最大值|更新为新 token|
|新分数等于当前最大值|保留旧 token|
|新分数小于当前最大值|保留旧 token|
|分数为 NaN|先按协议报错或屏蔽，不能交给普通比较|

### NaN 和负无穷要先处理

$-\infty$ 可以表达禁止候选；它不会成为有限 logit 的最大值。如果所有候选都被设为 $-\infty$，选择器没有合法 token，应触发显式失败、回退或约束修复，不能静默返回词表第一个 token。

NaN 不满足普通的大小关系。不同实现可能在遇到 NaN 时保留初始索引、传播 NaN 或产生未定义的 reduction 结果。推理协议应先检查：

$$
\operatorname{finite}(z_t(v))
$$

是否对所有允许候选成立。异常时记录请求、位置、logit processor 和候选 mask；不要把异常 token 当作模型的语义输出。

### 正温度不改变纯 argmax 的顺序

对于 $T>0$，温度分布为：

$$
p_T(v)
=
\operatorname{softmax}\left(\frac{z(v)}{T}\right).
$$

因为除以正数保持大小顺序：

$$
\operatorname*{arg\,max}_v\frac{z(v)}{T}
=
\operatorname*{arg\,max}_v z(v).
$$

所以如果最终仍然执行纯 argmax，$T=0.5$ 与 $T=2$ 选择相同 token；温度只会改变采样分布，不会改变无 tie 的贪心选择。工程中把 temperature 设为 0 往往表示“启用 greedy 分支”，而不是执行数学上的除以零。若实现先应用 temperature processor 再调用 argmax，必须检查它是否还包含其他截断、约束或浮点转换。

## 一步一步生成序列

### 自回归递推把 token 放回条件

给定初始 prompt $x$，第一个生成位置使用 prefill 的最后位置 logits，或者执行协议指定的首个 decode。选择 $y_t$ 后，下一步条件变为：

$$
y_t
=
\operatorname{Greedy}
\left(
F\left(
f_\theta(x,y_{<t}),
y_{<t},
c
\right)
\right),
\qquad
y_{<t+1}=y_{<t}\mathbin{\Vert}y_t.
$$

这里的 $\Vert$ 表示序列追加。第 $t+1$ 步的 logits 依赖已经选择的 $y_t$；贪心解码不是一次性对所有位置独立取最大值。

一个最小循环需要记录：

|状态|作用|更新时机|
|---|---|---|
|token 序列|作为下一步条件|选择后追加|
|position ID|确定位置编码和 cache offset|每次前向前推进|
|KV cache|保存历史 K/V|新 token 前向后写入|
|generated count|限制 max new tokens|选择或追加后更新|
|finished flag|EOS、stop、取消或超时|每步选择后检查|
|返回 token|区分已生成和已返回内容|后处理阶段确定|

[推理](../inference/inference/)固定了 prefill、decode、停止和服务调度的完整请求协议；[KV cache](../inference/kv-cache/)说明新 token 如何读取历史 K/V 并追加 cache。贪心解码只决定 logits 到 token 的选择，不改变 cache 的形状公式。

### EOS 的选择和返回是两个事件

如果 argmax 选择了 EOS，至少要区分：

1. EOS 是否写入该位置的 K/V cache；
2. EOS 是否放进内部生成 token 序列；
3. EOS 是否从返回文本中删除；
4. batch 中其他请求是否继续 decode；
5. stop sequence 是否在同一步优先触发。

因此，生成长度可以有多个口径：

|长度口径|是否包含 EOS|适用场景|
|---|---|---|
|selected tokens|通常包含|审计选择器输出|
|returned tokens|按返回协议|客户端看到的 token|
|visible text tokens|按 tokenizer 和后处理|文本展示|
|decode steps|按实际前向次数|性能和 cache 日志|

若把 EOS 从返回文本删除，却把它从内部计数也删除，max new tokens 的边界可能比模型实际执行多一步。日志必须同时给出停止原因和各长度口径。

## 局部最优不等于全局序列最优

### 语言模型优化的是整条序列的乘积

在固定输入下，一条候选序列的概率为：

$$
P(y_{1:T}\mid x)
=
\prod_{t=1}^{T}
p_\theta(y_t\mid x,y_{<t}).
$$

为了避免下溢，通常使用对数概率：

$$
\log P(y_{1:T}\mid x)
=
\sum_{t=1}^{T}
\log p_\theta(y_t\mid x,y_{<t}).
$$

贪心解码在第一个位置选择当前概率最大的 token，但不会保留第二名作为后续备选。它优化的是每一步当前条件下的局部概率，不是所有可能序列中的最大累计对数概率。[束搜索](../inference/beam-search/)会保留多个候选，正是为了延迟这个不可逆选择。

### 一个两步反例

假设第一步只有 A 和 B 两个候选：

|第一步 token|第一步概率|第二步的最佳条件概率|两步序列概率|
|---|---:|---:|---:|
|A|0.60|0.50|0.30|
|B|0.40|0.90|0.36|

贪心解码选择 A，因为 $0.60>0.40$；它随后得到概率为 $0.30$ 的路径。完整两步序列中，B 路径的 $0.36$ 更高。这个反例不说明贪心“错误”，它说明局部 argmax 没有访问未选择的分支，不能保证有限长度序列的全局最优。

如果第二步的条件分布还依赖更多历史，差异会继续扩大。即使每一步的局部选择都没有数值错误，最终文本仍可能低于另一条未保留路径的累计 log probability。

### 长度使“概率最大”更难直接比较

序列概率是多个小于 1 的概率相乘。长度越长，乘积通常越小。若比较不同长度的候选，常见报告会使用累计 log probability 或长度归一化分数：

$$
s_{\mathrm{norm}}(y)
=
\frac{1}{T^\alpha}
\sum_{t=1}^{T}\log p_\theta(y_t\mid x,y_{<t}),
\qquad
\alpha\ge0.
$$

贪心解码不需要维护这个累计分数，也不使用长度归一化来选择当前 token。若把 greedy 的逐步 max 与 beam 的累计分数放在同一个表中，必须说明它们优化的对象不同。

### PPL 和 greedy 成功率不能互换

[困惑度评估](../pretraining/evaluation-perplexity/)通常把真实 target token 作为条件，计算 teacher-forcing 下的每个 token log probability。贪心生成把自己刚刚选出的 token 放回下一步条件。一个早期选择会改变后续上下文；因此：

- 低 teacher-forcing loss 不保证 greedy 文本符合格式；
- greedy 生成失败不等于每个 gold token 的条件概率都低；
- 生成长度、EOS 位置和重复模式不能由 PPL 单独推出；
- 比较 greedy 输出时要固定 prompt、tokenizer、stop 和后处理。

## 选择器前后需要固定哪些变换

### 约束 mask 可以改变 argmax

如果候选集合为 $\mathcal V_t$，约束后的 logits 可以写成：

$$
z_t^{\mathrm{mask}}(v)
=
\begin{cases}
z_t(v),&v\in\mathcal V_t,\\
-\infty,&v\notin\mathcal V_t.
\end{cases}
$$

然后在 $\mathcal V_t$ 内执行 greedy。约束可以来自 JSON 状态机、代码语法、工具调用阶段、bad words、词法前缀或应用层白名单。约束是选择协议的一部分；同一 checkpoint 在不同 $\mathcal V_t$ 上可以生成不同 token。

如果 $\mathcal V_t$ 为空，服务需要明确策略：

- 返回约束失败；
- 回退到允许的结束 token；
- 放宽某一条约束；
- 终止当前请求并保留诊断。

静默取消全部 mask 会把一个约束 bug 变成看似正常的任意输出。

### repetition penalty 会改变顺序

重复惩罚依赖历史 token 集合，因此它不是一个固定的词表排序。一个抽象写法是：

$$
z_t^{\mathrm{rep}}(v)
=
R\left(z_t^{\mathrm{raw}}(v),y_{<t},v\right).
$$

某些实现对已出现 token 分别按正负 logit 方向缩放，另一些实现使用加法惩罚或频率惩罚。只要 $R$ 改变了 logit 顺序，greedy 的 token 就会改变。报告时必须给出 penalty 类型、系数、作用范围和重复 token 的定义。

### temperature 的位置不能含糊

纯 greedy 中，正温度不改变 argmax；但下面三种路径不是同一件事：

|路径|最终选择|温度的作用|
|---|---|---|
|raw greedy|raw logits argmax|无|
|processed greedy|变换后 logits argmax|可能通过 processor 改变顺序|
|temperature sampling|从 softmax(logits / T) 采样|改变随机分布和结果概率|

因此配置里同时出现 temperature 和 do_sample=false 时，不能只看字段名判断行为。需要查看实际执行分支和传入选择器的 logits。

## batch 与并行贪心

### 每条序列沿词表轴独立取最大值

对 batch 中第 $b$ 条请求，贪心选择是：

$$
y_{b,t}
=
\operatorname*{arg\,max}_{v}
z_{b,t}(v).
$$

不同请求不共享 token 选择。它们可以共享同一个矩阵乘 kernel、KV cache block 池或连续 batch，但每条序列的约束 mask、历史 token、EOS 状态和 position offset 仍然独立。

如果 logits shape 是 $[B,V]$，argmax 轴应是词表轴 $V$。如果生成一个 chunk 后 logits shape 是 $[B,R,V]$，通常只选择每条序列最后一个有效位置，或者按协议处理每个位置；不能把 batch、序列和词表轴混在一起。

### padding 和 finished request 不应继续生成

静态 batch 中，有些请求可能已经 EOS，有些仍在生成。完成请求可以：

- 从 active batch 移除；
- 保留 slot 但屏蔽其 logits 和 cache 写入；
- 用固定 pad token 占位，直到 batch 重新整理。

无论采用哪种方式，都要记录 active mask。对已完成请求继续执行 argmax 并写入 cache，会污染生成长度、吞吐和显存统计。

连续批处理还要同步 request ID、cache slot、当前 token 和 finished flag。[KV cache](../inference/kv-cache/)中的 batch 映射若与 logits 的 batch 顺序不一致，greedy 可能从 A 请求的 logits 选择 token，却把 token 写入 B 请求的 cache。

### 贪心没有 beam 的额外分支状态

单条 greedy 序列通常只需要一份：

- 当前 token 序列；
- 每层一份 K/V cache；
- 当前 logits；
- 停止状态和生成计数。

beam search 或多样本生成会把候选维度加入有效 batch，并需要累计分数、父索引和 cache 重排。用同一个 batch shape 实现两者时，不能把“batch 中有多条序列”误认为“greedy 正在搜索多个候选”；只有一个序列继续向前的规则才是 greedy。

## 确定性需要分层声明

### 固定协议不等于跨硬件位级一致

在理想实数计算中，给定相同 logits 和 tie-breaking，greedy 选择确定。实际输出还受以下层级影响：

|层级|需要固定的对象|可能的差异|
|---|---|---|
|模型|checkpoint、adapter、量化权重|logits 改变|
|输入|原文、tokenizer、模板、special token|token 序列改变|
|执行|prefill/decode、KV cache、position offset|上下文或舍入改变|
|数值|dtype、累加精度、kernel、归约顺序|近似 tie 的排序改变|
|选择|mask、penalty、tie-breaking|候选集合或 token 改变|
|停止|EOS、stop sequence、max new tokens|返回长度改变|
|后处理|解码、规范化、文本截断|展示字符串改变|

如果两个运行的原始 logits 在最大值附近差异低于数值误差，硬件、kernel 或归约顺序可能改变 argmax。应记录 top-k margin：

$$
\Delta_t
=
z_t(v_{(1)})-z_t(v_{(2)}),
$$

其中 $v_{(1)}$ 和 $v_{(2)}$ 是第一、第二高的候选。$\Delta_t$ 越小，选择越接近数值边界；它不能单独证明下一次运行一定改变，但能标记需要复核的位置。

### 复现报告要包含 token 级证据

一份可复核的 greedy 报告至少记录：

1. prompt 的原文 hash 和 input IDs；
2. checkpoint、tokenizer、模板与 generation config；
3. 每步选择前的 top-k token、logit、margin；
4. 应用的 mask、penalty 和 tie-breaking；
5. KV cache 的逻辑长度和 position offset；
6. 选择出的 token ID、EOS/stop 原因和返回文本；
7. dtype、设备、kernel 和确定性开关；
8. 同一请求重复运行的 token 级比较。

只比较最终字符串无法区分 tokenizer、后处理、停止规则和模型 logits 的差异。

## 运行方法

下面的标准库探针验证四件事：相同最大值的 tie-breaking、softmax 与 logits 的 argmax 一致性、正温度对纯 argmax 的不变性，以及局部 greedy 路径与全局两步概率的差异。

```python
from math import exp


def softmax(values):
    offset = max(values)
    weights = [exp(value - offset) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def greedy_index(values):
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


logits = [2.0, 2.0, 0.5, -1.0]
probabilities = softmax(logits)

print("logits=", logits)
print("softmax=", [round(value, 6) for value in probabilities])
print("greedy_id=", greedy_index(logits))
print("probability_argmax_id=", greedy_index(probabilities))

for temperature in (0.5, 2.0):
    scaled = [value / temperature for value in logits]
    print(
        "temperature",
        temperature,
        "greedy_id=",
        greedy_index(scaled),
    )

first_step = {"A": 0.60, "B": 0.40}
second_step_best = {"A": 0.50, "B": 0.90}
greedy_path = "A"
global_path = "B"
print("greedy_path_probability=", round(first_step[greedy_path] * second_step_best[greedy_path], 2))
print("global_best_path_probability=", round(first_step[global_path] * second_step_best[global_path], 2))
```

运行输出为：

```text
logits= [2.0, 2.0, 0.5, -1.0]
softmax= [0.439963, 0.439963, 0.098169, 0.021904]
greedy_id= 0
probability_argmax_id= 0
temperature 0.5 greedy_id= 0
temperature 2.0 greedy_id= 0
greedy_path_probability= 0.3
global_best_path_probability= 0.36
```

探针使用严格大于号，因此相同最大值保留较小索引 0。两种正温度仍保留相同排序。A 路径的第一步概率更高，但两步累计概率为 $0.30$；B 路径的累计概率为 $0.36$，对应局部 argmax 没有保留的分支。

## 失效模式和审计方法

### argmax 取错轴

词表轴不是 batch 轴，也不是序列轴。对 $[B,V]$ 取错轴会在不同请求之间比较 token；对 $[B,R,V]$ 取错轴会把多个位置的 logits 混在一起。审计要先打印 logits shape，再断言 argmax 输出 shape 为每条有效序列一个 token ID。

### 先四舍五入 logits 再比较

两个候选的原始差异可能很小。把 logits 打印到低精度后再执行比较，会制造人工 tie 或逆转顺序。比较应使用计算 dtype 的原始值；展示时另行格式化，并记录 top-1/top-2 margin。

### tie-breaking 依赖 reduction 实现

并行 max reduction 可能只返回最大值，不返回稳定索引。不同线程块、设备或库版本可能选择不同的相同最大值。显式实现最小 ID、词表顺序或优先级 tie-breaking，再用包含真实 tie 的输入测试。

### 把 sampling 配置当作 greedy

temperature、top-k、top-p 和随机 seed 只有在采样分支中才按概率抽样。配置字段存在不表示实际执行了采样；反过来，某些服务会在调用 greedy 前先执行 logit processor。检查最终传给选择器的分数和选择函数。

### EOS、PAD 和 stop sequence 混在一起

EOS 是词表 token；PAD 是 batch 对齐 token；stop sequence 是生成文本或 token 序列的停止模式。它们的判断层级不同。记录 selected token、cache 是否追加、返回文本和停止原因，才能定位 off-by-one。

### raw logits 与 processed logits 不一致

离线复现若直接对 raw logits 做 argmax，而服务先应用 mask、penalty 或约束，两个结果可以稳定地不同。导出每一步 processor 前后的 top-k，确认禁止集合和历史依赖。

### cache 映射与 token 映射错位

如果 token 选自第 $b$ 条 logits，却写入第 $b'$ 条 cache，下一步的差异不是 greedy 规则造成的。连续 batching、beam 维压缩、请求取消和 slot 复用都要使用同一个 permutation。使用不同请求的可识别 token 做双请求测试。

### 误把局部概率当作序列概率

当前最大概率不等于有限长度全局序列概率最大。需要全局搜索时使用 [束搜索](../inference/beam-search/)或其他候选保留规则，并明确累计 log probability 与长度归一化。greedy 的结果不能用“概率最大序列”描述。

### 以最终字符串判断确定性

文本解码可能移除 EOS、合并空格、规范化 Unicode 或丢弃控制 token。对比应先比较 token ID，再比较停止原因，最后比较返回文本。最终字符串相同也不代表中间 token、cache 路径和 logits 相同。

### 一份最小 greedy 审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|输入协议|checkpoint、tokenizer、模板和 prompt token 固定|版本、special token、input IDs|
|logits shape|沿词表轴 $V$ 做 argmax|batch/sequence/vocab 轴|
|选择分数|明确 raw 或 processed logits|mask、penalty、约束 processor|
|tie-breaking|相同最大值有固定索引规则|reduction、词表顺序、NaN|
|数值边界|记录 top-1/top-2 margin 和 dtype|低精度、归约顺序、kernel|
|温度语义|区分正温度 argmax 与 temperature sampling|do sample 分支、T=0 特判|
|历史更新|选择 token 进入正确的下一步条件|token append、position offset|
|KV cache|按同一 request/slot 追加 K/V|batch permutation、cache length|
|停止条件|EOS、stop、max new tokens 分开记录|返回 token、cache 写入、计数|
|batch 状态|finished 请求不再生成或写 cache|active mask、slot 回收|
|序列目标|局部 greedy 与累计序列分数分开|beam、长度归一化|
|复现证据|token 级输出和 top-k 日志可比较|只保留最终字符串|

贪心解码的实现可以很短，但验收不能只检查是否调用了 argmax。必须固定 argmax 的输入、轴、tie-breaking、历史更新和停止规则；否则同一个名称可能对应不同的执行协议。

## 相关词条

- [推理](../inference/inference/)：固定模型前向、解码循环、停止和服务调度的请求协议。
- [KV cache](../inference/kv-cache/)：说明贪心选择出的 token 如何进入增量 decode 和历史 K/V。
- [因果语言建模](../transformer-architectures/causal-language-modeling/)：推导 next-token 条件概率和训练目标。
- [Decoder-Only Transformer](../transformer-architectures/decoder-only/)：说明生成 logits 的 decoder-only 结构。
- [困惑度评估](../pretraining/evaluation-perplexity/)：区分 teacher-forcing 概率与自回归生成路径。
- [束搜索](../inference/beam-search/)：保留多个候选并比较累计 log probability。
- [温度采样](../inference/temperature-sampling/)：从温度调整后的分布随机选择 token。
- [Top-k 与 Top-p](../inference/top-k-top-p/)：截断候选集合后再采样。
- [重复惩罚](../inference/repetition-penalty/)：根据历史 token 修改选择前的 logits。
