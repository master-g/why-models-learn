---
title: "重复惩罚：用历史 token 调整下一步分数"
tags: ["why-models-learn"]
---

重复惩罚在生成第 $t$ 个 token 前读取历史 token 集合，再修改这些 token 的 next-token logits，使已经出现的 token 更难再次被选中。常见的分段规则是：已出现 token 的 logit 为正时除以惩罚系数，为负时乘以惩罚系数；这不是从所有分数中减去一个固定常数，也不是按出现次数自动线性累加。重复惩罚改变 processed logits 和后续候选集合，必须明确历史范围、token 粒度、惩罚系数、处理顺序以及与 top-k/top-p、temperature、EOS 和约束的交互。

![重复惩罚示意图：历史中的 token 0 和 token 2 经正负分段规则调整后，最高分从 token 0 变为 token 1](/assets/inference/svg/repetition-penalty.1.svg)

## 先固定历史与分数

### 历史是 token ID 序列

设当前请求在生成位置 $t$ 之前已经拥有 token 序列：

$$
x_{<t}=(x_0,\ldots,x_{t-1}).
$$

如果重复惩罚只按“是否出现过”工作，历史集合为：

$$
H_t
=
\{x_i:0\le i<t\}.
$$

集合 $H_t$ 丢失了每个 token 的出现次数和位置。token 0 出现一次与出现五次，在这种 repetition penalty 中都只产生一次 membership 命中。若服务希望次数越多惩罚越大，应使用显式 count-based 规则，不能把集合规则解释成频率规则。

历史范围需要单独配置：

|历史范围|集合内容|主要影响|
|---|---|---|
|仅生成部分|本次请求已经返回的 token|主要抑制新文本内部重复|
|prompt 加生成|输入和输出的全部 token|会降低复述用户关键词的概率|
|滑动窗口|最近 $W$ 个 token|长文本中旧 token 重新可用|
|beam 独立历史|每条 beam 的前缀 token|不同 parent 可以有不同惩罚|
|batch 共享历史|错误地混合多个请求 token|会把一个请求的历史泄漏给另一个请求|

默认 tokenizer 的 token ID 是历史单位。一个词可能被切成多个 subword token；对一个 token ID 的惩罚不等于对完整词字符串的惩罚。跨 tokenizer、跨模型或跨语言比较时，需要记录 tokenizer 版本和历史 token 序列。

### 输入必须是明确的 score 类型

模型前向通常输出 raw logits $z\in\mathbb R^V$。重复惩罚可以在 logits 上实现，也有实现把它作用于 log probability 或其他 processed score。三者不是同一个算法：

|输入|数值语义|需要确认|
|---|---|---|
|raw logits|未归一化的任意实数分数|正负分支以 0 为基准|
|log probability|归一化概率的对数|所有值通常不为正，分段效果会不同|
|已处理 logits|经过其他 processor 的分数|依赖 processor 顺序和 mask 状态|

本文固定最常见的 raw-logit 版本。配置文件不能只写 repetition penalty 的数值，还要写它接收哪一层 score。

## 常见 repetition penalty 的分段公式

### 惩罚系数大于 1

令惩罚系数为 $r>0$。对历史集合中的 token，常见分段变换为：

$$
z^{\mathrm{rep}}_v
=
\begin{cases}
z_v/r,&v\in H_t,\ z_v>0,\\
r z_v,&v\in H_t,\ z_v<0,\\
z_v,&v\notin H_t\ \text{或}\ z_v=0.
\end{cases}
$$

当 $r>1$ 时：

- 正 logit 除以 $r$，向 0 移动；
- 负 logit 乘以 $r$，远离 0；
- 未出现 token 保持原分数；
- 0 不变。

因此已出现 token 的分数会降低，但降低幅度取决于它原本的符号和大小。规则不是简单的 $z_v-r$ 或 $z_v-r\mathbf 1[v\in H_t]$。

### 分段规则为什么要看符号

设两个已出现 token 的分数分别为 $2$ 和 $-2$，$r=2$：

$$
2\longmapsto1,
\qquad
-2\longmapsto-4.
$$

两者都远离“更容易被选择”的方向。正分数变小，负分数变得更负。如果把所有分数都除以 $r$，负分数会向 0 移动，反而可能提高已出现负分 token 的相对概率，这与惩罚目标相反。

如果 $r=1$，变换是恒等映射：

$$
z^{\mathrm{rep}}=z.
$$

如果 $0<r<1$，方向反过来：已出现正 logit 会变大，已出现负 logit 会向 0 移动。它提高重复 token 的相对机会，通常应被命名为奖励或回放规则，而不是继续称为惩罚。

### 不合法系数不能静默处理

$ r\le0 $ 会使除法、符号方向或概率解释失去定义。服务应在配置解析时拒绝：

$$
r\le0.
$$

也不能把空值、NaN 或无穷值自动转换为默认惩罚。默认值应在请求进入推理前完成解析，并写入最终生成配置。

## 这不是频率惩罚，也不是存在惩罚

### Frequency penalty 按出现次数变化

令 token $v$ 在历史中的计数为：

$$
c_t(v)
=
\sum_{i=0}^{t-1}\mathbf 1[x_i=v].
$$

一种线性 frequency penalty 是：

$$
z^{\mathrm{freq}}_v
=
z_v-\alpha c_t(v),
\qquad
\alpha\ge0.
$$

token 每多出现一次，就额外减少 $\alpha$。它依赖计数，重复五次的 token 比重复一次的 token 多减少 $4\alpha$。

### Presence penalty 只按是否出现

一种 presence penalty 是：

$$
z^{\mathrm{pres}}_v
=
z_v-\beta\mathbf 1[v\in H_t],
\qquad
\beta\ge0.
$$

它只关心 token 是否已经出现，与出现一次还是多次无关。它和 repetition penalty 都使用历史集合，但修改函数不同：

|规则|历史统计|分数修改|对正负 logit 是否分段|
|---|---|---|---|
|repetition penalty|是否出现|按符号除法或乘法|是|
|frequency penalty|出现次数|减去 $\alpha c_t(v)$|通常否|
|presence penalty|是否出现|减去 $\beta$|通常否|
|no-repeat n-gram|局部序列状态|直接 mask 非法 token|不适用|

相同的参数数字不能在这四种规则之间互换。某个服务把字段命名为 penalty，并不说明它使用哪一类公式。

### No-repeat n-gram 是硬约束

设最近历史构成的 n-gram 状态不允许下一个 token $v$，硬约束可以写成：

$$
z^{\mathrm{mask}}_v
=
\begin{cases}
-\infty,&v\in\mathcal B_t,\\
z_v,&v\notin\mathcal B_t.
\end{cases}
$$

其中 $\mathcal B_t$ 是当前状态下禁止的 token 集合。它把概率质量设为 0，而 repetition penalty 只降低分数、通常仍保留非零概率。两者可以同时启用，但顺序和失败语义必须分别记录。

## 重复惩罚会改变排序

### 一个四 token 例子

取：

$$
z=(3,2,-0.5,-1.5),
\qquad
H_t=\{0,2\},
\qquad
r=2.
$$

token 0 和 token 2 已出现。应用分段规则：

$$
z^{\mathrm{rep}}
=
(1.5,2,-1,-1.5).
$$

未处理时的排序为：

$$
0>1>2>3.
$$

处理后的排序变为：

$$
1>0>2>3.
$$

token 0 原本是最高分，但由于它是已出现的正 logit，被除以 2 后低于未出现的 token 1。token 2 原本已经低于 token 0，乘以 2 后更低。

### softmax 之后的概率也会改变

对原始 logits 做稳定 softmax：

$$
p
\approx
(0.709630,0.261058,0.021429,0.007883).
$$

对处理后的 logits 做稳定 softmax：

$$
p^{\mathrm{rep}}
\approx
(0.359635,0.592939,0.029521,0.017905).
$$

token 1 的概率从约 $0.261$ 升到约 $0.593$，原因不是 token 1 的分数增加，而是 token 0 的分数降低、token 2 的分数进一步降低。所有概率都必须在处理后的 logits 上重新归一化。

### Repetition penalty 不保证没有重复

惩罚后的 token 仍可能具有最高分或最大概率：

- 历史 token 的原始优势足够大；
- 词表中其他 token 的分数更低；
- 语法、EOS 或工具调用 mask 删除了替代候选；
- top-k/top-p 截断后只剩历史 token；
- $r$ 接近 1；
- tokenizer 把可重复的文本拆成不同 token。

因此“启用 repetition penalty”只能说明分数变换已启用，不能说明输出一定不重复。应比较处理前后的候选集合和实际重复率。

## 绝对分数基准会影响变换

### 普通 softmax 对平移不敏感

对于任意常数 $c$，普通 softmax 满足：

$$
\operatorname{softmax}(z+c\boldsymbol 1)
=
\operatorname{softmax}(z).
$$

这是因为分子和分母同时乘以 $\exp(c)$。但分段 repetition penalty 以 0 为符号边界，通常不满足：

$$
R_r(z+c\boldsymbol 1)
=
R_r(z)+c\boldsymbol 1.
$$

所以先对 logits 加常数再应用 repetition penalty，可能得到不同结果。实现不能在 processor 前后随意做只对 softmax 安全的 logit 平移。

### 数值例子

对上面的 $z$ 加 4：

$$
z+4\boldsymbol 1=(7,6,3.5,2.5).
$$

对历史 token 0 和 2 使用 $r=2$：

$$
R_2(z+4\boldsymbol 1)
=
(3.5,6,1.75,2.5).
$$

原始路径得到 $(1.5,2,-1,-1.5)$，平移后再惩罚得到 $(3.5,6,1.75,2.5)$。即使两条路径在 repetition penalty 之前对应相同的 softmax 分布，processor 之后的排序和概率也不同。

这也是为什么要记录 processor 输入的分数基准。把 logits 先做中心化、裁剪或量化后再调用重复惩罚，已经改变了算法。

## 处理顺序与组合边界

### 常见顺序

一种明确的生成路径是：

$$
\text{raw logits}
\rightarrow
\text{repetition penalty}
\rightarrow
\text{frequency/presence penalty}
\rightarrow
\text{constraint mask}
\rightarrow
\text{temperature}
\rightarrow
\text{top-k/top-p}
\rightarrow
\text{sample}.
$$

也有系统先执行 mask，再执行 score processor。只要 mask 后的 $-\infty$ 不被错误变换、禁止 token 不会重新出现，路径可以成立；但必须有测试覆盖。

正温度除法与分段 repetition penalty 在精确实数上通常可以交换，因为 $T>0$ 不改变符号：

$$
R_r(z)/T
=
R_r(z/T).
$$

但这不表示所有实现都等价。输入是否是 logits、是否存在低精度舍入、其他 processor 是否插入、top-k/top-p 在哪一步执行，都会改变结果。top-k/top-p 不能与 repetition penalty 默认交换，因为它们会先删除候选。

### 先过滤再惩罚会丢失恢复机会

如果先 top-k：

$$
z\rightarrow\operatorname{TopK}(z)\rightarrow R_r,
$$

被 top-k 删除的 token 不会因为 repetition penalty 后分数相对升高而重新进入候选集合。若先惩罚：

$$
z\rightarrow R_r(z)\rightarrow\operatorname{TopK},
$$

排序会先反映历史惩罚。两条路径都可以实现，但它们对边界 token 的处理不同。常见设计把 repetition penalty 放在候选过滤之前。

### Mask 不能被 processor 解除

禁止 token 通常用 $-\infty$ 表示。惩罚处理器需要明确跳过非有限值：

$$
z_v=-\infty
\Longrightarrow
z^{\mathrm{rep}}_v=-\infty.
$$

如果代码把 $-\infty$ 乘以 $r$，仍然得到 $-\infty$；如果出现 NaN、先做有限值替换或错误地把 mask 转换为大负数，就可能让禁止 token 重新进入 top-k。审计时检查每个 processor 前后的 mask 集合。

### Temperature、Top-p 和 EOS

Repetition penalty 改变概率后，temperature 会进一步改变分布尖锐度，top-p 会重新决定候选数量。EOS 是否在历史集合中也要明确：

- prompt 中出现 EOS 字符对应 token，不等于请求已经完成；
- 已生成 EOS 后通常立即结束，不再执行下一次 repetition penalty；
- 若 EOS 被作为普通 token 参与候选，惩罚它可能改变结束概率；
- 最小生成长度可以先 mask EOS，再执行候选过滤。

输出长度的变化来自 EOS 概率和 stop 规则的共同结果，不能只归因于惩罚系数。

## 历史状态、batch 和 beam

### 每条请求维护自己的历史

在 batch 中，历史集合应按 request slot 独立维护。请求 A 的 token 不能写入请求 B 的 seen 集合。连续 batching 加入或移除请求时，需要同步更新：

- request ID 到 batch slot 的映射；
- token ID 历史或计数；
- repetition penalty 配置；
- EOS 和 stop 状态；
- KV cache slot 与 position offset。

只复制当前 batch 的 token tensor 而不复制 request state，会在重排后产生跨请求重复惩罚。

### Beam search 需要随 parent 重排

Beam search 中每条 beam 有自己的历史集合：

$$
H_t^{(b)}
=
\{x_i^{(b)}:0\le i<t\}.
$$

当 beam 选择 parent index 后，必须同时重排：

- token 历史或压缩的 seen mask；
- frequency count；
- repetition penalty 的配置状态；
- KV cache；
- 约束状态；
- cumulative score。

只重排 KV cache 而不重排历史惩罚状态，会让模型看到一个 parent，却按另一个 parent 的重复历史处理 logits。[束搜索](../inference/beam-search/)的 parent index 和 cache reorder 需要与这里的 history reorder 使用同一索引。

### 历史存储可以压缩，但语义不能改变

词表大小为 $V$ 时，seen mask 可用 $V$ 位表示：

$$
\text{memory}_{\mathrm{mask}}
=
\frac{V}{8}\ \text{bytes}.
$$

Count-based 规则需要计数数组或稀疏计数表。滑动窗口需要删除离开窗口的 token，并在计数降为 0 时清除 seen 状态。压缩 bitset、哈希表和 GPU mask 都必须通过相同 token 序列的候选结果验证。

## 运行成本和数值稳定

### 分数变换本身不是主要成本

对每个历史 token 执行一次分段变换，稀疏实现成本接近 $O(|H_t|)$；如果每一步都扫描完整词表，成本为 $O(V)$。主要工程选择是：

|实现方式|输入访问|适用边界|
|---|---|---|
|稀疏 token ID 列表|只写历史 token 的分数|历史短、词表大|
|dense seen mask|并行扫描全部词表|GPU fused kernel、历史较长|
|稀疏 count map|同时支持 frequency penalty|需要计数更新|
|融合 logit processor|在一个 kernel 内完成多种变换|需验证 mask、dtype 和顺序|

低精度分数会放大边界差异。一个 logit 原本接近 0 时，正负分支可能因舍入改变。实现应记录 dtype，并用接近 0、重复 token tie 和 $-\infty$ 的输入做单元测试。

### 非有限值需要显式处理

对每一步 processed logits 检查：

- NaN 不能进入排序或 softmax；
- +inf 通常表示前序 processor 失败；
- -inf 代表已屏蔽 token，应保持不可选；
- 全部为 -inf 表示约束失败；
- 惩罚系数不是有限正数时拒绝请求。

不要把 NaN 静默替换成 0，再把替换后的 token 写入 KV cache。错误应该在选择 token 之前暴露。

## 与其他解码规则的边界

|规则|历史依赖|对分数的作用|随机性|
|---|---|---|---|
|repetition penalty|是否出现过的 token 集合|按符号除法或乘法|取决于后续选择器|
|frequency penalty|每个 token 出现次数|按次数减去固定系数|取决于后续选择器|
|presence penalty|是否出现过|减去固定系数|取决于后续选择器|
|top-k/top-p|当前概率或排序|删除候选并重新归一化|通常需要 sample|
|greedy|无历史惩罚也可使用|选择最高 processed logit|不需要|
|beam search|每条 parent 的历史|累计 score 并保留多个分支|通常不需要|

Repetition penalty 是 logit processor，不是完整的 decoding strategy。它可以与 greedy、temperature sampling、top-k/top-p 或 beam search 组合。组合后应使用完整名称说明路径，例如“repetition penalty 后的 top-p sampling”，不能把所有结果都称为重复惩罚解码。

## 运行方法

下面的标准库探针复现分段 repetition penalty、排序、softmax 和平移敏感性。它只处理四个 logits，不运行神经网络。

```python
from math import exp


def repetition_penalty(logits, seen, penalty):
    adjusted = list(logits)
    for index in seen:
        if adjusted[index] > 0:
            adjusted[index] /= penalty
        elif adjusted[index] < 0:
            adjusted[index] *= penalty
    return adjusted


def softmax(logits):
    offset = max(logits)
    weights = [exp(value - offset) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


logits = [3.0, 2.0, -0.5, -1.5]
seen = {0, 2}

for penalty in (1.0, 2.0, 0.5):
    adjusted = repetition_penalty(logits, seen, penalty)
    ranking = sorted(
        range(len(adjusted)),
        key=lambda index: (-adjusted[index], index),
    )
    probabilities = [round(value, 6) for value in softmax(adjusted)]
    print(
        "penalty=",
        penalty,
        "adjusted=",
        adjusted,
        "ranking=",
        ranking,
    )
    print("probabilities=", probabilities)

shifted_logits = [value + 4.0 for value in logits]
shifted_adjusted = repetition_penalty(shifted_logits, seen, 2.0)
print("shifted_logits=", shifted_logits)
print("shifted_adjusted=", shifted_adjusted)
```

运行输出为：

```text
penalty= 1.0 adjusted= [3.0, 2.0, -0.5, -1.5] ranking= [0, 1, 2, 3]
probabilities= [0.70963, 0.261058, 0.021429, 0.007883]
penalty= 2.0 adjusted= [1.5, 2.0, -1.0, -1.5] ranking= [1, 0, 2, 3]
probabilities= [0.359635, 0.592939, 0.029521, 0.017905]
penalty= 0.5 adjusted= [6.0, 2.0, -0.25, -1.5] ranking= [0, 1, 2, 3]
probabilities= [0.979625, 0.017942, 0.001891, 0.000542]
shifted_logits= [7.0, 6.0, 3.5, 2.5]
shifted_adjusted= [3.5, 6.0, 1.75, 2.5]
```

$ r=2 $ 把 token 0 的正分数从 3 降到 1.5，把 token 2 的负分数从 $-0.5$ 推到 $-1$，所以 token 1 成为最高分。$r=0.5$ 产生相反方向的分数变化。平移后的结果不同，说明 repetition penalty 依赖 logit 的绝对符号基准。

## 失效模式和审计方法

### 把它实现成固定减法

$z_v-\alpha$、$z_v/r$ 和本文的正负分段规则是三个不同变换。固定减法不会对负 logit 执行“远离 0”的处理，也不能复现常见 repetition penalty。审计应逐个构造正分、负分、零分和未出现 token，核对四种分支。

### 把集合规则当成次数规则

常见 repetition penalty 对历史集合只命中一次。若实现每次扫描历史就再次除以 $r$，同一个 token 出现五次会被重复变换五次，结果已经变成另一种 count-based penalty。测试要分别覆盖出现一次、连续出现多次和离开滑动窗口。

### 只惩罚生成文本或只惩罚 prompt

历史范围改变候选分布。只惩罚生成文本允许模型复述 prompt；把 prompt 也纳入历史又可能降低必要关键词的概率。配置报告应标明 prompt、generated、all 或窗口长度，而不是只记录 penalty 数值。

### 在错误的 score 层应用

把 raw-logit 规则直接用于 log probability，会让大部分分数落在负分支。把处理后的分数再次当 raw logits 处理，也会重复修改。日志记录每个 processor 的输入输出摘要和 score type。

### 处理 mask 后的非有限值

对 -inf 做不兼容的类型转换、截断或 NaN 运算，可能让禁止 token 重新可选。每一步检查 mask 集合、有限值数量和最终候选集合。

### 先 top-k 再 repetition penalty

被 top-k 删除的 token 不会重新进入候选集合。若设计要求 repetition penalty 影响候选边界，应先惩罚再过滤，并用边界 logits 测试顺序。

### 惩罚系数越大就一定更好

$ r $ 越大，已出现 token 的分数变化越强，但可能破坏合法复述、代码标识符、数学符号或必要格式。选择系数时要按任务评估重复率、语义正确率、格式通过率和长度，而不是只看唯一 token 数。

### 只比较最终字符串

不同历史状态可能偶然生成同一个短字符串。审计应保存每步历史集合或摘要、原始/处理后 logits、候选排序、采样配置、EOS 状态和 token ID。

### Batch 或 beam 状态错位

request slot、beam parent、history mask、count map 和 KV cache 必须使用同一重排索引。用两个可识别 prompt 做交叉 batch，再用两个不同重复历史的 beam 做 parent reorder 测试。

### 一份最小重复惩罚审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|score type|明确输入是 raw logits、log probability 还是 processed score|processor 顺序、平移和归一化|
|penalty|有限且 $r>0$，$r=1$ 的恒等行为已测试|配置解析、NaN、零值|
|历史范围|prompt、generated、all 或窗口长度固定|token ID 序列、窗口淘汰|
|历史单位|固定 tokenizer 和 token ID 粒度|Unicode、subword 边界|
|正分支|历史正 logit 除以 $r$|是否误用固定减法|
|负分支|历史负 logit 乘以 $r$|符号判断、低精度舍入|
|零分支|0 保持 0|比较条件、NaN|
|mask|非有限禁止 token 仍不可选|类型转换、processor 跳过逻辑|
|组合顺序|repetition、temperature、top-k/top-p 顺序固定|候选边界、重复归一化|
|batch/beam|历史状态和 KV cache 同步重排|request slot、parent index|
|EOS/stop|停止规则与惩罚状态分开记录|最小长度、回退|
|复现|保留每步历史摘要、分数摘要和 token ID|只保存文本|

重复惩罚只是把历史信息写回下一步的 score。它不能替模型理解“重复”这个自然语言概念，也不能保证无重复输出。可复核的实现需要固定 token 历史、分段公式、score 类型、mask 与过滤顺序、状态重排和错误处理，才能解释某个 token 为什么从候选首位降下去。

## 相关词条

- [Top-k 与 Top-p](../inference/top-k-top-p/)：说明分数处理后如何截断候选并重新归一化。
- [温度采样](../inference/temperature-sampling/)：说明处理后的 logits 如何形成随机采样分布。
- [贪心解码](../inference/greedy-decoding/)：说明 repetition penalty 后如何执行 argmax。
- [束搜索](../inference/beam-search/)：说明每条 beam 的历史状态如何随 parent index 重排。
- [推理](../inference/inference/)：固定生成请求、停止条件、batch 和运行时账本。
- [KV cache](../inference/kv-cache/)：说明选出的 token 如何进入下一次 decode 和历史 K/V。
- [分词](../text-representation/tokenization/)：说明 token ID、subword 边界和文本字符串之间的差异。
