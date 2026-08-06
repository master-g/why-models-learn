---
title: "因果掩码：为什么当前位置不能读取未来 token"
tags: ["why-models-learn"]
---

Causal Masking（因果掩码）在自回归 attention 中把 query 位置 $t$ 对未来 key 位置 $s>t$ 的连接设为不可用，只保留 $s\le t$ 的位置。它让训练时可以并行计算整段序列，同时保持每个位置的输出只依赖当前及过去输入。本篇固定行是 query、列是 key 的约定，推导下三角 mask、数值与梯度边界，再处理 padding、KV cache 偏移、批次广播和增量解码。

![因果 attention 的下三角可访问区域、被屏蔽的未来区域，以及带前缀的增量解码 mask](/assets/attention/svg/causal-masking.1.svg)

## 先固定因果 mask 的方向

### 行是 query，列是 key

设单个 attention head 的 score 矩阵为：

$$
S\in\mathbb R^{T\times S}.
$$

第 $t$ 行表示 query 位置 $t$ 对每个 key 位置的匹配分数，第 $s$ 列表示被读取的 key/value 位置。self-attention 中常见 $T=S$；交叉 attention 中两者可以不同。

在长度相同的自回归 self-attention 中，位置编号从 $0$ 开始，允许集合为：

$$
\mathcal A_t=\{s\mid 0\le s\le t\}.
$$

第 $t$ 个 query 可以读取位置 $0$ 到 $t，包括自己；它不能读取 $t+1$ 到 $T-1$。

### 用加性 mask 表示不可用连接

定义因果 mask：

$$
M_{t,s}
=
\begin{cases}
0,&s\le t,\\
-\infty,&s>t.
\end{cases}
$$

缩放后的 score 加上 mask，再沿 key 轴做 softmax：

$$
A
=
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{QK^\mathsf T}{\sqrt{d_k}}+M
\right).
$$

对允许位置，mask 加 $0$，原 score 不变；对未来位置，mask 加 $-\infty$，softmax 后权重为 $0$。实际实现也可以使用布尔 allow mask，让 kernel 在 softmax 前跳过不允许的元素。

### 对角线是否保留

本篇使用 inclusive causal mask，即保留主对角线：

$$
\operatorname{allow}(t,s)=
\begin{cases}
1,&s\le t,\\
0,&s>t.
\end{cases}
$$

如果误用严格下三角 $s<t$，第 $0$ 行没有合法 key。decoder 的第一个输入位置会失去自己的输入，单 token 序列会出现全 mask 行。除非模型明确需要排除 self-loop，否则应保留对角线。

### 输入和标签要分开编号

训练 decoder 时，输入通常右移一位。例如：

|输入位置|$0$|$1$|$2$|
|---:|---:|---:|---:|
|decoder 输入|BOS|$y_0$|$y_1$|
|训练标签|$y_0$|$y_1$|EOS|

位置 $t$ 的 decoder hidden 可以读取输入位置 $0$ 到 $t$，再预测标签位置 $t$。因果约束作用在 decoder 输入的 key/value 序列，不是把目标标签提前提供给 query。

## 一个四位置的下三角例子

### 全部 score 相同

取 $T=4$，假设每个合法位置的 score 都为 $0$。因果 softmax 后的 attention matrix 为：

$$
A=
\begin{bmatrix}
1&0&0&0\\
\frac12&\frac12&0&0\\
\frac13&\frac13&\frac13&0\\
\frac14&\frac14&\frac14&\frac14
\end{bmatrix}.
$$

每一行的和为 $1$，上三角全部为 $0$。第 $0$ 行只能读取自己，第 $3$ 行可以读取四个位置。

### value 输出是不断扩大的前缀平均

令 value 是一维序列：

$$
V=
\begin{bmatrix}
10\\
20\\
30\\
40
\end{bmatrix}.
$$

矩阵乘法 $C=AV$ 得到：

$$
C=
\begin{bmatrix}
10\\
15\\
20\\
25
\end{bmatrix}.
$$

这里每一行都是当前位置及其前缀的平均。真实模型的 score 通常不相同，所以它会对前缀做加权平均；mask 只规定哪些位置可以参与。

### score 例子显示未来泄漏的数值差

取某个 query 行的未 masked score：

$$
S_t=[2,\ 1,\ 0,\ -1],
\qquad
V=[10,\ 20,\ 30,\ 40]^\mathsf T.
$$

如果当前是位置 $t=2$，因果 mask 只保留前三个 score：

$$
\operatorname{softmax}([2,1,0])
=
[0.665240956,\ 0.244728471,\ 0.090030573].
$$

对应的 context 为：

$$
C_t=14.247896174.
$$

不加 mask 时，四个位置的权重变为：

$$
\operatorname{softmax}([2,1,0,-1])
=
[0.643914260,\ 0.236882818,\ 0.087144319,\ 0.032058603],
$$

context 变为 $15.073472654$。未来 value $40$ 单独贡献约 $1.282344131$，当前位置的输出已经包含了未来输入的信息。这个差异不是数值近似造成的，而是连接集合不同造成的。

### 行熵随可访问位置增加

在前面的均匀 score 例子中，第 $t$ 行有 $t+1$ 个等概率位置，因此行熵为：

$$
\mathcal H_t=\log(t+1).
$$

四行熵分别为 $0$、$0.693147181$、$1.098612289$、$1.386294361$，平均为 $0.794513458$。熵的增加来自候选位置数增加，不代表模型一定在训练中使用了更远的 token。

## 因果性如何进入前向和反向

### 输出对未来输入不敏感

设第 $t$ 个 query 的输出为 $C_t$。在单层 causal self-attention 中，对未来位置 $j>t$ 的输入做扰动：

$$
x_j\leftarrow x_j+\Delta x_j,
\qquad
j>t.
$$

由于 $j$ 不在 $\mathcal A_t$ 中，$C_t$ 不应随该扰动改变。对允许位置 $j\le t$ 做同样扰动，输出通常会改变，除非当前参数或输入使该路径的局部影响恰好为零。

多层堆叠不会改变这个方向：第 $t$ 层输出只能依赖位置 $0$ 到 $t$ 的上一层表示，因此下一层仍不会得到未来位置的信息。

### 被 mask 的 score 没有有效梯度

理想的 $-\infty$ additive mask 或等价的布尔 kernel 使未来权重为零：

$$
s>t
\quad\Longrightarrow\quad
A_{t,s}=0,
\qquad
\frac{\partial\mathcal L}{\partial S_{t,s}}=0.
$$

未来 key/value 不会通过第 $t$ 行的 attention 路径影响损失。它们仍然可以作为自己的位置或更晚位置的 key/value，被允许的 query 读取。

### 因果性不是“整张矩阵依次计算”

训练时可以一次产生 $Q,K,V$，再并行计算所有 query 行。上三角元素在 mask 中被跳过或设为不可用，行与行之间没有依赖关系。并行训练与自回归因果性不冲突：

|阶段|可以并行计算什么|必须禁止什么|
|---|---|---|
|完整序列训练|所有 query 行的 score 与 softmax|每行读取未来 key/value|
|单 token 解码|当前 query 对历史 cache 的读取|读取尚未生成的位置|
|块状 prefill|当前块内所有 query 行|块内未来位置之间互相读取|

因果 mask 约束的是信息路径，不要求软件循环逐 token 执行完整训练前向。

## padding、批次和多头

### 因果 mask 和 padding mask 要同时满足

如果 batch 内序列长度不同，key 位置还要满足“不是 padding”。设 valid_key(s) 表示 key/value 位置有效，则允许条件为：

$$
\operatorname{allow}(t,s)
=
\mathbf 1\{s\le t\}
\land
\operatorname{valid\_key}(s).
$$

加性表示可以写成两个 mask 的和：

$$
M=M_{\mathrm{causal}}+M_{\mathrm{padding}}.
$$

只要某个 mask 把连接设为 $-\infty$，该位置就不会进入 softmax。

### padding key 和 padding query 是两个问题

屏蔽 padding key 可以防止有效 query 读取补齐位置；它不会自动删除 padding query 的输出。padding query 仍可能对有效历史位置产生一个合法的 attention 分布。

因此通常还要：

- 对 padding query 的 hidden 或 loss 使用 query mask；
- 在残差或输出层明确决定是否保留 padding query；
- 检查 loss reduction 的分母只包含有效标签。

若只检查 attention matrix 的列，可能漏掉 padding 行继续传播的问题。

### 左 padding 需要使用真实位置编号

右 padding 时，真实 token 通常占据位置 $0$ 到 $L-1$，实现较直接。左 padding 时，补齐位置在序列前方，真实 token 的绝对位置和局部位置可能不同。因果条件必须使用模型采用的实际位置编号，不能仅按 batch 中的列号套用一个统一下三角。

这会同时影响：

- causal mask 的允许区域；
- position id 或 RoPE 的位置；
- KV cache 写入位置；
- padding query 的有效性。

### 多头共享 mask 不共享 score

在标准 self-attention 中，causal mask 通常从：

$$
[T,S]
\quad\longrightarrow\quad
[B,1,T,S]
\quad\longrightarrow\quad
[B,h,T,S]
$$

广播到每个 head。mask 相同不表示不同 head 的 $QK^\mathsf T$ 相同。GQA/MQA 仍按 query head 生成各自的 score，只是某些 head 共享 K/V。

如果某个 head 有额外的局部窗口或稀疏规则，mask 可以扩展为 head-specific；此时要独立检查每个 head 的允许集合。

### packed batch 需要块对角因果 mask

为了减少 padding，多个短序列可能被拼到一个长序列中。拼接后不能让后一条序列的 query 读取前一条序列的 key。允许关系应同时满足：

$$
\operatorname{same\_sequence}(t,s)
\land
(s\le t).
$$

矩阵上它是多个独立的下三角块，而不是一张覆盖整个 packed 长度的下三角矩阵。FlashAttention varlen 等实现可以用每条序列的 cumulative length 直接表达这个边界。

## KV cache 中的偏移 causal mask

### 增量解码的 query 不是从位置零开始

假设已经缓存 $L_{\mathrm{past}}$ 个位置，现在一次输入 $T_q$ 个新 query。key/value 序列包含：

$$
S=L_{\mathrm{past}}+T_q.
$$

第 $i$ 个新 query 的真实位置是：

$$
p_i=L_{\mathrm{past}}+i,
\qquad
i=0,\ldots,T_q-1.
$$

它可以读取 key 位置 $s\le p_i$。因此偏移 mask 为：

$$
M_{i,s}
=
\begin{cases}
0,&s\le L_{\mathrm{past}}+i,\\
-\infty,&s>L_{\mathrm{past}}+i.
\end{cases}
$$

### 单 token decode 的行可以全为零

当 $T_q=1$ 时，第 $0$ 个 query 的真实位置为 $L_{\mathrm{past}}$。key 序列长度是 $L_{\mathrm{past}}+1$，它可以读取所有历史位置和当前新位置：

$$
M_{0,s}=0,
\qquad
s=0,\ldots,L_{\mathrm{past}}.
$$

如果把这张 $1\times(L_{\mathrm{past}}+1)$ 的 mask 错误地当成从位置零开始的普通下三角，可能只允许读取第 $0$ 个 key。这个 bug 会让生成结果偏离 full-sequence prefill，且不一定触发形状错误。

### 两个新 token 的偏移例子

取 $L_{\mathrm{past}}=3$、$T_q=2$。key 位置为 $0,1,2,3,4$，两个新 query 的真实位置为 $3,4$。mask 为：

$$
M=
\begin{bmatrix}
0&0&0&0&-\infty\\
0&0&0&0&0
\end{bmatrix}.
$$

第一行的 query 位置 $3$ 可以读取三个历史位置和自己，不能读取位置 $4$；第二行的 query 位置 $4$ 可以读取全部五个位置。

这张矩阵不是 $2\times2$ 的普通下三角。它是左侧历史前缀全可见、右侧当前块内部保持下三角的组合。

### prefill 和 decode 要使用同一位置语义

可以用同一条模型处理：

1. full prefill：一次处理整个 prompt，使用从零开始的下三角；
2. chunk prefill：已有前缀 cache，处理一段新 token，使用偏移下三角；
3. token decode：已有 cache，每次处理一个新 token，当前行对历史和自己全可见。

三种路径在相同输入和相同位置编码下，应产生一致的 logits。这个一致性比只检查 mask 的矩形形状更有判别力。

## 数值实现中的 mask 顺序

### 先加 mask，再做 softmax

标准顺序为：

$$
\begin{gathered}
S=\frac{QK^\mathsf T}{\sqrt{d_k}},\\
\widetilde S=S+M,\\
A=\operatorname{softmax}_{\mathrm{row}}(\widetilde S).
\end{gathered}
$$

如果先对未 masked score 做 softmax，再把未来权重置零而不重新归一化，剩余权重的和会小于 $1$，输出不再是 value 的凸组合。

如果置零后再按合法位置重新归一化，在精确算术下可以接近加性 mask 的结果；但实现需要额外处理全 mask 行、精度和梯度。把 mask 放进 softmax 的输入更容易保持统一语义。

### 负无穷和有限 sentinel

数学定义使用 $-\infty$。工程实现可能使用：

|表示|优点|需要核对|
|---|---|---|
|布尔 mask|kernel 可以直接跳过无效元素|softmax 实现是否处理全 mask 行|
|IEEE $-\infty$|语义直接|低精度 kernel 是否支持并保持稳定|
|有限负数 sentinel|部分硬件路径更方便|值是否足够小、全 mask 行如何处理|

对包含至少一个合法 score 的行，足够小的 sentinel 通常会使非法项的指数下溢为零。若一整行都是同一个有限 sentinel，稳定 softmax 先减去该行最大值，会得到全零，再输出均匀分布。因此有限 sentinel 不能单独解决全 mask 行。

### 全 mask 行必须有显式策略

全 mask 行可能来自：

- 严格下三角的第一个位置；
- padding query；
- packed sequence 的边界错误；
- cross-attention 中空 source；
- 错误的 cache offset。

可选策略取决于模型契约，例如返回全零 attention 和全零输出、跳过该 query 的 loss，或在构造 mask 时保证至少有一个合法 key。无论选哪种，都要避免 NaN 静默进入 residual 和后续层。

### 稳定 softmax 的减最大值不改变 mask 语义

对一行合法 score $\widetilde S_t$，稳定实现计算：

$$
A_{t,s}
=
\frac{\exp(\widetilde S_{t,s}-m_t)}
{\sum_j\exp(\widetilde S_{t,j}-m_t)},
\qquad
m_t=\max_j\widetilde S_{t,j}.
$$

只要非法项为真正不可用的 $-\infty$，它们在分子中仍为零。减最大值用于避免指数溢出，不会让未来位置重新获得权重。

## 因果 mask 与计算量

### 可访问连接约为完整矩阵的一半

长度为 $T$ 的 inclusive causal matrix 中，允许的位置数量为：

$$
N_{\mathrm{causal}}
=
\frac{T(T+1)}{2}.
$$

对比完整 $T\times T$ 矩阵：

|序列长度 $T$|完整元素|因果允许元素|允许比例|
|---:|---:|---:|---:|
|$4$|$16$|$10$|$0.625000$|
|$1024$|$1,048,576$|$524,800$|$0.500488$|
|$2048$|$4,194,304$|$2,098,176$|$0.500244$|

这表示逻辑连接数。普通 dense kernel 可能仍计算部分无效上三角，再用 mask 丢弃；专门的 causal kernel 可以跳过无效 tile。

### 一个 batch 和多头的资源账本

取 $B=2$、$h=16$、$T=2048$。只统计 attention score matrix 元素：

|项目|完整 score|因果允许区域|
|---|---:|---:|
|元素数|$134,217,728$|$67,141,632$|
|FP16 字节数|$268,435,456$|$134,283,264$|

FlashAttention 等实现可能根本不把完整 $A$ 写回显存，但仍须在 tile 内执行同样的因果判定。减少显式矩阵存储和减少逻辑 score 计算是两个不同的优化。

### causal mask 不改变模型宽度

causal mask 只改变 $T\times S$ 位置连接。它不改变：

- $d_{\mathrm{model}}$；
- $d_k$、$d_v$；
- query、key、value 的 head 数；
- 输出投影的形状。

因此它不会直接减少 Q/K/V 参数量。它可能让特定 kernel 跳过上三角计算，但这属于执行优化。

## 与其他 attention 的边界

### decoder self-attention 通常需要因果 mask

自回归语言模型在位置 $t$ 预测下一个 token 时，decoder 输入可以读取从起点到 $t$ 的前缀。causal mask 保证训练前向与推理时的可用信息集合一致。

[自注意力](../attention/self-attention/)描述 Q/K/V 在同一序列上的读取；本篇补上“哪些位置能连接”的约束。[多头注意力](../attention/multi-head-attention/)和 [GQA 与 MQA](../attention/gqa-and-mqa/)改变 head 和 K/V 的组织方式，但每个 query head 都可以使用同一 causal mask。

### cross-attention 默认不采用 target causal mask

decoder cross-attention 的 query 来自目标序列，key/value 来自 encoder 的源序列。一个目标位置通常可以读取整个有效 source，而不是只能读取 source 中同索引之前的位置。它需要 source padding mask，是否需要 source causal 约束取决于具体任务。

如果系统是流式 encoder、在线翻译或单调对齐，cross-attention 可能额外使用 source prefix 或单调 mask。那是另一个连接契约，不能把 decoder self-attention 的下三角直接复制过去。

### sparse、local 和 causal 可以组合

causal mask 保留“过去和当前”这一时间方向。local window 进一步限制过去范围，sparse pattern 选择若干合法位置，prefix-LM 可能只在部分区间使用双向连接。组合时先写出最终 allow 条件，再检查每个 query 的合法 key 集合。

例如 causal 加窗口 $w$ 可以写为：

$$
\operatorname{allow}(t,s)
=
\mathbf 1\{t-w+1\le s\le t\}.
$$

若再叠加 padding 或 packed sequence，还要继续加入有效位置和序列边界条件。

## 失效模式与审计方法

### 把上三角和下三角弄反

如果矩阵行是 query、列是 key，那么 decoder causal mask 的未来区域是上三角。先写一个三位置例子，手工确认第 $0$ 行只能看第 $0$ 列，再检查代码的比较符号。

### 错误排除对角线

严格使用 $s<t$ 会让第一个 query 没有合法 key。确认模型需要的是 inclusive causal，还是有明确理由排除 self-loop。

### 先 softmax 后置零但不重归一化

这种实现让每行 attention weight 的和小于 $1$。检查 A.sum(-1)，并用固定 value 核对输出是否仍是合法 value 的凸组合。

### cache offset 从零开始

带 $L_{\mathrm{past}}$ 个 cache 位置时，新 query 的真实位置从 $L_{\mathrm{past}}$ 开始。用单 token decode 和两 token chunk prefill 例子检查允许矩阵，不能只检查 Q_len 和 K_len。

### padding query 被当作有效 query

key padding mask 可以正确屏蔽列，但 padding 行仍可能产生输出。检查 query mask、loss reduction 和残差路径。

### packed sequence 穿过样本边界

把多个样本拼成一条序列后，单张下三角会允许后一个样本读取前一个样本。使用 block diagonal mask 或 varlen sequence metadata。

### 有限 sentinel 遇到全 mask 行

全 mask 行在稳定 softmax 中可能变成均匀分布，或在使用 $-\infty$ 时产生 NaN。对空行加显式测试，并检查输出和梯度。

### 把 causal mask 当作稀疏计算保证

逻辑上不允许的元素不一定在 dense kernel 中被跳过。分别记录允许连接数量、实际 FLOPs、显式 mask 存储和 kernel 运行时间。

### 用 attention heatmap 代替因果性证明

热图只能展示一次前向的权重。因果性还要通过未来 token 扰动、上三角 score 梯度和 full prefill 对比增量 decode 验证。

### 一份最小因果 mask 审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|矩阵方向|行是 query，列是 key|QK transpose|
|对角线|inclusive causal 包含 $s=t$|小于号与小于等于号|
|未来区域|$s>t$ 的 weight 为 $0$|mask 符号、广播方向|
|行归一化|每个有效 query 行的和为 $1$|softmax 轴、置零后是否重归一化|
|未来梯度|上三角 score 梯度为 $0$|mask 是否在 softmax 前生效|
|padding key|补齐列不可读|key mask|
|padding query|补齐行不进入有效 loss|query mask、reduction|
|cache offset|新 query 使用真实位置 $L_{\mathrm{past}}+i$|position id、mask 起点|
|chunk prefill|历史前缀全可读，当前块内部保持下三角|$T_q\times(L_{\mathrm{past}}+T_q)$ mask|
|packed batch|不同样本之间无连接|sequence boundary、block diagonal|
|数值稳定|无 NaN、无全 mask 静默输出|sentinel、softmax kernel|
|执行成本|区分逻辑连接与实际 kernel FLOPs|dense、tiled、FlashAttention|

## 相关词条

- [自注意力](../attention/self-attention/)
- [缩放点积注意力](../attention/scaled-dot-product/)
- [Attention 矩阵](../attention/attention-matrix/)
- [多头注意力](../attention/multi-head-attention/)
- [GQA 与 MQA](../attention/gqa-and-mqa/)
- [交叉注意力](../attention/cross-attention/)
- [注意力复杂度](../attention/attention-complexity/)
- [稀疏注意力](../attention/sparse-attention/)
- [FlashAttention](../attention/flash-attention/)
