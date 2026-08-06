---
title: "交叉注意力：为什么 query 读另一条序列的 K/V"
tags: ["why-models-learn"]
---

Cross-Attention（交叉注意力）让一条序列产生 query，去读取另一条序列产生的 key/value。经典 encoder-decoder 中，decoder 的目标位置产生 query，encoder 的源序列产生 key/value；因此 attention matrix 的行数是目标长度 $T$，列数是源长度 $S$。交叉注意力不自动使用 decoder 的因果下三角，通常只屏蔽 source padding；是否限制 source 的可见范围由任务的连接契约另行决定。本篇固定两条序列的方向、推导数值与梯度路径，再处理 decoder block、静态 KV cache、mask 和多头实现。

![目标序列产生 query、源序列产生 key/value，交叉注意力形成目标长度乘源长度的读取矩阵](/assets/attention/svg/cross-attention.1.svg)

## 先固定两条序列的方向

### query 来自目标，key/value 来自源

设目标序列隐藏状态和源序列隐藏状态分别为：

$$
\begin{gathered}
H_{\mathrm{tgt}}\in\mathbb R^{B\times T\times d_{\mathrm{tgt}}},\\
H_{\mathrm{src}}\in\mathbb R^{B\times S\times d_{\mathrm{src}}}.
\end{gathered}
$$

交叉注意力用两套输入产生 Q、K、V：

$$
\begin{gathered}
Q=H_{\mathrm{tgt}}W_Q,\\
K=H_{\mathrm{src}}W_K,\\
V=H_{\mathrm{src}}W_V.
\end{gathered}
$$

在拆出 head 后，第 $q$ 个 head 的形状为：

$$
\begin{gathered}
Q^{(q)}\in\mathbb R^{B\times T\times d_k},\\
K^{(q)}\in\mathbb R^{B\times S\times d_k},\\
V^{(q)}\in\mathbb R^{B\times S\times d_v}.
\end{gathered}
$$

因此 score、attention weight 和 context 的形状为：

$$
\begin{gathered}
S^{(q)}=Q^{(q)}(K^{(q)})^\mathsf T\in\mathbb R^{B\times T\times S},\\
A^{(q)}\in\mathbb R^{B\times T\times S},\\
C^{(q)}=A^{(q)}V^{(q)}\in\mathbb R^{B\times T\times d_v}.
\end{gathered}
$$

行仍是 query，列仍是 key；变化在于 query 和 key/value 来自不同的序列。

### self-attention 与 cross-attention 的形状差异

|注意力类型|Q 来源|K/V 来源|矩阵形状|常见位置约束|
|---|---|---|---|---|
|self-attention|同一序列|同一序列|$T\times T$|decoder self-attention 常用 causal mask|
|cross-attention|目标序列|源序列|$T\times S$|source padding mask，通常不加 target causal mask|
|encoder self-attention|源序列|源序列|$S\times S$|双向或任务指定的 source mask|

不要因为实现张量都叫 hidden state，就把 $T$ 和 $S$ 当成同一个长度。翻译、摘要、语音和视觉文本模型中，两条序列长度通常不同。

### 缩放、mask 和输出投影

每个 head 的标准计算为：

$$
\begin{gathered}
S^{(q)}
=
\frac{Q^{(q)}(K^{(q)})^\mathsf T}{\sqrt{d_k}},\\
A^{(q)}
=
\operatorname{softmax}_{\mathrm{row}}
\left(S^{(q)}+M^{(q)}\right),\\
C^{(q)}
=
A^{(q)}V^{(q)}.
\end{gathered}
$$

多个 head 沿目标位置的特征轴拼接，再经过输出投影：

$$
Y=
\operatorname{Concat}
\left(C^{(0)},\ldots,C^{(h-1)}\right)W_O.
$$

mask 的最后两个轴对应目标 query 和源 key。source padding 位于列轴，目标 padding 位于行轴；两者不能用同一个一维 mask 变量混淆。

## 一个两目标、三源位置的数值例子

### 先给出目标到源的 score

取两个目标 query 和三个源 key：

$$
S=
\begin{bmatrix}
2&0&-1\\
0&2&1
\end{bmatrix}.
$$

每一行对应一个目标位置，每一列对应一个源位置。这里不使用 causal mask，两个目标 query 都可以读取三个源位置。

令源 value 是二维向量：

$$
V=
\begin{bmatrix}
10&0\\
20&5\\
40&10
\end{bmatrix}.
$$

### row-wise softmax 产生两个不同的源读取

逐行 softmax 得到：

$$
A\approx
\begin{bmatrix}
0.843794734&0.114195199&0.042010066\\
0.090030573&0.665240956&0.244728471
\end{bmatrix}.
$$

每一行仍然和为 $1$，但两行的读取分布不同。context 为：

$$
C=AV\approx
\begin{bmatrix}
12.402253978&0.991076658\\
23.994263689&5.773489489
\end{bmatrix}.
$$

第一个目标位置主要读取源位置 $0$，第二个目标位置主要读取源位置 $1$。attention weight 表示当前 query 对源位置的软读取，不表示源 token 被复制到目标位置。

### source padding mask 只屏蔽列

假设源位置 $2$ 是 padding，把第三列 score 设为不可用：

$$
M_{\mathrm{src}}
=
\begin{bmatrix}
0&0&-\infty\\
0&0&-\infty
\end{bmatrix}.
$$

新的权重为：

$$
A_{\mathrm{masked}}\approx
\begin{bmatrix}
0.880797078&0.119202922&0\\
0.119202922&0.880797078&0
\end{bmatrix}.
$$

对应的输出为：

$$
C_{\mathrm{masked}}\approx
\begin{bmatrix}
11.192029220&0.596014610\\
18.807970780&4.403985390
\end{bmatrix}.
$$

source padding mask 改变了每个目标 query 的合法列集合；它没有把目标 query 行删掉。目标 padding 行仍需要独立的 query mask 或 loss mask。

## encoder-decoder 中的完整信息路径

### encoder 先建立源表示

以机器翻译为例，源句子先经过 source embedding 和 encoder layers：

$$
x_{0:S-1}
\longrightarrow
H_{\mathrm{src}}
\longrightarrow
K,V.
$$

这里的 K/V 编码源位置的内容和可匹配特征。它们不包含 decoder 当前步骤才产生的目标 token。

### decoder self-attention 先处理目标前缀

decoder 输入是右移后的目标前缀。decoder self-attention 使用 causal mask：

$$
y_{<t}
\longrightarrow
H_{\mathrm{tgt}}^{(t)}
\longrightarrow
Q.
$$

它保证目标位置 $t$ 的 query 只使用目标前缀信息。随后 cross-attention 用这个 query 读取完整有效源序列：

$$
Q_{\mathrm{tgt}}
\longleftrightarrow
(K_{\mathrm{src}},V_{\mathrm{src}}).
$$

一个 decoder layer 的常见顺序为：

|子层|query 来源|key/value 来源|主要 mask|
|---|---|---|---|
|decoder self-attention|目标前缀|目标前缀|target causal、target padding|
|cross-attention|self-attention 后的目标状态|encoder 输出|source padding，任务指定的 source visibility|
|feed-forward|cross-attention 后的目标状态|逐位置变换|query padding 或 loss mask|

Pre-Norm、Post-Norm 和并行 residual 会改变计算顺序，但不会改变 cross-attention 的 Q/K/V 来源定义。

### teacher forcing 不改变 cross-attention 的源方向

训练时 teacher forcing 提供真实的前一个目标 token，推理时使用模型自己生成的前一个 token。两种情况下，cross-attention 的 K/V 都来自 source encoder 输出；差异在目标 query 的前缀状态：

|阶段|目标 query 的来源|源 K/V|结果|
|---|---|---|---|
|teacher forcing|真实目标前缀|固定 source encoding|并行计算目标位置|
|free-running decode|已生成目标前缀|固定 source encoding|逐步生成目标 token|

因此可以在同一个 source 上比较两种目标 query 的 attention map。map 差异可能来自目标前缀不同，不能归因于 source K/V 被重新计算。

## 为什么 cross-attention 通常不加 target causal mask

### target 位置不是 source 位置

causal mask 的条件是“query 位置 $t$ 不能读取未来 key 位置 $s>t$”。在 cross-attention 中，$t$ 属于目标序列，$s$ 属于源序列；两者没有共同的时间索引。把 $s\le t$ 直接套到 $T\times S$ 矩阵，会人为禁止目标位置读取后面的源 token。

翻译中，目标词的正确对应关系可能位于源句子的任意位置。摘要中，当前目标 token 也可能需要源文档后部的信息。只要整个 source 在生成前已知，目标 query 可以读取所有有效 source key。

### source 可见性由任务另行定义

以下情况可能需要额外 source mask：

- source padding；
- streaming encoder 尚未产生的 source prefix；
- 多模态输入中的模态或区域可见性；
- 指定对齐窗口或单调读取；
- packed source 中不同样本的边界。

这些 mask 应写成目标行到源列的连接条件，不应被称为 target causal mask。若 source 是在线产生的，允许条件可能是：

$$
\operatorname{allow}(t,s)
=
\mathbf 1\{s\le a_t\},
$$

其中 $a_t$ 是目标位置 $t$ 当前可见的 source prefix 长度。它是任务的 source availability 约束。

### cross-attention 的对角线没有特殊地位

self-attention 中主对角线表示 query 读取自己的 key。cross-attention 的 query 和 key 属于不同序列，即使 $T=S$，索引相同也不意味着语义对应。因此默认没有“必须保留对角线”的规则。是否允许 $(t,t)$ 取决于 source-target 对齐任务，而不是矩阵形状。

## mask 的轴与 padding 细节

### source padding mask 广播到列轴

若 valid_source(s) 表示源位置是否有效，source padding mask 的逻辑条件为：

$$
\operatorname{allow}_{\mathrm{src}}(t,s)
=
\operatorname{valid\_source}(s).
$$

它可以从：

$$
[B,S]
\longrightarrow
[B,1,1,S]
\longrightarrow
[B,h,T,S]
$$

广播到 batch、head 和目标 query 轴。最后一维必须对齐 source key 位置。

### target padding mask 作用在行轴

target padding 不应作为合法 query 参与有效输出或 loss。若 target query mask 为 $m_{\mathrm{tgt}}\in\{0,1\}^{B\times T}$，可以在输出或 loss 中使用：

$$
\operatorname{valid\_query}(t)
=
m_{\mathrm{tgt}}(t).
$$

只把 target padding mask 广播到最后一维，会错误地把它当作 source key mask。检查一个 $T\ne S$ 的 batch 可以快速暴露这种轴错位。

### source 和 target 都有 padding 时

最终连接可以写成：

$$
\operatorname{allow}(t,s)
=
\operatorname{valid\_query}(t)
\land
\operatorname{valid\_source}(s)
\land
\operatorname{visible}(t,s).
$$

其中 visible 可以是全 source、source prefix、对齐窗口或 packed boundary。query 为 padding 时，即使形式上存在合法 source，也通常不应把该行计入训练目标。

### 全 source 被屏蔽的行

如果 source 为空、source 全是 padding，或 source availability mask 没有给当前目标 query 留下合法位置，某一行会全 mask。和 causal attention 一样，需要显式策略：

- 返回定义好的零 context；
- 跳过该目标 query 的 loss；
- 把空 source 作为输入错误拒绝；
- 使用任务规定的 null source token。

不能依靠有限负数 sentinel 让全 mask 行自然产生合理输出。

## 反向传播跨过两条序列

### query 梯度回到目标序列

cross-attention 的 query 投影来自目标 hidden，因此：

$$
\frac{\partial\mathcal L}{\partial W_Q}
=
H_{\mathrm{tgt}}^\mathsf T
\frac{\partial\mathcal L}{\partial Q}.
$$

目标序列还会通过 decoder self-attention 和 residual 收到其他梯度。cross-attention 只是其中一条从 source 到 target 的连接。

### key/value 梯度回到源序列

所有目标 query 行都读取同一份 source K/V。因此 source 的 K/V 梯度沿目标行聚合：

$$
\begin{gathered}
\frac{\partial\mathcal L}{\partial K_s}
=
\sum_{t=0}^{T-1}
\left.
\frac{\partial\mathcal L}{\partial K_s}
\right\rvert_{t},\\
\frac{\partial\mathcal L}{\partial V_s}
=
\sum_{t=0}^{T-1}
\left.
\frac{\partial\mathcal L}{\partial V_s}
\right\rvert_{t}.
\end{gathered}
$$

如果 source 位置被 padding mask 屏蔽，它不应通过该 cross-attention 行收到有效的 score/value 梯度。source encoder 还可能从 source-side loss 或其他 decoder layers 收到梯度。

### source 表示承载多个目标位置的训练信号

在一个目标序列中，同一个 source token 可能被多个 query 读取。它的 encoder 表示会接收这些目标位置贡献的总梯度。长 source、重复目标读取和尖锐 attention 会改变梯度聚合的分布，不能只看单个 attention weight 判断 source token 的训练重要性。

## cross-attention 的 KV cache

### source KV 可以预先计算

自回归 decoder 每一步产生新的目标 query，但 source encoder 输出在生成期间固定。可以先计算并保存：

$$
\begin{gathered}
K_{\mathrm{src,cache}}
\in
\mathbb R^{B\times h_{kv}\times S\times d_k},\\
V_{\mathrm{src,cache}}
\in
\mathbb R^{B\times h_{kv}\times S\times d_v}.
\end{gathered}
$$

每个 decoder step 只需生成当前目标 query，再读取静态 source cache。单层的 source KV 元素数为：

$$
N_{\mathrm{src,KV}}
=
B S h_{kv}(d_k+d_v).
$$

这与 decoder self-attention 的历史 KV cache 不同：

|cache 类型|K/V 来源|随生成长度变化|典型用途|
|---|---|---|---|
|decoder self KV|已生成目标前缀|按目标长度增长|避免重复计算 target prefix|
|cross-attention KV|固定 source encoder 输出|生成期间不增长|避免每步重新投影 source|

两种 cache 可能同时存在，不能用同一个长度变量覆盖。

### 一个 source KV 资源账本

取 $B=1$、$h_{kv}=32$、$S=4096$、$d_k=d_v=128$，FP16 每个元素占 2 bytes：

|项目|结果|
|---|---:|
|source K/V 元素数|$33,554,432$|
|FP16 字节数|$67,108,864$|
|容量|$64$ MiB|

这是单个 decoder layer 的 source KV cache。共享 K/V 的 GQA/MQA 可以把 $h_{kv}$ 降低，但每个目标 query 仍可能产生 $h_q$ 个 score 行。

### batch 和 beam search 会改变 cache 组织

不同 source 样本通常不能共享 K/V。beam search 可以让多个 beam 共享同一份 source cache 的物理存储，也可以复制 cache，取决于实现。检查：

- beam 是否只复制 target-side state；
- source cache 是否按 beam 视图广播；
- reorder beam 时是否错误重排了 source batch；
- 结束 beam 是否继续占用 source cache。

逻辑上，每个 beam 都有相同 source key/value；物理上可以使用引用计数、视图或复制。

## 多头、GQA 和 cross-attention

### 每个 query head 可以读取 source head

标准 multi-head cross-attention 的 query、key、value 都在各自的输入序列上投影。若使用 GQA/MQA，query head 数 $h_q$ 可以大于 source K/V head 数 $h_{kv}$：

$$
\begin{gathered}
Q\in\mathbb R^{B\times h_q\times T\times d_k},\\
K,V\in\mathbb R^{B\times h_{kv}\times S\times d_k/d_v}.
\end{gathered}
$$

第 $q$ 个目标 query head 使用 source K/V 组 $g(q)$。query 的序列来源与 K/V 的序列来源不同，但 head 映射规则仍然可以沿用。

### mask 的 head 轴仍按 query 展开

如果 source visibility 对所有 head 相同，可以使用：

$$
M\in\mathbb R^{B\times1\times T\times S}
$$

并广播到 $h_q$ 个 query head。若某些 head 使用不同的 source window，则使用：

$$
M\in\mathbb R^{B\times h_q\times T\times S}.
$$

K/V 的 head 数减少，不表示 mask 的 head 轴应该直接改成 $h_{kv}$。mask 约束的是 query 到 source key 的连接。

### output projection 回到目标宽度

cross-attention 输出的 token 轴仍是目标长度 $T$，因为每个目标 query 得到一个 context：

$$
C\in\mathbb R^{B\times T\times h_qd_v}
\longrightarrow
Y\in\mathbb R^{B\times T\times d_{\mathrm{model}}}.
$$

不要把输出误 reshape 成源长度 $S$。source 的长度只决定每个 query 能读取多少列，不决定输出 token 的数量。

## 计算量与实现轴

### score 和 value 计算分别依赖 $T$ 与 $S$

对 $B$ 个 batch、$h$ 个 head，cross-attention 的逻辑 attention matrix 元素为：

$$
N_A=B h T S.
$$

QK 点积和 AV 加权的维度项分别为：

$$
\begin{gathered}
N_{QK}=B h T S d_k,\\
N_{AV}=B h T S d_v.
\end{gathered}
$$

取 $B=2$、$h=16$、$T=128$、$S=512$、$d_k=d_v=64$：

|项目|结果|
|---|---:|
|attention matrix 元素|$2,097,152$|
|QK 点积维度项|$134,217,728$|
|AV 加权维度项|$134,217,728$|
|source K/V 元素|$2,097,152$|
|source K/V FP16 字节数|$4,194,304$|

当 $T\ll S$ 时，source 长度主导每个 target query 的读取成本；当 $S\ll T$ 时，目标 query 数主导 score 行数。不能用一个统一的序列长度替代两者。

### 常见布局变换

合并投影后，目标 Q 和源 K/V 的布局可能是：

$$
\begin{gathered}
[B,T,h_qd_k]\to[B,h_q,T,d_k],\\
[B,S,h_{kv}d_k]\to[B,h_{kv},S,d_k],\\
[B,S,h_{kv}d_v]\to[B,h_{kv},S,d_v].
\end{gathered}
$$

如果输入 batch 使用不同 source 长度，padding 或 varlen metadata 必须同时传给 cross-attention kernel。Q 的第二个序列轴是 $T$，K/V 的第二个序列轴是 $S$。

### source order 是内容语义的一部分

cross-attention 只能读取传入的 source 顺序和表示。如果 tokenizer、padding、pack/unpack、encoder position id 或 beam reorder 改变 source 位置，attention map 的列含义也随之改变。矩形形状仍可能正确，但读取内容已经错位。

因此对 source 做截断、拼接或重排时，必须同步验证：

- source token 到 encoder row 的映射；
- source padding 位置；
- source position id；
- KV cache 的写入和读取索引；
- 评估中显示 attention map 的列标签。

## 失效模式与审计方法

### 把 Q 和 K/V 的来源交换

若让 source 产生 Q、target 产生 K/V，矩阵形状可能仍然是某个合法的矩形，但每一行的语义会改变。先用 $T\ne S$ 的例子检查 query 行数和输出长度。

### 把 cross-attention 当成 target causal attention

目标位置 $t$ 通常可以读取所有有效 source 位置。把 $s\le t$ 套到 source 列会截断合法 source 信息。只有 source availability 或单调对齐明确要求时，才添加相应限制。

### 把 source padding mask 广播到行轴

source padding 应屏蔽列。如果将 [B,S] 误广播为 query 行 mask，可能把某些 target query 整行置零，或在 $T\ne S$ 时触发隐式广播。

### 只缓存 target self-attention 的 K/V

decoder 生成时，source K/V 也可以静态缓存。若每一步重新执行 source projection，输出可能正确但延迟和内存搬运增加。分别记录 target-growing cache 和 source-static cache。

### 用 $T$ 计算 source KV cache

cross-attention cache 的序列长度是 $S$。将其写成 $B h S(d_k+d_v)$；不能使用目标长度 $T$，也不能把 self-attention 的历史长度直接复用。

### 忘记 target padding query

source 列屏蔽后，target padding 行仍可能得到合法 context。检查 target mask、loss reduction、residual 和 decoder 输出是否忽略这些行。

### source cache 与 beam reorder 错位

beam 重新排序时只重排 target state，而 source cache 仍按原 batch index 读取，会让一个 beam 读取另一个 source。用两个不同 source 的 batch 和一次 beam reorder 检查。

### K/V 组映射顺序错误

在 GQA cross-attention 中，query head 的组映射必须与 source K/V head 的布局一致。用人工映射表检查连续分组和交错分组，不能只检查形状。

### 空 source 或全 mask 行静默传播

全 source 被屏蔽时，有限 sentinel 可能产生均匀分布，$-\infty$ 可能产生 NaN。为零长度、全 padding 和 source availability 为空的情况定义明确输出。

### 用一张 attention map 证明 source 因果关系

cross-attention weight 展示的是目标到源的读取分布，不自动证明 source token 对输出具有因果作用。还要做 source token 扰动、mask 对照、梯度或输出干预。

### 一份最小 cross-attention 审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|Q 来源|目标序列 hidden|decoder sublayer 顺序|
|K/V 来源|源序列 encoder hidden|source row 对齐|
|矩阵方向|行是目标 query，列是源 key|QK transpose|
|形状|$B\times h\times T\times S$|误用 $T=S$|
|softmax 轴|沿 source key 列归一化|axis、mask 广播|
|source padding|无效 source 列不可读|最后一维 mask|
|target padding|无效 target 行不进入有效输出或 loss|query mask、reduction|
|target causal|默认不添加|误套下三角|
|source visibility|按任务指定全 source 或 prefix|streaming、对齐窗口|
|输出 token 轴|保持目标长度 $T$|错误 reshape 到 $S$|
|source KV cache|容量按 $B S h_{kv}(d_k+d_v)$|误用 $T$、重复投影|
|beam reorder|target state 和 source cache 对齐|batch index|
|数值稳定|无 NaN、无全 mask 静默输出|sentinel、softmax kernel|
|梯度路径|Q 回目标，K/V 汇聚到源|detach、错误广播|

## 相关词条

- [自注意力](../attention/self-attention/)
- [缩放点积注意力](../attention/scaled-dot-product/)
- [Attention 矩阵](../attention/attention-matrix/)
- [多头注意力](../attention/multi-head-attention/)
- [GQA 与 MQA](../attention/gqa-and-mqa/)
- [因果掩码](../attention/causal-masking/)
- [Bahdanau 注意力](../rnn-lstm/bahdanau-attention/)
- [注意力作为软检索](../attention/attention-as-retrieval/)
- [注意力复杂度](../attention/attention-complexity/)
