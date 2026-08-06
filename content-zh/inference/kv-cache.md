---
title: "KV cache：用历史 K/V 换取增量解码"
tags: ["why-models-learn"]
---

KV cache 是自回归推理中跨 decode step 保存历史 key/value 投影的运行时状态。当前 query 读取这些历史 K/V 后就能完成 attention，不必在每一步重新投影已经处理过的 token。它不保存 query，也不等于训练时的 attention map；它按层、按请求、按 K/V head 和历史长度增长。本篇固定 cache 的张量合同、复杂度、显存公式、物理布局、批处理、前缀复用、beam 重排、精度和淘汰边界。

![KV cache 示意图：prefill 把 prompt 的 K/V 写入按层缓存，decode 读取历史行并追加新位置；下方对比 query head、K/V head 和缓存字节](/assets/inference/svg/kv-cache.1.svg)

## 先固定 cache 保存的对象

### K/V 来自每一层的投影

设 decoder 的 hidden width 为 $D$，batch size 为 $B$，query head 数为 $h_q$，K/V head 数为 $h_{kv}$，每个 head 的宽度为 $d_h$。在常见的均匀切分中：

$$
D=h_qd_h.
$$

一层输入 $X_\ell$ 的形状是：

$$
X_\ell\in\mathbb R^{B\times T\times D}.
$$

Q、K、V 投影后分别整理为：

$$
\begin{aligned}
Q_\ell&\in\mathbb R^{B\times h_q\times T\times d_h},\\
K_\ell&\in\mathbb R^{B\times h_{kv}\times T\times d_h},\\
V_\ell&\in\mathbb R^{B\times h_{kv}\times T\times d_h}.
\end{aligned}
$$

MHA 取 $h_{kv}=h_q$。GQA 和 MQA 只减少 K/V head；Q 仍然保留 $h_q$ 个独立视角。输入投影、输出投影和 head 的排列约定必须与模型 checkpoint 一致，不能只凭最终张量形状猜测。

推理开始后，cache 只保存已经处理过的位置的 K/V。若某层已有 $L$ 个历史位置：

$$
\mathcal C_\ell(L)
=
\left(
K_{\ell}^{\mathrm{cache}},
V_{\ell}^{\mathrm{cache}}
\right),
\qquad
K_{\ell}^{\mathrm{cache}},V_{\ell}^{\mathrm{cache}}
\in
\mathbb R^{B\times h_{kv}\times L\times d_h}.
$$

当前 decode 只产生新位置的 query、key 和 value。若一次追加 $R$ 个位置：

$$
\begin{aligned}
Q_{\ell}^{\mathrm{new}}&\in\mathbb R^{B\times h_q\times R\times d_h},\\
K_{\ell}^{\mathrm{new}},V_{\ell}^{\mathrm{new}}&\in\mathbb R^{B\times h_{kv}\times R\times d_h}.
\end{aligned}
$$

写入后，逻辑 cache 长度变为 $L+R$。Q 不需要跨步骤保存，因为下一次 attention 只读取下一次的新 query。

### K/V head 共享不等于 query head 共享

令 GQA 的每组 query head 数为：

$$
r=\frac{h_q}{h_{kv}},
\qquad
g(q)=\left\lfloor\frac{q}{r}\right\rfloor,
\qquad
0\le q<h_q.
$$

第 $q$ 个 query head 使用第 $g(q)$ 个 K/V head：

$$
\widetilde K_{\ell}^{(q)}
=
K_{\ell}^{(g(q))},
\qquad
\widetilde V_{\ell}^{(q)}
=
V_{\ell}^{(g(q))}.
$$

实现可以通过广播、gather 或 tile 内复用完成这个读取。它不需要在物理 cache 中复制 $r$ 份 K/V。每个 query head 仍然独立计算自己的 score 和 attention weight：

$$
A_{\ell}^{(q)}
=
\operatorname{softmax}
\left(
\frac{Q_{\ell}^{(q)}
\left(\widetilde K_{\ell}^{(q)}\right)^\mathsf T}
{\sqrt{d_h}}
+
M^{(q)}
\right).
$$

因此，减少 $h_{kv}$ 会减少 cache 元素和 K/V 读取量；它不会自动把 score head 数从 $h_q$ 改成 $h_{kv}$。[GQA 与 MQA](../attention/gqa-and-mqa/)展开了 head 映射、参数量和共享后的梯度路径。

### cache 不是 attention map

三个对象的生命周期不同：

|对象|形状中的长度轴|生命周期|是否跨 decode step 保存|
|---|---|---|---|
|Q|当前 query 长度 $R$|一次前向|否|
|K/V cache|历史长度 $L$|请求从 prefill 到结束|是|
|attention score 或 weight|当前 query 与可见 key|一次 attention|通常否|
|hidden state|当前前向的 token 数|一次或少量 checkpoint|通常否|

训练时可能物化完整的 $T\times T$ attention map。增量推理只需要当前 query 读取历史 K/V；两者都涉及 attention，但显存账本不能相加后再称为 cache。[注意力复杂度](../attention/attention-complexity/)分别核算了交互、物化矩阵和 KV cache。

## 为什么 cache 能降低重复计算

### 不使用 cache 时会反复处理历史

设 prompt 长度为 $P$，生成 $U$ 个新 token。若每次生成都把完整前缀重新送入 decoder，第 $u$ 次前向的长度为：

$$
T_u=P+u,
\qquad
0\le u<U.
$$

只计因果 self-attention 的逻辑位置交互，完整重算的数量为：

$$
N_{\mathrm{naive}}
=
h_q
\sum_{u=0}^{U-1}
\frac{T_u(T_u+1)}{2}.
$$

这个数包含每次前向对所有历史位置的 QK 交互。它还会重新计算历史 token 的 K/V 投影、FFN、归一化和其他中间量。

### 使用 cache 后只追加新位置

prefill 一次处理 prompt，因果 attention 的位置交互为：

$$
N_{\mathrm{prefill}}
=
h_q\frac{P(P+1)}{2}.
$$

之后每个 decode step 只有一个新 query，但它要读取当前完整历史。累计交互为：

$$
\begin{aligned}
N_{\mathrm{decode}}
&=
h_q\sum_{u=0}^{U-1}(P+u)\\
&=
h_q\left(UP+\frac{U(U-1)}{2}\right).
\end{aligned}
$$

因此 cache 路径的这部分总量是：

$$
N_{\mathrm{cache}}
=
N_{\mathrm{prefill}}+N_{\mathrm{decode}}.
$$

这不是把 attention 变成常数。每一步仍要读取随历史长度增长的 K/V，并计算新 query 的 score、weight 和 context。cache 删除的是历史 K/V 投影的重复工作，以及重新运行历史 token 的其他前向工作；它没有删除当前 token 的前向。

取 $P=4$、$U=3$、$h_q=8$。完整重算的三次长度为 $4,5,6$：

|路径|计数表达式|位置交互数|
|---|---|---:|
|prefill|$8(4\cdot5/2)$|80|
|cache decode|$8(4+5+6)$|120|
|cache 总量|$80+120$|200|
|每步完整重算|$8(4\cdot5/2+5\cdot6/2+6\cdot7/2)$|368|

这些数字只统计 attention 的位置交互。实际 wall-clock 时间还取决于 Q/K/V 投影、FFN、输出头、kernel 融合、内存带宽和调度。不能用 200 与 368 直接预测某一块 GPU 上的延迟。

### 第一个 token 的边界要明确

有两种常见约定：

1. prefill 最后位置的 logits 直接选择第一个生成 token，然后把该 token 写入 cache；
2. prefill 只建立 prompt cache，再额外执行一次 decode 得到第一个生成 token。

两种约定都可以实现，但 decode 次数、首 token 延迟和最终 cache 长度的日志会不同。审计时要记录：

|字段|需要固定的选择|
|---|---|
|prompt cache|prompt 最后一个位置是否已经写入|
|首 token|使用 prefill logits 还是额外 decode|
|追加时机|采样前还是采样后写入新 K/V|
|终止 token|EOS 是否写入 cache 和返回 token 序列|
|长度计数|生成预算按 token、字符还是请求时间计算|

如果把 prefill logits 生成的 token 错当作已经过了一次 decode，却又额外重复输入同一 token，结果会多生成一步，并可能把 cache 长度和 position offset 同时推移一位。

## cache 的生命周期由位置轴决定

### 空 cache 进入 prefill

请求开始时，每层 cache 都是空的。prefill 读取 prompt 的 token 和 mask，计算所有 prompt 位置的 K/V，再按逻辑位置 $0,\ldots,P-1$ 写入 cache：

$$
\begin{aligned}
K_{\ell}^{\mathrm{cache}}&\leftarrow K_{\ell,0:P},\\
V_{\ell}^{\mathrm{cache}}&\leftarrow V_{\ell,0:P}.
\end{aligned}
$$

这里的冒号只表示半开区间索引，不是数学条件。prefill 期间的 query 可以是完整 prompt 的所有位置；因果 mask 仍然限制每个位置只能读取自己和之前的位置。

### decode 追加当前 token

假设当前 cache 长度是 $L$，一次 decode 产生一个新位置。新 K/V 应写入逻辑位置 $L$：

$$
\begin{aligned}
K_{\ell}^{\mathrm{cache}}[L]&\leftarrow K_{\ell}^{\mathrm{new}}[0],\\
V_{\ell}^{\mathrm{cache}}[L]&\leftarrow V_{\ell}^{\mathrm{new}}[0],\\
L&\leftarrow L+1.
\end{aligned}
$$

如果一次 chunk prefill 追加 $R$ 个位置，写入位置是 $L,\ldots,L+R-1$。不能先把长度更新成 $L+R$，再用新长度计算当前 query 的 position；这样会让 position offset 多出一个 chunk。

一个请求的最小生命周期可以写成：

|时刻|逻辑 cache 内容|要记录的证据|
|---|---|---|
|请求开始|每层长度为 0|请求 ID、模型版本、是否命中前缀|
|prefill 完成|prompt 的 K/V|prompt token 数、位置起点、实际长度|
|decode 第 $u$ 步|历史加上当前前缀|query 长度、cache 长度、mask offset|
|追加完成|新 token 的 K/V 已写入|追加位置、序列 ID、停止状态|
|请求结束|释放、归还或进入共享池|释放时间、物理 block、租户边界|

### cache offset 必须进入 causal mask

设已有历史长度为 $L$，本次 chunk 有 $R$ 个新 query。拼接后的 key/value 长度是 $L+R$。新 query 的局部索引为 $i$，它的绝对位置是 $L+i$；拼接 key 的局部索引为 $j$。可见条件是：

$$
M_{ij}
=
\begin{cases}
0,&j\le L+i,\\
-\infty,&j>L+i,
\end{cases}
\qquad
0\le i<R,\quad
0\le j<L+R.
$$

当 $R=1$ 时，新 query 可以读取全部 $L$ 个历史位置和当前新位置。把这张矩阵错误地当作从位置零开始的普通下三角，会让它只读取很短的前缀，或者把历史位置误判为未来位置。[因果掩码](../attention/causal-masking/)给出下三角、padding 和 chunk prefill 的完整 mask 推导。

### position ID 和 RoPE 只应用一次

cache 的第 $t$ 行必须对应逻辑位置 $t$，或者对应明确的窗口位置映射。对于 RoPE，有两种合法实现：

- 在写入 cache 前把 K 旋转到绝对位置，并在读取时只旋转当前 Q；
- 保存未旋转 K，在 attention kernel 内使用同一 position offset 旋转 Q 和 K。

两种实现不能混用。已经旋转过的历史 K 再次按相同 position 旋转，会改变相位。重排 cache 时也不能重排 token 而忘记同步 position ID。

一个最小一致性测试是：固定同一 prompt，分别执行一次 full prefill 和逐 token decode，比较每个新位置的 logits。差异应在明确 dtype 和容差范围内；若差异在第一步就出现，优先检查 mask、position offset、RoPE 以及 cache 的 K/V 写入时机。

## 显存公式把 cache 变成可核对的数字

### 每个元素占多少字节

设模型有 $L_{\mathrm{layer}}$ 个 decoder layer，每个 K/V 标量占 $b$ bytes。若 batch 中每条序列都使用长度 $T$，cache 中有 K 和 V 两份张量：

$$
N_{\mathrm{KV}}
=
2L_{\mathrm{layer}}BTh_{kv}d_h.
$$

对应的字节数为：

$$
S_{\mathrm{KV}}
=
2L_{\mathrm{layer}}BTh_{kv}d_hb.
$$

这里的第一个 2 来自 K 和 V。它不是“两个 attention head”，也不是 FP16 的字节数。FP16 或 BF16 通常取 $b=2$；FP32 取 $b=4$；量化 cache 需要再加 scale、zero point、对齐和元数据。

如果每条序列的有效长度不同，不能直接把最大长度乘以 batch 作为有效 cache。理想的有效字节数是：

$$
S_{\mathrm{KV,varlen}}
=
2L_{\mathrm{layer}}h_{kv}d_hb
\sum_{i=1}^{B}T_i.
$$

预分配矩形 buffer 可能仍按 $BT_{\max}$ 占用。报告时要同时给出逻辑有效容量和物理分配容量。

### MHA、GQA 和 MQA 的 cache 比较

取 $L_{\mathrm{layer}}=32$、$B=1$、$T=2048$、$d_h=64$、FP16。固定 $h_q=16$，只改变 K/V head：

|配置|$h_{kv}$|K/V 元素总数|FP16 cache|
|---|---:|---:|---:|
|MHA|16|134,217,728|256 MiB|
|GQA-4|4|33,554,432|64 MiB|
|MQA|1|8,388,608|16 MiB|

这个表只计算 target self-attention 的动态 cache，不包含模型权重、激活、workspace、量化 scale 或 source cache。GQA-4 相对 MHA 将 K/V head 降为四分之一，cache 也降为四分之一；query head 仍有 16 个，score 的逻辑 head 数不随这张表变化。

### batch、beam 和生成长度都会乘上去

固定模型层数、head 数和 dtype 后，cache 对以下变量线性增长：

$$
S_{\mathrm{KV}}
\propto
L_{\mathrm{layer}}BTh_{kv}d_hb.
$$

因此：

- 并发请求数翻倍，若每条长度不变，cache 近似翻倍；
- 生成长度翻倍，单条请求的动态 cache 近似翻倍；
- beam 数并入有效 batch 后，cache 近似乘以 beam 数；
- GQA/MQA 减少 $h_{kv}$，但不改变 query 的 score head 数；
- 量化降低每个 K/V 元素的字节数，但会增加 scale 和反量化工作。

“模型能否装入显存”必须把固定的权重账和请求级的 cache 账分开。[参数量与资源账本](../transformer-components/parameter-count/)负责参数、激活、optimizer state 和运行时状态的边界。

### decode 的读取量和 score 数量不是同一个量

单个 decode query、单层、单条序列、历史长度为 $L$ 时，逻辑读取和 score 项可以分别写成：

$$
\begin{aligned}
N_{\mathrm{KV-read}}&=2Lh_{kv}d_h,\\
N_{\mathrm{score}}&=h_qL.
\end{aligned}
$$

乘以 dtype 字节数后，第一项给出 K/V 原始读取的下界；第二项给出 query-key 位置对数。实际 kernel 会通过 tile、共享内存和向量化复用 K/V，实际 DRAM 流量不一定等于这个下界。它仍然说明两个结构事实：

1. $h_{kv}$ 直接进入 cache 容量和 K/V 读取；
2. $h_q$ 直接进入 score 的逻辑数量。

如果 profiler 报告的 score head 数等于 $h_{kv}$，应检查它是否改变了 query 到 K/V 的连接规则，而不是只把 GQA 的命名当作证据。

## 物理布局决定寻址和碎片

### 逻辑轴和物理轴必须分开

论文和公式常写：

$$
K_{\ell}^{\mathrm{cache}}
\in
\mathbb R^{B\times h_{kv}\times T\times d_h}.
$$

实现也可能把内存布局写成：

- $[B,T,h_{kv},d_h]$，按 token 连续；
- $[B,h_{kv},T,d_h]$，按 head 连续；
- $[\text{block},\text{slot},h_{kv},d_h]$，按分页块存储；
- 将 batch、beam 或 sequence slot 合并到第一个物理轴。

这些布局在数学上可以表达同一个 cache，但 transpose、stride 和 kernel 的读取模式不同。审计不只检查 shape，还要检查每个逻辑位置最终落在哪个物理 slot：

|逻辑对象|需要固定的轴|常见错误|
|---|---|---|
|batch 或 request|请求 ID、sequence ID、物理 slot|重排后读到另一个请求|
|K/V head|$h_{kv}$|把 query head 当作 K/V head|
|位置|逻辑 token index、cache offset|追加到旧位置或跳过一行|
|head dimension|$d_h$|reshape 后最后一维错位|
|beam|样本到 beam 的映射|只重排 logits，不重排 cache|

### contiguous cache 的优点和代价

连续 cache 预先为每条序列分配一段最大长度。写入位置可以直接由当前长度得到，kernel 寻址简单；代价是：

- 需要预留最大上下文长度；
- 短请求占用未使用的尾部容量；
- 动态 batch 改变时可能需要搬迁；
- 多请求长度差异会造成矩形 padding；
- beam 扩展和请求取消可能留下空洞。

连续布局适合长度和 batch 较稳定的场景。它不等于每次都要物化完整 attention map；cache 仍只保存 K/V。

### paged cache 用 block table 把逻辑位置映射到物理块

分页 cache 把序列轴切成固定大小的 block。令 block size 为 $s_b$，逻辑位置为 $t$：

$$
p=\left\lfloor\frac{t}{s_b}\right\rfloor,
\qquad
o=t\bmod s_b.
$$

其中 $p$ 是逻辑 block 编号，$o$ 是 block 内偏移。调度器用 block table 把 $p$ 映射到物理 block：

$$
\operatorname{physical\_block}
=
\operatorname{block\_table}[p].
$$

真实地址还要加上 batch、K/V head 和 head dimension 的 stride。分页方式允许不同请求共享一个物理 block 池，减少按最大长度预分配带来的浪费，也便于请求结束后归还完整 block。

分页不改变 attention 的数学可见集合。它只改变 K/V 的寻址。block table、有效长度和 sequence ID 必须同时传给 kernel；只传一个物理起点不能表达非连续的 cache。

如果两个请求共享 prefix block，后续写入不能直接覆盖共享内容。实现需要 copy-on-write、只读标记或其他引用计数策略。否则一个请求的 decode 会修改另一个请求仍在读取的历史。

## batch 调度要同步 cache 和序列状态

### padding 会制造物理容量和计算浪费

把长度为 $T_1,\ldots,T_B$ 的请求补齐到 $T_{\max}$ 时，矩形 batch 的位置上界是 $BT_{\max}$，有效位置是：

$$
N_{\mathrm{valid}}
=
\sum_{i=1}^{B}T_i.
$$

padding 位置数为：

$$
N_{\mathrm{pad}}
=
BT_{\max}-N_{\mathrm{valid}}.
$$

若用物理容量定义 padding 比例：

$$
\rho_{\mathrm{pad}}
=
\frac{N_{\mathrm{pad}}}{BT_{\max}}.
$$

mask 可以阻止 PAD 影响结果，但不一定消除预分配、读取、写入和 kernel 调度。packed 或 varlen kernel 只有在实际跳过 padding 时，才会把部分计算转为有效长度总和。

### continuous batching 维护的是映射表

连续批处理允许新请求加入正在 decode 的 batch，也允许结束请求释放 slot。调度器至少要维护：

|状态|含义|更新时机|
|---|---|---|
|request ID|外部请求到内部序列的身份|入队、取消、完成|
|sequence ID|逻辑 token 序列的身份|分支、合并、重排|
|cache slot|K/V 的物理位置|分配、迁移、归还|
|logical length|已经写入的 token 数|每次追加后|
|position offset|下一 token 的绝对位置|每次追加前后|
|block table|逻辑 block 到物理 block|分页扩容、释放|
|finished flag|EOS、stop、超时或取消|每次生成后|

batch 重新排序时，logits、token、length、position 和 cache slot 必须使用同一个 permutation。只修改 batch 中的 token 排列，或者只更新 sequence length，都可能让请求读取另一个请求的历史。

### beam reorder 需要重排每一层 K/V

beam search 把一条输入扩展为多条候选。若当前有效 batch 是 $B$、beam 数是 $K$，cache 的逻辑第一轴可以看成 $B\times K$。选出新 beam 后，必须对每一层的 K 和 V 使用同一个父 beam 索引重排：

$$
\begin{aligned}
K_{\ell}^{\mathrm{new}}&=\operatorname{gather}
\left(K_{\ell}^{\mathrm{old}},\operatorname{parent}\right),\\
V_{\ell}^{\mathrm{new}}&=\operatorname{gather}
\left(V_{\ell}^{\mathrm{old}},\operatorname{parent}\right).
\end{aligned}
$$

只重排累计分数和当前 token，不重排 cache，会让下一步的 token 条件和历史分支不一致。beam 的 cache 容量通常还要乘以 $K$；如果实现共享完全相同的 prefix，则可以用只读 block 和 copy-on-write 避免立刻复制全部历史。

## prefix cache 复用的是完整的前缀状态

### 命中条件不能只比较字符串

多个请求有完全相同的前缀时，可以复用前缀 prefill 得到的 K/V。可复核的 cache key 至少要包含：

$$
\operatorname{key}
=
\operatorname{Hash}
\left(
\begin{gathered}
\text{checkpoint},\ \text{tokenizer},\ \text{template},\\
\text{input\_ids},\ \text{position rule},\ \text{mask rule},\\
\text{dtype},\ \text{adapter},\ \text{tenant policy}
\end{gathered}
\right).
$$

相同的原始字符串不保证相同的 input IDs。special token、聊天模板、BOS、Unicode 规范化、adapter、位置规则和 dtype 都可能改变 K/V。命中 prefix cache 后，应记录命中的 token 数、剩余 prefill token 数和 cache 的来源版本。

### partial hit 仍要记录边界

如果只命中前缀的一部分，命中区间的 K/V 可以复用，未命中 suffix 仍需要 prefill。逻辑上：

|阶段|需要处理的 token|cache 操作|
|---|---|---|
|完整 miss|整个 prompt|新建并写入全部 K/V|
|完整 hit|无新增 prompt|复用全部前缀 K/V|
|partial hit|未命中的 suffix|复用 prefix，追加 suffix K/V|
|模板改变|取决于 token 化结果|重新计算从变化点开始的状态|

前缀共享会改变内存所有权。共享 block 只读时，引用计数、租户隔离和过期策略必须进入审计。不能把 prefix cache 当作跨 checkpoint 或跨 adapter 的通用结果池。

## encoder-decoder 有静态 source cache

decoder-only 的 target self-attention cache 随输出长度增长。encoder-decoder 还可能在 encoder 完成后预先计算 source 的 K/V：

$$
\begin{aligned}
K_{\ell}^{\mathrm{src}}&\in\mathbb R^{B\times h_{kv}\times S\times d_h},\\
V_{\ell}^{\mathrm{src}}&\in\mathbb R^{B\times h_{kv}\times S\times d_h}.
\end{aligned}
$$

decoder 的每个 target query 读取固定的 source cache，source 长度 $S$ 不随目标生成长度 $U$ 增长。target self-attention 仍使用自己的动态 cache：

|cache 类型|key/value 来源|长度变化|主要用途|
|---|---|---|---|
|target self cache|已处理的目标 token|随 $U$ 增长|避免重复投影目标历史|
|static source cache|encoder source hidden state|通常固定为 $S$|避免每个 target step 重复 source 投影|
|prefix cache|共享 prompt 或 source 前缀|按命中区间复用|减少新请求 prefill|

如果把 source 长度当成 target cache 的增长长度，或者把 source cache 和 target self cache 使用同一个 position offset，cross-attention 会出现形状可运行但语义错位。[Encoder-Decoder Transformer](../transformer-architectures/encoder-decoder/)固定了 source、target、self-attention 和 cross-attention 的四条轴。

## 低精度和量化改变字节数与误差边界

### cache dtype 不等于权重 dtype

权重可以使用一种 dtype，K/V cache 可以使用另一种 dtype。部署报告至少要写：

|字段|需要记录的值|遗漏后的问题|
|---|---|---|
|权重 dtype|FP32、BF16、FP16 或量化格式|无法核对固定模型账|
|计算 dtype|矩阵乘和 softmax 的累加类型|无法解释 logits 差异|
|cache dtype|K/V 的存储格式|cache 字节数错误|
|scale dtype|量化 scale 的格式和粒度|遗漏元数据和带宽|
|反量化位置|读取前、tile 内或算子融合|无法复现误差和性能|

FP16 cache 的 $b=2$ 只代表两个字节的存储，不代表 kernel 一定用 FP16 完成点积累加。BF16、FP8、INT8 和其他格式还需要写出 scale、对齐和反量化规则。低精度 cache 可能降低显存和带宽，但会改变长上下文下的 attention score 与最终 logits；质量边界需要在目标模型和解码协议上测量。

### RoPE、量化和分页可以同时存在

分页只改变地址，量化改变元素表示，RoPE 改变位置条件。三者组合时要分别检查：

1. block table 找到的是正确的逻辑位置；
2. 量化 scale 与 K/V block 或 token 的对应关系未被重排破坏；
3. RoPE 只应用一次，并使用正确的绝对位置；
4. 解码 query 的计算 dtype 与缓存读取路径一致。

不能因为最终 cache shape 没有改变，就认为这三项实现可以任意交换。

## 窗口和淘汰会改变可见历史

### full cache 与 sliding window 不是同一个语义

完整上下文模式保存全部历史，当前长度为 $L$ 时 query 可以读取从 0 到 $L-1$ 的可见位置。滑动窗口只保留最近 $W$ 个位置：

$$
L_{\mathrm{effective}}
=
\min(L,W).
$$

当 $L>W$ 时，旧 K/V 会被淘汰或覆盖。这样可以把 cache 容量限制在：

$$
S_{\mathrm{window}}
=
2L_{\mathrm{layer}}BWh_{kv}d_hb.
$$

窗口 attention 不再等价于完整上下文 attention。需要同时固定：

|规则|需要核对|
|---|---|
|淘汰条件|按 token、block 还是时间|
|保留范围|最近 $W$ 个、局部窗口加全局 token，还是其他集合|
|position ID|保留绝对位置还是重新编号|
|RoPE offset|使用原始位置还是窗口局部位置|
|环形 buffer|覆盖前是否确认旧行不再被读取|
|上下文上限|拒绝、截断、窗口化还是分块摘要|

如果只把物理 cache 长度限制为 $W$，却仍使用完整上下文的 mask，kernel 可能读取失效地址；如果同时重编号位置，模型的相对位置关系也会改变。

### 请求结束必须释放动态状态

cache 不是模型权重。请求结束、取消、超时和错误路径都要释放或归还：

- contiguous buffer 的有效区间；
- paged cache 的 block table 和物理 blocks；
- prefix cache 的引用计数；
- beam 或多样本的分支状态；
- 量化 scale 和临时 workspace；
- request ID 到 cache slot 的映射。

只在正常 EOS 路径释放，会让取消请求和超时请求逐渐占满 cache 池。服务指标应区分逻辑 cache bytes、物理已分配 bytes、可复用 free blocks 和被引用的 prefix blocks。

## 运行方法

下面的标准库探针同时计算 cache 字节数、decode score 项和“完整重算”与“cache 路径”的因果位置交互。它不运行神经网络；数值只验证轴、长度和字节公式。

```python
def causal_pairs(length):
    return length * (length + 1) // 2


layers = 2
batch = 2
h_q = 8
h_kv = 4
head_dim = 8
bytes_per_value = 2
prompt = 4
new_tokens = 3
contexts = [4, 5, 6]


def cache_bytes(context_length):
    return (
        2
        * layers
        * batch
        * context_length
        * h_kv
        * head_dim
        * bytes_per_value
    )


decode_score_terms = [batch * h_q * length for length in contexts]
prefill_pairs = batch * h_q * causal_pairs(prompt)
cached_decode_pairs = batch * h_q * sum(contexts)
cached_total_pairs = prefill_pairs + cached_decode_pairs
naive_recompute_pairs = batch * h_q * sum(
    causal_pairs(length) for length in contexts
)

print("contexts=", contexts)
print("cache_bytes=", [cache_bytes(length) for length in contexts])
print("final_cache_bytes=", cache_bytes(prompt + new_tokens))
print("decode_score_terms=", decode_score_terms)
print("total_score_terms=", sum(decode_score_terms))
print(
    "cached_pairs=",
    {
        "prefill": prefill_pairs,
        "decode": cached_decode_pairs,
        "total": cached_total_pairs,
    },
)
print("naive_recompute_pairs=", naive_recompute_pairs)
```

运行输出为：

```text
contexts= [4, 5, 6]
cache_bytes= [2048, 2560, 3072]
final_cache_bytes= 3584
decode_score_terms= [64, 80, 96]
total_score_terms= 240
cached_pairs= {'prefill': 160, 'decode': 240, 'total': 400}
naive_recompute_pairs= 736
```

cache 字节数中的两个 K/V、两层、两个 batch 样本、四个 K/V head、八维 head 和 FP16 两字节都来自代码中的固定变量。最终长度为 $4+3=7$ 时，公式给出 $3584$ bytes。decode score 项按 $Bh_qL$ 计算；它随 $h_q$ 和历史长度增长，而不是随 $h_{kv}$ 直接减少。

## 失效模式和审计方法

### 把 cache 轴顺序当成数学合同

同一个逻辑 cache 可以有不同的物理 stride。若 kernel 以 $[B,T,h_{kv},d_h]$ 读取，调用方却按 $[B,h_{kv},T,d_h]$ 写入，形状仍可能兼容，数值会把 token 位置与 head 位置混在一起。审计要用一个小张量给每个轴写入可识别编号，再检查读取后的 K/V。

### 只缓存 K，不缓存 V

attention 需要 score 和 value 读取。只保存 K 可以计算权重，却无法得到 context。只保存 V 则无法计算 query-key score。每层、每个历史位置都要保存匹配的 K/V 对，且两者使用相同的 batch、head、position 映射。

### 追加到错误的位置

最常见的边界错误包括：

- prefill 后长度仍记录为 0；
- 追加前先把长度加一；
- chunk prefill 写入位置重叠；
- EOS 写入 cache 的规则与返回 token 规则不一致；
- 请求复用 slot 后没有清理旧长度。

要在每次追加后记录逻辑长度、首尾位置和物理 slot。不能只看最终生成文本。

### position offset 被重置

prefill 使用位置 $0,\ldots,P-1$ 后，第一次 decode 的新位置通常是 $P$。每次 batch 重排、prefix hit、sliding window 或 beam reorder 都要继续携带绝对位置规则。若 offset 重置为零，模型仍可能输出 token，但位置条件已经改变。

### mask 没有读取全部历史

decode query 的局部长度为 1，key 长度可能是 $L+1$。如果误用从零开始的 $1\times1$ mask，模型只能看到当前 token；如果把历史 padding 当作可见位置，结果会混入无效 K/V。应检查实际 mask 的 query absolute position、key absolute position、padding 和 block table。

### GQA 用错 head 数

GQA 的 cache 轴是 $h_{kv}$，score 的 query 轴是 $h_q$。把所有 attention 数都替换成 $h_{kv}$ 会低估 score、softmax 和 context 的工作。应分别打印 Q shape、K/V shape、group mapping 和 score shape。

### beam 只重排分数

beam 的父索引必须作用于每层 K 和 V。只重排 token、分数或 hidden state 的一部分，会让下一步使用错误历史。审计可以在两个 beam 生成不同标记 token 后，检查重排后的 K/V 是否与父 beam 一一对应。

### prefix cache 命中条件过宽

只按字符串前缀复用 cache 会忽略 tokenizer、模板、special token、position、mask、dtype、adapter 和租户。命中键要包含影响 K/V 的完整协议，命中日志要包含 checkpoint 和输入 token hash。

### 量化 scale 跟着错误的 block 移动

分页、beam reorder 或 batch compaction 后，K/V block 和 scale 必须使用同一个映射。scale 若仍留在旧 slot，误差可能只在特定长度或特定 head 出现。用 FP16 reference 与量化路径比较逐层 K/V 读取和最终 logits。

### 把窗口 cache 当作完整上下文

淘汰旧 K/V 后，模型的可见集合已经改变。不能仅修改物理长度而保留完整 mask、旧 position 或旧 RoPE offset。需要在报告中明确窗口大小、保留规则和与 full-context 结果的差异。

### 一份最小 KV cache 审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|模型结构|checkpoint 的 $h_q$、$h_{kv}$、$d_h$、layer 数固定|配置文件、权重 shape|
|Q/K/V shape|Q 使用 $h_q$，K/V 使用 $h_{kv}$|reshape、transpose、group mapping|
|cache 长度|每次追加 $R$ 后由 $L$ 变为 $L+R$|prefill 边界、EOS、chunk offset|
|mask|第 $i$ 个新 query 可读到绝对位置 $L+i$|causal offset、padding、block table|
|position|K/V 写入位置与 Q 的 position ID 一致|RoPE、窗口、重排|
|字节公式|$2L_{\mathrm{layer}}BTh_{kv}d_hb$ 与分配量对齐|K/V 两份、dtype、scale|
|decode 对比|full prefill 与 incremental logits 在容差内一致|历史 K/V、mask、kernel|
|batch 映射|request、sequence、slot、length 使用同一 permutation|continuous batching、compaction|
|beam|每层 K/V 使用父 beam 索引同步重排|gather、beam score|
|prefix cache|tokenizer、模板、checkpoint、adapter 和位置规则一致|命中键、租户隔离|
|分页|逻辑 block 能映射到有效物理 block|block table、free list、引用计数|
|淘汰|窗口、释放和超时路径都更新所有权|ring buffer、cancel、GC|

一个完整的运行报告至少应同时给出：模型配置、请求 prompt token 数、生成 token 数、K/V head 数、cache dtype、逻辑有效 bytes、物理分配 bytes、prefill 时间、逐 token decode 时间、是否命中 prefix cache、是否发生 beam 或 batch 重排，以及 full prefill 对照的误差。缺少其中任一项时，单独的“KV cache 大小”不能说明推理实现已经正确。

## 相关词条

- [推理](../inference/inference/)：固定一次冻结模型执行的输入、解码、停止和调度协议。
- [GQA 与 MQA](../attention/gqa-and-mqa/)：解释 query head 与 K/V head 的共享映射，以及共享对 cache 的影响。
- [因果掩码](../attention/causal-masking/)：推导 prefill、decode、padding 和 cache offset 的可见集合。
- [注意力复杂度](../attention/attention-complexity/)：分别核算 attention 交互、物化矩阵和 KV cache 的资源项。
- [Decoder-Only Transformer](../transformer-architectures/decoder-only/)：说明 target self-attention、prefill、decode 和位置条件。
- [Encoder-Decoder Transformer](../transformer-architectures/encoder-decoder/)：区分动态 target self cache 与静态 source cache。
- [参数量与资源账本](../transformer-components/parameter-count/)：分开模型权重、激活、运行时状态和 KV cache。
- [量化](../inference/quantization/)：讨论权重与运行时张量的低比特表示、scale 和误差边界。
- [长上下文](../inference/long-context/)：讨论上下文长度、显存、位置编码和窗口策略的扩展约束。
