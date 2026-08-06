---
title: "注意力复杂度：为什么序列交互会产生二次成本"
tags: ["why-models-learn"]
---

注意力复杂度描述一次 attention 计算需要处理多少 query-key 交互、乘加、激活内存和缓存字节。标准 dense attention 的核心交互数是 query 行数乘以 key 列数，即 $B h_q T S$；self-attention 取 $T=S=L$ 后才得到常说的二次项 $L^2$。本篇把逻辑交互数、实际 kernel 工作、物化 attention matrix、KV cache 和自回归解码分开核算，再说明 GQA、稀疏 attention、FlashAttention 与线性 attention 分别改变哪一项。

![self、cross 和 decode attention 的 query-key 交互数量，以及 GQA 只减少 K/V cache 不减少 query head 交互](/assets/attention/svg/attention-complexity.1.svg)

## 先把复杂度的对象分开

### $T$ 和 $S$ 是两条轴

设 batch size 为 $B$，query head 数为 $h_q$，K/V head 数为 $h_{kv}$，query 序列长度为 $T$，key/value 序列长度为 $S$。每个 head 的匹配维度和 value 维度分别为 $d_k$、$d_v$：

$$
\begin{gathered}
Q\in\mathbb R^{B\times h_q\times T\times d_k},\\
K\in\mathbb R^{B\times h_{kv}\times S\times d_k},\\
V\in\mathbb R^{B\times h_{kv}\times S\times d_v}.
\end{gathered}
$$

在 MHA 中通常有 $h_{kv}=h_q$。在 GQA 和 MQA 中，$h_{kv}<h_q$，K/V 会按组被多个 query head 读取。为了先数清 attention 的位置交互，假定 K/V 已按照实现约定广播或 gather 到每个 query head。

每个 query head 与每个 key 位置形成一个候选交互对：

$$
N_{\mathrm{pair}}
=
B h_q T S.
$$

这个数量回答“逻辑上有多少个 query-key 位置组合”。它还没有回答每个组合要做多少维度的点积，也没有回答实现是否把 score 或权重写到显存。

### 一个交互对包含两次矩阵乘法中的一个位置

标准缩放点积 attention 的两步矩阵乘法是：

$$
S^{(q)}
=
\frac{Q^{(q)}(K^{(q)})^\mathsf T}{\sqrt{d_k}},
\qquad
C^{(q)}
=
A^{(q)}V^{(q)},
$$

其中 $S^{(q)}$ 与 $A^{(q)}$ 的形状为 $T\times S$。对一个 query-key 对：

- $QK^\mathsf T$ 需要长度为 $d_k$ 的点积；
- $AV$ 需要把这个位置的权重乘到长度为 $d_v$ 的 value 上；
- softmax 还要沿每个 query 行扫描 $S$ 个 score，计算指数、求和和归一化。

若把一次乘法加一次加法计作一个 MAC，则两次矩阵乘法的 MAC 数为：

$$
N_{\mathrm{attn\text{-}MAC}}
=
B h_q T S(d_k+d_v).
$$

硬件文档常把一个 FMA 计作 2 FLOPs。因此只把两次矩阵乘法换算为 FLOPs 时：

$$
N_{\mathrm{attn\text{-}FLOP}}
\approx
2B h_q T S(d_k+d_v).
$$

这里的近似没有展开 softmax、mask、缩放和边界处理的标量指令。报告 FLOPs 时必须说明采用 MAC 还是 2-FLOP FMA 口径。

### 四个数字回答四个不同问题

|数字|对应对象|典型公式|不能替代的判断|
|---|---|---|---|
|交互对数|query 位置与 key 位置的逻辑组合|$B h_q T S$|不等于矩阵乘法 FLOPs|
|MAC/FLOPs|点积与 value 加权的算术量|$B h_q T S(d_k+d_v)$|不等于实际延迟|
|激活元素|score、权重、Q/K/V、context 的张量大小|按张量形状分别计数|不等于峰值显存|
|缓存字节|自回归步骤之间保存的 K/V|$B L h_{kv}(d_k+d_v)$|不等于训练期 attention map|

同一层可以有相同的交互对数、不同的物化策略和不同的 wall-clock latency。FlashAttention 主要改变中间矩阵的读写方式；GQA 主要改变 K/V head 轴；稀疏 attention 才会在 kernel 确实跳过无效位置时减少交互对。

## 从形状推导计算账本

### 投影层的成本随序列长度线性增长

令 query 输入宽度为 $d_{\mathrm{in,q}}$，K/V 输入宽度为 $d_{\mathrm{in,kv}}$，输出宽度为 $d_{\mathrm{out}}$。Q、K、V 和输出投影的 MAC 数为：

$$
\begin{aligned}
N_{\mathrm{proj\text{-}MAC}}
={}&
B\bigl(
T d_{\mathrm{in,q}} h_q d_k
+
S d_{\mathrm{in,kv}} h_{kv} d_k\\
&\quad+
S d_{\mathrm{in,kv}} h_{kv} d_v
+
T h_q d_v d_{\mathrm{out}}
\bigr).
\end{aligned}
$$

在 self-attention 的标准 MHA 配置中，令 $T=S=L$、$h_q=h_{kv}=h$、$d_k=d_v=d_{\mathrm{out}}/h=d_{\mathrm{model}}/h$，并令三个输入宽度都为 $d_{\mathrm{model}}$。四个投影合计：

$$
N_{\mathrm{proj\text{-}MAC}}
=
4B Ld_{\mathrm{model}}^2.
$$

这个项对序列长度是线性的。它仍然可能很大，因为 $d_{\mathrm{model}}^2$ 的常数比单个 head 的点积宽。

### attention 交互成本随两个长度的乘积增长

两次矩阵乘法合计为：

$$
N_{\mathrm{attn\text{-}MAC}}
=
B h_q T S(d_k+d_v).
$$

在标准 self-attention 配置下，若 $d_k=d_v=d_{\mathrm{model}}/h$，则：

$$
N_{\mathrm{attn\text{-}MAC}}
=
2B L^2d_{\mathrm{model}}.
$$

因此 self-attention 的长度项为 $L^2$。当 $T$、$S$ 不相等时，不能把它写成一个模糊的“序列长度平方”；cross-attention 的长度项是 $TS$。

### softmax 的数量级与矩阵乘法相同

对每个 query 行，softmax 至少要读取 $S$ 个 score。所有 head 和 batch 合计要扫描：

$$
B h_q T S
$$

个 score 元素。指数、求和、除法和 mask 的算术常数取决于实现，但长度数量级仍为 $\mathcal O(Bh_qTS)$。因此只核对 $QK^\mathsf T$ 而漏掉 softmax 的行扫描，会低估 dense attention 的工作。

### 输出投影和非 attention 子层另行计数

$AV$ 的结果拼接后进入 $W_O$。在标准 MHA 中，输出投影也是 $B Ld_{\mathrm{model}}^2$ MAC 的线性项。一个 Transformer block 还包含 layer normalization、残差和 MLP；它们有自己的参数、激活和算术账本。

因此，“attention 的二次成本”只描述 attention 子层的序列交互项。不能把整个 Transformer block 的所有 FLOPs 都归到 $L^2$，也不能因为短序列时投影或 MLP 占主导，就否认长序列时交互项的增长。

### 参数量与序列长度没有同一比例

在固定 head 配置下，Q/K/V 和 $W_O$ 的参数量与 $T$、$S$ 无关。输入序列变长时，新增的是投影激活、score、权重、context 和训练保存的中间量。一个简化账本如下：

|对象|self-attention 中的长度依赖|改变 head 数的影响|主要用途|
|---|---|---|---|
|Q 投影参数|无|随 $h_qd_k$ 改变|生成 query|
|K/V 投影参数|无|随 $h_{kv}(d_k+d_v)$ 改变|生成可共享的读取地址和内容|
|score 与权重元素|$T^2$|随 $h_q$ 改变|记录位置交互|
|Q/K/V 激活|$T$|Q 随 $h_q$，K/V 随 $h_{kv}$|前向和反向|
|KV cache|历史长度 $L$|随 $h_{kv}$ 改变|跨解码步骤复用 K/V|

把参数量、一次前向激活量和跨步骤 cache 放在同一列中相加，会得到无法解释的资源数字。

## 用数字看见 self-attention 的增长

### 固定宽度后扫描 $L$

取一个标准 MHA 例子：

$$
B=2,\qquad
h_q=h_{kv}=16,\qquad
d_{\mathrm{model}}=1024,\qquad
d_k=d_v=64,
\qquad
T=S=L.
$$

只计算 Q/K/V/输出投影与 $QK^\mathsf T$、$AV$ 两次矩阵乘法。一次 FP16 元素占 2 bytes，$1\ \mathrm{MiB}=2^{20}$ bytes。结果如下：

|$L$|交互对数 $Bh_qL^2$|attention MAC|四个投影 MAC|单个 $A$ 的 FP16 字节|
|---:|---:|---:|---:|---:|
|$512$|8,388,608|1,073,741,824|4,294,967,296|16 MiB|
|$1024$|33,554,432|4,294,967,296|8,589,934,592|64 MiB|
|$2048$|134,217,728|17,179,869,184|17,179,869,184|256 MiB|

把 $L$ 从 512 加倍到 1024，交互对数、attention MAC 和单个 attention matrix 的字节数都变为 4 倍；投影 MAC 只变为 2 倍。再把 $L$ 从 1024 加倍到 2048，比例保持不变。

表中的 attention MAC 已包含 $QK^\mathsf T$ 和 $AV$ 的长度维度，但没有把 softmax 标量操作折算进去。若实现同时物化 logits 和归一化后的 $A$，这两份 $T\times T$ 矩阵各需要 256 MiB；如果还保存反向所需的其他张量，峰值会更高。

### 长度交叉点不是普适常数

在上面的标准 MHA 配置中：

$$
N_{\mathrm{attn\text{-}MAC}}
=
2BL^2d_{\mathrm{model}},
\qquad
N_{\mathrm{proj\text{-}MAC}}
=
4BLd_{\mathrm{model}}^2.
$$

令两者相等，得到：

$$
2BL^2d_{\mathrm{model}}
=
4BLd_{\mathrm{model}}^2
\quad\Longrightarrow\quad
L=2d_{\mathrm{model}}.
$$

当 $d_{\mathrm{model}}=1024$ 时，这个简化交叉点是 $L=2048$。它只比较两类矩阵乘法的 MAC 数，不包含 MLP、softmax、内存访问和 kernel 效率，因此不能把 2048 当成所有硬件上的延迟拐点。它说明为什么短序列时投影项仍可能占主导，长度继续增加后交互项会快速追上。

### causal mask 只减少逻辑可见对

对于 inclusive causal mask，第 $t$ 个 query 可以读取 $t+1$ 个 key。单个 batch、单个 head 的可见位置数为：

$$
N_{\mathrm{causal,one\ head}}
=
\sum_{t=0}^{T-1}(t+1)
=
\frac{T(T+1)}{2}.
$$

加入 batch 和 query head 后：

$$
N_{\mathrm{causal}}
=
B h_q\frac{T(T+1)}{2}.
$$

在前面的 $B=2,h_q=16$ 配置中：

|$T$|dense 矩形元素 $Bh_qT^2$|causal 可见元素 $Bh_qT(T+1)/2$|可见比例|
|---:|---:|---:|---:|
|$1024$|33,554,432|16,793,600|约 $50.01\%$|
|$2048$|134,217,728|67,141,632|约 $50.00\%$|

这是逻辑连接数量。若 kernel 仍先生成完整 $T\times T$ score，再把上三角设为 mask，实际矩阵乘法和显存申请仍可能接近 dense 数量。只有实现真正跳过被屏蔽的块，算术和读写才会随可见区域下降。不能只看到下三角 mask，就把运行时间自动除以二。

### cross-attention 的乘积可以相同，资源位置仍不同

取：

$$
B=2,\qquad h_q=h_{kv}=16,\qquad d_k=d_v=64.
$$

下面三种 $T,S$ 组合的乘积都相同：

|$T$|$S$|交互对数 $Bh_qTS$|单个 FP16 attention matrix|query 激活元素 $Bh_qTd_k$|
|---:|---:|---:|---:|---:|
|$128$|$4096$|16,777,216|32 MiB|262,144|
|$512$|$1024$|16,777,216|32 MiB|1,048,576|
|$2048$|$256$|16,777,216|32 MiB|4,194,304|

三行的 $QK^\mathsf T$ 与 $AV$ 交互量相同，但 query 输出长度、source K/V 激活和投影成本不同。$T$ 是目标输出轴，$S$ 是可读取的 source 轴；交换两者不会保持整个系统的语义和缓存行为不变。

在 $T=128,S=4096$ 的第一行，attention 两次矩阵乘法需要：

$$
N_{\mathrm{attn\text{-}MAC}}
=
16{,}777{,}216\times(64+64)
=
2{,}147{,}483{,}648
$$

个 MAC。单个 score 或权重矩阵占 32 MiB FP16。这个数字只描述 attention 交互，不包含 encoder 对 4096 个 source token 的计算，也不包含 decoder 的 self-attention。

## 显存账本：物化矩阵与 KV cache

### 一个矩阵的元素数乘以字节数

设数据类型每个元素占 $b$ bytes。若实现物化完整 score 矩阵或完整 attention weight 矩阵：

$$
M_{\mathrm{map}}
=
bBh_qTS.
$$

若 logits 和 $A$ 同时存在：

$$
M_{\mathrm{logits}+A}
=
2bBh_qTS.
$$

对于 self-attention，$T=S=L$，所以物化矩阵的显存项为 $\mathcal O(Bh_qL^2)$。对于 cross-attention，它是 $\mathcal O(Bh_qTS)$。这两个表达式的长度变量不能混写。

激活峰值还包括：

$$
\begin{aligned}
N_Q&=Bh_qTd_k,\\
N_K&=Bh_{kv}Sd_k,\\
N_V&=Bh_{kv}Sd_v,\\
N_C&=Bh_qTd_v.
\end{aligned}
$$

把这些元素分别乘以 dtype 字节数，才能得到对应的存储量。训练时还可能保存 softmax 统计量、dropout mask、残差输入、layer normalization 中间量和反向所需的重计算边界。

### 逻辑存在不等于完整矩阵已经写入显存

attention matrix 在数学上定义了 $Bh_qTS$ 个位置的权重。实现可以有两种不同做法：

|实现选择|逻辑 $A$ 是否存在|完整 $A$ 是否写入高带宽显存|主要影响|
|---|---|---|---|
|朴素 dense kernel|是|通常是|显存峰值和读写量高|
|分块 dense kernel|是|可能只写部分块|峰值与 IO 下降|
|FlashAttention 类 kernel|是|不写完整 $A$|保持 dense 算术，减少中间矩阵 IO|
|稀疏 kernel|只对选中位置计算|只写选中块|交互数量与模式密度共同决定|

FlashAttention 用 tile 处理 $Q$、$K$、$V$，并用每行运行中的最大值和归一化因子完成稳定 softmax。对同一组 $Q,K,V$ 和 mask，它保留标准 attention 的矩阵语义，数值结果会受浮点舍入影响。它减少的是高带宽显存读写和峰值工作区，不会把 dense softmax 自动变成线性长度。

### 训练显存和推理 cache 不是一份账

训练要同时处理一批完整序列，反向还要使用前向中间量。activation checkpointing 可以少保存一部分中间量，但会在反向重算；它改变显存与计算的折中，不改变标准 attention 的逻辑交互数。

自回归推理的 KV cache 则跨时间步保存历史 K/V。设当前缓存长度为 $L$：

$$
N_{\mathrm{KV\text{-}cache}}
=
B L h_{kv}(d_k+d_v),
\qquad
M_{\mathrm{KV\text{-}cache}}
=
bB L h_{kv}(d_k+d_v).
$$

self-attention 的 $L$ 随已处理的 target token 增长。decoder cross-attention 的 source K/V 通常在 encoder 完成后一次写入，长度是固定的 $S$；它是静态 source cache，不是随每个输出 token 重新增长的 target cache。

### GQA/MQA 减少 cache 的 head 轴

取一个 decoder 层：

$$
B=1,\qquad
h_q=32,\qquad
L=4096,\qquad
d_k=d_v=128.
$$

比较 MHA、每组 4 个 query head 的 GQA 和 MQA。表中 score 交互仍按 $h_q$ 计算：

|配置|$h_{kv}$|KV cache 元素|FP16 KV cache|score 交互对数|attention MAC|
|---|---:|---:|---:|---:|---:|
|MHA|$32$|33,554,432|64 MiB|536,870,912|137,438,953,472|
|GQA-4|$8$|8,388,608|16 MiB|536,870,912|137,438,953,472|
|MQA|$1$|1,048,576|2 MiB|536,870,912|137,438,953,472|

GQA-4 和 MQA 把 K/V head 数从 32 减少到 8 或 1，因此 cache 和 K/V 投影参数下降。每个 query head 仍然对长度 4096 的 key 轴计算 score 和 value 加权；交互对数和 attention MAC 没有因 $h_{kv}$ 下降而改变。

若某个实现同时把 $h_q$ 的 score 矩阵改成 $h_{kv}$，它实现的是另一种连接规则，不能仅凭“GQA”这个名称接受。应检查运行时 score shape 和 query head 到 K/V group 的映射。

## 解码阶段：prefill 和 decode 的长度项不同

### prefill 一次处理完整前缀

prefill 将 prompt 的多个 token 一次送入模型。self-attention 的 query 与 key/value 都来自 prompt，长度为 $P$：

$$
N_{\mathrm{prefill,pair}}
=
B h_qP^2.
$$

causal mask 让位置 $t$ 只读到前缀中的 $0$ 到 $t$，但 dense kernel 是否跳过上三角仍是实现问题。

cross-attention 的 prefill 视具体架构而定：若 decoder 同时处理目标前缀，query 长度是 $T$，source 长度是 $S$，它的交互对数仍为 $B h_qTS$。这和 encoder 对 source 做的 self-attention $S^2$ 是两个不同项。

### decode 每次只增加一个 query

生成第 $u$ 个新 token 时，target self-attention 通常只有一个新的 query 行。若 prompt 长度为 $P$，已经生成了 $u$ 个 token，当前 self-attention 的 key/value 历史长度为 $P+u$：

$$
N_{\mathrm{decode,self,pair}}(u)
=
B h_q(P+u).
$$

连续生成 $U$ 个 token 后，self-attention 的累计 query-key 对数为：

$$
\begin{aligned}
N_{\mathrm{decode,self,pair,total}}
&=
B h_q\sum_{u=0}^{U-1}(P+u)\\
&=
B h_q\left(UP+\frac{U(U-1)}{2}\right).
\end{aligned}
$$

每一步仍有与当前历史长度成正比的读取工作，但不会重新计算所有历史 token 的 Q/K/V。KV cache 的作用是把历史 K/V 投影从每一步的重复工作中拿出来。

### cross-attention decode 的 source 长度保持固定

若 source K/V cache 已经建立，生成 $U$ 个 target token 的 cross-attention 交互对数为：

$$
N_{\mathrm{decode,cross,pair,total}}
=
B h_qUS.
$$

它对生成长度 $U$ 和 source 长度 $S$ 都是线性的。decoder 的 target self-attention 仍会产生随历史长度增长的项；不能把 cross-attention 的线性项当成整个 decoder 的总复杂度。

取：

$$
B=1,\quad h_q=32,\quad d_k=d_v=128,\quad
P=2048,\quad U=128,\quad S=4096.
$$

只比较 attention 的交互矩阵乘法：

|部分|累计交互对数|attention MAC|
|---|---:|---:|
|target self-attention decode|8,648,704|2,214,068,224|
|source cross-attention decode|16,777,216|4,294,967,296|

self-attention 的累计值使用 $UP+U(U-1)/2$；cross-attention 的每个新 query 都读取 4096 个 source key。两行都没有计入投影、softmax、MLP 和输出层。

在同一例子中，若 self-attention 使用 $h_{kv}=32$，完成 128 个新 token 后 target KV cache 的长度为 $P+U=2176$，占用 34 MiB FP16；若 cross-attention 的 source 长度为 4096，静态 source KV cache 占用 64 MiB FP16。两项属于不同 cache，实际 decoder 层还要加上其他激活和模型权重。

### batch 和 beam 会把缓存项再乘一次

上面的 cache 公式含有 $B$。beam search 通常把 beam 维并入有效 batch；若有 $B_{\mathrm{beam}}$ 条 beam，缓存容量近似乘以 beam 数。beam reorder 只改变样本到 cache 行的映射，不改变单条 beam 的 $L$、$S$ 和 head 轴复杂度。

变长 batch 还需要记录每条序列的真实长度。将 padding 长度直接代入 $T$、$S$ 会给出分配上界；packed 或 varlen kernel 若真的跳过 padding，实际工作可能按有效长度总和计算。报告时要区分“形状上界”和“有效 token 数”。

## 加速方法分别改动哪一项

### 局部窗口减少位置连接

若每个 query 只读取最多 $w$ 个相邻 key，稀疏位置对的数量近似为：

$$
N_{\mathrm{local,pair}}
\approx
B h_q T\min(S,w).
$$

self-attention 中 $T=S=L$ 时，局部窗口把 $L^2$ 变为约 $Lw$；当 $w$ 固定时，对长度是线性的。边界、双向窗口、causal 窗口和跨块连接会改变常数与精确计数。

只在 mask 中标记局部窗口而仍调用计算完整矩形的 dense GEMM，不会自动得到 $Lw$ 的运行时间。要减少实际工作，需要 kernel 或布局确实跳过窗口外的位置。

### block-sparse 由密度决定工作量

设保留的块占所有 query-key 块的比例为 $\rho$。理想的 block-sparse kernel 需要处理：

$$
N_{\mathrm{block\text{-}sparse,pair}}
\approx
\rho B h_qTS.
$$

$\rho$ 只是平均密度。块大小、块分布、负载均衡、索引读取和硬件矩阵乘法对齐都会影响实际延迟。一个有 10% 密度但块分散的实现，未必比一个有 25% 密度但布局规整的实现快。

### 线性 attention 改变了 softmax 的代数形式

标准 softmax attention 的分子含有：

$$
\sum_{s=1}^{S}
\exp\left(\frac{q_t^\mathsf T k_s}{\sqrt{d_k}}\right)v_s.
$$

由于指数和 query 同时参与每个 key 的匹配，标准 dense softmax 一般不能把 $QK^\mathsf T$ 与 $V$ 任意交换次序。线性 attention 常用特征映射近似：

$$
\exp(q^\mathsf T k)
\approx
\phi(q)^\mathsf T\phi(k),
$$

其中 $\phi$ 的输出维度记为 $r$。在这个近似下，可以先累积 source 的统计量：

$$
\begin{aligned}
G
&=
\sum_{s=1}^{S}\phi(k_s)v_s^\mathsf T,\\
z
&=
\sum_{s=1}^{S}\phi(k_s),\\
\widetilde c_t
&=
\frac{\phi(q_t)^\mathsf TG}
{\phi(q_t)^\mathsf Tz}.
\end{aligned}
$$

固定 $r$ 后，构造 $G,z$ 的成本随 $S$ 线性增长，查询成本随 $T$ 线性增长；一个粗略的单 head 账本为：

$$
\mathcal O\left(Sr(d_v+1)+Tr(d_v+1)\right).
$$

它不再物化 $T\times S$ 的标准 attention map，但输出也不再自动等于标准 row-wise softmax attention。是否保留相同的归一化、因果性、数值稳定性和模型质量，要由具体 feature map 与实现验证。

### FlashAttention 与线性 attention 不是同一类变化

|方法|逻辑位置对|完整 attention map|标准 softmax 语义|主要变化|
|---|---:|---|---|---|
|dense attention|$Bh_qTS$|通常物化|是|基线|
|causal dense|$Bh_qT(T+1)/2$ 个可见对|可能仍申请矩形|是|mask 约束可见位置|
|local / block sparse|按窗口或块密度减少|只保留选中区域|可保留|减少连接与 kernel 工作|
|FlashAttention|仍为 $Bh_qTS$|不物化完整矩阵|是，受舍入影响|减少 HBM IO 与峰值工作区|
|linear attention|不按全部 $TS$ 对计算|不物化标准矩阵|通常是近似形式|改变 softmax 代数以换取线性长度项|

FlashAttention 解决“如何高效执行 dense attention”；稀疏 attention 解决“哪些位置需要连接”；linear attention 解决“能否重排或近似矩阵计算”。三者可以分别组合，不能用同一个“复杂度降了”描述。

## 失效模式与审计方法

### 只写 $O(L^2)$，没有声明 head 和维度

$O(L^2)$ 只保留 self-attention 的长度项。若比较两个模型，还要记录 $B$、$h_q$、$d_k$、$d_v$、dtype 和是否物化矩阵。否则相同的 $L$ 可能对应不同的 MAC 和显存。

### 把 cross-attention 也写成 $L^2$

cross-attention 的矩阵形状是 $T\times S$。翻译、摘要和视觉文本模型通常有不同的 source/target 长度。先列出两条轴，再决定是否可以在特定实验中令 $T=S$。

### 用 $h_{kv}$ 计算 score 数量

GQA/MQA 减少 K/V head，但保留 $h_q$ 个独立 query。dense score 的 head 轴仍是 $h_q$。只有明确改变 query 连接的稀疏方案才会减少 score 头数。

### 把 mask 的逻辑零当成已经节省了 FLOPs

mask 只规定哪些位置不能进入 softmax。若 kernel 仍计算完整矩形，矩阵乘法数量不变。需要查看 kernel 的 tile 访问、实际 FLOPs 或 profiler 证据。

### 把 FlashAttention 写成线性 attention

FlashAttention 不改变 dense $TS$ 交互的数量级；它不物化完整 map，并降低 IO。linear attention 依赖特征映射或其他近似，标准 softmax 语义需要单独核对。

### 把一次 decode 的成本当成整个生成成本

decode 每一步只有一个新 query，但历史长度在增长。要报告单 token 成本和 $U$ 个 token 的累计和。prefill 的 $P^2$ 项也要单独列出。

### 把 KV cache 和训练 attention map 相加

KV cache 跨步骤保存历史 K/V；训练 attention map 是当前批次的 query-key 交互中间量。两者的长度轴、head 轴和生命周期不同。应分别报告元素和字节。

### 忽略 MAC/FLOP 口径

一个乘加在不同报告中可能记为 1 MAC 或 2 FLOPs。先写定义，再给数字。softmax、归一化、mask 和内存访问通常不在简单矩阵乘法 FLOPs 中。

### 用 FLOPs 直接预测延迟

不同 kernel 的内存访问、tile 对齐、并行度、数据类型和硬件利用率不同。FLOPs 适合比较算术量；延迟和吞吐需要在目标 batch、长度和硬件上实测。

### 变长 batch 只看 padding 上界

$T$、$S$ 可以代表张量上界，也可以代表有效长度。packed 或 varlen kernel 可能按有效 token 数运行。记录每条样本长度、padding 规则和 kernel 是否跳过 padding。

### 一份最小注意力复杂度审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|query 轴|明确 $T$ 和 query hidden 来源|self/cross 子层输入|
|key/value 轴|明确 $S$ 和 K/V hidden 来源|source 长度、cache 长度|
|query head|score 的 head 轴为 $h_q$|Q reshape、GQA 配置|
|K/V head|缓存和投影使用 $h_{kv}$|K/V reshape、group mapping|
|交互对|$Bh_qTS$ 或明确的稀疏子集|是否把 $h_{kv}$ 代入|
|点积 MAC|$Bh_qTSd_k$|$d_k$ 与 $d_{\mathrm{model}}$ 混淆|
|value MAC|$Bh_qTSd_v$|输出宽度、广播|
|softmax 轴|沿 key 轴归一化|mask 广播、row-wise axis|
|causal 计数|逻辑可见数为 $Bh_qT(T+1)/2$|dense kernel 是否跳过上三角|
|物化 map|按 $bBh_qTS$ 计算每份矩阵|logits 与 $A$ 是否同时存在|
|KV cache|按 $bBLh_{kv}(d_k+d_v)$ 计算|self 的 $L$、cross 的 $S$|
|prefill|self 通常有 $P^2$ 交互|是否错误按 decode 单行计算|
|decode|逐步 self 按历史长度增长|累计求和、source 固定|
|加速类型|区分 tile、稀疏和代数近似|把 Flash 写成 linear|
|报告口径|注明 MAC/FLOP、dtype、batch、长度|数字不可复现|

## 相关词条

- [Attention 矩阵](../attention/attention-matrix/)：定义 query 行、key 列和 $T\times S$ 的逻辑读取矩阵。
- [缩放点积注意力](../attention/scaled-dot-product/)：推导 $QK^\mathsf T/\sqrt{d_k}$、softmax 和 $AV$。
- [自注意力](../attention/self-attention/)：说明同一序列上的 $T\times T$ 交互和位置语义。
- [多头注意力](../attention/multi-head-attention/)：解释 query head、每头维度和输出投影。
- [GQA 与 MQA](../attention/gqa-and-mqa/)：说明 $h_q$ 与 $h_{kv}$ 的共享关系及 KV cache。
- [因果掩码](../attention/causal-masking/)：推导下三角可见区域和 cache offset。
- [交叉注意力](../attention/cross-attention/)：固定目标 $T$ 与源 $S$ 的矩形 attention。
- [FlashAttention](../attention/flash-attention/)：展开 tile、IO 和不物化完整 attention matrix。
- [稀疏注意力](../attention/sparse-attention/)：讨论局部、块稀疏和选择性位置连接。
- [线性注意力](../attention/linear-attention/)：讨论特征映射、顺序重排和标准 softmax 的近似边界。
