---
title: "FlashAttention：怎样不保存 attention matrix 仍保持 dense 语义"
tags: ["why-models-learn"]
---

FlashAttention 是一种 IO-aware 的 dense attention kernel：它把 query、key、value 和 score 分成 tile，在片上存储中完成缩放、mask、在线 softmax 和 value 加权，避免把完整的 $T\times S$ score 或 attention weight 矩阵写入高带宽显存。它保留标准 dense attention 的逻辑连接和 softmax 语义，主要减少中间矩阵的显存读写与峰值工作区；它不把 $T\times S$ 的 dense 算术量变成线性，也不等同于稀疏 attention 或 linear attention。

![FlashAttention 用 query/key/value tile 和在线 softmax 处理 dense attention，避免物化完整 T×S 矩阵](/assets/attention/svg/flash-attention.1.svg)

## 先区分语义、算术和 IO

### 标准 attention 有三个中间对象

对一个 head，令：

$$
Q\in\mathbb R^{T\times d_k},
\qquad
K\in\mathbb R^{S\times d_k},
\qquad
V\in\mathbb R^{S\times d_v}.
$$

标准缩放点积 attention 的逻辑步骤是：

$$
\begin{gathered}
S=QK^\mathsf T/\sqrt{d_k},\\
P=\operatorname{softmax}_{\mathrm{row}}(S+M),\\
O=PV.
\end{gathered}
$$

$S$ 是 score 或 logits 矩阵，$P$ 是逐行归一化后的权重矩阵，$O$ 是输出。$S$ 和 $P$ 都有 $T\times S$ 个元素；$O$ 只有 $T\times d_v$ 个元素。

标准定义关心的是：

$$
O_{t,:}
=
\sum_{s=1}^{S}P_{t,s}V_{s,:}.
$$

只要每一行的 $P_{t,:}$ 与 dense reference 一致到允许的浮点误差，输出语义就一致。实现不要求把 $P$ 作为一个完整的显存张量存在。

### 三种成本要分别报告

|成本|dense attention 的数量|FlashAttention 的变化|
|---|---:|---|
|逻辑 query-key 对|$B h_qTS$|不变|
|$QK^\mathsf T$ 与 $PV$ MAC|$B h_qTS(d_k+d_v)$|不变，受 kernel 融合和舍入影响|
|完整 score/weight HBM 中间矩阵|通常为 $B h_qTS$ 或两份|不物化完整矩阵|
|片上 score tile|—|约为 $B_rB_c$，随 tile 配置改变|
|输出和 row statistics|$B h_qTd_v$ 及行统计|写回输出和少量统计量|

HBM 是 GPU 上容量大但访问代价高的显存层；片上 SRAM、shared memory 或寄存器容量小但访问快。FlashAttention 的收益来自减少 HBM 往返，不是减少数学上的位置连接。

## tile 化如何避免完整矩阵

### 两条序列轴切成 query block 和 key block

把 query 轴切成宽度为 $B_r$ 的 block，把 key/value 轴切成宽度为 $B_c$ 的 block。对 query block $I$ 和 key block $J$，只在片上计算：

$$
S_{I,J}
=
Q_IK_J^\mathsf T/\sqrt{d_k}.
$$

这个 tile 的形状是 $B_r\times B_c$。随后在片上应用 mask、更新该 query block 的 softmax 统计量和输出累加器。处理完这个 key block 后，释放 tile，再加载下一个 key/value block。

完整的 $T\times S$ 矩阵从未作为一个整体写入 HBM。每个 query block 只需要最终输出和每行的归一化统计量。

### 在线 softmax 保存每一行的三个状态

对一个 query 行，维护：

- $m$：已经处理的 score 的最大值；
- $\ell$：以当前最大值为基准的指数和；
- $o$：同一基准下的 value 加权和。

初始状态为：

$$
m^{(0)}=-\infty,
\qquad
\ell^{(0)}=0,
\qquad
o^{(0)}=\mathbf 0.
$$

当前 tile 的 score 行为 $s_j$，先取：

$$
\widetilde m
=
\max_j s_j,
\qquad
m'
=
\max\left(m,\widetilde m\right).
$$

旧状态需要按新的最大值重标度：

$$
\begin{aligned}
\ell'
&=
\exp(m-m')\ell
+
\sum_j\exp(s_j-m'),\\
o'
&=
\exp(m-m')o
+
\sum_j\exp(s_j-m')v_j.
\end{aligned}
$$

处理完所有 key tile 后，输出为：

$$
o_{\mathrm{final}}
=
\frac{o'}{\ell'},
\qquad
\operatorname{LSE}
=
m'+\log\ell'.
$$

第一 tile 中 $m=-\infty$ 的旧项按零处理。这个更新等于把不同 tile 的指数和转换到同一个全局最大值下，因此不会因为直接计算 $\exp(s_j)$ 而轻易溢出。

### 一个三列 score 的在线 softmax 例子

取一个 query 行：

$$
s=(2,0,-1),
\qquad
v=(10,20,40).
$$

把前两个 score 放入第一个 tile，把最后一个 score 放入第二个 tile。第一个 tile 处理后：

$$
m_1=2,
\qquad
\ell_1=1+\exp(-2)=1.135335283,
$$

$$
o_1
=
10+20\exp(-2)
=
12.706705665.
$$

第二个 tile 的 score 为 $-1$，新的最大值仍为 2：

$$
\begin{aligned}
\ell_2
&=
1.135335283+\exp(-3)
=
1.185122352,\\
o_2
&=
12.706705665+40\exp(-3)
=
14.698188399.
\end{aligned}
$$

最后相除：

$$
\frac{o_2}{\ell_2}
=
12.402253978.
$$

直接对完整 score 做 row-wise softmax 得到的权重为：

$$
(0.843794734,\ 0.114195199,\ 0.042010066),
$$

对应 context 也是 $12.402253978$，差异只来自显示精度和浮点舍入。分块顺序改变时，浮点归约顺序可能带来最后几位差异；这不等于算法改变了 softmax 定义。

### mask 必须在最大值和指数之前生效

如果位置 $(t,s)$ 不可见，应在 tile 内把对应 score 视为 $-\infty$，再参与最大值、指数和归一化。否则被屏蔽位置可能影响 $m$ 或 $\ell$，最后仍泄漏到输出。

对 causal mask：

- 若 key tile 完全位于 query block 的未来，可以整块跳过；
- 若 key tile 完全位于允许区域，可以整块计算；
- 若是 query/key 对角 tile，只能对 tile 内部逐元素应用下三角 mask。

padding、prefix mask、source visibility 和 cache offset 都必须在同一 tile 坐标系中解释。mask 的轴仍然是 query 行与 key 列，不会因为分块而交换。

## 前向资源：少写矩阵，不少算术

### 一个长序列的 dense 账本

取：

$$
B=2,\qquad
h_q=16,\qquad
T=S=2048,\qquad
d_k=d_v=64.
$$

完整 dense attention 的 score 元素数为：

$$
Bh_qTS
=
2\cdot16\cdot2048^2
=
134{,}217{,}728.
$$

若每个元素为 FP16：

$$
134{,}217{,}728\times2
=
268{,}435{,}456
$$

bytes，即 256 MiB。若 logits 和归一化后的 $P$ 同时物化，两份矩阵合计 512 MiB，还没有计入 Q/K/V、输出、残差和其他 layer 激活。

两次矩阵乘法的 MAC 为：

$$
Bh_qTS(d_k+d_v)
=
17{,}179{,}869{,}184.
$$

FlashAttention 仍需计算这些 score 与 value 加权项。它把 score tile 在片上生成、消费和丢弃，避免把两份 $T\times S$ 中间矩阵写入 HBM。

### tile 配置决定片上工作区

取：

$$
B_r=64,\qquad B_c=128.
$$

query 轴有 $2048/64=32$ 个 block，key 轴有 $2048/128=16$ 个 block。每个 query block 需要遍历：

$$
32\times16=512
$$

个 score tile。单个 score tile 有：

$$
B_rB_c
=
64\times128
=
8192
$$

个元素，FP16 只占 16 KiB。实际片上工作区还要容纳 Q tile、K tile、V tile、输出累加器、row statistics 和实现的同步缓冲；16 KiB 只是 score tile 的下界，不是整个 kernel 的 shared memory 使用量。

在这一循环顺序下，每个 query block 可能重新加载一次 K/V tile。单个 head 的 K/V HBM 读量近似为：

$$
\frac{T}{B_r}S(d_k+d_v).
$$

因此 tile 不是把所有 HBM traffic 变成 $T+S$；它的核心变化是消除完整 $T\times S$ 中间结果的写回和再次读取。精确 HBM traffic 取决于 tile 尺寸、数据重用、线程布局和 kernel 版本。

### causal tile 数量与 tile 内浪费

若 $T=S=2048$ 且 $B_r=B_c=128$，两条轴各有 16 个 block。causal 可见的 block 对有：

$$
\frac{16\cdot17}{2}
=
136
$$

个，而完整矩形有 $16^2=256$ 个。完全位于未来的 block 可以跳过；对角 block 仍可能执行完整 $128\times128$ tile，再在 tile 内屏蔽未来元素。

因此至少要区分三个数字：

|对象|数量|
|---|---:|
|完整 dense token 对|$2048^2=4,194,304$|
|causal 合法 token 对|$2048\cdot2049/2=2,098,176$|
|causal 可见 block 对|$136$|

block 数减少不意味着每个对角 block 只计算三角形；kernel 是否能跳过对角 tile 内的无效元素，需要看具体实现。

## backward：保存 row statistics，再重算 tile

### 为什么不保存完整 $P$

训练时反向需要 attention 权重参与：

$$
\begin{aligned}
dV&=P^\mathsf TdO,\\
dP&=dOV^\mathsf T.
\end{aligned}
$$

若把完整 $P$ 保存下来，反向可以直接读取，但显存峰值包含 $T\times S$ 矩阵。FlashAttention 前向可以只保存每个 query 行的 log-sum-exp：

$$
\operatorname{LSE}_t
=
m_t+\log\ell_t,
$$

以及 Q/K/V、输出和其他必要边界。反向遍历同样的 tile 时，重新计算 score、mask 和归一化权重：

$$
P_{I,J}
=
\exp\left(S_{I,J}-\operatorname{LSE}_I\right).
$$

这样用重算换取显存。保存的 LSE 不能替代输出 $O$；它只提供恢复每行 softmax 归一化的统计量。

### softmax 的反向也可以按 tile 计算

对每个 query 行，令 $dP$ 为对权重的梯度。row-wise softmax 的梯度为：

$$
dS
=
P\odot
\left(
dP-
\sum_s(P_s dP_s)
\right).
$$

实现可以在当前 tile 内累积行的点积修正项，再计算：

$$
dQ=dS K/\sqrt{d_k},
\qquad
dK=dS^\mathsf TQ/\sqrt{d_k},
\qquad
dV=P^\mathsf TdO.
$$

完整梯度公式没有因为 tile 化而改变。改变的是 $P$ 和 $S$ 的生成、保存和重算时机。由于反向可能以不同顺序累计多个 tile 的梯度，最后几位浮点数与非分块 reference 不完全相同是正常的；应使用明确的 tolerance。

### dropout 需要可重现的 tile mask

attention dropout 通常在 softmax 后对 $P$ 乘一个随机 mask 和缩放因子。若前向不保存完整 dropout mask，反向重算对应 tile 时必须根据相同的随机种子、offset 和 counter 重新生成它。

以下差异会导致反向不匹配：

- tile 顺序改变了随机数 counter；
- batch/head/row/column 到 counter 的映射不同；
- causal 或 padding 位置也消耗了随机数；
- checkpoint 重算使用了不同的 RNG 状态；
- dropout 缩放是在 softmax 前还是后执行。

没有 dropout 时，LSE 和输入足以支持典型的 tile 重算；有 dropout 时还要审计 RNG 协议。

## 变长、padding 和 cache 坐标

### packed sequence 需要真实长度 metadata

batch 内序列长度不同时，dense 张量通常使用 padding。FlashAttention 的 varlen 版本可以使用每条序列的 cumulative sequence length，把多个样本压入一个 packed buffer。kernel 需要知道：

$$
\operatorname{cu\_seqlens}
=
(0,L_1,L_1+L_2,\ldots).
$$

这组 metadata 决定每个 query block 对应哪条序列、key block 的起止位置和 mask 边界。只把 padding token 的 score 设为负无穷，不一定减少读取和 tile 数；packed kernel 才可能按有效长度减少工作。

### cache offset 会改变 causal 比较

decode 或 chunk prefill 中，query 的逻辑位置可能从历史长度 $P$ 开始，key 的 cache 位置也带有 past offset。causal 判断不能只比较 tile 内的局部下标，应比较真实位置：

$$
\operatorname{allow}(t,s)
\Longleftrightarrow
\operatorname{key\_position}(s)
\le
\operatorname{query\_position}(t).
$$

如果只用 tile 内的 $i\le j$ 或 $i\ge j$，chunk 的第一行可能错误读取未来 token，或者错误屏蔽仍然合法的历史 cache。

### cross-attention 的两个 tile 轴不必等长

cross-attention 的 query 轴长度是 $T$，source key/value 轴长度是 $S$。FlashAttention 的 tile loop 直接使用 $B_r$ 和 $B_c$ 分别覆盖两条轴：

$$
S_{I,J}
=
Q_IK_J^\mathsf T/\sqrt{d_k},
\qquad
O_I\leftarrow O_I+\operatorname{softmax\_tile}(S_{I,J})V_J.
$$

source padding 应作用在 key 列，target padding 或 query mask 应作用在 query 行。不能因为 kernel 接口把两条轴都称为 sequence，就把 cross-attention 当作 $T=S$ 的 causal self-attention。

### GQA 的 K/V head 布局需要在 tile 前确定

对 GQA，query head 数为 $h_q$，K/V head 数为 $h_{kv}$。每个 query head 读取映射后的 K/V tile；score tile 数仍按 $h_q$ 计算。kernel 可以：

- 在多个 query head 之间复用同一个 K/V tile；
- 按 group 先加载 K/V，再让组内 query head 使用；
- 把 K/V broadcast 到临时布局；
- 使用专门的 grouped layout。

这些方案的逻辑输出可以相同，但 HBM 读量、shared memory 复用和寄存器压力不同。GQA 减少 K/V cache，不自动减少 query head 的 score tile。

## 数值稳定性和边界情况

### 所有 mask 行必须有定义

如果一整行 query 都被 padding 或 source visibility 屏蔽，tile 统计量可能是：

$$
m=-\infty,
\qquad
\ell=0.
$$

直接执行 $0/0$ 会产生 NaN。实现需要为全 mask 行定义输出，常见做法是输出零并在后续 loss 或 residual 路径屏蔽该行；具体行为必须与 reference 对齐。

### finite sentinel 不等于负无穷

为了避免硬件对 $-\infty$ 的特殊处理，有些 kernel 使用有限负 sentinel。若 score 范围足够大，sentinel 可能仍参与 softmax，产生微小但非零的权重。比较 reference 时要固定 sentinel、输入 dtype、累加 dtype 和 tolerance。

### 累加 dtype 与输入 dtype 可以不同

Q/K/V 可以使用 FP16 或 BF16，online softmax 的 $m,\ell,o$ 通常需要更高精度 accumulator。tile 间反复重标度会放大舍入差异，长序列和尖锐 score 尤其明显。

至少记录：

- score 的输入 dtype；
- $m,\ell,o$ 的 accumulator dtype；
- LSE 的保存 dtype；
- 是否使用 exp 的近似实现；
- forward/backward 的 tolerance；
- 是否启用 deterministic 模式。

“无 NaN”只说明边界没有失控，不说明输出已经足够接近 reference。

## FlashAttention 与其他方案的边界

|方法|dense $T\times S$ 算术|完整 map|softmax 语义|主要节省|
|---|---|---|---|---|
|naive dense|保留|通常物化|标准|实现简单|
|FlashAttention|保留|不物化|标准，受舍入影响|HBM IO、峰值工作区|
|sparse attention|只算选中连接|只存选中区域|可保留|位置连接与算术|
|linear attention|通常不按全部位置配对|不物化标准 map|通常为 kernel 近似|长度项与状态大小|
|GQA/MQA|score 仍按 $h_q$|K/V head 减少|标准或 tile 化|K/V 投影和 cache|

FlashAttention 可以与 causal mask、GQA、varlen 和 dropout 组合。若同时使用 sparse pattern，必须确认 kernel 真正跳过无效 tile；仅仅把 sparse mask 传给 dense FlashAttention 接口，不足以证明位置算术下降。

## 失效模式与审计方法

### 把不物化 map 写成减少了二次 FLOPs

FlashAttention 仍计算 dense 的 $QK^\mathsf T$ 和 $PV$。应分别报告 MAC、完整 map 是否写回和实际 HBM traffic。

### 把片上 tile 大小当成完整 map 大小

$B_rB_c$ 是单个工作区 tile。kernel 会遍历多个 tile；片上峰值和全局累计 traffic 不是同一个数字。

### 忽略 online softmax 的旧状态重标度

新 tile 的最大值变大时，旧 $\ell$ 和 $o$ 必须乘 $\exp(m-m')$。如果只把新 tile 加到旧状态，结果不会等于完整 softmax。

### 在 mask 后再计算最大值不一致

mask 必须在 max、exp、sum 和 value accumulator 的同一坐标约定下生效。部分 causal tile 特别容易把未来 score 留在最大值计算中。

### backward 保存了错误的统计量

LSE 是 $m+\log\ell$，不是简单的最大 score，也不是未经重标度的指数和。反向重算必须使用与 forward 相同的 scale、mask 和 dtype 约定。

### dropout 重算使用了不同随机序列

forward 和 backward 的 tile RNG counter 不一致时，梯度会错误，即使输出 shape 和 loss 正常。固定种子后比较 dropout mask 或梯度 reference。

### packed sequence 使用了 padding 坐标

变长 batch 的 cumulative lengths、tile 起点和 causal offset 必须使用真实序列位置。只比较 packed buffer 的局部 index 会让跨样本读取变得可能。

### GQA broadcast 造成额外读写

逻辑上共享 K/V 不等于实现上没有复制。如果先把 K/V broadcast 到 $h_q$，cache 可能没有节省，shared memory 读写还可能增加。检查实际 layout 和 profiler。

### 全 mask 行静默产生 NaN

padding、prefix mask 和 cross source visibility 都可能制造空行。对输出、LSE、梯度和 loss 做 NaN 检查，并记录空行策略。

### 用单次误差判断 kernel 正确

浮点归约顺序、tile 顺序、dropout 和 deterministic 设置会改变误差。至少比较多种长度、mask、dtype、causal/cross、全 mask 行和 backward 梯度。

### 一份最小 FlashAttention 审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|逻辑 shape|$Q,K,V$ 和输出轴正确|self/cross、head reshape|
|tile shape|$B_r,B_c$ 与硬件布局一致|shared memory、寄存器|
|scale|$1/\sqrt{d_k}$ 在 score tile 中使用|head dimension|
|online state|$m,\ell,o$ 按新最大值重标度|tile 合并顺序|
|mask|在 max、exp、sum、value 路径一致生效|causal、padding、offset|
|输出归一化|最终 $o/\ell$|分母、全 mask 行|
|中间矩阵|完整 score/P 不写入 HBM|kernel trace、显存峰值|
|算术量|dense 版本仍按 $Bh_qTS(d_k+d_v)$|误写成 linear|
|backward|保存 LSE 或等价统计量并重算 tile|梯度、recompute|
|dropout|RNG counter 可重现|checkpoint、tile 顺序|
|varlen|使用真实 cu_seqlens|padding、跨样本边界|
|cache offset|比较真实 query/key position|chunk prefill、decode|
|GQA|K/V group 映射与复用正确|broadcast、cache layout|
|dtype|输入、accumulator、LSE 分开记录|FP16/BF16、溢出|
|质量|与 dense reference 的输出/梯度误差在 tolerance 内|mask、归约顺序|
|性能|同时报告 latency、HBM traffic、峰值显存|不要只报 FLOPs|

## 相关词条

- [注意力复杂度](../attention/attention-complexity/)：区分位置交互、MAC、物化 map 和 cache。
- [Attention 矩阵](../attention/attention-matrix/)：定义逻辑 attention matrix 与 row-wise softmax。
- [缩放点积注意力](../attention/scaled-dot-product/)：说明 score scale、mask 和 softmax。
- [因果掩码](../attention/causal-masking/)：处理 causal、padding、cache offset 和全 mask 行。
- [GQA 与 MQA](../attention/gqa-and-mqa/)：说明 K/V head 共享和 cache layout。
- [交叉注意力](../attention/cross-attention/)：说明 $T\times S$ 的 source/target 轴。
- [稀疏注意力](../attention/sparse-attention/)：区分 tile IO 优化与实际稀疏连接。
- [线性注意力](../attention/linear-attention/)：区分 online softmax 与 feature kernel 重排。
