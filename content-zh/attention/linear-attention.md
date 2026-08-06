---
title: "线性注意力：怎样重排全局读取而不构造 T×S 矩阵"
tags: ["why-models-learn"]
---

Linear Attention（线性注意力）用特征映射把 query-key 相似度写成可分离的内积，再利用矩阵结合律先汇总 key/value，避免标准 dense attention 的 $T\times S$ attention matrix。对固定特征维度 $r$，序列长度项可以从 $\mathcal O(TS)$ 降到 $\mathcal O((T+S)r)$；代价是有限特征映射通常只近似 softmax kernel，归一化、位置偏置、mask、causal 顺序和数值稳定性都需要重新核对。本篇从标准 softmax attention 出发，推导可重排形式，再分别处理全序列、causal prefix、cross-attention、GQA 和 cache。

![标准 attention 先生成 T×S 矩阵，线性 attention 先汇总 phi(K) 转置 V 与 phi(K) 再读取](/assets/attention/svg/linear-attention.1.svg)

## 从标准 attention 到可重排形式

### 标准 softmax 把每个 query 和每个 key 配对

对一个 head，忽略 batch 和 head 下标，标准 attention 为：

$$
C
=
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{QK^\mathsf T}{\sqrt{d_k}}
\right)V.
$$

第 $t$ 个输出位置可以写成：

$$
c_t
=
\frac{
\displaystyle\sum_{s=1}^{S}
\exp\left(\frac{q_t^\mathsf T k_s}{\sqrt{d_k}}\right)v_s
}{
\displaystyle\sum_{s=1}^{S}
\exp\left(\frac{q_t^\mathsf T k_s}{\sqrt{d_k}}\right)
}.
$$

分子和分母都要对当前 query $q_t$ 与每个 key $k_s$ 计算一次相似度。所有 $t$、$s$ 组合合计 $TS$ 个位置对；这就是 dense attention 的序列二次项。

### 特征映射把相似度拆成 query 因子和 key 因子

设有一个特征映射：

$$
\phi:\mathbb R^{d_k}\longrightarrow\mathbb R^r.
$$

如果相似度 kernel 可以写成：

$$
\kappa(q,k)
=
\phi(q)^\mathsf T\phi(k),
$$

那么把标准 softmax kernel 换成 $\kappa$ 后，归一化 attention 为：

$$
\widetilde c_t
=
\frac{
\displaystyle\sum_{s=1}^{S}
\phi(q_t)^\mathsf T\phi(k_s)v_s
}{
\displaystyle\sum_{s=1}^{S}
\phi(q_t)^\mathsf T\phi(k_s)
}.
$$

利用标量与向量乘法的结合律，把与 $t$ 无关的 source 汇总移到外面：

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

$G$ 的形状是 $r\times d_v$，$z$ 的形状是 $r$。先计算 $G,z$，再让每个 query 读取这两个摘要，就不需要生成 $T\times S$ 的权重矩阵。

### 矩阵结合律是重排的关键

把所有 query 排成矩阵，记 $\Phi(Q)\in\mathbb R^{T\times r}$、$\Phi(K)\in\mathbb R^{S\times r}$。不带归一化时：

$$
\Phi(Q)\left(\Phi(K)^\mathsf TV\right)
=
\left(\Phi(Q)\Phi(K)^\mathsf T\right)V.
$$

右侧显式产生 $T\times S$ 矩阵；左侧先计算 $r\times d_v$ 的摘要，再计算 query 输出。只要 $r$ 远小于 $T$、$S$，左侧的长度项更小。

归一化版本再额外计算：

$$
\Phi(Q)\left(\Phi(K)^\mathsf T\mathbf 1_S\right),
$$

其中 $\mathbf 1_S$ 是长度为 $S$ 的全 1 向量。分子和分母共享同一个 key 汇总，逐 query 相除。

## 线性不等于标准 softmax 的精确等价

### 指数点积的精确特征维度通常不是有限数

标准 softmax 使用：

$$
\exp(q^\mathsf Tk).
$$

它有幂级数展开：

$$
\exp(q^\mathsf Tk)
=
\sum_{n=0}^{\infty}
\frac{(q^\mathsf Tk)^n}{n!}.
$$

这个展开可以对应一个无限维特征表示。有限维 $r$ 的 $\phi$ 通常只能近似它，或者直接选择另一个正 kernel。于是：

$$
\phi(q)^\mathsf T\phi(k)
\approx
\exp(q^\mathsf Tk)
$$

是近似假设，不是对所有 $q,k$ 的有限维恒等式。

有限特征维度带来一个明确的选择：$r$ 越大，kernel 近似可能更精确，汇总状态和计算也越大；$r$ 越小，线性成本更低，但 query-key 匹配的表达能力受限。

### 一个一维数值例子

取一个标量 query、三个标量 key 和三个标量 value：

$$
q=1,\qquad
k=(0,1,2),\qquad
v=(10,20,40).
$$

先用标准 softmax kernel，score 为 $(0,1,2)$。权重和 context 为：

$$
A_{\mathrm{softmax}}
\approx
(0.090030573,\ 0.244728471,\ 0.665240956),
$$

$$
c_{\mathrm{softmax}}
\approx
32.404513384.
$$

现在选一个简单的正 kernel：

$$
\phi(x)=
\begin{bmatrix}
1\\x
\end{bmatrix},
\qquad
\kappa(q,k)=1+qk.
$$

三个 kernel 权重的未归一化值为 $(1,2,3)$，归一化后 context 为：

$$
\widetilde c
=
\frac{1\cdot10+2\cdot20+3\cdot40}{1+2+3}
=
28.333333333.
$$

两种输出不同。这个例子不表示线性 attention 一定质量较低；它只证明有限特征 kernel 与标准 softmax 的权重和输出需要独立比较。若 $\phi$ 产生负 kernel，权重还可能失去非负概率的解释；需要使用正值映射或重新定义归一化。

### 正值分母是归一化的必要条件

标准 softmax 的分母严格为正。线性形式中的分母是：

$$
d_t
=
\phi(q_t)^\mathsf Tz.
$$

若 $\phi(q_t)$、$\phi(k_s)$ 的内积可以为负，$d_t$ 可能接近零、变成负数或在浮点计算中不稳定。实现通常选择非负 feature map，或加入数值稳定的截断和 epsilon：

$$
\widetilde c_t
=
\frac{\phi(q_t)^\mathsf TG}
{\max(d_t,\varepsilon)}
$$

这会改变精确公式，必须在数值测试中记录。不能把“先汇总再相除”直接当作无条件稳定的 softmax。

## 非因果线性 attention 的计算和内存

### source 汇总与 query 读取

设 batch size 为 $B$、head 数为 $h$、query 长度为 $T$、source 长度为 $S$、feature 维度为 $r$、value 维度为 $d_v$。每个 head 的 source 汇总需要：

$$
\begin{aligned}
N_G&=B h Srd_v,\\
N_z&=B h Sr.
\end{aligned}
$$

query 读取汇总需要：

$$
\begin{aligned}
N_{\mathrm{num}}&=B h Trd_v,\\
N_{\mathrm{den}}&=B h Tr.
\end{aligned}
$$

把这些乘加型项合在一起，得到近似长度账本：

$$
N_{\mathrm{linear}}
\approx
B h(T+S)r(d_v+1).
$$

这里的 $+1$ 表示归一化向量 $z$ 的额外维度项。特征映射本身、指数或激活、除法和投影层的成本没有全部展开。

标准 attention 的两次矩阵乘法则为：

$$
N_{\mathrm{dense}}
=
B hTS(d_k+d_v).
$$

当 $r,d_k,d_v$ 固定且 $T,S$ 增大时，linear attention 的长度项是 $T+S$，dense attention 的长度项是 $TS$。

### 状态大小不再随 $T\times S$ 增长

线性 attention 的全局摘要为：

$$
G\in\mathbb R^{r\times d_v},
\qquad
z\in\mathbb R^r.
$$

每个 batch、每个 head 的摘要元素数为：

$$
N_{\mathrm{state,one\ head}}
=
r(d_v+1).
$$

所有 batch 和 head 合计：

$$
N_{\mathrm{state}}
=
B h r(d_v+1).
$$

如果把 $G,z$ 以 FP16 保存，状态字节数为 $2N_{\mathrm{state}}$。训练仍可能保存 $\Phi(Q)$、$\Phi(K)$、输入和反向中间量；因此不能把小状态直接等同于整个训练峰值显存。

### 一个长序列资源例子

取：

$$
B=2,\qquad
h=16,\qquad
T=S=4096,\qquad
d_k=d_v=64,\qquad
r=64.
$$

标准 dense attention 的账本为：

- query-key 位置对：536,870,912；
- $QK^\mathsf T$ 与 $AV$ 的 MAC：68,719,476,736；
- 单个 FP16 attention map：1024 MiB。

线性 kernel 的近似项为：

|项|乘加型项|元素或状态|
|---|---:|---:|
|source $G$ 汇总|536,870,912|$G$ 元素为 $Bhrd_v=131,072$|
|source $z$ 汇总|8,388,608|$z$ 元素为 $Bhr=2,048$|
|query numerator|536,870,912|输出分子|
|query denominator|8,388,608|归一化分母|
|合计|1,090,519,040|状态 FP16 约 0.25390625 MiB|

这个比较只核对 attention 的核心项。线性版本仍要做 Q/K/V 投影、feature map、输出投影和其他 Transformer 子层；它不是把整个 block 的所有成本都删除。即使忽略投影，核心项也从 68.7B 降到约 1.09B，原因是先把 4096 个 source 的信息压进 $r\times d_v$ 状态。

改变 $r$ 会线性改变汇总和读取成本：

|feature 维度 $r$|核心乘加型项|摘要 FP16|
|---:|---:|---:|
|$32$|545,259,520|0.126953125 MiB|
|$64$|1,090,519,040|0.25390625 MiB|
|$128$|2,181,038,080|0.5078125 MiB|

$T$ 和 $S$ 固定时，增大 $r$ 会提高近似容量，也会线性增加状态和算力。比较两个实现时，必须同时固定 $r$、feature map、dtype 和投影配置。

## causal linear attention：把全局摘要变成 prefix state

### prefix state 的递推

causal attention 只允许位置 $t$ 读取 $s\le t$。线性形式可以在每个位置维护前缀状态：

$$
\begin{aligned}
G_t
&=
G_{t-1}+\phi(k_t)v_t^\mathsf T,\\
z_t
&=
z_{t-1}+\phi(k_t),\\
c_t
&=
\frac{\phi(q_t)^\mathsf TG_t}
{\phi(q_t)^\mathsf Tz_t}.
\end{aligned}
$$

初始状态为 $G_{-1}=0$、$z_{-1}=0$。更新先加入位置 $t$，所以这个约定包含当前 key；如果模型需要严格的 shifted input，必须把更新时刻和 query 时刻错开。

单个 token 的状态大小固定为 $r(d_v+1)$。若使用 $B h$ 个独立 head，state 数为 $Bhr(d_v+1)$。旧 token 不再以独立 K/V 行保留在 attention state 中，而是累积在 $G_t,z_t$ 内。

### 每个 decode step 的读取成本不随历史长度增长

在同一个 $B=2,h=16,r=64,d_v=64$ 配置下，单个新 query 的线性状态读取约为：

$$
B h r(d_v+1)
=
2\cdot16\cdot64\cdot65
=
133{,}120
$$

个乘加型项。

如果使用标准 dense self-attention，并且当前历史长度为 $L=4096$、$d_k=d_v=64$，单个 decode query 的交互 MAC 为：

$$
B hL(d_k+d_v)
=
2\cdot16\cdot4096\cdot128
=
16{,}777{,}216.
$$

这两者的长度依赖不同：linear causal attention 的 attention state 读取保持固定，dense attention 的历史读取随 $L$ 增长。投影和 MLP 仍然存在；“每步固定”只针对线性 attention 的状态部分。

### 训练并行和推理递推是不同问题

标准 dense causal attention 可以在训练时一次计算完整下三角矩阵，所有 query 行有较高并行度。linear causal attention 的递推在语义上有前缀依赖：

$$
(G_{t-1},z_{t-1})
\longrightarrow
(G_t,z_t).
$$

可以使用 associative scan 把部分前缀合并以提高训练并行度，但 scan 的临时状态、归约顺序和浮点误差要单独处理。推理阶段自然地逐 token 更新；训练阶段的总算力低，不代表 GPU kernel 一定获得同等吞吐。

反向传播还要经过每个 prefix state。如果完整保存所有 $G_t,z_t$，训练显存随 $T$ 线性增长；如果重算或使用 scan，计算与显存的折中会改变。不能只用推理 state 大小推断训练峰值。

### cache 语义从“保存历史行”变为“保存摘要”

标准 self-attention KV cache 保存：

$$
K_{0:L-1},V_{0:L-1}.
$$

causal linear attention 主要保存：

$$
G_{L-1},z_{L-1}.
$$

摘要 cache 的优点是容量不随 $L$ 增长。代价是不能从摘要中精确取回某个历史 token 的独立 K/V，也不能任意改变旧 token 的权重后再计算原始 dense softmax。以下操作因此需要额外设计：

- 删除某个历史 token；
- 修改一个旧 token 后只局部更新；
- beam reorder 时复制或重排摘要；
- 对一段 source 重新做局部 mask；
- 解释某个历史 token 对当前输出的单独权重。

线性 attention 的 cache 更小，不表示它保留了 dense attention cache 的所有可编辑结构。

## cross-attention 与多头共享

### 静态 source 可以先聚合

cross-attention 的 source 长度为 $S$，target query 长度为 $T$。若所有 target query 都可以读取同一份 source，source 摘要可以预先计算：

$$
\begin{aligned}
G_{\mathrm{src}}
&=
\sum_{s=1}^{S}\phi(k_s)v_s^\mathsf T,\\
z_{\mathrm{src}}
&=
\sum_{s=1}^{S}\phi(k_s),\\
c_t
&=
\frac{\phi(q_t)^\mathsf TG_{\mathrm{src}}}
{\phi(q_t)^\mathsf Tz_{\mathrm{src}}}.
\end{aligned}
$$

encoder 完成后，$G_{\mathrm{src}},z_{\mathrm{src}}$ 可以在多个 decoder query 和多个生成 step 中复用。source 长度的聚合只做一次；target 端每个 query 只读取固定大小的摘要。

### source visibility 会破坏单一摘要

如果 target query $t$ 只能读取 source 的某个子集 $E_t$，一个全 source 摘要不能同时回答所有 query。需要满足以下条件之一：

- 所有 $E_t$ 相同，可以使用一个摘要；
- $E_t$ 是 prefix，可以使用 prefix state；
- $E_t$ 是连续区间，可以使用可相减的 prefix summary；
- $E_t$ 有有限个固定块，可以为每个块建立摘要；
- 任意动态集合，需要重新选择或构造候选摘要。

因此 cross-attention 的 $TS$ 乘积可以在固定 source 全可见时被重排；带任意对齐窗口的 cross-attention 不会自动获得同样的线性形式。

### GQA 让摘要按 K/V group 共享

若有 $h_q$ 个 query head、$h_{kv}$ 个 K/V group，每个 group 使用一个 source summary：

$$
G_g
=
\sum_s\phi(k_{g,s})v_{g,s}^\mathsf T,
\qquad
z_g
=
\sum_s\phi(k_{g,s}).
$$

query head $q$ 通过 group mapping $g(q)$ 读取：

$$
c_q
=
\frac{\phi(q_q)^\mathsf TG_{g(q)}}
{\phi(q_q)^\mathsf Tz_{g(q)}}.
$$

状态元素从 MHA 的 $B h_qr(d_v+1)$ 变为：

$$
N_{\mathrm{state,GQA}}
=
B h_{kv}r(d_v+1).
$$

query 读取和 Q 投影仍按 $h_q$ 计数。linear attention 的特征维度、GQA 的 K/V head 数和序列连接 pattern 是三个独立轴。

## mask、位置和特征映射的限制

### arbitrary mask 通常不能只靠一次聚合

标准 dense attention 可以为每个 $(t,s)$ 位置加 mask。线性重排要求 source 汇总与 query $t$ 无关；如果 mask 是任意的 $M_{t,s}$，则：

$$
\sum_s
M_{t,s}\phi(k_s)v_s^\mathsf T
$$

会随 $t$ 改变，不能直接复用同一个 $G$。把任意 mask 仍写成一个全局 summary，会把不可见 source 泄漏到输出。

prefix mask 可以用递推；连续窗口可以使用两个 prefix state 的差，但需要保证 subtraction 后的数值和分母稳定；随机或 query-dependent mask 通常需要多个候选摘要或退回显式候选计算。

### 位置编码必须能被重排

如果位置编码加在输入 hidden 上，再生成 Q/K，某些绝对位置方案可以直接进入 $\phi(q_t),\phi(k_s)$。但任意 relative position bias 会产生：

$$
\kappa_{t,s}
=
\exp(q_t^\mathsf Tk_s+b_{t,s}),
$$

其中 $b_{t,s}$ 同时依赖 query 和 key 位置。若它不能分解成有限维的 query 因子与 key 因子，source 不能用一个与 $t$ 无关的 $G$ 汇总。

RoPE、相对位置 bias 或复杂位置核需要具体推导。不能看到 Q/K 仍有相同的最后一维，就认为 dense attention 的位置机制可以直接搬到线性重排中。

### feature map 的正性、尺度和归一化要一起验证

feature map 改变后需要检查：

- $\phi(q)^\mathsf T\phi(k)$ 是否非负；
- 分母是否远离 0；
- 输入尺度变化是否让 $G$ 溢出；
- 长序列累加是否损失精度；
- 是否需要 fp32 accumulator；
- epsilon 放在分母还是状态更新中；
- 不同 head 是否使用同一 feature map；
- causal prefix 的累加顺序是否固定。

混合精度下，$G_t$ 可能累加许多 outer product。即使每个 Q/K/V 是 FP16，也可能需要 FP32 state；dtype 改变会同时改变显存、吞吐和数值误差。

## 与 sparse、FlashAttention 和 dense attention 的边界

|方法|是否计算全部 $T\times S$ 位置|是否物化完整 map|标准 softmax 是否保留|长度主项|
|---|---|---|---|---|
|dense attention|是|通常是|是|$TS$|
|FlashAttention|是|否|是，受舍入影响|$TS$|
|sparse attention|只计算选中集合|只保存选中区域|可以保留|$\lvert E\rvert$|
|linear attention|通过摘要重排或近似|否|通常是近似 kernel|$(T+S)r$|

四者优化对象不同：

- dense attention 是标准语义基线；
- FlashAttention 改变 tile 和显存 IO；
- sparse attention 改变 query-key 连接集合；
- linear attention 改变相似度的代数表示，通常也改变标准 softmax 的函数。

一个实现可以先用线性 kernel，再在每个 query 上使用局部候选；也可以用 GQA 减少摘要 group，再使用 causal prefix。组合后的资源要按每个轴重新计数。

## 失效模式与审计方法

### 把结合律当成 softmax 的精确重排

矩阵乘法可以重排，row-wise softmax 的指数和归一化不能任意重排。先写出 feature kernel，再说明它与 $\exp(q^\mathsf Tk)$ 的关系。

### 把有限 feature map 当成无限精确 kernel

有限 $r$ 的 feature map 通常近似 softmax kernel。固定输入上比较 attention output、权重统计、梯度和任务 loss；不要只比较复杂度公式。

### 忽略分母接近零

非正 feature 或长序列累加会让 $\phi(q_t)^\mathsf Tz_t$ 变小。记录最小分母、NaN 数量、epsilon 和 accumulator dtype。

### 任意 mask 仍使用一个全局 summary

如果不同 query 有不同 source visibility，一个全 source summary 会混入不可见 value。检查 prefix、区间、块和动态集合是否有对应的 summary 结构。

### 把 causal recurrence 当成完全并行

推理可以逐 token 更新，但训练要处理 prefix dependency、scan 和反向保存。分别报告训练吞吐、训练峰值和 decode 延迟。

### 把摘要 cache 当成可编辑的 K/V cache

$G,z$ 不能提供单个历史 token 的精确 K/V。beam reorder、删除 token、局部重 mask 和 token 级归因都需要新的状态协议。

### 忽略位置编码的可分离性

relative bias 或 RoPE 的位置依赖可能改变 kernel 的因子化。对固定位置和长度做数值 reference，对不同位置偏移做一致性测试。

### 用 $r$ 代替 $d_k$ 时忘记投影成本

linear attention 的摘要维度是 $r$，但 Q/K 投影和输出投影仍然存在。对比端到端资源时同时列出 projection、feature map、summary 和 output projection。

### 用状态字节数代表全部训练显存

训练还可能保存 $\Phi(Q)$、$\Phi(K)$、prefix state、反向中间量、残差和 MLP 激活。只报告 $G,z$ 会低估峰值。

### 一份最小线性 attention 审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|相似度 kernel|明确 $\kappa(q,k)$ 和 feature map $\phi$|是否仍声称标准 softmax|
|feature 维度|记录 $r$、dtype 和 accumulator|状态大小、算力|
|分母|$\phi(q_t)^\mathsf Tz_t$ 有效且稳定|正性、epsilon、NaN|
|source summary|$G=\sum_s\phi(k_s)v_s^\mathsf T$|矩阵轴、value 转置|
|query read|$\phi(q_t)^\mathsf TG$|输出维度、head reshape|
|dense 对照|同一 Q/K/V/mask 有 reference|误差、归一化|
|causal state|$G_t,z_t$ 只包含允许的 prefix|更新顺序、cache offset|
|mask|能被同一 summary、prefix 或区间结构表示|任意 query mask|
|position|位置机制可因子化或有专门推导|relative bias、RoPE|
|cross source|source summary 是否可复用|source visibility|
|GQA|状态按 $h_{kv}$，query 按 $h_q$|group mapping|
|训练并行|scan、重算和反向路径有记录|prefix dependency|
|decode cache|保存摘要还是独立 K/V|beam reorder、删除 token|
|数值精度|state accumulator 与输入 dtype 分开记录|混合精度溢出|
|质量指标|输出误差、loss、长上下文和任务指标齐全|只看 FLOPs|

## 相关词条

- [注意力复杂度](../attention/attention-complexity/)：核算 dense、sparse、prefill、decode 和 cache 的资源项。
- [缩放点积注意力](../attention/scaled-dot-product/)：定义标准 $QK^\mathsf T/\sqrt{d_k}$、softmax 和 $AV$。
- [Attention 矩阵](../attention/attention-matrix/)：说明标准 attention map 的轴和 row-wise 归一化。
- [稀疏注意力](../attention/sparse-attention/)：说明连接集合、局部窗口和块稀疏。
- [FlashAttention](../attention/flash-attention/)：说明 dense attention 的 tile 与 IO 优化。
- [GQA 与 MQA](../attention/gqa-and-mqa/)：说明 query head 与 K/V group 的共享轴。
- [因果掩码](../attention/causal-masking/)：说明 causal prefix、cache offset 和全 mask 行。
- [交叉注意力](../attention/cross-attention/)：说明 source summary 与 target query 的矩形方向。
