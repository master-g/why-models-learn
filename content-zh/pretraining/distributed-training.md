---
title: "分布式训练：把模型、数据与通信放到多台设备上"
tags: ["why-models-learn"]
---

分布式训练是把一次模型训练拆到多个设备或进程上，并通过明确的 collective 通信保持参数、梯度、激活或专家状态的一致。数据并行复制模型并切分 batch，张量并行切分矩阵和 attention 头，流水线并行切分层，序列或上下文并行切分 token 轴，专家并行切分 MoE 专家，参数分片则切分权重、梯度和 optimizer state。每种方案都改变局部 tensor shape、通信路径、显存账本和 global batch；多卡运行不等于只把单卡脚本复制多份。

本文先固定 rank、张量和 collective 的合同，再推导数据并行的 global gradient、张量并行的列/行切分、流水线并行的 micro-batch bubble、序列与专家并行的通信方向，以及 ZeRO/FSDP 类参数分片的内存收益。最后用一个可复算的混合并行配置核对 global batch、pipeline efficiency、通信时间和每卡状态，并列出 deadlock、重复平均、shape 错位和扩缩容的审计方法。

![分布式训练示意：global batch 先在数据并行副本间切分，模型层再沿张量与流水线轴切分，collective 通信保持局部结果一致](/assets/pretraining/svg/distributed-training.1.svg)

## 先固定分布式合同

### Rank、设备与并行轴

设训练进程的集合为 $\mathcal R$，每个进程有一个 rank 和一张或多张设备。常见并行轴如下：

|并行方式|切分对象|主要通信|
| --- | --- | --- |
|data parallelism, DP|样本或有效 token 的 batch|gradient all-reduce、参数同步或梯度 reduce-scatter|
|tensor parallelism, TP|矩阵的输入/输出轴、attention head|all-gather、all-reduce、reduce-scatter|
|pipeline parallelism, PP|Transformer layer 或 block|stage 间 activation 与 gradient 发送|
|sequence/context parallelism, SP/CP|序列位置或上下文区间|沿 token 轴的 all-gather、ring 或 attention 交换|
|expert parallelism, EP|MoE expert|token dispatch 与 combine 的 all-to-all|
|parameter sharding|参数、梯度、optimizer state|按需 all-gather、reduce-scatter 和重分片|

若 DP、TP、PP、EP 组成笛卡尔 rank 网格，理想 world size 近似为

$$
W
=
W_{\mathrm{DP}}
W_{\mathrm{TP}}
W_{\mathrm{PP}}
W_{\mathrm{EP}}.
$$

SP/CP 有时复用 TP 或 DP 的 rank group，有时使用独立轴；不能只看 world size 推断每个轴的成员。每个进程都应记录 rank、world_size、各轴坐标、各 axis group 和通信 backend。

### Global tensor 与 local tensor

以 decoder-only Transformer 的隐藏状态为例，global tensor 可能是

$$
X
\in
\mathbb R^{B\times T\times D},
$$

其中 $B$ 是 global batch、$T$ 是序列长度、$D$ 是模型宽度。若 DP 切分 batch，单个 DP rank 看到的 local tensor 形状近似为

$$
X^{(r)}
\in
\mathbb R^{B_{\mathrm{local}}\times T\times D}.
$$

若 TP 沿隐藏宽度切分，则局部形状可能是

$$
X^{(r)}
\in
\mathbb R^{B_{\mathrm{local}}\times T\times D/W_{\mathrm{TP}}}.
$$

如果同时存在 SP/CP，$T$ 也可能变成 $T/W_{\mathrm{SP}}$，但 attention 的 query、key、value 交互仍然需要跨 rank 读取缺失的序列区间。每个 kernel 的输入和输出都应写出 global shape、local shape、shard axis 和 collective 之后的 shape。

### Collective 的职责

常见 collective 不只是通信 API 名称，它们决定了数值如何组合：

|操作|输入与输出|典型用途|
| --- | --- | --- |
|all-reduce|每个 rank 有一个 tensor，所有 rank 得到 sum 或 mean|数据并行梯度、张量并行 partial result|
|reduce-scatter|先求和，再把结果分片返回|分片梯度、行并行输出、参数分片|
|all-gather|每个 rank 有一片，所有 rank 得到完整 tensor|恢复完整权重、拼接序列或列并行输出|
|all-to-all|每个 rank 向其他 rank 发送不同片段|MoE token dispatch、专家输出 combine|
|point-to-point|指定 rank 之间发送/接收|流水线 stage 的 activation 和 gradient|

sum 与 mean 不等价。若 collective 返回 sum，后续必须按全局有效样本或 token 数归一化；若每个 rank 先平均、collective 又平均，梯度会被重复除以 world size。

## 数据并行

### 每个副本处理不同 batch

设 DP group 有 $W_{\mathrm{DP}}$ 个 rank，第 $r$ 个 rank 计算本地有效梯度

$$
\mathbf g^{(r)}
=
\frac{1}{Q_r}
\sum_{i\in\mathcal B_r}
m_i\nabla_\theta\ell_i,
$$

其中 $Q_r=\sum_{i\in\mathcal B_r}m_i$ 是本地有效事件数。若各 rank 的有效事件数相同，平均梯度为

$$
\overline{\mathbf g}
=
\frac{1}{W_{\mathrm{DP}}}
\sum_{r=1}^{W_{\mathrm{DP}}}
\mathbf g^{(r)}.
$$

参数更新使用 $\overline{\mathbf g}$ 后，各副本仍保持相同参数：

$$
\boldsymbol\theta_{s+1}^{(r)}
=
\boldsymbol\theta_s^{(r)}
-
\eta_s\overline{\mathbf g}.
$$

如果 $Q_r$ 不相同，应按有效事件数加权：

$$
\overline{\mathbf g}
=
\frac{1}{\sum_r Q_r}
\sum_{r=1}^{W_{\mathrm{DP}}}
Q_r\mathbf g^{(r)}.
$$

直接平均各 rank 的 local mean 只在 $Q_r$ 相同或权重已正确处理时等价。padding、drop-last、变长 packing 和 loss mask 都可能让 $Q_r$ 不同。

### Global batch 的形状

若每个 DP rank 的 micro-batch 有 $B_{\mathrm{micro}}$ 个序列，每个 optimizer update 累积 $K_{\mathrm{acc}}$ 个 micro-batch，global batch 的序列数为

$$
B_{\mathrm{global}}
=
W_{\mathrm{DP}}
B_{\mathrm{micro}}
K_{\mathrm{acc}}.
$$

固定序列长度 $T$ 时，理论 token 数是

$$
T_{\mathrm{global}}
=
W_{\mathrm{DP}}
B_{\mathrm{micro}}
K_{\mathrm{acc}}
T.
$$

实际 loss 分母应替换成有效 token 数：

$$
Q_{\mathrm{global}}
=
\sum_{r=1}^{W_{\mathrm{DP}}}
\sum_{j=1}^{K_{\mathrm{acc}}}
\sum_{i\in\mathcal B_{r,j}}
m_i.
$$

[大规模学习率调度](../pretraining/lr-schedules-at-scale/)使用的 step 和 token 计数必须与这里的 global update 合同一致。只增加设备数而不更新 global batch、warmup 和学习率，训练轨迹通常会改变。

### 数据副本和随机状态

数据并行要求不同 rank 通常读取不重叠的样本索引，但模型初始化、dropout、数据增强和 sampler RNG 又需要可控的差异。应分开记录：

1. 数据 shard、epoch 或 token cursor；
2. rank-specific seed 与 shared seed；
3. sampler 是否在每个 epoch 按 rank 改变顺序；
4. dropout mask 是否允许 rank 间不同；
5. 是否要求单卡与多卡 bitwise 一致，还是只要求统计指标一致。

多卡训练与单卡训练的数学等价需要相同数据顺序、相同 loss reduction、相同梯度归约、相同更新顺序和足够一致的浮点归约。通信顺序和低精度舍入可能仍会产生小差异。

## 张量并行

### 列并行矩阵

设一个线性层为

$$
Y
=
XW,
\qquad
X\in\mathbb R^{M\times D_{\mathrm{in}}},
\quad
W\in\mathbb R^{D_{\mathrm{in}}\times D_{\mathrm{out}}}.
$$

TP 沿输出列切分

$$
W
=
\left[
W_1\;W_2\;\cdots\;W_{W_{\mathrm{TP}}}
\right].
$$

每个 rank 计算

$$
Y_r
=
XW_r
\in
\mathbb R^{M\times D_{\mathrm{out}}/W_{\mathrm{TP}}},
$$

若下一个层需要完整的 $Y$，再做 all-gather：

$$
Y
=
\operatorname{all\text{-}gather}
\left(
Y_1,\ldots,Y_{W_{\mathrm{TP}}}
\right).
$$

如果后续层可以直接读取各列 shard，则可以延后 gather，减少通信。局部矩阵的维度、bias 是否分片和 gather 的时刻都要写进 layer contract。

### 行并行矩阵

沿输入行切分权重：

$$
W
=
\begin{bmatrix}
W_1\\
W_2\\
\vdots\\
W_{W_{\mathrm{TP}}}
\end{bmatrix},
\qquad
X
=
\left[
X_1\;X_2\;\cdots\;X_{W_{\mathrm{TP}}}
\right].
$$

各 rank 计算 partial output

$$
Y_r
=
X_rW_r,
$$

完整输出需要 all-reduce sum：

$$
Y
=
\sum_{r=1}^{W_{\mathrm{TP}}}Y_r.
$$

列并行通常产生拼接，行并行通常产生求和；把 all-gather 与 all-reduce 写反，会得到 shape 可能正确但数值错误的实现。

### Attention 与 FFN 的切分

在 multi-head attention 中，TP 可以把 head 集合分给不同 rank。每个 rank 计算局部的 $Q,K,V$ 和局部 attention output，输出投影再使用行并行或等价的求和路径。需要记录：

|对象|global shape|local shape 或通信|
| --- | --- | --- |
|Q/K/V head|$B\times T\times H\times d_h$|每 rank 负责 $H/W_{\mathrm{TP}}$ 个 head|
|attention output|$B\times T\times D$|局部 head 拼接后进入输出投影|
|FFN up projection|输出宽度切分|局部列并行结果，是否 gather 取决于下一个层|
|FFN down projection|输入宽度切分|partial output 后 all-reduce sum|

TP 不会自动改变 global 参数量。它把计算和状态分片到 rank，通信则出现在层间或子层内。[参数量总账](../transformer-components/parameter-count/)中的参数与 MAC 仍按 global 模型统计。

## 流水线并行

### 把层分到 stage

PP 把连续的 Transformer layer 分给 $P$ 个 stage。每个 stage 只保存自己的参数和激活，并在 stage 边界发送 activation：

$$
h^{(p+1)}_{u}
=
F_p\left(h^{(p)}_{u}\right),
$$

其中 $p$ 是 pipeline stage，$u$ 是 micro-batch 编号。反向时梯度沿相反方向返回。一个 global batch 会拆成 $m$ 个 micro-batch，才能让不同 stage 同时工作。

### Fill、steady state 与 drain

最简单的 fill-drain 调度中，stage 先等待前一 stage 的 activation，再依次执行 forward；最后一个 stage 产生 loss 后，反向梯度从后往前返回。stage 在填充和排空期间有空闲 bubble。

在每个 stage 计算时间近似相同、忽略通信和重计算时，$P$ 个 stage、$m$ 个 micro-batch 的理想 pipeline 利用率可近似为

$$
\rho_{\mathrm{PP}}
\approx
\frac{m}{m+P-1}.
$$

bubble 比例为

$$
1-\rho_{\mathrm{PP}}
\approx
\frac{P-1}{m+P-1}.
$$

增加 micro-batch 数会降低 bubble，但会增加 activation 生命周期、调度复杂度和 global batch。若 stage 计算量不均衡，实际利用率还会受最慢 stage 限制。

### 1F1B 与参数版本

1F1B 调度在进入 steady state 后交替执行一个 forward 和一个 backward，以减少同时保存的 activation。不同调度可能有：

|调度状态|需要核对的内容|
| --- | --- |
|fill|每个 stage 收到的 activation shape 与 micro-batch ID|
|steady state|forward/backward 次序、buffer 数和 stage 间发送|
|drain|最后一个 loss 是否回传到所有 stage|
|参数版本|一个 micro-batch 的 forward 与 backward 是否使用同一参数快照|
|loss 归约|每个 micro-batch 的 token 权重是否正确合并|

如果参数在不同 micro-batch 之间已更新，pipeline 可能出现 stale weight；使用同步更新、交错调度或权重版本缓存时，要说明接受的数值差异。

### 流水线数字例子

取 $P=4$ 个 stage、$m=8$ 个 micro-batch：

$$
\rho_{\mathrm{PP}}
=
\frac{8}{8+4-1}
=
\frac{8}{11}
=
0.727272727273.
$$

理想 bubble 比例为 $0.272727272727$。如果单个 micro-batch 的 stage 计算时间是 $2\ \mathrm{ms}$，只用该近似估计的有效时间比例仍是 $72.727\%$；通信、stage 不均衡、重计算和同步会降低实际值。

当 $m=1$ 时，利用率近似为 $1/P$；这说明只增加 pipeline stage 而不增加 micro-batch，吞吐会受到空闲阶段限制。

## 序列并行与上下文并行

### token 轴也可以切分

当 batch、hidden size 和 layer 数已经受显存限制时，可以沿序列轴切分：

$$
X
\in
\mathbb R^{B\times T\times D}
\longrightarrow
X_r
\in
\mathbb R^{B\times T/W_{\mathrm{SP}}\times D}.
$$

FFN 或逐 token 的 normalization 可以直接处理局部序列；但 full attention 中 query 位置需要读取 key/value 的其他序列区间。实现可能使用 all-gather、ring attention 或分块通信，必须说明：

1. query、key、value 各自的 local sequence range；
2. causal mask 是否使用 global position；
3. position ID、padding mask 和 loss mask 是否随 shard 保留；
4. attention 输出如何按原 token 顺序拼接；
5. KV 或 activation 的通信是否与计算重叠。

sequence parallel 与 context parallel 有不同的实现口径：前者常与 TP 共享局部 activation 轴，后者更关注长上下文的 attention 通信。名称不能替代 shape 和通信合同。

## 专家并行

### Token dispatch 的方向

MoE router 为每个 token 选择 expert 后，expert 可能分布在不同 rank。每个源 rank 需要把 token payload 发送到目标 expert rank：

$$
\mathcal X_r
\xrightarrow{\mathrm{all\text{-}to\text{-}all}}
\mathcal X_e
\xrightarrow{\mathrm{expert}}
\mathcal Y_e
\xrightarrow{\mathrm{all\text{-}to\text{-}all}}
\mathcal Y_r.
$$

通信 buffer 不只包含 hidden state，还可能包含 token index、expert index、gate 权重和 capacity padding。dispatch 的发送 bytes 与 combine 的返回 bytes 应分开记录。

### Capacity 与负载

若每个 expert 的 capacity 为 $C_e$，路由超过 capacity 的 token 可能被丢弃、转给其他 expert 或使用残差路径。要分别记录：

|轴|指标|
| --- | --- |
|token|每个 token 的 top-$k$ assignment 和 drop 状态|
|expert|每个 expert 接收的 token 数、计算时间和容量使用率|
|rank|每个 rank 的发送/接收 bytes 和等待时间|
|collective|all-to-all 的启动次数、消息大小和尾部延迟|

[混合专家模型](../transformer-architectures/mixture-of-experts/)中的 load-balancing loss 只能约束路由倾向，不能替代实际通信和 capacity 统计。

## 参数、梯度与 optimizer state 分片

### 复制与分片的区别

普通数据并行在每个 rank 复制完整参数、梯度和 optimizer state。参数同步只要求每个 rank 得到相同更新；显存占用不会随 DP rank 数理想下降。

参数分片把状态沿 DP group 分开保存。以 $P$ 个参数为例，若权重和梯度使用 2 bytes，Adam 的两个 moment 各使用 4 bytes，则忽略 master weight、activation 和 buffer 时：

$$
M_{\mathrm{full}}
=
P(2+2+4+4)
=
12P\ \mathrm{bytes}.
$$

理想地分给 $W_{\mathrm{DP}}$ 个 rank 后，每卡状态为

$$
M_{\mathrm{shard}}
\approx
\frac{12P}{W_{\mathrm{DP}}}.
$$

实际实现还需要 all-gather 参数以完成 forward，reduce-scatter 梯度以保存 local shard，并处理参数预取、释放、通信 buffer 和碎片化。

### ZeRO 与 FSDP 的阶段

可以按被分片的状态区分策略：

|阶段|分片对象|前向与反向的通信|
| --- | --- | --- |
|数据并行复制|无状态分片|梯度 all-reduce，参数全量常驻|
|optimizer state sharding|optimizer state|梯度归约后各 rank 保存自己的 state shard|
|gradient sharding|optimizer state 与梯度|reduce-scatter 后各 rank 只保留局部梯度|
|parameter sharding|optimizer state、梯度与参数|按 layer all-gather 参数，反向后 reduce-scatter 梯度|

ZeRO、FSDP 和其他 sharding 实现的具体通信时机不同，不能只按阶段名称估算峰值显存。审计时应记录 layer materialization 的粒度、prefetch、reshard、通信 buffer 和是否与计算重叠。

### 一亿与十亿参数的账本

取 $P=10^9$、BF16 权重和梯度、FP32 Adam moments：

$$
\begin{aligned}
M_{\mathrm{weights}}
&=
2\,000\,000\,000\ \mathrm{bytes},\\
M_{\mathrm{grad}}
&=
2\,000\,000\,000\ \mathrm{bytes},\\
M_{\mathrm{moments}}
&=
8\,000\,000\,000\ \mathrm{bytes},\\
M_{\mathrm{full}}
&=
12\,000\,000\,000\ \mathrm{bytes}
\approx
11.175870895386\ \mathrm{GiB}.
\end{aligned}
$$

若只把这三类状态理想分给 $W_{\mathrm{DP}}=8$ 个 rank，每卡是 $1\,500\,000\,000$ bytes，约 $1.396983861923$ GiB。这个数字没有包含 activation、KV、临时 all-gather buffer、master parameter、通信库 workspace、allocator 碎片和其他模型状态，所以它不是设备显存需求的最终值。

## 通信与设备拓扑

### Ring all-reduce 的一阶账本

对每个 rank 持有 $M$ bytes、通信 group 有 $p$ 个 rank 的 ring all-reduce，可用以下一阶模型估算：

$$
T_{\mathrm{allreduce}}
\approx
2(p-1)\alpha
+
\frac{2(p-1)}{p}\frac{M}{\mathcal B},
$$

其中 $\alpha$ 是一次通信启动的延迟，$\mathcal B$ 是有效带宽。第一项来自 reduce-scatter 和 all-gather 的启动次数，第二项是每个 rank 的传输量。真实时间还受 topology、消息分片、协议、竞争、同步和计算重叠影响。

取 $p=8$、$M=256\ \mathrm{MiB}$、$\alpha=2\ \mu\mathrm{s}$、$\mathcal B=100\ \mathrm{GB/s}$：

$$
T_{\mathrm{allreduce}}
\approx
2(7)(2\times10^{-6})
+
\frac{14}{8}
\frac{256\times2^{20}}{100\times10^9}
=
0.004725620480\ \mathrm{s}.
$$

这只是约 $4.725620480\ \mathrm{ms}$ 的无重叠估算。应与 profiler 的 NCCL、RCCL 或其他 backend 事件比较，不应把带宽标称值直接当作端到端训练吞吐。

### Group 与拓扑要匹配

并行轴不应任意分配 rank。通常希望：

1. TP 的高频小消息留在高速互联设备内；
2. PP 的 activation 发送映射到低延迟相邻设备；
3. DP 的梯度 collective 使用足够带宽的跨节点网络；
4. EP 的 all-to-all 避免把热点集中到少数链路；
5. parameter sharding 的 all-gather 与 reduce-scatter 不和其他大消息争用。

同一个 world size 可以有不同 rank mapping，性能差异来自 group 成员与物理 topology 的对应关系。启动日志应同时保存逻辑 rank 坐标、host、device、通信 group 和链路类型。

### 计算与通信重叠

若某 layer 的通信可以与下一段计算重叠，单步时间的一阶上界可能接近

$$
T_{\mathrm{step}}
\approx
\max\left(
T_{\mathrm{compute}},
T_{\mathrm{communication}}
\right).
$$

没有重叠时则更接近两者之和。重叠需要非阻塞 collective、正确的 stream/event 依赖和足够的 buffer；提前释放 buffer、错误同步或 collective 顺序不一致会把重叠变成数据竞争或 deadlock。

## 运行方法

### 形状与 global batch

取

$$
W_{\mathrm{DP}}=4,
\quad
W_{\mathrm{TP}}=2,
\quad
W_{\mathrm{PP}}=2,
\quad
B_{\mathrm{micro}}=2,
\quad
K_{\mathrm{acc}}=3,
\quad
T=8.
$$

完整的模型并行 world size 为 $2\times2=4$，若每个模型副本还复制到 4 个 DP rank，则总 rank 数为

$$
W
=
4\times2\times2
=
16.
$$

每个 optimizer update 的 global sequence 数和理论 token 数是

$$
\begin{aligned}
B_{\mathrm{global}}
&=
W_{\mathrm{DP}}B_{\mathrm{micro}}K_{\mathrm{acc}}
=
4\times2\times3
=
24,\\
T_{\mathrm{global}}
&=
B_{\mathrm{global}}T
=
24\times8
=
192.
\end{aligned}
$$

TP 和 PP 不应再乘进 global batch；它们复制的是同一个 batch 的计算路径。若 padding 或 loss mask 使有效事件只有 176 个，学习率、loss 分母和 token-based scheduler 应记录 176，而不是把 192 当作有效 token。

### 运行输出

下面的标准库脚本核对 global batch、pipeline bubble、状态分片和 ring all-reduce 估算：

```python
import math

dp, tp, pp = 4, 2, 2
micro_batch, accumulation, seq_len = 2, 3, 8
global_sequences = dp * micro_batch * accumulation
global_tokens = global_sequences * seq_len

pipeline_microbatches = 8
pipeline_efficiency = pipeline_microbatches / (
    pipeline_microbatches + 4 - 1
)

parameters = 1_000_000_000
full_bytes = parameters * (2 + 2 + 4 + 4)
sharded_bytes = full_bytes / 8

ranks = 8
message_bytes = 256 * 2**20
latency = 2e-6
bandwidth = 100e9
ring_seconds = (
    2 * (ranks - 1) * latency
    + 2 * (ranks - 1) / ranks * message_bytes / bandwidth
)

print(global_sequences, global_tokens)
print(f"{pipeline_efficiency:.12f}")
print(f"{full_bytes / 2**30:.12f}")
print(f"{sharded_bytes / 2**30:.12f}")
print(f"{ring_seconds * 1000:.12f}")
```

输出为：

```text
24 192
0.727272727273
11.175870895386
1.396983861923
4.725620480000
```

这些数字只核对逻辑账本。真实运行还要测 activation、通信、设备利用率、stage imbalance、collective tail latency 和有效 token mask。

## 同步、精度与故障边界

### Collective 顺序必须一致

collective 通常要求同一 group 的 rank 以相同顺序进入相同操作。以下差异可能导致永久等待：

|差异|表现|
| --- | --- |
|一个 rank 因异常跳过 all-reduce|其他 rank 在 collective 中等待|
|不同 rank 的 conditional branch 不同|后续 collective 顺序错位|
|某 stage 少发送一个 micro-batch|pipeline 接收端等待|
|EP route 产生不同 dispatch count|all-to-all payload 或 metadata 不匹配|
|checkpoint 恢复的 group size 不同|旧 rank 坐标与新 group 不一致|

遇到 hang 时先记录每个 rank 最后进入的 collective、group、tensor shape、payload bytes 和 stream event，不要只增加 timeout。

### 全局梯度裁剪

若要裁剪 global gradient norm，应先按并行方式汇总平方和：

$$
\lVert\mathbf g\rVert_2
=
\sqrt{
\sum_{r}
\lVert\mathbf g^{(r)}\rVert_2^2
}.
$$

TP、PP 和 parameter sharding 下，某个 rank 只持有梯度的一部分；局部裁剪再通信不等于全局裁剪。[梯度裁剪](../training-nn/gradient-clipping/)中的阈值和归约顺序要与分布式实现一致。

### Mixed precision 与 collective

低精度通信可以减少 bytes，但归约精度、overflow 检测和梯度 scaler 仍需明确：

1. all-reduce 前是否 unscale；
2. sum、mean 和 loss denominator 如何对应；
3. collective buffer 使用 FP16、BF16 还是 FP32；
4. 非有限值是否在所有 rank 同步；
5. 某 rank overflow 时是否所有 rank 同步跳过 update。

只在本地检测 NaN 而不广播失败状态，会让 rank 的 optimizer step 和 scheduler step 分叉。

## 失效模式

**把 DP、TP、PP 只当作设备数量。** 写出每个轴的 rank group、local shape、global shape 和 collective。

**把 TP 或 PP 乘进 global batch。** TP 和 PP 通常共同执行同一个 batch；global batch 主要由 DP、micro-batch 和 accumulation 决定。

**对 local mean 再做一次 mean。** 记录有效 token 分母，区分 local sum、local mean、all-reduce sum 和 all-reduce mean。

**把 all-gather 与 all-reduce 混用。** 列并行通常拼接输出，行并行通常求和；验证数值和 shape，不只验证通信完成。

**stage 不均衡却只增加 micro-batch。** 记录每个 stage 的 forward、backward、通信和 idle 时间；最慢 stage 仍然决定吞吐。

**pipeline bubble 占用主要时间。** 计算 $m/(m+P-1)$，再检查 activation 内存和实际调度，不把理论利用率写成实测吞吐。

**忽略 MoE dispatch。** 记录 expert、rank、token、bytes、capacity 和 all-to-all 等待时间；主任务 loss 不能说明路由通信健康。

**把理想分片内存当作显存需求。** 加上 activation、buffer、master weight、KV、通信 workspace 和碎片化，再做设备级测量。

**collective 顺序不一致。** 保存每个 rank 的 collective trace；异常路径必须同步失败和释放。

**扩缩容后沿用旧 scheduler 和 data cursor。** 重新计算 global batch、有效 token、rank group、学习率进度和 checkpoint 兼容性。

## 审计清单

一次可复核的分布式训练至少应保留：

|问题|证据|
| --- | --- |
|模型如何切分|DP、TP、PP、SP/CP、EP 的 group、坐标和 local shape|
|global batch 如何形成|DP rank、micro-batch、accumulation、有效 token 和 loss denominator|
|梯度如何归约|sum/mean、all-reduce/reduce-scatter、裁剪与 unscale 顺序|
|参数如何同步|参数版本、optimizer state、all-gather、reshard 和 checkpoint|
|通信是否成为瓶颈|collective bytes、延迟、带宽、overlap、idle 和 tail latency|
|流水线是否均衡|stage 计算时间、micro-batch 数、bubble 和 activation buffer|
|MoE 是否可扩展|expert/rank load、dispatch bytes、capacity、drop 和 all-to-all|
|故障是否一致|每个 rank 的非有限值、跳过更新、最后 collective 和恢复状态|
|结论覆盖什么|吞吐、训练损失、验证损失、内存、成本或下游指标|

如果这些证据缺失，最多能确认程序在多台设备上运行，不能确认它实现了预期的 global gradient、global batch 或并行效率。

## 相关词条

- [预训练](../pretraining/pretraining/)：说明分布式训练消费 token、更新参数和保存 checkpoint 的上下文。
- [计算最优](../pretraining/compute-optimal/)：把模型规模、数据规模和训练计算放到同一预算中比较。
- [大规模学习率调度](../pretraining/lr-schedules-at-scale/)：定义 global step、有效 token 和 scheduler 恢复。
- [参数量总账](../transformer-components/parameter-count/)：区分 global 参数量、active 参数、MAC、激活和 optimizer state。
- [混合专家模型](../transformer-architectures/mixture-of-experts/)：展开 router、expert capacity、dispatch 与 combine。
- [混合精度训练](../training-nn/mixed-precision/)：处理低精度状态、loss scaling、overflow 和通信归约。
- [梯度裁剪](../training-nn/gradient-clipping/)：定义全局范数与归约顺序。
- [训练稳定性](../pretraining/training-stability/)：记录非有限值、梯度异常和训练中断。
- [小批次随机梯度下降](../training-nn/minibatch-sgd/)：解释 global batch、梯度噪声与数据并行的基础。
