---
title: "Seq2Seq：用编码器状态条件化解码器"
---

Seq2Seq（sequence-to-sequence）模型把一条输入序列映射为另一条可能长度不同的输出序列。经典 RNN Seq2Seq 由两个循环网络组成：编码器按输入顺序读取序列，把历史压缩为 context；解码器以这个 context 为条件，逐步生成输出。训练时常用真实目标前缀（teacher forcing），推理时却只能使用自己已经生成的 token，这个差异构成了 exposure bias。

本篇只把经典 encoder-decoder 的计算图和训练/推理边界讲清楚。注意力会在后续词条中作为对固定 context 瓶颈的结构性修复出现。阅读时要持续区分：

1. 输入长度和输出长度可以不同；
2. decoder 的每一步都条件化于输入序列和已生成输出前缀；
3. teacher forcing 只改变训练时喂给 decoder 的前一 token；
4. beam search 只改变推理时如何选择前缀，不改变训练目标；
5. padding、BOS、EOS 和 loss 分母决定「一条序列」到底如何进入梯度。

![Seq2Seq 编码器、context 与解码器](/assets/rnn-lstm/svg/seq2seq.1.svg)

## 从输入序列到输出序列

设源序列为

$$
x_{1:S}=(x_1,x_2,\ldots,x_S),
$$

目标序列为

$$
y_{1:T}=(y_1,y_2,\ldots,y_T).
$$

编码器的递推可以写成

$$
s_i=f_{\mathrm{enc}}(s_{i-1},x_i),
\qquad
i=1,\ldots,S.
$$

最小的固定 context 取编码器末状态：

$$
c=s_S.
$$

解码器以一个初始状态 $d_0$ 和起始 token BOS 开始。最简单的条件递推为

$$
d_j=f_{\mathrm{dec}}(d_{j-1},u_j,c),
\qquad
j=1,\ldots,T,
$$

其中 $u_j$ 是第 $j$ 步喂入 decoder 的前一 token。训练时通常令

$$
u_1=\mathrm{BOS},
\qquad
u_j=y_{j-1}\quad(j\ge2).
$$

输出头把 $d_j$ 映射为词表上的 logits：

$$
\ell_j=W_od_j+b_o,
\qquad
p_j=\operatorname{softmax}(\ell_j).
$$

因此模型表达的条件概率是

$$
p(y_{1:T}\mid x_{1:S})
=
\prod_{j=1}^{T}p(y_j\mid y_{<j},x_{1:S}),
$$

其中

$$
y_{<j}=(y_1,\ldots,y_{j-1}).
$$

这不是把源序列和目标序列按位置一一相乘。encoder 负责读完源序列，decoder 负责在 context 和目标前缀条件下产生另一个时间轴。

| 对象 | 时间轴 | 典型长度 | 作用 |
| --- | --- | ---: | --- |
| $x_i$ | encoder time | $S$ | 源序列 token 或特征 |
| $s_i$ | encoder state | $S$ 个状态 | 累积源序列历史 |
| $c$ | bridge | 1 个向量 | 把编码器信息交给 decoder |
| $u_j$ | decoder input | $T$ 个 token | BOS 或上一目标 token |
| $d_j$ | decoder state | $T$ 个状态 | 累积目标前缀和 context |
| $y_j$ | decoder target | $T$ 个 token | 当前步监督目标 |

## Bridge：context 如何初始化 decoder

最简单的 bridge 有三种常见写法：

| bridge | decoder 初始条件 | 风险或代价 |
| --- | --- | --- |
| final-state copy | $d_0=s_S$ | encoder/decoder hidden width 必须匹配，长源序列压缩压力大 |
| learned projection | $d_0=W_cs_S+b_c$ | 增加参数，需检查 projection shape |
| separate context input | 每个 decoder step 都读 $c$ | 状态初始化不再是唯一注入点，但固定向量瓶颈仍在 |

如果 encoder 和 decoder 使用不同 hidden width，不能直接把 $s_S$ 当成 $d_0$：

$$
s_S\in\mathbb R^{d_e},
\qquad
d_0=W_cs_S+b_c\in\mathbb R^{d_d}.
$$

即使每个 decoder step 都把 $c$ 拼到输入，固定 context 仍然承载所有源信息：

$$
d_j=f_{\mathrm{dec}}\bigl(d_{j-1},[u_j;c]\bigr).
$$

与「只初始化一次」相比，这改变了读取频率，却没有让 decoder 访问每个 encoder 状态 $s_i$。后续注意力的关键正是把固定向量改成随 decoder step 变化的加权读取。

### 双向 encoder 的维度

若 encoder 有正向和反向两条循环路径：

$$
\overrightarrow s_i=f_{\rightarrow}(\overrightarrow s_{i-1},x_i),
\qquad
\overleftarrow s_i=f_{\leftarrow}(\overleftarrow s_{i+1},x_i),
$$

可以拼接

$$
s_i=[\overrightarrow s_i;\overleftarrow s_i].
$$

末状态 context 的维度变为 $2d_e$。若 decoder width 是 $d_d$，bridge 需要明确写成

$$
d_0=W_c[\overrightarrow s_S;\overleftarrow s_1]+b_c.
$$

不能因为「双向 encoder 最后有两个状态」就把其中一个静默丢掉或错误相加。

## Teacher forcing：训练时的输入不是模型自己的输出

训练时，teacher forcing 使用真实前一 token：

$$
u_j^{\mathrm{train}}=
\begin{cases}
\mathrm{BOS},&j=1,\\
y_{j-1},&j\ge2.
\end{cases}
$$

于是第 $j$ 步的预测条件是正确的目标前缀，即使模型在第 $j-1$ 步预测错了，错误也不会自动传给下一步。

推理时没有真实目标，必须使用已经生成的 token：

$$
u_j^{\mathrm{infer}}=\widehat y_{j-1}.
$$

两种输入路径是：

| 阶段 | decoder 前一输入 | 错误是否进入下一步 | 主要含义 |
| --- | --- | --- | --- |
| teacher forcing | 真实 $y_{j-1}$ | 通常不会 | 训练梯度稳定、条件前缀正确 |
| autoregressive inference | 生成 $\widehat y_{j-1}$ | 会 | 评估真实部署轨迹 |
| scheduled sampling | 真实与生成按规则混合 | 部分会 | 改变训练分布和梯度语义 |

### 暴露偏差的一个数字

假设第一步模型在两个候选 token 上给出

$$
p(A\mid\mathrm{BOS},x)=0.55,
\qquad
p(B\mid\mathrm{BOS},x)=0.45.
$$

如果第一步选 $A$，下一步正确延续概率只有 0.20；如果第一步选 $B$，下一步正确延续概率为 0.80。贪心解码会先选 $A$，得到两步前缀概率

$$
0.55\times0.20=0.11.
$$

另一条以较小首步概率开始的路径却有

$$
0.45\times0.80=0.36.
$$

teacher forcing 训练第 2 步时直接喂真实的 $y_1$，可能从未暴露过「第 1 步错成 $A$ 后如何恢复」的状态。beam search 可以保留 $B$ 前缀，但它不能替代训练分布的诊断。

## 训练目标：每个目标位置的条件 NLL

给定目标 token $y_j$，交叉熵损失为

$$
\ell_j=-\log p(y_j\mid y_{<j},x_{1:S}).
$$

一条没有 padding 的序列可以使用 sum 或 mean：

$$
\mathcal L_{\mathrm{sum}}=\sum_{j=1}^{T}\ell_j,
\qquad
\mathcal L_{\mathrm{mean}}=\frac1T\sum_{j=1}^{T}\ell_j.
$$

对 batch 中不同长度的目标，应使用有效位置 mask $M_{m,j}$：

$$
\mathcal L
=
-\frac{\sum_{m,j}M_{m,j}\log p(y_{m,j}\mid y_{m,<j},x_{m,1:S_m})}
{\sum_{m,j}M_{m,j}}.
$$

如果把 padding token 也放进分母，长短样本的相对权重会改变；如果只 mask loss 但仍让 padding 推进 decoder state，前向和反向图仍可能被污染。

### BOS、EOS 和 shift

目标 token 的监督通常需要一位错开：

| decoder step | 喂入 $u_j$ | 监督 $y_j$ |
| ---: | --- | --- |
| 1 | BOS | 第一个真实 token |
| 2 | 第一个真实 token | 第二个真实 token |
| 3 | 第二个真实 token | 第三个真实 token |
| $T$ | 上一个真实 token | EOS |

这里的 $T$ 包括 EOS 位置，但不把 BOS 当成需要预测的目标。常见错误有：

1. 用 $y_j$ 预测 $y_j$，造成目标泄漏；
2. 漏掉 EOS，模型没有停止信号；
3. 把 BOS 也计入 loss，改变了分母；
4. logits 和 target 长度相差一位却靠截断静默修复。

最小单元测试应打印一条短目标的完整 shift 表，而不是只检查 tensor shape。

### 三个 token 的 NLL 账本

假设三步真实目标 token 的条件概率分别是

$$
p_1=0.7,\qquad p_2=0.4,\qquad p_3=0.9.
$$

联合概率是

$$
p(y_{1:3}\mid x)=0.7\times0.4\times0.9=0.252.
$$

sum NLL 为

$$
-\log(0.7)-\log(0.4)-\log(0.9)
=1.378326191471,
$$

mean token NLL 为

$$
\frac{1.378326191471}{3}=0.459442063824.
$$

若第三步的 padding 被错误地加入分母，分母变成 4，数值会被写成 0.344581547868；这不是同一个训练目标。

## Encoder 和 decoder 的梯度路径

最终 loss 对 encoder 末状态的梯度来自所有 decoder 时间步：

$$
\frac{\partial\mathcal L}{\partial c}
=
\sum_{j=1}^{T}
\frac{\partial\mathcal L_j}{\partial c}.
$$

如果 $c=s_S$，梯度还要沿 encoder 时间轴回传：

$$
\frac{\partial\mathcal L}{\partial s_i}
=
\sum_{j=1}^{T}
\frac{\partial\mathcal L_j}{\partial c}
\frac{\partial c}{\partial s_i}
$$
（在固定 final-state bridge 下，$i=S$ 的直接项最明显）。

更准确地说，decoder 每一步通过 $d_j$ 和可能的 context 输入影响 loss，decoder 的状态梯度反向到 $d_0$，再经 bridge 回到 $s_S$。随后 encoder 仍然面对从 $S$ 到较早输入的时间 Jacobian 连乘。因此 Seq2Seq 并没有消除 RNN 的梯度问题，只是把它拆成：

1. decoder 目标时间轴；
2. bridge/context 路径；
3. encoder 源时间轴。

注意力会减少「所有源信息必须进入 $s_S$」的瓶颈，但 decoder 的自回归路径和 encoder/decoder 的局部梯度仍要审计。

### 参数共享与两条时间轴

encoder 参数在 $i=1,\ldots,S$ 间共享，decoder 参数在 $j=1,\ldots,T$ 间共享。它们可以不同：

$$
\theta_{\mathrm{enc}}\ne\theta_{\mathrm{dec}}.
$$

参数梯度分别按各自时间轴累加：

$$
\frac{\partial\mathcal L}{\partial\theta_{\mathrm{enc}}}
=\sum_{i=1}^{S}
\frac{\partial\mathcal L}{\partial s_i}
\frac{\partial s_i}{\partial\theta_{\mathrm{enc}}},
\qquad
\frac{\partial\mathcal L}{\partial\theta_{\mathrm{dec}}}
=\sum_{j=1}^{T}
\frac{\partial\mathcal L}{\partial d_j}
\frac{\partial d_j}{\partial\theta_{\mathrm{dec}}}.
$$

把 encoder 和 decoder 的参数拼成一个大矩阵并不意味着它们可以共享同一时间索引。形状账和时间账要分开。

## 推理：从局部概率到整条序列

decoder 每一步输出一个词表分布。常见解码方式包括：

| 方法 | 选择规则 | 优点 | 风险 |
| --- | --- | --- | --- |
| greedy | 每步取最大概率 token | 快、实现简单 | 局部最优可能堵住全局较好前缀 |
| sampling | 按分布采样 | 能表达多样性 | 结果随机，需控制温度和停止 |
| beam search | 保留 top-$B$ 个前缀 | 近似搜索更大前缀空间 | 计算量和长度偏好更复杂 |

一条未归一化的 beam score 通常是

$$
\operatorname{score}(y_{1:j})
=\sum_{k=1}^{j}\log p(y_k\mid y_{<k},x).
$$

因为 log 概率非正，序列越长通常分数越小，beam search 可能偏好过早 EOS。长度归一化可以写成一种可配置的形式：

$$
\operatorname{score}_{\alpha}(y_{1:j})
=
\frac{\sum_{k=1}^{j}\log p(y_k\mid y_{<k},x)}
{\left(\frac{5+j}{6}\right)^{\alpha}}.
$$

这不是无害的后处理；改变 $\alpha$ 会改变搜索目标。部署审计需要报告 beam width、长度惩罚、最小/最大长度、EOS 规则和重复惩罚。

### EOS 和停止

生成到 EOS 后，路径可以结束，但 batch 中其他 beam 可能仍继续。要明确：

1. EOS 是否计入最终 score；
2. 结束 beam 是否从候选集合移出；
3. 达到 max length 是否强制终止；
4. 没有 EOS 的路径如何处理；
5. BOS、EOS、PAD 是否共享词表 embedding。

一个输出字符串看起来正常，不足以证明停止逻辑没有泄漏。

## 长度与形状账

### Encoder

若 token embedding 维度为 $d_e$、encoder hidden 为 $d_h$：

$$
X\in\mathbb R^{B\times S\times d_e},
\qquad
S_{\mathrm{enc}}\in\mathbb R^{B\times S\times d_h}.
$$

### Decoder

若目标 embedding 维度为 $d_y$、decoder hidden 为 $d_d$、词表大小为 $V$：

$$
U\in\mathbb R^{B\times T\times d_y},
\qquad
D\in\mathbb R^{B\times T\times d_d},
\qquad
L\in\mathbb R^{B\times T\times V}.
$$

loss target 形状是

$$
Y\in\{0,\ldots,V-1\}^{B\times T}.
$$

如果使用 batch-first 还是 time-first，所有 encoder/decoder 轴必须一致：

| 布局 | encoder 输入 | decoder logits | 常见错位 |
| --- | --- | --- | --- |
| batch-first | $B\times S\times d_e$ | $B\times T\times V$ | 把 $S$ 当成 $T$ |
| time-first | $S\times B\times d_e$ | $T\times B\times V$ | transpose 后 target 未同步 |

### 不同长度的 batch

对 batch 中样本 $m$，源长度是 $S_m$，目标长度是 $T_m$。padding 后的矩形张量可以有

$$
M^{\mathrm{src}}_{m,i}
=\boldsymbol1_{\{i\le S_m\}},
\qquad
M^{\mathrm{tgt}}_{m,j}
=\boldsymbol1_{\{j\le T_m\}}.
$$

两个 mask 的时间轴不同。拿 source mask 去 mask target loss，或把 target mask 用在 encoder state 上，都会产生形状可能正确、语义错误的结果。

## 变体边界

### 只初始化 decoder 和每步输入 context

固定 context 可以只进入 $d_0$，也可以在每个 decoder step 拼接。前者的状态链更紧凑，后者缓解了 decoder 忘掉 context 的风险，但两者都无法按需选择不同源位置。

### Stacked encoder/decoder

多层 encoder 的 $s_S$ 可能是一个状态 tuple；多层 decoder 的 $d_0$ 也可能需要逐层 bridge：

$$
d_0^{(\ell)}=W_c^{(\ell)}s_S^{(\ell)}+b_c^{(\ell)}.
$$

只把最顶层状态传给 decoder，和把每层状态都传入，是两个不同的模型。

### Bidirectional encoder

双向 encoder 对离线翻译、摘要等任务可以使用整条源序列；如果任务要求在线输出，反向路径会读取未来源 token，不能直接当作 causal encoder。

### Attention 的动机

固定 context 迫使 encoder 把长源序列压成一个向量。attention 会让 decoder 第 $j$ 步读取

$$
c_j=\sum_{i=1}^{S}\alpha_{j,i}s_i,
\qquad
\sum_{i=1}^{S}\alpha_{j,i}=1,
$$

从而把源时间轴上的信息通过可变权重传给 decoder。它改变了图结构，后续需要单独讨论 score、softmax、mask 和对齐解释。Seq2Seq 的固定 context 仍是理解这次结构变化的基线。

## 失效模式：训练 loss 下降不代表生成链正确

### Target leakage

如果 decoder 第 $j$ 步输入了真实 $y_j$，模型可以直接利用当前答案，而不是根据 $y_{<j}$ 预测 $y_j$。检查 shift 表和第一个 token 的 BOS 位置。

### Teacher forcing 训练、自由运行评估

训练 loss 只在真实前缀上测量，推理却在模型自己的前缀上运行。要同时记录 teacher-forced NLL 和 free-running 的序列级指标，不能只看一个下降曲线。

### EOS/PAD 分母错误

EOS 漏掉会让模型不会停止，PAD 算入分母会改变不同长度样本的权重。固定一条含 EOS 和 PAD 的小 batch，手算有效 token 数。

### Encoder/decoder 状态错接

把 encoder final hidden 接到 decoder cell，或者把双向两个方向按错误顺序拼接，可能在 shape 上通过却破坏语义。打印每层 bridge 的输入输出 shape 和数值范围。

### 变长 mask 只屏蔽了 loss

decoder 在 PAD 位置继续更新 state，再在 loss 中 mask，并不等价于完全跳过 PAD。若状态更新依赖 padding embedding，后续有效位置仍会被污染。

### Greedy 误判模型质量

greedy 生成差不一定说明局部概率差，也可能是搜索路径问题；beam 更好也不一定说明训练目标正确，可能只是长度惩罚或 beam 宽度改变。把 teacher-forced token accuracy、free-running、greedy/beam 和 EOS 统计分开。

### 把 attention 的结论提前套回 fixed context

没有 attention 时，decoder 只能读取固定 $c$ 或其变换。若解释某个输出 token 时直接声称模型读取了源位置 $i$，却没有显式 attention 权重，那是超出了该模型图的证据。

### 复用状态造成跨样本泄漏

训练 batch、验证 batch 和推理请求之间复用 encoder/decoder hidden，可能让前一个样本影响后一个样本。独立请求、打乱 batch 顺序、显式 reset 是必要测试。

## Seq2Seq 审计协议

1. **画两条时间轴。** 分别标出 $x_{1:S}$、$s_{1:S}$、context、$u_{1:T}$、$d_{1:T}$ 和 $y_{1:T}$。
2. **固定 bridge。** 写出 $d_0$ 如何由 encoder 状态得到，记录双向和多层拼接顺序。
3. **打印 shift 表。** 核对 BOS、真实前缀、target、EOS 和 PAD 的每一位对齐。
4. **核对条件分解。** 确认第 $j$ 步只条件于 $x_{1:S}$ 和 $y_{<j}$，没有读到 $y_j$。
5. **分别算 loss 分母。** 区分 sum、mean、有效 token mask 和 EOS 是否计入。
6. **对比两种运行态。** 记录 teacher forcing 与 free-running 的 token/sequence 指标。
7. **检查推理搜索。** 报告 greedy、sampling 或 beam 的宽度、长度惩罚、EOS 和最大长度。
8. **按时间轴看梯度。** 分别检查 decoder 时间链、bridge 和 encoder 时间链的梯度范数。
9. **验证变长 batch。** 用不同 $S_m,T_m$ 的小 batch 检查 source/target mask 不串轴。
10. **做状态隔离。** reset、交换 batch 顺序、独立请求，确认没有跨样本 hidden 泄漏。
11. **做极小过拟合。** 固定两三条短样本，先验证 shift、loss 和自由运行生成，再扩大数据。

Seq2Seq 的基本结构是「一条序列读完，再以 context 为条件生成另一条序列」。它把 RNN 的时间传播分成 encoder 和 decoder 两段，又在训练与推理之间引入 teacher forcing 的分布差异。掌握 BOS/EOS、mask、bridge、beam 和两条时间轴之后，下一篇 teacher forcing 才能准确讨论：到底是在帮助优化，还是在把部署时会遇到的错误前缀藏起来。

## 相关词条

[序列建模](../rnn-lstm/sequence-modeling/)

[循环神经网络](../rnn-lstm/rnn/)

[门控循环单元](../rnn-lstm/gru/)

[长短期记忆网络](../rnn-lstm/lstm/)

[时间反向传播](../rnn-lstm/bptt/)

[Teacher forcing](../rnn-lstm/teacher-forcing/)

[双向循环网络](../rnn-lstm/bidirectional-rnn/)

[Bahdanau 注意力](../rnn-lstm/bahdanau-attention/)

[为什么需要注意力](../rnn-lstm/why-attention/)
