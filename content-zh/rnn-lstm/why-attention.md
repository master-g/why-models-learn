---
title: "为什么需要注意力：从固定摘要到可寻址的读取"
tags: ["why-models-learn"]
---

注意力（attention）出现的直接原因，是经典 RNN encoder-decoder 把长度为 $S$ 的源序列压成一个固定 context，再让 decoder 只通过这个向量生成长度为 $T$ 的目标序列。短序列上，这个接口简单有效；源序列变长、需要重排或复制多个局部事实时，它把所有证据挤进同一条信息通道，造成容量、梯度和优化上的瓶颈。

注意力的结构性改变是保留源时间轴上的多个状态，让 decoder 在第 $j$ 个目标 step 产生一组读取权重 $\alpha_{j,1:S}$，按需形成自己的 context $c_j$。因此它不是「给 RNN 加一层更复杂的激活」，而是把固定摘要接口改成可寻址的软读取接口。

这篇只回答为什么需要这个接口，以及它解决和没有解决什么：

1. 固定 context 的瓶颈具体在哪里；
2. 为什么长序列、重排和复制任务会放大瓶颈；
3. 注意力如何缩短信息路径并允许每步选择不同源位置；
4. 代价为何是 $T\times S$ 的 score、权重和缓存；
5. Bahdanau attention 如何成为这一动机的第一个具体实现；
6. 为什么 attention 不自动等于对齐、解释或因果性。

![左侧固定 context 把所有源状态压成一个摘要，右侧 attention 保留源时间轴并让 decoder 每步选择不同的读取权重](/assets/rnn-lstm/svg/why-attention.1.svg)

## 基线：经典 Seq2Seq 只有一个固定 context

设源序列为

$$
x_{1:S}=(x_1,x_2,\ldots,x_S).
$$

encoder 按源时间顺序更新 hidden state：

$$
s_i
=f_{\mathrm{enc}}(s_{i-1},x_i;\theta_{\mathrm{enc}}),
\qquad
i=1,\ldots,S.
$$

最小的经典 encoder-decoder 只把末状态交给 decoder：

$$
c=s_S.
$$

decoder 以 $c$ 为条件，自回归地生成目标序列：

$$
d_j
=f_{\mathrm{dec}}(d_{j-1},u_j,c;\theta_{\mathrm{dec}}),
$$

$$
p(y_j\mid y_{<j},x_{1:S})
=\operatorname{softmax}(W_o d_j+b_o).
$$

如果 decoder 只在初始化时看到 $c$，那么所有源信息必须先进入 $s_S$，再沿 decoder 的状态链影响输出。如果把 $c$ 拼到每一步输入，接口变成

$$
d_j
=f_{\mathrm{dec}}(d_{j-1},[u_j;c];\theta_{\mathrm{dec}}),
$$

但源序列仍然只有一个可读取的摘要。重复提供同一个向量，不等于恢复源时间轴。

### 两条时间轴和一条窄接口

可以把基线画成三段：

$$
x_1\longrightarrow s_1\longrightarrow\cdots\longrightarrow s_S
\longrightarrow c
\longrightarrow d_0\longrightarrow d_1\longrightarrow\cdots\longrightarrow d_T.
$$

源时间轴在 $s_S$ 处汇聚成一个节点；目标时间轴从这个节点重新展开。无论 $T$ 是多少，源到目标的跨轴接口都只有一个 $c$。

这并不意味着固定 context 在理论上绝对不能编码整条序列。只要允许无限精度、任意宽度和理想的可逆编码，一个向量可以携带非常多的信息。实际神经网络却受到 hidden width、有限数值精度、梯度噪声、优化时间和分布外长度的约束。工程问题不是「一个向量在抽象数学上能不能编码」，而是「训练是否能可靠地把 decoder 需要的每个源事实放进同一个向量，并在正确目标步取出来」。

## 瓶颈一：多个事实被迫混在一个表示里

固定 context 是一个压缩映射：

$$
(s_1,s_2,\ldots,s_S)\longmapsto c.
$$

压缩本身未必错误；问题是 decoder 之后没有一个显式地址可以请求原来的第 $i$ 个状态。所有源位置的影响先在 encoder 末端混合，再由 decoder 从混合物中反推需要的局部证据。

### 顺序信息的最小反例

用一个极小的线性玩具模型隔离这个问题。假设每个源状态就是输入本身，固定 context 取平均：

$$
s_i=x_i,
\qquad
c=\frac12(s_1+s_2).
$$

考虑两个不同的源序列：

$$
A=(1,-1),
\qquad
B=(-1,1).
$$

两者得到同一个 context：

$$
c_A=\frac12(1-1)=0,
\qquad
c_B=\frac12(-1+1)=0.
$$

如果 decoder 初始状态、目标前缀和参数都相同，它从这个接口看到的输入完全相同，不能仅凭 $c$ 区分「先出现 1」和「先出现 -1」。这不是 RNN 的所有实现都必然做平均，而是一个可验证的结构反例：一旦 encoder 映射把两个源序列压到同一个 context，后面的 decoder 没有信息可以恢复被丢掉的区别。

注意力保留源位置，并允许不同目标 step 使用不同权重。若对序列 $A$ 先读第一个位置、再读第二个位置：

$$
\alpha^{(1)}_A=(0.9,0.1),
\qquad
\alpha^{(2)}_A=(0.1,0.9),
$$

则两个 context 分别为

$$
c^{(1)}_A=0.9(1)+0.1(-1)=0.8,
\qquad
c^{(2)}_A=0.1(1)+0.9(-1)=-0.8.
$$

对序列 $B$，同一组「先左后右」的权重得到

$$
c^{(1)}_B=0.9(-1)+0.1(1)=-0.8,
\qquad
c^{(2)}_B=0.1(-1)+0.9(1)=0.8.
$$

动态读取把「值」和「地址」同时保留下来：同一个 decoder step 可以偏向源轴左侧或右侧，源顺序不必提前被一个固定平均值抹平。

### 多个局部事实的容量竞争

翻译、摘要和信息抽取常常需要同时处理多个事实：

| 目标行为 | 需要的源证据 | 固定 context 的压力 |
| --- | --- | --- |
| 复制人名或数字 | 精确的局部 token 及其顺序 | 不能只保留语义大意 |
| 重排短语 | 多个源位置的相对关系 | decoder 要从混合表示反推地址 |
| 处理长修饰语 | 远距离的局部边界 | 早期状态要穿过更长 encoder 链 |
| 多次复用同一实体 | 同一个源位置的重复读取 | 固定摘要没有显式地址 |
| 省略无关片段 | 哪些位置当前不重要 | 所有位置先混合，选择发生得太晚 |

注意力也不保证每个问题都被解决；它只是给模型一个可以表达这些读取策略的接口。

## 瓶颈二：信息路径随源长度变长

固定 context 中，源位置 $x_i$ 对目标输出的典型路径是

$$
x_i\longrightarrow s_i\longrightarrow s_{i+1}
\longrightarrow\cdots\longrightarrow s_S
\longrightarrow c\longrightarrow d_j\longrightarrow y_j.
$$

源越早，必须穿过的 encoder recurrent transition 越多。对一个目标位置的损失 $\mathcal L_j$，固定 context 的局部路径可以写成

$$
\frac{\partial\mathcal L_j}{\partial x_i}
=
\frac{\partial\mathcal L_j}{\partial d_j}
\frac{\partial d_j}{\partial c}
\frac{\partial c}{\partial s_S}
\frac{\partial s_S}{\partial s_{S-1}}
\cdots
\frac{\partial s_{i+1}}{\partial s_i}
\frac{\partial s_i}{\partial x_i}.
$$

中间的 Jacobian 连乘越长，越容易受到梯度消失、爆炸、激活饱和和优化噪声影响。LSTM 或 GRU 能改善状态 carry，却没有把源位置直接暴露给 decoder；它们缓解的是循环状态保存，不是固定 context 的寻址缺失。

### attention 提供一条显式源读取边

如果保存所有 encoder state $h_i$，每个目标位置都可以先计算

$$
e_{j,i}=\operatorname{score}(d_{j-1},h_i),
\qquad
\alpha_{j,i}
=\operatorname{softmax}_i(e_{j,1:S}),
$$

再读取

$$
c_j=\sum_{i=1}^{S}\alpha_{j,i}h_i.
$$

于是某个源 state 到当前 context 存在显式 value 路径：

$$
h_i\longrightarrow \alpha_{j,i}h_i
\longrightarrow c_j\longrightarrow d_j\longrightarrow y_j.
$$

它不需要让 $h_i$ 先被循环传播到唯一的末状态。score 路径仍然需要训练、仍然可能饱和，但模型现在可以学习「当前 query 应该从哪个源状态读取」。

从梯度角度，若暂时把 $\alpha_{j,i}$ 当作已知常数，

$$
\frac{\partial c_j}{\partial h_i}
=\alpha_{j,i}I.
$$

真实的 encoder state 还会通过 score 影响权重，因此完整梯度包含 value 路径和 score 路径；但即使只看这条直接路径，也能说明 attention 为什么改善信用分配：当前目标 step 对相关源状态有一个显式、可调权重的入口。

## 瓶颈三：每个目标 step 需要不同的源视角

固定 context 强迫所有目标位置使用同一个摘要：

$$
c_1=c_2=\cdots=c_T=c.
$$

decoder 可以对 $c$ 做不同的非线性变换，因此输出仍可能不同；但所有目标位置必须从同一个混合输入中恢复自己需要的源证据。

注意力把这个约束改成

$$
c_j=\sum_{i=1}^{S}\alpha_{j,i}h_i,
\qquad
\alpha_{j,\cdot}\ne\alpha_{k,\cdot}
\quad\text{通常允许}.
$$

「通常允许」很重要：模型可以学到多个 step 使用近似相同的权重，也可以学到分散或近似均匀的读取。attention 提供的是表达能力，不是强制单调、强制 one-hot 或强制每个位置只访问一次。

### 重排和复制

在一个需要重排的任务中，源序列和目标序列的时间顺序不同。固定 context 可以让 decoder 记住一个全局语义，但没有显式的源地址：

$$
\text{源位置 }i
\longrightarrow
\text{混合摘要 }c
\longrightarrow
\text{目标位置 }j.
$$

注意力增加了一个可学习的二维关系：

$$
(j,i)\longrightarrow\alpha_{j,i}.
$$

这个二维矩阵可以表达「目标第 $j$ 步主要读取源第 $i$ 步」，也可以表达一个目标 token 需要多个源位置。它因此特别适合解释为什么每个目标步的 context 不必相同。

### 长距离重复读取

若目标序列后面再次需要源序列早期的某个实体，固定 context 必须让 decoder 保存该实体的可用表示，直到需要它的时间步。attention 可以在每个目标 step 重新向源时间轴请求同一位置：

$$
\alpha_{j,i}\approx1
\quad\text{和}\quad
\alpha_{k,i}\approx1
\qquad
\text{即使 }k-j\text{ 很大}.
$$

这不是免费外部记忆：encoder states 仍需存储，score 仍需计算；但它把「长期保存一个事实」转成「保留状态并在需要时再次读取」。

## 注意力究竟改变了什么

把固定 context 和 attention 放在同一个账本中：

| 维度 | fixed context | step-wise attention |
| --- | --- | --- |
| encoder 交付物 | 一个 $c$ | 全部 $h_1,\ldots,h_S$ |
| decoder 源读取 | 读取同一个摘要 | 每步重新计算 $\alpha_{j,1:S}$ |
| 源地址 | 被混合后隐式存在 | 由源轴权重显式表示 |
| 目标视角 | 所有 step 共享输入摘要 | $c_j$ 可以随 $j$ 变化 |
| 梯度入口 | 经末状态和 decoder 链 | 增加每步 value/score 路径 |
| 代价 | context 读取便宜 | 大致产生 $T\times S$ score/weight |
| 因果性 | 取决于 encoder | 仍取决于 source encoder 和 mask |

因此准确的结论是：「attention 放松了固定摘要和单一源视角的约束」，而不是「attention 让模型记住一切」或「attention 自动解决长程依赖」。

## 第一个具体实现：Bahdanau attention

Bahdanau attention 把上面的动机写成一个可训练的软检索器。对第 $j$ 个目标 step，使用 decoder query 和第 $i$ 个 encoder state 计算加性 score：

$$
e_{j,i}
=v_a^{\mathsf T}
\tanh\bigl(W_d d_{j-1}+W_h h_i+b_a\bigr).
$$

沿源位置归一化：

$$
\alpha_{j,i}
=\frac{\exp(e_{j,i})}
{\sum_{k=1}^{S}\exp(e_{j,k})},
$$

形成每步 context：

$$
c_j=\sum_{i=1}^{S}\alpha_{j,i}h_i.
$$

这三行公式正好对应动机中的三件事：

1. score：当前目标状态询问每个源位置；
2. softmax：把询问结果变成可微的软地址；
3. weighted sum：把地址指向的源状态读回 decoder。

Bahdanau 的加性 score、mask、梯度和 shape 账本见[Bahdanau 注意力](../rnn-lstm/bahdanau-attention/)。这里不重复实现细节，只强调它为何是一个有针对性的结构回应：它保存源时间轴，同时让每个目标 step 生成自己的读取分布。

## 为什么不用只把 encoder 做得更大

扩大 encoder hidden width、使用 LSTM/GRU、堆叠更多层，都可能提高固定 context 的容量。但这些办法和 attention 解决的接口问题不同：

| 方案 | 主要改善 | 没有直接解决 |
| --- | --- | --- |
| 更宽 hidden | 单个摘要可承载更多维度 | decoder 仍没有源地址 |
| 更深 encoder | 更强的非线性和上下文混合 | 源位置仍在末状态处汇聚 |
| LSTM/GRU | 状态 carry 与时间梯度 | 固定 context 的单一读取口 |
| 双向 encoder | 每个位置看到左右源上下文 | decoder 仍只拿一个摘要时的寻址 |
| hierarchical encoder | 先局部压缩，再聚合层级 | 每个目标 step 是否能直接访问局部状态 |
| attention | 每步读取不同源状态 | 需要额外计算、mask 和稳定性检查 |

「更大的向量」和「可寻址的多个向量」不是同一个设计。前者增加管道宽度，后者改变管道拓扑。

## 代价：可寻址读取不是免费内存

保留 $S$ 个 encoder state，并对 $T$ 个目标 step 计算 score，通常需要

$$
\mathcal O(TS)
$$

个标量 score 和权重。若 score 内部宽度为 $d_a$、value 宽度为 $d_h$，粗略计算量为

$$
\mathcal O(TS d_a)
\;+\;
\mathcal O(TS d_h).
$$

训练时 attention map 大致占用

$$
\mathcal O(BTS)
$$

的存储，还要保存 encoder states、decoder states 和反向中间量。固定 context 的 decoder 读取不需要 $S$ 个 score；attention 用显式 source address 换来了时间和显存。

一个具体小账本可以取 $B=2$、$T=64$、$S=128$、$d_a=32$、$d_h=256$：

| 项目 | 算式 | 数量 |
| --- | --- | ---: |
| score 内部乘加规模 | $BTSd_a$ | $524288$ |
| value 加权规模 | $BTSd_h$ | $4194304$ |
| attention map 元素 | $BTS$ | $16384$ |
| 固定 context 的每步源 score | — | 0 |

表中的「0」只表示 fixed context 没有这张逐步源位置 score 图，不表示 encoder 或 decoder 没有计算。系统设计必须用真实长度分布、峰值显存和延迟测量来判断这个代价是否可接受。

### decoder 仍然是自回归的

RNN attention 可以在同一个 query 下并行计算所有源位置的 score，但目标 step 之间仍有

$$
d_j=f(d_{j-1},u_j,c_j)
$$

的状态依赖。因此 attention 不会把 decoder 变成一次性并行生成器，也不会自动消除 teacher forcing 与 free-running 的差异。

### mask 和变长序列

批量中不同源长度需要把 padding 从每个目标 step 的源分布中排除：

$$
\widetilde e_{b,j,i}
=
\begin{cases}
e_{b,j,i},&\text{有效源位置},\\
-\infty,&\text{padding 位置}.
\end{cases}
$$

mask 轴错、mask 后不重新归一化、全 mask 行产生 NaN，都会让「可寻址读取」变成「读取了填充或非有限值」。这也是为什么 attention 的收益必须和 mask 审计一起讨论。

## attention 不等于对齐、解释或因果性

attention map 有 $T\times S$ 个权重，很容易被画成热图并称为「对齐」。它确实可以提供一个有用的读取诊断，但不自动保证：

1. 最大权重位置就是人类认为的词对齐；
2. 每个目标 token 只对应一个源 token；
3. 权重是模型唯一使用的证据；
4. teacher forcing 下的热图等于 free-running 下的热图；
5. 双向 encoder 的 attention 具有在线因果性。

权重来自 score 参数、query、value 表示和归一化约定。多个源位置的混合 value 可能足以预测输出，低熵也不等于正确解释。若要把 attention 当作证据，应同时记录 mask、熵、输入位置扰动、输出变化和运行状态。

### 双向 encoder 的边界

若 encoder 使用双向 RNN，位置 $i$ 的状态可能是

$$
h_i=[\overrightarrow h_i;\overleftarrow h_i].
$$

attention 可以在源位置轴上选择 $h_i$，但每个 $h_i$ 已经包含当前位置左右两侧的源信息。完整源序列已知时，这是很好的离线表示；严格在线时，反向分支读取未来，attention 不能把未来信息变回过去。

### target-side causal boundary

attention 通常访问的是完整 source，而不是完整 target。目标侧仍然满足

$$
p(y_{1:T}\mid x_{1:S})
=\prod_{j=1}^{T}p(y_j\mid y_{<j},x_{1:S}).
$$

如果实现让第 $j$ 步 query 读取了未来 target token，那是 target leakage，不是 attention 的正常能力。源 mask 和目标 causal 约束是两个不同的轴。

## 从 RNN attention 到更一般的 attention

为什么需要注意力的动机并不绑定某一类 score：

1. 固定摘要让所有目标 step 共享同一个源摘要；
2. 可寻址状态序列允许每个 query 选择不同源位置；
3. score 函数可以是 additive、dot product 或其他匹配函数；
4. value 聚合可以是加权和或更一般的读取；
5. mask、归一化轴和资源账本始终需要保留。

Bahdanau attention 使用 decoder state 作为 query、encoder state 作为 key/value，是 encoder-decoder 中的 cross-attention。之后的 self-attention 会让同一序列内部的位置互相读取；更一般的 attention-as-retrieval 会把 query-key-value 看成可组合的读取抽象。它们可以使用不同 score 和并行化方式，但「从固定摘要走向可寻址读取」仍是共同动机。

## 失效模式：结构动机正确，接口仍可能错

### 把重复 context 当成动态读取

如果只是把同一个 $c$ 拼到每个 decoder 输入，$c_1,\ldots,c_T$ 仍然相同。要证明使用了 attention，必须看到随 $j$ 变化的源轴权重或 context，而不是只看到一个更宽的输入。

### 只增加宽度，不保留源状态

把 $s_S$ 投影成更宽的向量可能改善容量，但若 decoder 没有 $s_1,\ldots,s_{S-1}$ 的接口，它仍无法直接请求某个源位置。报告 width 增加时要同时报告信息路由是否变化。

### attention 权重轴颠倒

对每个目标位置，源轴权重应满足

$$
\sum_{i=1}^{S}\alpha_{j,i}=1.
$$

若误沿目标轴归一化，热图可能仍然有颜色和结构，但它不再表示「当前目标 step 如何分配源读取」。

### key/value 错位

score 对第 $i$ 个 key，value 却从第 $i+1$ 个位置取，最终 context shape 仍可能正确。需要用源位置 one-hot、单点扰动或显式索引追踪检查两条轴。

### 只在 teacher-forcing 下看热图

真实推理时 query 由模型自己的前缀产生。若前缀错误，decoder 状态、score 和 context 都可能漂移；只保存 teacher-forced attention map 不能证明部署时的读取链。

### 忽略资源边界

在短句上 attention 的 $T\times S$ 成本很小，在长输入、长输出或大 batch 上可能成为主导。至少记录 source length、target length、attention map、峰值显存和每 token 延迟。

### 过度解读最大权重

最大 $\alpha_{j,i}$ 只是当前 score 的最大项。它不是硬地址、不是充分解释，也不等价于因果贡献。把它写成「模型一定看了这个词」会超出计算图证据。

## 最小审计协议

面对「为什么需要 attention」的架构说明或实现，可以按以下顺序核对：

1. **画基线。** 标出 $x_{1:S}$、$s_{1:S}$、固定 $c$、$d_{1:T}$ 和唯一跨轴接口。
2. **找信息瓶颈。** 写出源位置到目标输出的路径，记录需要穿过多少个 encoder transition。
3. **做不可区分反例。** 用两个不同源序列和一个有碰撞的摘要映射，验证固定 context 丢掉的区别无法被 decoder 恢复。
4. **画动态读取。** 标出 $E,A,C$ 的 $T\times S$ 结构，以及每个 $j$ 可以有不同的 $\alpha_{j,\cdot}$。
5. **固定 query convention。** 记录 query 是 $d_{j-1}$ 还是 $d_j$，context 在 decoder 更新前还是更新后进入。
6. **核对源轴。** 断言 softmax 沿源位置归一化，padding 先 mask，valid weight 和为一。
7. **核对 key/value 对齐。** 用单点源扰动确认 score 位置和 value 位置没有错位。
8. **检查运行态。** 区分 teacher forcing、free-running、offline full source 和 strict online。
9. **检查梯度。** 同时覆盖 value 路径、score 路径、encoder 时间链和 decoder 时间链。
10. **记录资源。** 报告 $B,T,S,d_a,d_h$、attention map、激活缓存和真实长度下的延迟。

最小日志可以是

| 字段 | 示例 | 目的 |
| --- | --- | --- |
| baseline | $c=s_S$ | 固定摘要对照 |
| source states | $H\in\mathbb R^{B\times S\times d_h}$ | 保留源时间轴 |
| attention map | $A\in\mathbb R^{B\times T\times S}$ | 固定两个时间轴 |
| normalization | $\sum_i\alpha_{j,i}=1$ | 固定 softmax 轴 |
| query | $q_j=d_{j-1}$ | 固定时序 |
| mask | valid lengths $(S_1,\ldots,S_B)$ | 排除 padding |
| running mode | free-running / offline | 固定可见性 |
| resource | peak memory, latency/token | 接受 $T\times S$ 代价 |

## 结语

需要注意力，不是因为一个固定向量在抽象上绝对不能表示一条序列，而是因为经典 encoder-decoder 把「保存源信息」和「按目标位置读取源信息」压缩成了同一个末状态接口。源序列越长、目标行为越依赖局部事实、顺序越需要重排或复制，这个接口越难训练、越难传递梯度、越难让 decoder 找回正确地址。

attention 的回应是保留一组可寻址的 encoder states，让每个目标 query 产生自己的软读取分布：

$$
\text{固定摘要}
\quad\longrightarrow\quad
\text{源状态序列}
\quad\longrightarrow\quad
\text{每步可微读取}.
$$

它因此改变了信息路由，但也带来 $T\times S$ 的计算、mask、数值稳定性和解释边界。理解这笔交换，才能把 Bahdanau attention 看成一个具体而可审计的结构选择，而不是把所有性能提升都归因于一张热图。

## 相关词条

[序列到序列](../rnn-lstm/seq2seq/)

[Bahdanau 注意力](../rnn-lstm/bahdanau-attention/)

[教师强制](../rnn-lstm/teacher-forcing/)

[双向循环网络](../rnn-lstm/bidirectional-rnn/)

[序列建模](../rnn-lstm/sequence-modeling/)

[循环神经网络](../rnn-lstm/rnn/)

[RNN 时间展开](../rnn-lstm/rnn-unrolling/)

[时间反向传播](../rnn-lstm/bptt/)

[长短期记忆网络](../rnn-lstm/lstm/)

[门控循环单元](../rnn-lstm/gru/)

[注意力作为检索](../attention/attention-as-retrieval/)

[自注意力](../attention/self-attention/)

[交叉注意力](../attention/cross-attention/)

[注意力复杂度](../attention/attention-complexity/)
