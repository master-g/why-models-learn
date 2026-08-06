---
title: "Bahdanau 注意力：让解码器按需读取编码器状态"
tags: ["why-models-learn"]
---

Bahdanau 注意力（Bahdanau attention）是经典 RNN encoder-decoder 中的一种加性注意力（additive attention）。它不再把整条源序列压成一个固定 context，而是在 decoder 的每个目标时间步，用当前 decoder 状态作为 query，对所有 encoder hidden state 计算对齐分数，再用 softmax 得到权重，最后加权读取一份当步 context。

这篇词条固定一个时间约定：decoder 在生成第 $j$ 个目标 token 前，已经有状态 $d_{j-1}$ 和输入 $u_j$。先用 $d_{j-1}$ 与每个源状态 $h_i$ 计算注意力，再得到 $c_j$，然后更新为 $d_j$。不同实现可能把 query 写成 $d_j$，那是另一种时序约定；公式、缓存和测试必须从头到尾使用同一个约定。

它真正改变的不是「多加了一层 tanh」，而是信息路由合同：

1. encoder 保留源时间轴上的多个状态，而不是只交付一个末状态；
2. decoder 每一步都可以选择不同的源位置；
3. score、softmax、mask 和加权求和组成一条可微的读取路径；
4. attention weight 是归一化的路由系数，不自动等于离散词对齐或因果解释；
5. 源序列越长、目标序列越长，注意力矩阵带来的时间和显存成本越明显。

![Bahdanau 注意力用 decoder query 对每个 encoder state 打分，经 softmax 得到权重后形成当步 context](/assets/rnn-lstm/svg/bahdanau-attention.1.svg)

## 从固定 context 到每步 context

设源序列长度为 $S$，目标序列长度为 $T$。经典 Seq2Seq 可以只把 encoder 的最后状态传给 decoder：

$$
c=h_S.
$$

无论目标序列要生成多少步，所有源信息都必须经过同一个向量 $c$。即使把 $c$ 拼到每个 decoder 输入，信息瓶颈仍然存在：decoder 没有一个显式的接口去请求「源序列的第 $i$ 个位置」。

Bahdanau 注意力保留整个 encoder 状态序列：

$$
H=(h_1,h_2,\ldots,h_S),
\qquad
h_i\in\mathbb R^{d_h}.
$$

第 $j$ 个 decoder step 产生一组对齐分数 $e_{j,1},\ldots,e_{j,S}$，归一化为权重 $\alpha_{j,i}$，再形成当步 context：

$$
c_j
=\sum_{i=1}^{S}\alpha_{j,i}h_i,
\qquad
\sum_{i=1}^{S}\alpha_{j,i}=1.
$$

于是不同目标位置可以读取不同的源摘要：

$$
c_1\ne c_2\ne\cdots\ne c_T
$$

并不是说这些向量一定不同，而是说模型不再被迫复用同一个固定摘要。固定 context 与每步 context 的接口差异可以列成一张表：

| 结构 | decoder 可读取的源证据 | 源时间轴 | 主要瓶颈 |
| --- | --- | --- | --- |
| fixed context | 一个 $c$ | 通常只暴露末状态或 bridge 输出 | 长序列信息集中到一个向量 |
| Bahdanau attention | 每步一个 $c_j$ | 保留 $h_1,\ldots,h_S$ | 每个目标步都要计算 $S$ 个 score |
| masked attention | 每步一个被 mask 约束的 $c_j$ | 只读取有效源位置 | mask 轴、边界和全 mask 情况 |

这就是 attention 作为结构性修复的含义：它把「记住所有源信息」变成了「保存源状态，并在需要时可微地检索」。

## 一次读取的完整计算链

### 记号：query、key 和 value

在第 $j$ 个 decoder step，定义

$$
q_j=d_{j-1}\in\mathbb R^{d_d}.
$$

对每个源位置 $i$，encoder hidden state $h_i$ 同时充当 key 和 value：

$$
k_i=h_i\in\mathbb R^{d_h},
\qquad
v_i=h_i\in\mathbb R^{d_h}.
$$

key 参与「这个位置是否值得读取」的打分，value 参与「真正读出什么内容」的加权和。在经典 Bahdanau attention 里二者来自同一个 encoder state；把它们分开命名，是为了看清 score 路径和 value 路径，而不是暗示必须使用两套独立张量。

### 加性对齐分数

Bahdanau 的 score 先把 query 和 key 投影到一个共同的 attention 空间，再经过非线性和向量投影：

$$
e_{j,i}
=v_a^{\mathsf T}
\tanh\bigl(W_d d_{j-1}+W_h h_i+b_a\bigr).
$$

其中 attention 内部宽度记为 $d_a$，参数形状是

$$
W_d\in\mathbb R^{d_a\times d_d},
\qquad
W_h\in\mathbb R^{d_a\times d_h},
\qquad
b_a,v_a\in\mathbb R^{d_a}.
$$

括号里的向量在 attention 空间中相加：

$$
z_{j,i}
=W_d d_{j-1}+W_h h_i+b_a
\in\mathbb R^{d_a},
\qquad
e_{j,i}=v_a^{\mathsf T}\tanh(z_{j,i})\in\mathbb R.
$$

「加性」指的是投影后的 query 与 key 在进入 tanh 前相加，不是说最终 context 做了普通的向量相加。它允许 $d_d$ 与 $d_h$ 不相等，因为两者先分别由 $W_d$ 和 $W_h$ 投到 $d_a$。

### softmax 变成读取权重

如果所有源位置都有效，第 $j$ 步的权重为

$$
\alpha_{j,i}
=\frac{\exp(e_{j,i})}
{\sum_{k=1}^{S}\exp(e_{j,k})},
\qquad
i=1,\ldots,S.
$$

因此

$$
\alpha_{j,i}\ge 0,
\qquad
\sum_{i=1}^{S}\alpha_{j,i}=1.
$$

它们构成一条位于概率单纯形上的软路由。分数差距大时，权重会集中；分数接近时，权重会分散。softmax 不会把分数变成硬的 one-hot 索引，所以梯度可以同时通过多个源位置流回 encoder。

### 加权读取和 decoder 更新

把权重乘到 value 并求和：

$$
c_j
=\sum_{i=1}^{S}\alpha_{j,i}h_i
\in\mathbb R^{d_h}.
$$

在本文采用的时序约定中，decoder 用上一状态和当前输入、当步 context 更新：

$$
d_j
=f_{\mathrm{dec}}\bigl(d_{j-1},[u_j;c_j];\theta_{\mathrm{dec}}\bigr).
$$

随后输出目标 token 的条件分布，例如

$$
p(y_j\mid y_{<j},x_{1:S})
=\operatorname{softmax}\bigl(W_o[d_j;c_j]+b_o\bigr).
$$

有的实现把 context 只送入状态更新，有的实现把 $[d_j;c_j]$ 送入输出头；这不会改变 attention 的四步主链：

$$
\text{query}
\longrightarrow
\text{score}
\longrightarrow
\text{softmax weight}
\longrightarrow
\text{weighted value}.
$$

真正需要核对的是 query 属于哪个 decoder 时间点、context 进入哪条边，以及 loss 使用哪个输出。

## 一个可手算的加性注意力例子

先做一个一维的最小例子。为了让 score 子网络和 value 的作用可以同时看到，令 query 和三个 encoder state 都是标量：

$$
q=0.5,
\qquad
h_1=0.5,\quad h_2=0,\quad h_3=-1.
$$

取 $W_d=W_h=v_a=1$、$b_a=0$。于是

$$
e_i=\tanh(q+h_i),
$$

得到

$$
(e_1,e_2,e_3)
=(0.761594,\ 0.462117,\ -0.462117).
$$

逐项取指数并归一化：

| 位置 $i$ | $h_i$ | $e_i$ | $\exp(e_i)$ 的归一化权重 $\alpha_i$ |
| ---: | ---: | ---: | ---: |
| 1 | $0.5$ | $0.761594$ | $0.491318$ |
| 2 | $0$ | $0.462117$ | $0.364168$ |
| 3 | $-1$ | $-0.462117$ | $0.144515$ |

权重确实加起来为 $1$，context 是

$$
c
=0.491318(0.5)+0.364168(0)+0.144515(-1)
\approx 0.101144.
$$

这个结果不是选中了某个单独的 $h_i$，而是在 query 的条件下对三个 value 做了软组合。若把 query 改掉，三项 score、权重和 context 会一起改变；这正是「每个 decoder step 按需读取」的最小数学样本。

### 把 score 和 value 路径拆开核对

工程审计时，可以先假设 score 子网络已经输出

$$
e=(1,0,-1),
\qquad
h=(1,3,5).
$$

这一步暂时不追问 $e$ 是怎么得到的，只核对 softmax 与加权读取：

$$
\alpha
=(0.665241,\ 0.244728,\ 0.090031),
\qquad
c
=\sum_i\alpha_i h_i
\approx 1.849579.
$$

这个拆分很有用：如果权重已经正确但 context 错了，问题在 value 轴、广播或归约；如果 context 正确但权重不符合 softmax，问题在 score、归一化轴或 mask。

## 形状账本与参数账本

### 单样本和 batch 形状

对长度分别为 $S$ 和 $T$ 的 batch，使用 batch-first 约定时可以记录

$$
H\in\mathbb R^{B\times S\times d_h},
\qquad
D\in\mathbb R^{B\times T\times d_d}.
$$

对每个目标位置和源位置都计算一个标量 score：

$$
E\in\mathbb R^{B\times T\times S},
\qquad
A\in\mathbb R^{B\times T\times S}.
$$

value 加权后得到

$$
C\in\mathbb R^{B\times T\times d_h}.
$$

其中 $A$ 的最后一维必须是源时间轴，因为对每个固定的 $j$，softmax 应沿 $i$ 归一化：

$$
\sum_{i=1}^{S}A_{b,j,i}=1.
$$

如果把 tensor 排成 $B\times S\times T$，同一数学定义仍然成立，但归一化轴和矩阵乘法的转置必须同步改变。只检查输出 shape 往往抓不住轴错位，因为错误的广播也可能返回一个看起来合理的 shape。

### attention 参数

score 子网络的参数量为

$$
\begin{aligned}
N_{\mathrm{attn}}
&=d_a d_d+d_a d_h+d_a+d_a\\
&=d_a(d_d+d_h+2).
\end{aligned}
$$

四项分别来自 $W_d$、$W_h$、$b_a$ 和 $v_a$。例如 $d_d=4$、$d_h=6$、$d_a=3$ 时，

| 参数 | 形状 | 参数量 |
| --- | --- | ---: |
| $W_d$ | $3\times4$ | 12 |
| $W_h$ | $3\times6$ | 18 |
| $b_a$ | $3$ | 3 |
| $v_a$ | $3$ | 3 |
| 合计 | — | 36 |

这 36 个参数还不包括 encoder、decoder、embedding 和输出头。加性注意力的参数账本因此不能只写「attention 很小」；它是否值得，要和减少 fixed-context 瓶颈的收益、额外的 $T\times S$ 计算一起看。

### 每步的 shape 变换

固定一个 batch 项和目标位置 $j$，可以把一次读取写成下面的账本：

| 中间量 | 形状 | 作用 |
| --- | ---: | --- |
| $d_{j-1}$ | $d_d$ | decoder query |
| $h_i$ | $d_h$ | 第 $i$ 个 key/value |
| $W_d d_{j-1}$ | $d_a$ | query 投影 |
| $W_h h_i$ | $d_a$ | key 投影 |
| $z_{j,i}$ | $d_a$ | 相加并加 bias |
| $e_{j,i}$ | 标量 | 一个源位置的 score |
| $\alpha_{j,i}$ | 标量 | 沿源轴归一化后的权重 |
| $c_j$ | $d_h$ | 加权 value 的 context |

这个表也解释了为什么 $d_d$ 和 $d_h$ 可以不同，而 $W_d d_{j-1}$ 与 $W_h h_i$ 必须同宽。

## mask：先约束 score，再做 softmax

真实 batch 往往把不同长度的源序列补成同一个 $S_{\max}$。若某个样本的有效长度只有 $S_b$，padding 位置不能参与注意力。

令 $m_{b,i}=1$ 表示有效、$0$ 表示 padding。概念上应先修改 score：

$$
\widetilde e_{b,j,i}
=
\begin{cases}
e_{b,j,i},&m_{b,i}=1,\\
-\infty,&m_{b,i}=0,
\end{cases}
$$

再沿源轴计算

$$
\alpha_{b,j,i}
=\operatorname{softmax}_{i}
\bigl(\widetilde e_{b,j,1:S_{\max}}\bigr)_i.
$$

这样 padding 位置的权重为零，剩下的有效权重仍然加起来为 $1$。把 mask 乘在 softmax 之后但不重新归一化，会让有效权重和小于 $1$；把 padding 先放进 value 再希望 decoder 自己学会忽略，则把边界约束交给了一个不可靠的捷径。

用上面的核对分数 $e=(1,0,-1)$，假设第二个位置是 padding。mask 前后可以对照：

| 位置 | 未 mask 的 score | 是否有效 | mask 后参与 softmax 的 score | 最终权重 |
| ---: | ---: | :---: | ---: | ---: |
| 1 | 1 | 是 | 1 | $0.880797$ |
| 2 | 0 | 否 | $-\infty$ | $0$ |
| 3 | $-1$ | 是 | $-1$ | $0.119203$ |

若 $h=(1,3,5)$，masked context 是

$$
c_{\mathrm{mask}}
=0.880797(1)+0(3)+0.119203(5)
\approx 1.476812.
$$

这里有三个边界必须单独测试：

1. 有效长度为 $S_{\max}$ 时，mask 不应改变结果；
2. 有效长度为 $1$ 时，唯一有效位置的权重应为 $1$；
3. 有效长度为 $0$ 或整行被 mask 时，不能直接把全是 $-\infty$ 的向量送进普通 softmax，否则会出现 $0/0$ 和 NaN，必须在数据协议中拒绝、跳过或提供显式空 context。

## 梯度：一条读取路径，两个回传分支

attention 是可微的，梯度不需要先选择一个离散源位置。对 softmax 有

$$
\frac{\partial\alpha_i}{\partial e_k}
=\alpha_i(\delta_{ik}-\alpha_k).
$$

若 context 与 value 都是一维，记

$$
c=\sum_i\alpha_i h_i,
\qquad
g=\frac{\partial\mathcal L}{\partial c},
$$

则 energy 的梯度可以化为

$$
\frac{\partial\mathcal L}{\partial e_k}
=g\,\alpha_k(h_k-c).
$$

这条式子很有解释力：比当前加权平均值更大的 value 会把对应 score 往上推，比平均值更小的 value 会把对应 score 往下推；所有 energy 梯度之和为零，因为给所有 score 加同一个常数不会改变 softmax。

用 $e=(1,0,-1)$、$h=(1,3,5)$、上游梯度 $g=1$ 的例子，$c\approx1.849579$，得到

$$
\frac{\partial\mathcal L}{\partial e}
\approx(-0.565175,\ 0.281541,\ 0.283634),
\qquad
\sum_k\frac{\partial\mathcal L}{\partial e_k}\approx0.
$$

如果损失是

$$
\mathcal L=\frac12(c-2)^2,
$$

则 $g=c-2\approx-0.150421$，因此

$$
\frac{\partial\mathcal L}{\partial e}
\approx(0.085014,\ -0.042350,\ -0.042664).
$$

对 encoder state $h_k$，至少有两条回传路径：

1. value 路径：$h_k$ 直接出现在 $\sum_i\alpha_i h_i$ 中；
2. score 路径：$h_k$ 还通过 $W_hh_k$ 改变 $e_k$，再改变所有归一化权重。

把 score 路径错误地 detach，会让 attention 仍然能输出数值，但 encoder 学不到「什么样的 key 应该被读取」；把 value 路径错误地 detach，则会截断 context 内容对 encoder 的训练信号。对每个 decoder step，attention 参数梯度还要沿 $j=1,\ldots,T$ 累加：

$$
\frac{\partial\mathcal L}{\partial\theta_a}
=\sum_{j=1}^{T}
\frac{\partial\mathcal L}{\partial c_j}
\frac{\partial c_j}{\partial\theta_a}.
$$

这也是它和 BPTT 的连接：decoder 的状态链、attention 的源位置分支以及共享 score 参数共同构成展开图，不能只检查某一个 step 的局部梯度。

## 为什么是 additive，而不是直接点积

另一类常见 score 是点积：

$$
e_{j,i}=q_j^{\mathsf T}k_i.
$$

直接点积要求 query 和 key 已经位于同一个宽度；加性 attention 通过 $W_d$ 和 $W_h$ 分别投影，允许 decoder 与 encoder 使用不同的 hidden width。

| score | 典型形式 | 宽度要求 | 主要特征 |
| --- | --- | --- | --- |
| dot | $q_j^{\mathsf T}k_i$ | $q_j$ 与 $k_i$ 同宽 | 参数少、矩阵乘法直接 |
| projected dot | $(W_q q_j)^{\mathsf T}(W_k k_i)$ | 投影后同宽 | 先对齐表示空间 |
| additive | $v_a^{\mathsf T}\tanh(W_dq_j+W_hk_i+b_a)$ | 原始宽度可以不同 | 学习 query-key 的非线性匹配 |

不能只用参数量判断优劣。点积的尺度会随 hidden width 改变，通常需要缩放；加性 score 有额外投影和 tanh，也可能在大幅值处饱和。真正的选择要结合 hidden width、硬件上的矩阵乘法形态、序列长度和训练稳定性。

## attention weight 是不是对齐或解释

$\alpha_{j,i}$ 可以被画成一张 $T\times S$ 的热图，因此很容易把它称为「第 $j$ 个目标 token 对齐到第 $i$ 个源 token」。这个说法在某些翻译样例中有帮助，但它不是公式保证的语义：

1. 一个目标 token 可能需要多个源位置，权重会变宽；
2. 多个目标 token 可能复用同一源位置；
3. 重复 token、子词切分和同义信息会使对齐不唯一；
4. score 的参数化、value 的基底和 softmax 温度会改变权重形状；
5. decoder 可以从混合后的 value 中得到正确答案，即使最大权重位置不是人类认为的词；
6. teacher forcing 下的 query 使用真实前缀，free-running 下的 query 使用模型前缀，两张 attention 图不一定相同。

因此更稳妥的表述是：「这是模型在当前 query 下对源状态的可微读取分布」。如果要把它作为诊断证据，至少同时记录：

$$
\mathcal H_j
=-\sum_{i=1}^{S}\alpha_{j,i}\log\alpha_{j,i},
$$

以及最大权重、有效位置数量、mask 后权重和、输入位置扰动前后的输出变化。低熵不等于正确对齐，高熵也不等于模型没有使用源信息。

## 与 teacher forcing、双向 encoder 的边界

### attention 不会消除 decoder 的自回归约束

attention 只解决「从源序列哪里读」的问题，不改变目标序列的条件分解：

$$
p(y_{1:T}\mid x_{1:S})
=\prod_{j=1}^{T}
p(y_j\mid y_{<j},x_{1:S}).
$$

训练时 teacher forcing 可以把真实的 $y_{j-1}$ 送给 decoder；推理时只能把自己已经生成的 token 送回去。由于 query 是 decoder 状态，前缀差异会通过状态链改变 attention score 和 context。只在 teacher-forced batch 中检查 attention map，不能证明 free-running 的读取链也正确。

### 双向 encoder 会改变 key 的可见性

如果 encoder 是双向 RNN，可以令

$$
h_i=
\bigl[\overrightarrow h_i;\overleftarrow h_i\bigr]
\in\mathbb R^{2d_h}.
$$

Bahdanau score 仍然成立，只需把 $d_h$ 换成合并后的 key width。它的含义是：每个 key 已经包含源位置左右两侧的信息，decoder 再在源时间轴上选择位置。

这适合完整源序列已知的离线翻译、摘要和标注。如果系统要求严格在线，双向 encoder 已经读取了未来源 token；attention 再精确也不能恢复因果性。固定 context、双向 encoder、attention 和 decoder bridge 是四个不同的结构问题，不能用「有了 attention」一笔带过。

## 计算量、显存和可扩展性

设 score 的内部宽度为 $d_a$，encoder hidden 宽度为 $d_h$。训练时对所有 $T\times S$ 对计算 score 和 value 加权，粗略工作量为

$$
\mathcal O\bigl(BTSd_a\bigr)
\;+\;
\mathcal O\bigl(BTSd_h\bigr).
$$

attention map 本身占用

$$
\mathcal O(BTS)
$$

的存储；若还保存 score、softmax 中间量和 decoder 激活，反向传播的实际峰值会更高。

与 fixed context 对照：

| 资源 | fixed context | Bahdanau attention |
| --- | --- | --- |
| encoder 输出给 decoder | 一个 $d_h$ 向量 | $S$ 个 $d_h$ 向量 |
| 每个目标步读取源序列 | 不需要 $S$ 次 score | 需要 $S$ 个 score |
| 源位置选择 | 没有显式权重 | 有 $S$ 个 $\alpha_{j,i}$ |
| 训练时注意力矩阵 | 不产生 | 大致 $T\times S$ |
| decoder 时间串行性 | 仍然存在 | 仍然存在，attention 只在每步内并行源位置 |

在推理时可以一次只处理一个 query，避免同时保留完整 $T\times S$ 的 attention map；但 encoder states 仍要留在内存中，且每生成一个 token 都要重新对源位置打分。序列越长，所谓「可按需读取」越需要付出明确的计算代价。

## 变体和接口选择

### query 使用 $d_{j-1}$ 还是 $d_j$

本文的 convention 是先用 $d_{j-1}$ 计算 $c_j$，再更新 $d_j$。另一种实现可能先用输入更新出 $d_j$，再以 $d_j$ 作为 query。两者都能工作，但必须同步：

| 项目 | 本文 convention | 另一种 convention |
| --- | --- | --- |
| query | $d_{j-1}$ | $d_j$ |
| context 计算时机 | decoder 状态更新前 | decoder 状态更新后 |
| 初始 query | $d_0$ | 由第一个输入得到的 $d_1$ |
| 审计重点 | $u_j$ 是否先影响 context | $u_j$ 如何先进入 query |

最容易出错的是复制公式时只改了符号，没有改缓存、teacher-forcing 输入和第一步边界。

### context 如何进入 decoder

常见接口有三种：

1. 把 $[u_j;c_j]$ 作为 recurrent cell 的输入；
2. 先用 $u_j$ 更新状态，再把 $c_j$ 拼到输出头；
3. 两处都使用，但需要明确是否重复提供同一源证据。

它们会改变参数量、梯度路径和输出行为；attention score 本身不决定其中哪一种是正确实现。

### multi-layer decoder

多层 decoder 通常用顶层或指定层的状态作为 query：

$$
q_j=d_{j-1}^{(L)}
$$

再把 $c_j$ 送回顶层 cell 或输出头。若用底层状态做 query、顶层状态接 context，必须把这个接口写进 shape ledger；否则单看 $d_d$ 很难发现层间接错。

## 失效模式：权重看起来正常，读取链仍可能错误

### softmax 轴错了

对每个目标位置，权重应沿源轴归一化：

$$
\sum_{i=1}^{S}\alpha_{j,i}=1.
$$

若误沿目标轴归一化，某个源位置在不同目标步上的权重会加起来为 $1$，但每个 decoder step 看到的总权重不再是 $1$。这是最值得在单元测试中明确断言的轴错误。

### mask 放错位置

把 mask 乘到 softmax 后却不重新归一化，会缩小 context；把 mask 只施加到 loss，不会阻止 padding 参与 score。需要同时检查 masked weight 为零、有效 weight 和为一、context 不含 padding 污染。

### query off-by-one

用 $d_j$ 替换 $d_{j-1}$，但仍按本文的 decoder 更新顺序实现，会产生一条未声明的额外信息路径。第一步尤其容易暴露：$d_0$、BOS、$u_1$ 和 $c_1$ 的先后必须逐项画出。

### key/value 时间轴错位

反向 encoder、packed sequence 或 cache 重排后，score 可能对位置 $i$ 打分，却从另一个位置取 value。最终 shape 仍可能是 $B\times T\times d_h$；应该用 one-hot 源扰动或显式索引日志检查 score 与 value 的位置是否一致。

### 数值稳定性

直接计算 $\exp(e_i)$ 可能在大 score 时溢出。常用稳定写法是

$$
\operatorname{softmax}(e)_i
=\frac{\exp(e_i-\max_k e_k)}
{\sum_\ell\exp(e_\ell-\max_k e_k)}.
$$

减去最大值不改变权重，但能把最大指数控制在 $1$。同时还要检查 tanh 饱和、非有限 score、全 mask 行和空序列。

### 把 attention 图当成解释

最大权重位置只是当前归一化 score 的最大项，不是模型唯一使用的证据。对齐图需要配合输入扰动、输出变化、熵、teacher-forcing/free-running 条件和 mask 记录。

### 只测单步，不测展开图

单个 query 的 score 和 context 可能正确，但多个目标步共享参数时仍可能有缓存复用、状态携带、loss mask 或梯度累加错误。至少要同时做一条两步或三步展开的数值例子。

### 资源账本缺失

只报告「attention 提升了长句效果」而不报告源长、目标长、$T\times S$、峰值显存和推理延迟，会掩盖模型在真实长度分布上的代价。

## 最小审计协议

实现或阅读一段 Bahdanau attention，可以按以下顺序留下证据：

1. **冻结时间 convention。** 写明 query 是 $d_{j-1}$ 还是 $d_j$，并画出 BOS、$u_j$、$c_j$、$d_j$ 的先后。
2. **冻结轴。** 记录 $H$、$D$、$E$、$A$、$C$ 的 shape，断言 softmax 沿源轴归一化。
3. **复算 score。** 用一维例子检查 $W_dq+W_hh_i+b_a$、tanh 和 $v_a^{\mathsf T}$ 的顺序。
4. **复算 softmax。** 把一组已知 energy 直接送入手算或标准库，检查非负、和为一以及稳定实现的一致性。
5. **复算 value。** 检查 context 是否从同一源位置取出与 score 对齐的 value。
6. **加入 padding。** 覆盖全长、长度一和非法空长度；确认 mask 在 softmax 前生效。
7. **检查梯度分支。** 同时保留 score 路径和值路径，核对 softmax Jacobian 或有限差分。
8. **检查展开链。** 用至少两个 decoder step，覆盖 teacher forcing、free-running、状态携带和 loss mask。
9. **检查可见性。** 若使用双向 encoder，写明是离线完整源序列还是有 lookahead 的在线协议。
10. **记录资源。** 报告 $B,T,S,d_a,d_h$、attention map、激活保存量和一条真实长度下的延迟。

一个最小日志可以是

| 字段 | 示例 | 目的 |
| --- | --- | --- |
| query convention | $q_j=d_{j-1}$ | 固定时序 |
| source shape | $B\times S\times d_h=2\times5\times6$ | 固定 key/value 轴 |
| target shape | $B\times T\times d_d=2\times4\times4$ | 固定 query 轴 |
| score shape | $B\times T\times S=2\times4\times5$ | 固定 attention map |
| mask | valid lengths $(5,3)$ | 排除 padding |
| merge | $c_j=\sum_i\alpha_{j,i}h_i$ | 固定 value 读取 |
| decoder input | $[u_j;c_j]$ | 固定下游接口 |
| 可见性 | offline full source | 防止把双向结果当 causal |

## 结语

Bahdanau 注意力把 encoder 的多个 hidden state 变成一个可以被 decoder 逐步调用的可微读取接口。它的核心公式只有三层：

$$
\text{加性 score}
\longrightarrow
\text{沿源轴的 softmax}
\longrightarrow
\text{value 加权求和}.
$$

但可靠实现必须把这三层放回完整系统：query 属于哪个 decoder 时间点，mask 是否在 softmax 前生效，score 与 value 是否仍然按源位置对齐，梯度是否同时经过两条分支，teacher-forcing 评估是否代表真实生成，以及 $T\times S$ 的资源账本是否被接受。

理解这些边界后，attention 不再是「模型自动学会对齐」的黑箱，而是一张明确的可微路由图：decoder 提出 query，score 给源位置排序，softmax 形成软分配，context 把分配后的证据交给下一步状态。后续讨论更一般的 attention 时，仍应回到这张账本。

## 相关词条

[序列到序列](../rnn-lstm/seq2seq/)

[教师强制](../rnn-lstm/teacher-forcing/)

[双向循环网络](../rnn-lstm/bidirectional-rnn/)

[序列建模](../rnn-lstm/sequence-modeling/)

[循环神经网络](../rnn-lstm/rnn/)

[RNN 时间展开](../rnn-lstm/rnn-unrolling/)

[时间反向传播](../rnn-lstm/bptt/)

[长短期记忆网络](../rnn-lstm/lstm/)

[门控循环单元](../rnn-lstm/gru/)

[为什么需要注意力](../rnn-lstm/why-attention/)

[注意力作为检索](../attention/attention-as-retrieval/)

[交叉注意力](../attention/cross-attention/)

[注意力矩阵](../attention/attention-matrix/)
