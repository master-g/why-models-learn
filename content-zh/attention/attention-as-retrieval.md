---
title: "Attention 作为检索：query 如何从 key-value 记忆中软读取"
tags: ["why-models-learn"]
---

Attention 可以先抽象成一次可学习的 key-value 检索。query 提出当前读取条件，keys 提供可匹配的地址，values 保存被读取的内容。模型先计算 query 与每个 key 的 score，再沿候选位置归一化为权重，最后对 values 做加权求和：

$$
\mathbf c
=
\sum_{i=1}^{S}\alpha_i\mathbf v_i.
$$

这里 $\alpha_i$ 不是凭空出现的解释标签，而是由 query、key、score 函数、mask 和 softmax 共同产生的路由系数。Attention 的“看向哪里”与“读出什么”由 key 和 value 两条路径分别承载。

把 attention 看成检索接口，可以统一理解 encoder-decoder attention、self-attention、memory read 和部分稀疏 attention。它也明确了边界：高权重说明一次前向中的读取系数较大，不能单独证明某个输入是预测的因果原因。

![query 与 key 计算匹配分数，softmax 产生权重，再从 value memory 软读取出 context](/assets/attention/svg/attention-as-retrieval.1.svg)

## 把 Attention 写成 key-value 读取

### 三个角色的接口不同

设有 $S$ 个候选记忆槽位：

$$
\mathbf q\in\mathbb R^{d_k},
\qquad
K=
\begin{bmatrix}
\mathbf k_1^{\mathsf T}\\
\vdots\\
\mathbf k_S^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{S\times d_k},
\qquad
V=
\begin{bmatrix}
\mathbf v_1^{\mathsf T}\\
\vdots\\
\mathbf v_S^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{S\times d_v}.
$$

query 是一次读取请求，key 是用于匹配的地址，value 是被返回的内容：

| 对象 | 形状 | 作用 | 是否直接输出 |
| --- | --- | --- | --- |
| query $\mathbf q$ | $d_k$ | 提出当前读取条件 | 否 |
| key $\mathbf k_i$ | $d_k$ | 与 query 计算匹配分数 | 否 |
| value $\mathbf v_i$ | $d_v$ | 被权重加总的内容 | 是 |
| context $\mathbf c$ | $d_v$ | 一次 soft read 的结果 | 是 |

key 和 value 可以来自同一序列，也可以来自两个不同模块。self-attention 让同一组输入分别投影出 query、key、value；encoder-decoder attention 通常让 query 来自 decoder，让 key 和 value 来自 encoder。

### score 函数把 query 映射到地址分数

一般写成

$$
s_i
=
\operatorname{score}(\mathbf q,\mathbf k_i).
$$

最简单的点积 score 是

$$
s_i
=
\mathbf q\mathbf k_i^{\mathsf T}.
$$

也可以使用可学习的双线性形式：

$$
s_i
=
\mathbf qW\mathbf k_i^{\mathsf T},
$$

或者使用一个小网络输出标量。不同 score 函数改变的是寻址规则，不改变 key-value soft read 的接口。

### softmax 把分数变成路由权重

没有 mask 时：

$$
\alpha_i
=
\frac{\exp(s_i)}
{\sum_{j=1}^{S}\exp(s_j)}.
$$

权重有三个基本性质：

$$
\alpha_i\ge0,
\qquad
\sum_{i=1}^{S}\alpha_i=1.
$$

因此 context 是 values 的凸组合：

$$
\mathbf c
=
\sum_{i=1}^{S}\alpha_i\mathbf v_i.
$$

这是一种 soft read。它通常同时读取多个 value，而不是像数组下标那样只返回一个槽位。

## 一个可回放的数值例子

### query 选择 key

设一维 query 和三个一维 keys 为

$$
q=1,
\qquad
k_1=1,
\quad
k_2=0,
\quad
k_3=-1.
$$

使用点积作为 score：

$$
s=(s_1,s_2,s_3)=(1,0,-1).
$$

softmax 权重为

$$
\alpha
=
\operatorname{softmax}(1,0,-1)
\approx
(0.665240956,\ 0.244728471,\ 0.090030573).
$$

最大权重落在 $k_1$，但另外两个槽位仍保留非零读取量。soft read 会把不确定性保留在 context 中。

### value 决定读回的内容

令三个 value 是标量

$$
v_1=2,
\qquad
v_2=0,
\qquad
v_3=-1.
$$

读取结果为

$$
\begin{aligned}
c
&=
0.665240956\times2
+0.244728471\times0
+0.090030573\times(-1)\\
&\approx1.240451339.
\end{aligned}
$$

同一组权重如果配上另一组 values，会得到另一个 context。权重说明读取分布，value 决定读取内容；只看权重不能恢复 context 的全部信息。

### mask 改变可读取集合

如果第 3 个槽位被 mask，令

$$
m=(0,0,-\infty),
\qquad
s'=s+m=(1,0,-\infty).
$$

softmax 只在前两个有效槽位上归一化：

$$
\alpha'
=
\operatorname{softmax}(1,0,-\infty)
\approx
(0.731058579,\ 0.268941421,\ 0).
$$

对应同一组 values 的 context 为

$$
c'
=
0.731058579\times2
+0.268941421\times0
\approx1.462117158.
$$

mask 不是事后删掉权重，而是在归一化前改变竞争集合。padding、causal 约束和非法 memory 槽位都应在这个阶段处理。

## 为什么它像可寻址记忆

### key 是地址，value 是内容

把每个槽位写成 pair：

| memory 槽位 | key 表示什么 | value 表示什么 | 读取时发生什么 |
| --- | --- | --- | --- |
| $i$ | 可以被 query 匹配的地址 | 被返回的内容 | score 决定 $\alpha_i$ |
| $j$ | 另一种地址 | 另一份内容 | 与 $i$ 竞争归一化质量 |
| masked slot | 不可用地址 | 即使存在也不可读 | 权重固定为 0 |

key 和 value 不必具有相同维度。$d_k$ 决定匹配空间，$d_v$ 决定返回给后续层的内容宽度。

### 相同 key 会分摊读取质量

如果两个 keys 完全相同：

$$
\mathbf k_1=\mathbf k_2
\quad\Longrightarrow\quad
s_1=s_2.
$$

在没有其他差异时，它们会获得相同权重。若 values 不同，context 会把两份内容混合；若 values 也相同，两个槽位在当前 read 中不可区分。

这不是 score 函数的 bug，而是地址分辨率的限制。想区分重复内容，需要加入位置、类型、时间、来源或其他 key 特征。

### 一个 query 可以读取多个事实

softmax 的权重和为 1，所以 context 是一个混合结果。它可以同时携带多个槽位的信息，也会带来混叠：

1. 两个相关 key 可能共同贡献；
2. 一个高权重 value 可能覆盖低权重 value；
3. 相似的 keys 会使读出分布变宽；
4. 低权重槽位仍可能通过梯度影响训练；
5. 多个 query 可以从同一 memory 产生不同 context。

attention 不是只能做单点查找。它更接近可微的加权读操作。

## 温度、尖锐度与 hard read

### temperature 改变读取分布

用正温度 $\tau$ 缩放 score：

$$
\alpha_i(\tau)
=
\frac{\exp(s_i/\tau)}
{\sum_{j=1}^{S}\exp(s_j/\tau)}.
$$

$\tau$ 较小时，最大 score 的权重更集中；$\tau$ 较大时，分布更平。极限行为为：

$$
\tau\to0^+
\quad\Longrightarrow\quad
\alpha\text{ 接近 one-hot},
$$

而

$$
\tau\to+\infty
\quad\Longrightarrow\quad
\alpha_i\to\frac1S.
$$

这两个极限帮助解释 soft read 与平均池化的关系，但实际训练始终受有限 score、mask 和浮点精度影响。

### hard argmax 丢失梯度

硬读取可以写成

$$
i^\star
=
\operatorname{arg\,max}_{i}s_i,
\qquad
\mathbf c_{\text{hard}}
=
\mathbf v_{i^\star}.
$$

它返回一个槽位，读取路径清晰，但 argmax 在 score 相等或排序改变处不可微。softmax read 用连续权重保留梯度路径，代价是 context 可能混合多个 value。

| 读取方式 | 输出 | 梯度 | 典型影响 |
| --- | --- | --- | --- |
| hard argmax | 一个 value | 排序边界不可微 | 读取清晰，训练困难 |
| softmax read | values 的凸组合 | score 和 value 都可传播 | 训练稳定，可能混叠 |
| top-k soft read | 选中 k 个槽位后归一化 | 选择边界仍离散 | 降低成本，可能漏召回 |
| uniform average | 所有 value 等权 | 可微 | 不使用 query 匹配 |

top-k 需要先定义候选生成和 mask。若正确槽位没有进入 top-k，后续 softmax 无法恢复它。

## Attention 的梯度路径

### value 路径直接接收权重

对标量 value $v_i$：

$$
c=\sum_i\alpha_i v_i
\quad\Longrightarrow\quad
\frac{\partial c}{\partial v_i}
=
\alpha_i.
$$

高权重槽位从当前 context 收到更大的直接梯度；低权重槽位仍可能经 score 路径收到梯度。

### score 路径通过 softmax 耦合

对标量 values，softmax 的导数给出

$$
\frac{\partial c}{\partial s_i}
=
\alpha_i(v_i-c).
$$

如果 $v_i$ 高于当前 context，增加 $s_i$ 会倾向于增大 $c$；如果 $v_i$ 低于当前 context，增加 $s_i$ 会倾向于降低 $c$。这条式子说明 score 梯度不是只由 $\alpha_i$ 决定，还取决于 value 与当前混合结果的差异。

对向量 values，公式逐坐标成立：

$$
\frac{\partial\mathbf c}{\partial s_i}
=
\alpha_i(\mathbf v_i-\mathbf c).
$$

### mask 会切断非法位置的路径

被 mask 的位置有 $\alpha_i=0$。在理想的无穷小表示下，它不通过 value 加权路径接收梯度；实现中应避免用有限的大负数造成数值泄漏，并检查 dtype 下的 softmax 行为。

## 形状、复杂度与边界

### 一个 query 的读取成本

给定 $S$ 个 keys，一个 query 的主要匹配与读取量级为

$$
O(Sd_k)+O(Sd_v).
$$

如果有 $T$ 个 query，直接形成完整 attention map 的主要 pair 数为

$$
N_{\text{pair}}=TS.
$$

将 query、key、value 写成矩阵：

$$
Q\in\mathbb R^{T\times d_k},
\qquad
K\in\mathbb R^{S\times d_k},
\qquad
V\in\mathbb R^{S\times d_v}.
$$

点积 score 矩阵为

$$
S_{\text{score}}
=
QK^{\mathsf T}
\in\mathbb R^{T\times S}.
$$

按每个 query 的候选轴 softmax：

$$
A
=
\operatorname{softmax}_{\text{key}}(S_{\text{score}}+M)
\in\mathbb R^{T\times S}.
$$

最终 context：

$$
C=AV
\in\mathbb R^{T\times d_v}.
$$

### 形状错误会改变 attention 语义

| 张量 | 期望形状 | 轴含义 | 审计问题 |
| --- | --- | --- | --- |
| $Q$ | $T\times d_k$ | query 位置、query 特征 | query 是否来自目标序列 |
| $K$ | $S\times d_k$ | key 位置、key 特征 | key 是否来自记忆序列 |
| $V$ | $S\times d_v$ | value 位置、value 特征 | value 行是否与 key 对齐 |
| $A$ | $T\times S$ | query 对 key 的权重 | softmax 是否沿 key 轴 |
| $C$ | $T\times d_v$ | 每个 query 的读出 | 是否保留 query 位置 |

如果沿 query 轴做 softmax，得到的是另一种归一化关系。它可能仍然产生合法数值，但不再表示“每个 query 如何分配读取质量”。

### 资源成本由 query 数和 key 数共同决定

对于 encoder-decoder attention，通常 $T$ 是目标长度、$S$ 是源长度；对于 self-attention，二者都来自同一序列。不要把 $T\times S$ 的 map 元素和 $T\times d_v$ 的 context 元素混为一谈。

| 结构 | query 来源 | key/value 来源 | pair 数 |
| --- | --- | --- | ---: |
| encoder-decoder attention | decoder，$T$ | encoder，$S$ | $TS$ |
| self-attention | 同一序列，$T$ | 同一序列，$S=T$ | $T^2$ |
| memory read | 当前状态，$T$ | 外部 memory，$S$ | $TS$ |
| local attention | 当前窗口，$T$ | 局部候选，$w$ | $Tw$ |

局部或 top-k attention 降低 pair 数，但引入候选覆盖和边界选择问题。

## 从 soft read 到检索系统

### attention 权重是一种检索分布

对每个 query，$A_{t,:}$ 是 key 轴上的分布。可以报告：

$$
H_t
=
-\sum_{i=1}^{S}A_{t,i}\log A_{t,i}.
$$

$H_t$ 较低表示读取集中，较高表示读取分散。熵只描述权重分布，不说明 value 是否包含正确内容。

还可以报告有效候选数：

$$
S_{\text{eff},t}
=
\exp(H_t).
$$

它把一个分布转换成“等权候选数”的尺度，便于比较不同 $S$ 的读取集中程度。

### top-k 检索需要区分召回与重排

大 memory 中常先用近似检索取出候选，再用 attention score 重排：

1. candidate generator 产生候选集合 $\mathcal C_t$；
2. 只对 $\mathcal C_t$ 计算精确 score；
3. 在候选集合内做 mask 和 softmax；
4. 对候选 values 做加权读取。

如果真实相关 key 不在 $\mathcal C_t$，精确重排只能在错误候选内优化。报告中应区分 candidate recall、最终读取分布和下游任务结果。

### 读取分布不是因果解释

attention weight 是当前参数和输入下的一次路由系数。它可能与输入特征相关，但单独观察它不能证明：

1. 高权重位置对输出具有不可替代的因果影响；
2. 低权重位置完全没有影响；
3. 权重越大，value 的语义重要性越大；
4. 换一个 query、层或 mask 后仍然保留同一解释。

若要检验因果关系，应进行遮蔽、替换、激活干预或反事实实验，并比较输出变化。

## 与已有 attention 结构的关系

### Bahdanau attention 是一种 score 实现

Bahdanau attention 用 decoder query 与 encoder state 计算 additive energy，再沿源位置做 softmax。放进 key-value 接口：

| 抽象角色 | Bahdanau 结构中的对象 |
| --- | --- |
| query | 当前 decoder hidden state |
| key | encoder hidden state 的匹配投影 |
| value | encoder hidden state 或 value 投影 |
| score | additive energy |
| context | 按源位置加权的 encoder 状态和 |

抽象接口保持不变，score 函数和 query convention 发生变化。

### self-attention 复用同一序列的三份投影

self-attention 中，输入序列 $X$ 通常经过三套线性投影：

$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
$$

每个位置既可以提出 query，也可以作为 key 和 value 被其他位置读取。后续的 scaled-dot-product、multi-head、causal mask 和 KV cache 会进一步规定 score、分组和推理缓存；本篇只保留 key-value retrieval 的共同接口。

## 审计协议

### 先核对一次 forward 的读取合同

给定一个 batch，保存：

1. query、key、value 的来源和 shape；
2. score 函数、缩放因子和 temperature；
3. mask 的有效位置和 softmax 轴；
4. attention map 的 shape、行和以及零权重位置；
5. context 与手算 weighted sum 的差异；
6. value 行与 key 行是否一一对应；
7. dtype、近似检索候选和数值稳定策略。

### 再做四项独立检查

| 检查 | 预期结果 | 失败时先查什么 |
| --- | --- | --- |
| 权重非负 | $A_{t,i}\ge0$ | softmax 输入和 mask |
| 每行归一化 | 有效位置之和为 1 | softmax 轴和 padding |
| mask 生效 | 禁止位置权重为 0 | 加 mask 的阶段和 dtype |
| 读出一致 | $C=AV$ | value/key 排序和矩阵转置 |

对小规模 toy 输入，逐项打印 score、权重、mask 后权重、context 和梯度。只检查最终输出可能掩盖 key/value 错位。

### 一个可回放的 scalar toy

固定

$$
s=(1,0,-1),
\qquad
v=(2,0,-1).
$$

应得到

$$
\operatorname{softmax}(s)
\approx
(0.665240956,\ 0.244728471,\ 0.090030573),
\qquad
c\approx1.240451339.
$$

屏蔽第三个位置后：

$$
\operatorname{softmax}(1,0,-\infty)
\approx
(0.731058579,\ 0.268941421,\ 0),
\qquad
c'\approx1.462117158.
$$

这组数字同时检查 score 顺序、softmax 轴、mask 阶段和 value 对齐。

## 失效模式：有读取权重不等于有解释

### 1. 把 query、key、value 当成同一个对象

三者可以来自不同序列、不同投影和不同维度。应记录来源、shape 和行对齐关系。

### 2. 在错误轴上做 softmax

沿 query 轴归一化会改变权重的语义。应检查每个 query 的有效 key 权重和。

### 3. 在 softmax 后再删除 padding

事后删除会破坏归一化。应在 softmax 前对禁止位置加 mask，再核对有效权重和为 1。

### 4. 把最大权重当作唯一读取

softmax 通常给多个位置非零权重。要使用 hard read，必须明确离散选择和梯度边界。

### 5. 把高熵读取当作错误

高熵可能表示 query 需要组合多个 value，也可能表示 key 区分度不足。应和任务损失、value 内容及基线一起判断。

### 6. 只看 attention map，不看 value

相同权重配不同 values 会产生不同 context。应报告 map 与读出内容两条路径。

### 7. top-k 后忘记报告候选召回

未进入候选集的 key 不可能被重排找回。应分开报告 candidate recall 和最终读取结果。

### 8. 把权重可视化当作因果证据

高权重是路由观察。因果结论需要遮蔽、替换、干预或反事实评估。

## 结语

Attention 作为检索可以压缩为四步：用 query-key score 产生地址分数，用 mask 定义可读集合，用 softmax 得到读取分布，再对 value 做加权读出。这个接口解释了为什么 attention 可以在不同时间步读取不同内容，也解释了为什么 query、key、value 的来源与 shape 必须分开核对。

soft read 保留可微梯度，但会混合多个 value；hard 或 top-k read 提高稀疏性，却引入候选覆盖和离散边界。attention map 是读取路径证据，不是单独的因果解释。完整结论需要同时检查权重、value、mask、候选召回和任务输出。

## 相关词条

[Bahdanau 注意力](../rnn-lstm/bahdanau-attention/)

[为什么需要 Attention](../rnn-lstm/why-attention/)

[自注意力](../attention/self-attention/)

[Scaled Dot-Product Attention](../attention/scaled-dot-product/)

[Attention 矩阵](../attention/attention-matrix/)

[Embedding 几何](../text-representation/embedding-geometry/)

[余弦相似度](../text-representation/cosine-similarity/)
