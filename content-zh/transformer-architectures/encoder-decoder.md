---
title: "Encoder-Decoder：条件序列变换器"
tags: ["why-models-learn"]
---

Encoder-Decoder 是一条把源序列变成目标序列的条件计算链。encoder 对完整源序列做双向 self-attention，产生形状为 $(B,S,D)$ 的 encoder memory；decoder 对目标前缀做 causal self-attention，再用 cross-attention 从这份 memory 中读取信息，最后在每个目标位置产生词表 logits。这里的 $S$ 是源长度，$U$ 是目标长度；两个长度可以不同，两个词表也可以不同。

本文把经典 Seq2Seq 接口与 Transformer 实现放在同一份 shape 合同中。先固定两条序列的输入和目标移位，再展开 encoder、decoder masked self-attention、decoder cross-attention、padding/causal mask、teacher forcing 与自回归推理。随后用一个小配置独立核对参数量、MAC、cross-attention 的数字读取和静态源 K/V cache，最后比较 encoder-only、decoder-only 与 encoder-decoder 的边界。

![Encoder-Decoder 的两条序列和三条 attention 路径：encoder 产生源 memory，decoder 读取目标前缀并通过 cross-attention 读取源位置，最后输出目标 logits](/assets/transformer-architectures/svg/encoder-decoder.1.svg)

## 先固定两条序列的接口

### 源轴、目标轴和特征轴

设 batch size 为 $B$，源序列长度为 $S$，目标训练长度为 $U$，源词表和目标词表大小分别为 $V_{\mathrm s}$、$V_{\mathrm t}$。encoder 和 decoder 共用模型宽度 $D$。每个 attention 有 $h$ 个 query head，单个 head 的宽度为

$$
d_h=\frac{D}{h}.
$$

本文先使用标准多头 attention，使 encoder 和 decoder 的 query、key、value head 数都为 $h$。GQA 和 MQA 可以把 decoder 的 K/V head 数降为 $h_{\mathrm{kv}}$，但不改变 encoder memory 的源位置轴；它们的缓存账本见 [GQA 与 MQA](../attention/gqa-and-mqa/)。

两条序列的张量合同如下：

| 张量 | shape | 轴含义 | 产生位置 |
| --- | --- | --- | --- |
| source token id | $(B,S)$ | batch、源位置 | encoder 输入 |
| target input id | $(B,U)$ | batch、目标位置 | decoder 输入 |
| encoder hidden | $(B,S,D)$ | batch、源位置、特征 | encoder 输出 |
| decoder hidden | $(B,U,D)$ | batch、目标位置、特征 | decoder 输出 |
| target logits | $(B,U,V_{\mathrm t})$ | batch、目标位置、目标词表 | 输出头 |

$S$ 和 $U$ 不是同一个轴。encoder self-attention 的 score 有两个源位置轴，decoder masked self-attention 有两个目标位置轴，cross-attention 则同时拥有目标 query 轴和源 key 轴：

$$
\text{encoder self}: (B,h,S,S),
\qquad
\text{decoder self}: (B,h,U,U),
\qquad
\text{cross}: (B,h,U,S).
$$

最后一个 shape 是排查实现错误的关键。若 cross-attention 被写成 $(B,h,S,U)$，需要先说明交换的是哪两个逻辑轴；不能只依靠矩阵乘法能够运行来判断合同正确。

### 两个词表和两个 embedding

源 token id 属于源词表，目标 token id 属于目标词表：

$$
I^{\mathrm s}\in\{0,\ldots,V_{\mathrm s}-1\}^{B\times S},
\qquad
I^{\mathrm t}\in\{0,\ldots,V_{\mathrm t}-1\}^{B\times U}.
$$

最直接的参数化使用两个 embedding 矩阵：

$$
E_{\mathrm s}\in\mathbb R^{V_{\mathrm s}\times D},
\qquad
E_{\mathrm t}\in\mathbb R^{V_{\mathrm t}\times D}.
$$

查表得到：

$$
H^{\mathrm e}_{0,b,s,:}=E_{\mathrm s}[I^{\mathrm s}_{b,s},:],
\qquad
H^{\mathrm d}_{0,b,u,:}=E_{\mathrm t}[I^{\mathrm t}_{b,u},:].
$$

这里的方括号表示按 token id 选择矩阵行，不表示把离散 id 当作连续数值参与线性变换。源词表和目标词表可以相同，也可以不同：

| 选择 | 输入 embedding | 输出 head | 适用约束 |
| --- | --- | --- | --- |
| 独立词表 | $E_{\mathrm s},E_{\mathrm t}$ 分开 | 目标 head 独立 | 源/目标语言或符号集合差异大 |
| 共享词表 | 一个 $E$ | 可与目标 head 绑定 | token id、宽度和特殊符号合同一致 |
| 目标权重绑定 | $E_{\mathrm t}$ | $W_{\mathrm{lm}}=E_{\mathrm t}^{\mathsf T}$ | 输入输出目标词表相同 |
| 三者全共享 | 一个 $E$ | 同一矩阵转置 | 只有在源、目标 id 语义和词表都兼容时成立 |

共享参数减少计数，但不会自动改变计算图。必须同时检查 source id、target id、BOS/EOS/PAD 和输出 logits 的词表索引。

### 位置条件分别作用在两条时间轴

使用可学习位置表时，源和目标位置通常分别编号：

$$
H^{\mathrm e}_0=E_{\mathrm s}[I^{\mathrm s}]+P_{\mathrm s}[0:S],
\qquad
H^{\mathrm d}_0=E_{\mathrm t}[I^{\mathrm t}]+P_{\mathrm t}[0:U].
$$

源位置 $s$ 描述 encoder 读取到的源顺序，目标位置 $u$ 描述 decoder 正在预测的目标顺序。推理时，decoder 的目标位置会随着已经生成的 token 增长；encoder 的源位置不会因为每个 decode step 而重新编号。

RoPE 把位置作用到 Q/K，ALiBi 把相对位置作用到 score，可学习位置表把向量直接加到 hidden。三种方案的具体形式不同，但都必须保持以下接口：源 memory 仍为 $(B,S,D)$，目标 hidden 仍为 $(B,U,D)$，cross score 仍为 $(B,h,U,S)$。[位置编码](../transformer-components/positional-encoding/)、[RoPE](../transformer-components/rope/) 和 [ALiBi](../transformer-components/alibi/)分别展开这些位置路径。

## encoder：把源序列变成 memory

### pre-norm encoder block

设第 $\ell$ 层 encoder 输入为 $H^{\mathrm e}_{\ell-1}$，归一化模块为 $N^{\mathrm e}_{\ell,1}$ 和 $N^{\mathrm e}_{\ell,2}$。常见的 pre-norm block 写成：

$$
\begin{aligned}
R^{\mathrm e}_{\ell}
&=H^{\mathrm e}_{\ell-1}
  +\operatorname{MHA}^{\mathrm e}_{\ell}
    \left(N^{\mathrm e}_{\ell,1}(H^{\mathrm e}_{\ell-1});M^{\mathrm e}\right),\\
H^{\mathrm e}_{\ell}
&=R^{\mathrm e}_{\ell}
  +\operatorname{FFN}^{\mathrm e}_{\ell}
    \left(N^{\mathrm e}_{\ell,2}(R^{\mathrm e}_{\ell})\right).
\end{aligned}
$$

每一层的输入和输出 shape 都是 $(B,S,D)$。self-attention 混合源位置，FFN 在每个源位置独立地变换特征，残差加法把两条支路写回同一个 hidden 宽度。[LayerNorm 与残差流](../transformer-components/layernorm-residuals/)和 [Pre-Norm 与 Post-Norm](../transformer-components/pre-norm-vs-post-norm/)处理归一化、残差和 Jacobian 的关系；本文只固定数据流接口。

### encoder self-attention 读取完整源轴

令

$$
X=N^{\mathrm e}_{\ell,1}(H^{\mathrm e}_{\ell-1}).
$$

第 $r$ 个 head 的投影为：

$$
Q_r=XW^{Q,\mathrm e}_r,
\qquad
K_r=XW^{K,\mathrm e}_r,
\qquad
V_r=XW^{V,\mathrm e}_r,
$$

其中每个张量都可以整理为 $(B,S,d_h)$。score、权重和读取结果为：

$$
\begin{aligned}
Z_r&=\frac{Q_rK_r^{\mathsf T}}{\sqrt{d_h}}+M^{\mathrm e},\\
A_r&=\operatorname{softmax}_{\mathrm{key}}(Z_r),\\
O_r&=A_rV_r.
\end{aligned}
$$

$A_r$ 的每一行沿 key 轴归一化。对于一个没有 padding 的源序列，任意 query 位置都可以读取全部有效源位置；因此一个 head 的逻辑 score 是 $S\times S$。经过 head 拼接和输出投影后：

$$
\operatorname{MHA}^{\mathrm e}(X;M^{\mathrm e})
=\operatorname{Concat}(O_1,\ldots,O_h)W^{O,\mathrm e}.
$$

encoder 没有目标方向上的因果限制。源 token 在被允许的情况下可以读取左侧和右侧源 token；这使每个源位置的 hidden 成为完整源上下文的表示。padding 仍然必须从 key 轴屏蔽，否则空白位置会进入 memory。

### encoder 的 padding mask

设 $m^{\mathrm s}_{b,j}=1$ 表示源位置 $j$ 有效，$0$ 表示 PAD。按 key 轴广播的 mask 可以写成：

$$
M^{\mathrm e}_{b,i,j}
=
\begin{cases}
0,&m^{\mathrm s}_{b,j}=1,\\
-\infty,&m^{\mathrm s}_{b,j}=0.
\end{cases}
$$

这里 $i$ 是 query 源位置，$j$ 是 key 源位置。对于一个 batch 中长度不同的源序列，所有样本可以填充到同一 $S$，但每个样本的有效 key 集合不同。mask 要广播到 batch、head 和 query 轴；只在输入 embedding 阶段把 PAD 向量置零不能替代 score mask，因为后续线性投影和残差仍可能把该位置写入 attention 结果。

### memory 是可按源位置读取的中间接口

经过 $L_{\mathrm e}$ 层 encoder 后，得到：

$$
M^{\mathrm{enc}}=H^{\mathrm e}_{L_{\mathrm e}}\in\mathbb R^{B\times S\times D}.
$$

这里用 $M^{\mathrm{enc}}$ 表示 encoder memory，用 $M^{\mathrm e}$ 表示 attention mask。二者含义不同：前者是被 decoder 读取的连续表示，后者是限制读取连接的加性分数。

在后续 cross-attention 中，源位置轴仍然保留。第 $s$ 个 memory 向量可以携带局部 token、句法上下文或更远的源信息；decoder 不需要从一个固定长度的单向量中恢复整条源序列。经典 [Seq2Seq](../rnn-lstm/seq2seq/) 的 fixed context 版本会把源状态压缩成一个 bridge，[为什么需要 attention](../rnn-lstm/why-attention/)说明了保留可寻址源状态的动机。

## decoder：在目标前缀上读取 source memory

### 目标右移建立输入—标签合同

设目标真值为 $y_1,\ldots,y_U$。训练时 decoder 输入不是同一位置的目标，而是右移后序列：

$$
x^{\mathrm d}_1=\langle\mathrm{bos}\rangle,
\qquad
x^{\mathrm d}_u=y_{u-1}\quad (u=2,\ldots,U).
$$

于是第 $u$ 个 decoder hidden 预测 $y_u$。一个长度为 3 的例子如下：

| 目标位置 | decoder 输入 | 标签 | decoder self 可读取的输入 |
| --- | --- | --- | --- |
| $1$ | BOS | $y_1$ | BOS |
| $2$ | $y_1$ | $y_2$ | BOS、$y_1$ |
| $3$ | $y_2$ | $y_3$ | BOS、$y_1$、$y_2$ |

推理时没有真值 $y_u$，第 $u+1$ 步输入改为模型在第 $u$ 步选出的 token。训练和推理使用的 decoder block 可以相同，但输入前缀的来源不同。

### decoder masked self-attention 只读取目标前缀

令第 $\ell$ 层 decoder 的输入为 $H^{\mathrm d}_{\ell-1}$。masked self-attention 的 query、key、value 都来自目标 hidden：

$$
Q_r^{\mathrm{self}}=XW^{Q,\mathrm{self}}_r,
\qquad
K_r^{\mathrm{self}}=XW^{K,\mathrm{self}}_r,
\qquad
V_r^{\mathrm{self}}=XW^{V,\mathrm{self}}_r,
$$

其中 $X=N^{\mathrm d}_{\ell,1}(H^{\mathrm d}_{\ell-1})$，每个投影的逻辑序列轴长度为 $U$。causal mask 的有效区域是下三角：

$$
M^{\mathrm{causal}}_{u,j}
=
\begin{cases}
0,&j\leq u,\\
-\infty,&j>u.
\end{cases}
$$

目标位置 $u$ 可以读取当前位置的 decoder 输入和所有更早的输入，但不能读取 $u+1,\ldots,U$ 的目标输入。由于目标输入已右移，当前位置的输入是已知前缀，当前位置的输出才对应下一个目标标签。[因果掩码](../attention/causal-masking/)展开下三角、padding、packed sequence 和 decode offset。

### cross-attention 把目标 query 接到源 memory

masked self-attention 之后，decoder 用自己的 hidden 产生 query，用 encoder memory 产生 key/value。令

$$
X^{\mathrm d}=N^{\mathrm d}_{\ell,2}(R^{\mathrm d}_{\ell}),
\qquad
M^{\mathrm{enc}}=H^{\mathrm e}_{L_{\mathrm e}}.
$$

第 $r$ 个 head 的三组投影为：

$$
Q_r^{\mathrm{cross}}=X^{\mathrm d}W^{Q,\mathrm{cross}}_r,
\qquad
K_r^{\mathrm{cross}}=M^{\mathrm{enc}}W^{K,\mathrm{cross}}_r,
\qquad
V_r^{\mathrm{cross}}=M^{\mathrm{enc}}W^{V,\mathrm{cross}}_r.
$$

它们的逻辑 shape 分别为 $(B,U,d_h)$、$(B,S,d_h)$ 和 $(B,S,d_h)$。因此：

$$
\begin{aligned}
Z_r^{\mathrm{cross}}
&=\frac{Q_r^{\mathrm{cross}}(K_r^{\mathrm{cross}})^{\mathsf T}}{\sqrt{d_h}}
  +M^{\mathrm{cross}},\\
A_r^{\mathrm{cross}}
&=\operatorname{softmax}_{\mathrm{source}}(Z_r^{\mathrm{cross}}),\\
O_r^{\mathrm{cross}}
&=A_r^{\mathrm{cross}}V_r^{\mathrm{cross}}.
\end{aligned}
$$

交叉注意力权重的 shape 是 $(B,h,U,S)$。每一个目标 query 都沿源 key 轴归一化，得到一行长度为 $S$ 的读取权重。cross-attention 没有 decoder self-attention 那种目标下三角 mask：目标位置 $u$ 可以读取所有有效源位置，但不能读取未来目标位置，因为目标信息不在 cross 的 key/value 中。[交叉注意力](../attention/cross-attention/)专门推导这个 $U\times S$ 的读取接口。

### 一个两源位置的 cross-attention 数字例子

用一个 head 说明“目标 query 选择源位置”。取

$$
q=(1,0),
\qquad
k_1=(1,0),
\qquad
k_2=(0,1),
\qquad
d_h=2.
$$

令两个源位置提供的 value 为

$$
v_1=(1,0),
\qquad
v_2=(0,2).
$$

未归一化 score 为：

$$
\left(
\frac{q\cdot k_1}{\sqrt 2},
\frac{q\cdot k_2}{\sqrt 2}
\right)
=
\left(\frac{1}{\sqrt 2},0\right)
=
(0.707106781187,0).
$$

softmax 权重为：

$$
(a_1,a_2)
=(0.669761549327,0.330238450673).
$$

读取结果为：

$$
a_1v_1+a_2v_2
=(0.669761549327,0.660476901347).
$$

输出的两个坐标同时包含两个源 value，但第一个源位置权重更大。这个例子只核对读取算子；真实模型还会学习 Q/K/V 投影、输出投影和多头组合，不能把单个 attention row 当作独立的对齐真值。

### 一个 pre-norm decoder block

标准 Transformer decoder block 有三条顺序支路：masked self-attention、cross-attention、FFN。写成：

$$
\begin{aligned}
R^{\mathrm d}_{\ell}
&=H^{\mathrm d}_{\ell-1}
  +\operatorname{MHA}^{\mathrm{self}}_{\ell}
    \left(N^{\mathrm d}_{\ell,1}(H^{\mathrm d}_{\ell-1});M^{\mathrm{causal}},M^{\mathrm{pad}}\right),\\
C^{\mathrm d}_{\ell}
&=R^{\mathrm d}_{\ell}
  +\operatorname{MHA}^{\mathrm{cross}}_{\ell}
    \left(N^{\mathrm d}_{\ell,2}(R^{\mathrm d}_{\ell});M^{\mathrm{cross}}\right),\\
H^{\mathrm d}_{\ell}
&=C^{\mathrm d}_{\ell}
  +\operatorname{FFN}^{\mathrm d}_{\ell}
    \left(N^{\mathrm d}_{\ell,3}(C^{\mathrm d}_{\ell})\right).
\end{aligned}
$$

三条残差支路的 shape 都是 $(B,U,D)$。第一条支路混合目标前缀，第二条支路把源 memory 写入目标表示，第三条支路逐目标位置做特征变换。若实现只有 masked self-attention 和 FFN，就得到 decoder-only block；缺少 cross-attention 就没有源条件接口。

### decoder 的三个 mask 责任

三个 mask 处理的轴不同：

| mask | 作用位置 | 屏蔽的对象 | 是否改变 cross 的方向 |
| --- | --- | --- | --- |
| target causal | decoder self 的 key 轴 | 当前目标位置之后的 target key | 不涉及 source |
| target padding | decoder self 的 key/query 与 loss | PAD target key、无效 target query | 不涉及 source |
| source padding | encoder self 与 cross 的 source key | PAD source key | 只保留有效 source key |
| source visibility | cross 的 source key | 任务规定不可读的 source 区域 | 可以比 padding 更严格 |

source padding mask 要同时进入 encoder self-attention 和 decoder cross-attention。只在 encoder 内屏蔽 source PAD，不能保证 decoder 不读取 memory 中对应位置。target query mask 还要进入 loss reduction，否则 PAD 行会改变平均损失分母。

## 训练：teacher forcing 把整段目标并行化

### 条件概率分解

encoder 先读取整条源序列 $x_{1:S}$，decoder 再按目标前缀建模条件分布：

$$
p(y_{1:U}\mid x_{1:S})
=
\prod_{u=1}^{U}p(y_u\mid y_{<u},x_{1:S}).
$$

训练时可以把右移后的目标输入一次性送入 decoder。因果 mask 让第 $u$ 行只读取 $\langle\mathrm{bos}\rangle,y_1,\ldots,y_{u-1}$，所以 $U$ 个目标位置可以在一次矩阵前向中并行计算；并行计算不代表目标之间失去因果关系，因果关系由 mask 和右移共同保证。

### masked cross-entropy

设 decoder 输出 logits 为 $Z_{b,u,:}$，标签为 $y_{b,u}$，target 有效 mask 为 $m^{\mathrm t}_{b,u}$。逐位置负对数似然为：

$$
\ell_{b,u}
=-
\log\operatorname{softmax}(Z_{b,u,:})_{y_{b,u}}.
$$

只对有效目标位置做平均：

$$
L
=
\frac{\sum_{b=1}^{B}\sum_{u=1}^{U}m^{\mathrm t}_{b,u}\ell_{b,u}}
{\sum_{b=1}^{B}\sum_{u=1}^{U}m^{\mathrm t}_{b,u}}.
$$

如果把 source PAD 放进 target mask，或者把 target PAD 也计入分母，损失数值和梯度都会改变。source mask 决定 decoder 能读取哪些条件，target mask 决定哪些预测进入监督；两者不能用同一张矩形 mask 代替。

### teacher forcing 的输入来源

teacher forcing 在第 $u$ 个位置使用真实的 $y_{u-1}$ 作为输入。它使训练前向可并行，但训练条件分布和自由运行推理条件分布不同：

| 阶段 | 第 $u$ 步输入 | 可用源信息 | 目标序列计算方式 |
| --- | --- | --- | --- |
| teacher forcing | 真实 $y_{u-1}$ | encoder 全部有效 memory | 所有位置一次前向 |
| free-running | 模型上一步输出 | encoder 全部有效 memory | 每步追加一个 token |
| scheduled sampling | 按规则混合真值和模型 token | encoder 全部有效 memory | 依实现逐步或分段 |

teacher forcing 不是 decoder 架构本身，而是训练时提供前缀的协议。[Teacher Forcing](../rnn-lstm/teacher-forcing/)处理 exposure bias、scheduled sampling 和输入—标签对齐；本文只把它固定为 encoder-decoder 的训练接口。

## 推理：encoder 一次，decoder 逐步生成

### 两阶段推理路径

给定源 token $x_{1:S}$，自回归推理可以分成两阶段：

1. 把源 token 和 source position 送入 encoder，得到 $M^{\mathrm{enc}}\in\mathbb R^{B\times S\times D}$。
2. 为每个 decoder step 提供 BOS 或上一步生成的 token。
3. 用 decoder masked self-attention 读取已经存在的 target prefix。
4. 用 cross-attention 读取同一份 source memory。
5. 通过目标输出 head 得到 next-token logits。
6. 依据 greedy、sampling 或 beam search 选择下一个 target token。
7. 遇到 EOS 或长度上限后停止对应样本。

encoder 的结果可以在所有 decoder step 之间复用。新目标 token 只增加 decoder self-attention 的历史长度，不改变 source memory 的长度。

### prefill 与 decode

若已有目标前缀 $y_{1:t}$，可以先把整段前缀一次性送入 decoder，这一步称为 prefill。之后每一步只送入一个新 token，这一步称为 decode：

| 阶段 | decoder query 数 | target self 的 key 长度 | cross 的 key 长度 | source encoder |
| --- | ---: | ---: | ---: | --- |
| 训练 | $U$ | $U$ | $S$ | 一次 |
| prefill | $t$ | $t$ | $S$ | 已计算或一次 |
| 单步 decode | $1$ | $t+1$ | $S$ | 不重复计算 |

prefill 和逐 token decode 应该在相同 position、mask 和 dtype 条件下给出一致的 next-token logits。两条路径不一致时，优先检查目标 position offset、causal mask 的历史长度、RoPE 是否重复施加，以及 cache 是否沿 target sequence 轴追加。

### 两种 cache 的生长方式

decoder self-attention 的历史 K/V 随目标长度增长：

$$
K^{\mathrm{self}},V^{\mathrm{self}}
\in\mathbb R^{B\times h\times t\times d_h}.
$$

cross-attention 的 source K/V 可以在第一个 decoder step 之前从固定 memory 预计算：

$$
K^{\mathrm{cross}},V^{\mathrm{cross}}
\in\mathbb R^{B\times h\times S\times d_h}.
$$

因此单步 decode 时，target self cache 的长度从 $t$ 增长到 $t+1$，static source cache 的长度保持为 $S$。两者都属于运行时状态，但生命周期不同：source cache 随一条源请求固定，target self cache 随生成步数增长。

若使用 beam search，source memory 和 static source cache 要按 beam 选择结果同步重排；target self cache 也要按同一 beam 顺序重排。只重排 logits 或 token id，会让某个 beam 读取另一个 beam 的历史状态。[Beam Search](../inference/beam-search/)处理候选路径，[因果掩码](../attention/causal-masking/)处理 cache offset 与可读集合。

## 一个小配置的参数账本

### 固定数值

取一个便于手算的 encoder-decoder：

| 量 | 数值 | 含义 |
| --- | ---: | --- |
| $B$ | $1$ | batch size |
| $S$ | $2$ | 源序列长度 |
| $U$ | $3$ | 目标训练长度 |
| $V_{\mathrm s}$ | $6$ | 源词表大小 |
| $V_{\mathrm t}$ | $8$ | 目标词表大小 |
| $D$ | $4$ | hidden width |
| $h$ | $2$ | attention head 数 |
| $d_h$ | $2$ | 单 head width |
| $M$ | $8$ | FFN 中间宽度 |
| $L_{\mathrm e},L_{\mathrm d}$ | $1,1$ | encoder、decoder 层数 |

参数账本先排除 bias、额外 final norm 和 dropout 状态。这样每一项都能直接对应到一条计算支路。

### encoder 与 decoder layer 参数

一个标准 attention 的 Q/K/V/O 四个投影共有 $4D^2$ 个参数，一个两层 FFN 共有 $2DM$ 个参数，一个 LayerNorm 的 scale 和 bias 共有 $2D$ 个参数。

encoder layer 有一个 self-attention、一个 FFN 和两个 LayerNorm：

$$
P_{\mathrm e,layer}
=4D^2+2DM+4D
=4\cdot4^2+2\cdot4\cdot8+4\cdot4
=144.
$$

decoder layer 有一个 masked self-attention、一个 cross-attention、一个 FFN 和三个 LayerNorm：

$$
P_{\mathrm d,layer}
=8D^2+2DM+6D
=8\cdot4^2+2\cdot4\cdot8+6\cdot4
=216.
$$

cross-attention 的 Q 来自目标 hidden，K/V 来自 source memory，但参数量仍然是四个 $D\times D$ 投影。输入来源不同，不会把 projection 数减少一半。

### embedding、位置表和输出头

源 embedding、目标 embedding、源位置表和目标位置表的参数分别为：

$$
P_{\mathrm{emb}}
=V_{\mathrm s}D+V_{\mathrm t}D+SD+UD
=24+32+8+12
=76.
$$

若目标输出 head 与 $E_{\mathrm t}$ 绑定，$W_{\mathrm{lm}}=E_{\mathrm t}^{\mathsf T}$，则不再添加一份 $V_{\mathrm t}\times D$ 参数。总参数为：

$$
P_{\mathrm{tied}}
=P_{\mathrm{emb}}+P_{\mathrm e,layer}+P_{\mathrm d,layer}
=76+144+216
=436.
$$

若目标输出 head 不绑定目标 embedding，则增加 $V_{\mathrm t}D=32$：

$$
P_{\mathrm{untied}}=436+32=468.
$$

这个总数依赖于明确写出的简化条件。加入 bias、final norm、多层 encoder/decoder 或不同位置参数化后，应按实际模块逐项增加，不能把 $436$ 当作架构常数。

## 一个小配置的计算账本

### encoder self-attention

用 MAC 近似一次矩阵乘法的乘加数。encoder self-attention 的投影、score/value 和 FFN 三项为：

$$
\begin{aligned}
\mathrm{MAC}_{\mathrm e,proj}&=4SD^2,\\
\mathrm{MAC}_{\mathrm e,attn}&=2hS^2d_h,\\
\mathrm{MAC}_{\mathrm e,ffn}&=2SDM.
\end{aligned}
$$

代入 $S=2,D=4,h=2,d_h=2,M=8$：

$$
(\mathrm{MAC}_{\mathrm e,proj},
\mathrm{MAC}_{\mathrm e,attn},
\mathrm{MAC}_{\mathrm e,ffn})
=(128,32,128),
$$

所以一个 encoder layer 的主 MAC 为 $288$。这不包含 embedding 查表、LayerNorm、softmax 指数与归一化等逐元素开销。

### decoder masked self 与 cross-attention

目标 masked self-attention 的三项为：

$$
\begin{aligned}
\mathrm{MAC}_{\mathrm{d,self\text{-}proj}}&=4UD^2,\\
\mathrm{MAC}_{\mathrm{d,self\text{-}attn}}&=2h\frac{U(U+1)}{2}d_h,\\
\mathrm{MAC}_{\mathrm{d,ffn}}&=2UDM.
\end{aligned}
$$

取 $U=3$，得到 $(192,48,192)$。

cross-attention 的 Q/O 目标侧投影和 K/V 源侧投影为：

$$
\mathrm{MAC}_{\mathrm{cross,proj}}
=2(U+S)D^2
=160.
$$

cross 的 score 与 value 读取为：

$$
\mathrm{MAC}_{\mathrm{cross,attn}}
=2hUSd_h
=48.
$$

因此 decoder layer 的主 MAC 分解为：

$$
(192,48,160,48,192),
$$

总数为 $640$。顺序对应 masked self projection、masked self score/value、cross projection、cross score/value、FFN。

### 输出头和一次 decode step

训练前向还要把每个 decoder hidden 投影到目标词表：

$$
\mathrm{MAC}_{\mathrm{lm}}=UDV_{\mathrm t}=3\cdot4\cdot8=96.
$$

如果已预计算 source K/V，一次 decode step 只处理一个新目标 query。令已有 target cache 长度为 $t=2$，则单层的主要 MAC 为：

| 支路 | 单步 MAC | 计算条件 |
| --- | ---: | --- |
| target self projection | $4D^2=64$ | 只投影新 token |
| target self score/value | $2h(t+1)d_h=24$ | 新 query 读取 3 个 target key |
| cross Q/O projection | $2D^2=32$ | source K/V 已缓存 |
| cross score/value | $2hSd_h=16$ | 新 query 读取 2 个 source key |
| FFN | $2DM=64$ | 只处理新 token |
| target LM head | $DV_{\mathrm t}=32$ | 产生 next-token logits |
| 合计 | $232$ | 不含逐元素开销 |

若每步重新投影 source memory 的 K/V，需要额外加入 $2SD^2=64$，单步主 MAC 变为 $296$。static source KV cache 的价值可以直接从这两个数字看出：它消除了随 decode step 重复的 source projection，但没有消除每步对 $S$ 个 source key/value 的读取。

### cache bytes

标准多头下，一层 decoder 的 self K/V 和 cross K/V 元素数为：

$$
N_{\mathrm{cache}}
=2Bhd_h(t+S).
$$

上式中的因子 2 分别对应 K 和 V。若有 $L_{\mathrm d}$ 层、每个元素使用 $b$ bytes：

$$
\mathrm{bytes}_{\mathrm{cache}}
=2L_{\mathrm d}Bhd_h(t+S)b.
$$

在本例 $L_{\mathrm d}=1,B=1,h=2,d_h=2,t=2,S=2$ 且使用 FP16 时：

$$
2\cdot1\cdot1\cdot2\cdot2\cdot(2+2)\cdot2
=64\ \text{bytes}.
$$

若 decoder 层数改为 2，则为 $128$ bytes。decoder self cache 随 $t$ 增长，cross cache 的 $S$ 项保持固定；GQA/MQA 会把其中的 K/V head 从 $h$ 改为 $h_{\mathrm{kv}}$，但 query score 的逻辑读取仍需按 query head 计数。

## 三种架构的边界

encoder-decoder 保留两条序列和 cross-attention。删除其中一条路径，得到的不是同一种模型的不同命名，而是不同的可见性和推理协议：

| 架构 | encoder self | decoder self | cross-attention | 典型输出 |
| --- | --- | --- | --- | --- |
| encoder-only | 双向 | 无 | 无 | token/sequence 表示或分类 logits |
| decoder-only | 无 | causal | 无 | 给定前缀的 next-token logits |
| encoder-decoder | 双向 source | causal target | 有 | 给定 source 的目标序列 logits |
| full Transformer | 双向 source | causal target | 有 | 原始 encoder-decoder 计算链 |

full Transformer 是这一结构的完整实例；encoder-decoder 作为架构接口还可以替换位置方案、归一化、FFN、词表和解码策略。[完整 Transformer](../transformer-architectures/full-transformer/)给出从 token 到 logits 的整条数据流，[Encoder-Only](../transformer-architectures/encoder-only/)和 [Decoder-Only](../transformer-architectures/decoder-only/)分别固定两种删减后的 mask 与推理方式。

### 什么时候需要两条序列

当输入和输出承担不同角色，且输出长度或 token 顺序依赖输入内容时，encoder-decoder 的两个轴有直接含义：

| 任务形态 | source | target | cross-attention 读取的内容 |
| --- | --- | --- | --- |
| 翻译 | 源语言句子 | 目标语言句子 | 源语言上下文 |
| 摘要 | 长文档 | 摘要文本 | 文档中的可寻址片段 |
| 语音识别 | 声学帧或离散声学 token | 文本 token | 音频时间轴表示 |
| 图像到文本 | 图像 patch 表示 | 文本 token | 视觉 encoder memory |
| 条件生成 | 条件序列或结构化输入 | 生成序列 | 条件表示 |

如果输入和输出实际上是同一条前缀上的 continuation，decoder-only 往往可以把条件和目标拼到一个 causal 序列中；这会改变 mask、位置轴和训练标签对齐，不能只把 encoder 模块删除后宣称等价。

## 失效模式：两条序列接口中的边界错误

### 把 $S$ 和 $U$ 合并成一个长度

现象是代码在固定长度样例上可以运行，但 cross score、padding mask 或输出 logits 在源/目标长度不同的样例上发生转置或广播错误。先打印 encoder hidden、decoder hidden 和 cross score 的 shape，再用 $S=2,U=3$ 的非对称配置执行一次前向。

### cross-attention 使用了错误的来源

cross 的 query 必须来自 decoder hidden，key/value 必须来自 encoder memory。若三者都来自 decoder，就得到第二个 self-attention；若三者都来自 encoder，目标位置就没有产生读取条件。检查 Q 的长度为 $U$，K/V 的长度为 $S$。

### 给 cross-attention 施加目标 causal mask

cross 的 key 轴是 source 轴，不是 target 轴。把下三角 mask 直接复用到 cross 会让目标位置只能读取前几个 source token，结果会随源位置编号产生非任务约束的截断。cross 是否需要额外 source visibility mask，要按任务定义；不能按 decoder self 的下三角复制。

### encoder 忘记屏蔽 source PAD

source PAD 可能先进入 encoder self-attention，再作为 memory 的 key/value 被 decoder 读取。检查 source mask 是否同时进入 encoder self 和 cross；不要只检查输入 PAD embedding 是否为零。

### target 右移和 loss 标签错位

第 $u$ 个输入若包含 $y_u$，模型可以复制当前标签。第 $u$ 个 hidden 若对齐到 $y_{u-1}$，loss 则会把预测和标签错一位。对一条短序列逐行打印 BOS、decoder input、label、causal 可读集合。

### target PAD 改变平均损失

把所有矩形位置直接求 mean 会把短样本的 PAD 预测加入分母。独立记录有效 target token 数，核对分子和分母都使用 $m^{\mathrm t}$；source mask 不应代替 target loss mask。

### source cache 和 target cache 混用

source cross K/V 的长度应固定为 $S$，target self K/V 的长度应随 $t$ 增长。若把 source K/V 追加到 target cache，或者把 target 历史当作 cross key，shape 可能仍然满足矩阵乘法，但读取语义已经改变。

### beam reorder 只重排 token

生成候选发生 beam reorder 后，decoder hidden、target self cache、source memory 引用和 source cross cache 必须使用同一索引重排。只重排 token id 会把一个 beam 的历史与另一个 beam 的条件拼接。

### 共享 embedding 的词表条件不成立

source 和 target 词表即使大小相同，也不代表 token id 语义相同。共享 embedding 前检查 tokenizer、特殊符号、词表顺序、padding id、BOS/EOS id 和输出 head 的参数身份。

## 可复用的核验协议

审计一个 encoder-decoder 实现时，按以下顺序记录：

1. 写出 $B,S,U,V_{\mathrm s},V_{\mathrm t},D,h,h_{\mathrm{kv}},d_h,M,L_{\mathrm e},L_{\mathrm d}$ 和位置方案。
2. 打印 source id、target input id、target label、BOS/EOS/PAD id 和每条序列的有效长度。
3. 分别检查 encoder hidden $(B,S,D)$、decoder hidden $(B,U,D)$ 和 cross score $(B,h,U,S)$。
4. 用非对称 $S\neq U$ 的样例检查 source/target mask 的广播轴。
5. 用手算 query、key、value 核对一个 cross-attention row 的 softmax 和读取结果。
6. 核对 target 右移，确认第 $u$ 个输入不包含标签 $y_u$。
7. 独立计算 source/target embedding、位置表、encoder layer、decoder layer、输出 head 和 tied/untied 总参数。
8. 分开计算训练 MAC、prefill MAC、decode 单步 MAC、self cache bytes 和 static source cache bytes。
9. 比较 prefill 与逐 token decode 的 logits，记录 position offset、mask、cache 长度、dtype 和容差。
10. 用 source padding、target padding、不同 $S/U$、EOS 提前结束和 beam reorder 做回归。

第 3、5、6 步应先于 kernel 融合和量化。FlashAttention、fused FFN、paged cache 可以改变实现路径和访存，但不能改变 cross 的 $(U,S)$ 方向、target 的 causal 可读集合、source padding 的屏蔽范围和 cache 的生命周期。[参数量总账](../transformer-components/parameter-count/)处理参数、MAC、激活和状态账本；[Masked Language Modeling](../transformer-architectures/masked-language-modeling/)与 [Causal Language Modeling](../transformer-architectures/causal-language-modeling/)分别处理两类预训练目标，不应把训练目标名称当作架构名称。

## 相关词条

- [Seq2Seq](../rnn-lstm/seq2seq/)：处理 encoder 和 decoder 的条件序列接口、BOS/EOS/PAD 与变长输出。
- [Teacher Forcing](../rnn-lstm/teacher-forcing/)：处理真实前缀、free-running 和 exposure bias。
- [为什么需要 attention](../rnn-lstm/why-attention/)：解释 fixed context 瓶颈与可寻址源状态。
- [交叉注意力](../attention/cross-attention/)：展开目标 query、源 key/value 和 $U\times S$ 读取矩阵。
- [因果掩码](../attention/causal-masking/)：处理目标下三角、padding、packed sequence 和 cache offset。
- [Self-Attention](../attention/self-attention/)、[Multi-Head Attention](../attention/multi-head-attention/)：展开 Q/K/V、softmax 与多头投影。
- [完整 Transformer](../transformer-architectures/full-transformer/)：把 encoder、decoder、cross-attention 和 logits 组装成完整计算链。
- [Encoder-Only](../transformer-architectures/encoder-only/)、[Decoder-Only](../transformer-architectures/decoder-only/)：比较删除一侧后的可见性、输出与推理协议。
- [LayerNorm 与残差流](../transformer-components/layernorm-residuals/)、[Pre-Norm 与 Post-Norm](../transformer-components/pre-norm-vs-post-norm/)：处理 block 顺序和归一化路径。
- [前馈网络](../transformer-components/feedforward/)：处理每个 token 独立的特征变换支路。
- [参数量总账](../transformer-components/parameter-count/)：核对参数、MAC、激活和 KV cache。
