---
title: "Decoder-Only：因果自回归 Transformer"
tags: ["why-models-learn"]
---

Decoder-only Transformer 只保留 decoder stack，并删除原始 encoder-decoder 结构中的 encoder memory 和 cross-attention。它把输入前缀放入带 causal mask 的 self-attention：位置 $t$ 可以读取当前及过去位置，不能读取未来位置。最终 hidden 经过词表输出头得到 next-token logits；推理时每生成一个 token，就把它追加到下一步的前缀。

本文固定 decoder-only 的 $(B,T,D)$ shape 合同，先写右移输入、位置条件和 causal mask，再展开 pre-norm decoder block、LM head、权重绑定和训练时的并行前向。随后用一个小配置核对因果 attention、参数量和 MAC，最后把 prefill、单步 decode、KV cache、GQA、padding 和常见边界放在同一份审计协议中。完整的 next-token 损失与数据目标留给 [Causal Language Modeling](../transformer-architectures/causal-language-modeling/)。

![Decoder-only Transformer 的数据流：右移前缀进入带 causal mask 的 decoder stack，历史 K/V 写入 cache，最后一个 hidden 经过 LM head 产生下一个 token logits](/assets/transformer-architectures/svg/decoder-only.1.svg)

## 先固定 decoder-only 的接口

### 一条序列和一条 hidden 流

设 batch size 为 $B$，输入前缀长度为 $T$，词表大小为 $V$，模型宽度为 $D=d_{\mathrm{model}}$。token id 和 hidden 的 shape 为

$$
I\in\{0,\ldots,V-1\}^{B\times T},
\qquad
H_0\in\mathbb R^{B\times T\times D}.
$$

经过 $L$ 个 decoder block 后得到

$$
H_L\in\mathbb R^{B\times T\times D}.
$$

decoder-only 没有第二条 encoder 序列轴。self-attention 的 score 是 $(B,h,T,T)$；cross-attention 的 $(B,h,U,S)$ 在这个架构中不存在。

| 对象 | shape | 作用 | 位置轴 |
| --- | --- | --- | --- |
| token id | $(B,T)$ | 前缀的离散索引 | 有 |
| decoder hidden | $(B,T,D)$ | 受因果约束的上下文表示 | 有 |
| LM logits | $(B,T,V)$ | 每个位置对下一个 token 的分数 | 有 |
| next-token logits | $(B,V)$ | 当前前缀末端的下一 token 分数 | 无 |
| KV cache | $(B,h_{kv},T,d_h)$ | 保存历史 key/value | 有 |

训练时通常保留整条序列的 logits，推理时通常只读取最后一个位置的 logits。两者共享同一个 decoder stack，不共享同一个计算量。

### decoder-only 不读取未来输入

对目标 token 序列 $y_1,\ldots,y_T$，训练输入是右移后的前缀：

$$
x_1=\langle\mathrm{bos}\rangle,
\qquad
x_t=y_{t-1}\quad (t=2,\ldots,T).
$$

第 $t$ 个位置的输出负责预测 $y_t$。输入中没有当前位置的 $y_t$，否则模型可以从 token embedding 直接复制标签。

这条对齐合同与 causal mask 一起工作：右移阻止标签从输入端直接出现，causal mask 阻止当前位置从 attention 读取未来输入。两者有不同的职责，不能用其中一个代替另一个。

## 输入端：token embedding 和位置

### embedding 查表

词表 embedding 矩阵为

$$
E\in\mathbb R^{V\times D}.
$$

对输入 id 做行查表：

$$
X_{0,b,t,:}=E_{I_{b,t},:}.
$$

id 的整数大小没有距离含义。模型需要从 embedding 和后续层学习 token 的可用特征。

### 位置进入 Q/K 或 hidden

使用可加位置表时

$$
H_0=X_0+P_{0:T,:},
\qquad
P\in\mathbb R^{T_{\max}\times D}.
$$

decoder-only 常见的位置路径有：

| 方案 | 作用位置 | decode 时要记录 |
| --- | --- | --- |
| learned absolute | token hidden | 当前 cache 长度对应的位置表行 |
| sinusoidal | token hidden | 位置起点与频率 |
| RoPE | query/key | prefill 和 decode 的位置 offset |
| ALiBi | attention score | query/key 的真实距离 |

位置索引从 0 还是 1 开始、prompt 续写时是否从已有长度继续、packed sequence 是否在每段重置，都必须固定。[RoPE](../transformer-components/rope/)把位置作用到 Q/K，[ALiBi](../transformer-components/alibi/)把距离作用到 score。

## causal self-attention：把可读区域限制为前缀

### Q、K、V 的 shape

对某个 decoder block 的归一化输入 $X\in\mathbb R^{B\times T\times D}$，将宽度拆成 $h$ 个 head：

$$
d_h=\frac{D}{h}.
$$

第 $r$ 个 head 的 projection 为

$$
\begin{aligned}
Q_r&=XW^Q_r,\\
K_r&=XW^K_r,\\
V_r&=XW^V_r,
\end{aligned}
\qquad
Q_r,K_r,V_r\in\mathbb R^{B\times T\times d_h}.
$$

在训练的整段前向中，三个张量都可以一次算出；mask 决定每一行可以读取哪些列。[Self-Attention](../attention/self-attention/)的 Q/K/V 语义不变，因果性来自可读集合。

### causal mask 的下三角

令 $i$ 是 query 位置，$j$ 是 key 位置。causal mask 为

$$
M^{\mathrm{causal}}_{i,j}
=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

score 和读取结果为

$$
S_r=\frac{Q_rK_r^{\mathsf T}}{\sqrt{d_h}}+M^{\mathrm{causal}}+M^{\mathrm{pad}},
\qquad
A_r=\operatorname{softmax}_{\mathrm{key}}(S_r),
\qquad
O_r=A_rV_r.
$$

对长度为 4 的无 padding 序列，可读区域是：

| query 位置 | 可读 key | 被排除的 key |
| --- | --- | --- |
| $1$ | $1$ | $2,3,4$ |
| $2$ | $1,2$ | $3,4$ |
| $3$ | $1,2,3$ | $4$ |
| $4$ | $1,2,3,4$ | 无 |

位置 $i$ 可以读取自己，这是 inclusive causal mask。若实现使用 exclusive mask，第一位置没有可读 value，需要额外的 BOS 或特殊处理。

### padding 和 causal mask 要相交

变长 batch 中，causal mask 只表达时间顺序，padding mask 表达有效性：

$$
M^{\mathrm{pad}}_{b,i,j}
=
\begin{cases}
0,&m_{b,j}=1\\
-\infty,&m_{b,j}=0
\end{cases}
$$

实际可读集合是 causal 允许的列与有效 key 的交集。padding query 还要从 loss、输出和 cache 写入中排除；只给 key 加 mask 不会自动处理 padding query。

### 多头输出和残差接口

各 head 拼接并投影回 $D$ 维：

$$
\operatorname{MHA}_{\mathrm{causal}}(X)
=\operatorname{Concat}(O_1,\ldots,O_h)W^O.
$$

输出必须是 $(B,T,D)$，才能写回 residual stream。query head 数、K/V head 数和 cache 的 layout 可以由 MHA、GQA 或 MQA 决定，但 residual 宽度仍然是 $D$。[GQA 与 MQA](../attention/gqa-and-mqa/)展开共享 K/V 的参数和 cache 变化。

## 一个 pre-norm decoder block

### masked self-attention 和 FFN

设第 $\ell$ 层输入为 $H_{\ell-1}$，两个归一化模块为 $N_{\ell,1}$ 和 $N_{\ell,2}$。常见 pre-norm 结构为

$$
\begin{aligned}
R_\ell
&=H_{\ell-1}
 +\operatorname{MHA}^{\mathrm{causal}}_\ell
   \left(N_{\ell,1}(H_{\ell-1});M^{\mathrm{causal}},M^{\mathrm{pad}}\right),\\
H_\ell
&=R_\ell
 +\operatorname{FFN}_\ell\left(N_{\ell,2}(R_\ell)\right).
\end{aligned}
$$

两条支路的 shape 都是 $(B,T,D)$。attention 在 token 轴上读取历史，FFN 只在特征轴上计算。[前馈网络](../transformer-components/feedforward/)和[SwiGLU](../transformer-components/swiglu-ffn/)处理第二条支路的具体非线性。

post-norm 改成

$$
\begin{aligned}
R_\ell&=N_{\ell,1}\left(H_{\ell-1}+\operatorname{MHA}^{\mathrm{causal}}_\ell(H_{\ell-1})\right),\\
H_\ell&=N_{\ell,2}\left(R_\ell+\operatorname{FFN}_\ell(R_\ell)\right).
\end{aligned}
$$

这会同时改变前向函数和梯度 Jacobian。加载 decoder-only checkpoint 时，必须记录 norm 的顺序。

### final norm 和 LM head

pre-norm decoder stack 常在最后一层之后添加 final norm：

$$
\widetilde H_L=N_{\mathrm{final}}(H_L).
$$

输出头把最后一维映射到词表：

$$
Z=\widetilde H_LW_{\mathrm{out}}^{\mathsf T}+b_{\mathrm{out}},
\qquad
Z\in\mathbb R^{B\times T\times V}.
$$

推理时只读取最后一个位置：

$$
z_{\mathrm{next}}=Z_{:,T,:}\in\mathbb R^{B\times V}.
$$

如果使用 tied weights，通常令 $W_{\mathrm{out}}=E$。输入 embedding 与输出类别投影共享同一组参数，参数账不重复计数；梯度则累加输入和输出两条路径。

## 训练时：整段并行，信息仍然因果

### 并行不等于读取未来

训练可以把右移后的整段输入一次送入 decoder，得到 $(B,T,V)$ 的 logits。causal mask 在 score 上将未来列设为不可读，因此第 $t$ 行仍然只依赖 $x_1,\ldots,x_t$。

用矩阵并行计算只改变执行顺序，不改变依赖图：

| 阶段 | 计算方式 | 可读范围 | 输出 |
| --- | --- | --- | --- |
| training forward | 一次处理整段右移输入 | 每行的下三角前缀 | $(B,T,V)$ logits |
| prefill | 一次处理 prompt | prompt 内下三角前缀 | prompt logits / cache |
| decode | 每次处理新 token | 新 query 读取全部历史 cache | $(B,V)$ logits |

next-token 损失的标签、分母、ignore mask 和数据切分会影响训练目标，专篇见 [Causal Language Modeling](../transformer-architectures/causal-language-modeling/)。

### logits 与 tied embedding

若 $W_{\mathrm{out}}=E$ 且没有 output bias，第 $t$ 个 hidden 对词表第 $v$ 个 token 的 logit 是

$$
Z_{b,t,v}
=\widetilde H_{L,b,t,:}E_{v,:}^{\mathsf T}.
$$

同一 embedding 行有两个用途：作为 token 输入向量，作为一个候选输出 token 的分类向量。实现若只复制权重数值而没有共享参数对象，前向可能相同，更新路径却不同。

## 一个三位置的因果读取例子

### 同一组 Q/K/V 在不同 query 位置的结果

取 $d_h=2$，一个 query 和三个 key/value 为

$$
q=(1,0),
\qquad
k_1=(1,0),
\qquad
k_2=(0,1),
\qquad
k_3=(1,0),
$$

$$
v_1=(1,0),
\qquad
v_2=(0,2),
\qquad
v_3=(3,0).
$$

对第 3 个 query，三个 score 为

$$
s_1=\frac{1}{\sqrt 2}=0.707106781187,
\qquad
s_2=0,
\qquad
s_3=0.707106781187.
$$

它可以读取全部三个位置，因此权重和读取结果为

$$
a=(0.401112092680,0.197775814640,0.401112092680),
$$

$$
o=a_1v_1+a_2v_2+a_3v_3
=(1.604448370719,0.395551629281).
$$

对第 1 个 query，causal mask 只保留第 1 列，权重是 $(1,0,0)$，读取结果是 $v_1=(1,0)$。同一组投影值在不同 query 位置产生不同上下文，因为可读集合不同。

### 右移输入防止标签直达

设真值 token 是 $(y_1,y_2,y_3)$，decoder 输入为 $(\mathrm{BOS},y_1,y_2)$。第 3 个位置可以读取 BOS、$y_1$ 和 $y_2$，目标是 $y_3$。如果输入改成 $(y_1,y_2,y_3)$，第 3 个 hidden 可以通过自己的 embedding 直接看到 $y_3$，即使 causal mask 正确，训练也已经泄漏。

## 参数量和计算量账本

### 小配置

取一个便于手算的 decoder-only 配置：

$$
B=1,
\qquad
T=4,
\qquad
V=10,
\qquad
D=4,
\qquad
h=2,
\qquad
d_h=2,
\qquad
M=8,
\qquad
L=2.
$$

假定线性层没有 bias，每个 block 有两个 LayerNorm，stack 末尾有一个 final LayerNorm，使用长度为 4 的 learned position table，LM head 与 embedding 绑定。一个 decoder layer 的参数量为

$$
P_{\mathrm{layer}}
=4D^2+2DM+4D
=64+64+16
=144.
$$

两层 stack、embedding、位置表和 final LayerNorm 的参数量为

$$
\begin{aligned}
P_{\mathrm{total}}
&=LP_{\mathrm{layer}}+VD+TD+2D\\
&=288+40+16+8\\
&=352.
\end{aligned}
$$

若 LM head 不绑定 embedding，需要再增加 $VD=40$，总量变为 392。若 final norm 使用 RMSNorm 而不是带 scale/bias 的 LayerNorm，final norm 的参数项会从 $2D=8$ 变为 $D=4$。

### training 或 prefill 的主 MAC

只统计矩阵乘法以及 QK/AV，不统计 norm、softmax、激活和残差加法。一层、长度 $T=4$ 的主项是

$$
\begin{aligned}
\operatorname{MHA\ projection}&=4TD^2=256,\\
\operatorname{causal\ QK+AV}&=2hT^2d_h=128,\\
\operatorname{FFN}&=2TDM=256.
\end{aligned}
$$

合计为 $640$ MAC，两层 stack 为 $1280$ MAC。若 prefill 还计算完整 prompt 的 logits，绑定 LM head 的输出投影增加 $TDV=160$ MAC；这项来自词表宽度 $V$，不能混进 decoder layer 的参数账。

### 单步 decode 的主 MAC

假定当前 cache 已保存 $t=3$ 个历史位置，新 token 是第 4 个位置。只对新 token 做 Q/K/V/O projection：

$$
\begin{aligned}
\operatorname{new\ self\ projection}&=4D^2=64,\\
\operatorname{QK+AV\ with\ cache}&=2h(t+1)d_h=32,\\
\operatorname{FFN}&=2DM=64,\\
\operatorname{LM\ head}&=DV=40.
\end{aligned}
$$

单层合计为 $200$ MAC。两层 stack 为 $400$ MAC，忽略 norm、softmax 和 cache 追加。若没有 cache，历史三个位置的 K/V projection 也会在每一步重复计算。

### 参数、序列 MAC 和单步 MAC

| 账本项 | 是否随序列长度变化 | 小配置数值 |
| --- | --- | ---: |
| tied decoder layer 参数 | 否 | 144 / layer |
| token embedding | 否 | 40 |
| position table | 随最大长度变化 | 16 |
| training self-attention 交互 | 是 | 128 / layer |
| prefill LM projection | 是 | 160 / sequence |
| decode self-attention 交互 | 随 cache 长度变化 | 32 / step / layer |
| KV cache 元素 | 随历史长度变化 | 2 个张量 |

参数量只记录可学习张量。MAC 记录算术工作量。KV cache 记录运行时状态；三者不能用一个数字替代。

## prefill、decode 和 KV cache

### prefill 生成初始 cache

给定长度为 $T_0$ 的 prompt，prefill 一次计算完整 causal attention，并在每个 decoder layer 保存历史 key/value：

$$
K^{\mathrm{cache}}_\ell,V^{\mathrm{cache}}_\ell
\in\mathbb R^{B\times h_{kv}\times T_0\times d_h}.
$$

若使用标准 MHA，$h_{kv}=h$。若使用 GQA/MQA，K/V head 更少，query head 仍然可以保持 $h_q$ 个。

prefill 计算 prompt 内所有位置的 hidden 和 logits；只需要最后位置时，前面的 logits 可以丢弃，但前面的 K/V 仍然要写入 cache。

### decode 只计算一个新 query

当前 cache 长度为 $t$ 时，新 token 的 query shape 为

$$
Q_{\mathrm{new}}\in\mathbb R^{B\times h_q\times 1\times d_h}.
$$

历史和新位置拼接后的 K/V shape 为

$$
K_{\mathrm{all}},V_{\mathrm{all}}
\in\mathbb R^{B\times h_{kv}\times (t+1)\times d_h}.
$$

score 的逻辑 shape 是 $(B,h_q,1,t+1)$。新 query 可以读取所有历史和自己，不能读取尚未生成的位置。

### cache 的字节数

若 K/V 使用同一 dtype，每个元素占 $s$ bytes，$L$ 层、batch 为 $B$、历史长度为 $t$，缓存元素和字节数为

$$
N_{\mathrm{KV}}
=2LBh_{kv}td_h,
\qquad
\mathrm{bytes}_{\mathrm{KV}}=sN_{\mathrm{KV}}.
$$

对小配置 $L=2,B=1,h_{kv}=2,t=4,d_h=2$，FP16 的 cache 字节数为

$$
2\times2\times1\times2\times4\times2\times2=128\ \mathrm{bytes}.
$$

这个数值只包含 self-attention 的 K/V，不包含 hidden、allocator 对齐、beam 维度或 cross-attention 的静态源 cache。

### 位置 offset 必须与 cache 对齐

prefill 处理位置 $0,\ldots,T_0-1$ 后，下一次 decode 的新位置是 $T_0$。RoPE、ALiBi 和 learned position table 都要使用这个 offset。若新 token 重新使用位置 0，attention 仍可以运行，但位置条件已经错位。

## GQA/MQA：减少 cache，不改变因果图

decoder-only 可以使用 $h_q$ 个 query head 和 $h_{kv}$ 个 K/V head。query head 按组共享 K/V：

$$
r=\frac{h_q}{h_{kv}}.
$$

组内 query 仍然有独立的 score 和 attention weight。改变的主要是 K/V projection 参数、K/V cache 和读取时的广播或索引。

| 配置 | query head | K/V head | cache 相对 MHA |
| --- | ---: | ---: | ---: |
| MHA | $h_q$ | $h_q$ | 1 |
| GQA | $h_q$ | $h_{kv}$ | $h_{kv}/h_q$ |
| MQA | $h_q$ | $1$ | $1/h_q$ |

causal mask 仍然按 query 位置和 key 位置应用。GQA 不会把 decoder-only 变成双向，也不会减少逻辑 query-key 交互的 query head 数。

## 失效模式：因果 decoder 的边界错误

### mask 用成双向 attention

现象是第 1 行的 attention weight 在未来列非零，训练 loss 看起来过低，模型生成时表现明显下降。用长度 4 的全一 score 检查第一行、第二行和最后一行的非零列集合。

### 右移漏掉一位

现象是输入和标签完全相同，模型在训练中可以直接复制当前 token。逐位置打印 BOS、输入 id、标签 id，检查输入位置 $t$ 是否是标签位置 $t-1$ 的 token。

### decode 读取了错误 cache

常见错误包括把新 K/V 追加到错误层、沿 batch 轴而不是 sequence 轴拼接、beam reorder 后没有同步重排 cache、GQA cache 被错误复制到 query head。检查每层 cache 的轴和长度增长。

### prefill 与 decode 路径不一致

同一 prompt 用一次 prefill 和逐 token decode 得到的 next-token logits 应在明确的数值容差内一致。若不一致，优先检查 causal mask、position offset、RoPE 是否重复旋转以及 cache 中的 K/V 是否已经旋转。

### 只缓存 K/V，不记录长度

cache tensor 的 shape 可能包含预分配容量，真实有效长度仍需单独记录。把未写入的容量当作 key 会读取未初始化值或 padding 值。

### tied head 只复制权重

输入 embedding 与 LM head 复制同一初值不等于权重绑定。检查参数身份、optimizer 参数列表和反向梯度是否合并到同一张量。

### padding 进入生成位置

批量推理中，不同样本的有效末位置可能不同。直接取每行最后一个矩形位置会读取 padding hidden；应使用每个样本的有效长度索引。

### 把 decoder-only 当作 encoder-only

decoder-only 的 hidden 依赖当前及过去前缀；encoder-only 的 hidden 可以读取前后两侧有效 token。复制 block 代码时只改变类名或 position encoding，不会自动改变 mask 和推理协议。

## 可复用的核验协议

审计一个 decoder-only 模型时，按以下顺序记录：

1. 写出 $B,T,V,D,h,h_{kv},d_h,M,L$ 和位置方案；
2. 打印右移后的输入、标签、BOS/EOS/PAD id 和有效长度；
3. 对训练前向检查每层 score 是 $(B,h_q,T,T)$，对 decode 检查是 $(B,h_q,1,t+1)$；
4. 用长度 4 的手算 score 检查 causal mask 的下三角区域；
5. 独立计算 embedding、position table、每层 QKVO/FFN/norm、final norm 和 LM head 参数量；
6. 分开计算 training/prefill MAC、decode 单步 MAC、activation elements 和 KV cache bytes；
7. 比较 prefill 与逐 token decode 的 logits，明确 dtype 和容差；
8. 用 MHA、GQA、MQA 三个 head 配置检查 cache 轴和容量比例；
9. 检查 tied embedding 与 LM head 的参数身份和梯度累加；
10. 用不同 prompt 长度、padding、cache 续写和 beam reorder 做回归。

第 4 步和第 7 步必须先于 kernel 融合和量化。FlashAttention、paged KV cache 和 fused FFN 可以改变访存与实现路径，但不能改变 causal 可读集合、position offset 和 $(B,T,D)$ 的 residual 合同。[因果掩码](../attention/causal-masking/)处理 mask 方向和 cache offset，[参数量总账](../transformer-components/parameter-count/)处理参数、MAC、激活和状态账本。

## 相关词条

- [完整 Transformer](../transformer-architectures/full-transformer/)：把 encoder、decoder、cross-attention 和 logits 组装成完整计算链。
- [Encoder-Only](../transformer-architectures/encoder-only/)：比较双向 self-attention 与因果 self-attention。
- [因果掩码](../attention/causal-masking/)：推导下三角可读区域、padding 和 cache offset。
- [Self-Attention](../attention/self-attention/)、[Multi-Head Attention](../attention/multi-head-attention/)：展开 Q/K/V、softmax 和多头输出。
- [GQA 与 MQA](../attention/gqa-and-mqa/)：处理 K/V head 共享和 KV cache 容量。
- [RoPE](../transformer-components/rope/)、[ALiBi](../transformer-components/alibi/)：处理 decoder-only 的位置路径和 decode offset。
- [Pre-Norm 与 Post-Norm](../transformer-components/pre-norm-vs-post-norm/)、[残差流](../transformer-components/residual-streams/)：处理 block 顺序和主表示通道。
- [前馈网络](../transformer-components/feedforward/)、[SwiGLU](../transformer-components/swiglu-ffn/)：处理 decoder block 的 token-wise 非线性支路。
- [Causal Language Modeling](../transformer-architectures/causal-language-modeling/)：处理 next-token 目标、标签对齐和损失口径。
- [参数量总账](../transformer-components/parameter-count/)：分开核算参数、计算、激活和运行时状态。
