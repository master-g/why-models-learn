---
title: "完整 Transformer：从 token 到 logits"
tags: ["why-models-learn"]
---

完整 Transformer 是一条把离散 token 序列变成条件概率的计算链。原始 Transformer 同时包含编码器和解码器：编码器读取源序列，解码器在因果掩码下读取已经生成的目标前缀，并通过 cross-attention 读取编码器表示，最后把每个目标位置映射成词表上的 logits。现代模型常常只保留其中一侧，但仍然沿用这条计算链中的 embedding、位置条件、attention、残差、归一化、FFN 和输出头。

本文固定 batch、序列轴、hidden width、head 数和 mask 的合同，先从 token 进入模型写出完整的 encoder-decoder 前向，再分别展开 encoder self-attention、decoder masked self-attention 和 decoder cross-attention。随后用一个小配置核对 shape、参数量和一次解码步的 MAC，最后把 teacher forcing、KV cache、权重绑定和常见架构删减放进同一份审计表。

![完整 Transformer 的数据流：源 token 经过 embedding 和编码器，目标前缀经过 masked decoder，并通过 cross-attention 读取编码器表示，最后产生 logits](/assets/transformer-architectures/svg/full-transformer.1.svg)

## 先固定整条链的接口

### 轴和宽度

设源序列长度为 $S$，目标输入长度为 $U$，词表大小为 $V$，模型宽度为 $D=d_{\mathrm{model}}$。编码器和解码器都把每个 token 表示成 $D$ 维向量。若每个 attention 有 $h$ 个 head，单个 head 的宽度为

$$
d_h=\frac{D}{h}.
$$

这里假定 $h$ 整除 $D$。GQA 和 MQA 会把 K/V head 的数量改成 $h_{kv}$，但不改变 query 的主轴；同一问题见 [GQA 与 MQA](../attention/gqa-and-mqa/)。

四个批量张量的 shape 如下：

| 张量 | shape | 轴含义 | 所在阶段 |
| --- | --- | --- | --- |
| 源 token id | $(B,S)$ | batch、源位置 | 输入 |
| 目标输入 id | $(B,U)$ | batch、目标位置 | decoder 输入 |
| encoder hidden | $(B,S,D)$ | batch、源位置、特征 | encoder 输出 |
| decoder hidden | $(B,U,D)$ | batch、目标位置、特征 | decoder 输出 |

$B$ 是 batch size。$S$ 和 $U$ 可以不同；把两个长度都写成 $T$ 会掩盖 cross-attention 的两个位置轴。

### 离散 id 与连续表示

源 token id 和目标 token id 属于离散集合：

$$
I^{\mathrm{src}}\in\{0,\ldots,V-1\}^{B\times S},
\qquad
I^{\mathrm{tgt}}\in\{0,\ldots,V-1\}^{B\times U}.
$$

embedding 矩阵为 $E\in\mathbb R^{V\times D}$。查表不是把 id 当作连续数值相乘，而是按最后一个 id 轴取出对应行：

$$
H^{\mathrm{src}}_{0,b,s,:}=E_{I^{\mathrm{src}}_{b,s},:},
\qquad
H^{\mathrm{tgt}}_{0,b,u,:}=E_{I^{\mathrm{tgt}}_{b,u},:}.
$$

tokenization 决定了 $S$ 和 $U$ 具体包含多少个 token。embedding 只负责把每个 id 变成向量；它本身不告诉 attention 两个向量的先后顺序。

## 输入端：embedding、位置和目标移位

### 位置条件进入 hidden

如果使用可加的位置向量 $P$，输入 hidden 是

$$
H^{\mathrm{src}}_0=E[I^{\mathrm{src}}]+P^{\mathrm{src}},
\qquad
H^{\mathrm{tgt}}_0=E[I^{\mathrm{tgt}}]+P^{\mathrm{tgt}}.
$$

$P$ 可以是固定正弦表、可学习位置表，也可以不作为显式加法出现。RoPE 把位置作用到 Q/K，ALiBi 把位置差作用到 score；这些接口都必须保持后续 attention 的 shape 不变。[位置编码](../transformer-components/positional-encoding/)、[RoPE](../transformer-components/rope/) 和 [ALiBi](../transformer-components/alibi/)分别处理这三类路径。

位置向量的索引合同至少包括四项：

| 检查对象 | 合同 | 错误表现 |
| --- | --- | --- |
| 起始位置 | 通常从 0 开始 | 所有相位或位置表整体偏移 |
| padding | 不产生有效内容 | padding 被当成可读 token |
| packed sequence | 每段重新设定边界 | 一段读取下一段 |
| decode cache | 新 token 使用继续增长的位置 | 生成阶段位置重复 |

### decoder 的输入向右移一位

训练时，目标真值序列可以写成

$$
y_1,y_2,\ldots,y_U.
$$

decoder 的输入应该是带起始符的右移序列：

$$
x^{\mathrm{dec}}_1=\langle\mathrm{bos}\rangle,
\qquad
x^{\mathrm{dec}}_{u}=y_{u-1}\quad (u=2,\ldots,U).
$$

第 $u$ 个 decoder 位置只负责预测 $y_u$。因此输入和标签对齐为：

| decoder 位置 | 输入 token | 监督标签 | 可读取的目标 token |
| --- | --- | --- | --- |
| $1$ | BOS | $y_1$ | BOS |
| $2$ | $y_1$ | $y_2$ | BOS、$y_1$ |
| $u$ | $y_{u-1}$ | $y_u$ | $x^{\mathrm{dec}}_1,\ldots,x^{\mathrm{dec}}_u$ |

如果把 $y_u$ 本身放进第 $u$ 个输入，因果 mask 只能限制位置，不能修复标签已经泄漏到输入的问题。teacher forcing 的含义是使用真值前缀作为输入，而不是把当前标签提前给当前位置。

## encoder：读取完整源序列

### 一个 pre-norm encoder block

下面先写常见的 pre-norm 形式。设第 $\ell$ 层输入为 $H^{\mathrm e}_{\ell-1}$，$N^{\mathrm e}_{\ell,1}$ 和 $N^{\mathrm e}_{\ell,2}$ 是两个归一化模块：

$$
\begin{aligned}
R^{\mathrm e}_\ell
&=H^{\mathrm e}_{\ell-1}
  +\operatorname{MHA}^{\mathrm e}_\ell
    \left(N^{\mathrm e}_{\ell,1}(H^{\mathrm e}_{\ell-1});M^{\mathrm e}\right),\\
H^{\mathrm e}_\ell
&=R^{\mathrm e}_\ell
  +\operatorname{FFN}^{\mathrm e}_\ell
    \left(N^{\mathrm e}_{\ell,2}(R^{\mathrm e}_\ell)\right).
\end{aligned}
$$

每个加法都要求两侧 shape 相同，仍为 $(B,S,D)$。第一条支路让每个源位置读取其他源位置，第二条支路在每个位置独立地做非线性特征变换。[Pre-Norm 与 Post-Norm](../transformer-components/pre-norm-vs-post-norm/)讨论归一化放置对 Jacobian 的影响；这里先固定 block 的数据流。

### encoder self-attention

对输入 $X=N^{\mathrm e}_{\ell,1}(H^{\mathrm e}_{\ell-1})$，第 $r$ 个 head 的投影为

$$
\begin{aligned}
Q_r&=XW^Q_r,\\
K_r&=XW^K_r,\\
V_r&=XW^V_r,
\end{aligned}
\qquad
Q_r,K_r,V_r\in\mathbb R^{B\times S\times d_h}.
$$

attention score 和输出为

$$
A_r=\operatorname{softmax}_{\mathrm{key}}
\left(
\frac{Q_rK_r^{\mathsf T}}{\sqrt{d_h}}+M^{\mathrm e}
\right),
\qquad
O_r=A_rV_r.
$$

$A_r$ 的 shape 是 $(B,S,S)$。第一个 $S$ 是 query 位置，第二个 $S$ 是 key 位置。encoder self-attention 通常允许非 padding 源位置互相读取，因此有效区域近似是整张 $S\times S$ 矩阵；padding mask 仍然需要把无效 key 的 score 置为 $-\infty$。

各 head 拼接后再经过输出投影：

$$
\operatorname{MHA}^{\mathrm e}(X;M^{\mathrm e})
=\operatorname{Concat}(O_1,\ldots,O_h)W^O.
$$

[Self-Attention](../attention/self-attention/)和[Multi-Head Attention](../attention/multi-head-attention/)分别展开单头读取和多头拼接。这里要保留一个接口事实：attention 改变 token 之间的信息混合，FFN 不改变 token 轴上的位置数量。

### encoder 的 mask

对源 padding mask，令 $m^{\mathrm{src}}_{b,j}=1$ 表示位置 $j$ 有效，$0$ 表示 padding。可以把它广播到 query 位置：

$$
M^{\mathrm e}_{b,i,j}
=
\begin{cases}
0,&m^{\mathrm{src}}_{b,j}=1\\
-\infty,&m^{\mathrm{src}}_{b,j}=0
\end{cases}
$$

softmax 后 padding key 的权重应接近 0。query 位置本身如果是 padding，还需要在残差、FFN 和损失端继续保持无效，不能只依靠 key mask。

## decoder：先读前缀，再读 encoder

### 三个子层的顺序

decoder block 比 encoder block 多一个 cross-attention。对第 $\ell$ 层，先做目标端 masked self-attention：

$$
R^{\mathrm d}_\ell
=H^{\mathrm d}_{\ell-1}
 +\operatorname{MHA}^{\mathrm{mask}}_\ell
   \left(N^{\mathrm d}_{\ell,1}(H^{\mathrm d}_{\ell-1});M^{\mathrm{causal}}\right).
$$

再用 decoder 的 query 读取 encoder 的 key/value：

$$
\begin{aligned}
C^{\mathrm d}_\ell
&=R^{\mathrm d}_\ell
 +\operatorname{MHA}^{\mathrm{cross}}_\ell
   \left(N^{\mathrm d}_{\ell,2}(R^{\mathrm d}_\ell),
         H^{\mathrm e}_{L_{\mathrm e}};M^{\mathrm{src}}\right),\\
H^{\mathrm d}_\ell
&=C^{\mathrm d}_\ell
 +\operatorname{FFN}^{\mathrm d}_\ell
   \left(N^{\mathrm d}_{\ell,3}(C^{\mathrm d}_\ell)\right).
\end{aligned}
$$

三条残差支路都保持 $(B,U,D)$。cross-attention 的 key/value 来自 $(B,S,D)$ 的 encoder 输出，因此它的 score 矩阵 shape 是 $(B,U,S)$，不是 $(B,U,U)$。

### decoder masked self-attention

目标端的 causal mask 规定 query 位置 $i$ 不能读取未来 key 位置 $j>i$：

$$
M^{\mathrm{causal}}_{i,j}
=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

在 batch、head 和 padding 条件下，实际 mask 是这张三角矩阵与有效 token mask 的组合。对 $U=4$，可读区域为：

| query 位置 | 可读目标 key |
| --- | --- |
| $1$ | $1$ |
| $2$ | $1,2$ |
| $3$ | $1,2,3$ |
| $4$ | $1,2,3,4$ |

训练时可以一次计算整张下三角矩阵。并行计算不代表信息泄漏，因为每个未来 score 已经被 mask 去掉；如果实现先 softmax 再 mask，数值路径就不符合这个合同。

### cross-attention

对 $X=N^{\mathrm d}_{\ell,2}(R^{\mathrm d}_\ell)$ 和 $Y=H^{\mathrm e}_{L_{\mathrm e}}$，第 $r$ 个 cross-attention head 使用

$$
\begin{aligned}
Q_r&=XW^{Q,\mathrm{cross}}_r,\\
K_r&=YW^{K,\mathrm{cross}}_r,\\
V_r&=YW^{V,\mathrm{cross}}_r.
\end{aligned}
$$

此时

$$
\begin{aligned}
Q_r&\in\mathbb R^{B\times U\times d_h},\\
K_r,V_r&\in\mathbb R^{B\times S\times d_h},\\
A_r^{\mathrm{cross}}&\in\mathbb R^{B\times U\times S},\\
O_r&=A_r^{\mathrm{cross}}V_r\in\mathbb R^{B\times U\times d_h}.
\end{aligned}
$$

decoder 的每个位置都可以读取源序列中允许的 key。若源序列有 padding，$M^{\mathrm{src}}$ 应作用在 $S$ 这一列轴上；目标 causal mask 不应直接套到 cross-attention，因为 $S$ 与 $U$ 通常不同。

## 从输入到 logits 的完整前向

### encoder 端

设编码器层数为 $L_{\mathrm e}$。输入端得到 $H^{\mathrm e}_0$ 后，按层串行计算：

$$
H^{\mathrm e}_{L_{\mathrm e}}
=\operatorname{Encoder}_{L_{\mathrm e}}
\circ\cdots\circ
\operatorname{Encoder}_2
\circ\operatorname{Encoder}_1
(H^{\mathrm e}_0;M^{\mathrm e}).
$$

每层共享 shape 合同，但不共享参数。参数共享时，层索引会被移除，参数账不能按层数重复相加。

### decoder 端

给定右移后的目标输入，解码器得到 $H^{\mathrm d}_0$，再按层读取固定的 encoder memory：

$$
H^{\mathrm d}_{L_{\mathrm d}}
=\operatorname{Decoder}_{L_{\mathrm d}}
\circ\cdots\circ
\operatorname{Decoder}_1
(H^{\mathrm d}_0,H^{\mathrm e}_{L_{\mathrm e}};M^{\mathrm{causal}},M^{\mathrm{src}}).
$$

decoder 每层的 masked self-attention 只能读取目标前缀，cross-attention 读取完整源 memory。两种读取发生在同一个 decoder block 内，但它们的 key/value 来源和 mask 不同。

### 输出头

最终 decoder hidden 为 $H^{\mathrm d}_{L_{\mathrm d}}\in\mathbb R^{B\times U\times D}$。输出投影矩阵 $W_{\mathrm{out}}\in\mathbb R^{V\times D}$ 产生每个位置的 logits：

$$
Z=H^{\mathrm d}_{L_{\mathrm d}}W_{\mathrm{out}}^{\mathsf T}+b_{\mathrm{out}},
\qquad
Z\in\mathbb R^{B\times U\times V}.
$$

对一个有效位置，概率分布是

$$
p_{b,u,v}=\operatorname{softmax}_v(Z_{b,u,:})_v.
$$

如果把 embedding 和输出头绑定，且 embedding 行向量与输出类别使用同一词表，通常取 $W_{\mathrm{out}}=E$。这样输出头不新增 $VD$ 个参数，但 logits 的数值仍然通过转置后的 embedding 行产生。

### teacher forcing 的损失

设目标标签为 $y_{b,u}$，有效位置 mask 为 $m_{b,u}$。平均交叉熵应明确分母：

$$
\mathcal L
=-
\frac{1}{\sum_{b,u}m_{b,u}}
\sum_{b,u}m_{b,u}
\log p_{b,u,y_{b,u}}.
$$

padding 位置不能进入分子，也不能进入有效 token 的分母。若训练目标是 sum loss，分母可以省略，但不同 batch 的数值就不再代表同一平均口径。

## 一个小的数值前向

### 先只核对 token 和 shape

用一个不包含 bias 的玩具配置：

$$
B=1,
\qquad
S=2,
\qquad
U=3,
\qquad
V=6,
\qquad
D=4,
\qquad
h=2,
\qquad
d_h=2,
\qquad
M=8.
$$

令源 token 为 $(1,2)$，目标右移输入为 $(\mathrm{BOS},3,4)$。embedding 表只取与坐标对齐的几行：

$$
E_1=(1,0,0,0),
\qquad
E_2=(0,1,0,0),
\qquad
E_3=(0,0,1,0),
\qquad
E_4=(0,0,0,1).
$$

暂时令位置向量为零。则 encoder 和 decoder 的输入 shape、前几行表示为：

| 对象 | token id | hidden 的部分内容 | shape |
| --- | --- | --- | --- |
| 源输入 | $(1,2)$ | $(E_1,E_2)$ | $(1,2,4)$ |
| 目标输入 | $(\mathrm{BOS},3,4)$ | $(0,E_3,E_4)$ | $(1,3,4)$ |
| 目标标签 | $(y_1,y_2,y_3)$ | 不进入当前输入 | $(1,3)$ |

这个例子先验证两个事实：源长度是 2，目标长度是 3；decoder 的 cross-attention score 矩阵必须是 $(1,3,2)$。

### 一个 head 的 score 和 value 读取

选一个 $d_h=2$ 的 head，某个 query 和两个 key/value 为

$$
q=(1,0),
\qquad
k_1=(1,0),
\qquad
k_2=(0,1),
\qquad
v_1=(2,0),
\qquad
v_2=(0,4).
$$

缩放后的两个 score 是

$$
s_1=\frac{1}{\sqrt 2}=0.707106781187,
\qquad
s_2=0.
$$

softmax 权重为

$$
a_1=0.669761549327,
\qquad
a_2=0.330238450673.
$$

因此 value 读取结果为

$$
o=a_1v_1+a_2v_2
=(1.339523098653,1.320953802693).
$$

如果当前残差向量是 $x=(0,1)$，并且输出投影在这个玩具步骤中取为恒等映射，则 attention 残差结果是

$$
x+o=(1.339523098653,2.320953802693).
$$

这一步没有计算完整的 LayerNorm 和 FFN。它只把 query-key score、softmax 权重、加权 value 和残差相加的数值路径固定下来；完整 block 还必须继续经过归一化、FFN 和下一层。

## 参数量与计算量账本

### 一个小的 encoder-decoder 配置

继续使用 $D=4$、$M=8$、$h=2$、$V=6$，取 $L_{\mathrm e}=L_{\mathrm d}=2$。假定使用 LayerNorm、每个线性层无 bias、源和目标各使用可学习位置表，embedding 与输出头绑定。

单个 encoder layer 包含一组 self-attention、一个 FFN 和两个 LayerNorm：

$$
P_{\mathrm e,layer}
=4D^2+2DM+4D
=64+64+16
=144.
$$

单个 decoder layer 包含 masked self-attention、cross-attention、一个 FFN 和三个 LayerNorm：

$$
P_{\mathrm d,layer}
=8D^2+2DM+6D
=128+64+24
=216.
$$

因此两侧 stack 和输入表的参数量为：

| 账本项 | 计算 | 参数量 |
| --- | --- | ---: |
| 2 个 encoder layer | $2\times144$ | 288 |
| 2 个 decoder layer | $2\times216$ | 432 |
| token embedding | $VD=6\times4$ | 24 |
| 源与目标位置表 | $(S+U)D=5\times4$ | 20 |
| tied output head | 与 $E$ 共享 | 0 |
| 合计 | $288+432+24+20$ | 764 |

如果输出头不绑定 embedding，需要再增加 $VD=24$，合计变成 788。若把 bias 加回每个线性层和 LayerNorm 仿射项，也需要逐项加回，不能用“每层大约多少”替代 shape 计数。[参数量总账](../transformer-components/parameter-count/)给出同一口径下的更完整分解。

### 一次训练前向的 MAC

只计算矩阵乘法和 attention 的 QK/AV，不把 LayerNorm、softmax、激活函数和加法算作 MAC。encoder 的一条长度为 $S=2$ 的序列中：

$$
\begin{aligned}
\operatorname{MHA\ projection}
&=4SD^2=128,\\
\operatorname{QK+AV}
&=2hS^2d_h=32,\\
\operatorname{FFN}
&=2SDM=128.
\end{aligned}
$$

因此一个 encoder layer 的这部分工作量为 $128+32+128=288$ MAC。decoder 的 $U=3$ 个位置上：

$$
\begin{aligned}
\operatorname{masked\ self\ projection}&=4UD^2=192,\\
\operatorname{masked\ self\ QK+AV}&=2h\frac{U(U+1)}{2}d_h=48,\\
\operatorname{cross\ projection}&=2(U+S)D^2=160,\\
\operatorname{cross\ QK+AV}&=2hUSd_h=48,\\
\operatorname{FFN}&=2UDM=192.
\end{aligned}
$$

这些项合计 $640$ MAC。cross-attention 的 projection 是 $Q$ 和输出在 $U$ 个位置上计算，$K,V$ 在 $S$ 个 encoder 位置上计算，所以是 $2(U+S)D^2$，不能误写成 $4UD^2$。

### 计算账和参数账的区别

上面的 encoder self-attention 参数量不依赖 $S$，但 score 矩阵的工作量含有 $S^2$。decoder cross-attention 的参数量不依赖 $S$ 或 $U$，但交互工作量含有 $US$。同一组 projection 参数可以被所有位置复用，因此不能从参数量直接推出一次序列的 MAC。

| 对象 | 参数是否随长度变化 | 主要长度项 | 需要记录的 shape |
| --- | --- | --- | --- |
| token embedding | 否 | 查表次数 $BS$ | $(B,S,D)$ |
| encoder self-attention | 否 | $S^2$ 交互 | $(B,h,S,S)$ |
| decoder self-attention | 否 | 训练时约 $U^2$ | $(B,h,U,U)$ |
| decoder cross-attention | 否 | $US$ 交互 | $(B,h,U,S)$ |
| FFN | 否 | $U$ 或 $S$ | $(B,T,M)$ |
| output logits | 否 | $UV$ 乘法 | $(B,U,V)$ |

## 训练时的梯度路径

### 输出误差先回到 decoder hidden

对一个有效位置，softmax cross-entropy 对 logits 的梯度是

$$
\frac{\partial\mathcal L}{\partial Z_{b,u,v}}
=p_{b,u,v}-\mathbf 1\{v=y_{b,u}\}.
$$

若输出头不绑定 embedding，decoder hidden 的梯度为

$$
\frac{\partial\mathcal L}{\partial H^{\mathrm d}_{b,u,:}}
=\sum_{v=1}^{V}
\frac{\partial\mathcal L}{\partial Z_{b,u,v}}
W_{\mathrm{out},v,:}.
$$

cross-attention、decoder self-attention、FFN 和残差支路共同把这个梯度送回 decoder 输入。cross-attention 还会把梯度送入 encoder 的 K/V 投影和 encoder memory；因此 encoder 即使没有直接输出词表 logits，也会因目标端损失获得训练信号。

### 权重绑定会合并两条梯度

如果 $W_{\mathrm{out}}=E$，embedding 表一方面产生输入 hidden，另一方面产生输出类别的行向量。总梯度是两条使用路径的和：

$$
\frac{\partial\mathcal L}{\partial E}
=\left(\frac{\partial\mathcal L}{\partial E}\right)_{\mathrm{input}}
+\left(\frac{\partial\mathcal L}{\partial E}\right)_{\mathrm{output}}.
$$

实现中如果创建了两个参数对象再复制数值，就不再是共享；参数账可能看起来相同，梯度和更新却不同。

### mask 也决定梯度的可达区域

causal mask 让未来 key 的 attention 权重为零，因而当前位置不能通过 decoder self-attention 直接接收未来目标的梯度。teacher forcing 仍会让每个位置看到真值前缀；这不是违反因果性，而是训练条件与自由生成条件不同。

padding mask 应同时约束：

1. attention score 的 key 列；
2. padding query 的 residual 和中间状态是否继续传播；
3. loss 的分子和分母；
4. cross-attention 读取 encoder memory 的有效源位置。

只在 loss 端忽略 padding，不能阻止 padding 在中间层污染其他位置。

## 推理时：prefill、decode 和 KV cache

### prefill 一次读取完整前缀

自回归生成开始时，decoder 输入可能是 BOS 或一段已有前缀。prefill 阶段一次处理长度为 $U_0$ 的目标前缀，在每个 decoder layer 保存 masked self-attention 的 key/value：

$$
K^{\mathrm{cache}}_\ell
\in\mathbb R^{B\times h_{kv}\times U_0\times d_h},
\qquad
V^{\mathrm{cache}}_\ell
\in\mathbb R^{B\times h_{kv}\times U_0\times d_h}.
$$

encoder-decoder 模型还可以把 encoder 侧 cross-attention 的 K/V 预先投影并缓存。decoder-only 模型没有 encoder memory，但 self-attention cache 仍然存在。

### decode 每次只增加一个位置

生成下一个 token 时，只需对新位置计算 query，并把新位置的 key/value 追加到 cache。若当前 cache 长度是 $t$，当前 query 的 shape 是 $(B,h_q,1,d_h)$，读取的 K/V shape 是 $(B,h_{kv},t+1,d_h)$。逻辑 score shape 为 $(B,h_q,1,t+1)$。

不使用 cache 时，每一步都重新计算长度为 $t+1$ 的全部 K/V，累计 projection 工作量和内存读写都会增加。使用 cache 也不会消除 QK/AV 的长度项；它消除的是历史 token 的重复 K/V projection。

### 小配置的单步账

对前面的 $D=4,h=2,d_h=2$，假定 encoder 长度 $S=2$，当前 decoder cache 长度为 $t=2$。一次新增 token 的 decoder self-attention 近似包含：

$$
\begin{aligned}
\operatorname{self\ projection}&=4D^2=64,\\
\operatorname{self\ QK+AV}&=2h(t+1)d_h=24,\\
\operatorname{cross\ projection}&=2D^2=32,\\
\operatorname{cross\ QK+AV}&=2hSd_h=16,\\
\operatorname{FFN}&=2DM=64.
\end{aligned}
$$

合计为 $200$ MAC，忽略 norm、softmax 和加法。若不使用 cache，self-attention 的 projection 需要对三个目标位置重新做完整计算，且历史 K/V 还会被重复写入或读取。

## 原始结构与常见删减

### 三种结构共享同一组组件

“完整 Transformer”在这里指同时有 encoder stack 和 decoder stack 的 encoder-decoder 计算链。工程中常见的三种结构如下：

| 结构 | encoder | decoder | 主要 attention | 常见目标 |
| --- | --- | --- | --- | --- |
| encoder-only | 保留 | 删除 | 双向 self-attention | 表示、分类、抽取 |
| decoder-only | 删除 | 保留 masked self-attention | 因果 self-attention | next-token prediction |
| encoder-decoder | 保留 | 保留 | encoder self、decoder masked self、cross | 条件生成 |

encoder-only 和 decoder-only 不是把同一个 block 换一个 mask 就结束。删去一侧后，输入位置、输出头、训练目标和推理路径也会改变。后续 [Encoder-Only](../transformer-architectures/encoder-only/)、[Decoder-Only](../transformer-architectures/decoder-only/) 和 [Encoder-Decoder](../transformer-architectures/encoder-decoder/)分别处理三种结构的边界。

### 训练目标决定哪些路径有监督

| 目标 | 输入方式 | 监督位置 | 是否需要 cross-attention |
| --- | --- | --- | --- |
| encoder 表示 | 完整输入序列 | pooled 或 token label | 否 |
| decoder next-token | 右移目标前缀 | 每个下一个 token | 否，decoder-only |
| seq2seq generation | 源序列加右移目标 | 每个目标 token | 是，encoder-decoder |
| masked token prediction | 含 mask 的输入 | 被选中的位置 | 否，常见于 encoder-only |

相同的 Transformer 子层可以服务不同目标，但不能把目标函数、mask 和推理流程混写。

## 失效模式：整条链中最容易错的接口

### 把长度轴合并

encoder self-attention 的 score 是 $(B,h,S,S)$，decoder masked self-attention 是 $(B,h,U,U)$，cross-attention 是 $(B,h,U,S)$。如果代码统一用一个“$T$”，矩阵乘法仍可能广播成功，但源和目标位置会被错误混合。

### 目标没有右移

当前标签进入当前 decoder 输入时，训练 loss 会给出过高的结果。检查方法是逐位置打印输入 id 和标签 id，确认位置 $u$ 的输入来自 $y_{u-1}$，而不是 $y_u$。

### mask 方向反了

causal mask 应把 $j>i$ 的位置设成不可读。用一个长度为 4 的全一 attention score 做单元测试，检查第 1 行只有第 1 列有非零概率，第 4 行才可以读取 1 到 4 列。

### cross-attention 读取了错误的 memory

Q 来自 decoder 当前状态，K/V 来自 encoder 最终状态。若把 Q 也从 encoder memory 生成，模块退化成 encoder self-attention；若把 K/V 从 decoder 状态生成，模块不再读取源序列。

### cache 位置和 RoPE 位置不一致

prefill 的最后位置是 $U_0-1$ 时，下一次 decode 的位置应为 $U_0$。如果追加 cache 前后使用同一个位置索引，模型会把不同 token 解释成同一个相位。

### tied weight 只共享数值

复制 embedding 到 output head 不能实现参数绑定。检查参数对象身份、梯度累加位置和 optimizer 参数列表；三个检查都要指向同一份可更新权重。

### 统计了不存在的输出头

绑定 $W_{\mathrm{out}}=E$ 时，参数账不应再加一份 $VD$。相反，若 checkpoint 保存了独立的 lm head，就必须把它计入参数量和存储量。

### padding 只在一处处理

只给 loss 传 ignore_index，不能代替 attention key mask；只给 key mask，也不能代替 loss 分母过滤。至少分别检查 attention 权重、padding hidden、logits loss 和 cache 写入。

### pre-norm 和 post-norm 混用

把 norm 从残差前移动到残差后会改变 block 函数和梯度 Jacobian。加载 checkpoint 或复现论文时，逐子层记录“norm -> sublayer -> add”或“sublayer -> add -> norm”，不要从类名推断顺序。

## 可复用的核验协议

按以下顺序审计一个完整 Transformer：

1. 写出配置：$B,S,U,V,D,h,d_h,M,L_{\mathrm e},L_{\mathrm d}$；
2. 打印源 token、右移目标 token 和标签的 shape；
3. 为 encoder self、decoder self、cross-attention 分别打印 score shape；
4. 用全一或手算 score 检查 mask 的零概率区域；
5. 对一个位置核对 Q/K/V、softmax、加权 value 和残差；
6. 独立计算 embedding、每类 projection、FFN、norm 和 output head 参数量；
7. 分开记录训练全序列 MAC、推理单步 MAC、激活元素和 KV cache 元素；
8. 检查 teacher forcing 的输入标签错位一位；
9. 检查 tied weight 的参数对象和梯度是否共享；
10. 用长度增加、padding、cache 续写三组输入做回归。

其中第 3 步和第 4 步先于性能优化。FlashAttention、fused FFN 和量化可以改变实现成本，但不能改变 Q/K/V 的语义合同。[注意力复杂度](../attention/attention-complexity/)负责拆解交互和 cache 成本，[参数量总账](../transformer-components/parameter-count/)负责统一参数、激活和状态的计算口径。

## 相关词条

- [分词](../text-representation/tokenization/)：把原始文本映射成源序列和目标序列的离散 id。
- [Embedding](../text-representation/embeddings/)：把离散 token id 映射到连续向量。
- [位置编码](../transformer-components/positional-encoding/)：给 hidden 或 Q/K 加入位置条件。
- [Self-Attention](../attention/self-attention/)：展开同一序列内部的 Q/K/V 读取。
- [Multi-Head Attention](../attention/multi-head-attention/)：展开多头拆分、拼接和输出投影。
- [因果掩码](../attention/causal-masking/)：规定 decoder self-attention 的可读区域。
- [Cross-Attention](../attention/cross-attention/)：规定 decoder query 读取 encoder memory 的路径。
- [LayerNorm 与残差](../transformer-components/layernorm-residuals/)：比较归一化和残差加法的顺序。
- [Pre-Norm 与 Post-Norm](../transformer-components/pre-norm-vs-post-norm/)：分析两种 block 顺序的 Jacobian。
- [前馈网络](../transformer-components/feedforward/)：展开 token-wise FFN 的非线性特征变换。
- [参数量总账](../transformer-components/parameter-count/)：分开统计参数、计算、激活和运行时状态。
- [Encoder-Only](../transformer-architectures/encoder-only/)、[Decoder-Only](../transformer-architectures/decoder-only/)、[Encoder-Decoder](../transformer-architectures/encoder-decoder/)：分别处理三种 Transformer 结构。
