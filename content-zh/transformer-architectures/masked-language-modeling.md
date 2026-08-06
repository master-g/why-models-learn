---
title: "Masked Language Modeling：从腐蚀输入恢复 token"
tags: ["why-models-learn"]
---

Masked Language Modeling（MLM）是一种去噪式预训练目标：先从输入序列中选择一组 token 位置并构造腐蚀输入，再要求模型根据腐蚀后的完整上下文恢复这些位置的原 token。典型实现使用双向 encoder；模型在选中位置计算词表分布，损失只聚合选中位置。MLM 的名称描述训练目标，encoder-only 描述架构，二者属于不同层次的合同。

本文固定原序列、腐蚀序列、标签位置和三类 mask 的关系，推导 MLM 损失与 softmax 梯度。随后展开经典 15% 选择策略、[MASK]/随机替换/保持原 token 三种输入变换、动态 masking、整词 masking、span corruption，以及 selected-token loss 与全序列前向之间的计算差异。最后用一个小配置核对参数、MAC、logits、梯度和激活账本，并比较 MLM、Causal Language Modeling 与 encoder-decoder 去噪目标。

![Masked Language Modeling 的数据流：原 token 选择腐蚀策略形成输入，双向 encoder 产生 hidden，只抽取被选位置经过 MLM head 预测原 token](/assets/transformer-architectures/svg/masked-language-modeling.1.svg)

## 先固定 MLM 的任务接口

### 原序列、腐蚀序列和选中位置

设 tokenized 输入为

$$
x_{1:T}=(x_1,x_2,\ldots,x_T),
$$

其中 $T$ 包含本次输入的 token 位置。特殊 token 和 padding 通常不进入候选集合。令 $\mathcal M$ 为本次被选中的位置集合，腐蚀函数为 $g$，则模型看到的输入是

$$
\widetilde{x}_{1:T}=g(x_{1:T},\mathcal M).
$$

对 $i\in\mathcal M$，标签仍然是原 token $x_i$；对 $i\notin\mathcal M$，该位置没有 MLM 直接监督。批量输入的 shape 为：

| 张量 | shape | 轴含义 | 作用 |
| --- | --- | --- | --- |
| original token id | $(B,T)$ | batch、token 位置 | 产生标签和腐蚀输入 |
| corrupted token id | $(B,T)$ | batch、token 位置 | encoder 输入 |
| selected mask | $(B,T)$ | batch、token 位置 | 选择 MLM loss |
| encoder hidden | $(B,T,D)$ | batch、位置、特征 | 上下文表示 |
| selected logits | $(B,\lvert\mathcal M\rvert,V)$ | batch、选中位置、词表 | 计算恢复概率 |

实际实现常常先计算完整的 $(B,T,V)$ logits，再用 selected mask gather 出有效位置。表中的 $\lvert\mathcal M\rvert$ 可以理解为 batch 中有效选中位置数；不同样本的选中数量不同时，需要使用索引列表或布尔 gather，而不是假定每行数量相同。

### MLM 损失只看选中位置

设 encoder 和 MLM head 产生每个位置的 logits $z_{b,i,:}$。对 batch 中的有效选中位置，逐位置损失为

$$
\ell_{b,i}=-\log\operatorname{softmax}(z_{b,i,:})_{x_{b,i}}.
$$

令 $m_{b,i}=1$ 表示位置 $i$ 被选中且不是 padding，$0$ 表示该位置不参与 MLM loss，则平均损失为

$$
L_{\mathrm{MLM}}
=
\frac{\sum_{b=1}^{B}\sum_{i=1}^{T}m_{b,i}\ell_{b,i}}
{\sum_{b=1}^{B}\sum_{i=1}^{T}m_{b,i}}.
$$

分母是有效选中 token 数，不是 $BT$。如果一批样本分别选中 2、5、3 个位置，平均损失分母为 10；把矩形张量的所有位置直接 mean 会把没有标签的 hidden 当成零损失项，改变梯度尺度。

在同一层 encoder 中，未选中位置仍可能影响 selected hidden：双向 self-attention 会让被选位置读取它们的表示，梯度也会沿这条上下文路径回到未选中位置。未选中位置没有直接的词表交叉熵，但不等于没有梯度。

### 条件分布的含义

MLM 学习的是腐蚀上下文下的 token 条件分布：

$$
p_\theta(x_i\mid\widetilde{x}_{1:T}),
\qquad i\in\mathcal M.
$$

对一次选中的位置集合，可以把训练目标写成各位置损失之和：

$$
\mathcal L_{\mathcal M}
=
\sum_{i\in\mathcal M}\ell_i
=
\sum_{i\in\mathcal M}-\log p_\theta(x_i\mid\widetilde{x}_{1:T}).
$$

这里的逐位置分解是训练目标的聚合方式。多个被选位置共享同一个 encoder 上下文，但输出 head 通常对每个位置独立地产生词表分布；这不等价于 decoder 按顺序生成整条序列。

## 构造腐蚀输入

### 经典 15% 选择率

对每个可候选 token，以概率 $r=0.15$ 选择位置进入 $\mathcal M$。对已选中的位置，再按经典 BERT 风格的期望比例决定输入替换：

| 选择阶段 | 概率 | 输入中的 token | 监督标签 |
| --- | ---: | --- | --- |
| 未选中 | $0.85$ | 原 token $x_i$ | 无 MLM loss |
| 选中后替换为 mask token | $0.15\times0.80=0.12$ | $\langle\mathrm{mask}\rangle$ | 原 token $x_i$ |
| 选中后随机替换 | $0.15\times0.10=0.015$ | 随机候选 token | 原 token $x_i$ |
| 选中后保持原 token | $0.15\times0.10=0.015$ | 原 token $x_i$ | 原 token $x_i$ |

这些是期望比例。若一条序列只有 20 个候选 token，期望选中 3 个位置，期望分配为 2.4 个 mask token、0.3 个随机替换和 0.3 个保持原 token；一次具体采样不会要求这些数量为整数。被选中的位置无论采用哪种输入变换，都保留原 token 作为标签。

保持原 token 的分支让预训练输入分布与下游输入分布存在部分重合。它也会给某些样本提供较短的恢复路径，因此评估时要同时记录不同变换分支的比例，不能把“输入仍然是原 token”解释成该位置没有被监督。

### 随机替换的词表边界

随机替换 token 应从与输入 tokenizer 对应的候选词表中采样。特殊 token、padding token 和不允许作为内容的 id 通常排除在候选集合之外。若随机 token 恰好等于原 token，该样本在输入上看起来属于保持原 token 分支，但标签仍然来自原始数据。

随机替换的作用不是提供一个额外正确答案，而是让模型区分“上下文支持的 token”和“输入位置上暂时出现的 token”。如果把随机替换后的 token 当成标签，目标会从恢复原文变成复制腐蚀结果。

### 动态 masking 与静态 masking

静态 masking 在数据预处理阶段为每条样本固定一次 $\mathcal M$ 和替换结果。动态 masking 在每次读取样本或每个训练周期重新采样：

| 方案 | mask 何时产生 | 同一原序列的监督视角 | 主要检查点 |
| --- | --- | --- | --- |
| 静态 masking | 预处理一次 | 固定 | 数据缓存是否重复使用同一 corrupted input |
| 动态 masking | 取 batch 时或训练时 | 随 epoch 变化 | 随机种子、分布式 rank 和复现记录 |
| 在线 span masking | 取 batch 时选择连续 span | 连续片段 | span 长度分布和边界 |

动态 masking 可以让同一 token 在不同 step 进入或离开 $\mathcal M$，减少一个固定腐蚀视角造成的重复。它也使“同一随机种子复现”需要记录采样器状态、worker 状态和数据顺序。

### token、整词和 span 的选择单位

tokenizer 可能把一个自然语言词拆成多个 subword token。按 token 独立采样会产生只遮住词的一部分、其余 subword 仍可见的情况。整词 masking 先在词边界上选择单位，再把该词对应的所有 subword 加入 $\mathcal M$。span corruption 则选择连续位置区间：

$$
\mathcal M
=
\bigcup_{q=1}^{Q}
\{a_q,a_q+1,\ldots,b_q\}.
$$

选择单位改变监督覆盖和上下文难度，不改变 MLM loss 的基本形式。比较两种 masking 策略时，至少记录 token 级选中率、平均 span 长度、每条样本有效标签数和特殊 token 排除规则。

## 双向 encoder 前向

### corrupted input 进入 embedding

腐蚀后的 token id 查表并加入位置条件：

$$
H_0=E[\widetilde{x}]+P,
$$

其中 $H_0\in\mathbb R^{B\times T\times D}$。$E$ 把离散 id 映射到向量，$P$ 提供顺序条件；腐蚀操作已经发生在 id 或 embedding 选择之前，位置条件不负责告诉模型哪些位置被选中。

如果使用可学习的 mask token id，mask token 有自己的 embedding 行。如果把 mask token 直接替换成全零向量，模型仍然可能训练，但输入分布、参数使用和预训练协议已经不同，不能把两种实现的结果直接合并比较。

### encoder self-attention 保留左右上下文

对 encoder block 的输入 $X$，单头 self-attention 为

$$
Q_r=XW^Q_r,
\qquad
K_r=XW^K_r,
\qquad
V_r=XW^V_r.
$$

没有 causal mask 时，score 和读取为

$$
\begin{aligned}
Z_r&=\frac{Q_rK_r^{\mathsf T}}{\sqrt{d_h}}+M^{\mathrm{pad}},\\
A_r&=\operatorname{softmax}_{\mathrm{key}}(Z_r),\\
O_r&=A_rV_r.
\end{aligned}
$$

在没有 padding 的长度 $T$ 样本中，位置 $i$ 可以读取 $1,\ldots,T$ 的有效 key。若位置 $j$ 被腐蚀为 mask token，位置 $i$ 看到的是替换后的表示，不是原始 $x_j$；若 $j$ 处于选中集合但保持原 token，输入路径保留了该 token 的表面信息，标签仍然进入 loss。

### padding mask 与 token 选择 mask 分工

MLM 至少涉及三种 mask，名称相近但作用不同：

| mask 名称 | 作用对象 | 影响输入 token | 影响 attention 连接 | 影响 loss |
| --- | --- | --- | --- | --- |
| corruption selection | 原始 token 位置 | 决定是否替换输入 | 间接影响 hidden | 通常进入 selected mask |
| padding attention mask | padding key/query | 不替换内容 | 屏蔽 padding key，必要时屏蔽 padding query | 间接影响有效 hidden |
| selected loss mask | 监督位置 | 不改变输入 | 不改变可见性 | 选择进入 MLM loss 的位置 |

corruption selection 和 selected loss mask 在标准 MLM 中来自同一组位置，但仍应在实现中分开保存。padding attention mask 只表达序列长度，不表达一个内容 token 是否被选中；selected loss mask 也不能直接当作 attention mask。

### encoder block 和最终 hidden

使用 pre-norm encoder layer 时：

$$
\begin{aligned}
R_\ell
&=H_{\ell-1}
  +\operatorname{MHA}_\ell\left(N_{\ell,1}(H_{\ell-1});M^{\mathrm{pad}}\right),\\
H_\ell
&=R_\ell
  +\operatorname{FFN}_\ell\left(N_{\ell,2}(R_\ell)\right).
\end{aligned}
$$

经过 $L$ 层后，得到

$$
H_L\in\mathbb R^{B\times T\times D}.
$$

MLM 不需要把输入压缩成一个 sequence vector。每个 selected position 直接读取自己的 contextual hidden，再接同一个 MLM head。[Encoder-Only](../transformer-architectures/encoder-only/)展开双向 contextual hidden、CLS/mean pooling 和下游任务头；MLM 使用其中的 token 级 hidden。

## MLM head 与梯度

### 从 selected hidden 产生词表 logits

对 selected position $i$，可以用一个两层 head 写出通用形式：

$$
\begin{aligned}
u_i&=W_{\mathrm{dense}}h_i+b_{\mathrm{dense}},\\
\widehat{u}_i&=\operatorname{GELU}(u_i),\\
z_i&=W_{\mathrm{out}}\operatorname{LN}(\widehat{u}_i)+b_{\mathrm{out}}.
\end{aligned}
$$

实际模型可以删去 dense、GELU 或额外 LayerNorm。若输入 embedding 和输出 head 绑定，常见约定是

$$
W_{\mathrm{out}}=E^{\mathsf T},
$$

输出 bias 仍可单独存在。head 是否绑定影响参数量，不改变 selected position 的 loss 选择。

### softmax 交叉熵的局部梯度

设词表 logits 为 $z\in\mathbb R^V$，目标类别为 $y$，概率为

$$
p_j=\frac{\exp(z_j)}{\sum_{k=1}^{V}\exp(z_k)}.
$$

逐位置交叉熵为

$$
\ell=-\log p_y.
$$

对每个 logit 坐标的梯度为

$$
\frac{\partial\ell}{\partial z_j}
=p_j-\mathbb 1[j=y].
$$

目标类别的梯度为 $p_y-1$，其他类别的梯度为 $p_j$。因此 selected position 的 head 会增加目标类别相对其他类别的分数；梯度再通过 head、contextual hidden、attention 和 embedding 回到腐蚀输入的上下文路径。

### 三分类数字例子

取一个 selected position 的三类 logits：

$$
z^{(1)}=(1,0,-1),
$$

目标类别为第一个坐标。数值稳定 softmax 为

$$
p^{(1)}
=(0.665240955775,0.244728471055,0.090030573170).
$$

于是

$$
\ell_1=0.407605964444,
$$

局部梯度为

$$
\nabla_z\ell_1
=(-0.334759044225,0.244728471055,0.090030573170).
$$

第二个 selected position 取

$$
z^{(2)}=(0,1,-1),
\qquad
y^{(2)}=3.
$$

其概率、损失和梯度为

$$
\begin{aligned}
p^{(2)}
&=(0.244728471055,0.665240955775,0.090030573170),\\
\ell_2&=2.407605964444,\\
\nabla_z\ell_2
&=(0.244728471055,0.665240955775,-0.909969426830).
\end{aligned}
$$

两个位置采用 mean reduction 时

$$
\frac{\ell_1+\ell_2}{2}=1.407605964444.
$$

若把两个位置和未选中的 6 个位置一起除以 8，结果会变成 $0.351901491111$，梯度也缩小到 selected-token mean 的四分之一。这个差异来自 reduction 分母，不来自 softmax 实现。

## MLM 训练信号的结构

### 选中位置学习恢复关系

被替换位置的 hidden 要根据左右上下文、位置、其他 token 和训练到的参数预测原 token。一个简单局部样本：

$$
(\text{the},\text{cat},\text{sat})
\longrightarrow
(\text{the},\langle\mathrm{mask}\rangle,\text{sat})
\longrightarrow
\text{cat}.
$$

如果一个位置的左侧和右侧证据都存在，双向 encoder 可以同时使用两侧信息。若只保留左侧信息，目标函数仍能计算，但模型所学的条件分布已经接近单向 token prediction，且失去 MLM 的双向上下文合同。

### 未选中位置通过上下文路径获得梯度

设 selected hidden 近似为

$$
h_i=\sum_{j=1}^{T}a_{i,j}v_j.
$$

即使 $j\notin\mathcal M$，只要 $a_{i,j}$ 非零，$v_j$ 也会影响 selected position 的 logits。于是 selected loss 对 $v_j$ 的梯度可以非零：

$$
\frac{\partial\ell_i}{\partial v_j}
=a_{i,j}\frac{\partial\ell_i}{\partial h_i}.
$$

这解释了为什么 MLM 可以训练上下文表示：直接监督集中在被选位置，参数更新却通过 self-attention 传播到提供上下文的 token 表示。

### 多个 mask 的相互作用

一次输入中可以选择多个位置。一个 selected token 可能读取另一个 selected token 的 mask 或随机替换结果，模型需要在信息缺失的条件下联合使用剩余上下文。提高选择率会增加同一行中同时缺失的信息，降低每个标签的直接证据；降低选择率会让任务更接近从轻微噪声中恢复。

实际损失仍然按 selected position 聚合，不要求模型显式建模 selected token 之间的联合输出。比较不同 mask 率时，应同时固定有效标签数、训练 token 数和 batch loss reduction，避免把监督量变化误判为任务难度变化。

## 选择策略的设计边界

### 选择率

选择率 $r$ 决定每个输入平均产生多少监督位置：

$$
\mathbb E[\lvert\mathcal M\rvert]
=rN_{\mathrm{cand}},
$$

其中 $N_{\mathrm{cand}}$ 是排除特殊 token 和 padding 后的候选数。若一批有 $B$ 条相同长度的样本，每条有 20 个候选 token，$r=0.15$ 时平均每条选中 3 个位置，batch size 为 2 时平均有 6 个 MLM labels。

选择率变化会同时影响三个量：输入被腐蚀的比例、每次前向的监督密度、每个 selected token 可以利用的未腐蚀上下文。只比较最终 loss 而不记录这三个量，无法判断差异来自目标函数还是有效标签数。

### [MASK] 比例

选中位置使用 [MASK] 的比例越高，预训练输入与下游文本的表面分布差异越大；随机替换和保持原 token 分支会减小这种差异，但降低了输入中明确缺失的比例。经典分支不是三个互斥训练任务，而是同一个 MLM 目标下的三种腐蚀方式。

### whole-word masking

整词 masking 需要从 tokenizer 的词边界得到 token 分组。若一个词对应 subword 索引集合 $G_q$，选择该词时：

$$
G_q\subseteq\mathcal M.
$$

因此一个词的所有 subword 都进入 selected loss。词边界信息若来自预处理器而不是 tokenizer 本身，要记录对齐规则；否则“整词”可能只覆盖文本显示层的词，却没有覆盖对应的全部 token。

### span corruption

span corruption 选择连续片段，片段长度可以从几何分布或固定分布采样。连续片段使模型面对更长的缺口，也增加边界两侧的上下文依赖。若把 span 直接压缩成一个 sentinel token，则输入长度发生变化，目标形式更接近 encoder-decoder 去噪，而不是长度保持的 BERT-style MLM；两者都能使用 corrupted input，但 shape 和标签合同不同。

## 资源账本：监督稀疏，hidden 计算通常不稀疏

### 固定一个小配置

取

| 量 | 数值 | 含义 |
| --- | ---: | --- |
| $B$ | $2$ | batch size |
| $T$ | $8$ | token 序列长度 |
| $V$ | $10$ | MLM 目标词表大小 |
| $D$ | $4$ | hidden width |
| $h$ | $2$ | attention head 数 |
| $d_h$ | $2$ | 单 head width |
| $M$ | $8$ | FFN 中间宽度 |
| $L$ | $2$ | encoder 层数 |
| $\lvert\mathcal M\rvert$ | $3$ | 本批有效 selected token 总数 |

这个配置把 selected 数量写成 batch 总数。若各样本选中数量不同，仍按有效位置总数归约。

### 参数量

一个 encoder layer 的参数量为

$$
P_{\mathrm{layer}}
=4D^2+2DM+4D
=4\cdot4^2+2\cdot4\cdot8+4\cdot4
=144.
$$

输入 embedding、可学习位置表和最终 LayerNorm 的参数量为

$$
P_{\mathrm{input}}
=VD+TD+2D
=40+32+8
=80.
$$

若 MLM output head 与输入 embedding 绑定，且把输出 bias 省略，则总参数为

$$
P_{\mathrm{tied}}
=P_{\mathrm{input}}+LP_{\mathrm{layer}}
=80+2\cdot144
=368.
$$

若使用独立的 $V\times D$ 输出矩阵，则增加 $VD=40$：

$$
P_{\mathrm{untied}}=368+40=408.
$$

若再加入 dense/GELU/LayerNorm 的 BERT-style MLM head，需要把该 head 的参数单列；不能把 tied output head 的计数当作所有 MLM 实现的总参数。

### encoder MAC

对一个长度为 $T$ 的 encoder layer，主 MAC 可以按投影、attention score/value、FFN 分解：

$$
\begin{aligned}
\mathrm{MAC}_{\mathrm{proj}}&=4TD^2,\\
\mathrm{MAC}_{\mathrm{attn}}&=2hT^2d_h,\\
\mathrm{MAC}_{\mathrm{ffn}}&=2TDM.
\end{aligned}
$$

代入 $T=8,D=4,h=2,d_h=2,M=8$：

$$
(\mathrm{MAC}_{\mathrm{proj}},
\mathrm{MAC}_{\mathrm{attn}},
\mathrm{MAC}_{\mathrm{ffn}})
=(512,512,512).
$$

一个 layer 的主 MAC 为 $1536$，两层为 $3072$。如果按 batch size 2 计算，encoder 主 MAC 为 $6144$。

### full logits 与 selected logits

若 encoder 输出整段 hidden 后一次性产生完整词表 logits，输出头 MAC 为

$$
\mathrm{MAC}_{\mathrm{full\ head}}
=BTDV
=2\cdot8\cdot4\cdot10
=640.
$$

如果 head 只对 selected hidden 做矩阵乘法，且本批 $\lvert\mathcal M\rvert=3$：

$$
\mathrm{MAC}_{\mathrm{selected\ head}}
=\lvert\mathcal M\rvert DV
=3\cdot4\cdot10
=120.
$$

两种实现共享 encoder 的 $6144$ 主 MAC。selected head 减少输出投影和 logits 存储，但不能自动减少双向 encoder 对全部 $T$ 个位置的计算；要减少 encoder 计算，需要另外设计稀疏 attention 或局部上下文路径。

### 激活和 loss 缓冲

本配置的整段 hidden 元素数为

$$
BTD=2\cdot8\cdot4=64.
$$

FP16 hidden 占 $128$ bytes。selected hidden 元素数为

$$
\lvert\mathcal M\rvert D=3\cdot4=12,
$$

FP16 selected hidden 占 $24$ bytes。若保存完整 logits，logits 元素数为 $BTV=160$，FP16 占 $320$ bytes；只保存 selected logits 时为 $\lvert\mathcal M\rvert V=30$ 个元素，占 $60$ bytes。真实训练还要保存反向所需的中间激活、attention 权重或 kernel 重计算状态，以上数字只表示指定张量本身。

## MLM 与其他训练目标的边界

### attention mask、token corruption 和训练目标不要混名

下面三个概念都可能在代码中叫 mask：

| 名称 | 例子 | 改变什么 |
| --- | --- | --- |
| token corruption mask | 位置 3 替换为 [MASK] | 输入 token 内容 |
| padding attention mask | 位置 7 是 PAD | attention 可见 key |
| selected loss mask | 位置 3、6 进入标签 | loss 的求和与分母 |
| causal attention mask | 目标位置只能读取过去 | attention 可见 target key |

MLM encoder 需要 padding attention mask 和 selected loss mask；token corruption mask 决定输入如何被扰动。causal attention mask 属于自回归目标的可见性约束，直接把它加到双向 MLM encoder 会改变任务。

### MLM、Causal Language Modeling 和去噪 Seq2Seq

| 目标 | 输入 | 预测位置 | 典型架构 |
| --- | --- | --- | --- |
| MLM | 同长度的腐蚀序列 | 被选中的原 token | encoder-only |
| Causal Language Modeling | 目标前缀 | 每个 next-token 位置 | decoder-only |
| 去噪 Seq2Seq | 腐蚀源序列 | 目标序列 token | encoder-decoder |
| token classification | 原始或任务输入 | 每个标签位置 | encoder-only |

MLM 使用双向上下文并行预测被选位置；Causal Language Modeling 把历史前缀作为条件并维护时间方向；去噪 Seq2Seq 可以改变输入/输出长度并按目标序列自回归生成。三者都可能出现“mask”这个词，但训练输入、可见性和 loss 轴不同。[Encoder-Only](../transformer-architectures/encoder-only/)、[Decoder-Only](../transformer-architectures/decoder-only/)和 [Encoder-Decoder](../transformer-architectures/encoder-decoder/)分别固定三种架构接口；[Causal Language Modeling](../transformer-architectures/causal-language-modeling/)处理 next-token 目标的具体损失。

### MLM 与下游使用方式

预训练阶段的 selected loss 只要求恢复 token。下游可以把 encoder hidden 连接到序列分类、token 分类、span 预测或检索表示；下游 head 不必复用 MLM head。若继续使用 MLM head 评估，应明确输入中是否包含 [MASK]，以及评估位置是否来自同一 masking policy。

## 失效模式：恢复目标中的边界错误

### 对所有位置求 MLM loss

现象是 loss 看起来下降很快，模型学会复制未腐蚀 token，selected mask 对结果几乎没有影响。打印每个 batch 的 selected 数量，并确认 loss gather 只保留 $m_{b,i}=1$ 的位置。

### 把标签替换到腐蚀输入

现象是被选位置的输入仍然包含真实标签，模型可以走 identity path。逐样本保存 original id、corrupted id、label id，检查 label 只在 loss target 中出现，不被错误写回输入。

### 15% 选择率按字符而非 token 计算

现象是不同 tokenizer 或语言的有效 selected 数量差异很大，训练 token 预算无法比较。选择率应在 tokenized candidate 集合上定义；若改为整词或 span，记录实际 token 覆盖率。

### mask 到 special token 或 padding

现象是模型用 [CLS]、[SEP] 或 PAD 的固定模式获得监督，或者 loss 分母包含无效位置。候选集合和 loss mask 都要排除 special token、padding 和 packed sequence 边界。

### 使用 decoder causal mask

现象是被选位置只能读取左侧上下文，右侧证据被强制屏蔽，任务退化到单向恢复。用长度 4 的 score 矩阵检查每行可读 key 集合；MLM encoder 的有效区域应覆盖左右两侧有效位置。

### 动态 masking 未记录随机状态

现象是相同 checkpoint 和数据顺序无法复现 loss，分布式训练中不同 rank 产生重复或不一致的 mask。记录数据 worker、rank、epoch、样本索引和 masking RNG 状态。

### output head 的词表和 input embedding 不一致

现象是 logits 维度可以运行，但目标 id 对应到错误词表行。核对 tokenizer vocabulary、special id、output bias 长度、tied weight 的参数身份和 checkpoint key。

### 用候选数而不是有效 selected 数归一化

现象是 batch 中 padding 比例或样本长度变化时，loss 和梯度尺度随矩形长度变化。分母使用有效 selected token 数，不使用固定 $BT$ 或未排除 padding 的 $\lvert\mathcal M\rvert$。

### 把 span sentinel 当作普通 MLM

现象是输入长度、标签长度和 logits 轴对不上，或者一个 sentinel 位置需要预测多个被删除 token。先确认目标是长度保持的 token-level MLM，还是改变长度的 encoder-decoder span corruption，再选择数据结构和 loss。

## 可复用的核验协议

审计一个 MLM 实现时，按以下顺序记录：

1. 写出 tokenizer、候选 token 集合、$B,T,V,D,h,d_h,M,L$ 和特殊 token id。
2. 保存一条样本的 original id、corrupted id、selected mask、label id 和 padding mask。
3. 统计 selected token 数、[MASK]/随机替换/保持原 token 的实际比例和 token 覆盖率。
4. 检查 encoder score 的 shape 是否为 $(B,h,T,T)$，并用短序列确认左右两侧有效 key 都可见。
5. 检查 MLM head 的 logits 词表轴、selected gather 索引和目标 id 对齐。
6. 独立计算一个 selected position 的 softmax、交叉熵和 $p-\mathbb 1[j=y]$ 梯度。
7. 分别计算 input embedding、position table、encoder layers、final norm、tied/untied output head 的参数量。
8. 分开计算整段 encoder MAC、full logits head MAC、selected head MAC、hidden/logits 缓冲和有效 loss 分母。
9. 用动态 masking 的固定 RNG 做重复运行，比较 original/corrupted/selected 三份数据结构。
10. 用 source padding、短序列、多个 selected token、whole-word/span、mask rate 变化和 checkpoint reload 做回归。

第 4、5、6 步应先于 kernel 融合、混合精度和分布式数据并行。fused softmax、稀疏 output head 和动态 mask 可以改变实现路径，但不能改变双向可见集合、selected loss 的分母、label id 的词表位置和腐蚀输入的可审计记录。[Softmax](../neurons-and-activations/softmax/)处理概率和稳定计算，[交叉熵](../information-theory/cross-entropy/)处理逐位置损失，[标签平滑](../training-nn/label-smoothing/)处理目标分布的另一种参数化。

## 相关词条

- [Encoder-Only](../transformer-architectures/encoder-only/)：固定双向 encoder、$(B,T,D)$ contextual hidden 和 token 级输出。
- [分词](../text-representation/tokenization/)、[Embedding](../text-representation/embeddings/)：处理 token 候选集合、id、词表行和位置输入。
- [Softmax](../neurons-and-activations/softmax/)、[交叉熵](../information-theory/cross-entropy/)：推导 MLM head 的词表概率、损失和局部梯度。
- [标签平滑](../training-nn/label-smoothing/)：处理 one-hot token target 的平滑变体。
- [Causal Language Modeling](../transformer-architectures/causal-language-modeling/)：比较 next-token 目标、因果可见性和逐步生成。
- [Decoder-Only](../transformer-architectures/decoder-only/)：固定 causal decoder 的训练与推理协议。
- [Encoder-Decoder](../transformer-architectures/encoder-decoder/)：比较双向 source encoder、目标 decoder 和去噪 Seq2Seq。
- [完整 Transformer](../transformer-architectures/full-transformer/)：组装 encoder、decoder、cross-attention 和输出 logits。
- [因果掩码](../attention/causal-masking/)：说明下三角 attention mask 与 MLM 双向可见性的差异。
