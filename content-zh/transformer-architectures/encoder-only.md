---
title: "Encoder-Only：双向上下文编码器"
tags: ["why-models-learn"]
---

Encoder-only Transformer 只保留 encoder stack。它把一条完整输入序列同时提供给所有位置，每个非 padding token 可以读取同一序列中其他非 padding token 的表示，因此输出是双向的上下文表示。模型本身不按时间顺序生成 token，也不需要 decoder 的 causal mask 或 cross-attention；输出通常接 token classification、sequence classification、span prediction 或 embedding head。

本文固定 $(B,T,D)$ 的输入合同，先写 token embedding 和位置条件，再推导没有因果约束的 bidirectional self-attention、pre-norm encoder block 和最终 hidden。随后用一个三位置 attention 例子核对全局读取，用小配置计算 encoder、pooling 和分类头的参数量与 MAC，最后区分 padding、位置、pooling、任务头和训练目标的边界。MLM 的遮盖采样与交叉熵单独留给 [Masked Language Modeling](../transformer-architectures/masked-language-modeling/)。

![Encoder-only Transformer 的数据流：token 加入位置条件后经过双向 encoder stack，输出 contextual hidden，再分流到 token、CLS 池化和 span 任务头](/assets/transformer-architectures/svg/encoder-only.1.svg)

## 先固定 encoder-only 的接口

### 输入和输出的轴

设 batch size 为 $B$，序列长度为 $T$，词表大小为 $V$，模型宽度为 $D=d_{\mathrm{model}}$。token id 和 hidden 的 shape 为

$$
I\in\{0,\ldots,V-1\}^{B\times T},
\qquad
H_0\in\mathbb R^{B\times T\times D}.
$$

encoder stack 输出

$$
H_L\in\mathbb R^{B\times T\times D},
$$

其中 $L$ 是 encoder block 的层数。每个输出位置仍对应输入中的一个位置，但它的最后一维已经混合了上下文信息。

下面的对象使用不同的职责：

| 对象 | shape | 作用 | 是否保留位置轴 |
| --- | --- | --- | --- |
| token id | $(B,T)$ | 词表查表的离散索引 | 是 |
| token embedding | $(B,T,D)$ | 把 id 变成连续向量 | 是 |
| encoder hidden | $(B,T,D)$ | 双向上下文表示 | 是 |
| pooled hidden | $(B,D)$ | 序列级任务的固定长度表示 | 否 |
| token logits | $(B,T,C)$ | 每个位置的分类分数 | 是 |
| sequence logits | $(B,C)$ | 整条序列的分类分数 | 否 |

$C$ 是任务类别数。不要把 $(B,T,D)$ 的 contextual hidden 与 $(B,D)$ 的 pooled hidden 混成同一个输出接口。

### padding mask 是有效性合同

变长序列需要补成矩形。令

$$
m_{b,j}=
\begin{cases}
1,&\text{位置 }j\text{ 是有效 token}\\
0,&\text{位置 }j\text{ 是 padding}
\end{cases}
$$

attention 的 key mask 应把 padding 列排除。若一个 batch 的第一条样本长度为 3、第二条样本长度为 5，可以写成：

| 样本 | $T$ 的位置 | 有效 key |
| --- | --- | --- |
| $b=1$ | $1,2,3,4,5$ | $1,2,3$ |
| $b=2$ | $1,2,3,4,5$ | $1,2,3,4,5$ |

padding query 的输出和损失也要单独处理。只屏蔽 key 列会阻止有效 token 读取 padding，但不会自动让 padding query 的 hidden 和 loss 消失。

## 输入端：embedding 和位置条件

### token embedding

embedding 矩阵为

$$
E\in\mathbb R^{V\times D}.
$$

给定 token id，按行取出连续向量：

$$
X_{0,b,t,:}=E_{I_{b,t},:}.
$$

token id 的数值大小没有距离含义。id 为 7 的 token 不比 id 为 6 的 token 更接近；几何关系来自 $E$ 的训练结果。

### 位置让模型区分排列

没有位置条件时，self-attention 对输入位置的置换是等变的：重新排列 token，输出会按同样的排列重新排列。模型可以读取整组内容，却不能从表示中恢复原始顺序。

使用可加位置表 $P\in\mathbb R^{T_{\max}\times D}$ 时，输入为

$$
H_0=X_0+P_{0:T,:}.
$$

更一般地，位置可以通过以下路径进入：

| 位置方案 | 作用位置 | 长度限制的主要来源 |
| --- | --- | --- |
| fixed sinusoidal | 加到 token hidden | 公式可继续计算，但频率与数值范围仍需检查 |
| learned absolute | 加到 token hidden | 位置表最大行数 $T_{\max}$ |
| RoPE | 旋转 Q/K | 频率、角度和外推校准 |
| ALiBi | 加到 score | slope、距离和 mask |

encoder-only 的双向性不由位置编码提供。位置编码只告诉模型位置条件，能否读取未来位置由 attention mask 决定。

### 特殊 token 与 pooling

sequence classification 常在输入开头加入 CLS，末尾加入 SEP。CLS 不是数学上必须存在的 token，而是一个约定好的序列级读出位置。若使用 CLS pooling：

$$
h_{\mathrm{cls}}=H_{L,:,t_{\mathrm{cls}},:}\in\mathbb R^{B\times D},
$$

其中 $t_{\mathrm{cls}}$ 通常是固定的第一个位置。若位置可能变化，必须通过 mask 或显式索引确定它。

mean pooling 则按有效 token 做归约：

$$
h_{\mathrm{mean},b,:}
=\frac{\sum_{t=1}^{T}m_{b,t}H_{L,b,t,:}}
{\sum_{t=1}^{T}m_{b,t}}.
$$

分母是有效 token 数，不是填充后的 $T$。CLS pooling 和 mean pooling 产生的表示不是同一坐标，不能在未说明池化规则时直接比较。

## bidirectional self-attention：所有有效位置互相读取

### Q、K、V 的 shape

对某一层归一化后的输入 $X\in\mathbb R^{B\times T\times D}$，将 $D$ 拆成 $h$ 个 head，每个 head 的宽度为

$$
d_h=\frac{D}{h}.
$$

第 $r$ 个 head 的投影为

$$
\begin{aligned}
Q_r&=XW^Q_r,\\
K_r&=XW^K_r,\\
V_r&=XW^V_r,
\end{aligned}
\qquad
Q_r,K_r,V_r\in\mathbb R^{B\times T\times d_h}.
$$

所有 query、key、value 都来自同一序列，所以这是 self-attention。每个 query 都有 $T$ 个候选 key，padding mask 再从候选中移除无效列。

### score、mask 和 value

第 $r$ 个 head 的 score 为

$$
S_r=\frac{Q_rK_r^{\mathsf T}}{\sqrt{d_h}}+M^{\mathrm{pad}}.
$$

其 shape 是 $(B,T,T)$。行是 query 位置，列是 key 位置。padding mask 为

$$
M^{\mathrm{pad}}_{b,i,j}
=
\begin{cases}
0,&m_{b,j}=1\\
-\infty,&m_{b,j}=0
\end{cases}
$$

沿 key 列做 softmax：

$$
A_r=\operatorname{softmax}_{\mathrm{key}}(S_r),
\qquad
O_r=A_rV_r.
$$

对没有 padding 的长度为 $T$ 的序列，$A_r$ 的每一行都可以使用全部 $T$ 列。第 $i$ 行不需要满足下三角约束；这正是 encoder-only 与 decoder-only 的计算差异。

多头输出拼接后经过输出投影：

$$
\operatorname{MHA}(X;M^{\mathrm{pad}})
=\operatorname{Concat}(O_1,\ldots,O_h)W^O.
$$

[Self-Attention](../attention/self-attention/)负责单序列读取，[Multi-Head Attention](../attention/multi-head-attention/)负责多头分解和拼接。

### 双向不等于无 mask

encoder-only 通常没有 causal mask，但仍然有 padding mask、packed sequence 边界和任务特定 mask。几种约束的含义不同：

| mask | 允许读取的范围 | 用途 |
| --- | --- | --- |
| 无 mask | 全部有效位置 | 普通双向 encoder |
| padding mask | 有效 token | 变长 batch |
| block-diagonal mask | 同一 packed 段内的位置 | packed sequence |
| span mask | 指定局部或分段位置 | 特定预训练或结构任务 |
| causal mask | 当前及过去位置 | decoder 或自回归任务 |

把 causal mask 误加到 encoder-only 会把双向模型改成单向模型。只删除所有 mask 又会让 padding 或 packed 段边界泄漏。

## 一个 pre-norm encoder block

### 两个子层和两条残差

设第 $\ell$ 层输入为 $H_{\ell-1}$，$N_{\ell,1}$ 和 $N_{\ell,2}$ 是归一化模块。常见的 pre-norm block 为

$$
\begin{aligned}
R_\ell
&=H_{\ell-1}
 +\operatorname{MHA}_\ell
   \left(N_{\ell,1}(H_{\ell-1});M^{\mathrm{pad}}\right),\\
H_\ell
&=R_\ell
 +\operatorname{FFN}_\ell\left(N_{\ell,2}(R_\ell)\right).
\end{aligned}
$$

每个张量的 shape 都是 $(B,T,D)$。self-attention 在 token 轴上读取其他位置，FFN 只在最后的特征轴上计算。两条支路都必须把输出投影回 $D$ 维，才能执行逐分量残差加法。

如果使用 post-norm，归一化位置改为

$$
\begin{aligned}
R_\ell&=N_{\ell,1}\left(H_{\ell-1}+\operatorname{MHA}_\ell(H_{\ell-1})\right),\\
H_\ell&=N_{\ell,2}\left(R_\ell+\operatorname{FFN}_\ell(R_\ell)\right).
\end{aligned}
$$

这不是格式变化，而是不同的函数和 Jacobian。[LayerNorm 与残差](../transformer-components/layernorm-residuals/)、[Pre-Norm 与 Post-Norm](../transformer-components/pre-norm-vs-post-norm/)处理两种顺序的局部导数。

### final norm 和 residual stream

pre-norm stack 常在最后一层之后再执行一次归一化：

$$
\widetilde H_L=N_{\mathrm{final}}(H_L).
$$

输出头读取 $\widetilde H_L$ 还是 $H_L$ 是架构合同的一部分。残差流沿每层保留 $(B,T,D)$ 的主通道，attention 和 FFN 产生增量并写回。[残差流](../transformer-components/residual-streams/)给出相同 shape 合同下的读写与干预分析。

## 从 encoder hidden 到任务输出

### token classification

每个位置都有一个分类头：

$$
Z^{\mathrm{tok}}=H_LW_{\mathrm{tok}}^{\mathsf T}+b_{\mathrm{tok}},
\qquad
Z^{\mathrm{tok}}\in\mathbb R^{B\times T\times C}.
$$

序列标注、每 token 的实体类别和逐位置监督都使用这个接口。padding 位置的 label 必须从损失中排除。

### sequence classification

CLS pooling 后的 sequence hidden 为 $h_{\mathrm{cls}}\in\mathbb R^{B\times D}$。分类头为

$$
Z^{\mathrm{seq}}=h_{\mathrm{cls}}W_{\mathrm{seq}}^{\mathsf T}+b_{\mathrm{seq}},
\qquad
Z^{\mathrm{seq}}\in\mathbb R^{B\times C}.
$$

如果使用 mean pooling，先按有效 token 计算 $h_{\mathrm{mean}}$，再使用同一个线性头。head 的参数不会因为 pooling 改变而自动相同；输入表示不同，训练得到的 head 也可能不同。

### span prediction

抽取式问答或区间定位可以为每个 token 产生 start 和 end 分数：

$$
z^{\mathrm{start}}=H_Lw_{\mathrm{start}}+b_{\mathrm{start}},
\qquad
z^{\mathrm{end}}=H_Lw_{\mathrm{end}}+b_{\mathrm{end}}.
$$

两个向量的 shape 都是 $(B,T)$。上下文 token 以外的位置、padding 和不允许的答案区间需要从对应 softmax 或损失中排除。

### embedding head

若只需要序列表示，可以直接保存 pooled hidden，并在相似度或下游检索任务中使用。embedding head 的输出是表示，不是词表 logits；不应因为它来自 Transformer 就把它当成生成概率。

## 一个三位置的双向读取例子

### 固定一个 attention head

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

缩放后的 score 是

$$
s_1=\frac{1}{\sqrt 2}=0.707106781187,
\qquad
s_2=0,
\qquad
s_3=0.707106781187.
$$

因为没有 padding，也没有 causal mask，三个位置都会参加 softmax：

$$
a_1=0.401112092680,
\qquad
a_2=0.197775814640,
\qquad
a_3=0.401112092680.
$$

value 读取为

$$
o=a_1v_1+a_2v_2+a_3v_3
=(1.604448370719,0.395551629281).
$$

如果错误地套用长度为 3 的 causal mask，假设这个 query 是第一个位置，则只能读取 $v_1$，输出会变成 $(1,0)$。同一组 Q/K/V 得到两个不同结果，差异来自可读集合而不是 softmax 实现。

### padding 时的对照

若第三个 key 是 padding，则有效集合只保留前两个位置：

$$
a_1'=\frac{\exp(s_1)}{\exp(s_1)+\exp(s_2)},
\qquad
a_2'=\frac{\exp(s_2)}{\exp(s_1)+\exp(s_2)},
\qquad
a_3'=0.
$$

此时 $a_1'$ 和 $a_2'$ 仍然在有效集合内归一化为 1。把 padding score 设成一个很大的负数，再在有限精度下 softmax，必须确认不会出现 NaN；全 padding 的 query 行还需要显式处理。

## 参数量和 MAC 账本

### 小配置

取一个便于手算的 encoder-only 配置：

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
L=2,
\qquad
C=3.
$$

假定没有线性层 bias，使用 learned position table，两个 LayerNorm 都包含 scale 和 bias。encoder layer 的参数量是

$$
P_{\mathrm{layer}}
=4D^2+2DM+4D
=64+64+16
=144.
$$

token embedding 和长度为 4 的位置表分别是 $VD=40$ 和 $TD=16$。因此 encoder core 参数量为

$$
P_{\mathrm{core}}
=VD+TD+LP_{\mathrm{layer}}
=40+16+288
=344.
$$

如果再加一个 $D\times D$ 的 CLS pooler 及其 bias，增加 $D^2+D=20$；如果加一个三分类头，增加 $DC+C=15$。两者都加入时总量为

$$
P_{\mathrm{total}}=344+20+15=379.
$$

这个 379 是一个同时保存 pooler 和 sequence classifier 的小配置总量。token classification head、span head 和 sequence classifier 通常按任务选择，不应在实际 checkpoint 中无条件全部相加。

### 一层 encoder 的序列 MAC

只统计矩阵乘法以及 QK/AV，不统计 LayerNorm、softmax、激活函数和残差加法。长度 $T=4$ 时，一层 encoder 的三类主项为

$$
\begin{aligned}
\operatorname{MHA\ projection}&=4TD^2=256,\\
\operatorname{QK+AV}&=2hT^2d_h=128,\\
\operatorname{FFN}&=2TDM=256.
\end{aligned}
$$

合计为 $640$ MAC；两层 stack 的对应项为 $1280$ MAC。CLS pooler 对一个 pooled vector 做 $D^2=16$ MAC，sequence classifier 做 $DC=12$ MAC。embedding 查表不计作矩阵乘法。

如果把 T 从 4 增加到 8，参数量保持 379 不变，self-attention 的 QK/AV 从 $2hT^2d_h=128$ 增长到 512，而 FFN 从 256 增长到 512。长度会改变计算量和激活量，不会改变固定层的 projection 参数量。

### activation 和输出大小

在这个配置中，一层 residual hidden 有 $TD=16$ 个元素，中间 FFN 激活有 $TM=32$ 个元素。若用 FP16 保存，一个 hidden 张量占 $16\times2=32$ bytes，中间激活占 $32\times2=64$ bytes。真实训练还需要保存 Q/K/V、attention 权重或重算所需的统计量，峰值不能只用一个 hidden 张量估计。

sequence logits 有 $BC=3$ 个元素，token logits 有 $BTC=12$ 个元素。词表输出类任务的 logits 若使用 $V=10$，则每个位置有 10 个候选，输出元素为 $BT V=40$；这和 encoder core 的参数量是两个不同账本。

## encoder-only 与其他结构的边界

### 与 decoder-only 的差异

| 项目 | encoder-only | decoder-only |
| --- | --- | --- |
| self-attention 可读范围 | 全部有效位置 | 当前及过去位置 |
| 是否使用 causal mask | 通常不使用 | 必须使用 |
| 输出接口 | token/sequence 表示或分类 | next-token logits |
| 推理输入 | 通常一次提供完整序列 | 逐 token 追加前缀 |
| KV cache | 普通编码通常不需要 | 自回归 decode 通常需要 |
| 位置输出 | 每个输入位置都有 hidden | 每个前缀位置都有 hidden，但任务常读最后位置 |

encoder-only 不能直接用 next-token logits 的训练/推理协议代替；decoder-only 也不能因为使用同样的 block 组件就获得双向可见性。

### 与 encoder-decoder 的差异

encoder-decoder 有两条长度轴：源序列由 encoder 读取，目标前缀由 decoder 读取，decoder 还通过 cross-attention 读取 encoder memory。encoder-only 只有一条输入序列轴，self-attention 的 score 是 $(B,h,T,T)$，没有单独的 $(B,h,U,S)$ cross score。

如果一个任务只需要从完整输入提取表示，保留 decoder stack 会增加参数和计算路径。若任务需要根据源序列逐步生成目标序列，encoder-only 的双向 hidden 本身不提供自回归输出协议。

### 与 MLM 的差异

encoder-only 是架构类别，MLM 是训练目标。架构规定每个位置可以读取哪些输入；MLM 规定哪些位置计算监督、输入是否替换为 mask token、随机替换或保持原 token。相同的 encoder-only stack 可以接 token classification、sequence classification 或其他目标。

因此需要把三件事分开记录：

1. attention mask 是否允许双向读取；
2. input corruption 是否修改了哪些 token；
3. loss mask 是否在哪些位置计算监督。

## 训练与微调时的路径

### hidden 先经过任务头

对 token classification，任务头读取每个位置的 $H_{L,b,t,:}$；对 sequence classification，任务头只读取 pooled hidden。梯度先经过任务头，再按位置返回 encoder hidden。没有被任务头读取的位置可能只通过 attention 间接获得梯度，因此不能把“没有直接 label”解释成“没有训练信号”。

### padding 和 label mask

token 级损失应使用有效 label mask：

$$
\mathcal L_{\mathrm{tok}}
=-\frac{1}{\sum_{b,t}m^{\mathrm{label}}_{b,t}}
\sum_{b,t}m^{\mathrm{label}}_{b,t}
\log p_{b,t,y_{b,t}}.
$$

这个 $m^{\mathrm{label}}$ 可以同时排除 padding、未标注位置和任务不需要监督的位置，但它不代替 attention 的 key mask。两者在不同阶段约束不同张量。

### 冻结 encoder 与训练 head

参数高效设置常冻结 encoder，只更新任务头：

| 设置 | 更新参数 | 仍需前向的部分 | 主要风险 |
| --- | --- | --- | --- |
| full fine-tuning | encoder + head | 全部 stack | 显存和灾难性遗忘 |
| frozen encoder | task head | encoder 仍需运行 | 表示与任务不匹配 |
| adapter/LoRA | head + adapter | 全部相关 block | 只统计新增 trainable 参数会误读模型大小 |

参数账要同时报告 total parameter count 和 trainable parameter count。[参数量总账](../transformer-components/parameter-count/)分别处理两种数量以及 optimizer state。

## 失效模式：双向 encoder 的接口错误

### 误加 causal mask

现象是后部 token 无法影响前部 token，attention map 呈下三角。检查一个长度为 3 的全有效输入，确认第一个 query 的 key 权重在三个位置都有可能非零；若只剩第一列，说明 causal mask 仍在生效。

### 把 padding 当成内容

现象包括 CLS 或 mean pooled 表示随 padding 数量改变、token logits 在 padding 上出现稳定预测、attention 权重落在 padding 列。分别检查 key mask、query 输出清零和 loss mask。

### mean pooling 分母错误

若有效长度为 2、补齐长度为 4，却用 4 做分母，表示范数会随 padding 比例系统性缩小。用同一个有效 token 集合生成不同 padding 长度，检查 pooled output 是否保持不变。

### CLS 位置或特殊 token 不一致

训练 tokenizer、微调数据和推理输入必须使用同一个 CLS/SEP/PAD id 及位置。把 CLS 加到序列开头后又用第 0 个原 token 做 pooling，会让 head 读取错误位置。

### 位置表越界或位置偏移

learned absolute position table 只支持已有行。截断和 packed 输入还会决定位置是否重置；若一个 batch 内不同段共用连续位置，却没有段边界 mask，后段 token 会读取前段内容。

### 任务头读取了错误的层

最终 LayerNorm 前后的 hidden、最后一层与中间层、CLS 与 mean pooling 都不是同一个接口。checkpoint 加载和推理代码需要把 readout 层、归一化状态和 pooling 规则一起记录。

### 把架构与目标混写

encoder-only 不等于 BERT，不等于 MLM，也不等于 sequence classification。BERT 是一类具体模型和训练配方，MLM 是训练目标，classification 是下游头；三者必须分开填写。

### 只看参数量比较模型

两个 encoder-only 模型可以有相同参数量，却使用不同序列长度、head 数、位置表、pooling 和输出词表。比较资源时还要记录 MAC、激活、batch、长度和 dtype。

## 可复用的核验协议

审计一个 encoder-only 模型时，按以下顺序记录：

1. 写出 $B,T,V,D,h,d_h,M,L$ 和位置方案；
2. 打印 token id、special token、padding mask 和 position id；
3. 检查每层 Q/K/V、score、attention output 和 residual hidden 的 shape；
4. 用长度 3 的手算 score 验证没有 causal mask，并用 padding 例子验证无效 key 权重；
5. 独立计算 embedding、position table、每层 MHA/FFN/norm、pooler 和任务头的参数量；
6. 分别计算 sequence MAC、token logits、activation elements 和训练状态；
7. 对 CLS、mean pooling、token head、span head 明确实际 readout；
8. 用不同 padding 长度、不同位置长度和 packed 边界做回归；
9. 用一个冻结 encoder 与一个 full fine-tuning 配置核对 trainable parameter count；
10. 记录最终 hidden 的归一化位置、dtype、mask 版本和任务损失分母。

第 4 步必须先于性能优化。FlashAttention、fused FFN 和量化可以改变实现成本，但不能改变双向可读集合和 $(B,T,D)$ 的残差合同。[注意力复杂度](../attention/attention-complexity/)和[参数量总账](../transformer-components/parameter-count/)分别负责长度成本与参数/激活账本。

## 相关词条

- [完整 Transformer](../transformer-architectures/full-transformer/)：把 encoder、decoder、cross-attention 和 logits 组装成一条完整计算链。
- [Self-Attention](../attention/self-attention/)：展开同序列 Q/K/V 读取。
- [Multi-Head Attention](../attention/multi-head-attention/)：展开多头拆分、拼接和输出投影。
- [位置编码](../transformer-components/positional-encoding/)：把位置条件注入 token hidden 或 attention score。
- [RoPE](../transformer-components/rope/)、[ALiBi](../transformer-components/alibi/)：两种不使用可学习绝对位置表的位置信息路径。
- [LayerNorm 与残差](../transformer-components/layernorm-residuals/)、[Pre-Norm 与 Post-Norm](../transformer-components/pre-norm-vs-post-norm/)：处理 encoder block 的归一化和残差顺序。
- [残差流](../transformer-components/residual-streams/)、[前馈网络](../transformer-components/feedforward/)：处理 block 的主通道和逐 token 非线性支路。
- [Masked Language Modeling](../transformer-architectures/masked-language-modeling/)：处理 encoder-only 常见的遮盖训练目标。
- [Decoder-Only](../transformer-architectures/decoder-only/)、[Encoder-Decoder](../transformer-architectures/encoder-decoder/)：比较另外两种 Transformer 架构。
- [参数量总账](../transformer-components/parameter-count/)：分开核算参数、MAC、激活和运行时状态。
