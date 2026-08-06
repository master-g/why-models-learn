---
title: "Causal Language Modeling：按前缀预测下一个 token"
tags: ["why-models-learn"]
---

Causal Language Modeling（CLM）把 token 序列的联合概率分解为一串条件概率：第 $t$ 个位置只能根据更早的前缀预测下一个 token。训练时用右移后的输入和真实前缀一次性计算全部位置的 next-token loss；推理时把模型上一步生成的 token 追加到前缀，逐步产生后续 token。decoder-only Transformer 是 CLM 的典型架构，但 CLM 首先描述训练目标和可见性合同。

本文固定 BOS、EOS、PAD、输入标签移位、causal mask、loss reduction、teacher forcing、packed sequence 和 position offset。随后从自回归概率分解推导交叉熵与 perplexity，用一个小 logits 例子核对梯度，再分别计算训练、prefill、单步 decode、KV cache 和输出 head 的资源账本。最后比较 CLM、MLM 与 encoder-decoder 去噪目标的输入轴和监督轴。

![Causal Language Modeling 的数据流：右移输入进入因果 decoder，位置只读取已知前缀，输出 next-token logits 并与下一 token 标签计算损失](/assets/transformer-architectures/svg/causal-language-modeling.1.svg)

## 先固定前缀和标签的接口

### 一条序列如何产生 next-token 对

设一条 token 序列包含 BOS、内容 token 和 EOS：

$$
x_0=\langle\mathrm{bos}\rangle,
\qquad
x_T=\langle\mathrm{eos}\rangle.
$$

把 $x_0,\ldots,x_T$ 看成 $T$ 个预测位置对应的输入—标签边界，则 decoder 输入和监督标签为

$$
X_i=x_i,
\qquad
Y_i=x_{i+1},
\qquad
i=0,\ldots,T-1.
$$

输入的第 $i$ 行包含前缀末 token $x_i$，该行的 logits 预测下一 token $x_{i+1}$。一个具体序列的对齐如下：

| 预测位置 | decoder 输入 $X_i$ | 标签 $Y_i$ | 该行可读取的输入 |
| --- | --- | --- | --- |
| $0$ | BOS | $x_1$ | BOS |
| $1$ | $x_1$ | $x_2$ | BOS、$x_1$ |
| $2$ | $x_2$ | $x_3$ | BOS、$x_1$、$x_2$ |
| $T-1$ | $x_{T-1}$ | EOS | BOS 到 $x_{T-1}$ |

输入和标签的 shape 都是 $(B,T)$。模型输出 logits 的 shape 是 $(B,T,V)$，其中最后一个轴是词表。若使用已经包含 BOS 的 prompt，prompt 中已有 token 仍然可以作为输入上下文；只有需要预测的下一 token 才进入标签。

### BOS、EOS 和 PAD 的责任

BOS 给第一步提供一个合法的前缀，EOS 为可结束的序列提供标签。PAD 只把不同长度序列补成矩形，不属于自然语言序列的概率分解。批量输入需要保存每行的有效 prediction length：

| token | 进入 decoder 输入 | 可以作为标签 | 是否进入有效 loss |
| --- | --- | --- | --- |
| BOS | 第一行输入 | 通常不作为标签 | 通常不进入 |
| 普通内容 token | 后续输入 | 可以作为 next-token 标签 | 有效位置进入 |
| EOS | 作为最后一个标签后停止 | 可以作为结束标签 | 有效位置进入 |
| PAD | 只用于矩形填充 | 不作为内容标签 | 必须屏蔽 |

是否把 EOS 放进训练标签必须与推理停止规则一致。若训练数据没有 EOS，推理只能依赖长度上限或外部停止条件；若训练标签包含 EOS 但生成循环没有检查，模型可能继续从 EOS 后的 padding 或未定义状态生成。

### loss mask 与有效分母

设 $m_{b,i}=1$ 表示第 $i$ 个 prediction position 有效，$0$ 表示 PAD 或被排除的位置。逐位置交叉熵为

$$
\ell_{b,i}
=-\log\operatorname{softmax}(z_{b,i,:})_{Y_{b,i}}.
$$

有效 token mean 为

$$
L_{\mathrm{CLM}}
=
\frac{\sum_{b=1}^{B}\sum_{i=0}^{T-1}m_{b,i}\ell_{b,i}}
{\sum_{b=1}^{B}\sum_{i=0}^{T-1}m_{b,i}}.
$$

分母是有效 next-token 标签数量。不同 batch 的 padding 比例变化时，固定除以 $BT$ 会让同一个 token 的梯度尺度随无效位置数量变化。sum reduction 也可以使用，但要明确总 token 数和梯度缩放。

## 自回归概率与最大似然

### 条件概率分解

给定 BOS，CLM 把一条 token 序列的概率写成

$$
p_\theta(x_{1:T}\mid x_0)
=
\prod_{i=0}^{T-1}
p_\theta(x_{i+1}\mid x_{0:i}).
$$

取对数后，乘积变成逐位置相加：

$$
\log p_\theta(x_{1:T}\mid x_0)
=
\sum_{i=0}^{T-1}
\log p_\theta(x_{i+1}\mid x_{0:i}).
$$

最大化这份对数似然等价于最小化负对数似然：

$$
L_{\mathrm{NLL}}
=
-\sum_{i=0}^{T-1}
\log p_\theta(x_{i+1}\mid x_{0:i}).
$$

训练 batch 的 cross-entropy mean 是带 mask 的 NLL 除以有效标签数。teacher forcing 让每个位置的条件前缀都使用真实 token，因此一次训练前向可以同时计算 $T$ 个因果条件分布。

### softmax logits 的局部梯度

设某一位置的 logits 为 $z\in\mathbb R^V$，目标 token 的词表索引为 $y$：

$$
p_j
=
\frac{\exp(z_j)}
{\sum_{k=1}^{V}\exp(z_k)}.
$$

逐位置 NLL 为

$$
\ell=-\log p_y.
$$

对每个 logit 坐标：

$$
\frac{\partial\ell}{\partial z_j}
=p_j-\mathbb 1[j=y].
$$

目标类别的梯度为负值或趋近于零，其他类别的梯度为正值。梯度再经过输出 head、decoder hidden、causal self-attention 和 embedding 回到前缀表示。

### 三个位置的数字例子

取一个三类词表，三个 next-token 位置的 logits 和目标类别如下：

| 位置 | logits | 目标类别 | softmax 概率 | loss |
| --- | --- | ---: | --- | ---: |
| $0$ | $(1,0,-1)$ | $1$ | $(0.665240955775,0.244728471055,0.090030573170)$ | $0.407605964444$ |
| $1$ | $(0,1,-1)$ | $2$ | $(0.244728471055,0.665240955775,0.090030573170)$ | $0.407605964444$ |
| $2$ | $(0,0,0)$ | $3$ | $(0.333333333333,0.333333333333,0.333333333333)$ | $1.098612288668$ |

三个位置的 mean loss 为

$$
L
=
\frac{0.407605964444+0.407605964444+1.098612288668}{3}
=0.637941405852.
$$

第一个位置的局部梯度是

$$
\nabla_z\ell_0
=(-0.334759044225,0.244728471055,0.090030573170).
$$

第二个位置的局部梯度是

$$
\nabla_z\ell_1
=(0.244728471055,-0.334759044225,0.090030573170).
$$

均匀 logits 的第三个位置梯度是

$$
\nabla_z\ell_2
=(0.333333333333,0.333333333333,-0.666666666667).
$$

三个位置共享 decoder 参数，但每行都有自己的 target index 和 local gradient。mean reduction 只缩放三行梯度，不改变每行目标类别和非目标类别的方向。

### token-level perplexity

若有效 token 的平均负对数似然为 $\overline{\ell}$，perplexity 定义为

$$
\operatorname{PPL}
=\exp(\overline{\ell}).
$$

假设三个位置的正确 token 条件概率分别为 $0.8、0.5、0.25$，联合概率为

$$
0.8\cdot0.5\cdot0.25=0.1.
$$

总 NLL、平均 NLL 和 perplexity 为

$$
\begin{aligned}
L_{\mathrm{sum}}&=2.302585092994,\\
\overline{\ell}&=0.767528364331,\\
\operatorname{PPL}&=2.154434690032.
\end{aligned}
$$

perplexity 依赖 tokenizer、有效 token 集合、EOS/PAD 处理和平均分母。比较两个模型时，必须固定评估文本、tokenizer、loss mask 和 token 计数；字符级和 token 级 perplexity 不是同一个指标。[困惑度](../information-theory/perplexity/)展开这些归一化边界。

## decoder 的因果可见性

### masked self-attention

对 decoder layer 的输入 $H_{\ell-1}$，先做归一化：

$$
X=N_{\ell,1}(H_{\ell-1}).
$$

第 $r$ 个 head 产生

$$
Q_r=XW^Q_r,
\qquad
K_r=XW^K_r,
\qquad
V_r=XW^V_r.
$$

causal score、权重和读取为

$$
\begin{aligned}
Z_r&=\frac{Q_rK_r^{\mathsf T}}{\sqrt{d_h}}+M^{\mathrm{causal}}+M^{\mathrm{pad}},\\
A_r&=\operatorname{softmax}_{\mathrm{key}}(Z_r),\\
O_r&=A_rV_r.
\end{aligned}
$$

在长度 $T$ 且没有 padding 时，位置 $i$ 的可读 key 集合是

$$
\{0,1,\ldots,i\}.
$$

mask 的实现通常把未来位置的 score 设为 $-\infty$，softmax 后对应权重为零。对角线保留当前位置的 decoder 输入；当前位置输入是已知前缀的最后一个 token，当前位置输出预测下一 token。[因果掩码](../attention/causal-masking/)处理包含对角线、padding、cache offset 和 packed sequence 的细节。

### score shape 和可见集合

单头 score 的逻辑 shape 是 $(B,T,T)$，多头 score 是 $(B,h,T,T)$。两个 $T$ 的含义不同：

| 轴 | 含义 | 第 $i$ 行的约束 |
| --- | --- | --- |
| query 轴 | 当前要产生 logits 的输入位置 | 每行一个 next-token 条件 |
| key 轴 | 可被读取的历史输入位置 | 只保留 $j\leq i$ |
| value 轴 | 被权重加和的历史表示 | 与 key 轴一一对应 |

用全一 score 和长度 4 的 mask 检查时，四行的有效列数量应为 1、2、3、4。若第一行能看到第二列，模型已经使用未来 token；若第四行只能看到一列，历史 cache 或 mask offset 可能丢失。

### padding 和 packed sequence

padding mask 处理矩形 batch：

$$
M^{\mathrm{pad}}_{b,i,j}
=
\begin{cases}
0,&\text{key }j\text{ 有效},\\
\text{masked},&\text{key }j\text{ 为 PAD}.
\end{cases}
$$

这里的 masked score 在实现中使用 $-\infty$ 或等价的极小值。query 为 PAD 的行通常也从 loss 中排除；若仍计算该行 hidden，必须确保它不会被当作有效上下文写回其他样本。

packed sequence 把多个短序列拼到同一长度轴。每段内部保留下三角，不同段之间的 score 全部屏蔽：

$$
M^{\mathrm{packed}}_{i,j}
=
\begin{cases}
0,&\operatorname{seg}(i)=\operatorname{seg}(j)\ \text{且}\ j\leq i,\\
\text{masked},&\text{其他情况}.
\end{cases}
$$

仅使用全局下三角会让后一段读取前一段，造成样本间信息泄漏。packed sequence 还要保存每段的 position reset、loss 边界和 EOS 位置。

## 训练：一次前向计算多个因果条件

### teacher forcing

训练时，输入位置 $i$ 使用真实的 $x_i$，标签是 $x_{i+1}$。这就是 teacher forcing：每个位置看到的前缀来自数据，而非模型上一步的采样结果。[Teacher Forcing](../rnn-lstm/teacher-forcing/)进一步讨论真实前缀、free-running 和 exposure bias。

由于所有真实前缀已经在一个矩形张量中，decoder 可以并行计算 $T$ 行 hidden 和 logits；causal mask 负责把第 $i$ 行的未来 key 置为不可见。并行 kernel 不改变概率分解，改变的是同一 batch 内各条件分布的计算调度。

### 训练和推理的条件来源

| 阶段 | 当前输入 | 第 $i$ 步可见 token | logits 的使用 |
| --- | --- | --- | --- |
| teacher forcing 训练 | 数据中的真实 $x_i$ | $x_0,\ldots,x_i$ | 与真实 $x_{i+1}$ 算 loss |
| free-running 推理 | 模型上一步输出 | 已生成的前缀 | 选择或采样下一个 token |
| prefill | 已有完整 prompt | prompt 内部的因果前缀 | 取末位置或新块的 logits |
| decode | 新追加的一个 token | 历史 cache 加当前 token | 产生一个 next-token logits |

训练 loss 低只说明真实前缀条件下的平均概率较高。推理时前缀由模型自身产生，错误 token 会进入后续条件，输出路径与训练路径由此发生差异。

### 输入、标签和 loss mask 的三重审计

CLM 实现通常同时传递三类与位置有关的对象：

| 对象 | 形状 | 作用 | 典型错误 |
| --- | --- | --- | --- |
| shifted input | $(B,T)$ | 提供每行的已知前缀末 token | 与 label 未错开一位 |
| causal attention mask | $(B,h,T,T)$ 或可广播形式 | 屏蔽未来 key | mask 上三角方向反了 |
| loss mask | $(B,T)$ | 排除 PAD、prompt 或无效标签 | 分母包含 padding |

这三者不能用一张布尔矩阵代替。causal mask 决定 hidden 的可见集合，loss mask 决定哪些 logits 进入目标，shifted input 决定当前 logits 对齐到哪个标签。

## 推理：prefill、decode 和 cache

### prefill 一次读取 prompt

给定 prompt $x_{0:P-1}$，prefill 把完整 prompt 一次送入 decoder。第 $P-1$ 个 prompt input 位置产生下一个 token 的 logits；其余 prompt 位置的 logits 可以用于一致性检查，但生成循环通常只取末位置或新 chunk 的末位置。

prefill 的 causal score 逻辑交互数是

$$
\frac{P(P+1)}{2}
$$

乘以 batch、head 和 $d_h$ 的 score/value 两次乘法。dense kernel 可能仍然计算 $P^2$ 个矩形位置，再用 mask 把未来项清除，因此报告资源时必须标注“逻辑交互”还是“实际 dense 计算”。

### decode 只追加新 token

已有长度为 $t$ 的 target cache 时，新 token 的 query 数为 1，key/value 长度为 $t+1$。decoder self cache 按层保存：

$$
K_\ell^{\mathrm{cache}},V_\ell^{\mathrm{cache}}
\in\mathbb R^{B\times h\times t\times d_h}.
$$

当前步把新 token 的 K/V 沿历史长度轴追加。新 query 读取历史和当前 token，不重复计算旧 token 的 Q/K/V projection。若没有 cache，每一步都要重新对整个前缀做 decoder 前向，计算量随生成长度重复增长。

### position offset

cache 只保存 K/V 不会自动保存位置编号。第一个 decode step 的新 token position 应接在已有 prompt 或 target prefix 后面：

$$
p_{\mathrm{new}}=p_{\mathrm{start}}+t.
$$

RoPE、ALiBi 或可学习 position table 都需要使用同一 position offset。常见错误包括 prefill 已经加入一次 RoPE，decode 又对历史 K/V 重新旋转；或 batch 中不同样本的有效长度不同，却共用一个未对齐的 offset。

### prefill 与 decode 的一致性测试

对同一 prompt，分别使用两条路径：

1. 一次 prefill 得到末位置 next-token logits。
2. 逐 token 追加 prompt，读取最后一步 logits。

两者应在明确 dtype 和容差下接近。比较前先固定 tokenizer、position、causal mask、attention implementation、cache layout 和 dropout 状态。logits 差异若超过容差，先检查 shift、mask offset、K/V 追加轴和 position rotation。

## 资源账本：训练长度与生成步数不同

### 固定一个小配置

取一个 decoder-only CLM 配置：

| 量 | 数值 | 含义 |
| --- | ---: | --- |
| $B$ | $2$ | batch size |
| $T$ | $4$ | 训练 prediction positions |
| $V$ | $10$ | token 词表大小 |
| $D$ | $4$ | hidden width |
| $h$ | $2$ | query head 数 |
| $d_h$ | $2$ | 单 head width |
| $M$ | $8$ | FFN 中间宽度 |
| $L$ | $2$ | decoder 层数 |
| $t$ | $3$ | decode 前已有 target cache 长度 |

这里先排除 bias、额外 final norm 以外的实现差异；参数账本会明确计入最终 LayerNorm。

### 参数量

一个含 masked self-attention、FFN、三个 LayerNorm 的 decoder layer 为

$$
P_{\mathrm{layer}}
=8D^2+2DM+6D
=8\cdot4^2+2\cdot4\cdot8+6\cdot4
=216.
$$

输入 token embedding、可学习位置表和最终 LayerNorm 为

$$
P_{\mathrm{input}}
=VD+TD+2D
=40+16+8
=64.
$$

如果 LM head 与输入 embedding 绑定：

$$
P_{\mathrm{tied}}
=P_{\mathrm{input}}+LP_{\mathrm{layer}}
=64+2\cdot216
=496.
$$

如果输出 head 独立，再加 $VD=40$：

$$
P_{\mathrm{untied}}=496+40=536.
$$

输入 embedding 与输出 head 的参数绑定必须体现在参数身份和 optimizer 参数列表中；只复制初值不会合并反向梯度。

### 训练时的逻辑 attention MAC

一个 decoder layer 的线性 projection 和 FFN MAC 为

$$
\mathrm{MAC}_{\mathrm{proj}}=4TD^2=256,
\qquad
\mathrm{MAC}_{\mathrm{ffn}}=2TDM=256.
$$

因果下三角的逻辑 query-key pair 数是 $T(T+1)/2$，所以 score 和 value 的逻辑 MAC 为

$$
\mathrm{MAC}_{\mathrm{causal}}
=2h\frac{T(T+1)}{2}d_h
=80.
$$

按逻辑可读 pair 计数，一个 decoder layer 为

$$
(256,80,256),
\qquad
\mathrm{MAC}_{\mathrm{layer}}=592.
$$

两层、batch size 2 的逻辑主 MAC 为 $2\cdot2\cdot592=2368$。若实现使用完整 $T\times T$ dense 矩阵再施加 mask，attention 项为

$$
2hT^2d_h=128,
$$

对应单层 dense 主 MAC $(256,128,256)=640$，batch 两层为 $2560$。两组数字的差异来自“屏蔽后的逻辑交互”和“实际 dense kernel 乘法”两个口径。

### 输出 head 和单步 decode

训练时全序列 LM head 的 MAC 为

$$
\mathrm{MAC}_{\mathrm{train\ head}}
=BTDV
=2\cdot4\cdot4\cdot10
=320.
$$

decode 时只对一个新 token 产生 logits。已有 cache 长度 $t=3$ 时，单层主要 MAC 为：

| 支路 | 单步 MAC | 计算条件 |
| --- | ---: | --- |
| Q/K/V/O projection | $4D^2=64$ | 只投影新 token |
| causal score/value | $2h(t+1)d_h=32$ | 新 query 读取 4 个 key |
| FFN | $2DM=64$ | 只处理新 token |
| LM head | $DV=40$ | 产生一个 next-token logits |
| 合计 | $200$ | 不含逐元素开销 |

单步 attention 的 key 长度随 $t$ 增长，projection 和 FFN 不随历史长度增长；cache 把历史 projection 的重复计算变成历史 K/V 的读取。

### KV cache bytes

标准多头、FP16、两层 decoder、$B=2,h=2,d_h=2,t=3$ 时，self K/V cache 的元素数为

$$
N_{\mathrm{KV}}
=2LBh d_h t
=2\cdot2\cdot2\cdot2\cdot2\cdot3
=96.
$$

每个 FP16 元素占 2 bytes，因此 cache 为

$$
\mathrm{bytes}_{\mathrm{KV}}=96\cdot2=192\ \text{bytes}.
$$

GQA/MQA 把 K/V head 从 $h$ 改为 $h_{\mathrm{kv}}$，cache bytes 中对应的 head 因子也改为 $h_{\mathrm{kv}}$；query head 的逻辑输出仍按 $h$ 产生。[参数量总账](../transformer-components/parameter-count/)分别记录参数、MAC、激活和运行时状态。

## tokenizer、上下文和训练数据边界

### tokenization 改变预测事件

CLM 的一个 prediction position 对应一个 tokenizer token，不一定对应一个字符、词或字。换 tokenizer 会改变：

| 量 | 受 tokenizer 影响的部分 |
| --- | --- |
| 序列长度 $T$ | 同一文本被拆成多少预测事件 |
| 词表 $V$ | 输出 head 的类别数和参数量 |
| loss 分母 | 有效 token 数 |
| perplexity | 每 token 的平均不再可直接跨 tokenizer 比较 |
| context window | 固定 token 窗口覆盖的字符或词数量 |

因此 CLM 报告应同时记录 tokenizer、特殊 token、有效 token 数和截断策略。[分词](../text-representation/tokenization/)处理 token boundary、未知 token 和长度合同。

### 截断和滑动窗口

若原始文档超过最大长度，截断会删除部分条件上下文。滑动窗口可以让后续片段继续参与训练，但窗口之间是否携带状态、是否重复计数重叠 token，都要显式记录。把文档简单切成固定块并在每块前插入 BOS，会把跨块依赖改成独立样本。

### packed sequence

packed sequence 把多个短样本放进一个矩形 batch，能够减少 padding。它要求同时维护：

1. 每个 segment 的起止位置；
2. 每个 segment 的 position reset 或连续 position 方案；
3. block-diagonal causal mask；
4. 每个 segment 的 EOS 和 loss mask；
5. sample id 到输出 logits 的映射。

任意一项缺失都可能把一个样本的未来 token 暴露给另一个样本，或者把 padding 计入 loss。

## CLM、MLM 和 encoder-decoder 的边界

### 目标与架构的对应

| 目标 | 输入形式 | 可见性 | 预测方式 | 典型架构 |
| --- | --- | --- | --- | --- |
| CLM | 右移的真实前缀 | 当前位置及过去 | 每个位置预测下一个 token | decoder-only |
| MLM | 选中位置被腐蚀的同长序列 | 双向有效上下文 | 只恢复 selected token | encoder-only |
| 去噪 Seq2Seq | 腐蚀 source | encoder 双向，decoder 因果 | decoder 生成 target 序列 | encoder-decoder |

CLM 的 loss 对所有有效 next-token 位置通常都可计算；MLM 的 loss 只对 selected positions 计算；去噪 Seq2Seq 的标签在独立 target 轴上右移。三种目标都可以使用 cross-entropy，但输入、mask、位置轴和 loss 分母不同。[Masked Language Modeling](../transformer-architectures/masked-language-modeling/)与 [Encoder-Decoder](../transformer-architectures/encoder-decoder/)展开另外两种合同。

### CLM 的训练输出和生成输出

训练一次前向会产生整段 logits，推理一次 decode 通常只需要最后一个有效位置的 logits。若直接把训练输出的所有位置都当作生成结果，会重复读取已经知道的前缀；若推理只保存最后位置而没有 cache，则会在下一步重新计算历史。

生成策略还可以在 logits 后加入 temperature、top-k、top-p 或 beam search。它们改变 token 选择，不改变 CLM 的训练概率分解；评估 perplexity 时应使用未经过采样截断的模型概率。

## 失效模式：因果目标中的边界错误

### 输入和标签没有右移

现象是 loss 异常低，模型在训练中读取当前标签后直接复制。逐位置打印 input id 和 label id，检查 $Y_i=X_{i+1}$；第一个输入应为 BOS 或明确的 prompt 起点。

### causal mask 方向写反

现象是第一个 query 能读取未来列，训练结果好于推理结果。用长度 4 的全一 score 检查每行有效列数量是否为 1、2、3、4；不要只检查 mask 的矩阵 shape。

### PAD 进入 attention 或 loss

现象是短样本的 logits 受 padding 影响，perplexity 随 batch padding 比例变化。分别检查 padding key mask、padding query 处理和 loss denominator；不能只把 PAD embedding 设为零。

### EOS 训练和停止规则不一致

现象是模型生成内容正确但不停止，或训练 loss 统计漏掉 EOS。固定 EOS 是否进入 labels、生成循环的停止条件和评估 token 计数。

### packed sequence 跨段读取

现象是后一段样本的 query 可以读到前一段或其他段的 key，单样本测试正常，packed batch 测试异常。检查 segment id、block-diagonal causal mask 和 position reset。

### prefill 和 decode logits 不一致

现象是同一 prompt 的一次 prefill 与逐 token decode 产生不同 next-token logits。优先检查 position offset、RoPE 重复、K/V 追加轴、cache layer 顺序和 causal mask 历史长度。

### 只保存 cache tensor，不保存有效长度

预分配 cache 的容量可能大于真实历史长度。若把未写入容量也当作 key，模型会读取未初始化值或 padding。cache 需要同时记录 tensor、有效长度、position offset 和 batch/beam 排列。

### 用经过采样的 token 计算 perplexity

生成时 temperature、top-k 或 top-p 会改变选择路径。perplexity 应在原始模型 logits 上对真实标签计算，不应对采样后截断的分布直接取平均。

### 词表或 special id 发生漂移

现象是 logits 维度仍为 $V$，但 BOS、EOS、PAD 或普通 token 的 id 语义已经变化。加载 checkpoint 前核对 tokenizer 文件、词表顺序、special id 和 tied embedding 参数身份。

## 可复用的核验协议

审计一个 CLM 实现时，按以下顺序记录：

1. 写出 tokenizer、BOS/EOS/PAD id、$B,T,V,D,h,h_{\mathrm{kv}},d_h,M,L$ 和 position 方案。
2. 保存一条短序列的 input ids、label ids、loss mask、有效长度和 segment id。
3. 用长度 4 的手算 mask 检查每行可读 key 集合和 score shape $(B,h,T,T)$。
4. 独立计算一行 softmax、NLL 和 $p-\mathbb 1[j=y]$ 梯度，再核对 mean/sum reduction。
5. 用真实 token 条件概率重算联合概率、平均 NLL 和 perplexity。
6. 分别核对 embedding、position table、decoder layers、final norm、tied/untied LM head 参数量。
7. 分开计算 logical causal MAC、dense masked MAC、full training head、single-step decode 和 KV cache bytes。
8. 比较同一 prompt 的 prefill 与逐 token decode logits，记录 dtype、容差、position offset 和 cache length。
9. 用 padding、EOS 提前结束、不同长度、packed sequence 和跨 batch cache reset 做回归。
10. 再测试 temperature、top-k、top-p 或 beam search，确认它们只改变选择路径，不改训练 loss 和评估概率。

第 3、4、8 步应先于 fused attention、量化和分布式并行。实现可以改变 kernel 和存储布局，但不能改变右移合同、未来 key 的屏蔽范围、有效 loss 分母、position offset 和 cache 有效长度。[Decoder-Only](../transformer-architectures/decoder-only/)固定架构边界，[因果掩码](../attention/causal-masking/)固定可见性，[交叉熵](../information-theory/cross-entropy/)和 [困惑度](../information-theory/perplexity/)固定评估口径。

## 相关词条

- [Decoder-Only](../transformer-architectures/decoder-only/)：固定因果 decoder、masked self-attention、prefill/decode 和 KV cache。
- [因果掩码](../attention/causal-masking/)：推导下三角、padding、packed sequence 和 cache offset。
- [Teacher Forcing](../rnn-lstm/teacher-forcing/)：解释真实前缀训练与 free-running 推理的条件差异。
- [交叉熵](../information-theory/cross-entropy/)、[Softmax](../neurons-and-activations/softmax/)：推导 next-token logits 的概率、损失和梯度。
- [困惑度](../information-theory/perplexity/)：把平均 token NLL 转成评估指标并处理 tokenizer 边界。
- [分词](../text-representation/tokenization/)、[Embedding](../text-representation/embeddings/)：处理预测事件、词表 id 和输入向量。
- [Masked Language Modeling](../transformer-architectures/masked-language-modeling/)：比较双向腐蚀恢复与因果前缀预测。
- [Encoder-Decoder](../transformer-architectures/encoder-decoder/)：比较 source/target 两轴和去噪序列生成。
- [完整 Transformer](../transformer-architectures/full-transformer/)：组装 encoder、decoder、cross-attention 和输出 logits。
- [参数量总账](../transformer-components/parameter-count/)：核对参数、MAC、激活和 KV cache。
