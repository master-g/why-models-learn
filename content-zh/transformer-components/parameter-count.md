---
title: "参数量总账：四本账"
tags: ["why-models-learn"]
---

参数量是 checkpoint 中可训练标量的数量。它由权重矩阵、偏置向量、归一化参数、embedding、输出头和其他可学习张量共同组成。参数量不等于一次前向的计算量，也不等于训练或推理时的峰值显存：同一组参数可以被许多 token 复用，激活数量却随 batch 和序列长度增长；attention 的交互量还会随长度平方增长。

对一个 Transformer 模型，至少要分开记录四本账：

1. 参数账：有多少可学习标量，哪些张量共享或绑定；
2. 计算账：一次 token、一次序列或一次训练 step 做多少 MAC 或 FLOPs；
3. 激活账：前向和反向需要保存多少中间元素；
4. 状态账：梯度、FP32 master copy、optimizer moments、KV cache、量化 scale 和通信 buffer 占多少字节。

本文先固定计数单位和 shape 合同，再从单个线性层推到 FFN、SwiGLU、attention、embedding、归一化和完整 Transformer。随后把参数、计算、激活、优化器状态和 KV cache 放进同一个配置例子，最后给出公平比较与核验协议。

![参数量总账把同一模型配置拆成参数、计算、激活和运行时状态四本账](/assets/transformer-components/svg/parameter-count.1.svg)

## 先固定计数单位

### 参数是标量数量

一个矩阵

$$
W\in\mathbb R^{r\times c}
$$

包含

$$
P(W)=rc
$$

个参数。一个偏置向量

$$
b\in\mathbb R^r
$$

包含

$$
P(b)=r
$$

个参数。参数量先是无单位的标量计数，只有乘以 dtype 的字节数后才得到存储大小。

例如，$4096\times11008$ 的矩阵包含 $45\,088\,768$ 个参数。若每个参数以 FP16 保存，权重字节数为

$$
45\,088\,768\times2=90\,177\,536\ \mathrm{bytes}.
$$

使用二进制单位时，

$$
1\ \mathrm{MiB}=2^{20}\ \mathrm{bytes}.
$$

所以该矩阵占用

$$
\frac{90\,177\,536}{2^{20}}=86\ \mathrm{MiB}.
$$

这里的 $86\ \mathrm{MiB}$ 只描述一个 FP16 权重副本，不包含梯度、optimizer state、padding 和框架元数据。

### 四种计数对象

同一个张量在不同账本中的含义不同：

| 对象 | 计数单位 | 是否随 batch/序列长度增长 | 典型例子 |
| --- | --- | --- | --- |
| 参数量 | 标量个数 | 否 | $W\in\mathbb R^{m\times d}$ 的 $md$ |
| 参数存储 | bytes | 否 | FP16 权重的 $2P$ |
| 计算量 | MAC 或 FLOP | 是 | $BTmd$ |
| 激活存储 | elements 或 bytes | 是 | $BTm$ 的中间数组 |

一个矩阵乘法可以在所有 token 上复用同一个 $W$。因此参数账是固定的，计算账和激活账却受 $B$、$T$ 影响。

### 先说明边界

本文默认：

- 参数量只统计模型中保存的可学习标量；
- bias 是否存在单独写出；
- 一个矩阵乘法中的一次乘法加一次累加记为一个 MAC；
- 若报告 FLOPs，明确采用 $1$ MAC 等于 $2$ FLOPs 的换算；
- MiB 使用 $2^{20}$ bytes，MB 使用 $10^6$ bytes；
- padding、alignment、allocator rounding 和通信 buffer 不自动并入理论参数量。

如果模型使用 weight tying、共享层、低秩 adapter 或量化，必须把共享关系和额外张量单独列出。

## 单个线性层的参数与计算

### 列向量形式

设输入和输出分别为

$$
x\in\mathbb R^{d_{\mathrm{in}}},
\qquad
y\in\mathbb R^{d_{\mathrm{out}}}.
$$

带 bias 的仿射层为

$$
y=Wx+b,
\qquad
W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\qquad
b\in\mathbb R^{d_{\mathrm{out}}}.
$$

参数量为

$$
P_{\mathrm{affine}}=d_{\mathrm{out}}d_{\mathrm{in}}+d_{\mathrm{out}}.
$$

忽略 bias 时为

$$
P_{\mathrm{linear}}=d_{\mathrm{out}}d_{\mathrm{in}}.
$$

一个输入向量的矩阵乘法 MAC 为

$$
C_{\mathrm{MAC/token}}=d_{\mathrm{out}}d_{\mathrm{in}}.
$$

bias 加法和激活函数的标量操作没有包含在这个矩阵乘法 MAC 中。

### row-batch 形式

工程实现通常把 token 放在行上。设

$$
X\in\mathbb R^{N\times d_{\mathrm{in}}},
\qquad
N=BT.
$$

若权重仍按列向量约定保存，则

$$
Y=XW^{\mathsf T}+\boldsymbol 1b^{\mathsf T},
\qquad
Y\in\mathbb R^{N\times d_{\mathrm{out}}}.
$$

参数量仍然是

$$
d_{\mathrm{out}}d_{\mathrm{in}}+d_{\mathrm{out}}.
$$

批量矩阵乘法的 MAC 为

$$
C_{\mathrm{MAC}}=N\,d_{\mathrm{out}}d_{\mathrm{in}}.
$$

转置只改变存储和 GEMM 接口，不改变参数量。把 $W$ 的行列方向写反会同时改变 shape、参数账和计算账。

### 一个小矩阵例子

取

$$
d_{\mathrm{in}}=3,
\qquad
d_{\mathrm{out}}=2.
$$

无 bias 时，

$$
P_{\mathrm{linear}}=2\times3=6.
$$

带 bias 时，

$$
P_{\mathrm{affine}}=2\times3+2=8.
$$

对 $N=4$ 个 token，矩阵乘法 MAC 为

$$
C_{\mathrm{MAC}}=4\times2\times3=24.
$$

参数量仍然是 $6$ 或 $8$，不会因为输入 token 数变成 $4$ 而复制四份权重。

### 常见层的统一公式

| 层 | 权重 shape | 无 bias 参数量 | 带 bias 参数量 |
| --- | --- | ---: | ---: |
| 线性层 | $d_{\mathrm{out}}\times d_{\mathrm{in}}$ | $d_{\mathrm{out}}d_{\mathrm{in}}$ | $d_{\mathrm{out}}d_{\mathrm{in}}+d_{\mathrm{out}}$ |
| $1\times1$ 卷积 | $C_{\mathrm{out}}\times C_{\mathrm{in}}$ | $C_{\mathrm{out}}C_{\mathrm{in}}$ | $C_{\mathrm{out}}C_{\mathrm{in}}+C_{\mathrm{out}}$ |
| $k_h\times k_w$ 卷积 | $C_{\mathrm{out}}\times C_{\mathrm{in}}\times k_h\times k_w$ | $C_{\mathrm{out}}C_{\mathrm{in}}k_hk_w$ | 加 $C_{\mathrm{out}}$ |
| embedding | $V\times d$ | $Vd$ | 通常没有 bias |
| RMSNorm | $\gamma\in\mathbb R^d$ | $d$ | 通常没有 $\beta$ |
| LayerNorm | $\gamma,\beta\in\mathbb R^d$ | $2d$ | 参数合同本身已包含两组向量 |

表中的卷积参数量不随输出空间 $H\times W$ 增长。输出空间只会把同一组卷积核重复应用，从而增加 MAC 和激活数量。

## FFN 的参数量

### 普通两层 FFN

设 residual stream 宽度为 $d$，中间宽度为 $m$。列向量形式为

$$
F(h)=W_2\phi(W_1h+b_1)+b_2,
\qquad
h\in\mathbb R^d.
$$

其中

$$
W_1\in\mathbb R^{m\times d},
\qquad
W_2\in\mathbb R^{d\times m}.
$$

无 bias 时参数量为

$$
P_{\mathrm{FFN,no\ bias}}=md+dm=2dm.
$$

包含两个 bias 时为

$$
P_{\mathrm{FFN,bias}}=2dm+m+d.
$$

每个 token 的三次主要操作为第一层矩阵乘法、逐坐标激活和第二层矩阵乘法。矩阵乘法 MAC 为

$$
C_{\mathrm{FFN,MAC/token}}=2dm.
$$

对 $N=BT$ 个 token，

$$
C_{\mathrm{FFN,MAC}}=2BTdm.
$$

### SwiGLU 的三矩阵合同

SwiGLU 使用 gate、up 和 down 三个投影：

$$
g=W_gh+b_g,
\qquad
u=W_uh+b_u,
$$

$$
a=\operatorname{SiLU}(g)\odot u,
\qquad
y=W_da+b_d.
$$

其中

$$
W_g,W_u\in\mathbb R^{m\times d},
\qquad
W_d\in\mathbb R^{d\times m}.
$$

无 bias 时，

$$
P_{\mathrm{SwiGLU,no\ bias}}=3dm.
$$

包含三个 bias 时，

$$
P_{\mathrm{SwiGLU,bias}}=3dm+2m+d.
$$

矩阵乘法 MAC 为

$$
C_{\mathrm{SwiGLU,MAC/token}}=3dm.
$$

在相同 $d$ 和 $m$ 下，SwiGLU 的矩阵参数量与 MAC 都是普通 FFN 的

$$
\frac{3dm}{2dm}=1.5
$$

倍。公平比较必须同时说明中间宽度是否调整。

### 4096/11008 的直接账本

取

$$
d=4096,
\qquad
m=11008.
$$

先算单个矩阵：

$$
dm=4096\times11008=45\,088\,768.
$$

普通 FFN 的无 bias 参数量为

$$
P_{\mathrm{FFN}}=2dm=90\,177\,536.
$$

包含 bias 时为

$$
P_{\mathrm{FFN,bias}}=90\,177\,536+11\,008+4\,096=90\,192\,640.
$$

SwiGLU 的无 bias 参数量为

$$
P_{\mathrm{SwiGLU}}=3dm=135\,266\,304.
$$

包含 bias 时为

$$
P_{\mathrm{SwiGLU,bias}}=135\,266\,304+2\times11\,008+4\,096=135\,292\,416.
$$

单个 token 的 MAC 分别为 $90\,177\,536$ 和 $135\,266\,304$。一次 $B=2,T=4096$ 的 SwiGLU 投影 MAC 为

$$
BT\times3dm
=8192\times135\,266\,304
=1\,108\,101\,562\,368.
$$

这个数字没有包含 SiLU、逐元素乘法、bias 加法、读写和 kernel workspace。

### 相同参数预算下比较中间宽度

普通 FFN 取 $m_{\mathrm{FFN}}=4d$ 时，

$$
P_{\mathrm{FFN}}=2d(4d)=8d^2.
$$

令 SwiGLU 使用中间宽度 $m_{\mathrm{SwiGLU}}$，并忽略 bias。参数量相等要求

$$
3dm_{\mathrm{SwiGLU}}=8d^2.
$$

所以

$$
m_{\mathrm{SwiGLU}}=\frac83d.
$$

当 $d=4096$ 时，

$$
\frac83d=10\,922.666\ldots.
$$

实际实现通常选择硬件友好的整数。$m=11008$ 时，

$$
P_{\mathrm{SwiGLU}}=135\,266\,304,
$$

而普通 FFN 使用 $m=16384$ 时，

$$
P_{\mathrm{FFN,4d}}=134\,217\,728.
$$

二者比值为

$$
\frac{135\,266\,304}{134\,217\,728}=1.0078125.
$$

因此「SwiGLU 有三个矩阵」只说明结构，不足以说明两个模型的实际资源差异。必须写出 $d$、$m$、bias 和比较口径。

## Attention 的参数量

### 多头 attention 的四个投影

设模型宽度为 $d$，query head 数为 $h_q$，每个 head 的宽度为 $d_h$，并满足

$$
h_qd_h=d.
$$

标准 MHA 有四个投影：

$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V,
\qquad
O=ZW_O.
$$

若每个投影的输入和输出宽度都是 $d$，无 bias 参数量为

$$
P_{\mathrm{MHA,no\ bias}}=4d^2.
$$

四个 bias 都存在时为

$$
P_{\mathrm{MHA,bias}}=4d^2+4d.
$$

head 数改变张量的分块方式，但在总宽度 $d$ 固定且使用完整 MHA 时，不改变四个方阵的参数量。

### GQA 和 MQA

GQA 令 query 使用 $h_q$ 个 head，key/value 只使用 $h_{kv}$ 个 head。K/V 的总宽度为 $h_{kv}d_h$。无 bias 参数量为

$$
P_{\mathrm{GQA,no\ bias}}
=d^2+d(h_{kv}d_h)+d(h_{kv}d_h)+d^2
=2d^2+2dh_{kv}d_h.
$$

若四个投影都带 bias，Q 和 O 各贡献 $d$，K 和 V 各贡献 $h_{kv}d_h$：

$$
P_{\mathrm{GQA,bias}}
=2d^2+2dh_{kv}d_h+2d+2h_{kv}d_h.
$$

标准 MHA 是 $h_{kv}=h_q$。MQA 是 $h_{kv}=1$。GQA 只减少 K/V 的投影宽度和 KV cache 宽度，不减少 query 与每个 key/value 的交互次数。

### 4096 宽度的 MHA/GQA/MQA

取

$$
d=4096,
\qquad
h_q=32,
\qquad
d_h=128.
$$

标准 MHA 的无 bias 参数量为

$$
4d^2=67\,108\,864.
$$

GQA-8 的 K/V 总宽度为

$$
h_{kv}d_h=8\times128=1024.
$$

所以 GQA-8 的无 bias 参数量为

$$
2d^2+2d(8\times128)=41\,943\,040.
$$

MQA 的无 bias 参数量为

$$
2d^2+2d(1\times128)=34\,603\,008.
$$

对应的 K/V 投影和 cache 会减少，但 Q 投影、O 投影与 attention 交互的 query head 数仍由 $h_q$ 决定。

### attention 交互不是参数

对 self-attention，输入长度为 $T$ 时，QK 和 AV 的 MAC 分别为

$$
C_{QK}=Bh_qT^2d_h,
$$

$$
C_{AV}=Bh_qT^2d_h.
$$

合计为

$$
C_{\mathrm{interaction}}=2Bh_qT^2d_h.
$$

取 $B=1,T=4096,h_q=32,d_h=128$：

$$
C_{QK}=68\,719\,476\,736,
$$

$$
C_{AV}=68\,719\,476\,736,
$$

$$
C_{\mathrm{interaction}}=137\,438\,953\,472.
$$

这些 MAC 随 $T^2$ 增长，但不会新增可训练参数。把 attention matrix 的元素数加到模型参数量中，会把激活账误记为参数账。

## Embedding、输出头与位置参数

### token embedding

设词表大小为 $V$，模型宽度为 $d$。token embedding 矩阵为

$$
E\in\mathbb R^{V\times d}.
$$

参数量为

$$
P_{\mathrm{embed}}=Vd.
$$

取

$$
V=32\,000,
\qquad
d=4096.
$$

有

$$
P_{\mathrm{embed}}=32\,000\times4096=131\,072\,000.
$$

FP16 权重占用为

$$
131\,072\,000\times2
=262\,144\,000\ \mathrm{bytes}
=250\ \mathrm{MiB}.
$$

这个 embedding 参数量与序列长度无关。序列变长只会让查表后的激活 token 数增加。

### LM head 和 weight tying

输出 logits 通常由

$$
\operatorname{logits}=HE_{\mathrm{out}}^{\mathsf T}+b_{\mathrm{out}}
$$

产生，其中

$$
H\in\mathbb R^{N\times d},
\qquad
E_{\mathrm{out}}\in\mathbb R^{V\times d}.
$$

不共享权重时，输出矩阵新增

$$
P_{\mathrm{head,weight}}=Vd
$$

个参数；若有输出 bias，再增加 $V$：

$$
P_{\mathrm{head}}=Vd+V.
$$

weight tying 令

$$
E_{\mathrm{out}}=E.
$$

此时输出头不新增第二个 $V\times d$ 矩阵，但 logits 计算仍然存在。weight tying 减少参数存储，不会让词表投影的每个 token MAC 自动消失。

### 学习位置 embedding 与 RoPE

学习位置 embedding 使用

$$
P_{\mathrm{pos}}=L_{\max}d
$$

个参数。取 $L_{\max}=2048,d=4096$：

$$
P_{\mathrm{pos}}=2048\times4096=8\,388\,608.
$$

FP16 占用为 $16\ \mathrm{MiB}$。

正弦位置编码和 RoPE 使用固定函数或运行时旋转，通常没有同规模的可学习位置矩阵。它们可能增加逐元素计算，但不增加 $L_{\max}d$ 个训练参数。

### embedding 的词表效应

当 $V$ 增大时，embedding 和 untied LM head 都按 $Vd$ 线性增长。若输入 embedding 与输出 head 不共享，词表参数约为

$$
2Vd.
$$

若共享，约为

$$
Vd.
$$

在小模型中，词表参数可能占据总量的明显比例；在大模型中，block 参数可能占主导。比较两个模型时必须确认词表大小、是否加入额外 special tokens 和是否绑定输入输出权重。

## 归一化、bias 与其他小参数

### RMSNorm

RMSNorm 通常只有一个缩放向量

$$
\gamma\in\mathbb R^d.
$$

每个 RMSNorm 的参数量为

$$
P_{\mathrm{RMSNorm}}=d.
$$

如果一个 pre-norm Transformer block 在 attention 和 FFN 前各有一个 RMSNorm，则两者合计

$$
P_{\mathrm{block,norm}}=2d.
$$

RMSNorm 的参数量相对矩阵投影很小，但在精确复算时不能无声省略。

### LayerNorm

LayerNorm 通常有缩放和偏移：

$$
\gamma,\beta\in\mathbb R^d.
$$

每个 LayerNorm 的参数量为

$$
P_{\mathrm{LayerNorm}}=2d.
$$

两个 LayerNorm 合计 $4d$。把 RMSNorm 的 $d$ 误写成 LayerNorm 的 $2d$ 会在层数较大时产生可见差异。

### bias 的总量

对于矩阵宽度很大的层，bias 常比矩阵参数少几个数量级，但它仍然影响精确参数量。例如 4096/11008 SwiGLU 的三个 bias 合计

$$
2m+d=2\times11\,008+4\,096=26\,112.
$$

相对于 $135\,266\,304$ 个无 bias 参数，这个增量很小；相对于 checkpoint 对账，它必须被标记为存在或不存在。

### scale、gate 和额外参数

某些实现还包含：

| 参数类型 | 典型 shape | 计数方式 |
| --- | --- | --- |
| LayerNorm bias | $d$ | 每个向量的 $d$ |
| RMSNorm scale | $d$ | 每个向量的 $d$ |
| attention temperature | $1$ 或每 head 一个标量 | 按实际 shape |
| residual scale | $1$、每层一个或每通道一个 | 按实际 shape |
| learned mask/logit bias | $h\times L$ 或其他 | 不能按普通 bias 猜测 |
| quantization scale | 每 tensor、每 channel 或每 group | 另列存储，不默认算 trainable |

如果 scale 是从权重统计量生成的常量，它不属于训练参数；如果 scale 是 checkpoint 中可学习的张量，它应计入 trainable parameter count。

## 一个完整 Transformer block

### 统一 block 公式

设 block 包含一个 attention、一个 FFN 和两个归一化层。忽略 bias，标准 MHA 加普通 FFN、RMSNorm 的参数量为

$$
P_{\mathrm{block}}
=4d^2+2dm+2d.
$$

标准 MHA 加 SwiGLU 的参数量为

$$
P_{\mathrm{block,SwiGLU}}
=4d^2+3dm+2d.
$$

若使用 GQA，则把 attention 项替换为

$$
2d^2+2dh_{kv}d_h.
$$

如果使用 LayerNorm，则两个归一化层的项从 $2d$ 变成 $4d$。

### 4096/11008 的标准 MHA block

取

$$
d=4096,
\qquad
m=11008.
$$

标准 MHA、SwiGLU、两个 RMSNorm 的无 bias block 参数量为

$$
P_{\mathrm{block}}
=67\,108\,864+135\,266\,304+8\,192
=202\,383\,360.
$$

如果有 Q/K/V/O bias、SwiGLU 三个 bias 和 RMSNorm 没有 bias，则增加

$$
4d+(2m+d)
=16\,384+26\,112
=42\,496.
$$

带这些 bias 时，

$$
P_{\mathrm{block,bias}}=202\,425\,856.
$$

### GQA-8 block

GQA-8、SwiGLU、两个 RMSNorm 的无 bias block 参数量为

$$
P_{\mathrm{GQA8,block}}
=41\,943\,040+135\,266\,304+8\,192
=177\,217\,536.
$$

相对于标准 MHA block，节省

$$
202\,383\,360-177\,217\,536
=25\,165\,824
$$

个参数。节省来自 K/V 投影宽度，不来自 FFN。

### block 的 MAC

忽略逐元素操作，标准 MHA 加 SwiGLU 的每 token 投影 MAC 为

$$
C_{\mathrm{projection/token}}
=4d^2+3dm.
$$

在 4096/11008 配置中，

$$
C_{\mathrm{projection/token}}
=67\,108\,864+135\,266\,304
=202\,375\,168.
$$

self-attention 还要加入

$$
C_{\mathrm{interaction}}=2Bh_qT^2d_h.
$$

因此，同一个 block 的投影 MAC 是线性随 token 数增长的项，attention interaction 是二次随序列长度增长的项。

## 完整模型参数总账

### 总公式

设模型有 $L$ 个 block，输入 embedding 是否与输出 head 绑定由 $P_{\mathrm{head,new}}$ 表示，位置参数由 $P_{\mathrm{pos}}$ 表示，最终归一化参数为 $P_{\mathrm{final\ norm}}$。总参数量可以写为

$$
P_{\mathrm{model}}
=P_{\mathrm{embed}}
+LP_{\mathrm{block}}
+P_{\mathrm{head,new}}
+P_{\mathrm{pos}}
+P_{\mathrm{final\ norm}}
+P_{\mathrm{other}}.
$$

这里的 $P_{\mathrm{head,new}}$ 只统计没有在 embedding 中重复计数的输出权重。若 output head 与 embedding tying，则

$$
P_{\mathrm{head,new}}=0
$$

或只剩未共享的输出 bias。

### 一个 32 层配置

取

$$
L=32,
\qquad
d=4096,
\qquad
m=11008,
\qquad
V=32\,000.
$$

使用标准 MHA、SwiGLU、两个 RMSNorm、无 bias、输入输出 embedding tying，不加入学习位置 embedding：

$$
P_{\mathrm{block}}=202\,383\,360,
$$

$$
LP_{\mathrm{block}}
=32\times202\,383\,360
=6\,476\,267\,520.
$$

输入 embedding 为

$$
P_{\mathrm{embed}}=131\,072\,000.
$$

最终 RMSNorm 为

$$
P_{\mathrm{final\ norm}}=4096.
$$

所以总参数量为

$$
P_{\mathrm{model}}
=6\,476\,267\,520
+131\,072\,000
+4\,096
=6\,607\,343\,616.
$$

如果取消 weight tying，新增一个 $V\times d$ 输出矩阵：

$$
P_{\mathrm{untied}}
=6\,607\,343\,616+131\,072\,000
=6\,738\,415\,616.
$$

如果再加入输出 bias，还要增加 $V=32\,000$。如果改用 GQA-8，则每层减少 $25\,165\,824$ 个参数，32 层总共减少

$$
32\times25\,165\,824
=805\,306\,368.
$$

在实际模型中，vocabulary、bias、norm、position embedding、head tying 和 GQA 配置都必须从 checkpoint 或配置文件确认。

### 参数量按模块拆分

对上面的 tied、标准 MHA 配置：

| 模块 | 单层或全局参数量 | 占总账的职责 |
| --- | ---: | --- |
| 32 个 attention 投影 | $32\times67\,108\,864$ | Q/K/V/O |
| 32 个 SwiGLU | $32\times135\,266\,304$ | gate/up/down |
| 64 个 RMSNorm | $64\times4096$ | block 输入归一化 |
| token embedding | $131\,072\,000$ | token 到向量 |
| final RMSNorm | $4096$ | block 堆栈末端 |
| untied LM head | $0$ | 本配置使用 tying |

逐项相加时要避免把 32 个 attention 投影与 32 个 block 重复计算。表中每一行的名称对应唯一一组张量。

### 深度、宽度和词表的缩放

在中间宽度按 $m=rd$ 缩放时：

- attention 方阵参数按 $d^2$ 增长；
- FFN 参数按 $dm=rd^2$ 增长；
- block 数按 $L$ 线性增长；
- token embedding 按 $Vd$ 增长；
- 学习位置 embedding 按 $L_{\max}d$ 增长；
- sequence length 不改变参数量。

因此，模型变宽、变深、扩大词表和延长上下文会影响不同账本。不能用总参数量单独解释长上下文推理成本。

## 激活内存

### 激活元素数量

设 batch 为 $B$，序列长度为 $T$，模型宽度为 $d$，FFN 中间宽度为 $m$。一个 residual stream 的元素数量为

$$
N_{\mathrm{stream}}=BTd.
$$

一个 FFN 中间数组的元素数量为

$$
N_{\mathrm{FFN\ activation}}=BTm.
$$

如果保存 gate、up、SiLU 输出和门控结果，数量可能接近四个 $BTm$ 数组；实际实现可能复用 buffer 或重算。

### 4096/11008 的 FP16 激活

取

$$
B=2,
\qquad
T=4096,
\qquad
d=4096,
\qquad
m=11008.
$$

一个 residual stream 有

$$
BTd=2\times4096\times4096
=33\,554\,432
$$

个元素。FP16 下为

$$
33\,554\,432\times2
=67\,108\,864\ \mathrm{bytes}
=64\ \mathrm{MiB}.
$$

一个 FFN 中间数组有

$$
BTm=2\times4096\times11008
=90\,177\,536
$$

个元素。FP16 下为

$$
90\,177\,536\times2
=180\,355\,072\ \mathrm{bytes}
=172\ \mathrm{MiB}.
$$

四个中间数组的理论和为

$$
4\times172=688\ \mathrm{MiB}.
$$

这是保存策略的上界式账本，不是每个 fused kernel 的固定峰值。

### attention matrix 的激活

标准 dense self-attention 的 score 或 probability matrix 元素数为

$$
N_{\mathrm{attn\ map}}=Bh_qT^2.
$$

取 $B=1,h_q=32,T=4096$：

$$
N_{\mathrm{attn\ map}}
=32\times4096^2
=536\,870\,912.
$$

FP16 单个矩阵需要

$$
536\,870\,912\times2
=1\,073\,741\,824\ \mathrm{bytes}
=1024\ \mathrm{MiB}.
$$

如果同时保存 logits 和 probability 两份矩阵，理论上就是 $2048\ \mathrm{MiB}$，还不包含 Q/K/V、输出和其他反向状态。FlashAttention 通过 tile 和 online softmax 避免把完整矩阵写回 HBM，但不等于交互 MAC 消失。

### 训练峰值不是逐项简单相加

训练峰值取决于：

- 哪些中间量被保存；
- 是否在反向重算 gate、activation 或 attention tile；
- optimizer 是否与 forward 使用同一进程；
- allocator 是否复用已释放 buffer；
- data parallel、tensor parallel 和 pipeline parallel 如何切分；
- gradient accumulation 是否保留多个 microbatch 的状态。

因此报告激活内存时，应区分理论元素总和、实际保存集合、峰值 live bytes 和 allocator 观测值。

## 优化器状态与权重存储

### 纯权重存储

若参数量为 $P$，每个参数占 $s$ bytes，纯权重存储为

$$
M_{\mathrm{weight}}=Ps.
$$

常见 dtype 的单副本字节数为：

| dtype | bytes/parameter | 说明 |
| --- | ---: | --- |
| FP32 | $4$ | 常用高精度参考或 master copy |
| BF16 | $2$ | 训练/推理常见权重 dtype |
| FP16 | $2$ | 训练/推理常见权重 dtype |
| INT8 | $1$ | 仍需记录 scale、zero point 和布局 |
| INT4 | $0.5$ | 理论打包值，实际还有 scale 和 metadata |

INT4 的 $0.5$ bytes/parameter 只是数值位宽换算。按 group 存储的 scale、zero point、alignment 和 kernel metadata 必须另计。

### Adam 类状态

以 FP16 参数、FP16 gradient、FP32 master weight、FP32 一阶和二阶 moments 为例，每个原始参数可能需要：

- FP16 权重：$2$ bytes；
- FP16 梯度：$2$ bytes；
- FP32 master weight：$4$ bytes；
- FP32 一阶 moment：$4$ bytes；
- FP32 二阶 moment：$4$ bytes。

合计为

$$
2+2+4+4+4=16\ \mathrm{bytes/parameter}.
$$

这个数字不包含 activation、temporary buffer、通信和碎片。某些实现不保存 FP32 master weight，或使用 8-bit optimizer state，不能直接套用 16 bytes/parameter。

### 6.607B 参数配置的静态账

对上面的

$$
P=6\,607\,343\,616
$$

模型，FP16 权重理论字节数为

$$
2P=13\,214\,687\,232\ \mathrm{bytes}.
$$

若按 16 bytes/parameter 的 Adam 类静态账估计：

$$
16P=105\,717\,497\,856\ \mathrm{bytes}.
$$

这只是参数、梯度、master 和 moments 的理论合计。训练进程还要加入激活、数据 batch、通信 bucket、temporary GEMM workspace 和 CUDA allocator overhead。

### 参数量与 checkpoint 文件大小

checkpoint 文件的大小可能包含：

- 权重张量；
- optimizer state；
- scheduler state；
- RNG state；
- tokenizer 或配置；
- quantization scale 和 zero point；
- shard index 与 metadata；
- padding 对齐字节。

所以文件大小除以 dtype 字节数不能直接反推出参数量。需要先确认文件中保存的是 inference-only 权重还是完整训练 checkpoint。

## KV cache 与参数账的边界

### decode 时的 cache shape

自回归 decode 为每层的历史 token 保存 K/V。若 K/V head 数为 $h_{kv}$，head 宽度为 $d_h$，batch 为 $B$，历史长度为 $T$，层数为 $L$，K/V cache 元素数为

$$
N_{\mathrm{KV}}
=2LBh_{kv}Td_h.
$$

乘以 dtype bytes 得到 cache 存储。

### GQA cache 例子

取

$$
L=32,
\qquad
B=1,
\qquad
h_{kv}=8,
\qquad
T=4096,
\qquad
d_h=128.
$$

全模型 K/V cache 元素数为

$$
N_{\mathrm{KV}}
=2\times32\times1\times8\times4096\times128
=268\,435\,456.
$$

FP16 下为

$$
268\,435\,456\times2
=536\,870\,912\ \mathrm{bytes}
=512\ \mathrm{MiB}.
$$

如果改用标准 MHA 的 $h_{kv}=32$，其他条件不变，cache 变为

$$
2\times32\times32\times4096\times128
=1\,073\,741\,824
$$

个 FP16 元素，即 $2048\ \mathrm{MiB}$。

KV cache 不是 trainable parameter。它随 batch、历史长度、层数、$h_{kv}$ 和 dtype 增长，在 decode 过程中动态产生。GQA/MQA 减少 cache 的 K/V head 数，但不会把模型权重中的 query 投影删掉。

## 计算量与实际运行量

### prefill 和 decode

prefill 一次处理整段序列。对 self-attention，长度项近似为

$$
C_{\mathrm{prefill,interaction}}
=2Bh_qT^2d_h.
$$

decode 每一步通常只有一个新 query，但会读取历史长度 $T$ 的 K/V：

$$
C_{\mathrm{decode,interaction/step}}
=2Bh_qTd_h.
$$

所以 decode 单步的 attention 交互量随历史长度线性增长，而不是一次完整 prefill 的平方项。两种阶段的参数量相同，计算和 KV cache 账本不同。

### MAC 与 FLOP

如果一次乘法和一次加法记为一个 MAC，则矩阵乘法的 MAC 与 FLOP 可能按以下约定转换：

$$
1\ \mathrm{MAC}=2\ \mathrm{FLOPs}.
$$

有些硬件文档把 fused multiply-add 直接记为 2 FLOPs，也有报告把一个 MAC 记作一个运算单位。比较两个数字前，必须写出定义。softmax、LayerNorm、SiLU、mask、量化解码和内存访问通常不在简单 GEMM MAC 中。

### 理论 MAC 不等于延迟

实际 latency 还受以下因素影响：

| 影响因素 | 对账本的影响 |
| --- | --- |
| kernel 融合 | 减少中间读写，未必减少矩阵乘法 MAC |
| tile 与 layout | 改变访存和 occupancy |
| padding | 可能继续计算无效 token |
| tensor parallel | 改变通信与局部矩阵 shape |
| quantization | 减少权重字节，但增加解码和 scale 访问 |
| batch/sequence packing | 改变有效 token 数和 kernel 利用率 |
| hardware alignment | 可能引入 padding work |

因此理论 MAC 可用于结构比较，不能单独替代设备测量。

## 参数量相同不等于模型可比

### 需要同时固定的维度

公平比较两个 Transformer 结构，至少记录：

- 层数 $L$；
- 模型宽度 $d$；
- FFN 中间宽度 $m$；
- attention 的 $h_q$、$h_{kv}$、$d_h$；
- 词表大小 $V$；
- 输入输出 embedding 是否 tying；
- bias、norm 类型和位置参数；
- context length、batch、dtype；
- 是否包含 optimizer state 和 KV cache。

只写「两个模型都是 7B」无法确认它们的 block 结构、词表、head 或权重绑定是否相同。

### 等参数比较与等计算比较

可以有三种不同的比较条件：

| 比较口径 | 固定对象 | 适用问题 |
| --- | --- | --- |
| 等参数 | $P_{\mathrm{model}}$ | 模型容量与存储预算 |
| 等 MAC | 每 token 或每 step MAC | 理论算力与吞吐 |
| 等显存 | 权重、激活、状态峰值 | 设备可运行性 |
| 等 wall-clock | 实际 batch、硬件、kernel 和延迟 | 部署效率 |

等参数不保证等 MAC。等 MAC 不保证等显存。等显存也不保证等 wall-clock。报告结论时要写明使用哪一种口径。

### 普通 FFN 和 SwiGLU 的公平边界

相同 $d,m$ 时，SwiGLU 参数和 MAC 约为普通 FFN 的 1.5 倍。若目标是等参数，需要缩小 SwiGLU 的 $m$；若目标是等中间宽度，则接受资源增加；若目标是等 wall-clock，还要考虑 fused kernel 和 hardware shape。

这三种比较回答不同问题，不能把一个结果迁移到另一个口径。

## 参数高效微调的第二本参数账

### LoRA 的增量参数

对冻结的矩阵

$$
W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
$$

LoRA 用

$$
\Delta W=BA,
\qquad
B\in\mathbb R^{d_{\mathrm{out}}\times r},
\qquad
A\in\mathbb R^{r\times d_{\mathrm{in}}}.
$$

新增可训练参数量为

$$
P_{\mathrm{LoRA}}=r(d_{\mathrm{out}}+d_{\mathrm{in}}).
$$

不应把隐含的乘积矩阵 $\Delta W$ 再加一次。推理时是否合并到 $W$ 不改变训练期间的可训练参数账，但会改变运行时存储路径。

### 一个 attention LoRA 例子

取

$$
d_{\mathrm{in}}=d_{\mathrm{out}}=4096,
\qquad
r=8.
$$

单个方阵 adapter 的参数量为

$$
P_{\mathrm{LoRA,one}}
=8(4096+4096)
=65\,536.
$$

若给 Q、K、V、O 四个投影各加一个 adapter：

$$
P_{\mathrm{LoRA,4}}
=4\times65\,536
=262\,144.
$$

冻结的 base model 仍有原来的总参数量，但 trainable parameter count 只增加 $262\,144$，还可能包含 adapter bias 或 scaling 参数。

### 训练账与推理账

参数高效微调至少要报告：

| 账本 | 需要记录 |
| --- | --- |
| base total params | 冻结模型的全部参数 |
| trainable params | adapter、head、norm 等实际更新的参数 |
| optimizer states | 只为 trainable 参数保存，还是为全模型保存 |
| inference params | 合并前后是否仍分开保存 |
| activation/MAC | adapter 是否增加额外路径和读写 |

只报告「可训练参数很少」不能推出训练显存一定按同样比例下降，因为 frozen base 的前向权重仍然需要加载，激活和通信也可能占主导。

## 分片、量化与部署存储

### sharding 不改变逻辑参数量

tensor parallel 把一个矩阵沿输入或输出轴切分。例如

$$
W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

被分到 $p$ 个 rank 后，每个 rank 可能保存约 $P(W)/p$ 个逻辑参数，另有 padding 和通信 buffer。所有 rank 合计的逻辑参数量仍为 $P(W)$。

报告分片时同时写：

- global logical parameter count；
- 每 rank local tensor shape；
- replicated 参数；
- shard padding；
- all-gather/reduce-scatter buffer；
- checkpoint shard metadata。

### 量化后的存储

对每组 $g$ 个参数使用 $b$ bits 的量化值，并为每组保存 scale 与 zero point。理论数值存储为

$$
M_{\mathrm{value}}=P\frac{b}{8}.
$$

若每组有一个 FP16 scale，scale 额外存储为

$$
M_{\mathrm{scale}}=\frac{P}{g}\times2\ \mathrm{bytes}.
$$

如果 zero point 也是一个 byte，则还要加入

$$
M_{\mathrm{zero}}=\frac{P}{g}\times1\ \mathrm{byte}.
$$

实际布局还需要考虑 group padding 和对齐。量化存储大小不能只用 $Pb/8$ 结束。

### checkpoint 计数的可复现性

核对量化 checkpoint 时，需要说明：

- 量化的是 weights、activations 还是 optimizer states；
- group size 和 per-tensor/per-channel 规则；
- scale、zero point 的 dtype；
- 是否保存原始 FP16/FP32 master；
- shard padding 和索引文件是否计入；
- embedding/head 是否共享。

否则同一个逻辑参数量可能对应不同文件大小和运行时显存。

## 失效模式

### 把参数量当作显存

参数量是标量个数。显存还包括 dtype 字节数、梯度、optimizer state、activation、KV cache、temporary buffer 和 metadata。只用「参数量乘 2 bytes」估计训练显存会漏掉主要状态。

### 把激活或 attention map 加到参数量

$BTm$ 中间数组和 $Bh_qT^2$ attention map 是运行时激活。它们不属于 checkpoint 的 trainable parameters。应在激活账单独记录。

### 把权重共享重复计数

weight tying、跨层共享和复用同一 embedding 时，逻辑引用次数不等于存储张量份数。参数账应按唯一可学习张量计数。

### 把 MAC 当作 FLOP

先说明一个 MAC 的定义，再换算 FLOPs。softmax、激活、归约和访存不一定包含在 GEMM MAC 中。

### 把 head 数和模型宽度混淆

在总宽度固定的标准 MHA 中，head 数改变分块，不自动改变四个投影的参数量。GQA/MQA 改变的是 K/V 投影宽度和 cache，需要使用 $h_{kv}$ 重新计算。

### 把 bias 静默省略

bias 可能相对矩阵参数很小，但精确对账仍需标注。LayerNorm 的 $\beta$、LM head bias 和 gate/up/down bias 都不能凭经验猜测。

### 把等中间宽度当作等参数

SwiGLU 和普通 FFN 使用同一 $m$ 时，参数量不同。比较性能时必须说明是等宽度、等参数、等 MAC 还是等 wall-clock。

### 把 optimizer state 当作模型参数

Adam moments 与 master weight 是运行时状态，不是模型 forward 的 trainable parameter count。保存 checkpoint 时可以包含它们，但必须分开列项。

### 把 KV cache 当作权重

KV cache 是 decode 期间由输入产生的动态激活。它与 $B$、历史长度、层数、$h_{kv}$ 和 dtype 相关，不应并入参数量。

### 把理论激活和峰值显存等同

四个 $BTm$ 数组的理论总和不代表 fused kernel 一定同时物化四份。峰值需要依据保存、重算、allocator 和并行策略确认。

### 忽略量化附加项

INT4/INT8 的 value bits 不包含 scale、zero point、padding 和 metadata。部署文件大小需要用实际 checkpoint 结构核对。

### 用 wall-clock 反推参数量

kernel 融合、访存、通信、频率、batch 和硬件利用率都会影响延迟。延迟不能唯一反推出参数量，必须回到 checkpoint shape 和配置文件。

## 一个可复用的参数量核验协议

### 记录配置

1. 记录 $L$、$d$、$m$、$V$、$L_{\max}$。
2. 记录 attention 的 $h_q$、$h_{kv}$、$d_h$。
3. 记录 FFN 是普通两矩阵还是 GLU/SwiGLU 三矩阵。
4. 记录所有 bias、norm 类型和位置。
5. 记录 embedding、LM head、position embedding 是否共享。
6. 记录 LoRA、adapter 或其他 trainable additions。
7. 记录 dtype、quantization、group size 和 shard layout。

### 枚举唯一张量

1. 为每个权重记录名称、shape、dtype 和是否 trainable。
2. 为共享张量只保留一个参数项，记录所有引用位置。
3. 为 bias、norm scale、norm bias 单独建项。
4. 为量化 scale、zero point 和 metadata 建立存储项，不混入 trainable count。
5. 用 shape 乘积计算每项标量数量。

### 复算四本账

1. 逐项相加得到逻辑参数量。
2. 乘以 dtype bytes 得到纯权重存储。
3. 按 token、序列和 batch 复算 MAC。
4. 按保存策略复算激活元素与 peak live bytes。
5. 单列 optimizer state、KV cache、通信和 workspace。
6. 说明 MAC/FLOP、MiB/MB 和是否包含 bias 的口径。

### 做交叉检查

1. 用 embedding shape 核对 $Vd$。
2. 用每层 block count 乘层数核对堆栈总量。
3. 用 untied/tied 差值核对是否多一个 $Vd$ 矩阵。
4. 用 GQA/MQA 与 MHA 的 K/V shape 核对 attention 差值。
5. 用 FP16/FP32 字节数核对 checkpoint 文件的数量级。
6. 用 KV cache 公式核对 batch、长度、层数和 $h_{kv}$ 的变化。
7. 用独立脚本复算关键数字，避免从文章中的中间数字再次推导。

## 相关词条

- [前馈网络](../transformer-components/feedforward/)
- [SwiGLU：门控 FFN](../transformer-components/swiglu-ffn/)
- [注意力复杂度](../attention/attention-complexity/)
- [LayerNorm 与残差](../transformer-components/layernorm-residuals/)
- [残差流](../transformer-components/residual-streams/)
- [混合精度训练](../training-nn/mixed-precision/)
- [完整 Transformer](../transformer-architectures/full-transformer/)
