---
title: "LLaMA 2：从配置到前向的 decoder-only 复现"
tags: ["why-models-learn"]
---

LLaMA 2 不是一条孤立的网络层，而是一份由 tokenizer、模型配置、权重布局、位置规则、decoder block、KV cache 和生成策略共同组成的执行合同。实现一个能运行的 decoder-only Transformer 只需要把这些组件串起来；实现一个能加载并复现 checkpoint 的 LLaMA 2 风格模型，还需要让每个 shape、索引、转置、位置偏移和特殊 token 都与 checkpoint 的合同一致。

本文把 LLaMA 2 作为 LLaMA 风格 decoder-only Transformer 的具体案例，重点复现从 token id 到 next-token logits 的前向路径。文中的小配置只用于核对 shape、参数量、缓存容量和局部数值，不代表官方 checkpoint 的层数、宽度或词表大小。官方权重、tokenizer 文件和部署 kernel 仍然需要按实际 checkpoint 的配置读取，不能从模型名称推断。

整条路径可以压缩为：

$$
\begin{aligned}
\text{文本}
&\rightarrow
\text{token id}
\rightarrow
\text{embedding},\\
&\rightarrow
\text{decoder blocks},\\
&\rightarrow
\text{final norm}
\rightarrow
\text{LM head}
\rightarrow
\text{logits}.
\end{aligned}
$$

每个 block 的 attention 使用 RoPE 处理 query/key，使用 causal mask 限制未来位置，使用 MHA、GQA 或 MQA 约定 K/V head；前馈支路使用 SwiGLU。自回归 decode 把每层历史 K/V 写入 cache，下一步只计算新 token 的 query、key、value，再读取已有 cache。

![LLaMA 2 风格 decoder-only Transformer 的配置、GQA、RoPE、SwiGLU、KV cache 和 logits 前向路径](/assets/alignment/svg/llama2-from-scratch.1.svg)

## 复现目标是合同，不是重新训练官方模型

### 先固定输入和输出

设 batch size 为 $B$，输入长度为 $T$，模型宽度为 $D$，词表大小为 $V$，decoder block 数为 $L$。tokenizer 把文本变成整数 id 后，模型的主要张量合同为：

| 对象 | shape | 产生位置 | 后续消费者 |
| --- | --- | --- | --- |
| token id | $(B,T)$ | tokenizer 与特殊 token 规则 | embedding 查表 |
| hidden state | $(B,T,D)$ | embedding 与 residual stream | 每个 decoder block |
| query | $(B,h_q,T,d_h)$ | attention 的 Q projection | attention score |
| key/value | $(B,h_{kv},T,d_h)$ | attention 的 K/V projection | attention 读取与 KV cache |
| logits | $(B,T,V)$ | final RMSNorm 与 LM head | softmax、loss 或采样 |

行向量实现通常把线性层权重存成 (out_features, in_features)，因此输入 $X\in\mathbb R^{B\times T\times D}$ 使用

$$
Q_{\mathrm{flat}}=XW_Q^{\mathsf T},
\qquad
K_{\mathrm{flat}}=XW_K^{\mathsf T},
\qquad
V_{\mathrm{flat}}=XW_V^{\mathsf T}.
$$

再把最后一轴拆成 head：

$$
\begin{aligned}
Q&=\operatorname{reshape}(Q_{\mathrm{flat}})
 \in\mathbb R^{B\times h_q\times T\times d_h},\\
K&=\operatorname{reshape}(K_{\mathrm{flat}})
 \in\mathbb R^{B\times h_{kv}\times T\times d_h},\\
V&=\operatorname{reshape}(V_{\mathrm{flat}})
 \in\mathbb R^{B\times h_{kv}\times T\times d_h}.
\end{aligned}
$$

这条合同明确了权重的存储方向。若另一套实现使用列向量，公式中的转置位置会变化；只转置某几层会得到 shape 看似正确、数值却不一致的模型。

### “从零”覆盖哪些层

本文的复现范围包括：

| 层级 | 本文固定的内容 | 不由本文单独保证的内容 |
| --- | --- | --- |
| 输入 | tokenizer 输出的 token id、BOS/EOS、position id 和 loss 对齐 | 官方 tokenizer 文件的字节级一致性 |
| 配置 | $D$、$L$、$h_q$、$h_{kv}$、$d_h$、$m$、$\epsilon$ 和 RoPE base | 某个具体 checkpoint 的实际字段值 |
| block | pre-norm、causal attention、RoPE、GQA、SwiGLU、residual | 设备相关 fused kernel 的舍入顺序 |
| 推理 | prefill、decode、KV cache、next-token logits | 采样器、服务调度和吞吐优化 |
| 资格判断 | shape、参数账本、局部数值、缓存布局和回归样本 | 官方 benchmark 或完整训练复现 |

因此，前向复现通过只说明“这份配置和权重能按同一计算图运行”。它不自动说明 tokenizer、训练语料、微调数据、采样规则和公开评测结果都一致。

## 模型配置决定计算图

### 需要一起读取的字段

LLaMA 风格实现通常把模型宽度、层数、head 数、词表和数值开关放在配置中。不同实现的字段名可以不同，下面使用语义名：

| 语义字段 | 记号 | 作用 | 形状或单位 |
| --- | --- | --- | --- |
| vocabulary size | $V$ | embedding 与输出词表的行数 | token 数 |
| model dimension | $D$ | residual stream 的宽度 | hidden 特征数 |
| layer count | $L$ | decoder block 的重复次数 | 层数 |
| query head count | $h_q$ | 独立 query 与 score 的数量 | head 数 |
| K/V head count | $h_{kv}$ | key/value 表示与 cache 的数量 | head 数 |
| head dimension | $d_h$ | 每个 head 的最后一轴 | $D/h_q$ |
| intermediate width | $m$ | SwiGLU gate/up 的输出宽度 | hidden 特征数 |
| normalization epsilon | $\epsilon$ | RMSNorm 除法的稳定项 | 无量纲 |
| RoPE base | $\theta_{\mathrm{base}}$ | inverse frequency 的底数 | 无量纲 |
| maximum sequence length | $T_{\max}$ | position 与 cache 的上限 | token 数 |

一个配置可以把 $m$ 写成最终的整数，也可以由 multiple_of、扩展比和可选 multiplier 计算。加载 checkpoint 时应记录计算后的有效 $m$，因为参数量和权重 shape 依赖这个整数，而不是依赖某个未展开的公式。

### 形状不变量

常见 dense head 配置至少满足：

$$
D=h_qd_h,
\qquad
h_q\bmod h_{kv}=0,
\qquad
D_{kv}=h_{kv}d_h.
$$

RoPE 按二维坐标对旋转，因此标准实现还要求

$$
d_h\bmod 2=0,
\qquad
T\leq T_{\max}.
$$

| 检查 | 通过条件 | 违反时的结果 |
| --- | --- | --- |
| query 拆 head | $D=h_qd_h$ | Q reshape 不能保持元素总数 |
| GQA 分组 | $h_q/h_{kv}$ 为整数 | query head 无法确定 K/V 组 |
| RoPE 配对 | $d_h$ 为偶数 | 最后一个坐标没有旋转伙伴 |
| position 范围 | $T\leq T_{\max}$ | position 或 RoPE 表越界 |
| residual 写回 | 每个子层输出最后一轴为 $D$ | 残差加法 shape 不匹配 |

这些检查应在加载权重前执行。等到第一次矩阵乘法报错时，通常已经丢失了发生错误的字段和权重名。

## 输入端：tokenizer、embedding 和位置

### token id 不是字符编号

tokenizer 把文本分割成 token，并把每个 token 映射到固定整数。这个映射由 tokenizer 模型、词表文件、normalizer、special token 和版本共同决定。相同字符串在两个 tokenizer 中得到的 id 可以不同；相同 id 在两个词表中也可能对应不同 token。

对输入 id $I\in\{0,\ldots,V-1\}^{B\times T}$，embedding 矩阵为

$$
E\in\mathbb R^{V\times D}.
$$

行查表得到

$$
H_{0,b,t,:}=E_{I_{b,t},:}.
$$

decoder-only 模型通常不把一张 learned absolute position table 加到 hidden，而是把 position id 用在 RoPE 的 Q/K 旋转中。具体 checkpoint 仍可能有额外位置开关，所以应以配置和权重为准。

### BOS、EOS、PAD 和 loss mask

特殊 token 的含义必须同时写入 tokenizer 和训练/推理协议：

| token | 输入职责 | 生成或 loss 职责 |
| --- | --- | --- |
| BOS | 在需要时作为第一个条件 token | 通常不是独立目标 |
| EOS | 表示一条序列或模板段结束 | 生成命中后可以停止，是否计入 loss 由数据合同决定 |
| PAD | 把变长 batch 补成矩形 | attention key、query、loss 和 cache 都需要单独处理 |
| 文本 token | 由 embedding 变成 hidden | 作为后续位置的 next-token 标签或生成结果 |

右移后的输入和标签可以写成：

$$
X_t=I_t,
\qquad
Y_t=I_{t+1},
\qquad
t=0,\ldots,T-2.
$$

causal mask 只限制 attention 的可见性，loss mask 决定哪些标签进入分母。把 PAD 从 attention 中屏蔽，不会自动把 PAD 从 loss 中删除。

### 输出头是否绑定 embedding

若输出头与输入 embedding 绑定，使用同一个矩阵的转置：

$$
\operatorname{logits}=H_{\mathrm{final}}E^{\mathsf T}.
$$

若输出头独立，则使用 $W_{\mathrm{out}}\in\mathbb R^{V\times D}$：

$$
\operatorname{logits}=H_{\mathrm{final}}W_{\mathrm{out}}^{\mathsf T}.
$$

权重绑定减少一份 $VD$ 参数，但会改变 checkpoint 的 key 集合和输出头的自由度。不能因为两个矩阵 shape 相同，就把独立 output weight 默认为 tied weight。

## 一个 LLaMA 风格的 pre-norm block

### 残差流先后通过两条子层

设第 $\ell$ 层的输入为 $X_\ell\in\mathbb R^{B\times T\times D}$，RMSNorm 为 $N_{\ell,\mathrm{attn}}$ 和 $N_{\ell,\mathrm{ffn}}$。pre-norm decoder block 为：

$$
\begin{aligned}
U_\ell
&=X_\ell+
 \operatorname{Attention}_\ell
 \left(N_{\ell,\mathrm{attn}}(X_\ell)\right),\\
X_{\ell+1}
&=U_\ell+
 \operatorname{SwiGLU}_\ell
 \left(N_{\ell,\mathrm{ffn}}(U_\ell)\right).
\end{aligned}
$$

attention 和 SwiGLU 都只产生 $(B,T,D)$ 的 residual update。attention 在 token 轴读取其他位置；SwiGLU 固定每个 $(b,t)$ 后只在特征轴计算。

最后一层之后再执行 final RMSNorm：

$$
H_{\mathrm{final}}=N_{\mathrm{final}}(X_L).
$$

然后把它送入输出头。RMSNorm 是归一化类型，pre-norm 是归一化相对 residual 子层的位置；这两个开关必须分开记录。

### residual 顺序改变数值

把同样的权重改成 post-norm：

$$
\begin{aligned}
U_\ell
&=N_{\ell,\mathrm{attn}}
 \left(X_\ell+\operatorname{Attention}_\ell(X_\ell)\right),\\
X_{\ell+1}
&=N_{\ell,\mathrm{ffn}}
 \left(U_\ell+\operatorname{SwiGLU}_\ell(U_\ell)\right).
\end{aligned}
$$

这不只是把一个函数调用换到另一行。两种结构的 residual stream、梯度 Jacobian、最终尺度和 checkpoint 载入位置都不同。加载 LLaMA 风格权重时，norm 的 key 仍可能相似，真正的判断依据是计算图中的调用顺序。

## Attention：GQA、RoPE 和 causal mask

### Q、K、V 的宽度不同

采用 row-major 权重存储时，projection 权重的 shape 为：

| 权重 | shape | 逻辑 head 数 | 用途 |
| --- | --- | --- | --- |
| $W_Q$ | $(h_qd_h,D)$ | $h_q$ | 为每个 query head 产生独立 Q |
| $W_K$ | $(h_{kv}d_h,D)$ | $h_{kv}$ | 产生可共享的 K |
| $W_V$ | $(h_{kv}d_h,D)$ | $h_{kv}$ | 产生可共享的 V |
| $W_O$ | $(D,D)$ | 无 | 把拼接后的 attention 输出写回 residual stream |

GQA 的组大小为

$$
r=\frac{h_q}{h_{kv}},
\qquad
g(q)=\left\lfloor\frac{q}{r}\right\rfloor,
\qquad
q=0,\ldots,h_q-1.
$$

第 $q$ 个 query head 使用第 $g(q)$ 个 K/V head。相同 K/V 不意味着相同 score：每个 query head 仍然有自己的 $Q^{(q)}$。

### RoPE 只旋转 Q 和 K

设 head dimension 为偶数，把第 $i$ 个二维坐标对写成

$$
x_i=
\begin{bmatrix}
x_{2i}\\
x_{2i+1}
\end{bmatrix}.
$$

位置 $p$ 的旋转角度为

$$
\alpha_{p,i}=p\omega_i,
\qquad
\omega_i=\theta_{\mathrm{base}}^{-2i/d_h},
\qquad
i=0,\ldots,\frac{d_h}{2}-1.
$$

旋转矩阵为

$$
R(\alpha)=
\begin{bmatrix}
\cos\alpha&-\sin\alpha\\
\sin\alpha&\cos\alpha
\end{bmatrix}.
$$

对 Q/K 执行

$$
\widetilde q_{p,i}=R(p\omega_i)q_{p,i},
\qquad
\widetilde k_{s,i}=R(s\omega_i)k_{s,i},
\qquad
\widetilde v_{s,i}=v_{s,i}.
$$

因此 value 不携带 RoPE 旋转，attention score 为

$$
A^{(q)}_{p,:}
=\operatorname{softmax}
\left(
\frac{\widetilde q_p^{(q)}
(\widetilde K^{(g(q))})^{\mathsf T}}{\sqrt{d_h}}
+M_{p,:}
\right).
$$

M 同时承接 causal mask 和有效 key mask。长度为 $T$ 的无 padding 序列中，位置 $p$ 只允许读取索引不大于 $p$ 的 key。

### KV cache 保存什么

prefill 时得到整段的旋转后 K 和未旋转 V。decode 时把当前 token 的新 K/V 追加到每层 cache：

$$
\begin{aligned}
K_{\mathrm{cache}}^{(\ell)}
&\in\mathbb R^{B\times h_{kv}\times S\times d_h},\\
V_{\mathrm{cache}}^{(\ell)}
&\in\mathbb R^{B\times h_{kv}\times S\times d_h}.
\end{aligned}
$$

这里的 $S$ 是当前历史长度，不是模型层数 $L$。本文采用“cache 中的 K 已经完成 RoPE、V 保持原值”的合同；如果实现保存未旋转 K，则下一次读取前必须以原 position id 重新旋转，二者不能混用。

GQA 的收益主要来自 K/V head 数下降：

$$
N_{\mathrm{cache}}
=2BLh_{kv}Sd_h.
$$

其中因子 2 表示 K 和 V 两份状态。若每个元素占 $b$ bytes，则

$$
\operatorname{bytes}_{\mathrm{cache}}
=2BLh_{kv}Sd_hb.
$$

GQA 不减少 query head 的逻辑 score 数量。它减少 K/V projection 的宽度和 cache 容量，不能被表述为“所有 attention 计算都按 $h_{kv}$ 缩小”。

## RMSNorm 和 SwiGLU：两个非 attention 子层

### RMSNorm 只沿当前 token 的特征轴归约

对 $x\in\mathbb R^D$，RMSNorm 为

$$
r(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2+\epsilon},
\qquad
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{r(x)}.
$$

它不减去均值，也不把历史 cache 拼进归约集合。每个 $(b,t)$ 独立计算自己的 $r(x_{b,t,:})$。LLaMA 风格的规范实现通常只有 $\gamma$，没有 LayerNorm 的 $\beta$。

### SwiGLU 用三份投影

对归一化后的 token hidden $Z\in\mathbb R^{B\times T\times D}$，无 bias 的 SwiGLU 为

$$
\begin{aligned}
G&=ZW_g^{\mathsf T},
&W_g&\in\mathbb R^{m\times D},\\
U&=ZW_u^{\mathsf T},
&W_u&\in\mathbb R^{m\times D},\\
A&=\operatorname{SiLU}(G)\odot U,\\
F&=AW_d^{\mathsf T},
&W_d&\in\mathbb R^{D\times m}.
\end{aligned}
$$

gate 和 up 读取同一个 token，down 把中间宽度 $m$ 压回 $D$。这条支路没有 token 之间的 attention 读取；上下文已经由 attention 写入 residual stream。

### final norm 与 block norm 不是同一个参数

每个 block 有两条 RMSNorm 参数向量，final RMSNorm 还有一条独立向量：

| 参数位置 | 数量 | 读取时机 | 能否复用 |
| --- | ---: | --- | --- |
| attention norm | $D$ | attention projection 之前 | 不能与其他层默认复用 |
| FFN norm | $D$ | SwiGLU projection 之前 | 不能与 attention norm 默认复用 |
| final norm | $D$ | 所有 block 之后、LM head 之前 | 不能与最后一层 FFN norm 默认复用 |

把 final norm 漏掉通常不会造成 shape 错误，但会造成 logits 数值和 checkpoint 输出不一致。

## 参数账本：从小配置核对完整模型

### 单个 block 的参数量

假设所有线性层无 bias。attention 四份矩阵的参数量为

$$
\begin{aligned}
P_{\mathrm{attn}}
&=D(h_qd_h)+D(h_{kv}d_h)+D(h_{kv}d_h)+D^2\\
&=2D^2+2DD_{kv},
\qquad
D_{kv}=h_{kv}d_h.
\end{aligned}
$$

SwiGLU 三份矩阵的参数量为

$$
P_{\mathrm{SwiGLU}}=3Dm.
$$

两个 block 内 norm 各有 $D$ 个参数，所以

$$
P_{\mathrm{block}}
=2D^2+2DD_{kv}+3Dm+2D.
$$

若 embedding 和 output head 绑定，完整模型的参数量为

$$
P_{\mathrm{tied}}
=VD+LP_{\mathrm{block}}+D.
$$

若 output head 独立，则多出一份 $VD$：

$$
P_{\mathrm{untied}}
=P_{\mathrm{tied}}+VD.
$$

这份账本只计算模型参数，不包括 optimizer state、梯度、激活、workspace、量化 scale 或通信 buffer。

### toy 配置的逐项结果

取一个能手算的配置：

| 字段 | $V$ | $D$ | $L$ | $h_q$ | $h_{kv}$ | $d_h$ | $m$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| toy value | 10 | 8 | 2 | 4 | 2 | 2 | 16 |

此时 $D_{kv}=4$，GQA 每组包含 $r=2$ 个 query head。逐项账本为：

| 项目 | 公式 | toy value |
| --- | --- | ---: |
| attention projection | $2D^2+2DD_{kv}$ | 192 |
| SwiGLU projection | $3Dm$ | 384 |
| 两个 block norm | $2D$ | 16 |
| 一个 decoder block | $P_{\mathrm{attn}}+P_{\mathrm{SwiGLU}}+2D$ | 592 |
| tied 模型 | $VD+LP_{\mathrm{block}}+D$ | 1272 |
| untied 模型 | $P_{\mathrm{tied}}+VD$ | 1352 |

如果把 $h_{kv}$ 改成 $h_q$，attention projection 变成 MHA；如果令 $h_{kv}=1$，变成 MQA。FFN 和 norm 参数不随 GQA 的共享规则改变。

## Prefill、decode 和 next-token logits

### 训练时一次前向计算整段前缀

训练时输入右移后的 $I_{0:T-2}$，一次性计算所有位置的 logits。causal mask 使位置 $t$ 只能读取当前及过去的 token，loss 读取对应的 $I_{1:T-1}$。这称为 teacher forcing：

$$
\mathcal L
=-\frac{1}{N_{\mathrm{valid}}}
\sum_{b,t}
m_{b,t}\log
p_\theta(I_{b,t+1}\mid I_{b,\leq t}),
$$

其中 $m_{b,t}$ 是 loss mask，$N_{\mathrm{valid}}$ 是有效 target token 数。

### 推理时区分 prefill 和 decode

| 阶段 | 输入长度 | attention 读取 | cache 操作 | 输出 |
| --- | --- | --- | --- | --- |
| prefill | prompt 的全部 $T$ 个 token | 计算 prompt 内的 causal score | 写入每层全部历史 K/V | 最后位置的 logits |
| decode | 一个新 token | 新 query 读取历史与当前 K/V | 追加一个位置 | 当前 token 的 next-token logits |
| 长度增长 | 每步加 1 | query head 数不变，key 长度增长 | cache 沿序列轴增长 | 直到 EOS 或长度上限 |

prefill 仍然是训练式的整段矩阵计算；decode 不能把 prompt 的所有 token 再次投影一遍。KV cache 只保存 attention 需要跨步复用的 K/V，不保存 residual stream、SwiGLU 中间激活或最终 logits。

### toy cache 和 logits 账本

对 toy 配置取 $B=1$、$T=3$、FP16 每个元素 2 bytes：

$$
\begin{aligned}
N_{\mathrm{logits}}
&=BTV=30,\\
N_{\mathrm{next\text{-}token}}
&=BV=10,\\
N_{\mathrm{score}}
&=Bh_qT^2=36,\\
N_{\mathrm{causal\ visible}}
&=Bh_q\frac{T(T+1)}{2}=24,\\
\operatorname{bytes}_{\mathrm{cache}}
&=2BLTh_{kv}d_h\cdot2=96.
\end{aligned}
$$

score 的 36 个位置对是逻辑 dense 张量的大小；causal mask 只让其中 24 个位置进入有效读取。实际 kernel 是否物化完整矩阵，还要看 FlashAttention 或其他实现。

## Checkpoint 兼容性：shape 对上还不够

### 先建立权重 key 和 shape 清单

下面是常见 LLaMA 风格命名的语义示意。实际库可能把名称改成 layers.0.self_attn.q_proj.weight 等，但 shape 和职责必须一一对应：

| 权重 key 语义 | 存储 shape | 前向用途 | 常见错误 |
| --- | --- | --- | --- |
| token embedding | $(V,D)$ | token id 行查表 | 把 vocab 轴和 hidden 轴交换 |
| Q projection | $(h_qd_h,D)$ | 产生 query | 用 $h_{kv}$ 截断 query |
| K/V projection | $(h_{kv}d_h,D)$ | 产生共享 K/V | 按 MHA 的 $D$ 行加载 GQA 权重 |
| output projection | $(D,D)$ | attention 拼接后写回 | 忘记输出投影或权重转置错误 |
| gate/up projection | $(m,D)$ | SwiGLU 两条上投影 | 交换 gate 与 up 的 key |
| down projection | $(D,m)$ | 压回 residual 宽度 | 把 $(m,D)$ 当作存储 shape |
| attention/FFN norm | $(D)$ | 两个子层前归一化 | 复用不同层或交换 norm |
| final norm | $(D)$ | output head 前归一化 | 漏掉或误用最后一层 norm |
| output head | $(V,D)$ 或不存储 | 产生词表 logits | 误判 tied/untied |

加载时应同时记录 key、shape、dtype、设备和是否需要转置。只打印“所有权重都已加载”不能证明每个 weight 进入了正确的算子。

### tokenizer、config 和 checkpoint 是三件事

| 资产 | 决定什么 | 不能由什么替代 |
| --- | --- | --- |
| tokenizer 文件 | 文本到 token id、特殊 token 和词表行 | 不能由模型 config 推断 |
| model config | 层数、宽度、head、norm、RoPE 和词表尺寸 | 不能由权重总参数量唯一反推 |
| model checkpoint | embedding、各层矩阵、norm 和 output head 的数值 | 不能由同结构随机初始化替代 |
| generation config | temperature、top-p、top-k、stop 和最大长度 | 不能由 logits 自动确定 |

因此，模型“能生成文本”只证明四份接口已经足以执行一次请求；它不证明加载了目标 checkpoint，也不证明生成过程与原始推理配置一致。

## 一个可运行的最小核验探针

### 探针只使用 Python 标准库

下面的代码不实现高性能 tensor kernel，而是用标准库核对四类容易漂移的账本：head 维度、参数量、causal score、KV cache，以及 RMSNorm、SiLU 和 RoPE 的局部数值。

```python
from math import cos, exp, sin, sqrt

V = 10
D = 8
L = 2
h_q = 4
h_kv = 2
d_h = 2
m = 16
B = 1
T = 3
bytes_per = 2
assert D == h_q * d_h
assert h_q % h_kv == 0
D_kv = h_kv * d_h

attn_params = 2 * D * D + 2 * D * D_kv
swiglu_params = 3 * D * m
block_params = attn_params + swiglu_params + 2 * D
embedding_params = V * D
tied_params = embedding_params + L * block_params + D
untied_params = tied_params + V * D

logits_elements = B * T * V
next_logits_elements = B * V
causal_score_elements = B * h_q * T * T
visible_score_elements = B * h_q * T * (T + 1) // 2
kv_cache_bytes = 2 * B * L * T * h_kv * d_h * bytes_per

x = [1.0, 2.0, 3.0, 4.0]
eps = 1e-5
r = sqrt(sum(value * value for value in x) / len(x) + eps)

def silu(value):
    return value / (1.0 + exp(-value))

def rotate(pair, position, theta):
    angle = position * theta
    c, s = cos(angle), sin(angle)
    a, b = pair
    return [c * a - s * b, s * a + c * b]

q = [1.0, 2.0]
q_rot = rotate(q, 3, 0.5)

print("config_dims=", {
    "V": V, "D": D, "L": L, "h_q": h_q, "h_kv": h_kv,
    "d_h": d_h, "m": m, "B": B, "T": T,
})
print("head_group_size=", h_q // h_kv, "D_kv=", D_kv)
print("attention_params=", attn_params,
      "swiglu_params=", swiglu_params,
      "block_params=", block_params)
print("tied_params=", tied_params, "untied_params=", untied_params)
print("logits_elements=", logits_elements,
      "next_logits_elements=", next_logits_elements)
print("causal_score_elements=", causal_score_elements,
      "visible_score_elements=", visible_score_elements)
print("kv_cache_bytes_fp16=", kv_cache_bytes)
print("rmsnorm_r=", f"{r:.12f}", "silu_1=", f"{silu(1.0):.12f}")
print("rope_norm_before=",
      f"{sqrt(sum(v * v for v in q)):.12f}",
      "after=", f"{sqrt(sum(v * v for v in q_rot)):.12f}")
```

输出应为：

```text
config_dims= {'V': 10, 'D': 8, 'L': 2, 'h_q': 4, 'h_kv': 2, 'd_h': 2, 'm': 16, 'B': 1, 'T': 3}
head_group_size= 2 D_kv= 4
attention_params= 192 swiglu_params= 384 block_params= 592
tied_params= 1272 untied_params= 1352
logits_elements= 30 next_logits_elements= 10
causal_score_elements= 36 visible_score_elements= 24
kv_cache_bytes_fp16= 96
rmsnorm_r= 2.738614613267 silu_1= 0.731058578630
rope_norm_before= 2.236067977500 after= 2.236067977500
```

### 探针结果只证明局部合同

这个探针核对的是代数和 shape，不包含真实矩阵乘法、tokenizer、随机采样、量化误差、设备 kernel 或完整 checkpoint。它适合在加载真实模型之前捕捉以下错误：

| 探针信号 | 说明 | 仍需补充的证据 |
| --- | --- | --- |
| $D_{kv}$ 与 cache 字节数 | GQA 的 K/V 宽度使用了正确 head 数 | 真实权重 key 和 cache layout |
| tied 与 untied 参数差 | output head 的绑定决策进入账本 | checkpoint 是否真的存储 output weight |
| RoPE 前后范数相同 | 旋转矩阵保持局部长度 | position offset、频率表和轴顺序 |
| causal visible 数量 | mask 的下三角计数正确 | padding、packed sequence 和 kernel 实际 mask |
| RMSNorm、SiLU 数值 | 局部公式与 epsilon 约定正确 | 全模型多层前向和 dtype 舍入 |

## 常见失效模式

### 用模型名称填配置

同一系列可能包含不同层数、宽度、K/V head 数、词表和上下文设置。模型名称只提供索引，不能替代 checkpoint 配置。应先读取配置，再用权重 shape 交叉检查。

### 把 GQA 当成缩小版 MHA

GQA 只减少 K/V head。query head、score 的逻辑数量和 residual 宽度仍由 $h_q$、$T$、$D$ 决定。把 Q projection 也按 $h_{kv}$ 截断，会得到错误的 query 表示。

### RoPE 位置偏移少一位

prefill 后的第一个 decode token 使用已有 cache 长度作为 position id。若把它重新从零开始旋转，当前 Q 和历史 K 不在同一位置坐标系中，输出会在短序列上看似合理、在长 prompt 上逐步偏离。

### cache 中的 K 是否已旋转没有写入合同

“保存 K”不足以描述 cache。必须记录 K 是投影后、RoPE 后还是量化后，以及下一步是否再次旋转。重复 RoPE 或漏掉 RoPE 都可能保留相同 shape。

### 把 SwiGLU 当成普通两矩阵 FFN

SwiGLU 有 gate、up、down 三份投影。漏掉 gate 或交换 gate/up 的权重，参数量可能仍被手工代码掩盖，但输出和梯度都不一致。

### 把 RMSNorm 与 pre-norm 混为一个开关

RMSNorm 决定归一化算子，pre-norm 决定归一化相对子层的位置。使用 RMSNorm 的 post-norm block 与 LLaMA 风格 pre-norm block 不是同一前向函数。

### 只检查 logits shape

错误的 tokenizer、输出头绑定、权重转置、mask、位置和 dtype 都可能保留 $(B,T,V)$ 的 logits shape。资格核验至少要包含单层中间张量、参数 key、局部数值和短序列逐 token 回归。

## 运行方法

把“最小核验探针”代码保存为 llama2_probe.py，在 Python 3 环境运行：

```bash
python3 llama2_probe.py
```

加载真实 checkpoint 时，按以下顺序执行：

1. 读取 tokenizer 和 model config，记录文件版本、词表大小、特殊 token、$D$、$L$、$h_q$、$h_{kv}$、$d_h$、$m$、$\epsilon$ 和 RoPE base。
2. 按 key 和 shape 枚举 checkpoint，检查 embedding、Q/K/V/O、gate/up/down、两类 block norm、final norm 和 output head。
3. 用一个长度为 1 的 token 核对 embedding，再用长度为 2 或 3 的输入核对 causal attention、RoPE 和 residual shape。
4. 分别执行 prefill 和逐 token decode，比较两条路径最后位置的 logits；两者应在约定的 dtype 容差内一致。
5. 固定 tokenizer、模板、position、mask、dtype 和生成配置，再保存短 prompt 的逐步 logits 与 token 回归。

没有真实 checkpoint、tokenizer 和设备运行证据时，不能把 toy probe 报告为 LLaMA 2 的完整复现结果。

## 相关词条

- [Decoder-Only：因果自回归 Transformer](../transformer-architectures/decoder-only/)：固定 decoder-only 的输入、causal mask、LM head 和 KV cache 接口。
- [GQA 与 MQA](../attention/gqa-and-mqa/)：展开 query head 与 K/V head 的组映射、参数量和缓存容量。
- [RoPE](../transformer-components/rope/)：推导 Q/K 的旋转、相对相位、位置偏移和缓存约定。
- [RMSNorm](../transformer-components/rmsnorm/)：推导均方根归一化、epsilon、统计轴和残差接口。
- [SwiGLU](../transformer-components/swiglu-ffn/)：展开 gate/up/down 三矩阵、SiLU 门控和局部 Jacobian。
- [Pre-norm 与 Post-norm](../transformer-components/pre-norm-vs-post-norm/)：比较归一化位置、残差路径和最终 norm。
- [参数量与资源账本](../transformer-components/parameter-count/)：扩展参数、激活、MAC、optimizer state 和 KV cache 的资源核算。
- [Causal Language Modeling](../transformer-architectures/causal-language-modeling/)：说明右移输入、next-token 最大似然和有效 loss 分母。
- [推理数学](../inference/inference-math/)：补充 logits、temperature、top-k、top-p 和采样概率的数值合同。
- [量化](../inference/quantization/)：处理低 bit 权重、scale、误差和量化后 KV cache 的资源边界。
