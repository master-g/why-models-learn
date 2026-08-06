---
title: "RoPE：用 query/key 的相位表达相对位置"
tags: ["why-models-learn"]
---

RoPE（rotary positional embedding，旋转位置编码）把每个 attention head 的 query 和 key 拆成二维坐标对，再按各自的绝对位置旋转这些坐标。位置 $p$ 的 query 使用角度 $p\theta_r$，位置 $s$ 的 key 使用角度 $s\theta_r$；两者的点积会合并成位移 $s-p$ 的旋转。RoPE 保持向量范数，保持 Q/K 的 shape，也不需要一张可学习的绝对位置表。

对 head 内的 query、key、value，典型计算顺序是

$$
\begin{aligned}
q_p &= W_Qh_p,\\
k_s &= W_Kh_s,\\
v_s &= W_Vh_s,\\
\widetilde q_p &= \operatorname{RoPE}(q_p,p),\\
\widetilde k_s &= \operatorname{RoPE}(k_s,s),\\
\operatorname{score}(p,s)
&=\frac{\widetilde q_p^\mathsf T\widetilde k_s}{\sqrt{d_h}}+M_{p,s},\\
a_{p,s} &= \operatorname{softmax}_{s}\bigl(\operatorname{score}(p,s)\bigr),\\
o_p &= \sum_s a_{p,s}v_s.
\end{aligned}
$$

旋转只作用于 query 和 key。value 保留承载内容的原坐标，mask 仍然独立规定哪些位置可读。[位置编码](../transformer-components/positional-encoding/)词条建立了绝对位置输入和 padding 合同；这篇把位置条件放到 Q/K score 的具体路径中，再处理频率、长度、缓存、精度和核验方法。

![RoPE 对 query 和 key 的二维坐标对施加位置相关旋转，使 attention score 依赖相对相位](/assets/transformer-components/svg/rope.1.svg)

## 先固定旋转约定

### 二维旋转矩阵

把一个二维列向量写成

$$
x_r=
\begin{bmatrix}
x_{2r}\\
x_{2r+1}
\end{bmatrix}.
$$

采用逆时针旋转的约定：

$$
R(\alpha)
=
\begin{bmatrix}
\cos\alpha&-\sin\alpha\\
\sin\alpha&\cos\alpha
\end{bmatrix}.
$$

位置 $p$ 和频率 $\theta_r$ 共同决定角度：

$$
\alpha_{p,r}=p\theta_r.
$$

RoPE 对第 $r$ 对坐标执行

$$
\widetilde x_{p,r}
=R(p\theta_r)x_r.
$$

实际实现把所有坐标对拼回原来的 head 向量：

$$
\operatorname{RoPE}(x,p)
=
\operatorname{concat}_{r=0}^{d_h/2-1}
\left[R(p\theta_r)x_r\right].
$$

这里要求 $d_h$ 为偶数。若 head dimension 为奇数，最后一个坐标如何处理必须写入实现合同；直接假定每个坐标都能配对会产生 shape 或语义错误。

### 旋转保持长度

因为

$$
R(\alpha)^\mathsf TR(\alpha)=I,
$$

所以

$$
\lVert R(\alpha)x_r\rVert_2
=\lVert x_r\rVert_2.
$$

对所有坐标对拼接后仍有

$$
\lVert\operatorname{RoPE}(x,p)\rVert_2
=\lVert x\rVert_2.
$$

同一个向量在不同位置旋转后，单独的范数不变；不同位置的两个向量之间的点积会改变。这个区分是检查实现的重要基线。

### 频率安排

常见的 inverse frequency 规则为

$$
\theta_r
=\operatorname{base}^{-2r/d_h},
\qquad
r\in\left\{0,\ldots,\frac{d_h}{2}-1\right\}.
$$

许多实现把 $\operatorname{base}=10000$ 作为默认值，也有实现为长上下文调整 base 或频率。base 和坐标切片方式属于模型配置；加载 checkpoint 时必须与训练配置一致。

## 点积为什么变成相对位置

### 两个位置的旋转可以合并

令未旋转的 query 和 key 为 $q_p$、$k_s$。对一个频率对：

$$
\widetilde q_{p,r}=R(p\theta_r)q_{p,r},
\qquad
\widetilde k_{s,r}=R(s\theta_r)k_{s,r}.
$$

两者点积为

$$
\begin{aligned}
\widetilde q_{p,r}^\mathsf T\widetilde k_{s,r}
&=q_{p,r}^\mathsf TR(p\theta_r)^\math TR(s\theta_r)k_{s,r}\\
&=q_{p,r}^\mathsf TR((s-p)\theta_r)k_{s,r}.
\end{aligned}
$$

因为 $R(a)^\mathsf TR(b)=R(b-a)$，绝对位置在点积中合并成相对位移 $s-p$。把全部频率对相加：

$$
\widetilde q_p^\mathsf T\widetilde k_s
=
\sum_{r=0}^{d_h/2-1}
q_{p,r}^\mathsf TR((s-p)\theta_r)k_{s,r}.
$$

这个等式只说明位置旋转如何进入 Q/K 点积。$q_{p,r}$ 和 $k_{s,r}$ 仍然由内容 hidden 经过投影得到，最终 score 仍是内容与相对相位的共同结果。

### 一个二维频率对的展开

令

$$
q_r=
\begin{bmatrix}
a\\
b
\end{bmatrix},
\qquad
k_r=
\begin{bmatrix}
c\\
d
\end{bmatrix},
\qquad
\delta=s-p.
$$

则

$$
\begin{aligned}
q_r^\math TR(\delta\theta_r)k_r
={}&(ac+bd)\cos(\delta\theta_r)\\
&+(bc-ad)\sin(\delta\theta_r).
\end{aligned}
$$

第一项是未旋转点积按相对相位缩放，第二项来自二维方向的有符号交互。不同坐标对使用不同 $\theta_r$，因此一个 head 同时获得多种位移尺度。

### 数字例子

取

$$
q=(1,2,3,4),
\qquad
k=(5,6,7,8),
\qquad
p=2,
\qquad
s=5,
\qquad
\theta=(1,0.01).
$$

按每两个坐标一对旋转后：

| 对象 | 第 0 维 | 第 1 维 | 第 2 维 | 第 3 维 |
| --- | ---: | ---: | ---: | ---: |
| 未旋转 $q$ | 1.000000000 | 2.000000000 | 3.000000000 | 4.000000000 |
| 旋转后 $\widetilde q_2$ | -2.234741690 | 0.077003754 | 2.919405353 | 4.059196027 |
| 未旋转 $k$ | 5.000000000 | 6.000000000 | 7.000000000 | 8.000000000 |
| 旋转后 $\widetilde k_5$ | 7.171856575 | -3.092648261 | 6.591418469 | 8.339856269 |

未旋转点积为

$$
q^\mathsf Tk=70.
$$

旋转后点积为

$$
\widetilde q_2^\mathsf T\widetilde k_5
=36.830741379538.
$$

直接用相对位移 $\delta=3$ 的两个二维旋转计算，也得到 $36.830741379538$。query 和 key 各自的范数都保持不变：

$$
\lVert q\rVert_2
=\lVert\widetilde q_2\rVert_2
=5.477225575052.
$$

## RoPE 放在 Q/K 路径，不放在 value 路径

### 典型计算图

对输入 hidden $H$，单个 head 的计算可分成：

$$
\begin{aligned}
Q&=HW_Q,&K&=HW_K,&V&=HW_V,\\
\widetilde Q&=\operatorname{RoPE}(Q,I),&
\widetilde K&=\operatorname{RoPE}(K,I),&
\widetilde V&=V.
\end{aligned}
$$

然后执行

$$
O
=
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{\widetilde Q\widetilde K^\mathsf T}{\sqrt{d_h}}+M
\right)V.
$$

旋转后的 Q/K 影响每个 query 读取哪些 key；V 仍然提供被加权读出的内容。如果把 V 也按位置旋转，输出的坐标系会随 key 位置变化，语义已经变成另一种位置机制，不能再用上面的 RoPE score 推导覆盖。

### 旋转时机改变计算图

下面三种操作的输入不同：

| 位置注入方式 | 旋转对象 | score 依赖 | 需要单独核验 |
| --- | --- | --- | --- |
| Q/K 投影后旋转 | $Q,K$ | 相对相位进入点积 | 当前 RoPE 合同 |
| hidden 相加后再投影 | $H+P$ | 绝对向量进入 $W_Q,W_K$ | [位置编码](../transformer-components/positional-encoding/)合同 |
| score 上加 bias | score 矩阵 | 直接加入位置差函数 | 相对 bias 合同 |

实现中若把 RoPE 放在 Q/K 投影前，旋转会被 $W_Q$、$W_K$ 改写；它可以构成另一种模型，但不再等于标准的 Q/K rotary path。

### 缩放和 mask 的顺序

旋转保持 Q/K 的二范数，所以通常仍使用 scaled dot-product attention 的缩放：

$$
\operatorname{score}(p,s)
=
\frac{\widetilde q_p^\mathsf T\widetilde k_s}{\sqrt{d_h}}
+M_{p,s}.
$$

mask 在 softmax 前加入。causal mask、padding mask 和 RoPE 共同出现在同一个 score 计算中，但职责不同：

1. RoPE 改变可见位置之间的匹配分数；
2. mask 删除不可见连接；
3. softmax 在剩余 key 轴上归一化；
4. value 按权重混合内容。

全 mask 行、query padding 行和 loss reduction 仍要按照[因果掩码](../attention/causal-masking/)与[交叉注意力](../attention/cross-attention/)的约定处理。

## 频率、周期和长上下文

### 每个频率有自己的相位周期

第 $r$ 个频率的周期为

$$
\lambda_r
=\frac{2\pi}{\theta_r}
=2\pi\cdot\operatorname{base}^{2r/d_h}.
$$

以 $d_h=4$、$\operatorname{base}=10000$ 为例：

| 频率编号 $r$ | $\theta_r$ | 周期 $\lambda_r$ |
| ---: | ---: | ---: |
| 0 | 1.000000000 | 6.283185307180 |
| 1 | 0.010000000 | 628.318530717959 |

低编号频率变化快，能区分短位移的相位；高编号频率变化慢，能在更长范围内提供平滑变化。单个频率的相位会循环，多组频率共同决定完整向量。

### base 是长度合同的一部分

增大 base 会让多数频率变化更慢，改变训练和推理时同一位置差对应的相位。修改 base、重排频率或只旋转部分 head dimension，都可能改变 checkpoint 的 score 分布。

长度扩展方法可能对位置 ID 做缩放：

$$
p\longrightarrow \widehat p,
\qquad
\alpha_{p,r}=p\theta_r
\longrightarrow
\widehat p\,\widehat\theta_r.
$$

只要 $\widehat p$ 或 $\widehat\theta_r$ 的规则发生变化，就要把它当作位置机制变体，分别评估训练内长度和训练外长度。RoPE 能计算某个长位置的 sine/cosine，不等于后续层已经在该相位分布上训练过。

### 相位混叠需要单独测量

某一频率满足

$$
(p+\lambda_r)\theta_r
=p\theta_r+2\pi,
$$

因此该频率对的旋转回到原来的方向。完整 RoPE 向量包含多个频率，所有频率同时回到同一方向的周期通常很长；有限精度、长度分布和内容投影仍可能使长距离 score 变得难以区分。

长上下文测试应记录：

| 观测 | 作用 |
| --- | --- |
| 位置向量的 $\sin/\cos$ 范围 | 检查数值是否出现 NaN 或异常截断 |
| Q/K 旋转前后范数 | 检查旋转矩阵和 dtype |
| 固定内容下的 score 与 $s-p$ | 检查相对相位方向 |
| 不同长度的 score 分位数 | 检查长位置分布漂移 |
| full 与 decode 的 logits | 检查 cache offset 和旋转时机 |

## KV cache、GQA 和增量解码

### cache 中保存已经旋转的 key

假设历史 key 的真实位置是 $s$。可以在写入 cache 时计算

$$
\widetilde k_s=R(s\theta)k_s
$$

并把 $\widetilde k_s$ 存入 KV cache。当前 query 的真实位置为 $p$，只需计算

$$
\widetilde q_p=R(p\theta)q_p
$$

再与 cache 中的 $\widetilde k_s$ 做点积。不要在读取 cache 时再次对已经旋转的 key 施加 $R(s\theta)$。

另一种实现可以保存未旋转的 K 和 position ID，在 score kernel 内按需旋转。两种方案都可用，但 cache 内容的旋转状态必须明确。

### cache offset 不能从本轮张量长度推断

历史长度为 $L_{\mathrm{past}}$，本轮输入有 $U$ 个 token 时：

$$
p_i=L_{\mathrm{past}}+i,
\qquad
i\in\{0,\ldots,U-1\}.
$$

如果误用 $p_i=i$，本轮 query 会与历史 key 使用错误的相对相位。full prefill 和带 cache 的 chunk prefill 可能在第一个新 token 就产生不同 logits。

### GQA/MQA 共享 K/V，不共享 position ID

在 GQA 中，多个 query head 共享一个 K/V head。共享的是 K/V 表示和 cache 存储，不代表不同 query 位置可以共用 position ID。对同一时间位置 $s$，共享 K/V head 的 key 使用同一个 $R(s\theta)$；不同时间位置仍然使用不同旋转。

MQA 令全部 query head 共享一组 K/V。它减少 KV cache 容量，但不改变 RoPE 对每个 query/key 时间位置的旋转合同。[GQA 和 MQA](../attention/gqa-and-mqa/)词条处理 head 映射和 cache shape。

### beam reorder 必须保持位置状态

beam search 选择新候选后，通常需要重排 decoder hidden、KV cache 和与 cache 对齐的辅助状态。若 K 已经旋转，重排只移动对应的已旋转行；若 K 尚未旋转，重排 position ID 和 K 必须保持一致。把 beam index 应用到其中一项而漏掉另一项，会产生内容正确、位置错误的 score。

## 计算量、内存和精度

### 旋转开销相对 score 计算

每个二维向量需要 4 次乘法和 2 次加法。Q 和 K 都旋转时，每个 token、每个 head、每个 head dimension 约需 6 个标量算术操作：

$$
C_{\mathrm{rot}}
\approx6Bh_qTd_h.
$$

以 $B=2$、$h_q=32$、$T=4096$、$d_h=128$ 为例：

$$
C_{\mathrm{rot}}
=201,326,592.
$$

同一配置下，QK score 的 MAC 为

$$
C_{QK}
=Bh_qT^2d_h
=137,438,953,472.
$$

RoPE 没有改变 dense attention 的 $T^2$ 位置对；它增加逐 token 的 Q/K 旋转，并改变 score 的相位结构。

如果为每个位置缓存 cos/sin，缓存形状通常是

$$
[T,d_h/2]
$$

的两张表。上述 $T=4096,d_h=128$ 时，FP16 的两张表合计约 1 MiB；这份表可在 batch 和 head 之间复用。实现也可以在 kernel 内按位置计算，代价则转移到算术和特殊函数。

### 低精度下检查正交性

理论上旋转保持范数，FP16/BF16 的有限精度会产生舍入误差。应分别记录

$$
\Delta_{\mathrm{norm}}
=
\frac{\left\lvert\lVert\widetilde x\rVert_2-\lVert x\rVert_2\right\rvert}
{\lVert x\rVert_2+\varepsilon}.
$$

这里使用 $\lvert\cdot\rvert$ 表示标量绝对值，避免把 raw pipe 写入 Markdown 表格或数学区。误差阈值必须与 dtype、序列长度和 kernel 累加顺序一起确定。

常见实现会用 FP32 计算或缓存 inverse frequency、sin/cos，再把旋转结果转换到 Q/K 的计算 dtype。这个选择会改变误差分布，但不改变 RoPE 的数学定义。

## 和其他位置机制比较

| 机制 | 注入位置 | 位置表达 | 长度边界 | 主要审计点 |
| --- | --- | --- | --- | --- |
| 固定正弦位置编码 | 输入 hidden | 绝对向量，多频率相位 | 公式可计算，模型外推仍需验证 | 输入 shape、相加尺度 |
| 可学习位置表 | 输入 hidden | 每个绝对位置一行参数 | $L_{\max}$ 有硬边界 | 查表索引、训练长度 |
| RoPE | Q/K | 相对位移对应的多频率旋转 | base、频率和外推规则 | 旋转符号、Q/K 时机、cache offset |
| relative position bias | score | 位置差查表或函数 | bias 表或函数的长度边界 | score 加法和 mask 顺序 |
| ALiBi | score | 距离乘 head-specific slope | 依赖 slope 与长度分布 | slope、方向和长距离 score |

这些机制可以在同一模型族中被替换，但替换后需要重新核对 checkpoint、score 统计、mask、cache 和长度评估。[ALiBi](../transformer-components/alibi/)词条处理把距离偏置直接加入 score 的路径。

## 常见失效模式

| 失效模式 | 直接症状 | 检查动作 |
| --- | --- | --- |
| 把 V 也旋转 | 输出 value 坐标随 key 位置改变 | 对比只旋转 Q/K 与旋转 Q/K/V 的结果 |
| 在 Q/K 投影前旋转 | 与标准 RoPE score 不一致 | 固定 hidden，分别比较投影前后两种计算图 |
| 旋转矩阵符号反了 | $s-p$ 变成 $p-s$，某些任务仍可能部分通过 | 用非对称的 q/k 数值例子核对正负方向 |
| Q/K 坐标配对错位 | 范数可能正确，score 仍错误 | 检查偶数/奇数维度切片和 head 维度 |
| cache key 双重旋转 | decode 与 full prefill 不一致 | 记录 cache 写入前后的旋转状态 |
| cache offset 从零开始 | 只有带历史的 decode 出现位置偏差 | 比较 $p=L_{\mathrm{past}}+i$ 与 $p=i$ |
| base 或频率改变 | 长度变化时 score 分布漂移 | 固定 checkpoint，记录 inverse frequency 和相位 |
| 低精度相位误差累积 | 长位置范数或 logits 偏差增大 | 比较 FP32、FP16/BF16 的 norm 与 logits |
| mask 与位置混淆 | 不可见位置仍影响输出 | 独立检查 mask 前后的 score 和 softmax 行 |
| beam 只重排部分 cache | beam 结果内容对齐、位置错位 | 同时重排 K/V、position ID 和辅助状态 |

错误可能只在 decode、左 padding、长上下文或某个 head 出现。单独比较短序列的最终 token 不足以覆盖这些边界。

## 核验协议

对一个 RoPE 实现，至少执行以下检查：

1. 记录 head dimension、坐标配对顺序、旋转矩阵符号、base、频率公式和 position ID 起点。
2. 用 $p=0$ 检查旋转是否为恒等变换。
3. 用同一个 $x$ 在多个位置旋转，核对范数不变。
4. 用非对称 q/k 和 $p\ne s$ 的例子，核对旋转点积等于 $R((s-p)\theta)$ 的相对位移公式。
5. 固定 q/k 内容，改变 $s-p$，记录每个频率对的 score 变化。
6. 确认 value 没有被旋转，mask 在 softmax 前仍然生效。
7. 分别测试右 padding、左 padding、packed sequence 和 cache offset。
8. 比较 full prefill、chunk prefill 和增量 decode 的有效位置 logits。
9. 若使用 GQA/MQA 或 beam search，核对共享 K/V、cache 重排和 position ID 的对应关系。
10. 在 FP32 与实际计算 dtype 下比较 norm 误差、score 误差和长序列分位数。

数字基线可以使用前文例子：

| 检查项 | 结果 |
| --- | ---: |
| 未旋转 $q^\mathsf Tk$ | 70.000000000000 |
| $p=2$ 的旋转 query 第 0 维 | -2.234741690199 |
| $s=5$ 的旋转 key 第 0 维 | 7.171856575295 |
| 相对位移 $s-p=3$ 的旋转点积 | 36.830741379538 |
| $\lVert q\rVert_2=\lVert\widetilde q_2\rVert_2$ | 5.477225575052 |
| $B=2,h_q=32,T=4096,d_h=128$ 的 Q/K 旋转算术量 | 201,326,592 |
| 同配置 QK score MAC | 137,438,953,472 |

这些检查先验证旋转和索引，再验证 attention 和 cache；它们不能替代真实任务上的长上下文评估。

## 相关词条

[位置编码](../transformer-components/positional-encoding/)

[自注意力](../attention/self-attention/)

[缩放点积注意力](../attention/scaled-dot-product/)

[因果掩码](../attention/causal-masking/)

[GQA 和 MQA](../attention/gqa-and-mqa/)

[ALiBi](../transformer-components/alibi/)
