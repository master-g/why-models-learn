---
title: "ALiBi：用距离惩罚外推上下文长度"
tags: ["why-models-learn"]
---

ALiBi（Attention with Linear Biases，带线性偏置的注意力）把位置信息直接加入 attention score。对第 $h$ 个 head，query 位置 $i$ 与 key 位置 $j$ 的距离越大，score 受到的负偏置越大。它不为每个位置生成向量，不旋转 query 和 key，也不改写 value；它只在 softmax 前把一个由距离决定的标量加入 score。这个接口使训练长度与推理长度可以不同，但长度外推仍受斜率、内容分数、数值精度和缓存索引共同约束。

对 causal self-attention，设允许读取的 key 满足 $j\le i$。ALiBi 的核心计算可以写成

$$
\begin{aligned}
Q_h &= XW_{Q,h},\\
K_h &= XW_{K,h},\\
V_h &= XW_{V,h},\\
B_h(i,j) &= -m_h(i-j),\\
S_{h,i,j} &= \frac{Q_{h,i,:}K_{h,j,:}^{\mathsf T}}{\sqrt{d_h}}+B_h(i,j)+M_{i,j},\\
A_{h,i,j} &= \operatorname{softmax}_{j}\bigl(S_{h,i,j}\bigr),\\
O_{h,i,:} &= \sum_j A_{h,i,j}V_{h,j,:}.
\end{aligned}
$$

其中 $m_h>0$ 是 head 的固定斜率，$M_{i,j}$ 是 causal mask、padding mask 或其他可见性约束。若 $j>i$，通常令 $M_{i,j}=-\infty$；若 $j\le i$，$M_{i,j}$ 可以为零。位置条件进入 score 的加法路径，value 仍沿原坐标参与加权求和。

![ALiBi 在 attention score 上加入随距离线性减小的 head-specific bias，近距离读取得到更高先验分数](/assets/transformer-components/svg/alibi.1.svg)

## 先固定位置条件的接口

### 位置不进入 hidden 向量

绝对位置编码通常先构造位置向量 $P_{p,:}$，再把它与 token embedding 相加或拼接。RoPE 则在得到 Q/K 后旋转每个二维坐标对。ALiBi 不经过这两条路径：

1. 输入 hidden 不增加一张位置表；
2. $W_Q$、$W_K$、$W_V$ 的投影顺序不变；
3. Q/K 的数值只由内容和线性投影决定；
4. 位置差在 score 上以一个 head-specific 标量出现；
5. V 不携带 ALiBi 的距离偏置。

因此，ALiBi 的位置路径是

$$
\text{内容 }X
\longrightarrow
Q,K,V
\longrightarrow
\frac{QK^{\mathsf T}}{\sqrt{d_h}}
\longrightarrow
\text{加入 }B_h
\longrightarrow
\operatorname{softmax}
\longrightarrow
AV.
$$

这里的“线性”指偏置随距离 $\Delta$ 线性变化：

$$
B_h(i,j)=-m_h\Delta(i,j).
$$

它不是把 attention 的序列复杂度从二次降为线性，也不是 linear attention。对 dense attention，所有 query-key 位置对仍然存在。

### causal 与双向 attention 的距离

在 causal self-attention 中，未来 key 已经被 mask 删除，所以允许区域内

$$
\Delta(i,j)=i-j,\qquad j\le i.
$$

在双向 self-attention 中，query 可以读取左右两侧的 key，距离应写成

$$
\Delta(i,j)=\lvert i-j\rvert.
$$

两种写法使用同一个负斜率，但 causal 写法不需要绝对值。实现若在 causal 模式中仍写成带符号的 $j-i$，就会让更远的历史位置得到正偏置，方向会反转。

## 线性距离惩罚如何改变 softmax

### 相同内容分数下的距离先验

先忽略 mask，并设一个 head 的内容分数为 $C_{i,j}$。加入 ALiBi 后

$$
S_{i,j}=C_{i,j}-m\Delta(i,j).
$$

同一个 query 比较两个 key 时，softmax 分母会抵消：

$$
\frac{A_{i,j_1}}{A_{i,j_2}}
=
\exp\left(
C_{i,j_1}-C_{i,j_2}
-m\bigl(\Delta(i,j_1)-\Delta(i,j_2)\bigr)
\right).
$$

当两个 key 的内容分数相同，距离差一个 token 的位置，其未归一化权重比为 $\exp(-m)$。距离差 $\delta$ 个 token 时，远处 key 相对近处 key 还要乘以 $\exp(-m\delta)$。这个因子是先验，不是硬 mask；内容分数足够大时，远处 key 仍然可以获得较高权重。

### 一个单 head 的数值例子

设 query 位于 $i=4$，允许读取 key $j=0,1,2,3,4$，内容点积分数全部为零，斜率取 $m=0.5$。按 key 从旧到新排列，距离、偏置和未归一化权重为

| key 位置 $j$ | 距离 $\Delta=4-j$ | bias $-0.5\Delta$ | $\exp(-0.5\Delta)$ |
| ---: | ---: | ---: | ---: |
| 0 | 4 | -2.0 | 0.135335283237 |
| 1 | 3 | -1.5 | 0.223130160148 |
| 2 | 2 | -1.0 | 0.367879441171 |
| 3 | 1 | -0.5 | 0.606530659713 |
| 4 | 0 | 0.0 | 1.000000000000 |

未归一化权重的总和为

$$
Z
=0.135335283237+0.223130160148+0.367879441171
+0.606530659713+1
=2.332875544269.
$$

归一化后，权重约为

$$
(0.058012217398,\,
0.095645976785,\,
0.157693556382,\,
0.259992720659,\,
0.428655528777).
$$

最后一个 key 的权重约为 $42.87\%$。最远 key 与最近 key 的未归一化权重比为 $\exp(-2)$，最近 key 比它大约 $7.389$ 倍。若最远 key 的内容分数比最近 key 高 $2$，内容项正好抵消这段距离惩罚。

### 半衰距离

定义半衰距离 $\Delta_{1/2}$ 为距离增加后，保持相同内容分数的未归一化权重下降到原来一半的位置：

$$
\exp(-m\Delta_{1/2})=\frac{1}{2}.
$$

因此

$$
\Delta_{1/2}=\frac{\ln 2}{m}.
$$

斜率越大，head 的局部先验越强，半衰距离越短。几个斜率的量级如下：

| 斜率 $m$ | 半衰距离 $\ln 2/m$ | 距离增加 10 的权重因子 |
| ---: | ---: | ---: |
| 0.5 | 1.386294361120 | 0.006737946999 |
| 0.125 | 5.545177444480 | 0.286504796860 |
| 0.03125 | 22.180709777919 | 0.731615628946 |
| 0.0078125 | 88.722839111676 | 0.924848813217 |

半衰距离不是模型的实际 receptive field。内容相似度、mask、层间残差和其他 head 的读取都会改变最终依赖范围。它只描述在内容分数相同的受控条件下，距离项的衰减尺度。

## head-specific slope 的调度

### 斜率为什么按 head 分配

如果所有 head 使用同一个 $m$，每个 head 都有相同的距离先验。ALiBi 通常给不同 head 分配不同斜率：

1. 大斜率 head 更偏向近邻 key；
2. 小斜率 head 保留更长的历史范围；
3. 内容分数仍可在两个 head 中产生不同的选择；
4. 多个尺度的距离先验共同进入 head concat 和输出投影。

这是一种固定的 head 多样性。它不等于训练后每个 head 自己学习出了一个可解释的距离半衰点。

### $H$ 为 2 的幂时的常用序列

当 head 数 $H$ 是 2 的幂，常用斜率可写成

$$
m_h=2^{-8h/H},
\qquad h=1,2,\ldots,H.
$$

以 $H=8$ 为例，斜率从 $0.5$ 递减到 $0.00390625$：

| head $h$ | $m_h$ | 半衰距离 |
| ---: | ---: | ---: |
| 1 | 0.5 | 1.386294361120 |
| 2 | 0.25 | 2.772588722240 |
| 3 | 0.125 | 5.545177444480 |
| 4 | 0.0625 | 11.090354888959 |
| 5 | 0.03125 | 22.180709777919 |
| 6 | 0.015625 | 44.361419555839 |
| 7 | 0.0078125 | 88.722839111676 |
| 8 | 0.00390625 | 177.445678223351 |

head 的编号只是实现接口。训练完成后，不能在推理时随意交换 slope 与 head，因为 Q/K/V 的参数已经与原来的 slope 共同适应。

### head 数不是 2 的幂时

非 2 的幂 head 数需要明确 slope 生成算法。常见实现先取不超过 $H$ 的最大 2 的幂 $H_0$，生成 $H_0$ 个 slope；再从 $2H_0$ 的序列中隔一个取一个，补齐剩余 head：

$$
H_0=2^{\lfloor\log_2H\rfloor}.
$$

以 $H=12$ 为例，先取 $H_0=8$ 的序列，再从 $H=16$ 的序列交错取前四项。不同实现可能反转 head 顺序，但必须在训练和推理中保持同一组 slope 与 head 的绑定。

## 位置、长度与 KV cache

### 为什么不需要位置表

绝对位置表通常只能直接提供训练时覆盖的索引。若训练最大位置是 $L_{\text{train}}-1$，推理时访问新索引就需要插值、扩表或其他外推规则。ALiBi 的距离函数对任意非负整数 $\Delta$ 都有定义：

$$
B_h(\Delta)=-m_h\Delta.
$$

因此从 $L_{\text{train}}$ 扩展到 $L_{\text{test}}$ 时，不需要创建新的位置向量，也不需要对已有位置表做插值。以 $L_{\text{train}}=2048$、$L_{\text{test}}=8192$ 为例，新的最大 causal 距离从 $2047$ 变为 $8191$，偏置仍可直接计算。

这只消除了位置参数的索引边界。更长上下文会改变训练中出现的距离分布、softmax 竞争范围和累计内容误差。斜率过大时，远处信息可能被系统性压低；斜率过小时，距离先验可能不足以维持局部性。长度外推需要独立评测，不能由“没有位置表”直接推出质量不变。

### decode 时的绝对位置

自回归 decode 维护一个 KV cache。设已经缓存 $L_{\text{past}}$ 个 token，当前 chunk 内第 $t$ 个 query 的绝对位置为

$$
i=L_{\text{past}}+t.
$$

缓存中第 $j$ 个 key 的绝对位置为 $j$ 时，causal ALiBi 距离是

$$
\Delta(i,j)=i-j=L_{\text{past}}+t-j.
$$

例如 $L_{\text{past}}=3$、当前 query 为 $t=0$、缓存 key 为 $j=1$，正确距离是 $3+0-1=2$。把当前 chunk 的局部位置 $t$ 直接代替绝对位置，会把这个距离误算为 $0-1$，从而改变 bias 的符号和大小。

KV cache 的主要合同如下：

| 状态 | 正确处理 | 需要避免的处理 |
| --- | --- | --- |
| cached K | 保持原始 K，不需要旋转或加位置向量 | 把已缓存 K 当成带绝对位置向量的表示 |
| 新 query | 用 $L_{\text{past}}+t$ 计算与 cache 的距离 | 只用 chunk 内的 $t$ |
| causal mask | 结合绝对位置判断可见性 | 把 cache 与新 chunk 当成独立序列 |
| beam reorder | 重排 K/V cache，slope 仍按 head 保持 | 只重排 token，不重排对应 cache |
| GQA/MQA | 每个 query head 读取所属 KV head 的 K/V，并使用自己的 $m_h$ | 把 query head slope 错绑到 KV head |

ALiBi 不需要 RoPE 的 cos/sin cache，也不需要保存“已旋转 K”或位置偏移状态。它仍然需要保存 cache 长度，并在每次 score 计算中使用正确的绝对位置差。

### packed sequence 与 padding

padding 不属于真实序列位置。若一个 batch 中样本长度不同，距离必须在每个样本的有效坐标系内计算，padding key 还必须被 mask。把 batch padding 后的列索引直接当作连续语义位置，会让样本末尾与后续 padding 之间产生虚假的距离。

packed sequence 通常携带每个样本的起点和长度。实现至少需要同时核对：

1. packed token 的位置是否在样本边界内重新编号；
2. bias 的距离是否跨样本归零；
3. padding query 的输出是否被排除；
4. causal mask 是否与 bias 使用同一套有效位置。

## mask、score 与 value 的顺序

ALiBi 的偏置必须在 softmax 前加入 score：

$$
\begin{aligned}
C_{h,i,j} &= \frac{Q_{h,i,:}K_{h,j,:}^{\mathsf T}}{\sqrt{d_h}},\\
S_{h,i,j} &= C_{h,i,j}-m_h\Delta(i,j)+M_{i,j},\\
A_{h,i,j} &= \operatorname{softmax}_{j}(S_{h,i,j}),\\
O_{h,i,:} &= \sum_j A_{h,i,j}V_{h,j,:}.
\end{aligned}
$$

下表区分四个对象的职责：

| 组件 | 输入 | 改变的对象 | 不负责的对象 |
| --- | --- | --- | --- |
| 内容 score | Q、K | token 内容相似度 | 位置距离先验 |
| ALiBi bias | $\Delta$、$m_h$ | score 的标量偏移 | V 的内容坐标 |
| mask | 可见性规则 | softmax 前的可读集合 | 近远偏好 |
| value aggregation | A、V | 输出内容混合 | score 的位置惩罚 |

若先做 softmax 再把 bias 加到权重上，结果不再等价于对 logits 加 bias。若把 bias 乘到 V，距离项会直接改变被读取的内容，语义也发生变化。

### mask 的有限值实现

工程实现有时使用一个有限负数代替 $-\infty$。此时要检查该值是否足够负，以及全 mask 行是否有特殊处理。ALiBi 的 bias 会继续减小有效位置的 score；若 mask sentinel 不够负，未来位置可能得到非零权重。若全 mask 行直接送入 softmax，输出可能出现 NaN 或被任意归一化。

## 与其他位置机制的差异

ALiBi、绝对位置编码和 RoPE 都可以让 attention 使用位置信息，但位置条件进入 score 的方式不同：

| 机制 | 位置条件进入路径 | Q/K/V 是否改写 | 长度外推依赖 | 主要位置表达 |
| --- | --- | --- | --- | --- |
| 绝对正弦编码 | 加到 hidden | Q/K/V 间接改变 | 公式可继续计算，但效果需评估 | 绝对位置向量 |
| 可学习位置嵌入 | 加到 hidden | Q/K/V 间接改变 | 受位置表长度约束 | 绝对位置参数 |
| RoPE | 旋转 Q/K | Q/K 改写，V 不变 | 受相位速度、base 和精度影响 | 相对相位与内容交互 |
| ALiBi | score 加 $-m_h\Delta$ | Q/K/V 不改写 | bias 对新距离有定义，但质量需评估 | 内容无关的距离先验 |
| 相对位置 bias | score 加位置函数 | 通常不改写 | 取决于距离桶或函数定义 | 相对距离类别或函数 |

ALiBi 与一般 relative position bias 都把位置条件放在 score 上。区别在于 ALiBi 使用每个 head 的单调线性函数；距离桶、学习表或其他函数可以表达非单调关系，但可能遇到桶范围或参数索引边界。

RoPE 的位置项会进入 $QK^{\mathsf T}$ 的乘积，因而与内容坐标发生交互。ALiBi 在内容 score 之外加一个独立标量，解释上更直接，表达能力也更受限。两者不能因为都依赖相对位置就互换实现。

## 资源账本

### 参数与状态

ALiBi 不增加与最大位置长度成正比的可学习位置参数。每个 head 只需要一个固定 slope。若实现把 bias 融合进 attention kernel，可以在计算 score 时按位置差生成偏置，不必保存完整 bias 矩阵。

设 batch 为 $B$，head 数为 $H$，query 和 key 长度都为 $T$。完整 bias 矩阵的元素数为

$$
N_{\text{bias}}=BHT^2.
$$

以 $B=2$、$H=32$、$T=4096$ 为例：

| 项目 | 数值 |
| --- | ---: |
| bias 元素数 $BHT^2$ | 1,073,741,824 |
| FP16 完整 bias | 2,147,483,648 bytes，约 2 GiB |
| FP32 完整 bias | 4,294,967,296 bytes，约 4 GiB |
| 固定 slope 数量 $H$ | 32 |

这个账本说明“没有位置表”不等于“可以免费物化完整 bias”。高效实现应避免额外的全尺寸 bias 写回，把距离计算融合到 score、mask 和 softmax 的 kernel 中。

### dense attention 的复杂度没有改变

ALiBi 没有删除 query-key 位置对。忽略投影和输出投影，dense attention 的 score 仍然包含

$$
BHT^2d_h
$$

量级的点积算术，以及 $BHT^2$ 个位置对上的 bias 和 softmax 操作。它解决的是位置条件的参数和外推路径，不是 attention 的二次位置对成本。

## 失效模式

实现审查应先确认数值方向，再确认索引、mask 和 kernel 边界。常见问题如下：

| 现象 | 可能的具体错误 | 最小核对 |
| --- | --- | --- |
| 远处 key 权重反而增大 | causal 距离写成 $j-i$，或 bias 符号写反 | 固定内容分数，检查 $\Delta$ 增大时 score 是否下降 |
| 近邻优势消失 | bias 在 softmax 后加入，或距离没有进入 logits | 对零内容分数行比较 $\exp(-m\Delta)$ |
| 所有 head 的读取范围相同 | slope 被广播成一个标量 | 打印每个 head 的 $m_h$ 和半衰距离 |
| 新长度推理出现突变 | 使用了训练长度的截断距离或错误 bucket | 直接核对 $\Delta=L_{\text{test}}-1$ 的 bias |
| decode 结果与 full pass 不同 | cache offset 没有加入 query 绝对位置 | 对同一 token 比较 full 与 cached 的 $\Delta$ |
| padding 附近出现异常权重 | padding key 未 mask，或 bias 与样本坐标跨边界 | 单独构造不同长度 batch |
| 全 mask 行出现 NaN | 有限 sentinel 与 softmax 重标度不兼容 | 测试全 mask query 行和 loss reduction |
| 长上下文只剩最近 token | 所有 slope 过大，或 head slope 顺序错误 | 计算各 head 的半衰距离并比较远距权重 |
| GQA 输出依赖错误 | query head 的 slope 错绑到 KV head | 检查 Q head 到 KV head 的映射和 bias head 维 |
| cross-attention 距离没有语义 | 未定义 target/source 两套坐标的对应关系 | 明确 query 与 source 的位置原点和距离函数 |

其中“远处 key 权重反而增大”是方向性错误，“长上下文只剩最近 token”则可能是正确方向下的斜率配置问题。两者的排查路径不同。

## 一个可复现的数值核验

用 $H=8$ 的第一、第二个 head，取 $m_1=0.5$、$m_2=0.25$。令 query 位于 $i=4$，key 位置为 $j=0,\ldots,4$，内容 score 全为零，causal mask 只保留这些 key。对第一 head，bias 和 softmax 结果应为

$$
\begin{aligned}
B_1 &= (-2,-1.5,-1,-0.5,0),\\
\operatorname{softmax}(B_1)
&=(0.058012217398,\,
0.095645976785,\,
0.157693556382,\,
0.259992720659,\,
0.428655528777).
\end{aligned}
$$

第二 head 的距离惩罚较弱：

$$
B_2=(-1,-0.75,-0.5,-0.25,0).
$$

它的最远到最近权重比为 $\exp(-1)$，第一 head 的对应比为 $\exp(-2)$。因此不同 slope 使 head 在相同内容分数下形成不同距离尺度。

核验程序至少应断言以下关系：

1. $m_h>0$；
2. causal 区域中 $\Delta(i,j)=i-j\ge0$；
3. $\Delta$ 增大一个 token 时，bias 减少 $m_h$；
4. softmax 的每行权重和为 1；
5. 将常数加到一行所有 logits 不改变 softmax；
6. cache offset 加入后，full pass 与 decode 的距离一致；
7. mask 后不可见位置权重为 0；
8. V 未被 bias 直接改写。

## 审计协议

实现或翻译完成后，按以下顺序检查：

| 阶段 | 检查内容 | 证据 |
| --- | --- | --- |
| 记号 | 明确 $B$、$H$、$T$、$d_h$、$i$、$j$、$m_h$ 的维度和索引范围 | 公式与 shape 表 |
| 方向 | 固定内容分数，距离增加时 bias 单调减小 | 两个位置的 logits |
| slope | 记录生成算法、head 顺序和半衰距离 | slope 向量 |
| mask | 在 softmax 前合并 causal、padding 和 bias | logits 与权重 |
| cache | 用绝对位置核对 full pass、prefill、decode | offset 对照 |
| 长度 | 比较训练长度和新长度的距离范围 | 长上下文评测 |
| 数值 | 覆盖 FP32、FP16、全 mask 行和极端距离 | 断言与异常日志 |
| 性能 | 区分 bias 计算、完整 materialization 和 HBM 搬运 | kernel 或 profiler |
| 内容 | 检查 value 路径和输出 projection 未被误改 | Q/K/V 中间值 |

只检查短序列内的 score 方向，无法确认长度外推。只检查 full pass，无法确认 KV cache offset。只检查输出正确率，无法定位 slope、mask 和 value 路径的责任边界。

## 相关词条

[位置编码](../transformer-components/positional-encoding/)

[RoPE](../transformer-components/rope/)

[自注意力](../attention/self-attention/)

[因果掩码](../attention/causal-masking/)

[注意力矩阵](../attention/attention-matrix/)
