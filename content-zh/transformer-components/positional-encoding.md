---
title: "位置编码：把序列位置写进 Transformer 表示"
tags: ["why-models-learn"]
---

位置编码（positional encoding）是与 token 表示绑定的位置标记。它把第 $p$ 个 token 所处的顺序位置写入向量，使 Transformer 的自注意力能够区分相同 token 出现在不同位置的情况。对第 $p$ 个位置，最常见的输入接口是

$$
h_p=e_p+p_p,
$$

其中 $e_p\in\mathbb R^d$ 是 token embedding，$p_p\in\mathbb R^d$ 是位置向量，$h_p$ 才是送入后续层的 hidden state。位置编码规定了 $p_p$ 如何产生、位置索引从哪里开始、padding 和 cache 如何处理，以及长度超过训练范围时能否继续计算。

这篇先建立绝对位置编码的共同接口，再推导正弦位置编码的频率与相对位移性质，比较固定编码和可学习位置表，最后把位置索引、attention score、padding、缓存偏移和长度扩展放进同一份核验协议。RoPE 和 ALiBi 的具体计算分别在[RoPE](../transformer-components/rope/)和[ALiBi](../transformer-components/alibi/)词条展开。

![位置编码把 token embedding 与 position signal 相加，再影响正弦相位和 attention 位置对](/assets/transformer-components/svg/positional-encoding.1.svg)

## 没有位置时，注意力看见的是集合

### 自注意力本身不提供顺序

设一批 hidden state 按位置排成矩阵

$$
X\in\mathbb R^{T\times d}.
$$

没有位置特征时，单头 self-attention 可以写成

$$
\begin{aligned}
Q&=XW_Q,\\
K&=XW_K,\\
V&=XW_V,\\
A&=\operatorname{softmax}_{\mathrm{row}}\left(\frac{QK^\mathsf T}{\sqrt{d_k}}\right),\\
O&=AV.
\end{aligned}
$$

令 $\Pi$ 是交换序列位置的置换矩阵。把输入换成

$$
X'=\Pi X
$$

会同时得到

$$
Q'=\Pi Q,
\qquad
K'=\Pi K,
\qquad
V'=\Pi V.
$$

于是 score 和输出满足

$$
\begin{aligned}
Q'K'^\mathsf T
&=\Pi QK^\mathsf T\Pi^\mathsf T,\\
A'
&=\Pi A\Pi^\mathsf T,\\
O'
&=\Pi O.
\end{aligned}
$$

输出会跟着输入位置一起交换。模型能读取 token 内容和 token 之间的匹配关系，但仅凭这条计算图无法判断「甲在乙前面」还是「乙在甲前面」。[自注意力](../attention/self-attention/)需要额外的位置输入才能表达顺序。

### 一个最小反例

把三个 token 的 embedding 记为 $a,b,c$。没有位置向量时，序列

$$
(a,b,c)
\qquad\text{和}\qquad
(c,b,a)
$$

只相差一个位置置换。每个 token 仍然可以读取同一组内容；变化只体现在输出行的排列。若任务要求读取「前一个 token」、区分前缀和后缀，或判断两个 token 的相对距离，计算图还缺少位置条件。

位置编码为每个序列槽位增加一个随 $p$ 改变的向量：

$$
e_p\longrightarrow h_p=e_p+p_p.
$$

此后 token 的内容和它所在的槽位共同进入 $Q$、$K$、$V$ 投影。

## 绝对位置编码的接口

### 位置表的形状

对 batch-first 的输入，token embedding 可以写成

$$
E\in\mathbb R^{B\times T\times d}.
$$

若当前 batch 中的有效位置都使用 $0,1,\ldots,T-1$，位置矩阵为

$$
P_{0:T}\in\mathbb R^{T\times d}.
$$

按最后一维相加时，$P_{0:T}$ 会沿 batch 维广播：

$$
H_{b,p,:}=E_{b,p,:}+P_{p,:}.
$$

因此位置编码不改变 Transformer 的 model width。它把输入仍然保持在 $\mathbb R^{B\times T\times d}$ 中，后续的 Q/K/V 投影和 residual shape 不需要因为位置标记改变接口。

更一般地，位置 ID 矩阵可以是

$$
I\in\mathbb N^{B\times T},
$$

再通过位置表 gather：

$$
H_{b,p,:}=E_{b,p,:}+P_{I_{b,p},:}.
$$

这个接口能表达左 padding、packed sequence、cache offset 和每条样本不同的起始位置。位置 ID 不是 token ID；前者索引位置表，后者索引 token embedding。

### 位置索引从零还是从一开始

工程实现通常从 $p=0$ 开始，因为第一个位置可以直接对应数组下标 0。也可以从 $p=1$ 开始，但训练和推理必须使用同一约定。对于正弦编码，整体平移一个位置会改变每个维度的相位；对于可学习位置表，平移会直接换一行参数。

把一个训练好的模型从零起点改成一起点，会同时改变：

1. token 与 position vector 的配对；
2. 长度边界处最后一个位置的索引；
3. 增量解码时 cache offset 的计算；
4. 任何依赖绝对位置的 attention score。

这是模型输入协议的变化，不能只在 tokenizer 配置中改一个数字。

## 正弦位置编码

### 两个维度共享一个频率

设 $d$ 为偶数，位置 $p\in\{0,\ldots,L-1\}$，频率编号 $r\in\{0,\ldots,d/2-1\}$。经典正弦位置编码定义为

$$
\begin{aligned}
\operatorname{PE}_{p,2r}
&=\sin\left(p\omega_r\right),\\
\operatorname{PE}_{p,2r+1}
&=\cos\left(p\omega_r\right),\\
\omega_r
&=10000^{-2r/d}.
\end{aligned}
$$

也可以把每一对维度写成相位向量：

$$
\operatorname{PE}_{p,2r:2r+2}
=
\begin{bmatrix}
\sin(p\omega_r)\\
\cos(p\omega_r)
\end{bmatrix}.
$$

$r=0$ 的频率为 $1$，相邻的正弦和余弦维度每隔约 $2\pi$ 个位置转一周。随着 $r$ 增大，频率下降，波长变长。这里的「多频率」让短距离变化和长距离变化同时进入同一个位置向量。

### $d=4$ 的数字例子

当 $d=4$ 时，$\omega_0=1$、$\omega_1=0.01$。位置向量为：

| 位置 $p$ | 第 0 维 | 第 1 维 | 第 2 维 | 第 3 维 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.000000000 | 1.000000000 | 0.000000000 | 1.000000000 |
| 1 | 0.841470985 | 0.540302306 | 0.009999833 | 0.999950000 |
| 2 | 0.909297427 | -0.416146837 | 0.019998667 | 0.999800007 |

每一对正弦/余弦维度都满足

$$
\sin^2(p\omega_r)+\cos^2(p\omega_r)=1.
$$

所以 $d=4$ 的每个位置向量范数都为 $\sqrt 2\approx1.414213562373$。相邻位置 $p=1$ 和 $p=2$ 的点积为

$$
\operatorname{PE}_1^\mathsf T\operatorname{PE}_2
=\cos(1)+\cos(0.01)
\approx1.540252306285.
$$

这个数字同时可以由表中的四个分量直接相乘再相加得到。

### 频率和波长

第 $r$ 对维度的角频率为 $\omega_r$，对应位置周期为

$$
\lambda_r=\frac{2\pi}{\omega_r}
=2\pi\cdot10000^{2r/d}.
$$

在 $d=4$ 的例子中：

| 频率编号 $r$ | $\omega_r$ | 周期 $\lambda_r$ |
| ---: | ---: | ---: |
| 0 | 1 | 6.283185307180 |
| 1 | 0.01 | 628.318530717959 |

在 $d=1024$ 时，最后一对维度的周期约为 $61711.679832711$。频率范围覆盖得很宽，位置向量可以同时提供快速变化的局部相位和缓慢变化的长程相位。

### 平移可以写成二维旋转

对固定频率 $\omega$ 和位移 $k$，有

$$
\begin{bmatrix}
\sin((p+k)\omega)\\
\cos((p+k)\omega)
\end{bmatrix}
=
\begin{bmatrix}
\cos(k\omega)&\sin(k\omega)\\
-\sin(k\omega)&\cos(k\omega)
\end{bmatrix}
\begin{bmatrix}
\sin(p\omega)\\
\cos(p\omega)
\end{bmatrix}.
$$

右侧中间的矩阵只依赖位移 $k$，不依赖起点 $p$。因此固定频率下，从位置 $p$ 移动到 $p+k$ 可以由一个二维线性变换表示。这个性质说明正弦编码包含相对位移的可计算结构。

同一频率对的点积也只依赖位置差：

$$
\begin{aligned}
&\sin(p\omega)\sin(q\omega)
+\cos(p\omega)\cos(q\omega)\\
&=\cos((p-q)\omega).
\end{aligned}
$$

把所有频率对相加，未经过 Q/K 投影的位置向量点积仍是多个 $\cos((p-q)\omega_r)$ 的和。经过可学习投影后，attention score 还会包含内容与位置的交叉项，不能把这个点积公式直接当成最终 attention 权重。

## 可学习的绝对位置表

### 每个位置一行参数

可学习位置编码直接把位置表作为参数：

$$
P\in\mathbb R^{L_{\max}\times d}.
$$

位置 $p$ 使用第 $p$ 行 $P_{p,:}$。如果最大训练长度为 $L_{\max}$，位置表的参数量为

$$
N_{\mathrm{pos}}=L_{\max}d.
$$

以 $L_{\max}=2048$、$d=4096$ 为例：

$$
N_{\mathrm{pos}}
=2048\times4096
=8,388,608.
$$

若参数使用 FP16，每个参数占 2 bytes，位置表占

$$
\frac{8,388,608\times2}{2^{20}}
=16\ \mathrm{MiB}.
$$

这个开销相对于完整 Transformer 参数可能不大，但它直接绑定了最大位置索引，并且会随 $L_{\max}$ 线性增加。

### 固定表和可学习表的差异

| 方案 | 位置向量来源 | 参数量 | 长度边界 | 主要性质 |
| --- | --- | ---: | --- | --- |
| 正弦编码 | 由频率公式计算 | 0 | 可以计算更大索引 | 固定、连续、带多尺度相位 |
| 可学习位置表 | 训练得到的每行参数 | $L_{\max}d$ | 索引不能超过表长 | 能适应训练分布，外推需要额外方法 |
| 插值后的位置表 | 训练表经坐标变换得到 | 通常仍为 $L_{\max}d$ | 依赖插值规则 | 改变位置间距，必须重新验证模型行为 |

正弦公式能在数学上计算任意 $p$，但模型是否能在训练区间外使用这些向量，取决于训练长度、内容分布和后续层学到的函数。可计算性和泛化能力需要分别核验。

## 相加、拼接和尺度

### 相加保持 model width

最常见的接口是

$$
h_p=e_p+p_p,
\qquad
e_p,p_p\in\mathbb R^d.
$$

相加不新增每个位置的可学习投影，也不改变后续层看到的宽度。多头注意力、残差连接和前馈层可以继续使用同一个 $d_{\mathrm{model}}$。

相加会把两个信号放在同一组通道中。后续的 $W_Q$、$W_K$、$W_V$ 可以从组合向量中同时提取内容和位置，但不能要求模型在任意参数下都能无损恢复 $e_p$ 与 $p_p$ 的两个独立向量。位置编码的接口目标是提供可用的位置条件，不是保存两个输入的分离副本。

### 拼接需要恢复宽度

另一种接口是

$$
h_p=
\begin{bmatrix}
e_p\\
p_p
\end{bmatrix}
\in\mathbb R^{2d}.
$$

如果后续层仍要求宽度 $d$，需要增加投影

$$
W_{\mathrm{cat}}\in\mathbb R^{2d\times d}.
$$

相对于直接相加，额外投影的参数和每个位置的矩阵乘计数为：

| 原始宽度 $d$ | 额外权重参数 $2d^2$ | 加 bias 后参数 $2d^2+d$ | 每个位置 MAC |
| ---: | ---: | ---: | ---: |
| 4 | 32 | 36 | 32 |
| 4096 | 33,554,432 | 33,558,528 | 33,554,432 |

拼接保留了两个输入块的显式边界，但它增加了投影和激活带宽。选择相加还是拼接，应同时记录后续层的宽度、参数和计算量。

### 需要记录相对尺度

令 token embedding 和位置向量的均方根为

$$
\operatorname{rms}(e_p)
=\sqrt{\frac{1}{d}\sum_{j=1}^{d}e_{p,j}^2},
\qquad
\operatorname{rms}(p_p)
=\sqrt{\frac{1}{d}\sum_{j=1}^{d}p_{p,j}^2}.
$$

如果位置分支的尺度远大于 token 分支，$h_p$ 的早期变化会主要来自位置；如果位置分支太小，attention score 可能很难利用它。初始化和训练审计应记录两者的 RMS、相加后的 RMS 以及 Q/K 投影前后的分位数。

## 位置怎样进入 attention score

### 输入相加会产生四类项

设

$$
q_p=W_Q(e_p+p_p),
\qquad
k_s=W_K(e_s+p_s).
$$

忽略缩放因子时，点积展开为

$$
\begin{aligned}
q_p^\mathsf Tk_s
={}&e_p^\mathsf TW_Q^\mathsf TW_Ke_s\\
&+e_p^\mathsf TW_Q^\mathsf TW_Kp_s\\
&+p_p^\mathsf TW_Q^\mathsf TW_Ke_s\\
&+p_p^\mathsf TW_Q^\mathsf TW_Kp_s.
\end{aligned}
$$

四项分别包含内容—内容、内容—位置、位置—内容和位置—位置的相互作用。位置可以改变 query 与 key 的匹配，但它通过投影矩阵参与 score 的方式由训练得到。

在缩放点积注意力中，最终还要计算

$$
\operatorname{score}(p,s)
=\frac{q_p^\mathsf Tk_s}{\sqrt{d_k}}+M_{p,s},
$$

其中 $M_{p,s}$ 可能是 padding mask、causal mask 或其他 bias。位置编码和 attention mask 解决不同问题：前者提供位置条件，后者规定某条 query-key 连接是否可用。

### 绝对位置不等于相对距离

正弦编码的未投影点积具有位置差结构，可学习绝对表只给出每个位置一行向量。两者都把绝对位置送入模型，但都不自动保证最终 score 只依赖 $p-s$。内容交叉项、Q/K 投影、层归一化和后续层都会改变 score。

相对位置方法把位置信息放在不同接口中：

| 方法 | 位置作用位置 | 依赖形式 | 后续词条 |
| --- | --- | --- | --- |
| 绝对位置编码 | 输入 hidden | 每个位置一条 $p_p$ | 当前词条 |
| relative position bias | score | 依据 $p-s$ 加 bias | 相关实现另行说明 |
| RoPE | Q/K 特征 | 对每个频率对做相位旋转 | [RoPE](../transformer-components/rope/) |
| ALiBi | score | 依据距离加入带斜率项 | [ALiBi](../transformer-components/alibi/) |

这里的分类按信号进入计算图的位置区分。具体实现仍需核对 mask、长度和缓存合同。

## padding、packed sequence 和 cache offset

### padding 位置也必须有定义

一个 batch 可能包含长度不同的序列。假设两条样本长度分别为 3 和 2，右 padding 后：

| 样本 | token 槽位 | 有效 mask | 有效 position ID |
| --- | --- | --- | --- |
| A | $a_0,a_1,a_2,\mathrm{PAD}$ | 1,1,1,0 | 0,1,2 |
| B | $b_0,b_1,\mathrm{PAD},\mathrm{PAD}$ | 1,1,0,0 | 0,1 |

PAD 槽位是否分配 position ID 由实现决定。无效槽位必须被 key mask 排除，并且通常不应进入 loss reduction。给 PAD 一个数值不会使它成为有效位置；mask 才决定它是否能参与读取。

左 padding 会改变有效 token 的位置。对长度为 3 的 A，左 padding 后可以使用

$$
(\mathrm{PAD},a_0,a_1,a_2)
\longrightarrow
(0,0,1,2)
$$

也可以使用右对齐的绝对位置

$$
(\mathrm{PAD},a_0,a_1,a_2)
\longrightarrow
(0,1,2,3).
$$

两种合同都会出现于实际系统，但训练、mask、生成和 cache 必须采用同一约定。尤其是 decoder-only 模型左 padding 时，最后一个有效 token 的 position ID 常常需要与真实上下文长度一致。

### packed sequence 要重置边界

把多条短序列拼进一条长序列可以减少 padding，但必须同时保存：

1. 每条序列的起始位置；
2. 每条序列的有效长度；
3. 不同序列之间不可见的 block mask；
4. 每条序列内部是否从 $p=0$ 重新计数。

若第二条序列继承第一条序列的绝对位置，却在 attention mask 中禁止跨序列读取，模型会看到一组与独立运行不同的 position ID。这个变化可能是设计选择，也可能是 packed batch 的实现错误。

### 增量解码使用真实 cache 偏移

若历史 KV cache 已包含 $L_{\mathrm{past}}$ 个 token，本轮新 token 的真实位置通常为

$$
p_i=L_{\mathrm{past}}+i,
\qquad
i\in\{0,\ldots,U-1\}.
$$

新 query 的位置不能因为本轮输入张量只有 $U$ 行就重新从 0 开始。位置 ID、causal mask 和 KV cache 的历史长度必须共同使用同一个 offset。[因果掩码](../attention/causal-masking/)词条会进一步处理 prefill、decode、chunk prefill 和 block-diagonal mask。

## 长度扩展与外推

### 可学习位置表有硬边界

可学习位置表只定义了 $0$ 到 $L_{\max}-1$ 的行。输入长度 $T>L_{\max}$ 时，至少会遇到：

| 现象 | 直接原因 | 需要验证的处理 |
| --- | --- | --- |
| 查表越界 | 没有第 $T$ 个位置的参数 | 拒绝输入、扩展表或使用新的位置机制 |
| 复制最后一行 | 多个位置共用同一向量 | 位置区分能力和长程任务表现 |
| 插值旧表 | 新位置映射到旧坐标 | 插值坐标、attention score 和长上下文任务 |
| 切换位置方法 | 训练和推理的信号分布改变 | 重新校准并比较短、长长度 |

把最后一行复制到所有超长位置会让这些位置在输入侧失去区分。把位置表插值到更长长度可以构造向量，但不能由此推出模型已经学会新的距离尺度。

### 正弦公式可计算不代表模型已外推

正弦公式没有位置表的索引上限，因此可以计算 $p\ge L_{\mathrm{train}}$。但训练期间后续层只在有限位置分布上更新参数，超出该分布后可能出现：

1. 相位组合进入模型未见过的范围；
2. 内容与位置交叉项的统计分布发生变化；
3. attention score 的尖锐度或有效长度改变；
4. 训练中没有覆盖的 padding、cache offset 和长序列 mask 组合。

外推测试应按长度分桶报告结果，例如训练内长度、略超训练长度、明显超出长度，并同时记录位置向量范数、score 分位数和有效 attention 行数。

## 常见失效模式

| 失效模式 | 直接症状 | 检查动作 |
| --- | --- | --- |
| 没有加入位置编码 | 交换 token 顺序后输出只随行置换 | 对同一组 token 做两个置换并比较输出 |
| 位置起点偏移 | 首 token 的向量或 logits 改变 | 固定输入，分别打印 $p=0$ 和 $p=1$ 的查表结果 |
| padding 仍参与读取 | 有效 token 的输出依赖 PAD 数量 | 检查 key mask、query mask 和 loss mask |
| 左右 padding 混用 | 相同文本在 batch 中得到不同绝对位置 | 记录每个有效 token 的 position ID |
| cache offset 重置 | decode 首 token 与 full prefill 不一致 | 比较 full、prefill+decode 和单步 decode 的 logits |
| 位置表越界 | 长输入报错或重复使用位置向量 | 断言 $\max(I)<L_{\max}$ 并记录超长策略 |
| 位置尺度失衡 | Q/K score 方差或注意力熵异常 | 比较 token、position 和 $h$ 的 RMS 与分位数 |
| 把绝对位置当成距离 | 远近关系随内容改变而失效 | 单独固定内容，测试不同 $p-s$ 的 score |
| packed 边界未重置 | 不同样本之间出现位置漂移或可见性错误 | 同时检查 position ID、segment mask 和有效长度 |

这些症状可能来自位置编码，也可能来自 mask、tokenization、Q/K 投影或缓存重排。诊断时要保存完整输入合同，避免只看最终 logits。

## 核验协议

对一个实现，至少记录以下项目：

1. 固定位置起点、位置轴、最大长度、padding 方向和 cache offset 规则。
2. 用 $d=4$、$p=0,1,2$ 重算正弦向量，核对每个正弦/余弦对的平方和。
3. 核对位置表或位置公式的输出 shape 是否为 $[B,T,d]$，以及无效位置是否被 mask。
4. 做位置置换实验：同一组 token 改变顺序后，输出应表现出与位置机制相符的变化。
5. 比较相加前后的 token RMS、position RMS、hidden RMS 和 Q/K score 分布。
6. 在右 padding、左 padding、packed sequence 和不同 cache offset 下分别核对 position ID。
7. 让 full prefill、chunk prefill、增量 decode 使用同一输入，比较有效位置上的 logits。
8. 对 $T=L_{\max}$、$T=L_{\max}+1$ 和更长输入记录越界、插值或外推行为。
9. 需要长上下文时，分别评估训练内长度与训练外长度，不能只确认位置向量仍然可以计算。

以正弦 $d=4$ 例子为基线，独立数值结果应包含：

| 检查项 | 结果 |
| --- | ---: |
| $\lVert\operatorname{PE}_0\rVert_2$ | 1.414213562373 |
| $\lVert\operatorname{PE}_1\rVert_2$ | 1.414213562373 |
| $\operatorname{PE}_1^\mathsf T\operatorname{PE}_2$ | 1.540252306285 |
| $d=4$ 的短周期 | 6.283185307180 |
| $d=4$ 的长周期 | 628.318530717959 |
| $L_{\max}=2048,d=4096$ 的位置参数量 | 8,388,608 |

这些数字检查公式、索引和实现的基本一致性；它们不能替代真实模型在训练内外长度上的行为评估。

## 相关词条

[分词](../text-representation/tokenization/)

[自注意力](../attention/self-attention/)

[注意力矩阵](../attention/attention-matrix/)

[因果掩码](../attention/causal-masking/)

[RoPE](../transformer-components/rope/)

[ALiBi](../transformer-components/alibi/)
