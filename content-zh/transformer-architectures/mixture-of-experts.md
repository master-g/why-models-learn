---
title: "混合专家：用稀疏路由扩大 FFN 容量"
tags: ["why-models-learn"]
---

Mixture of Experts（MoE，混合专家）把一个 token 的前馈变换改成多个候选 expert 的稀疏加权组合：router 先为 token 产生 expert 分数，只选择 top-k 个 expert，再把这些 expert 的输出按门控权重相加。MoE 增加的是所有 expert 的参数容量，单个 token 访问的只是其中一部分；实际收益取决于 dispatch、容量限制、负载均衡和分布式通信是否与这个逻辑合同一致。

![token 先经过 router 选择 top-k expert，再经 dispatch、专家前馈和加权合并返回 residual stream](/assets/transformer-architectures/svg/mixture-of-experts.1.svg)

## 先固定 MoE 的计算合同

### expert 是逐 token 的前馈函数

设一个 batch 中有 $N$ 个有效 token，每个 token 的 residual stream 宽度为 $D$。把第 $t$ 个 token 写成行向量：

$$
\mathbf x_t\in\mathbb R^{D},
\qquad
t=1,\ldots,N.
$$

第 $e$ 个 expert 是一个独立的 FFN。用中间宽度 $H$、激活函数 $\phi$ 和带 bias 的两次线性变换表示：

$$
\mathbf f_e(\mathbf x)
=
\phi(\mathbf xW_{1,e}+\mathbf b_{1,e})W_{2,e}
+
\mathbf b_{2,e},
$$

其中：

$$
W_{1,e}\in\mathbb R^{D\times H},
\qquad
\mathbf b_{1,e}\in\mathbb R^{H},
\qquad
W_{2,e}\in\mathbb R^{H\times D},
\qquad
\mathbf b_{2,e}\in\mathbb R^{D}.
$$

行向量约定让每个 token 的输入形状为 $(D)$，expert 输出仍为 $(D)$，可以直接回到 Transformer block 的残差流。这里的 expert 可以是普通 FFN，也可以把 [SwiGLU FFN](../transformer-components/swiglu-ffn/) 作为每个 expert 的内部结构；MoE 的稀疏性来自 token 到 expert 的选择，不来自激活函数本身。

### router 只负责产生 expert 分数

router 读取同一个 $\mathbf x_t$，输出 $E$ 个 expert 的 logits：

$$
\mathbf z_t
=
\mathbf x_tW_r+\mathbf b_r
\in\mathbb R^{E},
$$

其中 $W_r\in\mathbb R^{D\times E}$。对 expert 轴做 softmax：

$$
p_{t,e}
=
\frac{\exp(z_{t,e})}
{\displaystyle\sum_{j=1}^{E}\exp(z_{t,j})},
\qquad
\sum_{e=1}^{E}p_{t,e}=1.
$$

router 的概率是候选排序和门控的输入。它不是 expert 输出，也不是 token 已经被处理的证明；还必须检查 top-k 选择、容量和 dispatch 结果。

### top-k 选择和门控权重是两步

令 $S_t$ 表示第 $t$ 个 token 选中的 expert 集合：

$$
S_t=\operatorname{TopK}(\mathbf p_t,k),
\qquad
\lvert S_t\rvert=k.
$$

下面采用 top-k 概率重新归一化的合同：

$$
g_{t,e}
=
\begin{cases}
\dfrac{p_{t,e}}{\displaystyle\sum_{j\in S_t}p_{t,j}},
&e\in S_t,\\[10pt]
0,
&e\notin S_t.
\end{cases}
$$

因此 $\sum_e g_{t,e}=1$。有些 top-2 实现直接使用原始 $p_{t,e}$，让被选中的 gate 之和小于 1；这会改变输出尺度和梯度，不能把两种约定混写。本文后面的 top-2 数值均使用重新归一化合同。

最终的 MoE 输出是：

$$
\mathbf y_t
=
\sum_{e\in S_t}
g_{t,e}\mathbf f_e(\mathbf x_t).
$$

若 $k=1$，输出只来自一个 expert。若 $k=E$，每个 token 都访问所有 expert，路由仍然可以产生权重，但条件计算已经退化为 dense mixture。

### padding token 不应参与路由负载

有效 token、padding token 和 loss mask 是三个不同集合。padding token 可以在实现中占据 batch 的位置，但应在 router 负载统计、容量计算和目标损失中使用独立的有效性标记。否则序列长度差异会被误报成 expert 偏好，容量也会被无效 token 消耗。

## 一个 token 如何被送入 expert

### 从 logits 到 top-2 gate

取一个 token 的 router logits：

$$
\mathbf z=(2,1,0).
$$

稳定 softmax 的结果为：

$$
\mathbf p
\approx
(0.665240955775,\ 0.244728471055,\ 0.090030573170).
$$

当 $E=3,k=2$ 时，选中 expert 1 和 expert 2。重新归一化后的 gate 为：

$$
\begin{aligned}
s
&=
0.665240955775+0.244728471055
=0.909969426830,\\
\mathbf g
&=
\left(
\frac{0.665240955775}{s},
\frac{0.244728471055}{s},
0
\right)\\
&\approx
(0.731058578630,\ 0.268941421370,\ 0).
\end{aligned}
$$

逐 expert 记录如下：

|expert|router probability $p_{t,e}$|是否进入 top-2|normalized gate $g_{t,e}$|
|---|---:|---|---:|
|$e_1$|0.665240955775|是|0.731058578630|
|$e_2$|0.244728471055|是|0.268941421370|
|$e_3$|0.090030573170|否|0|

选择集合和 gate 向量要同时保留。只保存 top-k 的 expert ID 而丢掉原始 probability，无法复核归一化方式；只保存 probability 而不保存最终容量裁剪，也无法复核实际执行的 expert。

### 加权合并的数字例子

设三个 expert 给出二维结果：

$$
\mathbf f_1(\mathbf x)=(1,0),
\qquad
\mathbf f_2(\mathbf x)=(0,2),
\qquad
\mathbf f_3(\mathbf x)=(4,4).
$$

因为 $e_3$ 未被选中，它的输出不进入这次前向。最终结果为：

$$
\begin{aligned}
\mathbf y
&=
0.731058578630(1,0)
+
0.268941421370(0,2)\\
&=
(0.731058578630,\ 0.537882842740).
\end{aligned}
$$

如果错误地把原始 probability 直接当作 gate，得到的结果会是：

$$
0.665240955775(1,0)
+
0.244728471055(0,2)
=
(0.665240955775,\ 0.489456942110),
$$

两者方向相近，但尺度不同。这个差异会进入残差流和反向梯度，必须在模型配置中明确记录。

### router 分数不能直接当作解释

router probability 表示当前输入、当前参数和当前 mask 下的选择分布。它可以说明 token 被哪些 expert 接收，但不能单独证明 expert 学到了某个语义类别，也不能证明被选 expert 对输出具有不可替代的因果作用。需要替换输入、屏蔽 expert 或固定路由后重新计算输出，才能检验这种因果影响。

## 参数量、计算量和激活量

### 所有 expert 的参数都要存储

每个带 bias 的普通 FFN 参数量为：

$$
m_{\mathrm{expert}}
=
DH+H+HD+D
=
2DH+H+D.
$$

若有 $E$ 个 expert，router 也带 bias，则 MoE 子层的参数量为：

$$
m_{\mathrm{MoE}}
=
E(2DH+H+D)+(DE+E).
$$

第一个项是所有 expert 的参数，第二个项是 router 参数。即使某个 batch 没有访问某个 expert，它的参数仍然属于 checkpoint、optimizer state 和分布式参数分片。

### 单个 token 的逻辑 MAC

忽略激活函数、比较和 bias 加法，只计算线性层的乘加。router 每个 token 需要：

$$
m_{\mathrm{router}}^{\mathrm{MAC}}=DE.
$$

访问 $k$ 个普通 FFN expert 需要：

$$
m_{\mathrm{experts}}^{\mathrm{MAC}}=k(2DH).
$$

因此稀疏 top-k 的逻辑 MAC 为：

$$
m_{\mathrm{token}}^{\mathrm{MAC}}
=
DE+2kDH.
$$

如果把所有 $E$ 个 expert 都执行，expert 部分变成 $2EDH$。这两个数字是逻辑工作量；padding、容量补齐、dispatch kernel、通信和实现中的 tile 取整还要单独列出。

### 一个小模型的资源账本

取：

$$
D=4,\qquad H=8,\qquad E=4.
$$

每个 expert 有 $2\cdot4\cdot8+8+4=76$ 个参数，router 有 $4\cdot4+4=20$ 个参数。MoE 子层共存储 $4\cdot76+20=324$ 个参数。对应的单 token 账本如下：

|结构|存储的 expert 数|每 token 访问数 $k$|子层参数量|router + expert MAC|
|---|---:|---:|---:|---:|
|单个 dense FFN|1|1|76|64|
|MoE top-1|4|1|324|80|
|MoE top-2|4|2|324|144|
|MoE all-expert|4|4|324|272|

MoE top-1 用 4 个 expert 的参数容量换取每 token 只访问一个 expert，但 router 仍然增加了 16 个乘加。MoE top-2 比 top-1 多一次 expert 前馈，激活和通信也随之增加。比较模型时，不能只报告总参数，也不能只报告 expert MAC。

### active parameter 不是 checkpoint parameter

一个 token 的 active parameter 可以近似写成 router 参数加上被访问 expert 的参数：

$$
m_{\mathrm{active}}
\approx
(DE+E)+k(2DH+H+D).
$$

这个数用于描述一次 token 的条件计算路径，不等于模型文件中的参数量。optimizer state 还可能为每个 expert 保存 momentum、variance 或 master weight。[参数量](../transformer-components/parameter-count/) 词条中的 parameter、activation、optimizer state 和 KV cache 四本账，在 MoE 中仍要分开。

### 激活内存随 token 分配而变化

router 输出的 $N\times E$ logits 可以很小，也可能在大 batch 上占据可观内存。dispatch buffer 还要按总 assignment 数 $Nk$ 和中间宽度 $D$ 保存 token：

$$
N_{\mathrm{assignment}}=Nk,
\qquad
\text{payload bytes}
=
N_{\mathrm{assignment}}D\,b_{\mathrm{dtype}}.
$$

容量补齐会把每个 expert 的 buffer 扩展到 $C$ 个槽位，实际分配量可能接近 $EC D b_{\mathrm{dtype}}$，而有效 token 数只有 $Nk$。因此应同时报告有效 payload、capacity padding 和通信缓冲区。

## top-k 的梯度与离散边界

### 固定选择集合时，梯度可以逐项展开

令上游梯度为 $\mathbf u_t=\partial L/\partial\mathbf y_t$，令 $\mathbf r_{t,e}=\mathbf f_e(\mathbf x_t)$。对选中 expert 的 gate：

$$
\frac{\partial L}{\partial g_{t,e}}
=
\mathbf u_t\cdot\mathbf r_{t,e}.
$$

在一次反向计算中先固定 $S_t$。对 selected logits 做重新归一化 softmax，有：

$$
\frac{\partial g_{t,j}}{\partial z_{t,e}}
=
g_{t,j}
(\mathbb 1_{j=e}-g_{t,e}).
$$

记 $q_{t,j}=\partial L/\partial g_{t,j}$，则：

$$
\frac{\partial L}{\partial z_{t,e}}
=
g_{t,e}
\left(
q_{t,e}
-
\sum_{j\in S_t}g_{t,j}q_{t,j}
\right),
\qquad e\in S_t.
$$

这个式子只覆盖 selected set 内部的 gate 归一化。如果 $e\notin S_t$，主输出路径中的 $g_{t,e}=0$，并且在选择集合固定时通常没有来自该 token 输出的 router 梯度。负载均衡损失或其他 router 正则可以为所有 expert 提供额外梯度。

### top-k 让选择集合在边界处跳变

softmax 对 logits 是连续的，但 TopK 会比较 logits 的大小。两个 expert 分数交换次序时，$S_t$ 可能突然变化：

1. 排名没有跨过第 $k$ 名时，selected set 不变，gate 在局部可以微分；
2. 第 $k$ 名和第 $k+1$ 名交换时，某个 expert 的前向路径突然被加入或移除；
3. 分数相等时，必须使用确定性的 tie-breaking，否则不同设备可能产生不同 dispatch；
4. gate 的梯度不等于选择集合的梯度，训练动态不能只用 softmax 的平滑性解释。

这也是为什么 router 的负载、容量和 overflow 需要作为训练日志，而不能只看总 loss。

### top-1 的负载均衡辅助损失

下面使用一个常见的 top-1 Switch-style 记号。设 $f_e$ 是 batch 中实际被分发到 expert $e$ 的 token 比例，$P_e$ 是 router 对 expert $e$ 的平均概率：

$$
f_e
=
\frac{1}{N}
\sum_{t=1}^{N}\mathbb 1\{S_t=\{e\}\},
\qquad
P_e
=
\frac{1}{N}
\sum_{t=1}^{N}p_{t,e}.
$$

辅助损失定义为：

$$
L_{\mathrm{aux}}
=
E\sum_{e=1}^{E}f_eP_e.
$$

如果四个 expert 的实际比例和平均概率都为 $1/4$，则：

$$
L_{\mathrm{aux}}
=
4\cdot4\cdot\frac{1}{16}=1.
$$

若：

$$
\begin{aligned}
\mathbf f&=(0.5,0.25,0.125,0.125),\\
\mathbf P&=(0.4,0.3,0.2,0.1).
\end{aligned}
$$

则：

$$
L_{\mathrm{aux}}
=
4(0.5\cdot0.4+0.25\cdot0.3+0.125\cdot0.2+0.125\cdot0.1)
=1.25.
$$

这个损失鼓励概率分布和实际选择同时铺开，但它不保证 expert 输出质量相同，也不保证每个 expert 会形成清晰的语义分工。top-k 实现需要先说明 $f_e$ 是按 token 计还是按 assignment 计，再选择相应的归一化；直接套用 top-1 公式会改变损失的尺度。

### router z-loss 处理 logits 尺度

为了限制 router logits 的整体尺度，可以加入基于 log-sum-exp 的正则：

$$
L_z
=
\frac{1}{N}
\sum_{t=1}^{N}
\left(
\log\sum_{e=1}^{E}\exp(z_{t,e})
\right)^2.
$$

它约束的是 logits 的数值尺度，不等于负载均衡损失。前者可以降低 softmax 溢出或极端尖锐分布的风险，后者关注 expert 使用比例。两者都存在时，训练日志应分别记录。

## 容量限制和 token overflow

### 每个 expert 需要有限的 slot

若有 $N$ 个有效 token、每个 token 选择 $k$ 个 expert，则 assignment 总数为：

$$
N_{\mathrm{assignment}}=Nk.
$$

在所有 expert 平均接收 assignment 的理想情形下，每个 expert 的期望 assignment 数为 $Nk/E$。设 capacity factor 为 $c$，采用按 assignment 分配 slot 的约定：

$$
C
=
\left\lceil
c\frac{Nk}{E}
\right\rceil.
$$

有些实现把 capacity factor 定义在 token 数而不是 assignment 数上，或者把 top-2 的第二路单独处理。公式本身不能跨实现直接比较，必须在配置中记录 $N$、$k$、$E$、$c$ 和 slot 定义。

### 一个容量裁剪例子

取 $N=8$、$E=4$、$k=2$、$c=1.25$：

$$
C
=
\left\lceil
1.25\frac{8\cdot2}{4}
\right\rceil
=5.
$$

假设一次 batch 的 assignment 计数为 $(8,4,3,1)$。总数为 $16$，但 expert 1 只能保留前 5 个 assignment：

|expert|assigned assignments|capacity $C$|kept|overflow|
|---|---:|---:|---:|---:|
|$e_1$|8|5|5|3|
|$e_2$|4|5|4|0|
|$e_3$|3|5|3|0|
|$e_4$|1|5|1|0|

overflow 的处理方式包括丢弃该路、退回另一条已选路、降低 gate 后重新归一化，或者采用 expert-choice 等不同路由合同。若实现静默丢弃，loss mask、有效 token 数和输出尺度都可能改变。报告中至少要记录 overflow 数量、被丢弃的 assignment、受影响 token 数和最终输出是否归一化。

### capacity factor 不是负载均衡保证

增大 $c$ 可以减少 overflow，但会增加每个 expert 的 buffer、padding 和通信量。如果 router 已经塌缩到一个 expert，任何有限的 $c$ 都只能截断更多 token；容量不是均衡器。反过来，辅助损失降低也不能保证容量从未溢出，因为平均概率和实际 top-k 选择可能不同。

### padding 和 packed sequence 的统计轴

变长序列常常先 padding 到同一个 $T$，再通过 attention mask 和 loss mask 排除无效位置。MoE 的路由统计还需要一个 valid-token mask：

$$
v_t
\in\{0,1\},
\qquad
N_{\mathrm{valid}}
=
\sum_{t=1}^{N_{\mathrm{padded}}}v_t.
$$

容量和 $f_e$ 应使用 $N_{\mathrm{valid}}$，而不是 $N_{\mathrm{padded}}$。packed sequence 则直接把多个样本的有效 token 拼成一条 dispatch 输入，但必须保留样本边界，避免 attention 或 loss 规则被错误共享。

## dispatch：从 token batch 到 expert batch

### 先建立 assignment 表

对每个 token 和每个 selected expert 建立一行 assignment：

|字段|含义|
|---|---|
|token index|原始 token 在有效 batch 中的位置|
|expert index|目标 expert 的编号|
|gate|该 expert 的加权系数|
|slot|该 expert buffer 中的写入位置|
|keep|容量裁剪后是否保留|

对于每个 expert，保留的 token 可以写成：

$$
X_e
=
\left[
\mathbf x_t:
e\in S_t
\land
\operatorname{keep}(t,e)=1
\right].
$$

经过 expert 前馈后：

$$
Y_e
=
\left[
\mathbf f_e(\mathbf x_t):
e\in S_t
\land
\operatorname{keep}(t,e)=1
\right].
$$

然后按 token index 逆置换，并用 gate 加权合并：

$$
\mathbf y_t
=
\sum_{e\in S_t}
\operatorname{keep}(t,e)
g_{t,e}\mathbf f_e(\mathbf x_t).
$$

如果某个 token 的所有 selected assignment 都被裁剪，公式中的有效和为空。实现必须定义零输出、残差直通、回退 expert 或其他行为，不能让空和静默变成 NaN。

### dispatch 的四个可审计阶段

|阶段|输入|输出|需要检查的字段|
|---|---|---|---|
|route|$X\in\mathbb R^{N\times D}$|$S_t,p_{t,e},g_{t,e}$|top-k、tie-breaking、padding mask|
|assign|token/expert 对|slot 化 assignment|capacity、overflow、token 重复数|
|expert compute|每个 expert 的 buffer|$Y_e$|每个 expert 的实际 token 数、dtype、shape|
|combine|$Y_e$ 和 inverse index|$Y\in\mathbb R^{N\times D}$|gate 合计、空 assignment、残差对齐|

route 的输出正确，不代表 combine 已经把 token 放回原位置。尤其在分布式 all-to-all 后，原始 token index、expert index 和 rank index 都必须保留。

## expert parallelism 和 all-to-all

### expert 的归属决定通信方向

设有 $R$ 个设备，每个设备保存一部分 expert。用 $r(e)$ 表示 expert $e$ 所在的 rank。若 token $t$ 当前位于 rank $r_t$，则 assignment 在：

$$
r_t\ne r(e)
$$

时需要跨 rank 发送。简化地，若远程 assignment 数为 $N_{\mathrm{remote}}$，输入 token 用 $b_{\mathrm{dtype}}$ 字节表示每个特征，则一次 dispatch payload 为：

$$
\text{bytes}_{\mathrm{dispatch}}
=
N_{\mathrm{remote}}D\,b_{\mathrm{dtype}}.
$$

expert 输出通常还要按 inverse mapping 返回，实际通信至少包含回程 payload、assignment metadata、padding 和 collective 的协议开销。

### 一个两 rank 的通信例子

取 $N=16$、$D=4$、$E=4$、$R=2$、$k=2$，总 assignment 数为：

$$
N_{\mathrm{assignment}}=16\cdot2=32.
$$

假设恰好一半 assignment 发送到远程 rank，且输入为 FP16：

$$
N_{\mathrm{remote}}=16,
\qquad
\text{bytes}_{\mathrm{dispatch}}
=
16\cdot4\cdot2
=128.
$$

返回 expert 输出的 payload 在相同假设下也是 128 bytes。这个数字没有包含 index、gate、capacity padding、collective header 和实际链路协议；它只是验证维度的最小账本。真实带宽占用应从通信 profiler 或 all-to-all 的实际 buffer 读取。

### 负载均衡要区分三个轴

一个 batch 可能在 token 轴上均衡，但在 rank 轴上不均衡。例如两个 rank 各保存两个 expert，expert assignment 计数为 $(8,4,3,1)$ 时，rank 0 可能收到 12 个 assignment，rank 1 只有 4 个。此时 expert-level 的辅助损失和 rank-level 的通信拥塞描述的是不同现象。

至少应分别记录：

1. 每个 expert 的 assignment 数；
2. 每个 rank 的接收和发送 bytes；
3. 每个 rank 的 expert compute 时间；
4. all-to-all 等待时间；
5. overflow 和 capacity padding。

只看平均 expert load 会漏掉最慢 rank。分布式 step 的尾延迟由最慢的 dispatch、expert compute 或 combine 路径决定。

## 训练、评估和生成时的路由

### 训练阶段需要同时优化两条路径

主任务损失通过被选 expert 的输出回传。router 还可能收到 load-balancing auxiliary loss 和 router z-loss。训练 batch 中每个 token 的路径为：

$$
\mathbf x_t
\longrightarrow
\mathbf z_t
\longrightarrow
S_t
\longrightarrow
\operatorname{dispatch}
\longrightarrow
\mathbf f_e(\mathbf x_t)
\longrightarrow
\mathbf y_t.
$$

其中 $S_t$ 是离散选择，dispatch 还可能引入 overflow。一个训练 loss 下降的 batch，可能同时出现 expert collapse、overflow 增加或通信变慢；这些指标必须单独记录。

### 评估阶段要固定随机性和容量规则

如果训练时 router 加入噪声或使用随机 tie-breaking，评估时应说明是否关闭噪声、是否固定 tie-breaking、是否仍然使用 capacity drop。评估时把 capacity 调大可能减少 token drop，但也改变了计算量和输出路径，不能把结果与训练配置下的 throughput 直接比较。

### 自回归生成阶段的 batch 形状变化

decoder-only 模型在 prefill 阶段一次处理多个前缀 token，在 decode 阶段每个序列通常每步增加一个 token。router 的 token 数、expert load、capacity padding 和 all-to-all 消息大小会随 batch 中的有效序列数变化。[Decoder-Only](../transformer-architectures/decoder-only/) 的 KV cache 减少了历史 attention 的重复计算，但不会消除 MoE router、dispatch、expert compute 和 combine。

当请求长度差异很大时，短序列完成后留下的 active batch 变小，原来均衡的 expert load 可能变得尖锐。部署报告要分开记录 prefill 和 decode 的路由统计。

## MoE 和其他前馈结构的边界

### MoE 不等于 SwiGLU

SwiGLU 只定义一个 FFN 内部的 gate branch、value branch 和 down projection。可以让每个 MoE expert 都使用 SwiGLU，但此时有两层 gate：

1. expert 内部 gate 沿特征轴调制中间向量；
2. MoE router 沿 expert 轴选择 token 的计算路径。

前者通常是 dense 的逐坐标乘法，后者才决定 token 是否访问某个 expert。把两者都称为“门控”会丢失路由轴。

### MoE 不等于稀疏注意力

[稀疏注意力](../attention/sparse-attention/) 在 query-key 位置之间删减连接；MoE 在 token-expert 之间选择前馈路径。两者的稀疏对象、mask、容量、梯度和通信都不同：

|结构|被选择的对象|稀疏轴|主要资源风险|
|---|---|---|---|
|dense FFN|一个固定 FFN|无|所有 token 共享同一容量|
|MoE|expert|token 到 expert|overflow、dispatch、expert load|
|SwiGLU FFN|内部特征分支|特征轴上的 dense gate|参数量和激活尺度|
|sparse attention|key 位置|query 到 key|连接集合、kernel 和 attention cache|

MoE 仍然可以放在 attention 后的 FFN 子层位置；它不会自动减少 attention 的 $T^2$ 交互，也不会自动减少 KV cache。

### 专家数量和参数预算要匹配

增加 $E$ 会线性增加 expert 参数和 checkpoint 存储。固定 $k$ 时，单 token 的 expert MAC 近似不随 $E$ 线性增加，但 router MAC 为 $DE$，负载均衡和通信候选数也会增加。若比较一个 dense FFN 和 MoE，至少要对齐：

|比较轴|需要固定或报告的量|
|---|---|
|容量|总 expert 参数、单 expert 宽度、active parameter|
|计算|router MAC、selected expert MAC、padding 和 tile 取整|
|数据|有效 token 数、序列长度分布、padding 比例|
|系统|expert/rank 布局、all-to-all bytes、通信重叠|
|训练|更新步数、aux loss 权重、overflow 和 optimizer state|

只把 MoE 的总参数与 dense 模型的参数相比较，或者只把 selected MAC 当作端到端 FLOPs，都会遗漏一部分预算。

## 独立数值核对

下面的数值由独立标准库脚本计算，再与正文公式逐项比对。脚本没有使用深度学习框架；它只实现指数、softmax、top-k 归一化、向量加权和和整数账本。

|核对项|输出|
|---|---|
|logits $(2,1,0)$ 的 softmax|$(0.665240955775,0.244728471055,0.090030573170)$|
|top-2 normalized gate|$(0.731058578630,0.268941421370,0)$|
|三个二维 expert 输出的合并结果|$(0.731058578630,0.537882842740)$|
|$D=4,H=8,E=4$ 的单 expert 参数|76|
|同配置 router 参数|20|
|MoE 子层总参数|324|
|top-1 / top-2 / all-expert token MAC|80 / 144 / 272|
|$N=8,E=4,k=2,c=1.25$ 的 capacity|5|
|assignment 计数 $(8,4,3,1)$ 的 overflow|$(3,0,0,0)$|
|top-1 负载均衡例子的 $L_{\mathrm{aux}}$|1.25|
|$N=16,D=4,k=2$ 且一半远程的 FP16 dispatch payload|128 bytes|

这些数字只验证局部合同。它们不能证明某个 router 在真实训练数据上形成了稳定的 expert specialization，也不能代替设备上的 kernel、通信和吞吐测量。

## 失效模式

### router collapse

大量 token 进入同一个 expert 时，其他 expert 的梯度和更新次数减少。应同时查看每个 expert 的 assignment、平均概率、梯度范数和实际 compute 时间；只看 auxiliary loss 可能漏掉主任务路径上的塌缩。

### 把 top-k 概率和 normalized gate 混用

原始 probability 的和可能小于 1，重新归一化后的 gate 的和为 1。两者会产生不同的 residual 输出尺度和 router 梯度。配置、checkpoint 和推理实现必须记录 gate contract。

### overflow 被静默丢弃

capacity drop 会改变某些 token 的有效 expert 数。若不记录 token drop 和 loss mask，训练 loss 可能对应另一组有效样本；推理输出也可能在长序列或高并发时突然变化。

### 把逻辑稀疏当作真实加速

top-k 只说明数学上选择了较少的 expert。若实现仍然把所有 token 复制到所有 expert、执行完整 dense kernel 或在同一 rank 上串行等待，端到端时间不会按 $k/E$ 缩短。应比较 router、dispatch、expert compute、combine 和通信的实测时间。

### 只报告 expert 平均负载

平均 load 可能接近均匀，但一个 rank 仍然可能承受更多远程 token 或更慢的 expert。报告应保留 expert 轴和 rank 轴的分布，尤其是 p95/p99 step 时间。

### 把 expert specialization 当作已验证事实

不同 expert 的输入分布可以不同，但输入分布差异不等于稳定的可解释语义分工。要验证 specialization，需要固定路由、替换 expert、跨 batch 统计输入类别和输出影响；router heatmap 只能作为观察。

### 用不同预算比较 top-1 和 top-2

top-2 会增加 expert MAC、dispatch assignment、capacity slot 和通信。若 top-2 结果更好，应同时报告 active compute、总参数、overflow、训练步数和吞吐，不能只比较最终 loss。

### 忽略 tie-breaking 和数值 dtype

接近相等的 logits 在不同 dtype、不同设备或不同排序实现下可能产生不同 top-k 集合。应固定排序规则，使用稳定 softmax，并在导出、量化和部署端重复边界输入。

## MoE 架构审计

拿到一个 MoE block 或训练日志时，可以按以下顺序核对：

1. 固定 residual stream 的输入形状 $(B,T,D)$，并说明有效 token 数 $N$；
2. 记录 expert 数 $E$、中间宽度 $H$、top-k $k$、router 是否带 bias；
3. 写出 expert 的内部 FFN，区分普通 FFN、SwiGLU 和其他变体；
4. 检查 router logits 的轴、softmax 轴、temperature 和数值稳定性；
5. 检查 top-k 的 tie-breaking、gate 是否重新归一化；
6. 检查 padding mask、valid-token mask、loss mask 是否分开；
7. 按 assignment 计数 capacity、slot、overflow 和 token drop；
8. 对比 expert load、rank load、dispatch bytes 和 all-to-all 等待时间；
9. 分开记录主任务损失、load-balancing loss、router z-loss 和梯度；
10. 把 total parameter、active parameter、activation、optimizer state 和通信 buffer 分开报告；
11. 分别测量 prefill、decode、不同 batch 长度和不同序列长度下的路由；
12. 用固定输入验证 route、dispatch、expert 输出、combine 和残差对齐。

这份账本把“有多个 expert”拆成了可计算的选择、容量和系统路径。只有当实际 dispatch 和 kernel 证据与逻辑 top-k 合同一致时，稀疏计算的资源结论才成立。

## 相关词条

[完整 Transformer](../transformer-architectures/full-transformer/)

[前馈网络](../transformer-components/feedforward/)

[SwiGLU FFN](../transformer-components/swiglu-ffn/)

[参数量](../transformer-components/parameter-count/)

[稀疏注意力](../attention/sparse-attention/)

[Decoder-Only](../transformer-architectures/decoder-only/)
