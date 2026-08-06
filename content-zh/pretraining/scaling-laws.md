---
title: "缩放定律：模型、数据与计算如何改变损失"
tags: ["why-models-learn"]
---

缩放定律是对一组受控训练实验的经验拟合：在 tokenizer、数据分布、训练目标和评估协议固定后，验证损失常在有限规模区间内随模型参数量、有效训练 token 数或训练计算量呈幂律式变化。它不是不需要检查边界的物理定律，也不表示模型规模增大后所有能力都按同一曲线增长。缩放定律真正提供的是边际收益、资源分配和外推不确定性的一个定量近似。

本文先定义 model size、data size、compute 和 held-out loss，再推导单变量幂律、模型与数据的联合拟合，以及固定 compute 下模型规模和数据规模的交换关系。随后说明如何设计网格实验、拟合 log-log 曲线、报告置信区间和识别曲线断点。最后用一个归一化数字例子核对边际收益与 compute 分配，并列出把拟合关系误读成能力定律的失效模式。

![缩放定律示意：在对数尺度上，模型规模增加使验证损失下降，但边际收益逐渐减小，固定计算量还需要在模型与数据之间分配](/assets/pretraining/svg/scaling-laws.1.svg)

## 缩放定律的对象

### 先固定损失和数据分布

设训练实验的配置为

$$
\xi
=
\left(
\mathcal D,\,
\mathrm{tokenizer},\,
\mathrm{objective},\,
\mathrm{architecture},\,
\mathrm{optimizer},\,
\mathrm{evaluation}
\right).
$$

缩放实验改变的变量应写在 $\xi$ 之外，并尽量固定 $\xi$ 中其他部分。对每次运行记录：

|量|记号|说明|
| --- | --- | --- |
|模型规模|$N$|参数量或预先约定的有效模型规模|
|数据规模|$D$|实际进入训练目标的有效 target token 数|
|训练计算量|$C$|按统一 FLOP 或 MAC 口径估算的训练计算|
|评估损失|$L$|固定验证事件集合上的平均 NLL 或 cross-entropy|
|训练状态|step、token、checkpoint|说明曲线对应的训练时刻|

如果模型使用不同 tokenizer、不同验证文本或不同 loss mask，数值变化同时包含评估口径变化，不能直接作为一个缩放变量的效果。[训练数据](../pretraining/training-data/)负责记录数据版本和有效 token；[下一词最大似然](../pretraining/next-token-as-mle/)负责固定 token 事件如何形成 NLL。

### model size 不是单一物理量

最常见的 $N$ 是可训练参数数量，但实际模型规模还可能受以下因素影响：

|配置|需要额外说明|
| --- | --- |
|dense Transformer|总参数通常也接近每个 token 的 active 参数|
|MoE|总参数、每 token active 参数和路由容量分开记录|
|共享权重|输入 embedding、输出 head 或跨层参数是否绑定|
|适配器训练|冻结 base 参数，只统计 trainable 参数还是总参数|
|量化训练|参数存储 dtype 与计算 dtype 分开|

在一组同构 dense 模型中，参数量可以作为模型规模的代理；跨架构比较时，参数量相同不保证表达能力、计算量或优化难度相同。[参数量总账](../transformer-components/parameter-count/)处理参数、MAC、激活和运行时状态的区分。

### data size 是有效训练事件数

对 decoder-only 语言模型，$D$ 应优先表示真正进入 loss denominator 的有效 target token 数，而不是原始文档数、字符数或 dataloader 产出的 padding 数：

$$
D
=
\sum_{\mathrm{updates}}
\sum_{b,t}
m_{b,t}.
$$

如果数据重复采样，同一 token 内容在训练目标中多次出现，$D$ 按实际消费次数累计。若每个 epoch 的数据量固定，使用 epoch 代替 token 数会让不同数据集规模的实验失去可比性。

### compute 需要声明估算口径

对 dense decoder-only Transformer，训练计算常可近似写成

$$
C
\approx
\kappa ND,
$$

其中 $\kappa$ 汇总前向、反向、attention、FFN、输出头、序列长度和实现细节。$\kappa$ 不是跨所有模型的常数：长上下文、稀疏路由、激活重计算、padding、通信和 fused kernel 都会改变实际成本。

因此报告 compute 时要说明是理论 dense MAC、估计 FLOP、设备计数还是 wall-clock GPU·小时。不同口径可以用于同一实验组内排序，但不能把数值直接跨口径比较。

## 单变量幂律

### 先看模型规模与验证损失

在数据规模和训练协议固定的实验区间内，一个常用近似是

$$
L(N)
\approx
L_\infty
+
A N^{-\alpha},
\qquad
A>0,\quad
\alpha>0.
$$

其中 $L_\infty$ 是拟合中的渐近 floor，$A$ 是尺度常数，$\alpha$ 是模型缩放指数。$\alpha$ 越大，模型规模变化带来的损失下降越快；但它仍然是实验区间内的有效指数，不必在更大模型上保持不变。

同样可以对 data size 或 compute 拟合：

$$
\begin{aligned}
L(D)
&\approx
L_\infty+B D^{-\beta},\\
L(C)
&\approx
L_\infty+G C^{-\gamma},
\end{aligned}
$$

其中 $\beta$ 和 $\gamma$ 描述数据、计算变量的有效边际收益。三个指数不应在没有实验依据时互相替换。

### 对数坐标把幂律变成直线

若 $L_\infty$ 已知或被可靠估计，模型规模关系可以变为

$$
\log\left(L-L_\infty\right)
\approx
\log A-\alpha\log N.
$$

在横轴 $\log N$、纵轴 $\log(L-L_\infty)$ 上，斜率近似为 $-\alpha$。这个变换有两个限制：

1. $L-L_\infty$ 必须为正；
2. $L_\infty$ 的误差会影响所有纵坐标。

如果 floor 未知，直接对 $\log L$ 做线性回归会把 floor 和指数混在一起。应报告拟合区间、floor 的估计方法和残差，而不是只给一个斜率。

### 边际收益随规模下降

对数规模的边际损失变化为

$$
\frac{\mathrm dL}{\mathrm d\log N}
=
N\frac{\mathrm dL}{\mathrm dN}
=
-\alpha A N^{-\alpha}.
$$

它的绝对值随 $N$ 增大而下降。把模型参数量扩大 $r$ 倍时，幂律项变为

$$
A(rN)^{-\alpha}
=
r^{-\alpha}A N^{-\alpha}.
$$

因此规模扩大一倍并不意味着损失下降一半；当 $\alpha$ 很小时，收益会较小但仍可能稳定。收益还要和增加的计算、数据和部署成本放在同一账本中。

### 一个单变量数字例子

取归一化模型规模 $n=N/N_0$，设

$$
L(n)
=
1.2+0.4n^{-0.1}.
$$

三个规模的拟合损失为：

|$n$|$L(n)$|相对前一项的模型误差|
| ---: | ---: | ---: |
|$1$|$1.600000$|$0.400000$|
|$10$|$1.517731$|$0.317731$|
|$100$|$1.452383$|$0.252383$|

从 $n=1$ 增加到 $10$，模型误差项减少约 $0.082269$；从 $10$ 增加到 $100$，只再减少约 $0.065348$。总损失仍然下降，但相同的数量级扩张带来的绝对收益变小。

## 模型与数据的联合缩放

### 两个有限资源项共同决定损失

在一个受控实验区间内，可以用

$$
L(N,D)
\approx
L_\infty
+
A N^{-\alpha}
+
B D^{-\beta}
$$

表示模型容量不足和数据不足对验证损失的两个近似贡献。该式不是唯一的拟合形式；交叉项、数据质量、重复率、优化不足和架构差异都可能需要额外项。

这个表达式的用途是拆开两个方向：

|观察|更可能需要检查的变量|
| --- | --- |
|增大 $N$ 后损失明显下降|模型容量仍是主要误差项|
|增大 $D$ 后损失明显下降|数据覆盖或重复率仍是主要误差项|
|同时增大 $N,D$ 才改善|交互、优化或 compute 约束可能重要|
|两者都不改善|可能已接近 floor、评估噪声或实现瓶颈|

这些解释需要由受控对照支持。单次大模型运行不能区分模型规模收益、数据版本变化和训练时间变化。

### 固定 compute 产生模型—数据交换

若把 $C\approx\kappa ND$ 视为固定资源约束，则

$$
D
\approx
\frac{C}{\kappa N}.
$$

代入联合损失：

$$
\begin{aligned}
L_C(N)
&\approx
L_\infty
+
A N^{-\alpha}
+
B\left(
\frac{C}{\kappa N}
\right)^{-\beta}\\
&=
L_\infty
+
A N^{-\alpha}
+
B\left(
\frac{\kappa}{C}
\right)^\beta
N^\beta.
\end{aligned}
$$

沿着固定 compute 曲线增大 $N$ 时，第一项下降，第二项因为可用数据 $D$ 减少而上升。模型过小会受到容量限制，模型过大则会在同一 compute 下缺少数据；两者之间可能存在一个验证损失最低的配置。

### 内部最优点的推导

对 $N$ 求导并令导数为 0：

$$
-\alpha A N^{-\alpha-1}
+
\beta B\left(
\frac{\kappa}{C}
\right)^\beta
N^{\beta-1}
=0.
$$

两边乘以 $N$ 后：

$$
\alpha A N^{-\alpha}
=
\beta B\left(
\frac{\kappa}{C}
\right)^\beta
N^\beta.
$$

整理得到

$$
N_\star^{\alpha+\beta}
=
\frac{\alpha A}{\beta B}
\left(
\frac{C}{\kappa}
\right)^\beta.
$$

因此在这个简化模型中：

$$
N_\star
\propto
C^{\frac{\beta}{\alpha+\beta}},
\qquad
D_\star
\propto
C^{\frac{\alpha}{\alpha+\beta}}.
$$

这条关系只说明两个幂律项和乘积 compute 约束下的内部解。实际 compute-optimal 训练还要考虑优化器、序列长度、数据重复、质量变化、并行效率和目标评估；下一篇 compute-optimal 会把资源分配单独展开。

### 损失 floor 不是自动的真实熵

$L_\infty$ 可能吸收多个未建模因素：

|来源|它对 floor 的影响|
| --- | --- |
|数据本身的不确定性|真实条件分布的熵或不可预测部分|
|tokenizer|事件粒度和词表分解|
|模型族限制|无法表达的数据条件关系|
|优化不足|训练没有到达当前模型的经验最优|
|评估噪声|验证集规模、采样和测量方差|
|拟合区间|在有限点上外推得到的参数|

只有在模型族、数据分布、优化过程和评估误差满足额外条件时，floor 才可能接近某种不可约损失。文章或报告应使用“拟合 floor”描述它，不把一个回归参数直接写成真实世界熵。

## 如何设计缩放实验

### 让一次实验只改变少数变量

一个可解释的实验网格至少要固定：

1. tokenizer、词表和 special token；
2. 训练数据版本、来源 mixture 和去重规则；
3. 目标函数、loss mask 和验证集；
4. 模型架构、优化器、学习率规则和 batch 合同；
5. 训练到的有效 token 数或 compute 计数；
6. 随机种子数量和 checkpoint 选择规则。

如果每个规模都使用不同数据清洗版本，拟合的不是单纯的 $N$ 缩放；如果大模型训练更久，拟合的可能是 compute 或优化时间。实验表应把变化变量放在单独列中。

### 网格要覆盖变化而不是只覆盖一个点

一个最小网格可以沿模型规模和数据规模各取多个点：

|实验轴|固定部分|变化部分|
| --- | --- | --- |
|model scaling|数据、token 数、架构、优化器|$N$|
|data scaling|模型、compute 口径、架构、优化器|$D$|
|compute scaling|模型和数据配比规则|$C$|
|allocation sweep|$C$|在 $N,D$ 之间改变比例|

两个点可以计算斜率，但无法识别断点、噪声和多种拟合形式。至少要保留一个验证点，不用它参与参数选择，用来检查外推误差。

### 同一训练时刻要有明确含义

训练损失随 token 消费量下降。比较不同模型规模时，以下三种条件不同：

|对齐方式|比较的问题|
| --- | --- |
|相同有效 token 数|给模型相同数据预算时，容量差异如何影响损失|
|相同 compute|固定资源时，模型和数据怎样分配|
|相同 wall-clock 时间|固定系统吞吐和工程约束时，得到什么损失|

“训练了相同步数”不等于“看了相同 token”，尤其在 batch、序列长度或 padding 不同的情况下。报告中要写清楚横轴是 token、update、FLOP 还是时间。

### 重复运行才能估计噪声

验证 loss 的差异可能来自随机初始化、数据顺序、dropout、评估 batch 和测量样本。对同一配置使用多个随机种子，可以估计

$$
\overline L
=
\frac1K\sum_{k=1}^{K}L_k,
\qquad
s_L^2
=
\frac1{K-1}
\sum_{k=1}^{K}
(L_k-\overline L)^2.
$$

若两条拟合曲线的差异小于这个噪声范围，就不能把差异写成稳定的缩放收益。单次运行的点可以进入探索图，但需要降低确认程度。

## 拟合和外推

### 先检查残差，再报告指数

对候选模型 $\widehat L(x)$，保留原始空间残差：

$$
r_i
=
L_i-\widehat L(x_i).
$$

至少检查：

|检查|发现什么|
| --- | --- |
|原始 loss 残差|是否存在系统性弯曲或断点|
|log-log 残差|幂律变换后是否仍有结构|
|不同 seed 的误差|点差异是否超过随机噪声|
|训练与验证 gap|是否出现过拟合或数据重复|
|横轴覆盖范围|外推是否远离观测区间|

拟合优度高不等于外推可靠。参数多的函数可以在观测点上拟合得更好，却在下一个数量级给出完全不同的预测。

### floor、指数和系数相互耦合

在

$$
L(N)=L_\infty+A N^{-\alpha}
$$

中，$L_\infty$、$A$ 和 $\alpha$ 会共同调整。同一组有限点可能由较高 floor 加较大指数，或较低 floor 加较小指数解释。应给出：

1. 参数估计和置信区间；
2. 训练点与拟合点的范围；
3. 是否固定 floor；
4. 拟合损失的权重和误差模型；
5. 留出的验证点预测误差。

不要只写“指数为 0.1”而省略 fitting range 和 uncertainty。

### 外推先区分插值和外推

若训练点覆盖 $N\in[N_{\min},N_{\max}]$，预测区间内的点是插值，预测 $N> N_{\max}$ 或 $N<N_{\min}$ 是外推。外推风险来自：

|风险|可能发生的变化|
| --- | --- |
|架构变化|参数量增加时 block、head 或词表比例改变|
|数据变化|新规模需要重复或加入新来源|
|优化变化|学习率、batch 或稳定性规则不再适用|
|系统变化|并行通信和吞吐让 $\kappa$ 改变|
|目标变化|loss 改善不再映射到下游指标|

外推报告应把点估计、区间和未验证假设放在一起。若没有新的实验点，不能把曲线预测写成已确认结果。

## 从 loss 到下游能力

### loss 是连续指标，能力可能有任务边界

验证 NLL 对规模变化通常是连续数值，但下游任务可能有离散阈值、提示格式、检索覆盖或工具调用约束。由 loss 的幂律下降不能直接推出所有 benchmark accuracy 也遵循同一个指数。

比较下游指标时要同时保存：

|量|需要固定什么|
| --- | --- |
|validation loss|tokenizer、mask、数据版本和分母|
|perplexity|平均 NLL 的事件集合|
|accuracy/F1|任务样本、prompt、解码和评分脚本|
|校准指标|概率温度、标签定义和分桶|
|生成质量|采样规则、长度、停止条件和人工协议|

若只选 loss 最低的 checkpoint，再报告其下游最好结果，选择过程需要单独写明；否则它不是一个独立测试估计。

### 能力拐点不是缩放指数

下游曲线出现拐点可能来自：

1. 评估指标的离散阈值；
2. prompt 或 few-shot 示例改变；
3. 任务样本量不足；
4. 解码策略切换；
5. 数据污染或 benchmark overlap；
6. 模型学会了某个必要的中间表示。

这些原因需要额外实验区分。把曲线的视觉拐点直接称为“涌现能力”，没有提供机制或排除评估效应。

## 独立数值核对

下面使用一个归一化的联合缩放模型：

$$
L(n,d)
=
1.2
+
0.4n^{-0.1}
+
0.3d^{-0.2},
\qquad
n=\frac{N}{N_0},
\quad
d=\frac{D}{D_0}.
$$

把归一化 compute 固定为 $nd=1$，比较四个模型—数据分配：

|配置|$n$|$d$|模型项|数据项|总损失|
| --- | ---: | ---: | ---: | ---: | ---: |
|A|1|1|0.400000|0.300000|1.900000|
|B|4|0.25|0.348220|0.395852|1.944073|
|C|0.25|4|0.459479|0.227357|1.886837|
|D|2|0.5|0.373213|0.344610|1.917823|

在这组人为设定的指数和系数下，配置 C 的损失最低，因为数据项的指数和系数使增加 $d$ 的收益超过增加 $n$ 的收益。这个结论只属于该 toy law；它不是跨模型规模的训练建议。

把导数条件代入同一组常数，得到内部平衡点：

$$
n_\star
=
0.258838656218,
\qquad
d_\star
=
3.863410568617,
\qquad
L(n_\star,d_\star)
=
1.886828545532.
$$

配置 C 接近这个连续解，但离散网格点仍有误差。对同一模型规模从 $n=1$ 增加到 $n=2$、保持 $d=1$ 时：

$$
\begin{aligned}
L(1,1)
&=
1.900000000000,\\
L(2,1)
&=
1.873213196615,\\
\Delta L
&=
0.026786803385.
\end{aligned}
$$

独立脚本重算的输出为配置 A/B/C/D 总损失 [1.900000000000, 1.944072598550, 1.886836826975, 1.917822703114]、连续平衡点 [0.258838656218, 3.863410568617] 和损失 1.886828545532。数值只核对幂律表达式、固定乘积约束和导数解，不替代真实训练实验。

## 失效模式：把经验曲线当成能力定律

### 改变了多个变量却只归因于模型规模

**现象：**大模型同时使用更多 token、更长训练时间、新数据或不同 tokenizer，报告却写成 model scaling。

**检查：**逐运行保存 $N,D,C$、数据版本、tokenizer、目标、架构和 optimizer；把变化变量拆开。

### 把参数量当成所有模型的同一尺度

**现象：**dense、MoE、共享权重和适配器模型只按总参数量排序。

**检查：**同时记录总参数、active 参数、trainable 参数、每 token compute 和存储 dtype。

### 用 epoch 代替有效 token

**现象：**不同数据集规模使用相同 epoch，却实际消费了不同数量的 token，曲线斜率失去含义。

**检查：**用 loss mask 后的 target token 计数对齐训练预算，并记录重复采样次数。

### 计算量口径不一致

**现象：**一组实验混用理论 FLOP、设备小时和 profiler MAC，固定 compute 曲线不再固定。

**检查：**写出 $\kappa$ 的定义、是否含前向和反向、是否含 padding、通信和重计算；在同一组实验中保持口径一致。

### 把训练损失当验证缩放

**现象：**训练集 loss 下降被报告成泛化收益，重复数据或过拟合没有被发现。

**检查：**固定独立 validation manifest，分开报告 train、validation 和 source-level metrics。

### floor 未知却直接线性拟合 log loss

**现象：**把 $\log L$ 的斜率直接当作 $\alpha$，不同拟合区间给出不同指数。

**检查：**比较包含 $L_\infty$ 的模型、固定 floor 的敏感性和原始残差；报告参数区间。

### 只用两个点外推多个数量级

**现象：**两次运行形成一条直线，下一代模型的损失被当成已知。

**检查：**增加规模点、保留外推验证点、报告断点风险和未验证假设。

### 用 loss 预测所有下游能力

**现象：**验证 NLL 下降被解释成所有 benchmark、事实正确率或生成质量必然提升。

**检查：**分别固定下游数据、prompt、解码和评分协议；验证 loss 与任务指标的相关性。

### 忽略数据重复和污染

**现象：**更大的 token budget 主要来自重复文档或 benchmark overlap，损失改善却被称为数据规模收益。

**检查：**按 source、dedup cluster、unique token 和 validation overlap 报告有效数据量。

### 把 wall-clock 改善当成 compute scaling

**现象：**硬件、并行度或 kernel 优化降低了训练时间，模型 loss 变化却被归因于计算量。

**检查：**分开记录理论 compute、实际吞吐、通信、设备小时和训练 token；不要用时间替代 FLOP。

## 可复用的缩放实验协议

审计一组 scaling law 实验时，按以下顺序记录：

1. 固定 tokenizer、词表、数据版本、去重规则和验证 manifest；
2. 定义 $N$、$D$、$C$ 的计数口径与是否包含 embedding、padding、反向和通信；
3. 固定模型架构族、目标函数、loss mask、optimizer 和学习率规则；
4. 设计覆盖多个数量级的 model、data、compute 或资源分配网格；
5. 为每个配置记录随机种子、checkpoint、有效 token、验证样本和 wall-clock；
6. 分别报告 train、validation、来源分项和下游任务指标；
7. 先在原始 loss 空间检查残差，再在 log-log 空间检查幂律近似；
8. 对 $L_\infty$、指数和系数报告拟合范围、误差模型和敏感性；
9. 保留至少一个不参与拟合的规模点检查插值或外推误差；
10. 检查数据重复、污染、截断、padding 和 source mixture 是否随规模变化；
11. 把理论 compute、实际吞吐、设备小时和优化器状态分开记账；
12. 将曲线只解释到实验支持的范围，把未验证外推和下游映射单独标记。

缩放定律适合回答“在受控范围内增加哪一种资源，验证损失如何变化”。它不能替代数据质量、优化稳定性、评估隔离和下游任务验证。下一篇 [计算最优训练](../pretraining/compute-optimal/)会在这个联合模型上继续处理固定 compute 下的资源分配。

## 相关词条

- [预训练](../pretraining/pretraining/)：定义有效 token、global batch、token budget 和 checkpoint。
- [训练数据](../pretraining/training-data/)：记录来源、清洗、去重、切分、manifest 和有效 token mixture。
- [下一词最大似然](../pretraining/next-token-as-mle/)：固定验证 NLL、token reduction 和 PPL 的事件集合。
- [参数量总账](../transformer-components/parameter-count/)：区分参数、active 参数、MAC、激活和运行时内存。
- [计算最优训练](../pretraining/compute-optimal/)：继续推导固定计算预算下模型和数据的分配。
- [损失函数](../training-nn/loss-functions/)：区分训练目标、归约和评估损失。
- [经验风险最小化](../learning-framework/empirical-risk-minimization/)：把有限样本平均损失放回经验风险框架。
- [模型容量](../learning-framework/model-capacity/)：解释参数化假设空间与表达能力边界。
