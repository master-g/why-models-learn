---
title: "Teacher Forcing：真实前缀如何改变序列训练"
---

Teacher forcing（教师强制）是序列生成训练中的一种输入策略：decoder 在第 $j$ 步接收真实的前一 token，而不是自己在第 $j-1$ 步生成的 token。它让每个训练位置都处在正确目标前缀上，通常能显著简化优化；代价是训练分布和推理分布不同，模型可能没有练习过如何从自己的错误前缀恢复。

这不是一个开关就能概括的「训练技巧」。要审计 teacher forcing，至少要把以下对象分开：

1. decoder 输入是 $y_{j-1}$ 还是 $\widehat y_{j-1}$；
2. loss 监督的是 $y_j$ 还是错位的 token；
3. 生成 token 的离散选择是否参与梯度；
4. 训练时的前缀分布和部署时的前缀分布是否一致；
5. 评估指标是在 teacher-forced 轨迹上，还是在 free-running 轨迹上。

上一篇 [Seq2Seq](../rnn-lstm/seq2seq/) 给出了 encoder-decoder 总图。本篇只沿 decoder 时间轴追踪前缀来源、梯度边界和暴露偏差，最后再比较 scheduled sampling、自由运行训练和序列级目标。

![Teacher forcing 与自由运行前缀](/assets/rnn-lstm/svg/teacher-forcing.1.svg)

## 两条 decoder 计算图

给定 encoder context $c$、目标序列 $y_{1:T}$ 和 decoder 状态 $d_{j-1}$，单步递推为

$$
d_j=f_\theta(d_{j-1},u_j,c),
\qquad
p_j=\operatorname{softmax}(W_od_j+b_o).
$$

teacher forcing 轨迹使用

$$
u_j^{\mathrm{TF}}=
\begin{cases}
\mathrm{BOS},&j=1,\\
y_{j-1},&j\ge2.
\end{cases}
$$

自由运行轨迹使用

$$
u_j^{\mathrm{free}}=
\begin{cases}
\mathrm{BOS},&j=1,\\
\widehat y_{j-1},&j\ge2,
\end{cases}
$$

其中 $\widehat y_{j-1}$ 可以来自 greedy、sampling 或 beam 的当前前缀。

| 轨迹 | 第 $j$ 步输入 | 前缀来源 | 是否暴露模型自己的错误 |
| --- | --- | --- | --- |
| teacher forcing | $y_{j-1}$ | 数据集 target | 通常不暴露 |
| free running | $\widehat y_{j-1}$ | 模型过去的选择 | 会暴露 |
| mixed / scheduled | 两者按规则选择 | 数据与模型的混合 | 按 schedule 暴露 |

训练目标通常只对 teacher-forced 轨迹写交叉熵：

$$
\mathcal L_{\mathrm{TF}}
=
-\sum_{j=1}^{T}
\log p_\theta(y_j\mid y_{<j},x).
$$

推理想要的却是

$$
\widehat y_{1:T}
\sim
p_\theta(\,\cdot\mid x)
\quad
\text{按自回归路径展开}.
$$

这两个对象相关但不相同。前者是在正确前缀条件下评价每个下一个 token，后者是从 BOS 开始真正滚动生成。

## Token shift：teacher forcing 的第一道闸

目标序列应先加特殊边界，再错开一位：

$$
\mathrm{decoder\ input}
=
[\mathrm{BOS},y_1,y_2,\ldots,y_{T-1}],
$$

$$
\mathrm{target}
=
[y_1,y_2,\ldots,y_{T-1},\mathrm{EOS}].
$$

第 $j$ 行的 logits 预测第 $j$ 个 target，而不是预测同一行喂入的 token。

| decoder step | 输入 token | 监督 token | 是否计入普通 token loss |
| ---: | --- | --- | --- |
| 1 | BOS | $y_1$ | 是 |
| 2 | $y_1$ | $y_2$ | 是 |
| $\cdots$ | $\cdots$ | $\cdots$ | 是 |
| $T$ | $y_{T-1}$ | EOS | 通常是 |

三种最小反例：

1. 把输入和 target 都设为 $[y_1,\ldots,y_T]$，模型可能读取当前答案；
2. 漏掉 EOS，推理没有显式终点；
3. 把 BOS 计入 target，loss 分母和指标含义改变。

只用张量 shape 检查无法发现第一种错误，因为输入和 target 仍然可以同长。应该把一条短目标的 token、位置和 mask 打印成表。

## 为什么 teacher forcing 优化更容易

在 teacher forcing 下，第 $j$ 步的条件前缀是真实的：

$$
p_\theta(y_j\mid y_{<j},x).
$$

即使模型在第 $j-1$ 步的概率分布很差，第 $j$ 步仍然收到正确的 token embedding。这样做有三个直接效果：

1. 每个位置都能得到稳定的 next-token 监督；
2. 长序列中的早期错误不会立刻污染后续输入；
3. BPTT 主要沿连续的 hidden/state 路径回传，而不是还要处理离散 argmax 的分支。

但这不是免费信息。训练阶段把正确前缀作为外部条件注入了每个时间步，模型学到的是「在正确上下文中补下一个 token」，而非完整的错误恢复策略。

### 离散 token 选择的梯度边界

若推理 token 是

$$
\widehat y_{j-1}=\operatorname*{argmax}_{v\in\mathcal V}p_{j-1}(v),
$$

这个离散 argmax 通常不可微。常规 teacher forcing 不需要对 token id 的选择求梯度，因为输入来自数据集；自由运行时若把 argmax 放入训练图，梯度不会穿过这个选择回到上一步 logits。

因此「把模型自己的 argmax 输出喂回去」并不自动等价于可以端到端优化的 free-running loss。要训练穿过选择的目标，需要 soft distribution、采样估计器、straight-through 近似或序列级强化学习等额外设计，每一种都会改变梯度估计。

## 一个三 token 数值账本

设目标是三个需要预测的 token，teacher-forced 条件概率为

$$
p_1=0.8,\qquad p_2=0.7,\qquad p_3=0.9.
$$

则 teacher-forced 的联合目标概率和 NLL 为

$$
p_{\mathrm{TF}}=0.8\times0.7\times0.9=0.504,
$$

$$
\mathcal L_{\mathrm{TF}}
=-\log(0.8)-\log(0.7)-\log(0.9)
=0.685179010911.
$$

若按三个有效 token 取 mean：

$$
\overline{\mathcal L}_{\mathrm{TF}}
=0.228393003637.
$$

现在假设模型第一步选错了一个 token，落入另一条前缀；在该错误前缀下，后两步对正确目标的概率变成 0.1 和 0.5。自由运行这条具体轨迹的目标概率为

$$
p_{\mathrm{wrong\ prefix}}
=0.8\times0.1\times0.5=0.04,
$$

对应 NLL 为

$$
-\log(0.04)=3.218875824868.
$$

teacher forcing 训练时第二步仍使用真实的第一 token，因此不会直接看到这条 0.04 的错误前缀轨迹。这个数字不是说每个错误都会如此放大，而是说明训练条件改变后，后续 token 的条件分布也改变了。

## 暴露偏差：错误如何进入后续状态

令真实前缀 embedding 为 $e(y_{j-1})$，模型前缀 embedding 为 $e(\widehat y_{j-1})$。一次错误输入的差为

$$
\Delta e_{j-1}
=e(\widehat y_{j-1})-e(y_{j-1}).
$$

在 decoder 的局部线性化下，状态差近似满足

$$
\Delta d_j
\approx
A_j\Delta d_{j-1}
+B_j\Delta e_{j-1},
$$

其中 $A_j$ 是对上一 hidden 的局部 Jacobian，$B_j$ 是对输入 embedding 的局部 Jacobian。

若标量近似 $A_j=0.8$，某一步输入错误造成单位扰动，之后只看状态路径的残余为：

| 之后的距离 | 状态差近似 $0.8^k$ | 解释 |
| ---: | ---: | --- |
| 1 | 0.80000 | 错误刚进入下一状态 |
| 2 | 0.64000 | 仍会影响下下步 |
| 5 | 0.32768 | 数值影响衰减，但离散 token 已可能改变 |
| 10 | 0.1073741824 | 连续状态影响变小，输出路径可能早已分叉 |

连续状态差可能衰减，但 token 选择是离散的：一个小概率差异只要改变了 argmax，就可能把下一步带到训练很少见的前缀。暴露偏差的关键不只是 hidden 范数是否爆炸，而是模型在自己产生的前缀分布上是否仍然得到合理条件概率。

### 独立错误概率的直觉

若把每个位置的错误近似为独立事件，单步错误率为 0.1，长度为 20 时至少出现一次错误的概率是

$$
1-(1-0.1)^{20}
=0.878423345409.
$$

真实序列错误通常并不独立，这个数不能当作模型实际失败率；它只说明「单步看起来不错」不能自动推出「整条长序列几乎不会遇到错误前缀」。

## Scheduled sampling：混合前缀的代价

scheduled sampling 在训练时按概率选择真实前缀或模型前缀。设第 $k$ 个训练阶段使用 teacher forcing 的概率为 $q_k$：

$$
u_j=
\begin{cases}
y_{j-1},&\text{概率 }q_k,\\
\widehat y_{j-1},&\text{概率 }1-q_k.
\end{cases}
$$

一种示意性的指数 schedule 是

$$
q_k
=
\max\left(q_{\min},q_0\gamma^{\lfloor k/K\rfloor}\right).
$$

例如 $q_0=0.9,\gamma=0.5,K=10,q_{\min}=0.1$ 时：

| 阶段 $k$ | $q_k$ | 模型前缀概率 $1-q_k$ |
| ---: | ---: | ---: |
| 0 | 0.9000 | 0.1000 |
| 10 | 0.4500 | 0.5500 |
| 20 | 0.2250 | 0.7750 |
| 30 | 0.1125 | 0.8875 |
| 40 及以后 | 0.1000 | 0.9000 |

这个表只是一个可核对的 schedule，不是普适最佳设置。它引入至少四个新变量：

1. $q_k$ 按 optimizer step、batch、epoch 还是 token 计；
2. 模型前缀来自 greedy、sample 还是 soft distribution；
3. 选择真实/模型前缀的随机状态是否可复现；
4. 目标 loss 是否仍然只对真实 target 计算。

### 为什么 schedule 不只是「逐渐关掉帮助」

当输入换成模型自己的 token 时，后续状态、attention/cache、mask 分支都可能改变。schedule 改变的是训练数据分布和计算图路径，而不仅是一个 dropout rate。若模型前缀来自离散采样，loss 仍通常只对当前 target 的 logit 求导，不会自动对「为什么采到这个 token」求导。

因此应该把 schedule 当作单独的训练目标实验，记录：

| 记录项 | 要回答的问题 |
| --- | --- |
| teacher-forced NLL | 正确前缀下的局部预测是否变好 |
| free-running NLL 或序列指标 | 自己前缀下是否更稳 |
| $q_k$ 实际均值 | schedule 是否按预期执行 |
| 前缀错误率 | 模型到底经历了多少非真实状态 |
| 梯度范数/裁剪率 | 混合轨迹是否引入更大方差 |
| seed 与采样日志 | 结果差异是否来自随机前缀 |

## 自由运行训练的几种目标

### Soft rollout

不把离散 token id 喂回去，而是把词表分布或 embedding 的期望作为下一输入：

$$
\bar e_{j-1}
=
\sum_{v\in\mathcal V}p_{j-1}(v)e(v).
$$

它保留可微路径，但 decoder 看到的不是一个真实 token embedding，而是混合 embedding。训练/推理仍存在差异，且大词表下计算和语义代价都不小。

### Sampling estimator

从模型分布采样真实 token，梯度可以用 score-function estimator 等方法估计。它更接近推理分布，但方差高，需要 baseline、奖励定义和随机状态审计。

### Sequence-level objective

直接评价整条生成序列，例如 BLEU、ROUGE、编辑距离或任务 reward，可以减少 token-level teacher forcing 与部署指标之间的距离；但序列级目标通常非光滑、稀疏或需要搜索/采样。

| 目标 | 前缀分布 | 梯度稳定性 | 评价对齐 |
| --- | --- | --- | --- |
| teacher-forced CE | 真实前缀 | 通常最好 | 局部 token 预测 |
| scheduled CE | 混合前缀 | 依 schedule 和采样而变 | 部分靠近部署 |
| soft rollout | 混合 embedding | 可微但语义不同 | 中等 |
| sampled sequence objective | 模型前缀 | 方差可能很高 | 更接近整句 |

没有一种方法在所有任务上同时最稳定、最可微、最接近真实部署。选择时应先写清目标指标和可接受的梯度方差。

## 评估：至少测两条轨迹

### Teacher-forced token metrics

在真实前缀下，可以测：

$$
\operatorname{accuracy}_{\mathrm{TF}}
=
\frac{\sum_j\boldsymbol1_{\{\widehat y_j^{\mathrm{TF}}=y_j\}}M_j}
{\sum_jM_j}.
$$

它适合诊断局部分类、词表概率和 label alignment，但不直接测自由生成。

### Free-running sequence metrics

从 BOS 开始生成，直到 EOS 或 max length，再测整句 exact match、编辑距离、BLEU/ROUGE 或任务指标。要记录 greedy/beam、temperature、max length 和 EOS 规则，否则同一模型可以得到不同的数字。

| 指标轨迹 | 前缀 | 适合发现 |
| --- | --- | --- |
| TF NLL | 真实目标前缀 | token 对齐、局部概率、标签问题 |
| free-running NLL | 自己生成的前缀 | exposure bias、错误恢复 |
| sequence exact match | 自己生成整句 | 早期错误和 EOS 问题 |
| beam/greedy gap | 不同搜索策略 | 局部概率与搜索误差 |

如果 TF NLL 很低而 free-running 质量很差，优先检查前缀分布、EOS、状态 reset 和解码超参，而不是只继续降低学习率。

## 与 BPTT、mask 和状态边界的关系

teacher forcing 不会取消 decoder 的时间反向传播。对 decoder 参数，所有时间步仍共享参数并累加梯度：

$$
\frac{\partial\mathcal L_{\mathrm{TF}}}{\partial\theta_{\mathrm{dec}}}
=
\sum_{j=1}^{T}
\frac{\partial\mathcal L_j}{\partial d_j}
\frac{\partial d_j}{\partial\theta_{\mathrm{dec}}}.
$$

如果使用 TBPTT，detach 仍会截断状态梯度；如果使用 padding mask，mask 仍要决定哪些时间步参与 loss 和状态更新。teacher forcing 只决定 $u_j$ 的来源，不能替代：

1. state carry/reset/detach；
2. source/target padding mask；
3. sum/mean 有效分母；
4. BOS/EOS shift；
5. encoder bridge 的梯度边界。

### 同一 batch 的三种 token

一个目标 batch 中至少有：

| 类型 | 用途 | 是否应计入普通 target loss |
| --- | --- | --- |
| BOS | decoder 起点输入 | 通常否 |
| 有效 target token | 当前监督目标 | 是 |
| EOS | 终止监督 | 通常是 |
| PAD | 对齐不同长度 | 否，除非任务明确把它当目标 |

把 PAD 作为 teacher-forced 输入可以是实现上的占位行为，但应避免它改变有效位置后的 state；把 PAD 计入 loss 则是另一项语义决定。

## 失效模式：看似稳定，实际没测到部署

### 只报告 teacher-forced loss

真实前缀下每一步都容易，不能证明从 BOS 开始会稳定生成。必须至少补一条 greedy free-running 轨迹。

### 把 scheduled sampling 概率当作模型置信度

$q_k$ 是训练输入来源的外部概率，不是模型对 token 的置信度。应记录模型自己的 token probability、prefix error rate 和 schedule 实际分布。

### 让 argmax 误入梯度图

离散 argmax 的结果作为 token id 参与下一步，通常没有可用的梯度路径。若声称端到端优化了生成选择，应明确使用了哪种 estimator 或 soft relaxation。

### 训练/验证模式的前缀来源不一致

验证时若仍 teacher forcing，曲线可能比部署指标乐观；验证时若随机 schedule，又可能和最终 greedy/beam 评估不一致。固定并命名每种评估轨迹。

### 目标错位被 accuracy 掩盖

如果输入和 target 同位，accuracy 可能异常高；如果 BOS/EOS shift 错一位，loss 仍能计算。打印 token 级 shift 表和一条人工预测。

### 采样随机性未固定

scheduled sampling、sampling decoder、dropout 和数据增强同时存在时，正负实验可能经历不同前缀。固定 seed、保存实际 token 前缀，才能比较 schedule。

### Beam search 结果被当成训练结果

beam width 和 length penalty 改变推理搜索，不改变 teacher-forced loss。报告时把模型目标、搜索目标和最终任务指标分开。

### 通过 reset 让 free-running 变好

每步错误后重置 decoder state 可能暂时减轻错误累积，却改变了自回归模型。若采用 reset，必须说明它是任务语义还是诊断实验。

## Teacher forcing 审计协议

1. **画输入来源。** 对每个 decoder step 标出 BOS、真实前缀、模型前缀和 target。
2. **核对 shift。** 手算一条含 BOS/EOS/PAD 的最短样本，确认每行监督对象。
3. **分离两个 loss。** 同时计算 teacher-forced token NLL 和 free-running 轨迹指标。
4. **固定 decoder 选择。** 记录 greedy、sampling 或 beam，以及 temperature、length penalty、max length。
5. **记录 schedule。** 保存 $q_k$、实际 teacher/model token 比例、随机 seed 和前缀错误率。
6. **检查梯度边界。** 确认 argmax/sample 是否可微、detach 在何处、BPTT 窗口多长。
7. **检查 mask 与分母。** BOS、EOS、PAD、有效 token 数和状态更新 mask 分开核对。
8. **做错误前缀反事实。** 手工替换一个前缀 token，观察后续 logits、state 范数和恢复概率。
9. **测长度曲线。** 按目标长度报告 TF NLL、free-running 错误率和 EOS 命中率。
10. **保存真实轨迹。** 至少保留几条 teacher/free/scheduled 的 token 序列，避免只剩汇总指标。
11. **小数据过拟合后自由运行。** 先确认模型能记住短样本，再判断 schedule 是否改变了部署轨迹。

Teacher forcing 的价值是把正确条件前缀提供给每个训练位置，让局部 token 监督容易优化；它的代价是模型可能没有学会在自己的错误前缀上恢复。理解这条边界后，scheduled sampling、soft rollout、序列级目标和 beam search 都可以被放回各自的位置：它们是在改变训练前缀、梯度估计还是推理搜索，而不是一个模糊的「让生成更稳」开关。

## 相关词条

[Seq2Seq](../rnn-lstm/seq2seq/)

[序列建模](../rnn-lstm/sequence-modeling/)

[时间反向传播](../rnn-lstm/bptt/)

[循环神经网络](../rnn-lstm/rnn/)

[门控循环单元](../rnn-lstm/gru/)

[长短期记忆网络](../rnn-lstm/lstm/)

[Beam search](../inference/beam-search/)

[困惑度](../information-theory/perplexity/)

[标签平滑](../training-nn/label-smoothing/)
