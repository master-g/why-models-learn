---
title: "训练稳定性：让训练轨迹保持有限、可控并可恢复"
tags: ["why-models-learn"]
---

训练稳定性是指训练循环中的数据、激活、损失、梯度、优化器状态和参数更新，在整个运行期间保持有限、尺度可控、持续产生有效进展，并能从 checkpoint 恢复到同一套状态。它不等于 loss 曲线没有尖峰，也不等于某一次运行没有出现 NaN；稳定性要求每一个边界都有可检查的数值合同、更新顺序和恢复协议。

![训练稳定性示意图：数据经过前向、损失、反向、梯度处理和参数更新，每个边界记录有限性与尺度；红色标记表示首个失败位置，绿色路径表示从 checkpoint 恢复](/assets/pretraining/svg/training-stability.1.svg)

## 稳定性是一份训练合同

训练状态可以写成

$$
\mathcal S_s
=
\left(
\theta_s,
\omega_s,
\eta_s,
d_s,
r_s,
c_s
\right),
$$

其中 $\theta_s$ 是参数，$\omega_s$ 是优化器状态，$\eta_s$ 是当前学习率，$d_s$ 是数据游标，$r_s$ 是随机状态，$c_s$ 是混合精度和分布式运行状态。一次成功的 optimizer step 应把 $\mathcal S_s$ 映射成下一份状态，而不是只把一份梯度传给 optimizer.step()。

可以把稳定性拆成四个可验证的性质：

|性质|可检查的合同|违反时的证据|
| --- | --- | --- |
|有限|参与下一步计算的标量、张量、梯度和状态都是 finite|NaN、+Inf、-Inf、非有限比例|
|有界|激活、梯度、参数和更新比保持在实验设定的范围内|范数、分位数、最大值、更新比持续越界|
|进展|训练目标、验证目标或指定能力指标在固定预算内产生可解释变化|长期平台、指标反向、有效 token 消耗后无变化|
|可恢复|参数、优化器、调度器、数据游标、随机状态和缩放器能共同恢复|重启后 step、loss、梯度或数据顺序分叉|

四个性质的观察时间不同。有限性需要逐步检查，有界性需要看分布和趋势，进展需要按固定 token 或 compute 比较，可恢复性需要做一次实际 resume 对照。只记录最终训练 loss，不能覆盖这四层合同。

## 一条更新路径有多个边界

设第 $s$ 次真正改变参数的更新从一批有效 token 开始。把一轮训练展开，可以得到：

$$
\begin{aligned}
x_s
&=\operatorname{batch}(d_s),\\
a_s
&=\operatorname{forward}(\theta_s,x_s),\\
\ell_s
&=\operatorname{loss}(a_s,y_s,m_s),\\
g_s
&=\nabla_{\theta_s}\ell_s,\\
\widetilde g_s
&=\operatorname{unscale}(g_s),\\
\widehat g_s
&=\operatorname{clip}(\widetilde g_s;\tau),\\
(\Delta\theta_s,\omega_{s+1})
&=\mathcal O(\theta_s,\widehat g_s,\omega_s,\eta_s),\\
\theta_{s+1}
&=\theta_s+\Delta\theta_s.
\end{aligned}
$$

这里的 $m_s$ 是 padding 或任务 mask，$\tau$ 是梯度裁剪阈值。实际实现还会插入混合精度缩放、collective 通信、梯度累积、学习率调度和 checkpoint 保存。每个插入点都可能改变稳定性：

|边界|应记录的量|要定位的问题|
| --- | --- | --- |
|数据进入模型前|shape、dtype、有效事件数、最小值、最大值、finite 比例|样本错位、量纲错误、空 mask、非有限输入|
|前向每层后|激活范数、均值、方差、分位数、finite 比例|初始化尺度、饱和、溢出、残差累积|
|损失计算后|每项 loss、reduction 分母、logits 范围、finite 比例|log(0)、softmax 溢出、mask 分母为零|
|反向每层后|梯度范数、最大值、非零比例、finite 比例|局部梯度爆炸、梯度消失、断图|
|优化器更新前|unscale 后范数、clip 系数、skip 标志|缩放顺序错误、局部裁剪、非有限梯度|
|优化器更新后|参数范数、更新范数、更新比、状态范数|学习率过大、状态污染、更新为零|
|同步和 checkpoint 后|rank 一致性、step、数据游标、随机状态、文件 hash|死锁、重复更新、恢复分叉|

先定位边界，再讨论超参数。若损失已经在前向中出现非有限值，继续调学习率只会把后果推迟到反向阶段。

## 第一非有限值比最后的 NaN 更有信息

设每个阶段都有一个有限性谓词：

$$
F(z)
=
\bigwedge_{i\in\operatorname{elements}(z)}
\operatorname{isfinite}(z_i).
$$

训练循环应保存第一个满足 $\neg F(z)$ 的阶段和 step。后续阶段的 NaN 只是传播结果，通常不能作为根因位置。

|阶段|常见首个失败|优先检查的量|
| --- | --- | --- |
|输入|NaN 或 Inf 进入模型|原始数据、归一化分母、token mask、dtype 转换|
|前向激活|某一层输出上溢或变成 NaN|线性层输入、残差相加、归一化方差、激活尾部|
|logits|logits 范围失控|权重尺度、attention score、mask 值、softmax 实现|
|损失|log 或除法产生非有限值|稳定 softmax、目标范围、reduction 分母、标签 mask|
|反向梯度|部分参数梯度先失败|对应层的局部 Jacobian、损失缩放、归约和通信|
|优化器状态|参数梯度有限而 $m$ 或 $v$ 失败|平方梯度、$\varepsilon$、状态 dtype、跳过协议|
|参数更新|状态有限而参数失败|学习率、权重衰减、更新顺序、参数 dtype|
|评估或保存|训练有限但指标或 checkpoint 失败|eval 模式、指标分母、序列化 dtype、文件写入|

loss=NaN 只说明至少一个上游量已经违反合同。需要把检查点沿计算图向前移动，找到第一个变化的边界。

## 标量二次目标给出最小稳定性判据

先看一个只有一个参数的目标：

$$
J(\theta)
=
\frac{\lambda}{2}\theta^2,
\qquad
\lambda>0.
$$

它的梯度为

$$
g(\theta)
=
\frac{\partial J}{\partial\theta}
=
\lambda\theta.
$$

裸 SGD 使用学习率 $\eta$ 更新时，

$$
\begin{aligned}
\theta_{s+1}
&=\theta_s-\eta\lambda\theta_s\\
&=(1-\eta\lambda)\theta_s.
\end{aligned}
$$

令 $r=1-\eta\lambda$，经过 $s$ 步后：

$$
\theta_s
=
r^s\theta_0.
$$

当 $\lvert r\rvert<1$ 时，参数幅值逐步衰减；当 $\lvert r\rvert=1$ 时，可能保持不变或持续振荡；当 $\lvert r\rvert>1$ 时，参数幅值增长。于是得到严格的线性稳定区间：

$$
0<\eta\lambda<2.
$$

这个区间区分了三种不同轨迹：

|$\eta\lambda$ 的位置|$r$ 的范围|参数轨迹|
| --- | --- | --- |
|接近 $0$|接近 $1$|稳定但下降很慢|
|接近 $1$|接近 $0$|快速衰减，可能需要关注噪声和离散化|
|介于 $1$ 与 $2$|介于 $-1$ 与 $0$|符号交替并衰减|
|大于或等于 $2$|$\lvert r\rvert\geq1$|振荡不衰减或发散|

取 $\lambda=1.2$、$\theta_0=1$。当 $\eta=0.5$ 时，$r=0.4$，轨迹为

$$
1,\quad
0.4,\quad
0.16,\quad
0.064,\quad
0.0256,\quad
0.01024,\quad
0.004096.
$$

当 $\eta=2.0$ 时，$r=-1.4$，轨迹为

$$
1,\quad
-1.4,\quad
1.96,\quad
-2.744,\quad
3.8416,\quad
-5.37824,\quad
7.529536.
$$

这个例子没有数据噪声、动量、非线性或混合精度。它只说明学习率与局部曲率的乘积决定了离散更新是否会跨过稳定区间。对真实网络，局部曲率会随参数和 batch 改变，因此 warmup 只能降低早期步长，不能把一个错误的梯度方向变成正确方向。[大规模学习率调度](../pretraining/lr-schedules-at-scale/)展开了按 step 和 token 选择步长的口径。

### 多维参数对应最大曲率

对于正定二次目标，

$$
J(\boldsymbol\theta)
=
\frac12
\boldsymbol\theta^\mathsf T
H\boldsymbol\theta,
$$

其中 Hessian $H$ 的特征值为 $\lambda_1,\ldots,\lambda_d$。梯度下降为

$$
\boldsymbol\theta_{s+1}
=
(I-\eta H)\boldsymbol\theta_s.
$$

沿着特征向量方向 $\boldsymbol q_i$ 分解参数：

$$
\boldsymbol\theta_s
=
\sum_i a_{i,s}\boldsymbol q_i,
\qquad
a_{i,s+1}
=
(1-\eta\lambda_i)a_{i,s}.
$$

每个方向都有自己的标量因子。要让所有方向同时线性稳定，需要

$$
0<\eta<\frac{2}{\lambda_{\max}(H)}.
$$

神经网络的 Hessian 通常非正定、随训练变化，而且优化器可能使用动量或逐坐标预条件。因此这个判据是局部分析工具，不是深度网络的全局保证。它仍然提供了一个可测问题：当学习率提高时，更新比、梯度范数和参数范数是否同时改变。

### 梯度路径会放大或压缩扰动

对多层复合函数，小扰动可以近似写成 Jacobian 的乘积：

$$
\delta h_L
=
J_LJ_{L-1}\cdots J_1\delta h_0.
$$

因此有上界

$$
\lVert\delta h_L\rVert_2
\leq
\left(
\prod_{k=1}^{L}
\lVert J_k\rVert_2
\right)
\lVert\delta h_0\rVert_2.
$$

如果许多层的谱范数小于 $1$，扰动和梯度可能沿深度方向逐步衰减；如果乘积大于 $1$，可能逐步放大。归一化、残差路径、初始化和学习率会共同改变这些 Jacobian，而单看最终 loss 无法区分是哪一段路径贡献了放大。[权重初始化](../training-nn/weight-initialization/)解释初始尺度，[梯度裁剪](../training-nn/gradient-clipping/)处理已经进入反向阶段的异常梯度。

## 有限浮点数会改变训练的数学对象

### 溢出、下溢和非有限传播

浮点计算中的异常有不同来源：

|现象|计算结果|训练中的表现|
| --- | --- | --- |
|上溢|有限值变成 Inf|激活、logits、平方梯度或状态突然变大|
|下溢|非零小量舍入为零|梯度、概率尾部或更新长期为零|
|NaN 传播|未定义结果继续进入后续算子|loss、梯度和参数很快全部失效|
|Inf - Inf|NaN|两个已经上溢的分支相减|
|0 × Inf|NaN|稀疏 mask 与非有限值相乘没有清除异常|
|log(0)|-Inf|交叉熵或似然实现直接取零概率的对数|
|除以零|Inf 或 NaN|空 mask、零方差或错误归一化分母|

有限性检查不能用“把 NaN 替换为零”代替。替换会改变梯度方向，并且可能把数据问题隐藏成一条看似能继续的训练曲线。恢复策略应区分跳过这次更新、降低尺度、回滚 checkpoint 和修复输入。

### 稳定 softmax 要先平移 logits

直接计算

$$
p_i
=
\frac{\exp(z_i)}
{\sum_j\exp(z_j)}
$$

会在 $z_i$ 较大时上溢，即使最终概率本来处于 $[0,1]$。令

$$
m=\max_j z_j,
$$

利用分子分母同时除以 $\exp(m)$，得到数值更稳定的形式：

$$
p_i
=
\frac{\exp(z_i-m)}
{\sum_j\exp(z_j-m)}.
$$

此时指数的最大输入为 $0$。这只处理指数计算的范围，不会修复已经为 NaN 的 logits，也不会替代正确的 padding 和 causal mask。[Softmax](../neurons-and-activations/softmax/)进一步展开归一化、温度和梯度。

### reduction 分母必须可审计

对每个位置的 loss $\ell_i$ 和有效 mask $m_i\in\{0,1\}$，按有效事件平均时：

$$
Q=\sum_i m_i,
\qquad
\mathcal L
=
\frac{\sum_i m_i\ell_i}{Q}.
$$

当 $Q=0$ 时，这个 batch 没有有效监督，直接相除会产生非有限值。实现必须在进入除法前记录 $Q$，并明确选择跳过 batch、返回零损失并不更新，或把数据合同判为失败。把空 batch 静默计入平均，会同时改变 loss 的尺度和学习率的有效含义。[下一词最大似然](../pretraining/next-token-as-mle/)中的 token-level loss 也依赖相同的有效 token 分母。

## 用尺度指标观察“有限但失控”

有限性通过不代表训练稳定。某些运行会保持 finite，却让参数范数、激活范围或更新步长逐步增长。至少需要保存以下指标：

$$
G_s
=
\lVert g_s\rVert_2,
\qquad
U_s
=
\lVert\Delta\theta_s\rVert_2,
\qquad
R_s
=
\frac{U_s}
{\lVert\theta_s\rVert_2+\varepsilon}.
$$

其中 $R_s$ 是 update-to-weight ratio。它不是普适阈值，而是跨模型、层和训练阶段比较更新相对参数尺度的量。还可以保存梯度裁剪系数：

$$
c_s
=
\min\left(
1,
\frac{\tau}{G_s+\varepsilon}
\right).
$$

|指标|至少保存的聚合|解释边界|
| --- | --- | --- |
|finite fraction|张量中 finite 元素比例|比例下降时记录首个 tensor 和 rank|
|范数|全局、按层、按参数组的 L2 范数|全局值会隐藏局部层的异常|
|分位数|p50、p90、p99、最大值|最大值容易受单个异常元素影响|
|梯度非零比例|非零且 finite 的元素比例|低比例可能来自稀疏结构，也可能来自下溢|
|clip fraction|被裁剪的 batch 或 update 比例|长期接近 1 表示阈值、学习率或梯度尺度需要复查|
|update ratio|$R_s$ 及其按层分布|更新为零、突增和层间差异都值得定位|
|优化器状态|$m$、$v$ 的范数和 finite 比例|状态失败可能晚于梯度失败|
|loss scale|当前缩放器、增长/回退次数、跳过次数|跳过次数是训练状态的一部分|
|有效 token|本地与 global 有效事件数|loss、学习率和比较预算的共同坐标|

记录均值时还要记录样本数量。一个小批次的均值、一个大批次的均值和一个被 mask 后的均值不能直接比较；百分位数也需要固定采样规则，否则不同运行的统计量没有同一口径。

## 混合精度、梯度裁剪和有限性检查必须按顺序组合

### 一个可复核的更新顺序

设损失缩放器使用正数 $a_s$。反向传播看到的是缩放损失：

$$
\widetilde{\mathcal L}_s
=
a_s\mathcal L_s,
\qquad
\widetilde{\boldsymbol g}_s
=
a_s\boldsymbol g_s.
$$

因此裁剪前必须反缩放：

$$
\boldsymbol g_s
=
\frac{\widetilde{\boldsymbol g}_s}{a_s}.
$$

对一个同步的 global update，可以使用下面的合同：

1. 计算 $\mathcal L_s$ 并乘以 $a_s$。
2. 执行 backward，得到缩放梯度。
3. 除以 $a_s$，把梯度还原到未缩放尺度。
4. 检查本地梯度和本地状态的 finite 标志。
5. 在分布式 group 内归约 finite 标志，形成 global skip 标志。
6. 对 global gradient 计算范数并执行一次全局梯度裁剪。
7. 在没有 skip 时更新优化器状态和参数。
8. 只在参数确实更新时推进成功的 optimizer step。
9. 按已定义的策略更新学习率调度器和 loss scaler。
10. 保存 step、缩放器、梯度统计和 checkpoint 状态。

第 5 步的通信位置可以因实现而变。某些实现先对缩放梯度做通信，再在归约结果上反缩放；只要所有 rank 使用相同尺度，并且裁剪使用反缩放后的 global gradient，数值合同仍然清楚。需要记录实际顺序，不能只记录“开启 AMP”。

### 跳过一次更新时哪些状态不动

当任意 rank 的梯度为非有限值时，所有 rank 都应使用同一个 skip 决策：

$$
b_{\mathrm{global}}
=
\max_{r\in\mathcal R}b_r,
\qquad
b_r\in\{0,1\}.
$$

若 $b_{\mathrm{global}}=1$，通常保持以下状态不变：

|状态|跳过时的常见处理|必须记录的例外|
| --- | --- | --- |
|参数 $\theta$|不更新|是否回滚到最近 checkpoint|
|Adam 的 $m,v$|不写入本次梯度|某些实现可能单独衰减状态|
|成功 optimizer step|不递增|另记 attempted update|
|学习率调度器|按项目合同决定|按 attempted step 计数会改变曲线|
|loss scaler|降低或保持并记录|增长窗口是否从零重新计数|
|数据游标|是否消费本批次要固定|重放 batch 还是跳过 batch|
|随机状态|与数据处理保持一致|恢复后是否要求 bitwise 对齐|

[混合精度训练](../training-nn/mixed-precision/)展开 loss scaling、主权重和 overflow skip；这里关注的是它与更新状态如何组合。若只有本地 rank 跳过，其他 rank 继续进入 collective 或 optimizer step，参数副本会分叉，并可能在下一次通信中死锁。

### 裁剪必须发生在 global gradient 上

若有多个数据并行 rank，第 $r$ 个 rank 的局部梯度为 $\boldsymbol g^{(r)}$。先分别裁剪再平均，通常不等于先平均再裁剪：

$$
\frac{1}{R}\sum_{r=1}^{R}
\operatorname{clip}
\left(
\boldsymbol g^{(r)};\tau
\right)
\neq
\operatorname{clip}
\left(
\frac{1}{R}\sum_{r=1}^{R}\boldsymbol g^{(r)};\tau
\right).
$$

更接近全局合同的做法是先形成正确归一化的 global gradient，再计算 global norm。对于已经分片的梯度，可以用局部平方和归约：

$$
G^2
=
\sum_{r=1}^{R}
\sum_i
\left(g_i^{(r)}\right)^2,
\qquad
c=\min\left(1,\frac{\tau}{\sqrt{G^2}+\varepsilon}\right).
$$

然后每个 rank 用同一个 $c$ 缩放自己持有的分片。[分布式训练](../pretraining/distributed-training/)展开有效 token 加权、reduce-scatter 和参数分片；这里的关键是有限性、范数和 skip 决策都必须有 global 语义。

## Adam 状态也属于稳定性

Adam 在第 $s$ 次成功更新时维护：

$$
\begin{aligned}
\boldsymbol m_s
&=
\beta_1\boldsymbol m_{s-1}
+
(1-\beta_1)\boldsymbol g_s,\\
\boldsymbol v_s
&=
\beta_2\boldsymbol v_{s-1}
+
(1-\beta_2)\boldsymbol g_s^{\odot2},\\
\widehat{\boldsymbol m}_s
&=
\frac{\boldsymbol m_s}{1-\beta_1^s},\\
\widehat{\boldsymbol v}_s
&=
\frac{\boldsymbol v_s}{1-\beta_2^s}.
\end{aligned}
$$

参数更新可以抽象为

$$
\Delta\boldsymbol\theta_s
=
-\eta_s
\frac{\widehat{\boldsymbol m}_s}
{\sqrt{\widehat{\boldsymbol v}_s}+\varepsilon}.
$$

这里的 $s$ 必须与状态更新次数一致。若 overflow skip 后仍然增加偏置修正用的 $s$，下一次有效更新会使用与历史不同的修正因子；若 scheduler 同时按另一套 attempted step 计数，学习率曲线也会分叉。[Adam](../training-nn/adam/)给出状态更新和偏置修正的具体推导。

参数有限而状态非有限时，下一次参数更新仍可能失败。检查顺序应覆盖：

1. 反缩放后的梯度；
2. 裁剪前后的梯度；
3. 一阶状态 $\boldsymbol m$；
4. 二阶状态 $\boldsymbol v$；
5. 偏置修正后的状态；
6. 参数更新 $\Delta\boldsymbol\theta$；
7. 新参数和权重衰减结果。

优化器状态不能只在 checkpoint 里保存，训练日志也应至少保存它们的范数、finite 标志和更新计数。状态 dtype 低于参数 dtype 时，还要单独记录累加器精度。

## 有效进展与数值稳定是两份证据

一条完全 finite 的曲线可能没有学习，也可能优化了错误目标。需要把数值指标和任务指标并列：

|观察|数值上可能稳定|仍需核对|
| --- | --- | --- |
|训练 loss 下降|梯度和更新有限|验证 loss、去重后的数据覆盖、token 预算|
|loss 几乎不变|更新比过小或梯度下溢|梯度非零比例、学习率、标签和 loss reduction|
|loss 快速下降|更新稳定但可能过拟合或泄漏|独立切分、污染证据、下游任务|
|clip fraction 很高|裁剪阻止了明显发散|原始梯度尺度、阈值、学习率和局部层|
|loss scale 持续回退|避免了立即溢出|哪一层先溢出、有效更新比例、吞吐|
|训练 loss 有周期|调度器和数据顺序可能稳定|按有效 token 对齐后周期是否仍存在|
|验证指标偶尔尖峰|单批次噪声或评估分母变化|逐样本指标、mask、eval 状态和 checkpoint|

固定比较轴很重要。对大规模训练，使用 optimizer step 比较两个运行可能把不同的 global batch、有效 token 或跳过次数混在一起；应按 token、compute 和 wall-clock 分别记录。[训练数据](../pretraining/training-data/)说明数据 provenance、过滤和有效 token mixture；训练稳定性检查不能用一条漂亮的 loss 曲线替代数据合同。

## 训练失败的定位协议

### 先冻结配置和证据

定位前先保存：

- 代码版本和依赖版本；
- 模型配置、数据 manifest、tokenizer 和 loss mask 规则；
- seed、rank、global batch、梯度累积和 scheduler 参数；
- dtype、AMP、loss scaler、optimizer 和 checkpoint 路径；
- 最近一次成功 checkpoint 的 step 与 hash；
- 从最近成功 step 开始的 loss、有效 token 和状态摘要。

没有这些信息时，重新运行可能同时改变数据顺序、初始化、通信布局和调度曲线，导致问题无法复现。

### 再把搜索区间缩小

从最近成功 checkpoint 重新运行，并以固定间隔保存摘要。若 step $s_a$ 有限、step $s_b$ 非有限，可以对区间做二分：

1. 从 checkpoint $s_a$ 恢复。
2. 运行到中点 $s_m$。
3. 检查所有边界的 finite 标志。
4. 把区间缩小到仍包含首个失败的半段。
5. 重复直到得到单个 step 或很小的 batch 区间。

在这个区间内再打开按层 hook，记录第一层出现非有限值的 tensor 名称。hook 应保持相同 dtype、reduction 和通信顺序；为了打印统计量而把整个张量复制到 CPU，可能改变异步执行和显存压力，需明确这项观测开销。

### 最后做恢复对照

找到首个失败位置后，构造至少三条恢复路径：

|路径|目的|结果解释|
| --- | --- | --- |
|原配置重跑|确认失败是否可重复|失败 step 和首个 tensor 应接近|
|降低峰值学习率或增加 warmup|检验步长边界|失败推迟但首个激活仍异常时，根因未定位|
|修复数值边界后恢复|验证修复是否改变合同|finite、更新比、clip fraction 和任务指标共同改善|

只看到“NaN 不再出现”还不够。若 loss scaler 把所有更新都跳过，曲线也可能保持有限但没有进展；需要同时检查成功更新数、有效 token、参数变化和验证指标。

## 一个标准库探针

下面的代码只使用 Python 标准库，分别核对标量更新的稳定区间、全局范数裁剪、update-to-weight ratio 和首个非有限阶段。它不模拟真实 GPU dtype 或 collective，因此输出用于检查公式和日志字段，不能替代目标设备运行。

```python
from math import isfinite, sqrt


def trajectory(eta, lam=1.2, theta0=1.0, steps=6):
    factor = 1.0 - eta * lam
    theta = theta0
    values = []
    for _ in range(steps + 1):
        values.append(theta)
        theta = factor * theta
    return factor, values


for eta in (0.5, 2.0):
    factor, values = trajectory(eta)
    rounded = [round(value, 6) for value in values]
    print(
        f"eta={eta:.1f} factor={factor:.1f} "
        f"trajectory={rounded}"
    )

gradient = (3.0, 4.0)
threshold = 2.0
gradient_norm = sqrt(sum(value * value for value in gradient))
clip_coef = min(1.0, threshold / gradient_norm)
clipped = tuple(clip_coef * value for value in gradient)
print(
    f"gradient_norm={gradient_norm:.12f} "
    f"clip_coef={clip_coef:.12f} "
    f"clipped={tuple(round(value, 12) for value in clipped)}"
)

theta = (1.0, -2.0)
delta = (-0.01, 0.02)
theta_norm = sqrt(sum(value * value for value in theta))
update_norm = sqrt(sum(value * value for value in delta))
update_ratio = update_norm / theta_norm
print(
    f"theta_norm={theta_norm:.12f} "
    f"update_norm={update_norm:.12f} "
    f"update_ratio={update_ratio:.12f}"
)

stages = [
    ("input", (1.0, 0.5)),
    ("forward", (2.0, 1.5)),
    ("loss", (float("nan"),)),
    ("backward", (float("nan"),)),
    ("update", (float("nan"),)),
]
first_bad = None
for name, values in stages:
    if not all(isfinite(value) for value in values):
        first_bad = name
        break

print(
    f"first_nonfinite_stage={first_bad} "
    f"checked_stages={len(stages)}"
)
```

输出为：

```text
eta=0.5 factor=0.4 trajectory=[1.0, 0.4, 0.16, 0.064, 0.0256, 0.01024, 0.004096]
eta=2.0 factor=-1.4 trajectory=[1.0, -1.4, 1.96, -2.744, 3.8416, -5.37824, 7.529536]
gradient_norm=5.000000000000 clip_coef=0.400000000000 clipped=(1.2, 1.6)
theta_norm=2.236067977500 update_norm=0.022360679775 update_ratio=0.010000000000
first_nonfinite_stage=loss checked_stages=5
```

第一行和第二行把 $\eta\lambda$ 对稳定区间的影响变成了可复算轨迹。第三行验证范数裁剪保留方向并把范数从 $5$ 缩到 $2$。第四行显示更新范数相对于参数范数为 $1\%$。最后一行说明有限性检查应在损失阶段停止，而不是等到 update 阶段才报告 NaN。

## 运行方法

把上一节的代码保存为 training_stability_probe.py，然后运行：

```bash
python3 training_stability_probe.py
```

要把探针接入真实训练，保留同样的输出字段，并为每个字段增加 step、rank、layer、dtype、checkpoint 标识。探针不应把非有限值静默替换为零，也不应在失败后继续写入新的 optimizer state。

## 失效模式

### 只看 loss

训练 loss 仍然有限时，某个层的激活可能已经接近 dtype 上限，或更新比已经远高于历史范围。只看 loss 会把首个失败位置和稳定性趋势隐藏起来。至少同时记录边界 finite 标志、按层范数、clip fraction、update ratio 和成功更新数。

### 用梯度裁剪掩盖前向错误

梯度裁剪只能限制进入优化器的梯度。它不修复输入、激活、logits、loss 或 optimizer state 的非有限值。若原始梯度长期被裁剪，先保存裁剪前证据，再检查学习率、初始化、损失实现和数据范围。

### 每个 rank 独立决定 skip

某一个 rank 发现 Inf 而其他 rank 继续更新，会让参数副本分叉。若后续 collective 需要所有 rank 参与，局部提前退出还可能造成死锁。finite 标志、clip 系数和 skip 决策都要有 global 语义。

### overflow skip 仍推进所有计数

把 attempted update、成功 optimizer step、scheduler step、loss scaler step 混成一个整数，会让 Adam 偏置修正、学习率曲线和数据游标出现不同步。日志中分开保存这些计数，并在 checkpoint 中恢复它们。

### checkpoint 只保存参数

只恢复 $\theta$ 而不恢复 $m$、$v$、更新计数、scheduler、scaler、随机状态和数据游标，会得到一条新的训练轨迹。新轨迹可能仍能下降，但它不再是原运行的 resume。

### 只裁剪每卡局部梯度

局部范数低于阈值，不代表 global gradient 低于阈值。数据并行与参数分片场景要用全局平方和或等价的 reduce-scatter 统计，再用同一个缩放系数处理各分片。

### 以 step 对齐不同运行

global batch、有效 token、跳过次数或 packing 长度不同，两个运行的同一个 step 不代表相同训练量。比较曲线时同步保存 step、有效 token、compute 和 wall-clock。

### 通过替换 NaN 继续训练

把 NaN 转成零会消除显式失败信号，却改变目标函数和更新方向。恢复流程应记录失败、保留最近 checkpoint，并在修复合同后重新运行。

## 稳定性审计清单

|范围|确认项|证据|
| --- | --- | --- |
|数据|输入、标签、mask、有效 token 和 dtype 有合同|manifest、样本摘要、mask 分母|
|前向|逐层激活 finite，范围和分位数有记录|层级统计、首个非有限 tensor|
|损失|softmax、log、reduction 和空 mask 有明确处理|单样本手算、loss 分项、有效事件数|
|反向|梯度 finite、非零比例和 global norm 可见|unscale 后统计、按层范数|
|裁剪|裁剪发生在 global gradient，原始范数保留|clip 系数、clip fraction、通信合同|
|优化器|状态 dtype、更新计数和状态 finite 可见|$m$、$v$、step、update ratio|
|调度|按 update 或 token 的计数与 skip 规则固定|scheduler 参数、step/token 日志|
|分布式|global finite、global clip 和 collective 顺序一致|rank flag、group、通信日志|
|checkpoint|参数、状态、数据和随机状态一起恢复|checkpoint manifest、hash、resume 对照|
|进展|训练与验证指标按固定预算比较|token、compute、wall-clock、下游指标|

通过这份清单后，才能把一次训练的 loss 变化解释为优化过程，而不是把运行时异常、数据合同错误和指标口径混在一起。

## 相关词条

- [大规模学习率调度](../pretraining/lr-schedules-at-scale/)
- [分布式训练](../pretraining/distributed-training/)
- [混合精度训练](../training-nn/mixed-precision/)
- [梯度裁剪](../training-nn/gradient-clipping/)
- [Adam](../training-nn/adam/)
- [训练调试](../training-nn/debugging-training/)
- [权重初始化](../training-nn/weight-initialization/)
- [Softmax](../neurons-and-activations/softmax/)
- [训练数据](../pretraining/training-data/)
- [下一词最大似然](../pretraining/next-token-as-mle/)
