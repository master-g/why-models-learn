---
title: "大规模学习率调度：按更新、token 与训练阶段控制步长"
tags: ["why-models-learn"]
---

大规模学习率调度是把参数更新使用的步长写成训练进度的函数，并明确这个进度按 optimizer update、有效 token、样本、epoch 还是 wall-clock 计数。warmup、稳定阶段和 decay 只能在计数口径、全局 batch、优化器状态、梯度累积、跳过更新和 checkpoint 恢复都清楚时比较。一个常见的调度形状不自动适合所有模型；峰值学习率、warmup 长度和终点学习率必须由稳定性与验证损失共同约束。

本文从单步参数更新开始，区分 micro-step、optimizer step 和有效 token，推导 linear warmup、cosine decay、inverse-square-root 与 WSD 调度。随后用数字例子和可运行输出核对学习率曲线，再处理大 batch、AdamW、梯度裁剪、混合精度、分布式训练与断点恢复，最后给出实验记录和审计清单。

![大规模学习率调度示意：学习率先 warmup 到峰值，再按余弦衰减或 stable-decay 进入训练后段](/assets/pretraining/svg/lr-schedules-at-scale.1.svg)

## 学习率调度的对象

### 学习率乘在什么量前面

最简单的随机梯度更新为

$$
\boldsymbol\theta_{s+1}
=
\boldsymbol\theta_s
-
\eta_s\mathbf g_s,
$$

其中 $s$ 是第 $s$ 次真正改变参数的 optimizer step，$\eta_s$ 是该步学习率，$\mathbf g_s$ 是按既定 loss reduction 得到的梯度。学习率不是损失、梯度范数或 token 数；它只定义更新方向前的尺度。

对 AdamW，参数更新可以抽象成

$$
\boldsymbol\theta_{s+1}
=
(1-\eta_s\lambda)\boldsymbol\theta_s
-
\eta_s
\frac{\widehat{\mathbf m}_s}
{\sqrt{\widehat{\mathbf v}_s}+\varepsilon},
$$

所以同一份 $\eta_s$ 同时影响自适应梯度项和解耦权重衰减项。[优化器](../training-nn/optimizers/)负责定义 $\widehat{\mathbf m}_s$、$\widehat{\mathbf v}_s$ 与状态更新；本文讨论 $\eta_s$ 如何随进度变化。

### 三个进度计数器

一个大规模训练运行至少有三个相关但不同的计数：

|计数|记号|含义|
| --- | --- | --- |
|micro-step|$r$|处理一个 device micro-batch 的前向与反向次数|
|optimizer step|$s$|梯度累积完成并真正更新参数的次数|
|有效 token|$t$|按 tokenizer、padding 和 loss mask 计入训练目标的 token 事件数|

若每次 optimizer update 都汇总相同数量的有效 token $q$，则

$$
t_s
=
\sum_{j=1}^{s}q_j
=
sq.
$$

但变长样本、packing、动态 batch 或跳过更新会让 $q_j$ 不同。epoch 也不适合流式语料：数据池可能无限，重复采样次数和有效 token 数比 epoch 更可复核。

调度器若按 $s$ 计数，必须在每次 optimizer update 后推进；若按 $t$ 计数，应根据实际消费的 token 推进。把每个 micro-step 都当成一次 optimizer step，会让 warmup 提前结束，decay 也会提前发生。

### 学习率调度的参数

一条调度曲线至少要记录以下量：

|参数|记号|作用|
| --- | --- | --- |
|峰值学习率|$\eta_{\max}$|warmup 结束后的最大梯度步长|
|初始学习率|$\eta_0$|第一个有效 optimizer update 的步长|
|warmup 长度|$S_{\mathrm w}$ 或 $T_{\mathrm w}$|达到峰值前的 optimizer step 或 token 数|
|衰减终点|$S_{\mathrm d}$ 或 $T_{\mathrm d}$|decay 结束的 step 或 token 位置|
|终点学习率|$\eta_{\min}$|训练后段保留的最小步长|
|进度单位|$s$、$t$ 或 wall-clock|调度器读取的时间轴|

只记录“用了 cosine”不足以复现训练。还要记录 warmup、总训练长度、终点学习率、step/token 口径、是否在 warmup 前更新参数以及 resume 后的计数器。

## 常见调度形状

### 常数学习率

最简单的调度保持

$$
\eta_s
=
\eta_{\max}.
$$

常数学习率适合做短期对照，也可能适合训练后段的 stable 阶段。它不会自动处理初始化阶段的更新风险，也不会随着接近训练终点而减小步长。使用常数曲线时要报告训练是否在验证损失仍下降时提前结束。

### Linear warmup

从 $\eta_0=0$ 开始的线性 warmup 为

$$
\eta_s
=
\eta_{\max}
\frac{s}{S_{\mathrm w}},
\qquad
0\le s\le S_{\mathrm w}.
$$

若希望第一步不为零，可以使用

$$
\eta_s
=
\eta_0
+
(\eta_{\max}-\eta_0)
\frac{s}{S_{\mathrm w}},
\qquad
0\le s\le S_{\mathrm w}.
$$

这里的 $s=0$ 可以表示尚未更新的初始状态，也可以表示第一步的调度索引。两种约定会产生一个 step 偏移，必须在实现和日志中固定。warmup 只减小早期步长；它不能修复错误的 loss scaling、梯度方向、数据切分或数值溢出。

### Warmup 加 cosine decay

训练总步数为 $S$，warmup 结束后到 $S$ 之间使用余弦衰减：

$$
\eta_s
=
\begin{cases}
\eta_{\max}\dfrac{s}{S_{\mathrm w}},
&0\le s<S_{\mathrm w},\\[6pt]
\eta_{\min}
+
\dfrac{\eta_{\max}-\eta_{\min}}{2}
\left[
1+
\cos\left(
\pi
\dfrac{s-S_{\mathrm w}}{S-S_{\mathrm w}}
\right)
\right],
&S_{\mathrm w}\le s\le S.
\end{cases}
$$

在 $s=S_{\mathrm w}$ 处，衰减段取值为 $\eta_{\max}$；在 $s=S$ 处，取值为 $\eta_{\min}$。余弦曲线在衰减起点的斜率为零，warmup 末端与它的斜率通常不相同，但学习率数值连续。

如果训练实际在 $S_{\mathrm{stop}}<S$ 处结束，模型从未走到 $\eta_{\min}$。如果运行超过 $S$，实现应明确 clamp 到 $\eta_{\min}$ 还是继续外推；未定义的外推会让恢复训练产生不同曲线。

### Inverse-square-root

原始 Transformer 配方常用带 warmup 的 inverse-square-root 形式：

$$
\eta_s
=
d_{\mathrm{model}}^{-1/2}
\min\left(
s^{-1/2},
sS_{\mathrm w}^{-3/2}
\right),
\qquad
s>0.
$$

当 $s\le S_{\mathrm w}$ 时，第二项较小，学习率随 $s$ 线性增加；之后第一项较小，学习率按 $s^{-1/2}$ 衰减。该公式把模型宽度和 warmup 一起写进尺度，但现代训练也常直接把峰值学习率作为需要调参的量。不同调度器不能只比较名称；应比较实际的 $\eta_s$、有效 token 位置和更新范数。

### WSD：warmup、stable、decay

WSD 把训练分成 warmup、stable 和 decay 三段。它可以写成

$$
\eta_s
=
\begin{cases}
\eta_{\max}\dfrac{s}{S_{\mathrm w}},
&0\le s<S_{\mathrm w},\\[6pt]
\eta_{\max},
&S_{\mathrm w}\le s<S_{\mathrm d},\\[6pt]
\eta_{\min}
+
\dfrac{\eta_{\max}-\eta_{\min}}{2}
\left[
1+
\cos\left(
\pi
\dfrac{s-S_{\mathrm d}}{S-S_{\mathrm d}}
\right)
\right],
&S_{\mathrm d}\le s\le S.
\end{cases}
$$

stable 段让训练在一段较长 token 区间使用相同峰值，最后的 decay 可以作为独立的短阶段。它适合需要先消费主要训练预算、再用额外 token 做低学习率收敛的实验，但不应把 stable 阶段的延长写成免费的训练：它会改变有效 token、计算预算和验证曲线。

## 学习率曲线的数字核对

### Warmup 加 cosine 的小例子

取总步数 $S=12$、warmup 步数 $S_{\mathrm w}=3$、峰值 $\eta_{\max}=10^{-3}$、终点 $\eta_{\min}=10^{-4}$。按上面的公式得到：

|optimizer step $s$|训练阶段|$\eta_s$|
| ---: | --- | ---: |
|$0$|warmup 起点|$0.000000000000$|
|$1$|warmup|$0.000333333333$|
|$2$|warmup|$0.000666666667$|
|$3$|峰值|$0.001000000000$|
|$6$|cosine 中段|$0.000775000000$|
|$9$|cosine 后段|$0.000325000000$|
|$12$|decay 终点|$0.000100000000$|

如果某一步梯度固定为 $\mathbf g=(2,-1)$，其范数是 $\sqrt5$。在 $s=1$ 的更新位移范数为

$$
\lVert\Delta\boldsymbol\theta_1\rVert_2
=
0.000333333333\sqrt5
=
0.000745355992.
$$

峰值步 $s=3$ 的位移范数为 $0.002236067978$。这只是固定梯度的几何例子；真实训练中梯度方向、梯度裁剪、Adam 预条件和权重衰减都会改变参数位移。

### 用标准库脚本复算

下面的脚本只使用 Python 标准库，直接实现上表中的 warmup 加 cosine 曲线：

## 运行方法

```python
import math

def warmup_cosine(step, total_steps, warmup_steps, peak, end):
    if step < warmup_steps:
        return peak * step / warmup_steps
    progress = min(
        1.0,
        (step - warmup_steps) / (total_steps - warmup_steps),
    )
    return end + 0.5 * (peak - end) * (
        1.0 + math.cos(math.pi * progress)
    )

for step in (0, 1, 2, 3, 6, 9, 12):
    value = warmup_cosine(step, 12, 3, 1e-3, 1e-4)
    print(f"{step:>2} {value:.12f}")
```

运行输出为：

```text
 0 0.000000000000
 1 0.000333333333
 2 0.000666666667
 3 0.001000000000
 6 0.000775000000
 9 0.000325000000
12 0.000100000000
```

代码把 $s=S$ 夹到终点，避免浮点误差让训练超过终点后继续改变学习率。实际实现还要测试 $S_{\mathrm w}=0$、$S=S_{\mathrm w}$、恢复步数超过 $S$ 和训练提前结束等边界。

### 峰值学习率受稳定性约束

对一维二次目标

$$
f(\theta)
=
\frac12\lambda\theta^2,
\qquad
\nabla f(\theta)=\lambda\theta,
$$

梯度下降的一步为

$$
\theta_{s+1}
=
(1-\eta_s\lambda)\theta_s.
$$

固定 $\lambda>0$ 时，线性系统的稳定条件是

$$
0<\eta_s\lambda<2.
$$

例如 $\lambda=1200$ 时，$\eta_s=0.001$ 的更新因子为 $-0.2$，误差绝对值缩小；$\eta_s=0.002$ 的更新因子为 $-1.4$，误差绝对值放大。深度网络不是一个固定二次函数，但这个例子说明峰值学习率和局部曲率共同决定稳定区间。warmup 只能暂时降低 $\eta_s$，不能替代梯度、激活和损失的数值检查。

## 大规模训练为什么更敏感

### Batch 改变了 step 轴

设每次 optimizer update 消费 $B_{\mathrm{tok}}$ 个有效 token，总训练 token 为 $T$，忽略最后不完整批次，则

$$
S
\approx
\frac{T}{B_{\mathrm{tok}}}.
$$

如果 global batch 增加 $k$ 倍，而总 token 不变，optimizer step 大约减少到原来的 $1/k$。同一条按 step 编写的曲线会在 token 轴上提前结束：warmup 消费的 token 变少，cosine decay 也更早到达终点。

在梯度方向近似稳定、优化器相同且噪声影响可以忽略的局部近似中，固定 token 数时可以用

$$
B'_{\mathrm{tok}}=kB_{\mathrm{tok}},
\qquad
\eta'_{\max}\approx k\eta_{\max}
$$

让每个 token 累计的标量更新量大致相近。这个线性缩放只是起始假设；batch 增大还会降低梯度噪声、改变泛化、改变 Adam 状态和显存占用，必须用验证损失与更新稳定性复核。

### 梯度累积不是额外的 optimizer step

若每个 optimizer step 汇总 $K$ 个 micro-batch，参数只在第 $K$ 个 micro-batch 后更新一次。对于有效 token 数分别为 $q_{s,1},\ldots,q_{s,K}$ 的一次累积，应该先定义有效总量

$$
Q_s
=
\sum_{j=1}^{K}q_{s,j},
$$

再使用与 loss reduction 一致的梯度。例如 token 平均目标可以写成

$$
\mathbf g_s
=
\frac{1}{Q_s}
\sum_{j=1}^{K}
\sum_{i\in\mathrm{micro}\,j}
m_i\nabla_\theta\ell_i.
$$

如果每个 micro-batch 先求平均再对 $K$ 个平均值等权相加，且每个 micro-batch 的有效 token 数不同，得到的梯度与 token 加权平均不同。学习率没有修正这个 reduction 差异；调度器只能控制已经形成的 $\mathbf g_s$ 的步长。

### Loss reduction 会改变学习率含义

若同一批样本的损失从 mean 改为 sum，梯度大约按有效事件数放大，原学习率不再表示同一更新尺度。对变长序列尤其要记录：

1. 分母是样本数、序列数还是有效 token 数；
2. padding 和 label mask 是否进入分母；
3. gradient accumulation 是按 token 加权还是按 micro-batch 加权；
4. loss scaling 和 gradient scaler 是否在 optimizer step 前还原梯度。

这些设置属于训练合同，不应在发现曲线不稳定后只调小峰值学习率来掩盖。

### Adam 的学习率与 moment 状态

Adam 的一阶和二阶状态可以写成

$$
\begin{aligned}
\mathbf m_s
&=
\beta_1\mathbf m_{s-1}
+
(1-\beta_1)\mathbf g_s,\\
\mathbf v_s
&=
\beta_2\mathbf v_{s-1}
+
(1-\beta_2)\mathbf g_s^2,\\
\widehat{\mathbf m}_s
&=
\frac{\mathbf m_s}{1-\beta_1^s},
\qquad
\widehat{\mathbf v}_s
=
\frac{\mathbf v_s}{1-\beta_2^s}.
\end{aligned}
$$

Adam 的归一化会让 $\eta_s$ 不再直接等于原始梯度的坐标步长，但它仍然线性缩放预条件后的方向。重置 optimizer state 而保留 scheduler step，或恢复 scheduler 而丢失 moment state，都会产生与原训练不同的更新。

### Gradient clipping 改变实际步长

全局范数裁剪可写成

$$
\mathbf g_s^{\mathrm{clip}}
=
\mathbf g_s
\min\left(
1,
\frac{\tau}{\lVert\mathbf g_s\rVert_2}
\right),
$$

其中 $\tau$ 是裁剪阈值。发生裁剪时，实际梯度项变成 $\eta_s\mathbf g_s^{\mathrm{clip}}$。应同时记录：

$$
\rho_s
=
\mathbf 1\left\{
\lVert\mathbf g_s\rVert_2>\tau
\right\},
$$

以及裁剪比例、梯度范数和更新范数。若 $\rho_s$ 长期为 1，降低学习率可能减少更新，但也可能只是把梯度爆炸隐藏起来；原因需要由激活、损失、数据和数值日志确认。

### Mixed precision 与跳过更新

混合精度训练可能在检测到 overflow 时跳过某个 optimizer update。此时要区分：

|事件|调度器计数|
| --- | --- |
|完成 micro-batch 反向|micro-step 增加|
|梯度累积完成但 overflow|是否更新参数取决于 scaler，通常不增加有效 optimizer step|
|参数与 optimizer state 成功更新|增加 scheduler step 和有效 update 计数|
|只完成验证或 checkpoint|不改变学习率进度|

如果 overflow 时参数没有更新，但 scheduler 仍然推进，下一次成功更新会使用更低的学习率。这个选择可以有意设计，但必须记录；稳定性问题不能由静默的计数偏移解释。

## 按 token 还是按 step 调度

### 把 step 曲线映射到 token 轴

如果每次更新的有效 token 数为 $q_s$，累计进度是

$$
t_s
=
\sum_{j=1}^{s}q_j.
$$

给定 token 轴上的函数 $\eta(t)$，第 $s$ 次更新使用

$$
\eta_s
=
\eta(t_s).
$$

当 $q_s=q$ 恒定时，step 调度和 token 调度只相差 $t_s=sq$ 的坐标变换。当 batch、序列长度或有效 mask 改变时，按 step 调度的曲线会改变实际 token warmup 和 decay；按 token 调度更接近“消费了多少训练信号”的定义。

按 token 调度也有边界：当前 batch 的 token 数在反向前才知道，scheduler 需要决定使用 batch 开始位置、结束位置还是有效 token 的中心位置。对于可复现训练，选择一种规则并写进 checkpoint。

### Warmup 应按什么比例设定

比起只写 warmup steps，更可复核的参数是 token 比例：

$$
r_{\mathrm w}
=
\frac{T_{\mathrm w}}{T_{\mathrm{train}}},
\qquad
r_{\mathrm d}
=
\frac{T_{\mathrm d}}{T_{\mathrm{train}}}.
$$

当全局 batch 改变而 $r_{\mathrm w}$ 固定时，warmup 消费的 token 仍近似不变；若只固定 warmup steps，warmup 在 token 轴上的占比会改变。实际配置仍要检查首段梯度、更新范数、验证损失和吞吐。

### 训练阶段与计算预算

大规模训练常在固定 token 或固定 compute 下结束。将 warmup、stable 和 decay 视为阶段，可以记录每一阶段的预算：

|阶段|主要目标|需要观察的量|
| --- | --- | --- |
|warmup|让更新从较小步长进入目标区间|梯度范数、overflow、更新比率、验证损失|
|stable|以峰值附近学习率消费主要训练预算|token 进度、loss 斜率、裁剪比例、吞吐|
|decay|减少更新噪声并在终点附近收敛|验证损失、权重漂移、下游回归|

改变某一阶段长度会改变总计算、有效 token 和 optimizer state 演化。[计算最优](../pretraining/compute-optimal/)选择模型与数据的预算分配，本文还要决定这个预算在训练轨迹上如何展开。

## 设计和验证大规模调度

### 先做小网格而不是只试一个峰值

在固定数据、模型、tokenizer、global batch 和评估协议后，可对以下变量做小网格：

|变量|候选方向|
| --- | --- |
|$\eta_{\max}$|按对数间隔覆盖稳定区间与明显过小区间|
|warmup 比例|按 token 或 step 取短、中、长三档|
|decay 终点|训练末端是否保留 $\eta_{\min}$|
|曲线形状|cosine、linear、inverse-square-root、WSD|
|batch 规则|固定 global batch 或随阶段变化|

每次只改变一组明确变量。不同 peak、warmup 和 batch 同时变化时，结果不能说明哪一个因素造成差异。

### 记录更新而不只记录学习率

每个 checkpoint 或固定 token 间隔至少保存：

$$
\left(
s,\,
t_s,\,
\eta_s,\,
\lVert\mathbf g_s\rVert_2,\,
\lVert\Delta\boldsymbol\theta_s\rVert_2,\,
\rho_s,\,
L_{\mathrm{train}},\,
L_{\mathrm{val}}
\right).
$$

更新范数与参数范数的比值为

$$
r_s
=
\frac{\lVert\Delta\boldsymbol\theta_s\rVert_2}
{\lVert\boldsymbol\theta_s\rVert_2+\varepsilon}.
$$

学习率曲线相同而 $r_s$ 差异很大，通常表示梯度规模、Adam 状态、裁剪比例、loss reduction 或权重衰减不同。只画 $\eta_s$ 不能证明训练步长相同。

### 选择 checkpoint

如果训练结束后从多个 checkpoint 中选择验证损失最低者，必须记录：

1. checkpoint 的 token 位置和 scheduler step；
2. 验证集是否被用于选择学习率或只用于最后报告；
3. 是否保存并恢复 optimizer、scheduler、scaler 和数据游标；
4. 最优 checkpoint 与最后 checkpoint 的下游差异。

最后 checkpoint 的学习率可能已经接近 $\eta_{\min}$，但它不一定是验证损失最低点。反过来，过早选择验证集最优 checkpoint 也会把开发集噪声带入调度决策。

## Checkpoint 恢复与分布式边界

### 恢复时要恢复哪些状态

一条可复现的 checkpoint 至少保存：

|状态|为什么影响调度|
| --- | --- |
|model parameters|决定下一次梯度与更新方向|
|optimizer moments|决定 Adam 预条件和 bias correction|
|optimizer step|决定 moment 修正与 step-based schedule|
|scheduler state|决定 warmup、stable、decay 的阶段|
|tokens seen|决定 token-based schedule 的位置|
|gradient scaler|决定 overflow 检测和跳过更新|
|data cursor 与 RNG|决定下一批数据和梯度噪声|

只恢复模型参数再从 step 0 开始 warmup，会产生另一条训练轨迹；只恢复 scheduler 而没有 optimizer moments，也会改变有效步长。

### 分布式训练的 global step

在数据并行中，scheduler 通常应该绑定 global optimizer step，而不是某一张设备的 local micro-step。一次 global update 的有效 token 可近似写成

$$
q_s
=
\sum_{d=1}^{D_{\mathrm{device}}}
\sum_{j=1}^{K_{\mathrm{acc}}}
q_{s,d,j}.
$$

设备掉队、最后不完整 batch、padding 和梯度累积会让各步的 $q_s$ 不同。若代码用固定 tokens_per_step 估计 token 进度，却没有记录真实有效 token，恢复或扩缩容后会出现曲线偏移。

### 扩缩容与恢复

增加设备数会改变 global batch 和每个 optimizer step 的有效 token；减少设备数会改变吞吐和 wall-clock。若目标是保持 token-based schedule，应从 checkpoint 的 tokens_seen 继续，而不是用新的 device count 乘以旧 step。若目标是保持 step-based schedule，应明确接受 token warmup 改变，并重新验证学习率峰值和稳定性。

## 失效模式

**把 micro-step 当作 optimizer step。** 记录梯度累积和真实参数更新次数；scheduler 只在约定的有效更新事件上推进。

**只记录 cosine，不记录参数。** 保存峰值、初始值、warmup、stable、decay、终点、总长度、计数单位、clamp 规则和 step 偏移。

**按 step 调度却改变 batch。** 同时报告有效 token 进度，或改用 token-based schedule。

**用线性 batch scaling 当作定律。** 把 $\eta'_{\max}\approx k\eta_{\max}$ 作为起点，检查梯度噪声、Adam 状态、更新比率和验证损失。

**让 loss reduction 改变更新尺度。** 固定有效 token 分母、padding mask、micro-batch 聚合和 loss scaling。

**梯度裁剪长期触发却只降学习率。** 记录裁剪比例并检查激活、数据、损失和数值溢出；原因未查明时不把裁剪率下降写成稳定性已确认。

**overflow 后静默推进 scheduler。** 分开记录 micro-step、成功 optimizer step、scaler 状态和 scheduler step。

**恢复 checkpoint 时重置状态。** 同时恢复 model、optimizer、scheduler、scaler、token 游标和随机状态。

**把最后 checkpoint 当作最优 checkpoint。** 用固定验证规则比较最后点、最佳点和下游指标，保留选择证据。

**把预训练调度迁移到后训练。** 后训练的数据、目标、batch 和梯度分布不同，需要重新定义峰值、warmup 和 decay。

## 审计清单

审计一条大规模学习率曲线时，至少确认：

|问题|证据|
| --- | --- |
|学习率作用在哪个更新|optimizer step 定义、梯度累积和成功更新计数|
|进度使用什么单位|step、有效 token、样本、epoch 或 wall-clock|
|曲线是否可复算|峰值、初始值、warmup、stable、decay、终点和 clamp|
|更新尺度是否一致|loss reduction、梯度范数、更新范数和裁剪比例|
|大 batch 是否改变曲线|global batch、每步有效 token、token 位置和噪声指标|
|恢复是否连续|optimizer、scheduler、scaler、token 游标和 RNG|
|数值是否稳定|overflow、NaN、跳过更新和梯度裁剪事件|
|结论覆盖什么|训练损失、验证损失、下游指标或部署成本|

学习率调度的结论只能覆盖已记录的计数、优化器、数据和评估协议。换 batch、换 loss reduction 或换 checkpoint 恢复方式后，应重新运行最小对照。

## 相关词条

- [计算最优](../pretraining/compute-optimal/)：决定固定训练预算如何分配到模型与有效 token。
- [缩放定律](../pretraining/scaling-laws/)：提供模型、数据和计算量的经验拟合。
- [预训练](../pretraining/pretraining/)：说明大规模 token 消费、目标函数和 checkpoint。
- [梯度下降](../training-nn/gradient-descent/)：定义学习率乘在梯度前的基础更新。
- [小批次随机梯度下降](../training-nn/minibatch-sgd/)：解释 batch、梯度噪声和梯度累积。
- [优化器](../training-nn/optimizers/)：展开 Adam、moment、权重衰减和参数更新状态。
- [混合精度训练](../training-nn/mixed-precision/)：处理 loss scaling、overflow 和跳过更新。
- [分布式训练](../pretraining/distributed-training/)：处理 global step、并行、通信和 global batch。
- [训练稳定性](../pretraining/training-stability/)：处理梯度异常、数值溢出和稳定性证据。
- [评估与困惑度](../pretraining/evaluation-perplexity/)：定义训练外损失和 checkpoint 评估协议。
