---
title: "Adam：一阶动量、二阶尺度与偏置修正"
tags: ["why-models-learn"]
---

Adam 把动量的方向平均和平方梯度的尺度平均放在同一个更新里，再用偏置修正抵消“状态从零开始”带来的早期缩小。它常被当作一个优化器名称直接交给框架，但真正决定训练轨迹的还有 $\beta_1$、$\beta_2$、$\varepsilon$、参数组、权重衰减语义、更新计数和学习率调度。本篇从逐步公式开始，解释每个状态量为何存在，用标准库实验区分偏置修正和 $\varepsilon$ 的实现差异，最后把 Adam 放回真实训练的梯度累积、混合精度、AdamW 与 checkpoint 协议中。

![Adam 更新图：梯度分成一阶方向和二阶尺度两条指数平均路径，经偏置修正后合成为参数位移](/assets/training-nn/svg/adam.1.svg)

## Adam 在优化器家族中的位置

设第 $t$ 次参数更新前有参数 $\boldsymbol\theta_t$ 和有效梯度 $\boldsymbol g_t$。SGD 只把梯度乘以学习率，动量在时间轴上保存历史方向；Adam 同时维护一阶和二阶的逐坐标统计量：

$$
\boldsymbol m_t\quad\text{记录梯度方向的指数平均},
\qquad
\boldsymbol v_t\quad\text{记录平方梯度的指数平均}.
$$

这里的“一阶”和“二阶”是对梯度随机变量的矩的称呼，不是说 Adam 保存了目标函数的 Hessian。$\boldsymbol v_t$ 的每个坐标只看对应的 $g_{t,i}^2$，没有描述参数坐标之间的旋转或耦合。因此 Adam 是逐坐标的自适应方法，不是 Newton 方法。

从外部看，一步 Adam 可以写成

$$
\begin{aligned}
(\boldsymbol u_t,\boldsymbol s_{t+1})
&=\mathcal O_{\mathrm{Adam}}
(\boldsymbol\theta_t,\boldsymbol g_t,\boldsymbol s_t,\eta_t,\boldsymbol h),\\
\boldsymbol\theta_{t+1}
&=\boldsymbol\theta_t+\boldsymbol u_t,
\end{aligned}
$$

其中 $\boldsymbol s_t$ 至少包含 $\boldsymbol m_t$、$\boldsymbol v_t$ 和更新计数 $t$。如果只保存参数而不保存这些状态，恢复后的程序虽然仍然能计算梯度，却不再是原来的 Adam 轨迹。

## 一步更新到底做了什么

先约定 $t$ 从 $1$ 开始，$\boldsymbol m_0=\boldsymbol0$、$\boldsymbol v_0=\boldsymbol0$。给定梯度 $\boldsymbol g_t$，Adam 先做两次指数滑动平均：

$$
\boldsymbol m_t
=\beta_1\boldsymbol m_{t-1}
+(1-\beta_1)\boldsymbol g_t,
$$

$$
\boldsymbol v_t
=\beta_2\boldsymbol v_{t-1}
+(1-\beta_2)\boldsymbol g_t^{\odot 2},
$$

其中 $\boldsymbol g_t^{\odot 2}$ 表示逐坐标平方，$0\leq\beta_1,\beta_2<1$。$\beta_1$ 控制方向平均保留多少过去，$\beta_2$ 控制平方梯度尺度保留多少过去。

由于两个状态都是从零开始，它们早期会偏小。Adam 用当前步数做修正：

$$
\widehat{\boldsymbol m}_t
=\frac{\boldsymbol m_t}{1-\beta_1^t},
\qquad
\widehat{\boldsymbol v}_t
=\frac{\boldsymbol v_t}{1-\beta_2^t}.
$$

最后按坐标计算梯度位移：

$$
\boldsymbol u_t
=-\eta_t
\frac{\widehat{\boldsymbol m}_t}
{\sqrt{\widehat{\boldsymbol v}_t}+\varepsilon},
\qquad
\boldsymbol\theta_{t+1}
=\boldsymbol\theta_t+\boldsymbol u_t.
$$

分子和分母都是向量逐坐标运算。$\varepsilon>0$ 防止分母为零，也决定极小二阶统计量附近的数值行为。常见默认值 $10^{-8}$ 不是数学上唯一的常数；改变它可能改变低精度或极小梯度下的更新。

把一次更新拆成这四步有助于排查问题：

1. 当前梯度是否已经完成正确的 batch 归约；
2. $\boldsymbol m_t$ 和 $\boldsymbol v_t$ 是否使用了同一个有效梯度；
3. 偏置修正的 $t$ 是否从正确的 optimizer step 计数；
4. 参数更新、权重衰减和学习率调度是否采用了约定的顺序。

## 偏置修正不是可选的装饰

先看最简单的情况：每一步都收到同一个标量梯度 $g_t=g$。由 $\boldsymbol m_0=0$ 可递推得到

$$
m_t
=(1-\beta_1^t)g.
$$

同理，平方梯度状态为

$$
v_t
=(1-\beta_2^t)g^2.
$$

所以偏置修正后

$$
\widehat m_t=g,
\qquad
\widehat v_t=g^2.
$$

当 $\varepsilon$ 很小时，Adam 的方向约为

$$
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}
\approx\operatorname{sign}(g).
$$

不做修正时，第一步的两个状态仍是 $(1-\beta_1)g$ 和 $(1-\beta_2)g^2$。取 $g=2$、$\beta_1=0.9$、$\beta_2=0.999$，第一步得到

$$
m_1=0.2,\qquad v_1=0.004,
$$

但正确修正后是 $\widehat m_1=2$、$\widehat v_1=4$。下面是连续三步的标准库输出；方向一列使用了偏置修正，未修正方向直接把 $m_t$ 和 $v_t$ 代入分式：

| 步数 | $m_t$ | $v_t$ | $\widehat m_t$ | $\widehat v_t$ | 修正后方向 | 未修正方向 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.200000 | 0.004000 | 2.000000 | 4.000000 | 1.000000 | 3.162277 |
| 2 | 0.380000 | 0.007996 | 2.000000 | 4.000000 | 1.000000 | 4.249591 |
| 3 | 0.542000 | 0.011988 | 2.000000 | 4.000000 | 1.000000 | 4.950235 |

未修正方向不但不是 $g/\lvert g\rvert$，还会随着步数上升。原因是分子按 $1-\beta_1^t$ 缩小，分母按 $\sqrt{1-\beta_2^t}$ 缩小；这两个因子通常不相等。偏置修正让“从零开始的状态”与“已经观察过很长历史的指数平均”在早期有可比较的尺度。

这不意味着修正后 Adam 的每一步都只由符号决定。$\varepsilon$、梯度随时间变化、$\beta_1$ 的方向滞后和 $\beta_2$ 的尺度滞后都会改变实际方向；符号近似只是稳定常值梯度下的第一步直觉。

## 第一小步为何容易接近固定长度

在第 $1$ 步，若梯度每个坐标都非零，偏置修正几乎恢复了

$$
\widehat m_{1,i}=g_{1,i},
\qquad
\widehat v_{1,i}=g_{1,i}^2.
$$

于是

$$
u_{1,i}
\approx-\eta_1\operatorname{sign}(g_{1,i}).
$$

例如 $g_{1,1}=1000$ 和 $g_{1,2}=0.001$，在 $\varepsilon$ 可以忽略时，这两个坐标的首步绝对位移都接近 $\eta_1$。这正是 Adam 对初始尺度不那么敏感的来源之一，也解释了为什么把 SGD 的学习率原样搬给 Adam 会产生不同的参数轨迹。

但“近似固定长度”不是安全边界。若 $g$ 与 $\varepsilon$ 同量级，分母中的 $\varepsilon$ 会明显缩短步长；若梯度方向不断翻转，$\widehat m_t$ 会发生抵消；若某坐标长期没有梯度，$\widehat v_t$ 会衰减而使有效步长重新变大。首步直觉只能帮助理解，不能代替训练日志。

## $\varepsilon$ 放在哪里会改变什么

常见公式把 $\varepsilon$ 放在平方根外：

$$
d_{\mathrm{out}}
=\sqrt{\widehat v_t}+\varepsilon,
\qquad
u_t=-\eta_t\frac{\widehat m_t}{d_{\mathrm{out}}}.
$$

也有实现把它放进平方根：

$$
d_{\mathrm{in}}
=\sqrt{\widehat v_t+\varepsilon},
\qquad
u_t=-\eta_t\frac{\widehat m_t}{d_{\mathrm{in}}}.
$$

如果 $\widehat v_t$ 远大于 $\varepsilon$，两者差异很小；如果尺度很小，差异会变大。令 $\widehat m_t=\widehat v_t^{1/2}=g>0$，$\varepsilon=10^{-8}$，两种写法的方向为：

| $g$ | 根号外的 $\varepsilon$ | 根号内的 $\varepsilon$ |
| ---: | ---: | ---: |
| $10^{-3}$ | 0.9999900001 | 0.9950371902 |
| $10^{-6}$ | 0.9900990099 | 0.0099995000 |
| $10^{-9}$ | 0.0909090909 | 0.0000100000 |

这些极小值不是常见训练的全部场景，却足以说明“公式看起来只差一行”会造成可测的差异。复现某个结果时，要记录框架版本、$\varepsilon$ 的默认值和它在实现中的位置，不能只写“使用 Adam”。

低精度还会放大这个问题。若 $\boldsymbol v_t$ 在半精度中下溢为零，$\varepsilon$ 既是数值保险，也是实际分母的一部分；通常应在更高精度的状态中保存矩，并在 unscale 后再更新。

## 两个 $\beta$ 是两条时间尺度

指数平均可以展开成

$$
\boldsymbol m_t
=(1-\beta_1)\sum_{j=0}^{t-1}\beta_1^j\boldsymbol g_{t-j},
$$

$$
\boldsymbol v_t
=(1-\beta_2)\sum_{j=0}^{t-1}\beta_2^j\boldsymbol g_{t-j}^{\odot2}.
$$

距当前 $j$ 步的历史权重按 $\beta^j$ 衰减。常用的数量级是有效记忆长度 $1/(1-\beta)$，若用半衰期衡量，则

$$
h_{1/2}=\frac{\log(1/2)}{\log\beta}.
$$

对 $\beta_1=0.9$ 与 $\beta_2=0.999$：

| 状态 | 参数 | $1/(1-\beta)$ | 半衰期（更新步） | 主要作用 |
| --- | ---: | ---: | ---: | --- |
| 一阶矩 | 0.9 | 10 | 6.5788 | 平滑方向、保留惯性 |
| 二阶矩 | 0.999 | 1000 | 692.8005 | 平滑尺度、抑制近期尖峰 |

“1000 步”是更新步，不是样本数，也不是 epoch。若有效 batch 从 32 变成 1024，每个 epoch 的 optimizer step 变少，矩看到的时间轴也跟着改变。调整 batch size 时，不能只按 epoch 复制 $\beta$ 的设置；至少要同时报告 step 数和样本访问量。

$\beta_1$ 变大，方向更平滑但响应更慢；$\beta_2$ 变大，分母更稳定但忘记异常尺度更慢。两条时间尺度组合起来后，Adam 可能仍沿着旧方向移动，同时用更早以前的平方梯度控制当前步长。

## 一个标准库实验：拆开偏置修正与衰减语义

下面的代码只使用 Python 标准库，包含三个可复核的小实验：

- 对常值梯度打印矩、偏置修正和未修正方向；
- 在二维二次目标上比较标准 Adam、去掉偏置修正的变体和根号内 $\varepsilon$；
- 在固定梯度上比较耦合 L2 与解耦的 AdamW 式权重衰减。

将代码保存后运行，输出中的每一列都可以从前面的状态方程逐步重算。

## 二维目标上的状态对照

取一个条件数为 $9$ 的二维二次目标：

$$
F(\boldsymbol\theta)
=\frac12\boldsymbol\theta^{\mathsf T}
\begin{bmatrix}
1&0\\
0&9
\end{bmatrix}
\boldsymbol\theta,
\qquad
\boldsymbol\theta_0=(1,1)^{\mathsf T}.
$$

第二个坐标的曲率是第一个坐标的 $9$ 倍。这个目标不模拟真实网络，却能把方向平均、逐坐标尺度和更新计数的影响隔离出来。代码会比较三种实现：

| 实现 | 改变的地方 |
| --- | --- |
| 标准 Adam | 一阶、二阶矩都做偏置修正，$\varepsilon$ 在根号外 |
| no-bias | 保留矩更新，但直接使用未修正的 $m_t$、$v_t$ |
| inside-eps | 使用偏置修正，但把 $\varepsilon$ 放进平方根 |

所有方法都使用 $\eta=0.05$。这不是说它们的最优学习率相同，而是让同一个标量在不同更新协议下产生可观察的差异。

## 耦合与解耦衰减的单坐标实验

再取一个原始梯度恒为 $0.2$ 的单坐标过程，设置 $\eta=0.1$、$\lambda=0.1$。耦合版本每一步把

$$
\widetilde g_t=0.2+0.1\theta_t
$$

写进 Adam 的两个状态；解耦版本让状态始终只看到 $0.2$，并在参数更新时额外乘以 $0.99$。这组实验只为比较状态输入和参数收缩的顺序，不是某个真实任务的调参建议。

## 运行结果怎样读

常值梯度的三步输出应显示修正后方向稳定在约 $1$，未修正方向从 $3.162277$ 增长到 $4.950235$。这验证了偏置修正的代数推导。

在二维目标上，标准 Adam 的第 $10$ 步 loss 为 $1.312222734$，最终参数约为 $(0.512293423,0.512293418)$。去掉偏置修正后，第 $5$ 步 loss 虽降到 $0.046738905$，第 $10$ 步却回到 $1.388269098$，最终参数约为 $(-0.526928708,-0.526928662)$。不同轨迹来自更新尺度不同，不应把这一次短轨迹解读成某个变体普遍更好。

根号内的 $\varepsilon$ 在正常尺度上只改变最后几位：第 $10$ 步 loss 为 $1.312222730$，最终参数约为 $(0.512293421,0.512293418)$。极小梯度实验才会放大这个位置差异。

衰减实验的五步参数分别为：

| 规则 | 五步后的参数序列 |
| --- | --- |
| 耦合 L2 | 0.900000003、0.800102714、0.700381533、0.600913544、0.501779472 |
| 解耦 AdamW | 0.890000005、0.781100010、0.673289015、0.566556130、0.460890573 |

耦合版本的状态输入从 $0.3$ 变成 $0.260091354$；解耦版本的状态输入每步仍是 $0.2$。如果复现实验只填写“Adam + weight decay”，这两种结果都可能被误认为正确，真正缺少的是衰减语义。完整的更新顺序见后文。

## AdamW 不是换一个名字

若把 L2 惩罚加入梯度，Adam 接收到的是

$$
\widetilde{\boldsymbol g}_t
=\boldsymbol g_t+\lambda\boldsymbol\theta_t.
$$

这会让 $\lambda\boldsymbol\theta_t$ 进入 $\boldsymbol m_t$ 和 $\boldsymbol v_t$。由于二阶统计量会按坐标缩放，参数衰减不再是统一比例的收缩。

AdamW 式解耦衰减则把梯度方向和参数收缩分开：

$$
\boldsymbol\theta_{t+1}
=(1-\eta_t\lambda)\boldsymbol\theta_t
-\eta_t
\frac{\widehat{\boldsymbol m}_t}
{\sqrt{\widehat{\boldsymbol v}_t}+\varepsilon}.
$$

这两个公式只在非常特殊的情形下相同。单坐标例子中，原始梯度恒为 $0.2$、$\eta=0.1$、$\lambda=0.1$。耦合版本的进入状态的梯度从 $0.3$ 降到 $0.260091354$；解耦版本每次都让状态只看到 $0.2$，但额外把参数乘以 $0.99$。如果实验报告只写“Adam + weight decay”，无法知道复现者应当采用哪一种。

还要说明衰减是否作用于所有参数。实践中常把 bias、归一化层的 scale 或其他不希望缩小的参数放在独立参数组中。参数组不仅改变 $\eta$，还改变 $\lambda$、状态张量和记录方式。

## 训练日志应该记录哪些量

只记录训练 loss 很难解释 Adam 的行为。每隔固定的 optimizer step，可以记录

$$
\begin{aligned}
\boldsymbol d_t
&=\frac{\widehat{\boldsymbol m}_t}
{\sqrt{\widehat{\boldsymbol v}_t}+\varepsilon},\\
\boldsymbol u_t
&=-\eta_t\boldsymbol d_t,\\
\text{update ratio}_t
&=\frac{\lVert\boldsymbol\theta_{t+1}-\boldsymbol\theta_t\rVert_2}
{\max(\lVert\boldsymbol\theta_t\rVert_2,\varepsilon)}.
\end{aligned}
$$

再配合梯度范数、矩范数和分母统计：

| 观测量 | 可以回答的问题 |
| --- | --- |
| $\lVert\boldsymbol g_t\rVert_2$ | 数据和反向传播给了多大的信号 |
| $\lVert\boldsymbol m_t\rVert_2$ | 方向平均是否滞后或发生抵消 |
| $\lVert\boldsymbol v_t\rVert_2$ | 梯度尺度是否被尖峰抬高 |
| $\lVert\boldsymbol u_t\rVert_2$ | 参数实际移动了多远 |
| update ratio | 当前移动相对参数大小是否失控 |
| 非有限坐标数 | 是输入、反向、矩还是更新先坏 |
| 每组 step 与学习率 | 参数组和 scheduler 是否同一时间轴 |

例如 loss 平稳但 update ratio 突然变大，可能是某坐标的二阶状态被清零、恢复时遗漏了 $\boldsymbol v$，或 $\varepsilon$ 与低精度尺度不匹配。若梯度范数很大但更新比率没有变大，Adam 的分母可能正在抑制它；这不自动表示问题已解决，还要检查验证曲线和矩的时间滞后。

## 梯度累积与混合精度的边界

Adam 的一次状态推进应对应一次定义好的有效梯度。若一个有效 batch 被拆成 $R$ 个 micro-batch，先按样本数形成

$$
\overline{\boldsymbol g}_t
=\frac{\sum_{r=1}^{R}n_r\boldsymbol g_{t,r}}
{\sum_{r=1}^{R}n_r},
$$

再只执行一次

$$
\begin{aligned}
\boldsymbol m_t
&=\beta_1\boldsymbol m_{t-1}
+(1-\beta_1)\overline{\boldsymbol g}_t,\\
\boldsymbol v_t
&=\beta_2\boldsymbol v_{t-1}
+(1-\beta_2)\overline{\boldsymbol g}_t^{\odot2}.
\end{aligned}
$$

若每个 micro-batch 都更新矩，再在最后更新参数，$\beta_1^R$ 和 $\beta_2^R$ 已经改变了状态衰减，学习率调度器的 step 计数也可能提前推进。那是另一个训练协议。

混合精度下，loss scale $a$ 可能使反向得到

$$
\boldsymbol g_t^{\mathrm{scaled}}
=a\boldsymbol g_t.
$$

进入 Adam 前应先 unscale：

$$
\boldsymbol g_t
=\frac{\boldsymbol g_t^{\mathrm{scaled}}}{a}.
$$

一个常见顺序是：

1. 完成梯度累积和有效 batch 归约；
2. unscale；
3. 检查 NaN 或 Inf；
4. 按全局范数或逐参数规则裁剪；
5. 按约定应用耦合或解耦 weight decay；
6. 推进 Adam 的矩和 step；
7. 更新参数与学习率调度器。

工程库可能把第 5 步拆到参数更新内部，也可能在第 6 步前处理。重要的是记录实际语义，并用一维测试确认：裁剪究竟作用于原始梯度、加上 L2 后的梯度，还是只作用于梯度更新而不作用于解耦衰减。

## step、学习率调度和 checkpoint 必须相互对齐

偏置修正中的 $t$ 是 Adam 的更新次数。它不应随着 epoch 重新归零，也不应在跳过非有限梯度的 step 后盲目递增。若一次更新因为 overflow 被跳过，常见协议是保持矩、参数和 optimizer step 不变，同时让 scaler 调整；具体行为要以实现为准并写入日志。

学习率可以是每个 optimizer step 的 $\eta_t$，也可以按 epoch 更新：

$$
\begin{aligned}
\boldsymbol d_t
&=\frac{\widehat{\boldsymbol m}_t}
{\sqrt{\widehat{\boldsymbol v}_t}+\varepsilon},\\
\boldsymbol\theta_{t+1}
&=\boldsymbol\theta_t-\eta_t\boldsymbol d_t.
\end{aligned}
$$

如果 scheduler 实际按 batch step，而报告按 epoch 写 warmup，恢复或改变 batch size 后就可能产生不同的 $\eta_t$。warmup 期间还会与偏置修正的早期阶段重叠；这不是错误，但需要分别记录两者。

一个可恢复的 Adam checkpoint 至少包含：

| 状态 | 为什么不能省略 |
| --- | --- |
| 参数张量 | 当前模型位置 |
| 一阶矩 $\boldsymbol m$ | 当前方向的历史 |
| 二阶矩 $\boldsymbol v$ | 当前坐标尺度的历史 |
| optimizer step | 偏置修正和调度器的时间轴 |
| 参数组配置 | 每组的学习率、衰减和状态映射 |
| scheduler 状态 | warmup 或衰减的当前位置 |
| scaler 状态 | 混合精度的 loss scale 和跳步策略 |
| sampler 与随机状态 | 后续 batch 顺序和噪声来源 |

只恢复参数会让下一次梯度可能看起来完全正常，但 $\boldsymbol m$、$\boldsymbol v$ 和 $t$ 已经从零重新开始。对 Adam 来说，这不是从 checkpoint 继续，而是在同一个参数点重新启动另一个优化器。

## 可复制的标准库实现

```python
import math


EPS = 1e-8


def adam_direction(
    gradient,
    state,
    beta1=0.9,
    beta2=0.999,
    eps=EPS,
    bias_correction=True,
    epsilon_inside=False,
):
    state["step"] += 1
    step = state["step"]
    state["first"] = [
        beta1 * old + (1.0 - beta1) * current
        for old, current in zip(state["first"], gradient)
    ]
    state["second"] = [
        beta2 * old + (1.0 - beta2) * current * current
        for old, current in zip(state["second"], gradient)
    ]

    if bias_correction:
        first = [
            value / (1.0 - beta1**step)
            for value in state["first"]
        ]
        second = [
            value / (1.0 - beta2**step)
            for value in state["second"]
        ]
    else:
        first = state["first"]
        second = state["second"]

    if epsilon_inside:
        return [
            current / math.sqrt(value + eps)
            for current, value in zip(first, second)
        ]
    return [
        current / (math.sqrt(value) + eps)
        for current, value in zip(first, second)
    ]


def fresh_state(size):
    return {"step": 0, "first": [0.0] * size, "second": [0.0] * size}


def constant_gradient_demo():
    state = fresh_state(1)
    for gradient_step in range(1, 4):
        state["step"] += 1
        step = state["step"]
        gradient = [2.0]
        state["first"][0] = (
            0.9 * state["first"][0] + 0.1 * gradient[0]
        )
        state["second"][0] = (
            0.999 * state["second"][0] + 0.001 * gradient[0] ** 2
        )
        first = state["first"][0] / (1.0 - 0.9**step)
        second = state["second"][0] / (1.0 - 0.999**step)
        corrected = first / (math.sqrt(second) + EPS)
        raw = state["first"][0] / (
            math.sqrt(state["second"][0]) + EPS
        )
        print(
            "constant",
            gradient_step,
            round(state["first"][0], 6),
            round(state["second"][0], 6),
            round(corrected, 6),
            round(raw, 6),
        )


def quadratic_loss(theta):
    return 0.5 * (theta[0] ** 2 + 9.0 * theta[1] ** 2)


def quadratic_gradient(theta):
    return [theta[0], 9.0 * theta[1]]


def run_quadratic(kind, steps=10, eta=0.05):
    theta = [1.0, 1.0]
    state = fresh_state(2)
    history = []
    for _ in range(steps):
        gradient = quadratic_gradient(theta)
        direction = adam_direction(
            gradient,
            state,
            bias_correction=kind != "no-bias",
            epsilon_inside=kind == "inside-eps",
        )
        theta = [
            current - eta * update
            for current, update in zip(theta, direction)
        ]
        history.append((theta[:], quadratic_loss(theta)))
    checkpoints = [1, 2, 5, 10]
    return (
        [round(history[index - 1][1], 9) for index in checkpoints],
        tuple(round(value, 9) for value in history[-1][0]),
    )


def run_decay(kind, steps=5):
    theta = 1.0
    first = 0.0
    second = 0.0
    beta1, beta2 = 0.9, 0.999
    eta, decay, gradient = 0.1, 0.1, 0.2
    values = []
    for step in range(1, steps + 1):
        used = gradient + decay * theta if kind == "coupled" else gradient
        first = beta1 * first + (1.0 - beta1) * used
        second = beta2 * second + (1.0 - beta2) * used * used
        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)
        direction = first_hat / (math.sqrt(second_hat) + EPS)
        if kind == "coupled":
            theta -= eta * direction
        else:
            theta = (1.0 - eta * decay) * theta - eta * direction
        values.append(round(theta, 9))
    return values


if __name__ == "__main__":
    constant_gradient_demo()
    for kind in ("adam", "no-bias", "inside-eps"):
        print(kind, run_quadratic(kind))
    print("coupled", run_decay("coupled"))
    print("decoupled", run_decay("decoupled"))
```

输出为：

```text
constant 1 0.2 0.004 1.0 3.162277
constant 2 0.38 0.007996 1.0 4.249591
constant 3 0.542 0.011988 1.0 4.950235
adam ([4.512500001, 4.050749222, 2.8235951, 1.312222734], (0.512293423, 0.512293418))
no-bias ([3.543861254, 1.991642051, 0.046738905, 1.388269098], (-0.526928708, -0.526928662))
inside-eps ([4.5125, 4.05074922, 2.823595097, 1.31222273], (0.512293421, 0.512293418))
coupled [0.900000003, 0.800102714, 0.700381533, 0.600913544, 0.501779472]
decoupled [0.890000005, 0.78110001, 0.673289015, 0.56655613, 0.460890573]

```

运行输出中的常值梯度三行显示修正后方向稳定在约 $1$，未修正方向从 $3.162277$ 增长到 $4.950235$。这验证了偏置修正的代数推导。

在二维目标上，标准 Adam 的第 $10$ 步 loss 为 $1.312222734$，最终参数约为 $(0.512293423,0.512293418)$。去掉偏置修正后，第 $5$ 步 loss 虽降到 $0.046738905$，第 $10$ 步却回到 $1.388269098$，最终参数约为 $(-0.526928708,-0.526928662)$。不同轨迹来自更新尺度不同，不应把这一次短轨迹解读成某个变体普遍更好。

根号内的 $\varepsilon$ 在正常尺度上只改变最后几位：第 $10$ 步 loss 为 $1.312222730$，最终参数约为 $(0.512293421,0.512293418)$。极小梯度实验才会放大这个位置差异。

衰减实验的五步参数分别为：

| 规则 | 五步后的参数序列 |
| --- | --- |
| 耦合 L2 | 0.900000003、0.800102714、0.700381533、0.600913544、0.501779472 |
| 解耦 AdamW | 0.890000005、0.781100010、0.673289015、0.566556130、0.460890573 |

耦合版本的状态输入从 $0.3$ 变成 $0.260091354$；解耦版本的状态输入每步仍是 $0.2$。如果复现实验只填写“Adam + weight decay”，这两种结果都可能被误认为正确，真正缺少的是衰减语义。

## 常见失效模式

**忘记偏置修正。** 代码能运行，前几步也可能下降，但第一步尺度已经改变。用常值梯度的精确例子检查 $\widehat m_1$ 和 $\widehat v_1$，比盯着真实网络的 loss 更容易定位。

**把二阶矩当作 Hessian。** $\boldsymbol v_t$ 只保存平方梯度的逐坐标平均；它不包含非对角曲率，也不能直接给出 Newton 方向。

**把 $\varepsilon$ 当作无关紧要的格式。** 大梯度时差别可能在最后几位，极小状态或低精度时则可能改变多个数量级。复现时要注明根号内外的位置。

**让每个 micro-batch 推进一次矩。** 这会改变 $\beta$ 的有效时间常数、optimizer step 和 scheduler 轨迹。先确定有效 batch，再更新一次状态。

**把 Adam 和 AdamW 混写。** 耦合衰减进入矩，解耦衰减直接收缩参数；两者的状态轨迹不同。

**恢复时漏掉 step 或参数组状态。** 偏置修正、学习率和每个参数张量的矩会从错误的时间点继续。

**把固定的 $\beta_2$ 当作固定的样本记忆。** 它控制更新步上的指数平均。batch、累积和分布式 worker 改变后，要重新解释它覆盖了多少样本。

**只凭二维二次目标选算法。** 标准库目标适合验证公式和状态顺序，不能替代真实网络的验证集、数据预算和泛化比较。

## 运行方法

将上面的代码保存为 adam.py，在项目根目录运行：

```bash
python3 adam.py
```

先核对常值梯度的三行输出，再核对二维目标和衰减的列表。若替换成框架实现，保持初始参数、梯度、$\beta$、$\varepsilon$ 和 step 顺序相同，逐步比较 $\boldsymbol m$、$\boldsymbol v$、偏置修正后的方向与参数；不要只比较最后一个 loss。

## 相关词条

- [优化器](../training-nn/optimizers/)：统一比较 SGD、动量、自适应方法与二阶方法的外部接口。
- [自适应学习率](../optimization-theory/adaptive-learning-rates/)：推导 AdaGrad、RMSProp 和 Adam 的尺度变换与理论边界。
- [动量与 Nesterov](../training-nn/momentum-and-nesterov/)：展开时间方向上的历史缓冲区和前瞻更新。
- [小批量随机梯度下降](../training-nn/minibatch-sgd/)：说明 Adam 接收的有效梯度如何由 batch 与累积形成。
- [梯度裁剪](../training-nn/gradient-clipping/)：处理进入优化器状态前的异常梯度范数。
- [学习率调度](../training-nn/learning-rate-schedules/)：安排 warmup、衰减与 optimizer step 的时间轴。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：区分 L2、权重衰减和优化路径的正则化作用。
- [混合精度](../training-nn/mixed-precision/)：说明 unscale、溢出检查与状态精度的关系。
