---
title: "学习率调度：把训练步长放到时间轴上"
tags: ["why-models-learn"]
---

学习率调度把每次参数更新使用的标量 $\eta_t$ 写成 optimizer step、epoch、样本访问量或验证反馈的函数。它不改变梯度方向，却会改变每一步移动多远、warmup 持续多久、衰减何时结束以及恢复训练后时间轴是否连续。本篇先区分样本、batch、optimizer step 和 epoch，再推导线性 warmup、余弦衰减、指数衰减、反平方根衰减与 plateau 调度，最后用标准库实验检查 off-by-one、batch size 变化和 checkpoint 恢复。

![学习率调度图：前段线性 warmup 到峰值，后段沿余弦曲线衰减；训练记录同时标出 optimizer step 与 epoch](/assets/training-nn/svg/learning-rate-schedules.1.svg)

## 学习率是哪个时间单位的函数

固定学习率的梯度更新可以写成

$$
\boldsymbol\theta_{t+1}
=\boldsymbol\theta_t-\eta\boldsymbol d_t,
$$

其中 $\boldsymbol d_t$ 是优化器给出的方向。带调度的更新则是

$$
\boldsymbol\theta_{t+1}
=\boldsymbol\theta_t-\eta_t\boldsymbol d_t,
\qquad
\eta_t=S(t;\boldsymbol h).
$$

这里的 $S$ 是调度函数，$\boldsymbol h$ 是峰值学习率、终点学习率、warmup 长度、总步数、衰减因子等超参数。它与 Adam 的一阶矩、二阶矩分工不同：Adam 改变 $\boldsymbol d_t$ 的逐坐标尺度，scheduler 改变整个方向前的标量 $\eta_t$。

写出 $t$ 之前，先固定四个计数：

| 计数 | 一次增加的时机 | 容易混淆的地方 |
| --- | --- | --- |
| 样本访问量 | 一个样本被 loader 取出 | 分布式训练中要说明是单卡还是全局 |
| batch | 一批样本完成一次前向与反向 | 不一定立刻更新参数 |
| optimizer step | 参数和优化器状态实际推进一次 | 梯度累积时多个 batch 才对应一步 |
| epoch | 训练集被完整访问一轮 | batch size 改变后每 epoch 的 step 会改变 |

例如有 $N=1000$ 个样本，不丢弃尾批次时：

- batch size 为 $32$，每 epoch 有 $\lceil1000/32\rceil=32$ 个 batch；
- batch size 为 $128$，每 epoch 有 $\lceil1000/128\rceil=8$ 个 batch。

同一个 400-step warmup，在两种设置下分别覆盖 $12.5$ 个 epoch 和 $50$ 个 epoch。若把 warmup 写成“前 10 个 epoch”，换 batch size 后它实际上已不是同一段学习率轨迹；若写成“前 400 个 optimizer step”，它的更新时间轴才保持一致。

梯度累积还会再除一次。若每个 micro-batch 都反向但累积 $A$ 次才更新参数，则

$$
\text{optimizer steps}
=\frac{\text{micro-batches}}{A},
$$

学习率调度通常也应在这一次参数更新时推进，而不是每次反向都推进。否则 warmup 和衰减会比参数实际移动快 $A$ 倍。

## 先决定边界：第几步使用哪个学习率

调度最容易出现的错误不是公式，而是边界约定。本文采用：

- optimizer step 从 $1$ 开始；
- 在第 $t$ 步计算 $\eta_t$；
- 用这一个 $\eta_t$ 完成 $\boldsymbol\theta_t\to\boldsymbol\theta_{t+1}$；
- step $T$ 的调度值仍然属于训练预算。

如果 warmup 有 $K$ 步，线性 warmup 写成

$$
\eta_t
=\eta_{\max}\frac{t}{K},
\qquad
1\leq t\leq K.
$$

这样第 $1$ 步不是零，第 $K$ 步恰好到达 $\eta_{\max}$。另一种代码把 step 记为 $0,1,\ldots$，写成 $\eta_s=\eta_{\max}(s+1)/K$；两种写法都可以，但不能把一个公式和另一个计数混用。

常见的三个边界问题是：

| 问题 | 表面现象 | 直接检查 |
| --- | --- | --- |
| warmup 第一步取零 | 初始参数几乎不动 | 打印 step 1 的 $\eta_1$ |
| 总步数多算或少算一 | 曲线终点提前或延后 | 打印 $t=T$ 和 $t=T+1$ |
| scheduler 在 step 前还是后推进 | 同一配置的首个 loss 不同 | 固定初始梯度做一步单元测试 |

一个可审计的训练循环应明确写出：

$$
\begin{aligned}
\boldsymbol g_t
&=\operatorname{backward}(\mathcal L_t),\\
\eta_t
&=S(t;\boldsymbol h),\\
(\boldsymbol d_t,\boldsymbol s_{t+1})
&=\mathcal O(\boldsymbol g_t,\boldsymbol s_t,\eta_t),\\
\boldsymbol\theta_{t+1}
&=\boldsymbol\theta_t-\eta_t\boldsymbol d_t.
\end{aligned}
$$

实际框架可能把 scheduler.step 放在 optimizer.step 前或后。只看调用顺序不够，还要输出第一个 optimizer step 使用的数值，确认代码语义与报告一致。

## 线性 warmup：先控制最大步长

线性 warmup 在 $K$ 个更新内把学习率从 $\eta_{\mathrm{start}}$ 提到 $\eta_{\max}$：

$$
\eta_t
=\eta_{\mathrm{start}}
+(\eta_{\max}-\eta_{\mathrm{start}})
\frac{t}{K},
\qquad
1\leq t\leq K.
$$

若 $\eta_{\mathrm{start}}=0$，第 $t$ 步的学习率就是 $\eta_{\max}t/K$。若从一个非零值开始，初始斜率由 $\eta_{\max}-\eta_{\mathrm{start}}$ 决定。

warmup 控制的是实际更新长度，不是梯度存在的时间。训练刚开始时，参数、Adam 的矩、归一化统计量和数据分布都可能处于过渡期；减小前几步的 $\eta_t$ 可以把这些变化分开观察。但 warmup 不是一个普遍安全开关：

- 若 $\eta_{\max}$ 本身过大，warmup 结束仍会跳到不稳定区域；
- 若 $\eta_{\max}$ 很小，长 warmup 只会浪费更新预算；
- 若从 checkpoint 恢复，warmup 是否已经完成必须由 scheduler state 决定；
- 若 batch size 改变，按 epoch 定义的 warmup 需要重新换算。

warmup 与 Adam 的偏置修正可以同时存在。前者直接改变 $\eta_t$，后者改变 $\boldsymbol d_t$；不要把 warmup 解释成“等待 Adam 的矩稳定”，也不要在恢复时只恢复其中一个。

## 余弦衰减：平滑地走到终点

完成 warmup 后，令剩余衰减区间的进度为

$$
p_t
=\frac{t-K}{T-K},
\qquad
K<t\leq T.
$$

余弦衰减从 $\eta_{\max}$ 平滑地走向 $\eta_{\min}$：

$$
\eta_t
=\eta_{\min}
+\frac12(\eta_{\max}-\eta_{\min})
\left(1+\cos(\pi p_t)\right).
$$

在 $p_t=0$ 时，$\eta_t=\eta_{\max}$；在 $p_t=1$ 时，$\eta_t=\eta_{\min}$。其前后端斜率都为零，所以比突然切换的阶梯衰减更平滑。

有限预算下要把进度截断：

$$
p_t
=\min\left(1,\max\left(0,\frac{t-K}{T-K}\right)\right).
$$

这样超过总步数后不会再次升高。若训练因为数据流结束提前停止，曲线可能还没有到 $\eta_{\min}$；若训练超过预算，应该明确选择继续保持终点、重启曲线，还是重新定义 $T$。

余弦曲线的平滑不等于验证指标一定平滑。随机 batch、数据增强、评估间隔和 Adam 的状态都会让 loss 产生波动；调度器只控制一个已知输入，不能替代验证集监控。

## 其他开环调度

不依赖验证反馈的调度可以在训练前完整写出，因此称为开环调度。常见形式包括：

指数衰减：

$$
\eta_t
=\eta_0\gamma^{t-1},
\qquad
0<\gamma<1.
$$

阶梯衰减：

$$
\eta_t
=\eta_0\delta^{\left\lfloor(t-1)/q\right\rfloor},
\qquad
0<\delta<1,
$$

其中每 $q$ 个 step 乘一次 $\delta$。

反平方根衰减常与 warmup 一起使用：

$$
\eta_t
=\eta_{\max}
\sqrt{\frac{K}{\max(t,K)}}.
$$

当 $t\leq K$ 时它保持在 $\eta_{\max}$，所以若需要从零升高，通常先接一段线性 warmup：

$$
\eta_t
=
\begin{cases}
\eta_{\max}t/K,&t\leq K,\\
\eta_{\max}\sqrt{K/t},&t>K.
\end{cases}
$$

多项式衰减则把终点速度作为参数：

$$
\eta_t
=\eta_{\min}
+(\eta_{\max}-\eta_{\min})
\left(1-\frac{t}{T}\right)^r,
\qquad
0\leq t\leq T.
$$

这些函数不是同一条曲线的不同画法。它们对早期步长、总更新量和尾部长度的分配不同。比较调度器时，除了画 $\eta_t$，还应报告离散总和

$$
A_T=\sum_{t=1}^{T}\eta_t.
$$

$A_T$ 不是训练效果的充分统计量，但可以揭示一个配置是否只是因为累计步长大得多才看起来下降更快。

## plateau 调度：让验证信号参与

plateau 调度不是预先写死 $S(t)$，而是根据最近的验证指标决定是否降低学习率。以要最小化的验证损失 $q_e$ 为例，维护当前最好值 $q_\star$、等待计数 $w$ 和 cooldown 计数：

1. 若 $q_e<q_\star-\text{threshold}$，更新最好值并把等待计数归零；
2. 否则等待计数加一；
3. 当等待计数达到 patience 且不在 cooldown 中，把学习率乘以 factor；
4. 把等待计数归零，进入 cooldown，并把学习率限制在 $\eta_{\min}$ 之上。

它的状态可以写成

$$
(\eta_{e+1},q_\star,w_{e+1},c_{e+1})
=R(\eta_e,q_e,q_\star,w_e,c_e;\boldsymbol h).
$$

这与余弦调度不同：同一 epoch 数在两次运行中可能得到不同学习率，因为验证噪声、随机种子和 threshold 比较会改变状态转移。

plateau 调度需要固定几个语义：

| 配置 | 必须说明 |
| --- | --- |
| monitor | 是验证 loss、accuracy 还是别的指标 |
| mode | 指标是越小越好还是越大越好 |
| threshold | 相对还是绝对改善，边界是否严格 |
| patience | 计数 epoch 还是评估事件 |
| factor | 降低后的比例 |
| cooldown | 降低后等待多少个评估事件 |
| min learning rate | 每个参数组相同还是分别设置 |

若每个 epoch 只评估一次，patience=3 表示连续三个评估事件没有足够改善，不一定等于三个 optimizer step。若验证集很小或指标方差很大，plateau 可能在噪声上反复触发；应先固定评估频率和随机性，再调 patience 与 threshold。

验证指标还不能来自测试集。用测试集决定何时降学习率会把测试信息写进训练过程，随后再报告测试结果就不再是封存测试。

## 一组十步曲线

取 $T=10$、warmup $K=3$、$\eta_{\max}=0.1$、$\eta_{\min}=0.01$。线性 warmup 加余弦衰减的十个学习率为

$$
(0.033333,\ 0.066667,\ 0.1,\ 0.095544,\ 0.083057,\ 0.065013,\ 0.044987,\ 0.026943,\ 0.014456,\ 0.01).
$$

同一组 step 上，几种调度的数值如下：

| 调度 | 第 1–10 步学习率 |
| --- | --- |
| constant | 0.1、0.1、0.1、0.1、0.1、0.1、0.1、0.1、0.1、0.1 |
| warmup + cosine | 0.033333、0.066667、0.1、0.095544、0.083057、0.065013、0.044987、0.026943、0.014456、0.01 |
| exponential $\gamma=0.8$ | 0.1、0.08、0.064、0.0512、0.04096、0.032768、0.026214、0.020972、0.016777、0.013422 |
| inverse square root | 0.1、0.1、0.1、0.086603、0.07746、0.070711、0.065465、0.061237、0.057735、0.054772 |
| step，每 4 步乘 0.1 | 0.1、0.1、0.1、0.1、0.01、0.01、0.01、0.01、0.001、0.001 |

它们在十步内的离散总和也不同：warmup + cosine 的总和是 $0.54$，constant 的总和是 $1.0$。因此只比较十步后的 loss 而不对齐累计步长，结论会把“走得更多”混成“调度更好”。

## 调度与优化器的组合

对 SGD，$\eta_t$ 直接控制梯度位移；对 Adam，真实位移是

$$
\Delta\boldsymbol\theta_t
=-\eta_t
\frac{\widehat{\boldsymbol m}_t}
{\sqrt{\widehat{\boldsymbol v}_t}+\varepsilon}.
$$

因此学习率曲线平滑，不代表参数更新范数平滑。二阶矩突然增大时，Adam 的有效方向会缩短；二阶矩衰减时，同一个 $\eta_t$ 可能产生更大的移动。

一个可用的日志拆分是：

| 量 | 所属层次 |
| --- | --- |
| $\eta_t$ | scheduler 的标量输入 |
| $\lVert\boldsymbol g_t\rVert_2$ | 数据与反向传播 |
| $\lVert\boldsymbol d_t\rVert_2$ | optimizer 预条件后的方向 |
| $\lVert\Delta\boldsymbol\theta_t\rVert_2$ | 参数实际位移 |
| update ratio | 位移相对参数尺度 |

如果 $\eta_t$ 下降而 update ratio 上升，问题可能在 Adam 的分母或状态恢复；如果 $\eta_t$ 没变而 update ratio 突降，可能是梯度尺度或二阶矩改变。把这四层量分开记录，才知道调度器是否真的执行了预期行为。

## batch size 改变时怎样保持比较公平

假定总样本访问预算为 $M$，全局有效 batch 为 $B_{\mathrm{eff}}$，则 optimizer step 近似为

$$
U=\left\lceil\frac{M}{B_{\mathrm{eff}}}\right\rceil.
$$

如果 scheduler 以总 step $T$ 结束，改变 $B_{\mathrm{eff}}$ 后要重新决定：

- 固定 step 数，允许访问更多或更少样本；
- 固定样本访问量，让总 step 随 batch 改变；
- 固定 epoch 数，接受每个 epoch 的更新次数改变。

三种选择都可能合理，但不能在报告里只写“训练 20 个 epoch”就假定预算相同。warmup、余弦终点、plateau patience 和 $\beta$ 的记忆长度都依赖所选时间单位。

梯度累积时，若单卡 micro-batch 为 $b$、设备数为 $D$、累积步数为 $A$，有效 batch 常写成

$$
B_{\mathrm{eff}}=bDA.
$$

调度器若按 optimizer step 推进，应把 $A$ 纳入总步数换算；若按样本访问量推进，则需记录每个 step 实际覆盖的样本数。

## checkpoint 与恢复

开环调度需要保存至少一个当前 step；plateau 调度还需要保存反馈状态：

| 状态 | 开环调度 | plateau 调度 |
| --- | --- | --- |
| 当前 optimizer step | 需要 | 需要 |
| 当前 epoch 或评估事件数 | 若按 epoch 调度需要 | 需要 |
| 当前 $\eta_t$ | 可重算但建议记录 | 建议记录 |
| 最好验证指标 | 不需要 | 需要 |
| patience 等待计数 | 不需要 | 需要 |
| cooldown 计数 | 不需要 | 需要 |
| 参数组学习率 | 需要 | 需要 |

恢复时若只加载模型和 Adam 状态，却让 scheduler 从第 $1$ 步重新开始，参数会在同一点收到不同的学习率。若只恢复 $\eta_t$ 而不恢复 plateau 的最好值与等待计数，下一次评估又会触发不同的降速时机。

一个简单的恢复测试是：先运行完整的 $T$ 步，再运行前 $r$ 步保存 checkpoint、恢复后运行 $T-r$ 步，逐步比较两条 $\eta_t$ 序列和参数。确定性调度下两者应在浮点容差内一致；若不一致，先查 step 的保存时机和 scheduler.step 的前后顺序。

## 标准库实验：曲线、二次目标与恢复

下面的代码不依赖训练框架，打印五种开环调度、一个一维二次目标的轨迹，以及 batch size 和恢复时间轴的数值。二次目标只用于检查调度实现，不用它替代真实网络上的验证选择。

```python
import math


def constant(step, eta=0.1):
    return eta


def warmup_cosine(
    step,
    total_steps=10,
    warmup_steps=3,
    eta_max=0.1,
    eta_min=0.01,
):
    if warmup_steps > 0 and step <= warmup_steps:
        return eta_max * step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(
        max((step - warmup_steps) / decay_steps, 0.0),
        1.0,
    )
    return eta_min + 0.5 * (eta_max - eta_min) * (
        1.0 + math.cos(math.pi * progress)
    )


def exponential(step, eta0=0.1, gamma=0.8):
    return eta0 * gamma ** (step - 1)


def inverse_sqrt(step, warmup_steps=3, eta_max=0.1):
    return eta_max * math.sqrt(
        warmup_steps / max(step, warmup_steps)
    )


def step_decay(step, eta0=0.1, drop=0.1, every=4):
    return eta0 * drop ** ((step - 1) // every)


SCHEDULES = {
    "constant": constant,
    "warmup-cosine": warmup_cosine,
    "exp": exponential,
    "inverse-sqrt": inverse_sqrt,
    "step": step_decay,
}


for name, schedule in SCHEDULES.items():
    values = [schedule(step) for step in range(1, 11)]
    print(name, [round(value, 6) for value in values])


def quadratic_run(schedule):
    x = 1.0
    losses = []
    for step in range(1, 11):
        eta = schedule(step)
        gradient = 8.0 * x
        x -= eta * gradient
        losses.append(0.5 * 8.0 * x * x)
    return (
        f"x10={x:.9f}",
        f"loss10={losses[-1]:.3e}",
        [f"{losses[index - 1]:.3e}" for index in (1, 3, 5, 10)],
    )


for name, schedule in SCHEDULES.items():
    print("quadratic", name, *quadratic_run(schedule))


samples = 1000
for batch_size in (32, 128):
    updates = math.ceil(samples / batch_size)
    print(
        "batch",
        batch_size,
        "updates_per_epoch",
        updates,
        "warmup_400_epoch",
        round(400 / updates, 4),
        "updates_in_20_epochs",
        updates * 20,
    )


full = [warmup_cosine(step) for step in range(1, 11)]
print(
    "resume",
    "step7_full",
    round(full[6], 6),
    "restart_step1",
    round(full[0], 6),
    "difference",
    round(full[0] - full[6], 6),
    "sum",
    round(sum(full), 9),
)
```

输出为：

```text
constant [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
warmup-cosine [0.033333, 0.066667, 0.1, 0.095544, 0.083057, 0.065013, 0.044987, 0.026943, 0.014456, 0.01]
exp [0.1, 0.08, 0.064, 0.0512, 0.04096, 0.032768, 0.026214, 0.020972, 0.016777, 0.013422]
inverse-sqrt [0.1, 0.1, 0.1, 0.086603, 0.07746, 0.070711, 0.065465, 0.061237, 0.057735, 0.054772]
step [0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001]
quadratic constant x10=0.000000102 loss10=4.194e-14 ['1.600e-01', '2.560e-04', '4.096e-07', '4.194e-14']
quadratic warmup-cosine x10=0.001061045 loss10=4.503e-06 ['2.151e+00', '1.874e-02', '1.172e-04', '4.503e-06']
quadratic exp x10=0.005230586 loss10=1.094e-04 ['1.600e-01', '4.938e-03', '7.781e-04', '1.094e-04']
quadratic inverse-sqrt x10=0.000029815 loss10=3.556e-09 ['1.600e-01', '2.560e-04', '3.494e-06', '3.556e-09']
quadratic step x10=0.001127962 loss10=5.089e-06 ['1.600e-01', '2.560e-04', '8.667e-06', '5.089e-06']
batch 32 updates_per_epoch 32 warmup_400_epoch 12.5 updates_in_20_epochs 640
batch 128 updates_per_epoch 8 warmup_400_epoch 50.0 updates_in_20_epochs 160
resume step7_full 0.044987 restart_step1 0.033333 difference -0.011653 sum 0.54
```

在一维二次目标 $f(x)=4x^2$ 上，梯度为 $8x$。固定 $\eta=0.1$ 时每一步都乘以 $0.2$，十步后 $x$ 约为 $1.02\times10^{-7}$；warmup + cosine 前三步较小，十步后 $x$ 为 $0.001061045$。这不是说 constant 一定优于 cosine：它们的累计学习率分别为 $1.0$ 和 $0.54$，且没有共享同一训练预算。实验先验证代码确实执行了预期曲线，再提醒比较时要对齐协议。

恢复示例中完整曲线的第 $7$ 步学习率是 $0.044987$；如果恢复时把 scheduler 当新对象从 step 1 重新开始，会得到 $0.033333$，差值为 $-0.011653$。模型参数相同不代表调度状态相同。

## 组合调度与重启

调度器可以组合，但每个组合都要定义边界。例如 warmup + cosine 是先选择 warmup 分支，再把剩余时间映射到 $[0,1]$。如果再加 step decay，必须说明乘法顺序：

$$
\eta_t
=S_{\mathrm{cosine}}(t)\,D_{\mathrm{step}}(t)
$$

与

$$
\eta_t
=S_{\mathrm{cosine}}\left(t;
\eta_{\max}D_{\mathrm{step}}(t)\right)
$$

通常不是同一条曲线。

cosine restart 会在某个周期末把进度重新映射到新的周期。它适合需要周期性探索的实验，但每次重启都改变了累计步长和验证曲线的解释。checkpoint 需要保存当前周期、周期内 step 和下一次重启位置，不能只保存一个全局 epoch。

one-cycle 类调度还会同时安排学习率上升/下降和动量变化。它不是“余弦学习率再加一个名字”，因为动量时间尺度也随时间改变；若使用它，应把 $\eta_t$ 与动量系数的两条曲线一起报告。

## 常见失效模式

**把 epoch 当作稳定的时间单位。** batch size、设备数或梯度累积改变后，每 epoch 的 optimizer step 会改变。优先报告总 step 或样本访问量。

**把 scheduler.step 放错位置。** 前置和后置调用会让首步或终点错一格。用固定梯度的单步测试打印实际使用的 $\eta_1$。

**warmup 和总步数使用不同口径。** warmup 按 optimizer step、总预算按 epoch 时，余弦进度的分母不再对应同一时间轴。

**恢复时只保存当前学习率。** 开环调度可能还能勉强继续，plateau、重启和组合调度会丢失等待计数或周期位置。

**只比较最终学习率。** 两条曲线可能在终点相同，但累计学习率、早期步长和峰值持续时间完全不同。

**把 plateau 的噪声当成模型趋势。** patience 太小、验证集太小或评估频率不固定都会让学习率过早下降。先固定 monitor、threshold 和评估事件。

**把测试集交给 plateau。** 调度过程已经使用测试信息，最后的测试指标不再是独立证据。

**超过总步数后让余弦继续外推。** 未截断的余弦可能重新上升，导致训练后段出现意外的大步长。对 $p_t$ 显式截断或明确重启。

## 运行方法

将上面的代码保存为 learning-rate-schedules.py，在项目根目录运行：

```bash
python3 learning-rate-schedules.py
```

先核对十步学习率和一维二次目标，再把调度函数接入真实训练。真实实验至少记录 optimizer step、样本访问量、$\eta_t$、验证评估事件和 checkpoint 中的 scheduler 状态；只贴最后一个学习率无法重建训练轨迹。

## 相关词条

- [梯度下降](../training-nn/gradient-descent/)：说明学习率在单步参数更新中的基本含义。
- [优化器](../training-nn/optimizers/)：区分 scheduler 的标量学习率与 Adam 的状态方向。
- [Adam](../training-nn/adam/)：查看偏置修正、二阶尺度和更新比率如何与 $\eta_t$ 组合。
- [小批量随机梯度下降](../training-nn/minibatch-sgd/)：定义有效 batch、梯度累积和 optimizer step。
- [动量与 Nesterov](../training-nn/momentum-and-nesterov/)：理解动量系数与学习率同时变化时的状态时间轴。
- [早停](../evaluation-and-generalization/early-stopping/)：把验证轨迹、checkpoint 和训练预算放在一起判断。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：区分衰减、训练时间和学习率路径的正则化影响。
