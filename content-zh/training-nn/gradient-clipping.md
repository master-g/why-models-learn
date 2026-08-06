---
title: "梯度裁剪：限制异常更新而不是掩盖根因"
tags: ["why-models-learn"]
---

梯度裁剪是在反向传播得到梯度后、优化器更新参数前，对梯度的大小施加上限。最常见的全局范数裁剪只把超过阈值的梯度整体缩短，因此保留方向；逐元素裁剪则会改变方向。本篇从这两种规则的公式出发，说明阈值、梯度累积、混合精度、分布式归约和优化器状态的顺序，再用标准库实验展示裁剪前后参数轨迹和可审计指标。

![梯度裁剪流程图：反向传播后先反缩放并归约梯度，再按全局范数裁剪，最后交给优化器；右侧比较原始向量、范数裁剪和逐元素裁剪](/assets/training-nn/svg/gradient-clipping.1.svg)

## 裁剪插在训练循环的哪里

一次普通更新可以拆成：

$$
\begin{aligned}
\boldsymbol g_t
&=\operatorname{backward}(\mathcal L_t),\\
\boldsymbol g_t'
&=\operatorname{clip}(\boldsymbol g_t;\tau),\\
(\boldsymbol d_t,\boldsymbol s_{t+1})
&=\mathcal O(\boldsymbol g_t',\boldsymbol s_t),\\
\boldsymbol\theta_{t+1}
&=\boldsymbol\theta_t-\eta_t\boldsymbol d_t.
\end{aligned}
$$

这里 $\boldsymbol g_t$ 是反向传播的原始梯度，$\boldsymbol g_t'$ 是裁剪后的梯度，$\boldsymbol s_t$ 是优化器状态。裁剪通常发生在 $\boldsymbol g_t$ 已经聚合完、但还没有送进动量或 Adam 矩统计的时刻。

这个位置有两个含义：

- 它限制的是本次进入优化器的梯度，不是前向激活、损失值或参数本身；
- 它只改变超出阈值的更新，没有超出阈值时应当成为恒等操作。

因此，日志中应同时保留原始范数和裁剪后范数。只记录后一项，会把“本来没有异常”和“异常已被压住”混成同一种曲线。

## 全局范数裁剪

把所有参数梯度展平后拼成一个向量 $\boldsymbol g$，定义全局二范数

$$
G=\lVert\boldsymbol g\rVert_2
=\sqrt{\sum_i g_i^2}.
$$

给定阈值 $\tau>0$，全局范数裁剪的缩放系数为

$$
c=\min\left(1,\frac{\tau}{G+\varepsilon}\right),
\qquad
\boldsymbol g'=c\boldsymbol g.
$$

当 $G\leq\tau$ 时，$c=1$；当 $G>\tau$ 时，$\lVert\boldsymbol g'\rVert_2$ 接近 $\tau$。若 $G=0$，梯度保持零向量。实现中的 $\varepsilon$ 只用于避免除零，不应被误解成一个额外的学习率。

### 为什么方向保持

在被裁剪的情况下，

$$
\boldsymbol g'=c\boldsymbol g,
\qquad
c>0.
$$

所以对于非零梯度，

$$
\frac{\boldsymbol g'}{\lVert\boldsymbol g'\rVert_2}
=\frac{\boldsymbol g}{\lVert\boldsymbol g\rVert_2}.
$$

它沿同一条射线移动，只是长度从 $G$ 变成 $\tau$。如果优化器是裸 SGD,这相当于把本步的有效学习率从 $\eta_t$ 降成 $\eta_tc$。

方向保持不等于优化结果不变。动量、Adam 的一阶矩和二阶矩都会把裁剪后的数值写入状态，所以一次裁剪可能影响后续多个 step。恢复 checkpoint 时，只恢复参数而漏掉这些状态，会让裁剪后的轨迹重新开始。

### 全局与逐张量的区别

若梯度分为 $\boldsymbol g^{(1)},\ldots,\boldsymbol g^{(m)}$，全局范数是

$$
G_{\mathrm{global}}
=\sqrt{\sum_{k=1}^{m}
\lVert\boldsymbol g^{(k)}\rVert_2^2}.
$$

全局裁剪使用同一个 $c$ 乘所有张量。逐张量裁剪则为每个张量单独计算

$$
c_k=\min\left(1,
\frac{\tau_k}{\lVert\boldsymbol g^{(k)}\rVert_2+\varepsilon}\right).
$$

逐张量规则可能让小张量保持原样、大张量被缩短，但它不再限制整个更新的总范数。两者都可以用，前提是日志明确写出阈值和归约范围。

## 逐元素裁剪不是同一件事

逐元素裁剪把每个坐标限制在 $[-v,v]$：

$$
g_i'=\operatorname{sign}(g_i)
\min(\lvert g_i\rvert,v).
$$

它对每个坐标单独处理，所以方向一般会改变。例如

$$
\boldsymbol g=(3,4),\qquad v=2
\quad\Longrightarrow\quad
\boldsymbol g'=(2,2).
$$

原向量的方向斜率是 $4/3$，逐元素裁剪后的斜率变为 $1$。如果某个坐标是噪声尖峰，逐元素裁剪可以限制它；如果梯度本来需要沿一条有意义的斜方向移动，这种改变也可能损失信息。

全局范数裁剪同一例子取 $\tau=2$：

$$
\lVert(3,4)\rVert_2=5,
\qquad
c=\frac25,
\qquad
\boldsymbol g'=(1.2,1.6).
$$

两个结果的范数都不超过约束，但只有全局范数裁剪保留了原方向。名称里都叫 clip,不能据此把它们当作可互换的实现。

## 裁剪能解决什么，不能解决什么

梯度爆炸可能来自深层 Jacobian 连乘、异常 batch、损失尺度、混合精度溢出或学习率过大。裁剪可以把当前这一步的梯度范数限制住，让训练不至于被一次异常更新直接推离可用区域。

它不能单独修复：

- 初始化导致的激活或梯度逐层消失；
- 过大的学习率让每一步都超出稳定范围；
- 所有 ReLU 长期处于负侧；
- 错误的 loss reduction 让梯度按 batch size 被重复放大；
- 数据或标签中持续存在的异常值。

如果几乎每个 step 都被裁剪，阈值很可能已经变成了隐藏的学习率控制器，而不是异常保护。此时应回头检查初始化、目标缩放、数据范围和优化器配置，而不是只把 $\tau$ 再调小。

## 阈值与日志

阈值 $\tau$ 没有脱离模型尺度的普适常数。参数量、loss reduction、batch size、混合精度和优化器都会改变原始范数的数量级。更有用的起点是先不裁剪或用很大的阈值跑一小段，记录 $G_t$ 的分位数，再选择一个能只拦住尾部异常的阈值。

每次更新至少记录：

| 指标 | 公式或含义 | 用途 |
| --- | --- | --- |
| raw norm | $G_t=\lVert\boldsymbol g_t\rVert_2$ | 观察未处理的梯度尺度 |
| clip coefficient | $c_t$ | 判断本步缩短了多少 |
| clipped norm | $\lVert\boldsymbol g_t'\rVert_2$ | 检查是否达到阈值 |
| clip fraction | $c_t<1$ 的 step 比例 | 判断裁剪是保护还是常态 |
| update ratio | $\lVert\Delta\boldsymbol\theta_t\rVert_2/\lVert\boldsymbol\theta_t\rVert_2$ | 连接梯度到实际参数位移 |
| max coordinate | $\max_i\lvert g_{t,i}\rvert$ | 发现单坐标尖峰 |

还要写明范数是在裁剪前还是反缩放后计算，是在单卡还是全局归约后计算，以及是否包括权重衰减项。否则两个实验报告里的“梯度范数”可能不是同一个量。

如果不同参数张量的尺度相差很大，也可以把固定绝对阈值换成自适应梯度裁剪（adaptive gradient clipping、AGC）。它先比较梯度范数与参数范数的比例：

$$
r_k
=\frac{\lVert\boldsymbol g^{(k)}\rVert_2}
{\lVert\boldsymbol\theta^{(k)}\rVert_2+\varepsilon},
\qquad
\boldsymbol g^{(k)\prime}
=\min\left(1,\frac{\lambda_{\mathrm{agc}}}{r_k}\right)
\boldsymbol g^{(k)}.
$$

AGC 的阈值 $\lambda_{\mathrm{agc}}$ 是相对尺度，不是全局范数裁剪的绝对 $\tau$；它也仍然需要记录被裁剪的张量比例，并检查参数范数很小时的下限。它适合解决参数组尺度差异，不能替代对 loss、数据、初始化和优化器的诊断。

## 梯度累积时先聚合还是先裁剪

设两个 micro-batch 的梯度分别为

$$
\boldsymbol g_1=(3,4),
\qquad
\boldsymbol g_2=(-3,4).
$$

若一次 optimizer step 使用它们的平均梯度，正确的平均值是

$$
\bar{\boldsymbol g}
=\frac{\boldsymbol g_1+\boldsymbol g_2}{2}
=(0,4).
$$

以 $\tau=2$ 做全局范数裁剪，得到

$$
\operatorname{clip}(\bar{\boldsymbol g};2)=(0,2).
$$

若每个 micro-batch 先裁剪再平均，则

$$
\frac{
(1.2,1.6)+(-1.2,1.6)
}{2}
=(0,1.6).
$$

两者都没有违反阈值，但它们不是同一个有效梯度。若目标是模拟一个由 $A$ 个 micro-batch 组成的有效 batch,通常应先完成梯度归约和平均，再对这一次参数更新的梯度裁剪。每个 micro-batch 单独裁剪是另一种明确的鲁棒聚合规则，不能悄悄混入。

梯度累积代码还要确认：

- 每次反向前是否清空或累加梯度；
- 梯度平均是在除以 $A$ 之前还是之后；
- scheduler 和 optimizer step 是否只在有效 batch 结束时推进；
- clip fraction 的分母是 micro-batch 数还是 optimizer step 数。

## 混合精度必须先反缩放

混合精度常把 loss 乘上一个 scale $s$，使低精度反向梯度不容易下溢。此时得到的是

$$
\boldsymbol g_{\mathrm{scaled}}
=s\boldsymbol g.
$$

如果直接对它计算范数，会得到

$$
\lVert\boldsymbol g_{\mathrm{scaled}}\rVert_2
=s\lVert\boldsymbol g\rVert_2.
$$

阈值判断会被 scale 人为放大。正确顺序是：

$$
\boldsymbol g_{\mathrm{scaled}}
\xrightarrow{\operatorname{unscale}}
\boldsymbol g
\xrightarrow{\operatorname{clip}}
\boldsymbol g'
\xrightarrow{\operatorname{optimizer}}
\boldsymbol\theta_{t+1}.
$$

还应先检查非有限值。如果梯度中已经有 NaN 或无穷大，把它们乘一个小系数不会恢复有效方向；应跳过本次更新并记录溢出原因。

## 分布式训练的全局范数

多卡训练中，每张卡先看到局部梯度 $\boldsymbol g^{(r)}$。如果目标是裁剪全局平均梯度，应先做与训练一致的 all-reduce,再计算

$$
\boldsymbol g_{\mathrm{global}}
=\frac1R\sum_{r=1}^{R}\boldsymbol g^{(r)},
\qquad
G_{\mathrm{global}}
=\lVert\boldsymbol g_{\mathrm{global}}\rVert_2.
$$

也有实现先归约各卡的平方范数再求和，但必须和梯度平均的约定匹配。每卡独立裁剪再 all-reduce 会让卡间的相对方向和幅度发生变化，它限制的是局部异常而不是全局更新。

检查分布式裁剪时，用一个固定小模型在单卡和多卡上喂相同的有效 batch,比较：

1. 归约后的原始范数；
2. 全局裁剪系数；
3. optimizer 更新前的梯度；
4. 参数 checksum。

四项都能对齐，才能确认“多卡梯度裁剪”等价于单卡基线。

## 裁剪与动量、Adam、权重衰减的顺序

对动量 SGD,一种清晰的顺序是先裁剪原始梯度，再写入速度：

$$
\begin{aligned}
\boldsymbol g_t'&=\operatorname{clip}(\boldsymbol g_t;\tau),\\
\boldsymbol v_{t+1}&=\beta\boldsymbol v_t+\boldsymbol g_t',\\
\boldsymbol\theta_{t+1}
&=\boldsymbol\theta_t-\eta\boldsymbol v_{t+1}.
\end{aligned}
$$

如果先把 $\boldsymbol g_t$ 写进动量再裁剪 $\boldsymbol v_{t+1}$，限制的是含历史信息的方向，语义已经不同。两种规则都可能有用，但阈值和日志不能混称为 raw-gradient clipping。

对 Adam,通常先裁剪进入一阶矩和二阶矩的梯度：

$$
\boldsymbol g_t'
\longrightarrow
\boldsymbol m_{t+1},\boldsymbol v_{t+1}
\longrightarrow
\widehat{\boldsymbol m}_{t+1}/
(\sqrt{\widehat{\boldsymbol v}_{t+1}}+\varepsilon).
$$

如果先做 Adam 的逐坐标预条件，再按预条件后的向量裁剪，约束的是实际方向的另一种尺度；它可能比 raw-gradient clipping 更接近参数更新，但必须用另一个名字和另一个阈值记录。

解耦权重衰减通常是独立的参数收缩：

$$
\boldsymbol\theta_{t+1}
=(1-\eta\lambda)\boldsymbol\theta_t
-\eta\boldsymbol d_t.
$$

要先说明裁剪对象是否包含 $-\lambda\boldsymbol\theta_t$。若把耦合衰减项混进梯度后裁剪，大参数的正则项也可能被截断；若使用解耦衰减，它通常不进入 raw gradient norm。

## 一个标准库数值实验

下面的短实验只用 Python 标准库，把三个容易混淆的差别放在同一张“演算纸”上：

1. 全局范数裁剪保留方向，逐元素裁剪不保留方向；
2. 梯度累积时，先平均再裁剪与每个 micro-batch 先裁剪再平均并不等价；
3. 一个异常大的梯度会把未裁剪的参数轨迹推过头，而裁剪把单步变化限制在阈值以内。

```python
import math

def norm(vector):
    return math.sqrt(sum(value * value for value in vector))

def clip_by_norm(vector, threshold):
    magnitude = norm(vector)
    factor = min(1.0, threshold / magnitude) if magnitude else 1.0
    return [factor * value for value in vector], factor

def clip_by_value(vector, limit):
    return [max(-limit, min(limit, value)) for value in vector]

single, factor = clip_by_norm([3.0, 4.0], 2.0)
value = clip_by_value([3.0, 4.0], 2.0)
g1 = [3.0, 4.0]
g2 = [-3.0, 4.0]
average = [(a + b) / 2.0 for a, b in zip(g1, g2)]
after, after_factor = clip_by_norm(average, 2.0)
micro1, _ = clip_by_norm(g1, 2.0)
micro2, _ = clip_by_norm(g2, 2.0)
before = [(a + b) / 2.0 for a, b in zip(micro1, micro2)]

def run(threshold=None):
    theta = 5.0
    history = []
    for impulse in [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]:
        gradient = theta + impulse
        if threshold is not None:
            gradient, _ = clip_by_norm([gradient], threshold)
            gradient = gradient[0]
        theta -= 0.2 * gradient
        history.append(theta)
    return history

print("single", f"norm={norm([3.0, 4.0]):.1f}", "factor=", f"{factor:.1f}", "global=", [round(v, 1) for v in single], "value=", [round(v, 1) for v in value])
print("accumulation", "average=", [round(v, 1) for v in average], "clip-after=", [round(v, 1) for v in after], "micro-clip-average=", [round(v, 1) for v in before])
print("trajectory", "no-clip=", [round(v, 4) for v in run()], "clip-2=", [round(v, 4) for v in run(2.0)])
```

输出是：

```text
single norm=5.0 factor= 0.4 global= [1.2, 1.6] value= [2.0, 2.0]
accumulation average= [0.0, 4.0] clip-after= [0.0, 2.0] micro-clip-average= [0.0, 1.6]
trajectory no-clip= [4.0, 3.2, -1.44, -1.152, -0.9216, -0.7373] clip-2= [4.6, 4.2, 3.8, 3.4, 3.0, 2.6]
```

第一行再次显示 $(3,4)$ 的二范数为 $5$，因而范数裁剪的系数是 $2/5=0.4$，结果仍在同一条射线上；逐元素裁剪则把两个坐标分别压到 $2$，方向改变。第二行的平均梯度是 $(0,4)$：先聚合再裁剪得到 $(0,2)$，每个 micro-batch 先裁剪再平均得到 $(0,1.6)$。第三行把一次 $20$ 的冲击放进简单二次目标；不裁剪的轨迹在第三步翻到负侧，阈值为 $2$ 时每步变化被限制，但也因此明显更慢。这里的 `clip-2` 只是说明机制，不是一个可直接搬到所有任务的推荐阈值。

## 失效模式：阈值也会被用错

梯度裁剪本身不是稳定性的证明。训练日志至少要能区分“原始梯度很大”“裁剪系数很小”“优化器预条件后更新很大”这三种情况：

| 现象 | 可能的误读或故障 | 应核对的证据 |
| --- | --- | --- |
| 每步 $c_t\ll1$ | 阈值过小，或损失缩放、学习率、初始化本身有问题 | raw norm、loss reduction、阈值和学习率 |
| 裁剪很少但模型不学习 | 裁剪了正常梯度，或问题不在异常大梯度 | clip fraction、平均 $c_t$、update-to-weight ratio |
| 单卡与多卡结果不同 | 各卡本地裁剪后才平均，而基线使用全局范数 | all-reduce 后的 norm 与全局系数 |
| AMP 下频繁跳过更新 | 在 scaled gradient 上裁剪，或没有先检查非有限值 | unscale 后 norm、overflow 标记、跳过次数 |
| 累积步结果不同 | 每个 micro-batch 都裁剪，改变了聚合顺序 | 梯度求和/平均的位置与裁剪时刻 |
| 动量后仍出现长尾更新 | 裁剪对象其实是速度，或恢复 checkpoint 时状态缺失 | raw gradient、velocity、checkpoint 中的优化器状态 |
| norm 正常但 update 很大 | Adam 预条件、权重衰减或参数组单位放大了实际更新 | preconditioned update、参数组学习率、weight decay |
| 梯度仍然趋近于零 | 裁剪只能限制上界，不能修复梯度消失 | 初始化、激活函数、各层梯度与激活统计 |

因此“loss 没有发散”还不够作为验收标准。至少要同时观察 loss、raw gradient norm、裁剪系数或裁剪比例、实际参数更新范数和验证集指标；否则裁剪可能只是把一个根因藏在更平滑的曲线后面。

## 运行方法

把上面的代码保存为一个文件，文件名是：

`gradient_clip_probe.py`

在项目环境中运行：

```bash
python3 gradient_clip_probe.py
```

接下来应把同一组固定输入接到实际训练循环，先写一个单步单元测试，验证 raw norm、裁剪系数和更新后的参数；再一次只加入一个变量，依次核对梯度累积、AMP 反缩放和分布式 all-reduce。只有这些边界都与基线对齐，才值得把阈值调参结果解释为模型或数据的性质。

## 相关词条

- [梯度消失与爆炸](../backpropagation/vanishing-and-exploding/)
- [权重初始化](../training-nn/weight-initialization/)
- [优化器](../training-nn/optimizers/)
- [Adam](../training-nn/adam/)
- [小批次随机梯度下降](../training-nn/minibatch-sgd/)
- [动量与 Nesterov](../training-nn/momentum-and-nesterov/)
- [学习率调度](../training-nn/learning-rate-schedules/)
