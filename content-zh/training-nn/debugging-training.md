---
title: "训练调试：把“loss 不对”拆成可定位的证据"
tags: ["why-models-learn"]
---

训练调试（debugging training）不是在学习率、优化器和正则化之间盲目试参，而是把一条训练循环拆成数据、前向、损失、反向、更新、评估和运行状态几个边界，找到第一个不符合预期的量。只要一个四点小数据集都无法被模型稳定拟合，就不应先讨论泛化；只有最小循环通过了数据合同、有限性、梯度和参数更新检查，训练曲线才值得解释。

![训练调试示意图：从数据合同经过前向损失、反向梯度和参数更新，每个边界都记录形状、范围、有限性与模式状态，直到定位第一个分叉](/assets/training-nn/svg/debugging-training.1.svg)

## 先把“训练失败”分成几层

一条最小训练循环可以抽象为

$$
\begin{aligned}
\widehat{y}_i&=f_\theta(x_i),\\
J(\theta)&=\frac{1}{N}\sum_{i=1}^{N}\ell(\widehat{y}_i,y_i),\\
g_t&=\nabla_\theta J(\theta_t),\\
\theta_{t+1}&=\operatorname{Update}(\theta_t,g_t,\text{state}_t).
\end{aligned}
$$

最后一个 loss 没有下降，只说明这条链的结果不符合预期，并没有告诉我们是 $x_i$ 错位、$f_\theta$ 输出范围错误、$\ell$ 的符号错误、$g_t$ 没有传回来、更新方向反了，还是评估模式与训练模式不一致。

| 层 | 要回答的问题 | 第一批证据 |
| --- | --- | --- |
| 数据合同 | 样本、标签、mask 和 batch 轴是否对应 | shape、dtype、标签范围、重复与泄漏 |
| 前向 | 输入经过每层后仍是有限且有尺度的值吗 | min/max、均值、方差、非有限比例 |
| 损失 | 计算的确实是目标函数吗 | 单样本手算、reduction、logits/probability 边界 |
| 反向 | 每个参数是否收到预期方向和尺度的梯度 | gradient norm、非零比例、有限性 |
| 更新 | 参数是否按正确符号和步长移动 | update norm、update ratio、参数快照 |
| 评估 | train/eval、mask、统计量和指标是否一致 | 模式、数据切分、逐样本预测 |
| 运行状态 | seed、optimizer、scheduler、AMP 和 checkpoint 是否对齐 | 配置快照、step、随机状态、scaler |

调试时要保存“第一个出错位置”，而不是只保存最终的 loss=nan。如果前向的第二层第一次出现 inf，后面的 loss、梯度和参数都是后果；如果梯度是有限的但 update 为零，问题已经从计算图转移到了优化器、参数 dtype 或状态更新。

## 先写出数据合同

数据合同是训练代码在进入模型前必须满足的可检查条件。对一个 $K$ 类互斥分类 batch，可以写成

$$
X\in\mathbb R^{N\times d},
\qquad
y\in\{0,1,\ldots,K-1\}^{N}.
$$

这两个形状只说明样本数和特征数对上了，还没有说明标签顺序正确。至少要检查：

1. $X$ 和 $y$ 的第一维相等；
2. 标签没有超出输出头的类别范围；
3. 输入中没有未处理的 NaN、Inf 或错误的量纲；
4. 训练、验证和测试的预处理参数只在训练集上拟合；
5. shuffle、mask、权重和标签使用同一个样本索引；
6. 任务的标签语义与损失的输出语义一致。

表格中“类别数正确”不能证明“标签对齐”。最危险的错误之一是分别打乱特征和标签：两个数组 shape 仍然完全正确，模型却只能把噪声当作规律。调试小数据时应打印几行原始样本、变换后样本和标签，而不是只打印 batch shape。

数据泄漏也属于数据合同错误。若标准化均值、类别表、未来窗口或标签派生特征从验证/测试集提前计算，训练 loss 仍然可以正常下降，甚至更好看；问题只会在独立部署数据上暴露。未毕业的交叉引用先用“数据泄漏”作为待核对类别；已毕业的切分与泛化条目则说明具体边界。

## 第一关是单样本和单批次检查

在完整训练前，先固定一个样本，手算或打印：

$$
x,\qquad
f_\theta(x),\qquad
\ell(f_\theta(x),y),\qquad
\nabla_\theta\ell.
$$

这一步要关闭会改变函数的随机层，或者固定并记录它们的掩码。若同一个输入连续两次前向结果不同，先解释随机性，再谈梯度。

接着用一个极小 batch 做过拟合测试。设 batch 只有 $m$ 个样本，训练目标是

$$
J_{\mathrm{tiny}}(\theta)
=\frac{1}{m}\sum_{i=1}^{m}\ell(f_\theta(x_i),y_i).
$$

测试时暂时关闭 weight decay、强数据增强和会改变 batch 语义的随机层，给模型足够的更新步数。一个能表达目标的模型，通常应把这个极小集合的损失压低并达到预期训练准确率；若做不到，应先查数据、损失、反向和更新。

“能过拟合一个小 batch”不是泛化证明。它只回答一个窄问题：在当前参数化和训练循环中，模型能否把一组固定监督信号变成更低的目标值。如果连这个问题都回答不了，增加数据、模型宽度或训练时间只会增加观测噪声。

## 前向要记录范围而不是只记录均值

对第 $l$ 层激活 $a^{(l)}$，建议至少记录

$$
\operatorname{finite}(a^{(l)}),
\quad
\min(a^{(l)}),
\quad
\max(a^{(l)}),
\quad
\operatorname{mean}(a^{(l)}),
\quad
\operatorname{std}(a^{(l)}),
\quad
\frac{\#\{a^{(l)}\ne0\}}{\#a^{(l)}}.
$$

均值接近零不能证明没有异常值：一个 batch 可以同时包含极大的正负值，均值恰好抵消；大量零值也会把均值和方差压低。分位数、最大绝对值和非零比例经常比单独的平均值更能定位问题。

不同层有不同的检查重点：

| 位置 | 重点信号 | 典型故障 |
| --- | --- | --- |
| 输入与归一化前 | 量纲、缺失值、异常样本 | 单位错、泄漏、NaN |
| 线性层输出 | 最大绝对值和分布跨度 | 初始化过大、学习率过大 |
| ReLU 输出 | 活动率和零值比例 | 死亡 ReLU、输入整体偏负 |
| sigmoid/tanh 输出 | 饱和比例和局部导数 | logits 过大、梯度消失 |
| softmax/logits | 行和、最大 logit、非有限值 | mask 错、指数不稳定 |
| 归一化层 | 统计轴、方差、训练/评估分支 | batch 太小、running stats 错 |

例如 sigmoid 的输出全部接近 0.5，可能是训练刚开始，也可能是梯度根本没有更新；输出全部接近 0 或 1，可能是分类已经很确定，也可能是 logits 过大导致饱和。不能只看输出的“好不好看”，要把前向统计和对应的梯度统计放在同一时间点。

## 反向要同时看梯度和更新

对每个参数张量 $\theta_l$，记录梯度范数

$$
G_l=\lVert\nabla_{\theta_l}J\rVert_2
$$

以及从参数快照得到的更新范数

$$
U_l=\lVert\theta_{l,t+1}-\theta_{l,t}\rVert_2.
$$

为了比较不同尺度的层，可以记录更新比率

$$
R_l
=\frac{U_l}{\lVert\theta_{l,t}\rVert_2+\varepsilon}.
$$

这三个量回答不同问题：

- $G_l=0$：反向路径没有给出该层梯度，或梯度在数值上被清零；
- $G_l$ 有限且非零、$U_l=0$：优化器没有执行、参数被冻结、dtype 吞掉了更新，或更新被跳过；
- $G_l$ 很大、$U_l$ 也很大：学习率、预条件、梯度裁剪或数值溢出需要检查；
- $R_l$ 远小于其它层：不一定是错误，可能是参数尺度不同，但需要和深度、激活和梯度一起看。

全局梯度范数不能替代逐层统计。一个层的梯度爆炸和另一个层的梯度消失可以在总和中互相掩盖；同一层不同通道也可能有完全不同的方向。[梯度消失与梯度爆炸](../backpropagation/vanishing-and-exploding/)篇展开了路径增益，[梯度裁剪](../training-nn/gradient-clipping/)篇则说明裁剪只能限制更新幅度，不能修复断开的计算图。

如果怀疑导数实现，用固定参数、固定数据、固定随机状态的高精度小图做有限差分。[梯度检查](../backpropagation/gradient-checking/)适合回答“解析梯度是否与当前前向函数局部一致”，不适合直接代替生产训练的 loss 曲线。

## 用故障特征区分三种常见错误

下面的四点数据使用线性二分类器。标签

$$
y_{\mathrm{or}}=(0,1,1,1)
$$

对应一个可被线性边界分开的 OR 任务；标签

$$
y_{\mathrm{xor}}=(0,1,1,0)
$$

对应线性模型无法完全拟合的 XOR 任务。代码故意比较三种运行：

1. 正确的梯度下降更新；
2. 把减号写成加号的反向更新；
3. 用线性模型训练 XOR 标签，并从零参数开始。

标准库探针输出如下：

| 运行 | loss（0→100） | accuracy（0→100） | 最终信号 |
| --- | --- | --- | --- |
| correct-or | 0.693147 → 0.156254 | 0.75 → 1.0 | 梯度 0.047104，参数最大绝对值 2.834662 |
| wrong-sign | 0.693147 → 51.783787 | 0.75 → 0.25 | 梯度 1.030776，参数最大绝对值 36.349408 |
| xor-labels | 0.693147 → 0.693147 | 0.5 → 0.5 | 梯度 0，最后一次更新 0 |

表格只列第 0 步和第 100 步，完整的五个检查点保留在后面的原始输出中；这样既能比较轨迹方向，也不会让结果表变成横向滚动的长字符串。

第一行说明一个可表达任务的模型正在学习；第二行说明损失实现可能没错，但更新方向反了；第三行不能简单解释成“梯度代码坏了”。对于对称的 XOR 数据和零初始化，线性模型在原点的平均梯度恰好为零，模型停在一个非最优的平坦点。换一个非线性模型、打破初始化对称或改变数据，就会得到不同的信号。

所以“梯度为零”必须和任务表达能力、初始化、激活状态以及标签结构一起解释。准确率 0.5 也不能单独证明标签随机；它可能是二分类基线、模型表达不足或预测阈值未校准。

## 稳定损失是调试的必要边界

对二分类 logits $z$ 和标签 $y\in\{0,1\}$，稳定的 BCE with logits 可以写成

$$
\ell(z,y)
=\max(z,0)-zy+\log\left(1+\exp(-\lvert z\rvert)\right).
$$

它与先计算 sigmoid 再取对数在实数算术中等价，但不会在 $z$ 很大或很小时直接计算危险的指数。探针令 $z=1000,y=1$：

- 直接计算 $\exp(1000)$ 会触发 OverflowError；
- 稳定 logits 公式的损失仍为 $0.000000$，因为模型对正确类别已经极度确定。

这不是说极端 logits 永远正确。它只说明“损失公式稳定”和“模型训练健康”是两个问题：稳定公式避免了实现先崩溃，仍要检查梯度是否饱和、标签是否正确、参数是否继续更新。

多分类、回归、mask 和权重也有相同的审计边界：

- logits 损失是否重复调用 sigmoid 或 softmax；
- mean 的分母是样本数、有效 mask 数还是权重和；
- 全部位置被 mask 时返回什么；
- 类别权重是否也改变了报告指标；
- 预测头的输出范围是否覆盖目标；
- 训练 loss 与评估 metric 是否使用同一批样本和同一单位。

损失数值越小不一定代表实现越正确。若把 reduction 多除了一次，loss 可能看起来更平滑；若把 padding 也计入平均，长序列的指标会被无效位置改变。先在一个能手算的 batch 上固定分母，再放大到完整数据。

## 学习率和优化器要在最小循环通过后调整

学习率影响的是已知梯度被转换成参数位移的尺度。对最简单的梯度下降，

$$
\Delta\theta_t
=-\eta_t g_t.
$$

如果 loss 爆炸且梯度方向经过高精度检查，先尝试缩小学习率；如果 loss 几乎不动且更新比率极小，才考虑增大学习率或检查梯度/参数是否被冻结。不要用降低学习率去修复标签错位、加号更新或 NaN。

调试学习率时可以做一个小范围扫描，但每次运行要固定：

| 要固定的量 | 原因 |
| --- | --- |
| 数据顺序和 batch 边界 | 否则比较的是不同噪声路径 |
| 初始化和随机种子 | 参数初值可能改变可达轨迹 |
| optimizer 与 scheduler step 语义 | 同一个 eta 不代表同一个有效更新 |
| loss reduction 和权重 | 目标整体缩放会改变有效步长 |
| 日志时间点 | update 前后混淆会制造假趋势 |

Adam、Momentum 和带权重衰减的优化器还会维护状态。修改学习率时不要忘记 state 是否从旧实验恢复；恢复一个不匹配的二阶矩，可能让首批更新与“新学习率”不相称。[优化器](../training-nn/optimizers/)、[Adam 优化器](../training-nn/adam/)和[学习率调度](../training-nn/learning-rate-schedules/)篇分别解释状态、预条件和时间索引。

## 训练态、评估态和随机性是同一个实验协议

训练和评估不是同一段代码只切换一个 training 布尔值就结束了。至少要固定：

- Dropout 是否抽取掩码；
- BatchNorm 是否用当前 batch 统计并更新 running buffer；
- 数据增强是否只作用于训练集；
- autocast、loss scaling 和梯度是否只存在于训练路径；
- 验证指标是否在 no-grad/inference 模式中计算；
- 评估时是否误用了训练 batch 的随机顺序或状态。

随机性要区分“可复现”和“统计等价”。固定一个 seed 只能让某些随机源重复，不能自动覆盖多进程数据加载、GPU kernel、异步通信、环境版本和非确定性算子。可复现调试应保存 seed、数据索引、模型初始化、配置、依赖版本和每次 update 的 step；统计实验则应报告多个 seed 的均值和波动。

若 train loss 下降而 eval loss 立刻异常，先比较同一批样本在 train/eval 两种模式的逐样本输出。这样能把模式分支与泛化问题分开；不要只看两条曲线的最终数字。

## 混合精度和分布式训练要查第一个分叉

混合精度中，loss scaling、unscale、有限性检查、梯度裁剪和 optimizer step 必须按固定顺序执行。[混合精度训练](../training-nn/mixed-precision/)篇的核心诊断是：若梯度在 scaled 路径中已经变成 Inf，后面再除以 scale 不能恢复；若 unscale 后梯度有限但参数不动，要查 master weights、optimizer state 和 skip 计数。

分布式训练还增加了设备之间的边界：

1. 每个 rank 的输入 batch 和标签是否按预期切分；
2. loss 是局部平均还是全局平均；
3. 梯度 all-reduce 前后是否重复除以 world size；
4. 某个 rank 先出现非有限值时，其他 rank 是否同步跳步；
5. 全局梯度裁剪使用的是局部范数还是汇总后的范数；
6. checkpoint 是否保存所有 rank 需要的 sampler 和通信状态。

“平均 loss 看起来对”不能证明通信正确。用一个两 rank、两样本的手算例，分别打印局部梯度、归约后梯度和最终更新，通常比直接跑大模型更快找到 world-size 或 reduction 错误。

## checkpoint 恢复是调试的一部分

训练脚本能从头跑完，不代表它能正确恢复。一次可复现恢复至少要保存：

| 状态 | 核对问题 |
| --- | --- |
| 模型参数 | 保存的是当前前向副本还是高精度主权重 |
| optimizer state | 动量、二阶矩和 step 是否连续 |
| scheduler | 恢复后下一次学习率是否与未中断轨迹相同 |
| scaler | scale、增长计数和 skip 状态是否连续 |
| 数据位置 | sampler、epoch、batch 和累积窗口是否对齐 |
| 随机状态 | dropout、增强和采样是否按协议继续 |

最小恢复测试是把一个短训练拆成“前 $r$ 步保存 + 恢复后继续”，然后逐步比较未拆分运行与恢复运行的 loss、梯度有限性、参数和 optimizer step。若使用非确定性 kernel，不能要求位级相同，但要先声明允许误差和哪些状态只要求统计一致。

恢复后第一步明显不同，先查 checkpoint 内容和 step 顺序；不要立刻把差异归因于“重新 warmup”。训练状态本身就是模型行为的一部分。

## 常见失效模式与第一证据

| 现象 | 可能原因 | 第一证据 |
| --- | --- | --- |
| loss 从第一步就是 NaN | 输入、logits、损失或归一化已非有限 | 首个非有限张量和原始输入 |
| loss 不变、梯度全零 | 标签对称、模型未连接、激活死区或参数被冻结 | 逐层梯度非零比例和 update norm |
| loss 上升、参数迅速变大 | 更新符号、学习率、reduction 或数据尺度错误 | 一步前后 loss、梯度方向、参数差 |
| 训练准确率也上不去 | 数据/标签错、输出头错、容量不足或优化未启动 | 单样本输出、小 batch 过拟合 |
| 训练很好、验证很差 | 切分、泄漏、模式、增强或过拟合 | 逐样本 train/eval 输出与切分记录 |
| loss 下降但指标不变 | metric 阈值、标签映射、mask 或单位不一致 | 预测、标签、逐样本损失对照 |
| 只有部分 rank 失败 | 数据切分、通信、非确定性或 rank 间状态不同 | rank-local finite、梯度和 step |
| 恢复后曲线断裂 | optimizer、scheduler、scaler、sampler 或 RNG 丢失 | checkpoint 字段与恢复后的第一步 |
| 低精度才失败 | 下溢、上溢、归约 dtype 或主权重缺失 | dtype 路径、scaled/unscaled 梯度 |

一个症状可以有多个原因。调试记录应保留已排除的假设和对应证据，否则重复试验会把同一个问题重新命名。

## 一个可复用的训练调试协议

按下面的顺序执行，直到某一关失败：

1. **固定输入。** 保存一个样本、一个小 batch、标签、mask、参数快照和随机状态。
2. **核对合同。** 检查 shape、dtype、标签范围、样本对齐、预处理和切分。
3. **核对前向。** 对每层记录范围、均值、方差、非零比例和有限性。
4. **核对损失。** 用一个手算样本验证输出头、logits/probability、reduction 和 mask 分母。
5. **核对反向。** 记录逐层梯度范数、非零比例和有限性；必要时做高精度有限差分。
6. **核对一步更新。** 比较参数前后快照、更新符号、更新范数和 update ratio。
7. **跑 tiny-batch overfit。** 关闭会改变目标的随机正则，确认固定小数据能下降。
8. **再调优化器。** 扫描学习率和状态协议，记录 scheduler、weight decay、裁剪和有效 batch。
9. **再打开复杂运行条件。** 逐一恢复增强、Dropout、BatchNorm、混合精度、分布式和 checkpoint。
10. **分开报告。** 训练风险、验证风险、部署输出和性能成本使用各自的证据，不用一个 loss 代替全部结论。

每一步都应有一个可以失败的断言。没有断言的“观察日志”很容易变成事后解释；有断言的最小循环才会告诉你下一步该看哪一层。

## 一个可复算的标准库调试探针

下面的代码实现一个没有第三方依赖的二分类 logits 训练循环。它故意保留三个对照：正确的 OR 任务、符号写反的更新、从零初始化的 XOR 标签，以及一个两样本 tiny-batch overfit。最后还对比不稳定指数和稳定 BCE with logits。

```python
import math


def sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def loss_grad(w, b, xs, ys):
    loss = 0.0
    gw = [0.0, 0.0]
    gb = 0.0
    correct = 0
    for x, y in zip(xs, ys):
        z = w[0] * x[0] + w[1] * x[1] + b
        p = sigmoid(z)
        loss += max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))
        dz = p - y
        gw[0] += dz * x[0]
        gw[1] += dz * x[1]
        gb += dz
        correct += int((p >= 0.5) == bool(y))
    n = len(xs)
    return loss / n, [g / n for g in gw], gb / n, correct / n


def norm(values):
    return math.sqrt(sum(v * v for v in values))


def train(xs, ys, lr=0.5, steps=100, sign=-1):
    w = [0.0, 0.0]
    b = 0.0
    history = []
    for step in range(steps + 1):
        loss, grad, gb, acc = loss_grad(w, b, xs, ys)
        grad_norm = norm([grad[0], grad[1], gb])
        if step in (0, 1, 10, 50, steps):
            history.append(
                (step, loss, acc, grad_norm,
                 max(abs(w[0]), abs(w[1]), abs(b)))
            )
        if step == steps:
            break
        old = [w[0], w[1], b]
        w[0] += sign * lr * grad[0]
        w[1] += sign * lr * grad[1]
        b += sign * lr * gb
        update_norm = norm(
            [w[0] - old[0], w[1] - old[1], b - old[2]]
        )
    return history, (w, b, update_norm)


xs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
or_ys = [0, 1, 1, 1]
xor_ys = [0, 1, 1, 0]

for name, ys, sign in [
    ("correct-or", or_ys, -1),
    ("wrong-sign", or_ys, 1),
    ("xor-labels", xor_ys, -1),
]:
    history, (w, b, update_norm) = train(
        xs, ys, steps=100, sign=sign
    )
    final = history[-1]
    checkpoints = [
        (step, round(loss, 6), round(acc, 3))
        for step, loss, acc, _, _ in history
    ]
    print(
        name,
        "checkpoints=",
        checkpoints,
        "final_grad_norm=",
        format(final[3], ".6f"),
        "final_max_param=",
        format(final[4], ".6f"),
        "last_update_norm=",
        format(update_norm, ".6f"),
    )

small_history, (small_w, small_b, small_update) = train(
    xs[:2], or_ys[:2], steps=200
)
print(
    "tiny-batch",
    "loss=",
    format(small_history[-1][1], ".6f"),
    "accuracy=",
    format(small_history[-1][2], ".3f"),
    "weights=",
    [round(value, 6) for value in small_w],
    "bias=",
    format(small_b, ".6f"),
)

try:
    math.exp(1000.0)
except OverflowError:
    stable = loss_grad([1000.0, 0.0], 0.0, [(1.0, 0.0)], [1])[0]
    print(
        "unstable-exp= OverflowError",
        "stable-bce-logits=",
        format(stable, ".6f"),
    )
```

运行输出：

```text
correct-or checkpoints= [(0, 0.693147, 0.75), (1, 0.60815, 0.75), (10, 0.401529, 0.75), (50, 0.23901, 1.0), (100, 0.156254, 1.0)] final_grad_norm= 0.047104 final_max_param= 2.834662 last_update_norm= 0.023709
wrong-sign checkpoints= [(0, 0.693147, 0.75), (1, 0.79565, 0.25), (10, 4.021528, 0.25), (50, 25.221287, 0.25), (100, 51.783787, 0.25)] final_grad_norm= 1.030776 final_max_param= 36.349408 last_update_norm= 0.515388
xor-labels checkpoints= [(0, 0.693147, 0.5), (1, 0.693147, 0.5), (10, 0.693147, 0.5), (50, 0.693147, 0.5), (100, 0.693147, 0.5)] final_grad_norm= 0.000000 final_max_param= 0.000000 last_update_norm= 0.000000
tiny-batch loss= 0.053613 accuracy= 1.000 weights= [0.0, 5.842057] bias= -2.704929
unstable-exp= OverflowError stable-bce-logits= 0.000000
```

这段程序没有模拟神经网络的隐藏层，但它足以验证调试顺序：正确更新能降低 OR loss，符号反了会让 loss 增大，XOR 的零梯度要结合模型表达能力解释，稳定 logits 损失能处理极端分数。把它替换成真实模型时，保留同样的观测字段和断言。

## 运行方法

将“一个可复算的标准库调试探针”中的代码保存为 `debug_training_probe.py`，在 Python 3 环境执行：

```bash
python3 debug_training_probe.py
```

若要把协议迁移到框架训练器，先让它支持一个固定 batch 的前向、反向和一步更新，再逐项加入数据加载、随机增强、验证、混合精度、分布式和 checkpoint。每加入一层复杂性就重新跑一次固定输入测试，失败时回到最近一个通过的边界。

## 相关词条

- [前向传播](../backpropagation/forward-pass/)：核对线性层、激活、logits 和输出头在计算图中的位置。
- [分类损失](../training-nn/classification-losses/)：比较 BCE with logits、softmax 交叉熵、mask 与 reduction。
- [梯度检查](../backpropagation/gradient-checking/)：把解析梯度和固定执行图的数值梯度对照。
- [梯度消失与梯度爆炸](../backpropagation/vanishing-and-exploding/)：按路径和层定位梯度尺度变化。
- [梯度裁剪](../training-nn/gradient-clipping/)：区分限制更新幅度与修复梯度来源。
- [混合精度训练](../training-nn/mixed-precision/)：检查 dtype、loss scaling、unscale、非有限值和主权重。
- [批量归一化](../training-nn/batch-normalization/)：核对训练/评估统计量和小 batch 行为。
- [Dropout](../training-nn/dropout/)：核对随机掩码、train/eval 和固定 seed。
- [优化器](../training-nn/optimizers/)：查看状态、更新顺序和参数组。
- [学习率调度](../training-nn/learning-rate-schedules/)：核对 scheduler 的 step 时间轴与恢复行为。
- [训练集、验证集与测试集](../learning-framework/train-validation-test/)：核对切分、泄漏和最终评估边界。
