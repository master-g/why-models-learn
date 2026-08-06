---
title: "Dropout：训练时随机断开，推理时恢复尺度"
tags: ["why-models-learn"]
---

Dropout 是在训练前向中按随机掩码暂时置零一部分激活，并按保留概率补偿其尺度的正则化方法。它让每次更新看到一个不同的子网络；评估时通常关闭随机掩码，直接使用完整激活。真正容易写错的地方不是“随机置零”四个字，而是丢弃概率与保留概率的方向、训练态缩放、反向复用同一掩码、掩码共享轴以及训练/评估模式是否真的切换。

![Dropout 示意图：训练时由 Bernoulli 掩码选择并按保留概率缩放激活，评估时取消随机掩码而保持期望尺度](/assets/training-nn/svg/dropout.1.svg)

## 先固定丢弃概率和保留概率

设一个标量激活为 $x_i$。本文用 $p$ 表示丢弃概率，用

$$
q=1-p
$$

表示保留概率。训练时为每个需要随机化的坐标采样一个二值掩码

$$
m_i\sim\operatorname{Bernoulli}(q),
\qquad
\Pr(m_i=1)=q,
\qquad
\Pr(m_i=0)=p.
$$

这里的 $p$ 与很多库的参数名约定相反：有的 API 参数叫 p，却表示丢弃率；有的底层实现直接接收 keep probability。写公式和读接口时，不能只看变量名，必须确认它代表哪一个事件。

| 记号 | 含义 | 代码审计问题 |
| --- | --- | --- |
| $p$ | 丢弃概率 $\Pr(m_i=0)$ | 接口的 p 是丢弃率还是保留率 |
| $q$ | 保留概率 $1-p$ | 是否在所有分支使用同一个 $q$ |
| $m_i$ | 本次前向实际采样的 0/1 掩码 | 反向和重算是否复用它 |
| $1/q$ | inverted Dropout 的训练缩放 | 是否把缩放重复做了两次 |

只要 $q=0$，$1/q$ 就没有定义，所以合法的随机失活要满足 $0\leq p<1$。当 $p=0$ 时，掩码恒为 1，Dropout 退化为恒等映射。

## Inverted Dropout 的前向公式

工程中更常用 inverted Dropout：训练时缩放，评估时不缩放。给定一批固定输入，训练态输出为

$$
\widetilde{x}_i
=\frac{m_i x_i}{q}.
$$

因为 $\mathbb E[m_i]=q$，逐坐标的期望是

$$
\begin{aligned}
\mathbb E[\widetilde{x}_i]
&=\mathbb E\left[\frac{m_i x_i}{q}\right]\\
&=\frac{x_i}{q}\mathbb E[m_i]\\
&=x_i.
\end{aligned}
$$

期望尺度保持不变，但每一次具体前向仍然可能不同。二阶矩为

$$
\mathbb E[\widetilde{x}_i^2]
=\frac{x_i^2}{q^2}\mathbb E[m_i^2]
=\frac{x_i^2}{q},
$$

因为二值掩码满足 $m_i^2=m_i$。因此方差为

$$
\begin{aligned}
\operatorname{Var}(\widetilde{x}_i)
&=\mathbb E[\widetilde{x}_i^2]
 -\mathbb E[\widetilde{x}_i]^2\\
&=\frac{x_i^2}{q}-x_i^2\\
&=\frac{p}{q}x_i^2.
\end{aligned}
$$

输入幅值越大，Dropout 引入的绝对噪声越大；$p$ 越接近 1，$p/q$ 越大。保持均值不变不等于保持方差不变，也不等于保持非线性网络输出不变。

另一种历史约定叫 original Dropout：训练时使用 $m_i x_i$，评估时再乘 $q$。两种约定在缩放位置一致、且中间没有改变语义的操作时可以得到相同的期望尺度，但实现不能把两套缩放同时使用。inverted Dropout 把评估分支保持为恒等映射，通常更容易与部署代码对齐。

| 约定 | 训练输出 | 评估输出 | 常见错误 |
| --- | --- | --- | --- |
| inverted | $m_i x_i/q$ | $x_i$ | 评估又乘一次 $q$ |
| original | $m_i x_i$ | $q x_i$ | 训练和评估都不缩放 |

## 一个三维数值例子

取

$$
x=(2,-1,4),
\qquad
p=0.5,
\qquad
q=0.5.
$$

如果本次掩码为 $m=(1,0,1)$，训练输出是

$$
\widetilde{x}
=\frac{(1,0,1)\odot(2,-1,4)}{0.5}
=(4,0,8).
$$

第二个坐标虽然被置零，但第一、第三个坐标被乘以 2。若对三个坐标的所有 $2^3$ 个掩码等权枚举，输出均值恰好回到 $(2,-1,4)$，三个坐标的方差分别为

$$
\frac{p}{q}(2^2,(-1)^2,4^2)=(4,1,16).
$$

这组数说明了两个容易混淆的事实：

- 单次输出不是输入的缩小版，而是部分坐标为零、其余坐标被放大的稀疏版本；
- 期望只描述大量随机前向的平均，不能把一次掩码输出直接替换成评估输出。

如果错误地把 $m\odot x$ 当作 inverted Dropout 的训练输出，例子会变成 $(2,0,4)$。它的期望是 $q x$，比评估分支小一半，后续层会在训练和评估之间看到不同尺度。

## 它给线性预测增加了什么

先看一个线性单元，令

$$
z=w^{\mathsf T}\widetilde{x}+b
=\sum_i w_i\frac{m_i x_i}{q}+b.
$$

在输入、权重和偏置固定，且不同坐标掩码独立时，

$$
\mathbb E[z]=w^{\mathsf T}x+b.
$$

独立性让方差可以逐坐标相加：

$$
\operatorname{Var}(z)
=\frac{p}{q}\sum_i(w_i x_i)^2.
$$

对平方损失和目标 $y$，使用

$$
\mathbb E[(y-z)^2]
=(y-\mathbb E[z])^2+\operatorname{Var}(z)
$$

可得

$$
\mathbb E[(y-z)^2]
=(y-w^{\mathsf T}x-b)^2
+\frac{p}{q}\sum_i w_i^2x_i^2.
$$

在这个简化的线性平方损失里，输入 Dropout 等价于在期望目标中增加一个按输入幅值加权的 L2 惩罚。它解释了 Dropout 为什么有正则化效果，但不能把这个结论无条件搬到深层非线性网络：一般有

$$
\mathbb E[f(\widetilde{x})]\ne f(\mathbb E[\widetilde{x}]).
$$

所以“评估时使用完整网络”是一个期望尺度近似和工程约定，不是说完整网络与所有随机子网络逐次相等。

也不能把 Dropout 只解释成“平均了很多子网络”。掩码共享轴、参数更新过程、非线性、优化器状态和训练数据顺序都会改变这些子网络之间的关系。这个视角有助于理解正则化，却不能替代实际的 train/eval 和数值检查。

## 反向传播必须复用本次掩码

设上游已经给出

$$
\delta_i=\frac{\partial L}{\partial\widetilde{x}_i}.
$$

对本次前向采样的 $m_i$，局部导数是

$$
\frac{\partial L}{\partial x_i}
=\frac{m_i}{q}\delta_i.
$$

因此被丢弃的坐标 $m_i=0$ 不接收来自这条 Dropout 路径的梯度；保留坐标的梯度被乘以 $1/q$。向量形式可以写成

$$
\nabla_x L
=\operatorname{diag}\left(\frac{m_1}{q},\ldots,\frac{m_d}{q}\right)
\nabla_{\widetilde{x}}L.
$$

实现反向时必须读取前向缓存的掩码，而不是根据输入重新采样一个掩码。否则前向计算的是

$$
\widetilde{x}^{(a)}=\frac{m^{(a)}\odot x}{q},
$$

反向却使用了另一条路径的局部导数

$$
\operatorname{diag}\left(\frac{m_1^{(b)}}{q},\ldots,\frac{m_d^{(b)}}{q}\right),
\qquad
m^{(b)}\ne m^{(a)}.
$$

此时形状可能完全正确，梯度数值却不再是当前执行图的梯度。有限差分检查也必须固定掩码；若正扰动、负扰动和解析梯度各自抽样，差商同时混入参数变化和随机噪声。梯度检查篇已经把这个协议展开为可执行的检查顺序，可参见 [梯度检查](../backpropagation/gradient-checking/)。

## 训练态、评估态和 MC Dropout

对 inverted Dropout，模块至少需要两种明确行为：

| 模式 | 是否抽掩码 | 输出规则 | 需要记录的证据 |
| --- | --- | --- | --- |
| training | 是 | $m\odot x/q$ | 随机种子、掩码形状、非零比例 |
| evaluation | 否 | $x$ | 多次调用输出应相同 |
| MC Dropout | 有意保留 | 多次 $m\odot x/q$ | 采样次数、随机状态、统计汇总 |

调用 no-grad 只表示不记录或不计算梯度，不会自动关闭 Dropout；调用 evaluation 只表示模块走评估分支，也不会替代 no-grad 对计算图的控制。二者是不同层次的状态，和 [自动微分](../backpropagation/autodiff-in-pytorch/)篇讨论的边界相同。

MC Dropout 是一种有意在推理时重复采样掩码、用输出分布估计不确定性的协议。它可以在评估阶段使用训练态 Dropout，但必须由调用方显式选择，并记录采样次数、聚合方式和随机状态。把 MC Dropout 当作普通 evaluation 会得到不稳定输出；把普通 evaluation 误写成 MC Dropout 则会让线上预测产生额外随机性。

训练态的随机性还会影响验证集和早停曲线。验证时若忘记切换 evaluation，指标波动可能只是掩码不同；若使用 MC Dropout，则应报告均值、方差或分位数，而不是把一次采样当作确定性指标。

## 选择丢弃率时先看尺度和任务

$p$ 没有脱离架构、数据量和优化协议的固定最佳值。可以先用以下边界做审计：

- $p=0$ 是关闭 Dropout，不应与“模块没有实现”混为一谈；
- $p$ 越大，保留坐标越少，单个保留坐标的幅值和梯度越大；
- 输入层通常需要较小的丢弃率，因为直接破坏原始特征会改变信息瓶颈；
- 隐藏层可以使用更大的丢弃率，但过大的 $p$ 会让有效路径过稀，训练噪声和方差上升；
- 最后输出头是否使用 Dropout 要看任务和校准要求，不能为了“全网统一”而默认添加；
- 若模型已经欠拟合、激活大面积死亡或数据量很小，继续增大 $p$ 可能把问题推得更远。

增大 batch 主要降低梯度估计的抽样噪声，不能消除每次前向的 Dropout 掩码噪声；增大 batch 也不会自动把多个 micro-batch 的随机前向变成一次共享掩码的前向。[小批量随机梯度下降](../training-nn/minibatch-sgd/)篇的有效 batch 讨论同样适用于这里。

## 掩码的共享轴决定了它到底在丢什么

“Dropout 率为 0.1”还没有描述完整算子。必须写清掩码的形状，以及它沿哪些轴广播：

| 变体 | 掩码共享方式 | 主要边界 |
| --- | --- | --- |
| elementwise | 每个样本、每个坐标独立 | 噪声最细，空间相邻值可能被独立打散 |
| feature/channel | 一个特征或通道在指定轴共享 | 保留整条特征或通道的结构，广播轴错会改变语义 |
| locked / variational | 同一序列的多个时间步共享 | 适合需要固定时间噪声的循环路径，不能当逐步独立采样 |
| attention dropout | 按实现对注意力权重或连接采样 | 要记录 softmax 前后、是否重归一化和 mask 轴 |
| stochastic depth | 整条残差分支或 block 被保留/丢弃 | 丢的是路径，不是每个激活坐标，见下文区别 |

卷积张量若为 $(N,C,H,W)$，elementwise 掩码可以与它同形；channel Dropout 可能只采样 $(N,C,1,1)$ 的掩码，再广播到空间轴。若代码把 $(N,C,H,W)$ 错广播成 $(1,C,1,1)$，不同样本也会共享一次随机选择；这可能是设计，也可能是隐蔽的 batch 间耦合。

循环网络还要区分“每个时间步重新采样”和“整条序列固定一个掩码”。前者的噪声沿时间变化，后者在同一条序列上保持一条固定的稀疏子路径。二者都可以叫 recurrent Dropout，但梯度路径、方差和复现实验结果不同。

## 它和 BatchNorm、残差分支的顺序不能靠猜

Dropout 会改变激活的零值比例和二阶矩，BatchNorm 会从输入统计中估计均值和方差。如果把 Dropout 放在 BatchNorm 之前，训练态的 batch statistics 会直接看到带随机零值的输入；评估态却看到未置零的完整输入，统计分支的差异会被放大。常见的顺序是先完成归一化和激活，再对要正则化的表示使用 Dropout，但具体模型仍要以实际算子顺序为准。

这不是一句“BN 后面一定更好”就能替代的规则。至少要分别记录：

1. BatchNorm 的统计轴和 train/eval 分支；
2. Dropout 的掩码形状和缩放位置；
3. 激活函数是在掩码之前还是之后；
4. 训练、评估和导出图是否保留相同的算子顺序。

残差结构也有相同的边界。若

$$
h_{\mathrm{out}}=h+F(h),
$$

可以只对残差分支使用 Dropout：

$$
h_{\mathrm{out}}
=h+\frac{m\odot F(h)}{q}.
$$

这与随机丢弃整个 $F$ 分支的 stochastic depth 不同，也与把 skip connection 中每个坐标独立置零不同。后两者改变的路径结构和方差都不一样；审计时要看掩码作用在哪一个张量，而不是只看模块名称。

## 随机状态、梯度累积和分布式训练

一次完整更新若由 $A$ 个 micro-batch 构成，通常每个 micro-batch 会经历一次独立的 Dropout 前向。把它们的梯度平均，不能还原成一次使用同一个大 batch、同一组掩码的前向；这和 BatchNorm 的统计不等价问题不同但并列存在。

| 场景 | 应记录什么 | 不能直接推出什么 |
| --- | --- | --- |
| 梯度累积 | 每个 micro-batch 的掩码与梯度 | 等价一次大 batch 随机前向 |
| 多卡训练 | rank、随机状态、掩码生成器 | 所有设备自然使用同一掩码 |
| checkpoint 恢复 | 参数、优化器、数据位置、RNG 状态 | 只恢复参数就能逐步复现 |
| 梯度检查 | 固定输入、掩码、模式 | 随机两次前向的差商仍是局部导数 |

分布式数据并行中，让各 rank 对不同样本使用独立掩码通常是合理的；如果为了复现实验而强制同步掩码，则要记录这个额外约束。随机种子相同也不一定意味着掩码相同：调用顺序、设备生成器、数据加载线程和 fused kernel 都可能消耗不同数量的随机数。

要重放一次训练前向，最可靠的证据是保存实际掩码或足以恢复它的 RNG 状态，而不是只保存一个全局 seed。恢复后应比较某一层的掩码非零比例、激活摘要、loss 和梯度摘要，才能知道是否复现了同一条随机路径。

## 和其他正则化方法的边界

Dropout 是激活上的随机乘法；它不等于以下方法：

| 方法 | 作用位置 | 需要区分的事实 |
| --- | --- | --- |
| Dropout | 激活或连接 | 每次前向采样掩码，评估通常关闭 |
| weight decay / L2 | 参数 | 惩罚参数大小，不直接置零激活 |
| data augmentation | 输入数据 | 改变样本，需检查标签和随机变换 |
| stochastic depth | 残差路径 | 以 block 为单位采样，非逐坐标失活 |
| label smoothing | 目标分布 | 改变监督目标，不改变中间激活掩码 |

这些方法可以组合，但组合后的有效噪声和目标尺度不能从单个方法的说明中直接相加。比如 Dropout 与 weight decay 同时使用时，前者改变每次前向的梯度，后者改变参数更新偏好；应分别记录两条路径。

## 一个标准库数值实验

下面的实验只使用 Python 标准库，固定掩码后检查四件事：inverted Dropout 的期望和方差、线性单元的期望输出、训练态与评估态的尺度，以及反向公式的中心差分。

```python
from itertools import product


def dropout(x, p, mask=None, training=True):
    q = 1.0 - p
    if not 0.0 <= p < 1.0:
        raise ValueError("p must satisfy 0 <= p < 1")
    if not training:
        return list(x)
    if mask is None:
        raise ValueError("the example requires an explicit mask")
    if len(mask) != len(x):
        raise ValueError("mask shape mismatch")
    return [value * keep / q for value, keep in zip(x, mask)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def loss_for_x(x, w, b, target, p, mask):
    z = dot(w, dropout(x, p, mask=mask, training=True)) + b
    return 0.5 * (z - target) ** 2


x = [2.0, -1.0, 4.0]
p = 0.5
q = 1.0 - p
mask = [1, 0, 1]
chosen = dropout(x, p, mask=mask)
all_masks = list(product((0, 1), repeat=len(x)))
samples = [dropout(x, p, mask=list(m)) for m in all_masks]
mean_output = [
    sum(row[i] for row in samples) / len(samples)
    for i in range(len(x))
]
variance = [
    sum((row[i] - mean_output[i]) ** 2 for row in samples) / len(samples)
    for i in range(len(x))
]

w = [0.5, -1.0, 0.25]
b = 0.3
target = 1.0
z_mean = dot(w, x) + b
linear_var = (p / q) * sum(
    (wi * xi) ** 2 for wi, xi in zip(w, x)
)

base_loss = loss_for_x(x, w, b, target, p, mask)
dz = dot(w, dropout(x, p, mask=mask)) + b - target
analytic_dx = [dz * wi * mi / q for wi, mi in zip(w, mask)]
eps = 1e-6
numeric_dx = []
for i in range(len(x)):
    plus = list(x)
    minus = list(x)
    plus[i] += eps
    minus[i] -= eps
    numeric_dx.append(
        (loss_for_x(plus, w, b, target, p, mask)
         - loss_for_x(minus, w, b, target, p, mask))
        / (2 * eps)
    )

print("chosen mask=", mask, "train output=",
      [round(v, 6) for v in chosen])
print("all-mask mean=", [round(v, 6) for v in mean_output],
      "variance=", [round(v, 6) for v in variance])
print("linear E[z]=", f"{z_mean:.6f}",
      "Var[z]=", f"{linear_var:.6f}",
      "eval z=", f"{z_mean:.6f}")
print("fixed-mask loss=", f"{base_loss:.6f}",
      "analytic dx=", [round(v, 9) for v in analytic_dx],
      "numeric dx=", [round(v, 9) for v in numeric_dx],
      "max-error=",
      f"{max(abs(a - n) for a, n in zip(analytic_dx, numeric_dx)):.3e}")
```

运行输出：

```text
chosen mask= [1, 0, 1] train output= [4.0, -0.0, 8.0]
all-mask mean= [2.0, -1.0, 4.0] variance= [4.0, 1.0, 16.0]
linear E[z]= 3.300000 Var[z]= 3.000000 eval z= 3.300000
fixed-mask loss= 5.445000 analytic dx= [3.3, -0.0, 1.65] numeric dx= [3.3, 0.0, 1.65] max-error= 3.724e-10
```

输出中的 -0.0 与 0.0 是同一个数值零，只是 Python 保留了乘法结果的符号。固定掩码后，解析梯度与中心差分的最大误差约为 $3.724\times10^{-10}$；如果把正负扰动之间的掩码改成另一组，差分就不再核对同一个函数。

## 实现审计和失效模式

接入框架模块时，可以把下面的检查写成小单测或一次训练前的断言：

| 检查项 | 应看到的事实 | 常见错误 |
| --- | --- | --- |
| 概率 | $q=1-p$ 且 $q>0$ | 把 keep rate 当 drop rate |
| 训练缩放 | 保留坐标乘 $1/q$ | 训练和评估都缩放 |
| 评估输出 | 同一输入多次调用相同 | eval 仍抽随机掩码 |
| 反向缓存 | 使用本次前向的 $m$ | backward 重新采样 |
| 掩码形状 | 与设计的共享轴一致 | 广播到错误的样本或时间轴 |
| 随机重放 | seed、RNG 或实际掩码可恢复 | 只保存参数 |
| 组合顺序 | BN、激活、残差分支顺序有记录 | 只看模块名称猜语义 |

**评估态仍随机。** 先用同一输入连续调用三次，比较逐元素输出；若不相同，检查模块状态、父模型状态、导出图以及是否有 MC Dropout 调用。

**训练和评估尺度不同。** 取固定输入和固定掩码，检查训练输出的样本均值是否接近输入；再检查评估分支是否直接返回输入。不要在模块外再手动乘 $q$ 或 $1/q$。

**梯度检查每次结果都漂移。** 固定掩码、输入、标签和模式，从同一个参数基准分别计算正扰动和负扰动。若使用自动微分，还要确认前向保存的掩码没有被原地覆盖。

**宽度变了但语义没变。** 先写出张量轴和掩码轴，再检查广播后的掩码实际形状。对卷积和序列尤其要检查样本轴、时间轴、通道轴是否被意外共享。

**把 Dropout 当成万能防过拟合开关。** Dropout 只改变训练时的随机前向和相应梯度；它不能替代数据切分、权重正则化、数据质量检查或正确的评估协议。训练误差上升、验证误差不降时，应先区分模型欠拟合、模式错误和随机指标噪声。

## 运行方法

把上面的代码保存为 dropout_probe.py，在项目环境中运行：

```bash
python3 dropout_probe.py
```

接入真实模块时，先用固定输入检查 $p=0$、一组手工掩码和 evaluation 输出；再检查随机掩码的非零比例、梯度检查、梯度累积、分布式 rank 和 checkpoint 恢复。若使用 fused Dropout kernel，还要把标准库结果与目标设备上的实际输出摘要对照，而不是只验证模块能成功构造。

## 相关词条

- [前向计算](../backpropagation/forward-pass/)：区分训练态与推理态，并把 Dropout 放回完整前向顺序。
- [梯度检查](../backpropagation/gradient-checking/)：固定随机状态后，用有限差分核对 Dropout 路径的局部导数。
- [小批量随机梯度下降](../training-nn/minibatch-sgd/)：解释梯度累积、micro-batch 与一次前向随机性的区别。
- [批量归一化](../training-nn/batch-normalization/)：比较另一种训练/评估分支，并审计 Dropout 与归一化的顺序。
- [自动微分](../backpropagation/autodiff-in-pytorch/)：区分 evaluation、no-grad、inference mode 和实际模块状态。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：把 Dropout 放进训练误差、验证误差和正则化的整体语境。
- [梯度消失与爆炸](../backpropagation/vanishing-and-exploding/)：检查 $1/q$ 放大和稀疏掩码对梯度尺度的影响。
