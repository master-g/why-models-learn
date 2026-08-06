---
title: "权重初始化：让信号从第一步就能穿过网络"
tags: ["why-models-learn"]
---

权重初始化是在训练开始前为参数赋值的规则。它既要用随机性打破神经元之间的对称，又要把前向激活和反向梯度放在可传播的尺度上。本篇从 $z=Wx+b$ 的方差传播出发，推导 Xavier/Glorot、LeCun 和 He/Kaiming 初始化，再用一个标准库实验比较过小、合适和过大的权重方差，最后把 fan-in、fan-out、卷积、输出头、归一化和 checkpoint 恢复中的边界列出来。

![权重初始化示意图：同一个输入经过仿射层和 ReLU 后，Xavier 的信号逐层衰减，He 初始化保持在可传播尺度，过大的方差逐层放大](/assets/training-nn/svg/weight-initialization.1.svg)

## 初始化先解决两个问题

一层隐藏层通常写成

$$
z_i=\sum_{j=1}^{n_{\mathrm{in}}}w_{ij}x_j+b_i,
\qquad
h_i=f(z_i).
$$

初始化需要同时处理两件不同的事。

第一，如果两个神经元拿到完全相同的权重和偏置，它们在每个样本上都会产生相同的预激活，也会收到相同的梯度。训练再久也不会自动把这两个复制品分开。随机权重负责打破这个对称。

第二，随机不等于合适。若每层都把输入的二阶矩乘以一个远小于 $1$ 的因子，深层激活和梯度会逐层消失；若乘以一个远大于 $1$ 的因子，它们会逐层放大。初始化是把这条信号链放到合适的初始尺度，不是给优化器挑一个学习率。

这里先区分三个量：

- **均值**描述信号是否整体偏向一侧；
- **方差**描述信号围绕均值的离散程度；
- **二阶矩**是 $\mathbb E[x^2]$，等于 $\operatorname{Var}(x)+\mathbb E[x]^2$。

ReLU 输出的均值通常为正，所以只追踪方差会漏掉一个事实：即使方差没有变大，正均值也会在后续仿射层中参与二阶矩传播。推导中用 $q=\mathbb E[x^2]$ 追踪能量，再单独观察均值和活动率。

## 前向方差如何穿过一层

先采用一个能看清机制的近似：

1. 输入分量独立同分布，二阶矩为 $q_x=\mathbb E[x_j^2]$；
2. 权重均值为零，方差为 $\sigma_w^2$；
3. 权重与输入独立；
4. 偏置先取零或与权重独立；
5. 宽度足够大，可以用平均统计量近似某一条具体路径。

由于权重均值为零，预激活的均值先近似为零。展开平方后，不同 $j$ 的交叉项在期望中消失：

$$
\begin{aligned}
\mathbb E[z_i^2]
&=\mathbb E\left[
\left(\sum_{j=1}^{n_{\mathrm{in}}}w_{ij}x_j+b_i\right)^2
\right]\\
&\approx n_{\mathrm{in}}\sigma_w^2q_x+\sigma_b^2.
\end{aligned}
$$

记 $q_z=\mathbb E[z_i^2]$，于是

$$
q_z\approx n_{\mathrm{in}}\sigma_w^2q_x+\sigma_b^2.
$$

若 $b_i=0$，权重和输入的尺度只通过 $n_{\mathrm{in}}\sigma_w^2$ 相乘。这个乘数就是初始化最先要控制的对象。

### 线性层

若 $f(z)=z$，那么 $q_h=q_z$。当偏置为零时，

$$
\frac{q_h}{q_x}
\approx n_{\mathrm{in}}\sigma_w^2.
$$

例如 $n_{\mathrm{in}}=64$ 且 $\sigma_w=0.01$，这个比例只有

$$
64\times0.01^2=0.0064.
$$

经过 12 层后，理想化的二阶矩比例是 $0.0064^{12}$，已经小到无法作为可用信号。这里还没有把激活导数算进反向传播。

### 对称输入上的 ReLU

对称的零均值 $z$ 经过 ReLU 后，正半轴大约保留一半概率。若 $z$ 的分布近似对称，

$$
\mathbb E[\operatorname{ReLU}(z)^2]
\approx \frac12\mathbb E[z^2].
$$

于是

$$
q_h
\approx \frac12 n_{\mathrm{in}}\sigma_w^2q_x.
$$

想让每层的二阶矩大致不变，就令

$$
\frac12 n_{\mathrm{in}}\sigma_w^2\approx1,
\qquad
\sigma_w^2\approx\frac{2}{n_{\mathrm{in}}}.
$$

这就是 He/Kaiming 初始化针对 ReLU 的核心比例。它保持的是二阶矩的近似，不是保证每个 batch、每个通道和每一步都完全相同。

### tanh 与 sigmoid

tanh 在零点附近满足 $\tanh(z)\approx z$，所以小信号区域可以先用线性近似；但输入幅值变大后，输出进入饱和区，局部导数变小。sigmoid 还会把输出均值推到正区间，在没有中心化措施时更容易把下一层的预激活推离零点。

因此，对 tanh 常用的目标是让初始预激活集中在近似线性的区域，而不是像 ReLU 那样只补偿一半正半轴。Xavier/Glorot 的 fan-in 与 fan-out 折中正是为这种双向传播目标设计的；它不是“所有激活函数的默认值”。

## 反向梯度也有自己的方差乘数

设 $\delta_i^\ell=\partial\mathcal L/\partial z_i^\ell$ 是第 $\ell$ 层预激活的梯度。对下一层的仿射变换反向展开：

$$
\delta_j^\ell
=f'(z_j^\ell)
\sum_{i=1}^{n_{\mathrm{out}}}
w_{ij}^{\ell+1}\delta_i^{\ell+1}.
$$

在独立性和零均值近似下，若 $q_\delta^\ell=\mathbb E[(\delta_j^\ell)^2]$，则

$$
q_\delta^\ell
\approx
n_{\mathrm{out}}\sigma_w^2
\mathbb E[f'(z^\ell)^2]\,
q_\delta^{\ell+1}.
$$

对对称输入的 ReLU,$f'(z)$ 以约一半概率为 $1$，所以

$$
\mathbb E[f'(z)^2]\approx\frac12.
$$

前向和反向的稳定条件分别带着 $n_{\mathrm{in}}$ 与 $n_{\mathrm{out}}$：

$$
\text{前向乘数}
\approx\frac12n_{\mathrm{in}}\sigma_w^2,
\qquad
\text{反向乘数}
\approx\frac12n_{\mathrm{out}}\sigma_w^2.
$$

当输入宽度和输出宽度不同，一个只用 fan-in 的选择可能保护前向却放大反向，反过来也一样。这个冲突解释了为什么 Xavier 使用两个 fan 的折中，而 He 的 `fan_in`/`fan_out` 是一个需要明确选择的实现参数。

对 leaky ReLU,

$$
f_\alpha(z)=
\begin{cases}
z,&z\geq0,\\
\alpha z,&z<0,
\end{cases}
\qquad
\mathbb E[f_\alpha'(z)^2]
\approx\frac{1+\alpha^2}{2}.
$$

按 fan-in 保护前向二阶矩时，得到

$$
\sigma_w^2
\approx
\frac{2}{(1+\alpha^2)n_{\mathrm{in}}}.
$$

当 $\alpha=0$ 时回到 ReLU 的 $2/n_{\mathrm{in}}$；当 $\alpha=1$ 时退化为线性层的 $1/n_{\mathrm{in}}$。

## Xavier、LeCun 与 He 的公式

设权重矩阵的形状为 $(n_{\mathrm{out}},n_{\mathrm{in}})$。常见公式如下：

| 初始化 | 正态分布的方差 | 均匀分布的采样区间 | 主要假设 |
| --- | ---: | ---: | --- |
| LeCun | $1/n_{\mathrm{in}}$ | $[-\sqrt{3/n_{\mathrm{in}}},\sqrt{3/n_{\mathrm{in}}}]$ | 以输入侧尺度为基准 |
| Xavier/Glorot | $2/(n_{\mathrm{in}}+n_{\mathrm{out}})$ | $[-\sqrt{6/(n_{\mathrm{in}}+n_{\mathrm{out}})},\sqrt{6/(n_{\mathrm{in}}+n_{\mathrm{out}})}]$ | 折中前向与反向 |
| He/Kaiming | $2/n_{\mathrm{in}}$ | $[-\sqrt{6/n_{\mathrm{in}}},\sqrt{6/n_{\mathrm{in}}}]$ | ReLU 的一半活动率 |

表中的均匀区间来自

$$
\operatorname{Var}(U[-a,a])=\frac{a^2}{3}.
$$

所以若目标正态方差为 $\sigma^2$，均匀分布的半宽就是 $a=\sqrt{3}\sigma$。分布形状改变了尾部和最大初值，但不能取代方差目标。

### Xavier/Glorot 为什么用两个 fan

如果只保护前向，可以取 $\sigma_w^2\approx1/n_{\mathrm{in}}$；如果只保护反向，可以取 $\sigma_w^2\approx1/n_{\mathrm{out}}$。Xavier 取两者的调和式：

$$
\sigma_w^2=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}.
$$

当 $n_{\mathrm{in}}=n_{\mathrm{out}}=n$ 时，它变成 $1/n$。对接近线性的对称激活，这个尺度让两侧的乘数都在 $1$ 附近；对 ReLU,由于少了 $\frac12$ 的门控补偿，会出现逐层衰减。

### He/Kaiming 为什么只看一侧

ReLU 把负侧输出置零，需要把权重方差提高到约 $2/n_{\mathrm{in}}$ 才补回一半活动率。正态版本为

$$
w_{ij}\sim\mathcal N\left(0,\frac{2}{n_{\mathrm{in}}}\right),
$$

均匀版本为

$$
w_{ij}\sim
U\left(
-\sqrt{\frac{6}{n_{\mathrm{in}}}},
\sqrt{\frac{6}{n_{\mathrm{in}}}}
\right).
$$

若代码把 ReLU 初始化的 `nonlinearity`、negative slope 或 `mode` 写错，公式本身正确也会得到错误尺度。至少记录以下配置：

| 字段 | 必须说明 |
| --- | --- |
| activation | 后面实际使用的激活和 negative slope |
| mode | 按 fan-in 还是 fan-out 保持尺度 |
| distribution | normal、uniform 还是截断分布 |
| fan convention | 对矩阵、卷积、转置卷积如何计数 |
| bias | 偏置是否为零、是否按先验设置 |

## 随机性为什么能打破对称

考虑一个隐藏层的两个神经元，若

$$
w_{1j}=w_{2j},
\qquad
b_1=b_2,
$$

那么对所有输入都有 $z_1=z_2$ 和 $h_1=h_2$。反向传播时两者的局部梯度也相同，于是一次更新仍然满足

$$
\Delta w_{1j}=\Delta w_{2j},
\qquad
\Delta b_1=\Delta b_2.
$$

这是一条不变的对称子空间。把所有权重都设为零不只是“初始信号小”，而是把网络放在了这条子空间上。

随机偏置不能替代随机权重。它有时能让神经元的输出不同，但权重列仍可能收到高度相似的信号；而且对深层矩阵，不同层的零权重会让路径结构消失。通常做法是权重用零均值随机分布，偏置设为零或按输出先验设置。

随机初始化也不能太大。设两个方案都能打破对称：

- $\sigma_w=10^{-6}$ 时，神经元确实不同，但所有激活和梯度几乎为零；
- $\sigma_w=10$ 时，神经元不同得很快，但一次前向就可能让 logits、损失或梯度溢出。

需要同时检查“不同”与“可传播”，不能把其中一个当成另一个的证明。

## 一个可复现的深度 ReLU 实验

下面的代码只依赖 Python 标准库。它构造 256 个宽度为 64 的输入，连续经过 12 个宽度为 64 的线性层和 ReLU,逐层记录激活二阶矩 $q=\operatorname{mean}(h^2)$。每个方案都用固定种子，但每层仍会抽取独立权重。

```python
import math
import random

def mean(values):
    values = list(values)
    return sum(values) / len(values)

def simulate(scheme, depth=12, width=64, samples=256):
    rng = random.Random(7)
    x = [[rng.gauss(0.0, 1.0) for _ in range(width)] for _ in range(samples)]
    q = []
    active = []
    for _ in range(depth):
        if scheme == "tiny":
            sigma = 0.01
        elif scheme == "xavier":
            sigma = math.sqrt(1.0 / width)
        elif scheme == "he":
            sigma = math.sqrt(2.0 / width)
        else:
            sigma = 0.20
        weights = [[rng.gauss(0.0, sigma) for _ in range(width)] for _ in range(width)]
        z = []
        for row in x:
            z.append([
                sum(weights[i][j] * row[j] for j in range(width))
                for i in range(width)
            ])
        x = [[max(0.0, value) for value in row] for row in z]
        q.append(mean(value * value for row in x for value in row))
        active.append(mean(
            value > 0.0 for row in z for value in row
        ))
    return q, active

for scheme in ("tiny", "xavier", "he", "large"):
    q, active = simulate(scheme)
    print(
        scheme,
        "q1/q6/q12",
        *(f"{q[i]:.6f}" for i in (0, 5, 11)),
        "active12",
        f"{active[-1]:.6f}",
    )
```

运行输出为：

```
tiny q1/q6/q12 0.003237 0.000000 0.000000 active12 0.442749
xavier q1/q6/q12 0.505765 0.009594 0.000057 active12 0.442749
he q1/q6/q12 1.011531 0.613987 0.231469 active12 0.442749
large q1/q6/q12 1.294759 2.700344 4.477267 active12 0.442749
```

有限宽度和有限样本会让 He 的二阶矩从 $1.01$ 漂到第 12 层的 $0.23$，所以“保持”不能理解为每一层都精确等于 $1$。它仍明显比 Xavier 的 $5.7\times10^{-5}$ 更能保留信号。过大方差的方案在第 12 层已经达到 $4.48$，如果再叠加更深的网络、非零偏置或不稳定的损失，梯度会更快进入异常区。

活动率在这组随机实验中约为 $0.443$，接近但不等于理想化的 $0.5$。它是另一个应记录的信号：二阶矩看起来正常时，如果活动率已经接近零，仍可能存在死亡 ReLU；如果活动率几乎全为一，则输入分布或偏置可能已经把层推离零点。

## fan-in、fan-out 与非全连接层

`fan_in` 是一个输出单元接收的独立输入数,`fan_out` 是一个输入单元把信号送出的输出数。对形状 $(n_{\mathrm{out}},n_{\mathrm{in}})$ 的普通线性层，两者就是两个维度；但对卷积层不能直接把 kernel 的四个维度随便相加。

对二维卷积，常见计数为

$$
\operatorname{fan}_{\mathrm{in}}
=C_{\mathrm{in}}K_hK_w,
\qquad
\operatorname{fan}_{\mathrm{out}}
=C_{\mathrm{out}}K_hK_w.
$$

分组卷积把输入和输出通道分到独立组，每个输出通道实际看到的是

$$
\operatorname{fan}_{\mathrm{in}}
=\frac{C_{\mathrm{in}}}{G}K_hK_w,
\qquad
\operatorname{fan}_{\mathrm{out}}
=\frac{C_{\mathrm{out}}}{G}K_hK_w.
$$

如果忽略 $G$，初始化方差会比实际连接密度低或高一个组数因子。深度可分离卷积还要分开看 depthwise 和 pointwise 两个算子，不能把整块参数量当作一个 fan。

转置卷积的权重张量布局可能与普通卷积不同。安全做法不是背某个库的维度顺序，而是问一句：一个输出位置实际累加了多少个独立权重？用这个数作为 fan-in,并用一个小输入测量首个 batch 的预激活二阶矩。

Embedding、分类头和 attention 投影也需要自己的口径：

- embedding 的每个向量通常由词表索引直接取出，其 fan 不等同于词表大小；
- Q/K/V 投影的矩阵按输入维度和输出维度计数，注意力点积还另有 $1/\sqrt{d_k}$ 的缩放；
- softmax 分类头的零权重会给出均匀 logits,这可能是合理起点，但不能把它推广到隐藏层所有权重；
- 回归头通常使用恒等输出，需要关注初始输出尺度而不是 ReLU 的一半活动率。

## 偏置、输出头与架构例外

### 偏置通常设零，但输出先验可以例外

隐藏层偏置设为零有两个好处：不会在第一步把预激活整体推到正侧或负侧，也不额外增加 $q_z$ 中的 $\sigma_b^2$。权重已经提供随机性时，不需要靠随机偏置打破神经元对称。

分类输出头可以根据训练集先验设偏置。若正类比例约为 $p$，希望初始 sigmoid 概率接近 $p$，可以取

$$
b_0=\log\frac{p}{1-p}.
$$

这不会改变隐藏层的初始化公式，但会改变第一批 loss 和梯度的尺度。报告实验时应把这个先验偏置和零偏置区分开。

### 归一化层改变了观测方式

BatchNorm、LayerNorm 或 RMSNorm 会重新缩放部分激活，因此仅凭归一化后的方差不能判断仿射层初始化是否正确。仍应在归一化前记录 $z$ 的均值、二阶矩和分位数；对残差分支还要同时记录分支输出相对于恒等路径的尺度。

归一化降低了对初始尺度的敏感性，但不能修复：

- 全零权重造成的对称；
- fan 计数错误造成的异常梯度；
- 偏置把所有预激活推到 ReLU 负侧；
- checkpoint 恢复时重新初始化了某个子模块。

### 残差块可以选择小分支或零末层

残差块的输出是

$$
h_{\mathrm{out}}=h_{\mathrm{in}}+F(h_{\mathrm{in}}).
$$

若 $F$ 与恒等路径一开始同样大，块的输出二阶矩可能翻倍；若把整个 $F$ 初始化成零，又会让分支内部的参数暂时收不到有效梯度。常见折中是让残差分支的末层较小或用可学习门从零开始，同时保留前面层的随机性。具体做法取决于架构，不能把“最后一层置零”误写成“整个网络置零”。

## 失效模式：初始化对了仍可能出问题

| 现象 | 可能原因 | 先记录什么 |
| --- | --- | --- |
| 所有隐藏单元输出一样 | 权重全零、复制了同一随机张量、加载错误 checkpoint | 每层权重的不同坐标数、行间相关性 |
| 激活二阶矩逐层趋零 | 权重方差太小、Xavier 用在深 ReLU、fan-in 计算过大 | 每层 $q_z$、$q_h$、活动率 |
| 激活或梯度逐层放大 | 方差过大、fan-in 计算过小、非零偏置叠加 | 每层 p99、最大值、梯度范数 |
| ReLU 活动率接近零 | 偏置太负、输入中心改变、更新第一步过大 | 预激活均值和正值比例 |
| 训练 loss 一开始就是 NaN | logits/exp 溢出、混合精度下初始范围过大、损失实现不稳定 | 首个 batch 的 logits 与 loss |
| 换 batch 或宽度后行为变化 | 使用固定绝对标准差，没有按 fan 缩放 | fan、权重样本方差和首层 q |
| 恢复训练后曲线突变 | 重新初始化、漏加载参数或 optimizer state | 参数 checksum、初始化 seed、checkpoint key |
| 归一化后看似正常但梯度异常 | 只看 norm 输出，没看归一化前的仿射层 | norm 前后统计与逐层梯度 |

尤其要把初始化问题和学习率问题分开做实验。固定初始化而改变 $\eta$ 可以测优化器稳定性；固定 $\eta$ 而改变初始化可以测信号传播。若同时改变两者，一次 loss 曲线无法告诉你是哪一层出了问题。

## 一套可执行的初始化审计

新建模型时，可以按下面的顺序做首批检查：

1. 写下每个可学习张量的形状、实际连接数和选择的 `fan_in`/`fan_out`；
2. 固定随机种子，初始化一次，记录权重均值、标准差、最小值、最大值和非零比例；
3. 用一小批固定输入跑一次前向，记录每个仿射层的 $z$ 均值、二阶矩、p1/p50/p99 和 ReLU 活动率；
4. 用固定标量 loss 反向一次，记录每层 $\lVert\delta\rVert_2$ 和参数梯度范数；
5. 检查输出头的初始概率、logits 或回归输出是否符合任务先验；
6. 保存初始化后的 checksum,恢复 checkpoint 后再次比较，确认没有静默重初始化。

这些记录应该出现在训练日志中，而不是只在初始化函数里打印一次。初始化的目标是让第一步可工作；真正的运行还要观察更新后统计是否迅速离开这个尺度。

## 运行方法

把上面的代码保存为 `weight_init_probe.py`，直接运行：

```bash
python3 weight_init_probe.py
```

如果把实验改成自己的层，至少保留四项输出：每层预激活二阶矩、激活二阶矩、活动率和反向梯度范数。不要只打印最终 loss；最终 loss 可能已经把中间层的饱和、裁剪或 NaN 传播掩盖掉。

## 相关词条

- [ReLU](../neurons-and-activations/relu/)
- [tanh 激活](../neurons-and-activations/tanh/)
- [梯度消失与爆炸](../backpropagation/vanishing-and-exploding/)
- [批归一化](../training-nn/batch-normalization/)
- [梯度裁剪](../training-nn/gradient-clipping/)
- [学习率调度](../training-nn/learning-rate-schedules/)
- [优化器](../training-nn/optimizers/)
