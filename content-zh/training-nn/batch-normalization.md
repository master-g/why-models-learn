---
title: "批量归一化：用批次统计重设训练尺度"
tags: ["why-models-learn"]
---

批量归一化（Batch Normalization，BatchNorm）在一层激活进入下一层之前，用当前批次的统计量把每个特征或通道重新居中、缩放，再用可学习的 $\gamma$ 和 $\beta$ 恢复有用的尺度与偏移。它不是把整个数据集预处理一次，也不是把训练态的批次统计直接带到评估态；真正容易出错的地方，是统计轴、训练/评估分支、running statistics 和分布式归约必须同时一致。

![批量归一化示意图：左侧把一批激活计算为均值和方差后归一化，再用 gamma、beta 做仿射变换；右侧比较训练态的批次统计与评估态的运行统计](/assets/training-nn/svg/batch-normalization.1.svg)

## 先确定统计轴

先用最简单的矩阵表示一批 dense 层激活：

$$
X=
\begin{bmatrix}
x_{1,1}&\cdots&x_{1,d}\\
\vdots&&\vdots\\
x_{m,1}&\cdots&x_{m,d}
\end{bmatrix}.
$$

$m$ 是 batch 中的样本数，$d$ 是特征数。对第 $k$ 个特征，BatchNorm 把 $x_{1,k},\ldots,x_{m,k}$ 当作一个统计集合；不同特征各自计算均值和方差，不把特征列混成一个总分布。

| 张量形状 | 每个归一化单元 | 统计轴 | $\gamma,\beta$ 的共享方式 |
| --- | --- | --- | --- |
| dense $(N,D)$ | 一个特征 $d$ | $N$ | 每个特征一组 |
| convolution $(N,C,H,W)$ | 一个通道 $c$ | $N,H,W$ | 同一通道的空间位置共享 |
| channels-last $(N,H,W,C)$ | 一个通道 $c$ | $N,H,W$ | 同一通道的空间位置共享 |
| sequence $(N,T,D)$ | 一个特征或通道 | 取决于实现的 $N,T$ 统计约定 | 不能把 padding 当真实 token |

所以“BatchNorm 的均值”不是一个没有下标的全局常数。对卷积层，常见做法是每个通道有一个 $\mu_{B,c}$ 和 $\sigma_{B,c}^2$，把 batch 轴和空间轴一起归约。若代码把特征轴也归约进去，得到的就不再是通常的 BatchNorm。

统计轴还决定了谁会互相影响：dense BatchNorm 中，同一特征的不同样本共享统计量；卷积 BatchNorm 中，同一通道的不同样本和空间位置共享统计量。这个跨样本耦合是它区别于 LayerNorm 的关键。

## 训练态的一次前向

对一个归一化单元，先把该统计集合写成 $x_1,\ldots,x_m$。训练态使用当前 batch 的均值

$$
\mu_B
=\frac{1}{m}\sum_{i=1}^{m}x_i
$$

和通常带偏的 batch 方差

$$
\sigma_B^2
=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2.
$$

加入数值稳定项 $\varepsilon>0$ 后，标准化和仿射恢复分成两步：

$$
\widehat{x}_i
=\frac{x_i-\mu_B}
{\sqrt{\sigma_B^2+\varepsilon}},
\qquad
y_i=\gamma\widehat{x}_i+\beta.
$$

$\widehat{x}$ 是临时的标准化值，$y$ 才是送入下一层的输出。$\gamma$ 和 $\beta$ 通常各对应一个特征或通道，并在所有样本、空间位置或 token 位置上共享。

### 一个四样本的数字例子

令一个特征在 batch 中取

$$
x=(1,2,3,4).
$$

则

$$
\mu_B=2.5,
\qquad
\sigma_B^2=1.25.
$$

若 $\varepsilon=10^{-5}$，标准差为约 $1.118038$，得到

$$
\widehat{x}
\approx
(-1.341635,-0.447212,0.447212,1.341635).
$$

设 $\gamma=1.5,\beta=-0.25$，输出就是

$$
y
\approx
(-2.262453,-0.920818,0.420818,1.762453).
$$

这里的方差分母是 $m$ 而不是 $m-1$。这是训练态常见的 batch 统计定义；某些实现会在写入 running variance 时使用无偏修正，不能看到一个 variance 字段就假设两个阶段的分母相同。

## 归一化、缩放与仿射恢复

这层实际上做了三个不同动作：

1. 减去 batch 均值，消除该统计单元在当前 batch 上的平移；
2. 除以 batch 标准差，把尺度放到相近范围；
3. 乘 $\gamma$、加 $\beta$，让网络可以重新选择有用的尺度和偏移。

如果只做前两步，网络被迫使用固定的零均值、单位方差表示，表达能力会受到限制。$\gamma$ 和 $\beta$ 让归一化可以退化为近似恒等变换：在固定评估统计量下，适当的 $\gamma$ 与 $\beta$ 能把尺度和中心重新调回去。

由于均值和方差来自同一个 batch，标准化值满足

$$
\frac{1}{m}\sum_{i=1}^{m}\widehat{x}_i
=0
$$

以及

$$
\frac{1}{m}\sum_{i=1}^{m}\widehat{x}_i^2
=\frac{\sigma_B^2}{\sigma_B^2+\varepsilon}.
$$

当方差远大于 $\varepsilon$ 时，第二个量接近 $1$；方差很小时，$\varepsilon$ 会有意把缩放压住。于是 $\varepsilon$ 不是可忽略的装饰参数，它决定了近常数输入经过归一化时会被放大多少。

### 对共同平移和缩放的近似不变性

对 $a>0$，把一个 batch 的输入改成

$$
x_i'=a x_i+b
$$

会同时把均值改成 $a\mu_B+b$、方差改成 $a^2\sigma_B^2$。若暂时忽略 $\varepsilon$，有

$$
\frac{x_i'-\mu_B'}
{\sqrt{\sigma_B'^2}}
=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2}}.
$$

因此前一层权重整体变大或变小，可能不会同比改变 BatchNorm 的输出；$\gamma$ 继续控制进入下一层的有效尺度。带 $\varepsilon$ 时这个等式变成近似等式，尤其在方差很小时差异会变大。

这个性质既有用也有代价：优化器看到的参数空间会出现尺度冗余，单纯比较前一层权重的绝对大小不能直接判断表示大小；权重衰减、$\gamma$ 的大小和实际更新范数仍要单独审计。

## 反向传播为什么会耦合同一批样本

设损失对输出的梯度为

$$
\delta_i=\frac{\partial\mathcal L}{\partial y_i}.
$$

先看仿射部分：

$$
\frac{\partial\mathcal L}{\partial\beta}
=\sum_{i=1}^{m}\delta_i,
\qquad
\frac{\partial\mathcal L}{\partial\gamma}
=\sum_{i=1}^{m}\delta_i\widehat{x}_i.
$$

对输入 $x_i$ 的梯度不能只乘一个固定的 $\gamma/\sqrt{\sigma_B^2+\varepsilon}$，因为 $x_i$ 同时影响 $\mu_B$、$\sigma_B^2$ 和自己的标准化值。令

$$
s=\sqrt{\sigma_B^2+\varepsilon},
\qquad
\widehat{x}_i=\frac{x_i-\mu_B}{s}.
$$

沿计算图逐项求导：

$$
\frac{\partial\mathcal L}{\partial\widehat{x}_i}
=\gamma\delta_i,
$$

同时均值和方差分支把所有样本的梯度重新汇合。整理后得到常用的紧凑形式

$$
\frac{\partial\mathcal L}{\partial x_i}
=\frac{\gamma}{m s}
\left(
m\delta_i
-\sum_{j=1}^{m}\delta_j
-\widehat{x}_i
\sum_{j=1}^{m}\delta_j\widehat{x}_j
\right).
$$

式子里的两个求和就是跨样本耦合的来源。一个样本的 $\delta_i$ 改变，会通过 batch 均值和方差影响同一个统计集合中的其他样本；BatchNorm 不是对每个样本独立应用的逐分量函数。

从这个公式还能看到两个约束趋势：

- 输入梯度的总和接近零，平移方向被中心化操作消掉；
- 输入梯度与 $\widehat{x}$ 的相关部分被抵消，整体缩放方向受到抑制。

有 $\varepsilon$ 时这些说法要理解为由实际公式决定的近似几何性质，而不是无条件的精确正交。

## 训练态、评估态和 running statistics

训练时每个 batch 都能计算 $\mu_B,\sigma_B^2$，但评估时不能依赖当前请求里恰好有哪几个样本。常见做法是在训练阶段维护运行统计量：

$$
\mu_{\mathrm{run}}
\leftarrow
\rho\mu_{\mathrm{run}}+(1-\rho)\mu_B,
$$

$$
\sigma_{\mathrm{run}}^2
\leftarrow
\rho\sigma_{\mathrm{run}}^2+(1-\rho)\sigma_B^2.
$$

评估态改用

$$
y
=\gamma
\frac{x-\mu_{\mathrm{run}}}
{\sqrt{\sigma_{\mathrm{run}}^2+\varepsilon}}
+\beta.
$$

$\rho$ 越接近 $1$，运行统计越平滑、记忆越长；但不同库把名为 momentum 的参数定义成 $\rho$ 或 $1-\rho$ 的情况都存在。审计时应看实际更新式，而不是只记录参数名。

### 有偏方差与无偏方差

训练态的 batch 方差常写成

$$
v_{\mathrm{biased}}
=\frac1m\sum_i(x_i-\mu_B)^2.
$$

当 $m>1$，对应的无偏估计为

$$
v_{\mathrm{unbiased}}
=\frac{m}{m-1}v_{\mathrm{biased}}.
$$

如果训练前向使用带偏版本、running buffer 写入却使用无偏版本，那么二者数值不同并不一定是 bug；真正的要求是训练和评估所采用的约定稳定、可追踪。小 batch 时这个修正比例很大，更不能凭一条日志猜测统计量含义。

### 切换模式时要检查四件事

1. 评估前是否真的调用了 eval 模式，而不是只关闭梯度；
2. running mean/variance 是否在训练期间更新、评估期间冻结；
3. checkpoint 是否同时保存了 $\gamma,\beta$ 和运行统计；
4. 微调新域时，是继续使用旧统计、重新估计，还是为域保留独立统计。

只调用 no-grad 不会自动把 BatchNorm 切换到评估态；只调用 eval 也不会阻止你手写的其他 buffer 更新。训练循环要分别记录模式和统计量版本。

## 它如何改变优化，而不等于“消除内部协变量偏移”

BatchNorm 常让激活和反向信号落在更可控的尺度，配合 $\gamma$ 使优化器面对的局部几何更平滑，因此一些网络可以使用更大的学习率或对初始化更宽容。但“训练更稳定”不能直接推出“它精确解决了内部协变量偏移”：

- batch 统计本身会引入随机噪声；
- 前一层参数改变时，$\widehat{x}$ 的关系仍然会改变；
- running statistics 只是指数平均，不是永远正确的总体分布；
- 小 batch、分布式分片和域迁移会改变统计估计的质量。

更稳妥的解释是：BatchNorm 重参数化了中间表示，并把当前 batch 的一阶、二阶尺度显式纳入计算；它可能改善条件数和更新尺度，但效果依赖统计轴、batch 大小、优化器和数据分布。

前一层权重的尺度冗余还会改变权重衰减的意义。若 $W$ 整体缩放后 BN 输出近似不变，L2 权重衰减仍会偏好更小的 $W$；这项正则化不能简单当成“不会影响函数”的无关项。$\gamma$ 也可能被衰减到很小，造成一条通道的有效信号被关闭。

## 小 batch、梯度累积和分布式训练

### 梯度累积不会自动扩大 BatchNorm 的统计 batch

假设一个有效更新由 $A$ 个 micro-batch 构成。梯度累积把它们的梯度加总或平均，但每个 micro-batch 内的 BatchNorm 已经用自己的 $\mu_B,\sigma_B^2$ 产生了前向输出。后面再平均梯度，不能把这 $A$ 次不同的归一化还原成一次更大 batch 的归一化。

因此要区分两件事：

| 机制 | 梯度的有效 batch | BatchNorm 统计的 batch |
| --- | --- | --- |
| 普通梯度累积 | 可以变大 | 仍是每个 micro-batch |
| Ghost BatchNorm | 通常保持小 | 有意切成虚拟小 batch |
| SyncBatchNorm | 可以跨卡统一 | 统计集合跨设备聚合 |
| LayerNorm | 不依赖 batch | 每个样本独立统计 |

当 batch 很小，方差估计会抖动；当某个通道在一个 batch 内近似常数，$\varepsilon$ 直接决定输出尺度。此时可以考虑 SyncBatchNorm、GroupNorm 或 LayerNorm，但替换归一化层会改变统计轴与训练语义，不能只改类名。

### 多卡时要先聚合充分统计量

第 $r$ 张卡有 $m_r$ 个样本。要计算全局均值，应聚合样本和与样本数：

$$
\mu_{\mathrm{global}}
=\frac{\sum_r\sum_{i=1}^{m_r}x_i^{(r)}}
{\sum_r m_r}.
$$

方差可以通过平方和写成

$$
\sigma_{\mathrm{global}}^2
=\frac{\sum_r\sum_i(x_i^{(r)})^2}
{\sum_r m_r}
-\mu_{\mathrm{global}}^2.
$$

直接平均各卡均值只有在每张卡样本数相同且权重正确时才等价；直接平均各卡方差还会漏掉卡间均值差异。实现 SyncBN 时通常 all-reduce 样本数、总和和平方和，或使用等价的稳定合并公式。

### padding 和 mask 不能进入统计量

序列、检测框或可变大小图像中常有 padding。若零填充直接参与均值和方差，padding 比例变化就会改变真实 token 或像素的归一化结果。要么用支持 mask 的统计实现，要么改用不依赖跨样本统计的归一化；不能只在 loss 上 mask 就认为 BatchNorm 已经忽略了 padding。

## 和 LayerNorm、GroupNorm 放在一起比较

| 层 | 统计集合 | 适用边界 |
| --- | --- | --- |
| BatchNorm | batch 与特征/通道 | 大而稳定的 batch、经典 CNN；评估使用 running stats |
| SyncBatchNorm | 跨卡的全局通道统计 | 单卡 batch 小而总 batch 足够大；需要跨卡同步 |
| LayerNorm | 单样本的特征维 | 序列模型、Transformer、小 batch；通常不区分训练和评估 |
| GroupNorm | 单样本内的通道组与空间位置 | 检测、分割和小 batch CNN；不依赖 batch |
| InstanceNorm | 单样本、单通道的空间位置 | 风格迁移等图像任务；是否使用 running stats 需看实现 |

没有一个归一化层在所有任务上都更好。选择依据应包括统计 batch 大小、设备数量、序列 mask、部署模式和是否需要 train/eval 行为一致，而不是把“normalization”当作一个可互换的标签。

## 位置、参数和数值实现的审计表

| 检查项 | 要确认的事实 | 常见错误 |
| --- | --- | --- |
| 位置 | 通常是 affine/conv 后、ReLU 前，或明确采用另一约定 | 训练时用一种顺序、导出时换了顺序 |
| 统计轴 | dense、channels-first、channels-last 的归约轴与实现一致 | 把通道混合，或把 batch 轴漏掉 |
| $\gamma,\beta$ | 形状按 feature/channel 广播，初值通常为 $1,0$ | 参数轴错位或重复广播 |
| $\varepsilon$ | 在目标 dtype 和计算 dtype 下都不会造成除零或过度放大 | fp16 中直接算方差导致 NaN |
| running stats | 只在训练态按明确 momentum 更新，并随 checkpoint 保存 | eval 仍更新，或恢复时重置 |
| 分布式 | 统计量的全局范围与基线相同 | 每卡本地统计却拿来和单卡全局结果比较 |
| mask | padding 和无效位置不进入统计 | 只 mask loss，不 mask normalization |
| 冻结策略 | 明确冻结 $\gamma,\beta$、running stats 或两者 | 以为冻结参数就会冻结 buffer |

对含 BN 的模型做梯度检查时，还要固定 batch 的样本集合和模式。若每次有限差分都重新抽样，数值梯度比较的是两个不同的函数；若在 eval 模式检查，却拿 train 模式的 batch-stat 公式解释，也会得到看似矛盾的结果。

## 一个标准库数值实验

下面的实验只用 Python 标准库，验证三个边界：

1. BatchNorm 对共同平移和正缩放近似不变，但 $\varepsilon$ 会留下可测的微小差异；
2. running mean/variance 是按 batch 顺序更新的状态，不是当前 batch 的即时统计；
3. 反向公式的输入梯度可以用中心差分独立核对。

```python
from math import sqrt


def forward(x, gamma=1.5, beta=-0.25, eps=1e-5):
    mean = sum(x) / len(x)
    var = sum((value - mean) ** 2 for value in x) / len(x)
    inv = 1.0 / sqrt(var + eps)
    xhat = [(value - mean) * inv for value in x]
    y = [gamma * value + beta for value in xhat]
    return mean, var, xhat, y


def backward(x, dy, gamma=1.2, eps=1e-5):
    mean = sum(x) / len(x)
    var = sum((value - mean) ** 2 for value in x) / len(x)
    inv = 1.0 / sqrt(var + eps)
    xhat = [(value - mean) * inv for value in x]
    total_dy = sum(dy)
    total_dyxhat = sum(a * b for a, b in zip(dy, xhat))
    m = len(x)
    dx = [
        gamma * inv / m * (m * d - total_dy - z * total_dyxhat)
        for d, z in zip(dy, xhat)
    ]
    dgamma = sum(d * z for d, z in zip(dy, xhat))
    dbeta = total_dy
    return dx, dgamma, dbeta


x = [1.0, 2.0, 3.0, 4.0]
scaled = [2.0 * value + 10.0 for value in x]
mean, var, xhat, y = forward(x)
scaled_mean, scaled_var, scaled_xhat, _ = forward(scaled)
print(
    "batch",
    "mean=",
    f"{mean:.6f}",
    "var=",
    f"{var:.6f}",
    "std=",
    f"{sqrt(var + 1e-5):.6f}",
    "xhat=",
    [round(v, 6) for v in xhat],
    "y=",
    [round(v, 6) for v in y],
)
print(
    "affine-invariance",
    "mean=",
    f"{scaled_mean:.6f}",
    "var=",
    f"{scaled_var:.6f}",
    "xhat=",
    [round(v, 6) for v in scaled_xhat],
    "max-diff=",
    f"{max(abs(a - b) for a, b in zip(xhat, scaled_xhat)):.12f}",
)

running_mean = 0.0
running_var = 0.0
for batch in ([1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]):
    batch_mean = sum(batch) / len(batch)
    batch_var = sum((value - batch_mean) ** 2 for value in batch) / len(batch)
    running_mean = 0.9 * running_mean + 0.1 * batch_mean
    running_var = 0.9 * running_var + 0.1 * batch_var
print("running", "mean=", f"{running_mean:.6f}", "var=", f"{running_var:.6f}")

dy = [0.5, -1.0, 0.25]
x3 = [1.0, 2.0, 4.0]
dx, dgamma, dbeta = backward(x3, dy)
h = 1e-5
numeric = []
for i in range(len(x3)):
    plus = x3[:]
    minus = x3[:]
    plus[i] += h
    minus[i] -= h
    plus_y = forward(plus, gamma=1.2, beta=0.0)[3]
    minus_y = forward(minus, gamma=1.2, beta=0.0)[3]
    numeric.append(
        sum(d * (a - b) for d, a, b in zip(dy, plus_y, minus_y)) / (2 * h)
    )
print(
    "backward",
    "dx=",
    [round(v, 9) for v in dx],
    "dgamma=",
    f"{dgamma:.9f}",
    "dbeta=",
    f"{dbeta:.9f}",
    "numeric=",
    [round(v, 9) for v in numeric],
    "max-error=",
    f"{max(abs(a - b) for a, b in zip(dx, numeric)):.12e}",
)
```

输出是：

```text
batch mean= 2.500000 var= 1.250000 std= 1.118038 xhat= [-1.341635, -0.447212, 0.447212, 1.341635] y= [-2.262453, -0.920818, 0.420818, 1.762453]
affine-invariance mean= 15.000000 var= 5.000000 xhat= [-1.341639, -0.447213, 0.447213, 1.341639] max-diff= 0.000004024892
running mean= 0.675000 var= 0.237500
backward dx= [0.58415469, -0.876232292, 0.292077602] dgamma= 0.066815096 dbeta= -0.250000000 numeric= [0.58415469, -0.876232292, 0.292077602] max-error= 5.262845714782e-12
```

第一行的 $x$ 与 $2x+10$ 经过归一化后只相差约 $4.0\times10^{-6}$，这正是 $\varepsilon=10^{-5}$ 留下的尺度差异；若没有 $\varepsilon$，正缩放和平移会严格抵消。第二行说明两批数据按 $\rho=0.9$ 更新后，运行均值是 $0.675$、运行方差是 $0.2375$，它们既不是第一批也不是第二批的即时统计。最后一行把反向公式得到的三个 $dx$ 与中心差分比较，最大误差约 $5.3\times10^{-12}$。

这个实验没有证明某个框架的 kernel、并行归约或 dtype 实现都正确；它只把公式层的约定钉住。接入实际模型时，还要固定张量布局、训练/评估模式、统计轴和 running buffer。

## 失效模式：归一化层也会把错误隐藏起来

| 现象 | 可能根因 | 应保留的证据 |
| --- | --- | --- |
| train 指标好，eval 指标突然掉 | running stats 过期、模式切换错误、训练/评估统计定义不同 | 同一 batch 的 batch stats、running stats 和两种输出 |
| 单卡正常，多卡偏移 | 每卡本地统计代替全局统计，或 all-reduce 权重不一致 | 卡数、每卡样本数、全局 sum/square-sum |
| 小 batch 输出抖动 | 方差估计噪声大，或某通道近似常数 | 每通道 batch mean/variance 的分位数 |
| 梯度累积仍与大 batch 不同 | 只累积了梯度，没有合并 BN 的前向统计 | micro-batch 尺寸、统计批次和有效 batch |
| 输出整体偏移或尺度异常 | $\gamma,\beta$ 广播轴错、running mean/variance 错位 | 参数形状、通道索引、前后统计 |
| 首轮或半精度出现 NaN | 方差计算溢出、$\varepsilon$ 太小或低精度归约 | 计算 dtype、raw variance、非有限值位置 |
| padding 比例一变，预测就变 | 无效位置进入均值和方差 | mask、有效样本数、padding 占比 |
| 冻结 BN 后结果仍漂移 | 只冻结了参数，buffer 仍在更新，或 eval 未生效 | requires-grad、buffer 写入日志、模块模式 |
| 某些通道逐渐失活 | $\gamma$ 被衰减到接近零，或前层信号本来就消失 | $\gamma$ 分布、前层激活、梯度范数 |

因此验收不能只看“训练 loss 是否下降”。至少要抽取一批固定输入，在 train/eval 两种模式下分别记录 batch stats、running stats、$\gamma,\beta$、归一化输出和实际更新；再按单卡、多卡、小 batch、恢复 checkpoint 的边界逐一比较。

## 运行方法

把上面的代码保存为一个文件，文件名是：

`batch_norm_probe.py`

在项目环境中运行：

```bash
python3 batch_norm_probe.py
```

接入真实模块时，先用固定的 $(N,C,H,W)$ 或 $(N,D)$ 输入写一个前向单元测试，确认均值、方差、$\gamma,\beta$ 广播和 train/eval 分支；再分别加入梯度累积、AMP、SyncBN、padding mask 和 checkpoint 恢复。每加一个边界，都要与标准库基线或框架 reference 对齐，不能把一条“最终 loss 相近”当成统计语义一致。

## 相关词条

- [权重初始化](../training-nn/weight-initialization/)
- [梯度裁剪](../training-nn/gradient-clipping/)
- [小批次随机梯度下降](../training-nn/minibatch-sgd/)
- [优化器](../training-nn/optimizers/)
- [梯度检查](../backpropagation/gradient-checking/)
- [ReLU](../neurons-and-activations/relu/)
- [tanh 激活](../neurons-and-activations/tanh/)
