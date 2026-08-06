---
title: "回归损失：从点预测到不确定性建模"
tags: ["why-models-learn"]
---

回归损失决定模型把连续目标理解成什么：平方损失把目标拉向条件均值，绝对损失把目标拉向条件中位数，分位数损失把高估和低估的代价做成不对称，概率回归损失则要求模型同时说明预测中心和不确定性。它们不是回归评估指标的不同叫法，而是训练时改变梯度、统计假设、异常点影响和预测含义的不同目标。

![回归损失把残差翻译成不同的统计假设：左侧比较平方、绝对、Huber 和 log-cosh，右侧展示不同分位数如何倾斜地惩罚高估与低估](/assets/training-nn/svg/regression-losses.1.svg)

## 先固定残差和输出对象

设输入为 $\boldsymbol x$，连续目标为 $y$，模型输出为点预测

$$
\widehat y=f_{\boldsymbol\theta}(\boldsymbol x).
$$

本文统一把残差定义为预测减真实值：

$$
r=\widehat y-y.
$$

$r>0$ 表示高估，$r<0$ 表示低估。这个符号约定尤其重要，因为分位数损失的正负两侧斜率并不对称；如果把残差换成 $y-\widehat y$ 却不同时换公式，最终学到的分位数方向会反过来。

逐样本回归损失写为

$$
\ell(\widehat y,y)=\ell(r).
$$

一批样本的训练目标还要指定归约：

$$
J(\boldsymbol\theta)
=\frac1n\sum_{i=1}^{n}
\ell\bigl(\widehat y_i,y_i\bigr).
$$

因此“选 MSE 还是 MAE”至少包含三层选择：单个残差如何计价，模型输出是点还是分布，以及一批样本如何加权。[损失函数总览](../training-nn/loss-functions/) 已经说明了目标、指标和部署代价的边界；本文只深入回归目标本身。

### 损失和指标的分工

MAE、MSE、RMSE 等名字既可以出现在训练损失里，也可以出现在评估报表里，但使用位置不同：

| 使用位置 | 需要回答的问题 | 额外约束 |
| --- | --- | --- |
| 训练损失 | 参数本轮应沿什么方向移动 | 梯度尺度、数值稳定、可归约 |
| 验证指标 | 冻结模型在样本上错得怎样 | 与历史版本、业务分组一致 |
| 部署代价 | 一次高估或低估会造成什么后果 | 阈值、库存、人工和风险规则 |

训练 MSE 后报告 MAE 不矛盾；它们只是在训练和报告时强调了不同的错误形状。真正危险的是只看其中一个数字，忘记说明单位、归约和样本权重。

## 平方损失：条件均值与高斯噪声

### 从残差斜率开始

半平方损失为

$$
\ell_{\mathrm{sq}}(r)=\frac12r^2.
$$

它的一阶和二阶导数是

$$
\frac{\mathrm d\ell_{\mathrm{sq}}}{\mathrm dr}=r,
\qquad
\frac{\mathrm d^2\ell_{\mathrm{sq}}}{\mathrm dr^2}=1.
$$

如果批量 MSE 不带 $\frac12$，则

$$
\operatorname{MSE}
=\frac1n\sum_{i=1}^{n}r_i^2,
\qquad
\frac{\partial\operatorname{MSE}}
{\partial\widehat y_i}
=\frac{2r_i}{n}.
$$

残差变成两倍时，半平方损失变成四倍，梯度变成两倍。这个远处越来越陡的斜率会让大错获得更强的修正信号，也会让一个离群标签主导一整个 batch 的更新。

### 为什么平方损失得到条件均值

固定输入 $X=\boldsymbol x$，令

$$
\mu(\boldsymbol x)
=\mathbb E[Y\mid X=\boldsymbol x].
$$

预测 $a$ 的条件平方风险可以分解为

$$
\begin{aligned}
\mathbb E[(a-Y)^2\mid X=\boldsymbol x]
&=\mathbb E[
((a-\mu)+(\mu-Y))^2
\mid X=\boldsymbol x]\\
&=(a-\mu)^2
+2(a-\mu)\mathbb E[\mu-Y\mid X=\boldsymbol x]\\
&\quad+\mathbb E[(\mu-Y)^2\mid X=\boldsymbol x]\\
&=(a-\mu)^2+\operatorname{Var}(Y\mid X=\boldsymbol x).
\end{aligned}
$$

中间项为零，最后一项不依赖 $a$，所以最小点是

$$
a^\star(\boldsymbol x)=\mu(\boldsymbol x).
$$

这并不意味着条件均值一定是一个常见的实际结果。若同一个输入对应两个相距很远的结果，均值可能落在两个峰之间；平方损失仍然会把点预测推向那个平均位置。

### 平方损失等价于固定方差的 Gaussian NLL

假设目标在给定输入和均值 $\mu$ 时服从方差固定的高斯分布：

$$
p(y\mid\mu,\sigma^2)
=\frac1{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(y-\mu)^2}{2\sigma^2}
\right).
$$

负对数似然为

$$
\ell_{\mathrm{Gaussian}}
=\frac12
\left(
\frac{(y-\mu)^2}{\sigma^2}
+\log\sigma^2
+\log(2\pi)
\right).
$$

若 $\sigma^2$ 固定，后两项是常数，第一项只是平方损失乘上固定权重。因此使用 MSE 隐含了一个噪声模型：大致相信残差是对称、轻尾并且共享一个尺度。这个假设可能很合适，也可能只是因为公式方便。

若把 $\sigma^2$ 也交给模型预测，损失就不再只是 MSE；下一节会说明为什么必须同时检查方差头的稳定性和校准。

## 绝对损失：条件中位数与 Laplace 噪声

绝对损失为

$$
\ell_{\mathrm{abs}}(r)=\lvert r\rvert.
$$

在 $r\neq0$ 时导数为

$$
\frac{\mathrm d\ell_{\mathrm{abs}}}{\mathrm dr}
=\operatorname{sign}(r).
$$

在 $r=0$ 处它没有唯一导数，常用次梯度集合 $[-1,1]$ 表示可接受的局部方向。一个实现可能选零，也可能由自动微分规则选择某个固定次梯度；检查梯度时不能把折点当作普通光滑点。

固定输入后，绝对风险的最小点是条件中位数。直观地说，把预测向右移动一点，会让所有预测在左侧的样本变近、右侧的样本变远；当两侧概率各达到一半时，左右拉力平衡。

绝对损失也有概率解释。若残差服从尺度为 $b$ 的 Laplace 分布：

$$
p(y\mid\mu,b)
=\frac1{2b}
\exp\left(-\frac{\lvert y-\mu\rvert}{b}\right),
$$

其负对数似然为

$$
\ell_{\mathrm{Laplace}}
=\frac{\lvert y-\mu\rvert}{b}
+\log(2b).
$$

固定 $b$ 后，最小化它等价于最小化绝对误差。与 Gaussian 相比，Laplace 分布的尾部更重，绝对损失不会让一个大残差的梯度无限增长，所以更能抵抗少量异常点。

### 一个均值和中位数分开的例子

考虑同一输入下的四个结果

$$
y\in\{0,0,0,10\}.
$$

平方风险在 $a=0$ 和 $a=2.5$ 处分别是

$$
\frac{0^2+0^2+0^2+10^2}{4}=25,
\qquad
\frac{2.5^2+2.5^2+2.5^2+7.5^2}{4}=18.75.
$$

所以平方损失更偏向 $2.5$。绝对风险则分别是

$$
\frac{0+0+0+10}{4}=2.5,
\qquad
\frac{2.5+2.5+2.5+7.5}{4}=3.75.
$$

所以绝对损失偏向 $0$。这不是优化器选得不同，而是两个损失在统计上定义了不同的“代表性预测”。

## Huber、pseudo-Huber 和 log-cosh

平方和绝对损失之间存在一整条稳健性折中路线。它们的区别不只体现在图像外观，还体现在零点的曲率、远处的影响函数和是否有明确的噪声解释。

### Huber：近处平滑，远处截坡

给定阈值 $\delta>0$，Huber 损失为

$$
\ell_{\delta}(r)
=
\begin{cases}
\frac12r^2,&\lvert r\rvert\leq\delta,\\
\delta\left(\lvert r\rvert-\frac12\delta\right),&\lvert r\rvert>\delta.
\end{cases}
$$

导数为

$$
\frac{\mathrm d\ell_{\delta}}{\mathrm dr}
=
\begin{cases}
r,&\lvert r\rvert\leq\delta,\\
\delta\operatorname{sign}(r),&\lvert r\rvert>\delta.
\end{cases}
$$

在两个连接点，函数值和一阶导数都连续；二阶导数从 $1$ 变成 $0$。因此 Huber 在小误差附近保留平方损失的局部曲率，在大误差处把梯度限制在 $\pm\delta$。它仍然计入异常点，只是不让异常点的影响随着残差平方增长。

$\delta$ 必须和残差的单位绑定。把目标从秒换成毫秒而不重新标定 $\delta$，会让几乎所有样本从平方区跳到线性区。常见做法是先用训练集尺度标准化目标，再在验证集上选择阈值；不能用测试残差反推一个看起来最好的 $\delta$。

### pseudo-Huber：用光滑曲线替代折点

pseudo-Huber 损失为

$$
\ell_{\delta}^{\mathrm{PH}}(r)
=\delta^2
\left(
\sqrt{1+\left(\frac r\delta\right)^2}-1
\right).
$$

它的导数是

$$
\frac{\mathrm d\ell_{\delta}^{\mathrm{PH}}}{\mathrm dr}
=\frac{r}
{\sqrt{1+(r/\delta)^2}}.
$$

当 $\lvert r\rvert\ll\delta$ 时，它近似 $\frac12r^2$；当 $\lvert r\rvert\gg\delta$ 时，导数趋近 $\operatorname{sign}(r)$。它没有 Huber 的二阶折点，适合需要连续曲率的优化或二阶近似，但仍然要校准 $\delta$。

### log-cosh：导数自然饱和

log-cosh 损失定义为

$$
\ell_{\mathrm{LC}}(r)=\log\cosh(r).
$$

因为

$$
\frac{\mathrm d\ell_{\mathrm{LC}}}{\mathrm dr}
=\tanh(r),
$$

它在零附近满足

$$
\log\cosh(r)
=\frac12r^2+O(r^4),
$$

在远处近似 $\lvert r\rvert-\log2$。因此它和 pseudo-Huber 一样让远处梯度饱和，但没有显式的阈值。直接计算 $\cosh(r)$ 在大残差时可能溢出，稳定形式是

$$
\log\cosh(r)
=\lvert r\rvert
+\log\left(1+\exp(-2\lvert r\rvert)\right)
-\log2.
$$

### 把四种点损失放在一张表里

| 损失 | 零点附近 | 远处斜率 | 典型统计含义 |
| --- | --- | --- | --- |
| 半平方 | 曲率固定 | 随残差增长 | 条件均值、轻尾噪声 |
| 绝对值 | 有折点 | 固定为正负一 | 条件中位数、较重尾噪声 |
| Huber | 平方且有阈值 | 截到正负 delta | 平滑稳健点预测 |
| pseudo-Huber | 平滑平方 | 连续地趋向饱和 | 需要连续曲率的稳健目标 |
| log-cosh | 平滑平方近似 | tanh 饱和 | 无显式折点的稳健目标 |

这张表不能单独决定选择。异常值是否是真实尾部、是否需要条件分位数、目标是否有严格正值约束、部署是否需要概率区间，都会改变合适的损失。

## 分位数损失：直接学习偏斜分布

### 正确写法取决于残差符号

本文的残差是 $r=\widehat y-y$。要学习条件分布的 $\tau$ 分位数，pinball 损失写成

$$
\rho_\tau(r)
=
\begin{cases}
(1-\tau)r,&r\geq0,\\
-\tau r,&r<0,
\end{cases}
\qquad 0<\tau<1.
$$

当模型高估时 $r>0$，斜率是 $1-\tau$；当模型低估时 $r<0$，斜率是 $-\tau$。因此 $\tau=0.9$ 会对低估施加 $0.9$ 的负侧斜率、只给高估 $0.1$ 的正侧斜率，模型会被推向条件分布的 90% 分位数。若采用相反残差 $u=y-\widehat y$，同一个损失会写成 $u\geq0$ 时 $\tau u$，两种写法完全等价，但不能混用。

### 为什么最小点是分位数

令 $F(a)=\mathbb P(Y\leq a\mid X=\boldsymbol x)$。在没有恰好落在 $a$ 上的离散质量时，条件 pinball 风险的导数为

$$
\frac{\mathrm d}{\mathrm da}
\mathbb E[\rho_\tau(a-Y)\mid\boldsymbol x]
= (1-\tau)F(a)-\tau(1-F(a)).
$$

令导数为零得到

$$
F(a)=\tau.
$$

所以损失的系数不是装饰：它把“低估和高估哪一个更贵”直接变成了目标分位数。

### 多个分位数和预测区间

可以让一个模型同时输出

$$
\widehat q_{\tau_1}(\boldsymbol x),
\ldots,
\widehat q_{\tau_K}(\boldsymbol x),
\qquad
\tau_1<\cdots<\tau_K,
$$

再对每个分位数求 pinball 损失的加权和。独立输出可能产生分位数交叉：

$$
\widehat q_{\tau_j}(\boldsymbol x)
>
\widehat q_{\tau_{j+1}}(\boldsymbol x),
$$

这在概率意义上不合理。可以用单调参数化、排序后的输出或交叉惩罚修复，但修复项也会改变训练目标，必须单独报告。

用下分位数和上分位数构成区间时，区间宽度和覆盖率是两个不同问题。区间很宽可以覆盖几乎所有目标，却没有决策价值；区间很窄可能看起来精确，却漏掉大量真实值。训练 pinball 只提供分位数方向，最终仍要在独立验证集上检查覆盖率、宽度和不同分组的稳定性。

## 概率回归：均值、方差和 NLL

点预测只输出 $\widehat y$，概率回归输出一个条件分布。最常见的 Gaussian 头让模型预测均值 $\mu(\boldsymbol x)$ 和对数方差

$$
s(\boldsymbol x)=\log\sigma^2(\boldsymbol x).
$$

对单个样本，Gaussian NLL 可以稳定地写成

$$
\ell_{\mathrm{NLL}}(\mu,s;y)
=\frac12
\left(
s+(y-\mu)^2\exp(-s)+\log(2\pi)
\right).
$$

对均值和 log-variance 的梯度分别是

$$
\frac{\partial\ell_{\mathrm{NLL}}}{\partial\mu}
=(\mu-y)\exp(-s),
$$

$$
\frac{\partial\ell_{\mathrm{NLL}}}{\partial s}
=\frac12
\left(
1-(y-\mu)^2\exp(-s)
\right).
$$

残差大时，增加方差可以降低标准化残差项，但也会增加 $s$ 本身；当预测的方差与残差尺度平衡时，两个效应相互抵消。对每个输入都预测方差，就是异方差回归；它可以表达“有些输入天生更不确定”，但也给模型一个通过放大方差逃避大错的通道。

### 训练方差头的边界

常见的安全边界包括：

| 边界 | 目的 | 需要观察 |
| --- | --- | --- |
| 预测 log-variance 而不是直接预测 variance | 保证正值并改善尺度 | $s$ 的分布与极端值 |
| 对 $s$ 设置合理下限或软约束 | 防止方差趋近零导致尖锐 NLL | 小残差区域的梯度 |
| 对均值和尺度分别记录梯度 | 区分中心错误与不确定性错误 | 梯度范数、更新比例 |
| 在冻结模型上做分箱校准 | 检查“预测的 90% 区间”是否真的覆盖约 90% | 覆盖率、宽度、分组漂移 |

如果只看 NLL 的平均值，无法知道模型是否通过把一部分样本的方差夸大来掩盖均值错误。应同时报告 MAE 或 RMSE、标准化残差分布、预测区间覆盖率和方差与残差大小的关系。

## 目标变换和计数目标

### 正值长尾目标可以在 log 空间建模

当目标严格非负且更关心乘法误差或相对比例时，可以在变换后的目标上训练：

$$
z=\log(1+y),
\qquad
\widehat z=\log(1+\widehat y),
$$

对应的 MSLE 为

$$
\ell_{\mathrm{MSLE}}
=\left(
\log(1+\widehat y)-\log(1+y)
\right)^2.
$$

它会让 $1$ 到 $2$ 与 $100$ 到 $200$ 的相对差异更接近，而不是把后者的原始单位误差放大一百倍。代价是目标必须满足非负，且从 log 空间还原到原空间时，均值和指数不能简单交换：

$$
\exp(\mathbb E[\log(1+Y)])-1
\neq
\mathbb E[Y]
$$

一般不相等。若业务最终关心原空间的均值，需要检查反变换偏差，而不能只报告 log 空间损失。

### 计数目标适合 Poisson 似然

对非负整数计数 $y\in\{0,1,2,\ldots\}$，可以让模型输出 log-rate $\eta$：

$$
\mu=\exp(\eta).
$$

Poisson 负对数似然为

$$
\ell_{\mathrm{Poisson}}(\eta;y)
=\exp(\eta)-y\eta+\log\Gamma(y+1).
$$

对 $\eta$ 的梯度很简单：

$$
\frac{\partial\ell_{\mathrm{Poisson}}}{\partial\eta}
=\exp(\eta)-y
=\mu-y.
$$

相比对计数直接用 MSE，Poisson 目标把均值和方差相等的分布假设写进了训练；若数据过度离散，应该检查负二项模型、稳健目标或额外尺度参数，而不是默默把所有计数错误当作同方差 Gaussian 噪声。

## 训练时怎样组合回归损失

一个实际目标可能同时包含点误差、分位数、权重和正则项：

$$
J(\boldsymbol\theta)
=\frac{\sum_i w_i\ell_i}
{\sum_i w_i}
+\lambda R(\boldsymbol\theta).
$$

组合时要把每项的单位和梯度尺度写清楚。比如把一个无量纲的区间覆盖惩罚直接加到单位为秒的 MSE 上，$\lambda$ 就承担了单位换算；它不是可以跨数据集照抄的常数。

下面的检查表适合在训练配置评审前使用：

| 问题 | 如果答案是“是” | 需要补的核验 |
| --- | --- | --- |
| 大残差可能是脏标签吗 | 先比较 Huber、log-cosh 或 Laplace NLL | 尾部样本梯度和分组误差 |
| 高估和低估代价不同吗 | 使用 pinball 或显式加权 | 阈值方向、分位数覆盖 |
| 输入决定噪声尺度吗 | 使用异方差 NLL | 方差头、标准化残差和校准 |
| 目标严格为正或为计数吗 | 考虑 log 变换或 Poisson | 零值、反变换、过度离散 |
| 样本重要性不同吗 | 使用 sample weight 或 mask | 权重分母、有效样本数 |

后续 [回归评估指标](../evaluation-and-generalization/regression-metrics/) 负责报告误差尺度和残差诊断；训练目标和评估指标应成对记录，才能看出“训练在优化什么、报表在衡量什么”。

## 一个统一的小例子

取三个回归样本

$$
\boldsymbol y=(1,4,10),
\qquad
\widehat{\boldsymbol y}=(0,5,8),
\qquad
\boldsymbol r=(-1,1,-2).
$$

对样本取平均，几种点损失为

$$
\operatorname{MAE}=\frac{4}{3},
\qquad
\operatorname{MSE}=2,
\qquad
L_{\mathrm{Huber},\delta=1}=\frac{2.5}{3}.
$$

同一组预测相对于预测值的梯度分别是

$$
\nabla\operatorname{MAE}
=\left(-\frac13,\frac13,-\frac13\right),
$$

$$
\nabla\operatorname{MSE}
=\left(-\frac23,\frac23,-\frac43\right),
$$

$$
\nabla L_{\mathrm{Huber}}
=\left(-\frac13,\frac13,-\frac13\right).
$$

MSE 让最后一个残差为 $-2$ 的样本贡献两倍于 Huber 的负向分量；Huber 已经把它截在 $\delta=1$。log-cosh 的梯度介于两者之间，因为 $\tanh(-2)$ 已经接近 $-1$，但还没有硬折点。

对 $\tau=0.9$ 的 pinball 损失，负残差的代价系数是 $0.9$，正残差的代价系数是 $0.1$：

$$
\rho_{0.9}(-1)=0.9,
\qquad
\rho_{0.9}(1)=0.1,
\qquad
\rho_{0.9}(-2)=1.8.
$$

平均值为 $0.933333333333$，它表达的是“宁可高估一点，也不要低估很多”的目标，而不是又一个对称误差指标。

## 运行方法

下面的脚本只使用 Python 标准库，逐项计算点损失、梯度、分位数损失、Gaussian NLL、Poisson NLL 和 log 目标变换。代码中的均值分母、残差符号与正文完全一致。

```python
import math

y = [1.0, 4.0, 10.0]
prediction = [0.0, 5.0, 8.0]
residual = [p - t for p, t in zip(prediction, y)]
n = len(y)

def huber(r, delta=1.0):
    if abs(r) <= delta:
        return 0.5 * r * r
    return delta * (abs(r) - 0.5 * delta)

def log_cosh(r):
    magnitude = abs(r)
    return (
        magnitude
        + math.log1p(math.exp(-2.0 * magnitude))
        - math.log(2.0)
    )

def pinball(r, tau):
    return (1.0 - tau) * r if r >= 0.0 else -tau * r

mae = sum(abs(r) for r in residual) / n
mse = sum(r * r for r in residual) / n
huber_mean = sum(huber(r) for r in residual) / n
log_cosh_mean = sum(log_cosh(r) for r in residual) / n

mae_grad = [
    (1.0 if r > 0.0 else -1.0 if r < 0.0 else 0.0) / n
    for r in residual
]
mse_grad = [2.0 * r / n for r in residual]
huber_grad = [
    max(-1.0, min(1.0, r)) / n
    for r in residual
]
log_cosh_grad = [math.tanh(r) / n for r in residual]

tau = 0.9
pinball_values = [pinball(r, tau) for r in residual]
pinball_mean = sum(pinball_values) / n

target_value = 3.0
mean_value = 2.5
log_variance = math.log(0.25)
gaussian_nll = 0.5 * (
    log_variance
    + (target_value - mean_value) ** 2 * math.exp(-log_variance)
    + math.log(2.0 * math.pi)
)

count = 3.0
log_rate = math.log(4.0)
poisson_nll = (
    math.exp(log_rate)
    - count * log_rate
    + math.lgamma(count + 1.0)
)
poisson_grad = math.exp(log_rate) - count

positive_y = [1.0, 10.0]
positive_prediction = [2.0, 8.0]
msle = sum(
    (
        math.log1p(p) - math.log1p(t)
    ) ** 2
    for p, t in zip(positive_prediction, positive_y)
) / len(positive_y)

conditional = [0.0, 0.0, 0.0, 10.0]
square_risk_at_zero = sum(value * value for value in conditional) / 4.0
square_risk_at_mean = sum(
    (value - 2.5) ** 2 for value in conditional
) / 4.0
absolute_risk_at_zero = sum(abs(value) for value in conditional) / 4.0
absolute_risk_at_mean = sum(
    abs(value - 2.5) for value in conditional
) / 4.0

print(
    f"point losses MAE={mae:.12f} MSE={mse:.12f} "
    f"Huber={huber_mean:.12f} log-cosh={log_cosh_mean:.12f}"
)
print("point gradients", mae_grad, mse_grad, huber_grad, log_cosh_grad)
print(
    f"pinball tau={tau:.1f} values={[round(value, 12) for value in pinball_values]} "
    f"mean={pinball_mean:.12f}"
)
print(
    f"probabilistic Gaussian-NLL={gaussian_nll:.12f} "
    f"Poisson-NLL={poisson_nll:.12f} "
    f"Poisson-log-rate-grad={poisson_grad:.12f}"
)
print(f"MSLE={msle:.12f}")
print(
    f"conditional square-risk-at-0={square_risk_at_zero:.12f} "
    f"at-mean={square_risk_at_mean:.12f}"
)
print(
    f"conditional absolute-risk-at-0={absolute_risk_at_zero:.12f} "
    f"at-mean={absolute_risk_at_mean:.12f}"
)
```

运行输出：

```text
point losses MAE=1.333333333333 MSE=2.000000000000 Huber=0.833333333333 log-cosh=0.730854802775
point gradients [-0.3333333333333333, 0.3333333333333333, -0.3333333333333333] [-0.6666666666666666, 0.6666666666666666, -1.3333333333333333] [-0.3333333333333333, 0.3333333333333333, -0.3333333333333333] [-0.2538647186519216, 0.2538647186519216, -0.321342526691939]
pinball tau=0.9 values=[0.9, 0.1, 1.8] mean=0.933333333333
probabilistic Gaussian-NLL=0.725791352645 Poisson-NLL=1.632876385868 Poisson-log-rate-grad=1.000000000000
MSLE=0.102335340955
conditional square-risk-at-0=25.000000000000 at-mean=18.750000000000
conditional absolute-risk-at-0=2.500000000000 at-mean=3.750000000000
```

这些数字把几种选择放在同一个尺度之外进行比较：MSE 的 2 和 Huber 的 0.833333 不能直接说谁“更好”，因为它们的函数定义不同；Gaussian NLL 还包含方差和常数项；pinball 的 $0.933333$ 则带有明确的不对称代价。

## 常见失效模式

### 符号换了，分位数方向却没换

这是 pinball 最隐蔽的错误。若代码使用 residual=prediction-target，就必须使用正侧 $1-\tau$、负侧 $-\tau$；若代码使用 target-prediction，则正侧为 $\tau$、负侧为 $-(1-\tau)$。用 $\tau=0.9$ 的三个残差做一个手算样本，通常比看训练曲线更快发现方向反了。

### 用 MSE 训练长尾标签，却只按 MAE 解释

MSE 会让极少数大残差获得更大的梯度；如果这些值是脏标签，模型可能反复追逐错误目标。如果最终业务确实关心大错，MSE 的选择有理由；如果大错主要来自观测污染，应比较 Huber、log-cosh、Laplace NLL 和分组尾部指标，而不是事后只报告 MAE。

### Huber 阈值没有跟单位一起变化

标准化目标、改变货币单位或把秒换成毫秒都会改变残差数值。固定 $\delta$ 会隐式改变平方区和线性区的比例。配置中要记录目标变换、残差尺度、$\delta$ 和恢复到原单位后的指标。

### 异方差模型用大方差掩盖均值错误

Gaussian NLL 允许模型提高方差来降低一个大残差的标准化代价。若没有分箱校准、标准化残差和均值指标，NLL 下降可能只是方差头变得更保守。检查 $s$ 的分布、均值梯度与尺度梯度的比例，以及高置信度区间的真实覆盖率。

### 分位数交叉或区间只追求覆盖

独立的多个分位数头可能输出上下界顺序颠倒；强行排序又可能让梯度和输出解释不透明。区间覆盖率接近目标也不够，必须同时检查平均宽度、条件覆盖率和不同时间/群组的偏差。

### 对零值或负值使用相对损失

MAPE、log 变换和 MSLE 都有前提。真实值为零时 MAPE 未定义，负值不能直接取 $\log(1+y)$，在原空间和 log 空间之间切换还会改变“平均”的含义。不要只用加一个很小的 epsilon 掩盖分母或定义域问题。

### 归约和权重改变了样本含义

对每个样本先平均、对每个用户先平均、对所有 token 一起平均，得到的是三个不同目标。mask、sample weight 和多任务损失的分母必须显式记录；否则 batch 长度、用户活跃度或类别频率会变成隐形权重。

## 核验协议

实现或替换一个回归损失时，可以按以下顺序核对：

1. 先在残差为负、为零、为正以及跨过 Huber 阈值的点上手算值和左右斜率；
2. 固定残差符号后，用中心差分核对光滑区域的梯度，在折点处改查左右差分和次梯度；
3. 对 Gaussian NLL 分别改变均值和 log-variance，确认两个梯度方向与公式一致；
4. 对多个分位数检查分位数顺序、覆盖率、区间宽度和不同分组的稳定性；
5. 把逐样本损失、归约分母、目标变换、权重、均值指标和部署代价放在同一份审计记录里。

还应在极端数值上测试稳定实现：大残差的 log-cosh、很小或很大的 log-variance、Poisson 的大 log-rate、接近零的目标和全为零的 mask。只有能同时解释损失值、梯度方向和最终指标，才算完成了回归损失的核验。

## 相关词条

- [损失函数总览](../training-nn/loss-functions/)
- [回归评估指标](../evaluation-and-generalization/regression-metrics/)
- [最大似然估计](../probability/maximum-likelihood/)
- [高斯分布](../probability/gaussian-distribution/)
- [期望](../probability/expectation/)
- [方差与协方差](../probability/variance-and-covariance/)
- [梯度下降](../training-nn/gradient-descent/)
