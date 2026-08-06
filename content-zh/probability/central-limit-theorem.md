---
title: "中心极限定理：平均值为何呈现钟形波动"
tags: ["why-models-learn"]
---

**中心极限定理**说明，许多独立同分布的随机变量相加或取平均后，经过中心化和标准化，分布会接近标准正态分布。若 $X_1,X_2,\ldots$ 独立同分布，且

$$
\mathbb E[X_i]=\mu,\qquad
\operatorname{Var}(X_i)=\sigma^2\in(0,\infty)
$$

令样本平均为

$$
\bar X_n=\frac1n\sum_{i=1}^nX_i
$$

则经典中心极限定理写成

$$
Z_n
=\frac{\bar X_n-\mu}{\sigma/\sqrt n}
=\frac{\sqrt n(\bar X_n-\mu)}{\sigma}
\xrightarrow{d}
\mathcal N(0,1)
$$

这里的 $\xrightarrow{d}$ 是依分布收敛。大数定律回答「平均值最终靠近哪里」，中心极限定理回答「平均值在这个位置周围怎样波动，以及这种波动何时可以用钟形曲线近似」。本篇从标准化的必要性出发，推导和与平均值的近似分布，解释正态曲线出现的原因，最后连接连续性修正、置信区间和机器学习中的批次噪声。

## 大数定律和中心极限定理分别控制什么

令

$$
S_n=X_1+\cdots+X_n
$$

大数定律给出

$$
\bar X_n=\frac{S_n}{n}
\xrightarrow{P}
\mu
$$

它的重点是极限位置：样本数足够大时，平均值偏离 $\mu$ 超过固定阈值的概率趋于 0。中心极限定理则保留这个偏离，但把它放大到一个不会消失的尺度：

$$
\sqrt n(\bar X_n-\mu)
\xrightarrow{d}
\mathcal N(0,\sigma^2)
$$

再除以 $\sigma$，就得到标准正态：

$$
\frac{\sqrt n(\bar X_n-\mu)}{\sigma}
\xrightarrow{d}
\mathcal N(0,1)
$$

可以把两条定律并排看成：

| 定律 | 研究的量 | 极限陈述 | 主要问题 |
| --- | --- | --- | --- |
| 大数定律 | $\bar X_n$ | $\bar X_n\to\mu$ | 平均值会靠近哪里 |
| 中心极限定理 | $\sqrt n(\bar X_n-\mu)/\sigma$ | $\to\mathcal N(0,1)$ | 偏离均值的形状和尺度 |

中心极限定理不是大数定律的更强版本。它研究的是另一个经过缩放的随机量：不缩放时，波动被压到 0；乘上 $\sqrt n$ 后，波动保留为有限的钟形随机量。

## 为什么必须乘上 $\sqrt n$

设每个样本的方差是 $\sigma^2$，且样本独立。方差的加法规则给出

$$
\operatorname{Var}(S_n)
=\sum_{i=1}^n\operatorname{Var}(X_i)
=n\sigma^2
$$

因此和的标准差是

$$
\operatorname{SD}(S_n)=\sigma\sqrt n
$$

平均值把和除以 $n$：

$$
\operatorname{Var}(\bar X_n)
=\operatorname{Var}\left(\frac{S_n}{n}\right)
=\frac{n\sigma^2}{n^2}
=\frac{\sigma^2}{n}
$$

所以平均值的典型波动是

$$
\operatorname{SD}(\bar X_n)=\frac{\sigma}{\sqrt n}
$$

这解释了标准化公式里的分母。若只看 $\bar X_n-\mu$，它的尺度随 $1/\sqrt n$ 消失；除以同一个尺度后：

$$
\frac{\bar X_n-\mu}{\sigma/\sqrt n}
=\frac{\sqrt n(\bar X_n-\mu)}{\sigma}
$$

得到一个均值为 0、方差为 1 的量。对满足条件的独立样本，中心极限定理告诉我们，这个量的整体分布会靠近标准正态，而不只是拥有相同的均值和方差。

## 从标准化和到平均值的近似分布

若 $Z_n$ 已经接近标准正态，就可以把等式

$$
\bar X_n
=\mu+\frac{\sigma}{\sqrt n}Z_n
$$

读成一个近似采样规则。因此

$$
\bar X_n
\approx
\mathcal N\left(\mu,\frac{\sigma^2}{n}\right)
$$

这里的 $\approx$ 不是说原始样本 $X_i$ 本身必须服从正态分布，而是说样本平均这个统计量在 $n$ 足够大时近似服从正态。原始分布可以是离散的、偏斜的，甚至只有有限个取值。

对任意区间 $[a,b]$，标准化端点：

$$
\begin{aligned}
P(a\leq\bar X_n\leq b)
&\approx
P\left(
\frac{a-\mu}{\sigma/\sqrt n}
\leq Z\leq
\frac{b-\mu}{\sigma/\sqrt n}
\right)\\
&=
\Phi\left(\frac{b-\mu}{\sigma/\sqrt n}\right)
-\Phi\left(\frac{a-\mu}{\sigma/\sqrt n}\right)
\end{aligned}
$$

其中 $Z\sim\mathcal N(0,1)$，$\Phi(z)=P(Z\leq z)$ 是标准正态分布的累积分布函数。概率问题因此被拆成三步：先减去 $\mu$，再除以标准误 $\sigma/\sqrt n$，最后查标准正态 CDF。

标准正态的几个中心区间是：

| 样本平均的近似区间 | 标准化边界 | 近似覆盖率 |
| --- | --- | --- |
| $\mu\pm\sigma/\sqrt n$ | $z=1$ | 0.6827 |
| $\mu\pm1.96\sigma/\sqrt n$ | $z=1.96$ | 0.9500 |
| $\mu\pm2\sigma/\sqrt n$ | $z=2$ | 0.9545 |
| $\mu\pm3\sigma/\sqrt n$ | $z=3$ | 0.9973 |

这些数字描述的是样本平均的抽样波动，不是说每一个原始 $X_i$ 都落在对应区间内。$n$ 越大，区间宽度按 $1/\sqrt n$ 缩小。

![大量独立贡献的标准化和趋近钟形曲线](/assets/probability/svg/central-limit-theorem.1.svg)

## 伯努利样本：离散频率如何变成连续钟形

令 $X_i$ 是伯努利变量：

$$
P(X_i=1)=p,\qquad
P(X_i=0)=1-p
$$

它的期望和方差是

$$
\mathbb E[X_i]=p,\qquad
\operatorname{Var}(X_i)=p(1-p)
$$

成功次数

$$
S_n=X_1+\cdots+X_n
$$

服从二项分布，样本平均

$$
\hat p=\bar X_n=\frac{S_n}{n}
$$

是成功比例。因此中心极限定理给出

$$
\frac{\hat p-p}{\sqrt{p(1-p)/n}}
\xrightarrow{d}
\mathcal N(0,1)
$$

以 $p=0.5,n=100$ 为例：

$$
\mathbb E[\hat p]=0.5,\qquad
\operatorname{SD}(\hat p)
=\sqrt{\frac{0.5\cdot0.5}{100}}
=0.05
$$

不做连续性修正时，区间 $0.4\leq\hat p\leq0.6$ 对应标准化边界 $-2$ 和 $2$：

$$
\begin{aligned}
P(0.4\leq\hat p\leq0.6)
&=P(40\leq S_{100}\leq60)\\
&\approx\Phi(2)-\Phi(-2)\\
&\approx0.9545
\end{aligned}
$$

但 $S_{100}$ 只能取整数，正态变量却可以取所有实数。连续性修正把整数区间两端向外扩半格：

$$
P(40\leq S_{100}\leq60)
\approx
P(39.5\leq Y\leq60.5),
\qquad
Y\sim\mathcal N(50,25)
$$

此时标准化端点是 $-2.1$ 和 $2.1$，近似概率变为

$$
\Phi(2.1)-\Phi(-2.1)
\approx0.9643
$$

精确的二项概率约为 $0.9648$，连续性修正把近似推得更近。它不是新的定理条件，而是把离散柱状概率用连续曲线覆盖时的几何修补；当 $n$ 很小或区间靠近 0、1 时，仍应优先使用精确的二项计算。

## 公平骰子：和与平均值的同一条正态近似

公平六面骰的单次期望和方差是

$$
\mu=3.5,\qquad
\sigma^2=\frac{35}{12}
$$

掷 $n=100$ 次，和与平均值分别满足近似关系

$$
S_{100}
\approx
\mathcal N\left(
350,\frac{875}{3}
\right)
$$

以及

$$
\bar X_{100}
\approx
\mathcal N\left(
3.5,\frac{35}{1200}
\right)
$$

平均值的标准误为

$$
\sqrt{\frac{35}{1200}}
\approx0.1708
$$

因此用标准正态的 1.96 倍标准误构造一个对称区间：

$$
3.5\pm1.96\sqrt{\frac{35}{1200}}
\approx
[3.1653,3.8347]
$$

在中心极限定理的近似意义下，样本平均落在这个区间内的概率约为 95%。这里的骰子分布只有六个取值，并不是钟形；钟形近似来自 100 次相加后的统计量，而不是来自单个骰子。

## 为什么许多分布都会出现正态曲线

中心极限定理的完整证明可以使用特征函数。下面的计算不是把定理的所有技术条件都证明一遍，而是展示正态形状从哪里出现。

对一个随机变量 $X$，特征函数定义为

$$
\varphi_X(t)=\mathbb E[e^{itX}]
$$

先把变量中心化并标准化：

$$
Y=\frac{X-\mu}{\sigma}
$$

于是 $\mathbb E[Y]=0$、$\operatorname{Var}(Y)=1$。在 $t=0$ 附近，有限方差允许写出二阶展开：

$$
\varphi_Y(t)
=1-\frac{t^2}{2}+o(t^2)
$$

令

$$
Z_n=\frac{Y_1+\cdots+Y_n}{\sqrt n}
$$

独立性使和的特征函数相乘：

$$
\begin{aligned}
\varphi_{Z_n}(t)
&=\prod_{i=1}^n\varphi_Y\left(\frac{t}{\sqrt n}\right)\\
&=\left[
1-\frac{t^2}{2n}+o\left(\frac1n\right)
\right]^n\\
&\longrightarrow e^{-t^2/2}
\end{aligned}
$$

$e^{-t^2/2}$ 正是标准正态的特征函数。这个骨架包含三个关键动作：每个小贡献的线性偏移已经被中心化消掉，二阶波动在 $n$ 次相乘后累积成有限的 $t^2/2$，更高阶的小量则在极限中消失。独立性不是装饰，它正是把 $n$ 个贡献变成乘积的条件。

## 有限样本时，近似误差有多大

中心极限定理只说误差在 $n\to\infty$ 时消失。若希望得到一个有限样本的误差控制，需要更多关于尾部的条件。Berry–Esseen 型不等式给出一个常用形式：若三阶绝对中心矩有限，

$$
\rho=\mathbb E[|X-\mu|^3]<\infty
$$

则存在与分布无关的常数 $C$，使得

$$
\sup_z
\left|
P(Z_n\leq z)-\Phi(z)
\right|
\leq
C\frac{\rho}{\sigma^3\sqrt n}
$$

右侧显示两个直觉：样本数增加 4 倍，最坏情形的误差界按 2 倍缩小；单次样本的偏斜和重尾越明显，三阶中心矩越大，正态近似可能越慢。这个上界是统一的最坏情况保证，通常比某个具体分布的真实误差宽松，不能把它当成每个区间的精确误差。

对称分布不等于任意小样本都适合正态近似。偏斜、厚尾、边界和离散格点都会让有限样本曲线与正态曲线存在明显差别。画直方图或直接计算精确分布，可以帮助判断近似是否已经足够。

## 置信区间：随机的是区间，不是未知参数

假设 $\mu$ 是未知但 $\sigma$ 已知。由中心极限定理，

$$
\frac{\bar X_n-\mu}{\sigma/\sqrt n}
\approx
\mathcal N(0,1)
$$

因此对显著性水平 $\alpha$，令 $z_{1-\alpha/2}$ 满足

$$
\Phi(z_{1-\alpha/2})=1-\frac{\alpha}{2}
$$

可以构造区间

$$
\left[
\bar X_n-z_{1-\alpha/2}\frac{\sigma}{\sqrt n},
\bar X_n+z_{1-\alpha/2}\frac{\sigma}{\sqrt n}
\right]
$$

重复抽样并重复构造区间时，约 $1-\alpha$ 的区间会覆盖真正的 $\mu$。更准确地说，这是对构造程序的长期覆盖率陈述；在一次区间已经算出后，不能把频率学派的 95% 直接解释成「$\mu$ 落在这一个固定区间里的概率是 95%」。

真实问题通常不知道 $\sigma$，会用样本标准差

$$
s^2=\frac1{n-1}\sum_{i=1}^n(X_i-\bar X_n)^2
$$

替换它。大样本下用 $s$ 的正态区间通常是渐近有效的；若原始数据本身是正态且样本较小，精确推断会用 t 分布。是否能把标准误估得可靠，还取决于抽样单位、独立性和尾部，而不仅是公式里出现了一个 $n$。

## 机器学习：批次平均和梯度噪声

对单个样本的损失记为

$$
L_i=\ell_\theta(X_i,Y_i)
$$

总体风险和批次经验风险分别是

$$
R(\theta)=\mathbb E[L_i],\qquad
\widehat R_B(\theta)=\frac1B\sum_{i=1}^BL_i
$$

在独立同分布的抽样近似下，中心极限定理给出

$$
\frac{\widehat R_B(\theta)-R(\theta)}
{\operatorname{SD}(L_i)/\sqrt B}
\xrightarrow{d}
\mathcal N(0,1)
$$

因此批次平均损失的随机波动通常按 $1/\sqrt B$ 缩小。梯度也有同样的结构。令

$$
G_i=\nabla_\theta\ell_\theta(X_i,Y_i),\qquad
\widehat G_B=\frac1B\sum_{i=1}^BG_i
$$

在一维方向 $v$ 上，若 $v^\mathsf TG_i$ 的方差有限：

$$
\frac{
v^\mathsf T(\widehat G_B-\mathbb E[G_i])
}{
\sqrt{\operatorname{Var}(v^\mathsf TG_i)/B}
}
\xrightarrow{d}
\mathcal N(0,1)
$$

这解释了增大 batch size 往往会让梯度更新更平滑，但不能推出训练一定更好。若一个 batch 里大量样本来自同一用户、同一视频或同一时间片，交叉协方差会改变 $\sigma^2/B$ 的缩减速度；若数据增强、缓存或标签泄漏让样本不再近似独立，简单的中心极限定理近似也会失效。多维梯度的完整版本需要多元中心极限定理，极限是协方差矩阵控制的多元高斯。

## 失效模式

**把中心极限定理当成大数定律。**大数定律告诉平均值靠近 $\mu$，中心极限定理告诉标准化后的偏离如何分布；只说「样本多所以接近」没有给出波动形状。

**忘记标准误。**平均值的标准差是 $\sigma/\sqrt n$，不是 $\sigma$。直接用单样本标准差去套平均值区间，会把不确定性夸大约 $\sqrt n$ 倍。

**把原始分布必须正态当成前提。**定理的经典版本允许骰子、伯努利变量等非正态原始分布；正态近似针对的是和或平均值。

**小样本和强偏斜时盲信正态曲线。**渐近定理不保证 $n=10$ 已经足够。要检查偏斜、尾部、边界和精确分布，离散分布还要考虑连续性修正。

**把近似概率当精确概率。**$P(40\leq S_{100}\leq60)$ 的二项精确值和正态近似值不同；中心极限定理控制的是极限，不会抹掉有限样本误差。

**把置信区间解释成参数的后验概率。**频率学派区间的 95% 是重复抽样覆盖率，不是一次观察后固定参数随机移动的概率。

**只数记录，不检查独立抽样单位。**同一用户的多条记录可能携带共同偏好；有效样本量未必等于行数，标准误也未必按 $1/\sqrt n$ 缩小。

**遇到无穷方差仍套用标准正态。**经典中心极限定理需要有限正方差；重尾和的极限可能属于稳定分布族，不能只因样本量大就默认钟形。

## 相关词条

- [大数定律](../probability/law-of-large-numbers/)：说明样本平均趋近总体期望，以及依概率与几乎处处收敛。
- [期望](../probability/expectation/)：提供总体平均、损失风险和函数期望的定义。
- [方差与协方差](../probability/variance-and-covariance/)：给出和、平均值和相关样本的波动计算。
- [高斯分布](../probability/gaussian-distribution/)：介绍标准正态、CDF、分位点和高斯区间。
- [随机变量](../probability/random-variables/)：区分随机对象、分布和观测值。
- [独立性](../probability/independence/)：说明样本乘积结构和交叉协方差为何可以消失。
- [抽样](../probability/sampling/)：继续讨论抽样单位、估计量和样本代表性。
