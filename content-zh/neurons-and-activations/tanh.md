---
title: "tanh：零中心的饱和激活"
tags: ["why-models-learn"]
---

tanh 是双曲正切激活函数：它把任意预激活值映射到 $(-1,1)$，在零点附近近似线性，在两端逐渐饱和。它和 sigmoid 的形状可以互相换算，但 tanh 以 0 为中心，正负信号保留在同一个尺度上；代价是它同样会在大绝对值处产生很小的导数。本文从双曲函数的定义推导 tanh 的导数和反函数，用数值表看清零中心与饱和如何同时出现，再把斜率、初始化、输出范围和循环网络中的时间梯度放到同一套检查里。

![tanh 曲线经过原点且上下对称，导数在原点达到峰值并向两端衰减](/assets/neurons-and-activations/svg/tanh.1.svg)

## tanh 是两个双曲函数的比值

双曲正弦和双曲余弦定义为

$$
\sinh z=\frac{e^z-e^{-z}}{2},
\qquad
\cosh z=\frac{e^z+e^{-z}}{2}.
$$

双曲正切是二者的比值：

$$
\tanh z
=\frac{\sinh z}{\cosh z}
=\frac{e^z-e^{-z}}{e^z+e^{-z}}.
$$

分子和分母同时乘以 $e^{-z}$，也可以写成

$$
\tanh z
=\frac{1-e^{-2z}}{1+e^{-2z}}.
$$

这三个形式各有用途：$\sinh z/\cosh z$ 方便推导，指数比值方便看极限，后一种形式方便和 sigmoid 联系。

当 $z$ 很大时，分子和分母都由 $e^z$ 主导；当 $z$ 很小时，二者都由 $e^{-z}$ 主导。因此

$$
\lim_{z\to+\infty}\tanh z=1,
\qquad
\lim_{z\to-\infty}\tanh z=-1,
\qquad
\tanh0=0.
$$

另外，$\sinh$ 是奇函数、$\cosh$ 是偶函数，所以

$$
\tanh(-z)=-\tanh z.
$$

这条奇对称性就是零中心的来源：正的预激活映射到正输出，负的预激活映射到负输出，恰好为 0 的预激活映射到 0。

## 一张数值表同时展示形状和饱和

下面用对称的预激活值看输出：

| $z$ | $\tanh(z)$ | 所在区域 |
| ---: | ---: | --- |
| $-4$ | $-0.999329$ | 负侧深度饱和 |
| $-2$ | $-0.964028$ | 负侧饱和 |
| $-1$ | $-0.761594$ | 过渡区 |
| $-0.5$ | $-0.462117$ | 靠近中心 |
| $0$ | $0$ | 中心 |
| $0.5$ | $0.462117$ | 靠近中心 |
| $1$ | $0.761594$ | 过渡区 |
| $2$ | $0.964028$ | 正侧饱和 |
| $4$ | $0.999329$ | 正侧深度饱和 |

输出被限制在 $(-1,1)$，但这个限制不是把所有输入都压成同一个数。$z=-1$ 和 $z=1$ 仍然能保留相反的方向；$z=-4$ 与 $z=4$ 则已经接近两端，再增大输入，输出改变得很少。

## 导数为什么是 $1-\tanh^2 z$

先从商法则出发。双曲函数的导数为

$$
\frac{\mathrm{d}}{\mathrm{d}z}\sinh z=\cosh z,
\qquad
\frac{\mathrm{d}}{\mathrm{d}z}\cosh z=\sinh z.
$$

因此

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d}z}\tanh z
&=\frac{\cosh z\cdot\cosh z-\sinh z\cdot\sinh z}{\cosh^2 z}\\
&=\frac{\cosh^2 z-\sinh^2 z}{\cosh^2 z}.
\end{aligned}
$$

双曲函数满足恒等式

$$
\cosh^2 z-\sinh^2 z=1.
$$

所以

$$
\tanh'(z)=\frac1{\cosh^2 z}.
$$

还可以把它完全写成输出的函数。因为 $\tanh z=\sinh z/\cosh z$，

$$
\begin{aligned}
1-\tanh^2 z
&=1-\frac{\sinh^2 z}{\cosh^2 z}\\
&=\frac{\cosh^2 z-\sinh^2 z}{\cosh^2 z}\\
&=\frac1{\cosh^2 z}.
\end{aligned}
$$

于是得到神经网络里最常用的形式：

$$
\tanh'(z)=1-\tanh^2 z.
$$

若记 $a=\tanh z$，反向传播只需要缓存前向输出 $a$：

$$
\frac{\partial a}{\partial z}=1-a^2.
$$

这个导数有三个立即可见的性质：

$$
0<1-a^2\leq1,
\qquad
\tanh'(0)=1,
\qquad
\lim_{\lvert z\rvert\to\infty}\tanh'(z)=0.
$$

在零点附近，tanh 不会把局部梯度放大；在两端，它会把梯度压小。

## 导数表揭示饱和的速度

把上一节的输出代入 $1-a^2$：

| $z$ | $\tanh(z)$ | $\tanh'(z)=1-\tanh^2(z)$ | 解释 |
| ---: | ---: | ---: | --- |
| $-4$ | $-0.999329$ | $0.001341$ | 几乎不动 |
| $-2$ | $-0.964028$ | $0.070651$ | 梯度明显变小 |
| $-1$ | $-0.761594$ | $0.419974$ | 仍能传递信号 |
| $0$ | $0$ | $1$ | 梯度最大 |
| $1$ | $0.761594$ | $0.419974$ | 仍能传递信号 |
| $2$ | $0.964028$ | $0.070651$ | 梯度明显变小 |
| $4$ | $0.999329$ | $0.001341$ | 几乎不动 |

例如 $z=2$ 时，输出已经是 $0.964028$，但导数只有 $0.070651$。如果一条标量路径连续经过 10 个处于这一尺度的 tanh，粗略的梯度因子是

$$
(0.070651)^{10}\approx3.1\times10^{-12}.
$$

这不是说任何含 tanh 的网络都会产生这个精确数字；它说明了一个量级：若许多层的预激活都落在尾部，小于 1 的导数会沿路径相乘。[饱和与梯度消失](../neurons-and-activations/saturation-and-vanishing/)会把这种乘积推广到一般的深层网络。

## 一阶导数之外：曲率在哪里

再对导数求一次导：

$$
\begin{aligned}
\tanh''(z)
&=\frac{\mathrm{d}}{\mathrm{d}z}\left(1-\tanh^2 z\right)\\
&=-2\tanh z\cdot\tanh'(z)\\
&=-2\tanh z\left(1-\tanh^2 z\right).
\end{aligned}
$$

因为 $\tanh$ 是奇函数，而 $\tanh'$ 是偶函数，所以 $\tanh''$ 是奇函数。它在 $z=0$ 处为 0，正侧为负，负侧为正：

$$
\tanh''(0)=0,
\qquad
\tanh''(1)\approx-0.639701,
\qquad
\tanh''(-1)\approx0.639701.
$$

这说明曲线在原点附近最陡，但不是在原点处拥有最大弯曲。它从中心向两侧变平，曲率的符号随中心翻转。

## tanh 与 sigmoid 只是中心和尺度不同

sigmoid 与 tanh 的换算关系是

$$
\tanh z=2\sigma(2z)-1.
$$

验证它：

$$
\begin{aligned}
2\sigma(2z)-1
&=\frac{2}{1+e^{-2z}}-1\\
&=\frac{1-e^{-2z}}{1+e^{-2z}}\\
&=\tanh z.
\end{aligned}
$$

反过来，

$$
\sigma(z)=\frac12\left(1+\tanh\frac z2\right).
$$

因此，若把 sigmoid 的输出先乘 2 再减 1，就得到一个零中心输出；若把 tanh 的输入缩小一半、输出平移并缩放，就得到 sigmoid。

二者的比较如下：

| 函数 | 输出范围 | 中点输出 | 中点导数 | 尾部行为 |
| --- | --- | ---: | ---: | --- |
| $\sigma(z)$ | $(0,1)$ | $0.5$ | $0.25$ | 向 0 或 1 饱和 |
| $\tanh(z)$ | $(-1,1)$ | $0$ | $1$ | 向 -1 或 1 饱和 |
| $2\sigma(2z)-1$ | $(-1,1)$ | $0$ | $1$ | 与 tanh 相同 |

所以「tanh 的梯度比 sigmoid 大」需要加上尺度条件。标准 sigmoid 在 $z=0$ 处的最大导数是 $1/4$，而标准 tanh 在 $z=0$ 处的最大导数是 $1$；但二者的输入缩放也不同，不能只比较函数名字。

## 零中心究竟改变了什么

sigmoid 的输出永远为正，tanh 的输出可以为正也可以为负。先看一组均值为 0 的预激活值：

$$
z\in\{-1.5,-0.5,0.5,1.5\}.
$$

对应的输出是

| 预激活 $z$ | sigmoid $\sigma(z)$ | tanh $\tanh(z)$ |
| ---: | ---: | ---: |
| $-1.5$ | $0.182426$ | $-0.905148$ |
| $-0.5$ | $0.377541$ | $-0.462117$ |
| $0.5$ | $0.622459$ | $0.462117$ |
| $1.5$ | $0.817574$ | $0.905148$ |
| 均值 | $0.500000$ | $0$ |

sigmoid 的均值为 0.5，不是错误，因为它的输出语义本来就在 0 和 1 之间；但作为隐藏表示，它会把整个坐标的基线推向正侧。tanh 在对称输入下均值为 0，正负方向都能参与下一层的线性组合。

设下一层的一条标量输出为

$$
u=\mathbf{v}^{\mathsf T}\mathbf{a}+c,
$$

上游梯度记为 $\delta=\partial L/\partial u$。则

$$
\frac{\partial L}{\partial\mathbf{v}}=\delta\mathbf{a},
\qquad
\frac{\partial L}{\partial c}=\delta.
$$

若一批样本的 $\mathbf{a}$ 总是正的，权重梯度中的方向会更容易被同一侧的偏置项带动；若 $\mathbf{a}$ 在零附近有正有负，下一层可以用相反符号的权重组合它们。这个观察只说明优化坐标的差异，不保证 tanh 在任何网络里都更快：输入分布、初始化、归一化和残差结构仍然决定实际梯度。

## 小输入区近似线性

指数展开给出

$$
e^z=1+z+\frac{z^2}{2}+\frac{z^3}{6}+O(z^4),
\qquad
e^{-z}=1-z+\frac{z^2}{2}-\frac{z^3}{6}+O(z^4).
$$

代入 tanh 的指数形式，分子和分母分别为

$$
e^z-e^{-z}=2z+\frac{z^3}{3}+O(z^5),
$$

$$
e^z+e^{-z}=2+z^2+O(z^4).
$$

相除得到

$$
\tanh z=z-\frac{z^3}{3}+O(z^5).
$$

所以当 $\lvert z\rvert$ 很小时，

$$
\tanh z\approx z,
\qquad
\tanh'(z)\approx1-z^2.
$$

这解释了为什么把预激活初始化在零附近通常有帮助：网络先工作在近似线性的高梯度区域，而不是一开始就把大量单元推入两端。初始化不能保证训练全程留在中心，只能改变起点。[激活函数](../neurons-and-activations/activation-functions/)会继续讨论激活分布、尺度和初始化之间的关系。

## 反函数把饱和输出拉回 logit 空间

当 $a=\tanh z$ 且 $a\in(-1,1)$ 时，反双曲正切为

$$
\operatorname{artanh}(a)
=\frac12\log\frac{1+a}{1-a}.
$$

从 $a=\tanh z$ 直接验证：

$$
\begin{aligned}
1+a
&=1+\frac{e^z-e^{-z}}{e^z+e^{-z}}
=\frac{2e^z}{e^z+e^{-z}},\\
1-a
&=1-\frac{e^z-e^{-z}}{e^z+e^{-z}}
=\frac{2e^{-z}}{e^z+e^{-z}}.
\end{aligned}
$$

因此

$$
\frac{1+a}{1-a}=e^{2z},
\qquad
\frac12\log\frac{1+a}{1-a}=z.
$$

反函数的导数是

$$
\frac{\mathrm{d}}{\mathrm{d}a}\operatorname{artanh}(a)
=\frac1{1-a^2}.
$$

它在 $a$ 接近 $\pm1$ 时很大。比如

$$
\operatorname{artanh}(0.9)\approx1.472219,
\qquad
\operatorname{artanh}(0.99)\approx2.646652.
$$

因此如果把一个已经饱和的 tanh 输出强行反变换回无界坐标，微小的输出误差会被放大。反函数适合推导和目标变换，不是把端点 $a=\pm1$ 当成普通可逆坐标的许可。

## 斜率和输出尺度是两个可分开的旋钮

工程中常见的不是裸的 $\tanh z$，而是

$$
a\tanh(kz+c)+d.
$$

其中 $k$ 控制横向过渡宽度，$a$ 控制纵向振幅，$c$ 平移输入，$d$ 平移输出。对 $z$ 求导：

$$
\frac{\mathrm{d}}{\mathrm{d}z}
\left[a\tanh(kz+c)+d\right]
=ak\left(1-\tanh^2(kz+c)\right).
$$

若 $a>0$，输出范围是

$$
(d-a,d+a).
$$

若要把目标 $y\in(0,1)$ 映射为 tanh 的目标，通常先变换为

$$
\widetilde y=2y-1\in(-1,1),
$$

网络输出 $\tanh(z)$ 后再变回

$$
\widehat y=\frac{\tanh(z)+1}{2}.
$$

这不会让 tanh 自动变成概率校准器，只是一个输出范围的坐标变换。概率语义、损失和阈值仍要单独检查。

## 在循环结构里，tanh 的优势和代价同时出现

循环网络常写出类似的隐藏状态更新：

$$
\mathbf{h}_t
=\tanh\left(
W\mathbf{x}_t+U\mathbf{h}_{t-1}+\mathbf{b}
\right).
$$

tanh 把每个隐藏坐标限制在 $(-1,1)$，不让一次异常大的线性组合直接把状态推到任意大的数。可是从时间 $t$ 往前传播时，隐藏状态的局部导数会进入连乘：

$$
\frac{\partial\mathbf{h}_t}{\partial\mathbf{h}_{t-1}}
=\operatorname{diag}\left(1-\tanh^2(\mathbf{z}_t)\right)U.
$$

经过多步以后，梯度包含

$$
\prod_{j=s+1}^{t}
\operatorname{diag}\left(1-\tanh^2(\mathbf{z}_j)\right)U.
$$

如果大多数 $\mathbf{z}_j$ 在饱和区，tanh 的对角因子会压小梯度；如果 $U$ 的谱范数也小于 1，衰减会更明显。门控循环结构、残差路径和更仔细的初始化，都是在改变这条时间路径。[循环神经网络](../rnn-lstm/rnn/)与 [LSTM](../rnn-lstm/lstm/)会把状态记忆和门控分开展开。

## 失效模式

**把零中心误读成不会饱和。** tanh 的中心是 0，但 $z=2$ 时导数已经只有约 $0.070651$，$z=4$ 时只有约 $0.001341$。零中心和非饱和是两个不同性质。

**只看输出范围，不看预激活分布。** 所有输出都落在 $(-1,1)$ 看起来很整齐，但如果大多数样本落在 $\lvert z\rvert>3$，网络实际上在用很小的梯度工作。每层都要记录预激活均值、标准差、分位数和饱和比例。

**把 tanh 当作概率输出。** tanh 的负值有自然的零中心解释，却不是概率。若目标是 $(0,1)$，需要显式的范围变换；若目标是二分类概率，使用 sigmoid 与对应的概率损失更直接。

**把对称输入当成保证。** 只有输入分布关于 0 对称时，奇函数的期望才会是 0。偏置、前一层偏移、数据切片和训练更新都会破坏对称性；看到均值漂移时要查分布，不要只背函数性质。

**在反函数端点使用精确的 $\pm1$。** $\operatorname{artanh}(1)$ 和 $\operatorname{artanh}(-1)$ 不在有限实数范围内。来自浮点计算的 $1.0000001$ 还会让实现产生 NaN，反变换前需要明确截断和误差语义。

**把放大斜率当作免费梯度。** 增大 $k$ 会让中心附近的最大斜率变成 $k$，但也会把过渡区变窄；同一批数据可能更快进入饱和。斜率、学习率和初始化尺度必须一起调。

**循环网络只监控梯度范数。** 总梯度很小不能告诉你是 tanh 饱和、矩阵谱半径过小，还是某条残差路径被截断。要同时记录 $\mathbf{z}_t$ 的饱和比例、$\lVert U\rVert$ 或谱信息以及不同时间距离的梯度。

## 一个可复用的 tanh 核验协议

遇到 tanh 隐藏层或输出头，可以依次检查：

1. 写出实际形式，是 $\tanh z$ 还是 $a\tanh(kz+c)+d$，不要把缩放藏在实现细节里。
2. 用 $z=-2,-1,0,1,2$ 检查奇对称性、零点输出和输出范围。
3. 用 $1-\tanh^2 z$ 独立重算导数，并检查导数在 $z=0$ 最大、两端接近零。
4. 记录每层预激活的均值、标准差和 $\lvert z\rvert$ 超过 2 或 3 的比例。
5. 若声称隐藏表示零中心，用同一批真实激活算均值，不要用函数的奇对称性代替数据证据。
6. 若用于有界回归，明确 $a,d$ 和目标的坐标变换；若用于概率输出，换用匹配的概率头与损失。
7. 若用于循环结构，按时间距离画梯度或 Jacobian 范数，并区分饱和因子与循环矩阵的作用。
8. 对极端输入和反函数边界做数值测试，确认不会出现 NaN、无穷或超出预期的剪裁。

这套协议把 tanh 的两面分开看：它在零附近给出零中心的高梯度表示，在两端提供有界输出但牺牲局部梯度。实际选择取决于数据尺度、网络路径和输出语义，不是由「零中心」三个字单独决定。

## 相关词条

- [什么是神经元](../neurons-and-activations/what-is-a-neuron/)：预激活和激活函数的基本位置
- [激活函数](../neurons-and-activations/activation-functions/)：tanh 与其他标量激活的比较坐标
- [阶跃与 sigmoid](../neurons-and-activations/heaviside-and-sigmoid/)：sigmoid、logit 和 BCE 的完整推导
- [为什么需要非线性](../neurons-and-activations/why-non-linearity/)：非线性如何改变表示空间
- [饱和与梯度消失](../neurons-and-activations/saturation-and-vanishing/)：小导数沿深度或时间的累积
- [循环神经网络](../rnn-lstm/rnn/)：tanh 隐藏状态的时间递推
- [LSTM](../rnn-lstm/lstm/)：用门控路径缓解长距离状态传递
- [ReLU](../neurons-and-activations/relu/)：非饱和正侧与负侧死区的另一种取舍
- [分类指标](../evaluation-and-generalization/classification-metrics/)：当有界输出参与分类决策时的阈值评估
