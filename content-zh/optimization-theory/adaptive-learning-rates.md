---
title: "自适应学习率：按坐标缩放步长"
tags: ["why-models-learn"]
---

自适应学习率方法根据各坐标过去的梯度大小，给不同坐标分配不同的有效步长。AdaGrad 把平方梯度累积起来，RMSProp 用指数滑动平均忘掉旧尺度，Adam 再把一阶方向和二阶尺度结合起来。它们都仍然是梯度法：分母是梯度历史的统计量，不是 Hessian 的逆，也不自动保证每一步目标下降。本文从坐标预条件的共同形式出发，推导三种方法的更新式、偏置修正和尺度行为，再用数值例子说明它们何时有帮助、何时会把学习率问题藏起来。

## 一个全局学习率为什么会卡住

先看最简单的二维二次目标：

$$
F(x,y)=\frac12\left(x^2+100y^2\right)
$$

梯度是

$$
\nabla F(x,y)=
\begin{bmatrix}
x\\
100y
\end{bmatrix}
$$

普通梯度下降的两个坐标分别满足

$$
x_{t+1}=(1-\eta)x_t,
\qquad
y_{t+1}=(1-100\eta)y_t
$$

因此稳定性要求

$$
0<\eta<\frac2{100}=0.02
$$

步长必须服从曲率最大的方向。取 $\eta=0.01$，从 $(x_0,y_0)=(1,1)$ 出发：

$$
(x_1,y_1)=(0.99,0),
\qquad
(x_{10},y_{10})=(0.99^{10},0)
\approx(0.9044,0)
$$

陡峭的 $y$ 方向一步就被压到谷底附近，平坦的 $x$ 方向十步后还剩约 $90.44\%$。如果把 $\eta$ 再调大，$y$ 方向会在谷底两侧振荡；如果把 $\eta$ 调小，$x$ 方向会更慢。一个全局标量无法同时表达「这个坐标的梯度已经很大」和「另一个坐标还需要更大步长」。

自适应方法不改变目标函数，而是把一个标量学习率改成随时间和坐标变化的预条件器。用 $g_t=\nabla F(\boldsymbol\theta_t)$ 表示第 $t$ 步梯度，设 $d_{t,i}>0$ 是第 $i$ 个坐标的尺度，则共同形式是

$$
\theta_{t+1,i}
=\theta_{t,i}
-\eta\frac{g_{t,i}}{d_{t,i}}
$$

有效学习率为

$$
\eta_{t,i}^{\mathrm{eff}}
=\frac{\eta}{d_{t,i}}
$$

当某个坐标的历史平方梯度较大时，$d_{t,i}$ 较大，该坐标的有效步长变小；历史梯度较小时，分母较小，更新相对变大。这个规则只知道「过去这个坐标的梯度有多大」，并不知道目标函数在该方向的真实曲率。

![自适应学习率把坐标梯度除以历史尺度](/assets/optimization-theory/svg/adaptive-learning-rates.1.svg)

## AdaGrad：把见过的梯度全部记住

AdaGrad 为每个坐标保存平方梯度的累积量：

$$
G_{t,i}
=G_{t-1,i}+g_{t,i}^2,
\qquad
G_{0,i}=0
$$

更新式为

$$
\theta_{t+1,i}
=\theta_{t,i}
-\eta\frac{g_{t,i}}{\sqrt{G_{t,i}}+\epsilon}
$$

$\epsilon$ 是防止分母为零的小正数。先暂时令 $\epsilon=0$，可以清楚看到平方根的作用。第一步梯度为 $\boldsymbol g_1=(4,1)$，第二步梯度为 $\boldsymbol g_2=(2,1)$，取 $\eta=0.1$：

| 步骤 | 梯度 | 累积平方梯度 | 归一化方向 $g_t/\sqrt{G_t}$ | 参数增量 |
| --- | --- | --- | --- | --- |
| 1 | $(4,1)$ | $(16,1)$ | $(1,1)$ | $(-0.1,-0.1)$ |
| 2 | $(2,1)$ | $(20,2)$ | $(0.4472,0.7071)$ | $(-0.04472,-0.07071)$ |

如果从 $\boldsymbol\theta_0=(0,0)$ 出发，两步后

$$
\boldsymbol\theta_2
\approx(-0.14472,-0.17071)
$$

第一坐标历史上见过更大的梯度，所以第二步的有效更新反而更小。第二坐标虽然当前梯度只有 $1$，但历史尺度也小，归一化后的更新更大。

### 为什么稀疏坐标会得到较大的步长

如果某个坐标长期没有梯度，$G_{t,i}$ 就不增长。它下一次出现非零梯度时，分母仍然小，因此可以走出相对大的步长。这个性质适合稀疏特征：不常出现的特征不应因为其他频繁出现的特征而被统一压小。

代价是累积量不会忘记旧历史。只要某坐标不断收到同尺度的非零梯度，就有

$$
G_{t,i}
\approx t\,\mathbb E[g_{t,i}^2]
$$

于是分母的数量级是 $\sqrt t$，有效学习率会随时间下降。AdaGrad 后期可能变得过于保守，即使当前梯度已经很小，早期的一次大梯度仍然留在分母里。

### 对正坐标缩放的抵消

假设把第 $i$ 个坐标的梯度历史整体放大 $c_i>0$：

$$
g'_{t,i}=c_i g_{t,i}
$$

那么

$$
G'_{t,i}
=\sum_{s=1}^{t}(c_i g_{s,i})^2
=c_i^2G_{t,i}
$$

因此在 $\epsilon=0$ 时

$$
\frac{g'_{t,i}}{\sqrt{G'_{t,i}}}
=\frac{c_i g_{t,i}}{\sqrt{c_i^2G_{t,i}}}
=\frac{g_{t,i}}{\sqrt{G_{t,i}}}
$$

梯度的单位缩放被分子和分母同时抵消了。$\epsilon$ 不为零时，接近零的尺度会受到额外影响；如果 $c_i<0$，那代表坐标方向本身翻转，结果会多出一个符号。

### AdaGrad 的一个简化保证

在一维凸在线问题中，若每一步使用投影使参数留在直径不超过 $D$ 的区间，并令

$$
G_t=\sum_{s=1}^{t}g_s^2
$$

则 AdaGrad 分析中的基本项可以写成

$$
\sum_{t=1}^{T}g_t(\theta_t-\theta^\star)
\leq
\frac{D^2}{2\eta}\sqrt{G_T}
+\frac{\eta}{2}
\sum_{t=1}^{T}\frac{g_t^2}{\sqrt{G_t}}
$$

因为

$$
\frac{g_t^2}{\sqrt{G_t}}
=\frac{G_t-G_{t-1}}{\sqrt{G_t}}
\leq 2\left(\sqrt{G_t}-\sqrt{G_{t-1}}\right)
$$

所以

$$
\sum_{t=1}^{T}\frac{g_t^2}{\sqrt{G_t}}
\leq 2\sqrt{G_T}
$$

得到一个一维的简化上界：

$$
\sum_{t=1}^{T}g_t(\theta_t-\theta^\star)
\leq
\left(\frac{D^2}{2\eta}+\eta\right)\sqrt{G_T}
$$

凸性把左侧的线性化损失差转成函数值差时，这类界说明了 AdaGrad 为什么能利用每个坐标的实际梯度尺度。它不是任意非凸神经网络训练的收敛证明，投影、有界梯度和凸性等假设不能被省略。

## RMSProp：只保留近期尺度

RMSProp 把累积量改成指数滑动平均：

$$
S_{t,i}
=\rho S_{t-1,i}
+(1-\rho)g_{t,i}^2,
\qquad
0<\rho<1
$$

更新为

$$
\theta_{t+1,i}
=\theta_{t,i}
-\eta\frac{g_{t,i}}{\sqrt{S_{t,i}}+\epsilon}
$$

旧平方梯度每一步乘上 $\rho$，因此不会永久留在分母中。$\rho=0.9$ 时，有效记忆长度约为

$$
\frac1{1-\rho}=10
$$

$\rho=0.99$ 时约为 $100$ 步。这个数量级不是硬窗口；第 $j$ 步前的平方梯度仍有权重，只是按 $\rho^j$ 衰减。

### 常梯度的数值例子

令一维梯度始终为 $g_t=2$，$\rho=0.9$，并从 $S_0=0$ 开始。三步的尺度和归一化梯度为

| 步骤 | $S_t$ | $2/\sqrt{S_t}$ |
| --- | --- | --- |
| 1 | $0.4$ | $3.1623$ |
| 2 | $0.76$ | $2.2942$ |
| 3 | $1.084$ | $1.9209$ |

无限步之后 $S_t$ 趋近于 $4$，所以 $2/\sqrt{S_t}$ 趋近于 $1$。起始时分母偏小，第一次更新会比稳定状态更大。这是指数平均从零开始的初始化偏差。

若做偏置修正：

$$
\widehat S_t
=\frac{S_t}{1-\rho^t}
$$

则常梯度的例子在第一步就有 $\widehat S_1=0.4/0.1=4$，归一化梯度为 $1$。RMSProp 的不同实现对是否修正二阶平均、$\epsilon$ 放在平方根内外有不同约定，阅读实现时要把公式对应上，不能只看算法名称。

### AdaGrad 与 RMSProp 的分界

| 特征 | AdaGrad | RMSProp |
| --- | --- | --- |
| 尺度统计 | $G_t=\sum_{s=1}^{t}g_s^2$ | $S_t=\rho S_{t-1}+(1-\rho)g_t^2$ |
| 旧梯度 | 永久保留 | 指数衰减 |
| 长期有效步长 | 通常继续变小 | 稳态时可保持在固定量级 |
| 稀疏梯度 | 未出现的坐标保留较大步长 | 近期没出现时尺度也会逐渐忘掉 |
| 主要风险 | 早期大梯度让后期过于保守 | 遗忘尺度后有效步长可能重新变大 |

## Adam：方向和尺度分别记忆

Adam 同时保存梯度的一阶指数平均和平方梯度的二阶指数平均：

$$
\boldsymbol m_t
=\beta_1\boldsymbol m_{t-1}
+(1-\beta_1)\boldsymbol g_t
$$

$$
\boldsymbol v_t
=\beta_2\boldsymbol v_{t-1}
+(1-\beta_2)\boldsymbol g_t^{\odot 2}
$$

这里的 $\boldsymbol g_t^{\odot 2}$ 表示逐分量平方，通常从 $\boldsymbol m_0=\boldsymbol0$、$\boldsymbol v_0=\boldsymbol0$ 开始。因为初始状态为零，前几步的平均值会系统性偏小。偏置修正是

$$
\widehat{\boldsymbol m}_t
=\frac{\boldsymbol m_t}{1-\beta_1^t},
\qquad
\widehat{\boldsymbol v}_t
=\frac{\boldsymbol v_t}{1-\beta_2^t}
$$

最后更新：

$$
\boldsymbol\theta_{t+1}
=\boldsymbol\theta_t
-\eta\frac{\widehat{\boldsymbol m}_t}
{\sqrt{\widehat{\boldsymbol v}_t}+\epsilon}
$$

$\boldsymbol m_t$ 决定主要前进方向，$\boldsymbol v_t$ 决定每个坐标的尺度。$\beta_1$ 越接近 $1$，方向记忆越长；$\beta_2$ 越接近 $1$，尺度记忆越长。

### 偏置修正的两步计算

取

$$
\beta_1=0.5,
\qquad
\beta_2=0.9,
\qquad
\boldsymbol g_1=(2,1),
\qquad
\boldsymbol g_2=(4,1)
$$

从零状态出发，二阶量按逐坐标计算：

| $t$ | $\boldsymbol g_t$ | $\boldsymbol m_t$ | $\boldsymbol v_t$ | $\widehat{\boldsymbol m}_t$ | $\widehat{\boldsymbol v}_t$ | $\widehat{\boldsymbol m}_t/\sqrt{\widehat{\boldsymbol v}_t}$ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | $(2,1)$ | $(1,0.5)$ | $(0.4,0.1)$ | $(2,1)$ | $(4,1)$ | $(1,1)$ |
| 2 | $(4,1)$ | $(2.5,0.75)$ | $(1.96,0.19)$ | $(3.3333,1)$ | $(10.3158,1)$ | $(1.0378,1)$ |

第一步若没有偏置修正，直接使用 $m_1/\sqrt{v_1}$ 会得到

$$
\frac{(1,0.5)}{\sqrt{(0.4,0.1)}}
\approx(1.5811,1.5811)
$$

而修正后是 $(1,1)$。偏置修正不是让梯度变得更准确的额外信息，而是把「从零开始的指数平均」恢复到与当前样本尺度相称的量级。

### Adam 其实是归一化的动量

动量方法保存历史方向，更新的是未归一化的速度：

$$
\boldsymbol v_{t+1}
=\beta\boldsymbol v_t+\boldsymbol g_t
$$

Adam 的 $\widehat{\boldsymbol m}_t$ 也保存方向历史，但再除以历史平方梯度的平方根：

$$
\text{更新方向}
\approx
\frac{\text{历史梯度平均}}
{\text{历史梯度幅度}}
$$

因此 Adam 常常让不同坐标的更新幅度更接近。这个特性会减弱「大梯度坐标走更大步」的关系，换来的好处是不同量纲和不同稀疏程度的坐标更容易共用一个基础学习率。

## 自适应不等于二阶方法

Newton 方法使用当前位置的曲率：

$$
\boldsymbol\theta_{t+1}
=\boldsymbol\theta_t
-H(\boldsymbol\theta_t)^{-1}\boldsymbol g_t
$$

AdaGrad、RMSProp 和 Adam 使用的是过去梯度的逐坐标统计量：

$$
D_t
=\operatorname{diag}\left(
\sqrt{G_{t,1}}+\epsilon,\ldots,
\sqrt{G_{t,d}}+\epsilon
\right)
$$

或将 $G_t$ 换成 $S_t$、$\widehat{\boldsymbol v}_t$，然后做

$$
\boldsymbol\theta_{t+1}
=\boldsymbol\theta_t-\eta D_t^{-1}\boldsymbol g_t
$$

$D_t$ 是一种对角预条件器，但通常不是 Hessian，更不是 Hessian 的逆。一个坐标的平方梯度很大，可能是因为它曲率大，也可能只是因为噪声大、损失尺度大，或者该坐标暂时遇到异常样本。只看梯度历史无法区分这些情况。

这也解释了为什么「自适应」不等于「自动找到最优步长」：

| 量 | 它回答的问题 | 它没有回答的问题 |
| --- | --- | --- |
| $g_{t,i}$ | 当前沿第 $i$ 个坐标往哪边上升 | 走多远仍然安全 |
| $G_{t,i}$ 或 $S_{t,i}$ | 这个坐标过去的梯度幅度多大 | 当前曲率是否变了 |
| $\widehat v_{t,i}$ | 近期平方梯度的尺度多大 | 分母是否来自可逆的局部模型 |
| Hessian | 局部曲率如何耦合各方向 | 大规模训练时是否值得显式求逆 |

## 归一化会破坏每步下降

归一化把梯度幅度压进分母，因此它不具备普通梯度下降由下降引理提供的同一种步长解释。最简单的单坐标例子是

$$
f(\theta)=50\theta^2,
\qquad
\theta_0=0.1
$$

此时

$$
g_1=f'(\theta_0)=10,
\qquad
f(\theta_0)=0.5
$$

对只含一个非零梯度的 Adam 第一步，若忽略 $\epsilon$，偏置修正后

$$
\widehat m_1=10,
\qquad
\widehat v_1=100,
\qquad
\frac{\widehat m_1}{\sqrt{\widehat v_1}}=1
$$

取 $\eta=0.3$，参数变成

$$
\theta_1=0.1-0.3=-0.2,
\qquad
f(\theta_1)=50\cdot0.2^2=2
$$

目标值从 $0.5$ 上升到 $2$。方法没有违反自己的更新规则；问题是学习率仍然需要由目标尺度和稳定性决定。分母把梯度变成近似符号，并没有替用户检查这一步会不会穿过谷底。

在含噪训练中，单步目标值上升更常见，也不必然代表算法失效；真正需要观察的是一段窗口内的趋势、验证目标和有效更新大小。但如果损失持续上升，首先应检查基础学习率、分母统计量和梯度是否出现异常，而不是把「自适应」当作稳定性保证。

## 尺度不变性有边界

AdaGrad 在 $\epsilon=0$ 时对正的坐标梯度缩放有抵消作用，RMSProp 和 Adam 在稳定统计阶段也有类似倾向。但这不是对所有参数化都不变。

把参数换成非线性坐标 $\theta=h(\phi)$ 后，梯度按链式法则变为

$$
\frac{\partial F}{\partial\phi}
=\frac{\partial F}{\partial\theta}
\frac{\partial h}{\partial\phi}
$$

历史平方梯度、参数更新和函数本身的几何形状都会一起改变。即使两个参数化表示同一个预测函数，逐坐标的自适应更新也不一定对应同一条函数空间轨迹。

因此「归一化后不同坐标更公平」只是一种局部、坐标依赖的说法。它有助于处理量纲差异，却不能替代对参数化和目标几何的检查。

## 权重衰减不要和 L2 梯度项混为一谈

如果把 L2 正则项直接加进梯度，更新是

$$
\boldsymbol\theta_{t+1}
=\boldsymbol\theta_t
-\eta\frac{\boldsymbol g_t+\lambda\boldsymbol\theta_t}
{\sqrt{\widehat{\boldsymbol v}_t}+\epsilon}
$$

此时正则项也被每个坐标的自适应分母缩放。另一种解耦的权重衰减先做收缩，再做梯度更新：

$$
\boldsymbol\theta_{t+1}
=(1-\eta\lambda)\boldsymbol\theta_t
-\eta\frac{\boldsymbol g_t}
{\sqrt{\widehat{\boldsymbol v}_t}+\epsilon}
$$

两式在普通 SGD 中可以非常接近，但在 Adam 这类坐标预条件器下通常不相同。讨论「正则化强度」时，要先说明使用的是加到梯度里的 L2 项，还是独立的 weight decay。

## 参数分别控制什么

| 参数 | 主要作用 | 过小或过大的典型表现 |
| --- | --- | --- |
| $\eta$ | 所有坐标的基础更新尺度 | 过大时仍会过冲，过小时收敛慢 |
| $\beta_1$ | Adam 一阶方向的记忆长度 | 太小方向抖，太大对方向变化反应慢 |
| $\beta_2$ 或 $\rho$ | 平方梯度尺度的记忆长度 | 太小分母噪声大，太大适应新尺度慢 |
| $\epsilon$ | 防止除零并限制极小分母 | 太大时归一化失效，太小时数值敏感 |
| 梯度裁剪阈值 | 限制单批次异常梯度 | 太小会改变方向，太大挡不住异常值 |

可以用

$$
\text{记忆长度}\approx\frac1{1-\beta}
$$

估算 $\beta_1$、$\beta_2$ 或 $\rho$ 看到多少步历史。这个估算只描述权重衰减，不表示算法真的只读取一个长度的窗口。

$\epsilon$ 的位置也要记录。常见的两种写法是

$$
\frac{g}{\sqrt v+\epsilon}
\qquad\text{和}\qquad
\frac{g}{\sqrt{v+\epsilon}}
$$

它们在 $v$ 很大时差别小，在 $v$ 接近零时差别明显，不能在复现实验时随意互换。

## 理论保证与工程表现不是一回事

AdaGrad 的累积平方梯度具有单调性，许多凸优化分析可以利用

$$
G_{t,i}\geq G_{t-1,i}
$$

把历史项逐步控制住。RMSProp 和 Adam 会遗忘旧尺度，$S_t$ 或 $v_t$ 可以下降，因而有效步长可能重新变大。不能把 AdaGrad 的凸问题证明直接移植到这两种方法上。

在一些凸在线构造中，未经额外条件的 Adam 可能出现不收敛的行为。修正方法会让二阶统计量保持单调，例如维护

$$
\widetilde v_{t,i}
=\max\left(\widetilde v_{t-1,i},v_{t,i}\right)
$$

再用 $\widetilde v_t$ 做分母。这个想法保留了 Adam 的方向平均，同时避免分母因为忘记旧尺度而任意减小。它说明「工程上常用」和「在某组假设下有证明」是两种不同的判断。

对实际训练，至少应同时记录下面几类量：

1. 原始梯度的范数和每个参数组的最大值；
2. $\sqrt{\widehat v}$ 或 RMSProp 分母的中位数、最大值和最小值；
3. 实际参数增量的范数，以及增量和参数范数的比值；
4. 训练目标、验证目标和梯度裁剪触发次数。

只看 loss 曲线无法判断是梯度变小、分母变大，还是学习率真的变小。

## 常见失效模式

**把自适应当成稳定性证明。** 分母只改变坐标尺度，$\eta$ 仍然可能太大。先在一维或二维二次目标上测过冲，再把学习率带到真实模型。

**忘记初始化偏置。** 从零开始的 EMA 在早期偏小，尤其是二阶量；要确认实现是否使用 $\widehat m_t$ 和 $\widehat v_t$，不能只凭算法名称猜。

**把一次异常梯度永久留下。** AdaGrad 会长期记住大梯度。若训练的梯度尺度发生阶段性变化，RMSProp 或 Adam 可能更合适，但遗忘也会带来新的过冲风险。

**把噪声当成曲率。** 某坐标的平方梯度大，可能只说明小批次方差大。分母变大后该坐标被压低，未必是在沿正确的曲率方向预条件。

**混淆 L2 正则和解耦权重衰减。** 在自适应分母存在时，两者对不同坐标产生不同效果；复现实验必须把更新式写清楚。

**只看平均 loss。** 自适应方法的单步目标值可以上升。应结合窗口趋势、验证集和有效步长，而不是因为一次上升就断言算法错误，或因为平均下降就忽略参数更新已经失控。

## 相关词条

- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：全局学习率、下降引理和稳定步长的基础。
- [随机梯度下降理论](../optimization-theory/stochastic-gradient-descent-theory/)：平方梯度统计量为何同时包含信号和噪声。
- [动量理论](../optimization-theory/momentum-theory/)：一阶历史方向与 Adam 的一阶平均之间的关系。
- [梯度](../calculus/gradient/)：逐坐标梯度和方向导数的定义。
- [Hessian 矩阵](../calculus/hessian/)：比较梯度历史预条件器与真实曲率。
- [二阶方法](../optimization-theory/second-order-methods/)：Newton、拟 Newton 与预条件更新。
- [曲率与条件数](../optimization-theory/curvature-and-conditioning/)：病态二次目标为何需要方向尺度处理。
- [损失景观](../optimization-theory/loss-landscapes/)：高维非凸目标中的鞍点、谷底和训练轨迹。
- [Adam](../training-nn/adam/)：神经网络训练语境下的 Adam 实现与经验。
- [学习率调度](../training-nn/learning-rate-schedules/)：在自适应基础学习率之外安排时间变化。
- [权重初始化](../training-nn/weight-initialization/)：初始化尺度如何影响早期梯度和分母统计。
