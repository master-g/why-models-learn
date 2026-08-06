---
title: "动量理论：历史梯度、振荡与加速"
tags: ["why-models-learn"]
---

动量方法不只看当前梯度，还保存一个带几何衰减的历史方向，再用这个方向更新参数。它可以在细长谷底中积累低曲率方向的进展，也可能把参数带过谷底而产生振荡。固定记号后，二次目标上的每个特征方向都变成一个二阶递推；特征根落在单位圆内才会收敛。本篇先推导重球法的稳定区间和条件数收益，再区分随机梯度下的缓冲区噪声、Nesterov 前瞻梯度以及「加速」和「每步下降」的不同含义。

## 先固定速度记号

本文采用下面的速度形式：

$$
\boldsymbol v_{k+1}
=\beta\boldsymbol v_k
+\nabla F(\boldsymbol\theta_k)
$$

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-\eta\boldsymbol v_{k+1}
$$

其中 $\eta>0$ 是学习率，$0\leq\beta<1$ 是动量系数，初始速度通常取 $\boldsymbol v_0=\boldsymbol0$。有些实现把 $\eta$ 乘进速度缓冲区，或者把新梯度乘以 $1-\beta$；这些记号只要前后一致，核心都是历史梯度的加权累积。

把速度展开：

$$
\begin{aligned}
\boldsymbol v_{k+1}
&=\beta^2\boldsymbol v_{k-1}
+\beta\nabla F(\boldsymbol\theta_{k-1})
+\nabla F(\boldsymbol\theta_k)\\
&=\cdots+
\sum_{j=0}^{k}
\beta^j\nabla F(\boldsymbol\theta_{k-j})
\end{aligned}
$$

越近的梯度权重越大，较早梯度的权重按 $\beta^j$ 衰减。若 $\beta=0$，速度就是当前梯度，更新退化成普通梯度下降。

### 动量等价于二阶参数递推

由

$$
\boldsymbol\theta_k-\boldsymbol\theta_{k-1}
=-\eta\boldsymbol v_k
$$

可把速度消掉：

$$
\begin{aligned}
\boldsymbol\theta_{k+1}
&=\boldsymbol\theta_k
-\eta\nabla F(\boldsymbol\theta_k)
+\beta(\boldsymbol\theta_k-\boldsymbol\theta_{k-1})
\end{aligned}
$$

最后一项是上一段参数位移的延续，所以也称为重球法。普通梯度下降只需要当前位置；动量方法还需要前一时刻的位置，递推阶数因此升高。

当 $\beta=0.9$ 时，当前梯度的权重是 $1$，前一步是 $0.9$，前两步是 $0.81$。几何权重的总和是

$$
\sum_{j=0}^{\infty}\beta^j
=\frac1{1-\beta}
$$

因此可以把

$$
\frac1{1-\beta}
$$

看作有效记忆长度的数量级。$\beta=0.9$ 对应约 $10$ 步，$\beta=0.99$ 对应约 $100$ 步；这不是硬截断窗口，较早的梯度仍然存在，只是权重越来越小。

## 二次函数把动量变成特征根问题

先看一维二次目标

$$
f(x)=\frac12\lambda x^2,
\qquad
\lambda>0
$$

梯度为 $\lambda x_k$。用重球形式：

$$
x_{k+1}
=x_k-\eta\lambda x_k+\beta(x_k-x_{k-1})
$$

整理成

$$
x_{k+1}
=(1+\beta-\eta\lambda)x_k-\beta x_{k-1}
$$

这是一个二阶线性递推。令试探解为 $x_k=r^k$，代入得到特征多项式

$$
r^2-(1+\beta-\eta\lambda)r+\beta=0
$$

只有所有特征根都满足

$$
\lvert r\rvert<1
$$

误差才会随 $k$ 趋于零。对 $0\leq\beta<1$，二阶多项式的单位圆稳定条件化为

$$
0<\eta\lambda<2(1+\beta)
$$

它包括三件事：学习率必须为正，缓冲区保留比例必须小于 $1$，而且学习率和曲率的乘积不能超过动量提高后的上限。

| 标量曲率条件 | 递推表现 | 结论 |
| --- | --- | --- |
| $0<\eta\lambda<2(1+\beta)$ | 所有特征根在单位圆内 | 误差趋于零 |
| $\eta\lambda=2(1+\beta)$ | 至少一个根落在单位圆上 | 临界方向不衰减 |
| $\eta\lambda>2(1+\beta)$ | 至少一个根的模大于 1 | 该方向发散 |
| $\beta=0$ | 退化为一阶梯度法 | 条件回到 $0<\eta\lambda<2$ |

注意，稳定不等于函数值每一步都下降。二阶递推可以在谷底两侧来回穿越，只要振荡幅度总体衰减，最终仍然收敛。

### 二维数值例子

取

$$
F(\boldsymbol\theta)
=\frac12\boldsymbol\theta^\mathsf TQ\boldsymbol\theta,
\qquad
Q=
\begin{bmatrix}
1&0\\
0&9
\end{bmatrix}
$$

初始参数为 $\boldsymbol\theta_0=(1,1)^\mathsf T$，并取 $\boldsymbol v_0=\boldsymbol0$。设

$$
\eta=0.25,
\qquad
\beta=0.25
$$

低曲率方向的特征根是

$$
r^2-r+0.25=(r-0.5)^2
$$

高曲率方向的特征根是

$$
r^2+r+0.25=(r+0.5)^2
$$

两个方向的根模都是 $0.5$。逐步更新：

$$
\begin{aligned}
\boldsymbol v_1&=(1,9)^\mathsf T,
&
\boldsymbol\theta_1&=(0.75,-1.25)^\mathsf T\\
\boldsymbol v_2&=(1,-9)^\mathsf T,
&
\boldsymbol\theta_2&=(0.5,1)^\mathsf T\\
\boldsymbol v_3&=(0.75,6.75)^\mathsf T,
&
\boldsymbol\theta_3&=(0.3125,-0.6875)^\mathsf T
\end{aligned}
$$

高曲率坐标的符号已经来回改变，但幅度从 $1$、$1.25$、$1$、$0.6875$ 逐渐受控。第一步的函数值是

$$
F(\boldsymbol\theta_1)
=\frac12(0.75^2+9\cdot1.25^2)
=7.3125
$$

初始函数值为 $5$，所以第一步反而上升。之后的根模分析仍然告诉我们，在这个二次目标上轨迹会逐步回到原点。动量提供的是累计方向和长期收敛速度，不是每一步的单调下降保证。

## 动量如何改善病态二次目标

令二次目标的特征值都落在 $[\mu,L]$，条件数为

$$
\kappa=\frac L\mu
$$

普通梯度法需要同时照顾最小曲率和最大曲率。学习率太小，低曲率方向前进缓慢；学习率太大，高曲率方向就会越过谷底。普通梯度法在二次目标上把最坏方向因子优化到

$$
\eta_{\mathrm{GD}}=\frac2{L+\mu},
\qquad
\rho_{\mathrm{GD}}
=\frac{L-\mu}{L+\mu}
=\frac{\kappa-1}{\kappa+1}
$$

经典重球法针对这个二次谱的参数是

$$
\eta_{\mathrm{HB}}
=\frac4{(\sqrt L+\sqrt\mu)^2}
$$

$$
\beta_{\mathrm{HB}}
=\left(
\frac{\sqrt L-\sqrt\mu}
{\sqrt L+\sqrt\mu}
\right)^2
$$

此时最坏特征根模为

$$
\rho_{\mathrm{HB}}
=\frac{\sqrt L-\sqrt\mu}
{\sqrt L+\sqrt\mu}
=\frac{\sqrt\kappa-1}
{\sqrt\kappa+1}
$$

当 $\kappa$ 很大时，$\rho_{\mathrm{HB}}$ 比 $\rho_{\mathrm{GD}}$ 更接近理想的快速衰减。这是二次目标上的精确谱结论；一般非线性目标的 Hessian 会随参数变化，不能把这组参数当作所有训练问题的固定答案。

回到 $Q=\operatorname{diag}(1,9)$，有 $\mu=1$、$L=9$、$\kappa=9$：

$$
\eta_{\mathrm{HB}}
=\frac4{(3+1)^2}
=0.25,
\qquad
\beta_{\mathrm{HB}}
=\left(\frac{3-1}{3+1}\right)^2
=0.25
$$

最坏根模为

$$
\rho_{\mathrm{HB}}
=\frac{3-1}{3+1}
=0.5
$$

而普通梯度法的最坏根模为

$$
\rho_{\mathrm{GD}}
=\frac{9-1}{9+1}
=0.8
$$

动量把条件数为 $9$ 的谱收缩成根模 $0.5$ 的二阶递推，但代价是参数轨迹会振荡，且对 $\eta$、$\beta$ 的偏离更加敏感。

## 随机梯度下的缓冲区噪声

把随机梯度写成

$$
\widehat{\boldsymbol g}_k
=\nabla F(\boldsymbol\theta_k)+\boldsymbol\xi_k,
\qquad
\mathbb E[\boldsymbol\xi_k\mid\boldsymbol\theta_k]
=\boldsymbol0
$$

速度中的噪声满足

$$
\boldsymbol v_{k+1}
=\beta\boldsymbol v_k+\boldsymbol\xi_k
$$

先假设参数暂时不动，噪声独立、均值为零、方差为 $\sigma^2$。标量方差递推是

$$
s_{k+1}
=\beta^2s_k+\sigma^2
$$

稳定后

$$
s_\infty
=\frac{\sigma^2}{1-\beta^2}
$$

因此不能简单说「动量一定降低梯度方差」。本文采用的未归一化速度会把历史噪声积累到缓冲区中；$\beta$ 越大，缓冲区的原始方差反而越大，只是高频变化被时间滤波了。

如果使用归一化的指数平均

$$
\boldsymbol m_{k+1}
=\beta\boldsymbol m_k
+(1-\beta)\widehat{\boldsymbol g}_k
$$

同样的独立噪声会满足

$$
\operatorname{Var}(m)_\infty
=\frac{(1-\beta)^2\sigma^2}{1-\beta^2}
=\frac{1-\beta}{1+\beta}\sigma^2
$$

这时缓冲区的数值方差确实下降，但更新式中的学习率尺度也随归一化约定变化。比较不同优化器时，必须同时注明缓冲区定义和学习率缩放。

例如 $\beta=0.9$ 时，未归一化缓冲区的稳态方差是

$$
\frac{\sigma^2}{1-0.9^2}
=5.263157894737\sigma^2
$$

归一化指数平均的稳态方差是

$$
\frac{1-0.9}{1+0.9}\sigma^2
=0.052631578947\sigma^2
$$

这两个数字并不矛盾：它们对应不同的状态变量定义。

## Nesterov 把梯度放到前瞻点

重球法在当前点 $\boldsymbol\theta_k$ 计算梯度，再把历史位移加入更新。Nesterov 加速梯度先沿当前速度看一眼前方：

$$
\boldsymbol y_k
=\boldsymbol\theta_k
+\beta(\boldsymbol\theta_k-\boldsymbol\theta_{k-1})
$$

再在前瞻点计算梯度：

$$
\boldsymbol\theta_{k+1}
=\boldsymbol y_k-\eta\nabla F(\boldsymbol y_k)
$$

两者的差异不在于是否保存历史，而在于梯度评估点：

| 方法 | 梯度在哪一点计算 | 历史方向的作用 |
| --- | --- | --- |
| 普通梯度下降 | $\boldsymbol\theta_k$ | 不保存 |
| 重球法 | $\boldsymbol\theta_k$ | 更新后加上历史位移 |
| Nesterov | 前瞻点 $\boldsymbol y_k$ | 先预测再用梯度修正 |

对凸且光滑目标，适当调度的 Nesterov 方法可以达到

$$
F(\boldsymbol\theta_k)-F(\boldsymbol\theta^\star)
=O\left(\frac{L\|\boldsymbol\theta_0-\boldsymbol\theta^\star\|_2^2}{k^2}\right)
$$

而基础梯度法的函数值率是 $O(1/k)$。强凸情形还可以得到与 $1-\sqrt{\mu/L}$ 相关的线性因子。这里的加速需要配套的参数调度和势函数证明；把一个任意 $\beta$ 加进梯度下降，并不会自动获得 $O(1/k^2)$。

## 如何理解加速和停止

动量方法的优势通常出现在一段连续方向相近的梯度上：历史速度会把这些小步叠加起来。如果梯度方向在狭窄谷底中交替变化，高动量又会把旧方向带到新方向，形成来回穿越。

因此以下信号要分开看：

- 训练目标是否逐步下降：动量允许暂时上升；
- 参数距离是否接近某个驻点：振荡轨迹可能暂时离开；
- 速度范数是否变大：可能是有效累计，也可能是即将失稳；
- 验证集指标是否改善：优化训练目标不等于泛化保证。

初始化 $\boldsymbol v_0=\boldsymbol0$ 时，第一步和普通梯度下降相同，历史效应从第二步开始出现。若从检查点恢复训练而忘记恢复动量缓冲区，参数相同但优化状态不同，后续轨迹也会不同。

## 失效模式

**只把当前梯度乘上一个系数。** 动量的状态是缓冲区递推；没有保存 $\boldsymbol v_k$，就不是同一个算法。

**把重球法和 Nesterov 混写。** 前者在当前位置求梯度，后者在前瞻点求梯度；两个公式的稳定性和参数解释不能互换。

**把稳定区间当成每步下降区间。** $0<\eta\lambda<2(1+\beta)$ 只说明一维二次递推的根可以落在单位圆内，函数值仍可能短暂上升。

**把 $\beta$ 越大理解成越好。** 有效记忆变长会平滑噪声，也会延长错误方向的影响；过大的 $\beta$ 配合较大的 $\eta$ 容易振荡。

**把缓冲区方差和更新噪声混为一谈。** 未归一化速度的原始方差可随 $\beta$ 增大，归一化指数平均的方差却会下降；比较前要先对齐定义。

**把二次目标的最佳参数搬到深层网络。** 重球法公式依赖固定的 $\mu$、$L$ 和不变 Hessian；网络训练中曲率会变化，随机梯度还会带来额外噪声。

**把加速率写成单步损失保证。** $O(1/k^2)$ 是凸问题中带调度的函数值上界，不是每一步都下降，也不是非凸网络的通用结论。

**恢复检查点时丢掉动量缓冲区。** 参数相同不代表优化状态相同；丢失历史速度会造成一次额外的方向变化。

![动量把历史方向带进当前更新](/assets/optimization-theory/svg/momentum-theory.1.svg)

## 相关词条

- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：提供没有历史缓冲区时的下降引理和步长基线。
- [随机梯度下降理论](../optimization-theory/stochastic-gradient-descent-theory/)：解释小批次无偏性、方差和噪声地板。
- [梯度](../calculus/gradient/)：说明当前点最陡下降方向的来源。
- [Hessian 矩阵](../calculus/hessian/)：用曲率和特征值解释动量的稳定性。
- [谱定理](../linear-algebra/spectral-theorem/)：把对称二次目标拆成相互独立的特征方向。
- [二次型](../linear-algebra/quadratic-forms/)：提供二次目标和方向曲率的代数形式。
- [凸集与凸函数](../optimization-theory/convex-sets-and-functions/)：给出凸函数值界和强凸结构。
- [自适应学习率](../optimization-theory/adaptive-learning-rates/)：继续比较按坐标调节步长的优化器。
- [二阶方法](../optimization-theory/second-order-methods/)：比较动量与使用曲率信息的 Newton 类方法。
- [曲率与条件数](../optimization-theory/curvature-and-conditioning/)：解释最小和最大特征值如何限制迭代速度。
