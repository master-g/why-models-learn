---
title: "时间反向传播：沿展开图累加循环网络的梯度"
tags: ["why-models-learn"]
---

时间反向传播（backpropagation through time，BPTT）不是一种和反向传播完全不同的微积分，而是把循环网络沿有限时间步展开后，在这张图上执行反向模式微分。RNN 的状态转移是

$$
h_t=\phi\bigl(W_{xh}x_t+W_{hh}h_{t-1}+b_h\bigr),
$$

BPTT 从末端 loss 往前传递每个状态的敏感度，经过每个时间步的激活 Jacobian 和 hidden-to-hidden 矩阵，并把同一组共享参数在不同时间点产生的梯度相加。

本文先固定列向量约定，逐项推导输出梯度、状态伴随量、局部预激活梯度和三组参数梯度，再用一个线性 RNN 的数值例子与中心差分核对结果。随后比较 full BPTT、truncated BPTT、carry、reset、detach、padding mask 和梯度裁剪的边界，最后给出实现审计协议。[RNN 时间展开](../rnn-lstm/rnn-unrolling/) 负责计算图的节点和索引，[RNN 中的梯度消失](../rnn-lstm/rnn-vanishing-gradient/) 会专门分析长时间 Jacobian 连乘的数值后果。

![BPTT 沿展开的时间链反向传递状态梯度：每步的局部梯度回到同一组共享参数并累加](/assets/rnn-lstm/svg/bptt.1.svg)

## 先固定一条展开链

### 前向方程

对 $t=1,\ldots,T$：

$$
\begin{aligned}
a_t&=W_{xh}x_t+W_{hh}h_{t-1}+b_h,\\
h_t&=\phi(a_t),\\
o_t&=W_{hy}h_t+b_y,\\
\widehat y_t&=\psi(o_t).
\end{aligned}
$$

$h_0$ 是初始状态，$x_t$ 是当前输入，$h_t$ 是传给下一步的状态。若是 next-step 预测，输出可能对应 $y_{t+1}$；为了推导不混淆，先把第 $t$ 步对状态的损失记为 $\mathcal L_t$，并令

$$
\mathcal L=\sum_{t=1}^{T}\mathcal L_t.
$$

many-to-one 只在末端有 loss，可以把 $\mathcal L_t=0$ 对 $t<T$；many-to-many 则多个 $\mathcal L_t$ 同时非零。推导不依赖是哪一种任务，只依赖每个时间步的 loss 是否存在。

### 列向量形状

使用列向量：

$$
x_t\in\mathbb R^{d_x},\qquad
h_t\in\mathbb R^{d_h},\qquad
o_t\in\mathbb R^{d_y}.
$$

参数形状为

$$
\begin{aligned}
W_{xh}&\in\mathbb R^{d_h\times d_x},&
W_{hh}&\in\mathbb R^{d_h\times d_h},&
b_h&\in\mathbb R^{d_h},\\
W_{hy}&\in\mathbb R^{d_y\times d_h},&
b_y&\in\mathbb R^{d_y}.
\end{aligned}
$$

反向公式写成列向量后，参数梯度是外积，状态梯度通过矩阵转置返回。若实现使用 row-batch，代码中的乘法方向会转置，但时间递推和梯度累加的语义不变。

### 缓存什么

完整 BPTT 通常需要保留：

| 缓存 | 反向用途 |
| --- | --- |
| $x_t$ | 形成 $W_{xh}$ 的外积梯度 |
| $h_{t-1}$ | 形成 $W_{hh}$ 的外积梯度 |
| $a_t$ 或 $h_t$ | 计算 $\phi'(a_t)$ |
| $o_t$ 或 logits | 计算输出头的局部梯度 |
| $m_t$ | 屏蔽 padding 和无目标位置 |

如果只保留 $h_T$，通常无法从头重建任意非线性 cell 的所有中间状态；要么重新前向，要么用检查点和重算换内存。

## 局部反向：先处理一个时间步

### 输出节点的梯度

定义输出局部梯度

$$
u_t
=\frac{\partial\mathcal L_t}{\partial o_t}.
$$

因为

$$
o_t=W_{hy}h_t+b_y,
$$

输出头对参数的局部贡献为

$$
\left.\frac{\partial\mathcal L}{\partial W_{hy}}\right\rvert_t
=u_th_t^{\mathsf T},
\qquad
\left.\frac{\partial\mathcal L}{\partial b_y}\right\rvert_t
=u_t.
$$

同时，输出节点把梯度传回状态：

$$
q_t^{\mathrm{out}}
=W_{hy}^{\mathsf T}u_t.
$$

如果第 $t$ 步没有有效 loss，mask 可以令 $u_t=\boldsymbol0$；如果输出头之后还有其它分支，所有分支返回 $h_t$ 的梯度都要相加。

### 状态节点收到两种梯度

$h_t$ 至少有两条出边：

1. 到当前输出 $o_t$；
2. 到下一步预激活 $a_{t+1}$。

因此 $h_t$ 的总敏感度不是只看当前 loss：

$$
q_t
=\frac{\partial\mathcal L}{\partial h_t}
=q_t^{\mathrm{out}}
+W_{hh}^{\mathsf T}r_{t+1},
$$

其中 $r_{t+1}=\partial\mathcal L/\partial a_{t+1}$，末端约定

$$
r_{T+1}=\boldsymbol0.
$$

第一项是当前输出分支传回来的梯度，第二项是未来状态沿 recurrent edge 传回来的梯度。many-to-one 在早期时间步可能没有当前输出分支，但仍然有第二项。

### 激活节点的局部梯度

状态由

$$
h_t=\phi(a_t)
$$

得到。逐分量激活时，令

$$
D_t=\operatorname{diag}\bigl(\phi'(a_t)\bigr),
$$

则

$$
r_t
=\frac{\partial\mathcal L}{\partial a_t}
=D_tq_t
=q_t\odot\phi'(a_t).
$$

这里的 $r_t$ 是该时间步进入线性层之前的误差向量。它把“状态收到的总梯度”乘上当前激活的局部导数；tanh 饱和时，$\phi'(a_t)$ 会让对应坐标的时间梯度变小。

## 参数梯度：每个时间步各贡献一份

### hidden-to-hidden 权重

线性层为

$$
a_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h.
$$

对 $W_{hh}$ 的微分：

$$
da_t=dW_{hh}h_{t-1}+\cdots.
$$

用 Frobenius 内积配对 $r_t$，得到第 $t$ 步的外积贡献：

$$
\left.\frac{\partial\mathcal L}{\partial W_{hh}}\right\rvert_t
=r_th_{t-1}^{\mathsf T}.
$$

因为同一个 $W_{hh}$ 在每个时间步复用，最终梯度是

$$
\frac{\partial\mathcal L}{\partial W_{hh}}
=\sum_{t=1}^{T}r_th_{t-1}^{\mathsf T}.
$$

不是只取最后一步，也不是把 $T$ 份矩阵梯度留在不同参数上。

### input-to-hidden 权重和 bias

同理：

$$
\frac{\partial\mathcal L}{\partial W_{xh}}
=\sum_{t=1}^{T}r_tx_t^{\mathsf T},
$$

以及

$$
\frac{\partial\mathcal L}{\partial b_h}
=\sum_{t=1}^{T}r_t.
$$

bias 在每一步都是同一个向量，所以沿时间做求和；这和 batch 维上的 bias reduction 是同一个局部规则叠加到两个轴。

### 状态梯度返回上一时间步

线性层对旧状态的微分为

$$
da_t=W_{hh}dh_{t-1}+\cdots,
$$

所以

$$
q_{t-1}^{\mathrm{from\ time}}
=W_{hh}^{\mathsf T}r_t.
$$

再把第 $t-1$ 步的输出分支加上，就得到前面的递推：

$$
q_{t-1}
=W_{hy}^{\mathsf T}u_{t-1}
+W_{hh}^{\mathsf T}r_t.
$$

这个转置乘法是最容易在 row/column 约定转换时写反的地方。用一个非对称 $W_{hh}$ 做数值检查，比用对称矩阵更能暴露方向错误。

## 反向递推算法

### 从 $T$ 走到 $1$

先把参数梯度清零，令 $r_{T+1}=\boldsymbol0$。然后对 $t=T,T-1,\ldots,1$：

$$
\begin{aligned}
q_t&=W_{hy}^{\mathsf T}u_t+W_{hh}^{\mathsf T}r_{t+1},\\
r_t&=q_t\odot\phi'(a_t),\\
G_{W_{hy}}&\mathrel{+}=u_th_t^{\mathsf T},\\
G_{b_y}&\mathrel{+}=u_t,\\
G_{W_{hh}}&\mathrel{+}=r_th_{t-1}^{\mathsf T},\\
G_{W_{xh}}&\mathrel{+}=r_tx_t^{\mathsf T},\\
G_{b_h}&\mathrel{+}=r_t.
\end{aligned}
$$

然后把 $r_t$ 通过 hidden-to-hidden 边传给更早状态；下一次循环会用 $W_{hh}^{\mathsf T}r_t$ 形成 $q_{t-1}$。公式中的 $\mathrel{+}=$ 不是一种新的导数，而是强调梯度累加。

### 为什么要先算总状态梯度

如果只用 $W_{hy}^{\mathsf T}u_t$ 计算 $q_t$，就漏掉未来输出对当前状态的影响。反过来，如果只沿 recurrent edge 回传，又会漏掉当前时间步的输出 loss。状态节点是汇合点，反向也必须在这里把各出边贡献相加。

### 输出头也共享时如何处理

若每个时间步都调用同一个 $W_{hy}$，它也要累加：

$$
G_{W_{hy}}
=\sum_{t=1}^{T}u_th_t^{\mathsf T},
\qquad
G_{b_y}
=\sum_{t=1}^{T}u_t.
$$

若 many-to-one 只在 $T$ 有输出，则除 $t=T$ 外的 $u_t$ 为空，但 $W_{hh}$ 和 $W_{xh}$ 仍可能从末端 loss 收到每一步的梯度。

## 数值核验：手算和中心差分对齐

### 线性三步 RNN

使用

$$
h_t=ah_{t-1}+w_xx_t+b,
\qquad
h_0=0,
$$

其中

$$
a=0.5,\qquad
w_x=1,\qquad
b=0,\qquad
(x_1,x_2,x_3)=(1,2,0).
$$

前向为

$$
(h_1,h_2,h_3)=(1,2.5,1.25).
$$

令输出就是 $o_3=h_3$，目标为 $y_3=1.2$：

$$
\mathcal L=\frac12(h_3-y_3)^2
=\frac12(1.25-1.2)^2
=0.00125.
$$

因为这里只有末端 loss：

$$
r_3=\frac{\partial\mathcal L}{\partial h_3}=0.05.
$$

线性激活的导数为 $1$，从后往前：

$$
\begin{aligned}
r_2&=a r_3=0.025,\\
r_1&=a r_2=0.0125.
\end{aligned}
$$

状态伴随量的完整序列可以列为

$$
(q_0,q_1,q_2,q_3)
=(0.00625,0.0125,0.025,0.05).
$$

参数梯度：

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial a}
&=\sum_{t=1}^{3}r_th_{t-1}
=0.0125\times0+0.025\times1+0.05\times2.5
=0.15,\\
\frac{\partial\mathcal L}{\partial w_x}
&=\sum_{t=1}^{3}r_tx_t
=0.0125\times1+0.025\times2+0.05\times0
=0.0625,\\
\frac{\partial\mathcal L}{\partial b}
&=\sum_{t=1}^{3}r_t
=0.0875.
\end{aligned}
$$

### 中心差分复核

对标量参数用

$$
\frac{\partial\mathcal L}{\partial\theta}
\approx
\frac{\mathcal L(\theta+\varepsilon)-\mathcal L(\theta-\varepsilon)}
{2\varepsilon}.
$$

取 $\varepsilon=10^{-6}$，独立重新前向得到：

| 参数 | BPTT 梯度 | 中心差分 | 最大绝对差 |
| --- | ---: | ---: | ---: |
| $a$ | $0.150000000000$ | $0.150000000007$ | $7.24\times10^{-12}$ |
| $w_x$ | $0.062500000000$ | $0.062499999998$ | $2.32\times10^{-12}$ |
| $b$ | $0.087500000000$ | $0.087500000001$ | $1.11\times10^{-12}$ |

这次检查验证的是三件事：状态梯度确实沿时间返回、共享参数梯度确实累加、前向和反向使用的是同一个 loss 归约。它不能证明任意框架实现都正确，但能把最小公式链锁定。

## full BPTT 与 truncated BPTT

### Full BPTT

完整 BPTT 让梯度穿过整条长度为 $T$ 的状态链：

$$
\frac{\partial h_T}{\partial h_s}
=
\prod_{t=s+1}^{T}
\frac{\partial h_t}{\partial h_{t-1}}.
$$

它保留完整的长期依赖梯度，代价是反向图深度和激活内存随 $T$ 增加。full 不表示梯度一定健康，只表示没有人为在时间边界切断。

### Truncated BPTT

把序列切成窗口长度 $K$：

$$
[1,T]
=
[1,K]\cup[K+1,2K]\cup\cdots.
$$

前向可以携带上一窗口的状态，但在窗口边界 detach：

$$
h_{j,0}
=\operatorname{detach}(h_{j-1,K}).
$$

这样每次反向只穿过 $K$ 个时间步。代价是跨窗口的 loss 不再对更早状态和参数提供完整梯度；前向仍然可能让远处信息影响当前状态，训练信号却被人为截断。

| 方案 | 前向历史 | 反向历史 | 资源 |
| --- | --- | --- | --- |
| full BPTT | 全序列 | 全序列 | 内存和时间随 $T$ 增加 |
| truncated BPTT | 可跨窗口 carry | 每窗最多 $K$ 步 | 近似按 $K$ 控制图深度 |
| reset + truncated | 每窗重新开始 | 每窗独立 | 丢失跨窗口前向信息 |

“截断”与“重置”不能互换。前者改变梯度图，后者改变状态轨迹。

### 截断窗口不是无偏替代

对共享参数 $\theta$，full 梯度包含

$$
\frac{\partial\mathcal L_j}{\partial\theta}
=
\sum_{s\le t\in\text{window }j}
\frac{\partial\mathcal L_j}{\partial h_t}
\frac{\partial h_t}{\partial h_s}
\frac{\partial h_s}{\partial\theta},
$$

以及跨越窗口边界的更早路径。detach 后这些路径被置零，因此 TBPTT 是一种有意的近似训练目标，而不是只改变实现方式。窗口长度应作为实验配置记录，不能在不同实验间静默变化。

## mask、变长与反向边界

### loss mask 会改变反向源头

设每步有效 mask 为 $m_t$：

$$
\mathcal L=\sum_{t=1}^{T}m_t\ell_t.
$$

若 $m_t=0$，对应输出源梯度为

$$
u_t
=m_t\frac{\partial\ell_t}{\partial o_t}
=\boldsymbol0.
$$

但这不一定意味着状态更新没有发生。padding 仍可能通过 $h_t$ 传给未来状态；如果任务要求 padding 不改变状态，还需在前向或状态边上使用 mask。

### many-to-one 的末状态

变长样本 $b$ 的有效末状态是

$$
h_{T_b}^{(b)},
$$

对应的 BPTT 只应从这个实际读取点回到有效前缀。如果错误地从 padding 后的统一末端 $h_{T_{\max}}^{(b)}$ 开始，梯度会沿不存在的时间步回传。

### 反向 mask 的三层检查

| 层次 | 要检查的对象 | 典型结果 |
| --- | --- | --- |
| 输出源 | $u_t$ | padding 和无 target 位置为零 |
| 状态边 | $q_t$ 到 $q_{t-1}$ | 是否允许 padding 状态继续传梯度 |
| 参数累加 | $r_th_{t-1}^{\mathsf T}$ | 只累加有效时间位置或明确的状态路径 |

不同任务对第二层的选择可能不同，但必须是明确选择。只在 loss 末端乘 mask，不能自动修复前向状态已被 padding 改写的事实。

## 梯度裁剪与数值保护

### 裁剪发生在 BPTT 之后

BPTT 先得到完整参数梯度 $G$，然后才可以进行全局范数裁剪：

$$
c=\min\left(1,\frac{\tau}{\lVert G\rVert_2+\varepsilon}\right),
\qquad
\widetilde G=cG.
$$

裁剪限制的是参数更新使用的梯度幅度，不会改变已经计算出的局部状态 Jacobian，也不能把消失的梯度放大回来。[梯度裁剪](../training-nn/gradient-clipping/) 会进一步比较 global norm、coordinate clipping 和自适应规则。

### 何时记录裁剪前后的量

至少记录：

$$
\lVert G\rVert_2,\qquad
\lVert\widetilde G\rVert_2,\qquad
c,\qquad
\lVert\Delta\theta\rVert_2.
$$

如果每一步都发生裁剪，说明模型在当前长度、初始化、输入尺度或学习率下频繁产生大梯度；不能只报告训练 loss 还能下降。

## 计算与内存账本

### 时间复杂度

对普通 RNN，每一步大致包含

$$
W_{xh}x_t,\qquad
W_{hh}h_{t-1},\qquad
W_{hy}h_t.
$$

若 batch 为 $B$，序列长度为 $T$，隐藏维为 $d_h$，输入维为 $d_x$，前向矩阵乘加量级近似为

$$
O\bigl(TB(d_hd_x+d_h^2+d_yd_h)\bigr).
$$

BPTT 也需要按 $T$ 访问每个缓存，量级同阶；主要差异是反向不能并行跳过时间依赖。

### 内存复杂度

完整反向通常保存 $T$ 个时间步的状态和激活：

$$
O(BTd_h)
$$

是状态部分的主项。若使用窗口 $K$ 的 TBPTT，单个反向图的主项可近似变成

$$
O(BKd_h),
$$

但还要加上参数、输入窗口、输出和优化器状态。若前向状态跨窗口 carry，数值历史仍然存在，只是旧计算图被 detach。

### 重算换内存

检查点方法可以只保留少数状态，在反向到达一个检查点时重新前向中间区间。它减少保存的激活，但增加前向计算。无论使用保存、重算还是截断，都要说明：

| 选择 | 减少什么 | 增加什么 |
| --- | --- | --- |
| full cache | 重新计算 | 激活内存 |
| checkpoint | 激活内存 | 重算时间和实现复杂度 |
| TBPTT | 单图时间深度 | 跨窗口梯度信息 |
| reset | 状态和图边界 | 跨段前向依赖 |

## 失效模式：反向公式对了，训练仍可能错

### 只回传 recurrent edge

如果公式只有

$$
q_t=W_{hh}^{\mathsf T}r_{t+1},
$$

就漏掉当前输出 $W_{hy}^{\mathsf T}u_t$。many-to-many 任务会因此丢失每步的直接监督，many-to-one 仍可能在末端看似正常。

### 只算当前输出梯度

如果公式只有

$$
q_t=W_{hy}^{\mathsf T}u_t,
$$

就把未来 loss 对当前状态的影响切掉，等价于在每个时间步都隐式 detach。短序列可能看不出差异，长依赖会失效。

### 参数梯度没有按时间累加

只保存最后一步的

$$
r_Th_{T-1}^{\mathsf T}
$$

会丢掉早期时间步对共享参数的贡献。检查方法是分别只启用第 $1$ 个和第 $T$ 个 loss，确认两个实验都能改变同一个参数梯度。

### 反向使用了错误版本的状态

若前向循环中 inplace 覆盖了 $h_{t-1}$，反向可能读到 $h_t$ 或最后一次状态。每个时间步的缓存必须保持不可混淆的版本，或者使用框架明确的 checkpoint/recompute 机制。

### 梯度检查没有固定状态

中心差分的正负扰动必须使用同一个输入、同一个 $h_0$、同一个 mask、同一个随机状态和同一个 loss 分母。若 dropout、采样或数据增强在正负两次前向中不同，差分误差不能用来判断 BPTT。

### 只看总梯度，不看时间分布

总梯度可能在时间上发生抵消：

$$
\sum_tG_t\approx0
$$

不代表每个 $G_t$ 都小。调试时可记录按时间的 $\lVert r_t\rVert_2$、按参数组的外积范数和累计和，区分消失、爆炸与符号抵消。

## BPTT 审计协议

1. **固定约定。** 写出列向量或 row-batch、$h_0$、输入时间轴和 target 对齐。
2. **固定前向缓存。** 至少保留 $x_t,h_{t-1},a_t,h_t$ 和有效 mask。
3. **先核对局部梯度。** 对一个 cell 分别检查输出头、激活、hidden-to-hidden 三条局部规则。
4. **再核对时间递推。** 从 $T$ 到 $1$ 打印 $q_t,r_t$，确认当前输出和未来状态都进入 $q_t$。
5. **核对参数求和。** 确认 $W_{xh},W_{hh},b_h,W_{hy},b_y$ 都按时间累加。
6. **做小数值中心差分。** 固定随机状态和 mask，至少核对 recurrent weight、input weight 与 bias。
7. **分离 full 与 truncated。** 记录窗口长度、carry/reset、detach 边界和有效 token 分母。
8. **记录梯度分布。** 同时记录总范数、时间分位数、非有限值首次位置和裁剪系数。
9. **检查资源边界。** 报告 $BTd_h$ 激活量级、重算策略、序列长度与并行/串行部分。

BPTT 的“through time”是一个具体的图结构事实：早期状态通过许多 recurrent edge 影响未来 loss，共享参数在不同时间节点收到多份局部贡献。把这两点写进数据、缓存和梯度账本，才有可能判断一个实现是在训练 RNN，还是只是在循环里反复调用了一个前馈层。

## 相关词条

[RNN 时间展开](../rnn-lstm/rnn-unrolling/)

[循环神经网络](../rnn-lstm/rnn/)

[序列建模](../rnn-lstm/sequence-modeling/)

[RNN 中的梯度消失](../rnn-lstm/rnn-vanishing-gradient/)

[梯度检查](../backpropagation/gradient-checking/)

[梯度裁剪](../training-nn/gradient-clipping/)

[训练调试](../training-nn/debugging-training/)

[长短期记忆网络](../rnn-lstm/lstm/)

[门控循环单元](../rnn-lstm/gru/)
