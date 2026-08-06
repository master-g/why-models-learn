---
title: "GRU：把保留与重写压缩到一个隐藏状态"
---

门控循环单元（Gated Recurrent Unit，GRU）把 LSTM 的显式细胞状态和隐藏状态合并为一个 $h_t$，用更新门 $z_t$ 在旧状态与候选状态之间插值，用重置门 $r_t$ 决定候选状态读取多少旧 hidden。它的核心不是「门更少所以一定更好」，而是用更少的状态变量换取一条仍然可学习的保留路径。

本篇固定如下约定：$z_t$ 表示写入候选状态的比例，因此

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\widetilde h_t.
$$

有些资料把 $z_t$ 定义为旧状态的保留比例，公式会写成

$$
h_t=z_t\odot h_{t-1}+(1-z_t)\odot\widetilde h_t.
$$

两者可以互相变换，但不能把门名和公式分开搬用。本文所有「$z$ 大」都表示「更倾向写入候选内容」。

GRU 的学习重点可以拆成四个问题：

1. 更新门是否真的提供了接近恒等的旧状态路径；
2. 重置门是否在需要时阻断旧状态对候选内容的影响；
3. 候选状态的 tanh 和两个 sigmoid 是否饱和；
4. 实现采用的门顺序、reset 放置位置和参数宽度是否与公式一致。

![GRU 的更新门、重置门与隐藏状态插值](/assets/rnn-lstm/svg/gru.1.svg)

## 两个门和一个候选状态

给定 $x_t\in\mathbb R^{d_x}$、$h_{t-1}\in\mathbb R^{d_h}$，采用 reset-before-matrix 的常见形式：

$$
\begin{aligned}
z_t&=\sigma(W_{xz}x_t+W_{hz}h_{t-1}+b_z),\\
r_t&=\sigma(W_{xr}x_t+W_{hr}h_{t-1}+b_r),\\
\widetilde h_t&=\tanh\bigl(W_{xh}x_t+W_{hh}(r_t\odot h_{t-1})+b_h\bigr),\\
h_t&=(1-z_t)\odot h_{t-1}+z_t\odot\widetilde h_t.
\end{aligned}
$$

各变量的职责如下：

| 变量 | 形状 | 数值范围 | 作用 |
| --- | --- | --- | --- |
| $z_t$ | $d_h$ | $(0,1)$ | 候选状态写入比例，$1-z_t$ 是旧 hidden 的直接保留比例 |
| $r_t$ | $d_h$ | $(0,1)$ | 候选状态读取旧 hidden 的比例 |
| $\widetilde h_t$ | $d_h$ | $(-1,1)$ | 当前输入和受 reset 调制的旧 hidden 生成的候选 |
| $h_t$ | $d_h$ | $(-1,1)$ | 旧 hidden 与候选的门控插值结果 |

和 LSTM 不同，GRU 没有单独的 $c_t$。因此「保存但暂不暴露」的状态仍然必须存在 $h_t$ 中；更新门既控制记忆保留，也控制候选内容何时对后续步骤可见。

### 更新门的两个读法

在本文约定中：

$$
z_t\approx0\Rightarrow h_t\approx h_{t-1},
\qquad
z_t\approx1\Rightarrow h_t\approx\widetilde h_t.
$$

这使 $1-z_t$ 直接出现在 cell-like carry 路径中。若阅读另一种 convention，应先定义

$$
z_t^{\mathrm{keep}}=1-z_t^{\mathrm{write}}
$$

再比较公式。只在文字上把两个版本都叫「update gate」，会让初始化、可视化和失效诊断全部反转。

## 参数账：两门加一个候选仿射

四组仿射变换的权重形状为

$$
\begin{aligned}
W_{xz},W_{xr},W_{xh}&\in\mathbb R^{d_h\times d_x},\\
W_{hz},W_{hr},W_{hh}&\in\mathbb R^{d_h\times d_h},\\
b_z,b_r,b_h&\in\mathbb R^{d_h}.
\end{aligned}
$$

总参数量为

$$
3(d_hd_x+d_h^2+d_h)
=3d_h(d_x+d_h+1).
$$

以 $d_x=3,d_h=4$ 为例：

$$
3(4\times3+4\times4+4)=96.
$$

接一个 $d_y=2$ 的线性输出头，需要 $2\times4+2=10$ 个参数，总数为 106。

| 结构 | 循环层参数量 | 输出头参数量 | 总参数量 |
| --- | ---: | ---: | ---: |
| vanilla RNN，$d_x=3,d_h=4,d_y=2$ | 32 | 10 | 42 |
| GRU，两个门加候选，$d_x=3,d_h=4,d_y=2$ | 96 | 10 | 106 |
| LSTM，四门，$d_x=3,d_h=4,d_y=2$ | 128 | 10 | 138 |

GRU 比 LSTM 少一组门，因此在相同 hidden width 下参数和乘加量通常更小；但公平比较还要说明 hidden width、projection、层数、双向方向和输出头是否相同。

### 融合矩阵

实现可以把三组 input-to-hidden 仿射合并：

$$
\begin{bmatrix}
a^z_t\\
a^r_t\\
a^h_t
\end{bmatrix}
=
W_xx_t+W_hh_{t-1}+b.
$$

这只适用于候选仿射仍然直接使用 $h_{t-1}$ 的布局。reset-before-matrix 的候选项需要在 $a^h_t$ 的 recurrent 输入处使用 $r_t\odot h_{t-1}$，不能把同一个未调制的 $W_{hh}h_{t-1}$ 当作所有变体的候选输入。

实际 kernel 可能采用不同的拼接顺序，例如 $(z,r,h)$ 或 $(r,z,h)$。参数量不变，门的含义会变；固定输入的单元级测试必须覆盖顺序。

## 更新门提供的插值路径

暂时把 $z_t$ 和 $\widetilde h_t$ 当作已算好的常数，只看旧 hidden 直接进入当前 hidden 的路径：

$$
\left.\frac{\partial h_t}{\partial h_{t-1}}\right\rvert_{z_t,\widetilde h_t}
=\operatorname{diag}(1-z_t).
$$

连续多步的直接 carry 增益是

$$
\left.\frac{\partial h_T}{\partial h_s}\right\rvert_{\mathrm{carry}}
=
\prod_{t=s+1}^{T}\operatorname{diag}(1-z_t).
$$

逐坐标写为

$$
\frac{\partial h_T^{(j)}}{\partial h_s^{(j)}}\bigg\rvert_{\mathrm{carry}}
=
\prod_{t=s+1}^{T}(1-z_t^{(j)}).
$$

因此本文约定下：

| 更新门 | 旧状态直接路径 | 候选状态写入 | 典型语义 |
| --- | --- | --- | --- |
| $z_t\approx0$ | 接近 1 | 很少 | 保留旧 hidden |
| $z_t\approx0.5$ | 约一半 | 约一半 | 平滑混合 |
| $z_t\approx1$ | 接近 0 | 接近全部 | 重写 hidden |

这条路径比 vanilla RNN 每一步都经过一个完整 tanh 的形式更容易保留，但它只是直接项。$z_t$ 和 $\widetilde h_t$ 依赖 $h_{t-1}$，完整状态 Jacobian 还包含门的间接导数。

### 完整 Jacobian 的分解

令

$$
q_t=W_{xh}x_t+W_{hh}(r_t\odot h_{t-1})+b_h,
\qquad
\widetilde h_t=\tanh(q_t).
$$

对 $h_t$ 求微分：

$$
dh_t
=
(1-z_t)\odot dh_{t-1}
+(\widetilde h_t-h_{t-1})\odot dz_t
+z_t\odot d\widetilde h_t.
$$

其中

$$
d\widetilde h_t
=
\operatorname{diag}(1-\widetilde h_t^2)
\left[
W_{hh}\,d(r_t\odot h_{t-1})
\text{输入项}
\right].
$$

而

$$
d(r_t\odot h_{t-1})
=
\operatorname{diag}(r_t)dh_{t-1}
+\operatorname{diag}(h_{t-1})dr_t.
$$

所以完整 Jacobian 包含四类来源：

1. 旧 hidden 的直接 carry；
2. 更新门改变插值比例；
3. reset gate 改变候选的 recurrent 输入；
4. 候选 tanh 和候选仿射的 recurrent 路径。

LSTM 的 cell highway 与 GRU 的 update carry 都是解释长程梯度的显式路径，但都不能替代完整的数值梯度检查。

## 三步标量前向账本

固定一维 hidden，令 $h_0=0.6$，并直接给出每一步已经算好的门和候选值：

$$
h_t=(1-z_t)h_{t-1}+z_t\widetilde h_t.
$$

| 时间步 | $h_{t-1}$ | $z_t$ | $r_t$ | $\widetilde h_t$ | $h_t$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.600000000000 | 0.20 | 0.50 | 0.80 | 0.640000000000 |
| 2 | 0.640000000000 | 0.70 | 0.90 | −0.20 | 0.052000000000 |
| 3 | 0.052000000000 | 0.10 | 0.20 | 0.50 | 0.096800000000 |

第一步是

$$
h_1=0.8\times0.6+0.2\times0.8=0.64.
$$

第二步中 $z_2=0.7$，所以候选负值会大幅重写旧状态：

$$
h_2=0.3\times0.64+0.7\times(-0.2)=0.052.
$$

第三步又主要保留 $h_2$：

$$
h_3=0.9\times0.052+0.1\times0.5=0.0968.
$$

直接 carry 从 $h_0$ 到 $h_3$ 的系数为

$$
(1-0.2)(1-0.7)(1-0.1)
=0.8\times0.3\times0.9=0.216.
$$

它说明第三步的输出仍可沿旧状态路径收到约 21.6% 的局部信号；完整梯度还要加上门值和候选值依赖旧 hidden 的分支。

### Reset gate 的反事实

在候选预激活中，旧 hidden 通过

$$
W_{hh}(r_t\odot h_{t-1})
$$

进入。若 $r_t=0$，候选不直接读取旧 hidden；若 $r_t=1$，候选完整读取它。注意这不是把当前 $h_t$ 重置为 0，也不是把 update gate 设为 0。reset 只改变候选的输入路径。

## 通过两个门的 BPTT

用上划线表示损失梯度：

$$
\bar h_t=\frac{\partial\mathcal L}{\partial h_t}.
$$

从插值式先得到三项局部梯度：

$$
\begin{aligned}
\bar h_{t-1}^{\mathrm{direct}}&=\bar h_t\odot(1-z_t),\\
\bar z_t&=\bar h_t\odot(\widetilde h_t-h_{t-1}),\\
\bar{\widetilde h}_t&=\bar h_t\odot z_t.
\end{aligned}
$$

更新门的预激活梯度为

$$
\delta^z_t=\bar z_t\odot z_t\odot(1-z_t).
$$

候选 tanh 的局部梯度为

$$
\delta^h_t
=
\bar{\widetilde h}_t\odot(1-\widetilde h_t^2).
$$

令候选预激活为

$$
q_t=W_{xh}x_t+W_{hh}(r_t\odot h_{t-1})+b_h.
$$

则候选路径先把梯度传回 reset 调制后的 recurrent 输入：

$$
\bar v_t=W_{hh}^{\mathsf T}\delta^h_t,
\qquad
v_t=r_t\odot h_{t-1}.
$$

于是

$$
\begin{aligned}
\bar r_t&=\bar v_t\odot h_{t-1},\\
\bar h_{t-1}^{\mathrm{candidate}}&=\bar v_t\odot r_t.
\end{aligned}
$$

reset gate 的预激活梯度为

$$
\delta^r_t=\bar r_t\odot r_t\odot(1-r_t).
$$

最后，两个 gate 的 recurrent 仿射层也把梯度传回上一 hidden：

$$
\bar h_{t-1}
=
\bar h_{t-1}^{\mathrm{direct}}
+\bar h_{t-1}^{\mathrm{candidate}}
+W_{hz}^{\mathsf T}\delta^z_t
+W_{hr}^{\mathsf T}\delta^r_t.
$$

参数梯度按时间步累加：

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial W_{xz}}
&\mathrel{+}=\delta^z_tx_t^{\mathsf T},&
\frac{\partial\mathcal L}{\partial W_{hz}}
&\mathrel{+}=\delta^z_th_{t-1}^{\mathsf T},\\
\frac{\partial\mathcal L}{\partial W_{xr}}
&\mathrel{+}=\delta^r_tx_t^{\mathsf T},&
\frac{\partial\mathcal L}{\partial W_{hr}}
&\mathrel{+}=\delta^r_th_{t-1}^{\mathsf T},\\
\frac{\partial\mathcal L}{\partial W_{xh}}
&\mathrel{+}=\delta^h_tx_t^{\mathsf T},&
\frac{\partial\mathcal L}{\partial W_{hh}}
&\mathrel{+}=\delta^h_t(r_t\odot h_{t-1})^{\mathsf T}.
\end{aligned}
$$

偏置梯度分别累加 $\delta^z_t,\delta^r_t,\delta^h_t$。候选 recurrent 外积使用的是 $r_t\odot h_{t-1}$，不是未经 reset 的 $h_{t-1}$；这是 GRU 参数梯度最容易漏掉的一个细节。

### 第二步的局部数值

用前向账本的第二步，取 $\bar h_2=1$，先只算插值式：

$$
\bar h_1^{\mathrm{direct}}=1-z_2=0.3,
\qquad
\bar z_2=\widetilde h_2-h_1=-0.2-0.64=-0.84,
\qquad
\bar{\widetilde h}_2=z_2=0.7.
$$

因此

$$
\delta^z_2=-0.84\times0.7\times0.3=-0.1764.
$$

候选 tanh 的局部导数给出

$$
\delta^h_2
=0.7(1-(-0.2)^2)=0.672.
$$

这两个数还没有包含 reset recurrent 权重产生的 $\bar h_1^{\mathrm{candidate}}$ 和 update/reset gate 的 recurrent 仿射回传。它们的用途是先隔离插值和激活两个局部规则，再逐步接回完整路径。

## Reset 放置位置：看似相近，导数不同

GRU 文献和框架中常见两类候选计算：

| 形式 | 候选预激活 | reset 作用的位置 | 审计重点 |
| --- | --- | --- | --- |
| reset-before-matrix | $W_{xh}x_t+W_{hh}(r_t\odot h_{t-1})+b_h$ | 先调制 hidden，再乘 recurrent 权重 | 参数梯度外积用 $r_t\odot h_{t-1}$ |
| reset-after-matrix | $W_{xh}x_t+r_t\odot(W_{hh}h_{t-1})+b_h$ | 先乘 recurrent 权重，再逐坐标调制 | reset 的维度和梯度顺序不同 |

两者在标量或特殊对角矩阵情形下可能看起来相同，但一般矩阵下

$$
W_{hh}(r_t\odot h_{t-1})
\ne
r_t\odot(W_{hh}h_{t-1}).
$$

写文章、复现论文或比较框架时，必须记录 reset 的放置位置。只说「使用 GRU」不足以确定完整计算图。

## GRU 与 LSTM：少一个状态变量的交换

| 比较项 | GRU | LSTM |
| --- | --- | --- |
| 显式状态 | 只有 $h_t$ | $c_t$ 与 $h_t$ |
| 门 | update $z_t$、reset $r_t$ | input $i_t$、forget $f_t$、output $o_t$ |
| 直接保留路径 | $(1-z_t)\odot h_{t-1}$ | $f_t\odot c_{t-1}$ |
| 新内容 | $\widetilde h_t$ 与旧 hidden 插值 | 候选内容写入 cell，再由 output 暴露 |
| 参数组 | 三组仿射 | 四组仿射 |
| 状态可见性 | 记忆和输出合在 hidden | cell 可以保留而暂不暴露 |
| 结构代价 | 通常较低 | 通常较高 |
| 关键风险 | update convention、reset 位置、门饱和 | cell/hidden 混淆、四门顺序、state 边界 |

GRU 的合并状态更紧凑，LSTM 的分离状态更容易描述「保留但不输出」。这不是一个无条件的优劣排序；应在相同参数预算、序列长度、训练稳定性和目标延迟下比较。

## 训练与序列边界

### Gate bias

在本文 convention 中，若希望初始时多保留旧状态，应让 $z_t$ 偏小，而不是机械地把名为 update gate 的 bias 设成正数。若另一实现把 $z_t$ 定义为 keep ratio，结论会反过来。初始化建议必须绑定公式 convention。

### 梯度裁剪

GRU 仍可能在候选 recurrent 路径、门的 recurrent 仿射和 stacked layers 中产生爆炸。全局范数裁剪为

$$
g\leftarrow
g\min\left(1,\frac{\tau}{\lVert g\rVert_2+\varepsilon}\right).
$$

它控制异常大更新，但不会修复 $z_t$ 饱和为 1 后的直接 carry 消失，也不会让 reset 重新打开被切断的候选路径。

### Dropout 和 recurrent mask

输入到门的 dropout、候选内容 dropout、每个时间步重新采样 recurrent mask 和一个序列内锁定的 variational mask，会改变不同的时间图。报告 GRU 结果时要写明：

1. mask 作用于 $x_t$、$h_{t-1}$ 还是候选值；
2. mask 是否在时间步间复用；
3. 评估态是否关闭；
4. state carry 是否跨 chunk。

### 变长与 reset

对变长 batch，padding 位置不应推进有效 hidden；如果使用 mask，则要同时屏蔽输出 loss 和状态更新。如果一个样本结束后没有 reset，下一条样本可能从错误的 $h_T$ 开始。GRU 没有独立 cell，状态泄漏会直接出现在唯一的 hidden 中，排查时尤其容易被误认为模型的长期记忆。

| 边界操作 | hidden 前向 | hidden 反向 | 语义 |
| --- | --- | --- | --- |
| carry | 跨窗口传递 | 可跨窗口回传 | 连续流 |
| carry + detach | 跨窗口传递 | 截断 | TBPTT |
| reset | 重新初始化 | 不跨边界 | 独立样本或 episode |

## 失效模式：少一个门不等于少一类错误

### 把 $z$ 的方向写反

如果实现公式是

$$
h_t=(1-z_t)h_{t-1}+z_t\widetilde h_t
$$

却按 $z_t$ 大表示保留旧状态来解释，所有 gate histogram、bias 初始化和时间视野结论都会反转。先从单步输出做反事实测试：固定 $\widetilde h_t\ne h_{t-1}$，增大 $z_t$，观察 $h_t$ 是靠近谁。

### Reset 影响了错误的路径

reset gate 只应改变候选状态如何读取旧 hidden。把它误乘到 update 插值或直接从最终 hidden 上清零，会得到另一个模型。

### 候选 recurrent 外积漏掉 reset

候选权重梯度应使用

$$
\delta^h_t(r_t\odot h_{t-1})^{\mathsf T}.
$$

若代码保存了未经 reset 的 $h_{t-1}$，前向数值可能仍然正确，参数梯度却会错。

### 忽略门的 sigmoid 饱和

$z_t$ 长期贴近 0 或 1 时，$z_t(1-z_t)$ 接近 0，更新门本身对其 preactivation 的梯度消失。直接 carry 可能很好，但门已经难以改变。应同时看门值和门的局部导数。

### 只看 hidden 范数

hidden 被 tanh 候选和插值约束在较小范围，并不表示候选 preactivation、gate logits 或参数梯度健康。至少记录 $a^z_t,a^r_t,a^h_t$、$z_t,r_t$ 和按时间的梯度。

### Batch 内状态泄漏

交换 batch 顺序、显式 reset，并比较同一条样本的输出。如果结果随 batch 排列变化，优先检查 hidden 初始化、mask 和 detach，而不是先调学习率。

### 公平比较遗漏宽度和方向

GRU 少一组门并不意味着在同一 hidden width、同一层数和同一双向设置下就一定公平。记录循环参数、输出头、projection、层数、方向数和训练 token 数。

## GRU 审计协议

1. **写出 convention。** 明确 $z$ 是写入比例还是保留比例，并写出完整插值公式。
2. **固定 reset 位置。** 区分 reset-before-matrix 与 reset-after-matrix，记录候选 recurrent 输入。
3. **核对参数量。** 逐组计算 $W_x,W_h,b$，再加输出头、projection、层数和方向。
4. **做单步反事实。** 固定旧 hidden 和候选，增大 $z$，检查输出是否靠近候选；固定其他项，改变 $r$，检查候选是否改变。
5. **做三步手算。** 复算 $h_t=(1-z_t)h_{t-1}+z_t\widetilde h_t$ 和直接 carry 乘积。
6. **分层核对 BPTT。** 先检查 direct、$z$、candidate tanh，再加入 reset 和两组 recurrent affine。
7. **检查候选参数外积。** 确认 recurrent 输入是 $r_t\odot h_{t-1}$ 或实现所声明的另一种形式。
8. **检查门饱和。** 记录 gate 值、gate derivative、候选 preactivation、hidden 范数和梯度范数。
9. **检查边界。** 写出 padding mask、packed sequence、carry/reset/detach 和 loss 分母。
10. **做数值梯度检查。** 对两个门、候选的 input/recurrent 权重和 bias 做中心差分，固定状态与随机 mask。

GRU 的简洁来自把「保留/重写」压缩成 hidden 上的一次插值。更新门提供直接 carry，重置门控制候选读取旧状态；但门值依赖、候选 tanh 和实现 convention 仍决定完整梯度。理解这三条路径后，GRU 就不再是需要背诵的门公式，而是一个可以逐项核对的时间计算图。

## 相关词条

[长短期记忆网络](../rnn-lstm/lstm/)

[RNN 中的梯度消失](../rnn-lstm/rnn-vanishing-gradient/)

[时间反向传播](../rnn-lstm/bptt/)

[循环神经网络](../rnn-lstm/rnn/)

[RNN 时间展开](../rnn-lstm/rnn-unrolling/)

[序列建模](../rnn-lstm/sequence-modeling/)

[梯度裁剪](../training-nn/gradient-clipping/)

[梯度消失与爆炸](../backpropagation/vanishing-and-exploding/)

[双向循环网络](../rnn-lstm/bidirectional-rnn/)
