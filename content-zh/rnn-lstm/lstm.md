---
title: "LSTM：用门控细胞状态保留长期梯度"
---

长短期记忆网络（Long Short-Term Memory，LSTM）是在循环网络中显式维护两种状态：暴露给输出的隐藏状态 $h_t$，以及承担长期信息通路的细胞状态 $c_t$。四个门控制哪些旧内容被保留、哪些新内容被写入、哪些内容被暴露。它解决梯度消失的关键，不是让所有矩阵乘积都变成 1，而是提供一条可以在许多时间步中接近恒等传递的 cell state 路径。

这条路径也有边界。遗忘门长期偏小会主动擦除历史，门的 sigmoid 仍会饱和，完整 Jacobian 仍包含门对隐藏状态的依赖，padding、detach、状态复用和 loss reduction 仍可能改变训练语义。本篇沿着一条细胞状态通路展开：

1. 先写出四门和 $c_t,h_t$ 的形状账；
2. 再把直接 carry 梯度与完整状态 Jacobian 区分开；
3. 用手工门值做三步前向和一小段 BPTT 数值核验；
4. 最后比较参数化、初始化、正则化、变长序列和失效模式。

![LSTM 的细胞状态通路与四个门](/assets/rnn-lstm/svg/lstm.1.svg)

## 两种状态：cell 负责携带，hidden 负责暴露

给定输入 $x_t\in\mathbb R^{d_x}$、隐藏状态 $h_{t-1}\in\mathbb R^{d_h}$ 和细胞状态 $c_{t-1}\in\mathbb R^{d_h}$，标准 LSTM 的四组门值为

$$
\begin{aligned}
i_t&=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i),\\
f_t&=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_f),\\
o_t&=\sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o),\\
\widetilde c_t&=\tanh(W_{xc}x_t+W_{hc}h_{t-1}+b_c).
\end{aligned}
$$

其中 $i_t$ 是输入门，$f_t$ 是遗忘门，$o_t$ 是输出门，$\widetilde c_t$ 是候选内容。状态更新为

$$
\begin{aligned}
c_t&=f_t\odot c_{t-1}+i_t\odot\widetilde c_t,\\
h_t&=o_t\odot\tanh(c_t).
\end{aligned}
$$

所有门都是逐坐标向量，$\odot$ 表示 Hadamard 乘积。输入门不是「把输入原样写进去」，而是控制候选内容的写入比例；输出门也不是「决定下一个状态」，而是控制当前细胞状态有多少暴露给 $h_t$。

| 变量 | 形状 | 数值范围 | 作用 |
| --- | --- | --- | --- |
| $i_t$ | $d_h$ | $(0,1)$ | 允许候选内容写入 cell |
| $f_t$ | $d_h$ | $(0,1)$ | 保留上一时刻 cell 的比例 |
| $o_t$ | $d_h$ | $(0,1)$ | 暴露 cell 到 hidden 的比例 |
| $\widetilde c_t$ | $d_h$ | $(-1,1)$ | 当前输入和旧 hidden 产生的候选内容 |
| $c_t$ | $d_h$ | 无硬边界 | 长期状态，数值上可以累积 |
| $h_t$ | $d_h$ | $(-1,1)$ | 传给下一步和输出头的隐藏状态 |

这里的「长期」不是说 $c_t$ 永远不变，而是说它有一条不必每次都穿过 tanh 的加法路径。遗忘门和输入门仍然可以有意地改变它。

## 参数账：四个门不是四个独立网络

若把四个门分别实现，权重形状为

$$
\begin{aligned}
W_{x\bullet}&\in\mathbb R^{d_h\times d_x},\\
W_{h\bullet}&\in\mathbb R^{d_h\times d_h},\\
b_\bullet&\in\mathbb R^{d_h},
\qquad
\bullet\in\{i,f,o,c\}.
\end{aligned}
$$

四门循环层的参数量为

$$
4(d_hd_x+d_h^2+d_h)
=4d_h(d_x+d_h+1).
$$

例如 $d_x=3,d_h=4$ 时，

$$
4(4\times3+4\times4+4)=128.
$$

若再接一个 $d_y=2$ 的线性输出头，增加 $2\times4+2=10$ 个参数，总数为 138。这个数字比同宽 vanilla RNN 的

$$
d_hd_x+d_h^2+d_h=32
$$

多四倍左右，是 LSTM 用额外门控换取路径控制的直接代价。

| 配置 | 循环层参数量 | 输出头参数量 | 总参数量 |
| --- | ---: | ---: | ---: |
| vanilla RNN，$d_x=3,d_h=4,d_y=2$ | 32 | 10 | 42 |
| LSTM，四门，$d_x=3,d_h=4,d_y=2$ | 128 | 10 | 138 |
| LSTM + projection，投影维度 $d_p$ | $4d_h(d_x+d_h+1)$ | 视输出头而定 | 还需计入投影矩阵 |

工程实现常把四个仿射变换拼成一次大矩阵乘法：

$$
\begin{bmatrix}
a^i_t\\
a^f_t\\
a^o_t\\
a^c_t
\end{bmatrix}
=
W_xx_t+W_hh_{t-1}+b,
$$

其中 $W_x\in\mathbb R^{4d_h\times d_x}$、$W_h\in\mathbb R^{4d_h\times d_h}$。这只是计算布局变化，不改变四个门的数学语义。审计时应确认拼接顺序，例如是 $(i,f,o,c)$ 还是框架约定的其他排列。

## Cell highway：直接 carry 路径为何更容易保留

只沿显式的上一 cell 到当前 cell 边，暂时把门值视作已算好的常数，有

$$
\left.\frac{\partial c_t}{\partial c_{t-1}}\right\rvert_{f_t,i_t,\widetilde c_t}
=\operatorname{diag}(f_t).
$$

连续经过 $s+1$ 到 $T$ 时刻的直接 carry 增益为

$$
\left.\frac{\partial c_T}{\partial c_s}\right\rvert_{\mathrm{carry}}
=
\operatorname{diag}(f_T)
\operatorname{diag}(f_{T-1})
\cdots
\operatorname{diag}(f_{s+1}).
$$

因为这些矩阵都是对角的，逐坐标看就是

$$
\frac{\partial c_T^{(j)}}{\partial c_s^{(j)}}\bigg\rvert_{\mathrm{carry}}
=
\prod_{t=s+1}^{T}f_t^{(j)}.
$$

当某个坐标的 $f_t^{(j)}$ 接近 1 时，它可以在很多步中接近恒等传递；当 $f_t^{(j)}$ 接近 0 时，它主动删除该坐标的历史。这是一个可学习的保留/擦除开关，而不是固定的「永远记住」开关。

### 直接路径不等于完整 Jacobian

门值本身依赖 $h_{t-1}$，而 $h_{t-1}$ 又依赖 $c_{t-1}$。因此完整的状态转移 Jacobian 还包含

$$
\frac{\partial c_t}{\partial c_{t-1}}
=
\operatorname{diag}(f_t)
+
\operatorname{diag}(c_{t-1})\frac{\partial f_t}{\partial c_{t-1}}
+
\operatorname{diag}(\widetilde c_t)\frac{\partial i_t}{\partial c_{t-1}}
+
\operatorname{diag}(i_t)\frac{\partial\widetilde c_t}{\partial c_{t-1}}.
$$

上式中的后三项经由 $h_{t-1}$ 间接产生。为了说明 LSTM 为什么改善长程梯度，通常先强调第一项的 cell highway；为了证明一个具体实现的梯度正确，则必须把门的依赖也纳入完整 BPTT。

这一区分很重要：

| 说法 | 可以推出什么 | 不能推出什么 |
| --- | --- | --- |
| $f_t$ 接近 1 | 直接 carry 路径的局部增益接近 1 | 完整 Jacobian 所有方向都接近 1 |
| $c_t$ 数值稳定 | cell 没有立即溢出 | 梯度没有消失或门没有饱和 |
| 输出门很小 | 当前 hidden 暴露较少 | cell 内没有信息或未来不能再读取 |
| LSTM 能学习长依赖 | 图中存在可训练的保留路径 | 任意初始化和任意序列都能学好 |

## 标量记忆账：遗忘门决定有效视野

为了只观察 carry，取一维 cell、固定输入写入为 0：

$$
c_t=f_tc_{t-1}.
$$

则

$$
\frac{\partial c_T}{\partial c_0}
=\prod_{t=1}^{T}f_t.
$$

如果每一步遗忘门相同，便退化为 $f^T$：

| 每步遗忘门 $f$ | 步数 $T$ | 直接 carry 增益 $f^T$ | 含义 |
| --- | ---: | ---: | --- |
| 0.5 | 5 | 0.03125 | 历史快速被擦除 |
| 0.8 | 20 | 0.011529215046 | 仍然是 vanilla RNN 式的明显收缩 |
| 0.95 | 20 | 0.358485922409 | 20 步后仍保留约 36% |
| 0.99 | 20 | 0.817906937597 | 短中程依赖较容易保留 |
| 0.99 | 100 | 0.366032341273 | 接近 1 仍会在很长距离上衰减 |
| 1.0 | 100 | 1 | 理想恒等路径，但不是 sigmoid 的普通输出 |

因此「遗忘门接近 1」是相对于要保留的时间尺度而言的。$f=0.99$ 对 20 步很宽松，对 1000 步仍然会衰减。门控结构提高了可学习性，却没有废除乘法。

### 遗忘门偏置

因为 $f_t=\sigma(a^f_t)$，把遗忘门 bias 初始化为正数，可以让初始门值偏向保留：

$$
\sigma(1)=0.731058578630,
\qquad
\sigma(2)=0.880797077978.
$$

这只是初始化时的偏好。输入、hidden 权重和后续更新仍会改变 $a^f_t$。如果序列语义要求快速重置，过大的 forget bias 反而可能让无关历史拖得太久。

## 三步前向：把门值、cell 和 hidden 分开记

下面固定一维门值，令 $c_0=1$。每一行都使用

$$
c_t=f_tc_{t-1}+i_t\widetilde c_t,
\qquad
h_t=o_t\tanh(c_t).
$$

| 时间步 | $f_t$ | $i_t$ | $\widetilde c_t$ | $o_t$ | $c_t$ | $h_t$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.90 | 0.20 | 0.50 | 0.80 | 1.000000000000 | 0.609275324765 |
| 2 | 0.80 | 0.30 | −0.40 | 0.70 | 0.680000000000 | 0.414063576802 |
| 3 | 0.95 | 0.10 | 0.30 | 0.60 | 0.676000000000 | 0.353347693657 |

第一步中

$$
c_1=0.9\times1+0.2\times0.5=1.
$$

第二步中，旧内容被保留 0.8，新候选内容为负并写入 0.3：

$$
c_2=0.8\times1+0.3\times(-0.4)=0.68.
$$

第三步中，cell 只从 0.68 变为 0.676，hidden 却还要经过输出门和 tanh。因此不能从 $h_t$ 的变化幅度直接推断 cell 中的信息是否被保留。

沿直接 carry 路径从 $c_0$ 到 $c_3$ 的增益是

$$
0.90\times0.80\times0.95=0.684.
$$

这 0.684 只描述旧 cell 通过遗忘门的路径，不包括每一步门值对 hidden 的间接依赖，也不包括输出头的梯度。

## 通过门的 BPTT：cell 梯度如何分流

用上划线表示损失对变量的梯度，例如

$$
\bar h_t=\frac{\partial\mathcal L}{\partial h_t},
\qquad
\bar c_t=\frac{\partial\mathcal L}{\partial c_t}.
$$

先从 $h_t=o_t\odot\tanh(c_t)$ 回传：

$$
\begin{aligned}
\bar o_t&=\bar h_t\odot\tanh(c_t),\\
\bar c_t&\mathrel{+}=\bar h_t\odot o_t\odot\bigl(1-\tanh^2(c_t)\bigr).
\end{aligned}
$$

再沿 cell 更新式把同一个 $\bar c_t$ 分给三条输入：

$$
\begin{aligned}
\bar f_t&=\bar c_t\odot c_{t-1},\\
\bar i_t&=\bar c_t\odot\widetilde c_t,\\
\bar{\widetilde c}_t&=\bar c_t\odot i_t,\\
\bar c_{t-1}&=\bar c_t\odot f_t.
\end{aligned}
$$

最后通过各门的激活函数：

$$
\begin{aligned}
\delta^i_t&=\bar i_t\odot i_t\odot(1-i_t),\\
\delta^f_t&=\bar f_t\odot f_t\odot(1-f_t),\\
\delta^o_t&=\bar o_t\odot o_t\odot(1-o_t),\\
\delta^c_t&=\bar{\widetilde c}_t\odot(1-\widetilde c_t^2).
\end{aligned}
$$

四个 gate preactivation 的梯度再通过各自的仿射层回到 $x_t$ 和 $h_{t-1}$：

$$
\bar h_{t-1}\mathrel{+}=
W_{hi}^{\mathsf T}\delta^i_t
+W_{hf}^{\mathsf T}\delta^f_t
+W_{ho}^{\mathsf T}\delta^o_t
+W_{hc}^{\mathsf T}\delta^c_t.
$$

参数梯度仍按时间累加。例如

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial W_{xi}}
&\mathrel{+}=\delta^i_tx_t^{\mathsf T},\\
\frac{\partial\mathcal L}{\partial W_{hi}}
&\mathrel{+}=\delta^i_th_{t-1}^{\mathsf T},\\
\frac{\partial\mathcal L}{\partial b_i}
&\mathrel{+}=\delta^i_t,
\end{aligned}
$$

遗忘、输出和候选内容三组参数同理。LSTM 改变的是状态路径，不改变共享参数必须把每个时间步贡献相加这一事实。

### 第二步的局部数值

取上面第二步的门值，并假设该步只收到 $\bar h_2=1$、没有来自未来的额外 $\bar c_2$。因为 $c_2=0.68,o_2=0.7$，

$$
\bar c_2
=0.7\bigl(1-\tanh^2(0.68)\bigr)
=0.455073363380.
$$

于是

$$
\begin{aligned}
\bar c_1&=0.455073363380\times0.8=0.364058690704,\\
\bar f_2&=0.455073363380\times1=0.455073363380,\\
\bar i_2&=0.455073363380\times(-0.4)=-0.182029345352,\\
\bar{\widetilde c}_2&=0.455073363380\times0.3=0.136522009014.
\end{aligned}
$$

再乘门的局部导数：

$$
\begin{aligned}
\delta^f_2&=0.072811738141,\\
\delta^i_2&=-0.038226162524,\\
\delta^c_2&=0.114678487572.
\end{aligned}
$$

这个小账本同时检查了三件事：输出门先影响 cell 梯度，遗忘门把 cell 梯度传给前一步，门的 sigmoid/tanh 再改变回到参数和 hidden 的梯度。若只检查 $f_2\bar c_2$，会漏掉后两层门控路径。

## 门的语义：不是四个独立的开关故事

### 输入门和候选内容必须成对看

$i_t$ 小不代表当前输入没有信息，也可能是候选内容 $\widetilde c_t$ 很大但被拒绝写入；$i_t$ 大也不代表写入内容重要，因为候选值可能接近 0。审计时应同时记录

$$
\lVert i_t\rVert,\qquad
\lVert\widetilde c_t\rVert,\qquad
\lVert i_t\odot\widetilde c_t\rVert.
$$

### 遗忘门是擦除比例，不是记忆评分

$f_t$ 只表示上一 cell 在当前一步被保留多少。某个信息已经不再需要时，$f_t$ 接近 0 是正确行为；因此不能把高 forget gate 平均值当成越大越好的模型质量指标。

### 输出门控制可见性

$o_t$ 很小会让 $h_t$ 很小，但 cell 仍可能保留有效状态。若任务在之后的时间步需要读取它，后续输出门可以再次打开。反过来，当前输出很小也可能只是输出门关闭，不等于模型没有更新 cell。

### 候选内容仍有 tanh

LSTM 的 cell highway 绕开的是每一步对旧 cell 的直接非线性变换；新写入的候选内容仍经过 tanh。若候选预激活长期饱和，模型会难以写入精细的新信息，即使旧信息的 carry 路径健康。

## 变体和实现边界

| 变体 | 改变什么 | 代价或风险 |
| --- | --- | --- |
| 标准 LSTM | 四个门分别由 $x_t,h_{t-1}$ 计算 | 参数多、门顺序实现容易错 |
| coupled input-forget gate | 令 $i_t=1-f_t$，减少一个独立门 | 不能独立控制保留和写入 |
| peephole LSTM | 让门看到 $c_{t-1}$ 或 $c_t$ | 增加逐坐标参数与依赖边 |
| projection LSTM | 用较大 cell、较小 projection 输出 hidden | 需分清 cell width 与 recurrent width |
| bidirectional LSTM | 正向和反向各维护一套状态 | 不能直接用于严格在线因果预测 |
| stacked LSTM | 多层之间传递 hidden 序列 | 同时增加深度方向和时间方向的梯度路径 |

### Projection 不是把 cell 截断

projection LSTM 中，内部 cell 维度可以是 $d_c$，对外 hidden 或 recurrent projection 维度是 $d_p$。参数账要明确：

$$
c_t\in\mathbb R^{d_c},
\qquad
h_t\in\mathbb R^{d_p},
\qquad
d_p\ne d_c
$$

具体门的输入权重、hidden 权重和投影矩阵形状随实现而变。只看到一个名为 hidden_size 的参数，不能推断 cell path 的实际宽度。

### Packed sequence 和 padding mask

变长 batch 中，padding 时间步不应产生有效的门更新或 loss。若框架用 packed sequence 跳过 padding，状态的时间索引不是简单的矩形网格；若用 mask，则要分别决定：

1. padding 位置是否更新 $c_t,h_t$；
2. padding 位置是否贡献输出 loss；
3. 一个样本结束后，状态是否 reset；
4. batch 中下一条样本是否错误继承了上一条状态。

一个有效位置 mask 只能解决损失归约的一部分问题，不能自动修正跨样本 state carry。

## 训练策略：针对路径，而不是只调学习率

### Forget bias 与初始时间尺度

正 forget bias 可以让初始 $f_t$ 偏大，减少刚开始训练时的快速擦除。应把它和目标序列的时间尺度一起选择：需要快速切换的任务不应盲目把所有遗忘门推向 1。

### 梯度裁剪

LSTM 仍可能在门的间接路径或 stacked 深度中发生梯度爆炸。全局范数裁剪可以限制异常更新：

$$
g\leftarrow
g\min\left(1,\frac{\tau}{\lVert g\rVert_2+\varepsilon}\right).
$$

但它不会把已经很小的 $\bar c_t$ 放大，也不会修复门饱和。应同时记录裁剪前范数和 cell carry 梯度。

### Dropout 的位置

对输入到门的连接施加 dropout，和对 recurrent state 每个时间步重新采样 dropout mask，不是同一个语义。后者会把额外随机扰动注入 cell highway，可能破坏长期路径。若使用 locked/variational dropout，应写明 mask 是否在一个序列内复用。

### State reset、carry 与 detach

跨 chunk carry 可以让下一个窗口看到前一个窗口的末状态，但在边界 detach 会切断梯度。三种行为分别是：

| 操作 | 前向状态 | 反向梯度 | 适用语义 |
| --- | --- | --- | --- |
| carry 不 detach | 跨窗口传递 | 跨窗口回传 | 真正需要连续长序列且资源允许 |
| carry + detach | 跨窗口传递 | 在边界截断 | TBPTT，限制显存和反向深度 |
| reset | 重新初始化 | 不跨边界 | 样本或 episode 明确独立 |

LSTM 的 cell highway 不会替你决定这些边界。一个很长的 $c_t$ 数值链，配上频繁 detach，仍然没有长程参数信用。

## 失效模式：门控存在不代表实现正确

### 四门切片顺序错误

拼接仿射输出时，如果实现按 $(i,f,o,c)$ 产生，而解码按 $(i,o,f,c)$ 读取，代码仍能运行，loss 也可能下降，但每个门的语义已错位。用固定输入和固定权重打印四门值，先核对切片顺序。

### 把 cell 和 hidden 混为一谈

$c_t$ 可以大于 1，$h_t=o_t\odot\tanh(c_t)$ 被限制在 $(-1,1)$。如果把 hidden 范围当作 cell 范围，或者用 hidden 的饱和推断 cell 已丢失，会得到错误诊断。

### Forget gate 全关或全开

$f_t$ 长期接近 0 会擦除所有历史；长期接近 1 又可能让无关历史和旧偏差拖延。两者都要结合任务标签对齐、输入变化和 cell 轨迹判断，不能只追求平均门值。

### 只检查 carry 项

直接 carry 项看起来稳定，不代表门的间接 Jacobian、输出头或 stacked layer 没有爆炸。完整 gradient check 应至少覆盖 $W_{xi},W_{xf},W_{xo},W_{xc}$ 和对应 hidden 权重。

### 输出门关闭造成假阴性

某个时间点 $h_t$ 很小可能是 $o_t$ 关闭，但 $c_t$ 仍携带信息。调试时要同时画 $c_t$、$\tanh(c_t)$、$o_t$ 和 $h_t$。

### Padding 污染 cell

若 padding 输入仍推进四个门，cell 可能在有效序列结束后继续漂移；若 padding 输出仍进入 mean loss，梯度分母也会被改变。状态 mask 和 loss mask 要分别验证。

### 状态跨样本泄漏

在 batch 循环中复用一个 hidden/cell 对象，可能让上一条样本的状态进入下一条样本。最小测试应交换 batch 顺序、显式 reset，并核对同一条样本的输出是否保持不变。

### Projection 的维度误读

把 projection width 当成 cell width 会导致门矩阵、输出头和参数量的审计全部错位。打印每个 gate 的输入输出 shape，不能只看模块构造函数的一个总宽度参数。

## LSTM 审计协议

1. **固定状态定义。** 明确 $c_t$、$h_t$、初始状态、列向量/row-batch 和输出头。
2. **固定门顺序。** 记录拼接矩阵的切片顺序，并用一个标量样例打印 $i,f,o,\widetilde c$。
3. **核对形状与参数量。** 分别计算四门的 input-to-hidden、hidden-to-hidden、bias 和 projection 参数。
4. **先做手工前向。** 固定门值，逐步复算 $c_t=f_tc_{t-1}+i_t\widetilde c_t$ 与 $h_t=o_t\tanh(c_t)$。
5. **分离 carry 梯度。** 记录 $\prod_t f_t$，同时说明它不是完整 Jacobian。
6. **核对 gate BPTT。** 分别检查输出门、cell、遗忘门、输入门、候选内容的局部梯度。
7. **按参数组做中心差分。** 至少覆盖四组 input 权重、四组 recurrent 权重与 bias。
8. **检查序列边界。** 写出 carry/reset/detach、padding/packed sequence 和 loss mask 的规则。
9. **记录运行时统计。** 包括 gate 分位数、cell/hidden 范数、$\bar c_t$、裁剪系数、非有限值首次出现位置。
10. **验证反事实。** 改变 forget bias、关闭输出门、交换 batch 顺序或缩短窗口，确认观察到的变化符合预期路径。

LSTM 的核心不是「有四个门」这句结构描述，而是 cell state 为时间图增加了一条可学习的加法通路。遗忘门决定旧信息留下多少，输入门和候选内容决定写入什么，输出门决定当前暴露什么；BPTT 则沿这些分支把梯度重新汇合。下一篇 GRU 会删掉显式 cell state，把保留和更新压缩到更少的门中，代价是路径语义也随之改变。

## 相关词条

[RNN 中的梯度消失](../rnn-lstm/rnn-vanishing-gradient/)

[时间反向传播](../rnn-lstm/bptt/)

[循环神经网络](../rnn-lstm/rnn/)

[RNN 时间展开](../rnn-lstm/rnn-unrolling/)

[门控循环单元](../rnn-lstm/gru/)

[序列建模](../rnn-lstm/sequence-modeling/)

[梯度裁剪](../training-nn/gradient-clipping/)

[梯度消失与爆炸](../backpropagation/vanishing-and-exploding/)

[双向循环网络](../rnn-lstm/bidirectional-rnn/)
