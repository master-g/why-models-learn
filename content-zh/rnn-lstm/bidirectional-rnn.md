---
title: "双向循环网络：让每个位置同时看到过去与未来"
tags: ["why-models-learn"]
---

双向循环网络（bidirectional recurrent neural network，BiRNN）在同一条输入序列上运行两条彼此独立的循环路径：一条从左到右累积过去，一条从右到左累积未来。第 $t$ 个位置的表示由两条路径合并而成，因此可以同时利用 $x_{1:t}$ 和 $x_{t:T}$。

它的核心不是「把一个 RNN 的 hidden size 乘二」，而是改变了信息可见性合同：

1. 正向状态只能使用当前位置及其左侧输入；
2. 反向状态从序列末端往回读，因此在位置 $t$ 可以使用右侧输入；
3. 两个方向通常使用不同参数，最后再按约定拼接、相加或投影；
4. 完整序列已知时，双向表示适合离线标注；需要立即输出时，反向路径会造成未来信息泄漏；
5. padding、反向索引、bridge 和输出宽度必须分别核对，不能只看最终 tensor shape。

![BiRNN 在同一条序列上同时沿正向与反向读取，再在每个位置合并两个状态](/assets/rnn-lstm/svg/bidirectional-rnn.1.svg)

## 两条路径：同一输入，不同时间方向

设输入序列为

$$
x_{1:T}=(x_1,x_2,\ldots,x_T),
$$

正向 RNN 从左到右运行：

$$
\overrightarrow h_t
=f_{\rightarrow}\bigl(\overrightarrow h_{t-1},x_t;\theta_{\rightarrow}\bigr),
\qquad
t=1,\ldots,T.
$$

反向 RNN 把序列从右到左读取：

$$
\overleftarrow h_t
=f_{\leftarrow}\bigl(\overleftarrow h_{t+1},x_t;\theta_{\leftarrow}\bigr),
\qquad
t=T,\ldots,1.
$$

用零向量表示两个扫描的边界时，初始条件是

$$
\overrightarrow h_0=\boldsymbol 0,
\qquad
\overleftarrow h_{T+1}=\boldsymbol 0.
$$

这里的「初始」依赖扫描方向：正向从 $t=1$ 开始，反向从 $t=T$ 开始。$\theta_{\rightarrow}$ 和 $\theta_{\leftarrow}$ 通常是两套参数；除非明确选择权重绑定，否则不能把它们当作同一个矩阵。

如果把状态转移展开，正向状态满足

$$
\overrightarrow h_t
=F_t(x_1,\ldots,x_t),
$$

而反向状态满足

$$
\overleftarrow h_t
=G_t(x_t,\ldots,x_T).
$$

因此双向层在位置 $t$ 的信息依赖是

$$
h_t
=\operatorname{merge}\bigl(\overrightarrow h_t,\overleftarrow h_t\bigr),
$$

它可以依赖整条已提供的序列；这正是它与 causal RNN 的根本区别。

| 状态 | 扫描方向 | 在位置 $t$ 可见的输入 | 边界状态 |
| --- | --- | --- | --- |
| $\overrightarrow h_t$ | $1\to T$ | $x_1,\ldots,x_t$ | $\overrightarrow h_0$ |
| $\overleftarrow h_t$ | $T\to1$ | $x_t,\ldots,x_T$ | $\overleftarrow h_{T+1}$ |
| $h_t$ | 合并 | 取决于两个方向 | 由 merge 决定 |

“反向”不是把输出序列倒过来就结束了。实现需要在反向执行后把每个状态重新对齐到原始位置，否则 $x_t$ 的正向状态会和另一个位置的反向状态拼接。

## Merge：两个 hidden 如何成为一个表示

最常见的是拼接：

$$
h_t
=
\begin{bmatrix}
\overrightarrow h_t\\
\overleftarrow h_t
\end{bmatrix}
=
[\overrightarrow h_t;\overleftarrow h_t]
\in\mathbb R^{2d_h}.
$$

如果两个方向的 hidden width 都是 $d_h$，拼接后的每个位置有 $2d_h$ 个特征。拼接保留了「过去通道」和「未来通道」的边界，下一层可以学习如何分别使用它们。

也可以相加：

$$
h_t
=\overrightarrow h_t+\overleftarrow h_t
\in\mathbb R^{d_h}.
$$

相加要求两个方向的宽度相同，并把方向信息混进同一个坐标。它节省输出宽度，却不能让后续层直接区分某个数值来自过去还是未来。

若下游希望使用第三个宽度 $d_m$，可以在拼接后投影：

$$
z_t
=W_m[\overrightarrow h_t;\overleftarrow h_t]+b_m,
\qquad
W_m\in\mathbb R^{d_m\times 2d_h}.
$$

这时需要把原始双向状态 $h_t$ 与投影后的 $z_t$ 分开命名。否则只记录一个 “hidden size” 很容易把循环宽度、合并宽度和下游输入宽度混为一谈。

| merge 方式 | 输入宽度 | 输出宽度 | 保留的方向边界 | 主要代价 |
| --- | ---: | ---: | --- | --- |
| concat | $d_h+d_h$ | $2d_h$ | 保留，顺序由实现定义 | 下一层输入变宽 |
| add | $d_h+d_h$ | $d_h$ | 不显式保留 | 两方向必须同宽 |
| concat + projection | $2d_h$ | $d_m$ | 投影前保留 | 增加 $W_m$ 与一次矩阵乘 |

批量输入的形状账本可以写成

$$
X\in\mathbb R^{B\times T\times d_x},
\qquad
\overrightarrow H,\overleftarrow H\in\mathbb R^{B\times T\times d_h}.
$$

采用拼接时

$$
H\in\mathbb R^{B\times T\times 2d_h},
$$

采用相加时

$$
H\in\mathbb R^{B\times T\times d_h}.
$$

若实现使用 time-first 或把特征轴放在前面，轴顺序会改变，但「每个原始时间位置有两个对齐方向状态」这个不变量不能改变。

## 一个线性例子：正向看左边，反向看右边

为了把方向差异从非线性中分离出来，取标量线性循环：

$$
\overrightarrow h_t
=\frac12\overrightarrow h_{t-1}+x_t,
\qquad
\overleftarrow h_t
=\frac12\overleftarrow h_{t+1}+x_t.
$$

令

$$
x_{1:4}=(1,2,0,3),
\qquad
\overrightarrow h_0=0,
\qquad
\overleftarrow h_5=0.
$$

正向扫描的计算是

$$
\begin{aligned}
\overrightarrow h_1&=1,\\
\overrightarrow h_2&=\frac12\cdot1+2=2.5,\\
\overrightarrow h_3&=\frac12\cdot2.5+0=1.25,\\
\overrightarrow h_4&=\frac12\cdot1.25+3=3.625.
\end{aligned}
$$

反向扫描从 $x_4$ 开始：

$$
\begin{aligned}
\overleftarrow h_4&=3,\\
\overleftarrow h_3&=\frac12\cdot3+0=1.5,\\
\overleftarrow h_2&=\frac12\cdot1.5+2=2.75,\\
\overleftarrow h_1&=\frac12\cdot2.75+1=2.375.
\end{aligned}
$$

对齐后每个位置的拼接状态为

| 位置 $t$ | $x_t$ | $\overrightarrow h_t$ | $\overleftarrow h_t$ | $[\overrightarrow h_t;\overleftarrow h_t]$ | 若相加 |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 1 | 1 | 2.375 | $(1,2.375)$ | 3.375 |
| 2 | 2 | 2.5 | 2.75 | $(2.5,2.75)$ | 5.25 |
| 3 | 0 | 1.25 | 1.5 | $(1.25,1.5)$ | 2.75 |
| 4 | 3 | 3.625 | 3 | $(3.625,3)$ | 6.625 |

位置 $t=2$ 的正向值只使用 $x_1,x_2$：

$$
\overrightarrow h_2=\frac12x_1+x_2=2.5.
$$

同一位置的反向值使用 $x_2,x_3,x_4$：

$$
\overleftarrow h_2=x_2+\frac12x_3+\frac14x_4=2.75.
$$

这不是「两个模型碰巧给了两个数字」，而是两条不同的信息路径。一般地，对衰减系数 $\alpha,\beta$，线性零边界循环可以展开为

$$
\overrightarrow h_t
=\sum_{i=1}^{t}\alpha^{t-i}x_i,
\qquad
\overleftarrow h_t
=\sum_{i=t}^{T}\beta^{i-t}x_i.
$$

因此 $\alpha$ 控制过去信息随距离的衰减，$\beta$ 控制未来信息随距离的衰减。真实 RNN 还会经过激活函数和矩阵变换，但两个方向对上下文的支持范围仍然遵守这张依赖图。

## 反向索引：执行顺序与原始位置必须分开

以零基索引描述批量输入：

$$
X\in\mathbb R^{B\times T\times d_x},
\qquad
\widetilde X_{b,k,:}=X_{b,T-1-k,:}.
$$

把 $\widetilde X$ 喂给一个普通的正向 RNN，得到的是反向执行顺序的状态 $\widetilde H$。对齐回原始时间轴时应使用

$$
\overleftarrow H_{b,t,:}
=\widetilde H_{b,T-1-t,:}.
$$

如果没有这次 flip，$\widetilde H_{b,0,:}$ 会被错误地放在原始位置 $0$；它其实对应原始序列的最后一个位置。对长度 $T=4$ 的序列，执行顺序和对齐关系是

| 反向执行步 | 读入的原始输入 | 产生的执行状态 | 应写回的原始位置 |
| ---: | --- | --- | ---: |
| 0 | $x_4$ | $\widetilde h_0$ | 4 |
| 1 | $x_3$ | $\widetilde h_1$ | 3 |
| 2 | $x_2$ | $\widetilde h_2$ | 2 |
| 3 | $x_1$ | $\widetilde h_3$ | 1 |

time-first、batch-first 和 packed sequence 只会改变这张表的索引写法，不会改变最后一列的语义。调试时同时打印原始时间下标、反向执行下标和 merge 前两条状态，往往比只检查输出 shape 更快发现问题。

## 因果性：双向层什么时候合法

正向状态对未来输入的偏导在因果 RNN 中为零：

$$
\frac{\partial\overrightarrow h_t}{\partial x_j}=0
\qquad
(j>t).
$$

反向状态则可能依赖任意未来位置：

$$
\frac{\partial\overleftarrow h_t}{\partial x_j}\ne0
\qquad
(j>t).
$$

若 merge 是拼接并且两条路径都真正使用输入，$h_t$ 通常依赖整条 $x_{1:T}$。所以双向层适合“等整段输入到齐后再输出”的任务，例如离线序列标注、整段文本编码或允许看完整窗口的识别；它不自动适合“输入到达一个 token 就必须立刻输出”的任务。

| 部署合同 | 位置 $t$ 能看到的输入 | 双向层是否直接满足 | 典型处理 |
| --- | --- | --- | --- |
| 离线整段标注 | $x_{1:T}$ | 是 | 直接使用两个方向 |
| 单向在线预测 | $x_{1:t}$ | 否 | 使用单向 RNN 或因果模型 |
| 固定 lookahead $K$ | $x_{1:\min(t+K,T)}$ | 不是完整双向 | 明确窗口边界和延迟 |
| 分块离线 | 当前 chunk 与允许的上下文 | 取决于 chunk 合同 | 每块重置或携带状态并记录 |

固定 lookahead 不是把完整 BiRNN 偷偷放进在线系统。它是另一种模型合同：反向路径最多读取 $K$ 个未来位置，输出需要等待相应延迟，chunk 边界还要说明是否允许跨块读取。

这里的未来泄漏是信息可见性问题，不是模型“太强”的抽象描述。若验证时把完整序列提供给双向层，而线上只能得到前缀，验证指标和部署指标就不在同一分布上。

## 参数、计算和内存账本

对一个 vanilla RNN 方向，设输入宽度为 $d_x$、hidden width 为 $d_h$。只计算循环单元参数：

$$
P_{\mathrm{dir}}
=d_hd_x+d_h^2+d_h
=(d_x+d_h+1)d_h.
$$

双向层有两套循环参数，因此

$$
P_{\mathrm{recurrent}}
=2(d_x+d_h+1)d_h.
$$

若每个位置的拼接状态接一个 $d_y$ 维输出头，则

$$
P_{\mathrm{concat}}
=2(d_x+d_h+1)d_h+d_y(2d_h+1).
$$

若使用相加 merge，输出头改为

$$
P_{\mathrm{add}}
=2(d_x+d_h+1)d_h+d_y(d_h+1).
$$

取 $d_x=3,d_h=4,d_y=5$：

| 配置 | 每方向循环参数 | 双向循环参数 | 输出头参数 | 总参数 |
| --- | ---: | ---: | ---: | ---: |
| 单向 RNN + $d_y$ 头 | 32 | 32 | 25 | 57 |
| 双向 + concat | 32 | 64 | 45 | 109 |
| 双向 + add | 32 | 64 | 25 | 89 |

这里的总参数没有计入额外 projection、LayerNorm、dropout 规则或多层结构。双向通常使循环部分参数约翻倍；concat 还会把下一层看到的输入宽度翻倍，从而增加下一层的矩阵乘成本。

对长度 $T$ 的序列，单个方向的循环计算量可以粗略写成

$$
O\bigl(T(d_xd_h+d_h^2)\bigr).
$$

双向有两条这样的依赖链：

$$
O\bigl(2T(d_xd_h+d_h^2)\bigr).
$$

两条方向可以在硬件上并行启动，但每一条内部仍然沿时间串行；“方向翻倍”不能被解释为“完全没有时间依赖”。训练时还要保存两条状态序列，concat 的状态激活大致是 $BT(2d_h)$，而 add 的输出只有 $BTd_h$，但两方向的中间激活仍需为反向传播保留。

## 梯度：merge 分开，输入再汇合

设每个位置的 loss 为 $\ell_t$，总损失为

$$
\mathcal L=\sum_{t=1}^{T}\ell_t(h_t,y_t).
$$

对 concat merge，写上游梯度为

$$
g_t
=\frac{\partial\mathcal L}{\partial h_t}
=
\begin{bmatrix}
g_t^{\rightarrow}\\
g_t^{\leftarrow}
\end{bmatrix}.
$$

反向传播在 merge 处把梯度拆回两个方向：

$$
\frac{\partial\mathcal L}{\partial\overrightarrow h_t}=g_t^{\rightarrow},
\qquad
\frac{\partial\mathcal L}{\partial\overleftarrow h_t}=g_t^{\leftarrow}.
$$

之后每条方向分别沿自己的时间顺序执行 BPTT。两套参数的梯度是各自时间步的累加：

$$
\frac{\partial\mathcal L}{\partial\theta_{\rightarrow}}
=
\sum_{t=1}^{T}
\frac{\partial\mathcal L}{\partial\overrightarrow h_t}
\frac{\partial\overrightarrow h_t}{\partial\theta_{\rightarrow}},
$$

$$
\frac{\partial\mathcal L}{\partial\theta_{\leftarrow}}
=
\sum_{t=1}^{T}
\frac{\partial\mathcal L}{\partial\overleftarrow h_t}
\frac{\partial\overleftarrow h_t}{\partial\theta_{\leftarrow}}.
$$

如果两个方向读取同一个输入，输入梯度在输入处再次相加：

$$
\frac{\partial\mathcal L}{\partial x_t}
=
\left.\frac{\partial\mathcal L}{\partial x_t}\right\rvert_{\rightarrow}
+
\left.\frac{\partial\mathcal L}{\partial x_t}\right\rvert_{\leftarrow}.
$$

这不表示两个方向共享 hidden 状态。它只表示同一个 $x_t$ 同时是两条计算图的父节点。若显式绑定 $\theta_{\rightarrow}=\theta_{\leftarrow}$，才需要把两条方向对同一参数的梯度合并；绑定参数是额外设计，不是 BiRNN 的默认含义。

相加 merge 的梯度路径不同：

$$
h_t=\overrightarrow h_t+\overleftarrow h_t
\quad\Longrightarrow\quad
\frac{\partial\mathcal L}{\partial\overrightarrow h_t}
=
\frac{\partial\mathcal L}{\partial h_t},
\qquad
\frac{\partial\mathcal L}{\partial\overleftarrow h_t}
=
\frac{\partial\mathcal L}{\partial h_t}.
$$

因此相加会把同一份上游梯度送入两个方向；concat 则按坐标切分。审计反向传播时，要先确认 merge 规则，再确认每个方向的时间反转是否在梯度路径中被正确处理。

## 变长序列：反向方向的 padding 更容易出错

设第 $b$ 个样本的有效长度为 $T_b$，右侧 padding 到 $T_{\max}$。有效位置掩码为

$$
M_{b,t}=\mathbf 1[t\le T_b].
$$

正向路径可以在有效位置更新，在 padding 位置保持边界状态：

$$
\overrightarrow h_{b,t}
=
\begin{cases}
f_{\rightarrow}(\overrightarrow h_{b,t-1},x_{b,t}),&M_{b,t}=1,\\
\overrightarrow h_{b,t-1},&M_{b,t}=0.
\end{cases}
$$

反向路径需要相反的边界：

$$
\overleftarrow h_{b,t}
=
\begin{cases}
f_{\leftarrow}(\overleftarrow h_{b,t+1},x_{b,t}),&M_{b,t}=1,\\
0,&t>T_b.
\end{cases}
$$

真正使用哪一种 padding 策略取决于框架的 packed sequence 或 mask API，但不变量是：padding 不能更新有效 token 的状态，也不能进入 loss。只把 padding embedding 设成零还不够，因为 recurrent bias 和旧状态仍可能改变结果。

看一个带偏置的反向标量循环：

$$
b_t=\frac12b_{t+1}+x_t+0.4,
\qquad
b_5=0.
$$

样本的有效输入是 $x_1=1,x_2=2$，$T_{\max}=4$，右侧两个位置是 PAD，PAD 数值设为 $0$。若把 padding 当普通时间步，结果为

$$
\begin{aligned}
b_4&=0.4,\\
b_3&=\frac12\cdot0.4+0.4=0.6,\\
b_2&=\frac12\cdot0.6+2+0.4=2.7,\\
b_1&=\frac12\cdot2.7+1+0.4=2.75.
\end{aligned}
$$

如果在有效序列末端使用长度感知边界，$b_3=b_4=0$ 不参与状态更新，则

$$
b_2=2.4,
\qquad
b_1=\frac12\cdot2.4+1+0.4=2.6.
$$

两个 $b_1$ 相差 $0.15$，$b_2$ 相差 $0.3$。这就是“零 padding 不等于不存在时间步”的具体数值证据。

| 检查对象 | 右侧 padding 的风险 | 应记录的证据 |
| --- | --- | --- |
| 反向边界 | 反向扫描先读到 PAD | 每个样本的 $T_b$ 与反向起点 |
| 状态更新 | bias 或旧状态在 PAD 处继续变化 | mask 前后 $\overleftarrow h_t$ |
| 输出 loss | PAD 位置被计入分母 | 有效 token 数与 mask 后 loss |
| 对齐 | reverse 后没有写回原始位置 | 原始 $t$、执行步和状态索引 |

## 多层与 Seq2Seq bridge

双向层若作为 encoder，完整源序列处理结束后，最有信息量的两个边界状态通常是

$$
\overrightarrow h_T
\quad\text{和}\quad
\overleftarrow h_1.
$$

注意反向状态是 $\overleftarrow h_1$，不是 $\overleftarrow h_T$。后者通常只包含从序列末端开始到 $T$ 的局部信息，而 $\overleftarrow h_1$ 才已经读过整个有效源序列。

拼接后可作为 context：

$$
c=
[\overrightarrow h_T;\overleftarrow h_1]
\in\mathbb R^{2d_h}.
$$

若 decoder 的 hidden width 为 $d_d$，bridge 应明确写出投影：

$$
d_0=W_cc+b_c,
\qquad
W_c\in\mathbb R^{d_d\times 2d_h}.
$$

多层双向 encoder 还要记录层和方向的顺序。对第 $\ell$ 层可以写成

$$
c^{(\ell)}
=
[\overrightarrow h_T^{(\ell)};\overleftarrow h_1^{(\ell)}].
$$

常见的 bridge 账本如下：

| encoder 输出 | 形状 | decoder bridge | 容易错的地方 |
| --- | --- | --- | --- |
| 单向末状态 | $d_h$ | 直接 copy 或投影 | 把双向模型仍按 $d_h$ 计 |
| 双向两个边界状态 | $2d_h$ | concat 后投影 | 方向顺序、$\overleftarrow h_1$ 下标 |
| 多层双向状态 | $L\times2d_h$ | 每层分别投影或重排 | 层轴、方向轴和 decoder 层对应关系 |
| 每个位置的双向输出 | $T\times2d_h$ | attention 或逐位置 head | 把 per-position output 错当 final bridge |

双向 encoder 和 causal decoder 可以组合：源序列在 decoder 开始前已经完整到齐，encoder 使用未来源 token 不构成目标生成方向的泄漏。相反，若把双向 decoder 放进左到右生成环节，就必须说明目标的未来 token 从哪里来；通常它不再是标准的 autoregressive decoding。

## BiLSTM、BiGRU 与其他变体

双向性是“沿两个方向各运行一个 cell”的外层结构，不等于 vanilla RNN 的专属功能。把每个方向的 cell 换成 LSTM 或 GRU，得到 BiLSTM 或 BiGRU：

$$
\overrightarrow h_t
=\operatorname{LSTM}_{\rightarrow}(\overrightarrow h_{t-1},x_t),
\qquad
\overleftarrow h_t
=\operatorname{LSTM}_{\leftarrow}(\overleftarrow h_{t+1},x_t).
$$

LSTM 还要分别记录两个方向的 cell state；GRU 则每个方向各有自己的 hidden state。不能因为最终输出形状都是 $B\times T\times2d_h$，就忽略方向内部的状态账本。

| 结构 | 两个方向是否独立 | 是否读取未来 | 额外审计点 |
| --- | --- | --- | --- |
| BiRNN | 通常是 | 完整序列时是 | vanilla 激活、时间反转 |
| BiLSTM | 是 | 完整序列时是 | $h$ 与 $c$ 两套方向边界 |
| BiGRU | 是 | 完整序列时是 | update/reset gate 的方向顺序 |
| 两层单向 RNN | 不等于双向 | 仍只看过去 | 层堆叠不能替代反向路径 |

“把序列反转后再跑一个 RNN”只实现了反向分支的一半。BiRNN 还必须把反向状态 flip 回原始时间轴，并与正向分支在每个位置 merge。

## 失败模式：shape 通过不代表语义正确

### 反向状态没有重新对齐

反向执行得到的第一项对应原始最后一个位置。若直接与正向输出按执行索引拼接，最终 tensor 的 shape 完全可能正确，但每一列表示了不同时间位置。用长度为 $4$ 的索引表逐项检查，而不是只比较维度。

### 把 $\overleftarrow h_T$ 当作反向 final state

反向循环从 $T$ 向 $1$ 读取；包含整条序列的边界是 $\overleftarrow h_1$。在 Seq2Seq bridge 中误取 $\overleftarrow h_T$ 会让 decoder 丢掉大部分源上下文。

### concat 与 add 的下游接口混用

concat 输出是 $2d_h$，add 输出是 $d_h$。如果下游线性层仍按旧宽度初始化，可能立刻 shape error；如果框架自动广播或隐式投影，问题也可能变成静默的语义变化。把 merge 规则写进模型配置和 checkpoint metadata。

### 两个方向意外共享参数

某些实现通过复用同一个 cell 实例让两方向共享权重。共享权重可以是有意的归纳偏置，但会改变参数量和梯度累加规则；不能因为参数量接近“两倍”就默认没有共享。

### padding 只屏蔽了 loss

若 padding 位置仍更新 reverse state，后面的有效位置已经被污染；这时仅在最后 loss 上乘 mask 太晚。要在状态更新、输出读取和 loss 归约三个边界分别检查 mask。

### 离线指标冒充在线指标

在完整序列上训练和验证的双向模型，天然获得未来上下文。若上线时逐 token 到达，必须重新定义评估输入为可见前缀或固定 lookahead；不能只把模型包进一个 streaming API 就声称它仍然因果。

### 方向和层次的状态顺序错接

多层 BiLSTM/GRU 的状态通常包含 layer、direction、batch、hidden 等轴。将 forward 与 backward 的排列错写，或者把 layer 轴当成 direction 轴，可能在尺寸相同的情况下悄悄改变 bridge 语义。

### 只比较参数量，不比较输出宽度和内存

BiRNN 的循环参数约是单向的两倍；concat 还使输出宽度翻倍并增加后续层成本。比较模型时同时报告方向数、每方向 hidden width、merge、每层输入宽度、激活存储和串行时间长度。

## BiRNN 审计协议

面对一个双向循环实现，可以按以下顺序留下最小可复核记录：

1. **画两个方向。** 标出正向边界 $\overrightarrow h_0$、反向边界 $\overleftarrow h_{T+1}$ 和每个原始时间位置。
2. **写依赖集合。** 确认 $\overrightarrow h_t$ 只依赖 $x_{1:t}$，$\overleftarrow h_t$ 只依赖 $x_{t:T}$。
3. **核对 reverse 对齐。** 记录反向执行下标到原始下标的映射，确认 merge 前两条状态都指向同一 $t$。
4. **冻结 merge 约定。** 记录 concat、add 或 projection，以及方向排列顺序和输出宽度。
5. **复算一个数字。** 用 $x=(1,2,0,3)$ 的线性例子核对正向 $2.5$、反向 $2.75$ 和位置 $2$ 的拼接结果。
6. **检查变长边界。** 打印每个 $T_b$、反向有效起点、padding 状态和 loss 分母。
7. **检查梯度分支。** 确认 concat 是切分上游梯度、add 是复制上游梯度，两个方向的参数梯度是否独立。
8. **确认部署合同。** 写明完整离线、固定 lookahead 或严格在线；让验证输入模拟同一可见性。
9. **记录资源账本。** 报告每方向参数、方向数、$BTd_h$ 激活、输出宽度、时间串行长度和 bridge 参数。

一个最小日志表可以是

| 字段 | 示例 | 目的 |
| --- | --- | --- |
| 输入 shape | $B\times T\times d_x=2\times5\times3$ | 固定轴顺序 |
| 方向 hidden | $d_h=4$，两个方向 | 固定循环宽度 |
| merge | concat，forward 在前 | 固定输出语义 |
| 输出 shape | $2\times5\times8$ | 核对下游接口 |
| 有效长度 | $(5,3)$ | 核对 reverse mask |
| 可见性 | offline full sequence | 排除错误的在线结论 |

## 结语

双向 RNN 用两条独立的时间路径把每个位置的上下文扩大为左右两侧：正向路径提供过去，反向路径提供未来，merge 决定下游如何读取这两种证据。它的收益来自信息可见性，代价也在于同一个事实：完整双向表示通常不是因果的。

真正可靠的 BiRNN 说明必须同时回答四个问题：反向状态是否对齐到原始位置，两个方向如何合并，padding 和 bridge 是否按有效边界处理，以及部署时未来输入是否真的可用。只要这四个问题没有被写进公式、shape 和评估协议，“双向”就仍然只是一个容易传错的模型名字。

## 相关词条

[序列建模](../rnn-lstm/sequence-modeling/)

[循环神经网络](../rnn-lstm/rnn/)

[RNN 时间展开](../rnn-lstm/rnn-unrolling/)

[时间反向传播](../rnn-lstm/bptt/)

[长短期记忆网络](../rnn-lstm/lstm/)

[门控循环单元](../rnn-lstm/gru/)

[序列到序列](../rnn-lstm/seq2seq/)

[教师强制](../rnn-lstm/teacher-forcing/)

[Bahdanau 注意力](../rnn-lstm/bahdanau-attention/)

[为什么需要注意力](../rnn-lstm/why-attention/)

[梯度裁剪](../training-nn/gradient-clipping/)
