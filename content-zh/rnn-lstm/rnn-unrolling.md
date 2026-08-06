---
title: "RNN 时间展开：把循环写成可审计的计算图"
tags: ["why-models-learn"]
---

RNN 的循环写法把很多次状态更新压缩成一行：

$$
h_t=f_\theta(h_{t-1};x_t)
$$

时间展开（unrolling）把长度为 $T$ 的这条递归链写成 $T$ 个计算节点：

$$
\begin{aligned}
h_1&=f_\theta(h_0,x_1),\\
h_2&=f_\theta(h_1,x_2),\\
&\ \vdots\\
h_T&=f_\theta(h_{T-1},x_T).
\end{aligned}
$$

展开后的节点各自有不同的输入和状态，却共享同一个 $\theta$。这不是把参数复制成 $T$ 份，而是把同一组参数在一张更长的图上重复使用。图展开后，输入轴、状态轴、输出轴、mask、缓存和梯度路径都能逐项检查；[时间反向传播](../rnn-lstm/bptt/) 会在这张图上反向计算，[循环神经网络](../rnn-lstm/rnn/) 负责定义单个 cell。

本文先固定索引约定，再用线性 RNN 完成一遍前向和路径贡献的手算，接着说明完整展开、动态循环、截断展开、可变长度和多层展开的差别，最后给出一份从节点到 loss 的审计清单。

![RNN 时间展开把共享的 cell 写成一条计算图：每个时间步读取当前输入和上一步状态，参数标签在各节点间复用](/assets/rnn-lstm/svg/rnn-unrolling.1.svg)

## 一行递推到底隐藏了什么

### 递推关系包含一条状态边

以 vanilla RNN 为例：

$$
\begin{aligned}
a_t&=W_{xh}x_t+W_{hh}h_{t-1}+b_h,\\
h_t&=\phi(a_t),\\
o_t&=W_{hy}h_t+b_y.
\end{aligned}
$$

每个时间步至少有三类节点：

| 节点 | 输入 | 输出 | 是否连接到下一步 |
| --- | --- | --- | --- |
| 输入节点 | $x_t$ | 当前观测向量 | 通过 $h_t$ 间接影响未来 |
| 状态节点 | $x_t,h_{t-1}$ | $a_t,h_t$ | $h_t$ 传给 $t+1$ |
| 输出节点 | $h_t$ | $o_t,\widehat y_t$ | 由损失决定是否回传 |

循环不是“把前一个输出当作下一个输入”这么简单。RNN 同时有输入轴和状态轴：$x_t$ 沿时间被读入，$h_t$ 沿时间传递，输出可能在每一步产生，也可能只在最后一步读取。

### 时间展开后的节点命名

长度为 $4$ 的展开图可以列成：

$$
\begin{aligned}
h_0&\xrightarrow[\ x_1\ ]{\theta}h_1
\xrightarrow[\ x_2\ ]{\theta}h_2
\xrightarrow[\ x_3\ ]{\theta}h_3
\xrightarrow[\ x_4\ ]{\theta}h_4,\\
o_t&=g_\theta(h_t).
\end{aligned}
$$

箭头上标 $\theta$ 表示四条状态边使用同一组参数，不表示每条边有一份独立权重。$h_0$ 是图的起点；如果它由外部上下文产生，图还应包含 $h_0=q_\theta(c)$ 这条上游路径。

### 展开是空间复制，参数不是

可以把图中出现的对象分成两类：

| 类型 | 例子 | 是否随 $T$ 增加 |
| --- | --- | --- |
| 参数 | $W_{xh},W_{hh},b_h,W_{hy},b_y$ | 不增加 |
| 时间状态 | $h_1,\ldots,h_T$ | 增加 $T$ 个 |
| 预激活与激活缓存 | $a_1,\ldots,a_T$ | 通常增加 $T$ 个 |
| 每步输出与 loss | $o_t,\ell_t$ | 按任务增加 |

参数共享使模型可以面对不同长度，状态缓存却仍然与长度成正比。把“参数量不变”误读成“运行内存不变”，会漏掉反向所需的中间激活。

## 索引合同：先决定每个下标表示什么

### 输入时间步从 $1$ 开始，状态从 $0$ 开始

最常用的约定是：

$$
x_{1:T}=(x_1,\ldots,x_T),\qquad
h_0\text{ 已知},\qquad
h_t=f_\theta(h_{t-1};x_t)
$$

这样 $x_1$ 与 $h_0$ 共同产生 $h_1$。如果实现把循环写成从数组下标 $0$ 开始，代码下标和数学下标可以错开，但文档必须说明：

| 数学对象 | 常见数组位置 | 需要确认 |
| --- | --- | --- |
| $x_1$ | X[batch; time; feature] 或 X[time; batch; feature] | 哪一维是时间 |
| $h_0$ | 单独的初始状态 | 是否包含在状态缓存中 |
| $h_T$ | 最后一次 cell 的返回值 | 是否被误当成 $h_{T-1}$ |
| $y_{t+1}$ | 右移后的 target | 最后一项是否有 mask |

### 当前输出和下一步目标是两个任务

逐步标注可以写成

$$
\widehat y_t=g_\theta(h_t),\qquad
\ell_t=\ell(y_t,\widehat y_t).
$$

next-step 预测则常写成

$$
\widehat y_{t+1}=g_\theta(h_t),\qquad
\ell_t=\ell(y_{t+1},\widehat y_{t+1}).
$$

展开图中的输出节点位置相同，目标边的下标不同。只看输出 tensor 的 shape 不能区分这两种任务；必须把输入、输出和 target 在同一条时间线画出来。

### 一个长度为三的对齐图

若输入是 $(x_1,x_2,x_3)$，next-step 目标是 $(y_2,y_3)$：

$$
\begin{array}{c@{\quad}ccc}
\text{状态} & h_1 & h_2 & h_3\\
\hline
\text{输入} & x_1 & x_2 & x_3\\
\text{预测} & \widehat y_2 & \widehat y_3 & \text{无目标}\\
\text{目标} & y_2 & y_3 & \text{mask}=0
\end{array}
$$

也可以把输入右移，用起始符号补齐长度；两种实现都可以，关键是同一位置的预测和目标具有相同的语义。

## 线性展开例子：每条时间边贡献多少

### 设一个可手算的 cell

为了不让 $\tanh$ 遮住图结构，先使用一维线性 RNN：

$$
h_t=0.5h_{t-1}+x_t,\qquad h_0=0.
$$

输入取

$$
(x_1,x_2,x_3)=(1,2,0).
$$

逐步展开：

$$
\begin{aligned}
h_1&=0.5\times0+1=1,\\
h_2&=0.5\times1+2=2.5,\\
h_3&=0.5\times2.5+0=1.25.
\end{aligned}
$$

如果输出头是 $\widehat y_t=2h_t$，输出序列为

$$
(\widehat y_1,\widehat y_2,\widehat y_3)=(2,5,2.5).
$$

这里的 $h_2$ 包含 $x_1$ 的一次衰减贡献和 $x_2$ 的直接贡献；$h_3$ 还继续携带它们：

$$
h_3
=0.5^2x_1+0.5x_2+x_3
=0.25+1+0
=1.25.
$$

### 末端损失如何看到每个输入

假设只在末端使用目标 $y_3=1$，损失为

$$
\mathcal L=\frac12(\widehat y_3-y_3)^2
=\frac12(2.5-1)^2
=1.125.
$$

先从输出往回：

$$
\frac{\partial\mathcal L}{\partial h_3}
=(\widehat y_3-y_3)\frac{\partial\widehat y_3}{\partial h_3}
=1.5\times2
=3.
$$

状态路径的局部导数是

$$
\frac{\partial h_t}{\partial h_{t-1}}=0.5.
$$

于是三个输入到末状态的路径系数与末端 loss 梯度为：

| 输入 | 到 $h_3$ 的路径系数 | 对 $\mathcal L$ 的直接贡献 |
| --- | ---: | ---: |
| $x_1$ | $\partial h_3/\partial x_1=0.5^2=0.25$ | $3\times0.25=0.75$ |
| $x_2$ | $\partial h_3/\partial x_2=0.5$ | $3\times0.5=1.5$ |
| $x_3$ | $\partial h_3/\partial x_3=1$ | $3$ |

这里的“直接贡献”指通过末状态这一条路径的导数，不包括其它输出 loss。它清楚显示了时间距离：越早的输入，要乘越多次状态转移导数。

### 共享 recurrent weight 的梯度

令 recurrent 系数为 $a$：

$$
h_t=ah_{t-1}+x_t.
$$

对上面的三步序列：

$$
h_3=a^2x_1+ax_2+x_3.
$$

所以

$$
\frac{\partial h_3}{\partial a}
=2ax_1+x_2.
$$

在 $a=0.5$ 处：

$$
\frac{\partial h_3}{\partial a}
=2\times0.5\times1+2=3,
\qquad
\frac{\partial\mathcal L}{\partial a}
=3\times3=9.
$$

这个 $9$ 是三个时间位置共同作用后的结果；如果错误地把每个时间步的 $a$ 当成独立参数，就不会得到同一个共享权重的时间求和。

## 形状展开：从三维输入到状态缓存

### batch-first 约定

设

$$
X\in\mathbb R^{B\times T\times d_x},
\qquad
H\in\mathbb R^{B\times T\times d_h}.
$$

第 $t$ 步取

$$
X_t=X[:,t,:]\in\mathbb R^{B\times d_x},
\qquad
H_t=H[:,t,:]\in\mathbb R^{B\times d_h}.
$$

row-batch 写法可以是

$$
A_t=X_tW_{xh}^{\mathsf T}
+H_{t-1}W_{hh}^{\mathsf T}
+\boldsymbol1 b_h^{\mathsf T},
\qquad
H_t=\phi(A_t).
$$

其中 $\boldsymbol1\in\mathbb R^{B}$ 广播到 batch。输出 logits 形状为

$$
O\in\mathbb R^{B\times T\times d_y}.
$$

### time-first 约定

如果输入排成

$$
X\in\mathbb R^{T\times B\times d_x},
$$

第 $t$ 步取的是 $X_t=X[t,:,:]$。数学递推不变，但数组切片位置变了。把 batch-first 张量直接交给 time-first cell，可能仍然产生可乘的矩阵，却把原本不同样本的向量当成时间连续输入。

### 展开后的形状账本

取 $B=2,T=4,d_x=3,d_h=5,d_y=2$：

| 对象 | 形状 | 作用 |
| --- | --- | --- |
| $X$ | $2\times4\times3$ | 两条样本、四个时间步、每步三维输入 |
| $h_0$ | $2\times5$ | 每条样本的起始状态 |
| $A_t$ | $2\times5$ | 第 $t$ 步预激活 |
| $H$ | $2\times4\times5$ | 四个时间步的状态缓存 |
| $O$ | $2\times4\times2$ | 每步两个输出分数 |
| $M$ | $2\times4$ | 有效时间步 mask |

若是 many-to-one，最终只读 $H[:,T-1,:]$，输出可以是 $2\times d_y$；若是 many-to-many，保留完整 $O$。状态缓存的形状不能由最终输出的形状反推。

## 完整展开与动态循环

### 静态图的优点

完整展开会显式构造

$$
h_1,h_2,\ldots,h_T,
\qquad
a_1,a_2,\ldots,a_T.
$$

优点是每条边和每个中间量都可定位：

| 观察对象 | 静态展开能直接回答的问题 |
| --- | --- |
| 输入边 | 第 $t$ 步究竟读了哪个 $x_t$ |
| 状态边 | $h_t$ 是否传给了正确的下一步 |
| 参数边 | 所有 cell 是否引用同一个 $\theta$ |
| 输出边 | loss 对应当前标签还是下一标签 |
| mask 边 | padding 是否在正确的时间位置被屏蔽 |

在调试阶段，可以只用 $T=3$ 的固定小张量，把每个节点打印出来，再扩展到真实长度。

### 动态循环的优点

动态循环用一个 cell 在运行时循环中重复调用：

$$
h\leftarrow h_0;
\qquad
\text{for }t=1,\ldots,T:
\quad
h\leftarrow f_\theta(h,x_t).
$$

它节省了显式图的编排代码，也自然支持不同长度；自动微分系统通常仍会记录每次调用产生的计算图节点。动态循环不等于没有展开，只是展开发生在运行时。

### 两者必须保持相同语义

| 方面 | 静态展开 | 动态循环 |
| --- | --- | --- |
| 参数 | 所有节点引用同一对象 | cell 被多次调用 |
| 状态 | 节点间显式传递 | 循环变量传递 |
| 长度 | 构图时固定 | 运行时读取长度 |
| 调试 | 节点容易逐个检查 | 需要记录每步缓存 |
| 反向 | 图中有完整时间路径 | tape 记录调用历史 |

如果静态和动态实现的 loss 不一致，先用同一组参数、同一 $h_0$ 和 $T=3$ 对齐 $a_t,h_t,o_t$，不要直接比较最终训练曲线。

## padding、变长与展开边界

### 统一到 $T_{\max}$ 不等于真的有 $T_{\max}$ 步

两个样本长度为 $4$ 和 $2$ 时，可以补成：

$$
M=
\begin{bmatrix}
1&1&1&1\\
1&1&0&0
\end{bmatrix}.
$$

状态缓存可以保持 $B\times T_{\max}\times d_h$，但 mask 至少要约束 loss：

$$
\mathcal L
=
\frac{\sum_{b,t}m_{b,t}\ell_{b,t}}
{\sum_{b,t}m_{b,t}}.
$$

如果 padding 时间步还会更新状态，则需要额外决定它是否会影响后续状态。常见选择是：

$$
h_t=
\begin{cases}
f_\theta(h_{t-1},x_t),&m_t=1,\\
h_{t-1},&m_t=0.
\end{cases}
$$

也可以让实现用 packed sequence 直接跳过无效 cell；无论用哪种 API，最终状态轨迹都要符合同一合同。

### 每个样本的末状态位置不同

若长度为 $T_b$，many-to-one 读取的是

$$
h_{T_b}^{(b)},
$$

不是统一 padding 张量的最后一个位置 $h_{T_{\max}}^{(b)}$。后者可能是 padding 继续更新后的状态。

实现中常见三种做法：

| 做法 | 读取位置 | 需要检查 |
| --- | --- | --- |
| 按长度 gather | $H[b,T_b-1,:]$ | 长度是否从 1 开始、是否越界 |
| 状态冻结 | 最后有效状态之后重复旧值 | mask 是否同步 |
| packed sequence | 直接返回各序列末状态 | packed 顺序和还原顺序 |

### 末端 target 的 mask

next-step 预测的最后一个输入没有下一个目标。若输入长度为 $T$，有效 next-step loss 通常只有 $T-1$ 个：

$$
m_t=
\begin{cases}
1,&1\le t<T,\\
0,&t=T.
\end{cases}
$$

可变长度 batch 中，还要把每个样本的最后一步分别置零，而不是只把整个 batch 的最后一列置零。

## chunk 与 truncated unroll

### 长流为什么要切块

完整长度 $T$ 的展开需要保存每个时间步的中间激活。若 batch 为 $B$、隐藏维为 $d_h$，只算状态缓存就有近似

$$
\text{memory}_{H}\propto BTd_h.
$$

预激活、输出、mask、优化器和其他层会增加常数。把长流切成每块 $K$ 步，可将反向图的时间深度限制在 $K$：

$$
(x_1,\ldots,x_T)
\longrightarrow
\underbrace{(x_1,\ldots,x_K)}_{\text{chunk 1}}
\longrightarrow
\underbrace{(x_{K+1},\ldots,x_{2K})}_{\text{chunk 2}}
\longrightarrow\cdots.
$$

### carry 和 detach 的组合

第 $j$ 个 chunk 接收上一块末状态：

$$
h_{j,0}=h_{j-1,K}.
$$

若做 truncated BPTT，通常使用

$$
h_{j,0}=\operatorname{detach}(h_{j-1,K}),
$$

让前向状态连续、反向只在块内传播。若 reset 成零，前向依赖也被切断，模型不再看到跨块历史。

| 边界动作 | 下一块看到的数值 | 梯度能否穿过边界 |
| --- | --- | --- |
| reset | 新的 $h_0$ | 不能 |
| carry | 上一块末状态 | 能，若不 detach |
| carry + detach | 上一块末状态 | 不能穿过，但块内仍可反向 |

“截断时间反向”只限制梯度图，不自动限制前向状态的数值影响；“重置状态”则同时改变前向和反向。

### chunk 的 loss 归约

如果每块的有效 token 数不同，不能简单平均每块 mean loss：

$$
\frac12\left(
\frac{\sum_{t\in\text{chunk 1}}\ell_t}{N_1}
+
\frac{\sum_{t\in\text{chunk 2}}\ell_t}{N_2}
\right)
$$

会让短块和长块获得相同块权重。全序列 token mean 应写成

$$
\frac{\sum_j\sum_{t\in j}m_t\ell_t}
{\sum_j\sum_{t\in j}m_t}.
$$

如果产品目标确实是每个 chunk 等权，则可以采用前一种归约，但必须把它当作明确的加权选择，而不是无意中的 padding 或 batch bug。

## 多层 RNN 的二维展开

### 时间轴和层轴同时展开

两层 RNN：

$$
\begin{aligned}
h_t^{(1)}&=f_{\theta_1}(h_{t-1}^{(1)},x_t),\\
h_t^{(2)}&=f_{\theta_2}(h_{t-1}^{(2)},h_t^{(1)}).
\end{aligned}
$$

展开后有两个方向的边：

| 边方向 | 连接 | 影响 |
| --- | --- | --- |
| 时间边 | $h_{t-1}^{(l)}\to h_t^{(l)}$ | 长度方向的状态与梯度路径 |
| 层间边 | $h_t^{(l-1)}\to h_t^{(l)}$ | 同一时间步的深度路径 |
| 输出边 | $h_t^{(L)}\to o_t$ | 任务 loss 的读取位置 |

总路径深度不是只看层数或只看序列长度。报告“二层 RNN 处理 100 个时间步”时，至少要同时说明沿时间经过多少次 cell、每层隐藏维、是否有截断。

### 参数共享的范围

同一层的不同时间步共享 $\theta_l$，不同层通常使用不同的 $\theta_l$：

$$
\theta_1\ne\theta_2,
\qquad
\theta_l^{(t)}=\theta_l^{(s)}
\quad\text{对同一层的时间步 }s,t.
$$

如果把所有层也共享成同一个参数集合，模型结构又变了；如果把同层时间步参数复制，模型也不再是普通 RNN。参数表应该同时标记 layer id 和 time id。

## 展开图上的梯度路径预览

### 状态 Jacobian 沿时间相乘

对

$$
h_t=\phi(W_{hh}h_{t-1}+W_{xh}x_t+b_h),
$$

有

$$
\frac{\partial h_t}{\partial h_{t-1}}
=D_tW_{hh},
\qquad
D_t=\operatorname{diag}\bigl(\phi'(a_t)\bigr).
$$

从时间 $s$ 回到时间 $T$：

$$
\frac{\partial h_T}{\partial h_s}
=
(D_TW_{hh})(D_{T-1}W_{hh})\cdots(D_{s+1}W_{hh}).
$$

展开图让这条产品变成一串实际边，而不是循环体里看不见的递归。若模型在每个时间步都有 loss，则从 $h_s$ 出发还会有多条通向不同输出的路径：

$$
\frac{\partial\mathcal L}{\partial h_s}
=
\sum_{t=s}^{T}
\frac{\partial\mathcal L_t}{\partial h_t}
\frac{\partial h_t}{\partial h_s}.
$$

[时间反向传播](../rnn-lstm/bptt/) 会把局部 VJP、参数梯度累加和截断算法写完整；[RNN 中的梯度消失](../rnn-lstm/rnn-vanishing-gradient/) 会专门检查长乘积造成的数值现象。

### 共享参数收到多条入边

把时间步看成参数节点 $\theta$ 的多次使用，参数梯度是所有使用点的贡献相加：

$$
\frac{\partial\mathcal L}{\partial\theta}
=
\sum_{t=1}^{T}
\frac{\partial\mathcal L}{\partial\theta^{(t)}}.
$$

这条求和是展开图中“多条边汇入同一参数”的数学表达。若实现用 inplace 覆盖中间量、错误清空梯度或把 cell 复制成不同参数，梯度图和模型语义都会改变。

## 失效模式：展开后才看得见的错

### 把参数共享写成参数复制

症状是序列长度固定时训练正常，换长度或测试不同位置的事件时表现骤降。检查每个时间节点的参数对象、权重 id 或梯度是否指向同一矩阵；数值相同的两份矩阵也不等于参数共享。

### $h_0$ 被错当成输入 $x_0$

$h_0$ 是状态起点，$x_1$ 是第一个观测。若把 padding、起始符号、初始状态混成一个变量，训练第一步和序列边界会悄悄改变。打印第一步的 $a_1$，分别列出 $W_{xh}x_1$、$W_{hh}h_0$ 和 $b_h$。

### 输出和 target 错位

损失下降不代表下标正确。用长度为 $3$ 的人工序列，让每个 $y_t$ 都是易识别的不同数字，逐个打印输出节点与 target 节点的对应关系。

### padding 仍然创建状态路径

统一长度后，padding 位置可能仍产生 $h_t$。如果它被拿来做 many-to-one 的末状态，短序列的预测会依赖补齐长度；如果它进入归一化分母，短样本的权重也会变化。

### detach 放在了错误边界

detach 太早会切断块内需要的梯度，detach 太晚会让计算图跨越很多 chunk、内存上涨。记录每块的起始状态是否有梯度、末状态的数值和图深度，才能区分这两种错误。

### 静态和动态实现只有最终 loss 对齐

最终 loss 偶然相同不能证明每一步相同。对固定 $T=3$，逐步比较 $a_t,h_t,o_t,\ell_t$；最早出现差异的节点就是修复边界。

## 一份时间展开审计协议

1. **写出索引。** 明确 $h_0$、$x_{1:T}$、输出是当前步还是下一步。
2. **画出短图。** 至少展开三步，标记输入边、状态边、输出边和共享参数。
3. **列形状。** 记录 $X,h_0,H,O,M$，注明 batch-first 或 time-first。
4. **核对参数身份。** 确认所有时间节点使用同一 cell 参数，不把共享误写成复制。
5. **复算节点。** 用小数值逐步算 $a_t,h_t$，至少覆盖两个时间步。
6. **核对末状态。** 变长序列按有效长度读取，不从 padding 末端盲取。
7. **核对 chunk。** 说明 reset、carry、detach、loss 分母和截断窗口。
8. **看反向路径。** 记录时间 Jacobian、梯度范数、非有限值与参数梯度是否跨时间累加。
9. **分别测短长序列。** 把前向正确性、内存增长、训练稳定性和泛化表现分开报告。

最终要得到的不是一张漂亮的展开图，而是一份能回答“哪个时间步读取了什么、状态在哪里传递、哪个 loss 通过哪条路径更新了共享参数”的账本。

## 相关词条

[序列建模](../rnn-lstm/sequence-modeling/)

[循环神经网络](../rnn-lstm/rnn/)

[时间反向传播](../rnn-lstm/bptt/)

[RNN 中的梯度消失](../rnn-lstm/rnn-vanishing-gradient/)

[教师强制](../rnn-lstm/teacher-forcing/)

[长短期记忆网络](../rnn-lstm/lstm/)

[门控循环单元](../rnn-lstm/gru/)

[序列到序列](../rnn-lstm/seq2seq/)

[双向循环网络](../rnn-lstm/bidirectional-rnn/)

[梯度检查](../backpropagation/gradient-checking/)

[训练调试](../training-nn/debugging-training/)
