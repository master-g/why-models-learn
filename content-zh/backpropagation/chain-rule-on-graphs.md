---
title: "计算图上的链式法则：局部 Jacobian 如何组成全局梯度"
tags: ["why-models-learn"]
---

**计算图上的链式法则**把一件长求导工作拆成两步：每条边只负责一个局部导数，整条输入到输出路径把局部导数相乘，多个路径在汇合点把贡献相加。对标量节点，这个规则表现为“路径乘积之和”；对向量节点，标量乘法升级为 Jacobian 的矩阵乘法。前向模式沿输入方向传播扰动，反向模式沿输出敏感度乘转置 Jacobian 拉回，它们不是两套微积分，而是同一条链式法则的两种遍历。

![计算图上的链式法则：前向局部值沿箭头前进，反向把每条路径的局部导数相乘并在汇合处相加](/assets/backpropagation/svg/chain-rule-on-graphs.1.svg)

本文先固定一张有分支的标量图，逐条算出路径贡献，再定义节点之间的局部 Jacobian 和图上的递推式。随后把同一结构写成向量 Jacobian、JVP 与 VJP，说明共享子图、广播、归约、ReLU 分支和时间展开如何影响求导。最后给出一个可以逐边记录的核验协议和失效模式。

## 一条边只表达局部变化

设有向边 $u\to v$ 表示节点 $v$ 依赖节点 $u$。如果节点都是标量，在当前前向值处定义局部导数

$$
J_{v,u}=\frac{\partial v}{\partial u}.
$$

这里的记号强调方向：下标前面的 $v$ 是输出，下标后面的 $u$ 是输入。小扰动满足局部近似

$$
\delta v
\approx
J_{v,u}\,\delta u.
$$

几个基本运算的局部规则是：

| 节点运算 | 对第一个输入的局部导数 | 对第二个输入的局部导数 |
| --- | --- | --- |
| $v=u+w$ | $\partial v/\partial u=1$ | $\partial v/\partial w=1$ |
| $v=uw$ | $\partial v/\partial u=w$ | $\partial v/\partial w=u$ |
| $v=u^2$ | $\partial v/\partial u=2u$ | — |
| $v=\sin u$ | $\partial v/\partial u=\cos u$ | — |
| $v=\operatorname{ReLU}(u)$ | $1_{u>0}$ | — |
| $v=\sum_i u_i$ | 对每个 $u_i$ 都是 $1$ | — |
| $v=\frac1n\sum_i u_i$ | 对每个 $u_i$ 都是 $1/n$ | — |

加法节点的两个输入都有一条边；平方节点只有一条输入边。局部规则要在当前前向值处取值：$u^2$ 的局部导数是 $2u$，不是一个与输入无关的常数；ReLU 的导数则取决于当前值落在哪一侧。

如果 $u=0$，ReLU 不可导。框架或手算必须先选定零点处的次梯度约定，例如取 $0$，并在核验记录中写明这个约定。不能一边把零点当成 $0$，另一边把它当成 $1$，再把两次结果当作同一条图的梯度。

## 标量图上的路径乘积

考虑一张有分支的图：

$$
u=x^2,
\qquad
v=xy,
\qquad
L=2u+v.
$$

从 $x$ 到 $L$ 有两条路径：

$$
x\longrightarrow u\longrightarrow L,
\qquad
x\longrightarrow v\longrightarrow L.
$$

从 $y$ 到 $L$ 只有一条路径：

$$
y\longrightarrow v\longrightarrow L.
$$

在 $x=2$、$y=3$ 处，前向值是

$$
u=2^2=4,
\qquad
v=2\cdot3=6,
\qquad
L=2\cdot4+6=14.
$$

### 对 $x$ 的两条路径

第一条路径 $x\to u\to L$ 的局部导数乘积是

$$
\frac{\partial L}{\partial u}
\frac{\partial u}{\partial x}
=2\cdot(2x)=2\cdot4=8.
$$

第二条路径 $x\to v\to L$ 的乘积是

$$
\frac{\partial L}{\partial v}
\frac{\partial v}{\partial x}
=1\cdot y=3.
$$

两条路径都改变了同一个输出 $L$，所以要相加：

$$
\frac{\partial L}{\partial x}=8+3=11.
$$

### 对 $y$ 的一条路径

因为 $u=x^2$ 不依赖 $y$，$y$ 只有通过 $v=xy$ 影响损失：

$$
\frac{\partial L}{\partial y}
=
\frac{\partial L}{\partial v}
\frac{\partial v}{\partial y}
=1\cdot x=2.
$$

直接把图合成一个函数也能核对结果：

$$
L(x,y)=2x^2+xy,
\qquad
\frac{\partial L}{\partial x}=4x+y,
\qquad
\frac{\partial L}{\partial y}=x.
$$

代入 $(x,y)=(2,3)$ 得 $(11,2)$。展开式适合检查这个小例子；当图包含共享层、矩阵运算和控制流时，保留局部边规则并按图传播更容易复用。

把这次计算压成表格，可以清楚看到“乘积”和“相加”分别发生在哪里：

| 起点 | 路径 | 局部导数链 | 路径贡献 |
| --- | --- | --- | --- |
| $x$ | $x\to u\to L$ | $\partial L/\partial u\cdot\partial u/\partial x=2\cdot4$ | $8$ |
| $x$ | $x\to v\to L$ | $\partial L/\partial v\cdot\partial v/\partial x=1\cdot3$ | $3$ |
| $y$ | $y\to v\to L$ | $\partial L/\partial v\cdot\partial v/\partial y=1\cdot2$ | $2$ |
| 汇合 | 所有到 $L$ 的 $x$ 路径 | $8+3$ | $\partial L/\partial x=11$ |

路径乘积只适用于固定了当前前向值、每条局部映射可导的情形。不可导点要有次梯度约定，循环要先展开成有限图，随机节点要固定本次执行的随机状态。

## 图上的一般递推式

令 $G=(V,E)$ 是一张 DAG，输出节点是标量损失 $L$。对每个节点 $v$ 定义伴随量

$$
\bar v=\frac{\partial L}{\partial v}.
$$

输出自身的伴随量是

$$
\bar L=\frac{\partial L}{\partial L}=1.
$$

若节点 $v$ 的后继集合为 $\operatorname{ch}(v)$，标量局部边导数是 $\partial w/\partial v$，则多元链式法则给出

$$
\bar v
=
\sum_{w\in\operatorname{ch}(v)}
\bar w\frac{\partial w}{\partial v}.
$$

这条递推从输出沿逆拓扑序执行。每个后继先把自己的敏感度乘上本地边导数，再把贡献加进当前节点的槽位。对上面的图：

$$
\bar L=1,
\qquad
\bar u=\bar L\frac{\partial L}{\partial u}=1\cdot2=2,
\qquad
\bar v=\bar L\frac{\partial L}{\partial v}=1\cdot1=1.
$$

然后回到 $x$：

$$
\bar x
=\bar u\frac{\partial u}{\partial x}
+\bar v\frac{\partial v}{\partial x}
=2\cdot4+1\cdot3=11.
$$

回到 $y$：

$$
\bar y
=\bar v\frac{\partial v}{\partial y}=1\cdot2=2.
$$

如果某个节点有三条后继边，递推式就有三项；如果节点只有一条后继，求和退化成一个乘积。梯度累加不是实现细节，而是分支图上的链式法则本身。

### 为什么必须逆拓扑序

计算 $\bar v$ 时，需要先知道所有后继 $w$ 的伴随量 $\bar w$。因此一个节点只有在所有输出方向的贡献都到达后，才能把结果传给自己的父节点。逆拓扑序保证了这一点。

若错误地先处理 $x$，只看到了 $x\to u\to L$ 这一支，就会暂时得到 $8$；等 $x\to v\to L$ 到达后还必须补上 $3$。若实现把槽位覆盖为 $3$ 而不是累加为 $11$，最终结果会少掉一整条合法路径。

### 路径公式

把递推式不断展开，可以得到标量节点的路径表示：

$$
\frac{\partial L}{\partial s}
=
\sum_{p\in\mathcal P(s,L)}
\prod_{(u\to v)\in p}
\frac{\partial v}{\partial u}.
$$

其中 $\mathcal P(s,L)$ 是从节点 $s$ 到 $L$ 的所有有向路径。每条路径内部是局部导数的乘积，路径之间是贡献的求和。实际自动微分系统通常不会显式枚举所有路径，因为路径数可能指数增长；它用伴随量槽位把相同的部分结果合并起来。

## 向量节点：标量乘法升级为 Jacobian

若 $u\in\mathbb R^{d_u}$，$v=f(u)\in\mathbb R^{d_v}$，局部导数不再是一个数，而是 Jacobian

$$
J_{v,u}
=
\frac{\partial v}{\partial u}
\in\mathbb R^{d_v\times d_u}.
$$

列向量扰动满足

$$
\delta v
\approx
J_{v,u}\delta u.
$$

如果 $u=f(x)$、$v=g(u)$，向量链式法则是

$$
J_{v,x}=J_{v,u}J_{u,x}.
$$

右边先把 $x$ 的扰动变成 $u$ 的扰动，再把它变成 $v$ 的扰动；矩阵顺序不能交换。

对标量损失 $L$，把梯度写成列向量，反向局部规则是

$$
\bar u
=
J_{v,u}^{\mathsf T}\bar v,
\qquad
\bar u=\nabla_u L,
\quad
\bar v=\nabla_v L.
$$

如果一个向量节点有多个父节点，$J_{v,u}$ 分别对应每个父节点的局部 Jacobian；如果一个节点有多个后继，各个后继的转置 Jacobian–向量积再相加。

### 一个向量 ReLU 图

取

$$
u=Ax,
\qquad
a=\operatorname{ReLU}(u),
\qquad
L=c^{\mathsf T}a,
$$

其中

$$
A=\begin{bmatrix}1&2\\-1&1\end{bmatrix},
\qquad
x=\begin{bmatrix}1\\2\end{bmatrix},
\qquad
c=\begin{bmatrix}1\\2\end{bmatrix}.
$$

前向值是

$$
u=Ax=\begin{bmatrix}5\\1\end{bmatrix},
\qquad
a=\begin{bmatrix}5\\1\end{bmatrix},
\qquad
L=c^{\mathsf T}a=7.
$$

两项预激活都在 ReLU 正侧，所以局部激活 Jacobian 是

$$
D=\operatorname{diag}(1,1)=I.
$$

输出到激活的梯度是 $\bar a=c$，因此

$$
\bar u=D^{\mathsf T}\bar a
=\begin{bmatrix}1\\2\end{bmatrix},
$$

$$
\bar x=A^{\mathsf T}\bar u
=
\begin{bmatrix}1&-1\\2&1\end{bmatrix}
\begin{bmatrix}1\\2\end{bmatrix}
=
\begin{bmatrix}-1\\4\end{bmatrix}.
$$

这里的矩阵顺序正是从输出往输入走：先乘激活 Jacobian 的转置，再乘线性层 Jacobian 的转置。

如果把输入沿方向

$$
d=\begin{bmatrix}0.1\\-0.2\end{bmatrix}
$$

扰动，前向模式先计算

$$
\dot u=Ad
=\begin{bmatrix}-0.3\\-0.3\end{bmatrix},
\qquad
\dot a=D\dot u=\begin{bmatrix}-0.3\\-0.3\end{bmatrix},
$$

再得到

$$
\dot L=c^{\mathsf T}\dot a=-0.9.
$$

反向得到的梯度与这个方向导数相容：

$$
\bar x^{\mathsf T}d
=(-1,4)\begin{bmatrix}0.1\\-0.2\end{bmatrix}
=-0.9.
$$

前向 JVP 给出一个指定方向的输出变化；反向 VJP 给出一个输出敏感度对所有输入坐标的汇总。二者在同一个标量结果上满足梯度与方向的内积关系。

## JVP 和 VJP 是两种遍历

### 前向模式：传播一条输入扰动

对每个节点保存值 $v$ 和沿给定方向的切向量 $\dot v$。若

$$
v=f(u_1,u_2),
$$

则

$$
\dot v
=J_{v,u_1}\dot u_1+J_{v,u_2}\dot u_2.
$$

例如 $v=u+w$ 时，$\dot v=\dot u+\dot w$；$v=uw$ 时，

$$
\dot v=w\dot u+u\dot w.
$$

一条 JVP 从输入方向开始，沿正拓扑序运行到输出。它适合输入维度较小、要计算少量方向导数，或需要 Hessian–向量积的场景。

### 反向模式：传播一个输出敏感度

反向模式从输出伴随量开始。对一个后继 $v=f(u)$，

$$
\bar u=J_{v,u}^{\mathsf T}\bar v.
$$

多个后继则是

$$
\bar u
=\sum_{v\in\operatorname{ch}(u)}
J_{v,u}^{\mathsf T}\bar v.
$$

一条 VJP 从输出沿逆拓扑序回到输入。标量损失只有一个输出方向，因此神经网络训练常用一次反向遍历就能得到许多参数的梯度；这不是因为反向规则更正确，而是输出维度和输入维度的计算预算不同。

### 维度决定选择

设函数 $f:\mathbb R^n\to\mathbb R^m$，完整 Jacobian 是 $m\times n$。若只需要一个输入方向的 JVP，可以避免显式构造整张矩阵；若只需要一个标量输出对全部参数的梯度，VJP 也避免为每个输入坐标单独做一次前向传播。

| 需求 | 传播的对象 | 典型遍历 | 结果 |
| --- | --- | --- | --- |
| 给定输入方向 $d$ | $\dot x=d$ | 正拓扑序 | $Jd$ |
| 给定输出权重 $r$ | $\bar y=r$ | 逆拓扑序 | $r^{\mathsf T}J$ 或 $J^{\mathsf T}r$ |
| 标量损失对参数 | $\bar L=1$ | 逆拓扑序 | 每个参数的梯度 |
| 完整 Jacobian | 多个方向 | 多次 JVP 或 VJP | $m\times n$ 矩阵 |

表中行向量和列向量写法只差转置，但实现必须统一。本文使用列梯度，所以 VJP 写成 $J^{\mathsf T}r$；若把伴随量写成行向量，则会看到 $r^{\mathsf T}J$。

## 共享子图、广播和归约

### 共享节点不复制梯度

如果一个节点 $s$ 同时供两个后继使用，例如

$$
q=s^2,
\qquad
r=3s,
\qquad
L=q+r,
$$

则

$$
\bar s
=\bar q\frac{\partial q}{\partial s}
+\bar r\frac{\partial r}{\partial s}.
$$

共享节点的值只需前向计算一次，但伴随量必须收集所有后继贡献。把一个共享节点复制成两份，前向数值可能仍相同；如果反向没有把两份对应的贡献合并，得到的却不是原图的梯度。

### 广播边在反向会变成求和

前向 batch bias 的广播可以写成

$$
Z=XW^{\mathsf T}+\mathbf 1_Bb^{\mathsf T}.
$$

一个长度为 $d$ 的 $b$ 被复制到 $B$ 行。若上游梯度为 $G_Z\in\mathbb R^{B\times d}$，则 bias 的梯度是沿 batch 轴求和：

$$
g_b=G_Z^{\mathsf T}\mathbf 1_B.
$$

这正是“多个复制边的贡献相加”。把广播误当成只存在一条边，会让 $g_b$ 少掉 batch 中其余样本的贡献。

### reduction 会改变局部 Jacobian

对向量 $u=(u_1,u_2,u_3)$，求和节点和平均节点分别是

$$
s=\sum_{i=1}^{3}u_i,
\qquad
m=\frac13\sum_{i=1}^{3}u_i.
$$

它们对输入的 Jacobian 是

$$
J_{s,u}=\begin{bmatrix}1&1&1\end{bmatrix},
\qquad
J_{m,u}=\begin{bmatrix}\frac13&\frac13&\frac13\end{bmatrix}.
$$

前向输出只差一个尺度，反向传回每个分量的局部系数也差三倍。对 mask 平均，分母还要改成有效元素个数；不能用固定总长度代替实际 reduction 规则。

## 非光滑节点与实际执行路径

链式法则在每条边上使用当前执行点的局部规则，因此以下情况需要把路径条件写进图语义。

### ReLU 的活动分支

对逐分量 ReLU，若 $u_i>0$，局部导数为 $1$；若 $u_i<0$，局部导数为 $0$。因此

$$
J_{a,u}=\operatorname{diag}(1_{u_1>0},\ldots,1_{u_d>0}).
$$

一个 batch 中不同样本可能选中不同的对角线 mask。反向需要使用本次前向实际得到的 mask，而不是根据另一批输入重新判断。

### 条件分支

如果程序根据输入选择 $v=f(u)$ 或 $v=g(u)$，当前执行图只包含实际走过的分支。沿未执行分支补一条梯度路径，会把另一个程序的导数混进来。若要对分支边界做分析，还需要单独讨论分段函数在边界处的连续性和可导性。

### 随机节点

dropout、采样或随机增强会让一次执行包含随机状态 $\omega$。固定 $\omega$ 后，链式法则作用于

$$
L=f_{\theta}(x;\omega).
$$

反向要复用同一个 mask 或随机选择；否则前向和反向对应的是两张不同的执行图。若目标是期望损失的梯度，还要进一步区分“固定随机样本的梯度估计”和“对随机变量取期望后的导数”。

## 时间展开只是更长的 DAG

循环更新

$$
h_t=F(h_{t-1},x_t;\theta),
\qquad
L=\sum_{t=1}^{T}\ell_t(h_t).
$$

在固定有限长度 $T$ 后，可以把 $h_0,h_1,\ldots,h_T$ 展开成一张 DAG。参数 $\theta$ 在每个时间步被共享，所以它到 $L$ 有多条跨时间路径：

$$
\frac{\partial L}{\partial\theta}
=\sum_{t=1}^{T}\sum_{s\le t}
\frac{\partial\ell_t}{\partial h_t}
\frac{\partial h_t}{\partial h_{t-1}}\cdots
\frac{\partial h_s}{\partial\theta}.
$$

这里的连乘正是路径上的局部 Jacobian 乘积，跨时间的求和正是共享参数收到的多条路径贡献。[前向计算](../backpropagation/forward-pass/)负责按时间顺序产生 $h_t$；[反向传播](../backpropagation/backpropagation/)或 BPTT 再按逆时间顺序累加。无限时间循环不是一张有限 DAG，需要额外的固定点或截断语义。

## 一份逐边核验协议

求导实现不应只拿最终参数梯度和一个黑盒库对比。先在一个小图上把节点、边、局部 Jacobian、前向值和伴随量都记下来：

| 检查项 | 要回答的问题 | 最小证据 |
| --- | --- | --- |
| 节点 | 输入、常量、运算和输出是否完整 | 节点表与拓扑序 |
| 前向值 | 每个局部规则使用的当前值是什么 | 每个节点的一次数值 |
| 边方向 | $u\to v$ 的输入输出顺序是否固定 | 边列表或依赖图 |
| Jacobian | 局部导数的形状和转置方向是什么 | $J_{v,u}$ 的 shape |
| 分支 | 一个节点是否有多个后继 | 每条路径贡献 |
| 累加 | 汇合节点是否把所有后继贡献相加 | 槽位更新记录 |
| 广播 | 复制到哪些轴，反向沿哪些轴求和 | 广播轴与归约轴 |
| 非光滑 | ReLU 零点或分段边界采用什么规则 | 次梯度约定 |
| 随机性 | 前向与反向是否共享 mask 或随机状态 | seed、mask 或状态 |
| 模式 | 训练态与推理态是否生成同一条图 | mode 与统计量来源 |
| 数值 | JVP、VJP 与有限差分是否相容 | 方向导数和差分结果 |

对标量图 $u=x^2,v=xy,L=2u+v$，固定 $(x,y)=(2,3)$ 时，最小记录是

$$
(u,v,L)=(4,6,14),
\qquad
(\bar u,\bar v,\bar x,\bar y)=(2,1,11,2).
$$

再取方向 $d=(0.1,-0.2)$，解析梯度给出的方向导数是

$$
\begin{bmatrix}11&2\end{bmatrix}
\begin{bmatrix}0.1\\-0.2\end{bmatrix}
=0.7.
$$

直接把输入改为 $(2+\varepsilon\cdot0.1,3-\varepsilon\cdot0.2)$，用中心差分核对

$$
\frac{L(x+\varepsilon d)-L(x-\varepsilon d)}{2\varepsilon}
\approx0.7.
$$

方向导数核对的是梯度在一个方向上的投影；多个独立方向或逐坐标有限差分可以进一步覆盖更多输入分量，但都不能替代对共享边和 reduction 轴的结构检查。

## 失效模式

**把局部导数当成全局梯度。** 一条边的 Jacobian 只说明相邻两个节点的变化关系；要得到参数到损失的导数，还需乘完路径上的所有局部规则。

**乘了路径却忘记相加。** 一个参数经多条分支影响同一损失时，每条路径的乘积都要加入同一个伴随量槽位。覆盖写入会丢掉先到达的贡献。

**Jacobian 顺序反了。** 对 $x\to u\to v$，正确顺序是 $J_{v,u}J_{u,x}$；反向列梯度则是先乘 $J_{v,u}^{\mathsf T}$，再乘 $J_{u,x}^{\mathsf T}$。方阵例子可能掩盖错误，非方阵探针更容易发现。

**把行梯度和列梯度混用。** $r^{\mathsf T}J$ 与 $J^{\mathsf T}r$ 表达同一 VJP 的两种记法，但不能在同一段实现中交替使用而不说明形状。

**在错误的前向点取局部导数。** 激活 mask、softmax 概率、归一化统计量和乘法另一侧的值都来自本次前向；拿另一批输入的缓存会产生另一张图的局部规则。

**广播只复制、不求和。** 前向 bias 被复制到多个样本，反向必须把这些复制边的贡献沿广播轴相加。

**把 sum、mean 和 mask mean 当成同一个节点。** 它们的前向尺度和局部 Jacobian 不同；只比较一个未标注 reduction 的 loss，无法判断梯度尺度。

**在未执行分支上补梯度。** 当前图只包含本次条件判断真正走过的路径；未执行代码不是这次函数的输入输出关系。

**随机状态不一致。** dropout mask 或采样索引在重算和反向时变化，等于把局部导数接到了另一张执行图上。

**在 ReLU 零点隐式换次梯度。** 数值恰好为零时，必须固定实现约定并在测试中覆盖该边界，不能让不同 kernel 自行选择。

**显式枚举所有路径。** 路径公式有助于理解，但深层共享图的路径数可能指数增长；实际实现应使用拓扑递推和累加槽位，不能把路径枚举当成可扩展算法。

## 相关词条

- [链式法则](../calculus/chain-rule/)：一元复合函数局部变化率相乘的基础。
- [向量链式法则](../calculus/vector-chain-rule/)：把标量乘积推广为 Jacobian 的矩阵复合。
- [Jacobian 矩阵](../calculus/jacobian/)：定义向量节点的局部线性映射。
- [计算图](../backpropagation/computational-graphs/)：定义节点、边、拓扑序、共享子图和一次执行记录。
- [前向计算](../backpropagation/forward-pass/)：沿拓扑顺序填入本次执行的节点值。
- [反向传播](../backpropagation/backpropagation/)：用逆拓扑递推计算标量损失对节点和参数的伴随量。
- [自动微分](../calculus/automatic-differentiation/)：在实际执行的基本运算上自动应用这些局部规则。
- [广播与归约求导](../calculus/broadcast-and-reduction-derivatives/)：展开复制边与 reduction 节点的局部 Jacobian。
- [梯度检查](../backpropagation/gradient-checking/)：用有限差分核对 JVP、VJP 和参数梯度。
- [梯度消失与爆炸](../backpropagation/vanishing-and-exploding/)：分析深层或时间展开路径上的 Jacobian 连乘尺度。
- [单神经元反向传播](../backpropagation/backprop-single-neuron/)：在最小网络上把图规则落到权重、偏置和输入。
