---
title: "GQA 与 MQA：为什么让多个 query head 共享 K/V"
tags: ["why-models-learn"]
---

Grouped-Query Attention（GQA，分组查询注意力）保留多个独立的 query head，只把 key/value head 的数量减少到 $h_{kv}$ 个；同一组 query head 共享一组 K/V。Multi-Query Attention（MQA，多查询注意力）是 GQA 的极端情形，所有 query head 共享同一组 K/V。两者改变的是 K/V 投影、KV cache 和反向梯度的汇合方式，query head 的数量与 attention score 的逻辑数量仍由 $h_q$ 决定。本篇从 Multi-Head Attention（MHA）的基线出发，推导 head 映射、参数量、缓存容量、梯度路径和实现中的轴约定。

![四个 query head 映射到两个 K/V 组，组内共享 K/V 但保留独立 query 和 attention weight，并对比 MHA、GQA 与 MQA 的 KV cache](/assets/attention/svg/gqa-and-mqa.1.svg)

## 先把三种 attention 放在同一张图里

### MHA、GQA 和 MQA 的 head 数量

设 decoder 或 self-attention 层有 $h_q$ 个 query head。标准 MHA 还使用 $h_q$ 个 K/V head，每个 query head 都有自己的 K/V。GQA 使用较少的 $h_{kv}$ 个 K/V head，并把 query head 分成大小相等的组。令：

$$
r=\frac{h_q}{h_{kv}},
\qquad
g(q)=\left\lfloor\frac{q}{r}\right\rfloor,
\qquad
q=0,\ldots,h_q-1.
$$

这里假定 $h_{kv}$ 整除 $h_q$，$r$ 是每组包含的 query head 数，$g(q)$ 是 query head $q$ 使用的 K/V 组编号。三种配置可以写成：

|模式|query head 数|K/V head 数|每组 query head 数|共享范围|
|---|---:|---:|---:|---|
|MHA|$h_q$|$h_q$|$1$|不共享 K/V|
|GQA|$h_q$|$h_{kv}$，$1<h_{kv}<h_q$|$r=h_q/h_{kv}$|组内共享 K/V|
|MQA|$h_q$|$1$|$h_q$|所有 query head 共享 K/V|

MQA 仍有多个 query head。它不是把整个 attention 层缩成一个 query head，而是只把 K/V head 压成一个。

### 从张量形状看出共享位置

输入隐藏状态为：

$$
H\in\mathbb R^{B\times T\times d_{\mathrm{model}}}.
$$

把 Q、K、V 分别投影并拆出 head 轴后，标准 GQA 的形状为：

$$
\begin{gathered}
Q\in\mathbb R^{B\times h_q\times T\times d_k},\\
K\in\mathbb R^{B\times h_{kv}\times S\times d_k},\\
V\in\mathbb R^{B\times h_{kv}\times S\times d_v}.
\end{gathered}
$$

在 self-attention 中通常有 $S=T$；在 cross-attention 中，query 序列长度 $T$ 可以和 key/value 序列长度 $S$ 不同。GQA 只减少 K/V 的 head 轴，未减少 query 的 head 轴。

第 $q$ 个 query head 先取第 $g(q)$ 组的 K/V：

$$
\begin{gathered}
S^{(q)}
=
\frac{Q^{(q)}(K^{(g(q))})^\mathsf T}{\sqrt{d_k}},\\
A^{(q)}
=
\operatorname{softmax}_{\mathrm{row}}
\left(S^{(q)}+M^{(q)}\right),\\
C^{(q)}
=
A^{(q)}V^{(g(q))}.
\end{gathered}
$$

最后仍然沿 query head 的特征轴拼接：

$$
Y=
\operatorname{Concat}
\left(C^{(0)},\ldots,C^{(h_q-1)}\right)W_O.
$$

如果 $d_k=d_v=d_{\mathrm{model}}/h_q$，拼接宽度仍为 $d_{\mathrm{model}}$。K/V head 变少不会改变每个 query head 的输出宽度。

### 共享 K/V 不等于共享 attention matrix

同一组 query head 使用同一个 $K^{(g)}$ 和 $V^{(g)}$，但它们的 $Q^{(q)}$ 不同。因此：

$$
Q^{(q_1)}\ne Q^{(q_2)}
\quad\Longrightarrow\quad
A^{(q_1)}\ \text{可以不同于}\ A^{(q_2)},
\qquad
g(q_1)=g(q_2).
$$

GQA 的共享单位是 K/V 表示，不是最终的读取权重。每个 query head 仍然可以从同一组位置中选择不同的权重。

## 一个两组 K/V 的数值例子

### 先固定 head 映射

取 $h_q=4$、$h_{kv}=2$，因此 $r=2$。四个 query head 的映射为：

|query head $q$|$0$|$1$|$2$|$3$|
|---:|---:|---:|---:|---:|
|K/V 组 $g(q)$|$0$|$0$|$1$|$1$|

第 $0$、$1$ 个 query head 共享第 $0$ 组 K/V，第 $2$、$3$ 个 query head 共享第 $1$ 组 K/V。

### 相同 K/V 产生不同读取

为了只观察 head 共享，直接给出每个 query head 的两个位置 score：

$$
\begin{gathered}
S^{(0)}=S^{(2)}=[2,\ 0],\\
S^{(1)}=S^{(3)}=[0,\ 2].
\end{gathered}
$$

第 $0$ 组和第 $1$ 组的 value 分别为：

$$
V^{(0)}
=
\begin{bmatrix}
10\\
20
\end{bmatrix},
\qquad
V^{(1)}
=
\begin{bmatrix}
100\\
200
\end{bmatrix}.
$$

逐行 softmax 后，四个 query head 的读取结果为：

|query head|K/V 组|attention weight|输出 $C^{(q)}$|
|---:|---:|---|---:|
|$0$|$0$|$(0.880797078,\ 0.119202922)$|$11.192029220$|
|$1$|$0$|$(0.119202922,\ 0.880797078)$|$18.807970780$|
|$2$|$1$|$(0.880797078,\ 0.119202922)$|$111.920292202$|
|$3$|$1$|$(0.119202922,\ 0.880797078)$|$188.079707798$|

第 $0$、$1$ 个 head 读取同一组 value，但因为 score 排序相反，输出不同。GQA 因此保留了多个 query 视角，同时减少了 K/V 表示的数量。

### 共享 K/V 限制的是内容坐标系

上面的两个 query head 仍然有两张不同的 attention weight。它们共享的是：

- 可被读取的 value 位置集合；
- value 的通道宽度 $d_v$；
- 产生 key 相似度的 K/V 投影参数。

它们没有共享：

- query 投影 $W_Q^{(q)}$；
- 每个 query head 的 score；
- 每个 query head 的 softmax 结果；
- 输出投影中对应的输入通道。

如果两组 query 产生相同的 score，输出才可能相同。共享 K/V 本身不能推出这一结论。

## KV cache 为什么是主要收益

### 自回归解码只需要保存 K/V

在 decoder 的第 $t$ 步，当前 token 产生一个新的 query，去读取此前已经生成的 $L$ 个位置的 key/value。过去位置的 K/V 可以缓存，避免每一步重新投影整个前缀：

$$
\begin{gathered}
K_{\mathrm{cache}}
\in
\mathbb R^{B\times h_{kv}\times L\times d_k},\\
V_{\mathrm{cache}}
\in
\mathbb R^{B\times h_{kv}\times L\times d_v}.
\end{gathered}
$$

当前 query 不需要像历史 K/V 一样跨步保存。单层、单设备、未量化时，KV cache 的元素数为：

$$
N_{\mathrm{KV}}
=
B L h_{kv}(d_k+d_v).
$$

若每个元素占 $b$ 字节，缓存字节数为：

$$
\operatorname{bytes}_{\mathrm{KV}}
=
B L h_{kv}(d_k+d_v)b.
$$

对相同的 $B,L,d_k,d_v$，MHA 到 GQA 的 cache 容量按 $h_{kv}$ 线性下降。GQA-8 表示有 8 个 K/V head 的配置，不表示有 8 个 query head。

### 一个单层 FP16 cache 账本

取 $B=1$、$L=2048$、$h_q=32$、$d_k=d_v=128$，FP16 每个元素占 2 bytes：

|模式|$h_{kv}$|K/V 元素数|FP16 字节数|容量|
|---|---:|---:|---:|---:|
|MHA|$32$|$16,777,216$|$33,554,432$|$32$ MiB|
|GQA-8|$8$|$4,194,304$|$8,388,608$|$8$ MiB|
|MQA|$1$|$524,288$|$1,048,576$|$1$ MiB|

这是单层的容量。一个有 $n_{\mathrm{layer}}$ 层的模型，在各层都保存同样长度的 cache 时，应再乘以 $n_{\mathrm{layer}}$。批次、序列长度、数据类型或 KV 量化改变时，也要重新计算。

### cache 变小不等于所有解码计算都变小

当前 query 仍有 $h_q$ 个 head。因此第 $t$ 步的逻辑 score 数量仍近似为：

$$
B h_q\cdot 1\cdot L.
$$

GQA 减少了需要保存和投影的 K/V head 数量，不会把这项改成 $B h_{kv}L$。每个 query head 仍需要读取同一组或对应组的 K/V，并形成自己的 attention weight。

实际延迟还取决于 kernel 是否在 K/V 读取时复用组内数据、cache 的布局、批次大小、GPU 内存层级和量化方式。容量的线性下降是形状事实，延迟下降需要运行时测量。

## 参数量如何随 K/V head 数量变化

### 写出四组投影矩阵

把所有 head 合并表示时，投影矩阵的形状为：

$$
\begin{gathered}
W_Q\in\mathbb R^{d_{\mathrm{model}}\times h_qd_k},\\
W_K\in\mathbb R^{d_{\mathrm{model}}\times h_{kv}d_k},\\
W_V\in\mathbb R^{d_{\mathrm{model}}\times h_{kv}d_v},\\
W_O\in\mathbb R^{h_qd_v\times d_{\mathrm{model}}}.
\end{gathered}
$$

忽略 bias，参数量为：

$$
P
=
d_{\mathrm{model}}
\left(
h_qd_k+h_{kv}d_k+h_{kv}d_v+h_qd_v
\right).
$$

当 $d_k=d_v=d_{\mathrm{model}}/h_q$ 时：

$$
\begin{aligned}
P
&=
d_{\mathrm{model}}^2
+
2\frac{h_{kv}}{h_q}d_{\mathrm{model}}^2
+
d_{\mathrm{model}}^2\\
&=
2d_{\mathrm{model}}^2
\left(1+\frac{h_{kv}}{h_q}\right).
\end{aligned}
$$

MHA 取 $h_{kv}=h_q$，得到 $4d_{\mathrm{model}}^2$。MQA 取 $h_{kv}=1$，K/V 部分只保留一个 query-head 宽度。

### $d_{\mathrm{model}}=512,h_q=8$ 的对照

取 $d_k=d_v=64$，先不计 bias：

|模式|$W_Q$|$W_K$|$W_V$|$W_O$|合计|
|---|---:|---:|---:|---:|---:|
|MHA，$h_{kv}=8$|$262,144$|$262,144$|$262,144$|$262,144$|$1,048,576$|
|GQA-2，$h_{kv}=2$|$262,144$|$65,536$|$65,536$|$262,144$|$655,360$|
|MQA，$h_{kv}=1$|$262,144$|$32,768$|$32,768$|$262,144$|$589,824$|

如果 Q、K、V、O 都有 bias，对应的 bias 数量分别为：

$$
d_{\mathrm{model}}
h_{kv}d_k
h_{kv}d_v
d_{\mathrm{model}}.
$$

在同一配置下，MHA、GQA-2、MQA 的含 bias 总参数量分别为 $1,050,624$、$656,640$、$590,976$。不同实现可能去掉某些 bias，比较 checkpoint 时必须先确认配置。

### 训练期参数收益与推理期 cache 收益不同

训练期仍然要为每个 query head 计算 score 和 value 加权。GQA 主要减少：

- K/V 投影矩阵的参数；
- K/V 投影后的激活；
- 生成时需要保存的 KV cache；
- 生成 kernel 读取 K/V 时可能重复搬运的独立副本。

它没有自动减少：

- query 投影矩阵；
- query head 的数量；
- 逻辑 attention matrix 的 query 行数；
- 每个 query head 的输出通道。

因此不能把参数下降、cache 下降和训练 FLOPs 下降写成同一个比例。三者对应不同的张量。

## 共享 K/V 如何改变反向传播

### query 梯度仍按 head 分开

第 $q$ 个 query head 使用自己的 $Q^{(q)}$，所以来自该 head 的 query 梯度只回到对应的 query 投影：

$$
\frac{\partial\mathcal L}{\partial W_Q^{(q)}}
=
H^\mathsf T
\frac{\partial\mathcal L}{\partial Q^{(q)}}.
$$

如果用合并的 $W_Q$，这些 head 梯度按照 query 列的布局写回不同的列块。GQA 没有把 query 参数合并成共享参数。

### K/V 梯度在组内相加

同一组 K/V 被多个 query head 使用。对第 $g$ 组，所有属于该组的 query head 都会贡献梯度：

$$
\frac{\partial\mathcal L}{\partial K^{(g)}}
=
\sum_{\{q\mid g(q)=g\}}
\left.
\frac{\partial\mathcal L}{\partial K^{(g)}}
\right\rvert_{q},
\qquad
\frac{\partial\mathcal L}{\partial V^{(g)}}
=
\sum_{\{q\mid g(q)=g\}}
\left.
\frac{\partial\mathcal L}{\partial V^{(g)}}
\right\rvert_{q}.
$$

这里的每一项都包含对应 query head 自己的 score、softmax 和上游梯度。共享参数不会让这些路径先平均成一张 attention matrix；反向传播先沿各个 query head 独立计算，再在共享 K/V 节点相加。

### 一个共享 value 的梯度例子

假设同一组的两个 query head 得到：

$$
A^{(0)}=[0.8,\ 0.2],
\qquad
A^{(1)}=[0.3,\ 0.7].
$$

若两个 head 的标量输出都收到上游梯度 $1$，value 的组内梯度为：

$$
\frac{\partial\mathcal L}{\partial V^{(g)}}
=
\begin{bmatrix}
0.8\\
0.2
\end{bmatrix}
+
\begin{bmatrix}
0.3\\
0.7
\end{bmatrix}
=
\begin{bmatrix}
1.1\\
0.9
\end{bmatrix}.
$$

MHA 中这两个 query head 若有两组独立 V，则它们分别保留 $[0.8,0.2]^\mathsf T$ 与 $[0.3,0.7]^\mathsf T$ 的梯度。GQA 让同一 K/V 组接收汇合后的训练信号。

### 从 MHA checkpoint 转成 GQA

一种常见的初始化方法是把同一组内的 MHA K/V 权重做平均：

$$
\begin{gathered}
W_K^{(g)}
\leftarrow
\frac{1}{r}
\sum_{q\in\mathcal G_g}
W_K^{(q)},\\
W_V^{(g)}
\leftarrow
\frac{1}{r}
\sum_{q\in\mathcal G_g}
W_V^{(q)}.
\end{gathered}
$$

其中 $\mathcal G_g$ 是映射到第 $g$ 组的 query head 集合。这个平均是初始化或 checkpoint 转换方法，不是保持前向函数不变的代数恒等式。原因有两个：

- 不同 MHA head 的 K/V 原本可以编码不同内容坐标；
- K 进入点积和 softmax，平均 K 后的 softmax 不等于平均后的 softmax。

转换后通常需要继续训练或校准，恢复程度取决于数据、训练预算、层位置和模型架构。不能只因为参数形状能复制，就断言两个 checkpoint 的输出相同。

## 实现时最容易错的 head 轴

### Q 和 K/V 的 reshape 宽度不同

合并投影的输出最后一维分别是：

$$
\begin{gathered}
\operatorname{width}(Q)=h_qd_k,\\
\operatorname{width}(K)=h_{kv}d_k,\\
\operatorname{width}(V)=h_{kv}d_v.
\end{gathered}
$$

典型的布局变换是：

$$
\begin{gathered}
[B,T,h_qd_k]\to[B,T,h_q,d_k]\to[B,h_q,T,d_k],\\
[B,S,h_{kv}d_k]\to[B,S,h_{kv},d_k]\to[B,h_{kv},S,d_k].
\end{gathered}
$$

把 K/V 按 $h_qd_k$ 去 reshape，会在形状检查阶段暴露错误；如果错误 reshape 后元素数恰好能广播，结果会继续运行但读取的是错误的 head。

### group mapping 必须与权重顺序一致

当 query head 按连续组排列时，$h_q=4,h_{kv}=2$ 的正确映射是：

$$
[0,0,1,1].
$$

某些张量 API 的普通 repeat 会得到：

$$
[0,1,0,1].
$$

这两个向量都满足 head 数量，但第二个交错了 K/V 组。使用 repeat_interleave、gather 或显式索引时，要先确认库函数复制的是每个组 $r$ 次，还是复制完整的组序列。

### 逻辑共享不要求物理复制

把 K/V 沿 head 轴真的复制成 $h_q$ 份，可以得到易读的参考实现，但会重新制造 GQA 想节省的激活和带宽。高效 kernel 可以通过广播、group index 或 tile 内复用，让多个 query head 读取同一份 K/V。

物理上不复制并不表示输出只有一份。每个 query head 仍然要用自己的 Q 计算 score 和权重，再产生自己的 $C^{(q)}$。

### Concat 顺序必须和输出投影匹配

输出投影 $W_O$ 的输入列顺序依赖 query head 的拼接顺序：

$$
Z=
C^{(0)}
\mathbin{\Vert}
C^{(1)}
\mathbin{\Vert}
\cdots
\mathbin{\Vert}
C^{(h_q-1)}.
$$

如果 reshape、transpose 或 gather 改变了 query head 的顺序，就必须同时改变 $W_O$ 的列布局。只检查最终张量形状无法发现这个排列错误。

### cache 的 head 轴和序列轴要分别检查

常见 cache 约定可能是 $[B,h_{kv},L,d_k]$、$[B,L,h_{kv},d_k]$ 或经过分页的 block layout。无论物理布局怎样变化，都要验证三件事：

- 第 $l$ 个位置的 K/V 没有移动到别的位置；
- 第 $q$ 个 query head 读取的是 $g(q)$ 组；
- causal mask 使用的是当前 query 位置和历史 key 位置，而不是 K/V head 编号。

分页 KV cache 改变的是存储寻址，不改变 $g(q)$ 的数学定义。

### mask 不随 K/V head 数量自动减少

典型 causal mask 可以广播为：

$$
M\in\mathbb R^{1\times1\times T\times S},
$$

也可以显式展开到 query head：

$$
M\in\mathbb R^{B\times h_q\times T\times S}.
$$

K/V 只有 $h_{kv}$ 个 head，不表示 mask 应该只有 $h_{kv}$ 个 head。mask 的 head 轴若存在，应根据 query head 或实际规则广播；head-specific mask 还需要检查每个 query head 的有效 key 集合。

## GQA 的边界和取舍

### 共享比例是可选的结构超参数

$h_{kv}$ 越小，K/V 参数和 cache 越少。$h_{kv}$ 越大，K/V 保留的内容坐标越多。均匀分组的 GQA 只使用一个固定映射，不保证每个 query head 的最佳共享对象都相同。

如果任务依赖许多不同的长程关系，减少 K/V head 可能降低表示容量。如果模型已经有冗余 K/V，GQA 可能以较小质量变化换取明显的 cache 节省。质量差异不能从 head 数量单独推出，应使用相同 checkpoint、数据、解码设置和评测指标实测。

### MQA 的优点和限制更集中

MQA 把所有 query head 的 K/V 都合并为一个组，cache 最小，权重也最少。它仍保留 $h_q$ 个 query 视角，但所有视角访问同一个 K/V 内容坐标系。

这种共享会提高缓存复用机会，也会减少 K/V 的多样性。模型若需要多个独立的 value 子空间，MQA 的质量代价可能高于 GQA。GQA-2、GQA-4、GQA-8 等中间配置提供了可测量的折中。

### GQA 不会自动提供稀疏 attention

GQA 仍然可以让每个 query head 读取全部 $S$ 个合法位置。它减少的是 K/V head 轴，不是序列位置轴。若 causal attention 的 score 矩阵仍为 $B\times h_q\times T\times S$，其位置复杂度仍是 $O(h_qTS)$。

[稀疏注意力](../attention/sparse-attention/)、局部窗口和线性 attention 处理的是位置连接或矩阵计算顺序；GQA 处理的是 K/V head 的共享。几种方法可以组合，但不能互相替代。

### cross-attention 的 cache 语义不同

decoder cross-attention 的 K/V 通常由固定 encoder 输出产生。它们不随 decoder 每一步增长，因此 GQA 的收益主要表现为投影参数、静态 K/V 激活和读取布局，而不是减少自回归前缀的增长量。

如果 encoder 序列很长，静态 cross-attention K/V 仍可能占用显著显存。仍要按 $B S h_{kv}(d_k+d_v)$ 和数据类型计算，而不是因为 cache 不增长就忽略其大小。

## 失效模式与审计方法

### 把 MQA 写成一个 query head

MQA 的定义是 $h_{kv}=1$，不是 $h_q=1$。先从权重形状或运行时张量读取 $h_q$，再检查 K/V 的 head 数。

### 用 $h_q$ 计算 K/V 参数

GQA 的 K/V 投影宽度是 $h_{kv}d_k$ 与 $h_{kv}d_v$。如果参数账本中的 K/V 仍使用 $h_q$，cache 和参数节省会被高估或完全消失。

### 复用了 K/V，却错误复用了 attention weight

同组 query head 共享 K/V，不共享 Q。检查每个 query head 的 score、softmax 行和以及输出是否独立计算。

### 组映射交错

把连续分组 [0,0,1,1] 写成交错分组 [0,1,0,1] 会让 query head 读取错误的 K/V。用一个人工索引表和一个单 batch 数值例子检查 gather 结果。

### 用平均 K/V 宣称函数等价

MHA checkpoint 的 K/V 平均会改变 score 和 softmax。将其标记为转换初始化，比较转换前后 logits、loss 和长上下文指标，不把权重平均当作等价重参数化。

### 只测 cache，不测输出

cache 字节数正确只说明存储形状正确。还要核对一个固定输入下的 Q/K/V、attention weight、输出和梯度，检查共享 K/V 的组内梯度是否求和。

### 把训练期 score 数量按 $h_{kv}$ 计算

训练期的 query head 仍为 $h_q$。如果资源账本把 $B h_qTS$ 写成 $B h_{kv}TS$，说明把 K/V 共享误当成 query 连接稀疏化。

### 一份最小 GQA/MQA 审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|query head 数|$h_q$ 与 Q 投影、输出拼接一致|Q 权重列布局|
|K/V head 数|$h_{kv}\le h_q$ 且按约定整除|模型配置、checkpoint 元数据|
|组大小|$r=h_q/h_{kv}$|分组假设|
|组映射|$g(q)=\lfloor q/r\rfloor$ 或明确的自定义映射|repeat 与 gather 顺序|
|Q 形状|$B\times h_q\times T\times d_k$|reshape、transpose|
|K/V 形状|$B\times h_{kv}\times S\times d_k/d_v$|错误使用 $h_q$|
|缩放|每个 query head 使用 $1/\sqrt{d_k}$|误用 $d_{\mathrm{model}}$|
|attention 权重|每个 query head 独立沿 key 轴归一化|错误复用 A|
|KV cache|元素数为 $B L h_{kv}(d_k+d_v)$|序列轴、dtype|
|共享梯度|每个 K/V 组累加组内 query head 梯度|detach、错误广播|
|输出拼接|按 query head 顺序沿特征轴拼接|head 排列、$W_O$|
|训练资源|score 连接仍按 $h_q$ 计数|把 GQA 当稀疏 attention|

## 相关词条

- [多头注意力](../attention/multi-head-attention/)
- [自注意力](../attention/self-attention/)
- [缩放点积注意力](../attention/scaled-dot-product/)
- [Attention 矩阵](../attention/attention-matrix/)
- [因果掩码](../attention/causal-masking/)
- [交叉注意力](../attention/cross-attention/)
- [注意力复杂度](../attention/attention-complexity/)
- [稀疏注意力](../attention/sparse-attention/)
- [FlashAttention](../attention/flash-attention/)
