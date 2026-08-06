---
title: "Multi-Head Attention：为什么要并行拆成多个 head"
tags: ["why-models-learn"]
---

Multi-Head Attention（多头注意力）把同一个输入投影到 $h$ 个独立的 query、key、value 子空间，在每个 head 内分别计算 attention，再把各个 head 的输出拼接起来，最后用一个输出投影混合这些通道。每个 head 有自己的读取矩阵和参数；各 head 共享输入序列，但不共享完整的 Q/K/V 投影。

固定输入 $H\in\mathbb R^{T\times d_{\mathrm{model}}}$，第 $r$ 个 head 的计算为：

$$
\begin{gathered}
Q^{(r)}=HW_Q^{(r)},\qquad
K^{(r)}=HW_K^{(r)},\qquad
V^{(r)}=HW_V^{(r)},\\
\operatorname{head}_r
=
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{Q^{(r)}K^{(r)\mathsf T}}{\sqrt{d_k}}
+M^{(r)}
\right)V^{(r)},\\
\operatorname{MHA}(H)
=
\operatorname{Concat}
\left(
\operatorname{head}_1,\ldots,\operatorname{head}_h
\right)W_O.
\end{gathered}
$$

拆头提供的是多组可学习的读取坐标系，不是把同一个 attention matrix 复制 $h$ 次。[Attention 矩阵](../attention/attention-matrix/)解释单个读取矩阵的行列语义，[缩放点积注意力](../attention/scaled-dot-product/)解释每个 head 为什么使用自己的 $d_k$ 缩放，本篇集中处理拆分、拼接、参数量和实现边界。

![同一输入进入多个独立 attention head，各 head 产生不同读取矩阵，输出拼接后经过输出投影混合](/assets/attention/svg/multi-head-attention.1.svg)

## 从一个输入拆出多个 head

### 每个 head 使用自己的投影

第 $r$ 个 head 的投影矩阵形状为：

$$
W_Q^{(r)},W_K^{(r)}
\in
\mathbb R^{d_{\mathrm{model}}\times d_k},
\qquad
W_V^{(r)}
\in
\mathbb R^{d_{\mathrm{model}}\times d_v}.
$$

如果 $H$ 的每一行是一个 token 位置，那么：

$$
Q^{(r)},K^{(r)}
\in
\mathbb R^{T\times d_k},
\qquad
V^{(r)}
\in
\mathbb R^{T\times d_v}.
$$

每个 head 都从完整的 $H$ 读取输入，只是使用不同的列空间：

$$
Q^{(r)}=H W_Q^{(r)}.
$$

因此「第 $r$ 个 head」不是输入特征的第 $r$ 段被动切片，而是一个独立的可学习线性投影。实现也可以先用一个大矩阵产生所有 head，再 reshape 成 head 轴；只要权重列和张量轴保持一致，数学结果相同。

### 标准设置让拼接宽度回到模型宽度

常见配置令：

$$
d_k=d_v=\frac{d_{\mathrm{model}}}{h}.
$$

第 $r$ 个 head 的输出为：

$$
\operatorname{head}_r
\in
\mathbb R^{T\times d_v}.
$$

沿最后一个特征轴拼接：

$$
Z
=
\operatorname{Concat}
\left(
\operatorname{head}_1,\ldots,\operatorname{head}_h
\right)
\in
\mathbb R^{T\times h d_v}
=
\mathbb R^{T\times d_{\mathrm{model}}}.
$$

再用：

$$
W_O
\in
\mathbb R^{h d_v\times d_{\mathrm{model}}},
\qquad
O=ZW_O.
$$

如果 $h d_v\ne d_{\mathrm{model}}$，输出投影仍然可以把拼接宽度映射回模型宽度；「拼接宽度等于模型宽度」是常见配置，不是矩阵乘法的必要条件。

### 形状账本

|对象|单个 head 的形状|全部 head 的形状|作用|
|---|---|---|---|
|输入 $H$|$T\times d_{\mathrm{model}}$|$T\times d_{\mathrm{model}}$|同一条输入序列|
|$Q^{(r)},K^{(r)}$|$T\times d_k$|$h\times T\times d_k$|匹配空间|
|$V^{(r)}$|$T\times d_v$|$h\times T\times d_v$|内容空间|
|$A^{(r)}$|$T\times T$|$h\times T\times T$|每个 head 的读取矩阵|
|$\operatorname{head}_r$|$T\times d_v$|$h\times T\times d_v$|各 head 的读取结果|
|$Z$|—|$T\times h d_v$|拼接后的通道|
|$O$|—|$T\times d_{\mathrm{model}}$|输出投影后的结果|

cross-attention 时 key/value 的长度可以是 $S$，对应形状把每个 head 的 $T\times T$ 改为 $T\times S$；head 维和匹配维的规则不变。

## 为什么要多个独立读取空间

### 一个 head 只有一张读取分布

对于给定 query 位置 $t$，一个 head 只有一行：

$$
A^{(r)}_{t,:}
=
\operatorname{softmax}
\left(
\frac{\mathbf q_t^{(r)}K^{(r)\mathsf T}}{\sqrt{d_k}}
+M_{t,:}^{(r)}
\right).
$$

这一行可以同时给多个 key 权重，但它只有一套 score 排序。多个 head 让同一个输入位置同时拥有多套排序：

$$
A^{(1)}_{t,:},
A^{(2)}_{t,:},
\ldots,
A^{(h)}_{t,:}.
$$

这些排序来自不同的 $W_Q^{(r)}$ 和 $W_K^{(r)}$。一个 head 可以偏向局部邻近位置，另一个 head 可以偏向重复实体或句法相关位置；这些是模型可能学到的分工，不是架构自动保证的标签。

### 投影列空间给每个 head 一种匹配坐标系

把 $W_Q^{(r)}$ 的每一列看作从模型表示提取一个 query 坐标，把 $W_K^{(r)}$ 的每一列看作提取一个 key 坐标。点积在这些坐标中比较：

$$
\mathbf q_t^{(r)}\mathbf k_i^{(r)\mathsf T}
=
\left(H_{t,:}W_Q^{(r)}\right)
\left(H_{i,:}W_K^{(r)}\right)^\mathsf T.
$$

不同 head 可以选择不同的投影列空间。它们仍然可以产生相同的读取位置，也可以在训练后出现冗余；多头提供的是可用的参数化自由度，不是每个 head 必须承担一个可命名功能的约束。

### head 输出不能只做平均

若把各 head 输出平均：

$$
\bar C
=
\frac{1}{h}
\sum_{r=1}^{h}\operatorname{head}_r,
$$

那么不同 head 的内容会在相加时混合，且输出宽度只剩 $d_v$。标准 MHA 先拼接：

$$
Z
=
\begin{bmatrix}
\operatorname{head}_1&
\operatorname{head}_2&
\cdots&
\operatorname{head}_h
\end{bmatrix},
$$

保留各 head 的通道，再由 $W_O$ 学习如何组合。$W_O$ 可以实现加权组合，也可以跨 head 混合不同通道；它比固定平均多出一层可学习线性映射。

## 两个 head 的数值例子

### 让两个 head 读取相反的方向

为了只观察拆头和拼接，直接给出两个 head 已经缩放后的 score 矩阵：

$$
S^{(1)}
=
\begin{bmatrix}
2&0\\
0&2
\end{bmatrix},
\qquad
S^{(2)}
=
\begin{bmatrix}
0&2\\
2&0
\end{bmatrix}.
$$

两个 head 的 score 排序相反。逐行 softmax 得到：

$$
A^{(1)}
\approx
\begin{bmatrix}
0.880797078&0.119202922\\
0.119202922&0.880797078
\end{bmatrix},
$$

$$
A^{(2)}
\approx
\begin{bmatrix}
0.119202922&0.880797078\\
0.880797078&0.119202922
\end{bmatrix}.
$$

令两个 head 的 value 都只有一个内容坐标，但数值范围不同：

$$
V^{(1)}
=
\begin{bmatrix}
10\\
20
\end{bmatrix},
\qquad
V^{(2)}
=
\begin{bmatrix}
100\\
200
\end{bmatrix}.
$$

各 head 的输出为：

$$
\operatorname{head}_1
\approx
\begin{bmatrix}
11.192029220\\
18.807970780
\end{bmatrix},
\qquad
\operatorname{head}_2
\approx
\begin{bmatrix}
188.079707798\\
111.920292202
\end{bmatrix}.
$$

拼接结果是：

$$
Z
\approx
\begin{bmatrix}
11.192029220&188.079707798\\
18.807970780&111.920292202
\end{bmatrix}.
$$

如果取 $W_O=I_2$，输出就保留这两个方向。固定平均会得到：

$$
\bar C
\approx
\begin{bmatrix}
99.635868509\\
65.364131491
\end{bmatrix},
$$

两个 head 的读取差异已经被压进一个数值，无法从该输出通道直接区分「第一个 head 的局部内容」和「第二个 head 的反向内容」。这个例子没有说明平均一定不能用；它说明拼接为输出投影保留了更多可学习的组合方式。

### 这个例子没有指定完整的输入投影

上面的 $S^{(1)}$、$S^{(2)}$ 是两个 head 的 post-scale score，$V^{(1)}$、$V^{(2)}$ 是投影后的 value。它们足以核对 softmax、读取和拼接，但没有给出同一个 $H$ 与四组 $W_Q/W_K/W_V$。若要复现完整层，还必须指定输入表示、每个投影矩阵、mask 和 $W_O$；不能把一个手写的 score 矩阵误称为完整模型前向输出。

## 每个 head 的缩放和 mask

### 缩放因子对应每个 head 的 $d_k$

标准多头配置中：

$$
d_k=\frac{d_{\mathrm{model}}}{h}.
$$

因此第 $r$ 个 head 使用：

$$
A^{(r)}
=
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{Q^{(r)}K^{(r)\mathsf T}}{\sqrt{d_k}}
+M^{(r)}
\right).
$$

如果 $d_{\mathrm{model}}=512$、$h=8$，每个 head 的 $d_k=64$，缩放因子是 8，而不是 $\sqrt{512}$。用模型宽度代替 head 宽度会把每个 head 的 score 压得过低。

如果不同 head 采用不同 $d_k^{(r)}$，则每个 head 需要使用自己的：

$$
\frac{1}{\sqrt{d_k^{(r)}}}.
$$

### mask 可以共享，也可以按 head 改变

因果 mask 通常对所有 head 相同：

$$
M^{(1)}
=
M^{(2)}
=
\cdots
=
M^{(h)}.
$$

但稀疏或结构化注意力可以让 $M^{(r)}$ 按 head 不同。无论是否共享，mask 都要在各 head 的 score 上沿同一 key 轴应用。某个 head 的全 mask 行仍然没有合法 softmax 分布，不能因为其他 head 有有效 key 就忽略。

### head 之间不共享 Q/K/V 的含义

标准 MHA 的每个 head 都有：

$$
W_Q^{(r)},W_K^{(r)},W_V^{(r)}.
$$

所以不同 head 的 key/value 缓存也各自不同。[GQA 与 MQA](../attention/gqa-and-mqa/)通过共享部分或全部 K/V 投影来降低缓存和带宽；它们保留多个 query head，但已经不是标准的完全独立多头 QKV。

## 拼接和输出投影

### Concat 是沿通道轴，不是沿 token 轴

每个 head 输出形状为 $T\times d_v$。正确拼接是：

$$
\operatorname{Concat}_{\mathrm{feature}}
\left(
\operatorname{head}_1,\ldots,\operatorname{head}_h
\right)
\in
\mathbb R^{T\times h d_v}.
$$

沿 token 轴拼接会得到 $(hT)\times d_v$，它把同一位置的多个 head 当成不同 token，后续 residual 无法与原始 $T\times d_{\mathrm{model}}$ 逐位置相加。这个错误可能在某些 reshape 后仍能运行，所以必须检查每个轴的含义。

### $W_O$ 混合 head 通道

写成单个位置的行向量：

$$
\mathbf z_t
=
\left[
\mathbf c_t^{(1)}
\ \mathbin{\Vert}\
\mathbf c_t^{(2)}
\ \mathbin{\Vert}\
\cdots
\ \mathbin{\Vert}\
\mathbf c_t^{(h)}
\right].
$$

输出投影为：

$$
\mathbf o_t
=
\mathbf z_t W_O.
$$

因此输出通道可以同时依赖多个 head。经过 $W_O$ 后，不能只根据最终通道位置断言它来自某一个 head；若要分析 head 贡献，需要做 head ablation、投影分块分析或其他干预。

### 输出投影不是另一个 attention

$W_O$ 只在每个位置的特征轴上做线性映射：

$$
W_O
\in
\mathbb R^{h d_v\times d_{\mathrm{model}}}.
$$

它不再比较 query 和 key，也不沿序列位置做 softmax。序列位置之间的读取只发生在各个 $A^{(r)}V^{(r)}$；$W_O$ 负责把多个 head 的内容通道整理回模型宽度。

## 反向传播如何拆回各个 head

### 先从输出投影回到拼接张量

令：

$$
O=ZW_O,
\qquad
G_O=\frac{\partial L}{\partial O}.
$$

则：

$$
\frac{\partial L}{\partial Z}
=
G_O W_O^\mathsf T,
\qquad
\frac{\partial L}{\partial W_O}
=
Z^\mathsf T G_O.
$$

$\partial L/\partial Z$ 沿最后一个特征轴切成 $h$ 段，分别得到：

$$
G_{\operatorname{head}_r}
=
\frac{\partial L}{\partial \operatorname{head}_r}.
$$

所以 head 在反向传播中共享上游损失，但通过各自的 value、attention matrix 和 Q/K 投影收到不同梯度。

### 单个 head 的反向路径

对：

$$
\operatorname{head}_r=A^{(r)}V^{(r)},
$$

令 $G_r=\partial L/\partial\operatorname{head}_r$，则：

$$
\frac{\partial L}{\partial A^{(r)}}
=
G_r V^{(r)\mathsf T},
\qquad
\frac{\partial L}{\partial V^{(r)}}
=
A^{(r)\mathsf T}G_r.
$$

$A^{(r)}$ 再通过逐行 softmax 连接到 $Q^{(r)}K^{(r)\mathsf T}/\sqrt{d_k}$，最后回到：

$$
Q^{(r)}=HW_Q^{(r)},
\qquad
K^{(r)}=HW_K^{(r)},
\qquad
V^{(r)}=HW_V^{(r)}.
$$

同一个 $H$ 会收到所有 head 的输入投影梯度之和。head 之间不是完全独立的模型，因为它们的输入和下游输出投影相连。

### head 范数不等于 head 重要性

某个 head 的输出范数较大，不等于它对任务更重要；$W_O$ 可以缩放或抵消该 head 的通道，多个 head 也可能通过不同方向共同作用。比较 head 时至少区分：

1. 去掉该 head 后的输出变化；
2. 去掉该 head 后的损失变化；
3. 该 head 的 attention 熵和读取位置；
4. 它与其他 head 的输出或梯度相似度。

单看热图面积或输出绝对值无法完成这个判断。

## 参数量与计算量

### 标准配置的参数量

忽略 bias，第 $r$ 个 head 的 Q/K/V 参数量为：

$$
d_{\mathrm{model}}d_k
+
d_{\mathrm{model}}d_k
+
d_{\mathrm{model}}d_v.
$$

$h$ 个 head 加上输出投影：

$$
P
=
h d_{\mathrm{model}}(2d_k+d_v)
+
h d_v d_{\mathrm{model}}.
$$

当 $d_k=d_v=d_{\mathrm{model}}/h$ 时：

$$
P
=
3d_{\mathrm{model}}^2+d_{\mathrm{model}}^2
=
4d_{\mathrm{model}}^2.
$$

拆成多个 head 不会在标准宽度配置下自动增加 QKV 加输出投影的参数总量；它改变的是参数如何分配到多个投影子空间。

### $d_{\mathrm{model}}=512,h=8$ 的账本

取 $d_k=d_v=64$：

|部分|参数计算|无 bias 参数量|
|---|---|---:|
|所有 head 的 $W_Q$|$512\times8\times64$|262,144|
|所有 head 的 $W_K$|$512\times8\times64$|262,144|
|所有 head 的 $W_V$|$512\times8\times64$|262,144|
|输出投影 $W_O$|$8\times64\times512$|262,144|
|合计|$4\times512^2$|1,048,576|
|加 QKV 与输出 bias|$3\times512+512$|1,050,624|

最后一行包含 bias，前五行都不包含 bias。若把所有 head 合并成一个大 QKV 投影矩阵，参数计数相同：

$$
W_{QKV}
\in
\mathbb R^{d_{\mathrm{model}}\times3d_{\mathrm{model}}}.
$$

拆头主要是对输出列和张量轴做结构化分组。

### attention 计算量如何分配

取 $B=2$、$h=8$、$T=S=512$、$d_k=d_v=64$：

|量|计算|结果|
|---|---|---:|
|所有 head 的 attention matrix 元素|$B h T S$|4,194,304|
|QK 点积的维度项|$B h T S d_k$|268,435,456|
|AV 加权的维度项|$B h T S d_v$|268,435,456|
|FP16 attention matrix 字节数|$B h T S\times2$|8,388,608 bytes = 8 MiB|
|单个 head 的 score 元素|$B T S$|524,288|

若保持 $d_{\mathrm{model}}=h d_k$，所有 head 的 QK 维度项为：

$$
B h T S d_k
=
B T S d_{\mathrm{model}}.
$$

与单个 head 使用 $d_k=d_{\mathrm{model}}$ 时，理论乘加项可以相同；MHA 把一个宽匹配空间换成 $h$ 个窄匹配空间，并让每个 head 拥有独立的 score 矩阵。逻辑 attention matrix 的存储元素数则从 $BTS$ 变为 $BhTS$，实际 fused kernel 可以选择不物化这些矩阵。

## 实现中的轴与 reshape

### 从合并投影到 head 轴

工程实现常先一次性计算：

$$
Q_{\mathrm{flat}}
\in
\mathbb R^{B\times T\times(hd_k)}.
$$

再 reshape 和 transpose：

$$
\begin{gathered}
Q_{\mathrm{flat}}
\longrightarrow
\widetilde Q
\in
\mathbb R^{B\times T\times h\times d_k},\\
\widetilde Q
\longrightarrow
Q
\in
\mathbb R^{B\times h\times T\times d_k}.
\end{gathered}
$$

K、V 使用相同的 head 轴约定。这个 transpose 把 head 轴移到序列轴前面，使每个 $B,h$ 切片都能进行 $T\times d_k$ 与 $S\times d_k$ 的点积。

### reshape 顺序必须与权重列顺序一致

如果合并投影的最后一维排列为：

$$
[h_1\text{ 的 }d_k\text{ 坐标}]
\ \Vert\
[h_2\text{ 的 }d_k\text{ 坐标}]
\ \Vert\
\cdots,
$$

reshape 时就必须按同样顺序恢复 head 轴。把交错布局误当成连续布局，会让一个 head 混入其他 head 的坐标，形状仍然正确但 score 和 value 都错位。

用小张量审计最容易发现该错误。令 $h=2,d_k=2$，某个 token 的合并 query 为：

$$
[q_{1,1},q_{1,2},q_{2,1},q_{2,2}]
=
[1,2,3,4].
$$

连续布局应恢复：

$$
q^{(1)}=[1,2],
\qquad
q^{(2)}=[3,4].
$$

如果实现恢复成 $[1,3]$ 与 $[2,4]$，它使用的是交错布局。先确定线性层输出协议，再确定 reshape 的 view/transpose 顺序。

### 输出方向的逆变换

各 head 输出通常是：

$$
C
\in
\mathbb R^{B\times h\times T\times d_v}.
$$

先 transpose 回：

$$
\widetilde C
\in
\mathbb R^{B\times T\times h\times d_v},
$$

再 reshape 成：

$$
Z
\in
\mathbb R^{B\times T\times(hd_v)}.
$$

只有最后这个通道轴才送入 $W_O$。把 $h$ 轴和 $T$ 轴交换后直接 flatten，会把不同位置的输出拼到同一个 token，残差连接会暴露出长度或语义错误。

## MHA、GQA 和 MQA 的边界

标准 MHA 为每个 query head 都提供独立的 K/V：

$$
h_q=h_k=h_v=h.
$$

GQA 把多个 query head 分成若干组，每组共享一个 K/V head：

$$
h_q>h_k=h_v.
$$

MQA 把所有 query head 共享一组 K/V：

$$
h_q>h_k=h_v=1.
$$

三者都可以继续使用 head-wise attention 和 $C=AV$，区别在于 K/V 投影、attention score 的广播方式以及推理时缓存的数量。[GQA 与 MQA](../attention/gqa-and-mqa/)会专门核对 group 映射、KV cache 和参数变化；本篇只把标准 MHA 作为基线。

## 失效模式与审计方法

### 把多个 head 当成复制品

如果所有 head 的 $W_Q/W_K/W_V$ 共享同一组参数，那么它们在相同 mask 和输入下会产生相同读取，拆头只是在复制计算。标准 MHA 需要检查每个 head 的投影列是否独立，或确认共享参数是有意的 GQA/MQA 设计。

### 使用错误的缩放维度

每个 head 的点积内维是 $d_k$。使用 $d_{\mathrm{model}}$ 会把 score 过度压平，使用比 $d_k$ 更小的数会增加饱和风险。打印实际 $Q^{(r)}$、$K^{(r)}$ 的最后一维，再核对每个 head 的 scale。

### 沿 token 轴拼接

正确输出需要保持 $T$ 个位置和 $h d_v$ 个通道。检查：

$$
\operatorname{shape}(Z)
=
(B,T,h d_v).
$$

如果出现 $(B,hT,d_v)$，说明把 head 轴拼到了序列轴。该错误应在 residual 相加前失败，而不是继续广播。

### 忘记输出投影或投影方向反了

拼接结果是 $B\times T\times h d_v$，$W_O$ 应作用在最后一个特征轴。把 $W_O$ 写成 $d_{\mathrm{model}}\times h d_v$ 并右乘，或把输出投影误做成沿 token 轴的矩阵，都会改变形状和语义。

### mask 只广播到部分 head

共享 causal/padding mask 时，检查它是否广播到所有 $h$ 个 head。head-specific mask 时，检查每个 $M^{(r)}$ 的有效集合。一个 head 的有效位置不能被误用为所有 head 的有效位置。

### 只看 head 热图给出固定命名

head 的读取模式会随层、checkpoint、输入和任务变化。命名或剪枝需要结合输出干预、损失变化和不同样本的稳定性；单张热图不能证明一个 head 永远只负责某个关系。

### 一份最小 MHA 审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|head 数|$h$ 与配置及权重切分一致|QKV 列布局|
|匹配维度|$d_k$ 是每个 head 的最后一维|误用 $d_{\mathrm{model}}$|
|value 维度|$d_v$ 与 head 输出宽度一致|Concat 宽度|
|attention 形状|每个 head 为 $T\times S$|Q/K transpose|
|softmax 轴|沿 key 轴逐行归一化|axis、mask|
|拼接轴|沿特征轴得到 $T\times h d_v$|误拼到 token 轴|
|输出投影|$W_O$ 形状为 $h d_v\times d_{\mathrm{model}}$|转置、残差宽度|
|参数量|标准配置无 bias 为 $4d_{\mathrm{model}}^2$|漏算或重复算投影|
|head 独立性|标准 MHA 每个 head 有自己的 Q/K/V|误把 MHA 当 GQA/MQA|
|梯度|合并投影后能按 head 切回|reshape、transpose、contiguous|

## 相关词条

- [自注意力](../attention/self-attention/)
- [缩放点积注意力](../attention/scaled-dot-product/)
- [Attention 矩阵](../attention/attention-matrix/)
- [GQA 与 MQA](../attention/gqa-and-mqa/)
- [因果掩码](../attention/causal-masking/)
- [交叉注意力](../attention/cross-attention/)
- [注意力复杂度](../attention/attention-complexity/)
- [残差连接](../cnn/residual-connections/)
