---
title: "Scaled Dot-Product Attention：为什么要除以 √d_k"
tags: ["why-models-learn"]
---

Scaled Dot-Product Attention（缩放点积注意力）先用 query 和 key 的点积计算匹配分数，再除以 $\sqrt{d_k}$，最后把分数送入 mask 和 softmax。这里 $d_k$ 是 key 向量的维数。缩放因子不改变 value 的内容，也不增加可学习参数；它把点积随维数增长的典型方差拉回到 softmax 可以处理的尺度。

单头 self-attention 的读取链可以写成：

$$
\begin{gathered}
S=\frac{QK^\mathsf T}{\sqrt{d_k}},\\
A=\operatorname{softmax}_{\mathrm{row}}(S+M),\\
C=AV.
\end{gathered}
$$

其中 $S$ 是缩放后的 score 矩阵，$M$ 是注意力掩码，$A$ 是沿每个 query 行归一化后的权重，$C$ 是读取结果。$\sqrt{d_k}$ 解决的是点积的数值尺度问题；它不会替代 mask、softmax 稳定实现、位置特征或后续归一化。[自注意力](../attention/self-attention/)给出完整的同序列读取结构，本篇只把「为什么这里要除以 $\sqrt{d_k}$」拆开推导。

![点积的方差随 key 维数增长，除以平方根后进入 softmax，避免随机分数直接把分布推向饱和](/assets/attention/svg/scaled-dot-product.1.svg)

## 先固定计算顺序与形状

### 缩放发生在 score 上

设 query、key、value 的行分别表示位置。矩阵形状为：

|对象|形状|含义|
|---|---|---|
|$Q$|$T\times d_k$|$T$ 个 query,每个 query 有 $d_k$ 个坐标|
|$K$|$T\times d_k$|$T$ 个 key,每个 key 有 $d_k$ 个坐标|
|$QK^\mathsf T$|$T\times T$|每个 query 与每个 key 的未缩放点积|
|$S$|$T\times T$|逐元素除以 $\sqrt{d_k}$ 后的 score|
|$M$|$T\times T$ 或可广播形状|允许读取的位置与被屏蔽的位置|
|$A$|$T\times T$|逐行 softmax 后的读取权重|
|$V$|$T\times d_v$|被读取的 value|
|$C$|$T\times d_v$|权重乘 value 后的上下文表示|

完整顺序是先计算 $QK^\mathsf T$，再除以 $\sqrt{d_k}$，然后加入 mask，最后沿 key 轴做 softmax：

$$
S_{t,i}
=
\frac{\mathbf q_t\mathbf k_i^\mathsf T}{\sqrt{d_k}},
\qquad
A_{t,:}
=
\operatorname{softmax}
\left(
S_{t,:}+M_{t,:}
\right).
$$

如果把除法放到 softmax 后，得到的不是同一个函数。softmax 需要先看到被缩放的 logits，才能决定这一行分布是集中还是分散。

### $d_k$ 是 key 的最后一维

多头注意力把模型宽度拆成 $h$ 个头。若每个头的 query 和 key 维数相同，则：

$$
d_k=\frac{d_{\mathrm{model}}}{h}.
$$

例如 $d_{\mathrm{model}}=512$、$h=8$ 时，$d_k=64$，缩放因子是 $\sqrt{64}=8$。这里不使用序列长度 $T$，也不直接使用未拆头的 $d_{\mathrm{model}}$。[多头注意力](../attention/multi-head-attention/)会继续处理拆头、拼接和输出投影；本篇只保留一个头来分析尺度。

若某个实现让 query 和 key 使用不同宽度，点积本身就未定义，除非先把它们投影到同一个匹配维度。缩放因子应对应这个实际的匹配维度，而不是 value 的维数 $d_v$。

## 点积为什么会随维数变大

### 把点积拆成许多随机项

设一个 query 和一个 key 的第 $j$ 个坐标分别为 $q_j$ 和 $k_j$。点积是 $d_k$ 个乘积的和：

$$
\mathbf q\mathbf k^\mathsf T
=
\sum_{j=1}^{d_k}q_jk_j.
$$

先采用推导缩放因子时的简化假设：

1. $q_j$ 与 $k_j$ 的均值为 0；
2. 不同坐标之间相互独立；
3. query 与 key 相互独立；
4. 所有 query 坐标的方差为 $\sigma_q^2$，所有 key 坐标的方差为 $\sigma_k^2$。

在这些假设下，一个乘积项的均值为：

$$
\mathbb E[q_jk_j]
=
\mathbb E[q_j]\mathbb E[k_j]
=0.
$$

它的二阶矩为：

$$
\mathbb E[(q_jk_j)^2]
=
\mathbb E[q_j^2]\mathbb E[k_j^2]
=
\sigma_q^2\sigma_k^2.
$$

因为该乘积项的均值为 0，它的方差也等于上式。不同坐标的乘积相互独立时，方差可以相加：

$$
\operatorname{Var}
\left(
\mathbf q\mathbf k^\mathsf T
\right)
=
\sum_{j=1}^{d_k}
\operatorname{Var}(q_jk_j)
=
d_k\sigma_q^2\sigma_k^2.
$$

因此点积的标准差为：

$$
\operatorname{Std}
\left(
\mathbf q\mathbf k^\mathsf T
\right)
=
\sqrt{d_k}\,\sigma_q\sigma_k.
$$

当 $\sigma_q=\sigma_k=1$ 时，点积的典型波动尺度是 $\sqrt{d_k}$。维数从 4 增长到 64，标准差的理论尺度从 2 增长到 8。单个样本的数值不一定等于这个标准差，但大量随机 pair 的分数会呈现同样的尺度趋势。

### 中心极限定理只说明典型形状

当 $d_k$ 足够大时，许多独立乘积相加，点积通常可以近似看成均值为 0、方差为 $d_k\sigma_q^2\sigma_k^2$ 的随机变量：

$$
\mathbf q\mathbf k^\mathsf T
\approx
\mathcal N
\left(
0,\,
d_k\sigma_q^2\sigma_k^2
\right).
$$

这个近似用于解释初始化或训练早期的尺度，不是对训练后所有 score 的分布承诺。学习会让 query 和 key 产生相关性，LayerNorm、RMSNorm、权重初始化和数据分布也会改变边际方差。真正稳定的判断仍然要测量某个模型的 score 分位数，而不是只套用正态假设。

### 非零均值会带来另一项增长

若坐标均值不是 0，记 $\mathbb E[q_j]=\mu_q$、$\mathbb E[k_j]=\mu_k$，则：

$$
\mathbb E
\left[
\mathbf q\mathbf k^\mathsf T
\right]
=
d_k\mu_q\mu_k.
$$

除以 $\sqrt{d_k}$ 后，这个均值项仍按 $\sqrt{d_k}$ 增长。因而 $\sqrt{d_k}$ 的推导依赖于中心化或近似零均值的设定。实际 Transformer 通常配合投影层和归一化控制坐标尺度；缩放因子只负责点积的维数效应，不能单独消除所有均值漂移。

### 四个维数的尺度账本

在单位方差、零均值的简化条件下：

|$d_k$|未缩放点积标准差|除以 $\sqrt{d_k}$ 后|若除以 $d_k$ 后|
|---:|---:|---:|---:|
|4|2|1|0.5|
|16|4|1|0.25|
|64|8|1|0.125|
|256|16|1|0.0625|

表中第二列描述点积的典型波动，第三列显示缩放后的目标尺度，第四列说明除以 $d_k$ 会把随机分数压得过小。这里的 1 是标准差的理论值，不是每一行实际测得的样本标准差。

## softmax 为什么会放大过大的分数

### softmax 只看相对差值

对一行 score $s=(s_1,\ldots,s_n)$，softmax 定义为：

$$
\operatorname{softmax}(s)_i
=
\frac{\exp(s_i)}
{\sum_{j=1}^{n}\exp(s_j)}.
$$

给每个分数加上同一个常数不会改变输出：

$$
\operatorname{softmax}(s+c\mathbf 1)
=
\operatorname{softmax}(s).
$$

因此实现可以先减去该行最大值来防止指数溢出：

$$
\operatorname{softmax}(s)_i
=
\frac{\exp(s_i-m)}
{\sum_{j=1}^{n}\exp(s_j-m)},
\qquad
m=\max_j s_j.
$$

减最大值解决的是有限精度下的指数范围；它不会改变大分数差异造成的分布尖锐化。

### 分数差异扩大后，分布趋向 one-hot

比较两组只有一个单位差异的 logits：

$$
\operatorname{softmax}(0,1)
\approx
(0.268941421,\ 0.731058579),
$$

$$
\operatorname{softmax}(0,4)
\approx
(0.017986210,\ 0.982013790).
$$

第二组没有改变排序，只把差异放大了四倍。较大的分量更接近 1，其余分量更接近 0。对于 attention，这表示一个 query 在训练早期可能过早集中到少数 key，其他位置收到的读取权重和梯度变小。

### softmax 的 Jacobian 会随饱和而变小

令 $p=\operatorname{softmax}(s)$，则：

$$
\frac{\partial p_i}{\partial s_j}
=
p_i(\delta_{ij}-p_j).
$$

当某个 $p_i$ 接近 1 时，$p_i(1-p_i)$ 接近 0；当某个 $p_i$ 接近 0 时，它收到的局部导数也变小。softmax 仍然可导，但有效的 score 梯度会集中在少数竞争边界附近。

注意力输出还包括 value 路径。若某一行输出是：

$$
c_t
=
\sum_{i=1}^{T}\alpha_{t,i}v_i,
$$

则固定权重时 $\partial c_t/\partial v_i=\alpha_{t,i}$；沿 score 路径还要乘 softmax 的 Jacobian。过大的 score 不会让整个网络立刻停止学习，但会改变梯度分配，使未选中的候选更难参与当前更新。

### 用熵测量分布是否集中

一行注意力权重的熵为：

$$
\mathcal H(A_{t,:})
=
-\sum_{i=1}^{T}
A_{t,i}\log A_{t,i}.
$$

熵低表示读取分布集中，熵高表示读取分布分散。熵不是质量指标：如果任务确实只需要一个 key，低熵可能合理；如果训练早期所有 query 都集中到同一个位置，低熵可能说明 score 尺度过大或 mask/初始化存在问题。诊断时应同时查看 score 分位数、有效 key 数和任务损失。

## 为什么是 $\sqrt{d_k}$

### 让随机 score 的方差回到常数尺度

把未缩放点积记为 $z=\mathbf q\mathbf k^\mathsf T$。根据前面的方差计算：

$$
\operatorname{Var}(z)
=
d_k\sigma_q^2\sigma_k^2.
$$

缩放后的 score 是：

$$
s
=
\frac{z}{\sqrt{d_k}}.
$$

方差按缩放因子的平方变化：

$$
\operatorname{Var}(s)
=
\frac{1}{d_k}\operatorname{Var}(z)
=
\sigma_q^2\sigma_k^2.
$$

当 query 和 key 的坐标方差保持在常数范围时，缩放后 score 的方差也保持在常数范围。softmax 看到的相对差异不再仅仅因为匹配维数变大而变大。

### 这不是把每一行做 z-score

除以 $\sqrt{d_k}$ 只使用一个由维数决定的全局标量。它不执行以下操作：

|操作|$\sqrt{d_k}$ 缩放是否完成|
|---|---|
|消除每一行 score 的均值|否|
|把每一行样本标准差精确设为 1|否|
|消除 query/key 的相关性|否|
|把向量归一化为单位长度|否|
|排除 mask 中的非法位置|否|
|使 softmax 输出具有固定熵|否|
|抵消点积中由维数引入的典型方差增长|是，在推导假设下|

因此它应被看作 attention score 的尺度校准，而不是一个完整的归一化层。[softmax 函数](../neurons-and-activations/softmax/)解释指数归一化和 Jacobian；[方差与协方差](../probability/variance-and-covariance/)解释这里使用的二阶矩。

### 信号也会被一起缩放

缩放不会只压低噪声，也会压低 query-key 的真实匹配信号。模型可以通过学习投影的方向、范数和后续权重，重新形成需要的 score 差异。固定因子提供的是一个可训练的起点和数值范围，不是对最终注意力尖锐度的硬约束。

若 query 与某个 key 的每个坐标都强烈同向，点积本身可能接近 $d_k$。除以 $\sqrt{d_k}$ 后仍接近 $\sqrt{d_k}$，所以高维且强匹配的分布仍可能很尖锐。缩放因子降低维数造成的随机波动，不能保证每个训练后 score 都处于温和范围。

## 一个四维的数值例子

### 先制造三个不同匹配程度

取一个 query 和三个 key：

$$
\mathbf q
=(1,1,1,1),
\qquad
K
=
\begin{bmatrix}
1&1&1&1\\
1&-1&1&-1\\
-1&-1&-1&-1
\end{bmatrix}.
$$

三个点积是：

$$
\mathbf qK^\mathsf T
=(4,0,-4).
$$

key 的维数是 $d_k=4$，所以标准缩放因子为 $\sqrt{d_k}=2$：

$$
\frac{\mathbf qK^\mathsf T}{\sqrt{d_k}}
=(2,0,-2).
$$

令三个 key 的 value 为 $(10,20,40)$。分别采用不缩放、除以 $\sqrt{d_k}$ 和除以 $d_k$：

|方案|送入 softmax 的 score|权重|context|
|---|---|---|---:|
|不缩放|$(4,0,-4)$|$(0.9816903928,0.0179802867,0.0003293204)$|10.189682481|
|除以 $\sqrt{d_k}$|$(2,0,-2)$|$(0.8668133322,0.1173104278,0.0158762400)$|11.649391478|
|除以 $d_k$|$(1,0,-1)$|$(0.6652409558,0.2447284711,0.0900305732)$|15.148201906|

三种方案的 context 都是 value 的凸组合，但分布差异明显。不缩放时第一个 key 获得 98.169% 的权重；除以 $\sqrt{d_k}$ 后仍偏向第一个 key，但其他两个 key 保留了更大的权重；除以 $d_k$ 则把差异压得更小。

### “更平”不等于“更正确”

在这个例子中，除以 $d_k$ 得到的分布更平，但这不说明它比标准缩放更好。标准因子来自方差尺度，而不是来自某个固定 toy value 的目标答案。模型训练需要在「保留匹配差异」与「避免随机分数饱和」之间找到可学习范围，过强和过弱的缩放都会改变这个范围。

### 缩放后仍可能产生尖锐分布

如果每个坐标都呈现强同向匹配，未缩放点积的信号也会随 $d_k$ 增长。对一个抽象的 score 三元组 $(8,0,-8)$：

$$
\operatorname{softmax}(8,0,-8)
\approx
(0.999664537,\ 0.000335350,\ 0.000000112).
$$

这个结果并不与缩放原理矛盾。$\sqrt{d_k}$ 只消除独立随机项造成的典型方差增长；它不会把所有真实匹配都改成相同强度。实际审计需要区分「随机背景 score 变大」和「模型学到的匹配信号变强」。

## 缩放因子与 temperature

### 两种写法描述同一个运算

对任意 score 向量 $s$，temperature 写法为：

$$
\operatorname{softmax}\left(\frac{s}{\tau}\right),
\qquad
\tau>0.
$$

标准缩放点积注意力等价于使用固定 temperature $\tau=\sqrt{d_k}$：

$$
\operatorname{softmax}
\left(
\frac{QK^\mathsf T}{\sqrt{d_k}}
\right)
=
\operatorname{softmax}
\left(
\frac{QK^\mathsf T}{\tau}
\right),
\qquad
\tau=\sqrt{d_k}.
$$

temperature 越大，分数差异被压得越小，分布通常越平；temperature 越小，分布通常越尖。固定 $\sqrt{d_k}$ 的选择来自维数方差，而验证集上的 temperature scaling 通常服务于概率校准，两者动机不同。

### 用同一组 score 比较温度

对 $s=(2,0,-2)$：

|$\tau$|softmax 权重|分布描述|
|---:|---|---|
|1|$(0.8668133322,0.1173104278,0.0158762400)$|集中|
|2|$(0.6652409558,0.2447284711,0.0900305732)$|中等|
|4|$(0.5064803911,0.3071958857,0.1863237232)$|分散|

如果在标准缩放后再加入可学习温度，可以写成：

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\mathsf T}{\sqrt{d_k}\tau_{\mathrm{learned}}}
+M
\right).
$$

这相当于在固定方差校准上再学习一个全局尖锐度。它引入了新的优化变量，也引入了温度过小导致 score 饱和的风险。

## mask、softmax 与数值实现

### 先缩放，再加入 mask

允许位置的 mask 值为 0，不允许位置在数学表达中取 $-\infty$：

$$
M_{t,i}
=
\begin{cases}
0,&\text{允许 query }t\text{ 读取 key }i,\\
-\infty,&\text{禁止读取}.
\end{cases}
$$

注意力行写成：

$$
A_{t,:}
=
\operatorname{softmax}
\left(
\frac{(QK^\mathsf T)_{t,:}}{\sqrt{d_k}}
+M_{t,:}
\right).
$$

在精确数学中，先把 $-\infty$ 除以 $\sqrt{d_k}$ 再加回仍是 $-\infty$；在有限浮点实现中，mask 通常使用一个大负 sentinel，而不是实际无穷。把 mask 加在缩放后，语义更直接，也避免 sentinel 的数值尺度随着实现顺序改变。

例如 $d_k=64$ 时，若误把 $-10^4$ 当成未缩放 score 的 mask，再整体除以 8，得到的是 $-1250$。它通常仍然足够小，但不再是原来设定的 sentinel；在低精度、极端正分数或自定义 kernel 中，剩余数值间隔可能影响下溢和梯度。mask 的有效性应以实际 dtype 和 kernel 验证。

### 用减最大值实现稳定 softmax

对每一行先定义：

$$
m_t
=
\max_i
\left(
\frac{(QK^\mathsf T)_{t,i}}{\sqrt{d_k}}
+M_{t,i}
\right).
$$

再计算：

$$
A_{t,i}
=
\frac{
\exp\left(
\frac{(QK^\mathsf T)_{t,i}}{\sqrt{d_k}}
+M_{t,i}
-m_t
\right)
}{
\sum_j
\exp\left(
\frac{(QK^\mathsf T)_{t,j}}{\sqrt{d_k}}
+M_{t,j}
-m_t
\right)
}.
$$

被 mask 的项应在减最大值前就被设为不可选。若某一整行全部被 mask，数学上的分母为 0，常见实现会得到 NaN。padding query、空 memory 或动态稀疏候选都要显式规定这种行如何处理。

### mask 不能只检查前向结果

一行中被禁止的位置前向权重应为 0，允许位置的权重和应为 1：

$$
A_{t,i}=0\quad\text{if }M_{t,i}=-\infty,
\qquad
\sum_{i\in\mathrm{valid}(t)}A_{t,i}=1.
$$

还要检查反向路径。有限 sentinel 可能在特定 dtype 下产生极小但非零的权重；如果下游损失对这些位置敏感，极小泄漏也会积累。[因果掩码](../attention/causal-masking/)会具体说明 decoder 的上三角禁读规则，[注意力掩码](../rnn-lstm/bahdanau-attention/)的术语约定来自已有 attention 词条。

## 缩放如何进入梯度

### 对 query 和 key 的直接导数

对一个 key 位置的 score：

$$
s_i
=
\frac{\mathbf q\mathbf k_i^\mathsf T}{\sqrt{d_k}}.
$$

固定另一侧向量时：

$$
\frac{\partial s_i}{\partial\mathbf q}
=
\frac{\mathbf k_i}{\sqrt{d_k}},
\qquad
\frac{\partial s_i}{\partial\mathbf k_i}
=
\frac{\mathbf q}{\sqrt{d_k}}.
$$

矩阵形式为：

$$
\mathrm dS
=
\frac{
(\mathrm dQ)K^\mathsf T
+Q(\mathrm dK)^\mathsf T
}{\sqrt{d_k}}.
$$

因此，固定 softmax 输入附近的局部变化时，缩放把 score 对 $Q/K$ 的直接导数乘以 $1/\sqrt{d_k}$。但完整梯度还要经过 softmax、value 加权、mask 和损失；不能把一次训练中的总梯度简单写成未缩放梯度除以 $\sqrt{d_k}$。

### 一个 score 梯度的数值核对

沿用四维例子中标准缩放后的第一行：

$$
\alpha
=(0.8668133322,\ 0.1173104278,\ 0.0158762400),
\qquad
c
=
15.148201906.
$$

若把 value 当作标量，score 路径满足：

$$
\frac{\partial c}{\partial s_i}
=
\alpha_i(v_i-c).
$$

代入 $v=(10,20,40)$ 得：

$$
\frac{\partial c}{\partial s}
\approx
(-3.424794756,\ 1.187373129,\ 2.237421627).
$$

三项之和约为 $3.1\times10^{-15}$，因为 softmax 对整行 score 同时加同一个常数不敏感。缩放改变了 $\alpha$，所以它同时改变 value 路径的直接权重和 score 路径的耦合梯度。

## 形状、计算量与参数量

### 批次和多头形状

在实现中通常把批次和 head 维放在前面：

|张量|形状|说明|
|---|---|---|
|$Q$|$B\times h\times T\times d_k$|每个 head 的 query|
|$K$|$B\times h\times S\times d_k$|每个 head 的 key|
|$QK^\mathsf T$|$B\times h\times T\times S$|每个目标位置与每个候选位置的 score|
|$M$|$B\times 1\times T\times S$ 或可广播形状|padding/causal 等约束|
|$A$|$B\times h\times T\times S$|沿 $S$ 轴 softmax|
|$V$|$B\times h\times S\times d_v$|候选 value|
|$C$|$B\times h\times T\times d_v$|每个 head 的读取结果|

在 self-attention 中 $S=T$；在 cross-attention 中，query 长度 $T$ 与 key/value 长度 $S$ 可以不同。缩放因子只依赖最后的匹配维 $d_k$，不依赖 $B$、$h$、$T$ 或 $S$。

### 一个资源账本

取 $B=2$、$h=16$、$T=S=512$、$d_k=d_v=64$：

|量|计算|结果|
|---|---|---:|
|score 元素数|$B h T S$|8,388,608|
|QK 点积乘加的维度项|$B h T S d_k$|536,870,912|
|AV 加权的维度项|$B h T S d_v$|536,870,912|
|FP16 score 矩阵字节数|$B h T S\times2$|16,777,216 bytes = 16 MiB|
|缩放因子参数数|固定 $\sqrt{64}$|0|

这里的乘加项是形状账本，不等同于某个具体 kernel 的指令数。FlashAttention 等实现可以避免把完整 $T\times S$ 矩阵写入显存，但不会改变缩放公式和逻辑上的位置对数量。[注意力复杂度](../attention/attention-complexity/)会进一步比较 $T^2$ 资源与近似方法。

### 缩放不增加 QKV 参数

若 $d_{\mathrm{model}}=512$、单头 $d_k=64$，三组无 bias 的 $Q/K/V$ 投影共有：

$$
3d_{\mathrm{model}}d_k
=
3\times512\times64
=
98{,}304
$$

个参数。$\sqrt{d_k}$ 是固定常数，因此不增加参数。若把 temperature 设为可学习标量，则只增加 1 个标量；若给每个 head 一个 temperature，则增加 $h$ 个标量。

## 与其他缩放方式比较

### 不缩放

直接使用 $QK^\mathsf T$ 会让零均值随机 score 的标准差按 $\sqrt{d_k}$ 增长。$d_k$ 越大，softmax 越容易受偶然的极端 score 支配。旧模型或特定任务可能通过较小的投影范数缓解这个问题，但那把数值责任转交给训练过程。

### 除以 $d_k$

除以 $d_k$ 会让方差变为：

$$
\operatorname{Var}
\left(
\frac{\mathbf q\mathbf k^\mathsf T}{d_k}
\right)
=
\frac{\sigma_q^2\sigma_k^2}{d_k}.
$$

标准差随 $1/\sqrt{d_k}$ 下降。大维数下随机 score 过于接近 0，softmax 更接近均匀分布，匹配差异需要依靠更大的学习信号才能显现。

### 先归一化为 cosine similarity

另一种选择是：

$$
\cos(\mathbf q,\mathbf k)
=
\frac{\mathbf q\mathbf k^\mathsf T}
{\lVert\mathbf q\rVert_2\lVert\mathbf k\rVert_2}.
$$

它去除了向量长度，只保留方向。缩放点积保留 query 和 key 的范数信息，只用 $\sqrt{d_k}$ 修正维数带来的典型尺度，因此二者表达的匹配函数不同。[余弦相似度](../text-representation/cosine-similarity/)展开这个几何差异。

### 学习 temperature 或 gain

可学习温度可以写成：

$$
A
=
\operatorname{softmax}
\left(
g\,QK^\mathsf T+M
\right),
\qquad
g>0.
$$

若初始化 $g=1/\sqrt{d_k}$，它从标准缩放的尺度开始；训练可以再改变全局尖锐度。这个方案增加了优化自由度，也需要监控 $g$ 是否过大、过小或在不同 head 之间失衡。

|方案|保留范数信息|随机 score 标准差的维数趋势|主要风险|
|---|---|---|---|
|$QK^\mathsf T$|是|$\sqrt{d_k}$ 增长|softmax 饱和|
|$QK^\mathsf T/\sqrt{d_k}$|是|在假设下保持常数|训练后仍可能有极端 score|
|$QK^\mathsf T/d_k$|是|$1/\sqrt{d_k}$ 衰减|分布过平|
|cosine similarity|否|由范数归一化决定|丢失长度信号|
|可学习 $gQK^\mathsf T$|通常是|由 $g$ 与训练共同决定|温度漂移或 head 间失衡|

## 失效模式与审计方法

### 把 $d_{\mathrm{model}}$ 当成 $d_k$

拆成多头后，$QK^\mathsf T$ 的最后一维是每个 head 的 $d_k$。若误用 $d_{\mathrm{model}}$，缩放因子过大，score 会被压得过低，注意力分布可能过平；若误用更小的维数，score 会被压得不够，softmax 更容易饱和。先打印实际 query/key 的最后一维，再确认缩放常数。

### 沿错误轴缩放或 softmax

缩放是对 score 的每个元素使用同一个 $1/\sqrt{d_k}$。softmax 必须沿 key 候选轴 $S$ 归一化，而不是沿 head、query 或最后的 value 维度。检查每行权重和与张量形状，比只检查一张可视化热图更可靠。

### 把 mask 放在错误位置

mask 需要与实际 dtype、sentinel 和 kernel 一起测试。只看未 mask 的随机输入，无法发现 padding、因果上三角、空候选行和广播维度的问题。

### 重复缩放或漏缩放

如果上游已经把 Q 或 K 乘过一个温度，attention 内部再除以 $\sqrt{d_k}$ 可能形成双重缩放；如果调用方期待 raw dot product，而模块内部又默认缩放，接口也会产生同样问题。把 score pipeline 写在日志或单元测试中，确认缩放恰好发生一次。

### 只看平均熵

平均熵可能掩盖少数 query 行的 score 爆炸、全 mask 行或某个 head 的极端温度。至少按 batch、head 和有效长度分组查看 score 的均值、标准差、最大值、最小值、权重熵和非法位置的最大权重。

### 只看前向，不看梯度

前向权重看似正常时，低精度 softmax 仍可能在反向传播中产生非有限值。检查 Q、K、score、A、输出和对应梯度的 finite 比例；对缩放常数、mask 和 softmax 采用同一 dtype 复核。

### 一份最小审计表

|检查项|应满足的条件|发现异常时先查|
|---|---|---|
|匹配维度|$Q$ 与 $K$ 的最后一维相同|head split、投影宽度|
|缩放常数|每个 score 除以 $\sqrt{d_k}$ 一次|d_model/d_k 混用、重复缩放|
|softmax 轴|每个 query 的有效 key 权重和为 1|transpose、axis 参数|
|mask 前向|禁止位置权重为 0|sentinel、广播、全 mask 行|
|score 尺度|训练早期标准差不随 $d_k$ 线性增长|初始化、norm、dtype|
|梯度有限性|Q/K/score/A 梯度无 NaN 或 Inf|指数溢出、全 mask、低精度|
|资源账本|score 逻辑形状为 $B\times h\times T\times S$|因果轴、cross-attention 长度|

## 相关词条

- [Attention 作为检索](../attention/attention-as-retrieval/)
- [自注意力](../attention/self-attention/)
- [softmax 函数](../neurons-and-activations/softmax/)
- [方差与协方差](../probability/variance-and-covariance/)
- [多头注意力](../attention/multi-head-attention/)
- [因果掩码](../attention/causal-masking/)
- [注意力复杂度](../attention/attention-complexity/)
- [余弦相似度](../text-representation/cosine-similarity/)
