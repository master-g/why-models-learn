---
title: "稀疏注意力：减少哪些 query-key 连接才真的省资源"
tags: ["why-models-learn"]
---

Sparse Attention（稀疏注意力）不让每个 query 读取全部 key，而是为每个 query 指定一个有效 key 集合。标准 dense attention 有 $B h_q T S$ 个 query-key 位置对；稀疏 attention 把这个数量改为有效连接数 $\lvert E\rvert$，并要求执行 kernel 真的跳过集合外位置，才能把逻辑稀疏转换为算力和显存节省。本篇从二值连接 mask 出发，推导局部窗口、块稀疏、全局 token、膨胀连接和动态 top-k 的计数，再区分稀疏 mask、FlashAttention、GQA 与 linear attention 的资源变化。

![dense、局部窗口和全局连接的 query-key 图案；只有 kernel 跳过集合外位置时稀疏 mask 才减少实际工作](/assets/attention/svg/sparse-attention.1.svg)

## 把 attention 写成连接集合

### mask 决定每一行能读取哪些 key

设 query 位置为 $t$，key 位置为 $s$。对固定的 head 和 batch，令：

$$
E_t\subseteq\{0,\ldots,S-1\}
$$

表示 query $t$ 可以读取的 key 集合。标准 attention 的 score 记为 $\ell_{t,s}$。只在 $E_t$ 上做 row-wise softmax：

$$
A_{t,s}
=
\begin{cases}
\dfrac{\exp(\ell_{t,s})}
{\displaystyle\sum_{u\in E_t}\exp(\ell_{t,u})},
& s\in E_t,\\[10pt]
0,
& s\notin E_t.
\end{cases}
$$

因此每一行仍然是一个概率分布，但归一化分母只包含合法连接。若 $E_t$ 为空，分母没有定义；实现必须在进入 softmax 前处理空行，不能让它静默产生 NaN 或任意均匀值。

把所有 query 行的边集合合在一起：

$$
E=\{(t,s):s\in E_t\}.
$$

若 batch 和 head 使用同一个结构化图案，逻辑有效位置数为：

$$
N_E
=
B h_q\sum_{t=0}^{T-1}\lvert E_t\rvert.
$$

若每个 batch、每个 head 的图案不同，则必须按实际集合计数：

$$
N_E
=
\sum_{b=0}^{B-1}
\sum_{q=0}^{h_q-1}
\sum_{t=0}^{T-1}
\lvert E_{b,q,t}\rvert.
$$

后一行是计数描述，不是可直接代入的程序公式；实现审计时应展开为每个 $b,q,t$ 的 $\lvert E_{b,q,t}\rvert$。

### dense、causal 和 sparse 的位置数量

三种连接集合的区别如下：

|连接规则|单行可见 key|总位置对|标准 softmax 是否保留|
|---|---:|---:|---|
|dense cross|$S$|$B h_qTS$|是|
|dense self|$T$|$B h_qT^2$|是|
|dense causal self|$t+1$|$B h_qT(T+1)/2$|是|
|结构化 sparse|$\lvert E_t\rvert$|$B h_q\sum_t\lvert E_t\rvert$|在选中集合上保留|

causal mask 只限制未来位置；local、block、global 和 dilated pattern 还会限制远处或特定布局。多种规则可以取交集或并集，但最终必须给出实际的 $E_t$。

### 有效边数与实际 kernel 工作不是同一个数字

一个实现可能收到稀疏 mask，却执行完整的 $T\times S$ 矩阵乘法，再把集合外 score 加上负无穷。此时：

- 数学输出等于 masked dense attention；
- 逻辑有效边数是 $N_E$；
- 矩阵乘法仍可能处理 $B h_qTS$ 个位置；
- dense 临时矩阵仍可能按 $T\times S$ 分配。

只有 sparse kernel、block-sparse kernel 或专门的变长布局真正不读取、不计算集合外位置时，MAC 和中间矩阵字节数才会接近 $N_E$ 的账本。审计不能只看 mask 张量的非零比例。

## 局部窗口：把 $T^2$ 变成约 $Tw$

### causal sliding window 的定义

设窗口宽度为 $w$，每个 query 读取自己和最多 $w-1$ 个过去位置：

$$
E_t
=
\{s:\max(0,t-w+1)\le s\le t\}.
$$

当 $T\ge w$ 时，前 $w$ 行逐步变长，之后每行保持 $w$ 个 key。因此每个 head 的有效位置数是：

$$
\begin{aligned}
\sum_{t=0}^{T-1}\lvert E_t\rvert
&=
\sum_{r=1}^{w}r+(T-w)w\\
&=
\frac{w(w+1)}{2}+(T-w)w.
\end{aligned}
$$

当 $w$ 固定且 $T$ 增大时，这个数量是 $\mathcal O(Tw)$。它比 dense causal 的 $\mathcal O(T^2)$ 小，但只保留有限历史。

### 一个长度为 8 的小例子

取 $T=8$、$w=3$。每行可读 key 数为：

$$
(1,2,3,3,3,3,3,3).
$$

因此：

|规则|每行连接数|总连接数（单 batch、单 head）|相对 dense $8^2$|
|---|---|---:|---:|
|dense self|$(8,8,8,8,8,8,8,8)$|64|100%|
|dense causal|$(1,2,3,4,5,6,7,8)$|36|56.25%|
|causal window $w=3$|$(1,2,3,3,3,3,3,3)$|21|32.8125%|

local window 的第 7 行只能读取位置 5、6、7。它不能直接读取位置 0 到 4；如果需要这些信息，模型必须通过多层传递、全局 token 或其他跨窗口连接获得。

### 长序列的资源差异

取：

$$
B=2,\qquad
h_q=16,\qquad
T=S=4096,\qquad
d_k=d_v=64.
$$

将 dense rectangular、dense causal 和 causal window $w=256$ 放在同一账本中。下表的 MAC 只计 $QK^\mathsf T$ 和 $AV$，FP16 map 只计选中位置在理想稀疏存储中的 2-byte 权重：

|规则|逻辑位置对|attention MAC|权重字节|约 MiB|
|---|---:|---:|---:|---:|
|dense rectangular|536,870,912|68,719,476,736|1,073,741,824|1024|
|dense causal|268,500,992|34,368,126,976|537,001,984|512.125|
|causal window $w=256$|32,509,952|4,161,273,856|65,019,904|62.0078125|

window 的有效位置约为 dense causal 的 12.11%。如果 kernel 以稀疏布局保存权重，权重字节也接近这个比例；如果 kernel 仍使用完整矩形，实际字节和算术量可能更接近第一行或第二行。

### 双向局部窗口有不同的边界

encoder 中常见双向局部窗口可以用半径 $r$ 表示：

$$
E_t
=
\{s:\max(0,t-r)\le s\le\min(S-1,t+r)\}.
$$

内部 query 的连接数约为 $2r+1$，边界 query 更少。它不等于 causal window $w=2r+1$：causal window 只向过去连接，双向窗口还读取未来位置。计算量相近时，信息可见性和任务契约不同。

### 窗口限制也会改变 cache 生命周期

在自回归 decode 中，如果模型只需要最近 $w$ 个 key，旧 K/V 可以从 active cache 中移出，cache 长度从历史长度 $L$ 限制到 $w$。但如果 pattern 还包含 global token、摘要 token 或跨块连接，就必须保留这些额外 K/V：

$$
L_{\mathrm{active}}
\approx
w+\text{仍需保留的全局或跨块位置数}.
$$

因此 local attention 的 cache 节省取决于实际连接规则，不是只看训练期的窗口宽度。

## block-sparse：用规整块换取 kernel 效率

### 块 mask 的两层计数

把 query 和 key 轴各切成宽度为 $b$ 的块。块索引为：

$$
i=\left\lfloor\frac{t}{b}\right\rfloor,
\qquad
j=\left\lfloor\frac{s}{b}\right\rfloor.
$$

设保留的块索引集合为 $\mathcal P$。若所有块都是完整的 $b\times b$ tile，kernel 计划处理的元素数为：

$$
N_{\mathrm{tile}}
=
B h_q b^2\lvert\mathcal P\rvert.
$$

实际 token 边界、causal 对角块和 padding 可能让有效边数小于 $N_{\mathrm{tile}}$。块稀疏的工作量通常按 tile 计，因为硬件矩阵乘法需要处理规整块；逻辑边数和 kernel 处理的 tile 元素必须分别报告。

### 一个 $16\times16$ 矩阵的块例子

取 $T=S=16$、块宽 $b=4$，共有 $4\times4$ 个块。保留对角块和紧邻的前一块：

$$
\mathcal P
=
\{(i,j):0\le i<4,\ i-j\in\{0,1\}\}.
$$

保留的块数是：

$$
4+3=7.
$$

因此完整 tile 位置数为 $7\times4^2=112$。若对角块内部再使用 causal mask，真正合法的 token 对数是 4 个对角三角形加 3 个完整前一块：

$$
4\cdot\frac{4\cdot5}{2}+3\cdot4^2
=
88.
$$

对照数值如下：

|布局|矩形位置|causal 合法位置|块 kernel 处理的完整 tile 位置|
|---|---:|---:|---:|
|dense rectangular|256|—|256|
|dense causal|—|136|可能是 256|
|块对角加前一块|—|88|112|

块稀疏减少了完整 tile 的数量，但对角块仍有块内无效位置。块宽越大，硬件利用率可能越高，边界和块内浪费也可能越多。

### 长序列的块带

回到 $T=S=4096$、$b=128$。共有 32 个块。若每个块只保留对角块和前一块，保留 63 个块，batch/head 为 $B=2,h_q=16$：

$$
N_{\mathrm{tile}}
=
2\cdot16\cdot63\cdot128^2
=
33{,}030{,}144.
$$

其中对角块内部的 causal 合法 token 对为：

$$
2\cdot16
\left(
31\cdot128^2
+
32\cdot\frac{128\cdot129}{2}
\right)
=
24{,}707{,}072.
$$

前一个数字接近 kernel 的完整 tile 工作量，后一个数字是应用 causal mask 后的逻辑有效边数。两者都小于 dense causal 的 268,500,992，但它们不能互相替代。

### 块布局必须适合硬件

块数量少不保证延迟低。还要检查：

- 块宽是否与矩阵乘法的 tile 对齐；
- 每个 query block 的非零块数量是否相近；
- block index 是否连续，是否需要大量间接寻址；
- K/V 是否按块连续加载；
- 不同 batch 或 head 的 pattern 是否造成负载不均；
- 边界块和 padding 是否被单独填充。

一个规则的带状 pattern 可能比相同非零率的随机块更容易达到高吞吐。稀疏率只能说明数量级，不能单独决定 wall-clock latency。

## 全局、膨胀和随机连接

### 全局 token 提供远距离路径

设有 $g$ 个全局位置。普通 query 除局部窗口外还可以读取这些全局 key；全局 query 可以读取全部 key。self-attention 中，若窗口宽度和 $g$ 固定，连接数的数量级可以写为：

$$
\mathcal O\left(Tw+Tg+gT\right)
=
\mathcal O\left(T(w+g)\right).
$$

精确数量取决于全局 token 是否已经落在局部窗口中、全局位置是否同时作为 query、边界如何处理。全局 token 让远处位置可以经过少量中间节点互相传递，但也增加了每一行或每个全局 query 的连接。

全局 token 不是“免费的位置”。它们的 K/V 仍占 cache，global query 的完整 key 读取仍产生 $\mathcal O(gT)$ 的行。

### 膨胀连接保持固定度数

膨胀窗口按间隔 $d$ 读取：

$$
E_t
=
\{t,\ t-d,\ t-2d,\ldots,t-(w-1)d\}
\cap\{0,\ldots,T-1\}.
$$

每行最多 $w$ 个连接，所以位置数量仍为 $\mathcal O(Tw)$。增大 $d$ 可以扩大单层覆盖的距离，但相邻 token 之间的直接连接减少，多个 layer 的可达路径和边界行为需要单独核对。

若不同 layer 使用不同 dilation，复合图案可能覆盖更广；“每层都是 $w$ 个连接”不等于“多层网络拥有 dense 的直接边”。

### 随机连接保持数量，不保持布局

每个 query 随机选择 $k$ 个 key 时，期望位置数约为 $Tk$，与局部窗口同阶。随机 pattern 的额外成本来自：

- 索引存储和 gather；
- 不连续的内存访问；
- batch/head 之间的负载差异；
- 每次生成或重建 pattern 的同步；
- 随机种子、训练和推理 pattern 不一致。

因此随机稀疏可以在逻辑数量上是线性的，但不一定在目标硬件上获得同样的吞吐。

### 信息路径和算术数量是两条检查线

把每个允许的 query-key 连接看作有向边。局部 pattern 的单层图可能没有远距离边；全局 token 可以缩短图上的路径；膨胀 pattern 可能减少邻近边却增加跨度。这个图结构回答“信息能否在若干层到达”，而 $N_E$ 回答“每层需要处理多少位置”。

一个 pattern 可以有很低的连接数，却让某些位置之间的路径过长。另一个 pattern 可以有相同连接数，却因为全局节点分布不同而有更短的路径。不能仅用稀疏率推断长程依赖质量。

## 动态 top-k：先选 key 还是先算 dense score

### 先算全部 score 再 top-k 不会省掉 score

如果对每个 query 先计算所有 $S$ 个 score，再保留其中 top-k：

$$
\ell_{t,0},\ldots,\ell_{t,S-1}
\longrightarrow
\operatorname{top\text{-}k}
\longrightarrow
E_t,
$$

则 top-k 之后的 $AV$ 可能只处理 $k$ 个 value，但 $QK^\mathsf T$ 仍然处理 $TS$ 个位置。候选选择、排序和索引也有成本。这个过程可以减少后续权重存储或 value 读取，但不能把端到端复杂度直接写成 $\mathcal O(Tk)$。

### 候选生成器必须纳入账本

若使用近似索引、聚类、路由器或低成本检索器预先生成候选集合，端到端计算至少包含：

$$
\text{候选生成成本}
+
\text{候选内 score}
+
\text{候选内 softmax}
+
\text{候选内 }AV.
$$

只有候选生成器本身不扫描全部 key，且候选召回满足任务需要时，才可能得到真实的端到端节省。候选召回降低会直接删掉未进入集合的 key，质量与资源需要一起测量。

### 静态 pattern 和动态 pattern 的工程差异

|pattern|集合何时确定|容易优化的部分|主要风险|
|---|---|---|---|
|local window|由位置固定确定|连续块、缓存窗口|远距离信息需要多层路径|
|block sparse|由块索引固定确定|tile、矩阵乘法对齐|块内无效位置和负载不均|
|global token|由特殊位置固定确定|全局列/行复用|全局行产生长读取|
|random / dilated|由规则或种子确定|固定度数|访问不连续、可达性不稳定|
|dynamic top-k|依赖输入或 query|候选内计算|候选生成和召回成本|

固定结构通常更容易编译和缓存；动态结构可以按内容改变连接，但需要把选择过程作为模型和 kernel 的一部分审计。

## 稀疏 attention 与其他注意力优化的边界

### GQA 改变 K/V head，不改变位置连接

GQA 令多个 query head 共享 K/V head。若每个 query 仍读取全部合法 key，score 位置仍有：

$$
B h_qTS
$$

个。稀疏 attention 改变的是每个 query 的 $E_t$；GQA 改变的是 K/V 的 head 轴。两者可以组合：

$$
N_{\mathrm{pair}}
=
B h_q\sum_t\lvert E_t\rvert,
\qquad
N_{\mathrm{KV\text{-}cache}}
=
B L h_{kv}(d_k+d_v).
$$

一个组合配置可以同时减少位置连接和 K/V cache，但不能把一个轴的节省写成另一个轴的节省。

### FlashAttention 改变 IO，不改变 dense 连接数

FlashAttention 仍然为每个 dense query-key 对计算 score、mask、softmax 和 value 加权，只把计算分成 tile，并使用在线 softmax 统计量避免写出完整 $T\times S$ 矩阵。它的逻辑位置数仍是 dense 的 $B h_qTS$，而不是稀疏集合的 $N_E$。

如果把局部 mask 交给一个只支持 dense tile 的 FlashAttention kernel，可能仍然处理窗口外的 tile；要得到 sparse 的位置节省，需要支持该 pattern 的 sparse kernel 或变长索引布局。

### linear attention 改变 softmax 的代数

linear attention 通常用特征映射或其他重排，把 source 的统计量先累积，再查询这些统计量。它可能得到与 $T+S$ 相关的复杂度，但输出不自动等于标准 softmax attention。稀疏 attention 则可以在选中边上保留标准 softmax，只是删掉连接。

因此要分别回答：

|问题|应检查的对象|
|---|---|
|是否少算了位置|有效边集合 $E$ 与实际 kernel 访问|
|是否少写了大矩阵|tile 工作区和 HBM IO|
|是否改变了 softmax|归一化公式、特征映射和数值输出|
|是否少了 K/V cache|$h_{kv}$、历史长度和 dtype|
|是否保留了长程信息|多层连接图、全局节点和任务指标|

## 训练、padding 和 decode 的实现边界

### dense mask 与 sparse kernel 的差异

以下两段逻辑都可以得到同一个 masked output：

1. 生成完整 $QK^\mathsf T$，在集合外加负无穷，再做 softmax；
2. 只读取 $E_t$ 中的 K，直接在候选集合上做 softmax。

第一种保持 dense GEMM 的规整布局，但可能没有算力和显存节省。第二种接近 $N_E$ 的账本，但要处理索引、块布局、变长行和负载均衡。

验证时至少记录：

- logits 或 score 是否完整生成；
- softmax 的归一化长度是否是 $\lvert E_t\rvert$；
- value gather 是否只读取有效 key；
- 输出是否与 masked dense reference 一致；
- profiler 中实际矩阵乘法和显存读写是否下降。

### padding 会制造虚假的稀疏率

一个 batch 的 padding key 可能在逻辑上被 mask，但 dense kernel 仍按最大 $S$ 计算。packed sequence 或 varlen kernel 只有在使用真实长度 metadata 并跳过 padding 时，才会降低实际工作。

对每条样本记录：

$$
T_b,\qquad S_b,\qquad
\lvert E_{b,t}\rvert.
$$

不要用 batch 中的最大长度替代所有样本的有效长度，也不要把 padding mask 的零值直接计作已经节省的 FLOPs。

### local decode 需要明确旧 cache 的处理

在 local causal decode 中，第一个新 query 可能只读取最近 $w$ 个历史位置。如果实现仍保留全部历史 K/V 并让 kernel 扫描全部 $L$，局部 mask 没有降低每 token 的读取量。要获得 $O(w)$ 的读取，需要：

- active cache 只保留窗口和必需的 global K/V；
- index 或 ring buffer 正确处理位置偏移；
- positional encoding 与 cache slot 对齐；
- 生成超过窗口后验证旧 token 不再进入 $E_t$。

### cross-attention 的 sparse 轴可以不同

cross-attention 的 $T$ 是 target query 轴，$S$ 是 source key 轴。局部对齐窗口可能定义为 source 位置和 target 位置之间的对齐区域，而不是 self-attention 的同一坐标。应直接写出 $E_t\subseteq\{0,\ldots,S-1\}$，不能把 $s\le t$ 自动套到两条不同序列上。

## 失效模式与审计方法

### 只加 mask，仍调用 dense kernel

这是最常见的静默误判。输出可以正确，逻辑非零率也可以很低，但 profiler 仍显示完整 $T\times S$ 矩阵乘法。检查 kernel、实际读取和峰值显存。

### 把稀疏率当成延迟比例

10% 的非零位置不保证延迟降到 10%。块对齐、索引、负载不均和内存访问会改变常数。报告逻辑 MAC 与真实 latency 两组数字。

### 把 dense causal 与 local causal 混为一谈

causal 只移除未来；local causal 还移除窗口外的过去。先列出每行的 $E_t$，再计算总边数。

### 只按完整 tile 计数

block-sparse kernel 可能处理包含无效 token 的完整块。报告 $N_E$ 和 $N_{\mathrm{tile}}$，不要只给其中一个。

### 让全局 token 失去全局可见性

如果 global query 也被 local mask 截断，它只是一个普通局部 token。检查 global row、global column 和 padding 的三个方向。

### 膨胀间隔跨过了有效位置

dilated pattern 的索引必须同时满足边界、causal 和 padding 约束。检查生成的索引是否重复、越界或读到 padding。

### top-k 之前已经算完 dense score

如果候选选择依赖完整 score，$QK^\mathsf T$ 的二次项仍在。把候选生成成本单独记录。

### 把 GQA 当成稀疏位置连接

GQA 减少 K/V head，不减少每个 query head 的 key 位置。分别检查 $h_q$、$h_{kv}$ 和 $E_t$。

### 把 FlashAttention 当成 sparse attention

不物化 attention matrix 不表示只计算非零连接。检查 kernel 是否跳过窗口外或块外位置。

### 全 mask 行产生 NaN 或均匀输出

padding、source visibility 和 dynamic top-k 都可能使某一行没有候选。定义空行输出和 loss mask，并加 NaN 检查。

### pattern 在训练和推理间不一致

训练用固定 local mask、推理用不同的 cache window 或 global token，会改变可见路径和输出分布。把训练、prefill、decode 三个阶段分别列出 pattern。

### 一份最小稀疏注意力审计表

|检查项|应满足的条件|异常时先查|
|---|---|---|
|有效集合|明确每个 query 的 $E_t$|mask 生成器、边界|
|softmax 分母|只包含 $E_t$ 中的 key|softmax axis、负无穷 mask|
|位置计数|$N_E=B h_q\sum_t\lvert E_t\rvert$|是否误用 dense $TS$|
|实际 kernel|不读取集合外位置|kernel 类型、profiler|
|块计数|同时报告 token 边和 tile 边|块内 causal、padding|
|causal 规则|future key 不进入集合|past offset、上三角|
|local window|窗口宽度和方向明确|双向/causal 混用|
|global token|global row 和 column 都核对|全局索引、mask 广播|
|dilation|步长、宽度、边界明确|重复、越界|
|动态 top-k|候选生成成本独立计算|dense score、召回|
|GQA 轴|score 用 $h_q$，cache 用 $h_{kv}$|head reshape、group map|
|KV cache|只保存仍会被读取的 K/V|window、global、ring buffer|
|padding|有效长度和 padding 位置正确|packed metadata|
|空行|有定义的输出和 loss 行为|NaN、均匀 softmax|
|阶段一致|train/prefill/decode pattern 对齐|cache offset、配置|
|质量指标|长程、局部和任务指标同时测量|只看稀疏率|

## 相关词条

- [注意力复杂度](../attention/attention-complexity/)：定义 dense、causal、cross 和 decode 的资源账本。
- [Attention 矩阵](../attention/attention-matrix/)：说明 row-wise softmax、逻辑矩阵和 mask 轴。
- [因果掩码](../attention/causal-masking/)：推导下三角可见区域、padding 和 cache offset。
- [GQA 与 MQA](../attention/gqa-and-mqa/)：区分 K/V head 共享与位置连接稀疏。
- [交叉注意力](../attention/cross-attention/)：说明不同 source/target 长度上的矩形连接。
- [FlashAttention](../attention/flash-attention/)：说明 tile、在线 softmax 和显存 IO。
- [线性注意力](../attention/linear-attention/)：说明特征映射和标准 softmax 的代数边界。
- [注意力作为软检索](../attention/attention-as-retrieval/)：说明候选召回、top-k 和 value 读取。
