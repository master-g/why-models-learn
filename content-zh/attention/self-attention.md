---
title: "Self-Attention：同一序列如何互相读取"
tags: ["why-models-learn"]
---

Self-attention（自注意力）让同一条序列中的每个位置都生成一个 query、key 和 value，再用每个 query 去读取整条序列的 value。它的输入来源是同一条序列，输出仍然按原位置排列，但每个输出位置已经混入了其他位置的信息。一个位置如何读取另一个位置，由 query-key 的匹配分数、掩码和 softmax 权重共同决定。

设输入表示为 $H\in\mathbb R^{T\times d_{\mathrm{model}}}$，其中 $T$ 是序列长度。一次 self-attention 的核心计算可以写成：

$$
\begin{gathered}
Q=HW_Q,\\
K=HW_K,\\
V=HW_V.
\end{gathered}
$$

$$
\operatorname{SA}(H)
=
\operatorname{softmax}_{\mathrm{row}}\left(
\frac{QK^\mathsf T}{\sqrt{d_k}}+M
\right)V.
$$

这里 $W_Q$、$W_K$、$W_V$ 是三组独立的可学习投影，$M$ 决定哪些位置可以互相读取。输出的第 $t$ 行是第 $t$ 个 query 读取所有允许 key 对应 value 后得到的上下文表示。后文先固定单头、稠密矩阵的情形；多头拆分、缩放点积的数值原因和矩阵实现分别由[多头注意力](../attention/multi-head-attention/)、[Scaled Dot-Product Attention](../attention/scaled-dot-product/)和[Attention 矩阵](../attention/attention-matrix/)展开。

![同一序列生成三份投影，query-key 形成 pairwise 权重，再读取 value 得到上下文表示](/assets/attention/svg/self-attention.1.svg)

## 自注意力的定义

### “自”指输入来源相同

把一条长度为 $T$ 的序列记为：

$$
H\in\mathbb R^{T\times d_{\mathrm{model}}},
\qquad
H_{t,:}=\mathbf h_t^\mathsf T.
$$

每一行 $\mathbf h_t$ 是一个位置在进入该层时的表示。三组投影都从这一个 $H$ 出发：

$$
Q=
\begin{bmatrix}
\mathbf q_1^\mathsf T\\
\mathbf q_2^\mathsf T\\
\vdots\\
\mathbf q_T^\mathsf T
\end{bmatrix},
\qquad
K=
\begin{bmatrix}
\mathbf k_1^\mathsf T\\
\mathbf k_2^\mathsf T\\
\vdots\\
\mathbf k_T^\mathsf T
\end{bmatrix},
\qquad
V=
\begin{bmatrix}
\mathbf v_1^\mathsf T\\
\mathbf v_2^\mathsf T\\
\vdots\\
\mathbf v_T^\mathsf T
\end{bmatrix}.
$$

“自”不表示 $\mathbf q_t=\mathbf k_t=\mathbf v_t$。它表示三者都从同一条序列产生。$W_Q$、$W_K$ 和 $W_V$ 可以不同，因此同一个输入位置在匹配空间和内容空间中承担不同角色。

|对象|形状|回答的问题|在一次读取中的作用|
|---|---|---|---|
|输入 $H$|$T\times d_{\mathrm{model}}$|每个位置当前携带什么表示|同时产生三条投影路径|
|query $Q$|$T\times d_k$|当前 query 要找什么|作为每一行的读取条件|
|key $K$|$T\times d_k$|每个位置能被什么条件匹配|提供地址特征|
|value $V$|$T\times d_v$|被读回的内容是什么|接受注意力权重并被加权|
|score $S$|$T\times T$|第 $t$ 个 query 与第 $i$ 个 key 匹配多少|列出所有位置对|
|权重 $A$|$T\times T$|第 $t$ 个 query 分给各 key 多少读取质量|沿每一行归一化|
|输出 $C$|$T\times d_v$|第 $t$ 个位置读回了什么|形成上下文表示|

### 一行输出读取整条序列

先忽略掩码和缩放，令分数为：

$$
s_{t,i}=\mathbf q_t\mathbf k_i^\mathsf T.
$$

沿 key 位置 $i$ 做 softmax：

$$
\alpha_{t,i}
=
\frac{\exp(s_{t,i})}
{\displaystyle\sum_{j=1}^{T}\exp(s_{t,j})}.
$$

第 $t$ 个输出位置为：

$$
\mathbf c_t
=
\sum_{i=1}^{T}\alpha_{t,i}\mathbf v_i.
$$

因此，一行输出不是只更新自己的 value。它会从所有允许位置读取；如果某个位置的 score 较高，该位置在凸组合中的系数就较大。若所有有效位置的分数相同，输出就是这些 value 的平均。

把所有 query 同时计算，可以得到：

$$
S=QK^\mathsf T,
\qquad
A=\operatorname{softmax}_{\mathrm{row}}(S),
\qquad
C=AV.
$$

矩阵形式保留了两条不同的轴：$A_{t,i}$ 的行 $t$ 是谁在读取，列 $i$ 是读取了谁。交换这两个索引会改变问题。

## 一个三位置的可回放例子

### 先固定投影后的 $q$、$k$、$v$

为了把一次 self-attention 写成可以逐项核对的纸面计算，令 $d_k=d_v=1$，并把投影后的标量列出。这里的三个 token 只用作位置标签，不引入词义解释：

|位置|token|$q_t$|$k_t$|$v_t$|
|---:|---|---:|---:|---:|
|$1$|甲|$1$|$1$|$10$|
|$2$|乙|$0$|$0$|$20$|
|$3$|丙|$-1$|$-1$|$40$|

因为 $d_k=1$，缩放因子 $\sqrt{d_k}=1$。分数矩阵为：

$$
S=QK^\mathsf T
=
\begin{bmatrix}
1&0&-1\\
0&0&0\\
-1&0&1
\end{bmatrix}.
$$

第一行表示位置 $1$ 的 query。它与位置 $1$ 的 key 同向，与位置 $3$ 的 key 反向，因此第一行的三个分数依次为 $1$、$0$、$-1$。第二行的 query 为零，所以它对三个 key 的分数相同。

### 每一行产生自己的读取分布

对 $S$ 按行做 softmax，得到：

$$
A
=
\begin{bmatrix}
0.665240956&0.244728471&0.090030573\\
0.333333333&0.333333333&0.333333333\\
0.090030573&0.244728471&0.665240956
\end{bmatrix}.
$$

三行的分布不同。位置 $1$ 偏向读取位置 $1$，位置 $2$ 均匀读取，位置 $3$ 偏向读取位置 $3$。这三个输出共享同一组 value，却使用不同的 query 行。

### 加权读取形成上下文表示

把 value 写成列向量：

$$
V=
\begin{bmatrix}
10\\
20\\
40
\end{bmatrix}.
$$

矩阵乘法给出：

$$
C=AV
=
\begin{bmatrix}
15.148201906\\
23.333333333\\
32.404513384
\end{bmatrix}.
$$

以第一行为例：

$$
c_1
=
0.665240956\times10
+0.244728471\times20
+0.090030573\times40
=15.148201906.
$$

位置 $1$ 的输出已经含有位置 $2$ 和位置 $3$ 的 value。位置 $2$ 的输出是 $10$、$20$、$40$ 的平均，因为它的 query 与所有 key 都没有偏好。

### 同一个分数矩阵也可能产生非对称权重

上面的 $S$ 是对称矩阵，因为这里直接令 $q_i=k_i$。但 $A$ 不是对称矩阵：

$$
A_{1,2}=0.244728471,
\qquad
A_{2,1}=0.333333333.
$$

原因是 softmax 的分母按行变化。$S_{t,i}$ 可以表示匹配分数，而 $A_{t,i}$ 还包含第 $t$ 行与其他 key 的相对竞争。Attention map 不能直接当作对称相似度矩阵。

## Self-Attention 如何改变表示

### 静态 embedding 变成上下文表示

静态 embedding 只按 token ID 查一行矩阵。同一个 token 在不同上下文中查到同一行：

$$
\mathbf e_t=E[\mathrm{id}(x_t)].
$$

Self-attention 先从当前序列生成 query、key、value，再得到：

$$
\mathbf c_t
=
\sum_{i=1}^{T}\alpha_{t,i}\mathbf v_i.
$$

如果输入序列或相邻位置变化，$Q$、$K$、$V$ 和 $A$ 都可能变化，所以同一个 token 的输出可以不同。后续[前馈网络](../transformer-components/feedforward/)、残差路径和归一化还会继续变换这个表示。

|表示|依赖的输入|同一 token 在不同上下文中是否必然相同|主要限制|
|---|---|---|---|
|静态 embedding|token ID|相同|不读取邻近 token|
|self-attention 输出|整条可见序列和 mask|不相同|需要处理 $T^2$ 个位置对|
|带位置的 self-attention 输出|整条序列、位置特征和 mask|不相同|位置注入方式会影响顺序信息|

### 一层就能建立远距离的直接路径

在单向 RNN 中，位置 $1$ 影响位置 $T$ 通常要经过中间状态链。Self-attention 的分数矩阵直接包含位置 $1$ 到位置 $T$ 的一对：

$$
A_{T,1}
\quad\text{直接连接}\quad
\mathbf v_1\longrightarrow\mathbf c_T.
$$

这里的“直接”只指计算图中存在一条 attention 边，不代表 $A_{T,1}$ 一定较大，也不代表该边对任务输出具有因果解释。训练仍然要学习 query、key 和 value 的投影，使需要的信息获得合适的分数和内容表示。

若没有掩码，一层 self-attention 的每个位置都可以读取 $T$ 个位置；若使用因果掩码，第 $t$ 个位置只能读取允许的前缀。可见范围改变时，路径数量和归一化分母也会改变。

### “互相读取”不等于“平均混合”

Self-attention 的输出是 value 的凸组合，但权重由 query-key 匹配决定：

$$
\alpha_{t,i}\ge0,
\qquad
\sum_{i=1}^{T}\alpha_{t,i}=1
\quad\text{（没有被 mask 的行）}.
$$

因此它可以实现三种不同的读取状态：

|权重分布|输出近似|可观察结果|
|---|---|---|
|一个位置接近 $1$|一个 value 的近似拷贝|读取集中|
|少数位置占主要质量|少数 value 的混合|读取稀疏但仍可微|
|所有有效位置接近相同|有效 value 的平均|query 没有形成明显偏好|

“每个 token 都看所有 token”描述的是候选集合，不描述最终读取质量。要判断一次运行中的读取对象，至少要同时看 score、mask、softmax 行和 value。

## 顺序信息的边界

### 没有位置输入时，self-attention 对排列等变

如果 $H$ 中没有位置特征，且 mask 不依赖绝对位置，那么 self-attention 具有排列等变性。令 $P$ 是一个只交换序列位置的置换矩阵，则：

$$
H'=PH.
$$

三份投影随之变为：

$$
Q'=P Q,
\qquad
K'=P K,
\qquad
V'=P V.
$$

分数矩阵满足：

$$
S'
=Q'K'^\mathsf T
=P QK^\mathsf T P^\mathsf T
=P S P^\mathsf T.
$$

因为 softmax 沿行计算，权重矩阵和输出满足：

$$
A'=P A P^\mathsf T,
\qquad
C'=A'V'=P C.
$$

这表示交换输入位置只会同步交换输出位置。模型可以计算内容之间的关系，却不能从没有提供的特征中恢复原始顺序。

### 位置特征打破这种限制

要区分“甲在乙前面”和“乙在甲前面”，输入必须携带位置相关信号。常见做法是把位置向量加到 token 表示：

$$
\tilde{\mathbf h}_t
=
\mathbf h_t+\mathbf p_t,
$$

也可以在 score 中加入位置相关偏置，或把相对位置信息直接写进 query-key 的计算。具体形式由[位置编码](../transformer-components/positional-encoding/)、[RoPE](../transformer-components/rope/)和后续的位置方法词条展开。

加入位置特征后，两个位置即使 token 内容相同，也可能因为 $\mathbf p_t$ 不同而生成不同的 $q_t$、$k_t$。位置特征改变的是输入接口，不会自动保证模型学到正确的顺序关系；顺序任务仍需要数据、目标和评测共同提供信号。

## 掩码定义可见的读取图

### 掩码在 softmax 前改变候选集合

带掩码的 self-attention 写成：

$$
A_{t,i}
=
\frac{
\exp\left(
\frac{\mathbf q_t\mathbf k_i^\mathsf T}{\sqrt{d_k}}+M_{t,i}
\right)
}{
\displaystyle\sum_{j=1}^{T}
\exp\left(
\frac{\mathbf q_t\mathbf k_j^\mathsf T}{\sqrt{d_k}}+M_{t,j}
\right)
}.
$$

允许的位置使用 $M_{t,i}=0$，禁止的位置使用足够小的值；在数学记号中常写作 $-\infty$：

$$
M_{t,i}
=
\begin{cases}
0,&\text{位置 }i\text{ 对 query }t\text{ 可见},\\
-\infty,&\text{位置 }i\text{ 对 query }t\text{ 不可见}.
\end{cases}
$$

掩码不是输出后的展示标记。它必须在归一化前进入分数，使禁止位置不参加分母。

### 三种读取图对应三种语义

|读取图|第 $t$ 个 query 可读的位置|主要用途|需要检查的边界|
|---|---|---|---|
|全连接 self-attention|全部有效位置|编码整条已知序列|是否错误读取了 padding|
|因果 self-attention|当前位置及其左侧位置|自回归预测|是否偷看未来 token|
|padding-masked self-attention|batch 中该样本的有效位置|可变长度 batch|被屏蔽列是否仍进入 softmax 分母|

因果掩码的详细索引约定、训练与推理差异由[因果掩码](../attention/causal-masking/)处理。当前只保留接口事实：掩码改变 $A$ 的有效列，进而改变每个输出的加权 value 集合。

### 在 softmax 后删除位置会改变归一化

假设某一行原本有两个有效候选，权重为 $(0.8,0.2)$。如果把第二个位置在 softmax 后直接乘成零，剩余权重之和变为 $0.8$，输出不再是按有效集合归一化的 convex combination。若再除以 $0.8$，结果才恢复为只在第一个候选上的权重 $1$。

这个差异在 padding 和因果约束中会传播到上下文表示。审计时要记录 mask 应用阶段、填充值、softmax 轴和全屏蔽行的处理规则。

## 形状、参数与复杂度

### 先画出每个张量的轴

对单个样本，常见 shape 合同如下：

|张量|形状|行轴|列轴|应核对的含义|
|---|---|---|---|---|
|$H$|$T\times d_{\mathrm{model}}$|位置 $t$|输入特征|输入序列的长度和通道|
|$Q$|$T\times d_k$|query 位置|query 特征|每行负责一个读取条件|
|$K$|$T\times d_k$|key 位置|key 特征|每行对应一个可被读取位置|
|$V$|$T\times d_v$|value 位置|value 特征|与 $K$ 的行一一对应|
|$QK^\mathsf T$|$T\times T$|query 位置|key 位置|所有位置对的匹配分数|
|$A$|$T\times T$|读取者|被读取者|每行沿 key 轴归一化|
|$AV$|$T\times d_v$|输出位置|输出特征|每行是一次加权读出|

最常见的 shape 错误不是矩阵乘法报错，而是转置后仍然能够相乘，却把行轴和列轴的语义交换。例如把 $V$ 的顺序独立重排后再乘 $A$，数值仍可能有结果，但第 $i$ 个 key 的权重已经读到了第 $j$ 个 value。

### 三组投影的参数量分开计算

若忽略 bias，三组投影的参数量为：

$$
\lvert W_Q\rvert=d_{\mathrm{model}}d_k,
\qquad
\lvert W_K\rvert=d_{\mathrm{model}}d_k,
\qquad
\lvert W_V\rvert=d_{\mathrm{model}}d_v.
$$

总量为：

$$
\lvert W_Q\rvert+\lvert W_K\rvert+\lvert W_V\rvert
=
d_{\mathrm{model}}(2d_k+d_v).
$$

这不是 $T$ 的函数。序列变长时，单个 token 的投影参数不增加；增加的是运行时的 pair 数和中间矩阵大小。

### $T^2$ 来自位置对，而不是来自参数矩阵

分数矩阵有 $T^2$ 个元素。对 batch size 为 $B$ 的输入，主要运行量可以按下表分解：

|步骤|单样本计算量量级|中间结果|随长度变化|
|---|---:|---|---|
|生成 $Q,K,V$|$O(Td_{\mathrm{model}}(2d_k+d_v))$|三份投影|线性|
|计算 $QK^\mathsf T$|$O(T^2d_k)$|分数矩阵|二次|
|softmax 与 mask|$O(T^2)$|权重矩阵|二次|
|计算 $AV$|$O(T^2d_v)$|上下文输出|二次|

若 $T$ 翻倍，pair 数从 $T^2$ 变为 $4T^2$。以 $T=256$ 和 $T=512$ 为例：

$$
256^2=65\,536,
\qquad
512^2=262\,144
=4\times65\,536.
$$

因此长上下文的主要压力来自 $T\times T$ 的 score、softmax 或权重缓存，而不只是 token embedding 的查表。稀疏、局部或线性方法改变的是这部分 pair 计算，不能只用参数量比较成本。

## 把 self-attention 看成消息传递

### 权重矩阵是有向加权图

把每个位置视为一个节点，把 $A_{t,i}$ 视为从节点 $i$ 到节点 $t$ 的消息权重，则：

$$
\mathbf c_t=\sum_{i=1}^{T}A_{t,i}\mathbf v_i.
$$

每一行的权重和为 $1$，所以节点 $t$ 收到的是允许邻居 value 的加权平均。没有 mask 时图接近完全有向图；有 mask 时，禁止边的权重为零；对角线表示位置读取自身 value 的 self-loop。

|图上的对象|self-attention 中的对象|改变它会改变什么|
|---|---|---|
|节点状态|$H$ 的一行|该位置产生的 query、key、value|
|边分数|$s_{t,i}$|读取者与被读取者的匹配偏好|
|边权重|$A_{t,i}$|消息在输出中的占比|
|消息内容|$\mathbf v_i$|被读取的信息方向和尺度|
|邻接约束|$M$|哪些位置可以传递消息|

这个视角可以解释为什么 value 不能被 attention map 替代。两次运行可能有相同的 $A$，但 $V$ 不同，输出 $AV$ 仍然不同。

### 对角线大不等于只看自己

如果 $A_{t,t}$ 较大，说明当前位置在这一次读取中保留了较多自己的 value。它仍可能读取其他位置；只有当其他权重都接近零时，输出才近似为自身 value。Residual path 也可能额外保留输入，但那不是 attention 对角线本身。

### 注意力权重不是 cosine similarity

匹配分数可以使用点积，也可以使用其他函数。即使 $q_t$ 和 $k_i$ 归一化后使用余弦相似度，softmax 仍然会把一行分数变成依赖竞争集合的概率分布。余弦相似度是 pair 的几何量，$A_{t,i}$ 是给定 query 行、mask 和 temperature 后的归一化读取权重。

## 梯度如何经过同一条读取路径

### value 路径直接接收读取权重

先把 value 和 context 暂时看作标量：

$$
c_t=\sum_{i=1}^{T}\alpha_{t,i}v_i.
$$

对某个 value 求偏导：

$$
\frac{\partial c_t}{\partial v_i}
=\alpha_{t,i}.
$$

权重较大的 value 对该位置的 context 具有更大的直接局部影响。这个导数只描述固定权重下的 value 路径，不等于整个任务输出对输入 token 的总因果效应。

### score 路径通过 softmax 耦合

因为 $\alpha_{t,i}$ 依赖整行 score，所以改变一个 score 会同时改变多个权重。对标量 value，有：

$$
\frac{\partial c_t}{\partial s_{t,i}}
=
\alpha_{t,i}(v_i-c_t).
$$

如果 $v_i$ 高于当前 context，增加它的 score 会提高 context；如果 $v_i$ 低于当前 context，增加它的 score 会降低 context。所有 score 导数之和为零：

$$
\sum_{i=1}^{T}
\frac{\partial c_t}{\partial s_{t,i}}
=0.
$$

这是 softmax 行归一化的局部约束。它不表示梯度为零，而表示在所有 score 同时平移相同常数时，权重不变。

### query、key、value 学到的内容不同

同一输入的三份投影在反向传播中收到不同路径的梯度：

|投影|前向职责|梯度主要回答的问题|
|---|---|---|
|$W_Q$|为读取者编码条件|当前位置应该寻找什么|
|$W_K$|为被读取者编码地址|什么条件可以匹配当前位置|
|$W_V$|编码被传递的内容|当前位置应该向其他位置提供什么|

这三个职责是接口分工，不是对训练后向量的语义保证。要确认某个维度学到了什么，需要做替换、遮蔽、线性探针或任务级干预，不能只按变量名推断。

## 失效模式：把“同一序列”误读成实现细节

### 1. 把 self 当成只读取对角线

Self-attention 的 self 指 Q、K、V 来自同一输入来源。一次完整读取通常包含 $T^2$ 个位置对；只保留对角线是另一种受限算子。

### 2. 把 $Q$、$K$、$V$ 当成同一个矩阵

三者的行索引都对应位置，但列空间和投影参数可以不同。把它们合并会丢失“匹配地址”和“传输内容”的接口区别。

### 3. 忘记位置特征

没有位置输入时，网络对输入排列等变。若任务依赖先后顺序，必须检查位置特征是否进入 $H$、score 或 mask。

### 4. 在错误轴上做 softmax

对 $T\times T$ 的分数矩阵，读取语义要求每个 query 行沿 key 列归一化。沿另一轴归一化会把“每个 query 分配读取质量”改成“每个 key 接收固定总质量”。

### 5. 在 softmax 后才删除 padding

先归一化再清零会让有效权重和小于 $1$，除非额外重新归一化。应在分母中排除禁止位置，并单独处理全屏蔽行。

### 6. 只看权重，不看 value

权重只描述读取比例。value 的内容、符号、方向和尺度决定加权后的上下文；相同 attention map 可以对应不同输出。

### 7. 把 attention map 当作因果证明

高权重表示一次前向中的读取系数较大。因果结论还需要遮蔽、替换、反事实或输出干预，并控制其他路径的补偿。

### 8. 只计算参数量，不计算 $T^2$

$W_Q$、$W_K$、$W_V$ 的参数量与序列长度无关，但分数矩阵和权重矩阵都包含 $T^2$ 个元素。报告资源时要分别列出参数、算力、激活内存和缓存。

## 审计协议

### 先核对一次 forward 的合同

给定一组固定的 $H$、投影矩阵和 mask，按以下顺序记录：

1. 检查 $H$、$Q$、$K$、$V$ 的 shape。
2. 检查 $QK^\mathsf T$ 的行是否对应 query、列是否对应 key。
3. 检查 mask 是否在 softmax 前加入。
4. 检查每个非空 query 行的权重是否非负且和为 $1$。
5. 检查 $C$ 是否等于按行权重读取对应 value 的结果。
6. 检查 value 的行顺序是否与 key 完全对齐。
7. 在改变输入顺序时，同时改变位置信号，区分排列等变性与位置编码效果。

### 用三位置例子做最小回放

本文的 toy 设置可作为不依赖框架的 sanity check：

|检查项|预期结果|
|---|---|
|$S=QK^\mathsf T$|$\begin{bmatrix}1&0&-1\\0&0&0\\-1&0&1\end{bmatrix}$|
|第一行 softmax|$(0.665240956,0.244728471,0.090030573)$|
|第二行 softmax|$(1/3,1/3,1/3)$|
|第三行 softmax|$(0.090030573,0.244728471,0.665240956)$|
|$C=AV$|$(15.148201906,23.333333333,32.404513384)^\mathsf T$|
|每个权重行的和|约为 $1$|
|$A_{1,2}$ 与 $A_{2,1}$|分别为 $0.244728471$ 与 $0.333333333$|

如果这些结果不一致，先查 $QK^\mathsf T$ 的转置、softmax 轴、value 排列和缩放因子，再查优化器或训练数据。

## 结语

Self-attention 的接口可以压缩为：

$$
H
\longrightarrow
(Q,K,V)
\longrightarrow
QK^\mathsf T+M
\longrightarrow
A
\longrightarrow
AV.
$$

输入来源相同，使每个位置能够直接读取同一序列中的其他位置；三组投影不同，使匹配条件、地址特征和传输内容可以分别学习。这个读取过程带来 $T^2$ 的位置对，也带来没有位置特征时的排列等变边界。判断一次实现是否符合 self-attention，需要同时核对 shape、softmax 轴、mask 阶段、value 对齐、位置输入和运行时复杂度。

## 相关词条

[Attention 作为检索](../attention/attention-as-retrieval/)

[Scaled Dot-Product Attention](../attention/scaled-dot-product/)

[Attention 矩阵](../attention/attention-matrix/)

[多头注意力](../attention/multi-head-attention/)

[因果掩码](../attention/causal-masking/)

[位置编码](../transformer-components/positional-encoding/)

[RoPE](../transformer-components/rope/)

[Embedding](../text-representation/embeddings/)

[残差连接](../cnn/residual-connections/)
