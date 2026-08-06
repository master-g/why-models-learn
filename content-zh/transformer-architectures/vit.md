---
title: "Vision Transformer：图像 patch 与全局注意力"
tags: ["why-models-learn"]
---

Vision Transformer（ViT）先把图像切成固定大小的 patch，把每个 patch 线性投影成一个 token，再把 patch token 和位置嵌入送入 encoder-only Transformer。它把空间局部性从卷积核的硬编码先验改成 token 化后的输入合同；patch size、位置嵌入和全局 attention 的序列长度共同决定模型的表达路径与资源账本。

![ViT 把图像切成 patch，投影为 token，加入 CLS 与位置嵌入后进入 Transformer encoder，再由分类头读取 CLS](/assets/transformer-architectures/svg/vit.1.svg)

## 从图像到 patch token

### 先固定图像轴

设一个 batch 中的单张图像为：

$$
\mathbf X\in\mathbb R^{C\times H\times W},
$$

其中 $C$ 是通道数，$H$ 和 $W$ 是高度与宽度。本文用 channel-first 记号；实现也可能使用 $H\times W\times C$，但 patch 的空间位置、通道顺序和内存布局必须保持一致。

选取正方形 patch 边长 $P$，并先要求 $P$ 整除 $H$ 和 $W$：

$$
n_h=\frac{H}{P},
\qquad
n_w=\frac{W}{P},
\qquad
N=n_hn_w.
$$

其中 $N$ 是 patch token 数，不包括 CLS token。若 $H$ 或 $W$ 不能被 $P$ 整除，就必须明确裁剪、padding 或保留不完整边界 patch 的规则，不能直接把 floor 后的数量当作图像已被完整覆盖。

### 每个 patch 的原始维度

按行索引 $r=0,\ldots,n_h-1$ 和列索引 $s=0,\ldots,n_w-1$，取出空间窗口：

$$
\mathbf X_{r,s}
=
\mathbf X[:,\,rP:(r+1)P,\,sP:(s+1)P]
\in\mathbb R^{C\times P\times P}.
$$

展平 patch：

$$
\mathbf q_{r,s}
=
\operatorname{vec}(\mathbf X_{r,s})
\in\mathbb R^{CP^2}.
$$

因此，patch projection 的输入维度是 $CP^2$。它不是一个已经具有 $D$ 维语义的 token；这一步只把局部像素块排列成向量。

### patch projection 把像素向量映射到模型宽度

设 Transformer 的 hidden width 为 $D$。使用同一组参数处理所有空间位置：

$$
\mathbf z_{r,s}
=
\mathbf q_{r,s}W_E+\mathbf b_E
\in\mathbb R^{D},
$$

其中：

$$
W_E\in\mathbb R^{CP^2\times D},
\qquad
\mathbf b_E\in\mathbb R^{D}.
$$

同一个 $W_E$ 在所有 patch 位置复用，这一点与卷积的权值共享相似。若把输入布局和参数排列对齐，patch projection 等价于 kernel size 和 stride 都为 $P$ 的卷积 stem：

$$
\operatorname{Conv2d}
\left(
\text{kernel}=P,\quad
\text{stride}=P,\quad
\text{in}=C,\quad
\text{out}=D
\right).
$$

等价关系只说明局部线性投影和权值共享，不意味着 ViT 后续层具有 CNN 的局部卷积连接或平移等变性。

### patch 的线性顺序必须固定

Transformer 接收一维序列，因此要把二维 patch 网格按固定顺序排列。常见做法是先按行、再按列：

$$
i=rn_w+s,
\qquad
0\le r<n_h,
\qquad
0\le s<n_w.
$$

于是：

$$
\mathbf z_i=\mathbf z_{r,s},
\qquad
i=0,\ldots,N-1.
$$

行优先、列优先、空间窗口的展平顺序以及通道顺序都属于输入协议。训练和推理只要有一项不同，模型看到的 token 位置就会改变，即使张量总 shape 没有变化。

## CLS、位置嵌入和 encoder 输入

### CLS token 不是图像 patch

分类模型常加入一个可学习的 CLS 向量：

$$
\mathbf x_{\mathrm{cls}}\in\mathbb R^{D}.
$$

把它放在 patch token 前面：

$$
\mathbf Z
=
\begin{bmatrix}
\mathbf x_{\mathrm{cls}}\\
\mathbf z_0\\
\vdots\\
\mathbf z_{N-1}
\end{bmatrix}
\in\mathbb R^{(N+1)\times D}.
$$

序列长度因此变为：

$$
T=N+1.
$$

CLS token 没有对应的像素窗口。它通过 self-attention 读取 patch token，最后由分类头使用；patch token 则保留各自的空间位置。若任务需要密集输出，可以不加入 CLS，或在输出时丢弃 CLS 后把剩余 token 还原成 patch 网格。

### 位置嵌入恢复二维位置条件

如果只把 patch token 当作一个集合，Transformer 的 self-attention 对 token 置换没有空间顺序偏好。ViT 通常加入位置嵌入：

$$
\mathbf Z_0
=
\mathbf Z+\mathbf E_{\mathrm{pos}},
\qquad
\mathbf E_{\mathrm{pos}}
\in\mathbb R^{(N+1)\times D}.
$$

每个位置都有一个 $D$ 维向量。CLS 位置和 patch 位置可以使用不同的初始化或参数索引。加入后，位于网格左上角和右下角的 token 不再只由像素内容区分，它们还携带不同的位置向量。

常见位置合同包括：

|位置合同|参数或计算|分辨率变化时的处理|
|---|---|---|
|learned 1D embedding|直接学习 $(N+1)\times D$ 表|按 patch 网格重排后插值|
|learned 2D embedding|分别对二维网格位置建模|沿高度和宽度插值|
|sinusoidal embedding|按位置函数计算|通常可以直接计算新长度|
|relative position|在 attention score 中加入相对位移项|需要定义新网格的相对坐标范围|

位置嵌入不是可有可无的装饰。训练分辨率改变后，位置表长度、CLS 行、patch 网格尺寸和插值方式都必须同时检查。

### encoder stack 的输入合同

把带位置的序列送入 encoder-only Transformer：

$$
\mathbf H^{(0)}=\mathbf Z_0,
\qquad
\mathbf H^{(\ell)}
=
\operatorname{EncoderBlock}_{\ell}
\left(\mathbf H^{(\ell-1)}\right),
\qquad
\ell=1,\ldots,L.
$$

每一层的 self-attention 允许任意两个序列位置交互。ViT 的全局 attention 轴是 token 轴，矩阵形状为：

$$
\mathbf A\in\mathbb R^{B\times h\times T\times T}.
$$

这里 $T=N+1$，包含 CLS；如果有 padding 或额外 special token，必须把它们计入 $T$。

## 一个小图像的完整 shape 账本

### 取 $8\times8$ RGB 图像

取：

$$
H=W=8,
\qquad
C=3,
\qquad
P=4,
\qquad
D=6.
$$

空间网格为：

$$
n_h=n_w=2,
\qquad
N=4,
\qquad
T=N+1=5.
$$

每个 patch 的原始维度是：

$$
CP^2=3\cdot4^2=48.
$$

从图像到 encoder 的 shape 如下：

|阶段|张量|shape|含义|
|---|---|---|---|
|输入|$\mathbf X$|$3\times8\times8$|RGB 图像|
|切块|$\mathbf X_{r,s}$|$3\times4\times4$|2×2 patch 网格中的一个窗口|
|展平|$\mathbf q_{r,s}$|$48$|一个 patch 的像素向量|
|投影|$\mathbf z_i$|$6$|一个 patch token|
|拼接|$\mathbf Z$|$5\times6$|1 个 CLS + 4 个 patch|
|加位置|$\mathbf H^{(0)}$|$5\times6$|encoder 输入|
|attention|$\mathbf A$|$h\times5\times5$|每个 head 的 token 交互|

一个实现若报告 encoder 输入为 $4\times6$，通常意味着它没有加入 CLS，或者报告的是不含 CLS 的 patch 子序列。两种形状都可以成立，但分类头读取的对象不同。

### 分类头读取 CLS 或 patch 平均

CLS 分类合同为：

$$
\mathbf h_{\mathrm{cls}}
=
\mathbf H^{(L)}_{0,:},
\qquad
\boldsymbol\ell
=
\mathbf h_{\mathrm{cls}}W_c+\mathbf b_c
\in\mathbb R^{K}.
$$

其中 $K$ 是类别数。另一种合同是对 patch token 求平均：

$$
\mathbf h_{\mathrm{mean}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf H^{(L)}_{i,:},
\qquad
\boldsymbol\ell
=
\mathbf h_{\mathrm{mean}}W_c+\mathbf b_c.
$$

CLS 和 mean pooling 不是同一个 readout。CLS 需要通过 attention 学习汇总接口；mean pooling 固定沿 patch 轴平均，可能保留更直接的整体统计。比较分类结果时应记录 pooling 方式。

### 密集任务保留 patch 网格

对于 patch-level 分类、分割或检测，需要保留 patch token：

$$
\mathbf H_{\mathrm{patch}}
=
\mathbf H^{(L)}_{1:N,:}
\in\mathbb R^{N\times D}.
$$

再按 $i=rn_w+s$ 还原：

$$
\mathbf G_{r,s,:}
=
\mathbf H_{\mathrm{patch},i,:}
\in\mathbb R^{n_h\times n_w\times D}.
$$

这个网格的分辨率是 patch 网格分辨率，不是原始像素分辨率。要输出像素级结果，还需要上采样、decoder、跨尺度特征或其他重建路径。直接把一个 patch token 当作一个像素预测会产生空间尺寸错误。

## 资源账本：patch size 影响什么

### patch projection 的 MAC

忽略 bias 加法，所有 patch projection 的 MAC 为：

$$
\begin{aligned}
M_{\mathrm{patch}}
&=
N(CP^2)D\\
&=
\left(\frac{H}{P}\frac{W}{P}\right)CP^2D\\
&=
HWCD.
\end{aligned}
$$

当 patch 不重叠、投影宽度 $D$ 固定时，改变 $P$ 不改变这项理想 MAC；它改变的是 token 数。真实实现仍可能受到 kernel tile、内存访问和 padding 的影响。

### attention 和 FFN 随 token 数增长

对 $B$ 张图像、$h$ 个 head、模型宽度 $D$、序列长度 $T$，忽略 bias、norm 和激活函数：

|计算|逻辑 MAC|
|---|---:|
|Q、K、V 和输出投影|$4BTD^2$|
|$QK^{\mathsf T}$|$BT^2D$|
|$AV$|$BT^2D$|
|两次 FFN 投影|$2BTDM$|

attention 的两项交互合计为 $2BT^2D$。因此，patch size 减小后，$T$ 增加，二次项会快速增长；patch projection 的理想 MAC 不变，不能用它代表整个 ViT 的计算量。

### 224×224 图像的 patch size 对照

忽略 CLS 以外的特殊 token，只比较 $H=W=224$ 的 patch 网格：

|patch size $P$|patch 数 $N$|含 CLS 的 $T$|$T^2$|相对 $P=16$ 的 $T^2$|
|---:|---:|---:|---:|---:|
|32|49|50|2,500|0.0644|
|16|196|197|38,809|1|
|8|784|785|616,225|15.8784|

从 $P=16$ 改为 $P=8$ 时，patch 数变为 4 倍，attention 位置对接近 16 倍。更小的 patch 保留更细的局部信息，但会直接增加 token 轴上的交互、激活和位置表长度。

### 小模型的参数和 MAC

回到 $H=W=8,C=3,P=4,D=6$，再取：

$$
h=2,
\qquad
d_h=3,
\qquad
M=12,
\qquad
K=3,
\qquad
L=2.
$$

带 bias 的 patch projection 参数量：

$$
m_{\mathrm{patch}}
=
(CP^2)D+D
=
48\cdot6+6
=294.
$$

CLS 向量和位置嵌入分别需要：

$$
m_{\mathrm{cls}}=D=6,
\qquad
m_{\mathrm{pos}}=TD=5\cdot6=30.
$$

每个 encoder block 由 MHA、FFN 和两个 LayerNorm 组成。若 Q/K/V/output projection 和 FFN 都带 bias：

$$
\begin{aligned}
m_{\mathrm{MHA}}
&=
4(D^2+D)
=168,\\
m_{\mathrm{FFN}}
&=
2DM+M+D
=162,\\
m_{\mathrm{LN}}
&=
2(2D)
=24,\\
m_{\mathrm{block}}
&=
m_{\mathrm{MHA}}+m_{\mathrm{FFN}}+m_{\mathrm{LN}}
=354.
\end{aligned}
$$

最终 LayerNorm 有 $2D=12$ 个参数，分类头有 $DK+K=21$ 个参数。总参数量为：

$$
\begin{aligned}
m_{\mathrm{ViT}}
&=
m_{\mathrm{patch}}+m_{\mathrm{cls}}+m_{\mathrm{pos}}
+Lm_{\mathrm{block}}+2D+(DK+K)\\
&=
294+6+30+2\cdot354+12+21\\
&=1071.
\end{aligned}
$$

忽略 norm、softmax、激活和 bias 加法，单张图像的逻辑 MAC 为：

$$
\begin{aligned}
M_{\mathrm{patch}}
&=
4\cdot48\cdot6
=1152,\\
M_{\mathrm{block}}
&=
4TD^2+2T^2D+2TDM\\
&=
4\cdot5\cdot6^2+2\cdot5^2\cdot6+2\cdot5\cdot6\cdot12\\
&=
1740,\\
M_{\mathrm{ViT}}
&=
M_{\mathrm{patch}}+LM_{\mathrm{block}}+DK\\
&=
1152+2\cdot1740+6\cdot3\\
&=4650.
\end{aligned}
$$

其中 $DK=18$ 只计分类头的矩阵乘法。不同实现是否包含 bias、norm、softmax、dropout 和 patch padding，会改变实测 FLOPs，但不应改变上述 shape 关系。

## self-attention 如何混合 patch

### patch token 通过全局连接交换信息

对第 $\ell$ 层的输入 $\mathbf H\in\mathbb R^{B\times T\times D}$，单个 head 的投影为：

$$
\mathbf Q=\mathbf HW_Q,
\qquad
\mathbf K=\mathbf HW_K,
\qquad
\mathbf V=\mathbf HW_V.
$$

缩放点积 attention 为：

$$
\operatorname{Attn}(\mathbf H)
=
\operatorname{softmax}
\left(
\frac{\mathbf Q\mathbf K^{\mathsf T}}{\sqrt{d_h}}
\right)\mathbf V.
$$

对于 ViT 的 patch token，score 矩阵的两个位置轴都覆盖 CLS 和所有 patch。左上 patch 可以在一层内读取右下 patch，不需要像小卷积核那样通过多层逐步扩大感受野。

### 全局 attention 不是二维局部先验

图像 patch 的二维相邻关系只通过位置嵌入、输入序列顺序和训练数据进入模型。attention 本身默认可以连接任意 token 对。若需要局部 attention、窗口 attention 或层级特征，就必须在 mask、架构或分辨率路径中明确加入，不能从“输入来自图像”推出局部连接。

这也是 ViT 和 CNN 的结构差异：

|属性|CNN|ViT|
|---|---|---|
|初始 token/feature|像素网格和局部卷积响应|非重叠 patch token|
|单层主要连接|局部窗口|默认全局 token 对|
|参数共享|卷积核跨空间位置共享|patch projection 跨 patch 位置共享，attention 投影跨 token 共享|
|位置合同|空间坐标由 feature map 保留|需要位置嵌入或相对位置机制|
|远距离交互|通过多层感受野扩大|一层 attention 可以直接交互|
|密集输出|保留 feature map|需丢弃 CLS 并重建 patch 网格|

ViT 可以学习局部模式，但这种局部性来自参数和数据中的可学习规律，不是由一个 $P\times P$ patch projection 自动保证的。

## 训练与推理的图像合同

### 预处理必须和 patch 化对齐

图像进入 patchify 之前通常会做 resize、crop、通道归一化和数据类型转换。每个步骤都可能改变 patch 的数值或空间位置：

1. 记录 resize 的目标 $H,W$ 和插值方式；
2. 记录 crop 的坐标和训练/评估是否一致；
3. 固定通道顺序、均值、标准差和数值范围；
4. 在归一化之后再执行 patch projection；
5. 用一张固定图像核对前几个 patch 的像素顺序和投影输入。

如果训练使用 RGB 而推理使用 BGR，或者训练按 channel-first 展平、推理按 channel-last 展平，模型参数仍能产生 shape 正确的输出，但 token 内容已经错位。

### 训练分辨率和推理分辨率

当推理分辨率变化时，patch 数变化为：

$$
N_{\mathrm{new}}
=
\frac{H_{\mathrm{new}}}{P}
\frac{W_{\mathrm{new}}}{P}.
$$

如果使用 learned absolute position embedding，原位置表通常不再匹配 $(N_{\mathrm{new}}+1)\times D$。常见处理是：

1. 拆出 CLS 的位置向量；
2. 把 patch 位置表重排为 $n_h\times n_w\times D$；
3. 在二维网格上插值到新尺寸；
4. 再把插值结果展平并和 CLS 位置拼回。

不能把包含 CLS 的一维表直接当成二维 patch 网格插值，否则会把分类 token 混入空间位置。

### 分类损失只监督图像级输出

图像分类中，最终 logits 通常是：

$$
\boldsymbol\ell\in\mathbb R^{K},
$$

每张图像对应一个标签。patch token 没有单独的类别标签，除非任务额外提供 patch-level supervision、遮挡重建或蒸馏目标。不能因为 attention 输出包含 $T$ 个 token，就把分类标签复制到每个 patch 位置而不改变目标定义。

### 推理时的 batch 和序列长度

ViT 没有 decoder-only 的逐 token KV cache 合同。一次图像前向通常同时处理整张图像的 $T$ 个 token；如果使用滑动窗口、多裁剪或视频帧，batch 和 token 数会按实际视图数量增长。多裁剪评估要记录每个 crop 的 patch 网格、位置插值和最终 logits 聚合方式。

## 位置嵌入与图像几何

### 绝对位置嵌入记录离散网格

learned absolute position embedding 为每个离散 patch 位置分配一个向量。它可以让模型区分相同局部内容出现在不同网格位置的情况，但它不自动表达连续平移、尺度变化或旋转关系。数据增强和插值策略会影响模型对这些变换的响应。

### 2D 位置不能被 1D 索引完全替代

row-major 索引把二维坐标 $(r,s)$ 映射为 $i=rn_w+s$。如果只用一个任意的 1D 位置序列，模型仍可以通过学习相邻 index 的关系来近似空间结构，但横向边界、纵向边界和同一列距离不会被显式区分。2D position embedding 或相对位置机制可以把这些坐标关系直接写入合同。

### 位置插值不是像素插值

位置表插值只对 $D$ 维位置向量做网格插值，不会改变图像像素。图像 resize 是输入预处理，位置插值是模型参数或中间表示的适配；两者发生在不同空间，不能用同一个 resize 配置替代。

## 与 CNN、encoder-only 和 patchify stem 的关系

### ViT 复用了 encoder-only 的序列合同

ViT 的主体是 encoder-only Transformer：输入序列可以双向 self-attention，输出仍然为每个 token 保留一个 contextual hidden。[Encoder-Only](../transformer-architectures/encoder-only/) 词条中的 padding、CLS/mean pooling、token head 和 sequence head 也适用于 ViT；ViT 的额外工作是把二维图像建立为 token 序列。

### patch projection 和 CNN stem 的边界

patch projection 可以等价为 stride 等于 kernel size 的一次卷积，但后续结构不同：

|模块|空间交互|下采样|输出形状|
|---|---|---|---|
|ViT patch projection|每个 patch 内部的线性组合|一次直接变成 patch 网格|$N\times D$|
|CNN convolution|局部重叠窗口，可多层堆叠|由 stride/pooling 逐级控制|$H'\times W'\times C'$|
|CNN patchify stem|较大 kernel/stride 的初始投影|stem 中完成初始下采样|低分辨率 feature map|

[ResNet 后续架构](../cnn/resnet-and-beyond/)中的 patchify stem 可以采用类似的初始下采样，但它仍然属于卷积 feature map 路径，不等于 ViT 的 global token attention。

### 全局 attention 的数据需求

ViT 在数据量较小时可能需要更强的数据增强、正则化或预训练，因为它没有 CNN 那样强的局部性和平移结构先验。这个现象不是“Transformer 不会看局部”，而是模型需要从样本中学习局部模式、空间关系和变换稳定性。

## 独立数值核对

下面的数值由独立标准库脚本计算，再与正文的 shape、参数和 MAC 公式逐项比对。脚本只实现整数算术和显式参数公式，没有使用深度学习框架。

|核对项|输出|
|---|---|
|配置 $(H,W,C,P)$|$(8,8,3,4)$|
|patch grid $(n_h,n_w)$|$(2,2)$|
|patch 数 $N$ / 含 CLS 的长度 $T$|4 / 5|
|patch 原始维度 $CP^2$|48|
|patch projection 参数|294|
|CLS / position 参数|6 / 30|
|一个 encoder block 参数|354|
|两层 block、final norm、3 类 head 的总参数|1071|
|patch projection MAC|1152|
|一个 block MAC|1740|
|完整小模型 MAC|4650|
|$224\times224$、$P=16$ 与 $P=8$ 的 $T^2$|38,809 / 616,225|

这些数字验证了 patch 化和资源账本。它们不能证明分类精度、数据效率或位置插值质量；这些结论需要在固定数据、训练预算和评估协议下另行测量。

## 失效模式

### 图像没有被完整覆盖

当 $H$ 或 $W$ 不是 $P$ 的整数倍时，floor、ceil、padding 和 crop 会产生不同 patch 数。检查输入尺寸、边界像素是否保留、padding 值和位置表长度，不要只检查最终 token shape。

### patch 展平顺序错位

channel-first 与 channel-last、行优先与列优先、通道交错与通道连续都会生成相同长度的向量。使用非对称的人工图像，让每个通道、每个行列位置具有不同数值，逐个比较第一个和最后一个 patch 的向量。

### CLS 位置或 pooling 合同错

分类头可能读取 CLS，也可能读取 patch mean。把不含 CLS 的序列传给读取第 0 个 CLS 的 head，或把 CLS 一并平均，会让输出仍然有正确的类别维度，但语义接口已经改变。

### learned position embedding 长度不匹配

改变分辨率后，patch 数和位置表长度不再一致。直接截断、重复或把 CLS 混入二维插值都可能让模型运行，但会改变位置合同。应记录原网格、新网格、CLS 行的处理和插值方法。

### 把 patch token 当作像素输出

一个 patch token 对应一个空间块，不对应一个像素。密集预测需要上采样、decoder 或其他重建路径，并且要记录 patch 网格和目标像素之间的对齐关系。

### 把全局 attention 当作局部先验

ViT 的 score 矩阵默认连接所有 token 对。若目标需要窗口、稀疏或层级的局部计算，必须检查 mask、attention kernel 或 backbone stage；输入是图像并不能证明这些连接已经存在。

### 只看 patch projection 的计算量

不重叠 patch 的 projection MAC 在理想公式中与 $P$ 抵消，但 attention 的 $T^2$ 项不会抵消。比较 patch size 时必须同时报告 token 数、attention 位置对、激活内存和实际吞吐。

### 训练与推理预处理不一致

resize、crop、归一化、通道顺序和 dtype 任何一项不一致，都可能使 patch 内容发生系统性偏移。固定图像的 patch 向量、投影输出和第一层 hidden 是比最终准确率更早的诊断信号。

## ViT 架构审计

面对一个 ViT 配置或 checkpoint，可以按以下顺序核对：

1. 固定输入通道、图像高度宽度、通道顺序和归一化；
2. 检查 $P$ 是否整除 $H,W$，并记录裁剪或 padding；
3. 计算 $n_h,n_w,N,T$，说明是否包含 CLS；
4. 核对 patch flatten 顺序、patch projection 权重形状和 bias；
5. 核对 patch token 的二维位置到一维 index 的映射；
6. 检查位置嵌入是 learned 1D、learned 2D、sinusoidal 还是 relative；
7. 检查改变分辨率时 CLS 行、patch 网格和插值方法；
8. 核对 encoder self-attention 的 $(B,h,T,T)$ 交互轴；
9. 核对分类头读取 CLS、mean pooling 或其他 token 子集；
10. 对密集任务核对 patch 网格到目标分辨率的重建；
11. 分别计算 patch projection、attention、FFN、head 的 MAC 与激活；
12. 用非对称固定图像验证 patch 内容、位置、首层 hidden 和最终 readout。

ViT 的输入不是“把图像改名为 token”这么简单。patch 网格、位置条件、全局交互和 readout 共同规定了模型实际能访问哪些信息，以及改变分辨率后哪些参数仍然有效。

## 相关词条

[Encoder-Only](../transformer-architectures/encoder-only/)

[完整 Transformer](../transformer-architectures/full-transformer/)

[位置编码](../transformer-components/positional-encoding/)

[Self-Attention](../attention/self-attention/)

[卷积神经网络](../cnn/cnn/)

[ResNet 后续架构](../cnn/resnet-and-beyond/)

[参数量](../transformer-components/parameter-count/)
