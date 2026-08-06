---
title: "ResNet 与后续架构：深度、宽度、基数与特征复用"
tags: ["why-models-learn"]
---

ResNet 解决了一个关键问题：让深层网络有机会沿恒等路径传递输入，并把每个 block 的学习目标改写成修正量。但残差连接不是架构设计的终点。得到稳定的加法路径之后，研究者还可以沿着不同轴继续扩展容量：

- 增加 **depth**，让网络拥有更多层级变换；
- 增加 **width**，让每个空间位置保留更多通道；
- 增加 **cardinality**，让多个并行变换共同形成一个 block；
- 增加 **connectivity**，让旧特征通过 concat 被后续层重复使用；
- 改变卷积、归一化和 stem 的组合，使卷积网络适应新的硬件和训练配方。

因此“ResNet 与后续架构”不是一张模型排行榜，而是一组结构实验：在尽量可比的参数量、计算量和分辨率下，改变一个结构轴，观察信息路径、梯度路径、激活内存和表达预算怎样变化。Wide ResNet 把一部分深度换成宽度，ResNeXt 把并行变换的数量作为 cardinality，DenseNet 用密集拼接复用历史特征，ConvNeXt 则从 ResNet 出发吸收大核深度卷积、LayerNorm、GELU 和分层下采样等现代设计。

本文先建立统一的账本，再从原始 ResNet 的 pre-activation identity mapping 推到四个后续方向。文中引用的主要边界是：[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)、[Wide Residual Networks](https://arxiv.org/abs/1605.07146)、[ResNeXt 的 CVPR 开放论文](https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html)、[DenseNet](https://arxiv.org/abs/1608.06993) 和 [ConvNeXt](https://arxiv.org/abs/2201.03545)。论文报告的 benchmark 结果属于论文当时的数据、训练配方和硬件条件；本文会把这些结果与独立的结构算术分开。

![ResNet 后续架构的四个扩展方向：恒等残差路径、并行变换的 cardinality、DenseNet 的特征拼接与 ConvNeXt 的现代卷积 block](/assets/cnn/svg/resnet-and-beyond.1.svg)

## 先固定五个比较坐标

设卷积特征图为

$$
x\in\mathbb R^{B\times C\times H\times W}.
$$

其中 $B$ 是 batch size，$C$ 是通道数，$H,W$ 是空间尺寸。一个架构改变了“深度”或“宽度”时，必须先说清楚改变的是哪一个量：

| 坐标 | 符号 | 它增加了什么 | 它的主要代价 |
| --- | --- | --- | --- |
| depth | $D$ | 串行的非线性变换次数 | 训练时间、梯度路径长度、串行延迟 |
| width | $C$ | 每个位置可保留的通道数 | 卷积参数和 MAC 常近似按 $C^2$ 增长 |
| cardinality | $G$ 或 $C_{\mathrm{ard}}$ | 并行变换分支的数量 | 分组规则、分支融合和内存访问 |
| growth rate | $k$ | Dense block 每层新增的通道数 | concat 后的通道与激活内存增长 |
| connectivity | — | 旧特征能被多少后续层直接读取 | 加法保持宽度，拼接增加接口宽度 |

这里用 $G$ 表示 cardinality 时，要和分组卷积里的 group 数区分语义；在 ResNeXt 的典型实现中两者对应，但“有 $G$ 组”不自动意味着它就是 ResNeXt。

### 单个卷积的资源账

忽略空间边界和 bias，一个 $k\times k$ 卷积从 $C_{\mathrm{in}}$ 到 $C_{\mathrm{out}}$ 的参数量和每个输出位置的 MAC 数分别为

$$
P_{\mathrm{conv}}
=k^2C_{\mathrm{in}}C_{\mathrm{out}},
\qquad
\operatorname{MAC}_{\mathrm{pos}}
=k^2C_{\mathrm{in}}C_{\mathrm{out}}.
$$

若有 $G$ 组且输入输出通道平均分组：

$$
P_{\mathrm{group}}
=k^2
\frac{C_{\mathrm{in}}C_{\mathrm{out}}}{G}.
$$

输出空间共有 $HW$ 个位置时，粗略 MAC 为

$$
\operatorname{MAC}_{\mathrm{layer}}
=HWk^2C_{\mathrm{in}}C_{\mathrm{out}}.
$$

这个账本不包括 activation、kernel launch、内存带宽、归一化和数据搬运，所以它只能作为结构比较的第一层证据。相同参数量不代表相同延迟，相同 FLOPs 也不代表相同训练显存。

### 深度与宽度的二次关系

若一串同形状的 $3\times3$ 卷积都使用通道数 $C$，每层参数近似为 $9C^2$。将通道扩大 $a$ 倍、层数缩小 $b$ 倍后，卷积参数相对量近似为

$$
\frac{P_{\mathrm{new}}}{P_{\mathrm{old}}}
\approx
\frac{a^2}{b}.
$$

例如，把 $C=64$ 的四个同形状卷积换成 $C=128$ 的一个 block：

$$
\frac{2^2}{4}
=1.
$$

这条计算不能直接拿来宣称“宽网络更省参数”，因为 block 内卷积数、stage 的空间尺寸、projection、归一化和 head 都可能不同。它只说明宽度对二次卷积项很敏感，而深度对同样 block 的重复次数近似线性。

## ResNet 原始路线：先把恒等路径做干净

### post-activation 的问题边界

普通残差 block 可以抽象成

$$
x_{l+1}
=\phi\left(x_l+F_l(x_l,\mathcal W_l)\right).
$$

在相加后使用 $\phi$ 时，shortcut 的数值路径虽然经过了加法，却仍会受到 merge 后激活导数的影响：

$$
\frac{\partial x_{l+1}}{\partial x_l}
=D_l\left(I+J_l\right),
$$

其中 $D_l$ 是激活的局部 Jacobian，$J_l$ 是 residual branch 的 Jacobian。当 ReLU 的某个坐标落在负半轴，$D_l$ 的对应元素可以为零，所谓“恒等梯度”就不再是完整的恒等映射。

### pre-activation 的递推

Identity Mappings 论文把归一化和激活移到 residual branch 内，抽象成

$$
x_{l+1}
=x_l+F_l\left(\operatorname{Norm}_l(x_l)\right).
$$

令

$$
\widetilde J_l
=
\frac{\partial F_l(\operatorname{Norm}_l(x_l))}
{\partial x_l}.
$$

则相加前的局部导数为

$$
\frac{\partial x_{l+1}}{\partial x_l}
=I+\widetilde J_l.
$$

前向递推可以展开为

$$
x_L
=x_l+\sum_{i=l}^{L-1}
F_i\left(\operatorname{Norm}_i(x_i)\right).
$$

反向传播则满足

$$
\frac{\partial L}{\partial x_l}
=
\frac{\partial L}{\partial x_L}
\left(
I+\sum_{i=l}^{L-1}
\frac{\partial F_i}{\partial x_l}
\right),
$$

这里的 $\partial F_i/\partial x_l$ 表示通过中间状态展开后的总导数。第一项对应从 $x_l$ 到 $x_L$ 的直接恒等贡献，后面的和代表经过不同 residual branch 的修正路径。它不是说残差分支之间彼此独立，而是说明 identity path 在 forward 和 backward 都有明确的代数位置。

### 一个线性小例子

令三层 pre-activation residual branch 都是

$$
F_i(x)=\alpha x,
\qquad
\alpha=0.02.
$$

每层导数为

$$
\frac{\partial x_{i+1}}{\partial x_i}=1.02.
$$

三层后的前向增益与反向增益相同：

$$
\frac{\partial x_3}{\partial x_0}
=1.02^3
=1.061208.
$$

如果没有 shortcut，三个纯变换层的增益是

$$
0.02^3
=0.000008.
$$

这不是一个训练性能定理，而是一个局部结构例子：把每层的目标从“产生整个输出”改为“产生小修正”，会把局部导数从 $0.02$ 平移到 $1.02$。当 $\alpha$、激活和归一化的真实 Jacobian 不是标量时，仍须回到按层统计。

### projection 是 stage 边界，不是免费 identity

当输入和输出 shape 不同时，shortcut 写成

$$
S_l(x_l)
=W_{s,l}*_{1\times1,\,s_l}x_l.
$$

block 变为

$$
x_{l+1}
=F_l(x_l)+S_l(x_l).
$$

projection 的局部导数是 $J_{S_l}$，所以

$$
\frac{\partial x_{l+1}}{\partial x_l}
=J_l+J_{S_l}.
$$

它不再包含尺寸不变的 $I$，但仍保留了一个较短的线性 shortcut。审计 stage transition 时要单独记录 projection 的 kernel、stride、输入输出通道和归一化；不能用“这是 ResNet”替代 shape ledger。

## ResNet 的 stage 账：容量在哪里增长

一个常见的 ImageNet ResNet-50 风格配置以 $224\times224$ 输入为例。不同实现可能在 stride 放置和 stem 细节上有差异，下面只固定常见的 stage 级接口：

| 阶段 | 空间尺寸 | bottleneck 内部宽度 | block 输出通道 | block 数量 |
| --- | ---: | ---: | ---: | ---: |
| stem | $56\times56$ | — | 64 | — |
| stage 1 | $56\times56$ | 64 | 256 | 3 |
| stage 2 | $28\times28$ | 128 | 512 | 4 |
| stage 3 | $14\times14$ | 256 | 1024 | 6 |
| stage 4 | $7\times7$ | 512 | 2048 | 3 |
| head | $1\times1$ | — | 1000 logits | — |

bottleneck 的通道路径是

$$
C_{\mathrm{out}}
\longrightarrow
\frac{C_{\mathrm{out}}}{4}
\longrightarrow
\frac{C_{\mathrm{out}}}{4}
\longrightarrow
C_{\mathrm{out}}.
$$

例如 stage 3 的内部宽度为 $256$、输出通道为 $1024$。这样做让 $3\times3$ 卷积不必直接在 1024 个通道上工作；代价是前后各有一次 $1\times1$ 投影。

### stage 转换会同时改变三件事

当空间边长减半、输出通道翻倍或翻四倍时：

1. 每层激活位置数减少；
2. 每个位置可保存的通道容量增加；
3. 后续卷积在原输入上的感受野步距增大。

若某 stage 的 activation 元素量近似为

$$
M=BCHW,
$$

把 $H,W$ 都减半、把 $C$ 翻倍后：

$$
\frac{M_{\mathrm{new}}}{M_{\mathrm{old}}}
=\frac{2C}{C}\times\frac{H}{2H}\times\frac{W}{2W}
=\frac12.
$$

因此常见 stage 设计让深层表示通道更宽而空间激活更少。若只记录参数量而不记录 $BCHW$，就会漏掉训练显存和特征图带宽。

### block 数量不是有效深度的全部

一个 bottleneck block 拥有三次卷积变换，BasicBlock 拥有两次；projection、stem、head 和可能的 downsample 也拥有可学习参数。比较“50 层”和“101 层”时，应固定：

| 账本 | 需要记录 |
| --- | --- |
| weighted depth | 哪些卷积、全连接和 projection 计入层数 |
| nonlinear depth | 实际经过多少次激活 |
| spatial depth | 每次下采样发生在哪个 block |
| parameter depth | shortcut 和 normalization 是否计入参数 |

历史论文的层数命名有自己的计数约定，不能把不同实现的字符串深度直接当成相同的计算图。

## Wide ResNet：用宽度换掉无效深度

### width factor 的定义

设窄版 residual block 的通道为 $C$，Wide ResNet 用宽度因子 $k$ 变为

$$
C_{\mathrm{wide}}=kC.
$$

若 block 使用两个同形状 $3\times3$ 卷积，忽略 bias：

$$
P_{\mathrm{thin}}
=2(9C^2),
\qquad
P_{\mathrm{wide}}
=2(9(kC)^2).
$$

所以同一个 block 的参数比为

$$
\frac{P_{\mathrm{wide}}}{P_{\mathrm{thin}}}
=k^2.
$$

$k=2$ 时每个同形状 block 的卷积参数变为 $4$ 倍；如果同时把 block 数减半，总卷积参数约为原来的 $2$ 倍，而串行深度减半。

### 四个窄 block 与两个宽 block

取 $C=64$、$k=2$：

$$
P_{\mathrm{four\ thin}}
=4\left(2\times9\times64^2\right)
=294912,
$$

$$
P_{\mathrm{two\ wide}}
=2\left(2\times9\times128^2\right)
=589824.
$$

这个例子没有构造“同预算”的宽窄网络，而是故意把账算出来：减半层数并不会自动抵消宽度的二次代价。要做公平比较，还要同步调整 block 数、内部 bottleneck、输入分辨率和 head。Wide ResNet 论文的核心不是一个普适的 $k=2$ 公式，而是用实验显示在特定 benchmark 和训练配方下，较浅但较宽的残差网络可以比极深窄网络更有效率。

### width 提升了什么，没提升什么

| 变化 | 可能收益 | 可能代价 |
| --- | --- | --- |
| 每个位置通道更多 | 局部特征并行容量更大 | 参数、MAC、激活读写增大 |
| block 数较少 | 串行训练和推理更短 | 层级组合次数减少 |
| residual branch 更宽 | 修正空间更大 | branch ratio 可能更难控制 |
| 使用 block 内 dropout | 提供显式随机正则 | mask、缩放和 eval 路径需核对 |

“宽比深好”只能是给定数据、预算和优化配置下的结论。若输入分辨率、batch size 或硬件改变，宽度带来的 memory-bound 代价可能反转比较结果。

## ResNeXt：把 cardinality 作为第三个容量轴

### 从一条变换到多条同构变换

ResNet 的 residual branch 是一个变换 $F(x)$。ResNeXt 把它改写成多个同拓扑变换的和：

$$
F(x)
=\sum_{g=1}^{G}T_g(x).
$$

其中 $G$ 是 cardinality，$T_g$ 是第 $g$ 个分支。merge 仍然是 addition，不是把分支结果沿 channel 轴 concat：

$$
y=x+\sum_{g=1}^{G}T_g(x).
$$

如果每个分支只读取输入的一部分通道，工程上通常用 grouped convolution 实现。group 数、每组宽度和总中间通道必须同时写在配置里。

### grouped bottleneck 的参数

设输入输出通道都为 $C_o$，中间总通道为 $Gd$。一个无 bias 的 ResNeXt bottleneck 参数量为

$$
P_{\mathrm{resnext}}
=C_o(Gd)
+9Gd^2
+(Gd)C_o.
$$

前后两个 $1\times1$ 卷积混合输入与总中间通道，中间 $3\times3$ grouped convolution 每组只处理 $d$ 个输入和 $d$ 个输出通道。若固定总中间宽度 $M=Gd$，则

$$
P_{\mathrm{resnext}}
=2C_oM+\frac{9M^2}{G}.
$$

在固定 $M$ 时，增加 $G$ 会减少中间 grouped convolution 的参数项；但分支数、内存访问和硬件利用率也会改变。

### 32×4d 与普通 bottleneck 的数值比较

取 $C_o=256$，比较一个中间宽度为 $64$ 的普通 bottleneck 和一个 $G=32,d=4$ 的 ResNeXt bottleneck：

| block | $1\times1$ 输入 | $3\times3$ 中间 | $1\times1$ 输出 | 总参数 |
| --- | ---: | ---: | ---: | ---: |
| ResNet bottleneck, $d=64$ | $256\times64=16384$ | $9\times64^2=36864$ | $64\times256=16384$ | $69632$ |
| ResNeXt $32\times4d$ | $256\times128=32768$ | $9\times32\times4^2=4608$ | $128\times256=32768$ | $70144$ |

两者参数量接近，但它们的中间结构不同：普通 bottleneck 用一条宽度 $64$ 的空间变换，ResNeXt 用 $32$ 条每组宽度 $4$ 的空间变换。这个对照说明 cardinality 并不是简单地把所有通道变宽；它重新分配了空间卷积的并行结构。

若误把 $32$ 个分支 concat 后再送入输出卷积，输出接口会从 $128$ 变成更大的通道张量，参数账和语义都不再是 ResNeXt 的 aggregated transform。审阅实现时要确认 grouped convolution 的权重形状和 residual merge 的位置。

### cardinality 的边界

增加 cardinality 可能带来更多相对独立的变换，但它不是免费正交基，也不保证每个分支学到互补特征。需要同时观察：

$$
\operatorname{MAC}_{\mathrm{group}}
=HW\cdot9Gd^2,
$$

以及每组输出的方差、跨组相关性和最终 merge 后的 residual ratio。若各组高度冗余，增大 $G$ 只增加了实现复杂度；若每组过窄，硬件 kernel 可能无法高效利用。

## DenseNet：用 concat 复用历史特征

### Dense connectivity 的递推

DenseNet 的第 $l$ 层不只读取前一层，而是读取所有先前 feature maps 的拼接：

$$
x_l
=H_l\left(
\left[x_0,x_1,\ldots,x_{l-1}\right]
\right).
$$

若初始通道数为 $C_0$、每层新增 $k$ 个通道，则第 $l$ 层的输入通道为

$$
C_{\mathrm{in},l}
=C_0+lk,
$$

输出后为

$$
C_{\mathrm{out},l}
=C_0+(l+1)k.
$$

$k$ 被称为 growth rate。它控制每层新增多少新信息，而不是当前 feature map 的总宽度。

### 三层 Dense block 的参数账

忽略 bias、使用三个普通 $3\times3$ 卷积，取

$$
C_0=64,
\qquad
k=32.
$$

三层的输入通道依次为 $64,96,128$，参数总量为

$$
P_{\mathrm{dense},3}
=9\times32(64+96+128)
=82944.
$$

三层同样的 $C=64$、输出仍为 $64$ 的 residual branch 参数为

$$
P_{\mathrm{residual},3}
=3\times9\times64^2
=110592.
$$

这不是严格公平的模型比较：Dense block 三层结束后的总通道是

$$
C_3=64+3\times32=160,
$$

而 residual branch 仍输出 $64$。DenseNet 用更小的每层新增宽度换取了历史特征的直接可见性，后续 transition layer 再负责压缩接口。

### concat 的内存账

如果 block 内所有历史 feature map 都保留，空间激活元素数近似为

$$
M_{\mathrm{dense}}
=BHW\sum_{l=0}^{L}
(C_0+lk).
$$

求和得到

$$
M_{\mathrm{dense}}
=BHW
\left(
(L+1)C_0+
k\frac{L(L+1)}{2}
\right).
$$

对 $C_0=64,k=32,L=3$：

$$
\sum_{l=0}^{3}(64+32l)
=64+96+128+160
=448.
$$

因此最终接口只有 $160$ 个通道，但 block 内部可能需要读取和保存合计 $448$ 个通道的历史表示。参数量较小不能直接推出训练显存较小。

### transition layer：控制增长而不是取消复用

一个 transition layer 可以用压缩系数 $\theta$ 和平均池化下采样：

$$
C_{\mathrm{out}}
=\left\lfloor\theta C_{\mathrm{in}}\right\rfloor,
\qquad
H_{\mathrm{out}}
=\left\lfloor\frac{H_{\mathrm{in}}}{2}\right\rfloor,
\qquad
W_{\mathrm{out}}
=\left\lfloor\frac{W_{\mathrm{in}}}{2}\right\rfloor.
$$

若 $C_{\mathrm{in}}=160,\theta=0.5$：

$$
C_{\mathrm{out}}
=\lfloor0.5\times160\rfloor
=80.
$$

压缩发生在 block 之间，不改变 block 内每层读取历史特征的递推；它只是把越来越宽的表示重新投影到下一 stage 的预算。

## 从残差到现代卷积：ConvNeXt 的设计拆解

### 把 ResNet 当作可改写的基线

ConvNeXt 论文不是凭空提出一个新 merge，而是从标准 ResNet 出发，逐步引入与现代视觉 Transformer 相近的设计，同时保持纯卷积计算图。一个简化的 ConvNeXt block 可以写成

$$
x_{l+1}
=x_l+
\operatorname{DropPath}\left(
\gamma\,
W_2\,
\operatorname{GELU}\left(
W_1\,
\operatorname{LN}\left(
\operatorname{DWConv}_{7\times7}(x_l)
\right)
\right)
\right).
$$

这里的 $W_1,W_2$ 是逐点通道投影，通常把通道先扩到 $4C$ 再还原；$\operatorname{DWConv}$ 每个通道独立使用空间核，$\gamma$ 是逐通道的 layer scale。这个公式把“残差路径”与“分支内部使用什么算子”分开了：外部仍是 $x+\text{update}$，内部可以换成深度卷积和 pointwise MLP。

### ConvNeXt block 的参数账

取 $C=96$、扩展比为 $4$，忽略 bias 和 normalization 的可学习参数：

| 部件 | 参数量 |
| --- | ---: |
| $7\times7$ depthwise convolution | $7^2\times96=4704$ |
| $1\times1$ expansion | $96\times(4\times96)=36864$ |
| $1\times1$ projection | $(4\times96)\times96=36864$ |
| 合计 | $78432$ |

同一通道数下，两个普通 $3\times3$ 卷积的空间卷积参数是

$$
P_{\mathrm{basic},96}
=2\times9\times96^2
=165888.
$$

ConvNeXt 的这个 block 用一次大核 depthwise convolution 和两次 pointwise projection 替代两次全通道空间卷积；参数较少，但 pointwise 部件仍然承担主要通道混合，真实延迟取决于硬件和 kernel 融合。

### 分层 stem 与 stage

一个 ConvNeXt-Tiny 风格的 stage 级 shape 可以写成：

| 阶段 | 入口空间 | 通道 | block 数 |
| --- | ---: | ---: | ---: |
| patchify stem | $224\times224$ | 96 | — |
| stage 1 | $56\times56$ | 96 | 3 |
| stage 2 | $28\times28$ | 192 | 3 |
| stage 3 | $14\times14$ | 384 | 9 |
| stage 4 | $7\times7$ | 768 | 3 |

stem 用 $4\times4$、stride 为 $4$ 的卷积把输入一次性映射到 $56\times56$；stage 之间用下采样层把空间边长减半、通道数翻倍。论文中的具体模型、训练配方和结果应以原文为准；这里的表只用于复算形状和容量轴。

### 现代化不是“把卷积换成 Transformer”

ConvNeXt block 仍然是局部卷积、逐点线性映射、归一化、激活和残差加法。它借鉴的是设计选择，而不是把 attention 作为隐含算子塞进卷积：

| 设计选择 | 在 block 中的作用 | 它改变的账本 |
| --- | --- | --- |
| $7\times7$ depthwise convolution | 扩大单层空间窗口，成本只按通道线性增长 | 感受野、空间 MAC、内存访问 |
| LayerNorm | 提供通道/特征维归一化合同 | 统计轴、训练/推理行为 |
| GELU | 提供平滑非线性 | 局部导数和数值范围 |
| $4C$ pointwise expansion | 增加通道内变换容量 | 参数和通道 MAC |
| layer scale $\gamma$ | 让更新分支初始较小 | residual ratio、优化稳定性 |
| stochastic depth | 训练时抽样 block 路径 | 梯度方差和 train/eval 合同 |

因此比较 ResNet 与 ConvNeXt 时，应问“哪些设计轴被替换，哪些保持不变”，而不是只比较模型名称。

## 四种架构的统一对照

| 家族 | 主连接 | 新增容量轴 | 结构重点 |
| --- | --- | --- | --- |
| ResNet | addition | depth | identity shortcut；projection 需对齐 |
| Wide ResNet | addition | width | 更宽的普通卷积 block；激活预算上升 |
| ResNeXt | addition + sum | cardinality | grouped bottleneck；核对每组宽度 |
| DenseNet | concat | connectivity + growth | 历史 feature map 复用；内存增长 |
| ConvNeXt | addition | operator/stem | depthwise $7\times7$、pointwise 与 Norm |

它们共享一个事实：短路径可以改善信息传播，但短路径的**合并方式**决定了下游接口。addition 保持通道宽度，concat 增长通道宽度，grouped branch 保持总接口但改变中间连接，depthwise convolution 保持通道独立直到 pointwise mixing。

### 在相同输出 shape 下比较 merge

假设输入和更新分支都有 $C$ 个通道，后续使用 $3\times3$ 卷积输出 $C$ 个通道：

$$
P_{\mathrm{add\ interface}}
=9C^2,
$$

$$
P_{\mathrm{concat\ interface}}
=9(2C)C
=18C^2.
$$

concat 让后续层的输入通道翻倍，所以即使前一个 Dense layer 没有大参数，后续接口也要为历史特征付费。addition 则把两条路径压到同一坐标，无法保留两份独立通道语义。

## 一个可复算的容量轴实验

设基线使用 $C=64$ 的两个 $3\times3$ 卷积 block，空间尺寸固定为 $H\times W$，忽略 bias：

$$
P_{\mathrm{base}}
=2\times9\times64^2
=73728,
$$

$$
\operatorname{MAC}_{\mathrm{base}}
=HW\times73728.
$$

分别改变一个轴：

| 变体 | 结构变化 | 参数/激活的可预期变化 |
| --- | --- | --- |
| deeper | block 数从 1 变 2 | 参数和串行路径约翻倍 |
| wider | 通道从 64 变 128 | 同形状空间卷积参数约变 4 倍 |
| cardinality | 设总中间宽度固定、增加 grouped branch 数 | 中间 grouped conv 参数约按 $1/G$ 降低 |
| dense | 每层新增 $k=32$ 通道并 concat | 每层输入通道递增，历史激活需要保留 |
| ConvNeXt-like | depthwise $7\times7$ + $4C$ pointwise | 空间核参数线性，通道投影参数仍按 $C^2$ |

这个实验的目的不是寻找“最优家族”，而是避免把参数、路径长度、表示宽度和历史复用混成一个“模型更大”的标量。若要比较准确率，必须同时固定数据增强、训练步数、优化器、学习率、batch size、输入分辨率和预训练状态。

## 连接方式的梯度边界

### addition 的梯度

对于

$$
y=x+F(x),
$$

有

$$
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial y}
\left(I+J_F\right).
$$

恒等项和 residual Jacobian 相加后共同决定每个方向的增益。若 $J_F$ 的特征值接近 $-1$，可能发生抵消；若长期为正且很大，仍可能爆炸。

### concat 的梯度

对于

$$
y=[x,z],
\qquad
z=F(x),
$$

下游梯度可以按拼接坐标拆开：

$$
\frac{\partial L}{\partial x}
=g_x^{\mathrm{direct}}
+J_F^{\mathsf T}g_z.
$$

这里也存在一条来自 $x$ 的直接路径，但它不是逐元素 addition 的恒等 Jacobian，而是把梯度放在 concat 对应的输入坐标，再叠加经过 $F$ 的回传。后续层若先用 $1\times1$ 卷积混合 concat 通道，梯度还会经过该投影。

### grouped branch 的梯度

若

$$
F(x)=\sum_{g=1}^{G}T_g(x),
$$

则

$$
\frac{\partial L}{\partial x}
=\sum_{g=1}^{G}
J_{T_g}^{\mathsf T}g_F.
$$

多个 branch 的梯度在输入处相加。它们不提供一个额外的 identity 项，identity 仍来自 outer shortcut；这就是为什么 cardinality 和 residual shortcut 是两个不同层次的设计。

## 常见误读与失效模式

| 误读 | 错误边界与最小证据 |
| --- | --- |
| 深度越大表达能力一定更强且训练一定更好 | 函数类、优化、预算和泛化不同；同时看 train/validation loss 与梯度 |
| 宽度翻倍只让参数翻倍 | 全通道卷积按 $C^2$ 增长；用 $k^2C_{\mathrm{in}}C_{\mathrm{out}}$ 复算 |
| ResNeXt 的 cardinality 就是 concat 分支数 | ResNeXt 在 branch 内求和；检查 grouped conv 后是否 addition |
| DenseNet 的 growth rate 是总通道数 | $k$ 是每层新增通道；记录 $C_l=C_0+lk$ |
| 短路径天然等于 identity | projection、Norm、activation 和 concat 都会改变路径；写出 $S(x)$ |
| 参数少就一定更快、更省显存 | depthwise、concat 和内存访问可能成为瓶颈；同时记录 MAC 与 activation |
| ConvNeXt 只是把 3×3 改成 7×7 | stem、Norm、激活、pointwise expansion 和 layer scale 也改变；核对完整顺序 |
| 论文 benchmark 数字可跨训练配方比较 | 数据、增强、epoch、分辨率和预训练会改变结果；固定实验协议 |
| stage 的 stride 放哪里都一样 | stride 改变 sampling phase、感受野 jump 和 branch shape；核对 shape ledger |

尤其要避免把 DenseNet 的“短路径很多”和 ResNet 的“有恒等加法路径”说成同一机制。二者都能让早期特征更容易到达后层，但一个通过 concat 保留坐标，另一个通过 addition 在同一坐标上合并。

## 一套跨家族审计协议

### 1. 固定输入与接口

对每个 stage 写出

$$
(B,C_{\mathrm{in}},H_{\mathrm{in}},W_{\mathrm{in}})
\longmapsto
(B,C_{\mathrm{out}},H_{\mathrm{out}},W_{\mathrm{out}}).
$$

同时记录 stem、downsample、projection、groups、growth rate 和 stage block 数。对 DenseNet 还要记录 concat 后的历史通道，对 ResNeXt 还要记录 $G$ 与 $d$。

### 2. 展开真实计算图

不要只写“残差块”或“Dense block”，而要把顺序写出来：

$$
x
\longmapsto
\operatorname{Norm}
\longmapsto
\operatorname{Act}
\longmapsto
\operatorname{Conv}
\longmapsto
\operatorname{Merge}.
$$

若是 ResNeXt，补上 grouped branch；若是 DenseNet，补上 concat；若是 ConvNeXt，补上 depthwise、LayerNorm、pointwise expansion、GELU、layer scale 和 DropPath。

### 3. 复算参数与 MAC

对每个带权重层使用

$$
P=k^2C_{\mathrm{in}}C_{\mathrm{out}},
\qquad
\operatorname{MAC}=HW P.
$$

对 grouped convolution 把中间通道拆成 $G$ 组；对 depthwise convolution 使用 $C_{\mathrm{in}}=C_{\mathrm{out}}=C$ 且每组一个空间核；对 Dense concat 用当前累计通道而不是初始通道。

### 4. 复算 activation memory

训练时至少估算

$$
M_{\mathrm{train}}
\approx
\sum_{\text{saved tensors}}
BCHW\cdot
\operatorname{bytes}(\mathrm{dtype}).
$$

Dense connectivity 重点查历史 feature map 是否被保存；ResNeXt 重点查 branch 中间结果是否并行驻留；ConvNeXt 重点查 NCHW 与 NHWC 变换是否产生额外 copy。

### 5. 做最小数值 probe

用小张量分别验证：

$$
F(x)=0
\quad\Longrightarrow\quad
y=S(x),
$$

$$
\operatorname{cardinality}=1
\quad\Longrightarrow\quad
\text{grouped branch 退化为单组变换},
$$

$$
k=0
\quad\Longrightarrow\quad
\text{Dense block 不新增通道但接口仍需明确定义}.
$$

再用中心差分核对输入、projection、grouped kernel、depthwise kernel 和 pointwise projection 的梯度。结构图通过不代表实现的 weight layout、group 索引和 concat 顺序正确。

### 6. 分层记录训练与部署证据

按 stage 记录：

| 证据 | 需要回答的问题 |
| --- | --- |
| residual ratio | 更新分支是否长期压过或完全消失 |
| activation statistics | width、concat 或 pointwise expansion 是否改变尺度 |
| gradient quantiles | 哪个 stage 先出现消失或爆炸 |
| train/eval difference | Norm、DropPath 和 running statistics 是否一致 |
| latency and memory | 参数/MAC 账是否与实际设备行为一致 |
| output parity | 导出、量化和混合精度是否改变 merge 后输出 |

架构选择的结论应写成“在明确预算、数据和设备下，某个轴的证据如何变化”，而不是“某个网络名字更先进”。

## 相关词条

- [残差连接：让深层网络学习对输入的修正](../cnn/residual-connections/)
- [从 LeNet 到 VGG：卷积网络如何变深](../cnn/lenet-to-vgg/)
- [梯度消失与爆炸](../backpropagation/vanishing-and-exploding/)
- [二维卷积](../cnn/convolution-2d/)
- [分组卷积与深度卷积](../cnn/convolution-2d/)
- [批量归一化](../training-nn/batch-normalization/)
- [Dropout](../training-nn/dropout/)
- [池化](../cnn/pooling/)
