---
title: "残差连接：让深层网络学习对输入的修正"
tags: ["why-models-learn"]
---

残差连接把一条网络分支改写成“输入加上一个修正量”。最常见的同形状残差块不是直接学习目标映射 $H(x)$，而是学习

$$
F(x)=H(x)-x,
$$

再输出

$$
y=x+F(x).
$$

这样做的关键不是多了一次加法，而是把恒等映射 $x\mapsto x$ 变成一条明确的、参数为零的路径。若新增的深层 block 暂时只需要保留已有表示，残差分支可以趋近于零；反向传播也会看到恒等项，而不必把所有信号都压过一串新的权重矩阵。

这解决的是深层**优化**中的退化问题，不是一个“层数越多越容易过拟合”的同义词。原始 ResNet 论文在 CIFAR-10 上比较了普通网络与更深的普通网络，发现更深的普通网络训练误差和测试误差都可能变差；作者把它称为 degradation problem，并用残差学习和 shortcut 让 152 层网络可以被有效训练。[原始 ResNet CVPR 论文](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) 是这里关于实验背景和原始 block 定义的主要来源。

本文固定一个简化但可复算的残差块，先区分普通深度网络的退化与梯度消失，再推导恒等路径的 Jacobian。随后处理分支形状、projection shortcut、BasicBlock、Bottleneck、post-activation 与 pre-activation，最后用参数账、数值例子和审计清单说明：残差连接改善的是参数化和信号路径，不能替代学习率、归一化、初始化和运行时证据。

![残差连接的三种形状关系：普通堆叠没有直接路径，同形状 block 通过恒等捷径相加，尺寸改变时由 1×1 projection 对齐后再相加](/assets/cnn/svg/residual-connections.1.svg)

## 先把普通深度网络的问题说准确

设一个没有 shortcut 的层级网络为

$$
h_{l+1}
=\phi_l(W_lh_l+b_l).
$$

如果在末端继续增加若干层，新的目标函数仍然可以在理论上表示旧网络：新增层学习恒等映射，后面的分类头保持不变。于是“函数类变大”本身不应让最优训练误差变差。

但“存在一组能实现恒等的参数”和“优化器容易找到这组参数”是两件事。新增层要用权重和激活的组合近似恒等，初始化、饱和、尺度和非凸优化都会影响搜索过程。实际训练中可能出现：

| 现象 | 直接证据 | 不能直接推出的结论 |
| --- | --- | --- |
| 更深的普通网络训练误差上升 | 训练集上的 loss 或 error 随深度变差 | 不是自动等于过拟合 |
| 训练集好、验证集差 | 训练/验证曲线分叉 | 这是泛化问题，需看正则化、数据和分布 |
| 早期层梯度接近零 | 按层梯度范数或 Jacobian 统计 | 可能是消失、饱和、掩码或损失路径问题 |
| 激活或梯度出现极端值 | batch 级分位数、NaN/Inf、更新比率 | 可能是爆炸、归一化、数据或数值精度问题 |

梯度消失与爆炸词条已经从标量乘积和 Jacobian 奇异值解释了跨层信号如何衰减或放大。残差连接提供一条有利的加法路径，但它不是对所有局部 Jacobian 的豁免证书。后文会把这两个层次分开。

### 恒等映射为什么是一个合理的深度目标

把浅层网络的表示记为 $h$。如果新增模块对任务没有有用的变化，理想目标是

$$
H(h)=h.
$$

普通堆叠要求一个或多个带激活的参数化层共同近似这个映射。残差参数化把目标改写为

$$
F(h)=H(h)-h=0.
$$

对新增 block 来说，学习“输出一个零修正”通常比从头拼出精确恒等映射更直接。这里的“通常”是优化假设，不是数学保证：残差分支仍可能有非零初始化、归一化统计、后置激活或形状变化。

### 退化问题不是训练集过拟合

过拟合的典型证据是训练误差继续下降而验证误差上升；退化问题的关键证据是训练误差本身随普通网络加深而上升。二者可能同时出现，但需要分别记录：

$$
\text{train error},\qquad
\text{validation error},\qquad
\lVert\nabla_{\theta_l}L\rVert_2.
$$

如果只看最终验证准确率，就无法判断是优化器没有把训练目标降下来，还是模型已经记住训练集却没有泛化。残差连接主要针对前者，同时也可能因为更易优化而改变后者；它不是独立的正则化定理。

## 残差块：学习修正而不是重写全部表示

### 同形状的基本定义

先考虑输入和输出具有相同 shape 的 block。设残差分支由若干带权重层组成：

$$
F(x,\mathcal W)
=F(x,\{W_i\}),
$$

shortcut 直接传递输入：

$$
S(x)=x.
$$

相加前的 block 输出为

$$
y=F(x,\mathcal W)+S(x)
=F(x,\mathcal W)+x.
$$

原始 ResNet 的常见 post-activation 写法还会在相加后施加非线性：

$$
H(x)=\phi\bigl(F(x,\mathcal W)+x\bigr).
$$

因此要说明“恒等路径”时必须注明讨论的是哪一层：相加前的主表达式确实有 $x$，但相加后的 ReLU 会改变负半轴。后文关于干净梯度路径的推导会显式采用不在相加后立即放置激活的简化块，或采用 pre-activation 形式，避免把两种 block 混成一个公式。

### 两条支路分别承担什么

| 部分 | 计算 | 学习到的内容 | 形状要求 |
| --- | --- | --- | --- |
| residual branch | $F(x,\mathcal W)$ | 对输入表示的局部或高阶修正 | 输出必须能与 shortcut 相加 |
| shortcut branch | $S(x)$ | 恒等传递或必要的线性投影 | 输出 shape 必须与 residual branch 相同 |
| merge | $F(x,\mathcal W)+S(x)$ | 逐元素相加，保留两条路径的信息 | batch、channel、height、width 都要对齐 |
| activation | $\phi(\text{merge})$ 或 merge 前的分支激活 | 非线性门控 | 位置由 block 变体决定 |

相加不是拼接。逐元素相加不会新增通道，也不会保留两份独立的张量轴；它要求两个分支在每个位置都有可对应的值。若把相加误写成 concatenation，后续层的通道数、参数量和表示语义都会改变。

### 零修正是一个可检查的边界状态

当残差分支输出为零时，

$$
F(x,\mathcal W)=0
\quad\Longrightarrow\quad
y=x.
$$

当残差分支很小时，可以把它看作输入上的扰动：

$$
\lVert y-x\rVert_2
=\lVert F(x,\mathcal W)\rVert_2.
$$

这给出了一个简单的运行时审计：记录 residual branch 输出相对 shortcut 输出的范数比

$$
\rho
=
\frac{\lVert F(x,\mathcal W)\rVert_2}
{\lVert x\rVert_2+\varepsilon}.
$$

$\rho$ 很小表示 block 主要保留输入，$\rho$ 很大表示修正支配了输出；两者都不自动代表好或坏。若 $\rho$ 逐层迅速变大，可能出现表示尺度和梯度风险；若长期几乎为零，可能是任务确实不需要该 block，也可能是初始化、归一化、学习率或数据路径阻止了分支学习。

## 恒等路径如何出现在反向传播中

### 简化块的 Jacobian

为了看清结构，先定义一个没有相加后激活的 block：

$$
x_{l+1}=x_l+F_l(x_l).
$$

令 $J_l$ 是残差分支对输入的 Jacobian：

$$
J_l
=\frac{\partial F_l(x_l)}{\partial x_l}.
$$

对 $x_l$ 求导得到

$$
\frac{\partial x_{l+1}}{\partial x_l}
=I+J_l.
$$

若上游梯度为列向量 $g_{l+1}=\nabla_{x_{l+1}}L$，则

$$
g_l
=\frac{\partial x_{l+1}}{\partial x_l}^{\mathsf T}
g_{l+1}
=(I+J_l)^{\mathsf T}g_{l+1}.
$$

展开 $L-l$ 个 block：

$$
\frac{\partial x_L}{\partial x_l}
=(I+J_{L-1})(I+J_{L-2})
\cdots(I+J_l).
$$

每个因子都有一个显式的 $I$。因此梯度可以沿恒等项逐 block 传递，或者在某个 block 经过 $J_l$ 进入残差分支。这个加法结构与普通堆叠的纯乘积不同：

$$
\frac{\partial h_L}{\partial h_l}
=A_{L-1}A_{L-2}\cdots A_l.
$$

残差结构改善的是“只有一条乘法路径”的参数化。它不意味着所有乘积方向都自动保持单位增益，因为 $I+J_l$ 的奇异值仍可能小于或大于 $1$。

### 两个 block 的展开

对两个简化 block：

$$
x_{l+1}=x_l+F_l(x_l),
\qquad
x_{l+2}=x_{l+1}+F_{l+1}(x_{l+1}).
$$

直接代入得到

$$
x_{l+2}
=x_l+F_l(x_l)+F_{l+1}(x_{l+1}).
$$

第一项是从输入到输出的直接恒等贡献；后两项是经过残差分支的修正。对更深的链，递推可以写成

$$
x_L
=x_l+\sum_{i=l}^{L-1}F_i(x_i).
$$

这里的 $x_i$ 仍然依赖之前的修正，所以这不是把所有分支都在同一个输入上独立计算，而是一个带递推状态的展开。它说明残差网络在状态空间中保留了基线表示，再逐步累加更新。

### 一个标量反向数值例子

设单个 block 的残差函数为

$$
F(x;\alpha,\beta)=\alpha x+\beta,
\qquad
x=3,\quad
\alpha=0.2,\quad
\beta=1.
$$

前向计算为

$$
F(x)=1.6,
\qquad
y=x+F(x)=4.6.
$$

若上游梯度为 $g_y=2$，则

$$
\frac{\partial y}{\partial x}=1+\alpha=1.2,
\qquad
\frac{\partial L}{\partial x}=g_y(1+\alpha)=2.4.
$$

同一例子中，参数梯度为

$$
\frac{\partial L}{\partial\alpha}
=g_yx=6,
\qquad
\frac{\partial L}{\partial\beta}
=g_y=2.
$$

如果没有 shortcut，输出若只写成 $y=F(x)$，对应的输入导数是 $\alpha=0.2$，而不是 $1.2$。这不是说 $1.2$ 一定更好，而是说明恒等项确实改变了局部梯度的结构。

### 恒等项不是稳定性保证

若每层都使用线性残差 $F_l(x)=\alpha x$，则单层 Jacobian 为

$$
1+\alpha.
$$

经过 $L$ 层：

$$
\frac{\partial x_L}{\partial x_0}
=(1+\alpha)^L.
$$

例如 $\alpha=0.01$、$L=20$ 时，

$$
(1.01)^{20}
\approx1.220190.
$$

若 $\alpha=-0.01$，同样深度得到

$$
(0.99)^{20}
\approx0.817907.
$$

当残差 Jacobian 的谱接近但不等于零时，$I+J_l$ 仍可能积累放大或衰减；当不同层的 Jacobian 方向旋转时，还会出现方向性放大、抵消和条件数恶化。因此实际诊断要同时看 residual ratio、梯度范数、激活统计和按层 Jacobian 的近似证据。

## 形状合同：什么时候可以直接相加

### 同形状 shortcut

对二维卷积特征图，设输入为

$$
x\in\mathbb R^{B\times C\times H\times W}.
$$

若残差分支输出也为

$$
F(x)\in\mathbb R^{B\times C\times H\times W},
$$

则可使用参数为零的恒等 shortcut：

$$
S(x)=x,
\qquad
y=F(x)+x.
$$

此时每个 batch、channel、height、width 坐标逐一相加。空间尺寸、通道数和 batch 维都不允许静默不同；广播若被框架接受，也不等于它符合残差 block 的结构合同。

### 形状变化需要 projection

如果残差分支改变了空间尺寸或通道数，直接使用 $x$ 不可行。例如：

$$
x\in\mathbb R^{B\times64\times56\times56},
\qquad
F(x)\in\mathbb R^{B\times128\times28\times28}.
$$

shortcut 可以用 stride 为 $2$ 的 $1\times1$ 卷积投影：

$$
S(x)=W_s*_{1\times1,\,s=2}x+b_s,
\qquad
S(x)\in\mathbb R^{B\times128\times28\times28}.
$$

相加才有定义：

$$
y=F(x)+S(x).
$$

投影不是“为了让公式看起来完整”的装饰，它承担了两个同时发生的坐标变换：$1\times1$ 核混合通道，stride $2$ 改变空间采样网格。若只改变通道而不改变空间，stride 应为 $1$；若只下采样而不改变通道，可使用输出通道等于输入通道的投影。

### projection 的参数账

含 bias 的 $1\times1$ projection 从 $C_{\mathrm{in}}$ 映射到 $C_{\mathrm{out}}$，参数量为

$$
P_{\mathrm{proj}}
=1^2C_{\mathrm{in}}C_{\mathrm{out}}+C_{\mathrm{out}}
=C_{\mathrm{in}}C_{\mathrm{out}}+C_{\mathrm{out}}.
$$

对 $64\to128$ 的例子：

$$
P_{\mathrm{proj}}
=64\times128+128
=8320.
$$

如果实现中的卷积后紧跟 BatchNorm，卷积 bias 常被关闭；那时应从账本中去掉 $C_{\mathrm{out}}$，而不是把两种约定的数字混在一起。

也可以用零填充 shortcut 把通道从 $64$ 扩到 $128$，再对空间做固定下采样。但它不提供学习到的跨通道线性映射，且空间下采样规则和信息保留方式必须另行定义。工程实现中使用 projection 时，应记录这是为了 shape 对齐还是为了额外表达能力。

## BasicBlock：两个小卷积组成一个修正

### 同形状 BasicBlock

经典 BasicBlock 的残差分支可以抽象成两个 $3\times3$ 卷积：

$$
F(x)
=W_2*\phi(W_1*x).
$$

若输入输出通道都是 $C$，stride 都为 $1$，含 bias 的参数量为

$$
P_{\mathrm{basic}}
=2(3^2C^2+C)
=18C^2+2C.
$$

取 $C=64$：

$$
P_{\mathrm{basic}}
=2(9\times64^2+64)
=73856.
$$

shortcut 是恒等映射时不增加权重参数。若每个卷积后有 BatchNorm，还要把每个输出通道的尺度和偏移计入另外的参数账；本文的卷积数字先固定为“含 bias、暂不计归一化参数”，便于不同 block 直接对比。

### 下采样 BasicBlock

继续使用输入 $64\times56\times56$、输出 $128\times28\times28$ 的例子。令第一层 $3\times3$ 卷积 stride 为 $2$，第二层保持输出 shape：

| 组件 | 输入通道 | 输出通道 | 核与 stride | 含 bias 参数 |
| --- | ---: | ---: | --- | ---: |
| residual conv1 | 64 | 128 | $3\times3,\ s=2$ | $3^2\times64\times128+128=73856$ |
| residual conv2 | 128 | 128 | $3\times3,\ s=1$ | $3^2\times128^2+128=147584$ |
| projection | 64 | 128 | $1\times1,\ s=2$ | $1^2\times64\times128+128=8320$ |
| 合计 | — | — | — | $229760$ |

残差 branch 和 projection branch 的空间输出都为 $28\times28$，通道输出都为 $128$，所以才能逐元素相加。若把 projection 的 stride 写成 $1$，它会得到 $56\times56\times128$；若把 residual conv1 的 stride 写成 $1$，它会得到 $56\times56\times128$，两种错误都应在 shape ledger 中被发现。

### BasicBlock 的感受野和 stride

把输入上的感受野边长记为 $r$、相邻特征点的输入步距记为 $j$。对 stride 为 $s$、核为 $k$ 的卷积：

$$
r_{\mathrm{out}}
=r_{\mathrm{in}}+(k-1)j_{\mathrm{in}},
\qquad
j_{\mathrm{out}}
=sj_{\mathrm{in}}.
$$

对同形状的两个 $3\times3$、stride 为 $1$ 的卷积，若输入 $r=1,j=1$：

$$
r_1=1+2=3,
\qquad
r_2=3+2=5,
\qquad
j_2=1.
$$

shortcut 仍然只携带输入的 $r=1$ 支持，但 merge 后的输出同时包含 identity 的局部基线和 residual branch 的较大感受野修正。不能把 block 的理论感受野简单写成 shortcut 与 residual 两者相加的尺寸；它们是逐元素相加的两种依赖来源，合并后的依赖集合是两者的并集。

## Bottleneck：用 $1\times1$ 压缩宽度

### 三层残差分支

当通道数较大时，ResNet 常用 bottleneck block：

$$
F(x)
=W_3*
\phi\left(
W_2*
\phi\left(
W_1*x
\right)
\right),
$$

其中三层卷积的通道路径是

$$
C
\longrightarrow C_b
\longrightarrow C_b
\longrightarrow C.
$$

典型地 $C_b=C/4$。第一和第三层是 $1\times1$ 通道投影，中间层是 $3\times3$ 空间卷积。对同形状、含 bias 的 block：

$$
P_{\mathrm{bottle}}
=CC_b+C_b
+9C_b^2+C_b
+C_bC+C.
$$

合并为

$$
P_{\mathrm{bottle}}
=2CC_b+9C_b^2+2C_b+C.
$$

### $C=256$ 的对照

取输出通道 $C=256$、瓶颈宽度 $C_b=64$：

| block | 逐层参数 | 总参数 |
| --- | --- | ---: |
| BasicBlock | $3\times3:256\to256$ 两次 | $2(9\times256^2+256)=1180160$ |
| Bottleneck | $1\times1:256\to64$，$3\times3:64\to64$，$1\times1:64\to256$ | $16448+36928+16640=70016$ |
| 比例 | Bottleneck / BasicBlock | $70016/1180160\approx0.0593$ |

这个参数节省来自让昂贵的 $3\times3$ 空间卷积在较窄的中间通道上运行。代价是新增两次 $1\times1$ 投影和更复杂的激活/归一化顺序；参数更少不等于延迟、内存访问或优化难度在所有设备上都更低。

若 bottleneck 的输入输出 shape 不同，仍然要对 shortcut 使用 projection。以 $256\to512$ 且空间 stride 为 $2$ 为例，projection 参数为

$$
P_{\mathrm{proj}}
=256\times512+512
=131584.
$$

不能因为 residual branch 已经有三层卷积，就把 shortcut 省掉；两条支路的 shape 合同仍然存在。

## post-activation 与 pre-activation

### post-activation：相加后再激活

原始 ResNet 论文中，一个典型 block 可以写为

$$
y=F(x,\mathcal W)+S(x),
\qquad
H(x)=\phi(y).
$$

这样做的好处是表达直观：两条支路先合并，再经过非线性。需要注意的是，$H(x)$ 对输入的完整 Jacobian 还包含激活导数：

$$
\frac{\partial H}{\partial x}
=D_\phi\left(J_F+J_S\right),
$$

其中 $D_\phi$ 是 merge 后激活的局部导数。即使 $J_S=I$，$D_\phi$ 也可能在某些坐标为零。

### pre-activation：把归一化和激活放到分支内

pre-activation 变体把归一化和非线性放到卷积之前，使主干可以抽象为

$$
x_{l+1}
=x_l+
W_{l,2}*
\phi\left(
\operatorname{Norm}_2
\left(
W_{l,1}*
\phi\left(
\operatorname{Norm}_1(x_l)
\right)
\right)
\right).
$$

在形状不变且没有额外 merge 后激活的抽象里，shortcut 的导数仍保持为 $I$。这有利于分析一条更干净的恒等路径，但不意味着 pre-activation 自动优于所有 post-activation 训练设置。归一化类型、参数初始化、学习率、数据规模和硬件实现仍会改变结果。

审阅代码或论文图时，至少记录：

| 问题 | 需要固定的答案 |
| --- | --- |
| 激活在哪里 | 卷积后、加法后，还是下一层卷积前 |
| 归一化在哪里 | 每个卷积前还是后，shortcut 是否有归一化 |
| shortcut 是否有参数 | identity、projection、零填充或其它映射 |
| 最后一个激活是否存在 | 它会不会截断所谓的恒等梯度路径 |
| block 的输出定义 | 是 $F(x)+x$ 还是 $\phi(F(x)+x)$ |

## 残差缩放、初始化与随机深度

### 显式缩放残差分支

可以给残差分支乘一个标量 $\alpha$：

$$
y=x+\alpha F(x).
$$

此时

$$
\frac{\partial y}{\partial x}
=I+\alpha J_F.
$$

当 $\alpha$ 较小，block 在初始化附近更接近恒等映射；当 $\alpha=0$，相加前的 block 完全是 identity。但如果 $\alpha$ 永远不变地很小，残差分支也可能学得过慢。缩放必须和学习率、归一化和参数初始化一起验证。

### 零初始化最后一个分支变换

一种常见思路是让残差分支最后一层的尺度参数初始化为零，使训练开始时

$$
F(x,\mathcal W_0)\approx0,
\qquad
y\approx x.
$$

这只是初始化状态，不是训练全过程的约束。归一化的 running statistics、偏置、后续激活和 projection shortcut 仍可能让整个 block 不完全等于 identity。报告时应区分“参数被零初始化”和“实际 residual ratio 在 batch 上接近零”。

### stochastic depth 的路径抽样

如果以保留概率 $q$ 随机保留残差分支，可以写成

$$
y=x+\frac{m}{q}F(x),
\qquad
m\sim\operatorname{Bernoulli}(q).
$$

在独立 mask 下，条件于输入的期望为

$$
\mathbb E_m[y\mid x]
=x+\mathbb E_m\left[\frac{m}{q}\right]F(x)
=x+F(x).
$$

但单个训练样本看到的是随机深度，梯度方差会改变；推理时通常关闭 mask，部署路径也必须和训练的缩放约定一致。它与逐坐标 Dropout 不同：随机深度按整个 residual branch 或 block 丢弃路径，而不是独立丢弃每个激活。

## 残差连接与其它结构的边界

| 结构 | 合并方式 | 直接保留输入路径 | 主要代价或风险 |
| --- | --- | --- | --- |
| plain stack | 逐层替换表示 | 没有显式 shortcut | 纯乘法 Jacobian，深度优化更难 |
| residual | $x+F(x)$ | 有 | 需要 shape 对齐，分支尺度需监控 |
| projected residual | $P(x)+F(x)$ | 有线性投影路径 | projection 增加参数、计算和可能的数值偏移 |
| dense connection | $\operatorname{concat}(x,F(x))$ | 以通道形式保留 | 通道数和内存随层增长 |
| gated residual | $x+g(x)\odot F(x)$ | 有但被门控 | 门可能饱和或关闭残差学习 |

Residual 的核心是加法和恒等基线；Dense 的核心是拼接和历史特征累积；两者都能缓解信息丢失，但参数账、内存账和后续层接口不同。看到“skip connection”这个泛称时，应继续追问它是 addition、concatenation 还是带门控的其它 merge。

### 加法与拼接的参数差别

假设输入和残差都是 $C$ 个通道，后续使用一个 $3\times3$ 卷积输出 $C$ 个通道：

| merge | merge 后通道数 | 后续卷积含 bias 参数 |
| --- | ---: | ---: |
| addition | $C$ | $3^2C^2+C$ |
| concatenation | $2C$ | $3^2(2C)C+C$ |

拼接把后续卷积的输入通道翻倍，参数也近似翻倍；加法保持宽度不变，但要求两个分支逐坐标同形状。不能只比较残差分支本身的层数而不比较 merge 后的接口。

## 一个可复算的两层残差例子

设初始标量为 $x_0=4$，两个 residual branch 分别为

$$
F_0(x)=0.1x,
\qquad
F_1(x)=-0.05x.
$$

两个 block 的前向结果：

$$
x_1=x_0+F_0(x_0)
=4+0.4
=4.4,
$$

$$
x_2=x_1+F_1(x_1)
=4.4-0.22
=4.18.
$$

局部导数分别为

$$
\frac{\partial x_1}{\partial x_0}=1.1,
\qquad
\frac{\partial x_2}{\partial x_1}=0.95.
$$

所以端到端导数是

$$
\frac{\partial x_2}{\partial x_0}
=1.1\times0.95
=1.045.
$$

这个例子同时展示两件事：恒等路径让每个因子围绕 $1$ 而不是围绕 $0$ 组织，但两个偏离 $1$ 的因子仍会相乘；残差连接减轻了深度优化压力，却没有取消深度上的尺度累积。

若把第二个分支改为 $F_1(x)=0.2x$，则

$$
x_2=4.4+0.88=5.28,
\qquad
\frac{\partial x_2}{\partial x_0}
=1.1\times1.2=1.32.
$$

因此只报告“有 residual block”不足以判断梯度是否稳定；至少要记录 block 的 Jacobian 近似或输入/输出梯度统计。

## 参数、计算和激活内存的审计

### 参数公式

对一个 $k\times k$ 卷积，从 $C_{\mathrm{in}}$ 到 $C_{\mathrm{out}}$，若启用 bias：

$$
P_{\mathrm{conv}}
=k^2C_{\mathrm{in}}C_{\mathrm{out}}+C_{\mathrm{out}}.
$$

若有 $G$ 组且通道平均分组：

$$
P_{\mathrm{group}}
=k^2
\frac{C_{\mathrm{in}}C_{\mathrm{out}}}{G}
+C_{\mathrm{out}}.
$$

一个残差 block 的总权重参数可写为

$$
P_{\mathrm{block}}
=P_F+P_S+P_{\mathrm{Norm}},
$$

其中 $P_S=0$ 表示 identity shortcut。若 shortcut 是 projection，必须把其卷积和归一化参数加进去。

### 乘加与加法

忽略边界和 bias，单个输出空间位置的卷积 MAC 数为

$$
\operatorname{MAC}_{\mathrm{conv}}
=k^2C_{\mathrm{in}}C_{\mathrm{out}}.
$$

残差 merge 还需要逐元素加法：

$$
\operatorname{AddCost}
=BCHW
$$

次加法量级。它通常比 $3\times3$ 卷积的乘加小很多，但在小通道、内存带宽受限或大量短 block 的设备上，kernel launch、读写和融合策略也会影响实际延迟。

### 激活存储

训练反向时，残差分支和 shortcut 的中间张量可能都需要保存。粗略记录：

$$
M_{\mathrm{act}}
\approx
\sum_l B C_l H_l W_l
\times
\operatorname{bytes}(\mathrm{dtype}).
$$

identity shortcut 不新增权重，但不代表不产生读写；projection shortcut 既增加参数和 MAC，也可能增加一个需要保存的中间激活。使用 checkpoint 或 recomputation 时，内存账和计算账会交换，不能只比较参数量。

## 失效模式：看起来像 residual，实际合同已坏

| 失效模式 | 具体症状 | 最小核验 |
| --- | --- | --- |
| 两支 shape 不同 | 加法报错或被意外广播 | 逐 batch、channel、height、width 打印 shape |
| 只给一支下采样 | block 首层能运行，merge 时尺寸不一致 | 记录两支每层 stride 和输出空间尺寸 |
| projection 通道数写错 | 空间相同但 channel 无法相加 | 检查 $C_{\mathrm{out}}$ 是否完全一致 |
| 把 addition 写成 concat | 后续层参数和内存突然增长 | 检查 merge 后 channel 是否翻倍 |
| 把 post-activation 当 pre-activation | 梯度路径公式与实现不符 | 标出每个 Norm、ReLU、conv、add 的顺序 |
| 忽略 projection 参数 | 参数量和 FLOPs 少算 | 单独列出 shortcut 分支 |
| 把训练误差退化当过拟合 | 只看验证曲线 | 同时记录 train error 和 validation error |
| residual branch 尺度失控 | $\rho$、激活或梯度分位数逐层增大 | 记录 $\lVert F(x)\rVert_2/(\lVert x\rVert_2+\varepsilon)$ |
| residual branch 长期为零 | loss 不改善或 block 没有贡献 | 查看最后一层权重、梯度、归一化统计 |
| 随机深度缩放错误 | train/eval 输出均值不一致 | 核对 $m/q$、推理 mask 和 checkpoint 配置 |
| BN 统计不一致 | 小 batch 或 eval 输出漂移 | 分离即时 batch statistics 与 running statistics |
| 以恒等路径替代全部诊断 | 仍出现消失、爆炸或错误预测 | 联合查看梯度、激活、shape、loss 和输出 |

其中最危险的误读是：“有 $I+J_F$ 就不会梯度消失”。当 $J_F$ 的特征值接近 $-1$ 时，某些方向可能被抵消；当它持续为正且较大时，也会爆炸。恒等路径是结构优势，不是监控替代品。

## 一套可执行的残差 block 核验协议

### 第一步：固定数学合同

对每个 block 写出：

$$
x\longmapsto F(x,\mathcal W),
\qquad
x\longmapsto S(x),
\qquad
y=F(x,\mathcal W)+S(x).
$$

同时记录：

1. 输入和两支输出的完整 shape；
2. 每个卷积的 kernel、stride、padding、dilation 和 groups；
3. activation、normalization、dropout 或 stochastic depth 的位置；
4. shortcut 是 identity、projection、零填充还是其它映射；
5. bias、BN scale/shift、dtype 和参数共享约定。

### 第二步：用小张量做前向和反向

选择可手算的输入，分别把 residual branch 和 shortcut 置零或置为 identity：

$$
F(x)=0
\quad\Longrightarrow\quad
y=S(x),
$$

$$
S(x)=0
\quad\Longrightarrow\quad
y=F(x).
$$

检查 merge 是否等于逐元素相加，而不是拼接或覆盖。对标量或小矩阵用有限差分核对输入和参数梯度；若有 post-activation，要把激活导数纳入参考答案。

### 第三步：核对形状与资源账

按层生成 shape ledger：

| 项目 | residual branch | shortcut | merge 后 |
| --- | --- | --- | --- |
| batch | $B$ | $B$ | $B$ |
| channel | $C_F$ | $C_S$ | 要求 $C_F=C_S$ |
| height | $H_F$ | $H_S$ | 要求 $H_F=H_S$ |
| width | $W_F$ | $W_S$ | 要求 $W_F=W_S$ |
| 参数 | $P_F$ | $P_S$ | $P_F+P_S+P_{\mathrm{Norm}}$ |

再计算 MAC、激活内存和 merge 加法。projection 只要存在，就不能把 shortcut 计为零成本。

### 第四步：观察训练证据

至少按 block 记录：

$$
\rho_l
=\frac{\lVert F_l(x)\rVert_2}
{\lVert S_l(x)\rVert_2+\varepsilon},
\qquad
G_l=\lVert\nabla_{x_l}L\rVert_2.
$$

对 $\rho_l$ 和 $G_l$ 报告均值、中位数、分位数、异常 batch 比例和随训练步的趋势。把训练/验证 loss、激活均值方差、NaN/Inf、学习率和更新比率放在同一时间轴上，才能判断是 block 设计、优化超参还是数据问题。

### 第五步：核对部署路径

训练态与推理态分别检查：

- normalization 使用的是即时 batch statistics 还是 running buffers；
- stochastic depth、dropout 是否关闭且缩放正确；
- projection 的 stride、padding 和 dtype 是否在导出后保持；
- fusion、量化或混合精度是否改变 merge 前后的数值范围；
- 单样本、最小 batch、全零输入和极端输入是否仍保持 shape 合同。

残差连接的正确性最终不是由一张结构图决定的，而是由公式、shape ledger、独立数值 probe、训练日志和部署输出共同决定的。

## 相关词条

- [梯度消失与爆炸](../backpropagation/vanishing-and-exploding/)
- [从 LeNet 到 VGG：卷积网络如何变深](../cnn/lenet-to-vgg/)
- [二维卷积](../cnn/convolution-2d/)
- [步幅、填充与膨胀](../cnn/stride-padding-dilation/)
- [批量归一化](../training-nn/batch-normalization/)
- [ReLU](../neurons-and-activations/relu/)
- [Dropout](../training-nn/dropout/)
- ResNet 与后续架构（后续词条）
