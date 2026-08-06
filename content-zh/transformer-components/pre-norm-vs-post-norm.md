---
title: "Pre-norm 与 Post-norm：残差相加前后的归一化顺序"
tags: ["why-models-learn"]
---

Pre-norm 和 Post-norm 描述归一化算子在残差子层两侧的位置。设输入为 $x$、残差分支为 $F$、归一化算子为 $N$：pre-norm 先计算 $F(N(x))$ 再与 $x$ 相加，post-norm 先计算 $x+F(x)$ 再对合并结果归一化。两种结构使用同一个 $N$ 时，forward 的调用顺序、局部 Jacobian、深层信号路径和最终输出尺度仍然不同。

这篇词条把 $N$ 抽象为 LayerNorm 或 RMSNorm，先固定两个 block 的计算图，再推导

$$
J_{\mathrm{pre}}=I+J_FJ_N,
\qquad
J_{\mathrm{post}}=J_N(I+J_F).
$$

随后用 $x=(1,2,3,4)$、LayerNorm 和线性分支 $F(x)=0.5x$ 核对一层的数值与局部特征值，再处理深层堆叠、attention/FFN、final norm、投影 shortcut、混合精度和审计协议。LayerNorm 与 RMSNorm 的单独公式分别见 [LayerNorm](../transformer-components/layernorm/) 和 [RMSNorm](../transformer-components/rmsnorm/)。

![pre-norm 让恒等路径直接进入残差相加，post-norm 在相加后让恒等路径经过归一化](/assets/transformer-components/svg/pre-norm-vs-post-norm.1.svg)

## 先固定残差 block 的两种计算图

### 一个子层的共同符号

设

$$
x\in\mathbb R^d
$$

是一个 token 的特征向量，$F:\mathbb R^d\to\mathbb R^d$ 是注意力或 FFN 子层，$N:\mathbb R^d\to\mathbb R^d$ 是保持 shape 的归一化算子。残差加法要求 $F(x)$ 与 $x$ 的 shape 相同；如果分支宽度不同，必须显式加入投影 shortcut，后文单独处理。

归一化可以是 LayerNorm：

$$
N(x)
=\gamma\odot
\frac{x-\mu\mathbf1}{\sqrt{v+\epsilon}}
+\beta,
$$

也可以是规范 RMSNorm：

$$
N(x)
=\gamma\odot
\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}.
$$

pre-norm 和 post-norm 只决定 $N$ 的位置，不决定 $N$ 采用哪一条公式。

### Pre-norm：归一化只进入分支

pre-norm 的单子层 block 为

$$
y_{\mathrm{pre}}
=x+F(N(x)).
$$

它有两条 forward 路径：

1. 主路径把 $x$ 直接送到加法节点；
2. 分支路径先计算 $N(x)$，再计算 $F$。

归一化后的表示只进入 $F$。加法后的 $y_{\mathrm{pre}}$ 不会自动重新归一化；如果下一层需要归一化，由下一个子层的 $N$ 或 final norm 负责。

### Post-norm：相加结果再归一化

post-norm 的单子层 block 为

$$
y_{\mathrm{post}}
=N\bigl(x+F(x)\bigr).
$$

它先计算未经归一化的分支输出和 shortcut 之和，再把合并向量送入 $N$。因此 shortcut 在输出和反向路径中都位于归一化之前。

### 顺序对照

| 结构 | 分支输入 | 加法节点 | block 输出 |
| --- | --- | --- | --- |
| pre-norm | $N(x)$ | $x+F(N(x))$ | 未必满足 $N$ 的统计约束 |
| post-norm | $x$ | $x+F(x)$ | $N(x+F(x))$ |

“都使用 LayerNorm”只说明归一化函数相同，不能据此推断 block 输出的均值、尺度或梯度路径相同。

## 两个子层的 Transformer 形式

### Pre-norm block

一个含 attention 和 FFN 的 pre-norm block 可以写成

$$
\begin{aligned}
x_1&=x+F_{\mathrm{attn}}(N_1(x)),\\
x_2&=x_1+F_{\mathrm{ffn}}(N_2(x_1)).
\end{aligned}
$$

第二个归一化 $N_2$ 看到的是第一个残差相加后的 $x_1$，而不是原始 $x$。两个归一化实例通常有独立的 $\gamma,\beta$ 或 $\gamma$ 参数。

### Post-norm block

对应的 post-norm block 为

$$
\begin{aligned}
x_1&=N_1\bigl(x+F_{\mathrm{attn}}(x)\bigr),\\
x_2&=N_2\bigl(x_1+F_{\mathrm{ffn}}(x_1)\bigr).
\end{aligned}
$$

每次相加后都执行归一化。于是第二个子层接收的 $x_1$ 已经经过 $N_1$；第一个 shortcut 也已经被 $N_1$ 处理。

### Q、K、V 的输入不同

若 attention 内部把 $N_1(x)$ 作为 hidden 输入，则 pre-norm 中

$$
\begin{aligned}
Q&=N_1(x)W_Q,\\
K&=N_1(x)W_K,\\
V&=N_1(x)W_V.
\end{aligned}
$$

post-norm 中同一位置通常是

$$
\begin{aligned}
Q&=xW_Q,\\
K&=xW_K,\\
V&=xW_V,
\end{aligned}
$$

而归一化发生在 attention 输出与 $x$ 相加之后。mask、QKV 投影和残差加法仍然是独立合同；把 post-norm 的归一化提前到 QKV 输入，就已经改成了 pre-norm 或另一种混合结构。

## Jacobian：先写链式法则

### Pre-norm 的局部导数

设 $J_F$ 是 $F$ 在 $N(x)$ 处的 Jacobian，$J_N$ 是 $N$ 在 $x$ 处的 Jacobian。对

$$
y_{\mathrm{pre}}=x+F(N(x))
$$

求微分：

$$
\begin{aligned}
dy_{\mathrm{pre}}
&=dx+J_F\,d(N(x))\\
&=dx+J_FJ_N\,dx\\
&=\left(I+J_FJ_N\right)dx.
\end{aligned}
$$

因此

$$
J_{\mathrm{pre}}
=I+J_FJ_N.
$$

恒等矩阵 $I$ 来自 shortcut 的直接微分。它不表示分支梯度很小，也不保证总 Jacobian 的所有特征值都接近 $1$；它表示输入存在一条不经过 $F$ 和 $N$ 的加法路径。

### Post-norm 的局部导数

令

$$
u=x+F(x).
$$

则

$$
du=(I+J_F)dx.
$$

对

$$
y_{\mathrm{post}}=N(u)
$$

再应用一次链式法则：

$$
dy_{\mathrm{post}}
=J_Ndu
=J_N(I+J_F)dx.
$$

所以

$$
J_{\mathrm{post}}
=J_N(I+J_F).
$$

这里的 $J_N$ 位于整个局部 Jacobian 的左侧。shortcut 的 $I$ 先与分支 Jacobian 相加，再一起经过 $J_N$；它没有 pre-norm 中单独留在外侧的恒等项。

### 顺序不能交换

一般情况下

$$
J_FJ_N\ne J_NJ_F.
$$

即使两个矩阵恰好可交换，下面两个表达式也不相同：

$$
I+J_FJ_N
\ne
J_N(I+J_F)
$$

除非额外满足特定条件。实现审计不能只检查“调用了两个相同模块”，还要检查加法节点和归一化节点的顺序。

## 一个四维线性分支例子

### 固定输入和分支

取

$$
x=(1,2,3,4),
\qquad
F(x)=\alpha x,
\qquad
\alpha=0.5.
$$

归一化取 $\gamma=\mathbf1$、$\beta=0$、$\epsilon=10^{-5}$ 的 LayerNorm。前文已经得到

$$
N(x)
=(-1.341635419969,-0.447211806656,
0.447211806656,1.341635419969).
$$

### 两种 forward 输出

pre-norm 输出为

$$
\begin{aligned}
y_{\mathrm{pre}}
&=x+0.5N(x)\\
&=(0.329182290016,1.776394096672,
3.223605903328,4.670817709984).
\end{aligned}
$$

post-norm 先做加法：

$$
u=x+0.5x=1.5x=(1.5,3,4.5,6).
$$

再对 $u$ 做 LayerNorm，得到

$$
y_{\mathrm{post}}
=(-1.341638401367,-0.447212800456,
0.447212800456,1.341638401367).
$$

两种输出的 shape 都是四维，数值语义完全不同。pre-norm 保留了输入的量级并加入一个归一化分支修正；post-norm 把相加结果重新投影到 LayerNorm 的输出尺度。

### 局部特征值

在 $x$ 处，LayerNorm 的 Jacobian 有三个方向类型：

1. 公共平移方向 $\mathbf1$ 的特征值为 $0$；
2. 中心化向量方向的特征值为 $\epsilon/s^3$；
3. 与 $\mathbf1$ 和中心化向量都正交的两个方向的特征值为 $1/s$。

pre-norm 的 $J_N$ 在原始输入 $x$ 处计算；post-norm 的 $J_N$ 在相加结果 $u=1.5x$ 处计算。两处的特征方向相同，但 $\epsilon$ 使尺度和局部特征值略有不同。本例在 $x$ 处有

$$
s=1.118038460877,
\qquad
\frac{\epsilon}{s^3}=7.155331664\times10^{-6},
\qquad
\frac{1}{s}=0.894423613313.
$$

post-norm 的相加结果满足

$$
s_u=\sqrt{2.8125+10^{-5}}=1.677053964546.
$$

因为 $J_F=\alpha I$，两种 block 的局部特征值为：

| 局部方向 | $J_N$ | $J_{\mathrm{pre}}=I+\alpha J_N$ | $J_{\mathrm{post}}=J_N(I+\alpha I)$ |
| --- | ---: | ---: | ---: |
| 公共平移 | $0$ | $1$ | $0$ |
| 中心化向量 | $0.000007155$ | $1.000003578$ | $0.000003180$ |
| 两个正交切向方向 | $0.894423613$ | $1.447211807$ | $0.894425601$ |

这个例子只描述一个输入点和一个线性分支。它显示了恒等项如何保留 pre-norm 的公共平移方向，也显示了 post-norm 如何让整个 shortcut 和分支的局部响应经过 $J_N$。

### 输入梯度的路径差异

对深层网络，局部 Jacobian 会沿层相乘：

$$
J_{\mathrm{stack}}
=J_{L-1}J_{L-2}\cdots J_0.
$$

pre-norm 的每个 $J_l$ 都有

$$
J_l=I+J_{F_l}J_{N_l},
$$

post-norm 的每个 $J_l$ 都有

$$
J_l=J_{N_l}(I+J_{F_l}).
$$

乘积中每一层的 $J_N$ 是否位于恒等项外侧，会改变不同方向的衰减和放大。不能只用单层输出均值或全局梯度范数替代逐层路径分析。

## 为什么 pre-norm 常保留更直接的梯度路径

### 恒等项位于 Jacobian 外侧

pre-norm 的局部导数可以写成

$$
dy=dx+\underbrace{J_FJ_Ndx}_{\text{残差分支修正}}.
$$

在分支 Jacobian 较小的区域，$dy$ 保留接近 $dx$ 的主项。深层乘积展开时，始终存在选择多个 $I$ 项的路径；这些项对应输入沿 shortcut 连续传递。

这不是“pre-norm 一定稳定”的定理。若 $J_FJ_N$ 在某些方向上持续为正且较大，$I+J_FJ_N$ 仍会放大；若分支更新改变表示尺度或注意力产生异常值，pre-norm 也会失效。这里的可检查事实是恒等 shortcut 在 Jacobian 中的位置。

### Post-norm 的恒等项被归一化包住

post-norm 的局部导数为

$$
dy=J_Ndx+J_NJ_Fdx.
$$

即使 $J_F=0$，也有

$$
J_{\mathrm{post}}=J_N.
$$

当 $N$ 有零方向或很小的径向响应时，输入在这些方向上的梯度会被直接压低。LayerNorm 的公共平移方向就是一个严格零方向；RMSNorm 在 $\epsilon=0$ 时的径向方向也是零方向。

### 残差更新尺度仍然重要

pre-norm 只保证 shortcut 存在，不保证分支更新 $F(N(x))$ 的幅度合适。应同时记录

$$
\rho_l
=\frac{\lVert F_l(N_l(x_l))\rVert_2}
{\lVert x_l\rVert_2+\delta},
$$

其中 $\delta>0$ 只用于避免零分母。若 $\rho_l$ 在深度方向持续增大，输入表示可能被残差更新主导；若持续接近零，分支可能几乎没有贡献。

## 深层堆叠的尺度账本

### 预激活与后归一化不是同一件事

pre-norm 的 block 输出是加法结果：

$$
x_{l+1}=x_l+F_l(N_l(x_l)).
$$

它的 residual stream 不一定在每层满足固定均值或均方约束。post-norm 的输出是

$$
x_{l+1}=N_l(x_l+F_l(x_l)),
$$

因此每层输出都会经过当前 $N_l$ 的统计变换。两者的表示尺度账本必须分别记录。

### 线性残差更新的数量级

若把分支局部近似为

$$
F_l(z)\approx\alpha_l z,
$$

并只做数量级估计，pre-norm 的更新是

$$
x_{l+1}\approx x_l+\alpha_lN_l(x_l).
$$

当 $N_l(x_l)$ 的范数与 $x_l$ 同量级时，$\alpha_l$ 直接决定每层残差增量的相对尺度。比如 $\alpha=0.05$、24 层的纯乘法因子 $(1+\alpha)^{24}$ 为

$$
(1.05)^{24}=3.225099944.
$$

这不是一个 pre-norm 网络的精确输出，只是说明“小的同向残差更新”在深度上也会累积。归一化位置不能替代残差比例和分支 Jacobian 的测量。

### Post-norm 的每层输出约束不等于梯度恒定

post-norm 可以让每层输出的统计尺度更受控，但

$$
J_N(I+J_F)
$$

仍然可能在某个方向变小或变大。前向方差稳定、反向梯度稳定和表示可分性是三个不同检查项，不能用其中一项替代另外两项。

### 24 步的局部对照

在一个简化的局部模型中，若每层某方向的 pre-norm 乘积因子为 $1.05$，24 层后为 $3.225099944$；若另一个方向的因子为 $0.95$，24 层后为

$$
(0.95)^{24}=0.291989024.
$$

这两个数字只是乘积示例，不是对任何架构的预测。实际 $J_F$、$J_N$、参数共享和注意力混合会让不同方向拥有不同因子；报告深层稳定性时必须说明局部模型和测量方向。

## Attention、FFN 和残差流的接口

### Attention 分支

pre-norm attention 分支通常为

$$
\Delta_{\mathrm{attn}}
=F_{\mathrm{attn}}(N_{\mathrm{attn}}(x)).
$$

然后

$$
x'=x+\Delta_{\mathrm{attn}}.
$$

post-norm 则为

$$
x'=N_{\mathrm{attn}}\bigl(x+F_{\mathrm{attn}}(x)\bigr).
$$

在两种形式中，causal mask、padding mask 和 attention 的 score 计算仍需要独立检查。改变 norm 位置不等于改变可见性规则。

### FFN 分支

pre-norm FFN 为

$$
y=x'+F_{\mathrm{ffn}}(N_{\mathrm{ffn}}(x')).
$$

post-norm FFN 为

$$
y=N_{\mathrm{ffn}}\bigl(x'+F_{\mathrm{ffn}}(x')\bigr).
$$

若 FFN 使用门控结构，$J_F$ 还包含门值和激活导数；不能把本节的 $\alpha I$ 例子当作真实 FFN Jacobian。

### Residual stream 的宽度

不论 pre-norm 还是 post-norm，加法节点要求

$$
\operatorname{shape}(x)
=\operatorname{shape}(F(x)).
$$

在 Transformer 中通常是 $(B,T,d_{\mathrm{model}})$。如果分支内部暂时变为 head 或 FFN 的中间宽度，必须在分支结束前投影回 $d_{\mathrm{model}}$。

残差流的定义和跨层传递见 [残差流](../transformer-components/residual-streams/)。本篇的 pre/post 判断发生在同一个残差加法节点附近，不改变 residual stream 的 shape 合同。

## Final norm 与输出头

### Pre-norm 常需要 final norm

pre-norm block 叠加后，最后一个 residual stream 可以写成

$$
x_L=x_0+\sum_{l=0}^{L-1}F_l(N_l(x_l)).
$$

这个 $x_L$ 没有自动经过最后一个 $N_L$。常见语言模型会在输出投影前加入

$$
h_{\mathrm{out}}=N_{\mathrm{final}}(x_L).
$$

final norm 不是任意一个 block norm 的别名；它有独立参数，且改变输出 logits 的尺度。

### Post-norm 的末端状态也要读取配置

post-norm 每个 block 的输出已经经过对应 $N_l$，但是否还放 final norm 由具体架构决定。不能根据“post-norm 已经归一化过”自行删除末端 norm，也不能把不同模型的 final norm 规则互换。

### 输出投影的顺序

若 hidden 到 logits 的投影为 $W_U$，常见顺序是

$$
\operatorname{logits}=h_{\mathrm{out}}W_U.
$$

是否使用 final norm、是否共享 embedding 权重、输出投影前是否还有额外缩放，都应与 pre/post 标签分开记录。归一化位置只解决 block 内的一个顺序问题。

## Norm 类型、残差位置和其他开关

### LayerNorm 与 RMSNorm 可以分别组合

以下四种组合逻辑上不同：

| 归一化函数 | pre-norm | post-norm |
| --- | --- | --- |
| LayerNorm | $x+F(\operatorname{LN}(x))$ | $\operatorname{LN}(x+F(x))$ |
| RMSNorm | $x+F(\operatorname{RMS}(x))$ | $\operatorname{RMS}(x+F(x))$ |

LayerNorm 是否减均值、RMSNorm 是否含 $\beta$ 属于 $N$ 的定义；$N$ 在加法前还是加法后属于 block 的定义。配置审计必须保留两列。

### Dropout 和 stochastic depth 不改变标签

如果分支包含 dropout：

$$
y=x+F_{\mathrm{dropout}}(N(x))
$$

仍然是 pre-norm；如果 dropout 位于加法后、norm 前，执行图需要逐算子记录。把随机算子放在不同位置会改变训练态的统计和梯度，但不应模糊 pre/post 的核心定义。

### Residual scaling 是第三个选择

有些架构在加法前缩放分支：

$$
y=x+\lambda F(N(x)).
$$

这仍是 pre-norm，但新增了 $\lambda$。$\lambda$、LayerScale、DropPath 和 norm 位置需要独立记录；不能把残差缩放误写成 post-norm。

## 数值精度与执行合同

### Norm 的统计精度

LayerNorm 或 RMSNorm 的 $J_N$ 都依赖归约结果。混合精度实现应固定：

| 项目 | 需要记录 |
| --- | --- |
| 输入 dtype | hidden 的存储格式 |
| 归约 dtype | mean、variance 或 mean square 的累加格式 |
| $\epsilon$ | 数值和放置位置 |
| 参数 dtype | $\gamma,\beta$ 或 $\gamma$ 的格式 |
| residual add dtype | shortcut 与分支相加的计算格式 |
| 输出 dtype | block 输出写回格式 |

pre/post 的差异发生在 forward 图上，不能用相同的 dtype 名称推断两种图的结果相同。

### 加法精度也属于残差合同

pre-norm 的加法是

$$
x+F(N(x)).
$$

若 shortcut 和分支在低精度中相加，残差小更新可能被舍入吞掉；若在 FP32 中加法再写回低精度，结果又不同。post-norm 还会把加法后的结果送进归一化，舍入误差会继续进入均方或方差统计。需要分别测试 residual add 和 norm。

### 训练与评估的随机分支

LayerNorm 和 RMSNorm 通常没有 running statistics，训练态和评估态的 norm 公式相同。dropout、DropPath、mask 和 cache 仍可能让两态执行图不同。复现模型时，先固定随机开关，再比较 norm 顺序。

## 失效模式

### 把 pre-norm 的公式读成 post-norm

把

$$
x+F(N(x))
$$

实现成

$$
N(x+F(x))
$$

会同时改变分支输入、shortcut 输出、Jacobian 和深层尺度。检查 forward 中 norm 的调用参数和加法节点，而不是只查模块名称。

### 把 post-norm 的 shortcut 当作恒等梯度

post-norm 的 shortcut 先与分支相加，再经过 $J_N$。即使代码中有一个加法操作，也不能把其梯度直接写成 $I$。应从

$$
J_N(I+J_F)
$$

开始追踪。

### 漏掉 final norm

pre-norm block 的最后 residual stream 与输出头之间常有 final norm。删除它可能让 hidden 和 logits 尺度变化，即使所有 block 内部的 norm 都存在。核对模块数量、参数前缀和输出头输入。

### 把 LayerNorm 与 RMSNorm 混用

两种 norm 的 Jacobian 零方向、$\beta$ 参数和公共平移行为不同。读取配置中的 norm type 和参数 shape，不能只看模块类名的缩写。

### 把 norm 放到错误的子层

attention 和 FFN 可以各自有 norm。若只给 attention 使用 pre-norm、FFN 使用 post-norm，这是混合布局；不能用一个总标签覆盖两个子层。逐个加法节点记录 $N$ 的位置。

### 忽略投影 shortcut

当 $F(x)$ 的宽度不同，shortcut 不是 $I$ 而是某个投影 $S$。继续使用

$$
J_{\mathrm{pre}}=I+J_FJ_N
$$

会得到错误梯度。需要把 $I$ 替换为 $J_S$，并核对投影是否也经过 norm。

### 用全局梯度范数替代方向分析

不同方向可能同时存在放大和衰减。全局 L2 梯度范数可能保持稳定，而公共平移、径向或 attention 子空间已经被压低。至少记录逐层范数、方向投影和 $J_N$ 的局部检查。

### 把单层数字外推成架构结论

线性 $F(x)=0.5x$ 的四维例子只展示链式法则和一个局部点。真实 $F$ 会随 token、参数、mask 和训练状态变化。使用例子验证公式，不把它当作所有深层模型的测量结果。

## 一个可复算的核验协议

### 先核对计算图

对一个子层，记录：

| 项目 | pre-norm | post-norm |
| --- | --- | --- |
| norm 输入 | $x$ | $x+F(x)$ |
| 分支输入 | $N(x)$ | $x$ |
| 加法输出 | $x+F(N(x))$ | $x+F(x)$ |
| block 输出 | 加法结果 | $N(x+F(x))$ |
| 局部 Jacobian | $I+J_FJ_N$ | $J_N(I+J_F)$ |

这张表先于任何数值比较。没有计算图，单看一组输出不能确定 norm 的位置。

### 再核对四维前向值

使用 LayerNorm、$x=(1,2,3,4)$、$\epsilon=10^{-5}$、$F(x)=0.5x$，应得到：

| 项目 | 参考值 |
| --- | ---: |
| $N(x)_1$ | $-1.341635419969$ |
| $y_{\mathrm{pre},1}$ | $0.329182290016$ |
| $y_{\mathrm{pre},4}$ | $4.670817709984$ |
| $y_{\mathrm{post},1}$ | $-1.341638401367$ |
| $y_{\mathrm{post},4}$ | $1.341638401367$ |

pre-norm 输出保留 residual stream 的量级，post-norm 输出接近归一化尺度；两个结果都应与同一个 $F$、同一个 $N$ 的配置相符。

### 核对局部 Jacobian

使用 $\alpha=0.5$ 的线性分支，参考特征值为：

$$
\begin{aligned}
\lambda_{\mathrm{pre},\mathbf1}&=1,\\
\lambda_{\mathrm{post},\mathbf1}&=0,\\
\lambda_{\mathrm{pre},\mathrm{tan}}&=1.447211806656,\\
\lambda_{\mathrm{post},\mathrm{tan}}&=1.341635419969.
\end{aligned}
$$

有限差分可以逐坐标核对 $J_{\mathrm{pre}}$ 和 $J_{\mathrm{post}}$。若两个矩阵相同，优先检查 norm 是否被放在同一个位置，或测试分支是否意外设为零。

### 最后执行深层和资源检查

至少记录：

1. 每层 residual ratio；
2. 每层 norm 输入与输出的均值、方差或均方；
3. 分支与 shortcut 的 dtype；
4. final norm 是否存在及其参数；
5. attention、FFN 是否使用同一 pre/post 位置；
6. projected shortcut 的 shape 与 Jacobian；
7. cache、padding mask 和 dropout 的执行态。

pre/post 是一条结构标签，不能替代这些逐项证据。

## 相关词条

- [LayerNorm](../transformer-components/layernorm/)
- [RMSNorm](../transformer-components/rmsnorm/)
- [LayerNorm 与残差](../transformer-components/layernorm-residuals/)
- [残差流](../transformer-components/residual-streams/)
- [梯度消失与爆炸](../backpropagation/vanishing-and-exploding/)
- [混合精度训练](../training-nn/mixed-precision/)
