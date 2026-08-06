---
title: "LayerNorm 与残差：归一化放在相加前还是相加后"
tags: ["why-models-learn"]
---

LayerNorm（Layer Normalization，层归一化）沿每个 token 的特征维度计算均值和方差，再用可学习的缩放与平移恢复表示尺度。残差连接把输入与子层输出逐分量相加。Transformer 中最重要的接口问题，是先归一化再进入子层，还是先完成残差相加再归一化：前者是 pre-norm，后者是 post-norm。两种顺序使用同一个 LayerNorm 公式，却产生不同的 block 函数和反向 Jacobian。

本文把 LayerNorm 当作残差 block 的组成件来分析。先固定统计轴、$\epsilon$ 和仿射参数，再用一个四维向量核对数值。随后推导 pre-norm 与 post-norm 的局部 Jacobian，说明深层堆叠中的恒等路径、最终归一化、padding、KV cache 和混合精度需要分别记录。LayerNorm 的完整独立实现与 RMSNorm 的对照留给后续词条。

![pre-norm 与 post-norm 的残差顺序：pre-norm 先经过 LayerNorm 再进入子层，post-norm 先将子层输出与输入相加再经过 LayerNorm](/assets/transformer-components/svg/layernorm-residuals.1.svg)

## 先固定 LayerNorm 的统计对象

### 输入张量与归约轴

设一个 Transformer block 的输入为

$$
x\in\mathbb R^{B\times T\times d},
$$

其中 $B$ 是 batch size，$T$ 是序列长度，$d=d_{\mathrm{model}}$ 是每个 token 的特征宽度。对固定的 batch 索引 $b$ 和 token 索引 $t$，LayerNorm 只在最后一维的 $d$ 个特征上计算统计量：

$$
\mu_{b,t}
=\frac{1}{d}\sum_{r=1}^{d}x_{b,t,r},
\qquad
\sigma^2_{b,t}
=\frac{1}{d}\sum_{r=1}^{d}(x_{b,t,r}-\mu_{b,t})^2.
$$

记

$$
\hat x_{b,t,r}
=\frac{x_{b,t,r}-\mu_{b,t}}
{\sqrt{\sigma^2_{b,t}+\epsilon}},
$$

则带仿射参数的 LayerNorm 为

$$
\operatorname{LN}(x_{b,t,r})
=\gamma_r\hat x_{b,t,r}+\beta_r.
$$

$\gamma,\beta\in\mathbb R^d$ 只沿特征维度广播。每个 token 有自己的 $\mu$、$\sigma^2$，所有 token 共享同一组 $\gamma$、$\beta$。

这里的方差分母是 $d$。它是当前 token 的总体方差约定，不能替换为统计学中样本方差的 $d-1$。$\epsilon>0$ 位于平方根内部，用于避免特征完全相同时除以零；它也改变低方差输入的实际缩放。

### 轴合同决定统计是否耦合

| 归一化 | 统计轴 | 不同 token 是否共享统计量 | 是否需要 running statistics |
| --- | --- | --- | --- |
| LayerNorm | 每个 token 的特征轴 | 否 | 否 |
| BatchNorm | batch、空间或实现指定的轴 | 通常是 | 训练态更新，评估态读取 |
| RMSNorm | 每个 token 的特征轴 | 否 | 否 |
| 误写成跨序列归一化 | token 轴或 token 与特征混合 | 会 | 取决于实现 |

LayerNorm 的 token 局部性使当前 token 的统计量不依赖 batch 中还有哪些样本，也不依赖同一序列的历史长度。这个性质适合自回归解码；BatchNorm 的训练/评估分支和统计轴见[批量归一化](../training-nn/batch-normalization/)。

## 一个四维向量的数值例子

### 先算均值、方差和标准化向量

取一个 token

$$
x=(1,2,3,4),
\qquad d=4,
\qquad \epsilon=10^{-5}.
$$

均值为

$$
\mu=\frac{1+2+3+4}{4}=2.5.
$$

中心化向量为 $c=x-\mu\mathbf 1=(-1.5,-0.5,0.5,1.5)$，方差为

$$
\sigma^2
=\frac{1.5^2+0.5^2+0.5^2+1.5^2}{4}
=1.25.
$$

因此

$$
\sqrt{\sigma^2+\epsilon}=1.11803846088,
$$

$$
\hat x
=(-1.34163541997,-0.44721180666,
0.44721180666,1.34163541997).
$$

当 $\gamma=(1,1,1,1)$、$\beta=(0,0,0,0)$ 时，输出就是 $\hat x$。当 $\gamma$ 和 $\beta$ 改变时，归一化的中心与尺度仍由当前 token 决定，但输出坐标可以恢复任务所需的可学习尺度和偏移。

### 统计量的三个可检查结果

在不考虑 $\epsilon$ 对尺度的微小影响时，标准化向量应满足

$$
\frac{1}{d}\sum_r\hat x_r\approx0,
\qquad
\frac{1}{d}\sum_r\hat x_r^2\approx1.
$$

使用上面的 $\epsilon=10^{-5}$，独立计算得到

| 检查项 | 数值 | 解释 |
| --- | ---: | --- |
| 均值 $\operatorname{mean}(\hat x)$ | $0$ | 中心化保持，浮点误差在零附近 |
| 二阶矩 $\operatorname{mean}(\hat x^2)$ | $0.99999200006$ | $\epsilon$ 使实际尺度略低于 1 |
| 范数 $\lVert\hat x\rVert_2$ | $1.99999200005$ | 四维向量的理想范数为 $\sqrt4=2$ |

如果实现把方差分母写成 $d-1$，或者在错误的轴上归约，这三项会出现系统偏差。仅检查输出 shape 无法发现这种错误。

## 两种残差顺序

设 $F_l$ 是第 $l$ 个 Transformer 子层，例如多头注意力或 FFN；设 $N_l$ 是带独立参数的 LayerNorm。为简化记号，先只看一个残差子层。

### Pre-norm：先归一化分支输入

pre-norm 把 LayerNorm 放在残差分支内部：

$$
x_{l+1}
=x_l+F_l\bigl(N_l(x_l)\bigr).
$$

输入 $x_l$ 直接进入加法。LayerNorm 和子层只作用在修正量的计算路径上。一个包含 attention 与 FFN 的 decoder block 可以写成

$$
h_l
=x_l+\operatorname{Attn}_l\bigl(N_{l,1}(x_l)\bigr),
$$

$$
x_{l+1}
=h_l+\operatorname{FFN}_l\bigl(N_{l,2}(h_l)\bigr).
$$

两次加法各自拥有一条未经过 LayerNorm 的 shortcut。这里的“未经过”只描述 shortcut 分支；下一层的 LayerNorm 仍会读取整个 residual stream。

### Post-norm：相加后归一化

post-norm 先计算子层修正，再把合并结果交给 LayerNorm：

$$
x_{l+1}
=N_l\bigl(x_l+F_l(x_l)\bigr).
$$

两个子层的写法为

$$
h_l
=N_{l,1}\bigl(x_l+\operatorname{Attn}_l(x_l)\bigr),
$$

$$
x_{l+1}
=N_{l,2}\bigl(h_l+\operatorname{FFN}_l(h_l)\bigr).
$$

post-norm 使每个子层输出都经历一次中心化、缩放和仿射变换。相加位置与 LayerNorm 位置必须从实现代码或计算图中逐项确认；“有残差”本身不足以确定 block 属于哪一种顺序。

### 顺序表

| 变体 | 一个子层的函数 | shortcut 是否直接到加法 | LayerNorm 看到的张量 |
| --- | --- | --- | --- |
| pre-norm | $x+F(N(x))$ | 是 | 子层输入 $x$ |
| post-norm | $N(x+F(x))$ | 先相加，再经过 $N$ | 合并结果 |
| 只在 block 末尾归一化 | $N(x+F_2(x+F_1(x)))$ | 中间加法仍可存在 | block 最终输出 |
| sandwich 变体 | $x+N_2(F(N_1(x)))$ 等 | 取决于具体定义 | 分支内部一个或多个位置 |

表中的第三、第四行是接口分类。它们的训练稳定性不能仅由名称推断，需要固定每个子层的实际图、参数共享方式和初始化。

## Jacobian：LayerNorm 放置如何改变梯度路径

### 先写局部导数

令 $J_F$ 表示 $F$ 对输入的局部 Jacobian，$J_N$ 表示 LayerNorm 对输入的局部 Jacobian。对 pre-norm，链式法则给出

$$
J_{\mathrm{pre}}
=I+J_FJ_N.
$$

对 post-norm，先经过残差相加，再经过 LayerNorm：

$$
J_{\mathrm{post}}
=J_N(I+J_F).
$$

矩阵乘法一般不交换，所以两者不能按同一个“残差加上一个归一化”公式处理。即使 $J_F$ 与 $J_N$ 在某个特例下可交换，前者包含显式的 $I$ 项，后者仍由 $J_N$ 左乘整个合并路径。

### LayerNorm 的局部 Jacobian

对单个 token，令

$$
c=x-\mu\mathbf 1,
\qquad
s=\sqrt{\sigma^2+\epsilon},
$$

并先把 $\gamma$ 看作逐坐标乘法。带仿射参数的 LayerNorm Jacobian 为

$$
J_N
=\frac{\operatorname{diag}(\gamma)}{s}
\left[
I-\frac{1}{d}\mathbf 1\mathbf 1^{\mathsf T}
-\frac{cc^{\mathsf T}}{d s^2}
\right].
$$

括号中的第一项保留局部坐标变化，第二项去掉共同平移方向，第三项校正方差变化。$\beta$ 是平移参数，因此不出现在输入 Jacobian 中。$\epsilon$ 增大时，$s$ 增大，低方差方向的导数幅度会下降。

对任意常数 $a$，LayerNorm 满足

$$
N(x+a\mathbf1)=N(x),
$$

所以共同平移方向的局部导数为零。对整体缩放，$\epsilon=0$ 时归一化还具有尺度不变性；$\epsilon>0$ 会把这种不变性变成近似关系。实际模型还会乘以 $\gamma$，因此不能把 LayerNorm 输出的所有方向都视为单位增益。

### 零分支的对照

令残差分支在某个点满足 $F(x)=0$，但局部导数 $J_F$ 可以单独保留。若只把分支函数暂时置零并令 $J_F=0$，则

$$
J_{\mathrm{pre}}=I,
\qquad
J_{\mathrm{post}}=J_N.
$$

这个边界状态揭示了两种顺序的结构差异：pre-norm 仍有一条纯 identity 的局部路径，post-norm 的合并结果随后要经过 LayerNorm。它描述的是 block 结构，不构成对完整训练的收敛保证。

### 一个线性分支的局部对照

再令 $F(x)=\alpha x$，则 $J_F=\alpha I$。局部近似为

$$
J_{\mathrm{pre}}
=I+\alpha J_N,
\qquad
J_{\mathrm{post}}
=(1+\alpha)J_N.
$$

当 $\alpha=0.1$ 时，pre-norm 的每个局部方向都保留 identity 项，再叠加归一化分支；post-norm 的所有方向都先经过 $J_N$。如果某个方向被 LayerNorm 的中心化或尺度项压低，post-norm 的 shortcut 也会承受这一局部变换。

## 深层堆叠中的信号尺度

### Pre-norm 的加法递推

把 pre-norm 的残差量记为

$$
r_l=F_l\bigl(N_l(x_l)\bigr),
\qquad
x_{l+1}=x_l+r_l.
$$

前向量值沿深度满足

$$
x_L=x_0+\sum_{l=0}^{L-1}r_l.
$$

这个展开式只表示每一层的残差加法；$r_l$ 仍依赖前面所有层的状态。反向微分则是

$$
\mathrm d x_{l+1}
=\mathrm d x_l
+J_{F_l}J_{N_l}\,\mathrm d x_l.
$$

其中第一项从每一层直接传递。残差量过大时，$\sum r_l$ 仍可能造成 residual stream 的尺度漂移；pre-norm 不能替代残差缩放、学习率和初始化审计。

### Post-norm 的层间尺度控制

post-norm 每层输出都形如

$$
x_{l+1}=N_l(u_l),
\qquad
u_l=x_l+F_l(x_l).
$$

LayerNorm 把每个 token 的中心和二阶尺度重新映射，再由 $\gamma_l,\beta_l$ 恢复可学习的坐标尺度。这提供了每层输出统计的控制点，但反向路径也包含 $J_{N_l}$。$\gamma_l$ 的大小、输入方差和 $\epsilon$ 共同决定局部增益。

### 最终 LayerNorm

常见的 pre-norm decoder 会在最后一个 residual block 之后增加 final LayerNorm，再送入输出投影：

$$
z=N_{\mathrm{final}}(x_L),
\qquad
\mathrm{logits}=zW_U+b_U.
$$

如果实现省略 final LayerNorm，输出头接收到的是未经最后一次尺度约束的 residual stream。是否省略属于架构合同，必须与 checkpoint、初始化和训练代码保持一致。不能从 block 内部的 pre-norm 推断 final LayerNorm 一定存在。

## Attention、FFN 与残差流的接口

### 注意力子层

对输入 $x$，pre-norm attention 子层可抽象为

$$
h=x+\operatorname{MHA}(N_{\mathrm{attn}}(x)).
$$

LayerNorm 先沿每个 token 的特征维度归一化；MHA 再按 head 拆分 Q/K/V。归一化不会改变 token 数量，也不会改变 causal mask、RoPE 或 ALiBi 的位置索引。MHA 的 head 轴与输出投影见[多头注意力](../attention/multi-head-attention/)，位置条件见[RoPE](./rope/)和[ALiBi](./alibi/)。

post-norm attention 则是

$$
h=N_{\mathrm{attn}}\bigl(x+\operatorname{MHA}(x)\bigr).
$$

两式中 MHA 的输入不同，Q/K/V 的数值尺度、score 的方差和 softmax 权重都可能改变。只把 LayerNorm 作为“输出再标准化”会漏掉它对 attention score 输入的作用。

### FFN 子层

pre-norm 的第二个子层读取 attention 残差后的状态：

$$
y=h+\operatorname{FFN}(N_{\mathrm{ffn}}(h)).
$$

这里的 $N_{\mathrm{ffn}}$ 通常与 $N_{\mathrm{attn}}$ 不共享 $\gamma,\beta$。共享 LayerNorm 参数会改变两个子层的尺度耦合，属于需要单独记录的实现选择。

### 残差流的 shape 合同

对于标准 decoder block，$x$、attention 输出、FFN 输出和每次相加后的 residual stream 都应具有

$$
(B,T,d_{\mathrm{model}}).
$$

LayerNorm 不负责把不同宽度的张量投影到同一维度。若子层改变宽度，必须先使用线性 projection 或调整子层输出；直接相加会在 shape 检查阶段失败，错误广播则可能悄悄改变语义。残差连接的一般 shape contract 见[残差连接](../cnn/residual-connections/)。

## 统计轴、padding 与 KV cache

### Padding token 的处理

LayerNorm 会对传入 block 的每个 token 执行计算，包括 padding token。attention mask 约束 key 的可见性，loss mask 约束哪些位置贡献目标；二者都不会自动取消 LayerNorm 的前向计算。

如果 padding embedding 是全零向量，且 $\beta=0$，它在理想算术中会输出零；当 $\beta\ne0$、后续有 bias 或残差累加时，padding 位置可能携带非零表示。因此需要明确记录：

| 位置 | 需要检查的合同 |
| --- | --- |
| LayerNorm | 是否对 padding token 计算，归约轴是否仍为特征轴 |
| attention | key padding mask 是否覆盖所有 attention head |
| residual | padding 位置是否需要在每个 block 后重新置零 |
| loss | token loss 的分母是否只统计有效 token |
| batch | 不同序列长度的 padding 是否改变了其它 token 的统计量 |

最后一行是 LayerNorm 与 BatchNorm 的重要边界：正确的 LayerNorm 不会让一个序列的 padding 改变另一个 token 的均值和方差。

### 自回归 KV cache

LayerNorm 的统计量只依赖当前 token 的 $d$ 个特征，因此把前缀放进 KV cache 后，当前新 token 的 LayerNorm 统计量不需要读取完整历史。实现仍需固定以下顺序：

$$
x_t
\xrightarrow{\ N\ }
\widetilde x_t
\xrightarrow{W_Q,W_K,W_V}
(q_t,k_t,v_t),
$$

缓存的是哪一个阶段的 $k_t,v_t$，必须与训练图相同。GQA 或 MQA 只改变 K/V head 的共享方式，不会改变 LayerNorm 的特征归约轴；相关 cache 账本见[GQA 与 MQA](../attention/gqa-and-mqa/)。RoPE 和 ALiBi 的位置处理仍使用解码偏移，不能因为 LayerNorm 是 token 局部操作就跳过位置条件。

### 混合精度

低精度实现需要分别记录输入、统计量、$\gamma/\beta$ 和输出的 dtype。常见稳定做法是：

1. 把求和、均值、方差累积到 FP32。
2. 在平方根内部加入约定的 $\epsilon$。
3. 用 FP32 统计量完成标准化。
4. 按 kernel 合同写回目标 dtype。

FP16 或 BF16 下直接累积长特征向量，可能增加均值和方差的舍入误差。将 $\epsilon$ 从 FP32 kernel 的值改成与 dtype 相关的另一数值，也会改变低方差 token 的输出和梯度。配置文件、融合 kernel 和参考实现要使用同一项数值合同。

## 参数、计算与激活内存

### LayerNorm 的参数量

每个 LayerNorm 的可学习参数为 $\gamma$ 和 $\beta$：

$$
P_{\mathrm{LN}}=2d.
$$

取 $d=4096$：

$$
P_{\mathrm{LN}}=8192.
$$

这些参数只占 8192 个标量；它们不随序列长度 $T$ 增长。若使用 FP16，参数存储为 $16384$ bytes，即 $16$ KiB；若使用 FP32，则为 $32768$ bytes，即 $32$ KiB。

### 一个长序列的统计账

取 $B=2$、$T=4096$、$d=4096$，token 数为

$$
BT=8192.
$$

每个 token 至少需要一个均值和一个方差统计量。若反向阶段以 FP32 保存这两项，统计量存储为

$$
8192\times2\times4
=65536\ \text{bytes}
=64\ \text{KiB}.
$$

输出激活仍有 $BTd=33554432$ 个元素；FP16 下仅计算一个输出张量就占 $64$ MiB。LayerNorm 的统计量开销是线性 token 账，attention score 的逻辑矩阵则沿 $T^2$ 增长；二者不能用同一个“显存占用”数字替代。

| 项目 | 公式 | $B=2,T=4096,d=4096$ |
| --- | --- | ---: |
| affine 参数 | $2d$ | $8192$ 个标量 |
| token 数 | $BT$ | $8192$ |
| FP32 均值与方差 | $BT\times2\times4$ bytes | $64$ KiB |
| 输出元素 | $BTd$ | $33554432$ |
| FP16 输出元素 | $BTd\times2$ bytes | $64$ MiB |

### 计算量的边界

LayerNorm 对每个 token 读取 $d$ 个特征，计算均值、平方差、倒数平方根、仿射变换，算术量与 $BTd$ 成正比。它不产生 $T\times T$ 的 query-key 交互矩阵，也不改变 attention 的二次复杂度。融合 LayerNorm、残差和 bias 可以减少显存读写，但融合前后必须保持同一个加法顺序、$\epsilon$ 和舍入合同。

## LayerNorm、RMSNorm 与 BatchNorm 的边界

### 三种统计规则

| 归一化 | 中心化 | 尺度统计 | 典型参数 | token 之间的统计耦合 |
| --- | --- | --- | --- | --- |
| LayerNorm | 减去均值 | 方差加 $\epsilon$ | $\gamma,\beta$ | 无 |
| RMSNorm | 不减均值 | 均方根加 $\epsilon$ | 通常只有 $\gamma$ | 无 |
| BatchNorm | 按指定 batch/空间轴减均值 | batch 方差与 running statistics | $\gamma,\beta$ 加 running state | 训练态通常有 |

LayerNorm 与 RMSNorm 都按 token 的特征轴归约，但是否去均值会改变输出的平移性质和局部 Jacobian。RMSNorm 的公式、参数变体和独立数值例子放在后续词条。BatchNorm 的训练态和评估态分支见[批量归一化](../training-nn/batch-normalization/)，范数的几何定义见[范数](../linear-algebra/norms/)。

### 归一化与残差是两个正交选择

残差连接决定输入如何与分支输出合并；归一化决定某个张量沿哪些轴如何重设中心和尺度。可以有残差而无 LayerNorm，也可以在没有 shortcut 的普通网络里使用 LayerNorm。审计时把两项分开记录：

| 结构问题 | 归一化问题 |
| --- | --- |
| 两分支是否 shape 对齐 | 均值、方差沿哪个轴计算 |
| shortcut 是否 identity 或 projection | 方差分母是 $d$ 还是其它数值 |
| merge 是加法还是 concat | $\epsilon$、$\gamma$、$\beta$ 的 dtype |
| merge 后是否还有 activation | 统计是否跨 token、batch 或空间位置 |

## 失效模式

### 把 pre-norm 和 post-norm 读反

代码中的调用顺序比类名更可靠。逐项标记 `norm`、子层、`add` 的执行顺序，并把一个 token 的计算图画出来。只看到 `ResidualBlock`、`TransformerLayer` 或 `Norm` 类名，无法确定实际变体。

### 在错误的轴上归一化

把 $(B,T,d)$ 的最后一维误写成序列维度，会让同一 token 的特征不再共同标准化；把 batch 维加入归约，还会引入样本间耦合。用两个 token、两个 batch 样本构造输入，改变其中一个样本后检查另一个样本的 LayerNorm 输出，可以直接发现这种错误。

### 把 $\epsilon$ 当作无关常数

高方差 token 对 $\epsilon$ 不敏感，低方差 token 对 $\epsilon$ 敏感。固定一个常数向量和一个近似常数向量，分别记录输出、倒数平方根和输入梯度。生产 kernel、参考实现和数值测试必须使用同一个 $\epsilon$ 位置与数值。

### 只比较均值和方差，不比较残差路径

两个实现都可能报告输出均值接近零、方差接近一，但一个实现为 $x+F(N(x))$，另一个实现为 $N(x+F(x))$。将 $F$ 置零后比较输出与输入的差异，再比较对输入的 Jacobian，能区分这两种计算图。

### 忽略 final LayerNorm

pre-norm block 的内部顺序不能推出 block 堆栈末尾是否有 final LayerNorm。读取模型配置、checkpoint 参数名和输出头前向调用，确认最终 residual stream 是否经过归一化。缺少 final norm 可能表现为 logits 尺度、loss 初始值或生成分布变化；原因需要结合具体实现和 checkpoint 核对。

### 让 padding 进入错误的聚合

LayerNorm 本身按 token 归约，padding 不会改变其它 token 的统计量。若观察到序列长度改变会影响有效 token 的 LayerNorm 输出，应先检查实际 kernel 是否混入 token 轴，再检查 embedding、attention mask、残差清零和 loss reduction。

### 混合精度参考不一致

参考实现使用 FP32 统计而 fused kernel 使用低精度累积时，输出差异在低方差 token 上会被放大。对比时固定输入 dtype、统计 dtype、$\epsilon$、参数 dtype、舍入模式和容差，再判断差异是否超出数值误差范围。

## 一个可复算的核验协议

| 步骤 | 输入或观察量 | 通过条件 |
| --- | --- | --- |
| 1. 固定 shape | $B,T,d$ 与 layout | LayerNorm 的归约轴是最后特征轴 |
| 2. 固定公式 | $\mu,\sigma^2,\epsilon,\gamma,\beta$ | 分母、$\epsilon$ 位置和仿射顺序与实现一致 |
| 3. 固定顺序 | `norm → sublayer → add` 或 `sublayer → add → norm` | 计算图与目标变体完全一致 |
| 4. 固定边界 | $F=0$、常数向量、近似常数向量 | pre-norm 保留 identity 输出/导数边界，post-norm 显示 $J_N$ 作用 |
| 5. 检查独立性 | 改变其它 batch 或 token | 当前 token 的统计量不随无关 token 改变 |
| 6. 检查 mask | padding、causal mask、loss mask | mask 只影响规定的读取或损失位置 |
| 7. 检查 cache | prefill 与 decode 的同一 token | norm、投影、缓存阶段与训练图一致 |
| 8. 检查 dtype | FP32 参考与目标 kernel | 误差在预先规定的容差内，低方差输入单独记录 |
| 9. 检查资源 | 参数、统计量、输出激活 | 数量与 $2d$、$BT$、$BTd$ 账本一致 |

核验报告至少保存一组输入向量、每个 token 的均值和方差、标准化结果、残差前后张量、$\gamma/\beta$、$\epsilon$、dtype 和顺序标签。只有输出 shape 正确的实现，不能据此视为 LayerNorm 与残差合同已满足。

## 相关词条

- [残差连接](../cnn/residual-connections/)
- [批量归一化](../training-nn/batch-normalization/)
- [范数](../linear-algebra/norms/)
- [多头注意力](../attention/multi-head-attention/)
- [GQA 与 MQA](../attention/gqa-and-mqa/)
- LayerNorm（后续词条）
- RMSNorm（后续词条）
- Pre-norm 与 Post-norm（后续词条）
- 残差流（后续词条）
