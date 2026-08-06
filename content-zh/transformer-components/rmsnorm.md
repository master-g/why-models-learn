---
title: "RMSNorm：只按均方根缩放，不减均值"
tags: ["why-models-learn"]
---

RMSNorm（Root Mean Square Layer Normalization，均方根归一化）沿一个 token 的特征轴计算均方根，再用这个标量缩放全部特征。它不减去均值，也不使用 LayerNorm 的 $\beta$ 偏移参数；规范形式只有一条逐特征可学习的 $\gamma$。输入为 $x\in\mathbb R^{B\times T\times d}$ 时，RMSNorm 只在每个 $x_{b,t,:}$ 内归约，统计量不跨 batch、token 或历史 cache 共享。

这篇词条从均方根和 L2 范数的关系开始，核对 RMSNorm 的数值例子、尺度不变性、径向 Jacobian 和反向梯度，再处理 $\epsilon$、广播、Transformer 残差接口、混合精度、资源账本以及与 LayerNorm、ScaleNorm、BatchNorm 的边界。LayerNorm 的中心化、$\beta$ 和完整 Jacobian 见 [LayerNorm](../transformer-components/layernorm/)；残差相加前后的顺序见 [LayerNorm 与残差](../transformer-components/layernorm-residuals/)。

![RMSNorm 对一个 token 的特征平方求均值并开平方，再用均方根缩放和 gamma 做逐特征输出](/assets/transformer-components/svg/rmsnorm.1.svg)

## 先固定均方根统计对象

### 一个 token 的特征轴

设中间表示为

$$
x\in\mathbb R^{B\times T\times d}.
$$

固定 batch 索引 $b$ 和 token 索引 $t$，得到

$$
x_{b,t,:}
=
\begin{bmatrix}
x_{b,t,1}&\cdots&x_{b,t,d}
\end{bmatrix}^{\mathsf T}
\in\mathbb R^d.
$$

下面把这条特征向量简写为 $x$。RMSNorm 的统计集合是这 $d$ 个坐标。它不先计算

$$
\mu=\frac{1}{d}\sum_i x_i,
$$

也不构造 $x-\mu\mathbf 1$。它直接计算坐标平方的平均。

### 均方根、缩放和输出

均方为

$$
m
=\frac{1}{d}\sum_{i=1}^{d}x_i^2
=\frac{1}{d}x^{\mathsf T}x.
$$

加入正数 $\epsilon$ 后，均方根为

$$
r
=\sqrt{m+\epsilon}.
$$

规范 RMSNorm 输出为

$$
y_i
=\gamma_i\frac{x_i}{r},
\qquad
y=\gamma\odot\frac{x}{r},
$$

其中 $\gamma\in\mathbb R^d$ 是可学习的逐特征尺度。$\gamma$ 沿 batch 和 token 轴广播，输出 shape 与输入相同。

原始 RMSNorm 形式不包含 $\beta$。如果某个实现写成

$$
y=\gamma\odot\frac{x}{r}+\beta,
$$

它是带可学习偏移的 RMSNorm 变体。这个变体可以实现，但参数量、平移行为和 checkpoint 合同都不同，不能在名称相同的情况下默认两者等价。

### 均方根与 L2 范数

当 $\epsilon=0$ 且 $x\ne0$ 时，

$$
r
=\sqrt{\frac{1}{d}x^{\mathsf T}x}
=\frac{\lVert x\rVert_2}{\sqrt d}.
$$

因此

$$
\frac{x}{r}
=\sqrt d\,\frac{x}{\lVert x\rVert_2}.
$$

RMSNorm 把向量投到半径约为 $\sqrt d$ 的球面，再用 $\gamma$ 做逐坐标缩放。$\epsilon>0$ 时，这个球面关系只近似成立；它把零向量和低范数向量的增益限制在有限范围。

### 统计轴决定 token 是否耦合

| 规则 | 统计集合 | 是否减均值 | 是否需要运行统计 |
| --- | --- | --- | --- |
| RMSNorm | 一个 token 的特征轴 | 否 | 否 |
| LayerNorm | 一个 token 的特征轴 | 是 | 否 |
| BatchNorm | batch、空间或实现指定的轴 | 是 | 通常需要 |
| 错误的跨 token 归约 | token 轴或 token 与特征混合 | 取决于实现 | 取决于实现 |

RMSNorm 的 token 局部性与 LayerNorm 相同，中心化规则不同。batch size 改变不会改变固定 token 的均方根；自回归解码只输入一个新 token 时，归约仍只包含该 token 的 $d$ 个特征。

## 四维向量的数值例子

### 先算均方和均方根

取

$$
x=(1,2,3,4),
\qquad
d=4,
\qquad
\epsilon=10^{-5}.
$$

均方为

$$
m
=\frac{1^2+2^2+3^2+4^2}{4}
=\frac{30}{4}
=7.5.
$$

均方根为

$$
r
=\sqrt{7.5+10^{-5}}
=2.738614613267.
$$

当 $\gamma=(1,1,1,1)$ 时，标准化部分为

$$
\frac{x}{r}
=(0.365148128238,0.730296256476,
1.095444384714,1.460592512952).
$$

RMSNorm 不会把这个向量的均值移到零。其输出均值约为 $0.912870$；平方平均约为

$$
\frac{m}{m+\epsilon}
=0.999998666668.
$$

### $\gamma$ 只改变逐特征尺度

取

$$
\gamma=(1,2,0.5,-1).
$$

输出为

$$
y
=(0.365148128238,1.460592512952,
0.547722192357,-1.460592512952).
$$

RMSNorm 的均方约束作用于 $x/r$，不作用于仿射后的 $y$。$\gamma$ 可以让不同特征保留不同的幅度，也可以通过接近零的参数压低某个输出坐标。

### 与 LayerNorm 例子的差别

同一个 $x=(1,2,3,4)$，LayerNorm 会先减去均值 $2.5$，所以其标准化向量包含正负对称坐标。RMSNorm 保留原始坐标的共同偏移，四个坐标仍然全部为正。这个差异来自是否减去 $\mu$，不是来自均方根和标准差的数值精度。

| 项目 | RMSNorm | LayerNorm |
| --- | ---: | ---: |
| 统计均值 | 不计算 | $2.5$ |
| 统计量 | $m=7.5$ | $v=1.25$ |
| 中心化 | 无 | $x-\mu\mathbf 1$ |
| $\gamma$ | 有 | 有 |
| $\beta$ | 规范形式无 | 有 |
| 标准化向量的均值 | 约 $0.912870$ | $0$ |

## RMSNorm 的几何性质

### 公共平移不会被消除

令

$$
x'=x+a\mathbf 1.
$$

则

$$
\frac{x'}{\operatorname{RMS}(x')}
\ne
\frac{x}{\operatorname{RMS}(x)}
$$

通常成立。即使 $\epsilon=0$，公共平移也会改变 $\lVert x'\rVert_2$ 和每个坐标在向量方向中的比例。RMSNorm 的不变性不是 LayerNorm 的平移不变性。

用

$$
x=(1,2,3,4),
\qquad
x'=(4,5,6,7)
$$

作对照，两个向量的均方分别为 $7.5$ 和 $31.5$，缩放后的方向也不同。若测试声称 RMSNorm 能消除公共偏移，应先检查是否误用了 LayerNorm 公式。

### 正比例缩放在 $\epsilon=0$ 时不变

对 $a>0$，有

$$
m(ax)=a^2m(x),
\qquad
r(ax)=a r(x)
$$

在 $\epsilon=0$ 时成立，因此

$$
\frac{ax}{r(ax)}
=\frac{x}{r(x)}.
$$

如果 $a<0$，输出方向整体翻转：

$$
\frac{ax}{r(ax)}
=-\frac{x}{r(x)}.
$$

当 $\epsilon>0$ 时，

$$
r(ax)=\sqrt{a^2m(x)+\epsilon},
$$

正比例缩放不再完全抵消。方差很小或 $a$ 很小时，$\epsilon$ 的影响尤其明显。

### 特征置换需要同时置换 $\gamma$

令 $\Pi$ 为置换矩阵。RMSNorm 的均方不依赖坐标排列，因此

$$
\operatorname{RMS}_{\Pi\gamma}(\Pi x)
=
\Pi\operatorname{RMS}_{\gamma}(x).
$$

只置换输入而保持 $\gamma$ 不变，通常不会保持输出坐标的对应关系。特征坐标和参数坐标必须使用同一个排列。

## Jacobian：径向方向与切向方向

### 对均方根求微分

令

$$
m=\frac{1}{d}x^{\mathsf T}x,
\qquad
r=\sqrt{m+\epsilon}.
$$

则

$$
dm=\frac{2}{d}x^{\mathsf T}dx,
\qquad
dr=\frac{1}{2r}dm
=\frac{x^{\mathsf T}dx}{d\,r}.
$$

对 $z=x/r$ 应用商法则：

$$
\begin{aligned}
dz
&=\frac{1}{r}dx-\frac{x}{r^2}dr\\
&=\frac{1}{r}dx-\frac{xx^{\mathsf T}}{d\,r^3}dx\\
&=\frac{1}{r}
\left(
I-\frac{xx^{\mathsf T}}{d\,r^2}
\right)dx.
\end{aligned}
$$

因此标准化部分的 Jacobian 为

$$
J_z
=\frac{1}{r}
\left(
I-\frac{xx^{\mathsf T}}{d\,r^2}
\right).
$$

令 $\Gamma=\operatorname{diag}(\gamma)$，规范 RMSNorm 输出 $y=\Gamma z$ 的 Jacobian 为

$$
J_y
=\frac{\Gamma}{r}
\left(
I-\frac{xx^{\mathsf T}}{d\,r^2}
\right).
$$

外积 $xx^{\mathsf T}$ 使各个输出坐标相互耦合。只写成逐坐标乘以 $1/r$ 会漏掉均方变化对所有坐标的共同影响。

### 径向方向是一个近似零方向

对全 1 向量，通常没有

$$
J_z\mathbf 1=0.
$$

RMSNorm 没有减均值，所以公共平移不是被消除的方向。对径向方向 $x$，有

$$
J_zx
=\frac{\epsilon}{r^3}x.
$$

当 $\epsilon=0$ 时，$J_zx=0$，因为正比例缩放不改变归一化方向。当 $\epsilon>0$ 时，径向响应变成小但非零的量。切向方向 $v$ 满足 $x^{\mathsf T}v=0$，则

$$
J_zv=\frac{1}{r}v.
$$

这把 RMSNorm 的局部作用分成两类：沿当前向量方向主要由 $\epsilon$ 决定，正交方向统一缩放 $1/r$。逐特征 $\gamma$ 会再对这些方向做坐标缩放。

### 本例的数值 Jacobian

对 $x=(1,2,3,4)$、$\epsilon=10^{-5}$ 和 $\gamma=(1,2,0.5,-1)$，Jacobian 近似为：

|  | 输入 1 | 输入 2 | 输入 3 | 输入 4 |
| --- | ---: | ---: | ---: | ---: |
| 输出 1 | $0.352976540$ | $-0.024343176$ | $-0.036514764$ | $-0.048686352$ |
| 输出 2 | $-0.048686352$ | $0.632923552$ | $-0.146059057$ | $-0.194745409$ |
| 输出 3 | $-0.018257382$ | $-0.036514764$ | $0.127801918$ | $-0.073029528$ |
| 输出 4 | $0.048686352$ | $0.097372704$ | $0.146059057$ | $-0.170402720$ |

矩阵不是对角矩阵。输入第 1 个坐标的变化会影响均方，因而影响四个输出坐标。

### 反向梯度只需要一个标量归约

设上游梯度为

$$
g=\frac{\partial L}{\partial y},
\qquad
u=g\odot\gamma.
$$

定义

$$
\overline{ux}
=\frac{1}{d}\sum_{j=1}^{d}u_jx_j.
$$

由 $J_y^{\mathsf T}u$ 得到

$$
\frac{\partial L}{\partial x_i}
=\frac{1}{r}
\left(
u_i-\frac{x_i}{r^2}\overline{ux}
\right).
$$

它只需要计算一个标量 $\overline{ux}$。与 LayerNorm 的反向相比，它没有均值归约项；与逐元素缩放相比，它仍然保留均方耦合项。

取上游梯度 $g=(1,0,0,0)$，本例的输入梯度为

$$
\frac{\partial L}{\partial x}
\approx
(0.352976540192,-0.024343176092,
-0.036514764137,-0.048686352183).
$$

四个分量不需要和为零。RMSNorm 没有 LayerNorm 的公共平移不变性，因此输入梯度不受零和约束。

## $\gamma$、$\beta$ 与参数合同

### 规范 RMSNorm 只有 $\gamma$

规范形式的可训练参数为

$$
\gamma\in\mathbb R^d,
\qquad
N_{\mathrm{param}}=d.
$$

例如 $d=4096$ 时，参数量为 $4096$；若以 FP32 保存，参数本身占

$$
4096\times4\ \mathrm{bytes}
=16384\ \mathrm{bytes}
=16\ \mathrm{KiB}.
$$

LayerNorm 在相同宽度下通常有 $\gamma$ 和 $\beta$ 两条参数向量，参数量为 $2d$。把 RMSNorm 的参数量直接按 LayerNorm 乘二，会让 checkpoint 账本多出一条并不存在的偏移向量。

### $\gamma_i=0$ 会关闭一个输出坐标

若 $\gamma_i=0$，则

$$
y_i=0,
\qquad
\frac{\partial y_i}{\partial x}=0.
$$

这一坐标既不输出当前输入的信息，也不向输入传梯度。初始化通常令 $\gamma$ 接近 $\mathbf1$；训练后 $\gamma_i$ 的分布可以作为特征尺度诊断，但不能单独作为重要性结论。

### 带 $\beta$ 的变体必须单独标记

若实现显式保存 $\beta$ 并执行

$$
y=\gamma\odot\frac{x}{r}+\beta,
$$

则 $\beta$ 只是 RMSNorm 之后的逐特征偏移，不参与均方统计。它会改变参数量和公共平移行为，但不会把算子变成 LayerNorm；LayerNorm 还包含减均值和方差统计。

迁移 checkpoint 时应检查：

| 项目 | 规范 RMSNorm | 带偏移变体 |
| --- | --- | --- |
| 均值统计 | 无 | 无 |
| $\gamma$ | 有 | 有 |
| $\beta$ | 无 | 有 |
| 参数量 | $d$ | $2d$ |
| 输出公共偏移 | 由输入保留 | 还可由 $\beta$ 添加 |

## $\epsilon$ 处理零范数和低范数

### 零向量的边界

当 $x=0$ 时

$$
m=0,
\qquad
r=\sqrt{\epsilon},
\qquad
y=0
$$

对规范的无 $\beta$ RMSNorm 成立。$\epsilon$ 防止除零，但不会凭空生成非零输出。

若使用带 $\beta$ 的变体，零向量输出为

$$
y=\beta.
$$

这也是 padding 向量可能产生非零输出的原因之一；是否允许它进入后续 attention 或 pooling，需要由对应 mask 控制。

### 近零向量的增益

取

$$
x=(0.001,0,0,0).
$$

其均方为

$$
m=2.5\times10^{-7}.
$$

两种 $\epsilon$ 给出：

| $\epsilon$ | $r$ | 第一个标准化坐标 |
| ---: | ---: | ---: |
| $10^{-5}$ | $0.003201562119$ | $0.3123475238$ |
| $10^{-3}$ | $0.03162672920$ | $0.0316188245$ |

增大 $\epsilon$ 会降低近零输入的增益。不同实现只要 $\epsilon$ 不同，就会在低范数 token 上产生不同输出，即使输入、$\gamma$ 和归约轴完全相同。

### $\epsilon$ 必须位于平方根内部

规范尺度为

$$
r=\sqrt{m+\epsilon}.
$$

下面的形式通常都不等价：

$$
\sqrt{m+\epsilon}
\ne
\sqrt m+\epsilon,
\qquad
\sqrt{m+\epsilon}
\ne
\sqrt{m}+\sqrt{\epsilon}.
$$

实现、配置文件和参考测试必须使用同一位置。只在高范数随机输入上比较结果，可能看不出这个差异；零向量和近零向量是必要边界。

## Shape、广播与 normalized shape

### 三维序列输入

对 $x\in\mathbb R^{B\times T\times d}$，常见 shape 合同为：

| 张量或参数 | shape | 规则 |
| --- | --- | --- |
| 输入 $x$ | $(B,T,d)$ | 每个 $(b,t)$ 独立沿最后一轴计算均方 |
| 均方 $m$ | $(B,T,1)$ | 在特征轴上广播 |
| 均方根 $r$ | $(B,T,1)$ | 在特征轴上广播 |
| $\gamma$ | $(d)$ | 沿 batch 和 token 轴广播 |
| 输出 $y$ | $(B,T,d)$ | 与输入保持相同 shape |

均方和均方根保留长度为 $1$ 的特征轴，避免把 token 轴和特征轴混淆。若实现把统计量 squeeze 成 $(B,T)$，必须明确后续广播规则。

### 拆分后的多头表示

若张量为

$$
x\in\mathbb R^{B\times T\times H\times d_h},
$$

规范选择取决于模型把哪个轴定义为 normalized shape：

1. 只归约 $d_h$，每个 token、每个 head 有自己的均方根；
2. 归约 $(H,d_h)$，一个 token 的所有 head 特征共享一个均方根；
3. 合并 head 后归约 $d_{\mathrm{model}}=H d_h$，与常见 Transformer hidden 接口一致。

这三种做法的输出 shape 可以都保持不变，但数值不同。checkpoint 中的 $\gamma$ 长度和 forward 中的 head reshape 必须一起核对。

### 不要把 mask 混进最后一轴

RMSNorm 的 mask 不存在于规范公式中。padding mask、causal mask 和 loss mask 分别属于 attention 可见性或目标聚合。若把 mask 数值拼入特征向量再计算均方，统计集合就改变了；若把整个序列沿 token 轴一起归约，RMSNorm 的 token 局部性也会丢失。

## RMSNorm 在 Transformer 中的接口

### pre-norm 形式

设 RMSNorm 为 $R_l$，注意力和 FFN 子层分别为 $F_{\mathrm{attn}}$、$F_{\mathrm{ffn}}$。pre-norm block 可写成

$$
\begin{aligned}
x'&=x+F_{\mathrm{attn}}(R_1(x)),\\
y&=x'+F_{\mathrm{ffn}}(R_2(x')).
\end{aligned}
$$

RMSNorm 先改变 residual branch 看到的 token 特征尺度，再把子层修正加回 residual stream。它不把输入的公共均值强制移到零。

### post-norm 形式

同一个 RMSNorm 放在残差相加之后时，形式变为

$$
\begin{aligned}
x'&=R_1(x+F_{\mathrm{attn}}(x)),\\
y&=R_2(x'+F_{\mathrm{ffn}}(x')).
\end{aligned}
$$

对应的 Jacobian 顺序与 LayerNorm 的 pre/post 选择相同，只是 $J_R$ 使用本文的均方根 Jacobian。[Pre-norm 与 Post-norm](../transformer-components/pre-norm-vs-post-norm/) 讨论残差路径；不要把“使用 RMSNorm”和“采用 pre-norm”当作同一个配置。

### final RMSNorm

许多使用 RMSNorm 的语言模型在最后一个 block 后、输出投影前再放置一个独立的 final RMSNorm：

$$
h_{\mathrm{out}}=R_{\mathrm{final}}(x_L).
$$

它拥有自己的 $\gamma_{\mathrm{final}}$。不能用最后一个 block 内的 $\gamma$ 替代它，也不能因它没有 $\beta$ 就从 checkpoint 中删除整个归一化模块。

## 训练态、推理态与 cache

### 没有 running statistics

规范 RMSNorm 对同一个输入在训练态和评估态执行相同的均方归约：

$$
m_{b,t}
=\frac{1}{d}\sum_{i=1}^{d}x_{b,t,i}^2.
$$

它没有 BatchNorm 式 running mean 或 running variance。训练态和评估态如果产生差异，通常来自 dropout、随机深度或其他独立算子。

### batch size 为一不改变统计集合

RMSNorm 的统计集合大小是 $d$，不是 $B$。$B=1$ 时，只要特征宽度有效，均方根仍由当前 token 的特征给出。不能把 BatchNorm 在小 batch 下的限制直接转移到 RMSNorm。

### KV cache 不参与当前 token 的均方根

自回归解码第 $t$ 步可以只计算一个新 hidden：

$$
x_t\in\mathbb R^d.
$$

过去 token 的 K/V cache 服务于 attention 读取，不进入 $x_t$ 的均方计算。若先把 cache 与当前 hidden 拼接，再执行 RMSNorm，归约集合会随历史长度改变，训练和解码的接口不一致。

### padding 仍需由下游 mask 处理

规范 RMSNorm 会对 padding token 自身的特征计算均方根。零 padding 经过无 $\beta$ RMSNorm 后仍为零，但经过带 $\beta$ 的变体后可能非零；即使保持为零，也不能代替 attention、pooling 或 loss 的 mask。

| 位置 | RMSNorm 的职责 | 需要另行检查的职责 |
| --- | --- | --- |
| token hidden | 沿特征轴计算均方根 | padding 表示是否进入下游 |
| attention score | 不决定可见性 | causal 与 key padding mask |
| 序列池化 | 不决定有效分母 | 只聚合有效 token |
| token loss | 不决定目标集合 | loss mask 和分母 |

## 数值实现：平方归约和逐元素缩放

### 推荐的前向顺序

对每个 token 执行：

1. 把输入转换到指定的归约精度；
2. 计算 $m=\operatorname{mean}(x_i^2)$；
3. 计算 $r=\operatorname{rsqrt}(m+\epsilon)$；
4. 计算 $y_i=\gamma_i x_i r$；
5. 按输出合同写回目标 dtype。

组合公式为

$$
y_i
=\gamma_i x_i
\operatorname{rsqrt}\left(
\frac{1}{d}\sum_jx_j^2+\epsilon
\right).
$$

与 LayerNorm 相比，RMSNorm 不需要先求均值、中心化和中心化平方，通常少一个统计量和一组逐元素减法。但平方归约本身仍然需要稳定的累加精度。

### FP16 输入不代表 FP16 归约

混合精度执行时，应明确：

| 项目 | 需要固定的值 |
| --- | --- |
| 输入 dtype | FP16、BF16、FP32 或其他格式 |
| 平方累加 dtype | 计算 $x_i^2$ 和均方时的类型 |
| $\epsilon$ dtype | 加入均方前的可表示精度 |
| $\gamma$ dtype | 参数存储和乘法类型 |
| 输出 dtype | 逐元素缩放后的写回类型 |

对长度为 $4096$ 的特征直接用 FP16 累加平方，可能产生更大的舍入误差。常见实现用 FP32 做平方和归约，再把结果转换到输出 dtype。

### 先平方再求平均

RMSNorm 的均方是非负项平均：

$$
m=\frac{1}{d}\sum_i x_i^2.
$$

它不存在 LayerNorm 的大均值减小方差消去，但大幅输入仍可能在平方和中溢出，低幅输入仍可能在低精度中下溢。检查最大输入、平方累加 dtype 和 $\epsilon$ 的表示范围。

### 不要引入无定义的均值状态

RMSNorm 不需要 $\mu$、running mean 或跨 token 的状态。分布式实现可以在每个 token 内完成平方归约；额外跨 batch 或跨设备同步会改变 token 局部统计合同。

## 参数、激活与计算账本

### 长序列示例

取

$$
B=2,
\qquad
T=4096,
\qquad
d=4096.
$$

token 数为

$$
BT=8192,
$$

输入和输出元素数为

$$
BTd=33554432.
$$

规范 RMSNorm 参数量为

$$
d=4096.
$$

### 统计量和输出内存

每个 token 需要一个均方根或 inverse RMS。若以 FP32 保存一个统计标量：

$$
BT\times4
=32768\ \mathrm{bytes}
=32\ \mathrm{KiB}.
$$

若实现为 backward 额外保存均方和 inverse RMS 两个 FP32 标量，则是

$$
2BT\times4
=65536\ \mathrm{bytes}
=64\ \mathrm{KiB}.
$$

若输出为 FP16：

$$
BTd\times2
=67108864\ \mathrm{bytes}
=64\ \mathrm{MiB}.
$$

统计量数量取决于 kernel 保存的中间结果，不能把一个实现的 workspace 账本写成所有实现都相同。

### 计算量

每个 token 需要一次平方和归约，以及一次 inverse RMS 广播和逐元素乘法：

$$
O(BTd).
$$

RMSNorm 不生成 $T\times T$ attention matrix，也不需要对 batch 维度做统计归约。它通常比 LayerNorm 少均值归约、中心化和中心化平方，但实际速度仍受 kernel 融合、内存访问和 dtype 影响。

## RMSNorm、LayerNorm、ScaleNorm 与 BatchNorm

### LayerNorm 多了中心化和 $\beta$

LayerNorm 的标准化部分为

$$
\operatorname{LN}(x)
=\frac{x-\mu\mathbf1}{\sqrt{v+\epsilon}},
\qquad
\mu=\operatorname{mean}(x),
\qquad
v=\operatorname{mean}\bigl((x-\mu\mathbf1)^2\bigr).
$$

RMSNorm 为

$$
\operatorname{RMS}(x)
=\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}.
$$

两者都沿 token 特征轴归约，也都不需要 running statistics；一个减均值，一个保留均值。LayerNorm 的 $\beta$ 和 RMSNorm 规范形式的无偏移参数也属于独立差异。

| 算子 | 统计量 | 规范可学习参数 | 公共平移不变性 |
| --- | --- | --- | --- |
| RMSNorm | $\operatorname{mean}(x^2)$ | $\gamma$ | 否 |
| LayerNorm | $\operatorname{mean}(x)$ 与中心化方差 | $\gamma,\beta$ | 是 |
| BatchNorm | batch 或空间相关统计 | scale、shift 与运行统计 | 取决于模式 |

“归一化”不是足够的实现标识。加载模型时要读取 norm 类型、$\epsilon$、归约轴和参数 shape。

### ScaleNorm 只有一个全局尺度

一种相关的 ScaleNorm 形式使用一个标量 $g$：

$$
y=g\frac{x}{\lVert x\rVert_2+\epsilon}.
$$

RMSNorm 使用长度为 $d$ 的 $\gamma$，而 ScaleNorm 使用一个全局尺度；两者的参数量和逐特征表达能力不同。若把 RMSNorm 的 $\gamma$ 强制成单个标量，模型的参数合同已经改变。

### BatchNorm 使用跨样本统计

BatchNorm 的常见训练路径把多个样本或空间位置放入同一个统计集合，并在评估时读取运行统计。RMSNorm 对每个 token 独立计算均方根，没有这种训练/评估统计分支。小 batch、序列长度变化和分布式同步的影响不能跨算子直接类比。

## 失效模式

### 把 RMSNorm 写成 LayerNorm

症状是输出均值被强制到零、输入整体加常数后输出不变、checkpoint 需要一条不应存在的 $\beta$。检查 forward 是否出现 $\mu$ 或 $x-\mu\mathbf1$。

### 把 LayerNorm 的参数量复制过来

规范 RMSNorm 只有 $\gamma$。如果加载器期待 $\beta$，要确认模型是否使用带偏移变体；不能用全零 $\beta$ 静默补齐而不记录这一变化。

### 使用错误的归约轴

输入从 $(B,T,d)$ 重排为 $(B,d,T)$ 后，最后一轴变成时间轴。沿最后一轴计算会让同一特征通道的多个 token 共享均方根。固定一个 token、改变另一个 token 是定位这个错误的直接测试。

### 把公共平移当成不变量

RMSNorm 不减均值。对 $x$ 和 $x+a\mathbf1$ 计算输出，结果通常不同。若测试断言公共平移不变，测试本身使用了 LayerNorm 的性质。

### 误放 $\epsilon$

将 $\epsilon$ 放在平方根外或放在除法之后，会在零向量和近零向量上给出不同结果。用 $x=0$、$x=(0.001,0,0,0)$ 逐项核对 $r$。

### 低精度平方溢出或下溢

平方和在低精度下可能溢出，微小特征可能在平方时下溢。固定平方计算、累加和输出 dtype，并用大幅与小幅输入分别测试。

### 把 cache 拼进归约

当前 token 的 RMSNorm 不应读取过去 K/V cache。若输出随历史长度改变，检查 hidden 和 cache 的拼接位置。

### 把 padding mask 当成 RMSNorm 逻辑

RMSNorm 只处理当前特征轴。padding 是否可读、是否进入池化、是否进入 loss，必须由下游 mask 和分母合同决定。

### 把 pre-norm 当成 RMSNorm 的同义词

RMSNorm 是统计函数，pre-norm 是 residual block 中的调用位置。RMSNorm 可以放在 pre-norm 或 post-norm 结构中；checkpoint 和 forward 图必须同时记录两项。

## 一个可复算的核验协议

### 前向数值

使用 $x=(1,2,3,4)$、$d=4$、$\epsilon=10^{-5}$、$\gamma=(1,2,0.5,-1)$，应得到：

| 项目 | 参考值 |
| --- | ---: |
| $m$ | $7.5$ |
| $r$ | $2.738614613267$ |
| $z_1$ | $0.365148128238$ |
| $y_2$ | $1.460592512952$ |
| $\operatorname{mean}(z)$ | $0.912870320595$ |
| $\operatorname{mean}(z^2)$ | $0.999998666668$ |

这些值能分别定位平方归约、$\epsilon$ 和 $\gamma$ 广播错误。

### 方向与局部性

执行四组检查：

1. 将输入乘以正标量，在 $\epsilon=0$ 的参考路径中检查输出不变；
2. 将输入整体加常数，检查输出通常改变；
3. 改变同 batch 或同序列的另一个 token，检查当前 token 输出不变；
4. 改变归约轴，检查局部性测试可以失败。

这四组检查把尺度性质、公共平移性质和 token 局部分开。

### 反向中心差分

对每个输入坐标用

$$
\frac{\partial L}{\partial x_j}
\approx
\frac{L(x+h e_j)-L(x-h e_j)}{2h}
$$

核对解析梯度。本文的四维例子、$\gamma=(1,2,0.5,-1)$ 和中心差分步长 $h=10^{-6}$ 给出的 Jacobian 最大绝对误差约为

$$
9.98\times10^{-11}.
$$

阈值必须随 dtype、步长和参考精度记录。不能用高阈值掩盖外积项缺失。

### 执行合同

| 检查 | 通过条件 |
| --- | --- |
| 统计量 | $m=\operatorname{mean}(x^2)$，不计算均值中心化 |
| $\epsilon$ | 位于 $m+\epsilon$ 的平方根内部 |
| 归约轴 | 只包含当前 token 的特征 |
| 参数 | 规范形式只有长度为 $d$ 的 $\gamma$ |
| 训练/评估 | 不读取 running statistics |
| 解码 | K/V cache 不进入当前 token 的均方根 |
| 精度 | 平方、累加、参数和输出 dtype 已记录 |
| 残差位置 | RMSNorm 与 pre/post-norm 分开记录 |

当这些条件都能由输入、配置、参数或 forward 图直接回答时，才可以把实现视为与目标 RMSNorm 合同一致。

## 相关词条

- [LayerNorm](../transformer-components/layernorm/)
- [LayerNorm 与残差](../transformer-components/layernorm-residuals/)
- [Pre-norm 与 Post-norm](../transformer-components/pre-norm-vs-post-norm/)
- [残差流](../transformer-components/residual-streams/)
- [混合精度训练](../training-nn/mixed-precision/)
- [批量归一化](../training-nn/batch-normalization/)
