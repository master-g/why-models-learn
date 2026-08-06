---
title: "LayerNorm：按 token 的特征轴归一化"
tags: ["why-models-learn"]
---

LayerNorm（Layer Normalization，层归一化）把一个 token 的特征向量作为统计集合，先减去该向量自己的均值，再除以自己的标准差，最后用每个特征一个可学习的缩放参数和偏移参数恢复表达能力。输入为 $x\in\mathbb R^{B\times T\times d}$ 时，LayerNorm 只沿最后的特征轴 $d$ 归约；不同 batch、不同 token 之间不共享即时均值和方差，也不维护 BatchNorm 式的 running statistics。

这篇词条从一个 token 的向量公式开始，推导 LayerNorm 的不变性、局部 Jacobian 和反向梯度，再把统计轴、$\epsilon$、广播、Transformer 接口、混合精度和内存账本写成可检查的合同。残差相加前后的放置顺序见 [LayerNorm 与残差](../transformer-components/layernorm-residuals/)；这里关注归一化算子本身。

![LayerNorm 沿单个 token 的特征轴计算均值和方差，再执行标准化与可学习的缩放平移](/assets/transformer-components/svg/layernorm.1.svg)

## 归一化对象是一个 token 的特征向量

### 从三维输入取出一个统计集合

设 Transformer 中间表示为

$$
x\in\mathbb R^{B\times T\times d},
$$

其中 $B$ 是 batch size，$T$ 是序列长度，$d$ 是每个 token 的特征宽度。固定 $b$ 和 $t$ 后，取出

$$
x_{b,t,:}
=
\begin{bmatrix}
x_{b,t,1}&\cdots&x_{b,t,d}
\end{bmatrix}^{\mathsf T}
\in\mathbb R^d.
$$

LayerNorm 的统计集合是这 $d$ 个特征。它不把 $T$ 个 token 拼成一个集合，也不把 batch 中的 $B$ 个样本拼成一个集合。为简化记号，下面把这个向量写成 $x$。

### 均值、方差、标准化和仿射输出

均值为

$$
\mu
=\frac{1}{d}\sum_{i=1}^{d}x_i.
$$

中心化向量为

$$
c=x-\mu\mathbf 1,
$$

其中 $\mathbf 1\in\mathbb R^d$ 是全 1 向量。LayerNorm 使用分母为 $d$ 的方差：

$$
v
=\sigma^2
=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
=\frac{1}{d}c^{\mathsf T}c.
$$

加入正数 $\epsilon$ 后，尺度为

$$
s=\sqrt{v+\epsilon}.
$$

标准化向量和最终输出分别为

$$
z=\frac{c}{s},
\qquad
y=\operatorname{LN}_{\gamma,\beta}(x)
=\gamma\odot z+\beta,
$$

其中 $\gamma,\beta\in\mathbb R^d$ 是可学习向量，$\odot$ 表示逐元素乘法。对三维输入，$\gamma$ 和 $\beta$ 沿 $B$、$T$ 轴广播，输出 shape 仍为 $(B,T,d)$。

这里的“层”不表示跨层统计，也不表示把整个网络的激活拼起来。它是一个沿指定特征轴执行的向量归一化算子。

### 分母是 $d$，不是 $d-1$

LayerNorm 的方差是当前向量的总体二阶中心矩：

$$
v=\frac{1}{d}\sum_i(x_i-\mu)^2.
$$

统计学中估计总体方差时可以使用 $d-1$ 的样本方差，但那是另一种估计约定。把实现中的 $d$ 改成 $d-1$ 会同时改变 $s$、$z$、反向 Jacobian 和后续层看到的尺度。归一化算子不能只按“方差通常除以样本数减一”的记忆替换分母。

### 统计轴决定 token 是否耦合

| 规则 | 当前统计集合 | 不同 token 是否共享即时统计量 | 是否需要运行统计 |
| --- | --- | --- | --- |
| LayerNorm | 一个 token 的特征轴 | 否 | 否 |
| RMSNorm | 一个 token 的特征轴 | 否 | 否 |
| BatchNorm | batch、空间或实现指定的轴 | 通常是 | 训练态更新，评估态读取 |
| 错误的跨 token 归约 | token 轴或 token 与特征混合 | 是 | 取决于实现 |

LayerNorm 的 token 局部性有两个直接结果。第一，batch size 改变不会改变同一个 token 的即时统计量。第二，自回归解码只输入一个新 token 时，归约集合仍然是这个 token 的 $d$ 个特征，而不是历史 cache 的全部 token。

## 四维向量把每一步算清楚

### 均值、中心化向量和方差

取

$$
x=(1,2,3,4),
\qquad
d=4,
\qquad
\epsilon=10^{-5}.
$$

均值为

$$
\mu
=\frac{1+2+3+4}{4}
=2.5.
$$

因此

$$
c=x-\mu\mathbf 1
=(-1.5,-0.5,0.5,1.5).
$$

方差为

$$
v
=\frac{1.5^2+0.5^2+0.5^2+1.5^2}{4}
=1.25.
$$

加入 $\epsilon$ 后

$$
s
=\sqrt{1.25+10^{-5}}
=1.118038460877.
$$

标准化向量为

$$
z
=\frac{c}{s}
=(-1.341635419969,-0.447211806656,
0.447211806656,1.341635419969).
$$

### 仿射参数改变输出坐标

若先取 $\gamma=(1,1,1,1)$、$\beta=(0,0,0,0)$，输出就是 $z$。为了把可学习参数的作用单独显示出来，取

$$
\gamma=(2,1,0.5,-1),
\qquad
\beta=(0,0,1,0).
$$

逐坐标相乘和相加得到

$$
y
=(-2.683270839938,-0.447211806656,
1.223605903328,-1.341635419969).
$$

$z$ 的均值接近 $0$、二阶矩接近 $1$，但 $y$ 不必满足这两个条件。归一化约束作用于仿射之前的 $z$；$\gamma$ 和 $\beta$ 为后续子层提供恢复尺度和偏移的自由度。

### 三个不变量检查

由中心化定义，标准化向量满足

$$
\frac{1}{d}\sum_i z_i=0
$$

在 $\epsilon>0$ 时，二阶矩为

$$
\frac{1}{d}\sum_i z_i^2
=\frac{v}{v+\epsilon},
$$

而不是严格的 $1$。本例中

| 检查项 | 数值 | 计算含义 |
| --- | ---: | --- |
| $\operatorname{mean}(z)$ | $0$ | 中心化后四个坐标的和为零 |
| $\operatorname{mean}(z^2)$ | $0.999992000064$ | $1.25/(1.25+10^{-5})$ |
| $\lVert z\rVert_2$ | $1.999992000048$ | 四维向量的理想范数为 $\sqrt4=2$ |
| $\operatorname{mean}(y)$ | $-0.812129$ | 仿射参数改变了输出均值 |

如果测试只断言输出 shape 或只断言均值为零，就会漏掉分母、$\epsilon$、$\gamma$ 和 $\beta$ 的错误。

## LayerNorm 的不变性与等变性

### 加上常数向量会被消除

对任意标量 $a$，令

$$
x'=x+a\mathbf 1.
$$

则

$$
\mu'=\mu+a,
\qquad
c'=x'-\mu'\mathbf 1=c,
\qquad
v'=v.
$$

所以在 $\gamma,\beta,\epsilon$ 相同的情况下

$$
\operatorname{LN}_{\gamma,\beta}(x+a\mathbf 1)
=\operatorname{LN}_{\gamma,\beta}(x).
$$

这个性质说明 LayerNorm 不会把所有特征同时增加的公共偏移传给输出。它不是对任意平移都不变；只有沿全 1 方向的平移被消除，改变各坐标之间的差异仍然会改变输出。

本例把输入改成 $(4,5,6,7)$，均值变为 $5.5$，中心化向量仍为 $(-1.5,-0.5,0.5,1.5)$，因此 $z$ 和带相同 $\gamma,\beta$ 的 $y$ 完全不变。

### 正比例缩放受 $\epsilon$ 影响

对 $a>0$，有

$$
x'=ax,
\qquad
c'=ac,
\qquad
v'=a^2v,
$$

因此

$$
\operatorname{LN}_{\gamma,\beta}(ax)
=
\gamma\odot
\frac{ac}{\sqrt{a^2v+\epsilon}}
\,+\beta.
$$

当 $\epsilon=0$ 时，正比例缩放会完全抵消。当 $\epsilon>0$ 时，分母中的固定项不随 $a$ 缩放，输出会有小幅变化；当 $v$ 很小时，这个变化会变得明显。负比例缩放还会翻转中心化向量的方向。

因此，“LayerNorm 消除了尺度”需要附带两个条件：尺度沿同一个 token 的全部特征共同变化，并且 $\epsilon$ 相对于方差可以忽略。

### 特征置换需要同时置换参数

令 $\Pi$ 是一个置换矩阵。由于均值和方差不依赖特征排列，

$$
\operatorname{LN}_{\Pi\gamma,\Pi\beta}(\Pi x)
=
\Pi\operatorname{LN}_{\gamma,\beta}(x).
$$

如果只置换输入而不置换 $\gamma,\beta$，这个等式通常不成立。实际模型中的特征索引带有固定语义，参数和输入的坐标合同必须同时保持。

## Jacobian：一个特征的改变会影响所有输出

### 先对均值和方差求微分

定义投影矩阵

$$
P=I-\frac{1}{d}\mathbf 1\mathbf 1^{\mathsf T}.
$$

因为 $c=Px$，所以

$$
d\mu=\frac{1}{d}\mathbf 1^{\mathsf T}dx,
\qquad
dc=Pdx.
$$

对方差

$$
v=\frac{1}{d}c^{\mathsf T}c
$$

求微分，得到

$$
dv
=\frac{2}{d}c^{\mathsf T}dc
=\frac{2}{d}c^{\mathsf T}dx.
$$

因为 $s=(v+\epsilon)^{1/2}$，

$$
ds
=\frac{1}{2s}dv
=\frac{c^{\mathsf T}dx}{d\,s}.
$$

### 标准化的 Jacobian

对 $z=c/s$ 应用乘积法则：

$$
\begin{aligned}
dz
&=\frac{1}{s}dc-\frac{c}{s^2}ds\\
&=\frac{1}{s}Pdx
-\frac{c\,c^{\mathsf T}}{d\,s^3}dx\\
&=\frac{1}{s}
\left(
P-\frac{cc^{\mathsf T}}{d\,s^2}
\right)dx.
\end{aligned}
$$

所以不含仿射参数的局部 Jacobian 为

$$
J_z
=\frac{1}{s}
\left(
I-\frac{1}{d}\mathbf 1\mathbf 1^{\mathsf T}
-\frac{cc^{\mathsf T}}{d\,s^2}
\right).
$$

写成输出 $y=\gamma\odot z+\beta$ 的 Jacobian，令 $\Gamma=\operatorname{diag}(\gamma)$，则

$$
J_y
=\frac{\Gamma}{s}
\left(
I-\frac{1}{d}\mathbf 1\mathbf 1^{\mathsf T}
-\frac{cc^{\mathsf T}}{d\,s^2}
\right).
$$

矩阵中的第一项去掉公共平移，第二个外积项传播方差变化。只实现逐坐标缩放而不实现这两个耦合项，会得到错误的反向梯度。

### 本例的数值 Jacobian

对上一节的 $x$、$\epsilon$ 和

$$
\gamma=(2,1,0.5,-1)
$$

代入公式，得到近似矩阵：

|  | 输入 1 | 输入 2 | 输入 3 | 输入 4 |
| --- | ---: | ---: | ---: | ---: |
| 输出 1 | $0.536660608$ | $-0.715536744$ | $-0.178886869$ | $0.357763006$ |
| 输出 2 | $-0.357768372$ | $0.626096887$ | $-0.178885080$ | $-0.089443435$ |
| 输出 3 | $-0.044721717$ | $-0.089442540$ | $0.313048444$ | $-0.178884186$ |
| 输出 4 | $-0.178881503$ | $0.089443435$ | $0.357768372$ | $-0.268330304$ |

每一行的和在数值误差范围内为零，因为输入沿 $\mathbf 1$ 方向的变化会被均值消除。矩阵不是对角矩阵；输入第 1 个坐标的变化会通过均值和方差影响四个输出坐标。

### $\epsilon$ 改变第二个零方向

当 $\epsilon=0$ 且 $c\ne0$ 时，LayerNorm 对正比例缩放也不变，因此

$$
J_z\mathbf 1=0,
\qquad
J_zc=0.
$$

加入 $\epsilon>0$ 后，第一条仍严格成立，但第二条变为

$$
J_yc
=\frac{\epsilon}{s^3}\Gamma c.
$$

所以 $\epsilon$ 把尺度方向从精确零方向变成小但非零的局部响应。方差远大于 $\epsilon$ 时，这个响应很小；方差接近零时，它不能忽略。若所有 $\gamma_i\ne0$，$\mathbf 1$ 方向至少保留一个零方向；若某个 $\gamma_i=0$，输出坐标本身还会引入额外退化。

### 反向梯度可以压缩成三个归约

设上游梯度为

$$
g=\frac{\partial L}{\partial y},
$$

令

$$
u=g\odot\gamma,
\qquad
\bar u=\frac{1}{d}\sum_i u_i,
\qquad
\overline{uz}=\frac{1}{d}\sum_i u_i z_i.
$$

则反向到输入的梯度为

$$
\frac{\partial L}{\partial x_i}
=\frac{1}{s}
\left(
u_i-\bar u-z_i\overline{uz}
\right).
$$

这个形式只需要计算两个标量归约 $\bar u$ 和 $\overline{uz}$，不需要显式构造 $d\times d$ 的 Jacobian。它仍然包含均值耦合和方差耦合。

用本例的 $\gamma=(2,1,0.5,-1)$，取上游梯度 $g=(1,0,0,0)$，得到

$$
\frac{\partial L}{\partial x}
\approx
(0.536660607786,-0.715536744051,
-0.178886869262,0.357763005527).
$$

这四个分量之和为零，与平移不变性相符。有限差分核验时，应比较这个输入梯度和每个坐标的中心差分，而不是只比较梯度范数。

## $\gamma$ 和 $\beta$ 恢复可学习的尺度

### 参数量是 $2d$

每个 LayerNorm 实例有两条长度为 $d$ 的参数向量：

$$
\gamma\in\mathbb R^d,
\qquad
\beta\in\mathbb R^d.
$$

因此可训练参数量为

$$
N_{\mathrm{param}}=d+d=2d.
$$

例如 $d=4096$ 时，参数量为 $8192$。如果参数以 FP32 保存，只计算这两条参数向量本身需要

$$
8192\times4\ \mathrm{bytes}=32768\ \mathrm{bytes}=32\ \mathrm{KiB}.
$$

优化器状态、梯度和参数副本属于训练系统的额外账本，不能把它们隐含在 LayerNorm 参数量中。

### $\gamma$ 为零会关闭一个特征的局部输出

若 $\gamma_i=0$，则

$$
y_i=\beta_i,
\qquad
\frac{\partial y_i}{\partial x}=0.
$$

该坐标的前向输出成为常数，且不会把当前输入的梯度传回。初始化通常让 $\gamma$ 接近 $1$、$\beta$ 接近 $0$，但训练过程中某些 $\gamma_i$ 变小仍可能改变表示和梯度的有效秩。

### $\beta$ 不是新的统计量

$\beta$ 只在标准化之后逐特征相加。它不会参与 $\mu$ 或 $v$ 的计算，也不会在 batch、序列或设备之间聚合。把 $\beta$ 放进统计集合会改变 LayerNorm 的定义；把同一个标量 $\beta$ 广播到所有特征则减少了模型的表达自由度。

## $\epsilon$ 是数值合同的一部分

### 常数向量的边界

若

$$
x=(a,a,\ldots,a),
$$

则 $\mu=a$、$c=0$、$v=0$，所以

$$
s=\sqrt{\epsilon},
\qquad
z=0,
\qquad
y=\beta.
$$

没有 $\epsilon$ 时，常数向量会产生除零。$\epsilon$ 不是只在异常输入上触发的开关；它定义了方差很小时的实际增益 $1/\sqrt{v+\epsilon}$。

### 近似常数向量的数值例子

取

$$
x=(1,1,1,1.001).
$$

此时

$$
\mu=1.00025,
\qquad
v=1.875\times10^{-7}.
$$

两种 $\epsilon$ 给出不同的尺度和标准化结果：

| $\epsilon$ | $s=\sqrt{v+\epsilon}$ | 前三个 $z_i$ | 最后一个 $z_4$ |
| ---: | ---: | ---: | ---: |
| $10^{-5}$ | $0.003191786334$ | $-0.0783260450$ | $0.2349781350$ |
| $10^{-3}$ | $0.03162574110$ | $-0.0079049531$ | $0.0237148593$ |

较大的 $\epsilon$ 把低方差输入的增益压得更低。两个实现即使使用相同的 $\gamma,\beta$，只要 $\epsilon$ 不同，也不是同一个函数。

### $\epsilon$ 的位置不能移动

正确的尺度是

$$
s=\sqrt{v+\epsilon}.
$$

下面两个式子通常不等价：

$$
\sqrt{v+\epsilon}
\ne
\sqrt v+\epsilon,
\qquad
\sqrt{v+\epsilon}
\ne
\sqrt{v}+\sqrt{\epsilon}.
$$

在 $v=0$ 时，第一种写法得到 $\sqrt\epsilon$，第二种写法得到 $\epsilon$，第三种才偶然相同；在一般 $v>0$ 时三者都不同。配置文件、checkpoint 和 kernel 必须使用同一 $\epsilon$ 约定。

## Shape、广播与归约轴

### 三维 Transformer 输入

对 $x\in\mathbb R^{B\times T\times d}$，最常见的合同如下：

| 张量或参数 | shape | 归约或广播规则 |
| --- | --- | --- |
| 输入 $x$ | $(B,T,d)$ | 每个 $(b,t)$ 独立沿最后一轴归约 |
| 均值 $\mu$ | $(B,T,1)$ | 在特征轴上广播回输入 |
| 方差 $v$ | $(B,T,1)$ | 在特征轴上广播回输入 |
| $\gamma,\beta$ | $(d)$ | 沿 batch 和 token 轴广播 |
| 输出 $y$ | $(B,T,d)$ | 与输入保持相同 shape |

均值和方差保留长度为 $1$ 的特征轴，便于执行逐元素减法和除法。实现如果把它们 squeeze 成 B,T，后续广播方向必须明确；否则可能把 token 轴和特征轴错配。

### 多头表示中的 normalized shape

若张量已经拆成

$$
x\in\mathbb R^{B\times T\times H\times d_h},
$$

归约哪个轴取决于实现合同：

1. 只沿 $d_h$ 归约，会为每个 head、每个 token 独立计算统计量；
2. 沿 $(H,d_h)$ 一起归约，会把一个 token 的所有 head 特征放入同一个统计集合；
3. 只沿最后一轴但把参数写成长度 $H d_h$，会发生 shape 不匹配或错误重排。

Transformer 常见的 LayerNorm 位于合并后的 $d_{\mathrm{model}}$ 特征轴上，而不是把 attention head 当作 batch。拆分和合并 head 的时机必须和归一化位置一起记录。

### 错误的跨 token 归约会制造耦合

如果误把 $x\in\mathbb R^{B\times T\times d}$ 沿 $T$ 轴求均值，某个 token 的输出就会依赖同一序列中其他 token 的数值。这个算子不再是标准 LayerNorm；在自回归场景中，它还会让当前 token 的输出间接依赖未来 padding 或未来 token。

因此，测试应当固定一个 token，改变同 batch 或同序列中另一个 token，检查该 token 的 LayerNorm 输出是否保持不变。这个局部性测试比只检查最终模型 logits 更容易定位轴错误。

## LayerNorm 在 Transformer 中处理什么

### 它通常进入 Q、K、V 的共同输入

设当前 block 输入为 $x$，注意力子层为 $F_{\mathrm{attn}}$，FFN 子层为 $F_{\mathrm{ffn}}$。pre-norm 的两个子层可以写成

$$
\begin{aligned}
x'&=x+F_{\mathrm{attn}}(\operatorname{LN}_1(x)),\\
y&=x'+F_{\mathrm{ffn}}(\operatorname{LN}_2(x')).
\end{aligned}
$$

当 $\operatorname{LN}_1(x)$ 进入多头注意力时，Q、K、V 的线性投影都从同一个归一化后的 token 表示产生，除非实现明确为某一路使用另一种归一化或缩放。

LayerNorm 不等于 attention mask。LayerNorm 决定每个 token 的特征统计，mask 决定 query 可以读取哪些 key；前者不能替代后者，后者也不应改变前者的特征归约轴。

### residual 顺序属于另一个选择

单个子层的两种抽象形式为

$$
\begin{aligned}
\text{pre-norm:}\quad&y=x+F(\operatorname{LN}(x)),\\
\text{post-norm:}\quad&y=\operatorname{LN}(x+F(x)).
\end{aligned}
$$

两者调用的是同一个 LayerNorm 定义，但 Jacobian 分别为

$$
\begin{aligned}
J_{\mathrm{pre}}&=I+J_FJ_{\mathrm{LN}},\\
J_{\mathrm{post}}&=J_{\mathrm{LN}}(I+J_F).
\end{aligned}
$$

这里的 $J_{\mathrm{LN}}$ 是本文推导的特征轴 Jacobian，$J_F$ 是注意力或 FFN 的局部 Jacobian。顺序、深层信号尺度和 final LayerNorm 的位置见 [LayerNorm 与残差](../transformer-components/layernorm-residuals/) 和 [Pre-norm 与 Post-norm](../transformer-components/pre-norm-vs-post-norm/)。

### final LayerNorm 仍是算子实例

许多 pre-norm Transformer 在最后一个 block 后、输出投影前再执行一次 LayerNorm：

$$
h_{\mathrm{out}}=\operatorname{LN}_{\mathrm{final}}(x_L).
$$

它拥有自己的 $\gamma_{\mathrm{final}},\beta_{\mathrm{final}}$，不能把 block 内的 $\gamma,\beta$ 复用到它。审计 checkpoint 时，要分别列出每个 block 的归一化参数和 final LayerNorm 参数。

## 训练态、评估态与自回归解码

### LayerNorm 不维护 running mean 和 running variance

对同一个输入，训练态和评估态的 LayerNorm 公式相同：

$$
(\mu,v)=\operatorname{stats}(x_{b,t,:}).
$$

没有 BatchNorm 式的运行 buffer，也不需要在评估前估计全数据集均值。训练态和评估态的差异通常来自 dropout、随机深度或其他独立算子，而不是 LayerNorm 的统计分支。

如果一个实现给 LayerNorm 增加 running mean，或者在评估态改用 batch 外部统计，它已经改变了算子语义。迁移 checkpoint 时，这类额外状态必须单独核对。

### batch size 为一仍然可以正常计算

LayerNorm 的统计集合大小是 $d$，不是 $B$。因此 $B=1$ 时，只要 $d>1$，均值和方差仍然由当前 token 的特征给出。把“batch size 为一时归一化不稳定”直接套用到 LayerNorm，混淆了 BatchNorm 的统计轴。

### KV cache 不改变当前 token 的统计集合

自回归解码第 $t$ 步可能只计算新 token 的 hidden：

$$
x_t\in\mathbb R^d.
$$

LayerNorm 仍然沿 $d$ 个特征计算 $\mu_t$ 和 $v_t$。过去 token 的 K/V cache 只服务于注意力读取，不参与当前 token 的 LayerNorm 统计。若实现把 cache 拼接回 hidden 后再归一化，输出就会依赖历史长度，且不再符合训练时的 token 局部合同。

### padding mask 不会自动跳过 LayerNorm

对 padding token 自身，LayerNorm 仍然会计算它的特征均值和方差。padding 是否能影响其他 token，取决于后续 attention、pooling 或 loss 是否正确使用 mask：

| 位置 | LayerNorm 的职责 | 需要另行检查的职责 |
| --- | --- | --- |
| padding token 自身 | 沿其特征轴计算统计 | padding 表示是否会进入下游 |
| attention score | 不参与 LayerNorm 统计 | key padding mask 和 causal mask |
| 序列池化 | 不改变 LayerNorm 公式 | pooling 的有效 token 分母 |
| token loss | 不改变 LayerNorm 公式 | loss mask 与分母 |

把 padding 置零不等于把它从所有聚合中删除。零向量经过带非零 $\beta$ 的 LayerNorm 后仍可能产生非零输出。

## 数值实现：先稳定归约，再逐元素仿射

### 推荐的计算顺序

对每个 token 的特征向量，执行以下顺序：

1. 把输入转换到指定的归约精度；
2. 计算特征均值 $\mu$；
3. 计算中心化值 $c_i=x_i-\mu$；
4. 计算 $v=\operatorname{mean}(c_i^2)$；
5. 计算 $r=\operatorname{rsqrt}(v+\epsilon)$；
6. 计算 $y_i=(\gamma_i c_i)r+\beta_i$。

对应公式为

$$
y_i
=\bigl(\gamma_i(x_i-\mu)\bigr)
\operatorname{rsqrt}\left(
\frac{1}{d}\sum_j(x_j-\mu)^2+\epsilon
\right)
+\beta_i.
$$

先中心化再平方，通常比直接用 $\operatorname{mean}(x^2)-\operatorname{mean}(x)^2$ 更容易控制大均值、小方差输入的消去误差。后者可以减少一次遍历，但两个接近的大数相减可能损失有效位。

### 归约精度与输出精度可以不同

混合精度执行时，常见合同是输入和输出使用 FP16 或 BF16，均值、方差和中间乘加使用 FP32。这样做不改变 LayerNorm 的数学归约轴，但会改变实际舍入路径。至少应记录：

| 项目 | 需要固定的值 |
| --- | --- |
| 输入 dtype | FP16、BF16、FP32 或其他格式 |
| 统计累加 dtype | 计算均值和方差时的类型 |
| $\epsilon$ dtype | 加入方差前的可表示精度 |
| 参数 dtype | $\gamma,\beta$ 的存储和计算类型 |
| 输出 dtype | 仿射后的写回类型 |

如果直接在 FP16 中累加长度为 $4096$ 的特征，均值和方差可能出现更大的舍入误差；如果 $\epsilon$ 在低精度中下溢为零，常数向量又会回到除零风险。

### 单遍公式不是无条件的替代

常见的单遍方差公式为

$$
v
=\operatorname{mean}(x^2)-\operatorname{mean}(x)^2.
$$

令 $x_i=C+\delta_i$，其中 $C$ 很大而 $\delta_i$ 很小。两个均值项都包含 $C^2$，相减时会丢掉表示 $\delta_i$ 所需的有效位。两遍中心化公式先减去 $\mu$，通常更适合 LayerNorm 的低方差输入。

如果 kernel 使用并行归约或 Welford 算法，也必须在相同输入、dtype 和 $\epsilon$ 下对照参考公式。算法名称不能替代数值误差测试。

### 不要把均值和方差写入跨 token 状态

LayerNorm 的归约可以在每个 token 内部并行完成。它不需要跨设备 all-reduce，也不需要把上一个 token 的 $\mu$ 或 $v$ 传给下一个 token。分布式实现若额外同步这些统计量，会制造跨 token 或跨 batch 的耦合。

## 参数、激活与计算账本

### 一个长序列的元素数

取

$$
B=2,
\qquad
T=4096,
\qquad
d=4096.
$$

token 数量为

$$
BT=8192,
$$

归一化输入元素数为

$$
BTd=33554432.
$$

每个 LayerNorm 的参数量仍只有

$$
2d=8192.
$$

这组数字区分了参数规模和激活规模：参数只沿特征轴存两条向量，激活则随 $B T d$ 增长。

### 统计量和输出的存储

每个 token 需要一个均值和一个方差。若统计量以 FP32 保存：

$$
2BT\times4
=65536\ \mathrm{bytes}
=64\ \mathrm{KiB}.
$$

若输出以 FP16 保存：

$$
BTd\times2
=67108864\ \mathrm{bytes}
=64\ \mathrm{MiB}.
$$

这不是完整 kernel 的峰值内存。输入、输出、残差、保存给 backward 的中心化值或 inverse standard deviation、workspace 和 allocator 对齐都可能增加峰值。账本必须区分“持久参数”“中间统计量”“输出激活”和“临时 workspace”。

### 计算量沿特征轴线性增长

每个 token 至少需要：

1. 对 $d$ 个特征做一次均值归约；
2. 对中心化平方做一次方差归约；
3. 对 $d$ 个特征做减法、乘法、逆平方根广播和仿射变换。

因此 LayerNorm 的算术量随 $d$ 和 token 数量 $BT$ 线性增长：

$$
O(BTd).
$$

它不生成 $T\times T$ 的 attention matrix。LayerNorm 可能成为 memory bandwidth 受限的逐元素算子，但不能因此把它的复杂度写成 attention 的二次复杂度。

## LayerNorm、RMSNorm 和 BatchNorm 的边界

### RMSNorm 删除了中心化步骤

RMSNorm 只计算均方根：

$$
r
=\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon},
\qquad
y_i=\gamma_i\frac{x_i}{r}.
$$

它没有 $\mu$ 和 $c=x-\mu\mathbf1$，所以不会消除公共平移。为只比较中心化差异，下面取 $\epsilon=0$。对 $x=(1,2,3,4)$，有

$$
r=\sqrt{\frac{1+4+9+16}{4}}
=\sqrt{7.5}
=2.738612787526.
$$

当 $\gamma=\mathbf1$ 时，RMSNorm 输出约为

$$
(0.3651483717,0.7302967433,
1.0954451150,1.4605934866).
$$

LayerNorm 和 RMSNorm 都沿 token 特征轴归约，也都不需要 running statistics；它们的中心化项不同，不能只通过“都有 norm”合并为同一个算子。

### BatchNorm 共享另一组统计对象

BatchNorm 的统计轴由数据布局和实现指定，常见情形会把 batch 中多个样本的同一 channel 放进统计集合，并在训练态维护 running mean 和 running variance。LayerNorm 则对每个 token 独立计算特征统计，不读取跨样本 buffer。

| 算子 | 主要统计集合 | 训练/评估统计分支 | 是否减均值 |
| --- | --- | --- | --- |
| LayerNorm | 单个 token 的特征轴 | 通常没有 | 是 |
| RMSNorm | 单个 token 的特征轴 | 通常没有 | 否 |
| BatchNorm | batch 或空间相关轴 | 通常有 | 是 |

[批量归一化](../training-nn/batch-normalization/) 的训练态和评估态合同不能移植到 LayerNorm；[RMSNorm](../transformer-components/rmsnorm/) 的均方根公式也不能用 LayerNorm 的中心化公式替换。

### 归一化选择和残差顺序是两条轴

选择 LayerNorm 或 RMSNorm，决定一个 token 的统计函数。选择 pre-norm 或 post-norm，决定这个统计函数在残差 block 中的调用位置。四种组合不是同一个设计：

| 统计函数 | pre-norm 形式 | post-norm 形式 |
| --- | --- | --- |
| LayerNorm | $x+F(\operatorname{LN}(x))$ | $\operatorname{LN}(x+F(x))$ |
| RMSNorm | $x+F(\operatorname{RMS}(x))$ | $\operatorname{RMS}(x+F(x))$ |

审计一个 checkpoint 时，至少要同时记录 norm 类型、归约轴、$\epsilon$、$\gamma,\beta$ 是否存在，以及它位于残差相加的哪一侧。

## 失效模式

### 把最后一轴当成实现细节

输入从 B,T,d 转成 B,d,T 后，最后一轴已经是时间轴。若代码仍调用“沿最后一轴的 LayerNorm”，统计集合就从一个 token 的特征改成了一个特征通道的时间片。用固定一个 token、改变另一个 token 的测试可以暴露这个错误。

### 使用 $d-1$ 的样本方差

症状是输出仍然近似零均值，但二阶矩、尺度和梯度出现稳定比例偏差。检查参考实现的分母，并在 $d=2$ 或 $d=4$ 的小向量上直接比较 $v$。

### 把 $\epsilon$ 放到平方根外

症状集中出现在低方差或常数 token。比较 $x=(1,1,1,1.001)$ 和 $x=(1,1,1,1)$，同时记录 $s$、$z$ 和是否出现 NaN。

### 只实现逐元素缩放

若反向代码只保留 $u_i/s$，就丢失了 $\bar u$ 和 $\overline{uz}$ 两个耦合项。用上面的 $g=(1,0,0,0)$ 例子比较四个输入梯度，可以直接看到其他坐标也有非零梯度。

### 广播了错误的参数 shape

将长度为 $d$ 的 $\gamma,\beta$ 广播到错误轴，或者把 H,d_h 的参数误用于只沿 $d_h$ 归约的输入，会产生静默的数值差异。测试应同时检查参数 shape、归约轴和输出 shape。

### 在低精度中完成统计归约

症状包括 batch 或序列长度改变后结果抖动、近似常数输入的输出偏差增大、不同 kernel 的误差超过预期。固定输入和累加 dtype，分别对照 FP32 参考值。

### 加入 running statistics

症状是同一输入在训练态和评估态产生不同 LayerNorm 输出，或者 batch size 改变后统计结果发生跨样本变化。检查模块是否注册了 running mean、running variance 或其他未写入算子定义的 buffer。

### 把 mask 当成统计过滤器

attention mask 不会自动从 LayerNorm 的特征集合中删除坐标。若需求是忽略 padding，应在相应的 attention、pooling 或 loss 聚合处处理，而不是随意修改 LayerNorm 的最后一轴。

### 把 norm 类型和残差位置混写

“使用 LayerNorm”只说明 $N$ 的定义，不说明是 $x+F(N(x))$ 还是 $N(x+F(x))$。加载 checkpoint 或复现论文时，必须同时检查 block 图、参数命名和 forward 顺序。

## 一个可复算的核验协议

### 先核对前向数值

使用 $x=(1,2,3,4)$、$d=4$、$\epsilon=10^{-5}$、$\gamma=(2,1,0.5,-1)$、$\beta=(0,0,1,0)$，应得到：

| 项目 | 参考值 |
| --- | ---: |
| $\mu$ | $2.5$ |
| $v$ | $1.25$ |
| $s$ | $1.118038460877$ |
| $z_1$ | $-1.341635419969$ |
| $y_3$ | $1.223605903328$ |
| $\operatorname{mean}(z)$ | $0$ |
| $\operatorname{mean}(z^2)$ | $0.999992000064$ |

这些检查把均值、方差、$\epsilon$、标准化和仿射参数分开定位。只比较最终向量的某一位，不能区分分母错误和参数广播错误。

### 再核对局部性与不变性

执行三组扰动：

1. 给同一个 token 的全部特征加常数，检查输出不变；
2. 只改变同 batch 的另一个 token，检查当前 token 输出不变；
3. 改变归约轴或转置输入，检查测试是否能失败。

第一组验证公共平移不变性，第二组验证 token 局部性，第三组验证测试没有只覆盖 shape。

### 用中心差分核对反向

对每个输入坐标 $x_j$，用小步长 $h$ 计算

$$
\frac{\partial L}{\partial x_j}
\approx
\frac{L(x+h e_j)-L(x-h e_j)}{2h}.
$$

或者直接对向量输出的 Jacobian 做中心差分。使用本文的四维例子和 $\gamma=(2,1,0.5,-1)$，独立中心差分得到的 Jacobian 最大绝对误差约为

$$
5.00\times10^{-10}.
$$

误差阈值应随 dtype、步长和参考实现精度记录。不能用 FP16 的宽松阈值掩盖归约轴或反向公式错误。

### 最后检查执行合同

核验清单如下：

| 检查 | 通过条件 |
| --- | --- |
| 分母 | 方差使用 $d$ |
| $\epsilon$ | 位于 $v+\epsilon$ 的平方根内部 |
| 统计轴 | 只包含当前 token 的特征 |
| 参数 | $\gamma,\beta$ 长度与归约轴相同 |
| 训练/评估 | 不读取 running statistics |
| 解码 | cache 不进入当前 token 的 LayerNorm 统计 |
| 精度 | 累加 dtype、参数 dtype、输出 dtype 已记录 |
| block 顺序 | norm 类型与 pre/post 位置分别记录 |

当这八项都能由输入、参数、模块配置或 forward 图直接回答时，才可以把一个 LayerNorm 实现视为与目标模型相同。

## 相关词条

- [LayerNorm 与残差](../transformer-components/layernorm-residuals/)
- [RMSNorm](../transformer-components/rmsnorm/)
- [Pre-norm 与 Post-norm](../transformer-components/pre-norm-vs-post-norm/)
- [残差流](../transformer-components/residual-streams/)
- [批量归一化](../training-nn/batch-normalization/)
- [混合精度训练](../training-nn/mixed-precision/)
- [自注意力](../attention/self-attention/)
