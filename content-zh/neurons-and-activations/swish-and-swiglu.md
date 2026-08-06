---
title: "Swish 与 SwiGLU：平滑门如何变成门控前馈层"
tags: ["why-models-learn"]
---

Swish 是把输入乘以 sigmoid 门的平滑激活，SwiGLU 则把这类门扩展为两条投影分支的逐坐标乘法：一条分支产生门，另一条分支携带值。Swish 的常用特例 SiLU 为 $\operatorname{SiLU}(x)=x\sigma(x)$；Transformer 的门控前馈层通常计算 $\operatorname{SiLU}(W_g\mathbf{x})\odot(W_u\mathbf{x})$，再用输出投影送回模型宽度。本文先推导 Swish 的导数和曲率，再从 GLU 的两分支结构推出 SwiGLU 的前向、反向、参数预算和核验边界。

![左侧比较 Swish 曲线与 sigmoid 门，右侧展示 SwiGLU 的门分支、值分支、逐坐标乘法和输出投影](/assets/neurons-and-activations/svg/swish-and-swiglu.1.svg)

## Swish 不是 sigmoid：输入还要乘上门

先记 sigmoid 为

$$
\sigma(t)=\frac{1}{1+e^{-t}}.
$$

带有斜率参数的 Swish 定义为

$$
\operatorname{Swish}_{\beta}(x)
=x\sigma(\beta x),
\qquad
\beta\geq0.
$$

$\beta=1$ 时的函数常写成 SiLU：

$$
\operatorname{SiLU}(x)
=x\sigma(x).
$$

这里的 sigmoid 只是门，不是最终输出。sigmoid 自身始终为正且小于 1，而 Swish 还乘以原输入，所以负输入可以产生负输出，正输入也可以大于 1。门的数值表示“缩放了多少”，但不应自动解释为概率或随机保留率。

当 $\beta>0$ 时，几个极限立即给出整体形状：

$$
\lim_{x\to+\infty}\operatorname{Swish}_{\beta}(x)=x,
\qquad
\lim_{x\to-\infty}\operatorname{Swish}_{\beta}(x)=0^{-},
\qquad
\operatorname{Swish}_{\beta}(0)=0.
$$

正输入很大时，sigmoid 门接近 1，输入几乎原样通过；负输入很大时，门接近 0，但 $x$ 仍为负，因此输出从负侧靠近 0。它和 ReLU 都有无界的正侧，却没有 ReLU 那样的硬折点和精确零输出。

## 导数和曲率说明它为什么不是单调门

令 $s=\sigma(\beta x)$。对 $x s$ 使用乘积法则，并利用

$$
\frac{\mathrm d}{\mathrm dx}\sigma(\beta x)
=\beta s(1-s),
$$

得到

$$
\begin{aligned}
\operatorname{Swish}_{\beta}'(x)
&=s+x\beta s(1-s)\\
&=\sigma(\beta x)
+\beta x\sigma(\beta x)\bigl(1-\sigma(\beta x)\bigr).
\end{aligned}
$$

在零点，

$$
\operatorname{Swish}_{\beta}'(0)=\frac12.
$$

继续求导：

$$
\begin{aligned}
\operatorname{Swish}_{\beta}''(x)
&=\beta s(1-s)
+\beta s(1-s)
+\beta^2x s(1-s)(1-2s)\\
&=\beta s(1-s)\bigl[2+\beta x(1-2s)\bigr].
\end{aligned}
$$

因此

$$
\operatorname{Swish}_{\beta}''(0)=\frac{\beta}{2}.
$$

对 SiLU，也就是 $\beta=1$，几个点的数值如下：

| $x$ | $\operatorname{SiLU}(x)$ | $\operatorname{SiLU}'(x)$ | $\operatorname{SiLU}''(x)$ |
| ---: | ---: | ---: | ---: |
| $-2$ | $-0.238406$ | $-0.090784$ | $0.050062$ |
| $-1$ | $-0.268941$ | $0.072329$ | $0.302366$ |
| $-0.5$ | $-0.188770$ | $0.260039$ | $0.441229$ |
| $0$ | $0$ | $0.500000$ | $0.500000$ |
| $0.5$ | $0.311230$ | $0.739961$ | $0.441229$ |
| $1$ | $0.731059$ | $0.927671$ | $0.302366$ |
| $2$ | $1.761594$ | $1.090784$ | $0.050062$ |

在 $x=-2$ 处导数仍为负，在 $x=-1$ 处已经转为正，所以 SiLU 的负侧不是单调递增的。数值求根得到极小点约为

$$
x_{\min}\approx-1.278465,
\qquad
\operatorname{SiLU}(x_{\min})\approx-0.278465.
$$

这不是一个需要避开的数学异常，而是门和输入相乘后的真实形状。若把 Swish 当成“处处递增的平滑 ReLU”，就会漏掉负侧的局部翻转；自定义算子、单调性假设和梯度检查都应覆盖这个区域。

## 和 GELU 的共同点不是相同的定义

GELU 使用标准正态 CDF 做门：

$$
\operatorname{GELU}(x)=x\Phi(x),
$$

而 SiLU 使用 sigmoid：

$$
\operatorname{SiLU}(x)=x\sigma(x).
$$

它们都有负侧小输出、零点导数 $1/2$ 和正侧近似线性，但门函数不同。用几个输入直接比较：

| $x$ | ReLU$(x)$ | GELU$(x)$ | SiLU$(x)$ | $\sigma(x)$ |
| ---: | ---: | ---: | ---: | ---: |
| $-2$ | $0$ | $-0.045500$ | $-0.238406$ | $0.119203$ |
| $-1$ | $0$ | $-0.158655$ | $-0.268941$ | $0.268941$ |
| $-0.5$ | $0$ | $-0.154269$ | $-0.188770$ | $0.377541$ |
| $0$ | $0$ | $0$ | $0$ | $0.500000$ |
| $0.5$ | $0.5$ | $0.345731$ | $0.311230$ | $0.622459$ |
| $1$ | $1$ | $0.841345$ | $0.731059$ | $0.731059$ |
| $2$ | $2$ | $1.954500$ | $1.761594$ | $0.880797$ |

GELU 的正侧输出在这些点更接近输入，SiLU 的门则由 logistic 曲线决定。二者都不是 sigmoid 输出头：把 $x\sigma(x)$ 或 $x\Phi(x)$ 的结果当作概率，需要另加有明确语义的归一化和损失接口。

## 标准正态输入经过 SiLU 后也不再零中心

设 $X\sim\mathcal N(0,1)$。对标准正态变量，Stein 恒等式为

$$
\mathbb E[Xf(X)]=\mathbb E[f'(X)].
$$

取 $f=\sigma$，就得到

$$
\begin{aligned}
\mathbb E[\operatorname{SiLU}(X)]
&=\mathbb E[X\sigma(X)]\\
&=\mathbb E[\sigma'(X)]\\
&=\mathbb E[\sigma(X)(1-\sigma(X))].
\end{aligned}
$$

数值积分给出

$$
\mathbb E[\operatorname{SiLU}(X)]
\approx0.206621.
$$

所以零均值的标准正态输入经过 SiLU 后，输出均值仍为正。这个结果和 sigmoid 的非零中心有相似的后果：后续层看到的不是一个自动保持零均值的信号。归一化、残差相加和偏置分析应记录真实的输出统计，而不能只看输入初始化。

## GLU 先把一个投影拆成门和值

门控线性单元 GLU 的核心不是一个新的逐分量函数，而是两条同宽分支的乘法。对两个向量 $\mathbf{a},\mathbf{b}$，原始形式可以写成

$$
\operatorname{GLU}(\mathbf{a},\mathbf{b})
=\mathbf{a}\odot\sigma(\mathbf{b}),
$$

其中 $\odot$ 表示逐坐标乘法。若输入为 $\mathbf{x}\in\mathbb R^d$，两条线性投影为

$$
\mathbf{u}=W_u\mathbf{x},
\qquad
\mathbf{g}=W_g\mathbf{x},
\qquad
W_u,W_g\in\mathbb R^{m\times d},
$$

那么 GLU 可以写成

$$
\mathbf{h}
=\mathbf{u}\odot\sigma(\mathbf{g}).
$$

$\mathbf{u}$ 是值分支，携带被传给下一层的幅度和符号；$\sigma(\mathbf{g})$ 是门分支，把每个中间坐标缩放到 $(0,1)$。这不是 dropout：每个输入和每个坐标的门都是确定性计算，而且所有坐标仍然参与反向传播，只是梯度可能被门缩小。

## SwiGLU 把 sigmoid 门换成 SiLU 门

SwiGLU 使用 Swish 或 SiLU 产生门分支。以最常见的 SiLU 为例：

$$
\operatorname{SwiGLU}(\mathbf{x})
=\operatorname{SiLU}(W_g\mathbf{x})
\odot(W_u\mathbf{x}).
$$

门分支的结果不再被限制在 $(0,1)$；它可以为负，也可以在正侧超过 1。因此 SwiGLU 的“门”是可学习的连续缩放，不是概率闸门。为了得到模型宽度为 $d$ 的输出，门控前馈层再加一个下投影：

$$
\begin{aligned}
\mathbf{g}&=W_g\mathbf{x},\\
\mathbf{u}&=W_u\mathbf{x},\\
\mathbf{h}&=\operatorname{SiLU}(\mathbf{g})\odot\mathbf{u},\\
\mathbf{y}&=W_d\mathbf{h},
\end{aligned}
$$

其中

$$
W_g,W_u\in\mathbb R^{m\times d},
\qquad
W_d\in\mathbb R^{d\times m}.
$$

普通两层前馈网络通常是

$$
\mathbf{y}=W_2\,\phi(W_1\mathbf{x}).
$$

它只有一条上投影和一条下投影；SwiGLU 则把上投影拆成门和值两条矩阵，再逐坐标相乘。这个乘法让一个分支可以依据当前输入，选择性地放大、压低或翻转另一分支的特征。

## 一个两维例子把门和值分开算

为了不把逐坐标乘法藏在黑盒里，取

$$
\mathbf{x}
=\begin{bmatrix}1\\-1\end{bmatrix},
\qquad
W_g=
\begin{bmatrix}1&0\\0&1\end{bmatrix},
\qquad
W_u=
\begin{bmatrix}2&0\\0&1\end{bmatrix}.
$$

于是

$$
\mathbf{g}=W_g\mathbf{x}
=\begin{bmatrix}1\\-1\end{bmatrix},
\qquad
\mathbf{u}=W_u\mathbf{x}
=\begin{bmatrix}2\\-1\end{bmatrix}.
$$

门和值分别是

$$
\operatorname{SiLU}(\mathbf{g})
=\begin{bmatrix}0.731059\\-0.268941\end{bmatrix},
\qquad
\mathbf{h}
=\operatorname{SiLU}(\mathbf{g})\odot\mathbf{u}
=\begin{bmatrix}1.462117\\0.268941\end{bmatrix}.
$$

第二个坐标尤其值得看：门为负，值也为负，两个负号相乘后得到正的中间输出。这个现象说明 SwiGLU 的门不是“只允许保留或抑制”的概率，它可以改变值分支的符号。

## 反向传播有两条独立的梯度路径

设上游梯度为 $\mathbf{q}=\partial L/\partial\mathbf{h}$，并记

$$
\mathbf{s}=\operatorname{SiLU}(\mathbf{g}),
\qquad
\mathbf{d}=\operatorname{SiLU}'(\mathbf{g}).
$$

由 $\mathbf{h}=\mathbf{s}\odot\mathbf{u}$，对两个中间分支分别求导：

$$
\frac{\partial L}{\partial\mathbf{g}}
=\mathbf{q}\odot\mathbf{u}\odot\mathbf{d},
\qquad
\frac{\partial L}{\partial\mathbf{u}}
=\mathbf{q}\odot\mathbf{s}.
$$

再通过两个线性投影得到

$$
\frac{\partial L}{\partial\mathbf{x}}
=W_g^{\mathsf T}\frac{\partial L}{\partial\mathbf{g}}
+W_u^{\mathsf T}\frac{\partial L}{\partial\mathbf{u}}.
$$

上面的两维例子取

$$
\mathbf{q}
=\begin{bmatrix}0.5\\-0.25\end{bmatrix}.
$$

在 $\mathbf{g}=(1,-1)$ 处，

$$
\operatorname{SiLU}'(\mathbf{g})
=\begin{bmatrix}0.927671\\0.072329\end{bmatrix}.
$$

所以

$$
\frac{\partial L}{\partial\mathbf{g}}
\approx
\begin{bmatrix}0.927671\\0.018082\end{bmatrix},
\qquad
\frac{\partial L}{\partial\mathbf{u}}
\approx
\begin{bmatrix}0.365529\\0.067235\end{bmatrix},
$$

并且

$$
\frac{\partial L}{\partial\mathbf{x}}
\approx
\begin{bmatrix}1.658729\\0.085318\end{bmatrix}.
$$

若门分支在某个坐标上接近 0，值分支的梯度 $\mathbf{q}\odot\mathbf{s}$ 会变小；若值分支接近 0，门分支的梯度 $\mathbf{q}\odot\mathbf{u}\odot\mathbf{d}$ 会变小。门控结构不是免费地保留两条完整梯度，而是让两条路径互相调制。

## GEGLU、ReGLU 和 SwiGLU 只替换门函数

可以把门函数抽象为 $\varphi$：

$$
\operatorname{GLU}_{\varphi}(\mathbf{x})
=\varphi(W_g\mathbf{x})\odot(W_u\mathbf{x}).
$$

常见变体为

| 变体 | 门函数 $\varphi$ | 门的范围或形状 | 需要检查的边界 |
| --- | --- | --- | --- |
| GLU | $\sigma$ | 严格在 $(0,1)$ | 饱和导致门梯度变小 |
| ReGLU | $\operatorname{ReLU}$ | 非负，负侧为 0 | 门分支死亡和稀疏性 |
| GEGLU | GELU | 负侧可小于 0，正侧近似线性 | CDF 或近似形式是否一致 |
| SwiGLU | SiLU | 负侧有小负谷，正侧无界 | sigmoid 稳定计算和负侧曲率 |

这些名称描述的是门分支，不是把值分支再激活一次。若把 $\varphi$ 错放到乘法之后，得到的是另一种函数，不能继续沿用同一个初始化、参数预算或算子名。

## 参数预算决定中间宽度不能照搬

普通前馈层若中间宽度为 $m_{\mathrm{ff}}$，忽略 bias，参数量是

$$
P_{\mathrm{plain}}
=dm_{\mathrm{ff}}+m_{\mathrm{ff}}d
=2dm_{\mathrm{ff}}.
$$

SwiGLU 有两个上投影和一个下投影。若其中间宽度为 $m_{\mathrm{swiglu}}$，参数量是

$$
P_{\mathrm{swiglu}}
=dm_{\mathrm{swiglu}}+dm_{\mathrm{swiglu}}+m_{\mathrm{swiglu}}d
=3dm_{\mathrm{swiglu}}.
$$

要保持大致相同的矩阵参数量，应满足

$$
3dm_{\mathrm{swiglu}}
\approx2dm_{\mathrm{ff}}
\quad\Longrightarrow\quad
m_{\mathrm{swiglu}}\approx\frac23m_{\mathrm{ff}}.
$$

例如模型宽度 $d=6$ 时：

| 结构 | 中间宽度 | 矩阵数量 | 矩阵参数量 |
| --- | ---: | ---: | ---: |
| 两层 ReLU FFN | $m_{\mathrm{ff}}=24$ | $2$ | $2\times6\times24=288$ |
| SwiGLU，等参数预算 | $m_{\mathrm{swiglu}}=16$ | $3$ | $3\times6\times16=288$ |
| SwiGLU，沿用宽度 24 | $24$ | $3$ | $3\times6\times24=432$ |

实际实现还要把 bias、分组方式、张量并行切分和硬件要求算进去，因此通常会把 $\frac23m_{\mathrm{ff}}$ 四舍五入到某个倍数。比较模型时若一边换成 SwiGLU、一边保留原中间宽度，测到的是结构和参数预算的混合差异。

## 门控前馈层仍然只在 token 内做变换

对序列中的每个 token，SwiGLU 的矩阵只作用在该 token 的隐藏向量：

$$
\mathbf{x}_t
\longmapsto
W_g\mathbf{x}_t,\;
W_u\mathbf{x}_t
\longmapsto
\operatorname{SiLU}(W_g\mathbf{x}_t)\odot(W_u\mathbf{x}_t)
\longmapsto
W_d\mathbf{h}_t.
$$

它不读取其他 token，也不替代 self-attention 的跨位置混合。可以把两者分工记成：

| 模块 | 主要混合对象 | 输入依赖的选择 |
| --- | --- | --- |
| self-attention | 序列位置之间 | 依据 query、key 选择其他位置的 value |
| SwiGLU FFN | 单个 token 的特征坐标 | 依据门分支缩放值分支 |
| LayerNorm 或 RMSNorm | 单个 token 的特征统计 | 依据范数重标定整个向量 |

因此“门控”并不意味着它掌握了 token 路由或稀疏专家选择。SwiGLU 是密集的逐坐标乘法；除非另加稀疏路由机制，否则每个 token 仍会计算两条投影分支。

## 数值稳定性和实现边界

SiLU 的核心是 sigmoid。直接计算 $e^{-x}$ 在 $x$ 很负时可能上溢，因此可以按符号写成稳定形式：

$$
\sigma(x)=
\begin{cases}
\dfrac1{1+e^{-x}},&x\geq0,\\[6pt]
\dfrac{e^x}{1+e^x},&x<0.
\end{cases}
$$

再计算 $\operatorname{SiLU}(x)=x\sigma(x)$。实际框架可能提供 fused SiLU、SwiGLU kernel 或硬件近似；训练、导出、量化和推理端要确认它们使用同一系数、同一 dtype 约定和同一分支顺序。

实现时还要明确：

1. $W_g$ 是门分支，$W_u$ 是值分支，不能因张量布局转换而交换；
2. 两条分支的中间宽度必须相同，才能做 $\odot$；
3. 激活应先作用在门分支，再与值分支相乘；
4. 输出投影应在逐坐标乘法之后进行；
5. 混合精度下要检查门、乘法和归一化的中间 dtype；
6. 不要把门值截断到 $(0,1)$，那会把 SwiGLU 偷换成 sigmoid GLU；
7. 不要把最终 SwiGLU 输出直接当成概率，前馈层输出仍是隐藏表示。

## 失效模式

**把 Swish、SiLU 和 GELU 当成同一个函数。** SiLU 是 $\beta=1$ 的 Swish；GELU 使用 $\Phi$ 而不是 $\sigma$。复现论文或迁移算子时，要把门函数和近似常数写完整。

**把 Swish 的门当成概率。** $\sigma(x)$ 是概率形状的函数，但 $x\sigma(x)$ 可以为负，也可以大于 1。SwiGLU 的 SiLU 门更不具有概率语义。

**沿用普通 FFN 的中间宽度。** SwiGLU 多了一条上投影，参数量会增加约 50%。若目标是公平比较，应先按参数预算调整中间宽度。

**只检查最终 loss。** 门和值分支可能有一个已经饱和、全零或尺度异常，但残差连接仍暂时遮住问题。应记录 $\mathbf{g}$、$\operatorname{SiLU}(\mathbf{g})$、$\mathbf{u}$、$\mathbf{h}$ 及两条梯度路径。

**把门控乘法放错位置。** $\operatorname{SiLU}(W_g\mathbf{x})\odot W_u\mathbf{x}$、$\operatorname{SiLU}(W_g\mathbf{x}\odot W_u\mathbf{x})$ 和 $W_d(\operatorname{SiLU}(W_g\mathbf{x}))\odot W_u\mathbf{x}$ 不是同一个算子。导出图和 fused kernel 要检查算子顺序。

**误以为门控结构会自动稀疏。** SwiGLU 是密集乘法，门值通常不是精确零；它改变特征尺度，不等于减少 FLOPs。需要稀疏计算时，必须另有可审计的稀疏路由或剪枝规则。

**忽略负侧门值。** SiLU 门在一段负区间可以为负，值分支的符号也可能因此翻转。若实现偷偷使用 max(0, gate)，就已经变成 ReGLU。

**只在正输入上做梯度检查。** 负侧是区分 GELU、SiLU、ReLU 及其门控变体的关键区域。至少覆盖 $x=-2,-1,0,1,2$，并对门和值分支分别做有限差分。

## 一个可复用的 Swish 与 SwiGLU 核验协议

遇到激活替换、门控 FFN 重构或自定义 kernel，可以按下面的顺序检查：

1. 写明使用 $\operatorname{Swish}_{\beta}$、SiLU、GELU、GLU 还是 SwiGLU，并记录 $\beta$、近似形式和参数布局。
2. 对 $x=-2,-1,0,1,2$ 独立计算前向值、一级导数和必要的二阶导数，确认负谷、零点斜率和正侧增长。
3. 固定一个小向量，分别算出门分支、值分支、逐坐标乘积和输出投影，检查矩阵方向没有交换。
4. 用上游梯度核对 $\partial L/\partial\mathbf{g}$、$\partial L/\partial\mathbf{u}$ 和 $\partial L/\partial\mathbf{x}$，不要只检查最终输出。
5. 记录真实训练中的门均值、负值比例、接近零比例、值分支尺度、乘积尺度和两条梯度范数。
6. 按 $m_{\mathrm{swiglu}}\approx2m_{\mathrm{ff}}/3$ 核对参数量、FLOPs 和显存，记录硬件对齐带来的取整。
7. 在训练、导出、量化和部署端重复同一组边界输入，确认 sigmoid 稳定式、dtype 和 fused kernel 的算子顺序一致。
8. 把门控 FFN 与 attention 分开评估：它改变 token 内特征变换，不应被当成跨位置混合或稀疏专家路由的证据。

这份协议把“函数形状”“两分支梯度”“参数预算”和“系统实现”分成四层证据。SwiGLU 是否值得采用，不能只由曲线更平滑或单次 loss 更低决定，还要看预算、稳定性、硬件实现和部署一致性。

## 相关词条

- [激活函数](../neurons-and-activations/activation-functions/)：建立输出范围、导数、饱和和输出头的比较坐标。
- [GELU](../neurons-and-activations/gelu/)：推导正态 CDF 门控及其近似形式。
- [ReLU](../neurons-and-activations/relu/)：比较硬门、稀疏零值和分段线性结构。
- [死亡 ReLU 与 Leaky ReLU](../neurons-and-activations/dead-relu-and-leaky/)：分析负侧梯度与恢复路径。
- [tanh](../neurons-and-activations/tanh/)：对比零中心饱和激活。
- [饱和与梯度消失](../neurons-and-activations/saturation-and-vanishing/)：分析门和导数在深层乘积中的尺度变化。
- [激活函数选择](../neurons-and-activations/choosing-activations/)：把激活形状放回任务和训练协议中比较。
- [Softmax](../neurons-and-activations/softmax/)：区分逐分量激活与向量耦合的概率输出头。
