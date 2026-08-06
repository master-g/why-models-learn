---
title: "困惑度：把平均 token 损失变成可读尺度"
tags: ["why-models-learn"]
---

**困惑度**是平均负对数似然取指数后的量。若模型在 $N$ 个有效 token 上给出的条件概率是 $q_t=q(x_t\mid x_{<t})$，使用自然对数时：

$$
\begin{aligned}
\operatorname{PPL}(q)
&=\exp\left(
-\frac1N\sum_{t=1}^N\ln q_t
\right)\\
&=\left(\prod_{t=1}^Nq_t^{-1}\right)^{1/N}
\end{aligned}
$$

如果每一步都在 $K$ 个 token 上均匀分配概率，困惑度就是 $K$；一般情况下它是模型面对数据时的**有效分支数**，不是词表大小，也不是模型实际会生成的候选 token 数。本篇从单 token 的负对数损失推导困惑度，再说明熵、交叉熵和 KL 散度如何进入公式，最后处理 padding、tokenizer、序列长度、teacher forcing 和语言模型评估边界。

## 从序列概率到平均损失

自回归模型把一条 token 序列的概率分解成

$$
q(x_{1:N})
=\prod_{t=1}^Nq(x_t\mid x_{<t})
$$

取自然对数把乘法变成加法：

$$
\ln q(x_{1:N})
=\sum_{t=1}^N\ln q(x_t\mid x_{<t})
$$

负对数似然是

$$
\operatorname{NLL}(x_{1:N};q)
=-\sum_{t=1}^N\ln q(x_t\mid x_{<t})
$$

它随着序列长度增加而增加，所以比较不同长度的序列时通常使用每 token 平均值：

$$
\ell_{\mathrm{nats/token}}
=-\frac1N\sum_{t=1}^N\ln q(x_t\mid x_{<t})
$$

困惑度只是把这个平均损失从对数尺度变回概率尺度：

$$
\operatorname{PPL}(q)=\exp\left(\ell_{\mathrm{nats/token}}\right)
$$

平均损失为 0 时，模型每一步都给真实 token 概率 1，困惑度为 1。平均损失越大，困惑度越大。因为指数是单调函数，在同一数据集、同一 tokenizer 和同一 token 计数方式下，比较困惑度等价于比较平均 NLL。

## 困惑度是逆概率的几何平均

每个 token 的惊奇度是

$$
s_t=-\ln q_t=\ln\frac1{q_t}
$$

取平均再指数化：

$$
\begin{aligned}
\operatorname{PPL}(q)
&=\exp\left(\frac1N\sum_ts_t\right)\\
&=\left(\prod_t\exp(s_t)\right)^{1/N}\\
&=\left(\prod_t\frac1{q_t}\right)^{1/N}
\end{aligned}
$$

因此困惑度是各步「有效候选数」$1/q_t$ 的几何平均，不是算术平均。一个极低的真实 token 概率会显著拉高乘积，不能被若干个很容易预测的 token 线性抵消。

### 三步数字例子

假设模型对三个真实 token 分别给出

$$
q_1=\frac12,
\qquad
q_2=\frac14,
\qquad
q_3=\frac18
$$

用 bit 作为对数单位，每一步的损失是

$$
-\log_2q_1=1,
\qquad
-\log_2q_2=2,
\qquad
-\log_2q_3=3
\quad\mathrm{bits}
$$

平均损失为

$$
\ell_{\mathrm{bits/token}}
=\frac{1+2+3}{3}
=2\ \mathrm{bits}
$$

在 bit 单位下应使用 $2$ 作为指数底：

$$
\operatorname{PPL}
=2^2
=4
$$

直接从概率相乘也得到

$$
\left(
\frac1{1/2}\cdot\frac1{1/4}\cdot\frac1{1/8}
\right)^{1/3}
=(2\times4\times8)^{1/3}
=4
$$

如果把相同的损失误当成自然对数再使用 $\exp$，会得到 $e^2$，这不是同一个单位下的困惑度。实际框架通常返回 nats/token，所以常见写法是 $\exp(\ell)$；若自己把日志换成 bit，就必须使用 $2^\ell$。

## 均匀分布和熵的基准

若 $Y$ 在 $K$ 个 token 上均匀分布：

$$
p(y)=\frac1K
$$

其熵是

$$
H(Y)=\ln K
$$

所以内在困惑度为

$$
\exp(H(Y))=K
$$

不均匀分布的熵更低，困惑度也更低。以 bit 为单位，考虑

$$
p=(0.5,0.25,0.125,0.125)
$$

熵为

$$
\begin{aligned}
H_2(p)
&=0.5\times1+0.25\times2
  +0.125\times3+0.125\times3\\
&=1.75\ \mathrm{bits}
\end{aligned}
$$

对应的内在困惑度是

$$
\operatorname{PPL}_2(p)
=2^{1.75}
=3.363585661015
$$

它介于最常见结果的 1 个选择和四个完全均匀选择之间。这里的「3.36 个有效选择」是平均意义，不代表每一步真的有 3.36 个 token。

## 模型困惑度其实是条件交叉熵

真实数据分布记为 $p$，模型分布记为 $q$。对上下文和下一个 token 的联合分布取平均，模型的 per-token 损失是条件交叉熵：

$$
\mathcal H_p(q)
=\mathbb E_{(X_{<t},X_t)\sim p}
\left[-\ln q(X_t\mid X_{<t})\right]
$$

因此模型的理论困惑度是

$$
\operatorname{PPL}(q)=\exp\left(\mathcal H_p(q)\right)
$$

条件交叉熵分解为真实条件熵和额外 KL 代价：

$$
\mathcal H_p(q)
=H_p(X_t\mid X_{<t})
 +\mathbb E_{X_{<t}\sim p}
D_{\mathrm{KL}}\left(
p(\cdot\mid X_{<t})
\middle\Vert
q(\cdot\mid X_{<t})
\right)
$$

指数化后：

$$
\operatorname{PPL}(q)
=\operatorname{PPL}(p)
\times
\exp\left(
\mathbb E_{X_{<t}}
D_{\mathrm{KL}}\left(p(\cdot\mid X_{<t})\Vert q(\cdot\mid X_{<t})\right)
\right)
$$

所以模型困惑度不能低于数据分布的内在困惑度；两者的比值是平均条件 KL 的指数。模型越接近真实条件分布，困惑度越接近数据本身的不可约预测难度。

### 一个四类别的数字例子

令真实分布为

$$
p=(0.5,0.25,0.125,0.125)
$$

模型输出

$$
q=(0.4,0.3,0.2,0.1)
$$

真实熵、模型交叉熵和 KL 额外代价分别为

$$
\begin{aligned}
H_2(p)&=1.75\ \mathrm{bits}\\
H_2(p,q)&=1.800687469707\ \mathrm{bits}\\
D_{\mathrm{KL},2}(p\Vert q)&=0.050687469707\ \mathrm{bits}
\end{aligned}
$$

所以

$$
\begin{aligned}
\operatorname{PPL}_2(p)
&=2^{1.75}
=3.363585661015\\
\operatorname{PPL}_2(q)
&=2^{1.800687469707}
=3.483861979569
\end{aligned}
$$

两者比值为

$$
\frac{\operatorname{PPL}_2(q)}{\operatorname{PPL}_2(p)}
=2^{0.050687469707}
=1.035758363448
$$

模型的平均编码代价比真实分布高约 $3.58\%$。这不是说每个 token 的概率都高出同一比例，而是平均对数代价经过指数化后的相对差异。

## padding 和 mask：只对有效 token 计分

批处理时不同序列经常补齐到同一长度。设 $m_t\in\{0,1\}$ 是有效 token mask，正确的平均损失是

$$
\ell
=-\frac{\sum_tm_t\ln q_t}{\sum_tm_t}
$$

困惑度则是

$$
\operatorname{PPL}
=\exp\left(
-\frac{\sum_tm_t\ln q_t}{\sum_tm_t}
\right)
$$

padding 位置不是来自数据分布的观测，不应当把它们放入分子，也不应当把它们放入有效 token 数分母。语言模型还要明确是否把 EOS 计入分数：只要不同实验的规则一致，结果可以比较；规则不一致时，一个额外的 EOS 会改变总 NLL 和 token 数。

例如两个有效 token 的概率是 $1/2$ 和 $1/4$，第三个位置是 padding。正确的 bit 损失为

$$
\ell=\frac{1+2}{2}=1.5\ \mathrm{bits/token}
$$

所以

$$
\operatorname{PPL}=2^{1.5}=2.828427124746
$$

如果错误地把 padding 当作第三个 token 并给它一个虚构的概率，评估值就不再对应这两个有效观测的交叉熵。

## 数据集要按 token 加权

对多个序列计算总困惑度时，应该先把所有有效 token 的 NLL 相加，再除以所有有效 token 数：

$$
\ell_{\mathrm{all}}
=\frac{\sum_s\sum_{t=1}^{N_s}
-\ln q(x_{s,t}\mid x_{s,<t})}
{\sum_sN_s}
$$

不能先算每条序列的 PPL，再对这些 PPL 做算术平均。例子：

- 序列 A 只有一个 token，模型给它概率 $1/4$，所以 NLL 是 2 bit，PPL 是 4。
- 序列 B 有三个 token，每个概率都是 $1/2$，所以 NLL 是 3 bit，PPL 是 2。

合并后的总 NLL 是 $5$ bit，总 token 数是 $4$，因此

$$
\operatorname{PPL}_{\mathrm{all}}
=2^{5/4}
=2.378414230005
$$

而简单平均两条序列的 PPL 会得到

$$
\frac{4+2}{2}=3
$$

两者不同，因为第一条序列只贡献一个 token，第二条贡献三个 token。困惑度是对 token 计数归一化后的指数，长度权重不能在指数化之后再补。

## tokenizer 决定了每 token 困惑度的单位

同一段字符可以被不同 tokenizer 切成不同数量的 token。设字符串「ab」的真实序列概率都是 $0.25$：

- tokenizer A 切成两个 token，$q(a)=0.5$、$q(b\mid a)=0.5$，总 NLL 是 2 bit，平均 NLL 是 1 bit/token，PPL 是 2。
- tokenizer B 切成一个 token「ab」，$q(ab)=0.25$，总 NLL 仍是 2 bit，但平均 NLL 是 2 bit/token，PPL 是 4。

模型给整段字符串的概率完全相同，per-token 困惑度却翻倍：

$$
\operatorname{PPL}_{A}=2,
\qquad
\operatorname{PPL}_{B}=4
$$

因此不能把不同 tokenizer、不同词表或不同 token 化规则下的 PPL 直接排成一条排行榜。更稳妥的跨单位指标是每字符或每字节的 bit 数：

$$
\operatorname{BPC}
=\frac{-\log_2q(\text{sequence})}
{\text{字符数}}
$$

上面两个 tokenizer 都得到总共 2 bit、2 个字符，所以 BPC 都是

$$
\operatorname{BPC}=\frac22=1\ \mathrm{bit/character}
$$

BPC 也不是万能转换器：字符定义、Unicode 规范化、空白处理和字节编码都必须写清楚。它解决的是单位变化，不会解决训练数据和评估数据不同的问题。

## teacher forcing 评估了什么

困惑度通常在真实前缀上计算：

$$
\operatorname{PPL}_{\mathrm{TF}}
=\exp\left(
-\frac1N\sum_t
\ln q_\theta(x_t\mid x_{<t}^{\mathrm{data}})
\right)
$$

这叫 teacher forcing 评估。第 $t$ 步的上下文来自数据，而不是模型上一步自己生成的 token。它正好对应语言模型训练时的最大似然分解，因此数值稳定、容易批量计算。

但生成时模型可能把自己的早期错误带入后续上下文：

$$
\hat x_t\sim q_\theta(\cdot\mid\hat x_{<t})
$$

此时上下文分布已经从数据分布变成了模型自身的分布。teacher forcing PPL 不能直接回答：

- 模型采样的文本是否连贯或有帮助。
- 模型是否会在长上下文中反复、跑题或失去约束。
- 模型输出的事实是否正确。
- 模型是否只把概率集中到少数模板而牺牲多样性。

低 PPL 说明模型在指定评估分布上给真实 token 较高概率，但生成质量还需要独立的任务指标和人工或程序化评估。

## 熵率和无限长序列

对平稳随机序列，熵率定义为

$$
h
=\lim_{T\to\infty}
\frac1T H(X_1,\ldots,X_T)
$$

若极限存在，源的内在困惑度率是

$$
\operatorname{PPL}_{\mathrm{rate}}=\exp(h)
$$

使用 bit 时则是

$$
\operatorname{PPL}_{\mathrm{rate},2}=2^{h_2}
$$

独立同分布的 $K$ 类均匀源有 $h=\ln K$，困惑度率为 $K$。有强结构的序列即使每个位置的边缘分布很均匀，条件熵率也可能更低；[互信息](../information-theory/mutual-information/) 正是在量化上下文从边缘熵中减少的部分。

前文的二元马尔可夫链例子中，每一步以概率 $0.1$ 翻转上一状态，条件熵率是

$$
h_2=h_2(0.1)=0.468995593589\ \mathrm{bits}
$$

所以困惑度率为

$$
\operatorname{PPL}_{\mathrm{rate},2}
=2^{0.468995593589}
=1.384145488462
$$

虽然单个状态的边缘分布是公平的，长序列的每一步有效分支数只有约 1.38，因为上一状态已经提供了预测信息。

![困惑度把平均 token 信息量取指数](/assets/information-theory/svg/perplexity.1.svg)

## 神经网络和语言模型实现

分类模型通常把 batch 内所有有效位置的交叉熵求和，再除以有效样本数。若返回的是自然对数损失「loss」：

$$
\operatorname{PPL}=\exp(\text{loss})
$$

实现时需要同时核对：

- loss 是「mean」还是「sum」，是否已经除过 token 数。
- padding、attention mask 和 label mask 是否使用同一套有效位置。
- logits 是否经过稳定的 log-softmax，而不是先把极小概率显式算出来。
- 训练、验证和测试是否使用相同 tokenizer、词表、EOS 规则和上下文截断策略。
- 分布式训练时分子和分母是否跨 worker 正确汇总，而不是平均每个 worker 的 PPL。

模型结构不会改变困惑度定义，但会改变条件概率的参数化和可达到的交叉熵。更大的模型、更多数据或更好的优化可能降低同一评估集上的 PPL；这仍然只是似然层面的证据，不等价于所有下游能力都提升。

## 失效模式

**把 nats 和 bits 混用。** nats/token 要用 $\exp$，bits/token 要用 $2^x$。底数错一次，整个困惑度就不在同一尺度。

**把 padding 当真实 token。** 这会改变分子和分母，得到的数不再是评估数据的平均 NLL。

**平均每条序列的 PPL。** 正确流程是先按有效 token 汇总 NLL，再除以总 token 数，最后指数化。

**比较不同 tokenizer 的 PPL。** token 数和 token 边界改变了单位；跨 tokenizer 应同时报告 tokenizer，或使用明确的 bit-per-byte/character 指标。

**把 PPL 当作生成质量。** teacher forcing 使用真实前缀，生成使用模型自己的前缀；二者面对的上下文分布不同。

**把困惑度当成词表大小。** 只有完全均匀的 $K$ 类分布才有 PPL $=K$；一般 PPL 是概率加权后的有效分支数。

**把模型熵和交叉熵混淆。** 评估真实数据时计算的是 $H(p,q)$，不是模型自己采样分布的 $H(q)$；模型熵低不保证它给真实 token 高概率。

**忽略零概率和数据泄漏。** 真实 token 若得到零概率，NLL 和 PPL 为无穷；测试数据若混入训练集，PPL 可能虚低，不能代表泛化。

## 相关词条

- [information-and-surprise](../information-theory/information-and-surprise/)：定义单个 token 的自信息量。
- [entropy](../information-theory/entropy/)：定义熵、联合熵、条件熵和熵率。
- [条件熵](../information-theory/conditional-entropy/)：解释序列中每一步剩余的不确定性。
- [cross-entropy](../information-theory/cross-entropy/)：推导语言模型 NLL 所属的交叉熵。
- [kl-divergence](../information-theory/kl-divergence/)：解释模型 PPL 相对数据内在 PPL 的额外代价。
- [互信息](../information-theory/mutual-information/)：量化上下文减少下一个 token 不确定性的部分。
- [maximum-likelihood](../probability/maximum-likelihood/)：说明自回归训练为何最小化平均负对数似然。
