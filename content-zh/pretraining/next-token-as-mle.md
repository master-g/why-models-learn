---
title: "下一词最大似然：语言模型如何拟合序列概率"
tags: ["why-models-learn"]
---

在固定 tokenizer、文档边界、特殊 token、loss mask 和样本权重后，decoder-only 语言模型对每个真实前缀预测下一个 token 的负对数损失，正好是 token 序列最大似然的可微写法。把联合概率按前缀分解，取对数把乘积变成求和，再对固定数据集做平均，就得到训练中使用的 next-token cross-entropy。这个等价关系属于被明确写出的训练目标；它不保证模型恢复真实世界分布，也不覆盖 label smoothing、正则化、采样加权和 free-running 生成。

本文先把“似然”放在 token 序列而不是字符或文档的层面，推导数据集似然、NLL 和交叉熵的关系。随后区分 token-level mean 与 sequence-level mean，解释 teacher forcing 为什么允许并行计算以及它为什么不等于推理时的生成过程。最后检查 tokenizer、EOS、packing、数据混合、MAP、label smoothing、验证 perplexity 和常见实现失效模式。

![下一词最大似然从 token 序列经过因果分解、对数求和与归约，形成可反向传播的训练目标](/assets/pretraining/svg/next-token-as-mle.1.svg)

## 先固定最大似然的对象

### 数据不是一串没有边界的字符

设 tokenizer、词表版本和 special token 已经固定。第 $n$ 条训练记录被编码为

$$
x^{(n)}_0,x^{(n)}_1,\ldots,x^{(n)}_{T_n},
$$

其中 $x^{(n)}_0$ 可以是 BOS，最后一个有效目标通常是 EOS，$T_n$ 表示有效的 next-token prediction 数量。这里的下标不是字符位置，也不一定对应自然语言词的位置；它只表示 token 序列中的预测轴。

训练数据还要保存文档 ID、来源、切分版本和边界规则。相同文字经过不同 tokenizer 会产生不同的 token 序列；相同 token 流使用独立窗口、EOS 串接或 block-diagonal packing，也会定义不同的条件上下文。因此，最大似然的第一个对象不是“文本本身”这个模糊概念，而是经过输入协议确定的有限 token 样本。

一条记录至少要把以下对象分开：

|对象|记号|作用|
| --- | --- | --- |
|输入 token|$x^{(n)}_{0:T_n-1}$|提供每个预测位置的真实前缀末 token|
|目标 token|$x^{(n)}_{1:T_n}$|作为下一个 token 的观测结果|
|有效长度|$T_n$|决定该记录产生多少个条件概率项|
|文档边界|BOS、EOS、segment id|决定前缀可以读取哪些 token|
|loss mask|$m^{(n)}_t\in\{0,1\}$|决定目标项是否进入训练归约|

如果 padding 被放在 batch 尾部，padding token 不会因为拥有一个整数 ID 就变成观测数据。它只能扩展张量形状；attention mask 和 loss mask 需要分别声明它能否被读取以及是否进入损失。

### 一个条件概率就是一个观测事件

给定前缀 $x^{(n)}_{0:t}$，模型输出词表上的条件分布：

$$
p_\theta\left(
x^{(n)}_{t+1}\mid x^{(n)}_{0:t}
\right).
$$

这个量回答的是“在当前模型参数和当前前缀下，下一个观测 token 被赋予多少概率”。它不是当前位置的类别是否正确，也不是生成时最终选中的 token 数量。argmax 可以选出概率最大的 token，但最大似然保留整组概率，因为损失需要知道目标 token 获得了多少质量。

对一条长度为 $T_n$ 的 token 序列，因果语言模型写出：

$$
p_\theta\left(
x^{(n)}_{1:T_n}\mid x^{(n)}_0
\right)
=
\prod_{t=0}^{T_n-1}
p_\theta\left(
x^{(n)}_{t+1}\mid x^{(n)}_{0:t}
\right).
$$

这是概率链式法则在自回归参数化下的直接应用。模型不需要假设 token 彼此独立；它只需要规定每个位置在给定前缀后的条件分布。[因果语言建模](../transformer-architectures/causal-language-modeling/)展开右移输入、causal mask 和这条分解的张量实现。

### 概率、似然和损失的方向

同一个数值 $p_\theta(x)$ 可以有不同的阅读方向：

|名称|固定什么|改变什么|在 next-token 训练中的位置|
| --- | --- | --- | --- |
|条件概率|参数 $\theta$ 和前缀|目标 token|被模型输出的一个概率|
|序列概率|参数 $\theta$ 和边界|整条 token 序列|条件概率的乘积|
|似然|观测序列|参数 $\theta$|用来选择参数的函数|
|对数似然|观测序列|参数 $\theta$|序列概率取对数后的和|
|NLL|观测序列|参数 $\theta$|训练时被最小化的相反数|

观测序列固定后，似然不是参数的概率分布。它不需要对 $\theta$ 归一化；模型条件概率只需要在每一个固定前缀下对词表 token 归一化。[最大似然](../probability/maximum-likelihood/)从一般概率模型的角度区分这两个方向。

## 从联合概率到 next-token loss

### 数据集似然是所有序列概率的乘积

给定 $N$ 条 token 序列，假设训练记录在模型参数条件下按数据集样本合同相乘，则数据集似然为

$$
\mathcal L_D(\theta)
=
\prod_{n=1}^{N}
p_\theta\left(
x^{(n)}_{1:T_n}\mid x^{(n)}_0
\right).
$$

把上一节的因果分解代入：

$$
\mathcal L_D(\theta)
=
\prod_{n=1}^{N}
\prod_{t=0}^{T_n-1}
p_\theta\left(
x^{(n)}_{t+1}\mid x^{(n)}_{0:t}
\right).
$$

直接相乘会在长序列上快速下溢。对数函数严格递增，所以最大化似然与最大化对数似然给出同一个参数选择：

$$
\begin{aligned}
\log\mathcal L_D(\theta)
&=
\sum_{n=1}^{N}\sum_{t=0}^{T_n-1}
\log p_\theta\left(
x^{(n)}_{t+1}\mid x^{(n)}_{0:t}
\right),\\
\widehat\theta_{\mathrm{MLE}}
&=
\operatorname*{arg\,max}_{\theta}
\log\mathcal L_D(\theta).
\end{aligned}
$$

取相反数后，最大似然变成最小化总 NLL：

$$
\mathcal J_{\mathrm{NLL}}(\theta)
=
-\sum_{n=1}^{N}\sum_{t=0}^{T_n-1}
\log p_\theta\left(
x^{(n)}_{t+1}\mid x^{(n)}_{0:t}
\right).
$$

这条目标没有引入新的神经网络层。softmax 把每个位置的 logits 变成词表分布，真实目标索引抽取其中一个概率，再由负对数把乘积目标改写成可求导的加法目标。

### 固定常数的平均不改变 MLE 解

令所有有效 target token 数为

$$
M
=
\sum_{n=1}^{N}T_n.
$$

当数据集、tokenizer、边界和 mask 固定时，$M$ 与 $\theta$ 无关。因此 token-level mean loss

$$
\mathcal J_{\mathrm{token}}(\theta)
=
\frac{1}{M}
\mathcal J_{\mathrm{NLL}}(\theta)
$$

和总 NLL 拥有相同的最优参数：

$$
\operatorname*{arg\,min}_{\theta}
\mathcal J_{\mathrm{token}}(\theta)
=
\operatorname*{arg\,min}_{\theta}
\mathcal J_{\mathrm{NLL}}(\theta).
$$

平均的作用是让 loss 的单位从“整份数据的总 surprise”变成“每个有效 token 的平均 NLL”。它改变梯度的固定尺度，却不改变固定数据集上的 MLE 目标。若 batch 中有效 token 数变化，实际训练还要记录 denominator，才能知道每一步梯度缩放是否一致。

### sequence-level mean 是另一个目标

变长记录常见两种归约。token-level mean 对每个 token 等权：

$$
\mathcal J_{\mathrm{token}}
=
\frac{
\displaystyle\sum_{n=1}^{N}\sum_{t=0}^{T_n-1}
\ell_{n,t}
}{
\displaystyle\sum_{n=1}^{N}T_n
},
\qquad
\ell_{n,t}
=
-\log p_\theta\left(
x^{(n)}_{t+1}\mid x^{(n)}_{0:t}
\right).
$$

sequence-level mean 先在每条记录内部平均，再让每条记录等权：

$$
\mathcal J_{\mathrm{sequence}}
=
\frac1N
\sum_{n=1}^{N}
\left[
\frac1{T_n}
\sum_{t=0}^{T_n-1}\ell_{n,t}
\right].
$$

两者只有在记录长度相同，或数据分布恰好满足特殊条件时才等价。标准数据集 MLE 对每一个观测 token 项求和；把总和除以固定的 $M$ 仍然是 MLE。把每条序列先除以自己的 $T_n$，则会给短序列更大的总权重，这是一种新的加权目标，不能继续无条件称为原始序列 MLE。

### padding 和 mask 只保留有效似然项

对 batch 中第 $b$ 行的第 $t$ 个位置，写出单项损失：

$$
\ell_{b,t}
=
-\log p_\theta(y_{b,t}\mid c_{b,t}),
$$

其中 $c_{b,t}$ 是实际可见前缀，$y_{b,t}$ 是 next-token label。引入有效 mask $m_{b,t}$ 后：

$$
\mathcal J_{\mathrm{masked}}
=
\frac{
\displaystyle\sum_{b,t}
m_{b,t}\ell_{b,t}
}{
\displaystyle\sum_{b,t}m_{b,t}
}.
$$

这里的分母不是名义 batch 大小，也不是固定上下文长度。若 EOS 是训练目标，就把 EOS 的 mask 设为 1；若某些 prompt 或 padding 位置不属于监督，就把对应 mask 设为 0。attention mask 决定 $c_{b,t}$，loss mask 决定 $\ell_{b,t}$ 是否进入目标，两者不能互相替代。

## 交叉熵是经验分布上的 NLL

### 把 token 位置看成条件事件

把每个有效位置压成一个条件事件：

$$
\mathcal E_D
=
\left\{
\left(c_{n,t},y_{n,t}\right)
\right\}_{n,t:\,m_{n,t}=1},
\qquad
M=\lvert\mathcal E_D\rvert.
$$

经验联合分布可以写成

$$
\widehat q_D(c,y)
=
\frac1M
\sum_{(c',y')\in\mathcal E_D}
\mathbb 1[c'=c,\ y'=y].
$$

模型在相同 context 上给出条件分布 $p_\theta(y\mid c)$。token-level mean NLL 就是经验条件事件对模型条件分布的交叉熵：

$$
\mathcal J_{\mathrm{token}}(\theta)
=
\mathbb E_{(C,Y)\sim\widehat q_D}
\left[
-\log p_\theta(Y\mid C)
\right].
$$

这也解释了为什么 next-token loss 既可以叫 NLL，也可以叫 token-level cross-entropy。前者强调概率模型的似然，后者强调经验事件和模型分布之间的平均编码代价；数值公式相同，报告的解释角度不同。[交叉熵](../information-theory/cross-entropy/)从熵与 KL 散度的角度展开这一关系。

### 充分表达能力时的局部最优分布

若某个 context $c$ 在数据中出现多次，目标 token 的经验条件频率为

$$
\widehat q_D(y\mid c)
=
\frac{N(c,y)}
{N(c)},
\qquad
N(c)=\sum_yN(c,y).
$$

在只考虑这个 context 的理想情况下，任意候选条件分布 $p(y\mid c)$ 的交叉熵满足

$$
\begin{aligned}
H\left(\widehat q_D(\cdot\mid c),p(\cdot\mid c)\right)
&=
H\left(\widehat q_D(\cdot\mid c)\right)\\
&\quad+
D_{\mathrm{KL}}\left(
\widehat q_D(\cdot\mid c)
\middle\Vert
p(\cdot\mid c)
\right).
\end{aligned}
$$

KL 散度非负，因此如果模型可以独立表达这个 context 的每个条件分布，最小值在 $p(y\mid c)=\widehat q_D(y\mid c)$ 处取得。实际语言模型共享 embedding、attention、FFN 和输出 head 参数，多个 context 之间受到同一参数化约束；训练得到的是模型族对经验分布的近似投影，而不是逐个 context 直接记下频率。

### 一个重复前缀的数字例子

假设同一个前缀 $c$ 在数据中出现三次，后续 token 是 $u,u,v$。经验分布为

$$
\widehat q_D(u\mid c)=\frac23,
\qquad
\widehat q_D(v\mid c)=\frac13.
$$

若模型给出 $p(u\mid c)=0.8$、$p(v\mid c)=0.2$，三次观测的平均 NLL 为

$$
\begin{aligned}
\mathcal J_c
&=
-\frac23\log0.8-\frac13\log0.2\\
&=
0.685241671688.
\end{aligned}
$$

经验分布自身的熵为

$$
H\left(\widehat q_D\right)
=
-\frac23\log\frac23-\frac13\log\frac13
=
0.636514168295.
$$

两者之差是该模型在这个 context 上的 KL 散度：

$$
D_{\mathrm{KL}}\left(
\widehat q_D\middle\Vert p
\right)
=
0.048727503393.
$$

模型把更高概率给了经验频率更高的 $u$，但仍然没有把 $u$ 推到概率 1。最大似然拟合的是数据中观察到的条件频率；它不要求每个 context 只有一个确定后续，也不提供数据之外的事实保证。

## teacher forcing 只改变条件前缀的来源

### 训练时可以并行计算

训练记录已知完整 token 序列，所以第 $t$ 个位置的条件输入可以直接取真实前缀：

$$
\begin{aligned}
X_{n,t}&=x^{(n)}_t,\\
Y_{n,t}&=x^{(n)}_{t+1},\\
C_{n,t}&=x^{(n)}_{0:t}.
\end{aligned}
$$

causal mask 让每个 query 只读取 $C_{n,t}$ 内的 token。虽然同一次 forward 同时产生所有位置 logits，但每一行仍对应一个不同的条件事件。并行计算没有把未来标签暴露给当前 query；它只利用了训练数据已经提供的真实前缀。

|阶段|当前位置的前缀|是否知道后续真实 token|主要计算|
| --- | --- | --- | --- |
|teacher forcing 训练|真实 token 前缀|知道整条训练记录，但 mask 屏蔽未来|整段 logits 和 token loss|
|prefill|用户提供的 prompt|知道 prompt，不知道生成部分|一次计算 prompt 各位置 hidden|
|自回归 decode|模型之前生成的 token|不知道尚未生成部分|读取 cache，产生下一步 logits|

teacher forcing 是损失计算的条件协议，不是一个额外的概率分布。它让每个训练位置的 context 与数据记录中的真实前缀对齐；推理时前缀可能包含模型自己生成的 token，因此条件分布路径会发生变化。[Teacher Forcing](../rnn-lstm/teacher-forcing/)词条同时处理这个训练—推理差异和暴露偏差。

### 局部梯度仍然是 $p-y$

某一位置的 logits 为 $z\in\mathbb R^V$，softmax 概率为 $p$，目标 token 的 one-hot 向量为 $e_y$。单项 NLL 的 logits 梯度是

$$
\frac{\partial\ell}{\partial z_j}
=
p_j-\mathbb 1[j=y].
$$

批量 token-level mean 的梯度只是有效位置梯度的加权平均：

$$
\nabla_\theta\mathcal J_{\mathrm{token}}
=
\frac1M
\sum_{n,t}
m_{n,t}\nabla_\theta\ell_{n,t}.
$$

例如 $p=(0.6,0.3,0.1)$、目标类别为第一个位置时，局部 logits 梯度为

$$
\nabla_z\ell=(-0.4,0.3,0.1).
$$

梯度下降会提高目标 logit 相对于其他 logit 的相对位置。共享参数使这一局部信号经过输出 head、hidden、attention 和 embedding 汇总到其他 token 事件；“最大似然”描述的是这些局部事件的总目标，不要求它们使用不同参数。

### teacher forcing 不等于生成质量

训练 NLL 测量的是模型在真实前缀上给真实下一个 token 的概率。生成质量还受以下因素影响：

1. 生成时前缀由模型自己产生，错误会进入后续条件；
2. temperature、top-k、top-p 和 beam search 会改变选择规则；
3. EOS、长度上限和重复惩罚会改变停止或候选路径；
4. 训练上下文、评估文本和部署 prompt 可能来自不同分布。

因此训练 NLL 下降可以说明经验目标改善，不能单独推出采样文本的事实正确率、长程规划能力或人类偏好分数改善。若需要比较生成策略，应保持模型参数和评估 prompt 不变，单独记录解码规则与额外成本。

## tokenizer 和文档边界改变似然

### token 序列似然不是字符序列似然

固定 tokenizer $g$ 后，文本 $s$ 被映射为 token 序列：

$$
g(s)=x_{0:T}.
$$

语言模型优化的是

$$
p_\theta(x_{1:T}\mid x_0),
$$

而不是直接对字符位置求和。若两个 tokenizer 把同一段文本切成不同数量和不同边界的 token，它们产生不同的条件事件集合、不同的分母和不同的 perplexity。比较结果时必须固定 tokenizer 版本、词表、special token 和 token 计数；不能用“每个字符平均 loss”去替代 token-level NLL 而不改变指标名称。

若 tokenizer 允许同一文本有多个合法分解，文本概率还需要对所有分解求和；常见确定性 tokenizer 只选择一条编码路径，因此模型报告通常是该 tokenization 路径的概率。这个概率可以用于比较同一 tokenizer 下的模型，不应直接解释成与另一 tokenizer 同尺度的文本概率。

### EOS 规定样本何时结束

设内容 token 为 $a,b$。下面两种序列定义的联合概率不同：

$$
\begin{aligned}
\text{无 EOS:}\quad
p_\theta(a,b\mid\mathrm{BOS})
&=
p_\theta(a\mid\mathrm{BOS})
p_\theta(b\mid\mathrm{BOS},a),\\
\text{含 EOS:}\quad
p_\theta(a,b,\mathrm{EOS}\mid\mathrm{BOS})
&=
p_\theta(a\mid\mathrm{BOS})
p_\theta(b\mid\mathrm{BOS},a)
p_\theta(\mathrm{EOS}\mid\mathrm{BOS},a,b).
\end{aligned}
$$

第二种合同多出一个训练事件，也给生成循环提供了停止标签。评估时如果训练计数包含 EOS 而验证计数排除 EOS，平均 NLL 和 perplexity 就不在同一个事件集合上。PAD 也不能代替 EOS：PAD 表示张量填充，EOS 表示序列概率的结束。

### packing 决定哪些前缀属于同一事件

多个短样本被放进一个长度窗口时，至少有三种目标解释：

|packing 合同|前缀可见性|对应的似然对象|
| --- | --- | --- |
|独立窗口|每个窗口只读取自己的历史|每个窗口独立产生一条序列|
|EOS 串接|后一文档可读取前一文档结尾的 token|一个带 EOS 分隔符的长 token 序列|
|block-diagonal causal mask|不同 segment 互不读取|多个独立序列共享矩形计算布局|

输入 tensor 都可能是 $(B,S)$，但三者的 $c_{n,t}$ 不同，因而最大似然目标不同。只检查 batch shape 不能证明文档边界正确；需要打印 segment id、可见 key 集合和有效 loss mask。

## 数据采样和归约会改变被拟合的分布

### 重复数据等价于增加观测次数

在原始 MLE 中，每个有效观测事件在总 NLL 中贡献一次。若同一序列被重复三次，它的对数概率也会在目标中出现三次。重复可以来自显式复制、数据采样概率、shard 重访或文档去重失败；从目标函数看，它们都会增加对应事件的权重。

设每个事件的非负权重为 $w_{n,t}$，加权 token 目标为

$$
\mathcal J_w(\theta)
=
\frac{
\displaystyle\sum_{n,t}
w_{n,t}\ell_{n,t}
}{
\displaystyle\sum_{n,t}w_{n,t}
}.
$$

当所有 $w_{n,t}=1$ 时，它退化为普通 token-level MLE。若代码按来源、语言、质量分数或长度设置不同权重，目标是加权经验分布上的 NLL；报告中应说明权重的单位、分母和是否随训练动态变化。

### 数据混合概率不等于 token 份额

假设来源 A、B 的抽样概率分别为 $0.75、0.25$。如果 A 的平均文档长度是 B 的三倍，按文档抽样并不意味着有效 token 份额也是 $0.75、0.25$。如果代码按 token shard 抽样，结果又会不同。应同时记录：

|配置或观测|含义|不能直接替代什么|
| --- | --- | --- |
|source sampling probability|抽取某来源记录或 shard 的概率|实际有效 token 份额|
|raw token count|清洗后来源拥有的 token 数|训练中被重复消费的 token 数|
|effective token count|经过 mask 和边界后进入 loss 的 token 数|原始字符数|
|observed mixture|训练日志累计的来源 token 比例|配置文件中的目标权重|

数据配比改变的是经验分布。模型即使把加权目标降到很低，也只说明它更好地拟合了加权后的观测事件；不能把该结果直接解释成未加权部署分布上的风险下降。

### 长度归约是隐含的采样策略

每条序列等权会让短记录的每个 token 获得更高平均权重；每个 token 等权会让长记录在总目标中贡献更多事件。两种归约都可以是有意的统计选择，但要把名称写清楚：

|目标名称|每条记录的总贡献|适合回答的问题|
| --- | --- | --- |
|总 NLL / token-level mean|与有效 token 数成正比|模型在所有目标 token 上的平均概率|
|sequence-level mean|每条记录相等|模型对随机抽取一条记录的平均 NLL|
|source-balanced mean|每个来源先归一化再平均|各来源等权时的平均表现|

把 sequence-level mean 输出成普通 token perplexity，会同时隐藏序列权重和分母变化。评估报告需要注明归约顺序。

## MLE、MAP 和训练中常见的附加项

### 最大似然只包含观测概率

普通 next-token MLE 的训练目标是

$$
\mathcal J_{\mathrm{MLE}}(\theta)
=
-\sum_{n,t}
\log p_\theta(y_{n,t}\mid c_{n,t}),
$$

其中 $y_{n,t}$ 和 $c_{n,t}$ 来自已固定的数据协议。它没有自动包含参数先验、权重范数、学习率、早停或优化器状态。

MAP 在同一似然上加入参数先验 $\pi(\theta)$：

$$
\begin{aligned}
\widehat\theta_{\mathrm{MAP}}
&=
\operatorname*{arg\,max}_{\theta}
\left[
\log\mathcal L_D(\theta)+\log\pi(\theta)
\right]\\
&=
\operatorname*{arg\,min}_{\theta}
\left[
\mathcal J_{\mathrm{NLL}}(\theta)-\log\pi(\theta)
\right].
\end{aligned}
$$

例如高斯先验可以产生与 L2 惩罚相似的形式，但具体系数、参数化和优化器更新仍需单独声明。写“模型用 MLE 训练”时，通常指数据似然部分；若实际目标含 weight decay、KL 项或其他正则化，应把它们分开报告。

### label smoothing 不再是硬标签 MLE

硬标签的目标分布是 one-hot 向量 $e_y$。label smoothing 把它改成软目标 $q$，损失为

$$
\ell_{\mathrm{smooth}}
=
-\sum_{j=1}^{V}
q_j\log p_\theta(j\mid c).
$$

当 $q\neq e_y$ 时，训练目标不再是只对“观测 token 恰好为 $y$”的硬标签 NLL。它可以被解释为对目标分布的交叉熵，或作为抑制过度自信的正则化；但不能在不说明 $\varepsilon$ 和参考分布的情况下继续把数值称为普通 MLE。[标签平滑](../training-nn/label-smoothing/)处理目标分布改变后的梯度和评估边界。

### 优化算法不等于统计目标

同一个 MLE 目标可以由 SGD、Adam、不同 batch 顺序、不同初始化和不同停止时机得到不同参数。反过来，同一个 optimizer 也可以最小化带 mask、加权、正则化或多任务项的不同目标。需要把下面三层分开：

|层|决定什么|例子|
| --- | --- | --- |
|统计目标|哪些观测被赋予多大概率|序列 MLE、加权 NLL、soft target|
|参数化|哪些条件分布可以同时表达|共享 Transformer、词表 head、上下文长度|
|优化过程|从初始化走到哪个近似解|Adam、学习率、batch、checkpoint、早停|

训练 loss 下降说明优化过程找到了更低的当前目标值。它不单独证明模型族能表达目标分布，也不单独证明验证分布上的 NLL 下降。

## 独立数值核对

下面的计算固定两条 token 记录，并把每个列出的条件概率当作模型在真实前缀上的输出。脚本只使用标准库，独立重算序列概率、总 NLL、token-level mean 和 sequence-level mean。

|记录|真实 next-token 条件概率|序列概率|序列 NLL|
| --- | --- | ---: | ---: |
|A|$0.8,\ 0.5,\ 0.25$|$0.100000000000$|$2.302585092994$|
|B|$0.5,\ 0.4$|$0.200000000000$|$1.609437912434$|

数据集的联合似然、总 NLL 和两种平均为

$$
\begin{aligned}
\mathcal L_D
&=
0.8\cdot0.5\cdot0.25\cdot0.5\cdot0.4
=0.020000000000,\\
\mathcal J_{\mathrm{NLL}}
&=
2.302585092994+1.609437912434
=3.912023005428,\\
\mathcal J_{\mathrm{token}}
&=
\frac{3.912023005428}{5}
=0.782404601086,\\
\mathcal J_{\mathrm{sequence}}
&=
\frac12
\left(
\frac{2.302585092994}{3}
+\frac{1.609437912434}{2}
\right)
=0.786123660274.
\end{aligned}
$$

总 NLL 与 token-level mean 只相差固定的 $5$ 倍，因此在固定这两条记录时选择相同的 MLE 参数。sequence-level mean 给两条记录相同权重，数值不同；它回答的是“随机抽一条记录时的平均 NLL”，而不是“随机抽一个有效 token 时的平均 NLL”。

同一核对还可以检查 perplexity：

$$
\operatorname{PPL}_{\mathrm{token}}
=
\exp(0.782404601086)
=
2.186724147887.
$$

这组数值不代表一个实际模型的质量；它只验证乘积、对数、token 归约和指数变换之间的关系。实际报告还要附 tokenizer、EOS、mask、样本来源和评估版本。

## 训练与评估的可比性

### validation NLL 使用同一事件定义

验证集可以用与训练相同的 token-level NLL 评估：

$$
\widehat{\mathcal J}_{\mathrm{val}}
=
\frac{
\displaystyle\sum_{(c,y)\in\mathcal E_{\mathrm{val}}}
-\log p_\theta(y\mid c)
}{
\displaystyle\lvert\mathcal E_{\mathrm{val}}\rvert
}.
$$

要比较两个 checkpoint，至少固定：

1. tokenizer 和词表版本；
2. BOS、EOS、PAD 以及其是否进入 loss；
3. 文档切分、packing 和 attention mask；
4. 目标 token 的有效集合与分母；
5. 验证文本、去重规则和 contamination 检查；
6. logits 的温度处理，通常使用未截断的原始模型分布。

模型在验证集上给真实 token 的概率越高，NLL 越低。若验证集含有训练数据的重复或近似内容，数值仍然可以计算，但它的泛化解释需要降低确认程度。

### perplexity 只是 NLL 的指数刻度

对 token-level mean NLL：

$$
\operatorname{PPL}
=
\exp\left(
\widehat{\mathcal J}_{\mathrm{token}}
\right).
$$

它可以被解释成等效的平均分支数，但不是词表大小，也不是生成文本的困惑程度。不同 tokenizer 会改变 token 事件数量和事件粒度；不同 sequence reduction 也会改变指数前的平均值。报告 PPL 时应同时报告平均 NLL、有效 token 数、序列数和 tokenizer。

如果生成过程使用 temperature、top-k 或 top-p，评估 PPL 仍然应在原始模型 logits 上对真实标签计算。对截断后的采样分布计算另一个数值，可以作为采样策略指标，但不能继续标成同一个模型的原始 MLE perplexity。

## 失效模式：目标名称和实现不一致

### 把当前 token 当作标签

**现象：**训练 loss 异常低，模型可以直接复制输入。

**检查：**打印一条短序列的输入和标签，确认 $X_t=x_t$、$Y_t=x_{t+1}$；检查 causal mask 没有允许 query 读取未来标签。若实现把 $Y_t=X_t$，优化的就不是 next-token MLE。

### 把每条序列的平均当成 token MLE

**现象：**短记录被赋予比长记录更高的每 token 权重，报告却写成普通 token-level perplexity。

**检查：**同时计算总 NLL、有效 token 数、sequence-level mean 和 token-level mean，记录归约顺序。

### PAD 进入分母

**现象：**同一模型在 padding 比例不同的 batch 上出现不同 loss 尺度。

**检查：**统计有效 target token 数；确认 numerator 和 denominator 使用同一个 loss mask；不要用 batch 的矩形长度替代有效事件数。

### EOS 口径在训练和验证之间漂移

**现象：**训练和验证的 token 数无法对齐，生成停止规则与训练标签不一致。

**检查：**固定 EOS 是否作为标签、是否计入 PPL，以及生成循环遇到 EOS 后的处理。

### tokenizer 版本变化

**现象：**文本相同，但 token 数、loss 和 PPL 同时发生明显变化。

**检查：**保存 tokenizer 文件哈希、词表大小、special token id 和编码后的 token 数；不要只比较字符数。

### 采样概率被误读成最终配比

**现象：**配置文件写了来源权重，训练日志中的有效 token 份额却不同。

**检查：**按来源累计有效 target token、平均文档长度、重复次数和过滤后的样本数；标明抽样单位是文档、token 还是 shard。

### label smoothing 被隐藏在损失函数中

**现象：**报告称使用硬标签 MLE，但相同 logits 下的 loss 与 $-\log p_y$ 不一致。

**检查：**打印目标分布、平滑系数、参考分布和 reduction；把 soft-target cross-entropy 与硬标签 NLL 分开记录。

### 用 free-running 生成概率替代 teacher-forced likelihood

**现象：**评估脚本先采样文本，再只对采样路径统计概率，结果依赖 temperature 或 top-p。

**检查：**验证时直接把真实 token 作为 label，把真实前缀作为 context，使用未截断 logits；生成质量单独评估。

### 只看到 loss 下降就声称学到了真实分布

**现象：**训练 NLL 下降，但验证 NLL、来源分项或污染审计没有证据。

**检查：**同时报告训练/验证事件集合、分项 token 数、数据重复率、tokenizer、checkpoint 和评估协议。训练 MLE 只确认当前经验目标被优化。

## 可复用的核验协议

审计一个 next-token MLE 实现时，按以下顺序记录：

1. 固定 tokenizer、词表、BOS、EOS、PAD 和文档边界；
2. 保存一条短序列的 input ids、label ids、segment id 和 loss mask；
3. 检查 $Y_t=X_{t+1}$，确认第一个有效 label 的前缀起点；
4. 打印长度 4 的 causal mask，检查每个 query 的可见 key 集合；
5. 独立计算一行 softmax、目标概率、NLL 和 $p-\mathbb 1[j=y]$；
6. 分别计算总 NLL、token-level mean、sequence-level mean 和有效分母；
7. 按来源统计 sampling probability、原始 token 数和有效 token 数；
8. 记录 EOS、padding、packing、position reset 与 cross-document 可见性；
9. 记录 label smoothing、sample weight、weight decay、KL 项和学习率；
10. 在独立验证文本上用原始 logits 计算 NLL 与 PPL；
11. 将 teacher-forced validation 与 free-running generation 分开；
12. 保存 tokenizer、数据版本、checkpoint、优化器状态和评估配置。

实现可以使用 fused cross-entropy、packed sequence、混合精度或分布式并行，但这些优化不能改变有效条件事件、标签移位、可见前缀和损失分母。若改变了采样权重、序列归约或目标分布，应同步改变目标名称和报告字段。

## 相关词条

- [因果语言建模](../transformer-architectures/causal-language-modeling/)：固定右移输入、causal mask、teacher forcing 和 next-token logits。
- [最大似然](../probability/maximum-likelihood/)：从一般概率模型解释似然、对数似然和 NLL。
- [交叉熵](../information-theory/cross-entropy/)、[Softmax](../neurons-and-activations/softmax/)：推导 token 概率和 logits 梯度。
- [经验风险最小化](../learning-framework/empirical-risk-minimization/)：说明有限样本平均损失与总体风险的边界。
- [困惑度](../information-theory/perplexity/)：把平均 token NLL 转成指数刻度并固定评估分母。
- [Teacher Forcing](../rnn-lstm/teacher-forcing/)：区分真实前缀训练与模型前缀生成。
- [分词](../text-representation/tokenization/)：固定文本到 token 事件的编码单位。
- [标签平滑](../training-nn/label-smoothing/)：说明软目标如何改变硬标签 NLL。
- [预训练](../pretraining/pretraining/)：记录语料、token budget、packing 和可恢复训练状态。
- [Masked Language Modeling](../transformer-architectures/masked-language-modeling/)：比较双向 selected-token 目标与因果 next-token 目标。
- [Encoder-Decoder](../transformer-architectures/encoder-decoder/)：比较独立 target 轴上的序列似然。
