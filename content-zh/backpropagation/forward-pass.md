---
title: "前向计算：沿网络逐层得到输出和损失"
tags: ["why-models-learn"]
---

**前向计算**（forward pass，也叫前向传播）是给定输入和当前参数后，沿一次具体的依赖顺序计算中间表示、模型输出以及可选损失的过程。它只负责求值，不计算参数梯度，也不更新参数；反向阶段读取其中一部分前向值，才把损失对参数的敏感度传回来。把这条边界守住，才不会把“模型给出了一个预测”和“模型已经完成一次训练更新”混成同一件事。

![前向计算沿拓扑顺序把输入变成隐藏表示、logits、概率和损失](/assets/backpropagation/svg/forward-pass.1.svg)

本文先从单个神经元写出多层网络的前向方程，再用一个两层分类器逐项算出隐藏值、logits、概率和交叉熵。随后把同一公式改写成 batch 矩阵形式，说明广播与 reduction 如何改变形状和数值尺度，再处理回归头、训练态与推理态、序列和 mask、数值稳定性以及供反向阶段读取的缓存。最后给出可以逐层记录的核验表和常见失效模式。

## 前向计算究竟在求什么

把模型记为

$$
f_{\theta}:\mathcal X\longrightarrow\mathcal Y,
$$

其中 $\theta$ 是当前参数，$x$ 是一条输入。一次前向计算固定 $\theta$ 和 $x$，执行

$$
\hat y=f_{\theta}(x).
$$

训练时通常还会把标签 $y$ 交给损失函数：

$$
\ell=\ell(\hat y,y),
\qquad
L=\operatorname{reduce}(\ell_1,\ldots,\ell_B).
$$

这里有三个不同层次：

- $\hat y$ 是模型的输出，可能是回归值、logit、概率或一组序列分数；
- $\ell_i$ 是第 $i$ 个样本的损失，只有在选择了标签和损失后才存在；
- $L$ 是把一批样本的损失按 `sum`、`mean` 或其他规则合成为的训练目标。

参数更新属于另一个步骤。例如梯度下降写成

$$
\theta_{\text{new}}
=
\theta_{\text{old}}-\eta\nabla_{\theta}L,
$$

它不属于前向计算。一次训练迭代可以包含前向、反向和更新，但一次前向本身不会改变 $\theta$。

### 架构、一次执行和数值记录

“两层感知器”是架构描述；“这个输入在当前权重下经过两层仿射变换、ReLU 和 softmax”才是一次具体的前向执行。把这几个层次分开，可以避免下面这种含糊说法：模型“有一个输出层”，却没有说明输出是 logits 还是概率，也没有说明 loss 是否对 batch 求平均。

| 对象 | 记录的内容 | 是否依赖本次输入 |
| --- | --- | --- |
| 架构 | 层的种类、宽度、连接和参数形状 | 通常不依赖 |
| 参数 | 权重、偏置、归一化参数 | 依赖训练状态，不依赖单条输入 |
| 前向节点 | 预激活、激活、logits、mask、归约值 | 依赖本次输入和模式 |
| 模型输出 | $\hat y$ 或 logits | 依赖本次输入 |
| 样本损失 | 每个样本的 $\ell_i$ | 依赖输入和标签 |
| batch 损失 | `sum`、`mean` 或加权归约结果 | 依赖 batch 组成 |

[计算图](../backpropagation/computational-graphs/)描述这些值之间的依赖边；前向计算则沿一次拓扑顺序实际填入节点值。相同架构在不同输入、不同 mask 或不同训练模式下，可能走过不同的具体节点。

## 从单个神经元到多层网络

单个神经元先做仿射变换，再做逐分量或标量激活。对输入 $x\in\mathbb R^{d}$，权重 $w\in\mathbb R^d$，偏置 $b\in\mathbb R$，有

$$
z=w^{\mathsf T}x+b,
\qquad
a=\phi(z).
$$

$z$ 叫预激活值，$a$ 是送往下一层的表示。若 $\phi$ 是 ReLU，

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

前向阶段只需读 $w$、$x$ 和 $b$，算出一个标量。多个神经元把标量堆成向量。设第 $l$ 层输入是 $a^{(l-1)}\in\mathbb R^{d_{l-1}}$，输出宽度是 $d_l$，则

$$
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)},
\qquad
a^{(l)}=\phi^{(l)}\bigl(z^{(l)}\bigr),
$$

其中

$$
W^{(l)}\in\mathbb R^{d_l\times d_{l-1}},
\qquad
b^{(l)}\in\mathbb R^{d_l}.
$$

最后一层的激活不一定和隐藏层相同。一个常见的多分类网络写成

$$
a^{(0)}=x,
$$

$$
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)},
\quad
a^{(l)}=\phi^{(l)}(z^{(l)}),
\quad
l=1,\ldots,L-1,
$$

$$
o=W^{(L)}a^{(L-1)}+b^{(L)},
\qquad
p=\operatorname{softmax}(o).
$$

这里 $o$ 是 logits，$p$ 才是概率向量。若真实类别为 $k$，softmax 交叉熵是

$$
\ell=-\log p_k.
$$

不要把最后一步固定写成 softmax。输出头取决于目标变量的结构：

| 任务 | 前向输出 | 常见逐样本损失 | 需要检查的性质 |
| --- | --- | --- | --- |
| 标量回归 | $\hat y=o\in\mathbb R$ | MSE、MAE 或 Huber | 输出单位与范围 |
| 多输出回归 | $\hat y=o\in\mathbb R^m$ | 分量损失再归约 | 输出维度和分量权重 |
| 二分类 | logit $o$ 或概率 $\sigma(o)$ | 带 logits 的 BCE | 不要重复 sigmoid |
| 互斥多分类 | logits $o$ 或概率 $p$ | softmax 交叉熵 | softmax 轴与标签索引 |
| 多标签分类 | 每个标签一个 logit | 多个 BCE 的归约 | 标签不是互斥类别 |

[激活函数](../neurons-and-activations/activation-functions/)决定隐藏层的局部变换；[softmax](../neurons-and-activations/softmax/)只在需要互斥概率时承担归一化。输出头和损失的组合不能只按“最后一层也用激活函数”的习惯决定。

### 每一层的形状

用列向量表示一条样本时，第 $l$ 层的形状关系可以写成：

| 量 | 形状 | 前向作用 |
| --- | --- | --- |
| $a^{(l-1)}$ | $d_{l-1}$ | 上一层表示 |
| $W^{(l)}$ | $d_l\times d_{l-1}$ | 把输入方向组合成输出方向 |
| $b^{(l)}$ | $d_l$ | 每个输出单元的平移 |
| $z^{(l)}$ | $d_l$ | 激活前的向量 |
| $a^{(l)}$ | $d_l$ | 激活后的向量 |

矩阵乘法的内维必须相同：$W^{(l)}$ 的列数是 $d_{l-1}$，输出的行数是 $d_l$。如果实际代码采用行向量 batch 约定，公式会把 $W$ 放在右边；这不是两套模型，而是同一个线性映射的存储约定。

## 一个两层分类器的完整数值例子

选一个输入维度为 $2$、隐藏宽度为 $2$、输出类别为 $2$ 的网络。用列向量约定，令

$$
x=\begin{bmatrix}1\\2\end{bmatrix},
\qquad
W^{(1)}=\begin{bmatrix}1&-1\\0.5&0.5\end{bmatrix},
\qquad
b^{(1)}=\begin{bmatrix}2\\0\end{bmatrix}.
$$

隐藏层使用 ReLU，输出层不加逐分量激活，输出矩阵和偏置为

$$
W^{(2)}=\begin{bmatrix}1&-1\\-1&1\end{bmatrix},
\qquad
b^{(2)}=\begin{bmatrix}0\\0\end{bmatrix}.
$$

真实标签是第二类，即 $k=2$。把每一步写开：

### 第一层仿射变换

$$
z^{(1)}
=
W^{(1)}x+b^{(1)}
=
\begin{bmatrix}1&-1\\0.5&0.5\end{bmatrix}
\begin{bmatrix}1\\2\end{bmatrix}
+
\begin{bmatrix}2\\0\end{bmatrix}
=
\begin{bmatrix}1\\1.5\end{bmatrix}.
$$

第一行得到 $1\cdot1+(-1)\cdot2+2=1$，第二行得到 $0.5\cdot1+0.5\cdot2=1.5$。这两个数仍然是预激活值，尚未经过非线性。

### 隐藏表示

$$
h=\operatorname{ReLU}\bigl(z^{(1)}\bigr)
=
\begin{bmatrix}\max(0,1)\\ \max(0,1.5)\end{bmatrix}
=
\begin{bmatrix}1\\1.5\end{bmatrix}.
$$

这次两个分量都在 ReLU 的正侧，所以没有一个分量被截成零。换一个输入时，某个预激活可能为负，后面的层就会看到不同的稀疏表示。

### 输出 logits

$$
o
=W^{(2)}h+b^{(2)}
=
\begin{bmatrix}1&-1\\-1&1\end{bmatrix}
\begin{bmatrix}1\\1.5\end{bmatrix}
=
\begin{bmatrix}-0.5\\0.5\end{bmatrix}.
$$

此时还不能把 $o$ 当作概率。它的两个分量之和是 $0$，但 logits 没有必须求和为 $1$ 的约束，也可以是负数。

### softmax 概率与交叉熵

$$
p_i=\frac{\exp(o_i)}{\exp(o_1)+\exp(o_2)},
$$

所以

$$
p
=
\begin{bmatrix}
\dfrac{e^{-0.5}}{e^{-0.5}+e^{0.5}}\\
\dfrac{e^{0.5}}{e^{-0.5}+e^{0.5}}
\end{bmatrix}
=
\begin{bmatrix}
0.268941421370\\
0.731058578630
\end{bmatrix}.
$$

目标是第二类，因此

$$
\ell=-\log p_2=0.313261687518.
$$

可以把这次执行压成一张逐阶段记录表：

| 阶段 | 运算 | 数值 | 形状 |
| --- | --- | --- | --- |
| 输入 | $x$ | $(1,2)$ | $2$ |
| 隐藏预激活 | $W^{(1)}x+b^{(1)}$ | $(1,1.5)$ | $2$ |
| 隐藏激活 | $\operatorname{ReLU}(z^{(1)})$ | $(1,1.5)$ | $2$ |
| 输出仿射 | $W^{(2)}h+b^{(2)}$ | $(-0.5,0.5)$ | $2$ |
| 概率头 | $\operatorname{softmax}(o)$ | $(0.268941421370,0.731058578630)$ | $2$ |
| 样本损失 | $-\log p_2$ | $0.313261687518$ | 标量 |

这张表既能作为解释，也能作为最小回归测试：如果以后更换矩阵乘法 kernel、精度或输出头，至少应核对每一个中间节点，而不是只看最后一个 loss。

### 换一个输出头，前向问题就不同

保留同一个隐藏表示 $h=(1,1.5)$，改成标量回归头

$$
w_{\mathrm{reg}}=\begin{bmatrix}0.4\\-0.2\end{bmatrix},
\qquad
b_{\mathrm{reg}}=0.1.
$$

输出为

$$
\hat y=w_{\mathrm{reg}}^{\mathsf T}h+b_{\mathrm{reg}}
=0.4\cdot1-0.2\cdot1.5+0.1=0.2.
$$

若标签是 $y=0.5$，使用半平方误差得到

$$
\ell=\frac12(\hat y-y)^2=0.045.
$$

隐藏层的前向值没有变，变的是输出头、输出的解释和损失。由此可以看出，“最后输出是一个向量”还不足以判断它代表 logits、概率还是连续目标。

## batch 不是把单样本公式藏起来

实际训练通常一次处理 $B$ 条样本。若把每条样本放在矩阵的一行，令

$$
X\in\mathbb R^{B\times d_0},
$$

则列向量公式的转置版本是

$$
Z^{(l)}=A^{(l-1)}\bigl(W^{(l)}\bigr)^{\mathsf T}+\mathbf 1_B\bigl(b^{(l)}\bigr)^{\mathsf T},
$$

$$
A^{(l)}=\phi^{(l)}\bigl(Z^{(l)}\bigr),
$$

其中

$$
A^{(l-1)}\in\mathbb R^{B\times d_{l-1}},
\quad
W^{(l)}\in\mathbb R^{d_l\times d_{l-1}},
\quad
b^{(l)}\in\mathbb R^{d_l},
\quad
Z^{(l)}\in\mathbb R^{B\times d_l}.
$$

$\mathbf 1_B$ 是长度为 $B$ 的全一列向量。它把同一个 bias 复制到每一行；实现里的 broadcasting 只是这次复制的紧凑表示，不是新的数学运算。

用上一个例子的参数，取两条输入

$$
X=
\begin{bmatrix}1&2\\2&1\end{bmatrix}.
$$

第一层得到

$$
Z^{(1)}
=X\bigl(W^{(1)}\bigr)^{\mathsf T}+\mathbf 1_2\bigl(b^{(1)}\bigr)^{\mathsf T}
=
\begin{bmatrix}1&1.5\\3&1.5\end{bmatrix},
$$

因此

$$
H=\operatorname{ReLU}\bigl(Z^{(1)}\bigr)
=
\begin{bmatrix}1&1.5\\3&1.5\end{bmatrix}.
$$

第二层的 logits 是

$$
O=H\bigl(W^{(2)}\bigr)^{\mathsf T}
=
\begin{bmatrix}-0.5&0.5\\1.5&-1.5\end{bmatrix}.
$$

沿最后一维分别做 softmax，得到近似概率

$$
P=
\begin{bmatrix}
0.268941421370&0.731058578630\\
0.952574126822&0.047425873178
\end{bmatrix}.
$$

假设第一条样本的标签是第二类，第二条样本的标签是第一类，则

$$
\ell_1=-\log 0.731058578630=0.313261687518,
\qquad
\ell_2=-\log 0.952574126822=0.048587351574.
$$

若 loss 使用平均归约，

$$
L_{\mathrm{mean}}
=\frac{\ell_1+\ell_2}{2}
=0.180924519546.
$$

若使用求和归约，

$$
L_{\mathrm{sum}}=\ell_1+\ell_2=0.361849039092.
$$

两者的预测概率相同，但 loss 相差 $2$ 倍；反向阶段的梯度也会相差相同的归约尺度。写训练日志时，应同时记录 batch size 和 reduction 规则。

### 广播、归约和轴

batch 公式里最容易被省略的是“沿哪个轴做什么”。下面的几件事看起来都叫加法或平均，形状语义却不同：

| 操作 | 输入形状 | 输出形状 | 前向含义 |
| --- | --- | --- | --- |
| bias 广播 | $B\times d$ 加 $d$ | $B\times d$ | 每个样本使用同一个 bias |
| 类别 softmax | $B\times C$ | $B\times C$ | 每一行内部的 $C$ 个类竞争 |
| token softmax | $B\times T\times C$ | $B\times T\times C$ | 每个位置沿类别轴归一化 |
| batch mean | $B$ 个样本损失 | 标量 | 跨样本求平均 |
| feature sum | $B\times d$ | $B$ 或 $1\times d$ | 沿指定特征轴归约 |

例如 $B\times T\times C$ 的 logits，若任务是每个 token 分类，softmax 应沿 $C$；若任务是注意力权重，归一化轴可能是键位置 $T$。同一个张量形状不能单独告诉你轴的含义，必须结合任务定义和 mask。

### 变长序列和 mask

对于长度不同的序列，通常把 batch 补齐成相同的 $T$，再用 mask 标出真实位置。若补齐位置不应参与分类 loss，可写成

$$
L
=
\frac{\displaystyle\sum_{b=1}^{B}\sum_{t=1}^{T}m_{b,t}\,\ell_{b,t}}
{\displaystyle\sum_{b=1}^{B}\sum_{t=1}^{T}m_{b,t}},
$$

其中 $m_{b,t}\in\{0,1\}$。分母是有效位置数，而不是固定的 $BT$；如果直接对补齐后的整张 loss 矩阵求 mean，短序列会因为补齐位置改变权重。

注意 mask 的两个作用不能混淆：它可以在前向时阻止无效位置进入归一化，也可以在 loss reduction 时排除无效标签。attention mask、padding loss mask 和 causal mask 的形状可能不同，不能看到一个布尔张量就默认它们语义相同。

## 输出、logits、概率和损失的边界

同一个数值向量在不同节点上的名字不是装饰。对二分类，logit $o$ 与概率 $p=\sigma(o)$ 的含义不同：

$$
o\in\mathbb R,
\qquad
p=\frac{1}{1+e^{-o}}\in(0,1).
$$

如果损失函数已经实现了“带 logits 的 BCE”，就应把 $o$ 直接交给它；先手动算 sigmoid 再交给同一个 logits 接口，会重复应用 sigmoid。多分类也类似：softmax 交叉熵通常直接接收 logits，并在内部用稳定的 log-sum-exp 计算。

输出解释可以按下面的顺序记录：

1. 节点产生了什么数值，是连续值、logit、概率还是归一化权重；
2. 数值沿哪个轴解释，batch、token、类别和特征是否分开；
3. 标签与输出的坐标如何对应；
4. loss 在哪些位置、以什么权重做 reduction。

只写“模型输出为 $p$”不够。至少要说明 $p$ 是由哪个 logit 轴归一化得到，以及空 mask、异常标签和 padding 怎么处理。

## 训练态和推理态不是同一条前向路径

有些层在训练时需要随机量或 batch 统计，推理时则使用固定规则。因此同一组参数和同一输入，在两种模式下可能得到不同的前向值。

### Dropout 的例子

设隐藏向量 $h=(2,4)$，dropout 概率为 $q=0.5$。训练时取 mask $m=(1,0)$，使用 inverted dropout：

$$
\widetilde h
=\frac{m\odot h}{1-q}
=\frac{(1,0)\odot(2,4)}{0.5}
=(4,0).
$$

因为每个分量保留的概率是 $1-q$，有

$$
\mathbb E[\widetilde h]=h.
$$

推理时不再抽 mask，直接使用 $h=(2,4)$。如果把训练态的随机 mask 错带进评估，预测会随每次调用变化；如果训练时忘记缩放，训练与推理的表示尺度会不一致。

### Batch normalization 的例子

对一个 batch 的某个通道，训练态可以使用当前 batch 的均值和方差：

$$
\mu_B=\frac1B\sum_{i=1}^{B}x_i,
\qquad
\sigma_B^2=\frac1B\sum_{i=1}^{B}(x_i-\mu_B)^2,
$$

$$
\widehat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}},
\qquad
y_i=\gamma\widehat x_i+\beta.
$$

推理态通常读取训练期间累积的 running mean 和 running variance，而不使用当前评估 batch 的统计量。于是“验证 batch 很小”不仅是速度问题，也可能改变前向输出。应把模式、统计量来源和 batch size 一起记录。

并非所有层都有训练/推理差异。线性层、固定 ReLU 和确定性的矩阵乘法在两种模式下通常相同；但只要图中存在 dropout、batch normalization、随机增强或采样，就必须显式确认模式。

## 数值稳定性属于前向语义的一部分

### softmax 先减最大值

朴素 softmax 写成

$$
p_i=\frac{e^{o_i}}{\sum_{j=1}^{C}e^{o_j}}.
$$

当 logits 很大时，$e^{o_i}$ 可能溢出。令 $m=\max_i o_i$，可以改写为

$$
p_i
=\frac{e^{o_i-m}}{\sum_{j=1}^{C}e^{o_j-m}}.
$$

分子分母同时乘上 $e^{-m}$，概率没有改变；但指数的最大输入现在是 $0$。例如 $o=(1000,999)$ 时，稳定计算得到

$$
p=(0.731058578630,0.268941421370),
$$

目标为第一类的交叉熵仍是 $0.313261687518$。如果实现先直接计算 $e^{1000}$，得到的可能是 `inf`，随后出现 `inf / inf` 的 `NaN`，即使数学结果本来完全正常。

### log-sum-exp 与带 logits 的损失

多分类交叉熵可以合并为

$$
\ell(o,k)
=-o_k+\log\left(\sum_{j=1}^{C}e^{o_j}\right).
$$

令 $m=\max_j o_j$ 后，稳定形式是

$$
\ell(o,k)
=-o_k+m+\log\left(\sum_{j=1}^{C}e^{o_j-m}\right).
$$

这解释了为什么“softmax 后再取 log”不是实现层面的必需步骤。前向接口若接收 logits，就可以在一次稳定的 log-sum-exp 中同时计算 loss 和需要的对数概率。

### 精度、设备和无效值

即使形状正确，以下情况仍会让前向结果失真：

- 半精度指数、平方或归约发生溢出；
- 低精度累加把很多小量抹掉；
- CPU 与 GPU 使用不同的舍入和归约顺序；
- 参数、输入和 mask 位于不同设备，或 dtype 自动提升与预期不同；
- 一个前向节点先产生 `NaN`，后续层只把它传播得更远。

调试时先检查输入、每层预激活和每层输出的有限性，再检查最终 loss。对概率头还应检查非负性和归一化；对 ReLU 等激活应检查输出范围；对 mask 归约应检查有效计数是否为零。

## 前向值为什么要保存

前向计算只求值，但反向阶段通常需要读取其中一些值。例如对

$$
a=\phi(z),
$$

局部导数 $\phi'(z)$ 可能由 $z$ 或 $a$ 计算；对 dropout，需要原来的 mask；对矩阵乘法，需要输入和另一侧的参数。一次执行记录中常见的缓存包括：

| 缓存 | 反向或调试用途 | 能否随意删除 |
| --- | --- | --- |
| 输入 $a^{(l-1)}$ | 计算权重梯度 | 不能，除非可重算 |
| 预激活 $z^{(l)}$ | 判断激活导数或重现输出 | 视激活而定 |
| 激活 $a^{(l)}$ | 下一层梯度与输出检查 | 视实现而定 |
| dropout mask | 恢复同一条随机路径 | 不能直接丢 |
| batch 统计量 | 重现归一化局部规则 | 不能与推理统计混用 |
| reduction 有效计数 | 恢复 mean 或 mask 权重 | 不能省略 |

保存全部激活会消耗内存。[反向传播](../backpropagation/backpropagation/)中的激活检查点可以只保存部分节点，在反向前重算其余前向片段；这改变的是存储和计算的取舍，不改变前向函数。调试时应先关闭重算和异步融合，以便每个节点的记录仍然对应源代码中的运算。

前向求值和梯度记录也不是同一层开关。推理可以关闭梯度追踪，但仍然要执行前向；训练可以暂时只求前向输出，但如果没有建立需要的记录，随后就不能凭空得到反向梯度。

## 一份可执行的前向核验表

给定一条输入和一组参数，可以按以下顺序记录。每一项都对应一个可观察的数值或形状，而不是只凭最终 loss 猜测实现正确。

| 检查项 | 要记录的问题 | 最小证据 |
| --- | --- | --- |
| 输入 | 输入的 batch、通道、token 顺序是否明确 | 输入 shape 与一条具体样本 |
| 参数 | 权重矩阵的行列方向是否固定 | 每层 $W$、$b$ 的 shape |
| 预激活 | 仿射结果是否符合手算或矩阵乘法 | $z^{(l)}$ 的一行或一个切片 |
| 激活 | 激活、mask 和训练模式是否正确 | $a^{(l)}$、mask 或模式名 |
| 输出头 | 输出是值、logit 还是概率 | 输出定义与数值范围 |
| 轴 | softmax、归约和 mask 沿哪一轴运行 | axis、有效计数和 keepdim |
| 标签 | 标签坐标是否与输出坐标一致 | 类别索引或回归目标 shape |
| 损失 | 每样本损失如何变成 batch loss | `sum`、`mean` 或加权公式 |
| 数值 | 是否存在溢出、下溢、NaN 或 Inf | 各层有限性与范围统计 |
| 模式 | 训练态和推理态是否选对统计量 | mode、随机种子、running stats |

对于上面的两层分类器，最小固定探针可以写成：输入 $x=(1,2)$，隐藏预激活 $(1,1.5)$，logits $(-0.5,0.5)$，概率 $(0.268941421370,0.731058578630)$，第二类交叉熵 $0.313261687518$。如果这五个节点都一致，才有理由进一步检查反向梯度；如果只有最终 loss 一致，却有中间节点不一致，不能把它当成通过。

## 失效模式

**把前向和更新写成一个动作。** 前向得到 $\hat y$ 和 $L$；反向得到 $\nabla_\theta L$；优化器才改变 $\theta$。混在一个函数名或日志行里，会让人误以为参数在产生预测的同时已经改变。

**把 logits 当概率。** logits 可以为负，也不要求和为 $1$。只有经过合适的归一化头后，才可以按概率解释。

**重复应用输出激活。** `BCE with logits` 已经包含稳定的 sigmoid 逻辑；softmax 交叉熵通常已经包含 log-sum-exp。重复应用会改变损失数值并损害极端值稳定性。

**矩阵方向错一层。** 列向量写法需要 $W a$，行向量 batch 写法需要 $A W^{\mathsf T}$。两者都能在某些方阵例子中运行，却可能在非方阵或参数梯度上暴露错误。

**bias 广播到错误轴。** $B\times d$ 的 bias 应是长度 $d$ 的向量；把长度 $B$ 的样本权重当成 bias，可能因为广播规则碰巧成功而不报形状错误。

**把 batch mean 当 batch sum。** 两种 reduction 的预测不变，但 loss 和梯度的尺度不同。跨 batch size 比较训练日志时，必须保留 reduction 记录。

**沿错误轴做 softmax。** $B\times T\times C$ 张量沿 $T$ 归一化和沿 $C$ 归一化都能得到和为 $1$ 的数，却表达了完全不同的竞争集合。

**训练态和推理态混用。** dropout 的随机 mask、batch normalization 的统计量和随机采样都会改变一次具体前向。评估前没有切到推理模式，结果可能不稳定或与部署不一致。

**只检查最终 loss。** 中间层的转置、广播和 mask 错误可能在某个对称样例上相互抵消。应固定非方阵、含负值和带 padding 的小探针，逐节点记录。

**忽略有限性。** `NaN` 往往在指数、除零、平方或第一次无效归约处产生；最终 loss 只是最晚暴露的位置。应按前向顺序定位第一个非有限节点。

**把重算当成不同的模型。** 激活检查点只改变何时保存和重算前向值；如果随机层没有保存相同 mask，重算路径就不再等价，反向会读到另一条执行轨迹。

## 相关词条

- [神经元是什么](../neurons-and-activations/what-is-a-neuron/)：前向计算中单个仿射单元的结构起点。
- [激活函数](../neurons-and-activations/activation-functions/)：解释隐藏层为什么在仿射变换后加入逐分量非线性。
- [计算图](../backpropagation/computational-graphs/)：定义前向节点、依赖边、拓扑序和一次执行记录。
- [softmax](../neurons-and-activations/softmax/)：展开 logits 到概率向量的归一化、轴和稳定计算。
- [损失函数](../training-nn/loss-functions/)：比较预测与标签并定义逐样本目标。
- [反向传播](../backpropagation/backpropagation/)：读取前向值后沿反向拓扑序累加梯度。
- [计算图上的链式法则](../backpropagation/chain-rule-on-graphs/)：把每个前向节点的局部导数接成全局导数。
- [单神经元反向传播](../backpropagation/backprop-single-neuron/)：在最小前向网络上手算参数梯度。
- [向量化反向传播](../backpropagation/vectorized-backprop/)：把 batch 前向节点对应的梯度写成矩阵形式。
- [梯度检查](../backpropagation/gradient-checking/)：用有限差分核对一条前向计算及其反向结果。
- [PyTorch 自动微分](../backpropagation/autodiff-in-pytorch/)：观察框架如何记录前向图并调用反向阶段。
- [Dropout](../training-nn/dropout/)：训练态随机丢弃表示、推理态关闭随机 mask 的正则化方法。
- [批归一化](../training-nn/batch-normalization/)：对比当前 batch 统计量与推理 running 统计量的前向差异。
