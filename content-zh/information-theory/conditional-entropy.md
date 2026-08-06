---
title: "条件熵：知道一部分之后还剩多少不确定性"
tags: ["why-models-learn"]
---

**条件熵**是已经观察到随机变量 $X$ 后，随机变量 $Y$ 剩余不确定性的平均值：

$$
\begin{aligned}
H_b(Y\mid X)
&=\mathbb E_{X}\left[H_b(Y\mid X=x)\right]\\
&=-\sum_{x,y}p(x,y)\log_b p(y\mid x)
\end{aligned}
$$

这里的 $H_b(Y\mid X=x)$ 是固定某一个 $x$ 后对 $Y$ 的熵，而 $H_b(Y\mid X)$ 还要按照 $p(x)$ 对所有条件加权。知道 $X$ 通常会让预测 $Y$ 更容易，因此离散变量满足 $H(Y\mid X)\leq H(Y)$；但这个不等式只对平均值成立，不保证每一个具体条件的熵都更小。本篇从条件分布的定义开始，推导联合熵的链式法则和条件化不增熵，再连接噪声信道、序列预测、条件交叉熵和语言模型的 next-token 损失。

## 先把随机性按已知信息分层

对满足 $p(x)>0$ 的条件，条件概率是

$$
p(y\mid x)=\frac{p(x,y)}{p(x)}
$$

固定 $x$ 后，它关于 $y$ 仍然是一个完整的概率分布：

$$
p(y\mid x)\geq0,
\qquad
\sum_y p(y\mid x)=1
$$

因此可以先计算这一层的熵：

$$
H_b(Y\mid X=x)
=-\sum_y p(y\mid x)\log_b p(y\mid x)
$$

但在真正观察 $X$ 之前，我们不知道将会落在哪个条件分支。对所有分支按 $p(x)$ 平均：

$$
\begin{aligned}
H_b(Y\mid X)
&=\sum_xp(x)H_b(Y\mid X=x)\\
&=-\sum_xp(x)\sum_yp(y\mid x)\log_bp(y\mid x)\\
&=-\sum_{x,y}p(x,y)\log_bp(y\mid x)
\end{aligned}
$$

这个最后的形式最适合推导。它的权重是联合概率 $p(x,y)$，但对数里的对象是给定 $x$ 后预测 $y$ 的概率。若 $p(x,y)=0$，该项约定为 0；若 $p(x)>0$ 而 $p(y\mid x)=0$，因为对应联合概率也为 0，它不会给真实分布的条件熵贡献无穷项。

条件熵不是「先算一个联合熵再随意除以另一个熵」。它是对每个条件分布各自计算熵，再用条件出现的概率做加权平均。

## 数字例子：两个条件分支的平均

令 $X,Y$ 都是二元变量，$X$ 的两个取值各以概率 $1/2$ 出现。给定 $X$ 后，$Y$ 的分布如下：

| $x$ | $p(x)$ | $p(Y=0\mid x)$ | $p(Y=1\mid x)$ | $H_2(Y\mid X=x)$ |
| --- | --- | --- | --- | --- |
| 0 | $0.5$ | $0.9$ | $0.1$ | $0.4690$ bit |
| 1 | $0.5$ | $0.2$ | $0.8$ | $0.7219$ bit |

第一行的条件熵是二元熵函数

$$
\begin{aligned}
H_2(Y\mid X=0)
&=-0.9\log_2 0.9-0.1\log_2 0.1\\
&\approx0.4690\ \mathrm{bits}
\end{aligned}
$$

第二行同理：

$$
\begin{aligned}
H_2(Y\mid X=1)
&=-0.2\log_2 0.2-0.8\log_2 0.8\\
&\approx0.7219\ \mathrm{bits}
\end{aligned}
$$

所以条件熵不是两行相加，而是按照 $p(x)$ 平均：

$$
\begin{aligned}
H_2(Y\mid X)
&=0.5\times0.4690+0.5\times0.7219\\
&\approx0.5955\ \mathrm{bits}
\end{aligned}
$$

对应的联合概率矩阵是

$$
p(x,y)
=
\begin{pmatrix}
0.45&0.05\\
0.10&0.40
\end{pmatrix}
$$

例如左上角的 $0.45$ 来自 $p(X=0)p(Y=0\mid X=0)=0.5\times0.9$。边缘化掉 $X$ 后：

$$
p(Y=0)=0.45+0.10=0.55,
\qquad
p(Y=1)=0.05+0.40=0.45
$$

所以不知道 $X$ 时，$Y$ 的熵为

$$
\begin{aligned}
H_2(Y)
&=-0.55\log_2 0.55-0.45\log_2 0.45\\
&\approx0.9928\ \mathrm{bits}
\end{aligned}
$$

知道 $X$ 后，平均剩余不确定性从 $0.9928$ bit 降到了 $0.5955$ bit。减少的部分不是凭空消失，而是 $X$ 对 $Y$ 提供了可预测的信息。

## 链式法则：联合熵拆成先后两笔

联合熵把一对结果当成一个整体：

$$
H(X,Y)=-\sum_{x,y}p(x,y)\log p(x,y)
$$

联合概率可以分解为

$$
p(x,y)=p(x)p(y\mid x)
$$

代入并使用对数的乘法法则：

$$
\begin{aligned}
H(X,Y)
&=-\sum_{x,y}p(x,y)\log\left[p(x)p(y\mid x)\right]\\
&=-\sum_{x,y}p(x,y)\log p(x)
  -\sum_{x,y}p(x,y)\log p(y\mid x)\\
&=-\sum_xp(x)\log p(x)
  -\sum_{x,y}p(x,y)\log p(y\mid x)\\
&=H(X)+H(Y\mid X)
\end{aligned}
$$

这就是熵的链式法则：

$$
H(X,Y)=H(X)+H(Y\mid X)
$$

同一份联合分布也可以先描述 $Y$，再描述给定 $Y$ 后的 $X$：

$$
H(X,Y)=H(Y)+H(X\mid Y)
$$

因此

$$
H(Y\mid X)=H(X,Y)-H(X)
$$

和

$$
H(X\mid Y)=H(X,Y)-H(Y)
$$

在上面的数字例子中，$H_2(X)=1$ bit，所以

$$
H_2(X,Y)
=1+0.595461844238
=1.595461844238\ \mathrm{bits}
$$

而

$$
H_2(X\mid Y)
=1.595461844238-0.992774453988
=0.602687390250\ \mathrm{bits}
$$

$H(Y\mid X)$ 与 $H(X\mid Y)$ 可以不同，因为两个方向的条件分布不同。联合熵本身对交换变量不敏感，但「先说谁、已知谁后再说谁」的拆分会改变。

### 一串变量的链式法则

把两变量公式反复展开，可以得到

$$
H(X_1,\ldots,X_n)
=\sum_{t=1}^nH(X_t\mid X_1,\ldots,X_{t-1})
$$

第一项约定为 $H(X_1)$。这条公式说的是：描述一串结果时，第 $t$ 项所需的平均信息量取决于前面已经知道的全部内容。

若变量独立同分布，则过去不会改变当前分布：

$$
H(X_t\mid X_1,\ldots,X_{t-1})=H(X_t)=H(X_1)
$$

于是

$$
H(X_1,\ldots,X_n)=nH(X_1)
$$

若变量是一阶马尔可夫链，满足

$$
p(x_t\mid x_1,\ldots,x_{t-1})=p(x_t\mid x_{t-1})
$$

则只需要最后一个状态：

$$
H(X_1,\ldots,X_n)
=H(X_1)+\sum_{t=2}^nH(X_t\mid X_{t-1})
$$

历史信息并没有凭空消失，而是通过当前状态保留了预测未来所需要的部分。

## 条件化平均不会增加离散熵

条件熵不超过未条件化的熵：

$$
H(Y\mid X)\leq H(Y)
$$

关键是离散熵函数的凹性。对每一个 $x$，条件分布 $p(\cdot\mid x)$ 是一个概率向量，而边缘分布是这些向量的混合：

$$
p(y)=\sum_xp(x)p(y\mid x)
$$

熵的凹性给出

$$
\begin{aligned}
H\left(\sum_xp(x)p(\cdot\mid x)\right)
&\geq\sum_xp(x)H\left(p(\cdot\mid x)\right)\\
H(Y)&\geq H(Y\mid X)
\end{aligned}
$$

等号成立的典型条件是 $X$ 与 $Y$ 独立，即对所有有正概率的 $x$：

$$
p(y\mid x)=p(y)
$$

这时观察 $X$ 不会改变对 $Y$ 的预测分布。两者之间的差值可以写成

$$
\begin{aligned}
I(X;Y)
&=H(Y)-H(Y\mid X)\\
&=\sum_{x,y}p(x,y)\log\frac{p(y\mid x)}{p(y)}\\
&=D_{\mathrm{KL}}\left(p(x,y)\,\middle\Vert\,p(x)p(y)\right)
\geq0
\end{aligned}
$$

这个差值叫互信息；它衡量知道 $X$ 平均减少了多少关于 $Y$ 的不确定性，完整定义和性质见后续的 [互信息](../information-theory/mutual-information/)。

上面的数字例子正好给出

$$
I(X;Y)
=0.992774453988-0.595461844238
=0.397312609750\ \mathrm{bits}
$$

## 三个边界例子

### 独立：知道了也没有帮助

若 $X$ 与 $Y$ 独立：

$$
p(y\mid x)=p(y)
$$

代入定义：

$$
\begin{aligned}
H(Y\mid X)
&=-\sum_{x,y}p(x)p(y)\log p(y)\\
&=-\sum_yp(y)\log p(y)\sum_xp(x)\\
&=H(Y)
\end{aligned}
$$

例如 $X$ 是一枚公平硬币，$Y$ 是另一枚独立公平硬币。观察 $X$ 之后，$Y$ 仍然有 1 bit 的不确定性。

### 确定：知道了就没有剩余不确定性

若 $Y=f(X)$ 是 $X$ 的确定函数，对每个 $x$ 都只有一个 $y$ 的条件概率为 1：

$$
p(y\mid x)=
\begin{cases}
1,&y=f(x)\\
0,&y\neq f(x)
\end{cases}
$$

确定性分布的熵为 0，所以

$$
H(Y\mid X)=0
$$

例如 $Y=X$ 时，只要看过 $X$，就能准确写出 $Y$。这不表示 $Y$ 自身没有随机性；若 $X$ 是公平硬币，$H(Y)=1$ bit，只是这份随机性完全由 $X$ 携带。

### 加噪声：剩余不确定性由噪声决定

令 $X\sim\operatorname{Bernoulli}(1/2)$，$N\sim\operatorname{Bernoulli}(\varepsilon)$，两者独立，并令

$$
Y=X\mathbin{\oplus}N
$$

也就是以概率 $\varepsilon$ 翻转 $X$。给定 $X$ 后，$Y$ 只剩下「是否翻转」的不确定性：

$$
H_2(Y\mid X)=h_2(\varepsilon)
= -\varepsilon\log_2\varepsilon
  -(1-\varepsilon)\log_2(1-\varepsilon)
$$

当 $\varepsilon=0.1$ 时：

$$
H_2(Y\mid X)=h_2(0.1)=0.468995593589\ \mathrm{bits}
$$

当 $\varepsilon=0.5$ 时，翻转与否完全随机，$H_2(Y\mid X)=1$ bit；$X$ 对 $Y$ 不再提供预测帮助。因为输入公平且对称，$Y$ 始终也是公平的，故

$$
H_2(Y)=1,
\qquad
I(X;Y)=1-h_2(\varepsilon)
$$

这就是最简单的二元对称信道：条件熵是噪声留下的量，互信息是信道仍然传递的量。

## 条件熵是平均值，不是逐条件比较

容易把

$$
H(Y\mid X)\leq H(Y)
$$

误读成对每一个 $x$ 都有 $H(Y\mid X=x)\leq H(Y)$。这并不成立。构造一个反例：

$$
p(X=0)=0.9,
\qquad
p(X=1)=0.1
$$

给定 $X=0$ 时令 $Y=0$ 确定发生，给定 $X=1$ 时令 $Y$ 是公平硬币。于是

$$
H(Y\mid X=0)=0,
\qquad
H(Y\mid X=1)=1\ \mathrm{bit}
$$

第二个分支的熵高于边缘熵。边缘概率为

$$
p(Y=1)=0.1\times0.5=0.05,
\qquad
p(Y=0)=0.95
$$

所以

$$
H_2(Y)=h_2(0.05)=0.286396957116\ \mathrm{bits}
$$

但加权平均仍然很小：

$$
H_2(Y\mid X)=0.9\times0+0.1\times1=0.1\ \mathrm{bits}
$$

条件熵比较的是平均预测难度。一个很少出现的条件分支可以比总体更不确定，只要其他高概率分支足够确定，平均值仍然下降。

## 更多条件不会增加剩余不确定性

如果又观察到变量 $Z$，则

$$
H(Y\mid X,Z)\leq H(Y\mid X)
$$

因为知道 $X,Z$ 至少不比只知道 $X$ 少信息。差值可以写成条件互信息：

$$
H(Y\mid X)-H(Y\mid X,Z)
=I(Y;Z\mid X)
\geq0
$$

它的含义是：在已经知道 $X$ 的前提下，$Z$ 还额外告诉了多少关于 $Y$ 的信息。若 $Z$ 完全由 $X$ 决定，增加 $Z$ 不会改变条件熵；若 $Z$ 直接揭示 $Y$，条件熵可以降到 0。

这个结论的使用边界仍然是「平均」。某个具体的 $(x,z)$ 分支可以比 $(x)$ 分支更均匀，但所有分支按联合概率平均后不会增加。

## 条件熵与条件交叉熵

真实数据分布给出 $p(x,y)$，模型为每个输入 $x$ 输出条件分布 $q(y\mid x)$。模型的条件交叉熵是

$$
\mathcal H_p(Y\mid X;q)
=-\sum_{x,y}p(x,y)\log q(y\mid x)
$$

在对数里插入真实条件概率：

$$
\begin{aligned}
\mathcal H_p(Y\mid X;q)
&=-\sum_{x,y}p(x,y)\log p(y\mid x)\\
&\quad+\sum_xp(x)\sum_yp(y\mid x)
  \log\frac{p(y\mid x)}{q(y\mid x)}\\
&=H_p(Y\mid X)
 +\sum_xp(x)D_{\mathrm{KL}}
 \left(p(\cdot\mid x)\Vert q(\cdot\mid x)\right)
\end{aligned}
$$

因此

$$
\mathcal H_p(Y\mid X;q)\geq H_p(Y\mid X)
$$

最优模型是在每一个有数据质量的输入上都给出真实条件分布。若 $q(y\mid x)=0$ 而 $p(x,y)>0$，对应的条件交叉熵为无穷大；这与 [KL 散度](../information-theory/kl-divergence/) 的支持集问题相同。

### 继续使用上面的数字例子

真实条件分布是

$$
p(Y\mid X=0)=(0.9,0.1),
\qquad
p(Y\mid X=1)=(0.2,0.8)
$$

设模型输出

$$
q(Y\mid X=0)=(0.8,0.2),
\qquad
q(Y\mid X=1)=(0.3,0.7)
$$

两个分支的条件交叉熵分别为

$$
\begin{aligned}
\mathcal H_2(p(\cdot\mid0),q(\cdot\mid0))
&=-0.9\log_2 0.8-0.1\log_2 0.2\\
&\approx0.521928094887\ \mathrm{bits}
\end{aligned}
$$

$$
\begin{aligned}
\mathcal H_2(p(\cdot\mid1),q(\cdot\mid1))
&=-0.2\log_2 0.3-0.8\log_2 0.7\\
&\approx0.759051657097\ \mathrm{bits}
\end{aligned}
$$

按 $p(X=0)=p(X=1)=0.5$ 平均：

$$
\mathcal H_2(p;q)
=0.5\times0.521928094887
 +0.5\times0.759051657097
=0.640489875992\ \mathrm{bits}
$$

它比真实条件熵 $0.595461844238$ bit 多

$$
0.640489875992-0.595461844238
=0.045028031754\ \mathrm{bits}
$$

这正是两个条件 KL 散度按输入分布加权后的结果。分类器训练中的平均交叉熵，不能低于数据分布本身的条件熵；模型容量、数据量和优化方法只能决定能接近多少。

## 序列预测：每一步的条件熵相加

对序列 $X_1,\ldots,X_T$，链式法则写成

$$
H(X_1,\ldots,X_T)
=\sum_{t=1}^T H(X_t\mid X_{<t})
$$

其中

$$
X_{<t}=(X_1,\ldots,X_{t-1})
$$

这正是自回归模型的分解方式。模型在第 $t$ 步根据前缀 $x_{<t}$ 输出 $q_\theta(x_t\mid x_{<t})$，一条序列的负对数似然为

$$
\mathcal L(\theta)
=-\sum_{t=1}^T\log q_\theta(x_t\mid x_{<t})
$$

对数据分布取平均后：

$$
\mathbb E_p[\mathcal L(\theta)]
=T H_p(X_t\mid X_{<t})
 +\text{模型失配造成的额外条件 KL}
$$

这里在平稳或平均意义下写成了每一步相同的记号；一般情况应保留每个 $t$ 的条件熵。训练 next-token 模型时，真实前缀通常来自数据集，这就是 teacher forcing 的概率分解。词条 [困惑度](../information-theory/perplexity/) 会把平均每 token 的自然对数损失再取指数。

如果序列是独立同分布的，前缀没有预测作用：

$$
H(X_t\mid X_{<t})=H(X_t)
$$

如果序列存在结构，前缀会降低条件熵。条件熵因此提供了一个描述数据本身可预测性的下限，模型的 NLL 还包含没有学会的部分。

### 一个有限长度的马尔可夫例子

令 $X_1$ 是公平二元变量，每一步以概率 $0.1$ 翻转上一状态。由前面的二元对称信道计算：

$$
H_2(X_1)=1,
\qquad
H_2(X_t\mid X_{t-1})=h_2(0.1)=0.468995593589
$$

因此四个状态的联合熵为

$$
\begin{aligned}
H_2(X_1,X_2,X_3,X_4)
&=1+3\times0.468995593589\\
&=2.406986780768\ \mathrm{bits}
\end{aligned}
$$

平均到每个状态是 $0.601746695192$ bit；序列变长后，首个状态的固定开销被摊薄，熵率趋近于

$$
\lim_{T\to\infty}\frac1T H(X_1,\ldots,X_T)
=h_2(0.1)
=0.468995593589\ \mathrm{bits\ per\ symbol}
$$

这与只看单个边缘变量的 1 bit 不同：边缘分布公平，并不意味着整条序列没有可预测结构。

![条件熵把各个条件分支的熵按分支概率加权](/assets/information-theory/svg/conditional-entropy.1.svg)

## 连续变量的条件微分熵

连续变量使用密度而不是概率质量。若联合密度为 $f(x,y)$，条件密度为 $f(y\mid x)$，条件微分熵定义为

$$
h(Y\mid X)
=-\int\int f(x,y)\log f(y\mid x)\,\mathrm dy\,\mathrm dx
$$

同样有链式法则：

$$
h(X,Y)=h(X)+h(Y\mid X)
$$

但不能把离散熵的所有直觉照搬过来。微分熵依赖坐标单位，可以是负数，也不具有「知道更多信息一定逐点降低」这样的朴素解释。

例如 $N$ 在宽度为 $1/2$ 的区间上均匀分布，$Y=X+N$。给定 $X$ 后，$Y$ 只是平移了 $N$，所以

$$
h(Y\mid X)=h(N)=\ln\frac12=-0.693147180560\ \mathrm{nats}
$$

负值不是概率出了问题，而是连续密度的数值受到测量单位影响。若把长度单位缩放，密度和微分熵都会改变。[熵](../information-theory/entropy/) 一文中的微分熵部分与这里使用同一边界：连续变量的可比对象通常是相对熵、互信息或带有明确参考测度的量。

## 神经网络语境：不可约不确定性与可学部分

在分类问题里，输入 $X$ 是图像或特征，标签 $Y$ 是类别。真实数据分布的

$$
H(Y\mid X)
$$

表示给定输入后标签仍然存在的平均不确定性。相同输入可能对应多个合法标签、标注者可能意见不一致、传感器可能丢失信息，这些都会让条件熵非零。模型输出 $q_\theta(y\mid x)$ 的条件交叉熵则可以拆成

$$
\mathcal H_p(Y\mid X;q_\theta)
=H_p(Y\mid X)
 +\mathbb E_{X\sim p}
 D_{\mathrm{KL}}\left(p(\cdot\mid X)\Vert q_\theta(\cdot\mid X)\right)
$$

第一项是数据分布本身的平均难度，第二项才是模型没有复现真实条件分布的部分。实际训练还会叠加有限样本、优化未收敛、正则化和分布偏移等因素，因此训练集交叉熵不必等于真实条件熵。

在语言模型中，$X$ 可以是前缀，$Y$ 是下一个 token。一个上下文若几乎总接同一个 token，局部条件熵很低；若多个续写都合理，局部条件熵较高。整个数据集的平均 next-token NLL 是这些局部条件熵和模型失配的加权总和。换 tokenizer 会改变「每个 token」的单位和序列长度，所以比较 perplexity 时必须同时说明 tokenizer 和评估数据。

## 失效模式

**把条件熵和某一条件混为一谈。** $H(Y\mid X=x)$ 是固定一个 $x$ 的数，$H(Y\mid X)$ 还要对 $x$ 平均。两者的下标写法不同，含义也不同。

**忘记按 $p(x)$ 加权。** 不同条件分支出现频率不同，直接平均各分支熵只有在 $X$ 均匀分布时才正确。

**把平均不增误读成逐点不增。** $H(Y\mid X)\leq H(Y)$ 是平均结论，罕见分支的 $H(Y\mid X=x)$ 可以高于边缘熵。

**以为条件熵为零就是 $Y$ 没有随机性。** $H(Y\mid X)=0$ 只说明给定 $X$ 后 $Y$ 几乎处处是确定的；$X$ 自身仍可以随机，因而 $Y=f(X)$ 也可以有很高的边缘熵。

**把联合熵的顺序当成无关紧要。** $H(X,Y)$ 对称，但 $H(Y\mid X)$ 和 $H(X\mid Y)$ 一般不相等；做链式分解时必须说明先观察谁。

**把连续微分熵当成离散熵。** $h(Y\mid X)$ 可以为负，且依赖单位；讨论连续分布的可预测信息时，优先检查是否真正需要微分熵，还是需要 KL 散度或互信息。

## 相关词条

- [information-and-surprise](../information-theory/information-and-surprise/)：从单个结果的负对数概率开始。
- [entropy](../information-theory/entropy/)：定义无条件熵、联合熵、条件熵和微分熵。
- [cross-entropy](../information-theory/cross-entropy/)：把条件版本推广到分类与回归的模型损失。
- [kl-divergence](../information-theory/kl-divergence/)：解释条件交叉熵与真实条件熵之间的额外代价。
- [互信息](../information-theory/mutual-information/)：完整讨论条件化减少的不确定性。
- [困惑度](../information-theory/perplexity/)：把序列平均 NLL 转成指数刻度。
- [独立性](../probability/independence/)：条件分布不随已知变量改变时的边界。
