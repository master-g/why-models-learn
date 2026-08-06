---
title: "期望：用概率加权平均随机变量"
tags: ["why-models-learn"]
---

**期望**是随机变量在其概率分布下的加权平均。离散变量把每个取值乘以它出现的概率再求和，连续变量把取值乘以密度再积分：

$$
\mathbb E[X]=\sum_xx\,p_X(x)
$$

或

$$
\mathbb E[X]=\int_{-\infty}^{\infty}x f_X(x)\,dx
$$

期望不是「最可能的取值」，也不保证是一次实验会观测到的结果。它把整个分布压缩成一个平均位置，随后可以用来描述期望损失、总体错误率、回归目标和采样平均。本篇从离散和连续的加权平均出发，推导函数期望、线性性、指标变量、条件期望和全期望，再把这些公式接到机器学习的风险最小化。

## 离散期望：把每个取值乘上概率

若 X 只取有限或可数多个值，期望定义为

$$
\mathbb E[X]=\sum_xx\,P(X=x)
$$

求和必须覆盖 X 的全部支持集。概率总和为 1，所以这确实是一个加权平均，而不是把各个数值等权相加。

两次公平投掷的正面数 X 满足

| $x$ | 0 | 1 | 2 |
| --- | ---: | ---: | ---: |
| $P(X=x)$ | $1/4$ | $1/2$ | $1/4$ |

于是

$$
\begin{aligned}
\mathbb E[X]
&=0\cdot\frac14+1\cdot\frac12+2\cdot\frac14\\
&=1
\end{aligned}
$$

一次投掷结果不会是 1 次正面的「平均样本」；实际结果只能是 0、1 或 2。若重复进行许多轮两次投掷，再把每轮的正面数相加并除以轮数，平均值会在抽样波动中靠近 1。

公平六面骰的期望是

$$
\mathbb E[X]
=\sum_{k=1}^6 k\cdot\frac16
=\frac{1+2+3+4+5+6}{6}
=\frac{21}{6}
=\frac72
=3.5
$$

3.5 不是骰子能掷出的点数，但它是长期平均每次点数。期望落在 3 和 4 之间，不表示有一个隐藏的 3.5 面骰子。

若概率不均匀，期望会向高概率区域偏移。例如

$$
P(X=0)=\frac12,\qquad
P(X=2)=\frac14,\qquad
P(X=6)=\frac14
$$

则

$$
\mathbb E[X]
=0\cdot\frac12+2\cdot\frac14+6\cdot\frac14
=2
$$

取值 6 只占四分之一，但它离 0 很远，所以把平均位置向右拉开。计算期望时，取值大小和概率大小缺一不可。

## 连续期望：把加权求和换成积分

若 X 有密度 $f_X$，期望是

$$
\mathbb E[X]
=\int_{-\infty}^{\infty}x f_X(x)\,dx
$$

密度不是每个点的概率，积分才是在区间上累加概率。要让这个加权平均有意义，通常要求

$$
\int_{-\infty}^{\infty}|x|f_X(x)\,dx<\infty
$$

这样正负部分不会出现无法相减的无穷大。

对 $U\sim\operatorname{Uniform}[2,5]$，密度在区间长度 3 上恒为 $1/3$：

$$
\begin{aligned}
\mathbb E[U]
&=\int_2^5x\cdot\frac13\,dx\\
&=\frac13\left[\frac{x^2}{2}\right]_2^5\\
&=\frac13\cdot\frac{25-4}{2}\\
&=\frac72
\end{aligned}
$$

均匀分布的期望位于区间中点。前面密度为 $2x$ 的分布则更偏向右侧：

$$
\begin{aligned}
\mathbb E[X]
&=\int_0^1x\cdot2x\,dx\\
&=2\int_0^1x^2\,dx\\
&=\frac23
\end{aligned}
$$

虽然支持区间仍是 $[0,1]$，概率更密集的右侧使平均位置从 $1/2$ 移到 $2/3$。

离散求和和连续积分只是同一个加权思想的两种写法。离散情形把质量放在一个个点上，连续情形把质量铺成密度；不能因为密度曲线在某点的高度较高，就直接说该点概率较大。

## 函数期望：先变换取值，再按原分布平均

很多时候需要的不是 X 本身，而是 $g(X)$。函数期望的规则是

$$
\mathbb E[g(X)]
=\sum_xg(x)p_X(x)
$$

或连续形式

$$
\mathbb E[g(X)]
=\int g(x)f_X(x)\,dx
$$

这里不需要先求出新随机变量 $g(X)$ 的 PMF 或密度；直接在 X 的原分布上计算函数值即可。这条规则常被称为函数期望公式。

对公平六面骰，平方的期望为

$$
\begin{aligned}
\mathbb E[X^2]
&=\sum_{k=1}^6k^2\cdot\frac16\\
&=\frac{1+4+9+16+25+36}{6}\\
&=\frac{91}{6}
\end{aligned}
$$

而

$$
\bigl(\mathbb E[X]\bigr)^2
=\left(\frac72\right)^2
=\frac{49}{4}
$$

两者不相等。先平均再平方和先平方再平均，是两个不同的运算顺序。损失函数、能量函数和概率模型中的对数似然，都需要先对每个样本计算函数，再对函数值取期望。

指标变量是函数期望最直接的特例。对事件 A 定义

$$
\mathbf1_A=
\begin{cases}
1,&A\text{ 发生},\\
0,&A\text{ 不发生}.
\end{cases}
$$

因为它只取 0 和 1：

$$
\begin{aligned}
\mathbb E[\mathbf1_A]
&=1\cdot P(A)+0\cdot P(A^{\complement})\\
&=P(A)
\end{aligned}
$$

所以概率本身就是一个指标变量的期望。比如一次掷两枚骰子，令 A 表示点数和为 7，则

$$
\mathbb E[\mathbf1_A]=P(A)=\frac6{36}=\frac16
$$

## 期望的线性性：求和不要求独立

期望最常用的代数性质是线性性：

$$
\mathbb E[aX+bY+c]
=a\mathbb E[X]+b\mathbb E[Y]+c
$$

其中 a、b、c 是常数。对有限和：

$$
\mathbb E\left[\sum_{i=1}^nX_i\right]
=\sum_{i=1}^n\mathbb E[X_i]
$$

离散情况下可以直接展开：

$$
\begin{aligned}
\mathbb E[aX+bY+c]
&=\sum_{x,y}(ax+by+c)p_{X,Y}(x,y)\\
&=a\sum_{x,y}x p_{X,Y}(x,y)
 +b\sum_{x,y}y p_{X,Y}(x,y)
 +c\sum_{x,y}p_{X,Y}(x,y)\\
&=a\mathbb E[X]+b\mathbb E[Y]+c
\end{aligned}
$$

关键是每一项都可以逐项相加，不需要把联合 PMF 分解成边缘 PMF 的乘积。即使 X、Y 有依赖，线性性仍然成立；独立性只有在处理乘积或联合分布分解时才会额外出现。

令 $H_i$ 表示第 i 次投掷是否为正面：

$$
H_i=\mathbf1\{\text{第 i 次为正面}\}
$$

三次投掷中正面总数为

$$
N=H_1+H_2+H_3
$$

只要每次边缘概率都是 $1/2$，不管三次之间是否独立，都有

$$
\mathbb E[N]
=\mathbb E[H_1]+\mathbb E[H_2]+\mathbb E[H_3]
=\frac12+\frac12+\frac12
=\frac32
$$

如果三次确实独立，N 还服从二项分布；但得到它的期望并不需要先使用二项分布。这个区别在相关样本计数、分层抽样和批次损失中都很实用。

## 乘积期望需要额外的联合信息

线性性给出的是和的期望，不会自动给出乘积：

$$
\mathbb E[XY]
=\sum_{x,y}xy\,p_{X,Y}(x,y)
$$

若 X、Y 独立，联合 PMF 可以分解，于是

$$
\begin{aligned}
\mathbb E[XY]
&=\sum_{x,y}xy\,p_X(x)p_Y(y)\\
&=\left(\sum_xx\,p_X(x)\right)
 \left(\sum_yy\,p_Y(y)\right)\\
&=\mathbb E[X]\mathbb E[Y]
\end{aligned}
$$

没有独立性时，必须使用真正的联合分布。甚至某一次碰巧满足 $\mathbb E[XY]=\mathbb E[X]\mathbb E[Y]$，也不能据此推出独立；那只说明一个乘积平均恰好相等。

例如令 X 均匀取 $-1,0,1$，并令 $Y=X^2$。它们显然有确定的非线性关系，但

$$
\mathbb E[X]=0,\qquad
\mathbb E[Y]=\frac23,\qquad
\mathbb E[XY]=\mathbb E[X^3]=0
$$

于是这个特定的乘积满足

$$
\mathbb E[XY]=0=\mathbb E[X]\mathbb E[Y]
$$

却不能说明 X、Y 独立。期望只看选定函数的一个平均值，独立性要检查所有事件函数。

## 条件期望：给定信息后的局部平均

给定 $Y=y$ 后，X 的条件期望是

$$
\mathbb E[X\mid Y=y]
=\sum_xx\,p_{X\mid Y}(x\mid y)
$$

连续情形是

$$
\mathbb E[X\mid Y=y]
=\int x f_{X\mid Y}(x\mid y)\,dx
$$

它通常是 y 的函数，不是一个固定常数。不同的信息层可能有不同的局部平均。

考虑联合 PMF 中三个有正概率的格子：

| $(x,y)$ | $(0,0)$ | $(0,1)$ | $(1,1)$ |
| --- | ---: | ---: | ---: |
| $p_{X,Y}(x,y)$ | $1/5$ | $1/2$ | $3/10$ |

Y=1 的概率为

$$
P(Y=1)=\frac12+\frac3{10}=\frac45
$$

所以

$$
\mathbb E[X\mid Y=1]
=0\cdot\frac58+1\cdot\frac38
=\frac38
$$

Y=0 时，X 只能为 0：

$$
\mathbb E[X\mid Y=0]=0
$$

这两个条件平均按 Y 的边缘概率混合后，恢复总体期望：

$$
\begin{aligned}
\mathbb E[X]
&=\mathbb E[X\mid Y=1]P(Y=1)
 +\mathbb E[X\mid Y=0]P(Y=0)\\
&=\frac38\cdot\frac45+0\cdot\frac15\\
&=\frac3{10}
\end{aligned}
$$

## 全期望定律：先分层，再混合

上面的计算是全期望定律的离散版本：

$$
\mathbb E[X]
=\mathbb E\bigl[\mathbb E[X\mid Y]\bigr]
$$

展开对 Y 的求和：

$$
\mathbb E[X]
=\sum_y\mathbb E[X\mid Y=y]p_Y(y)
$$

连续情形把外层求和换成积分：

$$
\mathbb E[X]
=\int\mathbb E[X\mid Y=y]f_Y(y)\,dy
$$

它与全概率公式的结构完全平行。全概率公式在每个层内计算事件概率，再按层概率加权；全期望定律在每个层内计算平均值，再按层概率加权。

条件期望还满足塔式关系。若信息分成两层，先知道 Z，再知道更细的 Y，则

$$
\mathbb E\bigl[\mathbb E[X\mid Y,Z]\mid Z\bigr]
=\mathbb E[X\mid Z]
$$

外层平均会把更细的信息重新混合回较粗的信息。这个关系不是把条件符号当作普通代数括号随意删掉，而是重复使用「在条件分布下取平均」的定义。

## 期望风险与回归中的条件均值

机器学习常把单个样本的损失写成

$$
\ell_\theta(X,Y)
$$

总体风险是联合数据分布下的期望：

$$
R(\theta)=\mathbb E_{(X,Y)}[\ell_\theta(X,Y)]
$$

训练集只能计算经验风险：

$$
\widehat R(\theta)
=\frac1n\sum_{i=1}^n\ell_\theta(x_i,y_i)
$$

在样本能够代表总体时，经验平均才会作为总体期望的近似。损失函数本身先作用于每个样本，不能先把输入和标签各自平均后再计算损失。

平方损失还解释了回归为什么指向条件均值。固定输入 x 和一个常数预测 a，考虑

$$
R_x(a)=\mathbb E[(Y-a)^2\mid X=x]
$$

令

$$
\mu_x=\mathbb E[Y\mid X=x]
$$

把 $Y-a=(Y-\mu_x)+(\mu_x-a)$ 展开：

$$
\begin{aligned}
R_x(a)
&=\mathbb E[(Y-\mu_x)^2\mid X=x]\\
&\quad+2(\mu_x-a)\mathbb E[Y-\mu_x\mid X=x]\\
&\quad+(\mu_x-a)^2
\end{aligned}
$$

中间项为 0，因为

$$
\mathbb E[Y-\mu_x\mid X=x]
=\mathbb E[Y\mid X=x]-\mu_x
=0
$$

因此

$$
R_x(a)
=\mathbb E[(Y-\mu_x)^2\mid X=x]+(\mu_x-a)^2
$$

第一项与 a 无关，第二项在 $a=\mu_x$ 时最小。也就是说，在平方损失下，给定输入后的最佳常数预测是

$$
a^*(x)=\mathbb E[Y\mid X=x]
$$

回归网络学习的不是每个输入对应的一个必然标签，而是在所选损失下逼近条件分布的某个统计量。换成绝对值损失，最优预测通常转向条件中位数；损失函数改变，期望要描述的目标也会改变。

![概率加权平均与平方损失下的条件均值](/assets/probability/svg/expectation.1.svg)

## 期望的存在与无穷平均

有限支持、或满足

$$
\mathbb E[|X|]<\infty
$$

的随机变量有一个有限的普通期望。对离散变量，这个条件是

$$
\sum_x|x|p_X(x)<\infty
$$

对连续变量则是绝对值积分有限。若 X 非负而这个和或积分发散，期望可以是 $+\infty$，不能把它当作普通实数继续做减法。

一个例子是对 $k=1,2,\ldots$ 定义

$$
P(X=k)=\frac6{\pi^2k^2}
$$

因为

$$
\sum_{k=1}^{\infty}\frac1{k^2}=\frac{\pi^2}{6}
$$

概率总和为 1，但

$$
\mathbb E[X]
=\sum_{k=1}^{\infty}k\cdot\frac6{\pi^2k^2}
=\frac6{\pi^2}\sum_{k=1}^{\infty}\frac1k
=+\infty
$$

尾部概率每项都很小，不代表取值乘上大小后仍然可加。实际建模中，若样本的极端值很常见，均值可能非常不稳定；此时需要检查尾部、截断规则和所使用的损失。

## 采样平均是期望的数值入口

从分布独立抽取 $x_1,\ldots,x_n$ 后，函数 g 的样本平均是

$$
\widehat{\mathbb E}[g(X)]
=\frac1n\sum_{i=1}^ng(x_i)
$$

它用有限计算近似

$$
\mathbb E[g(X)]
$$

这就是蒙特卡洛估计的基本形式。随着样本数增加，若抽样机制满足相应条件，样本平均会趋向总体期望；大数定律会在后面专门说明这种收敛需要什么假设。

神经网络训练中的 mini-batch 平均损失也是这个结构：用当前批次的经验分布估计总体数据分布下的期望风险。批次太小会有较大波动，批次来源有偏则平均值会稳定地偏向错误的分布。

## 失效模式

**把期望当成最可能值。**骰子期望是 3.5，众数可以是某一个概率最大的点；两个概念由不同的计算定义。

**把期望当成必然观测值。**期望是长期加权平均，不要求落在随机变量的离散支持集上。

**忘记乘概率或密度。**离散变量要用 $x p_X(x)$ 求和，连续变量要用 $x f_X(x)$ 积分；只把支持集的数值相加不是期望。

**把 $\mathbb E[g(X)]$ 写成 $g(\mathbb E[X])$。**非线性变换通常不能和期望交换。平方骰子的数值已经给出一个直接反例。

**以为求和的期望需要独立。**期望的线性性不需要独立；独立性主要用于乘积期望和联合分布分解。

**把一个乘积平均相等当成独立。**$\mathbb E[XY]=\mathbb E[X]\mathbb E[Y]$ 只检查一个函数，不能代替所有事件组合的判据。

**混淆条件期望和总体期望。**$\mathbb E[X\mid Y=y]$ 是某一个信息层内的平均，要再按 $Y$ 的概率混合才能得到 $\mathbb E[X]$。

**忽略期望不存在。**重尾分布可能只有无穷期望，或正负部分不能合法相减。先检查 $\mathbb E[|X|]$，再使用线性运算。

**把经验风险当成总体风险。**训练集平均值是有限样本估计，受抽样、分组和数据泄漏影响，不会自动等于总体期望。

## 相关词条

- [概率空间](../probability/probability-spaces/)：提供期望所依赖的概率测度。
- [随机变量](../probability/random-variables/)：定义随机取值、指标变量和变量变换。
- [离散分布](../probability/discrete-distributions/)：提供 PMF 求和与常见分布的具体例子。
- [连续分布](../probability/continuous-distributions/)：提供密度积分和连续期望的计算。
- [联合分布](../probability/joint-distributions/)：计算多个变量函数的联合期望。
- [边缘与条件分布](../probability/marginal-and-conditional/)：定义条件期望和按层重新加权。
- [独立性](../probability/independence/)：说明何时乘积期望可以分解。
- [方差与协方差](../probability/variance-and-covariance/)：从期望构造离散程度和共同变化。
- [大数定律](../probability/law-of-large-numbers/)：解释样本平均何时逼近总体期望。
