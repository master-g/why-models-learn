---
title: "单个神经元的反向传播：从误差信号到参数梯度"
tags: ["why-models-learn"]
---

单个神经元的反向传播，是把一条最小的学习计算图从损失逐边求导：神经元先把输入的加权和变成 logit，再经过激活得到输出，损失函数把输出和目标比较；反向阶段从损失出发，把一个误差信号传回预激活值，再分别乘上输入得到权重梯度、把它复制给偏置梯度。本文用一个二元分类神经元逐项计算这些量，再把公式推广到任意激活、向量输入和 batch，最后用有限差分检查每个坐标。

![单个神经元的前向值与反向梯度：输入和权重先汇入 logit，误差信号再沿各条边分配](/assets/backpropagation/svg/backprop-single-neuron.1.svg)

## 单个神经元就是一张可手算的图

设输入有 $d$ 个特征，权重向量为 $\boldsymbol w\in\mathbb R^d$，偏置为 $b\in\mathbb R$。神经元的仿射部分先计算

$$
z=\boldsymbol w^{\mathsf T}\boldsymbol x+b
=\sum_{j=1}^{d}w_jx_j+b.
$$

$z$ 叫预激活值或 logit。再用激活函数 $\phi$ 产生输出

$$
a=\phi(z).
$$

最后由目标 $y$ 和输出 $a$ 定义单样本损失

$$
L=\ell(a,y).
$$

这三行已经把计算图拆成了几个局部节点：

$$
\boldsymbol x,\boldsymbol w,b
\longrightarrow
z
\longrightarrow
a
\longrightarrow
L.
$$

反向传播要回答的不是“哪个参数更重要”这种模糊问题，而是三个具体的偏导：

$$
\nabla_{\boldsymbol w}L,
\qquad
\frac{\partial L}{\partial b},
\qquad
\nabla_{\boldsymbol x}L.
$$

前两个用于更新模型参数；最后一个通常用于解释输入敏感度、构造输入扰动或把梯度继续传给上一层。若这个神经元是网络中的中间节点，损失对 $a$ 的导数来自后继层，而不是由它自己决定。

## 先把二元分类损失写在 logit 上

二元分类中常用 sigmoid 激活

$$
\sigma(z)=\frac{1}{1+\exp(-z)},
\qquad
\sigma'(z)=\sigma(z)\bigl(1-\sigma(z)\bigr).
$$

它把任意实数 $z$ 映射到 $(0,1)$，所以可以把

$$
a=\sigma(z)
$$

解释成正类概率。目标 $y$ 取 $0$ 或 $1$ 时，二元交叉熵是

$$
L
=-y\log a-(1-y)\log(1-a).
$$

把 $a=\sigma(z)$ 代入并整理，可以得到更适合在 logit 上计算的形式

$$
L
=\log\bigl(1+\exp(z)\bigr)-yz.
$$

直接计算 $\exp(z)$ 在 $z$ 很大时可能溢出，所以实现通常使用稳定的 softplus 或带 logits 的 BCE。这里先用这个等价式推导导数，再在数值例子里保留足够精度。

对概率 $a$ 求导：

$$
\frac{\partial L}{\partial a}
=-\frac{y}{a}+\frac{1-y}{1-a}.
$$

对 sigmoid 使用链式法则：

$$
\begin{aligned}
\frac{\partial L}{\partial z}
&=\frac{\partial L}{\partial a}\frac{\partial a}{\partial z}\\
&=\left(-\frac{y}{a}+\frac{1-y}{1-a}\right)a(1-a)\\
&=-y(1-a)+(1-y)a\\
&=a-y.
\end{aligned}
$$

这个消去因子的结果很重要：sigmoid 与 BCE 配对后，传给仿射节点的误差信号只需要计算预测概率减去目标，而不必在实现中分别乘一遍 $\sigma'(z)$。这里的简化不是“sigmoid 没有导数”，而是交叉熵对概率的导数恰好与 sigmoid 导数相消。

## 一个二元分类神经元的完整数值例子

取二维输入

$$
\boldsymbol x=
\begin{bmatrix}
1\\
-2
\end{bmatrix},
\qquad
\boldsymbol w=
\begin{bmatrix}
0.4\\
-0.3
\end{bmatrix},
\qquad
b=0.1,
\qquad
y=1.
$$

目标是让这个样本输出正类概率，因此 $y=1$。

### 前向阶段

先算 logit：

$$
\begin{aligned}
z
&=0.4\times1+(-0.3)\times(-2)+0.1\\
&=1.1.
\end{aligned}
$$

再算概率：

$$
a=\sigma(1.1)
\approx0.750260105595.
$$

因为目标为 $1$，稳定的 BCE 为

$$
L
=\log\bigl(1+\exp(1.1)\bigr)-1.1
\approx0.287335325115.
$$

预测概率低于目标，所以损失为正，且接下来传回的 logit 误差信号应当为负数。

### 先得到一个误差信号

记损失对 logit 的导数为

$$
\delta
:=\frac{\partial L}{\partial z}.
$$

sigmoid 与 BCE 的组合给出

$$
\delta=a-y
\approx0.750260105595-1
=-0.249739894405.
$$

$\delta<0$ 的含义很具体：若只把 $z$ 增大一小点，损失会下降；若把 $z$ 减小，损失会增加。它还不是某个权重的梯度，因为每条参数边的局部导数不同。

### 从 logit 传回参数和输入

仿射节点的局部导数是

$$
\frac{\partial z}{\partial w_j}=x_j,
\qquad
\frac{\partial z}{\partial x_j}=w_j,
\qquad
\frac{\partial z}{\partial b}=1.
$$

因此对每个权重坐标：

$$
\frac{\partial L}{\partial w_j}
=\frac{\partial L}{\partial z}
\frac{\partial z}{\partial w_j}
=\delta x_j.
$$

代入本例：

$$
\nabla_{\boldsymbol w}L
=\delta\boldsymbol x
=
\begin{bmatrix}
-0.249739894405\\
0.499479788810
\end{bmatrix}.
$$

偏置边的局部导数是 $1$，所以

$$
\frac{\partial L}{\partial b}
=\delta
\approx-0.249739894405.
$$

输入梯度则是

$$
\nabla_{\boldsymbol x}L
=\delta\boldsymbol w
=
\begin{bmatrix}
-0.099895957762\\
0.074921968321
\end{bmatrix}.
$$

最后一个向量可以用来检查直觉：增加 $x_1$ 会沿 $w_1=0.4$ 的方向改变 logit，当前 $\delta<0$，所以 $x_1$ 的梯度为负；增加 $x_2$ 时，$w_2=-0.3$ 把方向反过来，所以 $x_2$ 的梯度为正。

### 用一行总公式回看

对于 sigmoid 加 BCE，单个神经元的所有参数梯度可以压成

$$
\boxed{
\delta=a-y,
\qquad
\nabla_{\boldsymbol w}L=\delta\boldsymbol x,
\qquad
\frac{\partial L}{\partial b}=\delta,
\qquad
\nabla_{\boldsymbol x}L=\delta\boldsymbol w
}.
$$

这四个量的形状也必须对得上：

$$
\delta\in\mathbb R,
\qquad
\boldsymbol x,\boldsymbol w,\nabla_{\boldsymbol w}L,\nabla_{\boldsymbol x}L\in\mathbb R^d,
\qquad
\frac{\partial L}{\partial b}\in\mathbb R.
$$

若代码返回了一个形状为 $d\times d$ 的权重梯度，或者把偏置梯度广播成了一个 batch 向量，首先检查的不是数值，而是这条形状约束。

## 为什么误差信号要先在 logit 处汇总

把每条边分别写出来：

$$
\frac{\partial L}{\partial w_j}
=
\underbrace{\frac{\partial L}{\partial a}}_{\text{损失边}}
\underbrace{\frac{\partial a}{\partial z}}_{\text{激活边}}
\underbrace{\frac{\partial z}{\partial w_j}}_{\text{仿射边}}.
$$

先乘前两项就是 $\delta=\partial L/\partial z$，再乘最后一项才得到权重坐标的梯度。这样安排有两个好处。

第一，所有传给同一个仿射节点的上游贡献都可以先相加。若神经元输出 $a$ 被两个损失项同时使用，应该先得到

$$
\frac{\partial L}{\partial a}
=\frac{\partial L_1}{\partial a}
+\frac{\partial L_2}{\partial a},
$$

再乘激活导数和仿射导数。覆盖而不是累加会丢掉一条路径。

第二，$\delta$ 与输入维度无关。无论是一个特征还是一百万个特征，sigmoid 和损失只需要计算一次 $\delta$；之后用外积或逐坐标乘法把这个标量分配给各条输入边。

如果输出头外面还有一个上游标量系数 $g=\partial L/\partial a$，通用写法是

$$
\delta
=g\,\phi'(z),
\qquad
\nabla_{\boldsymbol w}L=\delta\boldsymbol x,
\qquad
\frac{\partial L}{\partial b}=\delta.
$$

sigmoid 加 BCE 只是把前两项具体化为 $g=\partial L/\partial a$ 后的简化结果 $\delta=a-y$。

## 换激活或损失时，哪一项会改变

仿射节点的反向规则不变。变化只发生在从损失和输出回到 $z$ 的局部链上：

$$
\delta
=\frac{\partial L}{\partial z}
=\frac{\partial L}{\partial a}\phi'(z).
$$

| 输出与损失 | logit 误差信号 $\delta$ | 需要留意的条件 |
| --- | --- | --- |
| $a=z,\ L=\frac12(a-y)^2$ | $a-y$ | 恒等激活是无界回归头 |
| $a=\sigma(z),\ L=\operatorname{BCE}(a,y)$ | $a-y$ | BCE 与 sigmoid 配对后发生因子消去 |
| $a=\tanh(z),\ L=\frac12(a-y)^2$ | $(a-y)(1-a^2)$ | $\lvert a\rvert$ 接近 $1$ 时导数变小 |
| $a=\operatorname{ReLU}(z),\ L=\frac12(a-y)^2$ | $(a-y)1_{z>0}$ | $z<0$ 时梯度被活动掩码置零 |

例如使用 tanh 时，先算

$$
\frac{\partial L}{\partial a}=a-y,
\qquad
\frac{\partial a}{\partial z}=1-a^2,
$$

再相乘得到

$$
\delta=(a-y)(1-a^2).
$$

若在 tanh 输出上错误套用 $a-y$，就漏掉了激活边；若在 sigmoid+BCE 上又手动多乘一次 $a(1-a)$，则把同一条边重复计算，梯度会被缩小。

## 线性代数形式和梯度的形状

把输入和权重写成列向量：

$$
z=\boldsymbol w^{\mathsf T}\boldsymbol x+b.
$$

对 $\boldsymbol w$ 求梯度时，$z$ 对权重的 Jacobian 是 $\boldsymbol x^{\mathsf T}$；用列向量梯度记号写成

$$
\nabla_{\boldsymbol w}z=\boldsymbol x.
$$

于是

$$
\nabla_{\boldsymbol w}L
=\frac{\partial L}{\partial z}\nabla_{\boldsymbol w}z
=\delta\boldsymbol x.
$$

同理

$$
\nabla_{\boldsymbol x}L=\delta\boldsymbol w.
$$

如果把权重写成行向量，公式的转置方向会变化，但数值对象不变。实践中应该先固定一个布局，再检查矩阵乘法的形状。对单个神经元来说：

| 对象 | 形状 | 前向或反向作用 |
| --- | --- | --- |
| $\boldsymbol x$ | $d\times1$ | 输入特征 |
| $\boldsymbol w$ | $d\times1$ | 每个特征的权重 |
| $z$、$a$、$L$、$\delta$ | 标量 | 逐层传递的节点值或敏感度 |
| $\nabla_{\boldsymbol w}L$ | $d\times1$ | 更新权重 |
| $\nabla_{\boldsymbol x}L$ | $d\times1$ | 传给上一层或做敏感度分析 |

偏置看似没有输入特征，但它等价于在输入末尾追加常数 $1$：

$$
\widetilde{\boldsymbol x}
=
\begin{bmatrix}
\boldsymbol x\\
1
\end{bmatrix},
\qquad
\widetilde{\boldsymbol w}
=
\begin{bmatrix}
\boldsymbol w\\
b
\end{bmatrix},
\qquad
z=\widetilde{\boldsymbol w}^{\mathsf T}\widetilde{\boldsymbol x}.
$$

所以偏置梯度就是 $\delta$，并不是某个特殊的经验规则；它只是最后一条输入边的输入值等于 $1$。

## 从单个样本推广到 batch

现在有 $n$ 个样本，每行是一个输入：

$$
X\in\mathbb R^{n\times d},
\qquad
\boldsymbol w\in\mathbb R^d,
\qquad
\boldsymbol y\in\mathbb R^n.
$$

一个共享参数的单神经元对整个 batch 的前向计算是

$$
\boldsymbol z=X\boldsymbol w+b\boldsymbol1,
\qquad
\boldsymbol a=\sigma(\boldsymbol z).
$$

令逐样本 logit 误差信号为

$$
\boldsymbol\delta=\boldsymbol a-\boldsymbol y.
$$

若训练目标取样本损失的平均值

$$
L_{\mathrm{mean}}
=\frac1n\sum_{i=1}^{n}\ell_i,
$$

那么反向结果为

$$
\nabla_{\boldsymbol w}L_{\mathrm{mean}}
=\frac1n X^{\mathsf T}\boldsymbol\delta,
\qquad
\frac{\partial L_{\mathrm{mean}}}{\partial b}
=\frac1n\boldsymbol1^{\mathsf T}\boldsymbol\delta.
$$

输入矩阵的梯度按行是

$$
\nabla_XL_{\mathrm{mean}}
=\frac1n\boldsymbol\delta\boldsymbol w^{\mathsf T}.
$$

这里的外积形状是 $n\times1$ 乘 $1\times d$，结果正好是 $n\times d$。若目标是 sum reduction，三个公式都去掉 $1/n$；不能只对权重梯度除以 $n$ 而忘记偏置梯度。

### 一个两样本 batch

取

$$
X=
\begin{bmatrix}
1&-2\\
0.5&1
\end{bmatrix},
\qquad
\boldsymbol y=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\boldsymbol w=
\begin{bmatrix}
0.4\\
-0.3
\end{bmatrix},
\qquad
b=0.1.
$$

两个 logit 和概率分别为

$$
\boldsymbol z=
\begin{bmatrix}
1.1\\
0
\end{bmatrix},
\qquad
\boldsymbol a\approx
\begin{bmatrix}
0.750260105595\\
0.5
\end{bmatrix}.
$$

逐样本损失为

$$
\boldsymbol\ell
\approx
\begin{bmatrix}
0.287335325115\\
0.693147180560
\end{bmatrix},
\qquad
L_{\mathrm{mean}}\approx0.490241252838.
$$

误差信号为

$$
\boldsymbol\delta
\approx
\begin{bmatrix}
-0.249739894405\\
0.5
\end{bmatrix}.
$$

带入矩阵公式：

$$
\nabla_{\boldsymbol w}L_{\mathrm{mean}}
\approx
\begin{bmatrix}
0.000130052798\\
0.499739894405
\end{bmatrix},
\qquad
\frac{\partial L_{\mathrm{mean}}}{\partial b}
\approx0.125130052797.
$$

第一维权重梯度很小，不是因为两个样本都没有误差，而是第一列的两条贡献 $-0.249739894405$ 与 $0.5\times0.5=0.25$ 几乎抵消。把 batch 公式误写成“先平均 $\boldsymbol\delta$ 再逐元素乘平均输入”会失去这种样本与特征的配对关系；正确做法是先计算 $X^{\mathsf T}\boldsymbol\delta$。

### 广播和归约的边界

$b$ 在前向中从一个标量广播成 $n$ 个 logit：

$$
\boldsymbol z=X\boldsymbol w+b\boldsymbol1.
$$

反向时，广播的逆操作是求和：

$$
\frac{\partial L}{\partial b}
=\sum_{i=1}^{n}\frac{\partial L}{\partial z_i}
\quad\text{或}\quad
\frac1n\sum_{i=1}^{n}\frac{\partial L}{\partial z_i},
$$

具体取决于损失是 sum 还是 mean。把偏置梯度保留成长度为 $n$ 的向量，会让优化器收到错误的参数形状；只取第一行则会丢掉其余样本的贡献。

## 梯度下降只在反向之后发生

对本例取学习率 $\eta=0.1$，梯度下降更新是

$$
\boldsymbol w_{\mathrm{new}}
=\boldsymbol w-\eta\nabla_{\boldsymbol w}L,
\qquad
b_{\mathrm{new}}
=b-\eta\frac{\partial L}{\partial b}.
$$

代入数值：

$$
\boldsymbol w_{\mathrm{new}}
\approx
\begin{bmatrix}
0.424973989440\\
-0.349947978881
\end{bmatrix},
\qquad
b_{\mathrm{new}}\approx0.124973989440.
$$

新的 logit、概率和损失为

$$
z_{\mathrm{new}}\approx1.249843936643,
\qquad
a_{\mathrm{new}}\approx0.777272844691,
\qquad
L_{\mathrm{new}}\approx0.251963838785.
$$

损失下降是这个小步长在这个样本上的结果，不是反向传播的定义。若把更新写成 $\boldsymbol w+\eta\nabla_{\boldsymbol w}L$，这次会沿增大损失的方向走；若梯度正确但 $\eta$ 过大，仍可能越过低损失区域。诊断时要把“求导错”“符号错”“学习率不合适”分开记录。

## 前向阶段要缓存什么

对单个样本，反向时至少需要知道：

| 前向量或标量 | 为什么反向需要它 |
| --- | --- |
| $\boldsymbol x$ | 计算 $\nabla_{\boldsymbol w}L=\delta\boldsymbol x$ |
| $\boldsymbol w$ | 计算 $\nabla_{\boldsymbol x}L=\delta\boldsymbol w$ |
| $z$ | 计算 $\phi'(z)$，或在 sigmoid+BCE 中用于稳定损失 |
| $a$ | 计算 $\partial L/\partial a$，以及本例的 $\delta=a-y$ |
| $y$ | 计算损失和误差信号 |

参数在反向阶段不能先被更新再用来计算另一条梯度。正确的时序是：

$$
\text{读取同一组旧参数}
\longrightarrow
\text{前向并缓存}
\longrightarrow
\text{反向得到全部梯度}
\longrightarrow
\text{统一更新参数}.
$$

若先更新了 $w_1$，再用新 $w_1$ 计算 $x_1$ 或 $w_2$ 的梯度，后面的量就不再是同一张前向图的导数。深层网络把这个问题放大，因此通常把梯度计算和优化器的 step 调用明确分成两个阶段。

## 用有限差分检查解析梯度

解析梯度可以逐坐标和中心差分比较。对任意标量参数 $\theta$：

$$
\frac{\partial L}{\partial\theta}
\approx
\frac{L(\theta+\varepsilon)-L(\theta-\varepsilon)}{2\varepsilon}.
$$

本例取 $\varepsilon=10^{-5}$，只改变一个坐标，并重新执行完整前向。解析值和中心差分如下：

| 被扰动坐标 | 解析梯度 | 中心差分 |
| --- | ---: | ---: |
| $w_1$ | $-0.249739894405$ | $-0.249739894398$ |
| $w_2$ | $0.499479788810$ | $0.499479788818$ |
| $b$ | $-0.249739894405$ | $-0.249739894398$ |
| $x_1$ | $-0.099895957762$ | $-0.099895957772$ |
| $x_2$ | $0.074921968321$ | $0.074921968318$ |

这些差异约在 $10^{-12}$ 到 $10^{-11}$ 量级，符合双精度中心差分和小步长的舍入误差。检查时必须每次从同一份原始参数复制出 $\theta+\varepsilon$ 与 $\theta-\varepsilon$；如果第一次扰动没有还原，第二个坐标的差分就不是单坐标导数。

有限差分只能核对一个固定输入、固定标签和固定随机状态的局部导数。它不能证明 batch reduction、随机 dropout、控制流分支或参数更新协议都正确；这些边界需要单独构造测试。

## 失效模式：先查哪一条边

**把概率当成 logit。** sigmoid 的输出 $a$ 在 $(0,1)$，logit $z$ 可以是任意实数。BCE with logits 接收的是 $z$，如果已经传入概率又在内部套 sigmoid，数值和梯度都会错。

**漏掉激活导数。** 对一般损失，必须先计算 $\partial L/\partial a$，再乘 $\phi'(z)$。只有在 sigmoid 与 BCE 的特定配对下才可以化简为 $a-y$。

**把 batch 的 mean 当成 sum。** mean reduction 在每个样本的 $\delta_i$ 上额外乘 $1/n$。漏掉它会让 batch 越大，梯度越大；重复除一次则会让大 batch 的更新过小。

**偏置没有对广播求和。** 一个偏置被所有样本共享，反向时要把所有样本对它的贡献相加。形状为 $n$ 的临时向量不是最终的偏置梯度。

**覆盖而不是累加。** 如果输出同时进入多个损失项，或者同一个参数在图中被多次使用，梯度槽位必须累加。单个神经元的 $\boldsymbol w^{\mathsf T}\boldsymbol x$ 已经有多个乘法边，不能把它们当作互相独立的参数副本。

**在零点忽略激活约定。** ReLU 在 $z=0$ 不可导，框架会选一个次梯度约定。梯度检查要使用不落在零点的输入，或明确复现该约定。

**反向前修改缓存。** 先做更新、原地改写 $z$ 或清掉 $a$，会让后续局部导数来自另一组值。保留一次前向的缓存，等所有参数梯度计算完再更新。

**用差分步长掩盖错误。** $\varepsilon$ 太大有截断误差，太小有舍入误差；应在几个数量级上观察误差曲线，而不是只看一次“碰巧接近”的结果。

## 这个例子和更大网络的边界

单个神经元已经包含多层网络反向传播的三个核心动作：

1. 从标量损失得到上游敏感度；
2. 乘以当前节点的局部导数，得到传给父节点的误差信号；
3. 对每条参数边应用自己的局部导数，并在共享参数处累加。

多层网络只是把一个仿射节点的输出送入下一个节点。[两层网络的反向传播](../backpropagation/backprop-two-layer/) 会把一个 $\delta$ 变成每层的 $\boldsymbol\delta^{(l)}$，并展示隐藏层梯度为什么既依赖上游权重又依赖当前激活导数。[向量化反向传播](../backpropagation/vectorized-backprop/) 则把这里的逐样本乘法和累加整理成矩阵乘法。

单个神经元仍然保留两个在大模型中容易被藏起来的边界：激活和损失决定 $\delta$，输入与参数决定仿射节点的梯度形状。只要这两条边分开，看到一个错误数值时就能定位是损失、激活、广播、归约还是参数更新出了问题。

## 相关词条

- [计算图](../backpropagation/computational-graphs/)：把本条的神经元拆成节点、边和依赖顺序。
- [计算图上的链式法则](../backpropagation/chain-rule-on-graphs/)：解释局部导数相乘、共享路径相加的通用规则。
- [前向计算](../backpropagation/forward-pass/)：区分前向值、缓存、训练态和反向阶段。
- [反向传播](../backpropagation/backpropagation/)：把单个神经元的规则推广到任意计算图。
- [两层网络的反向传播](../backpropagation/backprop-two-layer/)：继续推导隐藏层的误差信号。
- [向量化反向传播](../backpropagation/vectorized-backprop/)：从单样本公式过渡到批量矩阵运算。
- [梯度检查](../backpropagation/gradient-checking/)：系统化比较解析梯度与数值微分。
- [sigmoid 函数](../neurons-and-activations/heaviside-and-sigmoid/)：展开 sigmoid 的范围、导数和饱和。
- [交叉熵](../information-theory/cross-entropy/)：从似然角度解释 BCE 与 logit 形式。
