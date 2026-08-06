---
title: "经典感知机：错误驱动地寻找线性分界"
tags: ["why-models-learn"]
---

经典感知机用一个带偏置的线性分数和硬阈值做二分类，遇到分错或落在边界上的样本就沿着纠错方向更新参数。它在线性可分数据上能在有限次错误后找到一个分界面；在线性不可分数据上则可能持续循环。感知机不是概率模型，也不把分数当作置信度，理解它的关键是把更新方向、几何边界和可分性放在同一个坐标里看。

![经典感知机在二维平面上移动线性分界并分开两类点](/assets/linear-models/svg/perceptron-classic.1.svg)

## 从加权和到硬分类

给定输入 $x\in\mathbb R^p$，感知机先计算

$$
s=w^{\mathsf T}x+b.
$$

把偏置并入特征，可以写成增广向量

$$
\widetilde x=
\begin{pmatrix}
1\\x
\end{pmatrix},
\qquad
\theta=
\begin{pmatrix}
b\\w
\end{pmatrix},
\qquad
s=\theta^{\mathsf T}\widetilde x.
$$

采用标签 $y\in\{-1,+1\}$。硬阈值输出为

$$
\widehat y=
\begin{cases}
+1,&s>0,\\
-1,&s<0,\\
0,&s=0.
\end{cases}
$$

这里的 $0$ 不是第三个类别，而是说明样本落在边界上，当前分数还不能支持任一标签。实现时也可以规定平局归入某一侧；对学习规则而言，边界样本通常按错误处理。

分界面由

$$
w^{\mathsf T}x+b=0
$$

给出。在二维中它是一条直线，在三维中是一个平面，在更高维中是超平面。分数的符号只看样本位于超平面的哪一侧，分数的绝对值暂时没有概率含义。

| 对象 | 数学形式 | 几何解释 | 感知机中的作用 |
| --- | --- | --- | --- |
| 分数 | $s=w^{\mathsf T}x+b$ | 点到有向边界的未归一化位置 | 决定阈值前的方向 |
| 边界 | $s=0$ | 两类之间的超平面 | 划分输入空间 |
| 预测 | $\operatorname{sign}(s)$ | 选择边界一侧 | 输出硬标签 |
| 参数 | $(w,b)$ | 边界的方向和平移 | 由错误样本更新 |

如果把 $w$ 和 $b$ 同时乘以正数，预测不变；这说明感知机参数也存在尺度冗余。只有经过归一化后，分数才可以和几何距离联系起来。

## 用带符号的 margin 判断错误

对标签 $y\in\{-1,+1\}$，定义带符号 margin

$$
m=y\,s=y\left(w^{\mathsf T}x+b\right).
$$

若 $m>0$，预测方向与标签一致；若 $m<0$，样本被分错；若 $m=0$，样本落在边界上。于是可以把更新条件统一写成

$$
m\le0.
$$

当样本出错时，感知机更新规则为

$$
\theta\leftarrow\theta+\eta y\widetilde x,
\qquad
\eta>0.
$$

这条式子同时覆盖两类：

- 若 $y=+1$，就加上 $\eta\widetilde x$，把分数往正方向推；
- 若 $y=-1$，就减去 $\eta\widetilde x$，把分数往负方向推。

更新后，同一个样本的带符号 margin 变成

$$
\begin{aligned}
m_{\mathrm{new}}
&=y\left(\theta+\eta y\widetilde x\right)^{\mathsf T}\widetilde x\\
&=y\theta^{\mathsf T}\widetilde x+\eta y^2\lVert\widetilde x\rVert_2^2\\
&=m+\eta\lVert\widetilde x\rVert_2^2.
\end{aligned}
$$

所以每次更新都会提高当前错误样本的 margin，增加量由输入向量的长度和学习率决定。这只描述当前样本，不保证其他样本的 margin 同时提高：一个边界移动后，原来正确的点可能被推到错误一侧。

感知机可以用一个分段的错误准则记录训练状态：

$$
\ell_{\mathrm P}(\theta;x,y)=
\begin{cases}
-y\theta^{\mathsf T}\widetilde x,&y\theta^{\mathsf T}\widetilde x\le0,\\
0,&y\theta^{\mathsf T}\widetilde x>0.
\end{cases}
$$

它和 logistic 交叉熵不同。已经正确的样本不会继续因为 margin 太小而产生损失；感知机只关心是否在边界正确一侧。若需要显式要求 margin 至少为 1，就进入 hinge loss 和支持向量机的设定，而不是经典感知机的原始规则。

## 一个两维更新例子

取四个已经线性可分的点：

$$
\begin{array}{c|c|c}
\text{样本}&x&y\\
A&(2,0)&+1\\
B&(0,2)&-1\\
C&(1,-1)&+1\\
D&(-1,1)&-1
\end{array}
$$

它们可以由边界 $x_1-x_2=0$ 分开。用增广向量、初始参数和学习率

$$
\widetilde x=(1,x_1,x_2)^{\mathsf T},
\qquad
\theta_0=(0,0,0)^{\mathsf T},
\qquad
\eta=1
$$

按 $A,B,C,D$ 的顺序扫描。第一步遇到 $A$ 时，分数为 0，按平局更新：

$$
\theta_1
=\theta_0+(+1)(1,2,0)^{\mathsf T}
=(1,2,0)^{\mathsf T}.
$$

接着看完整一轮：

| 样本 | 当前参数下的分数 $s$ | margin $m=ys$ | 动作 | 更新后参数 |
| --- | ---: | ---: | --- | --- |
| $A$ | $0$ | $0$ | 更新 | $(1,2,0)$ |
| $B$ | $1$ | $-1$ | 更新 | $(0,2,-2)$ |
| $C$ | $4$ | $4$ | 保持 | $(0,2,-2)$ |
| $D$ | $-4$ | $4$ | 保持 | $(0,2,-2)$ |

例如 $B$ 的增广向量是 $(1,0,2)^{\mathsf T}$，标签是 $-1$。在 $A$ 更新后的参数 $(1,2,0)^{\mathsf T}$ 下，分数为 1，负类样本的 margin 为 $-1$，所以第二次更新为

$$
(1,2,0)^{\mathsf T}-(1,0,2)^{\mathsf T}
=(0,2,-2)^{\mathsf T}.
$$

此后四个样本的 margin 分别为 $4,-4,4,-4$，乘上各自标签以后都是 4。第二轮扫描没有错误，算法可以停止。最终边界是

$$
0+2x_1-2x_2=0
\quad\Longleftrightarrow\quad
x_1-x_2=0.
$$

这个例子里感知机找到了一个解，但解并不唯一。任何正倍数的 $(0,2,-2)$ 都给出同一条边界，其他也能把四个点分开的直线同样可能被算法找到。

## 线性可分是什么意思

数据集线性可分，是指存在某个参数 $\theta^\star$，使每个样本都位于正确一侧：

$$
y_i{\theta^\star}^{\mathsf T}\widetilde x_i>0,
\qquad
i=1,\ldots,n.
$$

如果有限数据集满足这个条件，那么有限个正 margin 中存在最小值

$$
\gamma=\min_i y_i{\theta^\star}^{\mathsf T}\widetilde x_i>0.
$$

为了写出收敛界，再令

$$
R=\max_i\lVert\widetilde x_i\rVert_2.
$$

$\gamma$ 是当前参数尺度下的最小带符号 margin，$R$ 是样本增广向量的最大长度。真正与几何间隔相对应的是 $\gamma/\lVert\theta^\star\rVert_2$，因为把 $\theta^\star$ 放大只会同时放大 margin 和参数范数。

## 感知机收敛定理的证明

从 $\theta_0=0$ 开始，只在 $y_t\theta_t^{\mathsf T}\widetilde x_t\le0$ 时更新。假设前 $T$ 次更新所用样本分别为 $\widetilde x_1,\ldots,\widetilde x_T$。先看参数在可分解方向 $\theta^\star$ 上的投影：

$$
\begin{aligned}
\theta_{t+1}^{\mathsf T}\theta^\star
&=\theta_t^{\mathsf T}\theta^\star
+\eta y_t\widetilde x_t^{\mathsf T}\theta^\star\\
&\ge\theta_t^{\mathsf T}\theta^\star+\eta\gamma.
\end{aligned}
$$

归纳得到

$$
\theta_T^{\mathsf T}\theta^\star\ge T\eta\gamma.
$$

再看参数长度。因为更新发生在当前样本 margin 不正时，

$$
\begin{aligned}
\lVert\theta_{t+1}\rVert_2^2
&=\lVert\theta_t+\eta y_t\widetilde x_t\rVert_2^2\\
&=\lVert\theta_t\rVert_2^2
+2\eta y_t\theta_t^{\mathsf T}\widetilde x_t
+\eta^2\lVert\widetilde x_t\rVert_2^2\\
&\le\lVert\theta_t\rVert_2^2+\eta^2R^2.
\end{aligned}
$$

因此

$$
\lVert\theta_T\rVert_2\le\eta R\sqrt T.
$$

另一方面，Cauchy–Schwarz 不等式给出

$$
\theta_T^{\mathsf T}\theta^\star
\le\lVert\theta_T\rVert_2\lVert\theta^\star\rVert_2
\le\eta R\sqrt T\,\lVert\theta^\star\rVert_2.
$$

合并上下两个界，并消去正的 $\eta\sqrt T$：

$$
T\le
\left(
\frac{R\lVert\theta^\star\rVert_2}{\gamma}
\right)^2.
$$

这就是经典的感知机收敛界：线性可分时，错误更新次数有限。它不是“经过固定轮数一定得到某个唯一参数”的保证，也不是所有样本都只访问一次的保证。它只约束犯错次数，扫描顺序、样本尺度和数据 margin 都会影响实际运行轨迹。

对前面的四个点，可以取 $\theta^\star=(0,1,-1)^{\mathsf T}$。每个样本的带符号 margin 都是 2，$R=\sqrt5$，$\lVert\theta^\star\rVert_2=\sqrt2$，所以理论上界为

$$
\left(\frac{\sqrt5\sqrt2}{2}\right)^2=\frac52.
$$

错误次数是整数，因此这个界允许至多 2 次错误，和表格中的实际更新次数相符。

## 为什么不可分时会循环

如果不存在参数能把全部样本放在正确一侧，感知机就没有“零错误停止”的目标。一个最小反例是 XOR：

$$
\begin{array}{c|c}
x&y\\
(0,0)&-1\\
(0,1)&+1\\
(1,0)&+1\\
(1,1)&-1
\end{array}
$$

假设存在 $s=b+w_1x_1+w_2x_2$ 能正确分类。前三个约束先给出

$$
\begin{aligned}
(0,0),-1&:\quad b\le0,\\
(1,0),+1&:\quad b+w_1>0
\quad\Longrightarrow\quad w_1>-b,\\
(0,1),+1&:\quad b+w_2>0
\quad\Longrightarrow\quad w_2>-b.
\end{aligned}
$$

但第四个负类点要求

$$
b+w_1+w_2\le0.
$$

另一方面，由 $w_1>-b$ 和 $w_2>-b$ 得到

$$
b+w_1+w_2>b-2b=-b\ge0,
$$

这和第四个约束冲突。因此不存在这样的直线，偏置也不能解决 XOR 的不可分性。

在 XOR 上，感知机更新会反复修正不同的错误点。它可能在某个扫描顺序下出现参数循环，也可能在有限精度、停止阈值或数据顺序变化下呈现不同的轨迹；这些都不是“再多训练几轮就会找到解”的问题。要处理 XOR，需要增加非线性特征、使用隐藏层，或改用能表达不同决策边界的模型。

## 分数不是概率

感知机的输出只有硬标签，分数 $s$ 的大小不能直接读成概率。参数 $(w,b)$ 乘以正数后，分界和所有硬预测都不变，但分数会按同一倍数变化：

$$
\operatorname{sign}(c s)=\operatorname{sign}(s),
\qquad
c>0.
$$

因此“分数 2 比分数 1 更有 2 倍把握”没有感知机模型内的定义。它与 logistic 回归的区别也正在这里：logistic 回归把线性分数送入 sigmoid，并用 Bernoulli 似然训练概率；感知机只根据符号和错误事件更新。

感知机也没有把“正确但很靠近边界”的样本继续推远的机制。只要 $m>0$，经典更新就跳过它。若希望所有样本至少留出一个安全间隔，可以使用

$$
\ell_{\mathrm{hinge}}(s,y)=\max\bigl(0,1-ys\bigr),
$$

并通过最大间隔方法选择参数。这个目标与感知机共享线性边界，但训练偏好不同：感知机只惩罚错误侧，hinge loss 还惩罚 margin 小于 1 的正确样本。

## 特征扩展改变的是边界空间

感知机在原始特征上只能学习一个仿射超平面。如果先构造固定特征映射 $\phi(x)$，再使用

$$
s=\theta^{\mathsf T}
\begin{pmatrix}
1\\\phi(x)
\end{pmatrix},
$$

那么它在原始输入空间里对应的边界可能是曲线或其他非线性形状。模型仍然只对扩展后的特征做线性组合，非线性来自特征映射本身。

例如二维输入增加一个乘积特征 $x_1x_2$，分数可以写成

$$
s=b+w_1x_1+w_2x_2+w_{12}x_1x_2.
$$

在 $(x_1,x_2)$ 平面中，$s=0$ 通常不再是一条直线。这个思路可以解决某些简单不可分数据，但手工选择特征会遇到维度增长和泛化风险；神经网络通过学习中间表示，把特征变换也纳入训练。

## 与其他分类模型对照

| 模型 | 输出 | 训练信号 | 对线性不可分数据 | 概率解释 |
| --- | --- | --- | --- | --- |
| 经典感知机 | 硬标签、线性分数 | 分错样本的错误方向 | 原始特征上不能找到零错误边界 | 没有 |
| 逻辑回归 | 二分类概率 | Bernoulli 负对数似然 | 仍是线性边界，但可用概率损失 | 有 |
| Softmax 回归 | 多分类概率 | 分类分布交叉熵 | 仍是线性类别边界 | 有 |
| 线性 SVM | 硬标签或 margin 分数 | hinge loss 与最大间隔 | 仍受线性表达能力限制 | 没有直接概率 |
| 多层网络 | 任意结构的输出 | 取决于输出层损失 | 通过表示和非线性扩展边界 | 取决于输出层与校准 |

这里“线性”指对当前输入表示和参数的线性分数，不等于模型永远只能处理线性关系。固定的非线性特征、核映射和神经网络隐藏层都可以把问题送到更适合的表示空间；感知机本身的收敛定理只适用于给定表示下的线性可分性。

## 一个可复用的训练流程

1. 把标签统一编码为 $-1$ 和 $+1$，明确边界平局如何处理；
2. 把偏置并入增广特征，检查特征尺度和异常值；
3. 初始化 $\theta$，按数据顺序计算分数和带符号 margin；
4. 对 $m\le0$ 的样本执行 $\theta\leftarrow\theta+\eta y\widetilde x$；
5. 每轮记录错误次数、最小 margin、参数范数和数据顺序；
6. 线性可分时在连续若干轮零错误后停止，并保留验证集检查；
7. 多轮仍有错误时先检查不可分性、标签冲突和特征表示，不要只增加轮数；
8. 若需要概率、间隔、非线性边界或多分类，换用与目标匹配的模型。

数据顺序不是无关紧要的实现细节：在线更新会让早先样本影响后续边界。可以打乱顺序、使用小批量或保存多个随机种子的结果，但这些改变只影响运行轨迹，不会把不可分数据变成可分数据。

## 失效模式

**把感知机分数当成概率。** 线性分数的正负只决定硬分类，正倍数缩放就会改变数值而不改变预测；需要概率时应使用概率模型和校准流程。

**把收敛定理读成所有数据都会收敛。** 定理的前提是存在正的线性 margin；XOR、标签冲突和噪声点会破坏这个前提。

**只看训练错误，不检查更新是否循环。** 不可分数据上错误数可能在一个区间内反复波动；记录参数、样本顺序和重复状态才能识别循环。

**把边界样本当成已经正确。** $s=0$ 没有支持任一侧，必须明确平局政策；按错误更新是常见的安全选择。

**忽略特征尺度。** 大尺度特征会让一次更新的某些坐标变化远大于其他坐标，改变扫描轨迹和数值稳定性；标准化和记录尺度都很重要。

**把增加训练轮数当成解决表达能力不足。** 如果原始表示不可分，迭代只会继续改写边界；应增加特征、引入非线性或更换模型。

**把经典感知机和多层感知机混为一谈。** 经典感知机是单层硬阈值线性分类器，多层感知机可以包含隐藏层和可微激活，表达能力与训练问题都不同。

## 相关词条

- [逻辑回归](../linear-models/logistic-regression/)：比较 sigmoid 概率、Bernoulli 似然和感知机硬阈值。
- [Softmax 回归](../linear-models/softmax-regression/)：把线性分数扩展为竞争式多分类概率。
- [为什么需要非线性](../neurons-and-activations/why-non-linearity/)：用 XOR 说明隐藏表示如何突破线性可分限制。
- [间隔与支持向量机](../linear-models/margins-and-svm/)：展开 hinge loss、最大间隔和支持向量。
- [线性回归](../linear-models/linear-regression/)：对照连续标签下的最小二乘目标。
- [岭回归与 Lasso](../linear-models/ridge-and-lasso/)：比较显式正则化改变线性模型解偏好的方式。
- [分类指标](../evaluation-and-generalization/classification-metrics/)：检查硬标签、排序和概率评估的差异。
- [仿射空间与映射](../linear-algebra/affine-spaces-and-maps/)：理解偏置带来的平移和仿射边界。
