---
title: "分类损失：从标签结构到训练信号"
tags: ["why-models-learn"]
---

分类损失把离散标签、模型输出和错误代价接成一条可求导的训练路径。二分类的 Bernoulli 负对数似然、互斥多分类的 softmax 交叉熵、多标签任务的独立 BCE、focal loss 和 hinge loss 看起来都是“分类损失”，但它们对标签关系、置信度、类别不平衡和概率解释的假设不同。先确定标签结构，再选择输出头与损失；最后还要把训练损失、概率质量和部署阈值分开核验。

![分类损失既取决于标签关系，也取决于置信度：左侧比较 BCE 与 focal 对真实类别概率的惩罚，右侧把二分类、互斥多分类和多标签的输出契约并列](/assets/training-nn/svg/classification-losses.1.svg)

## 先看标签关系，再看损失

分类问题的第一问不是“用哪一个 API”，而是一个样本能同时拥有几个标签，以及标签是否有顺序。

| 标签结构 | 一个样本的真实结果 | 常见输出 | 首选损失 |
| --- | --- | --- | --- |
| 二分类 | 一个 $y\in\{0,1\}$ | 一个 logit $z$，经过 sigmoid 得到正类概率 | BCE with logits |
| 互斥多分类 | 恰好一个 $y\in\{1,\ldots,K\}$ | $K$ 个竞争的 logits，经过 softmax 得到联合分布 | softmax 交叉熵 |
| 多标签分类 | $\boldsymbol y\in\{0,1\}^{K}$，每个标签可独立出现 | $K$ 个独立 logits，逐标签 sigmoid | 独立 BCE with logits |
| 有序分类 | 类别之间有明确的等级 | 累积概率、阈值或有序 logit 结构 | 与等级结构匹配的有序损失 |

如果一张图片可以同时是“室内”“夜间”和“有人”，这不是把三个答案塞进一个 softmax 的多分类；如果一个样本只能属于“猫、狗、鸟”中的一个，三个独立 sigmoid 又没有表达互斥约束。损失函数不能替模型修复错误的标签语义。

还要区分四个层次：

| 层次 | 输入与输出 | 它回答的问题 |
| --- | --- | --- |
| 逐样本损失 | 一个标签与一个输出 | 这次错误的概率代价是多少 |
| 训练目标 | 损失、权重、mask、正则项的归约 | 参数本轮沿什么方向移动 |
| 评估指标 | 冻结模型的概率或硬预测 | 模型在一批样本上表现怎样 |
| 部署决策 | 概率、阈值与错误代价 | 这次是否采取某个行动 |

交叉熵可以在正确类别概率从 $0.51$ 增加到 $0.99$ 时继续下降，而 accuracy 可能完全不变。相反，改变阈值可以提高召回率，却不会改变已经训练好的交叉熵。若把这四层写成同一个数字，训练日志就很难解释。

## 二元交叉熵：Bernoulli 似然与 logit 梯度

设标签 $y\in\{0,1\}$，模型最后输出一个实数 logit $z$。sigmoid 把它变成正类概率：

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Bernoulli 分布给出的似然是

$$
P(y\mid p)=p^y(1-p)^{1-y}.
$$

取负对数得到二元交叉熵：

$$
\ell_{\mathrm{BCE}}(p,y)
=-y\log p-(1-y)\log(1-p).
$$

当 $y=1$ 时，损失是 $-\log p$；当 $y=0$ 时，损失是 $-\log(1-p)$。正确类别的概率越接近 $1$，损失越接近 $0$；把真实标签的概率压到接近 $0$，损失会迅速增大。

### 为什么 logits 形式更适合训练

把 $p=\sigma(z)$ 代回 BCE，并利用

$$
\log\sigma(z)=z-\log(1+e^z),
\qquad
\log(1-\sigma(z))=-\log(1+e^z),
$$

可以得到

$$
\begin{aligned}
\ell_{\mathrm{BCE}}(z,y)
&=-y\bigl(z-\log(1+e^z)\bigr)
-(1-y)\bigl(-\log(1+e^z)\bigr)\\
&=\log(1+e^z)-yz.
\end{aligned}
$$

记 softplus 为

$$
\operatorname{softplus}(z)=\log(1+e^z).
$$

直接计算 $e^z$ 在 $z$ 很大时会溢出，直接先算概率又可能把 $p$ 舍入成 $0$ 或 $1$。稳定的 softplus 写成

$$
\operatorname{softplus}(z)
=\max(z,0)+\log\bigl(1+e^{-\lvert z\rvert}\bigr),
$$

于是

$$
\ell_{\mathrm{BCE}}(z,y)
=\max(z,0)-yz
+\log\bigl(1+e^{-\lvert z\rvert}\bigr).
$$

这个式子在正负极端的 logit 上都避免了不必要的指数爆炸。实际接口若要求 logits，就应直接传入 $z$，不要在模型里先 sigmoid 再把概率当作 logit。

### 梯度为什么是预测减标签

对 logit 求导：

$$
\frac{\partial\ell}{\partial z}
=\frac{e^z}{1+e^z}-y
=\sigma(z)-y
=p-y.
$$

二阶导数为

$$
\frac{\partial^2\ell}{\partial z^2}
=p(1-p)\geq0.
$$

当 $y=1$ 而 $p$ 太小，梯度 $p-y$ 为负，梯度下降会把 $z$ 往上推；当 $y=0$ 而 $p$ 太大，梯度为正，会把 $z$ 往下推。这个方向同时说明 BCE 在 logit 上是凸的，但深度网络对参数通常仍是非凸的，因为 logit 是参数的非线性函数。

| 标签 $y$ | logit $z$ | 概率 $p$ | BCE | $\partial\ell/\partial z$ |
| ---: | ---: | ---: | ---: | ---: |
| 1 | $-2$ | 0.119203 | 2.126928 | -0.880797 |
| 1 | 0 | 0.500000 | 0.693147 | -0.500000 |
| 1 | 2 | 0.880797 | 0.126928 | -0.119203 |
| 0 | $-2$ | 0.119203 | 0.126928 | 0.119203 |
| 0 | 0 | 0.500000 | 0.693147 | 0.500000 |
| 0 | 2 | 0.880797 | 2.126928 | 0.880797 |

正确类别已经很有把握时，BCE 仍会给一个小但非零的梯度，继续要求概率更接近 $0$ 或 $1$。这也是它与只关心 margin 是否越过阈值的 hinge loss 的重要区别。

## 互斥多分类：softmax 交叉熵

现在每个样本只有一个真实类别 $c\in\{1,\ldots,K\}$，模型输出

$$
\boldsymbol z=(z_1,\ldots,z_K)\in\mathbb R^K.
$$

softmax 把相互竞争的分数变成一个联合概率：

$$
p_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}},
\qquad
\sum_{k=1}^{K}p_k=1.
$$

若使用 one-hot 标签 $\boldsymbol y$，多分类交叉熵是

$$
\ell_{\mathrm{CE}}
=-\sum_{k=1}^{K}y_k\log p_k.
$$

因为真实类别 $c$ 处 $y_c=1$，它也可以写成

$$
\ell_{\mathrm{CE}}=-\log p_c.
$$

将 softmax 代回损失，定义 log-sum-exp

$$
\operatorname{LSE}(\boldsymbol z)=\log\sum_{j=1}^{K}e^{z_j},
$$

便得到 logits 形式：

$$
\ell_{\mathrm{CE}}(\boldsymbol z,c)
=\operatorname{LSE}(\boldsymbol z)-z_c.
$$

### 梯度同时包含竞争关系

先对 log-sum-exp 求导：

$$
\frac{\partial\operatorname{LSE}(\boldsymbol z)}
{\partial z_k}
=\frac{e^{z_k}}{\sum_j e^{z_j}}
=p_k.
$$

所以

$$
\frac{\partial\ell_{\mathrm{CE}}}{\partial z_k}=p_k-y_k.
$$

这和 BCE 的形式相似，但这里的 $\boldsymbol p$ 必须和为 $1$，而且每个类别的梯度通过同一个归一化分母相互耦合。提高一个错误类别的 logit 会增加它自己的概率，也会相对压低其他类别的概率。

二阶导数是一个矩阵：

$$
\nabla_{\boldsymbol z}^{2}\ell
=\operatorname{diag}(\boldsymbol p)
-\boldsymbol p\boldsymbol p^{\mathsf T}.
$$

它的对角项是 $p_k(1-p_k)$，非对角项是 $-p_i p_j$。把 softmax 当作逐分量函数、只保留对角导数，会漏掉类别竞争。

### 稳定的 log-sum-exp

令

$$
m=\max_j z_j.
$$

因为

$$
\sum_j e^{z_j}=e^m\sum_j e^{z_j-m},
$$

所以

$$
\operatorname{LSE}(\boldsymbol z)
=m+\log\sum_j e^{z_j-m}.
$$

所有 $z_j-m\leq0$，指数项最大为 $1$。这不改变 softmax 的概率，因为给所有 logits 减去同一个常数不会改变概率比。它只是把数值计算搬到更安全的范围。

例如 $\boldsymbol z=(2,1,0)$ 时，真实类别取中间一类：

$$
\boldsymbol p\approx(0.665241,\ 0.244728,\ 0.090031),
$$

损失为

$$
-\log(0.244728)\approx1.407606.
$$

梯度约为

$$
(0.665241,\ -0.755272,\ 0.090031).
$$

正确类别的梯度为负，两个错误类别的梯度为正；每个 logit 都参与了一次更新。

## 软标签与 label smoothing

one-hot 标签不是分类目标的唯一形式。若目标是一个概率分布

$$
\boldsymbol q=(q_1,\ldots,q_K),
\qquad
q_k\geq0,
\qquad
\sum_kq_k=1,
$$

交叉熵写成

$$
\ell(\boldsymbol z,\boldsymbol q)
=-\sum_{k=1}^{K}q_k\log p_k.
$$

对 logits 求导仍然是

$$
\frac{\partial\ell}{\partial z_k}=p_k-q_k.
$$

把它拆开：

$$
\begin{aligned}
\ell(\boldsymbol q,\boldsymbol p)
&=-\sum_kq_k\log p_k\\
&=-\sum_kq_k\log q_k
+\sum_kq_k\log\frac{q_k}{p_k}\\
&=H(\boldsymbol q)+D_{\mathrm{KL}}(\boldsymbol q\Vert\boldsymbol p).
\end{aligned}
$$

固定目标 $\boldsymbol q$ 时，最优预测是 $\boldsymbol p=\boldsymbol q$。因此软标签不是把 one-hot“稍微模糊一下但最优答案不变”；它真的改变了训练目标所要求的分布。

label smoothing 的一个常见定义是

$$
q_k=(1-\varepsilon)\mathbf 1[k=c]+\frac{\varepsilon}{K},
\qquad
0\leq\varepsilon\leq1.
$$

当 $K=3$、$\varepsilon=0.1$ 且真实类别为第 $2$ 类时：

$$
\boldsymbol q=(0.033333,\ 0.933333,\ 0.033333).
$$

它减弱了把真实类推到概率 $1$、把其他类推到概率 $0$ 的压力，常常能改善过度自信，但也可能降低对硬标签分布的拟合。报告模型时要记录 $\varepsilon$，不能把 label smoothing 隐藏在数据管线或损失封装里。

mixup、知识蒸馏和人工置信度标签也会产生软目标。它们共享 $p-q$ 的梯度形式，却不共享同一个数据解释：label smoothing 是固定的目标平滑，蒸馏目标来自教师分布，mixup 目标来自样本标签的线性组合。分析校准时必须保留目标的来源。

## 多标签：每个标签独立的 BCE

多标签样本可以同时属于多个类别。此时模型输出

$$
\boldsymbol z=(z_1,\ldots,z_K),
\qquad
p_k=\sigma(z_k),
$$

每个标签都有自己的 Bernoulli 似然。逐标签损失的和为

$$
\ell_{\mathrm{ML}}
=\sum_{k=1}^{K}
\left[
-y_k\log p_k
-(1-y_k)\log(1-p_k)
\right].
$$

也可以对标签轴取平均：

$$
\ell_{\mathrm{ML,mean}}=\frac1K\ell_{\mathrm{ML}}.
$$

选择 sum 还是 mean 会改变梯度尺度，尤其是标签数量 $K$ 在不同实验间变化时。训练协议必须写清归约轴和分母。

多标签 BCE 没有 softmax 的竞争约束。三个标签都可以有较高概率，概率总和也不必为 $1$。这正是它适合“同时存在”的原因，而不是一个需要事后修复的数值问题。

| 任务 | 输出约束 | 一个类别变强时 | 常见错误 |
| --- | --- | --- | --- |
| 互斥多分类 | $\sum_kp_k=1$ | 会挤压其他类别 | 对每个类别独立 sigmoid |
| 多标签分类 | 每个 $p_k$ 独立落在 $(0,1)$ | 不必改变其他类别 | 把概率强行 softmax |
| 多任务二分类 | 每个头有自己的标签和损失 | 由共享骨干间接耦合 | 把不同任务拼成一个类别轴 |

如果业务真的有“至少一个标签、可能多个标签”的层级约束，独立 BCE 只是第一层建模；还需要另行设计结构约束或后处理，并在验证集上检查它是否损害召回和概率解释。

## 类别权重改变了什么

类别不平衡时，常见做法是给少数类更大的损失权重。二分类的加权 BCE 可写成

$$
\ell_{\alpha,\beta}(p,y)
=-\alpha y\log p
-\beta(1-y)\log(1-p),
$$

其中 $\alpha$ 是正类权重，$\beta$ 是负类权重。对 logit 求导：

$$
\frac{\partial\ell_{\alpha,\beta}}{\partial z}
=\bigl(\alpha y+\beta(1-y)\bigr)p-\alpha y.
$$

权重不只是让某些样本“多算几遍”。它会改变条件风险的最优概率。令

$$
\eta(x)=\mathbb P(Y=1\mid X=x),
$$

则固定输入时的期望损失为

$$
\mathcal R(p\mid x)
=-\alpha\eta\log p
-\beta(1-\eta)\log(1-p).
$$

令导数为零：

$$
-\frac{\alpha\eta}{p}
+\frac{\beta(1-\eta)}{1-p}=0.
$$

整理得到

$$
p^\star_{\mathrm{weighted}}
=\frac{\alpha\eta}
{\alpha\eta+\beta(1-\eta)}.
$$

当 $\alpha\neq\beta$ 时，最优输出不再是原始数据分布中的 $\eta$。例如 $\alpha=4,\beta=1$ 时，原始概率 $\eta=0.2$ 会被映射为

$$
\frac{4\times0.2}{4\times0.2+1\times0.8}=0.5.
$$

如果仍用加权输出的 $0.5$ 作阈值，它等价于在原始概率上使用

$$
\eta\geq\frac{\beta}{\alpha+\beta}=0.2
$$

的行动规则。这样做可能正是业务想要的代价敏感决策，但加权后的分数不能不加说明地当成原始部署分布中的校准概率。

多分类也有同样现象。设真实条件分布为 $\boldsymbol\pi$，类别权重为 $w_k$，加权交叉熵的条件风险是

$$
\mathcal R(\boldsymbol q)
=-\sum_{k=1}^{K}w_k\pi_k\log q_k,
\qquad
\sum_kq_k=1.
$$

用拉格朗日乘子求最小值，得到

$$
q_k^\star=\frac{w_k\pi_k}{\sum_jw_j\pi_j}.
$$

因此 class weight、过采样和后续阈值调整都要在“训练目标”与“概率解释”之间画一条明确的线：

| 做法 | 直接改变的对象 | 需要额外核验 |
| --- | --- | --- |
| sample weight | 单个样本的梯度贡献 | 权重归一化、有效样本数 |
| class weight | 某类样本的总体贡献与最优分布 | 校准、阈值和每类召回 |
| 过采样 | 训练看到的经验分布 | 采样概率、重复样本和先验修正 |
| 阈值调整 | 最后一步行动规则 | 不改变训练概率，需在验证集选择 |

逆频率权重、有效样本数权重等都是策略，不是从数据自动推出的唯一正确似然。少数类指标变好不代表总体概率仍然可直接解释。

## focal loss：把学习预算给难样本

交叉熵会持续奖励容易样本的高置信度预测。在极度不平衡的检测任务中，海量容易负样本可能压过少量困难正样本。focal loss 在交叉熵前面乘一个随正确类别概率增大而衰减的因子：

$$
p_t=
\begin{cases}
p,&y=1,\\
1-p,&y=0,
\end{cases}
$$

$$
\ell_{\mathrm{focal}}
=-\alpha_t(1-p_t)^\gamma\log p_t,
\qquad
\gamma\geq0.
$$

当 $\gamma=0$ 且 $\alpha_t=1$ 时，它退化为 BCE。若 $p_t=0.9$、$\gamma=2$，调制因子是 $(1-0.9)^2=0.01$；若 $p_t=0.1$，调制因子是 $0.81$。困难样本相对保留了更多梯度。

为了看清它不是一个神奇的“更强交叉熵”，先对 $p_t$ 求导：

$$
\frac{\partial\ell_{\mathrm{focal}}}{\partial p_t}
=\alpha_t
\left[
\gamma(1-p_t)^{\gamma-1}\log p_t
-\frac{(1-p_t)^\gamma}{p_t}
\right].
$$

再乘上 $p_t$ 对 logit 的导数：

$$
\frac{\partial p_t}{\partial z}
=(2y-1)p(1-p).
$$

这说明 focal 的梯度不再只是 $p-y$ 乘一个固定类别权重；它还包含 $\gamma$、当前难度和 $\log p_t$。实现时应使用稳定的 log-probability 或框架的专用实现，避免在极端概率上先做不安全的幂和对数。

focal loss 的代价是概率解释变复杂。它明确地改变了不同难度样本的相对权重，通常不再是未经修改数据分布下的 proper scoring rule。用它提升 rare-class AP 或 recall 后，仍要单独检查 log loss、Brier、可靠性图和阈值稳定性。

| 真实类别概率 $p_t$ | BCE 中的 $-\log p_t$ | $\gamma=2$ 的调制因子 | focal loss，$\alpha_t=1$ |
| ---: | ---: | ---: | ---: |
| 0.9 | 0.105361 | 0.01 | 0.001054 |
| 0.6 | 0.510826 | 0.16 | 0.081732 |
| 0.1 | 2.302585 | 0.81 | 1.865094 |

若困难样本本身包含标签噪声，focal 会主动放大它们的训练预算，可能把模型推向噪声。使用前要检查困难样本是否真的代表少数类或决策边界，而不是脏标签、遮挡或错误标注。

## margin 损失：hinge 与 squared hinge

概率损失并不是唯一的分类训练方式。对二分类标签采用

$$
y\in\{-1,+1\},
$$

令模型输出带符号分数 $z$，带符号 margin 为

$$
m=yz.
$$

$m>0$ 表示分数在正确一侧，$m$ 越大表示离边界越远。hinge loss 要求 margin 至少达到 $1$：

$$
\ell_{\mathrm{hinge}}(z,y)=\max(0,1-yz).
$$

当 $m\geq1$ 时损失为零；当 $m<1$ 时，损失线性惩罚 margin 不足。对 $m=1$ 的折点使用次梯度：

$$
\frac{\partial\ell_{\mathrm{hinge}}}{\partial z}
=
\begin{cases}
-y,&yz<1,\\
0,&yz>1.
\end{cases}
$$

折点处可以取区间 $[-y,0]$ 中的任意次梯度。平方 hinge 则是

$$
\ell_{\mathrm{sq-hinge}}=\max(0,1-yz)^2,
$$

在 margin 不足时给更大的连续惩罚，但仍然在 margin 超过 $1$ 后完全停止奖励。

与 logistic 损失比较：

| 损失 | 需要的标签 | 大 margin 时 | 是否直接给概率 |
| --- | --- | --- | --- |
| BCE / logistic | $0,1$ 或等价的符号编码 | 仍有非零梯度，持续奖励置信度 | 可以通过 sigmoid 解释，但仍需校准 |
| hinge | $-1,+1$ | margin 达到 $1$ 后损失为零 | 否 |
| squared hinge | $-1,+1$ | margin 达到 $1$ 后损失为零 | 否 |

hinge 很适合“边界两侧分开且留出间隔”的目标，例如线性 SVM；如果需要可比较的事件概率，不能把一个任意缩放的 hinge 分数直接读成概率。也不要把 $0,1$ 标签传给期待 $-1,+1$ 的 margin 公式，否则负类的方向会被破坏。

## 归约、mask 和无效标签

分类损失通常先得到一个 batch 或 token 网格：

$$
\boldsymbol\ell=(\ell_1,\ldots,\ell_n).
$$

无权平均是

$$
L=\frac1n\sum_{i=1}^{n}\ell_i.
$$

若样本有权重 $w_i$：

$$
L_w=\frac{\sum_iw_i\ell_i}{\sum_iw_i}.
$$

序列分类或语言模型常常需要 mask。令 $m_i\in\{0,1\}$ 表示 token 是否有效：

$$
L_{\mathrm{mask}}=\frac{\sum_i m_i\ell_i}{\sum_i m_i}.
$$

这里的分母应该是有效标签数，而不是带 padding 的矩形张量长度。若一批样本有更多 padding，却仍除以相同的总 token 数，有效 token 的梯度会被无效位置稀释。

ignore_index 一类接口通常实现的就是“被忽略位置不参与分子和分母”，但不同库可能对全被忽略的 batch、权重和 reduction 有不同约定。核验时至少构造三组输入：

1. 没有 mask 的单样本，确认逐样本损失；
2. 一半位置被 mask，确认分母只数有效位置；
3. 所有位置都被 mask，确认返回值、梯度和日志是否有明确约定。

类别权重、样本权重和 mask 可以同时存在。一个可审计的目标应明确写出每个因子和最终分母：

$$
L=\frac{\sum_i m_iw_i\ell_i}{\sum_i m_iw_i}.
$$

把无效标签先填成一个合法类别再依赖“它的权重很小”不是 mask；它仍可能在 logits 中制造梯度。无效位置应在损失计算前被明确排除。

## 如何选分类损失

选择时把“任务结构”和“概率/决策要求”分开：

| 需求 | 推荐起点 | 先检查什么 |
| --- | --- | --- |
| 二分类且需要概率 | BCE with logits | sigmoid、logit 稳定性、校准 |
| 互斥多分类且需要概率 | softmax CE with logits | 类别是否互斥、log-sum-exp、每类校准 |
| 多标签且标签可共存 | independent BCE with logits | 标签轴归约、每标签阈值与稀疏性 |
| 严重类别不平衡 | weighted BCE/CE 或 focal | 权重是否改变概率解释、少数类噪声 |
| 只关心间隔与排序边界 | hinge 或 squared hinge | 标签编码、margin、支持向量 |
| 目标本身是软分布 | soft-label CE | 目标来源、熵、分布是否归一化 |
| 序列中有 padding | 任一匹配输出的损失加 mask | 有效标签分母、全 mask batch |

一个常见的顺序是：先用未加权 BCE 或 CE 建立概率基线；再根据验证集中的部署代价选择阈值；只有当类别不平衡确实让梯度被容易样本淹没时，才引入权重或 focal。这样可以把“模型没学会排序”“概率没校准”和“阈值不合适”分开。

## 运行方法

下面的纯标准库脚本把几种输出契约放在同一组可复现数字上。它故意同时计算概率、损失和对 logit 的梯度：

```python
from math import exp, log, log1p


def sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + exp(-z))
    e = exp(z)
    return e / (1.0 + e)


def softplus(z):
    return max(z, 0.0) + log1p(exp(-abs(z)))


def bce_logits(z, y):
    return softplus(z) - y * z


def focal(z, y, gamma=2.0, alpha=1.0):
    p = sigmoid(z)
    p_t = p if y == 1 else 1.0 - p
    return -alpha * (1.0 - p_t) ** gamma * log(p_t)


def multiclass(logits, target):
    maximum = max(logits)
    lse = maximum + log(sum(exp(value - maximum) for value in logits))
    probabilities = [exp(value - lse) for value in logits]
    loss = lse - logits[target]
    gradient = [
        probability - (1.0 if index == target else 0.0)
        for index, probability in enumerate(probabilities)
    ]
    return loss, probabilities, gradient


def rounded(values):
    return [round(value, 12) for value in values]


binary_z = 2.0
binary_y = 1
binary_p = sigmoid(binary_z)
binary_loss = bce_logits(binary_z, binary_y)
binary_grad = binary_p - binary_y

multiclass_loss, multiclass_p, multiclass_grad = multiclass(
    [2.0, 1.0, 0.0],
    target=1,
)

multilabel_z = [2.0, -1.0, 0.0]
multilabel_y = [1, 0, 1]
multilabel_losses = [
    bce_logits(z, y)
    for z, y in zip(multilabel_z, multilabel_y)
]
multilabel_grads = [
    sigmoid(z) - y
    for z, y in zip(multilabel_z, multilabel_y)
]

eta = 0.2
alpha = 4.0
beta = 1.0
weighted_optimal_p = (
    alpha * eta
    / (alpha * eta + beta * (1.0 - eta))
)

print(
    f"binary p={binary_p:.12f} "
    f"BCE={binary_loss:.12f} "
    f"grad={binary_grad:.12f} "
    f"focal={focal(binary_z, binary_y):.12f}"
)
print(
    f"multiclass probs={rounded(multiclass_p)} "
    f"CE={multiclass_loss:.12f} "
    f"grad={rounded(multiclass_grad)}"
)
print(
    f"multilabel mean-BCE="
    f"{sum(multilabel_losses) / len(multilabel_losses):.12f} "
    f"grads={rounded(multilabel_grads)}"
)
print(
    f"weighted eta={eta:.1f} alpha={alpha:.0f} beta={beta:.0f} "
    f"calibrated-threshold={beta / (alpha + beta):.12f} "
    f"weighted-optimal-p={weighted_optimal_p:.12f}"
)
print(
    f"hinge z=0.5={max(0.0, 1.0 - 0.5):.12f} "
    f"squared-hinge={max(0.0, 1.0 - 0.5) ** 2:.12f}"
)
print(
    f"stable-extremes bce(z=1000,y=0)={bce_logits(1000.0, 0):.1f} "
    f"bce(z=-1000,y=1)={bce_logits(-1000.0, 1):.1f}"
)
```

输出为：

```text
binary p=0.880797077978 BCE=0.126928011043 grad=-0.119202922022 focal=0.001803562835
multiclass probs=[0.665240955775, 0.244728471055, 0.09003057317] CE=1.407605964444 grad=[0.665240955775, -0.755271528945, 0.09003057317]
multilabel mean-BCE=0.377778959707 grads=[-0.119202922022, 0.26894142137, -0.5]
weighted eta=0.2 alpha=4 beta=1 calibrated-threshold=0.200000000000 weighted-optimal-p=0.500000000000
hinge z=0.5=0.500000000000 squared-hinge=0.250000000000
stable-extremes bce(z=1000,y=0)=1000.0 bce(z=-1000,y=1)=1000.0
```

这里的最后一行不是说“极端样本的损失一定是 $1000$”，而是说明稳定实现可以在极端 logit 上返回有限、可解释的结果。对 $z=1000,y=0$，模型以极高置信度预测了错误类别，损失接近 $z$；对 $z=-1000,y=1$，同理损失接近 $-z$。

### 数字和梯度的三重核对

审计不能只比较一个 loss 标量：

1. 对 BCE，检查 $p$、$p-y$ 和有限差分是否方向一致；
2. 对 softmax CE，检查概率和为 $1$，梯度各项和为 $0$；
3. 对多标签 BCE，检查每个标签独立产生梯度，而不是被其他标签归一化；
4. 对加权损失，检查权重改变的是目标和阈值，不要把它误报告成原始校准概率；
5. 对 focal，至少比较容易样本和困难样本的相对权重；
6. 对 hinge，检查 margin 超过 $1$ 后损失与梯度是否归零；
7. 对 mask，检查分母只计算有效标签。

若代码同时提供 probability、logit 和 loss 三个接口，先用一个小数字例子逐项对齐，再接入大模型。分类损失的很多 bug 不是反向传播算法错，而是标签编码、输出契约、归约分母或稳定公式错位。

## 失效模式

**把互斥多分类写成独立 sigmoid。** 这会允许所有类别同时达到高概率，失去总和为 $1$ 的竞争约束。先确认一个样本是否只能有一个类别。

**把多标签任务强行 softmax。** softmax 会让一个标签变强时挤压其他标签；如果标签可以共存，应使用独立 sigmoid 和逐标签 BCE。

**把概率先 sigmoid 或 softmax，再传给 logits 损失。** 这会重复变换并破坏稳定的 softplus 或 log-sum-exp；接口若要求 logits，就直接传原始分数。

**直接用 $\log p$ 和 $\log(1-p)$ 处理极端概率。** 概率可能先舍入成 $0$ 或 $1$，产生无穷或 NaN。使用稳定的 logits 公式，并用 $\pm1000$ 做边界测试。

**把 class weight 当作只影响训练速度。** 加权风险的最优输出通常不再是原始条件概率。训练后要检查校准，并明确阈值是否已经吸收了业务代价。

**把 focal 当作免费的 BCE 替代品。** focal 会重新分配容易样本和困难样本的学习预算，可能放大标签噪声，也可能损害概率校准。指标改善要同时报告概率指标。

**把 hinge 分数当概率。** hinge 只要求 margin，不规定一个概率分布。若需要风险读数，应训练概率模型或另行校准分数。

**把标签值 $0,1$ 与 $-1,+1$ 混用。** BCE 和 hinge 的标签编码不是同一个接口契约。转换标签后要同时转换损失、梯度和阈值解释。

**忽略 label smoothing 已经改变目标。** 平滑后的最优分布是 $\boldsymbol q$，不是原始 one-hot。比较实验时必须记录 $\varepsilon$ 和软标签来源。

**mask 位置仍然参与分母或梯度。** padding 可能让不同批次的损失尺度随长度变化。检查有效标签数、全 mask batch 和每个归约轴。

**只看 accuracy 选择损失。** accuracy 不区分 $0.51$ 和 $0.99$ 的正确概率，也不说明少数类和错误代价。至少保留交叉熵、每类指标、校准和阈值下的决策指标。

## 一个可复用的核验协议

遇到新的分类损失或替换输出头，可以按下面顺序检查：

1. 写出标签空间：二分类、互斥多分类、多标签还是有序标签；
2. 写出模型最后一层的实际输出：logit、概率、概率向量还是 margin；
3. 对照接口约定，确认损失不会重复 sigmoid、softmax 或 log；
4. 用一个正确样本、一个自信错分样本和一个边界样本手算损失；
5. 逐 logit 核对解析梯度和中心差分，检查 BCE/CE 的 $p-y$ 形式；
6. 对 softmax 检查概率和为 $1$、梯度和为 $0$，对多标签检查各轴独立；
7. 对权重、focal 和 label smoothing，记录它们改变的目标分布；
8. 对 hinge 检查 $yz=1$ 两侧的次梯度和 margin 超界后的零梯度；
9. 对 mask 和 reduction 检查有效样本数、权重和以及全无效输入；
10. 在独立验证集上分开报告训练损失、概率校准和最终阈值决策。

这套协议的核心是把“输出是什么”“损失要求什么”和“部署要做什么”拆开。损失函数可以提供训练方向，却不会自动替你选择标签语义、概率校准或业务阈值。

## 相关词条

- [损失函数](../training-nn/loss-functions/)：统一比较逐样本损失、训练目标、指标和部署代价。
- [交叉熵](../information-theory/cross-entropy/)：从分布编码代价推导交叉熵、KL 散度和 softmax 形式。
- [阶跃与 sigmoid](../neurons-and-activations/heaviside-and-sigmoid/)：理解二分类输出、BCE 梯度和稳定的 sigmoid。
- [Softmax 回归](../linear-models/softmax-regression/)：展开互斥多分类的 logits、概率、梯度和类别竞争。
- [分类指标](../evaluation-and-generalization/classification-metrics/)：在冻结模型上评估混淆矩阵、排序、校准和阈值决策。
- [最大似然](../probability/maximum-likelihood/)：理解负对数似然为何成为分类训练目标。
- [标签平滑](../training-nn/label-smoothing/)：进一步讨论平滑目标对训练和校准的影响。
- [梯度下降](../training-nn/gradient-descent/)：把损失对 logit 的梯度接到参数更新。
