---
title: "交叉熵：用模型概率给真实结果编码"
tags: ["why-models-learn"]
---

**交叉熵**用一个分布 $q$ 给另一个分布 $p$ 产生的结果编码，定义为

$$
H(p,q)
=-\sum_xp(x)\log q(x)
$$

第一个分布 $p$ 是真实结果或数据分布，第二个分布 $q$ 是编码器、分类器或语言模型提供的概率。若 $q$ 把实际发生的结果赋予很小的概率，交叉熵就会很大；若 $q=p$，交叉熵退化为熵。分类交叉熵、二元交叉熵和 token 级负对数似然都是这个定义的不同展开。本篇先区分 $p$ 和 $q$ 的位置，再推导 logits、softmax、label smoothing、序列 mask 和连续高斯回归的损失。

## 两个分布的角色不能交换

离散分布 $p$ 产生结果，模型 $q$ 为这些结果分配编码代价。一次观察到 $x$ 时，模型给出的代价是

$$
-\log q(x)
$$

在真实分布 $p$ 下反复抽样并取平均：

$$
\begin{aligned}
H(p,q)
&=\mathbb E_{X\sim p}[-\log q(X)]\\
&=-\sum_xp(x)\log q(x)
\end{aligned}
$$

熵使用自己的概率：

$$
H(p)=H(p,p)
=-\sum_xp(x)\log p(x)
$$

交叉熵使用 $q$ 的概率：

$$
H(p,q)
=-\sum_xp(x)\log q(x)
$$

一般情况下

$$
H(p,q)\neq H(q,p)
$$

因为换了谁来产生样本，平均的对象就变了。训练分类器时，标签分布是 $p$ 的角色，模型输出是 $q$ 的角色；不能把模型输出当作加权分布后再解释成同一个量。

## 交叉熵等于真实熵加额外代价

在 $p(x)>0$ 的位置插入 $\log p(x)$：

$$
\begin{aligned}
H(p,q)
&=-\sum_xp(x)\log q(x)\\
&=-\sum_xp(x)\log p(x)
+\sum_xp(x)\log\frac{p(x)}{q(x)}\\
&=H(p)+D_{\mathrm{KL}}(p\Vert q)
\end{aligned}
$$

KL 散度满足

$$
D_{\mathrm{KL}}(p\Vert q)\geq0
$$

所以

$$
H(p,q)\geq H(p)
$$

且只有 $p=q$ 时达到相等。交叉熵中的第一部分是数据本身的平均不确定性，第二部分是模型和真实分布不匹配造成的额外编码代价。KL 的非负性和更多性质将在后续词条单独推导。

### 三类别数字例子

令真实分布为

$$
p=(0.7,0.2,0.1)
$$

模型输出为

$$
q=(0.6,0.3,0.1)
$$

用 bit 计算真实熵：

$$
H_2(p)
=-0.7\log_2 0.7
-0.2\log_2 0.2
-0.1\log_2 0.1
\approx1.1568
$$

交叉熵为

$$
\begin{aligned}
H_2(p,q)
&=-0.7\log_2 0.6
-0.2\log_2 0.3
-0.1\log_2 0.1\\
&\approx1.1955
\end{aligned}
$$

所以模型失配带来的额外代价约为 $0.0387$ bit。若模型把第三类概率从 $0.1$ 错降到 $10^{-6}$，只要第三类在真实分布中仍有概率 $0.1$，交叉熵就会大幅增加。

## 一个样本的分类交叉熵

有 $K$ 个类别，真实标签用概率向量

$$
\boldsymbol y=(y_1,\ldots,y_K),
\qquad
y_k\geq0,
\qquad
\sum_{k=1}^Ky_k=1
$$

模型输出

$$
\boldsymbol q=(q_1,\ldots,q_K)
$$

单个样本的交叉熵损失是

$$
\mathcal L(\boldsymbol y,\boldsymbol q)
=-\sum_{k=1}^Ky_k\log q_k
$$

如果标签是类别 $c$ 的 one-hot 向量：

$$
y_c=1,
\qquad
y_{k\neq c}=0
$$

那么损失简化为

$$
\mathcal L=-\log q_c
$$

模型只需要为真实类别分配概率，其他类别的概率通过归一化间接影响 $q_c$。正确类别的概率为 $1$ 时损失为 0；为 $1/2$ 时损失是 $\ln2$ nats；趋近 0 时损失趋向无穷。

### 为什么它是 proper scoring rule

假设真实类别分布为 $\boldsymbol p$，我们可以选择报告的分布为 $\boldsymbol q$。期望损失是

$$
\mathbb E_{Y\sim p}[-\log q_Y]
=H(p,q)
$$

由上一节的分解：

$$
H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)
$$

对固定的 $p$，最小值在 $q=p$ 处取得。也就是说，如果目标是最小化期望对数损失，诚实报告自己的概率分布是最优的；只报告一个类别名则丢掉了置信度信息。这是交叉熵作为概率预测评分规则的核心性质。

## 二元交叉熵：Bernoulli 的负对数似然

二分类标签 $y\in\{0,1\}$，模型预测正类概率 $q\in(0,1)$。Bernoulli 概率是

$$
q^y(1-q)^{1-y}
$$

取负对数得到二元交叉熵：

$$
\mathcal L(y,q)
=-y\log q-(1-y)\log(1-q)
$$

$y=1$ 时：

$$
\mathcal L=-\log q
$$

$y=0$ 时：

$$
\mathcal L=-\log(1-q)
$$

如果真实正类概率是 $p$，模型固定报告 $q$，期望损失为

$$
L(q)
=-p\log q-(1-p)\log(1-q)
$$

对 $q$ 求导：

$$
\frac{\mathrm dL}{\mathrm dq}
=-\frac pq+\frac{1-p}{1-q}
=\frac{q-p}{q(1-q)}
$$

令导数为 0：

$$
\widehat q=p
$$

二阶导数为

$$
\frac{\mathrm d^2L}{\mathrm dq^2}
=\frac p{q^2}+\frac{1-p}{(1-q)^2}>0
$$

所以二元交叉熵的期望值在真实概率处唯一最小。它不是强迫模型输出 0 或 1 的损失，恰恰相反，正确的最优输出是数据条件概率。

### 数字例子

真实正类概率 $p=0.8$，模型报告 $q=0.6$ 时，平均 BCE 为

$$
\begin{aligned}
L(0.6)
&=-0.8\ln0.6-0.2\ln0.4\\
&\approx0.5919\ \mathrm{nats}
\end{aligned}
$$

如果报告正确的 $q=0.8$：

$$
L(0.8)
=-0.8\ln0.8-0.2\ln0.2
\approx0.5004\ \mathrm{nats}
$$

差值约 $0.0915$ nat，就是这个错误概率分布带来的额外 KL 代价。

## 从 logits 到 softmax 交叉熵

神经网络通常输出未归一化 logits：

$$
\boldsymbol z=(z_1,\ldots,z_K)
$$

softmax 定义为

$$
q_k
=\frac{\exp(z_k)}
{\sum_{j=1}^K\exp(z_j)}
$$

对 one-hot 类别 $c$：

$$
\begin{aligned}
\mathcal L
&=-\log q_c\\
&=-z_c+\log\sum_{j=1}^K\exp(z_j)
\end{aligned}
$$

令

$$
\operatorname{LSE}(\boldsymbol z)
=\log\sum_{j=1}^K\exp(z_j)
$$

就是

$$
\mathcal L
=\operatorname{LSE}(\boldsymbol z)-z_c
$$

### 梯度为什么是预测减标签

对一般软标签 $\boldsymbol y$：

$$
\mathcal L(\boldsymbol y,\boldsymbol z)
=-\sum_ky_k\log q_k
=\operatorname{LSE}(\boldsymbol z)
-\sum_ky_kz_k
$$

因为

$$
\frac{\partial\operatorname{LSE}(\boldsymbol z)}
{\partial z_k}
=q_k
$$

而

$$
\frac{\partial}{\partial z_k}
\left(-\sum_jy_jz_j\right)
=-y_k
$$

所以

$$
\frac{\partial\mathcal L}{\partial z_k}
=q_k-y_k
$$

这个简洁梯度来自 softmax 的归一化和对数损失的配对，不是把预测减标签当作独立经验公式硬编码进去。

### 数字例子

取

$$
\boldsymbol z=(2,1,0)
$$

softmax 概率约为

$$
\boldsymbol q
\approx(0.6652,0.2447,0.0900)
$$

若真实类别是第一个类别：

$$
\mathcal L=-\log q_1
\approx0.4076\ \mathrm{nats}
$$

梯度向量为

$$
\nabla_{\boldsymbol z}\mathcal L
\approx(-0.3348,0.2447,0.0900)
$$

增加真实类别 logit 会降低损失，增加其他类别 logit 会提高损失。

## 数值稳定性：不要直接计算溢出的指数

直接计算 $\exp(z_k)$ 可能在 logits 很大时溢出。令

$$
m=\max_jz_j
$$

则

$$
\operatorname{LSE}(\boldsymbol z)
=m+\log\sum_j\exp(z_j-m)
$$

因为每个 $z_j-m\leq0$，指数项不会超过 1。交叉熵稳定地写成

$$
\mathcal L
=m+\log\sum_j\exp(z_j-m)-z_c
$$

二元情形令 $q=\operatorname{sigmoid}(z)$，损失可以写成

$$
\mathcal L(y,z)
=\max(z,0)-yz+\log(1+\exp(-|z|))
$$

这个形式在 $z$ 很大或很小时都避免了直接计算极端的 $\exp(z)$。框架中的 CrossEntropyLoss 通常接收 logits，而不是已经 softmax 后的概率，原因正是可以在同一个稳定表达式中完成归一化和对数。

## Label smoothing：改变目标分布

one-hot 标签把全部目标质量放在一个类别。label smoothing 用

$$
y_k^{(\varepsilon)}
=(1-\varepsilon)\mathbf 1_{\{k=c\}}
+\frac{\varepsilon}{K}
$$

替代原标签，其中 $\varepsilon\in[0,1]$。这仍然使用交叉熵：

$$
\mathcal L_{\mathrm{smooth}}
=-\sum_ky_k^{(\varepsilon)}\log q_k
$$

它的作用是让目标分布不再包含精确的 0 和 1，减弱模型把 logits 推向极端的压力。代价是训练目标已经不再是对原始 one-hot 经验分布的纯最大似然；$\varepsilon$ 是额外的建模选择。

如果 $K=4$、真实类别为 1、$\varepsilon=0.1$：

$$
\boldsymbol y^{(\varepsilon)}
=(0.925,0.025,0.025,0.025)
$$

正确类别仍然占主要权重，但其他类别也得到非零目标概率。它不等于人为声明每个错误类别同样可能，而是对硬标签不确定性做了一个固定的正则化近似。

## 批次、mask 和平均口径

对批次中的 $n$ 个样本，平均交叉熵通常写成

$$
\widehat L
=-\frac1n
\sum_{i=1}^n
\sum_{k=1}^Ky_{ik}\log q_{ik}
$$

但序列模型还要处理 padding。设 $m_{it}\in\{0,1\}$ 是 token mask：

$$
\widehat L_{\mathrm{token}}
=-
\frac{\sum_{i,t}m_{it}
\log q_\theta(x_{it}\mid x_{i,<t})}
{\sum_{i,t}m_{it}}
$$

分母应该是有效 token 数，而不是带 padding 的矩形总大小。否则短序列和长序列会被不同程度地加权，训练损失也不能直接比较。

类别权重会把目标改成

$$
\widehat L_{\mathrm{weighted}}
=-\frac1n
\sum_iw_{y_i}\log q_{i,y_i}
$$

它可以补偿类别不平衡，却不再是未经加权数据分布的普通交叉熵。报告结果时要说明是否使用了 class weight、采样重平衡或 label smoothing。

## 连续回归也可以使用负对数似然

交叉熵的积分版本是

$$
H(p,q)
=-\int p(x)\log q(x)\,\mathrm dx
$$

设模型使用固定方差 $\sigma^2$ 的高斯分布：

$$
q(x\mid\mu)
=\frac1{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$$

单个观测的负对数似然为

$$
-\log q(x\mid\mu)
=\frac12\log(2\pi\sigma^2)
+\frac{(x-\mu)^2}{2\sigma^2}
$$

固定 $\sigma^2$ 时，最小化它与最小化平方误差等价；但它还保留了概率单位和方差尺度。如果让模型同时预测 $\mu$ 和 $\sigma^2$：

$$
-\log q(x\mid\mu,\sigma^2)
=\frac12\log(2\pi\sigma^2)
+\frac{(x-\mu)^2}{2\sigma^2}
$$

模型可以用更大的方差解释难预测的样本，但第一项会惩罚无条件增大方差。此时 NLL 和 MSE 不再是同一个优化目标。

例如 $x=3$、$\mu=2$、$\sigma^2=4$：

$$
-\log q(3\mid2,4)
=\frac12\left[\ln(8\pi)+\frac14\right]
\approx1.7371\ \mathrm{nats}
$$

## 交叉熵和最大似然

对独立数据 $x_1,\ldots,x_n$，经验交叉熵是

$$
\widehat H(p_{\mathrm{data}},q_\theta)
=-\frac1n\sum_{i=1}^n
\log q_\theta(x_i)
$$

乘以 $n$ 后：

$$
n\widehat H
=-\sum_{i=1}^n\log q_\theta(x_i)
=-\log\prod_{i=1}^nq_\theta(x_i)
$$

因此最小化经验交叉熵就是最大化数据的对数似然。指数族词条中的 NLL、分类输出层和 MLE 都可以从这条等价关系接上；交叉熵不是一个只属于分类的特殊损失。

若训练分布是 $p_{\mathrm{train}}$，部署分布变成 $p_{\mathrm{test}}$，训练时最小化的是

$$
H(p_{\mathrm{train}},q_\theta)
$$

而部署时测量的是

$$
H(p_{\mathrm{test}},q_\theta)
$$

数据分布变化会同时改变真实熵和模型失配项，训练集交叉熵下降不能保证测试交叉熵也下降。

![交叉熵是用模型概率编码真实分布的平均代价](/assets/information-theory/svg/cross-entropy.1.svg)

## 失效模式

**把熵和交叉熵交换**：$H(p)$ 的第二个参数隐含为 $p$，交叉熵的第二个参数是模型 $q$。训练损失使用的是 $-\log q$，不是 $-\log p$。

**只看准确率不看概率**：两个分类器可以预测同样的类别，但一个给正确类别 0.51，另一个给 0.99。前者的交叉熵更大，因为它的概率预测更不确定。

**让正确类别概率为零**：one-hot 交叉熵中 $q_c=0$ 会产生无穷损失。数值稳定实现、平滑和支持集设计都要避免实际标签落在零概率上。

**softmax 后再交给 logits 损失**：如果接口期望 logits 却传入已经 softmax 的概率，会重复归一化并破坏数值稳定性。先确认损失函数的输入约定。

**平均分母包含 padding**：序列交叉熵要按有效 token mask 归一化。不同 padding 比例的批次不能用同一个矩形 token 数直接比较。

**把 label smoothing 当作无害实现细节**：它改变了目标分布和最优预测，可能改善校准，也可能降低对硬标签的拟合。实验报告要记录 $\varepsilon$。

**把 MSE 等同于所有回归 NLL**：只有固定方差高斯时，平方项的比例和常数才可以被忽略。异方差高斯、Laplace 或计数分布有不同的概率模型和损失。

**忽略分布漂移**：训练交叉熵是对训练分布的期望，部署数据改变后，原来的均值和分解都要重新评估。

## 相关词条

- [熵](../information-theory/entropy/)：交叉熵在 $p=q$ 时退化为熵。
- [信息量与惊奇度](../information-theory/information-and-surprise/)：单次负对数概率和 token 惊奇度。
- [KL 散度](../information-theory/kl-divergence/)：交叉熵超过真实熵的额外代价。
- [最大似然](../probability/maximum-likelihood/)：经验交叉熵最小化与 NLL 最大化似然等价。
- [指数族](../probability/exponential-family/)：分类、回归和计数输出分布的统一形式。
- [困惑度](../information-theory/perplexity/)：序列平均交叉熵的指数化指标。
- [变量变换](../probability/change-of-variables/)：连续密度交叉熵中的参考测度和 Jacobian 变化。
