---
title: "Softmax 回归：用竞争式概率模型拟合多分类"
tags: ["why-models-learn"]
---

Softmax 回归把每个类别的线性分数放进同一个归一化分母，得到总和为 1 的多分类概率。它用分类分布的似然或 softmax 交叉熵训练这些分数，因此类别之间会竞争；最后取最大概率只是一个决策步骤，不等于训练目标本身。

![Softmax 回归把多组 logits 归一化为类别概率并输出决策](/assets/linear-models/svg/softmax-regression.1.svg)

## 从 logits 到概率

设输入特征为 $xinmathbb R^p$，类别数为 $K$。对每个类别 $k$，先计算一个线性分数：

$$
z_k=b_k+x^{\mathsf T}w_k,
\qquad k=1,\ldots,K.
$$

把这些分数组成向量 $\boldsymbol z=(z_1,\ldots,z_K)^{\mathsf T}$。Softmax 定义为

$$
p_k=\operatorname{softmax}(\boldsymbol z)_k
=\frac{e^{z_k}}{\displaystyle\sum_{j=1}^{K}e^{z_j}}.
$$

分母同时看见所有类别。因为每个指数都为正，所以 $p_k>0$；把所有分子相加以后，分母正好抵消：

$$
\sum_{k=1}^{K}p_k
=\frac{\sum_{k=1}^{K}e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
=1.
$$

所以 $\boldsymbol p$ 可以作为一个分类分布的概率向量。这里要区分三个对象：$z$ 是未归一化的分数，$p$ 是概率，$\widehat y$ 是最后输出的类别。

| 层次 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| 线性层 | $x$ | $z_1,\ldots,z_K$ | 给每个类别一个分数 |
| 概率层 | logits | $p_1,\ldots,p_K$ | 归一化并让概率和为 1 |
| 决策层 | 概率或分数 | 一个类别 | 按规则选择行动 |
| 评估层 | 标签、概率或类别 | 指标 | 分别检查损失、排序和决策 |

## 为什么类别会相互竞争

考虑三个类别的 logits：

$$
\boldsymbol z=(2,1,0)^{\mathsf T}.
$$

三个指数约为 $7.389056$、$2.718282$、$1$，分母约为 $11.107338$，于是

$$
\boldsymbol p
\approx
(0.665241,\,0.244728,\,0.090031)^{\mathsf T}.
$$

第一类的分数只比第二类高 1，但它得到的概率约为第二类的 $2.718$ 倍，因为

$$
\frac{p_a}{p_b}=\frac{e^{z_a}}{e^{z_b}}=e^{z_a-z_b}.
$$

概率的比值只取决于两个分数的差，而不是它们的绝对位置。把所有 logits 同时加上常数 $c$：

$$
\operatorname{softmax}(\boldsymbol z+c\boldsymbol 1)_k
=\frac{e^{z_k+c}}{\sum_j e^{z_j+c}}
=\frac{e^{c}e^{z_k}}{e^{c}\sum_j e^{z_j}}
=p_k.
$$

因此 $\boldsymbol z$ 不是唯一的概率坐标。真正有意义的是类别之间的相对分数；数值实现也正是利用这个不变性来减小指数的大小。

由于指数是严格递增函数，最大分数和最大概率的位置相同：

$$
\arg\max_k p_k=\arg\max_k z_k.
$$

这条等价关系只说明 top-1 决策可以不计算 softmax。它不说明可以跳过 softmax 概率：交叉熵、校准、代价敏感决策和 top-k 概率都仍然需要归一化结果。

## 分类分布的似然与交叉熵

单个样本的标签 $y$ 是 $K$ 个类别中的一个。用独热向量 $\boldsymbol y$ 表示时，$y_k=1$ 只在真实类别处成立，其余位置为 0。分类分布的概率质量可以写成

$$
\Pr(Y=y\mid x)
=\prod_{k=1}^{K}p_k^{y_k}.
$$

只有真实类别那一项留下来：如果真实类别是 $c$，那么

$$
\Pr(Y=c\mid x)=p_c.
$$

对 $n$ 个独立样本取负对数并平均，得到 softmax 交叉熵：

$$
\operatorname{CE}
=-\frac1n\sum_{i=1}^{n}\sum_{k=1}^{K}y_{ik}\log p_{ik}.
$$

因为每个样本只有一个 $y_{ic}=1$，第 $i$ 个样本的损失就是 $-\log p_{ic}$。模型给真实类别的概率越小，损失增长越快；即使最终 top-1 类别没有改变，把 $0.51$ 提高到 $0.90$ 仍然会得到不同的训练信号。

把 softmax 代回损失，真实类别为 $c$ 时有

$$
\ell(\boldsymbol z,c)
=-\log\frac{e^{z_c}}{\sum_j e^{z_j}}
=\log\left(\sum_{j=1}^{K}e^{z_j}\right)-z_c.
$$

第一项是 log-sum-exp，第二项是真实类别 logit。对上面的 $\boldsymbol z=(2,1,0)$：

$$
\begin{aligned}
\ell(\boldsymbol z,1)&=\log(e^2+e^1+e^0)-2\approx0.407606,\\
\ell(\boldsymbol z,3)&=\log(e^2+e^1+e^0)-0\approx2.407606.
\end{aligned}
$$

同一组预测，如果真实类别从第一类换成第三类，损失相差 2，原因只是模型把第三类的概率压得很低。

## 梯度为什么是预测减标签

先看一个 log-sum-exp 坐标：

$$
L(\boldsymbol z)=\log\left(\sum_j e^{z_j}\right).
$$

对第 $k$ 个 logit 求导，分子留下 $e^{z_k}$，整个导数正好是 softmax 概率：

$$
\frac{\partial L}{\partial z_k}
=\frac{e^{z_k}}{\sum_j e^{z_j}}
=p_k.
$$

因此单样本交叉熵 $\ell=L-z_c$ 的梯度为

$$
\frac{\partial\ell}{\partial z_k}=p_k-y_k.
$$

这个表达式同时包含竞争关系。提高某个错误类别的概率，会增加它自己的梯度；提高真实类别的概率，会减少真实类别的梯度；所有坐标的梯度之和为零：

$$
\sum_k(p_k-y_k)=1-1=0.
$$

设样本矩阵 $X\in\mathbb R^{n\times p}$，权重矩阵 $W\in\mathbb R^{p\times K}$，偏置向量 $b\in\mathbb R^K$，则 logits 矩阵为

$$
Z=XW+\boldsymbol 1 b^{\mathsf T}.
$$

令 $P$ 和 $Y$ 分别收集所有样本的概率向量与独热标签，平均交叉熵对参数的梯度是

$$
\nabla_W\operatorname{CE}=\frac1nX^{\mathsf T}(P-Y),
\qquad
\nabla_b\operatorname{CE}=\frac1n\boldsymbol 1^{\mathsf T}(P-Y).
$$

它与二分类逻辑回归的 $p-y$ 结构相同，但每个样本现在产生一个 $K$ 维的竞争误差，而不是一个标量误差。固定前面的表示时，最后一层 softmax 回归仍然是关于 $W,b$ 的凸目标。

## Hessian 是概率协方差矩阵

Softmax 的 Jacobian 不是对角矩阵，因为改变一个 logit 会改变归一化分母。对任意 $j,k$：

$$
\frac{\partial p_k}{\partial z_j}
=p_k(\delta_{kj}-p_j).
$$

把所有坐标放在一起：

$$
J_{\mathrm{softmax}}
=\operatorname{diag}(\boldsymbol p)-\boldsymbol p\boldsymbol p^{\mathsf T}.
$$

对真实类别为 $c$ 的交叉熵，Hessian 就是这个矩阵。取任意向量 $v$，有

$$
\begin{aligned}
v^{\mathsf T}J_{\mathrm{softmax}}v
&=\sum_k p_kv_k^2-\left(\sum_kp_kv_k\right)^2\\
&=\operatorname{Var}_{K\sim\boldsymbol p}(v_K)\ge0.
\end{aligned}
$$

所以它半正定，关于 logits 的交叉熵是凸函数。由于给所有 logits 同时加常数不会改变概率，向量 $\boldsymbol 1$ 是零方向：

$$
J_{\mathrm{softmax}}\boldsymbol 1=\boldsymbol 0.
$$

例如三类均匀概率 $\boldsymbol p=(1/3,1/3,1/3)^{\mathsf T}$ 时，Hessian 为

$$
J_{\mathrm{softmax}}
=\frac19
\begin{pmatrix}
2&-1&-1\\
-1&2&-1\\
-1&-1&2
\end{pmatrix}.
$$

负的非对角项表示类别坐标并不独立：一个类别的概率增大，会挤压其他类别的概率。把这个 Jacobian 当成逐分量导数，会丢掉 softmax 的主要结构。

## 一个三分类梯度更新

继续使用

$$
\boldsymbol z=(2,1,0)^{\mathsf T},
\qquad
\boldsymbol y=(1,0,0)^{\mathsf T},
\qquad
\eta=0.1.
$$

由前面的概率，初始梯度为

$$
\boldsymbol g=\boldsymbol p-\boldsymbol y
\approx(-0.334759,\,0.244728,\,0.090031)^{\mathsf T}.
$$

沿梯度下降更新 logits：

$$
\boldsymbol z_{\mathrm{new}}
=\boldsymbol z-\eta\boldsymbol g
\approx(2.033476,\,0.975527,\,-0.009003)^{\mathsf T}.
$$

真实类别的分数上升，两个错误类别的分数下降。重新计算 softmax 得到

$$
\boldsymbol p_{\mathrm{new}}
\approx(0.677106,\,0.235069,\,0.087825)^{\mathsf T},
$$

真实类别概率从 $0.665241$ 增加到约 $0.677106$，交叉熵从 $0.407606$ 降到约 $0.389928$。这个例子只更新了 logits，实际训练时 logits 又由 $W,b$ 和输入共同决定，参数梯度还要乘上输入特征。

## 用 log-sum-exp 保证数值稳定

直接计算 $e^{z_k}$ 有两个风险：很大的 logit 会上溢，很小的 logit 会下溢。利用前面证明的平移不变性，令

$$
m=\max_j z_j.
$$

则

$$
\operatorname{softmax}(\boldsymbol z)_k
=\frac{e^{z_k-m}}{\sum_j e^{z_j-m}},
$$

所有指数都不超过 1。对应的稳定 log-sum-exp 是

$$
\operatorname{LSE}(\boldsymbol z)
=m+\log\left(\sum_j e^{z_j-m}\right).
$$

当 $\boldsymbol z=(1002,1001,1000)$ 时，直接算 $e^{1002}$ 很容易溢出；先减去最大值后，实际计算的指数仍然只是 $1,e^{-1},e^{-2}$，得到的概率与 $(2,1,0)$ 完全相同。

训练接口通常应直接接收 logits，在同一个稳定表达式里完成 log-sum-exp 和真实类别项：

$$
\ell(\boldsymbol z,c)
=\operatorname{LSE}(\boldsymbol z)-z_c.
$$

如果先把 logits softmax 成概率，再把概率传给一个期望 logits 的交叉熵接口，接口可能会再次做 log-sum-exp，结果既不稳定也不再是原来的损失。

## 决策不只等于 argmax

最简单的多分类决策是

$$
\widehat y=\arg\max_k p_k.
$$

它适用于每次必须选择一个类别、各类错误代价相近的场景。如果可以拒绝或转人工，可以设置置信度门槛：

$$
\widehat y=
\begin{cases}
\arg\max_k p_k,&\max_kp_k\ge\tau,\\
\text{拒绝或转人工},&\max_kp_k<\tau.
\end{cases}
$$

如果不同动作有不同代价，决策应最小化条件期望代价，而不是机械地取最大概率：

$$
\widehat a
=\arg\min_a\sum_{k=1}^{K}C(a,k)p_k.
$$

其中 $C(a,k)$ 是真实类别为 $k$ 时采取动作 $a$ 的代价。模型的概率输出、决策规则和评估指标可以分开变化：同一模型可以在不同业务代价下使用不同的 $C$ 或 $\tau$。

| 输出用途 | 需要什么 | 常见规则 |
| --- | --- | --- |
| 单一类别 | 最高概率类别 | argmax |
| top-k 检索 | 概率排序 | 取前 $k$ 个类别 |
| 低置信度转人工 | 最大概率和阈值 | $\max_kp_k\ge\tau$ |
| 代价敏感行动 | 概率和代价矩阵 | 最小化期望代价 |
| 概率决策 | 校准概率 | 直接计算风险或资源分配 |

把概率、决策和代价混在训练阶段，常常会让一个阈值变化牵连整个模型。只要部署目标仍是同一个条件分布，先拟合概率，再单独选择决策规则通常更容易审计。

## 温度改变尖锐程度，不改变排序

可以在推理时引入温度 $T>0$：

$$
p_k(T)=\frac{e^{z_k/T}}{\sum_j e^{z_j/T}}.
$$

$T<1$ 会放大分数差异，分布更尖；$T>1$ 会缩小分数差异，分布更平。由于除以正数不会改变大小关系，温度不会改变 argmax。对 $\boldsymbol z=(2,1,0)$：

$$
\begin{aligned}
\boldsymbol p(0.5)&\approx(0.866813,\,0.117310,\,0.015876)^{\mathsf T},\\
\boldsymbol p(2)&\approx(0.506480,\,0.307196,\,0.186324)^{\mathsf T}.
\end{aligned}
$$

如果模型的排序基本正确，但概率过于自信，可以在验证集上拟合一个温度做 temperature scaling。这个选择不能在测试集上反复调到最好；测试集要留给冻结后的最终报告。温度缩放调整概率的尖锐程度，不会修复类别标签错、排序错或分布已经改变的问题。

## 参数冗余与正则化

把每个类别的权重堆成矩阵 $W$，每一行对应一个类别。对任意向量 $a$ 和常数 $c$，同时做

$$
W'=W+\boldsymbol 1_Ka^{\mathsf T},
\qquad
b'=b+c\boldsymbol 1_K
$$

会让每个样本的所有 logits 同时增加 $a^{\mathsf T}x+c$，softmax 概率不变。这是参数不可识别方向。可以固定一个参考类别的权重和偏置为零，也可以保留冗余参数并让正则化和优化器处理它。

常见的 L2 正则化目标是

$$
J(W,b)
=\operatorname{CE}(W,b)
+\frac\lambda2\lVert W\rVert_F^2.
$$

它会倾向于选择较小的类间权重，改善病态和过拟合；是否惩罚偏置要按参数化约定决定。L1 正则化可以推动某些特征的整列权重变为零，但没有普通最小二乘那样简单的闭式解，需要使用合适的凸优化方法。正则化改变的是训练时的参数偏好，不能代替验证集上的温度、阈值和类别代价选择。

## 类别权重与标签平滑

如果少数类别的错误代价更高，可以使用类别权重：

$$
\operatorname{CE}_w
=-\frac1n\sum_{i=1}^{n}w_{y_i}\log p_{i,y_i}.
$$

权重会改变梯度中不同类别样本的相对贡献。它可能改善少数类召回，也可能使输出概率不再直接对应原始部署分布；训练后需要重新检查校准和决策阈值。

标签平滑把独热目标改成

$$
\widetilde y_{ik}
=(1-\varepsilon)y_{ik}+\frac\varepsilon K,
\qquad 0\le\varepsilon<1.
$$

于是模型不再被要求给真实类别概率精确等于 1。它可以减少极端 logits，但也改变了原始 one-hot 最大似然目标；如果需要严格概率解释，应把平滑、权重和后续校准一起记录。

Softmax 回归还带有一个重要的任务假设：每个样本只属于一个互斥类别。如果一个样本可以同时拥有多个标签，就不应强行让所有类别概率加起来为 1，而应考虑每个标签独立的 sigmoid 模型或其他多标签模型。

## Softmax 与 one-vs-rest 的区别

one-vs-rest 为每个类别训练一个二分类器，得到若干独立的正类分数或概率；softmax 则在一个分类分布中联合归一化。二者都可以用于多分类，但它们表达的约束不同：

| 方面 | Softmax 回归 | one-vs-rest |
| --- | --- | --- |
| 概率约束 | 所有类别概率和为 1 | 各二分类器通常独立 |
| 训练竞争 | 一个类别变强会相对挤压其他类别 | 每个分类器单独拟合 |
| 决策方式 | 通常取最大概率或最小期望代价 | 比较分数、阈值或校准后的概率 |
| 标签假设 | 单标签互斥多分类 | 可扩展到多个独立检测任务 |
| 置信度 | 联合分布中的相对概率 | 需要额外检查跨分类器可比性 |

如果类别天然互斥，softmax 的联合似然通常更直接；如果任务是多个属性是否存在，独立 sigmoid 更符合标签结构。选择哪一个取决于标签语义，而不是只看哪种输出层更常见。

## 如何评估多分类模型

交叉熵检查概率质量，但它不能替代所有部署指标。多分类评估至少要说明样本权重、类别平均方式和是否使用 top-k：

| 问题 | 指标或图 | 需要说明的细节 |
| --- | --- | --- |
| 单次决策是否正确 | accuracy、混淆矩阵 | 类别是否平衡、是否允许拒绝 |
| 少数类是否被忽略 | macro F1、每类召回率 | macro 还是 weighted 平均 |
| 排名中是否包含真类 | top-k accuracy | $k$ 的取值和候选集合 |
| 概率是否可信 | 多分类 log loss、Brier 分数、可靠性图 | 校准分箱和权重 |
| 代价是否可接受 | 期望代价 | 成本矩阵和部署先验 |

例如 top-1 accuracy 只检查最大类别，top-3 accuracy 允许真实类别出现在前三名；交叉熵还会区分“把真实类别排第二但概率接近”和“给真实类别几乎为零”这两种情况。模型选择应使用与实际行动相符的验证指标，最终测试报告要同时保留概率和硬决策证据。

## 一个可复用的训练流程

1. 先确认标签是互斥单标签，确定类别顺序和缺失类别的处理规则；
2. 按实体、时间或分层规则切分训练、验证和测试数据；
3. 只用训练集拟合标准化、特征处理、类别权重和标签映射；
4. 训练时传入 logits 和整数标签，使用稳定的 softmax 交叉熵；
5. 监控训练/验证交叉熵、每类召回率、混淆矩阵和概率校准；
6. 在验证集选择正则化、温度、拒绝阈值和代价矩阵；
7. 冻结选择后，在独立测试集报告 top-1、top-k、每类指标、log loss 和校准；
8. 在时间外、分布外和少数类子群上复查概率与决策是否仍然适用。

如果训练接口的损失函数期望 logits，就不要在模型前向过程中先做 softmax。保留原始 logits 也有助于调试平移不变性、温度缩放和数值稳定性。

## 失效模式

**把 logits 直接当成概率。** logits 可以是任意实数，也不保证总和为 1；先确认输出层和损失函数的输入约定。

**先 softmax 再交给 logits 损失。** 这会重复归一化，破坏稳定的 log-sum-exp 计算；框架若提供接收 logits 的交叉熵接口，应直接传入 logits。

**把 softmax 当成独立逐分量函数。** 分母包含全部类别，Jacobian 有负的非对角项；逐坐标求导会漏掉类别竞争。

**直接计算极大的指数。** 先减去最大 logit，再计算 softmax 或 log-sum-exp；不要把浮点溢出当成模型本身的证据。

**互斥标签却用独立 sigmoid，或多标签任务却用 softmax。** 输出约束必须和标签语义一致，概率和为 1 不是所有分类任务都需要的性质。

**只用 accuracy 选择模型。** 类别不平衡时准确率可能掩盖少数类失败；至少同时检查每类指标、交叉熵和混淆矩阵。

**把高置信度当成校准良好。** softmax 只保证归一化，不保证 $0.8$ 的样本长期有约 80% 属于预测类别；校准要在独立验证数据上检查。

**在测试集调温度和拒绝阈值。** 这会让最终评估反馈到模型选择，测试结果不再代表冻结流程的泛化表现。

**忘记参数不可识别方向。** 所有 logits 同时加常数不改变概率；解释权重或比较优化结果时，要固定参考类别或明确使用了何种正则化。

## 相关词条

- [逻辑回归](../linear-models/logistic-regression/)：二分类 Bernoulli 概率模型以及 sigmoid、阈值和校准。
- [交叉熵](../information-theory/cross-entropy/)：从分布编码代价推导 softmax 交叉熵和 log-sum-exp。
- [离散分布](../probability/discrete-distributions/)：理解分类分布的概率质量和独热标签。
- [最大似然估计](../probability/maximum-likelihood/)：从似然和对数似然的统一视角理解参数拟合。
- [岭回归与 Lasso](../linear-models/ridge-and-lasso/)：比较 L1/L2 正则化、收缩和验证选择。
- [训练、验证与测试集](../learning-framework/train-validation-test/)：规定正则化、温度和阈值的选择边界。
- [分类指标](../evaluation-and-generalization/classification-metrics/)：系统展开混淆矩阵、排序、概率和代价指标。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：连接参数惩罚、容量和训练外风险。
