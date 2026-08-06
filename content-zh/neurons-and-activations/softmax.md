---
title: "Softmax：让一组 logits 变成相互竞争的概率"
tags: ["why-models-learn"]
---

Softmax 把一组实数分数映射成总和为 $1$ 的概率向量。它不是把同一个标量函数逐个复制到每个坐标，而是让每个坐标共享同一个归一化分母：一个 logit 变大，通常会挤压其他坐标的概率。本文把 softmax 放回它真正的数学对象——概率单纯形上的向量函数——依次推导它的平移不变性、Jacobian、交叉熵梯度和数值稳定实现，再检查温度、归一化轴与 mask 如何改变它的语义。

![Softmax 从 logits 经过稳定归一化得到概率单纯形上的向量，并在轴与 mask 处接受检查](/assets/neurons-and-activations/svg/softmax.1.svg)

## 它处理的不是一个数，而是一条概率向量

设沿某个明确轴有 $K$ 个 logits：

$$
\boldsymbol z=(z_1,\ldots,z_K)^{\mathsf T}\in\mathbb R^K.
$$

Softmax 的第 $k$ 个输出是

$$
p_k=\operatorname{softmax}(\boldsymbol z)_k
=\frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}.
$$

由于每个有限的指数都为正，

$$
p_k>0,
\qquad
\sum_{k=1}^{K}p_k=1.
$$

因此 $\boldsymbol p$ 落在 $K$ 维概率单纯形的内部：

$$
\Delta^{K-1}
=\left\{\boldsymbol p\in\mathbb R^K:
p_k>0,\ \sum_{k=1}^{K}p_k=1\right\}.
$$

“内部”很重要：有限 logits 的 softmax 不会产生精确的 $0$ 或 $1$。极小概率只能无限接近 $0$；精确零通常来自显式 mask、稀疏替代函数，或低精度下的下溢。logits、概率和决策是三个不同层次：

| 对象 | 数值范围或约束 | 它回答的问题 |
| --- | --- | --- |
| logits $\boldsymbol z$ | 每个坐标可为任意实数 | 模型给出的相对分数是什么 |
| 概率 $\boldsymbol p$ | $p_k>0$ 且总和为 $1$ | 每个互斥类别分到多少质量 |
| 决策 $\widehat y$ | 通常是一个类别或行动 | 在代价规则下应该做什么 |

不能把 logits 直接当概率，也不能把取最大类别误认为 softmax 的训练目标。argmax 只保留排序，概率还包含置信度、交叉熵、校准和代价敏感决策需要的信息。

### 一个三分类数字例子

取

$$
\boldsymbol z=(2,0,-1)^{\mathsf T},
\qquad
m=\max_k z_k=2.
$$

先减去最大值，得到稳定计算时的中间量。各坐标如下：

| 类别 $k$ | $z_k$ | $z_k-m$ | $\exp(z_k-m)$ | $p_k$ |
| ---: | ---: | ---: | ---: | ---: |
| 1 | $2$ | $0$ | $1$ | $0.843795$ |
| 2 | $0$ | $-2$ | $0.135335$ | $0.114195$ |
| 3 | $-1$ | $-3$ | $0.049787$ | $0.042010$ |

所以

$$
\boldsymbol p
\approx(0.843794734481,\ 0.114195199385,\ 0.042010066134)^{\mathsf T},
\qquad
\sum_kp_k=1.
$$

最大 logit 对应最大概率，但第二类和第三类仍然保留了不同的概率质量。只报告第一类会丢掉这部分信息。

## 平移不变性让 logit 只有相对意义

对所有坐标同时加上常数 $c$：

$$
\begin{aligned}
\operatorname{softmax}(\boldsymbol z+c\boldsymbol 1)_k
&=\frac{\exp(z_k+c)}{\sum_j\exp(z_j+c)}\\
&=\frac{\exp(c)\exp(z_k)}
{\exp(c)\sum_j\exp(z_j)}
=p_k.
\end{aligned}
$$

因此 $\boldsymbol z$ 和 $\boldsymbol z+c\boldsymbol 1$ 表示完全相同的概率向量。实现中选择 $c=-m$，就是利用这个不变性把最大的指数变成 $1$。

两个类别的概率比值也只依赖 logit 间隔：

$$
\frac{p_a}{p_b}
=\frac{\exp(z_a)}{\exp(z_b)}
=\exp(z_a-z_b).
$$

所以把所有 logits 同时加大 $100$ 不会让模型更“自信”；拉大正确类别与其他类别之间的差距才会改变分布。由于指数严格递增，

$$
\arg\max_kp_k=\arg\max_kz_k.
$$

这只证明 top-1 决策可以直接比较 logits，不表示训练和概率应用都可以跳过 softmax。交叉熵、熵、top-k 概率、拒绝阈值和期望代价都需要归一化后的结果或等价的 log-probability。

## 它不是逐分量激活

逐分量激活满足 $a_k=f(z_k)$，改变 $z_j$ 不会直接影响 $a_k$（当 $j\neq k$ 时）。Softmax 的分母包含所有坐标，因此一次局部改变会沿整条概率向量传播。令

$$
S=\sum_{\ell=1}^{K}\exp(z_\ell),
\qquad
p_k=\frac{\exp(z_k)}{S}.
$$

对 $z_j$ 求导：

$$
\begin{aligned}
\frac{\partial p_k}{\partial z_j}
&=\frac{\delta_{kj}\exp(z_k)S-\exp(z_k)\exp(z_j)}{S^2}\\
&=p_k\left(\delta_{kj}-p_j\right),
\end{aligned}
$$

其中 $\delta_{kj}$ 是 Kronecker delta。于是同一坐标与不同坐标分别为

$$
\frac{\partial p_k}{\partial z_k}=p_k(1-p_k),
\qquad
\frac{\partial p_k}{\partial z_j}=-p_kp_j\quad(j\neq k).
$$

把全部偏导组织成矩阵：

$$
J_{\mathrm{softmax}}
=\frac{\partial\boldsymbol p}{\partial\boldsymbol z}
=\operatorname{diag}(\boldsymbol p)
-\boldsymbol p\boldsymbol p^{\mathsf T}.
$$

对前面的三分类例子，数值上

$$
J_{\mathrm{softmax}}
\approx
\begin{pmatrix}
0.131805&-0.096357&-0.035448\\
-0.096357&0.101155&-0.004797\\
-0.035448&-0.004797&0.040245
\end{pmatrix}.
$$

负的非对角项就是“竞争”的局部证据：提高一个类别的 logit，会压低其他类别的概率。矩阵的每一行和每一列都为零：

$$
J_{\mathrm{softmax}}\boldsymbol 1=\boldsymbol 0,
\qquad
\boldsymbol 1^{\mathsf T}J_{\mathrm{softmax}}=\boldsymbol 0^{\mathsf T}.
$$

输出只能在总和保持为 $1$ 的切平面内移动。进一步地，对任意向量 $\boldsymbol v$：

$$
\begin{aligned}
\boldsymbol v^{\mathsf T}J_{\mathrm{softmax}}\boldsymbol v
&=\sum_kp_kv_k^2-\left(\sum_kp_kv_k\right)^2\\
&=\operatorname{Var}_{K\sim\boldsymbol p}(v_K)\geq0.
\end{aligned}
$$

所以 Jacobian 半正定，但沿着全体 logits 同时平移的方向有零曲率。这不是逐元素导数表能表达的结构；若把 softmax 错当成 $K$ 个互不相干的标量函数，反向传播就会漏掉归一化分母的影响。

## 交叉熵把 Jacobian 压缩成 $p-y$

设目标向量 $\boldsymbol y$ 满足

$$
y_k\geq0,
\qquad
\sum_{k=1}^{K}y_k=1.
$$

它可以是 one-hot 标签，也可以是标签平滑后的软目标。以 softmax 概率计算交叉熵：

$$
\mathcal L(\boldsymbol y,\boldsymbol z)
=-\sum_{k=1}^{K}y_k\log p_k.
$$

把 $\log p_k=z_k-\operatorname{LSE}(\boldsymbol z)$ 代入，其中

$$
\operatorname{LSE}(\boldsymbol z)
=\log\sum_{j=1}^{K}\exp(z_j),
$$

得到

$$
\mathcal L
=\operatorname{LSE}(\boldsymbol z)
-\sum_{k=1}^{K}y_kz_k.
$$

对 log-sum-exp 求导：

$$
\frac{\partial\operatorname{LSE}(\boldsymbol z)}
{\partial z_k}
=\frac{\exp(z_k)}{\sum_j\exp(z_j)}
=p_k.
$$

所以

$$
\frac{\partial\mathcal L}{\partial z_k}
=p_k-y_k.
$$

这不是一个需要背下来的孤立经验式，而是 softmax 的归一化与对数损失恰好配对后的结果。梯度坐标之和为

$$
\sum_k\frac{\partial\mathcal L}{\partial z_k}
=\sum_kp_k-\sum_ky_k=0,
$$

与平移不变性一致：沿 $\boldsymbol 1$ 方向移动 logits 不会改变损失。

### 用数字看一次更新

对 $\boldsymbol z=(2,0,-1)$，假设第一类是真实类别：

$$
\boldsymbol y=(1,0,0)^{\mathsf T}.
$$

于是

$$
\mathcal L=-\log p_1\approx0.169846019556,
$$

$$
\boldsymbol g=\nabla_{\boldsymbol z}\mathcal L
\approx(-0.156205265519,\ 0.114195199385,\ 0.042010066134)^{\mathsf T}.
$$

若只为了观察方向，取步长 $\eta=0.1$ 做一次 logits 梯度下降：

$$
\boldsymbol z_{\mathrm{new}}
=\boldsymbol z-\eta\boldsymbol g
\approx(2.015621,\ -0.011420,\ -1.004201)^{\mathsf T}.
$$

重新计算得到

$$
\boldsymbol p_{\mathrm{new}}
\approx(0.847074,\ 0.111581,\ 0.041346)^{\mathsf T},
\qquad
\mathcal L_{\mathrm{new}}\approx0.165968.
$$

真实类别的 logit 上升，错误类别的 logits 下降，损失降低。真实训练中更新的是产生 logits 的权重和表示；输入特征会通过矩阵乘法继续出现在参数梯度中。

## 先减最大值，再计算指数

直接计算 $\exp(z_k)$ 会遇到上溢和下溢：

* 很大的正 logit 可能让指数超出当前 dtype 的可表示范围；
* 很小的负 logit 可能下溢成 $0$，让尾部概率和相应的对数损失信息消失；
* 先算概率再取对数还会把极小概率变成 $\log 0$。

令

$$
m=\max_jz_j.
$$

利用平移不变性，稳定的 softmax 为

$$
p_k
=\frac{\exp(z_k-m)}{\sum_j\exp(z_j-m)}.
$$

此时每个指数都不大于 $1$。稳定的 log-sum-exp 写成

$$
\operatorname{LSE}(\boldsymbol z)
=m+\log\sum_j\exp(z_j-m),
$$

稳定的 log-softmax 为

$$
\log p_k
=z_k-m-\log\sum_j\exp(z_j-m).
$$

例如 $\boldsymbol z=(1000,999,998)$ 与 $(2,1,0)$ 相差一个常数，所以概率相同；稳定计算的指数只需要处理 $1,\exp(-1),\exp(-2)$，而不是 $\exp(1000)$。对应的 log-sum-exp 约为

$$
\operatorname{LSE}(1000,999,998)
\approx1000.407605964444.
$$

训练接口若提供接收 logits 的交叉熵，应直接使用它。它可以在同一计算图里完成稳定的 log-sum-exp 与真实类别项，避免“先 softmax，再把概率传给仍然期待 logits 的损失”造成重复归一化。对只需要 top-1 的推理路径，直接取 $\arg\max\boldsymbol z$ 可以节省概率计算；但这不是交叉熵训练或概率校准的替代品。

| 目标 | 推荐计算 | 不要默认做什么 |
| --- | --- | --- |
| top-1 类别 | 直接比较 logits | 为了 argmax 先显式 softmax |
| 概率向量 | 稳定 softmax，沿正确轴减最大值 | 直接计算未平移的指数 |
| 对数概率 | 稳定 log-softmax 或 fused kernel | 先算极小概率再取对数 |
| 分类损失 | logits 加稳定交叉熵 | softmax 后再次交给 logits 损失 |

## 二分类是 softmax 的一个特例

对两个 logits $a,b$：

$$
\operatorname{softmax}(a,b)_1
=\frac{e^a}{e^a+e^b}
=\frac{1}{1+e^{-(a-b)}}
=\sigma(a-b).
$$

所以二分类 softmax 只依赖两个分数的差，等价于对 logit 差使用 sigmoid。二者并不是“一个逐元素、一个向量化”这么简单的实现差异：二分类用一个差值 logit 和一个 Bernoulli 损失，通常可以让接口更小；多分类需要在所有类别之间共享分母。

标签语义决定是否应该使用 softmax：

| 任务 | 每个样本的标签关系 | 输出约束 | 常见损失 |
| --- | --- | --- | --- |
| 互斥多分类 | 恰好一个类别 | 所有类别概率和为 $1$ | softmax 交叉熵 |
| 多标签分类 | 多个标签可同时成立 | 每个标签独立在 $(0,1)$ | 多个 BCE with logits |
| 多个独立二分类 | 标签彼此不构成一个分布 | 每个输出独立归一化 | 独立 sigmoid 与 BCE |

多标签问题若强行使用 softmax，会让一个标签变大时必然压低其他同时为真的标签；这不是优化器的问题，而是输出空间从一开始就写错了。

## 归一化轴是模型语义的一部分

实际张量通常有 batch、head、时间和类别等多个轴。Softmax 只应沿着代表“竞争集合”的那个轴计算；在哪个轴求和，决定了概率分布属于谁。

| 张量场景 | 典型形状 | 应归一化的轴 | 应满足的检查 |
| --- | --- | --- | --- |
| 分类输出 | $[\text{batch},\text{class}]$ | class | 每个样本的类别和为 $1$ |
| 多头注意力权重 | $[\text{batch},\text{head},\text{query},\text{key}]$ | key | 每个 query 的 key 权重和为 $1$ |
| 词表预测 | $[\text{batch},\text{time},\text{vocab}]$ | vocab | 每个时间位置的词表和为 $1$ |

最常见的轴错误是把 batch 轴或 hidden 轴当成类别轴：单个样本内部的概率和看似不为零，但真正应该竞争的集合没有归一化。实现审计时不要只看 API 参数名，应对一个小张量逐轴打印和验证和。

## 温度改变尖锐程度，也改变梯度尺度

温度 $T>0$ 定义为

$$
p_k(T)
=\frac{\exp(z_k/T)}{\sum_j\exp(z_j/T)}
=\operatorname{softmax}\left(\frac{\boldsymbol z}{T}\right)_k.
$$

对 $\boldsymbol z=(2,0,-1)$：

| 温度 $T$ | $p_1(T)$ | $p_2(T)$ | $p_3(T)$ | 形状 |
| ---: | ---: | ---: | ---: | --- |
| $0.5$ | $0.979629$ | $0.017943$ | $0.002428$ | 更尖 |
| $1$ | $0.843795$ | $0.114195$ | $0.042010$ | 原始 |
| $2$ | $0.628532$ | $0.231224$ | $0.140244$ | 更平 |

因为 $T$ 为正，温度不改变 argmax；当 $T$ 趋近于 $0$ 时，质量集中到最大 logit（若有并列则在并列者间分配），当 $T$ 变大时趋近均匀分布。对 logits 的梯度为

$$
\frac{\partial\boldsymbol p(T)}{\partial\boldsymbol z}
=\frac1T
\left(\operatorname{diag}(\boldsymbol p(T))
-\boldsymbol p(T)\boldsymbol p(T)^{\mathsf T}\right).
$$

因此温度不只是改变图形的“尖或平”，还改变局部梯度尺度。训练中的温度、蒸馏中的教师温度和部署时的 temperature scaling 必须记录清楚；在验证集上拟合的温度也不能反复用测试集调参。温度缩放只能修正概率的尖锐程度，不能修复类别排序错误或标签定义错误。

## Masked softmax 必须先屏蔽，再归一化

如果一部分位置不允许参与竞争，先构造允许集合 $A$：

$$
p_k=
\begin{cases}
\displaystyle\frac{\exp(z_k)}
{\sum_{j\in A}\exp(z_j)},&k\in A,\\[8pt]
0,&k\notin A.
\end{cases}
$$

工程上常把不允许的 logit 替换成 $-\infty$，再沿同一轴做稳定 softmax。对 $\boldsymbol z=(2,0,-1)$，屏蔽第二类后：

$$
\boldsymbol p_{\mathrm{mask}}
\approx(0.952574126822,\ 0,\ 0.047425873178)^{\mathsf T}.
$$

这与先得到完整概率、再把第二类清零而不重新归一化不同；后者的总和小于 $1$，已经不是允许集合上的概率分布。即使实现选择乘 mask 后再归一化，也要明确处理空集合和 dtype 误差，优先使用在 logits 阶段完成 mask 的 fused kernel。

全 mask 行是一个单独的无效输入：若所有位置都是 $-\infty$，最大值也是 $-\infty$，做 $z-m$ 会得到未定义的 $-\infty-(-\infty)$，随后可能出现 NaN。调用方必须预先保证每行至少有一个有效位置，或显式选择“报错、跳过该行、返回全零并在后续不计入损失”等策略；不能把 NaN 当作 softmax 的正常概率。

## 用不变量和小张量核验实现

一个 softmax 实现的正确性不应只靠一次端到端训练。可以在同一 dtype 和真实调用路径上逐层检查：

| 检查 | 预期结果 | 能抓住的错误 |
| --- | --- | --- |
| 概率总和 | 每个竞争集合约为 $1$ | 归一化轴、mask 或广播错误 |
| 非负与有限 | 允许位置为正且无 NaN、Inf | 溢出、下溢和无效空行 |
| 平移不变性 | $\operatorname{softmax}(\boldsymbol z)=\operatorname{softmax}(\boldsymbol z+c\boldsymbol1)$ | 未稳定化或偏置广播错误 |
| 排序一致性 | 概率最大坐标等于 logit 最大坐标 | 符号、轴或温度实现错误 |
| Jacobian 差分 | 数值差分接近 $p_k(\delta_{kj}-p_j)$ | 把向量函数错写成逐元素函数 |
| 交叉熵梯度 | logits 梯度接近 $\boldsymbol p-\boldsymbol y$ | 重复 softmax、标签索引或 reduction 错误 |
| mask 结果 | 禁止位置为零，有效集合和为 $1$ | mask 应用时机和空行策略错误 |

### 一个最小的验证协议

1. 用包含正负、重复和极大间隔的固定 logits，例如 $(2,0,-1)$、$(1000,999,998)$ 和 $(1000,0,-1000)$；
2. 分别在 float32 与部署 dtype 中检查有限性、总和和最大坐标；
3. 对所有 logits 加上 $10^3$ 或减去 $10^3$，比较概率与 log-probability；
4. 用中心差分扰动一个坐标，对照解析 Jacobian 的对应列；
5. 用 one-hot 与软标签分别对照 $p-y$，并确认 batch reduction 与文档约定一致；
6. 为每个真实张量轴构造一个小例子，确认只有语义上的竞争集合总和为 $1$；
7. 加入部分 mask 和全 mask 行，确认禁止位置、空行策略和反向梯度都符合调用方约定；
8. 最后比较独立实现、框架 fused kernel 与导出/量化模型的前向误差和 loss 误差。

对于概率输出，还应在冻结模型后检查校准、熵和长尾类别的 log loss。总和为 $1$ 只说明它是一个归一化向量，不说明它的 $0.9$ 真正代表约九成命中率。

## 失效模式

**把 softmax 当成逐元素函数。** 分母耦合所有坐标；只保留 $p_k(1-p_k)$ 会漏掉其他 logit 对它的影响。

**把 logits 当概率。** logits 可以为负、总和也没有约束；概率语义要由稳定 softmax 或等价 log-softmax 提供。

**先 softmax 再交给 logits 损失。** 这会重复归一化，并绕开损失接口本来可以使用的稳定 log-sum-exp。

**沿错误轴归一化。** batch、head、query、key 和 class 的轴名称不等于语义；必须验证正确竞争集合的和。

**直接计算极大的指数。** 先减最大 logit，计算对数概率时优先使用稳定 log-softmax；不要把 log(softmax(z)) 当成唯一实现。

**多标签任务使用 softmax。** softmax 强迫类别互斥；标签可以同时为 $1$ 时，应使用独立 sigmoid 或其他匹配的联合模型。

**在 softmax 后才清零 mask。** 不重新归一化会破坏总和；全 mask 行还会让稳定化步骤产生 NaN。

**把温度当作无害显示参数。** 温度会改变概率熵和梯度尺度；训练、蒸馏、校准和部署的温度必须进入实验记录。

**把归一化当成校准。** 概率和为 $1$ 是代数约束，不是泛化保证；需要独立数据上的 log loss、可靠性图和分组检查。

## 相关词条

- [激活函数选择](../neurons-and-activations/choosing-activations/)：把 softmax 放回互斥多分类的输出头，而不是把它当作隐藏层标量激活。
- [Softmax 回归](../linear-models/softmax-regression/)：把同一个概率映射接在线性 logits 上，形成多分类线性模型。
- [交叉熵](../information-theory/cross-entropy/)：从编码代价和负对数似然推导 softmax 交叉熵与 log-sum-exp。
- [逐元素导数](../calculus/elementwise-derivatives/)：区分逐分量函数与 softmax 这类坐标耦合的向量函数。
- [广播与归约求导](../calculus/broadcast-and-reduction-derivatives/)：理解共享分母、归约轴和反向广播为什么会耦合坐标。
- [离散分布](../probability/discrete-distributions/)：解释概率质量、互斥类别和单纯形约束。
- [分类指标](../evaluation-and-generalization/classification-metrics/)：把概率、排序、硬决策和校准分开评估。
- [熵](../information-theory/entropy/)：用预测分布的不确定性衡量 softmax 输出的尖锐程度。
