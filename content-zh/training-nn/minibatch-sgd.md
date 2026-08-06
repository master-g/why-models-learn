---
title: "小批量随机梯度下降：用抽样换取更新频率"
tags: ["why-models-learn"]
---

小批量随机梯度下降把完整数据集的梯度换成一小组样本的平均梯度。它的价值不是把同一个公式写短，而是用可控的噪声换取更低的单步计算成本、更高的参数更新频率和更好的硬件利用率。批次大小、抽样方式、归约分母、学习率和每个 epoch 的更新次数必须一起记录；否则“改了 batch size”实际上可能同时改了目标尺度、噪声水平和训练预算。

![小批量随机梯度下降在完整梯度附近采样：左侧展示不同批次的梯度估计，右侧比较小批次噪声、较大批次和全批量方向](/assets/training-nn/svg/minibatch-sgd.1.svg)

## 从经验风险拆出样本梯度

有 $N$ 个训练样本时，逐样本损失记作

$$
\ell_i(\boldsymbol\theta)
=\ell\bigl(f_{\boldsymbol\theta}(\boldsymbol x_i),y_i\bigr).
$$

全数据集的经验风险是

$$
F(\boldsymbol\theta)
=\frac1N\sum_{i=1}^{N}\ell_i(\boldsymbol\theta),
$$

梯度为

$$
\nabla F(\boldsymbol\theta)
=\frac1N\sum_{i=1}^{N}
\boldsymbol g_i(\boldsymbol\theta),
\qquad
\boldsymbol g_i(\boldsymbol\theta)
=\nabla\ell_i(\boldsymbol\theta).
$$

全批量梯度下降每次都计算这 $N$ 个样本梯度，再用它们的平均值更新参数。小批量方法从索引集合 $\mathcal B$ 中取出 $B$ 个样本，使用

$$
\widehat{\boldsymbol g}_{\mathcal B}(\boldsymbol\theta)
=\frac1B\sum_{i\in\mathcal B}
\boldsymbol g_i(\boldsymbol\theta).
$$

更新式仍然是

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k
-\eta_k\widehat{\boldsymbol g}_{\mathcal B_k}(\boldsymbol\theta_k).
$$

改变的是梯度来源，不是负梯度更新的定义。一次更新只看到 $\mathcal B_k$，所以它可能没有让完整目标 $F$ 下降；这是抽样噪声的正常结果，不等于每次上升都是实现错误。

### 为什么不直接取一个样本

$B=1$ 的随机梯度单步最便宜，更新频率最高，但每个样本的梯度可能与总体方向差异很大。$B=N$ 的全批量梯度没有抽样噪声，却必须在每次更新前处理全部数据。中间的 batch size 是三个量之间的折中：

| 批次大小 | 单步梯度 | 单步成本 | 每个 epoch 的更新次数 |
| --- | --- | --- | --- |
| $1$ | 波动最大 | 最低 | 最高 |
| $1<B<N$ | 有限方差的平均 | 中等 | 中等 |
| $N$ | 确定的完整平均 | 最高 | $1$ |

这里“每个 epoch 的更新次数”假设每个样本只访问一次且不丢弃尾批次。实际训练还要说明是否 drop-last、是否重复采样以及 batch 是否跨设备合并。

## 无偏性来自抽样口径

在固定参数 $\boldsymbol\theta$ 处，假设每个 batch 由有放回的均匀抽样得到，每个被抽到的索引 $I_j$ 都满足

$$
\mathbb P(I_j=i)=\frac1N.
$$

一个随机样本梯度的条件期望是

$$
\begin{aligned}
\mathbb E[
\boldsymbol g_{I_j}(\boldsymbol\theta)
\mid\boldsymbol\theta]
&=\sum_{i=1}^{N}
\frac1N\boldsymbol g_i(\boldsymbol\theta)\\
&=\nabla F(\boldsymbol\theta).
\end{aligned}
$$

对 $B$ 个样本取平均：

$$
\begin{aligned}
\mathbb E[
\widehat{\boldsymbol g}_{\mathcal B}(\boldsymbol\theta)
\mid\boldsymbol\theta]
&=\frac1B\sum_{j=1}^{B}
\mathbb E[
\boldsymbol g_{I_j}(\boldsymbol\theta)
\mid\boldsymbol\theta]\\
&=\nabla F(\boldsymbol\theta).
\end{aligned}
$$

因此小批量梯度是完整梯度的无偏估计量。无偏只描述许多次抽样的平均，不保证某一次 batch 恰好等于完整梯度，也不保证这一次更新后的完整 loss 下降。

如果采样概率不是均匀的，直接平均通常会偏向高概率样本。设样本 $i$ 以概率 $q_i$ 被抽到，可以用重要性权重

$$
\widehat{\boldsymbol g}
=\frac1B\sum_{j=1}^{B}
\frac{1}{Nq_{I_j}}
\boldsymbol g_{I_j}
$$

恢复均匀经验风险的期望。但权重会增大方差，实际系统需要在采样效率、权重稳定性和目标分布之间作出明确选择。类别平衡采样也是同样的边界：它可能改善少数类看到的频率，却不再是未经修正的经验风险。

### 不放回的一个 epoch 仍可做逐批次估计

实际训练常在一个 epoch 开始时打乱 $1,\ldots,N$，再把排列切成连续 batch。单个 batch 是不放回抽样，和有放回抽样不同，但若排列是均匀随机的，每个位置的边缘分布仍然是均匀的：

$$
\mathbb E[
\widehat{\boldsymbol g}_{\mathcal B}
\mid\boldsymbol\theta]
=\nabla F(\boldsymbol\theta).
$$

同一个 epoch 中不同 batch 之间不独立，因为它们不能重复使用同一个索引；这会影响方差和序列相关性，却不自动破坏单个 batch 的无偏性。分析“每一步噪声”时要说清是有放回随机 batch，还是一个打乱排列切片。

如果始终按原始数据顺序取 batch，估计量可能仍然无偏于某些数据集，但相邻更新会继承数据排序的结构。例如按类别排序的数据会让前几个 batch 几乎只包含一个类别，训练初期的梯度方向会被系统性地分段。shuffle 是训练协议的一部分，不只是性能优化。

## 方差告诉我们 batch size 买到了什么

令单样本梯度在当前参数处的均值为

$$
\boldsymbol\mu
=\nabla F(\boldsymbol\theta),
$$

噪声为

$$
\boldsymbol\xi_i
=\boldsymbol g_i-\boldsymbol\mu.
$$

有放回独立抽样时：

$$
\mathbb E[\boldsymbol\xi_i]=\boldsymbol0.
$$

如果单样本噪声的二阶大小为

$$
\mathbb E[
\lVert\boldsymbol\xi_i\rVert_2^2]
=\sigma^2,
$$

batch 平均的噪声是

$$
\overline{\boldsymbol\xi}_{\mathcal B}
=\frac1B\sum_{j=1}^{B}\boldsymbol\xi_{I_j}.
$$

独立性让交叉项的期望为零：

$$
\begin{aligned}
\mathbb E[
\lVert\overline{\boldsymbol\xi}_{\mathcal B}\rVert_2^2]
&=\frac1{B^2}
\sum_{j=1}^{B}
\mathbb E[
\lVert\boldsymbol\xi_{I_j}\rVert_2^2]\\
&=\frac{\sigma^2}{B}.
\end{aligned}
$$

所以 batch 从 $B$ 增大到 $4B$ 时，噪声方差变成原来的四分之一，噪声标准差变成原来的一半。更大的 batch 会让方向更稳定，却不会免费得到四倍的更新次数。

### 不放回抽样有有限总体修正

若从有限总体中不放回抽取 $B$ 个样本，定义单坐标梯度的总体均值和样本方差：

$$
\bar g=\frac1N\sum_{i=1}^{N}g_i,
\qquad
S^2=\frac1{N-1}\sum_{i=1}^{N}(g_i-\bar g)^2.
$$

简单随机抽样的 batch 均值方差是

$$
\operatorname{Var}(\widehat g_{\mathcal B})
=\frac1B
\left(1-\frac BN\right)S^2.
$$

括号中的

$$
1-\frac BN
$$

是有限总体修正。$B$ 接近 $N$ 时，同一个 epoch 中抽到的 batch 均值自然更稳定；当 $B\ll N$ 时，它接近有放回抽样的 $S^2/B$ 形式。这个公式使用的是 $1/(N-1)$ 定义的 $S^2$；若换成 $1/N$ 定义的总体方差，系数也要一起换。

梯度噪声不是一个固定常数。参数接近不同区域时，样本之间的梯度差异可能变大或变小；类别不平衡、标签噪声、数据增强和损失权重都会改变 $\sigma^2$。因此不能只根据 batch size 推断训练一定更稳定，最好在日志中测量 batch 梯度范数或多个 batch 的方差。

## 一个 epoch 有多少次更新

若数据集大小为 $N$、batch size 为 $B$，且不重复采样、不丢弃尾部，更新次数是

$$
U_{\mathrm{epoch}}
=\left\lceil\frac NB\right\rceil.
$$

如果设置 drop-last，则是

$$
U_{\mathrm{epoch}}
=\left\lfloor\frac NB\right\rfloor,
$$

最后不足 $B$ 的样本不会被看到。把 batch 从 $32$ 改成 $64$，但仍训练相同 epoch 数，会让每个 epoch 的更新次数大约减半；如果又把学习率、warmup 步数和总 epoch 保持不变，实验同时改变了更新预算。

因此训练预算至少有两种口径：

| 预算口径 | 定义 | 适合比较 |
| --- | --- | --- |
| epoch | 完整数据集被访问的次数 | 数据遍历和增强次数 |
| optimizer step | 参数实际更新次数 | 调度器、warmup 和 checkpoint |
| sample tokens seen | 已处理样本或 token 数 | 不同 batch size 的计算量 |
| wall-clock time | 实际运行时间 | 系统吞吐和部署约束 |

全批量每 epoch 更新一次，小批量每 epoch 更新很多次；如果只按 epoch 画曲线，横轴可能掩盖了更新频率的变化。报告 batch-size 实验时应同时报告总样本访问量和总 optimizer step。

### 尾批次不是无关紧要的实现细节

若 $N$ 不能被 $B$ 整除，最后一个 batch 的实际大小小于 $B$。如果损失已经在 batch 内取 mean，正确的 epoch 目标应按样本数重新加权：

$$
F_{\mathrm{epoch}}
=\frac{
\sum_{k}
B_k L_k
}{\sum_kB_k},
$$

其中 $B_k$ 是第 $k$ 个 batch 的有效样本数。直接对 batch mean 再取平均：

$$
\frac1K\sum_{k=1}^{K}L_k
$$

会让小尾批次和完整批次拥有相同的权重。除非每个 batch 大小相同，否则这两个量不相等。

如果使用加权样本或 mask，分母应替换成有效权重和：

$$
F_{\mathrm{epoch}}
=\frac{\sum_iw_i\ell_i}{\sum_iw_i},
$$

不能只按 batch 数平均日志里的标量。

## 梯度累积模拟更大的有效 batch

显存不够容纳目标 batch 时，可以把若干 micro-batch 的梯度累积后再更新。设每个 micro-batch 的有效样本数为 $b_r$，想得到一个大 batch 的平均梯度，应使用

$$
\widehat{\boldsymbol g}_{\mathrm{large}}
=\frac{
\sum_{r=1}^{R}b_r\widehat{\boldsymbol g}_r
}{
\sum_{r=1}^{R}b_r
}.
$$

若所有 micro-batch 大小都相同，则是它们梯度的简单平均。若最后一个 micro-batch 较小，不能无条件对每个 mean 梯度等权相加。

等价的实现方式有两种：

1. 每个 micro-batch 计算 sum loss，累加 $R$ 次后除以总有效样本数；
2. 每个 micro-batch 计算 mean loss，按 $b_r$ 加权后累加。

如果每个 micro-batch 的 mean 梯度直接相加却不除以 $R$，有效梯度会放大约 $R$ 倍；如果在累积过程中每次都更新参数，就不是大 batch，而是 $R$ 次小 batch 更新。

梯度累积还改变了 dropout、数据增强和 batch normalization 的语义。累积多个前向不会自动等价于一次真正的大 batch 前向：随机层可能每个 micro-batch 使用不同掩码，batch 统计量也只看到了局部样本。要宣称“等价大 batch”，必须说明哪些状态跨 micro-batch 共享。

## shuffle、采样和类别比例

一个常见的 epoch 流程是：

1. 只对训练集索引生成随机排列；
2. 按 batch size 切分排列；
3. 每个 batch 前向、反向并更新；
4. 记录有效样本数、loss 分子和分母；
5. 进入下一 epoch 时重新排列，并记录随机种子或随机状态。

验证集和测试集通常不参与这种随机重排后的训练更新，也不应因为训练 loader 的 shuffle 设置而改变评估样本的权重。评估可以按固定顺序读取，关键是每个样本只按预定口径计入一次。

### 类别平衡 batch 不是普通 batch

若训练集类别比例极不平衡，可以让每个 batch 包含更均衡的类别。这会降低少数类梯度的抽样等待时间，却也改变了一个 batch 代表的经验分布。此时有三种不同目标不要混在一起：

| 方式 | batch 看到的分布 | 目标含义 |
| --- | --- | --- |
| 均匀按样本采样 | 接近训练集先验 | 估计原始经验风险 |
| 类别平衡采样 | 类别先验被改写 | 更频繁训练少数类，需考虑先验修正 |
| 均匀采样加类别权重 | 采样分布不变 | 用损失显式改变错误代价 |

采样概率 $q_i$ 与原始经验风险的关系可以写成

$$
\mathbb E_{i\sim q}
\left[
\frac{1}{Nq_i}\ell_i
\right]
=\frac1N\sum_i\ell_i.
$$

但重要性权重可能带来高方差，权重裁剪又会引入偏差。工程上常见的 balanced sampler、class weight 和 threshold adjustment 分别作用于抽样、训练目标和部署决策，不能只用一个“少数类 recall 变高”来证明它们等价。

### 数据顺序和相关样本

随机排列主要打散相邻样本之间的相关性。若数据按用户、视频、时间窗口或同一设备聚簇，逐样本随机拆 batch 可能仍把高度相似样本放在同一批次，导致梯度方差被低估。更重要的是，不能先把同一主体的近重复样本拆到训练和验证，再用 shuffle 掩盖数据泄漏。

时序数据通常不能任意打乱未来和过去；这时要把“训练内 batch 的随机性”和“时间顺序约束”同时写进采样器。随机梯度理论中的均匀抽样假设不能自动覆盖所有时间序列训练协议。

## batch size 与学习率不是一个旋钮

增大 batch size 通常降低单步梯度方差，但也会减少固定 epoch 内的参数更新次数。学习率是否应该同时增大，取决于优化目标、归约方式、总样本访问量和调度器约定。

有时会使用线性缩放启发式：

$$
\eta_{\mathrm{new}}
\approx
\eta_{\mathrm{old}}
\frac{B_{\mathrm{new}}}{B_{\mathrm{old}}},
$$

让每个样本访问量对应的平均更新幅度大致保持一致。但这不是普适定理：

- 损失是 mean 还是 sum 会先改变梯度尺度；
- batch 变大后，单位 epoch 的更新次数减少；
- 大 batch 的噪声更小，可能离开原本有益的探索区域；
- warmup、衰减、动量和权重衰减的状态都按 step 或 epoch 解释；
- GPU 吞吐、通信开销和内存约束可能决定实际可用的 batch。

比较两个 batch size 时，至少固定或明确下面四个轴：

| 比较轴 | 要记录的量 |
| --- | --- |
| 数据预算 | 已处理样本数、token 数或 epoch |
| 更新预算 | optimizer step、gradient accumulation 次数 |
| 目标尺度 | loss reduction、mask 分母、权重 |
| 调度状态 | 每 step 的学习率、warmup、衰减和随机状态 |

一个稳妥流程是先固定总样本访问量，分别对几个 batch size 做短训练，再独立扫描学习率。不要在同时改 batch、学习率、epoch 和增强概率后，把结果差异归因于某一个旋钮。

### 调度器到底按什么计数

如果学习率调度器每个 optimizer step 更新一次，则一个 epoch 的调度推进量取决于 batch size：

$$
\text{steps per epoch}
=\left\lceil\frac NB\right\rceil.
$$

如果调度器按 epoch 更新，它的时间轴又是另一种。梯度累积时，一个 optimizer step 可能包含 $R$ 个 micro-batch；若调度器错误地把 micro-batch 也当作 optimizer step，实际衰减速度会快 $R$ 倍。

日志中应同时输出 epoch、micro-batch index、optimizer step 和当前学习率。恢复 checkpoint 时，不能只恢复 epoch 而丢失已经累计的 step。

## 如何测量 batch 梯度的噪声

在一个固定参数快照 $\boldsymbol\theta$ 上，可以取多个 batch 梯度：

$$
\widehat{\boldsymbol g}^{(1)},
\ldots,
\widehat{\boldsymbol g}^{(M)}.
$$

它们的均值

$$
\overline{\boldsymbol g}
=\frac1M\sum_{r=1}^{M}
\widehat{\boldsymbol g}^{(r)}
$$

可以和全批量梯度比较：

$$
\delta_{\mathrm{bias}}
=\lVert\overline{\boldsymbol g}
-\nabla F(\boldsymbol\theta)\rVert_2.
$$

batch 均值的离散程度可以用

$$
\widehat{\sigma}^2
=\frac1{M-1}
\sum_{r=1}^{M}
\left\lVert
\widehat{\boldsymbol g}^{(r)}
-\overline{\boldsymbol g}
\right\rVert_2^2.
$$

这不是对真实总体方差的完美估计：batch 可能重叠，数据增强会使同一索引产生不同梯度，类别采样可能改变抽样分布。但它比只看一条 noisy loss 曲线更能说明“这个 batch size 的方向有多不稳定”。

测量时应固定参数，不要在模型已经更新后把不同参数点的梯度混在一起。也应同时记录各个坐标或参数组的范数；总范数相近不代表每一层的噪声都相近。

## 分布式训练的有效 batch

在数据并行中，每个设备先用本地 batch 计算梯度，再做 all-reduce。若有 $W$ 个设备，每个设备的有效样本数为 $B_{\mathrm{local}}$，且梯度按设备平均，则全局有效 batch 通常是

$$
B_{\mathrm{global}}
=W B_{\mathrm{local}}.
$$

再加入 $R$ 个 micro-batch 的梯度累积：

$$
B_{\mathrm{effective}}
=W B_{\mathrm{local}}R.
$$

这个公式依赖每个设备样本不重复、每个 micro-batch 按有效样本正确归约以及 all-reduce 的平均约定。若通信操作做的是 sum 而不是 mean，学习率和 loss 分母必须相应调整。

分布式复现还要记录：

- 每个 epoch 的全局 shuffle seed；
- sampler 是否给不同设备分配了互斥索引；
- 是否 drop-last 以保证设备之间 batch 对齐；
- 梯度通信发生在每个 micro-batch 还是累积结束；
- batch normalization 或其他跨样本状态如何同步。

只写“用了 8 张卡、batch size 32”不够；还要说明 32 是本地、全局还是每次 optimizer step 的有效值。

## 运行方法

下面用四个一维二次样本观察完整梯度、所有二样本 batch 的梯度分布以及一个固定 shuffle epoch：

$$
\ell_i(\theta)=\frac12(\theta-a_i)^2,
\qquad
(a_1,a_2,a_3,a_4)=(0,2,4,6).
$$

在 $\theta_0=1$ 处，完整目标的最优点是样本均值 $3$，完整梯度是 $1-3=-2$。

```python
from itertools import combinations


targets = [0.0, 2.0, 4.0, 6.0]
theta0 = 1.0


def sample_gradient(theta, target):
    return theta - target


def batch_gradient(theta, indices):
    values = [
        sample_gradient(theta, targets[index])
        for index in indices
    ]
    return sum(values) / len(values)


def rounded(values):
    return [round(value, 12) for value in values]


per_sample = [
    sample_gradient(theta0, target)
    for target in targets
]
full_gradient = sum(per_sample) / len(per_sample)
all_batch_means = [
    batch_gradient(theta0, indices)
    for indices in combinations(range(len(targets)), 2)
]
batch_mean = sum(all_batch_means) / len(all_batch_means)
batch_variance = sum(
    (value - batch_mean) ** 2
    for value in all_batch_means
) / len(all_batch_means)

eta = 0.25
order = [2, 0, 3, 1]
theta = theta0
epoch_gradients = []
for start in range(0, len(order), 2):
    indices = order[start:start + 2]
    gradient = batch_gradient(theta, indices)
    epoch_gradients.append(gradient)
    theta -= eta * gradient

full_step_theta = theta0 - eta * full_gradient
accumulated_gradient = (
    2.0 * batch_gradient(theta0, [0, 1])
    + 2.0 * batch_gradient(theta0, [2, 3])
) / 4.0

print(f"per-sample gradients={rounded(per_sample)}")
print(
    f"full-gradient={full_gradient:.6f} "
    f"population-optimum={sum(targets) / len(targets):.6f}"
)
print(
    f"all 2-sample batch means={rounded(all_batch_means)} "
    f"variance={batch_variance:.6f}"
)
print(
    f"shuffle order={order} "
    f"batch-gradients={rounded(epoch_gradients)} "
    f"theta-after-one-epoch={theta:.6f}"
)
print(f"full-batch one-step theta={full_step_theta:.6f}")
print(f"accumulated mean gradient={accumulated_gradient:.6f}")
```

输出为：

```text
per-sample gradients=[1.0, -1.0, -3.0, -5.0]
full-gradient=-2.000000 population-optimum=3.000000
all 2-sample batch means=[0.0, -1.0, -2.0, -2.0, -3.0, -4.0] variance=1.666667
shuffle order=[2, 0, 3, 1] batch-gradients=[-1.0, -2.75] theta-after-one-epoch=1.937500
full-batch one-step theta=1.500000
accumulated mean gradient=-2.000000
```

所有二样本 batch 的梯度均值是 $-2$，但单个 batch 可以给出 $0$ 或 $-4$。固定 shuffle 顺序后，一个 epoch 的两次参数更新得到 $1.9375$，不同于只用一次完整梯度更新的 $1.5$；两者都没有违反各自的更新定义。最后一行验证了两个等大 micro-batch 的梯度按样本数累积后恢复完整平均梯度。

这个实验还说明“一个 epoch 后参数相同”不是小批量与全批量等价的定义。小批量在中途已经更新参数，第二个 batch 看到的是新的 $\theta$；即使一个 epoch 中每个样本都恰好访问一次，路径也不同。

## 失效模式

**把一次 batch 梯度当成完整梯度。** 单个 batch 只是估计量，可能偏离总体方向。调试时固定参数，比较多个 batch 均值与全批量梯度。

**把 batch mean 再按 batch 数平均。** 当最后一个 batch 较小时，这会给尾批次过大的权重。按有效样本数或有效权重重新归约。

**shuffle 训练集、验证集和测试集使用同一套随机状态。** 训练采样器可以随机，评估应固定数据边界和计分口径；随机增强也不能泄漏到验证或测试。

**类别平衡采样却仍按原始先验解释概率。** 采样改变了训练看到的分布。要么做重要性修正，要么明确训练目标是代价敏感目标，并单独校准部署概率。

**把 batch size 加倍但保持所有 step 预算不变。** 这会同时改变每 epoch 的样本访问量、梯度噪声和更新次数。比较实验至少固定总样本访问量或明确新的预算。

**梯度累积忘记除以有效样本数。** 多个 micro-batch 的 mean 梯度直接求和会放大更新；尾 micro-batch 还需要按有效大小加权。

**把梯度累积当成真正大 batch 前向。** dropout、batch normalization、数据增强和随机采样的状态可能不同。需要说明“等价”只针对梯度平均，还是也针对前向统计。

**不同设备重复抽到同一个样本。** 这会降低实际有效 batch，改变梯度方差和数据预算。检查 sampler 的 rank、seed、drop-last 和全局索引覆盖。

**调度器按错了时间单位。** epoch、micro-batch 和 optimizer step 不是同一个计数。恢复训练时同时核对当前学习率和已推进的 step。

**把一次 noisy loss 上升当成训练失败。** 随机梯度单步不保证完整目标下降。看固定参数下的 batch 方差、滑动平均、验证目标和异常样本，而不是只看相邻两个 batch。

## 一个可复用的核验协议

实现或修改小批量训练时，可以按下面顺序检查：

1. 在固定参数上计算全批量梯度，作为基线；
2. 采集多个 batch 梯度，比较均值、方差和全批量差异；
3. 明确采样是有放回、不放回、按 epoch shuffle 还是类别平衡；
4. 检查 batch loss 的分母、尾批次和有效 mask；
5. 记录每个 epoch 的样本访问量、optimizer step 和 micro-batch 数；
6. 用两个等大 micro-batch 核对梯度累积后的平均尺度；
7. 改变 batch size 时单独记录学习率、warmup、衰减和总预算；
8. 固定随机种子、数据顺序和增强状态，确认小数据集可复现；
9. 分布式训练检查全局 batch、all-reduce 的 sum/mean 和索引不重叠；
10. 在验证集和固定的概率/决策指标上比较，而不是只比较带噪训练 loss。

小批量随机梯度下降的核心不是“batch 越小越快”或“batch 越大越稳定”，而是把计算、噪声、更新频率和数据分布放进同一个协议。只有这些量都写清楚，两个 batch size 的训练结果才有可比性。

## 相关词条

- [梯度下降](../training-nn/gradient-descent/)：理解负梯度更新、步长、目标尺度和全批量基线。
- [随机梯度下降理论](../optimization-theory/stochastic-gradient-descent-theory/)：推导无偏随机梯度、期望下降和噪声地板。
- [损失函数](../training-nn/loss-functions/)：确认 sum、mean、weight 和 mask 的目标归约。
- [分类损失](../training-nn/classification-losses/)：查看类别权重、标签结构和分类输出契约。
- [梯度检查](../backpropagation/gradient-checking/)：在固定 batch 和参数上核对解析梯度与有限差分。
- [批归一化](../training-nn/batch-normalization/)：理解 batch 统计量为何使梯度累积不自动等价大 batch。
- [学习率调度](../training-nn/learning-rate-schedules/)：按 step 或 epoch 管理 warmup 与衰减。
- [提前停止](../evaluation-and-generalization/early-stopping/)：用验证证据选择停止点和最佳 checkpoint。
