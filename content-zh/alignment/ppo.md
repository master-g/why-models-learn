---
title: "PPO：用概率比裁剪策略更新"
tags: ["why-models-learn"]
---

PPO（Proximal Policy Optimization，近端策略优化）在旧策略采样的数据上更新当前策略，并用新旧策略的概率比限制单批更新对目标的推动。它把策略梯度的 advantage、重要性采样比、clipped surrogate、value function 和 entropy regularization 组合成一个训练循环。PPO 的裁剪降低了局部更新的激进程度，但不提供全局 KL 上界，也不能修复 reward、value、mask 或 rollout 版本错误。本文从概率比开始推导 clipped objective，再核对 GAE、value clipping、熵正则、旧策略快照和语言模型 token 级实现。[策略梯度](../alignment/policy-gradient/)提供 log-derivative 基础；[RLHF 与 DPO](../alignment/rlhf-dpo/)说明 PPO 如何进入语言模型对齐流程。

![PPO 示意图：旧策略生成 rollout，优势估计和新旧概率比进入裁剪目标，更新后用 KL 与独立结果检查策略漂移](/assets/alignment/svg/ppo.1.svg)

## PPO 解决的是策略更新过大的问题

设策略为 $\pi_\theta(a\mid s)$，目标是最大化折扣回报：

$$
J(\theta)
  =\mathbb E_{\tau\sim\pi_\theta}
    \left[
      \sum_{t=0}^{T-1}
      \gamma^t r_t
    \right],
\qquad
0\le\gamma\le1.
$$

[策略梯度](../alignment/policy-gradient/)使用 rollout 估计：

$$
\nabla_\theta J(\theta)
  =\mathbb E_t
    \left[
      \nabla_\theta\log\pi_\theta(a_t\mid s_t)
      \hat A_t
    \right].
$$

如果一批 rollout 只用于一次很小的更新，旧策略和新策略相近，估计仍然接近采样分布。若对同一批数据重复训练很多个 epoch，当前策略可能离开生成数据的策略，原始梯度估计就不再对应同一分布。

PPO 保存生成这批数据时的旧策略 $\pi_{\theta_{\mathrm{old}}}$，再用概率比衡量当前策略相对旧策略改变了多少。它在 surrogate objective 中惩罚超出局部区间的收益，允许一批数据被重复利用，同时保留一个可观测的策略漂移指标。

| 对象 | 符号 | 作用 |
|---|---|---|
| 当前策略 | $\pi_\theta$ | 接收梯度并产生下一轮策略 |
| 旧策略 | $\pi_{\theta_{\mathrm{old}}}$ | 生成当前 batch 的 rollout，冻结到该 batch 用完 |
| 参考策略 | $\pi_{\mathrm{ref}}$ | 可选的 KL 基准，常见于对齐训练 |
| advantage | $\hat A_t$ | 估计当前动作相对状态基线的好坏 |
| clip 范围 | $[1-\epsilon,1+\epsilon]$ | 限制单个样本的局部目标收益 |

旧策略和参考策略是两个不同对象。旧策略随 batch 更新；参考策略通常跨多个 batch 冻结。把两者的 log probability 混用，会同时破坏 PPO ratio 和 KL 监测。

## rollout 把策略问题变成有限批次

一条轨迹可以写为：

$$
\tau
  =(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_T).
$$

对每个时间步至少要保存：

| 字段 | 数学内容 | 失配时的结果 |
|---|---|---|
| state 或 observation | $s_t$ | value 和 policy 看到的输入不一致 |
| action | $a_t$ | 无法复算旧策略 log probability |
| reward | $r_t$ | return 和 advantage 方向改变 |
| done/truncated | 终止或截断状态 | bootstrap 边界错误 |
| old log probability | $\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$ | ratio 被重新计算成 1 |
| value prediction | $V_{\psi_{\mathrm{old}}}(s_t)$ | GAE 和 value target 不可复现 |
| action mask | 合法动作集合 | 无效动作进入概率归一化 |

rollout 是旧策略产生的证据，不是当前策略的实时查询结果。保存 batch 后，当前策略可以进行多个 minibatch epoch；但每次更新都必须保留同一份 old log probability。

### return 和 advantage 分开记录

Monte Carlo return 为：

$$
G_t
  =\sum_{l=0}^{T-t-1}
    \gamma^l r_{t+l}.
$$

value function 估计状态的期望 return：

$$
V^\pi(s_t)
  =\mathbb E_\pi[G_t\mid s_t].
$$

advantage 为：

$$
A^\pi(s_t,a_t)
  =Q^\pi(s_t,a_t)-V^\pi(s_t).
$$

它回答“这个动作比当前状态的平均动作好多少”。reward 高不代表 advantage 一定高；如果该状态下所有动作的 return 都高，value 也会一起升高。

## 概率比连接旧分布和当前分布

对从旧策略采样的动作，定义：

$$
\rho_t(\theta)
  =
  \frac{\pi_\theta(a_t\mid s_t)}
       {\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
  =
  \exp\left(
    \log\pi_\theta(a_t\mid s_t)
    -\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
  \right).
$$

当 $\rho_t=1$ 时，当前策略对这个动作的概率没有变化；当 $\rho_t=1.2$ 时，当前策略把这个动作的概率提高了 20%；当 $\rho_t=0.7$ 时，概率降低了 30%。这个比率只描述一个状态动作位置，不等于整条轨迹的 KL。

### 重要性采样为什么出现概率比

对旧策略采样的分布，可以用重要性采样改写当前策略下的期望：

$$
\mathbb E_{a\sim\pi_\theta}[f(a)]
  =
\mathbb E_{a\sim\pi_{\mathrm{old}}}
  \left[
    \frac{\pi_\theta(a\mid s)}
         {\pi_{\mathrm{old}}(a\mid s)}
    f(a)
  \right].
$$

因此 policy gradient 的一阶 surrogate 可以写成：

$$
L^{\mathrm{CPI}}(\theta)
  =
\mathbb E_t
  \left[
    \rho_t(\theta)\hat A_t
  \right].
$$

这个形式在旧策略和当前策略接近时有效。当前策略远离旧策略时，少量旧数据会被大概率比放大，估计方差和更新幅度都会增长。

### ratio 需要在同一 token 或动作上计算

如果 action 是离散环境动作，ratio 对一个动作直接计算。如果 action 是语言模型的 response token，常见 token 级形式为：

$$
\rho_t
  =
\exp\left(
  \log\pi_\theta(y_t\mid x,y_{<t})
  -\log\pi_{\mathrm{old}}(y_t\mid x,y_{<t})
\right).
$$

整条 response 的概率比是所有 token 比率的乘积：

$$
\rho_{\mathrm{seq}}
  =
\frac{\pi_\theta(y\mid x)}
     {\pi_{\mathrm{old}}(y\mid x)}
  =
\exp\left(
  \sum_{t=1}^{T}
  \left[
    \log\pi_\theta(y_t\mid x,y_{<t})
    -\log\pi_{\mathrm{old}}(y_t\mid x,y_{<t})
  \right]
\right).
$$

当 $T$ 较大时，$\rho_{\mathrm{seq}}$ 很容易远离 1。实现若对 token ratio 做 clip，不能把它解释为对 sequence ratio 做 clip；两者的目标不同。

## Clipped surrogate 如何限制单样本收益

PPO 的常用 clipped objective 为：

$$
L^{\mathrm{CLIP}}(\theta)
  =
\mathbb E_t
  \left[
    \min\left(
      \rho_t(\theta)\hat A_t,
      \operatorname{clip}
      \left(
        \rho_t(\theta),
        1-\epsilon,
        1+\epsilon
      \right)
      \hat A_t
    \right)
  \right].
$$

其中 $\epsilon>0$。训练时通常最大化这个目标，或者最小化其相反数。

### advantage 的符号决定哪一侧被限制

对单个样本，记 $\rho=\rho_t$、$A=\hat A_t$：

$$
\min(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A)
  =
\begin{cases}
A\min(\rho,1+\epsilon),&A\ge0,\\
A\max(\rho,1-\epsilon),&A<0.
\end{cases}
$$

当 $A>0$，提高动作概率是有利方向；超过 $1+\epsilon$ 后，额外提高不再增加目标。降低到 $1-\epsilon$ 以下不会被强行拉回，因为降低一个好动作的概率本身已经降低了目标。

当 $A<0$，降低动作概率是有利方向；低于 $1-\epsilon$ 后，额外降低不再增加目标。提高到 $1+\epsilon$ 以上仍然会继续降低目标，因为把坏动作的概率提高得更多仍然更差。

| 情况 | $\rho$ | $A$ | 未裁剪项 | 裁剪项 | PPO 取值 |
|---|---:|---:|---:|---:|---:|
| 提高好动作 | $1.2$ | $0.8$ | $0.96$ | $0.96$ | $0.96$ |
| 降低坏动作 | $0.5$ | $-0.8$ | $-0.40$ | $-0.64$ | $-0.64$ |
| 继续提高好动作 | $1.8$ | $0.8$ | $1.44$ | $0.96$ | $0.96$ |
| 继续降低坏动作 | $0.2$ | $-0.8$ | $-0.16$ | $-0.64$ | $-0.64$ |

裁剪只限制 surrogate 对一个样本的可获得收益。它不直接修改当前策略的概率分布，也不保证所有动作的 ratio 都在区间内。

### clip 不是硬 trust region

即使所有被采样动作的 clipped term 没有继续增加，也可能存在以下情况：

- 未采样动作的概率显著变化；
- 多个 token 的 sequence ratio 远离 1；
- 不同状态的更新方向叠加后使总 KL 增长；
- minibatch 顺序和多个 epoch 使参数继续移动；
- value、entropy 或辅助损失推动 policy objective 之外的更新。

因此需要同时记录 ratio 分位数、clip fraction、近似 KL 和独立任务结果。只记录 clipped loss 不能证明策略处于信赖域内。

## GAE 把 reward 传回早期动作

### TD error 是局部 bootstrap 残差

对 value prediction $V_\psi(s_t)$，TD error 为：

$$
\delta_t
  =r_t+\gamma V_\psi(s_{t+1})-V_\psi(s_t).
$$

如果 $s_{t+1}$ 是真正终止状态，通常不进行 bootstrap；如果只是时间截断，需要根据环境合同决定是否 bootstrap。把二者都当作 terminal 会低估长轨迹 return。

### GAE 在偏差和方差之间取值

广义优势估计为：

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
  =
\sum_{l=0}^{T-t-1}
  (\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda=0$ 只使用当前 TD error，方差较低但依赖 value 的准确性；$\lambda$ 接近 1 时更接近 Monte Carlo return，偏差较低但方差可能增加。两者都不是免费改动，必须在同一 reward scale 和 terminal mask 下比较。

### 三步轨迹的数字

设：

$$
\begin{aligned}
r&=(0,0.5,1),\\
V&=(0.2,0.4,0.8,0),\\
\gamma&=0.9,\qquad\lambda=0.95.
\end{aligned}
$$

TD error 为：

$$
\delta
  =(0.16,0.82,0.20).
$$

从后向前累加：

$$
\hat A
  =(1.007305,0.991000,0.200000).
$$

对应的 return target 为：

$$
\hat R_t=\hat A_t+V(s_t)
  =(1.207305,1.391000,1.000000).
$$

这里第三步的 reward 为 1，最后状态 value 为 0。若把最后状态错误地当成非终止并使用一个非零 bootstrap value，三个 advantage 都会变化。

## value loss 和 entropy 不是附属日志

### value head 预测 return target

常见 value loss 为：

$$
L_V(\psi)
  =
\mathbb E_t
  \left[
    \left(
      V_\psi(s_t)-\hat R_t
    \right)^2
  \right].
$$

有些 PPO 实现还裁剪 value 更新：

$$
\begin{aligned}
V_{\mathrm{clip}}(s_t)
  &=V_{\mathrm{old}}(s_t)
    +\operatorname{clip}
      \left(
        V_\psi(s_t)-V_{\mathrm{old}}(s_t),
        -\epsilon_V,\epsilon_V
      \right),\\
L_V^{\mathrm{clip}}
  &=\frac12\max\left(
    (V_\psi-\hat R)^2,
    (V_{\mathrm{clip}}-\hat R)^2
  \right).
\end{aligned}
$$

value clipping 防止 value head 在一个 batch 上移动过大，但它可能使 value loss 暂时不下降。需要分别记录 unclipped loss、clipped loss、value explained variance 和 advantage 方差。

### entropy 保持动作分布不立即塌缩

离散动作策略熵为：

$$
H(\pi_\theta(\cdot\mid s))
  =
  -\sum_{a}
    \pi_\theta(a\mid s)
    \log\pi_\theta(a\mid s).
$$

若最大化目标，常写为：

$$
L_{\mathrm{total}}
  =
  L^{\mathrm{CLIP}}
  -c_V L_V
  +c_H\mathbb E_t[H_t].
$$

$c_H$ 越大，训练越偏向保留随机性；它不能替代 reference KL，也不能保证生成回答安全。语言模型中常使用 token 分布熵或 response-level 近似，必须注明 mask 和 reduction。

### reward scale 会改变 advantage scale

如果 reward 全部乘以 $c>0$，在 value 拟合完全同步且没有其他尺度项的理想条件下，advantage 也会乘以 $c$：

$$
\hat A_t^{(c)}
  =c\hat A_t.
$$

但实际训练还有 value coefficient、entropy coefficient、gradient clipping、KL penalty 和学习率，因此 reward normalization 的切换会改变参数轨迹。PPO 配置至少要保存 reward mean/std、advantage mean/std、是否 whiten、clip range 和每个 loss 的权重。

## 一个可运行的 PPO 数值探针

下面的标准库探针覆盖一个二动作 batch、三步 GAE、value clipping、entropy 和两个 KL 方向。它把 PPO 的数值合同拆开输出，便于在真实实现中逐项对照。

```python
import math


old_probabilities = [0.60, 0.40]
new_probabilities = [0.72, 0.28]
advantages = [0.8, -0.8]
epsilon = 0.2


def ppo_terms(old, new, advantage):
    ratio = new / old
    clipped_ratio = min(
        max(ratio, 1.0 - epsilon),
        1.0 + epsilon,
    )
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    return ratio, clipped_ratio, unclipped, clipped


print(
    "old_probs=",
    [f"{value:.12f}" for value in old_probabilities],
)
print(
    "new_probs=",
    [f"{value:.12f}" for value in new_probabilities],
)

terms = [
    ppo_terms(old, new, advantage)
    for old, new, advantage in zip(
        old_probabilities,
        new_probabilities,
        advantages,
    )
]
print("ratios=", [
    f"{term[0]:.12f}" for term in terms
])
print("clipped_ratios=", [
    f"{term[1]:.12f}" for term in terms
])
print("unclipped_terms=", [
    f"{term[2]:.12f}" for term in terms
])
print("clipped_terms=", [
    f"{term[3]:.12f}" for term in terms
])
sample_min_terms = [
    min(term[2], term[3])
    for term in terms
]
print("sample_min_terms=", [
    f"{value:.12f}" for value in sample_min_terms
])
print(
    "mean_unclipped=",
    f"{sum(term[2] for term in terms) / len(terms):.12f}",
)
print(
    "mean_clipped=",
    f"{sum(term[3] for term in terms) / len(terms):.12f}",
)
print(
    "mean_min=",
    f"{sum(sample_min_terms) / len(sample_min_terms):.12f}",
)

kl_old_new = sum(
    old * math.log(old / new)
    for old, new in zip(
        old_probabilities,
        new_probabilities,
    )
)
kl_new_old = sum(
    new * math.log(new / old)
    for old, new in zip(
        old_probabilities,
        new_probabilities,
    )
)
entropy = -sum(
    probability * math.log(probability)
    for probability in new_probabilities
)
print("kl_old_new=", f"{kl_old_new:.12f}")
print("kl_new_old=", f"{kl_new_old:.12f}")
print("new_entropy=", f"{entropy:.12f}")

rewards = [0.0, 0.5, 1.0]
values = [0.2, 0.4, 0.8, 0.0]
gamma = 0.9
lambda_ = 0.95
td_errors = [
    rewards[index]
    + gamma * values[index + 1]
    - values[index]
    for index in range(3)
]
gae = [0.0, 0.0, 0.0]
running = 0.0
for index in range(2, -1, -1):
    running = (
        td_errors[index]
        + gamma * lambda_ * running
    )
    gae[index] = running
returns = [
    gae[index] + values[index]
    for index in range(3)
]
print("td_errors=", [
    f"{value:.12f}" for value in td_errors
])
print("gae=", [
    f"{value:.12f}" for value in gae
])
print("returns=", [
    f"{value:.12f}" for value in returns
])

old_value = 0.5
new_value = 0.8
target = 0.9
value_epsilon = 0.2
clipped_value = old_value + max(
    min(new_value - old_value, value_epsilon),
    -value_epsilon,
)
raw_value_loss = (new_value - target) ** 2
clipped_value_loss = (clipped_value - target) ** 2
print(
    "value_clipped=",
    f"{clipped_value:.12f}",
    "raw_loss=",
    f"{raw_value_loss:.12f}",
    "clipped_loss=",
    f"{clipped_value_loss:.12f}",
    "max_loss=",
    f"{max(raw_value_loss, clipped_value_loss):.12f}",
)

advantage_mean = sum(advantages) / len(advantages)
advantage_std = (
    sum(
        (value - advantage_mean) ** 2
        for value in advantages
    ) / len(advantages)
) ** 0.5
normalized = [
    (value - advantage_mean) / advantage_std
    for value in advantages
]
print("adv_mean=", f"{advantage_mean:.12f}")
print("adv_std=", f"{advantage_std:.12f}")
print("normalized=", [
    f"{value:.12f}" for value in normalized
])
print(
    "entropy_regularized_objective=",
    f"{sum(sample_min_terms) / len(sample_min_terms) + 0.01 * entropy:.12f}",
)
```

输出为：

```text
old_probs= ['0.600000000000', '0.400000000000']
new_probs= ['0.720000000000', '0.280000000000']
ratios= ['1.200000000000', '0.700000000000']
clipped_ratios= ['1.200000000000', '0.800000000000']
unclipped_terms= ['0.960000000000', '-0.560000000000']
clipped_terms= ['0.960000000000', '-0.640000000000']
sample_min_terms= ['0.960000000000', '-0.640000000000']
mean_unclipped= 0.200000000000
mean_clipped= 0.160000000000
mean_min= 0.160000000000
kl_old_new= 0.033277043499
kl_new_old= 0.031402536589
new_entropy= 0.592953317447
td_errors= ['0.160000000000', '0.820000000000', '0.200000000000']
gae= ['1.007305000000', '0.991000000000', '0.200000000000']
returns= ['1.207305000000', '1.391000000000', '1.000000000000']
value_clipped= 0.700000000000 raw_loss= 0.010000000000 clipped_loss= 0.040000000000 max_loss= 0.040000000000
adv_mean= 0.000000000000
adv_std= 0.800000000000
normalized= ['1.000000000000', '-1.000000000000']
entropy_regularized_objective= 0.165929533174
```

探针验证了四个局部合同。第一，正 advantage 的动作比率为 1.2，负 advantage 的动作比率为 0.7，后者被裁剪到 0.8，batch 的 mean-min objective 为 0.16。第二，旧策略到新策略和新策略到旧策略的 KL 不相同，方向必须写明。第三，GAE 的三个 TD error 经过 $\gamma\lambda$ 衰减后得到三个 advantage。第四，value clipping 取 raw loss 与 clipped loss 的较大值，在当前例子中为 0.04。

advantage 标准化后两个样本变成 $1$ 和 $-1$。这不会改变排序方向，却会改变 policy objective 的尺度；entropy coefficient 为 0.01 时，熵正则后的教学目标为 0.165929533174。真实训练还会叠加 value loss、KL penalty 和其他辅助项。

## 多个 epoch 会重新引入旧数据偏差

### old policy 是 batch 级快照

一个 PPO batch 的时间线应当是：

1. 用 $\pi_{\theta_{\mathrm{old}}}$ 生成 rollout；
2. 保存 action、reward、old log probability 和 value；
3. 固定这些字段，计算 advantage 和 return；
4. 用当前 $\pi_\theta$ 在多个 minibatch epoch 中更新；
5. 记录 ratio、clip fraction 和 KL；
6. 丢弃 batch，重新生成或重新采样。

第 4 步中可以多次使用数据，但 old log probability 不能随着当前参数重算。完成多个 epoch 后，ratio 分布通常会逐渐离开 1；继续使用同一批数据会增加 surrogate 与真实目标之间的偏差。

### clip fraction 只是局部报警器

定义：

$$
\operatorname{clipfrac}
  =\frac{1}{N}
    \sum_{t=1}^{N}
    \mathbf 1
    \left[
      \rho_t<1-\epsilon
      \ \text{or}\
      \rho_t>1+\epsilon
    \right].
$$

clip fraction 高表示很多样本落在裁剪区间外。它不表示更新一定失败，也不表示 clip fraction 低时策略一定安全。需要和 approximate KL、ratio 分位数、entropy、advantage 方差和独立指标一起解释。

### minibatch 顺序会影响优化路径

PPO 的目标在 batch 级别定义，但 Adam、gradient clipping、advantage normalization 和 minibatch 顺序会让不同排列产生不同参数轨迹。复现实验需要保存：

| 训练设置 | 需要固定 |
|---|---|
| batch size | 每次 rollout 包含的状态动作数量 |
| minibatch size | 每次参数更新使用的样本数 |
| epoch 数 | 同一批数据重复使用次数 |
| shuffle seed | minibatch 排列 |
| optimizer state | Adam 的一阶、二阶动量 |
| gradient clipping | norm 或 value 以及阈值 |
| advantage reduction | batch 内均值、标准差和 mask |

## KL 监测和 PPO clip 处在不同层

### 两个 Bernoulli 策略的 KL

对于二动作策略 $p=(p_0,p_1)$ 和 $q=(q_0,q_1)$：

$$
\operatorname{KL}(p\lVert q)
  =\sum_{i=0}^{1}
    p_i\log\frac{p_i}{q_i}.
$$

探针中：

$$
\operatorname{KL}(\pi_{\mathrm{old}}\lVert\pi_{\mathrm{new}})
  =0.033277043499,
\qquad
\operatorname{KL}(\pi_{\mathrm{new}}\lVert\pi_{\mathrm{old}})
  =0.031402536589.
$$

KL 非对称，因此日志必须标记方向。PPO clip 使用采样动作的概率比和 advantage；KL 需要对动作分布或采样近似取期望，两者不能互相替代。

### 对齐训练还会有 reference KL

在语言模型对齐中，可以同时存在：

1. $\pi_\theta$ 与 $\pi_{\mathrm{old}}$ 的 PPO ratio；
2. $\pi_\theta$ 与 $\pi_{\mathrm{ref}}$ 的 reference KL；
3. response token 上的 sampled log-ratio；
4. 独立任务结果和约束违反率。

[RLHF 与 DPO](../alignment/rlhf-dpo/)中的 $\pi_{\mathrm{ref}}$ 是偏好优化的长期参考；PPO 的 $\pi_{\mathrm{old}}$ 只负责当前 rollout batch。把 old/ref 统一命名为 reference 会隐藏这两个时间尺度。

## 语言模型中的 PPO 还要处理 token 合同

### action 是 response token

在语言模型中：

$$
s_t=(x,y_{<t}),
\qquad
a_t=y_t.
$$

策略概率是下一个 token 的条件分布。response 的总 log probability 是 token log probability 之和，但 PPO 的 advantage 可以是 response 级别的标量，也可以经过 token-level reward shaping 分配到各位置。

### prompt 和 padding 不应进入 policy loss

设 $m_t$ 是 response mask：

$$
\log\pi_\theta(y\mid x)
  =\sum_{t=1}^{T}
    m_t\log\pi_\theta(y_t\mid x,y_{<t}).
$$

至少要屏蔽：

- prompt token；
- padding token；
- EOS 之后的位置；
- 超过有效截断边界的位置；
- 工具调用或环境返回不属于策略动作的 token。

如果 chosen response 和 rejected response 的长度不同，使用 sequence reduction 会把长度直接带入 ratio；使用 token mean 会改变权重。需要在实验设置中固定 sum、mean、per-token advantage 和 response-level advantage。

### action mask 要在 old 和 new 策略中一致

合法动作 mask 会改变 softmax 的归一化分母。若 old policy 在采样时屏蔽了动作，而 new policy 计算 ratio 时没有屏蔽，两个概率不在同一个动作空间中。语言模型中的词表 mask、工具 action mask、EOS mask 和安全动作约束都需要同时应用到 old、new 和 reference log probability。

## 失效模式与审计清单

### 概率和目标

| 现象 | 可能原因 | 最小检查 |
|---|---|---|
| ratio 全部接近 1 | old logprob 被当前策略重算 | 抽查 rollout 保存的 old logprob |
| ratio 迅速极端化 | 学习率、epoch 或 advantage 过大 | ratio 分位数和 clip fraction |
| clip fraction 长期为 0 | 更新太小或 epsilon 太大 | 参数更新比率和 KL |
| clip fraction 接近 1 | 更新过大或数据太旧 | 减少 epoch，检查 batch 时间戳 |
| clip loss 下降但回报下降 | reward、advantage 或 mask 错误 | 独立回放单批 return |

### GAE 和 value

| 现象 | 可能原因 | 最小检查 |
|---|---|---|
| advantage 方差爆炸 | reward scale 或 terminal mask 错误 | TD error、return 和长度分组 |
| value loss 下降但策略不变 | value coefficient 或 advantage 过小 | policy gradient 和 advantage 均值 |
| value clipping 长期激活 | value 学习率或 target 尺度失配 | raw/clipped value loss |
| 长期任务性能下降 | gamma、lambda 或 bootstrap 边界错误 | terminal/truncated 分类 |

### 语言模型和对齐

| 现象 | 可能原因 | 最小检查 |
|---|---|---|
| 生成长度异常增加 | EOS reward、KL reduction 或 response mask 错误 | 每 token KL、EOS 率、长度分布 |
| KL 很小但回答错误 | reference 本身错误或 reward 只覆盖捷径 | 独立事实和安全切片 |
| 工具动作被错误优化 | 环境动作和文本 token 混在同一 mask | action type、权限和外部结果 |
| 中文或长上下文退化 | batch 分层、padding 或 position mask 失配 | 按语言和长度报告指标 |
| 新旧 checkpoint 不可复现 | old policy、optimizer state 或 seed 未保存 | hash、seed、minibatch 顺序 |

一次最小复现应固定一个 batch，导出 old/new/ref logits、action mask、old logprob、value、reward、done/truncated、advantage、return 和 loss 分解。先在这个 batch 上核对 ratio 与中心差分，再运行完整训练。

## 运行方法

将上一个 Python 代码块保存为 ppo-probe.py，再运行：

```bash
python3 ppo-probe.py
```

接入真实策略时，先导出一个固定 rollout batch，再逐条对照 old/new log probability、ratio、clip term、GAE、value clipping、entropy、KL 和 mask。修改 epsilon、gamma、lambda、epoch 或 reward normalization 后，重新记录这些指标和独立任务结果。

## 相关词条

- [策略梯度](../alignment/policy-gradient/)：从轨迹概率推导 log-derivative 和 advantage 更新。
- [强化学习基础](../alignment/rl-basics/)：定义 return、value、策略和终止状态。
- [RLHF 与 DPO](../alignment/rlhf-dpo/)：说明 PPO 在语言模型偏好优化中的 reward、KL 和 reference 边界。
- [奖励模型](../alignment/reward-model/)：提供 PPO 读取的代理 reward，并说明过优化风险。
- [训练稳定性](../pretraining/training-stability/)：检查 loss、梯度、精度、KL 和训练分布。
- [奖励投机](../alignment/reward-hacking/)：检查 reward 上升而独立结果下降的路径。
- [DPO](../alignment/dpo/)：比较不经过在线 PPO rollout 的直接偏好优化目标。
- [对齐问题](../alignment/alignment-problem/)：区分代理目标、真实意图和外部约束。
