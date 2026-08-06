---
title: "RLHF 与 DPO：在线奖励优化和离线偏好优化"
tags: ["why-models-learn"]
---

RLHF 与 DPO 都把“同一输入下人类更偏好哪个回答”的标签转换为策略更新，但两条路径保存的中间对象不同。RLHF 训练显式 reward model，再用在线或近在线策略优化和 KL 约束改变策略；DPO 从 KL 正则化目标的最优策略形式出发，把 reward 差分改写成当前策略与参考策略的 log probability 差分，直接在离线偏好对上训练。本文从共享数据合同开始，推导 RLHF 的 reward–advantage–PPO 路径和 DPO 的闭式损失，实跑一个二动作探针，并区分参考策略、beta、token mask、数据分布和独立评测的边界。[奖励模型](../alignment/reward-model/)和 [Bradley–Terry 模型](../alignment/bradley-terry/)提供偏好统计层；本文处理它们如何进入策略优化。

![RLHF 与 DPO 示意图：二者共享 SFT 和偏好数据，RLHF 经过 reward model、rollout、advantage 与 PPO，DPO 直接比较策略和参考策略的序列 log probability](/assets/alignment/svg/rlhf-dpo.1.svg)

## 两条路径共享偏好数据合同

设输入为 $x$，候选回答为 $y$。一条成对偏好记录至少包含：

$$
\mathcal D_{\mathrm{pref}}
  =\left\{
    (x,y_{\mathrm w},y_{\mathrm l},z)
  \right\},
$$

其中 $y_{\mathrm w}$ 是 chosen，$y_{\mathrm l}$ 是 rejected，$z$ 保存标签来源、指南版本、候选生成策略、平局或分歧状态等元数据。一个候选回答不是一个独立标量；它还绑定 tokenizer、prompt 模板、EOS 规则、截断长度和工具结果。

| 字段 | RLHF 使用方式 | DPO 使用方式 |
|---|---|---|
| 输入 $x$ | 生成 rollout 和计算 reward | 计算 chosen/rejected 的条件 log probability |
| $y_{\mathrm w},y_{\mathrm l}$ | 训练 reward model 或做独立比较 | 直接形成 DPO logit |
| 参考策略 | KL 惩罚和策略漂移基准 | 计算 reference log probability |
| 标签指南 | 决定 reward model 的监督目标 | 决定 DPO 的偏好方向 |
| 候选来源 | 影响 reward model 覆盖和 rollout 分布 | 决定离线数据覆盖 |
| 独立结果 | 检查 reward、策略和真实任务 | 检查 log-ratio 提升是否对应真实结果 |

如果 chosen/rejected 的顺序、模板或截断规则在训练与评测之间发生变化，损失数值仍然可以下降，但它不再对应同一个偏好任务。

### SFT 提供初始策略和参考分布

通常先用示范数据训练监督策略 $\pi_{\mathrm{SFT}}$。它至少承担两个作用：

1. 产生能完成基本任务的初始回答；
2. 提供后续参考策略 $\pi_{\mathrm{ref}}$，限制优化后的策略不要快速离开已知分布。

参考策略可以是 SFT checkpoint 的冻结副本，也可以是另一个明确版本。它不能在训练过程中随着当前策略无记录地更新，否则 KL 惩罚和 DPO log-ratio 都失去固定坐标。

## RLHF 先把偏好压成 reward

### reward model 学习成对标签

对 reward model $r_\phi(x,y)$，Bradley–Terry 偏好概率写为：

$$
p_\phi(y_{\mathrm w}\succ y_{\mathrm l}\mid x)
  =\sigma\left(
    r_\phi(x,y_{\mathrm w})
    -r_\phi(x,y_{\mathrm l})
  \right).
$$

成对负对数似然为：

$$
\mathcal L_{\mathrm{RM}}
  =-\mathbb E_{(x,y_{\mathrm w},y_{\mathrm l})\sim\mathcal D_{\mathrm{pref}}}
    \log\sigma\left(
      r_\phi(x,y_{\mathrm w})
      -r_\phi(x,y_{\mathrm l})
    \right).
$$

reward model 只被监督去排列候选，不会自动知道跨 prompt 的绝对分数，也不会自动确认事实正确性、工具权限或长期用户结果。[奖励模型](../alignment/reward-model/)词条展开了候选覆盖、长度偏差和 reward overoptimization；这里把它当作 RLHF 的一个输入层。

### 策略目标加入参考策略的 KL 约束

设当前策略为 $\pi_\theta$，参考策略为 $\pi_{\mathrm{ref}}$。一个常见的序列级目标是：

$$
J(\pi_\theta)
  =\mathbb E_{\substack{x\sim\mathcal D_x\\y\sim\pi_\theta(\cdot\mid x)}}
    \left[
      r_\phi(x,y)
      -\beta_{\mathrm{KL}}
       \log\frac{\pi_\theta(y\mid x)}
                     {\pi_{\mathrm{ref}}(y\mid x)}
    \right].
$$

因为：

$$
\operatorname{KL}
  \left(
    \pi_\theta(\cdot\mid x)
    \,\middle\lVert\,
    \pi_{\mathrm{ref}}(\cdot\mid x)
  \right)
  =
  \mathbb E_{y\sim\pi_\theta}
  \left[
    \log\frac{\pi_\theta(y\mid x)}
              {\pi_{\mathrm{ref}}(y\mid x)}
  \right],
$$

上式的采样形式可以理解为 reward 减去策略偏离参考分布的代价。对固定的 $x$，令：

$$
\mathcal Z(x)
  =\sum_y
    \pi_{\mathrm{ref}}(y\mid x)
    \exp\left(\frac{r_\phi(x,y)}{\beta_{\mathrm{KL}}}\right).
$$

对 $\pi$ 加入归一化约束并求最优解，得到：

$$
\pi^*(y\mid x)
  =\frac{1}{\mathcal Z(x)}
    \pi_{\mathrm{ref}}(y\mid x)
    \exp\left(
      \frac{r_\phi(x,y)}{\beta_{\mathrm{KL}}}
    \right).
$$

$\beta_{\mathrm{KL}}$ 越大，策略离开参考策略的代价越高，最优分布越接近参考分布。这个闭式形式是 DPO 推导的连接点，但 RLHF 仍然要面对 reward model 的拟合误差、rollout 分布变化和策略优化的数值问题。

### token 级 reward 需要明确归因

语言模型回答由 token 序列 $y_{1:T}$ 组成：

$$
\log\pi_\theta(y\mid x)
  =\sum_{t=1}^{T}
    \log\pi_\theta
    \left(
      y_t\mid x,y_{<t}
    \right).
$$

因此序列级 KL 样本可以写成：

$$
\log\frac{\pi_\theta(y\mid x)}
              {\pi_{\mathrm{ref}}(y\mid x)}
  =\sum_{t=1}^{T}
    \left[
      \log\pi_\theta(y_t\mid x,y_{<t})
      -\log\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})
    \right].
$$

实际实现可能把 KL 差分作为每个 token 的 shaping reward，也可能只在序列末尾加入 reward model 分数。两者的总和合同不同，不能只比较最终 reward 均值。

| 归因形式 | reward 位置 | 需要固定的量 |
|---|---|---|
| terminal reward | EOS 或序列末尾 | EOS、截断、长度和 advantage 回传 |
| per-token KL | 每个生成 token | causal mask、旧策略、参考 log probability |
| token reward + terminal reward | 两者都加入 | 加和顺序、尺度和 value target |
| response-only loss | 只对回答 token 计算 | prompt mask、padding mask、共享 tokenizer |

## PPO 让 RLHF 的策略更新保持局部

### rollout、advantage 和概率比

PPO 从旧策略 $\pi_{\theta_{\mathrm{old}}}$ 采样轨迹，再在同一批数据上更新当前策略。对状态 $s_t$ 和动作 $a_t$ 定义概率比：

$$
\rho_t(\theta)
  =\frac{\pi_\theta(a_t\mid s_t)}
          {\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
  =\exp\left(
    \log\pi_\theta(a_t\mid s_t)
    -\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
  \right).
$$

若 $\hat A_t$ 是 advantage，PPO 的 clipped surrogate 为：

$$
\mathcal L_{\mathrm{clip}}(\theta)
  =\mathbb E_t
    \left[
      \min\left(
        \rho_t(\theta)\hat A_t,
        \operatorname{clip}
        \left(
          \rho_t(\theta),
          1-\epsilon,
          1+\epsilon
        \right)\hat A_t
      \right)
    \right].
$$

这里的目标通常被最大化。$\hat A_t>0$ 时，概率比超过 $1+\epsilon$ 后不再继续增加收益；$\hat A_t<0$ 时，概率比低于 $1-\epsilon$ 后不再继续增加收益。裁剪限制了单批数据对策略的局部推动，但它不保证策略和参考策略的总 KL 一定小。

### value head 估计长程回报

PPO 需要用 value function 降低策略梯度方差。一个带 bootstrap 的 TD 残差为：

$$
\delta_t
  =r_t+\gamma V_\psi(s_{t+1})-V_\psi(s_t).
$$

GAE 估计为：

$$
\hat A_t
  =\sum_{l=0}^{T-t-1}
    (\gamma\lambda)^l\delta_{t+l}.
$$

response 结束、padding、截断和工具失败都可能改变 $s_{t+1}$ 是否存在。把 padding token 当作真实动作会把无效位置送进 value loss 和 policy loss，造成 advantage 和 KL 的错误平均。

### 一个小型 PPO 数值检查

设旧策略给动作的概率为 0.6，新策略给同一动作的概率为 0.72，$\hat A=0.8$，$\epsilon=0.2$。概率比为：

$$
\rho=\frac{0.72}{0.6}=1.2,
\qquad
\rho\hat A=0.96.
$$

它恰好位于上界，clipped objective 仍为 0.96。若旧概率为 0.4，新概率为 0.2，$\hat A=-0.6$，则 $\rho=0.5$，裁剪后概率比为 0.8，两个候选目标为 $-0.3$ 和 $-0.48$，取较小值 $-0.48$。这个取小值方向依赖 advantage 的符号，不能只把 clip 理解成把概率比截断后再相乘。

## DPO 把 reward 差分改写为 log-ratio

### 从 KL 正则最优策略反解 reward

为简化记号，令 $\beta>0$，并假设偏好标签服从：

$$
\Pr(y_{\mathrm w}\succ y_{\mathrm l}\mid x)
  =\sigma\left(
    r(x,y_{\mathrm w})-r(x,y_{\mathrm l})
  \right).
$$

KL 正则目标的最优策略满足：

$$
\pi^*(y\mid x)
  =\frac{1}{Z(x)}
    \pi_{\mathrm{ref}}(y\mid x)
    \exp\left(\frac{r(x,y)}{\beta}\right).
$$

取对数并整理：

$$
r(x,y)
  =\beta
    \left[
      \log\pi^*(y\mid x)
      -\log\pi_{\mathrm{ref}}(y\mid x)
    \right]
    +\beta\log Z(x).
$$

对 chosen 和 rejected 相减后，$\log Z(x)$ 消失：

$$
\begin{aligned}
r(x,y_{\mathrm w})-r(x,y_{\mathrm l})
  =\beta\Big[
    &\log\pi^*(y_{\mathrm w}\mid x)
     -\log\pi_{\mathrm{ref}}(y_{\mathrm w}\mid x)\\
    &-\log\pi^*(y_{\mathrm l}\mid x)
     +\log\pi_{\mathrm{ref}}(y_{\mathrm l}\mid x)
  \Big].
\end{aligned}
$$

DPO 用当前策略 $\pi_\theta$ 代替未知的 $\pi^*$，定义 log-ratio margin：

$$
m_\theta(x,y_{\mathrm w},y_{\mathrm l})
  =
  \left[
    \log\pi_\theta(y_{\mathrm w}\mid x)
    -\log\pi_{\mathrm{ref}}(y_{\mathrm w}\mid x)
  \right]
  -
  \left[
    \log\pi_\theta(y_{\mathrm l}\mid x)
    -\log\pi_{\mathrm{ref}}(y_{\mathrm l}\mid x)
  \right].
$$

把 $\beta m_\theta$ 代回 Bradley–Terry 概率，得到 DPO 的单样本损失：

$$
\mathcal L_{\mathrm{DPO}}
  =-\log\sigma\left(
    \beta m_\theta
  \right).
$$

它不需要显式训练 reward model，也不需要在每个更新批次生成新的 rollout；当前策略和冻结参考策略都要对同一对回答计算条件 log probability。

### DPO 梯度推动 chosen 的相对 log-ratio

令 $q=\sigma(\beta m_\theta)$。对单样本损失求导：

$$
\frac{\partial\mathcal L_{\mathrm{DPO}}}
     {\partial(\beta m_\theta)}
  =q-1.
$$

当 chosen 的相对 log-ratio 太低时，$q<1$，梯度会提高 chosen 的 log probability、降低 rejected 的 log probability。实际网络参数梯度还要经过 softmax 和共享 Transformer 表示；“提高 chosen、降低 rejected”是对两个独立 log probability 项的方向说明，不是每个 token 的独立参数更新。

### DPO 不是无条件的 reward-free 结论

DPO 的闭式损失依赖一组模型假设：

| 假设 | 在推导中的作用 | 违反时的结果 |
|---|---|---|
| 偏好服从 Bradley–Terry | 把 reward 差变成 sigmoid 概率 | log-ratio 不再对应同一概率模型 |
| 参考策略固定 | 让 $\pi_\theta/\pi_{\mathrm{ref}}$ 有固定坐标 | 训练目标随参考模型漂移 |
| KL 正则形式成立 | 得到指数倾斜的最优策略 | DPO 与原目标不再严格对应 |
| 偏好对覆盖当前任务 | 提供 chosen/rejected 的方向 | 策略可能学习数据捷径 |
| log probability 合同一致 | 正确计算序列分差 | prompt、EOS 或 mask 错配 |

因此“没有显式 reward model”描述的是训练路径，不表示没有代理目标、没有参考分布或没有 reward hacking 风险。

## 一个可运行的 RLHF 与 DPO 探针

下面的标准库探针用两个离散回答表示一个偏好对。它核对 DPO margin、损失、对 log probability 项的解析梯度、策略与参考策略的 KL、KL 正则目标的最优分布，以及两个 PPO clipping 例子。

```python
import math


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


beta = 0.5
reference_chosen = 0.60
reference_rejected = 0.40
policy_chosen = 0.75
policy_rejected = 0.25


def dpo_margin(policy_w, policy_l, reference_w, reference_l):
    return (
        math.log(policy_w)
        - math.log(reference_w)
        - math.log(policy_l)
        + math.log(reference_l)
    )


margin = dpo_margin(
    policy_chosen,
    policy_rejected,
    reference_chosen,
    reference_rejected,
)
dpo_logit = beta * margin
dpo_probability = sigmoid(dpo_logit)
dpo_loss = -math.log(dpo_probability)
gradient_logit = dpo_probability - 1.0
print("log_ratio_margin=", f"{margin:.12f}")
print("dpo_logit=", f"{dpo_logit:.12f}")
print(
    "dpo_preference_probability=",
    f"{dpo_probability:.12f}",
)
print("dpo_loss=", f"{dpo_loss:.12f}")
print(
    "dpo_grad_logp_chosen=",
    f"{beta * gradient_logit:.12f}",
)
print(
    "dpo_grad_logp_rejected=",
    f"{-beta * gradient_logit:.12f}",
)

policy_reference_kl = (
    policy_chosen
    * math.log(policy_chosen / reference_chosen)
    + policy_rejected
    * math.log(policy_rejected / reference_rejected)
)
print("policy_ref_kl=", f"{policy_reference_kl:.12f}")

reward_chosen = 1.0
reward_rejected = 0.0
unnormalized = [
    reference_chosen * math.exp(reward_chosen / beta),
    reference_rejected * math.exp(reward_rejected / beta),
]
normalizer = sum(unnormalized)
optimal_policy = [
    value / normalizer for value in unnormalized
]
print(
    "kl_regularized_optimal=",
    [f"{value:.12f}" for value in optimal_policy],
)
optimal_margin = dpo_margin(
    optimal_policy[0],
    optimal_policy[1],
    reference_chosen,
    reference_rejected,
)
print(
    "optimal_log_ratio_margin=",
    f"{optimal_margin:.12f}",
)
print(
    "optimal_dpo_probability=",
    f"{sigmoid(beta * optimal_margin):.12f}",
)


def ppo_clipped(old_probability, new_probability, advantage, epsilon):
    ratio = new_probability / old_probability
    clipped_ratio = min(
        max(ratio, 1.0 - epsilon),
        1.0 + epsilon,
    )
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    return ratio, clipped_ratio, unclipped, clipped, min(
        unclipped,
        clipped,
    )


for values in (
    (0.60, 0.72, 0.8, 0.2),
    (0.40, 0.20, -0.6, 0.2),
):
    result = ppo_clipped(*values)
    print(
        "ppo",
        "ratio=",
        f"{result[0]:.12f}",
        "clipped=",
        f"{result[1]:.12f}",
        "unclipped=",
        f"{result[2]:.12f}",
        "clipped_objective=",
        f"{result[3]:.12f}",
        "min_objective=",
        f"{result[4]:.12f}",
    )

for coefficient in (0.1, 0.5, 1.0):
    print(
        "kl_penalty_coeff=",
        coefficient,
        "penalty=",
        f"{coefficient * policy_reference_kl:.12f}",
    )
```

输出为：

```text
log_ratio_margin= 0.693147180560
dpo_logit= 0.346573590280
dpo_preference_probability= 0.585786437627
dpo_loss= 0.534799996740
dpo_grad_logp_chosen= -0.207106781187
dpo_grad_logp_rejected= 0.207106781187
policy_ref_kl= 0.049856756174
kl_regularized_optimal= ['0.917243097104', '0.082756902896']
optimal_log_ratio_margin= 2.000000000000
optimal_dpo_probability= 0.731058578630
ppo ratio= 1.200000000000 clipped= 1.200000000000 unclipped= 0.960000000000 clipped_objective= 0.960000000000 min_objective= 0.960000000000
ppo ratio= 0.500000000000 clipped= 0.800000000000 unclipped= -0.300000000000 clipped_objective= -0.480000000000 min_objective= -0.480000000000
kl_penalty_coeff= 0.1 penalty= 0.004985675617
kl_penalty_coeff= 0.5 penalty= 0.024928378087
kl_penalty_coeff= 1.0 penalty= 0.049856756174
```

第一组输出说明，当前策略相对于参考策略把 chosen 的 log-ratio 提高了 $\log 2$，在 $\beta=0.5$ 时 DPO logit 为 $\log 2/2$，预测偏好概率为 0.585786437627。损失仍然大于 0，说明策略还没有把 chosen 的相对优势推到无限大。对 chosen 和 rejected 的独立 log probability 项，梯度方向相反，绝对值均为 0.207106781187。

给定参考分布 $(0.6,0.4)$、reward 差为 1 和 $\beta=0.5$，KL 正则目标的最优策略为 $(0.917243097104,0.082756902896)$。它的 log-ratio margin 为 2，代入同一个偏好概率得到 0.731058578630。这个数值连接了 KL 正则的指数倾斜和 DPO 的 log-ratio 目标；它不表示真实人类选择概率已经达到该值。

PPO 的两个例子展示了 advantage 符号的重要性。正 advantage 的概率比 1.2 恰好落在上界；负 advantage 的概率比 0.5 被提高到 0.8 后，与原始目标相乘得到更小的 surrogate。KL penalty 还取决于系数，不能只报告 policy/reference KL 而不报告它进入 reward 的权重。

## RLHF 和 DPO 的差异来自优化时机

### RLHF 使用当前策略生成新样本

典型 RLHF 训练循环包含：

1. 从当前策略生成回答；
2. 用 reward model 评分；
3. 加入 reference KL、终止状态和约束奖励；
4. 用 value head 计算 return 和 advantage；
5. 用 PPO 等方法更新策略；
6. 用新策略重新生成下一批样本。

当前策略会改变采样分布，因此 reward model 的分布外行为会直接进入下一轮训练。每轮都要保存策略版本、候选、reward、KL、长度、拒答率、独立结果和约束违反率。

### DPO 在固定数据上更新策略

DPO 的基本训练循环是：

1. 固定偏好数据和参考策略；
2. 当前策略对 chosen/rejected 计算序列 log probability；
3. 计算两个相对参考的 log-ratio；
4. 用 DPO logistic loss 反向传播；
5. 在留出偏好对和独立任务上评测。

DPO 省略了每次更新的 rollout、reward model forward 和 value head，但仍然需要两个策略的前向计算、完整偏好对和可靠的独立评测。若反复用新策略生成数据再继续 DPO，训练就变成迭代式离线或近在线流程，不能继续把它描述成固定分布的一次性离线训练。

| 维度 | RLHF + PPO | DPO |
|---|---|---|
| 每轮数据 | 当前策略 rollout | 已保存的偏好对 |
| 显式 reward model | 通常需要 | 基础损失不需要 |
| value head | 用于 advantage | 不需要 |
| reference policy | KL 约束 | log-ratio 分母 |
| 策略更新 | 概率比、advantage、clip | 偏好 logit、logistic loss |
| 分布变化 | 训练过程中直接发生 | 固定数据阶段不发生 |
| 主要风险 | reward hacking、value 偏差、rollout 成本 | 数据覆盖、参考漂移、logprob 合同 |
| 监测重点 | reward、KL、advantage、独立结果 | chosen/rejected log-ratio、loss、校准、独立结果 |

DPO 的训练损失更简洁，不能因此推断它比 RLHF 对错误标签更不敏感。它把 reward model 的一部分统计假设和策略约束压缩进了 log-ratio 目标。

## beta、参考策略和 log probability 合同

### beta 同时控制目标尺度和策略漂移

在推导中，$\beta$ 出现在：

$$
\pi^*(y\mid x)
  \propto
  \pi_{\mathrm{ref}}(y\mid x)
  \exp\left(\frac{r(x,y)}{\beta}\right).
$$

因此增大 $\beta$ 会降低相同 reward 差产生的策略偏离；在 DPO 损失中，$\beta m_\theta$ 的绝对值也会改变 sigmoid 的饱和速度。工程配置可能把 beta 叫作 KL coefficient、inverse temperature 或 regularization coefficient，必须同时记录定义和公式，不能只看参数名。

### reference checkpoint 必须冻结并可复现

对同一条偏好对，DPO 使用：

$$
\Delta_{\theta}
  =
  \left[
    \log\pi_\theta(y_{\mathrm w}\mid x)
    -\log\pi_{\mathrm{ref}}(y_{\mathrm w}\mid x)
  \right]
  -
  \left[
    \log\pi_\theta(y_{\mathrm l}\mid x)
    -\log\pi_{\mathrm{ref}}(y_{\mathrm l}\mid x)
  \right].
$$

若 reference 模型、tokenizer、模板或 adapter 状态改变，$\Delta_\theta$ 改变，即使当前策略权重没有改变。应保存 base model hash、reference hash、tokenizer、chat template、LoRA 合并状态和量化配置。

### sequence log probability 只对 response token 求和

给定 prompt token 和 response token，条件序列 log probability 通常是：

$$
\log\pi_\theta(y\mid x)
  =\sum_{t=1}^{T}
    m_t
    \log\pi_\theta(y_t\mid x,y_{<t}),
\qquad
m_t\in\{0,1\}.
$$

$m_t$ 必须屏蔽 prompt、padding、越过 EOS 的位置和无效截断位置。chosen 与 rejected 必须使用相同的 mask 约定。对长回答使用总和会天然累积 token 数差异；改用平均值会改变原始序列概率目标，不能在实验中静默切换。

| 检查项 | 需要回答的问题 |
|---|---|
| prompt mask | 是否只把 response token 送进 loss |
| EOS mask | EOS 是否作为 response token，后续位置是否屏蔽 |
| padding mask | batch padding 是否影响 log probability |
| tokenizer | chosen/rejected/reference 是否共享分词器 |
| template | 训练和 reference 是否看到相同角色标记 |
| adapter state | policy/reference 是否加载了不同 adapter |
| precision | logsumexp 和 log probability 是否在稳定精度中计算 |
| reduction | sum、mean 和 batch reduction 的分母是什么 |

## 数据来源决定 DPO 学到的方向

### 偏好对的候选分布不是中性容器

如果所有 chosen 都来自更大的模型，DPO 可能同时学习模型血统、长度和格式；如果 rejected 只包含语法错误，模型没有数据判断事实错误和工具失败；如果偏好对来自旧策略，训练后的新策略可能进入未覆盖的区域。

应按以下轴保存分层统计：

| 轴 | 例子 | 需要监测 |
|---|---|---|
| 任务 | 问答、摘要、代码、工具调用 | 分组 loss 和独立正确性 |
| 难度 | 基础、边界、不可回答 | 长度、拒答和事实错误 |
| 候选来源 | SFT、当前策略、不同 checkpoint | 生成分布和排序变化 |
| 回答长度 | 短、中、长 | length bias 和 log-ratio |
| 语言/领域 | 中文、英文、专业领域 | 分组偏好和校准 |
| 安全状态 | 允许、拒绝、需确认 | 约束违反和过度拒答 |

### 训练 loss 不能替代独立结果

DPO loss 下降表示当前策略在给定 reference 和偏好对上增大了 chosen 的相对 log-ratio。它不直接测量事实正确性、代码可运行性、工具副作用、隐私边界或用户长期结果。至少要并列报告：

$$
\text{DPO loss},\quad
\Delta_{\theta},\quad
\operatorname{KL}(\pi_\theta\lVert\pi_{\mathrm{ref}}),\quad
\text{独立任务结果},\quad
\text{约束违反率}.
$$

如果只保存 loss 曲线，无法判断策略是在学习偏好内容，还是在利用长度、模板、拒答短语或 reference 的局部漏洞。

## RLHF 的 reward–advantage 合同

### reward scale 影响 PPO 的更新幅度

若所有 reward 乘以常数 $c>0$，最优动作顺序在某些条件下保持，但 return、advantage、value target 和 policy gradient 的尺度都会改变：

$$
\hat A_t^{(c)}
  =c\hat A_t
$$

在 value loss、entropy coefficient、gradient clipping 和学习率不同时缩放时，训练轨迹不会保持不变。RLHF 实验必须记录 reward normalization、whitening、per-token 或 per-sequence reduction。

### KL 样本和 KL 期望是两种量

对一个从当前策略采样的回答，可以计算样本 log-ratio：

$$
k(x,y)
  =\log\frac{\pi_\theta(y\mid x)}
              {\pi_{\mathrm{ref}}(y\mid x)}.
$$

它是一个随机变量；真正的 KL 是在当前策略下对所有回答取期望：

$$
\operatorname{KL}
  \left(
    \pi_\theta\lVert\pi_{\mathrm{ref}}
  \right)
  =\mathbb E_{y\sim\pi_\theta}[k(x,y)].
$$

实现日志中若记录 sampled mean、batch mean、token mean 和 sequence KL，必须明确分母。一个长回答的 token 平均值可能很小，但序列总 log-ratio 仍然很大。

### PPO 的旧策略不能被当前策略覆盖

概率比需要同一批数据上的旧 log probability：

$$
\rho_t
  =\exp(\log\pi_\theta-\log\pi_{\theta_{\mathrm{old}}}).
$$

如果数据加载后重新用当前策略计算分母，$\rho_t$ 接近 1，clip 失去作用；如果旧策略 checkpoint 和 rollout 策略不一致，advantage 与 ratio 也不再对应同一条轨迹。

## 失效模式和最小审计

### DPO 侧

| 现象 | 可能原因 | 最小检查 |
|---|---|---|
| loss 快速趋近 0 | beta 过大、标签简单或策略过拟合 | 分组 margin、留出准确率和 KL |
| chosen 变长且分数上升 | 长度或格式捷径 | 长度匹配、token-level margin、改写 |
| reference KL 异常大 | reference 错载、模板不一致或 beta 过小 | hash、模板、logprob 逐条抽样 |
| loss 下降但独立结果不变 | 偏好数据没有覆盖真实目标 | 增加难例和外部结果评测 |
| 中文或工具任务退化 | 数据分层不足 | 按语言、工具和任务报告 |
| 训练结果依赖 response reduction | sum/mean 或 mask 不一致 | 固定 mask、分母和 EOS 合同 |

### RLHF 侧

| 现象 | 可能原因 | 最小检查 |
|---|---|---|
| reward 上升、独立质量下降 | reward overoptimization 或 reward hacking | 训练分布外候选和独立评测 |
| KL 突然增长 | beta、学习率、value 或 rollout 版本错误 | old/ref/current logprob 和分位数 |
| advantage 方差爆炸 | reward scale、terminal mask 或 value 失配 | return、value error 和长度分组 |
| policy ratio 总在 1 附近 | old logprob 被覆盖 | 保存 rollout 时的旧策略输出 |
| 过度拒答 | 安全标签覆盖过窄或 reward 权重过高 | 允许/拒绝/需确认三组评测 |
| 训练不稳定 | batch、clip、KL、精度或 EOS 合同变化 | 固定实验配置并复现单批梯度 |

### 共享审计

先检查偏好对的 chosen/rejected 方向，再检查 reference 和 tokenizer hash。然后逐条复算一条短回答的 log probability、DPO margin、reward、KL 和 mask。最后对长度、任务、语言、候选来源和安全状态分组比较训练指标与独立结果。

外部系统权限不由 RLHF 或 DPO loss 授予。发送邮件、写入数据库、执行代码和读取敏感数据仍然需要权限检查、确认、沙箱、审计和回滚。

## 运行方法

将上一个 Python 代码块保存为 rlhf-dpo-probe.py，再运行：

```bash
python3 rlhf-dpo-probe.py
```

修改 beta、reference 概率、policy 概率、reward 差或 PPO 的 old/new probability 后，重新核对 DPO margin、梯度方向、KL 正则最优分布和 clipping 目标。接入真实模型时，再对同一条 chosen/rejected 样本导出 token mask 和逐 token log probability。

## 相关词条

- [奖励模型](../alignment/reward-model/)：说明偏好标签如何变成可优化的代理分数。
- [Bradley–Terry 模型](../alignment/bradley-terry/)：推导偏好标签的成对概率和最大似然排序。
- [策略梯度](../alignment/policy-gradient/)：说明 rollout reward 如何通过 log-derivative 更新策略。
- [强化学习基础](../alignment/rl-basics/)：定义状态、动作、回报、价值和策略。
- [对齐问题](../alignment/alignment-problem/)：区分偏好代理目标与真实意图、约束和独立结果。
- [监督微调](../finetuning/sft/)：提供偏好优化前的初始策略和参考分布。
- [PPO](../alignment/ppo/)：展开旧策略、概率比、advantage 和 clipping 目标。
- [DPO](../alignment/dpo/)：若实现按算法拆分，继续记录 DPO 的具体变体和工程差异。
- [奖励投机](../alignment/reward-hacking/)：检查策略提高代理分数但独立结果下降的路径。
- [训练稳定性](../pretraining/training-stability/)：审计 reward、KL、梯度、精度和分布漂移。
