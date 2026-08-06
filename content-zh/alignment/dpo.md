---
title: "DPO：用策略 log-ratio 直接学习偏好"
tags: ["why-models-learn"]
---

DPO（Direct Preference Optimization，直接偏好优化）在固定的 chosen/rejected 回答对上比较当前策略与参考策略的序列 log probability，把 KL 正则化 reward 优化的隐式 reward 差写成一个 logistic loss。它不要求训练一个显式 reward model，也不要求每个更新批次都重新 rollout，但仍然依赖 Bradley–Terry 偏好假设、冻结 reference、准确的 token mask、候选覆盖和独立评测。本文从 KL 正则最优策略推导 DPO 目标，展开序列 log-ratio、beta、label smoothing、softmax 梯度、sum/mean reduction 和 reference-free 风险，再用标准库探针逐项核对。[RLHF 与 DPO](../alignment/rlhf-dpo/)给出 RLHF 在线路径与 DPO 闭式推导的总览；本文聚焦 DPO 的实现合同和失效模式。

![DPO 示意图：偏好对同时经过当前策略和冻结参考策略，两个 log-ratio 的差形成偏好 logit，再直接更新当前策略](/assets/alignment/svg/dpo.1.svg)

## DPO 直接优化偏好对的相对概率

设输入为 $x$，chosen 回答为 $y_{\mathrm w}$，rejected 回答为 $y_{\mathrm l}$。数据集是：

$$
\mathcal D_{\mathrm{pref}}
  =\left\{
    (x,y_{\mathrm w},y_{\mathrm l},z)
  \right\},
$$

其中 $z$ 保存标注指南、候选来源、比较顺序、平局或分歧状态等元数据。DPO 不需要把每个回答标成一个绝对质量分数；它只需要同一输入下的偏好方向。

当前策略为 $\pi_\theta$，冻结参考策略为 $\pi_{\mathrm{ref}}$。对一对回答定义相对 reference 的 log-ratio：

$$
\Delta_\theta(x,y)
  =
  \log\pi_\theta(y\mid x)
  -\log\pi_{\mathrm{ref}}(y\mid x).
$$

chosen 相对 rejected 的 log-ratio margin 为：

$$
m_\theta
  =\Delta_\theta(x,y_{\mathrm w})
   -\Delta_\theta(x,y_{\mathrm l}).
$$

DPO 单样本损失为：

$$
\mathcal L_{\mathrm{DPO}}
  =-\log\sigma(\beta m_\theta),
\qquad
\beta>0.
$$

$m_\theta>0$ 表示当前策略相对 reference 更偏向 chosen；$\beta$ 把这个相对偏好转换成 logistic logit。损失下降的含义是 chosen 的相对 log-ratio 增大，不等于回答事实正确、工具动作安全或用户结果改善。

| 对象 | 计算内容 | 训练时是否更新 |
|---|---|---|
| 当前策略 $\pi_\theta$ | chosen/rejected 的 log probability | 是 |
| 参考策略 $\pi_{\mathrm{ref}}$ | 同一回答的基准 log probability | 否，通常冻结 |
| 偏好对 | chosen 与 rejected 的相对方向 | 不更新 |
| $\beta$ | log-ratio margin 的尺度 | 配置项，可调 |
| DPO loss | 偏好方向的负对数似然 | 最小化 |

## 从 KL 正则 reward 推导 DPO

### KL 正则目标的最优分布

假设存在一个隐式 reward $r(x,y)$，策略优化目标为：

$$
J(\pi)
  =
  \mathbb E_{y\sim\pi(\cdot\mid x)}
  \left[
    r(x,y)
  \right]
  -
  \beta
  \operatorname{KL}
  \left(
    \pi(\cdot\mid x)
    \lVert
    \pi_{\mathrm{ref}}(\cdot\mid x)
  \right).
$$

对固定 $x$，用归一化约束求最优分布，得到：

$$
\pi^*(y\mid x)
  =
  \frac{1}{Z(x)}
  \pi_{\mathrm{ref}}(y\mid x)
  \exp\left(
    \frac{r(x,y)}{\beta}
  \right).
$$

取对数：

$$
r(x,y)
  =
  \beta
  \left[
    \log\pi^*(y\mid x)
    -\log\pi_{\mathrm{ref}}(y\mid x)
  \right]
  +\beta\log Z(x).
$$

对 chosen 和 rejected 相减，归一化项消失：

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

### Bradley–Terry 把 reward 差变成偏好概率

假设偏好标签满足：

$$
\Pr(y_{\mathrm w}\succ y_{\mathrm l}\mid x)
  =
  \sigma\left(
    r(x,y_{\mathrm w})-r(x,y_{\mathrm l})
  \right).
$$

用当前策略 $\pi_\theta$ 近似未知的最优策略 $\pi^*$，并把上式的 reward 差替换为 $\beta m_\theta$：

$$
\Pr_\theta(y_{\mathrm w}\succ y_{\mathrm l}\mid x)
  =
  \sigma(\beta m_\theta).
$$

对这条概率取负对数，得到 DPO loss。推导需要同时满足 KL 正则目标、固定 reference 和 Bradley–Terry 标签假设。DPO 省略显式 reward model，是因为 reward 差在成对损失中被 log-ratio 代数替换；它没有消除代理目标。

### reference 的归一化常数为什么可以消失

$Z(x)$ 只依赖输入，不依赖候选 $y$。DPO 使用同一个输入下的 chosen/rejected 差分，因此：

$$
\left[\beta\log Z(x)\right]
-
\left[\beta\log Z(x)\right]
=0.
$$

这也是 DPO 直接使用成对数据的原因。若比较来自不同输入，两个归一化常数不同，不能把它们的 log-ratio 差直接当成同一偏好 logit。

## 序列 log probability 是实现核心

### 条件概率分解到 response token

令 response 为 $y_{1:T}$：

$$
\log\pi_\theta(y\mid x)
  =
  \sum_{t=1}^{T}
  \log\pi_\theta
  \left(
    y_t\mid x,y_{<t}
  \right).
$$

在 batch 中通常加入 response mask $m_t$：

$$
\log\pi_\theta(y\mid x)
  =
  \sum_{t=1}^{T}
  m_t
  \log\pi_\theta
  \left(
    y_t\mid x,y_{<t}
  \right),
\qquad
m_t\in\{0,1\}.
$$

$m_t$ 应屏蔽 prompt、padding、EOS 之后的位置和无效截断位置。chosen 与 rejected 使用相同的角色模板和 mask 规则；reference 也必须看到同一份 token 序列。

### sum 和 mean 不是同一个 DPO 目标

序列 sum 使用完整的条件概率：

$$
\Delta_\theta^{\mathrm{sum}}(x,y)
  =
  \sum_t
  m_t
  \left[
    \log\pi_\theta(y_t\mid x,y_{<t})
    -
    \log\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})
  \right].
$$

若改成 token mean：

$$
\Delta_\theta^{\mathrm{mean}}(x,y)
  =
  \frac{
    \sum_t m_t
    \left[
      \log\pi_\theta(y_t\mid x,y_{<t})
      -
      \log\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})
    \right]
  }{
    \sum_t m_t
  }.
$$

长回答和短回答的分母不同，chosen/rejected 的 margin 可能因此改变方向。mean 可以降低长度对梯度尺度的影响，但它对应的不是原始序列概率比。实验必须明确使用 sum、chosen/rejected 各自 mean，还是统一 token budget 后再比较。

### 一个长度导致方向反转的例子

设 chosen 有三个 token，rejected 有两个 token。各 token 的 policy/reference 概率比的对数分别为：

$$
\begin{aligned}
\ell_{\mathrm w}
  &=(0.064538521138,\ 0.074107972154,\ 0.117783035656),\\
\ell_{\mathrm l}
  &=(0.087011376990,\ 0.105360515658).
\end{aligned}
$$

sum 结果为：

$$
\Delta_{\mathrm w}^{\mathrm{sum}}
  =0.256429528948,
\qquad
\Delta_{\mathrm l}^{\mathrm{sum}}
  =0.192371892647,
\qquad
m^{\mathrm{sum}}
  =0.064057636300.
$$

chosen 的 sum margin 为正。若分别取 mean：

$$
\Delta_{\mathrm w}^{\mathrm{mean}}
  =0.085476509649,
\qquad
\Delta_{\mathrm l}^{\mathrm{mean}}
  =0.096185946324,
\qquad
m^{\mathrm{mean}}
  =-0.010709436675.
$$

同一组 token log-ratio，sum 把 chosen 排在前面，per-response mean 把 rejected 排在前面。这个例子不是在选择一种普遍正确的 reduction，而是在说明 reduction 属于训练目标，不能作为实现细节静默替换。

## beta 和 label smoothing 改变梯度

### beta 改变 sigmoid 的工作区间

令 $z=\beta m_\theta$、$q=\sigma(z)$：

$$
\mathcal L=-\log\sigma(z).
$$

对 $z$ 求导：

$$
\frac{\partial\mathcal L}{\partial z}
  =q-1.
$$

当 $m_\theta$ 已经很大时，$q$ 接近 1，梯度变小；当 $m_\theta$ 接近 0 时，梯度约为 $-0.5$。增大 $\beta$ 会让同一 margin 更快进入 sigmoid 饱和区，同时提高 logit 对参数的缩放。

在前面的长度例子中，sum margin 为 0.064057636300，$\beta=0.05,0.1,0.2,0.5$ 时的偏好概率分别为 0.500800719769、0.501601435431、0.503202838007、0.508006520095。概率仍然接近 0.5，说明当前策略相对 reference 的方向很弱。

### label smoothing 不只是数据清洗

对 chosen 标签使用 $\varepsilon$ 的 label smoothing，可以写成：

$$
\mathcal L_{\mathrm{smooth}}
  =
  -(1-\varepsilon)\log\sigma(z)
  -\varepsilon\log(1-\sigma(z)).
$$

其对 logit 的导数为：

$$
\frac{\partial\mathcal L_{\mathrm{smooth}}}{\partial z}
  =\sigma(z)-(1-\varepsilon).
$$

$\varepsilon>0$ 把目标概率从 1 移向 $1-\varepsilon$，防止模型被要求把每个偏好对推到无限自信。它也会改变收敛点和损失数值，必须记录在实验配置中。

### reference-free 不等于没有 reference 假设

某些实现把 reference log probability 设为常数、使用同一个模型的初始 logits，或采用其他 reference-free 目标。这样会改变：

- log-ratio 的坐标；
- beta 的数值解释；
- 长度和模板偏差；
- 策略漂移的监测基准；
- 与 KL 正则 reward 推导的对应关系。

如果没有冻结 reference，应把方法标记为具体变体，不能继续把它解释成标准 DPO 的同一目标。

## 梯度经过策略的 softmax

### 对 log probability 项的梯度

令：

$$
z=\beta
  \left[
    \log\pi_\theta(y_{\mathrm w}\mid x)
    -\log\pi_{\mathrm{ref}}(y_{\mathrm w}\mid x)
    -\log\pi_\theta(y_{\mathrm l}\mid x)
    +\log\pi_{\mathrm{ref}}(y_{\mathrm l}\mid x)
  \right],
\qquad
q=\sigma(z).
$$

则：

$$
\frac{\partial\mathcal L}{\partial\log\pi_\theta(y_{\mathrm w}\mid x)}
  =\beta(q-1),
\qquad
\frac{\partial\mathcal L}{\partial\log\pi_\theta(y_{\mathrm l}\mid x)}
  =-\beta(q-1).
$$

这两个方向相反，reference 分支不接收梯度。实际网络中 chosen 和 rejected 共享参数，softmax、Transformer 表示、adapter 和 padding mask 会把两个序列的梯度叠加到同一组参数上。

### 二动作 softmax 的有限差分核对

若 policy logits 为 $(0.4,-0.1)$，reference logits 为 $(0.1,-0.2)$，$\beta=0.2$，标准库探针得到：

$$
\begin{aligned}
\pi_{\mathrm{ref}}
  &=(0.574442516812,\ 0.425557483188),\\
\pi_\theta
  &=(0.622459331202,\ 0.377540668798),\\
m_\theta
  &=0.200000000000,\\
q
  &=0.509998666880.
\end{aligned}
$$

解析的 policy logit 梯度为 $(-0.098000266624,0.098000266624)$，中心差分为 $(-0.098000266568,0.098000266679)$，最大误差为 $5.553\times10^{-11}$。这个检查验证了 loss 对策略 logits 的链式导数；它不验证真实 Transformer 的 attention mask、分词或 batch reduction。

## 一个可运行的 DPO 数值探针

下面的标准库探针先用不同长度的 chosen/rejected token 序列检查 sum/mean margin，再检查 beta、label smoothing、二动作 softmax 梯度和交换标签。它不依赖深度学习框架，输出可以作为实现的最小参考。

```python
import math


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def softmax(logits):
    maximum = max(logits)
    exponentials = [
        math.exp(value - maximum)
        for value in logits
    ]
    normalizer = sum(exponentials)
    return [
        value / normalizer
        for value in exponentials
    ]


policy_chosen = [0.8, 0.7, 0.9]
reference_chosen = [0.75, 0.65, 0.8]
policy_rejected = [0.6, 0.5]
reference_rejected = [0.55, 0.45]

chosen_token_log_ratios = [
    math.log(policy / reference)
    for policy, reference in zip(
        policy_chosen,
        reference_chosen,
    )
]
rejected_token_log_ratios = [
    math.log(policy / reference)
    for policy, reference in zip(
        policy_rejected,
        reference_rejected,
    )
]
chosen_sum = sum(chosen_token_log_ratios)
rejected_sum = sum(rejected_token_log_ratios)
sum_margin = chosen_sum - rejected_sum
print("chosen_token_log_ratios=", [
    f"{value:.12f}"
    for value in chosen_token_log_ratios
])
print("rejected_token_log_ratios=", [
    f"{value:.12f}"
    for value in rejected_token_log_ratios
])
print(
    "chosen_sum=",
    f"{chosen_sum:.12f}",
    "rejected_sum=",
    f"{rejected_sum:.12f}",
    "sum_margin=",
    f"{sum_margin:.12f}",
)
chosen_mean = chosen_sum / len(chosen_token_log_ratios)
rejected_mean = rejected_sum / len(rejected_token_log_ratios)
print(
    "chosen_mean=",
    f"{chosen_mean:.12f}",
    "rejected_mean=",
    f"{rejected_mean:.12f}",
    "mean_margin=",
    f"{chosen_mean - rejected_mean:.12f}",
)

for beta in (0.05, 0.1, 0.2, 0.5):
    logit = beta * sum_margin
    probability = sigmoid(logit)
    loss = -math.log(probability)
    print(
        "beta",
        beta,
        "logit",
        f"{logit:.12f}",
        "prob",
        f"{probability:.12f}",
        "loss",
        f"{loss:.12f}",
    )

beta = 0.2
logit = beta * sum_margin
probability = sigmoid(logit)
for smoothing in (0.0, 0.1, 0.2):
    loss = (
        -(1.0 - smoothing) * math.log(probability)
        - smoothing * math.log(1.0 - probability)
    )
    gradient = probability - (1.0 - smoothing)
    print(
        "eps",
        smoothing,
        "smoothed_loss",
        f"{loss:.12f}",
        "grad_z",
        f"{gradient:.12f}",
    )

reference = softmax([0.1, -0.2])
policy = softmax([0.4, -0.1])
beta = 0.2
two_action_margin = (
    math.log(policy[0])
    - math.log(reference[0])
    - math.log(policy[1])
    + math.log(reference[1])
)
probability = sigmoid(beta * two_action_margin)
loss = -math.log(probability)
gradient_chosen = beta * (probability - 1.0)
gradient_rejected = -gradient_chosen
print(
    "ref_probs=",
    [f"{value:.12f}" for value in reference],
    "policy_probs=",
    [f"{value:.12f}" for value in policy],
)
print(
    "two_action_margin=",
    f"{two_action_margin:.12f}",
    "prob=",
    f"{probability:.12f}",
    "loss=",
    f"{loss:.12f}",
    "analytic_logit_grad=",
    [f"{value:.12f}" for value in (
        gradient_chosen,
        gradient_rejected,
    )],
)

finite_difference = []
step_size = 1e-6
for index in range(2):
    plus_logits = [0.4, -0.1]
    minus_logits = [0.4, -0.1]
    plus_logits[index] += step_size
    minus_logits[index] -= step_size
    plus_policy = softmax(plus_logits)
    minus_policy = softmax(minus_logits)
    plus_margin = (
        math.log(plus_policy[0])
        - math.log(reference[0])
        - math.log(plus_policy[1])
        + math.log(reference[1])
    )
    minus_margin = (
        math.log(minus_policy[0])
        - math.log(reference[0])
        - math.log(minus_policy[1])
        + math.log(reference[1])
    )
    plus_loss = -math.log(sigmoid(beta * plus_margin))
    minus_loss = -math.log(sigmoid(beta * minus_margin))
    finite_difference.append(
        (plus_loss - minus_loss) / (2.0 * step_size)
    )
print(
    "finite_difference_grad=",
    [f"{value:.12f}" for value in finite_difference],
)
print(
    "max_grad_error=",
    f"{max(abs(numerical - analytic) for numerical, analytic in zip(
        finite_difference,
        (gradient_chosen, gradient_rejected),
    )):.3e}",
)
print(
    "swapped_prob=",
    f"{1.0 - probability:.12f}",
    "swapped_loss=",
    f"{-math.log(1.0 - probability):.12f}",
)
```

输出为：

```text
chosen_token_log_ratios= ['0.064538521138', '0.074107972154', '0.117783035656']
rejected_token_log_ratios= ['0.087011376990', '0.105360515658']
chosen_sum= 0.256429528948 rejected_sum= 0.192371892647 sum_margin= 0.064057636300
chosen_mean= 0.085476509649 rejected_mean= 0.096185946324 mean_margin= -0.010709436675
beta 0.05 logit 0.003202881815 prob 0.500800719769 loss 0.691547021958
beta 0.1 logit 0.006405763630 prob 0.501601435431 loss 0.689949427962
beta 0.2 logit 0.012811527260 prob 0.503202838007 loss 0.686761933693
beta 0.5 logit 0.032028818150 prob 0.508006520095 loss 0.677260996653
eps 0.0 smoothed_loss 0.686761933693 grad_z -0.496797161993
eps 0.1 smoothed_loss 0.688043086419 grad_z -0.396797161993
eps 0.2 smoothed_loss 0.689324239145 grad_z -0.296797161993
ref_probs= ['0.574442516812', '0.425557483188'] policy_probs= ['0.622459331202', '0.377540668798']
two_action_margin= 0.200000000000 prob= 0.509998666880 loss= 0.673347167228 analytic_logit_grad= ['-0.098000266624', '0.098000266624']
finite_difference_grad= ['-0.098000266568', '0.098000266679']
max_grad_error= 5.553e-11
swapped_prob= 0.490001333120 swapped_loss= 0.713347167228
```

## 偏好对的构造决定 DPO 的覆盖

### chosen/rejected 方向必须可追溯

一条偏好对应保存：

| 字段 | 作用 |
|---|---|
| prompt | 确定条件分布和任务上下文 |
| chosen/rejected | 固定 loss 的正负方向 |
| 生成策略 | 记录候选来源和难度 |
| 标注指南版本 | 解释选择标准 |
| 标注者/评审器版本 | 分离不同偏好分布 |
| 比较顺序 | 检查位置和展示偏差 |
| 平局/跳过状态 | 防止无效标签静默变成胜负 |
| tokenizer/template | 重建序列 log probability |

如果把 rejected 和 chosen 交换，margin 变为 $-m_\theta$，预测概率从 $q$ 变为 $1-q$。探针中原概率为 0.509998666880，交换后为 0.490001333120；交换标签却仍然使用原 loss，是数据管线的方向错误。

### 候选来源会把捷径写进策略

DPO 只能根据偏好对中出现的差异更新策略。如果 chosen 普遍更长，模型会同时看到长度和偏好方向；如果 rejected 都有格式错误，模型不会获得事实核验的监督；如果数据来自一个旧 checkpoint，当前策略产生的新失败模式没有被比较。

应按任务、难度、语言、长度、候选 checkpoint、安全状态和工具结果分层。训练 loss、margin、length、reference KL 和独立指标都按这些组报告。

### 迭代式 DPO 会改变数据分布

一次固定数据的 DPO 更新不需要当前策略 rollout。若训练后用新策略生成候选、重新标注、再加入数据，下一轮的数据分布已经改变。这个流程可以有效扩大覆盖，但需要保存每轮：

$$
\text{policy checkpoint},
\quad
\text{candidate source},
\quad
\text{label version},
\quad
\text{reference checkpoint},
\quad
\text{independent result}.
$$

没有这些版本字段，无法判断性能变化来自策略更新、候选生成器还是标注协议。

## DPO 与 RLHF 的边界

### DPO 删除了显式 reward model

| 组件 | RLHF + PPO | DPO |
|---|---|---|
| 偏好监督 | 训练 reward model | 直接进入 log-ratio loss |
| 策略数据 | rollout 可随策略变化 | 固定偏好对，或迭代式扩充 |
| value function | 通常需要估计 advantage | 不需要 |
| old policy | 生成当前 PPO batch | 基础 DPO 不需要 |
| reference policy | KL 惩罚基准 | log probability 基准 |
| 主要数值量 | reward、advantage、ratio、KL | margin、beta、loss、KL |
| 主要覆盖风险 | reward model 外推和 rollout 漂移 | 偏好对覆盖和 reference 偏差 |

DPO 更少中间模块，不代表它更接近真实意图。它仍然学习一个代理偏好目标，并且会优化数据和 reference 中可利用的统计差异。

### DPO loss 不能授予外部权限

DPO 只更新生成策略的参数。它不能决定是否允许发送邮件、执行代码、写入数据库或读取隐私数据。工具系统仍需检查身份、权限、参数、确认、沙箱、审计和回滚。

## 失效模式与审计清单

| 现象 | 可能原因 | 最小核验 |
|---|---|---|
| loss 快速接近 0 | beta 过大、标签过易或策略过拟合 | 分组 margin、留出偏好和独立结果 |
| chosen 长度持续增长 | sum reduction 或长度捷径 | 长度匹配、mean 对照和改写 |
| reference KL 很大 | reference 错载、beta 太小或模板不一致 | hash、模板和逐条 logprob |
| margin 与手算不符 | prompt/EOS/padding mask 错误 | 导出 token mask 和 token logprob |
| chosen/rejected 交换后结果不变 | 标签方向或数据读取错误 | 单条样本 swap test |
| reference-free 结果不稳定 | 基准分布未固定 | 记录初始策略和变体公式 |
| loss 下降但任务不变 | 偏好数据没有覆盖任务目标 | 任务、难度、语言分层评测 |
| 长回答 reward 或概率占优 | token 数和模板泄漏 | 长度匹配与字段消融 |
| 多轮 DPO 后退化 | 新数据来自策略分布外或 reference 漂移 | 按轮保存候选来源和 checkpoint |

一次最小实现审计应输出：chosen/rejected 的完整 token 序列、response mask、当前/reference token log probability、sum/mean reduction、margin、beta、loss、label smoothing、梯度有限差分、reference hash、tokenizer/template hash 和独立评测切片。

## 运行方法

将上一个 Python 代码块保存为 dpo-probe.py，再运行：

```bash
python3 dpo-probe.py
```

修改 token 概率、beta、label smoothing、response mask 或 reduction 后，重新核对 sum/mean margin、偏好概率、softmax 梯度和 swap test。接入真实模型时，先用一个短 prompt 导出两条回答的逐 token log probability，再扩大 batch。

## 相关词条

- [RLHF 与 DPO](../alignment/rlhf-dpo/)：比较在线 reward 优化和离线偏好优化的整体路径。
- [Bradley–Terry 模型](../alignment/bradley-terry/)：说明偏好对如何由 reward 差形成选择概率。
- [奖励模型](../alignment/reward-model/)：对照显式 reward model 与 DPO 的隐式 reward 差。
- [PPO](../alignment/ppo/)：对照 old policy、rollout、advantage 和 clipped surrogate。
- [策略梯度](../alignment/policy-gradient/)：连接 log probability 梯度与策略参数更新。
- [监督微调](../finetuning/sft/)：提供 DPO 常用的初始策略和 reference checkpoint。
- [对齐问题](../alignment/alignment-problem/)：区分偏好代理目标、真实意图和外部约束。
- [奖励投机](../alignment/reward-hacking/)：检查策略利用代理信号而独立结果下降。
- [训练稳定性](../pretraining/training-stability/)：审计 beta、梯度、数值精度和分布漂移。
