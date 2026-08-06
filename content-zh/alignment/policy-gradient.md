---
title: "策略梯度：用回报直接更新动作概率"
tags: ["why-models-learn"]
---

策略梯度把策略写成带参数的概率分布，直接沿着期望回报对参数的梯度更新，而不是先把每个状态的最优动作写成标签。高回报轨迹提高其中动作的概率，低回报轨迹降低其中动作的概率。梯度估计通常有较高方差，因此需要 reward-to-go、baseline、advantage、critic、熵正则和行为策略修正。本文从轨迹概率推导 REINFORCE 的共同公式，再连接 actor–critic、on-policy/off-policy、连续动作和实现审计。

![策略梯度示意图：策略根据状态采样动作，环境返回轨迹回报，log probability 与 advantage 形成 actor 梯度，critic 用 TD 误差估计价值](/assets/alignment/svg/policy-gradient.1.svg)

## 策略梯度优化什么

### 策略把参数变成动作分布

给定状态 $s$，参数化策略为：

$$
\pi_\theta(a\mid s)
=
\Pr_\theta(A_t=a\mid S_t=s).
$$

离散动作可以用 logits 经过 softmax 得到概率。令 logits 为 $\theta_1,\ldots,\theta_K$：

$$
\pi_\theta(a=j\mid s)
=
\frac{\exp(\theta_j)}
{\sum_{k=1}^{K}\exp(\theta_k)}.
$$

实际模型通常令 $\theta_j=f_\vartheta(s)_j$，因此 $\theta$ 既可以表示直接的策略参数，也可以表示网络输出的局部 logits。后文把所有可训练参数统一记为 $\vartheta$。

### 目标是期望回报

沿用 [强化学习基础](../alignment/rl-basics/) 中的轨迹记号：

$$
\tau
=
(S_0,A_0,R_1,\ldots,A_{T-1},R_T,S_T).
$$

折扣回报为：

$$
G_t
=
\sum_{k=0}^{T-t-1}
\gamma^kR_{t+k+1}.
$$

策略目标是：

$$
J(\vartheta)
=
\mathbb E_{\tau\sim\pi_\vartheta}
\left[G_0\right].
$$

策略梯度要估计的是：

$$
\nabla_\vartheta J(\vartheta),
$$

而不是某一条轨迹的回报本身。一次成功轨迹可能提供有用方向，但它不能代表目标的期望值。

### 策略更新是概率更新

使用学习率 $\eta$ 时，梯度上升写成：

$$
\vartheta_{k+1}
=
\vartheta_k
+
\eta\widehat{\nabla_\vartheta J}.
$$

如果实现使用最小化损失，通常定义：

$$
\mathcal L_{\mathrm{actor}}
=
-
\widehat{\mathbb E}
\left[
\log\pi_\vartheta(A_t\mid S_t)\widehat A_t
\right].
$$

负号把“最大化回报”改成“最小化 actor loss”。这里的 $\widehat A_t$ 可以是回报、TD error 或 advantage 估计；符号、detach 和 reduction 必须记录。

## 从轨迹概率推导 log-derivative

### 轨迹概率中只有策略部分依赖参数

给定初始分布和环境转移，轨迹概率为：

$$
\Pr_\vartheta(\tau)
=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\vartheta(a_t\mid s_t)
P(s_{t+1},r_{t+1}\mid s_t,a_t).
$$

环境动力学 $P$ 和初始分布 $\rho_0$ 不由 actor 参数直接控制。因此对数轨迹概率的梯度只保留策略项：

$$
\nabla_\vartheta\log\Pr_\vartheta(\tau)
=
\sum_{t=0}^{T-1}
\nabla_\vartheta
\log\pi_\vartheta(a_t\mid s_t).
$$

### log-derivative identity 把不可微抽样变成可估计梯度

期望目标可以展开为：

$$
J(\vartheta)
=
\sum_\tau
\Pr_\vartheta(\tau)G_0(\tau).
$$

对参数求导：

$$
\begin{aligned}
\nabla_\vartheta J(\vartheta)
&=
\sum_\tau
G_0(\tau)
\nabla_\vartheta\Pr_\vartheta(\tau)\\
&=
\sum_\tau
\Pr_\vartheta(\tau)G_0(\tau)
\nabla_\vartheta\log\Pr_\vartheta(\tau)\\
&=
\mathbb E_{\tau\sim\pi_\vartheta}
\left[
G_0
\sum_{t=0}^{T-1}
\nabla_\vartheta
\log\pi_\vartheta(A_t\mid S_t)
\right].
\end{aligned}
$$

第二行使用：

$$
\nabla_\vartheta p
=
p\nabla_\vartheta\log p.
$$

这一步不要求离散动作的抽样操作本身可微。它把“从当前策略采样”与“对采样动作的 log probability 求梯度”分开。

### reward-to-go 删除当前动作不可能影响的过去奖励

在时刻 $t$ 选择 $A_t$ 后，$R_1,\ldots,R_t$ 已经确定，不会因当前动作改变。对这些过去奖励乘以 score function 后，期望为零。因此可以把整条轨迹回报换成从当前时刻开始的 reward-to-go：

$$
\nabla_\vartheta J(\vartheta)
=
\mathbb E
\left[
\sum_{t=0}^{T-1}
\nabla_\vartheta
\log\pi_\vartheta(A_t\mid S_t)
G_t
\right].
$$

这一步不改变期望梯度，却通常降低估计方差。代码中的 return 必须和动作时间轴对齐；把 $G_0$ 复制到每个时间步是一个可运行的估计，但通常比 reward-to-go 噪声更大。

### 软最大化不是策略梯度的必要条件

策略梯度可以直接学习随机策略。随机性由 $\pi_\vartheta$ 的分布决定，不需要每一步执行 argmax。若把策略变成确定性 argmax，动作选择对 logits 的梯度会在边界处不连续，探索也会消失。确定性策略梯度是另一种设置，需要单独定义连续动作和状态分布。

## softmax 策略的梯度结构

### 单个 logit 的 score function

对离散 softmax 策略：

$$
\log\pi_\theta(a=i\mid s)
=
\theta_i
-
\log\sum_{k=1}^{K}\exp(\theta_k).
$$

因此：

$$
\frac{\partial
\log\pi_\theta(a=i\mid s)}
{\partial\theta_j}
=
\mathbf 1[j=i]
-
\pi_\theta(j\mid s).
$$

被采样动作对应的 logit 得到正项，所有类别都通过归一化项得到负的概率项。梯度在 logits 轴上的和为零：

$$
\sum_{j=1}^{K}
\frac{\partial\log\pi_\theta(a=i\mid s)}
{\partial\theta_j}
=0.
$$

因此同时给所有 logits 加同一个常数不会改变策略，也不会产生有效更新。

### 一个两动作例子

设 $\theta=(0.4,-0.2)$，奖励分别是 $r=(1.0,0.2)$。softmax 概率为：

$$
\pi
=
(0.645656306,0.354343694).
$$

期望回报为：

$$
J
=
0.645656306\times1.0
+
0.354343694\times0.2
=0.716525045.
$$

直接对 logits 求导得到：

$$
\nabla_\theta J
=
\left(
0.183027392,
-
0.183027392
\right).
$$

提高第一个动作的 logit 会提高高回报动作概率，降低第二个动作概率；两个坐标的梯度相反，符合 softmax 的平移不变性。

## 一个可运行的策略梯度探针

下面的 Python 标准库探针计算两动作 softmax 策略的期望回报、解析梯度、中心差分、baseline 梯度、熵正则和一个重要性采样比率。它不执行完整环境训练，作用是核对策略梯度的局部数学和实现字段。

```python
from math import exp, log

def softmax(logits):
    pivot = max(logits)
    weights = [exp(value - pivot) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]

def objective(logits, rewards):
    probabilities = softmax(logits)
    return sum(
        probability * reward
        for probability, reward in zip(probabilities, rewards)
    )

def gradient(logits, rewards):
    probabilities = softmax(logits)
    value = objective(logits, rewards)
    return [
        probability * (reward - value)
        for probability, reward in zip(probabilities, rewards)
    ]

def entropy(logits):
    return -sum(
        probability * log(probability)
        for probability in softmax(logits)
    )

theta = [0.4, -0.2]
rewards = [1.0, 0.2]
probabilities = softmax(theta)
value = objective(theta, rewards)
exact_gradient = gradient(theta, rewards)
h = 1e-6
finite_difference = []
for index in range(len(theta)):
    plus = theta[:]
    minus = theta[:]
    plus[index] += h
    minus[index] -= h
    finite_difference.append(
        (objective(plus, rewards) - objective(minus, rewards))
        / (2 * h)
    )

print('probabilities=', [
    f'{value:.9f}' for value in probabilities
])
print('expected_return=', f'{value:.9f}')
print('exact_gradient=', [
    f'{value:.9f}' for value in exact_gradient
])
print('finite_difference=', [
    f'{value:.9f}' for value in finite_difference
])
print('max_gradient_error=', f'{
    max(abs(a - b) for a, b in zip(
        exact_gradient, finite_difference
    )):.3e}')

baseline = 0.6
baseline_gradient = []
for coordinate in range(len(theta)):
    estimate = 0.0
    for action, (probability, reward) in enumerate(
        zip(probabilities, rewards)
    ):
        score = (
            (1.0 if action == coordinate else 0.0)
            - probabilities[coordinate]
        )
        estimate += probability * (reward - baseline) * score
    baseline_gradient.append(estimate)
print('baseline=', f'{baseline:.1f}',
      'baseline_gradient=', [
          f'{value:.9f}' for value in baseline_gradient
      ])

entropy_value = entropy(theta)
entropy_coefficient = 0.05
regularized = value + entropy_coefficient * entropy_value
print('entropy=', f'{entropy_value:.9f}',
      'regularized_objective=', f'{regularized:.9f}')

behavior = softmax([0.1, -0.3])
target = softmax([0.7, -0.1])
ratio_action0 = target[0] / behavior[0]
print('behavior_prob_action0=', f'{behavior[0]:.9f}',
      'target_prob_action0=', f'{target[0]:.9f}',
      'importance_ratio_action0=', f'{ratio_action0:.9f}')
```

运行输出：

```text
probabilities= ['0.645656306', '0.354343694']
expected_return= 0.716525045
exact_gradient= ['0.183027392', '-0.183027392']
finite_difference= ['0.183027392', '-0.183027392']
max_gradient_error= 1.223e-11
baseline= 0.6 baseline_gradient= ['0.183027392', '-0.183027392']
entropy= 0.650094167 regularized_objective= 0.749029753
behavior_prob_action0= 0.598687660 target_prob_action0= 0.689974481 importance_ratio_action0= 1.152478207
```

探针中，解析梯度与中心差分的最大绝对误差为 $1.223\times10^{-11}$。baseline 取 $0.6$ 后，期望梯度仍为 $(0.183027392,-0.183027392)$；baseline 改变单条样本的权重和方差，不改变 action-independent baseline 下的期望。策略熵为 $0.650094167$，熵系数为 $0.05$ 时正则化目标为 $0.749029753$。行为策略对 action 0 的概率为 $0.598687660$，目标策略概率为 $0.689974481$，重要性采样比率为 $1.152478207$。

## baseline 和 advantage 降低估计方差

### action-independent baseline 的期望梯度为零

对只依赖状态的 baseline $b(s)$：

$$
\begin{aligned}
&\mathbb E_{a\sim\pi_\vartheta}
\left[
\nabla_\vartheta\log\pi_\vartheta(a\mid s)b(s)
\right]\\
&\quad
=
b(s)\sum_a
\pi_\vartheta(a\mid s)
\nabla_\vartheta
\log\pi_\vartheta(a\mid s)\\
&\quad
=
b(s)\nabla_\vartheta
\sum_a\pi_\vartheta(a\mid s)
=0.
\end{aligned}
$$

因此：

$$
\mathbb E
\left[
\nabla_\vartheta\log\pi_\vartheta(A_t\mid S_t)
(G_t-b(S_t))
\right]
$$

与不减 baseline 的期望相同。baseline 不能依赖当前动作，否则一般会改变期望梯度。

### 状态价值是常用 baseline

取 $b(S_t)=V^\pi(S_t)$，得到 advantage：

$$
A^\pi(S_t,A_t)
=
Q^\pi(S_t,A_t)-V^\pi(S_t).
$$

策略梯度可以写成：

$$
\nabla_\vartheta J
=
\mathbb E
\left[
\sum_t
\nabla_\vartheta
\log\pi_\vartheta(A_t\mid S_t)
A^\pi(S_t,A_t)
\right].
$$

如果动作结果高于状态平均，advantage 为正，提高该动作概率；如果低于平均，advantage 为负，降低该动作概率。critic 的任务是估计 baseline 或 advantage，不是替 actor 直接选择动作。

### baseline 的拟合误差会进入估计

实际中 $V^\pi$ 未知，用参数化 critic $V_\phi$ 拟合。若 critic 欠拟合、使用了未来动作信息或没有正确处理终止，advantage 会带偏。应保存 critic loss、TD error、advantage 均值/方差和 actor 更新比率。

批量标准化 advantage：

$$
\widehat A'_t
=
\frac{\widehat A_t-\operatorname{mean}(\widehat A)}
{\operatorname{std}(\widehat A)+\epsilon}
$$

通常改变梯度尺度和优化轨迹。它可以改善数值稳定性，但不应被描述为保持原始目标完全不变；是否跨 batch 标准化、是否 stop-gradient 和 epsilon 取值都要记录。

## actor–critic 把策略和价值放进同一循环

### TD error 提供在线 advantage 近似

对于 value critic：

$$
\delta_t
=
R_{t+1}
+
\gamma(1-d_t)V_\phi(S_{t+1})
-
V_\phi(S_t).
$$

可以把 $\delta_t$ 作为一步 advantage 估计：

$$
\widehat A_t\approx\delta_t.
$$

actor 的最小化损失为：

$$
\mathcal L_{\mathrm{actor}}
=
-
\widehat{\mathbb E}
\left[
\log\pi_\vartheta(A_t\mid S_t)
\operatorname{stopgrad}(\widehat A_t)
\right].
$$

critic 的平方损失为：

$$
\mathcal L_{\mathrm{critic}}
=
\frac12
\widehat{\mathbb E}
\left[
(y_t-V_\phi(S_t))^2
\right],
$$

其中 $y_t$ 可以是一部 TD 目标或 n-step return。actor 更新时通常不让梯度穿过 advantage 估计回到 critic，除非实现明确采用了不同的可微路径。

### GAE 在偏差和方差之间调节

把多个 TD error 按 $\gamma\lambda$ 衰减：

$$
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda=0$ 接近一步 TD，$\lambda$ 接近 $1$ 更接近 Monte Carlo。增大 $\lambda$ 通常让回报信息传播更远，也可能增加方差；它不是无代价的“更准确”开关。

### actor 和 critic 的更新频率需要记录

同一个 batch 可以先更新 critic 再更新 actor，也可以交替多次更新。critic 多步更新会改变 advantage 的尺度，actor 多步更新会让旧轨迹远离当前策略。以下字段应当进入 checkpoint 和日志：

|字段|需要固定的内容|变化的影响|
|---|---|---|
|critic target|TD、n-step 或 Monte Carlo|bias 和 variance|
|stop-gradient|advantage 是否回传到 critic|梯度路径|
|actor/critic ratio|每批更新次数|数据新鲜度|
|value coefficient|critic loss 权重|策略和价值的相对步长|
|entropy coefficient|探索正则权重|动作分布的尖锐程度|

## 熵正则保持探索

### 熵衡量动作分布的不确定性

离散策略熵为：

$$
\mathcal H(\pi_\vartheta(\cdot\mid s))
=
-
\sum_a
\pi_\vartheta(a\mid s)
\log\pi_\vartheta(a\mid s).
$$

最大化回报并加入熵正则：

$$
J_{\mathrm{reg}}(\vartheta)
=
J(\vartheta)
+
\alpha
\mathbb E_{s}
\left[
\mathcal H(\pi_\vartheta(\cdot\mid s))
\right],
\qquad
\alpha\ge0.
$$

$\alpha$ 较大时鼓励分布保持平坦，可能提高探索；$\alpha$ 过大时会阻止策略在任务已经明确时集中到高价值动作。训练中应报告原始回报、熵和熵正则项，不能只看合并后的 objective。

### 熵和 epsilon-greedy 解决的问题不同

epsilon-greedy 直接混合贪心动作和随机动作；熵正则改变优化目标，使策略倾向于保持高熵。一个策略可以同时使用两者，也可以只使用其中一种。比较实验时要把实际动作分布而不只是配置字段记录下来。

## on-policy 和 off-policy 修正

### 目标策略和行为策略的概率比

轨迹由行为策略 $\mu$ 采样，但要估计目标策略 $\pi$ 时，单步重要性比率是：

$$
\rho_t
=
\frac{\pi(A_t\mid S_t)}
{\mu(A_t\mid S_t)}.
$$

一段轨迹的累计比率为：

$$
\rho_{t:T}
=
\prod_{k=t}^{T}
\rho_k.
$$

比率修正了分布差异，但多个比率相乘容易产生高方差。行为策略必须覆盖目标策略可能选择的动作；若 $\mu(a\mid s)=0$ 而 $\pi(a\mid s)>0$，该目标动作无法从这份数据被无偏估计。

### 重用旧轨迹会带来策略滞后

on-policy 更新通常使用当前或近似当前策略采样的数据。数据更匹配目标分布，但样本利用率低。off-policy 可以重用 replay buffer 或历史策略数据，但需要重要性采样、保守目标、策略约束或其他分布修正。

如果旧数据上的 actor loss 下降，不能直接说明当前策略在环境中变好。应同时记录行为策略版本、目标策略版本、KL 距离、动作概率比的均值/分位数和 OOD 动作比例。

### 概率比裁剪会改变优化目标

工程实现可能把 $\rho_t$ 限制在区间：

$$
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon).
$$

裁剪降低极端更新，但被裁剪的样本不再按原始策略梯度完全优化。PPO 等方法把裁剪、旧策略和 advantage 组合成专门目标，相关边界放在 [PPO](../alignment/ppo/) 词条中。不能把“用了概率比”简化为“已经无偏”。

## 连续动作的策略梯度

### 高斯策略输出均值和尺度

对连续动作，常见策略为：

$$
a\sim\mathcal N
\left(
\mu_\vartheta(s),
\operatorname{diag}(\sigma_\vartheta^2(s))
\right).
$$

每个动作坐标的 log probability 为：

$$
\log\pi_\vartheta(a\mid s)
=
-
\frac12
\left[
\frac{(a-\mu_\vartheta(s))^2}
{\sigma_\vartheta^2(s)}
+
2\log\sigma_\vartheta(s)
+
\log(2\pi)
\right].
$$

网络通常输出 $\log\sigma$ 以保证 $\sigma>0$。动作裁剪、tanh squashing、单位换算和环境边界会改变实际动作分布，log probability 必须包含对应的变换修正。

### reparameterization 和 score-function 是两条梯度路径

高斯动作可以写成：

$$
a
=
\mu_\vartheta(s)
+
\sigma_\vartheta(s)\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I).
$$

如果环境可微，可以通过动作路径反向传播；真实环境通常不可微，仍使用 log-derivative 的 score-function 估计。不要因为采样代码中出现了 reparameterization 就假设环境结果对参数可微。

## 奖励、约束和对齐边界

### 策略梯度只追踪给定的 reward

策略梯度最大化的是：

$$
J_R(\vartheta)
=
\mathbb E_{\pi_\vartheta}
\left[
\sum_t\gamma^tR_{t+1}
\right].
$$

如果 $R$ 只代表点击、完成或格式得分，梯度不会自动加入事实正确、隐私、权限或长期副作用。[对齐问题](../alignment/alignment-problem/)中的代理目标偏差在策略梯度中表现为：模型越来越擅长提高 reward，但独立意图指标没有同步提升。

### 约束可以放在策略外部

高风险动作可以使用：

$$
\max_\vartheta J_R(\vartheta)
\quad
\mathrm{s.t.}\quad
C_j(\pi_\vartheta)\le c_j^{\max}.
$$

约束也可以由环境拦截器、权限服务、人工确认、沙箱和回滚组成。把所有约束塞进一个标量 reward 便于优化，但更容易让严重违反被其他样本的回报抵消。训练后仍要独立报告约束违反率。

### gradient ascent 的方向必须和 loss 符号一致

如果使用：

$$
\mathcal L_{\mathrm{actor}}
=
-
\log\pi_\vartheta(a\mid s)\widehat A,
$$

则最小化 loss 等价于提高正 advantage 动作的概率。如果漏掉负号，或者把已经带负号的 advantage 再取负，更新方向会反过来。最小单元测试应固定一个两动作状态，确认高回报动作的 logit 上升。

## 实现合同和常见失效模式

### 数据字段要按时间轴对齐

|字段|形状或语义|审计问题|
|---|---|---|
|observation/state|$B\times T$ 或环境状态结构|是否包含决策时可见信息|
|action|$B\times T$|是否来自记录的 behavior policy|
|reward|$B\times T$|下标是否对应动作之后的结果|
|old log probability|$B\times T$|是否与采样时策略一致|
|value|$B\times T$|是否在相同 mask 和终止语义下计算|
|advantage/return|$B\times T$|是否正确处理 padding 和 truncation|
|done|$B\times T$|自然终止与时间截断是否区分|

### 计算图要分离 old policy、critic 和 actor

常见实现需要：

1. 用当前策略计算 new log probability；
2. 用保存的 old log probability 构造概率比；
3. 对 advantage 或 return 按合同 stop-gradient；
4. 对 padding、invalid action 和 terminal position 应用 mask；
5. 分别记录 actor loss、critic loss、entropy 和约束损失。

如果把 old log probability 重新用当前策略计算，概率比恒为一；如果 advantage 没有 detach，critic 可能通过 actor loss 获得非预期梯度；如果 padding 没有 mask，平均 loss 会被无效位置稀释。

### 失效模式

|现象|优先检查|确认方法|
|---|---|---|
|高回报动作概率下降|actor loss 符号、advantage 符号、action index|两动作固定回归测试|
|梯度方差极大|回报尺度、回合长度、baseline 和 batch|比较 reward-to-go 与 baseline|
|熵快速降到零|entropy coefficient、学习率、概率比|记录熵和动作频率|
|critic loss 下降但策略不升|value target、advantage detach、探索|独立评估回报与 advantage 分布|
|旧轨迹更新不稳定|behavior/target 概率比、数据过期|记录 ratio 分位数和策略版本|
|终止后价值继续传播|done/truncation mask|自然终止与时间截断单测|
|连续动作 log prob 错误|tanh、clip、单位和尺度修正|固定动作逐维核对 log probability|
|reward 上升但任务质量下降|reward proxy 遗漏约束|独立结果、尾部风险和人工复核|

## 运行方法

将上一个 Python 代码块保存为 policy-gradient-probe.py，再运行 python3 policy-gradient-probe.py。修改 logits、动作奖励、baseline、entropy coefficient 或 behavior/target policy 后，重新核对解析梯度、中心差分、熵和概率比。

接入真实环境时，先固定 observation、action、reward、done、truncation、discount、mask 和策略版本，再分别运行 reward-to-go、baseline、critic、entropy 和概率比的最小回归。保存每个时间步的 old/new log probability、return、advantage、value、entropy、梯度范数和动作分布。

把训练回报和独立任务结果分开。对 on-policy 与 off-policy 运行分别报告行为策略、目标策略、数据年龄、动作覆盖、概率比、约束违反率和多种随机种子的回报分布。

## 相关词条

- [强化学习基础](../alignment/rl-basics/)：定义 MDP、回报、价值、探索和 TD 更新。
- [强化学习概览](../learning-framework/reinforcement-learning-overview/)：从交互循环建立强化学习总览。
- [梯度下降](../training-nn/gradient-descent/)：说明参数沿目标梯度更新的基本规则。
- [随机梯度下降理论](../optimization-theory/stochastic-gradient-descent-theory/)：分析采样梯度的噪声和收敛条件。
- [奖励模型](../alignment/reward-model/)：说明反馈如何变成可优化的 reward。
- [对齐问题](../alignment/alignment-problem/)：审计 reward proxy 与真实意图、约束和行动后果的偏差。
- [训练稳定性](../pretraining/training-stability/)：检查 actor、critic、bootstrap 和恢复合同。
- [PPO](../alignment/ppo/)：展开旧策略、概率比与裁剪目标。
- [DPO](../alignment/dpo/)：比较不经过在线策略梯度的偏好优化路径。
