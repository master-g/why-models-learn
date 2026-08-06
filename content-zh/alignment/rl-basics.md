---
title: "强化学习基础：从 MDP 到最优策略"
tags: ["why-models-learn"]
---

强化学习把连续的状态、动作、奖励和状态转移写成一个决策过程，并学习一条使长期回报最大化的策略。监督学习为每个输入提供目标标签，强化学习通常只在动作产生结果后得到奖励；奖励可以延迟、随机或稀疏，因此策略必须同时处理探索、信用分配、状态估计和环境变化。本文从有限马尔可夫决策过程出发，推导回报、策略价值、Bellman 方程、最优性、时序差分和策略梯度的共同结构，再核对一个可运行的四状态 MDP。

![强化学习基础示意图：状态经过策略选择动作，环境返回奖励并转移到新状态，价值估计和策略更新使用轨迹中的长期回报](/assets/alignment/svg/rl-basics.1.svg)

## 先把交互写成马尔可夫决策过程

### 五个对象组成一个 MDP

有限马尔可夫决策过程可以写成：

$$
\mathcal M
=
(\mathcal S,\mathcal A,P,R,\gamma).
$$

$\mathcal S$ 是状态集合，$\mathcal A$ 是动作集合，$P$ 描述状态转移，$R$ 描述奖励，$\gamma$ 是折扣因子。初始状态还需要一个分布 $\rho_0$；如果只研究固定起点，可以令 $\rho_0$ 是一个点质量。

|对象|符号|定义|需要检查|
|---|---|---|---|
|状态|$S_t\in\mathcal S$|时刻 $t$ 可用于决策的信息|是否包含预测未来所需的历史|
|动作|$A_t\in\mathcal A$|智能体在状态下选择的操作|是否受权限和可行动作约束|
|转移|$P(s',r\mid s,a)$|执行动作后得到结果和下一状态的联合分布|是否随机、是否依赖隐藏状态|
|奖励|$R_{t+1}$|环境对动作结果的数值反馈|是否延迟、稀疏或可被投机|
|策略|$\pi(a\mid s)$|给定状态时选择动作的概率|行为策略和目标策略是否相同|
|折扣|$\gamma\in[0,1)$|未来奖励相对当前奖励的权重|任务是有限时域、折扣还是平均奖励|

在时刻 $t$，智能体先观察 $S_t$，再按策略抽取 $A_t$。环境随后产生 $R_{t+1}$ 和 $S_{t+1}$：

$$
S_t
\xrightarrow[\pi]{A_t}
(R_{t+1},S_{t+1}).
$$

奖励下标是 $t+1$，因为奖励通常在动作执行并产生结果后才可见。动作本身不是奖励；策略也不能直接指定下一状态，下一状态由环境转移决定。

### Markov 条件约束状态的含义

如果 $S_t$ 真的是状态，给定当前状态和动作后，过去的信息不再增加对下一步的预测：

$$
\begin{aligned}
&\Pr(S_{t+1}=s',R_{t+1}=r
\mid S_0,A_0,\ldots,S_t=s,A_t=a)\\
&\qquad
=\Pr(S_{t+1}=s',R_{t+1}=r\mid S_t=s,A_t=a).
\end{aligned}
$$

这个条件不是说环境没有历史，而是说历史中与未来有关的信息已经被状态保留。如果机器人只看到当前图像却不知道速度，当前图像可能不足以构成 Markov 状态。可以把历史堆叠、维护 belief state 或使用带记忆的策略；这些方法改变的是状态表示和策略输入，不会自动消除观测不确定性。

### 一条轨迹记录了策略和环境的联合结果

有限回合长度为 $T$ 时，一条轨迹可以写成：

$$
\tau
=
(S_0,A_0,R_1,S_1,\ldots,A_{T-1},R_T,S_T).
$$

在给定初始分布、策略和环境后，轨迹概率是：

$$
\Pr_\pi(\tau)
=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi(a_t\mid s_t)
P(s_{t+1},r_{t+1}\mid s_t,a_t).
$$

这个乘积说明一次回报同时受三部分影响：初始状态抽到了什么，策略选择了什么动作，环境对动作返回了什么结果。只复现策略而改变初始状态或环境版本，评估结果仍可能变化。

## 回报把即时奖励连接到长期目标

### 折扣回报是奖励的加权和

从时刻 $t$ 开始的折扣回报为：

$$
G_t
=
\sum_{k=0}^{T-t-1}
\gamma^kR_{t+k+1}.
$$

它满足递归关系：

$$
G_t
=
R_{t+1}
+
\gamma G_{t+1}.
$$

递归关系是 Bellman 推导的起点。当前动作的价值不仅由即时奖励决定，还由后续策略在下一状态能得到的回报决定。

例如轨迹奖励为 $(1,0,2)$：

$$
\begin{aligned}
G_0(\gamma=0)&=1,\\
G_0(\gamma=0.5)&=1+0.5\times0+0.5^2\times2=1.5,\\
G_0(\gamma=1)&=1+0+2=3.
\end{aligned}
$$

$\gamma$ 越小，目标越接近即时奖励；$\gamma$ 越接近 $1$，远期结果的权重越大。有限回合可以使用 $\gamma=1$，无限时域通常要求 $\gamma<1$ 或使用平均奖励定义。不能只改变 $\gamma$ 后比较回报数字，却不说明目标已经改变。

### 策略目标是初始状态上的期望回报

策略的目标可以写成：

$$
J(\pi)
=
\mathbb E_{\substack{S_0\sim\rho_0\\ \tau\sim\pi}}
\left[G_0\right].
$$

期望包含初始状态、策略抽样和环境转移的随机性。一次高回报轨迹只能提供一个样本，不能代表 $J(\pi)$ 的精确值。评估时要报告回合数量、随机种子、均值、方差或分位数。

如果任务包含多个回合，平均回报通常写成：

$$
\widehat J_N(\pi)
=
\frac1N\sum_{i=1}^{N}G_0^{(i)}.
$$

当回报有重尾或严重失败时，平均值可能掩盖尾部。成功率、最坏分组、约束违反率和下分位数需要与平均回报并列。

### 奖励尺度影响数值，不一定改变策略

如果所有奖励乘以正数 $c$：

$$
R'_{t+1}=cR_{t+1},
$$

则每条轨迹的回报也乘以 $c$，固定策略之间的排序不变：

$$
G'_t=cG_t.
$$

但是价值函数的数值、TD 误差、梯度尺度和优化器步长都会改变。奖励裁剪、只保留符号或给不同结果添加不同偏置则可能改变策略排序，不能称作单纯的数值归一化。

## 策略、价值函数和 Bellman 方程

### 策略把状态变成动作分布

随机策略定义为：

$$
\pi(a\mid s)
=
\Pr(A_t=a\mid S_t=s),
\qquad
\sum_{a\in\mathcal A}\pi(a\mid s)=1.
$$

确定性策略可以写成 $a=\mu(s)$，也可以视为概率集中在一个动作上的策略。随机策略在探索、环境随机性或多个动作价值相近时有实际作用，不等于模型没有决策。

### 状态价值函数平均后续回报

给定策略 $\pi$，状态价值函数定义为：

$$
V^\pi(s)
=
\mathbb E_\pi[G_t\mid S_t=s].
$$

动作价值函数还固定当前动作：

$$
Q^\pi(s,a)
=
\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

策略对当前动作取平均后得到：

$$
V^\pi(s)
=
\sum_{a\in\mathcal A}
\pi(a\mid s)Q^\pi(s,a).
$$

动作优势表示某个动作相对当前策略平均水平的差值：

$$
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s).
$$

$A^\pi(s,a)>0$ 表示在当前状态下，动作 $a$ 的期望回报高于策略平均；它不是一个与策略无关的动作标签。

### Bellman 期望方程拆出第一步

从 $G_t=R_{t+1}+\gamma G_{t+1}$ 出发，对动作和环境结果取期望：

$$
\begin{aligned}
V^\pi(s)
&=
\sum_a\pi(a\mid s)
\sum_{s',r}
P(s',r\mid s,a)
\left[r+\gamma V^\pi(s')\right],\\
Q^\pi(s,a)
&=
\sum_{s',r}
P(s',r\mid s,a)
\left[
r+\gamma\sum_{a'}
\pi(a'\mid s')Q^\pi(s',a')
\right].
\end{aligned}
$$

这两个方程是定义的自洽表达，不是某一个特定算法。给定完整的转移和奖励模型，可以解方程得到价值；只有采样轨迹时，需要用 Monte Carlo、时序差分或模型学习近似它们。

### 终止状态要截断未来价值

如果 $s'$ 是终止状态，后续没有奖励：

$$
G_t=R_{t+1},
\qquad
V^\pi(s')=0.
$$

实现中通常用 done 标志把 bootstrap 项写成：

$$
y_t
=
R_{t+1}
+
\gamma(1-d_t)V(S_{t+1}),
\qquad
d_t\in\{0,1\}.
$$

把时间截断误当作自然终止，会丢掉截断点之后的价值；把自然终止误当作普通状态，又会把不应存在的未来奖励传播回来。两种边界都要在数据中区分。

## 最优价值和策略改进

### Bellman 最优性方程选择最好的动作

最优状态价值是所有策略中能达到的最大期望回报：

$$
V^\star(s)
=
\max_\pi V^\pi(s).
$$

对应的 Bellman 最优性方程为：

$$
V^\star(s)
=
\max_{a\in\mathcal A}
\sum_{s',r}
P(s',r\mid s,a)
\left[
r+\gamma V^\star(s')
\right].
$$

最优动作价值满足：

$$
Q^\star(s,a)
=
\sum_{s',r}
P(s',r\mid s,a)
\left[
r+\gamma\max_{a'}Q^\star(s',a')
\right].
$$

若有多个动作达到同一最大值，最优策略不一定唯一。选择其中一个确定动作可以得到确定性最优策略，按多个最大动作随机化也可以保持相同价值。

### Bellman 算子提供收敛结构

定义最优 Bellman 算子：

$$
(\mathcal T^\star V)(s)
=
\max_a
\sum_{s',r}
P(s',r\mid s,a)
\left[r+\gamma V(s')\right].
$$

在无穷范数下，当 $0\le\gamma<1$ 时：

$$
\left\lVert
\mathcal T^\star V
-
\mathcal T^\star W
\right\rVert_\infty
\le
\gamma
\lVert V-W\rVert_\infty.
$$

它是一个压缩映射，因此有唯一不动点 $V^\star$，反复应用 $\mathcal T^\star$ 可以在表格有限 MDP 中收敛。神经网络训练不直接享有这个表格收敛保证；函数逼近、bootstrap、采样分布和目标移动会引入额外误差。

### 策略改进比较同一个状态下的动作

给定策略 $\pi$，定义贪心策略：

$$
\pi'(s)
\in
\operatorname*{arg\,max}_{a}
Q^\pi(s,a).
$$

策略改进定理说明，在满足标准 MDP 条件时，逐状态选择不低于当前策略价值的动作不会降低价值：

$$
V^{\pi'}(s)\ge V^\pi(s).
$$

这个关系说明了 policy iteration 的结构：先评估当前策略，再对价值做贪心改进，反复进行直到策略稳定。它不说明用有限样本和近似 Q 值时一定单调；估计误差可能让贪心动作实际更差。

## 一个可运行的四状态 MDP 探针

下面的 Python 标准库探针对一个有限 MDP 做 value iteration，再计算起始状态的三个动作价值、epsilon-greedy 概率、完整轨迹回报和一步 TD 更新。start 的 long 动作经过 route 后获得奖励，short 立即得到较小奖励，risky 以一定概率得到较大的即时奖励但也可能进入 trap。

```python
gamma = 0.9
states = ['start', 'route', 'goal', 'trap']
actions = {
    'start': ['long', 'short', 'risky'],
    'route': ['finish'],
    'goal': [],
    'trap': [],
}
transitions = {
    ('start', 'long'): [(1.0, 'route', 0.0)],
    ('start', 'short'): [(1.0, 'goal', 0.6)],
    ('start', 'risky'): [(0.4, 'goal', 1.4), (0.6, 'trap', 0.0)],
    ('route', 'finish'): [(1.0, 'goal', 1.0)],
}

def action_value(values, state, action):
    return sum(
        probability
        * (reward + gamma * values[next_state])
        for probability, next_state, reward
        in transitions[(state, action)]
    )

values = {state: 0.0 for state in states}
for iteration in range(100):
    next_values = values.copy()
    for state in states:
        if actions[state]:
            next_values[state] = max(
                action_value(values, state, action)
                for action in actions[state]
            )
    if max(
        abs(next_values[state] - values[state])
        for state in states
    ) < 1e-12:
        values = next_values
        break
    values = next_values

q_start = {
    action: action_value(values, 'start', action)
    for action in actions['start']
}
print('value_iteration_steps=', iteration + 1)
print('optimal_values=', {
    state: f'{values[state]:.6f}'
    for state in states
})
print('start_q_values=', {
    action: f'{q_start[action]:.6f}'
    for action in q_start
})
print('greedy_action=', max(q_start, key=q_start.get))

epsilon = 0.2
best = max(q_start, key=q_start.get)
probabilities = {
    action: epsilon / len(q_start)
    for action in q_start
}
probabilities[best] += 1.0 - epsilon
print('epsilon=', f'{epsilon:.1f}',
      'action_probabilities=', {
          action: f'{probabilities[action]:.6f}'
          for action in probabilities
      })

trajectory = [
    ('start', 'long', 0.0),
    ('route', 'finish', 1.0),
]
trajectory_return = sum(
    (gamma ** index) * reward
    for index, (_, _, reward) in enumerate(trajectory)
)
print('trajectory_return=', f'{trajectory_return:.6f}')

v_start = 0.4
v_route = 0.8
alpha = 0.25
reward = 0.0
td_error = reward + gamma * v_route - v_start
v_updated = v_start + alpha * td_error
print('td_error=', f'{td_error:.6f}',
      'updated_v_start=', f'{v_updated:.6f}')

risk_expected_reward = 0.4 * 1.4 + 0.6 * 0.0
print('risky_expected_immediate_reward=',
      f'{risk_expected_reward:.6f}')
```

运行输出：

```text
value_iteration_steps= 3
optimal_values= {'start': '0.900000', 'route': '1.000000', 'goal': '0.000000', 'trap': '0.000000'}
start_q_values= {'long': '0.900000', 'short': '0.600000', 'risky': '0.560000'}
greedy_action= long
epsilon= 0.2 action_probabilities= {'long': '0.866667', 'short': '0.066667', 'risky': '0.066667'}
trajectory_return= 0.900000
td_error= 0.320000 updated_v_start= 0.480000
risky_expected_immediate_reward= 0.560000
```

value iteration 在第 3 次更新后得到 $V^\star(\mathrm{start})=0.900000$、$V^\star(\mathrm{route})=1.000000$。long 的动作价值为 $0.900000$，超过 short 的 $0.600000$ 和 risky 的 $0.560000$；risky 的期望即时奖励也是 $0.560000$，但它的结果分布包含进入 trap 的概率。$\epsilon=0.2$ 时，最优动作仍以 $0.866667$ 的概率选择，两个探索动作各占 $0.066667$。

轨迹 start → route → goal 的回报是 $0.900000$，因为第一步奖励为零，第二步奖励需要乘以 $\gamma=0.9$。如果当前估计 $V(\mathrm{start})=0.4$、$V(\mathrm{route})=0.8$，一步 TD 误差为 $0.320000$，学习率 $\alpha=0.25$ 后更新值为 $0.480000$。这个更新没有等待未知的完整环境模型，而是用下一状态的当前估计进行 bootstrap。

## 探索和利用需要同时设计

### epsilon-greedy 保留动作覆盖

设当前动作价值估计为 $Q(s,A)=3$、$Q(s,B)=2$，两个动作等概率探索，$\epsilon=0.2$ 时：

$$
\pi(A\mid s)
=
(1-\epsilon)+\frac{\epsilon}{2}
=0.9,
\qquad
\pi(B\mid s)
=\frac{\epsilon}{2}
=0.1.
$$

如果动作数为 $\lvert\mathcal A\rvert$，唯一最优动作的概率是：

$$
\pi(a^\star\mid s)
=
1-\epsilon+\frac{\epsilon}{\lvert\mathcal A\rvert}.
$$

当动作空间增大时，探索概率会分摊到更多动作。$\epsilon$ 衰减过快会让未尝试动作没有数据，衰减过慢又会让部署回报长期受随机动作影响。衰减计划需要和环境风险、样本成本和覆盖目标一起记录。

### softmax 探索按价值差分配概率

另一种策略是：

$$
\pi_\tau(a\mid s)
=
\frac{\exp(Q(s,a)/\tau)}
{\sum_{a'}\exp(Q(s,a')/\tau)}.
$$

$\tau$ 较大时分布更平，较小时更接近贪心。计算时要先减去最大值以避免指数溢出：

$$
\pi_\tau(a\mid s)
=
\frac{\exp((Q(s,a)-m)/\tau)}
{\sum_{a'}\exp((Q(s,a')-m)/\tau)},
\qquad
m=\max_{a'}Q(s,a').
$$

softmax 的探索与 epsilon-greedy 不等价。前者按动作价值差分配概率，后者把探索概率平均给动作；两者的随机种子、温度或 epsilon 变化都会改变数据分布。

### 行为策略和目标策略可以不同

生成数据的行为策略记为 $\mu$，希望评估或优化的目标策略记为 $\pi$。如果 $\mu=\pi$，称为 on-policy；如果两者不同，称为 off-policy。off-policy 能复用历史数据或探索策略产生的轨迹，但需要处理分布差异。

重要性采样用概率比修正一条轨迹：

$$
\rho_{t:T}
=
\prod_{k=t}^{T}
\frac{\pi(A_k\mid S_k)}
{\mu(A_k\mid S_k)}.
$$

当行为策略给某个目标策略会选择的动作极低概率时，比例会很大，估计方差也会变大。若行为策略从未执行某个动作，分母为零，目标策略关于该动作的价值无法从这份数据直接识别。

## Monte Carlo 和时序差分使用不同的信息

### Monte Carlo 等回合结束再更新

在回合结束后，Monte Carlo 方法使用观察到的完整回报 $G_t$ 更新价值：

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha\left[G_t-V(S_t)\right].
$$

它不需要 bootstrap，因此目标不依赖当前的价值估计；代价是必须等待回合结束，而且完整回报可能方差很大。长回合、稀疏奖励和随机环境会放大这个问题。

### TD(0) 只 bootstrap 一步

TD(0) 使用：

$$
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t),
$$

然后：

$$
V(S_t)
\leftarrow
V(S_t)+\alpha\delta_t.
$$

它可以在线更新，目标方差通常较低，但价值估计自身的偏差会被传播。$\delta_t$ 不是“真实误差”的直接测量；它是当前估计下的一步自洽残差。

### n-step 和 eligibility trace 折中

n-step 回报把多个实际奖励和一个较远的 bootstrap 拼接：

$$
G_t^{(n)}
=
\sum_{k=0}^{n-1}
\gamma^kR_{t+k+1}
+
\gamma^nV(S_{t+n}).
$$

$n=1$ 是 TD(0)；$n$ 接近回合剩余长度时更接近 Monte Carlo。eligibility trace 进一步把近期状态的更新权重按时间衰减，形成 TD($\lambda$) 家族。改变 $n$ 或 $\lambda$ 会改变 bias、variance 和奖励传播速度，需要在相同轨迹与评估合同下比较。

## 策略梯度直接优化动作分布

### 对数导数技巧把回报变成梯度

参数化策略 $\pi_\theta$ 的目标为：

$$
J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[G_0\right].
$$

在满足可交换条件时，策略梯度可以写成：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[
G_0
\sum_{t=0}^{T-1}
\nabla_\theta
\log\pi_\theta(A_t\mid S_t)
\right].
$$

它不需要对离散动作取 argmax 的梯度，而是使用采样动作的 log probability。回报越高，增大这条动作路径概率的更新越强；回报为负时，更新方向相反。

### baseline 可以降低方差

从回报中减去一个不依赖当前动作的 baseline，不改变期望梯度：

$$
\nabla_\theta J(\theta)
=
\mathbb E
\left[
\sum_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\bigl(G_t-b(S_t)\bigr)
\right].
$$

取 $b(S_t)=V^\pi(S_t)$ 时，括号中的量近似 advantage。baseline 能降低方差，但如果价值估计不准确，仍会引入估计误差。Actor–critic 同时学习策略 actor 和价值 critic，就是把这个结构放进一个训练循环。

### 策略梯度的审计字段

至少保存：

|字段|回答的问题|
|---|---|
|behavior policy|轨迹由哪条策略产生|
|log probability|采样动作的概率是否可重算|
|return / advantage|更新使用哪种回报和 baseline|
|entropy coefficient|探索压力是否随训练变化|
|value loss|critic 是否跟上策略分布|
|terminal / truncation|未来价值是否正确截断|

不记录 log probability、mask 和 done 语义时，训练 loss 即使下降，也无法确认策略梯度是否来自预期的轨迹。

## 奖励设计和奖励塑形

### 稀疏奖励增加信用分配难度

如果只有终点奖励，早期动作的梯度或价值更新要经过很长路径才能获得信号。长时域还会让 $\gamma^k$ 快速减小，导致远期奖励的数值影响变弱。改善方法可以是增加探索、使用 n-step、引入模型或设计密集反馈，但每种方法都会改变方差、偏差或任务定义。

### 势函数塑形在条件满足时保持策略

给定势函数 $\Phi(s)$，塑形奖励写成：

$$
F(s,a,s')
=
\gamma\Phi(s')-\Phi(s).
$$

新奖励为：

$$
R'(s,a,s')
=
R(s,a,s')+F(s,a,s').
$$

沿一条长度为 $T$ 的轨迹累加时，塑形项望远镜相消：

$$
\sum_{t=0}^{T-1}
\gamma^tF(S_t,A_t,S_{t+1})
=
-\Phi(S_0)+\gamma^T\Phi(S_T).
$$

如果终止状态势函数和折扣边界满足条件，不同策略之间的排序可以保持；任意把距离、速度或完成中间步骤加入奖励，不具备这个保证。

### 奖励投机是目标和结果的分离

当模型发现提高 reward 的行为没有提高实际任务质量时，出现的是代理目标问题，不是“模型没有努力”。需要把奖励计算过程、可操纵字段、外部结果和未观测后果分开。[对齐问题](../alignment/alignment-problem/)把这个问题扩展到偏好反馈、工具动作和安全约束。

## 函数逼近带来新的稳定性问题

### 表格方法和神经网络不是同一个收敛条件

有限 MDP 中，表格 value iteration 使用 Bellman 压缩映射，理论结构清晰。神经网络价值函数写成 $V_\phi(s)$ 后，更新目标又依赖当前或滞后的网络：

$$
y_t
=
R_{t+1}
+
\gamma V_{\phi^-}(S_{t+1}),
$$

其中 $\phi^-$ 可能是当前参数或 target network 的延迟副本。优化器要同时面对 bootstrap 目标、相关样本和非平稳策略分布，表格收敛结论不能直接迁移。

### 三个因素会互相放大

|因素|具体问题|需要观察|
|---|---|---|
|bootstrapping|目标包含估计值的误差|TD error 和目标漂移|
|off-policy|数据策略与学习策略不同|动作覆盖和重要性比|
|function approximation|参数共享不同状态|分组误差和梯度范数|

三者同时出现时，某个状态的更新可能改变其他状态的预测，新的策略又改变后续采样分布。target network、replay buffer、保守估计、梯度裁剪和较小更新都只能缓解特定路径，不能代替任务级评估。

### 经验回放改变样本相关性

在线轨迹相邻样本高度相关。replay buffer 随机抽取历史转移可以降低相邻相关性，但也会让数据分布滞后于当前策略。缓冲区容量、优先级、去重、终止标志和旧策略比例都会影响训练。

如果 buffer 中的动作覆盖不足，随机打乱不会创造新信息。应记录每个状态区域、动作和时间段的覆盖率，并单独测试策略对未覆盖动作的外推。

## 状态、环境和数据范围

### 部分可观测时需要 belief 或记忆

观测 $O_t$ 不一定等于真实状态 $S_t$。部分可观测模型可以用历史条件分布表示：

$$
b_t(s)
=
\Pr(S_t=s\mid O_{0:t},A_{0:t-1}).
$$

如果策略只接收 $O_t$ 而不接收历史，它可能把两个需要不同动作的真实状态混在一起。RNN、Transformer memory 或显式 belief state 解决的是信息保留问题，不会自动解决奖励错设。

### 在线学习可以看到新结果

在线 RL 在环境中执行动作并获得新轨迹。它能够收集当前策略需要的数据，但试错可能有成本，环境也可能被动作改变。真实系统需要安全探索、速率限制、人工批准、回滚和异常终止。

### 离线学习受行为数据限制

离线 RL 只使用固定数据集：

$$
\mathcal D
=
\{(s_t,a_t,r_{t+1},s_{t+1},d_t)\}.
$$

数据由行为策略 $\mu$ 产生。对数据中没有出现的动作，$Q(s,a)$ 主要依赖函数逼近外推；如果外推过于乐观，策略会选择未验证动作。数据覆盖、行为策略、时间切分和 OOD 动作率应当成为验收指标。

### 模拟环境需要现实验证

模拟器可以降低样本成本，但摩擦、延迟、传感器缺失、用户反应和资源限制可能与现实不同。模拟器内的高回报应当被视为条件证据，部署前需要独立环境、扰动、长时运行和失败恢复测试。

## 评估一个强化学习策略

### 训练回报不是完整验收

固定训练环境和随机种子容易让策略适应某个窄分布。最小评测矩阵应包含：

|切片|目的|需要保存|
|---|---|---|
|训练环境|确认更新是否生效|训练回报和策略版本|
|新初始状态|检查起点泛化|种子、状态分布和成功率|
|新环境参数|检查动力学偏移|参数扰动和失败类型|
|新任务组合|检查规则泛化|任务构成和分组结果|
|长时运行|观察延迟风险|回报曲线、崩溃和恢复|
|高风险动作|检查安全边界|违反率、严重度和回滚|

### 报告回报的分布而不是单个最好值

对于 $N$ 个独立回合，报告：

$$
\bar G
=
\frac1N\sum_{i=1}^NG_i,
\qquad
s^2
=
\frac1{N-1}\sum_{i=1}^N(G_i-\bar G)^2.
$$

近似的均值标准误为 $s/\sqrt N$，但重尾、相关回合和自适应停止会破坏简单的独立同分布假设。要说明回合是否独立、是否筛掉失败、是否选择最好 checkpoint，以及是否把探索动作留在评估中。

### 约束指标要和回报分开

一个策略可以提高回报，同时增加碰撞、越权、隐私泄露、资源超限或不可逆动作。把约束违反率单独写成：

$$
C_j(\pi)
=
\mathbb E_\pi
\left[
\sum_{t=0}^{T-1}
\gamma^t c_j(S_t,A_t,S_{t+1})
\right].
$$

如果 $C_j$ 代表风险成本，验收应设置阈值 $C_j\le c_j^{\max}$。不能用回报的平均提升抵消高影响动作的硬约束违反。

## 失效模式与审计清单

### 定义和实现错误

|现象|优先检查|确认方法|
|---|---|---|
|回报数值与手算不符|奖励下标、折扣和终止|逐步打印 $R_{t+1}$、done 和 $G_t$|
|策略选择未来最优动作|目标状态的 bootstrap 或 mask|检查因果顺序和终止截断|
|价值在终止后继续增长|done 未进入目标|终止转移的未来项应为零|
|探索概率不符合配置|动作数、随机分布或 epsilon|固定 seed 直接统计动作频率|
|短路奖励胜过长期目标|gamma 或 reward scale|比较动作价值和完整轨迹回报|

### 数据和训练错误

|现象|优先检查|确认方法|
|---|---|---|
|训练回报上升、部署下降|初始状态或环境分布过窄|新种子、参数扰动和独立环境|
|Q 值极高但动作失败|未覆盖动作的外推|统计 action coverage 和 OOD 动作|
|TD error 持续爆炸|奖励尺度、bootstrap 和学习率|记录 target、value、梯度和裁剪|
|更新不稳定|off-policy、相关样本和目标漂移|对照 on-policy 或 target network|
|策略只会重复一条轨迹|探索衰减过快或 buffer 偏斜|动作熵、状态覆盖和重采样检查|

### 安全和部署错误

|现象|优先检查|确认方法|
|---|---|---|
|高回报伴随高风险|奖励遗漏真实代价|独立约束指标和最坏分组|
|模拟成功、现实失败|动力学和观测不一致|扰动、延迟、传感器和长时测试|
|错误动作无法撤销|缺少行动合同|确认、幂等键、日志和回滚演练|
|失败发生后仍继续探索|异常终止和权限边界缺失|故障注入与人工接管|

### 最小审计字段

保存 checkpoint、环境版本、初始状态分布、策略版本、行为策略、随机种子、gamma、reward contract、done/truncation、动作覆盖、回报分布、约束指标、日志和回滚结果。没有这些字段，训练曲线只能说明某次运行产生了某些数字，不能说明策略在什么合同下学到了什么。

## 运行方法

将上一个 Python 代码块保存为 rl-basics-probe.py，再运行 python3 rl-basics-probe.py。修改 gamma、transition、epsilon 或初始价值后，重新核对 value iteration、动作价值、轨迹回报和 TD 更新。

接入真实环境时，先固定状态和动作定义、奖励合同、终止与截断语义、初始状态分布和安全约束，再分别运行训练环境、新种子、分布外参数和高风险动作测试。保存每条轨迹的状态摘要、动作、奖励、done、策略版本和环境状态。

对策略更新做独立回归。比较 greedy、epsilon-greedy、softmax 和部署策略的动作分布，记录行为策略与目标策略的差异；对离线数据统计动作覆盖和 OOD 选择，不用 value estimate 单独证明未见动作安全。

## 相关词条

- [强化学习概览](../learning-framework/reinforcement-learning-overview/)：从交互轨迹、回报、价值和 TD 更新建立总览。
- [学习是什么](../learning-framework/what-is-learning/)：区分反馈信号、假设空间、算法和部署风险。
- [期望](../probability/expectation/)：理解策略价值和环境随机性的平均方式。
- [条件熵](../information-theory/conditional-entropy/)：连接部分可观测和状态不确定性。
- [对齐问题](../alignment/alignment-problem/)：检查奖励、代理目标、约束与真实行动后果的偏差。
- [策略梯度](../alignment/policy-gradient/)：展开策略 log probability 和回报如何形成更新。
- [奖励模型](../alignment/reward-model/)：说明偏好或规则反馈如何变成可优化分数。
- [训练稳定性](../pretraining/training-stability/)：检查 bootstrap、梯度、数值和恢复行为。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：检查策略记忆训练环境后的泛化边界。
