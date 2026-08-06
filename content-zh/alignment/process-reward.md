---
title: "过程奖励：把最终结果的监督分配到步骤"
tags: ["why-models-learn"]
---

过程奖励在轨迹或推理序列的中间步骤上提供反馈，过程奖励模型（Process Reward Model，PRM）据此判断前缀是否保持在可行路径上；结果奖励（Outcome Reward，ORM）只在最终答案、任务成功或终止状态上提供反馈。过程奖励可以缩短信用分配路径、帮助搜索尽早剪枝，却会引入步骤边界、局部正确性、标签成本、奖励聚合和未来信息泄漏等新合同。本文从前缀价值和轨迹回报开始，比较 outcome/process 两类信号，推导潜在函数塑形的条件，说明 PRM 如何进入搜索与策略更新，再用标准库探针核对隐藏错误步骤、聚合方式和折扣奖励。

![过程奖励示意图：同一条轨迹的每个步骤获得局部评分，PRM 与最终 outcome reward 分开记录，搜索器据此保留前缀并由独立结果验收](/assets/alignment/svg/process-reward.1.svg)

## 结果奖励和过程奖励记录不同时间尺度

### 轨迹、前缀和终止结果

给定输入 $x$，模型生成步骤序列：

$$
\tau
 =
 \left(
   z_1,z_2,\ldots,z_T
 \right),
$$

第 $t$ 步的前缀为：

$$
h_t
 =
 \left(
   x,z_1,\ldots,z_t
 \right).
$$

结果奖励只读取终止状态或最终答案：

$$
R_{\mathrm O}(\tau)
 =
 R_{\mathrm O}(x,z_{1:T}).
$$

过程奖励在每一步产生：

$$
r_t
 =
 r_{\mathrm P}(h_{t-1},z_t),
\qquad
t=1,\ldots,T.
$$

把过程奖励聚合成轨迹分数的一种形式是：

$$
R_{\mathrm P}(\tau)
 =
 \sum_{t=1}^{T}
 \gamma^{t-1}r_t,
\qquad
0\leq\gamma\leq1.
$$

这里的 $r_t$ 可以表示步骤正确、前缀可行、引用有效、工具参数合规或风险降低。它不自动代表最终任务成功。

|信号|观察位置|优势|主要缺口|
|---|---|---|---|
|outcome reward|最终答案或终止状态|定义直接、标注目标清晰|信用分配稀疏、错误位置不明确|
|process reward|每个步骤或前缀|能定位错误、帮助搜索剪枝|步骤边界和局部标签成本高|
|terminal constraint|终止时检查安全、权限和外部状态|可以拦截不可接受结果|中间错误可能已改变轨迹|
|step constraint|每一步检查动作和状态|提前阻止危险路径|检查器需要读取中间状态|

### 两类 reward 可以同时存在

一个带终止条件的总回报可以写成：

$$
G(\tau)
 =
 \sum_{t=1}^{T}
 \gamma^{t-1}r_t
 +
 \gamma^{T}
 R_{\mathrm O}(\tau).
$$

系数、折扣和 terminal reward 的尺度决定过程信号与结果信号的相对影响。若过程分数尺度未校准，模型可能优化大量小的局部分数，忽略一次终止失败。

应把原始信号与聚合分数同时保存：

|字段|作用|
|---|---|
|step text / token span|确定步骤边界和可复现输入|
|step reward|记录每一步的原始评分|
|terminal outcome|记录最终正确、失败、超时或外部状态|
|mask / length|区分有效步骤、padding 和被截断轨迹|
|aggregation|记录 sum、mean、min、product 或 discount|
|source / version|固定 PRM、ORM、规则和 checkpoint|

## 过程奖励模型估计前缀的未来价值

### 前缀价值是自然目标

在策略 $\pi$ 下，前缀 $h_t$ 的结果价值为：

$$
V^\pi(h_t)
 =
 \mathbb E_{\tau\sim\pi}
 \left[
   R_{\mathrm O}(\tau)
 \mid h_t
 \right].
$$

如果最终成功用 $1$ 表示、失败用 $0$ 表示，$V^\pi(h_t)$ 就是从该前缀继续采样最终成功的概率：

$$
V^\pi(h_t)
 =
 \Pr_{\tau\sim\pi}
 \left(
   \mathrm{success}
 \mid h_t
 \right).
$$

PRM 可以近似每个前缀的局部正确性，也可以近似这个成功概率。两者需要区分：

|PRM 输出|解释|使用方式|
|---|---|---|
|step correctness|当前步骤是否满足局部规则|定位错误、构造 dense reward|
|prefix value|从当前前缀继续的成功概率|搜索排序、提前剪枝|
|risk score|前缀触发某个风险的概率|安全拦截、人工复核|
|confidence|评价器对上述判断的确定程度|阈值、拒答和校准|

局部正确步骤可能进入错误的全局假设；局部看似无关的步骤可能对最终答案有帮助。PRM 的输出语义必须写进标签协议。

### 过程标签的来源

步骤标签可以来自：

- 人工逐步核验：直接判断每步是否正确或可接受。
- 最终结果回溯：用可验证答案把部分前缀标为成功路径。
- 程序验证器：检查方程、类型、单元、代码测试或工具状态。
- 未来 rollout：从前缀继续采样多条轨迹，估计成功比例。
- AI 评价器：按规则批评和评分，再用独立样本校准。

标签来源不同，错误结构也不同。最终成功回溯可能把早期错误隐藏在后续修正中；程序验证器可能只覆盖形式正确；AI 评价器可能偏好更像标准解的格式。

## 信用分配和聚合方式会改变策略

### 终止 reward 的信用分配

若只有终止 reward $R_{\mathrm O}$，策略梯度可以写成：

$$
\nabla_\theta J
 =
 \mathbb E
 \left[
   R_{\mathrm O}(\tau)
   \sum_{t=1}^{T}
   \nabla_\theta
   \log\pi_\theta(z_t\mid h_{t-1})
 \right].
$$

同一个终止结果会把相同符号的信号乘到整条轨迹。长轨迹中，错误步骤和有用步骤共享信用；方差也会随着决策数量增加。

过程 advantage 可以缩短反馈路径：

$$
A_t
 =
 Q(h_{t-1},z_t)
 -
 V(h_{t-1}).
$$

若 PRM 提供的 $r_t$ 与未来成功相关，$A_t$ 更容易定位需要增加或降低概率的步骤。若 PRM 只反映格式，策略会把格式步骤的概率提高。

### sum、mean、min 和 product 的差异

给定步骤分数 $s_1,\ldots,s_T$，常见聚合方式为：

$$
S_{\mathrm{sum}}
 =
 \sum_{t=1}^{T}s_t,
 \qquad
S_{\mathrm{mean}}
 =
 \frac{1}{T}\sum_{t=1}^{T}s_t,
$$

$$
S_{\mathrm{min}}
 =
 \min_t s_t,
\qquad
S_{\mathrm{prod}}
 =
 \prod_{t=1}^{T}s_t.
$$

sum 偏向长轨迹，mean 把长度影响压低，min 对单个坏步骤敏感，product 把每个步骤都当成连续成功概率的近似。它们表达不同的任务假设。

|聚合|偏好|适用风险|
|---|---|---|
|sum|步骤多且每步都增加有效进展|冗余步骤获得额外 reward|
|mean|比较不同长度的平均步骤质量|一个关键坏步骤被平均掉|
|min|任何步骤失败都严重|误报一个局部低分就丢弃整条轨迹|
|product|每步分数可解释为条件成功概率|长轨迹概率快速变小、数值下溢|
|discounted sum|早期步骤权重更高或未来不确定|折扣改变真实任务优先级|

聚合方式必须和训练、搜索、验收保持一致。只在训练时使用 min、部署时只看 sum，会造成分布和选择口径不一致。

### 隐藏错误的数值例子

考虑三条四步轨迹：

|轨迹|步骤分数|最终结果|
|---|---|---:|
|clean|0.90、0.85、0.80、0.90|1|
|hidden error|0.95、0.95、0.20、0.95|0|
|slow valid|0.70、0.70、0.70、0.70|1|

clean 的 mean 为 $0.8625$，hidden error 的 mean 为 $0.7625$，slow valid 的 mean 为 $0.7000$。mean 能把 hidden error 排在 clean 后面，却仍然给它高于 slow valid 的分数。min 分别为 $0.80$、$0.20$、$0.70$，它能暴露隐藏错误，但也会把一个局部低分当成终止失败。

如果步骤分数代表条件成功概率，product 更接近整条路径的成功概率：

$$
S_{\mathrm{prod}}(\mathrm{clean})
 =
0.550800,
\qquad
S_{\mathrm{prod}}(\mathrm{hidden\ error})
 =
0.166963.
$$

这些数字依赖评分语义。PRM 输出是启发式分数时，不能把 product 自动解释成概率。

## Potential-based shaping 有明确的保持条件

### 潜在函数差分

在 Markov 状态 $s$、动作 $a$、下一状态 $s'$ 上，定义潜在函数 $\Phi$，塑形项为：

$$
F(s,a,s')
 =
 \gamma\Phi(s')
 -
 \Phi(s).
$$

总 reward 为：

$$
r'(s,a,s')
 =
r(s,a,s')
 +
F(s,a,s').
$$

沿一条长度为 $T$ 的轨迹，折扣塑形项之和为：

$$
\sum_{t=0}^{T-1}
\gamma^tF_t
 =
-\Phi(s_0)
 +
\gamma^T\Phi(s_T).
$$

中间潜在值发生了望远镜消去。若终止状态的潜在值固定，所有策略的回报只差一个由初始状态决定的常数，最优策略保持不变。

### 语言模型前缀不自动满足条件

语言模型的前缀可以作为状态，但需要检查：

- 前缀是否包含后续任务所需的完整状态。
- $\Phi$ 是否只依赖当前状态，而没有读取未来答案。
- 终止条件和 EOS 是否一致。
- padding、截断、工具外部状态是否进入状态。
- reward 是否在 beam、sample 和 rollout 之间使用同一尺度。

如果 PRM 直接读取最终答案、隐藏标签或未来 token，它可能泄漏未来信息。若状态不满足 Markov 近似，potential-based 形式也无法自动保持原始策略排序。

### 一个数值核对

设 $\gamma=0.9$，基础转移 reward 为 $(0,0,1)$，潜在值为：

$$
\Phi(s_0,s_1,s_2,s_3)
 =
(0.2,0.5,0.8,0).
$$

塑形项为：

$$
F
 =
(0.25,0.22,-0.80).
$$

基础折扣回报为 $0.81$，塑形项折扣和为 $-0.20$，总回报为 $0.61$。差值等于 $-\Phi(s_0)+\gamma^3\Phi(s_3)=-0.20$。这是一个轨迹级常数；如果不同终止状态的 $\Phi$ 不同，策略排序可能变化。

## PRM 如何进入搜索

### 前缀扩展和剪枝

对当前 beam 中的前缀 $h_t$ 扩展候选动作 $z_{t+1}$，搜索分数可以写成：

$$
S(h_{t+1})
 =
 S(h_t)
 +
 \lambda r_{\mathrm P}(h_t,z_{t+1})
 +
 \mu \log\pi_\theta(z_{t+1}\mid h_t).
$$

beam search、best-of-N、树搜索或 MCTS 都可以使用这个分数。$\lambda$ 和 $\mu$ 决定过程 reward 与策略概率的相对影响。

搜索器需要同时保存：

|字段|作用|
|---|---|
|prefix|重建当前推理或动作路径|
|parent|回溯搜索树和共享前缀|
|policy log probability|区分模型偏好和 PRM 偏好|
|process score|定位局部错误和排序依据|
|terminal status|区分完成、失败、截断和未评估|
|pruned reason|解释被删除的候选|

### 过程评分器会决定探索范围

若剪枝阈值过高，搜索器会删除需要暂时低分但后续可修正的路径。若阈值过低，搜索树增长和评测成本会迅速增加。

可以用保留率和最终成功率评估剪枝：

$$
\operatorname{recall}_{\mathrm{prefix}}
 =
 \frac{
   \text{被保留且最终成功的前缀数}
 }{
   \text{所有最终成功轨迹中的前缀数}
 }.
$$

只报告搜索后最终成功率会掩盖 PRM 已删除的有效路径。需要保存被剪掉的候选，至少保留一部分做反事实回放。

### 工具任务需要过程状态

在多步工具任务中，步骤 reward 可以检查：

- 工具名称和参数是否符合权限合同。
- 上一步返回值是否被正确解析。
- 当前动作是否依赖不存在的外部状态。
- 资源预算、幂等性和确认状态是否满足。
- 执行后状态是否与预期变化一致。

文本上看起来合理的步骤，可能没有改变外部系统；PRM 读取工具日志和独立状态比只读取模型文本更可靠。

## 过程奖励的标签和训练风险

### 步骤边界是模型外的协议

同一个回答可以按句子、方程行、代码块、工具调用或 token 分段。分段方式改变标签数量和 credit assignment：

|边界|优点|问题|
|---|---|---|
|句子|易于人工阅读|一个句子可能包含多个逻辑动作|
|方程/推导行|适合数学验证|跨行变量依赖难以单独判断|
|代码块/工具调用|适合执行状态|代码内部错误被聚合|
|token|粒度细、可用于 dense reward|标签成本和噪声高|
|模型自定义 step|贴合生成协议|不同 checkpoint 的边界可能不一致|

训练数据需要保存边界定义、token span、padding mask 和截断位置。边界变更等同于改变 reward 目标。

### PRM 可能学习格式捷径

过程奖励模型可能把以下特征当成步骤质量：

- “因此”“显然”“检查完毕”等模板词。
- 公式排版、步骤编号和固定解题句式。
- 更长的解释、更多的中间变量或免责声明。
- 与训练标签相似的答案结构。

应做语义固定的干预：保留同一逻辑，只改变措辞、顺序、长度和格式；同时报告 PRM 分数、最终结果和独立人工判断。

### 过程 reward 也会被投机

策略可以：

- 增加容易获得局部分数的步骤。
- 把错误隐藏在 PRM 看不到的工具状态或长上下文中。
- 生成大量低信息步骤，抬高 sum reward。
- 先输出符合规则的前缀，再在后续步骤改变目标。
- 利用 PRM 的阈值、tie-break 或截断逻辑。

因此 process reward 需要和 reward hacking 审计使用同一套代理/效用/约束账本。

## 过程奖励、结果奖励和搜索的组合

### 组合方式

可用的组合包括：

$$
R_{\mathrm{total}}
 =
 \alpha R_{\mathrm P}
 +
 (1-\alpha)R_{\mathrm O},
$$

$$
R_{\mathrm{gated}}
 =
 \mathbf 1
 \left[
   R_{\mathrm P}\geq\tau_{\mathrm P}
 \right]
 R_{\mathrm O},
$$

$$
R_{\mathrm{lex}}
 =
 \left(
   \operatorname{min}_t s_t,
   R_{\mathrm O}
 \right),
$$

其中最后一种表示先比较最弱步骤，再比较终止结果。每种组合对失败类型的容忍范围不同。

### 评测矩阵

|配置|过程信号|结果信号|检查重点|
|---|---|---|---|
|ORM only|无|有|终止结果、信用方差|
|PRM only|有|无或弱|局部捷径、全局失败|
|PRM + ORM|有|有|尺度、权重、冲突|
|PRM + verifier|有|程序/工具检查|验证覆盖、执行状态|
|PRM + search|有|终止重排|剪枝召回、搜索成本|
|PRM + PPO/RFT|作为训练 reward 或数据过滤|作为终点验收|策略是否写入局部捷径|

### 独立结果必须保持可见

训练曲线至少绘制：

$$
\left(
 \mathbb E[R_{\mathrm P}],
 \mathbb E[R_{\mathrm O}],
 \Pr(\mathrm{constraint\ violation}),
 \operatorname{coverage},
 \operatorname{cost}
\right).
$$

PRM loss 下降说明过程标签被拟合。它不说明最终任务成功率提高，也不说明工具状态或安全边界满足。

## 失效模式与审计清单

|现象|原因候选|最小核验|
|---|---|---|
|PRM 分数上升、最终成功率不变|步骤格式捷径或局部目标错误|语义固定改写、独立结果|
|某一步低分导致所有路径被剪掉|阈值过高或局部错误不可修正假设错误|保留剪枝样本、prefix recall|
|sum reward 偏好更长轨迹|每步小分累积、重复没有惩罚|mean/min 对照、长度分层|
|product reward 对长轨迹接近零|分数被当作概率但未校准|概率校准、对数聚合|
|PRM 判断正确、ORM 判断失败|局部步骤合理但全局目标错误|终止核验、后续 rollout|
|ORM 成功、步骤错误难以定位|终止反馈稀疏|人工逐步标注、回溯和程序验证|
|搜索成功率提高、有效路径召回下降|PRM 剪枝删除困难但正确前缀|prefix recall、反事实扩展|
|多轮训练后步骤模板固定|PRM 依赖格式和关键词|格式/语义干预、换评价器|
|工具文本正确、外部状态错误|PRM 只读文本没有读状态|动作前后状态、权限和日志|
|shape 或 mask 改变后分数异常|step boundary、padding 或 EOS 合同不一致|token span、mask、逐步重放|

### 最小审计输出

一次可复现的过程奖励实验应保存：

- 输入、完整轨迹、前缀、步骤边界、token span 和 padding mask。
- 每步 PRM score、聚合方式、terminal outcome、约束结果和独立评测。
- PRM/ORM/verifier 版本、提示模板、标签来源、校准切片和失败原因。
- 搜索树中的保留前缀、剪枝前缀、parent、policy log probability 和阈值。
- 训练的 reward scale、discount、lambda、beta、reference、checkpoint 和回滚点。
- 步骤长度、格式、语言、工具状态、错误类别和最终成功率的分组结果。

## 一个可运行的最小探针

下面的探针比较三条四步轨迹。hidden error 的前两步分数很高、第三步出现低分；它的最终 outcome 为失败。探针还核对 potential-based shaping 的望远镜差分。

```python
import math

paths = [
    {
        "name": "clean",
        "step_scores": (0.90, 0.85, 0.80, 0.90),
        "outcome": 1,
    },
    {
        "name": "hidden_error",
        "step_scores": (0.95, 0.95, 0.20, 0.95),
        "outcome": 0,
    },
    {
        "name": "slow_valid",
        "step_scores": (0.70, 0.70, 0.70, 0.70),
        "outcome": 1,
    },
]


for path in paths:
    scores = path["step_scores"]
    mean_score = sum(scores) / len(scores)
    product_score = math.prod(scores)
    print(
        "path=",
        path["name"],
        "mean=",
        f"{mean_score:.6f}",
        "min=",
        f"{min(scores):.6f}",
        "product=",
        f"{product_score:.6f}",
        "outcome=",
        path["outcome"],
    )

print(
    "mean_rank=",
    [
        path["name"]
        for path in sorted(
            paths,
            key=lambda path: sum(path["step_scores"])
            / len(path["step_scores"]),
            reverse=True,
        )
    ],
)
print(
    "min_rank=",
    [
        path["name"]
        for path in sorted(
            paths,
            key=lambda path: min(path["step_scores"]),
            reverse=True,
        )
    ],
)

gamma = 0.9
base_rewards = (0.0, 0.0, 1.0)
potential = (0.2, 0.5, 0.8, 0.0)
shaping = tuple(
    gamma * potential[index + 1] - potential[index]
    for index in range(len(base_rewards))
)
discounted_base = sum(
    (gamma**index) * reward
    for index, reward in enumerate(base_rewards)
)
discounted_shaping = sum(
    (gamma**index) * reward
    for index, reward in enumerate(shaping)
)
print(
    "shaping=",
    [f"{reward:.6f}" for reward in shaping],
    "discounted_base=",
    f"{discounted_base:.6f}",
    "discounted_shaping=",
    f"{discounted_shaping:.6f}",
    "shaped_total=",
    f"{discounted_base + discounted_shaping:.6f}",
)
print(
    "telescoping_difference=",
    f"{-potential[0] + (gamma**len(base_rewards)) * potential[-1]:.6f}",
)
```

输出为：

```text
path= clean mean= 0.862500 min= 0.800000 product= 0.550800 outcome= 1
path= hidden_error mean= 0.762500 min= 0.200000 product= 0.171475 outcome= 0
path= slow_valid mean= 0.700000 min= 0.700000 product= 0.240100 outcome= 1
mean_rank= ['clean', 'hidden_error', 'slow_valid']
min_rank= ['clean', 'slow_valid', 'hidden_error']
shaping= ['0.250000', '0.220000', '-0.800000'] discounted_base= 0.810000 discounted_shaping= -0.200000 shaped_total= 0.610000
telescoping_difference= -0.200000
```

## 运行方法

将上一个 Python 代码块保存为 process-reward-probe.py，再运行：

```bash
python3 process-reward-probe.py
```

修改步骤分数、终止结果、聚合方式、折扣和潜在函数后，重新比较路径排序、最终成功和 shaping 差值。接入真实 PRM 时，先固定 step boundary、mask、评分版本和独立 outcome，再把过程信号接入搜索或训练。

## 相关词条

- [奖励模型](../alignment/reward-model/)：说明代理分数、偏好标签、长度偏差和校准。
- [奖励投机](../alignment/reward-hacking/)：检查过程评分器是否被格式、长度或局部捷径利用。
- [拒绝采样](../alignment/rejection-sampling/)：比较按过程分数筛选候选和接受分布的变化。
- [强化学习基础](../alignment/rl-basics/)：提供轨迹、回报、价值函数和信用分配定义。
- [PPO](../alignment/ppo/)：说明 advantage、rollout、KL 和策略更新。
- [RLHF 与 DPO](../alignment/rlhf-dpo/)：比较 reward model、偏好对和策略优化。
- [训练稳定性](../pretraining/training-stability/)：检查过程 reward 尺度、折扣、梯度和多轮训练。
