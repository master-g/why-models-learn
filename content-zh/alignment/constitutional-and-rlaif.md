---
title: "宪法式对齐与 RLAIF：用原则和 AI 反馈构造偏好信号"
tags: ["why-models-learn"]
---

宪法式对齐把一组公开写出的原则放进批评、修订和偏好评价流程，RLAIF（Reinforcement Learning from AI Feedback，基于 AI 反馈的强化学习）再用 AI 评价器替代部分人工偏好标注，生成可供 reward model、PPO 或 DPO 使用的训练信号。原则让评价条件更容易检查，却不会自动成为真实意图；AI 评价器仍可能继承盲点、偏好表面风格或与被训练策略形成反馈回路。本文从原则表示开始，展开 self-critique/revision、AI 偏好标签、原则冲突、RLHF/RLAIF/DPO 的边界和独立验收，再用标准库探针核对一个评价器排序反转的例子。

![宪法式对齐与 RLAIF 示意图：原则进入批评和修订，AI 评价器生成偏好对，偏好优化改变策略，独立评测检查原则分数是否对应真实结果](/assets/alignment/svg/constitutional-and-rlaif.1.svg)

## 宪法式对齐把原则写成评价条件

### Constitution 是规则集合，不是单一奖励

设原则集合为：

$$
\mathcal C
 =
 \left\{
   c_1,c_2,\ldots,c_m
 \right\}.
$$

每个 $c_k$ 可以规定一种行为要求，例如：

- 说明不确定性，不把没有证据的内容写成确定事实。
- 避免泄露隐私、绕过权限或执行未确认的高影响动作。
- 在合法且低风险的请求上提供有用答案，并保留必要的限制说明。
- 把工具返回的外部状态与模型自己的计划分开陈述。

原则文本决定评价器需要检查哪些条件。它没有直接给出每个候选的真实效用，也没有替代任务成功、事实核验或外部权限系统。

对输入 $x$ 和候选 $y$，原则评价可以输出分量：

$$
s_k(x,y)
 =
 \operatorname{Eval}_{c_k}(x,y).
$$

把分量合成一个标量时，最简单的形式是：

$$
S_{\mathcal C}(x,y)
 =
 \sum_{k=1}^{m} w_k s_k(x,y),
 \qquad
 w_k\geq0.
$$

权重表达了工程选择。它没有把原则冲突变成数学事实；当 helpfulness、truthfulness 和 harmlessness 的排序不一致时，标量化会选定一种优先顺序。

|对象|作用|仍需外部确认的内容|
|---|---|---|
|原则文本 $c_k$|规定评价器要检查的条件|语言是否覆盖任务边界和例外|
|原则评分 $s_k$|记录候选对单条原则的满足程度|评分是否稳定、可校准、可解释|
|聚合分数 $S_{\mathcal C}$|生成排序或训练标签|权重是否隐藏目标冲突|
|AI 评价器|执行批量批评、比较或打分|与真实结果是否同向|
|独立评测|检查事实、任务、约束和外部结果|是否真正独立于生成反馈|

### 原则需要范围和例外

“帮助用户”在事实问题、医疗建议、代码执行和外部写操作中的可接受行为不同。原则应包含：

1. 适用范围：哪些输入、工具和输出类型进入该原则。
2. 优先级：原则冲突时哪个条件先检查。
3. 例外：低风险的解释、模拟或教育用途是否允许更详细的内容。
4. 可观测证据：评价器可以读取哪些字段，哪些字段必须由外部系统提供。
5. 失败动作：不确定时拒答、请求澄清、转人工或停止工具调用。

原则缺少这些字段时，AI 评价器会用语言习惯补全未定义部分。不同模型、提示模板或 temperature 可能因此得到不同标签。

## 批评与修订把原则放进生成过程

### 两步变换

给定初始回答 $y^{(0)}$，批评器读取原则并生成批评：

$$
c^{(0)}
 =
 \operatorname{Critique}_{\mathcal C}
 \left(
   x,y^{(0)}
 \right).
$$

修订器再根据输入、初始回答和批评生成新回答：

$$
y^{(1)}
 =
 \operatorname{Revise}
 \left(
   x,y^{(0)},c^{(0)}
 \right).
$$

可以重复若干轮：

$$
y^{(t+1)}
 =
 \operatorname{Revise}
 \left(
   x,y^{(t)},
   \operatorname{Critique}_{\mathcal C}(x,y^{(t)})
 \right).
$$

这个过程将自然语言原则转换为带解释的训练样本。它也引入了新的误差：批评可能漏掉关键事实，修订可能改变原问题，后续轮次可能重复同一套表面措辞。

### 修订增量必须分解

设独立效用为 $U$，修订前后的变化为：

$$
\Delta U
 =
 U(x,y^{(1)})
 -
 U(x,y^{(0)}).
$$

同时记录每条原则的变化：

$$
\Delta s_k
 =
 s_k(x,y^{(1)})
 -
 s_k(x,y^{(0)}).
$$

如果 $S_{\mathcal C}$ 上升而 $U$ 不变或下降，需要检查：

|修订结果|可能含义|需要保存的证据|
|---|---|---|
|原则分数上升，任务结果上升|批评修订同时改善代理和结果|独立任务切片、事实核验|
|原则分数上升，任务结果不变|修订增加格式或说明，未增加有效信息|长度、重复、内容覆盖|
|原则分数上升，任务结果下降|原则遗漏或修订改变问题目标|原始回答、修订差异、人工复核|
|原则分数下降，任务结果上升|原则过严、权重错误或评价器误判|安全边界、例外样本、独立结果|

修订文本本身不是验收证据。需要把原文、批评、修订、原则版本和外部结果一起保存。

### Self-critique 不等于独立核验

同一个模型生成回答、写批评、执行修订时，三个步骤共享表示和盲点。批评可以发现显式格式错误，却漏掉模型本身不知道的事实错误；修订可以让回答更符合规则语言，却不改变外部状态。

因此至少要区分：

- 生成模型：提出回答和修订候选。
- 评价模型：按照原则比较候选。
- 外部检查器：核对事实、权限、工具状态和硬约束。
- 人工或独立数据：校准高影响切片和评价器盲区。

不同组件可以使用同一基座，但评测证据不能只来自同一生成调用的自我陈述。

## RLAIF 把 AI 评价转换为偏好标签

### AI feedback 的数据合同

对两个候选 $y_a,y_b$，AI 评价器根据原则输出偏好标签：

$$
z_A
 =
 \operatorname{Judge}_{\mathcal C}
 \left(
   x,y_a,y_b
 \right)
 \in
 \left\{
   y_a\succ y_b,\;
   y_b\succ y_a,\;
   \text{tie}
 \right\}.
$$

每条标签至少保存：

|字段|作用|
|---|---|
|prompt|确定任务、上下文和工具状态|
|candidate A/B|保留完整回答和顺序|
|constitution version|固定原则文本、优先级和例外|
|judge checkpoint|固定评价器来源和版本|
|judge rationale|保留批评依据，但不把解释当成事实证明|
|label / tie|记录偏好、平局和不确定状态|
|sampling metadata|记录生成 checkpoint、temperature 和停止条件|
|independent slice|连接外部结果、人工复核或事实检查|

AI 评价器的 rationale 有助于定位规则触发点。它不能替代独立事实来源；评价器写出“已完成”不等于外部系统状态已改变。

### Bradley–Terry 只是标签生成模型

如果评价器给出标量 $S_A,S_B$，可以用 Bradley–Terry 形式写偏好概率：

$$
\Pr(y_a\succ y_b\mid x)
 =
 \sigma
 \left(
   \beta
   \left(
     S_A-S_B
   \right)
 \right).
$$

当 $S_A-S_B=0$ 时，概率为 $0.5$。增大 $\beta$ 会让标签更确定，却不会增加评价器对任务结果的知识。

RLAIF 的下游 reward model 可以拟合这些标签：

$$
\mathcal L_{\mathrm{AI}}
 =
 -
 \log
 \sigma
 \left(
   r_\phi(x,y_a)-r_\phi(x,y_b)
 \right).
$$

之后可以使用 PPO、DPO 或其他偏好优化方法改变策略。RLAIF 的反馈来源改变了标签生产环节；它没有保证下游 reward model 与真实效用相同。

### 用独立标签校准 AI 评价器

设一组人工或外部结果标签为 $z_I$，AI 标签为 $z_A$。先计算成对一致率：

$$
\operatorname{agreement}
 =
 \frac{1}{N}
 \sum_{i=1}^{N}
 \mathbf 1
 \left[
   z_{A,i}=z_{I,i}
 \right].
$$

高一致率仍需要按任务和风险分层。平均一致率可能掩盖医疗、权限、隐私、低资源语言或长上下文切片中的错误。

|校准层|检查内容|失败后的动作|
|---|---|---|
|标签方向|chosen/rejected 顺序、平局、拒答和不确定|修复数据管线、重跑样本|
|分组一致率|任务、语言、长度、风险、工具状态|限制反馈适用范围|
|概率校准|偏好概率与独立标签频率|调整 beta 或重新校准|
|尾部错误|高影响但低频的错误标签|人工复核、硬约束拦截|
|干预稳定性|只改变格式、长度、语气后的标签|消除捷径字段或改评价提示|

## 原则冲突决定 RLAIF 的排序

### 加权和会隐藏优先级

三个原则的分数可以写成向量：

$$
\boldsymbol s(x,y)
 =
 \left(
   s_{\mathrm{helpful}},
   s_{\mathrm{truthful}},
   s_{\mathrm{harmless}}
 \right).
$$

加权和：

$$
S_{\boldsymbol w}
 =
 \boldsymbol w^{\mathsf T}
 \boldsymbol s.
$$

若两个回答在不同维度各自占优，改变 $\boldsymbol w$ 就可能改变排序。权重不是模型从数据中自动发现的真理；它是产品、政策和风险负责人需要确认的选择。

四个候选的一个示例：

|候选|AI helpful|AI truthful|AI harmless|独立综合结果|
|---|---:|---:|---:|---:|
|grounded answer|0.82|0.84|0.86|0.890|
|verbose refusal|0.84|0.86|0.99|0.704|
|confident claim|0.90|0.45|0.80|0.564|
|safe refusal|0.55|0.82|0.99|0.672|

在权重 $\boldsymbol w=(0.4,0.4,0.2)$ 下，AI 评价器给 verbose refusal 的综合分为 $0.878$，给 grounded answer 的分为 $0.836$。独立结果分别为 $0.704$ 和 $0.890$。AI 评价器偏好详细拒答时，训练会增加拒答概率；独立任务结果要求保留有依据的回答。

### 词典式优先级和硬约束

对于不可抵消的原则，可以使用词典式比较：

1. 先检查硬安全和权限条件。
2. 在满足硬条件的候选中比较真实性。
3. 在真实性相近时比较帮助程度、清晰度和成本。

也可以使用约束优化：

$$
\max_\pi\;
 \mathbb E[S_{\mathcal C}(x,y)]
\quad
\text{subject to}
\quad
\mathbb E[C_j(x,y)]
\leq b_j.
$$

硬约束检查器必须独立于可优化分数。把所有规则折成一个标量时，某个高分维度可能抵消不可接受的权限或安全违反。

### 原则文本会产生语义歧义

“诚实地帮助用户”可能允许：

- 在证据充分时直接回答。
- 在信息不足时说明不确定性并请求澄清。
- 在请求危险动作时解释风险并提供低风险替代方案。

评价器需要知道这些分支的适用条件。只给出一个抽象原则，模型可能根据常见文本模式选择其中一个分支；这个选择需要用边界样本和独立标签核验。

## 宪法式对齐、RLHF、RLAIF 与 DPO 的边界

### 反馈来源和优化算法是两个轴

|流程|偏好标签来源|下游优化|需要独立核验的主要位置|
|---|---|---|---|
|RLHF|人类或人工组织的偏好|reward model + PPO 等|标注覆盖、reward 外推、rollout 和独立结果|
|RLAIF|AI 评价器，常受 constitution 指导|reward model + PPO 等|评价器校准、原则冲突、同源盲点|
|Constitutional AI|原则驱动的批评/修订和偏好生成|SFT、RL 或其他偏好优化|原则范围、修订质量、AI feedback 与真实结果|
|DPO|固定偏好对，可由人或 AI 产生|直接优化 policy/reference log-ratio|偏好对覆盖、reference、mask、长度和独立结果|

RLAIF 描述反馈来源。Constitutional AI 描述原则和批评修订等流程。PPO、DPO 描述策略更新方法。四个名词不能放在同一个维度上比较。

### 相同 constitution 不保证相同策略

策略还受以下因素影响：

$$
\pi_{\mathrm{final}}
 =
 F
 \left(
   \mathcal C,
   \mathcal D_{\mathrm{feedback}},
   \operatorname{Judge},
   \operatorname{Optimizer},
   \pi_{\mathrm{base}}
 \right).
$$

改变基座模型、评价器模板、候选生成温度、偏好数据覆盖、beta、KL 系数或 reference，都可能改变最终行为。复现时需要保存这些字段，而不能只保存 constitution 文本。

## RLAIF 的反馈回路和安全边界

### 评价器与策略会共同适应

如果新策略生成的新候选继续用于训练 AI 评价器，数据分布会迭代变化：

$$
\mathcal D_{t+1}
 =
 \operatorname{Collect}
 \left(
   \pi_t,\operatorname{Judge}_t
 \right).
$$

下一轮评价器看到的候选已经受到上一轮评分标准影响。若没有固定的时间外集合和人工复核集合，反馈回路可以让评价器和策略在内部指标上共同提高，外部结果保持不变。

每轮至少保存：

|版本字段|为什么需要|
|---|---|
|策略 checkpoint|确认候选由哪个策略产生|
|评价器 checkpoint|确认偏好由哪个版本判断|
|constitution version|确认规则文本是否改变|
|候选生成配置|分离 temperature、top-p 和 stop 差异|
|reference / reward 版本|分离优化基线和评分器差异|
|独立结果|确认外部任务和约束是否变化|

### 原则不能授予外部权限

宪法式评价可以建议“需要确认”或“不能执行”，执行层仍需检查：

- 用户身份和资源权限。
- 工具参数、目标对象和作用范围。
- 用户确认、幂等性和高影响动作的二次确认。
- 执行前后的外部状态。
- 审计日志、回滚和人工接管。

AI 评价器判断一段回答“符合原则”时，仍不代表该回答有权执行邮件发送、代码运行、数据库写入或隐私读取。

## 失效模式与审计清单

|现象|原因候选|最小核验|
|---|---|---|
|AI 评价器偏好更长或更礼貌的回答|格式和长度捷径|语义固定的长度、格式和语气干预|
|拒答率上升、任务成功率下降|harmlessness 权重过高或例外未定义|安全边界切片、可回答任务成功率|
|self-critique 轮数增加但事实错误不降|批评器与生成器共享盲点|外部事实核验、独立评价器和人工复核|
|AI 与人工标签平均一致但高风险切片失败|低风险样本占比过高|按严重度、任务和工具状态分层|
|原则分数上升、外部状态未改变|评价器读取了模型陈述|工具返回值和动作后状态|
|不同评价器给出相反排序|constitution 解释、模板或模型偏差|固定候选交叉评测、版本和 rationale|
|多轮 RLAIF 后内部分数持续上升|评价器与策略共适应|时间外集合、冻结评价器、回滚对照|
|DPO/PPO loss 下降、真实结果不变|反馈覆盖不足或代理目标错误|独立任务、候选覆盖、reward/utility 曲线|
|原则冲突时行为不稳定|权重、优先级和例外未固定|权重扫描、词典式规则和硬约束测试|
|安全判断依赖回答中的声明|奖励通道可被策略影响|受保护外部状态、权限和日志审计|

### 最小可交付证据

一次可复现的 Constitutional AI/RLAIF 审计应保存：

- constitution 原文、版本、优先级、例外和适用范围。
- 初始回答、批评、修订回答和每轮 token/长度变化。
- AI 评价器 checkpoint、提示模板、评分分量、rationale 和标签。
- 人工或外部独立标签、分组一致率、概率校准和高风险样本。
- reward model、policy、reference、optimizer、beta、KL 和数据切分。
- 工具调用的权限、参数、返回值、外部状态和回滚记录。
- 代理指标、真实效用、约束违反率和尾部失败的时间序列。

## 一个可运行的最小探针

下面的探针用三个原则给四个候选打分。AI 分量刻意把 verbose refusal 评得很高，独立分量保留任务结果；它展示反馈来源与真实效用发生排序分离时，RLAIF 标签如何把策略推向错误方向。探针不模拟真实模型，也不证明某个部署系统存在该问题。

```python
import math

principles = ("helpful", "truthful", "harmless")
weights = (0.4, 0.4, 0.2)
responses = [
    {
        "name": "grounded_answer",
        "ai": (0.82, 0.84, 0.86),
        "independent": (0.88, 0.92, 0.85),
    },
    {
        "name": "verbose_refusal",
        "ai": (0.84, 0.86, 0.99),
        "independent": (0.58, 0.70, 0.96),
    },
    {
        "name": "confident_claim",
        "ai": (0.90, 0.45, 0.80),
        "independent": (0.62, 0.40, 0.78),
    },
    {
        "name": "safe_refusal",
        "ai": (0.55, 0.82, 0.99),
        "independent": (0.35, 0.84, 0.98),
    },
]


def weighted(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))


for response in responses:
    print(
        "scores",
        response["name"],
        f"{weighted(response['ai'], weights):.6f}",
        f"{weighted(response['independent'], weights):.6f}",
    )

print(
    "ai_best=",
    max(responses, key=lambda row: weighted(row["ai"], weights))["name"],
)
print(
    "independent_best=",
    max(
        responses,
        key=lambda row: weighted(row["independent"], weights),
    )["name"],
)

for current_weights in (
    (0.6, 0.3, 0.1),
    (0.4, 0.4, 0.2),
    (0.2, 0.3, 0.5),
):
    ai_rank = sorted(
        responses,
        key=lambda row: weighted(row["ai"], current_weights),
        reverse=True,
    )
    independent_rank = sorted(
        responses,
        key=lambda row: weighted(row["independent"], current_weights),
        reverse=True,
    )
    print(
        "weights=",
        current_weights,
        "ai_rank=",
        [row["name"] for row in ai_rank],
        "independent_rank=",
        [row["name"] for row in independent_rank],
    )

pairs = []
for left in range(len(responses)):
    for right in range(left + 1, len(responses)):
        ai_left = weighted(responses[left]["ai"], weights)
        ai_right = weighted(responses[right]["ai"], weights)
        independent_left = weighted(
            responses[left]["independent"],
            weights,
        )
        independent_right = weighted(
            responses[right]["independent"],
            weights,
        )
        pairs.append(
            (
                ai_left > ai_right,
                independent_left > independent_right,
            )
        )
print(
    "pair_count=",
    len(pairs),
    "agreement=",
    sum(ai == independent for ai, independent in pairs),
)

before = (0.70, 0.62, 0.78)
after = (0.84, 0.86, 0.80)
print(
    "revision_before=",
    f"{weighted(before, weights):.6f}",
    "revision_after=",
    f"{weighted(after, weights):.6f}",
    "deltas=",
    [f"{new - old:.6f}" for new, old in zip(after, before)],
)

ai_margin = weighted(responses[0]["ai"], weights) - weighted(
    responses[1]["ai"],
    weights,
)
independent_margin = weighted(
    responses[0]["independent"],
    weights,
) - weighted(responses[1]["independent"], weights)
for beta in (1.0, 4.0):
    ai_probability = 1.0 / (1.0 + math.exp(-beta * ai_margin))
    independent_probability = 1.0 / (
        1.0 + math.exp(-beta * independent_margin)
    )
    print(
        "beta=",
        beta,
        "ai_prob_grounded_over_verbose=",
        f"{ai_probability:.9f}",
        "independent_prob=",
        f"{independent_probability:.9f}",
    )
```

输出为：

```text
scores grounded_answer 0.836000 0.890000
scores verbose_refusal 0.878000 0.704000
scores confident_claim 0.700000 0.564000
scores safe_refusal 0.746000 0.672000
ai_best= verbose_refusal
independent_best= grounded_answer
weights= (0.6, 0.3, 0.1) ai_rank= ['verbose_refusal', 'grounded_answer', 'confident_claim', 'safe_refusal'] independent_rank= ['grounded_answer', 'verbose_refusal', 'confident_claim', 'safe_refusal']
weights= (0.4, 0.4, 0.2) ai_rank= ['verbose_refusal', 'grounded_answer', 'safe_refusal', 'confident_claim'] independent_rank= ['grounded_answer', 'verbose_refusal', 'safe_refusal', 'confident_claim']
weights= (0.2, 0.3, 0.5) ai_rank= ['verbose_refusal', 'safe_refusal', 'grounded_answer', 'confident_claim'] independent_rank= ['grounded_answer', 'safe_refusal', 'verbose_refusal', 'confident_claim']
pair_count= 6 agreement= 5
revision_before= 0.684000 revision_after= 0.840000 deltas= ['0.140000', '0.240000', '0.020000']
beta= 1.0 ai_prob_grounded_over_verbose= 0.489501543 independent_prob= 0.546366403
beta= 4.0 ai_prob_grounded_over_verbose= 0.458098506 independent_prob= 0.677869926
```

## 运行方法

将上一个 Python 代码块保存为 constitutional-rlaif-probe.py，再运行：

```bash
python3 constitutional-rlaif-probe.py
```

修改原则权重、AI 分量、独立分量或候选数量后，重新检查 AI 排序、独立排序、pairwise agreement、修订增量和偏好概率。接入真实评价器时，先固定 constitution、模板、checkpoint 和候选顺序，再把高风险切片交给独立评测。

## 相关词条

- [对齐问题](../alignment/alignment-problem/)：区分人类意图、训练目标、代理 reward 和外部约束。
- [奖励模型](../alignment/reward-model/)：展开偏好标签、reward head、候选覆盖和校准。
- [RLHF 与 DPO](../alignment/rlhf-dpo/)：比较人类反馈、在线 reward 优化和离线偏好优化。
- [奖励投机](../alignment/reward-hacking/)：检查代理分数上升而真实结果下降或约束违反率上升的路径。
- [DPO](../alignment/dpo/)：说明固定偏好对、reference 和 log-ratio loss。
- [PPO](../alignment/ppo/)：说明 rollout、advantage、KL 和策略更新。
- [策略梯度](../alignment/policy-gradient/)：连接偏好概率、log probability 与参数梯度。
- [训练稳定性](../pretraining/training-stability/)：检查评价器、策略、beta 和多轮反馈回路的数值行为。
