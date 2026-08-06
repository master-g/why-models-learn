---
title: "奖励模型：把偏好反馈变成可优化的代理分数"
tags: ["why-models-learn"]
---

奖励模型用一个可训练函数给输入和候选输出打分，再把偏好、规则或环境反馈转换为后续优化可以使用的代理信号。它把“人类更喜欢哪个回答”压成一个标量或排序概率，便于策略优化，却也会继承标签偏差、覆盖不足、长度偏差和评审器漏洞。奖励模型的分数不是人类意图本身；训练、校准、分布外行为和独立结果都需要分别核验。

![奖励模型示意图：候选回答和偏好标签训练 reward model，模型输出代理分数后进入策略优化，独立评测检查分数提升是否带来真实结果改善](/assets/alignment/svg/reward-model.1.svg)

## 奖励模型学习的是偏好关系

### 输入、候选和标签组成反馈样本

一条成对偏好样本可以写成：

$$
d_i
=
(x_i,y_i^+,y_i^-,z_i),
$$

其中 $x_i$ 是输入或上下文，$y_i^+$ 是被选中的候选，$y_i^-$ 是被拒绝的候选，$z_i$ 表示标签来源、任务、版本和分歧等元数据。一个 reward model 为候选产生标量：

$$
r_\phi(x,y)\in\mathbb R.
$$

成对训练的基本要求是：

$$
r_\phi(x,y^+)
>
r_\phi(x,y^-)
$$

而不是要求 reward 的绝对数值等于某个客观真值。偏好关系只直接约束同一个输入下候选之间的相对顺序。

|字段|含义|缺失时无法确认|
|---|---|---|
|prompt/context|候选回答面对的输入、历史和约束|比较是否在同一任务上|
|chosen/rejected|被选和被拒的完整候选|偏好方向和截断边界|
|label source|专家、用户、规则或模型评审来源|标签是否同质|
|guide version|标注指南、语言和任务说明|标签语义是否变化|
|sampling config|候选生成 checkpoint、temperature、stop|候选分布是否被改变|
|split|训练、验证、测试和时间版本|是否重复使用反馈样本|

### 奖励模型不是环境真值函数

在经典 MDP 中，环境奖励可以由环境状态和动作定义。语言模型对齐中的 reward model 通常观察文本、上下文或工具结果的部分信息，学习的是反馈分布下的代理分数：

$$
\hat r_\phi(x,y)
\approx
\mathbb E[Z\mid x,y].
$$

如果标签 $Z$ 只记录“哪一个回答更讨喜”，它不一定记录事实是否正确、后续动作是否安全或用户长期结果是否改善。reward model 的训练目标越接近标注过程，它就越可能复现标注过程中的遗漏。

## Bradley–Terry 把成对偏好写成概率

### 分数差决定选择概率

最常见的成对模型是：

$$
\Pr(y^+\succ y^-\mid x)
=
\sigma\left(
r_\phi(x,y^+)-r_\phi(x,y^-)
\right),
$$

其中 sigmoid 为：

$$
\sigma(u)
=
\frac{1}{1+\exp(-u)}.
$$

如果两个候选分数相同，选择概率是 $0.5$；分数差增大时，模型对 chosen 的概率接近 $1$。这个概率描述的是标签模型的选择倾向，不是候选回答正确的概率。

### pairwise logistic loss 只看分差

令：

$$
\Delta_i
=
r_\phi(x_i,y_i^+)-r_\phi(x_i,y_i^-).
$$

chosen 标签为正时，负对数似然为：

$$
\mathcal L_i
=
-
\log\sigma(\Delta_i)
=
\operatorname{softplus}(-\Delta_i).
$$

对两个 reward 的导数为：

$$
\frac{\partial\mathcal L_i}
{\partial r_\phi(x_i,y_i^+)}
=
-
\left(1-\sigma(\Delta_i)\right),
$$

$$
\frac{\partial\mathcal L_i}
{\partial r_\phi(x_i,y_i^-)}
=
1-\sigma(\Delta_i).
$$

当 chosen 分数已经远高于 rejected 时，梯度变小；当排序反过来或分差接近零时，梯度更大。实现中应使用稳定的 log-sigmoid 或 softplus，不能先计算极端 sigmoid 再取对数。

### 三种反馈形式有不同的信息量

|反馈形式|训练信号|主要优点|主要问题|
|---|---|---|---|
|pointwise score|单个回答的等级或数值|可直接拟合尺度|标注尺度不一致|
|pairwise preference|两个回答的相对选择|比较通常比绝对打分稳定|只约束相对顺序|
|listwise ranking|多个候选的完整排序|利用一组排序关系|排序成本和一致性更高|

成对偏好适合把比较任务交给标注者，但它不会自动提供跨 prompt 的绝对尺度。一个回答在简单问题上得分高，不表示它和另一个复杂问题的分数具有相同含义。

## 奖励分数的平移和缩放

### 平移不改变成对排序

如果对所有候选加一个常数 $c$：

$$
r'(x,y)
=
r(x,y)+c,
$$

那么分差不变：

$$
r'(x,y^+)-r'(x,y^-)
=
r(x,y^+)-r(x,y^-).
$$

因此 pairwise loss 无法识别全局 reward 的零点。不同 checkpoint 的 reward 均值不能直接比较，除非定义了固定参考集和归一化合同。

### 缩放保持顺序却改变选择概率

如果：

$$
r'(x,y)=\kappa r(x,y),
\qquad
\kappa>0,
$$

排序不变，但 Bradley–Terry 概率变成：

$$
\sigma\left(
\kappa\Delta
\right).
$$

$\kappa$ 越大，概率越接近 0 或 1，reward model 看起来更自信。后续策略优化还会读取 reward 的绝对梯度和尺度，所以缩放可能改变更新幅度、KL 漂移和过优化速度。

### reward scale 需要和优化合同绑定

报告 reward 时至少保存：

|字段|作用|
|---|---|
|reference mean/std|定义跨 checkpoint 的参考尺度|
|pairwise margin|观察候选排序的分离程度|
|score quantile|检查尾部和异常高分|
|probability calibration|检查分数差对应的选择概率|
|policy update scale|检查 reward 是否改变策略梯度大小|

只报告“reward 从 0.4 上升到 0.8”没有足够含义。模型版本、样本分布、score normalization 和候选采样都可能改变数值。

## 一个可运行的 reward model 探针

下面的 Python 标准库探针用两个特征表示候选：第一个特征近似事实或任务质量，第二个特征近似输出长度。它训练 pairwise logistic reward model，检查解析梯度的方向、训练后的候选排序和显式长度偏差。

```python
from math import exp, log

def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    e = exp(value)
    return e / (1.0 + e)

def softplus(value):
    if value > 0:
        return value + log(1.0 + exp(-value))
    return log(1.0 + exp(value))

def dot(left, right):
    return sum(a * b for a, b in zip(left, right))

responses = {
    'grounded': [1.0, 0.2],
    'verbose': [0.8, 0.9],
    'hallucination': [0.2, 0.8],
    'cautious': [0.8, 0.1],
}
pairs = [
    ('grounded', 'verbose'),
    ('grounded', 'hallucination'),
    ('cautious', 'hallucination'),
]

weights = [0.0, 0.0]
learning_rate = 0.4
for _ in range(80):
    gradient = [0.0, 0.0]
    for chosen, rejected in pairs:
        difference = (
            dot(weights, responses[chosen])
            - dot(weights, responses[rejected])
        )
        probability = sigmoid(difference)
        delta = [
            a - b
            for a, b in zip(
                responses[chosen], responses[rejected]
            )
        ]
        scale = probability - 1.0
        for index in range(2):
            gradient[index] += scale * delta[index]
    for index in range(2):
        weights[index] -= (
            learning_rate * gradient[index] / len(pairs)
        )

scores = {
    name: dot(weights, features)
    for name, features in responses.items()
}
probabilities = []
for chosen, rejected in pairs:
    difference = scores[chosen] - scores[rejected]
    probabilities.append(sigmoid(difference))
print('trained_weights=', [
    f'{value:.9f}' for value in weights
])
print('trained_scores=', {
    name: f'{scores[name]:.9f}'
    for name in scores
})
print('pairwise_probabilities=', [
    f'{value:.9f}' for value in probabilities
])
print('ranked_responses=', sorted(
    scores, key=scores.get, reverse=True
))

probe_weights = [0.7, 0.3]
chosen = responses['grounded']
rejected = responses['verbose']
difference = dot(probe_weights, chosen) - dot(
    probe_weights, rejected
)
probability = sigmoid(difference)
loss = softplus(-difference)
delta = [a - b for a, b in zip(chosen, rejected)]
reward_gradient = [
    -(1.0 - probability) * value
    for value in delta
]
print('probe_difference=', f'{difference:.9f}',
      'preference_probability=', f'{probability:.9f}',
      'pairwise_loss=', f'{loss:.9f}')
print('gradient_wrt_reward_weights=', [
    f'{value:.9f}' for value in reward_gradient
])

length_biased = [0.8, 0.4]
length_scores = {
    name: dot(length_biased, features)
    for name, features in responses.items()
}
print('length_biased_scores=', {
    name: f'{score:.9f}'
    for name, score in length_scores.items()
})
print('length_biased_best=', max(
    length_scores, key=length_scores.get
))

for scale in [1.0, 2.0]:
    scaled_difference = scale * difference
    print('reward_scale=', f'{scale:.1f}',
          'probability=', f'{sigmoid(scaled_difference):.9f}',
          'loss=', f'{softplus(-scaled_difference):.9f}')
```

运行输出：

```text
trained_weights= ['2.104655726', '-2.919448683']
trained_scores= {'grounded': '1.520765989', 'verbose': '-0.943779234', 'hallucination': '-1.914627802', 'cautious': '1.391779712'}
pairwise_probabilities= ['0.921618629', '0.968792554', '0.964647973']
ranked_responses= ['grounded', 'cautious', 'verbose', 'hallucination']
probe_difference= -0.070000000 preference_probability= 0.482507142 pairwise_loss= 0.728759556
gradient_wrt_reward_weights= ['-0.103498572', '0.362245000']
length_biased_scores= {'grounded': '0.880000000', 'verbose': '1.000000000', 'hallucination': '0.480000000', 'cautious': '0.680000000'}
length_biased_best= verbose
reward_scale= 1.0 probability= 0.482507142 loss= 0.728759556
reward_scale= 2.0 probability= 0.465057055 loss= 0.765595182
```

训练后的质量权重为 $2.104655726$，长度权重为 $-2.919448683$；三组训练偏好的模型概率分别为 $0.921618629$、$0.968792554$ 和 $0.964647973$，排序为 grounded、cautious、verbose、hallucination。probe 权重为 $(0.7,0.3)$ 时，grounded 相对 verbose 的分差为 $-0.070000000$，偏好概率为 $0.482507142$，pairwise loss 为 $0.728759556$。如果长度权重改为正的 $0.4$，verbose 得分为 $1.000000000$，高于 grounded 的 $0.880000000$；评分器会把较长输出排在质量更高的输出之前。

reward scale 从 $1$ 变为 $2$ 没有改变候选顺序，却把相同分差的偏好概率从 $0.482507142$ 改为 $0.465057055$，并把 loss 从 $0.728759556$ 改为 $0.765595182$。这个结果提醒我们：分数尺度、概率解释和后续策略更新不能分开记录。

## 反馈数据的覆盖和质量

### 候选生成分布决定 reward model 看见什么

reward model 只在训练反馈覆盖的候选附近得到直接监督。如果所有候选都语法流畅，标签可能主要学习格式差异；如果所有候选都来自同一个 checkpoint，模型可能无法识别新策略产生的失败模式。

候选数据至少要覆盖：

|维度|需要变化的内容|遗漏时的结果|
|---|---|---|
|任务|问答、推理、代码、工具和写作|reward 只适合窄任务|
|难度|简单、边界和失败样本|平均偏好掩盖难例|
|长度|短答、长答和截断|长度成为捷径|
|语言/领域|部署语言和领域切片|分布外排序不稳定|
|安全状态|可回答、需澄清、应拒答|过度拒答或危险服从|
|候选来源|不同 checkpoint、temperature 和策略|同源偏差被重复学习|

候选覆盖不是样本数量的同义词。增加许多相似的普通样本，可能没有增加边界和尾部行为的信息。

### 标签指南需要把维度写开

“更好”需要拆成正确性、相关性、完整性、诚实性、安全性、风格和行动后果。标注指南还要规定：

1. 两个回答都错时如何标记；
2. 一个回答正确但不完整时如何比较；
3. 不确定时应该偏好说明限制还是自信猜测；
4. 应拒答和可安全回答的边界；
5. 外部工具结果和文字表述冲突时如何判定；
6. 候选含有隐私、提示注入或越权请求时如何处理。

如果指南把多个维度压成一个模糊的“整体感觉”，reward model 会把标注者的隐含偏好当作任务目标。

### 标注者分歧是数据的一部分

不同标注者可能对事实、风格、风险和用户目的有不同判断。应保留：

|字段|用途|
|---|---|
|annotator id 或匿名组|分析系统性分歧|
|raw vote|区分共识与多数票|
|tie/uncertain|保留不可判定样本|
|rationale|定位指南中的具体维度|
|adjudication|记录专家复核和版本|

把所有分歧强行压成一个标签会减少训练噪声表面，却丢掉任务本身的不可确定性。可以使用软标签、样本权重或不确定性切片，但应说明归约方法。

## reward head 如何从模型表示产生分数

### 最后一个位置的 hidden 可以接标量头

给定基础模型 hidden $h_\psi(x,y)$，一个最简单的 reward head 为：

$$
r_\phi(x,y)
=
w^{\mathsf T}h_\psi(x,y)+b.
$$

在序列任务中，需要明确从哪个位置读取表示：最后一个有效 token、EOS、池化向量或多个位置的聚合。padding、截断、response mask 和工具结果边界会改变这个表示。

### 冻结基础模型和联合训练是不同合同

如果只训练 reward head，基础模型表示固定，优化参数较少；如果同时更新基础模型，reward model 的容量和拟合速度更高，也更容易记住反馈样本或放大风格捷径。两种配置都要记录：

|配置|可训练参数|主要风险|
|---|---|---|
|frozen backbone|reward head|表示无法表达新维度|
|last-layer tuning|head 和末层|容易适应格式和局部模板|
|full reward model tuning|大部分基础参数|过拟合、漂移和迁移成本|
|adapter tuning|小型 adapter 和 head|目标模块、rank 和 base hash 影响结果|

reward model 训练不应把 assistant response 中的标签、分数或评审说明当作无意的额外输入。输入模板要和部署时评分模板一致，且对可操纵字段做消融。

### token-level reward 和 sequence-level reward 要区分

序列级 reward：

$$
r_\phi(x,y_{1:T})
$$

只在完整候选生成后打分。token-level reward：

$$
r_{\phi,t}(x,y_{\le t})
$$

可以提供更密集的反馈，但每个前缀的标签更难定义，且局部高分不保证完整答案质量。把序列级分数复制到每个 token 会产生一个可计算的训练张量，却不等于获得了真实的 token 贡献。

## 校准、排序和绝对分数

### pairwise accuracy 只检查排序

在保留集上，pairwise accuracy 为：

$$
\operatorname{Acc}_{\mathrm{pair}}
=
\frac1N
\sum_{i=1}^{N}
\mathbf 1
\left[
r_\phi(x_i,y_i^+)
>
r_\phi(x_i,y_i^-)
\right].
$$

它可以衡量排序方向，但无法说明分数差是否对应真实概率，也无法检查跨 prompt 的尺度。平局、标签分歧和严重度都应单独报告。

### 概率校准需要分箱和独立标签

若 reward model 输出：

$$
p_i
=
\sigma(\Delta_i),
$$

可以把样本按 $p_i$ 分箱，比较预测概率和实际 chosen 比例。高置信分箱中实际胜率明显低于预测值，说明模型过度自信。温度缩放可能改善概率校准，但不会修复候选覆盖缺失或目标遗漏。

### 跨 prompt 比较需要参考合同

在 prompt A 上的 reward 3 和 prompt B 上的 reward 2 不一定说明 A 的回答更好。reward head 可能学习了 prompt 难度、长度或模板偏移。跨 prompt 使用 reward 时，要先定义：

- 是否按 prompt 内排序；
- 是否使用 reference response 做差分；
- 是否进行长度或任务分组归一化；
- 是否把不同任务的分数送入同一个策略目标；
- 是否保留原始分数和归一化分数。

## 过优化和 reward hacking

### 策略会寻找高分区域

训练 reward model 时，候选通常来自一个有限分布。策略优化会主动寻找 reward model 的高分区域；一旦进入训练数据没有覆盖的区域，reward model 的外推可能变得不可靠：

$$
\max_\vartheta
\mathbb E_{x,y\sim\pi_\vartheta}
\left[
r_\phi(x,y)
\right].
$$

随着策略偏离 reward model 的训练分布，代理分数可以继续上升，独立质量却下降。这个过程不是 reward model 的 pairwise accuracy 能单独捕获的，因为评测样本本身可能没有跟随策略移动。

### 常见投机路径

|投机路径|模型提高的表面信号|独立检查|
|---|---|---|
|长度投机|更长、更完整的格式|质量按 token 和任务分组|
|自信投机|语气坚定、少用不确定词|事实、引用和校准|
|拒答投机|避开违规样本|可安全完成率和拒答精度|
|模板投机|加入评分器偏好的短语|去模板、改写和字段消融|
|评审器投机|直接迎合 evaluator|独立人类、规则和结果指标|
|标签泄露|复述 prompt 中的答案或分数|反事实和输入字段屏蔽|

### KL 约束限制策略漂移但不证明 reward 正确

策略优化常加入参考策略的 KL 惩罚：

$$
J_{\mathrm{reg}}(\vartheta)
=
\mathbb E
\left[
r_\phi(x,y)
-
\beta
\operatorname{KL}
\left(
\pi_\vartheta
\mathbin{\Vert}
\pi_{\mathrm{ref}}
\right)
\right].
$$

KL 惩罚限制策略离开 reference 的幅度，可能降低 reward hacking 的速度；它不能修复 reward model 已经把错误行为判为高分的情况。需要同时记录 reward、KL、独立质量和约束违反率。

### 过优化曲线应包含独立指标

沿策略更新 checkpoint 记录：

|轴或指标|作用|
|---|---|
|训练 reward|确认代理目标是否上升|
|pairwise holdout|检查排序是否保持|
|独立正确性|检查事实和任务结果|
|安全违反率|检查高影响尾部|
|KL to reference|检查策略漂移|
|长度/拒答/格式|查找可见捷径|

如果 reward 继续上升而独立指标先升后降，应把拐点当作过优化证据，而不是继续增加训练步数。

## 多目标 reward 需要保留约束

### 标量化会隐藏维度冲突

若有多个分数：

$$
\mathbf r(x,y)
=
(r_{\mathrm{quality}},
r_{\mathrm{truth}},
r_{\mathrm{safety}}),
$$

线性标量化为：

$$
r_w(x,y)
=
w^{\mathsf T}\mathbf r(x,y).
$$

改变 $w$ 会改变候选排序。若安全是硬边界，不能只把它作为一个可被质量分抵消的普通加权项；应设置拦截器或约束阈值。

### 分层 reward 不等于多任务平均

一种可审计的结构是：

$$
r_{\mathrm{total}}
=
r_{\mathrm{quality}}
+
r_{\mathrm{style}}
-
\lambda_{\mathrm{violation}}
\mathbf 1[\mathrm{violation}],
$$

同时保留每个分量和违反事件。只保存 total reward 会丢掉“质量上升、违反率也上升”的证据。

### reward model 可以提供排序，不能替代权限系统

文本 reward 不能决定是否允许发送邮件、写入数据库、执行代码或读取隐私数据。外部系统仍需执行权限检查、确认、沙箱、审计和回滚。reward model 的高分只能说明候选在某个评分器上的相对偏好。

## 失效模式与审计清单

### 数据和标签

|现象|优先检查|确认方法|
|---|---|---|
|训练 accuracy 高、部署排序差|候选分布和任务切片|按 checkpoint、语言、长度和任务拆分|
|长度越长分数越高|特征、标签指南和截断|长度匹配对照、删减和重写|
|拒答获得高 reward|安全标签把拒答当通用答案|可安全完成率和拒答理由|
|事实错误仍被选中|标注者只看流畅度|事实核验、引用和专家抽样|
|分歧被当成确定标签|没有 tie/uncertain 字段|保留 raw vote 和分歧切片|

### 模型和数值

|现象|优先检查|确认方法|
|---|---|---|
|极端 reward 或 NaN|softplus、log-sigmoid 和 dtype|极端分差单测|
|跨 checkpoint 分数不可比|平移、缩放和 normalization|固定 reference set|
|pairwise 概率过度自信|reward scale 和校准|可靠性图、温度缩放|
|token 位置泄露标签|pooling、mask 和模板|字段消融和位置置换|
|reward 只记住模板|重复 prompt 和候选格式|去模板、改写和新来源|

### 优化和部署

|现象|优先检查|确认方法|
|---|---|---|
|reward 上升、质量下降|策略过优化和分布漂移|checkpoint 曲线和独立评测|
|高分回答带来危险动作|reward 未包含行动后果|沙箱、权限和回滚测试|
|KL 很小但仍然错误|reference 本身或局部捷径|独立事实和安全切片|
|新策略超出 reward 训练分布|候选采样没有回流|主动采样和 OOD 检测|

### 最小 reward model 合同

保存 reward model checkpoint、base model hash、tokenizer、模板、pooling 位置、训练数据 hash、标签指南版本、候选生成配置、pairwise loss、score normalization、reference set、校准曲线、分组 accuracy、独立质量和策略优化曲线。没有候选和标签的原始记录，最终 reward 无法追溯。

## 运行方法

将上一个 Python 代码块保存为 reward-model-probe.py，再运行 python3 reward-model-probe.py。修改候选特征、偏好对、学习率或 reward scale 后，重新核对训练排序、pairwise probability、梯度方向和长度偏差。

接入真实反馈时，先固定 prompt、候选生成 checkpoint、temperature、stop、标签指南和 train/valid/test 切分，再记录原始投票、分歧、候选长度、任务分组和事实/安全验证结果。

沿策略优化过程保存 reward、KL、长度、拒答率、独立正确性、约束违反率和人工抽样。reward model 的 holdout accuracy 不能替代独立结果，也不能替代外部权限和回滚控制。

## 相关词条

- [对齐问题](../alignment/alignment-problem/)：说明 reward proxy 与真实意图、约束和行动后果的偏差。
- [强化学习基础](../alignment/rl-basics/)：说明奖励、策略、价值和环境交互。
- [策略梯度](../alignment/policy-gradient/)：说明 reward 如何通过 log probability 形成策略更新。
- [Bradley–Terry 模型](../alignment/bradley-terry/)：展开成对偏好到排序概率的模型。
- [指令数据](../finetuning/instruction-data/)：审计 prompt、任务约束、来源和过滤。
- [监督微调](../finetuning/sft/)：说明示范答案如何直接进入 next-token loss。
- [知识蒸馏](../finetuning/distillation/)：比较 teacher 行为和 reward proxy 的迁移。
- [RLHF 与 DPO](../alignment/rlhf-dpo/)：比较偏好反馈进入策略优化的两条路径。
- [奖励投机](../alignment/reward-hacking/)：分析高 reward 与真实结果分离的具体模式。
- [训练稳定性](../pretraining/training-stability/)：检查 reward、梯度、数值和恢复合同。
