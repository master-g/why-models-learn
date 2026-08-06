---
title: "拒绝采样：用候选筛选改变策略分布"
tags: ["why-models-learn"]
---

拒绝采样在对齐工程中通常表示：从策略生成多个候选，用 reward model、规则或独立检查器筛选，再保留达到阈值或得分最高的回答。它可以在推理时改变输出分布，也可以把保留样本用于 rejection fine-tuning；候选数量、筛选分数和接受规则共同决定新分布。经典拒绝采样还要求一个明确的目标分布和接受概率，best-of-N 只按代理分数挑选最高候选，二者需要分开讨论。本文从接受分布和顺序统计量推导这两种流程，核对采样数量如何放大 reward hacking 风险，再建立成本、覆盖、多样性、独立评测和回滚账本。

![拒绝采样示意图：基础策略产生多个候选，筛选器按阈值或 reward 排序保留样本，独立评测检查筛选后的真实效用和约束](/assets/alignment/svg/rejection-sampling.1.svg)

## 先区分经典拒绝采样和 best-of-N

### 经典拒绝采样的目标是重建分布

设 proposal 分布为 $p(y\mid x)$，目标分布为 $q^\star(y\mid x)$。如果存在常数 $M$ 使：

$$
q^\star(y\mid x)
 \leq
 M p(y\mid x)
$$

对所有候选成立，可以按接受概率：

$$
a(y\mid x)
 =
 \frac{q^\star(y\mid x)}
 {M p(y\mid x)}
$$

保留 proposal 样本。接受后的分布满足：

$$
q_{\mathrm{accepted}}(y\mid x)
 =
 \frac{
   p(y\mid x)a(y\mid x)
 }{
   \mathbb E_{y'\sim p}[a(y'\mid x)]
 }
 =
 q^\star(y\mid x).
$$

这里的接受概率依赖目标密度与 proposal 密度之比。只按 reward model 最高分保留候选时，通常没有这个密度比，也没有证明结果服从某个真实目标分布。

### 对齐工程中的筛选操作

在语言模型后训练中，常见流程是：

1. 用当前策略 $\pi_\theta$ 对同一个 prompt 采样 $N$ 个回答。
2. 用 reward model 或规则得到每个回答的分数。
3. 选择 top-1、top-k、超过阈值的回答，或删除失败样本。
4. 将保留样本直接返回，或加入下一轮训练数据。

这一步更准确的名称是 sample-and-rank、best-of-N、threshold filtering 或 rejection fine-tuning，工程文档仍可能统称 rejection sampling。记录名称时要说明目标是推理时筛选、数据过滤还是分布重构。

|流程|输入|输出|核心假设|
|---|---|---|---|
|经典 rejection sampling|proposal 与目标密度|目标分布样本|接受概率使用密度比|
|threshold filtering|策略样本与阈值|满足分数阈值的样本|阈值分数与真实结果同向|
|best-of-N|固定 prompt 的 N 个候选|代理分数最高候选|候选中存在高质量样本且评分器排序可靠|
|top-k filtering|候选集合与 k|前 k 个候选|保留的多样性和排序可接受|
|rejection fine-tuning|筛选后的回答|新策略 checkpoint|保留样本足以代表目标分布|

## 阈值筛选如何改变策略分布

### 接受事件的条件分布

设代理分数为 $R_\phi(x,y)$，接受事件为：

$$
A_\tau(x,y)
 =
 \mathbf 1
 \left[
   R_\phi(x,y)\geq\tau
 \right].
$$

接受率是：

$$
\alpha_\tau(x)
 =
 \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
 \left[
   A_\tau(x,y)
 \right].
$$

若不断采样直到接受，接受样本的条件分布为：

$$
q_\tau(y\mid x)
 =
 \frac{
   \pi_\theta(y\mid x)A_\tau(x,y)
 }{
   \alpha_\tau(x)
 }.
$$

采样次数的期望为：

$$
\mathbb E[N_{\mathrm{draw}}\mid x]
 =
 \frac{1}{\alpha_\tau(x)}.
$$

阈值越高，接受率通常越低，计算成本越高。接受率为 $0$ 时，流程无法返回样本；接受率随 prompt 变化时，筛选器也在改变任务分布的权重。

### 阈值筛选不只改变质量均值

假设某个任务切片的低分回答更集中在长上下文、低资源语言或工具失败状态。按同一阈值筛选后，保留数据的任务比例为：

$$
\Pr_{\mathrm{accepted}}(g)
 =
 \frac{
   \Pr_{\pi_\theta}(g)
   \Pr_{\pi_\theta}(A_\tau\mid g)
 }{
   \sum_h
   \Pr_{\pi_\theta}(h)
   \Pr_{\pi_\theta}(A_\tau\mid h)
 }.
$$

分组接受率不同会改变数据分布。总体 accepted reward 上升时，某些任务可能已经没有样本。

|切片|原始样本比例|接受率|筛选后影响|
|---|---:|---:|---|
|常见任务|高|高|在训练数据中占比继续上升|
|困难任务|中|低|保留样本减少，覆盖下降|
|低资源语言|低|低|可能被筛选流程删除|
|工具失败状态|低|接近零|失败路径不再进入训练数据|

需要同时报告原始分布、接受率和 accepted 分布，不能只报告保留样本的平均分。

## best-of-N 的选择分布

### 最高分候选的概率

固定输入 $x$，假设有有限候选类型 $i=1,\ldots,K$。候选类型 $i$ 的策略概率、代理分数和真实效用分别为 $p_i$、$R_i$、$U_i$。从策略独立采样 $N$ 次，选择代理分数最高的候选。

定义：

$$
F_{<}(R_i)
 =
 \sum_{j:R_j<R_i}p_j,
\qquad
F_{\leq}(R_i)
 =
 \sum_{j:R_j\leq R_i}p_j.
$$

当分数互异时，候选 $i$ 被选中的概率为：

$$
\Pr(i\text{ selected})
 =
 F_{\leq}(R_i)^N
 -
 F_{<}(R_i)^N.
$$

推导只用两个事件：所有 N 个样本的分数不超过 $R_i$，减去所有分数严格低于 $R_i$。$N$ 增大时，高分候选的选择概率上升，即使高分来自代理捷径。

### 代理 reward 的期望会升高

选择后的代理 reward 期望：

$$
\mathbb E[R_{\mathrm{selected}}]
 =
 \sum_{i=1}^{K}
 \Pr(i\text{ selected})R_i.
$$

真实效用期望：

$$
\mathbb E[U_{\mathrm{selected}}]
 =
 \sum_{i=1}^{K}
 \Pr(i\text{ selected})U_i.
$$

这两个表达式的权重相同，代入的数值不同。只有当 $R$ 的排序能够代表 $U$ 的排序时，增大 $N$ 才有稳定的真实收益。

四个候选的一个数值例子：

|候选|策略概率 $p_i$|代理 reward $R_i$|真实效用 $U_i$|
|---|---:|---:|---:|
|grounded|0.35|0.55|0.90|
|concise correct|0.30|0.80|0.82|
|verbose hallucination|0.20|0.95|0.40|
|safe refusal|0.15|0.65|0.75|

$N=1$ 时，代理 reward 期望为 $0.720000000$，真实效用期望为 $0.753500000$。$N=8$ 时，代理 reward 期望为 $0.924225720$，真实效用期望为 $0.470224648$。采样更多候选让评分器更容易看到 verbose hallucination；选择器把它的代理分数当作质量证据。

### top-k 和 top-1 的差异

top-1 只返回一个候选，方便直接部署，却丢失同一 prompt 下的备选信息。top-k 可以保留多个候选供人工、规则或二级检查器比较：

|策略|输出形式|优势|代价|
|---|---|---|---|
|top-1|只返回最高 reward|延迟和接口简单|极值误差直接进入输出|
|top-k|返回或保存前 k 个|保留人工和安全检查空间|仍依赖排序，成本更高|
|阈值 + 随机|从通过阈值的候选随机选|降低单个最高分的选择偏差|需要校准阈值和接受率|
|阈值 + 独立重排|先过滤硬约束，再用独立指标排序|分开安全和质量|需要多个可靠检查器|

如果候选之间高度相似，top-k 的名义数量不等于有效多样性。需要计算语义、任务路径、工具调用和错误类型的去重覆盖。

## 筛选分数决定候选会学到什么

### reward hacking 会随 N 放大

设代理 reward：

$$
R_\phi
 =
 U
 +
 B_\phi
 +
 \varepsilon_\phi.
$$

best-of-N 选择的是 $R_\phi$ 的极值。若捷径项 $B_\phi$ 在某类回答上偏高，N 增大后被选中的概率会快速上升。独立评测看到的真实效用可能下降。

因此应绘制 N 与以下指标的曲线：

$$
\left(
 \mathbb E[R],
 \mathbb E[U],
 \mathbb E[C_j],
 \operatorname{coverage},
 \operatorname{cost}
\right).
$$

一条曲线只显示 reward 上升时，无法判断筛选流程是否安全。

### 评分器校准必须覆盖极值区间

普通平均校准检查：

$$
\mathbb E[U\mid R=r].
$$

best-of-N 需要额外检查高分尾部：

$$
\mathbb E[U\mid R\geq q_{0.95}],
\qquad
\mathbb E[U\mid R=R_{\mathrm{max}}].
$$

评分器在中间分数区间表现良好，不代表 top-1 的极值分数可信。高分候选还可能来自异常长度、格式泄漏、评测器提示注入或重复样式。

### 阈值和 top-N 的选择要按切片校准

|配置|需要记录|失败表现|
|---|---|---|
|threshold $\tau$|总体和分组接受率|困难切片没有保留样本|
|N|采样次数、总 token、延迟|极值分数上升，真实效用下降|
|top-k|k、候选间距、去重率|多个候选只是同一错误的改写|
|二级检查器|版本、调用顺序、拒绝原因|某一检查器错误覆盖其他信号|
|回退策略|没有候选通过时的动作|系统返回未经检查的默认答案|

## 推理筛选和 rejection fine-tuning

### 推理时筛选不更新策略参数

推理时 best-of-N 的输出分布变为：

$$
\pi_{\mathrm{BoN}}(y\mid x)
 =
 \Pr_{\substack{
 y_1,\ldots,y_N\sim\pi_\theta\\
 \operatorname{select}(y_1,\ldots,y_N)=y
 }}
$$

基础策略 $\pi_\theta$ 没有改变。改变的是每个请求的计算量、输出选择规则和最终服务分布。部署日志需要保存：

- 基础策略 checkpoint。
- N、temperature、top-p、stop 和并发配置。
- 所有候选或候选摘要。
- 每个候选的 reward、规则分数、独立检查和拒绝原因。
- 最终选择和回退路径。

只保存最终回答时，无法重建选择偏差。

### 过滤样本再训练会更新策略

如果把 accepted 样本用于监督微调，训练目标近似为：

$$
\mathcal L_{\mathrm{RFT}}(\theta)
 =
 -
 \mathbb E_{(x,y)\sim\mathcal D_{\mathrm{accepted}}}
 \left[
   \log\pi_\theta(y\mid x)
 \right].
$$

数据分布：

$$
\mathcal D_{\mathrm{accepted}}
 \sim
 \mathcal D_{\mathrm{prompt}}
 \pi_{\mathrm{old}}(y\mid x)
 \mathbf 1
 \left[
   R_\phi(x,y)\geq\tau
 \right].
$$

新策略会增加 accepted 样本的概率。若筛选器的代理捷径进入 accepted 数据，RFT 会把捷径写入策略，而推理时筛选只在单次请求中使用它。

两种流程要分开报告：

|流程|参数是否更新|主要风险|
|---|---|---|
|inference-time BoN|否|延迟、极值误差、候选覆盖|
|threshold filtering|否或只做缓存|接受率变化、尾部失败|
|rejection fine-tuning|是|筛选偏差进入 checkpoint|
|迭代 RFT|多轮更新|策略与评分器共同适应|

### 被拒绝样本仍有信息

若只保存 accepted 样本，训练数据丢失：

- 哪些任务难以通过阈值。
- 评分器在 rejected 样本上的错误。
- chosen/rejected 的最小差异。
- 约束失败和工具失败的路径。

可以把 rejected 样本按失败原因保存，用于偏好对、hard negative、独立评测和阈值校准。被拒绝不等于低真实质量；它只表示没有通过当前筛选协议。

## 候选多样性和计算账本

### 增大 N 可能只重复同一模式

若每次采样都使用很低 temperature，N 个候选可能共享同一错误。可以用不同 seed、temperature、prompt 变体或候选生成 checkpoint 增加覆盖，但每种变化都要记录，因为它改变 proposal distribution。

有效候选数量需要按任务行为估计：

$$
N_{\mathrm{effective}}
 =
 \frac{
   \left(\sum_i w_i\right)^2
 }{
   \sum_i w_i^2
 },
$$

其中 $w_i$ 可以是候选聚类中每个模式的概率。$N_{\mathrm{effective}}$ 小时，名义 N 增加只增加重复计算。

### 成本不只是一倍采样

总成本可以拆成：

$$
\operatorname{Cost}
 =
 N\operatorname{Cost}_{\mathrm{generate}}
 +
 N\operatorname{Cost}_{\mathrm{score}}
 +
 \operatorname{Cost}_{\mathrm{independent}}
 +
 \operatorname{Cost}_{\mathrm{storage}}.
$$

独立检查器若只检查最终 top-1，可能漏掉 rejected 候选中的系统性问题；若检查全部 N 个候选，成本随 N 增长。可以先执行低成本硬过滤，再对少量候选做昂贵的事实、工具和人工核验，但要记录过滤顺序。

|阶段|可执行的低成本检查|高成本检查|
|---|---|---|
|候选生成后|长度、格式、权限参数、重复|事实检索、工具模拟|
|reward 排序后|分数异常、候选间距、分组接受率|人工比较、独立模型交叉评测|
|最终返回前|硬约束、敏感字段、外部状态|人工确认、真实环境 dry-run|
|RFT 入库前|去重、分布切片、checkpoint 追踪|长期任务结果和时间外评测|

## 失效模式与审计清单

|现象|原因候选|最小核验|
|---|---|---|
|N 增大、reward 上升、真实效用下降|代理 reward 尾部错误或 reward hacking|N 曲线、top-1 独立效用、失败样本|
|阈值提高后某些任务没有样本|分组接受率不一致|按任务、语言、难度报告 acceptance|
|top-k 候选高度相似|低温采样或模式坍缩|语义/错误类型聚类、有效候选数|
|accepted 数据长度持续增加|sum reward 或格式捷径|长度匹配、mean 对照、RFT 长度分布|
|筛选后安全率下降|安全检查器顺序或评分器可利用|独立硬约束和动作后状态|
|BoN 结果好、RFT 后退化|筛选分布过窄或把捷径写入策略|checkpoint 对照、未筛选评测、回滚|
|不同 N 使用不同任务分布|接受率和候选覆盖随 N 改变|原始/accepted 分布、分组权重|
|最终回答正确但工具状态错误|模型陈述替代外部事实|工具返回值、执行日志、外部状态|
|拒绝原因无法解释|只保存最终结果或覆盖了规则版本|候选、分数、规则、拒绝事件完整记录|
|N 增大后延迟和成本超限|生成、评分和独立评测重复执行|token、GPU 时间、评分调用和缓存命中|

### 最小审计输出

一次可复现的拒绝采样实验应保存：

- proposal policy、prompt 分布、采样参数和每个候选的完整 token 序列。
- reward model、规则检查器、独立评测器和版本 hash。
- N、top-k、阈值、tie-break、接受率、拒绝原因和回退策略。
- 每个候选的代理 reward、真实效用、约束结果、长度和错误类别。
- accepted 与 rejected 的分布、去重率、有效候选数量和计算成本。
- 推理时输出或 RFT checkpoint、训练步、reference、独立结果和回滚点。

## 一个可运行的最小探针

下面的探针固定四种回答和它们在基础策略中的概率。它计算 best-of-N 选择分布、阈值接受分布和至少一个候选达到阈值的概率。verbose hallucination 的代理 reward 最高，真实效用最低，用于展示采样数量增加后的选择偏差。

```python
import math

candidates = [
    {
        "name": "grounded",
        "p": 0.35,
        "proxy": 0.55,
        "utility": 0.90,
    },
    {
        "name": "concise_correct",
        "p": 0.30,
        "proxy": 0.80,
        "utility": 0.82,
    },
    {
        "name": "verbose_hallucination",
        "p": 0.20,
        "proxy": 0.95,
        "utility": 0.40,
    },
    {
        "name": "safe_refusal",
        "p": 0.15,
        "proxy": 0.65,
        "utility": 0.75,
    },
]


def selected_probabilities(sample_count):
    probabilities = []
    for row in candidates:
        lower = sum(
            candidate["p"]
            for candidate in candidates
            if candidate["proxy"] < row["proxy"]
        )
        upper = sum(
            candidate["p"]
            for candidate in candidates
            if candidate["proxy"] <= row["proxy"]
        )
        probabilities.append(upper**sample_count - lower**sample_count)
    return probabilities


for sample_count in (1, 2, 4, 8):
    probabilities = selected_probabilities(sample_count)
    expected_proxy = sum(
        probability * row["proxy"]
        for probability, row in zip(probabilities, candidates)
    )
    expected_utility = sum(
        probability * row["utility"]
        for probability, row in zip(probabilities, candidates)
    )
    print(
        "N=",
        sample_count,
        "selected_probs=",
        [f"{probability:.9f}" for probability in probabilities],
        "expected_proxy=",
        f"{expected_proxy:.9f}",
        "expected_utility=",
        f"{expected_utility:.9f}",
    )

threshold = 0.80
acceptance_rate = sum(
    row["p"] for row in candidates if row["proxy"] >= threshold
)
conditional_probabilities = [
    row["p"] / acceptance_rate
    if row["proxy"] >= threshold
    else 0.0
    for row in candidates
]
expected_proxy = sum(
    probability * row["proxy"]
    for probability, row in zip(conditional_probabilities, candidates)
)
expected_utility = sum(
    probability * row["utility"]
    for probability, row in zip(conditional_probabilities, candidates)
)
print(
    "threshold=",
    f"{threshold:.2f}",
    "acceptance_rate=",
    f"{acceptance_rate:.9f}",
    "expected_draws=",
    f"{1.0 / acceptance_rate:.9f}",
    "conditional_probs=",
    [f"{probability:.9f}" for probability in conditional_probabilities],
    "expected_proxy=",
    f"{expected_proxy:.9f}",
    "expected_utility=",
    f"{expected_utility:.9f}",
)

for sample_count in (2, 4, 8):
    at_least_threshold = 1.0 - (1.0 - acceptance_rate) ** sample_count
    print(
        "N=",
        sample_count,
        "at_least_threshold=",
        f"{at_least_threshold:.9f}",
    )
```

输出为：

```text
N= 1 selected_probs= ['0.350000000', '0.300000000', '0.200000000', '0.150000000'] expected_proxy= 0.720000000 expected_utility= 0.753500000
N= 2 selected_probs= ['0.122500000', '0.390000000', '0.360000000', '0.127500000'] expected_proxy= 0.804250000 expected_utility= 0.669675000
N= 4 selected_probs= ['0.015006250', '0.347100000', '0.590400000', '0.047493750'] expected_proxy= 0.877684375 expected_utility= 0.569907937
N= 8 selected_probs= ['0.000225188', '0.163865910', '0.832227840', '0.003681062'] expected_proxy= 0.924225720 expected_utility= 0.470224648
threshold= 0.80 acceptance_rate= 0.500000000 expected_draws= 2.000000000 conditional_probs= ['0.000000000', '0.600000000', '0.400000000', '0.000000000'] expected_proxy= 0.860000000 expected_utility= 0.652000000
N= 2 at_least_threshold= 0.750000000
N= 4 at_least_threshold= 0.937500000
N= 8 at_least_threshold= 0.996093750
```

## 运行方法

将上一个 Python 代码块保存为 rejection-sampling-probe.py，再运行：

```bash
python3 rejection-sampling-probe.py
```

修改基础策略概率、候选 reward、真实效用、阈值和 N 后，重新比较选择分布、接受率、代理期望、真实效用和计算次数。接入真实模型时，先保存 rejected 候选及其拒绝原因，再做 rejection fine-tuning。

## 相关词条

- [奖励模型](../alignment/reward-model/)：说明候选 reward、长度偏差、校准和 reward overoptimization。
- [奖励投机](../alignment/reward-hacking/)：检查 best-of-N 是否把代理评分器的捷径放大。
- [RLHF 与 DPO](../alignment/rlhf-dpo/)：比较 reward model、PPO 和离线偏好优化路径。
- [PPO](../alignment/ppo/)：说明 rollout、advantage、KL 和策略更新。
- [DPO](../alignment/dpo/)：说明固定偏好对、reference 和 log-ratio loss。
- [温度采样](../inference/temperature-sampling/)：控制 proposal policy 的候选分布和多样性。
- [Top-k 与 Top-p 采样](../inference/top-k-top-p/)：说明候选 token 过滤如何改变生成分布。
- [训练数据](../pretraining/training-data/)：检查 accepted/rejected 数据的覆盖、切分和版本。
