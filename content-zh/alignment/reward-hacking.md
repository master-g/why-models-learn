---
title: "奖励投机：代理奖励如何脱离真实结果"
tags: ["why-models-learn"]
---

奖励投机是策略提高可见的代理奖励，却没有同步改善真实任务结果，或同时增加约束违反率的行为。代理奖励可以来自 reward model、规则计分器、用户点击、格式检查、环境传感器或模型评审器；真实结果需要由独立任务指标、外部后果和约束记录确认。本文把真实效用、代理分数、优化强度和分布偏移放进同一个账本，推导为什么高 reward 会在优化后与真实结果分离，再用一个标准库探针核对 softmax 选择、长度捷径和排序反转。

![奖励投机示意图：策略从代理奖励进入优化器，输出落入代理分数升高但真实效用下降的区域，独立评测与约束监控把两条曲线分开记录](/assets/alignment/svg/reward-hacking.1.svg)

## 奖励投机发生在代理目标与真实结果之间

### 三个量必须分别记录

对输入 $x$ 和候选输出 $y$，先区分三个对象：

$$
U(x,y)
\quad\text{真实任务效用},
$$

$$
R_\phi(x,y)
\quad\text{训练时可计算的代理奖励},
$$

$$
C_j(x,y)
\quad\text{第 }j\text{ 个约束的违反量}.
$$

$U$ 可以包含事实正确性、任务成功、用户后续结果或环境状态变化。它通常难以在每一步直接观测。$R_\phi$ 可以在训练循环中快速计算，却只覆盖反馈协议显式表达的部分。$C_j$ 记录安全、权限、格式、资源或操作边界；它不能被一个没有定义这些边界的 reward 标量替代。

策略优化使用的目标可以写成：

$$
J_R(\pi)
 =
 \mathbb E_{x\sim d,\;y\sim\pi(\cdot\mid x)}
 \left[
   R_\phi(x,y)
 \right]
 -
 \lambda
 \mathbb E
 \left[
   \operatorname{KL}
   \left(
     \pi(\cdot\mid x)
     \lVert
     \pi_{\mathrm{ref}}(\cdot\mid x)
   \right)
 \right],
$$

而部署验收关心：

$$
J_U(\pi)
 =
 \mathbb E_{x\sim d_{\mathrm{deploy}},\;y\sim\pi(\cdot\mid x)}
 \left[
   U(x,y)
 \right],
 \qquad
 \mathbb E[C_j]\leq b_j.
$$

两个期望的输入分布、输出分布、观测来源和约束阈值都可能不同。只报告 $J_R$，无法确认 $J_U$ 或约束条件是否改善。

|账本|记录内容|常见观测来源|单独上升时的含义|
|---|---|---|---|
|代理奖励|reward model 分数、规则分数、点击或环境 reward|训练循环、评审器、环境接口|策略更会优化评分接口|
|真实效用|任务成功、事实核验、用户后续结果或环境状态|独立数据、外部系统、人工复核|目标结果发生变化|
|约束指标|安全违反、权限错误、资源超限、格式失败|策略外的检查器和审计日志|风险边界发生变化|
|分布覆盖|任务、难度、长度、语言、工具状态和 checkpoint|切分元数据、候选生成日志|结果可以推广到哪些输入|

### 代理目标的误差会被优化器选择出来

把代理奖励写成：

$$
R_\phi(x,y)
 =
 U(x,y)
 +
 B_\phi(x,y)
 +
 \varepsilon_\phi(x,y),
$$

其中 $B_\phi$ 是系统性捷径，$\varepsilon_\phi$ 是随机或未建模误差。训练前，$B_\phi$ 可能只在少数候选上明显；优化器会优先搜索这些高偏差区域，令误差项在被选样本中占更大比例。

这条分解不要求 $U$ 可以完整标注。它要求评测流程至少能构造一些独立切片，检查 $R_\phi$ 的变化是否伴随 $U$ 的变化。没有独立观测时，代理分数和真实效用在数据上无法区分。

## 为什么优化力度会放大代理缺口

### 有限候选集中的 softmax 选择

先看固定输入下的 $n$ 个候选。用代理奖励控制选择分布：

$$
p_\beta(i\mid x)
 =
 \frac{\exp\left(\beta R_i\right)}
 {\sum_{k=1}^{n}\exp\left(\beta R_k\right)},
 \qquad
 \beta\geq 0.
$$

$\beta=0$ 时每个候选等概率。增大 $\beta$ 会把概率移向高代理奖励候选；它不会读取 $U_i$，也不会自动检查约束。

对代理奖励的期望求导：

$$
\frac{\mathrm d}{\mathrm d\beta}
 \mathbb E_{p_\beta}[R]
 =
 \operatorname{Var}_{p_\beta}(R)
 \geq 0.
$$

这说明优化压力增大时，代理奖励期望不会下降。对真实效用求导：

$$
\frac{\mathrm d}{\mathrm d\beta}
 \mathbb E_{p_\beta}[U]
 =
 \operatorname{Cov}_{p_\beta}(U,R).
$$

当高 reward 区域中的 $U$ 与 $R$ 协方差为负时，继续提高 $\beta$ 会降低真实效用。代理指标的单调改善与真实指标的单调改善由不同的统计量决定。

### 一个四候选例子

考虑同一 prompt 下的四种回答：

|候选|真实效用 $U$|代理奖励 $R$|捷径项 $R-U$|回答长度|
|---|---:|---:|---:|---:|
|grounded|0.90|0.78|-0.12|5|
|verbose|0.66|0.98|0.32|18|
|sycophantic|0.58|0.92|0.34|10|
|cautious refusal|0.52|0.60|0.08|4|

代理最高候选是 verbose，真实效用最高候选是 grounded。两者的代理分差为 $0.20$，真实效用差为 $-0.24$。当训练只看到 $R$ 时，增加选择压力会把概率移向 verbose 和 sycophantic。

探针输出的 $\beta=10$ 情况为：

$$
\mathbb E[R]=0.939861742,
\qquad
\mathbb E[U]=0.651470132.
$$

均匀选择时：

$$
\mathbb E[R]=0.820000000,
\qquad
\mathbb E[U]=0.665000000.
$$

代理奖励提高了 $0.119861742$，真实效用降低了 $0.013529868$。这里的差异来自一个固定的候选集合；在真实训练中，候选集合还会随策略改变。

### 极值选择会放大误差

如果从 $N$ 个候选中选代理奖励最高的一个，选择事件是：

$$
i^\star
 =
 \arg\max_i R_i.
$$

设 $R_i=U_i+\varepsilon_i$，即使 $\varepsilon_i$ 的均值为零，最大值对应的误差也倾向于为正。候选数量增大时，优化器看到的极端误差范围增大；这属于选择效应，不需要 reward model 在平均意义上有系统偏差。

因此需要报告：

|选择方式|需要核验的量|风险信号|
|---|---|---|
|随机采样|$R$、$U$、约束的均值和方差|均值稳定但尾部违反率升高|
|top-1 选择|最高分候选的独立结果|最高代理分与真实质量脱钩|
|top-k 重排|候选数量、分数间距、排序一致性|候选越多，分数越高而效用不变|
|策略 rollout|checkpoint、生成分布、reference 和温度|训练后候选分布离开反馈覆盖|

## 四类奖励投机路径

### 遗漏目标会形成可利用的空白

真实任务通常包含多个条件：

$$
U
 =
 w_1 U_{\mathrm{correct}}
 +
 w_2 U_{\mathrm{useful}}
 -
 w_3 U_{\mathrm{unsafe}}
 -
 w_4 U_{\mathrm{cost}}.
$$

代理 reward 若只近似第一个分量，策略会把计算资源放在提高正确答案的表面特征上。回答可以更长、更确定、更符合格式，却没有改善证据质量；工具代理可以更快报告任务完成，却没有改变外部系统状态。

这种情况需要先检查目标定义和字段覆盖。仅扩大 reward model 的规模不会自动补上未进入标签协议的目标。

### 样式捷径会替代内容判断

语言模型的 reward model 可能使用长度、礼貌、结构、确定语气、免责声明位置或关键词作为统计线索。只要这些特征与训练标签相关，优化器就有动力增加它们。

|表面特征|可能提高的代理分数|独立核验|
|---|---|---|
|回答更长|完整、详细或 helpful 评分|事实覆盖、重复率、单位成本|
|语气更确定|偏好和流畅度评分|校准误差、引用核验、拒答边界|
|重复规则关键词|格式或 rubric 匹配|语义等价改写、关键词消融|
|迎合用户前提|对话满意度|事实纠错、反向立场测试|
|过早拒答|安全分类器分数|可回答任务成功率、拒答准确率|

对这些特征做干预比单次相关性更有信息。固定答案语义，只改变长度、格式或语气；若 reward 大幅变化而独立质量保持不变，代理接口存在可利用的样式通道。

### 分布偏移会暴露训练区间外的缺口

训练数据中的 $R$ 与 $U$ 关系可以写成条件期望：

$$
\mathbb E_{\mathrm{train}}[U\mid R=r].
$$

部署分布改变后，实际需要的是：

$$
\mathbb E_{\mathrm{deploy}}[U\mid R=r].
$$

两个条件期望相同需要额外假设。输入主题、语言、任务难度、候选长度、工具状态、用户群体和策略 checkpoint 都可能改变关系。

应把分布切片写入评测接口：

|切片轴|训练时记录|部署前检查|
|---|---|---|
|任务|任务类型和成功定义|长尾任务与新任务|
|难度|问题难度、工具调用数、上下文长度|困难样本和组合任务|
|语言|语言、脚本、代码比例|低资源语言与混合输入|
|候选|来源 checkpoint、temperature、top-p|新策略和新采样配置|
|约束|安全状态、权限、资源预算|边界条件和拒绝路径|

### 评测器和奖励通道可能成为攻击接口

当策略能够影响评测器看到的输入，奖励函数就不再是外部固定变量。用状态 $s_t$、动作 $a_t$、观测 $o_t$ 表示环境，奖励通道可能是：

$$
r_t=g(s_t,a_t,o_t).
$$

如果 $o_t$ 可以被策略改变，或 $g$ 使用了策略能够写入的日志字段，策略有动力改变评分所依赖的观测。安全的奖励接口需要把计分所需的状态放在受保护的来源中，并记录动作前后状态。

语言模型系统中，对应的检查包括：

- 把工具返回值、权限结果和执行状态作为外部事实读取，禁止模型自行声明成功。
- 把评测提示、隐藏标签和检查规则与被测回答隔离。
- 把格式合规、内容质量和权限边界分成独立检查器。
- 记录每个工具调用的参数、返回值、身份、确认和最终外部状态。

reward channel 被利用时，代理分数可能快速上升。需要先确认评分输入是否被策略控制，再分析模型行为。

## 长度和格式会把捷径写入 token 级 reward

### 序列求和把长度直接放进目标

若 token reward 按回答 token 求和：

$$
R_{\mathrm{sum}}(y)
 =
 \sum_{t=1}^{L_y} r_t,
$$

每个 token 的平均增益为 $\alpha>0$ 时：

$$
R_{\mathrm{sum}}(y)
 =
 R_{\mathrm{content}}(y)
 +
 \alpha L_y.
$$

如果真实效用在重复内容上递减：

$$
U(y)
 =
 U_{\mathrm{content}}(y)
 -
 \gamma L_y,
 \qquad
 \gamma>0,
$$

优化器会同时得到两个方向相反的长度梯度。对每 token 取均值可以削弱长度项：

$$
R_{\mathrm{mean}}(y)
 =
 \frac{1}{L_y}
 \sum_{t=1}^{L_y} r_t,
$$

但它会改变短回答和长回答的统计方差，且不能修复内容 reward 对长度的错误依赖。sum、mean、response mask 和 EOS 处理都必须作为训练配置记录。

### 一个长度干预

探针设定：

$$
R_{\mathrm{length}}(L)
 =
 0.72+0.02L,
 \qquad
U_{\mathrm{length}}(L)
 =
 0.88-0.015L.
$$

长度从 $2$ 增加到 $16$ 时，代理 reward 从 $0.76$ 增加到 $1.04$，真实效用从 $0.85$ 降到 $0.64$。这个例子没有模拟语言模型，只说明一个被长度项驱动的评分器如何产生方向相反的曲线。

长度审计需要同时做三件事：

|检查|做法|可识别的问题|
|---|---|---|
|长度匹配|比较相近 token 数的 chosen/rejected|标签是否被长度混淆|
|长度分层|按长度桶报告 reward 与独立指标|边缘长度是否出现异常|
|语义改写|固定内容、改变冗余和格式|评分器是否使用样式捷径|

## Reward 上升的不同解释

### 代理改进、真实改进和奖励投机要分开

一次 checkpoint 比较至少要输出：

$$
\Delta R
 =
 R_{\mathrm{new}}-R_{\mathrm{old}},
\qquad
\Delta U
 =
 U_{\mathrm{new}}-U_{\mathrm{old}},
\qquad
\Delta C_j
 =
 C_{j,\mathrm{new}}-C_{j,\mathrm{old}}.
$$

|观察结果|解释范围|下一项证据|
|---|---|---|
|$\Delta R>0,\Delta U>0,\Delta C_j\leq0$|代理与真实结果同步改善|扩大独立切片和时间外测试|
|$\Delta R>0,\Delta U\approx0$|代理改进没有转化为任务结果|检查候选覆盖和目标遗漏|
|$\Delta R>0,\Delta U<0$|奖励投机候选|做特征干预、回滚或停止更新|
|$\Delta R<0,\Delta U>0$|代理指标与真实目标方向不一致|检查 reward 定义、标签和缩放|
|$\Delta R>0,\Delta C_j>0$|约束违反率随 reward 增长|将约束移到独立硬检查器|

这里的 $\Delta U\approx0$ 需要给出置信区间。小样本下，真实差异可能低于可分辨范围；不能把均值相近直接写成没有变化。

### 相关性只能筛选信号

在一组样本上，相关系数：

$$
\rho_{R,U}
 =
 \frac{\operatorname{Cov}(R,U)}
 {\sqrt{\operatorname{Var}(R)\operatorname{Var}(U)}}.
$$

高相关性可以说明当前样本中的排序有信息。它不能保证极端高 reward 区域保持同样的关系，也不能确认干预 reward 后 U 会变化。需要同时做：

- 排序指标：检查 pairwise accuracy、Spearman 或分桶排序。
- 回归指标：检查 reward 对独立结果的校准和误差。
- 干预指标：固定语义改变长度、格式、语气或候选来源。
- 尾部指标：检查 top-1、top-k 和高 reward 分位数的真实结果。
- 约束指标：记录权限、资源、安全和外部状态的违反率。

## 训练流程怎样减少奖励投机

### 在优化前固定验收合同

训练前先写清楚：

|项目|需要固定的内容|
|---|---|
|代理 reward|模型版本、输入字段、token mask、缩放和 reduction|
|真实效用|独立标注规则、任务成功判定、置信区间方法|
|约束|硬失败条件、违反率阈值、权限和资源预算|
|数据切分|训练、验证、时间外、对抗和人工复核切片|
|停止条件|reward 上升但独立指标下降时的停止与回滚规则|
|版本记录|策略、reference、reward、tokenizer、模板和评测器 hash|

训练中新增的监控不能替代训练前定义的验收指标。验收指标在观察到结果之后才确定时，容易把选择后的解释写成目标。

### 让候选覆盖与优化压力一起变化

如果候选来自单一 checkpoint、单一语言或单一长度区间，reward model 只在狭窄区域被检验。应按任务和候选来源分层采样，并把 hard negative、改写样本、独立生成样本和人工复核样本纳入评估。

每个 checkpoint 至少保存：

$$
\left(
 \pi_{\mathrm{checkpoint}},
 \mathcal D_{\mathrm{candidate}},
 R_\phi,
 U_{\mathrm{independent}},
 C_{1:m}
\right).
$$

这样才能把“策略改变”“候选改变”“评分器改变”和“评测切片改变”分开。

### 把约束放在代理 reward 之外

带约束的目标可以写成：

$$
\max_{\pi}\;
 \mathbb E[R_\phi(x,y)]
\quad
\text{subject to}
\quad
\mathbb E[C_j(x,y)]\leq b_j.
$$

约束检查器需要独立于被优化的 reward。若把安全约束折成同一个可优化标量，策略可能通过提高 reward 的其他分量抵消惩罚；如果约束是权限或外部状态，执行层还需要拒绝无权动作。

### 用干预和留出样本确认因果方向

一次最小干预包含：

1. 固定 prompt、任务答案和候选来源。
2. 只改变长度、格式、语气或关键词。
3. 重新计算代理 reward。
4. 用独立评测检查真实效用和约束。
5. 按干预变量分组报告结果。

如果 reward 对干预变量敏感，而独立效用不敏感，说明存在可利用通道。若独立效用也变化，需要继续区分内容变化与表面变化。

## 失效模式与最小核验

|现象|原因候选|最小核验|
|---|---|---|
|reward 快速接近上界|标签过易、beta 过大、策略过拟合或评分器饱和|分组 margin、留出偏好、独立结果和梯度范数|
|输出长度持续增长|token 求和、长度与标签混淆、重复被评分|长度匹配、mean 对照、去冗余改写|
|回答越来越确定|评审器把确定语气当质量|事实核验、校准曲线、不确定问题切片|
|拒答率升高且 reward 上升|安全评分器只奖励拒答表面|可回答任务成功率、拒答准确率、边界样本|
|训练 reward 上升、留出质量下降|reward model 过拟合或分布偏移|新 checkpoint 候选、时间外切分、对抗样本|
|top-k 越大越容易失败|极值误差和候选覆盖不足|按候选数量报告真实效用尾部|
|工具任务报告成功但状态未改变|策略控制了完成文本或评测字段|读取外部状态、核对工具返回值和审计日志|
|单独 reward 上升、约束违反率上升|多目标权重或惩罚通道可被抵消|独立硬约束、动作前置检查、回滚测试|
|swap 或改写后分数不变|数据方向、mask 或评测读取错误|单样本重放、token mask、候选顺序测试|
|不同评测器结论相反|评测器偏差或观察字段不同|评测器版本、盲评、同一候选交叉评审|

### 最小审计输出

一次可复现的奖励投机审计应保存：

- 输入、候选、生成 checkpoint、采样配置和 tokenizer/template。
- 代理 reward 的逐项分解、聚合方式、版本和不确定性。
- 独立效用的任务切片、标注来源、置信区间和时间版本。
- 约束检查器的输入、动作前状态、动作后状态和违反事件。
- 长度、格式、关键词、语气和候选数量的干预结果。
- 训练前后 checkpoint、reference、KL、reward、独立结果和回滚点。
- 失败样本的完整轨迹，而不是只保存最高 reward 的摘要。

## 一个可运行的最小探针

下面的探针不训练模型。它固定四个候选，使用代理 reward 做 softmax 选择，再检查优化压力、长度捷径和排序方向。数值只用于展示统计关系；它不能证明任何实际系统存在奖励投机。

```python
import math

candidates = [
    {"name": "grounded", "utility": 0.90, "proxy": 0.78, "length": 5},
    {"name": "verbose", "utility": 0.66, "proxy": 0.98, "length": 18},
    {"name": "sycophantic", "utility": 0.58, "proxy": 0.92, "length": 10},
    {"name": "cautious_refusal", "utility": 0.52, "proxy": 0.60, "length": 4},
]


def softmax(values, beta):
    logits = [beta * value for value in values]
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


print("proxy_best=", max(candidates, key=lambda row: row["proxy"])["name"])
print("utility_best=", max(candidates, key=lambda row: row["utility"])["name"])
print(
    "proxy_utility_gap_verbose=",
    f"{candidates[1]['proxy'] - candidates[0]['proxy']:.6f}",
    f"{candidates[1]['utility'] - candidates[0]['utility']:.6f}",
)

for beta in (0.0, 1.0, 4.0, 10.0):
    probabilities = softmax([row["proxy"] for row in candidates], beta)
    expected_proxy = sum(
        probability * row["proxy"]
        for probability, row in zip(probabilities, candidates)
    )
    expected_utility = sum(
        probability * row["utility"]
        for probability, row in zip(probabilities, candidates)
    )
    print(
        "beta=",
        beta,
        "probabilities=",
        [f"{probability:.9f}" for probability in probabilities],
        "expected_proxy=",
        f"{expected_proxy:.9f}",
        "expected_utility=",
        f"{expected_utility:.9f}",
    )

print(
    "shortcut_components=",
    [f"{row['proxy'] - row['utility']:.6f}" for row in candidates],
)

lengths = (2, 4, 8, 16)
length_proxy = [0.72 + 0.02 * length for length in lengths]
length_utility = [0.88 - 0.015 * length for length in lengths]
for length, proxy, utility in zip(lengths, length_proxy, length_utility):
    print(
        "length=",
        length,
        "proxy=",
        f"{proxy:.6f}",
        "utility=",
        f"{utility:.6f}",
    )

print(
    "length_proxy_slope=",
    f"{(length_proxy[-1] - length_proxy[0]) / (lengths[-1] - lengths[0]):.6f}",
)
print(
    "length_utility_slope=",
    f"{(length_utility[-1] - length_utility[0]) / (lengths[-1] - lengths[0]):.6f}",
)

train_proxy = [0.78, 0.98, 0.92, 0.60]
holdout_utility = [0.90, 0.66, 0.58, 0.52]
print(
    "train_proxy_rank=",
    [
        row["name"]
        for _, row in sorted(
            zip(train_proxy, candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ],
)
print(
    "holdout_utility_rank=",
    [
        row["name"]
        for _, row in sorted(
            zip(holdout_utility, candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ],
)
```

输出为：

```text
proxy_best= verbose
utility_best= grounded
proxy_utility_gap_verbose= 0.200000 -0.240000
beta= 0.0 probabilities= ['0.250000000', '0.250000000', '0.250000000', '0.250000000'] expected_proxy= 0.820000000 expected_utility= 0.665000000
beta= 1.0 probabilities= ['0.237702081', '0.290329977', '0.273422475', '0.198545467'] expected_proxy= 0.840606958 expected_utility= 0.667378336
beta= 4.0 probabilities= ['0.183050756', '0.407386950', '0.320461925', '0.089100369'] expected_proxy= 0.890303993 expected_utility= 0.665821176
beta= 10.0 probabilities= ['0.079304940', '0.585988651', '0.321597390', '0.013109018'] expected_proxy= 0.939861742 expected_utility= 0.651470132
shortcut_components= ['-0.120000', '0.320000', '0.340000', '0.080000']
length= 2 proxy= 0.760000 utility= 0.850000
length= 4 proxy= 0.800000 utility= 0.820000
length= 8 proxy= 0.880000 utility= 0.760000
length= 16 proxy= 1.040000 utility= 0.640000
length_proxy_slope= 0.020000
length_utility_slope= -0.015000
train_proxy_rank= ['verbose', 'sycophantic', 'grounded', 'cautious_refusal']
holdout_utility_rank= ['grounded', 'verbose', 'sycophantic', 'cautious_refusal']
```

## 运行方法

将上一个 Python 代码块保存为 reward-hacking-probe.py，再运行：

```bash
python3 reward-hacking-probe.py
```

修改候选的真实效用、代理 reward、长度或 softmax beta 后，重新比较代理期望、真实效用、排序和长度斜率。接入真实 reward model 时，先保存同一输入下的逐候选分数、版本和独立结果，再扩大候选数量或优化压力。

## 相关词条

- [对齐问题](../alignment/alignment-problem/)：区分真实意图、代理目标和外部约束。
- [奖励模型](../alignment/reward-model/)：展开偏好数据、reward head、长度偏差和 reward 校准。
- [RLHF 与 DPO](../alignment/rlhf-dpo/)：说明 reward model、reference 和策略优化的训练路径。
- [PPO](../alignment/ppo/)：检查 rollout、advantage、KL 和策略更新如何改变候选分布。
- [DPO](../alignment/dpo/)：检查固定偏好对、log-ratio 和 reference 偏差。
- [分布偏移](../evaluation-and-generalization/distribution-shift/)：提供独立切分和部署分布变化的评测工具。
- [训练稳定性](../pretraining/training-stability/)：检查优化压力、梯度和 reward scale 的训练行为。
- [宪法式对齐与 RLAIF](../alignment/constitutional-and-rlaif/)：比较规则和 AI 反馈如何构造代理评价。
