---
title: "知识蒸馏：用 teacher 分布约束 student"
tags: ["why-models-learn"]
---

知识蒸馏用一个冻结的 teacher 模型提供软目标，让较小的 student 学习真实标签以外的类别相对关系、token 概率和行为分布。训练时需要同时保存 teacher/student 的 logits、温度、hard/soft loss 权重、tokenizer 对齐、mask、teacher 模式和缓存合同。蒸馏可以压缩模型，也可以迁移行为或保持旧能力；teacher 的错误、校准偏差和评测口径也会进入 student。

![知识蒸馏示意图：同一个输入分别经过冻结 teacher 和可训练 student，teacher logits 经温度得到 soft target，与真实标签共同形成 student 的训练损失](/assets/finetuning/svg/distillation.1.svg)

## teacher 和 student 先共享同一个输出接口

### logits 是未归一化的输出

对输入序列 $x$，teacher 和 student 分别输出：

$$
z_t=f_{\theta_t}(x),
\qquad
z_s=f_{\theta_s}(x).
$$

分类任务中 $z\in\mathbb R^C$，语言模型中 $z\in\mathbb R^{T\times V}$。logits 还没有经过 softmax；它们可以整体平移而不改变概率：

$$
\operatorname{softmax}(z)_i
=
\frac{\exp(z_i)}
{\sum_j\exp(z_j)}.
$$

实现 softmax 时应先减去最大 logit，防止指数溢出：

$$
\operatorname{softmax}(z)_i
=
\frac{\exp(z_i-\max_j z_j)}
{\sum_k\exp(z_k-\max_j z_j)}.
$$

### teacher 参数保持冻结

标准 offline distillation 的计算图为：

$$
p_t^{(\tau)}
=
\operatorname{softmax}\left(\frac{z_t}{\tau}\right),
\qquad
p_s^{(\tau)}
=
\operatorname{softmax}\left(\frac{z_s}{\tau}\right).
$$

teacher 处于 eval 模式并关闭梯度，student 处于 train 模式并接收 loss 梯度。teacher logits 可以在线计算，也可以在固定 tokenizer、模板和量化配置下预先缓存。

|对象|训练状态|输出|需要记录|
|---|---|---|---|
|teacher|eval、no grad|$z_t$ 或 $p_t$|checkpoint、dtype、版本|
|student|train|$z_s$|可训练参数、optimizer|
|hard label|固定 target|$y$|label mapping、mask|
|soft target|由 teacher 产生|$p_t^{(\tau)}$|温度、缓存精度|

### 输出空间必须可比较

若 teacher 和 student 共享分类标签或 tokenizer，logits 的最后一轴可以逐项比较。若词表不同，不能直接对不同 token ID 做 KL；需要建立 token、字符片段或序列级对齐。只把两个 vocab size 改成相同数字，不能证明语义类别对应。

## hard loss 和 soft loss

### hard target 提供任务标签

真实标签 $y$ 的交叉熵为：

$$
\mathcal L_{\mathrm{hard}}
=
-\log p_s^{(1)}(y).
$$

它直接约束 student 在标注任务上的目标类别或目标 token。数据标签错误时，hard loss 会把错误信号传入 student。

### KL loss 提供类别关系

teacher 与 student 的温度分布之间的蒸馏损失为：

$$
\mathcal L_{\mathrm{KD}}
=
\tau^2
\operatorname{KL}
\left(
p_t^{(\tau)}
\mathbin{\Vert}
p_s^{(\tau)}
\right).
$$

展开为：

$$
\mathcal L_{\mathrm{KD}}
=
\tau^2
\sum_i
p_{t,i}^{(\tau)}
\log
\frac{p_{t,i}^{(\tau)}}
{p_{s,i}^{(\tau)}}.
$$

最终损失通常写成：

$$
\mathcal L
=
\alpha\mathcal L_{\mathrm{hard}}
+
(1-\alpha)\mathcal L_{\mathrm{KD}}.
$$

$\alpha$、$\tau$ 和损失的归约方式共同决定真实标签与 teacher 分布的相对权重。不同实现可能把 $\alpha$ 定义为 soft loss 权重，实验记录必须写出完整公式。

### 为什么需要 $\tau^2$

温度增大后，softmax 的分布变平，student 的 soft target 梯度量级会按约 $1/\tau^2$ 缩小。乘上 $\tau^2$ 可以在改变分布平滑程度时保持 KD 梯度的数量级更接近：

$$
\frac{\partial\mathcal L_{\mathrm{KD}}}{\partial z_s}
\approx
\tau
\left(
p_s^{(\tau)}-p_t^{(\tau)}
\right)
\quad\text{若损失已经包含 }\tau^2.
$$

这里的近似依赖 softmax 和 KL 的具体写法。若实现把温度、KL 归约或额外 scale 放在别处，不能重复乘 $\tau^2$。

## 温度改变 teacher 信号的分辨率

### $\tau=1$ 保留原始分布

$\tau=1$ 使用标准 softmax。teacher 最大 logit 的概率可能很高，其他类别的相对关系在数值上变小。student 主要接收 top class 的信号。

### $\tau>1$ 暴露相对偏好

增大 $\tau$ 会让分布变平，非最大类别仍保留可区分的相对概率。它可以传递“次优类别之间的相似性”，但也会降低分布的置信度。温度过高时，teacher 的微小 logit 差异和数值噪声可能占据更大比例。

|温度|分布形状|soft target 信号|风险|
|---:|---|---|---|
|1|尖锐|主类别强|关系信息少|
|2|较平|次优类别可见|需匹配 scale|
|4|更平|更多类别有梯度|噪声和校准敏感|
|过高|接近均匀|类别区分变弱|KD 约束变松|

比较温度时，应固定 teacher checkpoint、student 初始化、hard/soft 权重、有效 token 和训练步数。只改变温度但不调整或报告 $\tau^2$，结果无法归因于分布平滑。

## 一个可运行的 logits 蒸馏探针

下面的探针使用 Python 标准库，核对稳定 softmax、hard CE、不同温度下的 KL、$\tau^2$ 缩放和混合损失。它还计算一个小 batch 的 logits 存储字节数，不包含 hidden、attention 和 optimizer state。

```python
from math import exp, log

def softmax(logits, temperature):
    scaled = [value / temperature for value in logits]
    pivot = max(scaled)
    weights = [exp(value - pivot) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]

def kl(target, predicted):
    return sum(a * log(a / b) for a, b in zip(target, predicted))

def fmt(values):
    return '[' + ', '.join(f'{value:.9f}' for value in values) + ']'

teacher = [2.0, 1.0, 0.0]
student = [1.5, 1.0, 0.5]
label = 0

hard_probability = softmax(student, 1.0)[label]
hard_ce = -log(hard_probability)
print('teacher_prob_tau1=', fmt(softmax(teacher, 1.0)))
print('student_prob_tau1=', fmt(softmax(student, 1.0)))
print('hard_ce=', f'{hard_ce:.9f}')

for tau in [1.0, 2.0]:
    teacher_prob = softmax(teacher, tau)
    student_prob = softmax(student, tau)
    raw_kl = kl(teacher_prob, student_prob)
    scaled_kd = tau * tau * raw_kl
    total = 0.5 * hard_ce + 0.5 * scaled_kd
    print('tau=', f'{tau:.1f}',
          'teacher_prob=', fmt(teacher_prob),
          'student_prob=', fmt(student_prob),
          'raw_kl=', f'{raw_kl:.9f}',
          'scaled_kd=', f'{scaled_kd:.9f}',
          'total=', f'{total:.9f}')

print('teacher_logits_bytes=', 2 * 4 * 5 * 2)
print('student_logits_bytes=', 2 * 4 * 5 * 2)
print('teacher_student_parameter_ratio=', f'{7_000_000_000 / 1_300_000_000:.6f}')
```

运行输出：

```text
teacher_prob_tau1= [0.665240956, 0.244728471, 0.090030573]
student_prob_tau1= [0.506480391, 0.307195886, 0.186323723]
hard_ce= 0.680269671
tau= 1.0 teacher_prob= [0.665240956, 0.244728471, 0.090030573] student_prob= [0.506480391, 0.307195886, 0.186323723] raw_kl= 0.060268897 scaled_kd= 0.060268897 total= 0.370269284
tau= 2.0 teacher_prob= [0.506480391, 0.307195886, 0.186323723] student_prob= [0.419228952, 0.326495836, 0.254275213] raw_kl= 0.019107581 scaled_kd= 0.076430323 total= 0.378349997
teacher_logits_bytes= 80
student_logits_bytes= 80
teacher_student_parameter_ratio= 5.384615
```

τ=1 时 teacher、student 的概率分别为 [0.665240956,0.244728471,0.090030573] 和 [0.506480391,0.307195886,0.186323723]，hard CE 为 0.680269671。τ=2 使分布变平，raw KL 从 0.060268897 降到 0.019107581；乘上 $\tau^2$ 后 scaled KD 为 0.076430323，和 hard loss 等权得到 0.378349997。2×4×5 的 FP16 logits 各占 80 bytes，teacher 参数量与 student 参数量的比率为 5.384615。

## teacher 数据和缓存策略

### offline distillation 先缓存 teacher 输出

offline 方案先用 teacher 对固定数据运行，保存 logits、top-k logits 或 soft target，再训练 student。它把 teacher 推理和 student 优化解耦，便于重复实验；代价是缓存文件会随样本数、序列长度和 vocab 增长。

若保存完整 logits，字节数近似为：

$$
M_{\mathrm{logits}}
=BTVb_{\mathrm{logit}}.
$$

保存 top-k 时，除了概率或 logits，还需要保存 token ID 和截断规则：

$$
M_{\mathrm{topk}}
\approx
BTK(b_{\mathrm{value}}+b_{\mathrm{id}}).
$$

完整 logits 和 top-k logits 的 KD 目标不同。top-k 之外的概率可以被聚合成 remainder bucket，也可以被丢弃；这两种处理不能混写。

### online distillation 同时运行 teacher 和 student

online 方案在同一个训练 step 中计算 teacher 和 student。teacher 结果始终 no grad，但 teacher 的参数、activation、workspace 和前向时间仍然会占资源。若显存不足，可以分批运行 teacher、低精度缓存 logits 或改用 offline。

### EMA teacher 会改变 teacher 定义

有些方案使用 student 参数的指数移动平均作为 teacher：

$$
\theta_{\mathrm{ema},k}
=
\beta\theta_{\mathrm{ema},k-1}
+
(1-\beta)\theta_{\mathrm{student},k}.
$$

此时 teacher 不是固定预训练 checkpoint。$\beta$、更新时机、是否 stop-gradient 和 checkpoint 恢复都必须记录。

|策略|teacher 来源|资源|可复现字段|
|---|---|---|---|
|offline|固定 checkpoint|缓存大、训练快|cache hash、dtype、版本|
|online|每步前向|计算和显存高|teacher eval/no-grad|
|EMA|student 的移动平均|额外 state|$\beta$、更新时机|
|self-distill|同一模型或历史 checkpoint|依赖实现|teacher checkpoint 边界|

## tokenizer 和 vocab 对齐

### 相同 tokenizer 可以逐 token 蒸馏

teacher 和 student 共享 tokenizer、词表和模板时，语言模型 logits 具有相同的 $(B,T,V)$ 合同。padding、BOS、EOS、role separator 和 target shift 也必须一致：

$$
z_t,z_s\in\mathbb R^{B\times T\times V}.
$$

对 assistant-only loss，KD mask 应与 student 的有效 target mask 对齐；prompt token 可以提供条件，但不应无意中进入损失分母。

### 不同 tokenizer 需要显式映射

teacher 和 student 使用不同词表时，直接计算同位置 KL 没有语义基础。可选策略包括：

1. 把双方输出映射到字符或 byte span，再比较 span 条件概率。
2. 按 student token 的字符串片段聚合 teacher 的多个 token 概率。
3. 使用序列级 teacher 生成和 student 生成做 response distillation。
4. 只在共享标签或分类 head 上做 task-level distillation。

每种映射都会改变目标分布和归约。应保存 tokenizer 版本、对齐算法、不可对齐 token 的处理和 loss mask。

|输出空间|蒸馏粒度|主要风险|
|---|---|---|
|相同 class label|class logits|label mapping 错位|
|相同 vocab|逐 token KL|position、mask、EOS 不一致|
|不同 subword vocab|span 概率|聚合与归一化错误|
|不同 tokenizer|sequence response|teacher sampling 方差|
|不同任务 head|feature 或 behavior|目标定义不唯一|

## 蒸馏不只有 logits

### response distillation

把 teacher 生成的答案作为 student 的监督 target，和 SFT 类似：

$$
\mathcal L_{\mathrm{response}}
=
\operatorname{CE}
\left(
y_{\mathrm{teacher}},
y_{\mathrm{student}}
\right).
$$

它可以迁移格式和表达风格，但 teacher 生成错误会变成训练 target，sampling、temperature、top-p、stop 和过滤策略需要固定。

### feature distillation

若 teacher 和 student 的 hidden size 不同，可以用投影层 $R$ 对齐：

$$
\mathcal L_{\mathrm{hidden}}
=
\left\lVert
R h_s-h_t
\right\rVert_2^2.
$$

投影层是否训练、从哪几层取 hidden、padding 如何 mask 和 feature 归一化都会影响结果。

### attention 和关系蒸馏

attention map 或 token 间关系矩阵可以作为额外目标：

$$
\mathcal L_{\mathrm{attn}}
=
\left\lVert
A_s-A_t
\right\rVert_2^2.
$$

不同 head 数、层数、位置编码和稀疏 attention 实现可能无法逐项对齐。关系蒸馏可以先聚合 head 或 layer，再比较矩阵；聚合规则必须明确。

|蒸馏类型|目标|适合检查|
|---|---|---|
|logit KD|teacher 概率|类别或 token 关系|
|response|teacher 生成|格式和序列行为|
|hidden|中间表示|表征和层间接口|
|attention|读取权重|token 关系和对齐|
|feature relation|距离或相似度|结构保持|

## student 容量决定能否承接 teacher

### teacher 分布不能绕过 student 上限

teacher 可以给出丰富的 soft target，但 student 的层数、hidden size、vocab、上下文长度和输出 head 限制了可表达函数族。若 student 过小，KD loss 可能持续下降有限，任务分数仍受容量限制。

蒸馏比较应至少包含：

|对照|目的|
|---|---|
|student from scratch|测初始化和容量|
|student hard-only|测 soft target 的增益|
|student hard+KD|测蒸馏损失|
|teacher|给出上限和错误边界|
|student with different $\tau$|测温度敏感性|

### teacher 错误会被平滑传播

teacher 的高置信错误、偏见、格式错误和安全边界会进入 soft target 或 response target。使用 teacher 生成数据时，应保留人工或规则过滤、事实校验和旧任务回归；teacher 分数不能作为 student 正确性的充分条件。

### calibration 影响 soft target

logits 的相对差异决定 soft target。teacher 过度自信时，$\tau$ 可以平滑分布，但不能恢复丢失的真实校准；teacher 过度平滑时，KD 会提供弱区分信号。需要报告 teacher 的 calibration、temperature、top-k 质量和任务分组表现。

## 运行方法

将上一个 Python 代码块保存为 distillation-logits-probe.py，再运行 python3 distillation-logits-probe.py。修改 teacher/student logits、temperature、hard/soft 权重或 batch shape 后，应同步检查 softmax、KL、$\tau^2$ 和缓存字节数。

接入真实模型时，先固定 teacher checkpoint、student architecture、tokenizer、template 和数据切分，再分别运行 hard-only、KD-only 和 mixed loss。训练中记录 teacher mode、student gradient、有效 token、temperature、logits cache、checkpoint 和分组评测。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|KD loss 下降但任务分数不升|student 容量、tokenizer 或 teacher 错误|加入 hard-only/student 对照|
|温度变大后训练变慢|$\tau^2$、soft loss 权重或梯度 scale|记录 raw/scaled KD 和 grad norm|
|teacher/student KL 无法计算|vocab、position 或 mask 不对齐|检查 logits shape 和 token mapping|
|offline 与 online 结果不同|cache dtype、teacher mode 或版本|保存 cache hash 和 no-grad 证据|
|teacher logits 缓存过大|完整 vocab、长度或 dtype|比较 full/top-k/remainder 存储|
|student 学到 teacher 的格式错误|response target 或过滤不足|按格式和事实切片回归|
|蒸馏后旧能力退化|teacher 输入覆盖不足或训练更新冲突|运行 old/new regression matrix|
|hidden/attention 蒸馏不稳定|层、head、shape 或归一化不匹配|记录投影、聚合和 mask|
|恢复训练结果不同|teacher cache、RNG、optimizer 或 sampler|固定输入运行 deterministic probe|
|报告只写 distillation|未说明目标、温度、权重和 teacher|保存完整 loss contract|

### 最小审计表

|审计层|应保存|验收问题|
|---|---|---|
|teacher|内容 hash、mode、dtype、版本|teacher 输出是否固定|
|student|架构、参数、optimizer、checkpoint|student 能否恢复|
|data|tokenizer、template、切分、有效 token|输入和 mask 是否一致|
|target|logit、response、hidden 或 attention|蒸馏目标是什么|
|temperature|$\tau$、$\tau^2$、hard/soft 权重|梯度 scale 是否明确|
|alignment|vocab、position、layer、head、span mapping|双方输出能否比较|
|cache|shape、dtype、hash、top-k/remainder|offline 输出能否复用|
|quality|teacher、student hard-only、KD、分组结果|蒸馏增益和错误来自哪里|
|deployment|decode、stop、format、safety|student 行为是否可复现|

知识蒸馏把 teacher 的分布、行为或中间表示变成 student 的训练信号。它不会消除 student 的容量上限，也不会自动修复 teacher 的错误。报告需要同时给出 hard loss、soft loss、温度、对齐规则、teacher 版本和 student 质量。

## 相关词条

- [灾难性遗忘](../finetuning/catastrophic-forgetting/)：检查蒸馏或新任务训练后的旧能力回归。
- [监督微调](../finetuning/sft/)：说明 hard target、mask 和 next-token loss。
- [指令数据](../finetuning/instruction-data/)：审计 teacher/student 使用的任务数据和过滤。
- [Softmax](../neurons-and-activations/softmax/)：推导 logits 到概率的稳定归一化。
- [交叉熵](../information-theory/cross-entropy/)：说明 hard label 和概率分布损失。
- [KL 散度](../information-theory/kl-divergence/)：说明 teacher/student 分布差异。
- [Temperature Sampling](../inference/temperature-sampling/)：比较温度对分布形状的影响。
- [模型容量](../learning-framework/model-capacity/)：说明 student 能表达的函数范围。
- [训练稳定性](../pretraining/training-stability/)：检查蒸馏训练的梯度、精度和恢复。
