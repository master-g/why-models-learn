---
title: "灾难性遗忘：新任务更新如何损伤旧任务"
tags: ["why-models-learn"]
---

灾难性遗忘描述连续训练中的旧任务性能下降：模型从旧任务 checkpoint $\theta_0$ 出发适配新任务，得到 $\theta_1$ 后，旧任务评测变差。旧任务损失上升、准确率下降、行为回归或安全边界变化都可以构成遗忘证据。它不是单一算法故障；数据混合、梯度方向、学习率、更新层、参数共享、评测切分和 checkpoint 对照共同决定观察结果。

![灾难性遗忘示意图：同一个基础 checkpoint 分别进入新任务更新、rehearsal、正则和 adapter 路径，旧任务回归结果与新任务适配结果需要同时记录](/assets/finetuning/svg/catastrophic-forgetting.1.svg)

## 先把“遗忘”写成可测量的差值

### 旧任务和新任务要有独立测试集

设旧任务测试集为 $\mathcal D_{\mathrm{old}}$，新任务测试集为 $\mathcal D_{\mathrm{new}}$。从基础 checkpoint $\theta_0$ 开始，在新任务数据上训练得到 $\theta_1$。旧任务准确率下降可以定义为：

$$
F_{\mathrm{acc}}
=
\operatorname{Acc}_{\mathrm{old}}(\theta_0)
-
\operatorname{Acc}_{\mathrm{old}}(\theta_1).
$$

损失上升则写成：

$$
F_{\mathrm{loss}}
=
\mathcal L_{\mathrm{old}}(\theta_1)
-
\mathcal L_{\mathrm{old}}(\theta_0).
$$

准确率差值为正表示旧任务分数下降，损失差值为正表示旧任务损失增加。两项要在相同数据、相同 decode、相同 scorer 和相同随机种子策略下计算。

|指标|定义|方向|
|---|---|---|
|old accuracy drop|$\operatorname{Acc}_{0}-\operatorname{Acc}_{1}$|越大越差|
|old loss increase|$\mathcal L_{1}-\mathcal L_{0}$|越大越差|
|retention ratio|$\operatorname{Acc}_{1}/\operatorname{Acc}_{0}$|越接近 1 越好|
|new task gain|$\operatorname{Acc}_{\mathrm{new}}(\theta_1)-\operatorname{Acc}_{\mathrm{new}}(\theta_0)$|越大表示适配更强|
|behavior regression|固定 prompt 的输出差异|按规则定义阈值|

### base 对照不能省略

只测训练后的新任务分数，无法判断旧任务是否退化。至少保留：

1. base：$\theta_0$ 在旧任务和新任务上的结果。
2. adapted：$\theta_1$ 在旧任务和新任务上的结果。
3. training protocol：数据、mask、tokenizer、模板、有效 token 和优化器设置。
4. checkpoint：训练前、训练中和训练后的参数或 adapter 标识。

若 base 在旧任务上本来就低分，训练后分数继续低不构成遗忘；若新任务评测切分发生变化，分数差值也不能直接归因于更新。

### 任务分数要拆到能力单元

一个总分可能掩盖局部回归。旧任务评测应按语言、领域、输出格式、难度、长度、工具调用和安全边界分组：

|分组字段|要回答的问题|
|---|---|
|task family|哪类旧能力下降|
|language/domain|退化集中在哪个分布|
|format|结构化输出、文本和代码是否同向变化|
|difficulty|简单样本保持是否掩盖难例遗忘|
|length|长输入或长输出是否更敏感|
|behavior rule|拒答、格式和工具边界是否变化|

## 新任务更新为什么会覆盖旧能力

### 参数共享让两项损失竞争

新任务优化更新为：

$$
\theta_{k+1}
=
\theta_k-\eta g_{\mathrm{new}},
\qquad
g_{\mathrm{new}}
=
\nabla_\theta\mathcal L_{\mathrm{new}}(\theta_k).
$$

旧任务损失变化的一阶近似为：

$$
\Delta\mathcal L_{\mathrm{old}}
\approx
-\eta
\nabla_\theta\mathcal L_{\mathrm{old}}^{\mathsf T}
\nabla_\theta\mathcal L_{\mathrm{new}}.
$$

当两个梯度内积为负时，新任务的一步更新会让旧任务损失上升；内积为正时，更新方向可能同时改善两项；内积接近零时，一阶影响较小，但高阶曲率、后续步骤和共享表示仍可能产生变化。

梯度余弦为：

$$
\cos(g_{\mathrm{old}},g_{\mathrm{new}})
=
\frac{
g_{\mathrm{old}}^{\mathsf T}g_{\mathrm{new}}
}{
\lVert g_{\mathrm{old}}\rVert
\lVert g_{\mathrm{new}}\rVert
}.
$$

实现时要处理零梯度分母。一个 batch 上的余弦只能描述局部方向，不能替代旧任务回归。

### 二次损失可以直接显示冲突

用一个标量参数 $w$ 构造两个任务：

$$
\mathcal L_{\mathrm{old}}(w)
=
\frac12(w-1)^2,
\qquad
\mathcal L_{\mathrm{new}}(w)
=
\frac12(w+1)^2.
$$

旧任务的最优点为 $w=1$，新任务的最优点为 $w=-1$。在 $w=0$ 处：

$$
g_{\mathrm{old}}=-1,
\qquad
g_{\mathrm{new}}=1,
\qquad
\cos(g_{\mathrm{old}},g_{\mathrm{new}})=-1.
$$

新任务更新会沿旧任务相反方向移动。模型参数很多时，这种冲突发生在部分层、部分 token 或部分行为子空间中，最终表现为能力单元的局部回归。

### 可塑性和保持需要分配同一更新

学习率大、训练 token 多、更新层多时，新任务拥有更高的参数移动自由度；冻结层、低 rank adapter 或较强保持正则会限制移动。两类约束都影响新任务的适配速度和最终质量：

|更新选择|旧能力风险|新任务自由度|需要测量|
|---|---|---|---|
|全量更新|较高|较高|old regression 和 update norm|
|冻结底层|中等|中等|分层梯度和新任务分数|
|LoRA/adapter|受 target module 影响|受 rank 影响|base hash、adapter 回归|
|prompt tuning|基础权重不改|受 prompt length 影响|prompt 路由和旧能力保持|
|小学习率/少 token|通常较低|可能不足|有效 token 和新任务 gain|

“使用 PEFT”只能说明基础权重更新路径受限；旧任务保持仍需要直接评测。

## 一个冲突损失的可运行探针

下面的 Python 标准库探针比较四步新任务更新、不同 rehearsal 权重和一个带旧参数惩罚的 EWC 风格二次损失。它用于核对差值和方向，不代表真实神经网络的完整训练动态。

```python
loss_old = lambda w: 0.5 * (w - 1.0) ** 2
loss_new = lambda w: 0.5 * (w + 1.0) ** 2
grad_new = lambda w: w + 1.0

w0 = 1.0
learning_rate = 0.25
w = w0
for _ in range(4):
    w -= learning_rate * grad_new(w)

print('base_w=', f'{w0:.6f}')
print('base_old_loss=', f'{loss_old(w0):.6f}')
print('base_new_loss=', f'{loss_new(w0):.6f}')
print('after_new_only_w=', f'{w:.6f}')
print('after_new_only_old_loss=', f'{loss_old(w):.6f}')
print('after_new_only_new_loss=', f'{loss_new(w):.6f}')
print('old_loss_increase=', f'{loss_old(w)-loss_old(w0):.6f}')

probe_w = 0.0
grad_old_probe = probe_w - 1.0
grad_new_probe = probe_w + 1.0
cosine = grad_old_probe * grad_new_probe / (abs(grad_old_probe) * abs(grad_new_probe))
print('gradient_cosine_at_probe=', f'{cosine:.6f}')

for old_weight in [0.25, 0.50, 0.75]:
    mixed_w = 2 * old_weight - 1
    print('mix_lambda=', f'{old_weight:.2f}',
          'w=', f'{mixed_w:.6f}',
          'old_loss=', f'{loss_old(mixed_w):.6f}',
          'new_loss=', f'{loss_new(mixed_w):.6f}')

fisher = 4.0
ewc_weight = 1.0
ewc_w = (-1.0 + ewc_weight * fisher * w0) / (1.0 + ewc_weight * fisher)
print('ewc_w=', f'{ewc_w:.6f}')
print('ewc_old_loss=', f'{loss_old(ewc_w):.6f}')
print('ewc_new_loss=', f'{loss_new(ewc_w):.6f}')
```

运行输出：

```text
base_w= 1.000000
base_old_loss= 0.000000
base_new_loss= 2.000000
after_new_only_w= -0.367188
after_new_only_old_loss= 0.934601
after_new_only_new_loss= 0.200226
old_loss_increase= 0.934601
gradient_cosine_at_probe= -1.000000
mix_lambda= 0.25 w= -0.500000 old_loss= 1.125000 new_loss= 0.125000
mix_lambda= 0.50 w= 0.000000 old_loss= 0.500000 new_loss= 0.500000
mix_lambda= 0.75 w= 0.500000 old_loss= 0.125000 new_loss= 1.125000
ewc_w= 0.600000
ewc_old_loss= 0.080000
ewc_new_loss= 1.280000
```

四步只训练新任务后，$w$ 从 1 移到 -0.367188，旧任务损失从 0 增加到 0.934601，新任务损失从 2 降到 0.200226。冲突点的梯度余弦为 -1。rehearsal 权重 $\lambda=0.75$ 时，旧/新损失为 0.125/1.125；$\lambda=0.25$ 时为 1.125/0.125。EWC 风格的旧参数惩罚把 $w$ 保持在 0.6，旧/新损失为 0.08/1.28。

这个例子把保持和适配的折中显式化。真实模型中的损失不是一个标量二次函数，旧能力可能分布在多个层和行为路径中，所以探针只能验证公式与读数方向。

## Rehearsal 把旧任务梯度放回更新

### 混合数据要按有效 token 统计

最直接的保持方法是在新任务训练中加入旧任务样本：

$$
\mathcal L_{\mathrm{mix}}
=
\lambda\mathcal L_{\mathrm{old}}
+
(1-\lambda)\mathcal L_{\mathrm{new}}.
$$

这里的 $\lambda$ 应按有效监督 token、样本或 batch 定义，并写明采样器是否重复旧样本。只说“加入 10% 旧数据”无法判断 token 比例、长度分布和 loss 分母。

|混合字段|需要保存|
|---|---|
|old sample share|旧样本在采样记录中的份额|
|old token share|旧任务有效 target token 份额|
|group split|旧/新数据按来源、任务和时间的切分|
|replay policy|全量 replay、固定 buffer 或分层采样|
|dedup|旧数据与新数据的重复率|
|privacy/license|旧样本是否允许重新训练|

### replay buffer 也会改变分布

固定大小的 replay buffer 需要定义保留策略。按随机样本保留可能丢失少数任务、长尾语言和安全边界；按任务均衡保留可能改变真实旧分布。应记录 buffer 构建时间、版本、容量、任务份额和去重方法。

如果旧数据不能保存，可以保存旧任务的统计、样本摘要、teacher logits 或可验证的合成回放数据；这些替代物的保持效果需要单独评测，不能等同于原始 replay。

## 正则化把参数拉回基础 checkpoint

### L2-SP 约束移动距离

一种简单保持项是惩罚参数偏离旧 checkpoint：

$$
\mathcal L_{\mathrm{sp}}
=
\mathcal L_{\mathrm{new}}
+
\frac{\lambda}{2}
\lVert\theta-\theta_0\rVert_2^2.
$$

它把所有参数使用同一个距离权重。不同层对旧能力的重要性可能不同，统一 $\lambda$ 可能过度限制某些适配路径，或者保护了对旧任务无关的参数。

### EWC 按重要性加权

EWC 风格的约束写成：

$$
\mathcal L_{\mathrm{ewc}}
=
\mathcal L_{\mathrm{new}}
+
\frac{\lambda}{2}
\sum_i F_i(\theta_i-\theta_{0,i})^2.
$$

$F_i$ 可以由旧任务 Fisher 信息近似得到。$F_i$ 大的参数移动代价更高；计算 Fisher 的数据、归约、裁剪和归一化必须记录。Fisher 估计错误时，正则会把更新预算分配到错误的位置。

### 正则系数改变新任务的可塑性

较强的 $\lambda$ 通常提高旧任务保持，可能降低新任务 gain；较弱的 $\lambda$ 提供更高自由度，可能增加旧任务回归。应通过固定协议画出 retention 与 new gain 的关系，不用单一 checkpoint 的结果选择系数。

## 蒸馏保留旧模型的行为

### logits 蒸馏提供软目标

冻结旧模型 $\theta_0$，对同一输入得到旧分布 $p_0$，新模型得到 $p_1$。可以加入：

$$
\mathcal L_{\mathrm{distill}}
=
\tau^2
\operatorname{KL}
\left(
p_0^{(\tau)}
\middle\|
p_1^{(\tau)}
\right).
$$

温度 $\tau$、蒸馏位置、旧模型输入和 token mask 影响保持信号。生成模型中可以对旧任务样本、旧模型生成 token 或指定行为位置做蒸馏。

### 蒸馏 target 也有覆盖范围

若只在新任务输入上蒸馏，旧任务没有被直接采样；若在旧任务输入上蒸馏，则成本接近 replay。若只保留 top-k logits，尾部概率和拒答边界可能未被约束。应报告蒸馏输入、温度、logit 截断、mask 和 KL 归约。

|蒸馏对象|保持信号|缺口|
|---|---|---|
|旧任务 logits|token 级行为|需要旧任务输入|
|旧模型生成|序列级行为|受 sampling 影响|
|隐藏状态|表征相似性|不等于输出行为|
|attention map|读取路径相似性|实现和层选择敏感|
|安全分类器分数|边界行为|覆盖任务有限|

## adapter 把任务更新与基础权重分开

### 独立 adapter 可以降低覆盖风险

全量微调直接修改 $\theta_0$。LoRA、prompt tuning 或其他 PEFT 方法可以把新任务变化保存为独立的 $\phi$：

$$
f_{\theta_0,\phi_{\mathrm{new}}}(x)
\quad\text{and}\quad
f_{\theta_0,\phi_{\mathrm{old}}}(x).
$$

切换旧任务时加载旧 adapter，基础权重保持不变。这个结构把任务间参数覆盖转化为 adapter 路由和存储问题，但单个 adapter 的能力仍可能挤压旧行为，多个 adapter 的 merge、cache 和 batch 也需要隔离。

### adapter 不能替代旧任务回归

如果部署只保留新 adapter，旧任务请求可能被路由到错误的 adapter；如果把多个 adapter 合并，增量可能发生干扰。应按 task ID、base hash、adapter hash 和 merged state 保存请求证据。

|方案|基础权重|任务隔离|主要代价|
|---|---|---|---|
|full fine-tuning|被修改|低|显存、回归和回滚成本高|
|LoRA adapter|冻结|高|target、rank、merge 和路由|
|soft prompt|冻结|高|位置、长度、cache 和模板|
|rehearsal full|被修改|中|旧数据、训练成本和隐私|
|distillation full|被修改|中|teacher 推理和蒸馏合同|

## 评测矩阵要包含保持和适配

### 最小四格结果

每个训练方案至少报告：

|checkpoint|旧任务|新任务|用途|
|---|---:|---:|---|
|base $\theta_0$|记录|记录|建立起点|
|new-only $\theta_1$|记录|记录|测遗忘与适配|
|rehearsal/regularized|记录|记录|比较保持策略|
|adapter|记录|记录|比较隔离方案|

如果只报告新任务 loss，不能知道新任务 gain 是否以旧任务回归换取。如果只报告旧任务 retention，不能知道模型是否拒绝了新任务更新。

### 评测协议本身也会造成假遗忘

下列变化会制造分数差异：

1. chat template、system prompt 或 stop rule 改变。
2. tokenizer、词表或 label mapping 改变。
3. decode temperature、top-p、max tokens 或 tool policy 改变。
4. 旧任务测试集发生去重、过滤或重采样。
5. scorer 版本、格式容忍度或答案规范化改变。

保存输入、输出、score、版本和失败样本。对结构化任务同时检查解析成功率与字段级正确率。

## 运行方法

将上一个 Python 代码块保存为 catastrophic-forgetting-probe.py，再运行 python3 catastrophic-forgetting-probe.py。修改 learning rate、更新步数、rehearsal weight 或 Fisher weight 后，应同步比较旧/新损失。

接入真实模型时，固定 base checkpoint，先建立旧/新测试集，再运行 new-only、rehearsal、regularized、adapter 四个对照。保存每个 checkpoint 的 old/new 分数、有效 token、参数更新范数、梯度余弦、数据份额和推理合同。

## 失效模式与审计清单

### 常见失效模式

|现象|优先检查|确认方法|
|---|---|---|
|新任务分数上升、旧任务下降|梯度冲突、学习率、token 预算|测 old/new gradient cosine 和回归矩阵|
|旧任务分数下降但 loss 不变|accuracy threshold、decode 或 scorer|保存 logits、输出和错误样本|
|rehearsal 没有保持效果|旧 token share、buffer 覆盖和 mask|统计 task/language/length 份额|
|正则后新任务不收敛|lambda、Fisher、冻结层或 LR|记录参数移动和 new gain|
|蒸馏后格式变差|蒸馏 mask、temperature、teacher 输出|比较 token/字段级回归|
|adapter 切换后旧任务失败|路由、base hash、cache 或 merge|按请求记录 adapter 和 cache|
|只测总分看不到回归|分组切片缺失|按 task、domain、format、length 重算|
|旧任务提升但新任务没有适配|replay 或正则过强|画 retention/new gain 曲线|
|恢复 checkpoint 后结果不同|optimizer、RNG、数据顺序或模板|运行同一输入的 deterministic probe|
|遗忘只在长输出出现|长度、截断、EOS 或 stop rule|按输出长度分层评测|

### 最小审计表

|审计层|应保存|验收问题|
|---|---|---|
|base|内容 hash、tokenizer、template、旧任务版本|遗忘差值的起点固定吗|
|data|旧/新样本、token 份额、buffer、去重|训练分布是否可重建|
|update|LR、步数、梯度、参数移动、cosine|更新方向和规模是什么|
|retention|old loss、accuracy、behavior slices|旧能力下降在哪里|
|adaptation|new loss、accuracy、格式和安全|新任务是否真的获得能力|
|mitigation|replay、lambda、Fisher、teacher、adapter|保持策略的输入和强度是什么|
|checkpoint|base、训练状态、adapter、merge、RNG|能否恢复同一状态|
|deployment|路由、cache、decode、scorer|线上行为是否匹配评测|

灾难性遗忘的结论需要同时说明旧任务变化、新任务变化和产生差值的训练合同。rehearsal、正则、蒸馏和 adapter 只改变保持—适配的分配方式；每种方案都需要在同一回归矩阵中比较。

## 相关词条

- [监督微调](../finetuning/sft/)：说明 next-token 更新、mask 和有效 token。
- [指令数据](../finetuning/instruction-data/)：检查新任务示范的质量、覆盖和切分。
- [全量微调与参数高效微调](../finetuning/full-vs-peft/)：比较参数空间、资源账和 checkpoint。
- [LoRA](../finetuning/lora/)：用低秩 adapter 限制任务更新路径。
- [Prompt Tuning](../finetuning/prompt-tuning/)：把任务条件保存为独立 soft prompt。
- [知识蒸馏](../finetuning/distillation/)：保留旧模型的 logits 或行为分布。
- [训练稳定性](../pretraining/training-stability/)：检查学习率、梯度、非有限值和恢复。
- [训练、验证与测试集](../learning-framework/train-validation-test/)：固定切分和评测边界。
