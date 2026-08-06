---
title: "分类评估指标：从混淆矩阵到阈值与排序"
tags: ["why-models-learn"]
---

分类评估指标不是一张固定的排行榜，而是把标签、预测分数和决策代价压缩成可比较的统计量。准确率回答“总体有多少判对”，精确率和召回率回答“正类预测错在哪里”，ROC/PR 曲线回答“改变阈值时排序能否保持”，校准和对数损失则检查“概率本身是否可信”。因此，先明确部署时要做什么决定，再选择与这个决定相称的指标。

![分类评估从混淆矩阵出发，经过阈值扫描连接 ROC 与 PR 曲线](/assets/evaluation-and-generalization/svg/classification-metrics.1.svg)

## 指标回答的是哪个决策问题

设第 $i$ 个样本的真实标签为 $y_i$，模型输出一个分数或概率 $s_i$，阈值和动作规则把它变成硬预测 $\widehat y_i$。一个评估指标可以写成

$$
\widehat m
=M\left(
\left\{
(y_i,s_i,\widehat y_i)
\right\}_{i=1}^{n}
\right).
$$

不同指标读取这组记录的不同部分：

| 部署问题 | 更直接的证据 | 不能单独回答什么 |
| --- | --- | --- |
| 所有样本总体是否判对 | accuracy、error rate | 少数类是否被忽略 |
| 报警中有多少是真的 | precision、PPV | 漏掉多少正类 |
| 正类中找回多少 | recall、TPR、sensitivity | 报警会不会太多 |
| 负类中排除了多少 | specificity、TNR | 正类的检出能力 |
| 改变阈值后排序是否稳定 | ROC-AUC、PR-AUC、average precision | 固定业务阈值的实际代价 |
| 输出的概率能否用于决策 | calibration、Brier score、log loss | 只看一个阈值的分类结果 |

这也区分了三种对象：训练损失通常给优化器连续的逐样本信号，评估指标用来比较方案，最终决策还要把预测映射为行动和代价。一个模型可以有更低的 log loss，却在某个固定阈值下有更低的召回率；这不是矛盾，而是读取了不同证据。

## 混淆矩阵是二分类指标的账本

先固定一个正类。对每个样本，把真实标签和硬预测放进四个格子：

| 真实 $\backslash$ 预测 | 正类 | 负类 |
| --- | ---: | ---: |
| 正类 | TP：真阳性 | FN：假阴性 |
| 负类 | FP：假阳性 | TN：真阴性 |

正类和负类的真实数量分别是

$$
P=TP+FN,
\qquad
N=TN+FP,
\qquad
n=P+N.
$$

假设一次测试集评估得到

| 真实 $\backslash$ 预测 | 预测正类 | 预测负类 | 合计 |
| --- | ---: | ---: | ---: |
| 真实正类 | $18$ | $2$ | $20$ |
| 真实负类 | $6$ | $74$ | $80$ |
| 合计 | $24$ | $76$ | $100$ |

这里 $TP=18$、$FN=2$、$FP=6$、$TN=74$。先保存这四个计数，再从它们派生指标，比只保存一个百分比更可审计：换一个正类定义、换一个阈值或换一个代价权重时，仍可以重新计算。

## 准确率、召回率和精确率看不同切面

准确率把两个对角格子放在同一个分子：

$$
\operatorname{accuracy}
=\frac{TP+TN}{n}
=\frac{18+74}{100}
=0.92.
$$

错误率是其补数：

$$
\operatorname{error\ rate}
=\frac{FP+FN}{n}
=0.08.
$$

召回率只在真实正类中计数：

$$
\operatorname{recall}
=\operatorname{TPR}
=\frac{TP}{TP+FN}
=\frac{18}{20}
=0.9.
$$

精确率只在预测正类中计数：

$$
\operatorname{precision}
=\operatorname{PPV}
=\frac{TP}{TP+FP}
=\frac{18}{24}
=0.75.
$$

负类也有对称的两个指标：

$$
\operatorname{specificity}
=\operatorname{TNR}
=\frac{TN}{TN+FP}
=\frac{74}{80}
=0.925,
$$

$$
\operatorname{NPV}
=\frac{TN}{TN+FN}
=\frac{74}{76}
\approx0.973684.
$$

因此，这个模型的 $92\%$ accuracy 不能替代 $90\%$ recall 和 $75\%$ precision。若正类是需要人工处理的报警，precision 直接影响人工队列里有多少无效工单；若正类是必须尽量找出的风险，recall 和 FN 的代价更重要。

精确率不是只由模型排序决定的，它还随正类比例变化。令总体正类率为

$$
\pi=\frac{P}{P+N},
$$

并令模型的 TPR 和 FPR 在分布变化前后保持不变，则

$$
\operatorname{precision}
=\frac{\operatorname{TPR}\pi}
{\operatorname{TPR}\pi+\operatorname{FPR}(1-\pi)}.
$$

例如固定 $\operatorname{TPR}=0.9$、$\operatorname{FPR}=0.05$ 时，若 $\pi=0.5$，precision 为

$$
\frac{0.9\times0.5}{0.9\times0.5+0.05\times0.5}
\approx0.947368;
$$

若部署中的正类率降为 $\pi=0.01$，同一个 TPR/FPR 对应的 precision 变成

$$
\frac{0.9\times0.01}{0.9\times0.01+0.05\times0.99}
\approx0.153846.
$$

这就是为什么离线测试集上的 precision 不能脱离部署基率解释。ROC 中的 TPR/FPR 可以在条件分布不变时较稳定，precision 却会因先验比例变化而明显改变。

## F1、平衡准确率和 MCC

precision 和 recall 的算术平均可能掩盖其中一项很小，因此常用调和平均 F1：

$$
F_1
=\frac{2\operatorname{precision}\operatorname{recall}}
{\operatorname{precision}+\operatorname{recall}}.
$$

在上面的混淆矩阵中，

$$
F_1
=\frac{2\times0.75\times0.9}{0.75+0.9}
\approx0.818182.
$$

F1 只使用 TP、FP 和 FN，不包含 TN。它适合“正类检出和报警纯度都重要”的场景，却不能表达把负类误报出去的额外业务代价，也不能判断概率是否校准。若需要偏向 recall 或 precision，可以用带权的 $F_\beta$：

$$
F_\beta
=(1+\beta^2)
\frac{\operatorname{precision}\operatorname{recall}}
{\beta^2\operatorname{precision}+\operatorname{recall}},
$$

其中 $\beta>1$ 更重视 recall，$0<\beta<1$ 更重视 precision。

类别不平衡时，可以把正类和负类的召回率先分别计算，再取平均：

$$
\operatorname{balanced\ accuracy}
=\frac{\operatorname{TPR}+\operatorname{TNR}}{2}
=\frac{0.9+0.925}{2}
=0.9125.
$$

它避免大量 TN 把少数类表现隐藏起来，但仍然是固定阈值的硬预测指标。MCC 则把四个格子同时纳入：

$$
\operatorname{MCC}
=\frac{TP\cdot TN-FP\cdot FN}
{\sqrt{
(TP+FP)(TP+FN)(TN+FP)(TN+FN)
}}.
$$

代入本例：

$$
\operatorname{MCC}
=\frac{18\times74-6\times2}
{\sqrt{24\times20\times80\times76}}
\approx0.772683.
$$

当四个格子都非退化时，MCC 的范围是 $[-1,1]$：$1$ 表示完全一致，$0$ 表示没有线性关联，$-1$ 表示完全相反。它对类别比例更敏感地使用 TN，但在某一行或某一列全为零时分母会退化，报告时要说明处理方式。

## 正类定义和代价决定阈值

模型输出概率并不会自动决定行动。给定阈值 $t$，硬预测通常写成

$$
\widehat y_i(t)
=\mathbb 1\{s_i\ge t\}.
$$

提高阈值会让预测正类变少，通常降低 FP 和 recall；降低阈值会扩大报警范围，通常提高 recall 但增加 FP。这个趋势不是每一份有限数据上都严格单调，因为分数可能有并列，实际实现还要规定 $s_i=t$ 时归入哪一侧。

如果一次 FN 的代价为 $c_{\mathrm{FN}}$，一次 FP 的代价为 $c_{\mathrm{FP}}$，可以直接在测试集上记录加权错误：

$$
\widehat C(t)
=c_{\mathrm{FN}}\,FN(t)
+c_{\mathrm{FP}}\,FP(t).
$$

在上面的例子中，若 $c_{\mathrm{FN}}=10$、$c_{\mathrm{FP}}=1$，代价为

$$
\widehat C=10\times2+1\times6=26.
$$

若模型输出的是经过校准的正类概率 $p(x)$，还可以对每个动作比较条件期望代价。设行动 $a_+$ 会把样本送入正类处理，行动 $a_-$ 则不处理，已知四种动作—结果代价，则应选择条件期望代价较小的行动：

$$
\mathbb E[C(a_+)\mid x]
=p(x)C(a_+,1)+(1-p(x))C(a_+,0),
$$

$$
\mathbb E[C(a_-)\mid x]
=p(x)C(a_-,1)+(1-p(x))C(a_-,0).
$$

所以阈值不应默认取 $0.5$。它由正类定义、错误代价、资源容量、人工队列和概率校准共同决定；在验证集上选择阈值，在测试集上只报告冻结后的结果。

## 阈值扫描把一个模型变成一串混淆矩阵

固定一组排序分数，阈值扫描就能展示模型在不同工作点的取舍。下面有 4 个正类和 6 个负类，分数从高到低排列：

| 排名 | 分数 $s_i$ | 真实标签 |
| ---: | ---: | --- |
| 1 | $0.95$ | 正类 |
| 2 | $0.90$ | 负类 |
| 3 | $0.85$ | 正类 |
| 4 | $0.80$ | 正类 |
| 5 | $0.70$ | 负类 |
| 6 | $0.60$ | 负类 |
| 7 | $0.55$ | 正类 |
| 8 | $0.50$ | 负类 |
| 9 | $0.40$ | 负类 |
| 10 | $0.10$ | 负类 |

用不同阈值截取列表前缀，得到：

| 阈值规则 | TP | FP | FN | TN | TPR | FPR | precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $s\ge0.95$ | 1 | 0 | 3 | 6 | 0.25 | 0 | 1 |
| $s\ge0.90$ | 1 | 1 | 3 | 5 | 0.25 | 0.1667 | 0.5 |
| $s\ge0.80$ | 3 | 1 | 1 | 5 | 0.75 | 0.1667 | 0.75 |
| $s\ge0.55$ | 4 | 3 | 0 | 3 | 1 | 0.5 | 0.5714 |
| $s\ge0.10$ | 4 | 6 | 0 | 0 | 1 | 1 | 0.4 |

表中的每一行都是一个可能的部署点。若人工团队每天只能处理少量报警，$s\ge0.80$ 可能比 $s\ge0.55$ 更合适；若漏掉一个正类的损失远高于多一次人工复核，则更低阈值可能值得承担。

## ROC 与 PR 曲线分别保留什么

ROC 曲线把每个阈值的点画成

$$
(\operatorname{FPR},\operatorname{TPR})
=\left(
\frac{FP}{FP+TN},
\frac{TP}{TP+FN}
\right).
$$

PR 曲线则画

$$
(\operatorname{recall},\operatorname{precision})
=\left(
\frac{TP}{TP+FN},
\frac{TP}{TP+FP}
\right).
$$

ROC 更强调正负类条件分布的排序分离，PR 更直接暴露“找回的正类里有多少是真的”。当正类很稀少时，大量负类会让 FPR 的绝对变化看起来很小，而 PR 曲线的 precision 会迅速反映报警纯度下降。

对上面的 10 个分数，4 个正类与 6 个负类一共有 $4\times6=24$ 对正负样本。按分数排序，正类分别战胜 6、5、5、3 个负类，因此 ROC-AUC 为

$$
\operatorname{ROC\text{-}AUC}
=\frac{6+5+5+3}{4\times6}
=\frac{19}{24}
\approx0.791667.
$$

这等价于随机抽一个正类和一个负类时，正类分数更高的概率；若出现并列，通常给并列半个胜场。它是排序指标，不等于某个阈值下的 accuracy，也不等于概率已经校准。

平均精度 average precision 可以只在每个正类被检索到的位置累加 precision。这个例子的正类排名为 $1,3,4,7$，所以

$$
\operatorname{AP}
=\frac14
\left(
1+\frac23+\frac34+\frac47
\right)
\approx0.747024.
$$

PR 曲线的随机基线等于正类率：

$$
\operatorname{precision}_{\mathrm{chance}}
=\pi=\frac4{10}=0.4.
$$

报告 ROC-AUC 时同时给出 PR-AUC 或 AP、正类率和关键阈值，通常比只报一个 AUC 更接近实际使用。曲线面积还依赖插值约定、分数并列处理和库的实现，复现实验时要固定这些细节。

## 多分类要先逐类再聚合

多分类混淆矩阵的行表示真实类别，列表示预测类别。以三类 $A,B,C$ 为例：

| 真实 $\backslash$ 预测 | $A$ | $B$ | $C$ |
| --- | ---: | ---: | ---: |
| $A$ | 40 | 5 | 5 |
| $B$ | 4 | 45 | 1 |
| $C$ | 6 | 2 | 42 |

对类别 $k$ 做 one-vs-rest 视角：对角线元素是 $TP_k$，第 $k$ 行其余部分是 $FN_k$，第 $k$ 列其余部分是 $FP_k$，矩阵之外的部分是 $TN_k$。于是每个类别都可以计算自己的 precision、recall 和 F1。

如果先对每个类别算指标，再平均，得到 macro 平均：

$$
\operatorname{macro}(m)
=\frac1K\sum_{k=1}^{K}m_k.
$$

它让小类别与大类别权重相同，适合关心每个类别都不能失守的任务。若按每类真实样本数 $n_k$ 加权：

$$
\operatorname{weighted}(m)
=\sum_{k=1}^{K}\frac{n_k}{n}m_k.
$$

micro 平均则先把所有类别的 TP、FP、FN 汇总，再计算一个总体指标：

$$
\operatorname{micro\ precision}
=\frac{\sum_kTP_k}{\sum_k(TP_k+FP_k)},
\qquad
\operatorname{micro\ recall}
=\frac{\sum_kTP_k}{\sum_k(TP_k+FN_k)}.
$$

单标签多分类中，micro precision、micro recall 和 accuracy 相等，因为每个样本恰好贡献一次预测。三类例子的正确数为 $40+45+42=127$，所以

$$
\operatorname{accuracy}
=\frac{127}{150}
\approx0.846667.
$$

同一结果同时给 macro 和 weighted 指标时，要把聚合规则写在指标名里。top-k accuracy 则允许真实类别出现在概率最高的前 $k$ 个类别中，适合类别很多且“候选集合命中”本身就是有用动作的场景；它不能替代单一类别决策的 precision 或成本分析。

## 概率预测还要检查校准

硬预测只保留了阈值的一侧信息。若模型输出 $p_i$ 作为正类概率，校准要求相近概率的样本在长期频率上也接近这个概率：

$$
\Pr(Y=1\mid p(X)\approx q)\approx q.
$$

例如一批预测概率约为 $0.8$ 的样本，若其中只有一半为正类，模型就过度自信；若其中几乎全为正类，模型则可能过于保守。两组模型可以拥有完全相同的混淆矩阵，却给出不同的概率质量。

二元 Brier score 是概率与标签的均方误差：

$$
\operatorname{Brier}
=\frac1n\sum_{i=1}^{n}(p_i-y_i)^2.
$$

对数损失会更强地惩罚把真实结果赋予极小概率：

$$
\operatorname{log\ loss}
=-\frac1n\sum_{i=1}^{n}
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right].
$$

可以把概率分箱，画每个箱的平均预测概率和实际正类频率；也可以报告 reliability diagram、Brier score 和 log loss。校准器本身需要在验证集上拟合，不能用测试标签反复调整后再声称测试结果独立。

## 评估要先冻结数据边界和选择规则

分类指标并不会修复数据泄漏。一个可解释的评估协议至少要固定：

| 选择项 | 应在何处决定 | 测试阶段如何处理 |
| --- | --- | --- |
| 特征与预处理参数 | 训练集 | 只应用已冻结的变换 |
| 模型超参数 | 验证集或交叉验证 | 不再根据测试结果修改 |
| 正类定义与标签版本 | 评估协议 | 测试前固定 |
| 决策阈值与代价权重 | 验证集和业务约束 | 测试时直接使用 |
| 指标集合与主指标 | 评估前 | 同时报告预先约定的辅助指标 |
| 分组、时间和去重规则 | 切分前 | 测试分片保持部署边界 |

训练、验证和测试的角色必须和数据切分方式一起描述。若同一用户、设备、文档或时间窗口的记录跨越分片，accuracy、AUC 和 calibration 都可能被高估。若正类定义在测试期间改变，应重新冻结标签版本并重新解释历史指标。

## 不确定性与模型比较

一个测试集上的指标是随机估计量，不是永恒常数。若把 accuracy 近似成独立 Bernoulli 结果，在 $\widehat a=0.92$、$n=100$ 时，标准误约为

$$
\operatorname{SE}(\widehat a)
\approx
\sqrt{\frac{\widehat a(1-\widehat a)}{n}}
=\sqrt{\frac{0.92\times0.08}{100}}
\approx0.027129.
$$

这个近似在小样本、分组相关、时间相关或类别极不平衡时可能不可靠。更通用的办法是对测试样本按部署单位做 bootstrap：每次重采样后重新计算整组指标，取分位数区间。若用户是独立单位，就按用户而不是按单条记录重采样；若指标关注少数类，还要报告每次重采样是否包含足够的正类。

比较两个模型时，应尽量在同一批测试样本上成对计算。这样可以减少“两个测试集抽样不同”带来的噪声；对硬分类器可以记录两个模型预测不一致的样本并使用配对检验，对概率模型则可以比较逐样本 log loss 或 Brier 差值的 bootstrap 区间。仅比较两个独立百分比的小数点后几位，通常没有足够证据说明模型真的不同。

分层报告也很重要。总体 recall 可能掩盖某个用户群、地区、时间段或设备类型的显著下降；应在不泄漏测试标签的前提下，按部署中有意义的分片报告支持数、混淆矩阵和区间。分片太小的时候，正确结论可能是“不确定”，而不是强行排名。

## 一个可复用的分类评估流程

1. 写清楚正类、负类、可用分数、最终动作和 FP/FN 的相对代价；
2. 根据部署抽样过程切分训练、验证和测试，按用户、设备、时间或空间边界去重；
3. 在训练边界内拟合预处理和模型，验证集只用于选择模型、校准器与阈值；
4. 在验证集上画阈值扫描，保存混淆矩阵、ROC/PR 曲线和候选工作点；
5. 预先指定主指标与辅助指标，同时记录正类率、支持数和指标定义；
6. 冻结模型、预处理、标签版本、阈值和聚合方式；
7. 在测试集上一次性计算混淆矩阵、主指标、概率损失、校准与关键分片；
8. 用 bootstrap、配对比较或合适的精确区间表达不确定性；
9. 把测试结果与部署基率、资源容量和实际错误代价一起解释；
10. 保存分数方向、阈值包含规则、并列处理、库版本和切分指纹，保证下一次评估可复现。

## 失效模式

**只报告 accuracy。** 类别不平衡时，大量 TN 可以让模型看似优秀；至少同时报告正类率、recall、precision 或 PR 指标。

**把 precision 和 recall 当作模型固有属性。** 它们依赖阈值、正类定义和部署基率；比较前要固定这些条件。

**只追求 F1。** F1 不看 TN、不表达错误代价，也不评价概率质量；需要把 balanced accuracy、MCC、成本和校准补上。

**只看 ROC-AUC。** AUC 是排序证据，不告诉你哪个阈值可部署；少数类报警还要看 PR 曲线和关键工作点。

**在测试集上挑阈值。** 这会把测试标签变成选择信号；阈值应在验证边界内决定，测试只做冻结后的估计。

**混淆 macro、micro 和 weighted。** 三种聚合回答不同问题，不能只写一个“平均 F1”而不说明权重。

**把概率当成天然可信。** sigmoid 或 softmax 输出落在 $[0,1]$ 里，不等于它已经校准；要用可靠性图、Brier 或 log loss 检查。

**忽略切分相关性和不确定性。** 同一实体跨分片、重复样本、时间泄漏和过小的少数类都会让指标过于乐观；报告支持数、切分规则和区间。

**把总体指标当成部署全貌。** 总体平均可能掩盖分片差异；应结合任务风险报告关键人群、时间段和资源约束。

## 相关词条

- [监督学习](../learning-framework/supervised-learning/)：理解分类任务、标签和条件决策的形式化。
- [训练、验证与测试集](../learning-framework/train-validation-test/)：规定模型、阈值和评估指标的选择边界。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：理解反复选择指标时产生的验证过拟合。
- [逻辑回归](../linear-models/logistic-regression/)：复习 sigmoid 概率、阈值和二分类 log loss。
- [Softmax 回归](../linear-models/softmax-regression/)：理解多分类概率、top-k 决策和交叉熵。
- [交叉熵](../information-theory/cross-entropy/)：理解 log loss 如何惩罚错误的概率质量。
- [间隔与支持向量机](../linear-models/margins-and-svm/)：对照分数、间隔和硬分类评估。
- [高斯混合模型与 EM](../linear-models/gmm-and-em/)：比较软责任度和硬标签评估的区别。
