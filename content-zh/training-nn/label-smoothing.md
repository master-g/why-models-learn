---
title: "标签平滑：把分类目标从尖点改成可控分布"
tags: ["why-models-learn"]
---

标签平滑（label smoothing）把互斥分类中的 one-hot 目标与一个参考分布混合，让正确类别仍然占最大质量、其他类别也保留少量质量。它改变的是交叉熵的目标分布，不是把 logits 做温度缩放，也不是在评估时把概率强行拉向均匀。要把它用对，必须先固定类别数、平滑系数、参考分布、损失归约、忽略标签和评估指标。

![标签平滑示意图：one-hot 目标与均匀分布按 epsilon 混合，交叉熵梯度从 p-y 变为 p-q](/assets/training-nn/svg/label-smoothing.1.svg)

## 先写出 hard target 和 soft target

设有 $K$ 个互斥类别，真实类别为 $c$。one-hot 目标 $\boldsymbol y$ 的第 $k$ 个坐标是

$$
y_k=\mathbf 1[k=c].
$$

它是一个概率分布：每个坐标非负，而且总和为 $1$。模型由 logits $\boldsymbol z$ 产生预测分布

$$
p_k=\frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}.
$$

标签平滑把 $\boldsymbol y$ 换成另一个目标分布 $\boldsymbol q$。最常见的均匀混合写成

$$
\boldsymbol q
=(1-\varepsilon)\boldsymbol y+\varepsilon\boldsymbol u,
\qquad
u_k=\frac{1}{K},
\qquad
0\leq\varepsilon\leq1.
$$

$\varepsilon$ 越大，目标越接近均匀分布；$\varepsilon=0$ 就是普通 hard-label 交叉熵。这个定义要求 $\boldsymbol q$ 仍然是概率分布：

$$
\sum_{k=1}^{K}q_k
=(1-\varepsilon)\sum_{k=1}^{K}y_k
+\varepsilon\sum_{k=1}^{K}u_k
=1.
$$

因此标签平滑不是把目标向量随意加上常数。混合之后仍要检查非负性、总和和正确类别的位置。

## 两种常见的平滑约定不是同一个 epsilon

实现中至少有两种写法。第一种把 $\varepsilon$ 的质量均匀分给包括正确类别在内的全部 $K$ 个类别：

$$
q_k=(1-\varepsilon)\mathbf 1[k=c]+\frac{\varepsilon}{K}.
$$

于是

$$
q_c=1-\varepsilon+\frac{\varepsilon}{K},
\qquad
q_{k\ne c}=\frac{\varepsilon}{K}.
$$

第二种只把质量分给错误类别：

$$
q_c=1-\varepsilon,
\qquad
q_{k\ne c}=\frac{\varepsilon}{K-1}.
$$

两种目标都合理，但相同的数值 $\varepsilon$ 不代表相同的正确类别质量或错误类别质量。$K=4,\varepsilon=0.2$ 时，第一种目标是 $(0.05,0.85,0.05,0.05)$（假设类别 1 正确），第二种目标是 $(0.066667,0.8,0.066667,0.066667)$。

| 约定 | 正确类质量 | 每个错误类质量 | 审计重点 |
| --- | --- | --- | --- |
| uniform-all | $1-\varepsilon+\varepsilon/K$ | $\varepsilon/K$ | 均匀质量包括正确类 |
| uniform-other | $1-\varepsilon$ | $\varepsilon/(K-1)$ | 只向错误类分配质量 |
| arbitrary prior | $(1-\varepsilon)y_c+\varepsilon r_c$ | $(1-\varepsilon)y_k+\varepsilon r_k$ | 参考分布 $r$ 的来源 |

读取损失接口时，要查清它使用哪一种约定；只把一个名为 smoothing 的参数当成两种实现都相同，会让实验之间的比较失去意义。

## 交叉熵为什么多了一项均匀损失

对目标分布 $\boldsymbol q$ 和模型分布 $\boldsymbol p$，多分类交叉熵是

$$
L(\boldsymbol z;\boldsymbol q)
=-\sum_{k=1}^{K}q_k\log p_k.
$$

代入 uniform-all 的 $\boldsymbol q=(1-\varepsilon)\boldsymbol y+\varepsilon\boldsymbol u$：

$$
\begin{aligned}
L_{\mathrm{smooth}}
&=-(1-\varepsilon)\log p_c
 -\frac{\varepsilon}{K}\sum_{k=1}^{K}\log p_k\\
&=(1-\varepsilon)L_{\mathrm{hard}}
 +\varepsilon L_{\mathrm{uniform}}.
\end{aligned}
$$

其中

$$
L_{\mathrm{hard}}=-\log p_c,
\qquad
L_{\mathrm{uniform}}=-\frac{1}{K}\sum_{k=1}^{K}\log p_k.
$$

所以标签平滑不是只把正确类别的 one-hot 值从 $1$ 改小，然后忽略其它项；每个类别都参与了交叉熵。均匀项会惩罚模型把任何类别的概率压得过低，因此对极端尖锐的分布施加额外压力。

如果参考分布是任意的 $\boldsymbol r$，同样可以写成

$$
L_{\mathrm{smooth}}
=(1-\varepsilon)L(\boldsymbol z;\boldsymbol y)
+\varepsilon L(\boldsymbol z;\boldsymbol r).
$$

这时正则项的含义由 $\boldsymbol r$ 决定，不一定是“趋向均匀”。例如按类别先验构造 $\boldsymbol r$，表达的是向一个先验分布收缩，而不是向所有类别等权收缩。

## logits 梯度从 p-y 变成 p-q

softmax 对 logits 的导数为

$$
\frac{\partial p_k}{\partial z_j}
=p_k(\mathbf 1[k=j]-p_j).
$$

把它代入交叉熵：

$$
\begin{aligned}
\frac{\partial L}{\partial z_j}
&=-\sum_{k=1}^{K}q_k
 \frac{1}{p_k}\frac{\partial p_k}{\partial z_j}\\
&=-\sum_{k=1}^{K}q_k
 (\mathbf 1[k=j]-p_j)\\
&=p_j-q_j.
\end{aligned}
$$

hard label 只是 $\boldsymbol q=\boldsymbol y$ 的特例。平滑之后，正确类别的梯度是

$$
\frac{\partial L}{\partial z_c}
=p_c-\left(1-\varepsilon+\frac{\varepsilon}{K}\right)
$$

（这里采用 uniform-all），错误类别 $k\ne c$ 的梯度是

$$
\frac{\partial L}{\partial z_k}
=p_k-\frac{\varepsilon}{K}.
$$

梯度总和仍然为零：

$$
\sum_{k=1}^{K}\frac{\partial L}{\partial z_k}
=\sum_{k=1}^{K}p_k-\sum_{k=1}^{K}q_k=0.
$$

若模型已经输出 $p_c$ 接近 1，hard label 的正确类梯度接近 0；label smoothing 会留下正的正确类梯度，梯度下降会适度降低过大的正确 logit。错误类如果已经低于它们的目标质量，则梯度为负，梯度下降会把它们的 logits 往上推。这个机制降低了极端置信度，但不会保证最终概率自动校准。

## 一个四分类数值例子

取 logits

$$
\boldsymbol z=(1.2,0.4,-0.3,-0.8),
\qquad
c=1,
\qquad
K=4,
\qquad
\varepsilon=0.2.
$$

稳定 softmax 得到

$$
\boldsymbol p
=(0.553160246,\ 0.248550921,\ 0.123426734,\ 0.074862099).
$$

正确类别是第二个坐标。三种目标和对应损失为：

| 目标 | 向量 | 交叉熵 |
| --- | --- | --- |
| hard | $(0,1,0,0)$ | $1.392107543$ |
| uniform-all | $(0.05,0.85,0.05,0.05)$ | $1.447107543$ |
| uniform-other | $(0.066667,0.8,0.066667,0.066667)$ | $1.465440876$ |

对应的 logits 梯度是

$$
\begin{aligned}
\boldsymbol g_{\mathrm{hard}}
&=(0.553160246,-0.751449079,0.123426734,0.074862099),\\
\boldsymbol g_{\mathrm{all}}
&=(0.503160246,-0.601449079,0.073426734,0.024862099),\\
\boldsymbol g_{\mathrm{other}}
&=(0.486493580,-0.551449079,0.056760068,0.008195432).
\end{aligned}
$$

uniform-all 与 hard 的区别不只在正确类：四个坐标的梯度都改变了。uniform-other 又是另一组梯度，不能只凭“都用了 0.2”判断它们等价。

## 平滑目标决定理想预测和有限的 logit 间隔

交叉熵可以拆成熵和 KL 散度：

$$
L(\boldsymbol z;\boldsymbol q)
=H(\boldsymbol q)+D_{\mathrm{KL}}(\boldsymbol q\Vert\boldsymbol p).
$$

只要模型能表达目标分布，最小值在

$$
\boldsymbol p=\boldsymbol q
$$

处取得，而不是在 $\boldsymbol p=\boldsymbol y$ 处取得。uniform-all 中每个类别的目标质量都为正，因此理想 logits 不需要把错误类别推到负无穷。两个类别的理想 logit 差可以写成概率比的对数：

$$
z_c-z_o
=\log\frac{q_c}{q_o}
=\log\frac{K-\varepsilon(K-1)}{\varepsilon},
\qquad
o\ne c.
$$

uniform-other 的对应差值为

$$
z_c-z_o
=\log\frac{(1-\varepsilon)(K-1)}{\varepsilon}.
$$

在 $K=4,\varepsilon=0.2$ 的例子中，这两个理想间隔分别是 $2.833213344$ 和 $2.484906650$。当 $\varepsilon$ 趋近于零，间隔才会重新趋向无穷大；增大 $\varepsilon$ 相当于给目标分布设置了一个有限的置信度上限。

这是“目标分布的最优解”，不是说训练好的网络一定会达到它。参数共享、特征不可分、权重衰减、优化没有收敛和数据分布错配都会使实际 $\boldsymbol p$ 偏离 $\boldsymbol q$。

## 它可能改善校准，但不能替代校准

硬标签只告诉模型一个类别，训练目标却允许它把正确类概率推得无限接近 1。标签平滑把每个样本的最优目标改成一个有正熵的分布，常会减轻过度自信，尤其是在模型分类边界已经足够好、但概率尖锐度过高的情形。

“常会”不是“必然”。实际校准还会受模型容量、类别不平衡、数据噪声、训练时增强、权重衰减、温度和评估分布影响。至少应在独立验证集上分别报告：

- hard-label accuracy 或 F1，回答类别决策是否正确；
- hard-label NLL，使用真实 one-hot 重新计算概率损失；
- Brier score 或 reliability diagram，检查概率数值；
- ECE 等分桶指标，同时报告分桶方式和样本数；
- 若采用温度缩放，记录验证集温度和是否与平滑训练分开拟合。

训练期间优化的是 $L(\boldsymbol z;\boldsymbol q)$，不能把它直接叫成真实标签 NLL。若训练日志只保留平滑后的 loss，实验之间就无法判断 loss 下降来自更好的分类，还是来自更大的目标平滑系数。

| 方法 | 发生在何处 | 改变的对象 |
| --- | --- | --- |
| label smoothing | 训练目标 | 每个样本的目标分布 $\boldsymbol q$ |
| temperature scaling | 通常在评估后处理 | logits 的相对尺度和预测概率 |
| confidence penalty | 训练损失的熵项 | 对模型分布熵的额外偏好 |
| calibration measurement | 独立评估 | 概率与事件频率的一致程度 |

温度缩放和标签平滑可以同时出现，但一个改变训练目标，一个调整评估概率；调试时要分别记录。

## 不同来源的软目标不能混成一个名字

标签平滑只是软目标的一种来源。若两个样本通过 mixup 混合，目标可能是

$$
\boldsymbol r
=\lambda\boldsymbol y^{(a)}
+(1-\lambda)\boldsymbol y^{(b)}.
$$

知识蒸馏中，教师 logits 经温度 $T$ 得到的分布也可以作为软目标。它携带教师对错误类别的相对偏好，而 uniform-all 的标签平滑对所有错误类别一视同仁。两者的梯度形式都可以写成 $p-q$，但 $q$ 的来源不同，保留的信息也不同。

如果对一个已有软目标 $\boldsymbol r$ 再做均匀平滑，目标是

$$
\boldsymbol q
=(1-\varepsilon)\boldsymbol r+\varepsilon\boldsymbol u.
$$

这不是“再次把 one-hot 变软”，而是把教师、mixup 或人工概率标签进一步向参考分布收缩。代码中若数据管线已经产生软标签，损失函数又自动做了一次 one-hot label smoothing，就会发生双重平滑。

## 类别不平衡、权重和 mask 会改变解释

uniform-all 向每个类别注入相同质量。若类别先验极不均衡，这个参考分布可能与任务先验不一致。可以选择先验分布 $\boldsymbol r$：

$$
\boldsymbol q=(1-\varepsilon)\boldsymbol y+\varepsilon\boldsymbol r,
\qquad
\sum_k r_k=1.
$$

但这时必须把 $\boldsymbol r$ 的估计数据、时间窗口和是否使用验证集写进实验记录。

类别权重也有多个不等价的位置。按真实类取一个权重的损失是

$$
L_{\mathrm{class-weight}}
=-w_c\sum_{k=1}^{K}q_k\log p_k,
$$

而按每个目标坐标加权则是

$$
L_{\mathrm{coordinate-weight}}
=-\sum_{k=1}^{K}w_kq_k\log p_k.
$$

前者仍把整条平滑目标按真实类别缩放，后者会改变目标分布的相对质量；不能只看到一个 class weight 参数就假设它们相同。

对 padding 或 ignore label，先判断该位置是否有效，再构造或参与平滑目标。被忽略的位置不应因为“先平滑、后 mask”而把均匀质量带入损失分母。序列任务还要记录有效 token 数和平均分母，不能只记录 batch size。

## 二分类和多标签任务不能直接套多分类公式

互斥多分类的目标在 $K$ 个坐标上总和为 1，使用 softmax 交叉熵。二分类 BCE 的每个样本只有一个 Bernoulli 概率；一种常见的二元平滑是把标量标签向 $1/2$ 混合：

$$
q=(1-\varepsilon)y+\frac{\varepsilon}{2}.
$$

这和把 $K=2$ 的 uniform-all 公式直接理解成“正确类变成 $1-\varepsilon$”不同。后者的正确类质量其实是 $1-\varepsilon/2$。接口参数的名称相同，目标语义仍可能不同。

多标签任务中每个标签独立，可以对每个 BCE 目标做二元平滑；但多个标签的目标不需要总和为 1，也不应把它们拼起来再做一次多分类 softmax。至少要记录输出头、标签编码、平滑参考值和阈值解释。

## 选择 epsilon 时保持可比较的基线

平滑系数应和模型、数据切分、类别数及其他正则化一起选择。一个可审计的实验矩阵可以包含 $\varepsilon=0$ 的 hard-label 基线，再比较几个小值，例如 $0.05$、$0.1$ 和 $0.2$；这只是搜索起点，不是普适答案。

注意 $K$ 会改变每个错误类获得的质量。uniform-all 固定 $\varepsilon$ 时，每个错误类的质量是 $\varepsilon/K$；类别数增大，单个错误类的质量反而减小，但所有错误类合计的质量仍为 $\varepsilon(K-1)/K$。因此跨数据集比较 $\varepsilon$ 时，不能只比较一个标量。

平滑过强常见的表现是训练 hard NLL 降不下去、决策边界变钝或少数类召回下降；平滑过弱则可能几乎没有改变过度自信。应把硬目标指标、平滑目标 loss、概率校准和类别分层结果放在同一份验证记录里。

## 一个标准库数值实验

下面只使用 Python 标准库，验证稳定 softmax、两种目标构造、三种交叉熵梯度和中心差分。代码把正确类别固定为索引 1，避免把“标签值”和“类别位置”混在一起。

```python
from math import exp, log


def softmax(z):
    shift = max(z)
    weights = [exp(value - shift) for value in z]
    total = sum(weights)
    return [value / total for value in weights]


def smooth_target(k, correct, eps, mode="all"):
    if mode == "all":
        return [
            (1.0 - eps) * (i == correct) + eps / k
            for i in range(k)
        ]
    return [
        (1.0 - eps) if i == correct else eps / (k - 1)
        for i in range(k)
    ]


def cross_entropy(probs, target):
    return -sum(q * log(p) for p, q in zip(probs, target))


def loss_from_logits(z, target):
    return cross_entropy(softmax(z), target)


z = [1.2, 0.4, -0.3, -0.8]
k = len(z)
correct = 1
eps = 0.2
probs = softmax(z)
hard = smooth_target(k, correct, 0.0, "all")
all_target = smooth_target(k, correct, eps, "all")
other_target = smooth_target(k, correct, eps, "other")
grad_hard = [p - q for p, q in zip(probs, hard)]
grad_all = [p - q for p, q in zip(probs, all_target)]
grad_other = [p - q for p, q in zip(probs, other_target)]
h = 1e-6
numeric = []
for i in range(k):
    plus = list(z)
    minus = list(z)
    plus[i] += h
    minus[i] -= h
    numeric.append(
        (loss_from_logits(plus, all_target)
         - loss_from_logits(minus, all_target))
        / (2 * h)
    )

print("prob=", [round(value, 9) for value in probs])
print("hard=", [round(value, 9) for value in hard],
      "uniform-all=", [round(value, 9) for value in all_target],
      "uniform-other=", [round(value, 9) for value in other_target])
print("loss hard/all/other=",
      *(f"{value:.9f}" for value in (
          cross_entropy(probs, hard),
          cross_entropy(probs, all_target),
          cross_entropy(probs, other_target),
      )))
print("grad hard=", [round(value, 9) for value in grad_hard])
print("grad all=", [round(value, 9) for value in grad_all])
print("grad other=", [round(value, 9) for value in grad_other])
print("target logit gap all/other=",
      f"{log(all_target[correct] / all_target[0]):.9f}",
      f"{log(other_target[correct] / other_target[0]):.9f}")
print("numeric grad=", [round(value, 9) for value in numeric],
      "max-error=",
      f"{max(abs(a - n) for a, n in zip(grad_all, numeric)):.3e}")
```

运行输出：

```text
prob= [0.553160246, 0.248550921, 0.123426734, 0.074862099]
hard= [0.0, 1.0, 0.0, 0.0] uniform-all= [0.05, 0.85, 0.05, 0.05] uniform-other= [0.066666667, 0.8, 0.066666667, 0.066666667]
loss hard/all/other= 1.392107543 1.447107543 1.465440876
grad hard= [0.553160246, -0.751449079, 0.123426734, 0.074862099]
grad all= [0.503160246, -0.601449079, 0.073426734, 0.024862099]
grad other= [0.48649358, -0.551449079, 0.056760068, 0.008195432]
target logit gap all/other= 2.833213344 2.484906650
numeric grad= [0.503160246, -0.601449079, 0.073426734, 0.024862099] max-error= 1.016e-10
```

中心差分与 uniform-all 的解析梯度最大误差约为 $1.016\times10^{-10}$。输出还显示 uniform-all 与 uniform-other 的损失和目标 logit 间隔都不同；只记录 epsilon 而不记录平滑约定，无法复现实验。

## 实现审计和失效模式

接入框架损失时，可以把下面的检查放到单元测试和训练配置中：

| 检查项 | 应看到的事实 | 常见错误 |
| --- | --- | --- |
| 类别与目标 | $K$、正确类索引和 $\boldsymbol q$ 总和明确 | 类别编号错位 |
| 平滑约定 | uniform-all、uniform-other 或先验分布已注明 | 同名 epsilon 直接比较 |
| logits 计算 | 使用稳定 log-sum-exp 或等价实现 | 大 logits 溢出 |
| 梯度 | 每个 logit 为 $p_k-q_k$，总和为 0 | 仍沿用 $p-y$ |
| ignore 位置 | 无效标签不进入目标和分母 | 先平滑再误计入 mask |
| soft target 来源 | label、mixup、蒸馏和先验分开记录 | 双重平滑 |
| 评估指标 | hard NLL、校准与训练 loss 分开 | 把平滑 loss 当真实 NLL |

**把 uniform-all 写成 uniform-other。** 用一个 $K=4$、$\varepsilon=0.2$ 的手算例比较正确类和错误类质量；不要只看损失函数的参数名。

**在概率上直接加 epsilon。** 正确做法是先构造目标 $\boldsymbol q$，再计算 $-\sum q_k\log p_k$。把 epsilon 加到预测概率、再重新归一化，是另一个函数，不能叫标签平滑。

**只减小正确类目标。** 其它类的平滑质量仍参与梯度。若实现只把 $1$ 改成 $1-\varepsilon$，却没有把剩余质量分配给其它类，目标总和就不是 1。

**对已经是软标签的目标再自动平滑。** mixup、蒸馏或人工概率标签已经提供了目标分布。需要再平滑时，应明确写成 $(1-\varepsilon)\boldsymbol r+\varepsilon\boldsymbol u$，并记录两次操作的来源。

**把平滑后的训练 loss 当作校准证据。** 训练目标改变后，loss 数值不能和 hard-label NLL 直接比较。至少用同一验证集按 one-hot 重新计算概率 NLL，并配合 Brier 或可靠性图。

**把多标签 BCE 当成互斥 softmax。** 多标签目标不要求各标签总和为 1；输出头、平滑公式和阈值都要按独立 Bernoulli 任务审计。

## 运行方法

把上面的代码保存为 label_smoothing_probe.py，在项目环境中运行：

```bash
python3 label_smoothing_probe.py
```

真实训练中先固定 $\varepsilon=0$ 的基线，再只改变平滑系数；同时保存目标分布摘要、训练 loss、hard-label NLL、准确率和校准指标。若还使用 mixup、知识蒸馏、class weight 或 ignore mask，要把目标构造顺序和每个归约分母写进配置。

## 相关词条

- [分类损失](../training-nn/classification-losses/)：先看 softmax CE、BCE、focal 和软目标的共同梯度形式。
- [Softmax 回归](../linear-models/softmax-regression/)：推导 logits、概率、交叉熵和 $p-y$ 梯度。
- [Softmax](../neurons-and-activations/softmax/)：检查归一化轴、平移不变性和数值稳定性。
- [交叉熵](../information-theory/cross-entropy/)：从概率分布和 KL 散度理解平滑目标的最优解。
- [分类指标](../evaluation-and-generalization/classification-metrics/)：分开报告 hard-label 决策、排序和概率校准。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：比较标签平滑与权重、数据和激活层面的正则化。
- [监督学习](../learning-framework/supervised-learning/)：区分训练目标、数据分布和部署决策。
