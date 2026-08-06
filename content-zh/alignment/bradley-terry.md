---
title: "Bradley–Terry 模型：从成对比较估计排序分数"
tags: ["why-models-learn"]
---

Bradley–Terry 模型把同一组候选的成对比较写成概率模型：每个候选有一个潜在分数，两个候选的分差决定其中一个被选中的概率。它的训练数据只有“在这次比较中谁胜出”，却能通过共享分数把局部比较合成为全局排序。本文从成对似然推导梯度和 Hessian，说明平移不可识别、比较图连通性、完全分离、平局和标签噪声，再用 Newton 探针核对最大似然估计的数值行为。[奖励模型](../alignment/reward-model/)经常用这个似然训练评分器，但 Bradley–Terry 模型本身是偏好数据的统计层，不是基础模型、奖励头或策略优化器。

![Bradley–Terry 模型示意图：候选比较图经过成对概率和最大似然估计得到排序分数，平移基准与独立评测分别保留](/assets/alignment/svg/bradley-terry.1.svg)

## 成对比较只观察相对胜负

设候选集合为 $\mathcal I=\{1,\ldots,m\}$。候选可以是回答、推荐结果、图像、动作或同一输入下的其他可比较对象。一次比较先固定上下文 $x$，再呈现两个候选 $y_i$ 和 $y_j$，记录标签：

$$
(x,y_i,y_j,z),\qquad
z=
\begin{cases}
1,&y_i\succ y_j,\\
0,&y_j\succ y_i.
\end{cases}
$$

这里的 $\succ$ 只表示标注协议下的选择，不自动表示客观质量。若同一对候选被比较多次，可以把统计结果写成：

| 符号 | 含义 | 需要固定的内容 |
|---|---|---|
| $w_{ij}$ | 候选 $i$ 胜过候选 $j$ 的次数 | 上下文、展示顺序、标签规则 |
| $w_{ji}$ | 候选 $j$ 胜过候选 $i$ 的次数 | 同一批候选和同一标签协议 |
| $n_{ij}=w_{ij}+w_{ji}$ | 这对候选的有效比较次数 | 是否排除平局、无效或超时标签 |
| $p_{ij}$ | 模型预测 $i$ 胜过 $j$ 的概率 | 分数参数和噪声尺度 |

成对数据没有直接给出“候选 $i$ 的绝对质量是 0.8”。它只给出一个方向和一个重复频率。若 $w_{ij}=8$、$w_{ji}=2$，观察到的胜率是 $0.8$；这个数字只描述这对候选在当前上下文和标签协议下的选择频率。

把所有比较画成图可以提前发现数据范围。每个候选是一个节点，存在比较就连接一条无向边；若保留胜负方向，则每次 $i$ 胜过 $j$ 产生一条 $i\to j$ 的有向边。图中没有跨组件的边时，不可能从数据估计组件之间的相对位置。

### 一个最小的比较记录

四个候选可以形成下面的记录：

| 候选对 | $i$ 胜出 | $j$ 胜出 | 观察胜率 $i$ |
|---|---:|---:|---:|
| A、B | 7 | 3 | 0.700 |
| A、C | 8 | 2 | 0.800 |
| A、D | 6 | 4 | 0.600 |
| B、C | 6 | 4 | 0.600 |
| B、D | 5 | 5 | 0.500 |
| C、D | 4 | 6 | 0.400 |

逐行比较只能得到六个局部胜率。Bradley–Terry 模型要求这些胜率由同一组候选分数共同解释，因此 A 对 C 的优势、C 对 D 的劣势和 B 对 D 的平局会同时影响每个候选的估计值。

## 分数差决定 Bradley–Terry 概率

给候选 $i$ 一个分数 $s_i\in\mathbb R$。经典 Bradley–Terry 概率为：

$$
p_{ij}
  =\Pr(i\succ j)
  =\frac{\exp(s_i)}{\exp(s_i)+\exp(s_j)}
  =\sigma(s_i-s_j).
$$

其中 $\sigma(t)=1/(1+\exp(-t))$。只有分差 $s_i-s_j$ 进入概率，所以两个候选分数同时增加同一个常数时，概率不变。

将概率改写成 odds：

$$
\frac{p_{ij}}{1-p_{ij}}
  =\exp(s_i-s_j),
\qquad
\log\frac{p_{ij}}{1-p_{ij}}
  =s_i-s_j.
$$

这说明 $s_i-s_j$ 是成对选择的对数几率。分差为 0 时，两个候选的概率都是 0.5；分差为 $\log 4$ 时，$i$ 的 odds 是 $j$ 的 4 倍，概率为 0.8。

### 从随机效用得到 logistic 形式

可以给每个候选定义一个潜在效用：

$$
U_i=s_i+\varepsilon_i,
\qquad
i\succ j\Longleftrightarrow U_i>U_j.
$$

如果 $\varepsilon_i$ 和 $\varepsilon_j$ 独立同分布，并且都服从相同尺度的 Gumbel 分布，那么两个噪声之差服从 logistic 分布。于是：

$$
\Pr(U_i>U_j)
  =\Pr(\varepsilon_j-\varepsilon_i<s_i-s_j)
  =\sigma\left(\frac{s_i-s_j}{\tau}\right).
$$

$\tau>0$ 是选择噪声尺度。将 $s_i$ 全部乘以常数与改变 $\tau$ 具有相同的效果，因此必须固定其中一个量。标准写法把噪声尺度吸收到分数中，使用 $\tau=1$。

| 分差 $s_i-s_j$ | $p_{ij}$ | odds $i:j$ | 解释 |
|---:|---:|---:|---|
| $0$ | $0.500000$ | $1:1$ | 比较没有方向 |
| $\log 2$ | $0.666667$ | $2:1$ | $i$ 的选择 odds 是两倍 |
| $\log 4$ | $0.800000$ | $4:1$ | $i$ 的选择 odds 是四倍 |
| $-\log 4$ | $0.200000$ | $1:4$ | $j$ 的选择 odds 是四倍 |

分数差的数值依赖噪声尺度。排序只依赖分数顺序，概率解释还依赖尺度是否在训练和评测中保持一致。

## 似然把局部标签合成为全局分数

对于一对候选，$w_{ij}$ 次记录了 $i$ 胜出，$w_{ji}$ 次记录了 $j$ 胜出。假设标签在给定分数后条件独立，成对对数似然是：

$$
\ell_{ij}
  =w_{ij}\log p_{ij}
   +w_{ji}\log(1-p_{ij}).
$$

将所有被比较的无序候选对记为 $\mathcal E$，整体对数似然为：

$$
\ell(s)
  =\sum_{(i,j)\in\mathcal E}
    \left[
      w_{ij}\log p_{ij}
      +w_{ji}\log(1-p_{ij})
    \right].
$$

训练时通常最小化负对数似然 $-\ell(s)$。当只有一对候选时，最大似然会让 $p_{ij}$ 接近观察胜率 $w_{ij}/n_{ij}$；当候选共享多条边时，每个分数必须同时满足多组比较，局部胜率不一定可以逐行复现。

### 梯度只累加比较边的残差

定义一条比较边的有向特征向量：

$$
x_{ij,k}
  =
  \begin{cases}
  1,&k=i,\\
  -1,&k=j,\\
  0,&\text{其他候选}.
  \end{cases}
$$

分差可以写成 $d_{ij}=s_i-s_j=x_{ij}^{\mathsf T}s$。对一条边求导：

$$
\begin{aligned}
\frac{\partial \ell_{ij}}{\partial d_{ij}}
  &=w_{ij}(1-p_{ij})-w_{ji}p_{ij}\\
  &=w_{ij}-n_{ij}p_{ij}.
\end{aligned}
$$

再用 $\partial d_{ij}/\partial s_k=x_{ij,k}$，得到：

$$
\nabla_s\ell(s)
  =\sum_{(i,j)\in\mathcal E}
    \left(w_{ij}-n_{ij}p_{ij}\right)x_{ij}.
$$

$w_{ij}-n_{ij}p_{ij}$ 是观察胜出次数与模型期望胜出次数的差。梯度为零时，每个候选的总观察胜出次数和总期望胜出次数在参数约束下平衡；单条边的概率不必等于该边的观察胜率。

### Hessian 是加权图 Laplacian

对概率求导：

$$
\frac{\partial p_{ij}}{\partial d_{ij}}
  =p_{ij}(1-p_{ij}).
$$

因此：

$$
\nabla_s^2\ell(s)
  =-\sum_{(i,j)\in\mathcal E}
    n_{ij}p_{ij}(1-p_{ij})
    x_{ij}x_{ij}^{\mathsf T}.
$$

每个外积 $x_{ij}x_{ij}^{\mathsf T}$ 都是比较图的一条边的 Laplacian 贡献，权重为 $n_{ij}p_{ij}(1-p_{ij})$。它是半负定的，说明未加正则化的对数似然是凹函数；固定一个基准或施加零均值约束后，在可识别条件下可以用 Newton 法寻找全局最大值。

| 量 | 数学形式 | 数据含义 |
|---|---|---|
| 观察次数 | $w_{ij}$ | 这条边上实际支持 $i$ 的标签数 |
| 期望次数 | $n_{ij}p_{ij}$ | 当前分数下模型预计支持 $i$ 的次数 |
| 梯度残差 | $w_{ij}-n_{ij}p_{ij}$ | 需要增加或减少分差的方向 |
| Hessian 权重 | $n_{ij}p_{ij}(1-p_{ij})$ | 这条边对曲率和信息量的贡献 |

当 $p_{ij}$ 接近 0 或 1 时，$p_{ij}(1-p_{ij})$ 变小。模型已经非常确定的边对局部曲率贡献较少；接近 0.5 的边提供更多区分相近分数的信息。

## 平移不可识别，比较图决定可估计范围

### 全局平移不改变任何预测

对任意常数 $c$，令 $s'_i=s_i+c$，则：

$$
s'_i-s'_j=s_i-s_j,
\qquad
p_{ij}(s')=p_{ij}(s).
$$

所以最大似然解不是一个孤立点，而是一条平移等价类。下面两种约束常用：

1. 固定一个候选为基准，例如 $s_D=0$；
2. 加入零均值约束 $\sum_i s_i=0$。

两种参数化得到的概率、排序和分差相同。基准分数本身没有质量含义，不能把 $s_D=0$ 解读成“D 的真实质量为零”。

### 无向连通性只解决平移，方向连通性还影响有限解

无向比较图连通时，所有候选至少通过比较边连接到同一个分数坐标系。如果图分成两个组件，可以给其中一个组件整体加常数而不改变任何已观测概率，因此组件之间的相对排名没有数据依据。

要让无正则化的最大似然保持有限，比较还需要避免完全分离。直观判据是：不能找到一个候选子集，使该子集中的候选在所有跨集合比较中都胜出，且没有反向损失。用有向胜负边表示时，强连通是常用的充分检查；若只存在从集合 A 指向集合 B 的边，增大 A 的分数会持续提高似然而不会被数据阻止。

| 数据状态 | 数学后果 | 审计动作 |
|---|---|---|
| 无向图有多个组件 | 组件间分数差不可识别 | 增加跨组件比较，或分组件报告 |
| 某候选在所有边上都获胜 | 分数可能趋向正无穷 | 记录分离，加入先验或收集反向样本 |
| 某候选只被比较一次 | 分数方差高 | 增加重复标签和相邻难度候选 |
| 比较边集中在同一类 prompt | 排序只覆盖局部任务分布 | 按任务、难度和生成来源分层评测 |
| 所有边的概率接近 0.5 | 排序差异低于分辨率 | 增加标签量或报告置信区间 |

“有一个全局排序”不等于“所有分数都能从当前数据估计”。排序范围由比较图和标签质量共同决定。

## 一个可运行的 Newton 探针

下面的探针固定 $s_D=0$，对四个候选的六组比较最大化 Bradley–Terry 对数似然。它显式构造梯度和 Hessian，再解 $H\Delta=-g$ 更新分数。数据图是连通的，每一对都有两个方向的胜出记录，因而不会用一个单向边界把分数推向无穷。

```python
import math


items = ["A", "B", "C", "D"]
anchor = "D"
free = ["A", "B", "C"]
pairs = [
    ("A", "B", 7, 3),
    ("A", "C", 8, 2),
    ("A", "D", 6, 4),
    ("B", "C", 6, 4),
    ("B", "D", 5, 5),
    ("C", "D", 4, 6),
]


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def solve_linear(matrix, vector):
    augmented = [
        [float(value) for value in row] + [float(rhs)]
        for row, rhs in zip(matrix, vector)
    ]
    size = len(vector)
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= (
                    factor * augmented[column][index]
                )
    return [augmented[row][size] for row in range(size)]


def gradient_hessian(theta):
    scores = dict(zip(free, theta))
    scores[anchor] = 0.0
    gradient = {item: 0.0 for item in free}
    hessian = [[0.0 for _ in free] for _ in free]

    for winner, loser, winner_count, loser_count in pairs:
        total = winner_count + loser_count
        probability = sigmoid(scores[winner] - scores[loser])
        edge_gradient = winner_count - total * probability
        edge = [
            (1 if item == winner else 0)
            - (1 if item == loser else 0)
            for item in free
        ]
        for row, item in enumerate(free):
            gradient[item] += edge_gradient * edge[row]
            for column in range(len(free)):
                hessian[row][column] -= (
                    total
                    * probability
                    * (1.0 - probability)
                    * edge[row]
                    * edge[column]
                )
    return [gradient[item] for item in free], hessian, scores


def negative_log_likelihood(scores):
    value = 0.0
    for winner, loser, winner_count, loser_count in pairs:
        difference = scores[winner] - scores[loser]
        value += winner_count * math.log1p(math.exp(-difference))
        value += loser_count * math.log1p(math.exp(difference))
    return value


theta = [0.0, 0.0, 0.0]
for step in range(20):
    gradient, hessian, scores = gradient_hessian(theta)
    if max(abs(value) for value in gradient) < 1e-12:
        break
    delta = solve_linear(hessian, [-value for value in gradient])
    theta = [value + change for value, change in zip(theta, delta)]

gradient, hessian, scores = gradient_hessian(theta)
print("newton_steps=", step + 1)
print("scores=", {
    item: f"{scores[item]:.12f}" for item in items
})
print(
    "negative_log_likelihood=",
    f"{negative_log_likelihood(scores):.12f}",
)
print("gradient=", [
    f"{value:.12e}" for value in gradient
])

for winner, loser, winner_count, loser_count in pairs:
    probability = sigmoid(scores[winner] - scores[loser])
    observed = winner_count / (winner_count + loser_count)
    print(
        f"{winner}>{loser}",
        "observed=",
        f"{observed:.12f}",
        "predicted=",
        f"{probability:.12f}",
    )

print("ranking=", sorted(
    items,
    key=lambda item: scores[item],
    reverse=True,
))

shifted = {item: score + 3.0 for item, score in scores.items()}
print(
    "shift_check_A>D=",
    f"{sigmoid(scores['A'] - scores['D']):.12f}",
    f"{sigmoid(shifted['A'] - shifted['D']):.12f}",
)

finite_difference_errors = []
for index in range(len(free)):
    step_size = 1e-5
    plus = theta[:]
    minus = theta[:]
    plus[index] += step_size
    minus[index] -= step_size
    plus_scores = dict(zip(free, plus))
    minus_scores = dict(zip(free, minus))
    plus_scores[anchor] = 0.0
    minus_scores[anchor] = 0.0
    numerical = (
        negative_log_likelihood(plus_scores)
        - negative_log_likelihood(minus_scores)
    ) / (2.0 * step_size)
    finite_difference_errors.append(
        abs(numerical + gradient[index])
    )
print(
    "max_negative_log_likelihood_gradient_error=",
    f"{max(finite_difference_errors):.3e}",
)

for scale in (0.5, 1.0, 2.0):
    probability = sigmoid(
        scale * (scores["A"] - scores["D"])
    )
    print(
        "scale=",
        scale,
        "A>D_probability=",
        f"{probability:.12f}",
    )
```

输出为：

```text
newton_steps= 5
scores= {'A': '0.645965697183', 'B': '-0.104444281687', 'C': '-0.532274329238', 'D': '0.000000000000'}
negative_log_likelihood= 38.382049557809
gradient= ['1.776356839400e-14', '-3.552713678801e-15', '-1.287858708565e-14']
A>B observed= 0.700000000000 predicted= 0.679268024956
A>C observed= 0.800000000000 predicted= 0.764631208283
A>D observed= 0.600000000000 predicted= 0.656100766761
B>C observed= 0.600000000000 predicted= 0.605355384956
B>D observed= 0.500000000000 predicted= 0.473912640000
C>D observed= 0.400000000000 predicted= 0.369986593239
ranking= ['A', 'D', 'B', 'C']
shift_check_A>D= 0.656100766761 0.656100766761
max_negative_log_likelihood_gradient_error= 1.776e-14
scale= 0.5 A>D_probability= 0.580051022706
scale= 1.0 A>D_probability= 0.656100766761
scale= 2.0 A>D_probability= 0.784473918301
```

探针得到的分数满足三个条件。第一，梯度的数量级约为 $10^{-14}$，已经接近双精度舍入误差。第二，A、B、C、D 的排序为 A、D、B、C；D 被固定为 0 只是坐标选择，不能解释为 D 的绝对质量。第三，把所有分数同时加 3 后，A 胜过 D 的概率保持不变，验证了平移不可识别。

预测胜率没有逐行等于观察胜率。例如 A 胜过 D 的观察胜率为 0.6，模型预测为 0.656100766761。这是全局拟合的结果：A、D 还分别和其他候选比较，单条边的误差由其他边共同承担。若只对 A、D 单独拟合，最大似然概率才会直接等于 0.6。

### Newton 更新为什么有效

在固定基准后，负对数似然的 Hessian 是正定或接近正定，Newton 更新使用当前位置的曲率把梯度残差转换为分数位移：

$$
\theta_{\text{new}}
  =\theta-H_{\text{NLL}}^{-1}g_{\text{NLL}}.
$$

探针写成最大化对数似然的形式 $H_{\ell}\Delta=-g_{\ell}$，与最小化负对数似然的写法相同。实际工程实现还要处理线搜索、阻尼、极端 logits、稀疏边和批量累加；不能把三维教学探针直接当作大规模训练器。

## 全局排序不等于逐边复制

### 两个候选时，分差接近胜率的 log odds

如果数据只包含一对候选，并且没有先验，最大似然满足：

$$
\hat p_{ij}
  =\frac{w_{ij}}{w_{ij}+w_{ji}},
\qquad
\hat s_i-\hat s_j
  =\log\frac{w_{ij}}{w_{ji}},
$$

前提是两种结果的计数都大于 0。若其中一个计数为 0，log odds 变成无穷，正是完全分离的两候选特例。

### 多个候选共享同一组分数

A、B、C、D 的数据包含环状约束：

$$
A\succ B,\quad
B\succ C,\quad
C\prec D,\quad
A\succ D.
$$

这些符号只表示观察中较多的一侧，不表示每次标签都一致。共享分数要求每条边使用同一套 $s$，因此模型会在相互不完全一致的比较之间进行最大似然折中。

| 目标 | 直接逐对估计 | Bradley–Terry 全局估计 |
|---|---|---|
| 输出 | 每条边一个胜率 | 每个候选一个分数 |
| 传递关系 | 需要额外排序规则 | 由共享分数自然产生 |
| 数据复用 | 一条边只影响自己 | 一条边会影响两端候选及相邻边 |
| 不一致比较 | 可能产生循环排序 | 以整体似然代价分配残差 |
| 跨 prompt 解释 | 不自动成立 | 仍需要固定上下文层和校准集 |

全局排序解决的是参数共享问题，不会自动解决候选集合变化、任务分布变化或标签标准变化。

## 平局、重复标签和选择噪声

标准 Bradley–Terry 只允许 $i$ 胜或 $j$ 胜。实际偏好数据常包含平局、跳过、无法判断和标注者分歧。把平局静默转换成半个胜场会改变似然，必须把转换规则写进数据版本。

### Davidson 模型显式表示平局

一种常见扩展在分母中增加平局项。设 $\nu>0$ 是平局倾向：

$$
\begin{aligned}
Z_{ij}
  &=\exp(s_i)+\exp(s_j)
    +\nu\exp\left(\frac{s_i+s_j}{2}\right),\\
\Pr(i\succ j)
  &=\frac{\exp(s_i)}{Z_{ij}},\\
\Pr(j\succ i)
  &=\frac{\exp(s_j)}{Z_{ij}},\\
\Pr(i\sim j)
  &=\frac{\nu\exp\left((s_i+s_j)/2\right)}{Z_{ij}}.
\end{aligned}
$$

分母中的三项共享同一组候选分数。$\nu$ 越大，在分差相同的情况下平局概率越高。若把平局拆为半胜半负，模型不会知道平局是“两个回答接近”还是“标注者无法判断”。

### 多次标签提供重复测量

同一比较的重复标签可以估计观察噪声，但不能把重复次数当作独立事实而不记录标注者和上下文。至少要保存：

| 字段 | 作用 |
|---|---|
| 标注者或评审器版本 | 区分不同偏好分布 |
| 比较顺序 | 检查位置偏差 |
| 标签状态 | 胜、负、平局、跳过、无效 |
| 指南版本 | 解释标准变化 |
| 候选生成 checkpoint | 记录比较覆盖范围 |
| 时间和任务分层 | 发现分布漂移 |

如果不同标注者的选择概率明显不同，可以使用标注者随机效应、分层噪声尺度或分组报告。先把标签混成一个计数再寻找解释，会丢失分歧来源。

## 比较图要提供可识别的信息

### 信息量在概率中间最大

一条比较边的 Hessian 权重为：

$$
q_{ij}=n_{ij}p_{ij}(1-p_{ij}).
$$

固定 $n_{ij}$ 时，$p_{ij}(1-p_{ij})$ 在 $p_{ij}=0.5$ 取得最大值 $0.25$。这不表示应该只采集“看起来平局”的样本，因为极端样本仍然能确定排序方向；它说明对两个分数接近的候选增加比较，通常比重复明显胜负更能缩小局部不确定性。

### 先保证跨组件边，再增加重复次数

若比较图不连通，再多重复组件内部的标签也不能估计组件间差异。一个实际的设计顺序是：

1. 检查无向图是否连通；
2. 对重要组件增加跨组件比较；
3. 为分数接近的候选增加重复标签；
4. 在不同 prompt、难度和生成来源上保留边；
5. 用未参与训练的比较边检查排序泛化。

| 采样目标 | 适合的比较 | 主要观察量 |
|---|---|---|
| 连接图 | 不同候选簇之间的边 | 组件是否合并 |
| 区分近邻 | 预测概率接近 0.5 的边 | 分差和置信区间 |
| 检查边界 | 已知容易失败或争议的候选 | 分离、标签分歧和异常概率 |
| 检查分布 | 不同任务和候选来源的同类比较 | 分组排序和校准 |
| 监测漂移 | 新 checkpoint 与固定参考候选比较 | 分数差、胜率和时间趋势 |

候选生成分布也是模型输入的一部分。只比较同一 checkpoint 的高质量回答，得到的排序可能无法覆盖策略优化后出现的格式投机、长度偏差和事实错误。

## 正则化和尺度约束改变什么

### L2 先验提供有限分数

当比较图分离或标签完全单向时，可以加入零均值的 L2 惩罚：

$$
\ell_{\lambda}(s)
  =\ell(s)
   -\frac{\lambda}{2}
    \sum_{i=1}^{m}(s_i-\bar s)^2,
\qquad
\bar s=\frac{1}{m}\sum_{i=1}^{m}s_i.
$$

它等价于对相对分数施加高斯先验，避免分数无界增长。$\lambda$ 越大，分数越靠近共同中心，预测概率越接近 0.5；这会降低方差，也会把真实的强差异收缩。报告结果时要同时保存 $\lambda$、锚点或零均值约束和训练数据范围。

### 分数尺度不只影响排序

若把分数乘以 $\kappa>0$，排序不变，但：

$$
p_{ij}^{(\kappa)}
  =\sigma\left(\kappa(s_i-s_j)\right).
$$

探针中 $\kappa=0.5$、$1$、$2$ 时 A 胜过 D 的概率分别为 0.580051022706、0.656100766761 和 0.784473918301。策略优化如果直接读取 reward 数值或梯度，尺度还会改变更新幅度。因此下面三件事不能混用：

| 记录 | 可以回答的问题 | 不能单独回答的问题 |
|---|---|---|
| 分数排序 | 哪个候选被评分器排在前面 | 概率是否校准 |
| Bradley–Terry 概率 | 给定噪声尺度下的选择概率 | 真实任务成功率 |
| 独立结果 | 事实正确性、用户结果或约束违反 | 偏好模型内部是否稳定 |

### 校准需要留出比较集

在留出集上，把模型概率按区间分桶，比较每个桶的平均预测概率与实际胜率。如果预测概率为 0.8 的样本实际胜率只有 0.6，排序可能仍然可用，但概率解释过于自信。温度缩放或其他校准方法只能修正保留集上观察到的概率关系，不能补齐未出现的候选类型。

## Bradley–Terry 在对齐流程中的位置

在语言模型对齐流程中，常见数据路径是：

$$
\text{候选回答}
\longrightarrow
\text{成对标签}
\longrightarrow
\text{Bradley–Terry 似然}
\longrightarrow
\text{reward model}
\longrightarrow
\text{策略优化}.
$$

这条路径中的对象不同：

| 对象 | 学习或计算的内容 | 不能替代的对象 |
|---|---|---|
| Bradley–Terry 模型 | 偏好标签在分数差下的概率 | 基础模型表示 |
| reward model | 输入和候选到代理分数的函数 | 人类意图和权限系统 |
| 策略梯度或其他优化器 | 根据分数改变候选分布 | 标签生成和独立评测 |
| 独立评测 | 真实任务、事实、安全和长期结果 | 训练损失本身 |

[奖励模型](../alignment/reward-model/)可以使用 Bradley–Terry 成对损失，但 reward model 的输入表示、pooling 位置、候选模板和训练参数仍然需要独立审计。[策略梯度](../alignment/policy-gradient/)读取的是奖励信号，不会自动知道这个信号是否由一个错误的偏好模型产生。[RLHF 与 DPO](../alignment/rlhf-dpo/)使用不同的优化路径，不能因为都出现偏好对就把模型、目标和推理边界视为同一件事。

## 失效模式与审计清单

### 先查参数是否可识别

- 检查候选节点数、无向边数和连通组件数。
- 检查是否存在只有单向胜负的分离集合。
- 检查锚点、零均值或正则化是否写入配置。
- 检查比较边是否跨越任务、难度和候选来源。

### 再查似然实现

- 用小数据手算一条边的概率、损失和梯度。
- 用中心差分核对解析梯度。
- 对极端分差使用稳定的 log-sigmoid 或 log-sum-exp。
- 检查 Hessian 的符号、对称性和基准约束。
- 检查重复标签、平局和无效样本的计数规则。

### 最后查解释边界

| 现象 | 可能原因 | 最小核验 |
|---|---|---|
| 分数越来越大 | 完全分离或缺少正则化 | 查看有向比较图和梯度范数 |
| 训练损失下降但留出排序变差 | 过拟合或候选分布漂移 | 按候选来源分组比较 |
| 概率接近 0 或 1 | 分数尺度过大或标签过少 | 保存分差分布并做校准 |
| 排序在不同 prompt 间反转 | 上下文交互或偏好标准不同 | 分层训练和分组评测 |
| 长回答稳定得分更高 | 长度特征或模板泄漏 | 长度匹配、字段消融和改写 |
| 平局被当作半胜后结果变化 | 标签转换改变了似然 | 比较显式平局模型和转换版本 |
| 新策略 reward 上升但独立质量下降 | 分布外外推或 reward hacking | 同时记录独立结果和约束违反率 |

一份可追溯的 Bradley–Terry 实验至少保存：候选和上下文的 hash、比较图、标签指南、标注者或评审器版本、胜负和平局计数、基准约束、噪声尺度、正则化、训练/留出划分、分数分布、pairwise accuracy、校准曲线和分组结果。只保存最终排序不能重建模型为什么给出该排序。

## 运行方法

将上一个 Python 代码块保存为 bradley-terry-probe.py，再运行：

```bash
python3 bradley-terry-probe.py
```

修改比较次数、候选集合或锚点后，重新核对 Newton 梯度、有限差分误差、平移不变性、排序和分数缩放。需要加入平局时，不要只把新标签静默折算成半胜；先固定 Davidson 参数和比较数据的版本。

## 相关词条

- [奖励模型](../alignment/reward-model/)：把偏好或规则反馈映射为后续优化可读取的代理分数。
- [对齐问题](../alignment/alignment-problem/)：区分偏好代理目标与真实意图、约束和独立结果。
- [策略梯度](../alignment/policy-gradient/)：说明策略如何读取奖励并通过 log-derivative 更新参数。
- [RLHF 与 DPO](../alignment/rlhf-dpo/)：比较偏好数据进入策略优化的两条路径。
- [逻辑回归](../linear-models/logistic-regression/)：复用 logistic 概率、对数几率和最大似然的基础形式。
- [最大似然估计](../probability/maximum-likelihood/)：说明如何用观测数据选择参数。
- [训练稳定性](../pretraining/training-stability/)：检查损失、梯度、数值范围和分布漂移。
