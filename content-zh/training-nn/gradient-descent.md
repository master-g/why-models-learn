---
title: "梯度下降：把损失变成参数更新"
tags: ["why-models-learn"]
---

梯度下降做的事情很具体：在当前参数处计算损失对参数的梯度，然后沿梯度的反方向移动一小步。反向传播负责回答“梯度是什么”，梯度下降负责回答“拿这个梯度移动多远”。方向、步长、目标尺度、批次来源和停止条件任何一项错位，训练曲线都可能看起来像一个优化器问题，实际却是更新协议的问题。

![梯度下降在不同曲率和步长下的轨迹：左侧展示细长谷底中的之字形，右侧比较稳定、过小和过大的步长](/assets/training-nn/svg/gradient-descent.1.svg)

## 梯度告诉我们最陡的上坡方向

设参数为 $\boldsymbol\theta\in\mathbb R^d$，目标函数为

$$
J(\boldsymbol\theta).
$$

在当前点 $\boldsymbol\theta$ 沿任意方向 $\boldsymbol v$ 移动一个很小的标量 $t$：

$$
J(\boldsymbol\theta+t\boldsymbol v)
\approx
J(\boldsymbol\theta)
+t\,\nabla J(\boldsymbol\theta)^{\mathsf T}\boldsymbol v.
$$

一阶变化由内积决定。若固定方向长度 $\lVert\boldsymbol v\rVert_2=1$，Cauchy–Schwarz 不等式给出

$$
\nabla J(\boldsymbol\theta)^{\mathsf T}\boldsymbol v
\geq
-\lVert\nabla J(\boldsymbol\theta)\rVert_2,
$$

等号在

$$
\boldsymbol v
=-\frac{\nabla J(\boldsymbol\theta)}
{\lVert\nabla J(\boldsymbol\theta)\rVert_2}
$$

时取得。因此负梯度是局部一阶近似中下降最快的单位方向。这里的“最快”只针对当前点、当前参数坐标和一阶变化，不保证走完一个有限距离后仍然最好。

梯度下降把这个方向写成

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k
-\eta_k\nabla J(\boldsymbol\theta_k),
$$

其中 $\eta_k>0$ 是学习率，也叫步长。梯度提供方向和每个坐标的相对尺度，学习率提供整个向量的共同缩放。

如果暂时忽略二阶项，一步的目标变化约为

$$
J(\boldsymbol\theta_{k+1})-J(\boldsymbol\theta_k)
\approx
-\eta_k\lVert\nabla J(\boldsymbol\theta_k)\rVert_2^2.
$$

只要梯度不为零，这个近似是负的。但“近似为负”依赖步长足够小；步长太大时二阶曲率项会超过一阶下降量。

## 一维二次函数把步长写成精确答案

先看最简单的目标：

$$
J(\theta)=\frac a2(\theta-\theta^\star)^2,
\qquad
a>0.
$$

梯度是

$$
J'(\theta)=a(\theta-\theta^\star).
$$

令误差 $e_k=\theta_k-\theta^\star$，更新后

$$
\begin{aligned}
e_{k+1}
&=\theta_{k+1}-\theta^\star\\
&=\theta_k-\eta a(\theta_k-\theta^\star)-\theta^\star\\
&=(1-\eta a)e_k.
\end{aligned}
$$

于是每一步都只是把误差乘上同一个因子：

$$
e_k=(1-\eta a)^ke_0.
$$

要让误差趋于零，必须满足

$$
\lvert1-\eta a\rvert<1,
$$

也就是

$$
0<\eta<\frac2a.
$$

这条边界解释了三种轨迹：

| 学习率范围 | 误差因子 | 轨迹 |
| --- | --- | --- |
| $0<\eta<1/a$ | $0<1-\eta a<1$ | 同侧平滑靠近最优点 |
| $\eta=1/a$ | $0$ | 一步到达一维二次函数的最优点 |
| $1/a<\eta<2/a$ | $-1<1-\eta a<0$ | 两侧交替但幅度缩小 |
| $\eta=2/a$ | $-1$ | 两侧等幅振荡，不收敛 |
| $\eta>2/a$ | $\lvert1-\eta a\rvert>1$ | 振荡并发散 |

以 $a=4$、$\theta^\star=0$、$\theta_0=2$ 为例：

$$
\eta=0.1
\quad\Longrightarrow\quad
e_{k+1}=0.6e_k,
$$

误差平滑下降；而

$$
\eta=0.6
\quad\Longrightarrow\quad
e_{k+1}=-1.4e_k,
$$

误差每一步换边且放大 $40\%$。训练日志里“loss 先下降后爆炸”常常就是局部有效曲率与学习率的乘积超过稳定范围。

### 损失值下降不等于每个坐标都单调

在 $\eta>1/a$ 的稳定区间内，误差会换符号。参数坐标可能来回穿过最优点，但目标值仍然下降，因为目标依赖的是误差平方：

$$
J(\theta_k)=\frac a2e_k^2.
$$

所以不能把“某个权重本轮变小”当成梯度下降正确的判据。应检查目标、梯度方向和更新后的参数是否与公式一致。

## 多维参数的步长同时面对所有曲率

对二次目标

$$
J(\boldsymbol\theta)
=\frac12
(\boldsymbol\theta-\boldsymbol\theta^\star)^{\mathsf T}
H
(\boldsymbol\theta-\boldsymbol\theta^\star),
$$

假设 $H$ 是对称正定矩阵。令误差

$$
\boldsymbol e_k
=\boldsymbol\theta_k-\boldsymbol\theta^\star.
$$

梯度为 $H\boldsymbol e_k$，更新变成

$$
\boldsymbol e_{k+1}
=(I-\eta H)\boldsymbol e_k.
$$

将 $H$ 做特征分解：

$$
H=U\Lambda U^{\mathsf T},
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_d).
$$

在特征坐标 $\boldsymbol z_k=U^{\mathsf T}\boldsymbol e_k$ 中，各方向独立更新：

$$
z_{k+1,i}
=(1-\eta\lambda_i)z_{k,i}.
$$

学习率必须同时对所有特征值稳定。二次正定情形的精确条件是

$$
0<\eta<\frac2{\lambda_{\max}}.
$$

最大曲率方向决定“不能太大”，最小曲率方向决定“沿谷底走得多慢”。若条件数

$$
\kappa=\frac{\lambda_{\max}}{\lambda_{\min}}
$$

很大，等高线会变成长条，梯度主要指向陡峭的横向，更新会在谷底两侧之字形摆动，而沿低曲率方向前进很慢。

二次目标的一个折中步长是

$$
\eta^\star
=\frac2{\lambda_{\max}+\lambda_{\min}},
$$

它让最大和最小曲率方向的误差收缩因子绝对值相等。这个公式只适用于已知谱的二次基线，不是深度网络的万能学习率；它的价值在于说明预条件和归一化为什么能改善不同方向的尺度差异。

## 从梯度到更新：训练循环中每一步的责任

一次确定性梯度下降迭代可以拆成六个可审计动作：

1. 取当前参数快照 $\boldsymbol\theta_k$；
2. 用同一个 batch 做前向计算，得到预测和逐样本损失；
3. 按约定的 sum、mean、权重和 mask 得到目标 $J_k$；
4. 反向传播得到 $\boldsymbol g_k=\nabla J(\boldsymbol\theta_k)$；
5. 对每个参数执行 $\boldsymbol\theta\leftarrow\boldsymbol\theta-\eta_k\boldsymbol g_k$；
6. 清空或替换梯度缓存，再进入下一轮。

第 2 步和第 3 步决定“正在优化什么”，第 4 步决定“这个目标对参数的敏感度”，第 5 步才是梯度下降本身。把梯度打印出来但不确认更新方向，无法证明参数真的沿负梯度移动。

在标量参数上，更新应满足

$$
\Delta\theta
=\theta_{k+1}-\theta_k
=-\eta g_k.
$$

因此可以用一个极小的单元测试验证：

| 梯度 $g_k$ | 学习率 $\eta$ | 应有更新 $\Delta\theta$ |
| ---: | ---: | ---: |
| 2 | 0.1 | -0.2 |
| -3 | 0.1 | 0.3 |
| 0 | 0.1 | 0 |

若结果符号反了，最常见原因是把 ascent 写成了 descent、优化器内部已经取过负号却又手动取一次，或把损失最大化和损失最小化的约定混在一起。

### 梯度缓存不是参数

实现中常把梯度保存在参数对象旁边。梯度缓存是本轮计算的导数，不是下一轮应继续累加的更新量。若没有清零，实际使用的是

$$
g_k^{\mathrm{used}}
=g_k+g_{k-1}+\cdots,
$$

而不是当前目标的梯度。某些框架故意允许梯度累加来模拟更大的 batch，但那必须是明确的 micro-batch 协议，不能由忘记清零产生。

更新参数时还要避免把更新操作重新接回自动微分图。数学上的 $\boldsymbol\theta_{k+1}$ 是新迭代的叶节点；实现上的参数更新通常应处在不记录梯度的上下文中。否则下一轮求导可能把优化器状态也当成模型计算图的一部分。

## 目标尺度会改变学习率含义

如果把目标乘以常数 $c>0$：

$$
\widetilde J(\boldsymbol\theta)=cJ(\boldsymbol\theta),
$$

那么

$$
\nabla\widetilde J=c\nabla J.
$$

用同一个学习率更新时：

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-\eta c\nabla J(\boldsymbol\theta_k).
$$

这和原目标使用有效学习率 $\eta_{\mathrm{eff}}=c\eta$ 相同。于是下列改动都可能改变训练动态：

- 把 batch loss 从 mean 改成 sum；
- 把 nats 的交叉熵换成 bits，乘上 $1/\log 2$；
- 改变多标签损失在类别轴上的 sum 或 mean；
- 给样本或类别添加权重；
- 改变正则项相对于数据损失的比例。

它们有时不改变某个理想的最优点，却会改变梯度大小、稳定步长和正则项相对强度。比较两次训练时，不要只记录“学习率是 0.01”；还要记录损失定义、归约分母、有效样本数和梯度范数。

### 参数和特征的单位也会改变曲率

对线性回归的平方损失

$$
J(\boldsymbol w)
=\frac12\lVert X\boldsymbol w-\boldsymbol y\rVert_2^2,
$$

Hessian 是

$$
H=X^{\mathsf T}X.
$$

如果某一列特征的单位从米换成毫米，列会放大 $1000$ 倍，对应的 Hessian 方向会放大约 $10^6$ 倍。一个在原单位下安全的学习率，换单位后可能立刻发散。标准化特征、合理初始化和预条件的作用之一，就是把不同方向的有效曲率拉回相近尺度。

这不是“学习率调小就全部解决”的问题。学习率只能用一个共同标量缩放所有方向；当曲率相差几个数量级时，过小的步长会让低曲率方向慢得不可接受。

## 全批量梯度下降是最清晰的基线

设经验风险为

$$
J(\boldsymbol\theta)
=\frac1N\sum_{i=1}^{N}
\ell_i(\boldsymbol\theta).
$$

全批量梯度下降每一步使用

$$
\nabla J(\boldsymbol\theta)
=\frac1N\sum_{i=1}^{N}\nabla\ell_i(\boldsymbol\theta).
$$

同一个参数快照、同一份完整数据和同一套归约会产生确定的梯度。它的好处是每次更新容易复现、容易做有限差分和目标变化测试；代价是每一步都要遍历完整数据集。

随机梯度和小批次梯度把完整平均换成抽样平均。它们是后续 SGD 词条的主题；在这里先保留一个边界：

| 方面 | 全批量梯度下降 | 小批次/随机梯度 |
| --- | --- | --- |
| 梯度 | 对当前数据集的确定平均 | 对总体梯度的随机估计 |
| 单步损失 | 在固定目标下更容易比较 | 可能因 batch 噪声上下波动 |
| 调试 | 适合逐参数核对 | 需要固定采样、顺序和随机种子 |
| 计算 | 每步成本高，更新次数少 | 单步便宜，更新频率高 |
| 需要的额外协议 | 主要是步长和归约 | 还要记录 batch、采样和方差 |

先用一个很小的数据集跑通全批量基线，可以把数据切分、随机采样和优化噪声从实现问题中移开。基线通过后，再逐项引入小批次、学习率调度或其他优化器。

## 学习率诊断要看三种证据

学习率太小、合适或太大，训练日志通常有不同形状，但不能只靠一条曲线下结论。

| 现象 | 可能含义 | 需要同时检查 |
| --- | --- | --- |
| loss 几乎不动，梯度也小 | 已接近平坦区或更新太小 | 参数移动量、梯度范数、loss 数值精度 |
| loss 缓慢下降，参数移动很小 | 学习率相对曲率太小 | 不同学习率的短跑对照、特征尺度 |
| loss 快速下降后震荡 | 步长接近高曲率稳定边界 | 最大梯度、参数更新比率、验证集 |
| loss 先下降后爆炸 | 步长超过局部稳定范围或出现数值异常 | logits/激活、梯度尖峰、NaN 首发位置 |
| train loss 下降，validation loss 上升 | 泛化或数据边界问题 | 过拟合、切分、正则化，而非只调学习率 |

一次短的学习率扫描比盲目训练几十个 epoch 更容易暴露数量级问题。固定初始化、数据顺序和步数，比较若干候选 $\eta$ 的前几次更新，至少记录：

$$
\frac{\lVert\Delta\boldsymbol\theta_k\rVert_2}
{\max(\lVert\boldsymbol\theta_k\rVert_2,\epsilon)},
\qquad
\lVert\nabla J(\boldsymbol\theta_k)\rVert_2.
$$

更新比率很小不一定坏，参数可能本来就在最优点附近；更新比率突然变大且伴随 loss 上升，则是步长、梯度尖峰或数值异常的强信号。

## 学习率调度改变的是每一轮的步长

固定学习率是最容易解释的基线，但训练中常让 $\eta_k$ 随时间变化：

| 调度 | 形式 | 作用 |
| --- | --- | --- |
| 常数 | $\eta_k=\eta_0$ | 适合作为基线，后期可能在噪声或离散误差附近徘徊 |
| 分段衰减 | 在指定轮次把 $\eta$ 乘以 $\gamma$ | 让前期快速移动，后期细化 |
| 指数衰减 | $\eta_k=\eta_0\gamma^k$ | 连续缩小步长，但可能过早变得很小 |
| 余弦衰减 | 随训练进度平滑下降 | 减少分段跳变，需预先知道总步数 |
| 预热后衰减 | 先从小值升到目标，再衰减 | 缓解初始化或大 batch 训练早期的巨大更新 |

调度器不是“免费提升”。它把学习率变成训练状态的一部分，因此恢复 checkpoint 时至少要一起恢复当前步数、调度器状态和优化器状态。只保存参数而忘记恢复 $\eta_k$，可能让恢复后的曲线和原训练不连续。

调度器还会改变不同阶段的证据含义：前期 loss 下降慢可能是预热，后期移动变小可能是衰减，而不是模型突然学不会。日志应同时记录当前学习率，而不是只记录 epoch。

## 停止、验证和 checkpoint

梯度下降没有一个只看参数移动量就能证明“已经学完”的通用停止条件。常见信号测量的是不同对象：

| 停止信号 | 实际测量 | 容易误解的地方 |
| --- | --- | --- |
| $\lVert\nabla J\rVert_2\leq\varepsilon$ | 当前目标的梯度大小 | 平坦点、鞍点或数值精度也会让梯度小 |
| $\lvert J_{k+1}-J_k\rvert\leq\varepsilon$ | 一步目标变化 | 学习率很小或曲率很低时会提前触发 |
| $\lVert\boldsymbol\theta_{k+1}-\boldsymbol\theta_k\rVert_2\leq\varepsilon$ | 参数移动长度 | 它等于 $\eta_k\lVert\nabla J\rVert$，受学习率直接影响 |
| validation loss 连续若干轮无改善 | 泛化目标是否改善 | 需要固定验证集和耐心值，不能窥视测试集 |
| 达到最大步数 | 计算预算 | 不是优化收敛或泛化的证明 |

实践中可以组合使用：训练目标和梯度作为优化诊断，验证集作为模型选择，最大步数作为资源上限。测试集只在模型、阈值和停止规则封存后使用。

每次保存 checkpoint 时应至少记录：

1. 参数 $\boldsymbol\theta$；
2. 当前步数、epoch、batch 顺序和随机状态；
3. 当前学习率或调度器状态；
4. 训练/验证损失及其归约分母；
5. 最佳验证指标与对应参数快照。

如果只保存“最后一轮”而不保存最佳验证快照，后续可能无法恢复已经出现过的最佳泛化状态。反过来，如果按测试集挑 checkpoint，测试集就不再是独立证据。

## 约束参数时要投影，而不是假装无约束

如果参数必须属于可行集 $\mathcal C$，普通梯度下降的一步可能把它带出可行域。投影梯度下降写成

$$
\boldsymbol\theta_{k+1}
=\Pi_{\mathcal C}
\left(
\boldsymbol\theta_k-\eta_k\nabla J(\boldsymbol\theta_k)
\right),
$$

其中 $\Pi_{\mathcal C}$ 把点映回可行集。对非负参数，投影可以是逐坐标取 $\max(0,\theta_i)$；对一个欧氏球，投影是超出半径时按比例缩回。

把参数更新后直接裁剪到范围内也可能是一种投影，但它改变了算法，不能再把它的轨迹解释成无约束梯度下降。若约束来自概率和、权重范数或物理范围，应该把投影、重新参数化和惩罚项三种做法区分记录。

权重衰减也不是投影。目标

$$
J_{\mathrm{reg}}(\boldsymbol\theta)
=J(\boldsymbol\theta)
+\frac\lambda2\lVert\boldsymbol\theta\rVert_2^2
$$

的梯度是

$$
\nabla J_{\mathrm{reg}}
=\nabla J+\lambda\boldsymbol\theta.
$$

它允许参数保持在整个欧氏空间，却额外惩罚大范数；投影则每一步都把参数限制在某个集合内。两者的最优点和训练轨迹都可能不同。

## 运行方法

下面用纯标准库实现二维二次目标：

$$
J(x,y)=\frac12(4x^2+y^2).
$$

它的两个曲率是 $4$ 和 $1$，因此最陡方向的稳定边界是 $\eta<2/4=0.5$。从 $(2,2)$ 出发，分别测试稳定区间内接近边界的步长、边界步长和超过边界的步长：

```python
def objective(theta):
    x, y = theta
    return 0.5 * (4.0 * x * x + y * y)


def gradient(theta):
    x, y = theta
    return [4.0 * x, y]


def rounded(values):
    return [round(value, 12) for value in values]


def run(eta, steps=5):
    theta = [2.0, 2.0]
    history = []
    for _ in range(steps):
        history.append(objective(theta))
        grad = gradient(theta)
        theta = [
            value - eta * direction
            for value, direction in zip(theta, grad)
        ]
    return theta, objective(theta), history


for eta in (0.20, 0.49, 0.50, 0.60):
    theta, loss, history = run(eta)
    print(
        f"eta={eta:.2f} theta={rounded(theta)} "
        f"loss={loss:.12f} "
        f"first={history[0]:.6f} last={history[-1]:.6f}"
    )
```

输出为：

```text
eta=0.20 theta=[0.00064, 0.65536] loss=0.214749184000 first=10.000000 last=0.335565
eta=0.49 theta=[-1.6307453952, 0.0690050502] loss=5.321041936409 first=10.000000 last=5.780270
eta=0.50 theta=[-2.0, 0.0625] loss=8.001953125000 first=10.000000 last=8.007812
eta=0.60 theta=[-10.75648, 0.02048] loss=231.403933696000 first=10.000000 last=118.064435
```

$\eta=0.20$ 同时缩小两个方向，最终 loss 明显下降；$\eta=0.49$ 虽然仍在理论稳定区间，却在高曲率方向来回摆动，五步后下降很慢；$\eta=0.50$ 处于最陡方向的边界，$x$ 坐标等幅翻转，整体目标由低曲率方向略微下降；$\eta=0.60$ 让高曲率方向的误差因子变成 $-1.4$，因此目标迅速变大。

这个例子有一个重要的审计优点：目标、梯度、更新和稳定边界都可以手算。真实神经网络的 loss landscape 不会这么规整，但一维或二维二次测试仍然可以抓出更新符号、学习率乘法、参数顺序和梯度缓存方面的实现错误。

## 失效模式

**把梯度方向当成更新方向。** 梯度指向局部上坡，最小化目标应使用 $\boldsymbol\theta-\eta\nabla J$。先用一个正梯度的标量单元测试验证更新量为负。

**学习率超过局部稳定范围。** 大步长可能先下降几轮再爆炸，也可能从第一步就出现 NaN。保留第一个异常 batch 的参数、梯度和激活，不要只看训练结束时的 NaN。

**把每个坐标都用同一个“合适步长”理解。** 多维曲率不同，一个标量学习率必须服从最大曲率方向的稳定要求，低曲率方向可能因此很慢。检查特征尺度、参数化和预条件。

**sum/mean 或权重变了但学习率没变。** 目标的常数缩放会等价地缩放梯度。记录归约分母和有效样本数，再比较学习率。

**忘记清理梯度缓存。** 这会把多个 batch 的梯度叠加，可能是有意的梯度累积，也可能是一个静默 bug。把每轮清零和累积步数写入训练协议。

**在参数更新时继续记录自动微分图。** 这样会让优化器状态混入下一轮计算图，增加内存并改变梯度语义。参数更新应与模型前向图分离。

**只因为训练 loss 下降就停止。** 下降只说明当前目标变小，不说明验证性能、概率校准或部署代价改善。保留验证集和最佳 checkpoint。

**把梯度很小解释成全局最优。** 平坦区域、鞍点、饱和激活、数值下溢和过小学习率都可能产生小梯度。结合目标曲率、参数移动和多个初始化判断。

**把裁剪、权重衰减和投影混成一个算法。** 它们作用在不同对象上：梯度、目标或可行域。改变其中任何一项都应在实验记录中明确写出。

**把全批量基线和小批次曲线直接比较。** 小批次引入了梯度噪声和不同的更新频率。比较时要固定数据规模、总样本访问量、学习率定义和归约口径。

## 一个可复用的核验协议

实现或更换梯度下降时，可以按下面顺序检查：

1. 用标量 $J(\theta)=\frac12a(\theta-\theta^\star)^2$ 核对梯度和更新符号；
2. 对同一个参数快照，比较解析梯度与中心差分；
3. 打印更新前后目标、梯度范数、参数移动量和当前学习率；
4. 在一个二维正定二次目标上测试稳定、边界和发散步长；
5. 改变 sum/mean、样本权重和 mask，确认学习率尺度随目标变化；
6. 固定全批量数据顺序，确认同一快照重复运行得到相同梯度；
7. 检查梯度缓存是否清零，若要累积则检查累积步数与分母；
8. 若有调度器，保存并恢复步数、当前学习率和 checkpoint；
9. 若有投影或权重衰减，单独记录它们发生在梯度、目标还是参数域；
10. 用验证集选择停止点，用封存测试集做最终报告。

梯度下降本身只是一个更新规则。它能否把模型训练好，取决于梯度是否对应正确目标、步长是否适合当前尺度、数据访问是否符合假设，以及你是否把训练曲线之外的证据也保存下来。

## 相关词条

- [损失函数](../training-nn/loss-functions/)：区分逐样本损失、训练目标、指标和部署代价。
- [分类损失](../training-nn/classification-losses/)：查看分类输出、损失和梯度的配套关系。
- [梯度](../calculus/gradient/)：从方向导数和线性近似理解梯度向量。
- [Hessian](../calculus/hessian/)：理解曲率、稳定步长和二阶项。
- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：在光滑性、凸性和强凸性假设下推导收敛界。
- [小批量随机梯度下降](../training-nn/minibatch-sgd/)：把全批量梯度换成带抽样噪声的更新。
- [动量与 Nesterov](../training-nn/momentum-and-nesterov/)：在梯度下降上加入历史方向和前瞻修正。
- [学习率调度](../training-nn/learning-rate-schedules/)：系统比较预热、衰减和周期性步长。
- [约束优化](../optimization-theory/constrained-optimization/)：进一步分析可行域、投影和拉格朗日方法。
