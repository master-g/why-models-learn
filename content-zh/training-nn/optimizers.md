---
title: "优化器：把梯度变成可审计的参数更新"
tags: ["why-models-learn"]
---

优化器接收当前梯度、参数和一组内部状态，决定本次参数如何移动以及下一步要保存什么。SGD 只保存学习率语义，动量多保存历史方向，自适应方法还保存逐坐标的平方梯度尺度；它们都可能在同一个 loss 上产生不同的训练轨迹。选择优化器不能只看名字或默认参数，还要把 batch 归约、梯度累积、权重衰减、裁剪、学习率调度和 checkpoint 状态写成同一个更新协议。本篇先建立这个共同接口，再比较常见家族、做一组可运行的标准库对照，最后给出公平比较与排查训练异常的记录方法。

![优化器把同一个梯度分成不同状态路径：SGD 直接更新，动量累积方向，自适应方法维护坐标尺度，Adam 同时维护方向和尺度](/assets/training-nn/svg/optimizers.1.svg)

## 优化器不是一个乘学习率的函数

设第 $k$ 次参数更新前有参数 $\boldsymbol\theta_k$、有效梯度 $\boldsymbol g_k$ 和优化器状态 $\boldsymbol s_k$。一个一般的优化器可以抽象为

$$
\begin{aligned}
(\boldsymbol u_k,\boldsymbol s_{k+1})
&=\mathcal O(
\boldsymbol\theta_k,
\boldsymbol g_k,
\boldsymbol s_k,
\eta_k,
\boldsymbol h),\\
\boldsymbol\theta_{k+1}
&=\boldsymbol\theta_k+\boldsymbol u_k,
\end{aligned}
$$

其中 $\boldsymbol u_k$ 是实际参数位移，$\boldsymbol h$ 是超参数集合。优化器的状态可能包括：

- 动量或速度缓冲区；
- 平方梯度的累积量或指数平均；
- Adam 的一阶、二阶矩；
- 当前更新次数，用于偏置修正；
- 参数组自己的学习率、衰减和统计量。

普通梯度下降是状态最少的特例：

$$
\boldsymbol u_k=-\eta_k\boldsymbol g_k,
\qquad
\boldsymbol s_k=\varnothing.
$$

因此一个训练 step 至少有两个层次：

1. 反向传播得到梯度；
2. 优化器根据梯度和历史状态决定位移。

如果梯度本身已经错误，换优化器不会修复计算图；如果梯度正确但学习率、状态恢复或归约分母错误，单看反向传播也找不到原因。

## 先固定有效梯度

优化器看到的梯度不一定是某一个样本的梯度。对一个有效 batch，若样本权重为 $w_i$、逐样本梯度为 $\boldsymbol g_i$，通常先形成

$$
\boldsymbol g_k
=\frac{\sum_{i\in\mathcal B}w_i\boldsymbol g_i}
{\sum_{i\in\mathcal B}w_i}.
$$

如果使用 $R$ 个 micro-batch 做梯度累积，且第 $r$ 个 micro-batch 的有效样本数为 $n_r$，应当先得到

$$
\overline{\boldsymbol g}_k
=\frac{\sum_{r=1}^{R}n_r\boldsymbol g_{k,r}}
{\sum_{r=1}^{R}n_r},
$$

然后让优化器推进一次状态：

$$
(\boldsymbol u_k,\boldsymbol s_{k+1})
=\mathcal O(
\boldsymbol\theta_k,
\overline{\boldsymbol g}_k,
\boldsymbol s_k,\eta_k,\boldsymbol h).
$$

如果每个 micro-batch 都更新一次动量或平方梯度统计量，实际执行的就不是一个大 batch，而是多个带状态的小 batch。有效 batch、噪声尺度、状态衰减和学习率含义都会改变。

混合精度还多一个边界。若反向传播使用 loss scale $a$，动量和自适应统计量应看到

$$
\boldsymbol g_k
=\frac{\boldsymbol g_k^{\mathrm{scaled}}}{a},
$$

而不是把被放大的梯度写进状态。常见顺序是 unscale、检查非有限值、梯度裁剪、归约确认，再进入优化器。

## 五类更新共享一张地图

常见训练优化器可以按保存的状态和改变的尺度分成几类：

| 家族 | 保存的状态 | 主要改变 | 典型使用语境 |
| --- | --- | --- | --- |
| SGD | 无 | 全局学习率 | 基线、可解释轨迹、强正则化实验 |
| Momentum / heavy-ball | 一阶历史方向 | 时间方向平滑 | 狭窄谷底、连续的梯度方向 |
| AdaGrad | 累积平方梯度 | 坐标步长持续变小 | 稀疏特征、稀疏更新 |
| RMSProp | 平方梯度指数平均 | 近期坐标尺度 | 非平稳梯度尺度 |
| Adam | 一阶平均 + 二阶平均 | 方向与尺度同时调节 | 深层网络的常用起点 |
| 二阶或拟二阶 | 曲率或曲率近似 | 坐标之间的耦合 | 小到中等规模、曲率信息值得付出的任务 |

这张表只描述机制，不给出无条件的优劣排序。比如 Adam 的状态更多、早期适应更快，但它和 SGD 的参数路径、噪声过滤以及隐式偏好不同；换优化器就相当于换了学习算法的一部分。

## SGD 是不可省略的基线

普通 SGD 的更新是

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-\eta_k\boldsymbol g_k.
$$

如果 $\boldsymbol g_k$ 是小批次无偏梯度，SGD 的噪声只来自数据抽样和其他随机训练机制。它的好处是状态少、尺度清楚、checkpoint 简单。缺点是同一个全局学习率必须同时适应所有参数和所有曲率方向。

在固定一维二次目标

$$
f(x)=\frac12\lambda x^2
$$

上，更新变成

$$
x_{k+1}
=(1-\eta\lambda)x_k.
$$

收敛需要

$$
0<\eta\lambda<2.
$$

这个小例子是非常有用的单元测试：当 $\eta\lambda=1$ 时下一步到达零；当 $\eta\lambda$ 接近 $2$ 时会交替振荡；当 $\eta\lambda>2$ 时误差放大。任何复杂优化器都应该先在这个基线上检查参数、梯度符号和 loss 计算。

SGD 还常常有较明确的隐式正则化路径。对比实验如果关心泛化，而不只是训练 loss，应该保留 SGD 作为参照，不要只比较哪个优化器更快把训练误差压低。

## 动量是在时间轴上做预条件

重球动量保存历史方向。用梯度速度 $\boldsymbol v_k$ 表示：

$$
\boldsymbol v_{k+1}
=\beta\boldsymbol v_k+\boldsymbol g_k,
\qquad
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-\eta_k\boldsymbol v_{k+1}.
$$

如果用参数位移 $\boldsymbol d_k$ 表示，则是

$$
\boldsymbol d_{k+1}
=\beta\boldsymbol d_k-\eta_k\boldsymbol g_k,
\qquad
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k+\boldsymbol d_{k+1}.
$$

动量没有按坐标除以梯度尺度；它是在时间轴上累积方向。历史权重为 $\beta^j$，有效记忆长度的数量级约为

$$
\frac1{1-\beta}.
$$

这能减少方向相近时的抖动，也会把错误方向和旧 batch 的噪声保留更久。动量系数变大时，通常要重新观察基础学习率和更新比率。动量篇已经用二维二次目标推导了特征根和 Nesterov 前瞻；这里把它视为优化器接口中的一种状态实现。

## AdaGrad、RMSProp 与 Adam 改变的是坐标尺度

自适应方法维护逐坐标的正数分母 $d_{k,i}$：

$$
\theta_{k+1,i}
=\theta_{k,i}
-\eta_k\frac{g_{k,i}}{d_{k,i}}.
$$

AdaGrad 使用永久累积的平方梯度：

$$
G_{k,i}
=G_{k-1,i}+g_{k,i}^2,
\qquad
d_{k,i}=\sqrt{G_{k,i}}+\epsilon.
$$

平方梯度大的坐标会越来越保守，长期不出现的稀疏坐标保留较大的相对步长。由于 $G_{k,i}$ 单调不减，AdaGrad 后期可能过于保守。

RMSProp 用指数平均忘掉较早尺度：

$$
S_{k,i}
=\rho S_{k-1,i}
+(1-\rho)g_{k,i}^2,
\qquad
d_{k,i}=\sqrt{S_{k,i}}+\epsilon.
$$

$\rho$ 越接近 $1$，尺度记忆越长；$S_{k,i}$ 可以下降，因此有效步长也可能重新变大。

Adam 额外保存一阶和二阶平均：

$$
\boldsymbol m_k
=\beta_1\boldsymbol m_{k-1}
+(1-\beta_1)\boldsymbol g_k,
$$

$$
\boldsymbol v_k
=\beta_2\boldsymbol v_{k-1}
+(1-\beta_2)\boldsymbol g_k^{\odot2}.
$$

从零初始化时，前几步需要偏置修正：

$$
\widehat{\boldsymbol m}_k
=\frac{\boldsymbol m_k}{1-\beta_1^k},
\qquad
\widehat{\boldsymbol v}_k
=\frac{\boldsymbol v_k}{1-\beta_2^k},
$$

更新为

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k
-\eta_k
\frac{\widehat{\boldsymbol m}_k}
{\sqrt{\widehat{\boldsymbol v}_k}+\epsilon}.
$$

这里的二阶量是逐坐标的梯度历史统计，不是 Hessian，也没有给出坐标之间的旋转耦合。自适应学习率篇已经展开 AdaGrad、RMSProp、Adam 的偏置修正、尺度不变性和理论边界；下一篇 Adam 再单独处理实现细节与训练协议。

## 同一个学习率不代表同一种步长

把同一个 $\eta$ 填进不同优化器，并不能构成公平的最终比较。对 SGD，$\eta$ 直接乘梯度；对动量，它还会乘上历史方向；对 Adam，当前坐标的梯度幅度被分母部分抵消。相同的字符串数值，实际更新范数可能差很多。

可以把每步真实移动量写成

$$
\Delta_k
=\boldsymbol\theta_{k+1}-\boldsymbol\theta_k,
\qquad
r_k
=\frac{\lVert\Delta_k\rVert_2}
{\max(\lVert\boldsymbol\theta_k\rVert_2,\varepsilon)}.
$$

比较优化器时，至少报告：

| 记录 | 为什么需要 |
| --- | --- |
| 基础学习率与调度器 | 说明标量尺度怎样随时间变化 |
| 实际更新范数 | 观察不同状态规则最后走了多远 |
| 梯度范数 | 区分梯度信号与优化器缩放 |
| 分母或缓冲区范数 | 解释坐标步长或历史方向 |
| optimizer step 数 | 对齐训练预算 |
| 样本访问量 | batch size 改变后仍能比较数据预算 |
| 验证曲线 | 防止只按训练 loss 选择算法 |

如果只记录 loss，常常无法判断“Adam 更快”是因为方向更好、实际步长更大，还是它在早期把每个坐标都归一化成了相近的幅度。

## 一个标准库对照实验

取同一个二维二次目标：

$$
F(\boldsymbol\theta)
=\frac12\boldsymbol\theta^{\mathsf T}
\begin{bmatrix}
1&0\\
0&9
\end{bmatrix}
\boldsymbol\theta,
\qquad
\boldsymbol\theta_0=(1,1)^{\mathsf T}.
$$

下面的代码实现 SGD、重球动量、AdaGrad、RMSProp 和带偏置修正的 Adam。所有方法使用 $\eta=0.05$，这不是声称该学习率对它们都最优，而是先展示“同一个数字在不同状态方程中意味着什么”。

```python
import math


Q = (1.0, 9.0)


def loss(theta):
    return 0.5 * sum(q * x * x for q, x in zip(Q, theta))


def grad(theta):
    return [q * x for q, x in zip(Q, theta)]


def run(
    name,
    steps=30,
    eta=0.05,
    beta=0.9,
    rho=0.9,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
):
    theta = [1.0, 1.0]
    momentum = [0.0, 0.0]
    square = [0.0, 0.0]
    first = [0.0, 0.0]
    second = [0.0, 0.0]
    history = []

    for step in range(1, steps + 1):
        current_grad = grad(theta)

        if name == "sgd":
            direction = current_grad

        elif name == "momentum":
            momentum = [
                beta * old + current
                for old, current in zip(momentum, current_grad)
            ]
            direction = momentum

        elif name == "adagrad":
            square = [
                old + current * current
                for old, current in zip(square, current_grad)
            ]
            direction = [
                current / (math.sqrt(old) + eps)
                for current, old in zip(current_grad, square)
            ]

        elif name == "rmsprop":
            square = [
                rho * old + (1.0 - rho) * current * current
                for old, current in zip(square, current_grad)
            ]
            direction = [
                current / (math.sqrt(old) + eps)
                for current, old in zip(current_grad, square)
            ]

        elif name == "adam":
            first = [
                beta1 * old + (1.0 - beta1) * current
                for old, current in zip(first, current_grad)
            ]
            second = [
                beta2 * old + (1.0 - beta2) * current * current
                for old, current in zip(second, current_grad)
            ]
            corrected_first = [
                value / (1.0 - beta1**step)
                for value in first
            ]
            corrected_second = [
                value / (1.0 - beta2**step)
                for value in second
            ]
            direction = [
                current / (math.sqrt(old) + eps)
                for current, old in zip(corrected_first, corrected_second)
            ]

        else:
            raise ValueError(name)

        theta = [
            current - eta * direction_value
            for current, direction_value in zip(theta, direction)
        ]
        history.append((step, theta[:], loss(theta)))

    return history


if __name__ == "__main__":
    checkpoints = (1, 2, 5, 10, 20, 30)
    for name in ("sgd", "momentum", "adagrad", "rmsprop", "adam"):
        history = run(name)
        values = [round(history[step - 1][2], 6) for step in checkpoints]
        print(name, "losses=", values, "theta30=",
              tuple(round(x, 6) for x in history[-1][1]))
```

输出为：

```text
sgd losses= [1.8125, 0.819031, 0.310767, 0.179272, 0.064256, 0.023035] theta30= (0.214639, 0.0)
momentum losses= [1.8125, 0.414931, 1.931342, 0.536021, 0.022509, 0.143486] theta30= (0.197061, -0.166045)
adagrad losses= [4.5125, 4.191274, 3.556456, 2.903142, 2.120579, 1.63054] theta30= (0.571059, 0.571059)
rmsprop losses= [3.543861, 2.715386, 1.375311, 0.454205, 0.024875, 0.000171] theta30= (0.005844, 0.005844)
adam losses= [4.5125, 4.050749, 2.823595, 1.312223, 0.061856, 0.052727] theta30= (-0.10269, -0.10269)
```

这里重球法在第 $5$ 步的 loss 又上升，说明动量轨迹可以穿过谷底；RMSProp 在这个固定目标上下降最快，AdaGrad 因为持续累积早期尺度而更保守。Adam 在第 $30$ 步还没有达到 RMSProp 的数值。这个结果不是排行榜：每种方法的 $\eta$、$\beta$、$\rho$、$\epsilon$ 都可以重新调，真实网络还会加入 batch 噪声、非线性曲率和正则化。实验的作用是让状态方程产生可检查的差异，而不是用一个二维目标决定工程选择。

## 权重衰减、裁剪和状态的顺序

优化器前的梯度处理要写清楚。若采用耦合的 L2 项，进入状态的是

$$
\widetilde{\boldsymbol g}_k
=\boldsymbol g_k+\lambda\boldsymbol\theta_k.
$$

如果采用解耦权重衰减，则参数收缩不进入动量或自适应分母：

$$
\boldsymbol\theta_{k+1}
=(1-\eta_k\lambda)\boldsymbol\theta_k
+\boldsymbol u_k^{\mathrm{gradient}}.
$$

二者在自适应方法下通常不等价。梯度裁剪也应在 unscale 之后、状态更新之前完成。若全局范数超过 $c$，可写成

$$
\widetilde{\boldsymbol g}_k
=\frac{c}{\lVert\boldsymbol g_k\rVert_2}
\boldsymbol g_k.
$$

一个常见的可审计顺序是：

$$
\text{unscale}
\longrightarrow
\text{finite check}
\longrightarrow
\text{clip}
\longrightarrow
\text{weight decay policy}
\longrightarrow
\text{optimizer state}
\longrightarrow
\text{parameter update}.
$$

项目也可以选择把耦合衰减放在裁剪之前或之后，但需要在实验记录中固定它。不同顺序会改变进入状态的向量，不能在比较中悄悄切换。

## 参数组和 checkpoint

不同参数组可以有不同的基础学习率、权重衰减和状态。第 $j$ 个参数组的状态应当只对应它自己的参数：

$$
\boldsymbol s_{k+1}^{(j)}
=\Phi_j\left(
\boldsymbol s_k^{(j)},
\boldsymbol g_k^{(j)}
\right),
\qquad
\boldsymbol\theta_{k+1}^{(j)}
=\boldsymbol\theta_k^{(j)}
+\boldsymbol u_k^{(j)}.
$$

恢复训练至少要保存：

- 参数张量；
- 每个参数组的 optimizer state；
- 当前 optimizer step；
- 学习率调度器的状态；
- 梯度累积计数和有效样本数；
- sampler、shuffle seed 或随机数状态；
- 混合精度 scaler；
- 最佳验证指标与对应 checkpoint 元数据。

只保存模型参数会把 SGD 之外的优化器都截断成一个新状态。Adam 的一阶、二阶矩丢失后，恢复点的第一个更新可能看起来没问题，后面的有效步长却已经不同；动量缓冲区丢失也会造成同样的跳变。

## 怎样公平比较优化器

一个有用的比较协议分成三层：

**先测实现。** 在一维和二维二次目标上固定初值，逐步打印参数、梯度、状态和 loss。用已知的稳定边界检查符号、偏置修正、状态初始化和第一步尺度。

**再冻结训练预算。** 同时固定数据划分、随机种子、总样本访问量、optimizer step 数、评估频率和停止规则。改 batch size 时，单独记录每 epoch 的 step 变化。

**最后调各自的验证超参数。** 在同一搜索预算下为每种方法选择学习率、动量系数、二阶记忆和衰减强度。用验证集选择，不要先看测试集再回头改配置。

报告不应只有“最终验证准确率”。至少应带上：

| 项目 | 最小记录 |
| --- | --- |
| 优化器定义 | 公式、库版本、Nesterov 和 weight decay 语义 |
| 基础学习率 | 初值、调度器、按 epoch 还是 step 推进 |
| 状态超参数 | $\beta$、$\rho$、$\beta_1$、$\beta_2$、$\epsilon$ |
| 训练预算 | optimizer step、样本访问量、有效 batch |
| 数值保护 | unscale、clip 阈值、非有限梯度处理 |
| 结果轨迹 | 训练 loss、验证 loss、更新比率和状态范数 |
| 恢复证据 | checkpoint 是否含 optimizer state 和 sampler state |

这样才能区分“算法更适合”与“它实际看到了更多样本、使用了更大的更新或恢复时保留了更多状态”。

## 常见失效模式

**只换优化器名，不重调学习率。** $\eta$ 在 SGD、动量和 Adam 中的有效含义不同。先做短程 sweep，再比较长程曲线。

**把每个 micro-batch 都当成 optimizer step。** 梯度累积时这会改变状态衰减、更新次数和学习率调度。先定义有效 batch，再推进一次优化器。

**忘记 Adam 的偏置修正。** 从零初始化的矩在早期偏小，直接使用会改变前几步的分母和有效方向。

**把平方梯度当作 Hessian。** 逐坐标统计不能表达参数之间的曲率耦合，也不能保证得到安全的 Newton 步。

**把 L2 项和解耦 weight decay 混为一谈。** 自适应分母或动量存在时，两种规则的参数轨迹不同。

**只看训练 loss 下降速度。** 优化器的路径本身可能改变隐式正则化和泛化。应在固定验证协议下比较，并记录更新范数。

**恢复时只加载模型权重。** 丢失动量、矩、step、scheduler 或 sampler 状态后，恢复训练不是原实验的继续。

**把某个二维目标的胜负推广到所有网络。** 二次目标可以暴露状态方程错误，却不能代替真实数据、模型和验证集上的选择。

## 运行方法

将上面的标准库代码保存为 optimizers.py，运行：

```bash
python3 optimizers.py
```

若要替换为真实框架实现，先让框架版本输出与标准库短轨迹的趋势一致，再逐项打开 batch 噪声、梯度累积、权重衰减、混合精度和学习率调度。每次只改变一个状态来源，才能定位差异来自哪里。

## 相关词条

- [梯度下降](../training-nn/gradient-descent/)：建立没有内部状态的全局学习率基线。
- [小批量随机梯度下降](../training-nn/minibatch-sgd/)：说明优化器实际接收到的 batch 梯度如何形成。
- [动量与 Nesterov](../training-nn/momentum-and-nesterov/)：处理历史方向、前瞻点与缓冲区恢复。
- [自适应学习率](../optimization-theory/adaptive-learning-rates/)：推导 AdaGrad、RMSProp、Adam 的坐标尺度与理论边界。
- [Adam](../training-nn/adam/)：继续展开 Adam 的实现选项、偏置修正和训练诊断。
- [学习率调度](../training-nn/learning-rate-schedules/)：固定优化器后安排 warmup、衰减和 step 预算。
- [梯度裁剪](../training-nn/gradient-clipping/)：限制进入优化器状态的异常梯度。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：区分显式正则项、权重衰减和优化路径偏好。
- [隐式正则化](../evaluation-and-generalization/implicit-regularization/)：分析优化器、参数化和训练时间共同产生的偏好。
