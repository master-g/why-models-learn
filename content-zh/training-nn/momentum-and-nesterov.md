---
title: "动量与 Nesterov：把历史方向接入训练循环"
tags: ["why-models-learn"]
---

动量方法在普通梯度下降的参数更新之外保存一个状态，把连续几步方向相近的梯度累积成更长的位移；Nesterov 方法进一步在“走出去之后的位置”计算梯度，再决定这一步怎么修正。它们不是给学习率加一个神秘倍数，而是改变了优化器的状态机：每个参数组都多了缓冲区，梯度累积、权重衰减、梯度裁剪、学习率调度和 checkpoint 都必须说明这个缓冲区在什么时候读写。本篇从可执行的更新顺序出发，对齐重球法、lookahead Nesterov 与常见库式写法，再用二维二次目标和训练审计信号说明怎样选参数、排查振荡以及恢复状态。

![动量与 Nesterov 的训练状态：左侧比较梯度下降、重球法与前瞻更新在狭窄谷底中的轨迹，右侧展示梯度、动量缓冲区和参数更新的顺序](/assets/training-nn/svg/momentum-and-nesterov.1.svg)

## 先把优化器写成状态机

设当前参数为 $\boldsymbol\theta_k$，当前批次产生的梯度为

$$
\boldsymbol g_k
=\nabla_{\boldsymbol\theta}L_k(\boldsymbol\theta_k).
$$

普通梯度下降只有参数状态：

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-\eta_k\boldsymbol g_k.
$$

动量方法还保存一个位移缓冲区 $\boldsymbol d_k$。采用参数位移记号时，重球法写成

$$
\boldsymbol d_{k+1}
=\beta\boldsymbol d_k-\eta_k\boldsymbol g_k,
\qquad
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k+\boldsymbol d_{k+1},
$$

其中 $\beta$ 是动量系数，通常满足 $0\leq\beta<1$。$\boldsymbol d_k$ 的单位和参数相同，所以它可以直接加到参数上。初始时若 $\boldsymbol d_0=\boldsymbol0$，第一步恰好退化成普通梯度下降：

$$
\boldsymbol d_1=-\eta_0\boldsymbol g_0,
\qquad
\boldsymbol\theta_1
=\boldsymbol\theta_0-\eta_0\boldsymbol g_0.
$$

Nesterov 的 lookahead 形式先用旧位移预测梯度评估点：

$$
\boldsymbol y_k
=\boldsymbol\theta_k+\beta\boldsymbol d_k,
$$

然后在 $\boldsymbol y_k$ 计算梯度：

$$
\boldsymbol g_k^{\mathrm{look}}
=\nabla L_k(\boldsymbol y_k),
$$

最后用这个前瞻点的梯度更新位移和参数：

$$
\boldsymbol d_{k+1}
=\beta\boldsymbol d_k-\eta_k\boldsymbol g_k^{\mathrm{look}},
\qquad
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k+\boldsymbol d_{k+1}.
$$

三种状态机的差异可以压缩成下面这张表：

| 方法 | 额外状态 | 梯度评估点 | 参数更新 |
| --- | --- | --- | --- |
| 梯度下降 | 无 | $\boldsymbol\theta_k$ | $-\eta_k\boldsymbol g_k$ |
| 重球动量 | $\boldsymbol d_k$ | $\boldsymbol\theta_k$ | $\boldsymbol d_{k+1}$ |
| lookahead Nesterov | $\boldsymbol d_k$ | $\boldsymbol y_k=\boldsymbol\theta_k+\beta\boldsymbol d_k$ | $\boldsymbol d_{k+1}$ |

“有动量”至少意味着这四个动作按固定顺序发生：读取旧缓冲区、计算或取得梯度、写入新缓冲区、更新参数。只把当前梯度乘上 $1+\beta$，却没有保存跨 step 的状态，并没有实现动量。

## 速度记号和位移记号如何互换

另一种常见写法把历史梯度而不是参数位移放进缓冲区：

$$
\boldsymbol v_{k+1}
=\beta\boldsymbol v_k+\boldsymbol g_k,
\qquad
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-\eta_k\boldsymbol v_{k+1}.
$$

当学习率 $\eta$ 固定时，两个写法只差一个尺度：

$$
\boldsymbol d_k=-\eta\boldsymbol v_k.
$$

将速度写法代回参数更新：

$$
\begin{aligned}
\boldsymbol d_{k+1}
&=-\eta\boldsymbol v_{k+1}\\
&=-\eta\left(\beta\boldsymbol v_k+\boldsymbol g_k\right)\\
&=\beta\boldsymbol d_k-\eta\boldsymbol g_k.
\end{aligned}
$$

这说明“速度”并不必然表示物理速度；有的库把未乘学习率的梯度累积称为 velocity，有的库把真正的参数位移称为 momentum buffer。审查 checkpoint 或迁移实现时，不能只看字段名，要看它最终乘没乘学习率。

如果学习率会变化，尺度差异会影响实现。速度缓冲区保存的是梯度单位，改变 $\eta_k$ 只改变本步的参数位移；位移缓冲区保存的是参数单位，旧位移本身已经包含过去的学习率。两者在固定学习率下等价，在学习率调度中需要单独确认：

| 缓冲区定义 | 保存的量 | 调整学习率后的含义 | 恢复时要核对 |
| --- | --- | --- | --- |
| 梯度速度 $\boldsymbol v$ | 历史梯度加权和 | 新 $\eta_k$ 只作用于当前更新 | $\boldsymbol v$ 与参数维度、梯度尺度 |
| 参数位移 $\boldsymbol d$ | 历史参数移动量 | 旧学习率已经留在缓冲区 | $\boldsymbol d$ 与当前参数单位 |
| 归一化平均 $\boldsymbol m$ | 历史梯度的指数平均 | 新梯度常乘 $1-\beta$ | 是否另有偏置修正或缩放 |

例如，用 $\boldsymbol v$ 写成

$$
\boldsymbol v_{k+1}
=\beta\boldsymbol v_k+\boldsymbol g_k
$$

时，展开前几项得到

$$
\boldsymbol v_{k+1}
=\boldsymbol g_k
+\beta\boldsymbol g_{k-1}
+\beta^2\boldsymbol g_{k-2}
+\cdots.
$$

动量系数 $\beta=0.9$ 时，旧梯度权重按 $1,0.9,0.81,\ldots$ 衰减；有效记忆长度的数量级约为

$$
H_{\mathrm{eff}}
=\frac1{1-\beta}.
$$

因此 $\beta=0.9$ 对应约 $10$ 步，$\beta=0.99$ 对应约 $100$ 步。这个数不是硬窗口，超过它的梯度仍有非零权重。若关心“旧影响减半要多久”，半衰期是

$$
h_{1/2}
=\frac{\log(1/2)}{\log\beta},
\qquad
0<\beta<1.
$$

$\beta=0.9$ 的半衰期约为 $6.58$ 步，$\beta=0.99$ 约为 $68.97$ 步。把这两个量混成同一个“记忆长度”会误读动量系数。

## Nesterov 不是把动量再乘一次

lookahead 版本先产生预测点：

$$
\boldsymbol y_k
=\boldsymbol\theta_k+\beta\boldsymbol d_k.
$$

如果 $\boldsymbol d_k$ 指向正在下降的方向，$\boldsymbol y_k$ 已经比当前参数更靠前；梯度在这个位置测量，能够提前看到“继续沿旧方向走会不会越过谷底”。

实现时必须区分两件事：

**前瞻位置。** 梯度输入是 $\boldsymbol y_k$，不是旧的 $\boldsymbol\theta_k$。如果模型前向仍然使用旧参数，代码只是重球法。

**前瞻后的位移。** 计算完 $\boldsymbol g_k^{\mathrm{look}}$ 后，缓冲区仍按 $\beta\boldsymbol d_k-\eta_k\boldsymbol g_k^{\mathrm{look}}$ 更新。不能把旧位移、当前梯度和前瞻梯度各加一次。

一些库式 Nesterov 写法先累积梯度速度：

$$
\boldsymbol v_{k+1}
=\beta\boldsymbol v_k+\boldsymbol g_k,
$$

再用

$$
\boldsymbol q_{k+1}
=\boldsymbol g_k+\beta\boldsymbol v_{k+1}
$$

作为参数方向：

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-\eta_k\boldsymbol q_{k+1}.
$$

这和显式把参数移动到 $\boldsymbol y_k$ 后再求 $\nabla L(\boldsymbol y_k)$ 不是逐项相同的算法。前者只在当前参数点计算一次梯度，再用代数修正模拟前瞻；后者需要把模型在前瞻参数处重新前向。两者都可能被称为 Nesterov，参数不能直接互换。

零初始化时，三种写法的第一步尤其能暴露差异：

| 写法 | 第一步的状态 | 第一步参数位移 |
| --- | --- | --- |
| lookahead 位移 | $\boldsymbol d_0=\boldsymbol0$，所以 $\boldsymbol y_0=\boldsymbol\theta_0$ | $-\eta_0\boldsymbol g(\boldsymbol\theta_0)$ |
| 梯度速度重球 | $\boldsymbol v_0=\boldsymbol0$ | $-\eta_0\boldsymbol g(\boldsymbol\theta_0)$ |
| 库式 Nesterov 修正 | 先写入 $\boldsymbol v_1=\boldsymbol g_0$，再形成 $\boldsymbol q_1$ | 可能是 $-\eta_0(1+\beta)\boldsymbol g_0$ |

若两个实现第一步就差了 $1+\beta$，不应先怀疑随机种子；先检查采用的是哪一种 Nesterov 约定、缓冲区是否在第一步参与了修正。

## 二维二次目标上的可运行对照

为了只观察优化器状态，取一个没有数据噪声的二次目标：

$$
F(\boldsymbol\theta)
=\frac12\boldsymbol\theta^{\mathsf T}Q\boldsymbol\theta,
\qquad
Q=
\begin{bmatrix}
1&0\\
0&9
\end{bmatrix},
\qquad
\boldsymbol\theta_0=
\begin{bmatrix}
1\\
1
\end{bmatrix}.
$$

梯度是

$$
\nabla F(\boldsymbol\theta)
=Q\boldsymbol\theta
=\begin{bmatrix}
\theta_1\\
9\theta_2
\end{bmatrix}.
$$

这个目标在第二坐标上的曲率是第一坐标的 $9$ 倍。为了让轨迹明显但仍保持稳定，实验使用 $\eta=0.1$、$\beta=0.5$，运行普通梯度下降、重球法和显式 lookahead Nesterov。代码只依赖 Python 标准库：

```python
Q = (1.0, 9.0)


def loss(theta):
    return 0.5 * sum(q * x * x for q, x in zip(Q, theta))


def grad(theta):
    return [q * x for q, x in zip(Q, theta)]


def run(kind, eta=0.1, beta=0.5, steps=12):
    theta = [1.0, 1.0]
    displacement = [0.0, 0.0]
    history = []

    for step in range(steps + 1):
        history.append((step, theta[:], loss(theta)))
        if step == steps:
            break

        if kind == "gd":
            displacement = [-eta * g for g in grad(theta)]
        elif kind == "heavy-ball":
            current_grad = grad(theta)
            displacement = [
                beta * old - eta * current
                for old, current in zip(displacement, current_grad)
            ]
        elif kind == "nesterov":
            lookahead = [
                current + beta * old
                for current, old in zip(theta, displacement)
            ]
            lookahead_grad = grad(lookahead)
            displacement = [
                beta * old - eta * current
                for old, current in zip(displacement, lookahead_grad)
            ]
        else:
            raise ValueError(kind)

        theta = [
            current + update
            for current, update in zip(theta, displacement)
        ]

    return history


if __name__ == "__main__":
    for kind in ("gd", "heavy-ball", "nesterov"):
        result = run(kind)
        losses = [round(row[2], 9) for row in result[:7]]
        final = result[-1]
        print(
            kind,
            "losses=", losses,
            "final_theta=",
            tuple(round(x, 6) for x in final[1]),
            "final_loss=",
            round(final[2], 12),
        )

    # 同一组参数并不适合任意 Nesterov 约定。
    heavy_ball = run("heavy-ball", eta=0.25, beta=0.25, steps=8)
    nesterov = run("nesterov", eta=0.25, beta=0.25, steps=8)
    print(
        "eta=0.25 beta=0.25",
        "heavy_ball_loss_8=",
        round(heavy_ball[-1][2], 12),
        "nesterov_loss_8=",
        round(nesterov[-1][2], 12),
    )
```

运行输出为：

```text
gd losses= [5.0, 0.45, 0.3285, 0.265725, 0.21523365, 0.174339221, 0.141214768] final_theta= (0.28243, 0.0) final_loss= 0.039883221538
heavy-ball losses= [5.0, 0.45, 1.16, 0.63218, 0.1195016, 0.205736904, 0.072831192] final_theta= (0.028922, 0.003933) final_loss= 0.000487864983
nesterov losses= [5.0, 0.45, 0.298125, 0.197507813, 0.126611613, 0.078746047, 0.047833587] final_theta= (0.061176, -0.0) final_loss= 0.001871246699
eta=0.25 beta=0.25 heavy_ball_loss_8= 0.011795043945 nesterov_loss_8= 17893.993221212
```

第一步的三个 loss 都是 $0.45$，因为缓冲区从零开始。重球法第二步上升到 $1.16$，但后面逐渐下降；这不是实现错误，而是高曲率坐标发生了受控的穿越。显式 Nesterov 在第一组参数上前瞻得更早，前六步的 loss 都比重球法平滑，但最终数值不必在每个步数上胜出。

最后一行更重要：把重球法的 $\eta=0.25,\beta=0.25$ 原样交给显式 lookahead Nesterov，八步后的 loss 从 $5$ 变成了 $17893.993221212$。这组参数对重球法可以收敛，对另一种前瞻递推却失稳。优化器名称相同或相近，不代表状态方程相同。

## 一次训练 step 中各操作的顺序

在神经网络里，动量不是独立于反向传播的第二套梯度。一个可审计的 step 可以写成：

1. 用当前参数做前向计算，得到 batch loss；
2. 反向传播，把梯度写到参数的 grad 字段；
3. 如果使用梯度累积，按有效样本数累加并在规定的 micro-batch 数后才进入优化器；
4. 在未缩放梯度上做梯度裁剪，或按项目约定明确裁剪位置；
5. 将耦合权重衰减加到梯度，或执行独立的解耦参数收缩；
6. 读取该参数组的动量缓冲区，写入新缓冲区；
7. 按重球或 Nesterov 规则计算参数位移；
8. 更新参数、学习率调度器和全局 step；
9. 清空或置空梯度，保存日志和 checkpoint。

用符号写，若先做耦合权重衰减和裁剪后得到 $\widetilde{\boldsymbol g}_k$，重球更新是

$$
\boldsymbol b_{k+1}
=\beta\boldsymbol b_k+\widetilde{\boldsymbol g}_k,
\qquad
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-\eta_k\boldsymbol b_{k+1}.
$$

如果保存的是参数位移，则同一件事是

$$
\boldsymbol d_{k+1}
=\beta\boldsymbol d_k-\eta_k\widetilde{\boldsymbol g}_k,
\qquad
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k+\boldsymbol d_{k+1}.
$$

### 梯度累积时只更新一次缓冲区

假设一个有效 batch 由 $R$ 个 micro-batch 组成，且第 $r$ 个 micro-batch 有 $n_r$ 个有效样本。正确的平均梯度是

$$
\overline{\boldsymbol g}
=\frac{\sum_{r=1}^{R}n_r\boldsymbol g_r}
{\sum_{r=1}^{R}n_r}.
$$

如果每个 micro-batch 大小都相同，可以把 $R$ 个 mean gradient 相加后除以 $R$。但动量缓冲区应当对 $\overline{\boldsymbol g}$ 更新一次：

$$
\boldsymbol b_{k+1}
=\beta\boldsymbol b_k+\overline{\boldsymbol g}.
$$

若每个 micro-batch 都执行一次 $\boldsymbol b\leftarrow\beta\boldsymbol b+\boldsymbol g_r$，就把一次“大 batch 更新”变成了 $R$ 次带动量的小更新：动量衰减次数、参数更新次数和学习率含义都变了。这不是单纯的显存优化。

尾 micro-batch 还要按有效样本数加权。假设两个 micro-batch 分别有 $3$ 和 $1$ 个样本，梯度是 $\boldsymbol g_1$ 和 $\boldsymbol g_2$，平均值应为

$$
\overline{\boldsymbol g}
=\frac{3\boldsymbol g_1+\boldsymbol g_2}{4},
$$

而不是 $\frac12(\boldsymbol g_1+\boldsymbol g_2)$。mask、padding 或被忽略的标签也应进入同一个有效分母。

### 混合精度和裁剪的边界

混合精度训练中，反向传播常先对 loss 乘一个 scale。动量缓冲区不应保存这个被放大的梯度；应先执行 unscale，再做非有限值检查和裁剪：

$$
\boldsymbol g_{\mathrm{usable}}
=\operatorname{unscale}(\boldsymbol g_{\mathrm{scaled}}).
$$

如果先把 scaled gradient 写进动量，缓冲区会把临时的 loss scale 当成真实梯度尺度。即使后来把当前梯度除回来，历史缓冲区也已经被污染。

裁剪也要先说明对象：按全局范数裁剪、按参数张量裁剪和按坐标裁剪不是同一操作。对全局范数裁剪，若

$$
\lVert\boldsymbol g\rVert_2>c,
$$

则可以使用

$$
\widetilde{\boldsymbol g}
=\frac{c}{\lVert\boldsymbol g\rVert_2}\boldsymbol g.
$$

常见的安全顺序是“unscale → 检查有限值 → 裁剪 → 动量更新”。如果把已经积累的缓冲区也裁剪，得到的是另一种算法，必须单独记录。

## 权重衰减不应偷偷改变动量语义

耦合权重衰减把参数项加入梯度：

$$
\widetilde{\boldsymbol g}_k
=\boldsymbol g_k+\lambda\boldsymbol\theta_k.
$$

随后这个总梯度进入动量缓冲区：

$$
\boldsymbol b_{k+1}
=\beta\boldsymbol b_k+\boldsymbol g_k+\lambda\boldsymbol\theta_k.
$$

因此衰减项也会被历史缓冲区带过去。若采用解耦衰减，则先用梯度动量更新，再单独收缩参数：

$$
\boldsymbol\theta_{k+1}
=(1-\eta_k\lambda)\boldsymbol\theta_k
-\eta_k\boldsymbol b_{k+1}.
$$

在普通 SGD 中，这两个写法可能看起来相近；加入动量或逐坐标自适应缩放后，它们的轨迹通常不同。对比实验要同时记录：

| 选择 | 衰减项进入缓冲区吗 | 主要影响 |
| --- | --- | --- |
| 耦合 weight decay | 是 | 历史状态也包含参数收缩方向 |
| 解耦 weight decay | 否 | 参数收缩不经过动量或自适应分母 |
| 无衰减 | 否 | 只观察损失梯度和优化器状态 |

不要只在配置里写 weight_decay=... 就认为两个框架含义相同；还要确认它在 optimizer step 的哪一行生效。

## 学习率、动量和调度器要一起解释

动量会把多步梯度合成为更长的位移，所以提高 $\beta$ 往往需要重新观察 $\eta$。稳定区间是由目标曲率和状态递推共同决定的；对固定一维二次目标

$$
f(x)=\frac12\lambda x^2
$$

重球递推的一个稳定条件是

$$
0<\eta\lambda<2(1+\beta).
$$

这个条件只说明线性递推的根可以落在单位圆内，不保证非线性网络、随机 batch 或每一步的 loss 都下降。实际调参时可以先从较小的 $\eta$ 开始，确认缓冲区和梯度尺度正常，再逐步提高 $\beta$；不要同时把两个旋钮推到边界后再猜是哪一个导致发散。

若 $\eta_k$ 在 warmup 或 cosine schedule 中变化，需要记录调度器的时间单位：

| 时间轴 | 一次计数代表 | 动量缓冲区发生什么 |
| --- | --- | --- |
| micro-batch | 一次前向/反向 | 若只累积梯度，缓冲区不应更新 |
| optimizer step | 一次参数更新 | 缓冲区衰减并吸收一次有效梯度 |
| epoch | 遍历训练数据一轮 | 可能触发调度器，不直接等于缓冲区更新次数 |

一个常见错配是：学习率调度器按 epoch 推进，但代码在每个 optimizer step 调用它；或者 batch size 改大后，仍用旧的总 step 数，导致有效训练时间改变。动量缓冲区无法替你修正这种预算变化。

## 参数组和状态字典

卷积层、归一化层或分类头可能采用不同的学习率、权重衰减和动量系数。若参数分成 $G$ 个组，第 $j$ 组有自己的状态：

$$
\boldsymbol b_{k+1}^{(j)}
=\beta_j\boldsymbol b_k^{(j)}
+\boldsymbol g_k^{(j)},
$$

$$
\boldsymbol\theta_{k+1}^{(j)}
=\boldsymbol\theta_k
-\eta_{k,j}\boldsymbol b_{k+1}^{(j)}.
$$

缓冲区的形状必须与对应参数完全相同；不能把所有参数拼成一个向量后再用错误的切片恢复。每个 checkpoint 至少应包含：

- 参数张量；
- 每个参数组的动量缓冲区；
- 当前 optimizer step；
- 学习率调度器的状态；
- 梯度累积计数和有效样本数；
- 数据采样器、shuffle seed 或可复现所需的随机状态；
- 混合精度 scaler 的状态。

恢复时只加载参数而不加载缓冲区，等价于把正在运动的球突然换成同一位置但速度为零的球。接下来的第一步可能仍然看起来合理，后续轨迹却已经改变。

## 训练中该看哪些信号

动量允许参数穿过低点，所以只看相邻两个 batch loss 很容易误判。至少同时记录以下量：

| 观测量 | 它回答的问题 | 异常时先查什么 |
| --- | --- | --- |
| 当前 batch loss | 这一个 batch 的拟合是否完成 | 数据、标签、归约和随机增强 |
| 固定训练子集 loss | 优化器是否真的降低了同一目标 | $\eta$、$\beta$、weight decay 和缓冲区 |
| 梯度范数 | 反向传播提供了多大信号 | unscale、非有限值和裁剪 |
| 缓冲区范数 | 历史方向是否积累过大 | 动量系数、梯度尺度和恢复状态 |
| 参数更新范数 | 这一步真正移动了多远 | 学习率、缓冲区单位和参数组 |
| 更新比率 | 位移相对参数有多大 | 过小的学习率或即将发散的步长 |
| 验证集指标 | 训练目标是否转化为泛化 | 数据边界、评估模式和停止点 |

可以用

$$
\rho_k
=\frac{\lVert\Delta\boldsymbol\theta_k\rVert_2}
{\max(\lVert\boldsymbol\theta_k\rVert_2,\varepsilon)}
$$

监控更新比率。若 $\rho_k$ 突然增大，先比较 $\lVert\boldsymbol b_k\rVert_2$、$\lVert\boldsymbol g_k\rVert_2$ 和学习率；不要只把 $\beta$ 调到更大来“平滑”曲线。

固定一小批训练样本也很有用。把数据顺序、增强和 dropout 状态固定后，比较重球、Nesterov 与普通 SGD 的 loss 曲线，可以把“优化器状态问题”和“数据随机性问题”分开。固定子集只用于诊断，不应替代独立验证集。

## 失效模式

**缓冲区在错误的时机更新。** 梯度累积期间每个 micro-batch 都推进一次动量，导致有效 batch、衰减次数和参数更新次数都改变。先打印 optimizer step，而不是只打印 dataloader step。

**把速度、位移和指数平均混用。** 三者的单位和学习率位置不同。迁移代码时先写出状态方程，再对照第一步和第二步的数值。

**用旧参数计算 Nesterov 梯度。** lookahead 算法的梯度评估点是 $\boldsymbol y_k$；如果没有前瞻前向，不能声称实现了显式 lookahead。

**把不同 Nesterov 公式共享同一组超参数。** 库式修正和显式前瞻的第一步、稳定边界及有效步长都可能不同。先用二维二次目标做短轨迹测试。

**在 scaled gradient 上裁剪或更新动量。** 混合精度的 loss scale 会被写进历史状态。先 unscale，再检查、裁剪、更新缓冲区。

**把 weight decay 自动归入“正则化一样”。** 耦合衰减会进入动量，解耦衰减不会；两者的参数轨迹和最终解释不同。

**只保存模型参数。** 丢失缓冲区、调度器、sampler 或 scaler 后，恢复训练不是从原来的优化状态继续。

**把暂时的 loss 上升当成发散。** 动量轨迹可以越过谷底。比较固定子集的滑动曲线、更新比率和缓冲区范数，确认是否呈现衰减振荡。

**只增加 $\beta$ 来解决高方差。** $\beta$ 延长记忆，也延长错误方向和旧 batch 噪声的影响；先检查 batch、归约分母和数据顺序。

## 运行方法

将上面的标准库代码保存为 momentum-and-nesterov.py，运行：

```bash
python3 momentum-and-nesterov.py
```

如果要把它变成真实训练循环，保留同样的最小对照：先固定一个小数据集和参数初值，分别记录梯度、缓冲区、参数位移和 loss，再打开随机增强、梯度累积或混合精度。这样每次增加一个训练机制时，都能知道变化发生在状态方程的哪一步。

## 相关词条

- [梯度下降](../training-nn/gradient-descent/)：先建立没有历史状态的负梯度更新和学习率诊断。
- [小批量随机梯度下降](../training-nn/minibatch-sgd/)：说明 batch 抽样、梯度方差、梯度累积和有效 batch。
- [动量理论](../optimization-theory/momentum-theory/)：推导重球法特征根、病态二次目标和理论加速条件。
- [随机梯度下降理论](../optimization-theory/stochastic-gradient-descent-theory/)：分析随机梯度的期望下降、方差和噪声地板。
- [梯度裁剪](../training-nn/gradient-clipping/)：处理进入动量缓冲区前的异常梯度范数。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：比较耦合正则项与解耦参数收缩。
- [学习率调度](../training-nn/learning-rate-schedules/)：说明 warmup、衰减和 optimizer step 的时间轴。
- [批归一化](../training-nn/batch-normalization/)：理解 batch 统计量为何会影响梯度累积的等价性。
- [混合精度](../training-nn/mixed-precision/)：解释 unscale、scaler 和非有限梯度检查的边界。
- [训练调试](../training-nn/debugging-training/)：把缓冲区、更新比率和 checkpoint 纳入训练审计。
