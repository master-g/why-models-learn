---
title: "混合精度训练：让低精度算得快，让关键状态保持准确"
tags: ["why-models-learn"]
---

混合精度训练（mixed-precision training）让神经网络的一部分前向与反向计算使用 float16 或 bfloat16，同时把容易溢出、下溢或积累舍入误差的量保留在 float32 等更高精度中。它不是把模型整体转换成一种更小的 dtype，而是为参数、激活、乘加归约、损失、梯度和优化器状态分别规定数值边界；loss scaling、有限性检查和高精度主权重正是这套边界的一部分。

![混合精度训练示意图：低精度激活和矩阵计算经过高精度归约，损失缩放后反向，再在有限性检查和梯度裁剪之后更新 float32 主权重](/assets/training-nn/svg/mixed-precision.1.svg)

## 精度不是一个开关

实数轴是连续的，浮点 dtype 只能表示有限个离散值。一个正常的二进制浮点数可以抽象写成

$$
x=(-1)^s(1.f)_2 2^e.
$$

$s$ 决定正负，$e$ 决定数量级，$f$ 决定同一数量级内能区分多细。指数位越多，能覆盖的范围通常越宽；尾数位越多，同一范围内的相对间隔通常越小。零、次正规数、无穷和 NaN 还要由具体格式的特殊编码处理。

因此一次训练计算至少有四个问题：

1. 这个张量以什么 dtype 存储；
2. 算子输入在什么 dtype 中相乘；
3. 乘加或归约在哪个 dtype 中累加；
4. 结果、梯度和状态在什么 dtype 中保存。

把四个问题都回答成 float16，才是“全半精度”；混合精度通常会让不同环节承担不同风险。

| 角色 | 常见选择 | 需要核对的风险 |
| --- | --- | --- |
| 参数或激活的存储 | float16 或 bfloat16 | 内存占用、带宽、表示范围和舍入 |
| 矩阵乘与卷积的输入 | float16 或 bfloat16 | 乘法输入是否过大或过小 |
| 乘加和归约累加器 | float32 或硬件指定的高精度累加 | 长求和、点积和方差是否溢出或丢位 |
| 损失、归一化统计和优化器状态 | 通常 float32 | 指数、除法、小量和长期累积 |
| 主权重与最终参数更新 | 通常 float32 | 很小的更新是否被参数存储吞掉 |

“使用低精度”若只写在配置文件里还不够。需要沿一条张量路径记录 dtype：输入进入哪一个算子、算子内部用什么累加、结果是否在下一个边界重新转换。两个 API 都叫 mixed precision，实际的算子白名单、累加器和设备内核也可能不同。

## float16 和 bfloat16 各自解决什么问题

常见格式的位宽可以先这样比较：

| 格式 | 指数位 | 尾数位 | 主要特点 |
| --- | ---: | ---: | --- |
| float16 / binary16 | 5 | 10 | 相对精度比 bfloat16 细，但有限范围窄 |
| bfloat16 | 8 | 7 | 指数范围接近 float32，但同一数量级的间隔更粗 |
| float32 | 8 | 23 | 范围和相对精度都较宽，代价是更多存储与带宽 |

这里的尾数位不包括正常数编码中隐含的最高位。表格用于理解格式，不能代替目标设备的内核行为：有些硬件会对次正规数采用 flush-to-zero，有些矩阵内核会使用比输入更高精度的累加器。

以 float16 的标准表示为例，最大有限值是 $65504$，最小正次正规数约为 $5.960464\times10^{-8}$。标准库模拟会得到：

- $2^{-24}$ 仍可表示为 $5.960464477539063\times10^{-8}$；
- $2^{-25}$ 舍入为 $0$；
- 两个 $40000$ 相加后再存回 float16 会得到 $\operatorname{inf}$。

这两个边界分别对应下溢和上溢。下溢会把一个本来非零的梯度或更新变成零；上溢会把一个有限值变成无穷，后续运算很容易传播 NaN。bfloat16 的指数位更多，通常更能容纳宽范围的激活和梯度，但它的尾数位更少，接近的数之间间隔更大。它不是在所有模型和硬件上都自动优于 float16，归约精度、损失实现和优化器状态仍然要单独审计。

## 混合精度是在计算图边界放置转换

一个实际的前向可以有这样的 dtype 分工：

| 操作 | 常见执行方式 | 为什么不能只看输入 dtype |
| --- | --- | --- |
| 线性层、矩阵乘、卷积 | 低精度输入，较高精度乘加 | 输出尺度取决于累加长度和实现 |
| 加法、残差连接 | 尽量保持共同 dtype，必要时显式转换 | 两个分支的尺度和舍入误差会合并 |
| 归一化与方差统计 | 统计量和归约使用 float32 | 平方、求和和除法更容易放大误差 |
| softmax、log-sum-exp 与分类损失 | 通常使用 float32 或稳定 logits 内核 | 指数、对数和极小概率的尾部很敏感 |
| 激活函数 | 按实现和输入范围决定 | 指数型激活的尾部可能先溢出 |

自动混合精度工具常把这一组规则封装在 autocast 作用域中，但 autocast 不是数学定理。它可能只改变某些算子的输入转换，不会自动修复不稳定的公式；自定义算子、融合内核、第三方扩展和显式 dtype 转换也可能绕过默认策略。

例如，先把 logits 直接取指数再归一化，即使输入张量能用 float16 保存，也不代表指数应该在 float16 中计算。稳定 softmax 会先减去最大 logit；归一化层会关注平方和和方差；损失会关注 $\log$ 的尾部。把这些高风险边界升级到 float32，通常比把所有算子都升级更节省资源，也更容易定位问题。

对一个残差块，要特别看相加之前两条支路的 dtype 和范围。如果一条支路在低精度、另一条支路在高精度，框架会在某个位置插入转换；若转换发生在饱和或舍入已经造成信息丢失之后，后面再转回高精度也无法恢复原值。记录转换位置比只记录模型的“整体精度”更有用。

## loss scaling 把小梯度暂时搬回可表示范围

设原始损失是 $L$，选择一个正的缩放因子 $S$，反向传播使用

$$
L_{\mathrm{scaled}}=S L.
$$

由链式法则，

$$
\nabla_\theta L_{\mathrm{scaled}}
=S\nabla_\theta L.
$$

因此缩放不会改变理想数学梯度的方向；在更新参数前必须除回 $S$：

$$
g
=\frac{1}{S}
\nabla_\theta L_{\mathrm{scaled}}.
$$

它解决的是表示范围问题。若原始梯度小于 float16 能表示的最小非零量，直接把梯度存入 float16 会得到零；先在低精度反向路径中把它放大，再把缩放后的结果转换到 float32 并除回，就能保留一个近似原始梯度的值。

本篇的标准库探针取原始梯度 $2^{-25}$ 和 $S=2^{10}$。输出显示，不缩放时它变成零，缩放后得到 $3.051757812500\times10^{-5}$，在 float32 中除回得到 $2.980232238770\times10^{-8}$。这个恢复值仍然受 float16 舍入影响，但没有在第一次转换时直接丢掉。

loss scaling 也有反方向的风险：$S$ 太大时，原本有限的梯度可能在反向中上溢为 $\operatorname{inf}$。所以缩放因子不是越大越好，训练需要同时做非有限值检测：

$$
\operatorname{finite}(g)
\Longleftrightarrow
\bigwedge_j
\operatorname{isfinite}(g_j).
$$

检测通常要在 unscale 之前或之后都明确其语义。若 scaled gradient 已经是 $\operatorname{inf}$，除以有限 $S$ 仍然是 $\operatorname{inf}$；真正重要的是在调用优化器前阻止非有限梯度改变参数和动量状态。

如果损失在梯度累积窗口中按 micro-batch 求平均，则应先固定归约约定。设窗口内有 $M$ 个 micro-batch，理想梯度是

$$
g
=\frac{1}{M}\sum_{i=1}^{M}\nabla_\theta L_i.
$$

若代码把每个 $L_i$ 乘上 $S$，还要在累积或最终 unscale 时除以 $M S$，不能只除 $S$。否则有效学习率会随累积步数改变。

## 为什么仍要保留 float32 主权重

一次参数更新写成

$$
\theta_{t+1}
=\theta_t-\eta u_t,
$$

其中 $u_t$ 可以是原始梯度、动量方向或 Adam 的归一化方向。即使 $\eta u_t$ 非零，把结果存回 float16 后也可能舍入回原来的 $\theta_t$。

探针取 $\theta_t=1$，更新量为 $2^{-12}$。直接以 float16 保存更新后的权重得到 $1.0$；用 float32 主权重计算则得到 $0.999755859375$。前者不是“学习率没有生效”，而是参数存储的离散间隔比本次更新更粗。

因此常见训练状态至少分成几层：

| 状态 | 常见保存精度 | 作用 |
| --- | --- | --- |
| 模型向前使用的参数副本 | float16 或 bfloat16 | 节约存储和计算带宽 |
| float32 master weights | float32 | 累积细小更新，再向前转换副本 |
| SGD 动量或 Adam 一阶、二阶矩 | 通常 float32 | 长时间累积，避免小量被吞掉 |
| loss scaler 的数值与增长计数 | float32 或整数 | 决定下一次缩放和跳步 |
| scheduler、optimizer step 与随机状态 | 按实现保存 | 保持恢复后的训练协议 |

有些框架会在优化器内部隐式维护 master weights，有些配置只保存模型的低精度参数。不能从 checkpoint 里看到一组 float16 权重，就假定主权重一定存在；应检查 state dict、优化器状态和恢复后的第一步更新。

这也解释了为什么 Adam 的二阶矩不适合随意放在低精度中。平方、平均和 $\varepsilon$ 分母同时涉及小量与长期累积；低精度下二阶矩可能下溢为零，或者在异常梯度出现时迅速失真。[Adam 优化器](../training-nn/adam/)篇中的更新式需要和实际状态 dtype 一起核对。

## 动态 loss scaling 是一个跳步状态机

固定缩放因子可以工作，但不同训练阶段的梯度范围可能变化。动态 scaler 通常维护一个当前缩放因子 $S_t$ 和一个连续成功计数器。一个常见协议是：

1. 用 $S_t L$ 做前向和反向；
2. 把梯度 unscale 到优化器使用的 dtype；
3. 检查所有梯度是否有限；
4. 若有非有限值，跳过本次参数和优化器状态更新，并降低 $S_t$；
5. 若梯度有限，执行裁剪和优化器更新，成功计数增加；
6. 连续成功达到增长间隔后提高 $S_t$，再把计数器清零。

可以抽象成

$$
S_{t+1}
=
\begin{cases}
S_t / d,&\text{检测到 overflow},\\
S_t g,&\text{连续有限步达到增长间隔},\\
S_t,&\text{其它有限步},
\end{cases}
$$

其中 $d>1$ 是回退因子，$g>1$ 是增长因子。不同框架可能使用不同的初始值、增长间隔、回退因子和检测粒度；算法不变，参数不能凭名称猜。

探针设置初始 $S=256$、两步成功后增长一倍、溢出时减半，得到：

| 步 | 梯度有限 | 动作 | 步后 scale | 成功计数 |
| ---: | :---: | :--- | ---: | ---: |
| 1 | 是 | update | 256 | 1 |
| 2 | 是 | update | 512 | 0 |
| 3 | 否 | skip | 256 | 0 |
| 4 | 是 | update | 256 | 1 |
| 5 | 是 | update | 512 | 0 |

这里的 skip 不是把 NaN 梯度当作零再更新。它意味着这一步的参数、动量和优化器 step 都不应接受这组非有限梯度；scaler 的状态则可以改变。scheduler 是否随一次跳步前进，要按训练循环的定义决定并记录，不能让它因为调用位置不同而悄悄改变有效训练步数。

动态 scaler 还可能掩盖一个更早的模型问题。如果第一层 logits 已经异常、归一化方差为零或损失公式不稳定，反复减小 $S$ 只能延后溢出，不会修复源头。日志至少要记录首个非有限张量、当前 scale、梯度范数、跳过次数和触发该步的输入范围。

## 梯度累加、分布式归约和裁剪要按同一尺度处理

梯度裁剪限制的是准备交给优化器的梯度，而不是带有人工缩放因子的 scaled gradient。全局二范数裁剪通常写成

$$
c
=\min\left(1,\frac{\tau}{\lVert g\rVert_2+\varepsilon}\right),
\qquad
g_{\mathrm{clip}}=c g.
$$

正确的顺序通常是：

$$
\text{scaled backward}
\;\longrightarrow\;
\text{unscale}
\;\longrightarrow\;
\text{finite check}
\;\longrightarrow\;
\text{all-reduce 或汇总}
\;\longrightarrow\;
\text{clip}
\;\longrightarrow\;
\text{optimizer step}.
$$

如果把 scaled gradient 直接拿去裁剪，阈值就随 $S$ 变化。探针固定 $S=256$，scaled gradient 是 $(0,512)$；unscale 后是 $(0,2)$，二范数为 $2$，阈值为 $1$ 时裁剪系数为 $0.5$，最终梯度是 $(0,1)$。这就是“先 unscale 再 clip”的具体差异。

梯度累加时要固定缩放因子和归约分母：

- 每个 micro-batch 的损失是否已经除以累积步数；
- scaled gradient 是在 float32 buffer 中累加，还是在低精度中累加；
- 一个窗口中途是否允许 scaler 改变；
- 溢出时是否丢弃整个窗口，而不是只丢弃一个局部贡献；
- 分布式 all-reduce 前后，有限性检查和全局范数使用什么约定。

在多个设备上，每卡都可能只看到局部 batch。若裁剪目标是全局范数，局部范数不能各自裁剪后再相加；需要按照通信协议先汇总平方和或等价的全局统计。[梯度裁剪](../training-nn/gradient-clipping/)篇的阈值公式，只有在这里的 dtype、缩放和归约顺序都固定后才有同样的含义。

## 训练、评估和检查点必须记录精度协议

评估时通常关闭 dropout 和参数更新，但是否用 autocast 进行推理是另一个选择。低精度推理可能节省带宽，也可能让边界样本的 logits、概率或归一化结果发生变化。训练态的 batch 统计、评估态的 running statistics 和推理 dtype 不能混成一个开关；[批量归一化](../training-nn/batch-normalization/)篇已经展示了统计模式本身会改变函数。

一个可审计的 checkpoint 至少要回答：

| 内容 | 缺失时可能发生什么 | 恢复检查 |
| --- | --- | --- |
| 模型参数与 float32 主权重 | 小更新丢失，恢复后轨迹改变 | 比较恢复前后的第一步参数差 |
| optimizer state 与 step | 动量、二阶矩或偏置修正重置 | 记录每个参数组的状态 dtype |
| scaler 的 scale 与成功计数 | 恢复后突然频繁 overflow 或过度放大 | 打印恢复前后的 scaler 状态 |
| scheduler、累积计数与数据位置 | 有效学习率和 batch 边界改变 | 对齐下一次 update 的输入 |
| 随机数状态 | dropout、采样和数据顺序改变 | 需要可复现时保存并比较 |

恢复测试不应只比较最终 loss。更直接的测试是保存一个训练窗口，在同一数据和随机状态下从 checkpoint 继续，逐步比较 scaler、梯度有限性、优化器 step、参数和指标；若只要求统计等价，也要先写出允许的浮点容差。[梯度检查](../backpropagation/gradient-checking/)适合在关闭随机性、升到 float64 的小图上核对导数；它不能证明生产低精度内核逐位相同。

## 低精度改变的是数值路径，不是反向公式

在精确算术中，softmax 交叉熵对 logits 的梯度仍然是

$$
\frac{\partial L}{\partial z_k}=p_k-y_k.
$$

混合精度训练没有把这个导数换成另一条学习规则。它改变的是每个中间量何时舍入、在哪个 dtype 中累加、是否因为上溢变成 $\operatorname{inf}$、是否因为下溢变成 $0$，以及最终更新是否被低精度参数存储吞掉。

因此“数学公式正确”与“训练实现正确”是两项不同检查：

- 数学检查：用高精度小图推导和验证导数、归约轴与损失分母；
- 数值检查：在目标设备和真实 dtype 下记录每个边界的范围、非有限比例和舍入误差；
- 协议检查：固定 autocast、scale、unscale、裁剪、通信、step 和 checkpoint 的顺序；
- 任务检查：比较低精度训练与高精度基线的收敛、验证指标和部署输出。

低精度下最后几位不同不自动表示错误；但出现 NaN、参数长期不动、恢复轨迹不一致或评估精度突降时，也不能用“浮点误差”四个字结束调查。要先找到第一个数值分叉的位置。

## 一个可复算的标准库模拟

下面的脚本只使用 Python 标准库中的 half 浮点打包，模拟 float16 存储舍入。它不模拟特定 GPU 的矩阵内核，因此输出用于解释下溢、上溢、主权重和 scaler 的关系，而不是硬件吞吐基准。

```python
import math
import struct


def f16(value):
    if math.isnan(value) or math.isinf(value):
        return value
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def f16_add(left, right):
    return f16(f16(left) + f16(right))


print("float16 max=", f16(65504), "small=", f16(2**-24),
      "underflow=", f16(2**-25))
print("float16 overflow add=", f16_add(40000, 40000))

small = 2**-25
scale = 2**10
scaled = f16(small * scale)
unscaled = scaled / scale
print("small grad=", format(small, ".12e"),
      "without scale=", format(f16(small), ".12e"),
      "scaled=", format(scaled, ".12e"),
      "unscaled fp32=", format(unscaled, ".12e"))

weight = 1.0
update = 2**-12
print("weight update=", format(update, ".12e"),
      "naive fp16=", f16(weight - update),
      "fp32 master=", format(weight - update, ".12e"))

scale = 256.0
tracker = 0
for step, finite in enumerate([True, True, False, True, True], 1):
    if finite:
        tracker += 1
        action = "update"
        if tracker == 2:
            scale *= 2
            tracker = 0
    else:
        action = "skip"
        scale /= 2
        tracker = 0
    print(f"scaler step={step} finite={finite} action={action} "
          f"scale={scale:.0f} tracker={tracker}")

clip_scale = 256.0
scaled_grad = (0.0, 512.0)
unscaled_grad = tuple(value / clip_scale for value in scaled_grad)
norm = math.sqrt(sum(value * value for value in unscaled_grad))
coefficient = min(1.0, 1.0 / norm)
clipped = tuple(value * coefficient for value in unscaled_grad)
print("clip after unscale=", "scaled", scaled_grad,
      "unscaled", unscaled_grad, "norm", format(norm, ".1f"),
      "coefficient", format(coefficient, ".1f"), "clipped", clipped)
```

运行输出：

```text
float16 max= 65504.0 small= 5.960464477539063e-08 underflow= 0.0
float16 overflow add= inf
small grad= 2.980232238770e-08 without scale= 0.000000000000e+00 scaled= 3.051757812500e-05 unscaled fp32= 2.980232238770e-08
weight update= 2.441406250000e-04 naive fp16= 1.0 fp32 master= 9.997558593750e-01
scaler step=1 finite=True action=update scale=256 tracker=1
scaler step=2 finite=True action=update scale=512 tracker=0
scaler step=3 finite=False action=skip scale=256 tracker=0
scaler step=4 finite=True action=update scale=256 tracker=1
scaler step=5 finite=True action=update scale=512 tracker=0
clip after unscale= scaled (0.0, 512.0) unscaled (0.0, 2.0) norm 2.0 coefficient 0.5 clipped (0.0, 1.0)
```

脚本有意把 float16 舍入放在存储边界，而把 unscale 和主权重更新写成 Python 高精度标量。真实框架会按设备和 kernel 使用 float32 或更具体的累加格式；阅读输出时应抓住“转换发生在哪里”，不要把这个小程序当成完整的训练器。

## 常见失效模式与审计顺序

| 现象 | 常见原因 | 先查什么 |
| --- | --- | --- |
| 第一批就出现 NaN 或 Inf | logits、平方、方差或归约上溢 | 第一个非有限张量与其输入范围 |
| 梯度长期为零 | 小梯度下溢、错误的 mask 或缩放未生效 | scaled 与 unscaled 梯度直方图 |
| scale 越降越低仍然溢出 | 模型或损失公式本身不稳定 | 稳定 softmax、归一化和首个 overflow 算子 |
| loss 有限但参数不动 | 低精度参数吞掉更新，或 step 被跳过 | master weights、更新范数和 skip 计数 |
| 裁剪结果随 scale 改变 | 在 unscale 前计算范数或裁剪 | 裁剪输入 dtype、范数和缩放因子 |
| 累积步数一变，收敛速度改变 | 损失分母、scale 或归约顺序不一致 | 每个 update 的有效梯度和样本数 |
| 恢复后马上频繁 overflow | checkpoint 丢失 scaler 或 optimizer 状态 | scale、计数器、矩和 step |
| 评估指标下降但训练 loss 正常 | 评估 dtype、归一化模式或部署算子不同 | train/eval、autocast 和输出 logits |

一个实际排查顺序可以压缩成：

1. 找到第一个非有限值，而不是只看最终 loss；
2. 对该张量记录 dtype、设备、最大绝对值、非零比例和累加 dtype；
3. 检查损失是否按预期缩放，反向后的梯度是否在 unscale 前后都有记录；
4. 检查有限性判断是否发生在 optimizer.step 之前；
5. 对累积和分布式归约固定分母、通信顺序和全局范数；
6. 确认 unscale 在裁剪之前，master weights 和 optimizer 状态没有被低精度覆盖；
7. 用 checkpoint 恢复测试验证 scaler、scheduler、随机状态和下一次 update 的边界。

## 运行方法

将“一个可复算的标准库模拟”中的代码保存为 `mixed_precision_probe.py`，在 Python 3 环境执行：

```bash
python3 mixed_precision_probe.py
```

它不需要安装第三方包。若要把结果用于真实训练诊断，还需要在目标框架中额外记录每个算子的 autocast 决策、梯度有限性、通信归约和设备内核；标准库脚本只负责给出可独立复核的格式边界。

## 相关词条

- [前向传播](../backpropagation/forward-pass/)：说明激活、线性变换和稳定 softmax 在计算图中的位置。
- [计算图](../backpropagation/computational-graphs/)：沿节点边界追踪 dtype 转换、缓存和反向依赖。
- [梯度消失与梯度爆炸](../backpropagation/vanishing-and-exploding/)：解释梯度范围为何会在深层路径上衰减或放大。
- [梯度裁剪](../training-nn/gradient-clipping/)：展开全局范数、坐标裁剪及其阈值语义。
- [Adam 优化器](../training-nn/adam/)：说明一阶、二阶状态和 $\varepsilon$ 如何参与更新。
- [批量归一化](../training-nn/batch-normalization/)：对照训练/评估统计量和高风险归约。
- [梯度检查](../backpropagation/gradient-checking/)：在高精度确定性小图中分离数学导数与数值误差。
- [学习率调度](../training-nn/learning-rate-schedules/)：核对跳步、累积和恢复后有效学习率的时间索引。
