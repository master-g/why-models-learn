---
title: "MNIST + MLP 训练循环：把一张图变成一次参数更新"
tags: ["why-models-learn"]
---

MNIST + MLP 训练循环是一个把前面各篇连接起来的实战词条：输入是一张 $28\times28$ 的灰度数字图像，模型把它展平成 $784$ 个特征，经过一个带 ReLU 的隐藏层产生十个 logits，交叉熵把 logits 和标签变成一个标量损失，反向传播与优化器再把这个标量变成参数位移。它的重点不是“用几行 API 得到一个准确率”，而是能逐项回答一批数据经过了什么形状、损失到底在比较什么、参数何时改变，以及评估数字是否来自未参与更新的数据。

![MNIST 加 MLP 的训练循环：批次数据经过展平、隐藏层和 logits，损失沿反向路径产生梯度，优化器完成参数更新，并在边界记录形状、标签和指标](/assets/training-nn/svg/mnist-mlp-training-loop.1.svg)

## 训练循环到底在闭合什么

一次完整的训练更新包含五个边界：

1. **数据边界**：取得一批图像 $X$ 和整数标签 $y$，核对样本轴、像素范围与标签范围；
2. **前向边界**：模型把 $X$ 映射为 logits $Z$，核对每个张量的形状和有限性；
3. **目标边界**：损失函数把 $Z$ 与 $y$ 归约成一个标量，确认 reduction 和标签语义一致；
4. **反向边界**：从标量损失计算每个参数的梯度，确认梯度确实存在并且方向合理；
5. **更新与评估边界**：优化器修改参数，评估循环只读取冻结后的模型，不把测试集梯度带回训练。

因此训练循环不是一条只有 forward 和 backward 的直线，而是一个带协议的执行链路：

$$
\text{batch}
\longrightarrow
\text{logits}
\longrightarrow
\text{loss}
\longrightarrow
\nabla_{\boldsymbol\theta}L
\longrightarrow
\boldsymbol\theta_{\text{new}}
\longrightarrow
\text{metric}.
$$

如果只看到最后的准确率，就看不到哪一个边界先偏离了预期。一个错误的标签、一个错误的 dim、一个没有执行的 step 和一个把测试集放进训练的循环，都可能在“程序正常结束”的情况下产生数字。

## MNIST 的数据合同

MNIST 的每个样本是一张单通道 $28\times28$ 的灰度图，标签是 $0$ 到 $9$ 的整数。一个 batch 的原始输入和标签应满足

$$
X\in\mathbb R^{B\times1\times28\times28},
\qquad
y\in\{0,\ldots,9\}^{B}.
$$

这里 $B$ 是批次大小。第一维是样本轴，第二维是通道轴，最后两维是图像的高和宽。标签不是十维 one-hot 向量，而是每个样本对应的一个类别索引；这正是多分类交叉熵通常要求的目标格式。

ToTensor 会把整数像素变成浮点张量，通常落在 $[0,1]$。这一步改变的是数值范围，不改变图像的空间形状。若再做标准化，必须把训练集和测试集使用的均值、标准差记录在同一个数据合同中：

$$
\widetilde X
=\frac{X-\mu}{\sigma}.
$$

标准化不是“让模型更聪明”的额外层，而是改变输入坐标尺度的预处理。如果训练时使用了标准化，评估、导出和部署时也必须使用同一组 $\mu$ 与 $\sigma$；只在测试侧标准化会制造输入分布不一致。

### 训练集和测试集的职责不同

训练集中的样本参与损失、梯度和参数更新。测试集只在训练完成后回答“这组参数对未参与更新的样本表现如何”。如果用测试准确率来挑选 epoch、学习率或隐藏层宽度，它就已经承担了验证集的职责，最终数字不再是一次真正的最终测试。[^mnist-test-boundary]

[^mnist-test-boundary]: 选择超参数时应使用独立验证集；测试集只用于最终一次独立报告。

在最小示例中可以直接使用官方 train/test 切分，但仍要在代码里保留两个独立的 loader。数据对象分开并不是形式主义：它让评估循环无法意外复用训练 batch，也让后面的审计可以独立检查样本数与标签分布。

## MLP 如何把图像变成 logits

### 展平只改变布局

MLP 不使用卷积的局部连接。它先把每张图像的三维坐标 $(1,28,28)$ 展平为一个长度为 $784$ 的向量：

$$
x\in\mathbb R^{1\times28\times28}
\longmapsto
\operatorname{vec}(x)\in\mathbb R^{784}.
$$

对整个 batch，展平操作是

$$
X_{\mathrm{flat}}
\in\mathbb R^{B\times784}.
$$

展平不会改变元素的数值，也不会执行学习；它只把后续全连接层所需的特征轴放到最后。若误把 batch 轴也展平，形状可能仍然“能乘上”，却会把不同样本拼成一条错误的输入，训练指标会失去意义。因此形状检查应在第一次前向前发生，而不是等矩阵乘法报错。

### 隐藏层和输出头

取一个宽度为 $128$ 的隐藏层，令第一层权重为 $W_1$、偏置为 $b_1$，输出层权重为 $W_2$、偏置为 $b_2$。每一批数据的前向计算可以写成

$$
\begin{aligned}
H
&=\operatorname{ReLU}
\left(
X_{\mathrm{flat}}W_1^{\mathsf T}
+\boldsymbol 1 b_1^{\mathsf T}
\right),
&H&\in\mathbb R^{B\times128},\\
Z
&=HW_2^{\mathsf T}
+\boldsymbol 1 b_2^{\mathsf T},
&Z&\in\mathbb R^{B\times10}.
\end{aligned}
$$

$Z$ 的每一行是一个样本对十个类别的十个未归一化分数。它们可以是任意实数，不要求先落在 $[0,1]$，也不要求每行加起来等于 $1$；“把 logits 当成概率”是分类训练中最常见的边界混淆之一。

### 参数量必须先算清楚

第一层有 $784\times128$ 个权重和 $128$ 个偏置，第二层有 $128\times10$ 个权重和 $10$ 个偏置。因此总参数量是

$$
\begin{aligned}
P
&=(784+1)\times128+(128+1)\times10\\
&=100480+1290\\
&=101770.
\end{aligned}
$$

这个数字是一个很便宜的结构检查。如果代码报告的参数量不是 $101770$，先查 Flatten 后的特征数、隐藏层宽度、输出类别数和是否意外加入了额外层，不要先调整学习率。

## logits、softmax 和交叉熵

对第 $i$ 个样本的 logits $z_i=(z_{i1},\ldots,z_{i10})$，softmax 概率为

$$
p_{ij}
=\frac{\exp(z_{ij})}
{\sum_{k=1}^{10}\exp(z_{ik})}.
$$

如果真实标签是 $y_i$，交叉熵的逐样本损失是

$$
\ell_i
=-\log p_{i,y_i}
=-z_{i,y_i}
+\log\sum_{k=1}^{10}\exp(z_{ik}).
$$

实际实现通常直接使用 logits 版本的交叉熵，让库在内部用稳定的 log-sum-exp 计算，而不是先手写 softmax 再取对数。对一个 batch，默认的 mean reduction 是

$$
L_{\mathcal B}
=\frac1B\sum_{i=1}^{B}\ell_i.
$$

在这个目标下，logits 的逐坐标导数具有清楚的形式：

$$
\frac{\partial\ell_i}{\partial z_{ij}}
=p_{ij}-\mathbf 1\{j=y_i\}.
$$

正确类别的梯度会推动其 logit 相对升高，错误类别的梯度会推动其相对降低。若模型输出头、标签编码和损失函数不匹配，这个方向解释就不成立。例如把类别索引转换成 one-hot 后仍传给只接受索引的交叉熵，可能得到形状错误，也可能在某些 API 组合中得到与预期不同的目标。

交叉熵、logits 和 softmax 的边界在[分类损失](../training-nn/classification-losses/)与[softmax 函数](../neurons-and-activations/softmax/)中展开；这里只保留训练循环需要的接口合同。

## 一次参数更新的严格顺序

对每个训练 batch，顺序应是：

1. 清空上一批次留下的梯度；
2. 前向得到 logits；
3. 计算一个标量 loss；
4. 调用反向传播，填充参数的梯度；
5. 调用优化器更新参数；
6. 记录这一次更新所对应的 loss、样本数和可选的梯度统计。

把第 5 步放到第 3 步之前，或者在 backward 前调用 step，都可能让程序继续运行，却不再表示“当前梯度推动当前参数”。在 PyTorch 中，梯度默认累加，所以“清空梯度”不是可有可无的整理动作：

$$
\boldsymbol g_t
=\frac{\partial L_{\mathcal B_t}}
{\partial\boldsymbol\theta_t}
$$

必须对应当前 batch 和当前参数；如果没有清空，实际拿到的是多个 batch 梯度的和。梯度累积是一个有意的训练协议，但它必须显式除以累积步数，并把 optimizer step 的时间轴写清楚。[反向传播](../backpropagation/backpropagation/)、[向量化反向传播](../backpropagation/vectorized-backprop/)和[优化器](../training-nn/optimizers/)分别解释这三个边界。

梯度下降的抽象更新仍然是

$$
\boldsymbol\theta_{t+1}
=\operatorname{Update}
\left(
\boldsymbol\theta_t,
\boldsymbol g_t,
\text{state}_t
\right).
$$

对普通 SGD，它近似为

$$
\boldsymbol\theta_{t+1}
=\boldsymbol\theta_t-\eta\boldsymbol g_t.
$$

对 Adam 或带动量的优化器，state 还包含动量、二阶矩和 step 等信息；因此只保存模型参数而不保存优化器状态，恢复后不一定沿着原来的轨迹继续。

## 一个 epoch 的 loss 和 accuracy 怎么算

假设第 $b$ 个 batch 有 $B_b$ 个样本，batch loss 是这些样本损失的平均值。整个 epoch 的样本平均 loss 应按样本数加权：

$$
L_{\mathrm{epoch}}
=\frac{\sum_b B_b L_b}
{\sum_b B_b}.
$$

如果所有 batch 恰好等大，直接平均 $L_b$ 没问题；如果最后一个 batch 更小，直接对 batch loss 做无权平均会让尾批次获得与大 batch 相同的影响。

> [!marginnote] 指标口径
> 每个 epoch 的 loss 按样本数加权；较小的尾批次不能获得与完整 batch 相同的权重。

准确率则先累计预测正确的样本数：

$$
\operatorname{accuracy}
=\frac{\sum_b
\sum_{i\in\mathcal B_b}
\mathbf 1\{\widehat y_i=y_i\}}
{\sum_b B_b},
\qquad
\widehat y_i=\arg\max_j z_{ij}.
$$

loss 的分母和 accuracy 的分母都应写进实现。一个常见错误是把 logits 的十个类别也纳入样本计数，或者在训练时累计了 batch 平均 loss、评估时却累计了逐样本 loss，最后的曲线就没有可比的尺度。

## 训练态和评估态不是同一个前向

训练循环需要 model.train()，评估循环需要 model.eval()。在当前这个没有 Dropout 和 BatchNorm 的最小 MLP 中，两者暂时看起来等价，但这只是模型结构的偶然简化。加上 Dropout 后，训练态会采样掩码；加上 BatchNorm 后，训练态会使用并更新 batch 统计，评估态会读取冻结的 running statistics。

评估还应使用 no-grad 协议：

$$
\nabla_{\boldsymbol\theta}
L_{\mathrm{test}}
\quad\text{不参与训练更新}.
$$

这不是为了让测试指标更高，而是为了防止评估保留计算图、消耗额外内存，或在无意中把测试损失反向传播到参数。评估阶段只做三件事：读取数据、前向计算、累计指标。

如果训练曲线和测试曲线同时打印，必须标注它们对应的时间点。训练 loss 是参数正在被更新时看到的 batch 平均，测试 loss 是一个 epoch 结束后用新参数重新遍历测试集的平均；两者不是同一组参数下的同一批样本。

## 一个可复算的完整 PyTorch 探针

下面的程序保留了最小但完整的边界：固定随机种子、明确数据转换、显式计算参数量、分离训练/评估函数、按样本累计指标，并且只在最后打印最终测试准确率。真实运行时需要先让 torchvision 能访问 MNIST 数据源；此前完整 MNIST 运行记录的最后准确率为 0.9774。

```python
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

root = Path("data/mnist")
transform = transforms.ToTensor()
train_set = datasets.MNIST(
    root=root, train=True, download=True, transform=transform
)
test_set = datasets.MNIST(
    root=root, train=False, download=True, transform=transform
)

generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_set, batch_size=128, shuffle=True,
    generator=generator, num_workers=0
)
test_loader = DataLoader(
    test_set, batch_size=256, shuffle=False, num_workers=0
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
loss_fn = nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train_epoch():
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, labels in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.numel()
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, labels in test_loader:
        logits = model(images)
        total_loss += loss_fn(logits, labels).item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.numel()
    return total_loss / total_count, total_correct / total_count


parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(
    "train_samples=", len(train_set),
    "test_samples=", len(test_set),
    "image_shape=", tuple(train_set.data.shape[1:]),
    "classes=", 10,
)
print("parameter_count=", parameter_count)

for epoch in range(5):
    train_loss, train_accuracy = train_epoch()
    test_loss, test_accuracy = evaluate()

print(f"epoch={epoch + 1} test_acc={test_accuracy:.4f}")
```

输出的结构应类似：

```text
train_samples= 60000 test_samples= 10000 image_shape= (28, 28) classes= 10
parameter_count= 101770
epoch=5 test_acc=0.9774
```

这里 train_set.data.shape[1:] 是原始数据对象的 $(28,28)$，而送入模型的 batch 经过 ToTensor 后是 $(B,1,28,28)$。不要因为日志中只打印了原始图像形状，就忘记通道轴已经由 transform 加入。

代码使用 reduction 设为 sum 再除以累计样本数，是为了让最后一个不完整 batch 不改变 epoch loss 的定义。若把损失改成默认 mean，就必须在累计时乘回当前 batch 的样本数；否则两个实现会报告不同的数值，却看起来都像“平均 loss”。

## 先用最小 batch 验证循环能闭合

在完整 MNIST 训练前，固定一个很小的 batch，例如 $32$ 张图像，关闭 Dropout、数据增强和复杂 scheduler，只训练这一批数据。一个容量足够的 MLP 应该能够把这批样本的训练 loss 压得很低，准确率接近 $1$。这个测试故意不回答泛化问题，它只回答：

- 输入和标签是否对应；
- 输出头是否产生十个 logits；
- 交叉熵是否在读取正确的类别；
- 梯度是否到达每一层；
- optimizer 是否真的改变了参数；
- train_epoch 的 loss 累计是否会下降。

如果连固定的 $32$ 张图都拟合不了，不要先换更大的模型。先记录第一批的图像范围、标签直方图、logits、loss、每层梯度范数和参数更新范数。[训练调试](../training-nn/debugging-training/)把这条最小训练链路推广成完整的证据协议。

## 两个故障对照：代码能跑不等于模型学对

### 去掉 ReLU 会把 MLP 变成线性模型

如果把

$$
H=\operatorname{ReLU}(X_{\mathrm{flat}}W_1^{\mathsf T}+b_1)
$$

改成恒等映射，那么两层线性变换可以合并：

$$
X_{\mathrm{flat}}W_1^{\mathsf T}W_2^{\mathsf T}
\longmapsto
X_{\mathrm{flat}}W^{\mathsf T}.
$$

参数虽然仍然分成两组，函数表达能力却退化成一个线性分类器。它可能达到一个看起来不差的准确率，因为 MNIST 的像素存在强烈的线性可分结构；但它不能表达隐藏层非线性带来的额外决策边界。此前的故障对照中，去掉 ReLU 的版本约在 $90\%$ 附近，而带 ReLU 的完整 MLP 达到 $0.9774$。这不是“反向传播坏了”，而是模型族变了。

### 学习率过大时，损失会先失去解释力

把 $\eta=0.1$ 改成 $\eta=10$，一次更新可能把 logits 推到极端范围。交叉熵会先出现很大的数值，随后模型可能把输出压到近似均匀分布。十分类均匀分布的交叉熵基线是

$$
-\log\frac1{10}=\log 10\approx2.302585.
$$

如果每一类都得到相同概率，准确率也会退化到约 $10\%$。此前故障运行的第一轮 loss 达到约 $2.7\times10^5$，之后出现接近 $\log10$ 的损失与 $10\%$ 的准确率。这个数字模式比单独看“loss 变大”更有诊断力：它提示参数已经把类别差异抹掉，而不是简单地需要多训练几轮。

学习率、损失 reduction、输入尺度和 optimizer 状态必须一起看。只把学习率调小可能让曲线重新下降，却没有解释为什么原来的更新跨过了可用的数值区域。[梯度下降](../training-nn/gradient-descent/)、[学习率调度](../training-nn/learning-rate-schedules/)和[权重初始化](../training-nn/weight-initialization/)分别处理这些前置条件。

## 训练日志要记录哪个时间轴

一行日志至少要能回答“这是第几个 epoch、处理了多少样本、完成了多少次参数更新”。$60000$ 个训练样本使用 batch size $128$ 时，一个完整 epoch 有

$$
\left\lceil\frac{60000}{128}\right\rceil=469
$$

次 optimizer step。epoch=5 不是“模型更新了五次”，而是训练集被遍历了五轮，参数实际被更新了约 $2345$ 次。

如果使用梯度累积，每次 optimizer step 之前可能已经读取多个 micro-batch；这时应同时记录：

$$
B_{\mathrm{effective}}
=B_{\mathrm{device}}
\times N_{\mathrm{device}}
\times K_{\mathrm{accum}},
$$

以及真正的 optimizer step 数。只报告 epoch 会掩盖 batch size、设备数和累积步数改变后训练预算的变化。

建议在日志中保留以下最小字段：

| 字段 | 含义 | 第一用途 |
| --- | --- | --- |
| epoch、step、samples | 训练时间轴 | 判断 scheduler 和恢复位置 |
| loss、accuracy | 当前数据上的目标与指标 | 看优化是否启动 |
| learning rate | 当前更新尺度 | 解释曲线拐点 |
| gradient norm、update norm | 反向与更新的尺度 | 区分无梯度和未更新 |
| train/eval mode | 前向协议 | 排查 Dropout、BatchNorm 分支 |
| seed、数据顺序、checkpoint | 实验状态 | 复现实验和恢复轨迹 |

指标不是装饰。没有时间轴的 loss 曲线只能说明“曾经出现过这些数字”，不能说明哪一次参数更新造成了变化。

## checkpoint 不只是保存模型参数

要从第 $t$ 步继续同一条训练轨迹，至少要保存：

$$
\mathcal C_t=
\left(
\boldsymbol\theta_t,
\text{optimizer\_state}_t,
\text{scheduler\_state}_t,
\text{epoch}_t,
\text{step}_t,
\text{random\_state}_t
\right).
$$

只保存 $\boldsymbol\theta_t$ 可以恢复函数近似，却不一定恢复更新轨迹。SGD 没有动量时状态较少，Adam、Momentum、学习率调度器和 DataLoader shuffle 都会引入额外状态。若恢复后下一批数据顺序、scheduler step 或 optimizer state 不同，新的曲线不是旧曲线的继续，哪怕起始参数逐位相同。

实际保存时还要写入数据预处理配置、类别映射、模型结构超参数和代码版本。MNIST 的类别顺序很直观，但真实项目中标签映射一旦变化，旧 checkpoint 仍可能成功加载，却对应完全不同的输出语义。

## 常见失效模式与第一证据

| 现象 | 可能原因 | 第一条证据 |
| --- | --- | --- |
| 第一批 loss 就是 NaN | 输入、参数初始化或 logits 已非有限 | 检查原始输入、首层输出和 loss |
| loss 不变、梯度全零 | 标签错位、模型未连接、激活死区或参数冻结 | 固定 batch 的逐层梯度与参数差 |
| 训练准确率也上不去 | 输出头、标签编码、loss 或数据合同错误 | 打印单样本 logits、标签和预测 |
| 训练很好、测试很差 | 过拟合、切分泄漏、评估模式或预处理不一致 | 逐样本 train/eval 输出与切分记录 |
| loss 快速变大 | 学习率、输入尺度、reduction 或更新符号错误 | 一步前后 loss、梯度和参数差 |
| 恢复后曲线突然改变 | optimizer、scheduler、shuffle 或随机状态未恢复 | 对比 checkpoint 中的全部状态 |
| 准确率约等于 $10\%$ | 输出近似均匀、标签映射错或模型已坍缩 | logits 的行间方差、类别计数和混淆矩阵 |

这些现象不能互相替代。准确率为 $10\%$ 不足以证明模型忘记了所有东西；loss 接近 $\log10$、logits 行间差异消失、每类预测计数接近均匀，三条证据合起来才支持“均匀输出坍缩”的解释。

## 这篇实战与前面各个概念的连接

这个例子把多个词条放进一个可以运行的训练链路：

- [什么是神经元](../neurons-and-activations/what-is-a-neuron/)：线性层、偏置和激活如何构成单个神经元与一层；
- [前向传播](../backpropagation/forward-pass/)：输入、隐藏表示、logits 和损失的依赖顺序；
- [分类损失](../training-nn/classification-losses/)：标签索引、logits 和交叉熵的接口；
- [softmax 函数](../neurons-and-activations/softmax/)：logits 变成概率时的归一化语义；
- [向量化反向传播](../backpropagation/vectorized-backprop/)：样本轴保留后，矩阵运算如何同时处理一批样本；
- [小批量随机梯度下降](../training-nn/minibatch-sgd/)：batch size、shuffle 和 epoch 更新次数；
- [优化器](../training-nn/optimizers/)：梯度、状态和参数位移的统一接口；
- [权重初始化](../training-nn/weight-initialization/)：让不同隐藏单元从不同方向开始学习；
- [训练集、验证集与测试集](../learning-framework/train-validation-test/)：评估边界和选择边界；
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：训练准确率和未见数据准确率的裂口。

MNIST 的价值不是它本身很复杂，而是它足够小，能让每一个边界都被看见。真正的大模型训练只是把相同的边界扩展到更多层、更多设备、更多状态和更长的时间轴。

## 运行方法

把上面的 Python 代码保存为 mnist_mlp_training_loop.py，在项目环境中运行：

```bash
uv run --with torch --with torchvision python mnist_mlp_training_loop.py
```

第一次运行需要下载 MNIST。若数据源不可达，先把四个 IDX 文件放到 data/mnist/MNIST/raw/，再保留代码中的 download=True 以便在文件已存在时通过完整性检查。运行时不要把测试集改成训练集，也不要为了让数字好看而在测试循环中调用 optimizer.step()。

## 相关词条

- [什么是神经元](../neurons-and-activations/what-is-a-neuron/)：线性层、偏置与激活的构件。
- [前向传播](../backpropagation/forward-pass/)：从输入到 logits 和损失的执行顺序。
- [分类损失](../training-nn/classification-losses/)：多分类交叉熵的标签与 reduction 语义。
- [softmax 函数](../neurons-and-activations/softmax/)：logits 到概率分布的归一化。
- [向量化反向传播](../backpropagation/vectorized-backprop/)：保留 batch 轴的矩阵化梯度。
- [小批量随机梯度下降](../training-nn/minibatch-sgd/)：batch 抽样、shuffle 与更新预算。
- [优化器](../training-nn/optimizers/)：参数更新和内部状态。
- [权重初始化](../training-nn/weight-initialization/)：隐藏单元的初始尺度与对称打破。
- [训练集、验证集与测试集](../learning-framework/train-validation-test/)：数据切分和最终评估边界。
- [训练调试](../training-nn/debugging-training/)：把异常拆成可定位的证据。
