---
title: "为什么需要非线性:XOR、空间折叠与深度的意义"
tags: ["why-models-learn"]
---

非线性激活函数是夹在神经网络层与层之间的逐点函数，它是整套架构里最小的构件；但没有它，网络再深也等价于单层。上一篇 [什么是神经元](../neurons-and-activations/what-is-a-neuron/) 给了代数版的必然性：由矩阵乘法的结合律，没有激活函数的深层网络坍缩成一层线性变换。本篇把理由补齐到几何与实验两面：先看 **XOR 问题**——四个点、两条不等式，就能证明线性模型存在它原则上做不到的事；再看一个两层 ReLU 网络如何解开它，从而看清非线性的几何本质是**折叠空间**，让数据在新表示下变得线性可分；最后用一组对照实验验证：同一个网络，加不加激活函数，结果完全不同。

## 线性模型的天花板：XOR 问题

XOR(异或)可能是最小的非平凡分类问题：输入两个比特 $(x_1, x_2)$，两比特**不同**时输出 1，相同时输出 0。四个样本就是全部数据：

$$
(0,0) \mapsto 0, \qquad (0,1) \mapsto 1, \qquad (1,0) \mapsto 1, \qquad (1,1) \mapsto 0
$$

**断言：任何线性分类器都解不了 XOR。** 证明只需四行。设线性模型 $y = w_1 x_1 + w_2 x_2 + b$，以阈值 $t$ 划分：$y > t$ 判 1，$y \le t$ 判 0。四个样本的要求写出来：

$$
b \le t, \qquad w_1 + w_2 + b \le t, \qquad w_2 + b > t, \qquad w_1 + b > t
$$

把两个 1 类的不等式相加：$w_1 + w_2 + 2b > 2t$；把两个 0 类的不等式相加：$w_1 + w_2 + 2b \le 2t$。同一个量既要大于 $2t$ 又要不超过 $2t$——**矛盾，这样的 $(w_1, w_2, b, t)$ 不存在**。注意这个证明不依赖任何训练算法：不是梯度下降没找到解，是解根本不在线性模型的假设空间里(假设空间这个话题见 [假设空间](../learning-framework/hypothesis-spaces/))。

这个结果的历史影响不小：1969 年 Minsky 与 Papert 在《Perceptrons》里强调了它，是神经网络研究进入第一次寒冬的直接原因之一(来龙去脉见 [经典感知机](../linear-models/perceptron-classic/))。也别把 XOR 当成玩具反例：真实数据——像素、文本、语音——几乎从不是线性可分的，XOR 只是把「线性不够」这件事压缩到四个点上的最小例子。

## 手工构造：两层 ReLU 如何解 XOR

一个输入层到两个神经元的隐藏层、再到一个输出神经元的网络，配 ReLU 激活 $\mathrm{relu}(z) = \max(0, z)$，就足以解开 XOR。隐藏层的两个神经元都算 $x_1 + x_2$，但阈值不同：

$$
h_1 = \mathrm{relu}(x_1 + x_2 - 0.5), \qquad h_2 = \mathrm{relu}(x_1 + x_2 - 1.5), \qquad y = h_1 - 3 h_2
$$

$h_1$ 度量「至少有一个比特是 1」的程度，$h_2$ 只在「两个比特都是 1」时启动。逐点代入：

$$
\begin{array}{c|c|c|c}
(x_1, x_2) & (h_1, h_2) & y = h_1 - 3 h_2 & \text{目标} \\ \hline
(0, 0) & (0,\ 0) & 0 & 0 \\
(0, 1) & (0.5,\ 0) & 0.5 & 1 \\
(1, 0) & (0.5,\ 0) & 0.5 & 1 \\
(1, 1) & (1.5,\ 0.5) & 0 & 0
\end{array}
$$

四个样本全部命中(阈值取 0.25 即可分开 0 与 0.5)。注意中间两行：**$(0,1)$ 与 $(1,0)$ 在隐藏空间里被映射到同一个点 $(0.5, 0)$**。在输入空间里，它们是两个点，被另外两个类夹在对面，任何直线都无法把它们与 $(0,0)$、$(1,1)$ 同时分开；在隐藏空间里，它们合并了--「两个比特不同」这个语义，被隐藏层直接算成了一个坐标。

## 空间折叠：非线性的几何本质

把两个空间并排画出来：

![XOR 的两个空间：输入空间里四个点任何直线都分不开(左)；隐藏空间里 (0,1) 与 (1,0) 合并，一条线就够(右)](/assets/neurons-and-activations/svg/why-non-linearity.1.svg)

左图是输入空间：两个 0 类点在对角、两个 1 类点在反对角，一条直线最多把三个点分对，第四个必然站错边。右图是隐藏层输出的 $(h_1, h_2)$ 空间：三个像点(有一个是两点合并)，一条直线干净利落地把 1 类隔在一边。**隐藏层做的事，是把输入空间折叠、剪切、重排，直到任务在新坐标下变成线性可分的**——然后输出层那条「线」随手一画就行。

这说明深度买到了什么。每一层非线性变换都是对空间的一次折叠：单层线性变换只能整体旋转、缩放、剪切，点的相对位置关系(哪些点在线的哪侧)被保形锁定；夹进非线性之后，空间可以弯折，原来缠在一起的类别被摊平。层层复合，折叠可以套折叠。所谓「表示学习」，说的就是这个过程：**网络真正在学的东西，是一组让任务变简单的坐标系**；至于「非线性复合的折叠能力有没有上限」--没有，这就是 [万有逼近](../neurons-and-activations/universal-approximation/) 定理的方向。

## 实验：有无激活的两种命运

理论到此，下面用实验验证。用 PyTorch 搭一个 $2 \to 4 \to 1$ 的小 MLP(隐藏层 ReLU，输出 sigmoid)，在 XOR 的四个样本上训练 2000 轮；再搭一个结构完全相同、唯独**去掉隐藏层激活函数**的对照网络，同样训练(完整代码见「运行方法」)：

$$
\begin{array}{c|c|c}
 & \text{带 ReLU} & \text{无激活} \\ \hline
\text{初始损失} & 0.8065 & 0.8023 \\
\text{2000 轮后} & 0.0002 & 0.6931 \\
\text{输出} & (0,\ 1,\ 0.999,\ 0) & (0.5,\ 0.5,\ 0.5,\ 0.5)
\end{array}
$$

带激活的网络 100 轮就把损失压到 $0.03$，最终输出与目标几乎逐位一致。无激活的网络 2000 轮后损失停在 $0.6931$——这个数字值得注意：它正是 $\ln 2 \approx 0.693$，即「永远输出 0.5」时的二分类交叉熵。这不是训练失败，是结构性无解：结合律保证这个「两层」网络整体仍是一个线性模型，而上一节刚证明过，线性模型在 XOR 上没有解——它能做的最好选择，就是把四个输出都推到 0.5，把损失停在理论下限 $\ln 2$。

同样的对照在真实数据上也成立：MNIST 上去掉激活函数的多层感知机退化到线性模型的精度水平，细节会在 [MNIST 训练循环](../training-nn/mnist-mlp-training-loop/) 重写时完整呈现。

## 运行方法

以下代码自包含，可直接运行(`uv run --with numpy --with torch` 临时环境)。三段：手工构造验证、torch 训练、去激活消融。

```python
import numpy as np

# --- 手工构造的两层 ReLU 解 ---
X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
Y = np.array([0, 1, 1, 0])
W1 = np.array([[1., 1.], [1., 1.]])
b1 = np.array([-0.5, -1.5])
w2 = np.array([1., -3.])
H = np.maximum(0, X @ W1.T + b1)
print(H)                 # (0,1) 与 (1,0) 在隐藏空间合并为 (0.5, 0)
print(H @ w2, Y)         # [0.  0.5 0.5 0. ] 全部命中

# --- torch 训练:带 ReLU ---
import torch, torch.nn as nn
torch.manual_seed(42)
Xt = torch.tensor(X, dtype=torch.float32)
Yt = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)
mlp = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid())
opt = torch.optim.Adam(mlp.parameters(), lr=0.05)
for epoch in range(2001):
    opt.zero_grad()
    loss = nn.BCELoss()(mlp(Xt), Yt)
    loss.backward(); opt.step()
print(loss.item(), mlp(Xt).detach().numpy().round(3).ravel())

# --- 消融:去掉激活函数 ---
torch.manual_seed(42)
lin = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 1), nn.Sigmoid())
opt2 = torch.optim.Adam(lin.parameters(), lr=0.05)
for epoch in range(2001):
    opt2.zero_grad()
    loss = nn.BCELoss()(lin(Xt), Yt)
    loss.backward(); opt2.step()
print(loss.item(), lin(Xt).detach().numpy().round(3).ravel())
```

运行输出(节选)：

```
[0.  0.5 0.5 0. ] [0 1 1 0]          # 手工构造:全部命中
0.0002 [0.    1.    0.999 0.   ]     # 带 ReLU:loss → 0.0002
0.6931 [0.5 0.5 0.5 0.5]             # 无激活:锁死 ln 2,输出全 0.5
```

## 失效模式与常见误区

**以为深度本身带来能力。** 消融实验的数字已经说过一遍：无激活的「两层」网络与带激活的同一个网络，参数量几乎相同、结构只差一个函数，结果是 $\ln 2$ 锁死与 $0.0002$ 之别。能力不在层数里，在层与层之间的非线性里。

**非线性放错位置。** 只在输出端加一个 sigmoid 没用：输出端之前仍然是纯线性复合，整个网络等价于「线性模型 + 输出压缩」，在 XOR 上同样无解。非线性必须夹在**每两层之间**--这就是「每层神经元都过一次 $f$」这个设计不可商量的原因。

**以为任何非线性都一样。** 「需要非线性」不等于「随便什么非线性都行」。sigmoid 在深层里梯度消失($\sigma' \le 0.25$，连乘衰减)，ReLU 有神经元死亡问题——选择是有代价的，失效分析在 [饱和与梯度消失](../neurons-and-activations/saturation-and-vanishing/) 与 [死亡 ReLU 与 Leaky 变体](../neurons-and-activations/dead-relu-and-leaky/)。

**把 XOR 当象牙塔反例。** 「我的数据又不是 XOR」是最常见的反驳。真实任务里类别在原始表示下几乎总是缠在一起的，XOR 只是最小的例子；隐藏层折叠空间这件事，在 784 维的 MNIST 上同样发生，只是肉眼看不见。

## 相关词条

- [什么是神经元](../neurons-and-activations/what-is-a-neuron/)：上一篇，代数版必然性(结合律坍缩)
- [激活函数](../neurons-and-activations/activation-functions/)：非线性的候选清单
- [ReLU](../neurons-and-activations/relu/)：本篇手工构造与实验用的激活
- [饱和与梯度消失](../neurons-and-activations/saturation-and-vanishing/)：选错非线性的代价
- [万有逼近](../neurons-and-activations/universal-approximation/)：折叠能力的理论上限
- [经典感知机](../linear-models/perceptron-classic/)：XOR 与第一次寒冬的来龙去脉
- [假设空间](../learning-framework/hypothesis-spaces/)：「解不在假设空间里」的正式说法
- [线性回归](../linear-models/linear-regression/)：线性模型能做到什么——它的适用边界正是本篇的主题
- [MNIST 训练循环](../training-nn/mnist-mlp-training-loop/)：真实数据上同一组对照的实战
