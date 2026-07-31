---
title: "什么是神经元:加权求和、偏置与一次非线性"
tags: ["why-models-learn"]
---

神经元是神经网络的基本计算单元：它吃进去一个向量 $\mathbf{x}$，吐出一个数 $y = f(\mathbf{w} \cdot \mathbf{x} + b)$，其中 $\mathbf{w}$ 是与输入同维的**权重向量**，$b$ 是一个叫**偏置**(bias)的数，$f$ 是一个固定的**激活函数**。这个定义里没有生物，只有三件数学构件——[点积](../linear-algebra/inner-products/)、加法、一个非线性函数——全部是 Part 0 已经备好的料。本篇是神经网络部分的开篇：我们把神经元解剖到公式级别，看清一个神经元就是一次点积、一层神经元就是一次[矩阵乘法](../linear-algebra/matrix-multiplication/)，并用代码验证它和 PyTorch 的 `nn.Linear` 是同一个东西；至于「为什么非要非线性不可」，本篇只给矩阵乘法级别的必然性论证，展开在 [为什么需要非线性](../neurons-and-activations/why-non-linearity/)。

## 解剖：四件构件

把定义逐个符号拆开。输入是 $n$ 维向量 $\mathbf{x} = (x_1, \dots, x_n)$；权重是同维向量 $\mathbf{w} = (w_1, \dots, w_n)$，每个 $w_i$ 是第 $i$ 个输入分量的「发言权」；偏置 $b$ 是一个数，把整体输出平移一份；激活函数 $f$ 是一个一元函数，逐点作用。计算分两步：

$$
z = \mathbf{w} \cdot \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b, \qquad y = f(z)
$$

第一步的 $z$ 叫**预激活值**(pre-activation)，是一个数；第二步过 $f$ 得到输出 $y$。取一组具体数字：$\mathbf{x} = (2, -1, 3)$、$\mathbf{w} = (0.5, 1.0, -0.3)$、$b = 0.1$:

$$
z = 0.5 \times 2 + 1.0 \times (-1) + (-0.3) \times 3 + 0.1 = -0.8
$$

激活函数选一个具体的，比如 sigmoid 函数 $\sigma(z) = \dfrac{1}{1 + e^{-z}}$(历史上最经典的激活，见 [阶跃与 sigmoid](../neurons-and-activations/heaviside-and-sigmoid/))：

$$
y = \sigma(-0.8) = \frac{1}{1 + e^{0.8}} \approx 0.310
$$

换一个激活函数，比如 $\mathrm{relu}(z) = \max(0, z)$(今天最常用的，见 [ReLU](../neurons-and-activations/relu/))，同一个 $z = -0.8$ 给出的输出是 $0$。**预激活值只有一个，输出长什么样，是激活函数说了算**——这就是「激活」二字的含义：它决定这个神经元对当前的加权输入做出什么形态的反应。

![神经元的解剖：输入按权重加权求和、加偏置得到 z,再过激活函数 f 得到输出 y](/assets/neurons-and-activations/svg/what-is-a-neuron.1.svg)

四件构件里只有 $\mathbf{w}$ 和 $b$ 带下标自由度——它们是**参数**，是「学习」要调整的东西；$\mathbf{x}$ 是数据，$f$ 是设计选择，一旦选定就不动。一个 784 维输入的神经元有 $784 + 1 = 785$ 个参数，这个账马上会和大一点的数字接上。

## 历史：一个被工程征用了八十年的名词

「神经元」这个名字来自神经科学，但它在机器学习里的含义经过两次征用后已经和原型相去甚远。1943 年 McCulloch 和 Pitts 提出形式神经元：输入加权和超过阈值就输出 1，否则输出 0——这是 $f$ 取阶跃函数的特例，也是「神经元可以计算逻辑」的第一次形式化。1958 年 Rosenblatt 的感知机(perceptron)给它配了学习规则：错了就按误差方向调权重——感知机的学习规则与兴衰是 [经典感知机](../linear-models/perceptron-classic/) 的事，本篇只要记住一点:**从第一天起，人工神经元就是「加权和 + 阈值」这个数学对象，生物类比只是命名时的借喻**。把它当成对大脑的模拟会系统性地误导直觉(见本篇失效模式)，把它当成一个可微的小函数，一切就位。

## 一个神经元是点积，一层神经元是矩阵乘法

单个神经元的核心运算是 $\mathbf{w} \cdot \mathbf{x}$——正是 [向量](../linear-algebra/vectors/) 篇说的点积、[矩阵乘法](../linear-algebra/matrix-multiplication/) 篇说的「行向量乘列向量得一个数」。这个观察微不足道，但把它推进一格就得到神经网络的基本组织方式:**一层神经元就是一组权重向量分别与同一个输入做点积**。$m$ 个神经元各有权重 $\mathbf{w}_1, \dots, \mathbf{w}_m$，把它们按行排成矩阵 $W$，一层的前向计算就是：

$$
\mathbf{z} = W\mathbf{x} + \mathbf{b}, \qquad \mathbf{y} = f(\mathbf{z})
$$

$W$ 是 $m \times n$ 矩阵(**行是神经元，列是输入**)，$\mathbf{b}$ 是 $m$ 维偏置向量，$f$ 逐分量作用。这正是矩阵乘法篇「全连接层 $\mathbf{y} = W\mathbf{x} + \mathbf{b}$」的来历：那一节是矩阵视角看层，这一篇是神经元视角看层，同一个对象。参数量账也接上了:128 个神经元的一层($n = 784$)有 $784 \times 128 = 100352$ 个权重，再加 128 个偏置，共 $100480$ 个参数——矩阵乘法篇数过的 100352，就是 128 个神经元各自的 785 个参数里权重的那部分。

工程里这个对象叫全连接层,PyTorch 的 `nn.Linear` 就是它。下面「运行方法」里的代码用 `nn.Linear(4, 2)` 建一个 4 输入、2 神经元的小层，再用 numpy 手算同一组权重下的输出，两者逐位一致——`nn.Linear` 没有任何神秘成分，就是 $W\mathbf{x} + \mathbf{b}$(实现细节上它存 $W$ 的形状是 $(m, n)$、内部算 $\mathbf{x}W^T$，与数学写法 $W\mathbf{x}$ 转置等价，正是矩阵乘法篇转置规律的日常应用)。

## 为什么必须非线性

激活函数不是装饰。把两层线性叠起来——第一层 $\mathbf{z} = W_1 \mathbf{x}$、第二层 $\mathbf{u} = W_2 \mathbf{z}$——由矩阵乘法的结合律：

$$
\mathbf{u} = W_2 (W_1 \mathbf{x}) = (W_2 W_1)\,\mathbf{x}
$$

两层线性等价于一层线性，权重矩阵是 $W_2 W_1$。「运行方法」的代码用一个 $2 \times 2$ 的例子真算了两遍：顺序过两层与合成一个矩阵，输出逐位相同。**叠加一百层线性，表达能力仍与一层相同——深度本身不带来任何新能力，带来新能力的是层与层之间夹着的非线性。** 所以每个神经元在加权求和之后都要过一次 $f$；没有了 $f$，整个神经网络坍缩成一次矩阵乘法。至于为什么非线性的层层复合就能逼近任意函数、激活函数具体怎么选，分别留给 [万有逼近](../neurons-and-activations/universal-approximation/) 与 [激活函数](../neurons-and-activations/activation-functions/)。

## 运行方法

以下代码自包含，可直接运行(`uv run --with numpy --with torch` 临时环境)。分四段：手算一个神经元、与 `nn.Linear` 对照、参数量账、线性层叠合坍缩。

```python
import numpy as np

# --- 手算一个神经元 ---
x = np.array([2.0, -1.0, 3.0])
w = np.array([0.5, 1.0, -0.3])
b = 0.1
z = w @ x + b                       # z = -0.8
sig = 1 / (1 + np.exp(-z))          # sigmoid(z) ≈ 0.310
print(z.round(4), sig.round(4), max(0.0, z))   # relu(z) = 0

# --- 与 nn.Linear 对照 ---
import torch
torch.manual_seed(7)
layer = torch.nn.Linear(4, 2)       # 4 输入,2 个神经元
xt = torch.tensor([1.0, 0.0, -1.0, 2.0])
out = layer(xt).detach().numpy()
W = layer.weight.detach().numpy()   # 形状 (2, 4):行是神经元
manual = W @ xt.numpy() + layer.bias.detach().numpy()
print(out.round(4), manual.round(4), np.allclose(out, manual))  # True

# --- 参数量账 ---
print(784 + 1, 784 * 128 + 128)     # 785, 100480

# --- 线性层叠合坍缩 ---
W1 = np.array([[2., 0.], [1., -1.]])
W2 = np.array([[1., 3.], [0., 2.]])
xx = np.array([1., 4.])
direct = W2 @ (W1 @ xx)             # 顺序过两层
fused = (W2 @ W1) @ xx              # 合并成一层
print(direct, fused, np.allclose(direct, fused))  # [-7. -6.] [-7. -6.] True
```

运行输出(节选)：

```
-0.8 0.3100 0.0
[0.0548 0.6364] [0.0548 0.6364] True
785 100480
[-7. -6.] [-7. -6.] True
```

第一行是手算神经元的三个量($z$、sigmoid 输出、ReLU 输出)；第二行证明 `nn.Linear` 的输出与手算 $W\mathbf{x} + \mathbf{b}$ 逐位一致；第三行是参数量账；最后一行是坍缩实验：两层线性顺序计算与合并成一个矩阵计算，输出完全相同。

## 失效模式与常见误区

**把生物隐喻当真。** 「神经元模拟大脑」的想象会导出错误预期：以为权重有兴奋/抑制之分、以为单个神经元在「检测」某个可读特征。工程现实是：神经元是一个可微参数化函数，权重正负只是加减号，单个神经元的「含义」由整层、整网的上下文共同决定。学这套理论，从第一天起就把它当数学对象。

**漏掉偏置。** 只写 $\mathbf{w} \cdot \mathbf{x}$ 不写 $b$，决策边界被钉死在原点——分类面必须穿过原点，多数真实问题立刻无解。偏置把线性变成仿射(这条边界的正式名称见 [仿射空间与仿射映射](../linear-algebra/affine-spaces-and-maps/))，它不是细节，是表达能力的一部分。

**以为堆线性层能加深能力。** 结合律保证 $W_2(W_1\mathbf{x}) = (W_2W_1)\mathbf{x}$：没有激活函数，深层网络与单层等价。如果模型深而不强，先检查每一层之间是否真的有非线性。

**形状想当然。** 一个神经元：行向量乘列向量得一个数；一层：$W$ 是 $(m, n)$，行是神经元。把 $W$ 的行列意义记反，参数量账和代码形状会对不上——`nn.Linear(4, 2)` 的 `weight` 是 $(2, 4)$，不是 $(4, 2)$。老规矩：动手前先标形状。

## 相关词条

- [向量](../linear-algebra/vectors/)：输入与权重的数学身份
- [内积](../linear-algebra/inner-products/)：一个神经元的核心运算
- [矩阵乘法](../linear-algebra/matrix-multiplication/)：一层神经元的前向计算
- [为什么需要非线性](../neurons-and-activations/why-non-linearity/)：激活函数缺席的必然性论证，展开版
- [激活函数](../neurons-and-activations/activation-functions/)：$f$ 的候选清单与选择
- [阶跃与 sigmoid](../neurons-and-activations/heaviside-and-sigmoid/)：本篇例子的第一个激活函数
- [ReLU](../neurons-and-activations/relu/)：今天默认的激活函数
- [经典感知机](../linear-models/perceptron-classic/)：学习规则的第一次登场
- [万有逼近](../neurons-and-activations/universal-approximation/)：非线性复合的表达上限
- [MNIST 训练循环](../training-nn/mnist-mlp-training-loop/)：神经元组装成网络的实战
