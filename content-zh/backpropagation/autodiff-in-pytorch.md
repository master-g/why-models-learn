---
title: "PyTorch 自动微分：实际运行的计算图与梯度"
tags: ["why-models-learn"]
---

PyTorch 自动微分是沿着一次真实前向执行记录的运算，反向应用链式法则并计算张量梯度的机制。它不是把 Python 源码先翻译成一张包含所有可能分支的静态公式，也不是优化器；`requires_grad` 决定哪些张量需要被追踪，前向运算产生 `grad_fn` 和反向所需的中间值，`backward()` 或 `torch.autograd.grad()` 再把输出敏感度传回输入。本文用一个标量图把这些对象对齐到手算公式，再展开叶张量、梯度累加、图生命周期、非标量输出、动态控制流、自定义 `Function`、`no_grad` 与 `detach` 的边界，并给出一份可复制的运行审计代码。

![PyTorch 自动微分在一次前向执行中记录 z、a、L，反向从损失把梯度传回 w、b 和 x](/assets/backpropagation/svg/autodiff-in-pytorch.1.svg)

## 自动微分究竟自动了什么

设模型在一次固定输入和参数下实现一个函数

$$
L=F(\boldsymbol\theta,\boldsymbol x).
$$

自动微分不通过有限差分反复扰动 $\boldsymbol\theta$，也不要求先手写整个函数的展开式。它把 $F$ 执行时经过的加法、乘法、矩阵乘法、激活函数和归约等基本运算接起来；每个基本运算都带有局部导数规则，反向阶段从 $L$ 的上游敏感度出发，按相反的依赖顺序使用这些规则。

这和 [梯度检查](../backpropagation/gradient-checking/) 的关系要分开：梯度检查把参数改成 $\theta_j+h$ 和 $\theta_j-h$，用函数值差商近似导数；自动微分只执行一次前向，再沿这次执行的计算图回传。前者是诊断方法，后者是训练时的导数计算方法。一个自动微分实现可以很快地算出错误梯度，所以仍然需要有限差分或小规模手算做独立核对。

PyTorch 的自动微分引擎通常被称为 autograd。官方机制说明把它描述为反向自动微分：前向执行同时建立表示运算历史的图，反向从输出节点沿图回到叶节点。这里的“图”不是一个脱离运行的数学承诺，而是这一轮实际执行经过的那一部分；下一轮前向会重新建立自己的图。可参阅 [PyTorch 的 Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)。

## 一个标量图：先按数学式子算一遍

取一个输入、一个权重和一个偏置：

$$
x=2,
\qquad
w=0.5,
\qquad
b=-0.2,
\qquad
y=0.25.
$$

定义带 tanh 激活的平方损失：

$$
z=wx+b,
\qquad
a=\tanh(z),
\qquad
L=\frac12(a-y)^2.
$$

前向值逐步是

$$
z=0.5\times2-0.2=0.8,
$$

$$
a=\tanh(0.8)\approx0.664036770268,
$$

$$
L=\frac12(0.664036770268-0.25)^2
\approx0.085713223567.
$$

把 $\delta$ 记作损失对预激活的导数。先穿过平方损失：

$$
\frac{\partial L}{\partial a}=a-y.
$$

再穿过 tanh：

$$
\frac{\partial a}{\partial z}=1-\tanh^2(z)=1-a^2,
$$

$$
\delta
:=\frac{\partial L}{\partial z}
=(a-y)(1-a^2)
\approx0.231469396049.
$$

最后穿过仿射节点：

$$
\frac{\partial L}{\partial w}=\delta x
\approx0.462938792099,
\qquad
\frac{\partial L}{\partial b}=\delta
\approx0.231469396049,
$$

$$
\frac{\partial L}{\partial x}=\delta w
\approx0.115734698025.
$$

图上的每条边只需要一个局部导数：$\partial z/\partial w=x$、$\partial z/\partial b=1$、$\partial z/\partial x=w$，以及 $\partial a/\partial z=1-a^2$。反向阶段把上游的 $\partial L/\partial a$ 乘进来，就得到这三条参数和输入梯度。PyTorch 自动微分要做的事情正是维护这组依赖与局部规则，而不是替代这套链式法则。

## `requires_grad`、叶张量和 `grad_fn`

PyTorch 中“一个张量需要梯度”至少涉及三个不同问题：是否允许 autograd 追踪它参与的运算、它是否是计算图的叶、反向之后梯度放在哪里。不能把它们都压缩成“张量有一个梯度字段”。

| 对象或属性 | 它回答的问题 | 标量例子中的位置 |
| --- | --- | --- |
| `requires_grad` | 以它为输入的后续运算是否需要被记录 | `w`、`b` 可设为 `True` |
| `is_leaf` | 这个张量是不是由当前图中的运算产生的叶节点 | 手工创建的 `w` 是叶，`z` 不是 |
| `grad_fn` | 这个结果由哪个已记录运算产生 | `z`、`a`、`loss` 通常非空 |
| `.grad` | 反向后把梯度保存在哪里 | 叶参数默认保存，非叶默认不保存 |
| `.retain_grad()` | 是否要求非叶张量也保留反向梯度 | 调试 `a` 或 `z` 时显式调用 |

如果至少有一个输入需要梯度，且当前处在默认 grad mode，PyTorch 会为产生的结果记录反向历史。因此即使 `x` 是常量，`z=w*x+b` 仍然需要追踪，因为它依赖 `w` 和 `b`。一个结果张量的 `grad_fn` 可以看成通向反向图的入口；它不是一个供业务代码依赖的稳定字符串，也不应拿具体类名做版本兼容判断。

叶张量通常指不是由本图运算产生的输入。把 `requires_grad=True` 直接放在 `torch.tensor`、`torch.randn` 等构造操作上得到的参数是叶张量；对它做乘法或加法后得到的结果是非叶张量。反向时，叶张量的 `.grad` 默认会被填入；非叶张量即使参与了梯度计算，也不会默认把梯度留在自己的 `.grad` 属性里。要检查中间激活，应在反向前写：

```python
a.retain_grad()
loss.backward()
print(a.grad)
```

`retain_grad()` 不会改变数学梯度，也不会让这个张量变成参数；它只改变调试时的保存行为，并可能增加内存占用。

### 同一个名字不等于同一个图节点

下面两次赋值的 `z` 不是一个节点：

```python
z = w * x + b
loss = 0.5 * (torch.tanh(z) - y).square()

z = w * x + b
other_loss = (z - 1.0).square()
```

第二次赋值只是 Python 名字重新指向了另一个前向结果。第一张图仍可能被 `loss` 引用，直到它反向完成或相关对象释放。调试时要记录张量的值、形状、dtype、device 和 `grad_fn`，不能只看变量名。

## `backward()`：从标量根节点往回累加

对标量损失调用

```python
loss.backward()
```

可以理解为把根节点的上游梯度设成 $1$，然后计算所有图叶节点对这个标量的梯度。对上面的例子，根节点满足

$$
\frac{\partial L}{\partial L}=1.
$$

之后每条局部边都接收已经从后方传来的敏感度。PyTorch 不需要用户把 $\partial L/\partial L=1$ 手动传入标量 `loss`；对非标量输出则不能省略上游向量，后文会单独说明。

### 梯度默认是累加的

`.backward()` 默认把本次结果加到叶张量已有的 `.grad` 上。设连续两次前向得到 $L_1$、$L_2$，若不清空梯度，则

$$
\texttt{w.grad}
\leftarrow
\texttt{w.grad}
+\frac{\partial L_1}{\partial w}
+\frac{\partial L_2}{\partial w}.
$$

这在梯度累积训练中有意为之，在普通的逐 batch 更新中则需要每轮显式清理：

```python
optimizer.zero_grad(set_to_none=True)
loss = model(x).sub(target).square().mean()
loss.backward()
optimizer.step()
```

`set_to_none=True` 让梯度字段回到 `None`，而不是先分配一个全零张量；它可以减少一次清零写入，也让“本轮是否产生了梯度”更容易观察。无论选哪种清理方式，关键是把清理时机固定下来，并知道是否正在做多步梯度累积。

### `backward()` 与参数更新是两件事

`loss.backward()` 只计算并累加梯度；它不会修改 `w`、`b` 的数值。`optimizer.step()` 才根据梯度和优化器状态更新参数。一次常见训练步的顺序是：

1. 清除上一轮残留梯度；
2. 用当前参数做一次前向；
3. 对当前损失反向；
4. 用同一轮产生的梯度执行更新。

如果在反向还没结束时先更新某一层，再用新参数继续计算另一层梯度，就把同一张前向图与两个参数点混在一起。这个错误不一定让形状报错，却会让梯度不再是同一个目标在同一个参数点的导数。

## `torch.autograd.grad`：需要返回值时不要污染 `.grad`

`torch.autograd.grad(outputs, inputs)` 计算并返回指定输入的梯度，默认不会把结果累加到这些输入的 `.grad` 属性。这适合梯度惩罚、元学习、Hessian–向量积和只想查询某一组输入的场景：

```python
g_w, g_b = torch.autograd.grad(loss, (w, b))
print(g_w, g_b)
```

和 `backward()` 的差别可以写成表：

| 调用 | 梯度去处 | 适合的用途 |
| --- | --- | --- |
| `loss.backward()` | 累加到图中叶张量的 `.grad` | 常规训练和优化器更新 |
| `torch.autograd.grad(loss, w)` | 作为返回值交给调用者 | 局部查询、高阶导数、梯度惩罚 |
| `loss.backward(v)` | 用 `v` 作为输出上游梯度并累加 | 非标量输出的指定 VJP |

二者都需要这张图仍然存在。如果同一前向结果已经执行过一次反向，默认情况下反向所需的中间缓存会被释放；此时再次对同一结果求导通常会报“图已经被释放”一类错误。与其盲目设置 `retain_graph=True`，更常见、更省内存的做法是重新做一次前向。只有确实需要在同一图上多次反向时才保留图，并把增加的内存成本写进调试记录。

## `create_graph=True`：把一阶导数也变成可微结果

默认反向得到的梯度通常是一个数值结果，下一次求导不再追踪“梯度是怎样算出来的”。要算高阶导数，需要在第一次求导时构建导数图：

```python
w = torch.tensor(2.0, dtype=torch.double, requires_grad=True)
f = 0.5 * (w - 3.0).square()
(g,) = torch.autograd.grad(f, w, create_graph=True)
(h,) = torch.autograd.grad(g, w)
print(f.item(), g.item(), h.item())  # 0.5, -1.0, 1.0
```

数学上

$$
f(w)=\frac12(w-3)^2,
\qquad
f'(2)=-1,
\qquad
f''(2)=1.
$$

`create_graph=True` 的含义是“记录求导过程产生的运算”，不是“自动生成完整 Hessian”。如果只需要 $H\boldsymbol v$，可以先求方向上的一阶导数，再对它求导；这样通常比显式构造 $H$ 少很多内存。`retain_graph` 与 `create_graph` 解决的不是同一个问题：前者保留当前反向要再次使用的原图，后者让导数结果拥有可继续反向的图。

## 非标量输出：反向默认需要一个上游向量

若 `output` 是向量 $\boldsymbol y=f(\boldsymbol x)$，每个输出坐标对每个输入坐标组成 Jacobian：

$$
J_{y,x}
=\frac{\partial\boldsymbol y}{\partial\boldsymbol x}.
$$

一个反向过程并不一定返回整个 $J$。给定输出空间的上游向量 $\boldsymbol v$，它计算

$$
\frac{\partial(\boldsymbol v^{\mathsf T}\boldsymbol y)}
{\partial\boldsymbol x}
=J_{y,x}^{\mathsf T}\boldsymbol v.
$$

这就是反向模式常见的向量–Jacobian 积写法。调用接口中，`grad_outputs` 或 `backward(gradient=...)` 的 `gradient` 参数就是这个 $\boldsymbol v$：

```python
x = torch.tensor([2.0, 3.0], dtype=torch.double, requires_grad=True)
y = torch.stack((x[0] * x[1], torch.exp(x[0]) + x[1]))
v = torch.tensor([0.5, -1.0], dtype=torch.double)
(vjp,) = torch.autograd.grad(y, x, grad_outputs=v)
print(vjp)
```

本例的 Jacobian 是

$$
J=
\begin{pmatrix}
3 & 2\\
e^2 & 1
\end{pmatrix},
$$

取 $\boldsymbol v=(0.5,-1)$，得到

$$
J^{\mathsf T}\boldsymbol v
=
\begin{pmatrix}
1.5-e^2\\
1-1
\end{pmatrix}
\approx
\begin{pmatrix}
-5.889056098931\\
0
\end{pmatrix}.
$$

如果把 `v` 换成全 1 向量，得到的是 $\sum_i y_i$ 对 $x$ 的梯度。很多“非标量不能直接 backward”的报错，本质上是在提醒调用者：请明确你到底要哪一个输出组合的导数。

### 需要整张 Jacobian 时再逐行或用变换

整张 Jacobian 的大小是输出元素数乘输入元素数，可能比一个 VJP 大得多。当前 PyTorch 的 `torch.func` 提供 `vjp`、`jvp`、`jacrev`、`jacfwd` 和 `hessian` 等函数变换：`vjp` 返回函数值以及一个可接收上游向量的闭包，`jvp` 沿输入方向传播，`jacrev` 用反向模式构造 Jacobian，`jacfwd` 用前向模式构造 Jacobian。具体 API 见 [torch.func reference](https://docs.pytorch.org/docs/stable/func.api)。选择哪一种仍取决于输入、输出的相对维度与内存预算；名字中有 `jac` 不代表应该在生产训练中无条件生成稠密矩阵。

## 前向图是动态的：运行了哪条分支就微分哪条分支

考虑一个带 Python 分支的函数：

```python
def f(x):
    if x.item() > 0:
        return x.square()
    return 3.0 * x
```

当 `x=2` 时，这一轮记录的是 $x\mapsto x^2$ 的运算，导数为 $2x=4$；当下一轮输入 `x=-2` 时，这一轮记录的是 $x\mapsto3x$，导数为 $3$。autograd 不会把两个分支拼成一个同时执行的公式，也不会对 `x.item() > 0` 这条 Python 布尔判断求导。分支条件决定了本轮走哪张图，图内可微运算决定了本轮的局部导数。

这就是动态计算图的工程含义：循环次数、分支路径、临时张量形状可以随输入改变。代价是调试时必须把“这一轮实际走了哪条路径”记录下来；只看模型源码而不看本轮输入和控制流，可能把正确的分支差异误判为梯度不稳定。

`x.item()` 还会把一个张量标量取回 Python 数值。在 GPU 上它可能引入设备同步，在分支条件里也明确地切断了这部分数值到后续图的连接。若条件本身需要可微，不能用 Python `if` 假装成一个可微门，而应选择合适的张量运算或连续近似，并重新定义数学目标。

## 保存的中间值：反向为什么需要前向缓存

很多局部导数需要前向中间值。例如 tanh 的导数可以用前向已经算出的 $a=\tanh(z)$：

$$
\frac{\partial\tanh(z)}{\partial z}=1-a^2.
$$

平方运算则可能保存输入 $x$，以便反向计算 $2x$。因此一次前向不只是产出最终输出，还可能留下若干供反向读取的 saved tensors。网络更深、更宽、序列更长时，激活缓存会成为显存的重要来源。

这也解释了两个常见现象：

- 第一次反向结束后不能无条件对同一 `loss` 再次 backward，因为引擎默认释放已经用过的缓存；
- 用 `torch.no_grad()` 或 `detach()` 得到的结果没有完整反向历史，之后再从它开始求原参数梯度就会断开。

激活检查点把策略改成“只保存少数节点，反向前重算其余前向”，用额外计算换取更低内存。它改变执行安排，不改变在没有随机状态和原地修改问题时的数学函数；若重算路径含随机算子，则还要保证随机状态协议一致。

## 自定义 `Function`：只有内置算子不够时才接管局部导数

如果一个新操作可以用现有 PyTorch 算子直接组合，通常让 autograd 记录这些内置算子更安全。只有当操作包含不可由 PyTorch 追踪的 NumPy/C++ 逻辑、需要专门的反向公式或确实需要合并保存缓存时，才考虑自定义 `torch.autograd.Function`。

一个平方操作的最小自定义版本如下：

```python
import torch

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2.0 * x

x = torch.tensor(2.0, dtype=torch.double, requires_grad=True)
y = Square.apply(x)                 # 通过 apply 调用,不要直接调用 forward
y.backward()
print(y.item(), x.grad.item())      # 4.0, 4.0
```

前向定义 $y=x^2$，局部导数是 $2x$。`grad_output` 是从后继节点传入的上游敏感度；如果 `Square` 后面还有一个乘法，它不一定等于 $1$。因此 `backward` 返回的是

$$
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}
=\texttt{grad\_output}\cdot2x,
$$

不是只返回裸的 $2x$。如果 `forward` 有多个 Tensor 输入，`backward` 必须按输入顺序返回对应梯度；对不需要梯度或非 Tensor 参数，可以返回 `None`。

`ctx.save_for_backward()` 应用于需要在反向使用的 Tensor，而普通元数据可以保存到 `ctx` 的其他属性。直接把大量 Tensor 挂到 `ctx` 可能让它们比必要时间活得更久。自定义反向必须用 `torch.autograd.gradcheck()` 或本文 [梯度检查](../backpropagation/gradient-checking/) 的中心差分协议在双精度、远离不可导边界的输入上核对；“代码能运行”只说明接口形状对上，不说明导数公式对。

## `no_grad`、`inference_mode`、`detach` 和 `eval` 不在同一层

这四个名字都常出现在“推理不需要梯度”的代码里，但它们控制的对象不同：

| 机制 | 改变什么 | 会不会让模型进入评估态 |
| --- | --- | --- |
| `torch.no_grad()` | 这一段运算不记录反向图 | 不会 |
| `torch.inference_mode()` | 更严格地关闭推理期 autograd 记录并减少相关开销 | 不会 |
| `tensor.detach()` | 从某个 Tensor 到它前驱图的梯度连接处切断 | 不会 |
| `model.eval()` | 把模块切换到 evaluation 行为 | 不会自动关闭 autograd |

训练完成后的推理通常同时需要：

```python
model.eval()
with torch.inference_mode():
    prediction = model(x)
```

`eval()` 影响 dropout、batch normalization 等模块的训练/评估行为；它不是一个全局梯度开关。反过来，`no_grad()` 也不会替你把 dropout 切到评估行为。只写其中一个，可能让模型行为或内存占用与预期不一致。

`detach()` 适合把一段结果当成常量交给后续损失、缓存或日志：

```python
features = encoder(x)
frozen_features = features.detach()
score = head(frozen_features)
```

此时 `score` 对 `head` 参数仍可以有梯度，但梯度不会穿过 `frozen_features` 回到 `encoder`。如果本意是冻结 encoder 的训练更新，最好同时清楚地区分参数的 `requires_grad`、前向的 `no_grad` 和优化器持有的参数列表；一个意外的 `detach` 可能让训练看似正常、实际只有后半段在学习。

## 运行方法

下面脚本把标量图、叶张量、中间梯度、`autograd.grad` 和高阶导数放在一起。它没有依赖随机数，输出应当与正文的独立双精度数字相符；`grad_fn is not None` 只检查“有记录”，不比较不同版本中的具体类名。

```python
import torch

torch.set_printoptions(precision=12)

# 标量计算图
x = torch.tensor(2.0, dtype=torch.double, requires_grad=True)
w = torch.tensor(0.5, dtype=torch.double, requires_grad=True)
b = torch.tensor(-0.2, dtype=torch.double, requires_grad=True)
y = torch.tensor(0.25, dtype=torch.double)

z = w * x + b
a = torch.tanh(z)
a.retain_grad()
loss = 0.5 * (a - y).square()

print("values", z.item(), a.item(), loss.item())
print("graph", x.is_leaf, w.is_leaf, z.is_leaf, z.grad_fn is not None)
loss.backward()
print("grads", x.grad.item(), w.grad.item(), b.grad.item(), a.grad.item())

# autograd.grad 返回值,不把查询结果写入另一个输入的 .grad
q = torch.tensor(2.0, dtype=torch.double, requires_grad=True)
f = 0.5 * (q - 3.0).square()
(first,) = torch.autograd.grad(f, q, create_graph=True)
(second,) = torch.autograd.grad(first, q)
print("higher", f.item(), first.item(), second.item())
```

本次环境的独立数值核验结果为：

```text
values z=0.800000000000 a=0.664036770268 loss=0.085713223567
grads dx=0.115734698025 dw=0.462938792099 db=0.231469396049 da=0.414036770268
higher f=0.500000000000 first=-1.000000000000 second=1.000000000000
```

这里的 `da` 是 $\partial L/\partial a=a-y$，而 `dz` 是 $\partial L/\partial z=0.231469396049$；代码打印 `a.grad`，所以它应对应 `da`。正文给出的 `dx`、`dw`、`db` 与这组值来自同一个标量函数。

本机当前没有安装 `torch`，临时环境通过 `uv` 获取 PyTorch 又被当前网络沙箱的 DNS 阻断，因此上面的代码块保留为可复制的 PyTorch 运行方法，文本中的这组数值明确标为独立双精度等价核验，不冒充本机 `torch` 的运行输出。真正需要确认版本特有行为时，应在安装目标 PyTorch 版本的环境中重新运行，并保存版本号、dtype、设备和完整输出。

### 运行输出还应该检查什么

只打印最终 `.grad` 不足以审计自动微分。建议在小张量上同时记录：

| 记录项 | 为什么要记录 |
| --- | --- |
| `torch.__version__` | API 与错误信息可能随版本变化 |
| `dtype`、`device` | 精度、设备同步和算子支持影响数值与性能 |
| 输入与参数的 `requires_grad` | 确认梯度入口没有被关闭 |
| 中间值的 `shape` 与 `grad_fn` 是否存在 | 确认图连到了预期位置 |
| `loss.ndim` 与 reduction | 确认标量根节点和 batch 缩放一致 |
| `grad is None`、梯度范数和非零比例 | 区分未连接、零梯度与数值下溢 |
| 随机种子、训练态/推理态 | 重现 dropout、增强和 batch 统计行为 |

对一个新自定义算子，还应在 double dtype、非边界输入和固定形状上跑 `gradcheck`；对需要二阶导数的算子再跑 `gradgradcheck`。这些检查覆盖局部导数，不覆盖标签语义、损失是否写对或模型是否学到了想要的任务。

## 失效模式：看到“梯度不对”先定位断点

### 参数的 `requires_grad` 没有打开

如果参数在创建时是 `requires_grad=False`，后续结果可能没有反向历史；若在 `no_grad` 或 `inference_mode` 中创建了关键中间量，也不会因为它的输入曾经需要梯度就自动恢复图。先检查梯度入口，再看损失。

### 误把非叶张量的 `.grad is None` 当成梯度没算

非叶张量默认不保留 `.grad`，但它的梯度可能已经被用于继续回传。对需要观察的激活调用 `retain_grad()`，不要为了让日志非空而把所有中间量都保留。

### 忘记清理，或者把梯度清理得太早

连续 batch 之间不清理会得到梯度累加；在 `optimizer.step()` 之前清理则会把刚算出的梯度丢掉。多步累积时，清理频率应与目标的平均/求和约定一起记录，不能只看代码缩进。

### 非标量输出没有给上游向量

`vector.backward()` 需要知道你要计算哪个输出组合的导数。若目标是所有坐标之和，可以传 `torch.ones_like(vector)`；若目标是某个加权和，传对应权重。不要为了消除报错随便传全 1，先写清楚损失或 VJP 的定义。

### 在错误的地方 `detach()` 或使用 `.data`

`detach()` 是有意切断图的操作，不是“把 Tensor 转成普通数”的无害写法。历史上的 `.data` 还可能绕过 autograd 的版本检查，导致保存给反向的值被悄悄原地修改；现代代码应使用明确的 `detach()` 或 `no_grad()`，并避免用原地操作改写反向需要的缓存。

### 原地修改了为反向保存的值

如果一个算子保存了前向中间量，之后的原地修改会让反向读到不一致的数据。PyTorch 通常会通过版本计数在 backward 时报错，这个报错是在保护梯度正确性，不应简单地用 `retain_graph` 或 `allow_unused` 压过去。先定位是哪一个张量被原地写入，再改成 out-of-place 运算或重新安排缓存生命周期。

### 训练态、推理态和梯度开关混用

`model.eval()` 不等于 `no_grad()`，`no_grad()` 也不等于 `detach()`。dropout 的随机性、batch normalization 的统计量、参数是否接收梯度、输出是否保留图，分别属于模块状态、grad mode、图连接和张量属性四个层次。排查时把四层分别记录，别用一个“eval 已调用”解释全部现象。

### 对同一图重复反向，或者为了避免报错永久保留图

重复 backward 的报错通常说明前向结果被复用了，或者高阶导数的图生命周期没有设计清楚。`retain_graph=True` 会增加内存，并不能修复错误的参数更新、错误的损失归约或断开的 `detach`。优先重新前向；只有同一图确实需要多次反向时才保留它。

### 以为 autograd 验证了数学意图

autograd 只会对实际执行的程序求导。如果把标签错位、mask 乘错、mean 写成 sum、logits 重复过 sigmoid 或参数送进了错误分支，自动微分仍会忠实地对这个错误函数求导。把前向数值、损失定义、独立手算、中心差分和梯度统计放在同一份审计记录中，才能区分“引擎没算对”和“程序本来就写成了另一个函数”。

## 小结

PyTorch 自动微分可以按四层理解：

1. `requires_grad` 决定从哪些张量开始记录；
2. 前向执行产生本轮实际经过的动态计算图、`grad_fn` 和反向缓存；
3. `backward()` 或 `autograd.grad()` 沿图应用局部链式法则，得到 VJP 或指定输入梯度；
4. `.grad` 的累加、图的释放、`no_grad`、`detach`、模块态和优化器更新决定结果如何被使用。

对标量损失，`backward()` 的根上游梯度是 $1$；对向量输出，必须提供想要的上游向量。叶张量默认保存 `.grad`，非叶张量要观察则显式 `retain_grad()`。高阶导数需要 `create_graph=True`，同一图重复使用才考虑 `retain_graph=True`。这些接口都不改变链式法则本身，只把计算图的生命周期和梯度存放位置交给程序员管理。

## 相关词条

- [自动微分](../calculus/automatic-differentiation/)：从导数模式、对偶数和 JVP/VJP 解释自动微分
- [计算图](../backpropagation/computational-graphs/)：把复合函数拆成节点、边与拓扑序
- [向量链式法则](../calculus/vector-chain-rule/)：统一 Jacobian 的方向与乘法顺序
- [反向传播](../backpropagation/backpropagation/)：手工推导同一反向模式的局部规则
- [前向计算](../backpropagation/forward-pass/)：先算哪些值、缓存哪些中间结果
- [梯度检查](../backpropagation/gradient-checking/)：用有限差分独立核对自动微分结果
- [梯度消失与爆炸](../backpropagation/vanishing-and-exploding/)：解释多层反向连乘的数值后果
- 激活检查点：以重算换取反向缓存内存
