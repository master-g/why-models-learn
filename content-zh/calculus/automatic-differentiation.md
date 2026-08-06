---
title: "自动微分:把链式法则装进计算图"
tags: ["why-models-learn"]
---

**自动微分**把一个程序拆成加法、乘法、指数等基本运算，在程序实际运行的数值点上保存每个节点的局部导数，再按链式法则组合出导数。它不是符号微分：不会把整个程序改写成一棵巨大代数表达式；也不是有限差分：不需要选择 $h$，不靠相近函数值相减。前向模式把一个输入方向送到输出，反向模式把一个输出敏感度拉回输入；这两个方向分别对应 Jacobian–向量积和向量–Jacobian 积。

## 三种“求导”方式先分开

设 $f(\boldsymbol x)$ 在当前点需要求导：

| 方法 | 做什么 | 主要代价或误差 |
| --- | --- | --- |
| 符号微分 | 改写表达式，生成导数表达式 | 表达式可能膨胀，需能操作符号 |
| 数值微分 | 用 $f(\boldsymbol x\pm h\boldsymbol e_i)$ 估计差商 | 有步长、舍入和函数调用误差 |
| 自动微分 | 沿实际执行的基本运算应用链式法则 | 需要可追踪计算图和每个算子的导数规则 |

自动微分的“自动”指的是程序替我们组织局部规则，不是绕开微积分。若基本算子的导数在当前点存在，AD 得到的是该计算路径上的链式法则结果，数值误差主要来自普通浮点运算。[数值微分](../calculus/numerical-differentiation/) 适合拿来检查 AD；AD 则适合在训练时反复计算高维参数梯度。

## 对偶数把值和导数放在一起

先看单个输入和一个方向。引入一个满足

$$
\epsilon^2=0
$$

的形式符号，把 $a$ 和方向导数 $a'$ 写成

$$
a+\epsilon a'
$$

这里的 $\epsilon$ 不是一个要取极限的普通小数；它的平方被规定为零。两个这样的量相加：

$$
(a+\epsilon a')+(b+\epsilon b')
=(a+b)+\epsilon(a'+b')
$$

相乘：

$$
\begin{aligned}
(a+\epsilon a')(b+\epsilon b')
&=ab+\epsilon(ab'+a'b)+\epsilon^2a'b'\\
&=ab+\epsilon(ab'+a'b)
\end{aligned}
$$

因为 $\epsilon^2=0$，乘法的 $\epsilon$ 系数正好是乘积法则。对一个可导标量函数：

$$
\phi(a+\epsilon a')
=
\phi(a)+\epsilon\phi'(a)a'
$$

这条规则把局部导数和当前值一起传下去。它不是把 $a'$ 当成“很小的数”后忽略高阶项，而是使用了一个专门让所有二阶项为零的代数。

取

$$
f(x)=x^2+3x+1
\qquad
x=2+\epsilon
$$

逐步计算：

$$
x^2=4+4\epsilon
\qquad
3x=6+3\epsilon
$$

所以

$$
f(2+\epsilon)
=
11+7\epsilon
$$

常数项 $11$ 是函数值，$\epsilon$ 的系数 $7$ 是

$$
f'(2)=2\cdot2+3=7
$$

对偶数一次运行就同时得到值和一个方向的导数。若输入不是标量而是 $\boldsymbol x+\epsilon\boldsymbol v$，得到的 $\epsilon$ 系数就是 $J_f(\boldsymbol x)\boldsymbol v$。

## 前向模式是 JVP

设

$$
F:\mathbb R^n\to\mathbb R^m
$$

在点 $\boldsymbol x$ 的 Jacobian 是 $J_F(\boldsymbol x)$。给一个输入方向 $\boldsymbol v$，前向模式传播的切向量满足

$$
\dot{\boldsymbol y}
=
J_F(\boldsymbol x)\boldsymbol v
$$

这个量叫 **Jacobian–vector product**，简称 JVP。它只回答一个方向的输出变化，不需要显式构造完整的 $m\times n$ Jacobian。

取

$$
F(u,v)
=
\begin{pmatrix}
u^2+3v\\
uv
\end{pmatrix}
\qquad
\boldsymbol x=
\begin{pmatrix}
1\\2
\end{pmatrix}
\qquad
\boldsymbol v=
\begin{pmatrix}
0.1\\0.3
\end{pmatrix}
$$

函数值为

$$
F(1,2)
=
\begin{pmatrix}
7\\2
\end{pmatrix}
$$

解析 Jacobian 是

$$
J_F(1,2)
=
\begin{pmatrix}
2&3\\
2&1
\end{pmatrix}
$$

所以前向切向量为

$$
J_F(1,2)\boldsymbol v
=
\begin{pmatrix}
2&3\\
2&1
\end{pmatrix}
\begin{pmatrix}
0.1\\0.3
\end{pmatrix}
=
\begin{pmatrix}
1.1\\0.5
\end{pmatrix}
$$

把 $\boldsymbol x+\epsilon\boldsymbol v$ 送入同一个运算过程，会在输出中得到

$$
F(\boldsymbol x+\epsilon\boldsymbol v)
=
F(\boldsymbol x)
+
\epsilon
\begin{pmatrix}
1.1\\0.5
\end{pmatrix}
$$

不需要把四个 Jacobian 元素先写出来；每个基本运算只需同时传递当前值和当前方向的切向量。

例如乘法节点 $z=ab$ 的前向规则是

$$
\dot z=\dot a\,b+a\,\dot b
$$

指数节点 $z=e^a$ 的规则是

$$
\dot z=e^a\dot a=z\dot a
$$

加法节点则把两条切向量相加。整张计算图沿输入到输出的方向执行这些局部规则，就是前向模式。

## 反向模式是 VJP

反向模式从一个输出敏感度开始。若 $F:\mathbb R^n\to\mathbb R^m$，给一个输出空间的行向量 $\boldsymbol u^{\mathsf T}$，反向传播

$$
\boldsymbol u^{\mathsf T}J_F(\boldsymbol x)
$$

这个量叫 **vector–Jacobian product**，简称 VJP。它把输出方向的加权变化拉回输入空间。若 $m=1$ 且 $\boldsymbol u=1$，VJP 就是标量函数的梯度转置。

用一个有分支的标量计算图：

$$
u=x^2
\qquad
v=3y
\qquad
w=xy
\qquad
L=u+v+w
$$

在 $x=1$、$y=2$：

$$
u=1
\qquad
v=6
\qquad
w=2
\qquad
L=9
$$

定义每个节点的反向敏感度

$$
\bar q=\frac{\partial L}{\partial q}
$$

从终点开始，先有

$$
\bar L=1
$$

加法节点把敏感度原样送到每个输入：

$$
\bar u=1
\qquad
\bar v=1
\qquad
\bar w=1
$$

再沿乘法节点反向：

$$
\bar x\mathrel{+}=\bar u(2x)
\qquad
\bar y\mathrel{+}=\bar v(3)
$$

以及

$$
\bar x\mathrel{+}=\bar w\,y
\qquad
\bar y\mathrel{+}=\bar w\,x
$$

代入当前值：

$$
\bar x=1\cdot2+1\cdot2=4
\qquad
\bar y=1\cdot3+1\cdot1=4
$$

直接展开

$$
L(x,y)=x^2+3y+xy
$$

也得到

$$
\nabla L(1,2)
=
\begin{pmatrix}
2x+y\\3+x
\end{pmatrix}_{(1,2)}
=
\begin{pmatrix}
4\\4
\end{pmatrix}
$$

反向模式的一次遍历同时给出 $x$ 和 $y$ 的梯度。它没有为每个输入分别运行一次完整函数，而是把共享的中间节点敏感度累加到各个前驱。

## 加法、乘法和共享节点的局部规则

把一个基本节点写成 $z=\phi(a,b)$，反向节点收到 $\bar z$ 后，对输入的贡献是

$$
\bar a\mathrel{+}=\bar z\frac{\partial z}{\partial a}
\qquad
\bar b\mathrel{+}=\bar z\frac{\partial z}{\partial b}
$$

常见规则为

| 前向节点 | 对 $a$ 的反向贡献 | 对 $b$ 的反向贡献 |
| --- | --- | --- |
| $z=a+b$ | $\bar z$ | $\bar z$ |
| $z=ab$ | $\bar z\,b$ | $\bar z\,a$ |
| $z=a^2$ | $\bar z\,2a$ | — |
| $z=\exp a$ | $\bar z\,\exp a$ | — |
| $z=\phi(a)$ | $\bar z\,\phi'(a)$ | — |

如果一个变量流向两个后继节点，它收到的反向梯度必须把两条路径的贡献相加。前面的 $x$ 同时进入 $u=x^2$ 和 $w=xy$，所以 $\bar x$ 是 $2+2$，不是只取其中一项。[广播与归约导数](../calculus/broadcast-and-reduction-derivatives/) 中的共享参数求和，是同一个局部伴随规则在形状复制上的表现。

## JVP 和 VJP 的形状

前向和反向都只是 Jacobian 的乘法方向不同：

$$
\operatorname{JVP}(\boldsymbol v)
=
J_F(\boldsymbol x)\boldsymbol v
$$

$$
\operatorname{VJP}(\boldsymbol u)
=
\boldsymbol u^{\mathsf T}J_F(\boldsymbol x)
$$

若上一个例子的 Jacobian 为

$$
J=
\begin{pmatrix}
2&3\\
2&1
\end{pmatrix}
$$

则输入方向 $\boldsymbol v=(0.1,0.3)^{\mathsf T}$ 的 JVP 是

$$
J\boldsymbol v=(1.1,0.5)^{\mathsf T}
$$

输出权重 $\boldsymbol u=(2,-1)^{\mathsf T}$ 的 VJP 是

$$
\boldsymbol u^{\mathsf T}J
=
\begin{pmatrix}
2&-1
\end{pmatrix}
\begin{pmatrix}
2&3\\
2&1
\end{pmatrix}
=
\begin{pmatrix}
2&5
\end{pmatrix}
$$

两者都不要求显式保存完整 Jacobian。JVP 的输入和输出形状分别是 $n$ 与 $m$；VJP 的输入权重和输出梯度形状分别是 $m$ 与 $n$。转置不是装饰，而是由内积和矩阵乘法方向决定的。

![自动微分的前向 JVP 与反向 VJP](/assets/calculus/svg/automatic-differentiation.1.svg)

## 为什么标量损失常用反向模式

设输入参数数量为 $n$，最终输出数量为 $m$。如果一次局部传播的成本与计算图大小同阶：

| 目标 | 前向模式 | 反向模式 |
| --- | --- | --- |
| 一个输入方向到许多输出 | 一次 JVP | 先构造输出权重，通常不划算 |
| 一个标量输出到许多参数 | 需要约 $n$ 个方向 | 一次 VJP 得到全部参数梯度 |
| 完整 $m\times n$ Jacobian | 约 $n$ 次 JVP | 约 $m$ 次 VJP |

神经网络通常有很多参数、一个标量损失，因此反向模式的计算次数更合适。若输入维度很小、输出维度很大，例如一个低维状态到许多观测的函数，前向模式可能更自然。模式的选择是输入和输出维度的选择，不是“反向永远更高级”。

反向模式需要正向阶段保留局部导数所需的中间值。例如 $z=a^2$ 的反向规则要用当前的 $a$。保存所有中间值会占用内存；检查点技术可以只保存部分节点，在反向时重新计算其余节点，用计算时间换内存。

## 神经网络中的一层反向传播

考虑线性层和逐分量激活：

$$
\boldsymbol z=W\boldsymbol x+\boldsymbol b
\qquad
\boldsymbol y=\Phi(\boldsymbol z)
$$

若上游给出列向量梯度 $\boldsymbol g_y=\nabla_{\boldsymbol y}L$，逐分量层的局部 Jacobian 是

$$
D_\Phi
=
\operatorname{diag}(\phi'(\boldsymbol z))
$$

所以

$$
\boldsymbol g_z
=
\boldsymbol g_y\odot\phi'(\boldsymbol z)
$$

线性层的微分为

$$
\mathrm d\boldsymbol z
=
(\mathrm dW)\boldsymbol x+W\,\mathrm d\boldsymbol x+\mathrm d\boldsymbol b
$$

对损失取微分：

$$
\mathrm dL
=
\boldsymbol g_z^{\mathsf T}
\left(
(\mathrm dW)\boldsymbol x+W\,\mathrm d\boldsymbol x+\mathrm d\boldsymbol b
\right)
$$

分别读出

$$
\nabla_WL
=
\boldsymbol g_z\boldsymbol x^{\mathsf T}
\qquad
\nabla_{\boldsymbol x}L
=
W^{\mathsf T}\boldsymbol g_z
\qquad
\nabla_{\boldsymbol b}L
=
\boldsymbol g_z
$$

批次输入时，$\boldsymbol b$ 会广播到每一行，所以 $\nabla_{\boldsymbol b}L$ 还要沿批次轴求和；权重梯度也会把每个样本的外积相加。这正是前面 JVP/VJP 规则、逐分量 Jacobian 和广播归约规则在一层网络中的组合。

## 正向值、反向值和计算图

自动微分系统通常需要区分三类信息：

- **正向值**：当前节点实际得到的数值，例如 $a$、$z$ 和激活输出；
- **局部规则所需的缓存**：例如乘法节点的另一个输入、平方节点的底数；
- **反向敏感度**：从最终输出传回当前节点的 $\bar q$。

正向模式把切向量和正向值一起传递，不一定需要保存整张反向 tape；反向模式先完成正向，再按逆拓扑顺序消费缓存。共享节点的反向敏感度必须累加，分支合并和广播合并都遵守这条规则。

如果在计算过程中显式切断一个节点和输入的依赖，后面的 AD 就不会沿被切断的边传播。把一个数值当成常量、调用 detach、在不支持梯度的自定义算子里返回普通数组，都会产生这种效果。它们有时是有意的停止梯度，有时是实现遗漏，检查时要区分。

## 高阶导数和 Hessian–向量积

自动微分可以再次作用在“已经包含导数的程序”上。对标量函数 $f(\boldsymbol x)$ 和方向 $\boldsymbol v$：

$$
H_f(\boldsymbol x)\boldsymbol v
=
\left.
\frac{\mathrm d}{\mathrm dt}
\nabla f(\boldsymbol x+t\boldsymbol v)
\right|_{t=0}
$$

这叫 Hessian–vector product，简称 HVP。它只计算 Hessian 作用在一个向量上的结果，不需要存储完整的 $n\times n$ Hessian。可以先反向得到梯度，再对这个梯度做一次前向方向传播，也可以按相反顺序组合两种模式。

如果只需要 Newton–CG、二阶方向搜索或曲率诊断，HVP 往往比显式 Hessian 更节省内存。若需要所有二阶偏导，仍然要付出与矩阵大小相应的计算或存储成本。[Hessian 矩阵](../calculus/hessian/) 的数值定义和这里的二阶 AD 是同一个局部对象。

## 分支、非光滑算子和自定义算子

AD 沿实际执行的路径传播导数。对

$$
f(x)=
\begin{cases}
x^2,&x>0\\
0,&x\le0
\end{cases}
$$

在 $x>0$ 时，当前路径的局部导数是 $2x$；在 $x<0$ 时是 $0$。在 $x=0$，经典导数是否存在要先由数学定义判断，框架选择的次梯度只是实现约定。

常见边界包括：

- max 的并列最大值需要指定梯度分配；
- 整数索引对索引本身通常不可微，但可以对被取出的值传播；
- 控制流改变路径时，AD 给出当前路径的导数，不会自动替你分析另一条路径；
- 原地修改可能覆盖反向需要的正向值；
- 自定义算子如果没有 JVP 或 VJP 规则，系统可能把它当成常量、拒绝求导，或使用一个不适合的近似；
- 复数、稀疏结构和外部黑盒调用需要各自的导数接口。

遇到“梯度是零”时，先问是数学上的局部导数为零，还是计算图边被切断、分支没有经过、参数没有参与当前输出。

## 自动微分和有限差分如何互相检查

对 AD 输出的梯度，有限差分仍然是小规模的外部检查器。以

$$
L(x,y)=x^2+3y+xy
$$

在 $(1,2)$ 为例，反向模式给出 $(4,4)^{\mathsf T}$。中心差分取 $h=10^{-3}$：

| 参数 | AD 梯度 | 中心差分 |
| --- | --- | --- |
| $x$ | $4$ | $4.000000$ |
| $y$ | $4$ | $4.000000$ |

这个检查不能证明整个框架没有错误，但能在一个具体点抓住转置、符号、共享节点累加和 reduction 约定的问题。对大模型应先缩成二维输入、关闭随机状态、固定损失 reduction，再逐步增加复杂度。[数值微分](../calculus/numerical-differentiation/) 篇说明了步长和相对误差的选择。

## 常见失效模式

- **把 AD 当成有限差分。** AD 不使用 $h$，而是逐节点应用局部链式规则。
- **正向值丢失。** 反向节点常需要正向缓存，例如乘法的另一个输入和激活值。
- **共享节点只回传一条路径。** 一个变量进入多个后继时，所有反向贡献都要相加。
- **把 JVP 和 VJP 的方向混写。** JVP 是 $J\boldsymbol v$，VJP 是 $\boldsymbol u^{\mathsf T}J$，形状先决定方向。
- **矩阵转置放错位置。** 线性层的输入梯度是 $W^{\mathsf T}\boldsymbol g_z$，不是 $W\boldsymbol g_z$。
- **广播参数梯度忘记归约。** 批次偏置的梯度要沿被复制的轴求和。
- **把不可导点的实现约定当成定理。** ReLU、max 和分支边界的零梯度或次梯度需要注明约定。
- **自定义算子没有局部导数规则。** 先确认它是否提供了与前向实现一致的 JVP/VJP。

## 相关词条

- [链式法则](../calculus/chain-rule/)：所有局部导数传播的标量基础。
- [向量链式法则](../calculus/vector-chain-rule/)：JVP 和 VJP 的矩阵形状来源。
- [计算图](../backpropagation/computational-graphs/)：把程序组织成可传播导数的节点和边。
- [梯度](../calculus/gradient/)：标量损失对参数的反向敏感度。
- [雅可比矩阵](../calculus/jacobian/)：JVP/VJP 所作用的局部矩阵。
- [逐分量导数](../calculus/elementwise-derivatives/)：激活节点的对角 Jacobian。
- [广播与归约导数](../calculus/broadcast-and-reduction-derivatives/)：批次参数的梯度复制与累加。
- [数值微分](../calculus/numerical-differentiation/)：独立检查自动微分结果。
- [Hessian 矩阵](../calculus/hessian/)：高阶自动微分和 HVP 的二阶对象。
