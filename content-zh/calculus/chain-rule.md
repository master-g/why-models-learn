---
title: "链式法则:复合函数如何传递变化率"
tags: ["why-models-learn"]
---

**链式法则**处理复合函数的导数。若先用 $g$ 把 $x$ 变成 $u=g(x)$，再用 $f$ 把 $u$ 变成 $y=f(u)$，那么输入的一个小变化要先经过内层，再经过外层：

$$
\frac{\mathrm dy}{\mathrm dx}
=\frac{\mathrm dy}{\mathrm du}
\frac{\mathrm du}{\mathrm dx}
$$

写成函数记号是

$$
(f\circ g)'(x)
=f'(g(x))g'(x)
$$

本篇从局部线性近似推导这条规则，区分复合与乘积，核对两层和三层数字例子，再把同一乘积读成神经网络计算图里的反向传播。多变量版本只说明结构，正式的偏导、梯度和向量链式法则留给后续词条。

## 先分清复合和乘积

复合函数

$$
(f\circ g)(x)=f(g(x))
$$

是“把 $g$ 的输出送进 $f$”。乘积

$$
(fg)(x)=f(x)g(x)
$$

是“在同一个输入上分别算 $f$ 和 $g$，再把两个结果相乘”。它们的导数完全不同：

$$
(f\circ g)'=f'(g)g'
\qquad
(fg)'=f'g+fg'
$$

取 $f(u)=u^2$、$g(x)=x+1$：

$$
(f\circ g)(x)=(x+1)^2
\qquad
(fg)(x)=x^2(x+1)
$$

前者只有一条输入到输出的路径，要乘内外层斜率；后者有两个因素同时变化，要把两条变化路径相加。

## 从局部线性近似推导

固定输入点 $a$，记内层输出为

$$
b=g(a)
$$

导数的局部线性近似告诉我们：

$$
g(a+h)=b+g'(a)h+o(h)
$$

令

$$
k=g(a+h)-g(a)
$$

那么 $k=g'(a)h+o(h)$，并且 $k\to0$。在外层点 $b$，同样有

$$
f(b+k)=f(b)+f'(b)k+o(k)
$$

把内层变化代进去：

$$
\begin{aligned}
f(g(a+h))
&=f(b)+f'(b)\big(g'(a)h+o(h)\big)+o(k)\\
&=f(b)+f'(b)g'(a)h+o(h)
\end{aligned}
$$

最后一步使用了 $k=O(h)$：外层剩余项满足 $o(k)/k\to0$，而 $k/h$ 有限，所以 $o(k)/h\to0$。若某个小步长恰好让 $k=0$，外层剩余项也是零，不影响这个结论。

因此复合函数在 $a$ 的一阶项系数就是

$$
(f\circ g)'(a)=f'(g(a))g'(a)
$$

这里的乘法不是把两个分数符号机械约掉，而是两个局部线性变化连续作用后的复合：内层把 $h$ 变成 $g'(a)h$，外层再把它变成 $f'(b)g'(a)h$。

## 两层复合的数字例子

取

$$
g(x)=3x+1
\qquad
f(u)=u^2
\qquad
y=f(g(x))=(3x+1)^2
$$

在 $x=2$：

$$
u=g(2)=7
\qquad
g'(2)=3
\qquad
f'(u)=2u=14
$$

所以链式法则给出

$$
y'(2)=f'(7)g'(2)=14\cdot3=42
$$

直接展开 $y=9x^2+6x+1$，再求导得到 $y'=18x+6$，代入 $x=2$ 也是 $42$。两种算法的角色不同：展开是这一个例子的核对，链式法则才是表达式变长时仍然可复用的算法。

![两层复合把局部变化率相乘：x 经过内层 g 再经过外层 f](/assets/calculus/svg/chain-rule.1.svg)

另一个例子把外层换成对数：

$$
y=\ln(x^2+1)
$$

在 $x=1$，内层 $u=x^2+1$ 的值为 $2$，内层导数为 $2$，外层 $\ln u$ 在 $u=2$ 的导数为 $1/2$，所以

$$
y'(1)=\frac12\cdot2=1
$$

如果只看外层并写成 $1/(x^2+1)$，会得到 $1/2$，漏掉了内层 $x^2+1$ 在 $x=1$ 的变化率。

## 三层复合：每一层都传一次

若

$$
x\longmapsto u(x)
\longmapsto v(u)
\longmapsto y(v)
$$

就要连续乘三层局部斜率：

$$
\frac{\mathrm dy}{\mathrm dx}
=\frac{\mathrm dy}{\mathrm dv}
\frac{\mathrm dv}{\mathrm du}
\frac{\mathrm du}{\mathrm dx}
$$

取

$$
y=\sin(e^{2x})
$$

分层写成

$$
u=2x
\qquad
v=e^u
\qquad
y=\sin v
$$

在 $x=0$：

$$
\frac{\mathrm du}{\mathrm dx}=2
\qquad
\frac{\mathrm dv}{\mathrm du}=e^0=1
\qquad
\frac{\mathrm dy}{\mathrm dv}=\cos(e^0)=\cos1
$$

所以

$$
y'(0)=\cos1\cdot1\cdot2
=2\cos1
\approx1.080604
$$

关键不是背住一条很长的公式，而是沿着表达式的嵌套结构从外向内列出每一层。每经过一层，就乘上该层在当前输入处的导数。

## 为什么记号像分数

在一元函数中，链式法则常写成

$$
\frac{\mathrm dy}{\mathrm dx}
=\frac{\mathrm dy}{\mathrm du}
\frac{\mathrm du}{\mathrm dx}
$$

这很像 $\mathrm du$ 可以约掉，但导数的定义并没有把 $\mathrm dy$、$\mathrm du$ 和 $\mathrm dx$ 当作三个先存在的普通数。它们是极限记号中的变化关系：

$$
\frac{\mathrm dy}{\mathrm dx}
=\lim_{h\to0}
\frac{y(x+h)-y(x)}{h}
$$

“分数相消”的直觉之所以可靠，是因为局部线性近似确实把输入小量映射成输出小量，而连续复合线性映射的系数就是乘积。遇到零导数、不可导点或多变量时，直接把符号当分数约分就可能越过定义域和存在性条件。

单位也会按链式法则相乘。若 $x$ 用秒计，$u=g(x)$ 用米计，$y=f(u)$ 用平方米计，那么 $\mathrm du/\mathrm dx$ 的单位是米/秒，$\mathrm dy/\mathrm du$ 的单位是平方米/米，乘起来就是平方米/秒，正好是 $\mathrm dy/\mathrm dx$ 的单位。

## 反向传播：沿计算图把导数乘回来

链式法则是反向传播的标量骨架。把一个简单计算拆成三层：

$$
z=wx+b
\qquad
a=z^2
\qquad
L=(a-5)^2
$$

固定

$$
x=3
\qquad
w=2
\qquad
b=1
$$

正向计算为

$$
z=7
\qquad
a=49
\qquad
L=1936
$$

反向沿相反方向传递局部导数：

$$
\frac{\partial L}{\partial a}=2(a-5)=88
\qquad
\frac{\partial a}{\partial z}=2z=14
\qquad
\frac{\partial z}{\partial w}=x=3
$$

因此

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
=88\cdot14\cdot3
=3696
$$

把正向表达式合成一个函数，

$$
L(w)=\big((3w+1)^2-5\big)^2
$$

直接求导并在 $w=2$ 代入，也得到 $3696$。反向传播没有创造新的微积分规则，它只是把链式法则按计算图的拓扑顺序重复使用；[计算图上的链式法则](../backpropagation/chain-rule-on-graphs/) 和 [反向传播](../backpropagation/backpropagation/) 篇会把多节点、分支和向量化情形展开。

如果沿偏置走，$\partial z/\partial b=1$，所以

$$
\frac{\partial L}{\partial b}=88\cdot14\cdot1=1232
$$

同一个损失对不同参数的导数不同，因为它们通向输出的局部路径不同。

## 多变量版本只先看结构

若标量输出依赖两个中间量：

$$
y=f(u,v)
\qquad
u=u(x)
\qquad
v=v(x)
$$

一维输入的变化会沿两条路径进入 $f$，结构是

$$
\frac{\mathrm dy}{\mathrm dx}
=\frac{\partial f}{\partial u}\frac{\mathrm du}{\mathrm dx}
+\frac{\partial f}{\partial v}\frac{\mathrm dv}{\mathrm dx}
$$

这里的 $\partial f/\partial u$ 和 $\partial f/\partial v$ 要在下一篇 [偏导数](../calculus/partial-derivatives/) 中定义；多个输入和输出时，标量乘法会升级为向量、矩阵的线性映射复合，分别由 [梯度](../calculus/gradient/)、[全导数](../calculus/total-derivative/)、[Jacobian 矩阵](../calculus/jacobian/) 和 [向量链式法则](../calculus/vector-chain-rule/) 接手。本篇先固定一元链式法则的骨架。

## 三组链式乘积的数字核对

把每层斜率单独列出来，最后再相乘：

| 复合函数 | 输入点 | 内层斜率 | 外层斜率 | 导数结果 |
| --- | --- | --- | --- | --- |
| $(3x+1)^2$ | $x=2$ | $3$ | $2\cdot7=14$ | $42$ |
| $\ln(x^2+1)$ | $x=1$ | $2$ | $1/2$ | $1$ |
| $\sin(e^{2x})$ | $x=0$ | $2\cdot1=2$ | $\cos1$ | $2\cos1\approx1.080604$ |
| $((3w+1)^2-5)^2$ | $w=2$ | $3\cdot14=42$ | $2\cdot44=88$ | $3696$ |

最后一行把中间两层先合成了 $42$，再乘最外层的 $88$。表格里的“内层”和“外层”是相对当前拆法的简称；表达式再多一层，就继续增加一列或按图逐层记录。

## 在机器学习中的读法

**每个节点只负责局部变化率**：线性层 $z=Wx+b$ 的局部导数是权重或输入，激活层提供自己的斜率，损失层提供对预测值的斜率。反向传播把它们按路径相乘，不需要为每一条完整输入输出关系重新求一个极限。

**小斜率会沿深度相乘**：若许多层的局部导数都小于 $1$，乘积可能快速变小；若很多层都大于 $1$，乘积可能快速变大。这是梯度消失和梯度爆炸的数学入口，具体的初始化、激活函数和残差连接留给后续词条。

**分支要把路径相加**：上面的多变量结构显示，同一个参数若通过多条路径影响损失，各条路径的链式乘积要相加。只沿一条路径回传会漏掉其他分支对同一参数的贡献。

**局部导数依赖当前点**：反向计算不能只保存“这个函数的公式”，还要保存正向经过的中间值。例如 $z^2$ 的局部导数是 $2z$，如果丢掉正向的 $z$，就无法在反向时得到正确斜率。

有限差分可以拿来检查一条简单计算图：

$$
\frac{\partial L}{\partial w}
\approx
\frac{L(w+h)-L(w)}{h}
$$

但深层网络的差分会遇到步长、舍入和计算成本问题；自动微分直接复用计算图结构传播导数，通常比逐参数差分更稳定。

## 容易混淆的地方

- **复合不是乘积**：$f(g(x))$ 用链式法则乘内外层斜率，$f(x)g(x)$ 用乘积法则把两条变化路径相加。
- **外层导数要在内层当前输出处取值**：$f'(g(a))$ 不是 $f'(a)$；先算 $g(a)$ 再把它送进外层导数。
- **不能漏掉内层导数**：$\ln(x^2+1)$ 的导数不只是 $1/(x^2+1)$，还要乘 $2x$。
- **多层不能只乘最外一层**：$\sin(e^{2x})$ 要乘 $\cos(e^{2x})$、$e^{2x}$ 和 $2$。
- **零导数不是公式失效**：某一层斜率为零时，沿该路径的一阶变化确实被压成零；要区分它和不可导。
- **分数记号不是无条件约分**：链式法则需要各层在相应点可导，符号直觉不能替代存在性检查。
- **分支路径要相加**：一个参数通过两个中间节点影响损失时，不能只保留其中一条路径。
- **反向传播仍然需要正向值**：局部导数常依赖中间激活，先把正向计算丢掉会使反向无法复现。

## 相关词条

- [导数](../calculus/derivatives/)：差商极限与局部线性近似
- [求导法则](../calculus/differentiation-rules/)：和、积、商、幂、指数、对数和三角函数
- [偏导数](../calculus/partial-derivatives/)：多变量函数的坐标方向变化率
- [梯度](../calculus/gradient/)：偏导数组成的向量
- [全导数](../calculus/total-derivative/)：多变量函数的整体线性近似
- [Jacobian 矩阵](../calculus/jacobian/)：向量值映射的导数矩阵
- [向量链式法则](../calculus/vector-chain-rule/)：矩阵形式的链式传递
- [计算图上的链式法则](../backpropagation/chain-rule-on-graphs/)：按节点和分支组织链式乘积
- [反向传播](../backpropagation/backpropagation/)：在神经网络中重复使用链式法则
- [自动微分](../calculus/automatic-differentiation/)：从计算结构传播导数
