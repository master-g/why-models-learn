---
title: "求导法则:把复杂函数拆开计算"
tags: ["why-models-learn"]
---

求导法则把导数定义里的极限变成一套可复用的代数操作。上一页用差商极限证明了常数函数、恒等函数和幂函数的基本导数；本篇继续从极限推导和式、常数倍、乘积和商法则，再整理幂、指数、对数与三角函数的常用结果。每条规则都说明它为什么成立，并用具体数字检查容易漏掉的项。复合函数的内外层传递属于 [链式法则](../calculus/chain-rule/)，不在这里提前展开。

## 和式与常数倍：导数的线性

若 $f$ 和 $g$ 在 $a$ 可导，先看它们的线性组合：

$$
\begin{aligned}
(\alpha f+\beta g)'(a)
&=\lim_{h\to0}
\frac{\alpha f(a+h)+\beta g(a+h)-\alpha f(a)-\beta g(a)}{h}\\
&=\alpha\lim_{h\to0}
\frac{f(a+h)-f(a)}{h}
+\beta\lim_{h\to0}
\frac{g(a+h)-g(a)}{h}\\
&=\alpha f'(a)+\beta g'(a)
\end{aligned}
$$

因此

$$
(f+g)'=f'+g'
\qquad
(\alpha f)'=\alpha f'
$$

这不是记号上的巧合：差商本身对函数值的加法和数乘就是线性的。常数函数是 $\alpha=0$ 的一部分：

$$
\frac{\mathrm d}{\mathrm dx}c=0
\qquad
\frac{\mathrm d}{\mathrm dx}x=1
$$

例如

$$
f(x)=x^2+3x+1
$$

由幂函数法则逐项求导：

$$
f'(x)=2x+3
\qquad
f'(2)=7
$$

不用规则而直接算差商也得到同一个数：

$$
\frac{f(2+h)-f(2)}{h}
=\frac{(11+7h+h^2)-11}{h}
=7+h
\longrightarrow7
$$

## 乘积法则：两个因素都在变化

乘积的变化不能只看一个因素。对 $F(x)=f(x)g(x)$，在差商分子中加上再减去 $f(a)g(a+h)$：

$$
\begin{aligned}
\frac{F(a+h)-F(a)}{h}
&=\frac{f(a+h)g(a+h)-f(a)g(a)}{h}\\
&=\frac{f(a+h)-f(a)}{h}g(a+h)
+f(a)\frac{g(a+h)-g(a)}{h}
\end{aligned}
$$

当 $h\to0$ 时，$g(a+h)\to g(a)$，因为可导函数先保证连续；两组差商分别趋近 $f'(a)$ 和 $g'(a)$。因此

$$
(fg)'(a)=f'(a)g(a)+f(a)g'(a)
$$

写成函数式：

$$
(fg)'=f'g+fg'
$$

每一项都保留一个原函数值。数字检查取

$$
f(x)=x^2
\qquad
g(x)=x+1
\qquad
a=2
$$

有

$$
f(2)=4,\quad f'(2)=4,\quad
g(2)=3,\quad g'(2)=1
$$

所以乘积法则给出

$$
(fg)'(2)=4\cdot3+4\cdot1=16
$$

直接展开 $f(x)g(x)=x^3+x^2$，得到 $3x^2+2x$，在 $x=2$ 也是 $12+4=16$。若误写成 $f'g'$，只会得到 $4$，少掉了两个函数值所代表的两条变化路径。

![乘积法则的两条变化路径与商法则的分母平方](/assets/calculus/svg/differentiation-rules.1.svg)

## 商法则：分母的变化不能漏

先对非零函数 $g$ 求倒数的导数。恒等式

$$
g(x)\frac1{g(x)}=1
$$

两边使用乘积法则：

$$
g'(x)\frac1{g(x)}
+g(x)\left(\frac1{g(x)}\right)'
=0
$$

所以

$$
\left(\frac1g\right)'
=-\frac{g'}{g^2}
$$

再把 $f/g$ 看成 $f\cdot(1/g)$，得到商法则：

$$
\left(\frac{f}{g}\right)'
=\frac{f'g-fg'}{g^2}
\qquad
(g\ne0)
$$

分子有“上导下减上导下”的结构，分母是整个 $g^2$，不是 $g$。

取

$$
f(x)=x+1
\qquad
g(x)=x+2
\qquad
a=1
$$

此时 $f(1)=2$、$f'(1)=1$、$g(1)=3$、$g'(1)=1$，商法则给出

$$
\left(\frac{x+1}{x+2}\right)'_{x=1}
=\frac{1\cdot3-2\cdot1}{3^2}
=\frac19
$$

把分式先改写成 $1-\frac1{x+2}$，导数是 $\frac1{(x+2)^2}$，在 $x=1$ 同样为 $1/9$。两个路径都保留了分母变化的影响。

## 幂函数：从整数指数延伸出去

上一页已经从二项式展开得到正整数幂：

$$
\frac{\mathrm d}{\mathrm dx}x^n=nx^{n-1}
\qquad
(n\in\mathbb{N})
$$

负整数指数由倒数规则得到。例如

$$
\frac{\mathrm d}{\mathrm dx}x^{-1}
=-\frac1{x^2}
=-x^{-2}
\qquad
(x\ne0)
$$

在定义域允许的区间上，幂函数法则可写成

$$
\frac{\mathrm d}{\mathrm dx}x^\alpha
=\alpha x^{\alpha-1}
$$

其中实数指数要限制在函数有定义且可导的区域。平方根可以不依赖记忆直接检查：在 $a=4$ 处，

$$
\begin{aligned}
\frac{\sqrt{4+h}-\sqrt4}{h}
&=\frac{\sqrt{4+h}-2}{h}\\
&=\frac{1}{\sqrt{4+h}+2}\\
&\longrightarrow\frac14
\end{aligned}
$$

所以

$$
\left.\frac{\mathrm d}{\mathrm dx}\sqrt{x}\right|_{x=4}
=\frac14
$$

有些公式在端点或零点会失效。比如 $\sqrt{x}$ 在 $x=0$ 的右差商是 $1/\sqrt{h}$，没有有限极限；负幂在 $x=0$ 没有定义。写出函数的定义域，是使用幂函数法则的一部分。

## 指数与对数

自然指数函数被定义为满足自身导数关系的特殊函数：

$$
\frac{\mathrm d}{\mathrm dx}e^x=e^x
$$

正因为导数仍是自身，指数增长在微分方程和连续时间模型里很方便。一般底数 $a>0$ 时，

$$
\frac{\mathrm d}{\mathrm dx}a^x
=a^x\ln a
$$

当 $a=e$ 时 $\ln e=1$，退回上一条。对数函数在 $x>0$ 上满足

$$
\frac{\mathrm d}{\mathrm dx}\ln x=\frac1x
$$

这两条互相对应：指数把加法坐标变成乘法尺度，对数把乘法尺度变回加法坐标。对数导数的定义仍然是差商极限，公式的完整证明会使用反函数和链式法则。

数字只检查函数值处的导数，不把近似当成精确值：

$$
\left.\frac{\mathrm d}{\mathrm dx}e^x\right|_{x=0}=1
\qquad
\left.\frac{\mathrm d}{\mathrm dx}\ln x\right|_{x=2}=\frac12
$$

若对数里还有一个随 $x$ 变化的表达式，例如 $\ln(x^2+1)$，不能只写成 $1/(x^2+1)$；里面的函数变化要由 [链式法则](../calculus/chain-rule/) 乘回来。

## 三角函数：先固定弧度

三角函数的导数公式默认角度用**弧度**。核心极限是

$$
\lim_{h\to0}\frac{\sin h}{h}=1
\qquad
\lim_{h\to0}\frac{\cos h-1}{h}=0
$$

用加法公式展开正弦差：

$$
\begin{aligned}
\frac{\sin(a+h)-\sin a}{h}
&=\sin a\frac{\cos h-1}{h}
+\cos a\frac{\sin h}{h}\\
&\longrightarrow\cos a
\end{aligned}
$$

因此

$$
\frac{\mathrm d}{\mathrm dx}\sin x=\cos x
$$

同理可得

$$
\frac{\mathrm d}{\mathrm dx}\cos x=-\sin x
$$

如果用角度而不是弧度，差商会多出一个单位换算因子 $\pi/180$。这不是三角函数公式变了，而是输入坐标的单位变了。

例如

$$
F(x)=\sin x+\cos x
$$

在 $x=0$ 处

$$
F'(0)=\cos0-\sin0=1
$$

而 $F(\pi/2)=1$、$F(0)=1$，区间平均变化率为 $0$；平均变化率为零不妨碍起点处的瞬时变化率为 $1$，因为两者测量的区间不同。

## 把规则排成一张表

在表达式有定义的区间上，常用规则可以压缩成：

| 函数 | 导数 | 使用边界 |
| --- | --- | --- |
| $c$ | $0$ | $c$ 是常数 |
| $x^\alpha$ | $\alpha x^{\alpha-1}$ | 先确认定义域 |
| $f+g$ | $f'+g'$ | 和式逐项求导 |
| $\alpha f$ | $\alpha f'$ | $\alpha$ 与 $x$ 无关 |
| $fg$ | $f'g+fg'$ | 两个因素各贡献一项 |
| $f/g$ | $(f'g-fg')/g^2$ | $g\ne0$ |
| $e^x$ | $e^x$ | 自然指数 |
| $a^x$ | $a^x\ln a$ | $a>0$ |
| $\ln x$ | $1/x$ | $x>0$ |
| $\sin x$ | $\cos x$ | 弧度 |
| $\cos x$ | $-\sin x$ | 弧度 |

表中的每一行都只处理“同一层”的运算。若表达式是 $e^{x^2}$、$\sin(3x+1)$ 或 $\ln(f(x))$，外层规则算完后还缺内层变化率，那是链式法则的工作。

## 综合例子：先看结构，再代规则

取

$$
F(x)=\frac{(x^2+1)(x+3)}{x+1}
$$

不需要把所有项先展开。令

$$
p(x)=x^2+1
\qquad
q(x)=\frac{x+3}{x+1}
$$

则 $F=pq$。在 $x=1$：

$$
p(1)=2,\qquad p'(1)=2
$$

商法则给出

$$
q(1)=2
\qquad
q'(1)
=\frac{1\cdot2-4\cdot1}{2^2}
=-\frac12
$$

因此

$$
F'(1)=p'(1)q(1)+p(1)q'(1)
=2\cdot2+2\cdot\left(-\frac12\right)
=3
$$

直接展开并求导也能核对这个数：

$$
F(x)=\frac{x^3+3x^2+x+3}{x+1}
\qquad
F'(1)=3
$$

结构化求导的好处不在于少写几个字符，而在于每个局部规则都有明确的适用条件；表达式变长时，检查“哪一层是乘积、哪一层是商、内层是否还在变化”比盲目展开更可靠。

## 在机器学习中的读法

权重衰减常把数据损失和参数惩罚加在一起。例如

$$
L(w)=(w-2)^2+\frac{\lambda}{2}w^2
$$

由和式、幂函数和常数倍规则：

$$
L'(w)=2(w-2)+\lambda w
$$

取 $\lambda=0.1$、$w=3$：

$$
L'(3)=2+0.3=2.3
$$

第一项是数据损失对参数的敏感度，第二项是正则化对参数的拉回作用。把正则项的导数漏掉，更新方向就不再对应写下来的目标函数。

乘积法则出现在门控和缩放结构中。若一个输出近似写成 $y(x)=g(x)h(x)$，输入变化会沿两条路径传到输出：

$$
\frac{\mathrm dy}{\mathrm dx}
=g'(x)h(x)+g(x)h'(x)
$$

每一条路径都要保留另一条路径当前的值。复合激活、损失和线性层串起来时，还要把每层的局部变化率相乘，这一步由 [链式法则](../calculus/chain-rule/) 接手。

指数和对数的导数在概率模型和损失函数里频繁出现。比如对数把乘积变成和，导数把每个因子的相对变化率暴露出来；具体的 softmax、交叉熵和反向传播会在后续机器学习词条中展开，不在这里把一阶规则误当成完整训练算法。

## 容易混淆的地方

- **乘积法则不能只导一边**：$(fg)'$ 有 $f'g$ 和 $fg'$ 两项；把它写成 $f'g'$ 会漏掉两条变化路径中的绝大部分。
- **商法则的分母是平方**：$(f/g)'$ 的分母是 $g^2$，分子还要保留减号和两个函数值。
- **幂函数法则要看定义域**：$\sqrt{x}$ 的公式在正数区间可用，负幂不能在零点硬代。
- **三角函数默认弧度**：换成角度后，输入单位改变，导数会多出 $\pi/180$ 的换算因子。
- **表达式有内外层就要想到链式法则**：$e^{x^2}$ 不是只把外层写成 $e^{x^2}$；$x^2$ 的变化率还没有传回来。
- **规则不是把函数当作字符串替换**：先识别最外层运算，再确认每个子函数的定义域和是否依赖当前变量。
- **精确导数和数值差商不同**：差商依赖步长 $h$，法则给的是 $h\to0$ 后的极限；两者的数值不应混写。

## 相关词条

- [导数](../calculus/derivatives/)：差商极限与局部线性近似
- [链式法则](../calculus/chain-rule/)：复合函数的内外层导数相乘
- [偏导数](../calculus/partial-derivatives/)：多变量表达式中的坐标方向求导
- [梯度](../calculus/gradient/)：偏导数组成的参数更新方向
- [全导数](../calculus/total-derivative/)：多变量函数的整体线性近似
- [范数](../linear-algebra/norms/)：误差和梯度大小的测量
- [梯度下降](../training-nn/gradient-descent/)：用导数决定参数更新方向
