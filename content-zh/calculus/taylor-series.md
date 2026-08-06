---
title: "Taylor 展开:用多项式逼近局部函数"
tags: ["why-models-learn"]
---

**Taylor 展开**用一个点处的各阶导数构造多项式，使这个多项式在该点附近尽量贴合原函数。以展开点 $a$ 为中心，$n$ 阶 Taylor 多项式是

$$
T_n(x;a)
=
\sum_{k=0}^{n}
\frac{f^{(k)}(a)}{k!}(x-a)^k
$$

它把函数值、斜率、曲率以及更高阶局部变化依次放进同一个表达式。有限的 $T_n$ 是多项式近似；让 $n\to\infty$ 得到的 Taylor 级数是否真的等于原函数，还要检查收敛区间和函数本身的性质。本篇先说明系数为什么是阶乘分母，再用余项估计误差，最后连接多变量 Hessian 和机器学习中的局部近似。

## 系数为什么是阶乘分母

假设我们要找一个 $n$ 阶多项式

$$
P_n(x)=c_0+c_1(x-a)+c_2(x-a)^2+\cdots+c_n(x-a)^n
$$

让它在 $a$ 处和 $f$ 有相同的前 $n$ 阶导数：

$$
P_n^{(j)}(a)=f^{(j)}(a)
\qquad
j=0,1,\ldots,n
$$

代入 $x=a$：

$$
P_n(a)=c_0=f(a)
$$

求一次导数后代入：

$$
P_n'(a)=c_1=f'(a)
$$

求二次导数时，只有 $c_2(x-a)^2$ 的二阶导数在 $a$ 不为零：

$$
P_n''(a)=2c_2=f''(a)
\qquad
c_2=\frac{f''(a)}{2!}
$$

继续下去，第 $k$ 项在求 $k$ 次导数后贡献 $k!c_k$，因此

$$
c_k=\frac{f^{(k)}(a)}{k!}
$$

这就是 Taylor 多项式的系数。$k!$ 不是记号上的装饰，而是幂函数连续求导产生的倍率。

低阶形式依次是

$$
T_0=f(a)
$$

$$
T_1=f(a)+f'(a)(x-a)
$$

$$
T_2=f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)^2
$$

所以 [导数](../calculus/derivatives/) 篇的一阶局部线性近似正是 $T_1$；加入 Hessian 或二阶导数后，才会看到局部弯曲。

## 余项告诉我们近似差多少

写成

$$
f(x)=T_n(x;a)+R_n(x;a)
$$

其中 $R_n$ 是余项。若 $f$ 在 $a$ 与 $x$ 之间有 $n+1$ 阶导数，Lagrange 余项给出某个介于 $a$ 和 $x$ 的 $\xi$：

$$
R_n(x;a)
=
\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}
$$

如果在这段区间上

$$
\left|f^{(n+1)}(t)\right|\le M
$$

就有可直接检查的上界：

$$
|R_n(x;a)|
\le
\frac{M|x-a|^{n+1}}{(n+1)!}
$$

阶数增加一阶，会多出一个位移因子和一个阶乘分母，但导数上界也可能随阶数变大。不能只看多项式项数就保证误差很小。

以 $e^x$ 在 $a=0$ 展开。所有阶导数在 $0$ 都是 $1$，所以三阶多项式是

$$
T_3(x;0)=1+x+\frac{x^2}{2}+\frac{x^3}{6}
$$

在 $x=0.2$：

$$
T_3(0.2;0)
=1+0.2+\frac{0.2^2}{2}+\frac{0.2^3}{6}
=1.221333333\ldots
$$

真实值为

$$
e^{0.2}=1.221402758\ldots
$$

误差约为 $0.000069425$。因为四阶导数仍是 $e^x$，在 $[0,0.2]$ 上可以取 $M=e^{0.2}$，余项上界为

$$
\frac{e^{0.2}(0.2)^4}{4!}
\approx0.00008143
$$

真实误差小于这个上界。余项不是只用来写证明，它能帮助决定当前阶数够不够。

## 常见函数的展开

展开点不同，系数和收敛性质都会改变。下面先列出在 $0$ 附近最常用的形式：

| 函数 | Taylor 级数或前几项 | 常见收敛范围 |
| --- | --- | --- |
| $e^x$ | $1+x+x^2/2!+x^3/3!+\cdots$ | 所有实数 $x$ |
| $\sin x$ | $x-x^3/3!+x^5/5!-\cdots$ | 所有实数 $x$ |
| $\cos x$ | $1-x^2/2!+x^4/4!-\cdots$ | 所有实数 $x$ |
| $\ln(1+x)$ | $x-x^2/2+x^3/3-\cdots$ | 通常先取 $\lvert x\rvert<1$ |
| $(1+x)^\alpha$ | $1+\alpha x+\alpha(\alpha-1)x^2/2!+\cdots$ | 由 $\alpha$ 与 $\lvert x\rvert$ 决定 |

例如

$$
\sin x
=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\cdots
$$

在 $x=0.5$，三阶多项式给出

$$
T_3(0.5)=0.5-\frac{0.5^3}{6}
=0.479166667\ldots
$$

加上五阶项后：

$$
T_5(0.5)
=0.5-\frac{0.5^3}{6}+\frac{0.5^5}{120}
=0.479427083\ldots
$$

而 $\sin(0.5)=0.479425539\ldots$。五阶结果比三阶结果更接近，但它仍是有限多项式，不是把无限级数在任意输入处无条件截断。

## Taylor 多项式和 Taylor 级数不是一回事

对每个固定的 $n$，$T_n(x;a)$ 都是有限多项式。Taylor 级数是形式上的无限和：

$$
\sum_{k=0}^{\infty}
\frac{f^{(k)}(a)}{k!}(x-a)^k
$$

要把它当成函数，至少要回答两个问题：

1. 这个级数在哪些 $x$ 上收敛？
2. 收敛后的和是否等于原函数 $f(x)$？

几何级数是清楚的例子：

$$
\frac1{1-x}
=1+x+x^2+x^3+\cdots
\qquad
|x|<1
$$

当 $x=2$ 时，右侧项根本不趋近于零，级数不可能收敛。即使在收敛区间内，也要检查它是否是目标函数的 Taylor 级数，而不是只做形式代数。

有些函数在一点处拥有所有阶导数，却不等于自己的 Taylor 级数。定义

$$
f(x)=
\begin{cases}
e^{-1/x^2},&x\ne0\\
0,&x=0
\end{cases}
$$

这个函数在 $0$ 处各阶导数都为 $0$，所以它在 $0$ 处的 Taylor 级数是恒等于 $0$ 的级数；但对任何 $x\ne0$，$f(x)>0$。它是光滑的，却不是在 $0$ 处由自己的 Taylor 级数表示的解析函数。实际使用中，有限 Taylor 多项式是局部工具，不能仅凭“所有导数存在”就宣布无限展开等于原函数。

## 多变量 Taylor 展开

多变量函数可以沿一条直线降成一元问题。令

$$
g(t)=f(\boldsymbol a+t\boldsymbol h)
$$

先前的梯度和 Hessian 关系给出

$$
g'(0)
=\nabla f(\boldsymbol a)^{\mathsf T}\boldsymbol h
$$

以及

$$
g''(0)
=\boldsymbol h^{\mathsf T}H_f(\boldsymbol a)\boldsymbol h
$$

于是二阶多变量展开是

$$
\begin{aligned}
f(\boldsymbol a+\boldsymbol h)
&\approx
f(\boldsymbol a)
+\nabla f(\boldsymbol a)^{\mathsf T}\boldsymbol h\\
&\quad
+\frac12\boldsymbol h^{\mathsf T}H_f(\boldsymbol a)\boldsymbol h
\end{aligned}
$$

更高阶时，$k$ 阶导数不再是一个矩阵，而是一个 $k$ 线性映射：

$$
f(\boldsymbol a+\boldsymbol h)
\approx
\sum_{k=0}^{n}
\frac1{k!}
D^k f(\boldsymbol a)
[\underbrace{\boldsymbol h,\ldots,\boldsymbol h}_{k\text{ 次}}]
$$

二阶时这个 $D^2f$ 就由 Hessian 表示。[Hessian 矩阵](../calculus/hessian/) 篇中

$$
f(x,y)=x^2+3xy+y^2
$$

在 $(1,2)$ 的例子之所以能被二阶式精确重建，是因为原函数本身没有三阶及更高阶项。取 $\boldsymbol h=(0.1,-0.2)$，一次项为 $-0.6$，二次项为 $-0.01$，于是从 $11$ 得到 $10.39$。

![Taylor 多项式的阶数增加后逐渐贴近原函数曲线](/assets/calculus/svg/taylor-series.1.svg)

## 机器学习中的局部多项式

非线性模型中，Taylor 展开常用于理解激活函数和损失在一个点附近的行为。以 sigmoid

$$
\sigma(x)=\frac1{1+e^{-x}}
$$

在 $0$ 附近的展开为

$$
\sigma(x)
=\frac12+\frac{x}{4}-\frac{x^3}{48}+O(x^5)
$$

取 $x=0.2$，三次近似是

$$
\frac12+\frac{0.2}{4}-\frac{0.2^3}{48}
=0.549833333\ldots
$$

直接计算 $\sigma(0.2)=0.549833998\ldots$，误差约为 $0.000000665$。在接近 $0$ 的输入上，有限多项式能解释 sigmoid 的中心区域；输入很大时，应该回到原函数的饱和行为，不能把中心展开当成全域公式。

对损失函数，二阶展开是

$$
L(\boldsymbol w+\boldsymbol\delta)
\approx
L(\boldsymbol w)
+\nabla L(\boldsymbol w)^{\mathsf T}\boldsymbol\delta
+\frac12\boldsymbol\delta^{\mathsf T}H_L(\boldsymbol w)\boldsymbol\delta
$$

梯度下降只保留一次项，Newton 和其他二阶方法还使用二次项。展开式把“步长太大时为什么线性预测失效”写成了具体的二阶误差，而不是一句模糊的近似提醒。

## 怎样选择展开阶数

设目标误差是 $\varepsilon$，余项上界满足

$$
\frac{M|x-a|^{n+1}}{(n+1)!}\le\varepsilon
$$

就可以选择足够大的 $n$。但有三个实际条件：

- 展开点应当靠近要估计的输入，Taylor 展开首先是局部工具；
- 需要知道或估计相应阶导数的上界 $M$；
- 如果输入远离展开点，换一个展开中心往往比盲目增加阶数更稳定。

对于交错级数如 $\sin x$，下一项常能提供简便的误差尺度；对于一般函数，要使用适用的余项定理或数值检查。高阶并不自动等于更好，浮点计算中极小项和大阶乘也会带来舍入问题。

## 常见失效模式

- **把有限多项式当成无限级数。** $T_n$ 只使用前 $n$ 阶信息；它是否收敛到原函数是另一个问题。
- **忽略收敛范围。** $\ln(1+x)$、几何级数等展开有明确边界，移出范围后项可能不趋于零。
- **把光滑当成解析。** 所有阶导数存在，并不保证函数等于自己的 Taylor 级数；平坦函数反例说明了这点。
- **忘记展开中心。** $T_n(x;a)$ 的每一项都依赖 $(x-a)$，把 $a$ 换掉而不重新计算导数会得到另一条多项式。
- **只写近似不写余项。** 没有误差上界，就不知道当前阶数和输入距离是否足够。
- **把局部模型外推到全域。** 激活函数和损失在远离展开点后可能进入完全不同的曲率或饱和区间。

## 相关词条

- [导数](../calculus/derivatives/)：Taylor 一阶项所依赖的局部线性系数。
- [求导法则](../calculus/differentiation-rules/)：计算各阶导数的规则。
- [梯度](../calculus/gradient/)：多变量 Taylor 展开的一次项。
- [Hessian 矩阵](../calculus/hessian/)：多变量展开的二次项。
- [二次型](../linear-algebra/quadratic-forms/)：解释二阶项的方向和符号。
- [Taylor 展开](../calculus/taylor-series/)：本条目的有限多项式与无限级数框架。
- [梯度下降](../training-nn/gradient-descent/)：只使用局部一次项的更新方法。
- [二阶优化方法](../optimization-theory/second-order-methods/)：使用局部二次模型的优化方法。
