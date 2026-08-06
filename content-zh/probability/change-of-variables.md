---
title: "变量变换：密度为何要乘 Jacobian"
tags: ["why-models-learn"]
---

**变量变换**研究随机变量经过函数映射后，概率分布如何改变。若 $Y=g(X)$，概率不会因为换了坐标而凭空产生或消失；改变的是同一份概率质量在新坐标轴上的密度。单调一维变换需要乘以逆函数导数的绝对值，多元变换需要乘以逆映射 Jacobian 行列式的绝对值。

对可逆的一维变换，核心公式是

$$
f_Y(y)
=f_X\bigl(g^{-1}(y)\bigr)
\left|\frac{d}{dy}g^{-1}(y)\right|
$$

对可逆的多元变换，核心公式是

$$
f_Y(y)
=f_X\bigl(g^{-1}(y)\bigr)
\left|\det J_{g^{-1}}(y)\right|
$$

本篇先从「概率区间不变」推导一维公式，再处理平方这类多对一变换，接着解释 Jacobian 如何测量面积体积伸缩，最后连接极坐标、对数密度和生成模型。离散变量也遵循同一个守恒原则，只是把密度换成概率质量求和。

## 变的是坐标，不是概率质量

设 $X$ 的密度为 $f_X$，令

$$
Y=g(X)
$$

一个小区间在变换前后代表同一件事：

$$
P(x\leq X\leq x+dx)
=P(g(x)\leq Y\leq g(x+dx))
$$

如果 $g$ 在该处递增且可微，$dy\approx g'(x)\,dx$。密度乘区间长度必须相等：

$$
f_X(x)\,dx
\approx
f_Y(y)\,dy
$$

因此

$$
f_Y(y)
\approx
f_X(x)\frac{dx}{dy}
=\frac{f_X(x)}{g'(x)}
$$

若变换递减，$g'(x)<0$，长度仍然是正数，所以必须取绝对值：

$$
f_Y(y)
=\frac{f_X(x)}{|g'(x)|},
\qquad x=g^{-1}(y)
$$

这不是一个记号技巧。若 $g$ 把一段区间拉长两倍，同样的概率质量被铺到两倍长度上，密度就减半；若把长度压缩一半，密度就加倍。

## 一维可逆变换的完整公式

若 $g$ 在支持集上严格单调且可逆，令 $x=g^{-1}(y)$。从 CDF 开始：

### 递增情形

当 $g$ 递增时：

$$
\begin{aligned}
F_Y(y)
&=P(Y\leq y)\\
&=P(g(X)\leq y)\\
&=P(X\leq g^{-1}(y))\\
&=F_X(g^{-1}(y))
\end{aligned}
$$

对 $y$ 求导：

$$
\begin{aligned}
f_Y(y)
&=f_X(g^{-1}(y))
\frac{d}{dy}g^{-1}(y)\\
&=f_X(g^{-1}(y))
\left|\frac{d}{dy}g^{-1}(y)\right|
\end{aligned}
$$

### 递减情形

当 $g$ 递减时，事件方向反过来：

$$
\begin{aligned}
F_Y(y)
&=P(g(X)\leq y)\\
&=P(X\geq g^{-1}(y))\\
&=1-F_X(g^{-1}(y))
\end{aligned}
$$

求导时出现一个负号，而逆函数导数本身也为负，最终仍得到

$$
f_Y(y)
=f_X(g^{-1}(y))
\left|\frac{d}{dy}g^{-1}(y)\right|
$$

所以递增和递减可以用一个绝对值公式统一。变换后的支持集也必须一起改变：密度公式只在 $y=g(x)$ 能到达的区域有效，其他地方密度为 0。

## 线性变换：单位改变如何改变密度

令

$$
Y=aX+b,\qquad a\neq0
$$

逆变换和逆导数是

$$
X=\frac{Y-b}{a},\qquad
\left|\frac{dX}{dY}\right|=\frac1{|a|}
$$

因此

$$
f_Y(y)
=\frac1{|a|}
f_X\left(\frac{y-b}{a}\right)
$$

平移 $b$ 只移动密度曲线，缩放 $a$ 同时改变横轴长度和纵轴高度。概率、CDF 的取值不因单位变化而改变，但密度的数值会改变，因为密度带有「每单位长度的概率」这一单位。

若 $X\sim\operatorname{Uniform}[0,1]$，取

$$
Y=2+3X
$$

逆变换为 $x=(y-2)/3$，支持集从 $[0,1]$ 变成 $[2,5]$，所以

$$
f_Y(y)=
\begin{cases}
\frac13,&2\leq y\leq5,\\
0,&\text{其他}.
\end{cases}
$$

原区间长度是 1，新区间长度是 3；总概率仍然是面积 $3\cdot(1/3)=1$。期望和方差也按熟悉的规则变换：

$$
\mathbb E[Y]=2+3\mathbb E[X]=3.5
$$

$$
\operatorname{Var}(Y)=3^2\operatorname{Var}(X)=\frac34
$$

直接套密度公式和先用期望方差的仿射规则，得到的是同一个结果。

## 多对一变换：每个原像都要贡献概率

公式只写一个 $g^{-1}(y)$ 的前提是 $g$ 可逆。平方函数

$$
Y=X^2
$$

在正负两侧会把两个不同的 $x$ 映射到同一个 $y$，不能只保留其中一个原像。一般的一维多分支公式是

$$
f_Y(y)
=\sum_{x_i:g(x_i)=y}
\frac{f_X(x_i)}{|g'(x_i)|}
$$

求和遍历所有满足 $g(x_i)=y$ 的原像，并且只保留原始支持集中的解。

### 例：均匀变量的平方

令 $X\sim\operatorname{Uniform}[-1,1]$，于是 $f_X(x)=1/2$。当 $0\leq y\leq1$：

$$
F_Y(y)
=P(X^2\leq y)
=P(-\sqrt y\leq X\leq\sqrt y)
=\sqrt y
$$

求导得到

$$
f_Y(y)=\frac1{2\sqrt y},
\qquad0<y<1
$$

从两个原像也能得到相同结果：

$$
\begin{aligned}
f_Y(y)
&=\frac{f_X(\sqrt y)}{2\sqrt y}
+\frac{f_X(-\sqrt y)}{2\sqrt y}\\
&=\frac1{4\sqrt y}+\frac1{4\sqrt y}\\
&=\frac1{2\sqrt y}
\end{aligned}
$$

虽然 $f_Y(y)$ 在 0 附近变大，但它仍然可积：

$$
\int_0^1\frac1{2\sqrt y}\,dy=1
$$

这里最容易犯的错误是只算 $x=\sqrt y$ 的分支，得到一半的总概率；另一个错误是忘记 $g'(0)=0$，把密度在边界的发散误判成概率无限大。密度可以在某点发散，单点概率仍然可以为 0。

### CDF 方法何时更安全

当变换不是单调、导数在某些点为 0，或支持集被分段切开时，先写

$$
F_Y(y)=P(g(X)\leq y)
$$

通常比直接背分支公式更安全。先解出关于 $X$ 的区间，再对原密度积分，最后对 $y$ 求导。CDF 方法会自动处理多个原像和支持集边界。

## 正态变量平方：为什么会出现卡方分布

令 $X\sim\mathcal N(0,1)$，仍取 $Y=X^2$。标准正态密度是

$$
f_X(x)=\frac1{\sqrt{2\pi}}e^{-x^2/2}
$$

两个原像贡献：

$$
\begin{aligned}
f_Y(y)
&=\frac{f_X(\sqrt y)}{2\sqrt y}
+\frac{f_X(-\sqrt y)}{2\sqrt y}\\
&=\frac1{\sqrt{2\pi y}}e^{-y/2},
\qquad y>0
\end{aligned}
$$

这正是自由度为 1 的卡方分布密度。它的均值可以直接从函数期望得到：

$$
\mathbb E[Y]=\mathbb E[X^2]=1
$$

平方变换把有正负方向的标准正态压到非负轴上，两个方向的概率质量在同一个 $y$ 处叠加。

## 多元变量变换：Jacobian 测量面积伸缩

令

$$
x=(x_1,\ldots,x_d),\qquad
y=g(x)=(g_1(x),\ldots,g_d(x))
$$

Jacobian 矩阵是所有一阶偏导数组成的矩阵：

$$
J_g(x)
=\frac{\partial y}{\partial x}
=
\begin{pmatrix}
\frac{\partial g_1}{\partial x_1}&\cdots&\frac{\partial g_1}{\partial x_d}\\
\vdots&\ddots&\vdots\\
\frac{\partial g_d}{\partial x_1}&\cdots&\frac{\partial g_d}{\partial x_d}
\end{pmatrix}
$$

在局部，$|\det J_g(x)|$ 是小体积被拉伸的倍数：

$$
dy
\approx
|\det J_g(x)|\,dx
$$

概率质量守恒：

$$
f_X(x)\,dx=f_Y(y)\,dy
$$

因此可逆变换满足

$$
f_Y(y)
=f_X(x)\frac1{|\det J_g(x)|},
\qquad y=g(x)
$$

也可以写成只含逆映射的形式：

$$
f_Y(y)
=f_X(g^{-1}(y))
\left|\det J_{g^{-1}}(y)\right|
$$

行列式的绝对值很重要。负行列式表示方向翻转，但概率体积不能是负数；行列式为 0 则表示局部把体积压到低维，普通的 $d$ 维密度公式不再直接适用。

对二维线性变换 $y=Ax$：

$$
J_g(x)=A,\qquad
f_Y(y)=\frac{f_X(A^{-1}y)}{|\det A|}
$$

若 $A$ 把面积扩大 6 倍，同样概率就会铺在 6 倍面积上，密度缩为原来的 $1/6$。

![坐标变换用 Jacobian 修正面积元素](/assets/probability/svg/change-of-variables.1.svg)

## 极坐标：面积元素为什么是 $r\,dr\,d\theta$

极坐标变换为

$$
x=r\cos\theta,\qquad
y=r\sin\theta
$$

它的 Jacobian 是

$$
J_{(r,\theta)\to(x,y)}
=
\begin{pmatrix}
\cos\theta&-r\sin\theta\\
\sin\theta&r\cos\theta
\end{pmatrix}
$$

行列式：

$$
\det J
=r\cos^2\theta+r\sin^2\theta
=r
$$

所以

$$
dx\,dy=r\,dr\,d\theta
$$

这就是极坐标面积元素中 $r$ 的来源。半径越大，同样的 $dr$ 和 $d\theta$ 扫出的扇形环越长，面积自然越大。

设 $(X,Y)$ 在单位圆盘内均匀分布：

$$
f_{X,Y}(x,y)=
\begin{cases}
\frac1\pi,&x^2+y^2\leq1,\\
0,&\text{其他}.
\end{cases}
$$

变到 $(R,\Theta)$ 后：

$$
f_{R,\Theta}(r,\theta)
=\frac1\pi r,
\qquad
0\leq r\leq1,\ 0\leq\theta<2\pi
$$

对角度积分得到半径密度：

$$
f_R(r)=\int_0^{2\pi}\frac r\pi\,d\theta=2r
$$

因此圆盘内均匀不等于半径在 $[0,1]$ 上均匀。半径越大的环面积越大：

$$
P(R\leq0.5)=\int_0^{0.5}2r\,dr=0.25
$$

$$
\mathbb E[R]=\int_0^1r\cdot2r\,dr=\frac23
$$

如果错误地把 $R$ 当作 Uniform[0,1]，会得到 $P(R\leq0.5)=0.5$ 和 $\mathbb E[R]=1/2$，这两个结果都违背圆盘的几何面积。

## 对数密度：变量变换进入生成模型

当 $y=g(x)$ 可逆时，密度公式取对数：

$$
\log f_Y(y)
=\log f_X(x)
-\log\left|\det J_g(x)\right|,
\qquad x=g^{-1}(y)
$$

第一项是简单基分布的 log-density，第二项是坐标变换对体积的修正。若变换把空间拉伸，行列式大于 1，单位体积中落入的概率密度就要降低；若压缩，密度就要升高。

Normalizing flow 选择一串可逆变换：

$$
x_0\longrightarrow x_1\longrightarrow\cdots\longrightarrow x_K=y
$$

链式法则给出

$$
\det J_{g_K\circ\cdots\circ g_1}
=\prod_{k=1}^K\det J_{g_k}
$$

所以

$$
\log f_Y(y)
=\log f_{X_0}(x_0)
-\sum_{k=1}^K\log|\det J_{g_k}(x_{k-1})|
$$

这解释了为什么流模型需要同时满足可逆、Jacobian 行列式可计算和数值稳定。只把神经网络输出当成新变量却忘记体积修正，会得到错误的密度和错误的似然。

## 离散变量和混合分布的版本

如果 $X$ 是离散变量，不能使用连续密度的导数公式。概率质量直接沿映射合并：

$$
P(Y=y)
=\sum_{x:g(x)=y}P(X=x)
$$

例如 $X$ 在 $\{-1,0,1\}$ 上均匀，$Y=X^2$ 后

$$
P(Y=0)=\frac13,\qquad
P(Y=1)=\frac23
$$

这和连续平方变换中「正负两个原像贡献相加」是同一个守恒思想，只是连续情形把求和换成了密度和 Jacobian。

若分布同时有点质量和连续密度，变量变换必须分别处理两部分：点质量沿映射移动并合并，连续部分按 Jacobian 变换。不能因为连续公式写得漂亮，就把原本的原子概率当成普通密度积分。

## 失效模式

**忘记逆导数。**密度不是把 $f_X(x)$ 直接换成 $f_X(g^{-1}(y))$；还必须乘逆函数导数的绝对值。

**只取一个原像。**平方、绝对值和三角函数等多对一变换会把多个区间压到同一个输出，必须对所有分支求和。

**漏掉支持集。**变换后的密度只在可达的 $y$ 区域非零。公式本身有值不代表该点属于随机变量的支持。

**把密度当概率。**$f_Y(y)$ 是单位长度或单位体积的概率质量，单点密度大不表示单点概率大。

**忽略绝对值。**递减变换或方向翻转会让导数、Jacobian 行列式为负，但概率密度不能为负。

**在 Jacobian 为零处机械套公式。**局部降维、边界点和奇异映射可能产生无穷密度或低维分布，需要回到 CDF、分支积分或测度的定义。

**在极坐标中漏掉 $r$。**$dr\,d\theta$ 不是笛卡尔面积，半径造成的扇形环伸缩必须由 Jacobian 补上。

**在生成模型中漏掉 log-det。**可逆网络改变了密度，不只是改变样本外观；似然计算必须包含体积修正。

## 相关词条

- [随机变量](../probability/random-variables/)：定义变量、分布和函数变换的概率对象。
- [连续分布](../probability/continuous-distributions/)：提供密度、CDF 和连续区间概率的基础。
- [高斯分布](../probability/gaussian-distribution/)：给出正态密度和平方变换中的标准正态例子。
- [向量](../linear-algebra/vectors/)：为多元变量和坐标表示提供线性代数语言。
- [Jacobian](../calculus/jacobian/)：专门推导偏导矩阵、局部线性化和行列式。
- [期望](../probability/expectation/)：计算变换后函数的期望。
- [正规化流](../generative-models/normalizing-flows/)：若词条尚未完成，本链接会在站点降级为纯文本。
