---
title: "全导数:把多变量变化写成一个线性映射"
tags: ["why-models-learn"]
---

对于多变量函数，**全导数**不是把几个偏导数并排写完，而是在一个点用一个线性映射同时近似所有输入方向的变化。若 $f$ 在 $\boldsymbol a$ 可微，全导数记为 $Df(\boldsymbol a)$，对小位移 $\boldsymbol h$ 满足

$$
f(\boldsymbol a+\boldsymbol h)
=
f(\boldsymbol a)
+Df(\boldsymbol a)[\boldsymbol h]
+o\left(\|\boldsymbol h\|_2\right)
$$

这里 $Df(\boldsymbol a)[\boldsymbol h]$ 是线性映射对位移的作用，不是把 $\boldsymbol h$ 代入原函数。标量函数的全导数可以用梯度的内积表示；向量值函数的全导数则是把输入扰动映射成输出扰动的线性变换。本篇先把这个定义写清楚，再用数字、路径和矩阵形状区分全导数、偏导、方向导数与梯度。

## 先找一个统一的线性近似

设 $f:\mathbb R^n\to\mathbb R^m$，在点 $\boldsymbol a$ 附近取一个小位移 $\boldsymbol h$。如果存在一个线性映射 $L:\mathbb R^n\to\mathbb R^m$，使得

$$
\lim_{\|\boldsymbol h\|_2\to0}
\frac{
\left\|
f(\boldsymbol a+\boldsymbol h)-f(\boldsymbol a)-L(\boldsymbol h)
\right\|_2
}{
\|\boldsymbol h\|_2
}
=0
$$

那么称 $f$ 在 $\boldsymbol a$ 可微，并把这个唯一的 $L$ 定义为 $Df(\boldsymbol a)$。分子是实际输出变化减去线性预测后的剩余，极限条件说的是：剩余相对于输入位移的长度会消失。

“线性”有两个要求：

$$
L(\boldsymbol h+\boldsymbol k)=L(\boldsymbol h)+L(\boldsymbol k)
\qquad
L(c\boldsymbol h)=cL(\boldsymbol h)
$$

因此它保留小位移的叠加和缩放关系。全导数把点 $\boldsymbol a$ 固定住，只描述这个点附近；同一个函数在不同点通常有不同的线性映射。

把余项写成小量记号，定义等价于

$$
f(\boldsymbol a+\boldsymbol h)
=
f(\boldsymbol a)
+Df(\boldsymbol a)[\boldsymbol h]
+o\left(\|\boldsymbol h\|_2\right)
$$

其中 $o(\|\boldsymbol h\|_2)$ 的意思是，除以 $\|\boldsymbol h\|_2$ 后趋近于零。它比“误差很小”更具体：位移缩小十倍时，余项相对于位移长度还会继续缩小。

## 标量函数中全导数就是梯度的内积

先看 $f:\mathbb R^2\to\mathbb R$。设坐标基向量为

$$
\boldsymbol e_1=(1,0)
\qquad
\boldsymbol e_2=(0,1)
$$

因为

$$
\frac{f(a+h,b)-f(a,b)}{h}
\longrightarrow f_x(a,b)
$$

全导数在线性映射上必须满足

$$
Df(a,b)[\boldsymbol e_1]=f_x(a,b)
\qquad
Df(a,b)[\boldsymbol e_2]=f_y(a,b)
$$

任意位移 $\boldsymbol h=(h,k)$ 都可以写成 $h\boldsymbol e_1+k\boldsymbol e_2$。利用线性性，

$$
\begin{aligned}
Df(a,b)[(h,k)]
&=
hDf(a,b)[\boldsymbol e_1]
+kDf(a,b)[\boldsymbol e_2]\\
&=
f_x(a,b)h+f_y(a,b)k\\
&=
\nabla f(a,b)\cdot(h,k)
\end{aligned}
$$

因此，梯度是全导数这个线性映射在欧氏内积下的向量表示；全导数本身是“输入位移到输出变化”的规则。梯度写成列向量时，全导数写成同一组数的行向量：

$$
\nabla f(a,b)
=
\begin{pmatrix}
f_x(a,b)\\
f_y(a,b)
\end{pmatrix}
\qquad
Df(a,b)
=
\begin{pmatrix}
f_x(a,b)&f_y(a,b)
\end{pmatrix}
$$

两种写法表达同一个线性作用：

$$
Df(a,b)
\begin{pmatrix}
h\\k
\end{pmatrix}
=
\nabla f(a,b)\cdot(h,k)
$$

这也是 [梯度](../calculus/gradient/) 与全导数最容易混淆的地方：梯度是向量，全导数是线性映射；在标量输出和标准欧氏坐标下，它们可以由转置互相表示。

## 一个二元函数的全导数

取

$$
f(x,y)=x^2+3xy+y^2
$$

在 $(1,2)$，

$$
f(1,2)=11
\qquad
\nabla f(1,2)=(8,7)
$$

因此对任意小位移 $(h,k)$，

$$
Df(1,2)[(h,k)]=8h+7k
$$

局部线性模型是

$$
f(1+h,2+k)
\approx
11+8h+7k
$$

取 $(h,k)=(0.1,-0.2)$：

$$
Df(1,2)[(0.1,-0.2)]
=0.8-1.4
=-0.6
$$

所以线性模型预测函数从 $11$ 变为 $10.4$。原函数直接计算：

$$
f(1.1,1.8)=10.39
$$

实际变化是 $-0.61$，与全导数的预测相差 $-0.01$。这里全导数接受的不是一个特殊的坐标方向，而是整个二维位移；沿哪个方向走，只是选择了传给这个映射的 $\boldsymbol h$。

对线性函数，余项会完全消失。比如

$$
q(x,y)=3x-2y+1
$$

对任意点和任意位移都有

$$
q(x+h,y+k)-q(x,y)=3h-2k=Dq(x,y)[(h,k)]
$$

全导数在这里不是近似，而是精确的变化公式。非线性函数才需要用 $o(\|\boldsymbol h\|_2)$ 描述剩余。

![全导数把输入位移映射为局部的输出变化](/assets/calculus/svg/total-derivative.1.svg)

## 方向导数和路径导数只是全导数的取值

如果 $\boldsymbol u$ 是单位向量，方向导数就是把 $\boldsymbol u$ 传给全导数：

$$
D_{\boldsymbol u}f(\boldsymbol a)=Df(\boldsymbol a)[\boldsymbol u]
$$

方向导数只比较单位位移；全导数可以接受任意长度的位移，所以

$$
Df(\boldsymbol a)[c\boldsymbol u]
=cD_{\boldsymbol u}f(\boldsymbol a)
$$

对一条参数路径 $\boldsymbol r(t)$，复合函数的导数则是

$$
\frac{\mathrm d}{\mathrm dt}f(\boldsymbol r(t))
=
Df(\boldsymbol r(t))[\boldsymbol r'(t)]
$$

这就是 [链式法则](../calculus/chain-rule/) 在线性映射记号下的写法：先用路径导数 $\boldsymbol r'(t)$ 产生输入扰动，再由全导数把它变成输出变化。

回到

$$
f(x,y)=x^2+3xy+y^2
$$

取路径

$$
\boldsymbol r(t)=(1+t,2+2t)
$$

在 $t=0$，

$$
\boldsymbol r'(0)=(1,2)
\qquad
Df(1,2)[(1,2)]=8+14=22
$$

直接把路径代入：

$$
\begin{aligned}
f(1+t,2+2t)
&=(1+t)^2+3(1+t)(2+2t)+(2+2t)^2\\
&=11+22t+11t^2
\end{aligned}
$$

在 $t=0$ 求导也是 $22$。同一组偏导在不同路径上会和不同的路径速度相乘；全导数把这个组合压缩成一次线性映射调用。

## 向量输出时，全导数仍然是线性映射

如果输出不再是一个数，而是向量，例如

$$
F(x,y)=
\begin{pmatrix}
x^2+3y\\
xy
\end{pmatrix}
$$

在 $(1,2)$ 对位移 $(h,k)$ 的一阶变化为

$$
DF(1,2)[(h,k)]
=
\begin{pmatrix}
2h+3k\\
2h+k
\end{pmatrix}
$$

把输入位移写成列向量，这个线性映射的矩阵是

$$
DF(1,2)
=
\begin{pmatrix}
2&3\\
2&1
\end{pmatrix}
$$

矩阵的第 $j$ 列描述第 $j$ 个输入坐标单独移动时的输出变化，矩阵乘以 $(h,k)^{\mathsf T}$ 后得到两个输出分量的线性预测。这个矩阵就是 [雅可比矩阵](../calculus/jacobian/)；标量函数的情形只是只有一行，不能把一般向量输出也压成一个梯度向量。

更一般地，若 $G:\mathbb R^n\to\mathbb R^p$、$F:\mathbb R^p\to\mathbb R^m$ 都可微，则

$$
D(F\circ G)(\boldsymbol a)
=
DF(G(\boldsymbol a))\circ DG(\boldsymbol a)
$$

用矩阵表示时，复合映射对应矩阵相乘；用线性映射表示时，先作用 $DG$，再作用 $DF$。[向量链式法则](../calculus/vector-chain-rule/) 会把这个结构用于更多坐标和矩阵形状。

## 机器学习：把参数扰动送进损失

把参数 $\boldsymbol w=(w_1,w_2)$ 看成输入，把损失看成标量函数：

$$
L(w_1,w_2)=w_1^2+3w_1w_2+w_2^2
$$

在 $\boldsymbol w_0=(1,2)$，

$$
L(\boldsymbol w_0)=11
\qquad
DL(\boldsymbol w_0)[(\delta_1,\delta_2)]
=8\delta_1+7\delta_2
$$

一次参数扰动 $\boldsymbol\delta$ 的一阶损失变化，直接由这个线性映射给出。梯度下降取

$$
\boldsymbol\delta=-\eta\nabla L(\boldsymbol w_0)
=(-8\eta,-7\eta)
$$

于是

$$
DL(\boldsymbol w_0)[\boldsymbol\delta]
=-113\eta
$$

对 $\eta=0.01$，一阶预测是 $11-1.13=9.87$，新参数为 $(0.92,1.93)$，原损失的真实值为

$$
L(0.92,1.93)=9.8981
$$

全导数给出的是当前点的线性预测；真实更新还包含二阶及更高阶项。步长变大时，线性预测的误差可能迅速增加，这也是 [梯度](../calculus/gradient/) 给方向而学习率还需要单独选择的原因。

## 用不同位移检验同一个全导数

固定 $(1,2)$，取方向向量

$$
\boldsymbol v=(0.1,-0.2)
\qquad
Df(1,2)[\boldsymbol v]=-0.6
$$

把实际位移缩放成 $\varepsilon\boldsymbol v$。对于当前二次函数，直接计算实际变化和线性预测如下：

| 缩放 $\varepsilon$ | 实际函数变化 | 全导数预测 |
| --- | --- | --- |
| $1$ | $f(1.1,1.8)-f(1,2)=-0.61$ | $-0.6$ |
| $0.1$ | $f(1.01,1.98)-f(1,2)=-0.0601$ | $-0.06$ |
| $0.01$ | $f(1.001,1.998)-f(1,2)=-0.006001$ | $-0.006$ |

当 $\varepsilon$ 从 $1$ 缩到 $0.01$，实际变化与线性预测的差从 $-0.01$ 缩到 $-0.000001$。误差相对于位移长度也在缩小，这正是全导数定义中的余项条件。

对任意向量 $\boldsymbol h$，标量输出还满足

$$
\left|Df(\boldsymbol a)[\boldsymbol h]\right|
=
\left|\nabla f(\boldsymbol a)\cdot\boldsymbol h\right|
\le
\|\nabla f(\boldsymbol a)\|_2\|\boldsymbol h\|_2
$$

所以梯度长度既是单位方向的最大变化率，也是这个线性映射在欧氏范数下的最大放大倍数。

## 常见失效模式

- **把全导数当成一个数。** 对标量输出，全导数作用在位移上才得到一个数；$Df(\boldsymbol a)$ 本身仍是线性映射，不能和某个点的函数值混为一谈。
- **只检查坐标轴就宣布可微。** $f_x$ 和 $f_y$ 只是全导数在两个基向量上的取值。分别存在偏导并不自动保证所有斜方向能由同一个线性映射统一近似；[偏导数](../calculus/partial-derivatives/) 篇的反例在原点两个偏导都存在，但函数不连续。
- **把一阶近似当成有限步长的恒等式。** 只有线性函数的余项恒为零；非线性函数的全导数只描述一个点附近的首阶变化。
- **忽略输入和输出的形状。** 标量输出对应一行线性映射，向量输出需要矩阵或更一般的线性算子。写更新式和链式法则时，必须检查矩阵相乘的维度。
- **混淆方向导数与全导数。** 方向导数只把单位向量传入全导数；全导数还能处理任意长度的组合位移和路径速度。

## 相关词条

- [偏导数](../calculus/partial-derivatives/)：沿单个坐标方向定义变化率。
- [梯度](../calculus/gradient/)：标量函数全导数在欧氏内积下的向量表示。
- [导数](../calculus/derivatives/)：从一元差商极限和局部线性近似开始。
- [链式法则](../calculus/chain-rule/)：沿路径组合局部变化率。
- [雅可比矩阵](../calculus/jacobian/)：向量值函数全导数的矩阵表示。
- [向量链式法则](../calculus/vector-chain-rule/)：组合多个线性映射的导数。
- [Hessian 矩阵](../calculus/hessian/)：记录二阶偏导和局部曲率。
- [梯度下降](../training-nn/gradient-descent/)：使用损失梯度反方向更新参数。
