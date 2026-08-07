---
title: "梯度下降理论：步长、光滑性与收敛保证"
tags: ["why-models-learn"]
---

梯度下降重复执行一条更新：计算当前梯度，沿反方向移动一小步，再在新位置重新计算。梯度决定方向，学习率决定步长。第一遍只需理解更新式和一维数字例子；L-光滑性、下降引理与收敛速度用于回答「为什么这个步长安全」，放在后半部分展开。

## 从局部线性近似得到更新

在点 $\boldsymbol x$ 附近沿位移 $\boldsymbol d$ 移动，局部线性近似是

$$
f(\boldsymbol x+\boldsymbol d)
\approx
f(\boldsymbol x)
+\nabla f(\boldsymbol x)^\mathsf T\boldsymbol d
$$

如果只比较同样长度的位移，柯西–施瓦茨不等式给出

$$
\nabla f(\boldsymbol x)^\mathsf T\boldsymbol d
\geq
-\|\nabla f(\boldsymbol x)\|_2\|\boldsymbol d\|_2
$$

等号在 $\boldsymbol d$ 与 $-\nabla f(\boldsymbol x)$ 同向时达到。因此，梯度给出局部线性模型中下降最快的方向。把移动距离写成梯度长度乘以正标量 $\eta$：

$$
\boldsymbol d=-\eta\nabla f(\boldsymbol x)
$$

就得到梯度下降更新

$$
\boldsymbol x_{k+1}
=\boldsymbol x_k-\eta\nabla f(\boldsymbol x_k)
$$

这里 $\eta$ 是学习率或步长。它不是方向信息的另一种写法，而是把方向向量放大或缩小的尺度参数。

### 一维例子

取

$$
f(x)=(x-3)^2
$$

则

$$
\nabla f(x)=2(x-3)
$$

用 $\eta=0.25$，更新式变成

$$
x_{k+1}
=x_k-0.25\cdot2(x_k-3)
=0.5x_k+1.5
$$

从 $x_0=0$ 出发：

$$
\begin{aligned}
x_1&=1.5,& f(x_1)&=2.25\\
x_2&=2.25,& f(x_2)&=0.5625\\
x_3&=2.625,& f(x_3)&=0.140625
\end{aligned}
$$

参数逐步接近 $3$，函数值也逐步下降。这个例子还没有说明 $\eta=0.25$ 为什么安全；要得到保证，必须控制二阶余项。

## 第一遍到这里需要掌握什么

梯度下降的更新式是

$$
\boldsymbol x_{k+1}
=\boldsymbol x_k-\eta\nabla f(\boldsymbol x_k)
$$

负号表示沿局部下降方向移动，$\eta>0$ 控制移动距离。每一步都必须在新位置重新计算梯度。学习率过小时，目标下降缓慢；学习率过大时，参数可能越过低点并产生振荡或发散。

后续训练词条第一次使用梯度下降时，以上内容已经足够。下面开始讨论步长保证和收敛速度，需要用到 Hessian、特征值、谱范数与凸性；这些概念可以在需要证明时回补。

> [!marginnote] 第二遍内容
> 如果当前目标是理解模型如何更新参数，可以从这里跳到「相关词条」。需要判断学习率范围或阅读收敛证明时，再继续下面的形式理论。

## L-光滑性控制二阶余项

称可微函数 $f$ 的梯度是 $L$-Lipschitz 连续的，或称 $f$ 是 $L$-光滑的，如果任意 $\boldsymbol x,\boldsymbol y$ 都满足

$$
\|\nabla f(\boldsymbol x)-\nabla f(\boldsymbol y)\|_2
\leq
L\|\boldsymbol x-\boldsymbol y\|_2
$$

$L$ 衡量的是梯度变化得有多快。它不是函数值本身的上限，也不是梯度的上限；函数可以在很远处很大，仍然拥有有限的梯度 Lipschitz 常数。

当 $f$ 二阶连续可微时，一个容易使用的充分条件是

$$
\|\nabla^2f(\boldsymbol x)\|_{\mathrm{op}}
\leq L
\qquad
\text{for all }\boldsymbol x
$$

其中 $\|\cdot\|_{\mathrm{op}}$ 是谱范数。对称 Hessian 的最大特征值绝对值不超过 $L$，就满足这个条件。二次函数的 Hessian 是常数矩阵，所以它的 $L$ 可以直接从特征值读出。

### 下降引理

L-光滑性给出一条二次上界：

$$
f(\boldsymbol x+\boldsymbol d)
\leq
f(\boldsymbol x)
+\nabla f(\boldsymbol x)^\mathsf T\boldsymbol d
+\frac L2\|\boldsymbol d\|_2^2
$$

这条不等式常称为下降引理。它不是把 Taylor 展开截断后当成等式，而是为余项提供了一个方向无关的上界。

推导如下。令

$$
\phi(t)=f(\boldsymbol x+t\boldsymbol d),
\qquad
0\leq t\leq1
$$

沿线积分得到

$$
f(\boldsymbol x+\boldsymbol d)-f(\boldsymbol x)
=\int_0^1\nabla f(\boldsymbol x+t\boldsymbol d)^\mathsf T\boldsymbol d\,\mathrm dt
$$

在积分中加上再减去 $\nabla f(\boldsymbol x)$：

$$
\begin{aligned}
f(\boldsymbol x+\boldsymbol d)-f(\boldsymbol x)
&=\nabla f(\boldsymbol x)^\mathsf T\boldsymbol d\\
&\quad+\int_0^1
\bigl(\nabla f(\boldsymbol x+t\boldsymbol d)-\nabla f(\boldsymbol x)\bigr)^\mathsf T
\boldsymbol d\,\mathrm dt
\end{aligned}
$$

Lipschitz 条件和柯西–施瓦茨不等式把积分项控制为

$$
\begin{aligned}
\left|
\bigl(\nabla f(\boldsymbol x+t\boldsymbol d)-\nabla f(\boldsymbol x)\bigr)^\mathsf T\boldsymbol d
\right|
&\leq
Lt\|\boldsymbol d\|_2^2\\
\int_0^1Lt\|\boldsymbol d\|_2^2\,\mathrm dt
&=\frac L2\|\boldsymbol d\|_2^2
\end{aligned}
$$

于是得到下降引理。

### 代入梯度步

把

$$
\boldsymbol d=-\eta\boldsymbol g,
\qquad
\boldsymbol g=\nabla f(\boldsymbol x)
$$

代入二次上界：

$$
\begin{aligned}
f(\boldsymbol x-\eta\boldsymbol g)
&\leq
f(\boldsymbol x)
-\eta\|\boldsymbol g\|_2^2
+\frac L2\eta^2\|\boldsymbol g\|_2^2\\
&=
f(\boldsymbol x)
-\eta\left(1-\frac{L\eta}{2}\right)\|\boldsymbol g\|_2^2
\end{aligned}
$$

因此只要

$$
0<\eta\leq\frac1L
$$

就有

$$
f(\boldsymbol x_{k+1})
\leq
f(\boldsymbol x_k)
-\frac{\eta}{2}\|\nabla f(\boldsymbol x_k)\|_2^2
$$

每一步至少实现半个线性模型下降量。这里的「至少」来自上界：实际函数可能下降得更多，但在只知道 $L$-光滑性的前提下，不能把更大的下降量当作定理。

## 二次函数把稳定性写成特征值

考虑最容易完整分析的二次目标

$$
f(\boldsymbol x)
=\frac12\boldsymbol x^\mathsf TQ\boldsymbol x
$$

其中 $Q=Q^\mathsf T\succ0$。它的唯一最小点是 $\boldsymbol0$，梯度和 Hessian 分别为

$$
\nabla f(\boldsymbol x)=Q\boldsymbol x,
\qquad
\nabla^2f(\boldsymbol x)=Q
$$

设 $Q$ 的特征分解为

$$
Q=U\Lambda U^\mathsf T,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_d)
$$

并用特征坐标 $\boldsymbol z_k=U^\mathsf T\boldsymbol x_k$ 表示误差。梯度下降在每个特征方向上独立变成

$$
\boldsymbol z_{k+1}
=(I-\eta\Lambda)\boldsymbol z_k
$$

也就是

$$
z_{k+1,i}
=(1-\eta\lambda_i)z_{k,i}
$$

要让第 $i$ 个方向的误差趋于零，必须有

$$
|1-\eta\lambda_i|<1
$$

对所有 $i$ 成立。由于最大的特征值记为

$$
L=\lambda_{\max}(Q)
$$

所以二次正定目标的精确收敛条件是

$$
0<\eta<\frac2L
$$

这比下降引理给出的 $\eta\leq1/L$ 宽。两者回答的问题不同：

| 步长范围 | 能保证什么 | 几何表现 |
| --- | --- | --- |
| $0<\eta\leq1/L$ | 每一步由下降引理保证下降 | 特征方向不换号，通常平稳靠近 |
| $1/L<\eta<2/L$ | 二次正定目标仍然收敛 | 高曲率方向可能来回振荡 |
| $\eta=2/L$ | 最大曲率方向因子为 $-1$ | 该方向不衰减 |
| $\eta>2/L$ | 至少一个方向因子绝对值大于 1 | 高曲率方向发散 |

### 数值例子：同一个目标的两个步长

取

$$
Q=
\begin{bmatrix}
1&0\\
0&9
\end{bmatrix},
\qquad
\boldsymbol x_0=
\begin{bmatrix}1\\1\end{bmatrix}
$$

此时 $L=9$，初始函数值为

$$
f(\boldsymbol x_0)
=\frac12(1+9)
=5
$$

先取 $\eta=0.1$，它小于 $1/L\approx0.111111$。两个坐标的缩放因子是 $0.9$ 和 $0.1$：

$$
\boldsymbol x_1=
\begin{bmatrix}0.9\\0.1\end{bmatrix},
\qquad
f(\boldsymbol x_1)
=\frac12(0.9^2+9\cdot0.1^2)
=0.45
$$

再走一步：

$$
\boldsymbol x_2=
\begin{bmatrix}0.81\\0.01\end{bmatrix},
\qquad
f(\boldsymbol x_2)
=\frac12(0.81^2+9\cdot0.01^2)
=0.3285
$$

低曲率方向的因子是 $0.9$，所以它下降得慢；高曲率方向的因子是 $0.1$，所以它几乎立即被压下去。

换成 $\eta=0.25$，已经超过 $2/L=2/9\approx0.222222$。第一步变成

$$
\boldsymbol x_1=
\begin{bmatrix}
1-0.25\\
1-2.25
\end{bmatrix}
=
\begin{bmatrix}
0.75\\
-1.25
\end{bmatrix}
$$

函数值反而是

$$
f(\boldsymbol x_1)
=\frac12(0.75^2+9\cdot1.25^2)
=7.3125
$$

高曲率方向的因子 $1-0.25\cdot9=-1.25$，负号表示振荡，绝对值大于 $1$ 表示振荡幅度还在放大。只看「梯度指向下降」而不看这个缩放因子，就解释不了这次失败。

## 凸性把下降升级为函数值收敛

光滑性控制的是「一步不要走坏」，凸性控制的是「当前函数值距离全局最小值还有多远」。可微函数 $f$ 是凸函数，当且仅当

$$
f(\boldsymbol y)
\geq
f(\boldsymbol x)
+\nabla f(\boldsymbol x)^\mathsf T(\boldsymbol y-\boldsymbol x)
$$

对任意 $\boldsymbol x,\boldsymbol y$ 成立。右侧是当前位置的一阶支撑平面；凸性要求函数图像永远不低于这个平面。

设 $\boldsymbol x^\star$ 是全局最小点，且 $f$ 凸并且 $L$-光滑。取 $\eta=1/L$，下降引理给出

$$
f(\boldsymbol x_{k+1})
\leq
f(\boldsymbol x_k)
-\frac1{2L}\|\nabla f(\boldsymbol x_k)\|_2^2
$$

凸性又给出

$$
f(\boldsymbol x_k)-f(\boldsymbol x^\star)
\leq
\nabla f(\boldsymbol x_k)^\mathsf T
(\boldsymbol x_k-\boldsymbol x^\star)
$$

令 $\boldsymbol g_k=\nabla f(\boldsymbol x_k)$，直接展开距离平方：

$$
\begin{aligned}
\frac L2\left(
\|\boldsymbol x_k-\boldsymbol x^\star\|_2^2
-\|\boldsymbol x_{k+1}-\boldsymbol x^\star\|_2^2
\right)
&=
\boldsymbol g_k^\mathsf T(\boldsymbol x_k-\boldsymbol x^\star)
-\frac1{2L}\|\boldsymbol g_k\|_2^2\\
&\geq
f(\boldsymbol x_{k+1})-f(\boldsymbol x^\star)
\end{aligned}
$$

于是

$$
f(\boldsymbol x_{k+1})-f(\boldsymbol x^\star)
\leq
\frac L2\left(
\|\boldsymbol x_k-\boldsymbol x^\star\|_2^2
-\|\boldsymbol x_{k+1}-\boldsymbol x^\star\|_2^2
\right)
$$

右侧会望远镜相消。因为函数值沿迭代不增加，累加前 $k$ 步后得到

$$
f(\boldsymbol x_k)-f(\boldsymbol x^\star)
\leq
\frac{L\|\boldsymbol x_0-\boldsymbol x^\star\|_2^2}{2k},
\qquad
k\geq1
$$

这就是凸且光滑目标的 $O(1/k)$ 函数值收敛率。它告诉我们达到函数值误差 $\varepsilon$ 至少需要数量级

$$
k=O\left(\frac{L\|\boldsymbol x_0-\boldsymbol x^\star\|_2^2}{\varepsilon}\right)
$$

的迭代；这是上界量级，不是每个数据集都会恰好用这么多步。

## 强凸性带来线性收敛

如果凸函数在所有方向都有统一的二次余量，就称它是 $\mu$-强凸的：

$$
f(\boldsymbol y)
\geq
f(\boldsymbol x)
+\nabla f(\boldsymbol x)^\mathsf T(\boldsymbol y-\boldsymbol x)
+\frac\mu2\|\boldsymbol y-\boldsymbol x\|_2^2
$$

其中 $\mu>0$。对二次函数，这相当于所有特征值都不小于 $\mu$。于是目标同时有最小曲率 $\mu$ 和最大曲率 $L$，条件数为

$$
\kappa=\frac L\mu
$$

先从强凸性得到一个梯度平方下界。令 $\boldsymbol x^\star$ 为最小点，在强凸不等式中取 $\boldsymbol y=\boldsymbol x^\star$：

$$
f(\boldsymbol x^\star)
\geq
f(\boldsymbol x)
+\boldsymbol g^\mathsf T(\boldsymbol x^\star-\boldsymbol x)
+\frac\mu2\|\boldsymbol x^\star-\boldsymbol x\|_2^2
$$

整理并令 $\boldsymbol r=\boldsymbol x-\boldsymbol x^\star$：

$$
f(\boldsymbol x)-f(\boldsymbol x^\star)
\leq
\boldsymbol g^\mathsf T\boldsymbol r
-\frac\mu2\|\boldsymbol r\|_2^2
\leq
\frac1{2\mu}\|\boldsymbol g\|_2^2
$$

最后一个不等式来自对 $\boldsymbol r$ 最大化右侧二次式。因此

$$
\|\nabla f(\boldsymbol x)\|_2^2
\geq
2\mu\bigl(f(\boldsymbol x)-f(\boldsymbol x^\star)\bigr)
$$

把它代回 $\eta=1/L$ 的下降引理：

$$
\begin{aligned}
f(\boldsymbol x_{k+1})-f(\boldsymbol x^\star)
&\leq
f(\boldsymbol x_k)-f(\boldsymbol x^\star)
-\frac1{2L}\|\nabla f(\boldsymbol x_k)\|_2^2\\
&\leq
\left(1-\frac\mu L\right)
\bigl(f(\boldsymbol x_k)-f(\boldsymbol x^\star)\bigr)
\end{aligned}
$$

递推 $k$ 次：

$$
f(\boldsymbol x_k)-f(\boldsymbol x^\star)
\leq
\left(1-\frac1\kappa\right)^k
\bigl(f(\boldsymbol x_0)-f(\boldsymbol x^\star)\bigr)
$$

这称为线性收敛，意思是误差每一步乘上固定小于 $1$ 的因子，而不是说误差关于 $k$ 呈一条直线。因为

$$
\left(1-\frac1\kappa\right)^k
\leq
\exp\left(-\frac k\kappa\right)
$$

要把误差缩小到原来的 $\varepsilon$ 倍，迭代次数的量级是 $\kappa\log(1/\varepsilon)$。条件数大时，细长谷底中的低曲率方向会拖慢整体收敛。

### 二次目标的最佳固定步长

对特征值都落在 $[\mu,L]$ 的二次函数，误差每个方向的因子是 $1-\eta\lambda$。若要最小化最坏方向的绝对因子，需要让两端曲率的因子大小相等、符号相反：

$$
1-\eta\mu
=-(1-\eta L)
$$

解得

$$
\eta_{\mathrm{opt}}
=\frac2{L+\mu},
\qquad
\max_{\lambda\in[\mu,L]}|1-\eta_{\mathrm{opt}}\lambda|
=\frac{L-\mu}{L+\mu}
=\frac{\kappa-1}{\kappa+1}
$$

这是已知 $\mu,L$ 时针对二次谱的最坏方向误差步长，不是所有非二次、非凸目标都可以直接套用的万能学习率。它也说明预条件的目标：把不同方向的有效曲率压缩到更窄的区间。

## 非凸光滑目标只能保证接近驻点

神经网络损失通常不满足全局凸性。假设 $f$ 仍是 $L$-光滑，且有有限下界

$$
f_\star=\inf_{\boldsymbol x}f(\boldsymbol x)>-\infty
$$

取 $\eta=1/L$，每一步都有

$$
\frac1{2L}\|\nabla f(\boldsymbol x_k)\|_2^2
\leq
f(\boldsymbol x_k)-f(\boldsymbol x_{k+1})
$$

对 $k=0,\ldots,T-1$ 求和：

$$
\frac1{2L}\sum_{k=0}^{T-1}
\|\nabla f(\boldsymbol x_k)\|_2^2
\leq
f(\boldsymbol x_0)-f(\boldsymbol x_T)
\leq
f(\boldsymbol x_0)-f_\star
$$

所以

$$
\frac1T\sum_{k=0}^{T-1}
\|\nabla f(\boldsymbol x_k)\|_2^2
\leq
\frac{2L\bigl(f(\boldsymbol x_0)-f_\star\bigr)}T
$$

至少存在一个迭代点满足

$$
\min_{0\leq k<T}
\|\nabla f(\boldsymbol x_k)\|_2^2
\leq
\frac{2L\bigl(f(\boldsymbol x_0)-f_\star\bigr)}T
$$

这给出的是 $\varepsilon$-驻点保证：若要求 $\|\nabla f\|_2\leq\varepsilon$，需要

$$
T\geq
\frac{2L\bigl(f(\boldsymbol x_0)-f_\star\bigr)}{\varepsilon^2}
$$

注意这里是梯度范数的平方，所以迭代复杂度对 $\varepsilon$ 是 $O(1/\varepsilon^2)$，不是凸情形函数值误差里的 $O(1/\varepsilon)$。

非凸保证不等于找到全局最小点。最简单的有界反例是

$$
f(x)=\sin x
$$

它满足 $|f''(x)|\leq1$，所以可以取 $L=1$；但 $x=\pi/2$ 的梯度为零，函数值却是局部最大值 $1$。梯度下降从这个点开始不会移动。光滑性负责控制一步，非凸性阻止我们把驻点自动解释成全局谷底。

如果函数虽非凸却满足 Polyak–Łojasiewicz 不等式

$$
\frac12\|\nabla f(\boldsymbol x)\|_2^2
\geq
\mu\bigl(f(\boldsymbol x)-f^\star\bigr)
$$

那么同样的下降引理会给出

$$
f(\boldsymbol x_{k+1})-f^\star
\leq
\left(1-\frac\mu L\right)
\bigl(f(\boldsymbol x_k)-f^\star\bigr)
$$

PL 条件可以带来线性函数值收敛，但它本身不要求函数是凸的；因此「非凸」和「没有任何速率」不是同一句话。

## 不知道 L 时用回溯线搜索

实际目标的全局 $L$ 往往未知，直接把 Hessian 最大特征值算出来又不现实。回溯线搜索从一个候选步长 $\eta_0$ 开始，检查 Armijo 条件：

$$
f(\boldsymbol x-\eta\boldsymbol g)
\leq
f(\boldsymbol x)
-c\eta\|\boldsymbol g\|_2^2,
\qquad
0<c<1
$$

如果不满足，就乘上 $0<\beta<1$：

$$
\eta\leftarrow\beta\eta
$$

直到条件成立，再更新参数。由下降引理可知，只要

$$
\eta\leq\frac{2(1-c)}L
$$

这个条件就一定成立。因此在 $L$ 有限时，几何缩小最终会找到可接受步长。

回溯线搜索把「猜一个全局学习率」换成「在当前点测试一个足够安全的局部步长」，代价是每次候选步长都要额外计算函数值。对小批次损失，函数值本身带随机噪声，直接套用确定性 Armijo 条件会变得不稳定；随机梯度下降需要另一套期望意义下的分析。

## 停止条件测量的是不同东西

梯度下降运行到某一步时，常见的停止信号并不等价：

| 停止信号 | 它实际检查什么 | 容易误读的地方 |
| --- | --- | --- |
| $\|\nabla f(\boldsymbol x_k)\|_2\leq\varepsilon$ | 一阶驻点残差 | 非凸时不保证极小，尺度改变也会改变梯度数值 |
| $\|\boldsymbol x_{k+1}-\boldsymbol x_k\|_2\leq\varepsilon$ | 本次移动的长度 | 等于 $\eta\|\nabla f\|$，小学习率会制造假象 |
| $\lvert f(\boldsymbol x_{k+1})-f(\boldsymbol x_k)\rvert\leq\varepsilon$ | 一步函数值变化 | 平坦区域、数值精度和损失抵消都可能让它很小 |
| 达到最大迭代数 | 计算预算 | 不是数学上的收敛证据 |

在强凸情形，梯度还可以给出函数值差的上界：

$$
f(\boldsymbol x)-f(\boldsymbol x^\star)
\leq
\frac1{2\mu}\|\nabla f(\boldsymbol x)\|_2^2
$$

但没有强凸性时，梯度很小只能说明一阶信息很小。它可能是局部最小、鞍点、平坦区域，或者只是浮点数已经无法分辨更小的变化。报告训练结果时，最好同时记录梯度范数、目标值、参数步长和验证集指标。

## 神经网络中的理论边界

把经验损失写成

$$
L(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
\ell\bigl(f_{\boldsymbol\theta}(\boldsymbol x_i),y_i\bigr)
$$

之后，梯度下降理论可以解释一个完整批次更新在什么条件下会下降。但深层网络的损失通常有以下差异：

- 全局 $L$ 可能很大，甚至在整个参数空间没有有用的统一估计；
- 网络一般非凸，梯度范数变小只给驻点证据；
- 参数重参数化会改变梯度尺度和有效条件数；
- 实际训练多用小批次梯度，噪声会打破每一步确定性下降；
- 学习率预热、衰减、权重衰减和梯度裁剪会改变原始更新式。

因此，理论中的 $\eta\leq1/L$ 是一个假设明确的充分条件，不是训练脚本里所有学习率都必须小于某个固定数字。它提供的是读懂实验的基线：损失突然上升时先查步长与局部曲率，收敛很慢时再查条件数、尺度和噪声来源。

![梯度下降的步长受最大曲率约束](/assets/optimization-theory/svg/gradient-descent-theory.1.svg)

## 失效模式

**把 $\eta\leq1/L$ 和 $\eta<2/L$ 当成同一个结论。** 前者由下降引理保证每步下降，后者在正定二次函数上保证误差收敛；中间区间可以振荡但仍收敛。

**只看梯度方向，不看曲率。** 高曲率方向的梯度可能很大，固定步长会在该方向越过谷底；二次函数中的因子 $1-\eta\lambda_i$ 能直接暴露这个问题。

**把凸函数的 $O(1/k)$ 率套到神经网络。** 这个率需要凸性、光滑性和全局最小点等假设；非凸光滑目标通常只能先保证某个点的梯度范数变小。

**把梯度范数为零写成找到最小值。** $\sin x$ 在 $x=\pi/2$ 的梯度为零，但那里是最大值；二阶信息或函数结构仍然必要。

**把全局 $L$ 当成容易知道的常数。** 大模型的参数空间可能没有实用的全局上界；可以用局部回溯、学习率试验和曲率诊断，但不能把试验值冒充定理中的 $L$。

**把训练损失的确定性下降当成小批次的必然现象。** 随机梯度的单步目标值可能上升，理论结论通常要改写成期望、方差或高概率界。

**只用一步损失变化作为停止条件。** 平坦区和小学习率都会让它变小；若没有梯度范数、验证集或预算信息，不能判断是否接近目标。

**忽略坐标尺度和条件数。** 同一个目标换一种参数化后，梯度大小和最大曲率都会变化；低曲率方向拖慢的现象可能需要预条件或重新缩放来处理。

## 相关词条

- [梯度](../calculus/gradient/)：从方向导数和局部线性近似定义下降方向。
- [Taylor 级数](../calculus/taylor-series/)：解释一次模型和二阶余项为什么决定步长安全性。
- [Hessian 矩阵](../calculus/hessian/)：用二阶偏导和特征值描述局部曲率。
- [凸集与凸函数](../optimization-theory/convex-sets-and-functions/)：说明凸性为什么能把局部信息连接到全局函数值。
- [局部与全局最小值](../optimization-theory/local-and-global-minima/)：区分局部停点、全局谷底和吸引盆。
- [一阶最优性条件](../optimization-theory/first-order-optimality/)：整理无约束梯度条件、KKT、次梯度和法锥。
- [二阶最优性条件](../optimization-theory/second-order-optimality/)：在一阶条件之外检查曲率和高阶退化。
- [随机梯度下降理论](../optimization-theory/stochastic-gradient-descent-theory/)：把确定性下降改写成带噪梯度的期望与方差分析。
- [动量方法理论](../optimization-theory/momentum-theory/)：研究累积历史方向后如何改变振荡和收敛率。
- [自适应学习率](../optimization-theory/adaptive-learning-rates/)：比较按坐标调整步长的方法与固定步长假设。
- [曲率与条件数](../optimization-theory/curvature-and-conditioning/)：进一步分析特征值跨度如何限制优化速度。
- [损失景观](../optimization-theory/loss-landscapes/)：把非凸参数空间中的谷底、鞍点和平坦方向放在更大尺度观察。
