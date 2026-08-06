---
title: "Hessian 矩阵:二阶变化与局部曲率"
tags: ["why-models-learn"]
---

对于标量函数 $f:\mathbb R^n\to\mathbb R$，**Hessian 矩阵**收集梯度各分量的一阶变化，也就是所有二阶偏导：

$$
H_f(\boldsymbol a)
=
J_{\nabla f}(\boldsymbol a)
=
\left[
\frac{\partial^2 f}{\partial x_i\partial x_j}(\boldsymbol a)
\right]_{n\times n}
$$

它回答的不是“当前往哪边走得快”，而是“梯度在不同方向怎样改变”。在可用二阶 Taylor 近似的点附近，

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

一次项由梯度给出，二次项由 Hessian 给出。本篇从梯度的变化推导这个矩阵，再用方向曲率、特征方向、极值分类和损失更新解释它在机器学习中的作用。

## Hessian 是梯度的 Jacobian

在二维中，

$$
\nabla f(x,y)=
\begin{pmatrix}
f_x(x,y)\\
f_y(x,y)
\end{pmatrix}
$$

对这个向量求 Jacobian：

$$
H_f(x,y)
=
\begin{pmatrix}
\dfrac{\partial f_x}{\partial x}&\dfrac{\partial f_x}{\partial y}\\
\dfrac{\partial f_y}{\partial x}&\dfrac{\partial f_y}{\partial y}
\end{pmatrix}
=
\begin{pmatrix}
f_{xx}&f_{xy}\\
f_{yx}&f_{yy}
\end{pmatrix}
$$

如果二阶偏导在邻域内连续，Clairaut 定理保证 $f_{xy}=f_{yx}$，于是 Hessian 是实对称矩阵。对称性让 [谱定理](../linear-algebra/spectral-theorem/) 可以使用正交特征基，把复杂的交叉项拆成互相垂直的曲率方向；没有足够的光滑性时，不能仅凭符号把两个混合偏导交换。

由于 Hessian 是梯度的导数，梯度在小位移下满足

$$
\nabla f(\boldsymbol a+\boldsymbol h)
\approx
\nabla f(\boldsymbol a)+H_f(\boldsymbol a)\boldsymbol h
$$

这是一阶函数近似的下一层：函数值的一阶变化由梯度控制，梯度的一阶变化由 Hessian 控制。

## 二阶 Taylor 公式

沿任意方向 $\boldsymbol h$ 取一条一元路径

$$
g(t)=f(\boldsymbol a+t\boldsymbol h)
$$

一阶链式法则给出

$$
g'(0)=\nabla f(\boldsymbol a)^{\mathsf T}\boldsymbol h
$$

再求一次导数：

$$
g''(0)
=
\boldsymbol h^{\mathsf T}H_f(\boldsymbol a)\boldsymbol h
$$

对 $g$ 使用一元二阶 Taylor 展开，在 $t=1$ 得到

$$
f(\boldsymbol a+\boldsymbol h)
=
f(\boldsymbol a)
+\nabla f(\boldsymbol a)^{\mathsf T}\boldsymbol h
+\frac12\boldsymbol h^{\mathsf T}H_f(\boldsymbol a)\boldsymbol h
+o(\|\boldsymbol h\|_2^2)
$$

其中

$$
\boldsymbol h^{\mathsf T}H_f(\boldsymbol a)\boldsymbol h
$$

是一个二次型。它把方向和曲率结合在一个数里：同一个 Hessian，换一条方向就可能得到不同的二阶变化。

## 一个二元函数的二阶核对

取

$$
f(x,y)=x^2+3xy+y^2
$$

梯度和 Hessian 分别为

$$
\nabla f(x,y)=
\begin{pmatrix}
2x+3y\\
3x+2y
\end{pmatrix}
\qquad
H_f(x,y)=
\begin{pmatrix}
2&3\\
3&2
\end{pmatrix}
$$

它的 Hessian 与位置无关。在 $\boldsymbol a=(1,2)$，

$$
f(\boldsymbol a)=11
\qquad
\nabla f(\boldsymbol a)=
\begin{pmatrix}
8\\7
\end{pmatrix}
$$

取位移

$$
\boldsymbol h=
\begin{pmatrix}
0.1\\-0.2
\end{pmatrix}
$$

一次项为

$$
\nabla f(\boldsymbol a)^{\mathsf T}\boldsymbol h
=(8,7)
\begin{pmatrix}
0.1\\-0.2
\end{pmatrix}
=-0.6
$$

二次项中的二次型为

$$
\begin{aligned}
\boldsymbol h^{\mathsf T}H_f\boldsymbol h
&=
(0.1,-0.2)
\begin{pmatrix}
2&3\\3&2
\end{pmatrix}
\begin{pmatrix}
0.1\\-0.2
\end{pmatrix}\\
&=-0.02
\end{aligned}
$$

所以二阶模型预测

$$
f(\boldsymbol a+\boldsymbol h)
\approx
11-0.6+\frac12(-0.02)
=10.39
$$

这次预测正好等于原二次函数的真实值。因为函数本身只有二次项，三阶及更高阶余项为零；对一般函数，二阶 Taylor 公式仍然只是在局部近似。

## 方向曲率和特征方向

若 $\boldsymbol u$ 是单位向量，沿它走的二阶变化率是

$$
\frac{\mathrm d^2}{\mathrm dt^2}
f(\boldsymbol a+t\boldsymbol u)\bigg|_{t=0}
=
\boldsymbol u^{\mathsf T}H_f(\boldsymbol a)\boldsymbol u
$$

这叫二阶方向导数。对上面的 Hessian，取两个正交单位向量

$$
\boldsymbol u_+
=\frac1{\sqrt2}
\begin{pmatrix}
1\\1
\end{pmatrix}
\qquad
\boldsymbol u_-
=\frac1{\sqrt2}
\begin{pmatrix}
1\\-1
\end{pmatrix}
$$

直接相乘：

$$
H_f\boldsymbol u_+=5\boldsymbol u_+
\qquad
H_f\boldsymbol u_-=-\boldsymbol u_-
$$

因此

$$
\boldsymbol u_+^{\mathsf T}H_f\boldsymbol u_+=5
\qquad
\boldsymbol u_-^{\mathsf T}H_f\boldsymbol u_-=-1
$$

沿 $\boldsymbol u_+$，二阶项使函数向上弯；沿 $\boldsymbol u_-$，二阶项使函数向下弯。对应的特征值 $5$ 和 $-1$ 是两个主曲率方向上的曲率强度。这里 Hessian 不定，和 [二次型](../linear-algebra/quadratic-forms/) 中同一矩阵存在正负方向的情形相同。

如果 Hessian 是对称矩阵，任意单位方向都能在正交特征基下写成

$$
\boldsymbol u=\sum_i c_i\boldsymbol q_i
\qquad
\sum_i c_i^2=1
$$

若 $H\boldsymbol q_i=\lambda_i\boldsymbol q_i$，则

$$
\boldsymbol u^{\mathsf T}H\boldsymbol u
=
\sum_i\lambda_i c_i^2
$$

方向曲率是特征值的加权平均，权重是方向在各特征向量上的平方投影。最大特征值给出最大的二阶上弯方向，最小特征值给出最强的下弯方向。

![Hessian 在两个特征方向上给出正负不同的曲率](/assets/calculus/svg/hessian.1.svg)

## 用 Hessian 分类临界点

只有在 $\nabla f(\boldsymbol a)=0$ 时，Hessian 才能直接判断点附近的二阶形状。若 Hessian 正定，则所有非零小方向的二次项为正，点是严格局部最小值；若负定，则是严格局部最大值；若不定，则同时存在上弯和下弯方向，点是鞍点。

对二维对称矩阵

$$
H=
\begin{pmatrix}
a&b\\b&c
\end{pmatrix}
$$

常用判据是：

| 条件 | 二阶形状 | 临界点结论 |
| --- | --- | --- |
| $a>0$ 且 $\det H>0$ | 正定 | 严格局部最小值 |
| $a<0$ 且 $\det H>0$ | 负定 | 严格局部最大值 |
| $\det H<0$ | 不定 | 鞍点 |
| $\det H=0$ | 半正定或半负定等退化情形 | 二阶检验不能决定 |

几个具体函数：

| 函数和点 | 梯度 | Hessian | 结论 |
| --- | --- | --- | --- |
| $x^2+2y^2$ 在 $(0,0)$ | $(0,0)$ | $\operatorname{diag}(2,4)$ | 严格局部最小 |
| $-x^2-2y^2$ 在 $(0,0)$ | $(0,0)$ | $\operatorname{diag}(-2,-4)$ | 严格局部最大 |
| $x^2-y^2$ 在 $(0,0)$ | $(0,0)$ | $\operatorname{diag}(2,-2)$ | 鞍点 |
| $x^4$ 在 $0$ | $0$ | $0$ | 最小值，二阶检验无结论 |
| $x^3$ 在 $0$ | $0$ | $0$ | 不是极值，二阶检验无结论 |

最后两行说明 Hessian 为零并不意味着函数没有结构，只是二阶项没有提供足够信息，需要继续看更高阶项或直接回到函数定义。反过来，如果梯度不为零，即使 Hessian 正定，也不能把当前点称为局部最小值；函数仍然有一阶下降方向。

## Newton 步骤使用局部二次模型

在点 $\boldsymbol w$，用二阶模型近似损失：

$$
L(\boldsymbol w+\boldsymbol\delta)
\approx
L(\boldsymbol w)
+\nabla L(\boldsymbol w)^{\mathsf T}\boldsymbol\delta
+\frac12\boldsymbol\delta^{\mathsf T}H_L(\boldsymbol w)\boldsymbol\delta
$$

对 $\boldsymbol\delta$ 求导并令其为零：

$$
\nabla L(\boldsymbol w)
+H_L(\boldsymbol w)\boldsymbol\delta=0
$$

如果 Hessian 可逆，Newton 步为

$$
\boldsymbol\delta_{\mathrm{Newton}}
=
-H_L(\boldsymbol w)^{-1}\nabla L(\boldsymbol w)
$$

取一个二次损失

$$
L(w_1,w_2)
=(w_1-2)^2+4(w_2+1)^2
$$

在 $\boldsymbol w_0=(0,0)$：

$$
\nabla L(\boldsymbol w_0)=
\begin{pmatrix}
-4\\8
\end{pmatrix}
\qquad
H_L=
\begin{pmatrix}
2&0\\0&8
\end{pmatrix}
$$

求解

$$
\begin{pmatrix}
2&0\\0&8
\end{pmatrix}
\boldsymbol\delta
=
\begin{pmatrix}
4\\-8
\end{pmatrix}
$$

得到 $\boldsymbol\delta=(2,-1)$，新点正好是 $(2,-1)$，此时损失为 $0$。这是二次函数的特殊情形：局部二次模型就是原函数本身，所以 Newton 一步到达最小点。一般神经网络的 Hessian 随位置变化且可能奇异或不定，不能把这个一步结果直接推广。

## 曲率怎样限制梯度步长

在 Hessian 的特征坐标中考虑

$$
q(a,b)=\frac12(5a^2+b^2)
$$

梯度下降更新为

$$
a_{\mathrm{new}}=(1-5\eta)a
\qquad
b_{\mathrm{new}}=(1-\eta)b
$$

两个方向使用同一个学习率，但曲率为 $5$ 的方向变化更快。若 $\eta=0.2$，$a$ 方向一步缩到 $0$，$b$ 方向乘以 $0.8$；若 $\eta=0.5$，$a$ 方向的因子是 $-1.5$，绝对值大于 $1$，该方向会来回放大。最大特征值因此会限制稳定学习率，条件数大的损失会同时包含快方向和慢方向。

这也是二阶方法想要利用的信息：梯度告诉参数往哪里走，Hessian 告诉不同方向的尺度和弯曲程度。代价是存储和求解矩阵都更昂贵，实践中经常只需要 Hessian-向量积而不显式构造整个 Hessian。

## Hessian-向量积

给定方向 $\boldsymbol v$，

$$
H_f(\boldsymbol a)\boldsymbol v
=
\frac{\mathrm d}{\mathrm dt}
\nabla f(\boldsymbol a+t\boldsymbol v)\bigg|_{t=0}
$$

所以 $H\boldsymbol v$ 是“沿 $\boldsymbol v$ 走时梯度怎样改变”，不必先把 $n^2$ 个矩阵元素全部写出来。对前面的

$$
H=
\begin{pmatrix}
2&3\\3&2
\end{pmatrix}
\qquad
\boldsymbol v=
\begin{pmatrix}
1\\-2
\end{pmatrix}
$$

有

$$
H\boldsymbol v=
\begin{pmatrix}
-4\\-1
\end{pmatrix}
$$

这个结果同时可以从梯度

$$
\nabla f(x,y)=
\begin{pmatrix}
2x+3y\\3x+2y
\end{pmatrix}
$$

沿 $(1,-2)$ 的变化读出。Hessian-向量积是共轭梯度、截断 Newton 和许多大模型二阶近似中常见的接口。

## 用中心差分核对二阶偏导

在 $(1,2)$ 对

$$
f(x,y)=x^2+3xy+y^2
$$

使用中心差分：

$$
\frac{f(1+h,2)-2f(1,2)+f(1-h,2)}{h^2}=2
$$

$$
\frac{f(1,2+h)-2f(1,2)+f(1,2-h)}{h^2}=2
$$

混合偏导用四个角点：

$$
\frac{
f(1+h,2+h)-f(1+h,2-h)-f(1-h,2+h)+f(1-h,2-h)
}{4h^2}
=3
$$

因此每一个 $h$ 都得到

| 步长 $h$ | $f_{xx}$ | $f_{xy}$ | $f_{yy}$ |
| --- | --- | --- | --- |
| $0.1$ | $2$ | $3$ | $2$ |
| $0.01$ | $2$ | $3$ | $2$ |
| $0.001$ | $2$ | $3$ | $2$ |

这是二次多项式的理想结果。对真实数值模型，二阶差分要除以 $h^2$，舍入误差会被放大得更快；先用解析 Hessian 或 Hessian-向量积核对，再选择合适的有限差分步长。

## 常见失效模式

- **在非临界点直接分类。** Hessian 描述弯曲，不会消除一阶斜率；必须先检查 $\nabla f(\boldsymbol a)=0$，才能使用极值二阶判据。
- **把半正定当成严格最小。** 零特征值方向的二阶项消失，函数可能是 $x^4$ 的平坦最小，也可能像 $x^3$ 一样没有极值。
- **忽略混合偏导的条件。** 没有足够连续性时，$f_{xy}$ 和 $f_{yx}$ 不一定能交换；数值近似中非对称还可能来自误差。
- **把特征值只当成函数值。** Hessian 的特征值是方向曲率，特征向量是参数空间中的方向，不是原函数的输出。
- **把局部二次模型当成全局形状。** 远离展开点后，Hessian 会改变；Newton 步需要重新计算并检查损失是否下降。
- **显式构造过大的 Hessian。** 参数有 $n$ 个时矩阵有 $n^2$ 个元素；许多算法只需要 $H\boldsymbol v$，不需要把所有元素存下来。

## 相关词条

- [梯度](../calculus/gradient/)：一阶变化和最陡方向。
- [全导数](../calculus/total-derivative/)：把一阶变化写成线性映射。
- [雅可比矩阵](../calculus/jacobian/)：Hessian 是梯度 Jacobian 的标量输出特例。
- [二次型](../linear-algebra/quadratic-forms/)：解释 $\boldsymbol h^{\mathsf T}H\boldsymbol h$ 的方向符号。
- [谱定理](../linear-algebra/spectral-theorem/)：为对称 Hessian 提供正交特征方向。
- [特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/)：解释主曲率大小和方向。
- [Taylor 展开](../calculus/taylor-series/)：把局部高阶近似系统化。
- [梯度下降](../training-nn/gradient-descent/)：只使用一阶梯度的参数更新。
- [二阶优化方法](../optimization-theory/second-order-methods/)：利用 Hessian 或其近似调整更新。
