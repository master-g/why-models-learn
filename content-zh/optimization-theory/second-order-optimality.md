---
title: "二阶最优性条件：用曲率区分谷底、峰顶和鞍点"
tags: ["why-models-learn"]
---

一阶条件只能告诉我们「一阶下降方向消失了」，二阶最优性条件继续检查当前位置沿不同方向是向上弯、向下弯还是平的。对无约束可微函数，正定 Hessian 足以保证严格局部最小，不定 Hessian 则给出鞍点；半正定而奇异时，二阶信息本身没有结论，必须查看更高阶项或直接比较函数值。等式约束把 Hessian 限制到约束的切空间，不等式约束还要限制到临界锥。本篇把这些必要条件和充分条件分开，再连接 Newton 步、信赖域和神经网络中的平坦方向。

## Taylor 展开把局部问题变成二次型

设 $f:\mathbb R^d\to\mathbb R$ 在 $\boldsymbol x^\star$ 附近二阶连续可微。对小位移 $\boldsymbol d$：

$$
\begin{aligned}
f(\boldsymbol x^\star+\boldsymbol d)
&=f(\boldsymbol x^\star)
+\nabla f(\boldsymbol x^\star)^\mathsf T\boldsymbol d\\
&\quad+\frac12\boldsymbol d^\mathsf T
H(\boldsymbol x^\star)\boldsymbol d
+o(\|\boldsymbol d\|_2^2)
\end{aligned}
$$

其中

$$
H(\boldsymbol x^\star)=\nabla^2f(\boldsymbol x^\star)
$$

是 Hessian。若 $\boldsymbol x^\star$ 已满足一阶条件

$$
\nabla f(\boldsymbol x^\star)=\boldsymbol0
$$

那么最先决定函数值变化的是

$$
q(\boldsymbol d)
=\boldsymbol d^\mathsf T
H(\boldsymbol x^\star)\boldsymbol d
$$

它是沿方向 $\boldsymbol d$ 的二次曲率。$q(\boldsymbol d)>0$ 表示二阶项把函数值推高，$q(\boldsymbol d)<0$ 表示推低，$q(\boldsymbol d)=0$ 表示该方向的二阶项没有提供信息。

### 三个二元数字例子

第一个函数是一个碗：

$$
f_+ (x,y)=x^2+2y^2
$$

在原点：

$$
\nabla f_+(0,0)=
\begin{bmatrix}0\\0\end{bmatrix},
\qquad
H_+=
\begin{bmatrix}
2&0\\
0&4
\end{bmatrix}
$$

对非零方向 $\boldsymbol d=(d_1,d_2)$：

$$
q_+(\boldsymbol d)=2d_1^2+4d_2^2>0
$$

所以从原点向任意方向移动都会增加函数值，原点是严格局部最小点。

第二个函数是一张鞍面：

$$
f_{\pm}(x,y)=x^2-y^2
$$

其 Hessian 为

$$
H_{\pm}=
\begin{bmatrix}
2&0\\
0&-2
\end{bmatrix}
$$

沿 $\boldsymbol d=(1,0)$ 有 $q_{\pm}=2$，沿 $\boldsymbol d=(0,1)$ 有 $q_{\pm}=-2$。同一个驻点既有上升方向又有下降方向，不能是局部极值。

第三个函数故意让二阶项失效：

$$
f_0(x,y)=x^4+y^2
$$

在原点 Hessian 是

$$
H_0=
\begin{bmatrix}
0&0\\
0&2
\end{bmatrix}
$$

沿 $x$ 轴的二次曲率为零，但函数值是 $x^4>0$；沿 $y$ 轴则由 $2y^2$ 提供正曲率。因此原点仍是严格局部最小点，结论来自四阶项，而不是来自 Hessian 的零特征值。

## 正定性决定无约束二阶分类

对称矩阵 $H$ 的四种性质对应不同的方向信息：

$$
\begin{aligned}
H\succ0
&\Longleftrightarrow
\boldsymbol d^\mathsf TH\boldsymbol d>0
\quad\text{for all }\boldsymbol d\neq\boldsymbol0\\
H\succeq0
&\Longleftrightarrow
\boldsymbol d^\mathsf TH\boldsymbol d\geq0
\quad\text{for all }\boldsymbol d\\
H\prec0
&\Longleftrightarrow
\boldsymbol d^\mathsf TH\boldsymbol d<0
\quad\text{for all }\boldsymbol d\neq\boldsymbol0
\end{aligned}
$$

如果有些方向给正值、有些方向给负值，则称 $H$ 不定。对于内部驻点，二阶判据如下：

| Hessian 条件 | 对应结论 | 条件强度 |
| --- | --- | --- |
| $H\succ0$ | 严格局部最小点 | 充分条件 |
| $H\prec0$ | 严格局部最大点 | 充分条件 |
| $H$ 不定 | 鞍点 | 充分排除极值 |
| $H\succeq0$ | 只能排除二阶下降方向 | 通常还不充分 |
| $H\preceq0$ | 只能排除二阶上升方向 | 通常还不充分 |

如果 $\boldsymbol x^\star$ 是二阶连续可微函数的内部局部最小点，那么 Hessian 必须半正定。证明很直接：沿任意 $\boldsymbol d$ 的一元函数

$$
\phi(t)=f(\boldsymbol x^\star+t\boldsymbol d)
$$

在 $t=0$ 处有局部最小，所以 $\phi''(0)\geq0$，而

$$
\phi''(0)
=\boldsymbol d^\mathsf TH(\boldsymbol x^\star)\boldsymbol d
$$

对每个方向都成立。注意这是必要条件；从「半正定」回到「局部最小」还缺少对零曲率方向的检查。

### 一维版本

在一维，Hessian 退化为二阶导数：

$$
f'(x^\star)=0
$$

之后：

$$
\begin{cases}
f''(x^\star)>0&\Rightarrow\text{严格局部最小}\\
f''(x^\star)<0&\Rightarrow\text{严格局部最大}\\
f''(x^\star)=0&\Rightarrow\text{二阶测试无结论}
\end{cases}
$$

$x^4$ 和 $x^3$ 都在 $0$ 处满足前两项导数为零、二阶导数为零，但前者是严格局部最小，后者不是极值。相同的二阶数据可以对应完全不同的函数形状。

## 特征值把曲率拆成正交方向

实对称 Hessian 可以正交对角化。若

$$
H=Q\Lambda Q^\mathsf T,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_d)
$$

且单位方向写成

$$
\boldsymbol u=\sum_{i=1}^dc_i\boldsymbol q_i,
\qquad
\sum_{i=1}^dc_i^2=1
$$

那么

$$
\boldsymbol u^\mathsf TH\boldsymbol u
=\sum_{i=1}^d\lambda_i c_i^2
$$

方向曲率是特征值的加权平均，权重是方向在各特征向量上的平方投影。因此：

- 所有特征值为正等价于 $H\succ0$；
- 所有特征值非负等价于 $H\succeq0$；
- 同时有正、负特征值等价于 $H$ 不定；
- 最小特征值是最容易向下弯的方向的曲率。

取

$$
H=
\begin{bmatrix}
2&0\\
0&8
\end{bmatrix},
\qquad
\boldsymbol u=\frac1{\sqrt2}
\begin{bmatrix}1\\-1\end{bmatrix}
$$

则

$$
\boldsymbol u^\mathsf TH\boldsymbol u
=\frac12(2+8)=5
$$

虽然方向同时含有两个坐标分量，曲率仍然只是 $2$ 和 $8$ 的加权平均。谱定理和特征分解提供了把交叉项换成这些独立方向的工具。

## 二阶退化时沿零空间继续看

设 $H\succeq0$ 但存在非零 $\boldsymbol z$ 满足

$$
H\boldsymbol z=\boldsymbol0
$$

那么沿 $\boldsymbol z$ 的二次项为零。接下来至少要做两件事：

1. 把函数限制到 $\boldsymbol x^\star+t\boldsymbol z$，检查第一个不为零的高阶项；
2. 同时检查零空间方向与非零曲率方向的组合，不能只看坐标轴。

### 正的四阶项

对

$$
f(x,y)=x^4+y^2
$$

原点 Hessian 的零空间是 $x$ 轴。限制到零空间：

$$
f(t,0)=t^4>0
\qquad
t\neq0
$$

再加上 $y$ 方向的 $y^2\geq0$，得到原点是严格局部最小点。

### 负的四阶项

把函数换为

$$
\widetilde f(x,y)=-x^4+y^2
$$

Hessian 仍然是

$$
\begin{bmatrix}
0&0\\
0&2
\end{bmatrix}
$$

但沿 $x$ 轴：

$$
\widetilde f(t,0)=-t^4<0
$$

沿 $y$ 轴则为 $y^2>0$。原点因此是鞍点。仅仅记录「Hessian 半正定」会把这两个函数混为一谈。

如果函数在某个方向上所有阶都保持不变，就可能有平坦的最小值集合。例如

$$
f(x,y)=y^2
$$

在整条 $x$ 轴上达到同一个全局最小值 0。此时原点是最小点，但不是严格最小点，Hessian 的零特征值对应一条等价解方向。

## 约束问题要把 Hessian 限制到切空间

考虑等式约束

$$
\min_{\boldsymbol x}f(\boldsymbol x)
\qquad
\text{subject to}\qquad
\boldsymbol h(\boldsymbol x)=\boldsymbol0
$$

在满足一阶条件的点 $(\boldsymbol x^\star,\boldsymbol\nu^\star)$，拉格朗日函数是

$$
\mathcal L(\boldsymbol x,\boldsymbol\nu)
=f(\boldsymbol x)
+\boldsymbol\nu^\mathsf T\boldsymbol h(\boldsymbol x)
$$

它关于原变量的 Hessian：

$$
H_{\mathcal L}
=\nabla^2f(\boldsymbol x^\star)
+\sum_{j=1}^m\nu_j^\star
\nabla^2h_j(\boldsymbol x^\star)
$$

约束曲面的切空间由

$$
\mathcal T(\boldsymbol x^\star)
=\left\{\boldsymbol d:
J_{\boldsymbol h}(\boldsymbol x^\star)\boldsymbol d
=\boldsymbol0\right\}
$$

给出。沿这些方向移动才是一阶上仍留在约束曲面的移动。因此二阶必要条件不是要求 $H_{\mathcal L}$ 在整个 $\mathbb R^d$ 上半正定，而是

$$
\boldsymbol d^\mathsf TH_{\mathcal L}\boldsymbol d\geq0
\qquad
\text{for all }\boldsymbol d\in\mathcal T(\boldsymbol x^\star)
$$

二阶充分条件则要求对每个非零切向方向严格为正：

$$
\boldsymbol d^\mathsf TH_{\mathcal L}\boldsymbol d>0
\qquad
\text{for all }\boldsymbol d\in\mathcal T(\boldsymbol x^\star),
\ \boldsymbol d\neq\boldsymbol0
$$

在相应的正则性和光滑性假设下，这能保证严格局部最小。法向方向不需要被目标 Hessian 判为正，因为那里的移动本来就违反等式约束。

### 投影例子的二阶核对

回看

$$
\min_{x_1,x_2}x_1^2+x_2^2
\qquad
\text{subject to}\qquad
x_1+x_2-1=0
$$

约束是仿射的，所以它的二阶导数为零，$H_{\mathcal L}=2I$。切空间方向满足

$$
d_1+d_2=0
$$

可以写成 $\boldsymbol d=(t,-t)$。于是

$$
\boldsymbol d^\mathsf TH_{\mathcal L}\boldsymbol d
=2t^2+2t^2
=4t^2>0
\qquad
t\neq0
$$

这说明投影点 $(1/2,1/2)$ 沿直线的每个非零方向都严格上升，确实是约束曲面上的严格局部最小；不需要要求 $2I$ 在约束法向和切向之外再满足什么额外条件。

### 不等式约束的临界锥

对 KKT 点，令活动不等式集合为

$$
\mathcal I(\boldsymbol x^\star)
=\left\{i:g_i(\boldsymbol x^\star)=0\right\}
$$

线性化可行方向满足

$$
\mathcal T_{\mathrm{lin}}
=\left\{\boldsymbol d:
\begin{aligned}
&\nabla h_j(\boldsymbol x^\star)^\mathsf T\boldsymbol d=0\\
&\nabla g_i(\boldsymbol x^\star)^\mathsf T\boldsymbol d\leq0,
\quad i\in\mathcal I(\boldsymbol x^\star)
\end{aligned}
\right\}
$$

其中只列活动约束，因为松约束在足够小的移动下仍有余量。临界锥进一步保留一阶目标变化为零的方向：

$$
\mathcal C(\boldsymbol x^\star)
=\left\{\boldsymbol d\in\mathcal T_{\mathrm{lin}}:
\nabla f(\boldsymbol x^\star)^\mathsf T\boldsymbol d=0\right\}
$$

在 KKT 和正则性假设下，二阶必要条件检查

$$
\boldsymbol d^\mathsf TH_{\mathcal L}\boldsymbol d\geq0
\qquad
\text{for all }\boldsymbol d\in\mathcal C(\boldsymbol x^\star)
$$

如果临界方向上的二次型严格为正，并满足相应的二阶充分条件，才能排除这些一阶上看不出变化的约束移动。对于边界点，临界锥可能只有零向量，此时二阶条件可能是空的，最优性主要已经由一阶约束方向决定。

## Newton 步依赖局部曲率

在当前位置 $\boldsymbol w$，把损失写成局部二次模型：

$$
L(\boldsymbol w+\boldsymbol\delta)
\approx
L(\boldsymbol w)
+\nabla L(\boldsymbol w)^\mathsf T\boldsymbol\delta
+\frac12\boldsymbol\delta^\mathsf TH_L(\boldsymbol w)\boldsymbol\delta
$$

对步长 $\boldsymbol\delta$ 求导并令其为零：

$$
\nabla L(\boldsymbol w)
+H_L(\boldsymbol w)\boldsymbol\delta
=\boldsymbol0
$$

如果 Hessian 可逆，Newton 步为

$$
\boldsymbol\delta_{\mathrm N}
=-H_L(\boldsymbol w)^{-1}\nabla L(\boldsymbol w)
$$

对严格凸二次函数

$$
L(\boldsymbol w)
=\frac12\boldsymbol w^\mathsf TQ\boldsymbol w
-\boldsymbol b^\mathsf T\boldsymbol w
$$

其中 $Q\succ0$，Hessian 恒为 $Q$，Newton 步一步就到达

$$
\boldsymbol w^\star=Q^{-1}\boldsymbol b
$$

例如

$$
Q=
\begin{bmatrix}
2&0\\
0&4
\end{bmatrix},
\qquad
\boldsymbol b=
\begin{bmatrix}2\\8\end{bmatrix}
$$

则

$$
\boldsymbol w^\star
=Q^{-1}\boldsymbol b
=\begin{bmatrix}1\\2\end{bmatrix}
$$

从原点出发，梯度为 $-\boldsymbol b$，Newton 步正好是 $(1,2)$。但如果 Hessian 不定，$-H^{-1}\nabla L$ 不一定是下降方向；如果 Hessian 奇异，线性方程还可能没有唯一解。

信赖域方法把二次模型限制在

$$
\|\boldsymbol\delta\|_2\leq\Delta
$$

的区域内，再决定这个局部模型值得信任多远。即使模型有负曲率，最优步也可能落在信赖域边界；这比直接把一个不定 Hessian 当作谷底曲率更稳妥。后续的二阶方法会继续处理线性方程、阻尼和有限精度。

## 神经网络中曲率不能直接当作泛化证明

对网络损失 $L(\boldsymbol\theta)$，Hessian 的特征值可以把参数扰动分成几类：

- 大正特征值方向，参数稍微移动就让训练损失快速上升；
- 小正特征值方向，损失变化平缓但仍是向上弯；
- 接近零的方向，可能来自参数冗余、对称性或真正的平坦区域；
- 负特征值方向，当前位置仍有二阶下降方向，不是严格局部最小。

这些都是训练目标在当前参数附近的局部事实。一个 Hessian 半正定的训练点不自动是全局最小，也不自动在验证集或总体分布上最好；一个负特征值则说明局部仍有下降方向，但不告诉我们哪一条有限步长轨迹能稳定到达更低点。

完整 Hessian 的元素数量随参数维度平方增长，实际计算常用 Hessian-向量积：

$$
H(\boldsymbol\theta)\boldsymbol v
=\left.
\frac{\mathrm d}{\mathrm dt}
\nabla L(\boldsymbol\theta+t\boldsymbol v)
\right|_{t=0}
$$

它能在不显式存储矩阵的情况下估计最大或最小曲率方向。有限批次还会让这个曲率带有抽样噪声，不能把一次小批次 Hessian 的符号直接当作总体风险的定理。

![二阶条件检查每个方向的曲率](/assets/optimization-theory/svg/second-order-optimality.1.svg)

## 失效模式

**把半正定当成严格局部最小。** 半正定只排除二阶负方向；零空间中的四阶负项仍可制造鞍点。

**只检查 Hessian 的一个坐标元素。** 曲率由 $\boldsymbol d^\mathsf TH\boldsymbol d$ 决定，交叉项和特征方向不能被逐元素符号替代。

**忘记先验证一阶条件。** 梯度非零时，一阶项会在足够小的某个方向主导二阶项；Hessian 正定也不能把当前点变成局部最小。

**把不定 Hessian 当成数值故障。** 负特征值可能是真实的鞍点或下弯方向；先区分数学结构和数值误差，再决定是否阻尼。

**把约束法向也拿来做二阶检验。** 等式约束只允许切向移动；应检查拉格朗日 Hessian 在切空间或临界锥上的二次型。

**把 KKT 条件和二阶充分条件混在一起。** KKT 主要描述一阶平衡，二阶条件还要检查一阶看不见的临界方向。

**把零特征值都解释成平坦最小谷。** 零曲率方向可能由高阶正项、负项或参数对称性产生；必须沿该方向继续检查。

**把 Newton 步当成总是下降。** 只有在合适的正定局部模型、步长控制和正则性条件下，Newton 方向才有直接的下降解释。

**把训练 Hessian 当成泛化曲率。** 参数空间局部二阶信息不等于验证集或总体风险的二阶信息。

**用有限差分的微小数值当作符号定理。** Hessian 特征值接近零时，步长、舍入和批次噪声都可能改变观察到的正负；需要报告容差和估计方式。

## 相关词条

- [一阶最优性条件](../optimization-theory/first-order-optimality/)：先检查梯度、可行方向、拉格朗日乘子和 KKT。
- [局部与全局最小值](../optimization-theory/local-and-global-minima/)：区分局部曲率结论和全局目标比较。
- [Hessian 矩阵](../calculus/hessian/)：从梯度 Jacobian、方向曲率和特征方向建立二阶工具。
- [二次型](../linear-algebra/quadratic-forms/)：理解 Hessian 二次型的正定、不定和等值面。
- [谱定理](../linear-algebra/spectral-theorem/)：把对称 Hessian 正交对角化为特征方向。
- [约束优化](../optimization-theory/constrained-optimization/)：继续处理约束下的二阶方法和可行域。
- [梯度下降理论](../optimization-theory/gradient-descent-theory/)：分析一阶更新、光滑性和收敛。
- [二阶方法](../optimization-theory/second-order-methods/)：展开 Newton、拟 Newton、阻尼和信赖域算法。
- [曲率与条件数](../optimization-theory/curvature-and-conditioning/)：连接特征值跨度、病态和优化速度。
- [损失景观](../optimization-theory/loss-landscapes/)：从高维参数空间讨论鞍点、平坦方向和局部几何。
