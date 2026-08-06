---
title: "长度与距离：范数如何变成度量"
tags: ["why-models-learn"]
---

**距离**(distance)是给一对点配一个非负数的规则：两点重合时距离为零，交换两点不改变距离，绕路不会比直达更短。若这些点来自向量空间，范数把位移向量的长度变成点间距离：

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|
$$

这里要分清两个对象：$\|\mathbf{x}\|$量的是从原点指向$\mathbf{x}$的位移长度，$d(\mathbf{x}, \mathbf{y})$量的是从$\mathbf{x}$走到$\mathbf{y}$的距离。[范数](../linear-algebra/norms/) 篇已经规定了长度必须满足的三条公理，本篇把它们逐项改写成距离的四条公理，再比较不同范数给出的几何、极限与机器学习行为。

## 四条距离公理

在集合$X$上，函数$d:X\times X\to\mathbb{R}$称为**度量**(metric)，如果任意$\mathbf{x},\mathbf{y},\mathbf{z}\in X$都满足：

**非负性**：

$$
d(\mathbf{x}, \mathbf{y}) \geq 0
$$

**同一性**：

$$
d(\mathbf{x}, \mathbf{y}) = 0 \quad\Longleftrightarrow\quad \mathbf{x} = \mathbf{y}
$$

**对称性**：

$$
d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})
$$

**三角不等式**：

$$
d(\mathbf{x}, \mathbf{z}) \leq d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})
$$

前两条说明距离是长度，而且只有重合的两点距离为零；第三条说明从$\mathbf{x}$到$\mathbf{y}$与从$\mathbf{y}$到$\mathbf{z}$的先后不重要；最后一条说明经过中间点的路线不会短于直接路线。

如果$X$只是一组对象，四条公理就是全部要求。$X$可以是平面上的点、文本、图像，也可以是一个有限集合。若$X$还有向量加法和数乘，范数就提供了一种结构特别规整的度量。

## 范数为什么能生成距离

设$V$是向量空间，取任意范数，定义

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|
$$

四条公理逐条来自范数的三条公理。

非负性与同一性直接继承正定性：

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\| \geq 0
$$

并且

$$
d(\mathbf{x}, \mathbf{y}) = 0
\Longleftrightarrow
\mathbf{x} - \mathbf{y} = \mathbf{0}
\Longleftrightarrow
\mathbf{x} = \mathbf{y}
$$

对称性使用齐次性里$|{-1}|=1$这一点：

$$
d(\mathbf{x}, \mathbf{y})
= \|\mathbf{x} - \mathbf{y}\|
= \|-(\mathbf{y} - \mathbf{x})\|
= \|\mathbf{y} - \mathbf{x}\|
= d(\mathbf{y}, \mathbf{x})
$$

三角不等式把位移拆成两段：

$$
\begin{aligned}
d(\mathbf{x}, \mathbf{z})
&= \|\mathbf{x} - \mathbf{z}\| \\
&= \|(\mathbf{x} - \mathbf{y}) + (\mathbf{y} - \mathbf{z})\| \\
&\leq \|\mathbf{x} - \mathbf{y}\| + \|\mathbf{y} - \mathbf{z}\| \\
&= d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})
\end{aligned}
$$

所以，范数不只告诉我们「一个向量有多长」，还自动告诉我们「两个点相隔多远」。这个由范数产生的度量叫**范数诱导度量**。

## 同一对点，三种距离

取平面上的两点

$$
\mathbf{a} = (1, 2), \qquad \mathbf{b} = (4, 6)
$$

它们的位移向量是

$$
\mathbf{b} - \mathbf{a} = (3, 4)
$$

位移相同，选的范数不同，距离就不同：

| 距离 | 计算 | 数值 |
| --- | --- | --- |
| 曼哈顿距离 $d_1$ | $\|(3, 4)\|_1 = 3 + 4$ | $7$ |
| 欧几里得距离 $d_2$ | $\|(3, 4)\|_2 = \sqrt{3^2 + 4^2}$ | $5$ |
| 切比雪夫距离 $d_\infty$ | $\|(3, 4)\|_\infty = \max(3, 4)$ | $4$ |

这三个数字都正确，只是回答了三个不同的问题。曼哈顿距离把每个坐标方向的移动量相加，欧几里得距离把位移当作直线，切比雪夫距离只看移动量最大的那个坐标。

取另一个点$\mathbf{c}=(4,2)$，走路线$\mathbf{a}\to\mathbf{c}\to\mathbf{b}$。三种尺子的直接距离与绕路距离分别是：

$$
\begin{array}{c|ccc}
 & d_1 & d_2 & d_\infty \\
\hline
d(\mathbf{a}, \mathbf{b}) & 7 & 5 & 4 \\
d(\mathbf{a}, \mathbf{c}) + d(\mathbf{c}, \mathbf{b}) & 7 & 7 & 7
\end{array}
$$

欧几里得尺子下，绕路$7$大于直达$5$；曼哈顿尺子下，这条横竖路线恰好就是最短路线，所以等号成立；切比雪夫尺子下，允许两个坐标同时移动，直接路线只需$4$个单位的时间步长，横竖绕路反而更长。

## 距离球：换一把尺子，邻域就换形状

给定中心点$\mathbf{a}$和半径$r$，**距离球**(metric ball)定义为

$$
B_d(\mathbf{a}, r) = \{\mathbf{x}: d(\mathbf{x}, \mathbf{a}) \leq r\}
$$

这里的「球」不是固定的圆。中心取原点、半径取$2$时，三种距离给出：

$$
\begin{aligned}
B_{d_1}(\mathbf{0}, 2) &= \{(x,y): |x| + |y| \leq 2\} \\
B_{d_2}(\mathbf{0}, 2) &= \{(x,y): x^2 + y^2 \leq 4\} \\
B_{d_\infty}(\mathbf{0}, 2) &= \{(x,y): \max(|x|,|y|) \leq 2\}
\end{aligned}
$$

第一种是菱形，第二种是圆盘，第三种是正方形。它们都是「离中心不超过$2$」的点，只是「不超过」由不同的范数解释。对应的边界若改成等号，就是距离恰好等于$2$的点，不再包含内部。

![左：点 a=(1, 2) 到点 b=(4, 6) 的位移 b−a=(3, 4)，三种距离为 7、5、4；右：同一个半径 r=2 的距离球在曼哈顿、欧几里得、切比雪夫三种尺子下分别是菱形、圆盘、正方形](/assets/linear-algebra/svg/lengths-and-distances.1.svg)

距离球把「附近」变成了一个数学对象。最近邻搜索是在比较多个中心的距离；聚类是在为每个点寻找合适的中心；迭代算法的停止条件也常写成「新旧参数的距离小于某个阈值」。因此，换距离并不是只换一个公式，而是在换邻域的边界。

## 平移、缩放与旋转

范数诱导的距离有两个直接的变换规律。给两点同时加上同一个位移$\mathbf{t}$，差向量不变：

$$
d(\mathbf{x}+\mathbf{t}, \mathbf{y}+\mathbf{t})
= \|(\mathbf{x}+\mathbf{t}) - (\mathbf{y}+\mathbf{t})\|
= \|\mathbf{x} - \mathbf{y}\|
= d(\mathbf{x}, \mathbf{y})
$$

这叫**平移不变性**：整张坐标纸一起移动，点对之间的距离不变。两点同时乘以标量$c$时，齐次性给出

$$
d(c\mathbf{x}, c\mathbf{y}) = \|c(\mathbf{x} - \mathbf{y})\| = |c|d(\mathbf{x}, \mathbf{y})
$$

整张图放大两倍，所有距离也放大两倍；反向只改变方向，不改变距离。

旋转则要看范数。对欧几里得范数，如果矩阵$R$满足$R^\mathsf{T}R=I$，就有

$$
\|R\mathbf{x} - R\mathbf{y}\|_2
= \|R(\mathbf{x} - \mathbf{y})\|_2
= \|\mathbf{x} - \mathbf{y}\|_2
$$

因为

$$
\|R\mathbf{v}\|_2^2 = (R\mathbf{v})^\mathsf{T}(R\mathbf{v}) = \mathbf{v}^\mathsf{T}R^\mathsf{T}R\mathbf{v} = \mathbf{v}^\mathsf{T}\mathbf{v} = \|\mathbf{v}\|_2^2
$$

$\ell_1$和$\ell_\infty$通常没有这个旋转不变性。取$\mathbf{u}=(1,0)$，逆时针旋转$45^\circ$后变成

$$
R\mathbf{u} = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)
$$

旋转前后三种长度为：

$$
\begin{array}{c|ccc}
 & \|\cdot\|_1 & \|\cdot\|_2 & \|\cdot\|_\infty \\
\hline
\mathbf{u} & 1 & 1 & 1 \\
R\mathbf{u} & \sqrt{2}\approx1.414 & 1 & 1/\sqrt{2}\approx0.707
\end{array}
$$

只有欧几里得长度保持不变。若任务的坐标轴没有特殊含义，欧几里得距离通常更符合「旋转不应改变相似度」的直觉；若坐标轴代表道路方向、棋盘方向或预算维度，曼哈顿和切比雪夫距离也可能更合适。

## 距离与极限

有了距离，就能不依赖坐标分量逐个趋近来定义极限：

$$
\mathbf{x}_k \to \mathbf{x}
\quad\Longleftrightarrow\quad
d(\mathbf{x}_k, \mathbf{x}) \to 0
$$

例如$\mathbf{x}_k=(1/k,2/k)$趋向零向量。三种距离都可以直接写出：

$$
\|\mathbf{x}_k\|_1 = \frac{3}{k}, \qquad
\|\mathbf{x}_k\|_2 = \frac{\sqrt{5}}{k}, \qquad
\|\mathbf{x}_k\|_\infty = \frac{2}{k}
$$

数值表为：

| $k$ | $d_1(\mathbf{x}_k,\mathbf{0})$ | $d_2(\mathbf{x}_k,\mathbf{0})$ | $d_\infty(\mathbf{x}_k,\mathbf{0})$ |
| --- | --- | --- | --- |
| $1$ | $3$ | $2.236$ | $2$ |
| $2$ | $1.5$ | $1.118$ | $1$ |
| $10$ | $0.3$ | $0.224$ | $0.2$ |
| $100$ | $0.03$ | $0.022$ | $0.02$ |

三列的数值速度不同，但都趋向$0$。有限维空间中任意两个范数等价，所以换用$\ell_1$、$\ell_2$或$\ell_\infty$不会改变「是否收敛」这个结论，只会改变距离数值与阈值的尺度。[范数](../linear-algebra/norms/) 篇给出了有限维范数等价的结论；这里把它改写成距离，就是同一组点列拥有相同的收敛点。

这也解释了优化里的停止条件：如果参数序列是$\mathbf{w}_0,\mathbf{w}_1,\dots$，可以检查

$$
d(\mathbf{w}_{k+1}, \mathbf{w}_k) = \|\mathbf{w}_{k+1} - \mathbf{w}_k\| < \varepsilon
$$

它表示相邻两次更新已经很近，不等于已经找到了全局最优点；停止阈值和范数选择仍要结合损失与任务判断。

## 距离不一定来自范数

「范数可以生成距离」不等于「每个距离都来自范数」。

先看实数轴上的离散度量：

$$
d_{\mathrm{disc}}(x,y) =
\begin{cases}
0, & x=y \\
1, & x\neq y
\end{cases}
$$

它满足四条距离公理。即使$x$和$y$相隔很远，只要不相等，距离就都是$1$。但它不可能由某个范数诱导：若$d(0,1)=\|1\|=1$，齐次性会要求$d(0,2)=\|2\|=2$；离散度量却给出$d(0,2)=1$。它是一个度量，却不是范数距离。

还有一个机器学习里很容易混淆的对象：**平方欧几里得距离**

$$
\delta(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2^2
$$

不满足三角不等式。在实数轴上取$0,1,2$：

$$
\delta(0,2) = 4
\;>\;
\delta(0,1) + \delta(1,2) = 1 + 1 = 2
$$

所以它不是度量。均方误差仍然有用，是因为在比较同一个目标时，平方是非负数上的单调变换，最小点与欧几里得距离相同；「适合优化」和「满足度量公理」是两件事。

## 神经网络里的距离选择

**最近邻看的是相对次序。** 给定查询向量$\mathbf{q}$和候选集合$\{\mathbf{x}_i\}$，最近邻是

$$
\operatorname*{arg\,min}_i d(\mathbf{q}, \mathbf{x}_i)
$$

如果把$d_2$换成$d_1$，候选点的排序可能改变。对$(1,2)$与$(4,6)$，三种距离已经给出$7$、$5$、$4$的不同尺度；在高维数据上，很多「谁更近」的判断都会依赖这把尺子。

**特征单位会进入距离。** 两个二维点$(1,0)$和$(0,1)$的欧几里得距离是$\sqrt{2}\approx1.414$。如果第二个特征改用一百倍的单位，点变成$(0,100)$，距离变成

$$
\sqrt{1^2 + 100^2} = \sqrt{10001} \approx 100.005
$$

这不是数据「真的更远」了，而是坐标单位变了。用距离做最近邻、聚类或相似度前，常要先统一单位或标准化特征，否则数值范围大的坐标会主导结果。

**损失里的平方要单独看。** [损失函数](../training-nn/loss-functions/) 常用平方误差，[范数](../linear-algebra/norms/) 篇已经说明它继承了欧几里得长度的最小点，却没有继承范数的齐次性。[梯度下降](../training-nn/gradient-descent/) 里用参数距离监控更新时，可以用$\ell_2$范数；优化目标写成平方$\ell_2$则是另一个选择。

## 失效模式与常见误区

**把点当成位移。** $\|\mathbf{x}\|$是点$\mathbf{x}$相对原点的长度；两点距离要写成$\|\mathbf{x}-\mathbf{y}\|$。如果把原点换了，单个坐标点的长度会变，点对之间的范数距离不会变。

**把平方距离叫作距离。** $\|\mathbf{x}-\mathbf{y}\|_2^2$在回归里常用，但$0,1,2$的反例已经说明它违反三角不等式。写下$ d $之前，先确认是不是开了平方。

**忘记说明使用哪把尺子。** 「两点距离是$5$」只有在范数已指定时才有意义；同一位移$(3,4)$在$\ell_1$、$\ell_2$、$\ell_\infty$下分别是$7$、$5$、$4$。

**把距离球误认成圆。** 只有欧几里得距离的二维球是圆盘。曼哈顿距离的球是菱形，切比雪夫距离的球是正方形；代码里的阈值条件必须与实际距离公式配套。

**忽略特征尺度。** 坐标换单位会改变$\ell_1$、$\ell_2$和$\ell_\infty$的数值。没有归一化就直接做最近邻或聚类，结果可能主要反映单位选择，而不是数据结构。

## 相关词条

- [范数](../linear-algebra/norms/)：长度的三条公理、$p$ 范数与单位球
- [内积](../linear-algebra/inner-products/)：欧几里得范数的内积来源与 Cauchy–Schwarz 不等式
- [角度与正交](../linear-algebra/angles-and-orthogonality/)：内积距离的角度、垂直与投影几何
- [正交投影](../linear-algebra/orthogonal-projections/)：点到子空间的最近距离
- [向量空间](../linear-algebra/vector-spaces/)：范数与距离定义所依赖的线性结构
- [损失函数](../training-nn/loss-functions/)：平方误差与距离尺度的选择
- [梯度下降](../training-nn/gradient-descent/)：用参数距离表述更新与停止条件
