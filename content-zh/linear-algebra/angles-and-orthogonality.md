---
title: "角度与正交:内积如何读出方向"
tags: ["why-models-learn"]
---

**角度**(angle)是两个非零向量方向差异的数值描述，内积把它写成

$$
\cos\theta
=
\frac{\langle \mathbf{x}, \mathbf{y}\rangle}
{\|\mathbf{x}\|\,\|\mathbf{y}\|}
$$

其中$\theta$是$\mathbf{x}$和$\mathbf{y}$的夹角。**正交**(orthogonality)则是内积为零：$\mathbf{x}\perp\mathbf{y}\Longleftrightarrow\langle\mathbf{x},\mathbf{y}\rangle=0$。前一篇[内积](../linear-algebra/inner-products/)讲了这套运算为什么合法，本篇把它翻译成方向、垂直、投影和最近点；[长度与距离](../linear-algebra/lengths-and-distances/)里的范数负责量大小，这里再把大小从内积中除掉，只留下方向关系。

## 先把内积除以长度

对两个非零向量，定义

$$
c(\mathbf{x},\mathbf{y})
=
\frac{\langle\mathbf{x},\mathbf{y}\rangle}
{\|\mathbf{x}\|\,\|\mathbf{y}\|}
$$

Cauchy–Schwarz 不等式给出

$$
\left|\langle\mathbf{x},\mathbf{y}\rangle\right|
\leq
\|\mathbf{x}\|\,\|\mathbf{y}\|
$$

所以

$$
-1\leq c(\mathbf{x},\mathbf{y})\leq1
$$

这个比值正好落在余弦函数的值域里，于是可以定义

$$
\theta=\arccos c(\mathbf{x},\mathbf{y}),\qquad 0\leq\theta\leq\pi
$$

分子是两个向量的内积，分母把两个向量各自的长度除掉。若把$\mathbf{x}$换成$5\mathbf{x}$，分子乘$5$，$\mathbf{x}$的长度也乘$5$，比值不变；向量变长了，方向没有变，夹角也不应该变。

内积的正负先给出粗略的方向判断：

| 情形 | $\langle\mathbf{x},\mathbf{y}\rangle$ | 夹角 |
| --- | --- | --- |
| 同向分量占优势 | $>0$ | $0\leq\theta<90^\circ$，锐角 |
| 没有共同方向分量 | $=0$ | $\theta=90^\circ$，正交 |
| 反向分量占优势 | $<0$ | $90^\circ<\theta\leq180^\circ$，钝角 |

这里的「共同方向分量」不是凭图形猜出来的，而是由内积算出来的数。两个向量都非零时，Cauchy–Schwarz 还说明$\theta=0^\circ$当且仅当它们同向，$\theta=180^\circ$当且仅当它们反向。

## 一个数字例子：同样的内积，不同的长度

取

$$
\mathbf{x}=(3,4),\qquad \mathbf{y}=(1,2)
$$

内积和长度分别是

$$
\langle\mathbf{x},\mathbf{y}\rangle
=3\times1+4\times2=11,\qquad
\|\mathbf{x}\|=5,\qquad
\|\mathbf{y}\|=\sqrt{5}
$$

因此

$$
\cos\theta
=\frac{11}{5\sqrt{5}}
\approx0.983870,\qquad
\theta\approx10.305^\circ
$$

这两个向量大致朝向同一个方向，但它们的长度并不相同。若只看内积$11$，会把「方向相近」和「向量很长」混在一起；除以长度之后，才得到只描述方向的余弦。

再看三组二维向量：

| $\mathbf{x}$ | $\mathbf{y}$ | $\langle\mathbf{x},\mathbf{y}\rangle$ | 夹角 |
| --- | --- | --- | --- |
| $(1,0)$ | $(1,1)$ | $1$ | $45^\circ$ |
| $(1,0)$ | $(0,1)$ | $0$ | $90^\circ$ |
| $(1,0)$ | $(-1,1)$ | $-1$ | $135^\circ$ |

第二行是最熟悉的坐标轴垂直，但正交不要求向量贴着坐标轴。取

$$
\mathbf{u}=(1,2),\qquad \mathbf{v}=(-2,1)
$$

有

$$
\langle\mathbf{u},\mathbf{v}\rangle
=1\times(-2)+2\times1=0
$$

所以$\mathbf{u}$和$\mathbf{v}$正交；它们的方向分别是斜率$2$和$-\frac12$，乘积为$-1$，正是平面中两条非竖直直线垂直的斜率判据。

## 正交不只是“看起来垂直”

在$\mathbb{R}^n$的点积下，正交的定义是

$$
\mathbf{x}\perp\mathbf{y}
\quad\Longleftrightarrow\quad
\langle\mathbf{x},\mathbf{y}\rangle=0
$$

它是一条代数判据，坐标系怎么画并不改变它的含义。对上面的$\mathbf{u}$和$\mathbf{v}$，如果把$\mathbf{u}$乘以任意标量$a$，仍有

$$
\langle a\mathbf{u},\mathbf{v}\rangle
=a\langle\mathbf{u},\mathbf{v}\rangle=0
$$

于是与$\mathbf{u}$正交的向量并不是一个孤立的箭头，而是一整个子空间。对

$$
U=\operatorname{span}\{(1,2)\}
$$

它的正交补是

$$
U^\perp
=\{\mathbf{z}\in\mathbb{R}^2:
\langle\mathbf{z},(1,2)\rangle=0\}
=\{(z_1,z_2):z_1+2z_2=0\}
=\operatorname{span}\{(-2,1)\}
$$

因此，正交补$U^\perp$收集的是「与$U$中每个向量都正交」的全部向量。因为$U$只有一个方向，二维平面里的正交补也只剩一个独立方向；更高维时，正交补可以包含多个方向。

内积还能直接推出勾股关系。展开平方：

$$
\|\mathbf{x}+\mathbf{y}\|^2
=\langle\mathbf{x}+\mathbf{y},\mathbf{x}+\mathbf{y}\rangle
=\|\mathbf{x}\|^2
+2\langle\mathbf{x},\mathbf{y}\rangle
+\|\mathbf{y}\|^2
$$

当且仅当$\mathbf{x}\perp\mathbf{y}$时，中间项为零，于是

$$
\mathbf{x}\perp\mathbf{y}
\quad\Longrightarrow\quad
\|\mathbf{x}+\mathbf{y}\|^2
=\|\mathbf{x}\|^2+\|\mathbf{y}\|^2
$$

反过来，如果这条等式成立，展开式中的内积就必须为零。因此，勾股关系不是另加的一条几何规律，它就是「交叉项消失」的内积语言。

## 投影：把一个向量拆成平行和垂直两部分

给定非零方向向量$\mathbf{u}$，希望用一段沿$\mathbf{u}$的向量近似$\mathbf{x}$。所有候选点都写成$t\mathbf{u}$，所以要最小化

$$
\|\mathbf{x}-t\mathbf{u}\|^2
=\langle\mathbf{x}-t\mathbf{u},\mathbf{x}-t\mathbf{u}\rangle
=\|\mathbf{x}\|^2
-2t\langle\mathbf{x},\mathbf{u}\rangle
+t^2\|\mathbf{u}\|^2
$$

把它配方：

$$
\begin{aligned}
\|\mathbf{x}-t\mathbf{u}\|^2
&=\|\mathbf{x}\|^2
-\frac{\langle\mathbf{x},\mathbf{u}\rangle^2}{\|\mathbf{u}\|^2}
+\|\mathbf{u}\|^2
\left(
t-\frac{\langle\mathbf{x},\mathbf{u}\rangle}{\|\mathbf{u}\|^2}
\right)^2
\end{aligned}
$$

最后一项总是非负，因此最小值出现在

$$
t^\ast
=\frac{\langle\mathbf{x},\mathbf{u}\rangle}{\|\mathbf{u}\|^2}
$$

沿$\mathbf{u}$方向的投影定义为

$$
\operatorname{proj}_{\mathbf{u}}\mathbf{x}
=t^\ast\mathbf{u}
=\frac{\langle\mathbf{x},\mathbf{u}\rangle}{\|\mathbf{u}\|^2}\mathbf{u}
$$

标量$t^\ast$回答「沿$\mathbf{u}$走多少」，投影向量回答「走完以后到达哪个向量」。分母是$\|\mathbf{u}\|^2$，不是$\|\mathbf{u}\|$；因为系数要把$\mathbf{u}$的实际长度也算进去。

取

$$
\mathbf{x}=(3,4),\qquad \mathbf{u}=(1,2)
$$

前面的内积是$\langle\mathbf{x},\mathbf{u}\rangle=11$，$\|\mathbf{u}\|^2=5$，所以

$$
t^\ast=\frac{11}{5},\qquad
\operatorname{proj}_{\mathbf{u}}\mathbf{x}
=\frac{11}{5}(1,2)
=\left(\frac{11}{5},\frac{22}{5}\right)
$$

剩下的部分叫正交残差：

$$
\mathbf{e}
=\mathbf{x}-\operatorname{proj}_{\mathbf{u}}\mathbf{x}
=\left(\frac45,-\frac25\right)
$$

它确实与$\mathbf{u}$正交：

$$
\langle\mathbf{e},\mathbf{u}\rangle
=\frac45\times1-\frac25\times2=0
$$

因此有分解

$$
\mathbf{x}
=\operatorname{proj}_{\mathbf{u}}\mathbf{x}+\mathbf{e},
\qquad
\operatorname{proj}_{\mathbf{u}}\mathbf{x}\parallel\mathbf{u},
\qquad
\mathbf{e}\perp\mathbf{u}
$$

数值核对如下：

| 对象 | 坐标或数值 |
| --- | --- |
| $\operatorname{proj}_{\mathbf{u}}\mathbf{x}$ | $(2.2,4.4)$ |
| $\mathbf{e}$ | $(0.8,-0.4)$ |
| $\langle\mathbf{e},\mathbf{u}\rangle$ | $0$ |
| $\|\mathbf{x}\|^2$ | $25$ |
| $\|\operatorname{proj}_{\mathbf{u}}\mathbf{x}\|^2+\|\mathbf{e}\|^2$ | $24.2+0.8=25$ |

图中的虚线段就是$\mathbf{e}$：它从投影点指向原向量的终点，并且与方向$\mathbf{u}$垂直。

![左：向量 x=(3, 4) 分解为沿 u=(1, 2) 的投影和正交残差；右：锐角、直角、钝角对应内积为正、零、负](/assets/linear-algebra/svg/angles-and-orthogonality.1.svg)

## 从一条直线到一个子空间

把一条直线换成子空间$U$，投影的判据不变：寻找$\mathbf{p}\in U$，使得$\mathbf{x}-\mathbf{p}$与$U$中的每个方向都正交。写成

$$
\mathbf{x}=\mathbf{p}+\mathbf{r},
\qquad
\mathbf{p}\in U,
\qquad
\mathbf{r}\in U^\perp
$$

如果这样的分解存在，它给出的$\mathbf{p}$就是$\mathbf{x}$在$U$上的正交投影，$\mathbf{r}$是无法由$U$表示的剩余部分。对一条由$\mathbf{u}$张成的直线，上一节的公式已经给出$\mathbf{p}$；对多个方向组成的子空间，需要先选一组方便计算的方向，下一篇[正交归一基](../linear-algebra/orthonormal-basis/)会处理如何把这些方向整理成彼此正交且长度为$1$的一组基。[正交投影](../linear-algebra/orthogonal-projections/)篇再把一般子空间的最近点问题完整展开。

正交条件带来唯一性。假设同一个$\mathbf{x}$有两种分解：

$$
\mathbf{x}=\mathbf{p}_1+\mathbf{r}_1=\mathbf{p}_2+\mathbf{r}_2,
\qquad
\mathbf{p}_1,\mathbf{p}_2\in U,
\qquad
\mathbf{r}_1,\mathbf{r}_2\in U^\perp
$$

相减得到

$$
\mathbf{p}_1-\mathbf{p}_2
=-(\mathbf{r}_1-\mathbf{r}_2)
$$

左边在$U$里，右边在$U^\perp$里，所以这个向量同时属于$U$和$U^\perp$。若它记为$\mathbf{q}$，则$\mathbf{q}\perp\mathbf{q}$，于是

$$
\|\mathbf{q}\|^2=\langle\mathbf{q},\mathbf{q}\rangle=0
\quad\Longrightarrow\quad
\mathbf{q}=\mathbf{0}
$$

所以$\mathbf{p}_1=\mathbf{p}_2$、$\mathbf{r}_1=\mathbf{r}_2$。最近点和正交残差不是许多种差不多的选择，而是由内积决定的唯一分解。

## 旋转不改变内积读出的角度

若矩阵$Q$满足

$$
Q^\mathsf{T}Q=I
$$

则对任意$\mathbf{x},\mathbf{y}$：

$$
\begin{aligned}
\langle Q\mathbf{x},Q\mathbf{y}\rangle
&=(Q\mathbf{x})^\mathsf{T}(Q\mathbf{y})\\
&=\mathbf{x}^\mathsf{T}Q^\mathsf{T}Q\mathbf{y}\\
&=\mathbf{x}^\mathsf{T}\mathbf{y}\\
&=\langle\mathbf{x},\mathbf{y}\rangle
\end{aligned}
$$

同样，$\|Q\mathbf{x}\|=\|\mathbf{x}\|$，所以余弦比值不变，正交关系也不变。二维旋转只是把所有箭头一起转动，箭头之间的角度没有被改写；反射也满足同一个代数条件。

这个结论有一个边界：任意可逆换基只保证坐标能互相转换，不保证标准点积的数值不变。非正交基下，两个坐标列的点积可能改变；如果要让几何量保持不变，需要同时转换内积的表示，或使用正交基。[换基](../linear-algebra/change-of-basis/)篇处理坐标如何变化，[正交矩阵与旋转](../linear-algebra/orthogonal-matrices-and-rotations/)篇再专门讨论满足$Q^\mathsf{T}Q=I$的矩阵。

## 神经网络里的方向关系

**余弦相似度去掉长度因素。** 文本嵌入或其他表示向量的相似度常写成

$$
\operatorname{cos\_sim}(\mathbf{x},\mathbf{y})
=
\frac{\mathbf{x}^{\mathsf{T}}\mathbf{y}}
{\|\mathbf{x}\|_2\|\mathbf{y}\|_2}
$$

它回答的是「两个表示朝向是否相近」，不是「两个向量的总能量有多大」。同一方向的$(1,1)$和$(100,100)$余弦相似度为$1$，原始点积却是$200$；如果把点积直接当相似度，向量长度会改变排序。

**注意力分数同时看长度和方向。** 自注意力中的未归一化分数是

$$
s(\mathbf{q},\mathbf{k})
=\frac{\mathbf{q}^{\mathsf{T}}\mathbf{k}}{\sqrt{d}}
=\frac{\|\mathbf{q}\|_2\|\mathbf{k}\|_2}{\sqrt{d}}\cos\theta
$$

所以缩放因子$\sqrt d$控制维度增长带来的数值尺度，但它没有把分数变成余弦相似度；只有另外除以$\|\mathbf{q}\|_2\|\mathbf{k}\|_2$，才只剩方向。

**分类器的权重也有方向。** 线性分类器的一个 logit 可以写成

$$
\mathbf{w}^{\mathsf{T}}\mathbf{x}+b
=\|\mathbf{w}\|_2\|\mathbf{x}\|_2\cos\theta+b
$$

权重方向决定输入朝哪个方向更容易得到高分，权重长度和输入长度则改变分数的幅度。把向量归一化后，分类边界和相似度的解释会更偏向角度；不归一化时，长度仍然参与决策。

**正交残差表示没有被当前方向解释的部分。** 在最小二乘、降维和表示压缩里，先把向量投影到一个子空间，再把正交残差留作误差，是同一个分解。具体的最小二乘公式见[最小二乘即投影](../linear-models/least-squares-as-projection/)篇；这里先记住判据：残差与已使用的方向正交。

## 失效模式与常见误区

**给零向量强行定义角度。** $\mathbf{0}$与任意向量的内积都是$0$，但角度公式的分母含$\|\mathbf{0}\|=0$。因此「零向量与所有向量正交」可以作为内积方程的结果，不能把它解释成零向量有一个确定的$90^\circ$方向。

**把内积为零和向量相等混在一起。** $\langle\mathbf{x},\mathbf{y}\rangle=0$表示正交，不表示$\mathbf{x}=\mathbf{y}$；反过来，$\mathbf{x}=\mathbf{y}\neq\mathbf{0}$时内积是$\|\mathbf{x}\|^2>0$，夹角为$0^\circ$。

**把原始点积当余弦相似度。** 点积会同时受到长度和方向影响。需要比较方向时，必须除以两个范数，并先检查两个向量都非零。

**投影分母少了一个长度。** 标量系数是$\langle\mathbf{x},\mathbf{u}\rangle/\|\mathbf{u}\|^2$，投影向量还要再乘$\mathbf{u}$。把分母写成$\|\mathbf{u}\|$会让结果随$\mathbf{u}$的缩放错误地变化。

**以为正交与坐标轴绑定。** $(1,2)$和$(-2,1)$的坐标都不贴轴，但内积为零。正交由内积定义；若换了加权内积，正交判据也会随度量结构改变，不能只凭欧几里得图形判断。

**把正交当成线性无关的同义词。** 两个非零正交向量一定线性无关，但一般的线性无关向量不必正交，例如$(1,0)$和$(1,1)$。正交是更强的几何条件，且会带来投影唯一性与勾股关系。

## 相关词条

- [内积](../linear-algebra/inner-products/)：三条公理、Cauchy–Schwarz 不等式与内积诱导范数
- [长度与距离](../linear-algebra/lengths-and-distances/)：范数如何生成度量与距离球
- [正交归一基](../linear-algebra/orthonormal-basis/)：把多个方向整理成便于投影和坐标计算的基
- [正交投影](../linear-algebra/orthogonal-projections/)：一般子空间的最近点与投影算子
- [正交矩阵与旋转](../linear-algebra/orthogonal-matrices-and-rotations/)：保持内积、长度和角度的线性映射
- [换基](../linear-algebra/change-of-basis/)：坐标变化与几何量保持的条件
- [最小二乘即投影](../linear-models/least-squares-as-projection/)：正交残差与最小二乘解
- [余弦相似度](../text-representation/cosine-similarity/)：归一化内积在表示学习中的用法
- [自注意力](../attention/self-attention/)：query-key 内积分数与$\sqrt d$缩放
