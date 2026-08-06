---
title: "岭回归与 Lasso：用正则化控制参数解的形状"
tags: ["why-models-learn"]
---

岭回归和 Lasso 都是在最小二乘目标外加入参数惩罚，但它们偏好的解不同：L2 惩罚把所有系数连续地向 0 收缩，L1 惩罚可以把一部分系数直接推到 0。前者通常更稳定，后者可以产生稀疏表示；这不是“一个更强、一个更弱”的关系，而是两种不同的归纳偏置。本文从目标函数、几何和最优性条件出发，推导两种方法的更新公式，再说明标准化、超参数选择、共线性和 Elastic Net 的边界。

![岭回归与 Lasso 从损失等高线到收缩和稀疏](/assets/linear-models/svg/ridge-and-lasso.1.svg)

## 先把拟合目标和参数偏好分开

设 $X\in\mathbb R^{n\times p}$ 是设计矩阵，$y\in\mathbb R^n$ 是标签，参数向量为 $\beta$。普通最小二乘只关心训练残差：

$$
\operatorname{RSS}(\beta)
=\lVert y-X\beta\rVert_2^2.
$$

正则化方法同时关心拟合误差和参数形状。采用带有 $\frac12$ 的统一记号：

$$
\begin{aligned}
J_{\mathrm{ridge}}(\beta)
&=\frac12\lVert y-X\beta\rVert_2^2
+\frac{\lambda}{2}\lVert\beta\rVert_2^2,\\
J_{\mathrm{lasso}}(\beta)
&=\frac12\lVert y-X\beta\rVert_2^2
+\lambda\lVert\beta\rVert_1,
\qquad \lambda\ge0.
\end{aligned}
$$

$\lambda$ 越大，模型越愿意牺牲训练拟合来换取更小或更稀疏的参数。$\lambda=0$ 时两者都退化为普通最小二乘；$\lambda>0$ 时，惩罚项改变的是优化问题本身，不是事后把已经训练好的系数随意截断。

带截距时通常把截距单独写出：

$$
\widehat y=\beta_0\mathbf 1+X\beta,
$$

只惩罚非截距系数 $\beta$。如果连截距也惩罚，模型会被额外推向“所有特征为 0 时预测为 0”，这通常不是想要的尺度偏好。

| 方法 | 目标中的惩罚 | 直接效果 | 典型偏好 |
| --- | --- | --- | --- |
| OLS | 无 | 只减小训练残差 | 在数据可辨识方向上完全追随拟合 |
| Ridge | $\frac{\lambda}{2}\lVert\beta\rVert_2^2$ | 连续收缩所有系数 | 稳定、保留相关特征 |
| Lasso | $\lambda\lVert\beta\rVert_1$ | 部分系数可变成 0 | 稀疏、选择少量特征 |

惩罚不是免费午餐。它通过引入偏差改变方差，是否有利要由验证分布上的风险决定，而不是只看训练 RSS。

## Ridge 的闭式解是方向依赖的收缩

Ridge 目标可微。展开数据项并求梯度：

$$
\begin{aligned}
\nabla_\beta J_{\mathrm{ridge}}
&=-X^{\mathsf T}(y-X\beta)+\lambda\beta\\
&=(X^{\mathsf T}X+\lambda I)\beta-X^{\mathsf T}y.
\end{aligned}
$$

令梯度为零：

$$
(X^{\mathsf T}X+\lambda I)\widehat\beta_{\mathrm{ridge}}
=X^{\mathsf T}y.
$$

当 $\lambda>0$ 时，$X^{\mathsf T}X+\lambda I$ 在实数情形下为正定矩阵，即使 $X$ 的列不满秩也可逆。因此

$$
\widehat\beta_{\mathrm{ridge}}
=
(X^{\mathsf T}X+\lambda I)^{-1}X^{\mathsf T}y.
$$

这个逆矩阵是推导中的表达，不意味着数值实现应该显式求逆。实际计算仍可用 Cholesky、QR、SVD 或稳定的线性系统求解器。

### SVD 说明每个方向收缩多少

令设计矩阵的紧 SVD 为

$$
X=U\Sigma V^{\mathsf T},
\qquad
\Sigma=\operatorname{diag}(s_1,\ldots,s_r),
\qquad
s_j>0.
$$

把标签投影到左奇异向量方向后，Ridge 系数在第 $j$ 个右奇异方向上的因子是

$$
\frac{s_j}{s_j^2+\lambda}\,u_j^{\mathsf T}y.
$$

相对于 OLS 在同一方向上的系数，它的缩放因子是

$$
\frac{s_j^2}{s_j^2+\lambda}.
$$

大奇异值方向的因子更接近 1，小奇异值方向的因子更小。Ridge 因而优先抑制数据中难以辨识、容易放大噪声的方向，而不是简单地对每个坐标使用同一个“百分比”。

若两个特征几乎重复，$X^{\mathsf T}X$ 会在“相减得到差异”的方向上接近奇异；Ridge 给这个方向加上 $\lambda$，让它不再无限放大微小扰动。

### 一维数字例子

取不带截距的

$$
X=
\begin{pmatrix}
1\\2
\end{pmatrix},
\qquad
y=
\begin{pmatrix}
1\\3
\end{pmatrix}.
$$

此时

$$
X^{\mathsf T}X=5,
\qquad
X^{\mathsf T}y=7.
$$

所以不同惩罚强度下

$$
\widehat\beta_{\mathrm{ridge}}(\lambda)
=\frac{7}{5+\lambda}.
$$

| $\lambda$ | Ridge 系数 | 小数近似 |
| --- | --- | --- |
| $0$ | $7/5$ | $1.4$ |
| $1$ | $7/6$ | $1.1667$ |
| $4$ | $7/9$ | $0.7778$ |

系数确实向 0 收缩，但训练标签、特征和损失都没有被修改。$\lambda$ 不是学习率，也不是把梯度乘小的临时开关；它定义了一个新的最优点。

### 重复列时 Ridge 仍能给出稳定坐标

令两列完全相同：

$$
X=
\begin{pmatrix}
1&1\\
1&1
\end{pmatrix},
\qquad
y=
\begin{pmatrix}
1\\3
\end{pmatrix},
\qquad
\lambda=1.
$$

有

$$
X^{\mathsf T}X=
\begin{pmatrix}
2&2\\
2&2
\end{pmatrix},
\qquad
X^{\mathsf T}y=
\begin{pmatrix}
4\\4
\end{pmatrix}.
$$

加上 $\lambda I$ 后：

$$
\begin{pmatrix}
3&2\\
2&3
\end{pmatrix}
\widehat\beta
=
\begin{pmatrix}
4\\4
\end{pmatrix},
\qquad
\widehat\beta_{\mathrm{ridge}}
=
\begin{pmatrix}
4/5\\4/5
\end{pmatrix}.
$$

两个重复特征得到相同系数，合计预测系数为 $8/5$。普通最小二乘只能识别 $\beta_1+\beta_2$，而不能识别它们如何分配；Ridge 用最小平方范数的偏好把分配稳定在对称解附近。它没有创造“哪个重复列更重要”的证据。

## Lasso 的关键是 0 点的次梯度

Lasso 的 $\lVert\beta\rVert_1$ 在任意非零坐标上可微，但在 $\beta_j=0$ 处有一个区间的次梯度：

$$
\partial\lvert\beta_j\rvert
=
\begin{cases}
\{1\},&\beta_j>0,\\
[-1,1],&\beta_j=0,\\
\{-1\},&\beta_j<0.
\end{cases}
$$

因此 Lasso 的最优性条件不是普通梯度等于零，而是存在 $z\in\partial\lVert\beta\rVert_1$，使

$$
X^{\mathsf T}(X\widehat\beta-y)
+\lambda z=0.
$$

对某个坐标 $\widehat\beta_j=0$，只要数据项的推动力落在区间

$$
\left\lvert x_j^{\mathsf T}(y-X\widehat\beta)\right\rvert\le\lambda
$$

内，$z_j$ 就可以在 $[-1,1]$ 中抵消它。这个区间条件正是系数能够精确等于 0 的来源。Ridge 的导数在 0 点只有一个值，不会因为跨过 0 就产生一个平坦的次梯度区间。

## 正交设计下软阈值公式一眼出现

假设特征已经正交归一：

$$
X^{\mathsf T}X=I.
$$

令 OLS 在坐标 $j$ 上的相关量为

$$
z_j=x_j^{\mathsf T}y.
$$

Lasso 目标可以按坐标分开。单个坐标要最小化

$$
\frac12(\beta_j-z_j)^2+\lambda\lvert\beta_j\rvert.
$$

它的解是软阈值函数

$$
S_\lambda(z)
=\operatorname{sign}(z)\max(\lvert z\rvert-\lambda,0),
$$

也可以写成三段：

$$
S_\lambda(z)=
\begin{cases}
z-\lambda,&z>\lambda,\\
0,&\lvert z\rvert\le\lambda,\\
z+\lambda,&z<-\lambda.
\end{cases}
$$

正数被减去 $\lambda$，负数被加回 $\lambda$，靠近 0 的坐标则被直接截成 0。取

$$
z=
\begin{pmatrix}
3/2\\-1/2\\1/10
\end{pmatrix},
\qquad
\lambda=\frac25,
$$

得到

$$
S_{2/5}(z)
=
\begin{pmatrix}
11/10\\-1/10\\0
\end{pmatrix}.
$$

这不是把小系数事后四舍五入，而是原目标在不可微尖角处的精确最优解。

## 坐标下降把软阈值推广到一般设计矩阵

一般情形下各列不正交，坐标之间会相互耦合。固定除第 $j$ 个坐标之外的所有系数，定义部分残差相关量

$$
\rho_j
=x_j^{\mathsf T}
\left(y-X_{-j}\beta_{-j}\right).
$$

只看 $\beta_j$ 时，Ridge 和 Lasso 的一维更新分别是

$$
\begin{aligned}
\beta_j^{\mathrm{ridge}}
&=\frac{\rho_j}{\lVert x_j\rVert_2^2+\lambda},\\
\beta_j^{\mathrm{lasso}}
&=\frac{S_\lambda(\rho_j)}{\lVert x_j\rVert_2^2}.
\end{aligned}
$$

算法循环选择坐标、计算其余项造成的残差、应用对应更新，直到目标函数和参数变化都足够小。Lasso 的阈值仍然存在，但阈值前的 $\rho_j$ 会随其他特征的系数变化；所以不能把正交设计下的软阈值公式不加条件地套到相关特征上。

坐标下降的停止条件必须与验证协议分开。训练目标变化很小，只说明当前优化近似收敛，不说明 $\lambda$ 已经选对，也不说明测试风险最低。

## L1 和 L2 的几何差异

把惩罚写成约束形式，可以直观看到两种解的形状：

$$
\min_\beta\ \lVert y-X\beta\rVert_2^2
\quad
\text{subject to}\quad
\lVert\beta\rVert_2\le t
$$

对应一个圆形或高维球约束，而

$$
\min_\beta\ \lVert y-X\beta\rVert_2^2
\quad
\text{subject to}\quad
\lVert\beta\rVert_1\le t
$$

对应带尖角的菱形或高维交叉多胞体。平方损失的等高线从外向内移动：

- 圆与椭圆通常在坐标轴之外接触，所以 Ridge 很少把某个坐标精确变成 0；
- 菱形的尖角正落在坐标轴上，所以 Lasso 更容易在某个系数为 0 的位置接触；
- 特征高度相关时，损失等高线的方向会旋转，Lasso 可能在相关特征中任意保留一部分，Ridge 则倾向于把权重分摊。

约束半径 $t$ 与惩罚强度 $\lambda$ 是两种等价的描述方式：在满足适当凸性条件时，每个惩罚问题都有对应的约束半径，反之亦然。工程上通常直接调 $\lambda$，因为它便于写进训练目标和交叉验证流程。

## 标准化不是装饰步骤

L1 和 L2 惩罚都直接作用在系数大小上。若一个特征从米改成厘米，保持同一预测需要把对应系数缩小 100 倍；未经标准化时，惩罚会把单位差异误当成特征重要性差异。

常见流程是：

1. 只用训练集估计每列的均值和尺度；
2. 用同一组训练统计量变换训练、验证和测试特征；
3. 在标准化坐标上选择 $\lambda$；
4. 若需要向人解释系数，再换回原单位并同时报告缩放规则。

中心化后通常把截距单独保留且不惩罚。稀疏性也依赖参数化：把同一个物理量拆成两列高度相关的编码，Lasso 选择哪一列可能改变，但预测和联合解释未必改变。

## 如何选择 Ridge、Lasso 或 Elastic Net

Ridge、Lasso 和 Elastic Net 的选择应由数据结构与部署目标共同决定：

| 情形 | 更自然的起点 | 原因 |
| --- | --- | --- |
| 特征很多且彼此相关 | Ridge | 共享权重通常比任意挑一列稳定 |
| 只有少量特征预期真正有用 | Lasso | 直接产生稀疏参数，便于压缩和筛选 |
| 相关特征希望成组保留 | Elastic Net | 同时利用 L1 的稀疏与 L2 的成组稳定 |
| 主要目标是预测而非解释 | 先比较三者 | 以验证风险和资源约束决定，不凭方法名选择 |
| 需要因果或政策结论 | 不能只靠正则化 | 惩罚解决的是拟合偏好，不是混杂识别 |

Elastic Net 把两种惩罚混合：

$$
J_{\mathrm{EN}}(\beta)
=\frac12\lVert y-X\beta\rVert_2^2
+\lambda\left[
\frac{1-\alpha}{2}\lVert\beta\rVert_2^2
+\alpha\lVert\beta\rVert_1
\right],
\qquad
0\le\alpha\le1.
$$

$\alpha=0$ 是 Ridge，$\alpha=1$ 是 Lasso，中间值同时提供收缩和稀疏。对于一组高度相关的特征，L2 部分会降低“只选其中一个”的不稳定性，但不保证得到唯一的科学解释。

## 验证集选择的是惩罚强度

对每个候选 $\lambda$，训练集拟合一个不同的模型。选择过程应该写清楚：

1. 在训练分片内拟合标准化参数和模型系数；
2. 在验证分片上记录 MSE、MAE 或实际业务损失；
3. 选择验证风险低且资源、稀疏度或稳定性满足要求的 $\lambda$；
4. 冻结惩罚、标准化和特征集合后，只在最后使用测试集；
5. 在重复切分或时间外样本上检查所选 $\lambda$ 和系数是否稳定。

如果反复查看测试集来选择 $\lambda$，测试集就被吸收进学习流程，最终分数不再是独立证据。[训练、验证与测试集](../learning-framework/train-validation-test/) 讨论的边界同样适用于正则化强度和惩罚混合比例。

正则化路径也值得观察：从大 $\lambda$ 逐步减小时，Ridge 系数平滑变大，Lasso 系数通常分段进入模型。路径的稳定性可以帮助发现共线性和数据泄漏，但不能代替独立测试。

## 失效模式

**把 Ridge 或 Lasso 当成自动消除过拟合的按钮。** 正则化只提供一种解偏好，$\lambda$ 过大时会欠拟合，过小时仍可能追随噪声。

**不标准化就比较系数或 Lasso 选择结果。** 单位变化会直接改变系数大小和惩罚强度；先固定训练集内的尺度，再讨论稀疏性。

**把 Lasso 的 0 系数当成因果上无效。** 相关特征、样本噪声、编码方式和 $\lambda$ 都会影响选择；0 表示当前优化问题的解，不是干预效应的证明。

**把相关特征中的任一保留列当成唯一真相。** Lasso 可能在等价或近似等价的列之间跳动，Ridge 的联合预测可能更稳定；应报告分组、联合效应和重采样稳定性。

**给截距也套同样惩罚却不说明。** 截距是否惩罚改变了零输入处的基准，通常应中心化特征并单独处理截距。

**用测试集挑选 $\lambda$ 或 $\alpha$.** 这样会把最终评估变成训练反馈；选择必须留在训练/验证协议内。

**把坐标下降收敛当成泛化保证。** 优化算法找到的是给定目标的近似最优解，不能证明特征、任务分布或损失函数选择正确。

## 相关词条

- [线性回归](../linear-models/linear-regression/)：回顾无惩罚最小二乘、设计矩阵、残差和 Ridge 数字例子。
- [最小二乘即投影](../linear-models/least-squares-as-projection/)：从列空间投影解释拟合误差和正规方程。
- [过拟合与正则化](../learning-framework/overfitting-and-regularization/)：比较显式、隐式正则化和验证集选择的泛化边界。
- [偏差—方差权衡](../learning-framework/bias-variance-tradeoff/)：解释收缩为何可能用偏差换取更低方差。
- [训练、验证与测试集](../learning-framework/train-validation-test/)：规定标准化、特征和 $\lambda$ 的选择边界。
- [伪逆](../linear-algebra/pseudoinverse/)：比较秩亏问题中的最小范数解与正则化解。
- [双下降](../learning-framework/double-descent/)：讨论插值附近参数化和解偏好对风险的影响。
