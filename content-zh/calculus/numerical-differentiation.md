---
title: "数值微分:用有限差分检查导数"
tags: ["why-models-learn"]
---

**数值微分**用有限个函数值近似导数，而不是直接沿计算图或符号表达式求导。最基本的做法是在 $x$ 附近取 $x-h$、$x$、$x+h$ 的函数值，用差商估计斜率。步长 $h$ 太大时，局部曲线被直线近似的截断误差明显；$h$ 太小时，两个几乎相等的浮点数相减又会放大舍入误差。本篇推导前向、后向和中心差分的误差阶，再把它们用于多元梯度、Jacobian、Hessian 和机器学习中的梯度检查。

## 有限差分在近似什么

导数的定义本身就是差商的极限：

$$
f'(x)
=
\lim_{h\to0}
\frac{f(x+h)-f(x)}{h}
$$

数值微分把 $h$ 保留为一个小但非零的数。前向差分为

$$
D_h^+f(x)
=
\frac{f(x+h)-f(x)}{h}
$$

后向差分为

$$
D_h^-f(x)
=
\frac{f(x)-f(x-h)}{h}
$$

对 $f(x+h)$ 做 Taylor 展开：

$$
f(x+h)
=
f(x)+hf'(x)+\frac{h^2}{2}f''(x)+O(h^3)
$$

代回前向差分：

$$
D_h^+f(x)
=
f'(x)+\frac h2f''(x)+O(h^2)
$$

所以前向差分的截断误差是 $O(h)$。同理

$$
D_h^-f(x)
=
f'(x)-\frac h2f''(x)+O(h^2)
$$

后向差分的误差阶相同，但一阶误差符号相反。

中心差分把两侧信息对称地放在一起：

$$
D_h^0f(x)
=
\frac{f(x+h)-f(x-h)}{2h}
$$

两侧展开后，偶数阶的一阶对称项抵消：

$$
D_h^0f(x)
=
f'(x)+\frac{h^2}{6}f^{(3)}(x)+O(h^4)
$$

中心差分的截断误差是 $O(h^2)$，在函数值计算代价相近时通常比单侧差分更准确。边界点没有 $x-h$ 时，前向或后向差分仍然有用；中心差分不是任何位置都能直接使用。

![有限差分在两侧取样，步长同时受到截断误差和舍入误差的限制](/assets/calculus/svg/numerical-differentiation.1.svg)

## 用 $e^x$ 看误差阶

取 $f(x)=e^x$，在 $x=0$ 处真实导数是 $f'(0)=1$。用几个步长计算：

| $h$ | 前向差分 | 后向差分 | 中心差分 |
| --- | --- | --- | --- |
| $10^{-1}$ | $1.051709$ | $0.951626$ | $1.001668$ |
| $10^{-2}$ | $1.005017$ | $0.995017$ | $1.000017$ |
| $10^{-3}$ | $1.000500$ | $0.999500$ | $1.000000$ |

前向和后向的误差大致随 $h$ 缩小十倍而缩小十倍；中心差分的误差大致缩小一百倍。以 $h=10^{-2}$ 为例，中心差分的误差约为 $1.67\times10^{-5}$，而前向差分的误差约为 $5.02\times10^{-3}$。

在这个例子中，中心差分的 $h=10^{-4}$ 结果为 $1.000000002$，已经足够接近 $1$。这不是说把 $h$ 继续缩小就一定更好，因为公式中的两个函数值会越来越接近。

## 二阶导数也可以用差分

把两侧 Taylor 展开相加，再减去 $2f(x)$：

$$
f(x+h)-2f(x)+f(x-h)
=
h^2f''(x)+\frac{h^4}{12}f^{(4)}(x)+O(h^6)
$$

因此中心二阶差分为

$$
D_{h,2}^0f(x)
=
\frac{f(x+h)-2f(x)+f(x-h)}{h^2}
=
f''(x)+O(h^2)
$$

仍取 $f(x)=e^x$、$x=0$，真实二阶导数也是 $1$：

| $h$ | 中心二阶差分 | 与 $1$ 的差 |
| --- | --- | --- |
| $10^{-1}$ | $1.000834$ | $8.34\times10^{-4}$ |
| $10^{-2}$ | $1.000008$ | $8.33\times10^{-6}$ |
| $10^{-3}$ | $1.000000$ | 约 $8.34\times10^{-8}$ |

二阶差分除以 $h^2$，所以它对函数值误差更敏感。若每次函数调用本身只精确到误差 $\delta$，分子中的三个数经过相减后，误差大约会被放大到 $\delta/h^2$ 的量级。

## 为什么步长不是越小越好

Taylor 展开只描述截断误差。浮点数计算还会把每次函数值写成近似值：

$$
\widehat f(x)=f(x)+\varepsilon_x
$$

前向差分中的两个函数值相减，误差项变成

$$
\frac{\varepsilon_{x+h}-\varepsilon_x}{h}
$$

当 $h$ 变小时，舍入误差项可能按 $1/h$ 增长。把两种误差粗略合在一起，可以写成

$$
\text{总误差}
\approx
C_{\mathrm{trunc}}h^p
+
C_{\mathrm{round}}\frac{\varepsilon}{h}
$$

其中 $p=1$ 对应前向/后向差分，$p=2$ 对应中心差分，$\varepsilon$ 是机器精度和函数值尺度共同决定的舍入尺度。第一项随 $h$ 变小，第二项随 $h$ 变大，曲线通常有一个中间谷底。

只看误差阶可以得到步长尺度：

$$
h_{\mathrm{forward}}\sim\sqrt{\varepsilon}
\qquad
h_{\mathrm{central}}\sim\varepsilon^{1/3}
$$

双精度机器精度约为 $2.22\times10^{-16}$，对应的两个尺度约为 $1.49\times10^{-8}$ 和 $6.06\times10^{-6}$。它们只是数量级提示，不是所有函数都应硬编码的步长：如果 $x$ 很大、函数输出很大，或导数非常小，还要把变量和函数的尺度放进去。

实践中可以先用相对步长

$$
h=\eta\max(1,|x|)
$$

再试几个相邻的 $\eta$。如果结果随 $h$ 缩小先稳定、后剧烈抖动，前一个稳定区间通常比最后一个最小 $h$ 更可信。

## 多元函数的偏导和梯度

令 $\boldsymbol e_j$ 是第 $j$ 个坐标方向。对 $f:\mathbb R^n\to\mathbb R$：

$$
\frac{\partial f}{\partial x_j}(\boldsymbol x)
\approx
\frac{f(\boldsymbol x+h\boldsymbol e_j)-f(\boldsymbol x-h\boldsymbol e_j)}{2h}
$$

把每个坐标的估计排起来，就得到数值梯度

$$
\widehat{\nabla f}(\boldsymbol x)
=
\begin{pmatrix}
\widehat{\partial_1f}\\
\vdots\\
\widehat{\partial_nf}
\end{pmatrix}
$$

取

$$
f(u,v)=u^2+3uv+v^2
\qquad
(\,u,v\,)=(1,2)
$$

解析梯度是

$$
\nabla f(1,2)
=
\begin{pmatrix}
8\\7
\end{pmatrix}
$$

取 $h=0.1$，前向差分给出

$$
\widehat{\nabla f}_{+}
\approx
\begin{pmatrix}
8.1\\7.1
\end{pmatrix}
$$

这是因为前向差分保留了二阶项。中心差分给出

$$
\widehat{\nabla f}_{0}
\approx
\begin{pmatrix}
8\\7
\end{pmatrix}
$$

这个函数是二次多项式，中心公式中的三阶导数为零，所以在精确算术中中心差分对任意 $h$ 都正好恢复梯度；浮点输出只会有末位误差。

方向导数也只需要沿一个方向取两次函数值。对单位向量 $\boldsymbol u$：

$$
D_{\boldsymbol u}f(\boldsymbol x)
\approx
\frac{f(\boldsymbol x+h\boldsymbol u)-f(\boldsymbol x-h\boldsymbol u)}{2h}
$$

在前面的点沿 $\boldsymbol u=(3/5,4/5)$，真实方向导数为

$$
\nabla f(1,2)^{\mathsf T}\boldsymbol u
=
8\cdot\frac35+7\cdot\frac45
=
10.4
$$

中心差分用 $h=0.1$ 得到 $10.400000$。如果只想检查一个具体方向，不必先计算所有偏导；如果要回传到每个参数，就需要完整的梯度。

## Jacobian 的逐列检查

对向量函数 $F:\mathbb R^n\to\mathbb R^m$，第 $j$ 列可以用

$$
\widehat J_{:,j}
=
\frac{
F(\boldsymbol x+h\boldsymbol e_j)
-
F(\boldsymbol x-h\boldsymbol e_j)
}{2h}
$$

独立计算。这样第 $j$ 列仍然表示输入第 $j$ 个方向的输出变化，和 [雅可比矩阵](../calculus/jacobian/) 的行列约定一致。

取

$$
F(u,v)
=
\begin{pmatrix}
u^2+3v\\
uv
\end{pmatrix}
$$

在 $(1,2)$，

$$
J_F(1,2)
=
\begin{pmatrix}
2&3\\
2&1
\end{pmatrix}
$$

用中心差分得到第一列约为 $(2,2)^{\mathsf T}$，第二列约为 $(3,1)^{\mathsf T}$。如果把每次扰动的输出变化写成一行，最后拼出来的矩阵会变成正确 Jacobian 的转置；检查时要先声明“每一列对应一个输入方向”。

向量函数的有限差分成本随输入维数增长：中心差分要为每个输入坐标调用两次 $F$。这适合小规模数值检查，不适合替代大模型的训练过程。

## Hessian 和混合偏导

对角 Hessian 元素用一维二阶差分：

$$
\frac{\partial^2f}{\partial x_i^2}(\boldsymbol x)
\approx
\frac{
f(\boldsymbol x+h\boldsymbol e_i)
-2f(\boldsymbol x)
+f(\boldsymbol x-h\boldsymbol e_i)
}{h^2}
$$

对 $i\ne j$，混合偏导可以用四个角点：

$$
\frac{\partial^2f}{\partial x_i\partial x_j}(\boldsymbol x)
\approx
\frac{
f(\boldsymbol x+h\boldsymbol e_i+h\boldsymbol e_j)
-f(\boldsymbol x+h\boldsymbol e_i-h\boldsymbol e_j)
-f(\boldsymbol x-h\boldsymbol e_i+h\boldsymbol e_j)
+f(\boldsymbol x-h\boldsymbol e_i-h\boldsymbol e_j)
}{4h^2}
$$

对

$$
f(u,v)=u^2+3uv+v^2
$$

真实 Hessian 是

$$
H_f=
\begin{pmatrix}
2&3\\
3&2
\end{pmatrix}
$$

在 $(1,2)$ 取 $h=0.1$，二阶差分分别给出 $2$、$3$、$2$，并且混合偏导的两个方向相同。数值 Hessian 不对称时，先检查步长、函数噪声和坐标索引；对于足够光滑的标量函数，解析 Hessian 的混合偏导应满足 [偏导数](../calculus/partial-derivatives/) 篇中说明的对称性条件。

二阶差分需要更多函数调用，也更容易放大函数值噪声。实际优化中常用 Hessian-向量积或自动微分，有限差分更适合验证小问题和局部实现。

## 机器学习中的梯度检查

设参数向量为 $\boldsymbol\theta$，解析代码给出 $\boldsymbol g$，数值检查对第 $i$ 个参数计算

$$
\widehat g_i
=
\frac{
L(\boldsymbol\theta+h\boldsymbol e_i)
-
L(\boldsymbol\theta-h\boldsymbol e_i)
}{2h}
$$

然后逐坐标比较，而不是只比较总损失。一个常用的相对误差是

$$
\operatorname{err}_i
=
\frac{|g_i-\widehat g_i|}
{\max(1,|g_i|,|\widehat g_i|)}
$$

分母中的 $1$ 避免两个都很小的梯度把相对误差放大到没有解释价值。也可以报告最大误差和对应参数索引。

取一个单神经元的平方损失：

$$
L(w_1,w_2)
=
\frac12(2w_1-w_2-3)^2
$$

在 $(w_1,w_2)=(1,0)$，残差是 $-1$，解析梯度为

$$
\nabla L=
\begin{pmatrix}
-2\\1
\end{pmatrix}
$$

用中心差分和 $h=10^{-3}$ 得

| 参数 | 解析梯度 | 中心差分 |
| --- | --- | --- |
| $w_1$ | $-2$ | $-2.000000$ |
| $w_2$ | $1$ | $1.000000$ |

这是一个二次函数，中心差分在精确算术中再次正好命中。真实网络的损失通常包含非线性，数值结果会有截断和浮点误差，但应该在一段步长范围内稳定接近解析梯度。

梯度检查的隔离条件很重要：

- 固定同一批输入、标签和随机数种子；
- 暂时关闭 dropout、随机增强和会改变状态的层；
- 只选少量参数，用高精度或较小模型先检查；
- 同时测试正梯度和负梯度，确认更新方向没有整体反号；
- 检查损失函数的 reduction 是 sum 还是 mean，避免把广播与归约的尺度问题误判成局部导数错误。

如果函数调用包含随机噪声，$L(\boldsymbol\theta+h\boldsymbol e_i)$ 和 $L(\boldsymbol\theta-h\boldsymbol e_i)$ 的差可能主要来自噪声，而不是参数扰动。固定随机状态只能减少一种误差；对本身随机的目标，还要重复采样并报告方差。

## 不光滑和不可重复函数的边界

有限差分默认附近函数足够平滑。以下情况要单独解释：

- ReLU 在零点左右斜率不同。对 $\operatorname{ReLU}(x)$ 在 $0$ 使用对称差分会得到 $(h-0)/(2h)=1/2$，这不是经典导数，也不等于实现选择的 $0$ 或 $1$ 次梯度。
- max 在并列最大值处可能没有唯一梯度。扰动方向不同，有限差分可能落到不同的最大值位置。
- 整数索引、取整、分支和离散采样不是小的连续扰动。改变一个参数可能让执行路径突然跳变。
- 归一化和 softmax 依赖多个坐标。只扰动一个坐标时，其他输出也可能变化，不能用逐分量公式期待非对角项为零。
- 缓存、随机状态或异步设备使两次函数调用不具备相同条件。先确认比较的是同一个函数，而不是两次不同的实验。

这些情况不是“把 $h$ 调得更小”就能修复。先确定目标是否在检查点可导，再决定应比较经典导数、次梯度、方向导数还是某种实现约定。

## 复步微分：避免相减消去

对适合扩展到复数且在点附近解析的函数，可以使用复步公式：

$$
f'(x)
\approx
\frac{\operatorname{Im}f(x+\mathrm ih)}{h}
$$

它不需要计算两个接近的实数之差，因此可以把 $h$ 取得很小而不遭遇同样的相消误差。由

$$
\sin(x+\mathrm ih)
=
\sin x\cosh h+\mathrm i\cos x\sinh h
$$

可见

$$
\frac{\operatorname{Im}\sin(x+\mathrm ih)}{h}
=
\cos x\frac{\sinh h}{h}
\to\cos x
$$

但复步法要求函数实现保留复数的虚部，不能经过绝对值、比较、ReLU 或只接受实数的黑盒算子。它是对解析函数的数值检查方法，不是任意机器学习计算图都能直接套用的接口。

## 数值微分和自动微分各自负责什么

有限差分只需要函数值，适合做黑盒验证；它的误差受步长、浮点精度、噪声和函数平滑性影响。[自动微分](../calculus/automatic-differentiation/) 沿计算图应用局部导数，通常不需要用相近函数值相减，但实现仍可能有转置、广播、归约和 reduction 约定错误。

因此两者的关系是交叉检查，不是谁把谁替代：

| 方法 | 需要什么 | 主要误差或成本 | 适合检查 |
| --- | --- | --- | --- |
| 前向/后向差分 | 函数值 | 一阶截断误差，单侧边界友好 | 快速粗查 |
| 中心差分 | 函数值 | 二阶截断误差，两侧函数调用 | 小规模梯度检查 |
| 复步微分 | 支持复数的函数值 | 解析性和复数实现限制 | 光滑黑盒函数 |
| 自动微分 | 可追踪计算图和局部导数 | 实现/显存/算子支持 | 训练和精确局部传播 |
| 符号微分 | 可操作的符号表达式 | 表达式膨胀 | 小型解析推导 |

检查一个新算子时，可以先用一个二维或三维输入做中心差分，再把结果和解析 Jacobian 或反向梯度比较；不要一开始就用整批大模型的损失差分来定位问题。

## 常见失效模式

- **把 $h$ 固定成一个魔法数字。** 试一组相邻步长，观察稳定区间，而不是只看一个输出。
- **误以为越小越准。** 截断误差会下降，但相消和舍入误差会在很小的 $h$ 下上升。
- **用前向差分期待中心差分的精度。** 前向/后向是 $O(h)$，中心一阶差分才是 $O(h^2)$。
- **把 Jacobian 的行列方向拼反。** 每次沿一个输入坐标扰动时，输出向量是对应的一列。
- **在不可导点把数值斜率当经典导数。** ReLU 零点的对称差分 $1/2$ 是取样结果，不是唯一导数。
- **梯度检查时让随机状态变化。** 两次损失评估必须尽量使用相同的输入、标签、随机状态和 reduction。
- **只检查梯度的绝对误差。** 大梯度和小梯度需要尺度化比较，同时报告索引和相对误差。
- **用有限差分代替训练期反向传播。** 它需要每个参数多次函数调用，且误差随维度和噪声迅速累积。

## 相关词条

- [导数](../calculus/derivatives/)：有限差分逼近的极限对象。
- [偏导数](../calculus/partial-derivatives/)：逐坐标扰动得到多元函数的偏导。
- [梯度](../calculus/gradient/)：把多次一维差分排成输入空间中的向量。
- [雅可比矩阵](../calculus/jacobian/)：把向量函数的逐列有限差分排成矩阵。
- [Hessian 矩阵](../calculus/hessian/)：二阶和混合差分对应的局部曲率矩阵。
- [矩阵微积分恒等式](../calculus/matrix-calculus-identities/)：解析矩阵梯度与有限差分的对照。
- [逐分量导数](../calculus/elementwise-derivatives/)：判断非对角有限差分是否应该为零。
- [广播与归约导数](../calculus/broadcast-and-reduction-derivatives/)：检查共享参数梯度的累加和缩放。
- [自动微分](../calculus/automatic-differentiation/)：训练时沿计算图传播局部导数。
