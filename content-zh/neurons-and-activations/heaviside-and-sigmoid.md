---
title: "阶跃与 sigmoid：从硬阈值到可训练概率"
tags: ["why-models-learn"]
---

阶跃函数把一个实数压成 0 或 1：过了阈值就开，没过阈值就关。sigmoid 则把同一个硬判定替换成平滑的 $(0,1)$ 输出：它在中间区域保留梯度，在两端逐渐饱和。本文从阶跃函数的决策边界出发，先说明它为什么适合表达分类规则却不适合直接做梯度训练，再推导 sigmoid 的导数、反函数、概率和对数几率解释，最后把损失函数、阈值、代价和数值稳定性放到同一张图里。

![左侧为在零点突然跳变的阶跃函数，右侧为经过零点并逐渐饱和的 sigmoid；二者的决策边界都在 z=0](/assets/neurons-and-activations/svg/heaviside-and-sigmoid.1.svg)

## 一个硬阈值就是一个分类器

设神经元先算出预激活值

$$
z=\mathbf{w}^{\mathsf T}\mathbf{x}+b.
$$

如果只关心「是否达到阈值」，可以用阶跃函数：

$$
H(z)=
\begin{cases}
0,&z<0,\\
1,&z\geq0.
\end{cases}
$$

于是神经元的输出是

$$
\hat y=H\left(\mathbf{w}^{\mathsf T}\mathbf{x}+b\right).
$$

这不是一个模糊的比喻。它把输入空间切成两半：

$$
\mathcal{R}_1
=\left\{\mathbf{x}:\mathbf{w}^{\mathsf T}\mathbf{x}+b\geq0\right\},
\qquad
\mathcal{R}_0
=\left\{\mathbf{x}:\mathbf{w}^{\mathsf T}\mathbf{x}+b<0\right\}.
$$

两块区域的分界面是

$$
\mathbf{w}^{\mathsf T}\mathbf{x}+b=0.
$$

在二维里它是一条直线，在三维里是一张平面，在更高维里是超平面。阶跃函数本身只负责把分界面的一侧标为 1、另一侧标为 0；真正决定分界面朝向和位置的是 $\mathbf{w}$ 与 $b$。

取

$$
\mathbf{w}=(1,-1),
\qquad
b=-0.5.
$$

此时

$$
z=x_1-x_2-0.5,
\qquad
x_1-x_2=0.5
$$

是决策边界。三个点的计算如下：

| 输入 | 预激活值 $z$ | 阶跃输出 $H(z)$ | 位置 |
| --- | ---: | ---: | --- |
| $(2,1)$ | $0.5$ | $1$ | 边界的正侧 |
| $(1,2)$ | $-1.5$ | $0$ | 边界的负侧 |
| $(0,0)$ | $-0.5$ | $0$ | 边界的负侧 |

这个例子也说明了偏置的作用：若 $b=0$，边界会被固定为 $x_1-x_2=0$，一定穿过原点；$b=-0.5$ 把整条边界平移到了另一处。

## 阶跃函数在零点发生了什么

阶跃函数的图像很简单，但它在零点有一个必须说清楚的约定。本文取 $H(0)=1$，也可以取 $H(0)=0$，或者在分类指标里把恰好落在边界的样本交给另一条规则。这个单点约定不会改变几乎所有输入的位置，却会改变边界样本的预测。

对任意 $z\ne0$，阶跃函数在一小段邻域里保持不变。因此它的导数是

$$
H'(z)=0
\qquad
(z\ne0),
$$

而在 $z=0$ 处不可导。用分段形式写就是

$$
H'(z)=
\begin{cases}
0,&z<0,\\
\text{不存在},&z=0,\\
0,&z>0.
\end{cases}
$$

不可导的单点不是主要麻烦。即使我们在 $z=0$ 人为选一个次梯度，绝大多数样本仍然落在 $z\ne0$ 的区域，梯度依旧是零。阶跃函数的问题是：它对「离边界多远」没有连续的反馈。

## 为什么普通梯度下降学不动阶跃函数

假设用平方损失看一个样本：

$$
L=\frac12\left(H(z)-y\right)^2,
\qquad
z=\mathbf{w}^{\mathsf T}\mathbf{x}+b.
$$

在 $z\ne0$ 的地方，链式法则只能给出

$$
\frac{\partial L}{\partial z}
=\left(H(z)-y\right)H'(z)=0.
$$

因此

$$
\frac{\partial L}{\partial\mathbf{w}}
=\frac{\partial L}{\partial z}\mathbf{x}
=\mathbf{0},
\qquad
\frac{\partial L}{\partial b}=0.
$$

举一个具体的错分样本：

$$
x=1,
\qquad
w=-0.2,
\qquad
b=0,
\qquad
y=1.
$$

它的预激活值是 $z=-0.2$，所以 $H(z)=0$，损失是

$$
L=\frac12(0-1)^2=0.5.
$$

但是 $z=-0.2$ 不在跳点，$H'(z)=0$，梯度仍是零。模型知道自己错了，却没有一个「应该向右移动多少」的微分信号。

这里要区分两种学习规则。感知机可以使用错分驱动的更新：

$$
\mathbf{w}
\leftarrow
\mathbf{w}+\eta(y-\hat y)\mathbf{x},
\qquad
b
\leftarrow
b+\eta(y-\hat y),
$$

其中 $\hat y=H(z)$。在上面的样本中，若 $\eta=0.5$，更新后

$$
w=-0.2+0.5(1-0)\cdot1=0.3,
\qquad
b=0+0.5(1-0)=0.5.
$$

新的预激活值是 $z=0.8$，预测翻到 1。这条规则直接使用了「预测错了」这个离散事件，不是对阶跃函数求导。[经典感知机](../linear-models/perceptron-classic/)展开了它的更新、收敛条件和不可分数据；本篇只要记住，感知机能训练阶跃输出，不等于阶跃函数适合现代反向传播。

## sigmoid 把硬开关换成平滑过渡

sigmoid，也叫 logistic 函数，定义为

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

它仍然是单调递增函数，但不再突然跳变。几个基本值直接来自定义：

$$
\lim_{z\to-\infty}\sigma(z)=0,
\qquad
\sigma(0)=\frac12,
\qquad
\lim_{z\to+\infty}\sigma(z)=1.
$$

因此它把整个实数轴映射到开区间 $(0,1)$。输出永远不会真正等于 0 或 1，但在足够远的两端会非常接近它们。

### 导数从定义逐步算出

把负指数暂时记为 $u=e^{-z}$，则

$$
\sigma(z)=(1+u)^{-1}.
$$

因为

$$
\frac{\mathrm{d}u}{\mathrm{d}z}=-e^{-z}=-u,
$$

所以

$$
\begin{aligned}
\sigma'(z)
&=-(1+u)^{-2}\frac{\mathrm{d}u}{\mathrm{d}z}\\
&=(1+u)^{-2}u\\
&=\frac{e^{-z}}{(1+e^{-z})^2}.
\end{aligned}
$$

再把 $\sigma(z)$ 与 $1-\sigma(z)$ 相乘：

$$
\sigma(z)\bigl(1-\sigma(z)\bigr)
=\frac{1}{1+e^{-z}}
\left(1-\frac{1}{1+e^{-z}}\right)
=\frac{e^{-z}}{(1+e^{-z})^2}.
$$

于是得到最常用的形式：

$$
\sigma'(z)=\sigma(z)\bigl(1-\sigma(z)\bigr).
$$

若记 $p=\sigma(z)$，那么 $p\in(0,1)$，并且

$$
0<\sigma'(z)=p(1-p)\leq\frac14.
$$

上界来自二次函数 $p(1-p)$ 在 $p=1/2$ 处达到最大值。也就是说，sigmoid 在 $z=0$ 附近最敏感：

$$
\sigma'(0)=\frac14.
$$

当 $p$ 接近 0 或 1 时，$p(1-p)$ 接近 0，这就是 sigmoid 的饱和区。

### 数值表比图像更能说明饱和

下面列出一组对称的预激活值。由于

$$
\sigma(-z)=1-\sigma(z),
\qquad
\sigma'(-z)=\sigma'(z),
$$

左右两端的导数相同，输出则围绕 $1/2$ 对称。

| $z$ | $\sigma(z)$ | $\sigma'(z)$ | 所在区域 |
| ---: | ---: | ---: | --- |
| $-4$ | $0.017986$ | $0.017663$ | 负侧饱和 |
| $-2$ | $0.119203$ | $0.104994$ | 负侧仍有梯度 |
| $-1$ | $0.268941$ | $0.196612$ | 过渡区 |
| $0$ | $0.500000$ | $0.250000$ | 最敏感 |
| $1$ | $0.731059$ | $0.196612$ | 过渡区 |
| $2$ | $0.880797$ | $0.104994$ | 正侧仍有梯度 |
| $4$ | $0.982014$ | $0.017663$ | 正侧饱和 |

所以「sigmoid 会饱和」不是一句只在大模型里才成立的口号：$z=4$ 时输出看起来已经很接近 1，但局部导数只有约 $0.0177$；若反向传播还要经过多层 sigmoid，这个小因子会被继续相乘。

## sigmoid 也可以从一个微分方程得到

sigmoid 的导数公式还给出一个反过来的构造：

$$
\frac{\mathrm{d}s}{\mathrm{d}z}=s(1-s),
\qquad
s(0)=\frac12.
$$

这是一个先增长、后变慢的自限增长方程。分离变量：

$$
\frac{\mathrm{d}s}{s(1-s)}=\mathrm{d}z.
$$

部分分式分解为

$$
\frac{1}{s(1-s)}
=\frac1s+\frac1{1-s}.
$$

注意对 $1/(1-s)$ 积分时会多出一个负号，因此

$$
\int\frac{\mathrm{d}s}{s(1-s)}
=\log s-\log(1-s)
=\log\frac{s}{1-s}.
$$

从初值 $s(0)=1/2$ 出发，积分常数为 0，于是

$$
\log\frac{s}{1-s}=z.
$$

两边取指数：

$$
\frac{s}{1-s}=e^z.
$$

解出 $s$：

$$
s=e^z(1-s)
\Longrightarrow
s(1+e^z)=e^z
\Longrightarrow
s=\frac{e^z}{1+e^z}
=\frac1{1+e^{-z}}.
$$

这条推导解释了 sigmoid 为什么在中点斜率最大、两端越来越平：增长速度本身就是当前输出与剩余空间 $1-s$ 的乘积。

## sigmoid 是平滑阈值，不是另一条决策边界

阶跃函数用 $z=0$ 做硬判定，而 sigmoid 的默认判定通常是

$$
\hat y=
\mathbf{1}\{\sigma(z)\geq0.5\}.
$$

因为 sigmoid 严格递增，且 $\sigma(0)=0.5$，所以

$$
\mathbf{1}\{\sigma(z)\geq0.5\}
=\mathbf{1}\{z\geq0\}
=H(z).
$$

这意味着，若只用默认的 0.5 阈值把概率转成类别，sigmoid 没有移动原来的边界。它改变的是边界附近的表达方式：阶跃只说「这边/那边」，sigmoid 还给出「离边界大约多远」的连续分数。

阈值不一定是 0.5。若要求

$$
\sigma(z)\geq\tau,
\qquad
0<\tau<1,
$$

利用 sigmoid 的反函数可得

$$
z\geq\operatorname{logit}(\tau)
=\log\frac{\tau}{1-\tau}.
$$

例如 $\tau=0.8$ 时，

$$
\operatorname{logit}(0.8)
=\log\frac{0.8}{0.2}
=\log4
\approx1.386294.
$$

同一个模型、同一组 sigmoid 输出，只把分类阈值从 0.5 提高到 0.8，就把 logit 边界从 0 推到了 1.386294。模型的连续输出没有变，改变的是最后一步的决策规则。

## 对数几率让概率与线性分数接上

sigmoid 的反函数是 logit：

$$
\operatorname{logit}(p)
=\log\frac{p}{1-p}.
$$

代入 $p=\sigma(z)$：

$$
\frac{p}{1-p}
=\frac{1/(1+e^{-z})}{e^{-z}/(1+e^{-z})}
=e^z,
$$

因此

$$
\operatorname{logit}(\sigma(z))=z.
$$

这个等式把三个量放在一条线上：

$$
\text{线性 logit }z
\quad\longleftrightarrow\quad
\text{几率 }\frac{p}{1-p}
\quad\longleftrightarrow\quad
\text{概率 }p.
$$

比如 $z=1$ 时

$$
p=\sigma(1)\approx0.731059,
\qquad
\frac{p}{1-p}=e\approx2.718282.
$$

这里的 2.718282 不是「正类概率为 2.7」，而是正类几率约为负类几率的 2.7 倍。$p=0.731059$ 仍然是一个小于 1 的概率。

把这个关系放到二分类模型里：

$$
\Pr(y=1\mid\mathbf{x})
=\sigma\left(\mathbf{w}^{\mathsf T}\mathbf{x}+b\right).
$$

于是

$$
\log\frac{\Pr(y=1\mid\mathbf{x})}
{1-\Pr(y=1\mid\mathbf{x})}
=\mathbf{w}^{\mathsf T}\mathbf{x}+b.
$$

这就是 logistic 回归的含义：不是说输入和概率之间是线性的，而是说输入和正负几率的对数之间是线性的。[逻辑回归](../linear-models/logistic-regression/)会进一步展开似然、梯度、Hessian、正则化和决策阈值。

## BCE 让 sigmoid 的梯度恰好变简单

对二分类标签 $y\in\{0,1\}$，给定概率 $p=\sigma(z)$，二元交叉熵是

$$
L
=-y\log p-(1-y)\log(1-p).
$$

先对 $p$ 求导：

$$
\frac{\partial L}{\partial p}
=-\frac{y}{p}+\frac{1-y}{1-p}.
$$

再乘上 sigmoid 的导数：

$$
\begin{aligned}
\frac{\partial L}{\partial z}
&=\frac{\partial L}{\partial p}\frac{\partial p}{\partial z}\\
&=\left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p)\\
&=-y(1-p)+(1-y)p\\
&=p-y.
\end{aligned}
$$

这一步的抵消很重要。若用平方损失，梯度是

$$
\frac{\partial}{\partial z}
\frac12(p-y)^2
=(p-y)p(1-p),
$$

会额外乘上一个在饱和区很小的 $p(1-p)$；BCE 与 sigmoid 配合后，梯度直接变成预测概率减标签。

取两个对称样本：

| 标签 $y$ | $z$ | $p=\sigma(z)$ | BCE | $\partial L/\partial z=p-y$ |
| ---: | ---: | ---: | ---: | ---: |
| $1$ | $-2$ | $0.119203$ | $2.126928$ | $-0.880797$ |
| $1$ | $2$ | $0.880797$ | $0.126928$ | $-0.119203$ |
| $0$ | $-2$ | $0.119203$ | $0.126928$ | $0.119203$ |
| $0$ | $2$ | $0.880797$ | $2.126928$ | $0.880797$ |

第一行是一个「很自信但预测错了」的正样本：模型给出 $0.119203$，损失很大，梯度为负，梯度下降会把 $z$ 往右推。第二行已经把正样本推到正确一侧，损失和梯度都变小。

这也说明不能随意把输出头、概率损失和标签语义拆开搭配。若网络输出的是 logit $z$，训练时可以直接使用带 logits 的 BCE 形式；若网络已经先算了 sigmoid，再使用概率形式 BCE；两次 sigmoid 或把概率当 logit 都会改变数值和梯度。

## 不同代价会产生不同阈值

默认阈值 0.5 只在两类决策代价对称时自然。设已经有一个校准概率 $p=\Pr(y=1\mid\mathbf{x})$，把样本判为正类的期望错误代价是

$$
C_{\mathrm{FP}}(1-p),
$$

判为负类的期望错误代价是

$$
C_{\mathrm{FN}}p.
$$

选择正类的条件是前者不大于后者：

$$
C_{\mathrm{FP}}(1-p)
\leq C_{\mathrm{FN}}p.
$$

整理得到

$$
p\geq
\frac{C_{\mathrm{FP}}}{C_{\mathrm{FP}}+C_{\mathrm{FN}}}.
$$

若误报一次的代价是 $C_{\mathrm{FP}}=4$，漏报一次的代价是 $C_{\mathrm{FN}}=1$，则

$$
p\geq\frac45=0.8,
\qquad
z\geq\log4\approx1.386294.
$$

这个方向符合直觉：误报更贵，就要求更高的正类概率才采取正类动作。阈值不是模型参数，也不是训练集上随手挑到最高 accuracy 的常数；它应该在代表部署代价的验证数据上选择，并封存测试集作最终报告。

如果概率没有校准，$\sigma(z)=0.8$ 只表示模型分数经过 sigmoid 后的数值，不保证相似分数的样本中恰好有 80% 为正类。阈值选择和概率校准是两个问题：前者决定行动，后者决定概率能不能被当作风险读数。

## 斜率参数控制过渡带

有时把 sigmoid 写成带斜率参数的形式：

$$
\sigma_k(z)=\frac{1}{1+e^{-kz}},
\qquad
k>0.
$$

求导得到

$$
\sigma_k'(z)
=k\sigma_k(z)\bigl(1-\sigma_k(z)\bigr),
\qquad
\sigma_k'(0)=\frac{k}{4}.
$$

$k$ 越大，过渡区越窄，曲线越接近阶跃；$k$ 越小，过渡区越宽，曲线越平。取 $z=1$：

| $k$ | $\sigma_k(1)$ | $\sigma_k'(1)$ | 形状 |
| ---: | ---: | ---: | --- |
| $0.5$ | $0.622459$ | $0.117502$ | 过渡较宽 |
| $1$ | $0.731059$ | $0.196612$ | 标准 sigmoid |
| $2$ | $0.880797$ | $0.209987$ | 更接近硬阈值 |
| $5$ | $0.993307$ | $0.033241$ | 在 $z=1$ 已接近饱和 |

注意 $k$ 增大时，零点处的最大斜率虽然增大，但指定的 $z=1$ 可能已经进入饱和区。不能只看 $\sigma_k'(0)$ 就说某个 $k$ 一定更好；还要看训练过程中预激活值实际落在什么范围。

## sigmoid 也能理解成带噪声的阶跃

设硬阈值不是固定在 0，而是受到一个随机阈值 $T$ 的扰动。若 $T$ 服从尺度为 $s$ 的 logistic 分布，其累积分布函数为

$$
\Pr(T\leq t)=\frac{1}{1+e^{-t/s}}.
$$

对给定的 $z$，硬阈值输出为 $H(z-T)$。它的期望是

$$
\begin{aligned}
\mathbb{E}_T[H(z-T)]
&=\Pr\bigl(H(z-T)=1\bigr)\\
&=\Pr(T\leq z)\\
&=\frac{1}{1+e^{-z/s}}\\
&=\sigma\left(\frac{z}{s}\right).
\end{aligned}
$$

所以 sigmoid 可以被看成许多带有不确定阈值的硬判定的平均。$s$ 大，阈值扰动大，输出过渡宽；$s$ 小，阈值集中在 0 附近，平均结果更像阶跃。这个解释说明平滑不是把类别含义改成了另一种东西，而是把边界附近的不确定性显式保留下来。

## 数值计算不能把 sigmoid 写得太天真

数学表达式

$$
\sigma(z)=\frac1{1+e^{-z}}
$$

在纸面上没有问题，但直接按这一个分式计算会遇到指数溢出。若 $z$ 很小，$e^{-z}$ 可能超过浮点数能表示的范围；若 $z$ 很大，先算 $1-\sigma(z)$ 又可能发生灾难性消减。

一个等价而稳定的分段形式是

$$
\sigma(z)=
\begin{cases}
\dfrac1{1+e^{-z}},&z\geq0,\\[6pt]
\dfrac{e^z}{1+e^z},&z<0.
\end{cases}
$$

两段都只计算不超过 1 的指数。对损失也应在 logit 空间计算。把

$$
\operatorname{softplus}(z)=\log(1+e^z)
$$

代入 BCE，可以写成

$$
L(z,y)=\operatorname{softplus}(z)-yz.
$$

稳定的等价形式是

$$
L(z,y)
=\max(0,z)-yz+\log\left(1+e^{-\lvert z\rvert}\right).
$$

最后一个指数的指数部分总是不大于 0。这里用绝对值是为了避免表格分隔符与 Markdown 的竖线混淆，也恰好提醒我们：稳定公式的关键是先取 $z$ 的大小，再选择安全的一侧。

实际检查时可以用四个极端值：

| $z$ | 理论上的 $\sigma(z)$ 趋势 | 应该检查什么 |
| ---: | --- | --- |
| $-1000$ | 接近 $0$ | 不出现 NaN，BCE 正类损失仍有限 |
| $-20$ | 很小的正数 | 梯度方向正确 |
| $20$ | 很接近 $1$ | 不把 $1-p$ 直接算成错误的 0 |
| $1000$ | 接近 $1$ | 不出现正向指数溢出 |

## 与 tanh 的关系

sigmoid 与 tanh 不是两条毫无关系的曲线：

$$
\tanh(z)=2\sigma(2z)-1.
$$

反过来，

$$
\sigma(z)=\frac12\left(1+\tanh\frac z2\right).
$$

因此两者的形状可以互相转换，但输出中心不同：

| 函数 | 输出范围 | 中点 | 最大导数 | 常见位置 |
| --- | --- | ---: | ---: | --- |
| sigmoid $\sigma(z)$ | $(0,1)$ | $0.5$ | $0.25$ | 二分类概率输出 |
| tanh$(z)$ | $(-1,1)$ | $0$ | $1$ | 需要零中心的表示 |

这张表不能单独用来决定隐藏层激活。sigmoid 输出不零中心、尾部会饱和；tanh 虽然零中心，也会在两端饱和；网络深度、初始化、归一化、残差连接和任务损失都会改变实际效果。[tanh 激活](../neurons-and-activations/tanh/)专门展开它的导数、零中心性质和饱和边界。

## 失效模式

**把阶跃函数直接放进需要反向传播的隐藏层。** 只要样本远离跳点，导数就是零。若确实要硬决策，可以把阶跃放到推理阶段，训练阶段保留可微代理，或者使用感知机式的错分更新。

**把 sigmoid 当成隐藏层的默认选项。** 它能表达非线性，但每一层都会把输出限制在 $(0,1)$，深层链式相乘还会遇到饱和。先看预激活分布和梯度统计，再决定是否使用它；不能因为输出像概率，就把它复制到所有隐藏层。

**把 sigmoid 输出的数值当成自动校准的概率。** sigmoid 只规定了一个单调映射。数据切分、正则化、类别不平衡和训练目标都会影响校准；需要概率语义时，必须在独立校准数据上检查可靠性图、分箱误差或其他校准指标。

**损失与输出头重复或错配。** 已经输出 logit 的模型不要再手动套一次 sigmoid 后交给要求 logits 的损失；已经得到概率的模型也不要把概率当作未变换的 logit。先写清网络最后一层到底输出什么，再选对应损失。

**默认阈值遮住代价。** 0.5 是对称代价下的方便起点，不是所有业务的正确答案。误报、漏报、资源容量和拒答选项都可能改变阈值；阈值要在验证协议中选择，不能用测试集反复试。

**极端 logit 触发数值问题。** 直接计算 $e^{-z}$、$\log(1-p)$ 或 $1-p$ 可能在尾部溢出或损失有效位。优先使用框架的稳定 logits 损失，并用 $z=\pm20,\pm1000$ 做单元测试。

## 一个可复用的核验协议

遇到阶跃、sigmoid 或任何二分类输出头，可以按下面顺序检查：

1. 先确认预激活的定义、偏置和符号，写出边界 $\mathbf{w}^{\mathsf T}\mathbf{x}+b=0$。
2. 对阶跃检查 $z<0$、$z=0$、$z>0$ 三个位置，确认边界约定。
3. 对 sigmoid 检查 $\sigma(0)=0.5$、$\sigma(-z)=1-\sigma(z)$ 和 $\sigma'(z)=\sigma(z)(1-\sigma(z))$。
4. 用默认阈值验证 $\sigma(z)\geq0.5$ 与 $z\geq0$ 是否一致；若阈值改变，换算为 logit 阈值。
5. 写清输出是 logit、概率还是硬标签，并让损失函数与它匹配。
6. 用至少一个正确样本和一个自信错分样本检查 BCE 的梯度方向，确认梯度为 $p-y$。
7. 用 $z=\pm20,\pm1000$ 检查实现不会产生 NaN、无穷或错误的全零梯度。
8. 若输出要参与行动，单独在验证数据上做校准与阈值选择，再在封存测试集上报告最终结果。

这套检查把三个容易混在一起的对象分开：函数形状、概率解释和行动阈值。阶跃负责硬决策，sigmoid 提供平滑分数，损失负责训练信号，阈值负责把分数变成具体动作。

## 相关词条

- [什么是神经元](../neurons-and-activations/what-is-a-neuron/)：预激活、偏置和激活函数的位置
- [为什么需要非线性](../neurons-and-activations/why-non-linearity/)：阶跃与 sigmoid 所在的非线性层为何不可缺
- [激活函数](../neurons-and-activations/activation-functions/)：激活函数的比较坐标与候选总览
- [经典感知机](../linear-models/perceptron-classic/)：不用导数也能训练硬阈值的错分更新
- [逻辑回归](../linear-models/logistic-regression/)：logit 线性、似然和正则化
- [tanh 激活](../neurons-and-activations/tanh/)：零中心的 sigmoid 变体
- [饱和与梯度消失](../neurons-and-activations/saturation-and-vanishing/)：尾部小导数如何沿深度累积
- [Softmax](../neurons-and-activations/softmax/)：从二分类 sigmoid 走向多分类概率
- [分类指标](../evaluation-and-generalization/classification-metrics/)：阈值改变后如何报告混淆矩阵与指标
