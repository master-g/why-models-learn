---
title: "激活函数：让每一层改变表示的形状"
tags: ["why-models-learn"]
---

激活函数是逐分量作用在神经元预激活值上的函数：先做仿射变换 $z=W\mathbf{x}+\mathbf{b}$，再得到 $\mathbf{a}=f(\mathbf{z})$。它的职责不是把数字“变得像生物神经元”，而是给层与层之间加入非线性，并决定信号的范围、中心、饱和方式和反向梯度。本文先建立一套比较激活函数的坐标，再用数值例子说明 sigmoid、tanh、ReLU 及其平滑变体各自改变了什么；阶跃函数、sigmoid、tanh、ReLU、GELU、Swish 和 softmax 的完整专题推导分别留在后续词条。

![激活函数从预激活值经过不同形状产生隐藏表示，输出头按任务选择回归或分类的映射](/assets/neurons-and-activations/svg/activation-functions.1.svg)

## 激活函数插在仿射层之后

一个神经元先计算

$$
z=\mathbf{w}^{\mathsf T}\mathbf{x}+b,
$$

再计算

$$
a=f(z).
$$

一层神经元写成向量形式：

$$
\mathbf{z}=W\mathbf{x}+\mathbf{b},
\qquad
\mathbf{a}=f(\mathbf{z}),
$$

其中 $f$ 默认逐分量作用：

$$
\mathbf{a}
=\bigl(f(z_1),f(z_2),\ldots,f(z_m)\bigr).
$$

这和 softmax 不同。逐分量激活分别看每个坐标；softmax 同时看所有 logit，输出分量之间相互耦合。后者是向量值的概率归一化，不是把一个标量函数简单复制到每个坐标上。

若连续两层没有激活：

$$
\mathbf{a}
=W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2
=(W_2W_1)\mathbf{x}+
\left(W_2\mathbf{b}_1+\mathbf{b}_2\right),
$$

仍然只是一个仿射变换。[为什么需要非线性](../neurons-and-activations/why-non-linearity/)用 XOR 和空间折叠证明了它为什么表达不足；这里关注的是另一个问题：既然必须加入 $f$，什么形状会让信号更容易训练、更容易解释。

## 一个导数把前向和反向连起来

设单个样本的平方损失为

$$
L=\frac12\left(f(z)-y\right)^2,
\qquad
z=\mathbf{w}^{\mathsf T}\mathbf{x}+b.
$$

对 $z$ 求导，链式法则给出

$$
\frac{\partial L}{\partial z}
=\left(f(z)-y\right)f'(z).
$$

对权重和偏置继续求导：

$$
\frac{\partial L}{\partial\mathbf{w}}
=\frac{\partial L}{\partial z}\mathbf{x},
\qquad
\frac{\partial L}{\partial b}
=\frac{\partial L}{\partial z}.
$$

因此激活函数同时决定三个量：

1. 预激活值落到哪里；
2. 输出误差如何被缩放；
3. 误差信号能否沿着网络传回去。

### 同一个预激活值的四种结果

取一组具体参数：

$$
\mathbf{x}=(2,-1),
\qquad
\mathbf{w}=(0.5,1),
\qquad
b=-0.5,
\qquad
y=0.25.
$$

于是

$$
z=0.5\times2+1\times(-1)-0.5=-0.5.
$$

对比几个标量激活：

| 激活 | $f(-0.5)$ | $f'(-0.5)$ | $\partial L/\partial z$ | 观察 |
| --- | ---: | ---: | ---: | --- |
| identity | $-0.500000$ | $1$ | $-0.750000$ | 不改变范围，适合回归输出 |
| sigmoid | $0.377541$ | $0.235004$ | $0.029973$ | 输出为正且梯度被压小 |
| tanh | $-0.462117$ | $0.786448$ | $-0.560043$ | 零中心，但负侧仍有梯度 |
| ReLU | $0$ | $0$ | $0$ | 负预激活被截断 |
| Leaky ReLU | $-0.005000$ | $0.01$ | $-0.002550$ | 负侧保留很小梯度 |

例如 sigmoid 行的误差项是 $0.377541-0.25\approx0.127541$，再乘以导数 $0.235004$，所以

$$
\frac{\partial L}{\partial z}
\approx0.127541\times0.235004
\approx0.029973.
$$

ReLU 行不是“算错了”：$z=-0.5$ 在它的负半轴上，输出和局部导数都为零。若某个神经元长期处在这个区域，它的梯度更新会停止；Leaky ReLU 通过人为保留斜率 $\alpha$ 改变了这一点。

## 比较激活函数的六个坐标

不能只问“哪个激活更强”。至少要同时看：

**是否非线性。** identity 的导数处处为 $1$，它不能让多层仿射变换获得新的函数形状。阶跃函数有非线性，但几乎处处导数为零，普通梯度下降无法直接训练它。

**输出范围。** 有界函数能限制输出，却可能在大绝对值区域饱和；无界函数可以传递更大的信号，却需要关注激活值和梯度的尺度。

**是否零中心。** sigmoid 输出在 $(0,1)$，隐藏层均值通常为正；tanh 输出在 $(-1,1)$，更容易让后续层看到正负两侧的变化。零中心不是所有输出头都需要的性质，概率输出本来就不以零为中心。

**导数在什么区域存在。** 平滑函数在数学上处处可导；ReLU 在 $0$ 处不可导，框架会选一个约定的次梯度。真正影响训练的往往不是一个点，而是大量样本是否落在零梯度或极小梯度区域。

**是否饱和。** 当 $f'(z)$ 在尾部接近零时，误差信号通过这一层会被压小。饱和既可能带来稳定的有界输出，也可能让深层网络难以更新。

**计算和硬件代价。** max、乘法和加法便宜；指数、正态分布函数和高精度特殊函数更贵。训练和推理是否能承受这个代价，要结合批量、设备和延迟约束判断。

这些坐标是选择标准，不是排名。输出范围、梯度尺度和损失函数要放在同一个训练协议里看。

## 常见标量激活的形状

### identity 和阶跃函数

identity 是

$$
f(z)=z,
\qquad
f'(z)=1.
$$

它不提供隐藏层所需的非线性，但在回归输出中很自然：模型可以直接预测任意实数，损失负责约束误差。若目标必须为正，可以考虑 softplus 或在业务层做明确的变换，而不是把所有回归问题都套 sigmoid。

阶跃函数可以写成

$$
f(z)=\mathbf{1}\{z\geq0\}.
$$

它适合表达硬判定，但在除跳点外的区域导数为零，在跳点又不可导。因此它是感知机和离散决策的重要概念，不是现代深度网络中直接用梯度训练的默认隐藏激活。[经典感知机](../linear-models/perceptron-classic/)讨论了它与误分类更新的关系。

### sigmoid

sigmoid 定义为

$$
\sigma(z)=\frac{1}{1+e^{-z}},
$$

它的导数可以只用输出表示：

$$
\sigma'(z)=\sigma(z)\left(1-\sigma(z)\right).
$$

输出落在 $(0,1)$，所以它适合把单个 logit 映射为二分类概率。它的最大导数是 $1/4$，并且当 $z$ 很正或很负时导数都趋近于零。把 sigmoid 放在深层隐藏单元中，会让多层导数相乘；把它放在二分类输出端，则要注意概率语义和数值稳定性。[阶跃与 sigmoid](../neurons-and-activations/heaviside-and-sigmoid/)会推导它的形状、对数几率和饱和。

### tanh

tanh 定义为

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}},
\qquad
\frac{\mathrm d}{\mathrm dz}\tanh(z)=1-\tanh^2(z).
$$

它的范围是 $(-1,1)$，并且是奇函数：

$$
\tanh(-z)=-\tanh(z).
$$

这让它在隐藏层中保持零中心输出。它仍会在大绝对值处饱和，导数回到零；零中心不等于没有梯度问题。[tanh](../neurons-and-activations/tanh/)单独处理它与 sigmoid 的关系、导数和饱和区。

### ReLU 和带负侧斜率的变体

ReLU 定义为

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

在正半轴它传递输入，在负半轴它输出零。它的优点是正侧不饱和、计算简单，并且会产生一部分精确为零的隐藏表示；代价是负侧梯度为零。

Leaky ReLU 用一个小的正斜率 $\alpha$ 保留负侧：

$$
\operatorname{LeakyReLU}_{\alpha}(z)
=\max(\alpha z,z),
\qquad
\alpha>0.
$$

当 $z<0$ 时，它的导数是 $\alpha$，因此不会因为一次负预激活就完全切断梯度。$\alpha$ 可以是固定超参数，也可以被设计成可学习参数；一旦让它学习，参数和正则化边界就需要单独说明。[ReLU](../neurons-and-activations/relu/)与[死亡 ReLU 与 Leaky 变体](../neurons-and-activations/dead-relu-and-leaky/)会进一步展开死区和初始化的影响。

### softplus、GELU 和 Swish

softplus 是 ReLU 的平滑近似：

$$
\operatorname{softplus}(z)=\log(1+e^z),
\qquad
\operatorname{softplus}'(z)=\sigma(z).
$$

直接计算 $\log(1+e^z)$ 在 $z$ 很大时可能溢出，实际实现会使用稳定形式。它没有 ReLU 的尖角，也没有完全为零的负侧梯度，但尾部仍可能非常小。

GELU 用标准正态分布函数 $\Phi$ 对输入做软门控：

$$
\operatorname{GELU}(z)=z\Phi(z).
$$

它不是简单的截断：负值通常被保留为小幅输出，正值大多被放行。若用精确的 $\Phi$，导数为

$$
\operatorname{GELU}'(z)
=\Phi(z)+z\phi(z),
$$

其中 $\phi$ 是标准正态密度。[GELU](../neurons-and-activations/gelu/)会说明精确形式、近似形式以及 Transformer 中的使用边界。

Swish 是可调的平滑门控：

$$
\operatorname{Swish}_{\beta}(z)
=z\sigma(\beta z).
$$

对 $z$ 求导：

$$
\operatorname{Swish}_{\beta}'(z)
=\sigma(\beta z)
+\beta z\sigma(\beta z)
\left(1-\sigma(\beta z)\right).
$$

$\beta$ 越大，门控越接近硬切换；$\beta$ 较小时，形状更平滑。[Swish 与 SwiGLU](../neurons-and-activations/swish-and-swiglu/)会把这个标量门与门控线性单元放到 Transformer 结构里比较。

## 一组输入上的数值形状

下面取 $z\in\{-2,-0.5,0,0.5,2\}$，看四个常见候选的输出：

| $z$ | sigmoid | tanh | ReLU | GELU |
| ---: | ---: | ---: | ---: | ---: |
| $-2$ | $0.119203$ | $-0.964028$ | $0$ | $-0.045500$ |
| $-0.5$ | $0.377541$ | $-0.462117$ | $0$ | $-0.154269$ |
| $0$ | $0.500000$ | $0$ | $0$ | $0$ |
| $0.5$ | $0.622459$ | $0.462117$ | $0.5$ | $0.345731$ |
| $2$ | $0.880797$ | $0.964028$ | $2$ | $1.954500$ |

这张表能看出三个差异：

1. sigmoid 把所有输入压到正区间；
2. tanh 保留正负方向，但两端趋近于常数；
3. ReLU 保留正侧幅度，GELU 则在零点附近平滑地决定放行多少。

在 $z=0.5$ 附近，ReLU 的输出已经是 $0.5$，GELU 只放行约 $0.345731$；在 $z=-0.5$，ReLU 完全截断，而 GELU 仍输出约 $-0.154269$。这些差异会改变下一层看到的分布，不能只在单个神经元的曲线图上做结论。

## 饱和、死区和梯度乘积

深层网络中的一条路径会累积多层导数。抽象地写，若某条路径上的权重和激活导数分别为 $W_1,\ldots,W_L$ 与 $f_1'(z_1),\ldots,f_L'(z_L)$，梯度包含类似的乘积：

$$
\frac{\partial L}{\partial \mathbf{x}}
\sim
\left(
W_1^{\mathsf T}D_1
W_2^{\mathsf T}D_2
\cdots
W_L^{\mathsf T}D_L
\right)
\frac{\partial L}{\partial \mathbf{a}_L},
$$

其中 $D_\ell$ 是由逐分量导数组成的对角矩阵。具体排列取决于网络约定，但结论不变：导数的大小和符号会沿深度重复出现。

sigmoid 的导数满足

$$
0<\sigma'(z)\leq\frac14.
$$

如果暂时忽略权重矩阵，只看 $L$ 个 sigmoid 导数，最宽松的乘积上界也是

$$
\left(\frac14\right)^L.
$$

这不是一个完整的梯度消失定理，因为权重、残差连接、归一化和数据分布都很重要；它只是说明为什么隐藏层使用 sigmoid 时要认真检查导数尺度。tanh 在零附近导数可以接近 $1$，但在尾部同样饱和。ReLU 正侧导数为 $1$，负侧导数为 $0$，所以它把“饱和尾部”换成了“死区风险”。

### ReLU 的死区不是所有零输出

ReLU 输出为零有两种容易混淆的情况：

1. 某个样本这一次的 $z<0$，它只是这一个样本在负侧；
2. 某个神经元对几乎所有训练样本都满足 $z<0$，它长期收不到梯度。

第一种是稀疏激活的正常来源，第二种才是常说的死亡 ReLU。学习率过大、偏置初始化过负、输入尺度变化或权重更新越过零点，都可能提高第二种情况的概率。诊断时应统计一段训练窗口内每个神经元的正侧比例，而不是看到一次零输出就判定死亡。

## 初始化会改变激活分布

设预激活 $Z$ 近似服从标准正态分布，并且分布关于零对称。对称性给出

$$
\mathbb E[\sigma(Z)]=\frac12,
\qquad
\mathbb E[\tanh(Z)]=0.
$$

ReLU 则只保留正半轴：

$$
\mathbb E[\operatorname{ReLU}(Z)]
=\frac{1}{\sqrt{2\pi}}
\approx0.398942,
$$

$$
\mathbb E[\operatorname{ReLU}(Z)^2]
=\frac12.
$$

所以即使 $Z$ 的均值为零，ReLU 输出的均值也会变成正数。层层传递后，激活均值、二阶矩和正侧比例可能发生漂移；初始化、归一化和残差结构都是在管理这个漂移。

这也是为什么“激活函数只决定曲线形状”不够完整。相同的 $f$ 放在不同权重尺度、偏置、归一化和数据分布后面，可能进入完全不同的工作区域。比较激活时要记录：

| 记录项 | 例子 | 它回答什么 |
| --- | --- | --- |
| 预激活均值和方差 | $\operatorname{mean}(z)$、$\operatorname{Var}(z)$ | 输入是否进入饱和或爆炸区域 |
| 正侧比例 | $\Pr(z>0)$ | ReLU 类是否大量处在死区 |
| 输出均值和方差 | $\operatorname{mean}(a)$、$\operatorname{Var}(a)$ | 下一层接收到的尺度 |
| 导数统计 | $\operatorname{mean}\lvert f'(z)\rvert$ | 反向信号是否被压小 |
| 分层轨迹 | 每层随 epoch 的曲线 | 问题发生在初始化还是训练途中 |

这些统计量比“这条激活在论文里更流行”更能解释一次具体训练为什么失败。

## 输出头不是隐藏激活的简单复制

隐藏层激活和输出层映射承担不同职责。输出头应由目标变量和损失函数共同决定：

| 任务 | 常见输出 | 训练时的注意点 |
| --- | --- | --- |
| 实数回归 | identity | 输出不应被 sigmoid 无故限制到 $(0,1)$ |
| 非负标量 | softplus 或显式正值参数化 | 记录零附近梯度和目标尺度 |
| 二分类概率 | sigmoid | 二元交叉熵优先直接接收 logit 的稳定实现 |
| 互斥多分类 | softmax | 多个类别共享归一化，概率和为 $1$ |
| 多标签分类 | 每类独立 sigmoid | 不应强制不同标签概率和为 $1$ |
| 计数或强度 | 依分布选择 log link、softplus 等 | 输出约束必须匹配似然和零值语义 |

二分类中，模型可以输出 logit $z$，把概率解释为 $\sigma(z)$，损失直接使用带 logits 的二元交叉熵。若先手动做 sigmoid，又把结果传给只接受 logit 的损失，数值稳定性和梯度都会被破坏。[逻辑回归](../linear-models/logistic-regression/)已经展示了 logit、概率和阈值的分离；这里的规则同样适用于神经网络最后一层。

多分类的 softmax 定义为

$$
\operatorname{softmax}(\mathbf{z})_i
=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}.
$$

它的分母同时读取所有类别，平移所有 logit 不改变输出：

$$
\operatorname{softmax}(\mathbf{z}+c\mathbf{1})
=\operatorname{softmax}(\mathbf{z}).
$$

因此 softmax 是向量头，不应和逐分量的 ReLU、GELU 混为一类。[Softmax](../neurons-and-activations/softmax/)会展开稳定计算、Jacobian 和温度。

## 怎样按任务选择隐藏激活

没有脱离任务和训练协议的总排名。可以按以下顺序缩小选择：

1. 先判断这一层是隐藏层还是输出头；
2. 明确输出范围、是否需要零中心和是否需要平滑导数；
3. 查看预激活分布会落在函数的哪一段；
4. 检查导数、正侧比例、输出均值方差和损失的数值稳定性；
5. 在同一数据切分、同一训练预算和同一随机种子协议下比较候选；
6. 同时报告质量、收敛速度、资源和失效样本，而不是只报最终损失。

一个可作为起点的选择表是：

| 场景 | 起点 | 需要特别检查 |
| --- | --- | --- |
| 普通前馈隐藏层 | ReLU 或其变体 | 死区、正侧比例、学习率 |
| 需要平滑门控 | GELU 或 Swish | 特殊函数代价和激活尺度 |
| 需要零中心且范围有限 | tanh | 大幅值输入的饱和 |
| 二分类输出 | sigmoid 或直接输出 logit | 损失是否重复套 sigmoid |
| 多分类输出 | softmax 或直接输出 logits | 类别耦合和数值稳定性 |
| 回归输出 | identity | 目标尺度、异常值和损失 |

“默认使用 ReLU”是一条工程起点，不是验证结论。网络深度、归一化、残差连接、初始化、数据尺度和优化器改变后，最合适的激活也可能改变。

## 失效模式

**把激活函数当成可忽略的装饰。** 没有隐藏非线性，多层仿射层仍然会合并成一层。先确认网络的非线性确实位于层与层之间。

**所有层都使用 sigmoid。** 输出范围看起来整齐，但深层导数容易在尾部变小。应观察预激活分布和每层导数，不要只看损失是否下降。

**看到 ReLU 输出为零就报告死亡。** 单个样本的负侧是正常稀疏性。要在多个 batch 和多个 epoch 上统计每个神经元的正侧比例。

**把零中心当成唯一标准。** sigmoid 的非零中心可能正是概率输出所需要的；tanh 的零中心也不能消除饱和。先写清楚输出语义和梯度工作区间。

**手动 sigmoid 后再次传给 logits 损失。** 这会重复做概率变换，破坏稳定实现。统一记录模型输出是 logit 还是概率，并让损失的接口匹配。

**只比较函数图，不比较训练协议。** 换激活时同时换了初始化、学习率、归一化和训练轮数，最后差异无法归因。先固定协议，再做单因素比较。

**把特殊函数的近似当成相同函数。** GELU、Swish 和 softplus 常有近似实现；论文、框架和部署端可能使用不同近似。记录精确式、近似式和误差容忍度。

**忽略输出头的任务约束。** 用 sigmoid 做无界回归、用 softmax 做多标签分类、用 identity 输出未经约束的方差，都会让模型目标和决策语义错位。

## 一个可复用的激活核验协议

改动激活函数时，可以固定一组小规模检查：

1. 记录每层预激活的均值、方差、分位数和正侧比例；
2. 记录每层输出的均值、方差、零比例和导数绝对值；
3. 用同一批输入检查几个代表性 $z$ 的前向值和反向值；
4. 确认输出头与损失函数的接口是 logit、概率还是受约束实数；
5. 固定数据切分、初始化协议、训练预算和随机种子再比较候选；
6. 对异常 loss、NaN、梯度全零和梯度爆炸保存第一层出现的位置；
7. 在部署精度、延迟和内存约束下重复检查，不只看训练集曲线；
8. 报告最终激活选择、版本、近似形式、超参数和未覆盖的输入范围。

这份协议把“换一个激活试试”变成可比较的实验：曲线只是入口，真正需要验证的是信号是否经过合理的工作区间，并且损失、输出头和部署动作是否仍然匹配。

## 相关词条

- [什么是神经元](../neurons-and-activations/what-is-a-neuron/)：把仿射计算和激活放回单个神经元。
- [为什么需要非线性](../neurons-and-activations/why-non-linearity/)：用 XOR 说明非线性对表达能力的必要性。
- [阶跃与 sigmoid](../neurons-and-activations/heaviside-and-sigmoid/)：推导阶跃和 sigmoid 的形状、概率与梯度。
- [tanh](../neurons-and-activations/tanh/)：展开零中心饱和激活。
- [ReLU](../neurons-and-activations/relu/)：展开分段线性激活。
- [死亡 ReLU 与 Leaky 变体](../neurons-and-activations/dead-relu-and-leaky/)：检查死区和负侧斜率。
- [GELU](../neurons-and-activations/gelu/)：推导正态门控和近似形式。
- [Swish 与 SwiGLU](../neurons-and-activations/swish-and-swiglu/)：连接平滑门控和 Transformer 结构。
- [饱和与梯度消失](../neurons-and-activations/saturation-and-vanishing/)：从导数乘积分析深层训练的信号衰减。
- [Softmax](../neurons-and-activations/softmax/)：处理向量耦合的多分类输出头。
- [逻辑回归](../linear-models/logistic-regression/)：区分 logit、概率和决策阈值。
- [万有逼近](../neurons-and-activations/universal-approximation/)：讨论非线性网络能表达的函数范围。
