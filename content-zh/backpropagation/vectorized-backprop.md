---
title: "向量化反向传播：把逐样本链式法则变成矩阵乘法"
tags: ["why-models-learn"]
---

向量化反向传播是把同一张计算图在一批样本上的逐样本反向传播，改写成矩阵运算的过程。它不改变模型、损失或链式法则，只改变计算和累加的组织方式：样本轴保留在张量中，许多个相同的标量运算合并为矩阵乘法、逐分量乘法和沿轴归约。这样，反向传播既能保持每个样本的局部梯度，又能把工作交给擅长批量线性代数的硬件。

本文固定“每个样本一行”的 row-batch 约定，从一层仿射变换和逐分量激活开始，用微分与 Frobenius 内积逐项推导四个反向公式。随后用一个不在 ReLU 边界上的数值例子核对形状、前向值和梯度，再说明广播的反向归约、batch 平均、深层网络、列向量布局、高阶张量和实现中的常见错位。

![向量化反向传播把整批样本的前向矩阵与反向矩阵连接起来](/assets/backpropagation/svg/vectorized-backprop.1.svg)

## 向量化改变的是执行形状

先看不使用向量化时的伪代码。对第 $i$ 个样本，仿射层会计算

$$
z_i=W x_i+b,
\qquad
a_i=\phi(z_i).
$$

如果这一批有 $n$ 个样本，直接写成循环就是重复执行 $n$ 次相同的程序。权重 $W$ 和偏置 $b$ 在样本之间共享；因此反向时，每个样本都会产生一份对同一组参数的贡献，最后必须相加：

$$
\frac{\partial L}{\partial W}
=\sum_{i=1}^{n}\frac{\partial L_i}{\partial W},
\qquad
\frac{\partial L}{\partial b}
=\sum_{i=1}^{n}\frac{\partial L_i}{\partial b}.
$$

向量化只是把这组循环一次性写成矩阵。它保留了样本之间的独立性和参数共享关系，也保留了反向图中的加法。若把样本误当成参数，或者把逐样本损失的平均误写成求和，得到的就不再是同一个目标。

## 先固定 row-batch 记号

设一批有 $n$ 个样本，输入维度为 $d$，输出维度为 $h$。每个样本是一行：

$$
X\in\mathbb R^{n\times d},
\qquad
W\in\mathbb R^{h\times d},
\qquad
b\in\mathbb R^h,
\qquad
\boldsymbol 1\in\mathbb R^n.
$$

$\boldsymbol1$ 是全为 $1$ 的列向量。把每个样本的仿射结果堆成行，得到

$$
Z=XW^{\mathsf T}+\boldsymbol1 b^{\mathsf T},
\qquad
Z\in\mathbb R^{n\times h}.
$$

偏置项的形状是 $n\times h$：广播把同一个 $b$ 复制到每个样本行。逐分量激活得到

$$
A=\phi(Z),
\qquad
A\in\mathbb R^{n\times h}.
$$

本篇用 $G_U$ 表示损失对 $U$ 的梯度：

$$
G_U:=\frac{\partial L}{\partial U}.
$$

因此 $G_A$、$G_Z$ 与 $A$、$Z$ 同形状，分别是 $n\times h$。矩阵梯度必须和它所对应的变量同形状，这是排查转置错误的第一条规则。

## 一层反向公式

### 先穿过逐分量激活

逐分量激活在每个坐标上都是一元函数。对第 $i$ 个样本和第 $j$ 个输出坐标：

$$
Z_{ij}\longmapsto A_{ij}=\phi(Z_{ij}).
$$

局部链式法则给出

$$
(G_Z)_{ij}=(G_A)_{ij}\phi'(Z_{ij}).
$$

把所有坐标同时写出，就是 Hadamard 积：

$$
G_Z=G_A\odot\phi'(Z).
$$

ReLU 的导数在非零点可以写成掩码 $M_{ij}=\mathbf1_{Z_{ij}>0}$，于是 $G_Z=G_A\odot M$。在 $Z_{ij}=0$ 的不可导点，具体实现必须采用已经约定的次梯度或边界规则，不能假装普通导数在那里唯一存在。

### 再穿过仿射层

仿射层的三个梯度是

$$
\begin{aligned}
G_X&=G_ZW,\\
G_W&=G_Z^{\mathsf T}X,\\
g_b&=G_Z^{\mathsf T}\boldsymbol1.
\end{aligned}
$$

逐个检查形状：

| 量 | 形状 | 计算来源 |
| --- | --- | --- |
| $X$ | $n\times d$ | 每个样本一行 |
| $W$ | $h\times d$ | 每个输出单元一行 |
| $Z,A,G_A,G_Z$ | $n\times h$ | 每个样本、每个输出坐标 |
| $G_X$ | $n\times d$ | $G_ZW$ |
| $G_W$ | $h\times d$ | $G_Z^{\mathsf T}X$ |
| $g_b$ | $h$ | 沿样本轴求和 |

$G_X$ 是把敏感度传回输入；$G_W$ 是所有样本的外积贡献之和；$g_b$ 是所有样本对同一偏置向量的贡献之和。这三个结果必须分别对齐 $X$、$W$ 和 $b$，不能因为某个矩阵乘法“能算出来”就认为转置方向正确。

## 用微分推导转置位置

形状记忆容易在多层网络或换布局后失效。更稳妥的做法是从微分开始。对

$$
Z=XW^{\mathsf T}+\boldsymbol1 b^{\mathsf T}
$$

取微分：

$$
dZ=dXW^{\mathsf T}+XdW^{\mathsf T}+\boldsymbol1 db^{\mathsf T}.
$$

用 Frobenius 内积定义上游梯度：

$$
dL=\langle G_Z,dZ\rangle_F
=\operatorname{tr}(G_Z^{\mathsf T}dZ).
$$

把三项分别代入。第一项为

$$
\operatorname{tr}(G_Z^{\mathsf T}dXW^{\mathsf T})
=\operatorname{tr}(W^{\mathsf T}G_Z^{\mathsf T}dX)
=\operatorname{tr}((G_ZW)^{\mathsf T}dX).
$$

矩阵梯度的定义是 $dL=\operatorname{tr}(G_X^{\mathsf T}dX)$，所以

$$
G_X=G_ZW.
$$

第二项利用 $\operatorname{tr}(M dW^{\mathsf T})=\operatorname{tr}(M^{\mathsf T}dW)$：

$$
\operatorname{tr}(G_Z^{\mathsf T}XdW^{\mathsf T})
=\operatorname{tr}((G_Z^{\mathsf T}X)^{\mathsf T}dW),
$$

因此

$$
G_W=G_Z^{\mathsf T}X.
$$

第三项是

$$
\operatorname{tr}(G_Z^{\mathsf T}\boldsymbol1 db^{\mathsf T})
=\operatorname{tr}((G_Z^{\mathsf T}\boldsymbol1)^{\mathsf T}db),
$$

从而

$$
g_b=G_Z^{\mathsf T}\boldsymbol1.
$$

这个推导揭示了三个公式的共同来源：矩阵乘法的一个因子被固定为微分，另一个因子沿迹循环到梯度的一侧。偏置没有独立的样本副本，所以它的微分只出现一次；广播的反向效果恰好就是沿复制轴求和。

## 数值例子：一批三行的完整反向

为了把公式和实际数字对上，取一个仿射层后接 ReLU。故意选择所有预激活都不等于 $0$，这样不会把边界次梯度混进核对。

$$
X=
\begin{pmatrix}
1&2\\
-1&0\\
0&1
\end{pmatrix},
\qquad
W=
\begin{pmatrix}
1&-0.5\\
0.5&1
\end{pmatrix},
\qquad
b=\begin{pmatrix}0.2\\-0.1\end{pmatrix}.
$$

### 前向矩阵

先算 $XW^{\mathsf T}$，再把偏置广播到三行：

$$
Z=XW^{\mathsf T}+\boldsymbol1b^{\mathsf T}
=
\begin{pmatrix}
0.2&2.4\\
-0.8&-0.6\\
-0.3&0.9
\end{pmatrix}.
$$

ReLU 只保留正数：

$$
A=\operatorname{ReLU}(Z)
=
\begin{pmatrix}
0.2&2.4\\
0&0\\
0&0.9
\end{pmatrix}.
$$

假设上游传来

$$
G_A=
\begin{pmatrix}
1&-2\\
0.5&0.3\\
-1&4
\end{pmatrix}.
$$

这代表后续图对 $A$ 的敏感度。为了得到一个可独立核对的标量，可以把本例理解为线性目标

$$
L=\langle G_A,A\rangle_F.
$$

它不必是训练中常见的最终损失；它的作用是给定 $G_A$ 后检验本层的反向规则。

### 反向矩阵

ReLU 掩码为

$$
M=\begin{pmatrix}
1&1\\
0&0\\
0&1
\end{pmatrix},
$$

所以

$$
G_Z=G_A\odot M
=
\begin{pmatrix}
1&-2\\
0&0\\
0&4
\end{pmatrix}.
$$

输入梯度为

$$
G_X=G_ZW
=
\begin{pmatrix}
0&-2.5\\
0&0\\
2&4
\end{pmatrix}.
$$

权重梯度为

$$
G_W=G_Z^{\mathsf T}X
=
\begin{pmatrix}
1&2\\
-2&0
\end{pmatrix}.
$$

偏置梯度是沿行相加：

$$
g_b=G_Z^{\mathsf T}\boldsymbol1
=
\begin{pmatrix}1\\2\end{pmatrix}.
$$

最后，线性目标的值为

$$
L=1\cdot0.2-2\cdot2.4+4\cdot0.9=-1.
$$

这里最值得观察的是第二行：它的 $Z$ 全为负，所以尽管 $G_A$ 第二行不为零，ReLU 仍把这行的 $G_Z$ 全部截成零。另一个关键点是 $g_b$：第一输出坐标收到 $1+0+0=1$，第二输出坐标收到 $-2+0+4=2$，不是只取最后一行，也不是把偏置梯度保留成 $3\times2$ 的复制矩阵。

### 用有限差分复核矩阵梯度

对任意一个参数坐标 $\theta$，用中心差分估计导数：

$$
\frac{\partial L}{\partial\theta}
\approx
\frac{L(\theta+\varepsilon)-L(\theta-\varepsilon)}{2\varepsilon}.
$$

取 $\varepsilon=10^{-5}$，对 $W$ 的四个坐标和 $b$ 的两个坐标分别扰动。由于本例的每个 $Z_{ij}$ 离 $0$ 至少有 $0.2$，扰动不会跨过 ReLU 边界；中心差分得到的六个结果与上面的 $G_W$、$g_b$ 一致，最大绝对误差约为 $10^{-11}$ 量级。这个核对同时验证了前向矩阵、ReLU 掩码、权重外积累加和偏置归约，而不只验证某一个乘法。

## 广播的反向就是归约

偏置公式最清楚地展示了广播导数。前向把一行 $b^{\mathsf T}$ 复制到每个样本：

$$
\boldsymbol1b^{\mathsf T}
=
\begin{pmatrix}
b^{\mathsf T}\\
\vdots\\
b^{\mathsf T}
\end{pmatrix}.
$$

反向必须把所有复制位置对同一个 $b$ 的敏感度加回来：

$$
g_b=\operatorname{sum}_{\text{样本轴}}(G_Z).
$$

更一般地，广播了一个低维张量，就沿着被复制的轴做求和。这个规则适用于 batch、时间步、空间位置和注意力头；区别只在于要明确到底哪些轴共享参数。

### sum 与 mean 不是同一个梯度尺度

设每个样本有损失 $\ell_i$。如果目标是求和：

$$
L_{\mathrm{sum}}=\sum_{i=1}^{n}\ell_i,
$$

那么每个样本的上游梯度直接进入 $G_Z$，偏置梯度随样本数增长。如果目标是平均：

$$
L_{\mathrm{mean}}=\frac1n\sum_{i=1}^{n}\ell_i,
$$

那么整个样本梯度通常多一个 $1/n$：

$$
G_Z^{\mathrm{mean}}=\frac1nG_Z^{\mathrm{sum}},
\qquad
g_b^{\mathrm{mean}}=\frac1n(G_Z^{\mathrm{sum}})^{\mathsf T}\boldsymbol1.
$$

把前向 loss 改成 mean 却继续使用 sum 梯度，会让有效学习率随 batch size 变化；反过来，把 sum 梯度再除一次 $n$，则会把梯度缩小两遍。实现中必须记录 reduction 发生的位置：它可能在逐样本损失之后，也可能在 loss head 的反向初始化时。

### 样本权重与 mask

若样本权重为 $q\in\mathbb R^n$，逐行乘到上游梯度：

$$
G_Z^{(q)}=\operatorname{diag}(q)G_Z.
$$

不需要真的构造对角矩阵；实际实现只需把第 $i$ 行乘以 $q_i$。如果权重目标是归一化加权平均，则还要除以权重和：

$$
L_q=\frac{\sum_iq_i\ell_i}{\sum_iq_i},
\qquad
G_Z^{(q)}=\frac{1}{\sum_iq_i}\operatorname{diag}(q)G_Z.
$$

padding mask、忽略标签和有效 token mask 都遵循同一个方向：先在损失归约处决定哪些位置有贡献，再把对应的零或权重传回 $G_Z$。如果只把前向 loss 的 masked 项置零，却忘记反向 mask，模型仍会从被忽略位置学习。

## 多层网络的向量化递推

向量化不会只适用于一层。对第 $l$ 层，令 $H^0=X$，并继续使用每个样本一行：

$$
Z^l=H^{l-1}(W^l)^{\mathsf T}+\boldsymbol1(b^l)^{\mathsf T},
\qquad
H^l=\phi_l(Z^l).
$$

若输出头已经产生了 $G_{H^l}$，则本层反向按同样的三步执行：

$$
\begin{aligned}
G_{Z^l}&=G_{H^l}\odot\phi_l'(Z^l),\\
G_{H^{l-1}}&=G_{Z^l}W^l,\\
G_{W^l}&=(G_{Z^l})^{\mathsf T}H^{l-1},\\
g_{b^l}&=(G_{Z^l})^{\mathsf T}\boldsymbol1.
\end{aligned}
$$

然后把 $G_{H^{l-1}}$ 交给前一层。输出层的 sigmoid 加二元交叉熵、softmax 加交叉熵、线性回归头虽然有不同的 $G_{Z^L}$ 初始化公式，一旦进入隐藏层，矩阵反向规则完全相同。

这也说明了“逐层误差信号”的含义：$G_{Z^l}$ 仍然保留每个样本和每个单元的位置；$G_{W^l}$ 和 $g_{b^l}$ 才在共享参数处沿样本轴汇合。向量化不是把所有样本压成一个标量梯度，而是在正确的轴上延迟汇合。

## 行布局、列布局与高阶张量

### 换成列向量并不会改变链式法则

有些线性代数教材把样本放在列中。若 $X_c\in\mathbb R^{d\times n}$，则

$$
Z_c=WX_c+b\boldsymbol1^{\mathsf T},
\qquad
Z_c\in\mathbb R^{h\times n}.
$$

令 $G_{Z_c}$ 与 $Z_c$ 同形状，反向公式变为

$$
G_{X_c}=W^{\mathsf T}G_{Z_c},
\qquad
G_W=G_{Z_c}X_c^{\mathsf T},
\qquad
g_b=G_{Z_c}\boldsymbol1.
$$

两组公式不是互相矛盾，而是同一个线性映射在不同数据布局下的表达。排查问题时不能只比较公式表面的转置；必须先问“样本在哪一维、权重的输出维在哪一维、广播沿哪一维发生”。

### 从二维 batch 到更高阶输入

序列输入常写成 $X\in\mathbb R^{B\times T\times d}$，图像或特征图也会有空间轴。对一个不混合位置的线性层，可以把前面的样本轴展平为 $BT$：

$$
X_{\mathrm{flat}}\in\mathbb R^{(BT)\times d},
\qquad
Z_{\mathrm{flat}}=X_{\mathrm{flat}}W^{\mathsf T}+\boldsymbol1b^{\mathsf T}.
$$

反向后把 $G_{X_{\mathrm{flat}}}$ 还原为 $B\times T\times d$。偏置梯度要沿 $B$ 和 $T$ 两个共享轴归约：

$$
g_b=\operatorname{sum}_{B,T}(G_Z).
$$

如果层本身在 token 之间做了注意力或卷积混合，就不能把它当作互不相干的 $BT$ 个样本；展平只是存储与 GEMM 的布局技巧，不能抹去算子真实连接的轴。padding、causal mask 和按 token 的 loss 也必须在各自的轴上保持一致。

## 为什么矩阵乘法适合反向传播

一层的主要工作是三次形如矩阵乘法的运算：

| 阶段 | 运算 | 直观含义 |
| --- | --- | --- |
| 前向 | $XW^{\mathsf T}$ | 所有样本同时做仿射变换 |
| 回传输入 | $G_ZW$ | 所有样本同时乘转置 Jacobian 的等价形式 |
| 累加权重 | $G_Z^{\mathsf T}X$ | 所有样本的外积贡献一次汇总 |

底层库常把这类密集矩阵乘法称为 GEMM。它可以利用连续内存、缓存、SIMD、GPU 线程块和专用矩阵单元；但“能调用 GEMM”不是正确性的证明。矩阵维度、数据布局、stride、dtype 和 reduction 轴仍然必须由计算图语义决定。

向量化通常减少解释器循环和小算子的调度开销，也让硬件有更大的计算块可处理。代价是中间矩阵可能更大：保存 $Z$ 和激活掩码会增加内存，激活检查点或算子融合则用额外重算换内存。性能优化应在公式已经通过独立梯度核对后进行，否则一个高效的转置错误只会更快地得到错误梯度。

## 一份可执行的审计顺序

遇到“向量化后训练不对”时，可以沿以下顺序逐层缩小范围：

| 检查 | 要确认的证据 | 常见错误 |
| --- | --- | --- |
| 1. 布局 | 样本轴、输出轴和权重轴的明确约定 | 行布局公式套到列布局 |
| 2. 前向 | $Z$ 的形状、一个样本的手算值和广播结果 | $W$ 与 $W^{\mathsf T}$ 混用 |
| 3. 局部导数 | $G_Z$ 是否等于 $G_A\odot\phi'(Z)$ | ReLU 掩码错位或使用了激活值的错误导数 |
| 4. 共享参数 | $G_W$、$g_b$ 是否沿正确轴汇总 | 把每个样本的权重梯度覆盖而非累加 |
| 5. reduction | sum、mean、mask、sample weight 的位置 | batch size 改变后学习率隐式改变 |
| 6. 数值 | 选取非边界坐标做中心差分 | 在 ReLU 边界处把次梯度差异误判为转置错误 |
| 7. 更新时序 | 所有梯度算完后才更新参数 | 先更新 $W$ 再用新 $W$ 回传 $X$ 或 $b$ |

先用一个仿射层隔离问题，再接激活，再接两层网络。每加一层都保留 $Z$、激活、掩码和梯度的形状记录。若只看最终 loss，转置错误、少除一次 batch size 和参数更新时序错误可能产生相似的曲线，难以定位。

## 最容易失败的地方

### 把样本轴当成参数轴

$G_W=G_Z^{\mathsf T}X$ 中的乘法正是在样本轴上做内积。若把 $XG_Z$ 当作权重梯度，形状有时仍能通过，但它表达的是另一种收缩。应把 $G_W$ 与 $W$ 的每个坐标用有限差分逐项对照。

### 偏置梯度忘记求和或多求一次

偏置在前向被广播，在反向必须沿所有被复制的轴求和。对 row-batch 的二维层是样本轴；对 $B\times T\times d$ 输入还可能包含时间轴。另一方面，如果上游已经含有 mean reduction，就不能在 bias backward 再秘密除一次 $n$。

### 用错激活掩码

ReLU 的掩码来自 $Z$ 的符号，不是来自 batch 行号，也不是来自另一个层的激活。缓存被原地覆盖、reshape 轴顺序改变或混用 dropout mask，都可能使 $G_Z$ 形状正确而数值错误。

### 用更新后的参数继续反向

一次反向传播应对应同一组前向参数和中间缓存。若先用 $G_W$ 更新了 $W$，再用新 $W$ 计算 $G_X$，就把两个不同参数点的局部导数拼在一起；这不是普通的同步梯度下降。

### 把向量化误当成独立样本完全相加

样本之间可能在 batch normalization、注意力、对比损失或跨样本负例中相互连接。此时不能简单套用“每行独立、只在参数处相加”的直觉，必须把跨样本算子也纳入计算图，并明确它的反向轴。

## 小结

在 row-batch 约定下，一层

$$
Z=XW^{\mathsf T}+\boldsymbol1b^{\mathsf T},
\qquad
A=\phi(Z)
$$

的反向传播只有一个局部激活公式和三个矩阵公式：

$$
G_Z=G_A\odot\phi'(Z),
\qquad
G_X=G_ZW,
\qquad
G_W=G_Z^{\mathsf T}X,
\qquad
g_b=G_Z^{\mathsf T}\boldsymbol1.
$$

它们分别表示局部导数、输入回传、共享权重的外积累加和广播偏置的反向归约。向量化的正确性不来自矩阵乘法看起来漂亮，而来自形状、共享轴、reduction 约定和有限差分证据同时对上。掌握这一层之后，多层 MLP 只是沿深度方向重复相同的递推；真正新增的风险，是布局、缓存和跨轴归约越来越多。

## 相关词条

- [反向传播：沿计算图把梯度累加回来](../backpropagation/)
- [计算图：把复合函数拆成可追踪的局部运算](../backpropagation/computational-graphs/)
- [前向传播：从输入到损失的值计算](../backpropagation/forward-pass/)
- [链式法则在计算图上的展开](../backpropagation/chain-rule-on-graphs/)
- [单神经元的反向传播](../backpropagation/backprop-single-neuron/)
- [两层网络的反向传播](../backpropagation/backprop-two-layer/)
- [广播与归约的导数](../calculus/broadcast-and-reduction-derivatives/)
- [矩阵微积分恒等式](../calculus/matrix-calculus-identities/)
- [梯度检查：用有限差分验证反向传播](../backpropagation/gradient-checking/)
