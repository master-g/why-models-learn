---
title: "低秩近似:只保留最重要的方向"
tags: ["why-models-learn"]
---

一个矩阵不一定要保留所有输入方向的作用。如果它的奇异值分解是

$$
\begin{aligned}
A&=\sum_{i=1}^r\sigma_i\mathbf{u}_i\mathbf{v}_i^{\mathsf T}\\
\sigma_1&\ge\sigma_2\ge\cdots\ge\sigma_r>0
\end{aligned}
$$

只取最大的 $k$ 个奇异值，就得到

$$
A_k=\sum_{i=1}^k\sigma_i\mathbf{u}_i\mathbf{v}_i^{\mathsf T}
$$

其中取 $0\le k\le r$。这叫**秩 $k$ 近似**（rank-$k$ approximation）。它保留矩阵在最强 $k$ 条正交通道上的作用，把其余方向当作误差。本篇先用精确小矩阵看见截断发生了什么，再证明 $A_k$ 在谱范数和 Frobenius 范数下分别达到最小误差，最后说明存储、降噪、PCA 和低秩参数化中的边界。

## 秩就是还剩多少条独立通道

在 SVD 中，非零奇异值的数量就是秩：

$$
\operatorname{rank}(A)=r
\quad\Longleftrightarrow\quad
\sigma_1,\dots,\sigma_r>0，
\quad
\sigma_{r+1}=\cdots=0
$$

每一项

$$
\sigma_i\mathbf{u}_i\mathbf{v}_i^{\mathsf T}
$$

最多贡献一条输入到输出的通道。$\mathbf{v}_i$ 负责从输入中取出一个方向，$\mathbf{u}_i$ 负责把它送到输出方向，$\sigma_i$ 是增益。[奇异值分解](../linear-algebra/svd/)把这三个量的来源和形状讲清楚。

保留前 $k$ 项有三个直接结果：

$$
\operatorname{rank}(A_k)\le k，
\qquad
A_k\mathbf{v}_i=\sigma_i\mathbf{u}_i\quad(i\le k)，
\qquad
A_k\mathbf{v}_i=\mathbf{0}\quad(i>k)
$$

所以近似矩阵并不是任意删掉 $A$ 的几列或几行，而是删掉奇异坐标中的弱通道。

## 先算一个三方向例子

取最容易完全核对的矩阵

$$
A=
\begin{pmatrix}
5&0&0\\
0&2&0\\
0&0&1
\end{pmatrix}
$$

它已经处在奇异坐标中：$U=V=I$，奇异值就是 $5$、$2$、$1$。外积展开是

$$
A=5\mathbf{e}_1\mathbf{e}_1^{\mathsf T}
 +2\mathbf{e}_2\mathbf{e}_2^{\mathsf T}
 +1\mathbf{e}_3\mathbf{e}_3^{\mathsf T}
$$

若只保留第一条通道，

$$
A_1=
\begin{pmatrix}
5&0&0\\
0&0&0\\
0&0&0
\end{pmatrix}，
\qquad
A-A_1=
\begin{pmatrix}
0&0&0\\
0&2&0\\
0&0&1
\end{pmatrix}
$$

第二、第三方向没有被近似矩阵处理。若保留前两条，

$$
A_2=
\begin{pmatrix}
5&0&0\\
0&2&0\\
0&0&0
\end{pmatrix}，
\qquad
A-A_2=
\begin{pmatrix}
0&0&0\\
0&0&0\\
0&0&1
\end{pmatrix}
$$

把误差逐项算出来：

| 近似 | 保留的奇异值 | 残差 $A-A_k$ 的 Frobenius 范数 | 残差的谱范数 |
| --- | --- | --- | --- |
| $A_0=0$ | 无 | $\sqrt{5^2+2^2+1^2}=\sqrt{30}$ | $5$ |
| $A_1$ | $5$ | $\sqrt{2^2+1^2}=\sqrt5$ | $2$ |
| $A_2$ | $5、2$ | $1$ | $1$ |
| $A_3=A$ | $5、2、1$ | $0$ | $0$ |

Frobenius 范数把所有元素的平方加起来再开方：

$$
\lVert M\rVert_F^2=\sum_{i,j}M_{ij}^2
$$

谱范数只看最大长度放大：

$$
\lVert M\rVert_2
=\max_{\mathbf{x}\ne\mathbf{0}}
\frac{\lVert M\mathbf{x}\rVert}{\lVert\mathbf{x}\rVert}
$$

在这个例子里，丢掉 $2$ 和 $1$ 两条通道后，Frobenius 误差的平方是 $4+1=5$，谱范数误差只由最大的被丢弃奇异值 $2$ 决定。保留的平方能量占比为

$$
\frac{5^2}{5^2+2^2+1^2}=\frac56
\quad(k=1)，
\qquad
\frac{5^2+2^2}{5^2+2^2+1^2}=\frac{29}{30}
\quad(k=2)
$$

「保留多少」必须先说明误差度量：谱范数关心最坏方向，Frobenius 范数关心全部元素的总平方误差。

![低秩近似保留最大的奇异值：奇异谱从 5、2、1 逐级下降，秩一近似只保留第一条通道](/assets/linear-algebra/svg/low-rank-approximation.1.svg)

## 截断为什么在谱范数下最优

结论是：若 $A$ 的奇异值按降序排列，那么对所有秩不超过 $k$ 的矩阵 $B$，

$$
\lVert A-A_k\rVert_2
=\sigma_{k+1}
\le
\lVert A-B\rVert_2
$$

先验证等号左边。写出残差：

$$
A-A_k
=\sum_{i=k+1}^r\sigma_i\mathbf{u}_i\mathbf{v}_i^{\mathsf T}
$$

对任意单位向量 $\mathbf{x}=\sum_i z_i\mathbf{v}_i$，

$$
\lVert(A-A_k)\mathbf{x}\rVert^2
=\sum_{i=k+1}^r\sigma_i^2z_i^2
\le\sigma_{k+1}^2\sum_{i=k+1}^r z_i^2
\le\sigma_{k+1}^2
$$

取 $\mathbf{x}=\mathbf{v}_{k+1}$ 时，等号成立，所以

$$
\lVert A-A_k\rVert_2=\sigma_{k+1}
$$

再看任意秩不超过 $k$ 的 $B$。令

$$
S=\operatorname{span}(\mathbf{v}_1,\dots,\mathbf{v}_{k+1})
$$

它的维数是 $k+1$。因为 $\operatorname{rank}(B)\le k$，秩-零度定理给出

$$
\dim\ker(B)\ge n-k
$$

于是 $S$ 与 $\ker(B)$ 不可能只有零向量交集：

$$
\dim S+\dim\ker(B)
\ge(k+1)+(n-k)=n+1>n
$$

因此可以找到单位向量 $\mathbf{x}\in S\cap\ker(B)$。它满足 $B\mathbf{x}=0$，而 $\mathbf{x}$ 只由前 $k+1$ 个右奇异向量组成，所以

$$
\lVert A\mathbf{x}\rVert^2
=\sum_{i=1}^{k+1}\sigma_i^2z_i^2
\ge\sigma_{k+1}^2\sum_{i=1}^{k+1}z_i^2
=\sigma_{k+1}^2
$$

从而

$$
\lVert A-B\rVert_2
\ge\lVert(A-B)\mathbf{x}\rVert
=\lVert A\mathbf{x}\rVert
\ge\sigma_{k+1}
$$

这说明任何秩 $k$ 矩阵都会丢掉某个由前 $k+1$ 个强方向组成的组合，那个方向的放大至少有 $\sigma_{k+1}$。截断 SVD 已经把最坏误差压到了这个下界。

## 截断为什么在 Frobenius 范数下也最优

Frobenius 范数的结论是

$$
\lVert A-A_k\rVert_F
=\sqrt{\sum_{i=k+1}^r\sigma_i^2}
\le
\lVert A-B\rVert_F
\qquad
(\operatorname{rank}(B)\le k)
$$

先给出下界。令 $P$ 是 $B$ 的行空间在输入空间中的正交投影。由于 $B$ 的每一行都在这个行空间中，

$$
B=BP，
\qquad
B(I-P)=0
$$

于是

$$
(A-B)(I-P)=A(I-P)
$$

右乘 $P$ 与右乘 $I-P$ 得到的两部分正交，因此

$$
\lVert A-B\rVert_F^2
\ge\lVert A(I-P)\rVert_F^2
$$

现在比较 $A(I-P)$ 的大小。因为

$$
\lVert A\rVert_F^2=\operatorname{tr}(A^{\mathsf T}A)=\sum_i\sigma_i^2
$$

并且 $P$ 是投影，

$$
\lVert AP\rVert_F^2
=\operatorname{tr}(P A^{\mathsf T}A)
$$

在右奇异向量基下令 $Q=V^{\mathsf T}PV$。$Q$ 仍是投影，所以其对角元满足 $0\le q_i\le1$，且

$$
\sum_iq_i=\operatorname{tr}(Q)=\operatorname{rank}(P)\le k
$$

因此

$$
\lVert AP\rVert_F^2
=\sum_i\sigma_i^2q_i
\le\sum_{i=1}^k\sigma_i^2
$$

因为要把总量不超过 $k$ 的权重放在降序排列的 $\sigma_i^2$ 上，最大值只能取前 $k$ 项。于是

$$
\begin{aligned}
\lVert A(I-P)\rVert_F^2
&=\lVert A\rVert_F^2-\lVert AP\rVert_F^2\\
&\ge\sum_{i=k+1}^r\sigma_i^2
\end{aligned}
$$

结合前面的不等式，得到所有秩不超过 $k$ 的 $B$ 都满足这个 Frobenius 下界。

取 $P=P_k$，其中

$$
P_k=\sum_{i=1}^k\mathbf{v}_i\mathbf{v}_i^{\mathsf T}
$$

则 $A_k=AP_k$，并且

$$
A(I-P_k)=\sum_{i=k+1}^r\sigma_i\mathbf{u}_i\mathbf{v}_i^{\mathsf T}
$$

不同外积项在 Frobenius 内积下正交，所以残差平方正好是

$$
\lVert A-A_k\rVert_F^2
=\sum_{i=k+1}^r\sigma_i^2
$$

下界被达到，定理得证。

## 存储量和计算量怎样变化

一个 $m\times n$ 的稠密矩阵需要存储 $mn$ 个数。秩 $k$ 的紧 SVD 存储

$$
U_k\in\mathbb{R}^{m\times k}，
\qquad
\Sigma_k\in\mathbb{R}^{k\times k}，
\qquad
V_k\in\mathbb{R}^{n\times k}
$$

共约 $k(m+n+1)$ 个数。只有在

$$
k(m+n+1)<mn
$$

时，因子化存储才比直接存矩阵少；如果 $k$ 接近 $\min(m,n)$，因子本身可能并不省空间。

应用近似矩阵也可以分三步做：

$$
A_k\mathbf{x}
=U_k\Sigma_k(V_k^{\mathsf T}\mathbf{x})
$$

先做 $V_k^{\mathsf T}\mathbf{x}$，再逐坐标乘奇异值，最后做 $U_k$。一次作用大约需要与 $k(m+n)$ 成正比的乘加，而不是稠密矩阵的 $mn$ 个乘加。这个估计只说明数量级，实际收益还取决于内存布局和硬件。

以 $1000\times1000$ 矩阵的秩 $10$ 近似为例，直接存储约 $10^6$ 个数，三个紧 SVD 因子约存储

$$
10(1000+1000+1)=20010
$$

个数。若误差允许，差异会很大；若原矩阵没有谱衰减，低秩近似就无法同时保持小误差和小存储。

## 数据降噪不是无条件删小奇异值

图像、词向量或样本矩阵常写成

$$
X=X_{\text{signal}}+X_{\text{noise}}
$$

如果有证据表明信号集中在前几个奇异方向，而噪声分散在许多较弱方向，取 $X_k$ 可能去掉一部分噪声。这是一个建模假设，不是 SVD 自动识别信号的定理。

反例很直接：如果真正有用的信号只出现在一个小奇异值方向，截断它会把信号一起删掉。选择 $k$ 时要在验证数据上比较任务误差，或使用噪声模型、交叉验证和保留能量阈值；只看奇异值曲线的拐点不保证下游任务最优。

PCA 是一个有明确目标的特例：对已经中心化的数据矩阵 $X$，取前 $k$ 个右奇异向量，得到在平方重构误差下最优的 $k$ 维线性子空间。协方差矩阵的特征值与 $X$ 的奇异值平方只差一个 $N-1$ 的归一化因子。

## 在机器学习中的低秩参数化

对一个权重矩阵 $W\in\mathbb{R}^{m\times n}$，直接把它替换为 $W_k$ 是事后压缩：先训练完整矩阵，再做 SVD 和截断。它减少推理存储，但需要检查截断后的精度。

LoRA 采用另一种参数化：冻结原权重 $W_0$，只学习

$$
\Delta W=BA，
\qquad
B\in\mathbb{R}^{m\times k}，
\quad
A\in\mathbb{R}^{k\times n}，
\quad
k\ll\min(m,n)
$$

因为

$$
\operatorname{rank}(BA)\le k
$$

所以更新量被限制在低秩子空间中。它与把训练后的 $W$ 做截断不是同一个流程：前者约束学习到的变化，后者压缩已经学到的矩阵。SVD 可以分析更新量的有效秩，但不能保证任意任务的最优 LoRA 因子。

连续线性层的低秩瓶颈还会减少可表达的方向数：如果某层只有 $k$ 个非零奇异值，它最多把输入的信息送进一个 $k$ 维像空间。后续的 [伪逆](../linear-algebra/pseudoinverse/)会用同一套奇异方向讨论哪些信息可以恢复，哪些信息已经在核中丢失。

## 容易混淆的地方

**把最大元素留下来。** 低秩截断按奇异值排序，不按矩阵元素的绝对值排序；矩阵的方向作用藏在整个外积项中。

**把 rank $k$ 当作保留 $k$ 个非零元素。** 一个秩一矩阵通常有很多非零元素，秩说的是独立通道数，不是稀疏度。

**只报告一个误差数字。** 谱范数误差是最坏方向的长度误差，Frobenius 误差是全体元素平方误差；同一截断在两者下的数字不同。

**任意删行或删列。** 删除原坐标中的行列通常没有达到奇异值截断的最优误差，除非矩阵恰好已经在相应坐标下对齐。

**忽略奇异值相等的情况。** 如果 $\sigma_k=\sigma_{k+1}$，最优的秩 $k$ 子空间可能不唯一，但最小误差数值仍由这些奇异值决定。

**看到谱衰减就断言噪声被删掉。** 小奇异值也可能承载任务信号；降噪假设必须用验证数据或噪声模型检查。

**忘记因子存储的开销。** $U_k$、$\Sigma_k$、$V_k$ 三个因子只有在 $k(m+n+1)<mn$ 时才比原矩阵更省。

## 相关词条

- [奇异值分解](../linear-algebra/svd/)：定义奇异值、左右奇异向量与外积展开。
- [秩](../linear-algebra/rank/)：解释非零奇异值数量与像空间维数。
- [伪逆](../linear-algebra/pseudoinverse/)：沿非零奇异方向反向缩放，处理最小二乘和秩亏。
- [矩阵范数](../linear-algebra/matrix-norms/)：展开谱范数、Frobenius 范数与奇异值的关系。
- [正交归一基](../linear-algebra/orthonormal-basis/)：构成截断近似中的输入输出方向。
- [主成分分析](../linear-models/pca/)：把截断 SVD 用于平方重构误差最小的子空间。
- [LoRA 低秩适配](../finetuning/lora/)：用低秩因子限制参数更新。
