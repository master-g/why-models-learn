---
title: "迹:矩阵对角线上的总和与线性不变量"
tags: ["why-models-learn"]
---

**迹**（trace）把一个方阵的主对角线元素相加：$\operatorname{tr}(A)=\sum_i a_{ii}$。它看起来只读取矩阵的几个位置，却不依赖同一个线性映射所选的基；更重要的是，迹对加法是线性的，对乘积满足循环不变性，并且等于全部特征值之和。[行列式](../linear-algebra/determinant/)记录有向体积因子，迹记录另一种更加法性的整体量。本篇先从对角线总和出发，再说明这些性质如何在投影、协方差和神经网络中出现。

## 定义：只对方阵加主对角线

设

$$
A=
\begin{pmatrix}
2&1\\
0&3
\end{pmatrix}
$$

它的主对角线是左上到右下的 $2$、$3$，所以

$$
\operatorname{tr}(A)=2+3=5
$$

三阶矩阵也一样：

$$
C=
\begin{pmatrix}
1&2&4\\
0&-1&3\\
2&0&6
\end{pmatrix},
\qquad
\operatorname{tr}(C)=1+(-1)+6=6
$$

迹的输入必须是方阵，因为它描述的是一个空间到自身的线性映射。长方形矩阵没有一条能把全部坐标成对对应起来的主对角线；不过如果 $X$ 是 $m\times n$、$Y$ 是 $n\times m$，那么 $XY$ 和 $YX$ 分别是方阵，两者都可以取迹，而且它们的迹相等，这正是后面要证明的循环性质。

主对角线上的数字不是矩阵的全部信息。非零矩阵

$$
N=
\begin{pmatrix}
0&1\\
0&0
\end{pmatrix}
$$

满足 $\operatorname{tr}(N)=0$。因此「迹为零」不能推出「矩阵为零」；迹是一个压缩后的数，不是矩阵内容的完整摘要。

![迹的两个读法：左侧把主对角线元素相加；右侧换基后矩阵元素和对角线都会变，但迹仍为 5](/assets/linear-algebra/svg/trace.1.svg)

## 迹的线性性

对同型方阵 $A$、$B$ 和标量 $c$，逐个看对角线位置就能得到

$$
\begin{aligned}
\operatorname{tr}(A+B)
&=\sum_i(a_{ii}+b_{ii})
=\sum_i a_{ii}+\sum_i b_{ii}
=\operatorname{tr}(A)+\operatorname{tr}(B),\\
\operatorname{tr}(cA)
&=\sum_i ca_{ii}
=c\sum_i a_{ii}
=c\operatorname{tr}(A)
\end{aligned}
$$

所以迹是「矩阵空间到标量空间」的线性映射。两个立即可用的特例是

$$
\operatorname{tr}(0)=0,
\qquad
\operatorname{tr}(I_n)=n
$$

不要把这种线性性误读成乘法性。取

$$
A=
\begin{pmatrix}
2&1\\
0&3
\end{pmatrix},
\qquad
B=
\begin{pmatrix}
1&2\\
4&0
\end{pmatrix}
$$

直接相乘：

$$
AB=
\begin{pmatrix}
6&4\\
12&0
\end{pmatrix},
\qquad
BA=
\begin{pmatrix}
2&7\\
8&4
\end{pmatrix}
$$

于是

$$
\operatorname{tr}(AB)=6,
\qquad
\operatorname{tr}(BA)=6,
\qquad
\operatorname{tr}(A)\operatorname{tr}(B)=5\cdot1=5
$$

乘积的迹是 $6$，不是两个迹的乘积 $5$。这里出现的 $\operatorname{tr}(AB)=\operatorname{tr}(BA)$ 不是巧合，而是迹最常用的乘积规则。

## 循环不变性：乘积可以循环移动

设 $X$ 是 $m\times n$，$Y$ 是 $n\times m$。虽然 $XY$ 和 $YX$ 的尺寸不同，但都为方阵。按下标展开：

$$
\begin{aligned}
\operatorname{tr}(XY)
&=\sum_{i=1}^{m}(XY)_{ii}
=\sum_{i=1}^{m}\sum_{j=1}^{n}x_{ij}y_{ji}\\
&=\sum_{j=1}^{n}\sum_{i=1}^{m}y_{ji}x_{ij}
=\sum_{j=1}^{n}(YX)_{jj}
=\operatorname{tr}(YX)
\end{aligned}
$$

中间只是交换了有限求和的顺序和标量乘法的顺序，没有要求 $XY=YX$。因此三项乘积也可以循环：

$$
\operatorname{tr}(ABC)
=\operatorname{tr}(BCA)
=\operatorname{tr}(CAB)
$$

但「循环」不等于「任意重排」。一般不能把 $B$、$C$ 随意交换成

$$
\operatorname{tr}(ABC)=\operatorname{tr}(ACB)
$$

矩阵乘法通常不交换；只有保持原顺序、把整个首段移到末尾，才是循环性质允许的操作。实际推导中常把一个大矩阵移到另一个因子旁边，利用这条性质缩短表达式。

## 相似变换下不变：同一映射换一组基

同一个线性映射在两组基下的矩阵形如

$$
A'=P^{-1}AP
$$

其中 $P$ 可逆。用循环不变性：

$$
\operatorname{tr}(P^{-1}AP)
=\operatorname{tr}(APP^{-1})
=\operatorname{tr}(A)
$$

所以矩阵的具体对角线会随基改变，但它们的和不变。取

$$
A=
\begin{pmatrix}
1&2\\
3&4
\end{pmatrix},
\qquad
P=
\begin{pmatrix}
1&1\\
0&1
\end{pmatrix},
\qquad
P^{-1}=
\begin{pmatrix}
1&-1\\
0&1
\end{pmatrix}
$$

计算得

$$
A'=P^{-1}AP=
\begin{pmatrix}
-2&-4\\
3&7
\end{pmatrix}
$$

原矩阵的对角线是 $1、4$，新矩阵的对角线是 $-2、7$；两者分别相加都为 $5$：

$$
\operatorname{tr}(A)=1+4=5,
\qquad
\operatorname{tr}(A')=-2+7=5
$$

这和 [换基](../linear-algebra/change-of-basis/) 篇的方向约定一致：$P$ 改变的是坐标表示，不是线性映射本身。迹、秩和行列式属于同一映射的数值性质；某个位置上的元素则属于所选坐标系。

## 迹与特征值、行列式的关系

对二维矩阵

$$
A=
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix}
$$

把特征值问题写成行列式方程：

$$
\begin{aligned}
\det(\lambda I-A)
&=\det\begin{pmatrix}
\lambda-a&-b\\
-c&\lambda-d
\end{pmatrix}\\
&=(\lambda-a)(\lambda-d)-bc\\
&=\lambda^2-(a+d)\lambda+(ad-bc)\\
&=\lambda^2-\operatorname{tr}(A)\lambda+\det(A)
\end{aligned}
$$

如果两个特征值是 $\lambda_1、\lambda_2$，这个多项式也写成

$$
(\lambda-\lambda_1)(\lambda-\lambda_2)
=\lambda^2-(\lambda_1+\lambda_2)\lambda+\lambda_1\lambda_2
$$

比较同次幂的系数：

$$
\operatorname{tr}(A)=\lambda_1+\lambda_2,
\qquad
\det(A)=\lambda_1\lambda_2
$$

高维情形同样成立：迹是特征多项式中次高项的相反数所对应的系数，等于全部特征值按代数重数相加；行列式是常数项的符号修正，等于特征值的乘积。[特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/) 篇会解释为什么这些数描述的是「某些方向只伸缩、不转向」的行为。这里先记住一个可检验的二维版本：对角线总和与特征值总和是同一个数的两种读法。

## 迹内积：把矩阵当成一个向量

矩阵也可以组成向量空间。对同型矩阵 $A、B$，Frobenius 内积可以写成

$$
\langle A,B\rangle_F
=\operatorname{tr}(A^{\mathsf T}B)
=\sum_{i,j}a_{ij}b_{ij}
$$

第一步来自迹的循环求和：

$$
\begin{aligned}
\operatorname{tr}(A^{\mathsf T}B)
&=\sum_i(A^{\mathsf T}B)_{ii}
=\sum_i\sum_j a_{ji}b_{ji}\\
&=\sum_{i,j}a_{ij}b_{ij}
\end{aligned}
$$

令 $B=A$，便得到 Frobenius 范数的平方：

$$
\|A\|_F^2
=\langle A,A\rangle_F
=\operatorname{tr}(A^{\mathsf T}A)
=\sum_{i,j}a_{ij}^2
$$

这和普通向量的 $\|x\|_2^2=x^{\mathsf T}x$ 是同一个结构，只是把一个矩阵的所有元素按某种顺序排成了一个长向量。注意这里是 $\operatorname{tr}(A^{\mathsf T}A)$，不是 $\operatorname{tr}(A^2)$；后者可以为负，也不等于元素平方和。

## 正交投影的迹等于投影维数

前面的 [正交投影](../linear-algebra/orthogonal-projections/) 给出：若 $Q\in\mathbb{R}^{n\times k}$ 的列是正交归一基，则投影矩阵为

$$
P_W=QQ^{\mathsf T}
$$

用循环不变性把两个较大的因子换序：

$$
\operatorname{tr}(P_W)
=\operatorname{tr}(QQ^{\mathsf T})
=\operatorname{tr}(Q^{\mathsf T}Q)
=\operatorname{tr}(I_k)
=k
$$

因此正交投影的迹直接数出它投到的子空间维数。二维直线 $W=\operatorname{span}\{(1,1)\}$ 的投影矩阵是

$$
P=
\frac12
\begin{pmatrix}
1&1\\
1&1
\end{pmatrix},
\qquad
P^2=P,
\qquad
\operatorname{tr}(P)=\frac12+\frac12=1
$$

它的秩也是 $1$。这里的等式不是说每个投影矩阵的某个对角线元素都是 $0$ 或 $1$；这个例子里两个对角线元素都是 $1/2$，只有总和稳定地给出维数。

## 机器学习里的迹

**协方差的总方差。** 对均值已经移到零的随机向量 $x$，协方差矩阵是

$$
\Sigma=\mathbb{E}[xx^{\mathsf T}]
$$

其对角线 $\Sigma_{ii}$ 是第 $i$ 个坐标的方差，所以

$$
\operatorname{tr}(\Sigma)
=\sum_i\Sigma_{ii}
=\mathbb{E}\left[\sum_i x_i^2\right]
=\mathbb{E}\left[\|x\|_2^2\right]
$$

迹把每个坐标方向的方差加成一个总量。若选取正交归一的 $k$ 维子空间 $Q$，投影后保留的平均能量为

$$
\mathbb{E}\left[\|Q^{\mathsf T}x\|_2^2\right]
=\operatorname{tr}(Q^{\mathsf T}\Sigma Q)
=\operatorname{tr}(\Sigma QQ^{\mathsf T})
$$

这正是 PCA 里比较候选子空间所用的量；[方差与协方差](../probability/variance-and-covariance/)、[协方差矩阵](../probability/covariance-matrix/) 篇会从统计定义继续展开。

**二次型的期望。** 对固定矩阵 $M$，同样的循环变换给出

$$
\mathbb{E}[x^{\mathsf T}Mx]
=\mathbb{E}\left[\operatorname{tr}(Mxx^{\mathsf T})\right]
=\operatorname{tr}(M\Sigma)
$$

这里把标量 $x^{\mathsf T}Mx$ 写成 $1\times1$ 矩阵的迹，再把期望和有限求和交换。高斯模型的二次损失、协方差拟合和很多矩阵微积分公式都在反复使用这一步。

**权重大小。** 神经网络中的参数矩阵 $W$ 若用 L2 正则化，惩罚项可以写成

$$
\|W\|_F^2=\operatorname{tr}(W^{\mathsf T}W)
$$

它与单独看 $\operatorname{tr}(W)$ 完全不同：后者只把主对角线相加，甚至不要求非对角元素很小；Frobenius 范数才会把所有权重平方后计入。

## 容易混淆的地方

**把迹当行列式。** 对 $\operatorname{diag}(2,3)$，迹是 $5$，行列式是 $6$。前者是对角线总和，后者是面积或体积因子，数值和用途都不同。

**把 $\operatorname{tr}(AB)$ 当成 $\operatorname{tr}(A)\operatorname{tr}(B)$。** 上面的数字例子已经给出 $6\ne5$；正确的简化是循环移动因子，不能拆成两个标量乘积。

**把迹为零当成矩阵为零。** 非零的 $N=\begin{pmatrix}0&1\\0&0\end{pmatrix}$ 就是反例。迹只保留一个标量，不能判断全部元素。

**把任意换坐标都当成相似变换。** 只有同一个线性映射在两组基下的表示才有 $P^{-1}AP$ 形式；若换的是映射本身或随意改动元素，迹没有理由保持。

**把循环移动扩大成任意换序。** $\operatorname{tr}(ABC)=\operatorname{tr}(BCA)$，但一般不等于 $\operatorname{tr}(ACB)$；检查矩阵乘法的顺序和尺寸仍然必要。

**忘记特征值要按代数重数计数。** 迹等于特征值之和的说法包含重数，也允许复特征值；它不是「随便挑几个显眼的方向」相加。

## 相关词条

- [行列式](../linear-algebra/determinant/)：有向体积因子与可逆性的判据。
- [特征值与特征向量](../linear-algebra/eigenvalues-and-eigenvectors/)：迹作为特征值总和的几何解释。
- [特征多项式](../linear-algebra/characteristic-polynomial/)：从多项式系数读取迹与行列式。
- [线性映射](../linear-algebra/linear-maps/)：同一映射换基后得到相似矩阵。
- [换基](../linear-algebra/change-of-basis/)：坐标表示改变而映射不变。
- [内积](../linear-algebra/inner-products/)：Frobenius 内积把矩阵空间纳入同一套结构。
- [正交投影](../linear-algebra/orthogonal-projections/)：投影矩阵的迹等于子空间维数。
- [方差与协方差](../probability/variance-and-covariance/)：协方差对角线与总方差。
- [协方差矩阵](../probability/covariance-matrix/)：迹在统计矩阵中的计算。
