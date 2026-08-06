---
title: 词汇表:术语译法与约定
---

本页汇总全库术语的译法约定:哪些翻译、哪些保留英文、首次出现如何对照。写词条时遇到新的术语决策,随手在本页加一行;查阅时按分组找。「首现」指该术语第一次出现的词条。

## 线性代数

| 英文 | 中文 | 约定 | 首现 |
| --- | --- | --- | --- |
| Hadamard product | Hadamard 积(阿达马积) | 首次出现中英对照,后文用 Hadamard 积;即逐分量乘法 | [vectors](../linear-algebra/vectors/) |
| element-wise | 逐分量 | 统一译「逐分量」,不译「逐元素」 | [vectors](../linear-algebra/vectors/) |
| dot product | 点积 | 译 | [vectors](../linear-algebra/vectors/) |
| outer product | 外积 | 译 | [vectors](../linear-algebra/vectors/) |
| span | 张成 | 译 | [vectors](../linear-algebra/vectors/) |
| broadcasting | 广播 | 译 | [vectors](../linear-algebra/vectors/) |
| residual connection | 残差连接 | 译 | [vectors](../linear-algebra/vectors/) |
| rank | 秩 | 译 | [rank](../linear-algebra/rank/) |
| column rank / row rank | 列秩 / 行秩 | 译,首次出现括注英文 | [rank](../linear-algebra/rank/) |
| full rank | 满秩 | 译 | [rank](../linear-algebra/rank/) |
| rank deficient | 秩亏 | 译 | [rank](../linear-algebra/rank/) |
| column space | 列空间 | 译(列张成的空间),正式处理见「核与像」 | [rank](../linear-algebra/rank/) |
| rank-nullity theorem | 秩零定理 | 译 | [rank](../linear-algebra/rank/) |
| kernel | 核 | 译,正式处理见「核与像」 | [rank](../linear-algebra/rank/) |
| numerical rank | 数值秩 | 译,首次出现括注英文 | [rank](../linear-algebra/rank/) |
| vector space | 向量空间 | 译 | [vector-spaces](../linear-algebra/vector-spaces/) |
| subspace | 子空间 | 译(子集配继承的运算),判别三条:含零、加封闭、数乘封闭 | [subspaces](../linear-algebra/subspaces/) |
| zero space / kernel | 零空间/核 | 两译并用,正式处理见「核与像」 | [subspaces](../linear-algebra/subspaces/) |
| linear combination | 线性组合 | 译 | [linear-combinations-and-span](../linear-algebra/linear-combinations-and-span/) |
| affine combination | 仿射组合 | 译(系数和为 1) | [linear-combinations-and-span](../linear-algebra/linear-combinations-and-span/) |
| convex combination | 凸组合 | 译(系数和为 1 且非负) | [linear-combinations-and-span](../linear-algebra/linear-combinations-and-span/) |
| linearly independent / dependent | 线性无关 / 线性相关 | 译,判定:零向量是否只有平凡配方 | [linear-independence](../linear-algebra/linear-independence/) |
| basis | 基 | 译;张成且无关的向量组 | [basis](../linear-algebra/basis/) |
| standard basis | 标准基 | 译 | [basis](../linear-algebra/basis/) |
| dimension | 维数 | 译;任一基的元素个数 | [dimension](../linear-algebra/dimension/) |
| finite / infinite dimensional | 有限维 / 无限维 | 译 | [dimension](../linear-algebra/dimension/) |
| coordinates | 坐标 | 译;记号 [v]_B,基有序 | [coordinates](../linear-algebra/coordinates/) |
| isomorphism | 同构 | 译(线性同构:保持结构的双射) | [coordinates](../linear-algebra/coordinates/) |
| change of basis | 换基 | 译 | [change-of-basis](../linear-algebra/change-of-basis/) |
| transition matrix | 过渡矩阵 | 译;列是「新基读旧基」,记号 P_{C←B} | [change-of-basis](../linear-algebra/change-of-basis/) |
| linear map | 线性映射 | 译(保持线性组合的映射) | [linear-maps](../linear-algebra/linear-maps/) |
| similar matrices | 相似矩阵 | 译;同一映射在不同基下的矩阵 | [linear-maps](../linear-algebra/linear-maps/) |
| image / range | 像 | 译(值域里被够着的全体) | [kernel-and-image](../linear-algebra/kernel-and-image/) |
| injective / surjective | 单射 / 满射 | 译;核为零 ⟺ 单射,像=值域 ⟺ 满射 | [kernel-and-image](../linear-algebra/kernel-and-image/) |
| inverse / invertible | 逆矩阵 / 可逆 | 译;不可逆称奇异(singular) | [matrix-inverse](../linear-algebra/matrix-inverse/) |
| trivial subspace | 平凡子空间 | 译({0} 与全空间自身) | [subspaces](../linear-algebra/subspaces/) |
| model soup | 模型汤 | 半译,权重平均技巧 | [subspaces](../linear-algebra/subspaces/) |
| group / Abelian group | 群 / Abel 群 | 译,Abel 音译不翻 | [vector-spaces](../linear-algebra/vector-spaces/) |
| closure | 封闭性 | 译 | [vector-spaces](../linear-algebra/vector-spaces/) |
| scalar multiplication | 数乘 | 译(与 vectors 篇一致) | [vector-spaces](../linear-algebra/vector-spaces/) |
| affine set | 仿射集 | 译,正式处理见「仿射空间与映射」 | [vector-spaces](../linear-algebra/vector-spaces/) |
| affine subspace | 仿射子空间 | 译(子空间的平移) | [affine-spaces-and-maps](../linear-algebra/affine-spaces-and-maps/) |
| affine map | 仿射映射 | 译(线性映射加平移) | [affine-spaces-and-maps](../linear-algebra/affine-spaces-and-maps/) |
| norm | 范数 | 译;三条公理:正定性、齐次性、三角不等式 | [norms](../linear-algebra/norms/) |
| p-norm | p 范数 | 译;欧几里得(p=2)、曼哈顿(p=1)、最大模(p=∞) | [norms](../linear-algebra/norms/) |
| unit ball / unit sphere | 单位球 | 译(本篇球面与球体不分,均指 ‖x‖=1 或 ‖x‖≤1) | [norms](../linear-algebra/norms/) |
| triangle inequality | 三角不等式 | 译 | [norms](../linear-algebra/norms/) |
| Minkowski inequality | Minkowski 不等式 | 不译(音译闵可夫斯基亦见,词条正文用英文) | [norms](../linear-algebra/norms/) |
| inner product | 内积 | 译;三条公理:对称性、双线性、正定性 | [inner-products](../linear-algebra/inner-products/) |
| inner product space | 内积空间 | 译(向量空间配内积) | [inner-products](../linear-algebra/inner-products/) |
| Cauchy–Schwarz inequality | Cauchy–Schwarz 不等式 | 不译(音译柯西-施瓦茨亦见,词条正文用英文) | [inner-products](../linear-algebra/inner-products/) |
| Frobenius inner product / norm | Frobenius 内积 / 范数 | 不译(人名);⟨A,B⟩=tr(AᵀB) | [inner-products](../linear-algebra/inner-products/) |
| parallelogram law | 平行四边形法则 | 译;内积诱导范数的判据 | [inner-products](../linear-algebra/inner-products/) |
| angle | 角度 | 译;由归一化内积给出余弦 | [angles-and-orthogonality](../linear-algebra/angles-and-orthogonality/) |
| orthogonality / orthogonal | 正交 | 译;内积为零,不依赖坐标轴外观 | [angles-and-orthogonality](../linear-algebra/angles-and-orthogonality/) |
| orthogonal complement | 正交补 | 译;与子空间中每个向量都正交的集合 | [angles-and-orthogonality](../linear-algebra/angles-and-orthogonality/) |
| projection / orthogonal projection | 投影 / 正交投影 | 译;最近点与正交残差的分解 | [angles-and-orthogonality](../linear-algebra/angles-and-orthogonality/) |
| projection matrix | 投影矩阵 | 译;正交投影满足 P²=P 且 Pᵀ=P | [orthogonal-projections](../linear-algebra/orthogonal-projections/) |
| normal equations | 正规方程 | 译;AᵀAc=Aᵀx,残差与设计矩阵列空间正交 | [orthogonal-projections](../linear-algebra/orthogonal-projections/) |
| least squares | 最小二乘 | 译;最小化 ‖y−Ac‖²,几何上是列空间投影 | [orthogonal-projections](../linear-algebra/orthogonal-projections/) |
| weighted least squares | 加权最小二乘 | 译;在加权内积下最小化残差平方,用 W 表示不同样本或方向的误差权重 | [least-squares-as-projection](../linear-models/least-squares-as-projection/) |
| QR decomposition | QR 分解 | 保留缩写;用正交归一列和上三角矩阵分解 A=QR | [orthogonal-projections](../linear-algebra/orthogonal-projections/) |
| orthonormal basis | 正交归一基 | 译;两两正交且每个向量长度为 1 的基 | [angles-and-orthogonality](../linear-algebra/angles-and-orthogonality/) |
| Gram-Schmidt process | Gram–Schmidt 过程 | 保留人名;逐步去掉已有方向分量并归一化,构造正交归一基 | [orthonormal-basis](../linear-algebra/orthonormal-basis/) |
| orthogonal matrix | 正交矩阵 | 译;满足 QᵀQ=I,保持内积与长度 | [orthogonal-matrices-and-rotations](../linear-algebra/orthogonal-matrices-and-rotations/) |
| orthogonal transformation | 正交变换 | 译;保持所选内积、长度、距离与角度的线性变换 | [orthogonal-matrices-and-rotations](../linear-algebra/orthogonal-matrices-and-rotations/) |
| rotation | 旋转 | 译;二维矩阵 Rθ 保持方向且满足 RθᵀRθ=I | [orthogonal-matrices-and-rotations](../linear-algebra/orthogonal-matrices-and-rotations/) |
| reflection | 反射 | 译;保持长度但翻转方向的正交变换 | [orthogonal-matrices-and-rotations](../linear-algebra/orthogonal-matrices-and-rotations/) |
| determinant | 行列式 | 译;线性变换的有向体积因子,非零等价于方阵可逆 | [determinant](../linear-algebra/determinant/) |
| signed area / volume | 有向面积 / 有向体积 | 译;绝对值是普通面积或体积,符号记录方向 | [determinant](../linear-algebra/determinant/) |
| cofactor expansion | 按余子式展开 | 译;沿行或列递归降为低阶行列式 | [determinant](../linear-algebra/determinant/) |
| Jacobian determinant | Jacobian 行列式 | 保留 Jacobian;局部体积因子为 det J 的绝对值 | [determinant](../linear-algebra/determinant/) |
| trace | 迹 | 译;方阵主对角线元素之和,相似变换下不变 | [trace](../linear-algebra/trace/) |
| cyclic property of trace | 迹的循环不变性 | 译;tr(XY)=tr(YX),多因子只能循环移动而非任意换序 | [trace](../linear-algebra/trace/) |
| characteristic polynomial | 特征多项式 | 译;det(λI−A),其系数包含迹与行列式 | [trace](../linear-algebra/trace/) |
| eigenvalue / eigenvector | 特征值 / 特征向量 | 译;Av=λv,特征向量方向经变换后只整体缩放 | [eigenvalues-and-eigenvectors](../linear-algebra/eigenvalues-and-eigenvectors/) |
| eigenspace | 特征空间 | 译;ker(A−λI),包含零向量的对应解空间 | [eigenvalues-and-eigenvectors](../linear-algebra/eigenvalues-and-eigenvectors/) |
| diagonalization | 对角化 | 译;有特征基时 A=PDP⁻¹,把变换改写为独立缩放 | [eigenvalues-and-eigenvectors](../linear-algebra/eigenvalues-and-eigenvectors/) |
| eigendecomposition | 特征分解 | 译;A=PΛP⁻¹,用特征基与对角特征值重写矩阵 | [eigendecomposition](../linear-algebra/eigendecomposition/) |
| spectral theorem | 谱定理 | 译;实对称矩阵有正交归一特征基,可写成 QΛQᵀ | [spectral-theorem](../linear-algebra/spectral-theorem/) |
| real symmetric matrix | 实对称矩阵 | 译;A=Aᵀ,谱定理适用的矩阵 | [spectral-theorem](../linear-algebra/spectral-theorem/) |
| positive semidefinite / positive definite | 半正定 / 正定 | 译;对称矩阵特征值全非负 / 全正 | [spectral-theorem](../linear-algebra/spectral-theorem/) |
| algebraic / geometric multiplicity | 代数重数 / 几何重数 | 译;根的重复次数 / 对应特征空间的维数 | [eigenvalues-and-eigenvectors](../linear-algebra/eigenvalues-and-eigenvectors/) |
| spectral radius | 谱半径 | 译;全部特征值绝对值的最大值 | [eigenvalues-and-eigenvectors](../linear-algebra/eigenvalues-and-eigenvectors/) |
| Rayleigh quotient | Rayleigh 商 | 保留人名;vᵀAv/vᵀv,用来比较方向上的二次型值 | [eigenvalues-and-eigenvectors](../linear-algebra/eigenvalues-and-eigenvectors/) |
| quadratic form | 二次型 | 译;q_A(x)=xᵀAx,同一向量放在矩阵两侧的标量函数 | [quadratic-forms](../linear-algebra/quadratic-forms/) |
| bilinear form | 双线性形式 | 译;xᵀAy,分别对两个向量线性 | [quadratic-forms](../linear-algebra/quadratic-forms/) |
| level set | 等值集 / 等值线 / 等值面 | 译;函数取同一常数的点集,二维称等值线或等高线,高维称等值面 | [quadratic-forms](../linear-algebra/quadratic-forms/) |
| indefinite quadratic form | 不定二次型 | 译;同一矩阵存在正值和负值方向 | [quadratic-forms](../linear-algebra/quadratic-forms/) |
| Hessian | Hessian 矩阵 | 保留 Hessian;二阶偏导组成的曲率矩阵 | [hessian](../calculus/hessian/) |
| Mahalanobis distance | Mahalanobis 距离 | 保留人名;用协方差逆矩阵加权的二次型距离 | [quadratic-forms](../linear-algebra/quadratic-forms/) |
| singular value decomposition / SVD | 奇异值分解 / SVD | 译;A=UΣVᵀ,把任意矩阵拆成正交换基与非负缩放 | [svd](../linear-algebra/svd/) |
| singular value | 奇异值 | 译;AᵀA 的非负特征值的平方根,表示对应方向的长度增益 | [svd](../linear-algebra/svd/) |
| left / right singular vector | 左奇异向量 / 右奇异向量 | 译;分别是输出方向与输入方向的正交基向量 | [svd](../linear-algebra/svd/) |
| thin / compact SVD | 紧 SVD | 译;只保留非零奇异值及其对应的左右奇异向量 | [svd](../linear-algebra/svd/) |
| low-rank approximation | 低秩近似 | 译;用前 k 个奇异通道近似矩阵,在谱范数和 Frobenius 范数下由截断 SVD 达到最小误差 | [low-rank-approximation](../linear-algebra/low-rank-approximation/) |
| rank-k approximation | 秩 k 近似 | 译;秩不超过 k 的矩阵近似,不等于只保留 k 个非零元素 | [low-rank-approximation](../linear-algebra/low-rank-approximation/) |
| Frobenius norm | Frobenius 范数 | 保留人名;所有元素平方和的平方根 | [low-rank-approximation](../linear-algebra/low-rank-approximation/) |
| Moore–Penrose pseudoinverse | Moore–Penrose 伪逆 | 保留人名;满足四个 Penrose 条件的唯一广义逆,用 SVD 对非零奇异值取倒数 | [pseudoinverse](../linear-algebra/pseudoinverse/) |
| Penrose equations | Penrose 条件 | 译;AA⁺A=A、A⁺AA⁺=A⁺及两个对称投影条件 | [pseudoinverse](../linear-algebra/pseudoinverse/) |
| minimum-norm solution | 最小范数解 | 译;在同样达到最小二乘残差的解中长度最小者 | [pseudoinverse](../linear-algebra/pseudoinverse/) |
| least-squares solution | 最小二乘解 | 译;使 ‖Ax−b‖² 最小的参数,不一定让 Ax=b 精确成立 | [pseudoinverse](../linear-algebra/pseudoinverse/) |
| matrix norm | 矩阵范数 | 译;给矩阵或线性映射量大小的函数,满足范数公理 | [matrix-norms](../linear-algebra/matrix-norms/) |
| induced p-norm / operator norm | 诱导 p 范数 / 算子范数 | 译;单位 p 范数球经 A 后的最大长度放大 | [matrix-norms](../linear-algebra/matrix-norms/) |
| spectral norm / operator 2-norm | 谱范数 / 算子 2 范数 | 译;诱导 2 范数,等于最大奇异值 | [matrix-norms](../linear-algebra/matrix-norms/) |
| nuclear norm / trace norm | 核范数 / 迹范数 | 译;奇异值之和,常作低秩正则化 | [matrix-norms](../linear-algebra/matrix-norms/) |
| condition number | 条件数 | 译;‖A‖·‖A⁻¹‖,衡量逆问题对扰动的敏感性 | [matrix-norms](../linear-algebra/matrix-norms/) |
| principal minor | 主子式 | 译;删去同组行列后形成的方阵行列式,特征多项式系数由其组合构成 | [characteristic-polynomial](../linear-algebra/characteristic-polynomial/) |
| Cayley–Hamilton theorem | Cayley–Hamilton 定理 | 保留人名;矩阵满足自己的特征多项式 p_A(A)=0 | [characteristic-polynomial](../linear-algebra/characteristic-polynomial/) |
| matrix polynomial | 矩阵多项式 | 译;把标量变量换成矩阵且常数项乘单位矩阵 | [characteristic-polynomial](../linear-algebra/characteristic-polynomial/) |
| spectral projector | 谱投影 | 译;在特征坐标中选取一个方向后换回原空间的幂等算子 | [eigendecomposition](../linear-algebra/eigendecomposition/) |
| metric / distance | 度量 / 距离 | 译;非负、同一性、对称性、三角不等式 | [lengths-and-distances](../linear-algebra/lengths-and-distances/) |
| metric space | 度量空间 | 译;集合配上满足四条距离公理的度量 | [lengths-and-distances](../linear-algebra/lengths-and-distances/) |
| metric ball | 距离球 | 译;B_d(a,r)={x:d(x,a)≤r} | [lengths-and-distances](../linear-algebra/lengths-and-distances/) |
| Euclidean / Manhattan / Chebyshev distance | 欧几里得 / 曼哈顿 / 切比雪夫距离 | 译;分别对应 ℓ₂、ℓ₁、ℓ∞ | [lengths-and-distances](../linear-algebra/lengths-and-distances/) |
| squared Euclidean distance | 平方欧氏距离 | 不称距离或度量;违反三角不等式,常用于均方误差 | [lengths-and-distances](../linear-algebra/lengths-and-distances/) |

## 微积分

| 英文 | 中文 | 约定 | 首现 |
| --- | --- | --- | --- |
| derivative / derivative function | 导数 / 导函数 | 译;差商在步长趋零时的极限,也表示局部线性系数 | [derivatives](../calculus/derivatives/) |
| difference quotient | 差商 | 译;(f(a+h)−f(a))/h,有限步长的平均变化率 | [derivatives](../calculus/derivatives/) |
| instantaneous rate of change | 瞬时变化率 | 译;一点处的导数值 | [derivatives](../calculus/derivatives/) |
| tangent line | 切线 | 译;在一点处斜率等于导数的直线 | [derivatives](../calculus/derivatives/) |
| differentiable / differentiability | 可导 / 可导性 | 译;差商极限存在且有限 | [derivatives](../calculus/derivatives/) |
| local linear approximation | 局部线性近似 | 译;f(a+h)=f(a)+f′(a)h+o(h) | [derivatives](../calculus/derivatives/) |
| chain rule | 链式法则 | 译;(f∘g)′(x)=f′(g(x))g′(x),复合函数的局部变化率相乘 | [chain-rule](../calculus/chain-rule/) |
| composition / composite function | 复合 / 复合函数 | 译;先算内层再把输出送入外层 | [chain-rule](../calculus/chain-rule/) |
| inner / outer function | 内层函数 / 外层函数 | 译;g 是 f(g(x)) 的内层,f 是外层 | [chain-rule](../calculus/chain-rule/) |
| partial derivative | 偏导数 | 译;固定其余坐标,只沿一个坐标方向求导 | [partial-derivatives](../calculus/partial-derivatives/) |
| mixed partial derivative | 混合偏导数 | 译;对不同坐标连续求偏导,如 fₓᵧ | [partial-derivatives](../calculus/partial-derivatives/) |
| Clairaut's theorem | Clairaut 定理 | 保留人名;二阶偏导连续时交换求导次序 | [partial-derivatives](../calculus/partial-derivatives/) |
| gradient | 梯度 | 译;标量函数对各坐标的偏导组成的向量,指向局部上升最快方向 | [gradient](../calculus/gradient/) |
| directional derivative | 方向导数 | 译;沿单位方向的变化率,等于梯度与方向向量的内积 | [gradient](../calculus/gradient/) |
| total derivative | 全导数 | 译;在一点用线性映射统一近似所有输入方向的变化 | [total-derivative](../calculus/total-derivative/) |
| differential | 微分 | 译;全导数作用于输入位移得到的一阶输出变化 | [total-derivative](../calculus/total-derivative/) |
| Fréchet derivative | Fréchet 导数 | 保留人名;用范数余项定义的无限维空间也可用的全导数 | [total-derivative](../calculus/total-derivative/) |
| Jacobian matrix | 雅可比矩阵 | 译;向量函数各输出对各输入的偏导组成的矩阵,行是输出列是输入 | [jacobian](../calculus/jacobian/) |
| local linear map | 局部线性映射 | 译;在一点用线性映射近似函数对小扰动的响应 | [jacobian](../calculus/jacobian/) |
| vector chain rule | 向量链式法则 | 译;复合向量函数的 Jacobian 按外层乘内层 | [vector-chain-rule](../calculus/vector-chain-rule/) |
| path-sum rule / path product | 路径求和规则 / 路径乘积 | 译;标量图上每条输入输出路径乘局部导数,不同路径的贡献在起点相加 | [chain-rule-on-graphs](../backpropagation/chain-rule-on-graphs/) |
| forward-mode differentiation | 前向模式微分 | 译;从输入方向出发逐层传播局部扰动 | [vector-chain-rule](../calculus/vector-chain-rule/) |
| reverse-mode differentiation | 反向模式微分 | 译;从输出敏感度出发逐层乘转置 Jacobian 拉回输入 | [vector-chain-rule](../calculus/vector-chain-rule/) |
| second directional derivative | 二阶方向导数 | 译;沿单位方向的二阶变化率,等于 uᵀHu | [hessian](../calculus/hessian/) |
| Hessian-vector product | Hessian-向量积 | 保留 Hessian;H 作用于方向向量而不必显式存储整个矩阵 | [hessian](../calculus/hessian/) |
| Taylor polynomial | Taylor 多项式 | 保留人名;在一点匹配函数各阶导数的有限多项式 | [taylor-series](../calculus/taylor-series/) |
| Taylor series | Taylor 级数 | 保留人名;Taylor 多项式阶数趋于无穷的形式级数,需检查收敛与等值 | [taylor-series](../calculus/taylor-series/) |
| Taylor remainder | Taylor 余项 | 保留人名;函数值与有限 Taylor 多项式之间的误差项 | [taylor-series](../calculus/taylor-series/) |
| radius of convergence | 收敛半径 | 译;幂级数围绕展开点收敛的最大距离 | [taylor-series](../calculus/taylor-series/) |
| elementwise function | 逐分量函数 | 译;第 i 个输出只依赖第 i 个输入的函数 | [elementwise-derivatives](../calculus/elementwise-derivatives/) |
| diagonal Jacobian | 对角 Jacobian | 译;逐分量函数的 Jacobian,非对角偏导为零 | [elementwise-derivatives](../calculus/elementwise-derivatives/) |
| separable function | 可分离函数 | 译;可写成各坐标函数之和或独立组合的函数 | [elementwise-derivatives](../calculus/elementwise-derivatives/) |
| reduction | 归约 | 译;沿一个或多个坐标轴把多个值合成较小形状 | [broadcast-and-reduction-derivatives](../calculus/broadcast-and-reduction-derivatives/) |
| broadcast derivative | 广播导数 | 译;共享输入的反向梯度沿被复制的轴求和 | [broadcast-and-reduction-derivatives](../calculus/broadcast-and-reduction-derivatives/) |
| adjoint map | 伴随映射 | 译;相对于内积把输出敏感度拉回输入的线性映射 | [broadcast-and-reduction-derivatives](../calculus/broadcast-and-reduction-derivatives/) |
| matrix gradient | 矩阵梯度 | 译;使 df=tr((∇X f)ᵀdX) 的矩阵,与 X 同形状 | [matrix-calculus-identities](../calculus/matrix-calculus-identities/) |
| matrix differential | 矩阵微分 | 译;表达矩阵变量局部线性变化的 dX 记号 | [matrix-calculus-identities](../calculus/matrix-calculus-identities/) |
| log-determinant | log-det / 对数行列式 | 保留常用缩写;正定矩阵行列式的对数,梯度为 X⁻ᵀ | [matrix-calculus-identities](../calculus/matrix-calculus-identities/) |
| finite difference | 有限差分 | 译;用有限个函数值的差商近似导数 | [numerical-differentiation](../calculus/numerical-differentiation/) |
| truncation error | 截断误差 | 译;用有限阶 Taylor 近似替代极限产生的误差 | [numerical-differentiation](../calculus/numerical-differentiation/) |
| roundoff error | 舍入误差 | 译;有限精度表示和相近数相减产生的误差 | [numerical-differentiation](../calculus/numerical-differentiation/) |
| gradient checking | 梯度检查 | 译;在固定目标、数据和执行状态下用数值微分逐坐标或沿方向核对解析/自动微分梯度 | [gradient-checking](../backpropagation/gradient-checking/) |
| analytic gradient / numerical gradient | 解析梯度 / 数值梯度 | 译;分别指链式法则或自动微分得到的梯度与由有限差分近似的梯度,两者必须作用于同一个标量目标 | [gradient-checking](../backpropagation/gradient-checking/) |
| central difference | 中心差分 | 译;用正负扰动函数值之差除以两倍步长的二阶有限差分公式 | [gradient-checking](../backpropagation/gradient-checking/) |
| relative gradient error | 相对梯度误差 | 译;用解析值与数值值之差除以两者尺度和下限的梯度检查指标 | [gradient-checking](../backpropagation/gradient-checking/) |
| directional derivative check | 方向导数检查 | 译;比较解析梯度与方向的内积和正负方向有限差分,用两次前向覆盖一组参数组合 | [gradient-checking](../backpropagation/gradient-checking/) |
| automatic differentiation | 自动微分 | 译;沿实际执行的基本运算应用链式法则计算导数 | [automatic-differentiation](../calculus/automatic-differentiation/) |
| dual number | 对偶数 | 译;用 ε²=0 同时携带数值和一个方向导数 | [automatic-differentiation](../calculus/automatic-differentiation/) |
| Jacobian-vector product | Jacobian–向量积 / JVP | 保留缩写;前向模式计算 Jv | [automatic-differentiation](../calculus/automatic-differentiation/) |
| vector-Jacobian product | 向量–Jacobian 积 / VJP | 保留缩写;反向模式计算 uᵀJ | [automatic-differentiation](../calculus/automatic-differentiation/) |
| computational graph | 计算图 | 译;把基本运算表示为节点和依赖边的有向图 | [computational-graphs](../backpropagation/computational-graphs/) |
| directed acyclic graph / DAG | 有向无环图 / DAG | 保留缩写;有向边不能沿依赖回到原节点,可用拓扑序求值 | [computational-graphs](../backpropagation/computational-graphs/) |
| topological order / topological sort | 拓扑序 / 拓扑排序 | 译;每条依赖边的父节点都排在子节点之前的节点顺序 | [computational-graphs](../backpropagation/computational-graphs/) |
| computational tape / tape | 计算轨迹 / tape | 保留常用词;某次运行中实际经过的运算、值和反向缓存记录 | [computational-graphs](../backpropagation/computational-graphs/) |
| graph tracing / tracing | 图追踪 / tracing | 译并保留常用词;用样例执行记录实际经过的计算节点和控制流 | [computational-graphs](../backpropagation/computational-graphs/) |
| common subexpression elimination | 公共子表达式消除 | 译;把数学上相同且可安全共享的子图合并,减少重复计算 | [computational-graphs](../backpropagation/computational-graphs/) |
| operator fusion | 算子融合 | 译;把连续运算合并为一个执行 kernel,减少中间张量读写 | [computational-graphs](../backpropagation/computational-graphs/) |
| backpropagation / backward pass | 反向传播 / 反向阶段 | 译;在固定前向计算图上从标量损失沿反向拓扑序累加局部梯度,本身不执行参数更新 | [backpropagation](../backpropagation/backpropagation/) |
| forward pass / forward propagation | 前向计算 / 前向传播 | 译;固定参数和输入后沿依赖顺序计算中间表示、输出和可选损失,不包含反向或参数更新 | [forward-pass](../backpropagation/forward-pass/) |
| training mode / evaluation mode | 训练态 / 评估态 | 译;决定 dropout 随机性、batch 统计量等层在一次前向中采用的规则 | [forward-pass](../backpropagation/forward-pass/) |
| loss function / pointwise loss | 损失函数 / 逐样本损失 | 译;把一次预测与真实结果的差异写成可比较、可求导的数值 | [loss-functions](../training-nn/loss-functions/) |
| surrogate loss | 替代损失 | 译;用可优化的连续或分段目标近似准确率等不可导决策指标 | [loss-functions](../training-nn/loss-functions/) |
| Bayes decision / conditional risk | Bayes 决策 / 条件风险 | 保留常用人名并译;固定输入后使条件期望损失最小的决策及其风险 | [loss-functions](../training-nn/loss-functions/) |
| loss reduction | 损失归约 | 译;把逐样本或逐位置损失按 sum、mean 或 mask 权重合成为训练目标的规则 | [loss-functions](../training-nn/loss-functions/) |
| adjoint variable / adjoint | 伴随量 | 译;节点 v 对标量损失的敏感度,记为 v̄=∂L/∂v | [backpropagation](../backpropagation/backpropagation/) |
| gradient accumulation | 梯度累加 | 译;同一变量经多条路径影响损失时,把各条边的梯度贡献相加 | [backpropagation](../backpropagation/backpropagation/) |
| activation checkpointing / recomputation | 激活检查点 / 重算 | 译;只保存部分前向值并在反向前重算,用额外计算换取显存 | [backpropagation](../backpropagation/backpropagation/) |
| autograd | autograd / PyTorch 自动微分引擎 | 保留框架名称并译;沿一次实际前向记录的运算图反向应用链式法则,不等于优化器 | [autodiff-in-pytorch](../backpropagation/autodiff-in-pytorch/) |
| requires_grad | requires_grad / 梯度追踪开关 | 保留属性名并译;决定默认 grad mode 下以该 Tensor 为输入的后续运算是否记录反向历史 | [autodiff-in-pytorch](../backpropagation/autodiff-in-pytorch/) |
| leaf tensor / non-leaf tensor | 叶张量 / 非叶张量 | 译;叶张量不是由当前图运算产生的输入,非叶张量由图中运算产生且默认不保留 `.grad` | [autodiff-in-pytorch](../backpropagation/autodiff-in-pytorch/) |
| grad_fn | grad_fn / 反向图入口 | 保留属性名并译;记录结果张量由哪个已追踪运算产生的反向节点引用 | [autodiff-in-pytorch](../backpropagation/autodiff-in-pytorch/) |
| saved tensor | 保存张量 / 反向缓存 | 译;前向为反向局部导数暂存的中间 Tensor,会影响反向内存和图生命周期 | [autodiff-in-pytorch](../backpropagation/autodiff-in-pytorch/) |
| grad mode / no-grad mode | grad mode / no-grad mode | 保留常用名称并译;默认记录梯度的模式与用上下文管理器关闭反向图记录的模式 | [autodiff-in-pytorch](../backpropagation/autodiff-in-pytorch/) |
| inference mode | inference mode / 推理模式 | 保留常用名称并译;比 no-grad 更严格、面向推理开销的 autograd 关闭模式,不等于 `model.eval()` | [autodiff-in-pytorch](../backpropagation/autodiff-in-pytorch/) |
| detach | detach / 脱离计算图 | 保留常用名称并译;返回与前驱图断开梯度连接的 Tensor,后续梯度不再回到被脱离分支 | [autodiff-in-pytorch](../backpropagation/autodiff-in-pytorch/) |
| custom autograd Function | 自定义 autograd Function | 保留 API 名并译;用 `forward`、`backward` 和 `save_for_backward` 接管一个局部算子的梯度规则 | [autodiff-in-pytorch](../backpropagation/autodiff-in-pytorch/) |

## 概率论

| 英文 | 中文 | 约定 | 首现 |
| --- | --- | --- | --- |
| probability space | 概率空间 | 译;样本空间、事件 σ-代数与概率测度的三元组 | [probability-spaces](../probability/probability-spaces/) |
| sample space | 样本空间 | 译;随机实验所有可能完整结果的集合 | [probability-spaces](../probability/probability-spaces/) |
| outcome / sample point | 结果 / 样本点 | 译;Ω 中的一个完整结果,不是由多个结果组成的事件 | [probability-spaces](../probability/probability-spaces/) |
| event | 事件 | 译;σ-代数中的集合,表示可被提问的一组样本点 | [probability-spaces](../probability/probability-spaces/) |
| sigma-algebra | σ-代数 | 保留希腊字母;对补集和可数并封闭的事件集合族 | [probability-spaces](../probability/probability-spaces/) |
| probability measure | 概率测度 | 译;满足非负、单位总量与可数可加性的事件函数 | [probability-spaces](../probability/probability-spaces/) |
| countable additivity | 可数可加性 | 译;两两不相交事件的可数并概率等于概率级数之和 | [probability-spaces](../probability/probability-spaces/) |
| mutually exclusive events | 互斥事件 | 译;交集为空,不能在同一样本点上同时发生 | [probability-spaces](../probability/probability-spaces/) |
| empirical frequency | 经验频率 | 译;有限样本中事件出现的比例,用来估计概率 | [probability-spaces](../probability/probability-spaces/) |

| random variable | 随机变量 | 译;从样本空间到数值空间的可测函数 | [random-variables](../probability/random-variables/) |
| probability mass function / PMF | 概率质量函数 / PMF | 保留缩写;离散随机变量各取值的概率 P(X=x) | [random-variables](../probability/random-variables/) |
| cumulative distribution function / CDF | 累积分布函数 / CDF | 保留缩写;F_X(t)=P(X≤t),统一描述分布 | [random-variables](../probability/random-variables/) |
| probability density function / PDF | 概率密度函数 / PDF | 保留缩写;连续分布中积分给出区间概率,函数值本身不是点概率 | [random-variables](../probability/random-variables/) |
| indicator random variable | 指标随机变量 | 译;事件发生取 1,不发生取 0 的随机变量 | [random-variables](../probability/random-variables/) |
| random vector | 随机向量 | 译;输出为 R^d 向量的随机变量 | [random-variables](../probability/random-variables/) |
| realization / observation | 实现 / 观测值 | 译;随机变量抽样后得到的具体数值 | [random-variables](../probability/random-variables/) |

| discrete distribution | 离散分布 | 译;把概率质量放在至多可数取值上的分布 | [discrete-distributions](../probability/discrete-distributions/) |
| Bernoulli distribution | 伯努利分布 | 译;一次成功/失败试验,取值为 0 或 1 | [discrete-distributions](../probability/discrete-distributions/) |
| binomial distribution | 二项分布 | 译;固定 n 次独立伯努利试验的成功次数 | [discrete-distributions](../probability/discrete-distributions/) |
| geometric distribution | 几何分布 | 译;第一次成功出现的试验编号 | [discrete-distributions](../probability/discrete-distributions/) |
| Poisson distribution | 泊松分布 | 译;固定区间内事件计数,参数 λ 是平均计数 | [discrete-distributions](../probability/discrete-distributions/) |
| categorical distribution | 分类分布 | 译;有限类别的概率向量 | [discrete-distributions](../probability/discrete-distributions/) |
| support | 支持集 | 译;随机变量所有正概率取值组成的集合 | [discrete-distributions](../probability/discrete-distributions/) |
| i.i.d. / independent and identically distributed | 独立同分布 / i.i.d. | 保留缩写;各变量独立且服从同一分布 | [discrete-distributions](../probability/discrete-distributions/) |
| convolution | 卷积 | 译;独立离散变量求和时按配对概率相加 | [discrete-distributions](../probability/discrete-distributions/) |

| continuous distribution | 连续分布 | 译;用密度和积分描述区间概率的分布 | [continuous-distributions](../probability/continuous-distributions/) |
| uniform distribution | 均匀分布 | 译;区间内密度恒定,概率与区间长度成比例 | [continuous-distributions](../probability/continuous-distributions/) |
| exponential distribution | 指数分布 | 译;连续等待时间模型,参数 λ 是速率 | [continuous-distributions](../probability/continuous-distributions/) |
| survival function | 生存函数 / 尾分布函数 | 译;S(t)=P(X>t)=1−F(t) | [continuous-distributions](../probability/continuous-distributions/) |
| mixed distribution | 混合分布 | 译;同时含点质量和连续密度部分的分布 | [continuous-distributions](../probability/continuous-distributions/) |
| likelihood | 似然 | 译;把观测数据固定后视为参数函数的密度或质量乘积 | [continuous-distributions](../probability/continuous-distributions/) |

| Gaussian / normal distribution | 高斯分布 / 正态分布 | 译;N(μ,σ²) 的钟形连续分布 | [gaussian-distribution](../probability/gaussian-distribution/) |
| standard normal distribution | 标准正态分布 | 译;均值 0、方差 1 的高斯分布 | [gaussian-distribution](../probability/gaussian-distribution/) |
| z-score / standard score | z 分数 / 标准分数 | 保留 z;z=(x−μ)/σ,表示离均值多少个标准差 | [gaussian-distribution](../probability/gaussian-distribution/) |
| standard deviation | 标准差 | 译;方差的平方根,与变量同单位 | [gaussian-distribution](../probability/gaussian-distribution/) |
| multivariate Gaussian / multivariate normal | 多元高斯 / 多元正态 | 译;由均值向量和协方差矩阵决定的多维高斯 | [gaussian-distribution](../probability/gaussian-distribution/) |

| joint distribution | 联合分布 | 译;同时描述多个随机变量取值组合的分布 | [joint-distributions](../probability/joint-distributions/) |
| joint PMF | 联合概率质量函数 / 联合 PMF | 保留缩写;离散变量同时取值的概率表 | [joint-distributions](../probability/joint-distributions/) |
| joint density / joint PDF | 联合密度 / 联合 PDF | 保留缩写;连续随机向量在平面或高维区域上的密度 | [joint-distributions](../probability/joint-distributions/) |
| marginal distribution | 边缘分布 | 译;从联合分布对其他坐标求和或积分得到的单变量分布 | [joint-distributions](../probability/joint-distributions/) |
| joint CDF | 联合累积分布函数 / 联合 CDF | 保留缩写;F_{X,Y}(s,t)=P(X≤s,Y≤t) | [joint-distributions](../probability/joint-distributions/) |
| product distribution | 乘积分布 | 译;独立变量联合分布等于边缘分布乘积 | [joint-distributions](../probability/joint-distributions/) |

| conditional probability | 条件概率 | 译;已知 B 发生后 A 的概率 P(A∩B)/P(B) | [marginal-and-conditional](../probability/marginal-and-conditional/) |
| conditional distribution | 条件分布 | 译;给定另一个变量取值后重新归一化的分布 | [marginal-and-conditional](../probability/marginal-and-conditional/) |
| conditional PMF / conditional density | 条件 PMF / 条件密度 | 保留缩写;联合质量或密度除以已知变量的边缘量 | [marginal-and-conditional](../probability/marginal-and-conditional/) |
| chain rule of probability | 概率乘法链式法则 | 译;联合概率拆成一连串条件概率的乘积 | [marginal-and-conditional](../probability/marginal-and-conditional/) |
| law of total probability | 全概率公式 | 译;按互斥分割情形对条件概率加权求和 | [marginal-and-conditional](../probability/marginal-and-conditional/) |
| conditional expectation | 条件期望 | 译;给定信息后的概率加权平均 | [marginal-and-conditional](../probability/marginal-and-conditional/) |

| Bayes' theorem | 贝叶斯定理 | 保留人名;P(A|B)=P(B|A)P(A)/P(B) | [bayes-theorem](../probability/bayes-theorem/) |
| prior probability | 先验概率 | 译;观察当前证据前对假设的概率 | [bayes-theorem](../probability/bayes-theorem/) |
| evidence / marginal likelihood | 证据 / 边缘似然 | 译;对所有假设加权后的观测总体概率 | [bayes-theorem](../probability/bayes-theorem/) |
| posterior probability | 后验概率 | 译;观察证据后对假设更新得到的概率 | [bayes-theorem](../probability/bayes-theorem/) |
| likelihood ratio | 似然比 | 译;两个假设下同一证据概率的比值 | [bayes-theorem](../probability/bayes-theorem/) |
| posterior odds | 后验赔率 | 译;后验概率与其补事件概率之比 | [bayes-theorem](../probability/bayes-theorem/) |
| Naive Bayes | 朴素贝叶斯 | 保留人名;给定类别时各特征条件独立的分类模型 | [bayes-theorem](../probability/bayes-theorem/) |

| independence | 独立性 | 译;联合概率等于边缘概率乘积,知道一个变量不改变另一个变量的分布 | [independence](../probability/independence/) |
| pairwise independence | 两两独立 | 译;任意两个变量独立,但不保证三个或更多变量整体独立 | [independence](../probability/independence/) |
| mutual independence | 相互独立 | 译;任取变量子集时联合概率都能分解为边缘乘积 | [independence](../probability/independence/) |
| conditional independence | 条件独立 | 译;给定第三个变量后联合条件分布分解为两个条件边缘 | [independence](../probability/independence/) |
| uncorrelated | 不相关 | 译;协方差为零的二阶关系,比独立性更弱 | [independence](../probability/independence/) |

| expectation / expected value | 期望 / 数学期望 | 译;按概率或密度对随机变量取加权平均 | [expectation](../probability/expectation/) |
| linearity of expectation | 期望的线性性 | 译;和的期望等于期望之和,不要求变量独立 | [expectation](../probability/expectation/) |
| law of total expectation | 全期望定律 | 译;先求各条件层期望,再按层概率加权 | [expectation](../probability/expectation/) |

| variance | 方差 | 译;围绕均值的平方偏离的概率加权平均 | [variance-and-covariance](../probability/variance-and-covariance/) |
| covariance | 协方差 | 译;两个变量偏离各自均值的乘积的期望 | [variance-and-covariance](../probability/variance-and-covariance/) |
| correlation coefficient | 相关系数 | 译;协方差除以两个标准差,无量纲且位于 −1 与 1 之间 | [variance-and-covariance](../probability/variance-and-covariance/) |
| law of total variance | 全方差定律 | 译;总体方差等于组内条件方差的平均加组间条件均值的方差 | [variance-and-covariance](../probability/variance-and-covariance/) |
| Bessel's correction | Bessel 修正 | 保留人名;样本方差除以 n−1 以获得总体方差的无偏估计 | [variance-and-covariance](../probability/variance-and-covariance/) |

| covariance matrix | 协方差矩阵 | 译;随机向量各坐标两两协方差组成的对称半正定矩阵 | [covariance-matrix](../probability/covariance-matrix/) |
| positive semidefinite / PSD | 半正定 | 译;对所有向量 a 都满足 aᵀΣa≥0 的矩阵性质 | [covariance-matrix](../probability/covariance-matrix/) |
| sample covariance | 样本协方差 | 译;中心化数据矩阵乘其转置并按 n 或 n−1 归一化 | [covariance-matrix](../probability/covariance-matrix/) |

| law of large numbers | 大数定律 | 译;样本平均在合适抽样条件下趋近总体期望 | [law-of-large-numbers](../probability/law-of-large-numbers/) |
| weak law of large numbers | 弱大数定律 | 译;偏离总体均值超过固定阈值的概率趋于 0 | [law-of-large-numbers](../probability/law-of-large-numbers/) |
| strong law of large numbers | 强大数定律 | 译;样本平均沿整条样本路径以概率 1 收敛到总体期望 | [law-of-large-numbers](../probability/law-of-large-numbers/) |
| sample mean | 样本均值 | 译;有限样本取值的算术平均 | [law-of-large-numbers](../probability/law-of-large-numbers/) |
| standard error | 标准误 | 译;估计量抽样分布的标准差,样本均值 iid 时为 σ/√n | [law-of-large-numbers](../probability/law-of-large-numbers/) |
| convergence in probability | 依概率收敛 | 译;固定阈值外的偏离概率趋于 0 | [law-of-large-numbers](../probability/law-of-large-numbers/) |
| almost sure convergence | 几乎处处收敛 / 几乎必然收敛 | 译;整条样本路径以概率 1 收敛 | [law-of-large-numbers](../probability/law-of-large-numbers/) |

| central limit theorem | 中心极限定理 | 译;独立样本的和或平均值经中心化和标准化后趋近标准正态分布 | [central-limit-theorem](../probability/central-limit-theorem/) |
| convergence in distribution | 依分布收敛 | 译;随机变量的分布函数在连续点趋近极限分布函数 | [central-limit-theorem](../probability/central-limit-theorem/) |
| continuity correction | 连续性修正 | 译;用连续正态曲线近似离散概率时把整数边界扩展半格 | [central-limit-theorem](../probability/central-limit-theorem/) |
| Berry–Esseen theorem | Berry–Esseen 定理 | 保留人名;用三阶绝对中心矩给出正态近似的有限样本误差界 | [central-limit-theorem](../probability/central-limit-theorem/) |
| characteristic function | 特征函数 | 译;E[e^{itX}] 形式的变换,用乘积刻画独立和的分布 | [central-limit-theorem](../probability/central-limit-theorem/) |
| confidence interval | 置信区间 | 译;按重复抽样覆盖率构造未知参数的区间估计 | [central-limit-theorem](../probability/central-limit-theorem/) |

| sampling | 抽样 | 译;从目标总体中按抽样设计选出有限观测以估计总体性质 | [sampling](../probability/sampling/) |
| population | 总体 | 译;希望结论适用的全部目标单位或生成分布 | [sampling](../probability/sampling/) |
| estimand | 目标参数 | 译;抽样前定义并希望估计的总体量 | [sampling](../probability/sampling/) |
| estimator | 估计量 | 译;由随机样本计算总体参数的随机函数 | [sampling](../probability/sampling/) |
| estimate | 数值估计 | 译;把观测样本代入估计量后得到的具体数值 | [sampling](../probability/sampling/) |
| simple random sampling | 简单随机抽样 | 译;每个总体单位具有相同入样机会的概率抽样 | [sampling](../probability/sampling/) |
| finite population correction / FPC | 有限总体修正 | 译;无放回抽样时因已观察总体一部分而降低方差的因子 | [sampling](../probability/sampling/) |
| sampling error | 抽样误差 | 译;样本估计量与目标参数之间由随机选样造成的偏离 | [sampling](../probability/sampling/) |
| sampling bias | 抽样偏差 | 译;覆盖不足或入样机会与结果相关造成的系统偏离 | [sampling](../probability/sampling/) |
| stratified sampling | 分层抽样 | 译;先按子群分层再在层内抽样并按总体权重合并 | [sampling](../probability/sampling/) |
| cluster sampling | 整群抽样 | 译;先抽取群组再观察群内单位的抽样设计 | [sampling](../probability/sampling/) |
| design effect | 设计效应 | 译;抽样设计相对独立简单随机抽样放大方差的倍数 | [sampling](../probability/sampling/) |
| Horvitz–Thompson estimator | Horvitz–Thompson 估计量 | 保留人名;按单位入样概率倒数加权的总体总量估计量 | [sampling](../probability/sampling/) |
| bootstrap | Bootstrap / 自助法 | 保留方法名;从经验分布有放回重抽以近似统计量抽样分布 | [sampling](../probability/sampling/) |
| nonresponse bias | 非响应偏差 | 译;被抽中但未回答的单位与回答单位系统不同造成的偏差 | [sampling](../probability/sampling/) |

| change of variables | 变量变换 | 译;把随机变量映射到新坐标并按长度或体积伸缩修正密度 | [change-of-variables](../probability/change-of-variables/) |
| change-of-variables formula | 变量变换公式 | 译;用逆函数导数或逆映射 Jacobian 行列式转换密度 | [change-of-variables](../probability/change-of-variables/) |
| many-to-one transformation | 多对一变换 | 译;多个原始取值映射到同一输出并合并各分支概率 | [change-of-variables](../probability/change-of-variables/) |
| polar coordinates | 极坐标 | 译;用半径与角度表示平面位置的坐标系统 | [change-of-variables](../probability/change-of-variables/) |
| area element | 面积元素 | 译;局部区域的面积微元,极坐标中为 r dr dθ | [change-of-variables](../probability/change-of-variables/) |
| log-Jacobian determinant | 对数 Jacobian 行列式 | 保留 Jacobian;生成模型密度变换中的 log-det 体积修正 | [change-of-variables](../probability/change-of-variables/) |

| maximum likelihood estimation / MLE | 最大似然估计 / MLE | 译并保留缩写;固定观测后选择使似然最大的参数 | [maximum-likelihood](../probability/maximum-likelihood/) |
| log-likelihood | 对数似然 | 译;似然取对数后把独立样本的概率乘积变成求和 | [maximum-likelihood](../probability/maximum-likelihood/) |
| negative log-likelihood / NLL | 负对数似然 / NLL | 译并保留缩写;最大化似然等价的最小化损失 | [maximum-likelihood](../probability/maximum-likelihood/) |
| score function | 得分函数 | 译;对数似然关于参数的一阶导数 | [maximum-likelihood](../probability/maximum-likelihood/) |
| Fisher information | Fisher 信息 | 保留人名;衡量观测对参数局部曲率和估计精度的信息量 | [maximum-likelihood](../probability/maximum-likelihood/) |
| sufficient statistic | 充分统计量 | 译;在给定模型下保留样本关于参数似然信息的统计量 | [maximum-likelihood](../probability/maximum-likelihood/) |
| maximum a posteriori / MAP | 最大后验估计 / MAP | 译并保留缩写;最大化似然与先验乘积的参数估计 | [maximum-a-posteriori](../probability/maximum-a-posteriori/) |
| posterior mode | 后验众数 | 译;后验密度达到最大值的参数点,即 MAP | [maximum-a-posteriori](../probability/maximum-a-posteriori/) |
| conjugate prior | 共轭先验 | 译;更新后仍属于同一分布族的先验 | [maximum-a-posteriori](../probability/maximum-a-posteriori/) |
| posterior predictive distribution | 后验预测分布 | 译;对参数后验加权积分得到未来观测的预测分布 | [maximum-a-posteriori](../probability/maximum-a-posteriori/) |
| Laplace approximation | Laplace 近似 | 保留人名;在后验峰附近用 Hessian 逆矩阵近似高斯后验 | [maximum-a-posteriori](../probability/maximum-a-posteriori/) |
| credible interval | 可信区间 | 译;后验概率覆盖指定比例的参数区间 | [maximum-a-posteriori](../probability/maximum-a-posteriori/) |
| identifiability | 可识别性 | 译;不同参数是否对应不同的观测分布 | [maximum-likelihood](../probability/maximum-likelihood/) |
| exponential family | 指数族 | 译;可写成自然参数与充分统计量内积减对数配分函数的分布族 | [exponential-family](../probability/exponential-family/) |
| natural parameter | 自然参数 | 译;指数族中与充分统计量线性配对的参数 | [exponential-family](../probability/exponential-family/) |
| log-partition function | 对数配分函数 | 译;保证指数族密度归一化且其导数生成矩的函数 | [exponential-family](../probability/exponential-family/) |
| canonical link | 规范链接函数 | 译;把响应均值映射到指数族自然参数的链接函数 | [exponential-family](../probability/exponential-family/) |
| Bregman divergence | Bregman 差 | 保留人名;由凸函数及其一阶近似定义的非负差值 | [exponential-family](../probability/exponential-family/) |
| self-information / surprisal | 自信息量 / 惊奇度 | 译并保留英文;单个结果的负对数概率 | [information-and-surprise](../information-theory/information-and-surprise/) |
| bit | 比特 | 译;以 2 为底的自信息量单位 | [information-and-surprise](../information-theory/information-and-surprise/) |
| nat | 奈特 | 译;以自然对数为底的自信息量单位 | [information-and-surprise](../information-theory/information-and-surprise/) |
| prefix code | 前缀码 | 译;任一码字都不是另一码字前缀的可无歧义编码 | [information-and-surprise](../information-theory/information-and-surprise/) |
| Kraft inequality | Kraft 不等式 | 保留人名;前缀码长度满足的幂和约束 | [information-and-surprise](../information-theory/information-and-surprise/) |
| entropy | 熵 | 译;自信息量的期望,衡量分布的平均不确定性 | [entropy](../information-theory/entropy/) |
| joint entropy | 联合熵 | 译;联合结果的平均自信息量 | [entropy](../information-theory/entropy/) |
| conditional entropy | 条件熵 | 译;已知一个变量后另一个变量剩余的平均不确定性,按条件变量的概率加权 | [conditional-entropy](../information-theory/conditional-entropy/) |
| chain rule of entropy | 熵的链式法则 | 译;联合熵等于先验熵加给定前项后的条件熵 | [conditional-entropy](../information-theory/conditional-entropy/) |
| conditional cross-entropy | 条件交叉熵 | 译;模型按输入条件预测标签时的平均负对数概率 | [conditional-entropy](../information-theory/conditional-entropy/) |
| mutual information | 互信息 | 译;观察一个变量后另一个变量平均减少的不确定性,也等于联合分布相对独立基线的 KL 散度 | [mutual-information](../information-theory/mutual-information/) |
| pointwise mutual information / PMI | 点互信息 / PMI | 译并保留缩写;单个联合结果相对独立基线的对数概率比 | [mutual-information](../information-theory/mutual-information/) |
| conditional mutual information | 条件互信息 | 译;在已知第三个变量后两个变量仍共享的信息量 | [mutual-information](../information-theory/mutual-information/) |
| Fano's inequality | Fano 不等式 | 保留人名;用条件熵约束分类错误概率的界 | [mutual-information](../information-theory/mutual-information/) |
| information bottleneck | 信息瓶颈 | 译;压缩输入信息同时保留任务相关信息的表示学习框架 | [mutual-information](../information-theory/mutual-information/) |
| perplexity / PPL | 困惑度 / PPL | 译并保留缩写;平均 token 负对数似然取指数后的有效分支数 | [perplexity](../information-theory/perplexity/) |
| bits per token | 每 token 比特数 | 译;以 2 为底的平均 token 负对数似然 | [perplexity](../information-theory/perplexity/) |
| bits per character / BPC | 每字符比特数 / BPC | 译并保留缩写;跨 tokenizer 比较时按字符或字节归一化的信息量 | [perplexity](../information-theory/perplexity/) |
| teacher forcing | teacher forcing / 教师强制 | 保留常用英文;用真实前缀而非模型生成前缀计算 next-token 损失 | [teacher-forcing](../rnn-lstm/teacher-forcing/) |
| optimization problem | 优化问题 | 译;在可行选择中寻找目标函数最小或最大的元素 | [optimization-problems](../optimization-theory/optimization-problems/) |
| decision variable | 决策变量 | 译;优化过程中允许改变的未知量 | [optimization-problems](../optimization-theory/optimization-problems/) |
| objective function | 目标函数 | 译;定义候选解优劣并被最小化或最大化的函数 | [optimization-problems](../optimization-theory/optimization-problems/) |
| feasible set | 可行集 | 译;满足全部等式和不等式约束的候选点集合 | [optimization-problems](../optimization-theory/optimization-problems/) |
| argmin | argmin / 最优解集合 | 保留常用记号;达到最小目标值的全部点 | [optimization-problems](../optimization-theory/optimization-problems/) |
| infimum | 下确界 | 译;所有函数值的最大下界,不一定被某个点达到 | [optimization-problems](../optimization-theory/optimization-problems/) |
| empirical risk / population risk | 经验风险 / 总体风险 | 译;分别是样本平均损失和数据分布下的期望损失 | [optimization-problems](../optimization-theory/optimization-problems/) |
| empirical risk minimization / ERM | 经验风险最小化 / ERM | 译并保留缩写;在假设空间中选择样本平均损失最小的规则 | [empirical-risk-minimization](../learning-framework/empirical-risk-minimization/) |
| empirical risk minimizer | 经验风险最小规则 | 译;达到假设空间内最小经验风险的候选规则 | [empirical-risk-minimization](../learning-framework/empirical-risk-minimization/) |
| approximate ERM | 近似经验风险最小化 | 译;经验风险只比空间内最优值高有限优化误差的选择 | [empirical-risk-minimization](../learning-framework/empirical-risk-minimization/) |
| regularized empirical risk | 正则化经验风险 | 译;样本损失加上复杂度惩罚后的训练目标 | [empirical-risk-minimization](../learning-framework/empirical-risk-minimization/) |
| weighted empirical risk / sample weight | 加权经验风险 / 样本权重 | 译;按样本错误代价对经验损失重新加权的目标 | [empirical-risk-minimization](../learning-framework/empirical-risk-minimization/) |
| penalty method | 罚函数法 | 译;把约束违反程度加进目标函数的近似约束方法 | [optimization-problems](../optimization-theory/optimization-problems/) |
| convex set | 凸集 | 译;包含任意两点连线段的集合 | [convex-sets-and-functions](../optimization-theory/convex-sets-and-functions/) |
| convex function | 凸函数 | 译;线段中间的函数值不高于两端函数值的同权插值 | [convex-sets-and-functions](../optimization-theory/convex-sets-and-functions/) |
| convex hull | 凸包 | 译;包含给定集合的最小凸集,由有限凸组合组成 | [convex-sets-and-functions](../optimization-theory/convex-sets-and-functions/) |
| epigraph | 上图集 | 译;函数图像上方且在定义域内的点集 | [convex-sets-and-functions](../optimization-theory/convex-sets-and-functions/) |
| Jensen's inequality | Jensen 不等式 | 保留人名;凸函数作用于平均值不超过函数值的加权平均 | [convex-sets-and-functions](../optimization-theory/convex-sets-and-functions/) |
| strongly convex | 强凸 | 译;带有统一二次余量的凸性,可排除不同的多个最优解 | [convex-sets-and-functions](../optimization-theory/convex-sets-and-functions/) |
| local minimum / local minimizer | 局部最小值 / 局部最小点 | 译;在可行域某个邻域内不低于该点的函数值或点 | [local-and-global-minima](../optimization-theory/local-and-global-minima/) |
| global minimum / global minimizer | 全局最小值 / 全局最小点 | 译;在整个可行域内不高于其他点的函数值或点 | [local-and-global-minima](../optimization-theory/local-and-global-minima/) |
| stationary point | 驻点 | 译;可微无约束情形下梯度为零的点,不一定是极值 | [local-and-global-minima](../optimization-theory/local-and-global-minima/) |
| saddle point | 鞍点 | 译;沿不同方向分别上升和下降的驻点 | [local-and-global-minima](../optimization-theory/local-and-global-minima/) |
| first-order optimality condition | 一阶最优性条件 | 译;用梯度、可行方向或乘子表达局部最优所需的一阶关系 | [first-order-optimality](../optimization-theory/first-order-optimality/) |
| feasible direction | 可行方向 | 译;从当前可行点出发仍能保持约束的局部移动方向 | [first-order-optimality](../optimization-theory/first-order-optimality/) |
| Lagrangian / Lagrange multiplier | 拉格朗日函数 / 拉格朗日乘子 | 译;把约束以乘子加进目标并用驻点关系求解 | [first-order-optimality](../optimization-theory/first-order-optimality/) |
| Karush–Kuhn–Tucker conditions / KKT conditions | Karush–Kuhn–Tucker 条件 / KKT 条件 | 保留人名与缩写;包含可行性、驻点、乘子符号和互补松弛的约束一阶条件 | [first-order-optimality](../optimization-theory/first-order-optimality/) |
| complementary slackness | 互补松弛 | 译;每个不等式约束的乘子与约束余量乘积为零 | [first-order-optimality](../optimization-theory/first-order-optimality/) |
| constraint qualification | 约束资格条件 | 译;保证约束梯度具有足够独立性以推出 KKT 必要性的正则条件 | [first-order-optimality](../optimization-theory/first-order-optimality/) |
| subgradient | 次梯度 | 译;在不可微点给出全局支撑下界的斜率或向量 | [first-order-optimality](../optimization-theory/first-order-optimality/) |
| normal cone | 法锥 | 译;与凸可行集内所有可行位移内积不为正的法向量集合 | [first-order-optimality](../optimization-theory/first-order-optimality/) |
| second-order optimality condition | 二阶最优性条件 | 译;用 Hessian 或拉格朗日 Hessian 在相关方向上的二次型判断局部曲率 | [second-order-optimality](../optimization-theory/second-order-optimality/) |
| second-order sufficient condition | 二阶充分条件 | 译;相关方向上的二次型严格为正时保证严格局部最小的条件 | [second-order-optimality](../optimization-theory/second-order-optimality/) |
| tangent space / tangent direction | 切空间 / 切向方向 | 译;一阶近似下沿约束曲面仍可移动的方向集合 | [second-order-optimality](../optimization-theory/second-order-optimality/) |
| critical cone | 临界锥 | 译;满足线性化可行性且一阶目标变化为零的约束方向集合 | [second-order-optimality](../optimization-theory/second-order-optimality/) |
| curvature | 曲率 | 译;沿方向的二阶变化率,由 Hessian 二次型给出 | [second-order-optimality](../optimization-theory/second-order-optimality/) |
| trust region | 信赖域 | 译;限制局部二次模型步长范围的区域 | [second-order-optimality](../optimization-theory/second-order-optimality/) |
| gradient descent | 梯度下降 | 译;沿当前目标梯度反方向更新参数的一阶迭代方法 | [gradient-descent](../training-nn/gradient-descent/) |
| learning rate / step size | 学习率 / 步长 | 译;梯度下降中乘在梯度前的正标量 $\eta$,决定每次移动的尺度 | [gradient-descent](../training-nn/gradient-descent/) |
| parameter update | 参数更新 | 译;用当前梯度和学习率把参数从 $\boldsymbol\theta_k$ 移到 $\boldsymbol\theta_{k+1}$ 的操作 | [gradient-descent](../training-nn/gradient-descent/) |
| effective learning rate | 有效学习率 | 译;目标缩放、参数单位或预条件共同作用后实际乘在某方向梯度上的步长尺度 | [gradient-descent](../training-nn/gradient-descent/) |
| learning-rate schedule | 学习率调度 | 译;把每次参数更新使用的学习率写成 step、epoch 或验证反馈的函数 | [learning-rate-schedules](../training-nn/learning-rate-schedules/) |
| warmup | warmup / 预热 | 保留常用名称并译;训练早期从较小学习率逐步升到目标学习率的阶段 | [learning-rate-schedules](../training-nn/learning-rate-schedules/) |
| cosine decay | 余弦衰减 | 译;用余弦函数把学习率从峰值平滑降到终点的开环调度 | [learning-rate-schedules](../training-nn/learning-rate-schedules/) |
| plateau scheduler | plateau 调度器 | 保留常用名称并译;根据验证指标在连续无改善时降低学习率的反馈调度 | [learning-rate-schedules](../training-nn/learning-rate-schedules/) |
| patience | patience / 等待轮数 | 保留常用名称并译;指标未达到最小改善量时允许继续检查的评估事件数 | [early-stopping](../evaluation-and-generalization/early-stopping/) |
| full-batch gradient descent | 全批量梯度下降 | 译;每次用完整数据集的确定性平均梯度更新参数的梯度法 | [gradient-descent](../training-nn/gradient-descent/) |
| projected gradient descent | 投影梯度下降 | 译;先按负梯度移动,再把参数投影回可行集的约束更新方法 | [gradient-descent](../training-nn/gradient-descent/) |
| objective scaling | 目标缩放 | 译;把损失、归约或单位乘以常数的变换,会按同样比例改变梯度和学习率含义 | [gradient-descent](../training-nn/gradient-descent/) |
| update ratio | 更新比率 | 译;参数移动范数相对参数范数的比例,用于监控步长是否过小或异常变大 | [gradient-descent](../training-nn/gradient-descent/) |
| L-smooth / Lipschitz-continuous gradient | L-光滑 / Lipschitz 连续梯度 | 译;梯度变化满足 $\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|$ 的函数条件 | [gradient-descent-theory](../optimization-theory/gradient-descent-theory/) |
| descent lemma | 下降引理 | 译;L-光滑函数的二次上界,给出一步更新的函数值变化 | [gradient-descent-theory](../optimization-theory/gradient-descent-theory/) |
| epsilon-stationary point | $\varepsilon$-驻点 | 译;梯度范数不超过给定容差 $\varepsilon$ 的点 | [gradient-descent-theory](../optimization-theory/gradient-descent-theory/) |
| Polyak–Łojasiewicz inequality / PL inequality | Polyak–Łojasiewicz 不等式 / PL 不等式 | 保留人名与缩写;用梯度平方下界函数值差,可在非凸情形给出线性函数值收敛 | [gradient-descent-theory](../optimization-theory/gradient-descent-theory/) |
| Armijo condition | Armijo 条件 | 保留人名;要求实际目标下降达到梯度方向预测下降的一定比例 | [gradient-descent-theory](../optimization-theory/gradient-descent-theory/) |
| backtracking line search | 回溯线搜索 | 译;从较大步长开始反复缩小直到满足下降条件 | [gradient-descent-theory](../optimization-theory/gradient-descent-theory/) |
| stochastic gradient descent / SGD | 随机梯度下降 / SGD | 译并保留缩写;用随机样本或小批次梯度近似总体梯度的迭代方法 | [minibatch-sgd](../training-nn/minibatch-sgd/) |
| mini-batch | 小批次 | 译;一次更新中抽取并平均的样本子集 | [minibatch-sgd](../training-nn/minibatch-sgd/) |
| unbiased stochastic gradient | 无偏随机梯度 | 译;给定当前参数时,抽样梯度期望等于完整经验风险梯度的估计量 | [minibatch-sgd](../training-nn/minibatch-sgd/) |
| gradient noise / variance | 梯度噪声 / 方差 | 译;随机梯度与完整梯度的差及其二阶大小,批次增大时通常下降 | [minibatch-sgd](../training-nn/minibatch-sgd/) |
| noise floor | 噪声地板 | 译;固定学习率下随机梯度使期望误差只能稳定在的非零尺度 | [stochastic-gradient-descent-theory](../optimization-theory/stochastic-gradient-descent-theory/) |
| batch size | 批次大小 | 译;一次随机梯度平均包含的样本数,通常记为 $B$ | [minibatch-sgd](../training-nn/minibatch-sgd/) |
| epoch | epoch / 数据轮次 | 保留常用名称并译;训练集被完整访问一次的计数口径,不等于参数更新次数 | [minibatch-sgd](../training-nn/minibatch-sgd/) |
| shuffle | shuffle / 随机打乱 | 保留常用名称并译;每个训练 epoch 对索引重新排列以减少顺序相关性的采样操作 | [minibatch-sgd](../training-nn/minibatch-sgd/) |
| drop-last | 丢弃尾批次 | 译;当样本数不能整除 batch size 时舍弃最后不足一批的样本的 loader 选项 | [minibatch-sgd](../training-nn/minibatch-sgd/) |
| effective batch size | 有效批次大小 | 译;考虑设备数、micro-batch 和梯度累积后一次参数更新实际平均的样本数 | [minibatch-sgd](../training-nn/minibatch-sgd/) |
| gradient accumulation steps | 梯度累积步数 | 译;在一次参数更新前累积多少个 micro-batch 梯度的计数 | [minibatch-sgd](../training-nn/minibatch-sgd/) |
| optimizer | 优化器 | 译;接收有效梯度与内部状态并计算参数位移的更新算法 | [optimizers](../training-nn/optimizers/) |
| optimizer state | 优化器状态 | 译;动量、平方梯度统计、矩和 step 等会影响后续更新的持久状态 | [optimizers](../training-nn/optimizers/) |
| Robbins–Monro conditions | Robbins–Monro 条件 | 保留人名;常见递减学习率条件 $\sum_k\eta_k=\infty$ 且 $\sum_k\eta_k^2<\infty$ | [stochastic-gradient-descent-theory](../optimization-theory/stochastic-gradient-descent-theory/) |
| momentum / heavy-ball method | 动量 / 重球法 | 译;把历史梯度累积到速度缓冲区后再更新参数的一阶加速方法 | [momentum-and-nesterov](../training-nn/momentum-and-nesterov/) |
| velocity / momentum buffer | 速度 / 动量缓冲区 | 译;保存历史梯度加权和的辅助状态向量 | [momentum-and-nesterov](../training-nn/momentum-and-nesterov/) |
| momentum coefficient | 动量系数 | 译;记为 $\beta$,控制上一时刻缓冲区保留的比例 | [momentum-and-nesterov](../training-nn/momentum-and-nesterov/) |
| Nesterov accelerated gradient / Nesterov momentum | Nesterov 加速梯度 / Nesterov 动量 | 保留人名;先沿历史方向前瞻再在前瞻点计算梯度的动量变体 | [momentum-and-nesterov](../training-nn/momentum-and-nesterov/) |
| acceleration | 加速 | 译;相对基础梯度法改善理论收敛阶或条件数依赖,不等于每一步目标下降 | [momentum-theory](../optimization-theory/momentum-theory/) |
| effective memory horizon | 有效记忆长度 | 译;几何衰减历史权重的数量级,动量系数为 $\beta$ 时约为 $1/(1-\beta)$ | [momentum-and-nesterov](../training-nn/momentum-and-nesterov/) |
| lookahead update | 前瞻更新 | 译;先沿历史位移得到前瞻参数,再在前瞻点计算梯度的更新步骤 | [momentum-and-nesterov](../training-nn/momentum-and-nesterov/) |
| coupled weight decay | 耦合权重衰减 | 译;把参数衰减项加入梯度并让它进入动量缓冲区的更新方式 | [momentum-and-nesterov](../training-nn/momentum-and-nesterov/) |
| adaptive learning rate | 自适应学习率 | 译;根据各坐标历史梯度尺度调整有效步长的方法 | [adaptive-learning-rates](../optimization-theory/adaptive-learning-rates/) |
| preconditioner | 预条件器 | 译;更新前对梯度做尺度变换的矩阵或逐坐标因子 | [adaptive-learning-rates](../optimization-theory/adaptive-learning-rates/) |
| AdaGrad | AdaGrad | 保留算法名;累积平方梯度并按坐标缩放更新 | [adaptive-learning-rates](../optimization-theory/adaptive-learning-rates/) |
| RMSProp | RMSProp | 保留算法名;用平方梯度的指数滑动平均作分母 | [adaptive-learning-rates](../optimization-theory/adaptive-learning-rates/) |
| exponential moving average | 指数滑动平均 | 译;当前值与历史平均按指数衰减权重组合 | [adaptive-learning-rates](../optimization-theory/adaptive-learning-rates/) |
| first moment / second moment | 一阶矩 / 二阶矩 | 译;Adam 中梯度平均与平方梯度平均的统计量 | [adaptive-learning-rates](../optimization-theory/adaptive-learning-rates/) |
| bias correction | 偏置修正 | 译;补偿从零初始化的指数平均在早期偏小 | [adaptive-learning-rates](../optimization-theory/adaptive-learning-rates/) |
| Adam | Adam | 保留算法名;结合一阶方向平均和二阶尺度平均的自适应方法 | [adaptive-learning-rates](../optimization-theory/adaptive-learning-rates/) |
| AdamW | AdamW | 保留算法名;把参数权重衰减从 Adam 的梯度矩更新中解耦 | [adam](../training-nn/adam/) |
| optimizer epsilon | 优化器 $arepsilon$ | 译;加入自适应分母以避免除零,其放在根号内外会影响极小尺度下的更新 | [adam](../training-nn/adam/) |
| decoupled weight decay | 解耦权重衰减 | 译;把参数收缩从自适应梯度更新中分离出来 | [adaptive-learning-rates](../optimization-theory/adaptive-learning-rates/) |
| AMSGrad | AMSGrad | 保留算法名;维护单调二阶分母以改善部分收敛分析 | [adaptive-learning-rates](../optimization-theory/adaptive-learning-rates/) |
| second-order method | 二阶方法 | 译;使用 Hessian 或曲率近似构造局部二次模型更新 | [second-order-methods](../optimization-theory/second-order-methods/) |
| Newton method / Newton step | Newton 方法 / Newton 步 | 保留人名;解 Hessian 线性系统 $Hp=-g$ 的二阶更新 | [second-order-methods](../optimization-theory/second-order-methods/) |
| local quadratic model | 局部二次模型 | 译;用梯度线性项和 Hessian 二次项近似附近目标 | [second-order-methods](../optimization-theory/second-order-methods/) |
| damping / damped Newton | 阻尼 / 阻尼 Newton | 译;给 Hessian 加 $\lambda I$ 或缩短步长以提高稳定性 | [second-order-methods](../optimization-theory/second-order-methods/) |
| line search | 线搜索 | 译;沿候选方向选择满足下降条件的步长 | [second-order-methods](../optimization-theory/second-order-methods/) |
| quasi-Newton method | 拟 Newton 方法 | 译;用梯度变化的割线信息近似 Hessian 或其逆 | [second-order-methods](../optimization-theory/second-order-methods/) |
| secant condition | 割线条件 | 译;要求近似 Hessian 满足 $B_{k+1}s_k=y_k$ | [second-order-methods](../optimization-theory/second-order-methods/) |
| BFGS | BFGS | 保留缩写;保持正定性的拟 Newton 矩阵更新方法 | [second-order-methods](../optimization-theory/second-order-methods/) |
| L-BFGS | L-BFGS | 保留缩写;只保存最近割线对的低存储拟 Newton 方法 | [second-order-methods](../optimization-theory/second-order-methods/) |
| negative curvature | 负曲率 | 译;Hessian 存在负特征值的方向,二次模型沿其不是局部最小 | [second-order-methods](../optimization-theory/second-order-methods/) |
| inexact Newton method | 不完全 Newton 方法 | 译;只把 Newton 线性系统解到随优化阶段变化的残差精度 | [second-order-methods](../optimization-theory/second-order-methods/) |
| constrained optimization | 约束优化 | 译;在等式或不等式可行域内最小化目标函数 | [constrained-optimization](../optimization-theory/constrained-optimization/) |
| feasible set / feasible point | 可行集 / 可行点 | 译;满足全部约束的点集及其中的点 | [constrained-optimization](../optimization-theory/constrained-optimization/) |
| projected gradient | 投影梯度 | 译;先做梯度步再投影回可行集合的更新 | [constrained-optimization](../optimization-theory/constrained-optimization/) |
| projection onto a convex set | 凸集投影 | 译;到凸集合最近的点,投影结果唯一 | [constrained-optimization](../optimization-theory/constrained-optimization/) |
| augmented Lagrangian | 增广拉格朗日 | 译;结合乘子项和二次约束残差项的函数 | [constrained-optimization](../optimization-theory/constrained-optimization/) |
| barrier method / log barrier | 障碍方法 / 对数障碍 | 译;在可行域内部用边界发散项逼近约束最优解 | [constrained-optimization](../optimization-theory/constrained-optimization/) |
| active-set method | 活动集方法 | 译;把当前猜测的边界不等式当作等式处理 | [constrained-optimization](../optimization-theory/constrained-optimization/) |
| alternating direction method of multipliers / ADMM | 交替方向乘子法 / ADMM | 保留缩写;分裂变量并交替求解增广拉格朗日子问题 | [constrained-optimization](../optimization-theory/constrained-optimization/) |
| primal residual / dual residual | 原始残差 / 对偶残差 | 译;分别衡量约束不满足程度和乘子更新的一致性 | [constrained-optimization](../optimization-theory/constrained-optimization/) |
| primal problem / primal variable | 原问题 / 原变量 | 译;直接优化目标函数的约束问题及其参数 | [duality](../optimization-theory/duality/) |
| dual problem / dual variable | 对偶问题 / 对偶变量 | 译;最大化拉格朗日对偶函数及其乘子 | [duality](../optimization-theory/duality/) |
| dual function | 对偶函数 | 译;固定乘子后对原变量取下确界得到的函数 | [duality](../optimization-theory/duality/) |
| weak duality / strong duality | 弱对偶 / 强对偶 | 译;对偶值不超过原值 / 两者最优值相等 | [duality](../optimization-theory/duality/) |
| duality gap | 对偶间隙 | 译;原问题最优值与对偶问题最优值之差 | [duality](../optimization-theory/duality/) |
| Slater condition | Slater 条件 | 保留人名;凸不等式约束存在严格可行点的充分条件 | [duality](../optimization-theory/duality/) |
| dual ascent | 对偶上升 | 译;沿对偶函数次梯度方向更新乘子的最大化方法 | [duality](../optimization-theory/duality/) |
| Fenchel conjugate / convex conjugate | Fenchel 共轭 / 凸共轭 | 保留人名;对线性项与函数差取上确界得到的共轭函数 | [duality](../optimization-theory/duality/) |
| Fenchel–Young inequality | Fenchel–Young 不等式 | 保留人名;函数与共轭之和不小于变量内积 | [duality](../optimization-theory/duality/) |
| strict saddle point | 严格鞍点 | 译;Hessian 同时有正特征值和负特征值的驻点 | [saddle-points](../optimization-theory/saddle-points/) |
| degenerate saddle | 退化鞍点 | 译;Hessian 有零特征值但高阶项仍产生上升和下降方向的鞍点 | [saddle-points](../optimization-theory/saddle-points/) |
| negative curvature direction | 负曲率方向 | 译;二阶方向导数为负的方向 | [saddle-points](../optimization-theory/saddle-points/) |
| stable / unstable subspace | 稳定/不稳定子空间 | 译;线性迭代中误差衰减/放大的特征方向张成的子空间 | [saddle-points](../optimization-theory/saddle-points/) |
| descent-ascent | 下降—上升 | 译;min-max 中对最小化变量下降、最大化变量上升的迭代 | [saddle-points](../optimization-theory/saddle-points/) |
| min-max optimization | min-max 优化 | 保留常用形式;对一组变量最小化、另一组变量最大化 | [saddle-points](../optimization-theory/saddle-points/) |
| loss landscape / loss surface | 损失景观 / 损失曲面 | 译;把参数映射到训练或验证损失的高维函数及其低维切片 | [loss-landscapes](../optimization-theory/loss-landscapes/) |
| valley | 谷 | 译;横向曲率较大而沿谷底方向变化较慢的局部景观结构 | [loss-landscapes](../optimization-theory/loss-landscapes/) |
| flat direction | 平坦方向 | 译;局部方向曲率为零或很小,参数变化对损失影响弱 | [loss-landscapes](../optimization-theory/loss-landscapes/) |
| parameter symmetry / permutation symmetry | 参数对称 / 置换对称 | 译;不同参数表示同一函数的等价关系,如隐藏单元交换 | [loss-landscapes](../optimization-theory/loss-landscapes/) |
| loss barrier / barrier height | 损失障碍 / 障碍高度 | 译;连接两个参数点的指定路径上超过端点损失的最高增量 | [loss-landscapes](../optimization-theory/loss-landscapes/) |
| sharpness | 锐度 | 译;局部最大曲率或固定扰动下损失上升幅度的指标 | [loss-landscapes](../optimization-theory/loss-landscapes/) |
| reparameterization | 重参数化 | 译;用新坐标表示同一模型函数,会改变参数空间中的曲率数值 | [loss-landscapes](../optimization-theory/loss-landscapes/) |
| directional curvature | 方向曲率 | 译;单位方向上的二阶导数 $\boldsymbol u^\mathsf T H\boldsymbol u$ | [curvature-and-conditioning](../optimization-theory/curvature-and-conditioning/) |
| curvature spectrum | 曲率谱 | 译;Hessian 的特征值及其对应方向,描述局部曲率尺度 | [curvature-and-conditioning](../optimization-theory/curvature-and-conditioning/) |
| optimization condition number | 优化条件数 | 译;正定曲率中最大与最小特征值之比,决定一阶法的尺度差异 | [curvature-and-conditioning](../optimization-theory/curvature-and-conditioning/) |
| optimal fixed step size | 最佳固定步长 | 译;正定二次目标上使最坏特征方向收缩因子最小的常数步长 | [curvature-and-conditioning](../optimization-theory/curvature-and-conditioning/) |
| convergence factor | 收缩因子 | 译;每轮误差范数最多保留的比例,线性收敛中小于 1 | [curvature-and-conditioning](../optimization-theory/curvature-and-conditioning/) |
| anisotropy | 各向异性 | 译;不同方向的曲率或尺度明显不同的性质 | [curvature-and-conditioning](../optimization-theory/curvature-and-conditioning/) |
| coordinate scaling | 坐标缩放 | 译;改变参数或特征单位以平衡不同方向尺度的变换 | [curvature-and-conditioning](../optimization-theory/curvature-and-conditioning/) |
| gradient flow | 梯度流 | 译;连续时间极限 $\mathrm d\boldsymbol\theta/\mathrm dt=-\nabla f$ | [curvature-and-conditioning](../optimization-theory/curvature-and-conditioning/) |
| entropy rate | 熵率 | 译;长随机序列每个符号的极限平均熵 | [entropy](../information-theory/entropy/) |
| differential entropy | 微分熵 | 译;连续密度的积分形式熵,依赖坐标和单位 | [entropy](../information-theory/entropy/) |
| maximum entropy principle | 最大熵原理 | 译;在满足已知约束时选择额外结构最少的分布 | [entropy](../information-theory/entropy/) |
| binary entropy | 二元熵 | 译;Bernoulli 概率参数对应的熵函数 | [entropy](../information-theory/entropy/) |
| cross-entropy | 交叉熵 | 译;用模型分布为真实分布产生的结果编码的平均负对数概率 | [cross-entropy](../information-theory/cross-entropy/) |
| binary cross-entropy | 二元交叉熵 | 译;Bernoulli 输出的负对数似然损失 | [classification-losses](../training-nn/classification-losses/) |
| softmax cross-entropy | softmax 交叉熵 | 译;logits 经 softmax 后的多分类交叉熵损失 | [classification-losses](../training-nn/classification-losses/) |
| log-sum-exp | log-sum-exp | 保留术语;稳定计算指数和对数的函数 | [cross-entropy](../information-theory/cross-entropy/) |
| label smoothing | 标签平滑 | 译;把 one-hot 标签混入均匀分布以软化目标 | [classification-losses](../training-nn/classification-losses/) |
| soft label / soft target | 软标签 / 软目标 | 译;用概率分布而不是单个类别指示表示监督目标 | [classification-losses](../training-nn/classification-losses/) |
| smoothing coefficient / epsilon | 平滑系数 / epsilon | 译;控制 hard target 与参考分布混合比例的 $ε$,必须注明是否包含正确类 | [label-smoothing](../training-nn/label-smoothing/) |
| uniform-all / uniform-other smoothing | 全类均匀 / 错类均匀平滑 | 译;前者把质量分给包括正确类的全部类别,后者只分给错误类别 | [label-smoothing](../training-nn/label-smoothing/) |
| prior-aware label smoothing | 先验感知标签平滑 | 译;把 one-hot 目标向估计的类别先验而不是均匀分布收缩 | [label-smoothing](../training-nn/label-smoothing/) |
| confidence penalty | 置信度惩罚 | 译;在训练损失中直接加入模型分布熵相关项以抑制过度尖锐,区别于修改目标标签 | [label-smoothing](../training-nn/label-smoothing/) |
| multilabel classification | 多标签分类 | 译;一个样本可以同时拥有多个独立标签的分类任务 | [classification-losses](../training-nn/classification-losses/) |
| focal loss | focal loss / 聚焦损失 | 保留常用名称并译;用 $(1-p_t)^\gamma$ 降低容易样本相对权重的分类损失 | [classification-losses](../training-nn/classification-losses/) |
| class weight | 类别权重 | 译;按类别改变损失与梯度贡献的加权策略,可能改变输出的概率解释 | [classification-losses](../training-nn/classification-losses/) |
| cost-sensitive classification | 代价敏感分类 | 译;把不同错误的业务代价写入权重或决策阈值的分类设定 | [classification-losses](../training-nn/classification-losses/) |
| proper scoring rule | 严格合宜评分规则 | 译;期望评分在诚实报告真实概率处分数最优的规则 | [cross-entropy](../information-theory/cross-entropy/) |
| KL divergence / relative entropy | KL 散度 / 相对熵 | 译并保留缩写;真实分布用模型分布编码时的额外平均对数代价 | [kl-divergence](../information-theory/kl-divergence/) |
| forward KL / reverse KL | 前向 KL / 反向 KL | 译;分别指 D(p∥q) 与 D(q∥p),方向不可交换 | [kl-divergence](../information-theory/kl-divergence/) |
| data processing inequality | 数据处理不等式 | 译;对分布施加变换后 KL 散度不会增加 | [kl-divergence](../information-theory/kl-divergence/) |
| Pinsker inequality | Pinsker 不等式 | 保留人名;用 KL 散度上界总变差距离 | [kl-divergence](../information-theory/kl-divergence/) |
| total variation distance | 总变差距离 | 译;两个离散分布概率质量差绝对值和的一半 | [kl-divergence](../information-theory/kl-divergence/) |
| variational inference | 变分推断 | 译;用可计算近似分布优化后验 KL 的推断方法 | [kl-divergence](../information-theory/kl-divergence/) |
| evidence lower bound / ELBO | 证据下界 / ELBO | 译并保留缩写;最小化后验 KL 等价优化的下界目标 | [kl-divergence](../information-theory/kl-divergence/) |
| concentration inequality | 集中不等式 | 译;用有限矩、支持区间或矩母函数控制随机变量尾部概率的上界 | [concentration-inequalities](../probability/concentration-inequalities/) |
| Markov inequality | Markov 不等式 | 保留人名;用非负随机变量期望控制超过阈值的概率 | [concentration-inequalities](../probability/concentration-inequalities/) |
| Chebyshev inequality | Chebyshev 不等式 | 保留人名;用方差控制偏离均值的双侧概率 | [concentration-inequalities](../probability/concentration-inequalities/) |
| Chernoff bound | Chernoff 界 | 保留人名;优化矩母函数得到的指数尾部上界 | [concentration-inequalities](../probability/concentration-inequalities/) |
| Hoeffding inequality | Hoeffding 不等式 | 保留人名;对独立有界样本给出指数集中界 | [concentration-inequalities](../probability/concentration-inequalities/) |
| Bernstein inequality | Bernstein 不等式 | 保留人名;同时利用方差和有界增量控制尾部 | [concentration-inequalities](../probability/concentration-inequalities/) |
| sub-Gaussian | 次高斯 | 译;矩母函数被高斯函数控制的尾部性质 | [concentration-inequalities](../probability/concentration-inequalities/) |
| union bound | 并集界 | 译;用各事件概率之和上界至少一个事件发生的概率 | [concentration-inequalities](../probability/concentration-inequalities/) |
| sample complexity | 样本复杂度 | 译;达到指定误差和失败概率所需的样本数量规模 | [concentration-inequalities](../probability/concentration-inequalities/) |

## 机器学习

| 英文 | 中文 | 约定 | 首现 |
| --- | --- | --- | --- |
| one-hot | one-hot(独热) | 首次出现括注「独热」,后文用 one-hot;词表中单个类别的标准基向量,只有一个坐标为 1 | [one-hot-and-distributed](../text-representation/one-hot-and-distributed/) |
| distributed representation | 分布式表示 | 译;把对象信息分散到多个连续坐标中的向量表示,embedding 是 token 的典型分布式表示 | [one-hot-and-distributed](../text-representation/one-hot-and-distributed/) |
| embedding lookup | embedding 查表 | 保留常用英文并解释;用 token ID 直接选择 embedding 矩阵对应行,数学上等价于 one-hot 乘矩阵 | [one-hot-and-distributed](../text-representation/one-hot-and-distributed/) |
| standard basis vector | 标准基向量 | 译;one-hot 在 $\mathbb R^V$ 中的坐标基,不同 token 的内积为 0 | [one-hot-and-distributed](../text-representation/one-hot-and-distributed/) |
| sparse update | 稀疏更新 | 译;一次 token embedding lookup 的梯度通常只写入词表矩阵被选中的行 | [one-hot-and-distributed](../text-representation/one-hot-and-distributed/) |
| multi-hot | multi-hot / 多热编码 | 保留常用英文并译;允许多个坐标同时为 1 的集合或多标签表示,不等于类别互斥的 one-hot | [one-hot-and-distributed](../text-representation/one-hot-and-distributed/) |
| hidden state | 隐藏状态 | 译,首次出现括注英文;序列模型中表示截至当前时间步历史摘要的状态向量 | [sequence-modeling](../rnn-lstm/sequence-modeling/) |
| sequence modeling / sequential modeling | 序列建模 | 译;利用有序观测、时间状态或条件分解预测序列中的输出,不等于对所有序列做生成建模 | [sequence-modeling](../rnn-lstm/sequence-modeling/) |
| time step | 时间步 | 译;序列中的一个位置或一次状态更新,用 t 标记 | [sequence-modeling](../rnn-lstm/sequence-modeling/) |
| state transition / recurrent update | 状态转移 / 循环更新 | 译;用上一步状态和当前输入计算下一状态的函数 h_t=f(h_{t−1},x_t) | [sequence-modeling](../rnn-lstm/sequence-modeling/) |
| recurrent neural network / RNN | 循环神经网络 / RNN | 译并保留缩写;在时间步共享状态转移参数、沿序列递归更新隐藏状态的网络 | [sequence-modeling](../rnn-lstm/sequence-modeling/) |
| vanilla RNN / simple RNN | vanilla RNN / 简单 RNN | 保留常用英文并译;用单一非门控隐藏状态 h_t=φ(W_{xh}x_t+W_{hh}h_{t−1}+b_h) 的基础循环网络 | [rnn](../rnn-lstm/rnn/) |
| RNN cell / recurrent cell | RNN cell / 循环单元 | 保留常用英文并译;一次时间步中把当前输入和上一状态映射为新状态的计算模块 | [rnn](../rnn-lstm/rnn/) |
| unrolling / time unfolding | 时间展开 | 译;把共享的循环 cell 沿有限时间步复制为显式计算节点和状态边 | [rnn-unrolling](../rnn-lstm/rnn-unrolling/) |
| unrolled computation graph | 展开计算图 | 译;包含每个时间步输入、状态、输出和共享参数使用点的计算图 | [rnn-unrolling](../rnn-lstm/rnn-unrolling/) |
| state carry / state reset | 状态携带 / 状态重置 | 译;分别把上一段末状态传给下一段或在序列边界重新初始化状态 | [rnn-unrolling](../rnn-lstm/rnn-unrolling/) |
| truncated backpropagation through time / TBPTT | 截断时间反向传播 / TBPTT | 译并保留缩写;前向状态跨 chunk 携带但在边界 detach,把反向时间深度限制在窗口内 | [rnn-unrolling](../rnn-lstm/rnn-unrolling/) |
| packed sequence | packed sequence / 打包序列 | 保留框架常用名称并译;按有效长度跳过 padding 时间步的变长序列表示 | [rnn-unrolling](../rnn-lstm/rnn-unrolling/) |
| backpropagation through time / BPTT | 时间反向传播 / BPTT | 译并保留缩写;沿显式时间展开图反向递推状态敏感度,并把共享参数在各时间步的局部梯度相加 | [bptt](../rnn-lstm/bptt/) |
| state sensitivity / hidden-state adjoint | 状态敏感度 / 隐藏状态伴随量 | 译;表示损失对某个时间步隐藏状态的总导数,同时包含当前输出和未来状态路径的贡献 | [bptt](../rnn-lstm/bptt/) |
| preactivation gradient | 预激活梯度 | 译;损失对循环单元预激活 a_t 的导数,由状态敏感度乘以激活函数的局部导数组成 | [bptt](../rnn-lstm/bptt/) |
| input-to-hidden / hidden-to-hidden / hidden-to-output | 输入到隐藏 / 隐藏到隐藏 / 隐藏到输出 | 译;分别指 W_{xh}、W_{hh} 与 W_{hy} 的三类参数连接 | [rnn](../rnn-lstm/rnn/) |
| recurrent weight | 循环权重 | 译;沿时间重复使用的 hidden-to-hidden 权重矩阵 W_{hh} | [rnn](../rnn-lstm/rnn/) |
| fixed point / equilibrium state | 固定点 / 平衡状态 | 译;状态转移满足 h*=F(h*) 的状态,局部 Jacobian 决定附近扰动的衰减或放大 | [rnn](../rnn-lstm/rnn/) |
| temporal vanishing gradient / recurrent gradient decay | 时间梯度消失 / 循环梯度衰减 | 译;梯度沿多个时间 Jacobian 连乘后,对较早状态或输入的信号变得极小 | [rnn-vanishing-gradient](../rnn-lstm/rnn-vanishing-gradient/) |
| time Jacobian / recurrent Jacobian | 时间 Jacobian / 循环 Jacobian | 译;状态转移对上一状态的局部导数 J_t=D_tW_{hh},决定一次时间边的方向性增益 | [rnn-vanishing-gradient](../rnn-lstm/rnn-vanishing-gradient/) |
| effective time horizon | 有效时间视野 | 译;以梯度增益低于给定阈值为准估计仍有可用信用的时间距离,不是硬记忆上限 | [rnn-vanishing-gradient](../rnn-lstm/rnn-vanishing-gradient/) |
| orthogonal recurrent weight | 正交循环权重 | 译;满足或接近 W_{hh}^{T}W_{hh}=I 的循环权重,可保持线性部分长度但不能消除激活饱和 | [rnn-vanishing-gradient](../rnn-lstm/rnn-vanishing-gradient/) |
| autoregressive modeling | 自回归建模 | 译;用前缀条件分布逐步分解联合序列概率并预测下一项 | [sequence-modeling](../rnn-lstm/sequence-modeling/) |
| many-to-one / many-to-many / one-to-many | 多对一 / 多对多 / 一对多 | 译;分别描述整条输入到单个输出、逐步输出和单个条件到序列输出的接口 | [sequence-modeling](../rnn-lstm/sequence-modeling/) |
| padding mask / sequence mask | padding 掩码 / 序列掩码 | 译;标记可变长度 batch 中有效时间步,同时约束状态更新或损失归约 | [sequence-modeling](../rnn-lstm/sequence-modeling/) |
| batch-first / time-first | batch-first / time-first | 保留常用轴名并译;分别把 batch 或时间放在三维输入的第一维 | [sequence-modeling](../rnn-lstm/sequence-modeling/) |
| bidirectional RNN | 双向循环网络 | 译;同时沿正向和反向读取序列并按 merge 规则合并两个方向的状态,不天然满足在线因果预测 | [bidirectional-rnn](../rnn-lstm/bidirectional-rnn/) |
| forward hidden state / backward hidden state | 正向隐藏状态 / 反向隐藏状态 | 译;分别沿过去到未来、未来到过去扫描并在原始时间位置对齐的两个状态序列 | [bidirectional-rnn](../rnn-lstm/bidirectional-rnn/) |
| bidirectional merge | 双向合并 | 译;把正向与反向 hidden 通过 concat、add 或 projection 变成下游输入的操作 | [bidirectional-rnn](../rnn-lstm/bidirectional-rnn/) |
| lookahead | 前视窗口 | 译;在线系统允许读取的有限未来范围,带来明确的输出延迟,不等于完整双向读取 | [bidirectional-rnn](../rnn-lstm/bidirectional-rnn/) |
| long short-term memory / LSTM | 长短期记忆网络 / LSTM | 译并保留缩写;用输入、遗忘、输出门和细胞状态显式控制写入、保留与暴露的循环网络 | [lstm](../rnn-lstm/lstm/) |
| cell state / memory state | 细胞状态 / 记忆状态 | 译;LSTM 中沿 carry 路径传递、可累积并由门控更新的长期状态 c_t | [lstm](../rnn-lstm/lstm/) |
| input gate | 输入门 | 译;用 sigmoid 比例控制候选内容写入细胞状态的逐坐标门 | [lstm](../rnn-lstm/lstm/) |
| forget gate | 遗忘门 | 译;用 sigmoid 比例控制上一时刻细胞状态保留多少的逐坐标门 | [lstm](../rnn-lstm/lstm/) |
| output gate | 输出门 | 译;控制 tanh(c_t) 有多少暴露为 LSTM 隐藏状态的逐坐标门 | [lstm](../rnn-lstm/lstm/) |
| candidate cell content | 候选细胞内容 | 译;由当前输入和上一隐藏状态经 tanh 产生、等待输入门筛选写入的向量 | [lstm](../rnn-lstm/lstm/) |
| cell highway / carry path | 细胞直通路径 / 携带路径 | 译;细胞状态沿遗忘门乘法和加法更新传递的显式时间路径,其梯度不是完整状态 Jacobian | [lstm](../rnn-lstm/lstm/) |
| forget bias | 遗忘门偏置 | 译;遗忘门预激活的初始 bias,正值通常让初始保留比例偏大但不保证长期记忆 | [lstm](../rnn-lstm/lstm/) |
| peephole connection | peephole 连接 / 窥视孔连接 | 保留变体名并译;让 LSTM 门直接读取旧或新细胞状态的额外连接 | [lstm](../rnn-lstm/lstm/) |
| projection LSTM | 投影 LSTM | 译;用内部较宽的细胞状态和较窄的 hidden projection 分离记忆宽度与循环输出宽度 | [lstm](../rnn-lstm/lstm/) |
| gated recurrent unit / GRU | 门控循环单元 / GRU | 译并保留缩写;用更新门在旧 hidden 与候选 hidden 之间插值、用重置门控制候选读取旧状态的循环网络 | [gru](../rnn-lstm/gru/) |
| update gate | 更新门 | 译;在本文 convention 中表示候选 hidden 的写入比例,其补数是旧 hidden 的直接保留比例 | [gru](../rnn-lstm/gru/) |
| reset gate | 重置门 | 译;控制候选状态计算时读取上一 hidden 多少的逐坐标 sigmoid 门 | [gru](../rnn-lstm/gru/) |
| candidate hidden state | 候选隐藏状态 | 译;由输入与 reset 调制的旧 hidden 经过仿射变换和 tanh 产生、等待更新门混合的向量 | [gru](../rnn-lstm/gru/) |
| reset-before-matrix / reset-after-matrix | 矩阵前重置 / 矩阵后重置 | 译;分别在 recurrent 矩阵乘法前后施加 reset gate 的 GRU 候选计算变体 | [gru](../rnn-lstm/gru/) |
| update-gate convention | 更新门约定 | 译;明确 z 是候选写入比例还是旧状态保留比例,决定公式、bias 初始化和门图解释方向 | [gru](../rnn-lstm/gru/) |
| sequence-to-sequence / Seq2Seq | 序列到序列 / Seq2Seq | 译并保留缩写;用 encoder 读取源序列、用 decoder 按目标前缀生成可变长度输出的条件模型 | [seq2seq](../rnn-lstm/seq2seq/) |
| encoder-decoder | 编码器—解码器 | 译;由 encoder 生成源 memory、由 decoder 按目标前缀并通过 cross-attention 生成目标序列的条件结构,不等于任意两层网络 | [encoder-decoder](../transformer-architectures/encoder-decoder/) |
| context vector / fixed context | context 向量 / 固定 context | 保留常用词并译;encoder 传给 decoder 的源序列摘要,固定版本通常由末状态或 bridge 产生 | [seq2seq](../rnn-lstm/seq2seq/) |
| fixed-context bottleneck | 固定 context 瓶颈 | 译;源序列先压成一个摘要后,decoder 没有显式源地址、只能从同一混合向量恢复不同目标 step 所需证据的接口限制 | [why-attention](../rnn-lstm/why-attention/) |
| addressable read / soft read | 可寻址读取 / 软读取 | 译;保留源状态序列并用 query 产生归一化权重,按需从指定源位置的 value 做可微加权读取 | [why-attention](../rnn-lstm/why-attention/) |
| attention as retrieval | attention 作为检索 | 译;把 query-key score、mask、softmax 权重和 value 加权读出统一为一次可微 key-value 读取 | [attention-as-retrieval](../attention/attention-as-retrieval/) |
| hard read | 硬读取 | 译;通过 argmax 或离散选择返回一个 value 槽位,读取路径清晰但选择边界不可微 | [attention-as-retrieval](../attention/attention-as-retrieval/) |
| candidate recall | 候选召回 | 译;近似或 top-k attention 在精确重排前把真实相关 key 放入候选集合的比例 | [attention-as-retrieval](../attention/attention-as-retrieval/) |
| content-addressable memory | 内容寻址记忆 | 译;用 query 与 key 的匹配而非固定数组位置选择 value 的记忆接口 | [attention-as-retrieval](../attention/attention-as-retrieval/) |
| bridge | bridge / 状态桥接 | 保留常用词并译;把 encoder 状态投影或初始化为 decoder 状态的连接 | [seq2seq](../rnn-lstm/seq2seq/) |
| exposure bias | 暴露偏差 | 译;teacher forcing 训练使用真实前缀、推理使用模型前缀造成的条件分布差异 | [teacher-forcing](../rnn-lstm/teacher-forcing/) |
| Bahdanau attention / additive attention | Bahdanau 注意力 / 加性注意力 | 译并保留人名;用 decoder query 与各个 encoder state 的加性 score 生成源轴权重,再形成每步 context 的注意力机制 | [bahdanau-attention](../rnn-lstm/bahdanau-attention/) |
| alignment score / energy | 对齐分数 / energy | 译并保留常用词;在 softmax 前衡量当前 decoder query 与某个 encoder key 匹配程度的标量 $e_{j,i}$ | [bahdanau-attention](../rnn-lstm/bahdanau-attention/) |
| query / key / value | query / key / value | 保留常用术语并解释;query 提出读取条件,key 参与匹配,value 提供被加权读出的内容 | [bahdanau-attention](../rnn-lstm/bahdanau-attention/) |
| attention weight | 注意力权重 | 译;沿源时间轴 softmax 后的非负读取系数 $\alpha_{j,i}$,每个目标 step 的有效权重和为一 | [bahdanau-attention](../rnn-lstm/bahdanau-attention/) |
| attention map | 注意力图 | 译;按目标位置和源位置排列的 $T\times S$ 权重可视化或记录,不自动等于因果解释 | [bahdanau-attention](../rnn-lstm/bahdanau-attention/) |
| step-wise context | 逐步 context / 每步 context | 译;decoder 每个目标时间步按当前权重从完整 encoder 状态序列读出的 $c_j$ | [bahdanau-attention](../rnn-lstm/bahdanau-attention/) |
| attention entropy | 注意力熵 | 译;用 $-\sum_i\alpha_{j,i}\log\alpha_{j,i}$ 衡量某个目标 step 的读取分布集中或分散程度 | [bahdanau-attention](../rnn-lstm/bahdanau-attention/) |
| self-attention / self-attention layer | 自注意力 / 自注意力层 | 译;从同一序列生成 query、key、value,用 $T\times T$ 的位置对权重让每个位置读取其他位置的 value | [self-attention](../attention/self-attention/) |
| permutation equivariance | 排列等变性 | 译;输入位置按置换矩阵交换时,没有位置特征的 self-attention 输出按同一置换交换,不自动恢复原始顺序 | [self-attention](../attention/self-attention/) |
| attention graph / message passing | 注意力图 / 消息传递 | 译;把位置视为节点、attention weight 视为有向边权、value 视为消息的 self-attention 解释 | [self-attention](../attention/self-attention/) |
| position pair | 位置对 | 译;一个 query 位置与一个 key 位置组成的交互单元,稠密 self-attention 在长度 $T$ 时有 $T^2$ 个候选对 | [self-attention](../attention/self-attention/) |
| attention complexity | 注意力复杂度 | 译;分别核算 query-key 交互、MAC/FLOPs、激活内存、KV cache 和 prefill/decode 资源的 attention 账本 | [attention-complexity](../attention/attention-complexity/) |
| query-key pair / attention pair | query-key 交互对 / 注意力交互对 | 译;一个 query 位置与一个 key 位置形成的逻辑交互单元,dense attention 的数量为 $B h_q T S$ | [attention-complexity](../attention/attention-complexity/) |
| multiply-accumulate / MAC | 乘加 / MAC | 保留缩写;一次乘法与加法的算术计数单位,报告中要说明一个 MAC 是否按 2 FLOPs 换算 | [attention-complexity](../attention/attention-complexity/) |
| FLOP | FLOP / 浮点运算次数 | 保留缩写并译;报告浮点算术量的单位,必须注明 FMA 或 MAC 的换算口径 | [attention-complexity](../attention/attention-complexity/) |
| materialized attention matrix | 物化注意力矩阵 | 译;把逻辑 $T\times S$ 的 score 或权重完整写入显存的中间张量,与 tile 工作区相对 | [attention-complexity](../attention/attention-complexity/) |
| sparse attention | 稀疏注意力 | 译;为每个 query 指定有限 key 集合 $E_t$,在保留连接上做 attention,并要求 kernel 实际跳过集合外位置 | [sparse-attention](../attention/sparse-attention/) |
| sparsity pattern / edge set | 稀疏模式 / 连接集合 | 译;规定 query-key 位置哪些边存在的结构化或动态索引集合,总边数决定逻辑稀疏 attention 的位置工作量 | [sparse-attention](../attention/sparse-attention/) |
| local attention / sliding window | 局部注意力 / 滑动窗口 | 译;每个 query 只读取邻近固定窗口内的 key,causal 版本只保留当前及有限历史 | [sparse-attention](../attention/sparse-attention/) |
| block-sparse attention | 块稀疏注意力 | 译;以规整 $b\times b$ 块为单位保留 query-key 连接,区分有效 token 边与 kernel 处理的完整 tile | [sparse-attention](../attention/sparse-attention/) |
| global token | 全局 token | 译;在局部或稀疏 pattern 中保留全序列可见的特殊位置,用于建立远距离信息路径 | [sparse-attention](../attention/sparse-attention/) |
| dilated attention | 膨胀注意力 | 译;按固定步长从历史或序列中抽取 key,在保持有限连接数的同时扩大单层跨度 | [sparse-attention](../attention/sparse-attention/) |
| dynamic top-k attention | 动态 top-k 注意力 | 译;按输入或 query 动态选择候选 key,必须把候选生成和召回误差与候选内 attention 分开核算 | [sparse-attention](../attention/sparse-attention/) |
| linear attention | 线性注意力 | 译;用 feature map 把 query-key kernel 写成可分离内积,先汇总 K/V 再由 query 读取,序列长度项通常为 $(T+S)r$ | [linear-attention](../attention/linear-attention/) |
| kernelized attention | 核化注意力 | 译;用 $\kappa(q,k)=\phi(q)^\mathsf T\phi(k)$ 替换或近似标准 softmax kernel 的 attention 形式 | [linear-attention](../attention/linear-attention/) |
| feature map | 特征映射 | 译;把输入映射到另一特征空间,用于改变可表达函数族或因子化相似度计算 | [kernel-trick](../linear-models/kernel-trick/) |
| prefix state | 前缀状态 | 译;causal linear attention 中累积历史 $\phi(k)v^\mathsf T$ 与 $\phi(k)$ 的摘要,每个 decode step 更新一次 | [linear-attention](../attention/linear-attention/) |
| associative scan | 结合律扫描 / associative scan | 译并保留常用英文;利用可结合的状态合并并行计算 prefix summary 的算法结构 | [linear-attention](../attention/linear-attention/) |
| positive kernel | 正 kernel | 译;保证 query-key 特征内积及归一化分母保持非负或远离零的 kernel 约束 | [linear-attention](../attention/linear-attention/) |
| FlashAttention | FlashAttention / 闪存注意力 | 保留名称并译;在片上 tile 中融合 score、mask、online softmax 和 value 加权,避免把完整 attention matrix 写入 HBM | [flash-attention](../attention/flash-attention/) |
| IO-aware attention | IO-aware attention / 面向 IO 的注意力 | 保留术语并译;以高带宽显存与片上存储之间的数据搬运为主要约束设计 attention kernel | [flash-attention](../attention/flash-attention/) |
| online softmax | 在线 softmax | 译;按 tile 维护行最大值、重标度指数和与 value 加权和,在不保存完整 score 的情况下得到稳定归一化结果 | [flash-attention](../attention/flash-attention/) |
| attention tile | 注意力 tile / 注意力分块 | 译;query block 与 key/value block 的局部交互矩阵,大小通常为 $B_r\times B_c$ | [flash-attention](../attention/flash-attention/) |
| log-sum-exp statistics / LSE | log-sum-exp 统计量 / LSE | 保留缩写并译;保存每个 query 行的 $m+\\log\\ell$,供 FlashAttention backward 重算 softmax tile | [flash-attention](../attention/flash-attention/) |
| positional encoding / position encoding | 位置编码 | 译;按序列位置生成向量或其他位置条件,并注入 token 表示或 attention score | [positional-encoding](../transformer-components/positional-encoding/) |
| absolute positional encoding | 绝对位置编码 | 译;位置条件直接绑定绝对索引 $p$,可用固定公式或可学习位置表产生 | [positional-encoding](../transformer-components/positional-encoding/) |
| sinusoidal positional encoding | 正弦位置编码 | 译;用不同频率的 sine/cosine 相位构造固定位置向量 | [positional-encoding](../transformer-components/positional-encoding/) |
| learned positional embedding | 可学习位置嵌入 | 译;把每个绝对位置作为位置表中的一行参数,通常受 $L_{\max}$ 限制 | [positional-encoding](../transformer-components/positional-encoding/) |
| position ID / position index | 位置 ID / 位置索引 | 译;索引位置表或计算位置公式的整数,与 token ID 的词表索引分开 | [positional-encoding](../transformer-components/positional-encoding/) |
| relative position | 相对位置 | 译;由 query 与 key 的位置差如 $p-s$ 表示的位移关系 | [positional-encoding](../transformer-components/positional-encoding/) |
| rotary positional embedding / RoPE | 旋转位置编码 / RoPE | 保留缩写并译;把每个 Q/K 的二维坐标对按绝对位置旋转,使点积包含相对位移相位 | [rope](../transformer-components/rope/) |
| rotary phase | 旋转相位 | 译;位置与 inverse frequency 的乘积 $p\theta$,决定每个二维坐标对的旋转角度 | [rope](../transformer-components/rope/) |
| inverse frequency | inverse frequency / 逆频率 | 保留常用术语并译;按 base 和 head dimension 生成 $\theta_r$,控制各坐标对的相位变化速度 | [rope](../transformer-components/rope/) |
| rotated query / rotated key | 旋转 query / 旋转 key | 译;RoPE 作用后的 Q/K 向量,缓存和 score 必须明确其是否已旋转 | [rope](../transformer-components/rope/) |
| relative phase | 相对相位 | 译;query 位置 $p$ 与 key 位置 $s$ 的旋转点积中出现的 $(s-p)\theta$ | [rope](../transformer-components/rope/) |
| Attention with Linear Biases / ALiBi | 带线性偏置的注意力 / ALiBi | 保留缩写并译;在 score 上加入每个 head 的负距离偏置,不改写 Q/K/V | [alibi](../transformer-components/alibi/) |
| linear attention bias | 线性距离偏置 | 译;按 $-m_h\Delta(i,j)$ 随 query-key 距离线性变化的 score 偏置 | [alibi](../transformer-components/alibi/) |
| head-specific slope | 每头斜率 / head-specific slope | 保留常用英文并译;为每个 attention head 分配的固定距离惩罚系数 $m_h$ | [alibi](../transformer-components/alibi/) |
| distance penalty | 距离惩罚 | 译;距离增大时从 score 中扣除的项,由 slope 和位置差共同决定 | [alibi](../transformer-components/alibi/) |
| half-life distance | 半衰距离 | 译;在内容分数相同条件下,距离偏置使未归一化权重降低到一半所需的距离 $\ln 2/m$ | [alibi](../transformer-components/alibi/) |
| pre-norm / pre-LN | pre-norm / 预归一化 | 保留常用术语并译;先对残差分支输入做归一化,再计算子层修正并与原输入相加,抽象形式为 $x+F(N(x))$ | [pre-norm-vs-post-norm](../transformer-components/pre-norm-vs-post-norm/) |
| post-norm / post-LN | post-norm / 后归一化 | 保留常用术语并译;先计算子层修正并与输入相加,再对合并结果做归一化,抽象形式为 $N(x+F(x))$ | [pre-norm-vs-post-norm](../transformer-components/pre-norm-vs-post-norm/) |
| identity gradient path | 恒等梯度路径 | 译;pre-norm Jacobian 中由 shortcut 产生的外侧 $I$,形式为 $J_{\mathrm{pre}}=I+J_FJ_N$ | [pre-norm-vs-post-norm](../transformer-components/pre-norm-vs-post-norm/) |
| projected shortcut | 投影 shortcut | 保留常用英文并译;当残差分支和输入宽度不同时用线性投影匹配 shape 的 shortcut $S(x)$ | [pre-norm-vs-post-norm](../transformer-components/pre-norm-vs-post-norm/) |
| final RMSNorm | final RMSNorm / 最终 RMSNorm | 保留常用术语并译;pre-norm block 堆栈末尾、输出投影之前执行的独立 RMSNorm | [pre-norm-vs-post-norm](../transformer-components/pre-norm-vs-post-norm/) |
| normalization Jacobian / LayerNorm Jacobian | 归一化 Jacobian / LayerNorm Jacobian | 译并保留常用英文;归一化输出对特征输入的局部导数矩阵,包含去均值、方差缩放和 $\gamma$ 的作用 | [layernorm-residuals](../transformer-components/layernorm-residuals/) |
| residual stream | 残差流 | 译;Transformer 各层沿同一 $d_{\mathrm{model}}$ 特征宽度逐层传递、并由子层增量更新的表示序列 | [residual-streams](../transformer-components/residual-streams/) |
| residual update / residual branch output | 残差更新 / 残差分支输出 | 译;子层从当前 stream 读取后产生、在写回前必须匹配 $d_{\mathrm{model}}$ 的增量 $\Delta_l$ | [residual-streams](../transformer-components/residual-streams/) |
| parallel residual branches | 并行残差分支 | 译;多个分支读取同一个 $x_l$、再在同一个加法节点合并的结构 | [residual-streams](../transformer-components/residual-streams/) |
| stream readout | 流读出 | 译;用固定方向 $w$ 把某层 residual stream 投影为标量 $w^{\mathsf T}x_l$ 的测量 | [residual-streams](../transformer-components/residual-streams/) |
| residual intervention | 残差流干预 | 译;在指定层的 stream 上加入向量或替换 stream 的解释性实验操作 | [residual-streams](../transformer-components/residual-streams/) |
| feedforward network / FFN | 前馈网络 / FFN | 译并保留缩写;对每个 token 独立执行两次特征投影与逐坐标非线性激活、再把输出压回 $d_{\mathrm{model}}$ 的 Transformer 子层 | [feedforward](../transformer-components/feedforward/) |
| intermediate width / FFN width | 中间宽度 / FFN 宽度 | 译;FFN 激活前后中间表示的特征数 $d_{\mathrm{ffn}}$,通常大于 residual stream 的 $d_{\mathrm{model}}$ | [feedforward](../transformer-components/feedforward/) |
| token-wise feedforward | 逐 token 前馈 | 译;固定 batch 和位置后只沿最后特征轴计算、不在 token 轴上读取其他位置的 FFN 方式 | [feedforward](../transformer-components/feedforward/) |
| FFN local Jacobian | FFN 局部 Jacobian | 译并保留常用英文;两次线性投影和激活组成的局部导数 $J_F=W_2D_\phi W_1$ | [feedforward](../transformer-components/feedforward/) |
| active fraction | 激活比例 | 译;以 ReLU 为例,一个 batch 中 pre-activation 为正的中间坐标比例,用于诊断稀疏性而非单独判断能力 | [feedforward](../transformer-components/feedforward/) |
| expansion ratio | 扩展比 | 译;FFN 中间宽度与 residual stream 宽度之比 $d_{\mathrm{ffn}}/d_{\mathrm{model}}$ | [feedforward](../transformer-components/feedforward/) |
| gate projection / gate branch | 门投影 / 门分支 | 译并保留常用英文;SwiGLU 中把输入 $h$ 映射为 gate pre-activation $g=W_gh+b_g$ 的投影及其分支 | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| up projection / value branch | 上投影 / value 分支 | 译并保留常用英文;SwiGLU 中把输入 $h$ 映射为待调制 value $u=W_uh+b_u$ 的投影及其分支 | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| down projection | 下投影 | 译;把门控结果 $a\in\mathbb R^m$ 压回 residual stream 宽度 $d$ 的矩阵 $W_d$ | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| gated FFN | 门控 FFN | 译;用 $\operatorname{SiLU}(g)\odot u$ 等输入相关乘法调制中间特征的前馈子层 | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| gate coefficient / gate value | 门控系数 / 门值 | 译;gate 分支经过门函数后的逐坐标系数 $s=\operatorname{SiLU}(g)$,不等同于 sigmoid 概率 | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| gate saturation | 门饱和 | 译;gate pre-activation 进入 SiLU 数值极端区域后,输出尺度或局部导数影响门控路径的状态 | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| SwiGLU local Jacobian | SwiGLU 局部 Jacobian | 译并保留常用英文;门分支与 value 分支共同决定的局部导数 $W_d[\operatorname{diag}(u\odot s')W_g+\operatorname{diag}(s)W_u]$ | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| gated product / activation product | 门控乘积 / 激活乘积 | 译;gate 输出与 value 逐坐标相乘的中间结果 $a=s\odot u$ | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| SwiGLU parameter budget | SwiGLU 参数预算 | 译;无 bias 时三矩阵的参数量 $3dm$,需要与普通 FFN 的 $2dm$ 和中间宽度共同比较 | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| fused SwiGLU | 融合 SwiGLU | 保留常用英文并译;在一个执行路径中融合 gate/up、SiLU、门控乘法和 down projection 的实现 | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| final LayerNorm | final LayerNorm / 最终层归一化 | 保留常用术语并译;在 pre-norm block 堆栈末尾、输出投影之前执行的 LayerNorm,属于架构合同 | [layernorm-residuals](../transformer-components/layernorm-residuals/) |
| RMSNorm / root mean square normalization | RMSNorm / 均方根归一化 | 保留缩写并译;沿 token 特征轴计算均方根并缩放,不减去特征均值,规范形式只有 $\gamma$ 没有 $\beta$ | [rmsnorm](../transformer-components/rmsnorm/) |
| root mean square / RMS | 均方根 / RMS | 译并保留缩写;特征平方平均后开平方的尺度 $\sqrt{\operatorname{mean}(x^2)+\epsilon}$ | [rmsnorm](../transformer-components/rmsnorm/) |
| inverse RMS | inverse RMS / 逆均方根 | 保留常用术语并译;RMSNorm 中乘在输入特征上的 $\operatorname{rsqrt}(\operatorname{mean}(x^2)+\epsilon)$ | [rmsnorm](../transformer-components/rmsnorm/) |
| radial direction in normalization | 归一化径向方向 | 译;沿当前特征向量 $x$ 的局部方向,RMSNorm 在 $\epsilon=0$ 时对该方向的 Jacobian 为零 | [rmsnorm](../transformer-components/rmsnorm/) |
| ScaleNorm | ScaleNorm / 全局尺度归一化 | 保留名称并译;用一个标量尺度除以向量 L2 范数,与逐特征 $\gamma$ 的 RMSNorm 参数合同不同 | [rmsnorm](../transformer-components/rmsnorm/) |
| attention matrix | 注意力矩阵 | 译;行对应 query、列对应 key 的 $T\times S$ 读取权重矩阵,逐行 softmax 后用 $C=AV$ 混合 value | [attention-matrix](../attention/attention-matrix/) |
| row-stochastic matrix | 行随机矩阵 | 译;元素非负且每行和为 1 的矩阵,普通 attention matrix 满足 $A\mathbf 1=\mathbf 1$ 但列和不必为 1 | [attention-matrix](../attention/attention-matrix/) |
| doubly stochastic matrix | 双随机矩阵 | 译;行和与列和都为 1 的非负矩阵,attention 的 row-wise softmax 不自动满足这一性质 | [attention-matrix](../attention/attention-matrix/) |
| multi-head attention | 多头注意力 | 译;同一输入在 $h$ 个独立 Q/K/V 子空间中并行读取,沿特征轴拼接各 head 输出后用 $W_O$ 混合 | [multi-head-attention](../attention/multi-head-attention/) |
| head dimension / per-head dimension | head 维度 / 每头维度 | 译;单个 head 的 query/key/value 子空间宽度,标准配置为 $d_{\mathrm{model}}/h$ | [multi-head-attention](../attention/multi-head-attention/) |
| output projection | 输出投影 | 译;把拼接后的 $h d_v$ 通道映射回 $d_{\mathrm{model}}$ 的 $W_O$ 线性层 | [multi-head-attention](../attention/multi-head-attention/) |
| head ablation | head 消融 | 译;移除或置零一个 attention head 后比较输出或损失变化的干预测试 | [multi-head-attention](../attention/multi-head-attention/) |
| grouped-query attention / GQA | 分组查询注意力 / GQA | 译;保留 $h_q$ 个独立 query head,让每组 query head 共享一个 K/V head,组大小为 $r=h_q/h_{kv}$ | [gqa-and-mqa](../attention/gqa-and-mqa/) |
| multi-query attention / MQA | 多查询注意力 / MQA | 译;GQA 的极端配置,保留多个 query head 但令 $h_{kv}=1$,所有 query head 共享一组 K/V | [gqa-and-mqa](../attention/gqa-and-mqa/) |
| query head / KV head | query head / K/V head | 译;分别表示产生独立 query 或承载共享 key/value 的 head,两者数量在 GQA 中可以不同 | [gqa-and-mqa](../attention/gqa-and-mqa/) |
| group mapping | 组映射 | 译;把 query head $q$ 映射到 K/V 组 $g(q)$ 的索引规则,标准等分配置为 $\lfloor q/r\rfloor$ | [gqa-and-mqa](../attention/gqa-and-mqa/) |
| KV cache | KV cache / K/V 缓存 | 保留常用英文并译;自回归解码时按层保存历史 key/value,容量与 $B L h_{kv}(d_k+d_v)$ 成正比 | [gqa-and-mqa](../attention/gqa-and-mqa/) |
| K/V sharing | K/V 共享 | 译;多个 query head 复用同一组 key/value 表示,不等同于复用 query、score 或 attention weight | [gqa-and-mqa](../attention/gqa-and-mqa/) |
| scaled dot-product attention | 缩放点积注意力 | 译;先计算 query-key 点积、除以 $\sqrt{d_k}$、加入 mask,再沿 key 轴 softmax 的 attention 形式 | [scaled-dot-product](../attention/scaled-dot-product/) |
| scaling factor | 缩放因子 | 译;作用在 attention score 上的固定或可学习乘除标量,本词条的固定因子为 $1/\sqrt{d_k}$ | [scaled-dot-product](../attention/scaled-dot-product/) |
| matching dimension | 匹配维度 | 译;query 与 key 用点积比较时的共同最后一维 $d_k$,多头注意力中等于每个 head 的宽度 | [scaled-dot-product](../attention/scaled-dot-product/) |
| score variance / logit scale | score 方差 / logit 尺度 | 译并保留常用英文;softmax 前匹配分数的二阶波动与整体数值范围,会影响权重尖锐度和梯度分配 | [scaled-dot-product](../attention/scaled-dot-product/) |
| BOS / beginning-of-sequence | BOS / 序列开始标记 | 保留缩写并译;decoder 第一步输入的特殊 token,通常不作为需要预测的目标 | [seq2seq](../rnn-lstm/seq2seq/) |
| EOS / end-of-sequence | EOS / 序列结束标记 | 保留缩写并译;目标序列中的停止 token,推理命中后可结束一条生成路径 | [seq2seq](../rnn-lstm/seq2seq/) |
| PAD / padding token | PAD / 填充 token | 保留缩写并译;把 batch 中不同长度序列补成矩形的特殊 token,通常需要 mask | [seq2seq](../rnn-lstm/seq2seq/) |
| autoregressive decoding | 自回归解码 | 译;把已经生成的 token 作为下一步 decoder 输入的推理过程 | [seq2seq](../rnn-lstm/seq2seq/) |
| beam search | beam search / 束搜索 | 保留常用英文并译;按累计 log probability 保留多个候选前缀的近似序列搜索 | [beam-search](../inference/beam-search/) |
| length penalty | 长度惩罚 | 译;对 beam 的累计 log probability 按序列长度重新加权的搜索超参数 | [beam-search](../inference/beam-search/) |
| scheduled sampling | scheduled sampling / 计划采样 | 保留常用英文并译;训练时按计划在真实前缀与模型生成前缀之间切换的策略 | [teacher-forcing](../rnn-lstm/teacher-forcing/) |
| representation | 表示 | 译 | [vectors](../linear-algebra/vectors/) |
| loss | 损失 | 首次出现对照「损失(loss)」,后文用「损失」 | [vectors](../linear-algebra/vectors/) |
| embedding | 嵌入 | 译(嵌入层、词嵌入);把离散 token 或其他索引映射为可学习连续向量,词条 slug 保留英文 | [embeddings](../text-representation/embeddings/) |
| embedding geometry | embedding 几何 | 保留常用英文并译;把 embedding 行作为点,在固定对象、metric、预处理和 checkpoint 下分析长度、方向、距离与邻域 | [embedding-geometry](../text-representation/embedding-geometry/) |
| embedding matrix | embedding 矩阵 | 保留常用英文并解释;按 token ID 保存向量行的 $V\times d$ 可学习矩阵,lookup 直接选择其中一行 | [embeddings](../text-representation/embeddings/) |
| static embedding | 静态 embedding | 保留常用英文并解释;只依赖 token ID 的固定查表向量,同一 token 在不同上下文中查到同一行 | [embeddings](../text-representation/embeddings/) |
| contextual representation / contextual embedding | 上下文表示 / contextual embedding | 译并保留英文;经过上下文层后同时依赖 token、位置、邻近输入和 mask 的运行时 hidden state | [embeddings](../text-representation/embeddings/) |
| skip-gram | skip-gram | 保留模型名;用中心 token 预测窗口内上下文 token 的 embedding 训练目标 | [embeddings](../text-representation/embeddings/) |
| continuous bag-of-words / CBOW | 连续词袋 / CBOW | 译并保留缩写;聚合上下文 token 向量后预测中心 token 的 embedding 训练目标 | [embeddings](../text-representation/embeddings/) |
| negative sampling | 负采样 | 译;用真实上下文正样本和从噪声分布抽取的负 token 近似完整 softmax 目标 | [embeddings](../text-representation/embeddings/) |
| tied weights / weight tying | 权重绑定 / 权重共享 | 译;把输入 embedding 矩阵的转置复用于输出词表投影的架构选择 | [embeddings](../text-representation/embeddings/) |
| Gram matrix | Gram 矩阵 | 保留人名并译;由向量两两内积组成的矩阵,可由对角线和非对角线恢复平方欧氏距离 | [embedding-geometry](../text-representation/embedding-geometry/) |
| nearest neighbor / k-nearest neighbors | 最近邻 / k 近邻 | 译;在指定候选集、metric 和预处理下按距离或相似度排序得到的邻居 | [embedding-geometry](../text-representation/embedding-geometry/) |
| hubness | hubness / 邻居集中 | 保留常用英文并解释;少数向量在许多查询的 top-k 邻域中反复出现的现象 | [embedding-geometry](../text-representation/embedding-geometry/) |
| centering / centered embedding | 中心化 / 中心化 embedding | 译;从每个向量减去候选集合均值的预处理,会改变原始距离与近邻 | [embedding-geometry](../text-representation/embedding-geometry/) |
| tokenization / tokenizer | tokenization / 分词 / tokenizer / 分词器 | 保留常用英文并译;把字符串经规范化、切分、词表映射和 special-token 后处理变成模型输入 ID 序列的完整组件 | [tokenization](../text-representation/tokenization/) |
| token | token / 标记 | 保留常用英文并译;词表定义的离散建模单位,可以是词、字符、byte 或子词,不等于 embedding | [tokenization](../text-representation/tokenization/) |
| vocabulary / vocab | 词表 / vocab | 译并保留常用缩写;允许 tokenizer 输出的 token 集合及其到整数 ID 的映射 | [tokenization](../text-representation/tokenization/) |
| vocabulary size | 词表大小 | 译;tokenizer 可直接输出的 token 条目数 $V$,会同时影响 embedding 参数、softmax 候选、覆盖率和序列长度 | [vocabulary-size-tradeoffs](../text-representation/vocabulary-size-tradeoffs/) |
| token fertility / fragmentation | token fertility / token 碎片化 | 保留常用英文并解释;每个自然词平均对应的 token 数及其长尾,用于比较词表粒度造成的序列长度压力 | [vocabulary-size-tradeoffs](../text-representation/vocabulary-size-tradeoffs/) |
| token coverage | token 覆盖率 | 译;输入片段被词表直接 token 或 fallback 协议覆盖的比例,需按语言和输入类型分层 | [vocabulary-size-tradeoffs](../text-representation/vocabulary-size-tradeoffs/) |
| token ID | token ID / 标记 ID | 保留术语并解释;词表中 token 的整数索引,只用于查 embedding 行,数值大小没有语义距离 | [tokenization](../text-representation/tokenization/) |
| subword token | 子词 token | 译;在词与字符之间复用常见片段、组合少见词形的离散 token | [tokenization](../text-representation/tokenization/) |
| byte-level / byte fallback | byte-level / byte fallback | 保留常用英文并解释;以 UTF-8 byte 覆盖任意输入或作为未知片段的最终回退层 | [tokenization](../text-representation/tokenization/) |
| out-of-vocabulary / OOV / UNK | 未登录词 / OOV / UNK | 保留缩写并译;词表无法直接覆盖的输入片段及其 unknown 特殊 token 表示 | [tokenization](../text-representation/tokenization/) |
| special token | 特殊 token | 译;BOS、EOS、PAD、UNK、SEP、CLS 等参与模型 shift、mask、模板或停止协议的离散符号 | [tokenization](../text-representation/tokenization/) |
| token length / token budget | token 长度 / token 预算 | 译;文本经 tokenizer 后的序列长度及其对上下文窗口、padding、截断和 attention 成本的约束 | [tokenization](../text-representation/tokenization/) |
| byte pair encoding / BPE | BPE 分词 / byte pair encoding | 保留缩写并译;从初始 alphabet 统计相邻 pair,按频率学习有顺序的 merge list,再用固定 rank 把新输入切成子词 token | [tokenization-bpe](../text-representation/tokenization-bpe/) |
| merge rule / merge rank | 合并规则 / merge rank | 保留常用英文并解释;把相邻 token pair 合成为新 token 的规则及其在编码时的优先级顺序,不等于 vocabulary ID | [tokenization-bpe](../text-representation/tokenization-bpe/) |
| initial alphabet | 初始 alphabet | 保留常用英文并解释;BPE 开始统计前用于展开输入的字符、Unicode code point 或 UTF-8 byte 集合 | [tokenization-bpe](../text-representation/tokenization-bpe/) |
| word-boundary marker | 词边界标记 | 译;附在词首或词尾以阻止跨词 merge 并帮助 decode 恢复空格的 tokenizer 协议符号 | [tokenization-bpe](../text-representation/tokenization-bpe/) |
| byte-level BPE | byte-level BPE | 保留常用英文并解释;先把字符串编码为 UTF-8 byte,再在 byte alphabet 上学习 merge 的 BPE 变体 | [tokenization-bpe](../text-representation/tokenization-bpe/) |
| pair frequency | pair 频率 | 译;当前训练序列中相邻符号 pair 的按语料频次加权计数,每轮 merge 后必须重新或增量更新 | [tokenization-bpe](../text-representation/tokenization-bpe/) |
| WordPiece | WordPiece 分词 | 保留模型名并译;用似然相关的 pair 得分学习子词词表,编码时常用词首 token、continuation marker 和最长匹配 | [wordpiece-and-sentencepiece](../text-representation/wordpiece-and-sentencepiece/) |
| continuation marker | continuation 标记 / 词内延续标记 | 保留常用英文并解释;WordPiece 中表示 token 位于同一预分词片段内部而非词首的前缀协议 | [wordpiece-and-sentencepiece](../text-representation/wordpiece-and-sentencepiece/) |
| SentencePiece | SentencePiece 分词器 | 保留工具名并解释;直接处理原始句子、把空格纳入 tokenizer 协议且可承载 BPE 或 Unigram 模型的工具包 | [wordpiece-and-sentencepiece](../text-representation/wordpiece-and-sentencepiece/) |
| whitespace marker | 空格标记 / whitespace marker | 译并保留英文;SentencePiece 常用的 U+2581 词首符号,用于在 token 中保留词间空格信息 | [wordpiece-and-sentencepiece](../text-representation/wordpiece-and-sentencepiece/) |
| Unigram language model / Unigram tokenizer | Unigram 语言模型 / Unigram 分词 | 保留模型名并译;为候选 token 学习概率,用所有合法分词路径的似然和及 Viterbi 或 sampling 产生序列 | [wordpiece-and-sentencepiece](../text-representation/wordpiece-and-sentencepiece/) |
| Viterbi segmentation | Viterbi 分词 / 最优路径分词 | 译;在 token 概率固定时用动态规划最大化一条完整分词路径的对数概率 | [wordpiece-and-sentencepiece](../text-representation/wordpiece-and-sentencepiece/) |
| attention mask | 注意力掩码 | 译;在 softmax 前把 padding 或不可读位置从注意力分布中排除的 mask | [bahdanau-attention](../rnn-lstm/bahdanau-attention/) |
| causal mask / causal attention | 因果掩码 / 因果注意力 | 译;在自回归 attention 中保留 $s\le t$ 的 key 位置,把 query 位置 $t$ 对未来位置设为不可读 | [causal-masking](../attention/causal-masking/) |
| inclusive causal mask | 包含对角线的因果掩码 | 译;保留 $s=t$ 的 self-loop 的 causal mask,第一个 query 至少可以读取自己 | [causal-masking](../attention/causal-masking/) |
| cache offset mask | cache 偏移掩码 | 译;已有 $L_{\mathrm{past}}$ 个历史 key 时,按真实位置 $L_{\mathrm{past}}+i$ 构造新 query 的因果连接 | [causal-masking](../attention/causal-masking/) |
| prefill / chunk prefill | prefill / 块状 prefill | 保留推理术语并译;一次处理完整 prompt 或带历史 KV cache 的新 token 块,使用相应的因果位置偏移 | [causal-masking](../attention/causal-masking/) |
| future information leakage | 未来信息泄漏 | 译;query 位置读取训练或推理时不可用的未来 token,导致因果约束失效 | [causal-masking](../attention/causal-masking/) |
| block-diagonal causal mask | 块对角因果掩码 | 译;packed batch 中每条序列内部保留下三角,不同序列之间不建立连接 | [causal-masking](../attention/causal-masking/) |
| cross-attention / encoder-decoder attention | 交叉注意力 / 编解码器注意力 | 译;目标序列产生 query,源序列产生 key/value,形成 $T\times S$ 的读取矩阵 | [cross-attention](../attention/cross-attention/) |
| source sequence / target sequence | 源序列 / 目标序列 | 译;分别表示被读取的 encoder 输入序列和产生 decoder query 的生成序列 | [cross-attention](../attention/cross-attention/) |
| source visibility | 源可见性 | 译;规定每个目标 query 可以读取哪些 source key 的连接条件,可包含 padding、prefix 或对齐窗口 | [cross-attention](../attention/cross-attention/) |
| static source KV cache | 静态源 KV cache | 译;source encoder 输出投影出的 K/V 在自回归生成期间不增长,可预先计算并复用 | [cross-attention](../attention/cross-attention/) |
| target query mask | 目标 query 掩码 | 译;标记 target padding 或无效目标行,防止其输出或 loss 进入有效路径 | [cross-attention](../attention/cross-attention/) |
| beam reorder | beam 重排 | 译;beam search 选出候选后按新 beam 顺序重排 target state,并保持 source cache 的 batch 对齐 | [cross-attention](../attention/cross-attention/) |
| gating mechanism | 门控机制 | 译 | [vectors](../linear-algebra/vectors/) |
| cosine similarity | 余弦相似度 | 译;归一化内积,只比较方向不比较长度 | [embedding-geometry](../text-representation/embedding-geometry/) |
| cosine distance / angular distance | 余弦差异 / 角距离 | 译;余弦差异常写为 $1-\cos$,角距离为 $\arccos(\cos)$,后者在单位球面上满足距离公理 | [cosine-similarity](../text-representation/cosine-similarity/) |
| total variance | 总方差 | 译;协方差矩阵迹,等于各坐标方差之和 | [trace](../linear-algebra/trace/) |
| principal component | 主成分 | 译;协方差矩阵最大特征值对应的方向及其投影 | [pca](../linear-models/pca/) |
| whitening | 白化 | 译;用协方差特征分解去相关并把各方向缩放到单位方差 | [pca](../linear-models/pca/) |
| machine learning / learning problem | 机器学习 / 学习问题 | 译;从有限样本中选择面向新输入的预测或行动规则 | [what-is-learning](../learning-framework/what-is-learning/) |
| hypothesis / hypothesis space | 假设 / 假设空间 | 译;一个候选规则及所有允许规则组成的集合 | [hypothesis-spaces](../learning-framework/hypothesis-spaces/) |
| version space | 版本空间 | 译;与训练样本标签完全一致的候选假设子集 | [hypothesis-spaces](../learning-framework/hypothesis-spaces/) |
| hypothesis class capacity | 假设类容量 | 译;假设空间在数据上能表达的标注模式或有效复杂度 | [hypothesis-spaces](../learning-framework/hypothesis-spaces/) |
| growth function | 增长函数 | 译;在 n 个输入点上假设类最多能产生的不同标注模式数量 | [hypothesis-spaces](../learning-framework/hypothesis-spaces/) |
| effective complexity | 有效复杂度 | 译;用于控制泛化偏差的候选空间复杂程度,不必等于参数数量 | [hypothesis-spaces](../learning-framework/hypothesis-spaces/) |
| realizable setting | 可实现情形 | 译;假设空间中存在一个规则能完全符合目标标签的学习设定 | [hypothesis-spaces](../learning-framework/hypothesis-spaces/) |
| agnostic learning | 不依赖可实现假设的学习 | 译;不假定假设空间内存在零误差规则而直接比较总体风险的设定 | [hypothesis-spaces](../learning-framework/hypothesis-spaces/) |
| hyperparameter | 超参数 | 译;训练前设定的空间、正则化或优化选择,不是由单次拟合直接得到的参数 | [hypothesis-spaces](../learning-framework/hypothesis-spaces/) |
| model selection | 模型选择 | 译;用训练外证据在候选假设空间或学习器之间选择方案 | [model-selection](../evaluation-and-generalization/model-selection/) |
| structural risk minimization / SRM | 结构风险最小化 / SRM | 译并保留缩写;在经验风险之外加入候选复杂度项后选择方案的原则 | [model-selection](../evaluation-and-generalization/model-selection/) |
| Akaike information criterion / AIC | 赤池信息准则 / AIC | 译并保留缩写;用最大对数似然和两倍参数数目的惩罚比较统计模型 | [model-selection](../evaluation-and-generalization/model-selection/) |
| one-standard-error rule | 一标准误规则 | 译;在最优分数加一个标准误的范围内选择复杂度较低的候选 | [model-selection](../evaluation-and-generalization/model-selection/) |
| no-free-lunch theorem | 无免费午餐定理 | 保留常用译法;没有额外任务假设时不存在对所有目标都占优的学习算法 | [no-free-lunch](../learning-framework/no-free-lunch/) |
| task distribution / task family | 任务分布 / 任务族 | 译;目标函数或学习任务的概率集合,决定哪些未见规律更可能出现 | [no-free-lunch](../learning-framework/no-free-lunch/) |
| task prior | 任务先验 | 译;观测训练样本前对目标函数可能性的权重分配 | [no-free-lunch](../learning-framework/no-free-lunch/) |
| learning algorithm / learner | 学习算法 / 学习器 | 译;把样本、反馈和假设空间映射为一个已选择规则的过程或系统 | [what-is-learning](../learning-framework/what-is-learning/) |
| data-generating distribution | 数据生成分布 | 译;产生输入和标签样本的未知联合分布 | [what-is-learning](../learning-framework/what-is-learning/) |
| generalization gap | 泛化间隙 | 译;同分布总体风险与经验风险之间的差,需与训练到部署的分布偏移分开 | [generalization-intuition](../evaluation-and-generalization/generalization-intuition/) |
| generalization bound | 泛化界 | 译;以复杂度、样本量和置信度上界控制总体风险与经验风险差距的概率保证 | [generalization-intuition](../evaluation-and-generalization/generalization-intuition/) |
| uniform convergence | 统一收敛 | 译;同一个样本集上同时控制候选假设类经验风险与总体风险差距的性质 | [generalization-intuition](../evaluation-and-generalization/generalization-intuition/) |
| Rademacher complexity | Rademacher 复杂度 | 保留人名并译;衡量函数类在给定输入上配合随机符号能力的复杂度 | [generalization-intuition](../evaluation-and-generalization/generalization-intuition/) |
| algorithmic stability | 算法稳定性 | 译;替换一个训练样本后学习器对测试损失变化不大的性质 | [generalization-intuition](../evaluation-and-generalization/generalization-intuition/) |
| multiple comparisons / selection overfitting | 多重比较 / 选择过拟合 | 译;候选方案越多越容易从开发集噪声中挑出偶然低分结果的现象 | [generalization-intuition](../evaluation-and-generalization/generalization-intuition/) |
| inductive bias | 归纳偏置 | 译;在有限样本下偏向某类规则的结构、先验或算法倾向 | [what-is-learning](../learning-framework/what-is-learning/) |
| approximation error | 表达误差 | 译;假设空间内最优风险与所有规则最优风险之间的差 | [what-is-learning](../learning-framework/what-is-learning/) |
| universal approximation | 通用逼近 | 译;在指定紧域和误差范数下,用足够宽的非线性网络逼近任意连续目标函数的存在性结论 | [universal-approximation](../neurons-and-activations/universal-approximation/) |
| uniform approximation | 一致逼近 | 译;用整个域上的最大绝对误差控制函数近似质量的方式 | [universal-approximation](../neurons-and-activations/universal-approximation/) |
| compact set | 紧集 | 译;在欧氏空间中闭且有界的集合,连续函数在其上具有一致连续性 | [universal-approximation](../neurons-and-activations/universal-approximation/) |
| modulus of continuity | 模连续性 | 译;用输入距离上限控制函数值变化上限的函数 $\omega_f(\delta)$ | [universal-approximation](../neurons-and-activations/universal-approximation/) |
| piecewise-linear interpolation | 分段线性插值 | 译;在相邻网格点之间用直线连接目标值的近似函数 | [universal-approximation](../neurons-and-activations/universal-approximation/) |
| estimation error | 估计误差 | 译;有限样本使经验最优规则偏离分布上最优规则的误差 | [what-is-learning](../learning-framework/what-is-learning/) |
| optimization error | 优化误差 | 译;实际算法输出的经验风险高于假设空间内经验最优值的部分 | [what-is-learning](../learning-framework/what-is-learning/) |
| memorization | 记忆化 | 译;逐个拟合已见样本而没有形成可迁移规则的行为 | [what-is-learning](../learning-framework/what-is-learning/) |
| distribution shift | 分布偏移 | 译;训练数据分布与部署数据分布不一致的情况,需进一步定位输入、类别比例或标签规则的变化 | [distribution-shift](../evaluation-and-generalization/distribution-shift/) |
| deployment distribution | 部署分布 | 译;模型实际运行时面对的输入和标签联合分布,风险应相对于它计算 | [distribution-shift](../evaluation-and-generalization/distribution-shift/) |
| covariate shift | 协变量偏移 | 译;输入边缘分布改变而条件标签分布近似不变的偏移 | [distribution-shift](../evaluation-and-generalization/distribution-shift/) |
| label shift / prior probability shift | 标签偏移 / 先验概率偏移 | 译;类别先验改变而类别条件输入分布近似不变的偏移 | [distribution-shift](../evaluation-and-generalization/distribution-shift/) |
| concept shift / conditional shift | 概念偏移 / 条件偏移 | 译;输入到标签的条件分布改变,旧标签规则不再直接适用 | [distribution-shift](../evaluation-and-generalization/distribution-shift/) |
| importance weighting | 重要性加权 | 译;用部署分布与训练分布的密度比重加训练样本,估计目标风险 | [distribution-shift](../evaluation-and-generalization/distribution-shift/) |
| density ratio | 密度比 | 译;目标分布密度除以参考分布密度,协变量偏移下用于风险重加权 | [distribution-shift](../evaluation-and-generalization/distribution-shift/) |
| support overlap / positivity | 支持集重叠 / 正性条件 | 译;部署出现的输入区域需被训练分布覆盖,否则密度比无法从样本识别 | [distribution-shift](../evaluation-and-generalization/distribution-shift/) |
| domain classifier | 域分类器 | 译;区分训练域与部署域样本的分类器,可用于发现差异和估计密度比 | [distribution-shift](../evaluation-and-generalization/distribution-shift/) |
| irreducible noise | 不可约噪声 | 译;在给定输入下标签仍有随机性的无法消除误差来源 | [what-is-learning](../learning-framework/what-is-learning/) |
| feedback signal | 反馈信号 | 译;学习问题用来评价预测或行动的标签、重构误差或奖励 | [what-is-learning](../learning-framework/what-is-learning/) |
| supervised learning | 监督学习 | 译;从带标签样本学习输入到目标输出的规则 | [supervised-learning](../learning-framework/supervised-learning/) |
| labeled dataset / supervised dataset | 带标签数据集 / 监督数据集 | 译;由输入与目标标签成对组成,用于计算监督损失的数据集 | [supervised-learning](../learning-framework/supervised-learning/) |
| feature / input feature | 特征 / 输入特征 | 译;模型在预测时可见的输入变量或其表示 | [supervised-learning](../learning-framework/supervised-learning/) |
| label / target | 标签 / 目标 | 译;监督样本中希望模型预测或估计的输出 | [supervised-learning](../learning-framework/supervised-learning/) |
| regression | 回归 | 译;以连续数值或数值向量为主要输出的监督任务 | [supervised-learning](../learning-framework/supervised-learning/) |
| linear regression | 线性回归 | 译;对参数保持线性的数值预测模型,特征可以是原始变量或固定变换 | [linear-regression](../linear-models/linear-regression/) |
| design matrix | 设计矩阵 | 译;按样本排列特征列的矩阵,把参数向量映射为全部训练预测 | [linear-regression](../linear-models/linear-regression/) |
| ordinary least squares / OLS | 普通最小二乘 / OLS | 译并保留缩写;最小化残差平方和的无惩罚线性回归解 | [linear-regression](../linear-models/linear-regression/) |
| coefficient / intercept | 系数 / 截距 | 译;系数是特征列的线性权重,截距是全一列对应的平移项 | [linear-regression](../linear-models/linear-regression/) |
| residual | 残差 | 译;观测标签减去模型预测的差,最小二乘残差与设计矩阵列空间正交 | [linear-regression](../linear-models/linear-regression/) |
| hat matrix / leverage | 帽子矩阵 / 杠杆值 | 译;H 把标签映到拟合值,对角线杠杆值表示输入位置对拟合的几何约束强度 | [linear-regression](../linear-models/linear-regression/) |
| multicollinearity | 多重共线性 | 译;特征列近似线性相关,可使单个系数不稳定而整体预测仍较稳定 | [linear-regression](../linear-models/linear-regression/) |
| ridge regression | 岭回归 | 译;在残差平方和外加入 L2 参数惩罚以换取收缩和更低方差 | [linear-regression](../linear-models/linear-regression/) |
| Lasso / L1 regularization | Lasso / L1 正则化 | 保留常用名称并译;用 L1 惩罚产生可为零的稀疏系数 | [ridge-and-lasso](../linear-models/ridge-and-lasso/) |
| soft-thresholding | 软阈值 | 译;把绝对值超过阈值的坐标向零收缩,低于阈值的坐标置零 | [ridge-and-lasso](../linear-models/ridge-and-lasso/) |
| coordinate descent | 坐标下降 | 译;固定其余坐标后轮流优化单个参数的算法 | [ridge-and-lasso](../linear-models/ridge-and-lasso/) |
| elastic net | Elastic Net | 保留常用名称;混合 L1 稀疏与 L2 收缩的正则化 | [ridge-and-lasso](../linear-models/ridge-and-lasso/) |
| logistic regression | 逻辑回归 | 译;对数几率为特征线性函数的 Bernoulli 概率模型 | [logistic-regression](../linear-models/logistic-regression/) |
| sigmoid | sigmoid 函数 | 保留常用名称;把实数分数映射到 0 和 1 之间的单调函数 | [heaviside-and-sigmoid](../neurons-and-activations/heaviside-and-sigmoid/) |
| logit / odds | 对数几率 / 几率 | 译;odds 是正类与负类概率之比,logit 是其对数 | [heaviside-and-sigmoid](../neurons-and-activations/heaviside-and-sigmoid/) |
| decision threshold | 决策阈值 | 译;把概率转换为类别或行动的分界值,与概率训练目标分开选择 | [heaviside-and-sigmoid](../neurons-and-activations/heaviside-and-sigmoid/) |
| complete separation | 完全分离 | 译;二分类或成对比较中存在单向完全分离,使无正则化最大似然参数趋向无穷 | [logistic-regression](../linear-models/logistic-regression/) |
| iteratively reweighted least squares / IRLS | 迭代重加权最小二乘 / IRLS | 译并保留缩写;用当前概率产生权重后反复解加权二次近似 | [logistic-regression](../linear-models/logistic-regression/) |
| softmax regression | Softmax 回归 | 译;用共享归一化分母把多组线性 logits 变成互斥类别概率的模型 | [softmax-regression](../linear-models/softmax-regression/) |
| softmax | softmax 函数 | 保留常用名称;把一组实数分数沿指定轴归一化为和为 1 的概率向量 | [softmax](../neurons-and-activations/softmax/) |
| probability simplex | 概率单纯形 | 译;分量非负且总和为 1 的概率向量集合,softmax 的输出落在其内部 | [softmax](../neurons-and-activations/softmax/) |
| softmax Jacobian | softmax Jacobian | 保留常用名称;记录共享归一化分母造成的坐标耦合导数矩阵 | [softmax](../neurons-and-activations/softmax/) |
| masked softmax | 掩码 softmax | 译;先排除不允许位置再在有效集合内归一化的 softmax | [softmax](../neurons-and-activations/softmax/) |
| normalization axis | 归一化轴 | 译;张量中代表竞争集合并沿其求和归一化的坐标轴 | [softmax](../neurons-and-activations/softmax/) |
| logits | logits / 未归一化分数 | 保留常用名称并解释;送入 softmax 或分类损失前的任意实数分数 | [softmax](../neurons-and-activations/softmax/) |
| log-softmax | log-softmax | 保留常用名称;以稳定 log-sum-exp 计算的对数概率输出 | [softmax](../neurons-and-activations/softmax/) |
| temperature scaling | 温度缩放 | 译;用验证集温度调整 logits 概率尖锐程度的校准方法 | [softmax](../neurons-and-activations/softmax/) |
| top-k accuracy | top-k 准确率 | 保留常用形式;真实类别出现在预测概率前 k 名的比例 | [softmax-regression](../linear-models/softmax-regression/) |
| one-vs-rest | one-vs-rest / 一对其余 | 保留常用形式;为每个类别训练独立二分类器的多分类方案 | [softmax-regression](../linear-models/softmax-regression/) |
| classification metric | 分类指标 | 译;从标签、分数或硬预测汇总分类模型表现的统计量 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| confusion matrix | 混淆矩阵 | 译;按真实类别与预测类别交叉记录 TP、FP、FN、TN 或多分类计数的表 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| true positive / TP | 真阳性 / TP | 译并保留缩写;真实为正类且预测为正类的样本 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| false positive / FP | 假阳性 / FP | 译并保留缩写;真实为负类却预测为正类的样本 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| false negative / FN | 假阴性 / FN | 译并保留缩写;真实为正类却预测为负类的样本 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| true negative / TN | 真阴性 / TN | 译并保留缩写;真实为负类且预测为负类的样本 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| accuracy / error rate | 准确率 / 错误率 | 译;分别是判对样本比例与判错样本比例 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| precision / positive predictive value / PPV | 精确率 / 阳性预测值 / PPV | 译并保留缩写;预测为正类的样本中真实为正类的比例 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| recall / true positive rate / sensitivity / TPR | 召回率 / 真阳性率 / 灵敏度 / TPR | 译并保留缩写;真实正类中被预测为正类的比例 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| specificity / true negative rate / TNR | 特异度 / 真阴性率 / TNR | 译并保留缩写;真实负类中被预测为负类的比例 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| negative predictive value / NPV | 阴性预测值 / NPV | 译并保留缩写;预测为负类的样本中真实为负类的比例 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| F1 score / F-beta score | F1 分数 / Fβ 分数 | 译并保留写法;分别用 precision 与 recall 的调和平均或带权调和平均平衡两类错误 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| balanced accuracy | 平衡准确率 | 译;二分类中正类召回率与负类特异度的平均 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| Matthews correlation coefficient / MCC | Matthews 相关系数 / MCC | 保留人名并译;同时使用二分类混淆矩阵四个格子的相关性指标 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| false positive rate / FPR | 假阳性率 / FPR | 译并保留缩写;真实负类中被误报为正类的比例 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| class prevalence / base rate | 类别率 / 基率 | 译;评估分布中正类或某类别所占的真实比例 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| ROC curve / ROC-AUC | ROC 曲线 / ROC-AUC | 保留缩写并译;扫描阈值时以 FPR 为横轴、TPR 为纵轴及其面积表示排序能力 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| precision-recall curve / PR curve | 精确率—召回率曲线 / PR 曲线 | 译;扫描阈值时以 recall 与 precision 表示正类检索取舍的曲线 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| average precision / AP | 平均精度 / AP | 译并保留缩写;按正类检索位置汇总 precision 的排序指标 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| macro average / micro average / weighted average | macro 平均 / micro 平均 / 加权平均 | 保留常用写法并译;分别按类别等权、先汇总计数或按支持数聚合指标 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| Brier score | Brier 分数 | 保留人名;预测概率与二元标签之间的均方误差 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| log loss | 对数损失 | 译;按真实标签惩罚错误概率的负对数似然指标 | [classification-metrics](../evaluation-and-generalization/classification-metrics/) |
| regression metric | 回归指标 | 译;从连续目标、预测值和部署代价汇总回归模型表现的统计量 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| squared loss / L2 loss | 平方损失 / L2 损失 | 译;以残差平方为点损失,对应条件均值并放大大残差 | [regression-losses](../training-nn/regression-losses/) |
| absolute loss / L1 loss | 绝对损失 / L1 损失 | 译;以残差绝对值为点损失,对应条件中位数且对离群点更稳健 | [regression-losses](../training-nn/regression-losses/) |
| pseudo-Huber loss | pseudo-Huber 损失 | 保留常用名称并译;用平滑平方根替代 Huber 折点的稳健损失 | [regression-losses](../training-nn/regression-losses/) |
| log-cosh loss | log-cosh 损失 | 保留常用名称并译;以 $\log\cosh(r)$ 为点损失,梯度为 $\tanh(r)$ | [regression-losses](../training-nn/regression-losses/) |
| Gaussian negative log-likelihood | Gaussian 负对数似然 | 译并保留分布名;同时学习条件均值与对数方差的概率回归目标 | [regression-losses](../training-nn/regression-losses/) |
| heteroscedastic regression | 异方差回归 | 译;让条件方差随输入变化而不是对所有样本共享一个噪声尺度的回归 | [regression-losses](../training-nn/regression-losses/) |
| quantile crossing | 分位数交叉 | 译;独立预测多个分位数时出现低分位数高于高分位数的次序违例 | [regression-losses](../training-nn/regression-losses/) |
| mean absolute error / MAE | 平均绝对误差 / MAE | 译并保留缩写;残差绝对值的平均,保持目标原单位且不特别放大大错 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| mean squared error / MSE | 均方误差 / MSE | 译并保留缩写;残差平方的平均,对大残差施加更强惩罚 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| root mean squared error / RMSE | 均方根误差 / RMSE | 译并保留缩写;MSE 开平方后回到目标原单位的指标 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| coefficient of determination / R-squared / R² | 决定系数 / R² | 译并保留写法;相对均值基线的残差平方和改善比例,可以小于零 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| residual sum of squares / SSE | 残差平方和 / SSE | 译并保留缩写;模型预测残差平方的总和 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| total sum of squares / SST | 总平方和 / SST | 译并保留缩写;真实值相对评估集均值的平方偏差总和 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| mean absolute percentage error / MAPE | 平均绝对百分比误差 / MAPE | 译并保留缩写;按真实值绝对规模归一化的平均绝对误差,零值时未定义 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| symmetric mean absolute percentage error / sMAPE | 对称平均绝对百分比误差 / sMAPE | 译并保留缩写;用真实值与预测值绝对规模之和作为分母的相对误差 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| weighted absolute percentage error / WAPE | 加权绝对百分比误差 / WAPE | 译并保留缩写;所有绝对误差之和除以所有真实值绝对规模之和 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| Huber loss | Huber 损失 | 保留人名并译;小残差用平方、大残差用线性的稳健损失 | [regression-losses](../training-nn/regression-losses/) |
| pinball loss / quantile loss | 分位数损失 / pinball loss | 译并保留常用名称;用不对称线性惩罚学习条件分位数的损失 | [regression-losses](../training-nn/regression-losses/) |
| prediction interval | 预测区间 | 译;给出未来真实值可能范围并需同时评估覆盖率与宽度的不确定性输出 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| interval coverage / interval width | 区间覆盖率 / 区间宽度 | 译;分别统计真实值落入区间的比例与区间平均长度 | [regression-metrics](../evaluation-and-generalization/regression-metrics/) |
| perceptron | 感知机 | 译;用线性分数和硬阈值进行分类的错误驱动模型 | [perceptron-classic](../linear-models/perceptron-classic/) |
| perceptron learning rule | 感知机学习规则 | 译;对 margin 不正的样本按标签和输入方向更新参数的规则 | [perceptron-classic](../linear-models/perceptron-classic/) |
| signed margin | 带符号 margin | 保留常用词并解释;标签乘以线性分数,正值表示样本在正确一侧 | [perceptron-classic](../linear-models/perceptron-classic/) |
| linear separability | 线性可分性 | 译;存在一个仿射超平面把所有样本放在各自正确一侧的性质 | [perceptron-classic](../linear-models/perceptron-classic/) |
| perceptron convergence theorem | 感知机收敛定理 | 译;线性可分且存在正 margin 时感知机错误更新次数有限的定理 | [perceptron-classic](../linear-models/perceptron-classic/) |
| hinge loss | hinge loss / 合页损失 | 保留常用形式并译;惩罚带符号 margin 小于 1 的分类损失,margin 达到 1 后归零 | [classification-losses](../training-nn/classification-losses/) |
| activation function / nonlinearity | 激活函数 / 非线性 | 译;作用在神经元预激活值上的逐分量函数,给层间表示加入非线性并改变梯度尺度 | [activation-functions](../neurons-and-activations/activation-functions/) |
| pre-activation | 预激活值 | 译;仿射变换 W x+b 在经过激活函数前的标量或向量 | [activation-functions](../neurons-and-activations/activation-functions/) |
| single neuron / artificial neuron | 单个神经元 / 人工神经元 | 译;把输入特征做加权和加偏置并经过激活得到一个输出的最小计算单元 | [backprop-single-neuron](../backpropagation/backprop-single-neuron/) |
| error signal / delta | 误差信号 / δ | 译并保留记号;损失对神经元预激活或 logit 的导数,把上游敏感度传给仿射输入 | [backprop-single-neuron](../backpropagation/backprop-single-neuron/) |
| hidden layer / hidden representation | 隐藏层 / 隐藏表示 | 译;输入与输出头之间的可学习层及其中间向量,不直接等同于最终概率或目标 | [backprop-two-layer](../backpropagation/backprop-two-layer/) |
| layerwise error signal | 逐层误差信号 | 译;损失对每层预激活的梯度,按下一层转置权重和当前激活导数逐层传回 | [backprop-two-layer](../backpropagation/backprop-two-layer/) |
| vectorized backpropagation / vectorization | 向量化反向传播 / 向量化 | 译;保留样本轴把逐样本链式法则合并为矩阵乘法、逐分量运算和沿共享轴归约,不改变模型或损失语义 | [vectorized-backprop](../backpropagation/vectorized-backprop/) |
| row-batch layout | row-batch 布局 / 每样本一行 | 保留常用名称并译;把 batch 的样本放在矩阵行、特征放在列的张量约定 | [vectorized-backprop](../backpropagation/vectorized-backprop/) |
| GEMM / general matrix multiplication | GEMM / 通用矩阵乘法 | 保留缩写并译;底层线性代数库用于高效执行密集矩阵乘法的计算内核,不是梯度正确性的证明 | [vectorized-backprop](../backpropagation/vectorized-backprop/) |
| identity activation | identity 激活 / 恒等激活 | 保留常用名称并译;输出等于输入,常用于无界回归头但不提供隐藏层非线性 | [activation-functions](../neurons-and-activations/activation-functions/) |
| step function / Heaviside function | 阶跃函数 / Heaviside 函数 | 译并保留人名;在阈值处跳变的硬判定函数,几乎处处导数为零 | [heaviside-and-sigmoid](../neurons-and-activations/heaviside-and-sigmoid/) |
| logistic function | logistic 函数 | 保留常用名称;由微分方程或硬阈值平滑得到的 sigmoid 函数 | [heaviside-and-sigmoid](../neurons-and-activations/heaviside-and-sigmoid/) |
| sigmoid derivative | sigmoid 导数 | 译;$\sigma'(z)=\sigma(z)(1-\sigma(z))$,在零点达到最大值 1/4 | [heaviside-and-sigmoid](../neurons-and-activations/heaviside-and-sigmoid/) |
| BCE with logits | 带 logits 的二元交叉熵 | 译;直接在未归一化分数上稳定计算二元交叉熵并避免重复 sigmoid | [classification-losses](../training-nn/classification-losses/) |
| tanh activation | tanh 激活 | 保留常用名称;输出在 (-1,1) 的零中心双曲正切激活,大幅值处会饱和 | [tanh](../neurons-and-activations/tanh/) |
| hyperbolic tangent | 双曲正切 | 译;由双曲正弦除以双曲余弦定义的奇函数,把实数映射到 (-1,1) | [tanh](../neurons-and-activations/tanh/) |
| tanh derivative | tanh 导数 | 译;$\tanh'(z)=1-\tanh^2(z)$,在零点达到 1 并在两端衰减 | [tanh](../neurons-and-activations/tanh/) |
| inverse hyperbolic tangent / artanh | 反双曲正切 / artanh | 译并保留记号;把 (-1,1) 输出映回无界实数的反函数 | [tanh](../neurons-and-activations/tanh/) |
| ReLU / rectified linear unit | ReLU / 修正线性单元 | 保留缩写并译;取 max(0,z) 的分段线性激活,正侧不饱和但负侧有死区 | [relu](../neurons-and-activations/relu/) |
| ReLU derivative / activity mask | ReLU 导数 / 活动掩码 | 译;在零点外由 0/1 门决定梯度是否通过,零点采用实现约定的次梯度 | [relu](../neurons-and-activations/relu/) |
| He initialization | He 初始化 | 保留人名;针对 ReLU 以权重方差约为 2/n 维持激活二阶矩的初始化 | [relu](../neurons-and-activations/relu/) |
| weight initialization | 权重初始化 | 译;训练开始前为参数设定分布,同时打破神经元对称并控制前向激活与反向梯度的尺度 | [weight-initialization](../training-nn/weight-initialization/) |
| fan-in / fan-out | fan-in / fan-out | 保留常用名称;分别表示一个输出单元接收的输入数与一个输入单元连接的输出数 | [weight-initialization](../training-nn/weight-initialization/) |
| Xavier / Glorot initialization | Xavier / Glorot 初始化 | 保留人名;以 2/(fan-in+fan-out) 的权重方差折中前向与反向传播 | [weight-initialization](../training-nn/weight-initialization/) |
| LeCun initialization | LeCun 初始化 | 保留人名;以约 1/fan-in 的权重方差维持输入侧尺度 | [weight-initialization](../training-nn/weight-initialization/) |
| variance propagation | 方差传播 | 译;用层宽、权重方差和激活导数追踪前向二阶矩或反向梯度的逐层乘数 | [weight-initialization](../training-nn/weight-initialization/) |
| symmetry breaking | 对称打破 | 译;用不同初始权重让原本相同的神经元收到不同信号并形成不同梯度 | [weight-initialization](../training-nn/weight-initialization/) |
| dead ReLU | 死亡 ReLU | 译;神经元对数据长期处于负侧,输出和权重梯度都接近 0 的状态 | [dead-relu-and-leaky](../neurons-and-activations/dead-relu-and-leaky/) |
| activation rate / dead-neuron fraction | 活动率 / 死亡神经元比例 | 译;用预激活为正的样本比例监控稀疏性,再按跨 batch 长期低活动识别死亡候选 | [dead-relu-and-leaky](../neurons-and-activations/dead-relu-and-leaky/) |
| negative slope | 负侧斜率 | 译;Leaky ReLU 或 PReLU 在负侧保留的正斜率参数 | [dead-relu-and-leaky](../neurons-and-activations/dead-relu-and-leaky/) |
| Leaky ReLU | Leaky ReLU / 带泄漏 ReLU | 保留常用名称并译;在负侧保留小正斜率以减少死亡 ReLU | [dead-relu-and-leaky](../neurons-and-activations/dead-relu-and-leaky/) |
| PReLU / parametric ReLU | PReLU / 参数化 ReLU | 保留缩写并译;把负侧斜率作为可学习参数的 ReLU 变体 | [dead-relu-and-leaky](../neurons-and-activations/dead-relu-and-leaky/) |
| softplus | softplus | 保留常用名称;log(1+exp(z)) 的平滑 ReLU 近似,导数为 sigmoid | [activation-functions](../neurons-and-activations/activation-functions/) |
| Gaussian error linear unit / GELU | Gaussian error linear unit / GELU | 保留英文和缩写;用标准正态分布函数对输入做平滑门控的激活 | [gelu](../neurons-and-activations/gelu/) |
| exact GELU | 精确 GELU | 译;按 xΦ(x) 或 erf 形式计算的 GELU,区别于 tanh 近似 | [gelu](../neurons-and-activations/gelu/) |
| approximate GELU | GELU 近似 | 译;用 tanh 或 sigmoid 近似标准正态 CDF 以降低实现成本的形式 | [gelu](../neurons-and-activations/gelu/) |
| GELU derivative | GELU 导数 | 译;$\operatorname{GELU}'(x)=\Phi(x)+x\phi(x)$,负侧可为负且正侧可超过 1 | [gelu](../neurons-and-activations/gelu/) |
| Swish / SiLU | Swish / SiLU | 保留常用名称;输入与 sigmoid(βz) 相乘的平滑门控激活,β=1 时为 SiLU | [swish-and-swiglu](../neurons-and-activations/swish-and-swiglu/) |
| GLU / gated linear unit | GLU / 门控线性单元 | 译并保留缩写;用一条值分支逐坐标乘以 sigmoid 门分支的两投影结构 | [swish-and-swiglu](../neurons-and-activations/swish-and-swiglu/) |
| SwiGLU | SwiGLU | 保留缩写;用 SiLU 门分支逐坐标缩放值分支的门控前馈结构 | [swiglu-ffn](../transformer-components/swiglu-ffn/) |
| GEGLU / ReGLU | GEGLU / ReGLU | 保留变体名;分别用 GELU 或 ReLU 替换 GLU 的门函数 | [swish-and-swiglu](../neurons-and-activations/swish-and-swiglu/) |
| activation saturation | 激活饱和 | 译;输入继续变大而输出变化变小,局部导数接近 0 的区域 |
| vanishing gradient | 梯度消失 | 译;反向路径上的权重与局部 Jacobian 连乘后,早期参数收到的梯度趋近 0 | [vanishing-and-exploding](../backpropagation/vanishing-and-exploding/) |
| exploding gradient | 梯度爆炸 | 译;反向路径上的局部增益连乘后,梯度范数快速变大的现象 | [vanishing-and-exploding](../backpropagation/vanishing-and-exploding/) |
| local Jacobian | 局部 Jacobian | 译;一层从输入扰动到输出扰动的导数矩阵,激活逐分量时含对角导数矩阵 | [chain-rule-on-graphs](../backpropagation/chain-rule-on-graphs/) |
| mean log gain | 平均对数增益 | 译;逐层局部增益绝对值取对数后的平均,用来判断深度方向的衰减趋势 | [vanishing-and-exploding](../backpropagation/vanishing-and-exploding/) |
| gradient norm | 梯度范数 | 译;用 L2、无穷范数或分位数概括一层、一个参数组或整批梯度的大小,不能替代逐方向统计 | [vanishing-and-exploding](../backpropagation/vanishing-and-exploding/) |
| batch normalization / BatchNorm | 批量归一化 / BatchNorm | 译并保留缩写;按 batch 与特征或通道的统计轴标准化激活,再用可学习尺度和偏移恢复表达能力 | [batch-normalization](../training-nn/batch-normalization/) |
| batch statistics | 批次统计量 | 译;当前训练 batch 上计算的均值与方差,会让同一统计集合中的样本相互耦合 | [batch-normalization](../training-nn/batch-normalization/) |
| running mean / running variance | 运行均值 / 运行方差 | 译;训练期间按指数更新、评估时替代即时 batch 统计的 buffer | [batch-normalization](../training-nn/batch-normalization/) |
| affine parameters / scale and shift | 仿射参数 / 缩放与平移 | 译;BatchNorm 中可学习的 $\gamma,\beta$,分别恢复尺度与偏移 | [batch-normalization](../training-nn/batch-normalization/) |
| SyncBatchNorm / SyncBN | 同步批量归一化 / SyncBN | 保留缩写并译;跨设备归约样本数、总和与平方和后计算全局通道统计 | [batch-normalization](../training-nn/batch-normalization/) |
| LayerNorm / layer normalization | LayerNorm / 层归一化 | 保留缩写并译;对单个 token 的特征轴计算总体均值与方差,再用 $\gamma,\beta$ 做仿射恢复,不依赖 batch 统计 | [layernorm](../transformer-components/layernorm/) |
| normalized shape | normalized shape / 归一化 shape | 保留常用术语并译;决定 LayerNorm 把输入的哪些尾部轴放入同一个统计集合,并约束 $\gamma,\beta$ 的 shape | [layernorm](../transformer-components/layernorm/) |
| population variance in normalization | 归一化总体方差 | 译;LayerNorm 沿归约轴使用分母 $d$ 的平方偏差平均,不使用样本方差的 $d-1$ 分母 | [layernorm](../transformer-components/layernorm/) |
| LayerNorm epsilon | LayerNorm epsilon / LayerNorm 的 $\epsilon$ | 保留常用术语并译;加入方差后再开平方的正数,同时决定低方差输入的实际增益 | [layernorm](../transformer-components/layernorm/) |
| token-local normalization | token-local normalization / token 局部归一化 | 保留常用术语并译;每个 token 独立沿特征轴计算统计量,不跨 token 或 batch 共享即时均值和方差 | [layernorm](../transformer-components/layernorm/) |
| GroupNorm / group normalization | GroupNorm / 组归一化 | 保留缩写并译;在单个样本内按通道组和空间位置计算统计,适合小 batch 边界 | [batch-normalization](../training-nn/batch-normalization/) |
| Ghost BatchNorm | Ghost BatchNorm / 虚拟批量归一化 | 保留常用名称并译;把大 batch 拆成虚拟小 batch 分别计算统计以保留噪声 | [batch-normalization](../training-nn/batch-normalization/) |
| mixed-precision training / automatic mixed precision / AMP | 混合精度训练 / 自动混合精度 / AMP | 译并保留缩写;让不同算子、归约、梯度和状态使用不同 dtype,以降低存储和计算成本并控制数值风险 | [mixed-precision](../training-nn/mixed-precision/) |
| float16 / binary16 | float16 / binary16 半精度 | 保留格式名并译;5 个指数位和 10 个尾数位的二进制浮点格式,范围较窄但同量级间隔比 bfloat16 更细 | [mixed-precision](../training-nn/mixed-precision/) |
| bfloat16 | bfloat16 脑浮点 | 保留格式名并译;沿用 float32 的指数位宽而减少尾数位,通常覆盖更宽范围但相对精度较粗 | [mixed-precision](../training-nn/mixed-precision/) |
| loss scaling / gradient scaling | 损失缩放 / 梯度缩放 | 译;在反向前把损失乘以缩放因子,再在优化器更新前 unscale 以避免小梯度下溢 | [mixed-precision](../training-nn/mixed-precision/) |
| dynamic loss scaling | 动态损失缩放 | 译;按梯度是否有限自动增减 scale,溢出时跳过更新并回退缩放因子 | [mixed-precision](../training-nn/mixed-precision/) |
| master weights | 主权重 | 译;训练中以更高精度保存并累积参数更新的权重副本,再转换为前向使用的低精度副本 | [mixed-precision](../training-nn/mixed-precision/) |
| autocast | autocast / 自动类型转换 | 保留常用名称并译;按算子策略在作用域内自动选择输入和计算 dtype,不等于所有算子都降为低精度 | [mixed-precision](../training-nn/mixed-precision/) |
| GradScaler | GradScaler / 梯度缩放器 | 保留 API 名称并译;管理 scale、unscale、非有限梯度检查和溢出跳步的训练状态 | [mixed-precision](../training-nn/mixed-precision/) |
| non-finite gradient | 非有限梯度 | 译;含 NaN 或 Inf 的梯度,在 optimizer step 前必须被检测并按协议跳过或处理 | [mixed-precision](../training-nn/mixed-precision/) |
| accumulator dtype | 累加器 dtype | 译;乘加、归约或统计内部保存部分和的数值类型,可能不同于输入和输出 dtype | [mixed-precision](../training-nn/mixed-precision/) |
| multilayer perceptron / MLP | 多层感知机 / MLP | 译并保留缩写;由全连接层和逐坐标非线性激活组成的前馈网络,Transformer FFN 是逐 token 的 MLP 子层 | [feedforward](../transformer-components/feedforward/) |
| training loop | 训练循环 | 译;按 batch 重复前向、损失、反向、优化器更新并按 epoch 评估的执行协议 | [mnist-mlp-training-loop](../training-nn/mnist-mlp-training-loop/) |
| data contract | 数据合同 | 译;对样本轴、形状、dtype、数值范围、标签编码和切分职责的明确接口约束 | [mnist-mlp-training-loop](../training-nn/mnist-mlp-training-loop/) |
| parameter count | 参数量 | 译;模型中需要学习的权重与偏置标量总数,用于核对结构和资源预算 | [parameter-count](../transformer-components/parameter-count/) |
| total parameter count | 总参数量 | 译;按唯一可学习张量的 shape 乘积相加得到的模型参数总数,不把激活、optimizer state 或 KV cache 混入 | [parameter-count](../transformer-components/parameter-count/) |
| trainable parameter count | 可训练参数量 | 译;当前训练任务实际更新的参数标量数,参数高效微调时小于冻结 base model 的 total parameter count | [parameter-count](../transformer-components/parameter-count/) |
| parameter storage | 参数存储 | 译;参数量乘以权重 dtype 字节数后的静态存储,需要另列量化 scale、padding 和 shard metadata | [parameter-count](../transformer-components/parameter-count/) |
| model size | 模型规模 | 译;由参数量、结构配置和权重存储共同描述的模型规模,不直接等同于单次推理显存或延迟 | [parameter-count](../transformer-components/parameter-count/) |
| parameter sharing | 参数共享 | 译;多个层、位置或分支引用同一组可学习张量,参数按唯一存储张量计数而非按引用次数重复计数 | [parameter-count](../transformer-components/parameter-count/) |
| activation ledger | 激活账 | 译;按 $B$、$T$、特征宽度和保存策略记录前向/反向中间张量元素、bytes 与峰值生命周期 | [parameter-count](../transformer-components/parameter-count/) |
| MAC/FLOP accounting | MAC/FLOP 账 | 译;明确一次 MAC 与 FLOP 的换算口径,并把矩阵乘法、逐元素算子、attention 交互和访存分开记录 | [parameter-count](../transformer-components/parameter-count/) |
| optimizer state memory | 优化器状态内存 | 译;梯度、master weight、一阶/二阶矩和 step 等训练状态的存储,不属于 forward 参数量 | [parameter-count](../transformer-components/parameter-count/) |
| KV cache memory | KV cache 内存 | 译;decode 期间按层、batch、历史长度和 K/V head 保存的动态缓存字节数,不属于 trainable parameters | [parameter-count](../transformer-components/parameter-count/) |
| LoRA parameter count | LoRA 参数量 | 译;冻结矩阵上低秩 adapter 的新增可训练参数 $r(d_{\mathrm{in}}+d_{\mathrm{out}})$,不重复计数隐含的 $\Delta W$ | [parameter-count](../transformer-components/parameter-count/) |
| training debugging / debugging a training loop | 训练调试 / 训练循环调试 | 译;按数据、前向、损失、反向、更新、评估和运行状态边界定位训练异常的过程 | [debugging-training](../training-nn/debugging-training/) |
| minimum reproducible example / MRE | 最小可复现例 / MRE | 译并保留缩写;用最少数据、模型、配置和随机状态重现一个训练现象的实验 | [debugging-training](../training-nn/debugging-training/) |
| sanity check | 合理性检查 | 译;在完整训练前用手算、小张量或固定样本验证 shape、范围、损失和更新方向 | [debugging-training](../training-nn/debugging-training/) |
| tiny-batch overfit test | 小批次过拟合测试 | 译;暂时关闭复杂正则并让模型拟合极小固定 batch,用来隔离数据、梯度和更新错误 | [debugging-training](../training-nn/debugging-training/) |
| label alignment | 标签对齐 | 译;确认样本、特征、标签、mask 和权重在 shuffle 与 batch 后仍指向同一观测 | [debugging-training](../training-nn/debugging-training/) |
| activation statistics | 激活统计 | 译;按层记录激活的范围、均值、方差、分位数、非零比例和有限性 | [debugging-training](../training-nn/debugging-training/) |
| gradient statistics | 梯度统计 | 译;按层记录梯度范数、非零比例、分位数和非有限值以诊断反向传播尺度 | [debugging-training](../training-nn/debugging-training/) |
| loss curve | 损失曲线 | 译;按训练 step 或 epoch 记录损失值的序列,必须结合数据切分和日志时间点解释 | [debugging-training](../training-nn/debugging-training/) |
| dropout / random dropout | Dropout / 随机失活 | 译并保留常用名称;训练时按随机掩码置零激活、通常用保留概率补偿尺度,评估时关闭随机性 | [dropout](../training-nn/dropout/) |
| inverted dropout | inverted Dropout / 训练态缩放随机失活 | 译并保留常用名称;训练输出除以保留概率,评估输出直接使用完整激活 | [dropout](../training-nn/dropout/) |
| drop probability / keep probability | 丢弃概率 / 保留概率 | 译;分别表示掩码为 0 与为 1 的概率,必须确认 API 的参数语义 | [dropout](../training-nn/dropout/) |
| dropout mask | Dropout 掩码 | 译;本次前向决定哪些激活保留的 0/1 随机张量,反向必须复用同一掩码 | [dropout](../training-nn/dropout/) |
| activation noise | 激活噪声 | 译;随机层在中间表示上引入的扰动,不等于参数本身的噪声 | [dropout](../training-nn/dropout/) |
| feature/channel dropout | 特征/通道 Dropout | 译;沿指定轴共享掩码以整条丢弃特征或通道,广播轴决定样本和空间是否共享 | [dropout](../training-nn/dropout/) |
| locked / variational dropout | 锁定 / 变分 Dropout | 译;同一序列的多个时间步共享一次掩码,区别于每个时间步重新采样 | [dropout](../training-nn/dropout/) |
| MC Dropout | MC Dropout / 蒙特卡洛 Dropout | 保留常用名称并译;推理时有意重复采样训练态掩码以估计输出不确定性 | [dropout](../training-nn/dropout/) |
| stochastic depth | 随机深度 | 译;按残差分支或 block 随机保留路径,不是逐激活坐标的 Dropout | [dropout](../training-nn/dropout/) |
| gradient clipping | 梯度裁剪 | 译;在参数更新前限制梯度范数或坐标幅度的操作,能限制异常更新但不能修复梯度消失根因 | [gradient-clipping](../training-nn/gradient-clipping/) |
| global norm clipping | 全局范数裁剪 | 译;把所有参数梯度合并后用同一个缩放系数限制整体二范数并保留方向 | [gradient-clipping](../training-nn/gradient-clipping/) |
| clip coefficient | 裁剪系数 | 译;梯度范数超过阈值时使用的缩放因子 $c=\min(1,\tau/(G+\varepsilon))$ | [gradient-clipping](../training-nn/gradient-clipping/) |
| adaptive gradient clipping / AGC | 自适应梯度裁剪 / AGC | 保留缩写;按梯度范数相对参数范数的比例限制更新,阈值不再是固定绝对量 | [gradient-clipping](../training-nn/gradient-clipping/) |
| residual Jacobian | 残差 Jacobian | 译;残差块 h+F(h) 的局部导数 I+J_F,恒等路径可改善传递但其奇异值仍需检查 | [residual-connections](../cnn/residual-connections/) |
| temporal gradient | 时间梯度 | 译;循环状态对较早时间步或输入的损失导数,沿时间展开的 Jacobian 连乘会使其消失或爆炸 | [vanishing-and-exploding](../backpropagation/vanishing-and-exploding/) |
| activation selection | 激活函数选择 | 译;按目标输出约束、隐藏层统计、梯度、架构和部署条件筛选激活函数的过程 |
| output head | 输出头 | 译;把最后一层表示映射为任务所需数值或概率的输出层及其激活/损失组合 | [forward-pass](../backpropagation/forward-pass/) |
| convolutional neural network / CNN | 卷积神经网络 / CNN | 译并保留缩写;把局部连接、权值共享和空间布局作为图像归纳偏置的网络 | [cnn](../cnn/cnn/) |
| local connectivity | 局部连接 | 译;一个输出位置只读取输入的局部窗口而不是整张图 | [cnn](../cnn/cnn/) |
| weight sharing | 权值共享 | 译;同一组卷积核参数在不同空间位置重复使用 | [cnn](../cnn/cnn/) |
| feature map / activation map | 特征图 / 激活图 | 译;一个核在所有空间位置的响应集合,也可指经过激活后的中间张量 | [cnn](../cnn/cnn/) |
| convolution kernel / filter | 卷积核 / 滤波器 | 译;对局部窗口做加权求和以产生响应的可学习参数张量 | [cnn](../cnn/cnn/) |
| receptive field | 感受野 | 译;某个输出单元在输入上可能依赖的空间区域 | [cnn](../cnn/cnn/) |
| LeNet-5 | LeNet-5 | 保留模型名;用局部卷积、可学习子采样和层级特征识别手写字符的经典卷积网络 | [lenet-to-vgg](../cnn/lenet-to-vgg/) |
| trainable subsampling | 可学习子采样 | 译;对局部平均等聚合结果再施加通道独立的可学习尺度、偏置和非线性 | [lenet-to-vgg](../cnn/lenet-to-vgg/) |
| AlexNet | AlexNet | 保留模型名;把 ReLU、GPU 卷积、重叠池化、dropout 与大规模图像训练组合起来的经典网络 | [lenet-to-vgg](../cnn/lenet-to-vgg/) |
| overlapping pooling | 重叠池化 | 译;窗口大小大于 stride、相邻池化窗口共享输入位置的下采样 | [lenet-to-vgg](../cnn/lenet-to-vgg/) |
| local response normalization / LRN | 局部响应归一化 / LRN | 保留缩写并译;在同一空间位置沿相邻通道响应做归一化的历史结构 | [lenet-to-vgg](../cnn/lenet-to-vgg/) |
| VGG | VGG | 保留模型族名;以连续小卷积核和 block 堆叠研究卷积网络深度的架构系列 | [lenet-to-vgg](../cnn/lenet-to-vgg/) |
| VGG-16 | VGG-16 | 保留模型名;13 个卷积层加 3 个全连接层、以 $3\times3$ 卷积为主的 VGG 配置 | [lenet-to-vgg](../cnn/lenet-to-vgg/) |
| small-kernel stack / 3×3 stack | 小核堆叠 / 3×3 堆叠 | 译;用多个带非线性的 $3\times3$ 卷积获得大于单层核的理论感受野 | [lenet-to-vgg](../cnn/lenet-to-vgg/) |
| weight layer | 带权重层 | 译;拥有可学习参数的卷积、全连接或其它参数化层,不把无参数激活和池化计入深度 | [lenet-to-vgg](../cnn/lenet-to-vgg/) |
| activation memory | 激活内存 | 译;训练时保存中间张量及其反向所需状态的内存,规模由 batch、序列/空间位置、特征宽度和保存策略共同决定 | [parameter-count](../transformer-components/parameter-count/) |
| residual connection / residual learning | 残差连接 / 残差学习 | 译;把目标映射改写为输入加残差修正 $y=x+F(x)$ 的参数化 | [residual-connections](../cnn/residual-connections/) |
| residual block | 残差块 | 译;由 residual branch、shortcut 和逐元素加法组成的网络模块 | [residual-connections](../cnn/residual-connections/) |
| shortcut connection | 捷径连接 / shortcut | 译;绕过一组参数化层并把输入直接或投影后送到 merge 的分支 | [residual-connections](../cnn/residual-connections/) |
| identity shortcut | 恒等捷径 | 译;在 shape 相同时直接传递输入、参数量为零的 shortcut | [residual-connections](../cnn/residual-connections/) |
| projection shortcut | 投影捷径 | 译;用通常为 $1\times1$ 卷积的可学习映射对齐 shortcut shape | [residual-connections](../cnn/residual-connections/) |
| BasicBlock | BasicBlock / 基本残差块 | 保留架构名并译;通常用两个 $3\times3$ 卷积构成 residual branch 的块 | [residual-connections](../cnn/residual-connections/) |
| bottleneck block | bottleneck block / 瓶颈残差块 | 保留常用名称并译;用 $1\times1$ 压缩、$3\times3$ 空间卷积和 $1\times1$ 恢复通道的块 | [residual-connections](../cnn/residual-connections/) |
| degradation problem | 退化问题 | 译;普通网络加深后训练误差本身上升、并非仅由泛化过拟合造成的深度优化现象 | [residual-connections](../cnn/residual-connections/) |
| residual ratio | 残差比例 | 译;用 $\lVert F(x)\rVert_2/(\lVert S(x)\rVert_2+\varepsilon)$ 监控修正分支相对 shortcut 的尺度 | [residual-connections](../cnn/residual-connections/) |
| identity mapping | 恒等映射 | 译;输入经过网络模块后保持为自身的映射,残差网络中可作为 shortcut 的基线 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| pre-activation residual unit | 预激活残差单元 | 译;把归一化和激活放到 residual branch 内、让相加后的主路径更接近恒等的 block 变体 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| wide residual network / WRN | 宽残差网络 / WRN | 保留缩写并译;减少部分串行深度、用 width factor 增加残差 block 通道的家族 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| cardinality | 基数 / cardinality | 保留常用英文并译;ResNeXt 中并行同构变换或 grouped branch 的数量轴 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| ResNeXt | ResNeXt | 保留模型名;在 residual branch 内聚合多个同拓扑变换、把 cardinality 作为容量轴的架构 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| dense connectivity / DenseNet | 密集连接 / DenseNet | 译并保留模型名;每层读取所有先前 feature maps 并通过 concat 复用历史特征的网络 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| growth rate | 增长率 | 译;Dense block 中每层新增的 feature-map 通道数 $k$,不是总通道数 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| transition layer | 过渡层 | 译;Dense block 之间用通道压缩和空间下采样控制接口预算的层 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| feature reuse | 特征复用 | 译;后续层直接读取已产生的历史特征、减少重复重建表示的连接属性 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| ConvNeXt | ConvNeXt | 保留模型名;从 ResNet 现代化得到、使用大核 depthwise convolution、LayerNorm、GELU 和残差更新的纯卷积家族 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| layer scale | layer scale / 层缩放 | 保留常用名称并译;用逐通道可学习尺度 $\gamma$ 缩小 residual update 初始幅度的设计 | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| patchify stem | patchify stem / patch 化 stem | 保留常用名称并译;用较大 kernel 和 stride 一次性把图像映射为低分辨率初始 feature map 的 stem | [resnet-and-beyond](../cnn/resnet-and-beyond/) |
| translation equivariance | 平移等变性 | 译;输入平移后输出按相同几何变换移动的性质 | [invariance-and-equivariance](../cnn/invariance-and-equivariance/) |
| translation invariance | 平移不变性 | 译;输入发生小幅平移时最终预测近似保持不变的性质 | [invariance-and-equivariance](../cnn/invariance-and-equivariance/) |
| transformation group / group action | 变换群 / 群作用 | 译;可组合的输入变换集合及其作用到输入或输出坐标的规则 | [invariance-and-equivariance](../cnn/invariance-and-equivariance/) |
| output representation | 输出表示 | 译;描述输入变换如何作用到特征图、mask、边界框或标量输出的映射 rho | [invariance-and-equivariance](../cnn/invariance-and-equivariance/) |
| equivariance error | 等变误差 | 译;输出先按目标变换对齐后与变换输入输出之间的相对差异 | [invariance-and-equivariance](../cnn/invariance-and-equivariance/) |
| invariance error | 不变误差 | 译;变换前后标量或不变输出之间的相对差异 | [invariance-and-equivariance](../cnn/invariance-and-equivariance/) |
| sampling phase | 采样相位 | 译;输入变换与 stride 网格窗口起点之间的相对位置 | [invariance-and-equivariance](../cnn/invariance-and-equivariance/) |
| effective kernel size | 有效核尺寸 | 译;考虑 dilation 后核覆盖的空间尺寸 $d(k-1)+1$ | [stride-padding-dilation](../cnn/stride-padding-dilation/) |
| padding / stride / dilation | 填充 / 步幅 / 膨胀 | 译;分别控制边界扩展、窗口移动间隔和核元素间距的卷积参数 | [stride-padding-dilation](../cnn/stride-padding-dilation/) |
| same padding | same 填充 | 保留常用模式名并解释;为目标输出尺寸计算总 padding 后再分配到各边界的约定 | [stride-padding-dilation](../cnn/stride-padding-dilation/) |
| asymmetric padding | 非对称填充 | 译;左右或上下 padding 数量不同、会改变输出空间相位的边界配置 | [stride-padding-dilation](../cnn/stride-padding-dilation/) |
| output phase | 输出相位 | 译;输出索引映射回原输入坐标时由 padding 分配和 stride 决定的空间对齐位置 | [stride-padding-dilation](../cnn/stride-padding-dilation/) |
| receptive-field jump | 感受野步距 | 译;连续卷积层中相邻特征点映射到原输入的坐标间隔,随 stride 递推 | [stride-padding-dilation](../cnn/stride-padding-dilation/) |
| output size arithmetic | 输出尺寸算术 | 译;把有效核、四侧 padding、stride 和取整规则逐层转换为空间输出 shape 的核对过程 | [output-size-arithmetic](../cnn/output-size-arithmetic/) |
| effective output length | 有效输出长度 | 译;沿一个空间轴能够容纳的完整窗口数量,由 padded 长度、有效核尺寸和 stride 决定 | [output-size-arithmetic](../cnn/output-size-arithmetic/) |
| output padding | 输出填充 | 译;转置卷积中用于选择相邻离散输出长度的 shape 参数,不是向输出末端追加真实零值 | [output-size-arithmetic](../cnn/output-size-arithmetic/) |
| transposed convolution | 转置卷积 | 译;对应卷积线性算子转置的重叠累加算子,常用于从低分辨率表示生成高分辨率空间 shape | [output-size-arithmetic](../cnn/output-size-arithmetic/) |
| ceil mode | ceil 模式 | 保留常用模式名并解释;保留尾部不完整窗口或按向上取整规则计算池化输出的约定 | [pooling](../cnn/pooling/) |
| shape ledger | 形状账 | 译;逐层记录输入输出空间尺寸、核参数、padding、取整和分支对齐约束的审计表 | [output-size-arithmetic](../cnn/output-size-arithmetic/) |
| Toeplitz matrix | Toeplitz 矩阵 | 保留人名;矩阵元素只依赖行列索引之差、卷积共享窗口形成的带状结构 | [convolution-as-toeplitz](../cnn/convolution-as-toeplitz/) |
| block Toeplitz with Toeplitz blocks / BTTB | 块 Toeplitz with Toeplitz blocks / BTTB | 保留术语并解释;二维卷积按行展开后形成的块级与块内 Toeplitz 结构 | [convolution-as-toeplitz](../cnn/convolution-as-toeplitz/) |
| convolution matrix | 卷积矩阵 | 译;把卷积窗口连接写成稀疏矩阵、便于检查边界、通道和反向传播的表示 | [convolution-as-toeplitz](../cnn/convolution-as-toeplitz/) |
| circulant matrix | 循环矩阵 | 译;周期边界下每一行是上一行循环移位的卷积矩阵 | [convolution-as-toeplitz](../cnn/convolution-as-toeplitz/) |
| im2col | im2col | 保留实现名并解释;把局部 patch 抽取成矩阵后用 GEMM 实现卷积的变换 | [convolution-as-toeplitz](../cnn/convolution-as-toeplitz/) |
| patch extraction | 局部块抽取 | 译;按卷积窗口把输入空间邻域排列为行或列的稀疏索引过程 | [convolution-as-toeplitz](../cnn/convolution-as-toeplitz/) |
| scatter-add | 散射累加 | 译;把转置矩阵或反向传播的多个输出梯度写回并累加到重叠输入位置 | [convolution-as-toeplitz](../cnn/convolution-as-toeplitz/) |
| pooling | 池化 | 译;在局部窗口内用固定聚合规则下采样或汇总特征的操作 | [pooling](../cnn/pooling/) |
| max pooling | 最大池化 | 译;在局部窗口中保留最大响应,梯度通常沿 argmax 位置传播 | [pooling](../cnn/pooling/) |
| average pooling | 平均池化 | 译;在局部窗口中求平均并按 padding 规则决定边界分母的固定聚合 | [pooling](../cnn/pooling/) |
| sum pooling | 求和池化 | 译;在局部窗口中累加响应、保留局部总量但不自动归一化的固定聚合 | [pooling](../cnn/pooling/) |
| global average pooling | 全局平均池化 | 译;沿每个 channel 的全部空间位置求平均,把特征图汇聚为通道统计量 | [pooling](../cnn/pooling/) |
| pooling indices | 池化索引 | 译;最大池化记录的窗口赢家位置,可供特定 unpooling 过程放回响应 | [pooling](../cnn/pooling/) |
| include-pad / exclude-pad | 包含填充 / 排除填充 | 译;平均池化边界分母是否把 padding 元素计入的两种数值合同 | [pooling](../cnn/pooling/) |
| anti-aliased downsampling | 抗混叠下采样 | 译;在 stride 抽取前用明确低通或平滑规则抑制高频折叠的下采样设计 | [pooling](../cnn/pooling/) |
| locality prior | 局部性先验 | 译;优先假设相邻输入位置的局部组合比远距离连接更有用 | [why-convolution](../cnn/why-convolution/) |
| stationarity | 平稳性 | 译;局部规律的统计含义大致不随空间坐标改变,支撑跨位置共享参数 | [why-convolution](../cnn/why-convolution/) |
| parameter tying | 参数绑定 | 译;把多个连接位置约束为使用同一个可学习参数 | [why-convolution](../cnn/why-convolution/) |
| sample efficiency | 样本效率 | 译;达到同等泛化质量所需的数据量与每个样本提供的有效证据量 | [why-convolution](../cnn/why-convolution/) |
| global context | 全局上下文 | 译;需要跨越较大空间范围才能判断的输入关系 | [why-convolution](../cnn/why-convolution/) |
| discrete convolution | 离散卷积 | 译;按相对索引对齐、逐项相乘并求和的离散序列或数组运算 | [discrete-convolution](../cnn/discrete-convolution/) |
| cross-correlation | 互相关 | 译;按窗口原方向逐项相乘并求和、通常不预先翻转核的滑动算子 | [cross-correlation-vs-convolution](../cnn/cross-correlation-vs-convolution/) |
| kernel reversal | 核翻转 | 译;把长度为 $M$ 的核位置 $j$ 映射为 $M-1-j$,用于对齐互相关与数学卷积 | [cross-correlation-vs-convolution](../cnn/cross-correlation-vs-convolution/) |
| symmetric / antisymmetric kernel | 对称核 / 反对称核 | 译;翻转后分别保持自身或只改变符号的核,可用于判断方向响应 | [cross-correlation-vs-convolution](../cnn/cross-correlation-vs-convolution/) |
| matched filter | 匹配滤波器 | 译;用模板与局部输入的相关响应定位相似模式的线性滤波器 | [cross-correlation-vs-convolution](../cnn/cross-correlation-vs-convolution/) |
| operator adjoint | 算子伴随 | 译;满足内积关系的转置/共轭转置算子,反向传播中的输入梯度由此前向算子决定 | [cross-correlation-vs-convolution](../cnn/cross-correlation-vs-convolution/) |
| convolution theorem | 卷积定理 | 译;把时域卷积对应为频域乘法、把带共轭的相关对应为共轭频谱乘积的关系 | [cross-correlation-vs-convolution](../cnn/cross-correlation-vs-convolution/) |
| two-dimensional convolution | 二维卷积 | 译;在高度和宽度两个空间轴上滑动局部核并对输入通道求和的卷积算子 | [convolution-2d](../cnn/convolution-2d/) |
| channel mixing | 通道混合 | 译;在同一空间位置对输入通道加权求和以产生输出通道的操作 | [convolution-2d](../cnn/convolution-2d/) |
| pointwise / 1×1 convolution | 逐点卷积 / 1×1 卷积 | 译;只在每个像素位置混合通道、不读取空间邻域的卷积层 | [convolution-2d](../cnn/convolution-2d/) |
| grouped convolution | 分组卷积 | 译;把输入和输出通道划组后只在对应组内连接的卷积结构 | [convolution-2d](../cnn/convolution-2d/) |
| depthwise convolution | 深度卷积 | 译;每个输入通道独立使用空间核、几乎不做跨通道混合的分组卷积 | [convolution-2d](../cnn/convolution-2d/) |
| depthwise separable convolution | 深度可分离卷积 | 译;先做逐通道空间卷积再用 1×1 卷积混合通道的分解结构 | [convolution-2d](../cnn/convolution-2d/) |
| channels-first / channels-last | 通道优先 / 通道末尾 | 译;分别把通道轴放在空间轴之前或之后的张量布局约定 | [convolution-2d](../cnn/convolution-2d/) |
| NCHW / NHWC | NCHW / NHWC | 保留布局缩写并解释;批次、通道、高度、宽度的两种常见轴顺序 | [convolution-2d](../cnn/convolution-2d/) |
| finite support | 有限支持 | 译;序列或函数只有有限索引位置取非零值的性质 | [discrete-convolution](../cnn/discrete-convolution/) |
| zero extension | 零延拓 | 译;把有限序列在支持集外定义为零以明确线性卷积边界 | [discrete-convolution](../cnn/discrete-convolution/) |
| full / valid / same convolution | full / valid / same 卷积 | 保留常用模式名并解释;分别保留全部重叠、完全落入和对齐长度的卷积输出 | [discrete-convolution](../cnn/discrete-convolution/) |
| discrete impulse / delta | 离散冲激 / delta | 保留符号并译;只在一个索引取 1、其余位置取 0 的卷积单位元 | [discrete-convolution](../cnn/discrete-convolution/) |
| impulse response | 冲激响应 | 译;线性系统对离散单位冲激的输出,可生成任意输入的卷积响应 | [discrete-convolution](../cnn/discrete-convolution/) |
| separable kernel | 可分离核 | 译;可写成少量一维向量外积之和、从而拆成多次一维卷积的核 | [discrete-convolution](../cnn/discrete-convolution/) |
| circular convolution | 循环卷积 | 译;索引按周期长度回绕的卷积,区别于边界外取零的线性卷积 | [discrete-convolution](../cnn/discrete-convolution/) |
| output support | 输出支撑集 | 译;一个输出头允许产生的数值集合,选择回归或分类头时必须覆盖目标的语义范围 |
| pre-activation statistics | 预激活统计 | 译;对激活前的均值、方差、分位数和尾部比例做的逐层记录 |
| fair parameter budget | 公平参数预算 | 译;比较不同结构或激活时固定参数量、宽度、FLOPs 和更新量等资源条件 | [parameter-count](../transformer-components/parameter-count/) |
| deployment parity | 部署一致性 | 译;训练、导出、量化和目标设备在算子、dtype、近似和输出误差上的可核对一致 |
| margin | 间隔 / margin | 保留常用词并解释;样本相对分类边界的带符号分数或规范化距离 | [margins-and-svm](../linear-models/margins-and-svm/) |
| geometric margin | 几何间隔 | 译;函数 margin 除以权重范数后得到的真实边界距离 | [margins-and-svm](../linear-models/margins-and-svm/) |
| support vector | 支持向量 | 译;SVM 中约束紧或对偶系数为正、决定边界的样本 | [margins-and-svm](../linear-models/margins-and-svm/) |
| support vector machine / SVM | 支持向量机 / SVM | 译并保留缩写;以最大几何间隔和 hinge loss 训练的分类模型 | [margins-and-svm](../linear-models/margins-and-svm/) |
| hard-margin SVM | 硬间隔支持向量机 | 译;要求所有样本满足规范化 margin 约束的 SVM | [margins-and-svm](../linear-models/margins-and-svm/) |
| soft-margin SVM | 软间隔支持向量机 | 译;用松弛变量或 hinge loss 允许样本违反 margin 的 SVM | [margins-and-svm](../linear-models/margins-and-svm/) |
| slack variable | 松弛变量 | 译;把不可行约束的违反程度显式写入优化问题的非负变量 | [margins-and-svm](../linear-models/margins-and-svm/) |
| KKT conditions | KKT 条件 | 保留缩写;约束优化中由驻点、可行性和互补松弛组成的最优性条件 | [margins-and-svm](../linear-models/margins-and-svm/) |
| kernel trick | 核技巧 | 译;只用核函数计算隐式特征空间内积而不显式构造映射的方法 | [kernel-trick](../linear-models/kernel-trick/) |
| kernel function | 核函数 | 译;返回隐式特征空间内积的函数 | [kernel-trick](../linear-models/kernel-trick/) |
| kernel matrix / Gram matrix | 核矩阵 / Gram 矩阵 | 译并保留人名;训练样本两两核值组成的对称矩阵 | [kernel-trick](../linear-models/kernel-trick/) |
| polynomial kernel | 多项式核 | 译;以点积加常数的幂表示多项式特征内积的核 | [kernel-trick](../linear-models/kernel-trick/) |
| RBF kernel / Gaussian kernel | RBF 核 / 高斯核 | 保留缩写并译;按欧氏距离指数衰减的局部相似核 | [kernel-trick](../linear-models/kernel-trick/) |
| representer theorem | Representer 定理 | 保留人名;正则化经验风险的最优函数落在训练样本核函数张成的空间中 | [kernel-trick](../linear-models/kernel-trick/) |
| Nyström approximation | Nyström 近似 | 保留人名;用少量 landmark 点的低秩结构近似完整核矩阵 | [kernel-trick](../linear-models/kernel-trick/) |
| random Fourier features | 随机 Fourier 特征 | 译并保留人名;用有限维随机特征近似平移不变核以便线性训练 | [kernel-trick](../linear-models/kernel-trick/) |
| classification | 分类 | 译;以离散类别或类别概率为主要输出的监督任务 | [supervised-learning](../learning-framework/supervised-learning/) |
| ranking | 排序 | 译;根据输入与标签学习候选项相对顺序的监督任务 | [supervised-learning](../learning-framework/supervised-learning/) |
| structured prediction | 结构化预测 | 译;输出为序列、集合、图或其他带内部约束对象的监督任务 | [supervised-learning](../learning-framework/supervised-learning/) |
| conditional Bayes rule | 条件 Bayes 规则 | 保留人名;对每个输入最小化条件风险的理想决策规则 | [supervised-learning](../learning-framework/supervised-learning/) |
| conditional mean / conditional median | 条件均值 / 条件中位数 | 译;平方损失 / 绝对损失下的条件风险最优预测 | [supervised-learning](../learning-framework/supervised-learning/) |
| 0–1 loss | 0–1 损失 | 译;预测类别正确取零、错误取一的分类损失 | [supervised-learning](../learning-framework/supervised-learning/) |
| calibration | 校准 | 译;预测概率与相应事件长期发生频率的一致程度 | [supervised-learning](../learning-framework/supervised-learning/) |
| label noise | 标签噪声 | 译;观测标签因标注错误或随机翻转而偏离目标的现象 | [supervised-learning](../learning-framework/supervised-learning/) |
| class imbalance | 类别不平衡 | 译;不同类别样本数量或先验概率相差悬殊的情况 | [supervised-learning](../learning-framework/supervised-learning/) |
| cost-sensitive learning | 代价敏感学习 | 译;按不同错误的实际代价加权训练或决策的监督方法 | [supervised-learning](../learning-framework/supervised-learning/) |
| data leakage / label leakage | 数据泄漏 / 标签泄漏 | 译;训练或评估时把标签、未来信息或不可用特征带入输入 | [supervised-learning](../learning-framework/supervised-learning/) |
| train / validation / test set | 训练集 / 验证集 / 测试集 | 译;分别用于拟合参数、选择方案和冻结后最终评估的互不重叠数据分片 | [train-validation-test](../learning-framework/train-validation-test/) |
| holdout set | 留出集 | 译;从可用数据中保留、在选择阶段不参与拟合的分片 | [train-validation-test](../learning-framework/train-validation-test/) |
| cross-validation | 交叉验证 | 译;轮流把各折作为验证并在其余折拟合,用来估计选择表现的方法 | [cross-validation](../evaluation-and-generalization/cross-validation/) |
| K-fold cross-validation | K 折交叉验证 | 译;把开发数据划成 K 个互不重叠折并轮流留出一折验证的协议 | [cross-validation](../evaluation-and-generalization/cross-validation/) |
| nested cross-validation | 嵌套交叉验证 | 译;内层选择超参数、外层估计整个选择流程表现的交叉验证协议 | [cross-validation](../evaluation-and-generalization/cross-validation/) |
| repeated cross-validation | 重复交叉验证 | 译;用多个随机折划分重复 K 折流程以降低单次划分偶然性的协议 | [cross-validation](../evaluation-and-generalization/cross-validation/) |
| stratified cross-validation | 分层交叉验证 | 译;让各折尽量保持类别比例但不自动消除实体或时间相关性的折法 | [cross-validation](../evaluation-and-generalization/cross-validation/) |
| group cross-validation / group split | 分组交叉验证 / 分组切分 | 译;按用户、设备或文档等实体组分配样本,避免同组跨分片 | [cross-validation](../evaluation-and-generalization/cross-validation/) |
| time-series cross-validation / walk-forward validation | 时间序列交叉验证 / 向前验证 | 译;只用过去训练并在未来窗口验证的时间顺序协议 | [cross-validation](../evaluation-and-generalization/cross-validation/) |
| leave-one-out cross-validation / LOOCV | 留一交叉验证 / LOOCV | 译并保留缩写;每次只留一个样本验证的 K=n 极端交叉验证 | [cross-validation](../evaluation-and-generalization/cross-validation/) |
| out-of-fold prediction / OOF prediction | 折外预测 / OOF 预测 | 译并保留缩写;由没有见过该样本的对应折模型生成的预测 | [cross-validation](../evaluation-and-generalization/cross-validation/) |
| model-selection bias | 模型选择偏差 | 译;在许多候选结果中挑选最小验证分数导致的乐观偏差 | [model-selection](../evaluation-and-generalization/model-selection/) |
| validation overfitting | 验证集过拟合 | 译;反复用验证结果选择后对验证噪声的适应,使泛化表现被高估 | [cross-validation](../evaluation-and-generalization/cross-validation/) |
| hyperparameter tuning / hyperparameter optimization | 超参数搜索 / 超参数优化 | 译;在训练参数之外用开发集证据选择结构、正则化、优化和预算方案的过程 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| search space | 搜索空间 | 译;规定候选超参数的变量类型、边界、变换和条件规则 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| grid search | 网格搜索 | 译;对每个维度预先列出的笛卡尔积候选逐一评估的搜索方法 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| random search | 随机搜索 | 译;从预先定义的候选分布抽取超参数配置并评估的搜索方法 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| log-uniform distribution | 对数均匀分布 | 译;变量的对数而不是变量本身服从均匀分布的采样尺度 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| Bayesian optimization | 贝叶斯优化 | 译;用历史试验拟合代理模型并由采集函数决定下一次试验的搜索方法 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| surrogate model | 代理模型 | 译;近似昂贵验证目标并预测未试配置表现的模型 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| acquisition function | 采集函数 | 译;在利用当前好区域与探索不确定区域之间选择下一候选的函数 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| expected improvement / EI | 期望改进 / EI | 译并保留缩写;对当前最好目标值的预期改善量 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| multi-fidelity optimization | 多保真优化 | 译;用不同训练资源预算逐级筛选候选的搜索方法 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| successive halving | successive halving / 逐级淘汰 | 保留常用名称并译;给许多候选少量资源后只给少数候选更多资源的策略 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| Pareto frontier / non-dominated configuration | Pareto 前沿 / 非支配配置 | 保留人名并译;不存在另一配置在所有目标不差且至少一项更好的多目标点集 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| trial budget / search budget | 试验预算 / 搜索预算 | 译;限制候选次数、折数、随机种子、训练资源或总成本的约束 | [hyperparameter-tuning](../evaluation-and-generalization/hyperparameter-tuning/) |
| evaluation protocol | 评估协议 | 译;规定切分、预处理、选择、测试和不确定性报告方式的完整规则 | [train-validation-test](../learning-framework/train-validation-test/) |
| preprocessing leakage | 预处理泄漏 | 译;在切分前用验证或测试信息拟合标准化等预处理参数 | [train-validation-test](../learning-framework/train-validation-test/) |
| overfitting / underfitting | 过拟合 / 欠拟合 | 译;分别是适应样本偶然细节或表达能力不足导致的训练外风险问题 | [overfitting-and-regularization](../learning-framework/overfitting-and-regularization/) |
| regularization | 正则化 | 译;在拟合数据之外加入复杂度、参数规模、训练时间或不变性偏好的方法 | [overfitting-and-regularization](../learning-framework/overfitting-and-regularization/) |
| explicit regularization | 显式正则化 | 译;把复杂度惩罚或约束直接写进训练目标的正则化 | [overfitting-and-regularization](../learning-framework/overfitting-and-regularization/) |
| implicit regularization | 隐式正则化 | 译;由初始化、优化器、参数化或训练过程产生而未直接写进目标的解偏好 | [implicit-regularization](../evaluation-and-generalization/implicit-regularization/) |
| implicit bias / optimization-induced bias | 隐式偏置 / 优化诱导偏好 | 译;算法在同样训练目标的多个解中因路径和参数化而偏向某些解的现象 | [implicit-regularization](../evaluation-and-generalization/implicit-regularization/) |
| minimum-norm interpolator | 最小范数插值器 | 译;在所有零训练误差解中选择参数范数最小者的规则,线性梯度下降需附带初始化与秩条件 | [implicit-regularization](../evaluation-and-generalization/implicit-regularization/) |
| max-margin direction | 最大间隔方向 | 译;可分分类器中使规范化几何间隔最大的归一化参数方向 | [implicit-regularization](../evaluation-and-generalization/implicit-regularization/) |
| weight decay | 权重衰减 | 译;通过更新规则持续收缩参数的正则化机制 | [overfitting-and-regularization](../learning-framework/overfitting-and-regularization/) |
| early stopping | 早停 | 译;在验证风险或其他停止规则指示的时机结束训练,把训练时间作为选择变量 | [early-stopping](../evaluation-and-generalization/early-stopping/) |
| early stopping criterion / stopping rule | 早停准则 / 停止规则 | 译;规定监控指标、方向、评估频率和预算后决定何时结束训练的规则 | [early-stopping](../evaluation-and-generalization/early-stopping/) |
| checkpoint / best checkpoint | 检查点 / 最佳检查点 | 译;保存某个训练时刻模型及可选优化状态的快照,最佳检查点由验证指标选出 | [early-stopping](../evaluation-and-generalization/early-stopping/) |
| min_delta / minimum improvement | min_delta / 最小改善量 | 保留参数名并译;验证指标必须超过的最小改善幅度,单位随监控指标变化 | [early-stopping](../evaluation-and-generalization/early-stopping/) |
| restore best weights | 恢复最佳权重 | 译;停止触发后加载验证指标历史最佳时刻的模型权重而非最后权重 | [early-stopping](../evaluation-and-generalization/early-stopping/) |
| update budget | 更新预算 | 译;用参数更新次数或已处理样本数规定训练量,避免 epoch 随数据规模改变而失去可比性 | [early-stopping](../evaluation-and-generalization/early-stopping/) |
| data augmentation | 数据增强 | 译;对训练输入施加保持任务语义的随机变换,把合理变化写进训练目标 | [data-augmentation](../evaluation-and-generalization/data-augmentation/) |
| label-preserving transformation | 标签保持变换 | 译;输入经过变换后标签仍成立的变换,需按任务定义而不是视觉相似度判断 | [data-augmentation](../evaluation-and-generalization/data-augmentation/) |
| augmentation distribution | 增强分布 | 译;规定变换种类、参数、概率与组合顺序的条件分布 q(T∣x) | [data-augmentation](../evaluation-and-generalization/data-augmentation/) |
| consistency regularization | 一致性正则化 | 译;要求同一原样本的合理变换视图产生相近预测的约束 | [data-augmentation](../evaluation-and-generalization/data-augmentation/) |
| test-time augmentation / TTA | 测试时增强 / TTA | 译并保留缩写;推理时对多个合理视图预测并聚合,需单独报告额外成本 | [data-augmentation](../evaluation-and-generalization/data-augmentation/) |
| invariant / equivariant | 不变 / 等变 | 译;不变要求输出不变,等变要求输出按 ρT 一起变换 | [invariance-and-equivariance](../cnn/invariance-and-equivariance/) |
| effective sample size / design effect | 等效样本数 / 设计效应 | 译;用同源视图相关性近似统计信息量,不等于实际数据行数 | [data-augmentation](../evaluation-and-generalization/data-augmentation/) |
| interpolating solution / interpolation regime | 插值解 / 插值区间 | 译;在训练样本上达到零损失或完全拟合的解及其模型区域 | [overfitting-and-regularization](../learning-framework/overfitting-and-regularization/) |
| double descent | 双下降 | 译;测试风险在插值阈值附近上升并在过参数化区域可能再次下降的非单调曲线 | [double-descent](../learning-framework/double-descent/) |
| interpolation threshold | 插值阈值 | 译;模型有效自由度与样本约束相当、训练风险开始可达零的区域 | [double-descent](../learning-framework/double-descent/) |
| overparameterization / overparameterized model | 过参数化 / 过参数化模型 | 译;参数或有效自由度超过训练样本约束的模型区域或模型 | [double-descent](../learning-framework/double-descent/) |
| effective degrees of freedom | 有效自由度 | 译;在给定数据、正则化和算法下实际参与拟合的自由度,不必等于参数数量 | [double-descent](../learning-framework/double-descent/) |
| bias–variance tradeoff | 偏差—方差权衡 | 译;在系统性误差与有限样本估计波动之间调整复杂度的关系 | [bias-variance-tradeoff](../learning-framework/bias-variance-tradeoff/) |
| statistical bias | 统计偏差 | 译;估计器或平均预测与目标量之间的系统性偏离 | [bias-variance-tradeoff](../learning-framework/bias-variance-tradeoff/) |
| squared bias | 偏差平方 | 译;平均预测偏离目标的平方,是平方风险分解的一项 | [bias-variance-tradeoff](../learning-framework/bias-variance-tradeoff/) |
| estimator variance | 估计器方差 | 译;换训练样本或随机重复时估计器输出围绕平均值的波动 | [bias-variance-tradeoff](../learning-framework/bias-variance-tradeoff/) |
| bias-variance decomposition | 偏差—方差分解 | 译;平方预测风险拆成偏差平方、估计方差和不可约噪声的恒等式 | [bias-variance-tradeoff](../learning-framework/bias-variance-tradeoff/) |
| shrinkage estimator | 收缩估计器 | 译;把无偏估计向指定中心收缩,以偏差换取更低方差的估计器 | [bias-variance-tradeoff](../learning-framework/bias-variance-tradeoff/) |
| ensemble prediction | 集成预测 | 译;平均或组合多个模型输出以降低部分不相关预测波动的方法 | [bias-variance-tradeoff](../learning-framework/bias-variance-tradeoff/) |
| model capacity | 模型容量 | 译;假设空间表达不同函数或标注模式的能力规模 | [model-capacity](../learning-framework/model-capacity/) |
| learning curve | 学习曲线 | 译;改变训练样本数时记录训练风险和验证风险以判断数据效应的曲线 | [overfitting-and-underfitting](../learning-framework/overfitting-and-underfitting/) |
| unsupervised learning | 无监督学习 | 译;只用输入样本学习结构、表示或生成分布 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| clustering | 聚类 | 译;按输入之间的相似性把样本划分为若干组的无监督任务 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| k-means | k-means | 保留常用写法;以平方欧氏距离和组内均值为目标的聚类算法 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| within-cluster distortion | 簇内畸变 | 译;样本到所属聚类中心的平均平方距离 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| Gaussian mixture model / GMM | 高斯混合模型 / GMM | 译并保留缩写;用多个带权高斯成分表示混合密度的概率模型 | [gmm-and-em](../linear-models/gmm-and-em/) |
| mixture component | 混合成分 | 译;GMM 中由权重、均值和协方差共同定义的一个高斯子分布 | [gmm-and-em](../linear-models/gmm-and-em/) |
| responsibility | 责任度 | 译;给定当前参数后样本属于某个混合成分的后验概率 | [gmm-and-em](../linear-models/gmm-and-em/) |
| expectation-maximization / EM | 期望最大化 / EM | 译并保留缩写;交替估计隐变量后验并最大化加权完整数据目标的算法 | [gmm-and-em](../linear-models/gmm-and-em/) |
| E-step / M-step | E 步 / M 步 | 保留字母并译;EM 中计算责任度的步骤和按责任度更新参数的步骤 | [gmm-and-em](../linear-models/gmm-and-em/) |
| covariance structure | 协方差结构 | 译;对 GMM 协方差施加 spherical、diagonal、tied 或 full 等参数约束 | [gmm-and-em](../linear-models/gmm-and-em/) |
| soft clustering | 软聚类 | 译;用属于各簇的概率或责任度而不是单一硬标签表示归属 | [gmm-and-em](../linear-models/gmm-and-em/) |
| Bayesian information criterion / BIC | 贝叶斯信息准则 / BIC | 译并保留缩写;用对数似然和随样本量增长的复杂度惩罚比较模型 | [gmm-and-em](../linear-models/gmm-and-em/) |
| singular covariance | 奇异协方差 | 译;协方差退化导致密度尖峰和似然无有限上界的危险情形 | [gmm-and-em](../linear-models/gmm-and-em/) |
| principal component analysis / PCA | 主成分分析 / PCA | 译并保留缩写;按投影方差从大到小选择正交方向的降维方法 | [pca](../linear-models/pca/) |
| explained variance ratio | 解释方差比例 | 译;某主成分特征值占全部方差的比例 | [pca](../linear-models/pca/) |
| PCA score | PCA 分数 | 保留常用形式;样本中心化后在主成分方向上的投影坐标 | [pca](../linear-models/pca/) |
| PCA loading | PCA 载荷 | 保留常用形式;描述原始变量与主成分关联的方向或缩放向量,需说明约定 | [pca](../linear-models/pca/) |
| scree plot | 碎石图 | 译;按主成分序号绘制特征值以观察谱的下降形状 | [pca](../linear-models/pca/) |
| reconstruction error | 重构误差 | 译;低维投影再映回原空间后与原样本之间的平方差或其平均 | [pca](../linear-models/pca/) |
| kernel PCA | 核 PCA / 核主成分分析 | 保留缩写并译;先用核矩阵定义隐式特征空间中的主成分 | [pca](../linear-models/pca/) |
| dimensionality reduction | 降维 | 译;把输入映射到较低维坐标同时保留选定结构的方法 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| density estimation | 密度估计 | 译;从样本估计输入分布或其概率密度的任务 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| log-likelihood / negative log-likelihood | 对数似然 / 负对数似然 | 译;样本在概率模型下的平均对数概率及其相反数 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| generative model | 生成模型 | 译;学习数据分布并可用于生成或模拟新样本的模型 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| autoencoder | 自编码器 | 译;用编码器压缩输入、再用解码器重构输入的模型 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| reconstruction loss | 重构损失 | 译;衡量模型输出与原输入之间差异的损失 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| representation learning | 表示学习 | 译;从数据中学习供压缩、检索或下游任务使用的中间表示 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| latent representation / latent variable | 潜在表示 / 潜变量 | 译;模型内部用于表达观测数据的不可直接观测坐标或变量 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| self-supervised learning | 自监督学习 | 译;从输入自身构造伪标签、视图或上下文关系来训练模型 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| pretext task | 代理任务 | 译;用于预训练表示而非最终部署目标的构造任务 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| pseudo-label | 伪标签 | 译;由输入、变换或模型规则自动生成而非人工直接提供的标签 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| contrastive learning | 对比学习 | 译;通过拉近正样本对并区分负样本对来学习表示的方法 | [unsupervised-learning](../learning-framework/unsupervised-learning/) |
| reinforcement learning | 强化学习 | 译;通过环境交互和延迟奖励学习策略或价值函数 | [reinforcement-learning-overview](../learning-framework/reinforcement-learning-overview/) |

## 保留不译

| 英文 | 原因 | 首现 |
| --- | --- | --- |
| MNIST | 数据集名;Modified National Institute of Standards and Technology 的手写数字图像数据集 | [mnist-mlp-training-loop](../training-nn/mnist-mlp-training-loop/) |
| Transformer | 架构名 | [full-transformer](../transformer-architectures/full-transformer/) |
| full Transformer | 完整 Transformer | 同时包含 encoder stack 和 decoder stack 的 encoder-decoder 计算链,由源序列、目标前缀、三种 attention、输出 logits 和训练/推理协议共同定义 | [full-transformer](../transformer-architectures/full-transformer/) |
| source memory | 源 memory / 源序列记忆 | 译并保留常用英文;encoder 最终输出的 $(B,S,D)$ 表示,按源位置保留并作为 decoder cross-attention 的 key/value 来源 | [encoder-decoder](../transformer-architectures/encoder-decoder/) |
| target prefix | 目标前缀 | 译;decoder 在当前位置预测前已经提供的 BOS、真实 target 前缀或已生成 token 序列 | [encoder-decoder](../transformer-architectures/encoder-decoder/) |
| source length / target length | 源长度 / 目标长度 | 译;分别记为 $S$ 和 $U$,决定 encoder self、decoder self 与 cross-attention 的两个位置轴 | [encoder-decoder](../transformer-architectures/encoder-decoder/) |
| encoder-decoder block | 编解码器 block | 译;由 encoder self-attention 或 decoder masked self-attention、cross-attention、FFN 及相应残差/归一化支路组成的层级模块 | [encoder-decoder](../transformer-architectures/encoder-decoder/) |
| encoder stack | encoder 堆栈 / 编码器堆栈 | 由多层 encoder block 串行组成、把源 token 表示变成 encoder memory 的模块 | [full-transformer](../transformer-architectures/full-transformer/) |
| decoder stack | decoder 堆栈 / 解码器堆栈 | 由 masked self-attention、cross-attention 和 FFN block 串行组成、按目标前缀产生 hidden 的模块 | [full-transformer](../transformer-architectures/full-transformer/) |
| encoder memory | encoder memory / 编码器记忆 | encoder stack 最终输出的 $(B,S,D)$ 表示,作为 decoder cross-attention 的 key/value 来源 | [full-transformer](../transformer-architectures/full-transformer/) |
| target shift / shifted target | 目标移位 / 右移目标 | 训练 decoder 时用 BOS 和真实前缀构造输入,使位置 $u$ 输入 $y_{u-1}$ 并预测 $y_u$ 的对齐合同 | [full-transformer](../transformer-architectures/full-transformer/) |
| attention score shape | attention score 形状 | 区分 encoder self 的 $(B,h,S,S)$、decoder self 的 $(B,h,U,U)$ 与 cross 的 $(B,h,U,S)$ 的张量合同 | [full-transformer](../transformer-architectures/full-transformer/) |
| encoder-only | Encoder-Only / 仅编码器架构 | 保留常用英文并译;只保留 encoder stack、用双向 self-attention 把完整输入序列变成 $(B,T,D)$ contextual hidden 的 Transformer 架构 | [encoder-only](../transformer-architectures/encoder-only/) |
| bidirectional encoder | 双向编码器 | 译;每个有效 query 可以读取同一序列中前后两侧的有效 key,不使用 decoder 的 causal mask | [encoder-only](../transformer-architectures/encoder-only/) |
| contextual hidden / contextual representation | 上下文 hidden / 上下文表示 | 译并保留英文;经过 encoder self-attention 后同时依赖当前 token、位置、其他有效 token 和 mask 的运行时表示 | [encoder-only](../transformer-architectures/encoder-only/) |
| CLS pooling / mean pooling | CLS 池化 / 均值池化 | 译;分别读取约定 CLS 位置或按有效 token 数归约序列 hidden,生成 $(B,D)$ 的序列级表示 | [encoder-only](../transformer-architectures/encoder-only/) |
| token classification / sequence classification | token 分类 / 序列分类 | 译;分别在每个 token hidden 或 pooled sequence hidden 上接任务头并计算监督 | [encoder-only](../transformer-architectures/encoder-only/) |
| span prediction | 区间预测 | 译;对每个 token 产生 start/end 分数,从上下文序列中选择一个连续答案区间的任务头 | [encoder-only](../transformer-architectures/encoder-only/) |
| decoder-only | Decoder-Only / 仅解码器架构 | 保留常用英文并译;只保留带 causal mask 的 decoder stack、没有 encoder memory 和 cross-attention、按前缀产生 next-token logits 的 Transformer 架构 | [decoder-only](../transformer-architectures/decoder-only/) |
| causal decoder | 因果解码器 | 译;每个 query 只读取当前及过去 key、通过下三角 self-attention 维持自回归依赖的 decoder stack | [decoder-only](../transformer-architectures/decoder-only/) |
| next-token logits | next-token logits / 下一个 token 分数 | 保留常用术语并译;decoder-only 在当前有效前缀末位置产生的 $(B,V)$ 词表未归一化分数 | [decoder-only](../transformer-architectures/decoder-only/) |
| decoder KV cache | decoder KV cache / 解码器 K/V 缓存 | 译并保留常用英文;自回归 decode 按层保存历史 K/V、让新 query 读取全部前缀而不重复计算历史 projection 的运行时状态 | [decoder-only](../transformer-architectures/decoder-only/) |
| LLaMA 2 | LLaMA 2 | 保留模型名;以 LLaMA 风格 decoder-only、RMSNorm、RoPE、SwiGLU 和可选 GQA 组成的语言模型系列,具体配置必须读取 checkpoint | [llama2-from-scratch](../alignment/llama2-from-scratch/) |
| LLaMA-style decoder | LLaMA 风格 decoder | 译并保留常用英文;以 pre-norm、RMSNorm、causal attention、RoPE、SwiGLU 和 residual stream 组成的 decoder-only block 组合 | [llama2-from-scratch](../alignment/llama2-from-scratch/) |
| model configuration contract | model configuration contract / 模型配置合同 | 译并保留常用英文;固定 $V$、$D$、$L$、$h_q$、$h_{kv}$、$d_h$、$m$、$epsilon$、RoPE base 和上下文上限的字段集合 | [llama2-from-scratch](../alignment/llama2-from-scratch/) |
| forward contract | forward contract / 前向合同 | 译并保留常用英文;规定 token id、权重 shape、位置、mask、残差和 logits 在一次模型前向中的接口与顺序 | [llama2-from-scratch](../alignment/llama2-from-scratch/) |
| checkpoint compatibility | checkpoint compatibility / checkpoint 兼容性 | 译并保留常用英文;tokenizer、config、权重 key、shape、dtype、转置和输出头绑定都与目标 checkpoint 一致的条件 | [llama2-from-scratch](../alignment/llama2-from-scratch/) |
| cache layout contract | cache layout contract / cache 布局合同 | 译并保留常用英文;规定每层 K/V 的 head 数、序列轴、dtype、RoPE 状态和追加位置,确保 prefill 与 decode 可复用 | [llama2-from-scratch](../alignment/llama2-from-scratch/) |
| tied / untied output head | 绑定/独立输出头 | 译并保留常用英文;分别表示 output logits 复用 embedding 转置或保存独立的 $V\times D$ 输出矩阵 | [llama2-from-scratch](../alignment/llama2-from-scratch/) |
| masked language modeling / MLM | 掩码语言建模 / MLM | 译并保留缩写;选择并腐蚀输入 token,只在 selected positions 预测原 token 的去噪式预训练目标 | [masked-language-modeling](../transformer-architectures/masked-language-modeling/) |
| corruption selection mask | 腐蚀选择 mask | 译;标记哪些原始 token 位置进入 [MASK]、随机替换或保持原 token 的输入变换 | [masked-language-modeling](../transformer-architectures/masked-language-modeling/) |
| selected loss mask | selected loss mask / 监督选择 mask | 译并保留常用英文;选择哪些位置进入 MLM loss,与 padding attention mask 和 causal attention mask 分开 | [masked-language-modeling](../transformer-architectures/masked-language-modeling/) |
| whole-word masking | 整词 masking | 译并保留常用英文;以自然语言词而非独立 subword token 为选择单位,把同一词的全部 token 加入 selected 集合 | [masked-language-modeling](../transformer-architectures/masked-language-modeling/) |
| span corruption | span 腐蚀 | 译并保留常用英文;选择连续 token 片段形成腐蚀输入的去噪策略,可保持长度或配合 sentinel 改变长度 | [masked-language-modeling](../transformer-architectures/masked-language-modeling/) |
| MLM head | MLM head / MLM 输出头 | 保留常用英文并译;把 selected contextual hidden 映射为目标词表 logits 的输出模块,可与输入 embedding 绑定 | [masked-language-modeling](../transformer-architectures/masked-language-modeling/) |
| causal language modeling / CLM | 因果语言建模 / CLM | 译并保留缩写;把序列联合概率按前缀分解,每个有效位置用 $x_{\leq i}$ 预测 next token $x_{i+1}$ 的自回归目标 | [causal-language-modeling](../transformer-architectures/causal-language-modeling/) |
| next-token prediction | next-token prediction / 下一个 token 预测 | 保留常用英文并译;用右移输入的每个位置产生下一 token 的词表 logits 和交叉熵监督 | [causal-language-modeling](../transformer-architectures/causal-language-modeling/) |
| causal factorization | 因果分解 | 译;把 $p(x_{1:T}\mid x_0)$ 写成 $\prod_i p(x_{i+1}\mid x_{0:i})$ 的联合概率分解 | [causal-language-modeling](../transformer-architectures/causal-language-modeling/) |
| shifted input / shifted label | 右移输入 / 移位标签 | 译;令 decoder 输入 $X_i=x_i$、监督标签 $Y_i=x_{i+1}$,使当前位置 logits 对齐下一个 token | [causal-language-modeling](../transformer-architectures/causal-language-modeling/) |
| mixture of experts / MoE | 混合专家 / MoE | 译并保留缩写;用 router 为每个 token 选择少量 expert,在增加总参数容量的同时只执行 top-k 条前馈路径 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| expert | expert / 专家 | 保留常用英文并译;一个独立的 token-wise FFN 或 SwiGLU 前馈函数,由 MoE router 按 token 选择 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| router | router / 路由器 | 保留常用英文并译;把 token hidden 映射为 expert logits 和选择概率的模块,不等同于最终 dispatch 结果 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| top-k gating | top-k 门控 | 译并保留常用英文;按 router probability 选择 k 个 expert,再使用原始或重新归一化的 gate 加权 expert 输出 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| normalized gate | 归一化 gate / 归一化门控权重 | 译并保留常用英文;只在 selected expert 集合内重新归一化 router probability,使被选 gate 之和为 1 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| capacity factor | capacity factor / 容量因子 | 保留常用英文并译;把每个 expert 的期望 assignment 数放大后取整以分配有限 slot 的系数 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| token overflow / token drop | token overflow / token 丢弃 | 译并保留常用英文;expert assignment 超过 capacity 后被裁剪、回退或丢弃的路由结果 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| dispatch buffer | dispatch buffer / 分发缓冲区 | 译并保留常用英文;按 expert 和 capacity slot 重排 token hidden、gate 与索引以供 expert 批量计算的缓冲区 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| load-balancing auxiliary loss | 负载均衡辅助损失 | 译;同时约束 router 平均概率与实际 expert assignment 比例,降低 expert collapse 和 capacity overflow 风险 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| router z-loss | router z-loss / 路由器 z 损失 | 保留常用英文并译;对 router logits 的 log-sum-exp 平方施加正则以控制尺度,不等同于 expert 负载均衡损失 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| expert parallelism | expert parallelism / expert 并行 | 译并保留常用英文;把不同 expert 放在不同设备上,通过 all-to-all 发送 token assignment 并返回 expert 输出的并行方式 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| expert collapse | expert collapse / 专家塌缩 | 保留常用英文并译;大量 token 长期集中到少数 expert,造成其他 expert 更新不足和容量溢出 | [mixture-of-experts](../transformer-architectures/mixture-of-experts/) |
| Vision Transformer / ViT | Vision Transformer / ViT / 视觉 Transformer | 保留模型名并译;把图像切成 patch token,加入位置嵌入后送入 encoder-only Transformer 的视觉架构 | [vit](../transformer-architectures/vit/) |
| image patch / patch token | 图像 patch / patch token | 译并保留常用英文;图像中固定 $P\times P$ 空间窗口展平并投影到模型宽度 $D$ 后形成的 token | [vit](../transformer-architectures/vit/) |
| patchify | patchify / patch 化 | 保留常用英文并译;按固定网格从图像抽取非重叠 patch 并排列成序列的输入步骤 | [vit](../transformer-architectures/vit/) |
| patch projection / linear patch embedding | patch projection / 线性 patch 嵌入 | 译并保留常用英文;把 $CP^2$ 维像素向量映射到 $D$ 维 patch token 的共享线性层 | [vit](../transformer-architectures/vit/) |
| patch size | patch size / patch 大小 | 保留常用英文并译;非重叠图像窗口的边长 $P$,决定 patch 网格和 token 数 | [vit](../transformer-architectures/vit/) |
| patch grid | patch 网格 | 译;图像高度宽度除以 patch size 后得到的 $n_h\times n_w$ 空间 token 网格 | [vit](../transformer-architectures/vit/) |
| class token / CLS token | class token / CLS token / 分类 token | 保留常用英文并译;拼在 patch token 前、通过 self-attention 汇总图像信息并供分类头读取的可学习向量 | [vit](../transformer-architectures/vit/) |
| patch token sequence | patch token 序列 | 译;按固定空间展平顺序排列的 $N$ 个 patch embedding,可与 CLS 和位置嵌入组成 encoder 输入 | [vit](../transformer-architectures/vit/) |
| 2D positional embedding | 二维位置嵌入 | 译;把 patch 的二维行列坐标映射为位置向量,用于在 token 序列中保留网格关系 | [vit](../transformer-architectures/vit/) |
| position interpolation | 位置插值 | 译;把 learned position embedding 从原 patch 网格插值到新分辨率,并单独处理 CLS 位置的适配步骤 | [vit](../transformer-architectures/vit/) |
| patch-level prediction | patch-level prediction / patch 级预测 | 保留常用英文并译;保留 patch token 网格并为每个 patch 产生类别、分割或其他局部输出的任务合同 | [vit](../transformer-architectures/vit/) |
| pretraining / pre-train | 预训练 / pretraining | 译并保留常用英文;在大规模语料上先训练基础模型,再用于继续预训练、监督微调或对齐的训练阶段 | [pretraining](../pretraining/pretraining/) |
| pretraining corpus | 预训练语料 | 译;经清洗、去重、过滤和来源标注后供基础模型训练的文档或 token 集合 | [pretraining](../pretraining/pretraining/) |
| effective target tokens | 有效 target token | 译;经过 padding、loss mask 和边界规则筛选后真正进入目标损失分母的 token 数 | [pretraining](../pretraining/pretraining/) |
| token budget / training token budget | token 预算 / 训练 token 预算 | 译;一次训练计划允许 optimizer 实际消费的有效 target token 总数,区别于文档数和字符数 | [pretraining](../pretraining/pretraining/) |
| global batch / global batch size | global batch / 全局批量 | 译并保留常用英文;跨设备、梯度累积后一次 optimizer update 汇总的样本或有效 token 批量 | [pretraining](../pretraining/pretraining/) |
| data mixture / mixture weight | 数据混合 / 混合权重 | 译;多个数据源按采样概率共同组成训练输入的配置及权重,需与实际 token 份额分开记录 | [pretraining](../pretraining/pretraining/) |
| sampling probability | 采样概率 | 译;数据源在文档、token、shard 或 batch 采样单位上的抽取概率,不自动等于原始语料占比 | [pretraining](../pretraining/pretraining/) |
| document boundary | 文档边界 | 译;决定相邻文档能否通过 EOS 串接、独立窗口或 block-diagonal mask 互相作为条件的输入协议 | [pretraining](../pretraining/pretraining/) |
| token packing | token packing / token 打包 | 译并保留常用英文;把多个 token 或文档按上下文长度排列成 batch 序列并配合 boundary、attention 和 loss mask 的过程 | [pretraining](../pretraining/pretraining/) |
| optimizer state checkpoint | optimizer state checkpoint / 优化器状态 checkpoint | 译并保留常用英文;和模型参数一起保存 moments、scheduler、RNG、数据游标和 token 计数以支持继续训练 | [pretraining](../pretraining/pretraining/) |
| data contamination | 数据污染 | 译;评估样本或近似内容进入训练语料,使结果可能反映记忆而非泛化的情况 | [pretraining](../pretraining/pretraining/) |
| next-token maximum likelihood / next-token MLE | 下一词最大似然 / next-token MLE | 译并保留缩写;在固定 tokenizer、边界、mask 和权重后,把真实 token 序列的联合概率按前缀分解并最小化有效 target token 的 NLL | [next-token-as-mle](../pretraining/next-token-as-mle/) |
| token-level mean | token-level mean / token 级平均 | 保留常用英文并译;对所有有效 next-token 事件等权平均 NLL,与固定数据集的总 NLL 具有相同最优参数 | [next-token-as-mle](../pretraining/next-token-as-mle/) |
| sequence-level mean | sequence-level mean / 序列级平均 | 保留常用英文并译;先在每条序列内平均 NLL,再让每条序列等权,通常不同于 token-level mean | [next-token-as-mle](../pretraining/next-token-as-mle/) |
| empirical conditional distribution | 经验条件分布 | 译;同一真实前缀下各 next-token 标签的观测频率,是 token-level NLL 所拟合的有限样本分布 | [next-token-as-mle](../pretraining/next-token-as-mle/) |
| effective likelihood event | 有效似然事件 | 译;由真实前缀、next-token 标签和 loss mask 确定、实际进入 NLL 分母的一条条件观测 | [next-token-as-mle](../pretraining/next-token-as-mle/) |
| token-level NLL | token 级 NLL | 译并保留缩写;对有效 next-token 条件事件的负对数概率,按有效 token 数归约后用于 PPL | [next-token-as-mle](../pretraining/next-token-as-mle/) |
| training data / training dataset | 训练数据 / 训练数据集 | 译;从原始记录经过来源登记、解析、清洗、去重、切分、tokenize 和 shard 生成后供模型消费的版本化数据集合 | [training-data](../pretraining/training-data/) |
| data provenance / data lineage | 数据 provenance / 数据血缘 | 保留常用英文并译;记录原始来源、派生文档、训练 segment、token shard 与处理版本之间的可追踪关系 | [training-data](../pretraining/training-data/) |
| canonical text / canonical record | 规范化文本 / canonical record | 译并保留常用英文;经过约定的 Unicode、换行和模板处理后用于 hash、去重和派生数据的文本记录 | [training-data](../pretraining/training-data/) |
| content hash | 内容哈希 | 译;对 raw bytes 或 canonical text 计算的内容指纹,必须标明输入层级和哈希版本 | [training-data](../pretraining/training-data/) |
| exact deduplication | 精确去重 | 译;按约定规范化后的相同内容哈希合并重复记录,仍保留来源成员映射 | [training-data](../pretraining/training-data/) |
| near-duplicate detection | 近似重复检测 | 译;用 shingle、Jaccard 或候选索引发现表面高度重叠但不完全相同的记录 | [training-data](../pretraining/training-data/) |
| shingle / k-shingle | shingle / k-shingle | 保留常用英文;文档中连续 $k$ 个 token、词或字符组成的局部片段集合,用于近似重叠比较 | [training-data](../pretraining/training-data/) |
| Jaccard similarity | Jaccard 相似度 | 保留人名并译;集合交集大小除以并集大小,用于衡量 shingle 集合的重叠比例 | [training-data](../pretraining/training-data/) |
| split manifest | split manifest / 切分清单 | 保留常用英文并译;冻结记录 ID、group ID、split、规则版本和计数的评估与训练边界索引 | [training-data](../pretraining/training-data/) |
| source-level split / group split | 来源级切分 / group split | 译并保留常用英文;按文档、来源、用户、仓库或重复簇整体切分以避免相关记录跨 split | [training-data](../pretraining/training-data/) |
| retention rate | 保留率 | 译;过滤或去重后输出记录数或有效 token 数除以对应输入计数的比例 | [training-data](../pretraining/training-data/) |
| effective token mixture | 有效 token 混合比例 | 译;按 tokenizer、截断、packing 和 loss mask 后真正进入目标分母的来源 token 份额 | [training-data](../pretraining/training-data/) |
| personally identifiable information / PII | 个人可识别信息 / PII | 译并保留缩写;可直接或结合其他字段识别个人的信息类别,数据管线需记录 review、redaction 或 exclusion 状态 | [training-data](../pretraining/training-data/) |
| scaling law / scaling laws | 缩放定律 | 译;在固定数据、目标和评估协议后,拟合模型规模、数据规模或计算量与验证损失关系的经验幂律近似 | [scaling-laws](../pretraining/scaling-laws/) |
| scaling exponent | 缩放指数 | 译;幂律项 $N^{-\alpha}$、$D^{-\beta}$ 或 $C^{-\gamma}$ 中描述边际收益衰减的指数 | [scaling-laws](../pretraining/scaling-laws/) |
| model scaling / data scaling / compute scaling | 模型缩放 / 数据缩放 / 计算缩放 | 译;分别固定其他实验条件后改变参数量、有效训练 token 或训练计算量的实验轴 | [scaling-laws](../pretraining/scaling-laws/) |
| loss floor / asymptotic loss floor | 损失 floor / 渐近损失下限 | 保留常用英文并译;缩放曲线中的 $L_\infty$ 拟合项,不自动等于真实分布熵或不可约误差 | [scaling-laws](../pretraining/scaling-laws/) |
| diminishing returns | 边际收益递减 | 译;资源继续增加时验证损失仍下降,但每个数量级或单位资源带来的下降幅度变小 | [scaling-laws](../pretraining/scaling-laws/) |
| iso-compute curve / fixed-compute frontier | 等计算量曲线 / 固定计算量前沿 | 译;满足 $C\approx\kappa ND$ 的模型规模与数据规模组合,用于比较资源分配 | [scaling-laws](../pretraining/scaling-laws/) |
| training compute | 训练计算量 | 译;按统一 FLOP、MAC 或其他约定累计前向、反向、attention、FFN 和相关训练成本的量 | [scaling-laws](../pretraining/scaling-laws/) |
| scaling grid | 缩放实验网格 | 译;在模型、数据、计算或资源分配轴上取多个受控配置以拟合曲线和检查断点的实验集合 | [scaling-laws](../pretraining/scaling-laws/) |
| scaling extrapolation | 缩放外推 | 译;把观测规模区间拟合的损失关系预测到未观测模型、数据或计算范围的过程,需单独报告不确定性 | [scaling-laws](../pretraining/scaling-laws/) |
| compute-optimal training | 计算最优训练 | 译;在固定训练计算预算、数据和硬件约束下选择模型规模与有效训练 token 以最小化指定验证损失的训练问题 | [compute-optimal](../pretraining/compute-optimal/) |
| compute budget | 计算预算 | 译;预先规定的 FLOP、设备计数、GPU·小时或 wall-clock 资源上限,必须声明计算口径 | [compute-optimal](../pretraining/compute-optimal/) |
| compute-optimal frontier | 计算最优前沿 | 译;固定计算量曲线上经拟合和离散验证后损失较低的候选区域,不等于一条架构无关的常数比例 | [compute-optimal](../pretraining/compute-optimal/) |
| token-to-parameter ratio | token/参数比 | 保留常用英文并译;有效训练 token 数与模型规模之比,只在固定数据、架构、目标和计算口径后比较 | [compute-optimal](../pretraining/compute-optimal/) |
| active parameters | active 参数 / 激活参数 | 译并保留常用英文;一次 token 前向实际参与计算的参数,与 MoE 总参数、可训练参数和共享权重分开记录 | [compute-optimal](../pretraining/compute-optimal/) |
| finite-data boundary | 有限数据边界 | 译;无约束计算最优解要求的数据量超过可用数据池时由 $D=D_{\mathrm{pool}}$ 形成的可行域边界 | [compute-optimal](../pretraining/compute-optimal/) |
| constrained allocation | 约束分配 | 译;在计算、数据、显存、时间或部署约束下把资源分给模型规模和训练 token 的优化过程 | [compute-optimal](../pretraining/compute-optimal/) |
| effective compute budget | 有效计算预算 | 译;从总预算中扣除候选间固定或预先计量的通信、评估、IO 等成本后用于模型训练项的预算 | [compute-optimal](../pretraining/compute-optimal/) |
| optimizer step | optimizer 更新步 | 保留常用英文并译;梯度累积完成且参数与 optimizer state 成功更新一次的计数,区别于 micro-step | [lr-schedules-at-scale](../pretraining/lr-schedules-at-scale/) |
| micro-step | micro-step / 微步 | 保留常用英文并译;一次 device micro-batch 完成前向与反向的计数,不自动表示参数发生更新 | [lr-schedules-at-scale](../pretraining/lr-schedules-at-scale/) |
| peak learning rate | 峰值学习率 | 译;warmup 结束后调度曲线达到的最大学习率,需与 global batch、loss reduction 和 optimizer 一起报告 | [lr-schedules-at-scale](../pretraining/lr-schedules-at-scale/) |
| token-based schedule | 基于 token 的调度 | 译;用累计有效训练 token 而不是 optimizer step 作为学习率曲线进度的调度方式 | [lr-schedules-at-scale](../pretraining/lr-schedules-at-scale/) |
| stable phase | stable 阶段 / 稳定阶段 | 保留常用英文并译;学习率在峰值附近保持不变、消费主要训练预算的阶段 | [lr-schedules-at-scale](../pretraining/lr-schedules-at-scale/) |
| warmup-stable-decay / WSD | warmup-stable-decay / WSD 调度 | 保留缩写并译;依次经过 warmup、stable 和后段 decay 的学习率调度形状 | [lr-schedules-at-scale](../pretraining/lr-schedules-at-scale/) |
| skipped optimizer update | 跳过 optimizer 更新 | 译;因 overflow、无效梯度或其他保护逻辑而完成反向但未改变参数的训练事件 | [lr-schedules-at-scale](../pretraining/lr-schedules-at-scale/) |
| update norm ratio | 更新范数比率 | 译;参数更新范数除以参数范数的监控量,用于比较不同学习率、batch 和 optimizer state 下的实际移动尺度 | [lr-schedules-at-scale](../pretraining/lr-schedules-at-scale/) |
| distributed training | 分布式训练 | 译;把一次训练拆到多个设备或进程并通过通信保持参数、梯度、激活或专家状态合同的训练方式 | [distributed-training](../pretraining/distributed-training/) |
| data parallelism / DP | 数据并行 / DP | 译并保留缩写;复制模型参数、把 batch 分到多个 rank 后归约梯度的并行方式 | [distributed-training](../pretraining/distributed-training/) |
| tensor parallelism / TP | 张量并行 / TP | 译并保留缩写;沿矩阵轴或 attention head 切分单层计算并用 collective 合并 partial result 的方式 | [distributed-training](../pretraining/distributed-training/) |
| pipeline parallelism / PP | 流水线并行 / PP | 译并保留缩写;沿 layer 或 block 切分模型、用 micro-batch 在 stage 间传递 activation 与 gradient 的方式 | [distributed-training](../pretraining/distributed-training/) |
| pipeline bubble | 流水线 bubble / 流水线空泡 | 保留常用英文并译;流水线填充、排空或 stage 不均衡产生的设备空闲时间 | [distributed-training](../pretraining/distributed-training/) |
| sequence parallelism / context parallelism | 序列并行 / 上下文并行 | 译并保留常用英文;沿 token 或上下文区间切分序列并通过通信完成 attention 读取的并行方式 | [distributed-training](../pretraining/distributed-training/) |
| collective communication | collective 通信 / 集体通信 | 译并保留常用英文;由通信 group 内全部 rank 按相同顺序参与的归约、收集或交换操作 | [distributed-training](../pretraining/distributed-training/) |
| all-reduce / all-gather / reduce-scatter | all-reduce / all-gather / reduce-scatter | 保留通信原语名并解释;分别用于全量归约、拼接分片和归约后分发分片 | [distributed-training](../pretraining/distributed-training/) |
| parameter sharding | 参数分片 | 译;把参数、梯度或 optimizer state 沿 rank group 分开保存并按需 materialize 的内存策略 | [distributed-training](../pretraining/distributed-training/) |
| pipeline stage | 流水线 stage / 阶段 | 保留常用英文并译;流水线并行中持有连续 layer 子集并处理 micro-batch 的计算分区 | [distributed-training](../pretraining/distributed-training/) |
| LoRA | LoRA / 低秩适配 | 译并保留缩写;冻结基础线性层、训练 A/B 两个低秩因子并用 alpha/r 缩放增量的参数高效微调方法 | [lora](../finetuning/lora/) |
| training stability | 训练稳定性 | 译;要求训练状态在有限性、尺度、有效进展和 checkpoint 恢复四个方面满足可检查合同的性质 | [training-stability](../pretraining/training-stability/) |
| finite training state | 有限训练状态 | 译;参数、激活、损失、梯度和优化器状态均不含 NaN 或 Inf 的训练状态 | [training-stability](../pretraining/training-stability/) |
| non-finite value | 非有限值 | 译;NaN、正负 Inf 或由溢出、下溢和未定义运算产生的非有限浮点结果 | [training-stability](../pretraining/training-stability/) |
| first non-finite stage | 首个非有限阶段 | 译;沿数据、前向、损失、反向、优化器和更新边界检查时第一次违反 finite 合同的位置 | [training-stability](../pretraining/training-stability/) |
| update-to-weight ratio | update-to-weight ratio / 更新与参数比 | 保留常用英文并译;参数更新范数除以参数范数,用于比较实际移动相对当前模型尺度的大小 | [training-stability](../pretraining/training-stability/) |
| gradient clip fraction | 梯度裁剪比例 | 译;一段训练中原始梯度范数超过阈值并实际被梯度裁剪的 batch 或 update 比例 | [training-stability](../pretraining/training-stability/) |
| global skip flag | global skip 标志 / 全局跳过标志 | 保留常用英文并译;把任意 rank 的非有限梯度归约为所有 rank 共用的跳过更新决定 | [training-stability](../pretraining/training-stability/) |
| overflow skip | overflow skip / 溢出跳过 | 保留常用英文并译;检测到溢出或非有限梯度后保持参数和优化器状态不变、调整缩放器并记录的保护动作 | [training-stability](../pretraining/training-stability/) |
| attempted update / successful update | attempted update / successful update / 尝试更新 / 成功更新 | 译并保留常用英文;分别表示发起一次更新流程和确实改变参数及优化器状态的更新计数 | [training-stability](../pretraining/training-stability/) |
| recovery checkpoint | 恢复 checkpoint | 译;保存参数、优化器、调度器、缩放器、数据游标和随机状态以便从最近成功 step 重新运行的 checkpoint | [training-stability](../pretraining/training-stability/) |
| stability contract | 稳定性合同 | 译;规定每个训练边界的 finite 检查、尺度指标、更新顺序和恢复语义的可审计约定 | [training-stability](../pretraining/training-stability/) |
| perplexity evaluation | 困惑度评估 | 译;在固定模型、文本、tokenizer、target mask、上下文窗口和聚合规则下计算语言模型平均 NLL 与 PPL 的协议 | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| target loss mask | target loss mask / 目标损失 mask | 保留常用英文并译;选择哪些右移标签进入 NLL 分母的 mask,区别于 causal attention mask 和 padding mask | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| completion-only evaluation | completion-only evaluation / 仅 completion 评估 | 保留常用英文并译;把 prompt 作为条件、只对 response 或指定 completion span 计分的评估口径 | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| context-window evaluation | 上下文窗口评估 | 译;在固定最大 context length、文档边界和上下文截断规则下计算 target 概率的评估方式 | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| sliding-window evaluation / stride | 滑动窗口评估 / stride | 译并保留常用英文;用重叠输入窗口保留左侧 context、只对不重复 target 区域计分的评估方法 | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| target region | target 区域 / 目标区间 | 保留常用英文并译;窗口内真正进入 loss 的不重叠位置集合,可与复用的 context 区域分开记录 | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| bits per byte / BPB | 每字节比特数 / BPB | 译并保留缩写;累计自然对数 NLL 除以原始 bytes 和 ln 2,用于补充跨 tokenizer 的信息量尺度 | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| token-weighted aggregation | token 加权聚合 | 译;把所有有效 target token 的 NLL 总和除以有效 token 总数的聚合规则 | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| document-level bootstrap | 文档级 bootstrap | 保留常用英文并译;以文档或 group 为重采样单位估计 PPL/NLL 不确定性的 bootstrap 方法 | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| clean evaluation set | 清洁评估集 | 译;经过 exact/near duplicate 与训练语料污染检查、并单独记录版本的评估子集 | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| log-sum-exp / logsumexp | log-sum-exp / 对数和指数 | 保留常用写法并译;从 logits 稳定计算归一化对数概率、避免先转低精度 probability 再取 log 的运算 | [evaluation-perplexity](../pretraining/evaluation-perplexity/) |
| emergent ability | emergent ability / 能力涌现 | 保留常用英文并译;在固定模型家族与评测协议下、较小模型未达到成功标准而较大模型达到的任务级行为观察 | [emergent-abilities](../pretraining/emergent-abilities/) |
| capability emergence | 能力涌现现象 | 译;把模型规模、训练资源、任务指标与成功阈值共同定义的能力曲线拐点作为研究对象 | [emergent-abilities](../pretraining/emergent-abilities/) |
| latent success probability | 潜在成功概率 | 译;假设模型在固定样本上产生正确结果的概率,用于和有限样本观察分数区分 | [emergent-abilities](../pretraining/emergent-abilities/) |
| task composition | 任务组合 | 译;要求多个子步骤同时正确的任务结构,可把平滑的子能力概率组合成更陡的整体成功曲线 | [emergent-abilities](../pretraining/emergent-abilities/) |
| metric thresholding | 指标阈值化 | 译;把连续分数按预先规定的阈值映射为通过或失败的离散报告规则 | [emergent-abilities](../pretraining/emergent-abilities/) |
| exact-match threshold | exact match 阈值 | 保留常用英文并译;只有规范化后的输出完全匹配参考答案时才记为成功的离散评分边界 | [emergent-abilities](../pretraining/emergent-abilities/) |
| model family | 模型家族 | 译;共享基座、tokenizer、训练目标和主要配方、只沿受控规模轴变化的一组模型 | [emergent-abilities](../pretraining/emergent-abilities/) |
| scaling axis | 缩放轴 | 译;实验中明确改变的参数量、有效 token、compute、训练时间或其他资源维度 | [emergent-abilities](../pretraining/emergent-abilities/) |
| prompt sensitivity | prompt 敏感性 / 提示敏感性 | 保留常用英文并译;任务分数随模板、few-shot 示例、角色消息或上下文格式改变的程度 | [emergent-abilities](../pretraining/emergent-abilities/) |
| continuous capability metric | 连续能力指标 | 译;保留概率、子步骤比例、数值误差或部分得分以观察离散成功前的渐进变化的指标 | [emergent-abilities](../pretraining/emergent-abilities/) |
| capability boundary | 能力边界 | 译;在有限模型网格中由未通过点和通过点共同确定的观察区间,不自动表示内部相变位置 | [emergent-abilities](../pretraining/emergent-abilities/) |
| inference request | 推理请求 | 译;由 checkpoint、输入 token、生成配置、上下文限制、设备和停止规则共同定义的一次冻结模型执行 | [inference](../inference/inference/) |
| decode step | decode step / 解码步 | 保留常用英文并译;自回归推理中读取当前 KV cache、计算一个新位置并选择下一个 token 的一次前向事件 | [inference](../inference/inference/) |
| time to first token / TTFT | 首 token 延迟 / TTFT | 译并保留缩写;从请求进入开始到返回第一个生成 token 的时间,包含排队、prefill 和首次 decode | [inference](../inference/inference/) |
| inter-token latency | token 间延迟 | 保留常用英文并译;相邻生成 token 返回之间的时间间隔,与 TTFT 和端到端延迟分开记录 | [inference](../inference/inference/) |
| continuous batching | continuous batching / 连续批处理 | 保留常用英文并译;在已有 decode batch 运行期间动态加入新请求、释放结束请求并重排 slot 的调度方式 | [inference](../inference/inference/) |
| prefix cache | prefix cache / 前缀缓存 | 保留常用英文并译;对完全相同的 tokenizer、模板、位置和 mask 前缀复用 prefill 产生的 KV 状态 | [inference](../inference/inference/) |
| padding waste | padding 浪费率 | 译;变长 batch 中理论补齐位置减去有效 token 后占总补齐位置的比例 | [inference](../inference/inference/) |
| stop sequence | stop sequence / 停止序列 | 保留常用英文并译;生成文本尾部匹配后触发停止、并按协议决定是否从返回文本移除的字符串序列 | [inference](../inference/inference/) |
| inference memory ledger | 推理显存账本 | 译;分别记录权重、KV cache、激活和 workspace 峰值显存的资源账本 | [inference](../inference/inference/) |
| generation budget | 生成预算 | 译;由 max new tokens、context limit、超时和并发资源共同规定的一次请求生成上限 | [inference](../inference/inference/) |
| cache slot mapping | cache slot 映射 | 译;continuous batching 中把 request/sequence ID 对应到物理 KV cache 位置的调度状态 | [inference](../inference/inference/) |
| beam width | beam width / 束宽 | 保留常用英文并译;每一步保留的 active 或 finished 候选数量,束宽为 1 时退化为 greedy | [beam-search](../inference/beam-search/) |
| beam score | beam score / 束分数 | 保留常用英文并译;候选序列各步 log probability 的累计值或经长度规则修正后的搜索分数 | [beam-search](../inference/beam-search/) |
| active beam | active beam / 活跃束 | 保留常用英文并译;尚未触发 EOS、stop 或取消、还会继续前向的候选序列 | [beam-search](../inference/beam-search/) |
| finished beam | finished beam / 完成束 | 保留常用英文并译;已满足结束条件、停止扩展并等待最终排序的候选序列 | [beam-search](../inference/beam-search/) |
| parent index | parent index / 父索引 | 保留常用英文并译;从展平的 parent-token 候选恢复旧 beam、以同步推进 token、score 和 KV cache 的索引 | [beam-search](../inference/beam-search/) |
| beam pruning | beam pruning / 束剪枝 | 保留常用英文并译;每步从全部 parent-token 扩展中只保留固定束宽、丢弃其余历史分支的操作 | [beam-search](../inference/beam-search/) |
| length-normalized beam score | 长度归一化束分数 | 译;用序列长度的幂或惩罚函数重新缩放累计 log probability 后用于候选排序的分数 | [beam-search](../inference/beam-search/) |
| constrained beam search | constrained beam search / 约束束搜索 | 保留常用英文并译;让每条 beam 按自身语法或格式状态屏蔽不允许 token 后继续搜索 | [beam-search](../inference/beam-search/) |
| diverse beam search | diverse beam search / 多样性束搜索 | 保留常用英文并译;对相同 token、group 或相似前缀加入多样性惩罚以减少候选重合的束搜索变体 | [beam-search](../inference/beam-search/) |
| greedy decoding | greedy decoding / 贪心解码 | 保留常用英文并译;每个生成位置对选择前 logits 执行固定 argmax、只保留一条自回归路径的确定性解码规则 | [greedy-decoding](../inference/greedy-decoding/) |
| argmax decoding | argmax 解码 | 译;在词表轴上选择最大 processed logit 对应 token 的逐步生成操作 | [greedy-decoding](../inference/greedy-decoding/) |
| tie-breaking | tie-breaking / 平局处理 | 保留常用英文并译;多个 token 具有相同最大分数时按固定词表顺序、token ID 或优先级选择一个 | [greedy-decoding](../inference/greedy-decoding/) |
| processed logits | processed logits / 处理后 logits | 保留常用英文并译;经过 mask、repetition penalty、语法约束或其他 logit processor 后交给选择器的分数 | [greedy-decoding](../inference/greedy-decoding/) |
| logit processor | logit processor / logit 处理器 | 保留常用英文并译;根据历史、生成配置或约束修改候选 token 分数的推理组件 | [greedy-decoding](../inference/greedy-decoding/) |
| top-1 margin | top-1 margin / 第一名间隔 | 保留常用英文并译;最高与第二高 logit 的差值,用于标记 argmax 对数值舍入和 kernel 差异的敏感位置 | [greedy-decoding](../inference/greedy-decoding/) |
| selection mask | selection mask / 选择掩码 | 译;把当前状态不允许生成的词表项设为不可选、再在剩余候选中执行解码规则的 mask | [greedy-decoding](../inference/greedy-decoding/) |
| local argmax | 局部 argmax | 译;只在当前前缀的下一 token 候选中选择最大分数,不维护未选分支的累计序列分数 | [greedy-decoding](../inference/greedy-decoding/) |
| sequence log probability | 序列对数概率 | 译;把自回归序列中每个条件 token 的 log probability 相加得到的累计路径分数 | [greedy-decoding](../inference/greedy-decoding/) |
| key/value projection | key/value 投影 | 译;每层把 hidden state 投影为供后续 query 读取的 key 和 value,其历史结果组成 KV cache | [kv-cache](../inference/kv-cache/) |
| cache length | cache 长度 | 译;当前请求已经写入 K/V 的逻辑 token 数,决定下一位置的 position offset 和可见历史范围 | [kv-cache](../inference/kv-cache/) |
| cache append | cache 追加 | 译;把当前 decode 或 chunk prefill 产生的新 K/V 写入从旧长度开始的连续逻辑位置 | [kv-cache](../inference/kv-cache/) |
| cache offset | cache offset / 缓存偏移 | 保留常用英文并译;已有历史长度进入 causal mask、position ID 和 chunk 写入地址的偏移量 | [kv-cache](../inference/kv-cache/) |
| paged KV cache | paged KV cache / 分页 K/V 缓存 | 保留常用英文并译;把逻辑序列切成固定 block、通过 block table 映射到物理 K/V block 的布局 | [kv-cache](../inference/kv-cache/) |
| block table | block table / 块表 | 保留常用英文并译;记录逻辑 cache block 到物理存储 block 的映射,供分页 kernel 寻址 | [kv-cache](../inference/kv-cache/) |
| beam cache reorder | beam cache 重排 | 译并保留常用英文;beam 选择新父分支后按同一 parent index 重排每层 K/V 的操作 | [kv-cache](../inference/kv-cache/) |
| KV cache quantization | KV cache 量化 | 译;用低比特格式存储 K/V 并配合 scale 或反量化 kernel 降低 cache 字节数 | [kv-cache](../inference/kv-cache/) |
| cache eviction | cache 淘汰 | 译;按窗口、block、超时或请求状态移除不再保留的历史 K/V 并更新所有权 | [kv-cache](../inference/kv-cache/) |
| sliding-window cache | sliding-window cache / 滑动窗口缓存 | 保留常用英文并译;只保存最近固定窗口的 K/V,以固定显存换取改变后的可见历史集合 | [kv-cache](../inference/kv-cache/) |
| incremental decode | incremental decode / 增量解码 | 保留常用英文并译;prefill 后每步只计算新 query、追加新 K/V 并读取历史 cache 的执行路径 | [kv-cache](../inference/kv-cache/) |
| temperature sampling | temperature sampling / 温度采样 | 保留常用英文并译;先把 logits 除以正温度、再从稳定归一化后的 categorical 分布随机选择下一 token 的解码规则 | [temperature-sampling](../inference/temperature-sampling/) |
| temperature decoding | temperature decoding / 温度解码 | 保留常用英文并译;用 temperature 参数改变 next-token categorical 分布、但不直接控制生成长度的解码配置 | [temperature-sampling](../inference/temperature-sampling/) |
| categorical sampling | categorical sampling / 分类分布抽样 | 保留常用英文并译;按词表概率向量从离散 token 类别中抽取一个结果的操作 | [temperature-sampling](../inference/temperature-sampling/) |
| inverse CDF sampling | inverse CDF sampling / 逆 CDF 抽样 | 保留常用英文并译;用均匀随机数落入累计概率区间来选择离散类别的抽样实现 | [temperature-sampling](../inference/temperature-sampling/) |
| CDF boundary | CDF boundary / CDF 边界 | 保留常用英文并译;累计概率与随机数相等时由小于或小于等于规则决定类别的边界协议 | [temperature-sampling](../inference/temperature-sampling/) |
| sampling RNG | sampling RNG / 抽样随机数生成器 | 保留常用英文并译;负责产生抽样随机数、并由算法、状态、精度和消耗顺序共同定义的运行时组件 | [temperature-sampling](../inference/temperature-sampling/) |
| request-level RNG | request-level RNG / 请求级随机数生成器 | 保留常用英文并译;为每条推理请求维护独立随机状态、降低 batch 重排对其他请求随机流影响的协议 | [temperature-sampling](../inference/temperature-sampling/) |
| sampling entropy | sampling entropy / 抽样熵 | 保留常用英文并译;用 categorical 分布的 Shannon entropy 衡量温度采样不确定性、与生成长度分开 | [temperature-sampling](../inference/temperature-sampling/) |
| candidate renormalization | candidate renormalization / 候选重新归一化 | 保留常用英文并译;mask 或 top-k/top-p 删除候选后把剩余概率重新缩放为总和 1 的步骤 | [temperature-sampling](../inference/temperature-sampling/) |
| top-k sampling | top-k sampling / top-k 抽样 | 保留常用英文并译;按分数保留固定数量的最高概率 token、重新归一化后进行 categorical 抽样的解码规则 | [top-k-top-p](../inference/top-k-top-p/) |
| top-p sampling / nucleus sampling | top-p sampling / nucleus sampling / top-p 抽样 | 保留常用英文并译;按概率降序保留累计质量达到阈值的最小 token 前缀、重新归一化后抽样的解码规则 | [top-k-top-p](../inference/top-k-top-p/) |
| top-k filter | top-k filter / top-k 过滤器 | 保留常用英文并译;在采样前把排名低于固定 top-k 边界的 token mask 掉的候选过滤器 | [top-k-top-p](../inference/top-k-top-p/) |
| top-p filter | top-p filter / top-p 过滤器 | 保留常用英文并译;在采样前按累计概率阈值保留最小前缀并 mask 其余 token 的候选过滤器 | [top-k-top-p](../inference/top-k-top-p/) |
| top-p threshold | top-p threshold / top-p 阈值 | 保留常用英文并译;决定累计概率前缀何时停止扩展的 $τ$,必须说明是否包含 crossing token | [top-k-top-p](../inference/top-k-top-p/) |
| crossing token | crossing token / 跨阈值 token | 保留常用英文并译;使累计概率首次达到或超过 top-p 阈值、因此必须保留的最后一个候选 token | [top-k-top-p](../inference/top-k-top-p/) |
| candidate mass | candidate mass / 候选质量 | 保留常用英文并译;截断前概率在保留候选集合上的总和,作为重新归一化的分母 | [top-k-top-p](../inference/top-k-top-p/) |
| min tokens to keep | min tokens to keep / 最少保留 token 数 | 保留常用英文并译;即使累计概率已达到阈值仍强制保留的最小候选数量约束 | [top-k-top-p](../inference/top-k-top-p/) |
| repetition penalty | repetition penalty / 重复惩罚 | 保留常用英文并译;对历史中已出现 token 的 raw logit 按正负符号分别除以或乘以惩罚系数的 logit processor | [repetition-penalty](../inference/repetition-penalty/) |
| repetition penalty processor | repetition penalty processor / 重复惩罚处理器 | 保留常用英文并译;读取 request history、修改 processed logits 并把结果交给 mask、temperature 或候选过滤器的推理组件 | [repetition-penalty](../inference/repetition-penalty/) |
| seen token set | seen token set / 已见 token 集合 | 保留常用英文并译;当前请求历史中至少出现过一次的 token ID 集合,用于 membership 型重复惩罚 | [repetition-penalty](../inference/repetition-penalty/) |
| repetition history | repetition history / 重复历史 | 保留常用英文并译;生成位置之前按约定 prompt、generated 或窗口范围保存的 token ID 序列或统计状态 | [repetition-penalty](../inference/repetition-penalty/) |
| frequency penalty | frequency penalty / 频率惩罚 | 保留常用英文并译;按 token 在历史中的出现次数从 logits 中减去线性惩罚的 score processor | [repetition-penalty](../inference/repetition-penalty/) |
| presence penalty | presence penalty / 存在惩罚 | 保留常用英文并译;只按 token 是否出现过、而不按出现次数从 logits 中减去固定惩罚的 score processor | [repetition-penalty](../inference/repetition-penalty/) |
| count-based penalty | count-based penalty / 计数型惩罚 | 保留常用英文并译;使用历史 token 计数而非仅使用 seen membership 改变下一步分数的惩罚规则 | [repetition-penalty](../inference/repetition-penalty/) |
| no-repeat n-gram | no-repeat n-gram / 不重复 n-gram | 保留常用英文并译;根据最近序列状态把会形成重复 n-gram 的 token 直接 mask 为不可选的硬约束 | [repetition-penalty](../inference/repetition-penalty/) |
| logit score processor | logit score processor / logit 分数处理器 | 保留常用英文并译;在 softmax 或候选过滤前按历史、约束或配置变换词表分数的推理组件 | [repetition-penalty](../inference/repetition-penalty/) |
| speculative decoding | speculative decoding / 投机解码 | 保留常用英文并译;由 draft model 提议多个 token、target model 一次验证并在首个拒绝处按残差修正的推理算法 | [speculative-decoding](../inference/speculative-decoding/) |
| speculative sampling | speculative sampling / 投机采样 | 保留常用英文并译;用 proposal 分布产生候选、用接受—拒绝和 residual distribution 保持 target 边际分布的采样过程 | [speculative-decoding](../inference/speculative-decoding/) |
| draft model | draft model / 草稿模型 | 保留常用英文并译;以较低单步成本自回归提出候选 token、由 target model 验证的辅助模型 | [speculative-decoding](../inference/speculative-decoding/) |
| target model | target model / 目标模型 | 保留常用英文并译;定义最终 next-token 分布、验证 draft proposal 并决定提交结果的权威模型 | [speculative-decoding](../inference/speculative-decoding/) |
| proposal distribution | proposal distribution / 提议分布 | 保留常用英文并译;draft model 在当前已提交前缀和草稿历史下用于提出 token 的概率分布 $q$ | [speculative-decoding](../inference/speculative-decoding/) |
| target distribution | target distribution / 目标分布 | 保留常用英文并译;经过最终 score processor 和采样配置后、投机解码需要保持边际一致的概率分布 $p$ | [speculative-decoding](../inference/speculative-decoding/) |
| acceptance probability | acceptance probability / 接受概率 | 保留常用英文并译;对 draft token 使用 $min(1,p(v)/q(v))$ 决定接受或进入残差修正的概率 | [speculative-decoding](../inference/speculative-decoding/) |
| acceptance rate | acceptance rate / 接受率 | 保留常用英文并译;每个验证位置的 $sum_vmin(p(v),q(v))$,衡量 draft 与 target 分布重叠程度 | [speculative-decoding](../inference/speculative-decoding/) |
| residual distribution | residual distribution / 残差分布 | 保留常用英文并译;按 $max(p(v)-q(v),0)$ 取正差并重新归一化、用于首个拒绝位置修正的分布 | [speculative-decoding](../inference/speculative-decoding/) |
| accepted prefix | accepted prefix / 接受前缀 | 保留常用英文并译;一轮验证中从当前请求前缀开始连续通过接受检验、可以提交并保留 cache 的 draft token 前缀 | [speculative-decoding](../inference/speculative-decoding/) |
| speculative rollback | speculative rollback / 投机回滚 | 保留常用英文并译;拒绝后删除未提交 draft/target token、回收临时 KV block 并为 correction token 重算状态的操作 | [speculative-decoding](../inference/speculative-decoding/) |
| target verification | target verification / 目标验证 | 译并保留常用英文;target model 在一次前向中计算多个 draft position 的概率、接受比率和首个拒绝位置的过程 | [speculative-decoding](../inference/speculative-decoding/) |
| inference math / inference arithmetic | 推理数学 / 推理算力账 | 译;把 prefill、decode、矩阵乘法、attention、LM head、KV cache、权重和延迟拆成可核对资源项的分析方法 | [inference-math](../inference/inference-math/) |
| prefill compute | prefill 计算 | 译;一次处理有效 prompt token 时由投影、$T^2$ attention 交互、FFN 和 LM head 组成的计算量 | [inference-math](../inference/inference-math/) |
| decode compute | decode 计算 | 译;每个新 token 的投影、读取长度为 $P$ 的 KV attention、FFN 和 LM head 计算量 | [inference-math](../inference/inference-math/) |
| KV width | K/V 宽度 | 译;每个 token 的 K 或 V 投影宽度 $D_{kv}=h_{kv}d_h$,由 K/V head 数和每头维度共同决定 | [inference-math](../inference/inference-math/) |
| prefill attention cost | prefill attention 成本 | 译;按 $2BT^2D$ 估计完整 prompt 的 query-key 与 attention-value 交互 MAC 数的主要项 | [inference-math](../inference/inference-math/) |
| decode attention cost | decode attention 成本 | 译;按 $2BPD$ 估计单个新 query 读取历史 K/V 的交互 MAC 数的主要项 | [inference-math](../inference/inference-math/) |
| arithmetic intensity | 算术强度 | 译;执行的 FLOPs 除以从内存层级搬运的 bytes,用于比较计算瓶颈和带宽瓶颈 | [inference-math](../inference/inference-math/) |
| roofline lower bound | roofline 下界 | 译;由 FLOP 峰值和内存带宽分别给出的理想运行时间下界,不包含排队、调度和 kernel 空隙 | [inference-math](../inference/inference-math/) |
| logical cache length / allocated cache length | 逻辑缓存长度 / 分配缓存长度 | 译;分别表示请求可见的有效 K/V token 数和物理 page、block 或 slot 已分配的容量,用于区分语义长度与显存占用 | [inference-math](../inference/inference-math/) |
| quantization | quantization / 量化 | 译;把浮点权重、激活或 KV cache 映射为有限位宽 code,用 scale 和可选 zero point 恢复近似值的表示变换 | [quantization](../inference/quantization/) |
| uniform quantization | uniform quantization / 均匀量化 | 译;用固定 scale 让相邻整数 code 对应等间距浮点区间的量化器 | [quantization](../inference/quantization/) |
| symmetric quantization | symmetric quantization / 对称量化 | 译;以浮点零和整数零为中心、通常固定 zero point 为 0 的量化方案 | [quantization](../inference/quantization/) |
| asymmetric quantization | asymmetric quantization / 非对称量化 | 译;按独立的 $x_{\min}$、$x_{\max}$ 映射到整数码范围、允许 zero point 偏离 0 的量化方案 | [quantization](../inference/quantization/) |
| quantization scale | quantization scale / 量化 scale | 保留常用英文并译;把浮点范围映射到相邻整数 code 间距的正标量 $s$,必须连同 dtype 和粒度保存 | [quantization](../inference/quantization/) |
| zero point | zero point / 零点 | 保留常用英文并译;使浮点零对应整数 code 的偏移 $z$,非对称量化需要把它纳入量化与反量化合同 | [quantization](../inference/quantization/) |
| quantization code | quantization code / 量化码 | 译;存储在 int4、int8 或 codebook 中、需要结合 scale 和 zero point 解读的有限位宽值 | [quantization](../inference/quantization/) |
| quantization error | quantization error / 量化误差 | 译;原始浮点值与反量化近似值之差,由舍入、截断、scale 和 metadata 共同决定 | [quantization](../inference/quantization/) |
| clipping range / calibration range | clipping range / calibration range / 截断范围 / 校准范围 | 译并保留常用英文;决定哪些浮点值映射到可表达端点、由 min-max、percentile、MSE 或固定配置得到的范围 | [quantization](../inference/quantization/) |
| saturation ratio | saturation ratio / 饱和比例 | 译;量化前超出范围并被截断到最小或最大 code 的元素比例 | [quantization](../inference/quantization/) |
| per-tensor quantization | per-tensor quantization / 张量级量化 | 保留常用英文并译;整个张量共享一个 scale 和可选 zero point 的粒度 | [quantization](../inference/quantization/) |
| per-channel quantization | per-channel quantization / 通道级量化 | 保留常用英文并译;沿指定 channel 轴为每个通道保存独立 scale 的粒度 | [quantization](../inference/quantization/) |
| per-group quantization | per-group quantization / 分组量化 | 保留常用英文并译;把连续权重元素切成固定 group、每组保存 scale 的粒度 | [quantization](../inference/quantization/) |
| group size | group size / 分组大小 | 保留常用英文并译;一个 scale 覆盖的连续权重元素数量,同时决定误差、metadata 和 kernel 布局 | [quantization](../inference/quantization/) |
| weight-only quantization | weight-only quantization / 仅权重量化 | 保留常用英文并译;只压缩权重存储、让激活保持 FP16 或 BF16 的推理路径,常见于 W4A16 | [quantization](../inference/quantization/) |
| activation quantization | activation quantization / 激活量化 | 译;把运行时 hidden、输入或中间输出映射为低比特表示,需要处理当前输入分布和非线性误差 | [quantization](../inference/quantization/) |
| static activation quantization | static activation quantization / 静态激活量化 | 保留常用英文并译;离线 calibration 后固定部署 scale、运行时不重新估计范围的激活量化 | [quantization](../inference/quantization/) |
| dynamic activation quantization | dynamic activation quantization / 动态激活量化 | 保留常用英文并译;按当前 batch、token 或 block 运行时计算 scale 的激活量化 | [quantization](../inference/quantization/) |
| W4A16 / W8A8 | W4A16 / W8A8 | 保留配置写法;分别表示权重与激活的存储或计算位宽,必须补充 scale、累加器和输出 dtype | [quantization](../inference/quantization/) |
| quantization calibration | quantization calibration / 量化校准 | 译;使用代表性输入估计激活或权重范围、scale、zero point 和误差目标的离线过程 | [quantization](../inference/quantization/) |
| calibration dataset | calibration dataset / 校准数据集 | 译;覆盖 prompt、长度、领域、batch 和目标任务分布、用于估计量化范围的固定输入集合 | [quantization](../inference/quantization/) |
| outlier channel | outlier channel / 离群通道 | 译;动态范围显著高于其他通道、会支配共享 scale 并降低普通值分辨率的通道 | [quantization](../inference/quantization/) |
| dequantization | dequantization / 反量化 | 译;按 scale、zero point 和 codebook 把低比特 code 恢复为计算域近似值的过程 | [quantization](../inference/quantization/) |
| requantization | requantization / 再量化 | 译;把计算结果从一个浮点或整数范围重新映射到目标低比特 code 域的步骤 | [quantization](../inference/quantization/) |
| codebook quantization | codebook quantization / 码本量化 | 译;用有限的非均匀浮点或整数 codebook 表示值、再按索引恢复近似值的量化方案 | [quantization](../inference/quantization/) |
| NF4 | NF4 / NormalFloat4 | 保留名称并译;按近似正态权重分布设计的 4 bit 非均匀 codebook,仍需记录 block scale 和实现版本 | [quantization](../inference/quantization/) |
| GPTQ / AWQ / SmoothQuant | GPTQ / AWQ / SmoothQuant | 保留算法名称;分别代表不同的权重误差补偿、激活感知权重保护或权重—激活尺度重分配方法,不能替代完整量化合同 | [quantization](../inference/quantization/) |
| long context / long-context | long context / long-context / 长上下文 | 译;在较长 token 序列上继续读取、推理或生成、并按长度与位置报告有效能力的推理场景 | [long-context](../inference/long-context/) |
| context window | context window / 上下文窗口 | 译;模型、位置机制和运行时允许处理的最大 token 数,不等同于任务上仍可靠的 effective context | [long-context](../inference/long-context/) |
| effective context | effective context / 有效上下文 | 译;在固定任务、长度、位置、干扰项和质量阈值下仍能可靠使用的信息范围 | [long-context](../inference/long-context/) |
| retrieval distance | retrieval distance / 检索距离 | 译;查询位置与证据 token 位置之间的序列距离,需和方向、文档边界及位置编码一起记录 | [long-context](../inference/long-context/) |
| usable generation budget | usable generation budget / 可用生成预算 | 译;在 context window 中扣除 prompt、系统消息和工具定义后、为新 token、EOS、stop 和超时保留的请求级上限 | [long-context](../inference/long-context/) |
| long-context evaluation | long-context evaluation / 长上下文评测 | 译;同时按总长度、证据位置、干扰项、任务类型和运行模式核验有效上下文的评测协议 | [long-context](../inference/long-context/) |
| lost in the middle | lost in the middle / 中间信息退化 | 保留常用英文并译;同一证据放在长序列中间时,相较开头或结尾位置的任务质量下降现象 | [long-context](../inference/long-context/) |
| evidence position | evidence position / 证据位置 | 译;答案所需证据在 token 序列中的起止位置或相对比例,用于构造位置分桶 | [long-context](../inference/long-context/) |
| position bucket | position bucket / 位置桶 | 译;把证据或 target token 按开头、前中、中间、后中、结尾等相对位置分组的评测区间 | [long-context](../inference/long-context/) |
| attention distance | attention distance / 注意力距离 | 译;query 与可读 key 之间的位置间隔,用于分析局部窗口、远距离读取和距离偏置 | [long-context](../inference/long-context/) |
| distractor document | distractor document / 干扰文档 | 译;包含相似、冲突或无关信息、用于测试模型证据选择和长上下文抗干扰能力的文档 | [long-context](../inference/long-context/) |
| sliding-window attention | sliding-window attention / 滑动窗口注意力 | 保留常用英文并译;每个 query 只读取最近固定窗口、通过局部可达路径控制 attention 成本的组织方式 | [long-context](../inference/long-context/) |
| context compression | context compression / 上下文压缩 | 译;用摘要、局部状态、检索片段或其他表示减少原始 token 数、同时保留任务所需信息的过程 | [long-context](../inference/long-context/) |
| long-context retrieval | long-context retrieval / 长上下文检索 | 译;在长文档中定位、选择并把远距离证据交给当前 query 的读取任务与系统路径 | [long-context](../inference/long-context/) |
| position extrapolation | position extrapolation / 位置外推 | 译;在训练位置范围之外使用位置机制并观察模型能否保持相对位置和任务质量的推理情形 | [long-context](../inference/long-context/) |
| context extension / context-extension | 上下文扩展 / context-extension | 译;把训练或原始配置覆盖的 token 长度延伸到更大目标范围、同时重新核对位置合同、质量和资源的方案 | [context-extension](../inference/context-extension/) |
| extension factor | 扩展因子 | 译;目标长度与训练长度的比值 $r=T_{\mathrm{target}}/T_{\mathrm{train}}$,用于描述位置或频率的缩放范围 | [context-extension](../inference/context-extension/) |
| direct position extrapolation | 直接位置外推 | 译;把训练范围外的新 position ID 原样送入位置机制、保留原始距离但访问未覆盖坐标的方案 | [context-extension](../inference/context-extension/) |
| linear position interpolation | 线性位置插值 | 译;按 $p'=pT_{\mathrm{train}}/T_{\mathrm{target}}$ 把目标位置压回训练区间的映射 | [context-extension](../inference/context-extension/) |
| RoPE scaling | RoPE scaling / RoPE 缩放 | 保留常用英文并译;通过位置映射、频率表或 base 改变 RoPE 相位合同以覆盖更长序列的方案 | [context-extension](../inference/context-extension/) |
| position mapping | position mapping / 位置映射 | 译并保留常用英文;把运行时目标位置 $p$ 转换为位置编码实际使用坐标的函数 | [context-extension](../inference/context-extension/) |
| frequency scaling | frequency scaling / 频率缩放 | 译并保留常用英文;按扩展规则改变 RoPE inverse frequency、从而改变各坐标对相位变化速度的操作 | [context-extension](../inference/context-extension/) |
| RoPE base | RoPE base / RoPE 基值 | 保留常用英文并译;生成各维 inverse frequency 的幂函数底数,改变它会同时改变整组频率 | [context-extension](../inference/context-extension/) |
| phase angle | phase angle / 相位角 | 译并保留常用英文;位置与频率的乘积 $\theta_i(p)=p\omega_i$,决定 RoPE 二维坐标对的旋转角度 | [context-extension](../inference/context-extension/) |
| continued pretraining for context extension | 上下文扩展继续预训练 | 译;在长序列数据和扩展位置合同上更新权重、让模型接触新的位置分布与跨段依赖的训练阶段 | [context-extension](../inference/context-extension/) |
| dynamic position scaling | dynamic position scaling / 动态位置缩放 | 译并保留常用英文;按请求长度或运行时规则选择扩展因子、并要求 prefill 与 decode 的 cache 合同一致 | [context-extension](../inference/context-extension/) |
| prefix cache contract | prefix cache contract / 前缀缓存合同 | 译并保留常用英文;规定 checkpoint、模板、位置规则、factor、mask、dtype 和 cache layout 必须一致的复用条件 | [context-extension](../inference/context-extension/) |
| short-context regression | short-context regression / 短上下文回归 | 译并保留常用英文;上下文扩展后在原长度的 logits、任务分数或格式能力下降的回归检查 | [context-extension](../inference/context-extension/) |
| effective length | effective length / 有效长度 | 译;在固定任务、位置和质量阈值下仍能可靠使用的 token 长度,不等同于运行时窗口上限 | [context-extension](../inference/context-extension/) |
| attention factor | attention factor / 注意力因子 | 译并保留常用英文;上下文扩展中作用于 attention score 的额外校准标量,必须和位置频率规则一起记录 | [context-extension](../inference/context-extension/) |
| supervised fine-tuning / SFT | 监督微调 / SFT | 译并保留缩写;从预训练 checkpoint 出发、用带输入和目标答案的示范数据更新参数的 next-token 训练阶段 | [sft](../finetuning/sft/) |
| SFT data contract | SFT 数据合同 | 译并保留常用英文;规定示范来源、role、chat template、tokenizer、loss mask、切分和版本的可复现数据接口 | [sft](../finetuning/sft/) |
| assistant-only loss | assistant-only loss / 仅 assistant 损失 | 译并保留常用英文;只让 assistant target token 进入交叉熵、把 system 和 user token 作为条件而不计入监督的口径 | [sft](../finetuning/sft/) |
| full-sequence loss | full-sequence loss / 全序列损失 | 译并保留常用英文;把模板化序列中包括 prompt 在内的多个 next-token 位置都计入损失的口径 | [sft](../finetuning/sft/) |
| last-turn-only loss | last-turn-only loss / 仅最后轮损失 | 译并保留常用英文;多轮对话中只对最后一个 assistant 响应计算损失、历史消息只作为条件的口径 | [sft](../finetuning/sft/) |
| chat template | chat template / 对话模板 | 译并保留常用英文;把带 role 的消息列表渲染为模型实际读取的字符串、separator 和 special token 序列的规则 | [sft](../finetuning/sft/) |
| generation prompt | generation prompt / 生成提示 | 译并保留常用英文;推理前追加的 assistant 起始标记或等价模板片段、决定模型从哪个 role 开始生成 | [sft](../finetuning/sft/) |
| role separator | role separator / 角色分隔符 | 译并保留常用英文;区分 system、user、assistant、tool 消息并参与 tokenization、mask 和生成边界的标记 | [sft](../finetuning/sft/) |
| supervised token count | supervised token count / 监督 token 数 | 译;经过 shift、padding、role 和 loss mask 后真正进入目标损失分母的 token 数 | [sft](../finetuning/sft/) |
| token-mean loss | token-mean loss / token 均值损失 | 译并保留常用英文;把所有有效 target token 的损失求和后除以有效 token 总数的归约方式 | [sft](../finetuning/sft/) |
| example-mean loss | example-mean loss / 样本均值损失 | 译并保留常用英文;先计算每条样本的有效 token 平均损失、再对样本平均的归约方式 | [sft](../finetuning/sft/) |
| SFT evaluation contract | SFT 评测合同 | 译并保留常用英文;固定 checkpoint、mask、模板、tokenizer、decode、stop、scorer 和回归切分的评测接口 | [sft](../finetuning/sft/) |
| instruction data / instruction-data | 指令数据 / instruction-data | 译并保留常用英文;把任务意图、输入上下文、约束、目标答案和来源元数据组织成监督示范的记录集合 | [instruction-data](../finetuning/instruction-data/) |
| instruction record | 指令数据记录 | 译;包含 instruction、input、context、constraint、output 和可追溯元数据的单条任务记录 | [instruction-data](../finetuning/instruction-data/) |
| task instruction | 任务指令 | 译;说明需要完成的动作、目标或问题、与作用对象 input 和答案 output 分开保存 | [instruction-data](../finetuning/instruction-data/) |
| task constraint | 任务约束 | 译;规定语言、长度、格式、字段、工具或安全边界的输出条件 | [instruction-data](../finetuning/instruction-data/) |
| target answer | 目标答案 | 译;在给定 instruction、input 和 context 后作为监督 target 的示范输出,需要按任务验证正确性与完整性 | [instruction-data](../finetuning/instruction-data/) |
| structured output schema | 结构化输出 schema | 译并保留常用英文;规定 JSON、工具参数或其他结构化答案的字段、类型、必填项和取值约束 | [instruction-data](../finetuning/instruction-data/) |
| instruction-data validator | 指令数据验证器 | 译并保留常用英文;按 schema、解析、事实、代码、格式或安全规则检查记录并保存版本与结果的程序 | [instruction-data](../finetuning/instruction-data/) |
| instruction quality vector | 指令数据质量向量 | 译并保留常用英文;分别记录正确性、相关性、完整性、格式和安全等质量分量,不把单一总分当作全部依据 | [instruction-data](../finetuning/instruction-data/) |
| coverage matrix | 覆盖矩阵 | 译;按 task family、language/domain 和 output format 记录每个组合是否出现及其样本和 token 数量 | [instruction-data](../finetuning/instruction-data/) |
| data card | 数据卡 | 译;记录数据来源、许可、schema、过滤、去重、分布、切分、tokenizer 和训练接口的数据版本说明 | [instruction-data](../finetuning/instruction-data/) |
| raw count / kept count | 原始数量 / 保留数量 | 译;分别表示过滤前记录数与通过当前 validator、去重和隐私规则后进入候选集的记录数 | [instruction-data](../finetuning/instruction-data/) |
| observed share | 实际份额 | 译;训练运行中按样本或有效 token 统计的实际来源占比,用于和采样配置中的 target share 比较 | [instruction-data](../finetuning/instruction-data/) |
| full fine-tuning / full-parameter fine-tuning | 全量微调 / 全参数微调 | 译;从预训练 checkpoint 出发、允许基础模型的大部分或全部参数获得梯度并更新的适配方案 | [full-vs-peft](../finetuning/full-vs-peft/) |
| parameter-efficient fine-tuning / PEFT | 参数高效微调 / PEFT | 译并保留缩写;冻结基础权重、只更新 adapter、低秩因子、prefix 或其他小参数集合的微调方案 | [full-vs-peft](../finetuning/full-vs-peft/) |
| trainable parameter set | 可训练参数集合 | 译;一次实验中实际参与梯度、optimizer state 和 checkpoint 更新的参数名、shape、dtype 集合 | [full-vs-peft](../finetuning/full-vs-peft/) |
| frozen base weight | 冻结基础权重 | 译;参与前向但不接收训练更新的预训练权重,仍可能占主要权重显存 | [full-vs-peft](../finetuning/full-vs-peft/) |
| adapter | adapter / 适配器 | 保留常用英文并译;插入基础模型路径、只保存少量可训练参数并在运行时提供任务增量的模块 | [full-vs-peft](../finetuning/full-vs-peft/) |
| adapter checkpoint | adapter checkpoint / 适配器 checkpoint | 译并保留常用英文;只保存 adapter 参数、配置和 base model 标识、需要与基础权重成对加载的 checkpoint | [full-vs-peft](../finetuning/full-vs-peft/) |
| target module | target module / 目标模块 | 译并保留常用英文;允许 PEFT 插入或更新的具体层、投影、embedding、LM head 或 norm 参数集合 | [full-vs-peft](../finetuning/full-vs-peft/) |
| trainable ratio | trainable ratio / 可训练参数比率 | 译并保留常用英文;可训练参数数除以基础模型参数数的比例 $P_{\mathrm{train}}/P$ | [full-vs-peft](../finetuning/full-vs-peft/) |
| adapter merge | adapter merge / 适配器合并 | 译并保留常用英文;把 adapter 增量写回基础权重、生成 merged checkpoint 的操作,必须保存 dtype、scale 和 hash | [full-vs-peft](../finetuning/full-vs-peft/) |
| merged checkpoint / unmerged checkpoint | 合并 checkpoint / 未合并 checkpoint | 译并保留常用英文;分别表示增量已写入基础权重或仍以独立 adapter 前向加载的部署形态 | [full-vs-peft](../finetuning/full-vs-peft/) |
| prefix tuning / prompt tuning | prefix tuning / prompt tuning | 保留常用英文并译;只训练虚拟 token 或 attention prefix 参数、把任务条件注入输入或读取路径的 PEFT 方案 | [prompt-tuning](../finetuning/prompt-tuning/) |
| gradient payload | 梯度 payload | 译并保留常用英文;一次数据并行更新需要同步的 trainable gradient 字节量,近似为参数数乘以梯度 dtype 字节数 | [full-vs-peft](../finetuning/full-vs-peft/) |
| LoRA factor | LoRA 因子 | 译并保留常用英文;低秩增量中的 A∈R^(r×d_in) 或 B∈R^(d_out×r),二者相乘得到不单独保存的更新矩阵 | [lora](../finetuning/lora/) |
| LoRA scale / alpha | LoRA scale / alpha / LoRA 缩放 | 译并保留常用英文;实际进入 forward 的增量乘数,常见约定为 s=alpha/r,训练和合并必须使用同一约定 | [lora](../finetuning/lora/) |
| zero-initialized adapter | 零增量初始化 adapter | 译;常用 B=0、A 为小幅随机值的 LoRA 初始化,使插入时的增量为零但首步两个因子梯度不对称 | [lora](../finetuning/lora/) |
| factor shape | 因子 shape | 译并保留常用英文;按输入维度、输出维度和 rank 记录 A/B 的矩阵形状,用于检查转置与参数量 | [lora](../finetuning/lora/) |
| rank budget | rank 预算 / 秩预算 | 译并保留常用英文;在目标层数量、每层 rank 和矩阵形状之间分配可训练参数的资源约束 | [lora](../finetuning/lora/) |
| fused QKV | 融合 QKV | 译并保留常用英文;把 query、key、value 投影沿输出轴拼成一个参数矩阵的实现,其 LoRA 参数量需按融合 shape 计算 | [lora](../finetuning/lora/) |
| low-rank update subspace | 低秩更新子空间 | 译;由 rank 不超过 r 的 BA 增量限制出的可训练矩阵集合,不等于先训练全量更新再做 SVD | [lora](../finetuning/lora/) |
| merged/unmerged equivalence | 合并/未合并等价性 | 译并保留常用英文;固定 dtype、scale、bias 和 dropout 后,运行时低秩路径与 W_0+sBA 合并路径应在舍入误差内一致 | [lora](../finetuning/lora/) |
| base model hash | 基础模型 hash | 译并保留常用英文;adapter 加载时用于确认 tokenizer、模块 shape 和基础权重与训练 checkpoint 相同的内容标识 | [lora](../finetuning/lora/) |
| QLoRA | QLoRA / 量化低秩适配 | 保留方法名并译;以低 bit 形式保存冻结基础权重、按计算 dtype 反量化、只训练 LoRA adapter 的参数高效微调方案 | [qlora](../finetuning/qlora/) |
| quantized base model | 量化基础模型 | 译并保留常用英文;冻结基础 checkpoint 经低 bit index、group scale 和可选二次量化保存、前向时反量化使用的模型 | [qlora](../finetuning/qlora/) |
| double quantization / nested quantization | 二次量化 / 嵌套量化 | 译并保留常用英文;再次量化第一层量化常数、用更大 nested group 保存第二层 metadata 的压缩步骤 | [qlora](../finetuning/qlora/) |
| nested group | nested group / 嵌套分组 | 译并保留常用英文;覆盖多个第一层 group scale、用于保存二次量化 scale metadata 的更大分组 | [qlora](../finetuning/qlora/) |
| compute dtype | compute dtype / 计算 dtype | 译并保留常用英文;反量化后的基础权重、LoRA 路径和矩阵累加实际使用的 BF16、FP16 或其他计算精度 | [qlora](../finetuning/qlora/) |
| paged optimizer | paged optimizer / 分页优化器 | 译并保留常用英文;把 optimizer state 按页在 GPU 与主存之间调度、降低显存峰值但不消除 state 的优化器实现 | [qlora](../finetuning/qlora/) |
| adapter-only optimizer state | 仅 adapter 优化器状态 | 译并保留常用英文;只为 LoRA A/B 等可训练参数维护权重、梯度和 Adam moments、冻结量化基础权重不进入 state 的账本 | [qlora](../finetuning/qlora/) |
| quantized base regression | 量化基础模型回归 | 译;用固定输入比较浮点基础模型与量化基础模型的 logits、top-k 或任务结果、分离量化误差与 adapter 训练效果 | [qlora](../finetuning/qlora/) |
| dequantized merged checkpoint / requantized merged checkpoint | 反量化合并 checkpoint / 再量化合并 checkpoint | 译并保留常用英文;分别表示在浮点近似权重中写回 LoRA 或写回后重新编码到低 bit 的部署产物 | [qlora](../finetuning/qlora/) |
| soft prompt | soft prompt / 软提示 | 保留常用英文并译;直接训练并拼接到真实 token embedding 前的连续向量矩阵,不经过 tokenizer 且不增加 vocabulary ID | [prompt-tuning](../finetuning/prompt-tuning/) |
| virtual token | virtual token / 虚拟 token | 保留常用英文并译;占用序列位置、参与 attention 但没有对应离散词表 ID 的可训练输入向量 | [prompt-tuning](../finetuning/prompt-tuning/) |
| prompt length | prompt length / prompt 长度 | 译并保留常用英文;软提示包含的虚拟 token 数 m,同时决定参数量、position 偏移、attention 和 KV cache 开销 | [prompt-tuning](../finetuning/prompt-tuning/) |
| input prompt / deep prompt | 输入级 prompt / 深层 prompt | 译并保留常用英文;分别表示只在输入 embedding 前拼接或在多个 Transformer 层接口注入可训练向量的方案 | [prompt-tuning](../finetuning/prompt-tuning/) |
| prompt position offset | prompt position offset / prompt 位置偏移 | 译并保留常用英文;虚拟 token 占用前置位置后、真实 token 和生成 token 需要使用的整体 position 偏移 | [prompt-tuning](../finetuning/prompt-tuning/) |
| prompt KV cache | prompt KV cache / prompt KV 缓存 | 译并保留常用英文;输入级虚拟 token 在 decoder 生成时产生并保留的各层 K/V 状态 | [prompt-tuning](../finetuning/prompt-tuning/) |
| prompt-only optimizer state | 仅 prompt 优化器状态 | 译并保留常用英文;只为软提示参数维护权重、梯度和 optimizer moments、冻结基础模型不进入 state 的账本 | [prompt-tuning](../finetuning/prompt-tuning/) |
| prompt checkpoint contract | prompt checkpoint 合同 | 译并保留常用英文;绑定 prompt tensor、length、dtype、base hash、tokenizer、template、position 和 mask 的加载接口 | [prompt-tuning](../finetuning/prompt-tuning/) |
| catastrophic forgetting | catastrophic forgetting / 灾难性遗忘 | 保留术语并译;连续训练新任务后、旧任务 loss 上升、accuracy 下降或行为回归的现象,必须相对同一 base 和评测合同测量 | [catastrophic-forgetting](../finetuning/catastrophic-forgetting/) |
| old-task regression / new-task gain | 旧任务回归 / 新任务增益 | 译并保留常用英文;分别表示 adapted checkpoint 相对 base 的旧任务损失/分数变化与新任务分数变化 | [catastrophic-forgetting](../finetuning/catastrophic-forgetting/) |
| gradient conflict | 梯度冲突 | 译;旧任务与新任务梯度内积为负、一次新任务更新的一阶效果会提高旧任务损失的局部状态 | [catastrophic-forgetting](../finetuning/catastrophic-forgetting/) |
| rehearsal / replay | rehearsal / replay / 回放 | 保留常用英文并译;在新任务训练中混入旧任务样本、token 或行为 target 以提供旧能力保持梯度的策略 | [catastrophic-forgetting](../finetuning/catastrophic-forgetting/) |
| replay buffer | replay buffer / 回放缓冲区 | 保留常用英文并译;按容量、任务份额、时间、去重和隐私规则保存旧任务候选样本的集合 | [catastrophic-forgetting](../finetuning/catastrophic-forgetting/) |
| L2-SP | L2-SP / 参数锚定正则 | 保留方法名并译;在新任务损失之外惩罚参数偏离旧 checkpoint 的二范数距离的保持方法 | [catastrophic-forgetting](../finetuning/catastrophic-forgetting/) |
| EWC | EWC / 弹性权重保持 | 保留方法名并译;按 Fisher 重要性对参数偏离旧 checkpoint 的幅度加权、限制重要参数移动的正则方法 | [catastrophic-forgetting](../finetuning/catastrophic-forgetting/) |
| old-task retention | old-task retention / 旧任务保持率 | 译并保留常用英文;adapted checkpoint 的旧任务分数相对 base 分数的保留比例,需按任务切片和评测协议计算 | [catastrophic-forgetting](../finetuning/catastrophic-forgetting/) |
| behavior regression | behavior regression / 行为回归 | 译并保留常用英文;固定输入、decode 和规则下,模型输出格式、安全边界、拒答或工具行为相对 base 发生的退化 | [catastrophic-forgetting](../finetuning/catastrophic-forgetting/) |
| retention-adaptation tradeoff | 保持—适配折中 | 译;旧任务 retention 与新任务 gain 随 replay、正则、蒸馏、学习率或 adapter 容量变化的共同曲线 | [catastrophic-forgetting](../finetuning/catastrophic-forgetting/) |
| knowledge distillation / distillation | 知识蒸馏 / 蒸馏 | 译;用冻结 teacher 的概率分布、生成结果或中间表示约束 student、把超出 hard label 的关系信息迁移到较小模型的训练方法 | [distillation](../finetuning/distillation/) |
| teacher model / student model | teacher 模型 / student 模型 | 保留常用英文并译;分别表示提供蒸馏目标的模型与接收目标、被训练或部署的模型 | [distillation](../finetuning/distillation/) |
| hard target / soft target | hard target / soft target / 硬目标 / 软目标 | 译并保留常用英文;hard target 是单个标签或 teacher response,soft target 是包含类别或 token 相对概率的分布 | [distillation](../finetuning/distillation/) |
| KD loss | KD loss / 蒸馏损失 | 译并保留常用英文;以温度分布上的 KL 散度为核心、可与 hard cross-entropy 按权重混合的 student 训练损失 | [distillation](../finetuning/distillation/) |
| temperature-scaled target | temperature-scaled target / 温度缩放目标 | 译并保留常用英文;把 teacher 和 student logits 除以正温度后再 softmax、用于暴露非最大类别相对关系的目标分布 | [distillation](../finetuning/distillation/) |
| tau-squared gradient scaling | tau-squared gradient scaling / tau 平方梯度缩放 | 译并保留常用英文;在 KD loss 中乘以 $\tau^2$、补偿温度升高后 softmax 梯度量级约按 $1/\tau^2$ 缩小的约定 | [distillation](../finetuning/distillation/) |
| offline distillation / online distillation | offline distillation / online distillation / 离线蒸馏 / 在线蒸馏 | 译并保留常用英文;offline 预先缓存 teacher 输出后训练 student,online 在同一训练 step 中同时前向 teacher 和 student | [distillation](../finetuning/distillation/) |
| logits cache / top-k logits cache | logits cache / top-k logits cache / logits 缓存 / top-k logits 缓存 | 译并保留常用英文;保存 teacher 完整词表或截断候选的 logits、概率、token ID、dtype、shape 和 hash 以复用 offline 目标 | [distillation](../finetuning/distillation/) |
| vocabulary alignment / tokenizer alignment | 词表对齐 / tokenizer 对齐 | 译并保留常用英文;建立 teacher 与 student 的 class、token、byte span 或序列映射、避免对不同 ID 直接计算 KL | [distillation](../finetuning/distillation/) |
| response distillation | response distillation / 响应蒸馏 | 译并保留常用英文;把 teacher 生成的响应作为 student 的序列监督、需要固定 sampling、stop、过滤和事实校验 | [distillation](../finetuning/distillation/) |
| feature distillation | feature distillation / 特征蒸馏 | 译并保留常用英文;用投影对齐 teacher 与 student 的 hidden、feature 或关系表示并计算额外目标 | [distillation](../finetuning/distillation/) |
| EMA teacher | EMA teacher / EMA teacher 模型 | 保留常用英文并译;用 student 参数的指数移动平均产生 teacher、必须记录 beta、更新时机、stop-gradient 和恢复边界 | [distillation](../finetuning/distillation/) |
| alignment problem | alignment problem / 对齐问题 | 保留常用英文并译;模型实际优化的目标、部署行为与人类意图之间的偏差问题,需要同时审计目标、反馈、分布、约束和行动后果 | [alignment-problem](../alignment/alignment-problem/) |
| intended utility | intended utility / 真实意图效用 | 译并保留常用英文;按正确性、帮助程度、诚实性、安全性和可控性等维度评价输入、输出与外部动作的目标函数 | [alignment-problem](../alignment/alignment-problem/) |
| proxy objective / proxy metric | proxy objective / proxy metric / 代理目标 / 代理指标 | 译并保留常用英文;训练中可计算、但只近似真实意图的 loss、reward、规则分数或偏好分数 | [alignment-problem](../alignment/alignment-problem/) |
| objective misspecification | objective misspecification / 目标错设 | 译并保留常用英文;训练目标遗漏、错误表达或错误权衡重要行为约束、导致优化指标与真实效用排序不一致的情况 | [alignment-problem](../alignment/alignment-problem/) |
| utility vector / value dimension | 效用向量 / 价值维度 | 译并保留常用英文;把正确、帮助、诚实、安全和可控等分量分开记录、避免单一标量隐藏目标冲突 | [alignment-problem](../alignment/alignment-problem/) |
| outer alignment / inner alignment | 外部对齐 / 内部对齐 | 译并保留常用英文;分别检查训练目标是否表达设计意图、模型学到的策略是否在新分布中继续追踪该意图 | [alignment-problem](../alignment/alignment-problem/) |
| specification gaming / reward hacking | 规范投机 / 奖励投机 | 译并保留常用英文;模型提高可见代理分数或满足表面规则、但没有改善甚至损伤真实结果的行为 | [alignment-problem](../alignment/alignment-problem/) |
| proxy optimization | 代理优化 | 译;直接最大化可计算代理目标的过程,需要与独立意图效用、约束违反率和行动后果同时比较 | [alignment-problem](../alignment/alignment-problem/) |
| independent evaluation | 独立评测 | 译;不把训练反馈、同源评分器或训练样本直接复用为验收证据、使用独立切分和独立观察来源的评测 | [alignment-problem](../alignment/alignment-problem/) |
| hard safety constraint | 硬安全约束 | 译并保留常用英文;不能用其他目标的平均提升抵消、需要设置阈值、拦截器或权限边界的安全条件 | [alignment-problem](../alignment/alignment-problem/) |
| action contract | 行动合同 | 译并保留常用英文;规定模型可触发的外部动作、权限、确认、幂等、日志、回滚和人工接管条件的接口 | [alignment-problem](../alignment/alignment-problem/) |
| abstention / clarification | abstention / clarification / 退出动作 / 请求澄清 | 译并保留常用英文;模型在事实、权限或风险不确定时停止当前回答或动作、改为说明限制并请求更多信息 | [alignment-problem](../alignment/alignment-problem/) |
| tail-risk evaluation | tail-risk evaluation / 尾部风险评测 | 译并保留常用英文;单独统计少量高影响失败、最坏分组和严重度,避免平均分掩盖不可接受行为 | [alignment-problem](../alignment/alignment-problem/) |
| Markov decision process / MDP | 马尔可夫决策过程 / MDP | 译并保留缩写;用状态、动作、转移、奖励和折扣描述连续决策问题的形式化对象 | [rl-basics](../alignment/rl-basics/) |
| trajectory | trajectory / 轨迹 | 译并保留常用英文;按时间记录初始状态、动作、奖励、下一状态和终止标记的交互序列 | [rl-basics](../alignment/rl-basics/) |
| return / discounted return | return / discounted return / 回报 / 折扣回报 | 译并保留常用英文;从当前时刻开始按折扣因子加权累加未来奖励的目标量 | [rl-basics](../alignment/rl-basics/) |
| discount factor | discount factor / 折扣因子 | 译并保留常用英文;控制未来奖励相对当前奖励权重的 $\gamma$,不能与奖励数值归一化混同 | [rl-basics](../alignment/rl-basics/) |
| policy | policy / 策略 | 保留常用英文并译;把状态映射为动作概率或确定动作、定义智能体如何与环境交互的规则 | [rl-basics](../alignment/rl-basics/) |
| state-value function / V function | 状态价值函数 / V 函数 | 译并保留常用英文;给定策略从某状态出发的期望折扣回报 $V^\pi(s)$ | [rl-basics](../alignment/rl-basics/) |
| action-value function / Q function | 动作价值函数 / Q 函数 | 译并保留常用英文;固定当前状态和第一步动作、再按策略执行得到的期望折扣回报 $Q^\pi(s,a)$ | [rl-basics](../alignment/rl-basics/) |
| Bellman equation / Bellman expectation equation | Bellman 方程 / Bellman 期望方程 | 保留人名并译;把当前价值拆成一步奖励与折扣未来价值期望的自洽关系 | [rl-basics](../alignment/rl-basics/) |
| Bellman optimality equation | Bellman 最优性方程 | 保留人名并译;对每个状态选择动作价值最大者、定义最优价值不动点的方程 | [rl-basics](../alignment/rl-basics/) |
| Bellman operator / contraction | Bellman 算子 / 压缩映射 | 保留人名并译;把价值映射为一步最优备份、在 $\gamma<1$ 的有限 MDP 中具有收缩性质 | [rl-basics](../alignment/rl-basics/) |
| advantage function | advantage function / 优势函数 | 译并保留常用英文;动作价值减去当前策略状态价值的差 $A^\pi(s,a)$,表示相对策略平均水平的动作增益 | [rl-basics](../alignment/rl-basics/) |
| epsilon-greedy | epsilon-greedy / epsilon-贪心 | 保留常用英文并译;以 $1-\epsilon$ 选择当前估计最优动作、以 $\epsilon$ 在动作中探索的策略 | [rl-basics](../alignment/rl-basics/) |
| softmax exploration | softmax exploration / softmax 探索 | 译并保留常用英文;按动作价值经温度 softmax 分配探索概率、温度越大分布越平 | [rl-basics](../alignment/rl-basics/) |
| on-policy / off-policy | on-policy / off-policy / 同策略 / 异策略 | 保留常用英文并译;分别表示行为策略与目标策略相同或不同的学习设置 | [rl-basics](../alignment/rl-basics/) |
| Monte Carlo return | Monte Carlo 回报 | 译并保留常用英文;等待回合结束后使用实际完整回报更新价值、不使用 bootstrap 的估计 | [rl-basics](../alignment/rl-basics/) |
| temporal-difference learning / TD learning | 时序差分学习 / TD 学习 | 译并保留缩写;用即时奖励和下一状态价值构造 bootstrap 目标、在线更新当前价值的学习方式 | [rl-basics](../alignment/rl-basics/) |
| bootstrapping | bootstrapping / 自举 | 保留常用英文并译;把当前价值估计放入下一次更新目标、用估计传播未来回报的操作 | [rl-basics](../alignment/rl-basics/) |
| n-step return / eligibility trace | n-step 回报 / eligibility trace / 资格迹 | 译并保留常用英文;在实际多步奖励与远端 bootstrap 之间调节回报传播范围的 TD 方法组件 | [rl-basics](../alignment/rl-basics/) |
| policy gradient | policy gradient / 策略梯度 | 译并保留常用英文;用采样回报或优势加权的 $\nabla_\theta\log\pi_\theta(a\mid s)$ 直接更新策略参数的方法 | [rl-basics](../alignment/rl-basics/) |
| actor-critic | actor-critic / 演员—评论家 | 保留常用英文并译;同时学习产生动作的 actor 策略和估计价值或优势的 critic 的结构 | [rl-basics](../alignment/rl-basics/) |
| reward shaping / potential-based shaping | 奖励塑形 / 势函数奖励塑形 | 译并保留常用英文;为提供更密集反馈而修改奖励、势差形式在满足边界条件时可保持策略排序 | [rl-basics](../alignment/rl-basics/) |
| experience replay / replay buffer | experience replay / replay buffer / 经验回放 / 回放缓冲区 | 译并保留常用英文;保存历史转移并随机重采样以降低相邻样本相关性的机制 | [rl-basics](../alignment/rl-basics/) |
| partial observability / belief state | partial observability / belief state / 部分可观测 / belief 状态 | 译并保留常用英文;当前观测不足以确定真实状态时、用历史或状态后验辅助策略决策的设置 | [rl-basics](../alignment/rl-basics/) |
| offline reinforcement learning | offline reinforcement learning / 离线强化学习 | 译并保留常用英文;只用固定行为策略产生的历史转移学习、需要处理动作覆盖和数据外推风险的设置 | [rl-basics](../alignment/rl-basics/) |
| action coverage / OOD action | 动作覆盖 / OOD 动作 | 译并保留常用英文;分别表示数据对状态—动作区域的观测程度与策略选择训练数据外动作的情况 | [rl-basics](../alignment/rl-basics/) |
| termination / truncation | 自然终止 / 时间截断 | 译;区分任务真正结束与因时间上限或采样边界停止、决定 bootstrap 是否保留未来价值的两个标记 | [rl-basics](../alignment/rl-basics/) |
| policy gradient / policy gradient theorem | policy gradient / 策略梯度 | 译并保留常用英文;用回报或 advantage 加权的策略 log probability 梯度直接更新动作分布参数的方法与期望梯度关系 | [policy-gradient](../alignment/policy-gradient/) |
| log-derivative trick / score function | log-derivative trick / score function / 对数导数技巧 / score 函数 | 译并保留常用英文;用 $\nabla p=p\nabla\log p$ 把不可微抽样分布的梯度改写为可采样的 log probability 梯度 | [policy-gradient](../alignment/policy-gradient/) |
| REINFORCE | REINFORCE / 回报加权策略梯度 | 保留方法名并译;用完整轨迹回报乘以每个动作 log probability 梯度、形成无 critic 的 Monte Carlo 策略梯度估计 | [policy-gradient](../alignment/policy-gradient/) |
| reward-to-go | reward-to-go / 从当前时刻起的回报 | 译并保留常用英文;只使用当前动作之后的折扣奖励估计当前动作贡献、去除已确定过去奖励带来的方差 | [policy-gradient](../alignment/policy-gradient/) |
| policy baseline | policy baseline / 策略梯度基线 | 译并保留常用英文;只依赖状态而不依赖当前动作、从回报中减去以降低梯度方差且不改变期望的估计 | [policy-gradient](../alignment/policy-gradient/) |
| generalized advantage estimation / GAE | 广义优势估计 / GAE | 译并保留缩写;把多个按 $\gamma\lambda$ 衰减的 TD error 累加为 advantage、在 bias 与 variance 之间调节的估计 | [policy-gradient](../alignment/policy-gradient/) |
| entropy regularization / entropy coefficient | 熵正则 / 熵系数 | 译并保留常用英文;在回报目标中加入策略熵及其权重、控制动作分布保持探索的正则项 | [policy-gradient](../alignment/policy-gradient/) |
| importance sampling ratio | importance sampling ratio / 重要性采样比率 | 译并保留常用英文;目标策略对行为策略在同一状态动作上的概率比、用于修正 off-policy 轨迹分布 | [policy-gradient](../alignment/policy-gradient/) |
| probability ratio clipping | probability ratio clipping / 概率比裁剪 | 译并保留常用英文;把新旧策略概率比限制在区间内以抑制极端更新、同时改变原始无偏目标的优化形式 | [policy-gradient](../alignment/policy-gradient/) |
| advantage normalization | advantage normalization / 优势标准化 | 译并保留常用英文;按 batch 对 advantage 去均值并按标准差缩放、改变梯度尺度但不等同于原始目标不变 | [policy-gradient](../alignment/policy-gradient/) |
| continuous-action policy / Gaussian policy | 连续动作策略 / Gaussian 策略 | 译并保留常用英文;用连续分布的均值与尺度生成动作、必须把裁剪、tanh 和 log probability 修正纳入合同 | [policy-gradient](../alignment/policy-gradient/) |
| actor loss / critic loss | actor 损失 / critic 损失 | 译并保留常用英文;分别更新动作策略参数和价值估计参数的目标、需要记录权重、detach 和时间 mask | [policy-gradient](../alignment/policy-gradient/) |
| reward model | reward model / 奖励模型 | 保留常用英文并译;从偏好、规则或环境反馈学习输入与候选输出的代理分数、供后续策略优化使用的模型 | [reward-model](../alignment/reward-model/) |
| preference data / preference label | preference data / preference label / 偏好数据 / 偏好标签 | 译并保留常用英文;记录同一输入下候选回答的选择、平局、分歧、标注指南和生成配置的数据 | [reward-model](../alignment/reward-model/) |
| pairwise preference / pairwise logistic loss | 成对偏好 / 成对 logistic 损失 | 译并保留常用英文;用 chosen 与 rejected 的 reward 分差形成 sigmoid 概率和负对数似然的训练形式 | [reward-model](../alignment/reward-model/) |
| Bradley–Terry model | Bradley–Terry 模型 | 保留人名并译;用潜在候选分数差的 sigmoid 把成对选择写成概率,再用最大似然估计全局排序 | [bradley-terry](../alignment/bradley-terry/) |
| worth / ability score | worth / ability score / 候选价值分数 / 能力分数 | 译并保留常用英文;Bradley–Terry 模型中决定候选相对选择概率的潜在分数,绝对零点由参数约束选择 | [bradley-terry](../alignment/bradley-terry/) |
| pairwise log-odds | pairwise log-odds / 成对对数几率 | 译并保留常用英文;两个候选的分数差等于成对选择概率的 log odds | [bradley-terry](../alignment/bradley-terry/) |
| comparison graph | comparison graph / 比较图 | 译并保留常用英文;候选作为节点、成对比较作为边的图,用于检查覆盖和分数可识别性 | [bradley-terry](../alignment/bradley-terry/) |
| score identifiability | score identifiability / 分数可识别性 | 译并保留常用英文;检查平移基准、比较图组件和标签方向是否足以确定相对分数 | [bradley-terry](../alignment/bradley-terry/) |
| Davidson model / tie propensity | Davidson model / tie propensity / Davidson 模型 / 平局倾向 | 保留模型名并译;在 Bradley–Terry 分母加入平局项、用参数表示两个候选无法区分的概率 | [bradley-terry](../alignment/bradley-terry/) |
| Laplacian Hessian | Laplacian Hessian / Laplacian Hessian / 图 Laplacian Hessian | 译并保留常用英文;成对比较外积按边累加形成的二阶导矩阵,权重由重复次数和 p(1-p) 决定 | [bradley-terry](../alignment/bradley-terry/) |
| reward head | reward head / 奖励头 | 保留常用英文并译;从基础模型 hidden、EOS 或池化表示产生标量 reward 的可训练输出模块 | [reward-model](../alignment/reward-model/) |
| reward scale / score normalization | reward scale / score normalization / 奖励尺度 / 分数归一化 | 译并保留常用英文;定义 reward 平移、缩放、参考集均值方差和后续策略更新幅度的数值合同 | [reward-model](../alignment/reward-model/) |
| pairwise accuracy | pairwise accuracy / 成对准确率 | 译并保留常用英文;测试 reward model 是否把 chosen 的分数排在 rejected 之前的排序指标 | [reward-model](../alignment/reward-model/) |
| reward calibration | reward calibration / 奖励校准 | 译并保留常用英文;检查 reward 分差转换出的偏好概率与独立标签频率是否一致的过程 | [reward-model](../alignment/reward-model/) |
| candidate coverage | candidate coverage / 候选覆盖 | 译并保留常用英文;反馈数据对任务、难度、长度、语言、安全状态和 checkpoint 候选分布的覆盖程度 | [reward-model](../alignment/reward-model/) |
| length bias | length bias / 长度偏差 | 译并保留常用英文;reward 把输出长度、格式完整或冗余程度当作质量捷径的偏差 | [reward-model](../alignment/reward-model/) |
| label disagreement | label disagreement / 标签分歧 | 译并保留常用英文;标注者对同一候选比较给出不同选择、平局或不确定判断的情况 | [reward-model](../alignment/reward-model/) |
| reward overoptimization | reward overoptimization / 奖励过优化 | 译并保留常用英文;策略继续提高 reward model 分数、但独立质量、约束或真实结果开始下降的状态 | [reward-model](../alignment/reward-model/) |
| RLHF / reinforcement learning from human feedback | RLHF / 基于人类反馈的强化学习 | 保留缩写并译;用偏好标签训练 reward model、再用 rollout 和策略优化改变语言模型分布的训练流程 | [rlhf-dpo](../alignment/rlhf-dpo/) |
| reference policy | reference policy / 参考策略 | 译并保留常用英文;冻结的策略分布,用于 RLHF 的 KL 惩罚或 DPO 的 log probability 基准 | [rlhf-dpo](../alignment/rlhf-dpo/) |
| KL penalty / KL coefficient | KL penalty / KL coefficient / KL 惩罚 / KL 系数 | 译并保留常用英文;按策略与参考策略的 log-ratio 惩罚分布漂移、需要同时记录系数和 reduction | [rlhf-dpo](../alignment/rlhf-dpo/) |
| rollout policy / old policy | rollout policy / old policy / 采样策略 / 旧策略 | 译并保留常用英文;生成 RLHF 轨迹或保存 PPO 概率比分母的策略版本 | [rlhf-dpo](../alignment/rlhf-dpo/) |
| PPO / proximal policy optimization | PPO / proximal policy optimization / 近端策略优化 | 保留缩写并译;在旧策略 rollout 上用概率比、advantage 和 clipped surrogate 更新当前策略的策略优化算法 | [ppo](../alignment/ppo/) |
| PPO clipped surrogate | PPO clipped surrogate / PPO 裁剪代理目标 | 译并保留常用英文;用旧新策略概率比和 advantage 的裁剪最小值限制单批策略更新 | [ppo](../alignment/ppo/) |
| PPO ratio / importance ratio | PPO ratio / importance ratio / PPO 概率比 / 重要性比率 | 译并保留常用英文;当前策略与旧策略在同一状态动作上的概率比、用于复用 rollout 数据 | [ppo](../alignment/ppo/) |
| PPO clip fraction | PPO 裁剪比例 | 译;概率比落在 PPO 裁剪区间外的样本比例,需要和 KL 与 ratio 分位数联合解释 | [ppo](../alignment/ppo/) |
| value clipping | value clipping / value 裁剪 | 译并保留常用英文;限制 value head 相对旧预测的单批移动、通常取 raw 与 clipped value loss 的较大值 | [ppo](../alignment/ppo/) |
| terminal state / truncated state | terminal state / truncated state / 终止状态 / 截断状态 | 译并保留常用英文;区分环境真正结束与时间或预算耗尽、决定 TD error 是否 bootstrap | [ppo](../alignment/ppo/) |
| sequence policy ratio | sequence policy ratio / 序列策略概率比 | 译并保留常用英文;response token log probability 差求和后指数化的整段新旧策略概率比 | [ppo](../alignment/ppo/) |
| entropy coefficient | entropy coefficient / 熵系数 | 译并保留常用英文;PPO 总目标中控制动作分布熵项权重的配置、不能替代 reference KL | [ppo](../alignment/ppo/) |
| DPO / direct preference optimization | DPO / direct preference optimization / 直接偏好优化 | 保留缩写并译;从 KL 正则最优策略反解 reward 差、直接用当前与参考策略的 log-ratio 训练偏好对的方法 | [rlhf-dpo](../alignment/rlhf-dpo/) |
| log-ratio margin | log-ratio margin / log-ratio 分差 | 译并保留常用英文;chosen 与 rejected 相对参考策略的序列 log probability 差,构成 DPO 的偏好 logit | [rlhf-dpo](../alignment/rlhf-dpo/) |
| response-only loss | response-only loss / 仅回答损失 | 译并保留常用英文;只在回答 token 上计算 log probability 或策略损失、屏蔽 prompt 和 padding 位置 | [rlhf-dpo](../alignment/rlhf-dpo/) |
| DPO label smoothing | DPO label smoothing / DPO 标签平滑 | 译并保留常用英文;把 chosen 标签与 rejected 标签按 epsilon 混合、降低偏好目标过度确信的损失和梯度 | [dpo](../alignment/dpo/) |
| response mask | response mask / 回答掩码 | 译并保留常用英文;标记只参与回答 log probability 聚合的 token、排除 prompt 和 padding | [dpo](../alignment/dpo/) |
| sum/mean sequence reduction | sum/mean sequence reduction / 序列求和/求均值归约 | 译并保留常用英文;决定多 token 回答的 log probability 按 token 求和还是取均值、影响长度偏差 | [dpo](../alignment/dpo/) |
| reference-free preference optimization | reference-free preference optimization / 无参考策略偏好优化 | 译并保留常用英文;移除显式 reference log probability 的偏好目标变体、需要固定初始策略或其他基线 | [dpo](../alignment/dpo/) |
| implicit reward | implicit reward / 隐式奖励 | 译并保留常用英文;由策略相对 reference 的 log-ratio 代表的 reward 差、无需单独训练 reward model | [dpo](../alignment/dpo/) |
| preference pair swap test | preference pair swap test / 偏好对交换测试 | 译并保留常用英文;交换 chosen/rejected 后检查 margin、概率和 loss 是否按标签方向反转的数据管线测试 | [dpo](../alignment/dpo/) |
| Goodhart's law | Goodhart's law / 古德哈特定律 | 保留人名并译;当代理指标成为优化目标后、原先由该指标代表的真实关系可能失效的经验规律 | [reward-hacking](../alignment/reward-hacking/) |
| proxy gap | proxy gap / 代理缺口 | 译并保留常用英文;代理 reward 与独立真实效用在优化后产生的排序、均值或尾部差异 | [reward-hacking](../alignment/reward-hacking/) |
| shortcut feature | shortcut feature / 捷径特征 | 译并保留常用英文;长度、格式、关键词或语气等可提高代理分数、但不承载任务内容的表面变量 | [reward-hacking](../alignment/reward-hacking/) |
| extremal selection | extremal selection / 极值选择 | 译并保留常用英文;从大量候选中按带误差的代理分数选最高值、使极端误差更容易被选中的过程 | [reward-hacking](../alignment/reward-hacking/) |
| reward channel integrity | reward channel integrity / 奖励通道完整性 | 译并保留常用英文;计分所需状态来自受保护外部来源、策略不能改写评分输入的接口属性 | [reward-hacking](../alignment/reward-hacking/) |
| counterfactual intervention | counterfactual intervention / 反事实干预 | 译并保留常用英文;固定任务语义只改变长度、格式、语气或关键词、比较代理 reward 与独立结果变化的测试 | [reward-hacking](../alignment/reward-hacking/) |
| constraint violation rate | constraint violation rate / 约束违反率 | 译并保留常用英文;安全、权限、资源或格式检查失败事件在样本或轨迹中的比例 | [reward-hacking](../alignment/reward-hacking/) |
| reward-utility covariance | reward-utility covariance / 奖励效用协方差 | 译并保留常用英文;softmax 优化压力下真实效用期望对 beta 的导数、由代理 reward 与真实效用的协方差决定 | [reward-hacking](../alignment/reward-hacking/) |
| Constitutional AI | Constitutional AI / 宪法式对齐 | 保留方法名并译;用明确原则驱动批评、修订和偏好生成、再把反馈用于策略训练的流程 | [constitutional-and-rlaif](../alignment/constitutional-and-rlaif/) |
| constitution / principle set | constitution / principle set / 宪法 / 原则集合 | 保留常用英文并译;规定评价范围、优先级、例外和失败动作的版本化规则集合 | [constitutional-and-rlaif](../alignment/constitutional-and-rlaif/) |
| RLAIF / reinforcement learning from AI feedback | RLAIF / reinforcement learning from AI feedback / 基于 AI 反馈的强化学习 | 保留缩写并译;用 AI 评价器生成偏好标签、再训练 reward model 或直接优化策略的反馈流程 | [constitutional-and-rlaif](../alignment/constitutional-and-rlaif/) |
| self-critique / revision | self-critique / revision / 自我批评 / 修订 | 保留常用英文并译;模型按原则指出回答问题并生成改写候选的两步过程、需要和外部事实核验分开 | [constitutional-and-rlaif](../alignment/constitutional-and-rlaif/) |
| AI feedback / AI judge | AI feedback / AI judge / AI 反馈 / AI 评价器 | 译并保留常用英文;由模型对候选进行比较、打分或给出 rationale 以生成训练标签的反馈来源 | [constitutional-and-rlaif](../alignment/constitutional-and-rlaif/) |
| principle aggregation / priority | principle aggregation / priority / 原则聚合 / 原则优先级 | 译并保留常用英文;用加权和、词典式比较或硬约束处理多条原则冲突的规则 | [constitutional-and-rlaif](../alignment/constitutional-and-rlaif/) |
| evaluator calibration | evaluator calibration / 评价器校准 | 译并保留常用英文;用人工或外部结果检查 AI 评价器的标签方向、分组一致率、概率和尾部错误 | [constitutional-and-rlaif](../alignment/constitutional-and-rlaif/) |
| pairwise agreement | pairwise agreement / 成对一致率 | 译并保留常用英文;AI 评价器与独立标签在同一候选对上的偏好方向相同的比例 | [constitutional-and-rlaif](../alignment/constitutional-and-rlaif/) |
| best-of-N / sample-and-rank | best-of-N / sample-and-rank / N 选优 / 采样排序 | 保留常用英文并译;同一 prompt 采样 N 个候选、按代理 reward 选择最高或前 k 个回答的推理时筛选流程 | [rejection-sampling](../alignment/rejection-sampling/) |
| threshold filtering | threshold filtering / 阈值筛选 | 译并保留常用英文;只保留代理 reward 达到阈值的候选、需要报告总体和分组接受率 | [rejection-sampling](../alignment/rejection-sampling/) |
| rejection fine-tuning | rejection fine-tuning / 拒绝采样微调 | 译并保留常用英文;把筛选后 accepted 回答作为监督数据更新策略 checkpoint 的训练流程 | [rejection-sampling](../alignment/rejection-sampling/) |
| accepted distribution | accepted distribution / 接受分布 | 译并保留常用英文;proposal policy 在通过阈值或筛选事件条件化后的输出分布 | [rejection-sampling](../alignment/rejection-sampling/) |
| selection bias | selection bias / 选择偏差 | 译并保留常用英文;按带误差的代理分数选择极值候选后、被选样本与原始策略分布之间的系统差异 | [rejection-sampling](../alignment/rejection-sampling/) |
| candidate diversity / effective candidate count | candidate diversity / effective candidate count / 候选多样性 / 有效候选数 | 译并保留常用英文;按语义、错误类型或任务路径估计 N 个候选包含多少独立模式的指标 | [rejection-sampling](../alignment/rejection-sampling/) |
| expected draw count | expected draw count / 期望采样次数 | 译并保留常用英文;不断采样直到通过阈值时的期望尝试次数、在接受率 alpha 下为 1/alpha | [rejection-sampling](../alignment/rejection-sampling/) |
| process reward / PRM | process reward / process reward model / PRM / 过程奖励 / 过程奖励模型 | 保留缩写并译;在轨迹中间步骤或前缀上评分、用于信用分配、搜索剪枝或策略训练的奖励模型 | [process-reward](../alignment/process-reward/) |
| outcome reward / ORM | outcome reward / outcome reward model / ORM / 结果奖励 / 结果奖励模型 | 保留缩写并译;在最终答案、终止状态或外部任务结果上评分的奖励信号 | [process-reward](../alignment/process-reward/) |
| prefix value | prefix value / 前缀价值 | 译并保留常用英文;从当前前缀继续采样时获得最终成功或目标结果的条件期望 | [process-reward](../alignment/process-reward/) |
| step boundary | step boundary / 步骤边界 | 译并保留常用英文;把推理、代码或工具轨迹切分为可单独评分步骤的 token span 和协议 | [process-reward](../alignment/process-reward/) |
| process reward aggregation | process reward aggregation / 过程奖励聚合 | 译并保留常用英文;用 sum、mean、min、product 或 discount 把步骤分数变成轨迹分数的规则 | [process-reward](../alignment/process-reward/) |
| prefix recall | prefix recall / 前缀召回率 | 译并保留常用英文;最终成功轨迹中的前缀被过程评分器保留、没有被搜索剪枝删除的比例 | [process-reward](../alignment/process-reward/) |
| credit assignment | credit assignment / 信用分配 | 译并保留常用英文;把终止结果或过程反馈分配到导致结果的动作、token 或步骤的过程 | [process-reward](../alignment/process-reward/) |
