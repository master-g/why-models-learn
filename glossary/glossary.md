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
| span | 张成(集) | 译,正式处理见「线性组合与张成」 | [subspaces](../linear-algebra/subspaces/) |
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

## 机器学习

| 英文 | 中文 | 约定 | 首现 |
| --- | --- | --- | --- |
| one-hot | one-hot(独热) | 首次出现括注「独热」,后文用 one-hot | [vectors](../linear-algebra/vectors/) |
| hidden state | 隐藏状态 | 译,首次出现括注英文 | [vectors](../linear-algebra/vectors/) |
| representation | 表示 | 译 | [vectors](../linear-algebra/vectors/) |
| loss | 损失 | 首次出现对照「损失(loss)」,后文用「损失」 | [vectors](../linear-algebra/vectors/) |
| embedding | 嵌入 | 译(嵌入层、词嵌入);词条 slug 保留英文 | [vectors](../linear-algebra/vectors/) |
| attention mask | 注意力掩码 | 译 | [vectors](../linear-algebra/vectors/) |
| gating mechanism | 门控机制 | 译 | [vectors](../linear-algebra/vectors/) |

## 保留不译

| 英文 | 原因 | 首现 |
| --- | --- | --- |
| MNIST | 数据集名 | [vectors](../linear-algebra/vectors/) |
| Transformer | 架构名 | [vectors](../linear-algebra/vectors/) |
| LoRA | 微调方法名(Low-Rank Adaptation) | [rank](../linear-algebra/rank/) |
