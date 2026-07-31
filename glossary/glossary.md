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
| group / Abelian group | 群 / Abel 群 | 译,Abel 音译不翻 | [vector-spaces](../linear-algebra/vector-spaces/) |
| closure | 封闭性 | 译 | [vector-spaces](../linear-algebra/vector-spaces/) |
| scalar multiplication | 数乘 | 译(与 vectors 篇一致) | [vector-spaces](../linear-algebra/vector-spaces/) |
| affine set | 仿射集 | 译,正式处理见「仿射空间与映射」 | [vector-spaces](../linear-algebra/vector-spaces/) |

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
