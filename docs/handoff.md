# Handoff:「从神经网络到大语言模型」知识库项目

> 本文档是项目启动的全部上下文。新建 repo 后,把本文件交给 Claude(或直接放入 repo 根目录)即可继续。
> 产生于 2026-07-27 happy-llm 工作区的一次 grilling 讨论,决策已定,勿重开。

## 定位

写给未来的自己的结构化学习库。**词条是学习成果的正式沉淀:一个概念写成词条才算学完。**
形式对齐 algebrica-zh:概念词条(一个概念一篇,互相链接)+ `sections.yaml` 章节骨架先行 + 空词条即 TODO(known_absent 模式,光明正大空着)。

背景:happy-llm 的笔记进度(Ch1-2 完成)超越真实水平,因此本项目从 NN 基础重新夯实。happy-llm 的 Ch3-Ch6 **暂停**,其 `reference/` 降级为素材参考书;写到第 4 章时回去做 Ch5/Ch6 作为实战。

## 大纲初稿(sections.yaml 按此建)

```
1. neural-network-basics 神经网络基础
   - what-is-a-neuron        神经元: 权重/偏置/加权和
   - activation-functions    激活函数: sigmoid/tanh/ReLU, 为什么需要非线性
   - loss-functions          损失函数: MSE/交叉熵
   - gradient-descent        梯度下降与学习率
   - backpropagation         反向传播: 链式法则
   - mnist-mlp-training-loop ★实战: MNIST + MLP 手写训练循环(第一个词条)

2. classic-architectures 经典架构(支线, 零代码, 概念词条+数学推导)
   - cnn                     CNN: 局部性/权值共享/池化
   - rnn                     RNN: 序列建模
   - lstm                    LSTM: 门控机制, 梯度消失
   - why-attention           从 LSTM 到注意力: 长程依赖/串行瓶颈/梯度消失

3. transformer
   - tokenization-bpe        分词: 手写 BPE (参考 Karpathy minbpe)
   - embeddings              词嵌入
   - self-attention          自注意力: Q/K/V, 缩放点积
   - multi-head-attention    多头注意力
   - positional-encoding     位置编码: 正弦 → RoPE
   - layernorm-residuals     归一化与残差连接
   - feedforward             FFN → SwiGLU
   - full-transformer        组装完整 Transformer

4. modern-llm GPT 与现代 LLM
   - decoder-only            Decoder-only 架构与因果掩码
   - modern-components       LLaMA 化组件: RMSNorm/RoPE/SwiGLU/GQA
   - pretraining             预训练: 数据/目标/超参
   - inference               推理: KV cache, 温度/top-p 采样
   - sft                     指令微调
   - rlhf-dpo                RLHF/DPO 概念
   - llama2-from-scratch     ★实战: 回 happy-llm 复现 Ch5(215M LLaMA2)
```

## 已定决策(勿重开)

| 决策 | 结论 |
|---|---|
| 写给谁 | 写给未来的自己;骨架先行,词条随学习写,空词条合法 |
| 内容模型 | 纯概念词条;原始笔记不进站,留在 happy-llm notebooks/ |
| 支线深度 | CNN/RNN/LSTM 只写概念词条+数学推导,零代码实战 |
| MNIST | 做,MLP 手写训练循环,半天;CNN 版留作可选升级 |
| 互动 | 静态优先(KaTeX+SVG);现成互动直接嵌入;自研 demo 只在写词条卡住时做,预算 ≤ 该词条写作时间。候选: softmax 温度滑块、注意力热力图、BPE 合并动画 |
| 写作顺序 | 按大纲顺序,从 NN 基础开始(因真实水平落后于 happy-llm 笔记进度) |
| License | CC BY-NC-SA 4.0(与 happy-llm 一致;全文在 happy-llm/LICENSE) |

## 技术落地步骤

1. 用户新建 GitHub repo(名字用户定,暂称 `nn-to-llm`),clone 到 `~/Documents/workspace/personal/`
2. 拷贝 algebrica-zh 骨架(`~/Documents/workspace/personal/algebrica-zh`):
   - 保留:`astro.config.mjs`、`package.json`、`src/`、`public/`、`tests/`、`scripts/`、`.gitignore`
   - 清空:`content-zh/` 下的词条内容(保留目录机制)、`docs/`
   - 重写:`sections.yaml`(按上面大纲)、站点名称/标题
   - 删除或简化:翻译追踪机制(`translation-failures.json`、frontmatter 里的 translation 字段、glossary 如用不上)
   - 以 `npm install && npm run dev` 跑通为准;`npm test` 有渲染检查测试
3. 加 `LICENSE`(CC BY-NC-SA 4.0 全文,happy-llm/LICENSE 可直接拷)、`README.md`
4. 本文件归档为 repo 的 `HANDOFF.md` 或 `docs/handoff.md`
5. 部署先不做;攒够 10 个词条再上 GitHub Pages

## 第一个里程碑

1. 骨架跑通,sections.yaml 大纲入库(空词条合法)
2. 词条 `mnist-mlp-training-loop`:PyTorch 手写 MLP 训练 MNIST,含完整训练循环(数据→前向→loss→反向→step),代码自包含可跑
3. 四件套词条:neuron / activation / loss / gradient-descent / backpropagation

每个词条验收标准:概念讲透(写给自己的费曼检验)+ 交叉链接到相关词条 + 实战类词条代码可运行。

## 与 happy-llm 的关系

- happy-llm repo: `~/Documents/workspace/personal/happy-llm`(github.com/master-g/happy-llm)
- Ch3-Ch6 暂停;`reference/`(上游 datawhalechina/happy-llm 克隆)作为素材库
- 写到第 4 章 `llama2-from-scratch` 词条时,回 happy-llm 做 Ch5/Ch6 实战
- happy-llm 的 OVERVIEW.md 路线图里的硬件建议(M5 MPS / RTX 3080Ti 分工)依然有效
