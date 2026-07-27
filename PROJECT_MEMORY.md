# 项目记忆 — nn-to-llm

> 跨会话的持久信息。完成任务后回写,约定见 [CLAUDE.md](./CLAUDE.md)。
> 长度上限约 400 行。超长时:**保留**「已验证的事实」与「下次运行」,**淘汰**最旧的「失败尝试」「上次会话」条目。
> 条目用 `- [YYYY-MM-DD]` 前缀,新条目追加在各小节**末尾**,便于按时间淘汰。

## 已验证的事实
<!-- 确认过的技术决策和约束。长期保留,不随压缩淘汰。 -->
- [2026-07-27] 架构:vault 是词条唯一事实源(`~/Documents/ObsidianVaults/Main/03 - AREAS/learning/nn-to-llm/<slug>.md`,feynman 卡片格式),repo 是渲染器;`content-zh/` 是 sync 纯产物,永不手改。
- [2026-07-27] 发布闸门:vault frontmatter `status: complete | reference` 才被同步;`active` 草稿不出 vault。毕业 = status 翻转 + sections.yaml 的 known_absent 移除 slug。
- [2026-07-27] 大纲定案 4 章 27 词条:评估并入 `overfitting-and-regularization`;GQA 并入 `multi-head-attention`;`modern-components` 已删;softmax 讲在 `loss-functions`;BPE 保持第 3 章首位(数据流顺序)。
- [2026-07-27] 词条格式三条硬规则:①开头一句话定义 ②`##` 分节 ③结尾 `## 相关词条`;实战类加 `## 运行方法`,代码全内嵌且贴运行输出当证据。
- [2026-07-27] vault 飞地约定:文件名 = slug(kebab-case),`[[slug]]` wikilink 写交叉链接,sync 按 sections.yaml 映射改写;禁用 embeds 与 Obsidian 附件图片。约定全文在飞地 `_README.md`。
- [2026-07-27] 骨架移植自 algebrica-zh,翻译管线(translate/glossary/英文源 collection/上游文章图谱)已整体拆除;`scripts/check-search.mjs` 的断言指向未写词条,FAIL 是预期 TODO 信号。
- [2026-07-27] obsidian-vault skill 的 vault 路径已从 `/mnt/d/...` 修正为 `/Users/mg/Documents/ObsidianVaults/Main/`(Windows 时代遗留)。

## 失败尝试
<!-- 踩过的坑、走不通的路径、被否决的方案及原因。超长时最旧条目优先淘汰。 -->
- [2026-07-27] 软连接 vault → content-zh 方案被否:git 只存断链指针,GitHub Pages 构建时 Actions 拿不到 vault 内容,部署必断——故用单向拷贝同步。
- [2026-07-27] 链接改写插件的 `sectionDirs` 最初从 slugMap(只含已写词条)推导,空章节缺席导致跨章节 known_absent 链接降级失效(报 new dangling 而非纯文本)——改为调用方从 sections.yaml 传入全量章节。

## 上次会话
<!-- 上次运行做了什么、停在何处。滚动记录,超长时最旧条目优先淘汰。 -->
- [2026-07-27] 骨架落地:拷 algebrica-zh、拆翻译管线、改单 collection、45/45 测试绿、构建 6 页、探针词条全链路验证(sync/渲染/链接降级/自动清理)后清理。加 LICENSE(CC BY-NC-SA 4.0)、README、docs/handoff.md 归档。
- [2026-07-27] grilling 定案大纲/格式/存放后全部落地:sections.yaml 修订、sync-from-vault.mjs、vault 飞地 + _README.md、CLAUDE.md、bootstrap 上下文文件。
- [2026-07-27] 停在:站点 0 词条、27 个 TODO 待写,尚未 git init。
- [2026-07-27] git init 完成并推送 github.com/master-g/nn-to-llm(root-commit b961a90,57 文件);README/BaseLayout 里的仓库地址假设已证实正确。

## 下次运行
<!-- 计划要做的任务和优先级。长期保留,不随压缩淘汰。 -->
- [2026-07-27] 写第一个里程碑词条:`mnist-mlp-training-loop`(PyTorch 手写 MLP 训练 MNIST,完整训练循环,代码自包含可跑,贴运行输出)。在 vault 飞地写,status 先 active,完成翻 complete 后 sync + 移出 known_absent。
- [2026-07-27] 之后按大纲顺序写四件套:what-is-a-neuron / activation-functions / loss-functions / gradient-descent / backpropagation。
- [2026-07-27] 仓库 github.com/master-g/nn-to-llm 已建已推;攒够 10 个词条再部署 Pages。
