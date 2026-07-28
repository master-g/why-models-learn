# 模型为什么会学习

写给未来的自己的结构化学习库:从向量与数学地基开始,到手写完整 Transformer,再到复现 LLaMA2。

**词条是学习成果的正式沉淀:一个概念写成词条才算学完。** 骨架先行——`sections.yaml` 定义全部章节与词条,空词条以 TODO 状态光明正大地空着(known_absent 模式),随学习进度逐个填充。

## 架构:vault 是事实源,本 repo 是渲染器

词条在 Obsidian vault 飞地 `Documents/ObsidianVaults/Main/03 - AREAS/learning/why-models-learn/` 里以 feynman 卡片书写(约定见该目录 `_README.md`);`content-zh/` 是 `npm run sync` 的纯产物,不手改。

## 写作工作流

1. 在 vault 飞地新建 `<slug>.md`(slug 须在 `sections.yaml` 中),frontmatter 带 `title` 与 `status: active`。
2. 写完 → `status` 翻为 `complete` → `npm run sync` → 从 `sections.yaml` 的 `known_absent` 移除该 slug。站内指向它的链接随即自动从纯文本变成真链接。
3. `npm run dev` 预览;`npm test` 跑渲染与样式契约检查。

词条验收标准:概念讲透(写给自己的费曼检验)+ 交叉链接到相关词条 + 实战类词条代码自包含可运行(输出贴进词条当证据)。

## 常用命令

`make`(无参数)列出全部命令,均为 npm 脚本的薄封装:

```bash
make install       # 安装依赖
make sync          # vault → content-zh 单向同步
make dev           # 本地预览
make build         # 构建到 dist/
make test          # 单元测试 + 视觉契约
make check-search  # 搜索冒烟(需先 build;词条未写齐时 FAIL 属预期)
```

## 结构

- `sections.yaml` — 章节骨架与 known_absent TODO 列表
- `content-zh/<section>/<slug>.md` — 同步产物(勿手改)
- `scripts/sync-from-vault.mjs` — 同步管道(wikilink 改写/callout 剥离/status 闸门)
- `docs/handoff.md` — 项目启动上下文(定位、已决策事项、路线图)

## 许可与致谢

内容以 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可发布。

站点骨架与视觉主题移植自 [Algebrica](https://algebrica.org) 的中文镜像项目 algebrica-zh(个人非商业用途);数学公式由 MathJax 渲染。
