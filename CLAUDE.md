# CLAUDE.md — nn-to-llm

「从神经网络到大语言模型」:写给未来的自己的结构化学习库。**词条是学习成果的正式沉淀:一个概念写成词条才算学完。**

## 架构:vault 是事实源,repo 是渲染器

- **词条唯一事实源**:Obsidian vault 飞地 `~/Documents/ObsidianVaults/Main/03 - AREAS/learning/nn-to-llm/<slug>.md`(feynman 卡片格式,约定见该目录 `_README.md`)。
- `content-zh/` 是 `npm run sync` 的**纯产物,永不手改**。同步只拷 `status: complete | reference`;wikilink 改写为站内链接;callout 标记剥掉;frontmatter 只透传 `title`/`tags`。
- 毕业信号:词条写完 → vault 里 `status` 翻为 `complete` → repo `sections.yaml` 的 `known_absent` 移除该 slug。指向未毕业词条的链接自动渲染为纯文本。

## 工作流

```bash
npm run sync    # vault → content-zh 单向同步
npm run dev     # 预览
npm test        # 单元测试 + 视觉契约
npm run build   # 构建 dist/
```

## 已决策事项(2026-07-27 grilling,勿重开)

- **大纲**:4 章 27 词条见 `sections.yaml`。评估并入 `overfitting-and-regularization`;GQA 并入 `multi-head-attention`;`modern-components` 已删;softmax 讲在 `loss-functions` 里。
- **词条格式**:三条硬规则——①开头一句话定义 ②`##` 分节 ③结尾 `## 相关词条`;实战类加 `## 运行方法`。
- **代码**:全部内嵌词条,写作时本地真跑,输出贴进词条当证据。大代码(LLaMA2)在 happy-llm。
- **草稿**:vault 内 `status: active`,不出 vault;演算纸笔记是消耗品。
- **支线**:CNN/RNN/LSTM 零代码,概念+数学推导。
- **部署**:攒够 10 个词条再上 GitHub Pages。

## 骨架

移植自 algebrica-zh(MathJax SVG 渲染、sectionize 主题、搜索、中文写作 lint)。原始上下文见 `docs/handoff.md`。

## 技术栈
- Astro ~7.1.0(静态站点,"type": "module",纯 ESM .mjs;无框架组件,仅 .astro)
- 数学管线:remark-math → rehype-mathjax/svg → rehype-sanitize(自定义 MathJax SVG 白名单)
- 内容:Astro content collection(glob loader 扫 `content-zh/*/*.md`),frontmatter 只需 `title`(+ 可选 `tags`)
- 测试:node:test + assert/strict(`node --test`),无测试框架

## 命令
- `npm run sync` — vault → content-zh 单向同步(写词条后必跑)
- `npm run dev` / `npm run build` / `npm test`(45 个测试,含视觉契约)

## 代码风格
- 管线文件注释用中文;内部插件文件名沿用 algebrica 命名(历史遗留,勿改)
- 词条正文:`##`(h2)分节(sectionize 插件依赖);交叉链接写相对路径 `../slug/` 或 `../section/slug/`
- vault 侧写作用 Obsidian 习惯([[wikilink]]、callout),同步脚本负责适配,站点侧不手写 wikilink

## 禁止文件
- `content-zh/` 下所有文件——同步产物,手改会被下次 sync 覆盖/清除
- `dist/`、`node_modules/`、`.astro/`、`data/`(gitignore)
- `public/theme/` 视觉资产(algebrica 移植,个人非商业用途,勿替换后外传)

## 审查规则
- 词条验收:概念讲透(费曼)+ 结尾 `## 相关词条` 交叉链接 + 实战类代码内嵌可跑且贴运行输出
- `npm test` 必须全绿;视觉契约(visual-contracts)约束 BaseLayout/index/[slug] 的类名结构,改页面时同步改测试
- sections.yaml 的 slug 与 vault 文件名一一对应;新词条先进 sections.yaml + known_absent

## 项目记忆 (回写约定)
跨会话的持久信息记录在 [PROJECT_MEMORY.md](./PROJECT_MEMORY.md)。
**完成每个重要任务后务必回写**: 把确认的决策写入「已验证的事实」、踩的坑写入「失败尝试」、用进展更新「上次会话」、把计划写入「下次运行」。
保持 PROJECT_MEMORY.md 在 300~400 行,超长时用 `scripts/memory.py compact` 压缩(保留事实与计划,淘汰最旧日志)。
