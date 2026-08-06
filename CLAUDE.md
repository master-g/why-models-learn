# CLAUDE.md — why-models-learn

「模型为什么会学习」:写给未来的自己的结构化学习库。**词条是学习成果的正式沉淀:一个概念写成词条才算学完。**

## 架构:vault 是事实源,repo 是渲染器

- **词条唯一事实源**:Obsidian vault 飞地 `~/Documents/ObsidianVaults/Main/03 - AREAS/learning/why-models-learn/<slug>.md`(feynman 卡片格式,约定见该目录 `_README.md`)。
- `content-zh/` 是 `npm run sync` 的**纯产物,永不手改**。同步只拷 `status: complete | reference`;wikilink 改写为站内链接;普通 callout 标记剥掉,受控 `marginfigure`/`fullwidth`/`epigraph` 标记经源校验后保留供渲染管线转换;frontmatter 只透传 `title`/`tags`。
- 毕业信号:词条写完 → vault 里 `status` 翻为 `complete` → repo `sections.yaml` 的 `known_absent` 移除该 slug。指向未毕业词条的链接自动渲染为纯文本。

## 工作流

```bash
npm run sync    # vault → content-zh 单向同步
npm run dev     # 预览
npm test        # 单元测试 + 视觉契约
npm run build   # 构建 dist/
```

## 已决策事项(2026-07-27 grilling,勿重开)

- **大纲**(2026-07-28 重设计,书目驱动):**6 部分 25 章 308 词条**见 `sections.yaml`,结构为 parts > sections > entries 三级。数学地基(Part 0,94 词条)先行——用户编程资深、数学偏弱,**词条写法向数学原理与推导侧重**。GQA 独立为 `gqa-and-mqa`;softmax 独立;评估拆为总览 `overfitting-and-regularization` + 细节篇;进阶理论(Part 6)为参考轨,长期 TODO 合法。
- **词条格式**:三条硬规则——①开头定义+定位段(无 TLDR callout)②`##` 分节 ③结尾 `## 相关词条`;实战类加 `## 运行方法`。**体裁是 algebrica 长文**(概念揉碎展开、推导不跳步、数字实例、失效模式一节),不是费曼卡片;参照 `mnist-mlp-training-loop`。
- **代码**:ML/实战类词条内嵌,写作时本地真跑,输出贴进词条当证据;**Part 0 数学词条不引代码**(2026-07-28 用户反馈,vectors 重写时定)。大代码(LLaMA2)在 happy-llm。
- **插图**(2026-07-28 vectors 补图时定):手绘 SVG 存 vault `svg/<slug>.<n>.svg`(点号分隔,防 slug 前缀歧义),正文 `![中文 alt](svg/x.svg)`;sync 校验归属/存在性(缺失=硬错误)→ 拷 `public/assets/<section>/svg/` 并改写绝对路径,与词条同生命周期。SVG 用系统字体栈(img 内用不了 webfont),配色从纸墨主题。禁 `![[embed]]`。
- **草稿**:vault 内 `status: active`,不出 vault;演算纸笔记是消耗品。
- **支线**:CNN/RNN/LSTM 零代码,概念+数学推导。
- **部署**:GitHub Pages 公开地址为 <https://master-g.github.io/why-models-learn/>。发布工作流先执行 `check:public-history` 与 `check:public-release`;历史清理完成前禁止发布。站内绝对路径必须经 `withBase()` 或 markdown 管线的 rehype-prefix-base,唯一事实源 `src/lib/base.mjs`。

## 骨架

数学渲染、sectionize、搜索与中文写作 lint 源自 algebrica-zh 的 MIT 许可代码。公开界面使用本项目的独立样式 `src/styles/site.css`,不包含 Algebrica 主题包、品牌图标与主题字体。原始上下文见 `docs/handoff.md`。

**渲染试验田**:repo 根 `playground/rendering.md`(手维护 fixture,独立 collection)→ `/playground/` 页,与词条同管线同版式;改管线/样式后肉眼回归用。不进 vault/大纲/首页/搜索。

## 技术栈

- Astro ~7.1.0(静态站点,"type": "module",纯 ESM .mjs;无框架组件,仅 .astro)
- 数学管线:remark-math → rehype-mathjax/svg → rehype-sanitize(自定义 MathJax SVG 白名单)
- 内容:Astro content collection(glob loader 扫 `content-zh/*/*.md`),frontmatter 只需 `title`(+ 可选 `tags`)
- 测试:node:test + assert/strict(`node --test`),无测试框架

## 命令

- `make`(无参数)列出全部命令——npm 脚本的薄封装:`make sync / dev / build / test / preview / check-search / install`
- `make sync` 写词条后必跑;`make test`(单元测试 + 视觉契约)须全绿

## 代码风格

- 管线文件注释用中文;内部插件文件名沿用 algebrica 命名(历史遗留,勿改)
- 词条正文:`##`(h2)分节(sectionize 插件依赖);交叉链接写相对路径 `../slug/` 或 `../section/slug/`
- vault 侧写作用 Obsidian 习惯(`[[wikilink]]`、callout),同步脚本负责适配,站点侧不手写 wikilink

## 禁止文件

- `content-zh/` 下所有文件——同步产物,手改会被下次 sync 覆盖/清除
- `dist/`、`node_modules/`、`.astro/`、`data/`(gitignore)
- `public/theme/`、`public/styles/zh-overrides.css`——旧 Algebrica 主题资产,不得进入当前树或可达 Git 历史

## 审查规则

- 词条验收:概念讲透(费曼)+ 结尾 `## 相关词条` 交叉链接 + 实战类代码内嵌可跑且贴运行输出
- 自动闸(移植自 algebrica 验收体系,2026-07-28 升级):`npm run sync` 对每个同步词条跑 copywriting-lint——LINT 是警告(盘古空格/半角标点/直引号/全角数字/重复标点,回 vault 修),LINT-ERROR 是硬错误(数学区残留标点等,sync exit 1 不写产物);`npm run build` 的 postbuild 检查 `mjx-error`、插图 404 和全站内部引用
- 公开发布必须先通过 `npm run check:public-history` 和 `npm run check:public-release`;前者失败时按 `docs/runbooks/public-repository-cutover.md` 清理历史
- `npm test` 必须全绿;视觉契约(visual-contracts)约束 BaseLayout/index/[slug] 的类名结构,改页面时同步改测试
- sections.yaml 的 slug 与 vault 文件名一一对应;新词条先进 sections.yaml + known_absent

## 项目记忆 (回写约定)

跨会话的持久信息记录在 [PROJECT_MEMORY.md](./PROJECT_MEMORY.md)。
**完成每个重要任务后务必回写**: 把确认的决策写入「已验证的事实」、踩的坑写入「失败尝试」、用进展更新「上次会话」、把计划写入「下次运行」。
保持 PROJECT_MEMORY.md 在 300~400 行,超长时用 `scripts/memory.py compact` 压缩(保留事实与计划,淘汰最旧日志)。
