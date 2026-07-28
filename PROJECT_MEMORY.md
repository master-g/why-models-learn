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
- [2026-07-28] 链接方案经首个真实词条验证:路由平铺 `/<slug>/`;sync 把 wikilink 改写为 `../<section>/<slug>/`,构建期 rehype-rewrite-algebrica 再规范化为 `/<slug>/`(已写)或降级纯文本(known_absent)——两层配合,作者只写 wikilink。
- [2026-07-28] **体裁定案(用户反馈修订 07-27 决策)**:词条要 algebrica 长文,不是费曼卡片——概念按内在结构揉碎展开、推导不跳步、抽象必配具体数字、关键论断贴真实运行输出、结尾前有失效模式/反例一节;开头是定义+定位段,不要 TLDR callout。模板 = `mnist-mlp-training-loop`。约定已同步进 vault `_README.md` 与 CLAUDE.md。
- [2026-07-28] **用户画像(影响一切写法)**:十多年编程经验,数学偏弱 → 词条向数学原理/推导侧重,代码细节可提速;数学预备单独立词条,后续词条直接引用不重复讲。
- [2026-07-28] **大纲 v2(书目驱动重设计)**:5 组代理并行抽取 12 本书目录(MML/Parr&Howard/Chaudhury/Fleuret/Petersen&Zech/Ananthaswamy/Smets/Nelson/Amidi/Xiao&Zhu/Prince/Goodfellow),按数学依赖链重排为 **6 部分 25 章 301 词条**(Part 0 数学地基 87 词先行;Part 6 进阶理论参考轨长期 TODO 合法)。schema 升为三级:sections.yaml 加 `parts` 列表 + 每章 `part` 字段,首页按部分分组渲染。原有 27 slug 全保留(多数降为章总览篇);`derivatives-and-chain-rule`/`probability-and-information` 溶解为原子词条。素材原件:书库 `/Users/mg/Documents/wps/Documents/books/AI-ML/`。注意 `Mathematical Theory of Deep Learning (Fang et al, 2024).pdf` 实为 Petersen & Zech 所著,文件名标错。
- [2026-07-28] **linear-algebra 章拆细 30→37(用户批准,参考 vault 内 Bright Side of Mathematics 65 集结构)**:`basis-and-dimension` 拆为 `basis`+`dimension`;新增 `subspaces`/`coordinates`/`kernel-and-image`/`rank-nullity`/`matrix-inverse`/`characteristic-polynomial`;`rank` 移到 linear-maps 之后(BoM 秩零定理阶梯)。总词条 301→308。不照搬 65 集:定义/例子分集太细,叉积/Sarrus/Cramer 等 ML 用不上的不收;SVD/低秩/伪逆他没有的保留。BoM 笔记路径:vault `03 - AREAS/learning/bright-side-math/Linear_Algebra/`。
- [2026-07-28] **自动闸两道(移植 algebrica 验收体系,用户批准)**:①`npm run sync` 对每个同步词条跑 copywriting-lint,reports/errors 以 `[sync] slug:line LINT[-ERROR]` 警告输出(只警告不改写,vault 是事实源);②`postbuild` 挂 `scripts/check-mjx-errors.mjs`,扫 dist 全部 html 的 `mjx-error`,有即构建失败。不移植项:torture 文件制度、validate-translation 四件套、双闸流程(均翻译项目专属)。
- [2026-07-28] **Part 0 数学词条不引代码(用户反馈,vectors 重写时定)**:数学词条纯推导;代码只属于 ML/实战类词条;「NN 语境」节保留但只概念提及。篇幅向 algebrica 单概念长文收敛(vectors 从 230 行重写为 126 行)。约定已同步 vault `_README.md` 与 CLAUDE.md。
- [2026-07-28] **渲染试验田**:repo 根 `playground/rendering.md` 独立 collection(content.config.ts 加 `playground`)→ `/playground/` 页(src/pages/playground.astro,复用词条版式类);不进 vault/大纲/首页/搜索。已覆盖 case:行内/展示数学、pmatrix/bmatrix/cases/align、长公式溢出、Python/shell/无语言围栏代码块、长行滚动、站内链接与 known_absent 降级、blockquote、列表表格。
- [2026-07-28] **代码块渲染修复链**:①sanitize 白名单剥掉 shiki span/pre 的 inline style(代码无色的根因)→ astro.config.mjs 对 pre/code/span 放行 `style`(元素均管线产物,无注入面);②`markdown.shikiConfig.theme` 对自定义 processor 同样生效,主题 github-dark → min-light;③zh-overrides.css 新增代码块层(暖白底 #fdfcfa !important 压 inline style、rgba(49,47,47,0.14) 边框、6px 圆角、Geist Mono 栈)与行内代码样式。主题色:纸 #f4efee、墨 #312f2f。

## 失败尝试
<!-- 踩过的坑、走不通的路径、被否决的方案及原因。超长时最旧条目优先淘汰。 -->
- [2026-07-27] 软连接 vault → content-zh 方案被否:git 只存断链指针,GitHub Pages 构建时 Actions 拿不到 vault 内容,部署必断——故用单向拷贝同步。
- [2026-07-27] 链接改写插件的 `sectionDirs` 最初从 slugMap(只含已写词条)推导,空章节缺席导致跨章节 known_absent 链接降级失效(报 new dangling 而非纯文本)——改为调用方从 sections.yaml 传入全量章节。
- [2026-07-28] copywriting-lint 从 algebrica 移植后暴露两个语料差异(algebrica 无代码块、数学单行):①围栏代码块未遮蔽,代码里的英文直引号被误报「建议改用「」」;②遮蔽用 `' '.repeat` 把换行也抹掉,多行代码块/多行 `$$` 后行号漂移。修法:代码围栏最先遮蔽,且遮蔽一律 `replace(/[^\r\n]/g, ' ')` 保留换行。已加单测(代码块跳过 + 行号保持)。
- [2026-07-28] 词条代码注释里写 numpy 输出 `[[14.]]` 被 sync 的 wikilink 改写当成 `[[slug]]` 报警——代码/输出块内的双方括号与 wikilink 语法冲突,写作时避开或改写表述。

## 上次会话
<!-- 上次运行做了什么、停在何处。滚动记录,超长时最旧条目优先淘汰。 -->
- [2026-07-27] 骨架落地:拷 algebrica-zh、拆翻译管线、改单 collection、45/45 测试绿、构建 6 页、探针词条全链路验证(sync/渲染/链接降级/自动清理)后清理。加 LICENSE(CC BY-NC-SA 4.0)、README、docs/handoff.md 归档。
- [2026-07-27] grilling 定案大纲/格式/存放后全部落地:sections.yaml 修订、sync-from-vault.mjs、vault 飞地 + _README.md、CLAUDE.md、bootstrap 上下文文件。
- [2026-07-27] 停在:站点 0 词条、27 个 TODO 待写,尚未 git init。
- [2026-07-27] git init 完成并推送 github.com/master-g/nn-to-llm(root-commit b961a90,57 文件);README/BaseLayout 里的仓库地址假设已证实正确。
- [2026-07-28] 首个词条 `mnist-mlp-training-loop` 全链路走通:vault 写入 → status complete → sync → 移出 known_absent → 45/45 测试 → 构建 7 页。代码用 `uv run --with torch --with torchvision` 真跑,5 epoch 输出 test_acc=0.9774 已贴进词条。check-search FAIL 仍为预期 TODO 信号。流程无框架性问题,未提交 git。
- [2026-07-28] 用户反馈首版是费曼卡片非所求,按 algebrica 长文体重写:补经验风险形式化、交叉熵手算例、参数量/batch 数量账、评估循环语义;新增「失效模式」节并真跑两个变体——去 ReLU(线性坍缩 ~90%,且后段 train/test 裂口=过拟合征兆)、lr=10(首 epoch 损失 27 万,坍缩到 ≈ln10 均匀分布锁死 10%)。教训:解读段落必须按真实数字写,先跑后写,别先拟稿套数字。
- [2026-07-28] 写 `vectors` 前先做验收闸分析(algebrica 移植评估)并落地两道自动闸:sync 挂 copywriting-lint 警告(用临时 vectors.md 注入半角标点验证 LINT-ERROR 触发)、postbuild 挂 check-mjx-errors.mjs(失败/通过路径均验证)。坑:块注释里写 `dist/**/*.html`,其中的 `*/` 提前闭合注释导致 SyntaxError。45/45 测试绿,构建+postbuild 通过。改动未提交:sync-from-vault.mjs(lint 接线)、check-mjx-errors.mjs(新)、package.json(postbuild)、CLAUDE.md、PROJECT_MEMORY.md。
- [2026-07-28] **`vectors` 毕业(1/301)**:Part 0 首篇,algebrica 长文体(三视角/运算定义+几何/四条性质/Hadamard/NN 三角色/失效模式四条),全部数字来自 `uv run --with numpy` 实跑(加法数乘/784 维拉平/广播陷阱/行列外积)。lint 闸立功:抓出代码块误报→给 lint 加代码围栏遮蔽+保换行(46 测试绿);sync 抓出 `[[14.]]` wikilink 冲突→改写注释。渲染验证:/vectors/ 36 个 mjx-container 零错误,未毕业交叉链接全纯文本,首页与 vault _index.md 均已更新。未提交:vault 词条 + sections.yaml + copywriting-lint.mjs + 测试 + PROJECT_MEMORY。

## 下次运行
<!-- 计划要做的任务和优先级。长期保留,不随压缩淘汰。 -->
- [2026-07-28] 沿 linear-algebra 章顺序推进:`vectors` ✅ → 下一篇 `matrices`。穿插 Part 2 保持手感。跑代码一律 `uv run --with <pkg>` 临时环境;解读按真实输出写,先跑后写。
- [2026-07-28] MNIST 词条退役回炉(用户决定「不保留,按新架构重规划」):vault 翻回 active、slug 回 known_absent,sync 自动清产物(退役链路验证通过)。重写时机 = Part 2 收尾,届时改为纯实战篇、数学全引用;其失效模式真跑素材(去 ReLU ~90%+过拟合裂口、lr=10 锁死 ln10)分流到 `saturation-and-vanishing`/`gradient-descent` 等对应词条。站点回到 0 成品 301 TODO。
- [2026-07-27] 仓库 github.com/master-g/nn-to-llm 已建已推;攒够 10 个词条再部署 Pages。
