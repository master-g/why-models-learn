# 模型为什么会学习

写给未来的自己的结构化学习库：从向量与数学地基开始，到手写完整
Transformer，再到理解预训练、微调、推理与对齐。

**词条是学习成果的正式沉淀：一个概念写成词条才算学完。**
`sections.yaml` 先定义全部章节与词条；未完成词条保留为 TODO，随后按学习依赖
逐篇编写、核验并发布。

## 公开发布状态

公开站点使用独立界面与中文排版，不发布 Algebrica 的主题包、品牌图标或主题
字体文件。GitHub Pages 工作流同时检查当前文件树、构建产物和 Git 可达历史；
任何旧主题对象仍在可达历史中时，工作流拒绝部署。

公开历史清理必须在隔离镜像中执行，并保留可恢复 bundle。执行步骤见
[`docs/runbooks/public-repository-cutover.md`](docs/runbooks/public-repository-cutover.md)。

## 架构：vault 是事实源，repo 是渲染器

词条唯一事实源位于 Obsidian vault 的
`03 - AREAS/learning/why-models-learn/`。`content-zh/` 是 `npm run sync` 的
纯产物，不手工修改。

写作完成后，把 vault 词条的 `status` 改为 `complete`，从 `sections.yaml` 的
`known_absent` 移除对应 slug，再运行同步、测试、构建与视觉验收。

新增或迁移插图时，遵循 [SVG 主题作者规范](docs/authoring/svg-theme.md)。

## 常用命令

```bash
npm run sync                  # vault → content-zh 单向同步
npm test                      # 单元测试与视觉契约
npm run build                 # 构建 dist/ 并检查公式、插图和静态链接
npm run check:licenses        # 检查依赖许可元数据与第三方声明
npm run check:public-release  # 检查当前文件树和构建产物
npm run check:public-history  # 检查全部可达 Git 历史
```

## 结构

- `sections.yaml`：章节骨架与 `known_absent` TODO 列表。
- `content-zh/<section>/<slug>.md`：同步产物。
- `public/assets/<section>/svg/`：从 vault 同步的原创插图。
- `scripts/sync-from-vault.mjs`：同步管道。
- `scripts/migrate-svg-theme.mjs`：迁移与检查插图的明暗主题角色。
- `playground/rendering.md`：渲染管线回归夹具。

## 许可

- 中文词条与原创插图按
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  提供。
- 原创软件代码采用 MIT License。
- 第三方材料继续遵循各自许可证。

完整边界见 [LICENSE.md](LICENSE.md)，第三方清单见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
