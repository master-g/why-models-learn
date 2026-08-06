# 学习路径与 Tufte 教学布局验收记录

日期：2026-08-06
分支：`codex/feat-learning-paths-tufte`
范围：学习路径数据、主线导航、数学层级提示、Tufte 块级语法、同步校验、Markdown AST、响应式样式、打印规则和小规模内容试点。

## 结果摘要

实现范围已完成，发布结论待授权。当前已完成一篇数学词条的 `marginfigure` 试点：`lengths-and-distances`。`mnist-mlp-training-loop` 暂不加入 `fullwidth` 源标记，因为本轮浏览器测量没有证明其表格在 680 px 正文列中不可读。渲染试验田和学习路径页仍覆盖三种块级布局，以及学习路径页的 `epigraph` 和 `newthought`。

以下结果区分自动化证据、浏览器实测和仍未覆盖的媒体边界。没有把 CSS 契约测试当作浏览器实测结论。

## 自动化证据

- `learning-paths.yaml` 将 94 篇数学词条划分为 17 篇最小前置、56 篇按需回补和 21 篇参考轨，重复、遗漏、未知 slug 和未知章节会使单元测试失败。
- `/learn/` 显示五个主线阶段、三个入口动作、五组最小数学前置、按需回补、可选经典架构和参考轨。
- 主线词条使用服务端生成的前后链接；可选支线返回 Transformer 主线；Part 6 词条保持目录专用导航；回补词条显示适用阶段的稳定返回链接。
- `marginfigure`、`fullwidth` 和 `epigraph` 在同步层先校验，再由 `rehype-tufte-blocks` 转成 `figure`、`figcaption`、`blockquote` 和 `footer`。
- 布局校验覆盖缺图、空 alt、多图、图片与表格混合、表格列数不一致、缺少题记来源、嵌套 callout、代码、展示公式和原始 HTML。
- 布局试点测试确认同步产物只有一个 `marginfigure` 标记，并确认模型词条尚未采用 `fullwidth`。
- 普通旁注仍由原有管线处理，布局标记不会被普通 callout 清理逻辑误删。
- 本轮 `npm run sync` 同步 274 个词条和 278 张插图；`npm test` 为 149/149；`npm run build` 生成 304 个页面，`mjx-error=0`，279 个插图引用通过，静态站点闸解析 10440 个站内引用。

复现命令：

```bash
npm run sync
npm test
npm run build
git diff --check
```

## 响应式与打印合同

代码合同规定以下目标尺寸。集成浏览器已完成浅色桌面、浅色移动、深色桌面和打印媒体的实测：

| 环境 | 目标行为 | 代码合同 | 实测状态 |
| --- | --- | --- | --- |
| 桌面浅色（1440 × 1000） | 旁注图进入页边 | `marginfigure` 宽 240 px，正文列 680 px | 通过：`playground` 与 `lengths-and-distances` 均为 240 px，页面 `scrollWidth = clientWidth = 1440` |
| 桌面浅色（1440 × 1000） | 通栏使用正文与页边总宽 | `fullwidth` 最大宽 1000 px | 通过：`playground` 为 1000 px，表格容器为 1000 px |
| 桌面深色（1440 × 1000） | 主题切换不破坏布局 | 同一布局规则 | 通过：主题为 `black`，旁注 240 px，通栏 1000 px，页面无横向溢出 |
| 移动浅色（390 × 844） | 布局回到正文流 | `marginfigure`、`fullwidth` 宽度 100%，表格容器自身滚动 | 通过：正文列与块宽 350 px，表格内容宽 680 px 且容器滚动，页面 `scrollWidth = clientWidth = 390` |
| 打印媒体（1440 × 1000） | 取消浮动与页面级溢出 | 布局宽度回到纸张可用宽度，表格取消 680 px 最小宽度 | 通过：`matchMedia('print')` 为真，三种块均取消浮动，表格 `display: table`、`min-width: 0` |
| reduced-motion | 不增加动画状态 | 仅关闭滚动平滑 | 通过媒体查询：`matchMedia('(prefers-reduced-motion: reduce)')` 为真；无控制台错误 |

目标页面矩阵为 `/`、`/learn/`、`/playground/`、`/lengths-and-distances/`、`/mnist-mlp-training-loop/`、`/cnn/` 和 `/category/approximation-theory/`。浅色矩阵覆盖 1440 × 1000 与 390 × 844；`playground` 另覆盖桌面深色主题和打印媒体。所有页面导航后均无控制台错误。

## 浏览器边界

本轮未使用 Playwright 自带缓存浏览器；通过仓库规定的集成浏览器驱动连接本地预览，完成真实 DOM、视口、主题、控制台和打印媒体检查。浏览器检查覆盖布局合同，不等同于真实纸张打印机输出；打印结果来自浏览器 `print` 媒体模拟。

下一次浏览器验收应：

1. 在真实打印机或导出 PDF 环境复核分页和字体替换。
2. 只有在 `mnist-mlp-training-loop` 的表格超出 680 px 或出现可读性损失时，才把它作为第二篇 `fullwidth` vault 试点。

## 删除增强内容检查

删除 `lengths-and-distances` 的 `marginfigure` 后，距离球定义、三种范数计算、机器学习用途和正文插图仍在。删除 `/learn/` 的题记与开篇短语后，五阶段标题、词条链接和回补关系仍在。布局内容没有承担必要定义或推导。

## 未执行事项

- 未新增第二篇 vault 布局试点；`fullwidth` 试点等待真实 680 px 可读性证据。
- 未在真实打印机或 PDF 导出器中验证分页；本轮只完成浏览器打印媒体模拟。
- 未提交、推送或发布分支。
