# SVG 明暗主题验收记录

日期：2026-08-06

范围：`why-models-learn` 全部原创插图的暗色适配，以及 SVG 主题契约、迁移工具和构建闸。

## 结果摘要

验收结论：通过。

- vault 源插图：278 张。
- 发布目录插图：279 张，其中包含 `playground` 回归夹具 1 张。
- 标准迁移组：10 张。
- 专用迁移组：268 张。
- `public/assets/` 主题契约检查：279/279 通过。
- `npm test`：97/97 通过。
- `npm run sync`：严格同步成功。
- `npm run build`：构建与 postbuild 闸通过。

## 实现边界

标准迁移路径只接受已登记的纸墨颜色。它把直接颜色属性改为 `svg-*` 语义类，并保持几何结构投影不变。

专用迁移路径按元素、绘制属性和上下文分配角色。它覆盖以下情况：

- 样式块中的颜色声明。
- 线性渐变和径向渐变的 `stop-color`。
- 同一个源色在文字、线条和填充中的不同语义。
- RGBA 的透明度。
- 大面积 `rect` 背景。
- 通过 `url(#...)` 使用的内部定义。

大面积浅色填充会生成 `svg-special-background-*` 角色。组元素上继承的填充会补充文字角色，以避免暗色主题下标签与填充混用同一低对比度颜色。旧的专用类保留在结构中，新角色按 CSS 后出现顺序覆盖其颜色，避免修改几何结构和元素顺序。

## 静态检查

逐图检查覆盖以下项目：

1. 根元素声明 `data-svg-theme="paper-ink-v1"`。
2. 每个使用的角色都有浅色定义。
3. 每个使用的角色都有暗色定义。
4. 暗色定义位于 `prefers-color-scheme: dark` 媒体规则中。
5. 标准角色使用固定纸墨颜色。
6. 文字、图形和普通填充达到各自的对比度阈值。
7. 颜色字面量不绕过语义角色类。
8. 内部 `id` 唯一，`url(#...)` 和 `href="#..."` 引用存在。
9. 迁移前后结构投影一致。

附加审计对 279 张发布 SVG 的专用文字与专用背景暗色组合逐一计算对比度。结果为 0 个低于 4.5:1 的组合。

代表性文件抽样覆盖：

| 类型 | 文件 |
| --- | --- |
| 标准纸墨图 | `public/assets/linear-algebra/svg/lengths-and-distances.1.svg` |
| 样式块与渐变 | `public/assets/evaluation-and-generalization/svg/data-augmentation.1.svg` |
| 继承文字与背景 | `public/assets/attention/svg/attention-as-retrieval.1.svg` |
| 微积分线图 | `public/assets/calculus/svg/gradient.1.svg` |
| 卷积结构图 | `public/assets/cnn/svg/convolution-2d.1.svg` |
| 概率分布图 | `public/assets/probability/svg/gaussian-distribution.1.svg` |
| Transformer 组件图 | `public/assets/transformer-components/svg/rope.1.svg` |
| 训练流程图 | `public/assets/training-nn/svg/adam.1.svg` |
| 序列模型图 | `public/assets/rnn-lstm/svg/sequence-modeling.1.svg` |
| 对齐流程图 | `public/assets/alignment/svg/reward-model.1.svg` |
| 文本表示图 | `public/assets/text-representation/svg/embeddings.1.svg` |
| 渲染试验田 | `public/assets/playground/svg/test-1.svg` |

## 浏览器抽样

在本地 Astro 开发站点打开 `/why-models-learn/lengths-and-distances/`，先以浅色主题加载，再点击主题按钮切换到黑色主题。页面无需刷新即可完成切换。

浏览器观测结果：

- 页面主题状态为 `black`。
- 文档根节点的 `color-scheme` 计算值为 `dark`。
- 文章插图 `lengths-and-distances.1.svg` 加载完成。
- 直接打开同一 SVG 时，浏览器的 `prefers-color-scheme: dark` 媒体规则匹配。
- 页面文字、数学公式和插图容器在暗色主题下保持可见。

浏览器自动化无法对内嵌 SVG 建立独立的裁剪上下文，因此本次不把单张截图作为逐图证明。逐图证明由发布目录静态契约检查提供，浏览器检查用于确认主题状态传播和代表性页面加载。

## 复现命令

```bash
npm run sync
npm run check:svg-theme
npm test
npm run build
git diff --check
```

以上命令在本次验收中均返回成功。`dist/` 是构建产物，不纳入提交。
