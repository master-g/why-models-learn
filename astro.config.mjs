import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax/svg';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import rehypeMarkStandaloneMath from './src/plugins/rehype-mark-standalone-math.mjs';
import rehypeRewriteAlgebrica from './src/plugins/rehype-rewrite-algebrica.mjs';
import rehypeSectionizeAlgebrica from './src/plugins/rehype-sectionize-algebrica.mjs';
import danglingJson from './src/lib/dangling-links.json' with { type: 'json' };
import { buildSlugMap } from './src/lib/slug-map.mjs';
import { getKnownAbsent, getSections } from './src/lib/sections.mjs';
import minLightTheme from '@shikijs/themes/min-light';

// min-light 默认是 Material 冷调色(饱和红/蓝/紫/橙),浮在暖纸上不协调;
// 按 Claude 设计 token(claude.design.md)整体重映射为其代码块指引的
// 「muted blues / oranges / grays」三族,紫色(设计禁止的第四色)并入 teal。
// theme 传对象而非名字:shiki 的 createHighlighter 接受 ThemeRegistration,Astro 原样透传。
const SHIKI_COLOR_MAP = {
  '#24292e': '#141413', // 基础前景 → ink
  '#212121': '#141413', // 标点/editor 前景 → ink
  '#d32f2f': '#cc785c', // 关键字:Material 红 → coral primary(哑橙)
  '#1976d2': '#2b5581', // 数字/常量 → 哑蓝(并入字符串一族)
  '#6f42c1': '#a9583e', // 函数/类型:紫(设计禁色)→ primary-active(teal #5db8a6 对比仅 ~2.5:1,实测太淡)
  '#ff9800': '#e8a55a', // 函数参数 → accent-amber
  '#22863a': '#2b5581', // 字符串/标签/正则/模板串 → 哑蓝(并入 string 族;绿会成三族外的第四色)
  '#c2c3c5': '#6c6a64', // 注释:太淡的冷灰 → muted 暖灰
  '#316bcd': '#2b5581', // diff info → 哑蓝
  '#cd9731': '#d4a017', // diff warn → warning
  '#cd3131': '#c64545', // diff error → error
  '#800080': '#6c6a64', // diff debug → muted
};
const shikiTheme = {
  ...minLightTheme,
  colors: { ...minLightTheme.colors, 'editor.foreground': '#141413' },
  tokenColors: minLightTheme.tokenColors.map((rule) => {
    const fg = rule.settings?.foreground;
    if (!fg) return rule;
    // min-light 部分色值带 alpha 后缀(#24292eff),先归一化再查表。
    const key = fg.slice(0, 7).toLowerCase();
    const mapped = SHIKI_COLOR_MAP[key];
    return mapped ? { ...rule, settings: { ...rule.settings, foreground: mapped } } : rule;
  }),
};

// 空词条合法(known_absent):slugMap 只含已写词条,strictEmpty 关闭。
const slugMap = buildSlugMap({ source: 'fs', strictCollisions: true, strictEmpty: false, silent: true });
// 指向未写词条的链接渲染为纯文本;写完从 known_absent 移除后自动变链接。
const dangling = { ...danglingJson, text: [...(danglingJson.text || []), ...getKnownAbsent()] };
console.log(`[astro-config] built slug map: ${slugMap.size} articles`);

/**
 * Extend the default rehype-sanitize schema to allow MathJax SVG output.
 * Keeps script stripping and javascript: URL blocking intact.
 */
function makeMathSchema(base) {
  return {
    ...base,
    // 关闭 id/name 的 user-content- 前缀改写:MathJax 的 <path id> 与 <use xlink:href>
    // 必须逐字对应,前缀化会让全部字形引用悬空(公式空白)。元素均为管线产出,无注入面。
    clobber: [],
    tagNames: [
      ...(base.tagNames || []),
      // MathJax SVG 的伴随 <style> 块必须保留为元素——标签被剥掉会把整段 CSS 以文本泄漏进页面。
      // remark 默认不允许原始 HTML,LLM 内容无法注入元素,放行 <style> 无注入面。
      'style',
      'mjx-container',
      'mjx-assistive-mml',
      'mjx-math',
      'mjx-mrow',
      'mjx-mi',
      'mjx-mo',
      'mjx-mn',
      'mjx-mtext',
      'mjx-mspace',
      'mjx-msub',
      'mjx-msup',
      'mjx-msubsup',
      'mjx-mfrac',
      'mjx-msqrt',
      'mjx-mroot',
      'mjx-munder',
      'mjx-mover',
      'mjx-munderover',
      'mjx-mtable',
      'mjx-mtr',
      'mjx-mtd',
      'mjx-semantics',
      'mjx-annotation',
      'svg',
      'g',
      'path',
      'defs',
      'use',
      'line',
      'rect',
      'circle',
      'ellipse',
      'polygon',
      'polyline',
      'text',
      'tspan',
      'clipPath',
      'linearGradient',
      'radialGradient',
      'stop',
      'symbol',
    ],
    attributes: {
      ...(base.attributes || {}),
      'mjx-container': ['class', 'jax', 'display', 'justify', 'width', 'role', 'style', 'tabIndex'],
      'mjx-assistive-mml': ['role'],
      'mjx-math': ['xmlns', 'display', 'alttext'],
      svg: ['xmlns', 'width', 'height', 'role', 'focusable', 'viewBox', 'xmlnsXLink', 'style', 'preserveAspectRatio'],
      g: ['stroke', 'fill', 'strokeWidth', 'transform', 'dataMmlNode', 'style'],
      path: ['id', 'd'],
      use: ['dataC', 'xLinkHref', 'href', 'transform'],
      line: ['x1', 'y1', 'x2', 'y2', 'stroke', 'strokeWidth'],
      rect: ['x', 'y', 'width', 'height', 'rx', 'ry', 'stroke', 'strokeWidth', 'fill'],
      circle: ['cx', 'cy', 'r'],
      ellipse: ['cx', 'cy', 'rx', 'ry'],
      polygon: ['points'],
      polyline: ['points'],
      text: ['x', 'y', 'dx', 'dy', 'fontSize', 'fontFamily', 'textAnchor'],
      tspan: ['x', 'y', 'dx', 'dy'],
      clipPath: ['id'],
      linearGradient: ['id', 'x1', 'y1', 'x2', 'y2'],
      radialGradient: ['id', 'cx', 'cy', 'r'],
      stop: ['offset', 'stopColor'],
      symbol: ['id'],
      a: ['href', 'title', 'target', 'rel', 'class'],
      // 词条插图(SVG):defaultSchema 只放行 src/aria,alt 会被剥掉。
      img: [...(base.attributes?.img || []), 'alt', 'title', 'width', 'height'],
      section: ['dataFootnotes', ['className', 'footnotes', 'post-section']],
      // shiki 的着色全靠 span/pre 的 inline style;remark 不开 raw HTML,
      // 元素均为管线产物(shiki/MathJax),放行 style 无注入面。
      pre: ['class', 'style', 'tabIndex'],
      code: ['class', 'style'],
      span: ['class', 'style'],
      '*': [...(base.attributes?.['*'] || []), 'className', 'class'],
    },
  };
}

export default defineConfig({
  compressHTML: true,
  trailingSlash: 'always',
  build: {
    format: 'directory',
  },
  markdown: {
    shikiConfig: { theme: shikiTheme },
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeMathjax,
        rehypeMarkStandaloneMath,
        [rehypeRewriteAlgebrica, { slugMap, dangling, warn: console.warn, sectionDirs: getSections().map((s) => s.dir) }],
        rehypeSectionizeAlgebrica,
        [rehypeSanitize, makeMathSchema(defaultSchema)],
      ],
    }),
  },
});
