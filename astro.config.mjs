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
      section: ['dataFootnotes', ['className', 'footnotes', 'post-section']],
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
