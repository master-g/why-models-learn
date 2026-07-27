import yaml from 'js-yaml';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax/svg';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import rehypeMarkStandaloneMath from '../plugins/rehype-mark-standalone-math.mjs';
import rehypeRewriteAlgebrica from '../plugins/rehype-rewrite-algebrica.mjs';
import rehypeSectionizeAlgebrica from '../plugins/rehype-sectionize-algebrica.mjs';
import danglingJson from '../lib/dangling-links.json' with { type: 'json' };
import { buildSlugMap } from '../lib/slug-map.mjs';
import { getKnownAbsent, getSections } from '../lib/sections.mjs';

// 与 astro.config.mjs 同一套规则:slugMap 只含已写词条,known_absent 链接降级为纯文本。
const slugMap = buildSlugMap({ source: 'fs', strictCollisions: true, strictEmpty: false, silent: true });
const dangling = { ...danglingJson, text: [...(danglingJson.text || []), ...getKnownAbsent()] };

function makeMathSchema(base) {
  return {
    ...base,
    // 关闭 id/name 的 user-content- 前缀改写(同 astro.config.mjs:防止字形引用悬空)。
    clobber: [],
    tagNames: [
      ...(base.tagNames || []),
      // MathJax SVG 的伴随 <style> 块必须保留为元素(同 astro.config.mjs)。
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

const processors = new Map();

function getProcessor(currentSection = null) {
  if (!processors.has(currentSection)) {
    processors.set(
      currentSection,
      createMarkdownProcessor({
        remarkPlugins: [remarkMath],
        rehypePlugins: [
          rehypeMathjax,
          rehypeMarkStandaloneMath,
          [rehypeRewriteAlgebrica, { slugMap, dangling, currentSection, sectionDirs: getSections().map((s) => s.dir) }],
          rehypeSectionizeAlgebrica,
          [rehypeSanitize, makeMathSchema(defaultSchema)],
        ],
      }),
    );
  }
  return processors.get(currentSection);
}

function parseFrontmatter(raw) {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (!match) return { data: {}, body: raw };
  return { data: yaml.load(match[1]) || {}, body: match[2] };
}

export async function renderPageMarkdown(raw, { currentSection } = {}) {
  const processor = await getProcessor(currentSection);
  const result = await processor.render(raw);
  return result.code;
}

export async function renderPageMarkdownWithFrontmatter(raw, { currentSection } = {}) {
  const { data, body } = parseFrontmatter(raw);
  const html = await renderPageMarkdown(body, { currentSection });
  return { data, html };
}
