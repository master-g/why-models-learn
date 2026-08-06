import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateSvgTheme } from '../../scripts/lib/svg-theme-contract.mjs';

const baseLayout = readFileSync('src/layouts/BaseLayout.astro', 'utf8');
const homePage = readFileSync('src/pages/index.astro', 'utf8');
const articlePage = readFileSync('src/pages/[slug].astro', 'utf8');
const siteCss = readFileSync('src/styles/site.css', 'utf8');
const playgroundMarkdown = readFileSync('playground/rendering.md', 'utf8');
const playgroundSvg = readFileSync('public/assets/playground/svg/test-1.svg', 'utf8');

describe('independent visual contracts', () => {
  it('loads the repository-owned stylesheet without copied theme paths', () => {
    assert.match(baseLayout, /import '..\/styles\/site\.css'/);
    assert.doesNotMatch(baseLayout, /\/theme\//);
    assert.doesNotMatch(baseLayout, /zh-overrides\.css/);
  });

  it('uses semantic site navigation and a text brand', () => {
    assert.match(baseLayout, /class="site-header"/);
    assert.match(baseLayout, /class="site-brand__name">模型为什么会学习/);
    assert.match(baseLayout, /class="site-menu"/);
    assert.doesNotMatch(baseLayout, /icon-algebrica/);
  });

  it('renders the home page as an independent book index', () => {
    assert.match(homePage, /class="home-page"/);
    assert.match(homePage, /class="home-hero"/);
    assert.match(homePage, /class="section-index__group"/);
  });

  it('centers standalone illustrations and display formulas', () => {
    assert.match(
      siteCss,
      /\.article-section p > img:only-child\s*\{[^}]*display:\s*block;[^}]*margin-inline:\s*auto;/s,
    );
    assert.match(
      siteCss,
      /\.article-section p\.standalone-math > mjx-container\s*\{[^}]*display:\s*block;[^}]*text-align:\s*center;/s,
    );
  });

  it('contains formulas and wide tables on narrow screens', () => {
    assert.match(
      siteCss,
      /\.article-section mjx-container\[display="true"\][\s\S]*overflow-x:\s*auto;/,
    );
    assert.match(
      siteCss,
      /\.article-section table\s*\{[^}]*display:\s*block;[^}]*max-width:\s*100%;[^}]*overflow-x:\s*auto;/s,
    );
    assert.match(
      siteCss,
      /\.article-section li > mjx-container:not\(\[display="true"\]\)\s*\{[^}]*max-width:\s*100%;[^}]*overflow-x:\s*auto;/s,
    );
  });

  it('keeps notes visible in flow on narrow screens and in the 240px desktop margin', () => {
    assert.match(
      siteCss,
      /\.article-section \.sidenote\s*\{[^}]*display:\s*block;[^}]*max-width:\s*100%;[^}]*background:\s*var\(--note\);/s,
    );
    assert.match(
      siteCss,
      /@media \(min-width:\s*1040px\)[\s\S]*\.article-section \.sidenote\s*\{[^}]*float:\s*right;[^}]*clear:\s*right;[^}]*width:\s*240px;[^}]*background:\s*transparent;/s,
    );
    assert.match(
      siteCss,
      /@media \(min-width:\s*1040px\)[\s\S]*\.article-section::after\s*\{[^}]*clear:\s*both;/s,
    );
    assert.doesNotMatch(siteCss, /sidenote[^{}]*checkbox|checkbox[^{}]*sidenote/i);
  });

  it('styles note labels, anchors and return links with visible focus', () => {
    assert.match(siteCss, /\.sidenote__label[\s\S]*font-family:\s*var\(--font-mono\)/);
    assert.match(siteCss, /\.sidenote-ref:focus-visible/);
    assert.match(siteCss, /\.sidenote__backref:focus-visible/);
    assert.match(siteCss, /\.sidenote:target/);
  });

  it('keeps a complete sidenote fixture in the rendering playground', () => {
    assert.match(playgroundMarkdown, /\[\^sidenote-first\]/);
    assert.match(playgroundMarkdown, /\[\^sidenote-second\]/);
    assert.match(playgroundMarkdown, /> \[!marginnote\] 符号提醒/);
    assert.match(playgroundMarkdown, /\[\^sidenote-math\]:[^\n]*\$[^$]+\$/);
    assert.match(playgroundMarkdown, /\[\^sidenote-link\]:[^\n]*\[[^\]]+\]\([^)]+\)/);
  });

  it('renders a scoped content-license notice on every article', () => {
    assert.match(articlePage, /<footer class="article-attribution">/);
    assert.match(articlePage, /正文与原创插图/);
    assert.match(articlePage, /软件代码与第三方材料不在此许可范围内/);
  });

  it('keeps the repository-owned illustration on the adaptive SVG contract', () => {
    const result = validateSvgTheme(playgroundSvg, { asset: 'playground/svg/test-1.svg' });
    assert.deepEqual(result.errors, []);
    assert.match(playgroundSvg, /prefers-color-scheme:\s*dark/);
    assert.doesNotMatch(playgroundSvg, /\b(?:fill|stroke)="#[0-9a-f]{3,8}"/i);
  });
});
