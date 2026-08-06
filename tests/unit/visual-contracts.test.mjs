import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const baseLayout = readFileSync('src/layouts/BaseLayout.astro', 'utf8');
const homePage = readFileSync('src/pages/index.astro', 'utf8');
const articlePage = readFileSync('src/pages/[slug].astro', 'utf8');
const siteCss = readFileSync('src/styles/site.css', 'utf8');

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

  it('renders a scoped content-license notice on every article', () => {
    assert.match(articlePage, /<footer class="article-attribution">/);
    assert.match(articlePage, /正文与原创插图/);
    assert.match(articlePage, /软件代码与第三方材料不在此许可范围内/);
  });
});
